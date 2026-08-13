#!/usr/bin/env python3
"""Refit H800 M1 while preserving the original exact historical holdout.

The 167 observations whose original partition role is ``holdout`` never enter
model selection, center fitting, residual calibration, expansion calibration or
OOM-guard fitting in this script.  They are scored once by models fitted on the
current observations, with or without the 158 historical calibration rows.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from common import read_json, sha256_json, write_json
from export_h800_observations import validate_canonical_observation, write_jsonl
from fit_h800_lora_source_disjoint_recalibration_v1 import (
    DEFAULT_COVERAGE,
    DEFAULT_FROZEN_BASELINE,
    DEFAULT_HARDWARE,
    DEFAULT_INVENTORY,
    VARIANT_FEATURES,
    VARIANT_M1,
    _collapse_records,
    _evaluate,
    _fit_bundle,
    _input_binding,
    _inventory_models,
    _is_critical_lora,
    _nested_loso,
    _outcome,
    _predict_stack,
    _prediction_detail,
    _read_jsonl,
    _source_id,
    _write_candidate,
)
from fit_h800_profile_aware_memory_challenger_v1 import _export_exact_jobs
from migrate_refit_h800_historical_memory_v1 import (
    DEFAULT_CANONICAL,
    DEFAULT_CURRENT_OLD,
    DEFAULT_DATASET_ANALYSIS,
    DEFAULT_NEW_QUEUE,
    DEFAULT_THEORY_BASIS,
    HISTORICAL_GROUP_PREFIX,
    _build_pairs,
    _connected_components,
    _content_hashes,
    _current_data_paths,
    _dataset_registry,
    _inject_historical_contract,
    _legacy_basis_ids,
    _metric_delta,
    _read_historical_exact,
)


SCHEMA = "sft_h800_fixed_historical_holdout_refit/v2"
IMPLEMENTATION_VERSION = (
    "sft_h800_fixed_historical_holdout_refit/"
    "2026-08-05.original-holdout-never-fit-v2"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "diagnostics"
    / "h800_fixed_historical_holdout_20260805"
)


def _original_role(observation: Mapping[str, Any]) -> str:
    partition = (observation.get("configuration") or {}).get(
        "calibration_partition"
    ) or {}
    return str(partition.get("role") or "unknown").lower()


def _add_outcome_to_cluster_id(records: Sequence[dict[str, Any]]) -> None:
    for record in records:
        record["cluster_id"] = f"{record['cluster_id']}-{_outcome(record)}"


def _score(
    records: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _prediction_detail(
            record,
            _predict_stack(record, bundle),
            variant=VARIANT_M1,
        )
        for record in records
    ]


def _critical(details: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in details if row.get("critical_lora") is True]


def _raw_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "outcomes": dict(Counter(_outcome(row) for row in records)),
        "independent_sources": len({_source_id(row) for row in records}),
        "critical_lora_rows": sum(_is_critical_lora(row) for row in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--theory-basis", type=Path, default=DEFAULT_THEORY_BASIS)
    parser.add_argument(
        "--dataset-analysis", type=Path, default=DEFAULT_DATASET_ANALYSIS
    )
    parser.add_argument("--new-queue", type=Path, default=DEFAULT_NEW_QUEUE)
    parser.add_argument("--current-old", type=Path, default=DEFAULT_CURRENT_OLD)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--hardware", type=Path, default=DEFAULT_HARDWARE)
    parser.add_argument(
        "--frozen-baseline", type=Path, default=DEFAULT_FROZEN_BASELINE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    args = parser.parse_args()
    if not 0.5 < args.coverage < 1.0:
        raise ValueError("coverage must be strictly between 0.5 and 1")

    registry, _analysis = _dataset_registry(args.dataset_analysis)
    components, historical_overlaps = _connected_components(registry)
    source_group_by_dataset: dict[str, str] = {}
    for index, component in enumerate(components, start=1):
        group = f"{HISTORICAL_GROUP_PREFIX}{index:02d}"
        for dataset_id in component:
            source_group_by_dataset[dataset_id] = group

    legacy_ids, legacy_basis_audit = _legacy_basis_ids(args.theory_basis)
    historical_exact, canonical_audit = _read_historical_exact(
        args.canonical, legacy_ids
    )
    historical_calibration_observations = [
        row for row in historical_exact if _original_role(row) == "calibration"
    ]
    historical_holdout_observations = [
        row for row in historical_exact if _original_role(row) == "holdout"
    ]
    if len(historical_calibration_observations) != 158:
        raise ValueError("expected 158 original historical calibration observations")
    if len(historical_holdout_observations) != 167:
        raise ValueError("expected 167 original historical holdout observations")
    training_observation_ids = {
        str(row["observation_id"]) for row in historical_calibration_observations
    }
    holdout_observation_ids = {
        str(row["observation_id"]) for row in historical_holdout_observations
    }
    if training_observation_ids & holdout_observation_ids:
        raise ValueError("historical calibration/holdout observation IDs overlap")

    migrated_historical_calibration = [
        _inject_historical_contract(
            row,
            registry=registry,
            source_group_by_dataset=source_group_by_dataset,
        )
        for row in historical_calibration_observations
    ]
    migrated_historical_holdout = [
        _inject_historical_contract(
            row,
            registry=registry,
            source_group_by_dataset=source_group_by_dataset,
        )
        for row in historical_holdout_observations
    ]

    queue = _read_jsonl(args.new_queue)
    if len(queue) != 77:
        raise ValueError("current new queue must contain 77 jobs")
    print("exporting current exact 77-job observations", flush=True)
    new_observations = _export_exact_jobs(queue)
    old_observations = _read_jsonl(args.current_old)
    if len(old_observations) != 28:
        raise ValueError("current old snapshot must contain 28 observations")
    current_observations = [*old_observations, *new_observations]
    for row in current_observations:
        errors = validate_canonical_observation(row)
        if errors:
            raise ValueError(f"current canonical validation failed: {errors}")

    current_paths = _current_data_paths(new_observations, old_observations)
    current_hashes: set[str] = set()
    for path in current_paths.values():
        current_hashes |= _content_hashes(path)
    historical_current_overlap = {
        dataset_id: len(set(binding["content_hashes"]) & current_hashes)
        for dataset_id, binding in sorted(registry.items())
    }
    if any(historical_current_overlap.values()):
        raise ValueError("historical and current data content overlap")

    inventory = read_json(args.inventory)
    model_by_id, fixed_lora = _inventory_models(inventory)
    hardware = read_json(args.hardware)
    padding_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    raw_max_cache: dict[str, int] = {}
    current_cutoff, current_effective = _build_pairs(
        current_observations,
        model_by_id=model_by_id,
        fixed_lora=fixed_lora,
        hardware=hardware,
        origin="current_19_sources",
        padding_cache=padding_cache,
        raw_max_cache=raw_max_cache,
    )
    historical_calibration_cutoff, historical_calibration_effective = _build_pairs(
        migrated_historical_calibration,
        model_by_id=model_by_id,
        fixed_lora=fixed_lora,
        hardware=hardware,
        origin="historical_original_calibration_158",
        padding_cache=padding_cache,
        raw_max_cache=raw_max_cache,
    )
    historical_holdout_cutoff, historical_holdout_effective = _build_pairs(
        migrated_historical_holdout,
        model_by_id=model_by_id,
        fixed_lora=fixed_lora,
        hardware=hardware,
        origin="historical_fixed_holdout_167",
        padding_cache=padding_cache,
        raw_max_cache=raw_max_cache,
    )
    _add_outcome_to_cluster_id(historical_calibration_cutoff)
    _add_outcome_to_cluster_id(historical_calibration_effective)
    _add_outcome_to_cluster_id(historical_holdout_cutoff)
    _add_outcome_to_cluster_id(historical_holdout_effective)

    historical_training_clusters = {
        str(row["cluster_id"]) for row in historical_calibration_effective
    }
    historical_holdout_clusters = {
        str(row["cluster_id"]) for row in historical_holdout_effective
    }
    overlapping_clusters = historical_training_clusters & historical_holdout_clusters
    if overlapping_clusters:
        raise ValueError("historical calibration/holdout configuration groups overlap")

    current_cutoff_collapsed, current_effective_collapsed, current_collapse = (
        _collapse_records(current_cutoff, current_effective)
    )
    training_cutoff, training_effective, training_collapse = _collapse_records(
        [*current_cutoff, *historical_calibration_cutoff],
        [*current_effective, *historical_calibration_effective],
    )
    holdout_cutoff, holdout_effective, holdout_collapse = _collapse_records(
        historical_holdout_cutoff,
        historical_holdout_effective,
    )
    del current_cutoff_collapsed, training_cutoff, holdout_cutoff
    print(
        f"training raw={len(current_effective) + len(historical_calibration_effective)} "
        f"unique={len(training_effective)} holdout raw={len(historical_holdout_effective)} "
        f"unique={len(holdout_effective)}",
        flush=True,
    )

    print("running current-only M1 nested leave-source-out", flush=True)
    baseline_nested = _nested_loso(
        current_effective_collapsed,
        variant=VARIANT_M1,
        coverage=args.coverage,
    )
    print("running augmented-training M1 nested leave-source-out", flush=True)
    augmented_nested = _nested_loso(
        training_effective,
        variant=VARIANT_M1,
        coverage=args.coverage,
    )

    baseline_bundle = _fit_bundle(
        current_effective_collapsed,
        names=VARIANT_FEATURES[VARIANT_M1],
        coverage=args.coverage,
        diagnostic_max_fallback=False,
    )
    augmented_bundle = _fit_bundle(
        training_effective,
        names=VARIANT_FEATURES[VARIANT_M1],
        coverage=args.coverage,
        diagnostic_max_fallback=False,
    )
    baseline_holdout_details = _score(holdout_effective, baseline_bundle)
    augmented_holdout_details = _score(holdout_effective, augmented_bundle)

    baseline_current_details = list(baseline_nested["details"])
    augmented_current_details = [
        row
        for row in augmented_nested["details"]
        if row.get("origin") == "current_19_sources"
    ]
    augmented_historical_training_fold_details = [
        row
        for row in augmented_nested["details"]
        if row.get("origin") == "historical_original_calibration_158"
    ]
    baseline_overall_details = [
        *baseline_current_details,
        *baseline_holdout_details,
    ]
    augmented_overall_details = [
        *augmented_current_details,
        *augmented_holdout_details,
    ]

    metrics = {
        "current_19_source_disjoint": {
            "current_only_training": _evaluate(baseline_current_details),
            "augmented_training": _evaluate(augmented_current_details),
            "comparison": _metric_delta(
                _evaluate(baseline_current_details),
                _evaluate(augmented_current_details),
            ),
        },
        "historical_fixed_holdout": {
            "current_only_training": _evaluate(baseline_holdout_details),
            "augmented_training": _evaluate(augmented_holdout_details),
            "comparison": _metric_delta(
                _evaluate(baseline_holdout_details),
                _evaluate(augmented_holdout_details),
            ),
        },
        "pooled_current_plus_historical_holdout": {
            "current_only_training": _evaluate(baseline_overall_details),
            "augmented_training": _evaluate(augmented_overall_details),
            "comparison": _metric_delta(
                _evaluate(baseline_overall_details),
                _evaluate(augmented_overall_details),
            ),
        },
        "critical_lora": {
            "current_19_source_disjoint": {
                "current_only_training": _evaluate(
                    _critical(baseline_current_details)
                ),
                "augmented_training": _evaluate(
                    _critical(augmented_current_details)
                ),
            },
            "historical_fixed_holdout": {
                "current_only_training": _evaluate(
                    _critical(baseline_holdout_details)
                ),
                "augmented_training": _evaluate(
                    _critical(augmented_holdout_details)
                ),
            },
            "pooled_current_plus_historical_holdout": {
                "current_only_training": _evaluate(
                    _critical(baseline_overall_details)
                ),
                "augmented_training": _evaluate(
                    _critical(augmented_overall_details)
                ),
            },
        },
        "historical_calibration_source_nested_fold": _evaluate(
            augmented_historical_training_fold_details
        ),
    }
    current_critical = metrics["critical_lora"]["current_19_source_disjoint"][
        "augmented_training"
    ]
    holdout_critical = metrics["critical_lora"]["historical_fixed_holdout"][
        "augmented_training"
    ]
    stage_two_triggers = {
        "current_critical_admission_recall_below_0p90": float(
            current_critical.get("admission_recall") or 0.0
        )
        < 0.90,
        "current_critical_reserved_mape_above_0p20": float(
            current_critical.get("reserved_center_source_equal_mape") or math.inf
        )
        > 0.20,
        "historical_holdout_false_safe_or_unsafe_admission": (
            int(holdout_critical.get("false_safe_oom") or 0) > 0
            or int(holdout_critical.get("unsafe_success_admitted") or 0) > 0
        ),
        "historical_holdout_upper_coverage_below_0p95": float(
            holdout_critical.get("reserved_upper_source_equal_coverage") or 0.0
        )
        < 0.95,
    }
    stage_two_required = any(stage_two_triggers.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    holdout_details_path = args.output_dir / "historical_fixed_holdout_predictions.jsonl"
    write_jsonl(holdout_details_path, augmented_holdout_details)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fixed_historical_holdout_evaluated",
        "publishable": False,
        "production_model_mutated": False,
        "model_variant_fixed_before_holdout_evaluation": VARIANT_M1,
        "inputs": {
            "canonical_observations": _input_binding(args.canonical),
            "theory_basis": _input_binding(args.theory_basis),
            "dataset_analysis": _input_binding(args.dataset_analysis),
            "new_queue": _input_binding(args.new_queue),
            "current_old_observations": _input_binding(args.current_old),
            "model_inventory": _input_binding(args.inventory),
            "hardware": _input_binding(args.hardware),
            "frozen_baseline": _input_binding(args.frozen_baseline),
        },
        "data_contract": {
            "current_observations": len(current_observations),
            "historical_original_calibration_observations": len(
                historical_calibration_observations
            ),
            "historical_fixed_holdout_observations": len(
                historical_holdout_observations
            ),
            "training_raw_observations": len(current_effective)
            + len(historical_calibration_effective),
            "training_unique_configuration_results": len(training_effective),
            "holdout_raw_observations": len(historical_holdout_effective),
            "holdout_unique_configuration_results": len(holdout_effective),
            "training_holdout_observation_id_overlap": 0,
            "historical_training_holdout_configuration_result_overlap": len(
                overlapping_clusters
            ),
            "holdout_used_for_variant_selection": False,
            "holdout_used_for_center_fit": False,
            "holdout_used_for_residual_or_expansion_calibration": False,
            "holdout_used_for_oom_guard_fit": False,
            "historical_and_current_sample_content_overlap": sum(
                historical_current_overlap.values()
            ),
            "historical_train_and_holdout_share_dataset_source_group": True,
            "interpretation": (
                "The fixed holdout is observation/configuration-disjoint and never fit, "
                "but it is not a new source-disjoint prospective dataset because its "
                "five historical datasets form the same connected content source group."
            ),
        },
        "raw_data_audit": {
            "current": _raw_audit(current_effective),
            "historical_calibration": _raw_audit(
                historical_calibration_effective
            ),
            "historical_holdout": _raw_audit(historical_holdout_effective),
            "current_collapse": current_collapse,
            "training_collapse": training_collapse,
            "holdout_collapse": holdout_collapse,
            "historical_dataset_content_overlaps": historical_overlaps,
            "historical_connected_components": components,
            "historical_current_content_overlap_rows": historical_current_overlap,
            "legacy_non_attempt_scoped_observations_excluded": legacy_basis_audit,
            "canonical_audit": canonical_audit,
        },
        "metrics": metrics,
        "release_gate": {
            "stage_two_calibration_required": stage_two_required,
            "stage_two_triggers": stage_two_triggers,
            "formal_new_source_prospective_acceptance_passed": False,
            "production_replacement_allowed": False,
        },
    }
    report["report_sha256"] = sha256_json(report)
    report_path = args.output_dir / "fixed_historical_holdout_report.json"
    write_json(report_path, report)

    candidate_path = args.output_dir / "candidate_model_strict_holdout.json"
    candidate = _write_candidate(
        candidate_path,
        selected_variant=VARIANT_M1,
        bundle=augmented_bundle,
        source_count=len({_source_id(row) for row in training_effective}),
        report_sha256=report["report_sha256"],
        stage_two_required=stage_two_required,
        stage_two_triggers=stage_two_triggers,
    )
    manifest = {
        "schema": "sft_h800_fixed_historical_holdout_manifest/v2",
        "generated_at_utc": report["generated_at_utc"],
        "files": {
            path.name: _input_binding(path)
            for path in (holdout_details_path, report_path, candidate_path)
        },
        "candidate_sha256": candidate["candidate_sha256"],
        "holdout_never_fit": True,
        "publishable": False,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(args.output_dir / "output_manifest.json", manifest)
    print(
        f"holdout={len(holdout_effective)} stage_two_required={stage_two_required} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
