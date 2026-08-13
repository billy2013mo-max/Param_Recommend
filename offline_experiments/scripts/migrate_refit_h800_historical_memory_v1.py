#!/usr/bin/env python3
"""Migrate exact historical H800 memory observations to the effective-sequence model.

The historical public-derived datasets share many samples, so connected datasets
are kept as one independent source group for weighting, cross-validation and tail
calibration.  Former historical holdout rows may be reclassified as consumed
calibration evidence because a completely new prospective holdout is required.
Legacy observations without attempt-scoped event binding remain excluded.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    MATRIX_DIR,
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)
from export_h800_observations import validate_canonical_observation, write_jsonl
from fit_h800_lora_source_disjoint_recalibration_v1 import (
    CANDIDATE_SCHEMA,
    DEFAULT_COVERAGE,
    DEFAULT_FROZEN_BASELINE,
    DEFAULT_HARDWARE,
    DEFAULT_INVENTORY,
    FITTED_VARIANTS,
    VARIANT_FEATURES,
    VARIANT_M1,
    VARIANT_M2,
    VARIANT_M3,
    _anchor_diagnostics,
    _build_record_pair,
    _collapse_records,
    _evaluate,
    _fit_bundle,
    _input_binding,
    _is_critical_lora,
    _nested_loso,
    _observation_binding,
    _outcome,
    _read_jsonl,
    _select_variant,
    _source_id,
    _write_candidate,
)
from fit_h800_profile_aware_memory_challenger_v1 import _export_exact_jobs
from h800_native_memory_calibration import _inventory_models, native_admission_reason


SCHEMA = "sft_h800_historical_memory_migration_refit/v1"
IMPLEMENTATION_VERSION = (
    "sft_h800_historical_memory_migration_refit/"
    "2026-08-05.exact-history-effective-sequence-v1"
)
HISTORICAL_GROUP_PREFIX = "historical_public_connected_component_"
HISTORICAL_DATASETS = (
    "short_512",
    "multiturn_2048",
    "multiturn_4096",
    "longtail_8192",
    "longcontext_32768",
)

DEFAULT_CANONICAL = ARTIFACT_DIR / "canonical_h800_observations.jsonl"
DEFAULT_THEORY_BASIS = ARTIFACT_DIR / "h800_theory_basis.json"
DEFAULT_DATASET_ANALYSIS = ARTIFACT_DIR / "dataset_analysis.json"
DEFAULT_NEW_QUEUE = MATRIX_DIR / "h800_lora_source_disjoint_jobs_v1.jsonl"
DEFAULT_CURRENT_OLD = (
    ARTIFACT_DIR / "h800_profile_aware_memory_calibration_observations_v1.jsonl"
)
DEFAULT_CURRENT_REPORT = (
    ROOT
    / "diagnostics"
    / "h800_lora_memory_recalibration_20260804"
    / "nested_ablation_results.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "diagnostics" / "h800_historical_memory_migration_20260805"
)


def _canonical_content_hash(row: Mapping[str, Any]) -> str:
    content = row.get("messages") or row.get("conversations") or row.get(
        "conversation"
    )
    if content is None:
        content = {
            key: row.get(key)
            for key in ("instruction", "input", "output", "prompt", "response")
            if key in row
        }
    payload = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _content_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                hashes.add(_canonical_content_hash(json.loads(line)))
    return hashes


def _dataset_registry(
    dataset_analysis_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    analysis = read_json(dataset_analysis_path)
    datasets = analysis.get("datasets") or {}
    registry: dict[str, dict[str, Any]] = {}
    for dataset_id in HISTORICAL_DATASETS:
        entry = datasets.get(dataset_id)
        if not isinstance(entry, Mapping):
            raise ValueError(f"dataset analysis has no entry for {dataset_id}")
        data_path = Path(str(entry["file"])).resolve()
        profile_path = (
            ARTIFACT_DIR
            / "dataset_profiles"
            / f"{dataset_id}.qwen3_nothink.jsonl"
        ).resolve()
        if not data_path.is_file() or not profile_path.is_file():
            raise ValueError(f"historical data/profile missing for {dataset_id}")
        expected_data_sha = str(entry.get("sha256") or "")
        actual_data_sha = sha256_file(data_path)
        if expected_data_sha and expected_data_sha != actual_data_sha:
            raise ValueError(f"historical data SHA-256 mismatch for {dataset_id}")
        maximum = 0
        profile_rows = 0
        with profile_path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                maximum = max(maximum, int(row["total_tokens"]))
                profile_rows += 1
        if profile_rows != int(entry.get("samples") or 0) or maximum <= 0:
            raise ValueError(f"historical profile row/max audit failed for {dataset_id}")
        registry[dataset_id] = {
            "dataset_id": dataset_id,
            "data_path": str(data_path),
            "data_sha256": actual_data_sha,
            "data_rows": int(entry["samples"]),
            "profile_path": str(profile_path),
            "profile_sha256": sha256_file(profile_path),
            "profile_rows": profile_rows,
            "raw_profile_max": maximum,
            "content_hashes": _content_hashes(data_path),
        }
    return registry, analysis


def _connected_components(
    registry: Mapping[str, Mapping[str, Any]],
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    ids = sorted(registry)
    adjacency: dict[str, set[str]] = {dataset_id: set() for dataset_id in ids}
    overlaps: list[dict[str, Any]] = []
    for index, left in enumerate(ids):
        left_hashes = set(registry[left]["content_hashes"])
        for right in ids[index + 1 :]:
            count = len(left_hashes & set(registry[right]["content_hashes"]))
            overlaps.append({"left": left, "right": right, "rows": count})
            if count:
                adjacency[left].add(right)
                adjacency[right].add(left)
    components: list[list[str]] = []
    remaining = set(ids)
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(adjacency[node] - component))
        remaining -= component
        components.append(sorted(component))
    return sorted(components), overlaps


def _legacy_basis_ids(path: Path) -> tuple[set[str], dict[str, Any]]:
    basis = read_json(path)
    rows = []
    for row in basis.get("records") or []:
        if not isinstance(row, Mapping):
            continue
        outcome = row.get("outcome")
        if isinstance(outcome, Mapping):
            outcome = outcome.get("class")
        if (
            (row.get("route") or {}).get("memory_boundary") is True
            and str(outcome or "").lower() in {"success", "oom"}
        ):
            rows.append(row)
    ids = {str(row["observation_id"]) for row in rows}
    return ids, {
        "rows": len(rows),
        "outcomes": dict(
            Counter(
                str(
                    (row.get("outcome") or {}).get("class")
                    if isinstance(row.get("outcome"), Mapping)
                    else row.get("outcome")
                ).lower()
                for row in rows
            )
        ),
        "evidence_tiers": dict(
            Counter(str(row.get("evidence_tier")) for row in rows)
        ),
    }


def _read_historical_exact(
    canonical_path: Path,
    legacy_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    admitted: list[dict[str, Any]] = []
    admission = Counter()
    legacy_found: set[str] = set()
    legacy_exclusions = Counter()
    total = 0
    with canonical_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            observation_id = str(row.get("observation_id"))
            if observation_id in legacy_ids:
                legacy_found.add(observation_id)
                for reason in (row.get("outcome") or {}).get(
                    "calibration_exclusion_reasons", []
                ):
                    legacy_exclusions[str(reason)] += 1
            reason = native_admission_reason(row)
            admission[reason] += 1
            if reason != "admitted":
                continue
            job = (row.get("configuration") or {}).get("job") or {}
            if str(job.get("dataset_id")) in HISTORICAL_DATASETS:
                admitted.append(row)
    if legacy_found != legacy_ids:
        raise ValueError(
            f"legacy basis/canonical binding mismatch: {len(legacy_found)} != "
            f"{len(legacy_ids)}"
        )
    if len(admitted) != 325:
        raise ValueError(f"expected 325 exact historical observations, got {len(admitted)}")
    for row in admitted:
        errors = validate_canonical_observation(row)
        if errors:
            raise ValueError(f"historical canonical validation failed: {errors}")
        attempt = row.get("attempt") or {}
        quality = row.get("quality") or {}
        if (
            attempt.get("attempt_scoped") is not True
            or quality.get("event_attempt_binding_complete") is not True
            or quality.get("terminal_label_verified") is not True
        ):
            raise ValueError("admitted historical observation lacks exact attempt binding")
    return admitted, {
        "canonical_rows": total,
        "admission_reasons": dict(admission),
        "legacy_basis_rows_found": len(legacy_found),
        "legacy_calibration_exclusion_reasons": dict(legacy_exclusions),
    }


def _current_data_paths(
    new_observations: Sequence[Mapping[str, Any]],
    old_observations: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for row in [*new_observations, *old_observations]:
        job = (row.get("configuration") or {}).get("job") or {}
        raw = job.get("data_path")
        if not raw:
            raise ValueError("current observation has no data path")
        path = Path(str(raw)).resolve()
        if not path.is_file():
            raise ValueError(f"current data path missing: {path}")
        paths[str(job.get("dataset_id"))] = path
    return paths


def _inject_historical_contract(
    observation: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
    source_group_by_dataset: Mapping[str, str],
) -> dict[str, Any]:
    migrated = copy.deepcopy(dict(observation))
    configuration = migrated["configuration"]
    job = configuration["job"]
    dataset_id = str(job["dataset_id"])
    binding = registry[dataset_id]
    original_partition = copy.deepcopy(configuration.get("calibration_partition"))
    original_role = (
        str((original_partition or {}).get("role") or "unknown").lower()
    )
    job["data_path"] = binding["data_path"]
    job["data_sha256"] = binding["data_sha256"]
    job["dataset_profile_path"] = binding["profile_path"]
    job["dataset_profile_sha256"] = binding["profile_sha256"]
    job["historical_migration"] = {
        "original_partition": original_partition,
        "original_role": original_role,
        "former_holdout_reclassified_as_consumed_calibration": (
            original_role == "holdout"
        ),
        "future_validation_eligible": False,
    }
    configuration["calibration_partition"] = {
        "role": "calibration",
        "policy": "historical_content_connected_source_group_v1",
        "split_unit_id": source_group_by_dataset[dataset_id],
        "original_role": original_role,
    }
    return migrated


def _build_pairs(
    observations: Sequence[dict[str, Any]],
    *,
    model_by_id: Mapping[str, dict[str, Any]],
    fixed_lora: Mapping[str, Any],
    hardware: Mapping[str, Any],
    origin: str,
    padding_cache: dict[tuple[str, int, int], dict[str, Any]],
    raw_max_cache: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cutoff_rows: list[dict[str, Any]] = []
    effective_rows: list[dict[str, Any]] = []
    for observation in observations:
        cutoff, effective = _build_record_pair(
            observation,
            model_by_id=model_by_id,
            fixed_lora=dict(fixed_lora),
            hardware=dict(hardware),
            padding_cache=padding_cache,
            raw_max_cache=raw_max_cache,
            origin=origin,
        )
        cutoff_rows.append(cutoff)
        effective_rows.append(effective)
    return cutoff_rows, effective_rows


def _metric_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    keys = (
        "allocated_center_source_equal_mape",
        "reserved_center_source_equal_mape",
        "reserved_upper_source_equal_coverage",
        "admission_recall",
        "unsafe_success_admitted",
        "false_safe_oom",
    )
    return {
        key: {
            "before": before.get(key),
            "after": after.get(key),
            "delta": (
                float(after[key]) - float(before[key])
                if before.get(key) is not None and after.get(key) is not None
                else None
            ),
        }
        for key in keys
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
    parser.add_argument("--current-report", type=Path, default=DEFAULT_CURRENT_REPORT)
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
    original_role_counts = Counter(
        str(
            (((row.get("configuration") or {}).get("calibration_partition") or {}).get(
                "role"
            ))
        )
        for row in historical_exact
    )
    historical_outcomes = Counter(
        str((row.get("outcome") or {}).get("class")) for row in historical_exact
    )

    queue = _read_jsonl(args.new_queue)
    if len(queue) != 77:
        raise ValueError("current new queue must contain 77 jobs")
    print("exporting current exact 77-job observations", flush=True)
    new_observations = _export_exact_jobs(queue)
    old_observations = _read_jsonl(args.current_old)
    if len(old_observations) != 28:
        raise ValueError("current old snapshot must contain 28 observations")
    for row in [*new_observations, *old_observations]:
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

    migrated_historical = [
        _inject_historical_contract(
            row,
            registry=registry,
            source_group_by_dataset=source_group_by_dataset,
        )
        for row in historical_exact
    ]

    inventory = read_json(args.inventory)
    model_by_id, fixed_lora = _inventory_models(inventory)
    hardware = read_json(args.hardware)
    padding_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    raw_max_cache: dict[str, int] = {}
    current_cutoff, current_effective = _build_pairs(
        [*old_observations, *new_observations],
        model_by_id=model_by_id,
        fixed_lora=fixed_lora,
        hardware=hardware,
        origin="current_19_sources",
        padding_cache=padding_cache,
        raw_max_cache=raw_max_cache,
    )
    historical_cutoff, historical_effective = _build_pairs(
        migrated_historical,
        model_by_id=model_by_id,
        fixed_lora=fixed_lora,
        hardware=hardware,
        origin="historical_exact_325",
        padding_cache=padding_cache,
        raw_max_cache=raw_max_cache,
    )

    # Historical boundary sweeps can contain both successes and OOMs for an
    # otherwise identical configuration.  Preserve both censoring classes
    # instead of treating them as contradictory repeats or letting one class
    # erase the other during collapse.
    historical_cluster_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    for row in historical_effective:
        historical_cluster_outcomes[str(row["cluster_id"])][_outcome(row)] += 1
    historical_mixed_outcome_groups = [
        {
            "base_cluster_id": cluster_id,
            "outcomes": dict(counts),
        }
        for cluster_id, counts in sorted(historical_cluster_outcomes.items())
        if len(counts) > 1
    ]
    for row in [*historical_cutoff, *historical_effective]:
        row["cluster_id"] = f"{row['cluster_id']}-{_outcome(row)}"

    current_sources = {_source_id(row) for row in current_effective}
    historical_sources = {_source_id(row) for row in historical_effective}
    if len(current_sources) != 19 or len(historical_sources) != len(components):
        raise ValueError("independent source grouping audit failed")
    if current_sources & historical_sources:
        raise ValueError("historical/current source ids overlap")

    collapsed_cutoff, collapsed_effective, collapse = _collapse_records(
        [*current_cutoff, *historical_cutoff],
        [*current_effective, *historical_effective],
    )
    print(
        f"combined raw={len(current_effective) + len(historical_effective)} "
        f"unique={len(collapsed_effective)} sources="
        f"{len(current_sources | historical_sources)}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bindings_path = args.output_dir / "historical_observation_bindings.jsonl"
    records_path = args.output_dir / "migrated_historical_effective_records.jsonl"
    write_jsonl(
        bindings_path,
        [
            {
                **_observation_binding(row),
                "dataset_id": ((row.get("configuration") or {}).get("job") or {}).get(
                    "dataset_id"
                ),
                "original_role": (
                    ((row.get("configuration") or {}).get("calibration_partition") or {}).get(
                        "role"
                    )
                ),
                "source_group": source_group_by_dataset[
                    str(
                        ((row.get("configuration") or {}).get("job") or {}).get(
                            "dataset_id"
                        )
                    )
                ],
            }
            for row in historical_exact
        ],
    )
    write_jsonl(records_path, historical_effective)

    nested: dict[str, Any] = {}
    for variant in FITTED_VARIANTS:
        nested[variant] = _nested_loso(
            collapsed_effective,
            variant=variant,
            coverage=args.coverage,
        )
        print(
            f"{variant}: {json.dumps(nested[variant]['metrics'], sort_keys=True)}",
            flush=True,
        )
    all_mechanism_selection = _select_variant(nested)
    candidate_scope_nested = {
        variant: {
            "metrics": _evaluate(
                [
                    row
                    for row in report_variant.get("details") or []
                    if row.get("critical_lora") is True
                ]
            ),
            "coefficient_stability": report_variant.get("coefficient_stability")
            or {},
        }
        for variant, report_variant in nested.items()
    }
    selection = _select_variant(candidate_scope_nested)
    selected_variant = selection["selected_variant"]
    full_bundle = (
        _fit_bundle(
            collapsed_effective,
            names=VARIANT_FEATURES[selected_variant],
            coverage=args.coverage,
            diagnostic_max_fallback=False,
        )
        if selected_variant
        else None
    )

    selected_details = (
        (nested.get(selected_variant) or {}).get("details") or []
        if selected_variant
        else []
    )
    combined_metrics = _evaluate(selected_details)
    current_metrics = _evaluate(
        [row for row in selected_details if row.get("origin") == "current_19_sources"]
    )
    historical_metrics = _evaluate(
        [row for row in selected_details if row.get("origin") == "historical_exact_325"]
    )
    current_critical_metrics = _evaluate(
        [
            row
            for row in selected_details
            if row.get("origin") == "current_19_sources"
            and row.get("critical_lora") is True
        ]
    )
    combined_critical_metrics = _evaluate(
        [row for row in selected_details if row.get("critical_lora") is True]
    )

    current_report = read_json(args.current_report)
    before_overall = current_report["nested_M1_M3"][VARIANT_M1]["metrics"]
    before_critical = current_report["selected_scope_diagnostics"][
        "critical_lora_2gpu_zero2_gc_off_nonpacking"
    ]

    critical_key = json.dumps(["lora", 2, False, 2, False], separators=(",", ":"))
    critical_expansion = (
        (((full_bundle or {}).get("reservation_expansion") or {}).get("entries") or {}).get(
            critical_key
        )
        or {}
    )
    ratio_stability = selection.get("ratio_stability") or {}
    expansion_upper = critical_expansion.get("expansion_upper")
    empirical_max_expansion = critical_expansion.get("empirical_max_expansion")
    stage_two_triggers = {
        "admission_recall_below_0p90": (
            float(current_critical_metrics.get("admission_recall") or 0.0) < 0.90
        ),
        "critical_reserved_center_source_equal_mape_above_0p20": (
            float(
                current_critical_metrics.get("reserved_center_source_equal_mape")
                or math.inf
            )
            > 0.20
        ),
        "critical_expansion_threshold_is_empirical_max": (
            expansion_upper is not None
            and empirical_max_expansion is not None
            and math.isclose(
                float(expansion_upper),
                float(empirical_max_expansion),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ),
        "ratio_coefficients_unstable": any(
            ratio_stability.get(variant) is False
            for variant in (VARIANT_M2, VARIANT_M3)
        ),
    }
    stage_two_required = bool(selected_variant) and any(stage_two_triggers.values())

    registry_report = {
        dataset_id: {
            key: value
            for key, value in binding.items()
            if key != "content_hashes"
        }
        for dataset_id, binding in sorted(registry.items())
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "shadow_candidate_requires_stage_two_calibration"
            if selected_variant and stage_two_required
            else (
                "shadow_candidate_requires_prospective_holdout"
                if selected_variant
                else "no_candidate_selected"
            )
        ),
        "publishable": False,
        "production_model_mutated": False,
        "inputs": {
            "canonical_observations": _input_binding(args.canonical),
            "theory_basis": _input_binding(args.theory_basis),
            "dataset_analysis": _input_binding(args.dataset_analysis),
            "new_queue": _input_binding(args.new_queue),
            "current_old_observations": _input_binding(args.current_old),
            "current_m1_report": _input_binding(args.current_report),
            "model_inventory": _input_binding(args.inventory),
            "hardware": _input_binding(args.hardware),
            "frozen_baseline": _input_binding(args.frozen_baseline),
        },
        "historical_migration_audit": {
            "strict_exact_observations_admitted": len(historical_exact),
            "strict_exact_outcomes": dict(historical_outcomes),
            "original_partition_roles": dict(original_role_counts),
            "former_holdout_rows_reclassified_as_consumed_calibration": int(
                original_role_counts.get("holdout", 0)
            ),
            "legacy_non_attempt_scoped_observations_excluded": legacy_basis_audit,
            "canonical_audit": canonical_audit,
            "dataset_registry": registry_report,
            "historical_dataset_content_overlaps": historical_overlaps,
            "historical_connected_components": components,
            "historical_independent_source_groups": sorted(historical_sources),
            "historical_mixed_outcome_configuration_groups": (
                historical_mixed_outcome_groups
            ),
            "historical_mixed_outcome_configuration_group_count": len(
                historical_mixed_outcome_groups
            ),
            "historical_current_content_overlap_rows": historical_current_overlap,
            "historical_current_content_overlap_total": sum(
                historical_current_overlap.values()
            ),
            "future_validation_policy": (
                "all migrated historical rows are consumed calibration evidence and "
                "must never be counted as prospective validation"
            ),
        },
        "combined_data_audit": {
            "current_raw_observations": len(current_effective),
            "historical_raw_observations": len(historical_effective),
            "combined_raw_observations": len(current_effective)
            + len(historical_effective),
            "current_independent_sources": len(current_sources),
            "historical_independent_sources": len(historical_sources),
            "combined_independent_sources": len(current_sources | historical_sources),
            "collapse": collapse,
            "collapsed_outcomes": dict(
                Counter(_outcome(row) for row in collapsed_effective)
            ),
            "source_weighting": "each independent source group has total weight one",
        },
        "effective_sequence_anchor_diagnostics": _anchor_diagnostics(
            collapsed_cutoff, collapsed_effective
        ),
        "nested_M1_M3": nested,
        "selection": {
            "candidate_scope": (
                "LoRA + ZeRO-2 + GC-off + 2 GPU + non-packing"
            ),
            "candidate_scope_selection": selection,
            "all_mechanism_diagnostic_selection": all_mechanism_selection,
            "all_mechanism_failures_do_not_expand_candidate_scope": True,
        },
        "selected_scope_diagnostics": {
            "combined_all": combined_metrics,
            "current_19_sources_all": current_metrics,
            "historical_exact_only": historical_metrics,
            "combined_critical_lora": combined_critical_metrics,
            "current_19_sources_critical_lora": current_critical_metrics,
        },
        "comparison_with_original_M1": {
            "current_19_sources_all": _metric_delta(before_overall, current_metrics),
            "current_19_sources_critical_lora": _metric_delta(
                before_critical, current_critical_metrics
            ),
            "comparison_contract": (
                "same current 19 source validation rows; migrated model folds additionally "
                "train on the historical connected source group"
            ),
        },
        "selected_full_fit": {
            "variant": selected_variant,
            "bundle": full_bundle,
            "critical_expansion": critical_expansion,
        },
        "release_gate": {
            "stage_two_calibration_required": stage_two_required,
            "stage_two_triggers": stage_two_triggers,
            "stage_two_plan": {
                "new_independent_sources": 20,
                "jobs_per_source": 3,
                "jobs": 60,
            },
            "formal_prospective_acceptance_passed": False,
            "production_replacement_allowed": False,
        },
    }
    report["report_sha256"] = sha256_json(report)
    report_path = args.output_dir / "historical_migration_refit_report.json"
    write_json(report_path, report)

    candidate_path = args.output_dir / "candidate_model.json"
    candidate = _write_candidate(
        candidate_path,
        selected_variant=selected_variant,
        bundle=full_bundle,
        source_count=len(current_sources | historical_sources),
        report_sha256=report["report_sha256"],
        stage_two_required=stage_two_required,
        stage_two_triggers=stage_two_triggers,
    )
    manifest = {
        "schema": "sft_h800_historical_memory_migration_manifest/v1",
        "generated_at_utc": report["generated_at_utc"],
        "files": {
            path.name: _input_binding(path)
            for path in (bindings_path, records_path, report_path, candidate_path)
        },
        "selected_variant": selected_variant,
        "candidate_sha256": candidate["candidate_sha256"],
        "publishable": False,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(args.output_dir / "output_manifest.json", manifest)
    print(
        f"selected={selected_variant} stage_two_required={stage_two_required} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
