#!/usr/bin/env python3
"""Ablate and recalibrate the H800 M1 safety upper bound.

The M1 effective-sequence center regression is held fixed across all safety
variants.  Variant selection uses nested leave-source-out predictions from the
current 105 observations plus the original 158 historical calibration rows.
The original 167 historical holdout rows never enter fitting or selection and
are scored only after the safety variant has been selected.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
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
    _fit_residual_hierarchy,
    _fit_ridge,
    _input_binding,
    _inventory_models,
    _is_critical_lora,
    _mechanism_key,
    _mode_gc_key,
    _observed_allocated,
    _observed_reserved,
    _oom_lower,
    _outcome,
    _predict_center,
    _predict_stack,
    _prediction_detail,
    _read_jsonl,
    _safe_limit,
    _select_residual,
    _source_id,
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
    _read_historical_exact,
)
from refit_h800_with_fixed_historical_holdout_v2 import (
    _add_outcome_to_cluster_id,
    _original_role,
)


SCHEMA = "sft_h800_m1_safety_upper_ablation/v2"
CANDIDATE_SCHEMA = "sft_h800_m1_safety_upper_shadow_candidate/v2"
IMPLEMENTATION_VERSION = (
    "sft_h800_m1_safety_upper_ablation/"
    "2026-08-05.single-joint-tail-no-marginal-max-product-v2"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "diagnostics"
    / "h800_m1_safety_upper_v2_20260805"
)

S0_STACKED = "S0_current_stacked_upper"
S1_RESERVED_ONLY = "S1_reserved_success_tail_only"
S2_RESERVED_OOM = "S2_reserved_success_tail_plus_oom_guard"
S3_JOINT = "S3_joint_center_single_success_oom_tail"
SAFETY_VARIANTS = (
    S0_STACKED,
    S1_RESERVED_ONLY,
    S2_RESERVED_OOM,
    S3_JOINT,
)
SELECTABLE_VARIANTS = (S0_STACKED, S2_RESERVED_OOM, S3_JOINT)
VARIANT_PRIORITY = {
    S3_JOINT: 0,
    S2_RESERVED_OOM: 1,
    S0_STACKED: 2,
}


def _source_balanced_log_center(
    rows: Sequence[Mapping[str, Any]], key: str | None
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if _outcome(row) != "success":
            continue
        allocated = _observed_allocated(row)
        reserved = _observed_reserved(row)
        if allocated is None or reserved is None:
            continue
        bucket = str(row[key]) if key else "pooled"
        grouped[bucket][_source_id(row)].append(
            math.log(float(reserved) / float(allocated))
        )
    result: dict[str, dict[str, Any]] = {}
    for bucket, by_source in sorted(grouped.items()):
        source_centers = [
            statistics.median(values) for values in by_source.values()
        ]
        result[bucket] = {
            "available": bool(source_centers),
            "independent_sources": len(source_centers),
            "rows": sum(len(values) for values in by_source.values()),
            "log_center": (
                statistics.median(source_centers) if source_centers else None
            ),
            "aggregation": (
                "median within source, then median across independent sources"
            ),
        }
    return result


def _records_with_keys(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "mechanism_key": _mechanism_key(row),
            "mode_gc_key": _mode_gc_key(row),
        }
        for row in records
    ]


def _fit_expansion_center(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    keyed = _records_with_keys(records)
    return {
        "score": "log(observed_reserved_bytes / observed_allocated_bytes)",
        "exact": _source_balanced_log_center(keyed, "mechanism_key"),
        "mode_gc": _source_balanced_log_center(keyed, "mode_gc_key"),
        "pooled": _source_balanced_log_center(keyed, None).get(
            "pooled",
            {"available": False, "log_center": None},
        ),
        "hierarchy": ["exact_mechanism", "mode_x_gc", "pooled"],
    }


def _select_expansion_center(
    record: Mapping[str, Any], hierarchy: Mapping[str, Any]
) -> dict[str, Any]:
    exact = (hierarchy.get("exact") or {}).get(_mechanism_key(record))
    if isinstance(exact, Mapping) and exact.get("available") is True:
        return {"source": "exact_mechanism", **dict(exact)}
    coarse = (hierarchy.get("mode_gc") or {}).get(_mode_gc_key(record))
    if isinstance(coarse, Mapping) and coarse.get("available") is True:
        return {"source": "mode_x_gc", **dict(coarse)}
    pooled = hierarchy.get("pooled") or {}
    if isinstance(pooled, Mapping) and pooled.get("available") is True:
        return {"source": "pooled", **dict(pooled)}
    return {"available": False, "source": None, "log_center": None}


def _joint_base(
    record: Mapping[str, Any],
    *,
    allocated_center: float,
    reserved_center: float,
    expansion_center: Mapping[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    selected = _select_expansion_center(record, expansion_center)
    if selected.get("available") is not True:
        return None, selected
    allocation_path = allocated_center * math.exp(
        float(selected["log_center"])
    )
    return max(reserved_center, allocation_path), {
        **selected,
        "allocation_center_expansion_bytes": allocation_path,
    }


def _fit_joint_safety(
    records: Sequence[Mapping[str, Any]],
    *,
    base_bundle: Mapping[str, Any],
    coverage: float,
    diagnostic_max_fallback: bool,
) -> dict[str, Any]:
    names = list(base_bundle["feature_names"])
    allocated_alpha = float(base_bundle["allocated_model"]["alpha"])
    reserved_alpha = float(base_bundle["reserved_model"]["alpha"])
    scores: list[dict[str, Any]] = []
    sources = sorted({_source_id(row) for row in records})
    for held in sources:
        training = [row for row in records if _source_id(row) != held]
        evaluation = [row for row in records if _source_id(row) == held]
        allocated_model = _fit_ridge(
            training,
            names=names,
            alpha=allocated_alpha,
            target="allocated",
        )
        reserved_model = _fit_ridge(
            training,
            names=names,
            alpha=reserved_alpha,
            target="reserved",
        )
        expansion_center = _fit_expansion_center(training)
        for row in evaluation:
            allocated_center = _predict_center(row, allocated_model)
            reserved_center = _predict_center(row, reserved_model)
            base, selected_expansion = _joint_base(
                row,
                allocated_center=allocated_center,
                reserved_center=reserved_center,
                expansion_center=expansion_center,
            )
            if base is None:
                continue
            score_base = {
                "source_id": held,
                "mechanism_key": _mechanism_key(row),
                "mode_gc_key": _mode_gc_key(row),
                "outcome": _outcome(row),
                "expansion_center_source": selected_expansion.get("source"),
            }
            if _outcome(row) == "success":
                reserved = _observed_reserved(row)
                if reserved is not None:
                    scores.append(
                        {
                            **score_base,
                            "score": math.log(float(reserved) / base),
                            "constraint": "observed_reserved",
                        }
                    )
            elif _outcome(row) == "oom":
                safe = _safe_limit(row)
                lower = _oom_lower(row)
                floors = [
                    float(value)
                    for value in (
                        lower,
                        (safe + 1.0) if safe is not None else None,
                    )
                    if value is not None
                ]
                if floors:
                    scores.append(
                        {
                            **score_base,
                            "score": math.log(max(floors) / base),
                            "constraint": "oom_right_censor_lower",
                        }
                    )
    return {
        "definition": (
            "base=max(reserved_center, allocated_center*source-balanced median "
            "reserved/allocated); one source-level upper quantile is fitted on "
            "success reserved observations and OOM right-censor lower bounds"
        ),
        "expansion_center": _fit_expansion_center(records),
        "joint_residual_upper": _fit_residual_hierarchy(
            scores,
            coverage=coverage,
            diagnostic_max_fallback=diagnostic_max_fallback,
        ),
        "inner_oof_scores": {
            "rows": len(scores),
            "success": sum(row["outcome"] == "success" for row in scores),
            "oom": sum(row["outcome"] == "oom" for row in scores),
            "independent_sources": len(
                {str(row["source_id"]) for row in scores}
            ),
        },
    }


def _fit_safety_bundle(
    records: Sequence[Mapping[str, Any]],
    *,
    coverage: float,
    diagnostic_max_fallback: bool,
) -> dict[str, Any]:
    bundle = _fit_bundle(
        records,
        names=VARIANT_FEATURES[VARIANT_M1],
        coverage=coverage,
        diagnostic_max_fallback=diagnostic_max_fallback,
    )
    bundle["joint_safety"] = _fit_joint_safety(
        records,
        base_bundle=bundle,
        coverage=coverage,
        diagnostic_max_fallback=diagnostic_max_fallback,
    )
    return bundle


def _common_centers(
    record: Mapping[str, Any], bundle: Mapping[str, Any]
) -> tuple[float, float]:
    return (
        _predict_center(record, bundle["allocated_model"]),
        _predict_center(record, bundle["reserved_model"]),
    )


def _predict_safety(
    record: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    safety_variant: str,
) -> dict[str, Any]:
    allocated_center, reserved_center = _common_centers(record, bundle)
    if safety_variant == S0_STACKED:
        prediction = dict(_predict_stack(record, bundle))
        prediction["safety_variant"] = safety_variant
        candidates = {
            "direct_reserved_upper": prediction.get("direct_reserved_upper_bytes"),
            "allocated_expansion_upper": prediction.get("expansion_upper_bytes"),
            "oom_upper": prediction.get("oom_upper_bytes"),
        }
        finite = {
            key: float(value)
            for key, value in candidates.items()
            if value is not None and math.isfinite(float(value))
        }
        prediction["upper_components"] = finite
        prediction["dominant_upper_component"] = (
            max(finite, key=finite.get) if finite else None
        )
        return prediction

    reserved_tail = _select_residual(record, bundle["reserved_residual_upper"])
    if reserved_tail.get("available") is not True:
        return {
            "available": False,
            "allocated_center_bytes": allocated_center,
            "reserved_center_bytes": reserved_center,
            "upper_bytes": None,
            "safety_variant": safety_variant,
            "issues": ["reserved_success_tail_unavailable"],
        }
    direct_reserved_upper = reserved_center * math.exp(
        max(0.0, float(reserved_tail["log_upper"]))
    )
    if safety_variant == S1_RESERVED_ONLY:
        return {
            "available": True,
            "allocated_center_bytes": allocated_center,
            "reserved_center_bytes": reserved_center,
            "upper_bytes": direct_reserved_upper,
            "direct_reserved_upper_bytes": direct_reserved_upper,
            "reserved_tail": reserved_tail,
            "safety_variant": safety_variant,
            "upper_components": {
                "direct_reserved_upper": direct_reserved_upper,
            },
            "dominant_upper_component": "direct_reserved_upper",
            "issues": ["diagnostic_success_only_no_oom_guard"],
        }

    if safety_variant == S2_RESERVED_OOM:
        oom = (bundle.get("oom_guards") or {}).get(_mechanism_key(record))
        oom_upper = None
        if isinstance(oom, Mapping):
            oom_upper = reserved_center * math.exp(
                max(0.0, float(oom["log_lower"]))
            )
        components = {"direct_reserved_upper": direct_reserved_upper}
        if oom_upper is not None:
            components["oom_upper"] = oom_upper
        dominant = max(components, key=components.get)
        return {
            "available": True,
            "allocated_center_bytes": allocated_center,
            "reserved_center_bytes": reserved_center,
            "upper_bytes": max(components.values()),
            "direct_reserved_upper_bytes": direct_reserved_upper,
            "oom_upper_bytes": oom_upper,
            "reserved_tail": reserved_tail,
            "oom_guard": dict(oom or {}),
            "safety_variant": safety_variant,
            "upper_components": components,
            "dominant_upper_component": dominant,
            "issues": ([] if oom_upper is not None else ["no_exact_oom_guard"]),
        }

    if safety_variant == S3_JOINT:
        joint = bundle["joint_safety"]
        base, selected_expansion = _joint_base(
            record,
            allocated_center=allocated_center,
            reserved_center=reserved_center,
            expansion_center=joint["expansion_center"],
        )
        tail = _select_residual(record, joint["joint_residual_upper"])
        if base is None or tail.get("available") is not True:
            return {
                "available": False,
                "allocated_center_bytes": allocated_center,
                "reserved_center_bytes": reserved_center,
                "upper_bytes": None,
                "safety_variant": safety_variant,
                "issues": ["joint_center_or_tail_unavailable"],
            }
        upper = base * math.exp(max(0.0, float(tail["log_upper"])))
        return {
            "available": True,
            "allocated_center_bytes": allocated_center,
            "reserved_center_bytes": reserved_center,
            "joint_center_bytes": base,
            "upper_bytes": upper,
            "joint_tail": tail,
            "expansion_center": selected_expansion,
            "safety_variant": safety_variant,
            "upper_components": {"joint_center_single_tail": upper},
            "dominant_upper_component": "joint_center_single_tail",
            "issues": [],
        }
    raise ValueError(f"unknown safety variant: {safety_variant}")


def _detail(
    record: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    safety_variant: str,
) -> dict[str, Any]:
    detail = _prediction_detail(record, prediction, variant=VARIANT_M1)
    detail["safety_variant"] = safety_variant
    detail["upper_components"] = dict(prediction.get("upper_components") or {})
    detail["dominant_upper_component"] = prediction.get(
        "dominant_upper_component"
    )
    detail["upper_over_safe_limit"] = (
        float(detail["upper_bytes"]) / float(detail["safe_limit_bytes"])
        if detail.get("upper_bytes") is not None
        else None
    )
    detail["issues"] = list(prediction.get("issues") or [])
    selector = record.get("selector") or {}
    detail["zero_stage"] = selector.get("zero_stage")
    detail["gradient_checkpointing"] = bool(
        selector.get("gradient_checkpointing")
    )
    detail["packing"] = bool(selector.get("packing"))
    return detail


def _nested_safety_ablation(
    records: Sequence[Mapping[str, Any]],
    *,
    coverage: float,
) -> dict[str, Any]:
    sources = sorted({_source_id(row) for row in records})
    details: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in SAFETY_VARIANTS
    }
    folds: list[dict[str, Any]] = []
    for index, held in enumerate(sources, start=1):
        print(f"safety outer fold {index}/{len(sources)} holdout={held}", flush=True)
        training = [row for row in records if _source_id(row) != held]
        evaluation = [row for row in records if _source_id(row) == held]
        bundle = _fit_safety_bundle(
            training,
            coverage=coverage,
            diagnostic_max_fallback=True,
        )
        for row in evaluation:
            for variant in SAFETY_VARIANTS:
                prediction = _predict_safety(
                    row,
                    bundle,
                    safety_variant=variant,
                )
                details[variant].append(
                    _detail(row, prediction, safety_variant=variant)
                )
        folds.append(
            {
                "held_source_id": held,
                "training_sources": len({_source_id(row) for row in training}),
                "evaluation_configurations": len(evaluation),
            }
        )
    return {
        "protocol": (
            "outer leave split_unit_id out; the same fold-specific M1 centers are "
            "used by every safety variant; all tails and OOM guards are regenerated "
            "using training-fold inner leave-source-out scores"
        ),
        "folds": folds,
        "details": details,
    }


def _scope_metrics(
    details: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant, rows in details.items():
        critical = [row for row in rows if row.get("critical_lora") is True]
        current = [
            row
            for row in critical
            if row.get("origin") == "current_19_sources"
        ]
        historical = [
            row
            for row in critical
            if row.get("origin") == "historical_original_calibration_158"
        ]
        result[variant] = {
            "selection_scope_combined_critical_lora": _evaluate(critical),
            "current_19_sources_critical_lora": _evaluate(current),
            "historical_calibration_source_critical_lora": _evaluate(historical),
            "all_mechanisms_diagnostic": _evaluate(rows),
        }
    return result


def _gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "prediction_availability_one": float(
            metrics.get("prediction_availability") or 0.0
        )
        == 1.0,
        "reserved_upper_source_equal_coverage_ge_0p95": float(
            metrics.get("reserved_upper_source_equal_coverage") or 0.0
        )
        >= 0.95,
        "unsafe_success_admitted_zero": int(
            metrics.get("unsafe_success_admitted") or 0
        )
        == 0,
        "false_safe_oom_zero": int(metrics.get("false_safe_oom") or 0) == 0,
    }
    return {"checks": checks, "all_passed": all(checks.values())}


def _select_safety_variant(scope_metrics: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        variant: _gate(
            scope_metrics[variant]["selection_scope_combined_critical_lora"]
        )
        for variant in SAFETY_VARIANTS
    }
    eligible: list[tuple[float, int, str]] = []
    for variant in SELECTABLE_VARIANTS:
        metrics = scope_metrics[variant][
            "selection_scope_combined_critical_lora"
        ]
        recall = metrics.get("admission_recall")
        if gates[variant]["all_passed"] and recall is not None:
            eligible.append(
                (-float(recall), VARIANT_PRIORITY[variant], variant)
            )
    selected = min(eligible)[2] if eligible else None
    return {
        "policy": (
            "select only on combined training-data critical-LoRA nested OOF; "
            "require availability=100%, source-equal upper coverage>=95%, zero "
            "unsafe-success admission and zero false-safe OOM; then maximize safe "
            "success admission recall; S1 is diagnostic-only because it omits OOM evidence"
        ),
        "gates": gates,
        "selectable_variants": list(SELECTABLE_VARIANTS),
        "diagnostic_only_variants": [S1_RESERVED_ONLY],
        "eligible_variants": [row[2] for row in sorted(eligible)],
        "selected_variant": selected,
    }


def _score_records(
    records: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    return {
        variant: [
            _detail(
                row,
                _predict_safety(row, bundle, safety_variant=variant),
                safety_variant=variant,
            )
            for row in records
        ]
        for variant in SAFETY_VARIANTS
    }


def _prepare_records(args: argparse.Namespace) -> dict[str, Any]:
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
    historical_calibration = [
        row for row in historical_exact if _original_role(row) == "calibration"
    ]
    historical_holdout = [
        row for row in historical_exact if _original_role(row) == "holdout"
    ]
    if len(historical_calibration) != 158 or len(historical_holdout) != 167:
        raise ValueError("unexpected original historical role counts")
    calibration_ids = {str(row["observation_id"]) for row in historical_calibration}
    holdout_ids = {str(row["observation_id"]) for row in historical_holdout}
    if calibration_ids & holdout_ids:
        raise ValueError("historical calibration and holdout observation IDs overlap")

    migrated_calibration = [
        _inject_historical_contract(
            row,
            registry=registry,
            source_group_by_dataset=source_group_by_dataset,
        )
        for row in historical_calibration
    ]
    migrated_holdout = [
        _inject_historical_contract(
            row,
            registry=registry,
            source_group_by_dataset=source_group_by_dataset,
        )
        for row in historical_holdout
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
        raise ValueError("historical and current sample content overlap")

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
        migrated_calibration,
        model_by_id=model_by_id,
        fixed_lora=fixed_lora,
        hardware=hardware,
        origin="historical_original_calibration_158",
        padding_cache=padding_cache,
        raw_max_cache=raw_max_cache,
    )
    historical_holdout_cutoff, historical_holdout_effective = _build_pairs(
        migrated_holdout,
        model_by_id=model_by_id,
        fixed_lora=fixed_lora,
        hardware=hardware,
        origin="historical_fixed_holdout_167",
        padding_cache=padding_cache,
        raw_max_cache=raw_max_cache,
    )
    for row in [*historical_holdout_cutoff, *historical_holdout_effective]:
        row["calibration_partition"] = {
            **dict(row.get("calibration_partition") or {}),
            "role": "fixed_holdout",
            "policy": "original_historical_holdout_never_fit_v2",
        }
    for records in (
        historical_calibration_cutoff,
        historical_calibration_effective,
        historical_holdout_cutoff,
        historical_holdout_effective,
    ):
        _add_outcome_to_cluster_id(records)

    calibration_clusters = {
        str(row["cluster_id"]) for row in historical_calibration_effective
    }
    holdout_clusters = {
        str(row["cluster_id"]) for row in historical_holdout_effective
    }
    if calibration_clusters & holdout_clusters:
        raise ValueError("historical calibration and holdout configurations overlap")

    _, current_collapsed, current_collapse = _collapse_records(
        current_cutoff,
        current_effective,
    )
    _, training_collapsed, training_collapse = _collapse_records(
        [*current_cutoff, *historical_calibration_cutoff],
        [*current_effective, *historical_calibration_effective],
    )
    _, holdout_collapsed, holdout_collapse = _collapse_records(
        historical_holdout_cutoff,
        historical_holdout_effective,
    )
    return {
        "current": current_collapsed,
        "training": training_collapsed,
        "holdout": holdout_collapsed,
        "audit": {
            "current_raw_observations": len(current_effective),
            "historical_calibration_raw_observations": len(
                historical_calibration_effective
            ),
            "training_raw_observations": len(current_effective)
            + len(historical_calibration_effective),
            "historical_holdout_raw_observations": len(
                historical_holdout_effective
            ),
            "current_unique_configuration_results": len(current_collapsed),
            "training_unique_configuration_results": len(training_collapsed),
            "historical_holdout_unique_configuration_results": len(
                holdout_collapsed
            ),
            "training_sources": len({_source_id(row) for row in training_collapsed}),
            "training_holdout_observation_id_overlap": 0,
            "training_holdout_configuration_result_overlap": 0,
            "historical_current_sample_content_overlap": sum(
                historical_current_overlap.values()
            ),
            "current_collapse": current_collapse,
            "training_collapse": training_collapse,
            "holdout_collapse": holdout_collapse,
            "historical_connected_components": components,
            "historical_dataset_content_overlaps": historical_overlaps,
            "legacy_non_attempt_scoped_observations_excluded": legacy_basis_audit,
            "canonical_audit": canonical_audit,
        },
    }


def _critical_tail_entry(
    bundle: Mapping[str, Any], safety_variant: str
) -> dict[str, Any]:
    critical_key = json.dumps(["lora", 2, False, 2, False], separators=(",", ":"))
    if safety_variant in (S1_RESERVED_ONLY, S2_RESERVED_OOM):
        entry = (
            (bundle["reserved_residual_upper"].get("exact") or {}).get(
                critical_key
            )
            or {}
        )
        return {"path": "reserved_success_residual", **dict(entry)}
    if safety_variant == S3_JOINT:
        entry = (
            (
                bundle["joint_safety"]["joint_residual_upper"].get("exact")
                or {}
            ).get(critical_key)
            or {}
        )
        return {"path": "joint_success_oom_residual", **dict(entry)}
    expansion = (
        (bundle.get("reservation_expansion") or {}).get("entries") or {}
    ).get(critical_key) or {}
    return {"path": "current_stacked_allocated_expansion", **dict(expansion)}


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

    prepared = _prepare_records(args)
    training = prepared["training"]
    holdout = prepared["holdout"]
    print(
        f"training unique={len(training)} sources="
        f"{len({_source_id(row) for row in training})} holdout unique={len(holdout)}",
        flush=True,
    )
    nested = _nested_safety_ablation(training, coverage=args.coverage)
    scope_metrics = _scope_metrics(nested["details"])
    selection = _select_safety_variant(scope_metrics)
    selected = selection["selected_variant"]
    if selected is None:
        raise ValueError("no safety variant passed the training-data selection gates")

    full_bundle = _fit_safety_bundle(
        training,
        coverage=args.coverage,
        diagnostic_max_fallback=False,
    )
    holdout_details = _score_records(holdout, full_bundle)
    holdout_metrics = {
        variant: {
            "all_mechanisms_diagnostic": _evaluate(rows),
            "critical_lora_diagnostic": _evaluate(
                [row for row in rows if row.get("critical_lora") is True]
            ),
        }
        for variant, rows in holdout_details.items()
    }
    selected_training_details = nested["details"][selected]
    selected_current_details = [
        row
        for row in selected_training_details
        if row.get("origin") == "current_19_sources"
        and row.get("critical_lora") is True
    ]
    selected_holdout_critical = [
        row
        for row in holdout_details[selected]
        if row.get("critical_lora") is True
    ]
    pooled_selected = _evaluate(
        [*selected_current_details, *selected_holdout_critical]
    )
    critical_tail = _critical_tail_entry(full_bundle, selected)
    tail_rank_is_empirical_max = bool(
        critical_tail.get("log_upper") is not None
        and critical_tail.get("empirical_max") is not None
        and math.isclose(
            float(critical_tail["log_upper"]),
            float(critical_tail["empirical_max"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    )
    selected_current_metrics = scope_metrics[selected][
        "current_19_sources_critical_lora"
    ]
    stage_two_triggers = {
        "current_critical_admission_recall_below_0p90": float(
            selected_current_metrics.get("admission_recall") or 0.0
        )
        < 0.90,
        "critical_tail_rank_is_empirical_max": tail_rank_is_empirical_max,
        "formal_new_source_prospective_acceptance_missing": True,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_details_path = args.output_dir / "training_nested_oof_predictions.jsonl"
    write_jsonl(
        training_details_path,
        [
            row
            for variant in SAFETY_VARIANTS
            for row in nested["details"][variant]
        ],
    )
    holdout_details_path = args.output_dir / "historical_holdout_diagnostics.jsonl"
    write_jsonl(
        holdout_details_path,
        [
            row
            for variant in SAFETY_VARIANTS
            for row in holdout_details[variant]
        ],
    )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "safety_upper_shadow_challenger_selected",
        "publishable": False,
        "production_model_mutated": False,
        "center_model_contract": {
            "variant": VARIANT_M1,
            "effective_sequence": (
                "round_up(min(cutoff_len, profile_max), 8); packing uses cutoff_len"
            ),
            "same_fold_specific_centers_across_all_safety_variants": True,
            "center_refit_is_not_the_ablation_variable": True,
        },
        "safety_variant_definitions": {
            S0_STACKED: (
                "current max(direct reserved tail, allocated tail times upper "
                "reservation expansion, OOM guard)"
            ),
            S1_RESERVED_ONLY: (
                "direct reserved-center success tail only; diagnostic, no OOM evidence"
            ),
            S2_RESERVED_OOM: (
                "direct reserved-center success tail plus exact-mechanism OOM guard"
            ),
            S3_JOINT: (
                "max(reserved center, allocated center times source-balanced central "
                "reservation expansion), followed by one joint success/OOM source tail"
            ),
        },
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
        "data_audit": prepared["audit"],
        "nested_protocol": {
            "protocol": nested["protocol"],
            "folds": nested["folds"],
            "historical_fixed_holdout_used_for_selection": False,
        },
        "training_nested_metrics": scope_metrics,
        "selection": selection,
        "selected_full_fit": {
            "safety_variant": selected,
            "critical_tail": critical_tail,
            "tail_rank_is_empirical_max": tail_rank_is_empirical_max,
            "bundle": full_bundle,
        },
        "historical_fixed_holdout_diagnostics": {
            "used_for_selection": False,
            "already_consumed_for_diagnostic_interpretation": True,
            "not_valid_for_future_formal_acceptance": True,
            "metrics": holdout_metrics,
        },
        "selected_scope_pooled_diagnostic": {
            "definition": (
                "selected-variant current-19-source critical-LoRA outer-fold rows "
                "plus selected-variant fixed historical critical-LoRA holdout rows"
            ),
            "metrics": pooled_selected,
        },
        "release_gate": {
            "stage_two_calibration_required": True,
            "stage_two_triggers": stage_two_triggers,
            "stage_two_plan": {
                "new_independent_sources": 20,
                "jobs_per_source": 3,
                "jobs": 60,
                "purpose": (
                    "increase critical-tail source count so the 95% finite-sample "
                    "rank is no longer the empirical maximum"
                ),
            },
            "new_post_freeze_prospective_holdout_required": True,
            "production_replacement_allowed": False,
        },
    }
    report["report_sha256"] = sha256_json(report)
    report_path = args.output_dir / "safety_upper_ablation_report.json"
    write_json(report_path, report)

    candidate: dict[str, Any] = {
        "schema": CANDIDATE_SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": report["generated_at_utc"],
        "status": "shadow_candidate_requires_stage_two_calibration",
        "publishable": False,
        "production_override_allowed": False,
        "scope": {
            "gpu_family": "H800",
            "training_mode": "lora",
            "zero_stage": 2,
            "gradient_checkpointing": False,
            "gpu_count": 2,
            "packing": False,
            "outside_scope_policy": "keep_current_frozen_model",
        },
        "center_variant": VARIANT_M1,
        "selected_safety_variant": selected,
        "source_count": len({_source_id(row) for row in training}),
        "stage_two_calibration_required": True,
        "stage_two_triggers": stage_two_triggers,
        "source_recalibration_report_sha256": report["report_sha256"],
        "model": full_bundle,
    }
    candidate["candidate_sha256"] = sha256_json(candidate)
    candidate_path = args.output_dir / "candidate_model_safety_upper_v2.json"
    write_json(candidate_path, candidate)

    manifest = {
        "schema": "sft_h800_m1_safety_upper_manifest/v2",
        "generated_at_utc": report["generated_at_utc"],
        "files": {
            path.name: _input_binding(path)
            for path in (
                training_details_path,
                holdout_details_path,
                report_path,
                candidate_path,
            )
        },
        "selected_safety_variant": selected,
        "candidate_sha256": candidate["candidate_sha256"],
        "publishable": False,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(args.output_dir / "output_manifest.json", manifest)
    print(
        f"selected={selected} current_critical_recall="
        f"{selected_current_metrics.get('admission_recall')} "
        f"stage_two_required=True output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
