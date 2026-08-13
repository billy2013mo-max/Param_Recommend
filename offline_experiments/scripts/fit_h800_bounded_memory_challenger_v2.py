#!/usr/bin/env python3
"""Fit the post-holdout bounded H800 memory challenger v2 on calibration only.

The consumed v1 final holdout is bound as architecture-diagnosis provenance,
but none of its measurements enters the numerical fit or model selection.
The resulting artifact is therefore a seen-failure repair candidate, not a
publishable model; it requires a newly frozen prospective holdout.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from common import ARTIFACT_DIR, read_json, read_jsonl, sha256_file, sha256_json, write_json
from fit_h800_profile_aware_memory_challenger_v1 import _build_points
from h800_bounded_memory_model import (
    ARTIFACT_SCHEMA,
    IMPLEMENTATION_VERSION,
    bounded_feature_values,
    feature_vector,
    predict_log_correction,
    selector_bucket_key,
    selector_feature_name,
)


GIB = float(1024**3)
ALPHA_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
STRESS_FACTOR_BOUNDS = (0.20, 3.00)
DEFAULT_CALIBRATION = ARTIFACT_DIR / "h800_profile_aware_memory_calibration_observations_v1.jsonl"
DEFAULT_OLD_MEMORY = ARTIFACT_DIR / "h800_challenger_modeling.json"
DEFAULT_V1_CHALLENGER = ARTIFACT_DIR / "h800_profile_aware_memory_challenger_v1.json"
DEFAULT_CONSUMED_ACCEPTANCE = ARTIFACT_DIR / "h800_final_unseen_holdout_acceptance_v1.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_bounded_memory_challenger_v2.json"
DEFAULT_MARKDOWN = ARTIFACT_DIR / "h800_bounded_memory_challenger_v2.md"


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _percentile(values: Sequence[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(value))
    if not ordered:
        return None
    position = min(1.0, max(0.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_summary(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(clean),
        "mean": fmean(clean) if clean else None,
        "median": _percentile(clean, 0.5),
        "p90": _percentile(clean, 0.9),
        "maximum": max(clean) if clean else None,
    }


def _scenario_equal_mean(
    details: Sequence[Mapping[str, Any]], key: str
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in details:
        grouped[str(row["split_unit_id"])].append(float(row[key]))
    return fmean(fmean(values) for values in grouped.values())


def _candidates(selector_names: Sequence[str]) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    selector_names = tuple(str(name) for name in selector_names)
    allocated = (
        {"name": "selector_only", "feature_names": selector_names},
        {
            "name": "selector_plus_lora_maximum_fraction",
            "feature_names": (*selector_names, "lora_maximum_fraction"),
        },
        {
            "name": "selector_plus_lora_p99_fraction",
            "feature_names": (*selector_names, "lora_p99_fraction"),
        },
    )
    reserved = (
        {"name": "selector_only", "feature_names": selector_names},
        {
            "name": "selector_plus_lora_maximum_fraction",
            "feature_names": (*selector_names, "lora_maximum_fraction"),
        },
        {
            "name": "selector_plus_lora_p99_fraction",
            "feature_names": (*selector_names, "lora_p99_fraction"),
        },
        {
            "name": "selector_plus_lora_maximum_and_fragmentation",
            "feature_names": (
                *selector_names,
                "lora_maximum_fraction",
                "lora_fragmentation_pressure",
            ),
        },
        {
            "name": "selector_plus_lora_p99_and_fragmentation",
            "feature_names": (
                *selector_names,
                "lora_p99_fraction",
                "lora_fragmentation_pressure",
            ),
        },
        {
            "name": "selector_plus_lora_maximum_and_risk_fragmentation",
            "feature_names": (
                *selector_names,
                "lora_maximum_fraction",
                "risk_fragmentation_pressure",
            ),
        },
        {
            "name": "selector_plus_lora_maximum_and_hierarchical_fragmentation",
            "feature_names": (
                *selector_names,
                "lora_maximum_fraction",
                "lora_fragmentation_pressure",
                "risk_fragmentation_pressure",
                "risk_fragmentation_pressure_x_log2_mbs",
            ),
        },
    )
    return allocated, reserved


def _values(
    point: Mapping[str, Any], supported_selector_keys: Sequence[str]
) -> dict[str, float]:
    return bounded_feature_values(
        point["record"],
        point["padding"],
        supported_selector_keys=supported_selector_keys,
    )


def _fit_ridge(
    points: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    alpha: float,
    target: str,
    supported_selector_keys: Sequence[str],
) -> dict[str, Any]:
    names = tuple(str(name) for name in candidate["feature_names"])
    rows = [_values(point, supported_selector_keys) for point in points]
    features = np.vstack([feature_vector(row, names) for row in rows])
    targets = np.asarray([float(point[target]) for point in points], dtype=float)
    counts = Counter(str(point["split_unit_id"]) for point in points)
    weights = np.asarray(
        [1.0 / counts[str(point["split_unit_id"])] for point in points],
        dtype=float,
    )
    means = np.average(features, axis=0, weights=weights)
    scales = np.sqrt(np.average((features - means) ** 2, axis=0, weights=weights))
    scales[scales < 1e-9] = 1.0
    standardized = (features - means) / scales
    design = np.column_stack((np.ones(len(points)), standardized))
    penalty = np.diag([0.0, *([float(alpha)] * len(names))])
    coefficients = np.linalg.pinv(
        design.T @ (weights[:, None] * design) + penalty
    ) @ (design.T @ (weights * targets))
    return {
        "available": True,
        "model_family": "scenario_weighted_bounded_log_residual_ridge",
        "candidate_name": str(candidate["name"]),
        "target": target,
        "alpha": float(alpha),
        "feature_names": list(names),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "fit_unique_configuration_rows": len(points),
        "fit_split_units": len(counts),
        "scenario_weighting": "equal total weight per split_unit_id",
    }


def _stress_test(
    model: Mapping[str, Any],
    supported_selector_keys: Sequence[str],
) -> dict[str, Any]:
    rows = []
    lower, upper = STRESS_FACTOR_BOUNDS
    for bucket in supported_selector_keys:
        material = json.loads(bucket)
        is_lora = material[0] == "lora"
        is_risk = bool(is_lora and material[2] == 2 and not material[3])
        log2_mbs = math.log2(float(material[5]))
        logs = []
        shape_grid = (0.0, 0.25, 0.5, 0.75, 1.0) if is_lora else (0.0,)
        for maximum in shape_grid:
            for p99 in shape_grid:
                for fragmentation in shape_grid:
                    values = {
                        "lora_maximum_fraction": maximum if is_lora else 0.0,
                        "lora_p99_fraction": p99 if is_lora else 0.0,
                        "lora_fragmentation_pressure": (
                            fragmentation if is_lora else 0.0
                        ),
                        "risk_fragmentation_pressure": (
                            fragmentation if is_risk else 0.0
                        ),
                        "risk_fragmentation_pressure_x_log2_mbs": (
                            fragmentation * log2_mbs if is_risk else 0.0
                        ),
                    }
                    for supported in supported_selector_keys:
                        values[selector_feature_name(supported)] = float(
                            bucket == supported
                        )
                    logs.append(predict_log_correction(values, model))
        rows.append(
            {
                "selector_bucket": bucket,
                "minimum_log_correction": min(logs),
                "maximum_log_correction": max(logs),
                "minimum_factor": math.exp(min(logs)),
                "maximum_factor": math.exp(max(logs)),
            }
        )
    observed_min = min(row["minimum_factor"] for row in rows)
    observed_max = max(row["maximum_factor"] for row in rows)
    return {
        "grid": "all bounded shape inputs in {0,.25,.5,.75,1}",
        "required_factor_bounds": {"minimum": lower, "maximum": upper},
        "observed_factor_bounds": {
            "minimum": observed_min,
            "maximum": observed_max,
        },
        "passed": observed_min >= lower and observed_max <= upper,
        "details": rows,
    }


def _cross_validate(
    points: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    alpha: float,
    target: str,
    supported_selector_keys: Sequence[str],
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for held_out in sorted({str(point["split_unit_id"]) for point in points}):
        train = [point for point in points if point["split_unit_id"] != held_out]
        test = [point for point in points if point["split_unit_id"] == held_out]
        model = _fit_ridge(
            train, candidate, alpha, target, supported_selector_keys
        )
        for point in test:
            log_correction = predict_log_correction(
                _values(point, supported_selector_keys), model
            )
            predicted = float(point["allocated_anchor_bytes"]) * math.exp(
                log_correction
            )
            observed = float(
                point[
                    "observed_allocated_bytes"
                    if target == "allocated_log_correction"
                    else "observed_reserved_bytes"
                ]
            )
            details.append(
                {
                    "split_unit_id": held_out,
                    "cluster_id": point["cluster_id"],
                    "selector_bucket": selector_bucket_key(point["record"]),
                    "allocated_anchor_bytes": float(point["allocated_anchor_bytes"]),
                    "predicted_bytes": predicted,
                    "observed_bytes": observed,
                    "absolute_percentage_error": abs(predicted / observed - 1.0),
                    "signed_percentage_error": predicted / observed - 1.0,
                    "log_underprediction_residual": math.log(observed / predicted),
                }
            )
    return {
        "scenario_equal_mape": _scenario_equal_mean(
            details, "absolute_percentage_error"
        ),
        "row_absolute_percentage_error": _metric_summary(
            [row["absolute_percentage_error"] for row in details]
        ),
        "maximum_log_underprediction_residual": max(
            row["log_underprediction_residual"] for row in details
        ),
        "details": details,
    }


def _select(
    points: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    target: str,
    supported_selector_keys: Sequence[str],
) -> dict[str, Any]:
    trials = []
    for candidate in candidates:
        for alpha in ALPHA_GRID:
            fitted = _fit_ridge(
                points, candidate, alpha, target, supported_selector_keys
            )
            stress = _stress_test(fitted, supported_selector_keys)
            evaluation = _cross_validate(
                points, candidate, alpha, target, supported_selector_keys
            )
            trials.append(
                {
                    "candidate": dict(candidate),
                    "alpha": alpha,
                    "scenario_equal_mape": evaluation["scenario_equal_mape"],
                    "row_absolute_percentage_error": evaluation[
                        "row_absolute_percentage_error"
                    ],
                    "stress_test": stress,
                }
            )
    eligible = [row for row in trials if row["stress_test"]["passed"]]
    if not eligible:
        raise ValueError(f"no stress-safe candidate remains for {target}")
    selected = min(
        eligible,
        key=lambda row: (
            float(row["scenario_equal_mape"]),
            len(row["candidate"]["feature_names"]),
            -float(row["alpha"]),
            str(row["candidate"]["name"]),
        ),
    )
    return {"selected": selected, "trials": trials}


def _nested_cv(
    points: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    target: str,
    supported_selector_keys: Sequence[str],
) -> dict[str, Any]:
    details = []
    selections = []
    for held_out in sorted({str(point["split_unit_id"]) for point in points}):
        train = [point for point in points if point["split_unit_id"] != held_out]
        test = [point for point in points if point["split_unit_id"] == held_out]
        choice = _select(
            train, candidates, target, supported_selector_keys
        )["selected"]
        candidate = choice["candidate"]
        alpha = float(choice["alpha"])
        model = _fit_ridge(
            train, candidate, alpha, target, supported_selector_keys
        )
        selections.append(
            {
                "held_out_split_unit_id": held_out,
                "candidate_name": candidate["name"],
                "feature_count": len(candidate["feature_names"]),
                "alpha": alpha,
            }
        )
        for point in test:
            predicted = float(point["allocated_anchor_bytes"]) * math.exp(
                predict_log_correction(
                    _values(point, supported_selector_keys), model
                )
            )
            observed = float(
                point[
                    "observed_allocated_bytes"
                    if target == "allocated_log_correction"
                    else "observed_reserved_bytes"
                ]
            )
            details.append(
                {
                    "split_unit_id": held_out,
                    "cluster_id": point["cluster_id"],
                    "absolute_percentage_error": abs(predicted / observed - 1.0),
                }
            )
    return {
        "scenario_equal_mape": _scenario_equal_mean(
            details, "absolute_percentage_error"
        ),
        "row_absolute_percentage_error": _metric_summary(
            [row["absolute_percentage_error"] for row in details]
        ),
        "outer_fold_selections": selections,
        "details": details,
    }


def _upper_model(
    points: Sequence[Mapping[str, Any]],
    fixed_reserved_cv: Mapping[str, Any],
    old_memory: Mapping[str, Any],
) -> dict[str, Any]:
    residuals: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in fixed_reserved_cv["details"]:
        residuals[str(row["selector_bucket"])].append(row)
    center_guards = {}
    for bucket, rows in sorted(residuals.items()):
        center_guards[bucket] = {
            "log_residual_upper": max(
                0.0, max(float(row["log_underprediction_residual"]) for row in rows)
            ),
            "leave_profile_out_rows": len(rows),
            "independent_split_units": len(
                {str(row["split_unit_id"]) for row in rows}
            ),
            "source": "fixed_family_leave_profile_out_maximum",
        }

    envelopes: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for point in points:
        envelopes[selector_bucket_key(point["record"])].append(point)
    anchor_envelopes = {}
    for bucket, rows in sorted(envelopes.items()):
        anchor_envelopes[bucket] = {
            "log_correction_upper": max(
                0.0,
                max(
                    math.log(
                        float(row["observed_reserved_bytes"])
                        / float(row["allocated_anchor_bytes"])
                    )
                    for row in rows
                ),
            ),
            "unique_configuration_rows": len(rows),
            "independent_split_units": len(
                {str(row["split_unit_id"]) for row in rows}
            ),
            "source": "calibration_selector_empirical_anchor_envelope",
        }
    legacy = old_memory["memory"]["frozen_model"]["tail"].get(
        "oom_log_residual_lower_by_exact_selector"
    ) or {}
    return {
        "available": True,
        "center_residual_guard_by_selector": center_guards,
        "anchor_envelope_by_selector": anchor_envelopes,
        "legacy_exact_selector_oom_guard": legacy,
        "formula": (
            "max(direct_center * exp(selector_LOPO_residual_guard), "
            "physical_anchor * exp(selector_anchor_envelope), "
            "physical_anchor * exp(legacy_exact_selector_OOM_guard))"
        ),
        "semantics": (
            "conservative empirical operational upper; not an identified P95"
        ),
    }


def _guarded_cv(
    fixed_reserved_cv: Mapping[str, Any], upper_model: Mapping[str, Any]
) -> dict[str, Any]:
    details = []
    for row in fixed_reserved_cv["details"]:
        bucket = str(row["selector_bucket"])
        residual = float(
            upper_model["center_residual_guard_by_selector"][bucket][
                "log_residual_upper"
            ]
        )
        envelope = float(
            upper_model["anchor_envelope_by_selector"][bucket][
                "log_correction_upper"
            ]
        )
        upper = max(
            float(row["predicted_bytes"]) * math.exp(residual),
            float(row["allocated_anchor_bytes"]) * math.exp(envelope),
        )
        details.append(
            {
                **dict(row),
                "operational_upper_bytes": upper,
                "covered": float(row["observed_bytes"]) <= upper,
            }
        )
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in details:
        grouped[str(row["split_unit_id"])].append(bool(row["covered"]))
    rates = [sum(values) / len(values) for values in grouped.values()]
    return {
        "row_coverage": sum(row["covered"] for row in details) / len(details),
        "scenario_equal_mean_coverage": fmean(rates),
        "scenario_equal_p05_coverage": _percentile(rates, 0.05),
        "details": details,
    }


def fit(
    *,
    calibration_path: Path,
    old_memory_path: Path,
    v1_challenger_path: Path,
    consumed_acceptance_path: Path,
) -> dict[str, Any]:
    observations = read_jsonl(calibration_path)
    if len(observations) != 28:
        raise ValueError("expected the immutable 28-row calibration snapshot")
    if any(
        row["fingerprint"]["calibration_evidence_eligible"] is not True
        or row["configuration"]["calibration_partition"]["role"] != "calibration"
        for row in observations
    ):
        raise ValueError("a non-calibration observation entered the v2 fit")
    old_memory = read_json(old_memory_path)
    _raw, points = _build_points(observations, old_memory)
    if len(points) != 18:
        raise ValueError("expected 18 repeat-collapsed calibration configurations")
    split_units = sorted({str(point["split_unit_id"]) for point in points})
    if len(split_units) != 4:
        raise ValueError("expected exactly four independent calibration profiles")
    for point in points:
        point["direct_reserved_log_correction"] = math.log(
            float(point["observed_reserved_bytes"])
            / float(point["allocated_anchor_bytes"])
        )

    supported_keys = sorted(
        {selector_bucket_key(point["record"]) for point in points}
    )
    if len(supported_keys) != 5:
        raise ValueError("expected five calibrated mechanism buckets")
    selector_names = [selector_feature_name(key) for key in supported_keys]
    allocated_candidates, reserved_candidates = _candidates(selector_names)
    allocated_selection = _select(
        points,
        allocated_candidates,
        "allocated_log_correction",
        supported_keys,
    )
    reserved_selection = _select(
        points,
        reserved_candidates,
        "direct_reserved_log_correction",
        supported_keys,
    )
    allocated_choice = allocated_selection["selected"]
    reserved_choice = reserved_selection["selected"]
    allocated_model = _fit_ridge(
        points,
        allocated_choice["candidate"],
        float(allocated_choice["alpha"]),
        "allocated_log_correction",
        supported_keys,
    )
    reserved_model = _fit_ridge(
        points,
        reserved_choice["candidate"],
        float(reserved_choice["alpha"]),
        "direct_reserved_log_correction",
        supported_keys,
    )
    fixed_allocated_cv = _cross_validate(
        points,
        allocated_choice["candidate"],
        float(allocated_choice["alpha"]),
        "allocated_log_correction",
        supported_keys,
    )
    fixed_reserved_cv = _cross_validate(
        points,
        reserved_choice["candidate"],
        float(reserved_choice["alpha"]),
        "direct_reserved_log_correction",
        supported_keys,
    )
    nested_allocated = _nested_cv(
        points, allocated_candidates, "allocated_log_correction", supported_keys
    )
    nested_reserved = _nested_cv(
        points,
        reserved_candidates,
        "direct_reserved_log_correction",
        supported_keys,
    )
    upper_model = _upper_model(points, fixed_reserved_cv, old_memory)
    guarded = _guarded_cv(fixed_reserved_cv, upper_model)
    v1 = read_json(v1_challenger_path)
    consumed = read_json(consumed_acceptance_path)

    report: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_post_holdout_repair_candidate_requires_new_prospective_holdout",
        "gpu_family": "H800",
        "publishable": False,
        "production_override_allowed": False,
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "source_bindings": {
            "calibration_snapshot": _binding(calibration_path),
            "physical_anchor_artifact": _binding(old_memory_path),
            "invalidated_v1_challenger": _binding(v1_challenger_path),
            "consumed_v1_acceptance_diagnosis": _binding(consumed_acceptance_path),
        },
        "governance": {
            "consumed_holdout_measurement_rows_used_in_numerical_fit": 0,
            "consumed_holdout_used_to_diagnose_model_family": True,
            "candidate_is_unseen_to_consumed_holdout": False,
            "new_prospective_holdout_required": True,
            "old_artifact_overwritten": False,
            "automatic_execution_allowed": False,
        },
        "calibration_admission": {
            "raw_observation_rows": len(observations),
            "unique_configuration_rows": len(points),
            "repeat_rows_collapsed": len(observations) - len(points),
            "independent_split_units": len(split_units),
            "split_unit_ids": split_units,
            "supported_selector_buckets": len(supported_keys),
            "training_observation_ids_sha256": sha256_json(
                sorted(row["observation_id"] for row in observations)
            ),
        },
        "diagnosed_v1_failure": {
            "v1_report_sha256": v1["report_sha256"],
            "consumed_acceptance_report_sha256": consumed["report_sha256"],
            "root_cause": (
                "risk-by-padding-pressure squared/cubed reservation terms "
                "extrapolated beyond the calibration pressure range"
            ),
            "repair_constraints": [
                "one direct reserved-center head over the physical anchor",
                "all dataset-shape features bounded to [0,1]",
                "no polynomial pressure powers or cross-head multiplication",
                "unknown execution-mechanism buckets fail closed",
                "candidate stress grid must keep correction factors in [0.20,3.00]",
            ],
        },
        "selection": {
            "protocol": (
                "calibration-only scenario-equal leave-profile-out MAPE after "
                "bounded stress gate; feature count then larger alpha are tie breakers"
            ),
            "alpha_grid": list(ALPHA_GRID),
            "stress_factor_bounds": list(STRESS_FACTOR_BOUNDS),
            "allocated_diagnostic": allocated_selection,
            "direct_reserved_center": reserved_selection,
        },
        "model": {
            "formula": (
                "reserved_center = physical_allocated_anchor * "
                "exp(direct_bounded_log_correction)"
            ),
            "allocated_diagnostic_formula": (
                "allocated_center = physical_allocated_anchor * "
                "exp(bounded_allocated_log_correction)"
            ),
            "supported_selector_keys": supported_keys,
            "selector_key_fields": [
                "training_mode",
                "gpu_count",
                "zero_stage",
                "gradient_checkpointing",
                "packing",
                "physical_mbs",
            ],
            "shape_features": {
                "lora_maximum_fraction": (
                    "I(lora) * maximum_clipped_tokens / cutoff_len"
                ),
                "lora_fragmentation_pressure": (
                    "I(lora) * E[random_batch_max]/cutoff_len * "
                    "(1 - truncation_fraction)"
                ),
            },
            "allocated_anchor": {
                "source_artifact": str(old_memory_path.resolve()),
                "source_artifact_sha256": sha256_file(old_memory_path),
                "source_report_sha256": old_memory["report_sha256"],
                "head": "memory.frozen_model.allocated_center_diagnostic",
            },
            "allocated_diagnostic": allocated_model,
            "direct_reserved_center": reserved_model,
            "operational_upper": upper_model,
        },
        "evaluation": {
            "unit": "repeat-collapsed unique configuration",
            "fixed_selected_allocated_leave_profile_out": fixed_allocated_cv,
            "nested_allocated_model_selection_leave_profile_out": nested_allocated,
            "fixed_selected_reserved_leave_profile_out": fixed_reserved_cv,
            "nested_reserved_model_selection_leave_profile_out": nested_reserved,
            "guarded_fixed_reserved_leave_profile_out": guarded,
        },
        "release_gate": {
            "new_holdout_campaign_id": None,
            "new_holdout_profiles_selected": False,
            "predictions_must_be_frozen_before_new_gpu_runs": True,
            "promotion_from_consumed_v1_holdout_forbidden": True,
            "automatic_execution_allowed": False,
        },
        "publication_blockers": [
            "the v2 architecture was designed after reading the v1 final holdout",
            "only four independent calibration profiles are available",
            "a new profile-disjoint prospective holdout has not been frozen or run",
            "packing=true and unseen selector buckets remain outside this candidate",
        ],
    }
    report["report_sha256"] = sha256_json(report)
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    fixed = report["evaluation"]["fixed_selected_reserved_leave_profile_out"]
    nested = report["evaluation"]["nested_reserved_model_selection_leave_profile_out"]
    allocated = report["model"]["allocated_diagnostic"]
    reserved = report["model"]["direct_reserved_center"]
    guarded = report["evaluation"]["guarded_fixed_reserved_leave_profile_out"]
    return "\n".join(
        [
            "# H800 有界显存 challenger v2",
            "",
            "> 状态：已冻结的 post-holdout 修复候选，不可发布；必须使用全新的 prospective holdout 验收。",
            "",
            "## 模型",
            "",
            "`reserved_center = physical_allocated_anchor × exp(direct_bounded_log_correction)`",
            "",
            f"- reserved 头：`{reserved['candidate_name']}`，{len(reserved['feature_names'])} 个变量，alpha={reserved['alpha']}。",
            f"- allocated 诊断头：`{allocated['candidate_name']}`，{len(allocated['feature_names'])} 个变量，alpha={allocated['alpha']}。",
            "- 画像只使用 [0,1] 内的最大长度占比与非截断形状压力；未知执行机制桶直接拒绝。",
            "- upper 取中心残差上界、物理锚点历史包络和旧 OOM guard 三者最大值。",
            "",
            "## 仅校准集结果",
            "",
            f"- 固定模型族 LOPO：scenario-equal center MAPE={100 * fixed['scenario_equal_mape']:.2f}%，row P90 APE={100 * fixed['row_absolute_percentage_error']['p90']:.2f}%。",
            f"- 含模型选择不确定性的 nested LOPO：MAPE={100 * nested['scenario_equal_mape']:.2f}%，row P90 APE={100 * nested['row_absolute_percentage_error']['p90']:.2f}%。",
            f"- operational upper 校准内覆盖率={100 * guarded['row_coverage']:.1f}%。",
            "",
            "## 治理结论",
            "",
            "v1 最终 holdout 只用于定位失真机制，没有进入数值拟合；但因为模型族是在揭盲后设计的，旧 holdout 不能再次证明泛化性。下一步必须先冻结新的数据画像与预测，再运行新实验。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--old-memory", type=Path, default=DEFAULT_OLD_MEMORY)
    parser.add_argument("--v1-challenger", type=Path, default=DEFAULT_V1_CHALLENGER)
    parser.add_argument(
        "--consumed-acceptance", type=Path, default=DEFAULT_CONSUMED_ACCEPTANCE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = fit(
        calibration_path=args.calibration,
        old_memory_path=args.old_memory,
        v1_challenger_path=args.v1_challenger,
        consumed_acceptance_path=args.consumed_acceptance,
    )
    write_json(args.output, report)
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(args.output.resolve()),
                "markdown": str(args.markdown.resolve()),
                "report_sha256": report["report_sha256"],
                "reserved_candidate": report["model"]["direct_reserved_center"][
                    "candidate_name"
                ],
                "fixed_lopo_mape": report["evaluation"][
                    "fixed_selected_reserved_leave_profile_out"
                ]["scenario_equal_mape"],
                "nested_lopo_mape": report["evaluation"][
                    "nested_reserved_model_selection_leave_profile_out"
                ]["scenario_equal_mape"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
