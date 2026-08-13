#!/usr/bin/env python3
"""Fit the H800 profile-aware memory challenger from 28 calibration jobs.

Only the two explicitly bound calibration queues are admitted.  Repeated runs
are retained in the evidence snapshot but collapsed before regression so they
do not masquerade as independent profile evidence.  Candidate selection uses
leave-profile-out cross-validation; a separate nested result exposes model
selection uncertainty.  The frozen unseen-profile selection is checked for
source overlap but is never read as training data.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np

from common import (
    ARTIFACT_DIR,
    MATRIX_DIR,
    RESULTS_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from export_h800_observations import (
    _load_json,
    _one_observation,
    _provenance_index,
    validate_canonical_observation,
    write_jsonl,
)
from h800_challenger_modeling import _predict_memory_center
from h800_native_memory_calibration import (
    _inventory_models,
    build_native_record,
    native_admission_reason,
)
from h800_profile_aware_memory_model import (
    ARTIFACT_SCHEMA,
    IMPLEMENTATION_VERSION,
    base_feature_values,
    predict_log_correction,
    selector_mbs_key,
    vector,
)


GIB = float(1024**3)
ALPHA_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_PROFILE_QUEUE = (
    MATRIX_DIR / "h800_profile_aware_memory_calibration_jobs_v1.jsonl"
)
DEFAULT_14B_QUEUE = (
    MATRIX_DIR / "h800_14b_full_boundary_calibration_jobs_v1.jsonl"
)
DEFAULT_PROFILE_DESIGN = (
    ARTIFACT_DIR / "h800_profile_aware_memory_calibration_design_v1.json"
)
DEFAULT_14B_DESIGN = (
    ARTIFACT_DIR / "h800_14b_full_boundary_calibration_design_v1.json"
)
DEFAULT_HOLDOUT_SELECTION = (
    ARTIFACT_DIR / "h800_final_unseen_holdout_selection_v1.json"
)
DEFAULT_OLD_MEMORY = ARTIFACT_DIR / "h800_challenger_modeling.json"
DEFAULT_SNAPSHOT = (
    ARTIFACT_DIR / "h800_profile_aware_memory_calibration_observations_v1.jsonl"
)
DEFAULT_AMENDMENT = (
    ARTIFACT_DIR / "h800_profile_aware_memory_model_amendment_v1.json"
)
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_profile_aware_memory_challenger_v1.json"
DEFAULT_MARKDOWN = ARTIFACT_DIR / "h800_profile_aware_memory_challenger_v1.md"


ALLOCATION_CANDIDATES = (
    {"name": "no_profile", "feature_names": (), "saturation_rate": 16.0},
    *(
        {
            "name": f"profile_saturation_k{rate}",
            "feature_names": (
                "is_lora",
                "activation_share",
                "lora_x_allocation_saturation",
                "activation_share_x_allocation_saturation",
                "log2_mbs",
                "log2_gpu_count",
            ),
            "saturation_rate": float(rate),
        }
        for rate in (4, 8, 12, 16)
    ),
)
RESERVATION_CANDIDATES = (
    {"name": "no_profile", "feature_names": (), "saturation_rate": 8.0},
    {
        "name": "selector_only",
        "feature_names": (
            "lora_zero2_gc_off",
            "log2_mbs",
            "log2_gpu_count",
        ),
        "saturation_rate": 8.0,
    },
    {
        "name": "padding_pressure_only",
        "feature_names": (
            "padding_pressure",
            "p99_fraction_of_cutoff",
            "coefficient_of_variation",
            "truncation_fraction",
        ),
        "saturation_rate": 8.0,
    },
    {
        "name": "bounded_selector_profile_interactions",
        "feature_names": (
            "reservation_saturation",
            "lora_zero2_gc_off",
            "log2_mbs",
            "log2_gpu_count",
            "risk_x_padding_pressure_squared",
            "risk_x_padding_pressure_cubed",
            "risk_x_padding_pressure_cubed_x_log2_mbs",
        ),
        "saturation_rate": 8.0,
    },
)
ORIGINAL_RATIO_CANDIDATES = (
    {"name": "no_profile", "feature_names": (), "saturation_rate": 8.0},
    RESERVATION_CANDIDATES[2],
    {
        "name": "original_bounded_seven_terms",
        "feature_names": (
            "padding_pressure",
            "p99_fraction_of_cutoff",
            "coefficient_of_variation",
            "truncation_fraction",
            "lora_zero2_gc_off",
            "log2_mbs",
            "log2_gpu_count",
        ),
        "saturation_rate": 8.0,
    },
)


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
    usable = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(usable),
        "mean": fmean(usable) if usable else None,
        "median": median(usable) if usable else None,
        "p90": _percentile(usable, 0.90),
        "maximum": max(usable) if usable else None,
    }


def _scenario_equal_mean(details: Sequence[Mapping[str, Any]], key: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in details:
        grouped[str(row["split_unit_id"])].append(float(row[key]))
    return fmean(fmean(values) for values in grouped.values())


def _export_exact_jobs(queue_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    hardware = _load_json(ROOT / "config" / "hardware.json")
    experiment = _load_json(ROOT / "config" / "experiment.json")
    provenance = _provenance_index(ROOT)
    observations: list[dict[str, Any]] = []
    for job in sorted(queue_rows, key=lambda row: str(row["job_id"])):
        job_id = str(job["job_id"])
        latest_path = RESULTS_DIR / job_id / "latest_attempt.json"
        latest = _load_json(latest_path)
        if (
            latest.get("job_id") != job_id
            or latest.get("state") != "complete"
            or latest.get("calibration_eligible") is not True
        ):
            raise ValueError(f"latest attempt is not calibration-eligible: {job_id}")
        attempt = RESULTS_DIR / job_id / str(latest["attempt_path"])
        row = _one_observation(
            attempt,
            ROOT,
            hardware,
            experiment,
            provenance,
            job_id_hint=job_id,
            attempt_scoped=True,
        )
        if row is None:
            raise ValueError(f"exporter rejected latest attempt: {job_id}")
        validation = validate_canonical_observation(row)
        if validation:
            raise ValueError(f"canonical validation failed for {job_id}: {validation}")
        if native_admission_reason(row) != "admitted":
            raise ValueError(
                f"native memory admission rejected {job_id}: "
                f"{native_admission_reason(row)}"
            )
        observations.append(row)
    return observations


def _fit_ridge(
    points: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    alpha: float,
    target: str,
) -> dict[str, Any]:
    names = tuple(str(name) for name in candidate["feature_names"])
    values = [
        base_feature_values(
            point["record"],
            point["padding"],
            allocation_saturation_rate=float(candidate["saturation_rate"]),
            reservation_saturation_rate=float(candidate["saturation_rate"]),
        )
        for point in points
    ]
    features = (
        np.vstack([vector(row, names) for row in values])
        if names
        else np.empty((len(points), 0), dtype=float)
    )
    targets = np.asarray([float(point[target]) for point in points], dtype=float)
    counts = Counter(str(point["split_unit_id"]) for point in points)
    weights = np.asarray(
        [1.0 / counts[str(point["split_unit_id"])] for point in points],
        dtype=float,
    )
    means = (
        np.average(features, axis=0, weights=weights)
        if names
        else np.asarray([], dtype=float)
    )
    scales = (
        np.sqrt(np.average((features - means) ** 2, axis=0, weights=weights))
        if names
        else np.asarray([], dtype=float)
    )
    scales[scales < 1e-9] = 1.0
    standardized = (features - means) / scales if names else features
    design = np.column_stack((np.ones(len(points)), standardized))
    penalty = np.diag([0.0, *([float(alpha)] * len(names))])
    coefficients = np.linalg.pinv(
        design.T @ (weights[:, None] * design) + penalty
    ) @ (design.T @ (weights * targets))
    return {
        "available": True,
        "model_family": "scenario_weighted_log_residual_ridge",
        "candidate_name": candidate["name"],
        "target": target,
        "alpha": float(alpha),
        "feature_names": list(names),
        "feature_parameters": {
            "saturation_rate": float(candidate["saturation_rate"]),
            "saturation_formula": "1 - exp(-rate * padding_pressure)",
        },
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "fit_unique_configuration_rows": len(points),
        "fit_split_units": len(counts),
        "scenario_weighting": "equal total weight per split_unit_id",
    }


def _values_for(point: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    return base_feature_values(
        point["record"],
        point["padding"],
        allocation_saturation_rate=float(candidate["saturation_rate"]),
        reservation_saturation_rate=float(candidate["saturation_rate"]),
    )


def _head_cv(
    points: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    alpha: float,
    target: str,
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    groups = sorted({str(point["split_unit_id"]) for point in points})
    for held_out in groups:
        train = [point for point in points if point["split_unit_id"] != held_out]
        test = [point for point in points if point["split_unit_id"] == held_out]
        model = _fit_ridge(train, candidate, alpha, target)
        for point in test:
            correction = math.exp(
                predict_log_correction(_values_for(point, candidate), model)
            )
            if target == "allocated_log_correction":
                predicted = float(point["allocated_anchor_bytes"]) * correction
                observed = float(point["observed_allocated_bytes"])
            elif target == "reservation_log_ratio":
                predicted = float(point["observed_allocated_bytes"]) * correction
                observed = float(point["observed_reserved_bytes"])
            else:
                raise ValueError(f"unknown target: {target}")
            details.append(
                {
                    "split_unit_id": held_out,
                    "cluster_id": point["cluster_id"],
                    "predicted_bytes": predicted,
                    "observed_bytes": observed,
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
        "details": details,
    }


def _select(
    points: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    target: str,
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    for candidate in candidates:
        for alpha in ALPHA_GRID:
            evaluation = _head_cv(points, candidate, alpha, target)
            trials.append(
                {
                    "candidate": dict(candidate),
                    "alpha": alpha,
                    "scenario_equal_mape": evaluation["scenario_equal_mape"],
                    "row_absolute_percentage_error": evaluation[
                        "row_absolute_percentage_error"
                    ],
                }
            )
    selected = min(
        trials,
        key=lambda row: (
            float(row["scenario_equal_mape"]),
            len(row["candidate"]["feature_names"]),
            -float(row["alpha"]),
            str(row["candidate"]["name"]),
        ),
    )
    return {"selected": selected, "trials": trials}


def _combined_cv(
    points: Sequence[Mapping[str, Any]],
    allocation_candidate: Mapping[str, Any],
    allocation_alpha: float,
    reservation_candidate: Mapping[str, Any],
    reservation_alpha: float,
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for held_out in sorted({str(point["split_unit_id"]) for point in points}):
        train = [point for point in points if point["split_unit_id"] != held_out]
        test = [point for point in points if point["split_unit_id"] == held_out]
        allocation = _fit_ridge(
            train,
            allocation_candidate,
            allocation_alpha,
            "allocated_log_correction",
        )
        reservation = _fit_ridge(
            train,
            reservation_candidate,
            reservation_alpha,
            "reservation_log_ratio",
        )
        for point in test:
            allocation_factor = math.exp(
                predict_log_correction(
                    _values_for(point, allocation_candidate), allocation
                )
            )
            reservation_factor = math.exp(
                predict_log_correction(
                    _values_for(point, reservation_candidate), reservation
                )
            )
            center = (
                float(point["allocated_anchor_bytes"])
                * allocation_factor
                * reservation_factor
            )
            observed = float(point["observed_reserved_bytes"])
            details.append(
                {
                    "split_unit_id": held_out,
                    "cluster_id": point["cluster_id"],
                    "job_ids": point["job_ids"],
                    "selector_mbs_key": selector_mbs_key(point["record"]),
                    "predicted_reserved_center_bytes": center,
                    "observed_reserved_bytes": observed,
                    "absolute_percentage_error": abs(center / observed - 1.0),
                    "signed_percentage_error": center / observed - 1.0,
                    "log_underprediction_residual": math.log(observed / center),
                }
            )
    errors = [row["absolute_percentage_error"] for row in details]
    return {
        "scenario_equal_mape": _scenario_equal_mean(
            details, "absolute_percentage_error"
        ),
        "row_absolute_percentage_error": _metric_summary(errors),
        "row_signed_percentage_error": _metric_summary(
            [row["signed_percentage_error"] for row in details]
        ),
        "details": details,
    }


def _nested_combined_cv(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for held_out in sorted({str(point["split_unit_id"]) for point in points}):
        train = [point for point in points if point["split_unit_id"] != held_out]
        test = [point for point in points if point["split_unit_id"] == held_out]
        allocation_selection = _select(
            train, ALLOCATION_CANDIDATES, "allocated_log_correction"
        )["selected"]
        reservation_selection = _select(
            train, RESERVATION_CANDIDATES, "reservation_log_ratio"
        )["selected"]
        allocation_candidate = allocation_selection["candidate"]
        reservation_candidate = reservation_selection["candidate"]
        allocation = _fit_ridge(
            train,
            allocation_candidate,
            float(allocation_selection["alpha"]),
            "allocated_log_correction",
        )
        reservation = _fit_ridge(
            train,
            reservation_candidate,
            float(reservation_selection["alpha"]),
            "reservation_log_ratio",
        )
        selections.append(
            {
                "held_out_split_unit_id": held_out,
                "allocated_candidate": allocation_candidate["name"],
                "allocated_alpha": allocation_selection["alpha"],
                "reservation_candidate": reservation_candidate["name"],
                "reservation_alpha": reservation_selection["alpha"],
            }
        )
        for point in test:
            center = float(point["allocated_anchor_bytes"])
            center *= math.exp(
                predict_log_correction(
                    _values_for(point, allocation_candidate), allocation
                )
            )
            center *= math.exp(
                predict_log_correction(
                    _values_for(point, reservation_candidate), reservation
                )
            )
            observed = float(point["observed_reserved_bytes"])
            details.append(
                {
                    "split_unit_id": held_out,
                    "cluster_id": point["cluster_id"],
                    "absolute_percentage_error": abs(center / observed - 1.0),
                    "log_underprediction_residual": math.log(observed / center),
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
        "outer_fold_selections": selections,
        "details": details,
    }


def _ratio_only_cv(
    points: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    alpha: float,
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for held_out in sorted({str(point["split_unit_id"]) for point in points}):
        train = [point for point in points if point["split_unit_id"] != held_out]
        test = [point for point in points if point["split_unit_id"] == held_out]
        model = _fit_ridge(
            train, candidate, alpha, "reservation_log_ratio"
        )
        for point in test:
            center = float(point["allocated_anchor_bytes"]) * math.exp(
                predict_log_correction(_values_for(point, candidate), model)
            )
            observed = float(point["observed_reserved_bytes"])
            details.append(
                {
                    "split_unit_id": held_out,
                    "absolute_percentage_error": abs(center / observed - 1.0),
                }
            )
    return {
        "scenario_equal_mape": _scenario_equal_mean(
            details, "absolute_percentage_error"
        ),
        "row_absolute_percentage_error": _metric_summary(
            [row["absolute_percentage_error"] for row in details]
        ),
    }


def _build_points(
    observations: Sequence[dict[str, Any]],
    old_memory: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory = read_json(ARTIFACT_DIR / "model_inventory.json")
    model_by_id, fixed_lora = _inventory_models(inventory)
    hardware = read_json(ROOT / "config" / "hardware.json")
    allocated_model = old_memory["memory"]["frozen_model"][
        "allocated_center_diagnostic"
    ]
    raw_points: list[dict[str, Any]] = []
    for row in observations:
        record = build_native_record(
            row,
            model_by_id=model_by_id,
            fixed_lora=fixed_lora,
            hardware=hardware,
        )
        job = row["configuration"]["job"]
        padding = job.get("expected_padding_pressure")
        if not isinstance(padding, Mapping):
            raise ValueError(f"job has no bound padding profile: {job['job_id']}")
        measurement = row["measurements"]["memory"]
        observed_allocated = float(measurement["max_allocated_bytes"])
        observed_reserved = float(measurement["max_reserved_bytes"])
        allocated_anchor = _predict_memory_center(record, allocated_model)
        cluster_material = {
            "split_unit_id": row["configuration"]["calibration_partition"][
                "split_unit_id"
            ],
            "model_id": job["model_id"],
            "train_type": job["train_type"],
            "gpu_count": job["gpu_count"],
            "zero_stage": job["zero_stage"],
            "gc": job["gc"],
            "mbs": job["mbs"],
            "cutoff_len": job["cutoff_len"],
            "packing": job["packing"],
        }
        raw_points.append(
            {
                "observation_id": row["observation_id"],
                "job_id": job["job_id"],
                "cluster_id": "memcluster-" + sha256_json(cluster_material)[:16],
                "split_unit_id": cluster_material["split_unit_id"],
                "record": record,
                "padding": dict(padding),
                "allocated_anchor_bytes": allocated_anchor,
                "observed_allocated_bytes": observed_allocated,
                "observed_reserved_bytes": observed_reserved,
                "allocated_log_correction": math.log(
                    observed_allocated / allocated_anchor
                ),
                "reservation_log_ratio": math.log(
                    observed_reserved / observed_allocated
                ),
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in raw_points:
        grouped[str(point["cluster_id"])].append(point)
    collapsed: list[dict[str, Any]] = []
    for cluster_id, repeats in sorted(grouped.items()):
        first = dict(repeats[0])
        first["job_ids"] = sorted(str(row["job_id"]) for row in repeats)
        first["observation_ids"] = sorted(
            str(row["observation_id"]) for row in repeats
        )
        first["repeat_count"] = len(repeats)
        for key in (
            "allocated_anchor_bytes",
            "observed_allocated_bytes",
            "observed_reserved_bytes",
            "allocated_log_correction",
            "reservation_log_ratio",
        ):
            first[key] = median(float(row[key]) for row in repeats)
        first.pop("job_id", None)
        first.pop("observation_id", None)
        collapsed.append(first)
    return raw_points, collapsed


def _tail_from_cv(
    details: Sequence[Mapping[str, Any]],
    old_memory: Mapping[str, Any],
) -> dict[str, Any]:
    by_bucket: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in details:
        by_bucket[str(row["selector_mbs_key"])].append(row)
    selector_mbs = {}
    for key, rows in sorted(by_bucket.items()):
        residuals = [float(row["log_underprediction_residual"]) for row in rows]
        selector_mbs[key] = {
            "log_residual_upper": max(0.0, max(residuals)),
            "unique_configuration_rows": len(rows),
            "independent_split_units": len(
                {str(row["split_unit_id"]) for row in rows}
            ),
            "split_unit_ids": sorted(
                {str(row["split_unit_id"]) for row in rows}
            ),
            "source": "fixed_selected_model_leave_profile_out_maximum",
        }
    all_residuals = [
        float(row["log_underprediction_residual"]) for row in details
    ]
    legacy = old_memory["memory"]["frozen_model"]["tail"].get(
        "oom_log_residual_lower_by_exact_selector"
    ) or {}
    return {
        "available": True,
        "selector_mbs": selector_mbs,
        "pooled": {
            "log_residual_upper": max(0.0, max(all_residuals)),
            "unique_configuration_rows": len(details),
            "independent_split_units": len(
                {str(row["split_unit_id"]) for row in details}
            ),
            "source": "fixed_selected_model_leave_profile_out_maximum",
        },
        "legacy_exact_selector_oom_guard": legacy,
        "formula": (
            "reserved_center * exp(max(selector_mbs_LOPO_max, "
            "legacy_exact_selector_OOM_guard, 0))"
        ),
        "statistical_claim": (
            "operational empirical guard; only 3-4 independent profiles per "
            "supported bucket, so a distribution-free P95 is not identified"
        ),
    }


def _guarded_evaluation(
    details: Sequence[Mapping[str, Any]], tail: Mapping[str, Any]
) -> dict[str, Any]:
    evaluated = []
    for row in details:
        bucket = tail["selector_mbs"][str(row["selector_mbs_key"])]
        guard = float(bucket["log_residual_upper"])
        upper = float(row["predicted_reserved_center_bytes"]) * math.exp(guard)
        observed = float(row["observed_reserved_bytes"])
        evaluated.append(
            {
                **dict(row),
                "operational_upper_reserved_bytes": upper,
                "covered": observed <= upper,
            }
        )
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in evaluated:
        grouped[str(row["split_unit_id"])].append(bool(row["covered"]))
    rates = [sum(values) / len(values) for values in grouped.values()]
    return {
        "row_coverage": sum(row["covered"] for row in evaluated) / len(evaluated),
        "scenario_equal_mean_coverage": fmean(rates),
        "scenario_equal_p05_coverage": _percentile(rates, 0.05),
        "details": evaluated,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    assumption = report["original_contract_assumption_test"]
    fixed = report["evaluation"]["fixed_selected_family_leave_profile_out"]
    nested = report["evaluation"]["nested_model_selection_leave_profile_out"]
    model = report["model"]
    return "\n".join(
        [
            "# H800 profile-aware 显存 challenger v1",
            "",
            f"- 训练证据：{report['calibration_admission']['raw_observation_rows']} 条完整 calibration attempt；折叠重复后 {report['calibration_admission']['unique_configuration_rows']} 个配置点；{report['calibration_admission']['independent_split_units']} 个独立数据画像。",
            f"- 原假设检验：旧 allocated 头在本批 profile-stratified 数据上的 MAPE 为 {100 * assumption['allocated_anchor']['scenario_equal_mape']:.2f}%，因此原先“allocated 头无需画像校正”的前提不成立。",
            f"- 冻结 allocated 校正头：`{model['allocated_profile_correction']['candidate_name']}`，{len(model['allocated_profile_correction']['feature_names'])} 个特征，alpha={model['allocated_profile_correction']['alpha']}。",
            f"- 冻结 reservation 头：`{model['reservation_ratio']['candidate_name']}`，{len(model['reservation_ratio']['feature_names'])} 个特征，alpha={model['reservation_ratio']['alpha']}。",
            f"- 固定模型族 leave-profile-out：scenario-equal center MAPE={100 * fixed['scenario_equal_mape']:.2f}%，row P90 APE={100 * fixed['row_absolute_percentage_error']['p90']:.2f}%。",
            f"- 含模型选择不确定性的 nested leave-profile-out：scenario-equal center MAPE={100 * nested['scenario_equal_mape']:.2f}%，row P90 APE={100 * nested['row_absolute_percentage_error']['p90']:.2f}%。",
            f"- 校准内 operational upper 覆盖：scenario-equal P05={100 * report['evaluation']['guarded_fixed_cv']['scenario_equal_p05_coverage']:.1f}%。",
            "",
            "该结果只允许进入新的 unseen-profile prospective holdout。它尚未通过发布验收，不能覆盖当前模型，也不能自动启动训练。",
            "",
        ]
    )


def fit(
    *,
    profile_queue_path: Path,
    boundary_queue_path: Path,
    profile_design_path: Path,
    boundary_design_path: Path,
    holdout_selection_path: Path,
    old_memory_path: Path,
    snapshot_path: Path,
    amendment_path: Path,
) -> dict[str, Any]:
    queue_rows = [
        *read_jsonl(profile_queue_path),
        *read_jsonl(boundary_queue_path),
    ]
    if len(queue_rows) != 28 or len({row["job_id"] for row in queue_rows}) != 28:
        raise ValueError("the bound calibration queues must contain 28 unique jobs")
    observations = _export_exact_jobs(queue_rows)
    if len(observations) != 28:
        raise ValueError("the exporter did not return exactly 28 observations")
    if any(
        row["fingerprint"]["quality"] != "complete"
        or row["fingerprint"]["calibration_evidence_eligible"] is not True
        or row["configuration"]["calibration_partition"]["role"] != "calibration"
        for row in observations
    ):
        raise ValueError("a non-calibration or incomplete observation was admitted")

    holdout_selection = read_json(holdout_selection_path)
    selected_source_hashes = {
        str(row["source_sha256"]) for row in holdout_selection["profiles"]
    }
    selected_source_paths = {
        str(Path(row["source_path"]).resolve())
        for row in holdout_selection["profiles"]
    }
    calibration_hashes = {
        str(row["configuration"]["job"]["data_sha256"])
        for row in observations
    }
    calibration_paths = {
        str(Path(row["configuration"]["job"]["data_path"]).resolve())
        for row in observations
    }
    if selected_source_hashes & calibration_hashes or selected_source_paths & calibration_paths:
        raise ValueError("unseen holdout source overlaps the calibration fit")

    write_jsonl(snapshot_path, observations)
    old_memory = read_json(old_memory_path)
    _raw_points, points = _build_points(observations, old_memory)
    if len(points) != 18:
        raise ValueError(f"expected 18 repeat-collapsed configurations, got {len(points)}")
    split_units = sorted({str(point["split_unit_id"]) for point in points})
    if len(split_units) != 4:
        raise ValueError("expected exactly four independent calibration profiles")

    allocated_anchor_details = [
        {
            "split_unit_id": point["split_unit_id"],
            "absolute_percentage_error": abs(
                float(point["allocated_anchor_bytes"])
                / float(point["observed_allocated_bytes"])
                - 1.0
            ),
        }
        for point in points
    ]
    allocated_anchor_test = {
        "scenario_equal_mape": _scenario_equal_mean(
            allocated_anchor_details, "absolute_percentage_error"
        ),
        "row_absolute_percentage_error": _metric_summary(
            [row["absolute_percentage_error"] for row in allocated_anchor_details]
        ),
    }
    original_trials = []
    for candidate in ORIGINAL_RATIO_CANDIDATES:
        for alpha in ALPHA_GRID:
            evaluation = _ratio_only_cv(points, candidate, alpha)
            original_trials.append(
                {
                    "candidate": candidate["name"],
                    "feature_count": len(candidate["feature_names"]),
                    "alpha": alpha,
                    **evaluation,
                }
            )
    original_selected = min(
        original_trials,
        key=lambda row: (
            row["scenario_equal_mape"],
            row["feature_count"],
            -row["alpha"],
            row["candidate"],
        ),
    )
    assumption_failed = bool(
        allocated_anchor_test["scenario_equal_mape"] > 0.06
        or allocated_anchor_test["row_absolute_percentage_error"]["p90"] > 0.12
    )
    if not assumption_failed:
        raise ValueError(
            "the design amendment is only valid when the allocated-anchor assumption fails"
        )

    amendment: dict[str, Any] = {
        "schema": "sft_h800_profile_aware_memory_model_amendment/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_unseen_profile_materialization_and_prediction",
        "reason": (
            "The calibration evidence falsified the original contract assumption "
            "that the cutoff-only physical allocated head remained accurate across "
            "short and long non-packing data profiles."
        ),
        "original_contract": read_json(profile_design_path)[
            "candidate_model_contract"
        ],
        "assumption_test": {
            "allocated_anchor": allocated_anchor_test,
            "failed": assumption_failed,
            "decision_thresholds": {"scenario_equal_mape": 0.06, "row_p90_ape": 0.12},
        },
        "amended_contract": {
            "formula": (
                "reserved_center = physical_allocated_anchor * "
                "exp(profile_allocated_correction) * exp(reservation_ratio)"
            ),
            "allocated_profile_head_maximum_features": 6,
            "reservation_head_maximum_features": 7,
            "repeat_policy": "collapse exact configuration repeats before fitting",
            "selection": "leave-profile-out CV with nested selection audit",
            "upper": "selector+MBS leave-profile-out maximum residual guard",
        },
        "governance": {
            "old_memory_artifact_overwritten": False,
            "old_24_row_holdout_used_for_fit": False,
            "unseen_holdout_source_rows_used_for_fit": False,
            "unseen_holdout_selection_changed": False,
            "unseen_holdout_predictions_already_generated": False,
            "gpu_jobs_launched_by_amendment": False,
            "promotion_still_requires_the_frozen_unseen_profile_holdout": True,
        },
        "source_bindings": {
            "profile_design": _binding(profile_design_path),
            "boundary_design": _binding(boundary_design_path),
            "holdout_selection": _binding(holdout_selection_path),
            "calibration_snapshot": _binding(snapshot_path),
            "old_memory_artifact": _binding(old_memory_path),
        },
    }
    amendment["report_sha256"] = sha256_json(amendment)
    write_json(amendment_path, amendment)

    allocation_selection = _select(
        points, ALLOCATION_CANDIDATES, "allocated_log_correction"
    )
    reservation_selection = _select(
        points, RESERVATION_CANDIDATES, "reservation_log_ratio"
    )
    allocated_choice = allocation_selection["selected"]
    reservation_choice = reservation_selection["selected"]
    allocated_candidate = allocated_choice["candidate"]
    reservation_candidate = reservation_choice["candidate"]
    allocated_model = _fit_ridge(
        points,
        allocated_candidate,
        float(allocated_choice["alpha"]),
        "allocated_log_correction",
    )
    reservation_model = _fit_ridge(
        points,
        reservation_candidate,
        float(reservation_choice["alpha"]),
        "reservation_log_ratio",
    )
    fixed_cv = _combined_cv(
        points,
        allocated_candidate,
        float(allocated_choice["alpha"]),
        reservation_candidate,
        float(reservation_choice["alpha"]),
    )
    nested_cv = _nested_combined_cv(points)
    tail = _tail_from_cv(fixed_cv["details"], old_memory)
    guarded = _guarded_evaluation(fixed_cv["details"], tail)

    report: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_shadow_candidate_requires_unseen_profile_holdout",
        "gpu_family": "H800",
        "publishable": False,
        "production_override_allowed": False,
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "source_bindings": {
            "profile_calibration_queue": _binding(profile_queue_path),
            "boundary_calibration_queue": _binding(boundary_queue_path),
            "profile_calibration_design": _binding(profile_design_path),
            "boundary_calibration_design": _binding(boundary_design_path),
            "calibration_snapshot": _binding(snapshot_path),
            "model_contract_amendment": _binding(amendment_path),
            "frozen_unseen_holdout_selection": _binding(holdout_selection_path),
            "allocated_anchor_artifact": _binding(old_memory_path),
            "model_inventory": _binding(ARTIFACT_DIR / "model_inventory.json"),
            "hardware": _binding(ROOT / "config" / "hardware.json"),
        },
        "calibration_admission": {
            "raw_observation_rows": len(observations),
            "unique_configuration_rows": len(points),
            "repeat_rows_collapsed": len(observations) - len(points),
            "independent_split_units": len(split_units),
            "split_unit_ids": split_units,
            "outcomes": dict(Counter(row["outcome"]["class"] for row in observations)),
            "complete_fingerprint_rows": sum(
                row["fingerprint"]["quality"] == "complete" for row in observations
            ),
            "calibration_eligible_rows": sum(
                row["fingerprint"]["calibration_evidence_eligible"] is True
                for row in observations
            ),
            "training_observation_ids_sha256": sha256_json(
                sorted(row["observation_id"] for row in observations)
            ),
            "unseen_source_overlap": False,
            "old_24_row_holdout_rows_in_training": 0,
        },
        "original_contract_assumption_test": {
            "allocated_anchor": allocated_anchor_test,
            "best_ratio_only_end_to_end": original_selected,
            "allocated_anchor_assumption_failed": assumption_failed,
            "amendment_required": True,
        },
        "selection": {
            "protocol": (
                "scenario-equal leave-profile-out MAPE; feature count and larger "
                "alpha are deterministic secondary tie breakers"
            ),
            "alpha_grid": list(ALPHA_GRID),
            "allocated": allocation_selection,
            "reservation": reservation_selection,
        },
        "model": {
            "formula": (
                "reserved_center = allocated_physical_anchor * "
                "exp(allocated_profile_correction) * exp(reservation_ratio)"
            ),
            "allocated_anchor": {
                "source_artifact": str(old_memory_path.resolve()),
                "source_artifact_sha256": sha256_file(old_memory_path),
                "source_report_sha256": old_memory["report_sha256"],
                "head": "memory.frozen_model.allocated_center_diagnostic",
            },
            "allocated_profile_correction": allocated_model,
            "reservation_ratio": reservation_model,
            "operational_upper": tail,
        },
        "evaluation": {
            "unit": "repeat-collapsed unique configuration",
            "fixed_selected_family_leave_profile_out": fixed_cv,
            "nested_model_selection_leave_profile_out": nested_cv,
            "guarded_fixed_cv": guarded,
            "interpretation": (
                "The fixed-family CV describes the frozen model family; the nested "
                "CV is the more conservative estimate including model-selection "
                "instability. Neither replaces the frozen unseen-profile holdout."
            ),
        },
        "release_gate": {
            "required_holdout_campaign_id": holdout_selection["campaign_id"],
            "required_acceptance": holdout_selection["minimum_acceptance"],
            "predictions_must_be_frozen_before_gpu": True,
            "new_exact_approval_required": True,
            "old_artifact_may_be_overwritten": False,
        },
        "publication_blockers": [
            "frozen unseen-profile prospective holdout has not run",
            "only four independent calibration profiles are available",
            "nested model-selection CV exceeds the desired six-percent center MAPE",
        ],
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-queue", type=Path, default=DEFAULT_PROFILE_QUEUE)
    parser.add_argument("--boundary-queue", type=Path, default=DEFAULT_14B_QUEUE)
    parser.add_argument("--profile-design", type=Path, default=DEFAULT_PROFILE_DESIGN)
    parser.add_argument("--boundary-design", type=Path, default=DEFAULT_14B_DESIGN)
    parser.add_argument("--holdout-selection", type=Path, default=DEFAULT_HOLDOUT_SELECTION)
    parser.add_argument("--old-memory", type=Path, default=DEFAULT_OLD_MEMORY)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = fit(
        profile_queue_path=args.profile_queue,
        boundary_queue_path=args.boundary_queue,
        profile_design_path=args.profile_design,
        boundary_design_path=args.boundary_design,
        holdout_selection_path=args.holdout_selection,
        old_memory_path=args.old_memory,
        snapshot_path=args.snapshot,
        amendment_path=args.amendment,
    )
    write_json(args.output, report)
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(args.output.resolve()),
                "report_sha256": report["report_sha256"],
                "allocated_candidate": report["model"]["allocated_profile_correction"]["candidate_name"],
                "reservation_candidate": report["model"]["reservation_ratio"]["candidate_name"],
                "fixed_cv_scenario_mape": report["evaluation"]["fixed_selected_family_leave_profile_out"]["scenario_equal_mape"],
                "nested_cv_scenario_mape": report["evaluation"]["nested_model_selection_leave_profile_out"]["scenario_equal_mape"],
                "guarded_scenario_p05_coverage": report["evaluation"]["guarded_fixed_cv"]["scenario_equal_p05_coverage"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
