#!/usr/bin/env python3
"""Fit an analysis-only unified H800 memory model from the stopped campaign.

The fit combines the frozen 273-row V5 training table with every complete,
calibration-eligible fit unit from the unified-resource campaign.  It preserves
the campaign contracts:

* repeated Critical and Packing runs are consumed only after the evaluator's
  required collapse;
* successful runs are exact reserved-memory centres;
* CUDA OOM is a right-censored capacity constraint, never an imputed peak;
* mechanism fields are inputs to one shared model, never product routes;
* model/regularisation selection is nested leave-one-source-out.

The command is CPU-only and writes a shadow diagnostic.  It cannot publish or
mutate the production predictor.
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

import numpy as np

from analyze_h800_fresh_memory_residual_v2 import profile_padding_statistics
from common import (
    ARTIFACT_DIR,
    MATRIX_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from h800_theory_basis import _model_geometry, memory_basis


SCHEMA = "sft_h800_unified_resource_partial_refit/v1"
IMPLEMENTATION_VERSION = (
    "sft_h800_unified_resource_partial_refit/"
    "2026-08-09.shared-censored-log-residual-nested-source-cv"
)
GIB = float(1 << 30)
SAFE_LIMIT_GIB = 132.83927001953126
ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
CENSOR_LOSS_WEIGHT = 1.0

DEFAULT_QUEUE = MATRIX_DIR / "h800_unified_resource_evidence_jobs_v1.jsonl"
DEFAULT_AUDIT = ARTIFACT_DIR / "h800_unified_resource_evidence_results_v1.json"
DEFAULT_INVENTORY = ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_v1.json"
DEFAULT_HARDWARE = ROOT / "config" / "hardware.json"
DEFAULT_PRIOR_ROWS = (
    ROOT
    / "diagnostics"
    / "h800_profile_expansion_offline_v1_20260809"
    / "profile_features.jsonl"
)
DEFAULT_STRICT = (
    ROOT
    / "diagnostics"
    / "h800_profile_expansion_offline_v1_20260809"
    / "strict_predictions.jsonl"
)
DEFAULT_ROUTE_STRICT = (
    ROOT
    / "diagnostics"
    / "h800_route_specific_expansion_offline_v2_20260809"
    / "strict_predictions.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "diagnostics" / "h800_unified_resource_partial_refit_20260809"
)

BASE_FEATURES = (
    "is_lora",
    "gradient_checkpointing",
    "zero2",
    "zero3",
    "log2_gpu_count",
    "log2_mbs",
    "log2_parameters_over_4b",
    "activation_share",
    "is_lora_x_activation_share",
    "is_lora_x_log2_parameters",
    "activation_share_x_log2_mbs",
    "log_effective_fraction",
    "critical_lora_x_log_effective_fraction",
)
PROFILE_FEATURES = (
    "profile_mean_fraction",
    "profile_p90_fraction",
    "profile_p99_fraction",
    "profile_expected_batch_max_fraction",
    "profile_maximum_fraction",
    "profile_cv_bounded",
    "profile_truncation_fraction",
    "profile_rare_tail_gap",
)
UNIFIED_INTERACTIONS = (
    "packing",
    "packing_x_log2_samples_per_pack",
    "packing_x_profile_mean_fraction",
    "packing_x_zero3",
    "packing_x_gradient_checkpointing",
    "zero3_x_log2_gpu_count",
    "gradient_checkpointing_x_log2_mbs",
    "is_lora_x_zero3",
)
FEATURE_SETS = {
    "analytic_intercept": (),
    "v5_feature_basis_refit": BASE_FEATURES,
    "pooled_profile_refit": (*BASE_FEATURES, *PROFILE_FEATURES),
    "unified_shared_interactions": (
        *BASE_FEATURES,
        *PROFILE_FEATURES,
        *UNIFIED_INTERACTIONS,
    ),
}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile(values: Sequence[float], probability: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    position = min(1.0, max(0.0, probability)) * (len(clean) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def _profile_values(
    statistics_row: Mapping[str, Any], *, cutoff_len: int
) -> dict[str, float]:
    cutoff = float(cutoff_len)

    def fraction(name: str) -> float:
        value = _finite(statistics_row.get(name))
        if value is None or cutoff <= 0:
            raise ValueError(f"profile statistic {name} is unavailable")
        return min(1.0, max(0.0, value / cutoff))

    expected = _finite(
        statistics_row.get("expected_random_batch_max_fraction_of_cutoff")
    )
    if expected is None:
        expected = fraction("expected_random_batch_max_tokens")
    expected = min(1.0, max(0.0, expected))
    maximum = fraction("maximum_clipped_tokens")
    cv = _finite(statistics_row.get("coefficient_of_variation"))
    truncation = _finite(statistics_row.get("truncation_fraction"))
    if cv is None or truncation is None:
        raise ValueError("profile CV/truncation is unavailable")
    return {
        "profile_mean_fraction": fraction("mean_clipped_tokens"),
        "profile_p90_fraction": fraction("p90_clipped_tokens"),
        "profile_p99_fraction": fraction("p99_clipped_tokens"),
        "profile_expected_batch_max_fraction": expected,
        "profile_maximum_fraction": maximum,
        "profile_cv_bounded": min(1.0, max(0.0, cv / 2.0)),
        "profile_truncation_fraction": min(1.0, max(0.0, truncation)),
        "profile_rare_tail_gap": max(0.0, maximum - expected),
    }


def _complete_feature_values(
    base: Mapping[str, Any], *, packing: bool, samples_per_pack: float
) -> dict[str, float]:
    values = {str(key): float(value) for key, value in base.items()}
    packing_value = float(bool(packing))
    log_samples = math.log2(max(1.0, float(samples_per_pack)))
    values.update(
        {
            "packing": packing_value,
            "packing_x_log2_samples_per_pack": packing_value * log_samples,
            "packing_x_profile_mean_fraction": (
                packing_value * values["profile_mean_fraction"]
            ),
            "packing_x_zero3": packing_value * values["zero3"],
            "packing_x_gradient_checkpointing": (
                packing_value * values["gradient_checkpointing"]
            ),
            "zero3_x_log2_gpu_count": (
                values["zero3"] * values["log2_gpu_count"]
            ),
            "gradient_checkpointing_x_log2_mbs": (
                values["gradient_checkpointing"] * values["log2_mbs"]
            ),
            "is_lora_x_zero3": values["is_lora"] * values["zero3"],
        }
    )
    missing = [
        name
        for name in FEATURE_SETS["unified_shared_interactions"]
        if name not in values or not math.isfinite(values[name])
    ]
    if missing:
        raise ValueError(f"model features are incomplete: {missing}")
    return values


def _basis_for_job(
    job: Mapping[str, Any],
    *,
    model_by_id: Mapping[str, Mapping[str, Any]],
    fixed_lora: Mapping[str, Any],
    capacity_bytes: int,
) -> tuple[dict[str, Any], int]:
    packing = bool(job.get("packing"))
    cutoff = int(job["cutoff_len"])
    sequence = cutoff if packing else int(job["aligned_effective_sequence"])
    effective_job = dict(job)
    effective_job["cutoff_len"] = sequence
    model = model_by_id[str(job["model_id"])]
    geometry = _model_geometry(effective_job, dict(model), dict(fixed_lora))
    return memory_basis(effective_job, geometry, capacity_bytes), sequence


def _current_features(
    job: Mapping[str, Any],
    *,
    model_by_id: Mapping[str, Mapping[str, Any]],
    fixed_lora: Mapping[str, Any],
    capacity_bytes: int,
    profile_cache: dict[tuple[str, int, int], dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    basis, sequence = _basis_for_job(
        job,
        model_by_id=model_by_id,
        fixed_lora=fixed_lora,
        capacity_bytes=capacity_bytes,
    )
    reference = float(basis["analytic_reference_bytes"])
    activation_share = float(basis["structural_activation_bytes"]) / reference
    cutoff = int(job["cutoff_len"])
    is_lora = float(job["train_type"] == "lora")
    gc = float(bool(job["gc"]))
    zero = int(job["zero_stage"])
    log_gpu = math.log2(float(job["gpu_count"]))
    log_mbs = math.log2(float(job["mbs"]))
    model_parameters = float(job["model_parameters"])
    fraction = min(1.0, max(1e-12, sequence / float(cutoff)))
    log_fraction = math.log(fraction)
    critical = float(
        str(job.get("mechanism_id")) == "lora_zero2_gc0_2gpu_pack0"
        and not bool(job.get("packing"))
    )
    base = {
        "is_lora": is_lora,
        "gradient_checkpointing": gc,
        "zero2": float(zero == 2),
        "zero3": float(zero == 3),
        "log2_gpu_count": log_gpu,
        "log2_mbs": log_mbs,
        "log2_parameters_over_4b": math.log2(model_parameters / 4_022_468_096.0),
        "activation_share": activation_share,
        "is_lora_x_activation_share": is_lora * activation_share,
        "is_lora_x_log2_parameters": (
            is_lora * math.log2(model_parameters / 4_022_468_096.0)
        ),
        "activation_share_x_log2_mbs": activation_share * log_mbs,
        "log_effective_fraction": log_fraction,
        "critical_lora_x_log_effective_fraction": critical * log_fraction,
    }
    profile_key = (
        str(Path(str(job["dataset_profile_path"])).resolve()),
        cutoff,
        int(job["mbs"]),
    )
    if profile_key not in profile_cache:
        profile_cache[profile_key] = profile_padding_statistics(
            Path(profile_key[0]), cutoff_len=cutoff, physical_mbs=int(job["mbs"])
        )
    base.update(_profile_values(profile_cache[profile_key], cutoff_len=cutoff))
    samples_per_pack = float(
        ((job.get("packing_contract") or {}).get("expected_samples_per_pack"))
        or 1.0
    )
    return reference, _complete_feature_values(
        base,
        packing=bool(job.get("packing")),
        samples_per_pack=samples_per_pack,
    )


def _reference_from_summary_row(
    row: Mapping[str, Any],
    *,
    model_by_id: Mapping[str, Mapping[str, Any]],
    fixed_lora: Mapping[str, Any],
    capacity_bytes: int,
) -> float:
    values = row["model_features"]
    cutoff = int(row["cutoff_len"])
    sequence = max(
        1,
        int(round(cutoff * math.exp(float(values["log_effective_fraction"])))),
    )
    model = model_by_id[str(row["model_id"])]
    job = {
        "model_parameters": int(model["actual_parameters"]),
        "train_type": str(row["training_mode"]),
        "gpu_count": int(row["gpu_count"]),
        "mbs": int(row["mbs"]),
        "cutoff_len": sequence,
        "zero": f"zero{int(row['zero_stage'])}",
        "gc": bool(row["gradient_checkpointing"]),
    }
    geometry = _model_geometry(job, dict(model), dict(fixed_lora))
    return float(memory_basis(job, geometry, capacity_bytes)["analytic_reference_bytes"])


def _prior_records(
    path: Path,
    *,
    model_by_id: Mapping[str, Mapping[str, Any]],
    fixed_lora: Mapping[str, Any],
    capacity_bytes: int,
) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if row.get("split") == "training"]
    if len(rows) != 273:
        raise ValueError(f"frozen V5 training table drifted: {len(rows)} != 273")
    records = []
    for row in rows:
        reference = _reference_from_summary_row(
            row,
            model_by_id=model_by_id,
            fixed_lora=fixed_lora,
            capacity_bytes=capacity_bytes,
        )
        values = _complete_feature_values(
            row["model_features"],
            packing=bool(row.get("packing")),
            samples_per_pack=1.0,
        )
        outcome = str(row["outcome"])
        if outcome not in {"success", "oom"}:
            raise ValueError(f"unsupported prior outcome: {outcome}")
        records.append(
            {
                "record_id": f"prior::{row['cluster_id']}",
                "source_id": str(row["source_id"]),
                "origin": "frozen_v5_training_273",
                "campaign": str(row.get("origin") or "prior_v5"),
                "role": "prior_v5_fit",
                "state": "exact" if outcome == "success" else "censored",
                "reference_bytes": reference,
                "target_reserved_bytes": (
                    float(row["observed_reserved_bytes"])
                    if outcome == "success"
                    else None
                ),
                "target_allocated_bytes": (
                    float(row["observed_allocated_bytes"])
                    if outcome == "success"
                    else None
                ),
                "censor_lower_bytes": (
                    float(capacity_bytes) if outcome == "oom" else None
                ),
                "features": values,
                "model_id": str(row["model_id"]),
                "train_type": str(row["training_mode"]),
                "gpu_count": int(row["gpu_count"]),
                "zero_stage": int(row["zero_stage"]),
                "gc": bool(row["gradient_checkpointing"]),
                "mbs": int(row["mbs"]),
                "cutoff_len": int(row["cutoff_len"]),
                "packing": bool(row.get("packing")),
                "profile_sha256": str(row.get("profile_sha256") or ""),
            }
        )
    return records


def _current_records(
    audit: Mapping[str, Any],
    queue: Sequence[Mapping[str, Any]],
    *,
    model_by_id: Mapping[str, Mapping[str, Any]],
    fixed_lora: Mapping[str, Any],
    capacity_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    jobs = {str(row["job_id"]): row for row in queue}
    observations = {str(row["job_id"]): row for row in audit["observations"]}
    profile_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    skipped = Counter()
    for unit in audit["memory_fit_units"]:
        state = str(unit["fit_state"])
        if state not in {"exact_success_center", "right_censored_oom"}:
            skipped[state] += 1
            continue
        member_ids = [str(value) for value in unit["member_job_ids"]]
        job = jobs[member_ids[0]]
        reference, values = _current_features(
            job,
            model_by_id=model_by_id,
            fixed_lora=fixed_lora,
            capacity_bytes=capacity_bytes,
            profile_cache=profile_cache,
        )
        allocated = [
            float((observations[job_id].get("measurements") or {})["max_allocated_bytes"])
            for job_id in member_ids
            if observations[job_id].get("classification") == "success"
            and (observations[job_id].get("measurements") or {}).get(
                "max_allocated_bytes"
            )
            is not None
        ]
        records.append(
            {
                "record_id": f"current::{unit['fit_unit_id']}",
                "source_id": str(job["split_unit_id"]),
                "origin": "unified_resource_campaign_partial",
                "campaign": str(job["campaign_id"]),
                "role": str(job["evidence_role"]),
                "state": (
                    "exact" if state == "exact_success_center" else "censored"
                ),
                "reference_bytes": reference,
                "target_reserved_bytes": (
                    float(unit["exact_center_target_bytes"])
                    if state == "exact_success_center"
                    else None
                ),
                "target_allocated_bytes": (
                    float(statistics.median(allocated)) if allocated else None
                ),
                "censor_lower_bytes": (
                    float(capacity_bytes)
                    if state == "right_censored_oom"
                    else None
                ),
                "features": values,
                "member_job_ids": member_ids,
                "model_id": str(job["model_id"]),
                "train_type": str(job["train_type"]),
                "gpu_count": int(job["gpu_count"]),
                "zero_stage": int(job["zero_stage"]),
                "gc": bool(job["gc"]),
                "mbs": int(job["mbs"]),
                "cutoff_len": int(job["cutoff_len"]),
                "packing": bool(job["packing"]),
                "profile_sha256": str(job["dataset_profile_sha256"]),
            }
        )
    return records, {
        "included": len(records),
        "included_states": dict(Counter(row["state"] for row in records)),
        "skipped_fit_unit_states": dict(skipped),
        "profile_cache_entries": len(profile_cache),
    }


def _physical_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["source_id"]),
        str(row["model_id"]),
        str(row["train_type"]),
        int(row["gpu_count"]),
        int(row["zero_stage"]),
        bool(row["gc"]),
        int(row["mbs"]),
        int(row["cutoff_len"]),
        bool(row["packing"]),
        str(row["profile_sha256"]),
    )


def _collapse_combined(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[_physical_key(row)].append(row)
    collapsed: list[dict[str, Any]] = []
    duplicate_groups = 0
    mixed_groups = 0
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        duplicate_groups += int(len(rows) > 1)
        states = {str(row["state"]) for row in rows}
        first = dict(rows[0])
        first["record_id"] = "combined::" + sha256_json(
            sorted(str(row["record_id"]) for row in rows)
        )[:24]
        first["origins"] = sorted({str(row["origin"]) for row in rows})
        first["member_record_ids"] = sorted(str(row["record_id"]) for row in rows)
        if states == {"exact"}:
            first["target_reserved_bytes"] = float(
                statistics.median(float(row["target_reserved_bytes"]) for row in rows)
            )
            allocated = [
                float(row["target_allocated_bytes"])
                for row in rows
                if row.get("target_allocated_bytes") is not None
            ]
            first["target_allocated_bytes"] = (
                float(statistics.median(allocated)) if allocated else None
            )
        elif states == {"censored"}:
            first["censor_lower_bytes"] = max(
                float(row["censor_lower_bytes"]) for row in rows
            )
        else:
            mixed_groups += 1
            first["state"] = "censored"
            first["target_reserved_bytes"] = None
            first["target_allocated_bytes"] = None
            first["censor_lower_bytes"] = max(
                float(row.get("censor_lower_bytes") or 0.0) for row in rows
            )
            first["mixed_success_oom_instability"] = True
        collapsed.append(first)
    return collapsed, {
        "raw_records": len(records),
        "unique_physical_records": len(collapsed),
        "duplicate_physical_groups": duplicate_groups,
        "mixed_success_oom_groups_kept_as_censored_only": mixed_groups,
        "states": dict(Counter(str(row["state"]) for row in collapsed)),
        "sources": len({str(row["source_id"]) for row in collapsed}),
    }


def _source_weights(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts = Counter(str(row["source_id"]) for row in records)
    raw = np.asarray(
        [1.0 / counts[str(row["source_id"])] for row in records], dtype=float
    )
    return raw * (len(records) / float(raw.sum()))


def _matrix(records: Sequence[Mapping[str, Any]], names: Sequence[str]) -> np.ndarray:
    if not names:
        return np.zeros((len(records), 0), dtype=float)
    return np.asarray(
        [[float(row["features"][name]) for name in names] for row in records],
        dtype=float,
    )


def _fit(
    records: Sequence[Mapping[str, Any]], *, names: Sequence[str], alpha: float
) -> dict[str, Any]:
    exact_count = sum(row["state"] == "exact" for row in records)
    if exact_count < 2:
        raise ValueError("a fit requires at least two exact centres")
    features = _matrix(records, names)
    weights = _source_weights(records)
    means = (
        np.average(features, axis=0, weights=weights)
        if len(names)
        else np.zeros(0, dtype=float)
    )
    scales = (
        np.sqrt(np.average((features - means) ** 2, axis=0, weights=weights))
        if len(names)
        else np.ones(0, dtype=float)
    )
    scales[scales < 1e-9] = 1.0
    standardized = (features - means) / scales
    design = np.column_stack((np.ones(len(records)), standardized))
    targets = np.asarray(
        [
            math.log(
                float(
                    row["target_reserved_bytes"]
                    if row["state"] == "exact"
                    else row["censor_lower_bytes"]
                )
                / float(row["reference_bytes"])
            )
            for row in records
        ],
        dtype=float,
    )
    exact = np.asarray([row["state"] == "exact" for row in records], dtype=bool)
    initial_design = design[exact]
    initial_weights = weights[exact]
    penalty = np.diag([0.0, *([float(alpha) * float(initial_weights.sum())] * len(names))])
    initial = np.linalg.pinv(
        initial_design.T @ (initial_weights[:, None] * initial_design) + penalty
    ) @ (initial_design.T @ (initial_weights * targets[exact]))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        prediction = design @ parameters
        residual = np.zeros(len(records), dtype=float)
        residual[exact] = prediction[exact] - targets[exact]
        censored = ~exact
        residual[censored] = np.minimum(0.0, prediction[censored] - targets[censored])
        effective_weights = weights.copy()
        effective_weights[censored] *= CENSOR_LOSS_WEIGHT
        loss = float(np.sum(effective_weights * residual**2) / effective_weights.sum())
        loss += float(alpha) * float(parameters[1:] @ parameters[1:])
        gradient = (
            2.0
            * (design.T @ (effective_weights * residual))
            / float(effective_weights.sum())
        )
        gradient[1:] += 2.0 * float(alpha) * parameters[1:]
        return loss, gradient

    # The squared-hinge objective is piecewise quadratic.  Solve it by an
    # active-set Ridge loop; this is algebraically identical to the generic
    # optimizer inside a stable active set and is orders of magnitude faster
    # for nested source CV.  L-BFGS-B remains a guarded fallback for a rare
    # active-set cycle at a censor boundary.
    parameters = np.asarray(initial, dtype=float)
    censored = ~exact
    total_effective_weight = float(
        np.sum(weights[exact])
        + CENSOR_LOSS_WEIGHT * np.sum(weights[censored])
    )
    seen_active_sets: set[tuple[int, ...]] = set()
    active_iterations = 0
    active_converged = False
    for active_iterations in range(1, 51):
        prediction = design @ parameters
        active_censored = censored & (prediction < targets - 1e-10)
        signature = tuple(np.flatnonzero(active_censored).tolist())
        if signature in seen_active_sets:
            break
        seen_active_sets.add(signature)
        active = exact | active_censored
        active_weights = weights[active].copy()
        active_weights[active_censored[active]] *= CENSOR_LOSS_WEIGHT
        active_design = design[active]
        active_targets = targets[active]
        active_penalty = np.diag(
            [0.0, *([float(alpha) * total_effective_weight] * len(names))]
        )
        updated = np.linalg.pinv(
            active_design.T @ (active_weights[:, None] * active_design)
            + active_penalty
        ) @ (active_design.T @ (active_weights * active_targets))
        updated_prediction = design @ updated
        updated_active = censored & (updated_prediction < targets - 1e-10)
        parameters = np.asarray(updated, dtype=float)
        if np.array_equal(updated_active, active_censored):
            active_converged = True
            break

    optimizer_summary: dict[str, Any]
    if active_converged:
        loss, gradient = objective(parameters)
        optimizer_summary = {
            "method": "piecewise_quadratic_active_set",
            "success": True,
            "message": "stable censor active set",
            "iterations": active_iterations,
            "objective": float(loss),
            "gradient_norm": float(np.linalg.norm(gradient)),
        }
    else:
        # Prediction-only imports do not need SciPy.  Keep the optional fitting
        # dependency local to the rare active-set fallback.
        from scipy.optimize import minimize

        optimized = minimize(
            lambda value: objective(value)[0],
            parameters,
            jac=lambda value: objective(value)[1],
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9},
        )
        if not optimized.success and float(np.linalg.norm(optimized.jac)) > 1e-5:
            raise RuntimeError(
                f"censored Ridge did not converge: {optimized.message}"
            )
        parameters = np.asarray(optimized.x, dtype=float)
        optimizer_summary = {
            "method": "L-BFGS-B_active_set_fallback",
            "success": bool(optimized.success),
            "message": str(optimized.message),
            "iterations": int(optimized.nit),
            "objective": float(optimized.fun),
            "gradient_norm": float(np.linalg.norm(optimized.jac)),
        }
    raw_coefficients = parameters[1:] / scales if len(names) else np.zeros(0)
    raw_intercept = float(parameters[0] - means @ raw_coefficients)
    return {
        "model_family": "shared_source_balanced_censored_log_residual_ridge",
        "target": "log(reserved_center_bytes / analytic_reference_bytes)",
        "censor_loss": "squared hinge below device-capacity lower bound",
        "censor_loss_weight": CENSOR_LOSS_WEIGHT,
        "feature_names": list(names),
        "alpha": float(alpha),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "intercept": float(parameters[0]),
        "coefficients": parameters[1:].tolist(),
        "raw_intercept": raw_intercept,
        "raw_coefficients": raw_coefficients.tolist(),
        "fit_records": len(records),
        "fit_exact_centres": exact_count,
        "fit_right_censored": len(records) - exact_count,
        "fit_sources": len({str(row["source_id"]) for row in records}),
        "optimizer": optimizer_summary,
    }


def _predict(row: Mapping[str, Any], model: Mapping[str, Any]) -> float:
    names = [str(name) for name in model["feature_names"]]
    features = np.asarray([float(row["features"][name]) for name in names])
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    correction = float(model["intercept"])
    if names:
        correction += float(((features - means) / scales) @ coefficients)
    return float(row["reference_bytes"]) * math.exp(correction)


def _prediction_details(
    train: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str],
    alpha: float,
    held_out_source: str | None = None,
    selected_feature_set: str | None = None,
) -> list[dict[str, Any]]:
    model = _fit(train, names=names, alpha=alpha)
    details = []
    for row in test:
        predicted = _predict(row, model)
        detail = {
            "record_id": row["record_id"],
            "source_id": row["source_id"],
            "origin": row["origin"],
            "role": row["role"],
            "state": row["state"],
            "model_id": row["model_id"],
            "train_type": row["train_type"],
            "gpu_count": row["gpu_count"],
            "zero_stage": row["zero_stage"],
            "gc": row["gc"],
            "packing": row["packing"],
            "reference_bytes": row["reference_bytes"],
            "predicted_reserved_bytes": predicted,
            "predicted_reserved_gib": predicted / GIB,
            "held_out_source": held_out_source,
            "selected_feature_set": selected_feature_set,
            "selected_alpha": float(alpha),
        }
        if row["state"] == "exact":
            observed = float(row["target_reserved_bytes"])
            detail.update(
                {
                    "observed_reserved_bytes": observed,
                    "observed_reserved_gib": observed / GIB,
                    "absolute_percentage_error": abs(predicted / observed - 1.0),
                    "signed_percentage_error": predicted / observed - 1.0,
                }
            )
        else:
            lower = float(row["censor_lower_bytes"])
            detail.update(
                {
                    "censor_lower_bytes": lower,
                    "censor_lower_gib": lower / GIB,
                    "censor_satisfied": predicted >= lower,
                    "censor_shortfall_fraction": max(0.0, 1.0 - predicted / lower),
                }
            )
        details.append(detail)
    return details


def _metrics(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact = [row for row in details if row["state"] == "exact"]
    censored = [row for row in details if row["state"] == "censored"]
    by_source: dict[str, list[float]] = defaultdict(list)
    for row in exact:
        by_source[str(row["source_id"])].append(float(row["absolute_percentage_error"]))
    errors = [float(row["absolute_percentage_error"]) for row in exact]
    signed = [float(row["signed_percentage_error"]) for row in exact]
    shortfalls = [float(row["censor_shortfall_fraction"]) for row in censored]
    return {
        "exact_centres": len(exact),
        "exact_sources": len(by_source),
        "row_mape": statistics.fmean(errors) if errors else None,
        "source_equal_mape": (
            statistics.fmean(statistics.fmean(values) for values in by_source.values())
            if by_source
            else None
        ),
        "median_ape": _percentile(errors, 0.5),
        "p90_ape": _percentile(errors, 0.9),
        "max_ape": max(errors) if errors else None,
        "signed_bias": statistics.fmean(signed) if signed else None,
        "right_censored": len(censored),
        "censor_satisfied": sum(bool(row["censor_satisfied"]) for row in censored),
        "censor_satisfaction_rate": (
            statistics.fmean(float(bool(row["censor_satisfied"])) for row in censored)
            if censored
            else None
        ),
        "mean_censor_shortfall_fraction": (
            statistics.fmean(shortfalls) if shortfalls else None
        ),
        "selection_score": (
            (
                statistics.fmean(
                    statistics.fmean(values) for values in by_source.values()
                )
                if by_source
                else 1.0
            )
            + 0.05
            * (
                1.0
                - statistics.fmean(
                    float(bool(row["censor_satisfied"])) for row in censored
                )
                if censored
                else 0.0
            )
            + 0.05 * (statistics.fmean(shortfalls) if shortfalls else 0.0)
        ),
    }


def _logo(
    records: Sequence[Mapping[str, Any]], *, names: Sequence[str], alpha: float
) -> list[dict[str, Any]]:
    details = []
    for held in sorted({str(row["source_id"]) for row in records}):
        train = [row for row in records if str(row["source_id"]) != held]
        test = [row for row in records if str(row["source_id"]) == held]
        details.extend(
            _prediction_details(
                train,
                test,
                names=names,
                alpha=alpha,
                held_out_source=held,
            )
        )
    return details


def _candidate_grid() -> list[dict[str, Any]]:
    return [
        {"feature_set": name, "feature_names": list(features), "alpha": alpha}
        for name, features in FEATURE_SETS.items()
        for alpha in ALPHAS
    ]


def _rank_candidates(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for candidate in _candidate_grid():
        details = _logo(
            records,
            names=candidate["feature_names"],
            alpha=float(candidate["alpha"]),
        )
        ranked.append({**candidate, "source_grouped_cv": _metrics(details)})
    return sorted(
        ranked,
        key=lambda row: (
            float(row["source_grouped_cv"]["selection_score"]),
            len(row["feature_names"]),
            float(row["alpha"]),
        ),
    )


def _nested_logo(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    sources = sorted({str(row["source_id"]) for row in records})
    for held in sources:
        outer_train = [row for row in records if str(row["source_id"]) != held]
        outer_test = [row for row in records if str(row["source_id"]) == held]
        ranked = _rank_candidates(outer_train)
        selected = ranked[0]
        fold_predictions = _prediction_details(
            outer_train,
            outer_test,
            names=selected["feature_names"],
            alpha=float(selected["alpha"]),
            held_out_source=held,
            selected_feature_set=str(selected["feature_set"]),
        )
        predictions.extend(fold_predictions)
        folds.append(
            {
                "held_out_source": held,
                "train_records": len(outer_train),
                "test_records": len(outer_test),
                "selected_feature_set": selected["feature_set"],
                "selected_alpha": selected["alpha"],
                "inner_source_grouped_cv": selected["source_grouped_cv"],
                "outer_metrics": _metrics(fold_predictions),
            }
        )
    return predictions, folds


def _breakdowns(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    definitions = {
        "origin": lambda row: str(row["origin"]),
        "role": lambda row: str(row["role"]),
        "train_type": lambda row: str(row["train_type"]),
        "gpu_count": lambda row: str(row["gpu_count"]),
        "packing": lambda row: str(bool(row["packing"])).lower(),
    }
    for label, key in definitions.items():
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in details:
            grouped[key(row)].append(row)
        result[label] = {name: _metrics(rows) for name, rows in sorted(grouped.items())}
    return result


def _strict_records(
    path: Path,
    *,
    model_by_id: Mapping[str, Mapping[str, Any]],
    fixed_lora: Mapping[str, Any],
    capacity_bytes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successes = []
    all_rows = read_jsonl(path)
    for row in all_rows:
        if row.get("outcome") != "success":
            continue
        reference = _reference_from_summary_row(
            row,
            model_by_id=model_by_id,
            fixed_lora=fixed_lora,
            capacity_bytes=capacity_bytes,
        )
        if row.get("analytic_reference_bytes") is not None and not math.isclose(
            reference,
            float(row["analytic_reference_bytes"]),
            rel_tol=0.0,
            abs_tol=1.0,
        ):
            raise ValueError("strict analytic reference reconstruction drifted")
        successes.append(
            {
                "record_id": f"strict::{row['observation_id']}",
                "source_id": str(row["source_id"]),
                "origin": "strict_unused_dataset_diagnostic",
                "campaign": str(row.get("origin") or "strict"),
                "role": "strict_unused_dataset",
                "state": "exact",
                "reference_bytes": reference,
                "target_reserved_bytes": float(row["observed_reserved_bytes"]),
                "target_allocated_bytes": float(row["observed_allocated_bytes"]),
                "censor_lower_bytes": None,
                "features": _complete_feature_values(
                    row["model_features"],
                    packing=bool(row.get("packing")),
                    samples_per_pack=1.0,
                ),
                "model_id": str(row["model_id"]),
                "train_type": str(row["training_mode"]),
                "gpu_count": int(row["gpu_count"]),
                "zero_stage": int(row["zero_stage"]),
                "gc": bool(row["gradient_checkpointing"]),
                "mbs": int(row["mbs"]),
                "cutoff_len": int(row["cutoff_len"]),
                "packing": bool(row.get("packing")),
                "profile_sha256": str(row.get("profile_sha256") or ""),
                "existing_predictions": {
                    "analytic_reference": reference,
                    "current_v5": float(row["current_v5_reserved_center_bytes"]),
                    "pooled_profile": float(
                        row["profile_expansion_reserved_center_bytes"]
                    ),
                },
                "cluster_id": str(row["cluster_id"]),
            }
        )
    return successes, all_rows


def _simple_prediction_metrics(
    rows: Sequence[Mapping[str, Any]], prediction: Mapping[str, float]
) -> dict[str, Any]:
    details = []
    for row in rows:
        predicted = float(prediction[str(row["record_id"])])
        observed = float(row["target_reserved_bytes"])
        details.append(
            {
                "state": "exact",
                "source_id": row["source_id"],
                "absolute_percentage_error": abs(predicted / observed - 1.0),
                "signed_percentage_error": predicted / observed - 1.0,
            }
        )
    return _metrics(details)


def _strict_evaluation(
    strict: Sequence[Mapping[str, Any]],
    *,
    combined_model: Mapping[str, Any],
    prior_only_model: Mapping[str, Any],
    route_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    route_by_cluster = (
        {str(row["cluster_id"]): row for row in read_jsonl(route_path)}
        if route_path.is_file()
        else {}
    )
    prediction_maps: dict[str, dict[str, float]] = defaultdict(dict)
    details = []
    for row in strict:
        row_id = str(row["record_id"])
        existing = row["existing_predictions"]
        for name, value in existing.items():
            prediction_maps[str(name)][row_id] = float(value)
        prediction_maps["prior_only_same_model"][row_id] = _predict(
            row, prior_only_model
        )
        prediction_maps["combined_unified_partial"][row_id] = _predict(
            row, combined_model
        )
        route = route_by_cluster.get(str(row["cluster_id"])) or {}
        if route.get("route_reserved_center_bytes") is not None:
            prediction_maps["route_specific_diagnostic"][row_id] = float(
                route["route_reserved_center_bytes"]
            )
        details.append(
            {
                "record_id": row_id,
                "source_id": row["source_id"],
                "model_id": row["model_id"],
                "train_type": row["train_type"],
                "gpu_count": row["gpu_count"],
                "observed_reserved_gib": float(row["target_reserved_bytes"]) / GIB,
                "predictions_gib": {
                    name: values[row_id] / GIB
                    for name, values in prediction_maps.items()
                    if row_id in values
                },
            }
        )
    metrics = {
        name: _simple_prediction_metrics(strict, values)
        for name, values in prediction_maps.items()
        if len(values) == len(strict)
    }
    return metrics, details


def _format_percent(value: Any) -> str:
    number = _finite(value)
    return "—" if number is None else f"{100.0 * number:.2f}%"


def _markdown(report: Mapping[str, Any]) -> str:
    audit = report["data_audit"]
    nested = report["nested_source_grouped_cv"]
    lines = [
        "# H800 统一资源模型：中途停止后的部分重拟合",
        "",
        "> CPU-only 影子诊断；没有修改生产模型或发布配置。",
        "",
        "## 数据状态",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        f"| 实验完成作业 | {audit['campaign_completed_jobs']} / {audit['campaign_expected_jobs']} |",
        f"| 旧 V5 折叠训练配置 | {audit['prior_records']} |",
        f"| 本轮可用折叠单元 | {audit['current_records']} |",
        f"| 合并去重物理配置 | {audit['combined']['unique_physical_records']} |",
        f"| 合并精确中心 | {audit['combined']['states'].get('exact', 0)} |",
        f"| 合并 OOM 右删失约束 | {audit['combined']['states'].get('censored', 0)} |",
        f"| 合并独立源 | {audit['combined']['sources']} |",
        "",
        "## 候选的整源留出结果",
        "",
        "| 特征族 | alpha | Source-equal MAPE | P90 APE | OOM 约束满足率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["candidate_source_grouped_cv"]:
        metrics = row["source_grouped_cv"]
        lines.append(
            f"| {row['feature_set']} | {row['alpha']} | "
            f"{_format_percent(metrics['source_equal_mape'])} | "
            f"{_format_percent(metrics['p90_ape'])} | "
            f"{_format_percent(metrics['censor_satisfaction_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## 嵌套整源留出（无选型泄漏）",
            "",
            "| 指标 | 结果 |",
            "|---|---:|",
            f"| 精确中心数 | {nested['metrics']['exact_centres']} |",
            f"| Source-equal MAPE | {_format_percent(nested['metrics']['source_equal_mape'])} |",
            f"| 行 MAPE | {_format_percent(nested['metrics']['row_mape'])} |",
            f"| P90 APE | {_format_percent(nested['metrics']['p90_ape'])} |",
            f"| 最大 APE | {_format_percent(nested['metrics']['max_ape'])} |",
            f"| OOM 约束满足率 | {_format_percent(nested['metrics']['censor_satisfaction_rate'])} |",
            "",
            "## 既有严格未见成功配置（29 条）",
            "",
            "| 模型 | Source-equal MAPE | 行 MAPE | P90 APE |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, metrics in report["strict_unused_success_diagnostic"]["metrics"].items():
        lines.append(
            f"| {name} | {_format_percent(metrics['source_equal_mape'])} | "
            f"{_format_percent(metrics['row_mape'])} | "
            f"{_format_percent(metrics['p90_ape'])} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 本轮中止任务及 12 个不完整 Packing 物理臂未进入拟合。",
            "- 当前完成的 Packing 中只有 2-GPU FULL 物理臂；不能据此宣称 4-GPU Packing 已校准。",
            "- OOM 只作为“需求超过设备容量”的右删失不等式，不伪造显存峰值。",
            "- 严格未见集是既有、已被分析过的历史诊断，不是新 prospective acceptance。",
            "- 该产物仅用于判断当前拟合方向和数据增益，`publishable=false`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--hardware", type=Path, default=DEFAULT_HARDWARE)
    parser.add_argument("--prior-rows", type=Path, default=DEFAULT_PRIOR_ROWS)
    parser.add_argument("--strict", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--route-strict", type=Path, default=DEFAULT_ROUTE_STRICT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    queue = read_jsonl(args.queue)
    audit = read_json(args.audit)
    if audit.get("campaign_id") != "h800_unified_resource_evidence_20260809_v1":
        raise ValueError("unified campaign audit identity drifted")
    if int(audit.get("completed_jobs") or 0) >= int(audit.get("expected_jobs") or 0):
        raise ValueError("this command is specifically the stopped partial-campaign refit")

    inventory = read_json(args.inventory)
    model_by_id = {str(row["id"]): row for row in inventory["models"]}
    fixed_lora = inventory["fixed_lora"]
    hardware = read_json(args.hardware)
    capacity_bytes = int(hardware["memory_bytes_reported_by_torch"])

    print("step 1/5: loading frozen V5 fit rows and complete campaign units", flush=True)
    prior = _prior_records(
        args.prior_rows,
        model_by_id=model_by_id,
        fixed_lora=fixed_lora,
        capacity_bytes=capacity_bytes,
    )
    current, current_audit = _current_records(
        audit,
        queue,
        model_by_id=model_by_id,
        fixed_lora=fixed_lora,
        capacity_bytes=capacity_bytes,
    )
    combined, collapse = _collapse_combined([*prior, *current])

    print("step 2/5: comparing shared feature families by source-grouped CV", flush=True)
    candidates = _rank_candidates(combined)
    best = candidates[0]

    print("step 3/5: running nested source-grouped model selection", flush=True)
    nested_predictions, nested_folds = _nested_logo(combined)
    nested_metrics = _metrics(nested_predictions)

    print("step 4/5: fitting final shadow and scoring existing strict sources", flush=True)
    final_model = _fit(
        combined,
        names=best["feature_names"],
        alpha=float(best["alpha"]),
    )
    prior_only_model = _fit(
        prior,
        names=best["feature_names"],
        alpha=float(best["alpha"]),
    )
    strict, strict_raw = _strict_records(
        args.strict,
        model_by_id=model_by_id,
        fixed_lora=fixed_lora,
        capacity_bytes=capacity_bytes,
    )
    strict_metrics, strict_details = _strict_evaluation(
        strict,
        combined_model=final_model,
        prior_only_model=prior_only_model,
        route_path=args.route_strict,
    )

    print("step 5/5: writing analysis-only model and report", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_artifact = {
        "schema": "sft_h800_unified_resource_memory_shadow_model/v1",
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "partial_campaign_shadow_fit",
        "analysis_only": True,
        "publishable": False,
        "production_model_mutated": False,
        "single_shared_model": True,
        "mechanism_fields_are_features_not_routes": True,
        "selected_feature_set": best["feature_set"],
        "selection": {
            "protocol": "source-grouped CV on the combined fit set",
            "selected_alpha": best["alpha"],
            "selected_metrics": best["source_grouped_cv"],
        },
        "model": final_model,
        "training_contract": {
            "exact_success_centres": True,
            "oom_right_censored_at_device_capacity": True,
            "mixed_success_oom_not_used_as_exact_centre": True,
            "repeats_collapsed_before_memory_fit": True,
            "source_balanced": True,
        },
    }
    model_artifact["artifact_sha256"] = sha256_json(model_artifact)
    write_json(args.output_dir / "candidate_model.json", model_artifact)
    write_jsonl(args.output_dir / "nested_oof_predictions.jsonl", nested_predictions)
    write_jsonl(args.output_dir / "strict_predictions.jsonl", strict_details)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "partial_campaign_shadow_refit_complete",
        "analysis_only": True,
        "publishable": False,
        "production_model_mutated": False,
        "data_audit": {
            "campaign_completed_jobs": int(audit["completed_jobs"]),
            "campaign_expected_jobs": int(audit["expected_jobs"]),
            "campaign_outcomes": dict(audit["raw_outcomes"]),
            "campaign_fit_unit_states": dict(audit["memory_fit_unit_states"]),
            "prior_records": len(prior),
            "current_records": len(current),
            "current": current_audit,
            "combined": collapse,
            "strict_rows_total": len(strict_raw),
            "strict_success_rows_scored": len(strict),
        },
        "inputs": {
            "queue": {"path": str(args.queue.resolve()), "sha256": sha256_file(args.queue)},
            "audit": {"path": str(args.audit.resolve()), "sha256": sha256_file(args.audit)},
            "inventory": {"path": str(args.inventory.resolve()), "sha256": sha256_file(args.inventory)},
            "hardware": {"path": str(args.hardware.resolve()), "sha256": sha256_file(args.hardware)},
            "prior_rows": {"path": str(args.prior_rows.resolve()), "sha256": sha256_file(args.prior_rows)},
            "strict": {"path": str(args.strict.resolve()), "sha256": sha256_file(args.strict)},
        },
        "model_contract": {
            "target": "reserved memory centre",
            "physics_anchor": "effective-sequence analytic reference",
            "one_shared_model": True,
            "mechanism_inputs_not_routes": True,
            "oom_treatment": "right-censored device-capacity constraint",
            "selection": "nested leave-one-source-out",
        },
        "candidate_source_grouped_cv": candidates,
        "selected_candidate": best,
        "nested_source_grouped_cv": {
            "protocol": (
                "outer leave-one-source-out; feature family and alpha selected only "
                "inside each outer training fold"
            ),
            "metrics": nested_metrics,
            "breakdowns": _breakdowns(nested_predictions),
            "folds": nested_folds,
            "selected_feature_set_counts": dict(
                Counter(str(row["selected_feature_set"]) for row in nested_folds)
            ),
        },
        "strict_unused_success_diagnostic": {
            "interpretation": (
                "existing source-disjoint historical diagnostic; previously inspected, "
                "therefore not prospective acceptance"
            ),
            "metrics": strict_metrics,
        },
        "campaign_grouped_sensitivity": {
            "comparison": "same selected model family/alpha, prior-only versus prior+current",
            "prior_only_strict_source_equal_mape": strict_metrics[
                "prior_only_same_model"
            ]["source_equal_mape"],
            "combined_strict_source_equal_mape": strict_metrics[
                "combined_unified_partial"
            ]["source_equal_mape"],
        },
        "limitations": [
            "campaign stopped at 186/220 jobs",
            "12 incomplete packing physical arms excluded",
            "current complete packing centres cover only 2-GPU FULL",
            "strict diagnostic has been previously inspected and is not prospective",
            "shadow model is not bound to the production predictor",
        ],
    }
    report["report_sha256"] = sha256_json(report)
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "selected_feature_set": best["feature_set"],
                "selected_alpha": best["alpha"],
                "nested_source_equal_mape": nested_metrics["source_equal_mape"],
                "nested_censor_satisfaction_rate": nested_metrics[
                    "censor_satisfaction_rate"
                ],
                "strict_combined_source_equal_mape": strict_metrics[
                    "combined_unified_partial"
                ]["source_equal_mape"],
                "publishable": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
