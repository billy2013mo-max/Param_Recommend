#!/usr/bin/env python3
"""Benchmark unified H800 reserved-memory centre models on fixed source folds.

This CPU-only diagnostic tests whether the current error is caused by the
linear residual model rather than by a lack of terminal observations.  Every
candidate remains one shared model: source/dataset identifiers are grouping
keys only and are never inputs.  OOM rows remain right-censored constraints.

The primary comparison is nested grouped cross-validation.  Candidate ranking
also includes leave-one-source-out diagnostics, but that ranking is not used as
an unbiased acceptance score.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fit_h800_unified_resource_partial_v1 as base
import numpy as np
from common import (
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)

SCHEMA = "sft_h800_memory_center_model_benchmark/v1"
IMPLEMENTATION_VERSION = (
    "sft_h800_memory_center_model_benchmark/"
    "2026-08-09.physics-anchor-scale-bounded-nonlinear-source-cv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "diagnostics" / "h800_memory_center_model_benchmark_20260809"
)
GIB = float(1 << 30)
DEVICE_CAPACITY_BYTES = float(
    read_json(base.DEFAULT_HARDWARE)["memory_bytes_reported_by_torch"]
)

LEGACY_FEATURE_NAMES = base.FEATURE_SETS["unified_shared_interactions"]
ANCHOR_SCALE_FEATURE_NAMES = (
    "reference_fraction_of_capacity",
    "activation_fraction_of_capacity",
    "nonactivation_fraction_of_capacity",
)
LORA_ZERO3_NOGC_ACTIVATION_FEATURE = "lora_zero3_nogc_activation_fraction_of_capacity"
LORA_ZERO3_NOGC_TRUNCATION_FEATURE = "lora_zero3_nogc_truncation_activation_pressure"
LORA_ZERO3_NOGC_RARE_TAIL_FEATURE = "lora_zero3_nogc_rare_tail_activation_pressure"
FULL_ZERO3_GC_REFERENCE_FEATURE = "full_zero3_gc_reference_fraction_of_capacity"
FULL_SINGLE_GPU_ZERO0_NOGC_MBS_FEATURE = (
    "full_single_gpu_zero0_nogc_mbs_reference_pressure"
)
TIED_EMBEDDING_GC_REFERENCE_FEATURE = "tied_embedding_gc_reference_pressure"
V3_MECHANISM_FEATURE_NAMES = (
    LORA_ZERO3_NOGC_ACTIVATION_FEATURE,
    LORA_ZERO3_NOGC_TRUNCATION_FEATURE,
    LORA_ZERO3_NOGC_RARE_TAIL_FEATURE,
    FULL_ZERO3_GC_REFERENCE_FEATURE,
    FULL_SINGLE_GPU_ZERO0_NOGC_MBS_FEATURE,
    TIED_EMBEDDING_GC_REFERENCE_FEATURE,
)
TIED_EMBEDDING_MODEL_IDS = frozenset({"qwen3_1p7b", "qwen3_4b"})
RAW_FEATURE_NAMES = (*LEGACY_FEATURE_NAMES, *ANCHOR_SCALE_FEATURE_NAMES)
GATE_NAMES = (
    "is_lora",
    "gradient_checkpointing",
    "zero2",
    "zero3",
    "packing",
)
ACTIVATION_MECHANISM_FEATURE_NAMES = tuple(
    f"{gate}_x_activation_fraction_of_capacity" for gate in GATE_NAMES
)
MECHANISM_CONTINUOUS_NAMES = (
    "log2_gpu_count",
    "log2_mbs",
    "log2_parameters_over_4b",
    "activation_share",
    "log_effective_fraction",
    "profile_mean_fraction",
    "profile_p90_fraction",
    "profile_p99_fraction",
    "profile_expected_batch_max_fraction",
    "profile_maximum_fraction",
    "profile_cv_bounded",
    "profile_truncation_fraction",
    "profile_rare_tail_gap",
    *ANCHOR_SCALE_FEATURE_NAMES,
)


def _load_records() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    queue = read_jsonl(base.DEFAULT_QUEUE)
    audit = read_json(base.DEFAULT_AUDIT)
    inventory = read_json(base.DEFAULT_INVENTORY)
    hardware = read_json(base.DEFAULT_HARDWARE)
    model_by_id = {str(row["id"]): row for row in inventory["models"]}
    capacity_bytes = int(hardware["memory_bytes_reported_by_torch"])
    prior = base._prior_records(
        base.DEFAULT_PRIOR_ROWS,
        model_by_id=model_by_id,
        fixed_lora=inventory["fixed_lora"],
        capacity_bytes=capacity_bytes,
    )
    current, current_audit = base._current_records(
        audit,
        queue,
        model_by_id=model_by_id,
        fixed_lora=inventory["fixed_lora"],
        capacity_bytes=capacity_bytes,
    )
    combined, collapse = base._collapse_combined([*prior, *current])
    strict, _ = base._strict_records(
        base.DEFAULT_STRICT,
        model_by_id=model_by_id,
        fixed_lora=inventory["fixed_lora"],
        capacity_bytes=capacity_bytes,
    )
    return (
        combined,
        strict,
        {
            "prior_records": len(prior),
            "current_records": len(current),
            "current_audit": current_audit,
            "combined": collapse,
            "capacity_bytes": capacity_bytes,
            "repeat_noise": _repeat_noise_audit(audit),
        },
    )


def _repeat_noise_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    observations = {str(row["job_id"]): row for row in audit.get("observations") or []}
    by_state: dict[str, list[float]] = defaultdict(list)
    for unit in audit.get("memory_fit_units") or []:
        values = [
            float(
                (observations[job_id].get("measurements") or {})["max_reserved_bytes"]
            )
            for job_id in unit.get("member_job_ids") or []
            if observations[job_id].get("classification") == "success"
            and (observations[job_id].get("measurements") or {}).get(
                "max_reserved_bytes"
            )
            is not None
        ]
        if len(values) >= 2:
            by_state[str(unit["fit_state"])].append(max(values) / min(values) - 1.0)
    return {
        state: {
            "repeat_groups": len(values),
            "median_max_min_spread": base._percentile(values, 0.5),
            "p90_max_min_spread": base._percentile(values, 0.9),
            "max_max_min_spread": max(values) if values else None,
        }
        for state, values in sorted(by_state.items())
    }


def _source_weights(
    records: Sequence[Mapping[str, Any]], *, power: float = 1.0
) -> np.ndarray:
    if not 0.0 <= power <= 1.0:
        raise ValueError("source weight power must be between zero and one")
    counts = Counter(str(row["source_id"]) for row in records)
    weights = np.asarray(
        [counts[str(row["source_id"])] ** (-power) for row in records], dtype=float
    )
    return weights * (len(records) / float(weights.sum()))


def _feature_value(row: Mapping[str, Any], name: str) -> float:
    if name in row["features"]:
        return float(row["features"][name])
    reference_fraction = float(row["reference_bytes"]) / DEVICE_CAPACITY_BYTES
    activation_share = float(row["features"]["activation_share"])
    if name == "reference_fraction_of_capacity":
        return reference_fraction
    if name == "activation_fraction_of_capacity":
        return reference_fraction * activation_share
    if name == "nonactivation_fraction_of_capacity":
        return reference_fraction * (1.0 - activation_share)
    if name == LORA_ZERO3_NOGC_ACTIVATION_FEATURE:
        return (
            float(row["features"]["is_lora"])
            * float(row["features"]["zero3"])
            * (1.0 - float(row["features"]["gradient_checkpointing"]))
            * reference_fraction
            * activation_share
        )
    is_lora = float(row["features"]["is_lora"])
    zero2 = float(row["features"]["zero2"])
    zero3 = float(row["features"]["zero3"])
    gc = float(row["features"]["gradient_checkpointing"])
    if name == LORA_ZERO3_NOGC_TRUNCATION_FEATURE:
        return (
            is_lora
            * zero3
            * (1.0 - gc)
            * reference_fraction
            * activation_share
            * float(row["features"]["profile_truncation_fraction"])
        )
    if name == LORA_ZERO3_NOGC_RARE_TAIL_FEATURE:
        return (
            is_lora
            * zero3
            * (1.0 - gc)
            * reference_fraction
            * activation_share
            * float(row["features"]["profile_rare_tail_gap"])
        )
    if name == FULL_ZERO3_GC_REFERENCE_FEATURE:
        return (1.0 - is_lora) * zero3 * gc * reference_fraction
    if name == FULL_SINGLE_GPU_ZERO0_NOGC_MBS_FEATURE:
        single_gpu = float(abs(float(row["features"]["log2_gpu_count"])) < 1.0e-12)
        zero0 = max(0.0, 1.0 - zero2 - zero3)
        return (
            (1.0 - is_lora)
            * single_gpu
            * zero0
            * (1.0 - gc)
            * reference_fraction
            * float(row["features"]["log2_mbs"])
        )
    if name == TIED_EMBEDDING_GC_REFERENCE_FEATURE:
        tied = float(str(row["model_id"]) in TIED_EMBEDDING_MODEL_IDS)
        return tied * gc * reference_fraction
    suffix = "_x_activation_fraction_of_capacity"
    if name.endswith(suffix):
        gate = name[: -len(suffix)]
        if gate not in GATE_NAMES:
            raise KeyError(name)
        return float(row["features"][gate]) * reference_fraction * activation_share
    raise KeyError(name)


def _raw_matrix(
    records: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [[_feature_value(row, name) for name in feature_names] for row in records],
        dtype=float,
    )


def _weighted_location_scale(
    matrix: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    means = np.average(matrix, axis=0, weights=weights)
    scales = np.sqrt(np.average((matrix - means) ** 2, axis=0, weights=weights))
    scales[scales < 1.0e-9] = 1.0
    return means, scales


def _basis_names(
    kind: str,
    feature_names: Sequence[str],
    nonlinear_indexes: Sequence[int],
) -> list[str]:
    names = list(feature_names)
    if kind in {
        "anchor_quadratic",
        "anchor_mechanism",
        "bounded_quadratic",
        "bounded_mechanism",
    }:
        names.extend(f"square::{feature_names[index]}" for index in nonlinear_indexes)
    if kind in {"anchor_mechanism", "bounded_mechanism"}:
        interaction_names = (
            ANCHOR_SCALE_FEATURE_NAMES
            if kind == "anchor_mechanism"
            else MECHANISM_CONTINUOUS_NAMES
        )
        for gate in GATE_NAMES:
            for continuous in interaction_names:
                names.append(f"gate::{gate}::{continuous}")
    return names


def _expand_basis(
    raw: np.ndarray,
    *,
    kind: str,
    raw_means: np.ndarray,
    raw_scales: np.ndarray,
    nonlinear_indexes: Sequence[int],
    feature_names: Sequence[str],
) -> np.ndarray:
    standardized = (raw - raw_means) / raw_scales
    if kind in {"legacy_linear", "linear"}:
        return standardized
    if kind in {"anchor_quadratic", "anchor_mechanism"}:
        columns = [standardized, standardized[:, nonlinear_indexes] ** 2]
        if kind == "anchor_mechanism":
            index_by_name = {name: index for index, name in enumerate(feature_names)}
            gates = raw[:, [index_by_name[name] for name in GATE_NAMES]]
            anchors = standardized[
                :, [index_by_name[name] for name in ANCHOR_SCALE_FEATURE_NAMES]
            ]
            columns.append(
                np.einsum("ni,nj->nij", gates, anchors).reshape(len(raw), -1)
            )
        return np.column_stack(columns)
    bounded = 2.0 * np.tanh(standardized / 2.0)
    columns = [bounded]
    if kind in {"bounded_quadratic", "bounded_mechanism"} and nonlinear_indexes:
        quadratic = bounded[:, nonlinear_indexes] ** 2
        columns.append(quadratic)
    if kind == "bounded_mechanism":
        index_by_name = {name: index for index, name in enumerate(feature_names)}
        gates = raw[:, [index_by_name[name] for name in GATE_NAMES]]
        continuous = bounded[
            :, [index_by_name[name] for name in MECHANISM_CONTINUOUS_NAMES]
        ]
        interactions = np.einsum("ni,nj->nij", gates, continuous).reshape(len(raw), -1)
        columns.append(interactions)
    if kind not in {
        "bounded_linear",
        "bounded_quadratic",
        "bounded_mechanism",
    }:
        raise ValueError(f"unknown basis kind: {kind}")
    return np.column_stack(columns)


def _targets(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    exact = np.asarray([row["state"] == "exact" for row in records], dtype=bool)
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
    return targets, exact


def _solve_weighted_ridge(
    design: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=float) * float(alpha) * float(weights.sum())
    penalty[0, 0] = 0.0
    normal = design.T @ (weights[:, None] * design) + penalty
    rhs = design.T @ (weights * targets)
    try:
        return np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(normal) @ rhs


def _fit_model(
    records: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    kind = str(candidate["basis_kind"])
    feature_variant = str(candidate.get("feature_variant") or "anchor_scale")
    if kind == "legacy_linear" or feature_variant == "legacy":
        feature_names = LEGACY_FEATURE_NAMES
    elif feature_variant == "activation_scale":
        feature_names = (*LEGACY_FEATURE_NAMES, "activation_fraction_of_capacity")
    elif feature_variant == "activation_mechanism":
        feature_names = (
            *LEGACY_FEATURE_NAMES,
            "activation_fraction_of_capacity",
            *ACTIVATION_MECHANISM_FEATURE_NAMES,
        )
    elif feature_variant == "anchor_scale":
        feature_names = RAW_FEATURE_NAMES
    elif feature_variant == "anchor_scale_lora_zero3_nogc_activation":
        feature_names = (
            *RAW_FEATURE_NAMES,
            LORA_ZERO3_NOGC_ACTIVATION_FEATURE,
        )
    elif feature_variant == "anchor_scale_v3_mechanisms":
        feature_names = (*RAW_FEATURE_NAMES, *V3_MECHANISM_FEATURE_NAMES)
    else:
        raise ValueError(f"unknown feature variant: {feature_variant}")
    raw = _raw_matrix(records, feature_names)
    source_weight_power = float(candidate.get("source_weight_power", 1.0))
    source_weights = _source_weights(records, power=source_weight_power)
    raw_means, raw_scales = _weighted_location_scale(raw, source_weights)
    unique_counts = [len(np.unique(raw[:, index])) for index in range(raw.shape[1])]
    nonlinear_indexes = [
        index for index, count in enumerate(unique_counts) if count >= 5
    ]
    if kind in {"anchor_quadratic", "anchor_mechanism"}:
        nonlinear_indexes = [
            list(feature_names).index(name) for name in ANCHOR_SCALE_FEATURE_NAMES
        ]
    expanded = _expand_basis(
        raw,
        kind=kind,
        raw_means=raw_means,
        raw_scales=raw_scales,
        nonlinear_indexes=nonlinear_indexes,
        feature_names=feature_names,
    )
    expanded_means, expanded_scales = _weighted_location_scale(expanded, source_weights)
    standardized = (expanded - expanded_means) / expanded_scales
    design = np.column_stack((np.ones(len(records)), standardized))
    targets, exact = _targets(records)
    censored = ~exact
    alpha = float(candidate["alpha"])
    huber_delta = candidate.get("huber_delta")
    censored_constraint_weight = float(candidate.get("censored_constraint_weight", 1.0))
    if censored_constraint_weight < 1.0:
        raise ValueError("censored constraint weight must be at least one")
    parameters = _solve_weighted_ridge(
        design[exact], targets[exact], source_weights[exact], alpha=alpha
    )
    active_signature: tuple[int, ...] | None = None
    for _iteration in range(100):
        prediction = design @ parameters
        active_censored = censored & (prediction < targets)
        active = exact | active_censored
        residual = prediction[active] - targets[active]
        if huber_delta is None:
            robust_active = np.ones(len(residual), dtype=float)
        else:
            delta = float(huber_delta)
            robust_active = np.minimum(
                1.0, delta / np.maximum(np.abs(residual), 1.0e-12)
            )
        constraint_weights = np.where(censored[active], censored_constraint_weight, 1.0)
        fit_weights = source_weights[active] * robust_active * constraint_weights
        updated = _solve_weighted_ridge(
            design[active], targets[active], fit_weights, alpha=alpha
        )
        signature = tuple(np.flatnonzero(active_censored).tolist())
        converged = (
            active_signature == signature
            and float(np.max(np.abs(updated - parameters))) < 1.0e-9
        )
        parameters = updated
        active_signature = signature
        if converged:
            break
    return {
        "model_family": "physics_anchored_shared_bounded_residual",
        "basis_kind": kind,
        "feature_variant": feature_variant,
        "raw_feature_names": list(feature_names),
        "basis_feature_names": _basis_names(kind, feature_names, nonlinear_indexes),
        "source_or_dataset_id_used_as_feature": False,
        "alpha": alpha,
        "huber_delta": huber_delta,
        "correction_shrinkage": float(candidate["correction_shrinkage"]),
        "source_weight_power": source_weight_power,
        "censored_constraint_weight": censored_constraint_weight,
        "raw_means": raw_means.tolist(),
        "raw_scales": raw_scales.tolist(),
        "nonlinear_indexes": nonlinear_indexes,
        "expanded_means": expanded_means.tolist(),
        "expanded_scales": expanded_scales.tolist(),
        "intercept": float(parameters[0]),
        "coefficients": parameters[1:].tolist(),
        "fit_records": len(records),
        "fit_exact_centres": int(exact.sum()),
        "fit_right_censored": int(censored.sum()),
        "fit_sources": len({str(row["source_id"]) for row in records}),
    }


def _predict_correction(
    records: Sequence[Mapping[str, Any]], model: Mapping[str, Any]
) -> np.ndarray:
    feature_names = [str(name) for name in model["raw_feature_names"]]
    raw = _raw_matrix(records, feature_names)
    expanded = _expand_basis(
        raw,
        kind=str(model["basis_kind"]),
        raw_means=np.asarray(model["raw_means"], dtype=float),
        raw_scales=np.asarray(model["raw_scales"], dtype=float),
        nonlinear_indexes=[int(value) for value in model["nonlinear_indexes"]],
        feature_names=feature_names,
    )
    standardized = (
        expanded - np.asarray(model["expanded_means"], dtype=float)
    ) / np.asarray(model["expanded_scales"], dtype=float)
    correction = float(model["intercept"]) + standardized @ np.asarray(
        model["coefficients"], dtype=float
    )
    return float(model["correction_shrinkage"]) * correction


def _prediction_details(
    train: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    *,
    fold_id: str,
) -> list[dict[str, Any]]:
    model = _fit_model(train, candidate)
    corrections = _predict_correction(test, model)
    details = []
    for row, correction in zip(test, corrections):
        predicted = float(row["reference_bytes"]) * math.exp(float(correction))
        detail = {
            "record_id": str(row["record_id"]),
            "source_id": str(row["source_id"]),
            "origin": str(row["origin"]),
            "role": str(row["role"]),
            "state": str(row["state"]),
            "model_id": str(row["model_id"]),
            "train_type": str(row["train_type"]),
            "gpu_count": int(row["gpu_count"]),
            "zero_stage": int(row["zero_stage"]),
            "gc": bool(row["gc"]),
            "packing": bool(row["packing"]),
            "reference_bytes": float(row["reference_bytes"]),
            "predicted_reserved_bytes": predicted,
            "predicted_reserved_gib": predicted / GIB,
            "fold_id": fold_id,
            "candidate_id": str(candidate["candidate_id"]),
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
    return {
        "exact_centres": len(exact),
        "exact_sources": len(by_source),
        "row_mape": statistics.fmean(errors) if errors else None,
        "source_equal_mape": (
            statistics.fmean(statistics.fmean(values) for values in by_source.values())
            if by_source
            else None
        ),
        "median_ape": base._percentile(errors, 0.5),
        "p90_ape": base._percentile(errors, 0.9),
        "max_ape": max(errors) if errors else None,
        "signed_bias": statistics.fmean(signed) if signed else None,
        "right_censored": len(censored),
        "censor_satisfaction_rate": (
            statistics.fmean(float(row["censor_satisfied"]) for row in censored)
            if censored
            else None
        ),
        "mean_censor_shortfall_fraction": (
            statistics.fmean(
                float(row["censor_shortfall_fraction"]) for row in censored
            )
            if censored
            else None
        ),
    }


def _candidate_grid() -> list[dict[str, Any]]:
    specifications = [
        ("legacy_linear", "legacy", (0.1,), (1.0,), (None,)),
        ("linear", "activation_scale", (0.01, 0.03, 0.1), (0.8, 1.0), (None, 0.20)),
        ("linear", "activation_mechanism", (0.03, 0.1, 0.3), (0.8, 1.0), (None, 0.20)),
        ("linear", "anchor_scale", (0.03, 0.1, 0.3), (0.6, 0.8, 1.0), (None, 0.20)),
        (
            "anchor_quadratic",
            "anchor_scale",
            (0.03, 0.1, 0.3),
            (0.8, 1.0),
            (None, 0.20),
        ),
        ("anchor_mechanism", "anchor_scale", (0.1, 0.3), (0.8, 1.0), (None, 0.20)),
        ("bounded_linear", "anchor_scale", (0.1,), (0.6, 0.8, 1.0), (None, 0.20)),
        ("bounded_quadratic", "anchor_scale", (0.3, 1.0), (0.6, 0.8, 1.0), (0.20,)),
        ("bounded_mechanism", "anchor_scale", (1.0, 3.0), (0.6, 0.8, 1.0), (0.20,)),
    ]
    candidates = []
    for kind, feature_variant, alphas, shrinkages, huber_deltas in specifications:
        for alpha in alphas:
            for shrinkage in shrinkages:
                for huber_delta in huber_deltas:
                    token = (
                        f"{kind}__{feature_variant}__a{alpha:g}__s{shrinkage:g}__"
                        f"h{'squared' if huber_delta is None else format(huber_delta, 'g')}"
                    )
                    candidates.append(
                        {
                            "candidate_id": token,
                            "basis_kind": kind,
                            "feature_variant": feature_variant,
                            "alpha": alpha,
                            "correction_shrinkage": shrinkage,
                            "huber_delta": huber_delta,
                        }
                    )
    return candidates


def _source_folds(
    records: Sequence[Mapping[str, Any]], fold_count: int
) -> list[set[str]]:
    counts = Counter(str(row["source_id"]) for row in records)
    actual = min(int(fold_count), len(counts))
    folds = [set() for _ in range(actual)]
    loads = [0 for _ in range(actual)]
    for source_id, count in sorted(
        counts.items(), key=lambda item: (-item[1], item[0])
    ):
        index = min(range(actual), key=lambda value: (loads[value], value))
        folds[index].add(source_id)
        loads[index] += count
    return folds


def _grouped_predictions(
    records: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    *,
    folds: Sequence[set[str]],
    prefix: str,
) -> list[dict[str, Any]]:
    details = []
    for index, held_sources in enumerate(folds):
        train = [row for row in records if str(row["source_id"]) not in held_sources]
        test = [row for row in records if str(row["source_id"]) in held_sources]
        details.extend(
            _prediction_details(train, test, candidate, fold_id=f"{prefix}{index:02d}")
        )
    return details


def _rank_candidates(
    records: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    folds: Sequence[set[str]],
) -> list[dict[str, Any]]:
    ranked = []
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"candidate {index}/{len(candidates)}: {candidate['candidate_id']}",
            flush=True,
        )
        details = _grouped_predictions(
            records, candidate, folds=folds, prefix="candidate_"
        )
        ranked.append({**candidate, "metrics": _metrics(details)})
    return sorted(
        ranked,
        key=lambda row: (
            float(row["metrics"]["source_equal_mape"]),
            float(row["metrics"]["p90_ape"]),
            str(row["candidate_id"]),
        ),
    )


def _nested_grouped_cv(
    records: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    outer_fold_count: int,
    inner_fold_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = []
    audits = []
    outer_folds = _source_folds(records, outer_fold_count)
    for outer_index, held_sources in enumerate(outer_folds):
        print(f"nested outer fold {outer_index + 1}/{len(outer_folds)}", flush=True)
        train = [row for row in records if str(row["source_id"]) not in held_sources]
        test = [row for row in records if str(row["source_id"]) in held_sources]
        inner_folds = _source_folds(train, inner_fold_count)
        ranked = _rank_candidates(train, candidates, folds=inner_folds)
        selected = ranked[0]
        fold_predictions = _prediction_details(
            train,
            test,
            selected,
            fold_id=f"outer_{outer_index:02d}",
        )
        predictions.extend(fold_predictions)
        audits.append(
            {
                "outer_fold": outer_index,
                "held_out_sources": sorted(held_sources),
                "train_records": len(train),
                "test_records": len(test),
                "selected_candidate_id": selected["candidate_id"],
                "inner_metrics": selected["metrics"],
                "outer_metrics": _metrics(fold_predictions),
            }
        )
    return predictions, audits


def _strict_details(
    train: Sequence[Mapping[str, Any]],
    strict: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    details = _prediction_details(train, strict, candidate, fold_id="strict_existing")
    return details, _metrics(details)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=5)
    args = parser.parse_args()

    print("loading exact centres and right-censored observations", flush=True)
    records, strict, data_audit = _load_records()
    candidates = _candidate_grid()
    comparison_folds = [
        {source_id} for source_id in sorted({str(row["source_id"]) for row in records})
    ]
    print("running fixed leave-one-source-out candidate comparison", flush=True)
    ranked = _rank_candidates(records, candidates, folds=comparison_folds)
    baseline_id = "legacy_linear__legacy__a0.1__s1__hsquared"
    baseline_rank = next(
        row for row in ranked if str(row["candidate_id"]) == baseline_id
    )
    stable_candidates = [
        row
        for row in ranked
        if float(row["metrics"]["max_ape"])
        <= float(baseline_rank["metrics"]["max_ape"])
        and float(row["metrics"]["censor_satisfaction_rate"])
        >= float(baseline_rank["metrics"]["censor_satisfaction_rate"]) - 0.05
    ]
    if not stable_candidates:
        raise RuntimeError("no candidate passed the explicit stability envelope")
    stable_candidate = min(
        stable_candidates,
        key=lambda row: (
            float(row["metrics"]["source_equal_mape"]),
            float(row["metrics"]["p90_ape"]),
            str(row["candidate_id"]),
        ),
    )
    shortlist_ids = {str(row["candidate_id"]) for row in ranked[: min(8, len(ranked))]}
    shortlist_ids.add(baseline_id)
    shortlist = [
        candidate
        for candidate in candidates
        if str(candidate["candidate_id"]) in shortlist_ids
    ]
    print(
        f"running nested grouped selection with {len(shortlist)} candidates",
        flush=True,
    )
    nested_predictions, nested_audit = _nested_grouped_cv(
        records,
        shortlist,
        outer_fold_count=args.outer_folds,
        inner_fold_count=args.inner_folds,
    )
    nested_metrics = _metrics(nested_predictions)

    outer_folds = _source_folds(records, args.outer_folds)
    baseline = next(
        candidate
        for candidate in candidates
        if candidate["candidate_id"] == baseline_id
    )
    baseline_predictions = _grouped_predictions(
        records, baseline, folds=outer_folds, prefix="baseline_outer_"
    )
    baseline_metrics = _metrics(baseline_predictions)

    stable_predictions = _grouped_predictions(
        records, stable_candidate, folds=outer_folds, prefix="stable_outer_"
    )
    stable_metrics = _metrics(stable_predictions)

    best_diagnostic = ranked[0]
    final_model = _fit_model(records, stable_candidate)
    strict_details, strict_metrics = _strict_details(records, strict, stable_candidate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "nested_predictions.jsonl", nested_predictions)
    write_jsonl(args.output_dir / "baseline_predictions.jsonl", baseline_predictions)
    write_jsonl(
        args.output_dir / "stable_candidate_predictions.jsonl", stable_predictions
    )
    write_jsonl(args.output_dir / "strict_predictions.jsonl", strict_details)
    model_artifact = {
        "schema": "sft_h800_memory_center_shadow_model/v1",
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "publishable": False,
        "production_model_mutated": False,
        "candidate": stable_candidate,
        "model": final_model,
    }
    model_artifact["artifact_sha256"] = sha256_json(model_artifact)
    write_json(args.output_dir / "candidate_model.json", model_artifact)

    report = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "publishable": False,
        "production_model_mutated": False,
        "data_audit": data_audit,
        "model_contract": {
            "target": "reserved memory centre",
            "physics_anchor": "analytic reference",
            "one_shared_model": True,
            "source_or_dataset_id_used_as_feature": False,
            "oom_treatment": "right-censored lower-bound constraint",
            "nonlinear_design": (
                "bounded standardized residual basis; optional additive quadratic "
                "and mechanism interactions"
            ),
        },
        "candidate_comparison": {
            "protocol": "fixed leave-one-source-out folds; diagnostic model ranking",
            "ranked": ranked,
            "diagnostic_lowest_mape_candidate": best_diagnostic,
            "stable_selection_policy": (
                "minimise source-equal MAPE subject to max APE no worse than the "
                "legacy baseline and OOM satisfaction no more than 5 percentage "
                "points below the legacy baseline"
            ),
            "stable_candidate": stable_candidate,
        },
        "primary_nested_grouped_cv": {
            "protocol": (
                f"{args.outer_folds}-fold outer source groups; {args.inner_folds}-fold "
                "inner source groups; candidate selection inside each outer fold"
            ),
            "shortlist_candidate_ids": sorted(shortlist_ids),
            "baseline_candidate_id": baseline_id,
            "baseline_metrics": baseline_metrics,
            "stable_candidate_metrics": stable_metrics,
            "selected_metrics": nested_metrics,
            "folds": nested_audit,
            "selected_candidate_counts": dict(
                Counter(str(row["selected_candidate_id"]) for row in nested_audit)
            ),
        },
        "existing_strict_diagnostic": {
            "interpretation": "previously inspected; comparison only, not acceptance",
            "candidate_id": stable_candidate["candidate_id"],
            "metrics": strict_metrics,
        },
        "inputs": {
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "prior_rows": {
                "path": str(base.DEFAULT_PRIOR_ROWS.resolve()),
                "sha256": sha256_file(base.DEFAULT_PRIOR_ROWS),
            },
            "campaign_audit": {
                "path": str(base.DEFAULT_AUDIT.resolve()),
                "sha256": sha256_file(base.DEFAULT_AUDIT),
            },
        },
    }
    report["report_sha256"] = sha256_json(report)
    write_json(args.output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "diagnostic_best": best_diagnostic,
                "stable_candidate": stable_candidate,
                "nested_baseline": baseline_metrics,
                "stable_grouped": stable_metrics,
                "nested_selected": nested_metrics,
                "strict_diagnostic": strict_metrics,
                "publishable": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
