#!/usr/bin/env python3
"""Fit an invariant, rank-first non-Packing throughput challenger.

This versioned challenger intentionally leaves the frozen V5 implementation and
artifact untouched.  It changes two contracts:

1. Non-Packing work mirrors the training collator: samples are truncated by
   ``cutoff_len``, each micro-batch is padded to its own longest sample, and the
   padded length is rounded up to a multiple of eight.  ``cutoff_len`` is not a
   direct regression feature.  Consequently, once cutoff exceeds the profiled
   maximum sample length, work, features and predictions are invariant.
2. Absolute scale and ordering are fitted by separate heads.  The final order
   is determined exclusively by a scenario-balanced pairwise head.  The
   absolute head only supplies the mean log-throughput of a candidate set.

The script is offline and diagnostic.  It launches no GPU work, mutates no
queue, and does not publish a production profile.  The recently consumed V5
dataset-extension outcomes may be used for retrospective/OOF diagnostics and
the final calibration fit, but are never described as a fresh holdout.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from common import ROOT, read_json, sha256_file, sha256_json, write_json
from structured_throughput_modeling import (
    COMPONENT_NAMES,
    PROFILE_SEEDS,
    StaticDatasetProfiles,
    _load_h800,
    _percentile,
    _static_structured_basis,
)
from structured_throughput_modeling import (
    FEATURE_NAMES as V5_FEATURE_NAMES,
)

SCHEMA = "sft_rank_first_throughput_challenger/v1"
IMPLEMENTATION_VERSION = (
    "sft_rank_first_throughput_challenger_impl/"
    "2026-08-05.invariant-nonpacking-two-head-v1"
)
PAD_TO_MULTIPLE_OF = 8
HUBER_DELTA = 0.35
MATERIAL_GAP = 0.05
MATERIAL_PAIR_WEIGHT = 4.0
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
EPSILON = 1.0e-12

REMOVED_CUTOFF_FEATURES = {
    "log2_cutoff_over_512",
    "mean_length_to_cutoff",
}
INVARIANT_EXTRA_FEATURE_NAMES = (
    "log1p_effective_tokens_per_step",
    "log1p_computed_tokens_per_step",
    "log1p_attention_pairs_per_step",
    "log2_expected_padded_batch_length_over_512",
    "truncated_sample_fraction",
    "truncated_token_fraction",
    "log_launch_seconds_at_physical_limit",
    "log_compute_seconds_at_physical_limit",
    "log_kernel_hbm_seconds_at_physical_limit",
    "log_optimizer_hbm_seconds_at_physical_limit",
    "log_communication_seconds_at_physical_limit",
)
FEATURE_NAMES = tuple(
    name for name in V5_FEATURE_NAMES if name not in REMOVED_CUTOFF_FEATURES
) + INVARIANT_EXTRA_FEATURE_NAMES


def _positive(value: Any, default: float = EPSILON) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number) or number <= 0:
        return default
    return number


def _ceil_multiple(value: int, multiple: int = PAD_TO_MULTIPLE_OF) -> int:
    if value <= 0 or multiple <= 0:
        raise ValueError("value and multiple must be positive")
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


class InvariantStaticProfiles:
    """Merged profiles with exact non-Packing dynamic-padding semantics."""

    def __init__(self, profile_dirs: Sequence[Path]) -> None:
        self.rows: dict[str, list[dict[str, Any]]] = {}
        self._sources: dict[str, dict[str, Any]] = {}
        self._cache: dict[tuple[str, int, int, bool], dict[str, Any]] = {}
        for directory in profile_dirs:
            current = StaticDatasetProfiles(Path(directory))
            for dataset_id, rows in current.rows.items():
                if dataset_id in self.rows and self.rows[dataset_id] != rows:
                    raise ValueError(f"Conflicting static profiles for {dataset_id}")
                self.rows[dataset_id] = rows
            self._sources.update(current.source_bindings())
        if not self.rows:
            raise ValueError("No static token profiles were loaded")

    @classmethod
    def from_rows(
        cls,
        rows: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> InvariantStaticProfiles:
        instance = cls.__new__(cls)
        instance.rows = {
            str(dataset_id): [dict(row) for row in current]
            for dataset_id, current in rows.items()
        }
        instance._sources = {}
        instance._cache = {}
        return instance

    def source_bindings(self) -> dict[str, Any]:
        return dict(sorted(self._sources.items()))

    def profile(
        self,
        dataset_id: str,
        *,
        cutoff_len: int,
        physical_mbs: int,
        packing: bool,
    ) -> dict[str, Any]:
        key = (str(dataset_id), int(cutoff_len), int(physical_mbs), bool(packing))
        if key in self._cache:
            return self._cache[key]
        if packing:
            raise ValueError("rank-first challenger v1 supports packing=false only")
        rows = self.rows.get(str(dataset_id))
        if rows is None:
            raise KeyError(f"No static token profile for {dataset_id}")
        cutoff = int(cutoff_len)
        mbs = int(physical_mbs)
        if cutoff <= 0 or mbs <= 0:
            raise ValueError("cutoff_len and physical_mbs must be positive")

        raw_lengths = [int(row.get("total_tokens") or 0) for row in rows]
        if not raw_lengths or min(raw_lengths) <= 0:
            raise ValueError(f"{dataset_id} has nonpositive token length")
        lengths = [min(cutoff, length) for length in raw_lengths]
        label_lengths = [
            min(length, int(row.get("label_tokens") or 0))
            for row, length in zip(rows, lengths)
        ]

        computed_per_sample_runs: list[float] = []
        attention_per_sample_runs: list[float] = []
        for seed in PROFILE_SEEDS:
            shuffled = list(lengths)
            random.Random(seed).shuffle(shuffled)
            computed = 0
            attention = 0
            samples = 0
            for start in range(0, len(shuffled), mbs):
                batch = shuffled[start : start + mbs]
                padded_length = _ceil_multiple(max(batch))
                computed += padded_length * len(batch)
                attention += padded_length * padded_length * len(batch)
                samples += len(batch)
            computed_per_sample_runs.append(computed / samples)
            attention_per_sample_runs.append(attention / samples)

        mean_length = statistics.fmean(lengths)
        computed_per_sample = statistics.fmean(computed_per_sample_runs)
        removed_tokens = sum(raw_lengths) - sum(lengths)
        result = {
            "dataset_id": str(dataset_id),
            "cutoff_len": cutoff,
            "physical_mbs": mbs,
            "packing": False,
            "samples_profiled": len(rows),
            "length": {
                "mean": mean_length,
                "std": statistics.pstdev(lengths),
                "p50": _percentile(lengths, 50),
                "p90": _percentile(lengths, 90),
                "p99": _percentile(lengths, 99),
                "maximum": max(lengths),
                "raw_maximum": max(raw_lengths),
                "mean_squared": statistics.fmean(length * length for length in lengths),
            },
            "label_token_ratio": sum(label_lengths) / _positive(sum(lengths)),
            "mean_turns": statistics.fmean(
                float(row.get("turns") or 0.0) for row in rows
            ),
            "padding_utilization": mean_length / computed_per_sample,
            "packing_fill_ratio": None,
            "mean_samples_per_pack": 1.0,
            "pad_to_multiple_of": PAD_TO_MULTIPLE_OF,
            "expected_padded_batch_length": computed_per_sample,
            "truncated_sample_fraction": sum(
                raw > cutoff for raw in raw_lengths
            )
            / len(raw_lengths),
            "truncated_token_fraction": removed_tokens / _positive(sum(raw_lengths)),
            "work_per_physical_sequence": {
                "effective_tokens": mean_length,
                "computed_tokens": computed_per_sample,
                "computed_attention_token_pairs": statistics.fmean(
                    attention_per_sample_runs
                ),
                "logical_samples": 1.0,
            },
            "source_is_pre_run_static_profile": True,
            "cutoff_invariance_contract": (
                "profile work is unchanged for cutoff_len >= raw_maximum"
            ),
        }
        self._cache[key] = result
        return result


def invariant_profile_signature(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Fields that must remain equal after cutoff ceases to truncate data."""

    return {
        "length": profile["length"],
        "padding_utilization": profile["padding_utilization"],
        "expected_padded_batch_length": profile["expected_padded_batch_length"],
        "truncated_sample_fraction": profile["truncated_sample_fraction"],
        "truncated_token_fraction": profile["truncated_token_fraction"],
        "work_per_physical_sequence": profile["work_per_physical_sequence"],
    }


def _invariant_basis(
    record: Mapping[str, Any],
    profiles: InvariantStaticProfiles,
    *,
    hardware_memory_bytes: float,
) -> dict[str, Any]:
    basis = _static_structured_basis(
        record,
        profiles,  # type: ignore[arg-type]
        hardware_memory_bytes=hardware_memory_bytes,
    )
    values = {
        name: float(value)
        for name, value in basis["feature_values"].items()
        if name not in REMOVED_CUTOFF_FEATURES
    }
    work = basis["work_per_step"]
    profile = basis["work_evidence"]["profile"]
    values.update(
        {
            "log1p_effective_tokens_per_step": math.log1p(
                float(work["effective_tokens"])
            ),
            "log1p_computed_tokens_per_step": math.log1p(
                float(work["computed_tokens"])
            ),
            "log1p_attention_pairs_per_step": math.log1p(
                float(work["computed_attention_token_pairs"])
            ),
            "log2_expected_padded_batch_length_over_512": math.log2(
                _positive(profile["expected_padded_batch_length"]) / 512.0
            ),
            "truncated_sample_fraction": float(
                profile["truncated_sample_fraction"]
            ),
            "truncated_token_fraction": float(
                profile["truncated_token_fraction"]
            ),
        }
    )
    components = basis["component_seconds_at_physical_limits"]
    for component, feature in zip(
        COMPONENT_NAMES,
        INVARIANT_EXTRA_FEATURE_NAMES[-5:],
    ):
        values[feature] = math.log(_positive(components[component]))
    features = np.asarray([values[name] for name in FEATURE_NAMES], dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("Invariant challenger features contain NaN/Inf")
    return {**basis, "invariant_feature_values": values, "invariant_features": features}


def _candidate_from_pooled(
    pooled: Mapping[str, Any],
    profiles: InvariantStaticProfiles,
    *,
    hardware_memory_bytes: float,
    source_role: str,
) -> dict[str, Any]:
    basis = _invariant_basis(
        pooled["record"],
        profiles,
        hardware_memory_bytes=hardware_memory_bytes,
    )
    return {
        "scenario_id": str(pooled["scenario_id"]),
        "record": pooled["record"],
        "candidate_key": pooled.get("candidate_key"),
        "job_id": None,
        "source_role": source_role,
        "historical": bool(pooled.get("historical")),
        "features": basis["invariant_features"],
        "basis": basis,
        "observed_log_throughput": float(pooled["observed_log_throughput"]),
    }


def _extension_record(
    row: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    configuration = prediction["configuration"]
    explanation = prediction["explanation"]
    hardware = explanation["hardware"]
    return {
        "scenario": {
            "model_id": configuration["model_id"],
            "dataset_id": configuration["dataset_id"],
            "train_type": configuration["training_mode"],
            "gpu_count": configuration["gpu_count"],
            "physical_mbs": configuration["physical_mbs"],
            "target_gbs": configuration["target_gbs"],
            "cutoff_len": configuration["cutoff_len"],
        },
        "selector": {
            "training_mode": configuration["training_mode"],
            "zero_stage": configuration["zero_stage"],
            "gradient_checkpointing": configuration["gradient_checkpointing"],
            "packing": configuration["packing"],
            "offload": configuration["offload"],
            "kernel_path": configuration["kernel_path"],
            "dtype": configuration["dtype"],
        },
        "model_basis": explanation["model_geometry"],
        "performance": {
            "gradient_accumulation_steps": configuration[
                "gradient_accumulation_steps"
            ],
            "physical_priors": {
                "dense_bf16_peak_flops_per_gpu": hardware[
                    "dense_bf16_peak_flops_per_gpu"
                ],
                "hbm_bandwidth_bytes_per_second": hardware[
                    "hbm_bandwidth_bytes_per_second"
                ],
                "intra_node_bandwidth_bytes_per_second": hardware[
                    "intra_node_bandwidth_bytes_per_second"
                ],
                "collective_latency_seconds": hardware[
                    "collective_latency_seconds"
                ],
            },
        },
    }


def _load_extension_candidates(
    evaluation_path: Path,
    predictions_path: Path,
    profiles: InvariantStaticProfiles,
    *,
    hardware_memory_bytes: float,
) -> list[dict[str, Any]]:
    evaluation = read_json(evaluation_path)
    frozen = read_json(predictions_path)
    predictions = frozen["predictions"]
    candidates: list[dict[str, Any]] = []
    for row in evaluation["rows"]:
        if row["state"] != "success_authoritative":
            continue
        job_id = str(row["job_id"])
        prediction = predictions[job_id]
        record = _extension_record(row, prediction)
        basis = _invariant_basis(
            record,
            profiles,
            hardware_memory_bytes=hardware_memory_bytes,
        )
        candidates.append(
            {
                "scenario_id": str(row["scenario_id"]),
                "record": record,
                "candidate_key": [
                    int(row["mbs"]),
                    int(row["zero_stage"]),
                    bool(row["gradient_checkpointing"]),
                ],
                "job_id": job_id,
                "source_role": str(row["source"]),
                "historical": False,
                "features": basis["invariant_features"],
                "basis": basis,
                "observed_log_throughput": math.log(
                    _positive(row["observed"]["effective_tokens_per_second"])
                ),
            }
        )
    return candidates


def _scenario_weights(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts = Counter(str(candidate["scenario_id"]) for candidate in candidates)
    return np.asarray(
        [1.0 / counts[str(candidate["scenario_id"])] for candidate in candidates],
        dtype=float,
    )


def _standardization(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.vstack([np.asarray(candidate["features"], dtype=float) for candidate in candidates])
    weights = _scenario_weights(candidates)
    means = np.average(raw, axis=0, weights=weights)
    scales = np.sqrt(np.average((raw - means) ** 2, axis=0, weights=weights))
    scales[scales < 1.0e-9] = 1.0
    return raw, means, scales


def _irls_ridge(
    design: np.ndarray,
    target: np.ndarray,
    base_weights: np.ndarray,
    *,
    alpha: float,
    penalize_intercept: bool,
) -> tuple[np.ndarray, int]:
    coefficients = np.zeros(design.shape[1], dtype=float)
    regularizer = np.eye(design.shape[1], dtype=float) * float(alpha)
    if not penalize_intercept:
        regularizer[0, 0] = 0.0
    iterations = 0
    for iterations in range(1, 51):
        residual = target - design @ coefficients
        robust = np.ones_like(residual)
        outside = np.abs(residual) > HUBER_DELTA
        robust[outside] = HUBER_DELTA / np.abs(residual[outside])
        weights = base_weights * robust
        normal = design.T @ (weights[:, None] * design) + regularizer
        rhs = design.T @ (weights * target)
        updated = np.linalg.pinv(normal) @ rhs
        if np.max(np.abs(updated - coefficients)) < 1.0e-9:
            coefficients = updated
            break
        coefficients = updated
    return coefficients, iterations


def _fit_absolute_head(
    candidates: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
) -> dict[str, Any]:
    raw, means, scales = _standardization(candidates)
    standardized = (raw - means) / scales
    design = np.column_stack([np.ones(len(candidates)), standardized])
    target = np.asarray(
        [float(candidate["observed_log_throughput"]) for candidate in candidates],
        dtype=float,
    )
    coefficients, iterations = _irls_ridge(
        design,
        target,
        _scenario_weights(candidates),
        alpha=alpha,
        penalize_intercept=False,
    )
    return {
        "model_family": "scenario_balanced_absolute_log_throughput_ridge",
        "role": "candidate_set_absolute_center_only",
        "feature_names": list(FEATURE_NAMES),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "alpha": float(alpha),
        "irls_iterations": iterations,
        "fit_candidates": len(candidates),
        "fit_scenarios": len({str(row["scenario_id"]) for row in candidates}),
    }


def _fit_rank_head(
    candidates: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
) -> dict[str, Any]:
    raw, means, scales = _standardization(candidates)
    standardized = (raw - means) / scales
    by_scenario: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_scenario[str(candidate["scenario_id"])].append(index)

    rows: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    material_pairs = 0
    for indexes in by_scenario.values():
        pairs = list(itertools.combinations(indexes, 2))
        if not pairs:
            continue
        for left, right in pairs:
            delta = (
                float(candidates[left]["observed_log_throughput"])
                - float(candidates[right]["observed_log_throughput"])
            )
            relative_gap = 1.0 - math.exp(-abs(delta))
            material = relative_gap >= MATERIAL_GAP
            material_pairs += int(material)
            rows.append(standardized[left] - standardized[right])
            targets.append(delta)
            weights.append(
                (MATERIAL_PAIR_WEIGHT if material else 1.0) / len(pairs)
            )
    if not rows:
        raise ValueError("Rank head has no within-scenario candidate pairs")
    design = np.vstack(rows)
    target = np.asarray(targets, dtype=float)
    base = np.asarray(weights, dtype=float)
    coefficients, iterations = _irls_ridge(
        design,
        target,
        base,
        alpha=alpha,
        penalize_intercept=True,
    )
    return {
        "model_family": "scenario_balanced_material_weighted_pairwise_ridge",
        "role": "exclusive_candidate_ordering_head",
        "formula": "rank_score=standardized_invariant_features@gamma",
        "feature_names": list(FEATURE_NAMES),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "coefficients": coefficients.tolist(),
        "alpha": float(alpha),
        "huber_delta_log_throughput": HUBER_DELTA,
        "material_gap": MATERIAL_GAP,
        "material_pair_weight": MATERIAL_PAIR_WEIGHT,
        "irls_iterations": iterations,
        "fit_candidates": len(candidates),
        "fit_scenarios": len(by_scenario),
        "fit_pairs": len(rows),
        "fit_material_pairs": material_pairs,
    }


def _linear_score(candidate: Mapping[str, Any], model: Mapping[str, Any]) -> float:
    features = np.asarray(candidate["features"], dtype=float)
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    score = float(((features - means) / scales) @ np.asarray(model["coefficients"]))
    return score + float(model.get("intercept") or 0.0)


def set_aware_log_predictions(
    absolute_logs: Sequence[float],
    rank_scores: Sequence[float],
) -> list[float]:
    """Preserve the absolute center while making rank score authoritative."""

    if len(absolute_logs) != len(rank_scores) or not absolute_logs:
        raise ValueError("absolute_logs and rank_scores must have equal nonzero length")
    absolute_center = statistics.fmean(float(value) for value in absolute_logs)
    rank_center = statistics.fmean(float(value) for value in rank_scores)
    return [absolute_center + float(score) - rank_center for score in rank_scores]


def _prediction_entries(
    candidates: Sequence[Mapping[str, Any]],
    absolute_head: Mapping[str, Any],
    rank_head: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], float, float, float]]:
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_scenario[str(candidate["scenario_id"])].append(candidate)
    entries: list[tuple[Mapping[str, Any], float, float, float]] = []
    for current in by_scenario.values():
        absolute = [_linear_score(candidate, absolute_head) for candidate in current]
        rank = [_linear_score(candidate, rank_head) for candidate in current]
        final = set_aware_log_predictions(absolute, rank)
        entries.extend(zip(current, final, absolute, rank))
    return entries


def _mechanism(candidate: Mapping[str, Any]) -> str:
    selector = candidate["record"].get("selector") or {}
    return "z{zero}_gc{gc}".format(
        zero=int(selector.get("zero_stage") or 0),
        gc=int(bool(selector.get("gradient_checkpointing"))),
    )


def _configuration(candidate: Mapping[str, Any]) -> str:
    scenario = candidate["record"].get("scenario") or {}
    selector = candidate["record"].get("selector") or {}
    return "MBS{mbs}/Z{zero}/GC-{gc}".format(
        mbs=int(scenario.get("physical_mbs") or 0),
        zero=int(selector.get("zero_stage") or 0),
        gc="on" if selector.get("gradient_checkpointing") else "off",
    )


def _evaluate_entries(
    entries: Sequence[tuple[Mapping[str, Any], float, float, float]],
) -> dict[str, Any]:
    by_scenario: dict[str, list[tuple[Mapping[str, Any], float, float, float]]] = defaultdict(list)
    for entry in entries:
        by_scenario[str(entry[0]["scenario_id"])].append(entry)

    final_apes: list[float] = []
    absolute_apes: list[float] = []
    pair_correct = pair_count = 0
    cross_correct = cross_count = 0
    material_cross_correct = material_cross_count = 0
    mechanism_pairs: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "comparisons": 0,
            "correct": 0,
            "material_comparisons": 0,
            "material_correct": 0,
        }
    )
    wrong_cross_pairs: list[dict[str, Any]] = []
    scenario_cross_accuracies: list[float] = []
    scenario_material_accuracies: list[float] = []
    regrets: list[float] = []
    exact_top1: list[float] = []
    details: list[dict[str, Any]] = []
    for scenario_id, rows in sorted(by_scenario.items()):
        current_cross_correct = current_cross_count = 0
        current_material_correct = current_material_count = 0
        for candidate, final_log, absolute_log, _ in rows:
            observed = float(candidate["observed_log_throughput"])
            final_apes.append(abs(math.exp(final_log - observed) - 1.0))
            absolute_apes.append(abs(math.exp(absolute_log - observed) - 1.0))
        for left, right in itertools.combinations(rows, 2):
            observed_delta = (
                float(left[0]["observed_log_throughput"])
                - float(right[0]["observed_log_throughput"])
            )
            if abs(observed_delta) <= 1.0e-12:
                continue
            predicted_delta = float(left[1]) - float(right[1])
            correct = (observed_delta > 0.0) == (predicted_delta > 0.0)
            pair_correct += int(correct)
            pair_count += 1
            if _mechanism(left[0]) != _mechanism(right[0]):
                mechanism_key = tuple(
                    sorted((_mechanism(left[0]), _mechanism(right[0])))
                )
                mechanism_pairs[mechanism_key]["comparisons"] += 1
                mechanism_pairs[mechanism_key]["correct"] += int(correct)
                cross_correct += int(correct)
                cross_count += 1
                current_cross_correct += int(correct)
                current_cross_count += 1
                relative_gap = 1.0 - math.exp(-abs(observed_delta))
                if relative_gap >= MATERIAL_GAP:
                    mechanism_pairs[mechanism_key]["material_comparisons"] += 1
                    mechanism_pairs[mechanism_key]["material_correct"] += int(
                        correct
                    )
                    material_cross_correct += int(correct)
                    material_cross_count += 1
                    current_material_correct += int(correct)
                    current_material_count += 1
                if not correct:
                    wrong_cross_pairs.append(
                        {
                            "scenario_id": scenario_id,
                            "left_configuration": _configuration(left[0]),
                            "right_configuration": _configuration(right[0]),
                            "left_observed_effective_tokens_per_second": math.exp(
                                float(left[0]["observed_log_throughput"])
                            ),
                            "right_observed_effective_tokens_per_second": math.exp(
                                float(right[0]["observed_log_throughput"])
                            ),
                            "observed_relative_gap": relative_gap,
                            "predicted_log_difference": predicted_delta,
                        }
                    )
        if current_cross_count:
            scenario_cross_accuracies.append(
                current_cross_correct / current_cross_count
            )
        if current_material_count:
            scenario_material_accuracies.append(
                current_material_correct / current_material_count
            )
        if len(rows) >= 2:
            selected = max(rows, key=lambda item: float(item[1]))
            oracle = max(
                rows,
                key=lambda item: float(item[0]["observed_log_throughput"]),
            )
            regret = max(
                0.0,
                1.0
                - math.exp(
                    float(selected[0]["observed_log_throughput"])
                    - float(oracle[0]["observed_log_throughput"])
                ),
            )
            regrets.append(regret)
            exact_top1.append(float(selected[0] is oracle[0]))
            details.append(
                {
                    "scenario_id": scenario_id,
                    "candidates": len(rows),
                    "selected_job_id": selected[0].get("job_id"),
                    "oracle_job_id": oracle[0].get("job_id"),
                    "top1_regret": regret,
                    "cross_mechanism_pairwise_accuracy": (
                        current_cross_correct / current_cross_count
                        if current_cross_count
                        else None
                    ),
                    "material_cross_mechanism_pairwise_accuracy": (
                        current_material_correct / current_material_count
                        if current_material_count
                        else None
                    ),
                }
            )

    return {
        "candidate_rows": len(entries),
        "scenario_rows": len(by_scenario),
        "all_pairwise_comparisons": pair_count,
        "all_pairwise_accuracy": pair_correct / pair_count if pair_count else None,
        "cross_mechanism_pairwise_comparisons": cross_count,
        "cross_mechanism_pairwise_accuracy": (
            cross_correct / cross_count if cross_count else None
        ),
        "scenario_equal_cross_mechanism_pairwise_accuracy": (
            statistics.fmean(scenario_cross_accuracies)
            if scenario_cross_accuracies
            else None
        ),
        "material_gap": MATERIAL_GAP,
        "material_cross_mechanism_pairwise_comparisons": material_cross_count,
        "material_cross_mechanism_pairwise_accuracy": (
            material_cross_correct / material_cross_count
            if material_cross_count
            else None
        ),
        "scenario_equal_material_cross_mechanism_pairwise_accuracy": (
            statistics.fmean(scenario_material_accuracies)
            if scenario_material_accuracies
            else None
        ),
        "mean_top1_regret": statistics.fmean(regrets) if regrets else None,
        "worst_top1_regret": max(regrets) if regrets else None,
        "exact_top1_fraction": statistics.fmean(exact_top1) if exact_top1 else None,
        "set_aware_absolute_mape": statistics.fmean(final_apes) if final_apes else None,
        "absolute_head_mape": statistics.fmean(absolute_apes) if absolute_apes else None,
        "mechanism_pair_breakdown": {
            "__vs__".join(key): {
                **counts,
                "accuracy": counts["correct"] / counts["comparisons"],
                "material_accuracy": (
                    counts["material_correct"] / counts["material_comparisons"]
                    if counts["material_comparisons"]
                    else None
                ),
            }
            for key, counts in sorted(mechanism_pairs.items())
        },
        "wrong_cross_mechanism_pairs": sorted(
            wrong_cross_pairs,
            key=lambda row: float(row["observed_relative_gap"]),
            reverse=True,
        ),
        "details": details,
    }


def _scenario_folds(
    candidates: Sequence[Mapping[str, Any]],
    fold_count: int = 5,
) -> list[set[str]]:
    counts = Counter(str(candidate["scenario_id"]) for candidate in candidates)
    fold_count = max(2, min(int(fold_count), len(counts)))
    folds = [set() for _ in range(fold_count)]
    loads = [0] * fold_count
    for scenario_id, count in sorted(
        counts.items(),
        key=lambda item: (-(item[1] * (item[1] - 1) // 2), item[0]),
    ):
        index = min(range(fold_count), key=lambda current: (loads[current], current))
        folds[index].add(scenario_id)
        loads[index] += count * (count - 1) // 2
    return folds


def _cross_validated_entries(
    candidates: Sequence[Mapping[str, Any]],
    *,
    absolute_alpha: float,
    rank_alpha: float,
    folds: Sequence[set[str]],
) -> list[tuple[Mapping[str, Any], float, float, float]]:
    entries: list[tuple[Mapping[str, Any], float, float, float]] = []
    for held in folds:
        train = [row for row in candidates if str(row["scenario_id"]) not in held]
        test = [row for row in candidates if str(row["scenario_id"]) in held]
        absolute = _fit_absolute_head(train, alpha=absolute_alpha)
        rank = _fit_rank_head(train, alpha=rank_alpha)
        entries.extend(_prediction_entries(test, absolute, rank))
    return entries


def _select_absolute_alpha(
    candidates: Sequence[Mapping[str, Any]],
    folds: Sequence[set[str]],
) -> dict[str, Any]:
    evaluated = []
    dummy_rank_alpha = 1.0
    for alpha in ALPHA_GRID:
        metrics = _evaluate_entries(
            _cross_validated_entries(
                candidates,
                absolute_alpha=alpha,
                rank_alpha=dummy_rank_alpha,
                folds=folds,
            )
        )
        evaluated.append({"alpha": alpha, "metrics": metrics})
    selected = min(
        evaluated,
        key=lambda item: (
            float(item["metrics"]["absolute_head_mape"]),
            float(item["alpha"]),
        ),
    )
    return {"selection_primary": "minimum_absolute_head_mape", "selected": selected, "candidates": evaluated}


def _rank_selection_key(item: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = item["metrics"]
    return (
        -float(metrics.get("scenario_equal_material_cross_mechanism_pairwise_accuracy") or 0.0),
        -float(metrics.get("material_cross_mechanism_pairwise_accuracy") or 0.0),
        float(metrics.get("mean_top1_regret") or 0.0),
        -float(metrics.get("scenario_equal_cross_mechanism_pairwise_accuracy") or 0.0),
        -float(metrics.get("cross_mechanism_pairwise_accuracy") or 0.0),
        float(item["alpha"]),
    )


def _select_rank_alpha(
    candidates: Sequence[Mapping[str, Any]],
    folds: Sequence[set[str]],
    *,
    absolute_alpha: float,
) -> dict[str, Any]:
    evaluated = []
    for alpha in ALPHA_GRID:
        metrics = _evaluate_entries(
            _cross_validated_entries(
                candidates,
                absolute_alpha=absolute_alpha,
                rank_alpha=alpha,
                folds=folds,
            )
        )
        evaluated.append({"alpha": alpha, "metrics": metrics})
    selected = min(evaluated, key=_rank_selection_key)
    return {
        "selection_primary": (
            "maximize scenario-equal material cross-mechanism pairwise accuracy; "
            "pooled material accuracy and top1 regret are subsequent tie-breakers"
        ),
        "selected": selected,
        "candidates": evaluated,
    }


def _extension_oof_entries(
    base: Sequence[Mapping[str, Any]],
    extension: Sequence[Mapping[str, Any]],
    *,
    absolute_alpha: float,
    rank_alpha: float,
) -> list[tuple[Mapping[str, Any], float, float, float]]:
    entries: list[tuple[Mapping[str, Any], float, float, float]] = []
    scenario_ids = sorted({str(row["scenario_id"]) for row in extension})
    for held in scenario_ids:
        train = [*base, *(row for row in extension if str(row["scenario_id"]) != held)]
        test = [row for row in extension if str(row["scenario_id"]) == held]
        entries.extend(
            _prediction_entries(
                test,
                _fit_absolute_head(train, alpha=absolute_alpha),
                _fit_rank_head(train, alpha=rank_alpha),
            )
        )
    return entries


def _counterfactual_cutoff_audit(
    template: Mapping[str, Any],
    profiles: InvariantStaticProfiles,
    absolute_head: Mapping[str, Any],
    rank_head: Mapping[str, Any],
    *,
    hardware_memory_bytes: float,
) -> dict[str, Any]:
    dataset_id = str(template["record"]["scenario"]["dataset_id"])
    raw_maximum = max(
        int(row["total_tokens"]) for row in profiles.rows[dataset_id]
    )
    cutoffs = [value for value in (512, 1024, 2048, 4096) if value >= raw_maximum]
    candidates = []
    signatures = []
    for cutoff in cutoffs:
        record = json.loads(json.dumps(template["record"]))
        record["scenario"]["cutoff_len"] = cutoff
        basis = _invariant_basis(
            record,
            profiles,
            hardware_memory_bytes=hardware_memory_bytes,
        )
        candidates.append(
            {
                **template,
                "record": record,
                "features": basis["invariant_features"],
                "basis": basis,
            }
        )
        signatures.append(
            invariant_profile_signature(basis["work_evidence"]["profile"])
        )
    absolute = [_linear_score(candidate, absolute_head) for candidate in candidates]
    rank = [_linear_score(candidate, rank_head) for candidate in candidates]
    return {
        "dataset_id": dataset_id,
        "profile_raw_maximum": raw_maximum,
        "cutoffs": cutoffs,
        "profile_signatures_identical": all(
            signature == signatures[0] for signature in signatures[1:]
        ),
        "feature_vectors_identical": all(
            np.array_equal(candidates[0]["features"], candidate["features"])
            for candidate in candidates[1:]
        ),
        "absolute_log_predictions": absolute,
        "rank_scores": rank,
        "absolute_prediction_range": max(absolute) - min(absolute),
        "rank_score_range": max(rank) - min(rank),
        "passes": (
            all(signature == signatures[0] for signature in signatures[1:])
            and all(
                np.array_equal(candidates[0]["features"], candidate["features"])
                for candidate in candidates[1:]
            )
            and max(absolute) - min(absolute) <= 1.0e-12
            and max(rank) - min(rank) <= 1.0e-12
        ),
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    extension = report["evaluation"]["extension"]
    base_only = extension["challenger_base_only"]
    oof = extension["challenger_leave_one_extension_scenario_out"]
    final = extension["challenger_final_fit_in_sample"]
    v5 = extension["v5_frozen"]

    def pct(value: Any) -> str:
        return "—" if value is None else f"{100.0 * float(value):.2f}%"

    return "\n".join(
        [
            "# Rank-first 非 Packing 吞吐 Challenger v1",
            "",
            "> 该报告是离线 challenger 诊断，不是新鲜 holdout 验收，也不自动替换 V5。",
            "",
            "## 实现合同",
            "",
            "非 Packing 的每个样本先截断：",
            "",
            "$$",
            "l_i=\\min(t_i,C)",
            "$$",
            "",
            "每个微批次按最长样本动态 padding，并补齐到 8 的倍数：",
            "",
            "$$",
            "L_b^{\\mathrm{pad}}=8\\left\\lceil\\frac{\\max_{i\\in b}l_i}{8}\\right\\rceil",
            "$$",
            "",
            "`cutoff_len` 不再是直接回归特征。最终候选顺序只由 pairwise ranking head 决定；absolute head 只提供候选集合的平均 log-throughput。",
            "",
            "## 最新扩展实验回放",
            "",
            "| 模型 | 跨机制 Pairwise | 重要差距跨机制 Pairwise | Top-1 平均遗憾 | 绝对 MAPE |",
            "|---|---:|---:|---:|---:|",
            f"| V5 frozen | {pct(v5['cross_mechanism_pairwise_accuracy'])} | — | {pct(v5['mean_top1_regret'])} | {pct(v5['absolute_mape'])} |",
            f"| challenger：仅旧训练集拟合 | {pct(base_only['cross_mechanism_pairwise_accuracy'])} | {pct(base_only['material_cross_mechanism_pairwise_accuracy'])} | {pct(base_only['mean_top1_regret'])} | {pct(base_only['set_aware_absolute_mape'])} |",
            f"| challenger：逐数据场景 OOF | {pct(oof['cross_mechanism_pairwise_accuracy'])} | {pct(oof['material_cross_mechanism_pairwise_accuracy'])} | {pct(oof['mean_top1_regret'])} | {pct(oof['set_aware_absolute_mape'])} |",
            f"| challenger：全部扩展结果拟合后回放 | {pct(final['cross_mechanism_pairwise_accuracy'])} | {pct(final['material_cross_mechanism_pairwise_accuracy'])} | {pct(final['mean_top1_regret'])} | {pct(final['set_aware_absolute_mape'])} |",
            "",
            "其中 OOF 只表示逐扩展数据场景留出的回放诊断；由于模型结构是在查看 V5 错误后设计的，它不是可用于发布的全新前瞻验收。",
            "",
            "## Cutoff 不变量",
            "",
            f"- profile 最大长度：{report['cutoff_invariance_audit']['profile_raw_maximum']} token",
            f"- 检查 cutoff：{report['cutoff_invariance_audit']['cutoffs']}",
            f"- Work、特征和预测完全一致：`{str(report['cutoff_invariance_audit']['passes']).lower()}`",
            "",
            "## 当前结论",
            "",
            "该 challenger 已实现指定的长度语义和排序优先结构。最终是否可替换 V5，仍需要依据本报告的回放结果和后续冻结的新鲜场景验收决定。",
            "",
        ]
    )


def build_report(
    *,
    profile_dirs: Sequence[Path],
    h800_observations_path: Path,
    h800_theory_basis_path: Path,
    h800_inventory_path: Path,
    h800_hardware_path: Path,
    h800_runtime_root: Path,
    extension_evaluation_path: Path,
    extension_predictions_path: Path,
    v5_artifact_path: Path,
) -> dict[str, Any]:
    profiles = InvariantStaticProfiles(profile_dirs)
    hardware = read_json(h800_hardware_path)
    hardware_memory_bytes = float(hardware["memory_bytes_reported_by_torch"])
    h800 = _load_h800(
        observation_path=h800_observations_path,
        theory_basis_path=h800_theory_basis_path,
        inventory_path=h800_inventory_path,
        hardware_path=h800_hardware_path,
        runtime_root=h800_runtime_root,
    )
    base = [
        _candidate_from_pooled(
            candidate,
            profiles,
            hardware_memory_bytes=hardware_memory_bytes,
            source_role="h800_strict_fit",
        )
        for candidate in h800["strict_fit_candidates"]
    ]
    holdout = [
        _candidate_from_pooled(
            candidate,
            profiles,
            hardware_memory_bytes=hardware_memory_bytes,
            source_role="h800_consumed_native_holdout",
        )
        for candidate in h800["holdout_candidates"]
    ]
    extension = _load_extension_candidates(
        extension_evaluation_path,
        extension_predictions_path,
        profiles,
        hardware_memory_bytes=hardware_memory_bytes,
    )

    folds = _scenario_folds(base)
    absolute_selection = _select_absolute_alpha(base, folds)
    absolute_alpha = float(absolute_selection["selected"]["alpha"])
    rank_selection = _select_rank_alpha(
        base,
        folds,
        absolute_alpha=absolute_alpha,
    )
    rank_alpha = float(rank_selection["selected"]["alpha"])

    base_absolute = _fit_absolute_head(base, alpha=absolute_alpha)
    base_rank = _fit_rank_head(base, alpha=rank_alpha)
    final_fit_population = [*base, *extension]
    final_absolute = _fit_absolute_head(final_fit_population, alpha=absolute_alpha)
    final_rank = _fit_rank_head(final_fit_population, alpha=rank_alpha)

    extension_evaluation = read_json(extension_evaluation_path)
    v5_metrics = extension_evaluation["metrics"]
    cutoff_audit = _counterfactual_cutoff_audit(
        extension[0],
        profiles,
        final_absolute,
        final_rank,
        hardware_memory_bytes=hardware_memory_bytes,
    )
    if not cutoff_audit["passes"]:
        raise ValueError("cutoff invariance audit failed")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "rank_first_challenger_fitted_and_replayed",
        "analysis_only": True,
        "publishable": False,
        "production_profile_generated": False,
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "model_contract": {
            "scope": "text SFT, packing=false, H800 diagnostic",
            "target": "effective_tokens_per_second",
            "feature_dimension": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "removed_direct_cutoff_features": sorted(REMOVED_CUTOFF_FEATURES),
            "cutoff_is_only_a_truncation_control": True,
            "dynamic_padding_to_multiple_of": PAD_TO_MULTIPLE_OF,
            "two_independent_heads": True,
            "ranking_head_is_authoritative_for_order": True,
            "absolute_head_role": "candidate_set_log_throughput_center_only",
            "set_aware_formula": (
                "log_T_i=mean_j(log_T_abs_j)+rank_score_i-mean_j(rank_score_j)"
            ),
        },
        "selection": {
            "base_training_scenario_folds": len(folds),
            "absolute_head": absolute_selection,
            "rank_head": rank_selection,
            "holdout_used_for_hyperparameter_selection": False,
            "ranking_is_primary": True,
        },
        "models": {
            "base_only_absolute_head": base_absolute,
            "base_only_rank_head": base_rank,
            "final_absolute_head": final_absolute,
            "final_rank_head": final_rank,
        },
        "data": {
            "base_fit_candidates": len(base),
            "base_fit_scenarios": len({str(row["scenario_id"]) for row in base}),
            "consumed_native_holdout_candidates": len(holdout),
            "extension_success_candidates": len(extension),
            "extension_scenarios": len({str(row["scenario_id"]) for row in extension}),
            "extension_results_are_fresh_holdout": False,
            "extension_results_used_in_final_fit": True,
        },
        "evaluation": {
            "base_cross_validation": rank_selection["selected"]["metrics"],
            "consumed_native_holdout_diagnostic": _evaluate_entries(
                _prediction_entries(holdout, base_absolute, base_rank)
            ),
            "extension": {
                "v5_frozen": {
                    "cross_mechanism_pairwise_accuracy": v5_metrics["ranking"][
                        "pooled_cross_mechanism_pairwise_accuracy"
                    ],
                    "mean_top1_regret": v5_metrics["ranking"]["v5_mean_top1_regret"],
                    "absolute_mape": v5_metrics["absolute"]["throughput_mape"],
                },
                "challenger_base_only": _evaluate_entries(
                    _prediction_entries(extension, base_absolute, base_rank)
                ),
                "challenger_leave_one_extension_scenario_out": _evaluate_entries(
                    _extension_oof_entries(
                        base,
                        extension,
                        absolute_alpha=absolute_alpha,
                        rank_alpha=rank_alpha,
                    )
                ),
                "challenger_final_fit_in_sample": _evaluate_entries(
                    _prediction_entries(extension, final_absolute, final_rank)
                ),
            },
        },
        "cutoff_invariance_audit": cutoff_audit,
        "source_bindings": {
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "profiles": profiles.source_bindings(),
            "h800_observations": {
                "path": str(h800_observations_path.resolve()),
                "sha256": sha256_file(h800_observations_path),
            },
            "extension_evaluation": {
                "path": str(extension_evaluation_path.resolve()),
                "sha256": sha256_file(extension_evaluation_path),
            },
            "extension_predictions": {
                "path": str(extension_predictions_path.resolve()),
                "sha256": sha256_file(extension_predictions_path),
            },
            "frozen_v5": {
                "path": str(v5_artifact_path.resolve()),
                "sha256": sha256_file(v5_artifact_path),
            },
        },
        "limitations": [
            "extension outcomes are consumed and not a fresh release holdout",
            "packing and VL are outside challenger v1 scope",
            "card generalization is intentionally not evaluated",
            "the challenger does not automatically replace the frozen V5 artifact",
        ],
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-profile-dir",
        type=Path,
        default=ROOT / "artifacts" / "dataset_profiles",
    )
    parser.add_argument(
        "--additional-dataset-profile-dir",
        type=Path,
        default=ROOT / "artifacts" / "h800_lora_safety_stage2_v1" / "profiles",
    )
    parser.add_argument(
        "--h800-observations",
        type=Path,
        default=ROOT / "artifacts" / "canonical_h800_observations.jsonl",
    )
    parser.add_argument(
        "--h800-theory-basis",
        type=Path,
        default=ROOT / "artifacts" / "h800_theory_basis.json",
    )
    parser.add_argument(
        "--h800-model-inventory",
        type=Path,
        default=ROOT / "artifacts" / "model_inventory.json",
    )
    parser.add_argument(
        "--h800-hardware",
        type=Path,
        default=ROOT / "config" / "hardware.json",
    )
    parser.add_argument(
        "--h800-runtime-root",
        type=Path,
        default=ROOT / "runtime",
    )
    parser.add_argument(
        "--extension-evaluation",
        type=Path,
        default=ROOT / "artifacts" / "v5_dataset_candidate_extension_h800_20260805.json",
    )
    parser.add_argument(
        "--extension-predictions",
        type=Path,
        default=ROOT / "artifacts" / "h800_v5_dataset_candidate_extension_frozen_predictions_v1.json",
    )
    parser.add_argument(
        "--v5-artifact",
        type=Path,
        default=ROOT / "artifacts" / "structured_throughput_modeling.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "rank_first_throughput_challenger_v1.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "artifacts" / "rank_first_throughput_challenger_v1.md",
    )
    args = parser.parse_args()

    report = build_report(
        profile_dirs=[args.dataset_profile_dir, args.additional_dataset_profile_dir],
        h800_observations_path=args.h800_observations,
        h800_theory_basis_path=args.h800_theory_basis,
        h800_inventory_path=args.h800_model_inventory,
        h800_hardware_path=args.h800_hardware,
        h800_runtime_root=args.h800_runtime_root,
        extension_evaluation_path=args.extension_evaluation,
        extension_predictions_path=args.extension_predictions,
        v5_artifact_path=args.v5_artifact,
    )
    write_json(args.output, report)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "markdown_output": str(args.markdown_output.resolve()),
        "report_sha256": report["report_sha256"],
        "extension": report["evaluation"]["extension"],
        "cutoff_invariance_audit": report["cutoff_invariance_audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
