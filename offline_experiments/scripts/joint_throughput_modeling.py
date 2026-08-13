#!/usr/bin/env python3
"""Fit a full-factor joint absolute-throughput and pairwise-ranking model.

The model does not force the historical fitted physical throughput estimate to
be an additive baseline.  Instead, it predicts positive step time from
configuration, model, workload, mechanism and hardware features.  Model
selection compares direct log-step regression with a residual around an
analytic FLOPs/HBM/communication time anchor:

    log(step_seconds_hat) = optional_log_analytic_anchor
                            + intercept
                            + standardized_features @ beta
    log(throughput_hat) = log(effective_tokens_per_step) - log(step_seconds_hat)

The fit jointly minimizes scenario-balanced absolute log-step error and
scenario-balanced pairwise log-step-difference error.  Physical quantities such
as FLOPs, HBM traffic and communication payload are ordinary input features,
not a baseline that the ranker must first undo.

This script is offline and diagnostic.  It launches no GPU experiments, mutates
no queue and publishes no production profile.
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

from common import ROOT, read_json, sha256_file, sha256_json, write_json
from h800_challenger_modeling import (
    SCHEMA as H800_HYBRID_SCHEMA,
    _build_native_throughput_record,
    _fit_pairwise_ranker,
    _inventory_models,
    _is_legacy,
    _observation_id,
    _outcome,
    _physics_throughput_score,
    _ranker_score,
    _read_observations,
    _same_split_audit,
    _throughput_candidates,
    _verify_bound_report,
    throughput_admission_reason,
)
from h800_theory_calibration import (
    _candidate_key,
    _fit_compute_model,
    _observed_step_seconds,
    _work_per_step,
    scenario_id,
    scenario_material,
)
from rtx4090_challenger_modeling import (
    DEFAULT_ALPHA as RTX4090_BASELINE_ALPHA,
    DEFAULT_COLLECTIVE_BANDWIDTH_BYTES_S,
    DEFAULT_HBM_BANDWIDTH_BYTES_S,
    _build_records as build_4090_records,
    _physical_priors as physical_4090_priors,
)


SCHEMA = "sft_joint_throughput_modeling/v1"
IMPLEMENTATION_VERSION = (
    "sft_joint_throughput_modeling_impl/"
    "2026-07-28.full-factor-two-head-throughput-ranker"
)
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
PAIR_WEIGHT_GRID = (0.25, 1.0, 4.0, 16.0)
TARGET_MODES = ("direct_log_step", "analytic_anchor_residual")
ABSOLUTE_DEVIATION_BLEND_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
HUBER_DELTA = 0.35
H800_HISTORICAL_SCENARIO_WEIGHT = 0.25
EPSILON = 1e-12


FEATURE_NAMES = (
    # Training mechanism and discrete configuration.
    "is_lora",
    "gradient_checkpointing",
    "packing",
    "zero1",
    "zero2",
    "zero3",
    "dtype_bf16",
    "kernel_fa2",
    "kernel_fa3",
    "kernel_liger",
    "kernel_fused_ce",
    "kernel_fused_optimizer",
    "kernel_compile",
    "dataset_short",
    "dataset_multiturn",
    "dataset_longtail",
    "dataset_longcontext",
    # Resource and batching.
    "log2_gpu_count",
    "log2_mbs",
    "log2_gradient_accumulation",
    "log2_target_gbs",
    "log2_cutoff_over_512",
    # Model architecture.
    "log2_base_parameters_over_1p7b",
    "log2_trainable_parameters_over_1p7b",
    "log2_loaded_parameters_over_1p7b",
    "log2_num_layers",
    "log2_hidden_size",
    "log2_intermediate_size",
    "log2_attention_heads",
    "log2_kv_width",
    "log2_vocab_size",
    "log2_lora_rank",
    # Dataset/work-per-step features available from a dataset profile.
    "log1p_effective_tokens_per_step",
    "log1p_computed_tokens_per_step",
    "log1p_attention_pairs_per_step",
    "log1p_logical_samples_per_step",
    "log2_effective_tokens_per_sample_over_512",
    "log2_computed_tokens_per_sample_over_512",
    "log1p_attention_pairs_per_computed_token",
    "effective_to_computed_token_ratio",
    # Analytic workload features.
    "log1p_total_flops",
    "attention_flop_share",
    "recompute_flop_share",
    "log1p_kernel_traffic_bytes",
    "log1p_optimizer_traffic_bytes",
    "log1p_communication_payload_bytes",
    "log1p_collective_count",
    "log1p_ideal_compute_seconds",
    "log1p_ideal_kernel_hbm_seconds",
    "log1p_ideal_optimizer_hbm_seconds",
    "log1p_ideal_collective_seconds",
    "log1p_flops_per_kernel_byte",
    "ideal_collective_time_share",
    # Hardware descriptors.  They are constant in a per-card fit but make the
    # feature contract suitable for a future multi-card shared model.
    "log1p_dense_peak_flops_per_gpu",
    "log1p_hbm_bandwidth_bytes_per_second",
    "log1p_link_bandwidth_bytes_per_second",
    # Selected mechanism/scale interactions.
    "lora_x_gc",
    "lora_x_zero2",
    "lora_x_zero3",
    "gc_x_zero2",
    "gc_x_zero3",
    "zero2_x_log2_gpu_count",
    "zero3_x_log2_gpu_count",
    "zero2_x_log2_parameters",
    "zero3_x_log2_parameters",
    "gc_x_log2_cutoff",
    "gc_x_log2_parameters",
    "log2_mbs_x_log2_cutoff",
    "log2_mbs_x_log2_parameters",
    "log2_gpu_count_x_log2_parameters",
)

COMPACT_FEATURE_NAMES = (
    # Every variable recommendation factor remains explicit.
    "is_lora",
    "gradient_checkpointing",
    "packing",
    "zero1",
    "zero2",
    "zero3",
    "log2_gpu_count",
    "log2_mbs",
    "log2_gradient_accumulation",
    "log2_target_gbs",
    "log2_cutoff_over_512",
    "log2_base_parameters_over_1p7b",
    # Dataset distribution is represented by quantitative pre-run work
    # expectations rather than dataset-name dummies.
    "log1p_effective_tokens_per_step",
    "log2_effective_tokens_per_sample_over_512",
    "log2_computed_tokens_per_sample_over_512",
    "log1p_attention_pairs_per_computed_token",
    "effective_to_computed_token_ratio",
    # Model architecture, kernel work and hardware limits are compressed into
    # analytic workload quantities to reduce collinearity during scale holdout.
    "log1p_total_flops",
    "attention_flop_share",
    "recompute_flop_share",
    "log1p_kernel_traffic_bytes",
    "log1p_optimizer_traffic_bytes",
    "log1p_communication_payload_bytes",
    "log1p_collective_count",
    "log1p_ideal_compute_seconds",
    "log1p_ideal_kernel_hbm_seconds",
    "log1p_ideal_optimizer_hbm_seconds",
    "log1p_ideal_collective_seconds",
    "log1p_flops_per_kernel_byte",
    "ideal_collective_time_share",
    # Configuration/scale interactions.
    "lora_x_gc",
    "lora_x_zero2",
    "lora_x_zero3",
    "gc_x_zero2",
    "gc_x_zero3",
    "zero2_x_log2_gpu_count",
    "zero3_x_log2_gpu_count",
    "zero2_x_log2_parameters",
    "zero3_x_log2_parameters",
    "gc_x_log2_cutoff",
    "gc_x_log2_parameters",
    "log2_mbs_x_log2_cutoff",
    "log2_mbs_x_log2_parameters",
    "log2_gpu_count_x_log2_parameters",
)

FEATURE_SETS = {
    "compact_analytic": COMPACT_FEATURE_NAMES,
    "full_factor": FEATURE_NAMES,
}


def _positive(value: Any, default: float = EPSILON) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number) or number <= 0:
        return default
    return number


def _nonnegative(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number) or number < 0:
        return 0.0
    return number


def _log2_ratio(value: Any, reference: float = 1.0) -> float:
    return math.log2(_positive(value) / reference)


def _ratio(numerator: Any, denominator: Any) -> float:
    return _nonnegative(numerator) / _positive(denominator)


def _record_features(record: Mapping[str, Any]) -> np.ndarray:
    scenario = record.get("scenario") or {}
    selector = record.get("selector") or {}
    model = record.get("model_basis") or {}
    performance = record.get("performance") or {}
    work = performance.get("work_per_step") or {}
    flops = performance.get("flops_per_step") or {}
    traffic = performance.get("traffic_bytes_per_rank_step") or {}
    communication = performance.get("communication") or {}
    ideal = performance.get("ideal_seconds") or {}
    priors = performance.get("physical_priors") or {}

    training_mode = str(
        selector.get("training_mode")
        or scenario.get("train_type")
        or ""
    ).lower()
    is_lora = float(training_mode == "lora")
    gc = float(bool(selector.get("gradient_checkpointing")))
    packing = float(bool(selector.get("packing")))
    zero = int(selector.get("zero_stage") or 0)
    zero1 = float(zero == 1)
    zero2 = float(zero == 2)
    zero3 = float(zero == 3)
    dtype = str(selector.get("dtype") or "").lower()
    kernel = str(selector.get("kernel_path") or "").lower()
    dataset = str(scenario.get("dataset_id") or "").lower()

    gpu_count = _positive(scenario.get("gpu_count"))
    mbs = _positive(scenario.get("physical_mbs"))
    accumulation = _positive(
        performance.get("gradient_accumulation_steps")
    )
    target_gbs = _positive(scenario.get("target_gbs"))
    cutoff = _positive(scenario.get("cutoff_len"), 512.0)
    parameters = _positive(model.get("base_parameters"))

    effective_tokens = _positive(work.get("effective_tokens"))
    computed_tokens = _positive(work.get("computed_tokens"))
    attention_pairs = _positive(
        work.get("computed_attention_token_pairs")
    )
    logical_samples = _positive(work.get("logical_samples"))
    total_flops = _positive(flops.get("total"))
    attention_flops = _nonnegative(flops.get("attention")) + _nonnegative(
        flops.get("recompute_attention")
    )
    recompute_flops = _nonnegative(
        flops.get("recompute_linear")
    ) + _nonnegative(flops.get("recompute_attention"))
    kernel_traffic = _positive(traffic.get("kernel_total"))
    optimizer_traffic = _positive(traffic.get("optimizer"))
    communication_payload = _positive(
        communication.get("payload_bytes_per_rank_step")
    )
    collective_count = _nonnegative(
        communication.get("collective_count")
    )
    ideal_compute = _nonnegative(ideal.get("compute_at_dense_peak"))
    ideal_kernel = _nonnegative(
        ideal.get("kernel_hbm_at_physical_peak")
    )
    ideal_optimizer = _nonnegative(
        ideal.get("optimizer_hbm_at_physical_peak")
    )
    ideal_collective = _nonnegative(
        ideal.get("collective_payload_at_link_peak")
    )
    ideal_total = (
        ideal_compute + ideal_kernel + ideal_optimizer + ideal_collective
    )

    log_gpu = math.log2(gpu_count)
    log_mbs = math.log2(mbs)
    log_parameters = _log2_ratio(parameters, 2_031_739_904.0)
    log_cutoff = _log2_ratio(cutoff, 512.0)

    values = {
        "is_lora": is_lora,
        "gradient_checkpointing": gc,
        "packing": packing,
        "zero1": zero1,
        "zero2": zero2,
        "zero3": zero3,
        "dtype_bf16": float(dtype == "bf16"),
        "kernel_fa2": float("fa2" in kernel),
        "kernel_fa3": float("fa3" in kernel),
        "kernel_liger": float("liger" in kernel),
        "kernel_fused_ce": float("fused_ce" in kernel),
        "kernel_fused_optimizer": float(
            "adamw_torch_fused" in kernel or "fused_optim" in kernel
        ),
        "kernel_compile": float(
            "compile=true" in kernel or "compile_true" in kernel
        ),
        "dataset_short": float(dataset.startswith("short")),
        "dataset_multiturn": float(dataset.startswith("multiturn")),
        "dataset_longtail": float(dataset.startswith("longtail")),
        "dataset_longcontext": float(dataset.startswith("longcontext")),
        "log2_gpu_count": log_gpu,
        "log2_mbs": log_mbs,
        "log2_gradient_accumulation": math.log2(accumulation),
        "log2_target_gbs": math.log2(target_gbs),
        "log2_cutoff_over_512": log_cutoff,
        "log2_base_parameters_over_1p7b": log_parameters,
        "log2_trainable_parameters_over_1p7b": _log2_ratio(
            model.get("trainable_parameters"), 2_031_739_904.0
        ),
        "log2_loaded_parameters_over_1p7b": _log2_ratio(
            model.get("loaded_parameters"), 2_031_739_904.0
        ),
        "log2_num_layers": _log2_ratio(model.get("num_layers")),
        "log2_hidden_size": _log2_ratio(model.get("hidden_size")),
        "log2_intermediate_size": _log2_ratio(
            model.get("intermediate_size")
        ),
        "log2_attention_heads": _log2_ratio(
            model.get("num_attention_heads")
        ),
        "log2_kv_width": _log2_ratio(model.get("kv_width")),
        "log2_vocab_size": _log2_ratio(model.get("vocab_size")),
        "log2_lora_rank": _log2_ratio(model.get("lora_rank")),
        "log1p_effective_tokens_per_step": math.log1p(effective_tokens),
        "log1p_computed_tokens_per_step": math.log1p(computed_tokens),
        "log1p_attention_pairs_per_step": math.log1p(attention_pairs),
        "log1p_logical_samples_per_step": math.log1p(logical_samples),
        "log2_effective_tokens_per_sample_over_512": _log2_ratio(
            effective_tokens / logical_samples, 512.0
        ),
        "log2_computed_tokens_per_sample_over_512": _log2_ratio(
            computed_tokens / logical_samples, 512.0
        ),
        "log1p_attention_pairs_per_computed_token": math.log1p(
            attention_pairs / computed_tokens
        ),
        "effective_to_computed_token_ratio": min(
            1.5, effective_tokens / computed_tokens
        ),
        "log1p_total_flops": math.log1p(total_flops),
        "attention_flop_share": attention_flops / total_flops,
        "recompute_flop_share": recompute_flops / total_flops,
        "log1p_kernel_traffic_bytes": math.log1p(kernel_traffic),
        "log1p_optimizer_traffic_bytes": math.log1p(optimizer_traffic),
        "log1p_communication_payload_bytes": math.log1p(
            communication_payload
        ),
        "log1p_collective_count": math.log1p(collective_count),
        "log1p_ideal_compute_seconds": math.log1p(ideal_compute),
        "log1p_ideal_kernel_hbm_seconds": math.log1p(ideal_kernel),
        "log1p_ideal_optimizer_hbm_seconds": math.log1p(
            ideal_optimizer
        ),
        "log1p_ideal_collective_seconds": math.log1p(
            ideal_collective
        ),
        "log1p_flops_per_kernel_byte": math.log1p(
            total_flops / kernel_traffic
        ),
        "ideal_collective_time_share": (
            ideal_collective / ideal_total if ideal_total > 0 else 0.0
        ),
        "log1p_dense_peak_flops_per_gpu": math.log1p(
            _positive(priors.get("dense_bf16_peak_flops_per_gpu"))
        ),
        "log1p_hbm_bandwidth_bytes_per_second": math.log1p(
            _positive(priors.get("hbm_bandwidth_bytes_per_second"))
        ),
        "log1p_link_bandwidth_bytes_per_second": math.log1p(
            _positive(
                priors.get("intra_node_bandwidth_bytes_per_second")
            )
        ),
        "lora_x_gc": is_lora * gc,
        "lora_x_zero2": is_lora * zero2,
        "lora_x_zero3": is_lora * zero3,
        "gc_x_zero2": gc * zero2,
        "gc_x_zero3": gc * zero3,
        "zero2_x_log2_gpu_count": zero2 * log_gpu,
        "zero3_x_log2_gpu_count": zero3 * log_gpu,
        "zero2_x_log2_parameters": zero2 * log_parameters,
        "zero3_x_log2_parameters": zero3 * log_parameters,
        "gc_x_log2_cutoff": gc * log_cutoff,
        "gc_x_log2_parameters": gc * log_parameters,
        "log2_mbs_x_log2_cutoff": log_mbs * log_cutoff,
        "log2_mbs_x_log2_parameters": log_mbs * log_parameters,
        "log2_gpu_count_x_log2_parameters": log_gpu * log_parameters,
    }
    vector = np.asarray([values[name] for name in FEATURE_NAMES], dtype=float)
    if not np.isfinite(vector).all():
        raise ValueError("Full-factor feature vector contains NaN/Inf")
    return vector


def _record_log_analytic_anchor(record: Mapping[str, Any]) -> float:
    performance = record.get("performance") or {}
    ideal = performance.get("ideal_seconds") or {}
    communication = performance.get("communication") or {}
    priors = performance.get("physical_priors") or {}
    compute = _nonnegative(ideal.get("compute_at_dense_peak"))
    kernel_hbm = _nonnegative(
        ideal.get("kernel_hbm_at_physical_peak")
    )
    optimizer_hbm = _nonnegative(
        ideal.get("optimizer_hbm_at_physical_peak")
    )
    collective_payload = _nonnegative(
        ideal.get("collective_payload_at_link_peak")
    )
    collective_latency = _nonnegative(
        priors.get("collective_latency_seconds")
    )
    collective_count = _nonnegative(
        communication.get("collective_count")
    )
    anchor_seconds = (
        max(compute, kernel_hbm)
        + optimizer_hbm
        + collective_payload
        + collective_count * collective_latency
    )
    return math.log(_positive(anchor_seconds))


def _pool_candidates(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        str, dict[tuple[Any, ...], list[Mapping[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[scenario_id(record)][_candidate_key(record)].append(record)
    candidates: list[dict[str, Any]] = []
    for current_scenario, by_candidate in sorted(grouped.items()):
        for key, replicates in sorted(
            by_candidate.items(),
            key=lambda item: tuple(str(value) for value in item[0]),
        ):
            log_throughputs = []
            log_effective_work = []
            for record in replicates:
                step = _observed_step_seconds(record)
                work = _work_per_step(record, "effective_tokens")
                if step is None or work is None or step <= 0 or work <= 0:
                    raise ValueError("Throughput candidate lacks step/work")
                log_throughputs.append(math.log(work / step))
                log_effective_work.append(math.log(work))
            feature_matrix = np.vstack(
                [_record_features(record) for record in replicates]
            )
            log_anchors = [
                _record_log_analytic_anchor(record)
                for record in replicates
            ]
            observed_log_throughput = statistics.median(log_throughputs)
            effective_log_work = statistics.median(log_effective_work)
            candidates.append(
                {
                    "scenario_id": current_scenario,
                    "scenario": scenario_material(replicates[0]),
                    "candidate_key": list(key),
                    "record": replicates[0],
                    "replicate_records": list(replicates),
                    "replicates": len(replicates),
                    "features": np.median(feature_matrix, axis=0),
                    "effective_log_work": effective_log_work,
                    "observed_log_throughput": observed_log_throughput,
                    "observed_log_step": (
                        effective_log_work - observed_log_throughput
                    ),
                    "log_analytic_anchor": statistics.median(
                        log_anchors
                    ),
                    "historical": all(
                        _is_legacy(record) for record in replicates
                    ),
                    "observation_ids": sorted(
                        _observation_id(record) for record in replicates
                    ),
                }
            )
    return candidates


def _scenario_weight(
    candidate: Mapping[str, Any],
    *,
    historical_weight: float,
) -> float:
    return (
        historical_weight
        if candidate.get("historical") is True
        else 1.0
    )


def _fit_joint_model(
    candidates: Sequence[Mapping[str, Any]],
    *,
    feature_set: str,
    target_mode: str,
    alpha: float,
    pair_weight: float,
    historical_weight: float,
) -> dict[str, Any]:
    usable = [
        candidate
        for candidate in candidates
        if historical_weight > 0 or candidate.get("historical") is not True
    ]
    if not usable:
        raise ValueError("No candidates for joint throughput model")
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unknown target mode {target_mode!r}")

    def target_value(candidate: Mapping[str, Any]) -> float:
        observed_log_step = float(candidate["observed_log_step"])
        if target_mode == "analytic_anchor_residual":
            return observed_log_step - float(
                candidate["log_analytic_anchor"]
            )
        return observed_log_step

    scenario_counts = Counter(
        str(candidate["scenario_id"]) for candidate in usable
    )
    absolute_weights = np.asarray(
        [
            _scenario_weight(
                candidate, historical_weight=historical_weight
            )
            / scenario_counts[str(candidate["scenario_id"])]
            for candidate in usable
        ],
        dtype=float,
    )
    selected_names = FEATURE_SETS[feature_set]
    selected_indexes = [FEATURE_NAMES.index(name) for name in selected_names]
    raw_features = np.vstack(
        [np.asarray(candidate["features"], dtype=float) for candidate in usable]
    )[:, selected_indexes]
    means = np.average(raw_features, axis=0, weights=absolute_weights)
    scales = np.sqrt(
        np.average(
            (raw_features - means) ** 2,
            axis=0,
            weights=absolute_weights,
        )
    )
    scales[scales < 1e-9] = 1.0
    standardized = (raw_features - means) / scales

    rows: list[np.ndarray] = []
    targets: list[float] = []
    base_weights: list[float] = []
    row_kinds: list[str] = []
    for index, candidate in enumerate(usable):
        rows.append(np.concatenate(([1.0], standardized[index])))
        targets.append(target_value(candidate))
        base_weights.append(float(absolute_weights[index]))
        row_kinds.append("absolute")

    by_scenario: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(usable):
        by_scenario[str(candidate["scenario_id"])].append(index)
    pair_count = 0
    pair_scenarios = 0
    for indexes in by_scenario.values():
        pairs = [
            (indexes[left], indexes[right])
            for left in range(len(indexes))
            for right in range(left + 1, len(indexes))
        ]
        if not pairs:
            continue
        pair_scenarios += 1
        scenario_weight = _scenario_weight(
            usable[indexes[0]], historical_weight=historical_weight
        )
        weight = float(pair_weight) * scenario_weight / len(pairs)
        for left, right in pairs:
            rows.append(
                np.concatenate(
                    ([0.0], standardized[left] - standardized[right])
                )
            )
            targets.append(
                target_value(usable[left]) - target_value(usable[right])
            )
            base_weights.append(weight)
            row_kinds.append("pairwise")
            pair_count += 1

    design = np.vstack(rows)
    target = np.asarray(targets, dtype=float)
    base = np.asarray(base_weights, dtype=float)
    coefficients = np.zeros(design.shape[1], dtype=float)
    regularizer = np.eye(design.shape[1], dtype=float) * float(alpha)
    regularizer[0, 0] = 0.0
    iterations = 0
    for iterations in range(1, 26):
        residual = target - design @ coefficients
        robust = np.ones_like(residual)
        outside = np.abs(residual) > HUBER_DELTA
        robust[outside] = HUBER_DELTA / np.abs(residual[outside])
        weights = base * robust
        normal = design.T @ (weights[:, None] * design) + regularizer
        rhs = design.T @ (weights * target)
        updated = np.linalg.pinv(normal) @ rhs
        if np.max(np.abs(updated - coefficients)) < 1e-9:
            coefficients = updated
            break
        coefficients = updated

    fitted = design @ coefficients
    residual = target - fitted
    absolute_mask = np.asarray(
        [kind == "absolute" for kind in row_kinds], dtype=bool
    )
    pair_mask = ~absolute_mask
    return {
        "available": True,
        "model_family": "robust_joint_log_step_ridge",
        "formula": (
            "log(step_hat) = optional_log_analytic_anchor + intercept + "
            "standardized_full_factors @ beta; "
            "log(throughput_hat) = log(effective_tokens_per_step) "
            "- log(step_hat)"
        ),
        "target_mode": target_mode,
        "feature_set": feature_set,
        "feature_names": list(selected_names),
        "feature_dimension": len(selected_names),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "alpha": float(alpha),
        "pair_weight": float(pair_weight),
        "historical_scenario_weight": float(historical_weight),
        "huber_delta_log_seconds": HUBER_DELTA,
        "irls_iterations": iterations,
        "fit_candidates": len(usable),
        "fit_scenarios": len(by_scenario),
        "fit_pairs": pair_count,
        "pair_scenarios": pair_scenarios,
        "fit_historical_candidates": sum(
            candidate.get("historical") is True for candidate in usable
        ),
        "fit_native_candidates": sum(
            candidate.get("historical") is not True for candidate in usable
        ),
        "absolute_log_step_rmse_in_sample": float(
            np.sqrt(np.mean(residual[absolute_mask] ** 2))
        ),
        "pairwise_log_step_rmse_in_sample": (
            float(np.sqrt(np.mean(residual[pair_mask] ** 2)))
            if pair_mask.any()
            else None
        ),
    }


def _predict_log_throughput(
    candidate: Mapping[str, Any],
    model: Mapping[str, Any],
) -> float:
    selected_indexes = [
        FEATURE_NAMES.index(name) for name in model["feature_names"]
    ]
    features = np.asarray(candidate["features"], dtype=float)[
        selected_indexes
    ]
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    log_step = float(model["intercept"]) + float(
        ((features - means) / scales) @ coefficients
    )
    if model["target_mode"] == "analytic_anchor_residual":
        log_step += float(candidate["log_analytic_anchor"])
    return float(candidate["effective_log_work"]) - log_step


def _fit_pairwise_head(
    candidates: Sequence[Mapping[str, Any]],
    *,
    feature_set: str,
    alpha: float,
    historical_weight: float,
) -> dict[str, Any]:
    """Fit a separate configuration-ordering head in log-throughput units."""
    usable = [
        candidate
        for candidate in candidates
        if historical_weight > 0 or candidate.get("historical") is not True
    ]
    if not usable:
        raise ValueError("No candidates for pairwise throughput head")
    selected_names = FEATURE_SETS[feature_set]
    selected_indexes = [
        FEATURE_NAMES.index(name) for name in selected_names
    ]
    scenario_counts = Counter(
        str(candidate["scenario_id"]) for candidate in usable
    )
    candidate_weights = np.asarray(
        [
            _scenario_weight(
                candidate, historical_weight=historical_weight
            )
            / scenario_counts[str(candidate["scenario_id"])]
            for candidate in usable
        ],
        dtype=float,
    )
    raw_features = np.vstack(
        [np.asarray(candidate["features"], dtype=float) for candidate in usable]
    )[:, selected_indexes]
    means = np.average(raw_features, axis=0, weights=candidate_weights)
    scales = np.sqrt(
        np.average(
            (raw_features - means) ** 2,
            axis=0,
            weights=candidate_weights,
        )
    )
    scales[scales < 1e-9] = 1.0
    standardized = (raw_features - means) / scales

    by_scenario: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(usable):
        by_scenario[str(candidate["scenario_id"])].append(index)
    rows: list[np.ndarray] = []
    targets: list[float] = []
    base_weights: list[float] = []
    pair_scenarios = 0
    for indexes in by_scenario.values():
        pairs = [
            (indexes[left], indexes[right])
            for left in range(len(indexes))
            for right in range(left + 1, len(indexes))
        ]
        if not pairs:
            continue
        pair_scenarios += 1
        scenario_weight = _scenario_weight(
            usable[indexes[0]], historical_weight=historical_weight
        )
        for left, right in pairs:
            rows.append(standardized[left] - standardized[right])
            left_log_throughput = (
                float(usable[left]["effective_log_work"])
                - float(usable[left]["observed_log_step"])
            )
            right_log_throughput = (
                float(usable[right]["effective_log_work"])
                - float(usable[right]["observed_log_step"])
            )
            targets.append(left_log_throughput - right_log_throughput)
            base_weights.append(scenario_weight / len(pairs))
    if not rows:
        raise ValueError("Pairwise throughput head has no comparable pairs")

    design = np.vstack(rows)
    target = np.asarray(targets, dtype=float)
    base = np.asarray(base_weights, dtype=float)
    coefficients = np.zeros(design.shape[1], dtype=float)
    regularizer = np.eye(design.shape[1], dtype=float) * float(alpha)
    iterations = 0
    for iterations in range(1, 26):
        residual = target - design @ coefficients
        robust = np.ones_like(residual)
        outside = np.abs(residual) > HUBER_DELTA
        robust[outside] = HUBER_DELTA / np.abs(residual[outside])
        weights = base * robust
        normal = design.T @ (weights[:, None] * design) + regularizer
        rhs = design.T @ (weights * target)
        updated = np.linalg.pinv(normal) @ rhs
        if np.max(np.abs(updated - coefficients)) < 1e-9:
            coefficients = updated
            break
        coefficients = updated
    residual = target - design @ coefficients
    return {
        "available": True,
        "model_family": "robust_pairwise_log_throughput_ridge_head",
        "formula": (
            "rank_score = standardized_selected_factors @ gamma; "
            "gamma is fitted to within-scenario log-throughput differences"
        ),
        "feature_set": feature_set,
        "feature_names": list(selected_names),
        "feature_dimension": len(selected_names),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "coefficients": coefficients.tolist(),
        "alpha": float(alpha),
        "historical_scenario_weight": float(historical_weight),
        "huber_delta_log_throughput": HUBER_DELTA,
        "irls_iterations": iterations,
        "fit_candidates": len(usable),
        "fit_scenarios": len(by_scenario),
        "fit_pairs": len(rows),
        "pair_scenarios": pair_scenarios,
        "pairwise_log_throughput_rmse_in_sample": float(
            np.sqrt(np.mean(residual**2))
        ),
    }


def _pairwise_head_score(
    candidate: Mapping[str, Any],
    model: Mapping[str, Any],
) -> float:
    selected_indexes = [
        FEATURE_NAMES.index(name) for name in model["feature_names"]
    ]
    features = np.asarray(candidate["features"], dtype=float)[
        selected_indexes
    ]
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    return float(((features - means) / scales) @ coefficients)


def _fit_two_head_model(
    candidates: Sequence[Mapping[str, Any]],
    *,
    absolute_settings: Mapping[str, Any],
    rank_feature_set: str,
    rank_alpha: float,
    absolute_deviation_blend: float,
    historical_weight: float,
) -> dict[str, Any]:
    absolute_head = _fit_joint_model(
        candidates,
        feature_set=str(absolute_settings["feature_set"]),
        target_mode=str(absolute_settings["target_mode"]),
        alpha=float(absolute_settings["alpha"]),
        pair_weight=float(absolute_settings["pair_weight"]),
        historical_weight=historical_weight,
    )
    rank_head = _fit_pairwise_head(
        candidates,
        feature_set=rank_feature_set,
        alpha=rank_alpha,
        historical_weight=historical_weight,
    )
    return {
        "available": True,
        "model_family": "set_aware_two_head_log_throughput_model",
        "formula": (
            "a_i=absolute_head_log_throughput(x_i); "
            "r_i=pairwise_head_score(x_i); "
            "log(throughput_hat_i)=mean_j(a_j) + lambda*"
            "(a_i-mean_j(a_j)) + (1-lambda)*(r_i-mean_j(r_j))"
        ),
        "candidate_set_contract": (
            "j ranges over feasible configurations for one fixed user "
            "scenario; a singleton falls back to the absolute head"
        ),
        "absolute_deviation_blend": float(absolute_deviation_blend),
        "absolute_head": absolute_head,
        "rank_head": rank_head,
    }


def _predict_two_head_entries(
    candidates: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], float]]:
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_scenario[str(candidate["scenario_id"])].append(candidate)
    absolute_head = model["absolute_head"]
    rank_head = model["rank_head"]
    blend = float(model["absolute_deviation_blend"])
    entries: list[tuple[Mapping[str, Any], float]] = []
    for current_candidates in by_scenario.values():
        absolute = np.asarray(
            [
                _predict_log_throughput(candidate, absolute_head)
                for candidate in current_candidates
            ],
            dtype=float,
        )
        rank = np.asarray(
            [
                _pairwise_head_score(candidate, rank_head)
                for candidate in current_candidates
            ],
            dtype=float,
        )
        absolute_center = float(np.mean(absolute))
        rank_center = float(np.mean(rank))
        predictions = (
            absolute_center
            + blend * (absolute - absolute_center)
            + (1.0 - blend) * (rank - rank_center)
        )
        entries.extend(zip(current_candidates, predictions.tolist()))
    return entries


def _mean(values: Sequence[float]) -> float | None:
    usable = [
        float(value) for value in values if math.isfinite(float(value))
    ]
    return statistics.fmean(usable) if usable else None


def _evaluate_entries(
    entries: Sequence[tuple[Mapping[str, Any], float]],
) -> dict[str, Any]:
    by_scenario: dict[
        str, list[tuple[Mapping[str, Any], float]]
    ] = defaultdict(list)
    for candidate, prediction in entries:
        by_scenario[str(candidate["scenario_id"])].append(
            (candidate, float(prediction))
        )

    absolute_errors = []
    absolute_log_errors = []
    scenario_mapes = []
    scenario_log_rmses = []
    pooled_pair_correct = pooled_pair_rows = 0
    scenario_pair_accuracies = []
    top1_regrets = []
    top1_hits = []
    gpu_regrets = []
    gpu_hits = []
    details = []
    for current_scenario, rows in sorted(by_scenario.items()):
        row_apes = []
        row_log_errors = []
        for candidate, prediction in rows:
            observed = float(candidate["observed_log_throughput"])
            log_error = prediction - observed
            ape = abs(math.exp(max(-50.0, min(50.0, log_error))) - 1.0)
            row_apes.append(ape)
            row_log_errors.append(log_error)
            absolute_errors.append(ape)
            absolute_log_errors.append(log_error)
        scenario_mapes.append(statistics.fmean(row_apes))
        scenario_log_rmses.append(
            math.sqrt(statistics.fmean(error**2 for error in row_log_errors))
        )

        pair_correct = pair_rows = 0
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                observed_delta = (
                    float(rows[left][0]["observed_log_throughput"])
                    - float(rows[right][0]["observed_log_throughput"])
                )
                if abs(observed_delta) <= 1e-12:
                    continue
                predicted_delta = rows[left][1] - rows[right][1]
                pair_correct += int(
                    (observed_delta > 0) == (predicted_delta > 0)
                )
                pair_rows += 1
        pooled_pair_correct += pair_correct
        pooled_pair_rows += pair_rows
        if pair_rows:
            scenario_pair_accuracies.append(pair_correct / pair_rows)

        if len(rows) >= 2:
            oracle = max(
                rows,
                key=lambda item: float(
                    item[0]["observed_log_throughput"]
                ),
            )
            selected = max(rows, key=lambda item: item[1])
            regret = max(
                0.0,
                1.0
                - math.exp(
                    float(selected[0]["observed_log_throughput"])
                    - float(oracle[0]["observed_log_throughput"])
                ),
            )
            top1_regrets.append(regret)
            top1_hits.append(float(regret <= 0.10))

        by_gpu: dict[
            int, list[tuple[Mapping[str, Any], float]]
        ] = defaultdict(list)
        for candidate, prediction in rows:
            gpu_count = int(candidate["record"]["scenario"]["gpu_count"])
            by_gpu[gpu_count].append((candidate, prediction))
        for gpu_count, gpu_rows in sorted(by_gpu.items()):
            if len(gpu_rows) < 2:
                continue
            oracle = max(
                gpu_rows,
                key=lambda item: float(
                    item[0]["observed_log_throughput"]
                ),
            )
            selected = max(gpu_rows, key=lambda item: item[1])
            regret = max(
                0.0,
                1.0
                - math.exp(
                    float(selected[0]["observed_log_throughput"])
                    - float(oracle[0]["observed_log_throughput"])
                ),
            )
            gpu_regrets.append(regret)
            gpu_hits.append(float(regret <= 0.10))
            details.append(
                {
                    "scenario_id": current_scenario,
                    "scenario": rows[0][0]["scenario"],
                    "gpu_count": gpu_count,
                    "candidates": len(gpu_rows),
                    "selected_candidate_key": selected[0]["candidate_key"],
                    "oracle_candidate_key": oracle[0]["candidate_key"],
                    "top1_regret": regret,
                    "hit_at_10_percent": regret <= 0.10,
                }
            )

    sorted_ape = sorted(absolute_errors)
    sorted_abs_log = sorted(abs(value) for value in absolute_log_errors)

    def percentile(values: Sequence[float], q: float) -> float | None:
        if not values:
            return None
        return float(np.quantile(np.asarray(values, dtype=float), q))

    return {
        "candidate_rows": len(entries),
        "scenario_rows": len(by_scenario),
        "comparable_scenario_rows": len(scenario_pair_accuracies),
        "pairwise_rows": pooled_pair_rows,
        "pairwise_correct_rows": pooled_pair_correct,
        "pooled_pairwise_accuracy": (
            pooled_pair_correct / pooled_pair_rows
            if pooled_pair_rows
            else None
        ),
        "scenario_equal_pairwise_accuracy": _mean(
            scenario_pair_accuracies
        ),
        "scenario_equal_top1_regret": _mean(top1_regrets),
        "scenario_equal_hit_at_10_percent": _mean(top1_hits),
        "gpu_group_rows": len(gpu_regrets),
        "scenario_gpu_equal_top1_regret": _mean(gpu_regrets),
        "scenario_gpu_equal_hit_at_10_percent": _mean(gpu_hits),
        "absolute_throughput_mape": _mean(absolute_errors),
        "scenario_equal_absolute_throughput_mape": _mean(
            scenario_mapes
        ),
        "absolute_throughput_ape_median": (
            statistics.median(sorted_ape) if sorted_ape else None
        ),
        "absolute_throughput_ape_p90": percentile(sorted_ape, 0.90),
        "absolute_log_error_rmse": (
            math.sqrt(
                statistics.fmean(
                    value**2 for value in absolute_log_errors
                )
            )
            if absolute_log_errors
            else None
        ),
        "scenario_equal_absolute_log_rmse": _mean(
            scenario_log_rmses
        ),
        "absolute_log_error_median": (
            statistics.median(absolute_log_errors)
            if absolute_log_errors
            else None
        ),
        "absolute_log_error_abs_p90": percentile(
            sorted_abs_log, 0.90
        ),
        "details": details,
    }


def _selection_objective(metrics: Mapping[str, Any]) -> float:
    log_rmse = float(metrics["scenario_equal_absolute_log_rmse"])
    pair_accuracy = metrics.get("scenario_equal_pairwise_accuracy")
    pair_penalty = (
        1.0 - float(pair_accuracy) if pair_accuracy is not None else 0.5
    )
    top1 = float(metrics.get("scenario_equal_top1_regret") or 0.0)
    gpu_top1 = float(
        metrics.get("scenario_gpu_equal_top1_regret") or 0.0
    )
    return log_rmse + 0.5 * pair_penalty + top1 + 0.5 * gpu_top1


def _scenario_folds(
    candidates: Sequence[Mapping[str, Any]],
    fold_count: int,
) -> list[set[str]]:
    by_scenario: dict[str, int] = Counter(
        str(candidate["scenario_id"]) for candidate in candidates
    )
    fold_count = max(2, min(fold_count, len(by_scenario)))
    folds = [set() for _ in range(fold_count)]
    pair_loads = [0] * fold_count
    candidate_loads = [0] * fold_count
    scenarios = sorted(
        by_scenario.items(),
        key=lambda item: (
            -(item[1] * (item[1] - 1) // 2),
            -item[1],
            item[0],
        ),
    )
    for current_scenario, count in scenarios:
        pair_count = count * (count - 1) // 2
        target = min(
            range(fold_count),
            key=lambda index: (
                pair_loads[index],
                candidate_loads[index],
                len(folds[index]),
                index,
            ),
        )
        folds[target].add(current_scenario)
        pair_loads[target] += pair_count
        candidate_loads[target] += count
    return folds


def _cross_validate_generic(
    candidates: Sequence[Mapping[str, Any]],
    *,
    feature_set: str,
    target_mode: str,
    alpha: float,
    pair_weight: float,
    historical_weight: float,
    fold_count: int,
) -> dict[str, Any]:
    entries: list[tuple[Mapping[str, Any], float]] = []
    folds = _scenario_folds(candidates, fold_count)
    fold_reports = []
    for held_out in folds:
        train = [
            candidate
            for candidate in candidates
            if str(candidate["scenario_id"]) not in held_out
        ]
        test = [
            candidate
            for candidate in candidates
            if str(candidate["scenario_id"]) in held_out
        ]
        model = _fit_joint_model(
            train,
            feature_set=feature_set,
            target_mode=target_mode,
            alpha=alpha,
            pair_weight=pair_weight,
            historical_weight=historical_weight,
        )
        fold_entries = [
            (candidate, _predict_log_throughput(candidate, model))
            for candidate in test
        ]
        entries.extend(fold_entries)
        fold_reports.append(
            {
                "held_out_scenario_ids": sorted(held_out),
                "train_candidates": len(train),
                "test_candidates": len(test),
                "metrics": _evaluate_entries(fold_entries),
            }
        )
    return {
        "method": f"deterministic scenario-balanced {len(folds)}-fold CV",
        "folds": fold_reports,
        "aggregate": _evaluate_entries(entries),
    }


def _cross_validate_h800_calibration(
    historical: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    *,
    feature_set: str,
    target_mode: str,
    alpha: float,
    pair_weight: float,
    historical_weight: float,
) -> dict[str, Any]:
    population = [*historical, *calibration]
    entries: list[tuple[Mapping[str, Any], float]] = []
    fold_reports = []
    calibration_scenarios = sorted(
        {str(candidate["scenario_id"]) for candidate in calibration}
    )
    for held_out in calibration_scenarios:
        train = [
            candidate
            for candidate in population
            if str(candidate["scenario_id"]) != held_out
        ]
        test = [
            candidate
            for candidate in calibration
            if str(candidate["scenario_id"]) == held_out
        ]
        model = _fit_joint_model(
            train,
            feature_set=feature_set,
            target_mode=target_mode,
            alpha=alpha,
            pair_weight=pair_weight,
            historical_weight=historical_weight,
        )
        fold_entries = [
            (candidate, _predict_log_throughput(candidate, model))
            for candidate in test
        ]
        entries.extend(fold_entries)
        fold_reports.append(
            {
                "held_out_scenario_id": held_out,
                "train_candidates": len(train),
                "test_candidates": len(test),
                "metrics": _evaluate_entries(fold_entries),
            }
        )
    return {
        "method": "native calibration leave-complete-scenario-out",
        "folds": fold_reports,
        "aggregate": _evaluate_entries(entries),
    }


def _cross_validate_model_ids(
    candidates: Sequence[Mapping[str, Any]],
    *,
    feature_set: str,
    target_mode: str,
    alpha: float,
    pair_weight: float,
    historical_weight: float,
) -> dict[str, Any]:
    model_ids = sorted(
        {
            str(candidate["record"]["scenario"]["model_id"])
            for candidate in candidates
        }
    )
    if len(model_ids) < 2:
        raise ValueError("Model-id CV requires at least two model ids")
    entries: list[tuple[Mapping[str, Any], float]] = []
    folds = []
    for held_model in model_ids:
        train = [
            candidate
            for candidate in candidates
            if str(candidate["record"]["scenario"]["model_id"])
            != held_model
        ]
        test = [
            candidate
            for candidate in candidates
            if str(candidate["record"]["scenario"]["model_id"])
            == held_model
        ]
        model = _fit_joint_model(
            train,
            feature_set=feature_set,
            target_mode=target_mode,
            alpha=alpha,
            pair_weight=pair_weight,
            historical_weight=historical_weight,
        )
        fold_entries = [
            (candidate, _predict_log_throughput(candidate, model))
            for candidate in test
        ]
        entries.extend(fold_entries)
        folds.append(
            {
                "held_out_model_id": held_model,
                "train_candidates": len(train),
                "test_candidates": len(test),
                "metrics": _evaluate_entries(fold_entries),
            }
        )
    return {
        "method": "leave-one-complete-model-id-out",
        "folds": folds,
        "aggregate": _evaluate_entries(entries),
    }


def _select_generic(
    candidates: Sequence[Mapping[str, Any]],
    *,
    historical_weight: float,
    fold_count: int,
) -> dict[str, Any]:
    evaluated = []
    for target_mode in TARGET_MODES:
        for feature_set in FEATURE_SETS:
            for alpha in ALPHA_GRID:
                for pair_weight in PAIR_WEIGHT_GRID:
                    cv = _cross_validate_generic(
                        candidates,
                        feature_set=feature_set,
                        target_mode=target_mode,
                        alpha=alpha,
                        pair_weight=pair_weight,
                        historical_weight=historical_weight,
                        fold_count=fold_count,
                    )
                    metrics = cv["aggregate"]
                    evaluated.append(
                        {
                            "target_mode": target_mode,
                            "feature_set": feature_set,
                            "feature_dimension": len(
                                FEATURE_SETS[feature_set]
                            ),
                            "alpha": alpha,
                            "pair_weight": pair_weight,
                            "objective": _selection_objective(metrics),
                            "metrics": metrics,
                        }
                    )
    selected = min(
        evaluated,
        key=lambda item: (
            float(item["objective"]),
            float(
                item["metrics"][
                    "scenario_equal_absolute_log_rmse"
                ]
            ),
            -float(
                item["metrics"].get(
                    "scenario_equal_pairwise_accuracy"
                )
                or 0.0
            ),
            int(item["feature_dimension"]),
            str(item["target_mode"]),
            float(item["pair_weight"]),
            float(item["alpha"]),
        ),
    )
    return {
        "selection_objective": (
            "scenario_equal_log_rmse + 0.5*(1-scenario_equal_pairwise_"
            "accuracy) + scenario_top1_regret + 0.5*fixed_gpu_top1_regret"
        ),
        "candidate_count": len(evaluated),
        "selected": selected,
        "candidates": evaluated,
    }


def _select_model_generalization(
    candidates: Sequence[Mapping[str, Any]],
    *,
    historical_weight: float,
) -> dict[str, Any]:
    evaluated = []
    for target_mode in TARGET_MODES:
        for feature_set in FEATURE_SETS:
            for alpha in ALPHA_GRID:
                for pair_weight in PAIR_WEIGHT_GRID:
                    cv = _cross_validate_model_ids(
                        candidates,
                        feature_set=feature_set,
                        target_mode=target_mode,
                        alpha=alpha,
                        pair_weight=pair_weight,
                        historical_weight=historical_weight,
                    )
                    metrics = cv["aggregate"]
                    evaluated.append(
                        {
                            "target_mode": target_mode,
                            "feature_set": feature_set,
                            "feature_dimension": len(
                                FEATURE_SETS[feature_set]
                            ),
                            "alpha": alpha,
                            "pair_weight": pair_weight,
                            "objective": _selection_objective(metrics),
                            "metrics": metrics,
                        }
                    )
    selected = min(
        evaluated,
        key=lambda item: (
            float(item["objective"]),
            float(
                item["metrics"][
                    "scenario_equal_absolute_log_rmse"
                ]
            ),
            -float(
                item["metrics"].get(
                    "scenario_equal_pairwise_accuracy"
                )
                or 0.0
            ),
            int(item["feature_dimension"]),
            str(item["target_mode"]),
            float(item["pair_weight"]),
            float(item["alpha"]),
        ),
    )
    return {
        "selection_objective": (
            "model-id CV: scenario_equal_log_rmse + "
            "0.5*(1-scenario_equal_pairwise_accuracy) + "
            "scenario_top1_regret + 0.5*fixed_gpu_top1_regret"
        ),
        "candidate_count": len(evaluated),
        "selected": selected,
        "candidates": evaluated,
    }


def _select_h800(
    historical: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    evaluated = []
    for target_mode in TARGET_MODES:
        for feature_set in FEATURE_SETS:
            for alpha in ALPHA_GRID:
                for pair_weight in PAIR_WEIGHT_GRID:
                    cv = _cross_validate_h800_calibration(
                        historical,
                        calibration,
                        feature_set=feature_set,
                        target_mode=target_mode,
                        alpha=alpha,
                        pair_weight=pair_weight,
                        historical_weight=(
                            H800_HISTORICAL_SCENARIO_WEIGHT
                        ),
                    )
                    metrics = cv["aggregate"]
                    evaluated.append(
                        {
                            "target_mode": target_mode,
                            "feature_set": feature_set,
                            "feature_dimension": len(
                                FEATURE_SETS[feature_set]
                            ),
                            "alpha": alpha,
                            "pair_weight": pair_weight,
                            "objective": _selection_objective(metrics),
                            "metrics": metrics,
                        }
                    )
    selected = min(
        evaluated,
        key=lambda item: (
            float(item["objective"]),
            float(
                item["metrics"][
                    "scenario_equal_absolute_log_rmse"
                ]
            ),
            -float(
                item["metrics"].get(
                    "scenario_equal_pairwise_accuracy"
                )
                or 0.0
            ),
            int(item["feature_dimension"]),
            str(item["target_mode"]),
            float(item["pair_weight"]),
            float(item["alpha"]),
        ),
    )
    return {
        "selection_objective": (
            "scenario_equal_log_rmse + 0.5*(1-scenario_equal_pairwise_"
            "accuracy) + scenario_top1_regret + 0.5*fixed_gpu_top1_regret"
        ),
        "candidate_count": len(evaluated),
        "selected": selected,
        "candidates": evaluated,
    }


def _generic_candidate_folds(
    candidates: Sequence[Mapping[str, Any]],
    fold_count: int,
) -> list[dict[str, Any]]:
    result = []
    for index, held_out in enumerate(
        _scenario_folds(candidates, fold_count)
    ):
        result.append(
            {
                "fold_id": index,
                "held_out_scenario_ids": sorted(held_out),
                "train": [
                    candidate
                    for candidate in candidates
                    if str(candidate["scenario_id"]) not in held_out
                ],
                "test": [
                    candidate
                    for candidate in candidates
                    if str(candidate["scenario_id"]) in held_out
                ],
            }
        )
    return result


def _model_candidate_folds(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    model_ids = sorted(
        {
            str(candidate["record"]["scenario"]["model_id"])
            for candidate in candidates
        }
    )
    if len(model_ids) < 2:
        raise ValueError("Two-head model-id CV needs at least two models")
    return [
        {
            "held_out_model_id": held_model,
            "train": [
                candidate
                for candidate in candidates
                if str(candidate["record"]["scenario"]["model_id"])
                != held_model
            ],
            "test": [
                candidate
                for candidate in candidates
                if str(candidate["record"]["scenario"]["model_id"])
                == held_model
            ],
        }
        for held_model in model_ids
    ]


def _h800_candidate_folds(
    historical: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    population = [*historical, *calibration]
    calibration_scenarios = sorted(
        {str(candidate["scenario_id"]) for candidate in calibration}
    )
    return [
        {
            "held_out_scenario_id": held_out,
            "train": [
                candidate
                for candidate in population
                if str(candidate["scenario_id"]) != held_out
            ],
            "test": [
                candidate
                for candidate in calibration
                if str(candidate["scenario_id"]) == held_out
            ],
        }
        for held_out in calibration_scenarios
    ]


def _cross_validate_two_head_folds(
    folds: Sequence[Mapping[str, Any]],
    *,
    absolute_settings: Mapping[str, Any],
    rank_feature_set: str,
    rank_alpha: float,
    absolute_deviation_blend: float,
    historical_weight: float,
) -> dict[str, Any]:
    entries: list[tuple[Mapping[str, Any], float]] = []
    fold_reports = []
    for fold in folds:
        train = fold["train"]
        test = fold["test"]
        model = _fit_two_head_model(
            train,
            absolute_settings=absolute_settings,
            rank_feature_set=rank_feature_set,
            rank_alpha=rank_alpha,
            absolute_deviation_blend=absolute_deviation_blend,
            historical_weight=historical_weight,
        )
        fold_entries = _predict_two_head_entries(test, model)
        entries.extend(fold_entries)
        identity = {
            key: value
            for key, value in fold.items()
            if key not in {"train", "test"}
        }
        fold_reports.append(
            {
                **identity,
                "train_candidates": len(train),
                "test_candidates": len(test),
                "metrics": _evaluate_entries(fold_entries),
            }
        )
    return {
        "folds": fold_reports,
        "aggregate": _evaluate_entries(entries),
    }


def _select_two_head(
    folds: Sequence[Mapping[str, Any]],
    *,
    absolute_settings: Mapping[str, Any],
    historical_weight: float,
    protocol: str,
) -> dict[str, Any]:
    evaluated = []
    for feature_set in FEATURE_SETS:
        for alpha in ALPHA_GRID:
            for blend in ABSOLUTE_DEVIATION_BLEND_GRID:
                cv = _cross_validate_two_head_folds(
                    folds,
                    absolute_settings=absolute_settings,
                    rank_feature_set=feature_set,
                    rank_alpha=alpha,
                    absolute_deviation_blend=blend,
                    historical_weight=historical_weight,
                )
                metrics = cv["aggregate"]
                evaluated.append(
                    {
                        "rank_feature_set": feature_set,
                        "rank_feature_dimension": len(
                            FEATURE_SETS[feature_set]
                        ),
                        "rank_alpha": alpha,
                        "absolute_deviation_blend": blend,
                        "objective": _selection_objective(metrics),
                        "metrics": metrics,
                    }
                )
    selected = min(
        evaluated,
        key=lambda item: (
            float(item["objective"]),
            float(
                item["metrics"][
                    "scenario_equal_absolute_log_rmse"
                ]
            ),
            -float(
                item["metrics"].get(
                    "scenario_equal_pairwise_accuracy"
                )
                or 0.0
            ),
            int(item["rank_feature_dimension"]),
            float(item["absolute_deviation_blend"]),
            float(item["rank_alpha"]),
        ),
    )
    return {
        "protocol": protocol,
        "selection_objective": (
            "scenario_equal_log_rmse + 0.5*(1-scenario_equal_pairwise_"
            "accuracy) + scenario_top1_regret + 0.5*fixed_gpu_top1_regret"
        ),
        "absolute_head_settings_frozen_before_rank_head_selection": {
            key: absolute_settings[key]
            for key in (
                "target_mode",
                "feature_set",
                "feature_dimension",
                "alpha",
                "pair_weight",
            )
        },
        "candidate_count": len(evaluated),
        "selected": selected,
        "candidates": evaluated,
    }


def _load_h800(
    *,
    observation_path: Path,
    theory_basis_path: Path,
    inventory_path: Path,
    hardware_path: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    observations = _read_observations(observation_path)
    theory_basis = read_json(theory_basis_path)
    inventory = read_json(inventory_path)
    hardware = read_json(hardware_path)
    _verify_bound_report(
        theory_basis,
        expected_schema="sft_h800_theory_basis/v1",
        name="H800 theory basis",
    )
    if "h800" not in str(
        hardware.get("name_reported_by_driver") or ""
    ).lower():
        raise ValueError("H800 joint-model input is not H800")
    model_by_id, fixed_lora = _inventory_models(inventory)
    admission = Counter()
    native_records = []
    for row in observations:
        reason = throughput_admission_reason(row)
        admission[reason] += 1
        if reason == "admitted":
            native_records.append(
                _build_native_throughput_record(
                    row,
                    model_by_id=model_by_id,
                    fixed_lora=fixed_lora,
                    hardware=hardware,
                    runtime_root=runtime_root,
                )
            )
    native_records.sort(key=_observation_id)
    split = _same_split_audit(native_records)
    if split["disjoint"] is not True:
        raise ValueError("H800 calibration/holdout split overlaps")
    calibration_records = [
        record
        for record in native_records
        if record["calibration_partition"]["role"] == "calibration"
    ]
    holdout_records = [
        record
        for record in native_records
        if record["calibration_partition"]["role"] == "holdout"
    ]
    basis_records = theory_basis.get("records")
    if not isinstance(basis_records, list):
        raise ValueError("H800 theory basis has no records")
    historical_records = [
        record
        for record in basis_records
        if isinstance(record, Mapping)
        and (record.get("route") or {}).get("throughput_primary") is True
        and _outcome(record) == "success"
    ]
    historical_candidates = _pool_candidates(historical_records)
    calibration_candidates = _pool_candidates(calibration_records)
    holdout_candidates = _pool_candidates(holdout_records)
    holdout_scenarios = {
        str(candidate["scenario_id"]) for candidate in holdout_candidates
    }
    strict_historical = [
        candidate
        for candidate in historical_candidates
        if str(candidate["scenario_id"]) not in holdout_scenarios
    ]
    strict_fit = [*strict_historical, *calibration_candidates]
    strict_overlap = sorted(
        {
            str(candidate["scenario_id"]) for candidate in strict_fit
        }
        & holdout_scenarios
    )
    if strict_overlap:
        raise ValueError("Strict H800 fit still overlaps holdout scenarios")
    return {
        "observations": observations,
        "admission": dict(sorted(admission.items())),
        "split": split,
        "historical_records": historical_records,
        "calibration_records": calibration_records,
        "holdout_records": holdout_records,
        "historical_candidates": historical_candidates,
        "strict_historical_candidates": strict_historical,
        "calibration_candidates": calibration_candidates,
        "holdout_candidates": holdout_candidates,
        "strict_fit_candidates": strict_fit,
        "strict_fit_scenario_overlap_with_holdout": strict_overlap,
    }


def _h800_analysis(
    data: Mapping[str, Any],
    *,
    hybrid_report: Mapping[str, Any],
    pure_report: Mapping[str, Any],
) -> dict[str, Any]:
    strict_historical = data["strict_historical_candidates"]
    calibration = data["calibration_candidates"]
    holdout = data["holdout_candidates"]
    selection = _select_h800(strict_historical, calibration)
    selected = selection["selected"]
    two_head_selection = _select_two_head(
        _h800_candidate_folds(strict_historical, calibration),
        absolute_settings=selected,
        historical_weight=H800_HISTORICAL_SCENARIO_WEIGHT,
        protocol="native calibration leave-complete-scenario-out",
    )
    two_head_selected = two_head_selection["selected"]
    fit_candidates = data["strict_fit_candidates"]
    single_head_model = _fit_joint_model(
        fit_candidates,
        feature_set=str(selected["feature_set"]),
        target_mode=str(selected["target_mode"]),
        alpha=float(selected["alpha"]),
        pair_weight=float(selected["pair_weight"]),
        historical_weight=H800_HISTORICAL_SCENARIO_WEIGHT,
    )
    two_head_model = _fit_two_head_model(
        fit_candidates,
        absolute_settings=selected,
        rank_feature_set=str(two_head_selected["rank_feature_set"]),
        rank_alpha=float(two_head_selected["rank_alpha"]),
        absolute_deviation_blend=float(
            two_head_selected["absolute_deviation_blend"]
        ),
        historical_weight=H800_HISTORICAL_SCENARIO_WEIGHT,
    )
    single_head_evaluation = _evaluate_entries(
        [
            (
                candidate,
                _predict_log_throughput(candidate, single_head_model),
            )
            for candidate in holdout
        ]
    )
    two_head_evaluation = _evaluate_entries(
        _predict_two_head_entries(holdout, two_head_model)
    )
    old = hybrid_report["throughput"]["same_native_holdout_comparison"]
    pure = pure_report["evaluation"]["same_native_holdout_comparison"]
    return {
        "protocol": (
            "select on native calibration LOSO after removing every "
            "historical scenario that overlaps native holdout; freeze on "
            "strict historical plus calibration; evaluate native holdout"
        ),
        "selection": selection,
        "two_head_selection": two_head_selection,
        "frozen_model": single_head_model,
        "frozen_two_head_challenger": two_head_model,
        "holdout": {
            "joint_single_head": single_head_evaluation,
            "joint_two_head": two_head_evaluation,
            "existing_physics_baseline": old["physics_baseline"],
            "existing_hybrid_ranker": old["pairwise_challenger"],
            "existing_pure_scenario_clean_ranker": pure[
                "pure_scenario_clean_ranker"
            ],
            "same_candidate_pair_signature": {
                "candidate_rows": two_head_evaluation["candidate_rows"],
                "pairwise_rows": two_head_evaluation["pairwise_rows"],
                "comparable_scenario_rows": two_head_evaluation[
                    "comparable_scenario_rows"
                ],
                "gpu_group_rows": two_head_evaluation["gpu_group_rows"],
            },
        },
        "training": {
            "strict_historical_candidates": len(strict_historical),
            "native_calibration_candidates": len(calibration),
            "fit_candidates": len(fit_candidates),
            "holdout_candidates": len(holdout),
            "fit_scenario_overlap_with_holdout": data[
                "strict_fit_scenario_overlap_with_holdout"
            ],
        },
    }


def _baseline_4090_fold(
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    *,
    priors: Mapping[str, Any],
) -> dict[str, list[tuple[Mapping[str, Any], float]]]:
    physical_model = _fit_compute_model(train_records, priors)
    if physical_model.get("available") is not True:
        raise ValueError("4090 fold physical model unavailable")
    train_candidates = _throughput_candidates(
        train_records, physical_model=physical_model, priors=priors
    )
    test_candidates = _throughput_candidates(
        test_records, physical_model=physical_model, priors=priors
    )
    hybrid = _fit_pairwise_ranker(
        train_candidates,
        feature_set="physical",
        alpha=RTX4090_BASELINE_ALPHA,
        historical_weight=0.0,
        use_physics_base=True,
    )
    pure = _fit_pairwise_ranker(
        train_candidates,
        feature_set="basic",
        alpha=RTX4090_BASELINE_ALPHA,
        historical_weight=0.0,
        use_physics_base=False,
    )
    return {
        "physical_baseline": [
            (
                candidate,
                _physics_throughput_score(
                    candidate["record"], physical_model, priors
                ),
            )
            for candidate in test_candidates
        ],
        "hybrid_pairwise": [
            (candidate, _ranker_score(candidate, hybrid))
            for candidate in test_candidates
        ],
        "pure_pairwise": [
            (candidate, _ranker_score(candidate, pure))
            for candidate in test_candidates
        ],
    }


def _nested_4090_scenario_cv(
    records: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    priors: Mapping[str, Any],
) -> dict[str, Any]:
    outer_folds = _scenario_folds(candidates, 5)
    entries: dict[
        str, list[tuple[Mapping[str, Any], float]]
    ] = defaultdict(list)
    fold_reports = []
    for fold_index, held_out in enumerate(outer_folds):
        train_candidates = [
            candidate
            for candidate in candidates
            if str(candidate["scenario_id"]) not in held_out
        ]
        test_candidates = [
            candidate
            for candidate in candidates
            if str(candidate["scenario_id"]) in held_out
        ]
        selection = _select_generic(
            train_candidates, historical_weight=0.0, fold_count=4
        )
        selected = selection["selected"]
        two_head_selection = _select_two_head(
            _generic_candidate_folds(train_candidates, 4),
            absolute_settings=selected,
            historical_weight=0.0,
            protocol="inner deterministic scenario-balanced 4-fold CV",
        )
        two_head_selected = two_head_selection["selected"]
        single_head_model = _fit_joint_model(
            train_candidates,
            feature_set=str(selected["feature_set"]),
            target_mode=str(selected["target_mode"]),
            alpha=float(selected["alpha"]),
            pair_weight=float(selected["pair_weight"]),
            historical_weight=0.0,
        )
        two_head_model = _fit_two_head_model(
            train_candidates,
            absolute_settings=selected,
            rank_feature_set=str(
                two_head_selected["rank_feature_set"]
            ),
            rank_alpha=float(two_head_selected["rank_alpha"]),
            absolute_deviation_blend=float(
                two_head_selected["absolute_deviation_blend"]
            ),
            historical_weight=0.0,
        )
        single_head_entries = [
            (
                candidate,
                _predict_log_throughput(candidate, single_head_model),
            )
            for candidate in test_candidates
        ]
        two_head_entries = _predict_two_head_entries(
            test_candidates, two_head_model
        )
        entries["joint_single_head"].extend(single_head_entries)
        entries["joint_two_head"].extend(two_head_entries)

        train_records = [
            record
            for record in records
            if scenario_id(record) not in held_out
        ]
        test_records = [
            record
            for record in records
            if scenario_id(record) in held_out
        ]
        baselines = _baseline_4090_fold(
            train_records, test_records, priors=priors
        )
        for name, current_entries in baselines.items():
            entries[name].extend(current_entries)
        fold_reports.append(
            {
                "fold": fold_index,
                "held_out_scenario_ids": sorted(held_out),
                "train_candidates": len(train_candidates),
                "test_candidates": len(test_candidates),
                "selected_alpha": selected["alpha"],
                "selected_pair_weight": selected["pair_weight"],
                "selected_target_mode": selected["target_mode"],
                "selected_feature_set": selected["feature_set"],
                "inner_selection_objective": selected["objective"],
                "selected_rank_feature_set": two_head_selected[
                    "rank_feature_set"
                ],
                "selected_rank_alpha": two_head_selected["rank_alpha"],
                "selected_absolute_deviation_blend": two_head_selected[
                    "absolute_deviation_blend"
                ],
                "inner_two_head_selection_objective": two_head_selected[
                    "objective"
                ],
                "joint_single_head_metrics": _evaluate_entries(
                    single_head_entries
                ),
                "joint_two_head_metrics": _evaluate_entries(
                    two_head_entries
                ),
            }
        )
    return {
        "method": (
            "nested deterministic scenario-balanced 5-fold outer CV; "
            "each outer training population uses 4-fold scenario CV for "
            "absolute-head selection, then conditional 4-fold CV for "
            "rank-head and head-blend selection"
        ),
        "metrics": {
            name: _evaluate_entries(current_entries)
            for name, current_entries in entries.items()
        },
        "folds": fold_reports,
    }


def _nested_4090_model_holdout(
    records: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    priors: Mapping[str, Any],
) -> dict[str, Any]:
    model_ids = sorted(
        {str(record["scenario"]["model_id"]) for record in records}
    )
    entries: dict[
        str, list[tuple[Mapping[str, Any], float]]
    ] = defaultdict(list)
    folds = []
    for held_model in model_ids:
        train_candidates = [
            candidate
            for candidate in candidates
            if str(candidate["record"]["scenario"]["model_id"])
            != held_model
        ]
        test_candidates = [
            candidate
            for candidate in candidates
            if str(candidate["record"]["scenario"]["model_id"])
            == held_model
        ]
        selection = _select_model_generalization(
            train_candidates, historical_weight=0.0
        )
        selected = selection["selected"]
        two_head_selection = _select_two_head(
            _model_candidate_folds(train_candidates),
            absolute_settings=selected,
            historical_weight=0.0,
            protocol="inner leave-one-complete-model-id-out",
        )
        two_head_selected = two_head_selection["selected"]
        single_head_model = _fit_joint_model(
            train_candidates,
            feature_set=str(selected["feature_set"]),
            target_mode=str(selected["target_mode"]),
            alpha=float(selected["alpha"]),
            pair_weight=float(selected["pair_weight"]),
            historical_weight=0.0,
        )
        two_head_model = _fit_two_head_model(
            train_candidates,
            absolute_settings=selected,
            rank_feature_set=str(
                two_head_selected["rank_feature_set"]
            ),
            rank_alpha=float(two_head_selected["rank_alpha"]),
            absolute_deviation_blend=float(
                two_head_selected["absolute_deviation_blend"]
            ),
            historical_weight=0.0,
        )
        single_head_entries = [
            (
                candidate,
                _predict_log_throughput(candidate, single_head_model),
            )
            for candidate in test_candidates
        ]
        two_head_entries = _predict_two_head_entries(
            test_candidates, two_head_model
        )
        entries["joint_single_head"].extend(single_head_entries)
        entries["joint_two_head"].extend(two_head_entries)

        train_records = [
            record
            for record in records
            if str(record["scenario"]["model_id"]) != held_model
        ]
        test_records = [
            record
            for record in records
            if str(record["scenario"]["model_id"]) == held_model
        ]
        baselines = _baseline_4090_fold(
            train_records, test_records, priors=priors
        )
        for name, current_entries in baselines.items():
            entries[name].extend(current_entries)
        folds.append(
            {
                "held_out_model_id": held_model,
                "train_candidates": len(train_candidates),
                "test_candidates": len(test_candidates),
                "selected_alpha": selected["alpha"],
                "selected_pair_weight": selected["pair_weight"],
                "selected_target_mode": selected["target_mode"],
                "selected_feature_set": selected["feature_set"],
                "selected_rank_feature_set": two_head_selected[
                    "rank_feature_set"
                ],
                "selected_rank_alpha": two_head_selected["rank_alpha"],
                "selected_absolute_deviation_blend": two_head_selected[
                    "absolute_deviation_blend"
                ],
                "joint_single_head_metrics": _evaluate_entries(
                    single_head_entries
                ),
                "joint_two_head_metrics": _evaluate_entries(
                    two_head_entries
                ),
            }
        )
    return {
        "method": (
            "leave-one-complete-model-id-out; each outer training "
            "population selects the absolute head and then the rank head "
            "by inner leave-one-model-id-out over remaining model ids"
        ),
        "metrics": {
            name: _evaluate_entries(current_entries)
            for name, current_entries in entries.items()
        },
        "folds": folds,
    }


def _4090_analysis(
    *,
    campaign_root: Path,
) -> dict[str, Any]:
    records, admission = build_4090_records(
        campaign_root,
        matrix_name="throughput_jobs.jsonl",
        hbm_bandwidth_bytes_s=DEFAULT_HBM_BANDWIDTH_BYTES_S,
        collective_bandwidth_bytes_s=(
            DEFAULT_COLLECTIVE_BANDWIDTH_BYTES_S
        ),
    )
    candidates = _pool_candidates(records)
    hardware = read_json(campaign_root / "config" / "hardware.json")
    priors = physical_4090_priors(
        hardware,
        hbm_bandwidth_bytes_s=DEFAULT_HBM_BANDWIDTH_BYTES_S,
        collective_bandwidth_bytes_s=(
            DEFAULT_COLLECTIVE_BANDWIDTH_BYTES_S
        ),
    )
    nested_scenario = _nested_4090_scenario_cv(
        records, candidates, priors=priors
    )
    model_holdout = _nested_4090_model_holdout(
        records, candidates, priors=priors
    )
    final_selection = _select_generic(
        candidates, historical_weight=0.0, fold_count=5
    )
    selected = final_selection["selected"]
    final_two_head_selection = _select_two_head(
        _generic_candidate_folds(candidates, 5),
        absolute_settings=selected,
        historical_weight=0.0,
        protocol="deterministic scenario-balanced 5-fold CV",
    )
    two_head_selected = final_two_head_selection["selected"]
    final_single_head_model = _fit_joint_model(
        candidates,
        feature_set=str(selected["feature_set"]),
        target_mode=str(selected["target_mode"]),
        alpha=float(selected["alpha"]),
        pair_weight=float(selected["pair_weight"]),
        historical_weight=0.0,
    )
    final_model = _fit_two_head_model(
        candidates,
        absolute_settings=selected,
        rank_feature_set=str(two_head_selected["rank_feature_set"]),
        rank_alpha=float(two_head_selected["rank_alpha"]),
        absolute_deviation_blend=float(
            two_head_selected["absolute_deviation_blend"]
        ),
        historical_weight=0.0,
    )
    return {
        "admission": admission,
        "records": len(records),
        "candidates": len(candidates),
        "nested_scenario_cv": nested_scenario,
        "nested_model_holdout": model_holdout,
        "final_selection": final_selection,
        "final_two_head_selection": final_two_head_selection,
        "frozen_model": final_single_head_model,
        "frozen_two_head_challenger": final_model,
        "in_sample_diagnostic": {
            "joint_single_head": _evaluate_entries(
                [
                    (
                        candidate,
                        _predict_log_throughput(
                            candidate, final_single_head_model
                        ),
                    )
                    for candidate in candidates
                ]
            ),
            "joint_two_head": _evaluate_entries(
                _predict_two_head_entries(candidates, final_model)
            ),
        },
    }


def build_report(
    *,
    h800_observations_path: Path,
    h800_theory_basis_path: Path,
    h800_inventory_path: Path,
    h800_hardware_path: Path,
    h800_runtime_root: Path,
    h800_hybrid_report_path: Path,
    h800_pure_report_path: Path,
    rtx4090_campaign_root: Path,
    rtx4090_report_path: Path,
) -> dict[str, Any]:
    h800_hybrid = read_json(h800_hybrid_report_path)
    h800_pure = read_json(h800_pure_report_path)
    rtx4090_report = read_json(rtx4090_report_path)
    _verify_bound_report(
        h800_hybrid,
        expected_schema=H800_HYBRID_SCHEMA,
        name="H800 hybrid report",
    )
    _verify_bound_report(
        h800_pure,
        expected_schema="sft_h800_pure_ranker_modeling/v1",
        name="H800 pure report",
    )
    if rtx4090_report.get("schema") != "sft_rtx4090_challenger_modeling/v1":
        raise ValueError("RTX 4090 comparison report schema mismatch")

    h800_data = _load_h800(
        observation_path=h800_observations_path,
        theory_basis_path=h800_theory_basis_path,
        inventory_path=h800_inventory_path,
        hardware_path=h800_hardware_path,
        runtime_root=h800_runtime_root,
    )
    h800 = _h800_analysis(
        h800_data,
        hybrid_report=h800_hybrid,
        pure_report=h800_pure,
    )
    rtx4090 = _4090_analysis(campaign_root=rtx4090_campaign_root)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "joint_full_factor_models_fitted_and_compared",
        "analysis_only": True,
        "publishable": False,
        "production_profile_generated": False,
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "model_contract": {
            "target": "effective_tokens_per_second",
            "predicted_latent": "log_step_seconds",
            "formula": (
                "log(throughput_hat) = log(effective_tokens_per_step) "
                "- (optional_log_analytic_anchor + intercept + "
                "standardized_selected_factors @ beta)"
            ),
            "training_objective": (
                "scenario-balanced robust absolute log-step loss + "
                "pair_weight * scenario-balanced robust pairwise log-step-"
                "difference loss + ridge regularization"
            ),
            "available_feature_dimension": len(FEATURE_NAMES),
            "available_feature_names": list(FEATURE_NAMES),
            "candidate_feature_sets": {
                name: list(feature_names)
                for name, feature_names in FEATURE_SETS.items()
            },
            "candidate_target_modes": list(TARGET_MODES),
            "set_aware_prediction": False,
            "two_head_set_aware_challenger_evaluated": True,
            "forced_physics_baseline": False,
            "physical_quantities_used_as_features": True,
            "runtime_outcome_features_used": False,
        },
        "selection": {
            "feature_sets": {
                name: len(feature_names)
                for name, feature_names in FEATURE_SETS.items()
            },
            "target_modes": list(TARGET_MODES),
            "alpha_grid": list(ALPHA_GRID),
            "pair_weight_grid": list(PAIR_WEIGHT_GRID),
            "absolute_deviation_blend_grid": list(
                ABSOLUTE_DEVIATION_BLEND_GRID
            ),
            "huber_delta_log_seconds": HUBER_DELTA,
            "objective": (
                "scenario_equal_log_rmse + 0.5*(1-scenario_equal_"
                "pairwise_accuracy) + scenario_top1_regret + "
                "0.5*fixed_gpu_top1_regret"
            ),
        },
        "model_decision": {
            "primary": "joint_single_head",
            "challenger": "joint_two_head",
            "reason": (
                "the two-head decomposition improved selected ranking "
                "slices but did not consistently improve nested scenario "
                "CV, cross-model holdout, absolute error and top-1 regret "
                "together; retain the simpler joint-loss model as primary"
            ),
        },
        "source_bindings": {
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "h800_observations": {
                "path": str(h800_observations_path.resolve()),
                "sha256": sha256_file(h800_observations_path),
            },
            "h800_theory_basis": {
                "path": str(h800_theory_basis_path.resolve()),
                "sha256": sha256_file(h800_theory_basis_path),
                "report_sha256": read_json(h800_theory_basis_path)[
                    "report_sha256"
                ],
            },
            "h800_model_inventory": {
                "path": str(h800_inventory_path.resolve()),
                "sha256": sha256_file(h800_inventory_path),
            },
            "h800_hardware": {
                "path": str(h800_hardware_path.resolve()),
                "sha256": sha256_file(h800_hardware_path),
            },
            "h800_hybrid_report": {
                "path": str(h800_hybrid_report_path.resolve()),
                "sha256": sha256_file(h800_hybrid_report_path),
                "report_sha256": h800_hybrid["report_sha256"],
            },
            "h800_pure_report": {
                "path": str(h800_pure_report_path.resolve()),
                "sha256": sha256_file(h800_pure_report_path),
                "report_sha256": h800_pure["report_sha256"],
            },
            "rtx4090_matrix": {
                "path": str(
                    (
                        rtx4090_campaign_root
                        / "matrix"
                        / "throughput_jobs.jsonl"
                    ).resolve()
                ),
                "sha256": sha256_file(
                    rtx4090_campaign_root
                    / "matrix"
                    / "throughput_jobs.jsonl"
                ),
            },
            "rtx4090_hardware": {
                "path": str(
                    (
                        rtx4090_campaign_root
                        / "config"
                        / "hardware.json"
                    ).resolve()
                ),
                "sha256": sha256_file(
                    rtx4090_campaign_root
                    / "config"
                    / "hardware.json"
                ),
            },
            "rtx4090_comparison_report": {
                "path": str(rtx4090_report_path.resolve()),
                "sha256": sha256_file(rtx4090_report_path),
            },
        },
        "h800": {
            "data_admission": {
                "counts": h800_data["admission"],
                "split": h800_data["split"],
                "historical_records": len(
                    h800_data["historical_records"]
                ),
                "native_calibration_records": len(
                    h800_data["calibration_records"]
                ),
                "native_holdout_records": len(
                    h800_data["holdout_records"]
                ),
            },
            **h800,
        },
        "rtx4090": rtx4090,
        "limitations": [
            "retrospective_analysis_not_prospective_publication_acceptance",
            "packing_effects_are_excluded_by_current_throughput_admission",
            "work_per_step_profile_features_currently_use_observed_run_counters_and_must_be_replaced_by_pre_run_dataset_profile_expectations",
            "rtx4090_only_covers_qwen3_0p6b_1p7b_4b",
            "h800_holdout_was_previously_inspected",
            "runtime_thermal_power_status_is_not_used_as_a_feature_but_can_add_irreducible_measurement_noise",
            "new_gpu_families_require_calibration_or_multi-card_training",
        ],
    }
    report["report_sha256"] = sha256_json(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema") != SCHEMA:
        raise ValueError("Joint throughput report schema mismatch")
    unsigned = dict(report)
    digest = unsigned.pop("report_sha256", None)
    if digest != sha256_json(unsigned):
        raise ValueError("Joint throughput report SHA-256 mismatch")
    if (
        report.get("gpu_experiments_launched") is not False
        or report.get("queues_mutated") is not False
        or report.get("publishable") is not False
        or report.get("production_profile_generated") is not False
    ):
        raise ValueError("Joint throughput report has unsafe state flags")
    contract = report.get("model_contract") or {}
    if (
        contract.get("available_feature_dimension") != len(FEATURE_NAMES)
        or contract.get("set_aware_prediction") is not False
        or contract.get("two_head_set_aware_challenger_evaluated")
        is not True
        or contract.get("forced_physics_baseline") is not False
        or contract.get("runtime_outcome_features_used") is not False
    ):
        raise ValueError("Joint throughput model contract drifted")

    def validate_head(
        model: Mapping[str, Any],
        *,
        require_target_mode: bool,
        name: str,
    ) -> None:
        feature_names = model.get("feature_names") or []
        if (
            model.get("feature_set") not in FEATURE_SETS
            or tuple(feature_names)
            != FEATURE_SETS[str(model.get("feature_set"))]
            or len(model.get("coefficients") or []) != len(feature_names)
        ):
            raise ValueError(f"{name} dimension drifted")
        if (
            require_target_mode
            and model.get("target_mode") not in TARGET_MODES
        ):
            raise ValueError(f"{name} target mode drifted")

    for card in ("h800", "rtx4090"):
        section = report.get(card) or {}
        model = section.get("frozen_model") or {}
        validate_head(
            model,
            require_target_mode=True,
            name=f"{card} frozen joint model",
        )
        challenger = section.get("frozen_two_head_challenger") or {}
        if (
            challenger.get("model_family")
            != "set_aware_two_head_log_throughput_model"
            or not (
                0.0
                <= float(challenger.get("absolute_deviation_blend"))
                <= 1.0
            )
        ):
            raise ValueError(f"{card} two-head challenger contract drifted")
        validate_head(
            challenger.get("absolute_head") or {},
            require_target_mode=True,
            name=f"{card} challenger absolute head",
        )
        validate_head(
            challenger.get("rank_head") or {},
            require_target_mode=False,
            name=f"{card} challenger rank head",
        )
    overlap = (
        ((report.get("h800") or {}).get("training") or {}).get(
            "fit_scenario_overlap_with_holdout"
        )
        or []
    )
    if overlap:
        raise ValueError("H800 joint fit overlaps holdout scenarios")


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "candidate_rows",
        "scenario_rows",
        "comparable_scenario_rows",
        "pairwise_rows",
        "pairwise_correct_rows",
        "pooled_pairwise_accuracy",
        "scenario_equal_pairwise_accuracy",
        "scenario_equal_top1_regret",
        "scenario_equal_hit_at_10_percent",
        "gpu_group_rows",
        "scenario_gpu_equal_top1_regret",
        "scenario_gpu_equal_hit_at_10_percent",
        "absolute_throughput_mape",
        "scenario_equal_absolute_throughput_mape",
        "absolute_throughput_ape_median",
        "absolute_throughput_ape_p90",
        "absolute_log_error_rmse",
        "scenario_equal_absolute_log_rmse",
    )
    return {field: metrics.get(field) for field in fields}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--h800-hybrid-report",
        type=Path,
        default=ROOT / "artifacts" / "h800_challenger_modeling.json",
    )
    parser.add_argument(
        "--h800-pure-report",
        type=Path,
        default=ROOT / "artifacts" / "h800_pure_ranker_modeling.json",
    )
    parser.add_argument(
        "--rtx4090-campaign-root",
        type=Path,
        default=ROOT / "campaigns" / "rtx4090_20260717",
    )
    parser.add_argument(
        "--rtx4090-report",
        type=Path,
        default=(
            ROOT
            / "campaigns"
            / "rtx4090_20260717"
            / "artifacts"
            / "rtx4090_challenger_modeling.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "joint_throughput_modeling.json",
    )
    args = parser.parse_args()
    report = build_report(
        h800_observations_path=args.h800_observations,
        h800_theory_basis_path=args.h800_theory_basis,
        h800_inventory_path=args.h800_model_inventory,
        h800_hardware_path=args.h800_hardware,
        h800_runtime_root=args.h800_runtime_root,
        h800_hybrid_report_path=args.h800_hybrid_report,
        h800_pure_report_path=args.h800_pure_report,
        rtx4090_campaign_root=args.rtx4090_campaign_root,
        rtx4090_report_path=args.rtx4090_report,
    )
    write_json(args.output, report)
    h800_holdout = report["h800"]["holdout"]
    rtx_scenario = report["rtx4090"]["nested_scenario_cv"]["metrics"]
    rtx_model = report["rtx4090"]["nested_model_holdout"]["metrics"]

    def selected_summary(selected: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "target_mode": selected["target_mode"],
            "feature_set": selected["feature_set"],
            "feature_dimension": selected["feature_dimension"],
            "alpha": selected["alpha"],
            "pair_weight": selected["pair_weight"],
            "objective": selected["objective"],
            "metrics": _metric_summary(selected["metrics"]),
        }

    def two_head_selected_summary(
        selected: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "rank_feature_set": selected["rank_feature_set"],
            "rank_feature_dimension": selected[
                "rank_feature_dimension"
            ],
            "rank_alpha": selected["rank_alpha"],
            "absolute_deviation_blend": selected[
                "absolute_deviation_blend"
            ],
            "objective": selected["objective"],
            "metrics": _metric_summary(selected["metrics"]),
        }

    print(
        json.dumps(
            {
                "schema": report["schema"],
                "output": str(args.output),
                "h800_holdout": {
                    name: _metric_summary(metrics)
                    for name, metrics in h800_holdout.items()
                    if isinstance(metrics, Mapping)
                    and "candidate_rows" in metrics
                },
                "rtx4090_nested_scenario_cv": {
                    name: _metric_summary(metrics)
                    for name, metrics in rtx_scenario.items()
                },
                "rtx4090_nested_model_holdout": {
                    name: _metric_summary(metrics)
                    for name, metrics in rtx_model.items()
                },
                "selected_hyperparameters": {
                    "h800": selected_summary(
                        report["h800"]["selection"]["selected"]
                    ),
                    "h800_two_head": two_head_selected_summary(
                        report["h800"]["two_head_selection"]["selected"]
                    ),
                    "rtx4090_final": selected_summary(
                        report["rtx4090"]["final_selection"]["selected"]
                    ),
                    "rtx4090_final_two_head": two_head_selected_summary(
                        report["rtx4090"][
                            "final_two_head_selection"
                        ]["selected"]
                    ),
                },
                "gpu_experiments_launched": report[
                    "gpu_experiments_launched"
                ],
                "publishable": report["publishable"],
                "report_sha256": report["report_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
