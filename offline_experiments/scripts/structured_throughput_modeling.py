#!/usr/bin/env python3
"""Fit a static-profile, structured and cross-card throughput model.

This analysis intentionally separates deterministic workload reconstruction
from learned hardware efficiency:

1. dataset token profiles plus MBS/GBS/cutoff reconstruct pre-run work;
2. model/config formulas reconstruct FLOPs, HBM traffic, communication and
   launch work;
3. a positive roofline-style time model learns shared inverse efficiencies
   with partially pooled per-card adapters;
4. a small bounded residual is trained with absolute, pairwise and listwise
   losses and still emits one effective-token/s prediction.

The script is CPU-only.  It launches no experiments and mutates no queue.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from common import ROOT, read_json, sha256_file, sha256_json, write_json
from h800_challenger_modeling import (
    _build_native_throughput_record,
    _inventory_models,
    _is_legacy,
    _observation_id,
    _outcome,
    _read_observations,
    _same_split_audit,
    _verify_bound_report,
    throughput_admission_reason,
)
from h800_theory_basis import (
    COMMUNICATION_BYTES,
    DEFAULT_ALL_GATHER_BUCKET_ELEMENTS,
    DEFAULT_PREFETCH_BUCKET_ELEMENTS,
    DEFAULT_REDUCE_BUCKET_ELEMENTS,
    GRADIENT_BYTES,
    OPTIMIZER_BYTES,
    PARAMETER_BYTES,
)
from h800_theory_calibration import (
    _candidate_key,
    _observed_step_seconds,
    _work_per_step,
    scenario_id,
    scenario_material,
)
from rtx4090_challenger_modeling import (
    DEFAULT_COLLECTIVE_BANDWIDTH_BYTES_S,
    DEFAULT_HBM_BANDWIDTH_BYTES_S,
    _build_records as build_4090_records,
)


SCHEMA = "sft_structured_throughput_modeling/v1"
STATIC_PROFILE_SCHEMA = "sft_static_workload_profiles/v1"
IMPLEMENTATION_VERSION = (
    "sft_structured_throughput_modeling_impl/"
    "2026-08-09.static-profile-hierarchical-card-adapter-multitask-analysis-extension"
)
PROFILE_SEEDS = (20260714, 20260715, 20260716)
SMOOTH_ROOFLINE_POWER = 4.0
KERNEL_LAUNCH_PRIOR_SECONDS = 5.0e-6
CORRECTION_LOG_LIMIT = 1.25
ABSOLUTE_HUBER_DELTA = 0.25
PAIRWISE_HUBER_DELTA = 0.20
PAIRWISE_ORDER_TEMPERATURE = 0.05
LISTWISE_TEMPERATURE = 0.10
EPSILON = 1.0e-12
H800_HISTORICAL_SCENARIO_WEIGHT = 0.25

# Fixed before blocked validation.  This report is a structural-model
# diagnostic, not a hyperparameter search against the held-out populations.
LOSS_WEIGHTS = {
    "absolute": 1.0,
    "pairwise_magnitude": 0.50,
    "pairwise_order": 0.02,
    "listwise": 0.05,
    "shared_residual_ridge": 0.02,
    "card_adapter_ridge": 0.10,
    "global_efficiency_prior": 0.02,
}
INITIAL_INVERSE_EFFICIENCY = {
    "launch": 5.0,
    "compute": 3.0,
    "kernel_hbm": 2.0,
    "optimizer_hbm": 2.0,
    "communication": 2.0,
}
COMPONENT_NAMES = tuple(INITIAL_INVERSE_EFFICIENCY)

FEATURE_NAMES = (
    # Mechanisms whose cost is not fully captured by byte/FLOP formulas.
    "is_lora",
    "gradient_checkpointing",
    "zero2",
    "zero3",
    "packing",
    "kernel_fa2",
    "kernel_fa3",
    "kernel_liger",
    "kernel_fused_ce",
    "kernel_fused_optimizer",
    "kernel_compile",
    # Resource geometry.
    "log2_gpu_count",
    "log2_mbs",
    "log2_gradient_accumulation",
    "log2_cutoff_over_512",
    # Dimensionless structured-work ratios.
    "log_compute_to_hbm_ratio",
    "log1p_comm_to_roof_ratio",
    "log1p_optimizer_to_roof_ratio",
    "log1p_launch_to_roof_ratio",
    "effective_to_computed_token_ratio",
    "attention_flop_share",
    "recompute_flop_share",
    "log1p_model_state_to_hbm_capacity",
    "mean_length_to_cutoff",
    "length_cv",
    "p90_to_mean_length",
    "p99_to_mean_length",
    "label_token_ratio",
    "mean_turns_over_10",
    # Hardware ratios, not card-name dummies.
    "log1p_machine_balance_flops_per_hbm_byte",
    "log1p_link_to_hbm_bandwidth_ratio",
    # Selected mechanism interactions.
    "lora_x_gc",
    "gc_x_zero3",
    "zero3_x_log2_gpu_count",
    "log2_mbs_x_length_cv",
)
CARD_RESIDUAL_FEATURE_NAMES = (
    "is_lora",
    "gradient_checkpointing",
    "zero3",
    "log2_gpu_count",
    "log2_mbs",
    "log_compute_to_hbm_ratio",
    "log1p_comm_to_roof_ratio",
    "log1p_launch_to_roof_ratio",
)
CARD_RESIDUAL_FEATURE_INDEXES = tuple(
    FEATURE_NAMES.index(name) for name in CARD_RESIDUAL_FEATURE_NAMES
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


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


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _ceil_div(value: float, divisor: int) -> int:
    return int(math.ceil(_nonnegative(value) / int(divisor)))


def _stable_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=float) / float(temperature)
    shifted = scaled - float(np.max(scaled))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def _huber(
    residual: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.abs(residual)
    quadratic = absolute <= delta
    loss = np.where(
        quadratic,
        0.5 * residual**2,
        delta * (absolute - 0.5 * delta),
    )
    gradient = np.where(
        quadratic,
        residual,
        delta * np.sign(residual),
    )
    return loss, gradient


def _inverse_softplus(value: float) -> float:
    target = max(EPSILON, float(value) - 1.0)
    return math.log(math.expm1(target))


def _mean(values: Sequence[float]) -> float | None:
    usable = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]
    return statistics.fmean(usable) if usable else None


def _pool_candidates(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        str, dict[tuple[Any, ...], list[Mapping[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[scenario_id(record)][_candidate_key(record)].append(record)
    candidates = []
    for current_scenario, by_candidate in sorted(grouped.items()):
        for key, replicates in sorted(
            by_candidate.items(),
            key=lambda item: tuple(str(value) for value in item[0]),
        ):
            log_steps = []
            log_works = []
            log_throughputs = []
            for record in replicates:
                step = _observed_step_seconds(record)
                work = _work_per_step(record, "effective_tokens")
                if step is None or work is None or step <= 0 or work <= 0:
                    raise ValueError("Successful throughput record lacks work")
                log_step = math.log(float(step))
                log_work = math.log(float(work))
                log_steps.append(log_step)
                log_works.append(log_work)
                log_throughputs.append(log_work - log_step)
            candidates.append(
                {
                    "scenario_id": current_scenario,
                    "scenario": scenario_material(replicates[0]),
                    "candidate_key": list(key),
                    "record": replicates[0],
                    "replicates": len(replicates),
                    "observed_log_step": statistics.median(log_steps),
                    "effective_log_work": statistics.median(log_works),
                    "observed_log_throughput": statistics.median(
                        log_throughputs
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
        raise ValueError("Structured H800 input is not H800")
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
        raise ValueError("H800 calibration and holdout overlap")
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
    historical = _pool_candidates(historical_records)
    calibration = _pool_candidates(calibration_records)
    holdout = _pool_candidates(holdout_records)
    holdout_ids = {
        str(candidate["scenario_id"]) for candidate in holdout
    }
    strict_historical = [
        candidate
        for candidate in historical
        if str(candidate["scenario_id"]) not in holdout_ids
    ]
    strict_fit = [*strict_historical, *calibration]
    overlap = sorted(
        {
            str(candidate["scenario_id"]) for candidate in strict_fit
        }
        & holdout_ids
    )
    if overlap:
        raise ValueError("H800 strict fit overlaps holdout")
    return {
        "admission": dict(sorted(admission.items())),
        "split": split,
        "strict_fit_candidates": strict_fit,
        "holdout_candidates": holdout,
        "strict_fit_scenario_overlap_with_holdout": overlap,
    }


def _scenario_folds(
    candidates: Sequence[Mapping[str, Any]],
    fold_count: int,
) -> list[set[str]]:
    counts = Counter(str(candidate["scenario_id"]) for candidate in candidates)
    actual_count = min(int(fold_count), len(counts))
    if actual_count < 2:
        raise ValueError("Scenario CV needs at least two folds")
    folds = [set() for _ in range(actual_count)]
    loads = [0 for _ in range(actual_count)]
    for scenario, count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        index = min(range(actual_count), key=lambda i: (loads[i], i))
        folds[index].add(scenario)
        loads[index] += count
    return folds


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
    log_errors = []
    scenario_mapes = []
    scenario_log_rmses = []
    pooled_pair_correct = 0
    pooled_pair_rows = 0
    scenario_pair_accuracies = []
    top1_regrets = []
    top1_hits = []
    gpu_regrets = []
    gpu_hits = []
    details = []
    for current_scenario, rows in sorted(by_scenario.items()):
        current_errors = []
        current_log_errors = []
        pair_correct = 0
        pair_rows = 0
        for candidate, prediction in rows:
            observed_log = float(candidate["observed_log_throughput"])
            observed = math.exp(observed_log)
            predicted = math.exp(prediction)
            error = abs(predicted - observed) / observed
            current_errors.append(error)
            absolute_errors.append(error)
            log_error = prediction - observed_log
            current_log_errors.append(log_error)
            log_errors.append(log_error)
        scenario_mapes.append(statistics.fmean(current_errors))
        scenario_log_rmses.append(
            math.sqrt(
                statistics.fmean(value * value for value in current_log_errors)
            )
        )
        for left, right in itertools.combinations(range(len(rows)), 2):
            observed_delta = (
                float(rows[left][0]["observed_log_throughput"])
                - float(rows[right][0]["observed_log_throughput"])
            )
            if abs(observed_delta) <= 1.0e-12:
                continue
            predicted_delta = rows[left][1] - rows[right][1]
            correct = (observed_delta > 0) == (predicted_delta > 0)
            pair_correct += int(correct)
            pair_rows += 1
        pooled_pair_correct += pair_correct
        pooled_pair_rows += pair_rows
        if pair_rows:
            scenario_pair_accuracies.append(pair_correct / pair_rows)

        if len(rows) >= 2:
            oracle_candidate, _ = max(
                rows,
                key=lambda item: float(
                    item[0]["observed_log_throughput"]
                ),
            )
            selected_candidate, _ = max(rows, key=lambda item: item[1])
            regret = max(
                0.0,
                1.0
                - math.exp(
                    float(
                        selected_candidate["observed_log_throughput"]
                    )
                    - float(
                        oracle_candidate["observed_log_throughput"]
                    )
                ),
            )
            top1_regrets.append(regret)
            top1_hits.append(float(regret <= 0.10))

        by_gpu: dict[
            int, list[tuple[Mapping[str, Any], float]]
        ] = defaultdict(list)
        for candidate, prediction in rows:
            by_gpu[
                int(candidate["record"]["scenario"]["gpu_count"])
            ].append((candidate, prediction))
        for gpu_count, gpu_rows in sorted(by_gpu.items()):
            if len(gpu_rows) < 2:
                continue
            oracle_candidate, _ = max(
                gpu_rows,
                key=lambda item: float(
                    item[0]["observed_log_throughput"]
                ),
            )
            selected_candidate, _ = max(
                gpu_rows,
                key=lambda item: item[1],
            )
            regret = max(
                0.0,
                1.0
                - math.exp(
                    float(
                        selected_candidate["observed_log_throughput"]
                    )
                    - float(
                        oracle_candidate["observed_log_throughput"]
                    )
                ),
            )
            gpu_regrets.append(regret)
            gpu_hits.append(float(regret <= 0.10))
            details.append(
                {
                    "scenario_id": current_scenario,
                    "scenario": rows[0][0].get("scenario"),
                    "gpu_count": gpu_count,
                    "candidates": len(gpu_rows),
                    "selected_candidate_key": selected_candidate[
                        "candidate_key"
                    ],
                    "oracle_candidate_key": oracle_candidate[
                        "candidate_key"
                    ],
                    "top1_regret": regret,
                    "hit_at_10_percent": regret <= 0.10,
                }
            )
    comparable_scenarios = sum(
        len(rows) >= 2 for rows in by_scenario.values()
    )
    return {
        "candidate_rows": len(entries),
        "scenario_rows": len(by_scenario),
        "comparable_scenario_rows": comparable_scenarios,
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
            statistics.median(absolute_errors)
            if absolute_errors
            else None
        ),
        "absolute_throughput_ape_p90": (
            _percentile(absolute_errors, 90)
            if absolute_errors
            else None
        ),
        "absolute_log_error_rmse": (
            math.sqrt(statistics.fmean(value * value for value in log_errors))
            if log_errors
            else None
        ),
        "scenario_equal_absolute_log_rmse": _mean(
            scenario_log_rmses
        ),
        "details": details,
    }


class StaticDatasetProfiles:
    """Pre-run dataset work expectations from saved token profiles."""

    def __init__(self, profile_dir: Path) -> None:
        self.profile_dir = profile_dir
        self.rows: dict[str, list[dict[str, Any]]] = {}
        for path in sorted(profile_dir.glob("*.qwen3_nothink.jsonl")):
            dataset_id = path.name.split(".", 1)[0]
            rows = _read_jsonl(path)
            if not rows:
                raise ValueError(f"Empty dataset profile {path}")
            self.rows[dataset_id] = rows
        if not self.rows:
            raise ValueError(f"No dataset profiles under {profile_dir}")
        self._cache: dict[tuple[str, int, int, bool], dict[str, Any]] = {}

    def source_bindings(self) -> dict[str, Any]:
        return {
            dataset_id: {
                "path": str(
                    (
                        self.profile_dir
                        / f"{dataset_id}.qwen3_nothink.jsonl"
                    ).resolve()
                ),
                "sha256": sha256_file(
                    self.profile_dir
                    / f"{dataset_id}.qwen3_nothink.jsonl"
                ),
            }
            for dataset_id in sorted(self.rows)
        }

    @staticmethod
    def _packed_bins(
        lengths: Sequence[int],
        capacity: int,
    ) -> list[list[int]]:
        # Deterministic best-fit decreasing approximation.  Packing is outside
        # the current primary admitted data, so this supplies an explicit
        # pre-run contract without claiming learned packing validation.
        bins: list[list[int]] = []
        remaining: list[int] = []
        for length in sorted((min(int(x), capacity) for x in lengths), reverse=True):
            eligible = [
                (space - length, index)
                for index, space in enumerate(remaining)
                if space >= length
            ]
            if not eligible:
                bins.append([length])
                remaining.append(capacity - length)
                continue
            _, index = min(eligible)
            bins[index].append(length)
            remaining[index] -= length
        return bins

    def profile(
        self,
        dataset_id: str,
        *,
        cutoff_len: int,
        physical_mbs: int,
        packing: bool,
    ) -> dict[str, Any]:
        key = (
            str(dataset_id),
            int(cutoff_len),
            int(physical_mbs),
            bool(packing),
        )
        if key in self._cache:
            return self._cache[key]
        rows = self.rows.get(str(dataset_id))
        if rows is None:
            raise KeyError(f"No static token profile for {dataset_id}")
        cutoff = int(cutoff_len)
        mbs = int(physical_mbs)
        if cutoff <= 0 or mbs <= 0:
            raise ValueError("cutoff_len and physical_mbs must be positive")
        lengths = [
            min(cutoff, int(row.get("total_tokens") or 0))
            for row in rows
        ]
        label_lengths = [
            min(
                length,
                int(row.get("label_tokens") or 0),
            )
            for row, length in zip(rows, lengths)
        ]
        if min(lengths) <= 0:
            raise ValueError(f"{dataset_id} has nonpositive token length")

        if packing:
            capacity = max(1, cutoff - 1)
            bins = self._packed_bins(lengths, capacity)
            used = [sum(current) for current in bins]
            attention = [
                sum(length * length for length in current)
                for current in bins
            ]
            logical = [len(current) for current in bins]
            work_per_physical_sequence = {
                "effective_tokens": statistics.fmean(used),
                "computed_tokens": float(cutoff),
                "computed_attention_token_pairs": statistics.fmean(
                    attention
                ),
                "logical_samples": statistics.fmean(logical),
            }
            padding_utilization = (
                statistics.fmean(used) / float(cutoff)
            )
            packing_fill_ratio = padding_utilization
            mean_samples_per_pack = statistics.fmean(logical)
        else:
            computed_per_sample_runs = []
            attention_per_sample_runs = []
            for seed in PROFILE_SEEDS:
                shuffled = list(lengths)
                random.Random(seed).shuffle(shuffled)
                computed = 0
                attention = 0
                samples = 0
                for start in range(0, len(shuffled), mbs):
                    batch = shuffled[start : start + mbs]
                    maximum = max(batch)
                    computed += maximum * len(batch)
                    attention += maximum * maximum * len(batch)
                    samples += len(batch)
                computed_per_sample_runs.append(computed / samples)
                attention_per_sample_runs.append(attention / samples)
            mean_length = statistics.fmean(lengths)
            computed_per_sample = statistics.fmean(
                computed_per_sample_runs
            )
            work_per_physical_sequence = {
                "effective_tokens": mean_length,
                "computed_tokens": computed_per_sample,
                "computed_attention_token_pairs": statistics.fmean(
                    attention_per_sample_runs
                ),
                "logical_samples": 1.0,
            }
            padding_utilization = mean_length / computed_per_sample
            packing_fill_ratio = None
            mean_samples_per_pack = 1.0

        mean_length = statistics.fmean(lengths)
        result = {
            "dataset_id": str(dataset_id),
            "cutoff_len": cutoff,
            "physical_mbs": mbs,
            "packing": bool(packing),
            "samples_profiled": len(rows),
            "length": {
                "mean": mean_length,
                "std": statistics.pstdev(lengths),
                "p50": _percentile(lengths, 50),
                "p90": _percentile(lengths, 90),
                "p99": _percentile(lengths, 99),
                "maximum": max(lengths),
                "mean_squared": statistics.fmean(
                    length * length for length in lengths
                ),
            },
            "label_token_ratio": (
                sum(label_lengths) / _positive(sum(lengths))
            ),
            "mean_turns": statistics.fmean(
                float(row.get("turns") or 0.0) for row in rows
            ),
            "padding_utilization": padding_utilization,
            "packing_fill_ratio": packing_fill_ratio,
            "mean_samples_per_pack": mean_samples_per_pack,
            "work_per_physical_sequence": work_per_physical_sequence,
            "source_is_pre_run_static_profile": True,
        }
        self._cache[key] = result
        return result

    def report(self) -> dict[str, Any]:
        datasets = {}
        for dataset_id, rows in sorted(self.rows.items()):
            inferred_cutoff = int(
                math.ceil(
                    max(int(row.get("total_tokens") or 0) for row in rows)
                    / 512
                )
                * 512
            )
            datasets[dataset_id] = {
                "cutoff_len": inferred_cutoff,
                "unpacked": {
                    str(mbs): self.profile(
                        dataset_id,
                        cutoff_len=inferred_cutoff,
                        physical_mbs=mbs,
                        packing=False,
                    )
                    for mbs in (1, 2, 4, 8, 16, 32)
                },
                "packing_mbs1": self.profile(
                    dataset_id,
                    cutoff_len=inferred_cutoff,
                    physical_mbs=1,
                    packing=True,
                ),
            }
        report: dict[str, Any] = {
            "schema": STATIC_PROFILE_SCHEMA,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "profile_seeds": list(PROFILE_SEEDS),
            "source_bindings": self.source_bindings(),
            "datasets": datasets,
            "uses_runtime_outcome_counters": False,
        }
        report["report_sha256"] = sha256_json(report)
        return report


def _scenario_material_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    record = candidate["record"]
    scenario = record.get("scenario") or {}
    selector = record.get("selector") or {}
    return (
        str(scenario.get("model_id")),
        str(
            selector.get("training_mode")
            or scenario.get("train_type")
            or ""
        ),
        str(scenario.get("dataset_id")),
        int(scenario.get("target_gbs") or 0),
    )


def _candidate_model_id(candidate: Mapping[str, Any]) -> str:
    return str(candidate["record"]["scenario"]["model_id"])


def _candidate_dataset_id(candidate: Mapping[str, Any]) -> str:
    return str(candidate["record"]["scenario"]["dataset_id"])


def _static_work_per_step(
    record: Mapping[str, Any],
    profiles: StaticDatasetProfiles,
) -> tuple[dict[str, float], dict[str, Any]]:
    scenario = record.get("scenario") or {}
    selector = record.get("selector") or {}
    performance = record.get("performance") or {}
    gpu_count = int(scenario.get("gpu_count") or 0)
    mbs = int(scenario.get("physical_mbs") or 0)
    target_gbs = int(scenario.get("target_gbs") or 0)
    packing = bool(selector.get("packing"))
    if gpu_count <= 0 or mbs <= 0 or target_gbs <= 0:
        raise ValueError("Invalid static work resource geometry")
    profile = profiles.profile(
        str(scenario.get("dataset_id")),
        cutoff_len=int(scenario.get("cutoff_len") or 0),
        physical_mbs=mbs,
        packing=packing,
    )
    per_sequence = profile["work_per_physical_sequence"]
    if packing:
        observed_ga = performance.get("gradient_accumulation_steps")
        if observed_ga is not None:
            gradient_accumulation = int(observed_ga)
        else:
            effective_mbs = _positive(
                profile.get("mean_samples_per_pack")
            )
            gradient_accumulation = max(
                1,
                round(target_gbs / (gpu_count * effective_mbs)),
            )
        physical_sequences = (
            gpu_count * mbs * gradient_accumulation
        )
    else:
        denominator = gpu_count * mbs
        if target_gbs % denominator:
            raise ValueError(
                "Unpacked target_gbs is not divisible by GPU*MBS"
            )
        gradient_accumulation = target_gbs // denominator
        physical_sequences = target_gbs
    work = {
        key: float(value) * physical_sequences
        for key, value in per_sequence.items()
    }
    return work, {
        "gradient_accumulation_steps": gradient_accumulation,
        "physical_sequences_per_step": physical_sequences,
        "profile": profile,
    }


def _static_structured_basis(
    record: Mapping[str, Any],
    profiles: StaticDatasetProfiles,
    *,
    hardware_memory_bytes: float,
) -> dict[str, Any]:
    scenario = record.get("scenario") or {}
    selector = record.get("selector") or {}
    model = record.get("model_basis") or {}
    performance = record.get("performance") or {}
    priors = performance.get("physical_priors") or {}
    work, work_evidence = _static_work_per_step(record, profiles)

    gpu_count = int(scenario["gpu_count"])
    mbs = int(scenario["physical_mbs"])
    cutoff = int(scenario["cutoff_len"])
    gradient_accumulation = int(
        work_evidence["gradient_accumulation_steps"]
    )
    computed_tokens = float(work["computed_tokens"])
    attention_pairs = float(
        work["computed_attention_token_pairs"]
    )
    linear_base = _positive(model.get("linear_applications_per_pass"))
    adapter_parameters = _nonnegative(model.get("adapter_parameters"))
    num_layers = _positive(model.get("num_layers"))
    hidden_size = _positive(model.get("hidden_size"))
    intermediate_size = _positive(model.get("intermediate_size"))
    kv_width = _positive(model.get("kv_width"))
    training_mode = str(
        selector.get("training_mode")
        or scenario.get("train_type")
        or ""
    ).lower()
    is_lora = float(training_mode == "lora")
    gc = float(bool(selector.get("gradient_checkpointing")))

    if training_mode == "full":
        linear_flops = 6.0 * linear_base * computed_tokens
    else:
        linear_flops = (
            4.0 * linear_base * computed_tokens
            + 6.0 * adapter_parameters * computed_tokens
        )
    attention_flops = (
        6.0 * num_layers * hidden_size * attention_pairs
    )
    recompute_linear = gc * (
        2.0 * linear_base * computed_tokens
        + (
            2.0 * adapter_parameters * computed_tokens
            if training_mode == "lora"
            else 0.0
        )
    )
    recompute_attention = (
        gc * 2.0 * num_layers * hidden_size * attention_pairs
    )
    total_flops = (
        linear_flops
        + attention_flops
        + recompute_linear
        + recompute_attention
    )

    local_tokens = computed_tokens / gpu_count
    per_token_layer = (
        6.0 * hidden_size
        + 2.0 * kv_width
        + 3.0 * intermediate_size
    )
    weight_traffic = (
        gradient_accumulation
        * 2.0
        * (linear_base + adapter_parameters)
        * PARAMETER_BYTES
    )
    activation_traffic = (
        4.0
        * local_tokens
        * num_layers
        * per_token_layer
        * PARAMETER_BYTES
    )
    attention_traffic = (
        4.0
        * local_tokens
        * (2.0 * hidden_size + 2.0 * kv_width)
        * PARAMETER_BYTES
    )
    recompute_traffic = (
        2.0 * (activation_traffic + attention_traffic) * gc
    )
    kernel_traffic = (
        weight_traffic
        + activation_traffic
        + attention_traffic
        + recompute_traffic
    )
    trainable_parameters = _positive(model.get("trainable_parameters"))
    loaded_parameters = _positive(model.get("loaded_parameters"))
    optimizer_traffic = (
        2.0
        * trainable_parameters
        / gpu_count
        * (PARAMETER_BYTES + GRADIENT_BYTES + OPTIMIZER_BYTES)
    )

    zero_stage = int(selector.get("zero_stage") or 0)
    ring = (
        (gpu_count - 1.0) / gpu_count if gpu_count > 1 else 0.0
    )
    communication_payload = 0.0
    collective_count = 0
    if gpu_count > 1 and zero_stage == 1:
        gradient_payload = (
            trainable_parameters * COMMUNICATION_BYTES
        )
        updated_payload = trainable_parameters * PARAMETER_BYTES
        communication_payload = ring * (
            2.0 * gradient_payload + updated_payload
        )
        collective_count = _ceil_div(
            trainable_parameters,
            DEFAULT_REDUCE_BUCKET_ELEMENTS,
        ) + _ceil_div(
            trainable_parameters,
            DEFAULT_ALL_GATHER_BUCKET_ELEMENTS,
        )
    elif gpu_count > 1 and zero_stage == 2:
        gradient_payload = (
            trainable_parameters * COMMUNICATION_BYTES
        )
        updated_payload = trainable_parameters * PARAMETER_BYTES
        communication_payload = ring * (
            2.0 * gradient_accumulation * gradient_payload
            + updated_payload
        )
        collective_count = gradient_accumulation * _ceil_div(
            trainable_parameters,
            DEFAULT_REDUCE_BUCKET_ELEMENTS,
        ) + _ceil_div(
            trainable_parameters,
            DEFAULT_ALL_GATHER_BUCKET_ELEMENTS,
        )
    elif gpu_count > 1 and zero_stage == 3:
        materializations = 3 if gc else 2
        communication_payload = (
            ring
            * gradient_accumulation
            * (
                materializations
                * loaded_parameters
                * PARAMETER_BYTES
                + trainable_parameters * COMMUNICATION_BYTES
            )
        )
        collective_count = gradient_accumulation * (
            materializations
            * _ceil_div(
                loaded_parameters,
                DEFAULT_PREFETCH_BUCKET_ELEMENTS,
            )
            + _ceil_div(
                trainable_parameters,
                DEFAULT_REDUCE_BUCKET_ELEMENTS,
            )
        )

    peak_flops = _positive(
        priors.get("dense_bf16_peak_flops_per_gpu")
    )
    hbm_bandwidth = _positive(
        priors.get("hbm_bandwidth_bytes_per_second")
    )
    link_bandwidth = _positive(
        priors.get("intra_node_bandwidth_bytes_per_second")
    )
    collective_latency = _positive(
        priors.get("collective_latency_seconds"),
        1.0e-5,
    )
    compute_seconds = total_flops / (gpu_count * peak_flops)
    kernel_hbm_seconds = kernel_traffic / hbm_bandwidth
    optimizer_hbm_seconds = optimizer_traffic / hbm_bandwidth
    communication_seconds = (
        communication_payload / link_bandwidth
        + collective_count * collective_latency
    )
    # Approximate groups of kernels, not individual mathematical operators.
    training_passes = 3.0 + gc
    launch_count_proxy = (
        gradient_accumulation
        * num_layers
        * training_passes
        * 80.0
        + gradient_accumulation * 200.0
    )
    launch_seconds = (
        launch_count_proxy * KERNEL_LAUNCH_PRIOR_SECONDS
    )
    roof_seconds = (
        compute_seconds**SMOOTH_ROOFLINE_POWER
        + kernel_hbm_seconds**SMOOTH_ROOFLINE_POWER
    ) ** (1.0 / SMOOTH_ROOFLINE_POWER)

    profile = work_evidence["profile"]
    lengths = profile["length"]
    effective_tokens = float(work["effective_tokens"])
    zero2 = float(zero_stage == 2)
    zero3 = float(zero_stage == 3)
    log_gpu = math.log2(gpu_count)
    log_mbs = math.log2(mbs)
    base_parameters = _positive(model.get("base_parameters"))
    if zero_stage == 3:
        parameter_state_bytes = (
            loaded_parameters * PARAMETER_BYTES / gpu_count
            + trainable_parameters
            * (GRADIENT_BYTES + OPTIMIZER_BYTES)
            / gpu_count
        )
    elif zero_stage in (1, 2):
        parameter_state_bytes = (
            loaded_parameters * PARAMETER_BYTES
            + trainable_parameters * GRADIENT_BYTES
            + trainable_parameters * OPTIMIZER_BYTES / gpu_count
        )
    else:
        parameter_state_bytes = (
            loaded_parameters * PARAMETER_BYTES
            + trainable_parameters
            * (GRADIENT_BYTES + OPTIMIZER_BYTES)
        )
    kernel = str(selector.get("kernel_path") or "").lower()
    feature_values = {
        "is_lora": is_lora,
        "gradient_checkpointing": gc,
        "zero2": zero2,
        "zero3": zero3,
        "packing": float(bool(selector.get("packing"))),
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
        "log2_gpu_count": log_gpu,
        "log2_mbs": log_mbs,
        "log2_gradient_accumulation": math.log2(
            gradient_accumulation
        ),
        "log2_cutoff_over_512": math.log2(cutoff / 512.0),
        "log_compute_to_hbm_ratio": math.log(
            (compute_seconds + EPSILON)
            / (kernel_hbm_seconds + EPSILON)
        ),
        "log1p_comm_to_roof_ratio": math.log1p(
            communication_seconds / (roof_seconds + EPSILON)
        ),
        "log1p_optimizer_to_roof_ratio": math.log1p(
            optimizer_hbm_seconds / (roof_seconds + EPSILON)
        ),
        "log1p_launch_to_roof_ratio": math.log1p(
            launch_seconds / (roof_seconds + EPSILON)
        ),
        "effective_to_computed_token_ratio": min(
            1.5,
            effective_tokens / _positive(work["computed_tokens"]),
        ),
        "attention_flop_share": attention_flops / total_flops,
        "recompute_flop_share": (
            recompute_linear + recompute_attention
        )
        / total_flops,
        "log1p_model_state_to_hbm_capacity": math.log1p(
            parameter_state_bytes / _positive(hardware_memory_bytes)
        ),
        "mean_length_to_cutoff": float(lengths["mean"]) / cutoff,
        "length_cv": float(lengths["std"])
        / _positive(lengths["mean"]),
        "p90_to_mean_length": float(lengths["p90"])
        / _positive(lengths["mean"]),
        "p99_to_mean_length": float(lengths["p99"])
        / _positive(lengths["mean"]),
        "label_token_ratio": float(profile["label_token_ratio"]),
        "mean_turns_over_10": float(profile["mean_turns"]) / 10.0,
        "log1p_machine_balance_flops_per_hbm_byte": math.log1p(
            peak_flops / hbm_bandwidth
        ),
        "log1p_link_to_hbm_bandwidth_ratio": math.log1p(
            link_bandwidth / hbm_bandwidth
        ),
        "lora_x_gc": is_lora * gc,
        "gc_x_zero3": gc * zero3,
        "zero3_x_log2_gpu_count": zero3 * log_gpu,
        "log2_mbs_x_length_cv": log_mbs
        * float(lengths["std"])
        / _positive(lengths["mean"]),
    }
    features = np.asarray(
        [feature_values[name] for name in FEATURE_NAMES],
        dtype=float,
    )
    components = np.asarray(
        [
            launch_seconds,
            compute_seconds,
            kernel_hbm_seconds,
            optimizer_hbm_seconds,
            communication_seconds,
        ],
        dtype=float,
    )
    if not np.isfinite(features).all() or not np.isfinite(components).all():
        raise ValueError("Static structured basis contains NaN/Inf")
    return {
        "work_per_step": work,
        "work_evidence": work_evidence,
        "flops": {
            "linear": linear_flops,
            "attention": attention_flops,
            "recompute_linear": recompute_linear,
            "recompute_attention": recompute_attention,
            "total": total_flops,
        },
        "traffic": {
            "kernel": kernel_traffic,
            "optimizer": optimizer_traffic,
            "communication": communication_payload,
            "collective_count": collective_count,
            "launch_count_proxy": launch_count_proxy,
        },
        "component_seconds_at_physical_limits": {
            name: float(value)
            for name, value in zip(COMPONENT_NAMES, components)
        },
        "components": components,
        "feature_values": feature_values,
        "features": features,
        "base_parameters": base_parameters,
    }


def _structured_candidates(
    pooled: Sequence[Mapping[str, Any]],
    *,
    card_id: str,
    profiles: StaticDatasetProfiles,
    hardware_memory_bytes: float,
    role: str,
) -> list[dict[str, Any]]:
    result = []
    for original in pooled:
        candidate = dict(original)
        basis = _static_structured_basis(
            candidate["record"],
            profiles,
            hardware_memory_bytes=hardware_memory_bytes,
        )
        candidate.update(
            {
                "card_id": str(card_id),
                "source_role": str(role),
                "original_scenario_id": str(candidate["scenario_id"]),
                "scenario_id": (
                    f"{card_id}:{candidate['scenario_id']}"
                ),
                "domain_scenario_key": _scenario_material_key(candidate),
                "structured_basis": basis,
                "structured_features": basis["features"],
                "physical_components": basis["components"],
                "static_log_work": math.log(
                    _positive(
                        basis["work_per_step"]["effective_tokens"]
                    )
                ),
                "observed_log_throughput": (
                    float(candidate["effective_log_work"])
                    - float(candidate["observed_log_step"])
                ),
            }
        )
        result.append(candidate)
    return result


def _scenario_weights(
    candidates: Sequence[Mapping[str, Any]],
    *,
    require_pairs: bool,
) -> dict[str, float]:
    by_card: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for candidate in candidates:
        by_card[str(candidate["card_id"])][
            str(candidate["scenario_id"])
        ].append(candidate)
    result: dict[str, float] = {}
    active_cards = []
    for card_id, scenarios in by_card.items():
        eligible = {
            scenario_id: rows
            for scenario_id, rows in scenarios.items()
            if not require_pairs or len(rows) >= 2
        }
        if eligible:
            active_cards.append((card_id, eligible))
    if not active_cards:
        return result
    card_weight = 1.0 / len(active_cards)
    for _, scenarios in active_cards:
        raw = {
            scenario_id: (
                H800_HISTORICAL_SCENARIO_WEIGHT
                if all(row.get("historical") is True for row in rows)
                else 1.0
            )
            for scenario_id, rows in scenarios.items()
        }
        total = sum(raw.values())
        for scenario_id, value in raw.items():
            result[scenario_id] = card_weight * value / total
    return result


def _training_arrays(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("Structured model received no candidates")
    cards = sorted({str(candidate["card_id"]) for candidate in candidates})
    card_to_index = {card: index for index, card in enumerate(cards)}
    raw_features = np.vstack(
        [
            np.asarray(candidate["structured_features"], dtype=float)
            for candidate in candidates
        ]
    )
    absolute_scenario_weights = _scenario_weights(
        candidates,
        require_pairs=False,
    )
    scenario_counts = Counter(
        str(candidate["scenario_id"]) for candidate in candidates
    )
    absolute_weights = np.asarray(
        [
            absolute_scenario_weights[str(candidate["scenario_id"])]
            / scenario_counts[str(candidate["scenario_id"])]
            for candidate in candidates
        ],
        dtype=float,
    )
    means = np.average(
        raw_features,
        axis=0,
        weights=absolute_weights,
    )
    scales = np.sqrt(
        np.average(
            (raw_features - means) ** 2,
            axis=0,
            weights=absolute_weights,
        )
    )
    scales[scales < 1.0e-9] = 1.0
    standardized = 2.0 * np.tanh(
        (raw_features - means) / (2.0 * scales)
    )

    by_scenario: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_scenario[str(candidate["scenario_id"])].append(index)
    pair_scenario_weights = _scenario_weights(
        candidates,
        require_pairs=True,
    )
    pair_left = []
    pair_right = []
    pair_weights = []
    for scenario_id, indexes in by_scenario.items():
        pairs = list(itertools.combinations(indexes, 2))
        if not pairs:
            continue
        scenario_weight = pair_scenario_weights[scenario_id]
        for left, right in pairs:
            pair_left.append(left)
            pair_right.append(right)
            pair_weights.append(scenario_weight / len(pairs))

    list_groups = [
        {
            "scenario_id": scenario_id,
            "indexes": np.asarray(indexes, dtype=int),
            "weight": pair_scenario_weights[scenario_id],
        }
        for scenario_id, indexes in by_scenario.items()
        if len(indexes) >= 2
    ]
    return {
        "cards": cards,
        "card_to_index": card_to_index,
        "card_indexes": np.asarray(
            [card_to_index[str(candidate["card_id"])] for candidate in candidates],
            dtype=int,
        ),
        "feature_means": means,
        "feature_scales": scales,
        "raw_features": raw_features,
        "standardized_features": standardized,
        "components": np.vstack(
            [
                np.asarray(candidate["physical_components"], dtype=float)
                for candidate in candidates
            ]
        ),
        "static_log_work": np.asarray(
            [float(candidate["static_log_work"]) for candidate in candidates],
            dtype=float,
        ),
        "observed_log_throughput": np.asarray(
            [
                float(candidate["observed_log_throughput"])
                for candidate in candidates
            ],
            dtype=float,
        ),
        "absolute_weights": absolute_weights,
        "pair_left": np.asarray(pair_left, dtype=int),
        "pair_right": np.asarray(pair_right, dtype=int),
        "pair_weights": np.asarray(pair_weights, dtype=float),
        "list_groups": list_groups,
        "scenario_count": len(by_scenario),
    }


def _parameter_layout(card_count: int) -> dict[str, Any]:
    cursor = 0
    global_components = slice(cursor, cursor + len(COMPONENT_NAMES))
    cursor = global_components.stop
    card_components = slice(
        cursor,
        cursor + card_count * len(COMPONENT_NAMES),
    )
    cursor = card_components.stop
    card_residuals = slice(
        cursor,
        cursor + card_count * len(CARD_RESIDUAL_FEATURE_NAMES),
    )
    cursor = card_residuals.stop
    intercept = cursor
    cursor += 1
    beta = slice(cursor, cursor + len(FEATURE_NAMES))
    cursor = beta.stop
    return {
        "global_components": global_components,
        "card_components": card_components,
        "card_residuals": card_residuals,
        "intercept": intercept,
        "beta": beta,
        "size": cursor,
        "card_count": card_count,
    }


def _prediction_and_jacobian(
    parameters: np.ndarray,
    arrays: Mapping[str, Any],
    layout: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    card_count = int(layout["card_count"])
    global_raw = parameters[layout["global_components"]]
    card_offsets = parameters[layout["card_components"]].reshape(
        card_count,
        len(COMPONENT_NAMES),
    )
    card_residuals = parameters[layout["card_residuals"]].reshape(
        card_count,
        len(CARD_RESIDUAL_FEATURE_NAMES),
    )
    intercept = float(parameters[int(layout["intercept"])])
    beta = parameters[layout["beta"]]
    card_indexes = arrays["card_indexes"]
    raw = global_raw[None, :] + card_offsets[card_indexes]
    multipliers = 1.0 + np.logaddexp(0.0, raw)
    multiplier_derivative = expit(raw)

    components = arrays["components"]
    launch = components[:, 0] * multipliers[:, 0]
    compute = components[:, 1] * multipliers[:, 1]
    hbm = components[:, 2] * multipliers[:, 2]
    optimizer = components[:, 3] * multipliers[:, 3]
    communication = components[:, 4] * multipliers[:, 4]
    power = SMOOTH_ROOFLINE_POWER
    roof = (compute**power + hbm**power) ** (1.0 / power)
    step_base = launch + roof + optimizer + communication

    standardized = arrays["standardized_features"]
    card_sensitive = standardized[:, CARD_RESIDUAL_FEATURE_INDEXES]
    raw_correction = (
        intercept
        + standardized @ beta
        + np.sum(
            card_sensitive * card_residuals[card_indexes],
            axis=1,
        )
    )
    correction = CORRECTION_LOG_LIMIT * np.tanh(
        raw_correction / CORRECTION_LOG_LIMIT
    )
    correction_derivative = 1.0 - np.tanh(
        raw_correction / CORRECTION_LOG_LIMIT
    ) ** 2
    log_step = np.log(np.maximum(EPSILON, step_base)) + correction
    prediction = arrays["static_log_work"] - log_step

    count = len(prediction)
    jacobian = np.zeros((count, int(layout["size"])), dtype=float)
    dstep_dmultiplier = np.zeros_like(components)
    dstep_dmultiplier[:, 0] = components[:, 0]
    roof_denominator = np.maximum(EPSILON, roof ** (power - 1.0))
    dstep_dmultiplier[:, 1] = (
        compute ** (power - 1.0)
        * components[:, 1]
        / roof_denominator
    )
    dstep_dmultiplier[:, 2] = (
        hbm ** (power - 1.0)
        * components[:, 2]
        / roof_denominator
    )
    dstep_dmultiplier[:, 3] = components[:, 3]
    dstep_dmultiplier[:, 4] = components[:, 4]
    dpred_draw = -(
        dstep_dmultiplier
        * multiplier_derivative
        / step_base[:, None]
    )
    jacobian[:, layout["global_components"]] = dpred_draw
    card_component_start = int(layout["card_components"].start)
    for row, card_index in enumerate(card_indexes):
        start = (
            card_component_start
            + int(card_index) * len(COMPONENT_NAMES)
        )
        jacobian[
            row,
            start : start + len(COMPONENT_NAMES),
        ] = dpred_draw[row]
    card_residual_start = int(layout["card_residuals"].start)
    for row, card_index in enumerate(card_indexes):
        start = (
            card_residual_start
            + int(card_index) * len(CARD_RESIDUAL_FEATURE_NAMES)
        )
        jacobian[
            row,
            start : start + len(CARD_RESIDUAL_FEATURE_NAMES),
        ] = -correction_derivative[row] * card_sensitive[row]
    jacobian[:, int(layout["intercept"])] = -correction_derivative
    jacobian[:, layout["beta"]] = -(
        correction_derivative[:, None] * standardized
    )
    return prediction, jacobian, {
        "step_base": step_base,
        "log_step": log_step,
        "multipliers": multipliers,
        "correction": correction,
    }


def _loss_and_gradient(
    parameters: np.ndarray,
    arrays: Mapping[str, Any],
    layout: Mapping[str, Any],
) -> tuple[float, np.ndarray, dict[str, float]]:
    prediction, jacobian, _ = _prediction_and_jacobian(
        parameters,
        arrays,
        layout,
    )
    observed = arrays["observed_log_throughput"]
    absolute_weights = arrays["absolute_weights"]
    absolute_loss, absolute_gradient = _huber(
        prediction - observed,
        ABSOLUTE_HUBER_DELTA,
    )
    components = {
        "absolute": float(
            np.sum(absolute_weights * absolute_loss)
        )
    }
    gradient_prediction = (
        LOSS_WEIGHTS["absolute"]
        * absolute_weights
        * absolute_gradient
    )
    total = LOSS_WEIGHTS["absolute"] * components["absolute"]

    left = arrays["pair_left"]
    right = arrays["pair_right"]
    if len(left):
        pair_weights = arrays["pair_weights"]
        predicted_difference = prediction[left] - prediction[right]
        observed_difference = observed[left] - observed[right]
        pair_loss, pair_gradient = _huber(
            predicted_difference - observed_difference,
            PAIRWISE_HUBER_DELTA,
        )
        components["pairwise_magnitude"] = float(
            np.sum(pair_weights * pair_loss)
        )
        pair_prediction_gradient = (
            LOSS_WEIGHTS["pairwise_magnitude"]
            * pair_weights
            * pair_gradient
        )
        total += (
            LOSS_WEIGHTS["pairwise_magnitude"]
            * components["pairwise_magnitude"]
        )

        signs = np.sign(observed_difference)
        order_argument = (
            -signs
            * predicted_difference
            / PAIRWISE_ORDER_TEMPERATURE
        )
        order_loss = np.logaddexp(0.0, order_argument)
        components["pairwise_order"] = float(
            np.sum(pair_weights * order_loss)
        )
        pair_prediction_gradient += (
            LOSS_WEIGHTS["pairwise_order"]
            * pair_weights
            * (
                -signs
                / PAIRWISE_ORDER_TEMPERATURE
                * expit(order_argument)
            )
        )
        total += (
            LOSS_WEIGHTS["pairwise_order"]
            * components["pairwise_order"]
        )
        np.add.at(gradient_prediction, left, pair_prediction_gradient)
        np.add.at(gradient_prediction, right, -pair_prediction_gradient)
    else:
        components["pairwise_magnitude"] = 0.0
        components["pairwise_order"] = 0.0

    listwise_loss = 0.0
    for group in arrays["list_groups"]:
        indexes = group["indexes"]
        target_probability = _stable_softmax(
            observed[indexes],
            LISTWISE_TEMPERATURE,
        )
        predicted_probability = _stable_softmax(
            prediction[indexes],
            LISTWISE_TEMPERATURE,
        )
        current = float(
            np.sum(
                target_probability
                * (
                    np.log(np.maximum(EPSILON, target_probability))
                    - np.log(
                        np.maximum(EPSILON, predicted_probability)
                    )
                )
            )
        )
        group_weight = float(group["weight"])
        listwise_loss += group_weight * current
        gradient_prediction[indexes] += (
            LOSS_WEIGHTS["listwise"]
            * group_weight
            * (
                predicted_probability - target_probability
            )
            / LISTWISE_TEMPERATURE
        )
    components["listwise"] = listwise_loss
    total += LOSS_WEIGHTS["listwise"] * listwise_loss
    gradient = jacobian.T @ gradient_prediction

    beta = parameters[layout["beta"]]
    beta_penalty = 0.5 * float(beta @ beta)
    total += LOSS_WEIGHTS["shared_residual_ridge"] * beta_penalty
    gradient[layout["beta"]] += (
        LOSS_WEIGHTS["shared_residual_ridge"] * beta
    )
    components["shared_residual_ridge"] = beta_penalty

    card_offsets = parameters[layout["card_components"]]
    card_residuals = parameters[layout["card_residuals"]]
    card_penalty = 0.5 * float(
        card_offsets @ card_offsets
        + card_residuals @ card_residuals
    )
    total += LOSS_WEIGHTS["card_adapter_ridge"] * card_penalty
    gradient[layout["card_components"]] += (
        LOSS_WEIGHTS["card_adapter_ridge"] * card_offsets
    )
    gradient[layout["card_residuals"]] += (
        LOSS_WEIGHTS["card_adapter_ridge"] * card_residuals
    )
    components["card_adapter_ridge"] = card_penalty

    global_raw = parameters[layout["global_components"]]
    global_multiplier = 1.0 + np.logaddexp(0.0, global_raw)
    prior = np.asarray(
        [INITIAL_INVERSE_EFFICIENCY[name] for name in COMPONENT_NAMES],
        dtype=float,
    )
    prior_residual = np.log(global_multiplier) - np.log(prior)
    prior_penalty = 0.5 * float(prior_residual @ prior_residual)
    total += (
        LOSS_WEIGHTS["global_efficiency_prior"] * prior_penalty
    )
    gradient[layout["global_components"]] += (
        LOSS_WEIGHTS["global_efficiency_prior"]
        * prior_residual
        / global_multiplier
        * expit(global_raw)
    )
    components["global_efficiency_prior"] = prior_penalty
    return float(total), gradient, components


def _fit_model(
    candidates: Sequence[Mapping[str, Any]],
    *,
    audit_gradient: bool = False,
) -> dict[str, Any]:
    arrays = _training_arrays(candidates)
    layout = _parameter_layout(len(arrays["cards"]))
    initial = np.zeros(int(layout["size"]), dtype=float)
    initial[layout["global_components"]] = np.asarray(
        [
            _inverse_softplus(INITIAL_INVERSE_EFFICIENCY[name])
            for name in COMPONENT_NAMES
        ],
        dtype=float,
    )

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        loss, gradient, _ = _loss_and_gradient(
            parameters,
            arrays,
            layout,
        )
        return loss, gradient

    bounds = []
    bounds.extend([(-8.0, 12.0)] * len(COMPONENT_NAMES))
    bounds.extend(
        [(-4.0, 4.0)]
        * (len(arrays["cards"]) * len(COMPONENT_NAMES))
    )
    bounds.extend(
        [(-2.0, 2.0)]
        * (
            len(arrays["cards"])
            * len(CARD_RESIDUAL_FEATURE_NAMES)
        )
    )
    bounds.append((-3.0, 3.0))
    bounds.extend([(-3.0, 3.0)] * len(FEATURE_NAMES))
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": 400,
            "ftol": 1.0e-11,
            "gtol": 1.0e-7,
            "maxls": 40,
        },
    )
    parameters = np.asarray(result.x, dtype=float)
    final_loss, _, components = _loss_and_gradient(
        parameters,
        arrays,
        layout,
    )
    gradient_audit = None
    if audit_gradient:
        rng = np.random.default_rng(20260728)
        probe = parameters + rng.normal(
            loc=0.0,
            scale=0.03,
            size=len(parameters),
        )
        for index, (lower, upper) in enumerate(bounds):
            probe[index] = float(
                np.clip(probe[index], lower + 0.1, upper - 0.1)
            )
        _, analytic_gradient, _ = _loss_and_gradient(
            probe,
            arrays,
            layout,
        )
        audit_indexes = sorted(
            {
                int(index)
                for index in np.linspace(
                    0,
                    len(parameters) - 1,
                    num=min(20, len(parameters)),
                )
            }
        )
        finite_difference_step = 1.0e-6
        rows = []
        for index in audit_indexes:
            plus = probe.copy()
            minus = probe.copy()
            plus[index] += finite_difference_step
            minus[index] -= finite_difference_step
            plus_loss, _, _ = _loss_and_gradient(plus, arrays, layout)
            minus_loss, _, _ = _loss_and_gradient(minus, arrays, layout)
            finite_difference = (
                plus_loss - minus_loss
            ) / (2.0 * finite_difference_step)
            analytic = float(analytic_gradient[index])
            absolute_error = abs(finite_difference - analytic)
            scaled_error = absolute_error / max(
                1.0e-8,
                abs(finite_difference),
                abs(analytic),
            )
            rows.append(
                {
                    "parameter_index": index,
                    "analytic": analytic,
                    "finite_difference": finite_difference,
                    "absolute_error": absolute_error,
                    "scaled_error": scaled_error,
                }
            )
        gradient_audit = {
            "probe": "deterministic_perturbation_of_fitted_parameters",
            "finite_difference_step": finite_difference_step,
            "checked_parameters": len(rows),
            "max_absolute_error": max(
                row["absolute_error"] for row in rows
            ),
            "max_scaled_error": max(
                row["scaled_error"] for row in rows
            ),
            "rows": rows,
        }
    global_raw = parameters[layout["global_components"]]
    card_offsets = parameters[layout["card_components"]].reshape(
        len(arrays["cards"]),
        len(COMPONENT_NAMES),
    )
    card_residuals = parameters[layout["card_residuals"]].reshape(
        len(arrays["cards"]),
        len(CARD_RESIDUAL_FEATURE_NAMES),
    )
    global_multiplier = 1.0 + np.logaddexp(0.0, global_raw)
    card_multipliers = {}
    for card_index, card_id in enumerate(arrays["cards"]):
        card_multipliers[card_id] = {
            name: float(value)
            for name, value in zip(
                COMPONENT_NAMES,
                1.0
                + np.logaddexp(
                    0.0,
                    global_raw + card_offsets[card_index],
                ),
            )
        }
    raw_features = arrays["raw_features"]

    def support_for_indexes(
        indexes: Sequence[int],
    ) -> dict[str, Any]:
        selected_candidates = [candidates[index] for index in indexes]
        selected_features = raw_features[np.asarray(indexes, dtype=int)]
        feature_support = {}
        for feature_index, name in enumerate(FEATURE_NAMES):
            values = selected_features[:, feature_index]
            feature_support[name] = {
                "min": float(np.min(values)),
                "p05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(np.max(values)),
            }
        base_parameters = np.asarray(
            [
                float(
                    candidate["structured_basis"]["base_parameters"]
                )
                for candidate in selected_candidates
            ],
            dtype=float,
        )
        scenarios = [
            candidate["record"]["scenario"]
            for candidate in selected_candidates
        ]
        selectors = [
            candidate["record"]["selector"]
            for candidate in selected_candidates
        ]
        return {
            "candidate_rows": len(selected_candidates),
            "scenario_rows": len(
                {
                    str(candidate["scenario_id"])
                    for candidate in selected_candidates
                }
            ),
            "model_ids": sorted(
                {
                    _candidate_model_id(candidate)
                    for candidate in selected_candidates
                }
            ),
            "dataset_ids": sorted(
                {
                    _candidate_dataset_id(candidate)
                    for candidate in selected_candidates
                }
            ),
            "feature_ranges": feature_support,
            "base_parameters": {
                "min": float(np.min(base_parameters)),
                "median": float(np.median(base_parameters)),
                "max": float(np.max(base_parameters)),
            },
            "gpu_counts": sorted(
                {
                    int(scenario["gpu_count"])
                    for scenario in scenarios
                }
            ),
            "physical_mbs": sorted(
                {
                    int(scenario["physical_mbs"])
                    for scenario in scenarios
                }
            ),
            "cutoff_lens": sorted(
                {
                    int(scenario["cutoff_len"])
                    for scenario in scenarios
                }
            ),
            "target_gbs": sorted(
                {
                    int(scenario["target_gbs"])
                    for scenario in scenarios
                }
            ),
            "training_modes": sorted(
                {
                    str(
                        selector.get("training_mode")
                        or scenario.get("train_type")
                    )
                    for selector, scenario in zip(
                        selectors,
                        scenarios,
                    )
                }
            ),
            "zero_stages": sorted(
                {
                    int(selector.get("zero_stage") or 0)
                    for selector in selectors
                }
            ),
            "packing_values": sorted(
                {
                    bool(selector.get("packing"))
                    for selector in selectors
                }
            ),
        }

    training_support = support_for_indexes(
        list(range(len(candidates)))
    )
    training_support["by_card"] = {
        card_id: support_for_indexes(
            [
                index
                for index, candidate in enumerate(candidates)
                if str(candidate["card_id"]) == card_id
            ]
        )
        for card_id in arrays["cards"]
    }
    training_support["interpretation"] = (
        "global support identifies complete extrapolation; by_card support "
        "identifies configurations transferred from the other fitted card; "
        "emit a numeric estimate but downgrade confidence outside either"
    )
    return {
        "available": True,
        "model_family": (
            "static_profile_hierarchical_card_adapter_multitask"
        ),
        "formula": (
            "log(T_hat)=log(static_effective_tokens)-"
            "[log(launch+smoothmax(compute,hbm)+optimizer+comm)+"
            "bounded(shared_residual+known_card_residual)]"
        ),
        "component_names": list(COMPONENT_NAMES),
        "smooth_roofline_power": SMOOTH_ROOFLINE_POWER,
        "global_component_raw_parameters": global_raw.tolist(),
        "global_inverse_efficiency_multipliers": {
            name: float(value)
            for name, value in zip(
                COMPONENT_NAMES,
                global_multiplier,
            )
        },
        "card_component_raw_offsets": {
            card_id: card_offsets[index].tolist()
            for index, card_id in enumerate(arrays["cards"])
        },
        "card_residual_feature_names": list(
            CARD_RESIDUAL_FEATURE_NAMES
        ),
        "card_residual_coefficients": {
            card_id: card_residuals[index].tolist()
            for index, card_id in enumerate(arrays["cards"])
        },
        "card_inverse_efficiency_multipliers": card_multipliers,
        "unknown_card_policy": (
            "use global component multipliers, zero card component offset, "
            "zero card residual adapter, and downgrade confidence"
        ),
        "feature_names": list(FEATURE_NAMES),
        "feature_dimension": len(FEATURE_NAMES),
        "feature_means": arrays["feature_means"].tolist(),
        "feature_scales": arrays["feature_scales"].tolist(),
        "feature_transform": (
            "2*tanh((x-mean)/(2*scale))"
        ),
        "correction_log_limit": CORRECTION_LOG_LIMIT,
        "correction_intercept": float(
            parameters[int(layout["intercept"])]
        ),
        "correction_coefficients": parameters[
            layout["beta"]
        ].tolist(),
        "loss_weights": dict(LOSS_WEIGHTS),
        "training_support": training_support,
        "fit_candidates": len(candidates),
        "fit_scenarios": int(arrays["scenario_count"]),
        "fit_pairs": len(arrays["pair_left"]),
        "fit_cards": list(arrays["cards"]),
        "optimizer": {
            "method": "L-BFGS-B_with_analytic_gradient",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "final_loss": final_loss,
            "loss_components_unweighted": components,
            "analytic_gradient_audit": gradient_audit,
        },
    }


def _predict_log_throughput(
    candidate: Mapping[str, Any],
    model: Mapping[str, Any],
) -> float:
    features = np.asarray(candidate["structured_features"], dtype=float)
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    standardized = 2.0 * np.tanh(
        (features - means) / (2.0 * scales)
    )
    global_raw = np.asarray(
        model["global_component_raw_parameters"],
        dtype=float,
    )
    card_offsets = model["card_component_raw_offsets"]
    offset = np.asarray(
        card_offsets.get(
            str(candidate["card_id"]),
            np.zeros(len(COMPONENT_NAMES)),
        ),
        dtype=float,
    )
    multipliers = 1.0 + np.logaddexp(0.0, global_raw + offset)
    components = np.asarray(
        candidate["physical_components"],
        dtype=float,
    )
    launch = components[0] * multipliers[0]
    compute = components[1] * multipliers[1]
    hbm = components[2] * multipliers[2]
    optimizer = components[3] * multipliers[3]
    communication = components[4] * multipliers[4]
    roof = (
        compute**SMOOTH_ROOFLINE_POWER
        + hbm**SMOOTH_ROOFLINE_POWER
    ) ** (1.0 / SMOOTH_ROOFLINE_POWER)
    step_base = launch + roof + optimizer + communication
    raw_correction = float(model["correction_intercept"]) + float(
        standardized
        @ np.asarray(model["correction_coefficients"], dtype=float)
    )
    card_residuals = model["card_residual_coefficients"]
    card_residual = np.asarray(
        card_residuals.get(
            str(candidate["card_id"]),
            np.zeros(len(CARD_RESIDUAL_FEATURE_NAMES)),
        ),
        dtype=float,
    )
    raw_correction += float(
        standardized[list(CARD_RESIDUAL_FEATURE_INDEXES)]
        @ card_residual
    )
    correction = CORRECTION_LOG_LIMIT * math.tanh(
        raw_correction / CORRECTION_LOG_LIMIT
    )
    log_step = math.log(_positive(step_base)) + correction
    return float(candidate["static_log_work"]) - log_step


def _evaluate(
    candidates: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    return _evaluate_entries(
        [
            (candidate, _predict_log_throughput(candidate, model))
            for candidate in candidates
        ]
    )


def _fit_and_evaluate(
    train: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = _fit_model(train)
    return model, _evaluate(test, model)


def _work_reconstruction_audit(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fields = (
        "effective_tokens",
        "computed_tokens",
        "computed_attention_token_pairs",
    )
    ratios = {field: [] for field in fields}
    absolute_relative_errors = {field: [] for field in fields}
    for candidate in candidates:
        observed = (
            candidate["record"]["performance"]["work_per_step"]
        )
        predicted = (
            candidate["structured_basis"]["work_per_step"]
        )
        for field in fields:
            ratio = float(predicted[field]) / _positive(observed[field])
            ratios[field].append(ratio)
            absolute_relative_errors[field].append(abs(ratio - 1.0))
    return {
        "candidate_rows": len(candidates),
        "uses_observed_work_as_model_input": False,
        "fields": {
            field: {
                "predicted_to_observed_ratio_median": statistics.median(
                    ratios[field]
                ),
                "predicted_to_observed_ratio_p10": _percentile(
                    ratios[field], 10
                ),
                "predicted_to_observed_ratio_p90": _percentile(
                    ratios[field], 90
                ),
                "absolute_relative_error_median": statistics.median(
                    absolute_relative_errors[field]
                ),
                "absolute_relative_error_p90": _percentile(
                    absolute_relative_errors[field], 90
                ),
            }
            for field in fields
        },
    }


def _h800_holdout_protocol(
    h800_train: Sequence[Mapping[str, Any]],
    h800_holdout: Sequence[Mapping[str, Any]],
    rtx4090: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    held_material = {
        tuple(candidate["domain_scenario_key"])
        for candidate in h800_holdout
    }
    cross_card_train = [
        candidate
        for candidate in rtx4090
        if tuple(candidate["domain_scenario_key"]) not in held_material
    ]
    native_model, native_metrics = _fit_and_evaluate(
        h800_train,
        h800_holdout,
    )
    shared_model, shared_metrics = _fit_and_evaluate(
        [*h800_train, *cross_card_train],
        h800_holdout,
    )
    return {
        "method": (
            "H800 strict historical+native calibration fit; no H800 "
            "holdout scenario overlap.  Cross-card variant also removes "
            "4090 rows with the same card-independent scenario material."
        ),
        "native_only": {
            "train_candidates": len(h800_train),
            "metrics": native_metrics,
            "model": native_model,
        },
        "cross_card_shared": {
            "train_h800_candidates": len(h800_train),
            "train_rtx4090_candidates": len(cross_card_train),
            "removed_cross_card_scenario_matches": (
                len(rtx4090) - len(cross_card_train)
            ),
            "metrics": shared_metrics,
            "model": shared_model,
        },
    }


def _rtx4090_scenario_cv(
    h800_train: Sequence[Mapping[str, Any]],
    rtx4090: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entries = []
    folds = []
    for fold_index, held_ids in enumerate(
        _scenario_folds(rtx4090, 5)
    ):
        test = [
            candidate
            for candidate in rtx4090
            if str(candidate["scenario_id"]) in held_ids
        ]
        held_material = {
            tuple(candidate["domain_scenario_key"])
            for candidate in test
        }
        train = [
            candidate
            for candidate in [*h800_train, *rtx4090]
            if str(candidate["scenario_id"]) not in held_ids
            and tuple(candidate["domain_scenario_key"])
            not in held_material
        ]
        model = _fit_model(train)
        fold_entries = [
            (candidate, _predict_log_throughput(candidate, model))
            for candidate in test
        ]
        entries.extend(fold_entries)
        folds.append(
            {
                "fold": fold_index,
                "held_out_scenario_ids": sorted(held_ids),
                "held_out_card_independent_scenario_rows": len(
                    held_material
                ),
                "train_candidates": len(train),
                "test_candidates": len(test),
                "optimizer_success": model["optimizer"]["success"],
                "metrics": _evaluate_entries(fold_entries),
            }
        )
    return {
        "method": (
            "deterministic 5-fold complete-scenario CV on RTX4090; "
            "the same model/dataset/mode/GBS material is removed from "
            "H800 shared training"
        ),
        "folds": folds,
        "aggregate": _evaluate_entries(entries),
    }


def _rtx4090_model_holdout(
    h800_train: Sequence[Mapping[str, Any]],
    rtx4090: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entries = []
    folds = []
    for model_id in sorted(
        {_candidate_model_id(candidate) for candidate in rtx4090}
    ):
        train = [
            candidate
            for candidate in [*h800_train, *rtx4090]
            if _candidate_model_id(candidate) != model_id
        ]
        test = [
            candidate
            for candidate in rtx4090
            if _candidate_model_id(candidate) == model_id
        ]
        model = _fit_model(train)
        fold_entries = [
            (candidate, _predict_log_throughput(candidate, model))
            for candidate in test
        ]
        entries.extend(fold_entries)
        folds.append(
            {
                "held_out_model_id": model_id,
                "train_candidates": len(train),
                "test_candidates": len(test),
                "optimizer_success": model["optimizer"]["success"],
                "metrics": _evaluate_entries(fold_entries),
            }
        )
    return {
        "method": (
            "leave-one-complete-model-id-out across both cards; "
            "evaluate the held model on RTX4090"
        ),
        "folds": folds,
        "aggregate": _evaluate_entries(entries),
    }


def _dataset_holdout(
    h800_train: Sequence[Mapping[str, Any]],
    rtx4090: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    population = [*h800_train, *rtx4090]
    entries = []
    folds = []
    for dataset_id in sorted(
        {_candidate_dataset_id(candidate) for candidate in population}
    ):
        train = [
            candidate
            for candidate in population
            if _candidate_dataset_id(candidate) != dataset_id
        ]
        test = [
            candidate
            for candidate in population
            if _candidate_dataset_id(candidate) == dataset_id
        ]
        model = _fit_model(train)
        fold_entries = [
            (candidate, _predict_log_throughput(candidate, model))
            for candidate in test
        ]
        entries.extend(fold_entries)
        folds.append(
            {
                "held_out_dataset_id": dataset_id,
                "train_candidates": len(train),
                "test_candidates": len(test),
                "test_cards": sorted(
                    {str(candidate["card_id"]) for candidate in test}
                ),
                "optimizer_success": model["optimizer"]["success"],
                "metrics": _evaluate_entries(fold_entries),
            }
        )
    return {
        "method": "leave-one-complete-dataset-distribution-out across cards",
        "folds": folds,
        "aggregate": _evaluate_entries(entries),
    }


def _card_holdout(
    h800_train: Sequence[Mapping[str, Any]],
    h800_holdout: Sequence[Mapping[str, Any]],
    rtx4090: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    h800_model, rtx_metrics = _fit_and_evaluate(
        h800_train,
        rtx4090,
    )
    rtx_model, h800_metrics = _fit_and_evaluate(
        rtx4090,
        h800_holdout,
    )
    return {
        "method": (
            "leave-complete-card-out; unseen cards use global component "
            "multipliers and no card adapter"
        ),
        "h800_to_rtx4090": {
            "train_candidates": len(h800_train),
            "test_candidates": len(rtx4090),
            "metrics": rtx_metrics,
            "optimizer_success": h800_model["optimizer"]["success"],
        },
        "rtx4090_to_h800": {
            "train_candidates": len(rtx4090),
            "test_candidates": len(h800_holdout),
            "metrics": h800_metrics,
            "optimizer_success": rtx_model["optimizer"]["success"],
        },
    }


def _source_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def build_report(
    *,
    profiles: StaticDatasetProfiles,
    static_profile_path: Path,
    h800_observations_path: Path,
    h800_theory_basis_path: Path,
    h800_inventory_path: Path,
    h800_hardware_path: Path,
    h800_runtime_root: Path,
    rtx4090_campaign_root: Path,
    prior_joint_report_path: Path,
    analysis_extension_model_ids: Sequence[str] = (),
) -> dict[str, Any]:
    h800_data = _load_h800(
        observation_path=h800_observations_path,
        theory_basis_path=h800_theory_basis_path,
        inventory_path=h800_inventory_path,
        hardware_path=h800_hardware_path,
        runtime_root=h800_runtime_root,
    )
    h800_hardware = read_json(h800_hardware_path)
    h800_train = _structured_candidates(
        h800_data["strict_fit_candidates"],
        card_id="h800",
        profiles=profiles,
        hardware_memory_bytes=float(
            h800_hardware["memory_bytes_reported_by_torch"]
        ),
        role="h800_strict_fit",
    )
    h800_holdout = _structured_candidates(
        h800_data["holdout_candidates"],
        card_id="h800",
        profiles=profiles,
        hardware_memory_bytes=float(
            h800_hardware["memory_bytes_reported_by_torch"]
        ),
        role="h800_native_holdout",
    )

    rtx_records, rtx_admission = build_4090_records(
        rtx4090_campaign_root,
        matrix_name="throughput_jobs.jsonl",
        hbm_bandwidth_bytes_s=DEFAULT_HBM_BANDWIDTH_BYTES_S,
        collective_bandwidth_bytes_s=(
            DEFAULT_COLLECTIVE_BANDWIDTH_BYTES_S
        ),
    )
    rtx_hardware_path = (
        rtx4090_campaign_root / "config" / "hardware.json"
    )
    rtx_hardware = read_json(rtx_hardware_path)
    rtx4090 = _structured_candidates(
        _pool_candidates(rtx_records),
        card_id="rtx4090",
        profiles=profiles,
        hardware_memory_bytes=float(
            rtx_hardware["memory_bytes_reported_by_torch"]
        ),
        role="rtx4090_main",
    )

    h800_protocol = _h800_holdout_protocol(
        h800_train,
        h800_holdout,
        rtx4090,
    )
    rtx_scenario = _rtx4090_scenario_cv(h800_train, rtx4090)
    rtx_model = _rtx4090_model_holdout(h800_train, rtx4090)
    dataset_holdout = _dataset_holdout(h800_train, rtx4090)
    card_holdout = _card_holdout(
        h800_train,
        h800_holdout,
        rtx4090,
    )
    final_train = [*h800_train, *rtx4090]
    frozen_model = _fit_model(final_train, audit_gradient=True)
    prior_joint = read_json(prior_joint_report_path)
    inventory = read_json(h800_inventory_path)
    inventory_model_ids = {
        str(model.get("id"))
        for model in (inventory.get("models") or [])
        if model.get("id")
    }
    extension_model_ids = sorted(set(analysis_extension_model_ids))
    unknown_extensions = sorted(set(extension_model_ids) - inventory_model_ids)
    if unknown_extensions:
        raise ValueError(
            "analysis model extensions are absent from the bound inventory: "
            f"{unknown_extensions}"
        )
    packing_training_rows = sum(
        bool(
            (
                (candidate.get("record") or {}).get("selector") or {}
            ).get("packing")
        )
        for candidate in final_train
    )
    packing_evidence = "present" if packing_training_rows else "absent"
    packing_confidence_policy = (
        "packing_true: use the matching blocked-validation reference and "
        "downgrade confidence outside observed packing support"
        if packing_training_rows
        else "packing_true: unsupported by primary training evidence; low confidence"
    )
    limitations = [
        "only_dense_qwen3_architecture_family_has_training_evidence",
        "only_two_gpu_families_are_available_for_card_transfer",
        "h800_holdout_has_been_inspected_in_prior_analysis",
        "complete_new_mechanisms_require_operator_basis_extension",
        "far_out_of_domain_inputs_must_not_receive_false_precision",
    ]
    if not packing_training_rows:
        limitations.insert(
            1,
            "packing_static_contract_exists_but_primary_model_has_no_packing_training_rows",
        )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "structured_model_fitted_and_blocked_cv_completed",
        "analysis_only": True,
        "publishable": False,
        "production_profile_generated": False,
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "model_contract": {
            "target": "effective_tokens_per_second",
            "single_output_used_for_absolute_and_ranking": True,
            "dataset_id_used_as_model_feature": False,
            "model_id_used_as_model_feature": False,
            "card_name_used_as_shared_model_feature": False,
            "runtime_outcome_features_used_as_inputs": False,
            "pre_run_static_profile_used": True,
            "feature_dimension": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "card_residual_feature_names": list(
                CARD_RESIDUAL_FEATURE_NAMES
            ),
            "positive_structured_components": list(COMPONENT_NAMES),
            "unknown_card_policy": (
                "global efficiency parameters, zero card adapter and "
                "out-of-domain confidence downgrade"
            ),
            "packing_contract_implemented_but_primary_training_evidence": (
                packing_evidence
            ),
            "packing_primary_training_rows": packing_training_rows,
            "analysis_extension_model_ids": extension_model_ids,
        },
        "objective": {
            "description": (
                "scenario/card-balanced robust absolute log-throughput + "
                "pairwise magnitude + pairwise logistic order + listwise "
                "KL + hierarchical regularization"
            ),
            "weights": dict(LOSS_WEIGHTS),
            "fixed_before_blocked_validation": True,
        },
        "data": {
            "h800_train_candidates": len(h800_train),
            "h800_train_scenarios": len(
                {candidate["scenario_id"] for candidate in h800_train}
            ),
            "h800_holdout_candidates": len(h800_holdout),
            "h800_holdout_scenarios": len(
                {candidate["scenario_id"] for candidate in h800_holdout}
            ),
            "rtx4090_candidates": len(rtx4090),
            "rtx4090_scenarios": len(
                {candidate["scenario_id"] for candidate in rtx4090}
            ),
            "rtx4090_admission": rtx_admission,
            "models": sorted(
                {
                    _candidate_model_id(candidate)
                    for candidate in final_train
                }
            ),
            "datasets": sorted(
                {
                    _candidate_dataset_id(candidate)
                    for candidate in final_train
                }
            ),
            "cards": ["h800", "rtx4090"],
        },
        "static_work_reconstruction": {
            "h800": _work_reconstruction_audit(
                [*h800_train, *h800_holdout]
            ),
            "rtx4090": _work_reconstruction_audit(rtx4090),
        },
        "validation": {
            "h800_native_holdout": h800_protocol,
            "rtx4090_complete_scenario_cv": rtx_scenario,
            "rtx4090_complete_model_holdout": rtx_model,
            "complete_dataset_holdout": dataset_holdout,
            "complete_card_holdout": card_holdout,
        },
        "uncertainty_reference": {
            "interpretation": (
                "empirical blocked-validation error references, not formal "
                "prediction intervals"
            ),
            "known_h800_native_p90_absolute_percentage_error": (
                h800_protocol["native_only"]["metrics"][
                    "absolute_throughput_ape_p90"
                ]
            ),
            "known_rtx4090_scenario_p90_absolute_percentage_error": (
                rtx_scenario["aggregate"][
                    "absolute_throughput_ape_p90"
                ]
            ),
            "unseen_model_p90_absolute_percentage_error": (
                rtx_model["aggregate"][
                    "absolute_throughput_ape_p90"
                ]
            ),
            "unseen_dataset_p90_absolute_percentage_error": (
                dataset_holdout["aggregate"][
                    "absolute_throughput_ape_p90"
                ]
            ),
            "unseen_rtx4090_from_h800_p90_absolute_percentage_error": (
                card_holdout["h800_to_rtx4090"]["metrics"][
                    "absolute_throughput_ape_p90"
                ]
            ),
            "unseen_h800_from_rtx4090_p90_absolute_percentage_error": (
                card_holdout["rtx4090_to_h800"]["metrics"][
                    "absolute_throughput_ape_p90"
                ]
            ),
            "confidence_policy": [
                "known_card_and_inside_training_support: use native/scenario reference",
                "unseen_model_or_dataset: use matching blocked-holdout reference",
                "unknown_card: zero all card adapters and use card-holdout reference",
                "outside_feature_or_resource_support: low confidence; require calibration experiment",
                packing_confidence_policy,
            ],
        },
        "prior_flat_joint_reference": {
            "path": str(prior_joint_report_path.resolve()),
            "sha256": sha256_file(prior_joint_report_path),
            "report_sha256": prior_joint["report_sha256"],
            "h800_holdout": prior_joint["h800"]["holdout"][
                "joint_single_head"
            ],
            "rtx4090_scenario_cv": prior_joint["rtx4090"][
                "nested_scenario_cv"
            ]["metrics"]["joint_single_head"],
            "rtx4090_model_holdout": prior_joint["rtx4090"][
                "nested_model_holdout"
            ]["metrics"]["joint_single_head"],
        },
        "frozen_model": frozen_model,
        "source_bindings": {
            "implementation": _source_binding(Path(__file__)),
            "static_workload_profiles": _source_binding(
                static_profile_path
            ),
            "h800_observations": _source_binding(
                h800_observations_path
            ),
            "h800_theory_basis": _source_binding(
                h800_theory_basis_path
            ),
            "h800_model_inventory": _source_binding(
                h800_inventory_path
            ),
            "h800_hardware": _source_binding(h800_hardware_path),
            "rtx4090_matrix": _source_binding(
                rtx4090_campaign_root
                / "matrix"
                / "throughput_jobs.jsonl"
            ),
            "rtx4090_hardware": _source_binding(rtx_hardware_path),
            "prior_joint_report": _source_binding(
                prior_joint_report_path
            ),
            "dataset_profiles": profiles.source_bindings(),
        },
        "limitations": limitations,
    }
    report["report_sha256"] = sha256_json(report)
    validate_report(report)
    return report


def validate_static_profile_report(report: Mapping[str, Any]) -> None:
    if report.get("schema") != STATIC_PROFILE_SCHEMA:
        raise ValueError("Static workload profile schema mismatch")
    unsigned = dict(report)
    digest = unsigned.pop("report_sha256", None)
    if digest != sha256_json(unsigned):
        raise ValueError("Static workload profile SHA mismatch")
    if report.get("uses_runtime_outcome_counters") is not False:
        raise ValueError("Static profile unexpectedly uses runtime outcomes")


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema") != SCHEMA:
        raise ValueError("Structured throughput report schema mismatch")
    unsigned = dict(report)
    digest = unsigned.pop("report_sha256", None)
    if digest != sha256_json(unsigned):
        raise ValueError("Structured throughput report SHA mismatch")
    if (
        report.get("gpu_experiments_launched") is not False
        or report.get("queues_mutated") is not False
        or report.get("publishable") is not False
        or report.get("production_profile_generated") is not False
    ):
        raise ValueError("Structured report unsafe-state flags drifted")
    contract = report.get("model_contract") or {}
    if (
        contract.get("runtime_outcome_features_used_as_inputs")
        is not False
        or contract.get("pre_run_static_profile_used") is not True
        or contract.get("feature_dimension") != len(FEATURE_NAMES)
        or contract.get("single_output_used_for_absolute_and_ranking")
        is not True
    ):
        raise ValueError("Structured model contract drifted")
    model = report.get("frozen_model") or {}
    if (
        model.get("feature_dimension") != len(FEATURE_NAMES)
        or tuple(model.get("feature_names") or ()) != FEATURE_NAMES
        or tuple(model.get("component_names") or ())
        != COMPONENT_NAMES
        or len(model.get("correction_coefficients") or ())
        != len(FEATURE_NAMES)
        or tuple(model.get("card_residual_feature_names") or ())
        != CARD_RESIDUAL_FEATURE_NAMES
        or any(
            len(coefficients) != len(CARD_RESIDUAL_FEATURE_NAMES)
            for coefficients in (
                model.get("card_residual_coefficients") or {}
            ).values()
        )
        or not model.get("training_support")
        or sorted(
            (
                model.get("training_support", {}).get("by_card")
                or {}
            )
        )
        != ["h800", "rtx4090"]
    ):
        raise ValueError("Frozen structured model dimensions drifted")
    if model.get("optimizer", {}).get("success") is not True:
        raise ValueError("Frozen structured model did not converge")
    gradient_audit = model.get("optimizer", {}).get(
        "analytic_gradient_audit"
    )
    if (
        not gradient_audit
        or gradient_audit.get("checked_parameters", 0) < 10
        or gradient_audit.get("max_scaled_error", math.inf) > 1.0e-5
    ):
        raise ValueError("Structured model analytic gradient audit failed")
    expected_models = {
        "qwen3_0p6b",
        "qwen3_14b",
        "qwen3_1p7b",
        "qwen3_4b",
        "qwen3_8b",
    }
    expected_models.update(contract.get("analysis_extension_model_ids") or ())
    if set(report["data"]["models"]) != expected_models:
        raise ValueError("Structured model inventory drifted")


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
        "--dataset-profile-dir",
        type=Path,
        default=ROOT / "artifacts" / "dataset_profiles",
    )
    parser.add_argument(
        "--static-profile-output",
        type=Path,
        default=ROOT / "artifacts" / "static_workload_profiles.json",
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
        "--rtx4090-campaign-root",
        type=Path,
        default=ROOT / "campaigns" / "rtx4090_20260717",
    )
    parser.add_argument(
        "--prior-joint-report",
        type=Path,
        default=ROOT / "artifacts" / "joint_throughput_modeling.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "structured_throughput_modeling.json",
    )
    parser.add_argument(
        "--analysis-extension-model-id",
        action="append",
        default=[],
        help=(
            "Explicit inventory-backed model ID admitted only for an analysis "
            "refit; repeat for multiple extensions."
        ),
    )
    args = parser.parse_args()

    profiles = StaticDatasetProfiles(args.dataset_profile_dir)
    static_report = profiles.report()
    validate_static_profile_report(static_report)
    write_json(args.static_profile_output, static_report)
    report = build_report(
        profiles=profiles,
        static_profile_path=args.static_profile_output,
        h800_observations_path=args.h800_observations,
        h800_theory_basis_path=args.h800_theory_basis,
        h800_inventory_path=args.h800_model_inventory,
        h800_hardware_path=args.h800_hardware,
        h800_runtime_root=args.h800_runtime_root,
        rtx4090_campaign_root=args.rtx4090_campaign_root,
        prior_joint_report_path=args.prior_joint_report,
        analysis_extension_model_ids=args.analysis_extension_model_id,
    )
    write_json(args.output, report)
    validation = report["validation"]
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "output": str(args.output),
                "static_profile_output": str(args.static_profile_output),
                "h800_native_holdout": {
                    name: _metric_summary(section["metrics"])
                    for name, section in validation[
                        "h800_native_holdout"
                    ].items()
                    if isinstance(section, Mapping)
                    and isinstance(section.get("metrics"), Mapping)
                },
                "rtx4090_scenario_cv": _metric_summary(
                    validation["rtx4090_complete_scenario_cv"][
                        "aggregate"
                    ]
                ),
                "rtx4090_model_holdout": _metric_summary(
                    validation["rtx4090_complete_model_holdout"][
                        "aggregate"
                    ]
                ),
                "dataset_holdout": _metric_summary(
                    validation["complete_dataset_holdout"]["aggregate"]
                ),
                "card_holdout": {
                    name: _metric_summary(section["metrics"])
                    for name, section in validation[
                        "complete_card_holdout"
                    ].items()
                    if isinstance(section, Mapping)
                    and isinstance(section.get("metrics"), Mapping)
                },
                "frozen_optimizer": report["frozen_model"]["optimizer"],
                "gpu_experiments_launched": report[
                    "gpu_experiments_launched"
                ],
                "report_sha256": report["report_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
