#!/usr/bin/env python3
"""Recalibrate the H800 critical-LoRA memory model from source-disjoint data.

This is a CPU-only, non-publishable modeling stage for the 2026-08-04
source-disjoint campaign.  It exports the exact 77 approved attempts, merges
the four earlier independent profile sources, collapses repeated configurations,
and compares the predeclared M0--M3 ablation:

* M0: the frozen bounded-v2 cutoff-based production control;
* M1: effective-sequence physical anchor, no ratio residual feature;
* M2: M1 plus log(effective_sequence / cutoff_len);
* M3: M2 plus a critical-LoRA interaction with that log ratio.

M1--M3 are evaluated with outer leave-source-out folds.  Center ridge penalties
are selected inside each outer fold, and success/OOM tails are regenerated from
inner leave-source-out predictions.  Repeats never increase the independent
source count.  With one outer source held out, only 18 critical sources remain;
the 95% finite-sample threshold is therefore reported as an empirical-maximum
diagnostic fallback (18/19 = 94.74%), not as formal 95% coverage.  The final
full-data critical-LoRA fit has 19 sources and can compute the finite-sample
rank-19 threshold, but it still requires a new prospective holdout before use.
"""

from __future__ import annotations

import argparse
import copy
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
    RESULTS_DIR,
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)
from export_h800_observations import validate_canonical_observation, write_jsonl
from fit_h800_profile_aware_memory_challenger_v1 import _export_exact_jobs
from h800_bounded_memory_model import predict_memory as predict_frozen_bounded
from h800_native_memory_calibration import (
    _inventory_models,
    build_native_record,
    native_admission_reason,
)
from h800_theory_basis import memory_basis
from h800_theory_calibration import (
    _observed_allocated,
    _observed_reserved,
    _oom_lower,
    _safe_limit,
)


GIB = float(1 << 30)
SCHEMA = "sft_h800_lora_source_disjoint_recalibration/v1"
CANDIDATE_SCHEMA = "sft_h800_critical_lora_memory_candidate/v1"
IMPLEMENTATION_VERSION = (
    "sft_h800_lora_source_disjoint_recalibration/"
    "2026-08-05.source-balanced-nested-m0-m3-v2"
)
DEFAULT_COVERAGE = 0.95
DEFAULT_PAD_MULTIPLE = 8
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

DEFAULT_NEW_QUEUE = MATRIX_DIR / "h800_lora_source_disjoint_jobs_v1.jsonl"
DEFAULT_OLD_OBSERVATIONS = (
    ARTIFACT_DIR / "h800_profile_aware_memory_calibration_observations_v1.jsonl"
)
DEFAULT_INVENTORY = ARTIFACT_DIR / "model_inventory.json"
DEFAULT_HARDWARE = ROOT / "config" / "hardware.json"
DEFAULT_FROZEN_BASELINE = ARTIFACT_DIR / "h800_bounded_memory_challenger_v2.json"
DEFAULT_OUTPUT_DIR = (
    ROOT / "diagnostics" / "h800_lora_memory_recalibration_20260804"
)

VARIANT_M0 = "M0_frozen_cutoff_control"
VARIANT_M1 = "M1_effective_sequence_no_ratio"
VARIANT_M2 = "M2_effective_sequence_global_log_ratio"
VARIANT_M3 = "M3_effective_sequence_critical_lora_interaction"
FITTED_VARIANTS = (VARIANT_M1, VARIANT_M2, VARIANT_M3)

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
)
VARIANT_FEATURES = {
    VARIANT_M1: BASE_FEATURES,
    VARIANT_M2: (*BASE_FEATURES, "log_effective_fraction"),
    VARIANT_M3: (
        *BASE_FEATURES,
        "log_effective_fraction",
        "critical_lora_x_log_effective_fraction",
    ),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _job(observation: Mapping[str, Any]) -> dict[str, Any]:
    configuration = observation.get("configuration") or {}
    value = configuration.get("job") if isinstance(configuration, Mapping) else None
    if not isinstance(value, Mapping):
        raise ValueError("canonical observation has no bound job")
    return dict(value)


def _source_id(record: Mapping[str, Any]) -> str:
    calibration = record.get("calibration_partition") or {}
    value = calibration.get("split_unit_id")
    if not isinstance(value, str) or not value:
        raise ValueError("record has no independent split_unit_id")
    return value


def _outcome(record: Mapping[str, Any]) -> str:
    value = record.get("outcome")
    if isinstance(value, Mapping):
        value = value.get("class")
    return str(value or "").lower()


def _round_up(value: int, multiple: int) -> int:
    value = int(value)
    multiple = int(multiple)
    if value <= 0 or multiple <= 0:
        raise ValueError("alignment operands must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def _profile_path(job: Mapping[str, Any]) -> Path:
    raw = job.get("dataset_profile_path")
    if not raw:
        padding = job.get("expected_padding_pressure") or {}
        raw = padding.get("path") if isinstance(padding, Mapping) else None
    if not raw:
        raise ValueError(f"job {job.get('job_id')} has no profile path")
    path = Path(str(raw))
    if not path.is_file():
        raise ValueError(f"profile does not exist: {path}")
    expected = job.get("dataset_profile_sha256")
    if expected and sha256_file(path) != expected:
        raise ValueError(f"profile SHA-256 mismatch: {path}")
    return path


def _raw_profile_max(path: Path) -> int:
    maximum = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            maximum = max(maximum, int(row["total_tokens"]))
    if maximum <= 0:
        raise ValueError(f"profile is empty: {path}")
    return maximum


def _mechanism_key(record: Mapping[str, Any]) -> str:
    selector = record.get("selector") or {}
    scenario = record.get("scenario") or {}
    return json.dumps(
        [
            str(selector.get("training_mode") or "unknown"),
            int(selector.get("zero_stage") or 0),
            bool(selector.get("gradient_checkpointing")),
            int(scenario.get("gpu_count") or 0),
            bool(selector.get("packing")),
        ],
        separators=(",", ":"),
    )


def _mode_gc_key(record: Mapping[str, Any]) -> str:
    selector = record.get("selector") or {}
    return json.dumps(
        [
            str(selector.get("training_mode") or "unknown"),
            bool(selector.get("gradient_checkpointing")),
        ],
        separators=(",", ":"),
    )


def _is_critical_lora(record: Mapping[str, Any]) -> bool:
    selector = record.get("selector") or {}
    scenario = record.get("scenario") or {}
    return bool(
        selector.get("training_mode") == "lora"
        and int(selector.get("zero_stage") or 0) == 2
        and not bool(selector.get("gradient_checkpointing"))
        and int(scenario.get("gpu_count") or 0) == 2
        and not bool(selector.get("packing"))
    )


def _cluster_id(record: Mapping[str, Any]) -> str:
    scenario = record.get("scenario") or {}
    selector = record.get("selector") or {}
    effective = record.get("effective_sequence") or {}
    material = {
        "source_id": _source_id(record),
        "model_id": scenario.get("model_id"),
        "training_mode": selector.get("training_mode"),
        "gpu_count": scenario.get("gpu_count"),
        "physical_mbs": scenario.get("physical_mbs"),
        "cutoff_len": scenario.get("cutoff_len"),
        "effective_sequence": effective.get("tokens"),
        "zero_stage": selector.get("zero_stage"),
        "gradient_checkpointing": selector.get("gradient_checkpointing"),
        "packing": selector.get("packing"),
    }
    return "memcfg-" + sha256_json(material)[:20]


def _observation_binding(observation: Mapping[str, Any]) -> dict[str, Any]:
    job = _job(observation)
    return {
        "observation_id": observation.get("observation_id"),
        "job_id": job.get("job_id"),
        "execution_attempt_id": (observation.get("attempt") or {}).get(
            "execution_attempt_id"
        ),
        "outcome": (observation.get("outcome") or {}).get("class"),
        "split_unit_id": (
            (observation.get("configuration") or {}).get("calibration_partition")
            or {}
        ).get("split_unit_id"),
        "model_id": job.get("model_id"),
        "train_type": job.get("train_type"),
        "dataset_id": job.get("dataset_id"),
        "gpu_count": job.get("gpu_count"),
        "mbs": job.get("mbs"),
        "cutoff_len": job.get("cutoff_len"),
        "raw_profile_max": job.get("raw_profile_max"),
        "aligned_effective_sequence": job.get("aligned_effective_sequence"),
        "experiment_group": job.get("experiment_group"),
        "repeat": job.get("repeat"),
        "memory": dict((observation.get("measurements") or {}).get("memory") or {}),
        "censoring": observation.get("censoring"),
        "quality": {
            key: (observation.get("quality") or {}).get(key)
            for key in (
                "rank_events_complete",
                "measured_ranks_complete",
                "fingerprint_quality",
                "evidence_verified",
                "event_attempt_binding_complete",
                "terminal_label_verified",
                "calibration_partition_bound",
            )
        },
    }


def _build_record_pair(
    observation: dict[str, Any],
    *,
    model_by_id: Mapping[str, dict[str, Any]],
    fixed_lora: dict[str, Any],
    hardware: dict[str, Any],
    padding_cache: dict[tuple[str, int, int], dict[str, Any]],
    raw_max_cache: dict[str, int],
    origin: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if native_admission_reason(observation) != "admitted":
        raise ValueError(
            f"observation rejected: {native_admission_reason(observation)}"
        )
    cutoff_record = build_native_record(
        observation,
        model_by_id=model_by_id,
        fixed_lora=fixed_lora,
        hardware=hardware,
    )
    job = _job(observation)
    profile_path = _profile_path(job)
    cutoff = int(job["cutoff_len"])
    mbs = int(job["mbs"])
    cache_key = (str(profile_path.resolve()), cutoff, mbs)
    if cache_key not in padding_cache:
        padding_cache[cache_key] = profile_padding_statistics(
            profile_path, cutoff_len=cutoff, physical_mbs=mbs
        )
    padding = dict(padding_cache[cache_key])
    raw_key = str(profile_path.resolve())
    if raw_key not in raw_max_cache:
        raw_max_cache[raw_key] = _raw_profile_max(profile_path)
    raw_max = int(raw_max_cache[raw_key])
    clipped_max = int(padding["maximum_clipped_tokens"])
    sequence = _round_up(min(cutoff, clipped_max), DEFAULT_PAD_MULTIPLE)
    declared_sequence = job.get("aligned_effective_sequence")
    if declared_sequence is not None and int(declared_sequence) != sequence:
        raise ValueError(
            f"effective sequence drifted for {job.get('job_id')}: "
            f"{declared_sequence} != {sequence}"
        )
    declared_raw = job.get("raw_profile_max")
    if declared_raw is not None and int(declared_raw) != raw_max:
        raise ValueError(
            f"raw profile max drifted for {job.get('job_id')}: "
            f"{declared_raw} != {raw_max}"
        )

    cutoff_record = copy.deepcopy(cutoff_record)
    effective_record = copy.deepcopy(cutoff_record)
    selector = effective_record["selector"]
    scenario = effective_record["scenario"]
    rebuilt = memory_basis(
        {
            "gpu_count": int(scenario["gpu_count"]),
            "mbs": int(scenario["physical_mbs"]),
            "cutoff_len": sequence,
            "zero": f"zero{int(selector.get('zero_stage') or 0)}",
            "gc": bool(selector.get("gradient_checkpointing")),
        },
        effective_record["model_basis"],
        int(hardware["memory_bytes_reported_by_torch"]),
    )
    rebuilt["observed"] = effective_record["memory"]["observed"]
    effective_record["memory"] = rebuilt
    contract = {
        "policy": "round_up(min(cutoff_len, profile_max), 8)",
        "tokens": sequence,
        "cutoff_len": cutoff,
        "fraction_of_cutoff": sequence / float(cutoff),
        "raw_profile_max": raw_max,
        "raw_profile_max_over_cutoff": raw_max / float(cutoff),
        "maximum_clipped_tokens": clipped_max,
        "profile_path": str(profile_path.resolve()),
        "profile_sha256": sha256_file(profile_path),
    }
    metadata = {
        "origin": origin,
        "campaign_id": job.get("campaign_id"),
        "experiment_group": job.get("experiment_group") or "legacy_profile_calibration",
        "repeat": int(job.get("repeat") or 0),
        "job_id": job.get("job_id"),
        "observation_id": observation.get("observation_id"),
        "padding": padding,
    }
    for record in (cutoff_record, effective_record):
        record["effective_sequence"] = dict(contract)
        record["recalibration"] = dict(metadata)
        record["cluster_id"] = _cluster_id(record)
    return cutoff_record, effective_record


def _collapse_records(
    cutoff_records: Sequence[dict[str, Any]],
    effective_records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cutoff_by_id = {str(row["observation_id"]): row for row in cutoff_records}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in effective_records:
        grouped[str(row["cluster_id"])].append(row)
    collapsed_effective: list[dict[str, Any]] = []
    collapsed_cutoff: list[dict[str, Any]] = []
    repeat_groups: list[dict[str, Any]] = []
    for cluster_id, repeats in sorted(grouped.items()):
        outcomes = {_outcome(row) for row in repeats}
        if len(outcomes) != 1:
            raise ValueError(f"repeat outcomes disagree for {cluster_id}: {outcomes}")
        first = copy.deepcopy(repeats[0])
        observation_ids = sorted(str(row["observation_id"]) for row in repeats)
        first["observation_id"] = "collapsed-" + sha256_json(observation_ids)[:24]
        first["recalibration"]["repeat_count"] = len(repeats)
        first["recalibration"]["observation_ids"] = observation_ids
        first["recalibration"]["job_ids"] = sorted(
            str((row.get("recalibration") or {}).get("job_id")) for row in repeats
        )
        if _outcome(first) == "success":
            allocated = [float(_observed_allocated(row)) for row in repeats]
            reserved = [float(_observed_reserved(row)) for row in repeats]
            observed = first["memory"]["observed"]
            observed["peak_allocated_diagnostic_bytes"] = statistics.median(allocated)
            observed["peak_reserved_target_bytes"] = statistics.median(reserved)
            observed["allocator_reserved_minus_allocated_bytes"] = (
                observed["peak_reserved_target_bytes"]
                - observed["peak_allocated_diagnostic_bytes"]
            )
            if len(repeats) > 1:
                repeat_groups.append(
                    {
                        "cluster_id": cluster_id,
                        "source_id": _source_id(first),
                        "outcome": "success",
                        "repeat_count": len(repeats),
                        "allocated_gib": [value / GIB for value in allocated],
                        "reserved_gib": [value / GIB for value in reserved],
                        "allocated_relative_range": (
                            (max(allocated) - min(allocated)) / statistics.median(allocated)
                        ),
                        "reserved_relative_range": (
                            (max(reserved) - min(reserved)) / statistics.median(reserved)
                        ),
                    }
                )
        elif len(repeats) > 1:
            repeat_groups.append(
                {
                    "cluster_id": cluster_id,
                    "source_id": _source_id(first),
                    "outcome": "oom",
                    "repeat_count": len(repeats),
                    "oom_reproduced": True,
                }
            )
        collapsed_effective.append(first)

        cutoff_first = copy.deepcopy(cutoff_by_id[observation_ids[0]])
        cutoff_first["observation_id"] = first["observation_id"]
        cutoff_first["recalibration"] = copy.deepcopy(first["recalibration"])
        cutoff_first["cluster_id"] = cluster_id
        cutoff_first["memory"]["observed"] = copy.deepcopy(first["memory"]["observed"])
        collapsed_cutoff.append(cutoff_first)
    return collapsed_cutoff, collapsed_effective, {
        "raw_rows": len(effective_records),
        "unique_configurations": len(collapsed_effective),
        "collapsed_repeat_rows": len(effective_records) - len(collapsed_effective),
        "repeat_groups": repeat_groups,
    }


def _feature_values(record: Mapping[str, Any]) -> dict[str, float]:
    selector = record.get("selector") or {}
    scenario = record.get("scenario") or {}
    memory = record.get("memory") or {}
    model = record.get("model_basis") or {}
    effective = record.get("effective_sequence") or {}
    reference = float(memory["analytic_reference_bytes"])
    activation = float(memory["structural_activation_bytes"])
    is_lora = float(selector.get("training_mode") == "lora")
    gc = float(bool(selector.get("gradient_checkpointing")))
    zero = int(selector.get("zero_stage") or 0)
    log_gpu = math.log2(float(scenario["gpu_count"]))
    log_mbs = math.log2(float(scenario["physical_mbs"]))
    log_params = math.log2(float(model["base_parameters"]) / 4_022_468_096.0)
    fraction = min(1.0, max(1e-12, float(effective["fraction_of_cutoff"])))
    log_fraction = math.log(fraction)
    critical = float(_is_critical_lora(record))
    activation_share = activation / reference
    return {
        "is_lora": is_lora,
        "gradient_checkpointing": gc,
        "zero2": float(zero == 2),
        "zero3": float(zero == 3),
        "log2_gpu_count": log_gpu,
        "log2_mbs": log_mbs,
        "log2_parameters_over_4b": log_params,
        "activation_share": activation_share,
        "is_lora_x_activation_share": is_lora * activation_share,
        "is_lora_x_log2_parameters": is_lora * log_params,
        "activation_share_x_log2_mbs": activation_share * log_mbs,
        "log_effective_fraction": log_fraction,
        "critical_lora_x_log_effective_fraction": critical * log_fraction,
    }


def _vector(record: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    values = _feature_values(record)
    return np.asarray([values[str(name)] for name in names], dtype=float)


def _label(record: Mapping[str, Any], target: str) -> float | None:
    if target == "allocated":
        return _observed_allocated(record)
    if target == "reserved":
        return _observed_reserved(record)
    raise ValueError(f"unsupported target: {target}")


def _fit_ridge(
    records: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str],
    alpha: float,
    target: str,
) -> dict[str, Any]:
    successes = [
        row
        for row in records
        if _outcome(row) == "success" and _label(row, target) is not None
    ]
    if not successes:
        raise ValueError("no success records for ridge")
    features = np.vstack([_vector(row, names) for row in successes])
    targets = np.asarray(
        [
            math.log(
                float(_label(row, target))
                / float(row["memory"]["analytic_reference_bytes"])
            )
            for row in successes
        ],
        dtype=float,
    )
    counts = Counter(_source_id(row) for row in successes)
    weights = np.asarray(
        [1.0 / counts[_source_id(row)] for row in successes], dtype=float
    )
    means = np.average(features, axis=0, weights=weights)
    scales = np.sqrt(np.average((features - means) ** 2, axis=0, weights=weights))
    scales[scales < 1e-9] = 1.0
    standardized = (features - means) / scales
    design = np.column_stack((np.ones(len(successes)), standardized))
    penalty = np.diag([0.0, *([float(alpha)] * len(names))])
    coefficients = np.linalg.pinv(
        design.T @ (weights[:, None] * design) + penalty
    ) @ (design.T @ (weights * targets))
    raw_coefficients = coefficients[1:] / scales
    raw_intercept = float(coefficients[0] - means @ raw_coefficients)
    return {
        "model_family": "source_balanced_analytic_log_residual_ridge",
        "target": target,
        "feature_names": list(names),
        "alpha": float(alpha),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "raw_intercept": raw_intercept,
        "raw_coefficients": raw_coefficients.tolist(),
        "fit_success_rows": len(successes),
        "fit_independent_sources": len(counts),
        "weighting": "each split_unit_id has total weight one",
    }


def _predict_center(record: Mapping[str, Any], model: Mapping[str, Any]) -> float:
    vector = _vector(record, model["feature_names"])
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    correction = float(model["intercept"]) + float(
        ((vector - means) / scales) @ coefficients
    )
    prediction = float(record["memory"]["analytic_reference_bytes"]) * math.exp(
        correction
    )
    if not math.isfinite(prediction) or prediction <= 0:
        raise ValueError("ridge produced a non-positive prediction")
    return prediction


def _source_equal_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None and math.isfinite(float(value)):
            grouped[str(row["source_id"])].append(float(value))
    if not grouped:
        return None
    return statistics.fmean(statistics.fmean(values) for values in grouped.values())


def _select_alpha(
    records: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str],
    target: str,
) -> dict[str, Any]:
    sources = sorted({_source_id(row) for row in records})
    candidates: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        details: list[dict[str, Any]] = []
        for held in sources:
            train = [row for row in records if _source_id(row) != held]
            test = [
                row
                for row in records
                if _source_id(row) == held
                and _outcome(row) == "success"
                and _label(row, target) is not None
            ]
            if not test:
                continue
            model = _fit_ridge(train, names=names, alpha=alpha, target=target)
            for row in test:
                observed = float(_label(row, target))
                predicted = _predict_center(row, model)
                details.append(
                    {
                        "source_id": held,
                        "ape": abs(predicted / observed - 1.0),
                    }
                )
        candidates.append(
            {
                "alpha": float(alpha),
                "source_equal_mape": _source_equal_mean(details, "ape"),
                "rows": len(details),
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            float(row["source_equal_mape"]),
            -float(row["alpha"]),
        ),
    )
    return {"selected": selected, "candidates": candidates}


def _inner_oof(
    records: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str],
    allocated_alpha: float,
    reserved_alpha: float,
) -> dict[str, list[dict[str, Any]]]:
    allocated_scores: list[dict[str, Any]] = []
    reserved_scores: list[dict[str, Any]] = []
    oom_scores: list[dict[str, Any]] = []
    sources = sorted({_source_id(row) for row in records})
    for held in sources:
        train = [row for row in records if _source_id(row) != held]
        test = [row for row in records if _source_id(row) == held]
        allocated_model = _fit_ridge(
            train, names=names, alpha=allocated_alpha, target="allocated"
        )
        reserved_model = _fit_ridge(
            train, names=names, alpha=reserved_alpha, target="reserved"
        )
        for row in test:
            base = {
                "source_id": held,
                "mechanism_key": _mechanism_key(row),
                "mode_gc_key": _mode_gc_key(row),
            }
            if _outcome(row) == "success":
                allocated = float(_observed_allocated(row))
                reserved = float(_observed_reserved(row))
                allocated_scores.append(
                    {
                        **base,
                        "score": math.log(
                            allocated / _predict_center(row, allocated_model)
                        ),
                    }
                )
                reserved_scores.append(
                    {
                        **base,
                        "score": math.log(
                            reserved / _predict_center(row, reserved_model)
                        ),
                    }
                )
            elif _outcome(row) == "oom":
                safe = _safe_limit(row)
                lower = _oom_lower(row)
                floors = [
                    float(value)
                    for value in (lower, (safe + 1.0) if safe is not None else None)
                    if value is not None
                ]
                if floors:
                    oom_scores.append(
                        {
                            **base,
                            "score": math.log(
                                max(floors) / _predict_center(row, reserved_model)
                            ),
                        }
                    )
    return {
        "allocated": allocated_scores,
        "reserved": reserved_scores,
        "oom": oom_scores,
    }


def _collapse_worst(
    rows: Sequence[Mapping[str, Any]], key: str | None
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        bucket = str(row[key]) if key else "pooled"
        source = str(row["source_id"])
        grouped[bucket][source] = max(
            float(row["score"]), grouped[bucket].get(source, -math.inf)
        )
    return dict(grouped)


def _upper_quantile(
    values: Mapping[str, float],
    *,
    coverage: float,
    diagnostic_max_fallback: bool,
) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values.values())
    rank = math.ceil((len(ordered) + 1) * float(coverage))
    formal = bool(ordered) and rank <= len(ordered)
    fallback = bool(ordered) and not formal and diagnostic_max_fallback
    selected = (
        ordered[rank - 1]
        if formal
        else (ordered[-1] if fallback else None)
    )
    return {
        "available": selected is not None,
        "formal_finite_sample": formal,
        "diagnostic_empirical_max_fallback": fallback,
        "requested_coverage": float(coverage),
        "independent_sources": len(ordered),
        "rank": rank,
        "maximum_guaranteed_coverage_if_fallback": (
            len(ordered) / float(len(ordered) + 1) if ordered else None
        ),
        "log_upper": selected,
        "empirical_max": ordered[-1] if ordered else None,
    }


def _fit_residual_hierarchy(
    scores: Sequence[Mapping[str, Any]],
    *,
    coverage: float,
    diagnostic_max_fallback: bool,
) -> dict[str, Any]:
    exact = _collapse_worst(scores, "mechanism_key")
    mode_gc = _collapse_worst(scores, "mode_gc_key")
    pooled = _collapse_worst(scores, None).get("pooled", {})
    return {
        "score_unit": "worst configuration per independent split_unit_id",
        "exact": {
            key: _upper_quantile(
                values,
                coverage=coverage,
                diagnostic_max_fallback=diagnostic_max_fallback,
            )
            for key, values in sorted(exact.items())
        },
        "mode_gc": {
            key: _upper_quantile(
                values,
                coverage=coverage,
                diagnostic_max_fallback=diagnostic_max_fallback,
            )
            for key, values in sorted(mode_gc.items())
        },
        "pooled": _upper_quantile(
            pooled,
            coverage=coverage,
            diagnostic_max_fallback=diagnostic_max_fallback,
        ),
        "hierarchy": ["exact_mechanism", "mode_x_gc", "pooled"],
    }


def _select_residual(
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
    return {"available": False, "source": None, "log_upper": None}


def _fit_expansion(
    records: Sequence[Mapping[str, Any]],
    *,
    coverage: float,
    diagnostic_max_fallback: bool,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    row_counts: Counter[str] = Counter()
    for row in records:
        if _outcome(row) != "success":
            continue
        allocated = _observed_allocated(row)
        reserved = _observed_reserved(row)
        if allocated is None or reserved is None:
            continue
        key = _mechanism_key(row)
        source = _source_id(row)
        score = math.log(float(reserved) / float(allocated))
        grouped[key][source] = max(score, grouped[key].get(source, -math.inf))
        row_counts[key] += 1
    entries = {}
    for key, values in sorted(grouped.items()):
        entry = _upper_quantile(
            values,
            coverage=coverage,
            diagnostic_max_fallback=diagnostic_max_fallback,
        )
        entry["rows"] = row_counts[key]
        entry["expansion_upper"] = (
            math.exp(float(entry["log_upper"]))
            if entry.get("available")
            else None
        )
        entry["empirical_max_expansion"] = (
            math.exp(float(entry["empirical_max"]))
            if entry.get("empirical_max") is not None
            else None
        )
        entries[key] = entry
    return {"entries": entries, "no_pooled_fallback": True}


def _fit_oom_guards(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = _collapse_worst(scores, "mechanism_key")
    return {
        key: {
            "independent_sources": len(values),
            "log_lower": max(values.values()),
        }
        for key, values in sorted(grouped.items())
        if values
    }


def _fit_bundle(
    records: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str],
    coverage: float,
    diagnostic_max_fallback: bool,
) -> dict[str, Any]:
    allocated_selection = _select_alpha(records, names=names, target="allocated")
    reserved_selection = _select_alpha(records, names=names, target="reserved")
    allocated_alpha = float(allocated_selection["selected"]["alpha"])
    reserved_alpha = float(reserved_selection["selected"]["alpha"])
    allocated_model = _fit_ridge(
        records, names=names, alpha=allocated_alpha, target="allocated"
    )
    reserved_model = _fit_ridge(
        records, names=names, alpha=reserved_alpha, target="reserved"
    )
    oof = _inner_oof(
        records,
        names=names,
        allocated_alpha=allocated_alpha,
        reserved_alpha=reserved_alpha,
    )
    return {
        "feature_names": list(names),
        "coverage": coverage,
        "diagnostic_max_fallback": diagnostic_max_fallback,
        "allocated_alpha_selection": allocated_selection,
        "reserved_alpha_selection": reserved_selection,
        "allocated_model": allocated_model,
        "reserved_model": reserved_model,
        "allocated_residual_upper": _fit_residual_hierarchy(
            oof["allocated"],
            coverage=coverage,
            diagnostic_max_fallback=diagnostic_max_fallback,
        ),
        "reserved_residual_upper": _fit_residual_hierarchy(
            oof["reserved"],
            coverage=coverage,
            diagnostic_max_fallback=diagnostic_max_fallback,
        ),
        "reservation_expansion": _fit_expansion(
            records,
            coverage=coverage,
            diagnostic_max_fallback=diagnostic_max_fallback,
        ),
        "oom_guards": _fit_oom_guards(oof["oom"]),
        "inner_oof": {key: len(value) for key, value in oof.items()},
    }


def _predict_stack(
    record: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    allocated_center = _predict_center(record, bundle["allocated_model"])
    reserved_center = _predict_center(record, bundle["reserved_model"])
    allocated_tail = _select_residual(
        record, bundle["allocated_residual_upper"]
    )
    reserved_tail = _select_residual(record, bundle["reserved_residual_upper"])
    if (
        allocated_tail.get("available") is not True
        or reserved_tail.get("available") is not True
    ):
        return {
            "available": False,
            "allocated_center_bytes": allocated_center,
            "reserved_center_bytes": reserved_center,
            "upper_bytes": None,
            "issues": ["success_residual_upper_unavailable"],
        }
    allocated_upper = allocated_center * math.exp(
        max(0.0, float(allocated_tail["log_upper"]))
    )
    direct_reserved_upper = reserved_center * math.exp(
        max(0.0, float(reserved_tail["log_upper"]))
    )
    mechanism = _mechanism_key(record)
    expansion = (
        (bundle.get("reservation_expansion") or {}).get("entries") or {}
    ).get(mechanism)
    expansion_upper = None
    if isinstance(expansion, Mapping) and expansion.get("available") is True:
        expansion_upper = allocated_upper * float(expansion["expansion_upper"])
    if _is_critical_lora(record) and expansion_upper is None:
        return {
            "available": False,
            "allocated_center_bytes": allocated_center,
            "reserved_center_bytes": reserved_center,
            "upper_bytes": None,
            "issues": ["critical_expansion_guard_unavailable_fail_closed"],
        }
    oom = (bundle.get("oom_guards") or {}).get(mechanism)
    oom_upper = None
    if isinstance(oom, Mapping):
        oom_upper = reserved_center * math.exp(max(0.0, float(oom["log_lower"])))
    candidates = [direct_reserved_upper]
    if expansion_upper is not None:
        candidates.append(expansion_upper)
    if oom_upper is not None:
        candidates.append(oom_upper)
    return {
        "available": True,
        "allocated_center_bytes": allocated_center,
        "reserved_center_bytes": reserved_center,
        "allocated_upper_bytes": allocated_upper,
        "direct_reserved_upper_bytes": direct_reserved_upper,
        "expansion_upper_bytes": expansion_upper,
        "oom_upper_bytes": oom_upper,
        "upper_bytes": max(candidates),
        "allocated_tail": allocated_tail,
        "reserved_tail": reserved_tail,
        "expansion": dict(expansion or {}),
        "issues": (
            [] if expansion_upper is not None else ["direct_reserved_path_only"]
        ),
    }


def _prediction_detail(
    record: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    variant: str,
) -> dict[str, Any]:
    available = prediction.get("available") is True
    allocated = _observed_allocated(record)
    reserved = _observed_reserved(record)
    upper = (
        float(prediction["upper_bytes"])
        if available and prediction.get("upper_bytes") is not None
        else None
    )
    safe = float(record["memory"]["safe_limit_bytes"])
    return {
        "variant": variant,
        "cluster_id": record.get("cluster_id"),
        "source_id": _source_id(record),
        "origin": (record.get("recalibration") or {}).get("origin"),
        "model_id": (record.get("scenario") or {}).get("model_id"),
        "training_mode": (record.get("selector") or {}).get("training_mode"),
        "gpu_count": (record.get("scenario") or {}).get("gpu_count"),
        "mbs": (record.get("scenario") or {}).get("physical_mbs"),
        "cutoff_len": (record.get("scenario") or {}).get("cutoff_len"),
        "effective_sequence": (record.get("effective_sequence") or {}).get("tokens"),
        "effective_fraction": (record.get("effective_sequence") or {}).get(
            "fraction_of_cutoff"
        ),
        "outcome": _outcome(record),
        "critical_lora": _is_critical_lora(record),
        "prediction_available": available,
        "allocated_center_bytes": prediction.get("allocated_center_bytes"),
        "reserved_center_bytes": prediction.get("reserved_center_bytes"),
        "upper_bytes": upper,
        "safe_limit_bytes": safe,
        "admitted": bool(available and upper is not None and upper <= safe),
        "observed_allocated_bytes": allocated,
        "observed_reserved_bytes": reserved,
        "allocated_ape": (
            abs(float(prediction["allocated_center_bytes"]) / allocated - 1.0)
            if allocated is not None and prediction.get("allocated_center_bytes")
            else None
        ),
        "reserved_ape": (
            abs(float(prediction["reserved_center_bytes"]) / reserved - 1.0)
            if reserved is not None and prediction.get("reserved_center_bytes")
            else None
        ),
        "reserved_covered": (
            bool(upper >= reserved) if upper is not None and reserved is not None else None
        ),
        "actually_safe_success": (
            bool(reserved <= safe) if _outcome(record) == "success" and reserved else None
        ),
        "issues": list(prediction.get("issues") or []),
    }


def _evaluate(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = [row for row in details if row["outcome"] == "success"]
    ooms = [row for row in details if row["outcome"] == "oom"]
    safe_successes = [row for row in successes if row["actually_safe_success"]]
    unsafe_successes = [row for row in successes if not row["actually_safe_success"]]
    available = [row for row in details if row["prediction_available"]]
    scored_successes = [row for row in successes if row["reserved_covered"] is not None]
    return {
        "configurations": len(details),
        "independent_sources": len({str(row["source_id"]) for row in details}),
        "prediction_available": len(available),
        "prediction_availability": len(available) / len(details) if details else None,
        "success_configurations": len(successes),
        "oom_configurations": len(ooms),
        "allocated_center_row_mape": (
            statistics.fmean(float(row["allocated_ape"]) for row in successes)
            if successes
            else None
        ),
        "allocated_center_source_equal_mape": _source_equal_mean(
            successes, "allocated_ape"
        ),
        "reserved_center_row_mape": (
            statistics.fmean(float(row["reserved_ape"]) for row in successes)
            if successes and all(row["reserved_ape"] is not None for row in successes)
            else None
        ),
        "reserved_center_source_equal_mape": _source_equal_mean(
            successes, "reserved_ape"
        ),
        "reserved_upper_row_coverage": (
            statistics.fmean(float(bool(row["reserved_covered"])) for row in scored_successes)
            if scored_successes
            else None
        ),
        "reserved_upper_source_equal_coverage": _source_equal_mean(
            [
                {**dict(row), "coverage": float(bool(row["reserved_covered"]))}
                for row in scored_successes
            ],
            "coverage",
        ),
        "reserved_upper_coverage_with_fail_closed": (
            (
                sum(bool(row["reserved_covered"]) for row in scored_successes)
                + len(successes)
                - len(scored_successes)
            )
            / len(successes)
            if successes
            else None
        ),
        "safe_success_configurations": len(safe_successes),
        "admitted_safe_success_configurations": sum(
            bool(row["admitted"]) for row in safe_successes
        ),
        "admission_recall": (
            sum(bool(row["admitted"]) for row in safe_successes) / len(safe_successes)
            if safe_successes
            else None
        ),
        "unsafe_success_admitted": sum(
            bool(row["admitted"]) for row in unsafe_successes
        ),
        "false_safe_oom": sum(bool(row["admitted"]) for row in ooms),
    }


def _nested_loso(
    records: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    coverage: float,
) -> dict[str, Any]:
    names = VARIANT_FEATURES[variant]
    sources = sorted({_source_id(row) for row in records})
    details: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for index, held in enumerate(sources, start=1):
        print(f"{variant}: outer fold {index}/{len(sources)} holdout={held}", flush=True)
        training = [row for row in records if _source_id(row) != held]
        evaluation = [row for row in records if _source_id(row) == held]
        bundle = _fit_bundle(
            training,
            names=names,
            coverage=coverage,
            diagnostic_max_fallback=True,
        )
        for row in evaluation:
            details.append(
                _prediction_detail(
                    row,
                    _predict_stack(row, bundle),
                    variant=variant,
                )
            )
        coefficient_rows.append(
            {
                "held_source_id": held,
                "allocated": dict(
                    zip(
                        bundle["allocated_model"]["feature_names"],
                        bundle["allocated_model"]["raw_coefficients"],
                    )
                ),
                "reserved": dict(
                    zip(
                        bundle["reserved_model"]["feature_names"],
                        bundle["reserved_model"]["raw_coefficients"],
                    )
                ),
            }
        )
        critical_key = json.dumps(["lora", 2, False, 2, False], separators=(",", ":"))
        expansion = (
            (bundle["reservation_expansion"].get("entries") or {}).get(critical_key)
            or {}
        )
        folds.append(
            {
                "held_source_id": held,
                "training_sources": len({_source_id(row) for row in training}),
                "evaluation_configurations": len(evaluation),
                "allocated_alpha": bundle["allocated_model"]["alpha"],
                "reserved_alpha": bundle["reserved_model"]["alpha"],
                "critical_expansion_sources": expansion.get("independent_sources"),
                "critical_expansion_formal": expansion.get("formal_finite_sample"),
                "critical_expansion_diagnostic_fallback": expansion.get(
                    "diagnostic_empirical_max_fallback"
                ),
            }
        )
    ratio_features = [
        name
        for name in (
            "log_effective_fraction",
            "critical_lora_x_log_effective_fraction",
        )
        if name in names
    ]
    stability = {}
    for target in ("allocated", "reserved"):
        for feature in ratio_features:
            values = [float(row[target][feature]) for row in coefficient_rows]
            nonzero = [value for value in values if abs(value) > 1e-12]
            sign = 1.0 if statistics.median(values) >= 0 else -1.0
            stability[f"{target}:{feature}"] = {
                "folds": len(values),
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
                "sign_consistency": (
                    sum(value * sign >= 0 for value in nonzero) / len(nonzero)
                    if nonzero
                    else 1.0
                ),
                "sign_changes_across_range": min(values) < 0 < max(values),
            }
    return {
        "protocol": (
            "outer leave split_unit_id out; alpha selected by inner leave-source-out; "
            "tails regenerated from inner OOF source scores; repeats collapsed"
        ),
        "variant": variant,
        "feature_names": list(names),
        "folds": folds,
        "metrics": _evaluate(details),
        "coefficient_stability": stability,
        "details": details,
    }


def _evaluate_frozen_m0(
    cutoff_records: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for record in cutoff_records:
        if (record.get("recalibration") or {}).get("origin") != "new_77":
            continue
        padding = (record.get("recalibration") or {}).get("padding") or {}
        result = predict_frozen_bounded(
            record,
            padding,
            allocated_anchor_bytes=float(record["memory"]["analytic_reference_bytes"]),
            artifact=artifact,
        )
        if result.get("available") is not True:
            unsupported.append(
                {
                    "cluster_id": record.get("cluster_id"),
                    "source_id": _source_id(record),
                    "model_id": (record.get("scenario") or {}).get("model_id"),
                    "training_mode": (record.get("selector") or {}).get(
                        "training_mode"
                    ),
                    "gpu_count": (record.get("scenario") or {}).get("gpu_count"),
                    "mbs": (record.get("scenario") or {}).get("physical_mbs"),
                    "issues": result.get("issues"),
                }
            )
            continue
        prediction = {
            "available": True,
            "allocated_center_bytes": result["allocated_center_bytes"],
            "reserved_center_bytes": result["reserved_center_bytes"],
            "upper_bytes": result["operational_upper_reserved_bytes"],
            "issues": [],
        }
        details.append(_prediction_detail(record, prediction, variant=VARIANT_M0))
    return {
        "evaluation_kind": "frozen_model_on_new_calibration_data_not_acceptance",
        "supported_configurations": len(details),
        "unsupported_configurations": len(unsupported),
        "metrics": _evaluate(details),
        "details": details,
        "unsupported": unsupported,
    }


def _common_support_metrics(
    m0: Mapping[str, Any], nested: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    clusters = {str(row["cluster_id"]) for row in m0.get("details") or []}
    result = {VARIANT_M0: m0["metrics"]}
    for variant, report in nested.items():
        details = [
            row
            for row in report.get("details") or []
            if row.get("origin") == "new_77" and str(row.get("cluster_id")) in clusters
        ]
        result[variant] = _evaluate(details)
    return {
        "cluster_count": len(clusters),
        "purpose": "same-row M0--M3 comparison on frozen-M0 supported new configurations",
        "metrics": result,
    }


def _gate_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "false_safe_oom_zero": int(metrics.get("false_safe_oom") or 0) == 0,
        "unsafe_success_admitted_zero": int(
            metrics.get("unsafe_success_admitted") or 0
        )
        == 0,
        "reserved_upper_source_equal_coverage_ge_0p95": float(
            metrics.get("reserved_upper_source_equal_coverage") or 0.0
        )
        >= 0.95,
        "allocated_center_source_equal_mape_le_0p10": float(
            metrics.get("allocated_center_source_equal_mape") or math.inf
        )
        <= 0.10,
        "prediction_availability_one": float(
            metrics.get("prediction_availability") or 0.0
        )
        == 1.0,
    }
    return {"checks": checks, "all_passed": all(checks.values())}


def _ratio_stable(report: Mapping[str, Any]) -> bool:
    entries = report.get("coefficient_stability") or {}
    if not entries:
        return True
    return all(
        float(entry.get("sign_consistency") or 0.0) >= 0.80
        and entry.get("sign_changes_across_range") is not True
        for entry in entries.values()
    )


def _select_variant(nested: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    gates = {
        variant: _gate_metrics(report["metrics"])
        for variant, report in nested.items()
    }
    base_mape = float(
        nested[VARIANT_M1]["metrics"]["allocated_center_source_equal_mape"]
    )
    eligible = []
    for variant in FITTED_VARIANTS:
        mape = float(
            nested[variant]["metrics"]["allocated_center_source_equal_mape"]
        )
        stability = _ratio_stable(nested[variant])
        material_improvement = variant == VARIANT_M1 or (base_mape - mape >= 0.01)
        if gates[variant]["all_passed"] and stability and material_improvement:
            eligible.append((mape, len(VARIANT_FEATURES[variant]), variant))
    selected = min(eligible)[2] if eligible else None
    if selected is None and gates[VARIANT_M1]["all_passed"]:
        selected = VARIANT_M1
    return {
        "policy": (
            "safety gates first; ratio variants require >=1 percentage-point "
            "source-equal allocated-MAPE improvement and >=80% coefficient sign stability; "
            "otherwise prefer M1"
        ),
        "gates": gates,
        "ratio_stability": {
            variant: _ratio_stable(nested[variant]) for variant in FITTED_VARIANTS
        },
        "selected_variant": selected,
        "eligible_variants": [row[2] for row in sorted(eligible)],
    }


def _anchor_diagnostics(
    cutoff_records: Sequence[Mapping[str, Any]],
    effective_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cutoff_by_cluster = {str(row["cluster_id"]): row for row in cutoff_records}
    rows = []
    for effective in effective_records:
        if _outcome(effective) != "success":
            continue
        cutoff = cutoff_by_cluster[str(effective["cluster_id"])]
        observed = float(_observed_allocated(effective))
        cutoff_anchor = float(cutoff["memory"]["analytic_reference_bytes"])
        effective_anchor = float(effective["memory"]["analytic_reference_bytes"])
        rows.append(
            {
                "source_id": _source_id(effective),
                "origin": (effective.get("recalibration") or {}).get("origin"),
                "training_mode": (effective.get("selector") or {}).get(
                    "training_mode"
                ),
                "effective_fraction": effective["effective_sequence"][
                    "fraction_of_cutoff"
                ],
                "cutoff_ape": abs(cutoff_anchor / observed - 1.0),
                "effective_ape": abs(effective_anchor / observed - 1.0),
            }
        )
    return {
        "success_configurations": len(rows),
        "cutoff_anchor_source_equal_mape": _source_equal_mean(rows, "cutoff_ape"),
        "effective_anchor_source_equal_mape": _source_equal_mean(
            rows, "effective_ape"
        ),
        "by_training_mode": {
            mode: {
                "rows": len(selected),
                "cutoff_anchor_mape": statistics.fmean(
                    float(row["cutoff_ape"]) for row in selected
                ),
                "effective_anchor_mape": statistics.fmean(
                    float(row["effective_ape"]) for row in selected
                ),
            }
            for mode in ("lora", "full")
            if (selected := [row for row in rows if row["training_mode"] == mode])
        },
    }


def _input_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_candidate(
    path: Path,
    *,
    selected_variant: str | None,
    bundle: Mapping[str, Any] | None,
    source_count: int,
    report_sha256: str,
    stage_two_required: bool,
    stage_two_triggers: Mapping[str, bool],
) -> dict[str, Any]:
    critical_key = json.dumps(["lora", 2, False, 2, False], separators=(",", ":"))
    candidate: dict[str, Any] = {
        "schema": CANDIDATE_SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            (
                "shadow_candidate_requires_stage_two_calibration"
                if stage_two_required
                else "shadow_candidate_requires_prospective_holdout"
            )
            if selected_variant and bundle
            else "no_candidate_selected"
        ),
        "publishable": False,
        "production_override_allowed": False,
        "selected_variant": selected_variant,
        "scope": {
            "gpu_family": "H800",
            "training_mode": "lora",
            "zero_stage": 2,
            "gradient_checkpointing": False,
            "gpu_count": 2,
            "packing": False,
            "mechanism_key": critical_key,
            "outside_scope_policy": "keep_current_frozen_model",
        },
        "effective_sequence_contract": (
            "round_up(min(cutoff_len, profile_max), 8); packing uses cutoff_len"
        ),
        "missing_or_mismatched_profile_policy": "unavailable_fail_closed",
        "independent_critical_sources": source_count,
        "formal_95_percent_threshold_requires_sources": 19,
        "stage_two_calibration_required": stage_two_required,
        "stage_two_triggers": dict(stage_two_triggers),
        "stage_two_plan": {
            "new_independent_sources": 20,
            "jobs_per_source": 3,
            "jobs": 60,
            "target_total_independent_sources": 39,
        },
        "prospective_holdout_required": True,
        "source_recalibration_report_sha256": report_sha256,
        "model": dict(bundle or {}),
    }
    candidate["candidate_sha256"] = sha256_json(candidate)
    write_json(path, candidate)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-queue", type=Path, default=DEFAULT_NEW_QUEUE)
    parser.add_argument(
        "--old-observations", type=Path, default=DEFAULT_OLD_OBSERVATIONS
    )
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

    queue = _read_jsonl(args.new_queue)
    if len(queue) != 77 or len({row.get("job_id") for row in queue}) != 77:
        raise ValueError("new campaign queue must contain exactly 77 unique jobs")
    print("exporting exact 77-job campaign attempts", flush=True)
    new_observations = _export_exact_jobs(queue)
    if len(new_observations) != 77:
        raise ValueError("exact exporter did not return 77 observations")
    for row in new_observations:
        errors = validate_canonical_observation(row)
        if errors:
            raise ValueError(f"canonical observation validation failed: {errors}")
    old_observations = _read_jsonl(args.old_observations)
    for row in old_observations:
        errors = validate_canonical_observation(row)
        if errors:
            raise ValueError(f"old canonical observation validation failed: {errors}")

    inventory = read_json(args.inventory)
    model_by_id, fixed_lora = _inventory_models(inventory)
    hardware = read_json(args.hardware)
    padding_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    raw_max_cache: dict[str, int] = {}
    new_cutoff: list[dict[str, Any]] = []
    new_effective: list[dict[str, Any]] = []
    old_cutoff: list[dict[str, Any]] = []
    old_effective: list[dict[str, Any]] = []
    for origin, observations, cutoff_rows, effective_rows in (
        ("new_77", new_observations, new_cutoff, new_effective),
        ("old_28", old_observations, old_cutoff, old_effective),
    ):
        for observation in observations:
            cutoff, effective = _build_record_pair(
                observation,
                model_by_id=model_by_id,
                fixed_lora=fixed_lora,
                hardware=hardware,
                padding_cache=padding_cache,
                raw_max_cache=raw_max_cache,
                origin=origin,
            )
            cutoff_rows.append(cutoff)
            effective_rows.append(effective)

    new_sources = {_source_id(row) for row in new_effective}
    old_sources = {_source_id(row) for row in old_effective}
    if len(new_sources) != 15 or len(old_sources) != 4 or new_sources & old_sources:
        raise ValueError(
            "source-disjoint contract failed: expected 15 new + 4 old sources"
        )
    all_cutoff = [*old_cutoff, *new_cutoff]
    all_effective = [*old_effective, *new_effective]
    collapsed_cutoff, collapsed_effective, collapse = _collapse_records(
        all_cutoff, all_effective
    )
    counts = Counter(_outcome(row) for row in all_effective)
    collapsed_counts = Counter(_outcome(row) for row in collapsed_effective)
    print(
        f"raw observations={len(all_effective)} outcomes={dict(counts)}; "
        f"unique configs={len(collapsed_effective)} outcomes={dict(collapsed_counts)}; "
        f"sources={len(new_sources | old_sources)}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_observations.jsonl"
    validated_path = args.output_dir / "validated_observations.jsonl"
    write_jsonl(raw_path, new_observations)
    write_jsonl(
        validated_path,
        [_observation_binding(row) for row in new_observations],
    )

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

    frozen = read_json(args.frozen_baseline)
    m0 = _evaluate_frozen_m0(collapsed_cutoff, frozen)
    common_support = _common_support_metrics(m0, nested)
    selection = _select_variant(nested)
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
    critical_key = json.dumps(["lora", 2, False, 2, False], separators=(",", ":"))
    critical_expansion = (
        (((full_bundle or {}).get("reservation_expansion") or {}).get("entries") or {}).get(
            critical_key
        )
        or {}
    )
    selected_metrics = (
        (nested.get(selected_variant) or {}).get("metrics") or {}
        if selected_variant
        else {}
    )
    selected_details = (
        (nested.get(selected_variant) or {}).get("details") or []
        if selected_variant
        else []
    )
    critical_scope_metrics = _evaluate(
        [row for row in selected_details if row.get("critical_lora") is True]
    )
    new_critical_scope_metrics = _evaluate(
        [
            row
            for row in selected_details
            if row.get("critical_lora") is True and row.get("origin") == "new_77"
        ]
    )
    ratio_stability = selection.get("ratio_stability") or {}
    expansion_upper = critical_expansion.get("expansion_upper")
    empirical_max_expansion = critical_expansion.get("empirical_max_expansion")
    stage_two_triggers = {
        "admission_recall_below_0p90": (
            float(critical_scope_metrics.get("admission_recall") or 0.0) < 0.90
        ),
        "critical_reserved_center_source_equal_mape_above_0p20": (
            float(
                critical_scope_metrics.get("reserved_center_source_equal_mape")
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
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            (
                "shadow_candidate_requires_stage_two_calibration"
                if stage_two_required
                else "shadow_candidate_requires_prospective_holdout"
            )
            if selected_variant
            else "recalibration_gate_failed_no_candidate"
        ),
        "publishable": False,
        "production_model_mutated": False,
        "evaluation_kind": (
            "source_disjoint_nested_calibration_diagnostic; not prospective acceptance"
        ),
        "inputs": {
            "new_queue": _input_binding(args.new_queue),
            "old_observations": _input_binding(args.old_observations),
            "inventory": _input_binding(args.inventory),
            "hardware": _input_binding(args.hardware),
            "frozen_baseline": _input_binding(args.frozen_baseline),
        },
        "data_audit": {
            "new_observations": len(new_observations),
            "old_observations": len(old_observations),
            "new_outcomes": dict(Counter((row.get("outcome") or {}).get("class") for row in new_observations)),
            "new_independent_sources": len(new_sources),
            "old_independent_sources": len(old_sources),
            "combined_independent_sources": len(new_sources | old_sources),
            "source_overlap": sorted(new_sources & old_sources),
            "raw_profile_runtime_binding_complete": True,
            "canonical_validation_complete": True,
            "collapse": collapse,
            "collapsed_outcomes": dict(collapsed_counts),
        },
        "contracts": {
            "effective_sequence": "round_up(min(cutoff_len, profile_max), 8)",
            "raw_profile_max_above_cutoff": (
                "recorded for diagnostics but clipped before physical and ratio features"
            ),
            "center_weighting": "equal total weight per independent split_unit_id",
            "repeat_policy": "exact configurations collapsed; repeats do not add sources",
            "tail_policy": "inner OOF scores collapsed to worst per independent source",
            "outer_95_percent_limit": (
                "18 training sources force empirical-max diagnostic fallback; maximum "
                "finite-sample guarantee is 18/19=94.74%"
            ),
            "full_fit_95_percent_rule": "19 sources -> rank 19 empirical maximum",
        },
        "anchor_diagnostics": _anchor_diagnostics(
            collapsed_cutoff, collapsed_effective
        ),
        "M0_frozen_control": m0,
        "nested_M1_M3": nested,
        "common_support_comparison": common_support,
        "selection": selection,
        "selected_scope_diagnostics": {
            "all_configurations": selected_metrics,
            "critical_lora_2gpu_zero2_gc_off_nonpacking": critical_scope_metrics,
            "new_15_source_critical_lora": new_critical_scope_metrics,
        },
        "selected_full_fit": {
            "variant": selected_variant,
            "bundle": full_bundle,
            "critical_expansion": critical_expansion,
        },
        "release_gate": {
            "calibration_checkpoint_reached": bool(selected_variant),
            "stage_two_calibration_required": stage_two_required,
            "stage_two_triggers": stage_two_triggers,
            "recall_trigger_metric": (
                "nested_leave-source-out critical LoRA 2-GPU/ZeRO-2/GC-off/"
                "non-packing admission recall"
            ),
            "stage_two_plan": {
                "new_independent_sources": 20,
                "jobs_per_source": 3,
                "jobs": 60,
                "target_total_independent_sources": 39,
            },
            "formal_prospective_acceptance_passed": False,
            "production_replacement_allowed": False,
            "required_next_action": (
                (
                    "add the planned 20-source/60-job stage-two calibration, refit and "
                    "freeze a new shadow candidate before any prospective holdout"
                )
                if stage_two_required
                else (
                    "freeze the selected shadow candidate, then execute a new "
                    "source-disjoint prospective holdout"
                )
            ),
        },
    }
    report["report_sha256"] = sha256_json(report)
    report_path = args.output_dir / "nested_ablation_results.json"
    write_json(report_path, report)
    candidate_path = args.output_dir / "candidate_model.json"
    candidate = _write_candidate(
        candidate_path,
        selected_variant=selected_variant,
        bundle=full_bundle,
        source_count=len(new_sources | old_sources),
        report_sha256=report["report_sha256"],
        stage_two_required=stage_two_required,
        stage_two_triggers=stage_two_triggers,
    )
    manifest = {
        "schema": "sft_h800_lora_recalibration_output_manifest/v1",
        "generated_at_utc": report["generated_at_utc"],
        "files": {
            path.name: _input_binding(path)
            for path in (raw_path, validated_path, report_path, candidate_path)
        },
        "selected_variant": selected_variant,
        "candidate_sha256": candidate["candidate_sha256"],
        "publishable": False,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(args.output_dir / "output_manifest.json", manifest)
    print(
        f"wrote {report_path}; selected_variant={selected_variant}; "
        f"candidate={candidate_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
