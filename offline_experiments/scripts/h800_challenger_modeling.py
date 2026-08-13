#!/usr/bin/env python3
"""Evaluate alternative H800 memory and throughput models without GPU runs.

The command is deliberately an offline challenger study:

* no experiment, queue, approval, or production profile is created;
* model families and hyperparameters are selected only by native
  ``calibration`` leave-scenario-out cross-validation;
* the selected models are frozen before the native ``holdout`` partition is
  evaluated;
* memory is modeled as a regularized multiplicative residual over an analytic
  physical reference, with mechanism interactions and a censored OOM safety
  tail;
* throughput is modeled as a pairwise ranking residual over the existing
  physical throughput score, directly optimizing configuration ordering.

The holdout used here was already inspected by the earlier stage-1 memory
report.  It is therefore a useful same-row comparison, not a fresh publication
acceptance set.  The output always remains non-publishable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any

import numpy as np

from audit_h800_calibration_readiness import (
    PACKING_EVIDENCE_CLASS,
    SUPPORTED_MBS,
    _selector as readiness_selector,
)
from common import ROOT, read_json, sha256_file, sha256_json
from h800_native_memory_calibration import (
    build_native_record,
    native_admission_reason,
)
from h800_theory_basis import build_record as build_theory_record
from h800_theory_calibration import (
    _candidate_key,
    _observed_allocated,
    _observed_reserved,
    _observed_step_seconds,
    _oom_lower,
    _predict_step_seconds,
    _safe_limit,
    _work_per_step,
    scenario_id,
    scenario_material,
)


SCHEMA = "sft_h800_challenger_modeling/v1"
IMPLEMENTATION_VERSION = (
    "sft_h800_challenger_modeling_impl/"
    "2026-07-27.physical-ridge-and-pairwise-ranking"
)
OBSERVATION_SCHEMA = "sft_efficiency_observation/v2"
CALIBRATION_ROLES = {"calibration", "holdout"}
GIB = float(1024**3)
MEMORY_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
MEMORY_HISTORICAL_WEIGHT_GRID = (0.0, 0.1, 0.25, 0.5, 1.0)
MEMORY_FEATURE_GRIDS = ("basic", "physical_shares", "full")
THROUGHPUT_ALPHA_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
THROUGHPUT_HISTORICAL_WEIGHT_GRID = (0.0, 0.1, 0.25, 0.5, 1.0)
THROUGHPUT_FEATURE_GRIDS = ("basic", "physical")
CONFORMAL_COVERAGE = 0.95


MEMORY_BASIC_FEATURES = (
    "is_lora",
    "gradient_checkpointing",
    "zero2",
    "zero3",
    "lora_x_gc",
    "lora_x_zero2",
    "lora_x_zero3",
    "gc_x_zero2",
    "gc_x_zero3",
    "lora_x_zero3_x_non_gc",
    "log2_gpu_count",
    "log2_mbs",
    "log2_cutoff_over_512",
    "log2_parameters_over_1p7b",
    "lora_x_log2_parameters",
    "gc_x_log2_cutoff",
    "zero2_x_log2_gpu_count",
    "zero3_x_log2_gpu_count",
    "log2_mbs_x_log2_cutoff",
)
MEMORY_COMPONENT_KEYS = (
    "parameters_bytes",
    "gradients_bytes",
    "optimizer_bytes",
    "saved_activations_bytes",
    "recompute_workspace_bytes",
    "attention_workspace_bytes",
    "logits_workspace_bytes",
    "zero_collective_workspace_bytes",
    "stage3_live_parameters_bytes",
)
MEMORY_SHARE_FEATURES = tuple(f"share_{key}" for key in MEMORY_COMPONENT_KEYS)
MEMORY_LOG_COMPONENT_FEATURES = tuple(
    f"log1p_gib_{key}" for key in MEMORY_COMPONENT_KEYS
)

THROUGHPUT_BASIC_FEATURES = (
    "gradient_checkpointing",
    "zero2",
    "zero3",
    "lora_x_gc",
    "lora_x_zero2",
    "lora_x_zero3",
    "gc_x_zero2",
    "gc_x_zero3",
    "lora_x_zero3_x_non_gc",
    "log2_gpu_count",
    "log2_mbs",
    "log2_gradient_accumulation",
    "gc_x_log2_cutoff",
    "gc_x_log2_parameters",
    "zero2_x_log2_gpu_count",
    "zero3_x_log2_gpu_count",
    "zero2_x_log2_parameters",
    "zero3_x_log2_parameters",
    "log2_mbs_x_log2_cutoff",
    "log2_mbs_x_log2_parameters",
    "log2_gpu_count_x_log2_parameters",
)
THROUGHPUT_PHYSICAL_FEATURES = (
    "log1p_ideal_compute_seconds",
    "log1p_ideal_kernel_hbm_seconds",
    "log1p_ideal_optimizer_hbm_seconds",
    "log1p_ideal_collective_seconds",
    "log1p_communication_payload_gib",
    "log1p_kernel_traffic_gib",
    "log1p_optimizer_traffic_gib",
)


def _verify_bound_report(
    report: Mapping[str, Any], *, expected_schema: str, name: str
) -> None:
    if report.get("schema") != expected_schema:
        raise ValueError(
            f"{name} schema mismatch: {report.get('schema')!r} != "
            f"{expected_schema!r}"
        )
    unsigned = dict(report)
    digest = unsigned.pop("report_sha256", None)
    if not isinstance(digest, str) or digest != sha256_json(unsigned):
        raise ValueError(f"{name} report SHA-256 mismatch")


def _read_observations(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("schema") != OBSERVATION_SCHEMA:
                raise ValueError(
                    f"Observation line {line_number} is not {OBSERVATION_SCHEMA}"
                )
            observation_id = row.get("observation_id")
            if not isinstance(observation_id, str) or not observation_id:
                raise ValueError(f"Observation line {line_number} has no id")
            if observation_id in seen:
                raise ValueError(f"Duplicate observation id {observation_id}")
            seen.add(observation_id)
            rows.append(row)
    return rows


def _job(row: Mapping[str, Any]) -> dict[str, Any]:
    configuration = row.get("configuration")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    job = configuration.get("job")
    return dict(job) if isinstance(job, Mapping) else {}


def _partition(row: Mapping[str, Any]) -> dict[str, Any]:
    configuration = row.get("configuration")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    partition = configuration.get("calibration_partition")
    return dict(partition) if isinstance(partition, Mapping) else {}


def _outcome(record: Mapping[str, Any]) -> str:
    value = record.get("outcome")
    if isinstance(value, Mapping):
        value = value.get("class")
    return str(value or "").lower()


def _observation_id(record: Mapping[str, Any]) -> str:
    value = record.get("observation_id")
    if not isinstance(value, str) or not value:
        raise ValueError("Record has no observation id")
    return value


def _is_legacy(record: Mapping[str, Any]) -> bool:
    return str(record.get("evidence_tier") or "").startswith("legacy")


def _mean(values: Sequence[float]) -> float | None:
    usable = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(usable) if usable else None


def _percentile(values: Sequence[float], probability: float) -> float | None:
    usable = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not usable:
        return None
    position = min(1.0, max(0.0, probability)) * (len(usable) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return usable[lower]
    fraction = position - lower
    return usable[lower] * (1 - fraction) + usable[upper] * fraction


def _summary(values: Sequence[float]) -> dict[str, Any]:
    usable = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(usable),
        "mean": _mean(usable),
        "median": statistics.median(usable) if usable else None,
        "p90": _percentile(usable, 0.90),
        "max": max(usable) if usable else None,
    }


def throughput_admission_reason(row: Mapping[str, Any]) -> str:
    """Return one mutually exclusive native throughput admission decision."""

    hardware = row.get("hardware")
    hardware = hardware if isinstance(hardware, Mapping) else {}
    if hardware.get("gpu_family") != "H800":
        return "not_exact_h800"
    fingerprint = row.get("fingerprint")
    fingerprint = fingerprint if isinstance(fingerprint, Mapping) else {}
    if fingerprint.get("quality") != "complete":
        return "fingerprint_not_complete"
    if fingerprint.get("calibration_evidence_eligible") is not True:
        return "fingerprint_not_calibration_eligible"
    outcome = row.get("outcome")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    if outcome.get("class") != "success":
        return "outcome_not_success"
    if outcome.get("usable_for_throughput_calibration") is not True:
        return "not_usable_for_throughput"
    partition = _partition(row)
    if str(partition.get("role") or "").lower() not in CALIBRATION_ROLES:
        return "partition_not_calibration_or_holdout"
    if not isinstance(partition.get("split_unit_id"), str) or not partition.get(
        "split_unit_id"
    ):
        return "partition_split_unit_missing"
    job = _job(row)
    if job.get("packing") is True or job.get(
        "calibration_evidence_class"
    ) == PACKING_EVIDENCE_CLASS:
        return "packing_effect_evidence_excluded"
    try:
        mbs = int(job.get("mbs"))
    except (TypeError, ValueError):
        return "mbs_missing_or_invalid"
    if mbs not in SUPPORTED_MBS:
        return "mbs_outside_supported_domain"
    _key, selector = readiness_selector(dict(row))
    if (
        not isinstance(selector.get("runtime_fingerprint"), str)
        or not selector.get("runtime_fingerprint")
        or selector.get("training_mode") not in {"full", "lora"}
        or selector.get("zero_stage") not in {0, 1, 2, 3}
        or not isinstance(selector.get("gradient_checkpointing"), bool)
        or selector.get("dtype") == "unknown"
        or "unknown" in str(selector.get("kernel_path"))
    ):
        return "mechanism_selector_incomplete"
    return "admitted"


def _inventory_models(
    inventory: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    models = inventory.get("models")
    if not isinstance(models, list):
        raise ValueError("Model inventory has no models")
    model_by_id = {
        str(model.get("id")): dict(model)
        for model in models
        if isinstance(model, Mapping)
    }
    if len(model_by_id) != len(models):
        raise ValueError("Model inventory ids must be present and unique")
    fixed_lora = inventory.get("fixed_lora")
    if not isinstance(fixed_lora, Mapping):
        raise ValueError("Model inventory has no fixed_lora contract")
    return model_by_id, dict(fixed_lora)


def _build_native_throughput_record(
    row: dict[str, Any],
    *,
    model_by_id: Mapping[str, dict[str, Any]],
    fixed_lora: dict[str, Any],
    hardware: dict[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    if throughput_admission_reason(row) != "admitted":
        raise ValueError("Cannot build an excluded throughput observation")
    job = _job(row)
    fingerprint = row.get("fingerprint") or {}
    runtime_fingerprint = str(
        fingerprint.get("runtime_mechanism_fingerprint_sha256") or ""
    )
    recovery = {
        "runtime": {
            "runtime_cohort_id": runtime_fingerprint,
            "runtime_cohort_material": {
                "runtime_mechanism_fingerprint_sha256": runtime_fingerprint
            },
        },
        "measurement_eligibility": {"throughput_primary": True},
        "source_observation_sha256": sha256_json(row),
        "recovery_id": None,
        "evidence_tier": "native_v2",
    }
    record = build_theory_record(
        row,
        recovery,
        model_by_id[str(job.get("model_id"))],
        fixed_lora,
        hardware,
        runtime_root,
    )
    _key, selector = readiness_selector(row)
    record["selector"] = {
        "runtime_cohort_id": runtime_fingerprint,
        "dtype": selector["dtype"],
        "kernel_path": selector["kernel_path"],
        "training_mode": selector["training_mode"],
        "zero_stage": selector["zero_stage"],
        "gradient_checkpointing": selector["gradient_checkpointing"],
        "packing": False,
    }
    partition = _partition(row)
    record["calibration_partition"] = {
        "policy": partition.get("policy"),
        "role": str(partition.get("role")).lower(),
        "split_unit_id": partition.get("split_unit_id"),
    }
    record["confidence"] = "native_v2"
    return record


def _memory_feature_names(kind: str) -> tuple[str, ...]:
    if kind == "basic":
        return MEMORY_BASIC_FEATURES
    if kind == "physical_shares":
        return MEMORY_BASIC_FEATURES + MEMORY_SHARE_FEATURES
    if kind == "full":
        return (
            MEMORY_BASIC_FEATURES
            + MEMORY_SHARE_FEATURES
            + MEMORY_LOG_COMPONENT_FEATURES
        )
    raise ValueError(f"Unknown memory feature set {kind!r}")


def _memory_features(record: Mapping[str, Any], kind: str) -> np.ndarray:
    selector = record.get("selector") or {}
    scenario = record.get("scenario") or {}
    memory = record.get("memory") or {}
    components = memory.get("components") or {}
    model_basis = record.get("model_basis") or {}
    reference = float(memory["analytic_reference_bytes"])
    is_lora = float(selector.get("training_mode") == "lora")
    gc = float(bool(selector.get("gradient_checkpointing")))
    zero = int(selector.get("zero_stage") or 0)
    zero2 = float(zero == 2)
    zero3 = float(zero == 3)
    log_gpu = math.log2(float(scenario["gpu_count"]))
    log_mbs = math.log2(float(scenario["physical_mbs"]))
    log_cutoff = math.log2(float(scenario["cutoff_len"]) / 512.0)
    log_parameters = math.log2(
        float(model_basis["base_parameters"]) / 2_031_739_904.0
    )
    basic = [
        is_lora,
        gc,
        zero2,
        zero3,
        is_lora * gc,
        is_lora * zero2,
        is_lora * zero3,
        gc * zero2,
        gc * zero3,
        is_lora * zero3 * (1.0 - gc),
        log_gpu,
        log_mbs,
        log_cutoff,
        log_parameters,
        is_lora * log_parameters,
        gc * log_cutoff,
        zero2 * log_gpu,
        zero3 * log_gpu,
        log_mbs * log_cutoff,
    ]
    if kind == "basic":
        return np.asarray(basic, dtype=float)
    shares = [float(components[key]) / reference for key in MEMORY_COMPONENT_KEYS]
    if kind == "physical_shares":
        return np.asarray([*basic, *shares], dtype=float)
    if kind == "full":
        logged = [
            math.log1p(float(components[key]) / GIB)
            for key in MEMORY_COMPONENT_KEYS
        ]
        return np.asarray([*basic, *shares, *logged], dtype=float)
    raise ValueError(f"Unknown memory feature set {kind!r}")


def _memory_label(record: Mapping[str, Any], label: str) -> float | None:
    if label == "reserved":
        return _observed_reserved(record)
    if label == "allocated":
        return _observed_allocated(record)
    raise ValueError(f"Unknown memory label {label!r}")


def _fit_memory_ridge(
    records: Sequence[Mapping[str, Any]],
    *,
    feature_set: str,
    alpha: float,
    historical_weight: float,
    label: str = "reserved",
) -> dict[str, Any]:
    successes = [
        record
        for record in records
        if _outcome(record) == "success"
        and _memory_label(record, label) is not None
        and (historical_weight > 0 or not _is_legacy(record))
    ]
    if not successes:
        raise ValueError("No success rows for memory ridge")
    features = np.vstack(
        [_memory_features(record, feature_set) for record in successes]
    )
    targets = np.asarray(
        [
            math.log(
                float(_memory_label(record, label))
                / float(record["memory"]["analytic_reference_bytes"])
            )
            for record in successes
        ],
        dtype=float,
    )
    scenario_counts = Counter(scenario_id(record) for record in successes)
    weights = np.asarray(
        [
            (historical_weight if _is_legacy(record) else 1.0)
            / scenario_counts[scenario_id(record)]
            for record in successes
        ],
        dtype=float,
    )
    means = np.average(features, axis=0, weights=weights)
    scales = np.sqrt(
        np.average((features - means) ** 2, axis=0, weights=weights)
    )
    scales[scales < 1e-9] = 1.0
    standardized = (features - means) / scales
    design = np.column_stack((np.ones(len(standardized)), standardized))
    penalty = np.diag([0.0, *([float(alpha)] * standardized.shape[1])])
    normal = design.T @ (weights[:, None] * design) + penalty
    target = design.T @ (weights * targets)
    coefficients = np.linalg.pinv(normal) @ target
    return {
        "available": True,
        "model_family": "analytic_reference_log_residual_ridge",
        "label": label,
        "feature_set": feature_set,
        "feature_names": list(_memory_feature_names(feature_set)),
        "alpha": float(alpha),
        "historical_scenario_weight": float(historical_weight),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "fit_success_rows": len(successes),
        "fit_scenarios": len(scenario_counts),
        "fit_legacy_rows": sum(_is_legacy(record) for record in successes),
        "fit_native_rows": sum(not _is_legacy(record) for record in successes),
    }


def _predict_memory_center(
    record: Mapping[str, Any], model: Mapping[str, Any]
) -> float:
    features = _memory_features(record, str(model["feature_set"]))
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    residual = float(model["intercept"]) + float(
        ((features - means) / scales) @ coefficients
    )
    prediction = float(record["memory"]["analytic_reference_bytes"]) * math.exp(
        residual
    )
    if not math.isfinite(prediction) or prediction <= 0:
        raise ValueError("Memory ridge produced a non-positive prediction")
    return prediction


def _memory_selector_key(record: Mapping[str, Any]) -> tuple[str, int, bool, bool]:
    selector = record.get("selector") or {}
    return (
        str(selector.get("training_mode") or "unknown"),
        int(selector.get("zero_stage") or 0),
        bool(selector.get("gradient_checkpointing")),
        bool(selector.get("packing")),
    )


def _memory_mode_gc_key(record: Mapping[str, Any]) -> tuple[str, bool]:
    selector = record.get("selector") or {}
    return (
        str(selector.get("training_mode") or "unknown"),
        bool(selector.get("gradient_checkpointing")),
    )


def _key_token(key: Sequence[Any]) -> str:
    return json.dumps(list(key), ensure_ascii=False, separators=(",", ":"))


def _finite_upper_quantile(
    values: Sequence[float], *, coverage: float = CONFORMAL_COVERAGE
) -> dict[str, Any]:
    usable = sorted(float(value) for value in values if math.isfinite(float(value)))
    rank = math.ceil((len(usable) + 1) * coverage)
    identifiable = bool(usable) and rank <= len(usable)
    return {
        "sample_count": len(usable),
        "coverage": coverage,
        "rank": rank,
        "available": identifiable,
        "log_residual_upper": usable[rank - 1] if identifiable else None,
    }


def _fit_memory_tail(
    records: Sequence[Mapping[str, Any]],
    *,
    feature_set: str,
    alpha: float,
    historical_weight: float,
) -> dict[str, Any]:
    success_selector: dict[str, list[float]] = defaultdict(list)
    success_mode_gc: dict[str, list[float]] = defaultdict(list)
    success_pooled: list[float] = []
    oom_selector: dict[str, list[float]] = defaultdict(list)
    scenarios = sorted({scenario_id(record) for record in records})
    for held_out in scenarios:
        train = [
            record for record in records if scenario_id(record) != held_out
        ]
        test = [
            record for record in records if scenario_id(record) == held_out
        ]
        model = _fit_memory_ridge(
            train,
            feature_set=feature_set,
            alpha=alpha,
            historical_weight=historical_weight,
        )
        for record in test:
            center = _predict_memory_center(record, model)
            selector = _key_token(_memory_selector_key(record))
            mode_gc = _key_token(_memory_mode_gc_key(record))
            if _outcome(record) == "success":
                observed = _observed_reserved(record)
                if observed is None:
                    continue
                residual = math.log(observed / center)
                success_selector[selector].append(residual)
                success_mode_gc[mode_gc].append(residual)
                success_pooled.append(residual)
            elif _outcome(record) == "oom":
                lower = _oom_lower(record)
                safe = _safe_limit(record)
                floors = [
                    value
                    for value in (
                        lower,
                        (safe + 1.0) if safe is not None else None,
                    )
                    if value is not None
                ]
                if floors:
                    oom_selector[selector].append(
                        math.log(max(floors) / center)
                    )
    return {
        "available": True,
        "residual_source": "inner_leave_scenario_out",
        "success_log_residual_conformal": {
            "selector": {
                key: _finite_upper_quantile(values)
                for key, values in sorted(success_selector.items())
            },
            "mode_gc": {
                key: _finite_upper_quantile(values)
                for key, values in sorted(success_mode_gc.items())
            },
            "pooled": _finite_upper_quantile(success_pooled),
        },
        "oom_log_residual_lower_by_exact_selector": {
            key: {
                "sample_count": len(values),
                "log_residual_lower": max(values),
            }
            for key, values in sorted(oom_selector.items())
        },
        "tail_hierarchy": ["selector", "mode_x_gc", "pooled"],
        "oom_hierarchy": "exact_selector_only",
        "formula": (
            "center * exp(max(success_OOF_log_q95, "
            "exact_selector_OOM_log_guard, 0))"
        ),
    }


def _predict_memory_upper(
    record: Mapping[str, Any],
    center_model: Mapping[str, Any],
    tail: Mapping[str, Any],
) -> dict[str, Any]:
    center = _predict_memory_center(record, center_model)
    success = tail.get("success_log_residual_conformal") or {}
    selector_token = _key_token(_memory_selector_key(record))
    mode_gc_token = _key_token(_memory_mode_gc_key(record))
    selector_bucket = (success.get("selector") or {}).get(selector_token)
    mode_bucket = (success.get("mode_gc") or {}).get(mode_gc_token)
    pooled_bucket = success.get("pooled") or {}
    if isinstance(selector_bucket, Mapping) and selector_bucket.get(
        "available"
    ) is True:
        bucket = selector_bucket
        source = "selector"
    elif isinstance(mode_bucket, Mapping) and mode_bucket.get("available") is True:
        bucket = mode_bucket
        source = "mode_x_gc"
    else:
        bucket = pooled_bucket
        source = "pooled"
    if not isinstance(bucket, Mapping) or bucket.get("available") is not True:
        return {
            "available": False,
            "reserved_center_bytes": center,
            "issues": ["finite_sample_success_q95_unavailable"],
        }
    success_q = float(bucket["log_residual_upper"])
    oom_bucket = (
        tail.get("oom_log_residual_lower_by_exact_selector") or {}
    ).get(selector_token)
    oom_guard = (
        float(oom_bucket["log_residual_lower"])
        if isinstance(oom_bucket, Mapping)
        and oom_bucket.get("log_residual_lower") is not None
        else 0.0
    )
    log_guard = max(0.0, success_q, oom_guard)
    return {
        "available": True,
        "reserved_center_bytes": center,
        "success_tail_source": source,
        "success_log_residual_upper": success_q,
        "oom_exact_selector_log_guard": oom_guard,
        "operational_p95_reserved_bytes": center * math.exp(log_guard),
        "issues": [],
    }


def _memory_evaluation(
    records: Sequence[Mapping[str, Any]],
    center_model: Mapping[str, Any],
    tail: Mapping[str, Any],
    *,
    include_details: bool,
) -> dict[str, Any]:
    absolute_percentage_errors: list[float] = []
    signed_percentage_errors: list[float] = []
    absolute_errors_gib: list[float] = []
    success_rows = success_covered = 0
    oom_rows = false_safe = unavailable = 0
    safe_successes = admitted_safe_successes = 0
    details: list[dict[str, Any]] = []
    for record in records:
        outcome = _outcome(record)
        if outcome not in {"success", "oom"}:
            continue
        prediction = _predict_memory_upper(record, center_model, tail)
        detail: dict[str, Any] = {
            "observation_id": _observation_id(record),
            "outcome": outcome,
            "scenario_id": scenario_id(record),
            "prediction_available": prediction.get("available") is True,
        }
        if prediction.get("available") is not True:
            unavailable += 1
            detail["issues"] = prediction.get("issues") or []
            if include_details:
                details.append(detail)
            continue
        center = float(prediction["reserved_center_bytes"])
        upper = float(prediction["operational_p95_reserved_bytes"])
        safe_limit = _safe_limit(record)
        detail.update(
            {
                "reserved_center_bytes": center,
                "operational_p95_reserved_bytes": upper,
                "safe_limit_bytes": safe_limit,
                "success_tail_source": prediction.get("success_tail_source"),
            }
        )
        if outcome == "success":
            observed = _observed_reserved(record)
            if observed is None or safe_limit is None:
                unavailable += 1
                detail["issues"] = ["success_reserved_or_safe_limit_missing"]
            else:
                error = center - observed
                absolute_percentage_errors.append(abs(error) / observed)
                signed_percentage_errors.append(error / observed)
                absolute_errors_gib.append(abs(error) / GIB)
                success_rows += 1
                covered = observed <= upper
                success_covered += int(covered)
                actual_safe = observed <= safe_limit
                if actual_safe:
                    safe_successes += 1
                    admitted_safe_successes += int(upper <= safe_limit)
                detail.update(
                    {
                        "observed_reserved_bytes": observed,
                        "center_absolute_percentage_error": abs(error)
                        / observed,
                        "covered": covered,
                        "actual_safe_success": actual_safe,
                        "predicted_admit": upper <= safe_limit,
                    }
                )
        else:
            if safe_limit is None or _oom_lower(record) is None:
                unavailable += 1
                detail["issues"] = ["oom_safe_limit_or_lower_missing"]
            else:
                oom_rows += 1
                is_false_safe = upper <= safe_limit
                false_safe += int(is_false_safe)
                detail.update(
                    {
                        "right_censor_lower_bytes": _oom_lower(record),
                        "false_safe": is_false_safe,
                    }
                )
        if include_details:
            details.append(detail)
    return {
        "reserved_center_absolute_percentage_error": _summary(
            absolute_percentage_errors
        ),
        "reserved_center_signed_percentage_error": _summary(
            signed_percentage_errors
        ),
        "reserved_center_absolute_error_gib": _summary(absolute_errors_gib),
        "success_rows": success_rows,
        "success_covered": success_covered,
        "success_p95_coverage": (
            success_covered / success_rows if success_rows else None
        ),
        "oom_rows": oom_rows,
        "false_safe_oom": false_safe,
        "false_safe_oom_rate": false_safe / oom_rows if oom_rows else None,
        "prediction_unavailable_rows": unavailable,
        "actual_safe_success_rows": safe_successes,
        "admitted_safe_success_rows": admitted_safe_successes,
        "false_reject_safe_success": (
            safe_successes - admitted_safe_successes
        ),
        "safe_success_admission_recall": (
            admitted_safe_successes / safe_successes
            if safe_successes
            else None
        ),
        "details": details if include_details else None,
    }


def _memory_center_cv(
    historical: Sequence[Mapping[str, Any]],
    native_calibration: Sequence[Mapping[str, Any]],
    *,
    feature_set: str,
    alpha: float,
    historical_weight: float,
) -> dict[str, Any]:
    augmented = [*historical, *native_calibration]
    errors: list[float] = []
    signed_errors: list[float] = []
    folds = []
    for held_out in sorted({scenario_id(record) for record in native_calibration}):
        train = [
            record
            for record in augmented
            if scenario_id(record) != held_out
        ]
        test = [
            record
            for record in native_calibration
            if scenario_id(record) == held_out and _outcome(record) == "success"
        ]
        model = _fit_memory_ridge(
            train,
            feature_set=feature_set,
            alpha=alpha,
            historical_weight=historical_weight,
        )
        fold_errors = []
        for record in test:
            observed = _observed_reserved(record)
            assert observed is not None
            predicted = _predict_memory_center(record, model)
            error = (predicted - observed) / observed
            errors.append(abs(error))
            signed_errors.append(error)
            fold_errors.append(abs(error))
        folds.append(
            {
                "held_out_scenario_id": held_out,
                "held_out_scenario": (
                    scenario_material(test[0])
                    if test
                    else scenario_material(
                        next(
                            record
                            for record in native_calibration
                            if scenario_id(record) == held_out
                        )
                    )
                ),
                "success_rows": len(test),
                "mean_absolute_percentage_error": _mean(fold_errors),
            }
        )
    return {
        "success_rows": len(errors),
        "mean_absolute_percentage_error": _mean(errors),
        "median_absolute_percentage_error": (
            statistics.median(errors) if errors else None
        ),
        "p90_absolute_percentage_error": _percentile(errors, 0.90),
        "mean_signed_percentage_error": _mean(signed_errors),
        "scenario_folds": len(folds),
        "folds": folds,
    }


def _select_memory_model(
    historical: Sequence[Mapping[str, Any]],
    native_calibration: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for feature_set in MEMORY_FEATURE_GRIDS:
        for historical_weight in MEMORY_HISTORICAL_WEIGHT_GRID:
            for alpha in MEMORY_ALPHA_GRID:
                cv = _memory_center_cv(
                    historical,
                    native_calibration,
                    feature_set=feature_set,
                    alpha=alpha,
                    historical_weight=historical_weight,
                )
                candidates.append(
                    {
                        "feature_set": feature_set,
                        "historical_scenario_weight": historical_weight,
                        "alpha": alpha,
                        "cv": {
                            key: value
                            for key, value in cv.items()
                            if key != "folds"
                        },
                    }
                )
    selected = min(
        candidates,
        key=lambda item: (
            float(item["cv"]["mean_absolute_percentage_error"]),
            float(item["cv"]["p90_absolute_percentage_error"]),
            len(_memory_feature_names(str(item["feature_set"]))),
            float(item["historical_scenario_weight"]),
            float(item["alpha"]),
        ),
    )
    return {
        "selection_policy": (
            "minimum native-calibration leave-scenario-out reserved-center "
            "mean APE; p90 APE then feature count are tie-breakers"
        ),
        "candidate_count": len(candidates),
        "selected": selected,
        "candidates": candidates,
    }


def _memory_nested_cv(
    historical: Sequence[Mapping[str, Any]],
    native_calibration: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    augmented = [*historical, *native_calibration]
    fold_reports: list[dict[str, Any]] = []
    all_details: list[dict[str, Any]] = []
    for held_out in sorted({scenario_id(record) for record in native_calibration}):
        train = [
            record
            for record in augmented
            if scenario_id(record) != held_out
        ]
        test = [
            record
            for record in native_calibration
            if scenario_id(record) == held_out
        ]
        model = _fit_memory_ridge(
            train,
            feature_set=str(selected["feature_set"]),
            alpha=float(selected["alpha"]),
            historical_weight=float(selected["historical_scenario_weight"]),
        )
        tail = _fit_memory_tail(
            train,
            feature_set=str(selected["feature_set"]),
            alpha=float(selected["alpha"]),
            historical_weight=float(selected["historical_scenario_weight"]),
        )
        evaluation = _memory_evaluation(
            test, model, tail, include_details=True
        )
        all_details.extend(evaluation["details"] or [])
        fold_reports.append(
            {
                "held_out_scenario_id": held_out,
                "held_out_scenario": scenario_material(test[0]),
                "train_rows": len(train),
                "test_rows": len(test),
                "train_observation_ids_sha256": sha256_json(
                    sorted(_observation_id(record) for record in train)
                ),
                "test_observation_ids_sha256": sha256_json(
                    sorted(_observation_id(record) for record in test)
                ),
                "metrics": {
                    key: value
                    for key, value in evaluation.items()
                    if key != "details"
                },
            }
        )
    success_details = [
        detail
        for detail in all_details
        if detail.get("outcome") == "success"
        and detail.get("center_absolute_percentage_error") is not None
    ]
    oom_details = [
        detail
        for detail in all_details
        if detail.get("outcome") == "oom"
        and detail.get("false_safe") is not None
    ]
    safe_success = [
        detail
        for detail in success_details
        if detail.get("actual_safe_success") is True
    ]
    return {
        "policy": (
            "outer leave-native-scenario-out; the same scenario is removed "
            "from historical rows; every outer tail uses inner LOSO residuals"
        ),
        "scenario_folds": len(fold_reports),
        "aggregate": {
            "reserved_center_absolute_percentage_error": _summary(
                [
                    float(detail["center_absolute_percentage_error"])
                    for detail in success_details
                ]
            ),
            "success_rows": len(success_details),
            "success_covered": sum(
                detail.get("covered") is True for detail in success_details
            ),
            "success_p95_coverage": (
                sum(detail.get("covered") is True for detail in success_details)
                / len(success_details)
                if success_details
                else None
            ),
            "oom_rows": len(oom_details),
            "false_safe_oom": sum(
                detail.get("false_safe") is True for detail in oom_details
            ),
            "false_safe_oom_rate": (
                sum(detail.get("false_safe") is True for detail in oom_details)
                / len(oom_details)
                if oom_details
                else None
            ),
            "actual_safe_success_rows": len(safe_success),
            "admitted_safe_success_rows": sum(
                detail.get("predicted_admit") is True
                for detail in safe_success
            ),
            "false_reject_safe_success": sum(
                detail.get("predicted_admit") is not True
                for detail in safe_success
            ),
            "prediction_unavailable_rows": sum(
                detail.get("prediction_available") is not True
                for detail in all_details
            ),
        },
        "folds": fold_reports,
    }


def _allocated_center_evaluation(
    records: Sequence[Mapping[str, Any]], model: Mapping[str, Any]
) -> dict[str, Any]:
    errors: list[float] = []
    signed: list[float] = []
    absolute_gib: list[float] = []
    for record in records:
        if _outcome(record) != "success":
            continue
        observed = _observed_allocated(record)
        if observed is None:
            continue
        predicted = _predict_memory_center(record, model)
        delta = predicted - observed
        errors.append(abs(delta) / observed)
        signed.append(delta / observed)
        absolute_gib.append(abs(delta) / GIB)
    return {
        "absolute_percentage_error": _summary(errors),
        "signed_percentage_error": _summary(signed),
        "absolute_error_gib": _summary(absolute_gib),
    }


def _throughput_feature_names(kind: str) -> tuple[str, ...]:
    if kind == "basic":
        return THROUGHPUT_BASIC_FEATURES
    if kind == "physical":
        return THROUGHPUT_BASIC_FEATURES + THROUGHPUT_PHYSICAL_FEATURES
    raise ValueError(f"Unknown throughput feature set {kind!r}")


def _throughput_features(
    record: Mapping[str, Any], kind: str
) -> np.ndarray:
    selector = record.get("selector") or {}
    scenario = record.get("scenario") or {}
    performance = record.get("performance") or {}
    ideal = performance.get("ideal_seconds") or {}
    communication = performance.get("communication") or {}
    traffic = performance.get("traffic_bytes_per_rank_step") or {}
    model_basis = record.get("model_basis") or {}
    is_lora = float(selector.get("training_mode") == "lora")
    gc = float(bool(selector.get("gradient_checkpointing")))
    zero = int(selector.get("zero_stage") or 0)
    zero2 = float(zero == 2)
    zero3 = float(zero == 3)
    log_gpu = math.log2(float(scenario["gpu_count"]))
    log_mbs = math.log2(float(scenario["physical_mbs"]))
    log_cutoff = math.log2(float(scenario["cutoff_len"]) / 512.0)
    log_parameters = math.log2(
        float(model_basis["base_parameters"]) / 2_031_739_904.0
    )
    log_accumulation = math.log2(
        float(performance["gradient_accumulation_steps"])
    )
    basic = [
        gc,
        zero2,
        zero3,
        is_lora * gc,
        is_lora * zero2,
        is_lora * zero3,
        gc * zero2,
        gc * zero3,
        is_lora * zero3 * (1.0 - gc),
        log_gpu,
        log_mbs,
        log_accumulation,
        gc * log_cutoff,
        gc * log_parameters,
        zero2 * log_gpu,
        zero3 * log_gpu,
        zero2 * log_parameters,
        zero3 * log_parameters,
        log_mbs * log_cutoff,
        log_mbs * log_parameters,
        log_gpu * log_parameters,
    ]
    if kind == "basic":
        return np.asarray(basic, dtype=float)
    physical = [
        math.log1p(float(ideal["compute_at_dense_peak"])),
        math.log1p(float(ideal["kernel_hbm_at_physical_peak"])),
        math.log1p(float(ideal["optimizer_hbm_at_physical_peak"])),
        math.log1p(float(ideal["collective_payload_at_link_peak"])),
        math.log1p(
            float(communication["payload_bytes_per_rank_step"]) / GIB
        ),
        math.log1p(float(traffic["kernel_total"]) / GIB),
        math.log1p(float(traffic["optimizer"]) / GIB),
    ]
    if kind == "physical":
        return np.asarray([*basic, *physical], dtype=float)
    raise ValueError(f"Unknown throughput feature set {kind!r}")


def _observed_throughput(record: Mapping[str, Any]) -> float:
    step = _observed_step_seconds(record)
    effective_tokens = _work_per_step(record, "effective_tokens")
    if step is None or effective_tokens is None:
        raise ValueError("Throughput record is missing step time or effective tokens")
    return effective_tokens / step


def _physics_throughput_score(
    record: Mapping[str, Any],
    model: Mapping[str, Any],
    priors: Mapping[str, Any],
) -> float:
    step, issues = _predict_step_seconds(record, model, priors)
    effective_tokens = _work_per_step(record, "effective_tokens")
    if step is None or effective_tokens is None:
        raise ValueError(
            "Physical throughput score unavailable: " + ", ".join(issues)
        )
    return math.log(effective_tokens / step)


def _throughput_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    physical_model: Mapping[str, Any],
    priors: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[
        str, dict[tuple[Any, ...], list[Mapping[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[scenario_id(record)][_candidate_key(record)].append(record)
    candidates: list[dict[str, Any]] = []
    for scenario, by_candidate in sorted(grouped.items()):
        for key, replicates in sorted(
            by_candidate.items(),
            key=lambda item: tuple(str(value) for value in item[0]),
        ):
            candidates.append(
                {
                    "scenario_id": scenario,
                    "scenario": scenario_material(replicates[0]),
                    "candidate_key": list(key),
                    "record": replicates[0],
                    "replicates": len(replicates),
                    "observed_log_throughput": statistics.median(
                        math.log(_observed_throughput(record))
                        for record in replicates
                    ),
                    "physics_log_throughput": statistics.median(
                        _physics_throughput_score(
                            record, physical_model, priors
                        )
                        for record in replicates
                    ),
                    "historical": all(_is_legacy(record) for record in replicates),
                    "observation_ids": sorted(
                        _observation_id(record) for record in replicates
                    ),
                }
            )
    return candidates


def _fit_pairwise_ranker(
    candidates: Sequence[Mapping[str, Any]],
    *,
    feature_set: str,
    alpha: float,
    historical_weight: float,
    use_physics_base: bool,
) -> dict[str, Any]:
    usable = [
        candidate
        for candidate in candidates
        if historical_weight > 0 or candidate.get("historical") is not True
    ]
    if not usable:
        raise ValueError("No candidates for throughput ranker")
    scenario_counts = Counter(
        str(candidate["scenario_id"]) for candidate in usable
    )
    candidate_weights = np.asarray(
        [
            (
                historical_weight
                if candidate.get("historical") is True
                else 1.0
            )
            / scenario_counts[str(candidate["scenario_id"])]
            for candidate in usable
        ],
        dtype=float,
    )
    features = np.vstack(
        [
            _throughput_features(candidate["record"], feature_set)
            for candidate in usable
        ]
    )
    means = np.average(features, axis=0, weights=candidate_weights)
    scales = np.sqrt(
        np.average((features - means) ** 2, axis=0, weights=candidate_weights)
    )
    scales[scales < 1e-9] = 1.0
    by_scenario: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(
        list
    )
    for index, candidate in enumerate(usable):
        by_scenario[str(candidate["scenario_id"])].append((index, candidate))
    dimension = features.shape[1]
    normal = np.zeros((dimension, dimension), dtype=float)
    target = np.zeros(dimension, dtype=float)
    pair_count = 0
    pair_scenarios = 0
    for scenario_candidates in by_scenario.values():
        pairs: list[tuple[np.ndarray, float]] = []
        for left_index in range(len(scenario_candidates)):
            for right_index in range(left_index + 1, len(scenario_candidates)):
                left_row, left = scenario_candidates[left_index]
                right_row, right = scenario_candidates[right_index]
                feature_delta = (
                    features[left_row] - features[right_row]
                ) / scales
                left_target = float(left["observed_log_throughput"])
                right_target = float(right["observed_log_throughput"])
                if use_physics_base:
                    left_target -= float(left["physics_log_throughput"])
                    right_target -= float(right["physics_log_throughput"])
                pairs.append((feature_delta, left_target - right_target))
        if not pairs:
            continue
        pair_scenarios += 1
        scenario_weight = (
            historical_weight
            if scenario_candidates[0][1].get("historical") is True
            else 1.0
        )
        pair_weight = scenario_weight / len(pairs)
        for feature_delta, target_delta in pairs:
            normal += pair_weight * np.outer(feature_delta, feature_delta)
            target += pair_weight * feature_delta * target_delta
            pair_count += 1
    coefficients = np.linalg.pinv(
        normal + float(alpha) * np.eye(dimension)
    ) @ target
    uncentered_scores = []
    observed = []
    for candidate, feature in zip(usable, features):
        score = (
            float(candidate["physics_log_throughput"])
            if use_physics_base
            else 0.0
        ) + float(((feature - means) / scales) @ coefficients)
        uncentered_scores.append(score)
        observed.append(float(candidate["observed_log_throughput"]))
    intercept = float(
        np.average(
            np.asarray(observed) - np.asarray(uncentered_scores),
            weights=candidate_weights,
        )
    )
    return {
        "available": True,
        "model_family": "scenario_equal_pairwise_ridge_ranker",
        "feature_set": feature_set,
        "feature_names": list(_throughput_feature_names(feature_set)),
        "alpha": float(alpha),
        "historical_scenario_weight": float(historical_weight),
        "use_physics_base": bool(use_physics_base),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "coefficients": coefficients.tolist(),
        "global_log_intercept_for_absolute_diagnostic": intercept,
        "fit_candidates": len(usable),
        "fit_scenarios": len(by_scenario),
        "pair_scenarios": pair_scenarios,
        "fit_pairs": pair_count,
        "fit_historical_candidates": sum(
            candidate.get("historical") is True for candidate in usable
        ),
        "fit_native_candidates": sum(
            candidate.get("historical") is not True for candidate in usable
        ),
    }


def _ranker_score(
    candidate: Mapping[str, Any], model: Mapping[str, Any]
) -> float:
    features = _throughput_features(
        candidate["record"], str(model["feature_set"])
    )
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    score = float(model["global_log_intercept_for_absolute_diagnostic"])
    if model.get("use_physics_base") is True:
        score += float(candidate["physics_log_throughput"])
    score += float(((features - means) / scales) @ coefficients)
    return score


def _baseline_ranker_model(feature_set: str = "basic") -> dict[str, Any]:
    dimension = len(_throughput_feature_names(feature_set))
    return {
        "available": True,
        "model_family": "frozen_historical_physics_baseline",
        "feature_set": feature_set,
        "feature_names": list(_throughput_feature_names(feature_set)),
        "alpha": None,
        "historical_scenario_weight": None,
        "use_physics_base": True,
        "feature_means": [0.0] * dimension,
        "feature_scales": [1.0] * dimension,
        "coefficients": [0.0] * dimension,
        "global_log_intercept_for_absolute_diagnostic": 0.0,
    }


def _ranking_evaluation(
    candidates: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    *,
    include_details: bool,
) -> dict[str, Any]:
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_scenario[str(candidate["scenario_id"])].append(candidate)
    pooled_pair_correct = pooled_pair_rows = 0
    scenario_pair_accuracy: list[float] = []
    top1_regrets: list[float] = []
    top1_hits: list[float] = []
    gpu_top1_regrets: list[float] = []
    gpu_top1_hits: list[float] = []
    absolute_percentage_errors: list[float] = []
    scenario_mape: list[float] = []
    details: list[dict[str, Any]] = []
    for scenario, rows in sorted(by_scenario.items()):
        pair_correct = pair_rows = 0
        row_errors = []
        for row in rows:
            predicted = math.exp(_ranker_score(row, model))
            observed = math.exp(float(row["observed_log_throughput"]))
            error = abs(predicted - observed) / observed
            row_errors.append(error)
            absolute_percentage_errors.append(error)
        if row_errors:
            scenario_mape.append(statistics.fmean(row_errors))
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                observed_delta = float(
                    rows[left]["observed_log_throughput"]
                ) - float(rows[right]["observed_log_throughput"])
                if abs(observed_delta) <= 1e-12:
                    continue
                predicted_delta = _ranker_score(
                    rows[left], model
                ) - _ranker_score(rows[right], model)
                correct = (observed_delta > 0) == (predicted_delta > 0)
                pair_correct += int(correct)
                pair_rows += 1
        pooled_pair_correct += pair_correct
        pooled_pair_rows += pair_rows
        if pair_rows:
            scenario_pair_accuracy.append(pair_correct / pair_rows)
        if len(rows) >= 2:
            oracle = max(
                rows, key=lambda row: float(row["observed_log_throughput"])
            )
            selected = max(rows, key=lambda row: _ranker_score(row, model))
            regret = max(
                0.0,
                1.0
                - math.exp(
                    float(selected["observed_log_throughput"])
                    - float(oracle["observed_log_throughput"])
                ),
            )
            top1_regrets.append(regret)
            top1_hits.append(float(regret <= 0.10))
        by_gpu: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_gpu[int(row["record"]["scenario"]["gpu_count"])].append(row)
        for gpu_count, gpu_rows in sorted(by_gpu.items()):
            if len(gpu_rows) < 2:
                continue
            oracle = max(
                gpu_rows,
                key=lambda row: float(row["observed_log_throughput"]),
            )
            selected = max(
                gpu_rows, key=lambda row: _ranker_score(row, model)
            )
            regret = max(
                0.0,
                1.0
                - math.exp(
                    float(selected["observed_log_throughput"])
                    - float(oracle["observed_log_throughput"])
                ),
            )
            gpu_top1_regrets.append(regret)
            gpu_top1_hits.append(float(regret <= 0.10))
            if include_details:
                details.append(
                    {
                        "scenario_id": scenario,
                        "scenario": rows[0]["scenario"],
                        "gpu_count": gpu_count,
                        "candidates": len(gpu_rows),
                        "selected_candidate_key": selected["candidate_key"],
                        "oracle_candidate_key": oracle["candidate_key"],
                        "top1_regret": regret,
                        "hit_at_10_percent": regret <= 0.10,
                    }
                )
    return {
        "candidate_rows": len(candidates),
        "scenario_rows": len(by_scenario),
        "pairwise_rows": pooled_pair_rows,
        "pooled_pairwise_accuracy": (
            pooled_pair_correct / pooled_pair_rows
            if pooled_pair_rows
            else None
        ),
        "scenario_equal_pairwise_accuracy": _mean(scenario_pair_accuracy),
        "scenario_equal_top1_regret": _mean(top1_regrets),
        "scenario_equal_hit_at_10_percent": _mean(top1_hits),
        "gpu_group_rows": len(gpu_top1_regrets),
        "scenario_gpu_equal_top1_regret": _mean(gpu_top1_regrets),
        "scenario_gpu_equal_hit_at_10_percent": _mean(gpu_top1_hits),
        "absolute_throughput_percentage_error": _summary(
            absolute_percentage_errors
        ),
        "scenario_equal_absolute_throughput_mape": _mean(scenario_mape),
        "details": details if include_details else None,
    }


def _throughput_cv(
    historical: Sequence[Mapping[str, Any]],
    native_calibration: Sequence[Mapping[str, Any]],
    *,
    feature_set: str,
    alpha: float,
    historical_weight: float,
    use_physics_base: bool,
) -> dict[str, Any]:
    population = [*historical, *native_calibration]
    folds = []
    for held_out in sorted(
        {str(candidate["scenario_id"]) for candidate in native_calibration}
    ):
        train = [
            candidate
            for candidate in population
            if str(candidate["scenario_id"]) != held_out
        ]
        test = [
            candidate
            for candidate in native_calibration
            if str(candidate["scenario_id"]) == held_out
        ]
        model = _fit_pairwise_ranker(
            train,
            feature_set=feature_set,
            alpha=alpha,
            historical_weight=historical_weight,
            use_physics_base=use_physics_base,
        )
        metrics = _ranking_evaluation(test, model, include_details=False)
        folds.append(metrics)

    def fold_mean(field: str) -> float | None:
        return _mean(
            [
                float(fold[field])
                for fold in folds
                if fold.get(field) is not None
            ]
        )

    return {
        "scenario_folds": len(folds),
        "scenario_equal_pairwise_accuracy": fold_mean(
            "pooled_pairwise_accuracy"
        ),
        "scenario_equal_top1_regret": fold_mean(
            "scenario_equal_top1_regret"
        ),
        "scenario_equal_hit_at_10_percent": fold_mean(
            "scenario_equal_hit_at_10_percent"
        ),
        "scenario_gpu_equal_top1_regret": fold_mean(
            "scenario_gpu_equal_top1_regret"
        ),
        "scenario_gpu_equal_hit_at_10_percent": fold_mean(
            "scenario_gpu_equal_hit_at_10_percent"
        ),
        "scenario_equal_absolute_throughput_mape": fold_mean(
            "scenario_equal_absolute_throughput_mape"
        ),
    }


def _select_throughput_model(
    historical: Sequence[Mapping[str, Any]],
    native_calibration: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for feature_set in THROUGHPUT_FEATURE_GRIDS:
        for historical_weight in THROUGHPUT_HISTORICAL_WEIGHT_GRID:
            for alpha in THROUGHPUT_ALPHA_GRID:
                for use_physics_base in (False, True):
                    cv = _throughput_cv(
                        historical,
                        native_calibration,
                        feature_set=feature_set,
                        alpha=alpha,
                        historical_weight=historical_weight,
                        use_physics_base=use_physics_base,
                    )
                    candidates.append(
                        {
                            "feature_set": feature_set,
                            "historical_scenario_weight": historical_weight,
                            "alpha": alpha,
                            "use_physics_base": use_physics_base,
                            "cv": cv,
                        }
                    )
    selected = min(
        candidates,
        key=lambda item: (
            float(item["cv"]["scenario_gpu_equal_top1_regret"]),
            -float(item["cv"]["scenario_equal_pairwise_accuracy"]),
            float(item["cv"]["scenario_equal_top1_regret"]),
            len(_throughput_feature_names(str(item["feature_set"]))),
            float(item["historical_scenario_weight"]),
            float(item["alpha"]),
            not bool(item["use_physics_base"]),
        ),
    )
    return {
        "selection_policy": (
            "minimum calibration-LOSO scenario×GPU top1 regret; "
            "pairwise accuracy then overall top1 regret are tie-breakers"
        ),
        "candidate_count": len(candidates),
        "selected": selected,
        "candidates": candidates,
    }


def _same_split_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    calibration = [
        record
        for record in records
        if (record.get("calibration_partition") or {}).get("role")
        == "calibration"
    ]
    holdout = [
        record
        for record in records
        if (record.get("calibration_partition") or {}).get("role") == "holdout"
    ]
    calibration_units = {
        (record.get("calibration_partition") or {}).get("split_unit_id")
        for record in calibration
    }
    holdout_units = {
        (record.get("calibration_partition") or {}).get("split_unit_id")
        for record in holdout
    }
    overlap = sorted(
        str(value)
        for value in calibration_units.intersection(holdout_units)
        if value is not None
    )
    return {
        "calibration_rows": len(calibration),
        "holdout_rows": len(holdout),
        "calibration_split_units": sorted(
            str(value) for value in calibration_units if value is not None
        ),
        "holdout_split_units": sorted(
            str(value) for value in holdout_units if value is not None
        ),
        "overlap": overlap,
        "disjoint": not overlap,
        "calibration_observation_ids_sha256": sha256_json(
            sorted(_observation_id(record) for record in calibration)
        ),
        "holdout_observation_ids_sha256": sha256_json(
            sorted(_observation_id(record) for record in holdout)
        ),
    }


def build_report(
    *,
    observation_path: Path,
    theory_basis_path: Path,
    theory_calibration_path: Path,
    native_memory_baseline_path: Path,
    inventory_path: Path,
    hardware_path: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    observations = _read_observations(observation_path)
    theory_basis = read_json(theory_basis_path)
    theory_calibration = read_json(theory_calibration_path)
    native_memory_baseline = read_json(native_memory_baseline_path)
    inventory = read_json(inventory_path)
    hardware = read_json(hardware_path)
    _verify_bound_report(
        theory_basis,
        expected_schema="sft_h800_theory_basis/v1",
        name="theory basis",
    )
    _verify_bound_report(
        theory_calibration,
        expected_schema="sft_h800_theory_calibration/v1",
        name="theory calibration",
    )
    _verify_bound_report(
        native_memory_baseline,
        expected_schema="sft_h800_native_memory_calibration/v1",
        name="native memory baseline",
    )
    if "h800" not in str(
        hardware.get("name_reported_by_driver") or ""
    ).lower():
        raise ValueError("Challenger modeling is H800-only")
    model_by_id, fixed_lora = _inventory_models(inventory)

    memory_admission = Counter()
    native_memory_records: list[dict[str, Any]] = []
    throughput_admission = Counter()
    native_throughput_records: list[dict[str, Any]] = []
    for row in observations:
        memory_reason = native_admission_reason(row)
        memory_admission[memory_reason] += 1
        if memory_reason == "admitted":
            native_memory_records.append(
                build_native_record(
                    row,
                    model_by_id=model_by_id,
                    fixed_lora=fixed_lora,
                    hardware=hardware,
                )
            )
        throughput_reason = throughput_admission_reason(row)
        throughput_admission[throughput_reason] += 1
        if throughput_reason == "admitted":
            native_throughput_records.append(
                _build_native_throughput_record(
                    row,
                    model_by_id=model_by_id,
                    fixed_lora=fixed_lora,
                    hardware=hardware,
                    runtime_root=runtime_root,
                )
            )
    native_memory_records.sort(key=_observation_id)
    native_throughput_records.sort(key=_observation_id)
    memory_split = _same_split_audit(native_memory_records)
    throughput_split = _same_split_audit(native_throughput_records)
    if not memory_split["disjoint"] or not throughput_split["disjoint"]:
        raise ValueError("Calibration and holdout split units overlap")
    memory_calibration = [
        record
        for record in native_memory_records
        if record["calibration_partition"]["role"] == "calibration"
    ]
    memory_holdout = [
        record
        for record in native_memory_records
        if record["calibration_partition"]["role"] == "holdout"
    ]
    throughput_calibration_records = [
        record
        for record in native_throughput_records
        if record["calibration_partition"]["role"] == "calibration"
    ]
    throughput_holdout_records = [
        record
        for record in native_throughput_records
        if record["calibration_partition"]["role"] == "holdout"
    ]

    basis_records = theory_basis.get("records")
    if not isinstance(basis_records, list):
        raise ValueError("Theory basis has no records")
    historical_memory = [
        record
        for record in basis_records
        if isinstance(record, Mapping)
        and (record.get("route") or {}).get("memory_boundary") is True
        and _outcome(record) in {"success", "oom"}
    ]
    historical_throughput_records = [
        record
        for record in basis_records
        if isinstance(record, Mapping)
        and (record.get("route") or {}).get("throughput_primary") is True
        and _outcome(record) == "success"
    ]

    # Select both challengers using calibration data only.
    memory_selection = _select_memory_model(
        historical_memory, memory_calibration
    )
    selected_memory = memory_selection["selected"]
    memory_nested_cv = _memory_nested_cv(
        historical_memory, memory_calibration, selected_memory
    )
    augmented_memory = [*historical_memory, *memory_calibration]
    frozen_memory_center = _fit_memory_ridge(
        augmented_memory,
        feature_set=str(selected_memory["feature_set"]),
        alpha=float(selected_memory["alpha"]),
        historical_weight=float(
            selected_memory["historical_scenario_weight"]
        ),
    )
    frozen_memory_tail = _fit_memory_tail(
        augmented_memory,
        feature_set=str(selected_memory["feature_set"]),
        alpha=float(selected_memory["alpha"]),
        historical_weight=float(
            selected_memory["historical_scenario_weight"]
        ),
    )
    allocated_model = _fit_memory_ridge(
        augmented_memory,
        feature_set=str(selected_memory["feature_set"]),
        alpha=float(selected_memory["alpha"]),
        historical_weight=float(
            selected_memory["historical_scenario_weight"]
        ),
        label="allocated",
    )

    physical_model = (
        theory_calibration["full_historical_bootstrap_fit"]["throughput"][
            "center"
        ]
    )
    physical_priors = theory_calibration["physical_priors"]["values"]
    historical_throughput = _throughput_candidates(
        historical_throughput_records,
        physical_model=physical_model,
        priors=physical_priors,
    )
    native_throughput_calibration = _throughput_candidates(
        throughput_calibration_records,
        physical_model=physical_model,
        priors=physical_priors,
    )
    native_throughput_holdout = _throughput_candidates(
        throughput_holdout_records,
        physical_model=physical_model,
        priors=physical_priors,
    )
    throughput_selection = _select_throughput_model(
        historical_throughput, native_throughput_calibration
    )
    selected_throughput = throughput_selection["selected"]
    frozen_ranker = _fit_pairwise_ranker(
        [*historical_throughput, *native_throughput_calibration],
        feature_set=str(selected_throughput["feature_set"]),
        alpha=float(selected_throughput["alpha"]),
        historical_weight=float(
            selected_throughput["historical_scenario_weight"]
        ),
        use_physics_base=bool(selected_throughput["use_physics_base"]),
    )

    # Both challengers are now selected and frozen.  Only from this point on may
    # either native holdout partition be evaluated.
    memory_holdout_evaluation = _memory_evaluation(
        memory_holdout,
        frozen_memory_center,
        frozen_memory_tail,
        include_details=True,
    )
    allocated_holdout_evaluation = _allocated_center_evaluation(
        memory_holdout, allocated_model
    )
    baseline_ranker = _baseline_ranker_model()
    throughput_calibration_baseline = _ranking_evaluation(
        native_throughput_calibration,
        baseline_ranker,
        include_details=False,
    )
    throughput_holdout_baseline = _ranking_evaluation(
        native_throughput_holdout,
        baseline_ranker,
        include_details=True,
    )
    throughput_holdout_challenger = _ranking_evaluation(
        native_throughput_holdout,
        frozen_ranker,
        include_details=True,
    )

    current_memory_holdout = native_memory_baseline[
        "frozen_native_holdout"
    ]["augmented_candidate"]
    current_memory_cv = native_memory_baseline[
        "native_calibration_augmented_loocv"
    ]["aggregate"]
    memory_challenger_pass = (
        memory_holdout_evaluation["success_p95_coverage"] is not None
        and float(memory_holdout_evaluation["success_p95_coverage"]) >= 0.95
        and memory_holdout_evaluation["false_safe_oom"] == 0
        and memory_holdout_evaluation["prediction_unavailable_rows"] == 0
    )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "offline_challengers_frozen_and_compared",
        "gpu_family": "H800",
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "production_profile_generated": False,
        "publishable": False,
        "source_bindings": {
            "canonical_observations": {
                "path": str(observation_path),
                "sha256": sha256_file(observation_path),
                "rows": len(observations),
            },
            "theory_basis": {
                "path": str(theory_basis_path),
                "sha256": sha256_file(theory_basis_path),
                "report_sha256": theory_basis.get("report_sha256"),
            },
            "theory_calibration": {
                "path": str(theory_calibration_path),
                "sha256": sha256_file(theory_calibration_path),
                "report_sha256": theory_calibration.get("report_sha256"),
            },
            "native_memory_baseline": {
                "path": str(native_memory_baseline_path),
                "sha256": sha256_file(native_memory_baseline_path),
                "report_sha256": native_memory_baseline.get("report_sha256"),
            },
            "model_inventory": {
                "path": str(inventory_path),
                "sha256": sha256_file(inventory_path),
            },
            "hardware": {
                "path": str(hardware_path),
                "sha256": sha256_file(hardware_path),
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "protocol": {
            "model_and_hyperparameter_selection": (
                "native calibration leave-scenario-out only"
            ),
            "holdout_used_for_fit_or_selection": False,
            "holdout_touched_after_both_challengers_frozen": True,
            "holdout_is_fresh_publication_acceptance": False,
            "holdout_limitation": (
                "the same holdout was inspected by the earlier stage-1 report; "
                "this is a comparative diagnostic, not a new unbiased acceptance"
            ),
            "packing_used": False,
            "supported_mbs": sorted(SUPPORTED_MBS),
        },
        "data_admission": {
            "memory": {
                "counts": dict(sorted(memory_admission.items())),
                "split": memory_split,
                "outcomes": dict(
                    sorted(
                        Counter(
                            _outcome(record)
                            for record in native_memory_records
                        ).items()
                    )
                ),
            },
            "throughput": {
                "counts": dict(sorted(throughput_admission.items())),
                "split": throughput_split,
                "native_calibration_records": len(
                    throughput_calibration_records
                ),
                "native_holdout_records": len(throughput_holdout_records),
                "native_calibration_candidates_after_replicate_pooling": len(
                    native_throughput_calibration
                ),
                "native_holdout_candidates_after_replicate_pooling": len(
                    native_throughput_holdout
                ),
            },
        },
        "memory": {
            "model_contract": {
                "center_formula": (
                    "analytic_reference_bytes * exp(beta0 + "
                    "standardized_physical_and_mechanism_features @ beta)"
                ),
                "ridge_objective": (
                    "scenario-equal weighted log-residual squared error + "
                    "alpha * ||beta||^2"
                ),
                "tail_formula": frozen_memory_tail["formula"],
                "oom_contract": (
                    "right-censored lower inequality; OOM peak is never imputed"
                ),
            },
            "selection": memory_selection,
            "selected_nested_calibration_loocv": memory_nested_cv,
            "current_model_calibration_loocv": current_memory_cv,
            "frozen_model": {
                "reserved_center": frozen_memory_center,
                "allocated_center_diagnostic": allocated_model,
                "tail": frozen_memory_tail,
                "training_rows": len(augmented_memory),
                "training_observation_ids_sha256": sha256_json(
                    sorted(
                        _observation_id(record)
                        for record in augmented_memory
                    )
                ),
                "holdout_rows_in_training": 0,
            },
            "same_native_holdout_comparison": {
                "current_augmented_model": current_memory_holdout,
                "challenger": {
                    "reserved_and_safety": memory_holdout_evaluation,
                    "allocated_center_diagnostic": allocated_holdout_evaluation,
                },
                "same_rows": True,
                "challenger_safety_gate_passed": memory_challenger_pass,
                "not_fresh_publication_acceptance": True,
            },
        },
        "throughput": {
            "model_contract": {
                "ranking_score_formula": (
                    "log(frozen_physics_baseline_throughput) + intercept + "
                    "standardized_configuration_and_physical_features @ beta"
                ),
                "pairwise_objective": (
                    "sum over scenarios of mean squared error between observed "
                    "and predicted log-throughput differences + alpha*||beta||^2"
                ),
                "absolute_prediction_is_diagnostic": True,
                "primary_use": "ordering_existing_successful_configurations",
            },
            "historical_primary_records": len(
                historical_throughput_records
            ),
            "selection": throughput_selection,
            "calibration_physics_baseline": throughput_calibration_baseline,
            "frozen_model": {
                **frozen_ranker,
                "training_observation_ids_sha256": sha256_json(
                    sorted(
                        observation_id
                        for candidate in [
                            *historical_throughput,
                            *native_throughput_calibration,
                        ]
                        for observation_id in candidate["observation_ids"]
                    )
                ),
                "holdout_rows_in_training": 0,
            },
            "same_native_holdout_comparison": {
                "physics_baseline": throughput_holdout_baseline,
                "pairwise_challenger": throughput_holdout_challenger,
                "same_rows": True,
                "not_fresh_publication_acceptance": True,
            },
        },
        "publication_blockers": [
            "challenger_report_is_diagnostic_and_never_auto_publishes",
            "holdout_was_previously_inspected_not_fresh_acceptance",
            "prospective_unseen_scenario_acceptance_still_required",
        ],
    }
    report["report_sha256"] = sha256_json(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema") != SCHEMA:
        raise ValueError("Challenger report schema mismatch")
    unsigned = dict(report)
    digest = unsigned.pop("report_sha256", None)
    if digest != sha256_json(unsigned):
        raise ValueError("Challenger report SHA-256 mismatch")
    if report.get("gpu_experiments_launched") is not False:
        raise ValueError("Challenger modeling cannot launch GPU experiments")
    if report.get("queues_mutated") is not False:
        raise ValueError("Challenger modeling cannot mutate queues")
    if (
        report.get("publishable") is not False
        or report.get("production_profile_generated") is not False
    ):
        raise ValueError("Challenger report must remain non-publishable")
    protocol = report.get("protocol")
    protocol = protocol if isinstance(protocol, Mapping) else {}
    if (
        protocol.get("holdout_used_for_fit_or_selection") is not False
        or protocol.get("holdout_touched_after_both_challengers_frozen")
        is not True
    ):
        raise ValueError("Holdout separation protocol is missing")
    admission = report.get("data_admission")
    admission = admission if isinstance(admission, Mapping) else {}
    for component in ("memory", "throughput"):
        split = (admission.get(component) or {}).get("split") or {}
        if split.get("disjoint") is not True or split.get("overlap"):
            raise ValueError(f"{component} calibration/holdout split overlaps")
    memory_model = (
        ((report.get("memory") or {}).get("frozen_model") or {})
    )
    throughput_model = (
        ((report.get("throughput") or {}).get("frozen_model") or {})
    )
    if (
        memory_model.get("holdout_rows_in_training") != 0
        or throughput_model.get("holdout_rows_in_training") != 0
    ):
        raise ValueError("A frozen challenger contains holdout rows")
    json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    validate_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(
                report,
                output,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        type=Path,
        default=ROOT / "artifacts" / "canonical_h800_observations.jsonl",
    )
    parser.add_argument(
        "--theory-basis",
        type=Path,
        default=ROOT / "artifacts" / "h800_theory_basis.json",
    )
    parser.add_argument(
        "--theory-calibration",
        type=Path,
        default=ROOT / "artifacts" / "h800_theory_calibration.json",
    )
    parser.add_argument(
        "--native-memory-baseline",
        type=Path,
        default=ROOT / "artifacts" / "h800_native_memory_calibration.json",
    )
    parser.add_argument(
        "--model-inventory",
        type=Path,
        default=ROOT / "artifacts" / "model_inventory.json",
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=ROOT / "config" / "hardware.json",
    )
    parser.add_argument(
        "--runtime-root", type=Path, default=ROOT / "runtime"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "h800_challenger_modeling.json",
    )
    args = parser.parse_args()
    report = build_report(
        observation_path=args.observations,
        theory_basis_path=args.theory_basis,
        theory_calibration_path=args.theory_calibration,
        native_memory_baseline_path=args.native_memory_baseline,
        inventory_path=args.model_inventory,
        hardware_path=args.hardware,
        runtime_root=args.runtime_root,
    )
    write_report(args.output, report)
    memory_current = report["memory"]["same_native_holdout_comparison"][
        "current_augmented_model"
    ]
    memory_challenger = report["memory"]["same_native_holdout_comparison"][
        "challenger"
    ]["reserved_and_safety"]
    throughput_comparison = report["throughput"][
        "same_native_holdout_comparison"
    ]
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "output": str(args.output),
                "gpu_experiments_launched": report[
                    "gpu_experiments_launched"
                ],
                "memory_holdout": {
                    "current_reserved_center_mean_ape": memory_current[
                        "center_accuracy"
                    ]["reserved_center"]["absolute_percentage_error"]["mean"],
                    "challenger_reserved_center_mean_ape": memory_challenger[
                        "reserved_center_absolute_percentage_error"
                    ]["mean"],
                    "current_false_safe_oom": memory_current[
                        "operational_safety"
                    ]["false_safe_oom"],
                    "challenger_false_safe_oom": memory_challenger[
                        "false_safe_oom"
                    ],
                    "challenger_safety_gate_passed": report["memory"][
                        "same_native_holdout_comparison"
                    ]["challenger_safety_gate_passed"],
                },
                "throughput_holdout": {
                    "baseline_pairwise_accuracy": throughput_comparison[
                        "physics_baseline"
                    ]["pooled_pairwise_accuracy"],
                    "challenger_pairwise_accuracy": throughput_comparison[
                        "pairwise_challenger"
                    ]["pooled_pairwise_accuracy"],
                    "baseline_gpu_top1_regret": throughput_comparison[
                        "physics_baseline"
                    ]["scenario_gpu_equal_top1_regret"],
                    "challenger_gpu_top1_regret": throughput_comparison[
                        "pairwise_challenger"
                    ]["scenario_gpu_equal_top1_regret"],
                },
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
