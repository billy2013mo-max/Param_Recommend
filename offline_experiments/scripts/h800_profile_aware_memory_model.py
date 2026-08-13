#!/usr/bin/env python3
"""Shared math for the versioned H800 profile-aware memory challenger.

The model deliberately remains a correction over the frozen physical-shares
allocated-memory head.  It has two independently auditable multiplicative
parts:

``allocated = physical_allocated * exp(profile_correction)``

``reserved = allocated * exp(reservation_ratio)``

The operational upper bound is a selector-and-MBS leave-profile-out residual
guard.  The small number of independent profiles is exposed in the artifact;
the guard is not described as a statistically identified P95 interval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

import numpy as np


ARTIFACT_SCHEMA = "sft_h800_profile_aware_memory_challenger/v1"
IMPLEMENTATION_VERSION = (
    "sft_h800_profile_aware_memory_model/"
    "2026-08-03.physical-anchor-two-stage-profile-correction"
)


def selector_mbs_key(record: Mapping[str, Any]) -> str:
    """Return the stable operational-tail bucket for one static record."""

    selector = record.get("selector") or {}
    scenario = record.get("scenario") or {}
    material = [
        str(selector.get("training_mode") or "unknown"),
        int(selector.get("zero_stage") or 0),
        bool(selector.get("gradient_checkpointing")),
        bool(selector.get("packing")),
        int(scenario.get("physical_mbs") or 0),
    ]
    return json.dumps(material, ensure_ascii=False, separators=(",", ":"))


def legacy_selector_key(record: Mapping[str, Any]) -> str:
    """Return the exact-selector key used by the older OOM guard."""

    selector = record.get("selector") or {}
    material = [
        str(selector.get("training_mode") or "unknown"),
        int(selector.get("zero_stage") or 0),
        bool(selector.get("gradient_checkpointing")),
        bool(selector.get("packing")),
    ]
    return json.dumps(material, ensure_ascii=False, separators=(",", ":"))


def base_feature_values(
    record: Mapping[str, Any],
    padding: Mapping[str, Any],
    *,
    allocation_saturation_rate: float = 16.0,
    reservation_saturation_rate: float = 8.0,
) -> dict[str, float]:
    """Build the bounded feature dictionary used by both model heads."""

    selector = record.get("selector") or {}
    scenario = record.get("scenario") or {}
    memory = record.get("memory") or {}
    reference = float(memory["analytic_reference_bytes"])
    activation = float(memory["structural_activation_bytes"])
    pressure = float(
        padding["expected_random_batch_max_fraction_of_cutoff"]
    )
    if not 0.0 <= pressure <= 1.0 + 1e-9:
        raise ValueError("padding pressure must be between zero and one")
    pressure = min(1.0, max(0.0, pressure))
    is_lora = float(selector.get("training_mode") == "lora")
    risk = float(
        selector.get("training_mode") == "lora"
        and int(selector.get("zero_stage") or 0) == 2
        and not bool(selector.get("gradient_checkpointing"))
    )
    log_mbs = math.log2(float(scenario["physical_mbs"]))
    log_gpu = math.log2(float(scenario["gpu_count"]))
    allocation_saturation = 1.0 - math.exp(
        -float(allocation_saturation_rate) * pressure
    )
    reservation_saturation = 1.0 - math.exp(
        -float(reservation_saturation_rate) * pressure
    )
    activation_share = activation / reference
    return {
        "is_lora": is_lora,
        "activation_share": activation_share,
        "allocation_saturation": allocation_saturation,
        "lora_x_allocation_saturation": is_lora * allocation_saturation,
        "activation_share_x_allocation_saturation": (
            activation_share * allocation_saturation
        ),
        "reservation_saturation": reservation_saturation,
        "padding_pressure": pressure,
        "p99_fraction_of_cutoff": float(padding["p99_clipped_tokens"])
        / float(scenario["cutoff_len"]),
        "coefficient_of_variation": float(
            padding["coefficient_of_variation"]
        ),
        "truncation_fraction": float(padding["truncation_fraction"]),
        "lora_zero2_gc_off": risk,
        "log2_mbs": log_mbs,
        "log2_gpu_count": log_gpu,
        "risk_x_padding_pressure": risk * pressure,
        "risk_x_padding_pressure_squared": risk * pressure**2,
        "risk_x_padding_pressure_cubed": risk * pressure**3,
        "risk_x_padding_pressure_cubed_x_log2_mbs": (
            risk * pressure**3 * log_mbs
        ),
    }


def vector(values: Mapping[str, float], names: Sequence[str]) -> np.ndarray:
    return np.asarray([float(values[name]) for name in names], dtype=float)


def predict_log_correction(
    values: Mapping[str, float], model: Mapping[str, Any]
) -> float:
    names = tuple(str(name) for name in model.get("feature_names") or [])
    features = vector(values, names)
    means = np.asarray(model.get("feature_means") or [], dtype=float)
    scales = np.asarray(model.get("feature_scales") or [], dtype=float)
    coefficients = np.asarray(model.get("coefficients") or [], dtype=float)
    if not (
        len(features) == len(means) == len(scales) == len(coefficients)
    ):
        raise ValueError("profile-aware model feature dimensions do not match")
    if np.any(scales <= 0.0):
        raise ValueError("profile-aware model contains a non-positive scale")
    result = float(model["intercept"])
    if len(features):
        result += float(((features - means) / scales) @ coefficients)
    if not math.isfinite(result):
        raise ValueError("profile-aware model produced a non-finite correction")
    return result


def predict_memory(
    record: Mapping[str, Any],
    padding: Mapping[str, Any],
    *,
    allocated_anchor_bytes: float,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Predict allocated center, reserved center and guarded upper bytes."""

    if artifact.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("profile-aware memory artifact schema mismatch")
    model = artifact.get("model") or {}
    allocation = model.get("allocated_profile_correction") or {}
    reservation = model.get("reservation_ratio") or {}
    values = base_feature_values(
        record,
        padding,
        allocation_saturation_rate=float(
            allocation["feature_parameters"]["saturation_rate"]
        ),
        reservation_saturation_rate=float(
            reservation["feature_parameters"]["saturation_rate"]
        ),
    )
    allocation_log = predict_log_correction(values, allocation)
    reservation_log = predict_log_correction(values, reservation)
    allocated_center = float(allocated_anchor_bytes) * math.exp(allocation_log)
    reserved_center = allocated_center * math.exp(reservation_log)

    tail = model.get("operational_upper") or {}
    bucket_key = selector_mbs_key(record)
    bucket = (tail.get("selector_mbs") or {}).get(bucket_key) or {}
    profile_guard = bucket.get("log_residual_upper")
    tail_source = "selector_mbs"
    if profile_guard is None:
        profile_guard = (tail.get("pooled") or {}).get("log_residual_upper")
        tail_source = "pooled"
    if profile_guard is None:
        return {
            "available": False,
            "allocated_center_bytes": allocated_center,
            "reserved_center_bytes": reserved_center,
            "issues": ["operational_upper_guard_unavailable"],
        }
    legacy = (tail.get("legacy_exact_selector_oom_guard") or {}).get(
        legacy_selector_key(record)
    ) or {}
    legacy_guard = float(legacy.get("log_residual_lower") or 0.0)
    guard = max(0.0, float(profile_guard), legacy_guard)
    upper = reserved_center * math.exp(guard)
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (allocated_center, reserved_center, upper)
    ):
        raise ValueError("profile-aware memory prediction is non-positive")
    return {
        "available": True,
        "allocated_anchor_bytes": float(allocated_anchor_bytes),
        "allocated_profile_log_correction": allocation_log,
        "allocated_center_bytes": allocated_center,
        "reservation_ratio_log": reservation_log,
        "reserved_center_bytes": reserved_center,
        "operational_upper_reserved_bytes": upper,
        "operational_log_guard": guard,
        "profile_guard_log_residual": float(profile_guard),
        "legacy_oom_guard_log_residual": legacy_guard,
        "tail_source": tail_source,
        "tail_bucket": bucket_key,
        "padding_features": values,
        "issues": [],
    }
