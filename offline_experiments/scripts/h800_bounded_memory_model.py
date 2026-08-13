#!/usr/bin/env python3
"""Shared inference math for the bounded H800 memory challenger v2.

The v2 admission center is a single correction over the physical allocated
anchor.  Dataset-shape inputs are bounded in ``[0, 1]`` and the model refuses
selectors that have no calibration bucket.  This prevents the unbounded
polynomial extrapolation that invalidated the v1 unseen-profile holdout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

import numpy as np

from h800_profile_aware_memory_model import legacy_selector_key


ARTIFACT_SCHEMA = "sft_h800_bounded_memory_challenger/v2"
IMPLEMENTATION_VERSION = (
    "sft_h800_bounded_memory_model/"
    "2026-08-03.direct-reserved-bounded-shape-v2"
)
SELECTOR_FEATURE_PREFIX = "selector::"


def selector_bucket_key(record: Mapping[str, Any]) -> str:
    """Return the exact mechanism bucket admitted by the bounded model."""

    selector = record.get("selector") or {}
    scenario = record.get("scenario") or {}
    material = [
        str(selector.get("training_mode") or "unknown"),
        int(scenario.get("gpu_count") or 0),
        int(selector.get("zero_stage") or 0),
        bool(selector.get("gradient_checkpointing")),
        bool(selector.get("packing")),
        int(scenario.get("physical_mbs") or 0),
    ]
    return json.dumps(material, ensure_ascii=False, separators=(",", ":"))


def selector_feature_name(bucket_key: str) -> str:
    return SELECTOR_FEATURE_PREFIX + str(bucket_key)


def bounded_feature_values(
    record: Mapping[str, Any],
    padding: Mapping[str, Any],
    *,
    supported_selector_keys: Sequence[str],
) -> dict[str, float]:
    """Build the bounded shape features and exact-selector one-hot values."""

    selector = record.get("selector") or {}
    scenario = record.get("scenario") or {}
    cutoff = float(scenario["cutoff_len"])
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff_len must be positive")

    maximum_fraction = float(padding["maximum_clipped_tokens"]) / cutoff
    p99_fraction = float(padding["p99_clipped_tokens"]) / cutoff
    batch_pressure = float(
        padding["expected_random_batch_max_fraction_of_cutoff"]
    )
    truncation = float(padding["truncation_fraction"])
    for name, value in (
        ("maximum_fraction", maximum_fraction),
        ("p99_fraction", p99_fraction),
        ("batch_pressure", batch_pressure),
        ("truncation_fraction", truncation),
    ):
        if not math.isfinite(value) or not -1e-9 <= value <= 1.0 + 1e-9:
            raise ValueError(f"{name} must be between zero and one")
    maximum_fraction = min(1.0, max(0.0, maximum_fraction))
    p99_fraction = min(1.0, max(0.0, p99_fraction))
    batch_pressure = min(1.0, max(0.0, batch_pressure))
    truncation = min(1.0, max(0.0, truncation))

    is_lora = float(selector.get("training_mode") == "lora")
    is_risk = float(
        selector.get("training_mode") == "lora"
        and int(selector.get("zero_stage") or 0) == 2
        and not bool(selector.get("gradient_checkpointing"))
    )
    log2_mbs = math.log2(float(scenario["physical_mbs"]))
    fragmentation_pressure = batch_pressure * (1.0 - truncation)
    bucket = selector_bucket_key(record)
    values = {
        "lora_maximum_fraction": is_lora * maximum_fraction,
        "lora_p99_fraction": is_lora * p99_fraction,
        "lora_fragmentation_pressure": is_lora * fragmentation_pressure,
        "risk_fragmentation_pressure": is_risk * fragmentation_pressure,
        "risk_fragmentation_pressure_x_log2_mbs": (
            is_risk * fragmentation_pressure * log2_mbs
        ),
    }
    for supported in supported_selector_keys:
        values[selector_feature_name(str(supported))] = float(bucket == supported)
    return values


def feature_vector(
    values: Mapping[str, float], names: Sequence[str]
) -> np.ndarray:
    return np.asarray([float(values[name]) for name in names], dtype=float)


def predict_log_correction(
    values: Mapping[str, float], model: Mapping[str, Any]
) -> float:
    names = tuple(str(name) for name in model.get("feature_names") or [])
    features = feature_vector(values, names)
    means = np.asarray(model.get("feature_means") or [], dtype=float)
    scales = np.asarray(model.get("feature_scales") or [], dtype=float)
    coefficients = np.asarray(model.get("coefficients") or [], dtype=float)
    if not (
        len(features) == len(means) == len(scales) == len(coefficients)
    ):
        raise ValueError("bounded memory model feature dimensions do not match")
    if np.any(scales <= 0.0):
        raise ValueError("bounded memory model contains a non-positive scale")
    result = float(model["intercept"])
    if len(features):
        result += float(((features - means) / scales) @ coefficients)
    if not math.isfinite(result):
        raise ValueError("bounded memory model produced a non-finite correction")
    return result


def predict_memory(
    record: Mapping[str, Any],
    padding: Mapping[str, Any],
    *,
    allocated_anchor_bytes: float,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Predict diagnostic allocated, direct reserved center and safe upper."""

    if artifact.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("bounded memory artifact schema mismatch")
    model = artifact.get("model") or {}
    supported = tuple(str(key) for key in model.get("supported_selector_keys") or [])
    bucket = selector_bucket_key(record)
    if bucket not in supported:
        return {
            "available": False,
            "selector_bucket": bucket,
            "issues": ["unseen_selector_bucket"],
        }

    values = bounded_feature_values(
        record,
        padding,
        supported_selector_keys=supported,
    )
    allocated_model = model.get("allocated_diagnostic") or {}
    reserved_model = model.get("direct_reserved_center") or {}
    allocated_log = predict_log_correction(values, allocated_model)
    reserved_log = predict_log_correction(values, reserved_model)
    anchor = float(allocated_anchor_bytes)
    if not math.isfinite(anchor) or anchor <= 0.0:
        raise ValueError("allocated anchor must be positive")
    allocated_center = anchor * math.exp(allocated_log)
    reserved_center = anchor * math.exp(reserved_log)

    upper_model = model.get("operational_upper") or {}
    residual = (upper_model.get("center_residual_guard_by_selector") or {}).get(
        bucket
    )
    envelope = (upper_model.get("anchor_envelope_by_selector") or {}).get(bucket)
    if not isinstance(residual, Mapping) or not isinstance(envelope, Mapping):
        return {
            "available": False,
            "selector_bucket": bucket,
            "allocated_center_bytes": allocated_center,
            "reserved_center_bytes": reserved_center,
            "issues": ["operational_upper_guard_unavailable"],
        }
    center_guard_log = max(0.0, float(residual["log_residual_upper"]))
    anchor_envelope_log = max(0.0, float(envelope["log_correction_upper"]))
    legacy = (upper_model.get("legacy_exact_selector_oom_guard") or {}).get(
        legacy_selector_key(record)
    ) or {}
    legacy_guard_log = max(0.0, float(legacy.get("log_residual_lower") or 0.0))

    center_guarded_upper = reserved_center * math.exp(center_guard_log)
    anchor_envelope_upper = anchor * math.exp(anchor_envelope_log)
    legacy_anchor_upper = anchor * math.exp(legacy_guard_log)
    upper = max(
        center_guarded_upper,
        anchor_envelope_upper,
        legacy_anchor_upper,
    )
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (allocated_center, reserved_center, upper)
    ):
        raise ValueError("bounded memory prediction is non-positive")
    return {
        "available": True,
        "selector_bucket": bucket,
        "allocated_anchor_bytes": anchor,
        "allocated_profile_log_correction": allocated_log,
        "allocated_center_bytes": allocated_center,
        "direct_reserved_log_correction": reserved_log,
        "reserved_center_bytes": reserved_center,
        "operational_upper_reserved_bytes": upper,
        "center_guarded_upper_bytes": center_guarded_upper,
        "anchor_envelope_upper_bytes": anchor_envelope_upper,
        "legacy_anchor_upper_bytes": legacy_anchor_upper,
        "center_residual_guard_log": center_guard_log,
        "anchor_envelope_guard_log": anchor_envelope_log,
        "legacy_oom_guard_log": legacy_guard_log,
        "padding_features": values,
        "issues": [],
    }
