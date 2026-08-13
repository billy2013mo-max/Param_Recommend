#!/usr/bin/env python3
"""Shadow-only H800 overlay for a frozen VL tower and projector.

The base text model remains responsible for language-backbone memory and step
time.  This module adds only the visual residual calibrated by the paired
real-media versus token-length-matched campaign.  It deliberately does not
make an admission decision: the current evidence is retrospective development
evidence and has no source-disjoint prospective acceptance set.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA = "sft_h800_frozen_vl_residual_overlay/v1"
PREDICTION_SCHEMA = "sft_h800_frozen_vl_overlay_prediction/v1"


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and non-negative") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def load_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError(f"unsupported VL overlay artifact: {artifact.get('schema')}")
    if (artifact.get("release_contract") or {}).get("mode") != "shadow_only":
        raise ValueError("VL overlay V1 must remain shadow-only")
    return artifact


def predict_shadow_overlay(
    *,
    text_memory_center_bytes: float | None,
    text_step_seconds: float | None,
    effective_tokens_per_step: float | None,
    logical_samples_per_step: float,
    vl_features: Mapping[str, Any],
    model_family: str,
    artifact: Mapping[str, Any],
    text_memory_right_censored: bool = False,
) -> dict[str, Any]:
    """Apply the frozen-VL residual heads without issuing a recommendation."""

    if artifact.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("VL overlay artifact schema mismatch")
    if (artifact.get("release_contract") or {}).get("mode") != "shadow_only":
        raise ValueError("VL overlay V1 must remain shadow-only")

    samples = _finite_nonnegative(logical_samples_per_step, "logical_samples_per_step")
    memory_proxy = _finite_nonnegative(
        (
            (vl_features.get("memory_proxies_bytes") or {}).get(
                "vision_dynamic_microbatch"
            )
        ),
        "vision_dynamic_microbatch",
    )
    flops_per_sample = _finite_nonnegative(
        (
            (vl_features.get("throughput_proxies_per_sample") or {}).get(
                "vision_forward_flops"
            )
        ),
        "vision_forward_flops",
    )

    memory_head = artifact.get("memory_center_head") or {}
    memory_coefficient = _finite_nonnegative(
        memory_head.get("coefficient_bytes_per_proxy_byte"),
        "memory coefficient",
    )
    visual_memory_delta = memory_coefficient * memory_proxy
    if text_memory_right_censored:
        memory_state = "right_censored_base_unsafe"
        memory_center = None
    else:
        base_memory = _finite_nonnegative(
            text_memory_center_bytes, "text_memory_center_bytes"
        )
        memory_state = "shadow_center_available"
        memory_center = base_memory + visual_memory_delta

    throughput_head = artifact.get("throughput_head") or {}
    coefficients = throughput_head.get("family_coefficients_seconds_per_pflop") or {}
    if model_family not in coefficients:
        raise ValueError(f"unsupported VL model family: {model_family}")
    throughput_coefficient = _finite_nonnegative(
        coefficients[model_family], f"throughput coefficient for {model_family}"
    )
    visual_pflop = flops_per_sample * samples / 1.0e15
    visual_seconds = throughput_coefficient * visual_pflop
    if text_step_seconds is None:
        step_seconds = None
        tokens_per_second = None
    else:
        base_seconds = _finite_nonnegative(text_step_seconds, "text_step_seconds")
        step_seconds = base_seconds + visual_seconds
        tokens = _finite_nonnegative(
            effective_tokens_per_step, "effective_tokens_per_step"
        )
        tokens_per_second = tokens / step_seconds if step_seconds > 0.0 else None

    return {
        "schema": PREDICTION_SCHEMA,
        "mode": "shadow_only",
        "automatic_admission_allowed": False,
        "automatic_ranking_allowed": False,
        "model_family": model_family,
        "memory": {
            "state": memory_state,
            "text_center_bytes": text_memory_center_bytes,
            "visual_dynamic_proxy_bytes": memory_proxy,
            "visual_residual_bytes": visual_memory_delta,
            "predicted_center_bytes": memory_center,
            "safety_upper_bytes": None,
        },
        "throughput": {
            "text_step_seconds": text_step_seconds,
            "visual_forward_pflop": visual_pflop,
            "visual_seconds": visual_seconds,
            "predicted_step_seconds": step_seconds,
            "effective_tokens_per_second": tokens_per_second,
        },
        "scope": {
            "vision_tower_frozen": True,
            "multimodal_projector_frozen": True,
            "packing": False,
            "hardware": "NVIDIA H800 140GB HBM3",
        },
    }
