#!/usr/bin/env python3
"""Hybrid-attention (Qwen3.5/3.6) memory prediction bridge.

Independent shadow channel for stage-1 dense hybrid memory coefficients.
Builds the same structural feature vector the stage-1 fit used (intercept +
state/saved_full/saved_linear/recompute/full_workspace/linear_workspace/
linear_state/logits/zero_workspace/zero3_saved) from the real checkpoint
config.json, and applies the fitted coefficient set.

This module does NOT touch the production V3 predictor.  It is the
forward-propagating piece a V3 integration would call, and it can be tested
in isolation against the stage-1 observations.
"""

from __future__ import annotations

from typing import Any, Mapping

from hybrid_attention_memory_features import build_dense_hybrid_features
from hybrid_attention_memory_features_v2 import (
    ZERO3_PARAM_WORKSPACE_KEY,
    build_dense_hybrid_features_v2,
)

PARAMETER_BYTES = 2


def hybrid_memory_design(
    model_row: Mapping[str, Any],
    job: Mapping[str, Any],
    fixed_lora: Mapping[str, Any],
    capacity_bytes: int,
) -> dict[str, float]:
    feats = build_dense_hybrid_features(
        job, model_row, fixed_lora=fixed_lora, capacity_bytes=capacity_bytes
    )
    memory = feats["memory"]
    act = memory["activation_components"]
    ws = memory["workspace_candidates"]
    zero_stage = str((job.get("zero") or "none").lower())
    saved_total = float(
        act["saved_full_attention_activations_bytes"]
        + act["saved_linear_attention_activations_bytes"]
    )
    return {
        "intercept": 1.0,
        "state": float(memory["state_bytes"]),
        "saved_full": float(act["saved_full_attention_activations_bytes"]),
        "saved_linear": float(act["saved_linear_attention_activations_bytes"]),
        "recompute": float(act["recompute_workspace_bytes"]),
        "full_workspace": float(ws["full_attention_workspace_bytes"]),
        "linear_workspace": float(ws["linear_attention_workspace_bytes"]),
        "linear_state": float(ws["linear_recurrent_state_bytes"]),
        "logits": float(ws["logits_workspace_bytes"]),
        "zero_workspace": float(ws["zero_collective_workspace_bytes"]),
        "zero3_saved": saved_total if zero_stage == "zero3" else 0.0,
    }


def predict_memory_center(
    model_row: Mapping[str, Any],
    job: Mapping[str, Any],
    coefficients_by_name: Mapping[str, float],
    fixed_lora: Mapping[str, Any],
    capacity_bytes: int,
) -> dict[str, Any]:
    design = hybrid_memory_design(
        model_row, job, fixed_lora=fixed_lora, capacity_bytes=capacity_bytes
    )
    center = sum(float(coefficients_by_name[k]) * v for k, v in design.items())
    return {"center_bytes": max(center, 0.0), "design_bytes": design}


def hybrid_memory_design_v2(
    model_row: Mapping[str, Any],
    job: Mapping[str, Any],
    fixed_lora: Mapping[str, Any],
    capacity_bytes: int,
) -> dict[str, float]:
    """Stage-2 design: the stage-1 columns plus the ZeRO-3 param workspace."""

    design = hybrid_memory_design(
        model_row, job, fixed_lora=fixed_lora, capacity_bytes=capacity_bytes
    )
    feats = build_dense_hybrid_features_v2(
        job, model_row, fixed_lora=fixed_lora, capacity_bytes=capacity_bytes
    )
    design["zero3_param_workspace"] = float(
        feats["memory"]["workspace_candidates"][ZERO3_PARAM_WORKSPACE_KEY]
    )
    return design


def hybrid_memory_design_v3(
    model_row: Mapping[str, Any],
    job: Mapping[str, Any],
    fixed_lora: Mapping[str, Any],
    capacity_bytes: int,
) -> dict[str, float]:
    """Stage-3 design: adds the GC-aware saved-activation split.

    When gradient checkpointing is OFF the full per-layer activation is saved;
    those bytes move into ``saved_*_gc0`` columns so the GC-off regime gets its
    own coefficient (stage-2 shared one coefficient and systematically
    under-predicted GC-off rows on novel sources).
    """

    design = hybrid_memory_design_v2(
        model_row, job, fixed_lora=fixed_lora, capacity_bytes=capacity_bytes
    )
    feats = build_dense_hybrid_features_v2(
        job, model_row, fixed_lora=fixed_lora, capacity_bytes=capacity_bytes
    )
    act = feats["memory"]["activation_components"]
    gc_off = bool(job.get("gc") if job.get("gc") is not None else job.get("gradient_checkpointing")) is False
    design["saved_full_gc0"] = float(act["saved_full_attention_activations_bytes"]) if gc_off else 0.0
    design["saved_linear_gc0"] = float(act["saved_linear_attention_activations_bytes"]) if gc_off else 0.0
    if gc_off:
        design["saved_full"] = 0.0
        design["saved_linear"] = 0.0
    return design


def predict_memory_center_v2(
    model_row: Mapping[str, Any],
    job: Mapping[str, Any],
    coefficients_by_name: Mapping[str, float],
    fixed_lora: Mapping[str, Any],
    capacity_bytes: int,
) -> dict[str, Any]:
    design = hybrid_memory_design_v2(
        model_row, job, fixed_lora=fixed_lora, capacity_bytes=capacity_bytes
    )
    center = sum(float(coefficients_by_name[k]) * v for k, v in design.items())
    return {"center_bytes": max(center, 0.0), "design_bytes": design}


def predict_memory_center_v3(
    model_row: Mapping[str, Any],
    job: Mapping[str, Any],
    coefficients_by_name: Mapping[str, float],
    fixed_lora: Mapping[str, Any],
    capacity_bytes: int,
) -> dict[str, Any]:
    design = hybrid_memory_design_v3(
        model_row, job, fixed_lora=fixed_lora, capacity_bytes=capacity_bytes
    )
    center = sum(float(coefficients_by_name[k]) * v for k, v in design.items())
    return {"center_bytes": max(center, 0.0), "design_bytes": design}
