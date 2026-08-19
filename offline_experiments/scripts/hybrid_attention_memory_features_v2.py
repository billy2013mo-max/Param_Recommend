#!/usr/bin/env python3
"""Stage-2 dense hybrid-attention memory features (adds a ZeRO-3 param workspace).

Stage-1 (``hybrid_attention_memory_features.py``) systematically under-predicts
peak reserved memory for large hybrid models on multi-GPU ZeRO-3 (qwen3p5_9b
under-estimated 17.9%, qwen3_6_27b 14.9% on the source-disjoint prospective
tasks).  The root cause is that ZeRO-3's per-device all-gather of the full
sharded parameters plus prefetch double-buffering is never modelled: it leaks
into ``stage3_live`` (a heuristic-bucket term folded into ``state_bytes`` whose
coefficient is pinned to [0.8, 1.25]) and is therefore squeezed too small.

V2 adds one explicit, freely-fit workspace column ``zero3_param_workspace_bytes``
so the ZeRO-3 constant/all-gather bias can be separated from ``state`` and the
saved-activation terms.  Everything else is reused verbatim from the stage-1
feature basis; the new column is a *structural proxy*, its coefficient is
learned from measurements.
"""

from __future__ import annotations

from typing import Any

from hybrid_attention_memory_features import (
    DEFAULT_ALL_GATHER_BUCKET_ELEMENTS,
    PARAMETER_BYTES,
    build_dense_hybrid_features,
)

SCHEMA = "sft_hybrid_attention_dense_features/v2"

# New coefficient introduced on top of the stage-1 basis.
ZERO3_PARAM_WORKSPACE_KEY = "zero3_param_workspace_bytes"

_ZERO_STAGE = {"none": 0, "zero0": 0, "zero1": 1, "zero2": 2, "zero3": 3}


def _zero_stage(value: Any) -> int:
    normalized = str(value or "none").strip().lower()
    if normalized not in _ZERO_STAGE:
        raise ValueError(f"unsupported ZeRO stage {value!r}")
    return _ZERO_STAGE[normalized]


def zero3_param_workspace_bytes(
    *,
    loaded_parameters: float,
    max_module_parameter_elements: float,
    gpu_count: int,
    zero_stage: int,
) -> float:
    """Structural proxy for the ZeRO-3 per-device parameter working set.

    Under ZeRO-3 with more than one GPU, the forward/backward pass all-gathers
    the sharded parameters back to full precision one bucket at a time and keeps
    a prefetch double-buffer live.  We model the active working set as the
    all-gather bucket plus the largest single module (double-buffer), capped by
    the full loaded parameter count.  A single GPU (no sharding) and non-ZeRO-3
    stages contribute zero; the regression coefficient absorbs the exact
    double-buffer multiple and bucket-fill fraction.
    """

    if zero_stage != 3 or int(gpu_count) <= 1:
        return 0.0
    workspace_elements = min(
        float(loaded_parameters),
        float(DEFAULT_ALL_GATHER_BUCKET_ELEMENTS)
        + float(max_module_parameter_elements),
    )
    return workspace_elements * PARAMETER_BYTES


def zero3_param_workspace_from_basis(
    feature_basis: dict[str, Any],
    configuration: dict[str, Any],
) -> float:
    """Derive the V2 column from an already-recorded stage-1 feature basis."""

    geometry = feature_basis["geometry"]
    return zero3_param_workspace_bytes(
        loaded_parameters=geometry["loaded_parameters"],
        max_module_parameter_elements=geometry["max_module_parameter_elements"],
        gpu_count=int(configuration["gpu_count"]),
        zero_stage=_zero_stage(configuration.get("zero")),
    )


def build_dense_hybrid_features_v2(
    job: dict[str, Any],
    model: dict[str, Any],
    fixed_lora: dict[str, Any],
    capacity_bytes: int,
) -> dict[str, Any]:
    """Build the stage-1 basis and append the ZeRO-3 param-workspace column."""

    basis = build_dense_hybrid_features(job, model, fixed_lora, capacity_bytes)
    extra = zero3_param_workspace_bytes(
        loaded_parameters=basis["geometry"]["loaded_parameters"],
        max_module_parameter_elements=basis["geometry"]["max_module_parameter_elements"],
        gpu_count=int(job["gpu_count"]),
        zero_stage=_zero_stage(job.get("zero")),
    )
    basis["memory"]["workspace_candidates"][ZERO3_PARAM_WORKSPACE_KEY] = float(extra)
    basis["schema"] = SCHEMA
    basis["assumptions"]["zero3_param_workspace"] = (
        "structural_proxy_all_gather_bucket_plus_max_module_double_buffer"
    )
    return basis
