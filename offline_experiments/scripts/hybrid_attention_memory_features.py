#!/usr/bin/env python3
"""Architecture-aware dense Qwen memory/compute features.

The module deliberately separates quantities that follow from the checkpoint
configuration from runtime liveness assumptions.  It supports dense Qwen full
attention and dense Qwen3.5-style hybrid attention.  MoE checkpoints are
rejected: expert routing needs a different state/activation model.

This is a feature basis, not a calibrated peak-memory predictor.  In
particular, saved-activation and operator-workspace terms are structural
proxies whose coefficients must be learned from successful measurements while
OOM observations remain right-censored lower bounds.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import read_json, sha256_file, sha256_json

SCHEMA = "sft_hybrid_attention_dense_features/v1"
ARCHITECTURE_SCHEMA = "sft_dense_architecture_signature/v1"

PARAMETER_BYTES = 2.0
GRADIENT_BYTES = 2.0
OPTIMIZER_BYTES = 12.0
LOGITS_BYTES = 4.0
DEFAULT_REDUCE_BUCKET_ELEMENTS = 500_000_000
DEFAULT_ALL_GATHER_BUCKET_ELEMENTS = 500_000_000
DEFAULT_MAX_LIVE_ELEMENTS = 1_000_000_000
DEFAULT_MAX_REUSE_ELEMENTS = 1_000_000_000
ACTIVATION_LIVENESS_BOUNDS = (0.25, 4.0)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _zero_stage(value: Any) -> int:
    normalized = str(value or "none").strip().lower()
    mapping = {"none": 0, "zero0": 0, "zero1": 1, "zero2": 2, "zero3": 3}
    if normalized not in mapping:
        raise ValueError(f"unsupported ZeRO stage {value!r}")
    return mapping[normalized]


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    if isinstance(nested, dict) and nested.get("hidden_size") is not None:
        return nested
    return config


def _state_dtype_bytes(value: Any) -> int:
    normalized = str(value or "float32").lower()
    if normalized in {"float64", "fp64", "torch.float64"}:
        return 8
    if normalized in {
        "float32",
        "fp32",
        "torch.float32",
        "int32",
        "torch.int32",
    }:
        return 4
    if normalized in {
        "float16",
        "fp16",
        "bfloat16",
        "bf16",
        "torch.float16",
        "torch.bfloat16",
    }:
        return 2
    raise ValueError(f"unsupported linear-attention state dtype {value!r}")


def _layer_runs(layer_types: list[str]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for layer_type in layer_types:
        if not runs or runs[-1]["type"] != layer_type:
            runs.append({"type": layer_type, "count": 1})
        else:
            runs[-1]["count"] += 1
    return runs


def architecture_signature(
    model: dict[str, Any], *, config_path: Path | None = None
) -> dict[str, Any]:
    """Return an immutable dense-language architecture signature.

    ``model`` is an inventory/job model row containing at least a local path.
    The real checkpoint ``config.json`` is authoritative; flattened inventory
    geometry is intentionally not enough for hybrid-attention models.
    """

    if config_path is None:
        model_path_value = model.get("path") or model.get("model_path")
        if not model_path_value:
            raise ValueError("model path is required for architecture signature")
        config_path = Path(str(model_path_value)) / "config.json"
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = read_json(config_path)
    text = _text_config(config)

    # Dense-vs-MoE is a hard routing boundary rather than a regression flag.
    expert_fields = {
        "num_experts": text.get("num_experts"),
        "num_experts_per_tok": text.get("num_experts_per_tok"),
        "moe_intermediate_size": text.get("moe_intermediate_size"),
        "decoder_sparse_step": text.get("decoder_sparse_step"),
    }
    if any(value not in (None, 0, "0") for value in expert_fields.values()):
        raise ValueError(
            "dense hybrid feature basis does not accept MoE checkpoints: "
            f"{expert_fields}"
        )

    hidden = _positive_int(text.get("hidden_size"), "hidden_size")
    intermediate = _positive_int(
        text.get("intermediate_size"), "intermediate_size"
    )
    layers = _positive_int(text.get("num_hidden_layers"), "num_hidden_layers")
    attention_heads = _positive_int(
        text.get("num_attention_heads"), "num_attention_heads"
    )
    kv_heads = _positive_int(
        text.get("num_key_value_heads") or attention_heads,
        "num_key_value_heads",
    )
    explicit_head_dim = text.get("head_dim")
    if explicit_head_dim is None:
        if hidden % attention_heads:
            raise ValueError("hidden_size is not divisible by num_attention_heads")
        head_dim = hidden // attention_heads
    else:
        head_dim = _positive_int(explicit_head_dim, "head_dim")

    raw_layer_types = text.get("layer_types")
    if raw_layer_types is None:
        layer_types = ["full_attention"] * layers
        layer_pattern_source = "implicit_all_full_attention"
    else:
        if not isinstance(raw_layer_types, list):
            raise ValueError("layer_types must be a list")
        layer_types = [str(value) for value in raw_layer_types]
        layer_pattern_source = "explicit_config"
    if len(layer_types) != layers:
        raise ValueError(
            f"layer_types length {len(layer_types)} != num_hidden_layers {layers}"
        )
    unsupported = sorted(set(layer_types) - {"full_attention", "linear_attention"})
    if unsupported:
        raise ValueError(f"unsupported dense layer types: {unsupported}")
    counts = Counter(layer_types)

    full_attention_layers = counts["full_attention"]
    linear_attention_layers = counts["linear_attention"]
    has_linear = linear_attention_layers > 0
    q_width = attention_heads * head_dim
    kv_width = kv_heads * head_dim
    attn_output_gate = bool(text.get("attn_output_gate", False))
    q_projection_width = q_width * (2 if attn_output_gate else 1)

    linear_num_key_heads = int(text.get("linear_num_key_heads") or 0)
    linear_num_value_heads = int(text.get("linear_num_value_heads") or 0)
    linear_key_head_dim = int(text.get("linear_key_head_dim") or 0)
    linear_value_head_dim = int(text.get("linear_value_head_dim") or 0)
    linear_conv_kernel_dim = int(text.get("linear_conv_kernel_dim") or 0)
    if has_linear and min(
        linear_num_key_heads,
        linear_num_value_heads,
        linear_key_head_dim,
        linear_value_head_dim,
        linear_conv_kernel_dim,
    ) <= 0:
        raise ValueError("hybrid config is missing linear-attention geometry")
    linear_key_width = linear_num_key_heads * linear_key_head_dim
    linear_value_width = linear_num_value_heads * linear_value_head_dim
    # Qwen3.5's fused projection carries two key streams and one value stream.
    linear_qkv_width = 2 * linear_key_width + linear_value_width
    state_dtype = str(text.get("mamba_ssm_dtype") or "float32")
    state_dtype_bytes = _state_dtype_bytes(state_dtype) if has_linear else 0

    runs = _layer_runs(layer_types)
    signature_core = {
        "schema": ARCHITECTURE_SCHEMA,
        "model_type": str(config.get("model_type") or text.get("model_type") or ""),
        "text_model_type": str(text.get("model_type") or config.get("model_type") or ""),
        "architecture_route": (
            "dense_hybrid_attention" if has_linear else "dense_full_attention"
        ),
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "vocab_size": _positive_int(text.get("vocab_size"), "vocab_size"),
        "num_hidden_layers": layers,
        "num_full_attention_layers": full_attention_layers,
        "num_linear_attention_layers": linear_attention_layers,
        "full_attention_interval": (
            int(text["full_attention_interval"])
            if text.get("full_attention_interval") is not None
            else None
        ),
        "layer_pattern_source": layer_pattern_source,
        "layer_pattern_sha256": hashlib.sha256(
            json.dumps(layer_types, separators=(",", ":")).encode()
        ).hexdigest(),
        "layer_runs": runs,
        "max_consecutive_full_attention_layers": max(
            (row["count"] for row in runs if row["type"] == "full_attention"),
            default=0,
        ),
        "max_consecutive_linear_attention_layers": max(
            (row["count"] for row in runs if row["type"] == "linear_attention"),
            default=0,
        ),
        "num_attention_heads": attention_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "query_width": q_width,
        "query_projection_width": q_projection_width,
        "kv_width": kv_width,
        "attention_output_gate": attn_output_gate,
        "linear_num_key_heads": linear_num_key_heads,
        "linear_num_value_heads": linear_num_value_heads,
        "linear_key_head_dim": linear_key_head_dim,
        "linear_value_head_dim": linear_value_head_dim,
        "linear_key_width": linear_key_width,
        "linear_value_width": linear_value_width,
        "linear_qkv_width": linear_qkv_width,
        "linear_conv_kernel_dim": linear_conv_kernel_dim,
        "linear_state_dtype": state_dtype if has_linear else None,
        "linear_state_dtype_bytes": state_dtype_bytes,
    }
    return {
        **signature_core,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "signature_sha256": sha256_json(signature_core),
    }


def lora_adapter_parameter_elements(
    signature: dict[str, Any], rank: int
) -> dict[str, int]:
    """Count language-tower LoRA elements for target=all linear modules.

    The count follows the actual Qwen3/Qwen3.5 projection shapes.  It excludes
    the Qwen3.5 depthwise conv because it is not a linear LoRA target.
    """

    rank = _positive_int(rank, "LoRA rank")
    hidden = int(signature["hidden_size"])
    intermediate = int(signature["intermediate_size"])
    q_width = int(signature["query_width"])
    q_projection_width = int(signature["query_projection_width"])
    kv_width = int(signature["kv_width"])
    full_per_layer = rank * (
        (hidden + q_projection_width)
        + 2 * (hidden + kv_width)
        + (q_width + hidden)
        + 3 * (hidden + intermediate)
    )

    linear_qkv_width = int(signature["linear_qkv_width"])
    linear_value_width = int(signature["linear_value_width"])
    value_heads = int(signature["linear_num_value_heads"])
    linear_per_layer = 0
    if int(signature["num_linear_attention_layers"]):
        linear_per_layer = rank * (
            (hidden + linear_qkv_width)
            + (hidden + linear_value_width)
            + 2 * (hidden + value_heads)
            + (linear_value_width + hidden)
            + 3 * (hidden + intermediate)
        )
    full_total = int(signature["num_full_attention_layers"]) * full_per_layer
    linear_total = int(signature["num_linear_attention_layers"]) * linear_per_layer
    return {
        "rank": rank,
        "full_attention_per_layer_elements": full_per_layer,
        "linear_attention_per_layer_elements": linear_per_layer,
        "full_attention_total_elements": full_total,
        "linear_attention_total_elements": linear_total,
        "total_elements": full_total + linear_total,
    }


def _model_geometry(
    job: dict[str, Any],
    model: dict[str, Any],
    signature: dict[str, Any],
    fixed_lora: dict[str, Any],
) -> dict[str, Any]:
    base_parameters = _positive_int(
        job.get("model_parameters") or model.get("actual_parameters"),
        "base parameters",
    )
    if model.get("actual_parameters") is not None:
        inventory_parameters = _positive_int(
            model.get("actual_parameters"), "inventory base parameters"
        )
        if inventory_parameters != base_parameters:
            raise ValueError(
                f"job/inventory parameter mismatch: {base_parameters} != "
                f"{inventory_parameters}"
            )
    mode = str(job.get("train_type") or "").lower()
    if mode not in {"full", "lora"}:
        raise ValueError(f"unsupported training mode {mode!r}")
    rank = _positive_int(fixed_lora.get("rank"), "LoRA rank")
    target = str(fixed_lora.get("target") or "")
    if target != "all":
        raise ValueError("dense feature basis currently requires LoRA target=all")
    adapter_breakdown = lora_adapter_parameter_elements(signature, rank)
    adapter_parameters = (
        adapter_breakdown["total_elements"] if mode == "lora" else 0
    )
    loaded = base_parameters + adapter_parameters
    trainable = adapter_parameters if mode == "lora" else base_parameters

    hidden = int(signature["hidden_size"])
    intermediate = int(signature["intermediate_size"])
    q_width = int(signature["query_width"])
    q_projection_width = int(signature["query_projection_width"])
    kv_width = int(signature["kv_width"])
    linear_qkv_width = int(signature["linear_qkv_width"])
    linear_value_width = int(signature["linear_value_width"])
    value_heads = int(signature["linear_num_value_heads"])

    full_linear_weights = (
        hidden * q_projection_width
        + 2 * hidden * kv_width
        + q_width * hidden
        + 3 * hidden * intermediate
    )
    linear_linear_weights = 0
    if int(signature["num_linear_attention_layers"]):
        linear_linear_weights = (
            hidden * linear_qkv_width
            + hidden * linear_value_width
            + 2 * hidden * value_heads
            + linear_value_width * hidden
            + 3 * hidden * intermediate
        )
    projection_weight_elements = (
        int(signature["num_full_attention_layers"]) * full_linear_weights
        + int(signature["num_linear_attention_layers"]) * linear_linear_weights
        + int(signature["vocab_size"]) * hidden
    )
    max_layer = max(full_linear_weights, linear_linear_weights)
    max_module = max(max_layer, int(signature["vocab_size"]) * hidden)
    persistent = min(
        loaded,
        int(signature["num_hidden_layers"])
        * (2 * hidden + 2 * int(signature["head_dim"]))
        + hidden,
    )
    return {
        "base_parameters": base_parameters,
        "adapter_parameters": adapter_parameters,
        "adapter_parameter_breakdown": adapter_breakdown,
        "loaded_parameters": loaded,
        "trainable_parameters": trainable,
        "frozen_parameters": loaded - trainable,
        "projection_weight_elements_per_pass": projection_weight_elements,
        "full_attention_projection_weight_elements_per_layer": full_linear_weights,
        "linear_attention_projection_weight_elements_per_layer": linear_linear_weights,
        "max_layer_parameter_elements": max_layer,
        "max_module_parameter_elements": max_module,
        "persistent_parameter_elements_structural": persistent,
    }


def build_dense_hybrid_features(
    job: dict[str, Any],
    model: dict[str, Any],
    fixed_lora: dict[str, Any],
    capacity_bytes: int,
) -> dict[str, Any]:
    """Build per-device memory and compute features for one dense job."""

    signature = architecture_signature(model)
    geometry = _model_geometry(job, model, signature, fixed_lora)
    num_gpus = _positive_int(job.get("gpu_count"), "gpu_count")
    micro_batch = _positive_int(job.get("mbs"), "mbs")
    sequence = _positive_int(job.get("cutoff_len"), "cutoff_len")
    zero_stage = _zero_stage(job.get("zero"))
    shards = float(num_gpus)
    loaded = geometry["loaded_parameters"]
    trainable = geometry["trainable_parameters"]

    if zero_stage == 0:
        parameters = loaded * PARAMETER_BYTES
        gradients = trainable * GRADIENT_BYTES
        optimizer = trainable * OPTIMIZER_BYTES
    elif zero_stage == 1:
        parameters = loaded * PARAMETER_BYTES
        gradients = trainable * GRADIENT_BYTES
        optimizer = trainable * OPTIMIZER_BYTES / shards
    elif zero_stage == 2:
        parameters = loaded * PARAMETER_BYTES
        gradients = trainable * GRADIENT_BYTES / shards
        optimizer = trainable * OPTIMIZER_BYTES / shards
    else:
        parameters = loaded * PARAMETER_BYTES / shards
        gradients = trainable * GRADIENT_BYTES / shards
        optimizer = trainable * OPTIMIZER_BYTES / shards

    hidden = int(signature["hidden_size"])
    intermediate = int(signature["intermediate_size"])
    full_layers = int(signature["num_full_attention_layers"])
    linear_layers = int(signature["num_linear_attention_layers"])
    q_width = int(signature["query_width"])
    q_projection_width = int(signature["query_projection_width"])
    kv_width = int(signature["kv_width"])
    full_per_token = (
        4 * hidden + q_projection_width + q_width + 2 * kv_width + 3 * intermediate
    )
    linear_qkv_width = int(signature["linear_qkv_width"])
    linear_value_width = int(signature["linear_value_width"])
    value_heads = int(signature["linear_num_value_heads"])
    linear_per_token = 0
    if linear_layers:
        # Structural proxy: fused qkv, gate/value branches, A/B state controls,
        # output and the gated MLP.  Runtime coefficients absorb kernel liveness.
        linear_per_token = (
            2 * hidden
            + 2 * linear_qkv_width
            + 2 * linear_value_width
            + 2 * value_heads
            + 3 * intermediate
        )

    checkpointing = bool(
        job.get("gc")
        if job.get("gc") is not None
        else job.get("gradient_checkpointing")
    )
    if checkpointing:
        saved_full = micro_batch * sequence * full_layers * hidden * PARAMETER_BYTES
        saved_linear = (
            micro_batch * sequence * linear_layers * hidden * PARAMETER_BYTES
        )
        recompute = (
            micro_batch
            * sequence
            * max(
                full_per_token if full_layers else 0,
                linear_per_token if linear_layers else 0,
            )
            * PARAMETER_BYTES
        )
    else:
        saved_full = (
            micro_batch
            * sequence
            * full_layers
            * full_per_token
            * PARAMETER_BYTES
        )
        saved_linear = (
            micro_batch
            * sequence
            * linear_layers
            * linear_per_token
            * PARAMETER_BYTES
        )
        recompute = 0.0
    saved = saved_full + saved_linear

    full_workspace = 0.0
    if full_layers:
        full_workspace = (
            micro_batch
            * sequence
            * (hidden + q_width + 2 * kv_width)
            * PARAMETER_BYTES
            + micro_batch
            * int(signature["num_attention_heads"])
            * sequence
            * 4
        )
    linear_workspace = 0.0
    linear_state = 0.0
    if linear_layers:
        linear_workspace = (
            micro_batch
            * sequence
            * (linear_qkv_width + linear_value_width)
            * PARAMETER_BYTES
        )
        linear_state = (
            micro_batch
            * value_heads
            * int(signature["linear_key_head_dim"])
            * int(signature["linear_value_head_dim"])
            * int(signature["linear_state_dtype_bytes"])
        )
        linear_workspace += linear_state
    attention_workspace = max(full_workspace, linear_workspace)
    logits = micro_batch * sequence * int(signature["vocab_size"]) * LOGITS_BYTES

    reduce_elements = min(trainable, DEFAULT_REDUCE_BUCKET_ELEMENTS)
    reduce_bytes = reduce_elements * GRADIENT_BYTES
    zero_workspace = 0.0
    stage3_live = 0.0
    if zero_stage == 1:
        zero_workspace = reduce_bytes
    elif zero_stage == 2:
        zero_workspace = max(
            reduce_bytes,
            min(trainable, DEFAULT_ALL_GATHER_BUCKET_ELEMENTS) * PARAMETER_BYTES,
        )
    elif zero_stage == 3:
        zero_workspace = reduce_bytes
        retained = min(
            loaded, DEFAULT_MAX_LIVE_ELEMENTS + DEFAULT_MAX_REUSE_ELEMENTS
        )
        live_elements = min(
            loaded,
            retained
            + geometry["max_module_parameter_elements"]
            + geometry["persistent_parameter_elements_structural"],
        )
        if num_gpus > 1:
            stage3_live = live_elements * PARAMETER_BYTES * (1 - 1 / shards)

    state_components = {
        "parameters_bytes": float(parameters),
        "gradients_bytes": float(gradients),
        "optimizer_bytes": float(optimizer),
        "stage3_live_parameters_bytes": float(stage3_live),
    }
    activation_components = {
        "saved_full_attention_activations_bytes": float(saved_full),
        "saved_linear_attention_activations_bytes": float(saved_linear),
        "saved_activations_bytes": float(saved),
        "recompute_workspace_bytes": float(recompute),
    }
    workspace_candidates = {
        "full_attention_workspace_bytes": float(full_workspace),
        "linear_attention_workspace_bytes": float(linear_workspace),
        "linear_recurrent_state_bytes": float(linear_state),
        "logits_workspace_bytes": float(logits),
        "zero_collective_workspace_bytes": float(zero_workspace),
    }
    state_bytes = sum(state_components.values())
    common_saved_bytes = saved + recompute
    peak_workspace = max(
        attention_workspace,
        logits,
        zero_workspace,
    )
    reference = state_bytes + common_saved_bytes + peak_workspace
    structural_activation = common_saved_bytes + attention_workspace
    non_activation = reference - structural_activation

    # Separate compute features retain the different S scaling: full attention
    # has a quadratic mixing term while Qwen3.5 linear recurrence is linear.
    tokens_per_microbatch = micro_batch * sequence
    projection_macs = (
        tokens_per_microbatch
        * int(geometry["projection_weight_elements_per_pass"])
    )
    full_attention_mixing_macs = (
        micro_batch * full_layers * q_width * sequence * sequence
    )
    linear_recurrence_macs = (
        micro_batch
        * linear_layers
        * sequence
        * int(signature["linear_num_value_heads"])
        * int(signature["linear_key_head_dim"])
        * int(signature["linear_value_head_dim"])
    )

    return {
        "schema": SCHEMA,
        "architecture_signature": signature,
        "geometry": geometry,
        "memory": {
            "state_components": state_components,
            "activation_components": activation_components,
            "workspace_candidates": workspace_candidates,
            "state_bytes": float(state_bytes),
            "common_saved_and_recompute_bytes": float(common_saved_bytes),
            "peak_workspace_bytes": float(peak_workspace),
            "structural_activation_bytes": float(structural_activation),
            "analytic_non_activation_bytes": float(non_activation),
            "analytic_reference_bytes": float(reference),
            "analytic_lower_bound_bytes": float(
                non_activation
                + ACTIVATION_LIVENESS_BOUNDS[0] * structural_activation
            ),
            "safe_limit_bytes": float(0.95 * capacity_bytes),
            "workspace_aggregation": "max_not_sum",
        },
        "compute": {
            "tokens_per_microbatch": tokens_per_microbatch,
            "projection_macs_per_forward": int(projection_macs),
            "full_attention_mixing_macs_per_forward": int(
                full_attention_mixing_macs
            ),
            "linear_recurrence_macs_per_forward": int(linear_recurrence_macs),
        },
        "assumptions": {
            "feature_status": "structural_basis_not_calibrated_predictor",
            "saved_activation_terms": "engineering_proxy_requires_architecture_specific_liveness_fit",
            "operator_workspace_terms": "engineering_proxy_runtime_uses_peak_candidate",
            "linear_recurrent_state": "persistent_per_microbatch_fp_state_proxy",
            "oom_semantics": "right_censored_lower_bound_never_exact_peak",
        },
    }
