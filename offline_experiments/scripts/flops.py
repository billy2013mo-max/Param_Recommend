#!/usr/bin/env python3
"""Structure-aware useful training FLOP estimates for dense Qwen SFT."""

from __future__ import annotations

from typing import Any


def forward_components(model: dict[str, Any], linear_tokens: int, attention_pairs: int, lora_rank: int) -> dict[str, float]:
    hidden = int(model["hidden_size"])
    intermediate = int(model["intermediate_size"])
    layers = int(model["num_hidden_layers"])
    heads = int(model["num_attention_heads"])
    kv_heads = int(model["num_key_value_heads"])
    vocab = int(model["vocab_size"])
    head_dim = hidden // heads
    query_width = heads * head_dim
    kv_width = kv_heads * head_dim

    # Matrix multiply FLOPs use 2*m*n*k. Qwen MLP is gate/up/down (SwiGLU).
    attention_projection_per_token = 2 * hidden * (query_width + 2 * kv_width) + 2 * query_width * hidden
    mlp_projection_per_token = 6 * hidden * intermediate
    layer_linear_forward = layers * (attention_projection_per_token + mlp_projection_per_token) * linear_tokens
    # Two causal attention matmuls (QK^T and P@V), each over roughly L^2/2 pairs.
    attention_core_forward = layers * 2 * query_width * attention_pairs
    lm_head_forward = 2 * hidden * vocab * linear_tokens

    # all-linear excludes lm_head in this LLaMA-Factory version.
    lora_in_plus_out_per_layer = 9 * hidden + 2 * kv_width + 3 * intermediate
    lora_forward = layers * 2 * lora_rank * lora_in_plus_out_per_layer * linear_tokens
    return {
        "layer_linear_forward": float(layer_linear_forward),
        "attention_core_forward": float(attention_core_forward),
        "lm_head_forward": float(lm_head_forward),
        "lora_forward": float(lora_forward),
    }


def training_flops(
    model: dict[str, Any],
    train_type: str,
    linear_tokens: int,
    attention_pairs: int,
    lora_rank: int = 32,
) -> dict[str, float]:
    components = forward_components(model, linear_tokens, attention_pairs, lora_rank)
    if train_type == "full":
        useful = 3.0 * (
            components["layer_linear_forward"]
            + components["attention_core_forward"]
            + components["lm_head_forward"]
        )
        formula = "FULL: 3x forward FLOPs (forward + activation/input gradient + weight gradient)"
    elif train_type == "lora":
        useful = (
            2.0 * components["layer_linear_forward"]
            + 3.0 * components["attention_core_forward"]
            + 2.0 * components["lm_head_forward"]
            + 3.0 * components["lora_forward"]
        )
        formula = (
            "LoRA: frozen base linears/lm_head use forward+dX (2x), attention core uses 3x, "
            "trainable LoRA A/B paths use 3x; lm_head is excluded from all-linear targets"
        )
    else:
        raise ValueError(train_type)
    return {"useful_flops": useful, "formula": formula, **components}


def result_flops(model: dict[str, Any], train_type: str, counters: dict[str, int], lora_rank: int = 32) -> dict[str, Any]:
    computed = training_flops(
        model,
        train_type,
        counters["computed_tokens"],
        counters["computed_attention_token_pairs"],
        lora_rank,
    )
    effective = training_flops(
        model,
        train_type,
        counters["effective_tokens"],
        counters["effective_attention_token_pairs"],
        lora_rank,
    )
    return {
        "computed_useful_flops": computed["useful_flops"],
        "effective_useful_flops": effective["useful_flops"],
        "formula": computed["formula"],
        "calibration_status": "analytic_v1; calibrate representative FULL/LoRA points with PyTorch Profiler",
    }
