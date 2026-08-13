"""Structural per-token matmul work, derived from real projection widths.

The historical ``linear_applications_per_pass`` in :mod:`h800_theory_basis`
assumes every attention layer spends ``2 * hidden * hidden`` on the q/o
projections and ``2 * hidden * kv_width`` on k/v.  That identity only holds
when ``num_attention_heads * head_dim == hidden_size`` and the layer is a
plain softmax-attention block.  Neither is guaranteed:

* Qwen3-0.6B / Qwen3-4B / Qwen3-32B set ``head_dim`` independently of
  ``hidden_size``, so ``q_proj`` is ``[heads * head_dim, hidden]`` and
  ``o_proj`` is ``[hidden, heads * head_dim]``.
* Qwen3.5 / Qwen3.6 additionally set ``attn_output_gate``, which doubles the
  ``q_proj`` output width, and replace most layers with a gated delta-net
  (``linear_attention``) whose projections have no counterpart in the
  historical formula at all.

This module recomputes the same quantity from the geometry that the modelling
code actually instantiates (verified against ``Qwen3_5Attention.__init__`` and
``Qwen3_5GatedDeltaNet.__init__`` in transformers 5.3.0), and is validated
against real checkpoint tensor shapes by
``tests/test_structural_linear_work.py``.

Only true per-token GEMMs are counted.  ``conv1d`` is depthwise (``groups ==
conv_dim``), and ``A_log`` / ``dt_bias`` are per-head vectors, so none of them
contribute a matmul term; they are reported separately as
``depthwise_elements_per_token`` for callers that want to model their launch
and bandwidth cost.

Nothing here predicts step time.  It replaces one analytic input to the
throughput basis and must not be read as a calibrated efficiency model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "FULL_ATTENTION",
    "LINEAR_ATTENTION",
    "layer_type_counts",
    "structural_linear_work",
]

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _geometry(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the sub-config carrying the text geometry.

    Mirrors ``inventory_models._geometry_config``: multimodal checkpoints nest
    the language geometry under ``text_config``.
    """
    if config.get("hidden_size") is not None:
        return config
    text_config = config.get("text_config")
    if isinstance(text_config, Mapping) and text_config.get("hidden_size") is not None:
        return text_config
    return config


def layer_type_counts(config: Mapping[str, Any]) -> dict[str, int]:
    """Count decoder layers by attention kind.

    ``layer_types`` is authoritative when present.  Absent it, the checkpoint
    is a uniform softmax-attention stack -- we do not infer a hybrid schedule
    from ``full_attention_interval`` alone, because a config carrying that key
    without ``layer_types`` has not been observed and would be a guess.
    """
    geom = _geometry(config)
    layers = _positive_int(geom.get("num_hidden_layers"), "num_hidden_layers")
    declared = geom.get("layer_types")
    if declared is None:
        return {FULL_ATTENTION: layers, LINEAR_ATTENTION: 0}
    if not isinstance(declared, (list, tuple)):
        raise ValueError(f"layer_types must be a sequence, got {declared!r}")
    if len(declared) != layers:
        raise ValueError(
            f"layer_types has {len(declared)} entries but num_hidden_layers is {layers}"
        )
    counts = {FULL_ATTENTION: 0, LINEAR_ATTENTION: 0}
    for entry in declared:
        if entry not in counts:
            raise ValueError(f"Unsupported layer type {entry!r}")
        counts[entry] += 1
    return counts


def _full_attention_elements(geom: Mapping[str, Any]) -> int:
    """Per-token matmul elements of one softmax-attention block.

    Widths follow ``Qwen3_5Attention.__init__``: q_proj gains a factor of two
    when ``attn_output_gate`` is set, and o_proj consumes ``heads * head_dim``
    rather than ``hidden_size``.
    """
    hidden = _positive_int(geom.get("hidden_size"), "hidden_size")
    heads = _positive_int(geom.get("num_attention_heads"), "num_attention_heads")
    kv_heads = _positive_int(
        geom.get("num_key_value_heads") or heads, "num_key_value_heads"
    )
    explicit_head_dim = geom.get("head_dim")
    if explicit_head_dim is not None:
        head_dim = _positive_int(explicit_head_dim, "head_dim")
    else:
        if hidden % heads:
            raise ValueError("hidden size is not divisible by attention heads")
        head_dim = hidden // heads

    query_width = heads * head_dim
    if geom.get("attn_output_gate"):
        query_width *= 2
    kv_width = kv_heads * head_dim
    return hidden * (query_width + 2 * kv_width + heads * head_dim)


def _linear_attention_elements(geom: Mapping[str, Any]) -> tuple[int, int]:
    """Per-token (matmul, depthwise) elements of one gated delta-net block.

    Widths follow ``Qwen3_5GatedDeltaNet.__init__``.  ``in_proj_a`` and
    ``in_proj_b`` are ``hidden -> num_v_heads`` and stay in the matmul term;
    the depthwise ``conv1d`` over ``conv_dim`` channels does not.
    """
    hidden = _positive_int(geom.get("hidden_size"), "hidden_size")
    num_k_heads = _positive_int(
        geom.get("linear_num_key_heads"), "linear_num_key_heads"
    )
    num_v_heads = _positive_int(
        geom.get("linear_num_value_heads"), "linear_num_value_heads"
    )
    head_k_dim = _positive_int(
        geom.get("linear_key_head_dim"), "linear_key_head_dim"
    )
    head_v_dim = _positive_int(
        geom.get("linear_value_head_dim"), "linear_value_head_dim"
    )
    conv_kernel = _positive_int(
        geom.get("linear_conv_kernel_dim"), "linear_conv_kernel_dim"
    )

    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = 2 * key_dim + value_dim

    matmul = hidden * (
        conv_dim          # in_proj_qkv
        + value_dim       # in_proj_z
        + 2 * num_v_heads  # in_proj_a, in_proj_b
        + value_dim       # out_proj
    )
    return matmul, conv_dim * conv_kernel


def _mlp_elements(geom: Mapping[str, Any]) -> int:
    hidden = _positive_int(geom.get("hidden_size"), "hidden_size")
    intermediate = _positive_int(
        geom.get("intermediate_size"), "intermediate_size"
    )
    return 3 * hidden * intermediate


def structural_linear_work(config: Mapping[str, Any]) -> dict[str, Any]:
    """Per-token matmul elements for one forward pass over the text stack.

    ``linear_applications_per_pass`` is the drop-in replacement for the
    historical geometry field.  The remaining keys are diagnostics: they let a
    caller see which layer kind dominates and compare against the historical
    formula without recomputing it.

    The language-model head is counted as ``vocab_size * hidden_size``, matching
    the historical convention.  A tied embedding still runs the head GEMM, so
    ``tie_word_embeddings`` does not change this term.
    """
    geom = _geometry(config)
    counts = layer_type_counts(config)
    hidden = _positive_int(geom.get("hidden_size"), "hidden_size")
    vocab = _positive_int(geom.get("vocab_size"), "vocab_size")

    mlp = _mlp_elements(geom)
    full_attn = _full_attention_elements(geom)
    full_layer = full_attn + mlp

    linear_layer = 0
    depthwise = 0
    if counts[LINEAR_ATTENTION]:
        linear_attn, conv = _linear_attention_elements(geom)
        linear_layer = linear_attn + mlp
        depthwise = conv * counts[LINEAR_ATTENTION]

    layer_total = (
        counts[FULL_ATTENTION] * full_layer + counts[LINEAR_ATTENTION] * linear_layer
    )
    head = vocab * hidden

    return {
        "linear_applications_per_pass": layer_total + head,
        "layer_elements_per_token": layer_total,
        "head_elements_per_token": head,
        "full_attention_layers": counts[FULL_ATTENTION],
        "linear_attention_layers": counts[LINEAR_ATTENTION],
        "full_attention_layer_elements_per_token": full_layer,
        "linear_attention_layer_elements_per_token": linear_layer,
        "depthwise_elements_per_token": depthwise,
        "is_hybrid_attention": counts[LINEAR_ATTENTION] > 0,
        "full_attention_layer_share": counts[FULL_ATTENTION]
        / (counts[FULL_ATTENTION] + counts[LINEAR_ATTENTION]),
        "attn_output_gate": bool(geom.get("attn_output_gate")),
    }
