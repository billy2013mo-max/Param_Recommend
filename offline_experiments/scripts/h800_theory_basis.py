#!/usr/bin/env python3
"""Export an auditable analytic basis for historical H800 observations.

This command performs no fitting and never touches a GPU.  It maps recovered
historical observations onto the same physical quantities used by the CPU-side
DeepSpeed planner: model state, activations, logits, ZeRO workspace/live state,
operator FLOPs, tensor traffic, optimizer traffic, and collective payloads.

The historical archive does not contain a native-v2 runtime mechanism identity
or a planner-trusted operator manifest.  Consequently this artifact is a
``theory_only`` bootstrap input and can never itself be published as a
calibrated planner profile.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from common import ROOT, read_json, sha256_file, sha256_json


SCHEMA = "sft_h800_theory_basis/v1"
RECORD_SCHEMA = "sft_h800_theory_basis_record/v1"
OBSERVATION_SCHEMA = "sft_efficiency_observation/v2"
RECOVERY_SCHEMA = "sft_h800_historical_recovery/v1"

PARAMETER_BYTES = 2.0
GRADIENT_BYTES = 2.0
OPTIMIZER_BYTES = 12.0
COMMUNICATION_BYTES = 2.0
LOGITS_BYTES = 4.0
H800_MEMORY_BANDWIDTH_BYTES_S = 3.35e12
H800_INTRA_NODE_BANDWIDTH_BYTES_S = 900e9
DEFAULT_HBM_EFFICIENCY = 0.60
DEFAULT_COLLECTIVE_EFFICIENCY = 0.70
DEFAULT_COLLECTIVE_LATENCY_SECONDS = 0.00001
DEFAULT_COMMUNICATION_OVERLAP = 0.0
DEFAULT_REDUCE_BUCKET_ELEMENTS = 500_000_000
DEFAULT_ALL_GATHER_BUCKET_ELEMENTS = 500_000_000
DEFAULT_PREFETCH_BUCKET_ELEMENTS = 50_000_000
DEFAULT_MAX_LIVE_ELEMENTS = 1_000_000_000
DEFAULT_MAX_REUSE_ELEMENTS = 1_000_000_000
ACTIVATION_LIVENESS_BOUNDS = (0.25, 4.0)
CALIBRATION_ROUTES = (
    "feasibility",
    "memory_boundary",
    "throughput_primary",
    "throughput_screen_only",
    "profiler",
    "packing_pair",
    "packing_memory_safety",
)
SELECTOR_KEYS = {
    "runtime_cohort_id",
    "dtype",
    "kernel_path",
    "training_mode",
    "zero_stage",
    "gradient_checkpointing",
    "packing",
}


def _read_observations(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
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
            if observation_id in rows:
                raise ValueError(f"Duplicate observation id {observation_id}")
            rows[observation_id] = row
    return rows


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and non-negative") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _ceil_div(value: int, divisor: int) -> int:
    if value <= 0:
        return 0
    if divisor <= 0:
        raise ValueError("collective bucket divisor must be positive")
    return (value + divisor - 1) // divisor


def _zero_stage(value: Any) -> int:
    normalized = str(value or "none").strip().lower()
    mapping = {"none": 0, "zero0": 0, "zero1": 1, "zero2": 2, "zero3": 3}
    if normalized not in mapping:
        raise ValueError(f"Unsupported ZeRO stage {value!r}")
    return mapping[normalized]


def _geom_source(model: dict[str, Any]) -> dict[str, Any]:
    """Return the dict carrying transformer geometry.

    Dense Qwen3 models expose ``hidden_size`` at the top level.  VL and newer
    nested architectures (qwen3_vl, qwen3_5) place the language-tower geometry
    inside ``text_config``; the vision tower is not modeled here.  Prefer an
    explicit inventory value, then fall back to ``text_config``.
    """
    if model.get("hidden_size") is not None:
        return model
    text_config = model.get("text_config")
    if isinstance(text_config, dict) and text_config.get("hidden_size") is not None:
        return text_config
    return model


def _model_geometry(
    job: dict[str, Any], model: dict[str, Any], fixed_lora: dict[str, Any]
) -> dict[str, int]:
    geom = _geom_source(model)
    hidden = _positive_int(geom.get("hidden_size"), "hidden_size")
    intermediate = _positive_int(
        geom.get("intermediate_size"), "intermediate_size"
    )
    layers = _positive_int(geom.get("num_hidden_layers"), "num_hidden_layers")
    attention_heads = _positive_int(
        geom.get("num_attention_heads"), "num_attention_heads"
    )
    kv_heads = _positive_int(
        geom.get("num_key_value_heads") or attention_heads,
        "num_key_value_heads",
    )
    # Honor an explicit head_dim when the config provides one (Qwen3-VL/3.x set
    # head_dim independently of hidden/heads, and hidden need not be divisible
    # by heads).  Dense Qwen3 omits it, where head_dim == hidden // heads.
    explicit_head_dim = geom.get("head_dim")
    if explicit_head_dim is not None:
        head_dim = _positive_int(explicit_head_dim, "head_dim")
    else:
        if hidden % attention_heads:
            raise ValueError("hidden size is not divisible by attention heads")
        head_dim = hidden // attention_heads
    kv_width = kv_heads * head_dim
    vocab = _positive_int(geom.get("vocab_size"), "vocab_size")
    base_parameters = _positive_int(
        job.get("model_parameters") or model.get("actual_parameters"),
        "base parameters",
    )
    inventory_parameters = _positive_int(
        model.get("actual_parameters"), "inventory base parameters"
    )
    if base_parameters != inventory_parameters:
        raise ValueError(
            f"Job/inventory parameter mismatch for {job.get('model_id')}: "
            f"{base_parameters} != {inventory_parameters}"
        )
    mode = str(job.get("train_type") or "").lower()
    if mode not in {"full", "lora"}:
        raise ValueError(f"Unsupported training mode {mode!r}")
    rank = _positive_int(fixed_lora.get("rank"), "LoRA rank")
    target = str(fixed_lora.get("target") or "")
    if target != "all":
        raise ValueError("Historical analytic LoRA basis requires target=all")
    # Projection widths: q_proj is [heads*head_dim, hidden] (doubled when the
    # model gates its attention output, e.g. Qwen3.5) and o_proj is
    # [hidden, heads*head_dim].  The historical hidden x hidden assumption
    # undercounted FLOPs and per-layer parameter elements for every model
    # with heads*head_dim != hidden (Qwen3-0.6B, Qwen3-4B, Qwen3-32B, ...).
    q_width = attention_heads * head_dim
    q_projection_width = (
        q_width * 2 if bool(geom.get("attn_output_gate", False)) else q_width
    )
    adapter_parameters = (
        rank
        * layers
        * (
            (hidden + q_projection_width)  # q_proj adapter
            + (q_width + hidden)  # o_proj adapter
            + 2 * (hidden + kv_width)  # k/v adapters
            + 3 * (hidden + intermediate)  # MLP adapters
        )
        if mode == "lora"
        else 0
    )
    loaded_parameters = base_parameters + adapter_parameters
    trainable_parameters = (
        adapter_parameters if mode == "lora" else base_parameters
    )
    linear_applications = layers * (
        hidden * q_projection_width
        + q_width * hidden
        + 2 * hidden * kv_width
        + 3 * hidden * intermediate
    ) + vocab * hidden
    max_layer = (
        hidden * q_projection_width
        + q_width * hidden
        + 2 * hidden * kv_width
        + 3 * hidden * intermediate
        + 2 * hidden
    )
    max_module = max(max_layer, vocab * hidden)
    # Qwen RMSNorm and q/k norm tensors are the known small persistent tensors.
    # This is historical Qwen structure evidence, not a trusted production
    # manifest and therefore cannot upgrade planner confidence.
    persistent_elements = min(
        loaded_parameters,
        layers * (2 * hidden + 2 * head_dim) + hidden,
    )
    return {
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_layers": layers,
        "num_attention_heads": attention_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "kv_width": kv_width,
        "vocab_size": vocab,
        "base_parameters": base_parameters,
        "adapter_parameters": adapter_parameters,
        "loaded_parameters": loaded_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": loaded_parameters - trainable_parameters,
        "linear_applications_per_pass": linear_applications,
        "max_layer_parameter_elements": max_layer,
        "max_module_parameter_elements": max_module,
        "persistent_parameter_elements_structural": persistent_elements,
        "lora_rank": rank,
    }


def memory_basis(
    job: dict[str, Any], geometry: dict[str, int], capacity_bytes: int
) -> dict[str, Any]:
    num_gpus = _positive_int(job.get("gpu_count"), "gpu_count")
    micro_batch = _positive_int(job.get("mbs"), "mbs")
    sequence = _positive_int(job.get("cutoff_len"), "cutoff_len")
    zero_stage = _zero_stage(job.get("zero"))
    loaded = geometry["loaded_parameters"]
    trainable = geometry["trainable_parameters"]
    shards = float(num_gpus)
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

    hidden = geometry["hidden_size"]
    intermediate = geometry["intermediate_size"]
    layers = geometry["num_layers"]
    kv_width = geometry["kv_width"]
    heads = geometry["num_attention_heads"]
    per_token_layer = 6 * hidden + 2 * kv_width + 3 * intermediate
    checkpointing = bool(job.get("gc"))
    if checkpointing:
        saved = micro_batch * sequence * layers * hidden * PARAMETER_BYTES
        recompute = micro_batch * sequence * per_token_layer * PARAMETER_BYTES
    else:
        saved = (
            micro_batch
            * sequence
            * layers
            * per_token_layer
            * PARAMETER_BYTES
        )
        recompute = 0.0
    attention = (
        micro_batch
        * sequence
        * (2 * hidden + 2 * kv_width)
        * PARAMETER_BYTES
        + micro_batch * heads * sequence * 4
    )
    logits = (
        micro_batch * sequence * geometry["vocab_size"] * LOGITS_BYTES
    )

    reduce_elements = min(trainable, DEFAULT_REDUCE_BUCKET_ELEMENTS)
    reduce_bytes = reduce_elements * GRADIENT_BYTES
    zero_workspace = 0.0
    stage3_live = 0.0
    if zero_stage == 1:
        zero_workspace = reduce_bytes
    elif zero_stage == 2:
        all_gather_bytes = (
            min(trainable, DEFAULT_ALL_GATHER_BUCKET_ELEMENTS) * PARAMETER_BYTES
        )
        zero_workspace = max(reduce_bytes, all_gather_bytes)
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
            stage3_live = (
                live_elements * PARAMETER_BYTES * (1 - 1 / float(num_gpus))
            )

    components = {
        "parameters_bytes": float(parameters),
        "gradients_bytes": float(gradients),
        "optimizer_bytes": float(optimizer),
        "saved_activations_bytes": float(saved),
        "recompute_workspace_bytes": float(recompute),
        "attention_workspace_bytes": float(attention),
        "logits_workspace_bytes": float(logits),
        "zero_collective_workspace_bytes": float(zero_workspace),
        "stage3_live_parameters_bytes": float(stage3_live),
    }
    structural_activation = saved + recompute + attention
    non_activation = sum(components.values()) - structural_activation
    reference = sum(components.values())
    return {
        "components": components,
        "analytic_non_activation_bytes": float(non_activation),
        "structural_activation_bytes": float(structural_activation),
        "analytic_reference_bytes": float(reference),
        "analytic_lower_bound_bytes": float(
            non_activation + ACTIVATION_LIVENESS_BOUNDS[0] * structural_activation
        ),
        "analytic_activation_envelope_bytes": {
            "lower": float(
                ACTIVATION_LIVENESS_BOUNDS[0] * structural_activation
            ),
            "upper": float(
                ACTIVATION_LIVENESS_BOUNDS[1] * structural_activation
            ),
        },
        "coefficient_bounds": {
            "activation_liveness": {
                "lower": ACTIVATION_LIVENESS_BOUNDS[0],
                "upper": ACTIVATION_LIVENESS_BOUNDS[1],
            },
            "runtime_or_allocator_fixed_bytes": {
                "lower": 0.0,
                "upper": float(capacity_bytes),
            },
        },
        "safe_limit_bytes": 0.95 * capacity_bytes,
        "assumptions": {
            "attention_kernel": "fa3_orig",
            "logits_workspace": "conservative_full_logits_because_fused_ce_chunk_shape_is_unbound",
            "zero_auto_resolution": "planner_deepspeed_0.19.2_conservative_defaults",
            "zero_overlap_communication": False,
            "stage3_structure": "historical_qwen_structural_not_planner_trusted",
        },
    }


def _gradient_accumulation(
    job: dict[str, Any], runtime_root: Path
) -> tuple[int, dict[str, Any]]:
    explicit = job.get("gradient_accumulation_steps")
    if explicit is not None:
        return _positive_int(explicit, "gradient_accumulation_steps"), {
            "kind": "canonical_job_configuration"
        }
    job_id = str(job.get("job_id") or "")
    config_path = runtime_root / "configs" / f"{job_id}.yaml"
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"Runtime config {config_path} is not a mapping")
        return (
            _positive_int(
                config.get("gradient_accumulation_steps"),
                "gradient_accumulation_steps",
            ),
            {
                "kind": "attempt_runtime_config",
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
        )
    target = _positive_int(job.get("target_gbs"), "target_gbs")
    denominator = _positive_int(job.get("mbs"), "mbs") * _positive_int(
        job.get("gpu_count"), "gpu_count"
    )
    if bool(job.get("packing")) or target % denominator:
        raise ValueError(f"Cannot reconstruct gradient accumulation for {job_id}")
    return target // denominator, {
        "kind": "derived_from_target_gbs_mbs_and_gpu_count",
        "formula": "target_gbs / (physical_mbs * gpu_count)",
    }


def performance_basis(
    row: dict[str, Any],
    job: dict[str, Any],
    geometry: dict[str, int],
    hardware: dict[str, Any],
    runtime_root: Path,
) -> dict[str, Any] | None:
    measurements = row.get("measurements") or {}
    step_seconds = measurements.get("mean_step_seconds")
    measured_steps = measurements.get("measured_step_count")
    if row.get("outcome", {}).get("class") != "success" or step_seconds is None:
        return None
    step_seconds = _finite_nonnegative(step_seconds, "mean_step_seconds")
    measured_steps = _positive_int(measured_steps, "measured_step_count")
    if step_seconds <= 0:
        raise ValueError("mean_step_seconds must be positive for a success")
    work = measurements.get("work") or {}

    def per_step(name: str) -> float:
        value = work.get(name)
        if value is None:
            raise ValueError(f"Successful performance row has no {name}")
        return _finite_nonnegative(value, name) / measured_steps

    computed_tokens = per_step("computed_tokens")
    effective_tokens = per_step("effective_tokens")
    attention_pairs = per_step("computed_attention_token_pairs")
    logical_samples = per_step("logical_samples")
    if computed_tokens <= 0 or effective_tokens <= 0 or logical_samples <= 0:
        raise ValueError("Successful performance work counters must be positive")

    linear_base = geometry["linear_applications_per_pass"]
    adapter = geometry["adapter_parameters"]
    mode = str(job.get("train_type"))
    if mode == "full":
        linear_flops = 6 * linear_base * computed_tokens
    else:
        linear_flops = (
            4 * linear_base * computed_tokens + 6 * adapter * computed_tokens
        )
    attention_flops = (
        6
        * geometry["num_layers"]
        * geometry["hidden_size"]
        * attention_pairs
    )
    recompute_linear = 0.0
    recompute_attention = 0.0
    if bool(job.get("gc")):
        recompute_linear = 2 * linear_base * computed_tokens
        if mode == "lora":
            recompute_linear += 2 * adapter * computed_tokens
        recompute_attention = (
            2
            * geometry["num_layers"]
            * geometry["hidden_size"]
            * attention_pairs
        )
    total_flops = (
        linear_flops
        + attention_flops
        + recompute_linear
        + recompute_attention
    )
    num_gpus = _positive_int(job.get("gpu_count"), "gpu_count")
    peak = _finite_nonnegative(
        hardware.get("bf16_dense_peak_flops_per_second_for_mfu"),
        "dense BF16 peak",
    )
    if peak <= 0:
        raise ValueError("dense BF16 peak must be positive")
    gradient_accumulation, gradient_accumulation_evidence = _gradient_accumulation(
        job, runtime_root
    )

    local_tokens = computed_tokens / num_gpus
    per_token_layer = (
        6 * geometry["hidden_size"]
        + 2 * geometry["kv_width"]
        + 3 * geometry["intermediate_size"]
    )
    weight_bytes_per_microstep = 2 * (
        linear_base + adapter
    ) * PARAMETER_BYTES
    weight_traffic = gradient_accumulation * weight_bytes_per_microstep
    activation_traffic = (
        4
        * local_tokens
        * geometry["num_layers"]
        * per_token_layer
        * PARAMETER_BYTES
    )
    attention_traffic = (
        4
        * local_tokens
        * (2 * geometry["hidden_size"] + 2 * geometry["kv_width"])
        * PARAMETER_BYTES
    )
    recompute_traffic = (
        2 * (activation_traffic + attention_traffic) if bool(job.get("gc")) else 0.0
    )
    optimizer_traffic = (
        2
        * geometry["trainable_parameters"]
        / num_gpus
        * (PARAMETER_BYTES + GRADIENT_BYTES + OPTIMIZER_BYTES)
    )
    kernel_traffic = (
        weight_traffic + activation_traffic + attention_traffic + recompute_traffic
    )

    zero_stage = _zero_stage(job.get("zero"))
    trainable = geometry["trainable_parameters"]
    loaded = geometry["loaded_parameters"]
    ring = (num_gpus - 1) / num_gpus if num_gpus > 1 else 0.0
    payload = 0.0
    collective_count = 0
    if num_gpus > 1 and zero_stage == 1:
        gradient_payload = trainable * COMMUNICATION_BYTES
        updated_payload = trainable * PARAMETER_BYTES
        payload = ring * (2 * gradient_payload + updated_payload)
        collective_count = _ceil_div(
            trainable, DEFAULT_REDUCE_BUCKET_ELEMENTS
        ) + _ceil_div(trainable, DEFAULT_ALL_GATHER_BUCKET_ELEMENTS)
    elif num_gpus > 1 and zero_stage == 2:
        gradient_payload = trainable * COMMUNICATION_BYTES
        updated_payload = trainable * PARAMETER_BYTES
        payload = ring * (
            2 * gradient_accumulation * gradient_payload + updated_payload
        )
        collective_count = gradient_accumulation * _ceil_div(
            trainable, DEFAULT_REDUCE_BUCKET_ELEMENTS
        ) + _ceil_div(trainable, DEFAULT_ALL_GATHER_BUCKET_ELEMENTS)
    elif num_gpus > 1 and zero_stage == 3:
        materializations = 3 if bool(job.get("gc")) else 2
        payload = ring * gradient_accumulation * (
            materializations * loaded * PARAMETER_BYTES
            + trainable * COMMUNICATION_BYTES
        )
        collective_count = gradient_accumulation * (
            materializations * _ceil_div(loaded, DEFAULT_PREFETCH_BUCKET_ELEMENTS)
            + _ceil_div(trainable, DEFAULT_REDUCE_BUCKET_ELEMENTS)
        )

    return {
        "physical_mbs": _positive_int(job.get("mbs"), "mbs"),
        "gradient_accumulation_steps": gradient_accumulation,
        "gradient_accumulation_evidence": gradient_accumulation_evidence,
        "work_per_step": {
            "computed_tokens": computed_tokens,
            "effective_tokens": effective_tokens,
            "computed_attention_token_pairs": attention_pairs,
            "logical_samples": logical_samples,
        },
        "flops_per_step": {
            "linear": float(linear_flops),
            "attention": float(attention_flops),
            "recompute_linear": float(recompute_linear),
            "recompute_attention": float(recompute_attention),
            "total": float(total_flops),
        },
        "traffic_bytes_per_rank_step": {
            "weights": float(weight_traffic),
            "linear_activations": float(activation_traffic),
            "attention": float(attention_traffic),
            "recompute": float(recompute_traffic),
            "kernel_total": float(kernel_traffic),
            "optimizer": float(optimizer_traffic),
        },
        "communication": {
            "payload_bytes_per_rank_step": float(payload),
            "collective_count": collective_count,
            "ideal_payload_seconds": float(
                payload / H800_INTRA_NODE_BANDWIDTH_BYTES_S
            ),
        },
        "ideal_seconds": {
            "compute_at_dense_peak": float(total_flops / (num_gpus * peak)),
            "kernel_hbm_at_physical_peak": float(
                kernel_traffic / H800_MEMORY_BANDWIDTH_BYTES_S
            ),
            "optimizer_hbm_at_physical_peak": float(
                optimizer_traffic / H800_MEMORY_BANDWIDTH_BYTES_S
            ),
            "collective_payload_at_link_peak": float(
                payload / H800_INTRA_NODE_BANDWIDTH_BYTES_S
            ),
        },
        "observed": {
            "mean_step_seconds": step_seconds,
            "effective_tokens_per_second": effective_tokens / step_seconds,
            "logical_samples_per_second": logical_samples / step_seconds,
            "computed_tokens_per_second": computed_tokens / step_seconds,
            "mfu_diagnostic": total_flops / (num_gpus * peak * step_seconds),
        },
        "physical_priors": {
            "dense_bf16_peak_flops_per_gpu": peak,
            "hbm_bandwidth_bytes_per_second": H800_MEMORY_BANDWIDTH_BYTES_S,
            "intra_node_bandwidth_bytes_per_second": H800_INTRA_NODE_BANDWIDTH_BYTES_S,
            "hbm_efficiency": {"center": DEFAULT_HBM_EFFICIENCY, "lower": 0.50},
            "collective_efficiency": {
                "center": DEFAULT_COLLECTIVE_EFFICIENCY,
                "lower": 0.60,
            },
            "communication_overlap": DEFAULT_COMMUNICATION_OVERLAP,
            "collective_latency_seconds": DEFAULT_COLLECTIVE_LATENCY_SECONDS,
        },
        "identifiability": {
            "compute_efficiency": "fit_allowed_with_bounds",
            "hbm_efficiency": "physical_prior_not_separately_identified_by_step_time",
            "collective_efficiency": "physical_prior_not_separately_identified_by_step_time",
            "collective_latency": "physical_prior_not_separately_identified_by_step_time",
        },
    }


def _memory_observation(row: dict[str, Any]) -> dict[str, Any]:
    outcome = str((row.get("outcome") or {}).get("class") or "")
    memory = (row.get("measurements") or {}).get("memory") or {}
    allocated = _finite_nonnegative(
        memory.get("max_allocated_bytes"), "max_allocated_bytes"
    )
    reserved = _finite_nonnegative(
        memory.get("max_reserved_bytes"), "max_reserved_bytes"
    )
    if memory.get("values_are_observed_not_imputed") is not True:
        raise ValueError("Memory observations must be observed, not imputed")
    if outcome == "success":
        return {
            "kind": "exact_success_peak",
            "peak_reserved_target_bytes": reserved,
            "peak_allocated_diagnostic_bytes": allocated,
            "allocator_reserved_minus_allocated_bytes": max(0.0, reserved - allocated),
            "right_censor_lower_bytes": None,
        }
    if outcome != "oom":
        raise ValueError(f"Unsupported calibration outcome {outcome!r}")
    censoring = row.get("censoring") or {}
    if censoring.get("kind") != "right_censored_memory_demand":
        raise ValueError("OOM row has no right-censored demand evidence")
    if censoring.get("demand_peak_bytes") is not None:
        raise ValueError("OOM demand peak must remain unknown, never imputed")
    requested = _finite_nonnegative(
        censoring.get("requested_allocation_bytes"), "requested allocation"
    )
    capacity = _finite_nonnegative(
        censoring.get("device_capacity_bytes_reported_in_error"),
        "OOM device capacity",
    )
    free = _finite_nonnegative(
        censoring.get("free_bytes_reported_in_error"), "OOM free bytes"
    )
    lower = max(reserved, allocated + requested, capacity - free + requested)
    return {
        "kind": "right_censored_oom",
        "peak_reserved_target_bytes": None,
        "peak_allocated_diagnostic_bytes": allocated,
        "allocator_reserved_minus_allocated_bytes": max(0.0, reserved - allocated),
        "right_censor_lower_bytes": lower,
        "right_censor_components": {
            "observed_reserved_bytes": reserved,
            "observed_allocated_plus_requested_bytes": allocated + requested,
            "capacity_minus_free_plus_requested_bytes": capacity - free + requested,
            "requested_allocation_bytes": requested,
        },
    }


def build_record(
    row: dict[str, Any],
    recovery: dict[str, Any],
    model: dict[str, Any],
    fixed_lora: dict[str, Any],
    hardware: dict[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    job = (row.get("configuration") or {}).get("job") or {}
    geometry = _model_geometry(job, model, fixed_lora)
    capacity = _positive_int(
        hardware.get("memory_bytes_reported_by_torch"), "H800 memory capacity"
    )
    memory = memory_basis(job, geometry, capacity)
    memory["observed"] = _memory_observation(row)
    runtime = recovery.get("runtime") or {}
    cohort_id = runtime.get("runtime_cohort_id")
    if not isinstance(cohort_id, str) or not cohort_id:
        raise ValueError("Recovered calibration record has no runtime cohort")
    zero_stage = _zero_stage(job.get("zero"))
    selector = {
        "runtime_cohort_id": cohort_id,
        "dtype": "bf16",
        "kernel_path": "fa3_orig+liger_fused_ce+adamw_torch_fused",
        "training_mode": str(job.get("train_type")),
        "zero_stage": zero_stage,
        "gradient_checkpointing": bool(job.get("gc")),
        "packing": bool(job.get("packing")),
    }
    scenario = {
        "model_id": job.get("model_id"),
        "train_type": job.get("train_type"),
        "dataset_id": job.get("dataset_id"),
        "target_gbs": job.get("target_gbs"),
        "gpu_count": job.get("gpu_count"),
        "physical_mbs": job.get("mbs"),
        "cutoff_len": job.get("cutoff_len"),
    }
    eligibility = recovery.get("measurement_eligibility") or {}
    routes = {
        route: True for route in CALIBRATION_ROUTES if eligibility.get(route) is True
    }
    return {
        "schema": RECORD_SCHEMA,
        "observation_id": row["observation_id"],
        "job_id": job.get("job_id"),
        "source_observation_sha256": recovery.get("source_observation_sha256"),
        "recovery_id": recovery.get("recovery_id"),
        "evidence_tier": recovery.get("evidence_tier"),
        "route": routes,
        "measurement_eligibility": routes,
        "outcome": (row.get("outcome") or {}).get("class"),
        "scenario": scenario,
        "scenario_id": sha256_json(
            {
                "model_id": scenario["model_id"],
                "train_type": scenario["train_type"],
                "dataset_id": scenario["dataset_id"],
                "target_gbs": scenario["target_gbs"],
            }
        ),
        "selector": selector,
        "runtime": {
            "runtime_cohort_id": cohort_id,
            "runtime_cohort_material": runtime.get("runtime_cohort_material"),
        },
        "model_basis": geometry,
        "memory": memory,
        "performance": performance_basis(
            row, job, geometry, hardware, runtime_root
        ),
        "confidence": "theory_only",
        "publishable": False,
    }


def build_report(
    observation_path: Path,
    recovery_path: Path,
    inventory_path: Path,
    hardware_path: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    observations = _read_observations(observation_path)
    recovery = read_json(recovery_path)
    if not isinstance(recovery, dict) or recovery.get("schema") != RECOVERY_SCHEMA:
        raise ValueError(f"Historical recovery is not {RECOVERY_SCHEMA}")
    recovery_records = recovery.get("records")
    if not isinstance(recovery_records, list):
        raise ValueError("Historical recovery has no records")
    inventory = read_json(inventory_path)
    models = inventory.get("models") if isinstance(inventory, dict) else None
    if not isinstance(models, list):
        raise ValueError("Model inventory has no models")
    model_by_id = {str(model.get("id")): model for model in models}
    if len(model_by_id) != len(models):
        raise ValueError("Model inventory contains duplicate ids")
    fixed_lora = inventory.get("fixed_lora") or {}
    hardware = read_json(hardware_path)
    if "h800" not in str(hardware.get("name_reported_by_driver") or "").lower():
        raise ValueError("Theory basis is H800-only")
    capacity = _positive_int(
        hardware.get("memory_bytes_reported_by_torch"), "H800 memory capacity"
    )

    records: list[dict[str, Any]] = []
    for recovered in recovery_records:
        eligibility = recovered.get("measurement_eligibility") or {}
        if eligibility.get("class") != "calibration_candidate":
            continue
        observation_id = recovered.get("source_observation_id")
        if observation_id not in observations:
            raise ValueError(f"Recovery references missing observation {observation_id}")
        row = observations[observation_id]
        job = (row.get("configuration") or {}).get("job") or {}
        model_id = str(job.get("model_id") or "")
        if model_id not in model_by_id:
            raise ValueError(f"No model inventory for {model_id}")
        records.append(
            build_record(
                row,
                recovered,
                model_by_id[model_id],
                fixed_lora,
                hardware,
                runtime_root,
            )
        )
    records.sort(key=lambda record: record["observation_id"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "gpu_family": "H800",
        "status": "bootstrap_theory_only",
        "confidence": "theory_only",
        "publishable": False,
        "coefficients_were_fit": False,
        "theory_scope": {
            "purpose": "physical_basis_for_bounded_historical_calibration",
            "gpu_execution_required": False,
            "fit_performed": False,
            "historical_qwen_structure_is_planner_trusted_exact": False,
        },
        "source_bindings": {
            "observations": {
                "path": str(observation_path),
                "sha256": sha256_file(observation_path),
            },
            "historical_recovery": {
                "path": str(recovery_path),
                "sha256": sha256_file(recovery_path),
                "report_sha256": recovery.get("report_sha256"),
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
            "observation_ids_sha256": sha256_json(
                [record["observation_id"] for record in records]
            ),
        },
        "counts": {
            "records": len(records),
            "outcomes": dict(sorted(Counter(record["outcome"] for record in records).items())),
            "routes": {
                route: sum(record["route"].get(route) is True for record in records)
                for route in CALIBRATION_ROUTES
            },
        },
        "theory_contract": {
            "dtype_bytes": {
                "parameter": PARAMETER_BYTES,
                "gradient": GRADIENT_BYTES,
                "optimizer": OPTIMIZER_BYTES,
                "communication": COMMUNICATION_BYTES,
                "logits": LOGITS_BYTES,
            },
            "activation_liveness_bounds": list(ACTIVATION_LIVENESS_BOUNDS),
            "safe_capacity_fraction": 0.95,
            "selector_keys": sorted(SELECTOR_KEYS),
            "forbidden_fit_features": ["model_id", "dataset_id", "target_gbs"],
        },
        "physical_priors": {
            "policy": "explicit_versioned_priors_null_is_never_imputed_or_fitted",
            "values": {
                "dense_peak_flops_per_s": _finite_nonnegative(
                    hardware.get("bf16_dense_peak_flops_per_second_for_mfu"),
                    "dense BF16 peak",
                ),
                "memory_bandwidth_bytes_per_s": H800_MEMORY_BANDWIDTH_BYTES_S,
                "collective_bandwidth_bytes_per_s": H800_INTRA_NODE_BANDWIDTH_BYTES_S,
                "hbm_efficiency": DEFAULT_HBM_EFFICIENCY,
                "optimizer_hbm_efficiency": DEFAULT_HBM_EFFICIENCY,
                "collective_efficiency": DEFAULT_COLLECTIVE_EFFICIENCY,
                "collective_latency_seconds": DEFAULT_COLLECTIVE_LATENCY_SECONDS,
                "microstep_latency_seconds": None,
                "framework_latency_seconds": None,
                "communication_overlap_by_stage": {
                    str(stage): DEFAULT_COMMUNICATION_OVERLAP
                    for stage in range(4)
                },
                "memory_capacity_bytes": capacity,
                "compute_efficiency": None,
            },
            "identifiability": {
                "compute_efficiency": "bounded_fit_allowed",
                "hbm_efficiency": "explicit_prior_not_separately_identified",
                "optimizer_hbm_efficiency": "explicit_prior_not_separately_identified",
                "collective_efficiency": "explicit_prior_not_separately_identified",
                "collective_latency_seconds": "explicit_prior_not_separately_identified",
                "communication_overlap_by_stage": "explicit_prior_not_separately_identified",
                "microstep_latency_seconds": "unidentified_and_unused_by_v1_basis",
                "framework_latency_seconds": "unidentified_and_unused_by_v1_basis",
            },
        },
        "records": records,
        "publication_blockers": [
            "historical_recovery_is_not_native_v2_publication_evidence",
            "runtime_mechanism_fingerprint_missing",
            "planner_trusted_exact_operator_manifest_missing",
            "prospective_acceptance_required",
        ],
    }
    report["report_sha256"] = sha256_json(report)
    validate_report(report)
    return report


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported theory basis schema {report.get('schema')}")
    digest = report.get("report_sha256")
    content = dict(report)
    content.pop("report_sha256", None)
    if digest != sha256_json(content):
        raise ValueError("Theory basis report SHA-256 mismatch")
    if report.get("publishable") is not False or report.get("coefficients_were_fit") is not False:
        raise ValueError("A theory basis cannot be fitted or publishable")
    scope = report.get("theory_scope") or {}
    if scope.get("gpu_execution_required") is not False or scope.get("fit_performed") is not False:
        raise ValueError("Theory basis must remain CPU-only and unfitted")
    priors = (report.get("physical_priors") or {}).get("values") or {}
    positive_priors = (
        "dense_peak_flops_per_s",
        "memory_bandwidth_bytes_per_s",
        "collective_bandwidth_bytes_per_s",
        "memory_capacity_bytes",
    )
    for name in positive_priors:
        if _finite_nonnegative(priors.get(name), name) <= 0:
            raise ValueError(f"Physical prior {name} must be positive")
    for name in (
        "hbm_efficiency",
        "optimizer_hbm_efficiency",
        "collective_efficiency",
    ):
        value = _finite_nonnegative(priors.get(name), name)
        if value <= 0 or value > 1:
            raise ValueError(f"Physical efficiency prior {name} must be in (0, 1]")
    if priors.get("compute_efficiency") is not None:
        raise ValueError("Compute efficiency belongs to the bounded fit, not the basis")
    if priors.get("microstep_latency_seconds") is not None or priors.get(
        "framework_latency_seconds"
    ) is not None:
        raise ValueError("Unidentified latency priors must remain null")
    overlap = priors.get("communication_overlap_by_stage")
    if not isinstance(overlap, dict) or set(overlap) != {"0", "1", "2", "3"}:
        raise ValueError("Communication overlap priors must cover stages 0-3")
    for stage, value in overlap.items():
        normalized = _finite_nonnegative(value, f"stage {stage} overlap")
        if normalized > 1:
            raise ValueError("Communication overlap prior must be in [0, 1]")
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Theory basis has no records")
    if (report.get("counts") or {}).get("records") != len(records):
        raise ValueError("Theory basis record count mismatch")
    ids: set[str] = set()
    for record in records:
        if record.get("schema") != RECORD_SCHEMA:
            raise ValueError("Theory basis contains an invalid record schema")
        observation_id = record.get("observation_id")
        if not isinstance(observation_id, str) or observation_id in ids:
            raise ValueError("Theory basis observation ids must be unique")
        ids.add(observation_id)
        selector = record.get("selector") or {}
        if set(selector) != SELECTOR_KEYS:
            raise ValueError("Theory basis selector keys are not mechanism-only")
        if {"model_id", "dataset_id", "target_gbs"}.intersection(selector):
            raise ValueError("Model/dataset identifiers cannot enter fit selectors")
        outcome = record.get("outcome")
        memory = record.get("memory") or {}
        lower = _finite_nonnegative(
            memory.get("analytic_lower_bound_bytes"), "analytic memory lower"
        )
        reference = _finite_nonnegative(
            memory.get("analytic_reference_bytes"), "analytic memory reference"
        )
        if lower > reference:
            raise ValueError("Analytic memory lower bound exceeds reference")
        observed = memory.get("observed") or {}
        if outcome == "oom":
            if observed.get("peak_reserved_target_bytes") is not None:
                raise ValueError("OOM demand cannot be used as an exact target")
            if (
                _finite_nonnegative(
                    observed.get("right_censor_lower_bytes"), "OOM censor lower"
                )
                <= 0
            ):
                raise ValueError("OOM censor lower must be positive")
            if record.get("performance") is not None:
                raise ValueError("OOM cannot contribute a performance target")
        elif outcome == "success":
            if observed.get("right_censor_lower_bytes") is not None:
                raise ValueError("Success cannot be right censored")
            if (
                _finite_nonnegative(
                observed.get("peak_reserved_target_bytes"), "success peak target"
                )
                <= 0
            ):
                raise ValueError("Success peak target must be positive")
            performance = record.get("performance") or {}
            if (
                _finite_nonnegative(
                    (performance.get("observed") or {}).get("mean_step_seconds"),
                    "success mean step seconds",
                )
                <= 0
            ):
                raise ValueError("Success mean step seconds must be positive")
        else:
            raise ValueError(f"Invalid theory basis outcome {outcome!r}")
    json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)


def write_report(path: Path, report: dict[str, Any]) -> None:
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
        "--historical-recovery",
        type=Path,
        default=ROOT / "artifacts" / "historical_h800_recovery.json",
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
        default=ROOT / "artifacts" / "h800_theory_basis.json",
    )
    args = parser.parse_args()
    report = build_report(
        args.observations,
        args.historical_recovery,
        args.model_inventory,
        args.hardware,
        args.runtime_root,
    )
    write_report(args.output, report)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "output": str(args.output),
                "records": report["counts"]["records"],
                "outcomes": report["counts"]["outcomes"],
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
