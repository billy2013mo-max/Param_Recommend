#!/usr/bin/env python3
"""Physical VL workload features shared by memory and throughput overlays."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


SCHEMA = "sft_vl_resource_features/v1"
SUPPORTED_PROFILE_SCHEMA = "sft_vl_workload_profile/v2"


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and non-negative") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _stat(summary: Mapping[str, Any], name: str, statistic: str = "mean") -> float:
    field = summary.get(name)
    if not isinstance(field, Mapping) or field.get(statistic) is None:
        raise ValueError(f"profile summary is missing {name}.{statistic}")
    return _finite_nonnegative(field[statistic], f"{name}.{statistic}")


def phase_max_reference_bytes(
    *,
    persistent_bytes: float,
    common_dynamic_bytes: float,
    language_dynamic_bytes: float,
    vision_dynamic_bytes: float,
) -> float:
    """Combine non-overlapping language/vision peaks without double counting."""

    values = {
        "persistent_bytes": persistent_bytes,
        "common_dynamic_bytes": common_dynamic_bytes,
        "language_dynamic_bytes": language_dynamic_bytes,
        "vision_dynamic_bytes": vision_dynamic_bytes,
    }
    clean = {name: _finite_nonnegative(value, name) for name, value in values.items()}
    return (
        clean["persistent_bytes"]
        + clean["common_dynamic_bytes"]
        + max(clean["language_dynamic_bytes"], clean["vision_dynamic_bytes"])
    )


def build_vl_resource_features(
    profile: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    physical_mbs: int,
    freeze_vision_tower: bool,
    freeze_multi_modal_projector: bool,
    dtype_bytes: int = 2,
) -> dict[str, Any]:
    """Build auditable memory and throughput proxies from a V2 profile."""

    if profile.get("schema") != SUPPORTED_PROFILE_SCHEMA:
        raise ValueError("VL resource features require workload profile V2")
    model_id = str(model.get("id") or "")
    if str((profile.get("model") or {}).get("model_id") or "") != model_id:
        raise ValueError("workload profile model binding does not match model inventory")
    if not freeze_vision_tower or not freeze_multi_modal_projector:
        raise ValueError(
            "VL overlay V1 only supports a frozen vision tower and frozen projector"
        )
    mbs = _positive_int(physical_mbs, "physical_mbs")
    dtype_bytes = _positive_int(dtype_bytes, "dtype_bytes")

    summary = profile.get("summary") or {}
    geometry = model.get("vision_geometry") or {}
    components = model.get("component_parameter_estimates") or {}
    depth = _positive_int(int(geometry.get("depth") or 0), "vision.depth")
    hidden = _positive_int(int(geometry.get("hidden_size") or 0), "vision.hidden_size")
    intermediate = _positive_int(
        int(geometry.get("intermediate_size") or 0),
        "vision.intermediate_size",
    )
    output_hidden = _positive_int(
        int(geometry.get("out_hidden_size") or model.get("hidden_size") or 0),
        "vision.out_hidden_size",
    )
    vision_parameters = _positive_int(
        int(components.get("vision_tower") or 0),
        "vision_tower_parameters",
    )
    projector_parameters = _positive_int(
        int(components.get("projector_or_merger") or 0),
        "projector_parameters",
    )

    image_count = _stat(summary, "images_per_sample")
    video_count = _stat(summary, "videos_per_sample")
    media_count = _stat(summary, "media_per_sample")
    visual_tokens = _stat(summary, "visual_tokens_total")
    raw_patches = _stat(summary, "raw_patch_units_total")
    pixel_elements = _stat(summary, "pixel_values_elements_total")
    sampled_frames = _stat(summary, "sampled_video_frames_total")
    total_tokens = _stat(summary, "total_tokens")
    text_tokens = _stat(summary, "text_tokens")
    patch_p95 = _stat(summary, "raw_patch_units_total", "p95")
    token_p95 = _stat(summary, "visual_tokens_total", "p95")

    # Forward-only proxies. They carry correct units and monotonic direction;
    # they are not asserted to equal measured CUDA peak or wall time.
    pixel_tensor_bytes = mbs * pixel_elements * dtype_bytes
    patch_embedding_bytes = mbs * raw_patches * hidden * dtype_bytes
    merged_embedding_bytes = mbs * visual_tokens * output_hidden * dtype_bytes
    vision_dynamic_proxy = (
        pixel_tensor_bytes + patch_embedding_bytes + merged_embedding_bytes
    )
    frozen_vision_state_bytes = (
        vision_parameters + projector_parameters
    ) * dtype_bytes

    per_patch_linear_flops = (
        8.0 * hidden * hidden + 4.0 * hidden * intermediate
    )
    vision_linear_flops_per_sample = depth * raw_patches * per_patch_linear_flops
    projector_flops_per_sample = 2.0 * projector_parameters * visual_tokens
    vision_forward_flops_per_sample = (
        vision_linear_flops_per_sample + projector_flops_per_sample
    )
    vision_hbm_bytes_per_sample = (
        (vision_parameters + projector_parameters) * dtype_bytes
        + pixel_elements * dtype_bytes
        + raw_patches * hidden * dtype_bytes
        + visual_tokens * output_hidden * dtype_bytes
    )

    deepstack_count = len((model.get("vision_config") or {}).get("deepstack_visual_indexes") or [])
    full_attention_count = len((model.get("vision_config") or {}).get("fullatt_block_indexes") or [])
    feature_values = {
        "is_vl": 1.0,
        "has_images": float(image_count > 0.0),
        "has_videos": float(video_count > 0.0),
        "log1p_images_per_sample": math.log1p(image_count),
        "log1p_videos_per_sample": math.log1p(video_count),
        "log1p_media_per_sample": math.log1p(media_count),
        "log1p_visual_tokens_per_sample": math.log1p(visual_tokens),
        "log1p_raw_patch_units_per_sample": math.log1p(raw_patches),
        "log1p_pixel_values_elements_per_sample": math.log1p(pixel_elements),
        "log1p_sampled_video_frames_per_sample": math.log1p(sampled_frames),
        "visual_token_fraction": visual_tokens / total_tokens if total_tokens else 0.0,
        "raw_patch_p95_to_mean": patch_p95 / raw_patches if raw_patches else 0.0,
        "visual_token_p95_to_mean": token_p95 / visual_tokens if visual_tokens else 0.0,
        "log2_vision_depth": math.log2(depth),
        "log2_vision_hidden": math.log2(hidden),
        "vision_intermediate_to_hidden": intermediate / hidden,
        "deepstack_layer_fraction": deepstack_count / depth,
        "full_attention_layer_fraction": full_attention_count / depth,
        "log2_physical_mbs": math.log2(mbs),
        "raw_patches_x_mbs_log1p": math.log1p(raw_patches * mbs),
        "sampled_frames_x_mbs_log1p": math.log1p(sampled_frames * mbs),
    }
    return {
        "schema": SCHEMA,
        "model_id": model_id,
        "profile_contract_sha256": (profile.get("processor_binding") or {}).get(
            "contract_sha256"
        ),
        "configuration": {
            "physical_mbs": mbs,
            "dtype_bytes": dtype_bytes,
            "freeze_vision_tower": True,
            "freeze_multi_modal_projector": True,
        },
        "language_work": {
            "mean_text_tokens_per_sample": text_tokens,
            "mean_visual_tokens_per_sample": visual_tokens,
            "mean_total_tokens_per_sample": total_tokens,
        },
        "vision_work": {
            "mean_images_per_sample": image_count,
            "mean_videos_per_sample": video_count,
            "mean_media_per_sample": media_count,
            "mean_raw_patch_units_per_sample": raw_patches,
            "mean_pixel_values_elements_per_sample": pixel_elements,
            "mean_sampled_video_frames_per_sample": sampled_frames,
        },
        "memory_proxies_bytes": {
            "frozen_vision_state": frozen_vision_state_bytes,
            "pixel_tensor_microbatch": pixel_tensor_bytes,
            "patch_embedding_microbatch": patch_embedding_bytes,
            "merged_embedding_microbatch": merged_embedding_bytes,
            "vision_dynamic_microbatch": vision_dynamic_proxy,
        },
        "throughput_proxies_per_sample": {
            "vision_forward_flops": vision_forward_flops_per_sample,
            "vision_linear_flops": vision_linear_flops_per_sample,
            "projector_flops": projector_flops_per_sample,
            "vision_hbm_bytes": vision_hbm_bytes_per_sample,
            "media_launch_units": media_count,
        },
        "feature_values": feature_values,
        "semantics": {
            "visual_tokens_feed_language_backbone": True,
            "raw_patch_units_feed_vision_tower": True,
            "vision_and_language_dynamic_peaks_are_phase_max_not_sum": True,
            "forward_proxies_require_paired_gpu_calibration": True,
        },
    }


def extend_throughput_work_per_step(
    base_work: Mapping[str, Any],
    vl_features: Mapping[str, Any],
    *,
    logical_samples_per_step: int,
) -> dict[str, float]:
    """Append vision work while leaving existing language work unchanged."""

    samples = _positive_int(logical_samples_per_step, "logical_samples_per_step")
    result = {str(key): _finite_nonnegative(value, str(key)) for key, value in base_work.items()}
    proxies = vl_features.get("throughput_proxies_per_sample") or {}
    result.update(
        {
            "vision_forward_flops": _finite_nonnegative(
                proxies.get("vision_forward_flops"), "vision_forward_flops"
            )
            * samples,
            "vision_hbm_bytes": _finite_nonnegative(
                proxies.get("vision_hbm_bytes"), "vision_hbm_bytes"
            )
            * samples,
            "vision_media_launch_units": _finite_nonnegative(
                proxies.get("media_launch_units"), "media_launch_units"
            )
            * samples,
        }
    )
    return result
