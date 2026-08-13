#!/usr/bin/env python3
"""Versioned, processor-bound VL workload profile primitives.

The functions here are CPU-only and intentionally separate visual encoder
work, text work and task semantics.  They accept already observed processor
dimensions or ``image_grid_thw``; they do not pretend that a header-only image
estimate is a runtime validation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
import statistics
from typing import Any

from runtime_evidence import sha256_json


SCHEMA = "sft_vl_workload_profile/v1"
PROFILE_VERSION = 1
_STAT_KEYS = ("min", "mean", "p50", "p90", "p95", "p99", "max")


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str, *, default: int = 0) -> int:
    if value is None:
        value = default
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and non-negative") from error
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


def visual_tokens_from_grid(
    grid_thw: Sequence[int],
    *,
    spatial_merge_size: int,
) -> int:
    """Convert processor ``image_grid_thw`` to merged visual tokens.

    ``grid_h`` and ``grid_w`` are patch-grid dimensions, so the spatial merge
    divisor is applied to both axes.  A processor should already make these
    dimensions divisible; rejecting non-divisible input catches a stale or
    mismatched processor contract instead of silently flooring it.
    """

    if not isinstance(grid_thw, Sequence) or len(grid_thw) != 3:
        raise ValueError("grid_thw must contain exactly [t, h, w]")
    temporal, height, width = (
        _positive_int(int(value), f"grid_thw[{index}]")
        for index, value in enumerate(grid_thw)
    )
    merge = _positive_int(spatial_merge_size, "spatial_merge_size")
    if height % merge or width % merge:
        raise ValueError(
            "image_grid_thw spatial dimensions must be divisible by "
            "spatial_merge_size"
        )
    return temporal * (height // merge) * (width // merge)


def dimensions_to_grid_thw(
    *,
    width: int,
    height: int,
    patch_size: int,
    spatial_merge_size: int,
    frames: int = 1,
    temporal_patch_size: int = 1,
) -> list[int]:
    """Convert processor-resized dimensions to an explicit grid contract."""

    width = _positive_int(width, "width")
    height = _positive_int(height, "height")
    patch = _positive_int(patch_size, "patch_size")
    frames = _positive_int(frames, "frames")
    temporal_patch = _positive_int(temporal_patch_size, "temporal_patch_size")
    if width % patch or height % patch:
        raise ValueError(
            "processor-resized width/height must be divisible by patch_size"
        )
    # A single image is represented by one temporal grid plane even for
    # processors whose video temporal patch size is two.  Multi-frame video
    # must still satisfy the temporal divisibility contract.
    if frames > 1 and frames % temporal_patch:
        raise ValueError(
            "video frame count must be divisible by temporal_patch_size"
        )
    temporal_grid = 1 if frames == 1 else frames // temporal_patch
    return [temporal_grid, height // patch, width // patch]


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Sequence[int | float]) -> dict[str, float | int | None]:
    """Return stable scalar statistics for a profile field."""

    clean = [float(value) for value in values]
    if not clean:
        return {key: None for key in _STAT_KEYS}
    result: dict[str, float | int | None] = {
        "min": min(clean),
        "mean": statistics.fmean(clean),
        "p50": _percentile(clean, 50),
        "p90": _percentile(clean, 90),
        "p95": _percentile(clean, 95),
        "p99": _percentile(clean, 99),
        "max": max(clean),
    }
    # Keep integer-valued minima/maxima readable while preserving fractional
    # quantiles and means.
    for key in ("min", "max"):
        if result[key] is not None and float(result[key]).is_integer():
            result[key] = int(result[key])
    return result


def _image_profile(
    image: Mapping[str, Any],
    *,
    patch_size: int,
    spatial_merge_size: int,
    temporal_patch_size: int,
) -> dict[str, Any]:
    if not isinstance(image, Mapping):
        raise ValueError("every image entry must be an object")
    frames = _nonnegative_int(image.get("frames"), "image.frames", default=1)
    frames = max(frames, 1)
    if image.get("grid_thw") is not None:
        grid = [int(value) for value in image["grid_thw"]]
        visual_tokens = visual_tokens_from_grid(
            grid,
            spatial_merge_size=spatial_merge_size,
        )
    else:
        width_value = image.get("width", image.get("model_width"))
        height_value = image.get("height", image.get("model_height"))
        if width_value is None or height_value is None:
            raise ValueError(
                "image requires grid_thw or processor-resized width/height"
            )
        width = _positive_int(int(width_value), "image.width")
        height = _positive_int(int(height_value), "image.height")
        grid = dimensions_to_grid_thw(
            width=width,
            height=height,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            frames=frames,
            temporal_patch_size=temporal_patch_size,
        )
        visual_tokens = visual_tokens_from_grid(
            grid,
            spatial_merge_size=spatial_merge_size,
        )
    declared_tokens = image.get("visual_tokens")
    if declared_tokens is not None:
        declared = _nonnegative_int(declared_tokens, "image.visual_tokens")
        if declared != visual_tokens:
            raise ValueError(
                "declared visual_tokens disagrees with image_grid_thw/processor "
                f"calculation: declared={declared}, calculated={visual_tokens}"
            )
    result = {
        "grid_thw": grid,
        "visual_tokens": visual_tokens,
        "frames": frames,
    }
    for source, target in (("width", "width"), ("height", "height")):
        if image.get(source) is not None:
            result[target] = _positive_int(int(image[source]), f"image.{source}")
    if image.get("pixels") is not None:
        result["pixels"] = _positive_int(int(image["pixels"]), "image.pixels")
    elif "width" in result and "height" in result:
        result["pixels"] = result["width"] * result["height"]
    return result


def profile_record(
    row: Mapping[str, Any],
    *,
    patch_size: int,
    spatial_merge_size: int,
    temporal_patch_size: int = 1,
    default_task_family: str = "causal_vl_sft",
) -> dict[str, Any]:
    """Normalize one media-aware sample without retaining raw text or URLs."""

    images = row.get("images") or []
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        raise ValueError("record.images must be a list")
    image_profiles = [
        _image_profile(
            image,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
        )
        for image in images
    ]
    text_tokens = row.get("text_tokens")
    if text_tokens is None and row.get("base_total_tokens_one_image_pad_each") is not None:
        # Existing Qwen2-VL business profiles count one image placeholder in
        # this field.  Keep the conversion explicit and versioned.
        text_tokens = int(row["base_total_tokens_one_image_pad_each"]) - len(images)
    text_tokens = _nonnegative_int(text_tokens, "text_tokens")
    label_tokens = _nonnegative_int(row.get("label_tokens"), "label_tokens")
    repeated_prompt_tokens = _nonnegative_int(
        row.get("repeated_prompt_tokens"),
        "repeated_prompt_tokens",
    )
    visual_tokens = sum(int(image["visual_tokens"]) for image in image_profiles)
    total_tokens = text_tokens + visual_tokens
    if total_tokens < label_tokens:
        raise ValueError("label_tokens cannot exceed text plus visual tokens")
    task_family = str(row.get("task_family") or default_task_family).strip()
    if not task_family:
        raise ValueError("task_family must be non-empty")
    sample_id = str(row.get("sample_id") or row.get("record_index") or "")
    return {
        "sample_id": sample_id,
        "task_family": task_family,
        "image_count": len(image_profiles),
        "images": image_profiles,
        "images_per_sample": len(image_profiles),
        "visual_tokens_total": visual_tokens,
        "text_tokens": text_tokens,
        "label_tokens": label_tokens,
        "repeated_prompt_tokens": repeated_prompt_tokens,
        "total_tokens": total_tokens,
        "image_token_ratio": (
            visual_tokens / total_tokens if total_tokens else 0.0
        ),
    }


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    image_counts = [int(record["image_count"]) for record in records]
    visual_tokens = [int(record["visual_tokens_total"]) for record in records]
    text_tokens = [int(record["text_tokens"]) for record in records]
    label_tokens = [int(record["label_tokens"]) for record in records]
    repeated_prompt = [int(record["repeated_prompt_tokens"]) for record in records]
    total_tokens = [int(record["total_tokens"]) for record in records]
    ratios = [float(record["image_token_ratio"]) for record in records]
    return {
        "records": len(records),
        "images_per_sample": distribution(image_counts),
        "visual_tokens_total": distribution(visual_tokens),
        "text_tokens": distribution(text_tokens),
        "label_tokens": distribution(label_tokens),
        "repeated_prompt_tokens": distribution(repeated_prompt),
        "total_tokens": distribution(total_tokens),
        "image_token_ratio": distribution(ratios),
        "visual_tokens_per_image": distribution(
            [
                int(image["visual_tokens"])
                for record in records
                for image in record["images"]
            ]
        ),
    }


def build_workload_profile(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    processor_name: str,
    processor_version: str,
    patch_size: int,
    spatial_merge_size: int,
    temporal_patch_size: int = 1,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
    default_task_family: str = "causal_vl_sft",
) -> dict[str, Any]:
    """Build an auditable profile grouped by task family."""

    if not str(model_id).strip():
        raise ValueError("model_id must be non-empty")
    if not str(processor_name).strip() or not str(processor_version).strip():
        raise ValueError("processor_name and processor_version are required")
    patch_size = _positive_int(patch_size, "patch_size")
    spatial_merge_size = _positive_int(
        spatial_merge_size,
        "spatial_merge_size",
    )
    temporal_patch_size = _positive_int(
        temporal_patch_size,
        "temporal_patch_size",
    )
    if image_min_pixels is not None:
        image_min_pixels = _positive_int(image_min_pixels, "image_min_pixels")
    if image_max_pixels is not None:
        image_max_pixels = _positive_int(image_max_pixels, "image_max_pixels")
    if (
        image_min_pixels is not None
        and image_max_pixels is not None
        and image_min_pixels > image_max_pixels
    ):
        raise ValueError("image_min_pixels cannot exceed image_max_pixels")

    records = [
        profile_record(
            row,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
            default_task_family=default_task_family,
        )
        for row in rows
    ]
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[str(record["task_family"])].append(record)
    processor_binding = {
        "name": str(processor_name),
        "version": str(processor_version),
        "patch_size": patch_size,
        "spatial_merge_size": spatial_merge_size,
        "temporal_patch_size": temporal_patch_size,
        "image_min_pixels": image_min_pixels,
        "image_max_pixels": image_max_pixels,
        "grid_contract": "image_grid_thw",
    }
    processor_binding["contract_sha256"] = sha256_json(processor_binding)
    return {
        "schema": SCHEMA,
        "profile_version": PROFILE_VERSION,
        "model": {"model_id": str(model_id)},
        "processor_binding": processor_binding,
        "records": records,
        "summary": _summarize_records(records),
        "summary_by_task_family": {
            task: _summarize_records(task_records)
            for task, task_records in sorted(by_task.items())
        },
        "privacy": {
            "raw_text_written": False,
            "raw_image_urls_written": False,
        },
        "gpu_training_started": False,
    }
