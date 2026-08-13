#!/usr/bin/env python3
"""Processor-bound image/video workload profiles for VL resource modeling.

V2 preserves two different workloads:

* merged visual tokens consumed by the language backbone;
* raw patch units and sampled frames consumed by the vision tower.

The module is CPU-only and never stores raw text or media URLs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from runtime_evidence import sha256_json
from vl_workload_profile import distribution, dimensions_to_grid_thw, visual_tokens_from_grid


SCHEMA = "sft_vl_workload_profile/v2"
PROFILE_VERSION = 2


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


def _optional_nonnegative_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be non-negative") from error
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _media_profile(
    media: Mapping[str, Any],
    *,
    modality: str,
    patch_size: int,
    spatial_merge_size: int,
    temporal_patch_size: int,
    input_channels: int,
) -> dict[str, Any]:
    if not isinstance(media, Mapping):
        raise ValueError(f"every {modality} entry must be an object")
    if modality not in {"image", "video"}:
        raise ValueError(f"unsupported modality {modality!r}")

    sampled_frames = _positive_int(
        int(media.get("sampled_frames") or media.get("frames") or 1),
        f"{modality}.sampled_frames",
    )
    if modality == "image" and sampled_frames != 1:
        raise ValueError("image.sampled_frames must equal one")

    if media.get("grid_thw") is not None:
        grid = [int(value) for value in media["grid_thw"]]
        if len(grid) != 3:
            raise ValueError(f"{modality}.grid_thw must contain [t, h, w]")
    else:
        width_value = media.get("processor_width", media.get("width"))
        height_value = media.get("processor_height", media.get("height"))
        if width_value is None or height_value is None:
            raise ValueError(
                f"{modality} requires grid_thw or processor width/height"
            )
        grid = dimensions_to_grid_thw(
            width=_positive_int(int(width_value), f"{modality}.width"),
            height=_positive_int(int(height_value), f"{modality}.height"),
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            frames=sampled_frames,
            temporal_patch_size=temporal_patch_size,
        )

    temporal, grid_height, grid_width = (
        _positive_int(int(value), f"{modality}.grid_thw[{index}]")
        for index, value in enumerate(grid)
    )
    expected_frames = 1 if modality == "image" else temporal * temporal_patch_size
    if sampled_frames != expected_frames:
        raise ValueError(
            f"{modality}.sampled_frames disagrees with grid_thw and "
            f"temporal_patch_size: {sampled_frames} != {expected_frames}"
        )

    visual_tokens = visual_tokens_from_grid(
        grid,
        spatial_merge_size=spatial_merge_size,
    )
    declared_tokens = media.get("visual_tokens")
    if declared_tokens is not None and int(declared_tokens) != visual_tokens:
        raise ValueError(
            f"{modality}.visual_tokens disagrees with grid_thw: "
            f"{declared_tokens} != {visual_tokens}"
        )

    raw_patch_units = temporal * grid_height * grid_width
    patch_vector_elements = (
        input_channels * temporal_patch_size * patch_size * patch_size
    )
    calculated_elements = raw_patch_units * patch_vector_elements
    declared_elements = media.get("pixel_values_elements")
    if declared_elements is not None and int(declared_elements) != calculated_elements:
        raise ValueError(
            f"{modality}.pixel_values_elements disagrees with processor grid: "
            f"{declared_elements} != {calculated_elements}"
        )

    result: dict[str, Any] = {
        "modality": modality,
        "grid_thw": [temporal, grid_height, grid_width],
        "sampled_frames": sampled_frames,
        "raw_patch_units": raw_patch_units,
        "visual_tokens": visual_tokens,
        "pixel_values_elements": calculated_elements,
    }
    for source, target in (
        ("processor_width", "processor_width"),
        ("processor_height", "processor_height"),
        ("width", "source_width"),
        ("height", "source_height"),
    ):
        if media.get(source) is not None:
            result[target] = _positive_int(int(media[source]), f"{modality}.{source}")
    for source in ("source_fps", "sample_fps", "duration_seconds"):
        value = _optional_nonnegative_float(media.get(source), f"{modality}.{source}")
        if value is not None:
            result[source] = value
    return result


def profile_record(
    row: Mapping[str, Any],
    *,
    patch_size: int,
    spatial_merge_size: int,
    temporal_patch_size: int,
    input_channels: int,
    default_task_family: str = "causal_vl_sft",
) -> dict[str, Any]:
    """Normalize one media-aware sample without retaining raw content."""

    images = row.get("images") or []
    videos = row.get("videos") or []
    for name, values in (("images", images), ("videos", videos)):
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"record.{name} must be a list")

    image_profiles = [
        _media_profile(
            media,
            modality="image",
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
            input_channels=input_channels,
        )
        for media in images
    ]
    video_profiles = [
        _media_profile(
            media,
            modality="video",
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
            input_channels=input_channels,
        )
        for media in videos
    ]
    media_profiles = [*image_profiles, *video_profiles]

    text_tokens = _nonnegative_int(row.get("text_tokens"), "text_tokens")
    label_tokens = _nonnegative_int(row.get("label_tokens"), "label_tokens")
    repeated_prompt_tokens = _nonnegative_int(
        row.get("repeated_prompt_tokens"),
        "repeated_prompt_tokens",
    )
    image_tokens = sum(int(media["visual_tokens"]) for media in image_profiles)
    video_tokens = sum(int(media["visual_tokens"]) for media in video_profiles)
    visual_tokens = image_tokens + video_tokens
    total_tokens = text_tokens + visual_tokens
    if label_tokens > total_tokens:
        raise ValueError("label_tokens cannot exceed total language-backbone tokens")

    task_family = str(row.get("task_family") or default_task_family).strip()
    if not task_family:
        raise ValueError("task_family must be non-empty")
    return {
        "sample_id": str(row.get("sample_id") or row.get("record_index") or ""),
        "task_family": task_family,
        "image_count": len(image_profiles),
        "video_count": len(video_profiles),
        "media_count": len(media_profiles),
        "images": image_profiles,
        "videos": video_profiles,
        "text_tokens": text_tokens,
        "label_tokens": label_tokens,
        "repeated_prompt_tokens": repeated_prompt_tokens,
        "image_visual_tokens": image_tokens,
        "video_visual_tokens": video_tokens,
        "visual_tokens_total": visual_tokens,
        "raw_patch_units_total": sum(
            int(media["raw_patch_units"]) for media in media_profiles
        ),
        "pixel_values_elements_total": sum(
            int(media["pixel_values_elements"]) for media in media_profiles
        ),
        "sampled_video_frames_total": sum(
            int(media["sampled_frames"]) for media in video_profiles
        ),
        "total_tokens": total_tokens,
        "visual_token_ratio": visual_tokens / total_tokens if total_tokens else 0.0,
    }


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def values(name: str) -> list[int | float]:
        return [record[name] for record in records]

    all_images = [image for record in records for image in record["images"]]
    all_videos = [video for record in records for video in record["videos"]]
    return {
        "records": len(records),
        "images_per_sample": distribution(values("image_count")),
        "videos_per_sample": distribution(values("video_count")),
        "media_per_sample": distribution(values("media_count")),
        "text_tokens": distribution(values("text_tokens")),
        "label_tokens": distribution(values("label_tokens")),
        "repeated_prompt_tokens": distribution(values("repeated_prompt_tokens")),
        "image_visual_tokens": distribution(values("image_visual_tokens")),
        "video_visual_tokens": distribution(values("video_visual_tokens")),
        "visual_tokens_total": distribution(values("visual_tokens_total")),
        "raw_patch_units_total": distribution(values("raw_patch_units_total")),
        "pixel_values_elements_total": distribution(
            values("pixel_values_elements_total")
        ),
        "sampled_video_frames_total": distribution(
            values("sampled_video_frames_total")
        ),
        "total_tokens": distribution(values("total_tokens")),
        "visual_token_ratio": distribution(values("visual_token_ratio")),
        "visual_tokens_per_image": distribution(
            [int(image["visual_tokens"]) for image in all_images]
        ),
        "visual_tokens_per_video": distribution(
            [int(video["visual_tokens"]) for video in all_videos]
        ),
        "raw_patch_units_per_image": distribution(
            [int(image["raw_patch_units"]) for image in all_images]
        ),
        "raw_patch_units_per_video": distribution(
            [int(video["raw_patch_units"]) for video in all_videos]
        ),
        "sampled_frames_per_video": distribution(
            [int(video["sampled_frames"]) for video in all_videos]
        ),
        "video_duration_seconds": distribution(
            [float(video["duration_seconds"]) for video in all_videos if "duration_seconds" in video]
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
    temporal_patch_size: int,
    input_channels: int = 3,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
    video_sample_fps: float | None = None,
    video_max_frames: int | None = None,
    default_task_family: str = "causal_vl_sft",
) -> dict[str, Any]:
    """Build a versioned processor-bound mixed-media workload profile."""

    if not rows:
        raise ValueError("rows must be non-empty")
    if not str(model_id).strip():
        raise ValueError("model_id must be non-empty")
    if not str(processor_name).strip() or not str(processor_version).strip():
        raise ValueError("processor_name and processor_version are required")
    patch_size = _positive_int(patch_size, "patch_size")
    spatial_merge_size = _positive_int(spatial_merge_size, "spatial_merge_size")
    temporal_patch_size = _positive_int(temporal_patch_size, "temporal_patch_size")
    input_channels = _positive_int(input_channels, "input_channels")
    for name, value in (
        ("image_min_pixels", image_min_pixels),
        ("image_max_pixels", image_max_pixels),
        ("video_min_pixels", video_min_pixels),
        ("video_max_pixels", video_max_pixels),
        ("video_max_frames", video_max_frames),
    ):
        if value is not None:
            _positive_int(value, name)
    if image_min_pixels and image_max_pixels and image_min_pixels > image_max_pixels:
        raise ValueError("image_min_pixels cannot exceed image_max_pixels")
    if video_min_pixels and video_max_pixels and video_min_pixels > video_max_pixels:
        raise ValueError("video_min_pixels cannot exceed video_max_pixels")
    video_sample_fps = _optional_nonnegative_float(video_sample_fps, "video_sample_fps")

    records = [
        profile_record(
            row,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
            input_channels=input_channels,
            default_task_family=default_task_family,
        )
        for row in rows
    ]
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[str(record["task_family"])].append(record)

    binding = {
        "name": str(processor_name),
        "version": str(processor_version),
        "patch_size": patch_size,
        "spatial_merge_size": spatial_merge_size,
        "temporal_patch_size": temporal_patch_size,
        "input_channels": input_channels,
        "image_min_pixels": image_min_pixels,
        "image_max_pixels": image_max_pixels,
        "video_min_pixels": video_min_pixels,
        "video_max_pixels": video_max_pixels,
        "video_sample_fps": video_sample_fps,
        "video_max_frames": video_max_frames,
        "grid_contract": "image_grid_thw_and_video_grid_thw_after_processor",
        "language_length_contract": "text_tokens_plus_merged_visual_tokens",
        "vision_work_contract": "raw_patch_units_before_spatial_merge",
    }
    binding["contract_sha256"] = sha256_json(binding)
    return {
        "schema": SCHEMA,
        "profile_version": PROFILE_VERSION,
        "model": {"model_id": str(model_id)},
        "processor_binding": binding,
        "records": records,
        "summary": _summarize_records(records),
        "summary_by_task_family": {
            task: _summarize_records(task_records)
            for task, task_records in sorted(by_task.items())
        },
        "privacy": {
            "raw_text_written": False,
            "raw_image_urls_written": False,
            "raw_video_urls_written": False,
        },
        "gpu_training_started": False,
    }
