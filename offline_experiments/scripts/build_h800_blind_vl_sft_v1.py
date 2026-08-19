#!/usr/bin/env python3
"""Build source-disjoint VL SFT sets from blind_data/vl (real image + synthetic prompt).

blind_data/vl holds 999 groups of 16 real business video frames with no
conversation/label.  For each requested ``--frames`` count N we emit a dataset
where every sample uses the first N frames of a group plus a fixed synthetic
instruction/answer.  Varying N varies the per-step vision patch/pixel load, which
is exactly the axis the frozen-vision-tower peak memory depends on; the V3
campaign uses the 1/2/4-frame sets to calibrate that dependence.

Per-image processor geometry (grid_thw, visual tokens, pixel elements) is
reproduced with the deterministic Qwen2-VL smart_resize.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from common import ARTIFACT_DIR, DATA_DIR, ROOT, sha256_file, sha256_json, write_json

BLIND_VL_DIR = ROOT / "blind_data" / "vl"
BASE_OUT_DIR = DATA_DIR / "blind_vl_prospective_v1"
BASE_PROFILE_DIR = ARTIFACT_DIR / "h800_blind_vl_workload_profiles_v1"
DEFAULT_SAMPLES = 64
CHANNELS = 3

SYNTHETIC_SYSTEM = "你是电商短视频画面理解助手。"
SYNTHETIC_INSTRUCTION = "请综合这些视频画面，描述其主要内容与商品信息。"
SYNTHETIC_ANSWER = "（合成占位答案：本样本仅用于冻结视觉塔的显存/吞吐前瞻验收，答案文本不参与业务评估。）"

MODEL_PROCESSOR = {
    "qwen2p5_vl_3b": {"patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2},
    "qwen3_vl_4b": {"patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2},
    "qwen3p5_4b": {"patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2},
}
TIER_PIXELS = {
    "low": {"image_min_pixels": 3136, "image_max_pixels": 200704},
    "high": {"image_min_pixels": 3136, "image_max_pixels": 589824},
}


def _smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _image_record(path: Path, processor: dict[str, int], tier: dict[str, int]) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
    patch = processor["patch_size"]
    merge = processor["spatial_merge_size"]
    temporal = processor["temporal_patch_size"]
    factor = patch * merge
    h_bar, w_bar = _smart_resize(height, width, factor, tier["image_min_pixels"], tier["image_max_pixels"])
    grid_h, grid_w = h_bar // patch, w_bar // patch
    raw_patch_units = grid_h * grid_w
    visual_tokens = raw_patch_units // (merge * merge)
    pixel_values_elements = raw_patch_units * CHANNELS * temporal * patch * patch
    return {
        "modality": "image",
        "grid_thw": [1, grid_h, grid_w],
        "sampled_frames": 1,
        "raw_patch_units": raw_patch_units,
        "visual_tokens": visual_tokens,
        "pixel_values_elements": pixel_values_elements,
        "processor_width": w_bar,
        "processor_height": h_bar,
    }


def _select_frames(frames: int) -> list[list[Path]]:
    samples: list[list[Path]] = []
    for group in sorted(BLIND_VL_DIR.iterdir()):
        if not group.is_dir():
            continue
        picked = sorted(group.glob("*.png"))[:frames]
        if len(picked) == frames:
            samples.append(picked)
    return samples


def _write_dataset(samples: list[list[Path]], demo: tuple[Path, Path, Path]) -> None:
    jsonl, registry, out_dir = demo
    out_dir.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w", encoding="utf-8") as out:
        for frames in samples:
            image_tags = "".join("<image>" for _ in frames)
            row = {
                "images": [str(p.resolve()) for p in frames],
                "messages": [
                    {"role": "system", "content": SYNTHETIC_SYSTEM},
                    {"role": "user", "content": image_tags + "\n" + SYNTHETIC_INSTRUCTION},
                    {"role": "assistant", "content": SYNTHETIC_ANSWER},
                ],
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(registry, {"vl_blind_prospective_v1": {
        "file_name": str(jsonl.resolve()),
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                 "assistant_tag": "assistant", "system_tag": "system"},
    }})


def _profile_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    def distribution(field: str) -> dict[str, float]:
        values = sorted(float(row.get(field) or 0.0) for row in records)
        n = len(values)

        def q(p: int) -> float:
            if not values:
                return 0.0
            idx = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
            return values[idx]

        return {"min": values[0], "mean": sum(values) / n, "p50": q(50), "p90": q(90), "p95": q(95), "p99": q(99), "max": values[-1]}

    return {
        "records": len(records),
        "images_per_sample": distribution("image_count"),
        "videos_per_sample": distribution("video_count"),
        "media_per_sample": distribution("media_count"),
        "visual_tokens_total": distribution("visual_tokens_total"),
        "raw_patch_units_total": distribution("raw_patch_units_total"),
        "pixel_values_elements_total": distribution("pixel_values_elements_total"),
        "sampled_video_frames_total": distribution("sampled_video_frames_total"),
        "total_tokens": distribution("total_tokens"),
        "text_tokens": distribution("text_tokens"),
    }


def _write_profiles(
    samples: list[list[Path]], profile_dir: Path
) -> list[dict[str, Any]]:
    written = []
    for model_id, processor in MODEL_PROCESSOR.items():
        for tier_name, tier in TIER_PIXELS.items():
            records = []
            for idx, frames in enumerate(samples):
                images = [_image_record(p, processor, tier) for p in frames]
                text_tokens = 64
                label_tokens = 48
                image_visual_tokens = sum(im["visual_tokens"] for im in images)
                records.append({
                    "sample_id": str(idx),
                    "task_family": "blind_vl_frame_understanding_synthetic_prompt",
                    "image_count": len(images),
                    "video_count": 0,
                    "media_count": len(images),
                    "images": images,
                    "videos": [],
                    "text_tokens": text_tokens,
                    "label_tokens": label_tokens,
                    "image_visual_tokens": image_visual_tokens,
                    "visual_tokens_total": image_visual_tokens,
                    "raw_patch_units_total": sum(im["raw_patch_units"] for im in images),
                    "pixel_values_elements_total": sum(im["pixel_values_elements"] for im in images),
                    "total_tokens": text_tokens + label_tokens + image_visual_tokens,
                })
            profile = {
                "schema": "sft_vl_workload_profile/v2",
                "profile_version": 2,
                "model": {"model_id": model_id},
                "provenance": {
                    "image_source": "blind_data/vl real business video frames (source-disjoint)",
                    "instruction": "synthetic_fixed_prompt",
                    "acceptance_target": "frozen_vision_tower_memory_and_throughput_only",
                },
                "processor_binding": {
                    "patch_size": processor["patch_size"],
                    "spatial_merge_size": processor["spatial_merge_size"],
                    "temporal_patch_size": processor["temporal_patch_size"],
                    "input_channels": CHANNELS,
                    "image_min_pixels": tier["image_min_pixels"],
                    "image_max_pixels": tier["image_max_pixels"],
                    "grid_contract": "image_grid_thw_after_smart_resize",
                    "reproduction": "deterministic_qwen2vl_smart_resize_no_processor_load",
                },
                "media_tier": tier_name,
                "source_id": f"blind_data/vl-{tier_name}",
                "summary": _profile_summary(records),
                "records": records,
            }
            path = profile_dir / f"{model_id}.blindvl.{tier_name}.json"
            write_json(path, profile)
            written.append({"model_id": model_id, "tier": tier_name, "path": str(path.resolve()), "sha256": sha256_file(path)})
    return written


def build(samples_limit: int, frames: int) -> dict[str, Any]:
    out_dir = BASE_OUT_DIR / f"f{frames}"
    jsonl = out_dir / f"blind_vl_images_f{frames}.jsonl"
    registry = out_dir / "registry" / "dataset_info.json"
    profile_dir = Path(str(BASE_PROFILE_DIR) + f"_f{frames}")
    samples = _select_frames(frames)[:samples_limit]
    if not samples:
        raise RuntimeError(f"no frame groups for frames={frames}")
    _write_dataset(samples, demo=(jsonl, registry, out_dir))
    profiles = _write_profiles(samples, profile_dir)
    design = {
        "schema": "sft_h800_blind_vl_sft_design/v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": "vl_blind_prospective_v1",
        "frames_per_sample": frames,
        "sample_count": len(samples),
        "dataset_jsonl": {"path": str(jsonl.resolve()), "sha256": sha256_file(jsonl)},
        "registry": {"path": str(registry.resolve()), "sha256": sha256_file(registry)},
        "profiles": profiles,
        "provenance_note": (
            f"Real source-disjoint frames ({frames} per sample), synthetic prompt; "
            "used only for frozen-vision-tower memory/throughput calibration/acceptance."
        ),
    }
    design["report_sha256"] = sha256_json(design)
    write_json(ARTIFACT_DIR / f"h800_blind_vl_sft_design_f{frames}_v1.json", design)
    return design


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--frames", type=int, required=True, help="frames per sample (1/2/4...)")
    args = parser.parse_args()
    print(json.dumps(build(args.samples, args.frames), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
