#!/usr/bin/env python3
"""Decode one real video through each exact training processor/plugin stack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import av
import torch
import transformers
from transformers import AutoProcessor

from common import ARTIFACT_DIR, read_json, read_jsonl, sha256_file, write_json
from prepare_h800_frozen_video_decomposition_v1 import WORKLOAD_MANIFEST
from prepare_h800_qwen35_vl_supplement_v1 import MODEL_SPECS
from prepare_vl_business_workload_profiles_v2 import TrainingEncoderV2, VIDEO_DATA


OUTPUT = ARTIFACT_DIR / "h800_vl_video_full_decode_validation_v1.json"
MODELS = ("qwen2p5_vl_3b", "qwen3_vl_4b", "qwen3p5_4b")


def validate() -> dict[str, Any]:
    specs = {str(row["id"]): row for row in MODEL_SPECS}
    source = read_jsonl(VIDEO_DATA)[0]
    video_path = Path(str(source["videos"][0]))
    profiles = {
        str(row["model_id"]): row
        for row in read_json(WORKLOAD_MANIFEST)["profiles"]
        if row.get("source") == "zltbjg_video" and row.get("tier") == "native_16"
    }
    rows = []
    for model_id in MODELS:
        spec = specs[model_id]
        processor = AutoProcessor.from_pretrained(
            spec["path"], trust_remote_code=True, local_files_only=True
        )
        processor.video_min_pixels = 16 * 16
        processor.video_max_pixels = 256 * 256
        processor.video_fps = 2.0
        processor.video_maxlen = 16
        encoder = TrainingEncoderV2(spec["path"], spec["template"])
        plugin = encoder.template.mm_plugin
        mm_inputs = plugin._get_mm_inputs([], [str(video_path.resolve())], [], processor)
        pixels = mm_inputs.get("pixel_values_videos")
        grid = mm_inputs.get("video_grid_thw")
        if not torch.is_tensor(pixels) or pixels.numel() <= 0:
            raise ValueError(f"{model_id} full decode produced no video pixel tensor")
        if not torch.is_tensor(grid) or grid.shape != (1, 3):
            raise ValueError(f"{model_id} full decode produced invalid video_grid_thw")
        expected_profile = read_json(Path(profiles[model_id]["path"]))
        expected_grid = expected_profile["records"][0]["videos"][0]["grid_thw"]
        actual_grid = [int(value) for value in grid[0].tolist()]
        if actual_grid != expected_grid:
            raise ValueError(
                f"{model_id} decoded grid drifted: {actual_grid} != {expected_grid}"
            )
        rows.append(
            {
                "model_id": model_id,
                "processor_class": type(processor).__name__,
                "plugin_class": type(plugin).__name__,
                "video_grid_thw": actual_grid,
                "pixel_tensor_shape": [int(value) for value in pixels.shape],
                "pixel_tensor_elements": int(pixels.numel()),
                "expected_pixel_tensor_elements": int(
                    expected_profile["records"][0]["pixel_values_elements_total"]
                ),
                "grid_exact": True,
                "pixel_tensor_positive": True,
            }
        )
    report = {
        "schema": "sft_h800_vl_video_full_decode_validation/v1",
        "all_passed": True,
        "runtime": {
            "transformers": transformers.__version__,
            "pyav": av.__version__,
        },
        "video": {
            "path": str(video_path.resolve()),
            "sha256": sha256_file(video_path),
        },
        "models": rows,
        "gpu_training_started": False,
    }
    write_json(OUTPUT, report)
    return report


def main() -> None:
    print(json.dumps(validate(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
