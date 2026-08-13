#!/usr/bin/env python3
"""Build processor-bound V2 profiles for frozen image/video business data.

Run this script with the fine-tuning launcher virtual environment so its
Transformers, PyAV and LLaMA-Factory versions match the training runtime.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import transformers
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from common import ARTIFACT_DIR, DATA_DIR, read_json, read_jsonl, sha256_file, sha256_json, write_json
from vl_workload_profile_v2 import build_workload_profile


SCHEMA = "sft_h800_vl_business_workload_profiles/v2"
OUTPUT_DIR = ARTIFACT_DIR / "h800_vl_business_workload_profiles_v2"
OUTPUT_MANIFEST = ARTIFACT_DIR / "h800_vl_business_workload_profiles_manifest_v2.json"
MODEL_INVENTORY = ARTIFACT_DIR / "h800_qwen35_vl_supplement_model_inventory_v1.json"
IMAGE_PROFILE_MANIFEST_V1 = ARTIFACT_DIR / "h800_qwen35_vl_supplement_processor_profiles_manifest_v1.json"
MEDIA_DOWNLOAD_MANIFEST = ARTIFACT_DIR / "h800_vl_media_business_download_manifest_v1.json"
QYPE_DATA = DATA_DIR / "vl_media_business_v1/derived/qype19_v7_images_local.jsonl"
VIDEO_DATA = DATA_DIR / "vl_media_business_v1/derived/zltbjg_v2_videos_local.jsonl"

MODELS = (
    {"id": "qwen2p5_vl_3b", "path": "/wanqing-models/Qwen2.5-VL-3B-Instruct", "template": "qwen2_vl"},
    {"id": "qwen3_vl_4b", "path": "/wanqing-models/Qwen3-VL-4B-Instruct", "template": "qwen3_vl"},
    {"id": "qwen3p5_4b", "path": "/wanqing-models/Qwen3.5-4B", "template": "qwen3_5_nothink"},
)
IMAGE_TIERS = (
    {"id": "low", "image_max_pixels": 448 * 448},
    {"id": "high", "image_max_pixels": 768 * 768},
)
VIDEO_TIERS = (
    {
        "id": "native_16",
        "video_min_pixels": 16 * 16,
        "video_max_pixels": 256 * 256,
        "video_fps": 2.0,
        "video_maxlen": 16,
    },
    {
        "id": "native_64",
        "video_min_pixels": 16 * 16,
        "video_max_pixels": 256 * 256,
        "video_fps": 2.0,
        "video_maxlen": 64,
    },
    {
        "id": "native_128",
        "video_min_pixels": 16 * 16,
        "video_max_pixels": 256 * 256,
        "video_fps": 2.0,
        "video_maxlen": 128,
    },
    {
        "id": "upscaled_64",
        "video_min_pixels": 640 * 360,
        "video_max_pixels": 640 * 360,
        "video_fps": 2.0,
        "video_maxlen": 64,
    },
)
IMAGE_PLACEHOLDER = "<image>"
VIDEO_PLACEHOLDER = "<video>"
ONE_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
ONE_VIDEO_TOKEN = "<|vision_start|><|video_pad|><|vision_end|>"
# The downloaded business calibration view currently has empty targets.  An
# empty target is not training-equivalent across templates: qwen3_5_nothink
# masks every label, while the older VL templates retain an EOS label.  Keep a
# minimal, model-independent supervised target so every architecture executes
# the same real forward/backward workload instead of silently stopping at step
# zero.  This is a calibration sentinel, not a fabricated semantic annotation.
VIDEO_CALIBRATION_RESPONSE = "0"


class TrainingEncoderV2:
    def __init__(self, model_path: str, template_name: str) -> None:
        from llamafactory.data.template import TEMPLATES

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True,
        )
        self.template = copy.deepcopy(TEMPLATES[template_name])
        self.template.fix_special_tokens(self.tokenizer)

    def encode(self, messages: list[dict[str, str]]) -> tuple[int, int]:
        processed = copy.deepcopy(messages)
        for message in processed:
            message["content"] = (
                str(message.get("content") or "")
                .replace(IMAGE_PLACEHOLDER, ONE_IMAGE_TOKEN)
                .replace(VIDEO_PLACEHOLDER, ONE_VIDEO_TOKEN)
            )
        system = None
        conversation = processed
        if processed and processed[0].get("role") == "system":
            system = str(processed[0].get("content") or "")
            conversation = processed[1:]
        pairs = self.template.encode_multiturn(
            self.tokenizer,
            conversation,
            system=system,
            tools=None,
        )
        source_tokens = sum(len(source_ids) for source_ids, _ in pairs)
        label_tokens = sum(len(target_ids) for _, target_ids in pairs)
        if self.template.efficient_eos:
            label_tokens += 1
        return source_tokens + label_tokens, label_tokens


def _inventory() -> dict[str, dict[str, Any]]:
    rows = {str(row["id"]): row for row in read_json(MODEL_INVENTORY)["models"]}
    missing = [model["id"] for model in MODELS if model["id"] not in rows]
    if missing:
        raise ValueError(f"model inventory is missing {missing}")
    return rows


def _processor_limits(processor: Any) -> tuple[int, int]:
    size = getattr(processor, "size", {}) or {}
    minimum = getattr(processor, "min_pixels", None)
    maximum = getattr(processor, "max_pixels", None)
    if minimum is None:
        minimum = size.get("shortest_edge") if isinstance(size, dict) else getattr(size, "shortest_edge", None)
    if maximum is None:
        maximum = size.get("longest_edge") if isinstance(size, dict) else getattr(size, "longest_edge", None)
    if type(minimum) is not int or type(maximum) is not int:
        raise ValueError(f"processor limits unavailable: {minimum}, {maximum}")
    return minimum, maximum


def _llamafactory_resize(width: int, height: int, *, maximum: int, minimum: int) -> tuple[int, int]:
    if width * height > maximum:
        factor = math.sqrt(maximum / (width * height))
        width, height = int(width * factor), int(height * factor)
    if width * height < minimum:
        factor = math.sqrt(minimum / (width * height))
        width, height = int(width * factor), int(height * factor)
    width, height = max(width, 28), max(height, 28)
    if width / height > 200:
        width = height * 180
    if height / width > 200:
        height = width * 180
    return width, height


def _binding(path: Path, profile: dict[str, Any], *, source: str, tier: str) -> dict[str, Any]:
    return {
        "model_id": profile["model"]["model_id"],
        "source": source,
        "modality": "video" if "video" in source else "image",
        "tier": tier,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "report_sha256": profile["report_sha256"],
        "summary": profile["summary"],
    }


def _convert_existing_image_profiles() -> list[dict[str, Any]]:
    manifest = read_json(IMAGE_PROFILE_MANIFEST_V1)
    selected = {
        (str(row["model_id"]), str(row["tier"])): row
        for row in manifest["profiles"]
        if row["model_id"] in {model["id"] for model in MODELS}
        and row["tier"] in {tier["id"] for tier in IMAGE_TIERS}
    }
    outputs = []
    for model in MODELS:
        for tier in IMAGE_TIERS:
            source_binding = selected[(model["id"], tier["id"])]
            source_path = Path(source_binding["path"])
            if sha256_file(source_path) != source_binding["sha256"]:
                raise ValueError(f"frozen V1 image profile drifted: {source_path}")
            source = read_json(source_path)
            processor = source["processor_binding"]
            rows = []
            for record in source["records"]:
                rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "task_family": record["task_family"],
                        "images": [
                            {
                                "grid_thw": image["grid_thw"],
                                "sampled_frames": 1,
                                "processor_width": image.get("width"),
                                "processor_height": image.get("height"),
                            }
                            for image in record["images"]
                        ],
                        "text_tokens": int(record["text_tokens"]),
                        "label_tokens": int(record["label_tokens"]),
                        "repeated_prompt_tokens": int(record.get("repeated_prompt_tokens") or 0),
                    }
                )
            profile = build_workload_profile(
                rows,
                model_id=model["id"],
                processor_name=str(processor["name"]),
                processor_version=str(processor["version"]),
                patch_size=int(processor["patch_size"]),
                spatial_merge_size=int(processor["spatial_merge_size"]),
                temporal_patch_size=int(processor["temporal_patch_size"]),
                image_min_pixels=int(source_binding["image_min_pixels"]),
                image_max_pixels=int(source_binding["image_max_pixels"]),
                default_task_family="full_reference_image_quality_sft",
            )
            profile["source_binding"] = {
                "source_profile_v1": str(source_path.resolve()),
                "source_profile_v1_sha256": source_binding["sha256"],
                "data": source["source_binding"],
            }
            profile["tier"] = tier["id"]
            profile["source_id"] = "pzfj38_v113"
            profile["conversion_contract"] = "V1 actual processor grids retained exactly; V2 adds raw patch and pixel tensor work"
            profile["report_sha256"] = sha256_json(profile)
            path = OUTPUT_DIR / f"{model['id']}.pzfj38.{tier['id']}.json"
            write_json(path, profile)
            outputs.append(_binding(path, profile, source="pzfj38_image", tier=tier["id"]))
    return outputs


def _prepare_qype_profiles(inventory: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = read_jsonl(QYPE_DATA)
    if len(rows) != 2011 or any(len(row.get("images") or []) != 2 for row in rows):
        raise ValueError("qype19 local dataset must contain 2,011 two-image rows")
    outputs = []
    for model in MODELS:
        processor = AutoProcessor.from_pretrained(
            model["path"], trust_remote_code=True, local_files_only=True
        )
        image_processor = processor.image_processor
        patch_size = int(image_processor.patch_size)
        merge_size = int(image_processor.merge_size)
        temporal_patch_size = int(image_processor.temporal_patch_size)
        processor_minimum, processor_maximum = _processor_limits(image_processor)
        image_floor = int(inventory[model["id"]]["image_min_pixels"])
        encoder = TrainingEncoderV2(model["path"], model["template"])
        encoded = [encoder.encode(row["messages"]) for row in rows]
        dimensions: dict[str, tuple[int, int]] = {}
        for row in rows:
            for path_value in row["images"]:
                if path_value in dimensions:
                    continue
                with Image.open(path_value) as image:
                    dimensions[path_value] = image.size

        for tier in IMAGE_TIERS:
            profile_rows = []
            expected_grids: dict[str, list[int]] = {}
            for index, (row, token_pair) in enumerate(zip(rows, encoded)):
                media_rows = []
                for path_value in row["images"]:
                    width, height = dimensions[path_value]
                    regular_width, regular_height = _llamafactory_resize(
                        width,
                        height,
                        maximum=int(tier["image_max_pixels"]),
                        minimum=image_floor,
                    )
                    resized_height, resized_width = smart_resize(
                        regular_height,
                        regular_width,
                        factor=patch_size * merge_size,
                        min_pixels=processor_minimum,
                        max_pixels=processor_maximum,
                    )
                    grid = [1, resized_height // patch_size, resized_width // patch_size]
                    expected_grids[path_value] = grid
                    media_rows.append(
                        {
                            "grid_thw": grid,
                            "sampled_frames": 1,
                            "width": width,
                            "height": height,
                            "processor_width": resized_width,
                            "processor_height": resized_height,
                        }
                    )
                total, labels = token_pair
                profile_rows.append(
                    {
                        "sample_id": str(index),
                        "task_family": "ecommerce_two_image_quality_sft",
                        "images": media_rows,
                        "text_tokens": total - 2,
                        "label_tokens": labels,
                    }
                )
            checked = []
            for path_value in list(dimensions)[:16]:
                with Image.open(path_value) as image:
                    image = image.convert("RGB")
                    regular = _llamafactory_resize(
                        image.width,
                        image.height,
                        maximum=int(tier["image_max_pixels"]),
                        minimum=image_floor,
                    )
                    image = image.resize(regular)
                    actual = image_processor(images=[image], return_tensors="pt")["image_grid_thw"][0].tolist()
                if actual != expected_grids[path_value]:
                    raise ValueError(f"qype19 processor grid mismatch for {model['id']}/{tier['id']}")
                checked.append(actual)
            profile = build_workload_profile(
                profile_rows,
                model_id=model["id"],
                processor_name=type(image_processor).__name__,
                processor_version=f"transformers-{transformers.__version__}",
                patch_size=patch_size,
                spatial_merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
                image_min_pixels=image_floor,
                image_max_pixels=int(tier["image_max_pixels"]),
                default_task_family="ecommerce_two_image_quality_sft",
            )
            profile["source_binding"] = {
                "data_path": str(QYPE_DATA.resolve()),
                "data_sha256": sha256_file(QYPE_DATA),
                "media_manifest_path": str(MEDIA_DOWNLOAD_MANIFEST.resolve()),
                "media_manifest_sha256": sha256_file(MEDIA_DOWNLOAD_MANIFEST),
            }
            profile["actual_processor_validation"] = {"checked_images": len(checked), "all_exact": True}
            profile["tier"] = tier["id"]
            profile["source_id"] = "qype19_v7"
            profile["report_sha256"] = sha256_json(profile)
            path = OUTPUT_DIR / f"{model['id']}.qype19.{tier['id']}.json"
            write_json(path, profile)
            outputs.append(_binding(path, profile, source="qype19_image", tier=tier["id"]))
    return outputs


def _video_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt = str(row.get("prompt") or "")
    if VIDEO_PLACEHOLDER not in prompt:
        prompt = f"{VIDEO_PLACEHOLDER}\n{prompt}"
    response = str(row.get("response") or "").strip() or VIDEO_CALIBRATION_RESPONSE
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def _prepare_video_profiles(inventory: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = read_jsonl(VIDEO_DATA)
    if len(rows) != 100 or any(len(row.get("videos") or []) != 1 for row in rows):
        raise ValueError("zltbjg local dataset must contain 100 one-video rows")
    media_manifest = read_json(MEDIA_DOWNLOAD_MANIFEST)
    video_by_path = {str(row["path"]): row for row in media_manifest["media"]["zltbjg_v2"]}
    outputs = []
    for model in MODELS:
        processor = AutoProcessor.from_pretrained(
            model["path"], trust_remote_code=True, local_files_only=True
        )
        video_processor = processor.video_processor
        patch_size = int(video_processor.patch_size)
        merge_size = int(video_processor.merge_size)
        temporal_patch_size = int(video_processor.temporal_patch_size)
        encoder = TrainingEncoderV2(model["path"], model["template"])
        plugin = encoder.template.mm_plugin

        for tier in VIDEO_TIERS:
            for name in ("video_min_pixels", "video_max_pixels", "video_fps", "video_maxlen"):
                setattr(processor, name, tier[name])
            profile_rows = []
            for index, row in enumerate(rows):
                path_value = str(row["videos"][0])
                metadata = plugin._get_qwen_video_grid_metadata([path_value], processor)
                if metadata is None:
                    raise ValueError(f"video metadata unavailable for {model['id']}/{tier['id']}/{index}")
                grid = [int(value) for value in metadata["video_grid_thw"][0].tolist()]
                sampled_before_pad = len(metadata["frames_indices"][0])
                media = video_by_path[path_value]
                processed_messages = plugin.process_messages(
                    _video_messages(row),
                    [],
                    [path_value],
                    [],
                    processor,
                )
                total, labels = encoder.encode(processed_messages)
                visual_tokens = math.prod(grid) // (merge_size**2)
                if labels <= 0:
                    raise ValueError(
                        f"processed video has no supervised labels for "
                        f"{model['id']}/{tier['id']}/{index}"
                    )
                if total <= visual_tokens:
                    raise ValueError(
                        f"processed video tokens invalid for {model['id']}/{tier['id']}/{index}: "
                        f"total={total}, visual={visual_tokens}"
                    )
                profile_rows.append(
                    {
                        "sample_id": str(index),
                        "task_family": "single_video_sft",
                        "videos": [
                            {
                                "grid_thw": grid,
                                "sampled_frames": grid[0] * temporal_patch_size,
                                "width": int(media["width"]),
                                "height": int(media["height"]),
                                "processor_width": grid[2] * patch_size,
                                "processor_height": grid[1] * patch_size,
                                "source_fps": float(media["fps"]),
                                "sample_fps": float(tier["video_fps"]),
                                "duration_seconds": float(media["duration_seconds"]),
                                "sampled_frames_before_temporal_pad": sampled_before_pad,
                            }
                        ],
                        # `text_tokens` includes all actual non-video-pad tokens,
                        # including Qwen3-VL per-frame timestamps and repeated
                        # vision boundary tokens.  The profile builder adds the
                        # `visual_tokens` video-pad work back exactly once.
                        "text_tokens": total - visual_tokens,
                        "label_tokens": labels,
                    }
                )
            profile = build_workload_profile(
                profile_rows,
                model_id=model["id"],
                processor_name=type(video_processor).__name__,
                processor_version=f"transformers-{transformers.__version__}",
                patch_size=patch_size,
                spatial_merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
                input_channels=int((inventory[model["id"]].get("vision_config") or {}).get("in_channels") or (inventory[model["id"]].get("vision_config") or {}).get("in_chans") or 3),
                video_min_pixels=int(tier["video_min_pixels"]),
                video_max_pixels=int(tier["video_max_pixels"]),
                video_sample_fps=float(tier["video_fps"]),
                video_max_frames=int(tier["video_maxlen"]),
                default_task_family="single_video_sft",
            )
            profile["source_binding"] = {
                "data_path": str(VIDEO_DATA.resolve()),
                "data_sha256": sha256_file(VIDEO_DATA),
                "media_manifest_path": str(MEDIA_DOWNLOAD_MANIFEST.resolve()),
                "media_manifest_sha256": sha256_file(MEDIA_DOWNLOAD_MANIFEST),
            }
            profile["metadata_validation"] = {
                "rows": len(profile_rows),
                "all_video_grids_from_llamafactory_plugin": True,
                "all_language_lengths_from_expanded_plugin_messages": True,
                "all_label_token_counts_positive": True,
                "empty_source_response_fallback": VIDEO_CALIBRATION_RESPONSE,
                "fallback_role": "calibration_sentinel_not_semantic_annotation",
                "pyav_available_in_profile_runtime": True,
                "full_decode_remains_runtime_canary_gate": True,
            }
            profile["tier"] = tier["id"]
            profile["source_id"] = "zltbjg_v2"
            profile["report_sha256"] = sha256_json(profile)
            path = OUTPUT_DIR / f"{model['id']}.zltbjg.{tier['id']}.json"
            write_json(path, profile)
            outputs.append(_binding(path, profile, source="zltbjg_video", tier=tier["id"]))
    return outputs


def prepare() -> dict[str, Any]:
    for required in (MODEL_INVENTORY, IMAGE_PROFILE_MANIFEST_V1, MEDIA_DOWNLOAD_MANIFEST, QYPE_DATA, VIDEO_DATA):
        if not required.is_file():
            raise FileNotFoundError(required)
    inventory = _inventory()
    profiles = [
        *_convert_existing_image_profiles(),
        *_prepare_qype_profiles(inventory),
        *_prepare_video_profiles(inventory),
    ]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "profile_schema": "sft_vl_workload_profile/v2",
        "profiles": profiles,
        "counts": {
            "total": len(profiles),
            "pzfj38_image": sum(row["source"] == "pzfj38_image" for row in profiles),
            "qype19_image": sum(row["source"] == "qype19_image" for row in profiles),
            "zltbjg_video": sum(row["source"] == "zltbjg_video" for row in profiles),
        },
        "runtime_binding": {
            "python": "/fine-tuning-launcher/.venv/bin/python",
            "transformers_version": transformers.__version__,
            "llamafactory_source": "/fine-tuning-launcher/LlamaFactory/src",
            "video_metadata_decoder": "PyAV",
        },
        "all_profiles_cpu_only": True,
        "gpu_training_started": False,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT_MANIFEST, report)
    return report


def main() -> None:
    report = prepare()
    print(
        json.dumps(
            {
                "manifest": str(OUTPUT_MANIFEST.resolve()),
                "report_sha256": report["report_sha256"],
                "counts": report["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
