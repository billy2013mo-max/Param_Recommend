#!/usr/bin/env python3
"""Build four processor-bound pzfj38 VL calibration profiles."""

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

from common import ARTIFACT_DIR, DATA_DIR, ROOT, read_json, sha256_file, sha256_json, write_json
from vl_workload_profile import build_workload_profile


SCHEMA = "sft_h800_vl_calibration_processor_profiles/v1"
MEDIA_MANIFEST = ARTIFACT_DIR / "h800_vl_calibration_media_manifest_v1.json"
DATA = DATA_DIR / "vl_calibration_v1" / "vl_pzfj38_calibration_v1.jsonl"
SOURCE = (
    ROOT.parent
    / "real_business_validation_20260730"
    / "datasets"
    / "dataset-pzfj38-1774860803"
    / "113"
    / "publish"
    / "extracted"
    / "dataset-pzfj38-1774860803-V113.jsonl"
)
OUTPUT_DIR = ARTIFACT_DIR / "h800_vl_calibration_profiles_v1"
OUTPUT_MANIFEST = ARTIFACT_DIR / "h800_vl_calibration_processor_profiles_manifest_v1.json"
IMAGE_PLACEHOLDER = "<image>"
ONE_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


MODELS = (
    {
        "id": "qwen2p5_vl_7b",
        "path": "/wanqing-models/Qwen2.5-VL-7B-Instruct",
        "template": "qwen2_vl",
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "image_min_pixels": 56 * 56,
    },
    {
        "id": "qwen3_vl_8b",
        "path": "/wanqing-models/Qwen3-VL-8B-Instruct",
        "template": "qwen3_vl",
        "patch_size": 16,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "image_min_pixels": 64 * 64,
    },
)
TIERS = (
    {"id": "low", "image_max_pixels": 448 * 448},
    {"id": "high", "image_max_pixels": 768 * 768},
)


class TrainingEncoder:
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
            message["content"] = message["content"].replace(
                IMAGE_PLACEHOLDER, ONE_IMAGE_TOKEN
            )
        system = None
        conversation = processed
        if processed and processed[0]["role"] == "system":
            system = processed[0]["content"]
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


def _llamafactory_resize(
    width: int, height: int, *, maximum: int, minimum: int
) -> tuple[int, int]:
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


def _processor_limits(image_processor: Any) -> tuple[int, int]:
    minimum = getattr(image_processor, "min_pixels", None)
    maximum = getattr(image_processor, "max_pixels", None)
    size = getattr(image_processor, "size", {}) or {}
    if minimum is None:
        minimum = size.get("shortest_edge")
    if maximum is None:
        maximum = size.get("longest_edge")
    if type(minimum) is not int or type(maximum) is not int:
        raise ValueError(f"processor pixel limits unavailable: {minimum}, {maximum}")
    return minimum, maximum


def _grid(
    width: int,
    height: int,
    *,
    cap: int,
    floor: int,
    patch_size: int,
    merge_size: int,
    processor_minimum: int,
    processor_maximum: int,
) -> tuple[list[int], tuple[int, int], tuple[int, int]]:
    regular_width, regular_height = _llamafactory_resize(
        width, height, maximum=cap, minimum=floor
    )
    resized_height, resized_width = smart_resize(
        regular_height,
        regular_width,
        factor=patch_size * merge_size,
        min_pixels=processor_minimum,
        max_pixels=processor_maximum,
    )
    return (
        [1, resized_height // patch_size, resized_width // patch_size],
        (regular_width, regular_height),
        (resized_width, resized_height),
    )


def _read_sources() -> list[dict[str, Any]]:
    rows = []
    with SOURCE.open(encoding="utf-8") as source:
        for line in source:
            if len(rows) >= 1000:
                break
            rows.append(json.loads(line))
    if len(rows) != 1000:
        raise ValueError("processor calibration requires exactly 1000 source rows")
    return rows


def _validate_actual_processor(
    *,
    processor: Any,
    media_by_digest: dict[str, dict[str, Any]],
    source_rows: list[dict[str, Any]],
    cap: int,
    floor: int,
    expected_by_digest: dict[str, list[int]],
) -> dict[str, Any]:
    checked = []
    seen: set[str] = set()
    for row in source_rows:
        for url in row["images"]:
            import hashlib

            digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            media = media_by_digest[digest]
            with Image.open(media["path"]) as opened:
                image = opened.convert("RGB")
            width, height = _llamafactory_resize(
                image.width, image.height, maximum=cap, minimum=floor
            )
            image = image.resize((width, height))
            actual = processor.image_processor(
                images=[image], return_tensors="pt"
            )["image_grid_thw"][0].tolist()
            expected = expected_by_digest[digest]
            if actual != expected:
                raise ValueError(
                    f"processor grid mismatch for {digest}: {actual} != {expected}"
                )
            checked.append({"url_sha256": digest, "grid_thw": actual})
            if len(checked) >= 16:
                return {"checked_images": checked, "all_exact": True}
    raise ValueError("not enough images for processor validation")


def prepare() -> dict[str, Any]:
    media_manifest = read_json(MEDIA_MANIFEST)
    if (
        media_manifest.get("schema") != "sft_h800_vl_calibration_media/v1"
        or media_manifest.get("derived", {}).get("rows") != 1000
        or media_manifest.get("derived", {}).get("all_decoded") is not True
        or media_manifest.get("derived", {}).get("sha256") != sha256_file(DATA)
    ):
        raise ValueError("VL calibration media manifest is not frozen/complete")
    media_by_digest = {str(row["url_sha256"]): row for row in media_manifest["media"]}
    source_rows = _read_sources()
    outputs = []
    for model in MODELS:
        processor = AutoProcessor.from_pretrained(
            model["path"], trust_remote_code=True, local_files_only=True
        )
        image_processor = processor.image_processor
        patch_size = int(getattr(image_processor, "patch_size"))
        merge_size = int(getattr(image_processor, "merge_size"))
        temporal_patch_size = int(getattr(image_processor, "temporal_patch_size"))
        if (
            patch_size != model["patch_size"]
            or merge_size != model["merge_size"]
            or temporal_patch_size != model["temporal_patch_size"]
        ):
            raise ValueError(f"{model['id']} processor geometry drifted")
        processor_minimum, processor_maximum = _processor_limits(image_processor)
        encoder = TrainingEncoder(model["path"], model["template"])
        encoded = [encoder.encode(row["messages"]) for row in source_rows]
        for tier in TIERS:
            profile_rows = []
            expected_by_digest: dict[str, list[int]] = {}
            for index, (source_row, token_pair) in enumerate(zip(source_rows, encoded)):
                images = []
                for url in source_row["images"]:
                    import hashlib

                    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
                    media = media_by_digest[digest]
                    grid, _, resized = _grid(
                        int(media["width"]),
                        int(media["height"]),
                        cap=int(tier["image_max_pixels"]),
                        floor=int(model["image_min_pixels"]),
                        patch_size=patch_size,
                        merge_size=merge_size,
                        processor_minimum=processor_minimum,
                        processor_maximum=processor_maximum,
                    )
                    expected_by_digest[digest] = grid
                    images.append(
                        {
                            "grid_thw": grid,
                            "width": resized[0],
                            "height": resized[1],
                            "pixels": resized[0] * resized[1],
                        }
                    )
                base_total_tokens, label_tokens = token_pair
                profile_rows.append(
                    {
                        "sample_id": str(index),
                        "task_family": "full_reference_image_quality_sft",
                        "images": images,
                        "text_tokens": base_total_tokens - len(images),
                        "label_tokens": label_tokens,
                        "repeated_prompt_tokens": 0,
                    }
                )
            profile = build_workload_profile(
                profile_rows,
                model_id=model["id"],
                processor_name=type(image_processor).__name__,
                processor_version=f"transformers-{transformers.__version__}",
                patch_size=patch_size,
                spatial_merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
                image_min_pixels=int(model["image_min_pixels"]),
                image_max_pixels=int(tier["image_max_pixels"]),
                default_task_family="full_reference_image_quality_sft",
            )
            validation = _validate_actual_processor(
                processor=processor,
                media_by_digest=media_by_digest,
                source_rows=source_rows,
                cap=int(tier["image_max_pixels"]),
                floor=int(model["image_min_pixels"]),
                expected_by_digest=expected_by_digest,
            )
            profile["source_binding"] = {
                "data_path": str(DATA.resolve()),
                "data_sha256": sha256_file(DATA),
                "media_manifest_path": str(MEDIA_MANIFEST.resolve()),
                "media_manifest_sha256": sha256_file(MEDIA_MANIFEST),
            }
            profile["llamafactory_pre_resize"] = {
                "image_min_pixels": int(model["image_min_pixels"]),
                "image_max_pixels": int(tier["image_max_pixels"]),
            }
            profile["processor_internal_limits"] = {
                "min_pixels": processor_minimum,
                "max_pixels": processor_maximum,
            }
            profile["actual_processor_validation"] = validation
            profile["tier"] = tier["id"]
            profile["report_sha256"] = sha256_json(profile)
            path = OUTPUT_DIR / f"{model['id']}.{tier['id']}.json"
            write_json(path, profile)
            outputs.append(
                {
                    "model_id": model["id"],
                    "tier": tier["id"],
                    "image_min_pixels": int(model["image_min_pixels"]),
                    "image_max_pixels": int(tier["image_max_pixels"]),
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "report_sha256": profile["report_sha256"],
                    "summary": profile["summary"],
                    "actual_processor_validation": validation,
                }
            )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "media_manifest": {
            "path": str(MEDIA_MANIFEST.resolve()),
            "sha256": sha256_file(MEDIA_MANIFEST),
            "report_sha256": media_manifest["report_sha256"],
        },
        "source": {"path": str(SOURCE.resolve()), "sha256": sha256_file(SOURCE)},
        "profiles": outputs,
        "all_actual_processor_checks_passed": all(
            row["actual_processor_validation"]["all_exact"] is True for row in outputs
        ),
        "raw_text_or_urls_written": False,
        "usage": "calibration_fit_only_never_prospective_acceptance",
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT_MANIFEST, report)
    return report


def main() -> None:
    report = prepare()
    print(
        json.dumps(
            {
                "manifest": str(OUTPUT_MANIFEST),
                "profiles": [
                    {
                        "model_id": row["model_id"],
                        "tier": row["tier"],
                        "sha256": row["sha256"],
                        "summary": row["summary"],
                    }
                    for row in report["profiles"]
                ],
                "all_actual_processor_checks_passed": report[
                    "all_actual_processor_checks_passed"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
