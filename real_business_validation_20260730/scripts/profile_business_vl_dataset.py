#!/usr/bin/env python3
"""Build a sampled token profile for the Qwen2.5-VL business dataset.

The JSONL contains remote image references rather than image files.  This
script therefore:

1. tokenizes the text of every record with the local Qwen2.5-VL tokenizer and
   the LLaMA-Factory ``qwen2_vl`` template;
2. deterministically samples records;
3. reads only the leading bytes of the sampled images to recover dimensions;
4. reproduces the Qwen2-VL resize/grid calculation to estimate visual tokens.

Artifacts contain hashes of image references, dimensions and token counts.
Raw text and raw image URLs are deliberately not copied.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    smart_resize,
)

from llamafactory.data.template import TEMPLATES


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "datasets"
    / "dataset-pzfj38-1774860803"
    / "113"
    / "publish"
    / "extracted"
    / "dataset-pzfj38-1774860803-V113.jsonl"
)
MODEL_PATH = Path("/wanqing-models/Qwen2.5-VL-7B-Instruct")
PROFILE_PATH = (
    ROOT / "profiles" / "business_pzfj38_v113.qwen2_vl.sampled.jsonl"
)
SUMMARY_PATH = ROOT / "profile_summaries" / "business_pzfj38_v113.json"

DATASET_ID = "business_pzfj38_v113"
SAMPLE_SEED = 20260730
DEFAULT_SAMPLE_RECORDS = 512
DEFAULT_WORKERS = 32
IMAGE_MAX_PIXELS = 768 * 768
IMAGE_MIN_PIXELS = 32 * 32
PROCESSOR_MIN_PIXELS = 56 * 56
PROCESSOR_MAX_PIXELS = 12845056
PATCH_SIZE = 14
MERGE_SIZE = 2
CUTOFF_GRID = (4096, 4608, 6144, 8192)
IMAGE_PLACEHOLDER = "<image>"
ONE_IMAGE_TOKEN = (
    "<|vision_start|><|image_pad|><|vision_end|>"
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


@dataclass(frozen=True)
class Record:
    index: int
    images: tuple[str, ...]
    messages: tuple[dict[str, str], ...]
    base_total_tokens: int
    label_tokens: int


@dataclass(frozen=True)
class ImageDimension:
    width: int
    height: int
    bytes_read: int
    content_type: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_hash(reference: str) -> str:
    return hashlib.sha256(reference.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl_atomic(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    os.replace(temporary, path)


class TrainingEncoder:
    """Encode a conversation with one placeholder token per image."""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True,
        )
        self.template = copy.deepcopy(TEMPLATES["qwen2_vl"])
        self.template.fix_special_tokens(self.tokenizer)

    def encode(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[int, int]:
        processed = copy.deepcopy(messages)
        for message in processed:
            message["content"] = message["content"].replace(
                IMAGE_PLACEHOLDER,
                ONE_IMAGE_TOKEN,
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


def read_records(encoder: TrainingEncoder) -> list[Record]:
    records = []
    with SOURCE.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or set(row) != {"images", "messages"}
                or not isinstance(row["images"], list)
                or not isinstance(row["messages"], list)
            ):
                raise ValueError(
                    f"{SOURCE}:{index + 1} has an unexpected schema"
                )
            images = row["images"]
            messages = row["messages"]
            if not images or not all(
                isinstance(value, str) for value in images
            ):
                raise ValueError(
                    f"{SOURCE}:{index + 1} has invalid image references"
                )
            if not messages or not all(
                isinstance(message, dict)
                and set(message) == {"role", "content"}
                and isinstance(message["role"], str)
                and isinstance(message["content"], str)
                for message in messages
            ):
                raise ValueError(
                    f"{SOURCE}:{index + 1} has invalid messages"
                )
            placeholder_count = sum(
                message["content"].count(IMAGE_PLACEHOLDER)
                for message in messages
            )
            if placeholder_count != len(images):
                raise ValueError(
                    f"{SOURCE}:{index + 1} has {len(images)} images but "
                    f"{placeholder_count} placeholders"
                )
            total_tokens, label_tokens = encoder.encode(messages)
            records.append(
                Record(
                    index=index,
                    images=tuple(images),
                    messages=tuple(messages),
                    base_total_tokens=total_tokens,
                    label_tokens=label_tokens,
                )
            )
    return records


def parse_png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or not data.startswith(PNG_SIGNATURE):
        return None
    if data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return (width, height) if width > 0 and height > 0 else None


def parse_jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    position = 2
    while position + 1 < len(data):
        if data[position] != 0xFF:
            position += 1
            continue
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            break
        marker = data[position]
        position += 1
        if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if position + 2 > len(data):
            break
        segment_length = int.from_bytes(
            data[position : position + 2],
            "big",
        )
        if segment_length < 2 or position + segment_length > len(data):
            break
        if marker in JPEG_SOF_MARKERS and segment_length >= 7:
            height = int.from_bytes(
                data[position + 3 : position + 5],
                "big",
            )
            width = int.from_bytes(
                data[position + 5 : position + 7],
                "big",
            )
            return (width, height) if width > 0 and height > 0 else None
        position += segment_length
    return None


def parse_image_dimensions(data: bytes) -> tuple[int, int] | None:
    dimensions = parse_png_dimensions(data) or parse_jpeg_dimensions(data)
    if dimensions is not None:
        return dimensions
    try:
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
    except Exception:
        return None
    return (width, height) if width > 0 and height > 0 else None


def encoded_reference(reference: str) -> str:
    parts = urlsplit(reference)
    hostname = parts.hostname
    if hostname is None:
        raise ValueError("image reference has no hostname")
    ascii_hostname = hostname.encode("idna").decode("ascii")
    if parts.port is not None:
        ascii_hostname = f"{ascii_hostname}:{parts.port}"
    if parts.username is not None or parts.password is not None:
        raise ValueError("image reference must not contain user info")
    return urlunsplit(
        (
            parts.scheme,
            ascii_hostname,
            quote(
                parts.path,
                safe="/%:@-._~!$&'()*+,;=",
            ),
            quote(
                parts.query,
                safe="=&%/:?@-._~!$'()*+,;",
            ),
            "",
        )
    )


def sanitized_error_code(error: Exception) -> str:
    if isinstance(error, HTTPError):
        return f"http_{error.code}"
    if isinstance(error, (TimeoutError, URLError)):
        return "network_or_timeout"
    message = str(error)
    if "dimensions not found" in message:
        return "unsupported_or_incomplete_image_header"
    if "hostname" in message or "user info" in message:
        return "invalid_image_reference"
    return type(error).__name__


def fetch_image_dimension(
    reference: str,
    timeout_seconds: float,
) -> ImageDimension:
    attempts = (65536, 262144)
    last_error = "unknown image format"
    request_url = encoded_reference(reference)
    for byte_limit in attempts:
        request = Request(
            request_url,
            headers={
                "Accept-Encoding": "identity",
                "Range": f"bytes=0-{byte_limit - 1}",
                "User-Agent": "param-recommend-dataset-profiler/1.0",
            },
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                data = response.read(byte_limit)
                content_type = response.headers.get("Content-Type")
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            continue
        dimensions = parse_image_dimensions(data)
        if dimensions is not None:
            width, height = dimensions
            return ImageDimension(
                width=width,
                height=height,
                bytes_read=len(data),
                content_type=content_type,
            )
        last_error = (
            f"dimensions not found in first {len(data)} bytes"
        )
    raise ValueError(last_error)


def regularized_size(width: int, height: int) -> tuple[int, int]:
    area = width * height
    if area > IMAGE_MAX_PIXELS:
        resize_factor = math.sqrt(IMAGE_MAX_PIXELS / area)
        width = int(width * resize_factor)
        height = int(height * resize_factor)
    elif area < IMAGE_MIN_PIXELS:
        resize_factor = math.sqrt(IMAGE_MIN_PIXELS / area)
        width = int(width * resize_factor)
        height = int(height * resize_factor)

    width = max(width, 28)
    height = max(height, 28)
    if width / height > 200:
        width = height * 180
    if height / width > 200:
        height = width * 180
    return width, height


def visual_tokens(width: int, height: int) -> tuple[int, int, int]:
    regularized_width, regularized_height = regularized_size(
        width,
        height,
    )
    resized_height, resized_width = smart_resize(
        height=regularized_height,
        width=regularized_width,
        factor=PATCH_SIZE * MERGE_SIZE,
        min_pixels=PROCESSOR_MIN_PIXELS,
        max_pixels=PROCESSOR_MAX_PIXELS,
    )
    tokens = (
        (resized_height // PATCH_SIZE)
        * (resized_width // PATCH_SIZE)
        // (MERGE_SIZE**2)
    )
    return tokens, resized_width, resized_height


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def wilson_interval(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    ratio = successes / total
    denominator = 1 + (z**2 / total)
    center = (ratio + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            ratio * (1 - ratio) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-records",
        type=int,
        default=DEFAULT_SAMPLE_RECORDS,
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    arguments = parser.parse_args()

    encoder = TrainingEncoder()
    records = read_records(encoder)
    if not records:
        raise ValueError("dataset is empty")
    sample_size = min(max(1, arguments.sample_records), len(records))
    sample_indices = sorted(
        random.Random(SAMPLE_SEED).sample(
            range(len(records)),
            sample_size,
        )
    )
    sampled_records = [records[index] for index in sample_indices]
    unique_references = sorted(
        {
            reference
            for record in sampled_records
            for reference in record.images
        }
    )

    dimensions: dict[str, ImageDimension] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(
                fetch_image_dimension,
                reference,
                arguments.timeout_seconds,
            ): reference
            for reference in unique_references
        }
        for future in as_completed(futures):
            reference = futures[future]
            try:
                dimensions[reference] = future.result()
            except Exception as error:
                errors[reference] = sanitized_error_code(error)

    profile_rows = []
    complete_total_tokens = []
    complete_visual_tokens = []
    complete_records = 0
    for record in sampled_records:
        image_rows = []
        record_complete = True
        record_visual_tokens = 0
        for reference in record.images:
            image_hash = reference_hash(reference)
            dimension = dimensions.get(reference)
            if dimension is None:
                record_complete = False
                image_rows.append(
                    {
                        "reference_sha256": image_hash,
                        "error": errors.get(
                            reference,
                            "dimension unavailable",
                        ),
                    }
                )
                continue
            tokens, resized_width, resized_height = visual_tokens(
                dimension.width,
                dimension.height,
            )
            record_visual_tokens += tokens
            complete_visual_tokens.append(tokens)
            image_rows.append(
                {
                    "reference_sha256": image_hash,
                    "original_width": dimension.width,
                    "original_height": dimension.height,
                    "model_width": resized_width,
                    "model_height": resized_height,
                    "visual_tokens": tokens,
                    "header_bytes_read": dimension.bytes_read,
                    "content_type": dimension.content_type,
                }
            )

        estimated_total = None
        if record_complete:
            complete_records += 1
            estimated_total = (
                record.base_total_tokens
                + record_visual_tokens
                - len(record.images)
            )
            complete_total_tokens.append(estimated_total)
        profile_rows.append(
            {
                "dataset_id": DATASET_ID,
                "record_index": record.index,
                "base_total_tokens_one_image_pad_each": (
                    record.base_total_tokens
                ),
                "label_tokens": record.label_tokens,
                "image_count": len(record.images),
                "images": image_rows,
                "estimated_total_tokens": estimated_total,
                "complete": record_complete,
            }
        )

    if not complete_total_tokens:
        raise RuntimeError("no sampled record has complete image dimensions")

    all_base_tokens = [record.base_total_tokens for record in records]
    all_label_tokens = [record.label_tokens for record in records]
    cutoff_analysis = {}
    for cutoff in CUTOFF_GRID:
        truncated = sum(
            value > cutoff for value in complete_total_tokens
        )
        cutoff_analysis[str(cutoff)] = {
            "cutoff_len": cutoff,
            "sample_records_truncated": truncated,
            "sample_truncation_rate": (
                truncated / len(complete_total_tokens)
            ),
            "truncation_rate_wilson_95": wilson_interval(
                truncated,
                len(complete_total_tokens),
            ),
            "sample_tokens_retained_ratio": (
                sum(
                    min(value, cutoff)
                    for value in complete_total_tokens
                )
                / sum(complete_total_tokens)
            ),
        }

    summary = {
        "schema": "business_vl_dataset_profile/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "source": {
            "relative_path": str(SOURCE.relative_to(ROOT)),
            "sha256": sha256_file(SOURCE),
            "records": len(records),
        },
        "model": {
            "model_id": "qwen2_5_vl_7b_instruct",
            "local_path": str(MODEL_PATH),
            "template": "qwen2_vl@llamafactory",
            "tokenizer_local_only": True,
        },
        "image_processing_assumptions": {
            "image_max_pixels": IMAGE_MAX_PIXELS,
            "image_min_pixels": IMAGE_MIN_PIXELS,
            "processor_min_pixels": PROCESSOR_MIN_PIXELS,
            "processor_max_pixels": PROCESSOR_MAX_PIXELS,
            "patch_size": PATCH_SIZE,
            "merge_size": MERGE_SIZE,
            "visual_token_formula": (
                "(resized_height / patch_size) * "
                "(resized_width / patch_size) / merge_size^2"
            ),
        },
        "full_dataset_text_profile": {
            "records": len(records),
            "base_total_tokens_one_image_pad_each": distribution(
                all_base_tokens
            ),
            "label_tokens": distribution(all_label_tokens),
        },
        "sample": {
            "method": "deterministic_uniform_without_replacement",
            "seed": SAMPLE_SEED,
            "requested_records": arguments.sample_records,
            "sampled_records": len(sampled_records),
            "complete_records": complete_records,
            "unique_image_references": len(unique_references),
            "successful_unique_image_headers": len(dimensions),
            "failed_unique_image_headers": len(errors),
            "estimated_total_tokens": distribution(
                complete_total_tokens
            ),
            "visual_tokens_per_image_observation": distribution(
                complete_visual_tokens
            ),
            "cutoff_analysis": cutoff_analysis,
        },
        "privacy": {
            "raw_text_written": False,
            "raw_image_urls_written": False,
            "image_reference_sha256_written": True,
        },
        "limitations": [
            (
                "Visual-token statistics are sampled estimates; text-token "
                "statistics cover all records."
            ),
            (
                "This is a token-shape profile, not a validated VL memory or "
                "throughput prediction."
            ),
        ],
        "gpu_training_started": False,
    }
    write_jsonl_atomic(PROFILE_PATH, profile_rows)
    write_json_atomic(SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
