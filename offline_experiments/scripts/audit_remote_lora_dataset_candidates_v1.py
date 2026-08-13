#!/usr/bin/env python3
"""Audit downloaded remote SFT candidates for the LoRA memory campaign.

The audit is CPU-only.  It accepts only pure-text ``system/prompt/response``
records, profiles them with the exact Qwen3 ``qwen3_nothink`` training
template, and reports exact normalized-row overlap between dataset IDs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

from llamafactory.data.template import TEMPLATES

from common import ARTIFACT_DIR, ROOT, sha256_file, sha256_json, write_json


DEFAULT_INPUT = ROOT / "data" / "remote_candidate_screen_20260804"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_lora_remote_candidate_audit_v1.json"
DEFAULT_TOKENIZER = Path("/wanqing-models/Qwen3-8B")
MEDIA_KEYS = {"image", "images", "video", "videos", "audio", "audios"}
REQUIRED_FIELDS = ("system", "prompt", "response")


def _iter_objects(path: Path) -> Iterable[Any]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if isinstance(value, list):
                yield from value
            else:
                yield value


def _normalized(row: dict[str, str]) -> str:
    return "\0".join(row[field].strip() for field in REQUIRED_FIELDS)


def _percentile(values: list[int], probability: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return int(ordered[index])


class TrainingEncoder:
    def __init__(self, tokenizer_path: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True,
        )
        self.template = copy.deepcopy(TEMPLATES["qwen3_nothink"])
        self.template.fix_special_tokens(self.tokenizer)

    def encode(self, row: dict[str, str]) -> tuple[int, int]:
        conversation = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
        pairs = self.template.encode_multiturn(
            self.tokenizer,
            conversation,
            system=row["system"] or None,
            tools=None,
        )
        source_tokens = sum(len(source_ids) for source_ids, _ in pairs)
        label_tokens = sum(len(target_ids) for _, target_ids in pairs)
        if self.template.efficient_eos:
            label_tokens += 1
        return source_tokens + label_tokens, label_tokens


def _tokenizer_binding(path: Path) -> dict[str, Any]:
    names = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    files = {
        name: sha256_file(path / name)
        for name in names
        if (path / name).is_file()
    }
    return {
        "path": str(path.resolve()),
        "template": "qwen3_nothink",
        "files": files,
        "files_sha256": sha256_json(files),
    }


def audit(input_dir: Path, tokenizer_path: Path) -> dict[str, Any]:
    encoder = TrainingEncoder(tokenizer_path)
    token_cache: dict[str, tuple[int, int]] = {}
    fingerprints_by_dataset: dict[str, set[str]] = {}
    datasets: list[dict[str, Any]] = []

    paths = sorted(input_dir.glob("dataset-*/[0-9]*/publish/*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no downloaded publish JSONL files under {input_dir}")

    for path in paths:
        dataset_id = path.relative_to(input_dir).parts[0]
        objects = list(_iter_objects(path))
        schema_counts: Counter[str] = Counter()
        media_rows = 0
        normalized_rows: list[tuple[dict[str, str], str]] = []
        system_counts: Counter[str] = Counter()

        for value in objects:
            if not isinstance(value, dict):
                schema_counts["non_object"] += 1
                continue
            if set(value) & MEDIA_KEYS:
                media_rows += 1
            if all(isinstance(value.get(field), str) for field in REQUIRED_FIELDS):
                schema_counts["system_prompt_response"] += 1
                row = {field: value[field] for field in REQUIRED_FIELDS}
                normalized = _normalized(row)
                fingerprint = hashlib.sha256(normalized.encode()).hexdigest()
                normalized_rows.append((row, fingerprint))
                system_counts[hashlib.sha256(row["system"].encode()).hexdigest()] += 1
            elif isinstance(value.get("messages"), list):
                schema_counts["messages"] += 1
            elif "chosen" in value and "rejected" in value:
                schema_counts["preference"] += 1
            else:
                schema_counts["other"] += 1

        accepted = (
            bool(objects)
            and len(normalized_rows) == len(objects)
            and media_rows == 0
        )
        lengths: list[int] = []
        label_lengths: list[int] = []
        if accepted:
            for row, fingerprint in normalized_rows:
                encoded = token_cache.get(fingerprint)
                if encoded is None:
                    encoded = encoder.encode(row)
                    token_cache[fingerprint] = encoded
                total_tokens, label_tokens = encoded
                lengths.append(int(total_tokens))
                label_lengths.append(int(label_tokens))

        fingerprints = {fingerprint for _, fingerprint in normalized_rows}
        if accepted:
            fingerprints_by_dataset[dataset_id] = fingerprints
        dominant_system = system_counts.most_common(1)
        datasets.append(
            {
                "dataset_id": dataset_id,
                "accepted_pure_text_sft": accepted,
                "rejection_reasons": [
                    reason
                    for reason, rejected in (
                        ("empty", not objects),
                        ("mixed_or_unsupported_schema", len(normalized_rows) != len(objects)),
                        ("media_fields_present", media_rows > 0),
                    )
                    if rejected
                ],
                "local_path": str(path.resolve()),
                "remote_key": "datasets/" + str(path.relative_to(input_dir)),
                "file_size_bytes": path.stat().st_size,
                "file_sha256": sha256_file(path),
                "raw_rows": len(objects),
                "unique_normalized_rows": len(fingerprints),
                "within_dataset_duplicate_fraction": (
                    1.0 - len(fingerprints) / len(normalized_rows)
                    if normalized_rows
                    else None
                ),
                "media_rows": media_rows,
                "schema_counts": dict(sorted(schema_counts.items())),
                "dominant_system_sha256": dominant_system[0][0] if dominant_system else None,
                "dominant_system_fraction": (
                    dominant_system[0][1] / len(normalized_rows)
                    if dominant_system and normalized_rows
                    else None
                ),
                "token_profile": (
                    {
                        "rows": len(lengths),
                        "minimum": min(lengths),
                        "p50": _percentile(lengths, 0.50),
                        "p90": _percentile(lengths, 0.90),
                        "p95": _percentile(lengths, 0.95),
                        "p99": _percentile(lengths, 0.99),
                        "maximum": max(lengths),
                        "label_maximum": max(label_lengths),
                    }
                    if lengths
                    else None
                ),
            }
        )

    inverted: dict[str, list[str]] = defaultdict(list)
    for dataset_id, fingerprints in fingerprints_by_dataset.items():
        for fingerprint in fingerprints:
            inverted[fingerprint].append(dataset_id)
    intersections: Counter[tuple[str, str]] = Counter()
    for owners in inverted.values():
        owners = sorted(set(owners))
        for left_index, left in enumerate(owners):
            for right in owners[left_index + 1 :]:
                intersections[(left, right)] += 1

    overlaps: list[dict[str, Any]] = []
    for (left, right), shared in intersections.items():
        left_count = len(fingerprints_by_dataset[left])
        right_count = len(fingerprints_by_dataset[right])
        union = left_count + right_count - shared
        overlaps.append(
            {
                "left_dataset_id": left,
                "right_dataset_id": right,
                "shared_rows": shared,
                "left_containment": shared / left_count,
                "right_containment": shared / right_count,
                "jaccard": shared / union,
            }
        )
    overlaps.sort(
        key=lambda row: (
            -max(float(row["left_containment"]), float(row["right_containment"])),
            -int(row["shared_rows"]),
            str(row["left_dataset_id"]),
            str(row["right_dataset_id"]),
        )
    )

    report: dict[str, Any] = {
        "schema": "sft_h800_lora_remote_candidate_audit/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "input_dir": str(input_dir.resolve()),
        "tokenizer_binding": _tokenizer_binding(tokenizer_path),
        "candidate_files": len(paths),
        "accepted_pure_text_sft_files": sum(
            row["accepted_pure_text_sft"] is True for row in datasets
        ),
        "unique_encoded_row_cache_entries": len(token_cache),
        "datasets": datasets,
        "exact_normalized_row_overlaps": overlaps,
        "selection_contract": {
            "dataset_id_must_be_unique": True,
            "accepted_pure_text_sft_required": True,
            "cross_dataset_containment_limit": 0.05,
            "same_dominant_system_is_not_alone_proof_of_independence": True,
            "final_source_revision_and_sha256_must_be_frozen_before_gpu": True,
        },
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(args.input, args.tokenizer)
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "candidate_files": report["candidate_files"],
                "accepted_pure_text_sft_files": report["accepted_pure_text_sft_files"],
                "unique_encoded_row_cache_entries": report[
                    "unique_encoded_row_cache_entries"
                ],
                "overlap_pairs": len(report["exact_normalized_row_overlaps"]),
                "report_sha256": report["report_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
