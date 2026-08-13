#!/usr/bin/env python3
"""Materialize three pre-frozen S3 profiles for bounded-memory v2 acceptance.

This is CPU-only.  It validates complete-source SHA bindings, preserves each
source's token-length CDF with a deterministic rank grid, and emits a separate
longest-row dataset for the pre-registered tail-forced safety point.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from llamafactory.data.template import TEMPLATES
from transformers import AutoTokenizer

from common import (
    ARTIFACT_DIR,
    DATA_DIR,
    percentile,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)


SCHEMA = "sft_h800_bounded_memory_v2_fresh_data/v1"
SELECTION_SCHEMA = "sft_h800_bounded_memory_v2_fresh_selection/v1"
DEFAULT_SELECTION = ARTIFACT_DIR / "h800_bounded_memory_v2_fresh_selection_v1.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_bounded_memory_v2_fresh_data_v1.json"
OUTPUT_DIR = DATA_DIR / "bounded_memory_v2_fresh_holdout_v1"
ARTIFACT_ROOT = ARTIFACT_DIR / "bounded_memory_v2_fresh_holdout_v1"
PROFILE_DIR = ARTIFACT_ROOT / "profiles"
SPLIT_MANIFEST = ARTIFACT_ROOT / "split_manifest.json"
PROCESSOR_CONTRACT = ARTIFACT_ROOT / "processor_contract.json"
MODEL_PATHS = {
    "qwen3_8b": Path("/wanqing-models/Qwen3-8B"),
    "qwen3_14b": Path("/wanqing-models/Qwen3-14B"),
}


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _normalize(value: Any, *, path: Path, line_number: int) -> dict[str, str]:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number} is not an SFT object")
    required = ("system", "prompt", "response")
    if not all(isinstance(value.get(field), str) for field in required):
        raise ValueError(f"{path}:{line_number} lacks string system/prompt/response")
    return {field: str(value[field]) for field in required}


class TrainingEncoder:
    def __init__(self, model_path: Path, template_name: str = "qwen3_nothink") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True,
        )
        self.template = copy.deepcopy(TEMPLATES[template_name])
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
        "model_path": str(path.resolve()),
        "files": files,
        "files_sha256": sha256_json(files),
        "template": "qwen3_nothink",
    }


def _rank_grid(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count < 2 or count > len(rows):
        raise ValueError("rank-grid count must be in [2, source rows]")
    ordered = sorted(rows, key=lambda row: (row["total_tokens"], row["line_number"]))
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise ValueError("rank-grid selection produced duplicate indices")
    return sorted((ordered[index] for index in indices), key=lambda row: row["line_number"])


def _statistics(lengths: list[int], cutoff_len: int) -> dict[str, Any]:
    clipped = [min(length, cutoff_len) for length in lengths]
    return {
        "rows": len(lengths),
        "minimum_total_tokens": min(lengths),
        "mean_total_tokens": sum(lengths) / len(lengths),
        "p50_total_tokens": percentile(lengths, 50),
        "p90_total_tokens": percentile(lengths, 90),
        "p95_total_tokens": percentile(lengths, 95),
        "p99_total_tokens": percentile(lengths, 99),
        "maximum_total_tokens": max(lengths),
        "cutoff_len": cutoff_len,
        "truncation_fraction": sum(length > cutoff_len for length in lengths) / len(lengths),
        "tokens_retained_ratio": sum(clipped) / sum(lengths),
    }


def prepare(selection_path: Path) -> dict[str, Any]:
    selection = read_json(selection_path)
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("bounded v2 fresh selection schema mismatch")
    if (
        selection.get("challenger_frozen_before_source_selection") is not True
        or selection.get("selection_frozen_before_gpu_results") is not True
        or selection.get("gpu_training_started") is not False
    ):
        raise ValueError("fresh selection governance contract is not frozen")

    encoders = {model_id: TrainingEncoder(path) for model_id, path in MODEL_PATHS.items()}
    processor = {
        "schema": "sft_qwen3_text_processor_contract/v4",
        "template": "qwen3_nothink",
        "models": {model_id: _tokenizer_binding(path) for model_id, path in MODEL_PATHS.items()},
        "shared_profile_allowed_only_if_all_row_lengths_identical": True,
    }

    profiles: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    checked_rows = 0
    for spec in selection["profiles"]:
        source = Path(spec["source_path"])
        if not source.is_file() or sha256_file(source) != spec["source_sha256"]:
            raise ValueError(f"frozen source is missing or changed: {source}")
        encoded_source: list[dict[str, Any]] = []
        with source.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                row = _normalize(json.loads(line), path=source, line_number=line_number)
                encoded = {model_id: encoder.encode(row) for model_id, encoder in encoders.items()}
                if len(set(encoded.values())) != 1:
                    raise ValueError(
                        f"Qwen3 tokenizer/template lengths differ at {source}:{line_number}: {encoded}"
                    )
                total_tokens, label_tokens = next(iter(encoded.values()))
                encoded_source.append(
                    {
                        "line_number": line_number,
                        "row": row,
                        "total_tokens": total_tokens,
                        "label_tokens": label_tokens,
                    }
                )
        if len(encoded_source) != int(spec["source_rows"]):
            raise ValueError(f"frozen source row count changed: {source}")
        checked_rows += len(encoded_source)

        selected = _rank_grid(encoded_source, int(spec["selection"]["rows"]))
        tail = sorted(
            encoded_source,
            key=lambda row: (-row["total_tokens"], row["line_number"]),
        )[: int(spec["selection"]["tail_forced_rows"])]
        cutoff_len = int(spec["cutoff_len"])
        data_path = OUTPUT_DIR / f"{spec['profile_id']}.jsonl"
        tail_data_path = OUTPUT_DIR / f"{spec['profile_id']}__tail_forced.jsonl"
        profile_path = PROFILE_DIR / f"{spec['profile_id']}.qwen3_nothink.jsonl"
        tail_profile_path = PROFILE_DIR / f"{spec['profile_id']}__tail_forced.qwen3_nothink.jsonl"

        def data_rows(items: list[dict[str, Any]], suffix: str = "") -> list[dict[str, Any]]:
            return [
                {
                    "sample_id": f"{spec['profile_id']}{suffix}:source_line:{item['line_number']}",
                    **item["row"],
                }
                for item in items
            ]

        def profile_rows(items: list[dict[str, Any]], suffix: str = "") -> list[dict[str, Any]]:
            return [
                {
                    "sample_id": f"{spec['profile_id']}{suffix}:source_line:{item['line_number']}",
                    "total_tokens": item["total_tokens"],
                    "label_tokens": item["label_tokens"],
                    "turns": 3 if item["row"]["system"] else 2,
                    "assistant_turns": 1,
                }
                for item in items
            ]

        _atomic_jsonl(data_path, data_rows(selected))
        _atomic_jsonl(tail_data_path, data_rows(tail, "__tail_forced"))
        _atomic_jsonl(profile_path, profile_rows(selected))
        _atomic_jsonl(tail_profile_path, profile_rows(tail, "__tail_forced"))

        selected_lengths = [int(item["total_tokens"]) for item in selected]
        tail_lengths = [int(item["total_tokens"]) for item in tail]
        profile = {
            **{key: value for key, value in spec.items() if key != "selection"},
            "dataset_id": spec["profile_id"],
            "data_path": str(data_path.resolve()),
            "data_sha256": sha256_file(data_path),
            "profile_path": str(profile_path.resolve()),
            "profile_sha256": sha256_file(profile_path),
            "profile_statistics": _statistics(selected_lengths, cutoff_len),
            "tail_forced_dataset_id": f"{spec['profile_id']}__tail_forced",
            "tail_forced_data_path": str(tail_data_path.resolve()),
            "tail_forced_data_sha256": sha256_file(tail_data_path),
            "tail_forced_profile_path": str(tail_profile_path.resolve()),
            "tail_forced_profile_sha256": sha256_file(tail_profile_path),
            "tail_forced_profile_statistics": _statistics(tail_lengths, cutoff_len),
        }
        profiles.append(profile)
        split_rows.append(
            {
                "profile_id": spec["profile_id"],
                "source_dataset_id": spec["source_dataset_id"],
                "source_path": str(source.resolve()),
                "source_sha256": sha256_file(source),
                "source_rows": len(encoded_source),
                "selection_method": spec["selection"]["method"],
                "selected_source_line_numbers": [item["line_number"] for item in selected],
                "tail_forced_source_line_numbers": [item["line_number"] for item in tail],
                "data_path": str(data_path.resolve()),
                "data_sha256": sha256_file(data_path),
                "tail_forced_data_path": str(tail_data_path.resolve()),
                "tail_forced_data_sha256": sha256_file(tail_data_path),
                "profile_path": str(profile_path.resolve()),
                "profile_sha256": sha256_file(profile_path),
                "tail_forced_profile_path": str(tail_profile_path.resolve()),
                "tail_forced_profile_sha256": sha256_file(tail_profile_path),
            }
        )

    processor["cross_model_length_identity"] = {
        "checked_rows": checked_rows,
        "all_identical": True,
    }
    write_json(PROCESSOR_CONTRACT, processor)
    split_manifest: dict[str, Any] = {
        "schema": "sft_h800_bounded_memory_v2_fresh_split_manifest/v1",
        "selection_binding": {
            "path": str(selection_path.resolve()),
            "sha256": sha256_file(selection_path),
            "payload_sha256": sha256_json(selection),
        },
        "fit_overlap": {
            "selected_sources_used_for_v2_fit": False,
            "selected_sources_used_for_prior_holdout": False,
            "policy": "complete S3 source and all derived rows remain prospective holdout only",
        },
        "splits": split_rows,
    }
    split_manifest["report_sha256"] = sha256_json(split_manifest)
    write_json(SPLIT_MANIFEST, split_manifest)

    bundle: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": selection["campaign_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "materialized_cpu_only_before_prediction_freeze",
        "gpu_training_started": False,
        "model_fitting_performed": False,
        "profiles": profiles,
        "candidate_policy": selection["frozen_candidate_policy"],
        "source_bindings": {
            "selection": {"path": str(selection_path.resolve()), "sha256": sha256_file(selection_path)},
            "split_manifest": {
                "path": str(SPLIT_MANIFEST.resolve()),
                "sha256": sha256_file(SPLIT_MANIFEST),
                "report_sha256": split_manifest["report_sha256"],
            },
            "processor_contract": {
                "path": str(PROCESSOR_CONTRACT.resolve()),
                "sha256": sha256_file(PROCESSOR_CONTRACT),
            },
        },
    }
    bundle["report_sha256"] = sha256_json(bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    bundle = prepare(args.selection)
    write_json(args.output, bundle)
    print(
        json.dumps(
            {
                "bundle": str(args.output.resolve()),
                "report_sha256": bundle["report_sha256"],
                "profiles": [
                    {
                        "profile_id": row["profile_id"],
                        **row["profile_statistics"],
                        "tail_forced": row["tail_forced_profile_statistics"],
                    }
                    for row in bundle["profiles"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
