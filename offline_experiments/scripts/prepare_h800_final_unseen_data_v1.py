#!/usr/bin/env python3
"""Materialize the two pre-frozen unseen H800 holdout data profiles.

The source selection, row counts and reservoir seeds come exclusively from
``h800_final_unseen_holdout_selection_v1.json``.  This command performs no
model fitting and launches no GPU work.  Both Qwen3 tokenizer/template paths
must produce identical lengths before a shared profile is emitted.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
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


SCHEMA = "sft_h800_final_unseen_business_data/v1"
DEFAULT_SELECTION = ARTIFACT_DIR / "h800_final_unseen_holdout_selection_v1.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_final_unseen_business_data_v1.json"
OUTPUT_DIR = DATA_DIR / "final_unseen_holdout_v1"
ARTIFACT_ROOT = ARTIFACT_DIR / "final_unseen_holdout_v1"
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
                output.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
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
        raise ValueError(f"{path}:{line_number} is not an Alpaca record")
    required = ("system", "prompt", "response")
    if not all(isinstance(value.get(field), str) for field in required):
        raise ValueError(
            f"{path}:{line_number} lacks string system/prompt/response"
        )
    return {field: str(value[field]) for field in required}


def _reservoir(
    path: Path, *, count: int, seed: int
) -> tuple[list[tuple[int, dict[str, str]]], int]:
    rng = random.Random(seed)
    selected: list[tuple[int, dict[str, str]]] = []
    seen = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = _normalize(json.loads(line), path=path, line_number=line_number)
            item = (line_number, row)
            if seen < count:
                selected.append(item)
            else:
                replacement = rng.randint(0, seen)
                if replacement < count:
                    selected[replacement] = item
            seen += 1
    if seen < count:
        raise ValueError(f"{path} has only {seen} rows; {count} required")
    return sorted(selected, key=lambda item: item[0]), seen


class TrainingEncoder:
    def __init__(self, model_path: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
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
        "model_path": str(path.resolve()),
        "files": files,
        "files_sha256": sha256_json(files),
        "template": "qwen3_nothink",
    }


def prepare(selection_path: Path) -> dict[str, Any]:
    selection = read_json(selection_path)
    if selection.get("schema") != "sft_h800_final_unseen_holdout_selection/v1":
        raise ValueError("unseen holdout selection schema mismatch")
    if selection.get("frozen_before_challenger_coefficient_inspection") is not True:
        raise ValueError("unseen profile selection was not frozen before fitting")
    encoders = {
        model_id: TrainingEncoder(path) for model_id, path in MODEL_PATHS.items()
    }
    processor = {
        "schema": "sft_qwen3_text_processor_contract/v3",
        "template": "qwen3_nothink",
        "models": {
            model_id: _tokenizer_binding(path)
            for model_id, path in MODEL_PATHS.items()
        },
        "shared_profile_allowed_only_if_all_row_lengths_identical": True,
    }

    profiles: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    for spec in selection["profiles"]:
        source = Path(spec["source_path"])
        if not source.is_file() or sha256_file(source) != spec["source_sha256"]:
            raise ValueError(f"frozen source is missing or changed: {source}")
        selected, source_rows = _reservoir(
            source,
            count=int(spec["selection"]["rows"]),
            seed=int(spec["selection"]["seed"]),
        )
        if source_rows != int(spec["source_rows"]):
            raise ValueError(f"frozen source row count changed: {source}")
        data_rows: list[dict[str, Any]] = []
        profile_rows: list[dict[str, Any]] = []
        for line_number, row in selected:
            sample_id = f"{spec['profile_id']}:source_line:{line_number}"
            encoded = {
                model_id: encoder.encode(row)
                for model_id, encoder in encoders.items()
            }
            if len(set(encoded.values())) != 1:
                raise ValueError(
                    f"Qwen3 tokenizer/template lengths differ for {sample_id}: {encoded}"
                )
            total_tokens, label_tokens = next(iter(encoded.values()))
            data_rows.append({"sample_id": sample_id, **row})
            profile_rows.append(
                {
                    "sample_id": sample_id,
                    "total_tokens": total_tokens,
                    "label_tokens": label_tokens,
                    "turns": 3 if row["system"] else 2,
                    "assistant_turns": 1,
                }
            )
        data_path = OUTPUT_DIR / f"{spec['profile_id']}.jsonl"
        profile_path = PROFILE_DIR / f"{spec['profile_id']}.qwen3_nothink.jsonl"
        _atomic_jsonl(data_path, data_rows)
        _atomic_jsonl(profile_path, profile_rows)
        lengths = [int(row["total_tokens"]) for row in profile_rows]
        split = {
            "profile_id": spec["profile_id"],
            "source_path": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "source_rows": source_rows,
            "selection_seed": int(spec["selection"]["seed"]),
            "selected_source_line_numbers": [line for line, _ in selected],
            "selected_rows": len(selected),
            "data_path": str(data_path.resolve()),
            "data_sha256": sha256_file(data_path),
            "profile_path": str(profile_path.resolve()),
            "profile_sha256": sha256_file(profile_path),
        }
        splits.append(split)
        profiles.append(
            {
                **{key: value for key, value in spec.items() if key != "selection"},
                "dataset_id": spec["profile_id"],
                "data_path": str(data_path.resolve()),
                "data_sha256": sha256_file(data_path),
                "profile_path": str(profile_path.resolve()),
                "profile_sha256": sha256_file(profile_path),
                "profile_statistics": {
                    "rows": len(lengths),
                    "mean_total_tokens": sum(lengths) / len(lengths),
                    "p50_total_tokens": percentile(lengths, 50),
                    "p90_total_tokens": percentile(lengths, 90),
                    "p99_total_tokens": percentile(lengths, 99),
                    "maximum_total_tokens": max(lengths),
                },
            }
        )
    processor["cross_model_length_identity"] = {
        "checked_rows": sum(row["selected_rows"] for row in splits),
        "all_identical": True,
    }
    write_json(PROCESSOR_CONTRACT, processor)
    split_manifest = {
        "schema": "sft_h800_final_unseen_split_manifest/v1",
        "selection_binding": {
            "path": str(selection_path.resolve()),
            "sha256": sha256_file(selection_path),
            "payload_sha256": sha256_json(selection),
        },
        "fit_overlap": {
            "selected_source_rows_used_for_fit": False,
            "selected_dataset_ids_used_for_fit": False,
            "policy": "complete source file and derived profile are holdout-only",
        },
        "splits": splits,
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
            "selection": {
                "path": str(selection_path.resolve()),
                "sha256": sha256_file(selection_path),
                "payload_sha256": sha256_json(selection),
            },
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
