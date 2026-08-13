#!/usr/bin/env python3
"""Freeze four real-business H800 holdout slices and Qwen3 token profiles.

This is a CPU-only preparation step.  It deterministically selects 1,000 rows
from each source, normalizes them to the exact Alpaca schema consumed by the
training runtime, and profiles those same rows with the local Qwen3 tokenizer
and ``qwen3_nothink`` template.  It never launches a GPU process.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable

from transformers import AutoTokenizer
from llamafactory.data.template import TEMPLATES

from common import ARTIFACT_DIR, DATA_DIR, ROOT, percentile, sha256_file, sha256_json, write_json


SCHEMA = "sft_h800_fresh_business_data_bundle/v2"
ROW_COUNT = 1000
SEED = 20260802
PROFILE_DIR = ARTIFACT_DIR / "fresh_holdout_v2" / "profiles"
DATA_OUTPUT_DIR = DATA_DIR / "fresh_holdout_v2"
PROCESSOR_CONTRACT = ARTIFACT_DIR / "fresh_holdout_v2" / "processor_contract.json"
SPLIT_MANIFEST = ARTIFACT_DIR / "fresh_holdout_v2" / "split_manifest.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_fresh_business_data_bundle_v2.json"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    source: Path
    source_shape: str
    model_id: str
    model_path: Path
    training_mode: str
    cutoff_len: int
    transition: str
    dataset_category: str
    business_scene: str


BUSINESS_ROOT = ROOT.parent / "real_business_validation_20260730"
SPECS = (
    DatasetSpec(
        dataset_id="fresh_business_education_1000",
        source=BUSINESS_ROOT / "datasets" / "dataset-yt0jrk-1775555547" / "2" / "publish" / "dataset-yt0jrk-1775555547-V2.jsonl",
        source_shape="singleton_wrapped_alpaca",
        model_id="qwen3_8b",
        model_path=Path("/wanqing-models/Qwen3-8B"),
        training_mode="lora",
        cutoff_len=4096,
        transition="1_to_2",
        dataset_category="longtail",
        business_scene="education merchant SKU classification",
    ),
    DatasetSpec(
        dataset_id="fresh_business_relevance_1000",
        source=BUSINESS_ROOT / "datasets" / "dataset-flieht-1763953980" / "4" / "publish" / "dataset-flieht-1763953980-V4.jsonl",
        source_shape="singleton_wrapped_alpaca",
        model_id="qwen3_14b",
        model_path=Path("/wanqing-models/Qwen3-14B"),
        training_mode="full",
        cutoff_len=4096,
        transition="2_to_4",
        dataset_category="longtail",
        business_scene="search/query relevance classification",
    ),
    DatasetSpec(
        dataset_id="fresh_business_title_1000",
        source=BUSINESS_ROOT / "recommendations" / "qwen35_lora_1x4090_20260731" / "data" / "dataset_177870.jsonl",
        source_shape="alpaca",
        model_id="qwen3_8b",
        model_path=Path("/wanqing-models/Qwen3-8B"),
        training_mode="full",
        cutoff_len=512,
        transition="2_to_4",
        dataset_category="short",
        business_scene="short-video title selection",
    ),
    DatasetSpec(
        dataset_id="fresh_business_audience_1000",
        source=BUSINESS_ROOT / "recommendations" / "qwen35_lora_1x4090_20260731" / "data" / "dataset_71014.jsonl",
        source_shape="alpaca",
        model_id="qwen3_14b",
        model_path=Path("/wanqing-models/Qwen3-14B"),
        training_mode="lora",
        cutoff_len=2048,
        transition="1_to_2",
        dataset_category="longtail",
        business_scene="short-video audience/title generation",
    ),
)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
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
        raise ValueError(f"{path}:{line_number} is not an Alpaca record")
    required = ("system", "prompt", "response")
    if not all(isinstance(value.get(field), str) for field in required):
        raise ValueError(f"{path}:{line_number} lacks string system/prompt/response")
    return {field: value[field] for field in required}


def _reservoir(path: Path, *, count: int, seed: int) -> tuple[list[tuple[int, dict[str, str]]], int]:
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
            model_path, trust_remote_code=True, use_fast=True, local_files_only=True
        )
        self.template = copy.deepcopy(TEMPLATES["qwen3_nothink"])
        self.template.fix_special_tokens(self.tokenizer)

    def encode(self, row: dict[str, str]) -> tuple[int, int]:
        conversation = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
        pairs = self.template.encode_multiturn(
            self.tokenizer, conversation, system=row["system"] or None, tools=None
        )
        source_tokens = sum(len(source_ids) for source_ids, _ in pairs)
        label_tokens = sum(len(target_ids) for _, target_ids in pairs)
        if self.template.efficient_eos:
            label_tokens += 1
        return source_tokens + label_tokens, label_tokens


def _tokenizer_binding(model_path: Path) -> dict[str, Any]:
    names = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    manifest = {
        name: sha256_file(model_path / name)
        for name in names
        if (model_path / name).is_file()
    }
    return {
        "model_path": str(model_path.resolve()),
        "files": manifest,
        "files_sha256": sha256_json(manifest),
        "template": "qwen3_nothink",
    }


def prepare(*, row_count: int = ROW_COUNT, seed: int = SEED) -> dict[str, Any]:
    missing = [str(spec.source) for spec in SPECS if not spec.source.is_file()]
    if missing:
        raise FileNotFoundError(f"business source files are missing: {missing}")
    encoders: dict[Path, TrainingEncoder] = {}
    processor_models = {
        spec.model_id: _tokenizer_binding(spec.model_path) for spec in SPECS
    }
    processor_contract = {
        "schema": "sft_qwen3_text_processor_contract/v2",
        "template": "qwen3_nothink",
        "profile_row_fields": [
            "sample_id", "total_tokens", "label_tokens", "turns", "assistant_turns"
        ],
        "models": processor_models,
    }
    write_json(PROCESSOR_CONTRACT, processor_contract)

    split_rows: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(SPECS):
        selected, source_rows = _reservoir(
            spec.source, count=row_count, seed=seed + spec_index * 1009
        )
        data_rows: list[dict[str, Any]] = []
        profile_rows: list[dict[str, Any]] = []
        if spec.model_path not in encoders:
            encoders[spec.model_path] = TrainingEncoder(spec.model_path)
        encoder = encoders[spec.model_path]
        for line_number, source_row in selected:
            sample_id = f"{spec.dataset_id}:source_line:{line_number}"
            data_rows.append({"sample_id": sample_id, **source_row})
            total_tokens, label_tokens = encoder.encode(source_row)
            profile_rows.append(
                {
                    "sample_id": sample_id,
                    "total_tokens": total_tokens,
                    "label_tokens": label_tokens,
                    "turns": 3 if source_row["system"] else 2,
                    "assistant_turns": 1,
                }
            )
        data_path = DATA_OUTPUT_DIR / f"{spec.dataset_id}.jsonl"
        profile_path = PROFILE_DIR / f"{spec.dataset_id}.qwen3_nothink.jsonl"
        _atomic_jsonl(data_path, data_rows)
        _atomic_jsonl(profile_path, profile_rows)
        lengths = [row["total_tokens"] for row in profile_rows]
        split_rows.append(
            {
                "dataset_id": spec.dataset_id,
                "source_path": str(spec.source.resolve()),
                "source_sha256": sha256_file(spec.source),
                "source_rows": source_rows,
                "selection_seed": seed + spec_index * 1009,
                "selected_source_line_numbers": [line for line, _ in selected],
                "selected_rows": len(selected),
                "data_path": str(data_path.resolve()),
                "data_sha256": sha256_file(data_path),
                "profile_path": str(profile_path.resolve()),
                "profile_sha256": sha256_file(profile_path),
            }
        )
        scenarios.append(
            {
                "scenario_id": f"{spec.dataset_id}__{spec.model_id}_{spec.training_mode}",
                "dataset_profile_id": spec.dataset_id,
                "dataset_id": spec.dataset_id,
                "data_path": str(data_path.resolve()),
                "data_sha256": sha256_file(data_path),
                "profile_path": str(profile_path.resolve()),
                "profile_sha256": sha256_file(profile_path),
                "model_id": spec.model_id,
                "model_path": str(spec.model_path.resolve()),
                "training_mode": spec.training_mode,
                "cutoff_len": spec.cutoff_len,
                "target_gbs": 64,
                "scale_out_transition": spec.transition,
                "dataset_category": spec.dataset_category,
                "business_scene": spec.business_scene,
                "profile_statistics": {
                    "rows": len(lengths),
                    "mean_total_tokens": sum(lengths) / len(lengths),
                    "p50_total_tokens": percentile(lengths, 50),
                    "p90_total_tokens": percentile(lengths, 90),
                    "p99_total_tokens": percentile(lengths, 99),
                    "max_total_tokens": max(lengths),
                    "truncation_rate_at_cutoff": sum(
                        value > spec.cutoff_len for value in lengths
                    ) / len(lengths),
                },
            }
        )
    canonical_ids: set[str] = set()
    canonical_path = ARTIFACT_DIR / "canonical_h800_observations.jsonl"
    if canonical_path.is_file():
        with canonical_path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                observation = json.loads(line)
                dataset_id = (observation.get("scenario") or {}).get("dataset_id")
                if dataset_id:
                    canonical_ids.add(str(dataset_id))
    selected_ids = {spec.dataset_id for spec in SPECS}
    overlap = sorted(selected_ids & canonical_ids)
    if overlap:
        raise ValueError(f"fresh business dataset IDs overlap canonical observations: {overlap}")
    split_manifest = {
        "schema": "sft_h800_fresh_split_manifest/v2",
        "campaign_seed": seed,
        "selection_unit": "source line",
        "prior_modeling_overlap_check": {
            "selected_dataset_ids": sorted(selected_ids),
            "canonical_observation_dataset_ids_sha256": sha256_json(sorted(canonical_ids)),
            "matched_existing_canonical_observation_dataset_ids": overlap,
            "passed": not overlap,
            "note": "These business dataset IDs are absent from canonical_h800_observations.jsonl; only post-freeze GPU outcomes are held out.",
        },
        "splits": split_rows,
    }
    write_json(SPLIT_MANIFEST, split_manifest)
    bundle = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "row_count_per_scenario": row_count,
        "processor_contract": {
            "path": str(PROCESSOR_CONTRACT.resolve()),
            "sha256": sha256_file(PROCESSOR_CONTRACT),
        },
        "split_manifest": {
            "path": str(SPLIT_MANIFEST.resolve()),
            "sha256": sha256_file(SPLIT_MANIFEST),
        },
        "runtime_dataset_registry": {
            "path": str((DATA_DIR / "dataset_info.json").resolve()),
            "sha256": sha256_file(DATA_DIR / "dataset_info.json"),
            "registered_dataset_ids": [spec.dataset_id for spec in SPECS],
        },
        "scenarios": scenarios,
    }
    bundle["report_sha256"] = sha256_json(bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=ROW_COUNT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.rows <= 0:
        raise SystemExit("--rows must be positive")
    bundle = prepare(row_count=args.rows, seed=args.seed)
    write_json(args.output, bundle)
    print(f"wrote {args.output}; scenarios={len(bundle['scenarios'])}; rows_each={args.rows}")


if __name__ == "__main__":
    main()
