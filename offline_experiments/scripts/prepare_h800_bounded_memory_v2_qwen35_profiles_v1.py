#!/usr/bin/env python3
"""Build Qwen3.5-specific profile aliases for the three fresh S3 datasets."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from llamafactory.data.template import TEMPLATES
from transformers import AutoTokenizer

from common import ARTIFACT_DIR, percentile, read_json, sha256_file, sha256_json, write_json


DEFAULT_BUNDLE = ARTIFACT_DIR / "h800_bounded_memory_v2_fresh_data_v1.json"
DEFAULT_INVENTORY = ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_v1.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_bounded_memory_v2_qwen35_profiles_v1.json"
PROFILE_DIR = ARTIFACT_DIR / "bounded_memory_v2_fresh_holdout_v1" / "profiles"
MODEL_PATH = Path("/wanqing-models/Qwen3.5-4B")


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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


def build(bundle_path: Path, inventory_path: Path) -> dict[str, Any]:
    bundle = read_json(bundle_path)
    inventory = read_json(inventory_path)
    if bundle.get("schema") != "sft_h800_bounded_memory_v2_fresh_data/v1":
        raise ValueError("fresh data bundle schema mismatch")
    models = {str(row["id"]): row for row in inventory["models"]}
    model = models.get("qwen3p5_4b")
    if not model or model.get("path") != str(MODEL_PATH):
        raise ValueError("Qwen3.5-4B transfer inventory binding missing")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    template = copy.deepcopy(TEMPLATES["qwen3_5_nothink"])
    template.fix_special_tokens(tokenizer)
    reports = []
    for profile in bundle["profiles"]:
        data_path = Path(profile["data_path"])
        if sha256_file(data_path) != profile["data_sha256"]:
            raise ValueError(f"fresh data changed: {data_path}")
        alias_id = f"{profile['profile_id']}__qwen35"
        output_path = PROFILE_DIR / f"{alias_id}.qwen3_nothink.jsonl"
        rows: list[dict[str, Any]] = []
        lengths: list[int] = []
        with data_path.open(encoding="utf-8") as source:
            for line in source:
                item = json.loads(line)
                messages = [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["response"]},
                ]
                pairs = template.encode_multiturn(
                    tokenizer,
                    messages,
                    system=item["system"] or None,
                    tools=None,
                )
                source_tokens = sum(len(source_ids) for source_ids, _ in pairs)
                label_tokens = sum(len(target_ids) for _, target_ids in pairs)
                if template.efficient_eos:
                    label_tokens += 1
                total_tokens = source_tokens + label_tokens
                lengths.append(total_tokens)
                rows.append(
                    {
                        "sample_id": f"{alias_id}:{item['sample_id']}",
                        "total_tokens": total_tokens,
                        "label_tokens": label_tokens,
                        "turns": 3 if item["system"] else 2,
                        "assistant_turns": 1,
                    }
                )
        _atomic_jsonl(output_path, rows)
        reports.append(
            {
                "dataset_id": alias_id,
                "source_profile_id": profile["profile_id"],
                "data_path": str(data_path.resolve()),
                "data_sha256": profile["data_sha256"],
                "profile_path": str(output_path.resolve()),
                "profile_sha256": sha256_file(output_path),
                "profile_tokenizer_id": "qwen3p5_4b_local",
                "profile_template_id": "qwen3_5_nothink",
                "statistics": {
                    "rows": len(lengths),
                    "mean_total_tokens": sum(lengths) / len(lengths),
                    "p50_total_tokens": percentile(lengths, 50),
                    "p90_total_tokens": percentile(lengths, 90),
                    "p99_total_tokens": percentile(lengths, 99),
                    "maximum_total_tokens": max(lengths),
                },
            }
        )
    tokenizer_files = {
        name: sha256_file(MODEL_PATH / name)
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
        if (MODEL_PATH / name).is_file()
    }
    report: dict[str, Any] = {
        "schema": "sft_h800_bounded_memory_v2_qwen35_profiles/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "transfer_diagnostic_profiles_frozen_before_gpu",
        "gpu_training_started": False,
        "profiles": reports,
        "processor_binding": {
            "model_path": str(MODEL_PATH.resolve()),
            "template": "qwen3_5_nothink",
            "tokenizer_files": tokenizer_files,
            "tokenizer_files_sha256": sha256_json(tokenizer_files),
        },
        "source_bindings": {
            "fresh_data_bundle": {"path": str(bundle_path.resolve()), "sha256": sha256_file(bundle_path)},
            "transfer_model_inventory": {"path": str(inventory_path.resolve()), "sha256": sha256_file(inventory_path)},
        },
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--model-inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.bundle, args.model_inventory)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
