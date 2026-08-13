#!/usr/bin/env python3
"""Create a privacy-preserving token-length profile for baseline_2gpu.

The encoder intentionally uses the exact ``qwen3`` template from the user
configuration.  The compatibility suffix ``qwen3_nothink`` is retained only
because the frozen predictor currently discovers additional profiles by that
filename convention.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any

from transformers import AutoTokenizer

from llamafactory.data.template import TEMPLATES


RUN_DIR = Path(__file__).resolve().parent
SOURCE = Path(
    "/wanqing-develop/chengjin/2卡baseline/baseline_2gpu_lf.jsonl"
)
MODEL = Path("/wanqing-models/Qwen3-14B")
PROFILE = (
    RUN_DIR / "profiles" / "baseline_2gpu_case.qwen3_nothink.jsonl"
)
SUMMARY = RUN_DIR / "profile_summary.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    template = copy.deepcopy(TEMPLATES["qwen3"])
    template.fix_special_tokens(tokenizer)

    PROFILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = PROFILE.with_name(PROFILE.name + ".tmp")
    lengths: list[int] = []
    labels: list[int] = []
    with SOURCE.open(encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as output:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"system", "prompt", "response"}:
                raise ValueError(f"Unexpected schema at source row {index + 1}")
            conversation = [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            ]
            pairs = template.encode_multiturn(
                tokenizer,
                conversation,
                system=row["system"],
                tools=None,
            )
            source_tokens = sum(len(source_ids) for source_ids, _ in pairs)
            label_tokens = sum(len(target_ids) for _, target_ids in pairs)
            if template.efficient_eos:
                label_tokens += 1
            total_tokens = source_tokens + label_tokens
            lengths.append(total_tokens)
            labels.append(label_tokens)
            output.write(
                json.dumps(
                    {
                        "sample_id": f"baseline_2gpu_case:{index}",
                        "total_tokens": total_tokens,
                        "label_tokens": label_tokens,
                        "turns": 2,
                        "assistant_turns": 1,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    os.replace(temporary, PROFILE)

    p50 = percentile(lengths, 50)
    p90 = percentile(lengths, 90)
    p99 = percentile(lengths, 99)
    if p90 <= 512:
        category = "short"
    elif p99 / max(p50, 1.0) >= 1.75:
        category = "longtail"
    elif p50 >= 8192:
        category = "longcontext"
    else:
        category = "multiturn"

    summary = {
        "schema": "baseline_2gpu_profile/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": "baseline_2gpu_case",
        "source": {
            "path": str(SOURCE),
            "sha256": sha256_file(SOURCE),
            "records": len(lengths),
            "raw_text_copied_to_profile": False,
        },
        "model": str(MODEL),
        "template": "qwen3",
        "profile": {
            "path": str(PROFILE),
            "sha256": sha256_file(PROFILE),
            "minimum": min(lengths),
            "mean": statistics.fmean(lengths),
            "p50": p50,
            "p90": p90,
            "p95": percentile(lengths, 95),
            "p99": p99,
            "maximum": max(lengths),
            "label_token_ratio": sum(labels) / sum(lengths),
            "suggested_dataset_category": category,
        },
    }
    write_json_atomic(SUMMARY, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
