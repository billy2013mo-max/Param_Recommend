#!/usr/bin/env python3
"""Build privacy-preserving token profiles for the two business text datasets.

The script uses the installed LLaMA-Factory ``qwen3_nothink`` template and
the tokenizer belonging to each requested model. It writes only token counts
and aggregate statistics; raw system, prompt and response text are never
copied into the profile artifacts.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
from typing import Any, Iterable

import numpy as np
from transformers import AutoTokenizer

from llamafactory.data.template import TEMPLATES


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "profiles"
SUMMARY_DIR = ROOT / "profile_summaries"
CUTOFF_GRID = (512, 2048, 4096, 8192, 32768)
MBS_GRID = (1, 2, 4, 8, 16)
SHUFFLE_SEEDS = (17, 43, 97, 193, 389)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    source: Path
    model_id: str
    model_path: Path
    template: str


SPECS = (
    DatasetSpec(
        dataset_id="business_yt0jrk_v2",
        source=(
            ROOT
            / "datasets"
            / "dataset-yt0jrk-1775555547"
            / "2"
            / "publish"
            / "dataset-yt0jrk-1775555547-V2.jsonl"
        ),
        model_id="qwen3_8b",
        model_path=Path("/wanqing-models/Qwen3-8B"),
        template="qwen3_nothink",
    ),
    DatasetSpec(
        dataset_id="business_flieht_v4",
        source=(
            ROOT
            / "datasets"
            / "dataset-flieht-1763953980"
            / "4"
            / "publish"
            / "dataset-flieht-1763953980-V4.jsonl"
        ),
        model_id="qwen3_14b",
        model_path=Path("/wanqing-models/Qwen3-14B"),
        template="qwen3_nothink",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
                json.dumps(row, ensure_ascii=False, sort_keys=True)
                + "\n"
            )
    os.replace(temporary, path)


class TrainingEncoder:
    def __init__(self, spec: DatasetSpec) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model_path,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True,
        )
        self.template = copy.deepcopy(TEMPLATES[spec.template])
        self.template.fix_special_tokens(self.tokenizer)

    def encode(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[int, int]:
        system = None
        conversation = messages
        if messages and messages[0]["role"] == "system":
            system = messages[0]["content"]
            conversation = messages[1:]
        pairs = self.template.encode_multiturn(
            self.tokenizer,
            conversation,
            system=system,
            tools=None,
        )
        source_tokens = sum(
            len(source_ids) for source_ids, _ in pairs
        )
        label_tokens = sum(
            len(target_ids) for _, target_ids in pairs
        )
        if self.template.efficient_eos:
            label_tokens += 1
        return source_tokens + label_tokens, label_tokens


def read_messages(path: Path) -> Iterable[list[dict[str, str]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not isinstance(row, list)
                or len(row) != 1
                or not isinstance(row[0], dict)
            ):
                raise ValueError(
                    f"{path}:{line_number} is not a wrapped triplet"
                )
            item = row[0]
            expected = {"system", "prompt", "response"}
            if set(item) != expected or not all(
                isinstance(item[field], str) for field in expected
            ):
                raise ValueError(
                    f"{path}:{line_number} has an unexpected schema"
                )
            yield [
                {"role": "system", "content": item["system"]},
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response"]},
            ]


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def best_fit_bins(lengths: list[int], capacity: int) -> list[list[int]]:
    bins: list[list[int]] = []
    remaining: list[int] = []
    for length in sorted(
        (min(int(value), capacity) for value in lengths),
        reverse=True,
    ):
        eligible = [
            (space - length, index)
            for index, space in enumerate(remaining)
            if space >= length
        ]
        if not eligible:
            bins.append([length])
            remaining.append(capacity - length)
            continue
        _, index = min(eligible)
        bins[index].append(length)
        remaining[index] -= length
    return bins


def unpacked_utilization(lengths: list[int], mbs: int) -> float:
    runs = []
    for seed in SHUFFLE_SEEDS:
        shuffled = list(lengths)
        random.Random(seed).shuffle(shuffled)
        effective = computed = 0
        for start in range(0, len(shuffled), mbs):
            batch = shuffled[start : start + mbs]
            effective += sum(batch)
            computed += max(batch) * len(batch)
        runs.append(effective / computed)
    return statistics.fmean(runs)


def cutoff_analysis(
    raw_lengths: list[int],
    cutoff: int,
) -> dict[str, Any]:
    lengths = [min(value, cutoff) for value in raw_lengths]
    capacity = max(1, cutoff - 1)
    bins = best_fit_bins(lengths, capacity)
    packed_effective = sum(sum(current) for current in bins)
    packed_computed = len(bins) * cutoff
    fill = packed_effective / packed_computed
    by_mbs = {}
    for mbs in MBS_GRID:
        plain = unpacked_utilization(lengths, mbs)
        by_mbs[str(mbs)] = {
            "unpacked_padding_utilization": plain,
            "packing_fill_ratio": fill,
            "theoretical_linear_token_efficiency_ratio": (
                fill / plain
            ),
        }
    return {
        "cutoff_len": cutoff,
        "samples_truncated": sum(
            value > cutoff for value in raw_lengths
        ),
        "sample_truncation_rate": (
            sum(value > cutoff for value in raw_lengths)
            / len(raw_lengths)
        ),
        "tokens_retained_ratio": (
            sum(lengths) / sum(raw_lengths)
        ),
        "mean_tokens_after_cutoff": statistics.fmean(lengths),
        "packing": {
            "algorithm": "deterministic_best_fit_decreasing",
            "capacity": capacity,
            "packs": len(bins),
            "fill_ratio": fill,
            "mean_logical_samples_per_pack": (
                len(lengths) / len(bins)
            ),
        },
        "by_mbs": by_mbs,
    }


def infer_category(lengths: list[int]) -> dict[str, Any]:
    p50 = percentile(lengths, 50)
    p99 = percentile(lengths, 99)
    ratio = p99 / max(p50, 1.0)
    if percentile(lengths, 90) <= 512:
        category = "short"
        reason = "P90 is at most 512 tokens"
    elif ratio >= 1.75:
        category = "longtail"
        reason = "P99/P50 is at least 1.75"
    elif p50 >= 8192:
        category = "longcontext"
        reason = "median length is at least 8192 tokens"
    else:
        category = "multiturn"
        reason = (
            "non-short distribution without a strong length tail; "
            "the frozen v4b category is only a coarse proxy"
        )
    return {
        "suggested_dataset_category": category,
        "rule_reason": reason,
        "p99_over_p50": ratio,
        "requires_review": True,
    }


def profile_dataset(spec: DatasetSpec) -> dict[str, Any]:
    encoder = TrainingEncoder(spec)
    profile_rows = []
    for index, messages in enumerate(read_messages(spec.source)):
        total_tokens, label_tokens = encoder.encode(messages)
        if total_tokens <= 0 or not 0 <= label_tokens <= total_tokens:
            raise ValueError(
                f"Invalid token counts for {spec.dataset_id}:{index}"
            )
        profile_rows.append(
            {
                "sample_id": f"{spec.dataset_id}:{index}",
                "total_tokens": total_tokens,
                "label_tokens": label_tokens,
                "turns": len(messages),
                "assistant_turns": 1,
            }
        )

    lengths = [row["total_tokens"] for row in profile_rows]
    labels = [row["label_tokens"] for row in profile_rows]
    profile_path = (
        PROFILE_DIR / f"{spec.dataset_id}.qwen3_nothink.jsonl"
    )
    write_jsonl_atomic(profile_path, profile_rows)
    tokenizer_config = spec.model_path / "tokenizer_config.json"
    model_config = spec.model_path / "config.json"
    summary = {
        "schema": "business_text_dataset_profile/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": spec.dataset_id,
        "source": {
            "path": str(spec.source.resolve()),
            "sha256": sha256_file(spec.source),
            "records": len(profile_rows),
            "raw_text_copied_to_profile": False,
        },
        "model_binding": {
            "model_id": spec.model_id,
            "model_path": str(spec.model_path),
            "template": spec.template,
            "model_config_sha256": sha256_file(model_config),
            "tokenizer_config_sha256": sha256_file(
                tokenizer_config
            ),
        },
        "profile": {
            "path": str(profile_path.resolve()),
            "sha256": sha256_file(profile_path),
            "length_tokens": {
                "minimum": min(lengths),
                "mean": statistics.fmean(lengths),
                "std": statistics.pstdev(lengths),
                "p50": percentile(lengths, 50),
                "p90": percentile(lengths, 90),
                "p95": percentile(lengths, 95),
                "p99": percentile(lengths, 99),
                "maximum": max(lengths),
            },
            "label_tokens": {
                "mean": statistics.fmean(labels),
                "ratio_of_total": sum(labels) / sum(lengths),
            },
            "category": infer_category(lengths),
            "cutoff_analysis": {
                str(cutoff): cutoff_analysis(lengths, cutoff)
                for cutoff in CUTOFF_GRID
            },
        },
        "packing_decision_is_validated": False,
        "gpu_training_started": False,
    }
    summary_path = SUMMARY_DIR / f"{spec.dataset_id}.json"
    write_json_atomic(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        help="Profile only the selected dataset id; repeat as needed.",
    )
    args = parser.parse_args()
    selected = set(args.dataset_id)
    specs = [
        spec
        for spec in SPECS
        if not selected or spec.dataset_id in selected
    ]
    if selected - {spec.dataset_id for spec in SPECS}:
        raise ValueError(
            "Unknown dataset ids: "
            + ", ".join(sorted(selected - {s.dataset_id for s in SPECS}))
        )
    reports = []
    for spec in specs:
        print(f"Profiling {spec.dataset_id}", flush=True)
        reports.append(profile_dataset(spec))
    print(
        json.dumps(
            {
                report["dataset_id"]: report["profile"][
                    "length_tokens"
                ]
                for report in reports
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
