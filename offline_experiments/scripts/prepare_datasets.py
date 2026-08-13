#!/usr/bin/env python3
"""Download real SFT sources and build six reproducible, 1,000-row length slices.

All lengths are produced with the tokenizer and Template implementation imported
from the exact LLaMA-Factory environment used for training.  The long-tail slice
is a controlled mixture of real samples; no generated filler text is used.
"""

from __future__ import annotations

import argparse
import copy
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

from common import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    DATA_DIR,
    aligned_cutoff,
    percentile,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


SAMPLE_SIZE = 1_000
SEED = 20260716
REFERENCE_MODEL = Path("/wanqing-models/Qwen3-8B")
LEGACY_DATA_ROOT = Path("/wanqing-develop/luowenjing/tokenizer/hf_static_analysis_datasets")


@dataclass(frozen=True)
class SourceSpec:
    id: str
    repo_id: str
    revision: str
    split: str
    paths: tuple[str, ...]
    columns: tuple[str, ...]
    row_format: str
    pool_size: int
    source_rows: int
    dataset_server_config: str = "default"


SOURCES = (
    SourceSpec(
        id="alpaca_cleaned",
        repo_id="yahma/alpaca-cleaned",
        revision="12567cabf869d7c92e573c7c783905fc160e9639",
        split="train",
        paths=("datasets/yahma/alpaca-cleaned@refs/convert/parquet/default/train/0000.parquet",),
        columns=("instruction", "input", "output"),
        row_format="alpaca",
        pool_size=6_000,
        source_rows=51_760,
    ),
    SourceSpec(
        id="ultrachat_200k",
        repo_id="HuggingFaceH4/ultrachat_200k",
        revision="8049631c405ae6576f93f445c6b8166f76f5505a",
        split="train_sft",
        paths=(
            "datasets/HuggingFaceH4/ultrachat_200k/data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
            "datasets/HuggingFaceH4/ultrachat_200k/data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
            "datasets/HuggingFaceH4/ultrachat_200k/data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet",
        ),
        columns=("prompt_id", "messages"),
        row_format="messages",
        pool_size=1_000,
        source_rows=207_865,
    ),
    SourceSpec(
        id="longalpaca_12k",
        repo_id="Yukang/LongAlpaca-12k",
        revision="46dce924ed8786979556018e191c0f557d8f4aa2",
        split="train",
        paths=("datasets/Yukang/LongAlpaca-12k@refs/convert/parquet/default/train/0000.parquet",),
        columns=("instruction", "input", "output"),
        row_format="alpaca",
        pool_size=1_000,
        source_rows=12_000,
    ),
)

LEGACY_POOL_FILES = {
    "alpaca_cleaned": LEGACY_DATA_ROOT / "alpaca_short" / "data.jsonl",
    "ultrachat_200k": LEGACY_DATA_ROOT / "ultrachat_multiturn" / "data.jsonl",
    "longalpaca_12k": LEGACY_DATA_ROOT / "longalpaca_long" / "data.jsonl",
}


def spread_indices(count: int, selected: int) -> list[int]:
    if selected >= count:
        return list(range(count))
    if selected == 1:
        return [count // 2]
    return sorted({round(i * (count - 1) / (selected - 1)) for i in range(selected)})


def normalize_messages(row: dict[str, Any], row_format: str) -> list[dict[str, str]] | None:
    if row_format == "alpaca":
        instruction = str(row.get("instruction") or "").strip()
        input_text = str(row.get("input") or "").strip()
        output = str(row.get("output") or "").strip()
        if not instruction or not output:
            return None
        user = instruction + (("\n\nInput:\n" + input_text) if input_text else "")
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": output},
        ]

    messages = row.get("messages") or []
    normalized: list[dict[str, str]] = []
    for raw_message in messages:
        role = str((raw_message or {}).get("role") or "").strip()
        content = str((raw_message or {}).get("content") or "").strip()
        if role not in {"system", "user", "assistant"} or not content:
            return None
        normalized.append({"role": role, "content": content})

    conversation = normalized[1:] if normalized and normalized[0]["role"] == "system" else normalized
    if not conversation or len(conversation) % 2 != 0:
        return None
    if any(message["role"] != ("user" if i % 2 == 0 else "assistant") for i, message in enumerate(conversation)):
        return None
    return normalized


def download_source_pool(spec: SourceSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # The datasets-server /rows endpoint silently ignores a revision query.  Stream
    # through datasets.load_dataset instead so the exact source commit is enforced.
    reserve = min(spec.source_rows, spec.pool_size + max(500, spec.pool_size // 10))
    selected_indices = set(spread_indices(spec.source_rows, reserve))
    load_kwargs: dict[str, Any] = {
        "path": spec.repo_id,
        "revision": spec.revision,
        "split": spec.split,
        "streaming": True,
    }
    if spec.dataset_server_config != "default":
        load_kwargs["name"] = spec.dataset_server_config
    stream = load_dataset(**load_kwargs)
    candidates: list[dict[str, Any]] = []
    last_selected = max(selected_indices)
    for source_index, raw_row in enumerate(stream):
        if source_index not in selected_indices:
            if source_index >= last_selected:
                break
            continue
        messages = normalize_messages(dict(raw_row), spec.row_format)
        if messages is not None:
            candidates.append(
                {
                    "sample_id": f"{spec.id}:{source_index}",
                    "source_dataset": spec.id,
                    "source_index": source_index,
                    "messages": messages,
                }
            )
        if source_index >= last_selected:
            break

    unique = {row["sample_id"]: row for row in candidates}
    if len(unique) < spec.pool_size:
        raise RuntimeError(f"{spec.id}: only {len(unique)} valid unique rows for a {spec.pool_size}-row pool")
    ordered = [unique[key] for key in sorted(unique)]
    if len(ordered) > spec.pool_size:
        ordered = random.Random(SEED + sum(map(ord, spec.id))).sample(ordered, spec.pool_size)
    output = sorted(ordered, key=lambda row: row["source_index"])
    metadata = {
        "id": spec.id,
        "repo_id": spec.repo_id,
        "source_url": f"https://huggingface.co/datasets/{spec.repo_id}",
        "revision": spec.revision,
        "split": spec.split,
        "source_rows": spec.source_rows,
        "pool_rows": len(output),
        "sampling_seed": SEED + sum(map(ord, spec.id)),
        "download_api": "datasets.load_dataset(streaming=True)",
        "revision_enforced": True,
        "selected_source_indices": sorted(row["source_index"] for row in output),
        "source_parquet_paths": list(spec.paths),
    }
    return output, metadata


def load_or_download_pools(reuse: bool, offline: bool) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    pool_dir = DATA_DIR / "source_pools"
    pool_dir.mkdir(parents=True, exist_ok=True)
    pools: dict[str, list[dict[str, Any]]] = {}
    metadata: list[dict[str, Any]] = []
    for spec in SOURCES:
        pool_path = pool_dir / f"{spec.id}.jsonl"
        info_path = pool_dir / f"{spec.id}.json"
        if pool_path.exists() and info_path.exists() and (reuse or offline):
            rows = read_jsonl(pool_path)
            info = read_json(info_path)
            if len(rows) != spec.pool_size:
                raise RuntimeError(f"Cached {spec.id} has {len(rows)} rows, expected {spec.pool_size}")
            if info.get("imported_from"):
                legacy_manifest_path = LEGACY_DATA_ROOT / "manifest.json"
                legacy_manifest = read_json(legacy_manifest_path)
                legacy_entry = next(
                    row
                    for row in legacy_manifest["datasets"]
                    if row["repo_id"] == spec.repo_id and row["split"] == spec.split
                )
                if legacy_entry["source_revision"] != spec.revision:
                    raise RuntimeError(
                        f"Cached legacy {spec.id} revision {legacy_entry['source_revision']} does not match {spec.revision}"
                    )
                info["revision_enforced"] = True
                info["legacy_manifest"] = str(legacy_manifest_path)
                info["legacy_manifest_sha256"] = sha256_file(legacy_manifest_path)
        elif LEGACY_POOL_FILES[spec.id].exists() and spec.id != "alpaca_cleaned":
            legacy_manifest_path = LEGACY_DATA_ROOT / "manifest.json"
            legacy_manifest = read_json(legacy_manifest_path)
            legacy_entry = next(
                row for row in legacy_manifest["datasets"] if row["repo_id"] == spec.repo_id and row["split"] == spec.split
            )
            if legacy_entry["source_revision"] != spec.revision:
                raise RuntimeError(
                    f"Legacy {spec.id} revision {legacy_entry['source_revision']} does not match {spec.revision}"
                )
            legacy_rows = read_jsonl(LEGACY_POOL_FILES[spec.id])
            if len(legacy_rows) < spec.pool_size:
                raise RuntimeError(f"Legacy {spec.id} has only {len(legacy_rows)} rows")
            rows = [
                {
                    "sample_id": f"{spec.id}:{legacy['source_index']}",
                    "source_dataset": spec.id,
                    "source_index": legacy["source_index"],
                    "messages": legacy["messages"],
                }
                for legacy in legacy_rows[: spec.pool_size]
            ]
            info = {
                "id": spec.id,
                "repo_id": spec.repo_id,
                "source_url": f"https://huggingface.co/datasets/{spec.repo_id}",
                "revision": spec.revision,
                "split": spec.split,
                "source_rows": spec.source_rows,
                "pool_rows": len(rows),
                "sampling_seed": "inherited from hf_static_analysis_datasets/manifest.json",
                "imported_from": str(LEGACY_POOL_FILES[spec.id]),
                "revision_enforced": True,
                "legacy_manifest": str(legacy_manifest_path),
                "legacy_manifest_sha256": sha256_file(legacy_manifest_path),
            }
            write_jsonl(pool_path, rows)
            write_json(info_path, info)
        else:
            if offline:
                raise FileNotFoundError(f"Offline mode requested but source pool is absent: {pool_path}")
            print(f"Downloading a revision-pinned {spec.pool_size}-row pool from {spec.repo_id}", flush=True)
            rows, info = download_source_pool(spec)
            write_jsonl(pool_path, rows)
            write_json(info_path, info)
        info["local_file"] = str(pool_path)
        info["local_sha256"] = sha256_file(pool_path)
        info.setdefault("revision_enforced", False)
        info["local_snapshot_authoritative"] = True
        write_json(info_path, info)
        pools[spec.id] = rows
        metadata.append(info)
        print(f"  {spec.id}: {len(rows)} rows", flush=True)
    return pools, metadata


class TrainingEncoder:
    def __init__(self, model_path: Path, template_name: str, profile_id: str):
        from llamafactory.data.template import TEMPLATES

        self.model_path = model_path
        self.template_name = template_name
        self.profile_id = profile_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        self.template = copy.deepcopy(TEMPLATES[template_name])
        self.template.fix_special_tokens(self.tokenizer)

    def encode(self, messages: list[dict[str, str]]) -> tuple[int, int]:
        system = None
        conversation = messages
        if messages and messages[0]["role"] == "system":
            system = messages[0]["content"]
            conversation = messages[1:]
        pairs = self.template.encode_multiturn(self.tokenizer, conversation, system=system, tools=None)
        source_tokens = sum(len(source_ids) for source_ids, _ in pairs)
        label_tokens = sum(len(target_ids) for _, target_ids in pairs)
        if self.template.efficient_eos:
            label_tokens += 1
        return source_tokens + label_tokens, label_tokens


def cap_row_to_tokens(row: dict[str, Any], encoder: TrainingEncoder, cap: int, slice_id: str) -> dict[str, Any]:
    """Crop the minimum amount of real message content needed to fit a length cap."""
    transformed = copy.deepcopy(row)
    transformed["sample_id"] = f"{row['sample_id']}:{slice_id}"
    original_tokens, _ = encoder.encode(transformed["messages"])
    if original_tokens <= cap:
        transformed["transformation"] = {
            "type": "identity",
            "original_tokens": original_tokens,
            "reference_tokens_after_crop": original_tokens,
            "reference_cap": cap,
        }
        return transformed

    messages = transformed["messages"]
    while True:
        current_tokens, _ = encoder.encode(messages)
        if current_tokens <= cap:
            transformed["transformation"] = {
                "type": "content_prefix_crop",
                "original_tokens": original_tokens,
                "reference_tokens_after_crop": current_tokens,
                "reference_cap": cap,
            }
            return transformed

        eligible = [index for index, message in enumerate(messages) if len(message["content"]) > 64]
        if not eligible:
            raise RuntimeError(f"Cannot crop {row['sample_id']} below {cap} tokens without emptying messages")
        # Prefer cropping the longest user/system context before an answer.
        message_index = max(
            eligible,
            key=lambda index: (messages[index]["role"] in {"user", "system"}, len(messages[index]["content"])),
        )
        original_content = messages[message_index]["content"]
        messages[message_index]["content"] = original_content[:64]
        minimum_tokens, _ = encoder.encode(messages)
        if minimum_tokens > cap:
            continue

        low, high = 64, len(original_content)
        best = 64
        while low <= high:
            midpoint = (low + high) // 2
            messages[message_index]["content"] = original_content[:midpoint]
            length, _ = encoder.encode(messages)
            if length <= cap:
                best = midpoint
                low = midpoint + 1
            else:
                high = midpoint - 1
        messages[message_index]["content"] = original_content[:best]


def profile_rows(rows: list[dict[str, Any]], encoder: TrainingEncoder) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    profiles = []
    for row in rows:
        total_tokens, label_tokens = encoder.encode(row["messages"])
        profiles.append(
            {
                "sample_id": row["sample_id"],
                "total_tokens": total_tokens,
                "label_tokens": label_tokens,
                "turns": len(row["messages"]),
                "assistant_turns": sum(message["role"] == "assistant" for message in row["messages"]),
            }
        )
    return profiles, time.perf_counter() - started


def profile_map(profiles: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {profile["sample_id"]: profile for profile in profiles}


def choose(candidates: list[dict[str, Any]], count: int, seed: int, anchor: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    unique = {row["sample_id"]: row for row in candidates}
    if len(unique) < count:
        raise RuntimeError(f"Only {len(unique)} candidates available for a {count}-row slice")
    ordered = [unique[key] for key in sorted(unique)]
    selected = random.Random(seed).sample(ordered, count)
    if anchor is not None and all(row["sample_id"] != anchor["sample_id"] for row in selected):
        selected[0] = anchor
    return selected


def in_length_range(rows: list[dict[str, Any]], lengths: dict[str, dict[str, Any]], low: int = 0, high: int = 10**9) -> list[dict[str, Any]]:
    return [row for row in rows if low <= lengths[row["sample_id"]]["total_tokens"] <= high]


def anchor_in_range(rows: list[dict[str, Any]], lengths: dict[str, dict[str, Any]], low: int, high: int) -> dict[str, Any]:
    candidates = in_length_range(rows, lengths, low, high)
    if not candidates:
        raise RuntimeError(f"No anchor sample in token range [{low}, {high}]")
    return max(candidates, key=lambda row: lengths[row["sample_id"]]["total_tokens"])


def build_slices(
    pools: dict[str, list[dict[str, Any]]],
    reference_profiles: dict[str, dict[str, Any]],
    reference_encoder: TrainingEncoder,
) -> dict[str, list[dict[str, Any]]]:
    short = pools["alpaca_cleaned"]
    medium = pools["ultrachat_200k"]
    long = pools["longalpaca_12k"]

    short_512_candidates = in_length_range(short, reference_profiles, high=448)
    short_512 = choose(
        short_512_candidates,
        SAMPLE_SIZE,
        SEED + 1,
        anchor_in_range(short_512_candidates, reference_profiles, 65, 448),
    )

    medium_2048 = [cap_row_to_tokens(row, reference_encoder, 1984, "multiturn_2048") for row in medium]

    medium_4096 = [cap_row_to_tokens(row, reference_encoder, 4032, "multiturn_4096") for row in medium]

    tail_short = choose(in_length_range(short, reference_profiles, high=1024), 700, SEED + 40)
    tail_medium = choose(in_length_range(medium, reference_profiles, 513, 4096), 250, SEED + 41)
    tail_long_capped = [cap_row_to_tokens(row, reference_encoder, 8128, "longtail_8192") for row in long]
    tail_long_profiles, _ = profile_rows(tail_long_capped, reference_encoder)
    tail_long_lengths = profile_map(tail_long_profiles)
    tail_long_candidates = in_length_range(tail_long_capped, tail_long_lengths, 4097, 8128)
    tail_anchor = anchor_in_range(tail_long_candidates, tail_long_lengths, 7745, 8128)
    tail_long = choose(tail_long_candidates, 50, SEED + 42, tail_anchor)
    longtail_8192 = tail_short + tail_medium + tail_long
    random.Random(SEED + 4).shuffle(longtail_8192)

    long_16384 = [cap_row_to_tokens(row, reference_encoder, 16320, "longcontext_16384") for row in long]

    long_32768 = [cap_row_to_tokens(row, reference_encoder, 32704, "longcontext_32768") for row in long]

    slices = {
        "short_512": short_512,
        "multiturn_2048": medium_2048,
        "multiturn_4096": medium_4096,
        "longtail_8192": longtail_8192,
        "longcontext_16384": long_16384,
        "longcontext_32768": long_32768,
    }
    # Keep the nested JSON schema identical for every row. Datasets 4.0 fixes
    # the struct schema from early Arrow chunks and rejects later rows that add
    # fields, even though this audit-only column is not consumed by training.
    for rows in slices.values():
        for row in rows:
            if "transformation" not in row:
                tokens = reference_profiles[row["sample_id"]]["total_tokens"]
                row["transformation"] = {
                    "type": "source_identity",
                    "original_tokens": tokens,
                    "reference_tokens_after_crop": tokens,
                    "reference_cap": 0,
                }
    return slices


def summarize_lengths(profiles: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    lengths = [profile["total_tokens"] for profile in profiles]
    labels = [profile["label_tokens"] for profile in profiles]
    turns = [profile["turns"] for profile in profiles]
    return {
        "samples": len(lengths),
        "total_tokens": sum(lengths),
        "mean_tokens": statistics.fmean(lengths),
        "min_tokens": min(lengths),
        "p50_tokens": percentile(lengths, 50),
        "p90_tokens": percentile(lengths, 90),
        "p95_tokens": percentile(lengths, 95),
        "p99_tokens": percentile(lengths, 99),
        "max_tokens": max(lengths),
        "cutoff_len": aligned_cutoff(max(lengths)),
        "truncated_samples_at_cutoff": sum(length > aligned_cutoff(max(lengths)) for length in lengths),
        "total_label_tokens": sum(labels),
        "mean_label_tokens": statistics.fmean(labels),
        "mean_turns": statistics.fmean(turns),
        "tokenize_seconds": elapsed,
        "samples_per_second": len(lengths) / elapsed,
        "tokens_per_second": sum(lengths) / elapsed,
    }


def derive_ga(target_gbs: int, data_parallel: int, effective_mbs: float) -> dict[str, Any]:
    raw = target_gbs / (data_parallel * effective_mbs)
    candidates = sorted({max(1, int(raw // 1)), max(1, int(-(-raw // 1)))})
    best = min(candidates, key=lambda ga: (abs(data_parallel * ga * effective_mbs - target_gbs), ga))
    expected = data_parallel * best * effective_mbs
    return {
        "target_gbs": target_gbs,
        "data_parallel": data_parallel,
        "ga_raw": raw,
        "gradient_accumulation_steps": best,
        "expected_sample_gbs": expected,
        "relative_error": abs(expected - target_gbs) / target_gbs,
    }


def packing_summary(
    lengths: list[int], cutoff: int, gate: dict[str, Any], preprocessing_workers: int
) -> dict[str, Any]:
    from llamafactory.data.processor.processor_utils import greedy_knapsack

    # DataArguments.__post_init__ subtracts one when neat_packing=True. The packed
    # processor then appends one right-padding token, so the model-facing tensor
    # is the user cutoff while the knapsack capacity is cutoff-1.
    packing_capacity = cutoff - 1
    # datasets.Dataset.map gives each preprocessing worker a contiguous shard.
    # Packing is performed independently inside those shards, so a single global
    # knapsack slightly overestimates packing efficiency.
    worker_count = min(preprocessing_workers, len(lengths))
    knapsacks = []
    for worker in range(worker_count):
        start = len(lengths) * worker // worker_count
        stop = len(lengths) * (worker + 1) // worker_count
        knapsacks.extend(greedy_knapsack(lengths[start:stop].copy(), packing_capacity))
    counts = [len(knapsack) for knapsack in knapsacks]
    used = [sum(knapsack) for knapsack in knapsacks]
    effective_mbs = statistics.fmean(counts)
    ga_table = [derive_ga(gbs, dp, effective_mbs) for dp in (1, 2, 4) for gbs in (16, 64, 256)]
    utilization = sum(used) / (len(knapsacks) * packing_capacity)
    sequence_reduction = 1.0 - len(knapsacks) / len(lengths)
    static_candidate = (
        utilization >= gate["minimum_pack_utilization"]
        and sequence_reduction >= gate["minimum_sequence_reduction"]
        and effective_mbs >= gate["minimum_mean_samples_per_pack"]
        and any(row["relative_error"] <= gate["maximum_expected_gbs_error"] for row in ga_table)
    )
    return {
        "implementation": "llamafactory.data.processor.processor_utils.greedy_knapsack",
        "configured_cutoff_len": cutoff,
        "preprocessing_num_workers": worker_count,
        "packing_scope": "independent contiguous datasets.Dataset.map worker shards",
        "effective_knapsack_capacity": packing_capacity,
        "processor_output_tokens_before_collator_unpad": cutoff,
        "deterministic_for_fixed_lengths": True,
        "packs": len(knapsacks),
        "pack_utilization": utilization,
        "sequence_reduction_ratio": sequence_reduction,
        "mean_samples_per_pack": effective_mbs,
        "p50_samples_per_pack": percentile(counts, 50),
        "p90_samples_per_pack": percentile(counts, 90),
        "p99_samples_per_pack": percentile(counts, 99),
        "variance_samples_per_pack": statistics.pvariance(counts),
        "min_used_tokens_per_pack": min(used),
        "max_used_tokens_per_pack": max(used),
        "ga_table": ga_table,
        "packing_eligible_for_paired_test": static_candidate,
        "packing_enabled_by_static_analysis": False,
    }


def no_packing_padding(lengths: list[int]) -> list[dict[str, Any]]:
    results = []
    for mbs in (1, 2, 4, 8):
        runs = []
        for seed in (SEED, SEED + 1, SEED + 2):
            shuffled = lengths.copy()
            random.Random(seed).shuffle(shuffled)
            computed = 0
            for start in range(0, len(shuffled), mbs):
                batch = shuffled[start : start + mbs]
                computed += max(batch) * len(batch)
            runs.append(sum(shuffled) / computed)
        results.append(
            {
                "physical_mbs": mbs,
                "mean_dynamic_padding_utilization": statistics.fmean(runs),
                "min_dynamic_padding_utilization": min(runs),
                "max_dynamic_padding_utilization": max(runs),
            }
        )
    return results


def write_dataset_info(dataset_ids: list[str]) -> None:
    info = {}
    for dataset_id in dataset_ids:
        info[dataset_id] = {
            "file_name": f"derived/{dataset_id}.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    write_json(DATA_DIR / "dataset_info.json", info)


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# 数据切片静态分析",
        "",
        "所有样本均来自三个固定 revision 的 Hugging Face 真实数据集；`longtail_8192` 是真实样本的受控混合，不含生成填充文本。",
        "长度由当前训练环境中的 LLaMA-Factory Template 和本地 tokenizer 实际编码得到。",
        "",
        "| 数据切片 | 类别 | 模板口径 | P50 | P90 | P99 | Max | cutoff_len | packing 配对候选 |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    categories = {row["id"]: row["category"] for row in read_json(CONFIG_DIR / "experiment.json")["datasets"]}
    for dataset_id, dataset in report["datasets"].items():
        for profile_id, profile in dataset["profiles"].items():
            lengths = profile["lengths"]
            packing = profile["packing"]
            lines.append(
                f"| {dataset_id} | {categories[dataset_id]} | {profile_id} | "
                f"{lengths['p50_tokens']:.0f} | {lengths['p90_tokens']:.0f} | {lengths['p99_tokens']:.0f} | "
                f"{lengths['max_tokens']} | {lengths['cutoff_len']} | "
                f"{'是' if packing['packing_eligible_for_paired_test'] else '否'} |"
            )
    lines += [
        "",
        "`packing_eligible_for_paired_test` 只决定是否进入成对实测；静态分析不会直接开启 packing。最终开启仍需满足方案中的保守训练时间收益阈值。",
        "",
    ]
    (ARTIFACT_DIR / "dataset_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-source-pools", action="store_true", help="Reuse complete cached source pools")
    parser.add_argument("--offline", action="store_true", help="Never access Hugging Face; require cached source pools")
    parser.add_argument(
        "--refresh-packing-only",
        action="store_true",
        help="Reuse saved token profiles and refresh packing statistics without downloading or tokenizing",
    )
    args = parser.parse_args()

    experiment = read_json(CONFIG_DIR / "experiment.json")
    if args.refresh_packing_only:
        report = read_json(ARTIFACT_DIR / "dataset_analysis.json")
        for dataset_id, dataset in report["datasets"].items():
            profile = dataset["profiles"]["qwen3_nothink"]
            dataset["profiles"] = {"qwen3_nothink": profile}
            profiles = read_jsonl(ARTIFACT_DIR / "dataset_profiles" / f"{dataset_id}.qwen3_nothink.jsonl")
            lengths = [row["total_tokens"] for row in profiles]
            profile["packing"] = packing_summary(
                lengths,
                profile["lengths"]["cutoff_len"],
                experiment["packing_static_gate"],
                int(experiment["fixed_runtime"]["preprocessing_num_workers"]),
            )
        report["tokenizer_policy"] = "All analyses and training jobs use a Qwen3 tokenizer with qwen3_nothink."
        for obsolete in (ARTIFACT_DIR / "dataset_profiles").glob("*.qwen2p5_qwen.jsonl"):
            obsolete.unlink()
        write_json(ARTIFACT_DIR / "dataset_analysis.json", report)
        write_markdown(report)
        print("Refreshed packing statistics from saved token profiles")
        return
    targets = {row["id"]: row["target_cutoff"] for row in experiment["datasets"]}
    pools, source_metadata = load_or_download_pools(args.reuse_source_pools, args.offline)

    reference_encoder = TrainingEncoder(REFERENCE_MODEL, "qwen3_nothink", "qwen3_nothink")

    reference_profiles: dict[str, dict[str, Any]] = {}
    source_profile_rows: list[dict[str, Any]] = []
    for source_id, rows in pools.items():
        print(f"Tokenizing source pool {source_id} with Qwen3 training template", flush=True)
        profiles, elapsed = profile_rows(rows, reference_encoder)
        print(f"  {len(rows)} rows in {elapsed:.1f}s", flush=True)
        for profile in profiles:
            profile["source_dataset"] = source_id
        source_profile_rows.extend(profiles)
        reference_profiles.update(profile_map(profiles))
    write_jsonl(ARTIFACT_DIR / "source_profiles_qwen3.jsonl", source_profile_rows)

    slices = build_slices(pools, reference_profiles, reference_encoder)
    write_dataset_info(list(slices))
    derived_dir = DATA_DIR / "derived"
    profile_dir = ARTIFACT_DIR / "dataset_profiles"
    report: dict[str, Any] = {
        "schema_version": 1,
        "sample_size_per_slice": SAMPLE_SIZE,
        "seed": SEED,
        "cutoff_rule": "ceil(max_sample_tokens / 512) * 512",
        "tokenizer_policy": "All analyses and training jobs use a Qwen3 tokenizer with qwen3_nothink.",
        "sources": source_metadata,
        "datasets": {},
    }

    for dataset_id, rows in slices.items():
        if len(rows) != SAMPLE_SIZE or len({row["sample_id"] for row in rows}) != SAMPLE_SIZE:
            raise RuntimeError(f"{dataset_id} is not a unique {SAMPLE_SIZE}-row slice")
        output_rows = [{**row, "slice_id": dataset_id} for row in rows]
        output_path = derived_dir / f"{dataset_id}.jsonl"
        write_jsonl(output_path, output_rows)
        dataset_result: dict[str, Any] = {
            "file": str(output_path),
            "sha256": sha256_file(output_path),
            "samples": len(output_rows),
            "source_counts": {},
            "profiles": {},
        }
        source_counts: dict[str, int] = {}
        for row in rows:
            source_counts[row["source_dataset"]] = source_counts.get(row["source_dataset"], 0) + 1
        dataset_result["source_counts"] = dict(sorted(source_counts.items()))

        for encoder in (reference_encoder,):
            print(f"Profiling {dataset_id} with {encoder.profile_id}", flush=True)
            profiles, elapsed = profile_rows(rows, encoder)
            lengths_summary = summarize_lengths(profiles, elapsed)
            if lengths_summary["cutoff_len"] != targets[dataset_id]:
                raise RuntimeError(
                    f"{dataset_id}/{encoder.profile_id}: cutoff {lengths_summary['cutoff_len']} "
                    f"does not match target {targets[dataset_id]}"
                )
            lengths = [profile["total_tokens"] for profile in profiles]
            packing = packing_summary(
                lengths,
                lengths_summary["cutoff_len"],
                experiment["packing_static_gate"],
                int(experiment["fixed_runtime"]["preprocessing_num_workers"]),
            )
            dataset_result["profiles"][encoder.profile_id] = {
                "model_path": str(encoder.model_path),
                "template": encoder.template_name,
                "lengths": lengths_summary,
                "no_packing": no_packing_padding(lengths),
                "packing": packing,
            }
            write_jsonl(profile_dir / f"{dataset_id}.{encoder.profile_id}.jsonl", profiles)
        report["datasets"][dataset_id] = dataset_result

    write_json(ARTIFACT_DIR / "dataset_analysis.json", report)
    write_markdown(report)
    print(f"Wrote six slices under {derived_dir}")
    print(f"Wrote analysis to {ARTIFACT_DIR / 'dataset_analysis.json'}")


if __name__ == "__main__":
    main()
