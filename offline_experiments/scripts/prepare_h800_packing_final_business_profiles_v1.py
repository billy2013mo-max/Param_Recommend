#!/usr/bin/env python3
"""Freeze source-disjoint business DataProfiles for final Packing acceptance.

This is an upload-time, CPU-only preparation step.  It binds raw BS3 objects,
the exact Qwen3 tokenizer/template, and LLaMA-Factory's worker-sharded greedy
packer.  Only aggregate profiles and six fixed cutoff points are retained for
the recommendation contract; raw rows remain offline experiment inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

from transformers import AutoTokenizer
from llamafactory.data.template import TEMPLATES

from common import ARTIFACT_DIR, ROOT, percentile, sha256_file, sha256_json, write_json
from packing_gbs_contract import derive_packing_gbs_contract
from static_packing_predictor import _pack_in_worker_shards


SCHEMA = "sft_h800_packing_final_business_profiles/v1"
INPUT_ROOT = ROOT / "data" / "packing_final_business_screen_20260810"
PRIOR_ROOTS = (
    ROOT / "data" / "remote_candidate_screen_20260804",
    ROOT / "data" / "source_pools",
    ROOT / "data" / "bounded_memory_v2_fresh_holdout_v1",
    ROOT.parent / "real_business_validation_20260730" / "datasets",
)
OUTPUT_DIR = ARTIFACT_DIR / "h800_packing_final_business_profiles_v1"
OUTPUT = ARTIFACT_DIR / "h800_packing_final_business_profile_manifest_v1.json"
TOKENIZER = Path("/wanqing-models/Qwen3-8B")
TEMPLATE = "qwen3_nothink"
WORKERS = 8
CUTOFFS = (1_024, 2_048, 4_096, 8_192, 16_384, 32_768)
TARGET_GBS_VALUES = (64, 128, 256, 512)
GPU_COUNTS = (4, 5, 6, 7)
FIELDS = ("system", "prompt", "response")

# The split is frozen before any final GPU measurement.  Fit sources may tune
# coefficients or margins; holdout sources may only decide pass/fail.
SOURCES = (
    ("PF01", "dataset-o6x7si-1783495294", "fit", "extreme_short"),
    ("PF02", "dataset-li2sye-1780465080", "fit", "short"),
    ("PF03", "dataset-rzoehw-1778321510", "fit", "short_broad"),
    ("PF04", "dataset-udwtrx-1785229698", "fit", "medium_concentrated"),
    ("PF05", "dataset-kxvcnz-1780930601", "fit", "medium_long"),
    ("PF06", "dataset-mooxf7-1778662463", "fit", "long_concentrated"),
    ("PH01", "dataset-ellxoz-1784727551", "prospective_holdout", "very_short"),
    ("PH02", "dataset-zefhaf-1780147888", "prospective_holdout", "short_tail"),
    ("PH03", "dataset-bpgwrk-1784805111", "prospective_holdout", "medium_broad"),
    ("PH04", "dataset-bigkex-1778126018", "prospective_holdout", "medium_long_concentrated"),
    ("PH05", "dataset-islt4f-1785055079", "prospective_holdout", "long_near_cutoff"),
    ("PH06", "dataset-51m5uv-1784027387", "prospective_holdout", "extreme_long_tail"),
)


def _iter_objects(path: Path) -> Iterable[Any]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, list):
                yield from value
            else:
                yield value


def _find_source(root: Path, dataset_id: str) -> Path:
    paths = sorted((root / dataset_id).glob("[0-9]*/publish/*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one downloaded JSONL for {dataset_id}, got {paths}")
    return paths[0]


def _normalized(row: dict[str, str]) -> str:
    return "\0".join(row[field].strip() for field in FIELDS)


def _rows_and_fingerprints(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    rows: list[dict[str, str]] = []
    fingerprints: set[str] = set()
    for value in _iter_objects(path):
        if not isinstance(value, dict) or not all(
            isinstance(value.get(field), str) for field in FIELDS
        ):
            raise ValueError(f"selected source is not pure system/prompt/response SFT: {path}")
        row = {field: value[field] for field in FIELDS}
        rows.append(row)
        fingerprints.add(hashlib.sha256(_normalized(row).encode()).hexdigest())
    if not rows:
        raise ValueError(f"empty selected source: {path}")
    return rows, fingerprints


class Encoder:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True,
        )
        self.template = copy.deepcopy(TEMPLATES[TEMPLATE])
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
        source_tokens = sum(len(source) for source, _ in pairs)
        label_tokens = sum(len(label) for _, label in pairs)
        if self.template.efficient_eos:
            label_tokens += 1
        return source_tokens + label_tokens, label_tokens


def _distribution(values: list[int]) -> dict[str, Any]:
    return {
        "minimum": min(values),
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.pstdev(values),
        "p10": percentile(values, 10),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "maximum": max(values),
    }


def _curve(lengths: list[int]) -> list[dict[str, Any]]:
    total_tokens = sum(lengths)
    result: list[dict[str, Any]] = []
    for cutoff in CUTOFFS:
        capacity = cutoff - 1
        retained_lengths = [min(value, capacity) for value in lengths]
        packs = _pack_in_worker_shards(
            retained_lengths,
            capacity=capacity,
            workers=WORKERS,
        )
        counts = [len(pack) for pack in packs]
        retained_tokens = sum(retained_lengths)
        sample_distribution = _distribution(counts)
        contracts: list[dict[str, Any]] = []
        for gpu_count in GPU_COUNTS:
            for target_gbs in TARGET_GBS_VALUES:
                contract = derive_packing_gbs_contract(
                    target_gbs=target_gbs,
                    data_parallel=gpu_count,
                    samples_per_pack=sample_distribution,
                    epsilon_gbs=0.10,
                    maximum_center_relative_error=0.05,
                )
                contracts.append(
                    {
                        "gpu_count": gpu_count,
                        "target_gbs": target_gbs,
                        "gradient_accumulation_steps": contract[
                            "gradient_accumulation_steps"
                        ],
                        "expected_sample_gbs": contract[
                            "expected_epoch_sample_gbs"
                        ],
                        "relative_error": contract[
                            "expected_epoch_sample_gbs_relative_error"
                        ],
                        "admissible": contract["gates"]["candidate_admissible"],
                        "reason_codes": contract["gates"]["reason_codes"],
                    }
                )
        result.append(
            {
                "cutoff_len": cutoff,
                "packing_capacity": capacity,
                "packs": len(packs),
                "pack_utilization": retained_tokens / (len(packs) * capacity),
                "model_facing_pack_fill_ratio": retained_tokens / (len(packs) * cutoff),
                "sequence_reduction_ratio": 1.0 - len(packs) / len(lengths),
                "sample_truncation_rate": sum(value > capacity for value in lengths)
                / len(lengths),
                "tokens_retained_ratio": retained_tokens / total_tokens,
                "samples_per_pack": sample_distribution,
                "gbs_contract_grid": contracts,
            }
        )
    return result


def _tokenizer_binding() -> dict[str, Any]:
    names = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    files = {
        name: sha256_file(TOKENIZER / name)
        for name in names
        if (TOKENIZER / name).is_file()
    }
    return {
        "path": str(TOKENIZER.resolve()),
        "template": TEMPLATE,
        "files": files,
        "files_sha256": sha256_json(files),
    }


def prepare(output: Path) -> dict[str, Any]:
    prior_fingerprints: set[str] = set()
    prior_files = 0
    for root in PRIOR_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            prior_files += 1
            for value in _iter_objects(path):
                if isinstance(value, dict) and all(
                    isinstance(value.get(field), str) for field in FIELDS
                ):
                    prior_fingerprints.add(
                        hashlib.sha256(_normalized(value).encode()).hexdigest()
                    )

    selected_rows: dict[str, list[dict[str, str]]] = {}
    selected_fingerprints: dict[str, set[str]] = {}
    paths: dict[str, Path] = {}
    for _, dataset_id, _, _ in SOURCES:
        path = _find_source(INPUT_ROOT, dataset_id)
        rows, fingerprints = _rows_and_fingerprints(path)
        if 1.0 - len(fingerprints) / len(rows) > 0.05:
            raise ValueError(f"within-source duplicate fraction exceeds 5%: {dataset_id}")
        prior_overlap = len(fingerprints & prior_fingerprints) / len(fingerprints)
        if prior_overlap > 0.05:
            raise ValueError(f"prior-source containment exceeds 5%: {dataset_id}")
        selected_rows[dataset_id] = rows
        selected_fingerprints[dataset_id] = fingerprints
        paths[dataset_id] = path

    overlap_rows: list[dict[str, Any]] = []
    ids = [dataset_id for _, dataset_id, _, _ in SOURCES]
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            shared = len(selected_fingerprints[left] & selected_fingerprints[right])
            left_containment = shared / len(selected_fingerprints[left])
            right_containment = shared / len(selected_fingerprints[right])
            if max(left_containment, right_containment) > 0.05:
                raise ValueError(f"selected-source containment exceeds 5%: {left}, {right}")
            if shared:
                overlap_rows.append(
                    {
                        "left_dataset_id": left,
                        "right_dataset_id": right,
                        "shared_rows": shared,
                        "left_containment": left_containment,
                        "right_containment": right_containment,
                    }
                )

    encoder = Encoder()
    token_cache: dict[str, tuple[int, int]] = {}
    profiles: list[dict[str, Any]] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for workload_id, dataset_id, split_role, shape_role in SOURCES:
        lengths: list[int] = []
        label_lengths: list[int] = []
        for row in selected_rows[dataset_id]:
            fingerprint = hashlib.sha256(_normalized(row).encode()).hexdigest()
            encoded = token_cache.get(fingerprint)
            if encoded is None:
                encoded = encoder.encode(row)
                token_cache[fingerprint] = encoded
            lengths.append(encoded[0])
            label_lengths.append(encoded[1])
        source_path = paths[dataset_id]
        profile: dict[str, Any] = {
            "schema": "sft_packing_data_profile/final_business_v1",
            "workload_id": workload_id,
            "dataset_id": dataset_id,
            "split_role": split_role,
            "shape_role": shape_role,
            "source_binding": {
                "local_path": str(source_path.resolve()),
                "remote_key": "datasets/" + str(source_path.relative_to(INPUT_ROOT)),
                "revision": int(source_path.relative_to(INPUT_ROOT).parts[1]),
                "size_bytes": source_path.stat().st_size,
                "sha256": sha256_file(source_path),
            },
            "records": len(lengths),
            "unique_normalized_rows": len(selected_fingerprints[dataset_id]),
            "within_source_duplicate_fraction": 1.0
            - len(selected_fingerprints[dataset_id]) / len(lengths),
            "length_tokens": _distribution(lengths),
            "label_tokens": _distribution(label_lengths),
            "packing_curve": _curve(lengths),
        }
        profile["profile_sha256"] = sha256_json(profile)
        profile_path = OUTPUT_DIR / f"{workload_id.lower()}_{dataset_id}.json"
        write_json(profile_path, profile)
        profiles.append(
            {
                "workload_id": workload_id,
                "dataset_id": dataset_id,
                "split_role": split_role,
                "shape_role": shape_role,
                "profile_path": str(profile_path.resolve()),
                "profile_sha256": sha256_file(profile_path),
                "records": len(lengths),
                "p50_tokens": profile["length_tokens"]["p50"],
                "p90_tokens": profile["length_tokens"]["p90"],
                "maximum_tokens": profile["length_tokens"]["maximum"],
            }
        )

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "input_root": str(INPUT_ROOT.resolve()),
        "prior_overlap_scan": {
            "roots": [str(root.resolve()) for root in PRIOR_ROOTS],
            "jsonl_files": prior_files,
            "unique_normalized_rows": len(prior_fingerprints),
        },
        "tokenizer_binding": _tokenizer_binding(),
        "packer_binding": {
            "implementation": "llamafactory-compatible largest-fitting greedy knapsack",
            "implementation_path": str(
                (ROOT / "scripts" / "static_packing_predictor.py").resolve()
            ),
            "implementation_sha256": sha256_file(
                ROOT / "scripts" / "static_packing_predictor.py"
            ),
            "preprocessing_num_workers": WORKERS,
            "packing_capacity": "cutoff_len - 1",
            "worker_scope": "independent contiguous dataset-map shards",
        },
        "product_cache_contract": {
            "cutoff_len_candidates": list(CUTOFFS),
            "stored_per_cutoff": [
                "pack_utilization",
                "samples_per_pack.mean",
                "samples_per_pack.p99",
                "samples_per_pack.maximum",
                "sample_truncation_rate",
                "tokens_retained_ratio",
            ],
            "recommendation_time_raw_rows_required": False,
            "recommendation_time_raw_token_lengths_required": False,
        },
        "split_contract": {
            "fit_source_count": sum(row[2] == "fit" for row in SOURCES),
            "prospective_holdout_source_count": sum(
                row[2] == "prospective_holdout" for row in SOURCES
            ),
            "holdout_may_tune_coefficients_or_margins": False,
            "maximum_pairwise_normalized_row_containment": 0.05,
            "maximum_prior_screen_containment": 0.05,
            "maximum_within_source_duplicate_fraction": 0.05,
        },
        "selected_source_count": len(SOURCES),
        "selected_record_count": sum(row["records"] for row in profiles),
        "profiles": profiles,
        "nonzero_selected_pair_overlaps": overlap_rows,
        "unique_tokenized_rows": len(token_cache),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(output, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = prepare(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "selected_source_count": report["selected_source_count"],
                "selected_record_count": report["selected_record_count"],
                "fit_sources": report["split_contract"]["fit_source_count"],
                "holdout_sources": report["split_contract"][
                    "prospective_holdout_source_count"
                ],
                "nonzero_pair_overlaps": len(report["nonzero_selected_pair_overlaps"]),
                "manifest_sha256": report["manifest_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
