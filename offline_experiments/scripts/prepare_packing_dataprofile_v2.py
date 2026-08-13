#!/usr/bin/env python3
"""Freeze Packing DataProfile v2 and screen W1-W9 without using a GPU.

The expensive path belongs to dataset upload/calibration time.  It may read a
frozen token-length snapshot and run the exact static packer.  The emitted
profile contains only aggregate statistics and a compact cutoff curve, so the
recommendation path never needs raw rows, token IDs, or the length vector.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable

import pyarrow.parquet as pq

from common import ARTIFACT_DIR, ROOT, percentile, sha256_file, sha256_json, write_json
from packing_gbs_contract import derive_packing_gbs_contract
from static_packing_predictor import _pack_in_worker_shards, load_policy


SCHEMA_ID = "sft_packing_data_profile/v2"
MANIFEST_SCHEMA = "sft_packing_data_profile_manifest/v2"
SCREEN_SCHEMA = "sft_packing_cutoff_dp_gbs_screen/v2"
MODEL_PATH = Path("/wanqing-models/Qwen3-8B")
MODEL_ID = "qwen3_8b"
TEMPLATE = "qwen3_nothink"
WORKERS = 8
MODEL_MAX = 40_960
TARGET_GBS_VALUES = (32, 64, 128)
DATA_PARALLEL_VALUES = (1, 2, 4)
EPSILON_GBS = 0.10
CENTER_ERROR_MAX = 0.05
SEED = 20260804
SAMPLE_RECORDS = 4_096

OUTPUT_DIR = ARTIFACT_DIR / "packing_dataprofile_v2"
JSON_SCHEMA = ARTIFACT_DIR / "packing_data_profile_schema_v2.json"
MANIFEST = ARTIFACT_DIR / "packing_data_profiles_w1_w9_manifest_v2.json"
SCREEN = ARTIFACT_DIR / "packing_cutoff_dp_gbs_screen_w1_w9_v2.json"
SCREEN_MARKDOWN = ARTIFACT_DIR / "packing_cutoff_dp_gbs_screen_w1_w9_v2.md"
POLICY = ARTIFACT_DIR / "static_packing_policy_v1.json"

PROFILE_ROOT = ARTIFACT_DIR / "real_business_packing_cutoff_mbs_v1" / "profiles"
DATASET_PROFILE_ROOT = ARTIFACT_DIR / "dataset_profiles"
OPEN_CODE = (
    ROOT
    / "data/packing_profile_v2_sources/opencodeinstruct/data/train-00000-of-00050.parquet"
)
ORCA_MATH = (
    ROOT
    / "data/packing_profile_v2_sources/orca_math/data/train-00000-of-00001.parquet"
)


@dataclass(frozen=True)
class ExistingSpec:
    workload_id: str
    display_name: str
    role: str
    path: Path
    coverage_status: str
    auto_recommendation_eligible: bool
    diagnostic_cutoffs: tuple[int, ...] = ()


EXISTING = (
    ExistingSpec(
        "W1", "极短集中", "high_samples_per_pack",
        PROFILE_ROOT / "real_177870_short_qwen3_v1.qwen3_nothink.jsonl",
        "calibration_snapshot", True,
    ),
    ExistingSpec(
        "W2", "短文本稀有长尾", "rare_long_tail",
        PROFILE_ROOT / "real_71014_short_tail_qwen3_v1.qwen3_nothink.jsonl",
        "calibration_snapshot", True,
    ),
    ExistingSpec(
        "W3", "自然多轮中等长度", "natural_multiturn",
        DATASET_PROFILE_ROOT / "multiturn_4096.qwen3_nothink.jsonl",
        "calibration_snapshot", True,
    ),
    ExistingSpec(
        "W4", "宽长尾", "broad_long_tail",
        PROFILE_ROOT / "real_4500_content_longtail_qwen3_v1.qwen3_nothink.jsonl",
        "calibration_snapshot", True,
    ),
    ExistingSpec(
        "W5", "近 cutoff 饱和", "near_cutoff_saturated",
        DATASET_PROFILE_ROOT / "longcontext_16384.qwen3_nothink.jsonl",
        "calibration_snapshot", True,
    ),
    ExistingSpec(
        "W6", "高截断压力", "truncation_pressure_diagnostic",
        DATASET_PROFILE_ROOT / "longcontext_32768.qwen3_nothink.jsonl",
        "diagnostic_only", False, (8_192, 16_384),
    ),
)


def schema_document() -> dict[str, Any]:
    nonnegative = {"type": "number", "minimum": 0}
    positive = {"type": "number", "exclusiveMinimum": 0}
    binding = {
        "type": "object",
        "required": ["path", "sha256", "role"],
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "role": {"type": "string", "minLength": 1},
            "repo_id": {"type": "string"},
            "revision": {"type": "string"},
            "sampling_rule": {"type": "string"},
        },
        "additionalProperties": True,
    }
    curve = {
        "type": "object",
        "required": [
            "cutoff_len", "packing_capacity", "pack_utilization", "packs",
            "samples_per_pack", "sample_truncation_rate", "tokens_retained_ratio",
        ],
        "properties": {
            "cutoff_len": {"type": "integer", "minimum": 2},
            "packing_capacity": {"type": "integer", "minimum": 1},
            "pack_utilization": {"type": "number", "minimum": 0, "maximum": 1},
            "packs": {"type": "integer", "minimum": 1},
            "sample_truncation_rate": {"type": "number", "minimum": 0, "maximum": 1},
            "tokens_retained_ratio": {"type": "number", "minimum": 0, "maximum": 1},
            "samples_per_pack": {
                "type": "object",
                "required": ["minimum", "mean", "standard_deviation", "p50", "p90", "p95", "p99", "maximum"],
                "properties": {
                    "minimum": positive, "mean": positive,
                    "standard_deviation": nonnegative,
                    "p50": positive, "p90": positive, "p95": positive,
                    "p99": positive, "maximum": positive,
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": True,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "Packing DataProfile v2",
        "type": "object",
        "required": [
            "schema", "profile_id", "workload_id", "profile_revision",
            "source_bindings", "tokenizer_binding", "packer_fingerprint",
            "aggregate", "packing_curve", "recommendation_contract",
        ],
        "properties": {
            "schema": {"const": SCHEMA_ID},
            "profile_id": {"type": "string", "minLength": 1},
            "workload_id": {"type": "string", "pattern": "^W[1-9]$"},
            "profile_revision": {"type": "string", "minLength": 1},
            "source_bindings": {"type": "array", "minItems": 1, "items": binding},
            "tokenizer_binding": {"type": "object"},
            "packer_fingerprint": {"type": "object"},
            "aggregate": {
                "type": "object",
                "required": ["records", "length_tokens", "label_tokens", "turns"],
                "properties": {
                    "records": {"type": "integer", "minimum": 1},
                    "length_tokens": {"type": "object"},
                    "label_tokens": {"type": "object"},
                    "turns": {"type": "object"},
                },
            },
            "packing_curve": {"type": "array", "minItems": 1, "items": curve},
            "recommendation_contract": {
                "type": "object",
                "required": ["raw_rows_read", "raw_lengths_read", "full_packer_run"],
                "properties": {
                    "raw_rows_read": {"const": False},
                    "raw_lengths_read": {"const": False},
                    "full_packer_run": {"const": False},
                },
            },
        },
        "additionalProperties": True,
    }


def _binding(path: Path, *, role: str, **extra: Any) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "role": role, **extra}


def _read_profile(path: Path) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            length = int(row["total_tokens"])
            label = int(row.get("label_tokens", 0))
            turns = int(row.get("turns", 2))
            if length <= 0 or label < 0 or label > length or turns <= 0:
                raise ValueError(f"invalid profile row {path}:{line_number}")
            rows.append({
                "sample_id": str(row.get("sample_id") or f"{path.stem}:{line_number}"),
                "total_tokens": length,
                "label_tokens": label,
                "turns": turns,
            })
    if not rows:
        raise ValueError(f"empty profile: {path}")
    return rows


class TrainingEncoder:
    def __init__(self) -> None:
        # Keep these training-runtime-only imports off the pure schema/test path.
        from transformers import AutoTokenizer
        from llamafactory.data.template import TEMPLATES

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, use_fast=True, local_files_only=True
        )
        self.template = copy.deepcopy(TEMPLATES[TEMPLATE])
        self.template.fix_special_tokens(self.tokenizer)

    def encode(self, messages: list[dict[str, str]]) -> tuple[int, int]:
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
        conversation = messages[1:] if system is not None else messages
        pairs = self.template.encode_multiturn(
            self.tokenizer, conversation, system=system, tools=None
        )
        source_tokens = sum(len(source_ids) for source_ids, _ in pairs)
        label_tokens = sum(len(target_ids) for _, target_ids in pairs)
        if self.template.efficient_eos:
            label_tokens += 1
        return source_tokens + label_tokens, label_tokens


def _reservoir(path: Path, *, fields: tuple[str, ...], limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    seen = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=2_048, columns=list(fields)):
        for row in batch.to_pylist():
            seen += 1
            if len(selected) < limit:
                selected.append(row)
            else:
                replacement = rng.randrange(seen)
                if replacement < limit:
                    selected[replacement] = row
    if len(selected) != min(limit, seen):
        raise ValueError(f"reservoir sampling failed for {path}")
    rng.shuffle(selected)
    return selected


def _external_rows(encoder: TrainingEncoder) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    code_raw = _reservoir(
        OPEN_CODE, fields=("id", "input", "output"), limit=SAMPLE_RECORDS, seed=SEED + 8
    )
    math_raw = _reservoir(
        ORCA_MATH, fields=("question", "answer"), limit=SAMPLE_RECORDS, seed=SEED + 9
    )
    outputs: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for workload_id, raw_rows, make_messages in (
        (
            "W8", code_raw,
            lambda row: [
                {"role": "user", "content": str(row["input"])},
                {"role": "assistant", "content": str(row["output"])},
            ],
        ),
        (
            "W9", math_raw,
            lambda row: [
                {"role": "user", "content": str(row["question"])},
                {"role": "assistant", "content": str(row["answer"])},
            ],
        ),
    ):
        profiled = []
        for index, row in enumerate(raw_rows):
            messages = make_messages(row)
            total, label = encoder.encode(messages)
            profiled.append({
                "sample_id": str(row.get("id") or f"{workload_id}:{index}"),
                "total_tokens": total,
                "label_tokens": label,
                "turns": len(messages),
            })
        outputs[workload_id] = (profiled, raw_rows)
    return outputs


def _candidate_cutoffs(maximum: int, diagnostics: Iterable[int] = ()) -> list[int]:
    data_min = math.ceil((maximum + 1) / 512) * 512
    values = set(int(value) for value in diagnostics)
    if data_min <= MODEL_MAX:
        values.add(data_min)
        for value in (data_min - 512, data_min + 512):
            if value >= 512:
                values.add(value)
    values.update(range(512, min(4_096, MODEL_MAX) + 1, 512))
    values.update(range(4_096, min(16_384, MODEL_MAX) + 1, 2_048))
    values.update(range(16_384, MODEL_MAX + 1, 4_096))
    values.update((7_168, 29_184, 31_744, 32_256, 32_768, MODEL_MAX))
    return sorted(value for value in values if 512 <= value <= MODEL_MAX)


def _distribution(values: list[int]) -> dict[str, Any]:
    return {
        "minimum": min(values),
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.pstdev(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "maximum": max(values),
    }


def _curve(lengths: list[int], cutoffs: list[int]) -> list[dict[str, Any]]:
    total = sum(lengths)
    rows = []
    for cutoff in cutoffs:
        capacity = cutoff - 1
        packed_lengths = [min(value, capacity) for value in lengths]
        knapsacks = _pack_in_worker_shards(
            packed_lengths, capacity=capacity, workers=WORKERS
        )
        counts = [len(pack) for pack in knapsacks]
        retained = sum(packed_lengths)
        rows.append({
            "cutoff_len": cutoff,
            "packing_capacity": capacity,
            "packs": len(knapsacks),
            "pack_utilization": retained / (len(knapsacks) * capacity),
            "model_facing_pack_fill_ratio": retained / (len(knapsacks) * cutoff),
            "sample_truncation_rate": sum(value > capacity for value in lengths) / len(lengths),
            "tokens_retained_ratio": retained / total,
            "samples_per_pack": _distribution(counts),
        })
    return rows


def _profile(
    *, workload_id: str, display_name: str, role: str,
    rows: list[dict[str, Any]], source_bindings: list[dict[str, Any]],
    coverage_status: str, auto_recommendation_eligible: bool,
    diagnostic_cutoffs: Iterable[int] = (),
) -> dict[str, Any]:
    lengths = [int(row["total_tokens"]) for row in rows]
    labels = [int(row["label_tokens"]) for row in rows]
    turns = [int(row["turns"]) for row in rows]
    curve = _curve(lengths, _candidate_cutoffs(max(lengths), diagnostic_cutoffs))
    profile: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "profile_id": f"packing_{workload_id.lower()}_qwen3_nothink_v2",
        "workload_id": workload_id,
        "display_name": display_name,
        "profile_role": role,
        "profile_revision": "2026-08-04-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage_status": coverage_status,
        "auto_recommendation_eligible": auto_recommendation_eligible,
        "source_bindings": source_bindings,
        "tokenizer_binding": {
            "model_id": MODEL_ID,
            "path": str(MODEL_PATH),
            "template": TEMPLATE,
            "tokenizer_config_sha256": sha256_file(MODEL_PATH / "tokenizer_config.json"),
            "model_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        },
        "packer_fingerprint": {
            "algorithm_id": load_policy(POLICY)["packing_algorithm"]["id"],
            "preprocessing_num_workers": WORKERS,
            "worker_sharding": "contiguous_equal_ranges",
            "bin_packing": "largest_fitting_item_greedy",
            "capacity": "cutoff_len_minus_one",
            "profile_order": "frozen_snapshot_order",
        },
        "aggregate": {
            "records": len(rows),
            "length_tokens": _distribution(lengths),
            "label_tokens": {
                **_distribution(labels),
                "ratio_of_total": sum(labels) / sum(lengths),
            },
            "turns": _distribution(turns),
        },
        "packing_curve": curve,
        "recommendation_contract": {
            "raw_rows_read": False,
            "raw_lengths_read": False,
            "full_packer_run": False,
            "cache_lookup_key_fields": [
                "profile_revision", "tokenizer_binding", "packer_fingerprint", "cutoff_len"
            ],
        },
    }
    profile["profile_fingerprint_sha256"] = sha256_json(profile)
    return profile


def _screen(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    summaries = []
    for profile in profiles:
        maximum = int(profile["aggregate"]["length_tokens"]["maximum"])
        workload_rows = []
        for point in profile["packing_curve"]:
            cutoff = int(point["cutoff_len"])
            covers_max = int(point["packing_capacity"]) >= maximum
            for dp in DATA_PARALLEL_VALUES:
                for target in TARGET_GBS_VALUES:
                    contract = derive_packing_gbs_contract(
                        target_gbs=target,
                        data_parallel=dp,
                        samples_per_pack=point["samples_per_pack"],
                        epsilon_gbs=EPSILON_GBS,
                        maximum_center_relative_error=CENTER_ERROR_MAX,
                    )
                    gates = contract["gates"]
                    admissible = bool(
                        covers_max
                        and gates["candidate_admissible"]
                        and profile["auto_recommendation_eligible"]
                    )
                    reason_codes = []
                    if not covers_max:
                        reason_codes.append("cutoff_truncates_profile")
                    reason_codes.extend(gates["reason_codes"])
                    if not profile["auto_recommendation_eligible"]:
                        reason_codes.append("profile_not_auto_recommendation_eligible")
                    row = {
                        "workload_id": profile["workload_id"],
                        "profile_id": profile["profile_id"],
                        "cutoff_len": cutoff,
                        "data_parallel": dp,
                        "target_gbs": target,
                        "gradient_accumulation_steps": contract["gradient_accumulation_steps"],
                        "expected_epoch_sample_gbs": contract["expected_epoch_sample_gbs"],
                        "expected_epoch_sample_gbs_relative_error": contract[
                            "expected_epoch_sample_gbs_relative_error"
                        ],
                        "n_pack_mean": point["samples_per_pack"]["mean"],
                        "n_pack_step_p99": point["samples_per_pack"]["p99"],
                        "global_microstep_sample_gbs_p99": contract[
                            "global_microstep_sample_gbs"
                        ]["p99"],
                        "cutoff_covers_profile_max": covers_max,
                        "center_integer_representable": gates["center_integer_representable"],
                        "gbs_controllable_at_ga_floor": gates[
                            "gbs_controllable_at_ga_floor"
                        ],
                        "candidate_admissible": admissible,
                        "reason_codes": reason_codes,
                        "memory_gate_status": "not_evaluated_cpu_screen",
                    }
                    rows.append(row)
                    workload_rows.append(row)
        admissible_rows = [row for row in workload_rows if row["candidate_admissible"]]
        summaries.append({
            "workload_id": profile["workload_id"],
            "display_name": profile["display_name"],
            "coverage_status": profile["coverage_status"],
            "auto_recommendation_eligible": profile["auto_recommendation_eligible"],
            "screened_candidates": len(workload_rows),
            "admissible_candidates": len(admissible_rows),
            "admissible_by_dp_target": {
                f"dp{dp}_g{target}": sum(
                    row["candidate_admissible"]
                    and row["data_parallel"] == dp
                    and row["target_gbs"] == target
                    for row in workload_rows
                )
                for dp in DATA_PARALLEL_VALUES
                for target in TARGET_GBS_VALUES
            },
        })
    report: dict[str, Any] = {
        "schema": SCREEN_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_execution_used": False,
        "recommendation_path_raw_data_read": False,
        "contract": {
            "model_max_cutoff": MODEL_MAX,
            "data_parallel_values": list(DATA_PARALLEL_VALUES),
            "target_gbs_values": list(TARGET_GBS_VALUES),
            "epsilon_gbs": EPSILON_GBS,
            "maximum_center_relative_error": CENTER_ERROR_MAX,
            "memory_gate": "deferred_to_hardware_candidate_stage",
        },
        "profile_summaries": summaries,
        "rows": rows,
    }
    report["report_sha256"] = sha256_json(report)
    return report


def _markdown(screen: dict[str, Any]) -> str:
    lines = [
        "# Packing W1-W9 cutoff×DP×GBS CPU 复筛 v2", "",
        "本报告只使用上传期冻结画像和静态 packer，不使用 GPU。admissible 尚未包含硬件显存 gate。", "",
        "| 画像 | 覆盖状态 | 自动推荐画像 | screened | admissible | DP1/G64 | DP2/G64 | DP2/G128 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in screen["profile_summaries"]:
        counts = row["admissible_by_dp_target"]
        lines.append(
            f"| {row['workload_id']} {row['display_name']} | {row['coverage_status']} | "
            f"{row['auto_recommendation_eligible']} | {row['screened_candidates']} | "
            f"{row['admissible_candidates']} | {counts['dp1_g64']} | "
            f"{counts['dp2_g64']} | {counts['dp2_g128']} |"
        )
    lines.extend(("", "W6 保持 diagnostic-only；W8/W9 是按冻结公开源新生成的 calibration snapshot，在 prospective 验收前不自动发布。", ""))
    return "\n".join(lines)


def prepare() -> dict[str, Any]:
    for path in (POLICY, OPEN_CODE, ORCA_MATH, *[spec.path for spec in EXISTING]):
        if not path.is_file():
            raise FileNotFoundError(path)
    write_json(JSON_SCHEMA, schema_document())

    profiles: list[dict[str, Any]] = []
    existing_rows: dict[str, list[dict[str, Any]]] = {}
    for spec in EXISTING:
        rows = _read_profile(spec.path)
        existing_rows[spec.workload_id] = rows
        profiles.append(_profile(
            workload_id=spec.workload_id,
            display_name=spec.display_name,
            role=spec.role,
            rows=rows,
            source_bindings=[_binding(spec.path, role="frozen_token_profile")],
            coverage_status=spec.coverage_status,
            auto_recommendation_eligible=spec.auto_recommendation_eligible,
            diagnostic_cutoffs=spec.diagnostic_cutoffs,
        ))

    short = _read_profile(DATASET_PROFILE_ROOT / "short_512.qwen3_nothink.jsonl")
    long = existing_rows["W5"]
    bimodal = short[:1_000] + long[:1_000]
    profiles.append(_profile(
        workload_id="W7", display_name="双峰混合", role="deterministic_bimodal_50_50",
        rows=bimodal,
        source_bindings=[
            _binding(DATASET_PROFILE_ROOT / "short_512.qwen3_nothink.jsonl", role="short_mode"),
            _binding(EXISTING[4].path, role="near_cutoff_mode"),
        ],
        coverage_status="derived_calibration_snapshot",
        auto_recommendation_eligible=True,
    ))

    external = _external_rows(TrainingEncoder())
    profiles.append(_profile(
        workload_id="W8", display_name="结构化/代码", role="code_structured",
        rows=external["W8"][0],
        source_bindings=[_binding(
            OPEN_CODE, role="frozen_public_source_shard",
            repo_id="nvidia/OpenCodeInstruct",
            revision="8f3ba5bafe4d6e8db46082cf7ae6741bc370604d",
            sampling_rule=f"seeded reservoir sample n={SAMPLE_RECORDS} from shard 0/50 seed={SEED + 8}",
        )],
        coverage_status="new_calibration_snapshot_pending_prospective_validation",
        auto_recommendation_eligible=False,
    ))
    profiles.append(_profile(
        workload_id="W9", display_name="高标签占比推理", role="reasoning_long_label",
        rows=external["W9"][0],
        source_bindings=[_binding(
            ORCA_MATH, role="frozen_public_source",
            repo_id="microsoft/orca-math-word-problems-200k",
            revision="29255d1770cc4eac66e5e7fa378cba542c026350",
            sampling_rule=f"seeded reservoir sample n={SAMPLE_RECORDS} from full parquet seed={SEED + 9}",
        )],
        coverage_status="new_calibration_snapshot_pending_prospective_validation",
        auto_recommendation_eligible=False,
    ))

    profiles.sort(key=lambda row: int(str(row["workload_id"])[1:]))
    if [row["workload_id"] for row in profiles] != [f"W{value}" for value in range(1, 10)]:
        raise ValueError("W1-W9 profile set is not exact")
    profile_bindings = []
    for profile in profiles:
        path = OUTPUT_DIR / f"{str(profile['workload_id']).lower()}_packing_dataprofile_v2.json"
        write_json(path, profile)
        profile_bindings.append({
            "workload_id": profile["workload_id"],
            "profile_id": profile["profile_id"],
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "coverage_status": profile["coverage_status"],
            "auto_recommendation_eligible": profile["auto_recommendation_eligible"],
        })

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_binding": _binding(JSON_SCHEMA, role="json_schema"),
        "profile_count": len(profiles),
        "workload_ids": [row["workload_id"] for row in profiles],
        "profiles": profile_bindings,
        "recommendation_path_contract": {
            "online_tokenization": False,
            "raw_dataset_read": False,
            "raw_length_vector_read": False,
            "full_static_packer_run": False,
            "compact_curve_lookup_only": True,
        },
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(MANIFEST, manifest)

    screen = _screen(profiles)
    screen["profile_manifest"] = _binding(MANIFEST, role="profile_manifest")
    screen["report_sha256"] = sha256_json({k: v for k, v in screen.items() if k != "report_sha256"})
    write_json(SCREEN, screen)
    SCREEN_MARKDOWN.write_text(_markdown(screen), encoding="utf-8")
    return {
        "schema": str(JSON_SCHEMA), "manifest": str(MANIFEST),
        "screen": str(SCREEN), "screen_markdown": str(SCREEN_MARKDOWN),
        "profiles": len(profiles), "screen_rows": len(screen["rows"]),
    }


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
