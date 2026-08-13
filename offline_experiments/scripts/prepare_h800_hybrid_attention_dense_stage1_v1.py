#!/usr/bin/env python3
"""Prepare the first dense hybrid-attention H800 experiment batch.

This CPU-only command creates model-bound exact-length text controls, an
immutable model/architecture inventory, a four-job compatibility canary and a
208-job formal queue.  It never creates an approval and never starts GPU work.

The formal design covers a full-attention control (Qwen3-8B) and three hybrid
dense checkpoints (Qwen3.5-4B/9B and Qwen3.6-27B).  MoE and real-media vision
training are separate routes and are intentionally outside this first batch.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    DATA_DIR,
    MATRIX_DIR,
    percentile,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from hybrid_attention_memory_features import architecture_signature
from inventory_models import inventory_model
from prepare_h800_qwen35_vl_supplement_v1 import (
    RUNTIME_CONTRACT,
    _environment_overlay,
)
from prepare_h800_vl_calibration_profiles_v1 import TrainingEncoder
from run_job import validate_job

CAMPAIGN_ID = "h800_hybrid_attention_dense_stage1_20260812_v1"
CANARY_PHASE_ID = "h800_hybrid_attention_dense_stage1_canary_v1"
FORMAL_PHASE_ID = "h800_hybrid_attention_dense_stage1_formal_v1"
JOB_SCHEMA = "sft_h800_hybrid_attention_dense_stage1_job/v1"
DESIGN_SCHEMA = "sft_h800_hybrid_attention_dense_stage1_design/v1"
PROFILE_SCHEMA = "sft_h800_exact_text_profile/v1"
GENERATED_AT_UTC = "2026-08-12T00:00:00+00:00"
TARGET_GBS = 64
ROWS_PER_DATASET = 64
LENGTHS = (2048, 8192, 16384)
AUTHORIZED_GPU_IDS = (0, 1, 2, 3, 4, 5, 6)

QWEN3_INVENTORY = ARTIFACT_DIR / "model_inventory.json"
QWEN35_INVENTORY = (
    ARTIFACT_DIR / "h800_qwen35_vl_supplement_model_inventory_v1.json"
)
MODEL_INVENTORY = (
    ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_model_inventory_v1.json"
)
FEATURE_SOURCE = Path(__file__).resolve().parent / "hybrid_attention_memory_features.py"

SOURCE_DATASETS = {
    "业务短样本": DATA_DIR
    / "bounded_memory_v2_fresh_holdout_v1"
    / "fresh_s3_rare_tail_short_v1.jsonl",
    "业务长尾样本": DATA_DIR
    / "bounded_memory_v2_fresh_holdout_v1"
    / "fresh_s3_broad_nontruncated_longtail_v1.jsonl",
}

DATA_OUTPUT_DIR = DATA_DIR / "hybrid_attention_dense_stage1_v1"
DATASET_REGISTRY_DIR = DATA_OUTPUT_DIR / "registry"
DATASET_REGISTRY = DATASET_REGISTRY_DIR / "dataset_info.json"
PROFILE_OUTPUT_DIR = (
    ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_profiles_v1"
)
PROFILE_MANIFEST = (
    ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_profiles_manifest_v1.json"
)
CANARY_QUEUE = (
    MATRIX_DIR / "h800_hybrid_attention_dense_stage1_canary_v1.jsonl"
)
FORMAL_QUEUE = (
    MATRIX_DIR / "h800_hybrid_attention_dense_stage1_formal_v1.jsonl"
)
DESIGN = ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_design_v1.json"

FIXED_LORA = {"rank": 32, "alpha": 32, "dropout": 0.0, "target": "all"}

MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "qwen3_8b",
        "source_inventory": "qwen3",
        "family": "qwen3",
        "release_family": "qwen3",
        "path": "/wanqing-models/Qwen3-8B",
        "tokenizer_path": "/wanqing-models/Qwen3-8B",
        "template": "qwen3_nothink",
        "enable_liger_kernel": True,
    },
    {
        "id": "qwen3p5_4b",
        "source_inventory": "qwen3_5",
        "family": "qwen3_5",
        "release_family": "qwen3_5",
        "path": "/wanqing-models/Qwen3.5-4B",
        "tokenizer_path": "/wanqing-models/Qwen3.5-4B",
        "template": "qwen3_5_nothink",
        "enable_liger_kernel": True,
    },
    {
        "id": "qwen3p5_9b",
        "source_inventory": "qwen3_5",
        "family": "qwen3_5",
        "release_family": "qwen3_5",
        "path": "/wanqing-models/Qwen3.5-9B",
        "tokenizer_path": "/wanqing-models/Qwen3.5-9B",
        "template": "qwen3_5_nothink",
        "enable_liger_kernel": True,
    },
    {
        "id": "qwen3p6_27b",
        "source_inventory": "new_scan",
        "family": "qwen3_5",
        "release_family": "qwen3_6",
        "path": "/wanqing-models/Qwen3.6-27B",
        "tokenizer_path": "/wanqing-models/Qwen3.6-27B",
        "template": "qwen3_5_nothink",
        "enable_liger_kernel": True,
        "train_types": ["lora"],
    },
)

PARALLEL_ROUTES: dict[str, dict[str, Any]] = {
    "D1_Z0": {"gpu_count": 1, "zero": "none"},
    "D2_Z2": {"gpu_count": 2, "zero": "zero2"},
    "D2_Z3": {"gpu_count": 2, "zero": "zero3"},
    "D4_Z2": {"gpu_count": 4, "zero": "zero2"},
    "D4_Z3": {"gpu_count": 4, "zero": "zero3"},
}


def _source_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    if len(rows) < ROWS_PER_DATASET:
        raise ValueError(f"source has fewer than {ROWS_PER_DATASET} rows: {path}")
    return rows


def _messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = str(row.get("system") or "")
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(
        [
            {"role": "user", "content": str(row["prompt"])},
            {"role": "assistant", "content": str(row["response"])},
        ]
    )
    return messages


def _append_exact_filler(
    encoder: TrainingEncoder,
    messages: list[dict[str, str]],
    *,
    target_total_tokens: int,
) -> tuple[list[dict[str, str]], int, int]:
    base = copy.deepcopy(messages)
    base_total, base_labels = encoder.encode(base)
    if target_total_tokens < base_total:
        raise ValueError(
            f"target total {target_total_tokens} is below base {base_total}"
        )
    user_index = max(
        index for index, message in enumerate(base) if message["role"] == "user"
    )
    original = base[user_index]["content"]
    filler_tokens = target_total_tokens - base_total
    for _ in range(8):
        candidate = copy.deepcopy(base)
        candidate[user_index]["content"] = original + (" x" * filler_tokens)
        total, labels = encoder.encode(candidate)
        if labels != base_labels:
            raise ValueError("user-side filler changed assistant label tokens")
        if total == target_total_tokens:
            return candidate, total, labels
        filler_tokens += target_total_tokens - total
        if filler_tokens < 0:
            raise ValueError("exact filler crossed below zero")
    raise ValueError(f"cannot construct exact length {target_total_tokens}")


def _statistics(values: list[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("statistics require non-empty values")
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "maximum": max(values),
    }


def _existing_inventory_rows() -> dict[str, dict[str, Any]]:
    qwen3 = read_json(QWEN3_INVENTORY)
    qwen35 = read_json(QWEN35_INVENTORY)
    rows = {
        str(row["id"]): row
        for report in (qwen3, qwen35)
        for row in report.get("models") or []
    }
    return rows


def _materialize_inventory() -> dict[str, Any]:
    existing = _existing_inventory_rows()
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        if spec["source_inventory"] == "new_scan":
            entry = inventory_model(dict(spec))
        else:
            if spec["id"] not in existing:
                raise ValueError(f"missing source inventory row {spec['id']}")
            entry = copy.deepcopy(existing[spec["id"]])
            for field in (
                "family",
                "release_family",
                "path",
                "tokenizer_path",
                "template",
                "enable_liger_kernel",
            ):
                entry[field] = spec[field]
        entry["architecture_signature"] = architecture_signature(entry)
        rows.append(entry)
    report_core = {
        "schema": "sft_h800_hybrid_attention_dense_stage1_model_inventory/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": GENERATED_AT_UTC,
        "gpu_training_started": False,
        "fixed_lora": FIXED_LORA,
        "models": rows,
    }
    report = {**report_core, "report_sha256": sha256_json(report_core)}
    write_json(MODEL_INVENTORY, report)
    return report


def _materialize_profiles(
    inventory: dict[str, Any],
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, Any]]:
    source_rows = {
        source_id: _source_rows(path)
        for source_id, path in SOURCE_DATASETS.items()
    }
    registry: dict[str, Any] = {}
    profiles: dict[tuple[str, str, int], dict[str, Any]] = {}
    profile_manifest_rows: list[dict[str, Any]] = []
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model in inventory["models"]:
        model_id = str(model["id"])
        encoder = TrainingEncoder(str(model["tokenizer_path"]), str(model["template"]))
        for source_id, raw_rows in source_rows.items():
            source_path = SOURCE_DATASETS[source_id]
            for target in LENGTHS:
                output_rows: list[dict[str, Any]] = []
                profile_rows: list[dict[str, Any]] = []
                for raw in raw_rows:
                    base_messages = _messages(raw)
                    base_total, _ = encoder.encode(base_messages)
                    if base_total > target:
                        continue
                    exact_messages, total, labels = _append_exact_filler(
                        encoder,
                        base_messages,
                        target_total_tokens=target,
                    )
                    sample_id = str(raw.get("sample_id") or f"source-{len(output_rows)}")
                    output_rows.append(
                        {
                            "messages": exact_messages,
                            "sample_id": sample_id,
                            "source_dataset_id": source_id,
                        }
                    )
                    profile_rows.append(
                        {
                            "schema": PROFILE_SCHEMA,
                            "sample_id": sample_id,
                            "source_dataset_id": source_id,
                            "model_id": model_id,
                            "total_tokens": total,
                            "label_tokens": labels,
                            "target_tokens": target,
                        }
                    )
                    if len(output_rows) == ROWS_PER_DATASET:
                        break
                if len(output_rows) != ROWS_PER_DATASET:
                    raise ValueError(
                        f"not enough <= {target} rows for {model_id}/{source_id}"
                    )
                if any(row["total_tokens"] != target for row in profile_rows):
                    raise ValueError("exact-length profile construction failed")

                slug = "source_a" if source_id == "业务短样本" else "source_b"
                dataset_id = f"hybrid_dense_stage1_{model_id}_{slug}_{target}"
                data_path = DATA_OUTPUT_DIR / f"{model_id}.{slug}.{target}.jsonl"
                profile_path = PROFILE_OUTPUT_DIR / f"{model_id}.{slug}.{target}.jsonl"
                write_jsonl(data_path, output_rows)
                write_jsonl(profile_path, profile_rows)
                registry[dataset_id] = {
                    "file_name": str(data_path.resolve()),
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
                label_values = [int(row["label_tokens"]) for row in profile_rows]
                profile = {
                    "schema": "sft_h800_hybrid_attention_dense_stage1_profile/v1",
                    "model_id": model_id,
                    "template": model["template"],
                    "tokenizer_path": model["tokenizer_path"],
                    "tokenizer_json_sha256": sha256_file(
                        Path(model["tokenizer_path"]) / "tokenizer.json"
                    ),
                    "source_dataset_id": source_id,
                    "source_path": str(source_path.resolve()),
                    "source_sha256": sha256_file(source_path),
                    "dataset_id": dataset_id,
                    "data_path": str(data_path.resolve()),
                    "data_sha256": sha256_file(data_path),
                    "profile_path": str(profile_path.resolve()),
                    "profile_sha256": sha256_file(profile_path),
                    "rows": ROWS_PER_DATASET,
                    "target_total_tokens": target,
                    "all_total_tokens_exact": True,
                    "label_token_statistics": _statistics(label_values),
                    "source_sample_ids_sha256": sha256_json(
                        [row["sample_id"] for row in profile_rows]
                    ),
                }
                profiles[(model_id, source_id, target)] = profile
                profile_manifest_rows.append(profile)

    DATASET_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    write_json(DATASET_REGISTRY, registry)
    manifest_core = {
        "schema": "sft_h800_hybrid_attention_dense_stage1_profiles_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "gpu_training_started": False,
        "profiles": profile_manifest_rows,
        "checks": {
            "profile_count": len(profile_manifest_rows),
            "expected_profile_count": len(MODEL_SPECS)
            * len(SOURCE_DATASETS)
            * len(LENGTHS),
            "all_exact": all(
                row["all_total_tokens_exact"] for row in profile_manifest_rows
            ),
            "all_64_rows": all(
                row["rows"] == ROWS_PER_DATASET for row in profile_manifest_rows
            ),
        },
        "dataset_registry": {
            "path": str(DATASET_REGISTRY.resolve()),
            "sha256": sha256_file(DATASET_REGISTRY),
        },
    }
    manifest = {**manifest_core, "report_sha256": sha256_json(manifest_core)}
    write_json(PROFILE_MANIFEST, manifest)
    return profiles, manifest


def _zero_stage(zero: str) -> int:
    return 0 if zero == "none" else int(zero[-1])


def _base_job(
    *,
    model: dict[str, Any],
    profile: dict[str, Any],
    phase_id: str,
    evidence_role: str,
    design_arm: str,
    parallel_route_id: str,
    gpu_count: int,
    zero: str,
    gc: bool,
    mbs: int,
    warmup_steps: int,
    measure_steps: int,
    repeat: int,
) -> dict[str, Any]:
    denominator = gpu_count * mbs
    if TARGET_GBS % denominator:
        raise ValueError(f"GBS {TARGET_GBS} is not divisible by {denominator}")
    signature = model["architecture_signature"]
    row: dict[str, Any] = {
        "schema": JOB_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": phase_id,
        "track": "dense_hybrid_attention_memory_stage1",
        "evidence_role": evidence_role,
        "design_arm": design_arm,
        "model_id": model["id"],
        "model_family": model["family"],
        "release_family": model["release_family"],
        "model_path": model["path"],
        "tokenizer_path": model["tokenizer_path"],
        "model_parameters": int(model["actual_parameters"]),
        "template": model["template"],
        "train_type": "lora",
        "target_gbs": TARGET_GBS,
        "gpu_count": gpu_count,
        "zero": zero,
        "zero_stage": _zero_stage(zero),
        "gc": gc,
        "gradient_checkpointing": gc,
        "mbs": mbs,
        "gradient_accumulation_steps": TARGET_GBS // denominator,
        "packing": False,
        "offload": False,
        "kind": "throughput",
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "fidelity": f"{'canary' if warmup_steps == 0 else 'formal'}_{warmup_steps}plus{measure_steps}",
        "repeat": repeat,
        "parallel_route_id": parallel_route_id,
        "dataset_id": profile["dataset_id"],
        "dataset_dir": str(DATASET_REGISTRY_DIR.resolve()),
        "dataset_registry_sha256": sha256_file(DATASET_REGISTRY),
        "data_path": profile["data_path"],
        "data_sha256": profile["data_sha256"],
        "dataset_profile_path": profile["profile_path"],
        "dataset_profile_sha256": profile["profile_sha256"],
        "source_dataset_id": profile["source_dataset_id"],
        "source_dataset_sha256": profile["source_sha256"],
        "cutoff_len": int(profile["target_total_tokens"]),
        "exact_total_tokens_per_sample": int(profile["target_total_tokens"]),
        "all_total_tokens_exact": True,
        "max_samples": ROWS_PER_DATASET,
        "enable_liger_kernel": bool(model["enable_liger_kernel"]),
        "effective_kernel_path": "fa3_orig+liger_fused_ce+adamw_torch_fused",
        "gpu_type": "NVIDIA H800 140GB HBM3",
        "required_runtime_gpu_name": "NVIDIA H800",
        "hardware_id": "local_h800_140g",
        "requested_gpu_pool": list(AUTHORIZED_GPU_IDS),
        "parallel_class": "gpu_partitionable",
        "requires_external_node_idle": False,
        "declared_model_manifest_path": str(MODEL_INVENTORY.resolve()),
        "declared_model_manifest_sha256": sha256_file(MODEL_INVENTORY),
        "architecture_route": signature["architecture_route"],
        "architecture_signature_sha256": signature["signature_sha256"],
        "model_config_sha256": signature["config_sha256"],
        "num_full_attention_layers": signature["num_full_attention_layers"],
        "num_linear_attention_layers": signature["num_linear_attention_layers"],
        "feature_basis_path": str(FEATURE_SOURCE),
        "feature_basis_sha256": sha256_file(FEATURE_SOURCE),
        "visual_runtime_evidence_required": False,
        "calibration_partition": {
            "role": (
                "canary_excluded"
                if "canary" in evidence_role
                else "calibration"
            ),
            "split_unit_id": profile["source_dataset_id"],
            "policy": (
                "software_semantics_only_not_fit_or_acceptance"
                if "canary" in evidence_role
                else "source_grouped_fit_only_never_acceptance_v1"
            ),
        },
    }
    if model.get("is_vision_language"):
        row.update(
            {
                "train_scope_id": "language_only",
                "freeze_vision_tower": True,
                "freeze_multi_modal_projector": True,
                "freeze_language_model": False,
                "environment_overlay": _environment_overlay(),
            }
        )
    row["scenario_id"] = (
        f"{model['id']}__{profile['source_dataset_id']}__"
        f"s{profile['target_total_tokens']}__{parallel_route_id}__"
        f"gc{int(gc)}__mbs{mbs}__{design_arm}__r{repeat}"
    )
    row["job_id"] = stable_id("h800hybriddense", row)
    validate_job(row)
    return row


def _canary_jobs(
    models: dict[str, dict[str, Any]],
    profiles: dict[tuple[str, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    safe_routes = {
        "qwen3_8b": ("D1_Z0", PARALLEL_ROUTES["D1_Z0"]),
        "qwen3p5_4b": ("D1_Z0", PARALLEL_ROUTES["D1_Z0"]),
        "qwen3p5_9b": ("D2_Z3", PARALLEL_ROUTES["D2_Z3"]),
        "qwen3p6_27b": ("D4_Z3", PARALLEL_ROUTES["D4_Z3"]),
    }
    jobs = []
    for model_id, (route_id, route) in safe_routes.items():
        jobs.append(
            _base_job(
                model=models[model_id],
                profile=profiles[(model_id, "业务短样本", 2048)],
                phase_id=CANARY_PHASE_ID,
                evidence_role="dense_hybrid_runtime_canary",
                design_arm="canary",
                parallel_route_id=route_id,
                gpu_count=int(route["gpu_count"]),
                zero=str(route["zero"]),
                gc=True,
                mbs=1,
                warmup_steps=0,
                measure_steps=2,
                repeat=0,
            )
        )
    if len(jobs) != 4 or len({row["job_id"] for row in jobs}) != 4:
        raise ValueError("canary queue must have four unique jobs")
    return jobs


def _formal_jobs(
    models: dict[str, dict[str, Any]],
    profiles: dict[tuple[str, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []

    def add(
        *,
        model: dict[str, Any],
        source_id: str,
        length: int,
        route_id: str,
        gc: bool,
        mbs: int,
        arm: str,
        repeat: int = 0,
    ) -> None:
        route = PARALLEL_ROUTES[route_id]
        jobs.append(
            _base_job(
                model=model,
                profile=profiles[(str(model["id"]), source_id, length)],
                phase_id=FORMAL_PHASE_ID,
                evidence_role="dense_hybrid_memory_calibration",
                design_arm=arm,
                parallel_route_id=route_id,
                gpu_count=int(route["gpu_count"]),
                zero=str(route["zero"]),
                gc=gc,
                mbs=mbs,
                warmup_steps=3,
                measure_steps=10,
                repeat=repeat,
            )
        )

    for model in models.values():
        # Core 120 jobs: architecture x length x GC x data-parallel/ZeRO route.
        for length in LENGTHS:
            for gc in (True, False):
                for route_id in PARALLEL_ROUTES:
                    add(
                        model=model,
                        source_id="业务短样本",
                        length=length,
                        route_id=route_id,
                        gc=gc,
                        mbs=1,
                        arm="核心正交",
                    )
        # MBS contrast at a controlled 8K shape.
        for gc in (True, False):
            for route_id in PARALLEL_ROUTES:
                add(
                    model=model,
                    source_id="业务短样本",
                    length=8192,
                    route_id=route_id,
                    gc=gc,
                    mbs=2,
                    arm="微批扩展",
                )
        # Source invariance at the same exact 8K shape.
        for gc in (True, False):
            for route_id in PARALLEL_ROUTES:
                add(
                    model=model,
                    source_id="业务长尾样本",
                    length=8192,
                    route_id=route_id,
                    gc=gc,
                    mbs=1,
                    arm="数据源复验",
                )
        # One-card repeat quantifies allocator/run variance for each GC state.
        for gc in (True, False):
            add(
                model=model,
                source_id="业务短样本",
                length=8192,
                route_id="D1_Z0",
                gc=gc,
                mbs=1,
                arm="重复方差",
                repeat=1,
            )
    if len(jobs) != 208 or len({row["job_id"] for row in jobs}) != 208:
        raise ValueError("formal queue must have 208 unique jobs")
    return jobs


def _counts(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "jobs": len(jobs),
        "by_model": dict(sorted(Counter(row["model_id"] for row in jobs).items())),
        "by_architecture_route": dict(
            sorted(Counter(row["architecture_route"] for row in jobs).items())
        ),
        "by_gpu_count": dict(
            sorted(Counter(str(row["gpu_count"]) for row in jobs).items())
        ),
        "by_zero": dict(sorted(Counter(row["zero"] for row in jobs).items())),
        "by_gc": dict(sorted(Counter(str(row["gc"]) for row in jobs).items())),
        "by_mbs": dict(sorted(Counter(str(row["mbs"]) for row in jobs).items())),
        "by_length": dict(
            sorted(Counter(str(row["cutoff_len"]) for row in jobs).items())
        ),
        "by_source": dict(
            sorted(Counter(row["source_dataset_id"] for row in jobs).items())
        ),
        "by_design_arm": dict(
            sorted(Counter(row["design_arm"] for row in jobs).items())
        ),
    }


def _audit_formal(jobs: list[dict[str, Any]]) -> dict[str, bool]:
    expected_models = {str(row["id"]) for row in MODEL_SPECS}
    return {
        "exactly_208_jobs": len(jobs) == 208,
        "all_job_ids_unique": len({row["job_id"] for row in jobs}) == len(jobs),
        "all_models_present": {row["model_id"] for row in jobs} == expected_models,
        "every_model_has_52_jobs": all(
            sum(row["model_id"] == model_id for row in jobs) == 52
            for model_id in expected_models
        ),
        "all_parallel_routes_present": {
            row["parallel_route_id"] for row in jobs
        }
        == set(PARALLEL_ROUTES),
        "all_lengths_present": {row["cutoff_len"] for row in jobs}
        == set(LENGTHS),
        "both_gc_states_present": {row["gc"] for row in jobs} == {True, False},
        "both_mbs_values_present": {row["mbs"] for row in jobs} == {1, 2},
        "both_sources_present": {row["source_dataset_id"] for row in jobs}
        == set(SOURCE_DATASETS),
        "packing_always_disabled": all(row["packing"] is False for row in jobs),
        "all_profiles_exact": all(row["all_total_tokens_exact"] for row in jobs),
        "all_jobs_validate": all((validate_job(row) is None) for row in jobs),
    }


def prepare() -> dict[str, Any]:
    if not RUNTIME_CONTRACT.is_file():
        raise FileNotFoundError(RUNTIME_CONTRACT)
    inventory = _materialize_inventory()
    models = {str(row["id"]): row for row in inventory["models"]}
    profiles, profile_manifest = _materialize_profiles(inventory)
    canary = _canary_jobs(models, profiles)
    formal = _formal_jobs(models, profiles)
    checks = _audit_formal(formal)
    if not all(checks.values()):
        raise ValueError(f"formal queue audit failed: {checks}")
    write_jsonl(CANARY_QUEUE, canary)
    write_jsonl(FORMAL_QUEUE, formal)

    design_core = {
        "schema": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": GENERATED_AT_UTC,
        "gpu_training_started": False,
        "approval_promoted": False,
        "scope": {
            "included": [
                "dense full-attention control",
                "dense Qwen3.5/Qwen3.6 hybrid attention",
                "text-only samples with frozen visual tower/projector where present",
                "LoRA rank 32 target all",
            ],
            "excluded": [
                "MoE",
                "real image/video media",
                "packing",
                "full-parameter fine-tuning",
                "acceptance/publication evaluation",
            ],
        },
        "models": [
            {
                "model_id": row["id"],
                "release_family": row["release_family"],
                "actual_parameters": row["actual_parameters"],
                "architecture_route": row["architecture_signature"][
                    "architecture_route"
                ],
                "full_attention_layers": row["architecture_signature"][
                    "num_full_attention_layers"
                ],
                "linear_attention_layers": row["architecture_signature"][
                    "num_linear_attention_layers"
                ],
                "architecture_signature_sha256": row["architecture_signature"][
                    "signature_sha256"
                ],
            }
            for row in inventory["models"]
        ],
        "factors": {
            "exact_total_tokens": list(LENGTHS),
            "gradient_checkpointing": [True, False],
            "micro_batch_size": [1, 2],
            "parallel_routes": PARALLEL_ROUTES,
            "source_datasets": {
                source_id: {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for source_id, path in SOURCE_DATASETS.items()
            },
            "packing": [False],
        },
        "required_gpu_pool": {
            "gpu_ids": list(AUTHORIZED_GPU_IDS),
            "max_gpu_count_per_job": 4,
            "allow_gpu_ids_outside_pool": False,
            "source": "user explicitly authorized GPU 0-6",
        },
        "design_arms": {
            "核心正交": 120,
            "微批扩展": 40,
            "数据源复验": 40,
            "重复方差": 8,
        },
        "canary": {
            "queue_path": str(CANARY_QUEUE.resolve()),
            "queue_sha256": sha256_file(CANARY_QUEUE),
            "counts": _counts(canary),
            "gate": "all four jobs must succeed before formal approval is created",
        },
        "formal": {
            "queue_path": str(FORMAL_QUEUE.resolve()),
            "queue_sha256": sha256_file(FORMAL_QUEUE),
            "counts": _counts(formal),
            "checks": checks,
            "oom_policy": "right-censored lower bound; never copy capacity as exact peak",
        },
        "bindings": {
            "model_inventory_path": str(MODEL_INVENTORY.resolve()),
            "model_inventory_sha256": sha256_file(MODEL_INVENTORY),
            "profile_manifest_path": str(PROFILE_MANIFEST.resolve()),
            "profile_manifest_sha256": sha256_file(PROFILE_MANIFEST),
            "dataset_registry_path": str(DATASET_REGISTRY.resolve()),
            "dataset_registry_sha256": sha256_file(DATASET_REGISTRY),
            "feature_basis_path": str(FEATURE_SOURCE),
            "feature_basis_sha256": sha256_file(FEATURE_SOURCE),
            "runtime_contract_path": str(RUNTIME_CONTRACT.resolve()),
            "runtime_contract_sha256": sha256_file(RUNTIME_CONTRACT),
        },
        "execution": {
            "prepare_command": (
                "PYTHONPATH=offline_experiments/scripts "
                "/fine-tuning-launcher/.venv/bin/python "
                "offline_experiments/scripts/prepare_h800_hybrid_attention_dense_stage1_v1.py"
            ),
            "canary_preview_command": (
                "python offline_experiments/scripts/scheduler.py --input "
                f"{CANARY_QUEUE}"
            ),
            "formal_preview_command": (
                "python offline_experiments/scripts/scheduler.py --input "
                f"{FORMAL_QUEUE}"
            ),
            "execute_requires_separately_promoted_approval": True,
            "approval_must_bind_gpu_ids": list(AUTHORIZED_GPU_IDS),
            "current_shared_experiment_config_is_not_modified_by_prepare": True,
        },
        "profile_manifest_checks": profile_manifest["checks"],
    }
    design = {**design_core, "report_sha256": sha256_json(design_core)}
    write_json(DESIGN, design)
    return design


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    design = prepare()
    print(
        json.dumps(
            {
                "gpu_training_started": design["gpu_training_started"],
                "canary": design["canary"]["counts"],
                "formal": design["formal"]["counts"],
                "checks": design["formal"]["checks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
