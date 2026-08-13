#!/usr/bin/env python3
"""Materialize the exact eight-job 4/5/6/7-GPU Packing executor canary."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from common import (
    ARTIFACT_DIR,
    DATA_DIR,
    MATRIX_DIR,
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)


CAMPAIGN_ID = "h800_packing_final_4to7gpu_20260810_v1"
PHASE_ID = "h800_packing_final_executor_canary_v1"
JOB_SCHEMA = "sft_h800_packing_final_executor_canary_job/v1"
DESIGN_SCHEMA = "sft_h800_packing_final_executor_canary_design/v1"
GPU_IDS = tuple(range(8))
GPU_COUNTS = (4, 5, 6, 7)
MODEL_ID = "qwen3_1p7b"
MODEL_PATH = Path("/wanqing-models/Qwen3-1.7B")
TEMPLATE = "qwen3_nothink"
DATASET_ID = "packing_final_pf02_canary_v1"
WORKLOAD_ID = "PF02"
CUTOFF_LEN = 1024
PACKED_GA = 2
UNPACKED_GA = 14
WARMUP_STEPS = 0
MEASURE_STEPS = 2

SOURCE = (
    DATA_DIR
    / "packing_final_business_screen_20260810"
    / "dataset-li2sye-1780465080"
    / "1"
    / "publish"
    / "dataset-li2sye-1780465080-V1.jsonl"
)
RUNTIME_DATA = (
    DATA_DIR
    / "packing_final_4to7gpu_v1"
    / "pf02_dataset-li2sye-1780465080.jsonl"
)
PROFILE = (
    ARTIFACT_DIR
    / "h800_packing_final_business_profiles_v1"
    / "pf02_dataset-li2sye-1780465080.json"
)
MODEL_INVENTORY = ARTIFACT_DIR / "model_inventory.json"
DATASET_INFO = DATA_DIR / "dataset_info.json"
PLAN = ARTIFACT_DIR / "h800_packing_final_4to7gpu_experiment_design_v1.json"
QUEUE = MATRIX_DIR / "h800_packing_final_executor_canary_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_packing_final_executor_canary_design_v1.json"
STATIC = ARTIFACT_DIR / "h800_packing_final_executor_canary_static_v1.json"
STAGING_DIR = ROOT / "packing_final_4to7gpu_staging"
EXPERIMENT = STAGING_DIR / "experiment.h800_packing_final_executor_canary_v1.json"
JOBS_DIR = ARTIFACT_DIR / "h800_packing_final_executor_canary_jobs_v1"
EXPECTED_JOBS = 8


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> None:
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
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _normalized_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with SOURCE.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            value = json.loads(line)
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            if not isinstance(value, dict) or not all(
                isinstance(value.get(field), str)
                for field in ("system", "prompt", "response")
            ):
                raise ValueError(f"{SOURCE}:{line_number} is not pure text SFT")
            rows.append(
                {field: str(value[field]) for field in ("system", "prompt", "response")}
            )
    if len(rows) != 11_690:
        raise ValueError(f"PF02 row count drifted: {len(rows)}")
    return rows


def _profile_contract() -> dict[str, Any]:
    profile = read_json(PROFILE)
    if profile.get("workload_id") != WORKLOAD_ID or int(profile.get("records", 0)) != 11_690:
        raise ValueError("PF02 profile identity drifted")
    curve = next(
        row for row in profile["packing_curve"] if int(row["cutoff_len"]) == CUTOFF_LEN
    )
    samples = curve["samples_per_pack"]
    return {
        "workload_id": WORKLOAD_ID,
        "dataset_id": str(profile["dataset_id"]),
        "cutoff_len": CUTOFF_LEN,
        "preprocessing_num_workers": 8,
        "pack_utilization": float(curve["pack_utilization"]),
        "model_facing_pack_fill_ratio": float(curve["model_facing_pack_fill_ratio"]),
        "samples_per_pack": {
            "minimum": int(samples["minimum"]),
            "mean": float(samples["mean"]),
            "p99": float(samples["p99"]),
            "maximum": int(samples["maximum"]),
        },
        "sample_truncation_rate": float(curve["sample_truncation_rate"]),
        "tokens_retained_ratio": float(curve["tokens_retained_ratio"]),
        "profile": _binding(PROFILE),
    }


def _experiment() -> dict[str, Any]:
    current = read_json(ROOT / "config" / "experiment.json")
    return {
        "schema_version": 1,
        "training_scope": {
            "phase_id": PHASE_ID,
            "model_ids": [MODEL_ID],
            "gpu_ids": list(GPU_IDS),
            "exclusive_node_gpu_ids": list(GPU_IDS),
            "max_gpu_count": 7,
            "stage": "sft",
            "precision": "bf16",
            "gpu_type": "NVIDIA H800 140GB HBM3",
            "gpu_counts": list(GPU_COUNTS),
            "global_batch_sizes": [count * UNPACKED_GA for count in GPU_COUNTS],
            "gradient_checkpointing": [True],
            "zero_by_gpu_count": {str(count): ["zero3"] for count in GPU_COUNTS},
            "objective": "prove exact packed and unpacked 4/5/6/7-rank execution",
        },
        "fixed_runtime": current["fixed_runtime"],
        "measurement": {
            **current["measurement"],
            "throughput_warmup_steps": WARMUP_STEPS,
            "throughput_measure_steps": MEASURE_STEPS,
            "performance_parallelism": "exclusive_pool",
            "scheduler_order_policy": "strict_queue_order",
            "formal_throughput_requires_exclusive_node": True,
        },
        "datasets": [
            {"id": DATASET_ID, "category": "executor_canary", "target_cutoffs": [CUTOFF_LEN]}
        ],
        "packing_static_gate": {
            "minimum_pack_utilization": 0.90,
            "minimum_sequence_reduction": 0.0,
            "minimum_mean_samples_per_pack": 1.0,
            "maximum_expected_gbs_error": 0.10,
            "note": "Phase 0 only; no model fit or publication.",
        },
    }


def prepare() -> dict[str, Any]:
    rows = _normalized_rows()
    _atomic_jsonl(RUNTIME_DATA, rows)
    contract = _profile_contract()
    write_json(
        STATIC,
        {
            "schema": "sft_h800_packing_final_executor_canary_static/v1",
            "generated_before_gpu": True,
            "raw_dataset_read_at_recommendation_time": False,
            "source": _binding(SOURCE),
            "runtime_data": _binding(RUNTIME_DATA),
            "contract": contract,
        },
    )
    write_json(EXPERIMENT, _experiment())

    model = next(
        row for row in read_json(MODEL_INVENTORY)["models"] if row["id"] == MODEL_ID
    )
    jobs: list[dict[str, Any]] = []
    order = tuple(
        (gpu_count, packing)
        for gpu_count in GPU_COUNTS
        for packing in ((False, True) if gpu_count % 2 == 0 else (True, False))
    )
    mean_samples = float(contract["samples_per_pack"]["mean"])
    for sequence_index, (gpu_count, packing) in enumerate(order):
        ga = PACKED_GA if packing else UNPACKED_GA
        expected_gbs = (
            mean_samples * gpu_count * ga if packing else float(gpu_count * ga)
        )
        target_gbs = gpu_count * UNPACKED_GA
        job: dict[str, Any] = {
            "schema": JOB_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "phase_id": PHASE_ID,
            "candidate_role": "executor_only_canary",
            "scenario_id": f"{WORKLOAD_ID}-qwen3_1p7b-lora-dp{gpu_count}",
            "arm_id": "packed" if packing else "unpacked",
            "repeat": 0,
            "model_id": MODEL_ID,
            "model_family": "qwen3",
            "model_path": str(MODEL_PATH),
            "tokenizer_path": str(MODEL_PATH),
            "template": TEMPLATE,
            "model_parameters": int(model["actual_parameters"]),
            "train_type": "lora",
            "dataset_id": DATASET_ID,
            "source_dataset_id": contract["dataset_id"],
            "workload_id": WORKLOAD_ID,
            "data_path": str(RUNTIME_DATA.resolve()),
            "data_sha256": sha256_file(RUNTIME_DATA),
            "dataset_profile_path": str(PROFILE.resolve()),
            "dataset_profile_sha256": sha256_file(PROFILE),
            "cutoff_len": CUTOFF_LEN,
            "target_gbs": target_gbs,
            "gpu_count": gpu_count,
            "zero": "zero3",
            "zero_stage": 3,
            "gc": True,
            "gradient_checkpointing": True,
            "mbs": 1,
            "gradient_accumulation_steps": ga,
            "packing": packing,
            "expected_sample_gbs": expected_gbs,
            "expected_sample_gbs_relative_error": abs(expected_gbs - target_gbs) / target_gbs,
            "static_packing_contract": contract,
            "offload": False,
            "gpu_type": "NVIDIA H800 140GB HBM3",
            "required_runtime_gpu_name": "NVIDIA H800",
            "hardware_id": "local_h800_140g",
            "kind": "throughput",
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "max_samples": len(rows),
            "fidelity": "executor_canary_0plus2",
            "parallel_class": "exclusive_pool",
            "requires_external_node_idle": True,
            "strict_queue_order": True,
            "execution_sequence_index": sequence_index,
            "publication_allowed": False,
        }
        job["job_id"] = stable_id("h800packfinalcanary", job)
        jobs.append(job)
    if len(jobs) != EXPECTED_JOBS or len({row["job_id"] for row in jobs}) != EXPECTED_JOBS:
        raise ValueError("executor canary must contain eight unique jobs")
    write_jsonl(QUEUE, jobs)
    for job in jobs:
        write_json(JOBS_DIR / f"{job['job_id']}.json", job)

    bindings = {
        "plan": _binding(PLAN),
        "queue": _binding(QUEUE),
        "experiment": _binding(EXPERIMENT),
        "static": _binding(STATIC),
        "dataset_info": _binding(DATASET_INFO),
        "model_inventory": _binding(MODEL_INVENTORY),
    }
    design: dict[str, Any] = {
        "schema": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_gpu_waiting_for_exact_approval",
        "gpu_training_started": False,
        "automatic_successor_expansion_allowed": False,
        "publication_allowed": False,
        "authorized_gpu_ids": list(GPU_IDS),
        "gpu_counts": list(GPU_COUNTS),
        "ordered_job_ids": [str(row["job_id"]) for row in jobs],
        "ordered_job_payload_sha256": sha256_json(jobs),
        "jobs": EXPECTED_JOBS,
        "measurement_contract": {
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "exact_rank_count_required": True,
            "exact_physical_gpu_uuid_binding_required": True,
            "authoritative_token_and_batch_shape_ledgers_required": True,
            "packed_physical_mbs": 1,
            "packed_runtime_semantics_required_on_every_rank": True,
            "static_raw_sample_gbs_relative_error_max": 0.20,
            "static_effective_token_relative_error_max": 0.15,
        },
        "bindings": bindings,
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    return {
        "queue": _binding(QUEUE),
        "design": _binding(DESIGN),
        "experiment": _binding(EXPERIMENT),
        "static": _binding(STATIC),
        "jobs": len(jobs),
        "gpu_counts": list(GPU_COUNTS),
    }


def main() -> None:
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
