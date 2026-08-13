#!/usr/bin/env python3
"""Materialize the 36-arm 4/5/6/7-GPU virtual-MBS transfer experiment."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
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
PHASE_ID = "h800_packing_final_virtual_mbs_v1"
JOB_SCHEMA = "sft_h800_packing_final_virtual_mbs_job/v1"
DESIGN_SCHEMA = "sft_h800_packing_final_virtual_mbs_design/v1"
GPU_IDS = tuple(range(8))
GPU_COUNTS = (4, 5, 6, 7)
MODEL_ID = "qwen3_8b"
MODEL_PATH = Path("/wanqing-models/Qwen3-8B")
TEMPLATE = "qwen3_nothink"
WARMUP_STEPS = 3
MEASURE_STEPS = 20
EXPECTED_JOBS = 36
CAPACITY_REPETITION = {
    "PF02": ("packing_final_pf02_capacity_v1", 3),
    "PF05": ("packing_final_pf05_capacity_v1", 10),
    "PF06": ("packing_final_pf06_capacity_v1", 16),
}

PROFILE_DIR = ARTIFACT_DIR / "h800_packing_final_business_profiles_v1"
RUNTIME_DATA_DIR = DATA_DIR / "packing_final_4to7gpu_v1"
MODEL_INVENTORY = ARTIFACT_DIR / "model_inventory.json"
DATASET_INFO = DATA_DIR / "dataset_info.json"
PLAN = ARTIFACT_DIR / "h800_packing_final_4to7gpu_experiment_design_v1.json"
QUEUE = MATRIX_DIR / "h800_packing_final_virtual_mbs_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_packing_final_virtual_mbs_design_v1.json"
STATIC = ARTIFACT_DIR / "h800_packing_final_virtual_mbs_static_v1.json"
STAGING_DIR = ROOT / "packing_final_4to7gpu_staging"
EXPERIMENT = STAGING_DIR / "experiment.h800_packing_final_virtual_mbs_v1.json"
JOBS_DIR = ARTIFACT_DIR / "h800_packing_final_virtual_mbs_jobs_v1"


@dataclass(frozen=True)
class Workload:
    workload_id: str
    source_dataset_id: str
    runtime_dataset_id: str
    profile_name: str
    cutoff_len: int
    sample_gbs_per_gpu: int


WORKLOADS = (
    Workload(
        "PF02",
        "dataset-li2sye-1780465080",
        "packing_final_pf02_canary_v1",
        "pf02_dataset-li2sye-1780465080.json",
        1024,
        112,
    ),
    Workload(
        "PF05",
        "dataset-kxvcnz-1780930601",
        "packing_final_pf05_fit_v1",
        "pf05_dataset-kxvcnz-1780930601.json",
        8192,
        40,
    ),
    Workload(
        "PF06",
        "dataset-mooxf7-1778662463",
        "packing_final_pf06_fit_v1",
        "pf06_dataset-mooxf7-1778662463.json",
        8192,
        32,
    ),
)


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


def _normalize_source(source: Path, destination: Path, expected_rows: int) -> None:
    rows: list[dict[str, str]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            if not isinstance(value, dict) or not all(
                isinstance(value.get(field), str)
                for field in ("system", "prompt", "response")
            ):
                raise ValueError(f"{source}:{line_number} is not pure text SFT")
            rows.append(
                {field: str(value[field]) for field in ("system", "prompt", "response")}
            )
    if len(rows) != expected_rows:
        raise ValueError(f"{source} row count drifted: {len(rows)} != {expected_rows}")
    _atomic_jsonl(destination, rows)


def _materialize_capacity_data(
    contracts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Repeat normalized records without changing their empirical distribution."""

    dataset_info = read_json(DATASET_INFO)
    capacity: dict[str, dict[str, Any]] = {}
    for contract in contracts:
        workload_id = str(contract["workload_id"])
        dataset_id, repetition_factor = CAPACITY_REPETITION[workload_id]
        source = Path(str(contract["runtime_data"]["path"]))
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        expected_rows = int(contract["records"])
        if len(rows) != expected_rows or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"{workload_id} normalized runtime data drifted")
        destination = source.with_name(
            f"{source.stem}.capacity{repetition_factor}x.jsonl"
        )
        _atomic_jsonl(
            destination,
            (row for _ in range(repetition_factor) for row in rows),
        )
        repeated_rows = expected_rows * repetition_factor
        dataset_info[dataset_id] = {
            "file_name": str(destination.resolve().relative_to(DATA_DIR.resolve())),
            "columns": {
                "prompt": "prompt",
                "response": "response",
                "system": "system",
            },
        }
        capacity[workload_id] = {
            "dataset_id": dataset_id,
            "data_path": str(destination.resolve()),
            "data_sha256": sha256_file(destination),
            "records": repeated_rows,
            "repetition_factor": repetition_factor,
            "distribution_preserving_repetition": True,
        }
    write_json(DATASET_INFO, dataset_info)
    return capacity


def _contract(workload: Workload) -> dict[str, Any]:
    profile_path = PROFILE_DIR / workload.profile_name
    profile = read_json(profile_path)
    if (
        profile.get("workload_id") != workload.workload_id
        or profile.get("dataset_id") != workload.source_dataset_id
        or profile.get("split_role") != "fit"
    ):
        raise ValueError(f"{workload.workload_id} profile identity drifted")
    curve = next(
        row
        for row in profile["packing_curve"]
        if int(row["cutoff_len"]) == workload.cutoff_len
    )
    samples = curve["samples_per_pack"]
    source = Path(str(profile["source_binding"]["local_path"]))
    runtime_data = RUNTIME_DATA_DIR / f"{workload.profile_name[:-5]}.jsonl"
    _normalize_source(source, runtime_data, int(profile["records"]))
    mean_samples = float(samples["mean"])
    floor_mbs = math.floor(mean_samples)
    ceil_mbs = math.ceil(mean_samples)
    if floor_mbs == ceil_mbs:
        raise ValueError(f"{workload.workload_id} does not bracket virtual MBS")
    if (
        workload.sample_gbs_per_gpu % floor_mbs
        or workload.sample_gbs_per_gpu % ceil_mbs
    ):
        raise ValueError(f"{workload.workload_id} unpacked GBS is not divisible")
    packed_ga = round(workload.sample_gbs_per_gpu / mean_samples)
    expected_per_gpu = mean_samples * packed_ga
    return {
        "workload_id": workload.workload_id,
        "source_dataset_id": workload.source_dataset_id,
        "runtime_dataset_id": workload.runtime_dataset_id,
        "cutoff_len": workload.cutoff_len,
        "sample_gbs_per_gpu": workload.sample_gbs_per_gpu,
        "records": int(profile["records"]),
        "source": _binding(source),
        "runtime_data": _binding(runtime_data),
        "profile": _binding(profile_path),
        "pack_utilization": float(curve["pack_utilization"]),
        "model_facing_pack_fill_ratio": float(curve["model_facing_pack_fill_ratio"]),
        "samples_per_pack": {
            "minimum": int(samples["minimum"]),
            "mean": mean_samples,
            "p99": float(samples["p99"]),
            "maximum": int(samples["maximum"]),
        },
        "floor_virtual_mbs": floor_mbs,
        "ceil_virtual_mbs": ceil_mbs,
        "packed_gradient_accumulation_steps": packed_ga,
        "expected_packed_sample_gbs_per_gpu": expected_per_gpu,
        "expected_packed_gbs_relative_error": abs(
            expected_per_gpu - workload.sample_gbs_per_gpu
        )
        / workload.sample_gbs_per_gpu,
        "sample_truncation_rate": float(curve["sample_truncation_rate"]),
        "tokens_retained_ratio": float(curve["tokens_retained_ratio"]),
    }


def _experiment(
    contracts: list[dict[str, Any]], capacity: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    current = read_json(ROOT / "config" / "experiment.json")
    global_batch_sizes = sorted(
        {
            int(contract["sample_gbs_per_gpu"]) * gpu_count
            for contract in contracts
            for gpu_count in GPU_COUNTS
        }
    )
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
            "global_batch_sizes": global_batch_sizes,
            "gradient_checkpointing": [True],
            "zero_by_gpu_count": {str(count): ["zero3"] for count in GPU_COUNTS},
            "objective": "measure direct virtual-MBS transfer for Packing throughput",
        },
        "fixed_runtime": current["fixed_runtime"],
        "measurement": {
            **current["measurement"],
            "throughput_warmup_steps": WARMUP_STEPS,
            "throughput_measure_steps": MEASURE_STEPS,
            "performance_parallelism": "disjoint_gpu_masks",
            "scheduler_order_policy": "strict_queue_order",
            "formal_throughput_requires_exclusive_node": False,
        },
        "datasets": [
            {
                "id": str(capacity[str(contract["workload_id"])]["dataset_id"]),
                "category": "virtual_mbs_fit",
                "target_cutoffs": [int(contract["cutoff_len"])],
                "execution_only_distribution_preserving_repetition": True,
            }
            for contract in contracts
        ],
        "packing_static_gate": {
            "minimum_pack_utilization": 0.0,
            "minimum_sequence_reduction": 0.0,
            "minimum_mean_samples_per_pack": 1.0,
            "maximum_expected_gbs_error": 0.05,
            "note": "GA is derived only from the frozen upload-time DataProfile.",
        },
    }


def prepare() -> dict[str, Any]:
    contracts = [_contract(workload) for workload in WORKLOADS]
    capacity = _materialize_capacity_data(contracts)
    static: dict[str, Any] = {
        "schema": "sft_h800_packing_final_virtual_mbs_static/v1",
        "generated_before_gpu": True,
        "split_role": "fit",
        "recommendation_time_raw_rows_required": False,
        "contracts": contracts,
    }
    static["report_sha256"] = sha256_json(static)
    write_json(STATIC, static)
    write_json(EXPERIMENT, _experiment(contracts, capacity))

    model = next(
        row for row in read_json(MODEL_INVENTORY)["models"] if row["id"] == MODEL_ID
    )
    jobs: list[dict[str, Any]] = []
    for workload_index, contract in enumerate(contracts):
        for gpu_count in GPU_COUNTS:
            available_pool_execution = contract["workload_id"] in {"PF05", "PF06"}
            use_original_execution_data = (
                contract["workload_id"] == "PF02" and gpu_count == 4
            )
            execution_data = (
                {
                    "dataset_id": contract["runtime_dataset_id"],
                    "data_path": contract["runtime_data"]["path"],
                    "data_sha256": contract["runtime_data"]["sha256"],
                    "records": contract["records"],
                    "repetition_factor": 1,
                    "distribution_preserving_repetition": False,
                }
                if use_original_execution_data
                else capacity[str(contract["workload_id"])]
            )
            arm_order = (
                ("packed", "floor", "ceil")
                if (workload_index + gpu_count) % 2
                else ("floor", "packed", "ceil")
            )
            for arm_id in arm_order:
                packing = arm_id == "packed"
                if arm_id == "packed":
                    mbs = 1
                    ga = int(contract["packed_gradient_accumulation_steps"])
                    expected_gbs = (
                        float(contract["samples_per_pack"]["mean"])
                        * gpu_count
                        * ga
                    )
                else:
                    mbs = int(contract[f"{arm_id}_virtual_mbs"])
                    ga = int(contract["sample_gbs_per_gpu"]) // mbs
                    expected_gbs = float(gpu_count * mbs * ga)
                target_gbs = int(contract["sample_gbs_per_gpu"]) * gpu_count
                job: dict[str, Any] = {
                    "schema": JOB_SCHEMA,
                    "campaign_id": CAMPAIGN_ID,
                    "phase_id": PHASE_ID,
                    "candidate_role": "virtual_mbs_transfer_fit",
                    "scenario_id": f"{contract['workload_id']}-qwen3_8b-lora-dp{gpu_count}",
                    "matched_group_id": f"{contract['workload_id']}-dp{gpu_count}",
                    "arm_id": arm_id,
                    "repeat": 0,
                    "model_id": MODEL_ID,
                    "model_family": "qwen3",
                    "model_path": str(MODEL_PATH),
                    "tokenizer_path": str(MODEL_PATH),
                    "template": TEMPLATE,
                    "model_parameters": int(model["actual_parameters"]),
                    "train_type": "lora",
                    "dataset_id": execution_data["dataset_id"],
                    "source_dataset_id": contract["source_dataset_id"],
                    "workload_id": contract["workload_id"],
                    "data_path": execution_data["data_path"],
                    "data_sha256": execution_data["data_sha256"],
                    "dataset_profile_path": contract["profile"]["path"],
                    "dataset_profile_sha256": contract["profile"]["sha256"],
                    "cutoff_len": int(contract["cutoff_len"]),
                    "target_gbs": target_gbs,
                    "gpu_count": gpu_count,
                    "zero": "zero3",
                    "zero_stage": 3,
                    "gc": True,
                    "gradient_checkpointing": True,
                    "mbs": mbs,
                    "gradient_accumulation_steps": ga,
                    "packing": packing,
                    "virtual_mbs": float(contract["samples_per_pack"]["mean"]),
                    "expected_sample_gbs": expected_gbs,
                    "expected_sample_gbs_relative_error": abs(expected_gbs - target_gbs)
                    / target_gbs,
                    "static_packing_contract": contract,
                    "offload": False,
                    "gpu_type": "NVIDIA H800 140GB HBM3",
                    "required_runtime_gpu_name": "NVIDIA H800",
                    "hardware_id": "local_h800_140g",
                    "kind": "throughput",
                    "warmup_steps": WARMUP_STEPS,
                    "measure_steps": MEASURE_STEPS,
                    "max_samples": int(execution_data["records"]),
                    "fidelity": "formal_throughput_3plus20",
                    "parallel_class": (
                        "available_pool" if available_pool_execution else "exclusive_pool"
                    ),
                    "requires_external_node_idle": not available_pool_execution,
                    "strict_queue_order": True,
                    "execution_sequence_index": len(jobs),
                    "publication_allowed": False,
                }
                if int(execution_data["repetition_factor"]) > 1:
                    job.update(
                        {
                            "execution_data_repetition_factor": int(
                                execution_data["repetition_factor"]
                            ),
                            "execution_data_distribution_preserving_repetition": True,
                        }
                    )
                if available_pool_execution:
                    job.update(
                        {
                            "allow_available_pool_for_large_job": True,
                            "performance_isolation": "selected_gpu_mask_idle_only",
                            "external_busy_gpus_allowed": True,
                        }
                    )
                job["job_id"] = stable_id("h800packvirtualmbs", job)
                jobs.append(job)
    if len(jobs) != EXPECTED_JOBS or len({row["job_id"] for row in jobs}) != EXPECTED_JOBS:
        raise ValueError("virtual-MBS queue must contain 36 unique jobs")
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
        "automatic_repeat_expansion_allowed": False,
        "publication_allowed": False,
        "authorized_gpu_ids": list(GPU_IDS),
        "gpu_counts": list(GPU_COUNTS),
        "ordered_job_ids": [str(row["job_id"]) for row in jobs],
        "ordered_job_payload_sha256": sha256_json(jobs),
        "jobs": EXPECTED_JOBS,
        "matched_groups": 12,
        "arms": ["packed", "floor", "ceil"],
        "measurement_contract": {
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "exclusive_whole_node": False,
            "pf02_completed_under_exclusive_whole_node": True,
            "pf05_pf06_available_pool_authorized": True,
            "throughput_truth": "measured logical samples divided by measured seconds",
            "token_truth": "authoritative consumed token ledger",
            "conditional_repeat_block_cv_threshold": 0.03,
            "conditional_repeat_top2_gap_threshold": 0.03,
            "repeat_decision_after_all_base_arms": True,
        },
        "candidate_models": [
            "T0 direct virtual-MBS substitution",
            "T1 one global multiplicative Packing coefficient",
            "T2 predeclared regularized Packing residual",
        ],
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
        "matched_groups": 12,
    }


def main() -> None:
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
