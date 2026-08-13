#!/usr/bin/env python3
"""Prepare 15 short H800 probes targeted at the v2 validation failures."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    MATRIX_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)

CAMPAIGN_ID = "h800_unified_bounded_targeted_20260810_v1"
PHASE_ID = "h800_unified_bounded_targeted_v1"
JOB_SCHEMA = "sft_h800_unified_bounded_targeted_job/v1"
DESIGN_SCHEMA = "sft_h800_unified_bounded_targeted_design/v1"
TARGET_GBS = 32
WARMUP_STEPS = 1
MEASURE_STEPS = 2
EXPECTED_JOBS = 15
GPU_IDS = tuple(range(8))

DEFAULT_QUEUE = MATRIX_DIR / "h800_unified_bounded_targeted_jobs_v1.jsonl"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_unified_bounded_targeted_design_v1.json"
STAGING_DIR = ROOT / "unified_bounded_targeted_staging"
DEFAULT_EXPERIMENT = STAGING_DIR / "experiment.h800_unified_bounded_targeted_v1.json"
MODEL_INVENTORY = ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_v1.json"
TEMPLATE_QUEUE = MATRIX_DIR / "h800_unified_resource_evidence_jobs_v1.jsonl"
DATASET_INFO = ROOT / "data" / "dataset_info.json"
PROFILE_DIR = ARTIFACT_DIR / "dataset_profiles"

DATASETS = {
    "multiturn_4096": "multiturn",
    "short_512": "short",
    "longtail_8192": "longtail",
}

# Five probes reproduce and bracket the one historical OOM admitted by v2.
WORKLOADS: tuple[dict[str, Any], ...] = (
    *(
        {
            "probe_family": "q14_lora_z3_profile_boundary",
            "model_id": "qwen3_14b",
            "train_type": "lora",
            "dataset_id": dataset_id,
            "gpu_count": 2,
            "mbs": mbs,
            "zero_stage": 3,
            "gc": False,
        }
        for dataset_id, mbs in (
            ("multiturn_4096", 1),
            ("multiturn_4096", 2),
            ("multiturn_4096", 4),
            ("short_512", 2),
            ("longtail_8192", 2),
        )
    ),
    # Six matched small-model GC contrasts target the +55% to +155% old-domain bias.
    *(
        {
            "probe_family": "small_model_gc_contrast",
            "model_id": model_id,
            "train_type": train_type,
            "dataset_id": "multiturn_4096",
            "gpu_count": 1,
            "mbs": 2,
            "zero_stage": 0,
            "gc": gc,
        }
        for model_id, train_type, gc in (
            ("qwen3_1p7b", "lora", False),
            ("qwen3_1p7b", "lora", True),
            ("qwen3_4b", "lora", False),
            ("qwen3_4b", "lora", True),
            ("qwen3_1p7b", "full", False),
            ("qwen3_1p7b", "full", True),
        )
    ),
    # Four matched ZeRO/GC cells determine whether the GC interaction transfers at 8B.
    *(
        {
            "probe_family": "q8_lora_zero_gc_contrast",
            "model_id": "qwen3_8b",
            "train_type": "lora",
            "dataset_id": "multiturn_4096",
            "gpu_count": 2,
            "mbs": 2,
            "zero_stage": zero_stage,
            "gc": gc,
        }
        for zero_stage, gc in ((2, False), (2, True), (3, False), (3, True))
    ),
)


def _profile_max(path: Path) -> int:
    maximum = 0
    for row in read_jsonl(path):
        maximum = max(maximum, int(row["total_tokens"]))
    if maximum <= 0:
        raise ValueError(f"profile is empty: {path}")
    return maximum


def _dataset_binding(dataset_id: str) -> dict[str, Any]:
    registry = read_json(DATASET_INFO)
    entry = registry[dataset_id]
    data_path = ROOT / "data" / str(entry["file_name"])
    profile_path = PROFILE_DIR / f"{dataset_id}.qwen3_nothink.jsonl"
    raw_max = _profile_max(profile_path)
    cutoff = 4096
    return {
        "dataset_id": dataset_id,
        "dataset_category": DATASETS[dataset_id],
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "dataset_profile_path": str(profile_path.resolve()),
        "dataset_profile_sha256": sha256_file(profile_path),
        "raw_profile_max": raw_max,
        "aligned_effective_sequence": int(math.ceil(min(cutoff, raw_max) / 8.0) * 8),
    }


def _models() -> dict[str, dict[str, Any]]:
    inventory = read_json(MODEL_INVENTORY)
    return {str(row["id"]): dict(row) for row in inventory["models"]}


def _template() -> dict[str, Any]:
    rows = read_jsonl(TEMPLATE_QUEUE)
    return dict(
        next(
            row
            for row in rows
            if row["model_id"] == "qwen3_8b"
            and row["train_type"] == "lora"
            and int(row["gpu_count"]) == 2
            and not bool(row["packing"])
        )
    )


def _job(
    workload: dict[str, Any],
    *,
    models: dict[str, dict[str, Any]],
    template: dict[str, Any],
) -> dict[str, Any]:
    model = models[str(workload["model_id"])]
    dataset = _dataset_binding(str(workload["dataset_id"]))
    gpu_count = int(workload["gpu_count"])
    mbs = int(workload["mbs"])
    if TARGET_GBS % (gpu_count * mbs):
        raise ValueError("target GBS is not divisible by gpu_count * mbs")
    zero_stage = int(workload["zero_stage"])
    zero = "none" if zero_stage == 0 else f"zero{zero_stage}"
    identity = {**workload, "campaign_id": CAMPAIGN_ID}
    job = dict(template)
    job.update(
        {
            "schema": JOB_SCHEMA,
            "job_id": stable_id("h800ubt", identity),
            "campaign_id": CAMPAIGN_ID,
            "phase_id": PHASE_ID,
            "experiment_group": "UNIFIED_BOUNDED_TARGETED",
            "evidence_role": "targeted_mechanism_calibration",
            "purpose": str(workload["probe_family"]),
            "probe_family": str(workload["probe_family"]),
            **dataset,
            "cutoff_len": 4096,
            "model_id": str(workload["model_id"]),
            "model_family": str(model["family"]),
            "model_path": str(model["path"]),
            "tokenizer_path": str(model["path"]),
            "model_parameters": int(model["actual_parameters"]),
            "template": "qwen3_nothink",
            "train_type": str(workload["train_type"]),
            "gpu_count": gpu_count,
            "mbs": mbs,
            "zero_stage": zero_stage,
            "zero": zero,
            "gc": bool(workload["gc"]),
            "gradient_checkpointing": bool(workload["gc"]),
            "gradient_accumulation_steps": TARGET_GBS // (gpu_count * mbs),
            "target_gbs": TARGET_GBS,
            "max_samples": 256,
            "fidelity": "memory_probe_1plus2",
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "repeat": 0,
            "packing": False,
            "offload": False,
            "parallel_class": "gpu_partitionable",
            "requires_external_node_idle": False,
            "publication_allowed": False,
            "oom_role": "right_censored_lower_bound",
            "oom_is_expected_evidence": True,
            "mechanism_id": (
                f"{workload['train_type']}_zero{zero_stage}_"
                f"gc{int(bool(workload['gc']))}_{gpu_count}gpu_pack0"
            ),
            "scenario_id": stable_id("ubtscn", identity),
            "split_unit_id": str(workload["dataset_id"]),
            "calibration_partition": {
                "role": "fit",
                "policy": "unified_bounded_targeted_mechanism_v1",
                "split_unit_id": str(workload["dataset_id"]),
            },
        }
    )
    for key in (
        "packing_contract",
        "packing_dataprofile_path",
        "packing_dataprofile_sha256",
        "samples_per_pack",
    ):
        job.pop(key, None)
    return job


def _experiment() -> dict[str, Any]:
    source = read_json(
        ROOT
        / "unified_resource_staging"
        / "experiment.h800_unified_resource_evidence_v1.json"
    )
    experiment = dict(source)
    experiment["training_scope"] = {
        "phase_id": PHASE_ID,
        "model_ids": sorted({str(row["model_id"]) for row in WORKLOADS}),
        "gpu_ids": list(GPU_IDS),
        "exclusive_node_gpu_ids": list(GPU_IDS),
        "max_gpu_count": 2,
        "stage": "sft",
        "precision": "bf16",
        "gpu_type": "NVIDIA H800 140GB HBM3",
        "gpu_counts": [1, 2],
        "global_batch_sizes": [TARGET_GBS],
        "gradient_checkpointing": [False, True],
        "zero_by_gpu_count": {"1": ["none"], "2": ["zero2", "zero3"]},
        "objective": "short targeted memory probes after frozen-v2 validation failure",
    }
    experiment["measurement"] = {
        **dict(experiment.get("measurement") or {}),
        "memory_probe_max_steps": WARMUP_STEPS + MEASURE_STEPS,
        "throughput_warmup_steps": WARMUP_STEPS,
        "throughput_measure_steps": MEASURE_STEPS,
        "performance_parallelism": "disjoint_gpu_masks",
        "scheduler_order_policy": "parallel_queue",
        "rerun_on_unhealthy_result": False,
    }
    experiment["datasets"] = [
        {"id": dataset_id, "category": category, "target_cutoffs": [4096]}
        for dataset_id, category in DATASETS.items()
    ]
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    existing = [
        str(path)
        for path in (args.queue, args.design, args.experiment)
        if path.exists()
    ]
    if existing:
        raise SystemExit(f"refusing to overwrite existing outputs: {existing}")

    models = _models()
    template = _template()
    jobs = [_job(dict(row), models=models, template=template) for row in WORKLOADS]
    if (
        len(jobs) != EXPECTED_JOBS
        or len({row["job_id"] for row in jobs}) != EXPECTED_JOBS
    ):
        raise ValueError("targeted queue must contain exactly 15 unique jobs")
    for row in jobs:
        model = models[str(row["model_id"])]
        if (
            Path(str(row["model_path"])).resolve() != Path(str(model["path"])).resolve()
            or Path(str(row["tokenizer_path"])).resolve()
            != Path(str(model["path"])).resolve()
            or int(row["model_parameters"]) != int(model["actual_parameters"])
        ):
            raise ValueError(f"model identity mismatch in {row['job_id']}")

    args.queue.parent.mkdir(parents=True, exist_ok=True)
    args.design.parent.mkdir(parents=True, exist_ok=True)
    args.experiment.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.queue, jobs)
    write_json(args.experiment, _experiment())
    design: dict[str, Any] = {
        "schema": DESIGN_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "status": "prepared_waiting_execution",
        "gpu_training_started": False,
        "execution_authorized": True,
        "publication_allowed": False,
        "objective": (
            "Reproduce the admitted historical q14 LoRA/Z3/no-GC OOM, measure "
            "small-model GC centre bias, and isolate Q8 ZeRO/GC interactions."
        ),
        "not_the_old_resume_queue": True,
        "old_missing_34_jobs_included": False,
        "probe_family_counts": dict(Counter(row["probe_family"] for row in jobs)),
        "ordered_job_ids": [str(row["job_id"]) for row in jobs],
        "ordered_job_payload_sha256": sha256_json(jobs),
        "bindings": {
            "queue": {
                "path": str(args.queue.resolve()),
                "sha256": sha256_file(args.queue),
            },
            "experiment": {
                "path": str(args.experiment.resolve()),
                "sha256": sha256_file(args.experiment),
            },
            "model_inventory": {
                "path": str(MODEL_INVENTORY.resolve()),
                "sha256": sha256_file(MODEL_INVENTORY),
            },
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(args.design, design)
    print(
        json.dumps(
            {
                "queue": str(args.queue),
                "design": str(args.design),
                "experiment": str(args.experiment),
                "jobs": len(jobs),
                "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in jobs),
                "probe_family_counts": design["probe_family_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
