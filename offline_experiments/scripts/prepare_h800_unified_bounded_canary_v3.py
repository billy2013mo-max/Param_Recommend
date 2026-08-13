#!/usr/bin/env python3
"""Freeze a small prospective H800 canary for the immutable v3 artifact."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fit_h800_unified_resource_partial_v1 as base
import h800_unified_bounded_memory_v3_data as v3_data
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
from h800_unified_bounded_memory_model import load_artifact, predict_records

CAMPAIGN_ID = "h800_unified_bounded_canary_20260810_v3"
PHASE_ID = "h800_unified_bounded_canary_v3"
JOB_SCHEMA = "sft_h800_unified_bounded_canary_job/v3"
DESIGN_SCHEMA = "sft_h800_unified_bounded_canary_design/v3"
PREDICTION_SCHEMA = "sft_h800_unified_bounded_canary_frozen_predictions/v3"
TARGET_GBS = 32
WARMUP_STEPS = 1
MEASURE_STEPS = 4
EXPECTED_JOBS = 10
GPU_IDS = tuple(range(8))

DEFAULT_QUEUE = MATRIX_DIR / "h800_unified_bounded_canary_jobs_v3.jsonl"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_unified_bounded_canary_design_v3.json"
DEFAULT_PREDICTIONS = (
    ARTIFACT_DIR / "h800_unified_bounded_canary_frozen_predictions_v3.json"
)
STAGING_DIR = ROOT / "unified_bounded_canary_staging"
DEFAULT_EXPERIMENT = STAGING_DIR / "experiment.h800_unified_bounded_canary_v3.json"
MODEL_INVENTORY = ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_v1.json"
TEMPLATE_QUEUE = MATRIX_DIR / "h800_unified_resource_evidence_jobs_v1.jsonl"
DATASET_INFO = ROOT / "data" / "dataset_info.json"
PROFILE_DIR = ARTIFACT_DIR / "dataset_profiles"
V2_ARTIFACT = ARTIFACT_DIR / "h800_unified_bounded_memory_candidate_v2.json"
V3_ARTIFACT = ARTIFACT_DIR / "h800_unified_bounded_memory_candidate_v3.json"

DATASET_CATEGORIES = {
    "short_512": "short",
    "multiturn_2048": "multiturn",
    "multiturn_4096": "multiturn",
    "longtail_8192": "longtail",
    "longcontext_16384": "longcontext",
}

# Eight v3-admitted cases deliberately sit close to the 95%-capacity boundary.
# Two just-over-boundary controls measure false rejection without being canary
# traffic.  All ten configurations were selected after the artifact checksum
# was frozen and are configuration-disjoint from the 531-row development set.
WORKLOADS: tuple[dict[str, Any], ...] = (
    {
        "probe_role": "admitted_canary",
        "model_id": "qwen3_1p7b",
        "train_type": "full",
        "dataset_id": "multiturn_4096",
        "cutoff_len": 4096,
        "gpu_count": 1,
        "mbs": 8,
        "zero_stage": 0,
        "gc": False,
    },
    {
        "probe_role": "admitted_canary",
        "model_id": "qwen3_1p7b",
        "train_type": "full",
        "dataset_id": "longtail_8192",
        "cutoff_len": 4096,
        "gpu_count": 1,
        "mbs": 8,
        "zero_stage": 0,
        "gc": False,
    },
    {
        "probe_role": "admitted_canary",
        "model_id": "qwen3_4b",
        "train_type": "lora",
        "dataset_id": "longcontext_16384",
        "cutoff_len": 8192,
        "gpu_count": 2,
        "mbs": 2,
        "zero_stage": 3,
        "gc": False,
    },
    {
        "probe_role": "admitted_canary",
        "model_id": "qwen3_8b",
        "train_type": "lora",
        "dataset_id": "multiturn_2048",
        "cutoff_len": 4096,
        "gpu_count": 1,
        "mbs": 8,
        "zero_stage": 0,
        "gc": False,
    },
    {
        "probe_role": "admitted_canary",
        "model_id": "qwen3_14b",
        "train_type": "full",
        "dataset_id": "short_512",
        "cutoff_len": 4096,
        "gpu_count": 2,
        "mbs": 1,
        "zero_stage": 3,
        "gc": True,
    },
    {
        "probe_role": "admitted_canary",
        "model_id": "qwen3_8b",
        "train_type": "full",
        "dataset_id": "short_512",
        "cutoff_len": 4096,
        "gpu_count": 2,
        "mbs": 16,
        "zero_stage": 3,
        "gc": False,
    },
    {
        "probe_role": "admitted_canary",
        "model_id": "qwen3_8b",
        "train_type": "lora",
        "dataset_id": "longtail_8192",
        "cutoff_len": 4096,
        "gpu_count": 2,
        "mbs": 4,
        "zero_stage": 2,
        "gc": False,
    },
    {
        "probe_role": "admitted_canary",
        "model_id": "qwen3_4b",
        "train_type": "full",
        "dataset_id": "multiturn_2048",
        "cutoff_len": 4096,
        "gpu_count": 2,
        "mbs": 8,
        "zero_stage": 3,
        "gc": False,
    },
    {
        "probe_role": "rejected_boundary_control",
        "model_id": "qwen3_14b",
        "train_type": "full",
        "dataset_id": "multiturn_2048",
        "cutoff_len": 4096,
        "gpu_count": 2,
        "mbs": 1,
        "zero_stage": 3,
        "gc": True,
    },
    {
        "probe_role": "rejected_boundary_control",
        "model_id": "qwen3_8b",
        "train_type": "full",
        "dataset_id": "longcontext_16384",
        "cutoff_len": 8192,
        "gpu_count": 2,
        "mbs": 1,
        "zero_stage": 3,
        "gc": False,
    },
)


def _profile_max(path: Path) -> int:
    maximum = max(int(row["total_tokens"]) for row in read_jsonl(path))
    if maximum <= 0:
        raise ValueError(f"profile is empty: {path}")
    return maximum


def _dataset_binding(dataset_id: str, cutoff_len: int) -> dict[str, Any]:
    registry = read_json(DATASET_INFO)
    entry = registry[dataset_id]
    data_path = ROOT / "data" / str(entry["file_name"])
    profile_path = PROFILE_DIR / f"{dataset_id}.qwen3_nothink.jsonl"
    raw_max = _profile_max(profile_path)
    return {
        "dataset_id": dataset_id,
        "dataset_category": DATASET_CATEGORIES[dataset_id],
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "dataset_profile_path": str(profile_path.resolve()),
        "dataset_profile_sha256": sha256_file(profile_path),
        "raw_profile_max": raw_max,
        "aligned_effective_sequence": int(
            math.ceil(min(cutoff_len, raw_max) / 8.0) * 8
        ),
    }


def _models() -> dict[str, dict[str, Any]]:
    inventory = read_json(MODEL_INVENTORY)
    return {str(row["id"]): dict(row) for row in inventory["models"]}


def _template() -> dict[str, Any]:
    return dict(
        next(
            row
            for row in read_jsonl(TEMPLATE_QUEUE)
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
    cutoff = int(workload["cutoff_len"])
    dataset = _dataset_binding(str(workload["dataset_id"]), cutoff)
    gpu_count = int(workload["gpu_count"])
    mbs = int(workload["mbs"])
    if TARGET_GBS % (gpu_count * mbs):
        raise ValueError("target GBS is not divisible by gpu_count * mbs")
    zero_stage = int(workload["zero_stage"])
    identity = {**workload, "campaign_id": CAMPAIGN_ID}
    job = dict(template)
    job.update(
        {
            "schema": JOB_SCHEMA,
            "job_id": stable_id("h800ubc3", identity),
            "campaign_id": CAMPAIGN_ID,
            "phase_id": PHASE_ID,
            "experiment_group": "UNIFIED_BOUNDED_CANARY_V3",
            "evidence_role": "prospective_frozen_v3_canary",
            "probe_role": str(workload["probe_role"]),
            "purpose": "frozen_v3_prospective_boundary_validation",
            **dataset,
            "cutoff_len": cutoff,
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
            "zero": "none" if zero_stage == 0 else f"zero{zero_stage}",
            "gc": bool(workload["gc"]),
            "gradient_checkpointing": bool(workload["gc"]),
            "gradient_accumulation_steps": TARGET_GBS // (gpu_count * mbs),
            "target_gbs": TARGET_GBS,
            "max_samples": 256,
            "fidelity": "prospective_memory_canary_1plus4",
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "repeat": 0,
            "packing": False,
            "offload": False,
            "parallel_class": "gpu_partitionable",
            "requires_external_node_idle": False,
            "publication_allowed": False,
            "oom_role": "prospective_admission_failure_if_v3_admitted",
            "oom_is_expected_evidence": True,
            "mechanism_id": (
                f"{workload['train_type']}_zero{zero_stage}_"
                f"gc{int(bool(workload['gc']))}_{gpu_count}gpu_pack0"
            ),
            "scenario_id": stable_id("ubc3scn", identity),
            "split_unit_id": str(workload["dataset_id"]),
            "calibration_partition": {
                "role": "prospective_acceptance",
                "policy": "unified_bounded_v3_frozen_canary_v1",
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


def _prediction_record(
    job: dict[str, Any],
    *,
    inventory: dict[str, Any],
    models: dict[str, dict[str, Any]],
    capacity: int,
    profile_cache: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    feature_job = dict(job)
    runtime_parameters = v3_data.RUNTIME_BASE_PARAMETERS.get(str(job["model_id"]))
    if runtime_parameters is not None:
        feature_job["model_parameters"] = runtime_parameters
    reference, features = base._current_features(
        feature_job,
        model_by_id=models,
        fixed_lora=inventory["fixed_lora"],
        capacity_bytes=capacity,
        profile_cache=profile_cache,
    )
    return {
        "record_id": f"prospective_canary::{job['job_id']}",
        "source_id": f"prospective::{job['dataset_profile_sha256']}",
        "origin": CAMPAIGN_ID,
        "role": str(job["probe_role"]),
        "state": "exact",
        "reference_bytes": reference,
        "target_reserved_bytes": 1.0,
        "censor_lower_bytes": None,
        "features": features,
        "model_id": str(job["model_id"]),
        "train_type": str(job["train_type"]),
        "gpu_count": int(job["gpu_count"]),
        "zero_stage": int(job["zero_stage"]),
        "gc": bool(job["gc"]),
        "mbs": int(job["mbs"]),
        "cutoff_len": int(job["cutoff_len"]),
        "packing": False,
        "profile_sha256": str(job["dataset_profile_sha256"]),
    }


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
        "objective": "frozen-v3 prospective boundary canary",
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
        {"id": dataset_id, "category": category, "target_cutoffs": [4096, 8192]}
        for dataset_id, category in DATASET_CATEGORIES.items()
    ]
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    existing = [
        str(path)
        for path in (args.queue, args.design, args.predictions, args.experiment)
        if path.exists()
    ]
    if existing:
        raise SystemExit(f"refusing to overwrite existing outputs: {existing}")

    inventory = read_json(MODEL_INVENTORY)
    models = {str(row["id"]): dict(row) for row in inventory["models"]}
    feature_models = {key: dict(value) for key, value in models.items()}
    for model_id, parameters in v3_data.RUNTIME_BASE_PARAMETERS.items():
        feature_models[model_id]["actual_parameters"] = parameters
    capacity = int(read_json(base.DEFAULT_HARDWARE)["memory_bytes_reported_by_torch"])
    template = _template()
    jobs = [_job(dict(row), models=models, template=template) for row in WORKLOADS]
    if (
        len(jobs) != EXPECTED_JOBS
        or len({row["job_id"] for row in jobs}) != EXPECTED_JOBS
    ):
        raise ValueError("canary queue must contain exactly ten unique jobs")
    profile_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    records = [
        _prediction_record(
            job,
            inventory=inventory,
            models=feature_models,
            capacity=capacity,
            profile_cache=profile_cache,
        )
        for job in jobs
    ]
    v2 = load_artifact(V2_ARTIFACT)
    v3 = load_artifact(V3_ARTIFACT)
    v2_predictions = predict_records(records, v2)
    v3_predictions = predict_records(records, v3)
    frozen_rows = []
    for workload, job, record, old, new in zip(
        WORKLOADS, jobs, records, v2_predictions, v3_predictions
    ):
        expected_admit = workload["probe_role"] == "admitted_canary"
        if bool(new["admitted"]) != expected_admit:
            raise ValueError(f"frozen v3 decision drifted for {job['job_id']}")
        frozen_rows.append(
            {
                "job_id": str(job["job_id"]),
                "probe_role": str(workload["probe_role"]),
                "record": record,
                "v2": old,
                "v3": new,
            }
        )
    prediction_report: dict[str, Any] = {
        "schema": PREDICTION_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_any_canary_outcome",
        "outcomes_observed": 0,
        "model_or_margin_refit_allowed": False,
        "production_model_mutated": False,
        "v2_artifact": {
            "path": str(V2_ARTIFACT.resolve()),
            "sha256": sha256_file(V2_ARTIFACT),
        },
        "v3_artifact": {
            "path": str(V3_ARTIFACT.resolve()),
            "sha256": sha256_file(V3_ARTIFACT),
        },
        "ordered_job_payload_sha256": sha256_json(jobs),
        "rows": frozen_rows,
    }
    prediction_report["report_sha256"] = sha256_json(prediction_report)

    args.queue.parent.mkdir(parents=True, exist_ok=True)
    args.design.parent.mkdir(parents=True, exist_ok=True)
    args.experiment.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.queue, jobs)
    write_json(args.experiment, _experiment())
    write_json(args.predictions, prediction_report)
    design: dict[str, Any] = {
        "schema": DESIGN_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "status": "prepared_waiting_for_eight_h800s_to_be_idle",
        "gpu_training_started": False,
        "execution_authorized": True,
        "publication_allowed": False,
        "objective": "Prospectively validate frozen-v3 center and admission at the 95%-capacity boundary.",
        "old_missing_34_jobs_included": False,
        "admitted_canary_jobs": sum(
            row["probe_role"] == "admitted_canary" for row in WORKLOADS
        ),
        "rejected_boundary_controls": sum(
            row["probe_role"] == "rejected_boundary_control" for row in WORKLOADS
        ),
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
            "frozen_predictions": {
                "path": str(args.predictions.resolve()),
                "sha256": sha256_file(args.predictions),
            },
            "v3_artifact": {
                "path": str(V3_ARTIFACT.resolve()),
                "sha256": sha256_file(V3_ARTIFACT),
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
                "predictions": str(args.predictions),
                "experiment": str(args.experiment),
                "jobs": len(jobs),
                "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in jobs),
                "v3_admitted": sum(bool(row["v3"]["admitted"]) for row in frozen_rows),
                "v3_rejected_controls": sum(
                    not bool(row["v3"]["admitted"]) for row in frozen_rows
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
