#!/usr/bin/env python3
"""Freeze a prospective center-accuracy validation on four real business datasets."""

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

CAMPAIGN_ID = "h800_unified_bounded_business_generalization_20260810_v3"
PHASE_ID = "h800_unified_bounded_business_generalization_v3"
JOB_SCHEMA = "sft_h800_unified_bounded_business_generalization_job/v3"
DESIGN_SCHEMA = "sft_h800_unified_bounded_business_generalization_design/v3"
PREDICTION_SCHEMA = "sft_h800_unified_bounded_business_generalization_predictions/v3"
TARGET_GBS = 32
WARMUP_STEPS = 1
MEASURE_STEPS = 4
EXPECTED_JOBS = 12
GPU_IDS = tuple(range(8))

DEFAULT_QUEUE = MATRIX_DIR / "h800_unified_bounded_business_generalization_jobs_v3.jsonl"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_unified_bounded_business_generalization_design_v3.json"
DEFAULT_PREDICTIONS = ARTIFACT_DIR / "h800_unified_bounded_business_generalization_frozen_predictions_v3.json"
STAGING_DIR = ROOT / "unified_bounded_business_generalization_staging"
DEFAULT_EXPERIMENT = STAGING_DIR / "experiment.h800_unified_bounded_business_generalization_v3.json"
MODEL_INVENTORY = ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_v1.json"
TEMPLATE_QUEUE = MATRIX_DIR / "h800_unified_resource_evidence_jobs_v1.jsonl"
DATASET_INFO = ROOT / "data" / "dataset_info.json"
PROFILE_DIR = ARTIFACT_DIR / "real_business_packing_cutoff_mbs_v1" / "profiles"
V3_ARTIFACT = ARTIFACT_DIR / "h800_unified_bounded_memory_candidate_v3.json"
V3_DEVELOPMENT_PREDICTIONS = (
    ROOT
    / "diagnostics"
    / "h800_unified_bounded_memory_v3_20260810"
    / "final_fit_predictions.jsonl"
)

DATASETS = {
    "real_177870_short_qwen3_v1": {
        "category": "very_short_concentrated",
        "cutoff_len": 1024,
    },
    "real_71014_short_tail_qwen3_v1": {
        "category": "short_long_tail",
        "cutoff_len": 4096,
    },
    "real_4500_education_qwen3_v1": {
        "category": "education_medium_long",
        "cutoff_len": 8192,
    },
    "real_4500_content_longtail_qwen3_v1": {
        "category": "content_broad_long_tail",
        "cutoff_len": 16384,
    },
}

# Three mechanisms per real business dataset.  These configurations are new to
# the v3 development table and differ from the old qwen3-8B/LoRA/1-GPU/GC-on
# real-business packing campaign.
WORKLOADS: tuple[dict[str, Any], ...] = (
    {"dataset_id": "real_177870_short_qwen3_v1", "model_id": "qwen3_4b", "train_type": "full", "gpu_count": 1, "mbs": 4, "zero_stage": 0, "gc": False},
    {"dataset_id": "real_177870_short_qwen3_v1", "model_id": "qwen3_8b", "train_type": "lora", "gpu_count": 1, "mbs": 4, "zero_stage": 0, "gc": False},
    {"dataset_id": "real_177870_short_qwen3_v1", "model_id": "qwen3_14b", "train_type": "lora", "gpu_count": 2, "mbs": 4, "zero_stage": 3, "gc": False},
    {"dataset_id": "real_71014_short_tail_qwen3_v1", "model_id": "qwen3_4b", "train_type": "lora", "gpu_count": 1, "mbs": 4, "zero_stage": 0, "gc": False},
    {"dataset_id": "real_71014_short_tail_qwen3_v1", "model_id": "qwen3_8b", "train_type": "full", "gpu_count": 2, "mbs": 2, "zero_stage": 3, "gc": True},
    {"dataset_id": "real_71014_short_tail_qwen3_v1", "model_id": "qwen3_14b", "train_type": "full", "gpu_count": 2, "mbs": 1, "zero_stage": 3, "gc": True},
    {"dataset_id": "real_4500_education_qwen3_v1", "model_id": "qwen3_4b", "train_type": "full", "gpu_count": 1, "mbs": 2, "zero_stage": 0, "gc": True},
    {"dataset_id": "real_4500_education_qwen3_v1", "model_id": "qwen3_8b", "train_type": "lora", "gpu_count": 2, "mbs": 2, "zero_stage": 2, "gc": False},
    {"dataset_id": "real_4500_education_qwen3_v1", "model_id": "qwen3_14b", "train_type": "lora", "gpu_count": 2, "mbs": 1, "zero_stage": 3, "gc": True},
    {"dataset_id": "real_4500_content_longtail_qwen3_v1", "model_id": "qwen3_4b", "train_type": "lora", "gpu_count": 1, "mbs": 1, "zero_stage": 0, "gc": True},
    {"dataset_id": "real_4500_content_longtail_qwen3_v1", "model_id": "qwen3_8b", "train_type": "full", "gpu_count": 2, "mbs": 1, "zero_stage": 3, "gc": True},
    {"dataset_id": "real_4500_content_longtail_qwen3_v1", "model_id": "qwen3_14b", "train_type": "full", "gpu_count": 2, "mbs": 1, "zero_stage": 3, "gc": True},
)

ACCEPTANCE_THRESHOLDS = {
    "minimum_success_jobs": 10,
    "minimum_success_per_dataset": 2,
    "center_source_equal_mape_max": 0.15,
    "absolute_center_signed_bias_max": 0.10,
    "center_p90_ape_max": 0.30,
    "maximum_oom_jobs": 0,
}


def _profile_max(path: Path) -> int:
    maximum = max(int(row["total_tokens"]) for row in read_jsonl(path))
    if maximum <= 0:
        raise ValueError(f"profile is empty: {path}")
    return maximum


def _dataset_binding(dataset_id: str) -> dict[str, Any]:
    spec = DATASETS[dataset_id]
    cutoff = int(spec["cutoff_len"])
    registry = read_json(DATASET_INFO)
    entry = registry[dataset_id]
    data_path = ROOT / "data" / str(entry["file_name"])
    profile_path = PROFILE_DIR / f"{dataset_id}.qwen3_nothink.jsonl"
    raw_max = _profile_max(profile_path)
    return {
        "dataset_id": dataset_id,
        "dataset_category": str(spec["category"]),
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "dataset_profile_path": str(profile_path.resolve()),
        "dataset_profile_sha256": sha256_file(profile_path),
        "raw_profile_max": raw_max,
        "aligned_effective_sequence": int(math.ceil(min(cutoff, raw_max) / 8.0) * 8),
    }


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


def _job(workload: dict[str, Any], *, models: dict[str, dict[str, Any]], template: dict[str, Any]) -> dict[str, Any]:
    dataset_id = str(workload["dataset_id"])
    cutoff = int(DATASETS[dataset_id]["cutoff_len"])
    dataset = _dataset_binding(dataset_id)
    model = models[str(workload["model_id"])]
    gpu_count = int(workload["gpu_count"])
    mbs = int(workload["mbs"])
    if TARGET_GBS % (gpu_count * mbs):
        raise ValueError("target GBS is not divisible by gpu_count * mbs")
    zero_stage = int(workload["zero_stage"])
    identity = {**workload, "cutoff_len": cutoff, "campaign_id": CAMPAIGN_ID}
    job = dict(template)
    job.update(
        {
            "schema": JOB_SCHEMA,
            "job_id": stable_id("h800ubg3", identity),
            "campaign_id": CAMPAIGN_ID,
            "phase_id": PHASE_ID,
            "experiment_group": "UNIFIED_BOUNDED_BUSINESS_GENERALIZATION_V3",
            "evidence_role": "prospective_business_dataset_center_acceptance",
            "purpose": "frozen_v3_center_generalization_on_real_business_data",
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
            "fidelity": "prospective_business_center_1plus4",
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "repeat": 0,
            "packing": False,
            "offload": False,
            "parallel_class": "gpu_partitionable",
            "requires_external_node_idle": False,
            "publication_allowed": False,
            "oom_role": "right_censored_and_business_acceptance_failure",
            "oom_is_expected_evidence": True,
            "mechanism_id": f"{workload['train_type']}_zero{zero_stage}_gc{int(bool(workload['gc']))}_{gpu_count}gpu_pack0",
            "scenario_id": stable_id("ubg3scn", identity),
            "split_unit_id": dataset_id,
            "calibration_partition": {
                "role": "prospective_acceptance",
                "policy": "unified_bounded_v3_real_business_generalization_v1",
                "split_unit_id": dataset_id,
            },
        }
    )
    for key in ("packing_contract", "packing_dataprofile_path", "packing_dataprofile_sha256", "samples_per_pack"):
        job.pop(key, None)
    return job


def _prediction_record(job: dict[str, Any], *, inventory: dict[str, Any], models: dict[str, dict[str, Any]], capacity: int, profile_cache: dict[tuple[str, int, int], dict[str, Any]]) -> dict[str, Any]:
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
        "record_id": f"prospective_business::{job['job_id']}",
        "source_id": str(job["dataset_id"]),
        "origin": CAMPAIGN_ID,
        "role": "prospective_business_dataset_center_acceptance",
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
    experiment = dict(read_json(ROOT / "unified_resource_staging" / "experiment.h800_unified_resource_evidence_v1.json"))
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
        "objective": "frozen-v3 center generalization on real business datasets",
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
        {"id": dataset_id, "category": spec["category"], "target_cutoffs": [spec["cutoff_len"]]}
        for dataset_id, spec in DATASETS.items()
    ]
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    existing = [str(path) for path in (args.queue, args.design, args.predictions, args.experiment) if path.exists()]
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
    if len(jobs) != EXPECTED_JOBS or len({row["job_id"] for row in jobs}) != EXPECTED_JOBS:
        raise ValueError("business generalization queue size or uniqueness drifted")

    development_sources = {
        str(row["source_id"]) for row in read_jsonl(V3_DEVELOPMENT_PREDICTIONS)
    }
    source_overlap = sorted(set(DATASETS) & development_sources)
    if source_overlap:
        raise ValueError(f"business datasets entered v3 development: {source_overlap}")

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
    artifact = load_artifact(V3_ARTIFACT)
    predictions = predict_records(records, artifact)
    if not all(bool(row["admitted"]) for row in predictions):
        rejected = [job["job_id"] for job, row in zip(jobs, predictions) if not row["admitted"]]
        raise ValueError(f"center-accuracy jobs must all be v3-admitted: {rejected}")
    frozen_rows = [
        {"job_id": str(job["job_id"]), "dataset_id": str(job["dataset_id"]), "record": record, "v3": prediction}
        for job, record, prediction in zip(jobs, records, predictions)
    ]
    frozen: dict[str, Any] = {
        "schema": PREDICTION_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_any_business_outcome",
        "outcomes_observed": 0,
        "model_or_margin_refit_allowed": False,
        "production_model_mutated": False,
        "v3_artifact": {"path": str(V3_ARTIFACT.resolve()), "sha256": sha256_file(V3_ARTIFACT)},
        "ordered_job_payload_sha256": sha256_json(jobs),
        "rows": frozen_rows,
    }
    frozen["report_sha256"] = sha256_json(frozen)

    args.queue.parent.mkdir(parents=True, exist_ok=True)
    args.design.parent.mkdir(parents=True, exist_ok=True)
    args.experiment.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.queue, jobs)
    write_json(args.experiment, _experiment())
    write_json(args.predictions, frozen)
    design: dict[str, Any] = {
        "schema": DESIGN_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "status": "prepared_waiting_for_eight_h800s_to_be_idle",
        "gpu_training_started": False,
        "execution_authorized": True,
        "publication_allowed": False,
        "objective": "Prospectively validate frozen-v3 center accuracy on four real business datasets excluded from v3 development.",
        "evidence_contract": {
            "dataset_source_id_overlap_with_v3_development": source_overlap,
            "configuration_outcomes_frozen_before_execution": True,
            "raw_datasets_existed_before_v3": True,
            "claim": "prospective configuration-level business validation; not a never-before-seen raw-data collection",
        },
        "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
        "old_missing_34_jobs_included": False,
        "datasets": list(DATASETS),
        "jobs_per_dataset": {dataset_id: sum(row["dataset_id"] == dataset_id for row in WORKLOADS) for dataset_id in DATASETS},
        "ordered_job_ids": [str(row["job_id"]) for row in jobs],
        "ordered_job_payload_sha256": sha256_json(jobs),
        "bindings": {
            "queue": {"path": str(args.queue.resolve()), "sha256": sha256_file(args.queue)},
            "experiment": {"path": str(args.experiment.resolve()), "sha256": sha256_file(args.experiment)},
            "frozen_predictions": {"path": str(args.predictions.resolve()), "sha256": sha256_file(args.predictions)},
            "v3_artifact": {"path": str(V3_ARTIFACT.resolve()), "sha256": sha256_file(V3_ARTIFACT)},
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(args.design, design)
    print(json.dumps({
        "queue": str(args.queue),
        "design": str(args.design),
        "predictions": str(args.predictions),
        "experiment": str(args.experiment),
        "jobs": len(jobs),
        "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in jobs),
        "datasets": list(DATASETS),
        "v3_center_gib": {row["job_id"]: prediction["center_bytes"] / (1 << 30) for row, prediction in zip(jobs, predictions)},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
