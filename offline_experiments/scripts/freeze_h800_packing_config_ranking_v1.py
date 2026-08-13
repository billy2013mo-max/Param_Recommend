#!/usr/bin/env python3
"""Freeze one split of the Packing configuration-ranking campaign.

Promotion always requires an explicit authorization string supplied at launch
time.  Merely preparing or inspecting this campaign never grants GPU execution.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from approval_gate import build_provenance_binding, build_queue_binding
from common import (
    ARTIFACT_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from prepare_h800_packing_config_ranking_v1 import (
    BASE_RANKER,
    CAMPAIGN_DATA_DIR,
    CAMPAIGN_ID,
    DATASET_REGISTRY,
    DESIGN,
    EXECUTION_POLICY_VERSION,
    EXPECTED_FILL_REPEATS_PER_SPLIT,
    EXPERIMENT,
    EXPECTED_JOBS_PER_SPLIT,
    EXPECTED_RANKING_GROUPS_PER_SPLIT,
    GPU_COUNTS,
    GPU_IDS,
    HOLDOUT_PREDICTIONS,
    JOB_SCHEMA,
    JOBS_DIR,
    INVALIDATED_PILOT,
    MEASURE_STEPS,
    MODEL_INVENTORY,
    PHASE_ID,
    PROFILE_MANIFEST,
    QUEUE_FIT,
    QUEUE_HOLDOUT,
    STATIC,
    WARMUP_STEPS,
)
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch, validate_job


SCRIPTS = ROOT / "scripts"
STAGING = ROOT / "packing_config_ranking_staging"
PREDICTION_SCHEMA = "sft_h800_packing_config_ranking_frozen_predictions/v2"


def queue_for_split(split: str) -> Path:
    if split == "fit":
        return QUEUE_FIT
    if split == "prospective_holdout":
        return QUEUE_HOLDOUT
    raise ValueError(f"unsupported split={split}")


def candidate_for_split(split: str) -> Path:
    suffix = "fit" if split == "fit" else "holdout"
    return ARTIFACT_DIR / f"approval_design_h800_packing_config_ranking_{suffix}_v2_candidate.json"


def _hardware() -> dict[str, Any]:
    hardware = probe_hardware(required_gpu_ids=list(GPU_IDS))
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError("GPU 0-7 are not an exact eight-H800 pool")
    if hardware.get("selected_gpu_compute_processes"):
        raise RuntimeError("one or more authorized GPUs are busy")
    hardware.update(
        {
            "occupancy_is_authorization_time_preflight": True,
            "join_busy_pool_required": True,
            "preemption_allowed": False,
            "gpu_ids_outside_authorized_pool_must_not_be_used": True,
        }
    )
    return hardware


def _validate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not HOLDOUT_PREDICTIONS.is_file():
        raise FileNotFoundError(
            "holdout predictions must be frozen before holdout execution: "
            f"{HOLDOUT_PREDICTIONS}"
        )
    payload = read_json(HOLDOUT_PREDICTIONS)
    if payload.get("schema") != PREDICTION_SCHEMA:
        raise ValueError("holdout prediction schema drifted")
    binding = payload.get("queue_binding") or {}
    if (
        binding.get("path") != str(QUEUE_HOLDOUT.resolve())
        or binding.get("sha256") != sha256_file(QUEUE_HOLDOUT)
        or payload.get("training_outcomes_read") is not False
    ):
        raise ValueError("holdout predictions do not bind the untouched holdout queue")
    predictions = payload.get("predictions")
    expected = {str(row["job_id"]) for row in rows}
    observed: set[str] = set()
    if not isinstance(predictions, list):
        raise ValueError("holdout predictions must be a list")
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            raise ValueError("every holdout prediction must be an object")
        job_id = str(prediction.get("job_id") or "")
        score = prediction.get("predicted_ranking_score")
        if (
            job_id in observed
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not isinstance(prediction.get("admitted"), bool)
        ):
            raise ValueError(f"invalid holdout prediction for {job_id}")
        observed.add(job_id)
    if observed != expected:
        raise ValueError("holdout predictions must cover the exact holdout queue")
    existing = [
        str(row["job_id"])
        for row in rows
        if (ROOT / "results" / str(row["job_id"]) / "latest_attempt.json").is_file()
    ]
    if existing:
        raise ValueError(
            "holdout predictions were not frozen before holdout outcomes existed: "
            f"{existing[:5]}"
        )
    return payload


def _validate(split: str, rows: list[dict[str, Any]]) -> None:
    expected_role = "fit" if split == "fit" else "prospective_holdout"
    if (
        len(rows) != EXPECTED_JOBS_PER_SPLIT
        or len({str(row.get("job_id")) for row in rows})
        != EXPECTED_JOBS_PER_SPLIT
        or [int(row.get("execution_sequence_index", -1)) for row in rows]
        != list(range(EXPECTED_JOBS_PER_SPLIT))
    ):
        raise ValueError("queue is not the exact ordered 132-job split")
    groups: dict[str, list[dict[str, Any]]] = {}
    repeats: list[dict[str, Any]] = []
    for row in rows:
        validate_job(row)
        if row.get("ranking_eligible") is True:
            groups.setdefault(str(row["ranking_group_id"]), []).append(row)
        else:
            repeats.append(row)
        gpu_count = int(row.get("gpu_count", 0))
        expected_parallel_class = (
            "exclusive_pool"
            if gpu_count == 8
            else "disjoint_wave"
            if gpu_count == 4
            else "gpu_partitionable"
        )
        if (
            row.get("schema") != JOB_SCHEMA
            or row.get("campaign_id") != CAMPAIGN_ID
            or row.get("phase_id") != PHASE_ID
            or row.get("split_role") != expected_role
            or gpu_count not in GPU_COUNTS
            or bool(row.get("packing")) is not True
            or int(row.get("mbs", 0)) != 1
            or int(row.get("warmup_steps", -1)) != WARMUP_STEPS
            or int(row.get("measure_steps", -1)) != MEASURE_STEPS
            or row.get("execution_policy_version") != EXECUTION_POLICY_VERSION
            or row.get("parallel_class") != expected_parallel_class
            or bool(row.get("allow_disjoint_wave_for_large_job")) != (gpu_count == 4)
            or row.get("requires_external_node_idle") is not False
            or row.get("homogeneous_card_count_wave") is not True
            or row.get("strict_queue_order") is not True
            or row.get("publication_allowed") is not False
            or int(row.get("repeat", -1)) not in {0, 1}
            or (int(row.get("repeat", -1)) == 0) != (row.get("ranking_eligible") is True)
        ):
            raise ValueError(f"queue row left campaign scope: {row.get('job_id')}")
        for path_key, digest_key in (
            ("data_path", "data_sha256"),
            ("dataset_profile_path", "dataset_profile_sha256"),
        ):
            path = Path(str(row[path_key]))
            if not path.is_file() or sha256_file(path) != row[digest_key]:
                raise ValueError(f"job binding drifted: {row['job_id']}:{path_key}")
        if row.get("dataset_registry_sha256") != sha256_file(DATASET_REGISTRY):
            raise ValueError(f"dataset registry binding drifted: {row['job_id']}")
    if len(groups) != EXPECTED_RANKING_GROUPS_PER_SPLIT:
        raise ValueError("ranking group count drifted")
    if (
        len(repeats) != EXPECTED_FILL_REPEATS_PER_SPLIT
        or any(int(row["gpu_count"]) != 1 for row in repeats)
        or any(row.get("measurement_role") != "full_utilization_repeat" for row in repeats)
    ):
        raise ValueError("full-utilization repeat contract drifted")
    for group_id, current in groups.items():
        gpu_count = int(current[0]["gpu_count"])
        expected_candidates = 6 if gpu_count == 1 else 12
        if len(current) != expected_candidates:
            raise ValueError(
                f"ranking group {group_id} has {len(current)} != {expected_candidates} candidates"
            )
    if split == "prospective_holdout":
        _validate_predictions(rows)


def freeze(split: str, *, authorization: str) -> Path:
    queue = queue_for_split(split)
    rows = read_jsonl(queue)
    _validate(split, rows)
    design = read_json(DESIGN)
    if (
        design.get("campaign_id") != CAMPAIGN_ID
        or design.get("phase_id") != PHASE_ID
        or design.get("gpu_training_started") is not False
        or design.get("bindings", {}).get(
            "fit_queue" if split == "fit" else "holdout_queue", {}
        ).get("sha256")
        != sha256_file(queue)
        or design.get("bindings", {}).get("experiment", {}).get("sha256")
        != sha256_file(EXPERIMENT)
        or design.get("bindings", {}).get("static", {}).get("sha256")
        != sha256_file(STATIC)
    ):
        raise ValueError("campaign design bindings drifted")
    live = read_json(ROOT / "config" / "experiment.json")
    scope = live.get("training_scope") or {}
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("gpu_ids") != list(GPU_IDS)
        or scope.get("exclusive_node_gpu_ids") != list(GPU_IDS)
        or scope.get("gpu_counts") != list(GPU_COUNTS)
        or int(scope.get("max_gpu_count", 0)) != 8
        or live.get("measurement", {}).get("performance_parallelism")
        != "disjoint_gpu_masks"
    ):
        raise ValueError("live experiment config is not the exact campaign scope")

    hardware = _hardware()
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(queue, rows, ROOT)
    scoped_sources = [
        SCRIPTS / "approval_gate.py",
        SCRIPTS / "common.py",
        SCRIPTS / "metrics_callback.py",
        SCRIPTS / "promote_approval_candidate.py",
        SCRIPTS / "run_job.py",
        SCRIPTS / "runtime_evidence.py",
        SCRIPTS / "scheduler.py",
        SCRIPTS / "train_entry.py",
        SCRIPTS / "prepare_h800_packing_config_ranking_v1.py",
        SCRIPTS / "freeze_h800_packing_config_ranking_v1.py",
        SCRIPTS / "evaluate_h800_packing_config_ranking_v1.py",
        SCRIPTS / "freeze_h800_packing_config_ranking_holdout_predictions_v1.py",
        STAGING / "launch_h800_packing_config_ranking_v1.py",
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "hardware.json",
        ROOT / "config" / "models.json",
        ROOT / "config" / "deepspeed" / "ds_z2.json",
        ROOT / "config" / "deepspeed" / "ds_z3.json",
    ]
    provenance_binding = build_provenance_binding(ROOT, source_paths=scoped_sources)
    bound_files = {
        queue,
        DESIGN,
        EXPERIMENT,
        STATIC,
        INVALIDATED_PILOT,
        DATASET_REGISTRY,
        PROFILE_MANIFEST,
        MODEL_INVENTORY,
        BASE_RANKER,
        ARTIFACT_DIR / "provenance.json",
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "hardware.json",
        ROOT / "config" / "models.json",
        *scoped_sources,
    }
    if split == "prospective_holdout":
        bound_files.add(HOLDOUT_PREDICTIONS)
    for row in rows:
        bound_files.add(Path(str(row["data_path"])))
        bound_files.add(Path(str(row["dataset_profile_path"])))
        bound_files.add(JOBS_DIR / f"{row['job_id']}.json")
    missing = [str(path) for path in bound_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound paths are absent: {missing}")
    file_manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=str)
        if path.resolve().is_relative_to(ROOT.resolve())
    }
    candidate = candidate_for_split(split)
    approval_design = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{split}",
        "authorization": authorization,
        "campaign_design": {
            "path": str(DESIGN.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(DESIGN),
            "report_sha256": design["report_sha256"],
        },
        "file_sha256": file_manifest,
        "execution_order": [PHASE_ID],
        "allowed_job_ids": queue_binding["ordered_job_ids"],
        "authorized_gpu_ids": list(GPU_IDS),
        "max_gpu_count": 8,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": (
                "strict GPU-count blocks; 1/2/4/8-card homogeneous full-node "
                "concurrency is 8/4/2/1; no cross-card-count overlap"
            ),
        },
        "oom_policy": "record CUDA OOM as evidence; no automatic retry",
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": {
            "allowed_job_ids": queue_binding["ordered_job_ids"],
            "queue_path": queue_binding["path"],
            "queue_sha256": queue_binding["sha256"],
            "ordered_job_ids": queue_binding["ordered_job_ids"],
            "ordered_job_payload_sha256": queue_binding[
                "ordered_job_payload_sha256"
            ],
            "job_payload_sha256": queue_binding["job_payload_sha256"],
        },
    }
    write_json(candidate, approval_design)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=("fit", "prospective_holdout"), required=True
    )
    parser.add_argument("--authorization", default="GPU execution not authorized")
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    if args.promote and args.authorization == "GPU execution not authorized":
        parser.error("--promote requires an explicit --authorization")
    candidate = freeze(args.split, authorization=args.authorization)
    digest = sha256_file(candidate)
    report = promote_candidate(
        candidate_path=candidate,
        expected_candidate_sha256=digest,
        project_root=ROOT,
        authorization=args.authorization,
        approved_by="user",
        promote=args.promote,
    )
    print(
        json.dumps(
            {
                "candidate": str(candidate.resolve()),
                "sha256": digest,
                "split": args.split,
                "jobs": EXPECTED_JOBS_PER_SPLIT,
                "promotion": report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
