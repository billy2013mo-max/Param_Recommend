#!/usr/bin/env python3
"""Materialize the exact 13-job eight-GPU resume for Packing boundary stage 1."""

from __future__ import annotations

from datetime import datetime, timezone
import copy
import json
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    MATRIX_DIR,
    RESULTS_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)


CAMPAIGN_ID = "h800_packing_memory_boundary_20260805_v1"
PHASE_ID = "h800_packing_memory_boundary_stage1_v1"
RESUME_REVISION = 2
GPU_IDS = tuple(range(8))
PARENT_QUEUE = MATRIX_DIR / "h800_packing_memory_boundary_stage1_v1.jsonl"
PARENT_DESIGN = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_design_v1.json"
PARENT_STATIC = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_static_v1.json"
PARENT_MANIFEST = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_queue_manifest_v1.json"
SELECTION_SOURCE = ARTIFACT_DIR / "h800_packing_memory_boundary_selection_v1.json"
QUEUE = MATRIX_DIR / "h800_packing_memory_boundary_stage1_resume_v2.jsonl"
SELECTION = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_resume_selection_v2.json"
DESIGN = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_resume_design_v2.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_resume_queue_manifest_v2.json"
JOB_DIR = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_resume_jobs_v2"

VALID_TERMINAL_IDS = (
    "h800packboundary1-090943c94be4504e",
    "h800packboundary1-af535b0a18331863",
    "h800packboundary1-00564396fc5eb22f",
)
INVALID_OOM_ID = "h800packboundary1-17c90fc6ae2bc6bc"
PENDING_IDS = (
    "h800packboundary1-e10dbd64a3e56024",
    "h800packboundary1-c0b02b27147c20d6",
    "h800packboundary1-12d2d33a04427ad0",
    "h800packboundary1-f5f00c60b966c5b7",
    "h800packboundary1-b9face75ecbdd936",
    "h800packboundary1-5e8595c393063961",
    "h800packboundary1-2f604ef66169dd09",
    "h800packboundary1-8e94d1211a8baae6",
    "h800packboundary1-b297aa89ec753c26",
    "h800packboundary1-24128aaf2e960a73",
    "h800packboundary1-81bfdd5c1da174dc",
    "h800packboundary1-71b3f86b8e385754",
)


def _binding(path: Path, **extra: Any) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), **extra}


def _status_evidence(job_id: str, *, expected: str, eligible: bool) -> tuple[dict[str, Any], list[Path]]:
    result_dir = RESULTS_DIR / job_id
    status_path = result_dir / "status.json"
    status = read_json(status_path)
    if (
        status.get("classification") != expected
        or status.get("calibration_eligible") is not eligible
        or (eligible and status.get("execution_fingerprint_quality") != "complete")
        or status.get("job_id") != job_id
    ):
        raise ValueError(f"{job_id}: terminal evidence does not match the frozen resume split")
    paths = [status_path]
    for relative in ("latest_attempt.json", "execution_fingerprint.json", "terminal_classification.json"):
        path = result_dir / relative
        if path.is_file():
            paths.append(path)
    return (
        {
            "job_id": job_id,
            "classification": expected,
            "calibration_eligible": eligible,
            "execution_fingerprint_quality": status.get("execution_fingerprint_quality"),
            "status": _binding(status_path),
        },
        paths,
    )


def prepare() -> dict[str, Any]:
    parent = read_jsonl(PARENT_QUEUE)
    parent_by_id = {str(row["job_id"]): row for row in parent}
    if (
        len(parent) != 16
        or len(parent_by_id) != 16
        or tuple(parent_by_id) != (*VALID_TERMINAL_IDS[:1], INVALID_OOM_ID, *VALID_TERMINAL_IDS[1:], *PENDING_IDS)
    ):
        raise ValueError("parent queue identity/order drifted")

    completed: list[dict[str, Any]] = []
    evidence_paths: list[Path] = []
    completed_specs = (
        (VALID_TERMINAL_IDS[0], "success"),
        (VALID_TERMINAL_IDS[1], "oom"),
        (VALID_TERMINAL_IDS[2], "oom"),
    )
    for job_id, classification in completed_specs:
        evidence, paths = _status_evidence(job_id, expected=classification, eligible=True)
        completed.append(evidence)
        evidence_paths.extend(paths)
    invalid, invalid_paths = _status_evidence(INVALID_OOM_ID, expected="oom", eligible=False)
    evidence_paths.extend(invalid_paths)
    invalid_status = read_json(Path(invalid["status"]["path"]))
    errors = invalid_status.get("execution_fingerprint_errors") or []
    if (
        invalid_status.get("execution_fingerprint_quality") != "incomplete"
        or not any("runtime_model_manifest" in str(error) for error in errors)
    ):
        raise ValueError("invalid B2 OOM is not the exact missing-runtime-manifest case")

    pending: list[dict[str, Any]] = []
    for job_id in PENDING_IDS:
        status_path = RESULTS_DIR / job_id / "status.json"
        if status_path.exists():
            raise ValueError(f"pending job unexpectedly became terminal: {job_id}")
        pending.append(parent_by_id[job_id])

    retry = copy.deepcopy(parent_by_id[INVALID_OOM_ID])
    retry.update(
        {
            "resume_revision": RESUME_REVISION,
            "retry_of_job_id": INVALID_OOM_ID,
            "retry_ordinal": 1,
            "retry_reason": "prior_cuda_oom_had_incomplete_runtime_model_manifest",
            "prior_status_path": str(Path(invalid["status"]["path"]).resolve()),
            "prior_status_sha256": invalid["status"]["sha256"],
            "prior_classification": "oom",
            "prior_calibration_eligible": False,
            "execution_sequence_index": len(parent),
        }
    )
    retry.pop("job_id", None)
    retry["job_id"] = stable_id("h800packboundary1retry", retry)
    if (RESULTS_DIR / retry["job_id"]).exists():
        raise ValueError("resume retry job already has a result directory")
    rows = [*pending, retry]
    if len(rows) != 13 or len({str(row["job_id"]) for row in rows}) != 13:
        raise ValueError("resume queue must contain 13 unique jobs")
    write_jsonl(QUEUE, rows)
    for row in rows:
        write_json(JOB_DIR / f"{row['job_id']}.json", row)

    selection: dict[str, Any] = {
        "schema": "sft_h800_packing_memory_boundary_stage1_resume_selection/v2",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": "retain calibration-eligible success/OOM; resume exact nonterminal rows; retry the one ineligible OOM once",
        "parent_queue": _binding(PARENT_QUEUE, jobs=16),
        "valid_terminal_count": 3,
        "valid_terminal": completed,
        "invalid_terminal_count": 1,
        "invalid_terminal": invalid,
        "unchanged_pending_count": len(pending),
        "unchanged_pending_job_ids": list(PENDING_IDS),
        "retry_count": 1,
        "retry_job_id": retry["job_id"],
        "resume_jobs": len(rows),
        "resume_gpu_job_equivalents": sum(int(row["gpu_count"]) for row in rows),
        "high_cutoff_jobs_in_resume": 0,
        "automatic_stage2_release_allowed": False,
    }
    selection["report_sha256"] = sha256_json(selection)
    write_json(SELECTION, selection)

    source_files: dict[str, Path] = {
        "experiment_config": ROOT / "config/experiment.json",
        "hardware_config": ROOT / "config/hardware.json",
        "deepspeed_z2": ROOT / "config/deepspeed/ds_z2.json",
        "deepspeed_z3": ROOT / "config/deepspeed/ds_z3.json",
        "dataset_registry": ROOT / "data/dataset_info.json",
        "model_inventory": ARTIFACT_DIR / "model_inventory.json",
        "dataset_analysis": ARTIFACT_DIR / "dataset_analysis.json",
        "parent_queue": PARENT_QUEUE,
        "parent_design": PARENT_DESIGN,
        "parent_static": PARENT_STATIC,
        "parent_manifest": PARENT_MANIFEST,
        "boundary_selection": SELECTION_SOURCE,
        "resume_selection": SELECTION,
        "preparer": Path(__file__).resolve(),
        "freezer": ROOT / "scripts/freeze_h800_packing_memory_boundary_stage1_resume_v2.py",
        "approval_gate": ROOT / "scripts/approval_gate.py",
        "promoter": ROOT / "scripts/promote_approval_candidate.py",
        "run_job": ROOT / "scripts/run_job.py",
        "scheduler": ROOT / "scripts/scheduler.py",
        "common": ROOT / "scripts/common.py",
        "train_entry": ROOT / "scripts/train_entry.py",
        "metrics_callback": ROOT / "scripts/metrics_callback.py",
        "profiler_callback": ROOT / "scripts/profiler_callback.py",
        "runtime_evidence": ROOT / "scripts/runtime_evidence.py",
    }
    for index, path in enumerate(evidence_paths):
        source_files[f"terminal_evidence_{index}"] = path
    for row in rows:
        for key in ("data_path", "dataset_profile_path", "packing_dataprofile_path", "declared_model_manifest_path"):
            source_files[f"{key}:{row['job_id']}"] = Path(row[key])
    source_bindings = {
        name: _binding(path) for name, path in sorted(source_files.items())
    }
    design: dict[str, Any] = {
        "schema": "sft_h800_packing_memory_boundary_stage1_resume/v2",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "resume_revision": RESUME_REVISION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_gpu_waiting_for_exact_13_job_resume_approval",
        "gpu_training_started": False,
        "objective": "Complete stage-1 U/P repeats and repair the one non-calibratable B2 OOM on eight H800s.",
        "resume_contract": {
            "valid_parent_terminals_reused": 3,
            "unchanged_parent_jobs_resumed": 12,
            "ineligible_oom_retries": 1,
            "resume_jobs": 13,
            "high_cutoff_jobs": 0,
            "oom_role": "right_censored_lower_bound_only_when_calibration_eligible",
            "software_or_evidence_failure_role": "repair_and_rerun_same_configuration",
            "automatic_stage2_release_allowed": False,
        },
        "required_gpu_pool": {
            "gpu_ids": list(GPU_IDS),
            "max_gpu_count_per_job": 2,
            "maximum_parallel_jobs_when_all_idle": 4,
            "preview_two_gpu_masks_when_all_idle": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "join_busy_pool": True,
            "preemption_allowed": False,
        },
        "selection": _binding(SELECTION),
        "queue": _binding(
            QUEUE,
            jobs=len(rows),
            gpu_job_equivalents=sum(int(row["gpu_count"]) for row in rows),
            ordered_job_ids=[str(row["job_id"]) for row in rows],
        ),
        "source_bindings": source_bindings,
        "automatic_next_batch_allowed": False,
        "automatic_packing_recommendation_allowed": False,
        "publication_allowed": False,
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    manifest: dict[str, Any] = {
        "schema": "sft_h800_packing_memory_boundary_stage1_resume_queue_manifest/v2",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "resume_revision": RESUME_REVISION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": _binding(DESIGN),
        "selection": _binding(SELECTION),
        "queue": _binding(QUEUE, jobs=len(rows), ordered_job_ids=[str(row["job_id"]) for row in rows]),
        "high_cutoff_jobs_materialized": 0,
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(QUEUE_MANIFEST, manifest)
    return {
        "design": _binding(DESIGN),
        "selection": _binding(SELECTION),
        "queue": _binding(QUEUE),
        "valid_parent_terminals": 3,
        "unchanged_pending": 12,
        "retries": 1,
        "resume_jobs": len(rows),
        "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in rows),
        "gpu_ids": list(GPU_IDS),
    }


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
