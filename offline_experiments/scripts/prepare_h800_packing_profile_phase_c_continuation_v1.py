#!/usr/bin/env python3
"""Materialize the exact 20-job Phase-C continuation after a safe gate halt."""

from __future__ import annotations

from datetime import datetime, timezone
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
    write_json,
    write_jsonl,
)


CAMPAIGN_ID = "h800_packing_profile_phase_c_20260805_v1"
PHASE_ID = "h800_packing_profile_phase_c_v1"
SCHEMA = "sft_h800_packing_profile_phase_c_continuation_design/v1"
PARENT_QUEUE = MATRIX_DIR / "h800_packing_profile_phase_c_v1.jsonl"
PARENT_DESIGN = ARTIFACT_DIR / "h800_packing_profile_phase_c_design_v1.json"
PARENT_MANIFEST = ARTIFACT_DIR / "h800_packing_profile_phase_c_queue_manifest_v1.json"
PARENT_STATIC = ARTIFACT_DIR / "h800_packing_profile_phase_c_static_v1.json"
PARENT_PREDICTIONS = ARTIFACT_DIR / "h800_packing_profile_phase_c_memory_predictions_v1.json"
QUEUE = MATRIX_DIR / "h800_packing_profile_phase_c_continuation_v1.jsonl"
SELECTION = ARTIFACT_DIR / "h800_packing_profile_phase_c_continuation_selection_v1.json"
DESIGN = ARTIFACT_DIR / "h800_packing_profile_phase_c_continuation_design_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_profile_phase_c_continuation_queue_manifest_v1.json"

EXPECTED_COMPLETED_IDS = (
    "h800packphasec-e90b3aa3f8d58820",
    "h800packphasec-84d9c29b1d5dbcf4",
    "h800packphasec-5d5b659c88c1bb44",
    "h800packphasec-00cfdc77270da867",
)
EXPECTED_REMAINING_IDS = (
    "h800packphasec-d6253d9c955587c7",
    "h800packphasec-bfeb2ef4ae2adfd5",
    "h800packphasec-6c9869ab006180f0",
    "h800packphasec-cb8874b61025e51a",
    "h800packphasec-adbe44503ee05109",
    "h800packphasec-e8749f68e79a8c92",
    "h800packphasec-994171dfe379c625",
    "h800packphasec-c611331db26d4308",
    "h800packphasec-db8b849b2196278e",
    "h800packphasec-5eb178772028f341",
    "h800packphasec-8bb2c3d340c21778",
    "h800packphasec-b1caa6a1b189362f",
    "h800packphasec-44293400e7c5ef5c",
    "h800packphasec-d35ee727eee7a0d8",
    "h800packphasec-16d687f8b4775c40",
    "h800packphasec-91b3407d6dea508f",
    "h800packphasec-c93972754f7b7229",
    "h800packphasec-7d5c113c4925d25a",
    "h800packphasec-68bacad1c6ba6c29",
    "h800packphasec-6431a49434d9f09f",
)


def _binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _success_evidence(job_id: str) -> tuple[dict[str, Any], dict[str, Path]]:
    result_root = RESULTS_DIR / job_id
    status_path = result_root / "status.json"
    latest_path = result_root / "latest_attempt.json"
    status = read_json(status_path)
    latest = read_json(latest_path)
    if (
        status.get("job_id") != job_id
        or status.get("classification") != "success"
        or status.get("calibration_eligible") is not True
        or latest.get("job_id") != job_id
        or latest.get("classification") != "success"
        or latest.get("calibration_eligible") is not True
        or latest.get("state") != "complete"
    ):
        raise ValueError(f"completed Phase-C result is not a durable eligible success: {job_id}")
    attempt = result_root / str(latest["attempt_path"])
    fingerprint = attempt / "execution_fingerprint.json"
    summaries = sorted((attempt / "metrics").glob("summary.rank*.json"))
    expected_ranks = int(
        (status.get("classification_evidence") or {}).get("expected_ranks") or 0
    )
    if not fingerprint.is_file() or expected_ranks != 2 or len(summaries) != expected_ranks:
        raise ValueError(f"completed Phase-C evidence is incomplete: {job_id}")
    paths = {
        f"completed_status:{job_id}": status_path,
        f"completed_latest:{job_id}": latest_path,
        f"completed_fingerprint:{job_id}": fingerprint,
    }
    for rank, path in enumerate(summaries):
        paths[f"completed_summary_rank{rank}:{job_id}"] = path
    return (
        {
            "job_id": job_id,
            "classification": "success",
            "calibration_eligible": True,
            "execution_attempt_id": status["execution_attempt_id"],
            "status": _binding(status_path),
            "latest_attempt": _binding(latest_path),
            "execution_fingerprint": _binding(fingerprint),
            "metric_summaries": [_binding(path) for path in summaries],
        },
        paths,
    )


def prepare() -> dict[str, Any]:
    parent = read_jsonl(PARENT_QUEUE)
    if len(parent) != 24 or len({str(row["job_id"]) for row in parent}) != 24:
        raise ValueError("Phase-C parent queue is not the exact 24-job queue")

    completed: list[dict[str, Any]] = []
    completed_files: dict[str, Path] = {}
    remaining: list[dict[str, Any]] = []
    for row in parent:
        job_id = str(row["job_id"])
        status_path = RESULTS_DIR / job_id / "status.json"
        if status_path.is_file():
            evidence, paths = _success_evidence(job_id)
            completed.append(evidence)
            completed_files.update(paths)
        else:
            remaining.append(row)
    completed_ids = tuple(row["job_id"] for row in completed)
    remaining_ids = tuple(str(row["job_id"]) for row in remaining)
    if completed_ids != EXPECTED_COMPLETED_IDS or remaining_ids != EXPECTED_REMAINING_IDS:
        raise ValueError(
            "live Phase-C partition is not the frozen 4-success/20-pending set: "
            f"completed={completed_ids}, remaining={remaining_ids}"
        )
    # One rejected pre-launch attempt may have a runtime input snapshot, but it
    # must not have produced durable result state for any pending row.
    if any((RESULTS_DIR / job_id / "status.json").exists() for job_id in remaining_ids):
        raise ValueError("a pending continuation job unexpectedly has durable status")
    write_jsonl(QUEUE, remaining)

    selection: dict[str, Any] = {
        "schema": "sft_h800_packing_profile_phase_c_continuation_selection/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "selection_policy": "exact ordered set difference after fail-closed provenance drift",
        "halt_reason": "the shared source manifest changed after the original approval; the scheduler stopped new launches and allowed four in-flight jobs to finish",
        "parent_queue": {**_binding(PARENT_QUEUE), "jobs": 24},
        "completed_success_count": 4,
        "completed": completed,
        "remaining_count": 20,
        "remaining_gpu_job_equivalents": 40,
        "remaining_job_ids": list(remaining_ids),
    }
    selection["report_sha256"] = sha256_json(selection)
    write_json(SELECTION, selection)

    source_files: dict[str, Path] = {
        "experiment_config": ROOT / "config" / "experiment.json",
        "hardware_config": ROOT / "config" / "hardware.json",
        "dataset_registry": ROOT / "data" / "dataset_info.json",
        "parent_queue": PARENT_QUEUE,
        "parent_design": PARENT_DESIGN,
        "parent_manifest": PARENT_MANIFEST,
        "parent_static": PARENT_STATIC,
        "parent_predictions": PARENT_PREDICTIONS,
        "continuation_selection": SELECTION,
        "preparer": Path(__file__).resolve(),
        "freezer": ROOT / "scripts" / "freeze_h800_packing_profile_phase_c_continuation_v1.py",
        "evaluator": ROOT / "scripts" / "evaluate_h800_packing_profile_phase_c_v1.py",
        "run_job": ROOT / "scripts" / "run_job.py",
        "scheduler": ROOT / "scripts" / "scheduler.py",
        "train_entry": ROOT / "scripts" / "train_entry.py",
        "metrics_callback": ROOT / "scripts" / "metrics_callback.py",
    }
    source_files.update(completed_files)
    for row in remaining:
        source_files[f"slice:{row['job_id']}"] = Path(row["data_path"])
        source_files[f"profile:{row['job_id']}"] = Path(row["dataset_profile_path"])
        source_files[f"dataprofile:{row['job_id']}"] = Path(row["packing_dataprofile_path"])

    design: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_gpu_waiting_for_exact_twenty_job_continuation_approval",
        "gpu_training_started": False,
        "continuation_only": True,
        "automatic_next_batch_allowed": False,
        "publication_allowed": False,
        "methodology_invariant": "The 20 queue rows are unchanged JSON objects copied from the original Phase-C queue; no training, safety, or modeling field changed.",
        "required_gpu_pool": {
            "gpu_ids": list(range(8)),
            "max_gpu_count_per_job": 2,
            "two_gpu_masks": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "max_parallel_jobs": 4,
            "preemption_allowed": False,
            "join_busy_pool": True,
        },
        "selection": _binding(SELECTION),
        "queue": {
            **_binding(QUEUE),
            "job_count": 20,
            "gpu_job_equivalents": 40,
            "ordered_job_ids": list(remaining_ids),
        },
        "source_bindings": {
            name: _binding(path) for name, path in sorted(source_files.items())
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)

    manifest: dict[str, Any] = {
        "schema": "sft_h800_packing_profile_phase_c_continuation_queue_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "design": _binding(DESIGN),
        "selection": _binding(SELECTION),
        "queue": {
            **_binding(QUEUE),
            "job_count": 20,
            "ordered_job_ids": list(remaining_ids),
        },
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(QUEUE_MANIFEST, manifest)
    return {
        "design": _binding(DESIGN),
        "selection": _binding(SELECTION),
        "queue": _binding(QUEUE),
        "completed_successes": 4,
        "remaining_jobs": 20,
        "remaining_gpu_job_equivalents": 40,
        "max_parallel_jobs": 4,
    }


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
