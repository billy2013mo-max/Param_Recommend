#!/usr/bin/env python3
"""Materialize the exact seven-job resume after continuation-v1 stopped safely."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, RESULTS_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json, write_jsonl
from prepare_h800_packing_platform_v4_interactions_batch1_continuation_v1 import CAMPAIGN_ID, PHASE_ID


SCHEMA = "sft_h800_packing_platform_v4_interactions_batch1_continuation_resume/v2"
PARENT_QUEUE = MATRIX_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_v1.jsonl"
PARENT_DESIGN = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_design_v1.json"
PARENT_SELECTION = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_selection_v1.json"
QUEUE = MATRIX_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_resume_v2.jsonl"
SELECTION = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_resume_selection_v2.json"
DESIGN = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_resume_design_v2.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_resume_queue_manifest_v2.json"
EXPECTED_COMPLETED_IDS = (
    "h800packv4int1cont-d814423c2c409b22",
    "h800packv4int1cont-bf81a330ce27c922",
    "h800packv4int1cont-c1a1ac38dde78914",
)
EXPECTED_REMAINING_IDS = (
    "h800packv4int1cont-c027f1e65ab40835",
    "h800packv4int1cont-0f88767ad9e630bf",
    "h800packv4int1cont-046e4cbe01f88310",
    "h800packv4int1cont-eef71c827e5c11ce",
    "h800packv4int1cont-0aa327da781c7c70",
    "h800packv4int1cont-569eb0c3f561316b",
    "h800packv4int1cont-ecc537d9ae534966",
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
        raise ValueError(f"continuation predecessor is not a durable eligible success: {job_id}")
    attempt = result_root / str(latest["attempt_path"])
    fingerprint = attempt / "execution_fingerprint.json"
    summaries = sorted((attempt / "metrics").glob("summary.rank*.json"))
    expected_ranks = int((status.get("classification_evidence") or {}).get("expected_ranks") or 0)
    if not fingerprint.is_file() or len(summaries) != expected_ranks:
        raise ValueError(f"continuation predecessor evidence is incomplete: {job_id}")
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
    if len(parent) != 10 or len({str(row["job_id"]) for row in parent}) != 10:
        raise ValueError("continuation-v1 parent queue is not the exact ten-job queue")
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
            "live continuation partition is not the frozen 3-success/7-pending set: "
            f"completed={completed_ids}, remaining={remaining_ids}"
        )
    if any((RESULTS_DIR / job_id / "latest_attempt.json").exists() for job_id in remaining_ids):
        raise ValueError("a pending resume job unexpectedly has an execution attempt")
    write_jsonl(QUEUE, remaining)

    selection: dict[str, Any] = {
        "schema": "sft_h800_packing_platform_v4_interactions_batch1_continuation_resume_selection/v2",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "selection_policy": "exact ordered set difference after a second fail-closed provenance source-path change",
        "halt_reason": "scripts/fit_h800_effective_sequence_challenger.py appeared after continuation-v1 provenance capture",
        "parent_queue": {**_binding(PARENT_QUEUE), "jobs": 10},
        "completed_success_count": 3,
        "completed": completed,
        "remaining_count": 7,
        "remaining_gpu_job_equivalents": sum(int(row["gpu_count"]) for row in remaining),
        "remaining_job_ids": list(remaining_ids),
    }
    selection["report_sha256"] = sha256_json(selection)
    write_json(SELECTION, selection)

    source_files: dict[str, Path] = {
        "plan": ROOT.parent / "Neat_Packing联合搜索_数据实验与建模完整计划_2026-08-04.md",
        "packing_decision_v2": ROOT.parent / "Packing 决策逻辑 v2.md",
        "experiment_config": ROOT / "config" / "experiment.json",
        "hardware_config": ROOT / "config" / "hardware.json",
        "dataset_registry": ROOT / "data" / "dataset_info.json",
        "parent_queue": PARENT_QUEUE,
        "parent_design": PARENT_DESIGN,
        "parent_selection": PARENT_SELECTION,
        "resume_selection": SELECTION,
        "preparer": Path(__file__).resolve(),
        "freezer": ROOT / "scripts" / "freeze_h800_packing_platform_v4_interactions_batch1_continuation_resume_v2.py",
        "combined_evaluator": ROOT / "scripts" / "evaluate_h800_packing_platform_v4_interactions_batch1_continuation_v1.py",
        "run_job": ROOT / "scripts" / "run_job.py",
        "scheduler": ROOT / "scripts" / "scheduler.py",
        "train_entry": ROOT / "scripts" / "train_entry.py",
        "metrics_callback": ROOT / "scripts" / "metrics_callback.py",
    }
    source_files.update(completed_files)
    for row in remaining:
        source_files[f"slice:{row['job_id']}"] = Path(row["data_path"])
        source_files[f"profile:{row['job_id']}"] = Path(row["dataset_profile_path"])

    design: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "resume_revision": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_gpu_waiting_for_exact_seven_job_resume_approval",
        "gpu_training_started": False,
        "continuation_only": True,
        "automatic_next_batch_allowed": False,
        "publication_allowed": False,
        "methodology_invariant": "The seven queue rows are byte-equivalent JSON objects copied from continuation-v1; no training or modeling field changed.",
        "required_gpu_pool": {
            "gpu_ids": [0, 1, 4, 5, 6, 7],
            "max_gpu_count_per_job": 2,
            "two_gpu_masks": [[0, 1], [4, 5], [6, 7]],
            "preemption_allowed": False,
            "excluded_busy_gpu_ids": [2, 3],
            "join_busy_pool": True,
        },
        "selection": _binding(SELECTION),
        "queue": {
            **_binding(QUEUE),
            "job_count": 7,
            "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in remaining),
            "ordered_job_ids": list(remaining_ids),
        },
        "source_bindings": {name: _binding(path) for name, path in sorted(source_files.items())},
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    manifest: dict[str, Any] = {
        "schema": "sft_h800_packing_platform_v4_interactions_batch1_continuation_resume_queue_manifest/v2",
        "campaign_id": CAMPAIGN_ID,
        "design": _binding(DESIGN),
        "selection": _binding(SELECTION),
        "queue": {**_binding(QUEUE), "job_count": 7, "ordered_job_ids": list(remaining_ids)},
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(QUEUE_MANIFEST, manifest)
    return {
        "design": _binding(DESIGN),
        "selection": _binding(SELECTION),
        "queue": _binding(QUEUE),
        "completed_continuation_successes": 3,
        "remaining_jobs": 7,
        "remaining_gpu_job_equivalents": sum(int(row["gpu_count"]) for row in remaining),
    }


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
