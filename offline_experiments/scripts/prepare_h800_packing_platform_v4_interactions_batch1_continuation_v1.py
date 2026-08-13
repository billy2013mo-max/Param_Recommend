#!/usr/bin/env python3
"""Materialize the exact 10-job continuation of Packing interactions batch 1."""

from __future__ import annotations

from datetime import datetime, timezone
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


SCHEMA = "sft_h800_packing_platform_v4_interactions_batch1_continuation_design/v1"
JOB_SCHEMA = "sft_h800_packing_platform_v4_interactions_batch1_continuation_job/v1"
CAMPAIGN_ID = "h800_packing_platform_v4_interactions_batch1_continuation_20260804_v1"
PHASE_ID = "h800_packing_platform_v4_interactions_batch1_continuation_v1"
PARENT_CAMPAIGN_ID = "h800_packing_platform_v4_interactions_batch1_20260804_v1"
PARENT_PHASE_ID = "h800_packing_platform_v4_interactions_batch1_v1"
GPU_POOL = (0, 1, 4, 5, 6, 7)

PARENT_QUEUE = MATRIX_DIR / "h800_packing_platform_v4_interactions_batch1_v1.jsonl"
PARENT_DESIGN = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_design_v1.json"
PARENT_STATIC = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_static_features_v1.json"
PARENT_APPROVAL = ARTIFACT_DIR / "approval_design_h800_packing_platform_v4_interactions_batch1_v1_candidate.json"
INVENTORY = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_model_inventory_v1.json"
SELECTION = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_selection_v1.json"
QUEUE = MATRIX_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_design_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_queue_manifest_v1.json"

EXPECTED_REMAINING_SOURCE_IDS = (
    "h800packv4int1-f4b408a1ae485b3b",
    "h800packv4int1-785af99d84d61480",
    "h800packv4int1-9ad5faba4a1b7bbc",
    "h800packv4int1-087aa39e536744dd",
    "h800packv4int1-16167af6f1a4a737",
    "h800packv4int1-c59abeaa42d52512",
    "h800packv4int1-889a3a7d6af7fc5f",
    "h800packv4int1-a546aaa4dea7f103",
    "h800packv4int1-6493873ba9fe554d",
    "h800packv4int1-65ff26c6eb063391",
)


def _binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _completed_evidence(job_id: str) -> tuple[dict[str, Any], dict[str, Path]]:
    result_root = RESULTS_DIR / job_id
    status_path = result_root / "status.json"
    latest_path = result_root / "latest_attempt.json"
    status = read_json(status_path)
    latest = read_json(latest_path)
    if (
        status.get("classification") != "success"
        or status.get("calibration_eligible") is not True
        or latest.get("classification") != "success"
        or latest.get("calibration_eligible") is not True
        or latest.get("state") != "complete"
        or latest.get("job_id") != job_id
    ):
        raise ValueError(f"predecessor result is not an eligible durable success: {job_id}")
    attempt_root = result_root / str(latest["attempt_path"])
    fingerprint_path = attempt_root / "execution_fingerprint.json"
    summaries = sorted((attempt_root / "metrics").glob("summary.rank*.json"))
    expected_ranks = int((status.get("classification_evidence") or {}).get("expected_ranks") or 0)
    if not fingerprint_path.is_file() or len(summaries) != expected_ranks:
        raise ValueError(f"predecessor execution evidence is incomplete: {job_id}")
    paths = {
        f"completed_status:{job_id}": status_path,
        f"completed_latest:{job_id}": latest_path,
        f"completed_fingerprint:{job_id}": fingerprint_path,
    }
    for rank, path in enumerate(summaries):
        paths[f"completed_summary_rank{rank}:{job_id}"] = path
    return (
        {
            "source_job_id": job_id,
            "classification": "success",
            "calibration_eligible": True,
            "execution_attempt_id": status["execution_attempt_id"],
            "status": _binding(status_path),
            "latest_attempt": _binding(latest_path),
            "execution_fingerprint": _binding(fingerprint_path),
            "metric_summaries": [_binding(path) for path in summaries],
        },
        paths,
    )


def continuation_row(source: dict[str, Any], index: int) -> dict[str, Any]:
    """Map one untouched pending parent payload into the continuation namespace."""

    row = dict(source)
    source_job_id = str(row.pop("job_id"))
    row.update(
        {
            "schema": JOB_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "phase_id": PHASE_ID,
            "candidate_role": "formal_packing_execution_interaction_continuation",
            "source_job_id": source_job_id,
            "continuation_reason": "fail_closed_provenance_source_path_set_change",
            "execution_sequence_index": index,
        }
    )
    row["job_id"] = stable_id("h800packv4int1cont", row)
    return row


def prepare() -> dict[str, Any]:
    parent_rows = read_jsonl(PARENT_QUEUE)
    if (
        len(parent_rows) != 24
        or len({str(row.get("job_id")) for row in parent_rows}) != 24
        or any(row.get("campaign_id") != PARENT_CAMPAIGN_ID for row in parent_rows)
        or any(row.get("phase_id") != PARENT_PHASE_ID for row in parent_rows)
    ):
        raise ValueError("parent queue is not the exact frozen 24-job batch")

    completed: list[dict[str, Any]] = []
    completed_source_files: dict[str, Path] = {}
    remaining_sources: list[dict[str, Any]] = []
    for source in parent_rows:
        job_id = str(source["job_id"])
        status_path = RESULTS_DIR / job_id / "status.json"
        if status_path.is_file():
            evidence, paths = _completed_evidence(job_id)
            completed.append(evidence)
            completed_source_files.update(paths)
        else:
            remaining_sources.append(source)
    remaining_ids = tuple(str(row["job_id"]) for row in remaining_sources)
    if len(completed) != 14 or remaining_ids != EXPECTED_REMAINING_SOURCE_IDS:
        raise ValueError(
            "live predecessor partition is not the frozen 14-success/10-pending set: "
            f"completed={len(completed)}, remaining={remaining_ids}"
        )
    if any((RESULTS_DIR / job_id / "latest_attempt.json").exists() for job_id in remaining_ids):
        raise ValueError("a frozen pending predecessor unexpectedly has an execution attempt")

    jobs = [continuation_row(source, index) for index, source in enumerate(remaining_sources)]
    if len({str(row["job_id"]) for row in jobs}) != 10:
        raise ValueError("continuation job IDs are not unique")
    write_jsonl(QUEUE, jobs)
    jobs_dir = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_jobs_v1"
    for job in jobs:
        write_json(jobs_dir / f"{job['job_id']}.json", job)

    selection: dict[str, Any] = {
        "schema": "sft_h800_packing_platform_v4_interactions_batch1_continuation_selection/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "selection_policy": "exact ordered set difference after fail-closed provenance rejection; no outcome-dependent candidate redesign",
        "halt_reason": "project source path set changed when an external scripts/experiment_effective_sequence_memory_basis.py appeared",
        "parent_queue": {**_binding(PARENT_QUEUE), "jobs": 24},
        "completed_success_count": 14,
        "completed": completed,
        "remaining_count": 10,
        "remaining_gpu_job_equivalents": sum(int(row["gpu_count"]) for row in jobs),
        "remaining_source_job_ids": list(remaining_ids),
        "continuation_job_ids": [str(row["job_id"]) for row in jobs],
        "source_to_continuation_job_id": {
            str(row["source_job_id"]): str(row["job_id"]) for row in jobs
        },
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
        "parent_static": PARENT_STATIC,
        "parent_approval_candidate": PARENT_APPROVAL,
        "continuation_selection": SELECTION,
        "model_inventory": INVENTORY,
        "preparer": Path(__file__).resolve(),
        "freezer": ROOT / "scripts" / "freeze_h800_packing_platform_v4_interactions_batch1_continuation_v1.py",
        "combined_evaluator": ROOT / "scripts" / "evaluate_h800_packing_platform_v4_interactions_batch1_continuation_v1.py",
        "parent_evaluator": ROOT / "scripts" / "evaluate_h800_packing_platform_v4_interactions_batch1_v1.py",
        "run_job": ROOT / "scripts" / "run_job.py",
        "scheduler": ROOT / "scripts" / "scheduler.py",
        "train_entry": ROOT / "scripts" / "train_entry.py",
        "metrics_callback": ROOT / "scripts" / "metrics_callback.py",
    }
    source_files.update(completed_source_files)
    for row in jobs:
        source_files[f"slice:{row['source_job_id']}"] = Path(row["data_path"])
        source_files[f"profile:{row['source_job_id']}"] = Path(row["dataset_profile_path"])

    design: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "parent_campaign_id": PARENT_CAMPAIGN_ID,
        "parent_phase_id": PARENT_PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_gpu_waiting_for_exact_continuation_approval",
        "gpu_training_started": False,
        "continuation_only": True,
        "automatic_next_batch_allowed": False,
        "publication_allowed": False,
        "objective": "Complete only the ten missing rows of the frozen 24-job Packing interaction design.",
        "methodology_invariant": "Every modeling field is copied from its parent row; only provenance namespace and execution index change.",
        "required_gpu_pool": {
            "gpu_ids": list(GPU_POOL),
            "max_gpu_count_per_job": 2,
            "maximum_parallel_gpu_slots": 6,
            "two_gpu_masks": [[0, 1], [4, 5], [6, 7]],
            "preemption_allowed": False,
            "excluded_busy_gpu_ids": [2, 3],
            "join_busy_pool": True,
        },
        "selection": _binding(SELECTION),
        "queue": {
            **_binding(QUEUE),
            "job_count": len(jobs),
            "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in jobs),
            "ordered_job_ids": [str(row["job_id"]) for row in jobs],
            "ordered_source_job_ids": [str(row["source_job_id"]) for row in jobs],
        },
        "measurement_contract": {
            "inherited_from_parent": _binding(PARENT_DESIGN),
            "warmup_steps": 2,
            "measure_steps": 8,
            "token_source": "consumed_token_ledger/v1",
            "global_metrics_sum_across_ranks": True,
            "combine_with_parent_successes_for_final_24_job_evaluation": True,
        },
        "source_bindings": {name: _binding(path) for name, path in sorted(source_files.items())},
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    manifest: dict[str, Any] = {
        "schema": "sft_h800_packing_platform_v4_interactions_batch1_continuation_queue_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "design": _binding(DESIGN),
        "selection": _binding(SELECTION),
        "queue": {
            **_binding(QUEUE),
            "job_count": len(jobs),
            "ordered_job_ids": [str(row["job_id"]) for row in jobs],
            "ordered_source_job_ids": [str(row["source_job_id"]) for row in jobs],
        },
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(QUEUE_MANIFEST, manifest)
    return {
        "design": _binding(DESIGN),
        "selection": _binding(SELECTION),
        "queue": _binding(QUEUE),
        "completed_parent_successes": 14,
        "continuation_jobs": len(jobs),
        "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in jobs),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
