#!/usr/bin/env python3
"""Materialize the exact four single-GPU jobs left in Packing batch 1."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, RESULTS_DIR, ROOT, read_jsonl, sha256_json, write_json, write_jsonl
from prepare_h800_packing_platform_v4_interactions_batch1_continuation_resume_v2 import _binding, _success_evidence
from prepare_h800_packing_platform_v4_interactions_batch1_continuation_v1 import CAMPAIGN_ID, PHASE_ID


PARENT_QUEUE = MATRIX_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_v1.jsonl"
PARENT_DESIGN = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_design_v1.json"
PARENT_RESUME_DESIGN = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_resume_design_v2.json"
QUEUE = MATRIX_DIR / "h800_packing_platform_v4_interactions_batch1_final_resume_v3.jsonl"
SELECTION = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_final_resume_selection_v3.json"
DESIGN = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_final_resume_design_v3.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_final_resume_queue_manifest_v3.json"
EXPECTED_COMPLETED_IDS = (
    "h800packv4int1cont-d814423c2c409b22",
    "h800packv4int1cont-bf81a330ce27c922",
    "h800packv4int1cont-c1a1ac38dde78914",
    "h800packv4int1cont-046e4cbe01f88310",
    "h800packv4int1cont-569eb0c3f561316b",
    "h800packv4int1cont-ecc537d9ae534966",
)
EXPECTED_REMAINING_IDS = (
    "h800packv4int1cont-c027f1e65ab40835",
    "h800packv4int1cont-0f88767ad9e630bf",
    "h800packv4int1cont-eef71c827e5c11ce",
    "h800packv4int1cont-0aa327da781c7c70",
)


def prepare() -> dict[str, Any]:
    parent = read_jsonl(PARENT_QUEUE)
    if len(parent) != 10:
        raise ValueError("parent continuation queue must contain ten jobs")
    completed: list[dict[str, Any]] = []
    completed_files: dict[str, Path] = {}
    remaining: list[dict[str, Any]] = []
    for row in parent:
        job_id = str(row["job_id"])
        if (RESULTS_DIR / job_id / "status.json").is_file():
            evidence, paths = _success_evidence(job_id)
            completed.append(evidence)
            completed_files.update(paths)
        else:
            remaining.append(row)
    completed_ids = tuple(row["job_id"] for row in completed)
    remaining_ids = tuple(str(row["job_id"]) for row in remaining)
    if completed_ids != EXPECTED_COMPLETED_IDS or remaining_ids != EXPECTED_REMAINING_IDS:
        raise ValueError(f"expected exact 6-success/4-pending split, got {completed_ids=} {remaining_ids=}")
    if any((RESULTS_DIR / job_id / "latest_attempt.json").exists() for job_id in remaining_ids):
        raise ValueError("a final pending job unexpectedly has an execution attempt")
    if any(int(row["gpu_count"]) != 1 for row in remaining):
        raise ValueError("final resume must contain only four single-GPU jobs")
    write_jsonl(QUEUE, remaining)

    selection: dict[str, Any] = {
        "schema": "sft_h800_packing_platform_v4_interactions_batch1_final_resume_selection/v3",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "selection_policy": "exact ordered set difference after provenance fail-closed stops; no candidate or payload redesign",
        "parent_queue": {**_binding(PARENT_QUEUE), "jobs": 10},
        "completed_success_count": 6,
        "completed": completed,
        "remaining_count": 4,
        "remaining_gpu_job_equivalents": 4,
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
        "parent_resume_design": PARENT_RESUME_DESIGN,
        "final_selection": SELECTION,
        "preparer": Path(__file__).resolve(),
        "freezer": ROOT / "scripts" / "freeze_h800_packing_platform_v4_interactions_batch1_final_resume_v3.py",
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
        "schema": "sft_h800_packing_platform_v4_interactions_batch1_final_resume/v3",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "resume_revision": 3,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_gpu_waiting_for_exact_four_job_final_resume_approval",
        "gpu_training_started": False,
        "continuation_only": True,
        "automatic_next_batch_allowed": False,
        "publication_allowed": False,
        "methodology_invariant": "All four rows are unchanged objects from continuation-v1 and complete only the missing W1 matched pairs.",
        "required_gpu_pool": {
            "gpu_ids": [0, 1, 4, 5, 6, 7],
            "max_gpu_count_per_job": 1,
            "preemption_allowed": False,
            "excluded_busy_gpu_ids": [2, 3],
            "join_busy_pool": True,
        },
        "selection": _binding(SELECTION),
        "queue": {**_binding(QUEUE), "job_count": 4, "gpu_job_equivalents": 4, "ordered_job_ids": list(remaining_ids)},
        "source_bindings": {name: _binding(path) for name, path in sorted(source_files.items())},
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    manifest: dict[str, Any] = {
        "schema": "sft_h800_packing_platform_v4_interactions_batch1_final_resume_queue_manifest/v3",
        "campaign_id": CAMPAIGN_ID,
        "design": _binding(DESIGN),
        "selection": _binding(SELECTION),
        "queue": {**_binding(QUEUE), "job_count": 4, "ordered_job_ids": list(remaining_ids)},
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(QUEUE_MANIFEST, manifest)
    return {"design": _binding(DESIGN), "selection": _binding(SELECTION), "queue": _binding(QUEUE), "completed": 6, "remaining": 4}


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
