#!/usr/bin/env python3
"""Freeze the ten untouched 32B jobs after the invalid packing job stopped resume 1."""

from __future__ import annotations

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
    write_json,
    write_jsonl,
)
from prepare_h800_final_memory_business_blind_v2 import implementation as prep

CAMPAIGN_ID = "h800_final_memory_business_blind_resume2_20260810_v2"
PHASE_ID = "h800_final_memory_business_blind_resume2_v2"
ORIGINAL_QUEUE = prep.DEFAULT_QUEUE
ORIGINAL_DESIGN = prep.DEFAULT_DESIGN
ORIGINAL_PREDICTIONS = prep.DEFAULT_PREDICTIONS
PRIOR_RESUME_DESIGN = ARTIFACT_DIR / "h800_final_memory_business_blind_resume_design_v2.json"
SCHEDULER_EVENTS = ROOT / "runtime" / "scheduler_events.jsonl"
DEFAULT_QUEUE = MATRIX_DIR / "h800_final_memory_business_blind_resume2_jobs_v2.jsonl"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_final_memory_business_blind_resume2_design_v2.json"
STAGING_DIR = ROOT / "final_memory_business_blind_resume2_v2_staging"
DEFAULT_EXPERIMENT = STAGING_DIR / "experiment.h800_final_memory_business_blind_resume2_v2.json"
EXPECTED_RESUME = 10
SCENARIOS = {"upper_tail_full", "upper_tail_lora"}


def _attempt_state(job_id: str) -> dict[str, Any]:
    root = ROOT / "results" / job_id
    status = root / "status.json"
    latest = root / "latest_attempt.json"
    attempts = root / "attempts"
    attempt_entries = list(attempts.iterdir()) if attempts.is_dir() else []
    return {
        "result_root_exists": root.exists(),
        "status_exists": status.is_file(),
        "latest_attempt_exists": latest.is_file(),
        "attempt_entries": [str(path) for path in attempt_entries],
    }


def main() -> None:
    outputs = (DEFAULT_QUEUE, DEFAULT_DESIGN, DEFAULT_EXPERIMENT)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite frozen resume-2 outputs: {existing}")
    original = read_jsonl(ORIGINAL_QUEUE)
    selected = [dict(row) for row in original if str(row.get("scenario")) in SCENARIOS]
    if len(selected) != EXPECTED_RESUME or len({str(row["job_id"]) for row in selected}) != EXPECTED_RESUME:
        raise ValueError("the untouched 32B subset drifted")
    states = {str(row["job_id"]): _attempt_state(str(row["job_id"])) for row in selected}
    touched = {
        job_id: state
        for job_id, state in states.items()
        if state["status_exists"] or state["latest_attempt_exists"] or state["attempt_entries"]
    }
    if touched:
        raise ValueError(f"resume-2 jobs are no longer untouched: {touched}")
    if not all(
        int(row["gpu_count"]) == 4
        and int(row["zero_stage"]) == 3
        and not bool(row["packing"])
        and str(row["model_id"]) == "qwen3_32b"
        for row in selected
    ):
        raise ValueError("resume-2 escaped the expected 32B/4GPU/ZeRO-3 mechanism")

    write_jsonl(DEFAULT_QUEUE, selected)
    experiment = read_json(prep.DEFAULT_EXPERIMENT)
    experiment["training_scope"] = {
        **dict(experiment["training_scope"]),
        "phase_id": PHASE_ID,
        "model_ids": ["qwen3_32b"],
        "gpu_counts": [4],
        "zero_by_gpu_count": {"4": ["zero3"]},
        "objective": "resume the ten untouched 32B jobs from the frozen final business-blind queue",
    }
    write_json(DEFAULT_EXPERIMENT, experiment)
    design: dict[str, Any] = {
        "schema": "sft_h800_final_memory_business_blind_resume2_design/v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "status": "frozen_exact_payload_resume_after_invalid_packing_launcher_failure",
        "gpu_training_started_for_resume2": False,
        "new_gpu_outcomes_observed": 0,
        "production_model_mutated": False,
        "authorized_gpu_ids": list(prep.GPU_IDS),
        "resume_jobs": EXPECTED_RESUME,
        "resume_job_ids": [str(row["job_id"]) for row in selected],
        "resume_ordered_payload_sha256": sha256_json(selected),
        "payload_policy": "exact rows copied from the pre-outcome frozen v2 queue; predictions and job mechanisms are unchanged",
        "selection_reason": "the only untouched executor-valid rows after resume 1 stopped at an invalid physical-MBS packing row",
        "bindings": {
            "original_queue": {"path": str(ORIGINAL_QUEUE.resolve()), "sha256": sha256_file(ORIGINAL_QUEUE)},
            "original_design": {"path": str(ORIGINAL_DESIGN.resolve()), "sha256": sha256_file(ORIGINAL_DESIGN)},
            "original_predictions": {"path": str(ORIGINAL_PREDICTIONS.resolve()), "sha256": sha256_file(ORIGINAL_PREDICTIONS)},
            "prior_resume_design": {"path": str(PRIOR_RESUME_DESIGN.resolve()), "sha256": sha256_file(PRIOR_RESUME_DESIGN)},
            "scheduler_events": {"path": str(SCHEDULER_EVENTS.resolve()), "sha256": sha256_file(SCHEDULER_EVENTS)},
            "resume_queue": {"path": str(DEFAULT_QUEUE.resolve()), "sha256": sha256_file(DEFAULT_QUEUE)},
            "experiment": {"path": str(DEFAULT_EXPERIMENT.resolve()), "sha256": sha256_file(DEFAULT_EXPERIMENT)},
        },
        "preparation_states": states,
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DEFAULT_DESIGN, design)


if __name__ == "__main__":
    main()
