#!/usr/bin/env python3
"""Freeze the exact pending-valid subset after the ZeRO-0 launcher failure."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json, write_jsonl
from prepare_h800_final_memory_business_blind_v2 import implementation as prep

CAMPAIGN_ID = "h800_final_memory_business_blind_resume_20260810_v2"
PHASE_ID = "h800_final_memory_business_blind_resume_v2"
ORIGINAL_QUEUE = prep.DEFAULT_QUEUE
ORIGINAL_DESIGN = prep.DEFAULT_DESIGN
ORIGINAL_PREDICTIONS = prep.DEFAULT_PREDICTIONS
SCHEDULER_EVENTS = ROOT / "runtime" / "scheduler_events.jsonl"
DEFAULT_QUEUE = MATRIX_DIR / "h800_final_memory_business_blind_resume_jobs_v2.jsonl"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_final_memory_business_blind_resume_design_v2.json"
STAGING_DIR = ROOT / "final_memory_business_blind_resume_v2_staging"
DEFAULT_EXPERIMENT = STAGING_DIR / "experiment.h800_final_memory_business_blind_resume_v2.json"
EXPECTED_COMPLETED = 20
EXPECTED_INVALID = 5
EXPECTED_RESUME = 35


def _terminal_binding(job_id: str) -> dict[str, Any] | None:
    path = ROOT / "results" / job_id / "status.json"
    if not path.is_file():
        return None
    status = read_json(path)
    if status.get("classification") not in {"success", "oom"}:
        raise ValueError(f"unexpected terminal classification in original run: {job_id}")
    return {"job_id": job_id, "classification": status["classification"], "path": str(path.resolve()), "sha256": sha256_file(path)}


def main() -> None:
    outputs = (DEFAULT_QUEUE, DEFAULT_DESIGN, DEFAULT_EXPERIMENT)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite frozen resume outputs: {existing}")
    original = read_jsonl(ORIGINAL_QUEUE)
    if len(original) != prep.EXPECTED_JOBS:
        raise ValueError("original 60-job queue drifted")
    completed = []
    invalid = []
    resume = []
    for row in original:
        terminal = _terminal_binding(str(row["job_id"]))
        incompatible = int(row["gpu_count"]) > 1 and int(row["zero_stage"]) not in {2, 3}
        if terminal is not None:
            if incompatible:
                raise ValueError("an executor-incompatible job unexpectedly has a terminal label")
            completed.append(terminal)
        elif incompatible:
            invalid.append({
                "job_id": str(row["job_id"]),
                "scenario": str(row["scenario"]),
                "gpu_count": int(row["gpu_count"]),
                "zero_stage": int(row["zero_stage"]),
                "reason": "run_job requires ZeRO-2 or ZeRO-3 for multi-card jobs",
            })
        else:
            if (ROOT / "results" / str(row["job_id"])).exists():
                raise ValueError(f"nonterminal resume job has an existing result directory: {row['job_id']}")
            resume.append(dict(row))
    if (len(completed), len(invalid), len(resume)) != (EXPECTED_COMPLETED, EXPECTED_INVALID, EXPECTED_RESUME):
        raise ValueError(f"resume partition drifted: {len(completed)}, {len(invalid)}, {len(resume)}")
    if {row["scenario"] for row in invalid} != {"long_packed_qwen35_lora"}:
        raise ValueError("executor-incompatible set escaped the one known scenario")
    write_jsonl(DEFAULT_QUEUE, resume)
    experiment = read_json(prep.DEFAULT_EXPERIMENT)
    experiment["training_scope"] = {
        **dict(experiment["training_scope"]),
        "phase_id": PHASE_ID,
        "objective": "resume the 35 untouched valid jobs from the frozen final business-blind queue",
    }
    write_json(DEFAULT_EXPERIMENT, experiment)
    design: dict[str, Any] = {
        "schema": "sft_h800_final_memory_business_blind_resume_design/v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "status": "frozen_exact_payload_resume_after_launcher_validation_failure",
        "gpu_training_started_for_resume": False,
        "new_gpu_outcomes_observed": 0,
        "production_model_mutated": False,
        "authorized_gpu_ids": list(prep.GPU_IDS),
        "completed_original_jobs": completed,
        "excluded_executor_incompatible_jobs": invalid,
        "resume_jobs": EXPECTED_RESUME,
        "resume_job_ids": [str(row["job_id"]) for row in resume],
        "resume_ordered_payload_sha256": sha256_json(resume),
        "payload_policy": "byte-identical rows copied from the pre-outcome frozen v2 queue; no prediction or configuration changes",
        "bindings": {
            "original_queue": {"path": str(ORIGINAL_QUEUE.resolve()), "sha256": sha256_file(ORIGINAL_QUEUE)},
            "original_design": {"path": str(ORIGINAL_DESIGN.resolve()), "sha256": sha256_file(ORIGINAL_DESIGN)},
            "original_predictions": {"path": str(ORIGINAL_PREDICTIONS.resolve()), "sha256": sha256_file(ORIGINAL_PREDICTIONS)},
            "scheduler_events": {"path": str(SCHEDULER_EVENTS.resolve()), "sha256": sha256_file(SCHEDULER_EVENTS)},
            "resume_queue": {"path": str(DEFAULT_QUEUE.resolve()), "sha256": sha256_file(DEFAULT_QUEUE)},
            "experiment": {"path": str(DEFAULT_EXPERIMENT.resolve()), "sha256": sha256_file(DEFAULT_EXPERIMENT)},
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DEFAULT_DESIGN, design)


if __name__ == "__main__":
    main()
