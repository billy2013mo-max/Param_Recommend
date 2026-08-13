#!/usr/bin/env python3
"""Freeze the VL-only continuation after transparent Packing counter repair."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, RESULTS_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json


CAMPAIGN_ID = "h800_packing_vl_canary_20260803_v1"
PHASE_ID = "h800_packing_vl_canary_v1"
V1_DESIGN = ARTIFACT_DIR / "h800_packing_vl_canary_design_v1.json"
V1_ACCEPTANCE = ARTIFACT_DIR / "h800_packing_semantic_canary_acceptance_v1.json"
V2_CORRECTION = ARTIFACT_DIR / "h800_packing_semantic_canary_acceptance_v2_prefetch_corrected.json"
VL_QUEUE = MATRIX_DIR / "h800_vl_media_canary_v1.jsonl"
OUTPUT = ARTIFACT_DIR / "h800_packing_vl_canary_continuation_design_v2.json"


def _binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def main() -> None:
    design = read_json(V1_DESIGN)
    v1 = read_json(V1_ACCEPTANCE)
    correction = read_json(V2_CORRECTION)
    vl_jobs = read_jsonl(VL_QUEUE)
    if (
        design.get("campaign_id") != CAMPAIGN_ID
        or v1.get("all_passed") is not False
        or correction.get("all_passed") is not True
        or len(vl_jobs) != 2
        or any(job.get("campaign_id") != CAMPAIGN_ID for job in vl_jobs)
    ):
        raise ValueError("continuation sources do not satisfy the exact Packing-to-VL transition")
    packing_results = []
    for job_id in design["stages"]["packing"]["job_ids"]:
        status_path = RESULTS_DIR / job_id / "status.json"
        status = read_json(status_path)
        if status.get("classification") != "success" or status.get("calibration_eligible") is not True:
            raise ValueError(f"Packing result is not successful: {status}")
        packing_results.append(
            {
                "job_id": job_id,
                "status": _binding(status_path),
                "execution_attempt_id": status["execution_attempt_id"],
                "execution_fingerprint_sha256": status["execution_fingerprint_sha256"],
            }
        )
    source_files = {
        "v1_design": V1_DESIGN,
        "v1_packing_acceptance": V1_ACCEPTANCE,
        "v2_prefetch_correction": V2_CORRECTION,
        "vl_queue": VL_QUEUE,
        "frozen_decisions": ARTIFACT_DIR / "h800_packing_vl_canary_frozen_decisions_v1.json",
        "model_inventory": ARTIFACT_DIR / "h800_packing_vl_canary_model_inventory_v1.json",
        "vl_media_manifest": ARTIFACT_DIR / "h800_vl_canary_media_manifest_v1.json",
        "vl_data": ROOT / "data" / "packing_vl_canary_v1" / "vl_pzfj38_canary_v1.jsonl",
        "dataset_registry": ROOT / "data" / "dataset_info.json",
        "experiment_config": ROOT / "config" / "experiment.json",
        "prefetch_evaluator": ROOT / "scripts" / "evaluate_h800_packing_prefetch_correction_v2.py",
        "continuation_preparer": Path(__file__).resolve(),
        "continuation_freezer": ROOT / "scripts" / "freeze_h800_packing_vl_continuation_v2.py",
        "vl_evaluator": ROOT / "scripts" / "evaluate_h800_packing_vl_canary_v1.py",
        "run_job": ROOT / "scripts" / "run_job.py",
        "scheduler": ROOT / "scripts" / "scheduler.py",
        "train_entry": ROOT / "scripts" / "train_entry.py",
        "metrics_callback": ROOT / "scripts" / "metrics_callback.py",
        "model_structure_manifest": ROOT / "scripts" / "model_structure_manifest.py",
    }
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_vl_canary_continuation_design/v2",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "packing_executed_v1_counter_defect_transparently_corrected_vl_not_started",
        "packing_gpu_training_started": True,
        "vl_gpu_training_started": False,
        "remaining_execution_order": ["vl"],
        "v1_design": _binding(V1_DESIGN),
        "v1_acceptance": _binding(V1_ACCEPTANCE),
        "v2_prefetch_correction": _binding(V2_CORRECTION),
        "packing_results": packing_results,
        "vl_queue": {**_binding(VL_QUEUE), "job_ids": [job["job_id"] for job in vl_jobs]},
        "source_bindings": {name: _binding(path) for name, path in sorted(source_files.items())},
        "required_gpu_pool": {"gpu_ids": [0, 1], "max_gpu_count": 2, "preemption_allowed": False},
        "correction_policy": {
            "threshold_changed": False,
            "model_changed": False,
            "queue_changed": False,
            "packing_rerun": False,
            "only_remove_collated_but_unconsumed_prefetch": True,
        },
        "publication_allowed": False,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256_file(OUTPUT), "report_sha256": report["report_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
