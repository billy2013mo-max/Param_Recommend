#!/usr/bin/env python3
"""Freeze a resource-only continuation for the seven remaining H800 jobs.

The predictor outputs and candidate choices stay bound to the original
prospective design.  This manifest changes only the physical GPU pool after
17 jobs have completed successfully, and proves that the continuation queue
is the exact set difference rather than a post-outcome redesign.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, RESULTS_DIR, ROOT, read_json, read_jsonl, sha256_file, write_json
from prepare_h800_prospective_holdout import probe_hardware
from prepare_h800_prospective_holdout_v2 import build_requirements


SCHEMA = "sft_h800_prospective_holdout_continuation/v2"
CAMPAIGN_ID = "h800_fresh_business_physical_v4b_holdout_20260802_remaining_gpu0_3"
DEFAULT_PARENT_DESIGN = ARTIFACT_DIR / "h800_fresh_holdout_design_v2.json"
DEFAULT_PARENT_APPROVAL_DESIGN = (
    ROOT
    / "runtime"
    / "approval_history"
    / "approval-20260802T103117Z-de389bc9f8eb"
    / "candidate_approval_design.json"
)
DEFAULT_FULL_QUEUE = ROOT / "matrix" / "h800_fresh_holdout_jobs_v2.jsonl"
DEFAULT_REMAINING_QUEUE = ROOT / "matrix" / "h800_fresh_holdout_remaining_0_3_v2.jsonl"
DEFAULT_BUNDLE = ARTIFACT_DIR / "h800_fresh_business_data_bundle_v2.json"
DEFAULT_REQUIREMENTS = ARTIFACT_DIR / "h800_fresh_profile_requirements_continuation_gpu0_3_v2.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_fresh_holdout_continuation_gpu0_3_v2.json"
GPU_IDS = (0, 1, 2, 3)


def _binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _result_bindings(job_ids: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bindings: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for job_id in job_ids:
        result_root = RESULTS_DIR / job_id
        status_path = result_root / "status.json"
        latest_path = result_root / "latest_attempt.json"
        if not status_path.is_file() or not latest_path.is_file():
            raise FileNotFoundError(f"missing durable result for {job_id}")
        status = read_json(status_path)
        latest = read_json(latest_path)
        if status.get("classification") != "success" or latest.get("state") != "complete":
            raise ValueError(f"completed predecessor {job_id} is not a durable success")
        attempt_root = result_root / str(latest["attempt_path"])
        fingerprint_path = attempt_root / "execution_fingerprint.json"
        if not fingerprint_path.is_file():
            raise FileNotFoundError(f"missing execution fingerprint for {job_id}")
        bindings[f"completed_status:{job_id}"] = _binding(status_path)
        bindings[f"completed_latest:{job_id}"] = _binding(latest_path)
        bindings[f"completed_fingerprint:{job_id}"] = _binding(fingerprint_path)
        rows.append(
            {
                "job_id": job_id,
                "classification": "success",
                "status_sha256": sha256_file(status_path),
                "execution_fingerprint_sha256": sha256_file(fingerprint_path),
            }
        )
    return bindings, rows


def build_continuation(
    *,
    parent_design_path: Path,
    parent_approval_design_path: Path,
    full_queue_path: Path,
    remaining_queue_path: Path,
    bundle_path: Path,
) -> dict[str, Any]:
    parent = read_json(parent_design_path)
    parent_approval = read_json(parent_approval_design_path)
    parent_binding = parent_approval.get("campaign_design") or {}
    if parent_binding.get("sha256") != sha256_file(parent_design_path):
        raise ValueError("historical approval does not bind the unchanged parent design")
    full = read_jsonl(full_queue_path)
    remaining = read_jsonl(remaining_queue_path)
    full_by_id = {str(row["job_id"]): row for row in full}
    remaining_ids = [str(row["job_id"]) for row in remaining]
    if len(full_by_id) != 24 or len(remaining_ids) != 7 or len(set(remaining_ids)) != 7:
        raise ValueError("expected an exact 24-job parent and seven unique remaining jobs")
    if any(job_id not in full_by_id for job_id in remaining_ids):
        raise ValueError("remaining queue contains a job outside the parent queue")
    if any(int(row["gpu_count"]) != 4 for row in remaining):
        raise ValueError("resource continuation accepts only the seven four-GPU jobs")
    completed_ids = [job_id for job_id in full_by_id if job_id not in set(remaining_ids)]
    if len(completed_ids) != 17:
        raise ValueError("remaining queue is not the exact 17-success/7-pending partition")
    result_bindings, completed_rows = _result_bindings(completed_ids)
    experiment = read_json(ROOT / "config" / "experiment.json")
    scope = experiment["training_scope"]
    if scope.get("gpu_ids") != list(GPU_IDS) or scope.get("exclusive_node_gpu_ids") != list(GPU_IDS):
        raise ValueError("live experiment config is not scoped exactly to GPU 0-3")
    parent_slots = {
        str(row["candidate_slot_id"]): row for row in parent["candidate_slots"]
    }
    remaining_slots = [parent_slots[str(row["candidate_slot_id"])] for row in remaining]
    common_bindings = {
        "parent_prospective_design": _binding(parent_design_path),
        "parent_promoted_approval_design": _binding(parent_approval_design_path),
        "parent_full_queue": _binding(full_queue_path),
        "remaining_queue": _binding(remaining_queue_path),
        "experiment_config": _binding(ROOT / "config" / "experiment.json"),
        "hardware_config": _binding(ROOT / "config" / "hardware.json"),
        "frozen_predictions": _binding(ARTIFACT_DIR / "h800_frozen_predictions_before_fresh_holdout_v2.json"),
        "business_data_bundle": _binding(bundle_path),
        "campaign_gate": _binding(ROOT / "scripts" / "check_h800_campaign_gate.py"),
        "approval_freezer": _binding(ROOT / "scripts" / "freeze_h800_prospective_live_approval_v2.py"),
        "scheduler": _binding(ROOT / "scripts" / "scheduler.py"),
        "run_job": _binding(ROOT / "scripts" / "run_job.py"),
        "continuation_implementation": _binding(Path(__file__).resolve()),
    }
    return {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "queues_mutated": False,
        "materialization_allowed": True,
        "publication_allowed": False,
        "continuation_only": True,
        "methodology_invariant": (
            "candidate selection, memory admission and v4b scores are copied byte-for-byte "
            "from the pre-outcome parent design; only the physical GPU IDs change"
        ),
        "predecessor": {
            "campaign_id": parent["campaign_id"],
            "design": _binding(parent_design_path),
            "historical_promoted_approval_design": _binding(parent_approval_design_path),
            "completed_success_count": len(completed_rows),
            "remaining_count": len(remaining),
            "completed_results": completed_rows,
        },
        "required_gpu_pool": {
            "gpu_ids": list(GPU_IDS),
            "expected_name_contains": "H800",
            "availability_policy": "exact_idle_pool",
            "join_busy_pool_allowed": False,
            "hardware_probe_at_design": probe_hardware(required_gpu_ids=GPU_IDS),
        },
        "acceptance_contract": parent["acceptance_contract"],
        "frozen_bindings": {**common_bindings, **result_bindings},
        "scenarios": parent["scenarios"],
        "candidate_slots": remaining_slots,
        "candidate_count": len(remaining_slots),
        "remaining_job_ids": remaining_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-design", type=Path, default=DEFAULT_PARENT_DESIGN)
    parser.add_argument("--parent-approval-design", type=Path, default=DEFAULT_PARENT_APPROVAL_DESIGN)
    parser.add_argument("--full-queue", type=Path, default=DEFAULT_FULL_QUEUE)
    parser.add_argument("--remaining-queue", type=Path, default=DEFAULT_REMAINING_QUEUE)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    design = build_continuation(
        parent_design_path=args.parent_design,
        parent_approval_design_path=args.parent_approval_design,
        full_queue_path=args.full_queue,
        remaining_queue_path=args.remaining_queue,
        bundle_path=args.bundle,
    )
    write_json(args.output, design)
    requirements = build_requirements(design, args.output, read_json(args.bundle))
    write_json(args.requirements, requirements)
    print(
        f"wrote {args.output}; completed_success=17; remaining={design['candidate_count']}; "
        "resource_pool=0,1,2,3"
    )


if __name__ == "__main__":
    main()
