#!/usr/bin/env python3
"""Freeze and promote the evidence-bound unfinished jobs of interrupted Phase 1."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from approval_gate import (
    build_provenance_binding,
    build_queue_binding,
    canonical_job_sha256,
)
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
from evaluate_h800_packing_final_virtual_mbs_v1 import _job_result
from freeze_h800_packing_final_virtual_mbs_v1 import _validate
from prepare_h800_packing_final_virtual_mbs_v1 import (
    CAMPAIGN_ID,
    DATASET_INFO,
    DESIGN,
    EXPERIMENT,
    GPU_COUNTS,
    GPU_IDS,
    MODEL_INVENTORY,
    PHASE_ID,
    QUEUE,
    STATIC,
)
from promote_approval_candidate import promote_candidate
from prepare_h800_prospective_holdout import probe_hardware
from run_job import live_runtime_identity, live_runtime_patch


EXPECTED_COMPLETED_JOBS = 22
EXPECTED_REMAINING_JOBS = 14
CONTINUATION_QUEUE = (
    MATRIX_DIR / "h800_packing_final_virtual_mbs_continuation_v1.jsonl"
)
CONTINUATION_DESIGN = (
    ARTIFACT_DIR / "h800_packing_final_virtual_mbs_continuation_design_v1.json"
)
CANDIDATE = (
    ARTIFACT_DIR
    / "approval_design_h800_packing_final_virtual_mbs_continuation_v1_candidate.json"
)
SCRIPTS = ROOT / "scripts"
STAGING = ROOT / "packing_final_4to7gpu_staging"
AUTHORIZATION = (
    "用户在2026-08-11明确要求使用目前空闲的卡运行未完成Packing任务。"
    "当前36任务中22个已有完整成功证据。"
    "本批准只允许运行其余14个任务，并允许4至7卡任务使用当前空闲GPU组成的mask；"
    "外部忙卡不得抢占，任务之间保持严格串行，选中mask必须在启动时空闲。"
    "不得重跑已完成22项，不得自动补第二重复、启动Phase 2/4或发布模型。"
)


def _available_pool_hardware() -> dict[str, Any]:
    hardware = probe_hardware(required_gpu_ids=list(GPU_IDS))
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError("GPU 0-7 are not an exact eight-H800 pool")
    busy_uuids = {
        str(row.get("gpu_uuid"))
        for row in hardware.get("selected_gpu_compute_processes", [])
    }
    idle_gpu_ids = [
        int(row["index"])
        for row in hardware.get("selected_gpu_rows", [])
        if str(row.get("uuid")) not in busy_uuids
    ]
    if len(idle_gpu_ids) < 4:
        raise RuntimeError(f"fewer than four H800 GPUs are idle: {idle_gpu_ids}")
    hardware.update(
        {
            "idle_gpu_ids_at_approval": idle_gpu_ids,
            "minimum_idle_gpu_count_to_start": 4,
            "external_busy_gpus_allowed": True,
            "join_busy_pool_required": True,
            "preemption_allowed": False,
            "selected_mask_rechecked_immediately_before_each_job": True,
        }
    )
    return hardware


def _completed_evidence(row: dict[str, Any]) -> dict[str, Any] | None:
    result = _job_result(row)
    if result.get("all_passed") is not True:
        return None
    root = RESULTS_DIR / str(row["job_id"])
    latest_path = root / "latest_attempt.json"
    latest = read_json(latest_path)
    attempt_id = str(latest.get("execution_attempt_id") or "")
    attempt_root = root / "attempts" / attempt_id
    status_path = attempt_root / "status.json"
    inputs_path = attempt_root / "execution_inputs.json"
    status = read_json(status_path)
    inputs = read_json(inputs_path)
    expected_payload = canonical_job_sha256(row)
    checks = {
        "latest_complete_success": latest.get("state") == "complete"
        and latest.get("classification") == "success",
        "status_success": status.get("classification") == "success",
        "calibration_eligible": status.get("calibration_eligible") is True,
        "job_snapshot_exact": inputs.get("job_snapshot") == row,
        "job_payload_exact": inputs.get("job_payload_sha256") == expected_payload,
        "phase_evaluator_all_passed": result.get("all_passed") is True,
    }
    if not all(checks.values()):
        raise ValueError(f"completed-prefix evidence drifted for {row['job_id']}: {checks}")
    return {
        "job_id": row["job_id"],
        "execution_attempt_id": attempt_id,
        "job_payload_sha256": expected_payload,
        "latest_attempt_path": str(latest_path.resolve().relative_to(ROOT.resolve())),
        "latest_attempt_sha256": sha256_file(latest_path),
        "status_path": str(status_path.resolve().relative_to(ROOT.resolve())),
        "status_sha256": sha256_file(status_path),
        "execution_inputs_path": str(inputs_path.resolve().relative_to(ROOT.resolve())),
        "execution_inputs_sha256": sha256_file(inputs_path),
        "checks": checks,
        "phase_result": result,
    }


def _split_queue() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(QUEUE)
    _validate(rows)
    completed: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for row in rows:
        item = _completed_evidence(row)
        if item is None:
            remaining.append(row)
        else:
            completed.append(row)
            evidence.append(item)
    if len(completed) != EXPECTED_COMPLETED_JOBS:
        raise ValueError(
            f"expected exactly {EXPECTED_COMPLETED_JOBS} successful jobs, got {len(completed)}"
        )
    if len(remaining) != EXPECTED_REMAINING_JOBS:
        raise ValueError(f"expected {EXPECTED_REMAINING_JOBS} remaining jobs, got {len(remaining)}")
    return completed, remaining, evidence


def freeze() -> Path:
    completed, remaining, evidence = _split_queue()
    write_jsonl(CONTINUATION_QUEUE, remaining)

    dataset_registry = read_json(DATASET_INFO)
    for row in remaining:
        dataset_id = str(row["dataset_id"])
        entry = dataset_registry.get(dataset_id)
        if not isinstance(entry, dict):
            raise ValueError(f"runtime dataset is not registered: {dataset_id}")
        registered_path = DATASET_INFO.parent / str(entry.get("file_name") or "")
        if (
            registered_path.resolve() != Path(str(row["data_path"])).resolve()
            or not registered_path.is_file()
            or sha256_file(registered_path) != row["data_sha256"]
        ):
            raise ValueError(f"runtime dataset registration drifted: {dataset_id}")

    base_rows = read_jsonl(QUEUE)
    base_design = read_json(DESIGN)
    if (
        base_design.get("campaign_id") != CAMPAIGN_ID
        or base_design.get("phase_id") != PHASE_ID
        or base_design.get("ordered_job_ids")
        != [str(row["job_id"]) for row in base_rows]
        or base_design.get("bindings", {}).get("queue", {}).get("sha256")
        != sha256_file(QUEUE)
    ):
        raise ValueError("base Phase 1 design is stale")

    live = read_json(ROOT / "config" / "experiment.json")
    scope = live.get("training_scope") or {}
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("gpu_ids") != list(GPU_IDS)
        or scope.get("exclusive_node_gpu_ids") != list(GPU_IDS)
        or scope.get("gpu_counts") != list(GPU_COUNTS)
        or int(scope.get("max_gpu_count", 0)) != 7
        or live.get("measurement", {}).get("performance_parallelism")
        != "disjoint_gpu_masks"
    ):
        raise ValueError("live experiment config is not the exact Phase 1 scope")

    continuation_design: dict[str, Any] = {
        "schema": "sft_h800_packing_final_virtual_mbs_continuation_design/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "reason": "user authorized remaining PF05/PF06 tasks to use currently idle GPUs",
        "base_queue": {
            "path": str(QUEUE.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(QUEUE),
            "jobs": len(completed) + len(remaining),
        },
        "completed_job_ids": [str(row["job_id"]) for row in completed],
        "completed_evidence": evidence,
        "continuation_queue": {
            "path": str(CONTINUATION_QUEUE.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(CONTINUATION_QUEUE),
            "ordered_job_ids": [str(row["job_id"]) for row in remaining],
            "ordered_job_payload_sha256": sha256_json(remaining),
            "jobs": len(remaining),
        },
        "must_not_rerun_completed_jobs": True,
        "available_pool_large_jobs_authorized": True,
        "external_busy_gpus_must_not_be_preempted": True,
        "automatic_repeat_expansion_allowed": False,
        "phase2_or_phase4_launch_allowed": False,
        "publication_allowed": False,
    }
    continuation_design["report_sha256"] = sha256_json(continuation_design)
    write_json(CONTINUATION_DESIGN, continuation_design)

    hardware = _available_pool_hardware()
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")

    queue_binding = build_queue_binding(CONTINUATION_QUEUE, remaining, ROOT)
    continuation_script = Path(__file__).resolve()
    continuation_launcher = (
        STAGING / "launch_h800_packing_final_virtual_mbs_continuation_v1.py"
    )
    scoped_sources = [
        SCRIPTS / "approval_gate.py",
        SCRIPTS / "common.py",
        SCRIPTS / "metrics_callback.py",
        SCRIPTS / "promote_approval_candidate.py",
        SCRIPTS / "run_job.py",
        SCRIPTS / "runtime_evidence.py",
        SCRIPTS / "scheduler.py",
        SCRIPTS / "train_entry.py",
        SCRIPTS / "prepare_h800_packing_final_virtual_mbs_v1.py",
        SCRIPTS / "freeze_h800_packing_final_virtual_mbs_v1.py",
        SCRIPTS / "evaluate_h800_packing_final_virtual_mbs_v1.py",
        continuation_script,
        STAGING / "launch_h800_packing_final_virtual_mbs_v1.py",
        continuation_launcher,
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "hardware.json",
        ROOT / "config" / "models.json",
        ROOT / "config" / "deepspeed" / "ds_z3.json",
    ]
    provenance_binding = build_provenance_binding(ROOT, source_paths=scoped_sources)
    bound_files: set[Path] = {
        QUEUE,
        DESIGN,
        CONTINUATION_QUEUE,
        CONTINUATION_DESIGN,
        EXPERIMENT,
        STATIC,
        DATASET_INFO,
        MODEL_INVENTORY,
        ARTIFACT_DIR / "provenance.json",
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "hardware.json",
        ROOT / "config" / "models.json",
        *scoped_sources,
    }
    for row in remaining:
        bound_files.add(Path(str(row["data_path"])))
        bound_files.add(Path(str(row["dataset_profile_path"])))
    for item in evidence:
        for key in ("latest_attempt_path", "status_path", "execution_inputs_path"):
            bound_files.add(ROOT / str(item[key]))
    missing = [str(path) for path in bound_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"approval-bound paths are absent: {missing}")
    file_manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=str)
        if path.is_file() and path.resolve().is_relative_to(ROOT.resolve())
    }
    approval_design = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}:available_pool_continuation_after_22",
        "authorization": AUTHORIZATION,
        "campaign_design": {
            "path": str(CONTINUATION_DESIGN.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(CONTINUATION_DESIGN),
            "report_sha256": continuation_design["report_sha256"],
        },
        "file_sha256": file_manifest,
        "execution_order": [f"{PHASE_ID}_available_pool_continuation_after_22"],
        "allowed_job_ids": queue_binding["ordered_job_ids"],
        "authorized_gpu_ids": list(GPU_IDS),
        "max_gpu_count": 7,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": "strict serial continuation on currently idle masks; external busy GPUs are never preempted",
        },
        "oom_policy": "unexpected_fit_measurement_failure_no_automatic_retry",
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
        "completed_evidence": evidence,
    }
    write_json(CANDIDATE, approval_design)
    return CANDIDATE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = freeze()
    digest = sha256_file(candidate)
    report = promote_candidate(
        candidate_path=candidate,
        expected_candidate_sha256=digest,
        project_root=ROOT,
        authorization=AUTHORIZATION,
        approved_by="user",
        promote=args.promote,
    )
    print(
        json.dumps(
            {
                "candidate": str(candidate),
                "sha256": digest,
                "completed_jobs": EXPECTED_COMPLETED_JOBS,
                "remaining_jobs": EXPECTED_REMAINING_JOBS,
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
