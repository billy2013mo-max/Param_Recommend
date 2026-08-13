#!/usr/bin/env python3
"""Freeze and optionally promote the exact GPU-4,5 14B Full calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch


AUTHORIZATION = (
    "用户明确要求继续仅使用 H800 GPU 4、5 运行剩余必要实验；本 approval 只允许执行"
    "新设计的八个 Qwen3-14B Full 两卡边界校准任务，不得占用其他 GPU、抢占外部进程、"
    "复用旧 holdout approval 或扩展到未批准任务。"
)
CAMPAIGN_ID = "h800_qwen3_14b_full_2gpu_boundary_calibration_20260802_v1"
PHASE_ID = "h800_qwen3_14b_full_2gpu_boundary_calibration_v1"
JOB_SCHEMA = "sft_h800_qwen3_14b_full_boundary_calibration_job/v1"
DEFAULT_QUEUE = ROOT / "matrix" / "h800_14b_full_boundary_calibration_jobs_v1.jsonl"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_14b_full_boundary_calibration_design_v1.json"
DEFAULT_QUEUE_MANIFEST = ARTIFACT_DIR / "h800_14b_full_boundary_calibration_queue_manifest_v1.json"


def freeze() -> Path:
    rows = read_jsonl(DEFAULT_QUEUE)
    if (
        len(rows) != 8
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
        or any(int(row.get("gpu_count", 0)) != 2 for row in rows)
    ):
        raise ValueError("queue is not the exact eight-job 14B Full calibration")
    experiment = read_json(ROOT / "config" / "experiment.json")
    scope = experiment["training_scope"]
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("model_ids") != ["qwen3_14b"]
        or scope.get("gpu_ids") != [4, 5]
        or scope.get("exclusive_node_gpu_ids") != [4, 5]
        or scope.get("max_gpu_count") != 2
        or scope.get("gpu_counts") != [2]
    ):
        raise ValueError("experiment config is not the exact GPU-4,5 14B boundary scope")
    hardware = probe_hardware(required_gpu_ids=(4, 5))
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"GPU 4,5 are not an exact-H800 pool: {hardware}")
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(DEFAULT_QUEUE, rows, ROOT)
    provenance_binding = build_provenance_binding(ROOT)
    campaign = read_json(DEFAULT_DESIGN)
    bound_files = {
        DEFAULT_QUEUE,
        DEFAULT_DESIGN,
        DEFAULT_QUEUE_MANIFEST,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "data" / "dataset_info.json",
    }
    for binding in campaign["source_bindings"].values():
        bound_files.add(Path(binding["path"]))
    completion = campaign["source_campaign_completion"]
    bound_files.add(Path(completion["queue_path"]))
    for attempt in completion["attempts"]:
        bound_files.add(Path(attempt["latest_attempt_path"]))
        bound_files.add(Path(attempt["status_path"]))
        bound_files.add(Path(attempt["execution_fingerprint_path"]))
    for row in rows:
        bound_files.add(Path(row["data_path"]))
        bound_files.add(Path(row["dataset_profile_path"]))
    missing = [str(path) for path in bound_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files are absent: {missing}")
    manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=lambda value: str(value))
    }
    ids = [str(row["job_id"]) for row in rows]
    stage = {
        "allowed_job_ids": ids,
        "queue_path": queue_binding["path"],
        "queue_sha256": queue_binding["sha256"],
        "ordered_job_ids": queue_binding["ordered_job_ids"],
        "ordered_job_payload_sha256": queue_binding["ordered_job_payload_sha256"],
        "job_payload_sha256": queue_binding["job_payload_sha256"],
    }
    design = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}",
        "campaign_design": {
            "path": str(DEFAULT_DESIGN.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(DEFAULT_DESIGN),
        },
        "file_sha256": manifest,
        "execution_order": [PHASE_ID],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": [4, 5],
        "max_gpu_count": 2,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": "wait for both GPU 4 and 5 to be idle; never preempt an external process",
        },
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": stage,
    }
    candidate = ARTIFACT_DIR / "approval_design_h800_14b_full_boundary_calibration_v1_candidate.json"
    write_json(candidate, design)
    return candidate


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
    print(json.dumps({"candidate": str(candidate), "sha256": digest, "promotion": report}, ensure_ascii=False, indent=2))
    if report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
