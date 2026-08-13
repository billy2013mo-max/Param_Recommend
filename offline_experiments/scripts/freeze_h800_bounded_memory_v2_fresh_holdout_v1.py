#!/usr/bin/env python3
"""Freeze and optionally promote one exact stage of the bounded-memory v2 holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from approval_gate import build_provenance_binding, build_queue_binding
from common import (
    ARTIFACT_DIR,
    RESULTS_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch


AUTHORIZATION = (
    "用户在 2026-08-03 明确要求改用 H800 GPU 0、1，并等待 GPU 0 上的推理结束后 "
    "执行剩余必要实验的授权。本 approval 仅允许执行冻结的新 S3 数据源 bounded-memory v2 "
    "prospective holdout 当前阶段；不得占用其他 GPU、抢占外部进程、修改 challenger、"
    "复用旧 approval 或扩展任务。"
)
CAMPAIGN_ID = "h800_bounded_memory_v2_fresh_holdout_20260803_v1"
PHASE_ID = "h800_bounded_memory_v2_fresh_holdout_v1"
JOB_SCHEMA = "sft_h800_bounded_memory_v2_fresh_holdout_job/v1"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_bounded_memory_v2_fresh_holdout_design_v1.json"
DEFAULT_QUEUE_MANIFEST = ARTIFACT_DIR / "h800_bounded_memory_v2_fresh_holdout_queue_manifest_v1.json"
DEFAULT_PREDICTIONS = ARTIFACT_DIR / "h800_frozen_predictions_before_bounded_memory_v2_fresh_holdout_v1.json"
DEFAULT_CHALLENGER = ARTIFACT_DIR / "h800_bounded_memory_challenger_v2.json"
DEFAULT_INVENTORY = ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_v1.json"
STAGES = {
    "canary": {
        "queue": ROOT / "matrix" / "h800_bounded_memory_v2_fresh_holdout_canary_v1.jsonl",
        "expected_jobs": 1,
        "require_canary": True,
    },
    "formal": {
        "queue": ROOT / "matrix" / "h800_bounded_memory_v2_fresh_holdout_formal_v1.jsonl",
        "expected_jobs": 20,
        "require_canary": False,
    },
}


def _canary_success(manifest: dict[str, Any]) -> dict[str, Any]:
    job_id = str(manifest["software_canary_job_id"])
    status_path = RESULTS_DIR / job_id / "status.json"
    if not status_path.is_file():
        raise RuntimeError(f"formal stage remains locked: canary status absent: {status_path}")
    status = read_json(status_path)
    if (
        status.get("job_id") != job_id
        or status.get("classification") != "success"
        or status.get("calibration_eligible") is not True
        or status.get("execution_fingerprint_quality") != "complete"
    ):
        raise RuntimeError(f"formal stage remains locked: canary did not pass: {status}")
    return {
        "job_id": job_id,
        "status_path": str(status_path.resolve().relative_to(ROOT.resolve())),
        "status_sha256": sha256_file(status_path),
        "execution_attempt_id": status.get("execution_attempt_id"),
        "classification": status.get("classification"),
        "calibration_eligible": status.get("calibration_eligible"),
        "execution_fingerprint_sha256": status.get("execution_fingerprint_sha256"),
    }


def freeze(stage_name: str) -> Path:
    stage_spec = STAGES[stage_name]
    queue_path = Path(stage_spec["queue"])
    rows = read_jsonl(queue_path)
    expected_jobs = int(stage_spec["expected_jobs"])
    if (
        len(rows) != expected_jobs
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
        or any(int(row.get("gpu_count", 0)) not in {1, 2} for row in rows)
        or any((row.get("calibration_partition") or {}).get("role") != "holdout" for row in rows)
        or any(Path(row.get("declared_model_manifest_path", "")).resolve() != DEFAULT_INVENTORY.resolve() for row in rows)
        or any(row.get("declared_model_manifest_sha256") != sha256_file(DEFAULT_INVENTORY) for row in rows)
    ):
        raise ValueError(f"queue is not the exact frozen {stage_name} stage")
    canary_flags = sum(row.get("software_canary") is True for row in rows)
    if canary_flags != (1 if stage_spec["require_canary"] else 0):
        raise ValueError(f"{stage_name} queue has an invalid software-canary composition")

    design = read_json(DEFAULT_DESIGN)
    manifest = read_json(DEFAULT_QUEUE_MANIFEST)
    predictions = read_json(DEFAULT_PREDICTIONS)
    challenger = read_json(DEFAULT_CHALLENGER)
    staged_manifest = (manifest.get("staged_queues") or {}).get(stage_name) or {}
    if (
        design.get("campaign_id") != CAMPAIGN_ID
        or design.get("phase_id") != PHASE_ID
        or design.get("gpu_training_started") is not False
        or design.get("execution_authorized") is not False
        or design.get("frozen_prediction_binding", {}).get("sha256") != sha256_file(DEFAULT_PREDICTIONS)
        or design.get("frozen_prediction_binding", {}).get("report_sha256") != predictions.get("report_sha256")
        or challenger.get("publishable") is not False
        or manifest.get("design", {}).get("sha256") != sha256_file(DEFAULT_DESIGN)
        or staged_manifest.get("sha256") != sha256_file(queue_path)
        or staged_manifest.get("job_count") != expected_jobs
    ):
        raise ValueError("fresh holdout design, prediction or staged queue binding drifted")

    experiment = read_json(ROOT / "config" / "experiment.json")
    scope = experiment["training_scope"]
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("model_ids") != ["qwen3_8b", "qwen3_14b", "qwen3p5_4b"]
        or scope.get("gpu_ids") != [0, 1]
        or scope.get("exclusive_node_gpu_ids") != [0, 1]
        or scope.get("max_gpu_count") != 2
        or scope.get("gpu_counts") != [1, 2]
        or scope.get("zero_by_gpu_count") != {"1": ["none"], "2": ["zero2", "zero3"]}
    ):
        raise ValueError("experiment config is not the exact GPU-0,1 fresh holdout scope")

    canary_evidence = None
    if stage_name == "formal":
        canary_evidence = _canary_success(manifest)

    hardware = probe_hardware(required_gpu_ids=(0, 1))
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"GPU 0,1 are not an exact-H800 pool: {hardware}")
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(queue_path, rows, ROOT)
    provenance_binding = build_provenance_binding(ROOT)

    bound_files = {
        queue_path,
        DEFAULT_DESIGN,
        DEFAULT_QUEUE_MANIFEST,
        DEFAULT_PREDICTIONS,
        DEFAULT_CHALLENGER,
        DEFAULT_INVENTORY,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "data" / "dataset_info.json",
    }
    for binding in design["source_bindings"].values():
        bound_files.add(Path(binding["path"]))
    for row in rows:
        bound_files.add(Path(row["data_path"]))
        bound_files.add(Path(row["dataset_profile_path"]))
    missing = [str(path) for path in bound_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files are absent: {missing}")
    file_manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=lambda value: str(value))
    }
    ids = [str(row["job_id"]) for row in rows]
    queue_stage = {
        "allowed_job_ids": ids,
        "queue_path": queue_binding["path"],
        "queue_sha256": queue_binding["sha256"],
        "ordered_job_ids": queue_binding["ordered_job_ids"],
        "ordered_job_payload_sha256": queue_binding["ordered_job_payload_sha256"],
        "job_payload_sha256": queue_binding["job_payload_sha256"],
    }
    approval = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}:{stage_name}",
        "campaign_design": {
            "path": str(DEFAULT_DESIGN.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(DEFAULT_DESIGN),
            "report_sha256": design["report_sha256"],
        },
        "frozen_predictions": {
            "path": str(DEFAULT_PREDICTIONS.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(DEFAULT_PREDICTIONS),
            "report_sha256": predictions["report_sha256"],
            "generated_before_gpu": True,
        },
        "challenger": {
            "path": str(DEFAULT_CHALLENGER.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(DEFAULT_CHALLENGER),
            "report_sha256": challenger["report_sha256"],
        },
        "stage": stage_name,
        "canary_evidence": canary_evidence,
        "file_sha256": file_manifest,
        "execution_order": [f"{PHASE_ID}:{stage_name}"],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": [0, 1],
        "max_gpu_count": 2,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": "wait until GPU 0,1 are both idle, then run the frozen queue only on that pair; never preempt an external process",
        },
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": queue_stage,
    }
    candidate = ARTIFACT_DIR / f"approval_design_h800_bounded_memory_v2_fresh_holdout_{stage_name}_v1_candidate.json"
    write_json(candidate, approval)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = freeze(args.stage)
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
