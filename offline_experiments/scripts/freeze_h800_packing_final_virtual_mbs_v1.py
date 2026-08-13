#!/usr/bin/env python3
"""Freeze and optionally promote the exact 36-job virtual-MBS experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_packing_final_virtual_mbs_v1 import (
    CAMPAIGN_ID,
    DATASET_INFO,
    DESIGN,
    EXPECTED_JOBS,
    EXPERIMENT,
    GPU_COUNTS,
    GPU_IDS,
    JOB_SCHEMA,
    MEASURE_STEPS,
    MODEL_INVENTORY,
    PHASE_ID,
    QUEUE,
    STATIC,
    WARMUP_STEPS,
)
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch, validate_job


AUTHORIZATION = (
    "用户在2026-08-10确认精简后的Packing最终验收计划并明确说‘开跑吧’；Phase 0随后8/8通过。"
    "本批准只允许Phase 1冻结的36个virtual-MBS任务：PF02/PF05/PF06、4/5/6/7卡、"
    "packed与unpacked floor/ceil三臂，Qwen3-8B LoRA、ZeRO-3、GC开、每项3步warmup加20步测量。"
    "任务整机串行，不得抢占外部进程，不得自动补第二重复、不得启动Phase 2/4、不得发布模型。"
)
CANDIDATE = ARTIFACT_DIR / "approval_design_h800_packing_final_virtual_mbs_v1_candidate.json"
SCRIPTS = ROOT / "scripts"
STAGING = ROOT / "packing_final_4to7gpu_staging"


def _hardware() -> dict[str, Any]:
    hardware = probe_hardware(required_gpu_ids=list(GPU_IDS))
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError("GPU 0-7 are not an exact eight-H800 pool")
    if hardware.get("selected_gpu_compute_processes"):
        raise RuntimeError("one or more authorized GPUs are busy")
    hardware.update(
        {
            "occupancy_is_authorization_time_preflight": True,
            "join_busy_pool_required": True,
            "preemption_allowed": False,
            "gpu_ids_outside_authorized_pool_must_not_be_used": True,
        }
    )
    return hardware


def _validate(rows: list[dict[str, Any]]) -> None:
    if (
        len(rows) != EXPECTED_JOBS
        or len({str(row.get("job_id")) for row in rows}) != EXPECTED_JOBS
        or [int(row.get("execution_sequence_index", -1)) for row in rows]
        != list(range(EXPECTED_JOBS))
    ):
        raise ValueError("queue is not the exact ordered 36-job experiment")
    by_group: dict[str, set[str]] = {}
    for row in rows:
        validate_job(row)
        by_group.setdefault(str(row["matched_group_id"]), set()).add(str(row["arm_id"]))
        available_pool_execution = row.get("workload_id") in {"PF05", "PF06"}
        if (
            row.get("schema") != JOB_SCHEMA
            or row.get("campaign_id") != CAMPAIGN_ID
            or row.get("phase_id") != PHASE_ID
            or int(row.get("gpu_count", 0)) not in GPU_COUNTS
            or row.get("zero") != "zero3"
            or row.get("gc") is not True
            or int(row.get("warmup_steps", -1)) != WARMUP_STEPS
            or int(row.get("measure_steps", -1)) != MEASURE_STEPS
            or row.get("parallel_class")
            != ("available_pool" if available_pool_execution else "exclusive_pool")
            or bool(row.get("requires_external_node_idle"))
            is available_pool_execution
            or bool(row.get("allow_available_pool_for_large_job"))
            is not available_pool_execution
            or row.get("strict_queue_order") is not True
            or row.get("publication_allowed") is not False
            or int(row.get("repeat", -1)) != 0
        ):
            raise ValueError(f"queue row left Phase 1 scope: {row.get('job_id')}")
        if bool(row["packing"]) is (row["arm_id"] != "packed"):
            raise ValueError(f"Packing arm label drifted: {row['job_id']}")
        if bool(row["packing"]) and int(row["mbs"]) != 1:
            raise ValueError(f"Packed physical MBS drifted: {row['job_id']}")
        for path_key, digest_key in (
            ("data_path", "data_sha256"),
            ("dataset_profile_path", "dataset_profile_sha256"),
        ):
            path = Path(str(row[path_key]))
            if not path.is_file() or sha256_file(path) != row[digest_key]:
                raise ValueError(f"job binding drifted: {row['job_id']}:{path_key}")
    if len(by_group) != 12 or any(arms != {"packed", "floor", "ceil"} for arms in by_group.values()):
        raise ValueError("each matched group must contain packed/floor/ceil")


def freeze() -> Path:
    rows = read_jsonl(QUEUE)
    _validate(rows)
    design = read_json(DESIGN)
    if (
        design.get("campaign_id") != CAMPAIGN_ID
        or design.get("phase_id") != PHASE_ID
        or design.get("gpu_training_started") is not False
        or design.get("automatic_repeat_expansion_allowed") is not False
        or design.get("ordered_job_ids") != [str(row["job_id"]) for row in rows]
        or design.get("ordered_job_payload_sha256") != sha256_json(rows)
        or design.get("bindings", {}).get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or design.get("bindings", {}).get("experiment", {}).get("sha256") != sha256_file(EXPERIMENT)
        or design.get("bindings", {}).get("static", {}).get("sha256") != sha256_file(STATIC)
    ):
        raise ValueError("Phase 1 design bindings drifted")
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

    hardware = _hardware()
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(QUEUE, rows, ROOT)
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
        STAGING / "launch_h800_packing_final_virtual_mbs_v1.py",
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "hardware.json",
        ROOT / "config" / "models.json",
        ROOT / "config" / "deepspeed" / "ds_z3.json",
    ]
    provenance_binding = build_provenance_binding(ROOT, source_paths=scoped_sources)
    bound_files = {
        QUEUE,
        DESIGN,
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
    for row in rows:
        bound_files.add(Path(str(row["data_path"])))
        bound_files.add(Path(str(row["dataset_profile_path"])))
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
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}",
        "authorization": AUTHORIZATION,
        "campaign_design": {
            "path": str(DESIGN.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(DESIGN),
            "report_sha256": design["report_sha256"],
        },
        "file_sha256": file_manifest,
        "execution_order": [PHASE_ID],
        "allowed_job_ids": queue_binding["ordered_job_ids"],
        "authorized_gpu_ids": list(GPU_IDS),
        "max_gpu_count": 7,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": "strict serial whole-node execution; exact first N mask for 4/5/6/7 ranks",
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
            "ordered_job_payload_sha256": queue_binding["ordered_job_payload_sha256"],
            "job_payload_sha256": queue_binding["job_payload_sha256"],
        },
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
                "jobs": EXPECTED_JOBS,
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
