#!/usr/bin/env python3
"""Freeze and promote the exact 20-job Packing Phase-C continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from approval_gate import build_provenance_binding, build_queue_binding
from common import (
    ARTIFACT_DIR,
    ROOT,
    gpu_process_snapshot,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from prepare_h800_packing_profile_phase_c_continuation_v1 import (
    CAMPAIGN_ID,
    DESIGN,
    EXPECTED_COMPLETED_IDS,
    EXPECTED_REMAINING_IDS,
    PARENT_DESIGN,
    PARENT_MANIFEST,
    PARENT_PREDICTIONS,
    PARENT_QUEUE,
    PARENT_STATIC,
    PHASE_ID,
    QUEUE,
    QUEUE_MANIFEST,
    SELECTION,
)
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch


AUTHORIZATION = (
    "用户于2026-08-05明确要求继续推进下一步并恢复8卡并发。原Phase-C队列已有4项"
    "calibration-eligible success，随后因共享源码provenance变化按门禁安全停止。"
    "本approval只允许在H800 GPU 0至7的空闲双卡mask上执行原冻结队列的精确20项差集；"
    "不得抢占、重复已完成4项、修改训练字段、扩展候选、自动启动下一批或自动发布。"
)
GPU_IDS = list(range(8))
EXPECTED_MASKS = [[0, 1], [2, 3], [4, 5], [6, 7]]


def _validate_queue(rows: list[dict[str, Any]]) -> list[str]:
    parent = read_jsonl(PARENT_QUEUE)
    parent_by_id = {str(row["job_id"]): row for row in parent}
    ids = [str(row.get("job_id") or "") for row in rows]
    if (
        tuple(ids) != EXPECTED_REMAINING_IDS
        or rows != [parent_by_id[job_id] for job_id in EXPECTED_REMAINING_IDS]
        or len(set(ids)) != 20
        or any(int(row.get("gpu_count", 0)) != 2 for row in rows)
        or sum(int(row.get("gpu_count", 0)) for row in rows) != 40
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
        or any(int(row.get("target_gbs", 0)) != 128 for row in rows)
        or any(int(row.get("mbs", 0)) != 1 for row in rows)
        or any(float(row.get("expected_sample_gbs_relative_error", 1)) > 0.05 for row in rows)
        or any(row.get("parallel_class") != "gpu_partitionable" for row in rows)
        or any(row.get("requires_external_node_idle") is not False for row in rows)
        or any(row.get("publication_allowed") is not False for row in rows)
        or any(
            float(row.get("memory_execution_guarded_upper_bytes", float("inf")))
            > float(row.get("memory_execution_limit_bytes", 0))
            for row in rows
        )
    ):
        raise ValueError("continuation queue is not the exact unchanged 20-row set difference")
    return ids


def freeze() -> Path:
    rows = read_jsonl(QUEUE)
    ids = _validate_queue(rows)
    experiment = read_json(ROOT / "config" / "experiment.json")
    scope = experiment["training_scope"]
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("model_ids") != ["qwen3_8b"]
        or scope.get("gpu_ids") != GPU_IDS
        or scope.get("exclusive_node_gpu_ids") != GPU_IDS
        or scope.get("max_gpu_count") != 2
        or scope.get("gpu_counts") != [2]
        or scope.get("global_batch_sizes") != [128]
        or scope.get("gradient_checkpointing") != [True, False]
        or scope.get("zero_by_gpu_count") != {"2": ["zero2", "zero3"]}
        or experiment.get("measurement", {}).get("performance_parallelism")
        != "disjoint_gpu_masks"
    ):
        raise ValueError("experiment config drifted from the Phase-C scope")

    design = read_json(DESIGN)
    selection = read_json(SELECTION)
    manifest = read_json(QUEUE_MANIFEST)
    if (
        design.get("schema")
        != "sft_h800_packing_profile_phase_c_continuation_design/v1"
        or design.get("campaign_id") != CAMPAIGN_ID
        or design.get("gpu_training_started") is not False
        or design.get("automatic_next_batch_allowed") is not False
        or design.get("publication_allowed") is not False
        or design.get("required_gpu_pool", {}).get("gpu_ids") != GPU_IDS
        or design.get("required_gpu_pool", {}).get("two_gpu_masks") != EXPECTED_MASKS
        or design.get("required_gpu_pool", {}).get("max_parallel_jobs") != 4
        or design.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or design.get("queue", {}).get("ordered_job_ids") != ids
        or selection.get("completed_success_count") != 4
        or tuple(row["job_id"] for row in selection.get("completed") or [])
        != EXPECTED_COMPLETED_IDS
        or selection.get("remaining_count") != 20
        or selection.get("remaining_job_ids") != ids
        or manifest.get("design", {}).get("sha256") != sha256_file(DESIGN)
        or manifest.get("selection", {}).get("sha256") != sha256_file(SELECTION)
        or manifest.get("queue", {}).get("sha256") != sha256_file(QUEUE)
    ):
        raise ValueError("continuation design, selection, or manifest drifted")
    for evidence in selection["completed"]:
        status = read_json(Path(evidence["status"]["path"]))
        if (
            status.get("job_id") != evidence["job_id"]
            or status.get("classification") != "success"
            or status.get("calibration_eligible") is not True
        ):
            raise ValueError(f"bound completed Phase-C result drifted: {evidence['job_id']}")

    hardware = probe_hardware(required_gpu_ids=GPU_IDS)
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"selected GPUs are not an exact H800 pool: {hardware}")
    live_processes = gpu_process_snapshot(GPU_IDS).get("processes") or []
    blocked = sorted({int(row["gpu_index"]) for row in live_processes})
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(QUEUE, rows, ROOT)

    bound_files = {
        QUEUE,
        DESIGN,
        SELECTION,
        QUEUE_MANIFEST,
        PARENT_QUEUE,
        PARENT_DESIGN,
        PARENT_MANIFEST,
        PARENT_STATIC,
        PARENT_PREDICTIONS,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ARTIFACT_DIR / "dataset_analysis.json",
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "hardware.json",
        ROOT / "config" / "deepspeed" / "ds_z2.json",
        ROOT / "config" / "deepspeed" / "ds_z3.json",
        ROOT / "data" / "dataset_info.json",
        ROOT / "scripts" / "approval_gate.py",
        ROOT / "scripts" / "common.py",
        ROOT / "scripts" / "profiler_callback.py",
        ROOT / "scripts" / "runtime_evidence.py",
    }
    source_bindings = list((design.get("source_bindings") or {}).values())
    for binding in source_bindings:
        bound_files.add(Path(binding["path"]))
    for row in rows:
        for key in ("data_path", "dataset_profile_path", "packing_dataprofile_path"):
            bound_files.add(Path(row[key]))
    missing = [str(path) for path in bound_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files are absent: {missing}")
    mismatched = [
        str(binding["path"])
        for binding in source_bindings
        if sha256_file(Path(binding["path"])) != binding.get("sha256")
    ]
    if mismatched:
        raise ValueError(f"continuation source bindings drifted: {mismatched}")
    root = ROOT.resolve()
    file_manifest = {
        str(path.resolve().relative_to(root)): sha256_file(path)
        for path in sorted(bound_files, key=lambda value: str(value))
        if path.resolve().is_relative_to(root)
    }
    execution_source_paths = [
        root / relative
        for relative in file_manifest
        if relative.startswith("scripts/") or relative.startswith("config/")
    ]
    provenance_binding = build_provenance_binding(
        ROOT, source_paths=execution_source_paths
    )
    stage = {
        "allowed_job_ids": ids,
        "queue_path": queue_binding["path"],
        "queue_sha256": queue_binding["sha256"],
        "ordered_job_ids": queue_binding["ordered_job_ids"],
        "ordered_job_payload_sha256": queue_binding["ordered_job_payload_sha256"],
        "job_payload_sha256": queue_binding["job_payload_sha256"],
    }
    approval: dict[str, Any] = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}:continuation_v1",
        "campaign_design": {
            "path": str(DESIGN.resolve().relative_to(root)),
            "sha256": sha256_file(DESIGN),
            "report_sha256": design["report_sha256"],
        },
        "file_sha256": file_manifest,
        "execution_order": [f"{PHASE_ID}:continuation_v1"],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": GPU_IDS,
        "max_gpu_count": 2,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": "run the exact 20-job set difference on four idle disjoint H800 pairs; wait rather than preempt",
        },
        "continuation_evidence": {
            "selection_path": str(SELECTION.resolve().relative_to(root)),
            "selection_sha256": sha256_file(SELECTION),
            "completed_successes": 4,
            "remaining_jobs": 20,
            "remaining_gpu_job_equivalents": 40,
        },
        "hardware_preflight": hardware,
        "idle_process_preflight": {
            "gpu_ids": GPU_IDS,
            "processes": live_processes,
            "initial_blocked_gpu_ids": blocked,
            "all_idle": not blocked,
            "join_busy_pool": True,
            "preemption_allowed": False,
        },
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": stage,
    }
    candidate = ARTIFACT_DIR / "approval_design_h800_packing_profile_phase_c_continuation_v1_candidate.json"
    write_json(candidate, approval)
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
    print(
        json.dumps(
            {"candidate": str(candidate), "sha256": digest, "promotion": report},
            ensure_ascii=False,
            indent=2,
        )
    )
    if report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
