#!/usr/bin/env python3
"""Freeze and optionally promote the exact 13-job eight-GPU boundary resume."""

from __future__ import annotations

import argparse
import copy
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
from prepare_h800_packing_memory_boundary_stage1_resume_v2 import (
    CAMPAIGN_ID,
    DESIGN,
    GPU_IDS,
    INVALID_OOM_ID,
    PARENT_QUEUE,
    PENDING_IDS,
    PHASE_ID,
    QUEUE,
    QUEUE_MANIFEST,
    RESUME_REVISION,
    SELECTION,
    VALID_TERMINAL_IDS,
)
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch


AUTHORIZATION = (
    "用户于2026-08-05在获知Packing显存边界第一阶段因provenance漂移停止后明确要求恢复，并在8卡上跑起来。"
    "本approval仅允许在H800 GPU 0至7上执行冻结的13个双卡续跑任务：12个未形成终态的原始低cutoff作业，"
    "以及1个因runtime manifest不完整而不可用于标定的B2 Packed OOM同配置重试。保留已有1个有效成功和2个有效OOM；"
    "不得抢占、重复有效终态、物化或启动高cutoff阶段、自动发布或自动开启Packing推荐。"
)
GPU_IDS_LIST = list(GPU_IDS)


def _validate_queue(rows: list[dict[str, Any]]) -> list[str]:
    parent = read_jsonl(PARENT_QUEUE)
    parent_by_id = {str(row["job_id"]): row for row in parent}
    ids = [str(row.get("job_id") or "") for row in rows]
    if (
        len(rows) != 13
        or len(set(ids)) != 13
        or tuple(ids[:12]) != PENDING_IDS
        or rows[:12] != [parent_by_id[job_id] for job_id in PENDING_IDS]
        or any(int(row.get("gpu_count", 0)) != 2 for row in rows)
        or any(int(row.get("mbs", 0)) != 1 for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
        or any(row.get("parallel_class") != "gpu_partitionable" for row in rows)
        or any(row.get("requires_external_node_idle") is not False for row in rows)
        or any(row.get("publication_allowed") is not False for row in rows)
        or any(float(row.get("expected_sample_gbs_relative_error", 1)) > 0.10 for row in rows)
    ):
        raise ValueError("queue is not the exact 12-pending plus one-retry resume")
    retry = copy.deepcopy(rows[-1])
    retry_id = retry.pop("job_id")
    parent_invalid = copy.deepcopy(parent_by_id[INVALID_OOM_ID])
    parent_invalid.pop("job_id", None)
    for key in (
        "resume_revision",
        "retry_of_job_id",
        "retry_ordinal",
        "retry_reason",
        "prior_status_path",
        "prior_status_sha256",
        "prior_classification",
        "prior_calibration_eligible",
    ):
        retry.pop(key, None)
    retry["execution_sequence_index"] = parent_invalid["execution_sequence_index"]
    if (
        retry != parent_invalid
        or not retry_id.startswith("h800packboundary1retry-")
        or rows[-1].get("retry_of_job_id") != INVALID_OOM_ID
        or rows[-1].get("prior_calibration_eligible") is not False
        or int(rows[-1].get("resume_revision", 0)) != RESUME_REVISION
    ):
        raise ValueError("B2 retry payload is not the exact repaired configuration")
    return ids


def freeze() -> Path:
    rows = read_jsonl(QUEUE)
    ids = _validate_queue(rows)
    experiment = read_json(ROOT / "config/experiment.json")
    scope = experiment["training_scope"]
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("model_ids") != ["qwen3_8b", "qwen3_14b"]
        or scope.get("gpu_ids") != GPU_IDS_LIST
        or scope.get("exclusive_node_gpu_ids") != GPU_IDS_LIST
        or scope.get("max_gpu_count") != 2
        or scope.get("gpu_counts") != [2]
        or scope.get("global_batch_sizes") != [128, 256]
        or scope.get("gradient_checkpointing") != [False, True]
        or scope.get("zero_by_gpu_count") != {"2": ["zero2", "zero3"]}
        or experiment.get("measurement", {}).get("performance_parallelism") != "disjoint_gpu_masks"
    ):
        raise ValueError("experiment config is not the exact eight-GPU resume scope")

    design = read_json(DESIGN)
    selection = read_json(SELECTION)
    manifest = read_json(QUEUE_MANIFEST)
    if (
        design.get("schema") != "sft_h800_packing_memory_boundary_stage1_resume/v2"
        or design.get("resume_revision") != RESUME_REVISION
        or design.get("gpu_training_started") is not False
        or design.get("resume_contract", {}).get("valid_parent_terminals_reused") != 3
        or design.get("resume_contract", {}).get("unchanged_parent_jobs_resumed") != 12
        or design.get("resume_contract", {}).get("ineligible_oom_retries") != 1
        or design.get("resume_contract", {}).get("high_cutoff_jobs") != 0
        or design.get("required_gpu_pool", {}).get("gpu_ids") != GPU_IDS_LIST
        or design.get("required_gpu_pool", {}).get("preview_two_gpu_masks_when_all_idle") != [[0, 1], [2, 3], [4, 5], [6, 7]]
        or design.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or design.get("queue", {}).get("ordered_job_ids") != ids
        or design.get("publication_allowed") is not False
        or selection.get("valid_terminal_count") != 3
        or selection.get("unchanged_pending_job_ids") != list(PENDING_IDS)
        or selection.get("retry_count") != 1
        or selection.get("retry_job_id") != ids[-1]
        or selection.get("high_cutoff_jobs_in_resume") != 0
        or manifest.get("design", {}).get("sha256") != sha256_file(DESIGN)
        or manifest.get("selection", {}).get("sha256") != sha256_file(SELECTION)
        or manifest.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or manifest.get("high_cutoff_jobs_materialized") != 0
    ):
        raise ValueError("resume design/selection/manifest drifted")

    for evidence in selection["valid_terminal"]:
        status = read_json(Path(evidence["status"]["path"]))
        if (
            status.get("job_id") != evidence["job_id"]
            or status.get("classification") != evidence["classification"]
            or status.get("calibration_eligible") is not True
            or status.get("execution_fingerprint_quality") != "complete"
        ):
            raise ValueError(f"valid parent terminal drifted: {evidence['job_id']}")
    invalid_status = read_json(Path(selection["invalid_terminal"]["status"]["path"]))
    if (
        invalid_status.get("job_id") != INVALID_OOM_ID
        or invalid_status.get("classification") != "oom"
        or invalid_status.get("calibration_eligible") is not False
    ):
        raise ValueError("invalid parent OOM drifted")
    for job_id in PENDING_IDS:
        if (ROOT / "results" / job_id / "status.json").exists():
            raise ValueError(f"pending parent job became terminal before approval: {job_id}")
    if (ROOT / "results" / ids[-1]).exists():
        raise ValueError("retry job acquired a result before approval")

    hardware = probe_hardware(required_gpu_ids=GPU_IDS_LIST)
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"selected GPUs are not an exact eight-H800 pool: {hardware}")
    live_processes = gpu_process_snapshot(GPU_IDS_LIST).get("processes") or []
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
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ARTIFACT_DIR / "dataset_analysis.json",
        ROOT / "config/experiment.json",
        ROOT / "config/hardware.json",
        ROOT / "config/deepspeed/ds_z2.json",
        ROOT / "config/deepspeed/ds_z3.json",
        ROOT / "data/dataset_info.json",
    }
    source_bindings = list((design.get("source_bindings") or {}).values())
    for binding in source_bindings:
        path = Path(binding["path"])
        bound_files.add(path)
        if sha256_file(path) != binding.get("sha256"):
            raise ValueError(f"resume source binding drifted: {path}")
    for row in rows:
        for key in ("data_path", "dataset_profile_path", "packing_dataprofile_path", "declared_model_manifest_path"):
            bound_files.add(Path(row[key]))
    missing = [str(path) for path in bound_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files missing: {missing}")
    root = ROOT.resolve()
    file_manifest = {
        str(path.resolve().relative_to(root)): sha256_file(path)
        for path in sorted(bound_files, key=str)
        if path.resolve().is_relative_to(root)
    }
    execution_source_paths = [
        root / relative
        for relative in file_manifest
        if relative.startswith("scripts/") or relative.startswith("config/")
    ]
    provenance_binding = build_provenance_binding(ROOT, source_paths=execution_source_paths)
    # Scoped provenance entries must also be bound by the approval manifest.
    for relative, digest in provenance_binding["scoped_source_manifest"].items():
        if file_manifest.get(relative) != digest:
            raise ValueError(f"scoped provenance path is absent from file manifest: {relative}")
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
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}:resume_v2",
        "campaign_design": {
            "path": str(DESIGN.resolve().relative_to(root)),
            "sha256": sha256_file(DESIGN),
            "report_sha256": design["report_sha256"],
        },
        "file_sha256": file_manifest,
        "execution_order": [f"{PHASE_ID}:resume_v2"],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": GPU_IDS_LIST,
        "max_gpu_count": 2,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": "use only idle disjoint masks [0,1], [2,3], [4,5], [6,7]; wait rather than preempt",
        },
        "continuation_evidence": {
            "selection_path": str(SELECTION.resolve().relative_to(root)),
            "selection_sha256": sha256_file(SELECTION),
            "valid_parent_terminals": 3,
            "unchanged_pending_jobs": 12,
            "ineligible_oom_retries": 1,
            "resume_jobs": 13,
            "resume_gpu_job_equivalents": 26,
        },
        "hardware_preflight": hardware,
        "idle_process_preflight": {
            "gpu_ids": GPU_IDS_LIST,
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
    candidate = ARTIFACT_DIR / "approval_design_h800_packing_memory_boundary_stage1_resume_v2_candidate.json"
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
    print(json.dumps({"candidate": str(candidate), "sha256": digest, "promotion": report}, ensure_ascii=False, indent=2))
    if report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
