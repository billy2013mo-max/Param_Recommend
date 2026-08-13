#!/usr/bin/env python3
"""Freeze and optionally promote the exact 16-job Packing boundary stage 1."""

from __future__ import annotations

import argparse
from collections import Counter
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
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch


AUTHORIZATION = (
    "用户于2026-08-05在查看Packing显存边界点表后明确要求把这些实验在GPU 0、1、2、3上跑起来。"
    "本approval仅允许在H800 GPU 0至3上执行冻结的第一阶段16个低cutoff双卡U/P配对作业；"
    "GPU 1、2当前外部任务不得抢占，调度器先使用空闲且全NVLink互联的0、3，随后等待1、2释放。"
    "高cutoff第二阶段不得自动物化或启动；"
    "CUDA OOM仅作为右删失显存下界，软件故障必须修复后原任务重跑，不得自动发布或开启Packing推荐。"
)
CAMPAIGN_ID = "h800_packing_memory_boundary_20260805_v1"
PHASE_ID = "h800_packing_memory_boundary_stage1_v1"
JOB_SCHEMA = "sft_h800_packing_memory_boundary_stage1_job/v1"
GPU_IDS = [0, 3, 1, 2]
QUEUE = ROOT / "matrix/h800_packing_memory_boundary_stage1_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_design_v1.json"
STATIC = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_static_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_queue_manifest_v1.json"
SELECTION = ARTIFACT_DIR / "h800_packing_memory_boundary_selection_v1.json"


def _validate_queue(rows: list[dict[str, Any]]) -> list[str]:
    ids = [str(row.get("job_id") or "") for row in rows]
    expected_settings = {
        "B1_8B_LoRA_W8_Z2_GCoff-c20480": ("qwen3_8b", "lora", 20_480, "zero2", False, 256, 3),
        "B2_14B_LoRA_W7_Z2_GCoff-c8192": ("qwen3_14b", "lora", 8_192, "zero2", False, 128, 24),
        "B3_14B_Full_W3_Z3_GCoff_G2-c8192": ("qwen3_14b", "full", 8_192, "zero3", False, 128, 9),
        "B4_14B_Full_W3_Z2_GCon_G2-c14336": ("qwen3_14b", "full", 14_336, "zero2", True, 128, 5),
    }
    if (
        len(rows) != 16
        or len(set(ids)) != 16
        or any(not job_id for job_id in ids)
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
        or any(int(row.get("boundary_stage", 0)) != 1 for row in rows)
        or any(int(row.get("execution_order_within_chain", 0)) != 1 for row in rows)
        or any(int(row.get("gpu_count", 0)) != 2 for row in rows)
        or any(int(row.get("mbs", 0)) != 1 for row in rows)
        or any(row.get("parallel_class") != "gpu_partitionable" for row in rows)
        or any(row.get("requires_external_node_idle") is not False for row in rows)
        or any(row.get("high_cutoff_auto_release_allowed") is not False for row in rows)
        or any(row.get("publication_allowed") is not False for row in rows)
        or [row.get("execution_sequence_index") for row in rows] != list(range(16))
    ):
        raise ValueError("queue is not the exact 16-job boundary stage-1 batch")
    if {str(row["boundary_setting_id"]) for row in rows} != set(expected_settings):
        raise ValueError("stage-1 setting set drifted")
    for setting_id, expected in expected_settings.items():
        subset = [row for row in rows if row["boundary_setting_id"] == setting_id]
        seen = Counter((bool(row["packing"]), int(row["repeat"])) for row in subset)
        model, train_type, cutoff, zero, gc, target_gbs, packed_ga = expected
        if len(subset) != 4 or len(seen) != 4 or set(seen.values()) != {1}:
            raise ValueError(f"{setting_id}: U/P repeats are not exact")
        for row in subset:
            ga = packed_ga if row["packing"] else target_gbs // 2
            if (
                row["model_id"] != model
                or row["train_type"] != train_type
                or int(row["cutoff_len"]) != cutoff
                or row["zero"] != zero
                or row["gc"] is not gc
                or int(row["target_gbs"]) != target_gbs
                or int(row["gradient_accumulation_steps"]) != ga
                or float(row["expected_sample_gbs_relative_error"]) > 0.10
                or row.get("first_epoch_capacity", {}).get("unpacked", {}).get("passed") is not True
                or row.get("first_epoch_capacity", {}).get("packed", {}).get("passed") is not True
            ):
                raise ValueError(f"{setting_id}: frozen mechanism/GBS contract drifted")
    return ids


def freeze() -> Path:
    rows = read_jsonl(QUEUE)
    ids = _validate_queue(rows)
    experiment = read_json(ROOT / "config/experiment.json")
    scope = experiment["training_scope"]
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("model_ids") != ["qwen3_8b", "qwen3_14b"]
        or scope.get("gpu_ids") != GPU_IDS
        or scope.get("exclusive_node_gpu_ids") != GPU_IDS
        or scope.get("max_gpu_count") != 2
        or scope.get("gpu_counts") != [2]
        or scope.get("global_batch_sizes") != [128, 256]
        or scope.get("gradient_checkpointing") != [False, True]
        or scope.get("zero_by_gpu_count") != {"2": ["zero2", "zero3"]}
        or experiment.get("measurement", {}).get("performance_parallelism") != "disjoint_gpu_masks"
    ):
        raise ValueError("experiment config is not the exact boundary stage-1 scope")
    design = read_json(DESIGN)
    static = read_json(STATIC)
    manifest = read_json(QUEUE_MANIFEST)
    if (
        design.get("schema") != "sft_h800_packing_memory_boundary_stage1_design/v1"
        or design.get("campaign_id") != CAMPAIGN_ID
        or design.get("gpu_training_started") is not False
        or design.get("stage_policy", {}).get("high_cutoff_jobs_in_this_queue") != 0
        or design.get("stage_policy", {}).get("stage2_automatic_release_allowed") is not False
        or design.get("required_gpu_pool", {}).get("gpu_ids") != GPU_IDS
        or design.get("required_gpu_pool", {}).get("preview_two_gpu_masks_when_all_idle") != [[0, 3], [1, 2]]
        or design.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or design.get("queue", {}).get("ordered_job_ids") != ids
        or design.get("publication_allowed") is not False
        or static.get("all_exact_runtime_gbs_contracts_passed") is not True
        or static.get("all_first_epoch_capacity_checks_passed") is not True
        or static.get("physical_p95_is_execution_admission_guard") is not False
        or manifest.get("design", {}).get("sha256") != sha256_file(DESIGN)
        or manifest.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or manifest.get("stage2_materialized") is not False
    ):
        raise ValueError("design/static/manifest drifted")

    hardware = probe_hardware(required_gpu_ids=GPU_IDS)
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"selected GPUs are not an exact four-H800 pool: {hardware}")
    live_processes = gpu_process_snapshot(GPU_IDS).get("processes") or []
    blocked = sorted({int(row["gpu_index"]) for row in live_processes})
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(QUEUE, rows, ROOT)
    provenance_binding = build_provenance_binding(ROOT)

    bound_files = {
        QUEUE,
        DESIGN,
        STATIC,
        QUEUE_MANIFEST,
        SELECTION,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ARTIFACT_DIR / "dataset_analysis.json",
        ROOT / "config/experiment.json",
        ROOT / "config/hardware.json",
        ROOT / "data/dataset_info.json",
    }
    source_bindings = list((design.get("source_bindings") or {}).values())
    for binding in source_bindings:
        path = Path(binding["path"])
        bound_files.add(path)
        if sha256_file(path) != binding.get("sha256"):
            raise ValueError(f"campaign source binding drifted: {path}")
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
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}",
        "campaign_design": {
            "path": str(DESIGN.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(DESIGN),
            "report_sha256": design["report_sha256"],
        },
        "file_sha256": file_manifest,
        "execution_order": [PHASE_ID],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": GPU_IDS,
        "max_gpu_count": 2,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": "use only idle fully-NVLinked masks [0,3] and [1,2]; wait rather than preempt",
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
    candidate = ARTIFACT_DIR / "approval_design_h800_packing_memory_boundary_stage1_v1_candidate.json"
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
