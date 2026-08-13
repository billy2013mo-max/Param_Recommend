#!/usr/bin/env python3
"""Freeze and promote the exact Packing interactions batch-1 continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_packing_platform_v4_interactions_batch1_continuation_v1 import (
    CAMPAIGN_ID,
    DESIGN,
    EXPECTED_REMAINING_SOURCE_IDS,
    INVENTORY,
    JOB_SCHEMA,
    PARENT_CAMPAIGN_ID,
    PARENT_DESIGN,
    PARENT_QUEUE,
    PHASE_ID,
    QUEUE,
    QUEUE_MANIFEST,
    SELECTION,
    continuation_row,
)
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch


AUTHORIZATION = (
    "用户在 2026-08-04 明确要求继续后面的 GPU 实验。原 24 项 Packing×GC/ZeRO 批次因项目中出现新的 Python 源文件而按 provenance 门禁停止；"
    "已完成的 14 项均为 calibration-eligible success。本 approval 仅允许在 H800 GPU 0、1、4、5、6、7 上执行冻结的 10 项精确差集，"
    "其中 GPU 2、3 始终排除；允许等待空闲卡加入资源池，但不得抢占、重复运行前 14 项、扩展候选、自动启动下一批或自动发布。"
)
GPU_IDS = [0, 1, 4, 5, 6, 7]
PARENT_STATIC = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_static_features_v1.json"


def _validate_exact_queue(rows: list[dict[str, Any]]) -> None:
    parent_rows = read_jsonl(PARENT_QUEUE)
    parent_by_id = {str(row["job_id"]): row for row in parent_rows}
    expected = [
        continuation_row(parent_by_id[source_id], index)
        for index, source_id in enumerate(EXPECTED_REMAINING_SOURCE_IDS)
    ]
    if rows != expected:
        raise ValueError("continuation queue is not the exact deterministic parent-row mapping")
    ids = [str(row.get("job_id") or "") for row in rows]
    if (
        len(rows) != 10
        or len(set(ids)) != 10
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
        or [row.get("execution_sequence_index") for row in rows] != list(range(10))
        or tuple(str(row.get("source_job_id")) for row in rows) != EXPECTED_REMAINING_SOURCE_IDS
        or sum(int(row.get("gpu_count", 0)) == 1 for row in rows) != 4
        or sum(int(row.get("gpu_count", 0)) == 2 for row in rows) != 6
        or sum(int(row.get("gpu_count", 0)) for row in rows) != 16
        or any(int(row.get("mbs", 0)) != 1 for row in rows)
        or any(float(row.get("expected_sample_gbs_relative_error", 1)) > 0.05 for row in rows)
        or any(row.get("parallel_class") != "gpu_partitionable" for row in rows)
        or any(row.get("requires_external_node_idle") is not False for row in rows)
        or any(row.get("publication_allowed") is not False for row in rows)
    ):
        raise ValueError("continuation queue violates its exact frozen scope")


def freeze() -> Path:
    rows = read_jsonl(QUEUE)
    _validate_exact_queue(rows)
    ids = [str(row["job_id"]) for row in rows]

    experiment = read_json(ROOT / "config" / "experiment.json")
    scope = experiment["training_scope"]
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("model_ids") != ["qwen3_8b"]
        or scope.get("gpu_ids") != GPU_IDS
        or scope.get("exclusive_node_gpu_ids") != GPU_IDS
        or scope.get("max_gpu_count") != 2
        or scope.get("gpu_counts") != [1, 2]
        or scope.get("global_batch_sizes") != [64]
        or scope.get("gradient_checkpointing") != [True, False]
        or scope.get("zero_by_gpu_count") != {"1": ["none"], "2": ["zero2", "zero3"]}
        or experiment.get("measurement", {}).get("performance_parallelism") != "disjoint_gpu_masks"
    ):
        raise ValueError("experiment config is not the exact continuation scope")

    design = read_json(DESIGN)
    selection = read_json(SELECTION)
    queue_manifest = read_json(QUEUE_MANIFEST)
    if (
        design.get("schema") != "sft_h800_packing_platform_v4_interactions_batch1_continuation_design/v1"
        or design.get("campaign_id") != CAMPAIGN_ID
        or design.get("parent_campaign_id") != PARENT_CAMPAIGN_ID
        or design.get("gpu_training_started") is not False
        or design.get("continuation_only") is not True
        or design.get("automatic_next_batch_allowed") is not False
        or design.get("publication_allowed") is not False
        or design.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or design.get("queue", {}).get("ordered_job_ids") != ids
        or design.get("queue", {}).get("ordered_source_job_ids") != list(EXPECTED_REMAINING_SOURCE_IDS)
        or selection.get("gpu_training_started") is not False
        or selection.get("completed_success_count") != 14
        or selection.get("remaining_count") != 10
        or selection.get("remaining_source_job_ids") != list(EXPECTED_REMAINING_SOURCE_IDS)
        or selection.get("continuation_job_ids") != ids
        or queue_manifest.get("design", {}).get("sha256") != sha256_file(DESIGN)
        or queue_manifest.get("selection", {}).get("sha256") != sha256_file(SELECTION)
        or queue_manifest.get("queue", {}).get("sha256") != sha256_file(QUEUE)
    ):
        raise ValueError("continuation design, selection or queue manifest drifted")

    completed_ids = [str(row["source_job_id"]) for row in selection["completed"]]
    if len(completed_ids) != 14 or set(completed_ids) & set(EXPECTED_REMAINING_SOURCE_IDS):
        raise ValueError("completed predecessor set overlaps the continuation")
    for evidence in selection["completed"]:
        status = read_json(Path(evidence["status"]["path"]))
        if (
            status.get("job_id") != evidence["source_job_id"]
            or status.get("classification") != "success"
            or status.get("calibration_eligible") is not True
        ):
            raise ValueError(f"completed predecessor status drifted: {evidence['source_job_id']}")

    hardware = probe_hardware(required_gpu_ids=GPU_IDS)
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"selected GPUs are not an exact H800 pool: {hardware}")
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(QUEUE, rows, ROOT)
    provenance_binding = build_provenance_binding(ROOT)

    bound_files = {
        QUEUE,
        DESIGN,
        SELECTION,
        QUEUE_MANIFEST,
        PARENT_QUEUE,
        PARENT_DESIGN,
        PARENT_STATIC,
        INVENTORY,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ARTIFACT_DIR / "dataset_analysis.json",
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "hardware.json",
        ROOT / "data" / "dataset_info.json",
    }
    source_bindings = list((design.get("source_bindings") or {}).values())
    for binding in source_bindings:
        bound_files.add(Path(binding["path"]))
    for row in rows:
        bound_files.add(Path(row["data_path"]))
        bound_files.add(Path(row["dataset_profile_path"]))
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
            "path": str(DESIGN.resolve().relative_to(root)),
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
            "policy": "use only idle disjoint H800 masks [0,1], [4,5], [6,7] for 2-GPU jobs and any idle authorized card for 1-GPU jobs; wait rather than preempt; exclude GPUs 2/3",
        },
        "continuation_evidence": {
            "selection_path": str(SELECTION.resolve().relative_to(root)),
            "selection_sha256": sha256_file(SELECTION),
            "completed_parent_successes": 14,
            "remaining_jobs": 10,
            "remaining_gpu_job_equivalents": 16,
        },
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": stage,
    }
    candidate = ARTIFACT_DIR / "approval_design_h800_packing_platform_v4_interactions_batch1_continuation_v1_candidate.json"
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
