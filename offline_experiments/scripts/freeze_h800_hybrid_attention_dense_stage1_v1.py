#!/usr/bin/env python3
"""Freeze and promote an exact dense hybrid-attention stage-1 queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from approval_gate import build_provenance_binding, build_queue_binding
from common import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from prepare_h800_hybrid_attention_dense_stage1_v1 import (
    AUTHORIZED_GPU_IDS,
    CAMPAIGN_ID,
    CANARY_PHASE_ID,
    CANARY_QUEUE,
    DATASET_REGISTRY,
    DESIGN,
    FORMAL_PHASE_ID,
    FORMAL_QUEUE,
    JOB_SCHEMA,
    MODEL_INVENTORY,
    PROFILE_MANIFEST,
    RUNTIME_CONTRACT,
)
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch, validate_job

AUTHORIZATION = (
    "用户在 2026-08-12 明确要求开跑已设计的 Qwen3/Qwen3.5/Qwen3.6 dense "
    "混合注意力实验。本审批只授权 GPU 0-6 上当前冻结的 canary 或正式队列；"
    "不允许 GPU 7、抢占其他任务、扩展模型/数据/作业、启用 packing/offload、"
    "修改 LoRA 配置或把 OOM 当作精确峰值。必须先完成并通过四个 canary，才可"
    "提升 208 个正式任务的独立审批。成功任务记录精确 allocator 峰值与吞吐，"
    "OOM 只记录右删失下界；本批只用于标定，不能直接作为前瞻验收或发布证据。"
)
MAX_GPU_COUNT = 4
CANARY_ACCEPTANCE = (
    ARTIFACT_DIR
    / "h800_hybrid_attention_dense_stage1_canary_acceptance_v1.json"
)
STAGES = {
    "canary": {
        "phase_id": CANARY_PHASE_ID,
        "queue": CANARY_QUEUE,
        "jobs": 4,
    },
    "formal": {
        "phase_id": FORMAL_PHASE_ID,
        "queue": FORMAL_QUEUE,
        "jobs": 208,
    },
}

SCOPED_SOURCE_RELATIVE_PATHS = (
    "config/experiment.json",
    "config/hardware.json",
    "config/models.json",
    "config/deepspeed/ds_z2.json",
    "config/deepspeed/ds_z3.json",
    "scripts/approval_gate.py",
    "scripts/common.py",
    "scripts/promote_approval_candidate.py",
    "scripts/hybrid_attention_memory_features.py",
    "scripts/prepare_h800_hybrid_attention_dense_stage1_v1.py",
    "scripts/freeze_h800_hybrid_attention_dense_stage1_v1.py",
    "scripts/evaluate_h800_hybrid_attention_dense_stage1_v1.py",
    "scripts/run_job.py",
    "scripts/scheduler.py",
    "scripts/train_entry.py",
    "scripts/metrics_callback.py",
    "scripts/model_structure_manifest.py",
    "scripts/runtime_evidence.py",
    "scripts/gpu_telemetry.py",
    "scripts/collect_results.py",
)


def _prerequisite(stage: str) -> dict[str, Any] | None:
    if stage == "canary":
        return None
    if not CANARY_ACCEPTANCE.is_file():
        raise RuntimeError("formal stage remains locked: canary acceptance is absent")
    report = read_json(CANARY_ACCEPTANCE)
    if (
        report.get("schema")
        != "sft_h800_hybrid_attention_dense_stage1_canary_acceptance/v1"
        or report.get("campaign_id") != CAMPAIGN_ID
        or report.get("all_passed") is not True
    ):
        raise RuntimeError("formal stage remains locked: canary did not pass")
    return {
        "path": str(CANARY_ACCEPTANCE.resolve().relative_to(ROOT.resolve())),
        "sha256": sha256_file(CANARY_ACCEPTANCE),
        "report_sha256": report.get("report_sha256"),
        "all_passed": True,
    }


def _validate_live_scope(stage: str) -> None:
    experiment = read_json(CONFIG_DIR / "experiment.json")
    scope = experiment.get("training_scope") or {}
    expected_phase = STAGES[stage]["phase_id"]
    if (
        scope.get("phase_id") != expected_phase
        or scope.get("gpu_ids") != list(AUTHORIZED_GPU_IDS)
        or scope.get("exclusive_node_gpu_ids") != list(AUTHORIZED_GPU_IDS)
        or scope.get("max_gpu_count") != MAX_GPU_COUNT
        or scope.get("gpu_counts") != [1, 2, 4]
        or scope.get("global_batch_sizes") != [64]
    ):
        raise ValueError("live dense hybrid experiment scope drifted")


def freeze(stage: str) -> Path:
    spec = STAGES[stage]
    queue_path = Path(spec["queue"])
    rows = read_jsonl(queue_path)
    design = read_json(DESIGN)
    if (
        len(rows) != int(spec["jobs"])
        or len({str(row.get("job_id") or "") for row in rows}) != len(rows)
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != spec["phase_id"] for row in rows)
    ):
        raise ValueError(f"{stage} queue identity/count drifted")
    if design.get("gpu_training_started") is not False:
        raise ValueError("campaign design says GPU training already started")
    for row in rows:
        validate_job(row)
        if row.get("requested_gpu_pool") != list(AUTHORIZED_GPU_IDS):
            raise ValueError(f"job GPU pool drifted: {row['job_id']}")
    _validate_live_scope(stage)
    prerequisite = _prerequisite(stage)

    hardware = probe_hardware(required_gpu_ids=AUTHORIZED_GPU_IDS)
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"GPU 0-6 are not an exact H800 pool: {hardware}")
    if hardware.get("selected_gpu_compute_processes"):
        raise RuntimeError("refusing to freeze while GPU 0-6 have compute processes")
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(queue_path, rows, ROOT)

    scoped_paths = [ROOT / relative for relative in SCOPED_SOURCE_RELATIVE_PATHS]
    missing_sources = [str(path) for path in scoped_paths if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"approval source files are absent: {missing_sources}")
    provenance_binding = build_provenance_binding(ROOT, source_paths=scoped_paths)
    bound_files = {
        queue_path,
        DESIGN,
        MODEL_INVENTORY,
        PROFILE_MANIFEST,
        DATASET_REGISTRY,
        RUNTIME_CONTRACT,
        ARTIFACT_DIR / "provenance.json",
        *scoped_paths,
    }
    for row in rows:
        bound_files.add(Path(row["data_path"]))
        bound_files.add(Path(row["dataset_profile_path"]))
        overlay = row.get("environment_overlay") or {}
        if overlay:
            bound_files.add(Path(overlay["contract_path"]))
    if prerequisite is not None:
        bound_files.add(CANARY_ACCEPTANCE)
    missing = [str(path) for path in bound_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files are absent: {missing}")
    file_manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=str)
    }
    ids = [str(row["job_id"]) for row in rows]
    queue_stage = {
        "allowed_job_ids": ids,
        "queue_path": queue_binding["path"],
        "queue_sha256": queue_binding["sha256"],
        "ordered_job_ids": queue_binding["ordered_job_ids"],
        "ordered_job_payload_sha256": queue_binding[
            "ordered_job_payload_sha256"
        ],
        "job_payload_sha256": queue_binding["job_payload_sha256"],
    }
    approval_design: dict[str, Any] = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{spec['phase_id']}",
        "campaign_design": {
            "path": str(DESIGN.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(DESIGN),
            "report_sha256": design.get("report_sha256"),
        },
        "stage": stage,
        "stage_prerequisite": prerequisite,
        "file_sha256": file_manifest,
        "execution_order": [str(spec["phase_id"])],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": list(AUTHORIZED_GPU_IDS),
        "max_gpu_count": MAX_GPU_COUNT,
        "scheduler_execution": {
            "join_busy_pool": False,
            "preemption_allowed": False,
            "policy": (
                "use only idle GPU 0-6; disjoint 1/2-card masks; repository "
                "conservative pool-exclusive 4-card jobs"
            ),
        },
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": queue_stage,
    }
    candidate = (
        ARTIFACT_DIR
        / f"approval_design_h800_hybrid_attention_dense_stage1_{stage}_v1_candidate.json"
    )
    write_json(candidate, approval_design)
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
    print(
        json.dumps(
            {
                "candidate": str(candidate),
                "sha256": digest,
                "stage": args.stage,
                "promoted": bool(args.promote),
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
