#!/usr/bin/env python3
"""Freeze and promote the 4096-anchor supplement queue.

This closes the qwen3.6-27B × cutoff=4096 gap in the dense hybrid attention
stage-1 fit dataset.  It follows the same shape as
``freeze_h800_hybrid_attention_dense_stage1_v1.py`` but scopes the approval
to the 20-job supplement queue produced by
``prepare_h800_hybrid_attention_dense_stage1_supplement_4096_v1.py``.

Two intentional divergences from the original stage-1 freeze:

1. No canary/formal split.  The supplement contains 20 jobs total; the first
   one dispatched serves as an implicit canary and any failure surfaces
   immediately in the collected observations.

2. The build-time provenance tree for the DeepSpeed ZeRO-3 mixed-dtype fix
   was lost when ``/fine-tuning-launcher/`` was rebuilt on 2026-08-19.
   Correctness now rides on the installed DeepSpeed source being pinned to a
   patched upstream commit (see ``finetuning-launcher/pyproject.qwen36.toml``
   and the ``partition_parameters.py`` shipped in ``qwen36_venv``).
   ``freeze_lora_zero3_fix.validate_patch`` was updated on 2026-08-25 to
   report this as ``mode=runtime_source_only`` with ``all_passed=True`` when
   the source-level fix is verifiable and the (now deprecated) build-time
   provenance tree is absent in its entirety.
"""

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
    JOB_SCHEMA,
)
from prepare_h800_hybrid_attention_dense_stage1_supplement_4096_v1 import (
    CAMPAIGN_ID,
    DATASET_REGISTRY,
    DESIGN,
    FORMAL_PHASE_ID,
    FORMAL_QUEUE,
    MODEL_INVENTORY,
    PROFILE_MANIFEST,
    RUNTIME_CONTRACT,
    SUPPLEMENT_MODEL_IDS,
)
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch, validate_job

AUTHORIZATION = (
    "用户在 2026-08-25 明确要求开跑 qwen3.6-27B × cutoff=4096 的补数拟合队列，"
    "以修复 hybrid dense 显存模型在该组的过预测（拟合数据缺 4096 训练点，导致"
    "v3 前瞻验收在 27B/4096 组预测 163-187 GB、实测 95-116 GB）。本审批只授权 "
    "GPU 0-6 上当前冻结的 20 个补数任务，含 5 个并行路由 × 2 gradient "
    "checkpointing × 2 micro-batch size，全部使用 27B 与业务短样本、cutoff=4096。"
    "不允许 GPU 7、抢占其他任务、扩展模型/数据/作业、启用 packing/offload、"
    "修改 LoRA 配置或把 OOM 当作精确峰值。DeepSpeed ZeRO-3 数值修正由已 pin 的"
    " upstream commit 保证；OOM 只记录右删失下界；本批只用于补齐显存标定。"
)
MAX_GPU_COUNT = 4

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
    "scripts/prepare_h800_hybrid_attention_dense_stage1_supplement_4096_v1.py",
    "scripts/freeze_h800_hybrid_attention_dense_stage1_supplement_4096_v1.py",
    "scripts/run_job.py",
    "scripts/scheduler.py",
    "scripts/train_entry.py",
    "scripts/metrics_callback.py",
    "scripts/model_structure_manifest.py",
    "scripts/runtime_evidence.py",
    "scripts/gpu_telemetry.py",
    "scripts/collect_results.py",
    "scripts/freeze_lora_zero3_fix.py",
)


def _validate_live_scope() -> None:
    experiment = read_json(CONFIG_DIR / "experiment.json")
    scope = experiment.get("training_scope") or {}
    if scope.get("phase_id") != FORMAL_PHASE_ID:
        raise ValueError(
            f"training_scope.phase_id must be {FORMAL_PHASE_ID!r}, got "
            f"{scope.get('phase_id')!r}"
        )
    if scope.get("gpu_ids") != list(AUTHORIZED_GPU_IDS):
        raise ValueError("training_scope.gpu_ids drifted")
    if scope.get("exclusive_node_gpu_ids") != list(AUTHORIZED_GPU_IDS):
        raise ValueError("training_scope.exclusive_node_gpu_ids drifted")
    if scope.get("max_gpu_count") != MAX_GPU_COUNT:
        raise ValueError("training_scope.max_gpu_count drifted")
    if scope.get("gpu_counts") != [1, 2, 4]:
        raise ValueError("training_scope.gpu_counts drifted")
    if scope.get("global_batch_sizes") != [64]:
        raise ValueError("training_scope.global_batch_sizes drifted")
    live_model_ids = scope.get("model_ids") or []
    if set(live_model_ids) != set(SUPPLEMENT_MODEL_IDS):
        raise ValueError(
            "training_scope.model_ids must exactly cover the supplement models: "
            f"expected {sorted(SUPPLEMENT_MODEL_IDS)}, got {sorted(live_model_ids)}"
        )


def freeze() -> Path:
    rows = read_jsonl(FORMAL_QUEUE)
    design = read_json(DESIGN)
    # NOTE: the supplement is a phase inside the stage-1 v1 campaign (it
    # reuses _base_job, so rows carry the parent campaign_id).  Assert
    # phase and schema are pinned to the supplement values; assert campaign
    # is consistent across the queue without hard-coding which campaign.
    row_campaigns = {row.get("campaign_id") for row in rows}
    if (
        len(rows) != 20
        or len({str(row.get("job_id") or "") for row in rows}) != len(rows)
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or len(row_campaigns) != 1
        or any(row.get("phase_id") != FORMAL_PHASE_ID for row in rows)
    ):
        raise ValueError(
            f"supplement queue identity/count drifted: "
            f"n={len(rows)} campaigns={row_campaigns} "
            f"phases={ {r.get('phase_id') for r in rows} }"
        )
    row_campaign_id = next(iter(row_campaigns))
    if design.get("gpu_training_started") is not False:
        raise ValueError("campaign design says GPU training already started")
    for row in rows:
        validate_job(row)
        if row.get("requested_gpu_pool") != list(AUTHORIZED_GPU_IDS):
            raise ValueError(f"job GPU pool drifted: {row['job_id']}")
        if row.get("model_id") not in SUPPLEMENT_MODEL_IDS:
            raise ValueError(
                f"job {row['job_id']} model_id={row.get('model_id')!r} is outside "
                f"supplement scope {SUPPLEMENT_MODEL_IDS}"
            )
        if int(row.get("exact_total_tokens_per_sample") or 0) != 4096:
            raise ValueError(
                f"job {row['job_id']} is not a 4096-anchor sample "
                f"(exact_total_tokens_per_sample={row.get('exact_total_tokens_per_sample')})"
            )
    _validate_live_scope()

    hardware = probe_hardware(required_gpu_ids=AUTHORIZED_GPU_IDS)
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"GPU 0-6 are not an exact H800 pool: {hardware}")
    if hardware.get("selected_gpu_compute_processes"):
        raise RuntimeError("refusing to freeze while GPU 0-6 have compute processes")
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    # runtime_patch.all_passed=True is required.  With the build-time
    # provenance tree gone, validate_patch now reports mode=runtime_source_only
    # and derives all_passed purely from the installed DeepSpeed source (safe
    # per-parameter dtype present, unsafe first-parameter dtype absent).
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(FORMAL_QUEUE, rows, ROOT)

    scoped_paths = [ROOT / relative for relative in SCOPED_SOURCE_RELATIVE_PATHS]
    missing_sources = [str(path) for path in scoped_paths if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"approval source files are absent: {missing_sources}")
    provenance_binding = build_provenance_binding(ROOT, source_paths=scoped_paths)
    bound_files = {
        FORMAL_QUEUE,
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
        "ordered_job_payload_sha256": queue_binding["ordered_job_payload_sha256"],
        "job_payload_sha256": queue_binding["job_payload_sha256"],
    }
    approval_design: dict[str, Any] = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{row_campaign_id}:{FORMAL_PHASE_ID}",
        "campaign_design": {
            "path": str(DESIGN.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(DESIGN),
            "report_sha256": design.get("report_sha256"),
        },
        "stage": "formal",
        "stage_prerequisite": None,
        "file_sha256": file_manifest,
        "execution_order": [str(FORMAL_PHASE_ID)],
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
        / "approval_design_h800_hybrid_attention_dense_stage1_supplement_4096_formal_v1_candidate.json"
    )
    write_json(candidate, approval_design)
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
            {
                "candidate": str(candidate),
                "sha256": digest,
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
