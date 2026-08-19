#!/usr/bin/env python3
"""Freeze and promote the business-data blind-test approval.

Locks the 0105 business-text queue, the models, the data and the current
runtime identity into an approval candidate.  GPU 0-6 must be idle.  The
run is a fresh generalization blind test: rows were never used to fit any
predictor coefficient.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from approval_gate import build_provenance_binding, build_queue_binding
from common import (
    ARTIFACT_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from prepare_h800_business_blind_v1 import (
    AUTHORIZED_GPU_IDS,
    CAMPAIGN_ID,
    DESIGN_SCHEMA,
)
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch

AUTHORIZATION = (
    "用户在 2026-08-16 授权跑 0105 短视频业务文本的业务盲测泛化实验。"
    "覆盖 qwen3/qwen3.5/qwen3.6 的 LoRA SFT、packing 开/关、ZeRO-2/3、1-2 卡、"
    "GC 开关、cutoff 2048/4096。仅 GPU 0-6，最多 2 卡/任务，不允许抢占其他任务、"
    "扩展模型/数据/作业或把 OOM 当作精确峰值。本批用于泛化盲测，不用于校准或发布。"
)
# Approval-gate scope must match config/experiment.json.  Individual jobs in
# this campaign remain capped at two GPUs by the frozen queue and policy below.
MAX_GPU_COUNT = 4

SCOPED_SOURCE_RELATIVE_PATHS = (
    "scripts/approval_gate.py",
    "scripts/common.py",
    "scripts/promote_approval_candidate.py",
    "scripts/run_job.py",
    "scripts/scheduler.py",
    "scripts/train_entry.py",
    "scripts/prepare_h800_business_blind_v1.py",
    "scripts/freeze_h800_business_blind_v1.py",
    "scripts/hybrid_attention_memory_features.py",
)


def freeze() -> Path:
    import prepare_h800_business_blind_v1 as prep
    queue_path = Path(prep.QUEUE_PATH).resolve()
    rows = read_jsonl(queue_path)
    if not rows:
        raise ValueError("blind queue is empty")
    design = read_json(prep.DESIGN_PATH)
    if design.get("schema") != DESIGN_SCHEMA or design.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("blind design schema/campaign drifted")
    if design.get("gpu_training_started") is not False:
        raise ValueError("blind design says GPU training already started")

    hardware = probe_hardware(required_gpu_ids=list(AUTHORIZED_GPU_IDS))
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
    missing = [str(p) for p in scoped_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"approval source files are absent: {missing}")
    provenance_binding = build_provenance_binding(ROOT, source_paths=scoped_paths)

    bound_files = {
        queue_path,
        Path(prep.DESIGN_PATH).resolve(),
        ROOT / "artifacts" / "provenance.json",
        *(p.resolve() for p in scoped_paths),
    }
    missing = [str(p) for p in bound_files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files are absent: {missing}")
    file_manifest = {
        str(p.relative_to(ROOT.resolve())): sha256_file(p)
        for p in sorted(bound_files, key=str)
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
    approval_design: dict = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:business_blind",
        "campaign_design": {
            "path": str(
                Path(prep.DESIGN_PATH).resolve().relative_to(ROOT.resolve())
            ),
            "sha256": sha256_file(prep.DESIGN_PATH),
        },
        "stage": "blind",
        "stage_prerequisite": None,
        "file_sha256": file_manifest,
        "execution_order": ["h800_business_blind_v1"],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": list(AUTHORIZED_GPU_IDS),
        "max_gpu_count": MAX_GPU_COUNT,
        "scheduler_execution": {
            "join_busy_pool": False,
            "preemption_allowed": False,
            "policy": "use only idle GPU 0-6; disjoint 1/2-card masks; max 2 cards/job",
        },
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": queue_stage,
    }
    candidate = ARTIFACT_DIR / "approval_design_h800_business_blind_v1_candidate.json"
    write_json(candidate, approval_design)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument(
        "--source",
        choices=("datatest", "inference", "inference2"),
        default="datatest",
    )
    args = parser.parse_args()
    import prepare_h800_business_blind_v1 as prep
    spec = prep.SOURCES[args.source]
    prep.TEXT_SOURCE = prep.BLIND_DATA_DIR / "text" / spec["sft"]
    prep.TEXT_PROFILE = prep.BLIND_DATA_DIR / "text" / spec["profile"]
    prep.DATASET_ID = spec["dataset_id"]
    prep.QUEUE_PATH = prep.MATRIX_DIR / f"h800_business_{args.source}_v1.jsonl"
    prep.DESIGN_PATH = prep.ARTIFACT_DIR / f"h800_business_{args.source}_design_v1.json"
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
    checks = report.get("checks") or {}
    failed_checks = {k: v for k, v in checks.items() if v is False}
    prov = report.get("provenance") or {}
    prov_checks = prov.get("checks") or {}
    prov_failed = {k: v for k, v in prov_checks.items() if v is False}
    summary = {
        "candidate": str(candidate),
        "sha256": digest,
        "promote_flag": bool(args.promote),
        "mode": report.get("mode"),
        "all_passed": report.get("all_passed"),
        "installed_approval_sha256": report.get("installed_approval_sha256"),
        "failed_checks": failed_checks,
        "failed_provenance_checks": prov_failed,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
