#!/usr/bin/env python3
"""Freeze and optionally promote formal VL calibration after canary acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_frozen_video_decomposition_v1 import FORMAL_DESIGN as VIDEO_FORMAL_DESIGN, PROFILE_MANIFEST as VIDEO_PROFILE_MANIFEST
from prepare_h800_frozen_vl_combined_canary_v1 import AUTHORIZED_GPU_IDS
from prepare_h800_frozen_vl_combined_formal_v1 import CAMPAIGN_ID, CANARY_ACCEPTANCE, DESIGN, PHASE_ID, QUEUE, STAGING_CONFIG
from prepare_h800_frozen_vl_decomposition_v1 import FORMAL_DESIGN as IMAGE_FORMAL_DESIGN, INVENTORY, PROFILE_MANIFEST as IMAGE_PROFILE_MANIFEST
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch, resolve_dataset_dir, validate_job
from validate_h800_frozen_video_decomposition_v1 import validate as validate_video
from validate_h800_frozen_vl_decomposition_v1 import validate as validate_image


CANDIDATE = ARTIFACT_DIR / "approval_design_h800_frozen_vl_combined_formal_v1_candidate.json"
AUTHORIZATION = (
    "用户要求按既定实施顺序修复 Qwen3.5、Qwen2.5/Qwen3 VL 图片和视频实验并真正运行，"
    "且在 2026-08-11 明确要求继续推进。本 formal 授权只在 12/12 canary 通过后生效，"
    "只覆盖冻结的 87 个拟合/诊断作业；用户在 2026-08-12 明确授权使用物理 GPU 0-6；不得占用 GPU 7，"
    "不得抢占进程、扩展队列、启用 packing/offload，也不得把这些结果当作前瞻验收或发布证据。"
)

SCOPED_SOURCE_RELATIVE_PATHS = (
    "config/experiment.json",
    "config/hardware.json",
    "config/models.json",
    "scripts/approval_gate.py",
    "scripts/common.py",
    "scripts/promote_approval_candidate.py",
    "scripts/prepare_h800_frozen_vl_decomposition_v1.py",
    "scripts/validate_h800_frozen_vl_decomposition_v1.py",
    "scripts/prepare_h800_frozen_video_decomposition_v1.py",
    "scripts/validate_h800_frozen_video_decomposition_v1.py",
    "scripts/validate_h800_vl_video_full_decode_v1.py",
    "scripts/prepare_h800_frozen_vl_combined_formal_v1.py",
    "scripts/freeze_h800_frozen_vl_combined_formal_v1.py",
    "scripts/evaluate_h800_frozen_vl_combined_formal_v1.py",
    "scripts/run_job.py",
    "scripts/scheduler.py",
    "scripts/train_entry.py",
    "scripts/metrics_callback.py",
    "scripts/model_structure_manifest.py",
    "scripts/runtime_evidence.py",
    "scripts/gpu_telemetry.py",
    "scripts/collect_results.py",
)


def freeze() -> Path:
    acceptance = read_json(CANARY_ACCEPTANCE) if CANARY_ACCEPTANCE.is_file() else {}
    if (
        acceptance.get("schema")
        != "sft_h800_frozen_vl_combined_canary_acceptance/v1"
        or acceptance.get("all_passed") is not True
    ):
        raise RuntimeError("formal remains locked: exact combined canary has not passed")
    if validate_image().get("all_passed") is not True or validate_video().get("all_passed") is not True:
        raise RuntimeError("source decomposition static validation failed")
    rows = read_jsonl(QUEUE)
    design = read_json(DESIGN)
    unsigned = dict(design)
    claimed = unsigned.pop("report_sha256", None)
    if (
        len(rows) != 87
        or len({str(row["job_id"]) for row in rows}) != 87
        or claimed != sha256_json(unsigned)
        or design.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or design.get("queue", {}).get("ordered_job_ids")
        != [str(row["job_id"]) for row in rows]
    ):
        raise ValueError("combined formal design/queue binding drifted")
    for row in rows:
        validate_job(row)
        resolve_dataset_dir(row)
        if row.get("packing") is not False or row.get("offload") is not False:
            raise ValueError(f"job {row['job_id']} relaxed packing/offload contract")

    live_config = ROOT / "config/experiment.json"
    if sha256_file(live_config) != sha256_file(STAGING_CONFIG):
        raise ValueError("live experiment config is not the exact combined formal config")
    scope = read_json(live_config)["training_scope"]
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("gpu_ids") != AUTHORIZED_GPU_IDS
        or scope.get("exclusive_node_gpu_ids") != AUTHORIZED_GPU_IDS
        or scope.get("gpu_counts") != [1]
        or scope.get("max_gpu_count") != 1
    ):
        raise ValueError("combined formal GPU scope drifted")

    hardware = probe_hardware(required_gpu_ids=tuple(AUTHORIZED_GPU_IDS))
    if hardware.get("exact_h800_pool") is not True or hardware.get("selected_gpu_compute_processes"):
        raise RuntimeError("GPU 0-6 hardware/idle preflight failed")
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(QUEUE, rows, ROOT)
    scoped_paths = [ROOT / relative for relative in SCOPED_SOURCE_RELATIVE_PATHS]
    missing_sources = [str(path) for path in scoped_paths if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"formal approval sources are missing: {missing_sources}")
    provenance_binding = build_provenance_binding(ROOT, source_paths=scoped_paths)
    bound_files: set[Path] = {
        QUEUE,
        DESIGN,
        IMAGE_FORMAL_DESIGN,
        VIDEO_FORMAL_DESIGN,
        IMAGE_PROFILE_MANIFEST,
        VIDEO_PROFILE_MANIFEST,
        ARTIFACT_DIR / "h800_vl_video_full_decode_validation_v1.json",
        CANARY_ACCEPTANCE,
        INVENTORY,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        *scoped_paths,
    }
    for row in rows:
        bound_files.add(Path(row["data_path"]))
        bound_files.add(Path(row["dataset_profile_path"]))
        bound_files.add(Path(row["dataset_dir"]) / "dataset_info.json")
        if row.get("media_manifest_path"):
            bound_files.add(Path(row["media_manifest_path"]))
        overlay = row.get("environment_overlay") or {}
        if overlay:
            bound_files.add(Path(overlay["contract_path"]))
    missing = [str(path) for path in bound_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"formal approval-bound files are missing: {missing}")
    file_manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=str)
    }
    ids = [str(row["job_id"]) for row in rows]
    approval_design: dict[str, Any] = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}",
        "campaign_design": {
            "path": str(DESIGN.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(DESIGN),
            "report_sha256": design["report_sha256"],
        },
        "stage": "formal",
        "stage_prerequisite": {
            "path": str(CANARY_ACCEPTANCE.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(CANARY_ACCEPTANCE),
            "report_sha256": acceptance.get("report_sha256"),
            "all_passed": True,
        },
        "file_sha256": file_manifest,
        "execution_order": [PHASE_ID],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": AUTHORIZED_GPU_IDS,
        "max_gpu_count": 1,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": "use only idle physical GPUs 0-6; GPU7 and external processes are excluded",
        },
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": {
            "allowed_job_ids": ids,
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
    promotion = promote_candidate(
        candidate_path=candidate,
        expected_candidate_sha256=digest,
        project_root=ROOT,
        authorization=AUTHORIZATION,
        approved_by="user",
        promote=args.promote,
    )
    print(json.dumps({"candidate": str(candidate.resolve()), "sha256": digest, "promoted": bool(promotion.get("promoted")), "promotion": promotion}, ensure_ascii=False, indent=2))
    if promotion.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
