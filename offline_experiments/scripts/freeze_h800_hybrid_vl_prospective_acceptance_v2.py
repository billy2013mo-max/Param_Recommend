#!/usr/bin/env python3
"""Freeze and optionally promote the 54-job hybrid/VL prospective acceptance V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_hybrid_vl_prospective_acceptance_v2 import (
    CAMPAIGN_ID,
    COMBINED_QUEUE,
    DESIGN,
    FROZEN_PREDICTIONS,
    HYBRID_QUEUE,
    PHASE_ID,
    SCHEMA,
    UPPER_V3,
    VL_QUEUE,
)
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch, resolve_dataset_dir, validate_job


AUTHORIZED_GPU_IDS = list(range(8))
EXPECTED_JOBS = 54
EXPECTED_HYBRID = 36
EXPECTED_VL = 18
AUTHORIZATION = (
    "用户在 2026-08-18 要求运行 V2 源隔离前瞻验收采集：36 个非 packing 混合模型任务"
    "（qwen3p5_4b/9b、qwen3p6_27b，ZeRO0/2/3，1/2/4 卡，重点 9b/27b 多卡 ZeRO3）"
    "和 18 个冻结视觉塔图像 VL 任务（qwen2p5_vl_3b/qwen3_vl_4b/qwen3p5_4b），"
    "数据源为全新隔离数据 0105_inference2 与 blind_data/vl。"
    "批准 GPU 0-7；部分 GPU 可能被外部进程占用，调度器只使用当前空闲的 GPU，"
    "OOM 作为右删失下界不重试，不抢占进程。"
    "结果不得回填拟合；验收通过前不开启自动准入/自动发布。"
)
CANDIDATE = ARTIFACT_DIR / "approval_design_h800_hybrid_vl_prospective_acceptance_v2_candidate.json"

SCOPED_SOURCE_RELATIVE_PATHS = (
    "config/experiment.json",
    "config/hardware.json",
    "config/models.json",
    "config/deepspeed/ds_z2.json",
    "config/deepspeed/ds_z3.json",
    "scripts/approval_gate.py",
    "scripts/common.py",
    "scripts/promote_approval_candidate.py",
    "scripts/prepare_h800_hybrid_vl_prospective_acceptance_v2.py",
    "scripts/freeze_h800_hybrid_vl_prospective_acceptance_v2.py",
    "scripts/evaluate_h800_hybrid_vl_prospective_acceptance_v2.py",
    "scripts/fit_h800_hybrid_attention_dense_stage2_v1.py",
    "scripts/fit_h800_hybrid_vl_safety_upper_v3.py",
    "scripts/hybrid_attention_memory_features_v2.py",
    "scripts/hybrid_memory_bridge.py",
    "scripts/build_h800_blind_vl_sft_v1.py",
    "scripts/vl_resource_features.py",
    "scripts/h800_resource_predictor.py",
    "scripts/h800_unified_v3_throughput_v5_predictor.py",
    "scripts/throughput_predictor.py",
    "scripts/run_job.py",
    "scripts/scheduler.py",
    "scripts/train_entry.py",
    "scripts/metrics_callback.py",
    "scripts/runtime_evidence.py",
    "scripts/gpu_telemetry.py",
    "scripts/collect_results.py",
)


def _validate_internal_hash(payload: dict[str, Any], *, name: str) -> None:
    unsigned = dict(payload)
    expected = unsigned.pop("report_sha256", None)
    if expected != sha256_json(unsigned):
        raise ValueError(f"{name} internal hash mismatch")


def _validate_inputs() -> list[dict[str, Any]]:
    hybrid = read_jsonl(HYBRID_QUEUE)
    vl = read_jsonl(VL_QUEUE)
    rows = read_jsonl(COMBINED_QUEUE)
    if (
        rows != [*hybrid, *vl]
        or len(hybrid) != EXPECTED_HYBRID
        or len(vl) != EXPECTED_VL
        or len(rows) != EXPECTED_JOBS
    ):
        raise ValueError("combined queue is not the exact ordered 36+18 prospective design")
    ids = [str(row.get("job_id") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("prospective job IDs must be non-empty and unique")
    for row in rows:
        if (
            row.get("campaign_id") != CAMPAIGN_ID
            or row.get("phase_id") != PHASE_ID
            or row.get("evidence_role") != "prospective_acceptance_frozen_before_outcomes"
            or row.get("packing") is not False
            or row.get("offload") is not False
            or int(row.get("gpu_count", 0)) not in {1, 2, 4}
            or row.get("requested_gpu_pool") != AUTHORIZED_GPU_IDS
        ):
            raise ValueError(f"job escaped the prospective scope: {row.get('job_id')}")
        validate_job(row)
        resolve_dataset_dir(row)
        if (ROOT / "results" / str(row["job_id"]) / "status.json").is_file():
            raise RuntimeError(f"outcome exists before approval freeze: {row['job_id']}")

    design = read_json(DESIGN)
    predictions = read_json(FROZEN_PREDICTIONS)
    _validate_internal_hash(design, name="prospective design V2")
    _validate_internal_hash(predictions, name="frozen predictions V2")
    queues = design.get("queues") or {}
    disjoint = design.get("source_disjointness") or {}
    if (
        design.get("schema") != SCHEMA
        or design.get("campaign_id") != CAMPAIGN_ID
        or design.get("phase_id") != PHASE_ID
        or design.get("gpu_training_started") is not False
        or design.get("scope", {}).get("automatic_execution_allowed") is not False
        or queues.get("combined", {}).get("sha256") != sha256_file(COMBINED_QUEUE)
        or queues.get("hybrid", {}).get("sha256") != sha256_file(HYBRID_QUEUE)
        or queues.get("vl_image", {}).get("sha256") != sha256_file(VL_QUEUE)
        or design.get("frozen_predictions", {}).get("sha256") != sha256_file(FROZEN_PREDICTIONS)
        or predictions.get("schema") != "sft_h800_hybrid_vl_prospective_frozen_predictions/v2"
        or predictions.get("safety_upper", {}).get("sha256") != sha256_file(UPPER_V3)
        or len(predictions.get("hybrid", {}).get("predictions") or []) != EXPECTED_HYBRID
        or len(predictions.get("vl_image", {}).get("predictions") or []) != EXPECTED_VL
        or disjoint.get("hybrid_fit_sources_exclude_0105_inference2") is not True
        or disjoint.get("vl_fit_sources") != ["pzfj38_v113", "zltbjg_v2", "qype19_v7"]
        or disjoint.get("vl_prospective_source") != "blind_data/vl synthetic-prompt frames"
    ):
        raise ValueError("prospective design or frozen prediction binding drifted")
    return rows


def freeze(candidate_path: Path) -> Path:
    rows = _validate_inputs()
    hardware = probe_hardware(required_gpu_ids=tuple(AUTHORIZED_GPU_IDS))
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError("GPU 0-7 are not an exact H800 pool")

    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")

    scope = read_json(ROOT / "config" / "experiment.json")["training_scope"]
    if (
        scope.get("gpu_ids") != AUTHORIZED_GPU_IDS
        or scope.get("exclusive_node_gpu_ids") != AUTHORIZED_GPU_IDS
        or 1 not in scope.get("gpu_counts", [])
        or 2 not in scope.get("gpu_counts", [])
        or 4 not in scope.get("gpu_counts", [])
        or int(scope.get("max_gpu_count", 0)) < 4
    ):
        raise ValueError("live experiment config cannot execute the frozen 1/2/4-GPU queue")

    queue_binding = build_queue_binding(COMBINED_QUEUE, rows, ROOT)
    scoped_paths = [ROOT / name for name in SCOPED_SOURCE_RELATIVE_PATHS]
    missing_sources = sorted(str(path) for path in scoped_paths if not path.is_file())
    if missing_sources:
        raise FileNotFoundError(f"scoped execution sources are absent: {missing_sources}")
    provenance_binding = build_provenance_binding(ROOT, source_paths=scoped_paths)

    bound_files: set[Path] = {
        COMBINED_QUEUE,
        HYBRID_QUEUE,
        VL_QUEUE,
        DESIGN,
        FROZEN_PREDICTIONS,
        UPPER_V3,
        ARTIFACT_DIR / "h800_hybrid_memory_artifact_v2.json",
        ARTIFACT_DIR / "h800_hybrid_attention_dense_stage2_fit_report_v1.json",
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "data" / "dataset_info.json",
        ROOT / "data" / "blind_vl_prospective_v1" / "registry" / "dataset_info.json",
        ROOT / "data" / "blind_vl_prospective_v1" / "blind_vl_images.jsonl",
        *scoped_paths,
    }
    for row in rows:
        for field in (
            "data_path",
            "dataset_profile_path",
            "media_manifest_path",
            "declared_model_manifest_path",
        ):
            value = row.get(field)
            if value:
                bound_files.add(Path(str(value)))
        overlay = row.get("environment_overlay") or {}
        if overlay.get("contract_path"):
            bound_files.add(Path(str(overlay["contract_path"])))
    missing = sorted(str(path) for path in bound_files if not path.is_file())
    if missing:
        raise FileNotFoundError(f"approval-bound files are absent: {missing}")
    manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=str)
        if path.resolve().is_relative_to(ROOT.resolve())
    }
    ids = [str(row["job_id"]) for row in rows]
    stage = {
        "allowed_job_ids": ids,
        "queue_path": queue_binding["path"],
        "queue_sha256": queue_binding["sha256"],
        "ordered_job_ids": queue_binding["ordered_job_ids"],
        "ordered_job_payload_sha256": queue_binding["ordered_job_payload_sha256"],
        "job_payload_sha256": queue_binding["job_payload_sha256"],
    }
    approval_design = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}",
        "authorization": AUTHORIZATION,
        "campaign_design": {"path": str(DESIGN.relative_to(ROOT)), "sha256": sha256_file(DESIGN)},
        "frozen_predictions": {"path": str(FROZEN_PREDICTIONS.relative_to(ROOT)), "sha256": sha256_file(FROZEN_PREDICTIONS)},
        "file_sha256": manifest,
        "execution_order": [PHASE_ID],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": AUTHORIZED_GPU_IDS,
        "max_gpu_count": int(scope["max_gpu_count"]),
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": (
                "partial-pool adaptive: allocate only currently idle GPUs; "
                "pool expands as external processes vacate 4-7"
            ),
            "queue_max_gpu_count": 4,
        },
        "oom_policy": "no_retry; OOM is a right-censored lower bound",
        "publication_policy": "prospective acceptance only; no automatic promotion",
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": stage,
    }
    write_json(candidate_path, approval_design)
    return candidate_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = freeze(args.candidate.resolve())
    report = promote_candidate(
        candidate_path=candidate,
        expected_candidate_sha256=sha256_file(candidate),
        project_root=ROOT,
        authorization=AUTHORIZATION,
        approved_by="user",
        promote=args.promote,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
