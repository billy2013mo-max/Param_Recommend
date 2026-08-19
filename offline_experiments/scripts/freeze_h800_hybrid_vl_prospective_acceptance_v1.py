#!/usr/bin/env python3
"""Freeze and optionally promote the 27-job hybrid/VL prospective acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_hybrid_vl_prospective_acceptance_v1 import (
    CAMPAIGN_ID,
    COMBINED_QUEUE,
    DESIGN,
    FROZEN_PREDICTIONS,
    HYBRID_QUEUE,
    PHASE_ID,
    SAFETY_ARTIFACT,
    SCHEMA,
    VL_QUEUE,
)
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch, resolve_dataset_dir, validate_job


AUTHORIZED_GPU_IDS = list(range(7))
EXPECTED_JOBS = 27
AUTHORIZATION = (
    "用户在 2026-08-17 要求继续完成四步：用独立来源实测单侧显存上界与排序，"
    "通过后才开启自动准入和排序。本批准仅允许运行冻结的 27 个前瞻验收任务："
    "9 个非 packing 混合模型任务和 18 个冻结视觉塔/投影层的图像视觉语言任务；"
    "仅 GPU 0-6，每任务最多 2 卡，不重试 OOM，不抢占其他进程。"
    "OOM 只作为右删失下界；结果不得回填拟合，不得自动发布，验收不通过不得开启自动推荐。"
)
CANDIDATE = ARTIFACT_DIR / "approval_design_h800_hybrid_vl_prospective_acceptance_v1_candidate.json"

SCOPED_SOURCE_RELATIVE_PATHS = (
    "config/experiment.json",
    "config/hardware.json",
    "config/models.json",
    "config/deepspeed/ds_z2.json",
    "config/deepspeed/ds_z3.json",
    "scripts/approval_gate.py",
    "scripts/common.py",
    "scripts/promote_approval_candidate.py",
    "scripts/prepare_h800_hybrid_vl_prospective_acceptance_v1.py",
    "scripts/freeze_h800_hybrid_vl_prospective_acceptance_v1.py",
    "scripts/evaluate_h800_hybrid_vl_prospective_acceptance_v1.py",
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
    if rows != [*hybrid, *vl] or len(hybrid) != 9 or len(vl) != 18 or len(rows) != EXPECTED_JOBS:
        raise ValueError("combined queue is not the exact ordered 9+18 prospective design")
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
            or int(row.get("gpu_count", 0)) not in {1, 2}
            or row.get("requested_gpu_pool") != AUTHORIZED_GPU_IDS
        ):
            raise ValueError(f"job escaped the prospective scope: {row.get('job_id')}")
        validate_job(row)
        resolve_dataset_dir(row)
        if (ROOT / "results" / str(row["job_id"]) / "status.json").is_file():
            raise RuntimeError(f"outcome exists before approval freeze: {row['job_id']}")

    design = read_json(DESIGN)
    predictions = read_json(FROZEN_PREDICTIONS)
    _validate_internal_hash(design, name="prospective design")
    _validate_internal_hash(predictions, name="frozen predictions")
    queues = design.get("queues") or {}
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
        or predictions.get("schema") != "sft_h800_hybrid_vl_prospective_frozen_predictions/v1"
        or predictions.get("safety_artifact", {}).get("sha256") != sha256_file(SAFETY_ARTIFACT)
        or len(predictions.get("hybrid", {}).get("predictions") or []) != 9
        or len(predictions.get("vl_image", {}).get("predictions") or []) != 18
    ):
        raise ValueError("prospective design or frozen prediction binding drifted")
    disjoint = design.get("source_disjointness") or {}
    if (
        disjoint.get("hybrid_fit_sources_exclude_0105_inference") is not True
        or disjoint.get("vl_fit_sources") != ["pzfj38_v113", "zltbjg_v2"]
        or disjoint.get("vl_prospective_source") != "qype19_v7"
    ):
        raise ValueError("source-disjoint contract is absent")
    return rows


def freeze(candidate_path: Path) -> Path:
    rows = _validate_inputs()
    hardware = probe_hardware(required_gpu_ids=tuple(AUTHORIZED_GPU_IDS))
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError("GPU 0-6 are not an exact H800 pool")
    if hardware.get("selected_gpu_compute_processes"):
        raise RuntimeError("one or more authorized H800s are busy")

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
        or int(scope.get("max_gpu_count", 0)) < 2
    ):
        raise ValueError("live experiment config cannot execute the frozen 1/2-GPU queue")

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
        SAFETY_ARTIFACT,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "data" / "dataset_info.json",
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
            "join_busy_pool": False,
            "preemption_allowed": False,
            "policy": "parallel disjoint one/two-GPU masks across GPU 0-6",
            "queue_max_gpu_count": 2,
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
