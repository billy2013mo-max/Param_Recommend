#!/usr/bin/env python3
"""Freeze and optionally promote the exact six-GPU calibration queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch


AUTHORIZATION = (
    "用户明确要求使用当前空闲的六张 H800 GPU 0、1、4、5、6、7 启动 LoRA 显存补充实验；"
    "本次只允许先执行已物化且绑定的 20 个 profile-aware 校准作业，不得占用 GPU 2、3，"
    "不得终止或抢占外部进程，不得复用上一批 approval 或扩展到未批准任务。"
)
CAMPAIGN_ID = "h800_profile_aware_memory_calibration_20260802_v1"
PHASE_ID = "h800_profile_aware_memory_calibration_v1"
JOB_SCHEMA = "sft_h800_profile_aware_memory_calibration_job/v1"
DEFAULT_QUEUE = ROOT / "matrix" / "h800_profile_aware_memory_calibration_jobs_v1.jsonl"
DEFAULT_CAMPAIGN_DESIGN = ARTIFACT_DIR / "h800_profile_aware_memory_calibration_design_v1.json"
DEFAULT_QUEUE_MANIFEST = ARTIFACT_DIR / "h800_profile_aware_memory_calibration_queue_manifest_v1.json"
AUTHORIZED_GPU_IDS = (0, 1, 4, 5, 6, 7)


def freeze(queue: Path, campaign_design: Path, queue_manifest: Path) -> Path:
    rows = read_jsonl(queue)
    if (
        len(rows) != 20
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
    ):
        raise ValueError("queue is not the exact 20-job profile-aware calibration campaign")
    partitions = [row.get("calibration_partition") or {} for row in rows]
    if any(
        part.get("role") != "calibration"
        or part.get("policy") != "profile_padding_stratified_scenario_disjoint_v1"
        or not part.get("split_unit_id")
        for part in partitions
    ):
        raise ValueError("every queue row must carry the frozen calibration partition")

    experiment = read_json(ROOT / "config" / "experiment.json")
    scope = experiment["training_scope"]
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("gpu_ids") != list(AUTHORIZED_GPU_IDS)
        or scope.get("exclusive_node_gpu_ids") != list(AUTHORIZED_GPU_IDS)
        or scope.get("max_gpu_count") != 2
        or scope.get("gpu_counts") != [1, 2]
    ):
        raise ValueError("experiment config is not the exact GPU-4,5 calibration scope")

    hardware = probe_hardware(required_gpu_ids=AUTHORIZED_GPU_IDS)
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(
            f"GPU pool {list(AUTHORIZED_GPU_IDS)} is not exact H800: {hardware}"
        )

    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(queue, rows, ROOT)
    provenance_binding = build_provenance_binding(ROOT)

    bound_files = {
        queue,
        campaign_design,
        queue_manifest,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "data" / "dataset_info.json",
    }
    campaign = read_json(campaign_design)
    for binding in (campaign.get("source_bindings") or {}).values():
        bound_files.add(Path(str(binding["path"])))
    for row in rows:
        bound_files.add(Path(str(row["data_path"])))
        bound_files.add(Path(str(row["dataset_profile_path"])))
    missing = [str(path) for path in bound_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files are absent: {missing}")
    manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=lambda value: str(value))
    }
    ids = [str(row["job_id"]) for row in rows]
    stage_binding = {
        "allowed_job_ids": ids,
        "queue_path": queue_binding["path"],
        "queue_sha256": queue_binding["sha256"],
        "ordered_job_ids": queue_binding["ordered_job_ids"],
        "ordered_job_payload_sha256": queue_binding["ordered_job_payload_sha256"],
        "job_payload_sha256": queue_binding["job_payload_sha256"],
    }
    design = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}",
        "campaign_design": {
            "path": str(campaign_design.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(campaign_design),
        },
        "file_sha256": manifest,
        "execution_order": [PHASE_ID],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": list(AUTHORIZED_GPU_IDS),
        "max_gpu_count": 2,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": (
                "start only on idle members of the exact GPU 0,1,4,5,6,7 pool; join a busy member "
                "only after its external process exits naturally"
            ),
        },
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": stage_binding,
    }
    candidate = (
        ARTIFACT_DIR
        / "approval_design_h800_profile_aware_memory_calibration_six_gpu_v2_candidate.json"
    )
    write_json(candidate, design)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--campaign-design", type=Path, default=DEFAULT_CAMPAIGN_DESIGN)
    parser.add_argument("--queue-manifest", type=Path, default=DEFAULT_QUEUE_MANIFEST)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = freeze(
        args.queue.resolve(),
        args.campaign_design.resolve(),
        args.queue_manifest.resolve(),
    )
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
