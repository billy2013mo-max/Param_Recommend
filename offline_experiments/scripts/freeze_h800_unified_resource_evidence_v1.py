#!/usr/bin/env python3
"""Freeze and optionally promote the unified-resource evidence queue.

The freezer accepts either the complete 220-job queue or an exact ordered
subset generated for resume.  A subset is authorized only when every payload
is byte-equivalent (under canonical JSON hashing) to its row in the sealed full
queue.  Creating the candidate never launches GPU work; ``--promote`` only
installs the approval that the scheduler subsequently consumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
from prepare_h800_prospective_holdout import probe_hardware
from prepare_h800_unified_resource_evidence_v1 import (
    CAMPAIGN_ID,
    DESIGN_SCHEMA,
    JOB_SCHEMA,
    PHASE_ID,
)
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch, validate_job

AUTHORIZATION = (
    "用户在 2026-08-09 要求完成统一资源推荐器的实验审查和完整实验脚本，并明确说明本机 "
    "GPU 0-7 共 8 张卡均可作为该实验的资源池。本 approval 仅绑定已冻结的统一模型证据作业；"
    "机制字段只能作为单一模型的输入，OOM 只能作为右删失下界，重复测量不得冒充独立样本。"
    "脚本生成和 approval promotion 均不等于启动训练；只有 launch 脚本的显式 --execute 才会启动。"
)
EXPECTED_JOB_COUNT = 220
AUTHORIZED_GPU_IDS = list(range(8))
MAX_GPU_COUNT = 4

FULL_QUEUE = ROOT / "matrix" / "h800_unified_resource_evidence_jobs_v1.jsonl"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_unified_resource_evidence_design_v1.json"
DEFAULT_QUEUE_MANIFEST = (
    ARTIFACT_DIR / "h800_unified_resource_evidence_queue_manifest_v1.json"
)
CANDIDATE = (
    ARTIFACT_DIR / "approval_design_h800_unified_resource_evidence_v1_candidate.json"
)

SCOPED_SOURCE_RELATIVE_PATHS = (
    "config/experiment.json",
    "config/hardware.json",
    "config/models.json",
    "config/deepspeed/ds_z2.json",
    "config/deepspeed/ds_z3.json",
    "scripts/approval_gate.py",
    "scripts/common.py",
    "scripts/promote_approval_candidate.py",
    "scripts/prepare_h800_unified_resource_evidence_v1.py",
    "scripts/freeze_h800_unified_resource_evidence_v1.py",
    "scripts/evaluate_h800_unified_resource_evidence_v1.py",
    "scripts/run_job.py",
    "scripts/scheduler.py",
    "scripts/train_entry.py",
    "scripts/metrics_callback.py",
    "scripts/runtime_evidence.py",
    "scripts/gpu_telemetry.py",
    "scripts/collect_results.py",
    "unified_resource_staging/launch_h800_unified_resource_evidence_v1.py",
)


def _validate_row(row: dict[str, Any]) -> None:
    if row.get("schema") != JOB_SCHEMA:
        raise ValueError(f"unexpected job schema: {row.get('schema')}")
    if row.get("campaign_id") != CAMPAIGN_ID or row.get("phase_id") != PHASE_ID:
        raise ValueError(f"job lies outside the unified campaign: {row.get('job_id')}")
    if row.get("train_type") not in {"lora", "full"}:
        raise ValueError("only LoRA and FULL SFT belong to this campaign")
    if row.get("offload") is not False:
        raise ValueError("offload is outside the frozen domain")
    if int(row.get("gpu_count", 0)) not in {1, 2, 4}:
        raise ValueError(f"unsupported gpu_count: {row.get('gpu_count')}")
    if row.get("calibration_partition", {}).get("policy") != (
        "unified_model_source_grouped_v1"
    ):
        raise ValueError("job does not declare the unified grouped-split policy")
    validate_job(row)


def _validate_queue(queue_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full = read_jsonl(FULL_QUEUE)
    if len(full) != EXPECTED_JOB_COUNT:
        raise ValueError(f"sealed full queue must contain {EXPECTED_JOB_COUNT} jobs")
    rows = read_jsonl(queue_path)
    if not rows:
        raise ValueError("execution queue is empty")
    ids = [str(row.get("job_id") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("execution queue job IDs must be non-empty and unique")
    full_by_id = {str(row["job_id"]): row for row in full}
    unknown = sorted(set(ids) - set(full_by_id))
    if unknown:
        raise ValueError(f"resume queue contains unknown jobs: {unknown}")
    for row in full:
        _validate_row(row)
    for row in rows:
        _validate_row(row)
        if row != full_by_id[str(row["job_id"])]:
            raise ValueError(f"job payload drifted from full queue: {row['job_id']}")
    positions = {str(row["job_id"]): index for index, row in enumerate(full)}
    if [positions[job_id] for job_id in ids] != sorted(positions[job_id] for job_id in ids):
        raise ValueError("resume queue must preserve the sealed full-queue order")
    return rows, full


def _validate_scope(full_rows: list[dict[str, Any]]) -> None:
    scope = read_json(ROOT / "config" / "experiment.json")["training_scope"]
    zero_by_gpu: dict[str, set[str]] = {}
    for row in full_rows:
        zero_by_gpu.setdefault(str(int(row["gpu_count"])), set()).add(str(row["zero"]))
    expected = {
        "phase_id": PHASE_ID,
        "gpu_ids": AUTHORIZED_GPU_IDS,
        "exclusive_node_gpu_ids": AUTHORIZED_GPU_IDS,
        "max_gpu_count": MAX_GPU_COUNT,
        "gpu_counts": sorted({int(row["gpu_count"]) for row in full_rows}),
        "model_ids": sorted({str(row["model_id"]) for row in full_rows}),
        "zero_by_gpu_count": {
            key: sorted(values) for key, values in sorted(zero_by_gpu.items())
        },
        "global_batch_sizes": sorted({int(row["target_gbs"]) for row in full_rows}),
        "gradient_checkpointing": sorted(
            {bool(row["gradient_checkpointing"]) for row in full_rows}
        ),
    }
    for key, want in expected.items():
        got = scope.get(key)
        if got != want:
            raise ValueError(
                f"experiment scope mismatch on {key}: expected={want!r}, got={got!r}"
            )


def _validate_campaign_design(full_rows: list[dict[str, Any]]) -> dict[str, Any]:
    campaign = read_json(DEFAULT_DESIGN)
    if (
        campaign.get("schema") != DESIGN_SCHEMA
        or campaign.get("campaign_id") != CAMPAIGN_ID
        or campaign.get("phase_id") != PHASE_ID
    ):
        raise ValueError("campaign design identity mismatch")
    for flag in ("gpu_training_started", "execution_authorized", "publication_allowed"):
        if campaign.get(flag) is not False:
            raise ValueError(f"campaign design must still have {flag}=false")
    unsigned = dict(campaign)
    expected_hash = unsigned.pop("report_sha256", None)
    if expected_hash != sha256_json(unsigned):
        raise ValueError("campaign design internal hash mismatch")
    if campaign.get("ordered_job_ids") != [str(row["job_id"]) for row in full_rows]:
        raise ValueError("campaign design job order drifted")
    if campaign.get("ordered_job_payload_sha256") != sha256_json(full_rows):
        raise ValueError("campaign design does not bind the full queue payload")
    contract = campaign.get("fit_contract") or {}
    required = (
        "single_shared_model",
        "mechanism_fields_are_features_not_routes",
        "oom_rows_are_right_censored_lower_bounds",
        "critical_seed_rows_collapse_to_four_profile_scenarios",
        "packing_repeats_collapse_by_physical_arm_for_memory_center",
        "source_grouped_cross_validation_required",
    )
    if not all(contract.get(key) is True for key in required):
        raise ValueError("campaign design weakened a required fit contract")
    return campaign


def freeze(queue_path: Path) -> Path:
    rows, full_rows = _validate_queue(queue_path)
    _validate_scope(full_rows)
    _validate_campaign_design(full_rows)

    hardware = probe_hardware(required_gpu_ids=tuple(AUTHORIZED_GPU_IDS))
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"authorized GPUs are not an exact H800 pool: {hardware}")
    if hardware.get("selected_gpu_compute_processes"):
        raise RuntimeError("refusing to freeze while one of GPU 0-7 is busy")

    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")

    queue_binding = build_queue_binding(queue_path, rows, ROOT)
    scoped_paths = [ROOT / name for name in SCOPED_SOURCE_RELATIVE_PATHS]
    missing_sources = sorted(str(path) for path in scoped_paths if not path.is_file())
    if missing_sources:
        raise FileNotFoundError(f"scoped execution sources are absent: {missing_sources}")
    provenance_binding = build_provenance_binding(ROOT, source_paths=scoped_paths)

    bound_files = {
        queue_path,
        FULL_QUEUE,
        DEFAULT_DESIGN,
        DEFAULT_QUEUE_MANIFEST,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "data" / "dataset_info.json",
        *scoped_paths,
    }
    for row in full_rows:
        bound_files.add(Path(row["data_path"]))
        bound_files.add(Path(row["dataset_profile_path"]))
    missing = sorted(str(path) for path in bound_files if not path.is_file())
    if missing:
        raise FileNotFoundError(f"approval-bound files are absent: {missing}")
    manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=str)
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
    design = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}",
        "campaign_design": {
            "path": str(DEFAULT_DESIGN.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(DEFAULT_DESIGN),
        },
        "file_sha256": manifest,
        "execution_order": [PHASE_ID],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": AUTHORIZED_GPU_IDS,
        "max_gpu_count": MAX_GPU_COUNT,
        "scheduler_execution": {
            "join_busy_pool": False,
            "preemption_allowed": False,
            "policy": (
                "start only when GPU 0-7 are idle; pack disjoint 1/2-card jobs, "
                "keep the repository's conservative pool-exclusive 4-card policy"
            ),
        },
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": stage,
        "resume_subset": {
            "is_subset": len(rows) < len(full_rows),
            "jobs_in_execution_queue": len(rows),
            "jobs_in_full_design": len(full_rows),
            "subset_policy": "exact ordered payload subset of sealed full queue",
        },
    }
    write_json(CANDIDATE, design)
    return CANDIDATE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=FULL_QUEUE)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="replace the live approval; never launches training",
    )
    args = parser.parse_args()
    candidate = freeze(args.queue)
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
                "queue": str(args.queue),
                "job_count": len(read_jsonl(args.queue)),
                "authorized_gpu_ids": AUTHORIZED_GPU_IDS,
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
