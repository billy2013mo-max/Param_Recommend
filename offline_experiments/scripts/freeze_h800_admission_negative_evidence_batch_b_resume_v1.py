#!/usr/bin/env python3
"""Freeze and optionally promote the batch B RESUME approval.

The scheduler has no resume mode: relaunching the full batch would re-run the 31
jobs that already hold usable terminal evidence.  So this approval binds only the
11 jobs still missing a calibration-eligible outcome -- all single-GPU, so the
training scope narrows to gpu_counts=[1] and max_gpu_count=1.

Confined to GPUs 0-3 at the user's instruction so 4-7 stay available to others.

Creating the candidate is always safe; only ``--promote`` replaces the live
approval, and neither path launches GPU training.
"""

from __future__ import annotations

import argparse
import json
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
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch


AUTHORIZATION = (
    "用户在 2026-08-08 明确批准续跑：仅在 GPU 0-3 上运行批次 B 剩余的 11 个单卡作业"
    "（FULL 1 卡/检查点开 6 个 + LoRA 1 卡/检查点开 5 个）。"
    "GPU 4-7 留给外部使用，不得占用。本 approval 只允许执行冻结的这 42 个作业，"
    "不得抢占外部进程、不得复用旧 campaign 的 approval、不得扩展到 30 个 4 卡作业或已完成的 31 个作业，"
    "也不得放宽任何显存安全阈值。本批故意压到 OOM 以制造失败样本，OOM 是预期产物。"
)
CAMPAIGN_ID = "h800_admission_negative_evidence_20260807_v1"
PHASE_ID = "h800_admission_negative_evidence_v1"
JOB_SCHEMA = "sft_h800_admission_negative_evidence_job/v1"
EXPECTED_JOB_COUNT = 11
AUTHORIZED_GPU_IDS = [0, 1, 2, 3]
MAX_GPU_COUNT = 1

DEFAULT_QUEUE = (
    ROOT / "matrix" / "h800_admission_negative_evidence_batch_b_resume_jobs_v1.jsonl"
)
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_admission_negative_evidence_design_v1.json"
DEFAULT_QUEUE_MANIFEST = (
    ARTIFACT_DIR / "h800_admission_negative_evidence_queue_manifest_v1.json"
)
CANDIDATE = (
    ARTIFACT_DIR
    / "approval_design_h800_admission_negative_evidence_batch_b_resume_v1_candidate.json"
)

# Scoped provenance: bind only what this campaign actually executes with.  A
# parallel session edits the throughput line in the same tree, so a full-project
# snapshot would either fail closed on their in-flight work or silently certify
# code this campaign never runs.  Still fail-closed -- just on these files.
SCOPED_SOURCE_RELATIVE_PATHS = (
    "config/experiment.json",
    "config/hardware.json",
    "config/models.json",
    "config/deepspeed/ds_z2.json",
    "config/deepspeed/ds_z3.json",
    "scripts/approval_gate.py",
    "scripts/common.py",
    "scripts/promote_approval_candidate.py",
    "scripts/prepare_h800_admission_negative_evidence_v1.py",
    "scripts/freeze_h800_admission_negative_evidence_batch_b_resume_v1.py",
    "scripts/run_job.py",
    "scripts/scheduler.py",
    "scripts/train_entry.py",
    "scripts/metrics_callback.py",
    "scripts/runtime_evidence.py",
    "scripts/gpu_telemetry.py",
    "scripts/collect_results.py",
)


def _validate_queue(rows: list[dict]) -> None:
    if len(rows) != EXPECTED_JOB_COUNT:
        raise ValueError(
            f"batch B must hold exactly {EXPECTED_JOB_COUNT} jobs, got {len(rows)}"
        )
    for row in rows:
        if row.get("schema") != JOB_SCHEMA:
            raise ValueError(f"unexpected job schema: {row.get('schema')}")
        if row.get("campaign_id") != CAMPAIGN_ID:
            raise ValueError(f"unexpected campaign: {row.get('campaign_id')}")
        if row.get("phase_id") != PHASE_ID:
            raise ValueError(f"unexpected phase: {row.get('phase_id')}")
        if row.get("packing") or row.get("offload"):
            raise ValueError("this campaign is unpacked and offload-free")
        gpu_count = int(row.get("gpu_count", 0))
        if gpu_count != 1:
            raise ValueError(
                f"this resume batch is single-GPU only; found gpu_count={gpu_count}. "
                "Other jobs already hold usable terminal evidence or need their own approval."
            )
        if gpu_count > MAX_GPU_COUNT:
            raise ValueError("job exceeds the approved max GPU count")
        if gpu_count == 1 and row.get("zero") != "none":
            raise ValueError("single-card jobs must not declare DeepSpeed")
    if len({row["job_id"] for row in rows}) != len(rows):
        raise ValueError("queue contains duplicate job ids")


def _validate_scope(rows: list[dict]) -> None:
    scope = read_json(ROOT / "config" / "experiment.json")["training_scope"]
    zero_by_gpu: dict[str, set[str]] = {}
    for row in rows:
        key = str(int(row["gpu_count"]))
        zero_by_gpu.setdefault(key, set()).add(str(row["zero"]))
    expected = {
        "phase_id": PHASE_ID,
        "gpu_ids": AUTHORIZED_GPU_IDS,
        "exclusive_node_gpu_ids": AUTHORIZED_GPU_IDS,
        "max_gpu_count": MAX_GPU_COUNT,
        "gpu_counts": sorted({int(row["gpu_count"]) for row in rows}),
        "model_ids": sorted({str(row["model_id"]) for row in rows}),
        "zero_by_gpu_count": {
            key: sorted(value) for key, value in sorted(zero_by_gpu.items())
        },
        "global_batch_sizes": sorted({int(row["target_gbs"]) for row in rows}),
        "gradient_checkpointing": sorted(
            {bool(row["gradient_checkpointing"]) for row in rows}
        ),
    }
    for key, want in expected.items():
        got = scope.get(key)
        if isinstance(want, list) and isinstance(got, list) and key != "gpu_ids":
            got, want = sorted(got), sorted(want)
        if got != want:
            raise ValueError(
                f"experiment scope mismatch on {key}: approved={want!r} config={got!r}"
            )


def freeze() -> Path:
    rows = read_jsonl(DEFAULT_QUEUE)
    _validate_queue(rows)
    _validate_scope(rows)

    campaign = read_json(DEFAULT_DESIGN)
    if campaign.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("campaign design does not match this campaign")
    for flag in ("gpu_training_started", "execution_authorized", "publication_allowed"):
        if campaign.get(flag) is not False:
            raise ValueError(f"campaign design must still have {flag}=false")
    unsigned = dict(campaign)
    expected_hash = unsigned.pop("report_sha256", None)
    if expected_hash != sha256_json(unsigned):
        raise ValueError("campaign design internal hash mismatch")
    approved = set(campaign["ordered_job_ids"])
    unknown = sorted(str(row["job_id"]) for row in rows if row["job_id"] not in approved)
    if unknown:
        raise ValueError(f"batch B holds job ids absent from the design: {unknown}")

    hardware = probe_hardware(required_gpu_ids=tuple(AUTHORIZED_GPU_IDS))
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"authorized GPUs are not an exact-H800 pool: {hardware}")
    busy = hardware.get("selected_gpu_compute_processes") or []
    if busy:
        raise RuntimeError(
            "refusing to freeze while GPU 0-3 carry external compute processes; "
            f"never preempt: {busy}"
        )

    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")

    queue_binding = build_queue_binding(DEFAULT_QUEUE, rows, ROOT)
    scoped_paths = [ROOT / name for name in SCOPED_SOURCE_RELATIVE_PATHS]
    absent = sorted(str(path) for path in scoped_paths if not path.is_file())
    if absent:
        raise FileNotFoundError(f"scoped execution sources are absent: {absent}")
    provenance_binding = build_provenance_binding(ROOT, source_paths=scoped_paths)

    bound_files = {
        DEFAULT_QUEUE,
        DEFAULT_DESIGN,
        DEFAULT_QUEUE_MANIFEST,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "data" / "dataset_info.json",
        *scoped_paths,
    }
    for row in rows:
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
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": (
                "fill idle GPUs within 0-3 with 1- and 2-GPU jobs; GPU 4-7 are "
                "reserved for other users and must never be selected; wait for an "
                "externally busy GPU instead of preempting it"
            ),
        },
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": stage,
    }
    write_json(CANDIDATE, design)
    return CANDIDATE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="replace the live approval; without it only the candidate is written",
    )
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
                "job_count": EXPECTED_JOB_COUNT,
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
