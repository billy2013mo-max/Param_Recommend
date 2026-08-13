#!/usr/bin/env python3
"""Freeze and optionally promote the FULL admission backfill approval.

Binds the exact 46-job queue produced by
``prepare_h800_full_admission_backfill_v1.py`` to an approval design covering
all eight H800s.  Creating the candidate is always safe; only ``--promote``
replaces the live approval, and neither path launches GPU training.
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
    "用户在 2026-08-06 明确同意：按 8 卡排期运行 FULL 显存准入补数实验，并同意为此"
    "修改 config/experiment.json 的 training_scope（max_gpu_count 由 2 放宽到 4、"
    "加入 1/4 卡与 1.7B/4B/8B 模型、允许全参配置）。本 approval 只允许执行冻结的 46 个"
    "FULL 显存准入补数作业，不得抢占外部进程、不得复用旧 campaign 的 approval、"
    "不得扩展到未批准作业，也不得放宽任何显存安全阈值。"
)
CAMPAIGN_ID = "h800_full_admission_backfill_20260806_v1"
PHASE_ID = "h800_full_admission_backfill_v1"
JOB_SCHEMA = "sft_h800_full_admission_backfill_job/v1"
EXPECTED_JOB_COUNT = 46
AUTHORIZED_GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
MAX_GPU_COUNT = 4

DEFAULT_QUEUE = ROOT / "matrix" / "h800_full_admission_backfill_jobs_v1.jsonl"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_full_admission_backfill_design_v1.json"
DEFAULT_QUEUE_MANIFEST = (
    ARTIFACT_DIR / "h800_full_admission_backfill_queue_manifest_v1.json"
)
CANDIDATE = (
    ARTIFACT_DIR / "approval_design_h800_full_admission_backfill_v1_candidate.json"
)

# Scoped provenance: bind only the files this campaign actually executes with.
# A parallel session is advancing the throughput line in the same working tree,
# so a full-project snapshot would either fail closed on their in-flight edits or
# silently certify code this campaign never runs.  `approval_file_manifest_scoped_v1`
# is still fail-closed -- it just fails on *these* files.
SCOPED_SOURCE_RELATIVE_PATHS = (
    "config/experiment.json",
    "config/hardware.json",
    "config/models.json",
    "config/deepspeed/ds_z2.json",
    "config/deepspeed/ds_z3.json",
    "scripts/approval_gate.py",
    "scripts/common.py",
    "scripts/promote_approval_candidate.py",
    "scripts/prepare_h800_full_admission_backfill_v1.py",
    "scripts/freeze_h800_full_admission_backfill_v1.py",
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
            f"queue must hold exactly {EXPECTED_JOB_COUNT} jobs, got {len(rows)}"
        )
    for row in rows:
        if row.get("schema") != JOB_SCHEMA:
            raise ValueError(f"unexpected job schema: {row.get('schema')}")
        if row.get("campaign_id") != CAMPAIGN_ID:
            raise ValueError(f"unexpected campaign: {row.get('campaign_id')}")
        if row.get("phase_id") != PHASE_ID:
            raise ValueError(f"unexpected phase: {row.get('phase_id')}")
        if row.get("train_type") != "full":
            raise ValueError("this campaign is FULL fine-tuning only")
        if row.get("packing") or row.get("offload"):
            raise ValueError("this campaign is unpacked and offload-free")
        if int(row.get("gpu_count", 0)) not in (1, 2, 4):
            raise ValueError(f"unexpected gpu_count: {row.get('gpu_count')}")
        if int(row.get("gpu_count", 0)) > MAX_GPU_COUNT:
            raise ValueError("job exceeds the approved max GPU count")
    if len({row["job_id"] for row in rows}) != len(rows):
        raise ValueError("queue contains duplicate job ids")


def _validate_scope(rows: list[dict]) -> None:
    scope = read_json(ROOT / "config" / "experiment.json")["training_scope"]
    zero_by_gpu: dict[str, set[str]] = {}
    for row in rows:
        key = str(int(row["gpu_count"]))
        zero_by_gpu.setdefault(key, set()).add(f"zero{int(row['zero_stage'])}")
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
        if isinstance(want, list) and isinstance(got, list):
            got = sorted(got) if key != "gpu_ids" else got
            want = sorted(want) if key != "gpu_ids" else want
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
    if campaign["ordered_job_payload_sha256"] != sha256_json(rows):
        raise ValueError("campaign design does not bind this exact queue payload")

    hardware = probe_hardware(required_gpu_ids=tuple(AUTHORIZED_GPU_IDS))
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"authorized GPUs are not an exact-H800 pool: {hardware}")
    busy = hardware.get("selected_gpu_compute_processes") or []
    if busy:
        raise RuntimeError(
            "refusing to freeze while authorized GPUs carry external compute "
            f"processes; never preempt: {busy}"
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
                "fill idle H800s across GPU 0-7 with 1-, 2-, and 4-GPU FULL jobs; "
                "wait for any externally busy GPU instead of preempting it"
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
