#!/usr/bin/env python3
"""Freeze and optionally promote the exact 12-job ZeRO-3 memory boundary campaign.

Scoped to GPUs 4-7.  Cards 0-3 are reserved for other users: they are never
inspected for idleness and never scheduled onto.
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
from prepare_h800_lora_2gpu_zero3_boundary_v1 import (
    CAMPAIGN_ID,
    CUTOFF_LEN,
    DEFAULT_DESIGN,
    DEFAULT_QUEUE,
    GPU_COUNT,
    JOB_SCHEMA,
    MBS_LADDER,
    MODEL_ID,
    PHASE_ID,
    REPEATS,
    SPLIT_POLICY,
    SPLIT_ROLE,
    TARGET_DATASETS,
    TARGET_GBS,
)
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch

AUTHORIZATION = (
    "用户在2026-08-08明确授权：在GPU 4/5/6/7四张卡上跑ZeRO-3显存边界补源实验（方案A+），"
    "明确要求不要动0/1/2/3。本批准仅允许执行Qwen3-14B LoRA、两卡、cutoff 4096、GBS 64、"
    "ZeRO-3、关梯度检查点、不packing不offload条件下的MBS 2/4/8阶梯，两个未消费stage2数据集，"
    "共12个作业。OOM是本实验的预期产物，按右删失下界记录，不补跑不插值。"
    "调度器使用两个互斥GPU对并行，不得抢占外部进程、不得运行Packing/VL/全参、"
    "不得依据本批结果发布模型或调整吞吐模型。"
)
AUTHORIZED_GPU_IDS = (4, 5, 6, 7)
AUTHORIZED_PAIRS = ((4, 5), (6, 7))
EXPECTED_JOBS = len(TARGET_DATASETS) * len(MBS_LADDER) * REPEATS
DATASET_IDS = frozenset(str(entry["dataset_id"]) for entry in TARGET_DATASETS)
SCRIPTS = ROOT / "scripts"
DEFAULT_CANDIDATE = (
    ARTIFACT_DIR / "approval_design_h800_lora_2gpu_zero3_boundary_v1_candidate.json"
)


def _hardware() -> dict[str, object]:
    hardware = probe_hardware(required_gpu_ids=AUTHORIZED_GPU_IDS)
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(
            f"GPUs {list(AUTHORIZED_GPU_IDS)} are not an exact H800 pool: {hardware}"
        )
    if hardware.get("selected_gpu_compute_processes"):
        raise RuntimeError(
            f"authorized GPUs {list(AUTHORIZED_GPU_IDS)} are not idle; "
            "refusing to freeze while they carry compute processes"
        )
    hardware.update(
        {
            "occupancy_is_authorization_time_preflight": True,
            "busy_gpu_ids_at_freeze": [],
            "idle_authorized_pairs_at_freeze": [list(p) for p in AUTHORIZED_PAIRS],
            "join_busy_pool_required": False,
            "preemption_allowed": False,
            "gpu_ids_reserved_for_others": [0, 1, 2, 3],
        }
    )
    return hardware


def freeze(queue: Path, design_path: Path, candidate_path: Path) -> Path:
    rows = read_jsonl(queue)
    if (
        len(rows) != EXPECTED_JOBS
        or len({str(r.get("job_id")) for r in rows}) != EXPECTED_JOBS
        or any(r.get("schema") != JOB_SCHEMA for r in rows)
        or any(r.get("campaign_id") != CAMPAIGN_ID for r in rows)
        or any(r.get("phase_id") != PHASE_ID for r in rows)
        or any(r.get("train_type") != "lora" for r in rows)
        or any(str(r.get("model_id")) != MODEL_ID for r in rows)
        or any(int(r.get("gpu_count", 0)) != GPU_COUNT for r in rows)
        or any(int(r.get("cutoff_len", 0)) != CUTOFF_LEN for r in rows)
        or any(int(r.get("target_gbs", 0)) != TARGET_GBS for r in rows)
        or any(int(r.get("zero_stage", 0)) != 3 for r in rows)
        or any(bool(r.get("gradient_checkpointing")) for r in rows)
        or any(r.get("packing") is not False for r in rows)
        or any(r.get("offload") is not False for r in rows)
        or any(int(r.get("mbs", 0)) not in set(MBS_LADDER) for r in rows)
        or any(int(r.get("warmup_steps", 0)) != 3 for r in rows)
        or any(int(r.get("measure_steps", 0)) != 10 for r in rows)
    ):
        raise ValueError("queue is not the exact 12-job ZeRO-3 boundary set")
    if {str(r["dataset_id"]) for r in rows} != DATASET_IDS:
        raise ValueError("queue does not contain the exact two target datasets")
    for row in rows:
        if int(row["gradient_accumulation_steps"]) != TARGET_GBS // (
            GPU_COUNT * int(row["mbs"])
        ):
            raise ValueError("a job does not preserve the exact global batch size")
        # The entire purpose of this campaign: every job must carry the partition
        # inside its authorized payload, because the exporter refuses to accept it
        # afterwards and the run would be wasted.
        partition = row.get("calibration_partition") or {}
        if (
            partition.get("role") != SPLIT_ROLE
            or partition.get("policy") != SPLIT_POLICY
            or not str(partition.get("split_unit_id") or "").strip()
        ):
            raise ValueError(
                f"job {row.get('job_id')} lacks a bound calibration partition; "
                "the resulting observation would be rejected by the exporter"
            )
        if row.get("oom_role") != "right_censored_lower_bound":
            raise ValueError("boundary probes must declare their OOM role")

    design = read_json(design_path)
    ids = [str(r["job_id"]) for r in rows]
    if (
        design.get("campaign_id") != CAMPAIGN_ID
        or design.get("ordered_job_ids") != ids
        or design.get("ordered_job_payload_sha256") != sha256_json(rows)
        or design.get("bindings", {}).get("queue", {}).get("sha256") != sha256_file(queue)
    ):
        raise ValueError("design binding drifted from the queue")

    experiment = read_json(ROOT / "config" / "experiment.json")
    scope = experiment["training_scope"]
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("model_ids") != [MODEL_ID]
        or scope.get("gpu_ids") != list(AUTHORIZED_GPU_IDS)
        or scope.get("exclusive_node_gpu_ids") != list(AUTHORIZED_GPU_IDS)
        or scope.get("max_gpu_count") != GPU_COUNT
        or scope.get("gpu_counts") != [GPU_COUNT]
        or scope.get("global_batch_sizes") != [TARGET_GBS]
        or experiment.get("measurement", {}).get("performance_parallelism")
        != "disjoint_gpu_masks"
    ):
        raise ValueError("experiment config is not the exact boundary-campaign scope")

    hardware = _hardware()
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(queue, rows, ROOT)

    # Scoped provenance: bind only what can change this batch's behaviour, so an
    # unrelated script landing in the tree mid-run does not invalidate the queue.
    scoped_source_paths = [
        SCRIPTS / "approval_gate.py",
        SCRIPTS / "common.py",
        SCRIPTS / "metrics_callback.py",
        SCRIPTS / "profiler_callback.py",
        SCRIPTS / "promote_approval_candidate.py",
        SCRIPTS / "run_job.py",
        SCRIPTS / "runtime_evidence.py",
        SCRIPTS / "scheduler.py",
        SCRIPTS / "train_entry.py",
        SCRIPTS / "prepare_h800_lora_2gpu_zero3_boundary_v1.py",
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "hardware.json",
        ROOT / "config" / "models.json",
        ROOT / "config" / "deepspeed" / "ds_z3.json",
    ]
    provenance_binding = build_provenance_binding(ROOT, source_paths=scoped_source_paths)

    bound_files = {
        queue,
        design_path,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "data" / "dataset_info.json",
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "models.json",
    }
    bound_files.update(scoped_source_paths)
    for row in rows:
        bound_files.add(Path(str(row["data_path"])))
        bound_files.add(Path(str(row["dataset_profile_path"])))
    missing = [str(p) for p in bound_files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files are absent: {missing}")
    file_manifest = {
        str(p.resolve().relative_to(ROOT.resolve())): sha256_file(p)
        for p in sorted(bound_files, key=str)
    }

    approval_design = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}",
        "authorization": AUTHORIZATION,
        "campaign_design": {
            "path": str(design_path.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(design_path),
        },
        "file_sha256": file_manifest,
        "execution_order": [PHASE_ID],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": list(AUTHORIZED_GPU_IDS),
        "gpu_ids_reserved_for_others": [0, 1, 2, 3],
        "max_gpu_count": GPU_COUNT,
        "scheduler_execution": {
            "join_busy_pool": False,
            "preemption_allowed": False,
            "policy": (
                "run two disjoint two-GPU jobs concurrently on pairs (4,5) and "
                "(6,7); GPUs 0-3 are reserved for other users and are never "
                "scheduled onto; fail closed if an authorized GPU is externally busy"
            ),
        },
        "oom_policy": "expected_evidence_right_censored_lower_bound_no_retry",
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
    write_json(candidate_path, approval_design)
    return candidate_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = freeze(
        args.queue.resolve(), args.design.resolve(), args.candidate.resolve()
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
            {
                "candidate": str(candidate),
                "sha256": digest,
                "jobs": EXPECTED_JOBS,
                "authorized_gpu_ids": list(AUTHORIZED_GPU_IDS),
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
