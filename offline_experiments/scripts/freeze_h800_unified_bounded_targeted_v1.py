#!/usr/bin/env python3
"""Freeze and promote the exact 15-job unified-bounded targeted campaign."""

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
from prepare_h800_unified_bounded_targeted_v1 import (
    CAMPAIGN_ID,
    DEFAULT_DESIGN,
    DEFAULT_QUEUE,
    EXPECTED_JOBS,
    GPU_IDS,
    JOB_SCHEMA,
    MEASURE_STEPS,
    MODEL_INVENTORY,
    PHASE_ID,
    TARGET_GBS,
    WARMUP_STEPS,
    WORKLOADS,
)
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch

AUTHORIZATION = (
    "用户在2026-08-10明确授权：不要继续原来未跑完的34个慢任务，改为运行助手认为有针对性的实验，"
    "随后继续shadow和canary。本批准只允许15个短显存探针：Qwen3-14B LoRA两卡ZeRO-3/无GC的"
    "profile与MBS边界、Qwen3-1.7B/4B的小模型GC对照、Qwen3-8B LoRA两卡ZeRO/GC对照。"
    "使用GPU 0-7、每任务1或2卡、cutoff 4096、GBS32、1步warmup+2步measure；"
    "不packing、不offload，OOM按右删失证据记录。不得执行原34个resume任务，不得据此直接发布模型。"
)
DEFAULT_EXPERIMENT = (
    ROOT
    / "unified_bounded_targeted_staging"
    / "experiment.h800_unified_bounded_targeted_v1.json"
)
DEFAULT_CANDIDATE = (
    ARTIFACT_DIR / "approval_design_h800_unified_bounded_targeted_v1_candidate.json"
)
SCRIPTS = ROOT / "scripts"


def _hardware() -> dict[str, object]:
    hardware = probe_hardware(required_gpu_ids=GPU_IDS)
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError("GPU 0-7 are not an exact eight-H800 pool")
    if hardware.get("selected_gpu_compute_processes"):
        raise RuntimeError("one or more authorized H800s are busy")
    hardware.update(
        {
            "occupancy_is_authorization_time_preflight": True,
            "busy_gpu_ids_at_freeze": [],
            "join_busy_pool_required": False,
            "preemption_allowed": False,
        }
    )
    return hardware


def freeze(
    queue: Path, design_path: Path, experiment_path: Path, candidate_path: Path
) -> Path:
    rows = read_jsonl(queue)
    inventory = read_json(MODEL_INVENTORY)
    models = {str(row["id"]): row for row in inventory["models"]}
    expected_workloads = {
        (
            str(row["probe_family"]),
            str(row["model_id"]),
            str(row["train_type"]),
            str(row["dataset_id"]),
            int(row["gpu_count"]),
            int(row["mbs"]),
            int(row["zero_stage"]),
            bool(row["gc"]),
        )
        for row in WORKLOADS
    }
    actual_workloads = {
        (
            str(row.get("probe_family")),
            str(row.get("model_id")),
            str(row.get("train_type")),
            str(row.get("dataset_id")),
            int(row.get("gpu_count", 0)),
            int(row.get("mbs", 0)),
            int(row.get("zero_stage", -1)),
            bool(row.get("gc")),
        )
        for row in rows
    }
    if (
        len(rows) != EXPECTED_JOBS
        or len({str(row.get("job_id")) for row in rows}) != EXPECTED_JOBS
        or actual_workloads != expected_workloads
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
        or any(int(row.get("target_gbs", 0)) != TARGET_GBS for row in rows)
        or any(int(row.get("warmup_steps", -1)) != WARMUP_STEPS for row in rows)
        or any(int(row.get("measure_steps", -1)) != MEASURE_STEPS for row in rows)
        or any(int(row.get("cutoff_len", 0)) != 4096 for row in rows)
        or any(int(row.get("gpu_count", 0)) not in {1, 2} for row in rows)
        or any(row.get("packing") is not False for row in rows)
        or any(row.get("offload") is not False for row in rows)
        or any(row.get("oom_role") != "right_censored_lower_bound" for row in rows)
    ):
        raise ValueError("queue is not the exact 15-job targeted campaign")
    for row in rows:
        model = models[str(row["model_id"])]
        if (
            Path(str(row["model_path"])).resolve() != Path(str(model["path"])).resolve()
            or Path(str(row["tokenizer_path"])).resolve()
            != Path(str(model["path"])).resolve()
            or int(row["model_parameters"]) != int(model["actual_parameters"])
        ):
            raise ValueError(f"model identity mismatch: {row['job_id']}")
        if (
            int(row["gradient_accumulation_steps"])
            * int(row["gpu_count"])
            * int(row["mbs"])
            != TARGET_GBS
        ):
            raise ValueError(f"GBS mismatch: {row['job_id']}")
        partition = row.get("calibration_partition") or {}
        if (
            partition.get("role") != "fit"
            or partition.get("policy") != "unified_bounded_targeted_mechanism_v1"
            or not partition.get("split_unit_id")
        ):
            raise ValueError(f"calibration partition missing: {row['job_id']}")

    design = read_json(design_path)
    if (
        design.get("campaign_id") != CAMPAIGN_ID
        or design.get("ordered_job_ids") != [str(row["job_id"]) for row in rows]
        or design.get("ordered_job_payload_sha256") != sha256_json(rows)
        or design.get("old_missing_34_jobs_included") is not False
        or design.get("bindings", {}).get("queue", {}).get("sha256")
        != sha256_file(queue)
        or design.get("bindings", {}).get("experiment", {}).get("sha256")
        != sha256_file(experiment_path)
    ):
        raise ValueError("design binding drifted from targeted queue")

    live_experiment = read_json(ROOT / "config" / "experiment.json")
    scope = live_experiment.get("training_scope") or {}
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("gpu_ids") != list(GPU_IDS)
        or scope.get("exclusive_node_gpu_ids") != list(GPU_IDS)
        or int(scope.get("max_gpu_count", 0)) != 2
        or scope.get("global_batch_sizes") != [TARGET_GBS]
    ):
        raise ValueError("live experiment config is not the targeted scope")

    hardware = _hardware()
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(queue, rows, ROOT)
    scoped_source_paths = [
        SCRIPTS / "approval_gate.py",
        SCRIPTS / "common.py",
        SCRIPTS / "metrics_callback.py",
        SCRIPTS / "promote_approval_candidate.py",
        SCRIPTS / "run_job.py",
        SCRIPTS / "runtime_evidence.py",
        SCRIPTS / "scheduler.py",
        SCRIPTS / "train_entry.py",
        SCRIPTS / "prepare_h800_unified_bounded_targeted_v1.py",
        SCRIPTS / "evaluate_h800_unified_bounded_targeted_v1.py",
        Path(__file__).resolve(),
        ROOT
        / "unified_bounded_targeted_staging"
        / "launch_h800_unified_bounded_targeted_v1.py",
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "hardware.json",
        ROOT / "config" / "models.json",
        ROOT / "config" / "deepspeed" / "ds_z2.json",
        ROOT / "config" / "deepspeed" / "ds_z3.json",
    ]
    provenance_binding = build_provenance_binding(
        ROOT, source_paths=scoped_source_paths
    )
    bound_files = {
        queue,
        design_path,
        experiment_path,
        MODEL_INVENTORY,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "data" / "dataset_info.json",
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "models.json",
        *scoped_source_paths,
    }
    for row in rows:
        bound_files.add(Path(str(row["data_path"])))
        bound_files.add(Path(str(row["dataset_profile_path"])))
        bound_files.add(Path(str(row["model_path"])))
    missing = [str(path) for path in bound_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"approval-bound paths are absent: {missing}")
    file_manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=str)
        if path.is_file() and path.resolve().is_relative_to(ROOT.resolve())
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
        "allowed_job_ids": [str(row["job_id"]) for row in rows],
        "authorized_gpu_ids": list(GPU_IDS),
        "max_gpu_count": 2,
        "scheduler_execution": {
            "join_busy_pool": False,
            "preemption_allowed": False,
            "policy": "parallel disjoint one/two-GPU masks across GPU 0-7",
        },
        "oom_policy": "expected_evidence_right_censored_lower_bound_no_retry",
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": {
            "allowed_job_ids": queue_binding["ordered_job_ids"],
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
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = freeze(
        args.queue.resolve(),
        args.design.resolve(),
        args.experiment.resolve(),
        args.candidate.resolve(),
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
                "authorized_gpu_ids": list(GPU_IDS),
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
