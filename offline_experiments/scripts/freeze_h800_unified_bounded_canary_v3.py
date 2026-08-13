#!/usr/bin/env python3
"""Freeze and optionally promote the exact ten-job v3 canary approval."""

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
from prepare_h800_unified_bounded_canary_v3 import (
    CAMPAIGN_ID,
    DEFAULT_DESIGN,
    DEFAULT_EXPERIMENT,
    DEFAULT_PREDICTIONS,
    DEFAULT_QUEUE,
    EXPECTED_JOBS,
    GPU_IDS,
    JOB_SCHEMA,
    MEASURE_STEPS,
    MODEL_INVENTORY,
    PHASE_ID,
    TARGET_GBS,
    V3_ARTIFACT,
    WARMUP_STEPS,
    WORKLOADS,
)
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch

AUTHORIZATION = (
    "用户在2026-08-10明确要求依次完成1、2、针对性实验、4和5，并明确不要继续原来34个慢任务。"
    "本批准仅允许冻结v3后的10个前瞻小规模显存canary：8个v3已放行的边界配置和2个v3已拒绝的边界对照；"
    "使用GPU 0-7，每任务1或2卡，GBS32，1步warmup+4步measure，不packing、不offload、不重试。"
    "不得抢占其他用户进程，不得执行旧34任务，不得在看到结果后修改冻结预测，不得直接发布或覆盖线上模型。"
)
DEFAULT_CANDIDATE = (
    ARTIFACT_DIR / "approval_design_h800_unified_bounded_canary_v3_candidate.json"
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
    queue: Path,
    design_path: Path,
    predictions_path: Path,
    experiment_path: Path,
    candidate_path: Path,
) -> Path:
    rows = read_jsonl(queue)
    inventory = read_json(MODEL_INVENTORY)
    models = {str(row["id"]): row for row in inventory["models"]}
    expected = {
        (
            str(row["probe_role"]),
            str(row["model_id"]),
            str(row["train_type"]),
            str(row["dataset_id"]),
            int(row["cutoff_len"]),
            int(row["gpu_count"]),
            int(row["mbs"]),
            int(row["zero_stage"]),
            bool(row["gc"]),
        )
        for row in WORKLOADS
    }
    actual = {
        (
            str(row.get("probe_role")),
            str(row.get("model_id")),
            str(row.get("train_type")),
            str(row.get("dataset_id")),
            int(row.get("cutoff_len", 0)),
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
        or actual != expected
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
        or any(int(row.get("target_gbs", 0)) != TARGET_GBS for row in rows)
        or any(int(row.get("warmup_steps", -1)) != WARMUP_STEPS for row in rows)
        or any(int(row.get("measure_steps", -1)) != MEASURE_STEPS for row in rows)
        or any(int(row.get("gpu_count", 0)) not in {1, 2} for row in rows)
        or any(row.get("packing") is not False for row in rows)
        or any(row.get("offload") is not False for row in rows)
    ):
        raise ValueError("queue is not the exact ten-job frozen-v3 canary")
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
            partition.get("role") != "prospective_acceptance"
            or partition.get("policy") != "unified_bounded_v3_frozen_canary_v1"
        ):
            raise ValueError(f"prospective partition missing: {row['job_id']}")

    design = read_json(design_path)
    predictions = read_json(predictions_path)
    if (
        design.get("campaign_id") != CAMPAIGN_ID
        or design.get("ordered_job_ids") != [str(row["job_id"]) for row in rows]
        or design.get("ordered_job_payload_sha256") != sha256_json(rows)
        or design.get("old_missing_34_jobs_included") is not False
        or design.get("bindings", {}).get("queue", {}).get("sha256")
        != sha256_file(queue)
        or design.get("bindings", {}).get("experiment", {}).get("sha256")
        != sha256_file(experiment_path)
        or design.get("bindings", {}).get("frozen_predictions", {}).get("sha256")
        != sha256_file(predictions_path)
        or predictions.get("status") != "frozen_before_any_canary_outcome"
        or predictions.get("outcomes_observed") != 0
        or predictions.get("ordered_job_payload_sha256") != sha256_json(rows)
        or predictions.get("v3_artifact", {}).get("sha256") != sha256_file(V3_ARTIFACT)
    ):
        raise ValueError("canary design or prediction binding drifted")

    live_experiment = read_json(ROOT / "config" / "experiment.json")
    scope = live_experiment.get("training_scope") or {}
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("gpu_ids") != list(GPU_IDS)
        or scope.get("exclusive_node_gpu_ids") != list(GPU_IDS)
        or int(scope.get("max_gpu_count", 0)) != 2
        or scope.get("global_batch_sizes") != [TARGET_GBS]
    ):
        raise ValueError("live experiment config is not the canary scope")

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
        SCRIPTS / "prepare_h800_unified_bounded_canary_v3.py",
        SCRIPTS / "evaluate_h800_unified_bounded_canary_v3.py",
        Path(__file__).resolve(),
        ROOT
        / "unified_bounded_canary_staging"
        / "launch_h800_unified_bounded_canary_v3.py",
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
        predictions_path,
        experiment_path,
        MODEL_INVENTORY,
        V3_ARTIFACT,
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
        "frozen_predictions": {
            "path": str(predictions_path.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(predictions_path),
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
        "oom_policy": "no_retry; admitted OOM is acceptance failure; rejected-control OOM is correctly rejected evidence",
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
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = freeze(
        args.queue.resolve(),
        args.design.resolve(),
        args.predictions.resolve(),
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
