#!/usr/bin/env python3
"""Freeze and optionally promote the GPU 0-3 peak-replay calibration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from approval_gate import build_provenance_binding, build_queue_binding
from common import (
    ARTIFACT_DIR,
    DATA_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from prepare_h800_final_memory_peak_replay_stage1_v1 import (
    CAMPAIGN_ID,
    DATASET_INFO,
    DEFAULT_DATA_BUNDLE,
    DEFAULT_DESIGN,
    DEFAULT_EXPERIMENT,
    DEFAULT_PREDICTIONS,
    DEFAULT_QUEUE,
    EXPECTED_JOBS,
    EXPECTED_PAIRS,
    GPU_IDS,
    JOB_SCHEMA,
    MODEL_INVENTORY,
    PHASE_ID,
    REPLAY_MEASURE_STEPS,
    REPLAY_WARMUP_STEPS,
    SELECTION,
    TARGET_GBS,
    V3_ARTIFACT,
    WORKLOADS,
)
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch

AUTHORIZATION = (
    "用户在2026-08-10明确要求下载所需业务数据、制定完整显存模型补充实验计划并最终只在GPU 0-3运行，随后要求继续下一步。"
    "本批准仅允许阶段一的8个测量校准任务：四个预注册开发数据集各执行一次完整覆盖和一次精确峰值batch回放；"
    "只使用GPU 0-3，每任务1或2卡，不packing、不offload、不自动重试。"
    "不得读取或运行六个最终验收数据集，不得执行旧34任务，不得抢占其他用户进程，不得修改或发布线上模型。"
)
DEFAULT_CANDIDATE = (
    ARTIFACT_DIR
    / "approval_design_h800_final_memory_peak_replay_stage1_v1_candidate.json"
)
SCRIPTS = ROOT / "scripts"


def _hardware() -> dict[str, object]:
    hardware = probe_hardware(required_gpu_ids=GPU_IDS)
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError("GPU 0-3 are not an exact four-H800 selected pool")
    if hardware.get("selected_gpu_compute_processes"):
        raise RuntimeError("one or more authorized GPUs are busy")
    hardware.update(
        {
            "occupancy_is_authorization_time_preflight": True,
            "busy_gpu_ids_at_freeze": [],
            "join_busy_pool_required": False,
            "preemption_allowed": False,
            "gpu_ids_outside_authorized_pool_must_not_be_used": True,
        }
    )
    return hardware


def _validate_queue(
    rows: list[dict[str, object]], data_bundle: dict[str, object]
) -> None:
    if (
        len(rows) != EXPECTED_JOBS
        or len({str(row.get("job_id")) for row in rows}) != EXPECTED_JOBS
        or len({str(row.get("pair_id")) for row in rows}) != EXPECTED_PAIRS
    ):
        raise ValueError("queue does not contain exactly four unique pairs")
    development_ids = {
        str(row["source_dataset_id"])
        for row in data_bundle["profiles"]  # type: ignore[index]
    }
    expected_development_ids = {str(row["source_dataset_id"]) for row in WORKLOADS}
    if development_ids != expected_development_ids:
        raise ValueError("development source set drifted")
    by_pair: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair_id"]), []).append(row)
        if (
            row.get("schema") != JOB_SCHEMA
            or row.get("campaign_id") != CAMPAIGN_ID
            or row.get("phase_id") != PHASE_ID
            or str(row.get("source_dataset_id")) not in development_ids
            or int(row.get("gpu_count", 0)) not in {1, 2}
            or int(row.get("target_gbs", 0)) != TARGET_GBS
            or row.get("packing") is not False
            or row.get("offload") is not False
            or row.get("requires_external_node_idle") is not False
            or row.get("publication_allowed") is not False
        ):
            raise ValueError(
                f"queue row left the frozen stage-one scope: {row.get('job_id')}"
            )
        gpu_count = int(row["gpu_count"])
        mbs = int(row["mbs"])
        if int(row["gradient_accumulation_steps"]) * gpu_count * mbs != TARGET_GBS:
            raise ValueError(f"GBS identity mismatch: {row['job_id']}")
        for path_key, digest_key in (
            ("data_path", "data_sha256"),
            ("dataset_profile_path", "dataset_profile_sha256"),
            ("runtime_dataset_profile_path", "runtime_dataset_profile_sha256"),
        ):
            path = Path(str(row[path_key]))
            if not path.is_file() or sha256_file(path) != row[digest_key]:
                raise ValueError(
                    f"job file binding changed: {row['job_id']}:{path_key}"
                )
        if not Path(str(row["data_path"])).resolve().is_relative_to(DATA_DIR.resolve()):
            raise ValueError(f"runtime dataset escaped data root: {row['job_id']}")
        partition = row.get("calibration_partition") or {}
        if (
            partition.get("role") != "development_measurement"  # type: ignore[union-attr]
            or partition.get("policy") != "final_memory_peak_replay_stage1_v1"  # type: ignore[union-attr]
        ):
            raise ValueError(f"development partition missing: {row['job_id']}")
    for pair_id, pair_rows in by_pair.items():
        modes = {str(row["replay_mode"]) for row in pair_rows}
        if len(pair_rows) != 2 or modes != {"full_coverage", "peak_replay"}:
            raise ValueError(f"pair does not contain both exact arms: {pair_id}")
        full = next(row for row in pair_rows if row["replay_mode"] == "full_coverage")
        replay = next(row for row in pair_rows if row["replay_mode"] == "peak_replay")
        expected_full_steps = math.ceil(int(full["max_samples"]) / TARGET_GBS)
        if (
            int(full["warmup_steps"]) != 0
            or int(full["measure_steps"]) != expected_full_steps
            or int(full["full_plan_optimizer_steps"]) != expected_full_steps
            or int(replay["warmup_steps"]) != REPLAY_WARMUP_STEPS
            or int(replay["measure_steps"]) != REPLAY_MEASURE_STEPS
            or replay["replay_sampler_verification"].get(
                "all_rank_microbatches_exact_peak_batch"
            )
            is not True
        ):
            raise ValueError(f"full/replay measurement contract drifted: {pair_id}")
        identity_keys = (
            "source_dataset_id",
            "model_id",
            "train_type",
            "gpu_count",
            "zero_stage",
            "gc",
            "mbs",
            "cutoff_len",
            "gradient_accumulation_steps",
            "dataset_profile_sha256",
        )
        if any(full[key] != replay[key] for key in identity_keys):
            raise ValueError(f"pair mechanism identity differs: {pair_id}")


def freeze(
    queue: Path,
    design_path: Path,
    predictions_path: Path,
    data_bundle_path: Path,
    experiment_path: Path,
    candidate_path: Path,
) -> Path:
    rows = read_jsonl(queue)
    data_bundle = read_json(data_bundle_path)
    _validate_queue(rows, data_bundle)
    design = read_json(design_path)
    predictions = read_json(predictions_path)
    if (
        design.get("campaign_id") != CAMPAIGN_ID
        or design.get("phase_id") != PHASE_ID
        or design.get("authorized_gpu_ids") != list(GPU_IDS)
        or design.get("old_missing_34_jobs_included") is not False
        or design.get("ordered_job_ids") != [str(row["job_id"]) for row in rows]
        or design.get("ordered_job_payload_sha256") != sha256_json(rows)
        or design.get("bindings", {}).get("queue", {}).get("sha256")
        != sha256_file(queue)
        or design.get("bindings", {}).get("experiment", {}).get("sha256")
        != sha256_file(experiment_path)
        or design.get("bindings", {}).get("frozen_predictions", {}).get("sha256")
        != sha256_file(predictions_path)
        or design.get("bindings", {}).get("data_bundle", {}).get("sha256")
        != sha256_file(data_bundle_path)
        or predictions.get("status") != "frozen_before_any_stage1_gpu_outcome"
        or predictions.get("outcomes_observed") != 0
        or predictions.get("ordered_job_payload_sha256") != sha256_json(rows)
        or predictions.get("v3_artifact", {}).get("sha256") != sha256_file(V3_ARTIFACT)
        or data_bundle.get("final_acceptance_sources_read") != 0
    ):
        raise ValueError("stage-one design or prediction binding drifted")

    live_experiment = read_json(ROOT / "config" / "experiment.json")
    scope = live_experiment.get("training_scope") or {}
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("gpu_ids") != list(GPU_IDS)
        or scope.get("exclusive_node_gpu_ids") != list(GPU_IDS)
        or int(scope.get("max_gpu_count", 0)) != 2
        or scope.get("global_batch_sizes") != [TARGET_GBS]
    ):
        raise ValueError("live experiment config is not the GPU 0-3 stage-one scope")

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
        SCRIPTS / "prepare_h800_final_memory_peak_replay_stage1_v1.py",
        SCRIPTS / "evaluate_h800_final_memory_peak_replay_stage1_v1.py",
        Path(__file__).resolve(),
        ROOT
        / "final_memory_peak_replay_stage1_staging"
        / "launch_h800_final_memory_peak_replay_stage1_v1.py",
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
        data_bundle_path,
        experiment_path,
        MODEL_INVENTORY,
        V3_ARTIFACT,
        SELECTION,
        DATASET_INFO,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "models.json",
        *scoped_source_paths,
    }
    for row in rows:
        bound_files.add(Path(str(row["data_path"])))
        bound_files.add(Path(str(row["dataset_profile_path"])))
        bound_files.add(Path(str(row["runtime_dataset_profile_path"])))
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
            "policy": "parallel disjoint one/two-GPU masks across GPU 0-3 only",
        },
        "oom_policy": "unexpected_measurement_calibration_failure_no_retry",
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
    parser.add_argument("--data-bundle", type=Path, default=DEFAULT_DATA_BUNDLE)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = freeze(
        args.queue.resolve(),
        args.design.resolve(),
        args.predictions.resolve(),
        args.data_bundle.resolve(),
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
                "pairs": EXPECTED_PAIRS,
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
