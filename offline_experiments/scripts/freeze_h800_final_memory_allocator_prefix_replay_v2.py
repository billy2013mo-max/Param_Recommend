#!/usr/bin/env python3
"""Freeze and optionally promote the two-job allocator-prefix repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, DATA_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_final_memory_allocator_prefix_replay_v2 import (
    CAMPAIGN_ID,
    DATASET_INFO,
    DEFAULT_DATA_BUNDLE,
    DEFAULT_DESIGN,
    DEFAULT_EXPERIMENT,
    DEFAULT_QUEUE,
    EXPECTED_JOBS,
    GPU_IDS,
    JOB_SCHEMA,
    PHASE_ID,
    TARGET_GBS,
    V1_RESULTS,
)
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch

AUTHORIZATION = (
    "用户在2026-08-10要求继续完成显存模型最终验收。阶段一发现LoRA冷峰值回放的reserved显存"
    "低估14.60%和19.70%，因此本批准只允许在GPU 0-3上运行两个开发集LoRA allocator前缀回放修复任务；"
    "任务严格复用已成功的完整覆盖基线，不读取六个最终业务盲测结果，不运行旧34任务，不抢占其他任务，"
    "不修改、不发布线上模型。"
)
DEFAULT_CANDIDATE = ARTIFACT_DIR / "approval_design_h800_final_memory_allocator_prefix_replay_v2_candidate.json"
SCRIPTS = ROOT / "scripts"


def _hardware() -> dict[str, Any]:
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


def _validate_baselines(data_bundle: dict[str, Any]) -> None:
    baselines = data_bundle.get("v1_full_baselines") or []
    if len(baselines) != EXPECTED_JOBS:
        raise ValueError("data bundle does not bind exactly two v1 full baselines")
    for baseline in baselines:
        for key in ("latest_attempt", "status"):
            path = Path(str(baseline[f"{key}_path"]))
            if not path.is_file() or sha256_file(path) != baseline[f"{key}_sha256"]:
                raise ValueError(f"v1 baseline binding changed: {path}")
        summaries = baseline.get("summary_bindings") or []
        if not summaries:
            raise ValueError("v1 baseline has no bound rank summaries")
        for binding in summaries:
            path = Path(str(binding["path"]))
            if not path.is_file() or sha256_file(path) != binding["sha256"]:
                raise ValueError(f"v1 baseline summary changed: {path}")


def _validate_queue(rows: list[dict[str, Any]]) -> None:
    if (
        len(rows) != EXPECTED_JOBS
        or len({str(row.get("job_id")) for row in rows}) != EXPECTED_JOBS
        or len({str(row.get("v1_pair_id")) for row in rows}) != EXPECTED_JOBS
    ):
        raise ValueError("repair queue does not contain exactly two unique LoRA jobs")
    for row in rows:
        if (
            row.get("schema") != JOB_SCHEMA
            or row.get("campaign_id") != CAMPAIGN_ID
            or row.get("phase_id") != PHASE_ID
            or row.get("train_type") != "lora"
            or row.get("replay_mode") != "allocator_prefix_replay"
            or int(row.get("gpu_count", 0)) not in {1, 2}
            or int(row.get("target_gbs", 0)) != TARGET_GBS
            or row.get("packing") is not False
            or row.get("offload") is not False
            or row.get("publication_allowed") is not False
            or row.get("oom_is_expected_evidence") is not False
        ):
            raise ValueError(f"repair job left frozen scope: {row.get('job_id')}")
        contract = row.get("allocator_prefix_contract") or {}
        verification = row.get("allocator_prefix_sampler_verification") or {}
        if (
            verification.get("all_rank_prefixes_exact") is not True
            or int(row["warmup_steps"]) != 0
            or int(row["measure_steps"]) != int(contract["prefix_optimizer_steps"])
            or int(row["max_samples"]) != int(row["measure_steps"]) * TARGET_GBS
            or int(row["gradient_accumulation_steps"])
            * int(row["gpu_count"])
            * int(row["mbs"])
            != TARGET_GBS
        ):
            raise ValueError(f"allocator-prefix contract drifted: {row['job_id']}")
        partition = row.get("calibration_partition") or {}
        if (
            partition.get("role") != "development_measurement"
            or partition.get("policy") != "final_memory_allocator_prefix_replay_v2"
        ):
            raise ValueError(f"development partition missing: {row['job_id']}")
        for path_key, digest_key in (
            ("data_path", "data_sha256"),
            ("dataset_profile_path", "dataset_profile_sha256"),
            ("runtime_dataset_profile_path", "runtime_dataset_profile_sha256"),
        ):
            path = Path(str(row[path_key]))
            if not path.is_file() or sha256_file(path) != row[digest_key]:
                raise ValueError(f"job file binding changed: {row['job_id']}:{path_key}")
        if not Path(str(row["data_path"])).resolve().is_relative_to(DATA_DIR.resolve()):
            raise ValueError(f"runtime dataset escaped data root: {row['job_id']}")


def freeze(
    queue: Path,
    design_path: Path,
    data_bundle_path: Path,
    experiment_path: Path,
    candidate_path: Path,
) -> Path:
    rows = read_jsonl(queue)
    design = read_json(design_path)
    data_bundle = read_json(data_bundle_path)
    _validate_queue(rows)
    _validate_baselines(data_bundle)
    bindings = design.get("bindings") or {}
    if (
        design.get("campaign_id") != CAMPAIGN_ID
        or design.get("phase_id") != PHASE_ID
        or design.get("authorized_gpu_ids") != list(GPU_IDS)
        or design.get("old_missing_34_jobs_included") is not False
        or design.get("ordered_job_ids") != [str(row["job_id"]) for row in rows]
        or design.get("ordered_job_payload_sha256") != sha256_json(rows)
        or bindings.get("queue", {}).get("sha256") != sha256_file(queue)
        or bindings.get("experiment", {}).get("sha256") != sha256_file(experiment_path)
        or bindings.get("data_bundle", {}).get("sha256") != sha256_file(data_bundle_path)
        or bindings.get("v1_results", {}).get("sha256") != sha256_file(V1_RESULTS)
        or data_bundle.get("gpu_outcomes_observed") != 0
        or data_bundle.get("final_acceptance_sources_read") != 0
    ):
        raise ValueError("repair design or data binding drifted")

    live_experiment = read_json(ROOT / "config" / "experiment.json")
    scope = live_experiment.get("training_scope") or {}
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("gpu_ids") != list(GPU_IDS)
        or scope.get("exclusive_node_gpu_ids") != list(GPU_IDS)
        or int(scope.get("max_gpu_count", 0)) != 2
    ):
        raise ValueError("live experiment config is not the allocator-prefix repair scope")

    hardware = _hardware()
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(queue, rows, ROOT)
    launcher = ROOT / "final_memory_allocator_prefix_replay_v2_staging" / "launch_h800_final_memory_allocator_prefix_replay_v2.py"
    scoped_source_paths = [
        SCRIPTS / "approval_gate.py",
        SCRIPTS / "common.py",
        SCRIPTS / "metrics_callback.py",
        SCRIPTS / "promote_approval_candidate.py",
        SCRIPTS / "run_job.py",
        SCRIPTS / "runtime_evidence.py",
        SCRIPTS / "scheduler.py",
        SCRIPTS / "train_entry.py",
        SCRIPTS / "prepare_h800_final_memory_allocator_prefix_replay_v2.py",
        SCRIPTS / "evaluate_h800_final_memory_allocator_prefix_replay_v2.py",
        Path(__file__).resolve(),
        launcher,
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "hardware.json",
        ROOT / "config" / "models.json",
        ROOT / "config" / "deepspeed" / "ds_z2.json",
    ]
    provenance_binding = build_provenance_binding(ROOT, source_paths=scoped_source_paths)
    bound_files = {
        queue,
        design_path,
        data_bundle_path,
        experiment_path,
        V1_RESULTS,
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
    for baseline in data_bundle["v1_full_baselines"]:
        bound_files.add(Path(str(baseline["latest_attempt_path"])))
        bound_files.add(Path(str(baseline["status_path"])))
        bound_files.update(Path(str(item["path"])) for item in baseline["summary_bindings"])
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
        "oom_policy": "unexpected_measurement_repair_failure_no_retry",
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
    parser.add_argument("--data-bundle", type=Path, default=DEFAULT_DATA_BUNDLE)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = freeze(
        args.queue.resolve(),
        args.design.resolve(),
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
