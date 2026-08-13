#!/usr/bin/env python3
"""Freeze and optionally promote the 60-job final V3 business blind."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, DATA_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_final_memory_business_blind_v2 import implementation as prep
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch

AUTHORIZATION = (
    "用户在2026-08-10要求继续完成显存模型最终业务盲测。本批准只允许已在任何GPU结果前冻结的"
    "V3六业务数据集60任务初始矩阵，仅使用GPU 0-3；非packing任务使用已通过等价性门槛的allocator前缀回放，"
    "packing任务必须完整覆盖packed计划。OOM是有效验收标签且不重试；不得运行旧34任务、不得使用GPU 4-7、"
    "不得抢占其他任务、不得根据结果修改预测、阈值或线上模型。"
)
DEFAULT_CANDIDATE = ARTIFACT_DIR / "approval_design_h800_final_memory_business_blind_v2_candidate.json"
SCRIPTS = ROOT / "scripts"


def _hardware() -> dict[str, Any]:
    hardware = probe_hardware(required_gpu_ids=prep.GPU_IDS)
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError("GPU 0-3 are not an exact four-H800 selected pool")
    if hardware.get("selected_gpu_compute_processes"):
        raise RuntimeError("one or more authorized GPUs are busy")
    hardware.update({
        "occupancy_is_authorization_time_preflight": True,
        "busy_gpu_ids_at_freeze": [],
        "join_busy_pool_required": False,
        "preemption_allowed": False,
        "gpu_ids_outside_authorized_pool_must_not_be_used": True,
    })
    return hardware


def _validate_queue(rows: list[dict[str, Any]], predictions: dict[str, Any]) -> None:
    if (
        len(rows) != prep.EXPECTED_JOBS
        or len({str(row.get("job_id")) for row in rows}) != prep.EXPECTED_JOBS
        or len({str(row.get("scenario")) for row in rows}) != prep.EXPECTED_SCENARIOS
        or any(
            count != len(prep.TARGET_PRESSURES)
            for count in Counter(str(row.get("scenario")) for row in rows).values()
        )
    ):
        raise ValueError("final blind queue size, scenarios, or uniqueness drifted")
    prediction_rows = predictions.get("rows") or []
    if len(prediction_rows) != prep.EXPECTED_JOBS:
        raise ValueError("frozen prediction row count drifted")
    prediction_ids = {str(row["job_id"]) for row in prediction_rows}
    if prediction_ids != {str(row["job_id"]) for row in rows}:
        raise ValueError("prediction and queue job IDs differ")
    by_scenario: dict[str, set[float]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario"]), set()).add(float(row["target_pressure"]))
        if (
            row.get("schema") != prep.JOB_SCHEMA
            or row.get("campaign_id") != prep.CAMPAIGN_ID
            or row.get("phase_id") != prep.PHASE_ID
            or int(row.get("gpu_count", 0)) not in {1, 2, 4}
            or (
                int(row.get("gpu_count", 0)) > 1
                and int(row.get("zero_stage", -1)) not in {2, 3}
            )
            or row.get("offload") is not False
            or row.get("publication_allowed") is not False
            or row.get("oom_is_expected_evidence") is not True
            or row.get("warmup_steps") != 0
            or int(row.get("measure_steps", 0)) <= 0
        ):
            raise ValueError(f"final job left frozen scope: {row.get('job_id')}")
        partition = row.get("calibration_partition") or {}
        if partition.get("role") != "final_acceptance_blind" or partition.get("policy") != "six_source_content_disjoint_v1":
            raise ValueError(f"final blind partition missing: {row['job_id']}")
        mode = str(row.get("measurement_mode"))
        contract = row.get("measurement_contract") or {}
        if bool(row.get("packing")):
            if mode != "full_packed_dataset_coverage" or contract.get("reason") != "packing allocator-prefix equivalence is not established":
                raise ValueError(f"packing shortcut is forbidden: {row['job_id']}")
        elif mode != "validated_allocator_prefix_replay" or (contract.get("verification") or {}).get("all_rank_prefixes_exact") is not True:
            raise ValueError(f"nonpacking prefix is not exact: {row['job_id']}")
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
    if any(values != set(prep.TARGET_PRESSURES) for values in by_scenario.values()):
        raise ValueError("one or more scenario pressure ladders drifted")


def freeze(queue: Path, design_path: Path, predictions_path: Path, data_path: Path, experiment_path: Path, candidate_path: Path) -> Path:
    rows = read_jsonl(queue)
    design = read_json(design_path)
    predictions = read_json(predictions_path)
    data = read_json(data_path)
    _validate_queue(rows, predictions)
    bindings = design.get("bindings") or {}
    if (
        design.get("campaign_id") != prep.CAMPAIGN_ID
        or design.get("phase_id") != prep.PHASE_ID
        or design.get("authorized_gpu_ids") != list(prep.GPU_IDS)
        or design.get("old_missing_34_jobs_included") is not False
        or design.get("ordered_job_ids") != [str(row["job_id"]) for row in rows]
        or design.get("ordered_job_payload_sha256") != sha256_json(rows)
        or bindings.get("queue", {}).get("sha256") != sha256_file(queue)
        or bindings.get("experiment", {}).get("sha256") != sha256_file(experiment_path)
        or bindings.get("data_bundle", {}).get("sha256") != sha256_file(data_path)
        or bindings.get("frozen_predictions", {}).get("sha256") != sha256_file(predictions_path)
        or bindings.get("v3_artifact", {}).get("sha256") != sha256_file(prep.V3_ARTIFACT)
        or bindings.get("measurement_gate", {}).get("sha256") != sha256_file(prep.MEASUREMENT_GATE)
        or predictions.get("status") != "frozen_before_any_final_gpu_outcome"
        or predictions.get("outcomes_observed") != 0
        or predictions.get("ordered_job_payload_sha256") != sha256_json(rows)
        or data.get("gpu_outcomes_observed") != 0
        or data.get("final_acceptance_gpu_outcomes_read") != 0
    ):
        raise ValueError("final blind design, prediction, or data binding drifted")
    live = read_json(ROOT / "config" / "experiment.json")
    scope = live.get("training_scope") or {}
    if (
        scope.get("phase_id") != prep.PHASE_ID
        or scope.get("gpu_ids") != list(prep.GPU_IDS)
        or scope.get("exclusive_node_gpu_ids") != list(prep.GPU_IDS)
        or int(scope.get("max_gpu_count", 0)) != 4
    ):
        raise ValueError("live experiment config is not the final business-blind scope")

    hardware = _hardware()
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(queue, rows, ROOT)
    launcher = ROOT / "final_memory_business_blind_v2_staging" / "launch_h800_final_memory_business_blind_v2.py"
    source_paths = [
        SCRIPTS / "approval_gate.py", SCRIPTS / "common.py", SCRIPTS / "metrics_callback.py",
        SCRIPTS / "promote_approval_candidate.py", SCRIPTS / "run_job.py", SCRIPTS / "runtime_evidence.py",
        SCRIPTS / "scheduler.py", SCRIPTS / "train_entry.py",
        SCRIPTS / "prepare_h800_final_memory_business_blind_v1.py",
        SCRIPTS / "prepare_h800_final_memory_business_blind_v2.py",
        SCRIPTS / "evaluate_h800_final_memory_business_blind_v2.py",
        Path(__file__).resolve(), launcher,
        ROOT / "config" / "experiment.json", ROOT / "config" / "hardware.json", ROOT / "config" / "models.json",
        ROOT / "config" / "deepspeed" / "ds_z2.json", ROOT / "config" / "deepspeed" / "ds_z3.json",
    ]
    provenance_binding = build_provenance_binding(ROOT, source_paths=source_paths)
    bound_files = {
        queue, design_path, predictions_path, data_path, experiment_path,
        prep.V3_ARTIFACT, prep.MEASUREMENT_GATE, prep.SELECTION, prep.MODEL_INVENTORY,
        prep.DATASET_INFO, ARTIFACT_DIR / "provenance.json", ARTIFACT_DIR / "model_inventory.json",
        ROOT / "config" / "experiment.json", ROOT / "config" / "models.json", *source_paths,
    }
    for row in rows:
        bound_files.update({Path(str(row["data_path"])), Path(str(row["dataset_profile_path"])), Path(str(row["runtime_dataset_profile_path"])), Path(str(row["model_path"]))})
    missing = [str(path) for path in bound_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"approval-bound paths are absent: {missing}")
    manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=str)
        if path.is_file() and path.resolve().is_relative_to(ROOT.resolve())
    }
    approval_design = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{prep.CAMPAIGN_ID}:{prep.PHASE_ID}",
        "authorization": AUTHORIZATION,
        "campaign_design": {"path": str(design_path.resolve().relative_to(ROOT.resolve())), "sha256": sha256_file(design_path)},
        "file_sha256": manifest,
        "execution_order": [prep.PHASE_ID],
        "allowed_job_ids": [str(row["job_id"]) for row in rows],
        "authorized_gpu_ids": list(prep.GPU_IDS),
        "max_gpu_count": 4,
        "scheduler_execution": {"join_busy_pool": False, "preemption_allowed": False, "policy": "parallel disjoint 1/2-GPU jobs and exclusive 4-GPU jobs on GPU 0-3 only"},
        "oom_policy": "valid_final_acceptance_outcome_no_retry",
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": {
            "allowed_job_ids": queue_binding["ordered_job_ids"],
            "queue_path": queue_binding["path"], "queue_sha256": queue_binding["sha256"],
            "ordered_job_ids": queue_binding["ordered_job_ids"],
            "ordered_job_payload_sha256": queue_binding["ordered_job_payload_sha256"],
            "job_payload_sha256": queue_binding["job_payload_sha256"],
        },
    }
    write_json(candidate_path, approval_design)
    return candidate_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=prep.DEFAULT_QUEUE)
    parser.add_argument("--design", type=Path, default=prep.DEFAULT_DESIGN)
    parser.add_argument("--predictions", type=Path, default=prep.DEFAULT_PREDICTIONS)
    parser.add_argument("--data-bundle", type=Path, default=prep.DEFAULT_DATA_BUNDLE)
    parser.add_argument("--experiment", type=Path, default=prep.DEFAULT_EXPERIMENT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = freeze(args.queue.resolve(), args.design.resolve(), args.predictions.resolve(), args.data_bundle.resolve(), args.experiment.resolve(), args.candidate.resolve())
    digest = sha256_file(candidate)
    report = promote_candidate(
        candidate_path=candidate, expected_candidate_sha256=digest, project_root=ROOT,
        authorization=AUTHORIZATION, approved_by="user", promote=args.promote,
    )
    print(json.dumps({"candidate": str(candidate), "sha256": digest, "jobs": prep.EXPECTED_JOBS, "authorized_gpu_ids": list(prep.GPU_IDS), "promotion": report}, ensure_ascii=False, indent=2))
    if report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
