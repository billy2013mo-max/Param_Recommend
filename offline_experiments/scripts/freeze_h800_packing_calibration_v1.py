#!/usr/bin/env python3
"""Freeze and promote the exact 24-job Packing calibration approval."""

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
    "用户在 2026-08-03 明确要求继续按既定执行顺序推进 Packing 与 VL。"
    "本 approval 仅允许先在 H800 GPU 0、1 上执行已冻结的 24 个 text neat-Packing calibration jobs；"
    "不得占用其他 GPU、抢占外部任务、把 calibration 当作 prospective acceptance，或自动发布模型。"
)
CAMPAIGN_ID = "h800_packing_calibration_20260803_v1"
PHASE_ID = "h800_packing_calibration_v1"
JOB_SCHEMA = "sft_h800_packing_calibration_job/v1"
QUEUE = ROOT / "matrix" / "h800_packing_calibration_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_packing_calibration_design_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_calibration_queue_manifest_v1.json"
DECISIONS = ARTIFACT_DIR / "h800_packing_calibration_frozen_decisions_v1.json"
PREDICTIONS = ARTIFACT_DIR / "h800_packing_calibration_unpacked_predictions_v1.json"
INVENTORY = ARTIFACT_DIR / "h800_packing_calibration_model_inventory_v1.json"
CLOSEOUT = ARTIFACT_DIR / "h800_packing_vl_canary_closeout_v2.json"


def freeze() -> Path:
    rows = read_jsonl(QUEUE)
    if (
        len(rows) != 24
        or len({str(row.get("job_id")) for row in rows}) != 24
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
        or any((row.get("calibration_partition") or {}).get("role") != "calibration" for row in rows)
        or any(row.get("parallel_class") != "exclusive_node" for row in rows)
        or any(row.get("requires_external_node_idle") is not True for row in rows)
        or any(row.get("strict_queue_order") is not True for row in rows)
        or [row.get("execution_sequence_index") for row in rows] != list(range(24))
        or any(int(row.get("gpu_count", 0)) not in {1, 2} for row in rows)
        or any(row.get("packed_model_prediction_available_before_calibration") is not False for row in rows)
    ):
        raise ValueError("queue is not the exact frozen 24-job Packing calibration")
    counts = {}
    for row in rows:
        key = (str(row["family_id"]), str(row["packing_treatment"]))
        counts[key] = counts.get(key, 0) + 1
    if counts != {(family, treatment): 3 for family in ("C1", "C2", "C3", "C4") for treatment in ("unpacked", "packed")}:
        raise ValueError(f"Packing treatment balance drifted: {counts}")

    experiment = read_json(ROOT / "config" / "experiment.json")
    scope = experiment["training_scope"]
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("model_ids") != ["qwen3_8b", "qwen3_14b"]
        or scope.get("gpu_ids") != [0, 1]
        or scope.get("exclusive_node_gpu_ids") != [0, 1]
        or scope.get("max_gpu_count") != 2
        or scope.get("gpu_counts") != [1, 2]
        or experiment.get("measurement", {}).get("performance_parallelism") != "exclusive_pool"
        or experiment.get("measurement", {}).get("scheduler_order_policy") != "strict_queue_order"
    ):
        raise ValueError("experiment config is not the exact GPU-0,1 Packing calibration scope")

    design = read_json(DESIGN)
    queue_manifest = read_json(QUEUE_MANIFEST)
    decisions = read_json(DECISIONS)
    predictions = read_json(PREDICTIONS)
    closeout = read_json(CLOSEOUT)
    if (
        design.get("schema") != "sft_h800_packing_calibration_design/v1"
        or design.get("campaign_id") != CAMPAIGN_ID
        or design.get("phase_id") != PHASE_ID
        or design.get("gpu_training_started") is not False
        or design.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or design.get("queue", {}).get("ordered_job_ids") != [row["job_id"] for row in rows]
        or queue_manifest.get("design", {}).get("sha256") != sha256_file(DESIGN)
        or queue_manifest.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or decisions.get("generated_before_gpu") is not True
        or decisions.get("observed_decision_map") != {"C1": "off", "C2": "on", "C3": "off", "C4": "off"}
        or predictions.get("gpu_experiments_launched") is not False
        or closeout.get("all_passed") is not True
        or closeout.get("next_stage_allowed") != "packing_and_vl_calibration_materialization"
    ):
        raise ValueError("Packing calibration design or prerequisite drifted")

    hardware = probe_hardware(required_gpu_ids=(0, 1))
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"GPU 0,1 are not an exact H800 pool: {hardware}")
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(QUEUE, rows, ROOT)
    provenance_binding = build_provenance_binding(ROOT)

    bound_files = {
        QUEUE,
        DESIGN,
        QUEUE_MANIFEST,
        DECISIONS,
        PREDICTIONS,
        INVENTORY,
        CLOSEOUT,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "data" / "dataset_info.json",
    }
    for binding in (design.get("source_bindings") or {}).values():
        bound_files.add(Path(binding["path"]))
    for row in rows:
        bound_files.add(Path(row["data_path"]))
        bound_files.add(Path(row["dataset_profile_path"]))
    missing = [str(path) for path in bound_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files are absent: {missing}")
    file_manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=lambda value: str(value))
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
    approval = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}",
        "campaign_design": {
            "path": str(DESIGN.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(DESIGN),
            "report_sha256": design["report_sha256"],
        },
        "file_sha256": file_manifest,
        "execution_order": [PHASE_ID],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": [0, 1],
        "max_gpu_count": 2,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": "strict queue order; every job is exclusive within GPU 0,1; never preempt external processes",
        },
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": stage,
    }
    candidate = ARTIFACT_DIR / "approval_design_h800_packing_calibration_v1_candidate.json"
    write_json(candidate, approval)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true")
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
    print(json.dumps({"candidate": str(candidate), "sha256": digest, "promotion": report}, ensure_ascii=False, indent=2))
    if report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
