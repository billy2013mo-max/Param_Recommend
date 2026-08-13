#!/usr/bin/env python3
"""Freeze and promote the 15-row executor-corrected prospective supplement."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from freeze_h800_final_memory_business_blind_v2 import _hardware
import prepare_h800_final_memory_business_blind_corrections_v1 as prep
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch, validate_job

AUTHORIZATION = (
    "用户要求继续在GPU 4-7完成并验收显存模型。本批准仅允许在GPU 4-7运行15个前瞻性校正任务：5个Qwen3.5短文本任务绑定"
    "既有验证运行时，10个packing任务固定物理MBS=1，其中双卡Qwen3.5使用ZeRO-2；V3及安全余量不得重拟合，"
    "冻结后不得改动payload，不重跑原始失败任务，不修改线上模型。"
)
DEFAULT_CANDIDATE = ARTIFACT_DIR / "approval_design_h800_final_memory_business_blind_corrections_v1_candidate.json"
SCRIPTS = ROOT / "scripts"


def _validate(rows: list[dict], predictions: dict, design: dict) -> dict[str, bool]:
    by_scenario = Counter(str(row["scenario"]) for row in rows)
    qwen35 = [row for row in rows if str(row["model_id"]) == "qwen3p5_4b"]
    packing = [row for row in rows if bool(row["packing"])]
    return {
        "15_unique_jobs": len(rows) == prep.EXPECTED_JOBS and len({str(row["job_id"]) for row in rows}) == prep.EXPECTED_JOBS,
        "three_complete_ladders": len(by_scenario) == 3 and set(by_scenario.values()) == {5},
        "five_target_pressures_per_scenario": all({float(row["target_pressure"]) for row in rows if row["scenario"] == scenario} == {0.8, 0.93, 0.99, 1.01, 1.08} for scenario in by_scenario),
        "packing_physical_mbs_one": len(packing) == 10 and all(int(row["mbs"]) == 1 for row in packing),
        "multi_card_zero2_or_zero3": all(int(row["gpu_count"]) == 1 or int(row["zero_stage"]) in {2, 3} for row in rows),
        "qwen35_native_template": len(qwen35) == 10 and all(str(row["template"]) == "qwen3_5_nothink" for row in qwen35),
        "qwen35_overlay_exact": len(qwen35) == 10 and all((row.get("environment_overlay") or {}).get("contract_path") == str(prep.QWEN35_RUNTIME_CONTRACT.resolve()) and (row.get("environment_overlay") or {}).get("contract_sha256") == sha256_file(prep.QWEN35_RUNTIME_CONTRACT) for row in qwen35),
        "predictions_frozen_for_own_outcomes": predictions.get("status") == "frozen_before_any_correction_gpu_outcome" and predictions.get("outcomes_observed") == 0 and predictions.get("prior_campaign_gpu_outcomes_observed_at_freeze") == 45,
        "prediction_payload_bound": predictions.get("ordered_job_payload_sha256") == sha256_json(rows),
        "design_payload_bound": design.get("ordered_job_ids") == [str(row["job_id"]) for row in rows] and design.get("ordered_job_payload_sha256") == sha256_json(rows),
        "no_existing_results": not any((ROOT / "results" / str(row["job_id"])).exists() for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(prep.DEFAULT_QUEUE)
    predictions = read_json(prep.DEFAULT_PREDICTIONS)
    design = read_json(prep.DEFAULT_DESIGN)
    checks = _validate(rows, predictions, design)
    if not all(checks.values()):
        raise ValueError(f"correction freeze validation failed: {checks}")
    for row in rows:
        validate_job(row)
    live = read_json(ROOT / "config" / "experiment.json")
    scope = live.get("training_scope") or {}
    if scope.get("phase_id") != prep.PHASE_ID or scope.get("gpu_ids") != list(prep.GPU_IDS) or int(scope.get("max_gpu_count", 0)) != 4:
        raise ValueError("live config is not the correction scope")
    hardware = _hardware()
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError("runtime patch is unhealthy")
    queue_binding = build_queue_binding(prep.DEFAULT_QUEUE, rows, ROOT)
    launcher = ROOT / "final_memory_business_blind_corrections_v1_staging" / "launch_h800_final_memory_business_blind_corrections_v1.py"
    sources = [
        SCRIPTS / "approval_gate.py", SCRIPTS / "common.py", SCRIPTS / "metrics_callback.py",
        SCRIPTS / "promote_approval_candidate.py", SCRIPTS / "run_job.py", SCRIPTS / "runtime_evidence.py",
        SCRIPTS / "scheduler.py", SCRIPTS / "train_entry.py",
        SCRIPTS / "prepare_h800_bounded_memory_v2_fresh_data_v1.py",
        SCRIPTS / "prepare_h800_final_memory_business_blind_v1.py",
        SCRIPTS / "prepare_h800_final_memory_business_blind_v2.py",
        SCRIPTS / "prepare_h800_final_memory_business_blind_corrections_v1.py",
        SCRIPTS / "evaluate_h800_final_memory_business_blind_corrections_v1.py",
        ROOT / "final_memory_allocator_prefix_replay_v2_staging" / "launch_h800_final_memory_allocator_prefix_replay_v2.py",
        Path(__file__).resolve(), launcher,
        ROOT / "config" / "experiment.json", ROOT / "config" / "hardware.json", ROOT / "config" / "models.json",
        ROOT / "config" / "deepspeed" / "ds_z2.json",
    ]
    provenance = build_provenance_binding(ROOT, source_paths=sources)
    bound = {
        prep.DEFAULT_QUEUE, prep.DEFAULT_DESIGN, prep.DEFAULT_EXPERIMENT,
        prep.DEFAULT_PREDICTIONS, prep.DEFAULT_DATA_BUNDLE,
        prep.V3_ARTIFACT, prep.MEASUREMENT_GATE, prep.DATASET_INFO, prep.QWEN35_RUNTIME_CONTRACT,
        ARTIFACT_DIR / "provenance.json", ARTIFACT_DIR / "model_inventory.json", *sources,
    }
    for row in rows:
        bound.update({Path(str(row["data_path"])), Path(str(row["dataset_profile_path"])), Path(str(row["runtime_dataset_profile_path"])), Path(str(row["model_path"]))})
    missing = [str(path) for path in bound if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound, key=str)
        if path.is_file() and path.resolve().is_relative_to(ROOT.resolve())
    }
    candidate = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{prep.CAMPAIGN_ID}:{prep.PHASE_ID}",
        "authorization": AUTHORIZATION,
        "campaign_design": {"path": str(prep.DEFAULT_DESIGN.resolve().relative_to(ROOT.resolve())), "sha256": sha256_file(prep.DEFAULT_DESIGN)},
        "file_sha256": manifest,
        "execution_order": [prep.PHASE_ID],
        "allowed_job_ids": [str(row["job_id"]) for row in rows],
        "authorized_gpu_ids": list(prep.GPU_IDS),
        "max_gpu_count": 4,
        "scheduler_execution": {"join_busy_pool": False, "preemption_allowed": False, "policy": "prospective executor-corrected supplement on GPU 4-7"},
        "oom_policy": "valid_correction_evidence_no_retry",
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance,
        "queue_binding": queue_binding,
        "correction_static_checks": checks,
        "throughput_screen_delta": {
            "allowed_job_ids": queue_binding["ordered_job_ids"], "queue_path": queue_binding["path"],
            "queue_sha256": queue_binding["sha256"], "ordered_job_ids": queue_binding["ordered_job_ids"],
            "ordered_job_payload_sha256": queue_binding["ordered_job_payload_sha256"], "job_payload_sha256": queue_binding["job_payload_sha256"],
        },
    }
    write_json(DEFAULT_CANDIDATE, candidate)
    digest = sha256_file(DEFAULT_CANDIDATE)
    report = promote_candidate(
        candidate_path=DEFAULT_CANDIDATE, expected_candidate_sha256=digest, project_root=ROOT,
        authorization=AUTHORIZATION, approved_by="user", promote=args.promote,
    )
    print(json.dumps({"candidate": str(DEFAULT_CANDIDATE), "sha256": digest, "jobs": prep.EXPECTED_JOBS, "promotion": report}, ensure_ascii=False, indent=2))
    if report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
