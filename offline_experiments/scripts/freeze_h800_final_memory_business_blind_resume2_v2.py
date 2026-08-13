#!/usr/bin/env python3
"""Freeze and promote the exact ten-job 32B final-blind resume-2 subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from freeze_h800_final_memory_business_blind_v2 import _hardware
import prepare_h800_final_memory_business_blind_resume2_v2 as resume
from prepare_h800_final_memory_business_blind_v2 import implementation as prep
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch, validate_job

AUTHORIZATION = (
    "用户要求继续完成最终显存盲测。本批准仅允许续跑原结果前冻结队列中从未启动的10个32B、4卡、ZeRO-3任务；"
    "每行payload必须与原队列完全一致，仅使用GPU 0-3，不重跑已有结果，不运行Qwen3.5或packing无效任务，"
    "不修改冻结预测或线上模型。"
)
DEFAULT_CANDIDATE = ARTIFACT_DIR / "approval_design_h800_final_memory_business_blind_resume2_v2_candidate.json"
SCRIPTS = ROOT / "scripts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(resume.DEFAULT_QUEUE)
    original = {str(row["job_id"]): row for row in read_jsonl(resume.ORIGINAL_QUEUE)}
    design = read_json(resume.DEFAULT_DESIGN)
    if (
        len(rows) != resume.EXPECTED_RESUME
        or len({str(row["job_id"]) for row in rows}) != resume.EXPECTED_RESUME
        or any(original.get(str(row["job_id"])) != row for row in rows)
        or any(str(row["scenario"]) not in resume.SCENARIOS for row in rows)
        or any(int(row["gpu_count"]) != 4 or int(row["zero_stage"]) != 3 or bool(row["packing"]) for row in rows)
        or any((ROOT / "results" / str(row["job_id"]) / "status.json").exists() for row in rows)
        or design.get("resume_job_ids") != [str(row["job_id"]) for row in rows]
        or design.get("resume_ordered_payload_sha256") != sha256_json(rows)
        or design.get("bindings", {}).get("resume_queue", {}).get("sha256") != sha256_file(resume.DEFAULT_QUEUE)
        or design.get("bindings", {}).get("original_predictions", {}).get("sha256") != sha256_file(resume.ORIGINAL_PREDICTIONS)
    ):
        raise ValueError("exact resume-2 subset or frozen binding drifted")
    for row in rows:
        validate_job(row)
    live = read_json(ROOT / "config" / "experiment.json")
    scope = live.get("training_scope") or {}
    if scope.get("phase_id") != resume.PHASE_ID or scope.get("gpu_ids") != list(prep.GPU_IDS) or int(scope.get("max_gpu_count", 0)) != 4:
        raise ValueError("live config is not the exact resume-2 scope")
    hardware = _hardware()
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError("runtime patch is unhealthy")
    queue_binding = build_queue_binding(resume.DEFAULT_QUEUE, rows, ROOT)
    launcher = ROOT / "final_memory_business_blind_resume2_v2_staging" / "launch_h800_final_memory_business_blind_resume2_v2.py"
    sources = [
        SCRIPTS / "approval_gate.py", SCRIPTS / "common.py", SCRIPTS / "metrics_callback.py",
        SCRIPTS / "promote_approval_candidate.py", SCRIPTS / "run_job.py", SCRIPTS / "runtime_evidence.py",
        SCRIPTS / "scheduler.py", SCRIPTS / "train_entry.py",
        SCRIPTS / "prepare_h800_final_memory_business_blind_resume2_v2.py",
        Path(__file__).resolve(), launcher,
        ROOT / "config" / "experiment.json", ROOT / "config" / "hardware.json", ROOT / "config" / "models.json",
        ROOT / "config" / "deepspeed" / "ds_z3.json",
    ]
    provenance = build_provenance_binding(ROOT, source_paths=sources)
    bound = {
        resume.DEFAULT_QUEUE, resume.DEFAULT_DESIGN, resume.DEFAULT_EXPERIMENT,
        resume.ORIGINAL_QUEUE, resume.ORIGINAL_DESIGN, resume.ORIGINAL_PREDICTIONS,
        resume.PRIOR_RESUME_DESIGN, prep.V3_ARTIFACT, prep.MEASUREMENT_GATE, prep.DATASET_INFO,
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
        "design_purpose": f"{resume.CAMPAIGN_ID}:{resume.PHASE_ID}",
        "authorization": AUTHORIZATION,
        "campaign_design": {"path": str(resume.DEFAULT_DESIGN.resolve().relative_to(ROOT.resolve())), "sha256": sha256_file(resume.DEFAULT_DESIGN)},
        "file_sha256": manifest,
        "execution_order": [resume.PHASE_ID],
        "allowed_job_ids": [str(row["job_id"]) for row in rows],
        "authorized_gpu_ids": list(prep.GPU_IDS),
        "max_gpu_count": 4,
        "scheduler_execution": {"join_busy_pool": False, "preemption_allowed": False, "policy": "exact untouched 32B subset on GPU 0-3"},
        "oom_policy": "valid_original_final_acceptance_outcome_no_retry",
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance,
        "queue_binding": queue_binding,
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
    print(json.dumps({"candidate": str(DEFAULT_CANDIDATE), "sha256": digest, "jobs": resume.EXPECTED_RESUME, "promotion": report}, ensure_ascii=False, indent=2))
    if report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
