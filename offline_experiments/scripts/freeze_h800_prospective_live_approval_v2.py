#!/usr/bin/env python3
"""Freeze and optionally promote the exact H800 fresh-holdout queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch


AUTHORIZATION = (
    "用户明确要求把剩余的 prospective holdout 四卡任务切换到空闲 H800 GPU 0,1,2,3；"
    "不得占用 GPU 4-7、终止外部进程、重跑已成功任务或扩展到未批准任务。"
)
DEFAULT_QUEUE = ROOT / "matrix" / "h800_fresh_holdout_jobs_v2.jsonl"
DEFAULT_CAMPAIGN_DESIGN = ARTIFACT_DIR / "h800_fresh_holdout_design_v2.json"
DEFAULT_REQUIREMENTS = ARTIFACT_DIR / "h800_fresh_profile_requirements_v2.json"


def freeze(
    queue: Path,
    campaign_design: Path,
    requirements: Path,
    stage: str,
) -> Path:
    rows = read_jsonl(queue)
    experiment = read_json(ROOT / "config" / "experiment.json")
    queue_binding = build_queue_binding(queue, rows, ROOT)
    provenance_binding = build_provenance_binding(ROOT)
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    ids = [str(row["job_id"]) for row in rows]
    bound_files = [
        queue,
        campaign_design,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "h800_frozen_predictions_before_fresh_holdout_v2.json",
        requirements,
        ARTIFACT_DIR / "h800_fresh_business_data_bundle_v2.json",
        ROOT / "data" / "dataset_info.json",
    ]
    manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in bound_files
    }
    stage_binding = {
        "allowed_job_ids": ids,
        "queue_path": queue_binding["path"],
        "queue_sha256": queue_binding["sha256"],
        "ordered_job_ids": queue_binding["ordered_job_ids"],
        "ordered_job_payload_sha256": queue_binding["ordered_job_payload_sha256"],
        "job_payload_sha256": queue_binding["job_payload_sha256"],
    }
    scope = experiment["training_scope"]
    design = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{read_json(campaign_design)['campaign_id']}:{stage}",
        "campaign_design": {
            "path": str(campaign_design.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(campaign_design),
        },
        "file_sha256": manifest,
        "execution_order": [stage],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": list(scope["gpu_ids"]),
        "max_gpu_count": int(scope["max_gpu_count"]),
        "scheduler_execution": {
            "join_busy_pool": False,
            "preemption_allowed": False,
            "policy": "require the exact GPU 0-3 pool to be idle before scheduler start",
        },
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": stage_binding,
    }
    candidate = ARTIFACT_DIR / "approval_design_h800_fresh_holdout_v2_candidate.json"
    write_json(candidate, design)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--campaign-design", type=Path, default=DEFAULT_CAMPAIGN_DESIGN)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--stage", default="h800_fresh_business_holdout_v2")
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = freeze(
        args.queue.resolve(),
        args.campaign_design.resolve(),
        args.requirements.resolve(),
        args.stage,
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
    print(json.dumps({"candidate": str(candidate), "sha256": digest, "promotion": report}, ensure_ascii=False, indent=2))
    if report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
