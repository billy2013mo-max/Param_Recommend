#!/usr/bin/env python3
"""Validate and freeze a single-GPU throughput recovery queue for re-approval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, RESULTS_DIR, ROOT, read_json, read_jsonl, sha256_file, write_json
from rebuild_queue import active_job_ids, classification
from validate_setup import freeze_approval_design


def validate_recovery_jobs(
    source_jobs: list[dict[str, Any]],
    recovery_jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    source_by_id = {str(job["job_id"]): job for job in source_jobs}
    recovery_ids = [str(job["job_id"]) for job in recovery_jobs]
    duplicate_ids = sorted({job_id for job_id in recovery_ids if recovery_ids.count(job_id) > 1})
    outside_source = sorted(set(recovery_ids) - set(source_by_id))
    invalid_shape = sorted(
        job_id
        for job_id, job in zip(recovery_ids, recovery_jobs)
        if job.get("kind") != "throughput" or int(job.get("gpu_count") or 0) != 1
    )
    return {
        "source_jobs": len(source_jobs),
        "recovery_jobs": len(recovery_jobs),
        "duplicate_job_ids": duplicate_ids,
        "outside_source_job_ids": outside_source,
        "invalid_non_single_gpu_throughput_job_ids": invalid_shape,
        "all_passed": not duplicate_ids and not outside_source and not invalid_shape,
    }


def validate_smoke(job_id: str) -> dict[str, Any]:
    result_dir = RESULTS_DIR / job_id
    status = read_json(result_dir / "status.json")
    rendered = read_json(result_dir / "rendered_run.json")
    job = rendered["job"]
    current_provenance = sha256_file(ARTIFACT_DIR / "provenance.json")
    checks = {
        "classification_success": status.get("classification") == "success",
        "single_gpu": int(job.get("gpu_count") or 0) == 1 and rendered.get("gpu_mask") == "1",
        "bounded_smoke": (
            job.get("kind") == "smoke"
            and job.get("model_id") == "qwen3_1p7b"
            and job.get("dataset_id") == "short_512"
            and int(job.get("warmup_steps") or 0) == 0
            and int(job.get("measure_steps") or 0) == 1
        ),
        "current_provenance": rendered.get("provenance_sha256") == current_provenance,
    }
    return {
        "job_id": job_id,
        "status_path": str((result_dir / "status.json").relative_to(ROOT)),
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recovery-queue", type=Path, required=True)
    parser.add_argument("--smoke-job-id", required=True)
    args = parser.parse_args()

    source_path = args.source.resolve()
    recovery_queue_path = args.recovery_queue.resolve()
    source_jobs = read_jsonl(source_path)
    recovery_jobs = read_jsonl(recovery_queue_path)
    queue_check = validate_recovery_jobs(source_jobs, recovery_jobs)
    recovery_ids = [str(job["job_id"]) for job in recovery_jobs]
    live = sorted(active_job_ids(recovery_ids))
    statuses = {
        job_id: outcome
        for job_id in recovery_ids
        if (outcome := classification(job_id)) is not None
    }
    smoke = validate_smoke(args.smoke_job_id)
    all_passed = queue_check["all_passed"] and not live and not statuses and smoke["all_passed"]
    validation = {
        "schema_version": 1,
        "purpose": "single-GPU throughput recovery while an already-approved GPU-2 job remains active",
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "recovery_queue": str(recovery_queue_path),
        "recovery_queue_sha256": sha256_file(recovery_queue_path),
        "queue": queue_check,
        "live_recovery_job_ids": live,
        "recovery_jobs_with_status": statuses,
        "smoke": smoke,
        "all_passed": all_passed,
    }
    validation_path = ARTIFACT_DIR / "recovery_validation.json"
    write_json(validation_path, validation)
    if not all_passed:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    approval_design = freeze_approval_design(
        extra_files=(recovery_queue_path, validation_path),
        extra_metadata={
            "recovery_scope": {
                "kind": "throughput",
                "gpu_count": 1,
                "gpu_ids": [1, 2, 3, 4],
                "jobs": len(recovery_jobs),
                "queue_sha256": validation["recovery_queue_sha256"],
                "validation_sha256": sha256_file(validation_path),
            }
        },
    )
    print(
        json.dumps(
            {
                "recovery_validation": str(validation_path),
                "jobs": len(recovery_jobs),
                "approval_design": approval_design,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
