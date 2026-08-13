#!/usr/bin/env python3
"""Validate and freeze the narrowly scoped H800 LoRA + ZeRO-3 fix canaries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    MATRIX_DIR,
    RESULTS_DIR,
    ROOT,
    RUNTIME_DIR,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from run_job import live_runtime_identity
from validate_setup import freeze_approval_design


QUEUE_PATH = MATRIX_DIR / "h800_lora_zero3_fix_canary_jobs.jsonl"
VALIDATION_PATH = ARTIFACT_DIR / "h800_lora_zero3_fix_validation.json"
PENDING_PATH = RUNTIME_DIR / "pipeline" / "pending-h800-lora-zero3-fix-canary.jsonl"
INSTALLED_DS_SOURCE = Path(
    "/fine-tuning-launcher/.venv/lib/python3.11/site-packages/"
    "deepspeed/runtime/zero/partition_parameters.py"
)
RUNTIME_PATCHER = Path("/fine-tuning-launcher/hack/patches/apply_deepspeed_zero3_mixed_dtype_fix.py")
BUILD_PATCHER = Path(
    "/fine-tuning-launcher/dev/finetuning-launcher/"
    "hack/patches/apply_deepspeed_zero3_mixed_dtype_fix.py"
)
BUILD_ENTRYPOINT = Path("/fine-tuning-launcher/dev/finetuning-launcher/hack/entrypoint.sh")
BUILD_DOCKERFILES = (
    Path("/fine-tuning-launcher/dev/finetuning-launcher/Dockerfile"),
    Path("/fine-tuning-launcher/dev/finetuning-launcher/Dockerfile.qwen36"),
)


def validate_jobs(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {
        "canary-z3fix-qwen3-8b-2g-20260721": ("qwen3_8b", 2),
        "canary-z3fix-qwen3-8b-4g-20260721": ("qwen3_8b", 4),
        "canary-z3fix-qwen3-14b-4g-20260721": ("qwen3_14b", 4),
    }
    ids = [str(job.get("job_id")) for job in jobs]
    shape_errors = []
    for job in jobs:
        expected_shape = expected.get(str(job.get("job_id")))
        actual_shape = (job.get("model_id"), int(job.get("gpu_count") or 0))
        fixed_fields_ok = all(
            (
                job.get("kind") == "runtime_canary",
                job.get("train_type") == "lora",
                job.get("zero") == "zero3",
                job.get("dataset_id") == "short_512",
                int(job.get("cutoff_len") or 0) == 512,
                int(job.get("mbs") or 0) == 1,
                int(job.get("target_gbs") or 0) == 16,
                job.get("gc") is False,
                int(job.get("warmup_steps") or 0) == 0,
                int(job.get("measure_steps") or 0) == 3,
            )
        )
        if expected_shape != actual_shape or not fixed_fields_ok:
            shape_errors.append(str(job.get("job_id")))
    return {
        "jobs": len(jobs),
        "job_ids": ids,
        "duplicate_job_ids": sorted({job_id for job_id in ids if ids.count(job_id) > 1}),
        "missing_job_ids": sorted(set(expected) - set(ids)),
        "unexpected_job_ids": sorted(set(ids) - set(expected)),
        "shape_errors": sorted(shape_errors),
    }


def validate_results(jobs: list[dict[str, Any]], current_ds_source_sha256: str) -> dict[str, Any]:
    rows = []
    for job in jobs:
        job_id = str(job["job_id"])
        result_dir = RESULTS_DIR / job_id
        status_path = result_dir / "status.json"
        if not status_path.is_file():
            rows.append({"job_id": job_id, "state": "pending", "all_passed": True})
            continue
        status = read_json(status_path)
        rendered = read_json(result_dir / "rendered_run.json")
        runtime_identity = read_json(result_dir / "runtime_identity.json")
        summaries = [read_json(path) for path in sorted((result_dir / "metrics").glob("summary.rank*.json"))]
        train_results_path = result_dir / "trainer_output" / "train_results.json"
        train_results = read_json(train_results_path) if train_results_path.is_file() else {}
        log = (result_dir / "train.log").read_text(encoding="utf-8", errors="replace")
        loss = train_results.get("train_loss")
        checks = {
            "classification_success": status.get("classification") == "success",
            "runtime_fingerprint_bound": (
                status.get("runtime_fingerprint_sha256")
                and status.get("runtime_fingerprint_sha256") == rendered.get("runtime_fingerprint_sha256")
            ),
            "patched_deepspeed_source": (
                runtime_identity.get("framework_source_sha256", {}).get(
                    "deepspeed_zero_partition_parameters"
                )
                == current_ds_source_sha256
            ),
            "patcher_bound": runtime_identity.get("launcher_patch_sha256") == sha256_file(RUNTIME_PATCHER),
            "rank_summaries_complete": len(summaries) == int(job["gpu_count"]),
            "three_optimizer_steps": bool(summaries)
            and all(
                summary.get("failure") is None
                and int(summary.get("total_steps") or 0) >= 3
                and int(summary.get("measured_steps") or 0) >= 3
                for summary in summaries
            ),
            "finite_loss": isinstance(loss, (int, float)) and math.isfinite(float(loss)),
            "fp32_lora_preserved": "DeepSpeed ZeRO3 detected, remaining trainable params in float32" in log,
            "dtype_failure_absent": "output tensor must have the same type as input tensor" not in log,
        }
        rows.append(
            {
                "job_id": job_id,
                "state": "success" if all(checks.values()) else "invalid",
                "checks": checks,
                "train_loss": loss,
                "runtime_fingerprint_sha256": status.get("runtime_fingerprint_sha256"),
                "all_passed": all(checks.values()),
            }
        )
    return {
        "rows": rows,
        "successful_job_ids": [row["job_id"] for row in rows if row["state"] == "success"],
        "pending_job_ids": [row["job_id"] for row in rows if row["state"] == "pending"],
        "invalid_job_ids": [row["job_id"] for row in rows if row["state"] == "invalid"],
        "all_passed": all(row["all_passed"] for row in rows),
    }


def validate_patch() -> dict[str, Any]:
    required = (INSTALLED_DS_SOURCE, RUNTIME_PATCHER, BUILD_PATCHER, BUILD_ENTRYPOINT, *BUILD_DOCKERFILES)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"missing": missing, "all_passed": False}
    source = INSTALLED_DS_SOURCE.read_text(encoding="utf-8")
    unsafe = "dtype=param_list[0].ds_tensor.dtype" in source
    safe = (
        "dtype=param_list[param_idx].ds_tensor.dtype" in source
        or "dtype=param.ds_tensor.dtype" in source
    )
    patchers_match = sha256_file(RUNTIME_PATCHER) == sha256_file(BUILD_PATCHER)
    entrypoint = BUILD_ENTRYPOINT.read_text(encoding="utf-8")
    fine_tune_only = '${CLOUD_MAAS_CMD:-train}" = "train"' in entrypoint
    dockerfiles_apply = all(
        "apply_deepspeed_zero3_mixed_dtype_fix.py" in path.read_text(encoding="utf-8")
        for path in BUILD_DOCKERFILES
    )
    return {
        "missing": [],
        "installed_deepspeed_source_sha256": sha256_file(INSTALLED_DS_SOURCE),
        "runtime_patcher_sha256": sha256_file(RUNTIME_PATCHER),
        "build_patcher_sha256": sha256_file(BUILD_PATCHER),
        "patchers_match": patchers_match,
        "unsafe_first_parameter_dtype_absent": not unsafe,
        "safe_per_parameter_dtype_present": safe,
        "fine_tune_only_entrypoint": fine_tune_only,
        "dockerfiles_apply_patch": dockerfiles_apply,
        "build_files_sha256": {str(path): sha256_file(path) for path in (BUILD_ENTRYPOINT, *BUILD_DOCKERFILES)},
        "all_passed": not unsafe and safe and patchers_match and fine_tune_only and dockerfiles_apply,
    }


def main() -> None:
    jobs = read_jsonl(QUEUE_PATH)
    job_check = validate_jobs(jobs)
    job_check["all_passed"] = not any(
        job_check[key]
        for key in (
            "duplicate_job_ids",
            "missing_job_ids",
            "unexpected_job_ids",
            "shape_errors",
        )
    )
    patch_check = validate_patch()
    result_check = validate_results(jobs, patch_check.get("installed_deepspeed_source_sha256", ""))
    jobs_by_id = {str(job["job_id"]): job for job in jobs}
    pending_jobs = [jobs_by_id[job_id] for job_id in result_check["pending_job_ids"]]
    write_jsonl(PENDING_PATH, pending_jobs)
    runtime_identity = live_runtime_identity()
    validation = {
        "schema_version": 1,
        "purpose": "H800 Qwen3-8B/14B LoRA + ZeRO-3 mixed-dtype fix canary",
        "upstream_fix": {
            "pull_request": "https://github.com/deepspeedai/DeepSpeed/pull/8073",
            "commit": "b5b3fded4049d5e623aecdd0720d4b6b96a947af",
        },
        "queue_path": str(QUEUE_PATH.relative_to(ROOT)),
        "queue_sha256": sha256_file(QUEUE_PATH),
        "jobs": job_check,
        "results": result_check,
        "pending_queue_path": str(PENDING_PATH.relative_to(ROOT)),
        "pending_queue_sha256": sha256_file(PENDING_PATH),
        "patch": patch_check,
        "runtime_identity": runtime_identity,
        "all_passed": job_check["all_passed"] and patch_check["all_passed"] and result_check["all_passed"],
    }
    write_json(VALIDATION_PATH, validation)
    if not validation["all_passed"]:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    frozen = freeze_approval_design(
        extra_files=(QUEUE_PATH, PENDING_PATH, VALIDATION_PATH),
        extra_metadata={
            "design_purpose": validation["purpose"],
            "authorization_date": "2026-07-21",
            "authorized_gpu_ids": [1, 2, 3, 4],
            "allowed_job_ids": result_check["pending_job_ids"],
            "runtime_fix": validation["upstream_fix"],
            "completed_canary_job_ids": result_check["successful_job_ids"],
            "canary_policy": "Only pending jobs from the three optimizer-step canaries; no full-pipeline resume.",
        },
    )
    print(json.dumps({"validation": str(VALIDATION_PATH), "approval_design": frozen}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
