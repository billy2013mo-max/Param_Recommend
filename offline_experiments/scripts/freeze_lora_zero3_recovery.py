#!/usr/bin/env python3
"""Freeze only the memory families blocked by DeepSpeed's LoRA ZeRO-3 dtype bug."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, RESULTS_DIR, ROOT, RUNTIME_DIR, read_json, sha256_file, write_json, write_jsonl
from freeze_lora_zero3_fix import VALIDATION_PATH as CANARY_VALIDATION_PATH
from freeze_lora_zero3_fix import validate_patch
from run_pipeline import memory_progress
from validate_setup import freeze_approval_design


KNOWN_ERROR = "TypeError: output tensor must have the same type as input tensor"
QUEUE_PATH = RUNTIME_DIR / "pipeline" / "pending-h800-lora-zero3-fix-families.jsonl"
VALIDATION_PATH = ARTIFACT_DIR / "h800_lora_zero3_recovery_validation.json"


def validate_family_shape(family: dict[str, Any]) -> bool:
    return all(
        (
            family.get("kind") == "memory_boundary",
            family.get("model_id") in {"qwen3_8b", "qwen3_14b"},
            family.get("train_type") == "lora",
            family.get("zero") == "zero3",
            int(family.get("gpu_count") or 0) in {2, 4},
            bool(family.get("mbs_candidates")),
        )
    )


def known_failure_evidence(family: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mbs in family.get("mbs_candidates") or ():
        trial_job_id = f"{family['job_id']}-mbs{mbs}"
        result_dir = RESULTS_DIR / trial_job_id
        status_path = result_dir / "status.json"
        log_path = result_dir / "train.log"
        if not status_path.is_file() or not log_path.is_file():
            continue
        status = read_json(status_path)
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if status.get("classification") != "failed" or KNOWN_ERROR not in log:
            continue
        rendered_path = result_dir / "rendered_run.json"
        rendered = read_json(rendered_path) if rendered_path.is_file() else {}
        config_path = Path(rendered.get("config_path", "")) if rendered.get("config_path") else None
        rows.append(
            {
                "trial_job_id": trial_job_id,
                "mbs": int(mbs),
                "classification": status.get("classification"),
                "error_signature": KNOWN_ERROR,
                "status_sha256": sha256_file(status_path),
                "train_log_sha256": sha256_file(log_path),
                "rendered_run_sha256": sha256_file(rendered_path) if rendered_path.is_file() else None,
                "rendered_provenance_sha256": rendered.get("provenance_sha256"),
                "config_sha256": sha256_file(config_path) if config_path and config_path.is_file() else None,
            }
        )
    return rows


def main() -> None:
    progress = memory_progress()
    families = list(progress["missing"])
    shape_errors = sorted(str(family.get("job_id")) for family in families if not validate_family_shape(family))
    evidence = {str(family["job_id"]): known_failure_evidence(family) for family in families}
    missing_evidence = sorted(family_id for family_id, rows in evidence.items() if not rows)
    duplicate_ids = sorted(
        {
            str(family["job_id"])
            for family in families
            if [str(row["job_id"]) for row in families].count(str(family["job_id"])) > 1
        }
    )
    canary = read_json(CANARY_VALIDATION_PATH)
    canary_successes = set((canary.get("results") or {}).get("successful_job_ids") or ())
    expected_canaries = {
        "canary-z3fix-qwen3-8b-2g-20260721",
        "canary-z3fix-qwen3-8b-4g-20260721",
        "canary-z3fix-qwen3-14b-4g-20260721",
    }
    patch = validate_patch()
    concrete_job_ids = [
        f"{family['job_id']}-mbs{mbs}"
        for family in families
        for mbs in family.get("mbs_candidates") or ()
    ]
    write_jsonl(QUEUE_PATH, families)
    checks = {
        "exactly_twenty_missing_families": len(families) == 20,
        "no_existing_exclusions": not progress["excluded"],
        "all_family_shapes_expected": not shape_errors,
        "no_duplicate_family_ids": not duplicate_ids,
        "all_families_have_archived_known_bug_evidence": not missing_evidence,
        "all_three_canaries_passed": canary.get("all_passed") is True and canary_successes == expected_canaries,
        "runtime_patch_valid": patch.get("all_passed") is True,
    }
    validation = {
        "schema_version": 1,
        "purpose": "Retry only the 20 H800 LoRA + ZeRO-3 memory families blocked by DeepSpeed PR #8073",
        "known_error": KNOWN_ERROR,
        "checks": checks,
        "memory_progress_before_recovery": {
            "total": progress["total"],
            "summarized": progress["summarized"],
            "excluded": len(progress["excluded"]),
            "missing": len(progress["missing"]),
        },
        "family_job_ids": [str(family["job_id"]) for family in families],
        "shape_errors": shape_errors,
        "duplicate_family_ids": duplicate_ids,
        "missing_failure_evidence": missing_evidence,
        "archived_failure_evidence": evidence,
        "potential_concrete_job_ids": concrete_job_ids,
        "queue_path": str(QUEUE_PATH.relative_to(ROOT)),
        "queue_sha256": sha256_file(QUEUE_PATH),
        "canary_validation_path": str(CANARY_VALIDATION_PATH.relative_to(ROOT)),
        "canary_validation_sha256": sha256_file(CANARY_VALIDATION_PATH),
        "patch": patch,
        "all_passed": all(checks.values()),
    }
    write_json(VALIDATION_PATH, validation)
    if not validation["all_passed"]:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    frozen = freeze_approval_design(
        extra_files=(QUEUE_PATH, CANARY_VALIDATION_PATH, VALIDATION_PATH),
        extra_metadata={
            "design_purpose": validation["purpose"],
            "authorization_date": "2026-07-21",
            "authorized_gpu_ids": [1, 2, 3, 4],
            "allowed_family_job_ids": validation["family_job_ids"],
            "allowed_job_ids": concrete_job_ids,
            "runtime_fix": canary["upstream_fix"],
            "recovery_policy": (
                "Adaptive boundary retry for exactly 20 known-bug families; stop each family at first OOM; "
                "any non-OOM failure remains blocking."
            ),
        },
    )
    print(
        json.dumps(
            {
                "families": len(families),
                "potential_concrete_jobs": len(concrete_job_ids),
                "validation": str(VALIDATION_PATH),
                "approval_design": frozen,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
