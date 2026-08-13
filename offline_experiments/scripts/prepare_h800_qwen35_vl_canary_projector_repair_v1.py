#!/usr/bin/env python3
"""Freeze the exact two-job projector-discovery Canary repair queue.

The original job payloads and job IDs are preserved.  Only the repository
runtime implementation changes, so a new attempt can replace the semantically
invalid language-only observations without fabricating a new experimental arm.
This script writes artifacts only; it never promotes approval or starts GPUs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json, write_jsonl
from prepare_h800_qwen35_vl_supplement_v1 import (
    CAMPAIGN_ID,
    CANARY_PHASE_ID,
    CANARY_QUEUE,
    DESIGN_SCHEMA,
    JOB_SCHEMA,
)


REPAIR_ID = "h800_qwen35_vl_canary_projector_repair_v1"
REPAIR_QUEUE = ROOT / "matrix" / f"{REPAIR_ID}.jsonl"
REPAIR_DESIGN = ARTIFACT_DIR / f"{REPAIR_ID}_design.json"
REPAIR_MANIFEST = ARTIFACT_DIR / f"{REPAIR_ID}_queue_manifest.json"
SOURCE_ACCEPTANCE = (
    ARTIFACT_DIR / "h800_qwen35_vl_supplement_canary_acceptance_v1.json"
)
EXPECTED_JOB_IDS = (
    "h800q35vl-6e523850dc7aebcd",
)


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _validate_failure_evidence() -> dict[str, Any]:
    report = read_json(SOURCE_ACCEPTANCE)
    semantic = {
        str(row.get("job_id")): row
        for row in report.get("media_and_train_scope_checks") or []
    }
    failed = sorted(
        job_id
        for job_id, row in semantic.items()
        if row.get("all_ranks_passed") is not True
    )
    if (
        report.get("schema")
        != "sft_h800_qwen35_vl_supplement_canary_acceptance/v1"
        or report.get("campaign_id") != CAMPAIGN_ID
        or report.get("phase_id") != CANARY_PHASE_ID
        or report.get("all_passed") is not False
        or failed != sorted(EXPECTED_JOB_IDS)
    ):
        raise ValueError(
            "repair source must be the exact two projector-trainability failures"
        )
    results = {
        str(row.get("job_id")): row for row in report.get("results") or []
    }
    for job_id in EXPECTED_JOB_IDS:
        rank_checks = semantic[job_id].get("rank_checks") or []
        if rank_checks:
            if any(
                (rank.get("checks") or {}).get("projector_trainability_exact")
                is not False
                for rank in rank_checks
            ):
                raise ValueError(f"{job_id} is not an observed projector failure")
            continue
        result = results.get(job_id) or {}
        status_path = Path(str(result.get("status_path") or ""))
        failure_log = status_path.parent / "train.log"
        if (
            result.get("classification") != "failed"
            or not status_path.is_file()
            or not failure_log.is_file()
            or "Projector+language LoRA discovery found no linear module below"
            not in failure_log.read_text(encoding="utf-8", errors="replace")
            or "['visual.merger']"
            not in failure_log.read_text(encoding="utf-8", errors="replace")
        ):
            raise ValueError(f"{job_id} is not the exact wrapper-path repair failure")
    return report


def prepare() -> dict[str, Any]:
    source_report = _validate_failure_evidence()
    parent_rows = read_jsonl(CANARY_QUEUE)
    by_id = {str(row.get("job_id")): row for row in parent_rows}
    if len(parent_rows) != 15 or set(EXPECTED_JOB_IDS) - set(by_id):
        raise ValueError("parent Canary queue identity drifted")
    rows = [by_id[job_id] for job_id in EXPECTED_JOB_IDS]
    for row in rows:
        if (
            row.get("schema") != JOB_SCHEMA
            or row.get("campaign_id") != CAMPAIGN_ID
            or row.get("phase_id") != CANARY_PHASE_ID
            or row.get("track") != "train_scope_canary"
            or row.get("train_scope_id") != "projector_plus_language"
            or row.get("freeze_vision_tower") is not True
            or row.get("freeze_multi_modal_projector") is not False
            or row.get("freeze_language_model") is not False
        ):
            raise ValueError(f"repair row scope drifted: {row.get('job_id')}")

    write_jsonl(REPAIR_QUEUE, rows)
    queue = {
        **_binding(REPAIR_QUEUE),
        "job_count": len(rows),
        "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in rows),
        "ordered_job_ids": [str(row["job_id"]) for row in rows],
        "ordered_job_payload_sha256": sha256_json(rows),
    }
    design: dict[str, Any] = {
        "schema": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": CANARY_PHASE_ID,
        "repair_id": REPAIR_ID,
        "generated_at_utc": "2026-08-09T16:45:00+00:00",
        "status": "frozen_projector_discovery_repair_before_gpu",
        "gpu_training_started": False,
        "execution_authorized": False,
        "publication_allowed": False,
        "fit_allowed": False,
        "acceptance_allowed": False,
        "continuation_only": True,
        "same_job_ids_and_payloads": True,
        "required_gpu_pool": {
            "gpu_ids": list(range(8)),
            "max_gpu_count": 1,
            "preemption_allowed": False,
        },
        "queue": queue,
        "repair_contract": {
            "root_cause": (
                "installed LLaMA-Factory all-target discovery unconditionally "
                "excluded registered multimodal projector paths"
            ),
            "implementation": (
                "repository-scoped train_entry compatibility wrapper adds only "
                "linear suffixes found below registered projector paths; upstream "
                "freeze/conflict filtering remains authoritative"
            ),
            "rerun_policy": (
                "rerun only the two semantically failed projector+language Canary "
                "jobs and evaluate the latest attempts together with the other 13"
            ),
            "formal_gate": "full 15/15 semantic Canary must pass after repair",
        },
        "source_bindings": {
            "parent_canary_queue": _binding(CANARY_QUEUE),
            "failed_canary_acceptance": _binding(SOURCE_ACCEPTANCE),
            "preparer": _binding(Path(__file__).resolve()),
            **{
                f"failed_status_{job_id}": _binding(
                    Path(
                        next(
                            row["status_path"]
                            for row in source_report["results"]
                            if row["job_id"] == job_id
                        )
                    )
                )
                for job_id in EXPECTED_JOB_IDS
            },
            **{
                f"failed_log_{job_id}": _binding(
                    Path(
                        next(
                            row["status_path"]
                            for row in source_report["results"]
                            if row["job_id"] == job_id
                        )
                    ).parent
                    / "train.log"
                )
                for job_id in EXPECTED_JOB_IDS
            },
        },
        "failed_acceptance_report_sha256": source_report.get("report_sha256"),
    }
    design["report_sha256"] = sha256_json(design)
    write_json(REPAIR_DESIGN, design)
    manifest: dict[str, Any] = {
        "schema": "sft_h800_qwen35_vl_canary_projector_repair_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": CANARY_PHASE_ID,
        "repair_id": REPAIR_ID,
        "gpu_training_started": False,
        "design": _binding(REPAIR_DESIGN),
        "queue": queue,
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(REPAIR_MANIFEST, manifest)
    return {
        "repair_id": REPAIR_ID,
        "queue": str(REPAIR_QUEUE),
        "jobs": [str(row["job_id"]) for row in rows],
        "design": str(REPAIR_DESIGN),
        "manifest": str(REPAIR_MANIFEST),
        "gpu_training_started": False,
    }


def main() -> None:
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
