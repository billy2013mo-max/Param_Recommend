#!/usr/bin/env python3
"""Freeze and promote only the unfinished rows of the combined VL formal queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from approval_gate import build_queue_binding
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
from promote_approval_candidate import promote_candidate


FULL_QUEUE = MATRIX_DIR / "h800_frozen_vl_combined_formal_v1.jsonl"
RESUME_QUEUE = MATRIX_DIR / "h800_frozen_vl_combined_formal_resume_v1.jsonl"
RESUME_DESIGN = ARTIFACT_DIR / "h800_frozen_vl_combined_formal_resume_design_v1.json"
CANDIDATE = (
    ARTIFACT_DIR
    / "approval_design_h800_frozen_vl_combined_formal_resume_v1_candidate.json"
)
LIVE_DESIGN = RUNTIME_DIR / "approval_design.json"
PREVIOUS_FORMAL_DESIGN_SHA256 = (
    "b49e0e2af4faeb079fd9555b091c3d0fdcfe9f9a4cdb993eacbfdc161192d443"
)
AUTHORIZATION = (
    "用户在 2026-08-12 明确要求改回来并继续运行 Qwen3.5/VL 实验；"
    "此前已明确授权使用物理 GPU 0-6。此恢复审批仅排除已有完整成功证据的 21 个作业，"
    "运行原 Formal 队列中尚未开始的 66 个作业；GPU7、抢占、packing、offload 和队列扩展均不允许。"
)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _completed_evidence(job: dict[str, Any]) -> dict[str, Any] | None:
    job_id = str(job["job_id"])
    result_dir = RESULTS_DIR / job_id
    status_path = result_dir / "status.json"
    latest_path = result_dir / "latest_attempt.json"
    if not status_path.is_file() and not latest_path.is_file():
        return None
    if not status_path.is_file() or not latest_path.is_file():
        raise RuntimeError(f"partial previous result cannot be skipped: {job_id}")
    status = read_json(status_path)
    latest = read_json(latest_path)
    attempt_id = str(latest.get("execution_attempt_id") or "")
    attempt_path = result_dir / str(latest.get("attempt_path") or "")
    attempt_status_path = attempt_path / "status.json"
    summary_path = attempt_path / "metrics" / "summary.rank0.json"
    if not attempt_status_path.is_file() or not summary_path.is_file():
        raise RuntimeError(f"previous success evidence is incomplete: {job_id}")
    attempt_status = read_json(attempt_status_path)
    summary = read_json(summary_path)
    checks = {
        "root_status_success": status.get("classification") == "success",
        "latest_complete_success": latest.get("state") == "complete"
        and latest.get("classification") == "success",
        "attempt_status_success": attempt_status.get("classification") == "success",
        "attempt_id_exact": bool(attempt_id)
        and status.get("execution_attempt_id") == attempt_id
        and attempt_status.get("execution_attempt_id") == attempt_id
        and summary.get("execution_attempt_id") == attempt_id,
        "job_id_exact": status.get("job_id") == job_id
        and attempt_status.get("job_id") == job_id
        and summary.get("job_id") == job_id,
        "return_code_zero": status.get("return_code") == 0
        and attempt_status.get("return_code") == 0,
        "measured_steps_exact": int(summary.get("measured_steps") or 0)
        == int(job["measure_steps"]),
        "positive_measurement": float(summary.get("measured_seconds") or 0.0) > 0.0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"previous result is not a complete success: {job_id}: {checks}")
    return {
        "job_id": job_id,
        "execution_attempt_id": attempt_id,
        "status_path": _relative(status_path),
        "status_sha256": sha256_file(status_path),
        "attempt_status_path": _relative(attempt_status_path),
        "attempt_status_sha256": sha256_file(attempt_status_path),
        "summary_path": _relative(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "measured_steps": int(summary["measured_steps"]),
        "checks": checks,
    }


def prepare(*, promote: bool) -> dict[str, Any]:
    if sha256_file(LIVE_DESIGN) != PREVIOUS_FORMAL_DESIGN_SHA256:
        raise RuntimeError("resume must be derived from the exact newly promoted 87-job design")
    full_rows = read_jsonl(FULL_QUEUE)
    if len(full_rows) != 87 or len({str(row["job_id"]) for row in full_rows}) != 87:
        raise RuntimeError("full formal queue is not the exact 87-row source queue")

    completed: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for row in full_rows:
        evidence = _completed_evidence(row)
        if evidence is None:
            remaining.append(row)
        else:
            completed.append(evidence)
    if len(completed) != 21 or len(remaining) != 66:
        raise RuntimeError(
            f"expected exact 21-success/66-pending split, got {len(completed)}/{len(remaining)}"
        )

    write_jsonl(RESUME_QUEUE, remaining)
    resume_design = {
        "schema": "sft_h800_frozen_vl_combined_formal_resume/v1",
        "source_queue": {"path": _relative(FULL_QUEUE), "sha256": sha256_file(FULL_QUEUE)},
        "resume_queue": {
            "path": _relative(RESUME_QUEUE),
            "sha256": sha256_file(RESUME_QUEUE),
            "ordered_job_ids": [str(row["job_id"]) for row in remaining],
        },
        "completed_count": len(completed),
        "remaining_count": len(remaining),
        "completed_evidence": completed,
        "policy": "skip_only_complete_success_with_exact_eight_step_summary",
    }
    write_json(RESUME_DESIGN, resume_design)

    approval_design = read_json(LIVE_DESIGN)
    approval_design["training_started"] = False
    approval_design["stage"] = "formal_resume"
    approval_design["design_purpose"] = (
        str(approval_design["design_purpose"]) + ":resume_21_complete_66_pending"
    )
    approval_design["allowed_job_ids"] = [str(row["job_id"]) for row in remaining]
    queue_binding = build_queue_binding(RESUME_QUEUE, remaining, ROOT)
    approval_design["queue_binding"] = queue_binding
    approval_design["throughput_screen_delta"] = {
        "allowed_job_ids": list(approval_design["allowed_job_ids"]),
        "queue_path": queue_binding["path"],
        "queue_sha256": queue_binding["sha256"],
        "ordered_job_ids": queue_binding["ordered_job_ids"],
        "ordered_job_payload_sha256": queue_binding["ordered_job_payload_sha256"],
        "job_payload_sha256": queue_binding["job_payload_sha256"],
    }
    approval_design["resume"] = {
        "design_path": _relative(RESUME_DESIGN),
        "design_sha256": sha256_file(RESUME_DESIGN),
        "completed_job_ids": [row["job_id"] for row in completed],
        "remaining_job_ids": list(approval_design["allowed_job_ids"]),
    }
    manifest = dict(approval_design["file_sha256"])
    for path in (RESUME_QUEUE, RESUME_DESIGN, Path(__file__)):
        manifest[_relative(path)] = sha256_file(path)
    for evidence in completed:
        for path_key, sha_key in (
            ("status_path", "status_sha256"),
            ("attempt_status_path", "attempt_status_sha256"),
            ("summary_path", "summary_sha256"),
        ):
            manifest[str(evidence[path_key])] = str(evidence[sha_key])
    approval_design["file_sha256"] = dict(sorted(manifest.items()))
    write_json(CANDIDATE, approval_design)

    digest = sha256_file(CANDIDATE)
    promotion = promote_candidate(
        candidate_path=CANDIDATE,
        expected_candidate_sha256=digest,
        project_root=ROOT,
        authorization=AUTHORIZATION,
        approved_by="user",
        promote=promote,
    )
    return {
        "candidate": str(CANDIDATE.resolve()),
        "candidate_sha256": digest,
        "resume_queue": str(RESUME_QUEUE.resolve()),
        "resume_queue_sha256": sha256_file(RESUME_QUEUE),
        "completed": len(completed),
        "remaining": len(remaining),
        "promoted": bool(promotion.get("promoted")),
        "promotion_all_passed": promotion.get("all_passed") is True,
        "transaction_id": promotion.get("transaction_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    report = prepare(promote=args.promote)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["promotion_all_passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
