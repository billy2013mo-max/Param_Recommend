#!/usr/bin/env python3
"""Evaluate prospective outcomes against predictions frozen before execution."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from prepare_h800_unified_bounded_canary_v3 import (
    CAMPAIGN_ID,
    DEFAULT_DESIGN,
    DEFAULT_PREDICTIONS,
    DEFAULT_QUEUE,
    EXPECTED_JOBS,
    PHASE_ID,
    V3_ARTIFACT,
)

SCHEMA = "sft_h800_unified_bounded_canary_results/v3"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_unified_bounded_canary_results_v3.json"
DEFAULT_MARKDOWN = ARTIFACT_DIR / "h800_unified_bounded_canary_results_v3.md"
RUNTIME_BASE_PARAMETERS = {
    "qwen3_1p7b": 1_720_574_976,
    "qwen3_4b": 4_022_468_096,
    "qwen3_8b": 8_190_735_360,
    "qwen3_14b": 14_768_307_200,
}


def _attempt_dir(job_id: str) -> Path | None:
    pointer = ROOT / "results" / job_id / "latest_attempt.json"
    if not pointer.is_file():
        return None
    row = read_json(pointer)
    path = pointer.parent / str(row.get("attempt_path") or "")
    return path if path.is_dir() else None


def _summaries(attempt: Path) -> list[dict[str, Any]]:
    return [
        read_json(path) for path in sorted(attempt.glob("metrics/summary.rank*.json"))
    ]


def _structure_audit(
    attempt: Path, *, expected_parameters: int, expected_world_size: int
) -> dict[str, Any]:
    paths = sorted(attempt.glob("metrics/model_structure_manifest*.json"))
    rows = [read_json(path) for path in paths]
    observed = [
        int(
            (
                ((row.get("components") or {}).get("language_model") or {}).get(
                    "base_parameter_elements"
                )
            )
            or 0
        )
        for row in rows
    ]
    return {
        "manifest_count": len(rows),
        "expected_manifest_count": expected_world_size,
        "expected_base_parameters": expected_parameters,
        "observed_base_parameters": observed,
        "all_passed": len(rows) == expected_world_size
        and observed == [expected_parameters] * expected_world_size,
        "paths": [str(path) for path in paths],
    }


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metrics(observations: list[dict[str, Any]]) -> dict[str, Any]:
    success = [row for row in observations if row["classification"] == "success"]
    oom = [row for row in observations if row["classification"] == "oom"]
    safe = [row for row in success if row["actually_safe_success"]]
    unsafe = [row for row in success if not row["actually_safe_success"]]
    admitted_canary = [
        row for row in observations if row["probe_role"] == "admitted_canary"
    ]
    rejected_controls = [
        row for row in observations if row["probe_role"] == "rejected_boundary_control"
    ]
    errors = [float(row["absolute_percentage_error"]) for row in success]
    signed = [float(row["signed_percentage_error"]) for row in success]
    admitted_safe = sum(bool(row["frozen_v3_admitted"]) for row in safe)
    admitted_oom = sum(bool(row["frozen_v3_admitted"]) for row in oom)
    return {
        "jobs": len(observations),
        "success": len(success),
        "oom": len(oom),
        "center_mape": statistics.fmean(errors) if errors else None,
        "center_signed_bias": statistics.fmean(signed) if signed else None,
        "center_p90_ape": _percentile(errors, 0.9),
        "actual_safe_success": len(safe),
        "admitted_safe_success": admitted_safe,
        "safe_success_admission_rate": admitted_safe / len(safe) if safe else None,
        "actual_unsafe_success": len(unsafe),
        "admitted_unsafe_success": sum(
            bool(row["frozen_v3_admitted"]) for row in unsafe
        ),
        "admitted_oom": admitted_oom,
        "oom_admission_rate": admitted_oom / len(oom) if oom else 0.0,
        "admitted_canary": {
            "jobs": len(admitted_canary),
            "success": sum(
                row["classification"] == "success" for row in admitted_canary
            ),
            "oom": sum(row["classification"] == "oom" for row in admitted_canary),
            "unsafe_success": sum(
                row["classification"] == "success" and not row["actually_safe_success"]
                for row in admitted_canary
            ),
        },
        "rejected_boundary_controls": {
            "jobs": len(rejected_controls),
            "success": sum(
                row["classification"] == "success" for row in rejected_controls
            ),
            "oom": sum(row["classification"] == "oom" for row in rejected_controls),
            "safe_success": sum(
                row["classification"] == "success" and row["actually_safe_success"]
                for row in rejected_controls
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    jobs = read_jsonl(args.queue)
    design = read_json(args.design)
    frozen = read_json(args.predictions)
    if len(jobs) != EXPECTED_JOBS or len(frozen.get("rows") or []) != EXPECTED_JOBS:
        raise ValueError("canary input size drifted")
    if (
        design["bindings"]["queue"]["sha256"] != sha256_file(args.queue)
        or design["bindings"]["frozen_predictions"]["sha256"]
        != sha256_file(args.predictions)
        or frozen["v3_artifact"]["sha256"] != sha256_file(V3_ARTIFACT)
        or frozen["ordered_job_payload_sha256"] != sha256_json(jobs)
        or frozen["outcomes_observed"] != 0
        or frozen["status"] != "frozen_before_any_canary_outcome"
    ):
        raise ValueError("frozen canary binding drifted")
    frozen_by_job = {str(row["job_id"]): row for row in frozen["rows"]}
    observations = []
    incomplete = []
    for job in jobs:
        job_id = str(job["job_id"])
        status_path = ROOT / "results" / job_id / "status.json"
        attempt = _attempt_dir(job_id)
        if not status_path.is_file() or attempt is None:
            incomplete.append(job_id)
            continue
        status = read_json(status_path)
        classification = str(status.get("classification") or "")
        if classification not in {"success", "oom"}:
            incomplete.append(job_id)
            continue
        structure = _structure_audit(
            attempt,
            expected_parameters=RUNTIME_BASE_PARAMETERS[str(job["model_id"])],
            expected_world_size=int(job["gpu_count"]),
        )
        if not structure["all_passed"]:
            raise RuntimeError(
                f"runtime model identity mismatch: {job_id}: {structure}"
            )
        summaries = _summaries(attempt)
        max_reserved = (
            max(float(row["max_reserved"]) for row in summaries) if summaries else None
        )
        max_allocated = (
            max(float(row["max_allocated"]) for row in summaries) if summaries else None
        )
        prediction = dict(frozen_by_job[job_id]["v3"])
        safe_limit = float(prediction["safe_limit_bytes"])
        observation: dict[str, Any] = {
            "job_id": job_id,
            "campaign_id": CAMPAIGN_ID,
            "phase_id": PHASE_ID,
            "probe_role": str(job["probe_role"]),
            "dataset_id": str(job["dataset_id"]),
            "model_id": str(job["model_id"]),
            "train_type": str(job["train_type"]),
            "gpu_count": int(job["gpu_count"]),
            "zero_stage": int(job["zero_stage"]),
            "gc": bool(job["gc"]),
            "mbs": int(job["mbs"]),
            "cutoff_len": int(job["cutoff_len"]),
            "classification": classification,
            "wall_seconds": float(status.get("wall_seconds") or 0.0),
            "max_reserved_bytes": max_reserved,
            "max_allocated_bytes": max_allocated,
            "runtime_model_identity": structure,
            "frozen_v3_center_bytes": float(prediction["center_bytes"]),
            "frozen_v3_upper_bytes": float(prediction["admission_upper_bytes"]),
            "frozen_v3_admitted": bool(prediction["admitted"]),
            "safe_limit_bytes": safe_limit,
        }
        if classification == "success":
            if max_reserved is None:
                raise RuntimeError(f"success has no memory summary: {job_id}")
            observation.update(
                {
                    "actually_safe_success": max_reserved <= safe_limit,
                    "absolute_percentage_error": abs(
                        float(prediction["center_bytes"]) / max_reserved - 1.0
                    ),
                    "signed_percentage_error": (
                        float(prediction["center_bytes"]) / max_reserved - 1.0
                    ),
                }
            )
        else:
            observation["actually_safe_success"] = False
        observations.append(observation)
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(f"canary campaign is incomplete: {incomplete}")

    metrics = _metrics(observations)
    checks = {
        "all_ten_jobs_terminal": not incomplete and len(observations) == EXPECTED_JOBS,
        "all_runtime_model_identities_passed": all(
            row["runtime_model_identity"]["all_passed"] for row in observations
        ),
        "admitted_canary_zero_oom": metrics["admitted_canary"]["oom"] == 0,
        "admitted_canary_zero_unsafe_success": metrics["admitted_canary"][
            "unsafe_success"
        ]
        == 0,
        "overall_zero_oom_admission": metrics["admitted_oom"] == 0,
        "safe_admission_at_least_0p80": metrics["safe_success_admission_rate"]
        is not None
        and metrics["safe_success_admission_rate"] >= 0.80,
        "center_mape_at_most_0p15": metrics["center_mape"] is not None
        and metrics["center_mape"] <= 0.15,
        "absolute_center_bias_at_most_0p10": metrics["center_signed_bias"] is not None
        and abs(metrics["center_signed_bias"]) <= 0.10,
        "center_p90_at_most_0p30": metrics["center_p90_ape"] is not None
        and metrics["center_p90_ape"] <= 0.30,
    }
    complete = not incomplete
    passed = complete and all(checks.values())
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "prospective_canary_passed"
            if passed
            else "prospective_canary_failed"
            if complete
            else "incomplete"
        ),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "production_model_mutated": False,
        "publication_allowed": False,
        "model_or_margin_refit_during_canary": False,
        "old_missing_34_jobs_executed": False,
        "queue": {"path": str(args.queue.resolve()), "sha256": sha256_file(args.queue)},
        "frozen_predictions": {
            "path": str(args.predictions.resolve()),
            "sha256": sha256_file(args.predictions),
            "frozen_before_outcomes": True,
        },
        "frozen_v3_artifact": {
            "path": str(V3_ARTIFACT.resolve()),
            "sha256": sha256_file(V3_ARTIFACT),
        },
        "terminal_counts": dict(Counter(row["classification"] for row in observations)),
        "incomplete_job_ids": incomplete,
        "metrics": metrics,
        "checks": checks,
        "all_canary_checks_passed": passed,
        "observations": observations,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(args.output, report)
    lines = [
        "# H800 统一有界显存 v3 前瞻 canary",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 状态 | {report['status']} |",
        f"| success / OOM | {metrics['success']} / {metrics['oom']} |",
        f"| 中心 MAPE | {metrics['center_mape']:.2%} |"
        if metrics["center_mape"] is not None
        else "| 中心 MAPE | n/a |",
        f"| 中心偏差 | {metrics['center_signed_bias']:+.2%} |"
        if metrics["center_signed_bias"] is not None
        else "| 中心偏差 | n/a |",
        f"| P90 APE | {metrics['center_p90_ape']:.2%} |"
        if metrics["center_p90_ape"] is not None
        else "| P90 APE | n/a |",
        f"| 安全放行 | {metrics['admitted_safe_success']}/{metrics['actual_safe_success']} = {metrics['safe_success_admission_rate']:.2%} |"
        if metrics["safe_success_admission_rate"] is not None
        else "| 安全放行 | n/a |",
        f"| OOM 放行 | {metrics['admitted_oom']}/{metrics['oom']} |",
        "",
    ]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
