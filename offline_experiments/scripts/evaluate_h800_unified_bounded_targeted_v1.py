#!/usr/bin/env python3
"""Evaluate the 15 targeted probes and verify runtime model identity."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fit_h800_unified_resource_partial_v1 as base
from common import (
    ARTIFACT_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from h800_unified_bounded_memory_model import load_artifact, predict_records
from prepare_h800_unified_bounded_targeted_v1 import (
    CAMPAIGN_ID,
    DEFAULT_QUEUE,
    EXPECTED_JOBS,
    MODEL_INVENTORY,
    PHASE_ID,
)

SCHEMA = "sft_h800_unified_bounded_targeted_results/v1"
DEFAULT_ARTIFACT = ARTIFACT_DIR / "h800_unified_bounded_memory_candidate_v2.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_unified_bounded_targeted_results_v1.json"
DEFAULT_MARKDOWN = ARTIFACT_DIR / "h800_unified_bounded_targeted_results_v1.md"

# The frozen inventory's qwen3_1p7b ``actual_parameters`` uses the untied-LM-head
# theory convention (2,031,739,904).  The checkpoint has tied embeddings and its
# runtime tensor inventory is consistently 1,720,574,976 across prior and current
# attempts.  Model identity is checked against the runtime tensor convention.
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
    observed = []
    for row in rows:
        components = row.get("components") or {}
        language = components.get("language_model") or {}
        observed.append(int(language.get("base_parameter_elements") or 0))
    return {
        "manifest_count": len(rows),
        "expected_manifest_count": expected_world_size,
        "expected_base_parameters": expected_parameters,
        "observed_base_parameters": observed,
        "all_passed": len(rows) == expected_world_size
        and observed == [expected_parameters] * expected_world_size,
        "paths": [str(path) for path in paths],
    }


def _metrics(observations: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in observations if row["classification"] == "success"]
    oom = [row for row in observations if row["classification"] == "oom"]
    safe = [row for row in successes if row["actually_safe_success"]]
    errors = [float(row["absolute_percentage_error"]) for row in successes]
    signed = [float(row["signed_percentage_error"]) for row in successes]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_family[str(row["probe_family"])].append(row)

    def family_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        exact = [row for row in rows if row["classification"] == "success"]
        censored = [row for row in rows if row["classification"] == "oom"]
        family_safe = [row for row in exact if row["actually_safe_success"]]
        return {
            "jobs": len(rows),
            "success": len(exact),
            "oom": len(censored),
            "center_mape": (
                statistics.fmean(row["absolute_percentage_error"] for row in exact)
                if exact
                else None
            ),
            "center_signed_bias": (
                statistics.fmean(row["signed_percentage_error"] for row in exact)
                if exact
                else None
            ),
            "safe_success": len(family_safe),
            "safe_admitted": sum(bool(row["admitted"]) for row in family_safe),
            "oom_admitted": sum(bool(row["admitted"]) for row in censored),
        }

    return {
        "jobs": len(observations),
        "success": len(successes),
        "oom": len(oom),
        "center_mape": statistics.fmean(errors) if errors else None,
        "center_signed_bias": statistics.fmean(signed) if signed else None,
        "safe_success": len(safe),
        "safe_admitted": sum(bool(row["admitted"]) for row in safe),
        "oom_admitted": sum(bool(row["admitted"]) for row in oom),
        "by_probe_family": {
            name: family_metrics(rows) for name, rows in sorted(by_family.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    jobs = read_jsonl(args.queue)
    if len(jobs) != EXPECTED_JOBS:
        raise ValueError("targeted queue size drifted")
    inventory = read_json(MODEL_INVENTORY)
    model_by_id = {str(row["id"]): row for row in inventory["models"]}
    hardware = read_json(base.DEFAULT_HARDWARE)
    capacity = int(hardware["memory_bytes_reported_by_torch"])
    artifact = load_artifact(args.artifact)
    profile_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
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
        reference, features = base._current_features(
            job,
            model_by_id=model_by_id,
            fixed_lora=inventory["fixed_lora"],
            capacity_bytes=capacity,
            profile_cache=profile_cache,
        )
        record = {
            "record_id": f"targeted::{job_id}",
            "reference_bytes": reference,
            "features": features,
        }
        prediction = predict_records([record], artifact)[0]
        safe_limit = float(prediction["safe_limit_bytes"])
        observation: dict[str, Any] = {
            "job_id": job_id,
            "campaign_id": CAMPAIGN_ID,
            "phase_id": PHASE_ID,
            "probe_family": str(job["probe_family"]),
            "dataset_id": str(job["dataset_id"]),
            "model_id": str(job["model_id"]),
            "train_type": str(job["train_type"]),
            "gpu_count": int(job["gpu_count"]),
            "zero_stage": int(job["zero_stage"]),
            "gc": bool(job["gc"]),
            "mbs": int(job["mbs"]),
            "cutoff_len": int(job["cutoff_len"]),
            "classification": classification,
            "calibration_eligible": bool(status.get("calibration_eligible")),
            "wall_seconds": float(status.get("wall_seconds") or 0.0),
            "reference_bytes": reference,
            "max_reserved_bytes": max_reserved,
            "max_allocated_bytes": max_allocated,
            "censor_lower_bytes": float(capacity) if classification == "oom" else None,
            "runtime_model_identity": structure,
            **dict(prediction),
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
        observations.append(observation)
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(f"targeted campaign is incomplete: {incomplete}")

    metrics = _metrics(observations)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if not incomplete else "incomplete",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "production_model_mutated": False,
        "old_missing_34_jobs_executed": False,
        "queue": {
            "path": str(args.queue.resolve()),
            "sha256": sha256_file(args.queue),
        },
        "frozen_v2_artifact": {
            "path": str(args.artifact.resolve()),
            "sha256": sha256_file(args.artifact),
        },
        "terminal_counts": dict(Counter(row["classification"] for row in observations)),
        "incomplete_job_ids": incomplete,
        "runtime_model_identity_all_passed": all(
            row["runtime_model_identity"]["all_passed"] for row in observations
        ),
        "runtime_parameter_convention": {
            "qwen3_1p7b_inventory_declared_parameters": int(
                model_by_id["qwen3_1p7b"]["actual_parameters"]
            ),
            "qwen3_1p7b_runtime_tied_embedding_base_parameters": RUNTIME_BASE_PARAMETERS[
                "qwen3_1p7b"
            ],
            "interpretation": "known parameter-count convention difference; model path and repeated runtime tensor inventory agree",
        },
        "metrics": metrics,
        "observations": observations,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(args.output, report)
    lines = [
        "# H800 统一有界模型定向实验",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 任务 | {metrics['jobs']} |",
        f"| success / OOM | {metrics['success']} / {metrics['oom']} |",
        f"| 冻结 v2 中心 MAPE | {metrics['center_mape']:.2%} |"
        if metrics["center_mape"] is not None
        else "| 冻结 v2 中心 MAPE | n/a |",
        f"| 冻结 v2 中心偏差 | {metrics['center_signed_bias']:+.2%} |"
        if metrics["center_signed_bias"] is not None
        else "| 冻结 v2 中心偏差 | n/a |",
        f"| 安全放行 | {metrics['safe_admitted']}/{metrics['safe_success']} |",
        f"| OOM 放行 | {metrics['oom_admitted']}/{metrics['oom']} |",
        "",
    ]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
