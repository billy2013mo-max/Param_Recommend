#!/usr/bin/env python3
"""Evaluate frozen-v3 center predictions on prospective real-business runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from evaluate_h800_unified_bounded_canary_v3 import _attempt_dir, _structure_audit, _summaries
from prepare_h800_unified_bounded_business_generalization_v3 import (
    ACCEPTANCE_THRESHOLDS,
    CAMPAIGN_ID,
    DATASETS,
    DEFAULT_DESIGN,
    DEFAULT_PREDICTIONS,
    DEFAULT_QUEUE,
    EXPECTED_JOBS,
    PHASE_ID,
    V3_ARTIFACT,
)

SCHEMA = "sft_h800_unified_bounded_business_generalization_results/v3"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_unified_bounded_business_generalization_results_v3.json"
DEFAULT_MARKDOWN = ARTIFACT_DIR / "h800_unified_bounded_business_generalization_results_v3.md"
DEFAULT_PREDICTION_ROWS = ARTIFACT_DIR / "h800_unified_bounded_business_generalization_predictions_v3.jsonl"
RUNTIME_BASE_PARAMETERS = {
    "qwen3_1p7b": 1_720_574_976,
    "qwen3_4b": 4_022_468_096,
    "qwen3_8b": 8_190_735_360,
    "qwen3_14b": 14_768_307_200,
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


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = [row for row in rows if row["classification"] == "success"]
    oom = [row for row in rows if row["classification"] == "oom"]
    errors = [float(row["absolute_percentage_error"]) for row in success]
    signed = [float(row["signed_percentage_error"]) for row in success]
    return {
        "jobs": len(rows),
        "success": len(success),
        "oom": len(oom),
        "center_mape": statistics.fmean(errors) if errors else None,
        "center_signed_bias": statistics.fmean(signed) if signed else None,
        "center_p90_ape": _percentile(errors, 0.9),
    }


def _grouped(observations: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        groups[str(row[key])].append(row)
    return {name: _group_metrics(rows) for name, rows in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--prediction-rows", type=Path, default=DEFAULT_PREDICTION_ROWS)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    jobs = read_jsonl(args.queue)
    design = read_json(args.design)
    frozen = read_json(args.predictions)
    if len(jobs) != EXPECTED_JOBS or len(frozen.get("rows") or []) != EXPECTED_JOBS:
        raise ValueError("business generalization input size drifted")
    if (
        design["bindings"]["queue"]["sha256"] != sha256_file(args.queue)
        or design["bindings"]["frozen_predictions"]["sha256"] != sha256_file(args.predictions)
        or frozen["v3_artifact"]["sha256"] != sha256_file(V3_ARTIFACT)
        or frozen["ordered_job_payload_sha256"] != sha256_json(jobs)
        or frozen["outcomes_observed"] != 0
        or frozen["status"] != "frozen_before_any_business_outcome"
        or design.get("acceptance_thresholds") != ACCEPTANCE_THRESHOLDS
    ):
        raise ValueError("frozen business-validation binding drifted")

    frozen_by_job = {str(row["job_id"]): row for row in frozen["rows"]}
    observations: list[dict[str, Any]] = []
    incomplete: list[str] = []
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
            raise RuntimeError(f"runtime model identity mismatch: {job_id}: {structure}")
        summaries = _summaries(attempt)
        max_reserved = max((float(row["max_reserved"]) for row in summaries), default=None)
        max_allocated = max((float(row["max_allocated"]) for row in summaries), default=None)
        prediction = dict(frozen_by_job[job_id]["v3"])
        observation: dict[str, Any] = {
            "job_id": job_id,
            "dataset_id": str(job["dataset_id"]),
            "dataset_category": str(job["dataset_category"]),
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
            "safe_limit_bytes": float(prediction["safe_limit_bytes"]),
        }
        if classification == "success":
            if max_reserved is None:
                raise RuntimeError(f"success has no memory summary: {job_id}")
            signed = float(prediction["center_bytes"]) / max_reserved - 1.0
            observation.update(
                {
                    "actually_safe_success": max_reserved <= float(prediction["safe_limit_bytes"]),
                    "absolute_percentage_error": abs(signed),
                    "signed_percentage_error": signed,
                }
            )
        else:
            observation["actually_safe_success"] = False
        observations.append(observation)

    if incomplete and not args.allow_incomplete:
        raise RuntimeError(f"business generalization campaign is incomplete: {incomplete}")

    overall = _group_metrics(observations)
    by_dataset = _grouped(observations, "dataset_id")
    by_model = _grouped(observations, "model_id")
    by_train_type = _grouped(observations, "train_type")
    dataset_mapes = [
        float(metrics["center_mape"])
        for metrics in by_dataset.values()
        if metrics["center_mape"] is not None
    ]
    source_equal_mape = statistics.fmean(dataset_mapes) if dataset_mapes else None
    thresholds = ACCEPTANCE_THRESHOLDS
    checks = {
        "all_jobs_terminal": not incomplete and len(observations) == EXPECTED_JOBS,
        "all_runtime_model_identities_passed": all(row["runtime_model_identity"]["all_passed"] for row in observations),
        "minimum_success_jobs": overall["success"] >= int(thresholds["minimum_success_jobs"]),
        "minimum_success_per_dataset": all(
            by_dataset.get(dataset_id, {}).get("success", 0) >= int(thresholds["minimum_success_per_dataset"])
            for dataset_id in DATASETS
        ),
        "zero_oom": overall["oom"] <= int(thresholds["maximum_oom_jobs"]),
        "source_equal_mape": source_equal_mape is not None and source_equal_mape <= float(thresholds["center_source_equal_mape_max"]),
        "absolute_center_bias": overall["center_signed_bias"] is not None and abs(float(overall["center_signed_bias"])) <= float(thresholds["absolute_center_signed_bias_max"]),
        "center_p90": overall["center_p90_ape"] is not None and float(overall["center_p90_ape"]) <= float(thresholds["center_p90_ape_max"]),
    }
    complete = not incomplete
    passed = complete and all(checks.values())
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "business_generalization_passed" if passed else "business_generalization_failed" if complete else "incomplete",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "production_model_mutated": False,
        "publication_allowed": False,
        "model_or_margin_refit_during_validation": False,
        "old_missing_34_jobs_executed": False,
        "evidence_contract": design["evidence_contract"],
        "acceptance_thresholds": thresholds,
        "queue": {"path": str(args.queue.resolve()), "sha256": sha256_file(args.queue)},
        "frozen_predictions": {"path": str(args.predictions.resolve()), "sha256": sha256_file(args.predictions), "frozen_before_outcomes": True},
        "frozen_v3_artifact": {"path": str(V3_ARTIFACT.resolve()), "sha256": sha256_file(V3_ARTIFACT)},
        "terminal_counts": dict(Counter(row["classification"] for row in observations)),
        "incomplete_job_ids": incomplete,
        "metrics": {
            "overall": overall,
            "center_source_equal_mape": source_equal_mape,
            "by_dataset": by_dataset,
            "by_model": by_model,
            "by_train_type": by_train_type,
        },
        "checks": checks,
        "all_business_generalization_checks_passed": passed,
        "observations": observations,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(args.output, report)
    args.prediction_rows.parent.mkdir(parents=True, exist_ok=True)
    args.prediction_rows.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in observations),
        encoding="utf-8",
    )

    def fmt(value: Any, signed: bool = False) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):+.2%}" if signed else f"{float(value):.2%}"

    lines = [
        "# H800 统一有界显存 V3：真实业务数据集泛化验收",
        "",
        f"状态：`{report['status']}`。预测在任何本轮运行结果产生前冻结，验证期间没有重拟合。",
        "",
        "| 范围 | 成功 / OOM | 中心 MAPE | 中心偏差 | P90 APE |",
        "|---|---:|---:|---:|---:|",
        f"| 整体（逐配置） | {overall['success']} / {overall['oom']} | {fmt(overall['center_mape'])} | {fmt(overall['center_signed_bias'], True)} | {fmt(overall['center_p90_ape'])} |",
        f"| 四数据集等权 | {overall['success']} / {overall['oom']} | {fmt(source_equal_mape)} | — | — |",
    ]
    for dataset_id in DATASETS:
        metrics = by_dataset.get(dataset_id, {})
        lines.append(
            f"| `{dataset_id}` | {metrics.get('success', 0)} / {metrics.get('oom', 0)} | {fmt(metrics.get('center_mape'))} | {fmt(metrics.get('center_signed_bias'), True)} | {fmt(metrics.get('center_p90_ape'))} |"
        )
    lines.extend(["", "OOM 只作为右删失证据，不进入中心 MAPE。", ""])
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
