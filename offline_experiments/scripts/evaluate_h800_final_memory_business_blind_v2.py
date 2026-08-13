#!/usr/bin/env python3
"""Evaluate the three primary metrics of the frozen V3 business blind."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, ROOT, percentile, read_json, read_jsonl, sha256_file, sha256_json, write_json
from evaluate_h800_final_memory_peak_replay_stage1_v1 import _observed_max_padded_sequence
from evaluate_h800_unified_bounded_canary_v3 import _attempt_dir, _structure_audit, _summaries
from prepare_h800_final_memory_business_blind_v2 import implementation as prep

SCHEMA = "sft_h800_final_memory_business_blind_results/v2"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_final_memory_business_blind_results_v2.json"
DEFAULT_MARKDOWN = ARTIFACT_DIR / "h800_final_memory_business_blind_results_v2.md"


def _terminal(job: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any] | None:
    job_id = str(job["job_id"])
    status_path = ROOT / "results" / job_id / "status.json"
    attempt = _attempt_dir(job_id)
    if not status_path.is_file() or attempt is None:
        return None
    status = read_json(status_path)
    classification = str(status.get("classification") or "")
    if classification not in {"success", "oom"}:
        return None
    structure = _structure_audit(
        attempt,
        expected_parameters=int(job["model_parameters"]),
        expected_world_size=int(job["gpu_count"]),
    )
    if not structure["all_passed"]:
        raise RuntimeError(f"runtime model identity mismatch: {job_id}: {structure}")
    summaries = _summaries(attempt)
    max_reserved = max((float(row.get("max_reserved") or 0) for row in summaries), default=0.0) or None
    max_allocated = max((float(row.get("max_allocated") or 0) for row in summaries), default=0.0) or None
    completed_steps = min((int(row.get("total_steps") or 0) for row in summaries), default=0)
    observed_padded = _observed_max_padded_sequence(summaries)
    expected_padded = None
    contract = job.get("measurement_contract") or {}
    if job.get("measurement_mode") == "validated_allocator_prefix_replay":
        expected_padded = max(
            int(row["rank_max_padded_sequence_length"])
            for row in contract["rank_contracts"]
        )
    if classification == "success" and max_reserved is None:
        raise RuntimeError(f"successful final job has no exact reserved peak: {job_id}")
    observed = max_reserved if classification == "success" else None
    safe_limit = float(prediction["safe_limit_bytes"])
    center = float(prediction["center_bytes"])
    signed_error = (center - observed) / observed if observed else None
    actual_class = (
        "oom" if classification == "oom"
        else "safe_success" if observed is not None and observed <= safe_limit
        else "unsafe_success"
    )
    pressure = float(prediction["admission_upper_bytes"]) / safe_limit
    return {
        "job_id": job_id,
        "scenario": str(job["scenario"]),
        "source_dataset_id": str(job["source_dataset_id"]),
        "model_id": str(job["model_id"]),
        "train_type": str(job["train_type"]),
        "gpu_count": int(job["gpu_count"]),
        "zero_stage": int(job["zero_stage"]),
        "gc": bool(job["gc"]),
        "packing": bool(job["packing"]),
        "mbs": int(job["mbs"]),
        "cutoff_len": int(job["cutoff_len"]),
        "target_pressure": float(job["target_pressure"]),
        "frozen_pressure": pressure,
        "classification": classification,
        "actual_class": actual_class,
        "wall_seconds": float(status.get("wall_seconds") or 0.0),
        "max_reserved_bytes": max_reserved,
        "max_allocated_bytes": max_allocated,
        "center_bytes": center,
        "risk_head_bytes": float(prediction["risk_head_bytes"]),
        "admission_upper_bytes": float(prediction["admission_upper_bytes"]),
        "safe_limit_bytes": safe_limit,
        "admitted": bool(prediction["admitted"]),
        "signed_percentage_error": signed_error,
        "absolute_percentage_error": abs(signed_error) if signed_error is not None else None,
        "boundary_safe": actual_class == "safe_success" and observed is not None and observed >= 0.90 * safe_limit,
        "difficult_oom": actual_class == "oom" and pressure <= 1.05,
        "expected_optimizer_steps": int(job["measure_steps"]),
        "completed_optimizer_steps": completed_steps,
        "expected_max_padded_sequence_length": expected_padded,
        "observed_max_padded_sequence_length": observed_padded,
        "measurement_shape_observed": (
            True if classification == "oom" or expected_padded is None
            else observed_padded is not None and observed_padded >= expected_padded
        ),
        "runtime_model_identity": structure,
    }


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=prep.DEFAULT_QUEUE)
    parser.add_argument("--design", type=Path, default=prep.DEFAULT_DESIGN)
    parser.add_argument("--predictions", type=Path, default=prep.DEFAULT_PREDICTIONS)
    parser.add_argument("--data-bundle", type=Path, default=prep.DEFAULT_DATA_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    jobs = read_jsonl(args.queue)
    design = read_json(args.design)
    predictions_artifact = read_json(args.predictions)
    data = read_json(args.data_bundle)
    bindings = design.get("bindings") or {}
    if (
        len(jobs) != prep.EXPECTED_JOBS
        or len(predictions_artifact.get("rows") or []) != prep.EXPECTED_JOBS
        or design.get("ordered_job_ids") != [str(row["job_id"]) for row in jobs]
        or design.get("ordered_job_payload_sha256") != sha256_json(jobs)
        or predictions_artifact.get("ordered_job_payload_sha256") != sha256_json(jobs)
        or predictions_artifact.get("outcomes_observed") != 0
        or predictions_artifact.get("status") != "frozen_before_any_final_gpu_outcome"
        or data.get("final_acceptance_gpu_outcomes_read") != 0
        or bindings.get("queue", {}).get("sha256") != sha256_file(args.queue)
        or bindings.get("frozen_predictions", {}).get("sha256") != sha256_file(args.predictions)
        or bindings.get("data_bundle", {}).get("sha256") != sha256_file(args.data_bundle)
    ):
        raise ValueError("final business-blind frozen binding drifted")
    predictions = {str(row["job_id"]): row for row in predictions_artifact["rows"]}
    observations = []
    incomplete = []
    for job in jobs:
        row = _terminal(job, predictions[str(job["job_id"])])
        if row is None:
            incomplete.append(str(job["job_id"]))
        else:
            observations.append(row)
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(f"final business blind is incomplete: {len(incomplete)} jobs")

    successes = [row for row in observations if row["classification"] == "success"]
    safe = [row for row in successes if row["actual_class"] == "safe_success"]
    unsafe = [row for row in successes if row["actual_class"] == "unsafe_success"]
    oom = [row for row in observations if row["actual_class"] == "oom"]
    boundary = [row for row in safe if row["boundary_safe"]]
    difficult = [row for row in oom if row["difficult_oom"]]

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successes:
        by_dataset[str(row["source_dataset_id"])].append(row)
    dataset_metrics = {}
    for source_id, rows in sorted(by_dataset.items()):
        dataset_metrics[source_id] = {
            "successes": len(rows),
            "mape": statistics.fmean(float(row["absolute_percentage_error"]) for row in rows),
            "bias": statistics.fmean(float(row["signed_percentage_error"]) for row in rows),
        }
    dataset_equal_mape = statistics.fmean(row["mape"] for row in dataset_metrics.values()) if dataset_metrics else None
    dataset_equal_bias = statistics.fmean(row["bias"] for row in dataset_metrics.values()) if dataset_metrics else None
    row_mape = statistics.fmean(float(row["absolute_percentage_error"]) for row in successes) if successes else None
    p90_ape = percentile([float(row["absolute_percentage_error"]) for row in successes], 90) if successes else None
    train_type_metrics = {}
    for train_type in ("full", "lora"):
        rows = [row for row in successes if row["train_type"] == train_type]
        train_type_metrics[train_type] = {
            "successes": len(rows),
            "mape": statistics.fmean(float(row["absolute_percentage_error"]) for row in rows) if rows else None,
        }

    safe_admission = _rate(sum(bool(row["admitted"]) for row in safe), len(safe))
    boundary_admission = _rate(sum(bool(row["admitted"]) for row in boundary), len(boundary))
    unsafe_admitted = _rate(sum(bool(row["admitted"]) for row in unsafe), len(unsafe))
    oom_admission = _rate(sum(bool(row["admitted"]) for row in oom), len(oom))
    difficult_oom_admission = _rate(sum(bool(row["admitted"]) for row in difficult), len(difficult))
    thresholds = design["acceptance_thresholds"]
    hard_failure = unsafe_admitted["numerator"] > 0 or oom_admission["numerator"] > 0
    denominator_checks = {
        "minimum_exact_successes": len(successes) >= int(thresholds["minimum_exact_successes"]),
        "minimum_safe_successes": len(safe) >= int(thresholds["minimum_safe_successes"]),
        "minimum_ooms": len(oom) >= int(thresholds["minimum_ooms"]),
        "minimum_difficult_ooms": len(difficult) >= int(thresholds["minimum_difficult_ooms"]),
        "full_minimum_successes": train_type_metrics["full"]["successes"] >= 12,
        "lora_minimum_successes": train_type_metrics["lora"]["successes"] >= 12,
        "each_scored_dataset_minimum_successes": len(dataset_metrics) == 6 and all(row["successes"] >= 4 for row in dataset_metrics.values()),
        "minimum_boundary_safe": len(boundary) >= 20,
    }
    metric_checks = {
        "dataset_equal_center_mape": dataset_equal_mape is not None and dataset_equal_mape <= float(thresholds["dataset_equal_center_mape_max"]),
        "absolute_center_bias": dataset_equal_bias is not None and abs(dataset_equal_bias) <= float(thresholds["absolute_center_bias_max"]),
        "p90_ape": p90_ape is not None and p90_ape <= float(thresholds["p90_ape_max"]),
        "train_type_mape": all(row["mape"] is not None and row["mape"] <= float(thresholds["train_type_mape_max"]) for row in train_type_metrics.values()),
        "per_dataset_mape": len(dataset_metrics) == 6 and all(row["mape"] <= float(thresholds["per_dataset_mape_max"]) for row in dataset_metrics.values()),
        "safe_admission_rate": safe_admission["rate"] is not None and safe_admission["rate"] >= float(thresholds["safe_admission_rate_min"]),
        "boundary_safe_admission_rate": boundary_admission["rate"] is not None and boundary_admission["rate"] >= float(thresholds["boundary_safe_admission_rate_min"]),
        "unsafe_success_admitted_zero": unsafe_admitted["numerator"] == 0,
        "oom_admitted_zero": oom_admission["numerator"] == 0,
        "difficult_oom_admitted_zero": difficult_oom_admission["numerator"] == 0,
    }
    execution_checks = {
        "all_jobs_terminal": not incomplete and len(observations) == prep.EXPECTED_JOBS,
        "all_runtime_model_identities_passed": all(row["runtime_model_identity"]["all_passed"] for row in observations),
        "all_success_measurement_shapes_observed": all(row["measurement_shape_observed"] for row in successes),
        "all_success_plans_completed": all(row["completed_optimizer_steps"] == row["expected_optimizer_steps"] for row in successes),
    }
    all_denominators = all(denominator_checks.values())
    all_metrics = all(metric_checks.values())
    all_execution = all(execution_checks.values())
    if hard_failure and all_execution:
        status = "rejected_model_failure"
    elif all_execution and all_denominators and all_metrics:
        status = "accepted_for_production_shadow"
    elif all_execution:
        status = "inconclusive_measurement_or_denominator"
    else:
        status = "incomplete"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "campaign_id": prep.CAMPAIGN_ID,
        "phase_id": prep.PHASE_ID,
        "production_model_mutated": False,
        "frozen_model_refit_after_outcomes": False,
        "jobs_expected": prep.EXPECTED_JOBS,
        "jobs_observed": len(observations),
        "incomplete_job_ids": incomplete,
        "terminal_counts": dict(Counter(row["classification"] for row in observations)),
        "three_primary_metrics": {
            "center": {
                "exact_successes": len(successes),
                "dataset_equal_mape": dataset_equal_mape,
                "dataset_equal_bias": dataset_equal_bias,
                "row_mape": row_mape,
                "p90_ape": p90_ape,
                "by_train_type": train_type_metrics,
                "by_dataset": dataset_metrics,
            },
            "safe_admission": safe_admission,
            "oom_admission": oom_admission,
        },
        "secondary_metrics": {
            "safe_successes": len(safe),
            "unsafe_successes": len(unsafe),
            "boundary_safe": len(boundary),
            "boundary_safe_admission": boundary_admission,
            "unsafe_success_admission": unsafe_admitted,
            "difficult_ooms": len(difficult),
            "difficult_oom_admission": difficult_oom_admission,
        },
        "acceptance_thresholds": thresholds,
        "execution_checks": execution_checks,
        "denominator_checks": denominator_checks,
        "metric_checks": metric_checks,
        "hard_failure_observed": hard_failure,
        "expansion_required": status == "inconclusive_measurement_or_denominator",
        "cases": {
            "center_ape_over_20pct": [row for row in observations if row["absolute_percentage_error"] is not None and row["absolute_percentage_error"] > 0.20],
            "safe_false_rejections": [row for row in safe if not row["admitted"]],
            "unsafe_successes": unsafe,
            "ooms": oom,
            "oom_false_admissions": [row for row in oom if row["admitted"]],
        },
        "observations": observations,
        "bindings": {
            "queue": {"path": str(args.queue.resolve()), "sha256": sha256_file(args.queue)},
            "design": {"path": str(args.design.resolve()), "sha256": sha256_file(args.design)},
            "frozen_predictions": {"path": str(args.predictions.resolve()), "sha256": sha256_file(args.predictions)},
            "data_bundle": {"path": str(args.data_bundle.resolve()), "sha256": sha256_file(args.data_bundle)},
        },
    }
    report["report_sha256"] = sha256_json(report)
    write_json(args.output, report)

    def pct(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.2%}"

    lines = [
        "# H800 V3 最终业务盲测三指标",
        "",
        f"状态：`{status}`；完成 {len(observations)}/{prep.EXPECTED_JOBS}。",
        "",
        "| 主指标 | 结果 | 分母 |",
        "|---|---:|---:|",
        f"| 六数据集等权中心 MAPE | {pct(dataset_equal_mape)} | {len(successes)} 个精确成功 |",
        f"| 安全配置放行率 | {pct(safe_admission['rate'])} | {safe_admission['denominator']} |",
        f"| OOM 配置放行率 | {pct(oom_admission['rate'])} | {oom_admission['denominator']} |",
        "",
        f"不安全成功错误放行：{unsafe_admitted['numerator']}/{unsafe_admitted['denominator']}；困难 OOM：{difficult_oom_admission['numerator']}/{difficult_oom_admission['denominator']}。",
        "",
    ]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
