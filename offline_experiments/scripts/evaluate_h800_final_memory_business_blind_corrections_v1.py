#!/usr/bin/env python3
"""Evaluate the corrected supplement alone and together with valid original rows."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, ROOT, percentile, read_json, read_jsonl, sha256_file, sha256_json, write_json
from evaluate_h800_final_memory_business_blind_v2 import _terminal
import prepare_h800_final_memory_business_blind_corrections_v1 as correction
from prepare_h800_final_memory_business_blind_v2 import implementation as original

DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_final_memory_business_blind_correction_results_v1.json"
DEFAULT_MARKDOWN = ARTIFACT_DIR / "h800_final_memory_business_blind_correction_results_v1.md"


def _observations(queue: Path, predictions_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    jobs = read_jsonl(queue)
    predictions = {str(row["job_id"]): row for row in read_json(predictions_path)["rows"]}
    observed: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for job in jobs:
        row = _terminal(job, predictions[str(job["job_id"])])
        if row is None:
            incomplete.append(str(job["job_id"]))
        else:
            observed.append(row)
    return observed, incomplete


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator, "rate": numerator / denominator if denominator else None}


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["classification"] == "success"]
    safe = [row for row in successes if row["actual_class"] == "safe_success"]
    unsafe = [row for row in successes if row["actual_class"] == "unsafe_success"]
    oom = [row for row in rows if row["actual_class"] == "oom"]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successes:
        by_dataset[str(row["source_dataset_id"])].append(row)
    dataset_metrics = {
        source_id: {
            "successes": len(group),
            "mape": statistics.fmean(float(row["absolute_percentage_error"]) for row in group),
            "bias": statistics.fmean(float(row["signed_percentage_error"]) for row in group),
        }
        for source_id, group in sorted(by_dataset.items())
    }
    train_type = {}
    for name in ("full", "lora"):
        group = [row for row in successes if row["train_type"] == name]
        train_type[name] = {
            "successes": len(group),
            "mape": statistics.fmean(float(row["absolute_percentage_error"]) for row in group) if group else None,
            "bias": statistics.fmean(float(row["signed_percentage_error"]) for row in group) if group else None,
        }
    safe_admission = _rate(sum(bool(row["admitted"]) for row in safe), len(safe))
    oom_admission = _rate(sum(bool(row["admitted"]) for row in oom), len(oom))
    unsafe_admission = _rate(sum(bool(row["admitted"]) for row in unsafe), len(unsafe))
    center = {
        "exact_successes": len(successes),
        "dataset_equal_mape": statistics.fmean(value["mape"] for value in dataset_metrics.values()) if dataset_metrics else None,
        "dataset_equal_bias": statistics.fmean(value["bias"] for value in dataset_metrics.values()) if dataset_metrics else None,
        "row_mape": statistics.fmean(float(row["absolute_percentage_error"]) for row in successes) if successes else None,
        "p90_ape": percentile([float(row["absolute_percentage_error"]) for row in successes], 90) if successes else None,
        "by_dataset": dataset_metrics,
        "by_train_type": train_type,
    }
    return {
        "terminal_counts": dict(Counter(row["classification"] for row in rows)),
        "three_primary_metrics": {"center": center, "safe_admission": safe_admission, "oom_admission": oom_admission},
        "unsafe_successes": len(unsafe),
        "unsafe_success_admission": unsafe_admission,
        "hard_failure": oom_admission["numerator"] > 0 or unsafe_admission["numerator"] > 0,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    jobs = read_jsonl(correction.DEFAULT_QUEUE)
    design = read_json(correction.DEFAULT_DESIGN)
    predictions = read_json(correction.DEFAULT_PREDICTIONS)
    data = read_json(correction.DEFAULT_DATA_BUNDLE)
    bindings = design.get("bindings") or {}
    if (
        len(jobs) != correction.EXPECTED_JOBS
        or len(predictions.get("rows") or []) != correction.EXPECTED_JOBS
        or design.get("ordered_job_ids") != [str(row["job_id"]) for row in jobs]
        or design.get("ordered_job_payload_sha256") != sha256_json(jobs)
        or predictions.get("ordered_job_payload_sha256") != sha256_json(jobs)
        or predictions.get("outcomes_observed") != 0
        or predictions.get("status") != "frozen_before_any_correction_gpu_outcome"
        or data.get("gpu_outcomes_observed") != 0
        or bindings.get("queue", {}).get("sha256") != sha256_file(correction.DEFAULT_QUEUE)
        or bindings.get("frozen_predictions", {}).get("sha256") != sha256_file(correction.DEFAULT_PREDICTIONS)
        or bindings.get("data_bundle", {}).get("sha256") != sha256_file(correction.DEFAULT_DATA_BUNDLE)
    ):
        raise ValueError("corrected supplement frozen binding drifted")

    correction_rows, correction_incomplete = _observations(correction.DEFAULT_QUEUE, correction.DEFAULT_PREDICTIONS)
    original_rows, original_incomplete = _observations(original.DEFAULT_QUEUE, original.DEFAULT_PREDICTIONS)
    if correction_incomplete and not args.allow_incomplete:
        raise RuntimeError(f"corrected supplement is incomplete: {len(correction_incomplete)} jobs")
    supplement_metrics = _metrics(correction_rows)
    combined_metrics = _metrics([*original_rows, *correction_rows])
    primary = combined_metrics["three_primary_metrics"]
    center = primary["center"]
    metric_checks = {
        "dataset_equal_center_mape_le_15pct": center["dataset_equal_mape"] is not None and center["dataset_equal_mape"] <= 0.15,
        "absolute_center_bias_le_5pct": center["dataset_equal_bias"] is not None and abs(center["dataset_equal_bias"]) <= 0.05,
        "p90_ape_le_30pct": center["p90_ape"] is not None and center["p90_ape"] <= 0.30,
        "safe_admission_ge_95pct": primary["safe_admission"]["rate"] is not None and primary["safe_admission"]["rate"] >= 0.95,
        "zero_admitted_oom": primary["oom_admission"]["numerator"] == 0,
        "zero_admitted_unsafe_success": combined_metrics["unsafe_success_admission"]["numerator"] == 0,
    }
    denominator_checks = {
        "all_15_corrections_terminal": len(correction_rows) == correction.EXPECTED_JOBS and not correction_incomplete,
        "combined_exact_successes_ge_40": int(center["exact_successes"]) >= 40,
        "combined_ooms_ge_60": primary["oom_admission"]["denominator"] >= 60,
        "all_six_business_sources_scored": len(center["by_dataset"]) == 6,
    }
    hard_failure = bool(combined_metrics["hard_failure"])
    if hard_failure:
        status = "rejected_model_failure"
    elif all(metric_checks.values()) and all(denominator_checks.values()):
        status = "accepted_for_production_shadow"
    elif not correction_incomplete:
        status = "inconclusive_requires_targeted_oom_denominator_or_metric_repair"
    else:
        status = "incomplete"
    report = {
        "schema": "sft_h800_final_memory_business_blind_correction_results/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "production_model_mutated": False,
        "v3_refit_after_any_reported_outcome": False,
        "interpretation_policy": {
            "original_rows": "pre-family-blind evidence; infrastructure failures and unlaunched invalid rows are excluded",
            "correction_rows": "prospective executor-compatible supplement, frozen before its own outcomes; not relabeled as original blind",
        },
        "correction": {**supplement_metrics, "jobs_expected": correction.EXPECTED_JOBS, "jobs_observed": len(correction_rows), "incomplete_job_ids": correction_incomplete},
        "combined_valid_evidence": {**combined_metrics, "jobs_observed": len(original_rows) + len(correction_rows)},
        "original_incomplete_or_infrastructure_job_ids": original_incomplete,
        "metric_checks": metric_checks,
        "denominator_checks": denominator_checks,
        "hard_failure": hard_failure,
        "bindings": {
            "correction_queue": {"path": str(correction.DEFAULT_QUEUE.resolve()), "sha256": sha256_file(correction.DEFAULT_QUEUE)},
            "correction_predictions": {"path": str(correction.DEFAULT_PREDICTIONS.resolve()), "sha256": sha256_file(correction.DEFAULT_PREDICTIONS)},
            "original_queue": {"path": str(original.DEFAULT_QUEUE.resolve()), "sha256": sha256_file(original.DEFAULT_QUEUE)},
            "original_predictions": {"path": str(original.DEFAULT_PREDICTIONS.resolve()), "sha256": sha256_file(original.DEFAULT_PREDICTIONS)},
        },
    }
    report["report_sha256"] = sha256_json(report)
    write_json(args.output, report)
    lines = [
        "# V3 最终业务盲测校正组结果",
        "",
        f"- 状态：`{status}`",
        f"- 可计分任务：{len(original_rows) + len(correction_rows)}（原始有效 {len(original_rows)}，校正组 {len(correction_rows)}/{correction.EXPECTED_JOBS}）",
        f"- 中心 dataset-equal MAPE：{center['dataset_equal_mape']:.2%}" if center["dataset_equal_mape"] is not None else "- 中心 MAPE：N/A",
        f"- 中心 dataset-equal bias：{center['dataset_equal_bias']:+.2%}" if center["dataset_equal_bias"] is not None else "- 中心 bias：N/A",
        f"- 安全配置放行率：{primary['safe_admission']['numerator']}/{primary['safe_admission']['denominator']} = {primary['safe_admission']['rate']:.2%}" if primary["safe_admission"]["rate"] is not None else "- 安全配置放行率：N/A",
        f"- OOM 放行率：{primary['oom_admission']['numerator']}/{primary['oom_admission']['denominator']} = {primary['oom_admission']['rate']:.2%}" if primary["oom_admission"]["rate"] is not None else "- OOM 放行率：N/A",
        "",
        "校正组是前瞻性补充证据，不冒充原始 pre-family-blind 结果。",
    ]
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": status, "jobs": len(correction_rows), "primary": primary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
