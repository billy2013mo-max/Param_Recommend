#!/usr/bin/env python3
"""Evaluate Phase 1 throughput transfer and freeze the conditional-repeat set."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, RESULTS_DIR, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_packing_final_virtual_mbs_v1 import (
    CAMPAIGN_ID,
    EXPECTED_JOBS,
    MEASURE_STEPS,
    PHASE_ID,
    QUEUE,
    WARMUP_STEPS,
)


OUTPUT = ARTIFACT_DIR / "h800_packing_final_virtual_mbs_results_v1.json"
REPEAT_SELECTION = ARTIFACT_DIR / "h800_packing_final_virtual_mbs_repeat_selection_v1.json"
BLOCK_SIZE = 5
BLOCK_CV_THRESHOLD = 0.03
TOP2_GAP_THRESHOLD = 0.03


def _events(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _block_cv(summaries: list[dict[str, Any]], metrics_dir: Path) -> tuple[float | None, list[float]]:
    step_samples: dict[int, float] = {}
    step_seconds: dict[int, float] = {}
    for summary in summaries:
        rank = int(summary["rank"])
        for step in (summary.get("batch_shape_evidence") or {}).get("measured_steps", []):
            global_step = int(step["global_step"])
            step_samples[global_step] = step_samples.get(global_step, 0.0) + float(
                step["logical_sample_count"]
            )
        for event in _events(metrics_dir / f"events.rank{rank}.jsonl"):
            if event.get("event") != "step_end" or event.get("is_warmup") is not False:
                continue
            global_step = int(event["global_step"])
            step_seconds[global_step] = max(
                step_seconds.get(global_step, 0.0), float(event["step_seconds"])
            )
    steps = sorted(set(step_samples) & set(step_seconds))
    if len(steps) != MEASURE_STEPS:
        return None, []
    block_tps = []
    for start in range(0, len(steps), BLOCK_SIZE):
        block = steps[start : start + BLOCK_SIZE]
        if len(block) != BLOCK_SIZE:
            return None, []
        block_tps.append(
            sum(step_samples[step] for step in block)
            / sum(step_seconds[step] for step in block)
        )
    mean = statistics.fmean(block_tps)
    cv = statistics.pstdev(block_tps) / mean if mean else None
    return cv, block_tps


def _job_result(job: dict[str, Any]) -> dict[str, Any]:
    root = RESULTS_DIR / str(job["job_id"])
    status_path = root / "status.json"
    result: dict[str, Any] = {
        "job_id": job["job_id"],
        "matched_group_id": job["matched_group_id"],
        "workload_id": job["workload_id"],
        "gpu_count": int(job["gpu_count"]),
        "arm_id": job["arm_id"],
        "packing": bool(job["packing"]),
        "mbs": int(job["mbs"]),
        "virtual_mbs": float(job["virtual_mbs"]),
        "classification": "missing",
        "all_passed": False,
    }
    if not status_path.is_file():
        return result
    status = read_json(status_path)
    result["classification"] = status.get("classification")
    latest = read_json(root / "latest_attempt.json")
    attempt_id = str(latest.get("execution_attempt_id") or "")
    attempt_root = root / "attempts" / attempt_id
    metrics_dir = attempt_root / "metrics"
    summaries = [read_json(path) for path in sorted(metrics_dir.glob("summary.rank*.json"))]
    gpu_count = int(job["gpu_count"])
    ranks_exact = (
        len(summaries) == gpu_count
        and {int(row.get("rank", -1)) for row in summaries} == set(range(gpu_count))
        and all(int(row.get("world_size", 0)) == gpu_count for row in summaries)
        and all(int(row.get("measured_steps", 0)) == MEASURE_STEPS for row in summaries)
    )
    ledgers = bool(summaries) and all(
        (row.get("token_ledger_evidence") or {}).get("authoritative") is True
        and (row.get("batch_shape_evidence") or {}).get("authoritative") is True
        for row in summaries
    )
    semantics = (
        all(
            ((row.get("runtime_batch_evidence") or {}).get("packing") or {}).get("semantic_checks_passed") is True
            for row in summaries
        )
        if bool(job["packing"])
        else True
    )
    global_samples = sum(
        float((row.get("measured_totals") or {}).get("logical_samples") or 0)
        for row in summaries
    )
    global_effective_tokens = sum(
        float((row.get("measured_totals") or {}).get("effective_tokens") or 0)
        for row in summaries
    )
    measured_seconds = max(
        (float(row.get("measured_seconds") or 0) for row in summaries), default=0.0
    )
    logical_tps = global_samples / measured_seconds if measured_seconds else None
    effective_tps = global_effective_tokens / measured_seconds if measured_seconds else None
    block_cv, block_tps = _block_cv(summaries, metrics_dir) if ranks_exact else (None, [])
    checks = {
        "success": status.get("classification") == "success",
        "calibration_eligible": status.get("calibration_eligible") is True,
        "rank_world_and_steps_exact": ranks_exact,
        "authoritative_ledgers": ledgers,
        "packing_semantics": semantics,
        "throughput_positive": logical_tps is not None and logical_tps > 0,
        "four_complete_five_step_blocks": block_cv is not None and len(block_tps) == 4,
    }
    result.update(
        {
            "execution_attempt_id": attempt_id,
            "measured_global_logical_samples": global_samples,
            "measured_seconds_conservative": measured_seconds,
            "logical_samples_per_second": logical_tps,
            "effective_tokens_per_second": effective_tps,
            "block_tps": block_tps,
            "block_cv": block_cv,
            "checks": checks,
            "all_passed": all(checks.values()),
        }
    )
    return result


def _group_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {str(row["arm_id"]): row for row in rows}
    complete = set(by_arm) == {"packed", "floor", "ceil"} and all(
        row["all_passed"] for row in rows
    )
    result: dict[str, Any] = {
        "matched_group_id": rows[0]["matched_group_id"],
        "workload_id": rows[0]["workload_id"],
        "gpu_count": rows[0]["gpu_count"],
        "complete": complete,
        "arms": by_arm,
    }
    if not complete:
        return result
    packed = float(by_arm["packed"]["logical_samples_per_second"])
    floor_tps = float(by_arm["floor"]["logical_samples_per_second"])
    ceil_tps = float(by_arm["ceil"]["logical_samples_per_second"])
    floor_mbs = float(by_arm["floor"]["mbs"])
    ceil_mbs = float(by_arm["ceil"]["mbs"])
    virtual_mbs = float(by_arm["packed"]["virtual_mbs"])
    weight = (virtual_mbs - floor_mbs) / (ceil_mbs - floor_mbs)
    interpolated = math.exp((1.0 - weight) * math.log(floor_tps) + weight * math.log(ceil_tps))
    ratio = packed / interpolated
    throughputs = sorted(
        (float(row["logical_samples_per_second"]), str(row["arm_id"])) for row in rows
    )
    top, second = throughputs[-1][0], throughputs[-2][0]
    top2_gap = (top - second) / top
    max_block_cv = max(float(row["block_cv"]) for row in rows)
    repeat_required = max_block_cv > BLOCK_CV_THRESHOLD or top2_gap < TOP2_GAP_THRESHOLD
    result.update(
        {
            "unpacked_interpolated_virtual_mbs_tps": interpolated,
            "packed_tps": packed,
            "packed_over_virtual_unpacked_ratio": ratio,
            "t0_absolute_percentage_error": abs(ratio - 1.0),
            "top2_relative_gap": top2_gap,
            "maximum_arm_block_cv": max_block_cv,
            "repeat_required": repeat_required,
            "repeat_reasons": [
                reason
                for condition, reason in (
                    (max_block_cv > BLOCK_CV_THRESHOLD, "block_cv_above_3pct"),
                    (top2_gap < TOP2_GAP_THRESHOLD, "top2_gap_below_3pct"),
                )
                if condition
            ],
        }
    )
    return result


def evaluate() -> dict[str, Any]:
    jobs = read_jsonl(QUEUE)
    if len(jobs) != EXPECTED_JOBS or any(row.get("campaign_id") != CAMPAIGN_ID for row in jobs):
        raise ValueError("queue is not the exact Phase 1 experiment")
    job_results = [_job_result(job) for job in jobs]
    groups = []
    for group_id in sorted({str(row["matched_group_id"]) for row in job_results}):
        groups.append(_group_result([row for row in job_results if row["matched_group_id"] == group_id]))
    complete_groups = [row for row in groups if row["complete"]]
    t0_errors = [float(row["t0_absolute_percentage_error"]) for row in complete_groups]
    ratios = [float(row["packed_over_virtual_unpacked_ratio"]) for row in complete_groups]
    global_scalar = math.exp(statistics.fmean(math.log(value) for value in ratios)) if ratios else None
    t1_errors = [abs(value / global_scalar - 1.0) for value in ratios] if global_scalar else []
    repeat_groups = [str(row["matched_group_id"]) for row in complete_groups if row["repeat_required"]]
    repeat_job_ids = [
        str(job["job_id"])
        for job in jobs
        if str(job["matched_group_id"]) in set(repeat_groups)
    ]
    selection: dict[str, Any] = {
        "schema": "sft_h800_packing_final_virtual_mbs_repeat_selection/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_after_all_base_results": len(complete_groups) == 12,
        "policy": {
            "block_cv_threshold": BLOCK_CV_THRESHOLD,
            "top2_gap_threshold": TOP2_GAP_THRESHOLD,
            "repeat_whole_matched_group": True,
        },
        "repeat_group_ids": repeat_groups,
        "source_repeat_job_ids": repeat_job_ids,
        "conditional_repeat_launches": len(repeat_job_ids),
        "maximum_allowed": 36,
    }
    selection["report_sha256"] = sha256_json(selection)
    write_json(REPEAT_SELECTION, selection)
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_final_virtual_mbs_results/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue": {"path": str(QUEUE.resolve()), "sha256": sha256_file(QUEUE)},
        "completion": {
            "jobs": len(job_results),
            "passed": sum(row["all_passed"] for row in job_results),
            "groups_complete": len(complete_groups),
            "classifications": dict(Counter(str(row["classification"]) for row in job_results)),
        },
        "direct_transfer": {
            "t0_group_equal_mape": statistics.fmean(t0_errors) if t0_errors else None,
            "t0_max_ape": max(t0_errors) if t0_errors else None,
            "t1_global_scalar": global_scalar,
            "t1_group_equal_mape": statistics.fmean(t1_errors) if t1_errors else None,
        },
        "noise": {
            "block_cv_p90": sorted(
                float(row["block_cv"]) for row in job_results if row.get("block_cv") is not None
            )[min(35, math.ceil(0.9 * 36) - 1)]
            if len(job_results) == 36 and all(row.get("block_cv") is not None for row in job_results)
            else None,
            "repeat_groups": repeat_groups,
            "conditional_repeat_launches": len(repeat_job_ids),
        },
        "gates": {
            "all_36_base_arms_passed": len(complete_groups) == 12 and all(row["all_passed"] for row in job_results),
            "t0_direct_substitution_fit_gate_mape_le_12pct": bool(t0_errors) and statistics.fmean(t0_errors) <= 0.12,
            "within_run_block_cv_p90_le_3pct": len(job_results) == 36
            and all(row.get("block_cv") is not None for row in job_results)
            and sorted(float(row["block_cv"]) for row in job_results)[math.ceil(0.9 * 36) - 1] <= 0.03,
            "automatic_publication_allowed": False,
        },
        "repeat_selection": {
            "path": str(REPEAT_SELECTION.resolve()),
            "sha256": sha256_file(REPEAT_SELECTION),
        },
        "groups": groups,
        "jobs": job_results,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    return report


def main() -> None:
    report = evaluate()
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "completion": report["completion"],
                "direct_transfer": report["direct_transfer"],
                "noise": report["noise"],
                "gates": report["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["gates"]["all_36_base_arms_passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
