#!/usr/bin/env python3
"""Audit the unified-resource evidence campaign and collapse repeat units.

This evaluator never fits or publishes a model.  It preserves successful
``max_reserved`` values as exact center observations, marks CUDA OOM as a
right-censored constraint without inventing a peak, collapses Critical seed
runs to four profile scenarios, and collapses Packing repeats to physical arms
for memory fitting.  Throughput repeats remain separate observations.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    RESULTS_DIR,
    read_json,
    read_jsonl,
    sha256_json,
    write_json,
)
from prepare_h800_unified_resource_evidence_v1 import CAMPAIGN_ID, PHASE_ID, QUEUE

GIB = float(1 << 30)
SCHEMA = "sft_h800_unified_resource_evidence_results/v1"
EXPECTED_JOBS = 220
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_unified_resource_evidence_results_v1.json"
DEFAULT_MARKDOWN = ARTIFACT_DIR / "h800_unified_resource_evidence_results_v1.md"


def _latest_attempt_dir(job_id: str, results_root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    status_view = results_root / job_id / "status.json"
    if not status_view.is_file():
        return None, None
    status_path = status_view.resolve()
    try:
        status = read_json(status_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None, None
    if status.get("job_id") != job_id:
        return None, None
    return status_path.parent, status


def _summaries(attempt_dir: Path) -> list[dict[str, Any]]:
    parsed = []
    for path in sorted((attempt_dir / "metrics").glob("summary.rank*.json")):
        try:
            parsed.append(read_json(path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return parsed


def _success_measurements(
    job: dict[str, Any], summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    ranks_exact = len(summaries) == int(job["gpu_count"])
    measured_steps = {int(row.get("measured_steps") or 0) for row in summaries}
    steps_exact = ranks_exact and len(measured_steps) == 1 and next(iter(measured_steps)) > 0
    steps = next(iter(measured_steps)) if steps_exact else None
    logical_samples = sum(
        float((row.get("measured_totals") or {}).get("logical_samples") or 0)
        for row in summaries
    )
    peak_reserved = max(
        (int(row.get("max_reserved") or 0) for row in summaries), default=0
    )
    peak_allocated = max(
        (int(row.get("max_allocated") or 0) for row in summaries), default=0
    )
    token_ledger_authoritative = bool(
        ranks_exact
        and all(
            (row.get("token_ledger_evidence") or {}).get("authoritative") is True
            for row in summaries
        )
    )
    shape_ledger_authoritative = bool(
        ranks_exact
        and all(
            (row.get("batch_shape_evidence") or {}).get("authoritative") is True
            for row in summaries
        )
    )
    shape_memory_steps = []
    for rank, summary in enumerate(summaries):
        evidence = summary.get("batch_shape_evidence") or {}
        for step in evidence.get("measured_steps") or []:
            shape_memory_steps.append({"rank": rank, **dict(step)})
    return {
        "metrics_rank_count_exact": ranks_exact,
        "measured_steps_consistent": steps_exact,
        "measured_steps": steps,
        "global_logical_samples": logical_samples if steps_exact else None,
        "observed_sample_gbs": logical_samples / steps if steps else None,
        "observed_sample_gbs_relative_error": (
            abs(logical_samples / steps - float(job["target_gbs"]))
            / float(job["target_gbs"])
            if steps
            else None
        ),
        "global_logical_samples_per_second": sum(
            float(row.get("logical_samples_per_second") or 0) for row in summaries
        )
        if ranks_exact
        else None,
        "global_effective_tokens_per_second": sum(
            float(row.get("effective_tokens_per_second") or 0) for row in summaries
        )
        if ranks_exact
        else None,
        "max_reserved_bytes": peak_reserved or None,
        "max_reserved_gib": peak_reserved / GIB if peak_reserved else None,
        "max_allocated_bytes": peak_allocated or None,
        "max_allocated_gib": peak_allocated / GIB if peak_allocated else None,
        "token_ledger_authoritative_all_ranks": token_ledger_authoritative,
        "batch_shape_ledger_authoritative_all_ranks": shape_ledger_authoritative,
        "shape_memory_steps": shape_memory_steps,
    }


def collect(queue_path: Path, results_root: Path) -> list[dict[str, Any]]:
    jobs = read_jsonl(queue_path)
    observations = []
    for job in jobs:
        job_id = str(job["job_id"])
        attempt_dir, status = _latest_attempt_dir(job_id, results_root)
        row = {
            "job_id": job_id,
            "campaign_id": job.get("campaign_id"),
            "phase_id": job.get("phase_id"),
            "evidence_role": job["evidence_role"],
            "scenario_id": job["scenario_id"],
            "mechanism_id": job["mechanism_id"],
            "split_unit_id": job["split_unit_id"],
            "model_id": job["model_id"],
            "train_type": job["train_type"],
            "gpu_count": job["gpu_count"],
            "zero_stage": job["zero_stage"],
            "gc": job["gc"],
            "mbs": job["mbs"],
            "cutoff_len": job["cutoff_len"],
            "packing": job["packing"],
            "repeat": job["repeat"],
            "seed": job["seed"],
            "classification": "missing",
            "calibration_eligible": False,
            "memory_observation_kind": "missing",
            "exact_center_target_bytes": None,
            "oom_peak_is_unknown_not_imputed": None,
        }
        if status is None or attempt_dir is None:
            observations.append(row)
            continue
        classification = str(status.get("classification") or "unknown")
        row.update(
            {
                "classification": classification,
                "calibration_eligible": status.get("calibration_eligible") is True,
                "execution_attempt_id": status.get("execution_attempt_id"),
                "wall_seconds": status.get("wall_seconds"),
                "gpu_mask": status.get("gpu_mask"),
            }
        )
        if classification == "success":
            measurements = _success_measurements(job, _summaries(attempt_dir))
            row["measurements"] = measurements
            row["memory_observation_kind"] = "exact_success_center"
            row["exact_center_target_bytes"] = measurements["max_reserved_bytes"]
            row["oom_peak_is_unknown_not_imputed"] = False
        elif classification == "oom":
            row["memory_observation_kind"] = "right_censored_oom"
            row["exact_center_target_bytes"] = None
            row["oom_peak_is_unknown_not_imputed"] = True
            row["right_censor_source"] = (
                "export_h800_observations.py derives the lower inequality from the "
                "attempt-bound CUDA OOM error; this evaluator never substitutes a peak"
            )
        observations.append(row)
    return observations


def _fit_group_key(row: dict[str, Any]) -> tuple[str, ...]:
    role = str(row["evidence_role"])
    if role == "critical_same_max_order_variance_diagnostic":
        return ("critical_profile", str(row["split_unit_id"]))
    if role == "unified_packing_matched_formal_fit":
        return (
            "packing_arm",
            str(row["split_unit_id"]),
            str(row["mechanism_id"]),
            "packed" if row["packing"] else "unpacked",
        )
    return ("single_nonpacking_configuration", str(row["job_id"]))


def collapse_memory_fit_units(observations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[_fit_group_key(row)].append(row)
    units = []
    for key, rows in sorted(grouped.items()):
        classifications = Counter(str(row["classification"]) for row in rows)
        eligible = [row for row in rows if row.get("calibration_eligible") is True]
        eligible_states = Counter(str(row["classification"]) for row in eligible)
        success_targets = [
            int(row["exact_center_target_bytes"])
            for row in eligible
            if row.get("classification") == "success"
            and row.get("exact_center_target_bytes") is not None
        ]
        if len(eligible) != len(rows):
            fit_state = "incomplete_or_ineligible"
        elif eligible_states == {"success": len(rows)}:
            fit_state = "exact_success_center"
        elif eligible_states == {"oom": len(rows)}:
            fit_state = "right_censored_oom"
        elif set(eligible_states) <= {"success", "oom"}:
            fit_state = "mixed_success_oom_instability"
        else:
            fit_state = "non_calibratable_failure"
        target = (
            statistics.median(success_targets)
            if fit_state == "exact_success_center" and success_targets
            else None
        )
        units.append(
            {
                "fit_unit_id": "::".join(key),
                "fit_unit_kind": key[0],
                "member_job_ids": [str(row["job_id"]) for row in rows],
                "member_count": len(rows),
                "classifications": dict(classifications),
                "eligible_classifications": dict(eligible_states),
                "fit_state": fit_state,
                "exact_center_target_bytes": target,
                "exact_center_target_gib": target / GIB if target is not None else None,
                "right_censored_peak_is_unknown_not_imputed": fit_state
                == "right_censored_oom",
                "repeat_policy": (
                    "median across repeated successes; mixed success/OOM is instability, "
                    "not an exact center"
                    if len(rows) > 1
                    else "single physical configuration"
                ),
            }
        )
    return units


def summarize(observations: list[dict[str, Any]]) -> dict[str, Any]:
    units = collapse_memory_fit_units(observations)
    completed = [row for row in observations if row["classification"] != "missing"]
    eligible = [row for row in completed if row["calibration_eligible"]]
    success = [row for row in eligible if row["classification"] == "success"]
    shape_ready = [
        row
        for row in success
        if (row.get("measurements") or {}).get(
            "batch_shape_ledger_authoritative_all_ranks"
        )
        is True
    ]
    return {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "model_refit": False,
        "publishable": False,
        "expected_jobs": EXPECTED_JOBS,
        "completed_jobs": len(completed),
        "calibration_eligible_jobs": len(eligible),
        "raw_outcomes": dict(Counter(str(row["classification"]) for row in observations)),
        "raw_evidence_roles": dict(Counter(str(row["evidence_role"]) for row in observations)),
        "memory_fit_units_after_required_collapse": len(units),
        "memory_fit_unit_states": dict(Counter(unit["fit_state"] for unit in units)),
        "authoritative_batch_shape_success_jobs": len(shape_ready),
        "campaign_complete": len(completed) == EXPECTED_JOBS,
        "fit_contract": {
            "success_max_reserved_is_exact_center": True,
            "oom_is_right_censored_and_never_imputed": True,
            "critical_20_runs_collapse_to_4_profile_units": True,
            "packing_72_runs_collapse_to_24_physical_memory_arms": True,
            "packing_repeats_remain_separate_for_throughput_noise": True,
            "clean_prospective_acceptance_is_separate": True,
        },
        "next_step": (
            "when all 220 jobs are calibration-eligible, export canonical H800 observations, "
            "fit one shared physics-informed model with grouped validation, freeze it, then "
            "generate a source-disjoint prospective acceptance queue"
        ),
        "memory_fit_units": units,
        "throughput_repeat_rows": [
            row
            for row in observations
            if row["evidence_role"] == "unified_packing_matched_formal_fit"
        ],
        "observations": observations,
    }


def _markdown(report: dict[str, Any]) -> str:
    outcomes = report["raw_outcomes"]
    states = report["memory_fit_unit_states"]
    return "\n".join(
        [
            "# 统一资源推荐器实验结果审计",
            "",
            f"- 完成作业：{report['completed_jobs']} / {report['expected_jobs']}",
            f"- 校准合格作业：{report['calibration_eligible_jobs']}",
            f"- 原始结果：`{json.dumps(outcomes, ensure_ascii=False, sort_keys=True)}`",
            f"- 折叠后显存拟合单元：{report['memory_fit_units_after_required_collapse']}",
            f"- 拟合单元状态：`{json.dumps(states, ensure_ascii=False, sort_keys=True)}`",
            f"- 长度—显存逐 step 遥测合格成功作业：{report['authoritative_batch_shape_success_jobs']}",
            "",
            (
                "成功作业的 `max_reserved` 是精确中心标签；OOM 只保留右删失不等式，"
                "绝不伪造显存峰值。20 个 Critical seed 作业只形成 4 个 profile 单元；"
                "72 个 Packing 作业只形成 24 个显存物理臂，重复仍用于吞吐噪声估计。"
            ),
            "",
            "本报告不能替代独立前瞻验收；发布前仍需在模型与候选策略冻结后生成新数据源队列。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write a progress report before all 220 terminal jobs exist",
    )
    args = parser.parse_args()
    jobs = read_jsonl(args.queue)
    if len(jobs) != EXPECTED_JOBS or any(
        job.get("campaign_id") != CAMPAIGN_ID for job in jobs
    ):
        raise ValueError("evaluator requires the sealed 220-job full queue")
    report = summarize(collect(args.queue, args.results))
    report["report_sha256"] = sha256_json(report)
    if not report["campaign_complete"] and not args.allow_incomplete:
        raise RuntimeError(
            f"campaign is incomplete: {report['completed_jobs']}/{EXPECTED_JOBS}; "
            "use --allow-incomplete for a progress-only report"
        )
    write_json(args.output, report)
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "markdown": str(args.markdown),
                "completed_jobs": report["completed_jobs"],
                "fit_units": report["memory_fit_units_after_required_collapse"],
                "publishable": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
