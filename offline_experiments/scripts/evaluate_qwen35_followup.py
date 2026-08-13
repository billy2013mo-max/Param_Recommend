#!/usr/bin/env python3
"""Replay frozen H800 predictors on the Qwen3.5 follow-up holdout.

The 25 follow-up outcomes are labels only.  This script never refits a
predictor and keeps the nine memory-boundary jobs out of throughput scoring.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np

import compare_memory_model_generations as memory_replay
import evaluate_qwen25_qwen35_holdout as replay
from common import read_json, sha256_file, sha256_json, write_json
from throughput_predictor import ThroughputPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "offline_experiments"
CAMPAIGN = PROJECT_ROOT / "offline_experiments_qwen35_tilelang_20260729"
QUEUE = CAMPAIGN / "matrix" / "queue_qwen35_followup.jsonl"
RESULTS = CAMPAIGN / "results"
FREEZE = CAMPAIGN / "artifacts" / "predictor_freeze_before_holdout.json"
HOTFIX = CAMPAIGN / "artifacts" / "qwen35_mbs1_transformers_hotfix.json"
DEFAULT_OUTPUT = (
    CAMPAIGN / "artifacts" / "qwen35_followup_frozen_model_replay.json"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_queue() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in QUEUE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _collect_outcomes() -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for queue_index, job in enumerate(_read_queue()):
        job_id = str(job["job_id"])
        status, attempt = replay._latest_terminal_attempt(RESULTS, job_id)
        state = str(status["classification"])
        if state not in {"success", "oom"}:
            raise ValueError(
                f"Follow-up is not ready: {job_id} has terminal state {state!r}"
            )
        observed = (
            replay._aggregate_success(job, status, attempt)
            if state == "success"
            else None
        )
        outcomes.append(
            {
                "cohort": "qwen35_generalization_followup",
                "queue_path": str(QUEUE.resolve()),
                "queue_index": queue_index,
                "memory_evidence": True,
                "throughput_role": (
                    "primary_formal"
                    if str(job["kind"]) == "throughput"
                    else "excluded_memory_probe"
                ),
                "job": dict(job),
                "job_id": job_id,
                "state": state,
                "status": dict(status),
                "attempt_path": str(attempt.resolve()),
                "observed": observed,
            }
        )
    return outcomes


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), quantile))


def _memory_scope(
    details: Sequence[Mapping[str, Any]],
    job_ids: set[str],
) -> dict[str, Any]:
    rows = [row for row in details if str(row["job_id"]) in job_ids]
    success = [row for row in rows if row["outcome"] == "success"]
    oom = [row for row in rows if row["outcome"] == "oom"]
    safe_success = [
        row for row in success if row["actual_safe_success"] is True
    ]
    observed_over_safe = [
        row for row in success if row["actual_safe_success"] is False
    ]
    apes = [
        100.0 * float(row["center_absolute_percentage_error"])
        for row in success
        if row.get("center_absolute_percentage_error") is not None
    ]
    admitted = [row for row in rows if row["predicted_admit"] is True]
    false_safe = [row for row in oom if row["predicted_admit"] is True]
    admitted_unsafe_success = [
        row
        for row in observed_over_safe
        if row["predicted_admit"] is True
    ]
    return {
        "rows": len(rows),
        "state_counts": dict(
            sorted(Counter(str(row["outcome"]) for row in rows).items())
        ),
        "center_accuracy_success_only": {
            "count": len(apes),
            "mape_percent": (
                statistics.fmean(apes) if apes else None
            ),
            "median_ape_percent": (
                statistics.median(apes) if apes else None
            ),
            "p90_ape_percent": _percentile(apes, 90.0),
            "max_ape_percent": max(apes) if apes else None,
        },
        "operational_upper": {
            "success_rows": len(success),
            "success_upper_coverage_count": sum(
                row["upper_covers_observed"] is True for row in success
            ),
            "success_upper_coverage_percent": (
                100.0
                * sum(
                    row["upper_covers_observed"] is True
                    for row in success
                )
                / len(success)
                if success
                else None
            ),
        },
        "admission": {
            "predicted_admitted_rows": len(admitted),
            "oom_rows": len(oom),
            "false_safe_oom": len(false_safe),
            "false_safe_oom_rate_percent": (
                100.0 * len(false_safe) / len(oom) if oom else None
            ),
            "actual_safe_success_rows": len(safe_success),
            "admitted_safe_success_rows": sum(
                row["predicted_admit"] is True for row in safe_success
            ),
            "false_reject_safe_success_rows": sum(
                row["predicted_admit"] is False for row in safe_success
            ),
            "safe_success_admission_recall_percent": (
                100.0
                * sum(
                    row["predicted_admit"] is True
                    for row in safe_success
                )
                / len(safe_success)
                if safe_success
                else None
            ),
            "observed_over_safe_line_success_rows": len(
                observed_over_safe
            ),
            "admitted_observed_over_safe_line_success_rows": len(
                admitted_unsafe_success
            ),
            "memory_admission_safety_failures": (
                len(false_safe) + len(admitted_unsafe_success)
            ),
        },
        "details": rows,
    }


def _throughput_replay(
    outcomes: Sequence[Mapping[str, Any]],
    physical_admission: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    predictor = ThroughputPredictor(
        model_inventory=replay.NEW_INVENTORY,
        strict_bindings=False,
    )
    entries_by_job = replay._throughput_entries(outcomes, predictor)
    formal = [
        entry
        for entry in entries_by_job.values()
        if entry["throughput_role"] == "primary_formal"
    ]
    successful = [
        entry for entry in formal if entry["state"] == "success"
    ]
    raw_rates = replay._rates_for_scope(successful)
    raw = replay._score_throughput_scope(
        successful,
        rates=raw_rates,
    )

    admitted = [
        entry
        for entry in formal
        if physical_admission[str(entry["job_id"])]["predicted_admit"]
    ]
    admitted_rates = replay._rates_for_scope(admitted)
    admitted_success = [
        entry for entry in admitted if entry["state"] == "success"
    ]
    filtered = replay._score_throughput_scope(
        admitted_success,
        rates=admitted_rates,
    )
    pipeline = replay._pipeline_metrics(
        formal,
        physical_admission,
        admitted_rates,
    )
    return {
        "evaluation_contract": {
            "new_outcomes_refit_any_model": False,
            "formal_jobs": len(formal),
            "formal_success_rows": len(successful),
            "formal_oom_rows": sum(
                entry["state"] == "oom" for entry in formal
            ),
            "scenarios": len(
                {str(entry["scenario_id"]) for entry in formal}
            ),
            "exact_repeats_per_configuration": 2,
            "ranking_unit": (
                "exact repeats averaged before pairwise, Top-1 and regret"
            ),
            "v4b_candidate_set_policy": (
                "V4b is recomputed on the exact candidate set for raw and "
                "physical-shares-filtered scoring"
            ),
        },
        "raw_success_only": raw,
        "physical_shares_filtered": {
            "admitted_candidates_including_terminal_oom": len(admitted),
            "admitted_success_rows": len(admitted_success),
            "admitted_oom_rows": sum(
                entry["state"] == "oom" for entry in admitted
            ),
            "models": filtered,
        },
        "end_to_end_physical_shares_plus_throughput": pipeline,
        "predictor_binding_mismatches": predictor.binding_mismatches,
    }


def _freeze_audit() -> dict[str, Any]:
    freeze = read_json(FREEZE)
    checks = []
    for item in freeze["artifacts"]:
        path = Path(item["path"])
        current = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "path": str(path),
                "expected_sha256": str(item["sha256"]),
                "current_sha256": current,
                "matches": current == str(item["sha256"]),
            }
        )
    return {
        "path": str(FREEZE.resolve()),
        "sha256": sha256_file(FREEZE),
        "all_frozen_artifacts_unchanged": all(
            row["matches"] for row in checks
        ),
        "checks": checks,
    }


def build_report(output: Path) -> dict[str, Any]:
    outcomes = _collect_outcomes()
    memory, physical_admission = replay._memory_replay(outcomes)
    details = memory["models"][memory_replay.MODEL_PHYSICAL]["overall"][
        "details"
    ]
    boundary_ids = {
        str(row["job_id"])
        for row in outcomes
        if str(row["job"]["kind"]) == "memory_probe"
    }
    throughput_ids = {
        str(row["job_id"])
        for row in outcomes
        if str(row["job"]["kind"]) == "throughput"
    }
    memory["physical_shares_scopes"] = {
        "memory_boundary_only": _memory_scope(details, boundary_ids),
        "formal_throughput_candidates": _memory_scope(
            details, throughput_ids
        ),
        "all_followup_jobs": _memory_scope(
            details, boundary_ids | throughput_ids
        ),
    }
    report = {
        "schema": "qwen35_followup_frozen_model_replay/v1",
        "implementation_version": (
            "qwen35_followup_frozen_model_replay/2026-07-29.v1"
        ),
        "generated_at_utc": _utc_now(),
        "evaluation_contract": {
            "gpu_experiments_launched_by_this_script": False,
            "queue_mutated": False,
            "new_outcomes_used_as_labels_only": True,
            "new_outcomes_used_to_refit_models": False,
            "memory_models_replayed": list(replay.MEMORY_MODEL_IDS),
            "throughput_models_replayed": list(
                replay.THROUGHPUT_MODEL_IDS
            ),
        },
        "source_bindings": {
            "queue": {
                "path": str(QUEUE.resolve()),
                "sha256": sha256_file(QUEUE),
                "jobs": len(outcomes),
            },
            "predictor_freeze": {
                "path": str(FREEZE.resolve()),
                "sha256": sha256_file(FREEZE),
            },
            "runtime_hotfix": (
                {
                    "path": str(HOTFIX.resolve()),
                    "sha256": sha256_file(HOTFIX),
                }
                if HOTFIX.is_file()
                else None
            ),
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
        },
        "freeze_audit": _freeze_audit(),
        "outcome_snapshot": {
            "rows": len(outcomes),
            "state_counts": dict(
                sorted(
                    Counter(
                        str(outcome["state"]) for outcome in outcomes
                    ).items()
                )
            ),
            "memory_boundary_rows": len(boundary_ids),
            "formal_throughput_rows": len(throughput_ids),
        },
        "memory_replay": memory,
        "throughput_replay": _throughput_replay(
            outcomes, physical_admission
        ),
    }
    report["report_sha256"] = sha256_json(report)
    write_json(output, report)
    return report


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def _print_summary(report: Mapping[str, Any]) -> None:
    print("outcomes", report["outcome_snapshot"]["state_counts"])
    boundary = report["memory_replay"]["physical_shares_scopes"][
        "memory_boundary_only"
    ]
    print(
        "physical_shares_boundary",
        "MAPE=" + _fmt(
            boundary["center_accuracy_success_only"]["mape_percent"]
        ),
        "coverage=" + _fmt(
            boundary["operational_upper"][
                "success_upper_coverage_percent"
            ]
        ),
        "false_safe="
        + f"{boundary['admission']['false_safe_oom']}/"
        + f"{boundary['admission']['oom_rows']}",
        "safe_recall=" + _fmt(
            boundary["admission"][
                "safe_success_admission_recall_percent"
            ]
        ),
        "safety_failures="
        + str(
            boundary["admission"][
                "memory_admission_safety_failures"
            ]
        ),
    )
    raw = report["throughput_replay"]["raw_success_only"]
    for model_id in replay.THROUGHPUT_MODEL_IDS:
        metrics = raw[model_id]["metrics"]
        print(
            "throughput",
            model_id,
            "MAPE="
            + _fmt(metrics["absolute"]["throughput_mape_percent"]),
            "pairwise="
            + _fmt(
                metrics["ranking"][
                    "pooled_pairwise_accuracy_percent"
                ]
            ),
            "top1="
            + _fmt(
                metrics["ranking"]["top_k"][
                    "exact_top1_hit_fraction"
                ]
            ),
            "hit90="
            + _fmt(
                metrics["ranking"]["top_k"][
                    "selected_at_least_90pct_of_best_fraction"
                ]
            ),
            "regret="
            + _fmt(metrics["ranking"]["mean_top1_regret_percent"]),
        )
    print("report_sha256", report["report_sha256"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.output.resolve())
    _print_summary(report)


if __name__ == "__main__":
    main()
