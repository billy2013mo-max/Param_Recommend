#!/usr/bin/env python3
"""Evaluate the frozen RTX 4090 Qwen2.5/Qwen3.5 holdout without refitting."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Any

from common import read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_rtx4090_small_family_generalization_20260730 import (
    DEFAULT_CAMPAIGN_ROOT,
    MODELS,
)


SCHEMA = "sft_rtx4090_small_family_frozen_replay/v1"
MODEL_IDS = tuple(MODELS)
DEFAULT_OUTPUT_NAME = "frozen_replay_evaluation.json"
GIB = float(2**30)


def _latest_status(result_root: Path) -> tuple[dict[str, Any], Path] | None:
    status_path = result_root / "status.json"
    if status_path.is_file():
        status = read_json(status_path)
        attempt_id = status.get("execution_attempt_id")
        attempt = (
            result_root / "attempts" / str(attempt_id)
            if attempt_id
            else result_root
        )
        if attempt.is_dir():
            return status, attempt
    candidates = []
    for path in (result_root / "attempts").glob("*/status.json"):
        status = read_json(path)
        candidates.append(
            (
                float(status.get("finished_unix") or 0.0),
                path.stat().st_mtime,
                status,
                path.parent,
            )
        )
    if not candidates:
        return None
    _, _, status, attempt = max(candidates)
    return status, attempt


def _observed(
    job: Mapping[str, Any],
    status: Mapping[str, Any],
    attempt: Path,
) -> dict[str, Any] | None:
    if status.get("classification") != "success":
        return None
    paths = sorted((attempt / "metrics").glob("summary.rank*.json"))
    if len(paths) != int(job["gpu_count"]):
        raise ValueError(
            f"{job['job_id']} has {len(paths)} summaries, "
            f"expected {job['gpu_count']}"
        )
    summaries = [read_json(path) for path in paths]
    measured_steps = {int(row["measured_steps"]) for row in summaries}
    if len(measured_steps) != 1:
        raise ValueError(f"{job['job_id']} rank measured_steps disagree")
    measured_seconds = max(
        float(row["measured_seconds"]) for row in summaries
    )
    effective_tokens = sum(
        int(row["measured_totals"]["effective_tokens"])
        for row in summaries
    )
    return {
        "measured_steps": measured_steps.pop(),
        "measured_seconds": measured_seconds,
        "effective_tokens": effective_tokens,
        "effective_tokens_per_second": effective_tokens / measured_seconds,
        "max_allocated_bytes": max(
            float(row["max_allocated"]) for row in summaries
        ),
        "max_reserved_bytes": max(
            float(row["max_reserved"]) for row in summaries
        ),
        "summary_paths": [str(path.resolve()) for path in paths],
    }


def _load(campaign_root: Path) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    freeze_path = campaign_root / "FREEZE.json"
    freeze = read_json(freeze_path)
    unsigned = dict(freeze)
    digest = unsigned.pop("freeze_sha256", None)
    if digest != sha256_json(unsigned):
        raise ValueError("FREEZE.json checksum mismatch")
    if (
        freeze.get("gpu_experiments_launched") is not False
        or freeze.get("results_consumed_for_model_fit") is not False
    ):
        raise ValueError("Prospective freeze contract drifted")
    prediction_path = Path(freeze["prediction_freeze"]["path"])
    if sha256_file(prediction_path) != freeze["prediction_freeze"]["sha256"]:
        raise ValueError("Frozen prediction checksum mismatch")
    jobs_path = Path(freeze["jobs"]["path"])
    if sha256_file(jobs_path) != freeze["jobs"]["sha256"]:
        raise ValueError("Frozen job queue checksum mismatch")
    return freeze, read_json(prediction_path), read_jsonl(jobs_path)


def _collect(
    campaign_root: Path,
    jobs: Sequence[Mapping[str, Any]],
    prediction: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    predicted = {
        str(row["request_id"]): row
        for row in prediction["predictions"]
    }
    rows = []
    missing = []
    for job in jobs:
        job_id = str(job["job_id"])
        result_root = (
            campaign_root
            / str(job["model_id"])
            / "results"
            / job_id
        )
        latest = _latest_status(result_root)
        if latest is None:
            missing.append(job_id)
            continue
        status, attempt = latest
        state = str(status.get("classification") or "unknown")
        if state not in {"success", "oom"}:
            raise ValueError(f"Unexpected terminal state {state}: {job_id}")
        request_id = str(job["prediction_request_id"])
        if request_id not in predicted:
            raise ValueError(f"Missing frozen prediction: {request_id}")
        rows.append(
            {
                "job": dict(job),
                "job_id": job_id,
                "request_id": request_id,
                "prediction": predicted[request_id],
                "state": state,
                "status_path": str((attempt / "status.json").resolve()),
                "observed": _observed(job, status, attempt),
            }
        )
    return rows, missing


def _memory_detail(row: Mapping[str, Any]) -> dict[str, Any]:
    prediction = row["prediction"]["memory"]
    observed = row["observed"]
    safe_limit = float(prediction["safe_limit_bytes"])
    predicted_admit = bool(prediction["admitted"])
    observed_reserved = (
        float(observed["max_reserved_bytes"])
        if observed is not None
        else None
    )
    actual_safe_success = bool(
        row["state"] == "success"
        and observed_reserved is not None
        and observed_reserved <= safe_limit
    )
    unsafe_actual = not actual_safe_success
    center = float(prediction["reserved_center_bytes"])
    return {
        "job_id": row["job_id"],
        "request_id": row["request_id"],
        "model_id": row["job"]["model_id"],
        "model_family": row["job"]["model_family"],
        "phase": row["job"]["phase_id"],
        "train_type": row["job"]["train_type"],
        "state": row["state"],
        "predicted_admit": predicted_admit,
        "actual_safe_success": actual_safe_success,
        "false_safe": predicted_admit and unsafe_actual,
        "false_reject": (not predicted_admit) and actual_safe_success,
        "safe_limit_gib": safe_limit / GIB,
        "predicted_reserved_center_gib": center / GIB,
        "predicted_unsafe_score": prediction.get("unsafe_score"),
        "observed_reserved_gib": (
            observed_reserved / GIB
            if observed_reserved is not None
            else None
        ),
        "center_absolute_error_gib": (
            abs(center - observed_reserved) / GIB
            if observed_reserved is not None
            else None
        ),
        "center_absolute_percentage_error": (
            abs(center - observed_reserved) / observed_reserved
            if observed_reserved
            else None
        ),
    }


def _memory_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    details = [_memory_detail(row) for row in rows]
    safe_successes = [
        row for row in details if row["actual_safe_success"]
    ]
    successful = [
        row for row in details if row["state"] == "success"
    ]
    apes = [
        float(row["center_absolute_percentage_error"])
        for row in successful
        if row["center_absolute_percentage_error"] is not None
    ]
    errors = [
        float(row["center_absolute_error_gib"])
        for row in successful
        if row["center_absolute_error_gib"] is not None
    ]
    return {
        "evaluated_jobs": len(details),
        "successes": len(successful),
        "ooms": sum(row["state"] == "oom" for row in details),
        "predicted_admitted": sum(
            row["predicted_admit"] for row in details
        ),
        "false_safe_count": sum(row["false_safe"] for row in details),
        "false_reject_count": sum(row["false_reject"] for row in details),
        "safe_successes": len(safe_successes),
        "safe_success_admission_recall": (
            sum(row["predicted_admit"] for row in safe_successes)
            / len(safe_successes)
            if safe_successes
            else None
        ),
        "center_mape_successes": statistics.fmean(apes) if apes else None,
        "center_mae_gib_successes": (
            statistics.fmean(errors) if errors else None
        ),
        "details": details,
    }


def _pairwise_accuracy(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    correct = total = 0
    for left, right in itertools.combinations(candidate_rows, 2):
        predicted_delta = (
            float(left["predicted_rate"]) - float(right["predicted_rate"])
        )
        observed_delta = (
            float(left["observed_rate"]) - float(right["observed_rate"])
        )
        if predicted_delta == 0.0 or observed_delta == 0.0:
            continue
        total += 1
        correct += (predicted_delta > 0) == (observed_delta > 0)
    return correct, total


def _throughput_metrics(
    rows: Sequence[Mapping[str, Any]],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    ranking_groups = {
        str(group["comparison_group"]): group
        for group in prediction["ranking_groups"]
    }
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["job"]["phase_id"] == "multi_candidate_ranking":
            groups[str(row["job"]["evaluation_group"])].append(row)
    scenarios = []
    pairwise_correct = pairwise_total = 0
    apes = []
    for group_id, group_rows in sorted(groups.items()):
        by_request: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_request[str(row["request_id"])].append(row)
        candidates = []
        for request_id, repeats in by_request.items():
            successful = [
                row for row in repeats if row["state"] == "success"
            ]
            prediction_row = repeats[0]["prediction"]
            predicted_rate = prediction_row["throughput"].get(
                "predicted_effective_tokens_per_second"
            )
            observed_rate = (
                statistics.median(
                    float(row["observed"]["effective_tokens_per_second"])
                    for row in successful
                )
                if successful
                else None
            )
            candidates.append(
                {
                    "request_id": request_id,
                    "candidate_label": repeats[0]["job"]["candidate_label"],
                    "gpu_count": repeats[0]["job"]["gpu_count"],
                    "predicted_admit": bool(
                        prediction_row["memory"]["admitted"]
                    ),
                    "predicted_rate": predicted_rate,
                    "observed_rate": observed_rate,
                    "successful_repeats": len(successful),
                    "total_repeats": len(repeats),
                }
            )
        observed_candidates = [
            row for row in candidates if row["observed_rate"] is not None
        ]
        predicted_candidates = [
            row
            for row in observed_candidates
            if row["predicted_rate"] is not None
        ]
        if not observed_candidates or not predicted_candidates:
            scenarios.append(
                {
                    "comparison_group": group_id,
                    "status": "insufficient_successful_candidates",
                    "candidates": candidates,
                }
            )
            continue
        observed_best = max(
            observed_candidates,
            key=lambda row: float(row["observed_rate"]),
        )
        ranked_ids = ranking_groups[group_id]["ranked_request_ids"]
        predicted_top = next(
            (
                row
                for request_id in ranked_ids
                for row in observed_candidates
                if row["request_id"] == request_id
            ),
            None,
        )
        if predicted_top is None:
            scenarios.append(
                {
                    "comparison_group": group_id,
                    "status": "predicted_top1_has_no_successful_measurement",
                    "candidates": candidates,
                }
            )
            continue
        correct, total = _pairwise_accuracy(predicted_candidates)
        pairwise_correct += correct
        pairwise_total += total
        for candidate in predicted_candidates:
            apes.append(
                abs(
                    float(candidate["predicted_rate"])
                    - float(candidate["observed_rate"])
                )
                / float(candidate["observed_rate"])
            )
        best_rate = float(observed_best["observed_rate"])
        predicted_top_rate = float(predicted_top["observed_rate"])
        scenarios.append(
            {
                "comparison_group": group_id,
                "status": "evaluated",
                "observed_best_request_id": observed_best["request_id"],
                "predicted_top1_request_id": predicted_top["request_id"],
                "exact_top1_hit": (
                    observed_best["request_id"] == predicted_top["request_id"]
                ),
                "hit_at_90_percent_of_best": (
                    predicted_top_rate >= 0.9 * best_rate
                ),
                "top1_regret": 1.0 - predicted_top_rate / best_rate,
                "pairwise_correct": correct,
                "pairwise_total": total,
                "candidates": candidates,
            }
        )
    evaluated = [
        row for row in scenarios if row["status"] == "evaluated"
    ]
    return {
        "evaluated_scenarios": len(evaluated),
        "pairwise_accuracy": (
            pairwise_correct / pairwise_total
            if pairwise_total
            else None
        ),
        "pairwise_correct": pairwise_correct,
        "pairwise_total": pairwise_total,
        "exact_top1_hit_fraction": (
            sum(row["exact_top1_hit"] for row in evaluated) / len(evaluated)
            if evaluated
            else None
        ),
        "hit_at_90_percent_of_best_fraction": (
            sum(row["hit_at_90_percent_of_best"] for row in evaluated)
            / len(evaluated)
            if evaluated
            else None
        ),
        "mean_top1_regret": (
            statistics.fmean(float(row["top1_regret"]) for row in evaluated)
            if evaluated
            else None
        ),
        "throughput_mape_secondary": (
            statistics.fmean(apes) if apes else None
        ),
        "scenarios": scenarios,
    }


def evaluate(campaign_root: Path) -> dict[str, Any]:
    freeze, prediction, jobs = _load(campaign_root)
    rows, missing = _collect(campaign_root, jobs, prediction)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_root": str(campaign_root.resolve()),
        "status": "pending_measurements" if missing else "complete",
        "results_used_for_refit": False,
        "source_bindings": {
            "freeze_sha256": freeze["freeze_sha256"],
            "prediction_sha256": freeze["prediction_freeze"]["sha256"],
            "jobs_sha256": freeze["jobs"]["sha256"],
        },
        "terminal_jobs": len(rows),
        "expected_jobs": len(jobs),
        "missing_job_ids": missing,
    }
    if not missing:
        memory = _memory_metrics(rows)
        throughput = _throughput_metrics(rows, prediction)
        acceptance = {
            "zero_false_safe": memory["false_safe_count"] == 0,
            "safe_success_recall_at_least_0p75": (
                memory["safe_success_admission_recall"] is not None
                and memory["safe_success_admission_recall"] >= 0.75
            ),
            "three_ranking_scenarios_evaluated": (
                throughput["evaluated_scenarios"] == 3
            ),
            "pairwise_accuracy_at_least_0p75": (
                throughput["pairwise_accuracy"] is not None
                and throughput["pairwise_accuracy"] >= 0.75
            ),
            "hit90_at_least_two_thirds": (
                throughput["hit_at_90_percent_of_best_fraction"] is not None
                and throughput["hit_at_90_percent_of_best_fraction"]
                >= 2 / 3
            ),
        }
        report.update(
            {
                "memory": memory,
                "throughput": throughput,
                "acceptance": acceptance,
                "promotion_ready": all(acceptance.values()),
            }
        )
    report["report_sha256"] = sha256_json(report)
    write_json(
        campaign_root / "artifacts" / DEFAULT_OUTPUT_NAME,
        report,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT,
    )
    args = parser.parse_args()
    report = evaluate(args.campaign_root.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
