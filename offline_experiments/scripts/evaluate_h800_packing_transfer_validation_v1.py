#!/usr/bin/env python3
"""Evaluate the pre-frozen 24-job Packing transfer validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

from common import ARTIFACT_DIR, read_json, read_jsonl, sha256_file, sha256_json, write_json, write_jsonl
from evaluate_h800_packing_config_ranking_v1 import collect_job
from prepare_h800_packing_transfer_validation_v1 import (
    CAMPAIGN_ID,
    FROZEN_PREDICTIONS,
    PHASE_ID,
    QUEUE,
)


SCHEMA = "sft_h800_packing_transfer_validation_results/v1"
OBSERVATION_SCHEMA = "sft_h800_packing_transfer_validation_observation/v1"
PREDICTION_SCHEMA = "sft_h800_packing_transfer_validation_frozen_predictions/v1"
OUTPUT = ARTIFACT_DIR / "h800_packing_transfer_validation_results_v1.json"
OBSERVATIONS = ARTIFACT_DIR / "h800_packing_transfer_validation_observations_v1.jsonl"


def _predictions(jobs: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = read_json(FROZEN_PREDICTIONS)
    binding = payload.get("queue_binding") or {}
    if (
        payload.get("schema") != PREDICTION_SCHEMA
        or payload.get("validation_outcomes_read") is not False
        or payload.get("coefficients_fitted") is not False
        or binding.get("path") != str(QUEUE.resolve())
        or binding.get("sha256") != sha256_file(QUEUE)
    ):
        raise ValueError("frozen predictions do not bind the exact untouched queue")
    rows = payload.get("predictions")
    if not isinstance(rows, list):
        raise ValueError("prediction rows are absent")
    by_id = {str(row.get("job_id")): row for row in rows if isinstance(row, dict)}
    expected = {str(job["job_id"]) for job in jobs}
    if len(by_id) != len(rows) or set(by_id) != expected:
        raise ValueError("predictions must cover all 24 jobs exactly once")
    for job_id, row in by_id.items():
        score = row.get("predicted_ranking_score")
        if (
            not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or float(score) <= 0.0
            or row.get("admitted") is not True
        ):
            raise ValueError(f"invalid frozen prediction for {job_id}")
    return by_id, payload


def _throughput(row: Mapping[str, Any]) -> float | None:
    outcome = row["outcome"]
    value = (outcome.get("throughput") or {}).get("global_logical_samples_per_second")
    return float(value) if outcome.get("usable_for_modeling") is True and value is not None else None


def _ranking(
    observations: list[dict[str, Any]], predictions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        if row.get("ranking_eligible") is True:
            groups[str(row["ranking_group_id"])].append(row)

    pair_correct = pair_total = 0
    hit90 = selected_oom = 0
    regrets: list[float] = []
    details: list[dict[str, Any]] = []
    for group_id, candidates in sorted(groups.items()):
        successful = [row for row in candidates if _throughput(row) is not None]
        current_correct = current_total = 0
        for left, right in itertools.combinations(successful, 2):
            left_truth = _throughput(left)
            right_truth = _throughput(right)
            assert left_truth is not None and right_truth is not None
            if math.isclose(left_truth, right_truth, rel_tol=0.0, abs_tol=1.0e-12):
                continue
            left_score = float(predictions[str(left["job_id"])]["predicted_ranking_score"])
            right_score = float(predictions[str(right["job_id"])]["predicted_ranking_score"])
            correct = (left_truth > right_truth) == (left_score > right_score)
            current_correct += int(correct)
            current_total += 1

        selected = max(
            candidates,
            key=lambda row: float(predictions[str(row["job_id"])]["predicted_ranking_score"]),
        )
        selected_tps = _throughput(selected)
        selected_class = str(selected["outcome"].get("classification") or "missing")
        selected_oom += int(selected_class == "oom")
        if successful:
            oracle = max(successful, key=lambda row: float(_throughput(row) or 0.0))
            oracle_tps = float(_throughput(oracle) or 0.0)
            regret = 1.0 if selected_tps is None else max(0.0, 1.0 - selected_tps / oracle_tps)
            within90 = selected_tps is not None and selected_tps >= 0.90 * oracle_tps
            oracle_id = str(oracle["job_id"])
        else:
            regret = 1.0
            within90 = False
            oracle_id = None
        pair_correct += current_correct
        pair_total += current_total
        hit90 += int(within90)
        regrets.append(regret)
        details.append(
            {
                "group_id": group_id,
                "candidate_count": len(candidates),
                "successful_candidate_count": len(successful),
                "pairwise_correct": current_correct,
                "pairwise_total": current_total,
                "predicted_top1_job_id": str(selected["job_id"]),
                "predicted_top1_classification": selected_class,
                "observed_top1_job_id": oracle_id,
                "hit90": within90,
                "top1_regret": regret,
            }
        )

    accuracy = pair_correct / pair_total if pair_total else None
    worst_regret = max(regrets) if regrets else None
    return {
        "scenario_count": len(groups),
        "pairwise_correct": pair_correct,
        "pairwise_total": pair_total,
        "pairwise_accuracy": accuracy,
        "hit90_hits": hit90,
        "hit90_required": len(groups),
        "selected_oom_count": selected_oom,
        "mean_top1_regret": statistics.fmean(regrets) if regrets else None,
        "worst_top1_regret": worst_regret,
        "details": details,
    }


def _repeat_diagnostic(observations: list[dict[str, Any]]) -> dict[str, Any]:
    repeat = next((row for row in observations if row["measurement_role"] == "transfer_noise_repeat"), None)
    if repeat is None:
        return {"available": False}
    key = (
        repeat["ranking_group_id"],
        repeat["candidate"]["cutoff_len"],
        repeat["candidate"]["zero_stage"],
        repeat["candidate"]["gradient_checkpointing"],
    )
    primary = next(
        (
            row
            for row in observations
            if row["measurement_role"] == "primary_ranking_candidate"
            and (
                row["ranking_group_id"],
                row["candidate"]["cutoff_len"],
                row["candidate"]["zero_stage"],
                row["candidate"]["gradient_checkpointing"],
            )
            == key
        ),
        None,
    )
    values = [value for value in (_throughput(primary) if primary else None, _throughput(repeat)) if value is not None]
    cv = statistics.pstdev(values) / statistics.fmean(values) if len(values) == 2 else None
    return {
        "available": len(values) == 2,
        "primary_job_id": str(primary["job_id"]) if primary else None,
        "repeat_job_id": str(repeat["job_id"]),
        "logical_samples_per_second": values,
        "coefficient_of_variation": cv,
    }


def evaluate() -> dict[str, Any]:
    jobs = read_jsonl(QUEUE)
    if len(jobs) != 24 or any(row.get("campaign_id") != CAMPAIGN_ID for row in jobs):
        raise ValueError("transfer queue drifted")
    predictions, frozen = _predictions(jobs)
    observations = []
    for job in jobs:
        row = collect_job(job)
        row.update({"schema": OBSERVATION_SCHEMA, "campaign_id": CAMPAIGN_ID, "phase_id": PHASE_ID})
        observations.append(row)
    write_jsonl(OBSERVATIONS, observations)
    classifications = Counter(str(row["outcome"]["classification"]) for row in observations)
    ranking = _ranking(observations, predictions)
    terminal = sum(classifications[name] for name in ("success", "oom"))
    admitted_oom = classifications["oom"]
    gates = {
        "all_24_terminal_success_or_oom": terminal == 24,
        "all_successes_modeling_usable": all(
            row["outcome"]["classification"] != "success"
            or row["outcome"].get("usable_for_modeling") is True
            for row in observations
        ),
        "six_scenes_evaluated": ranking["scenario_count"] == 6,
        "pairwise_accuracy_at_least_0p90": ranking["pairwise_accuracy"] is not None
        and float(ranking["pairwise_accuracy"]) >= 0.90,
        "hit90_six_of_six": ranking["hit90_hits"] == 6,
        "worst_top1_regret_below_0p10": ranking["worst_top1_regret"] is not None
        and float(ranking["worst_top1_regret"]) < 0.10,
        "no_predicted_safe_candidate_oom": admitted_oom == 0,
        "predicted_top1_oom_count_zero": ranking["selected_oom_count"] == 0,
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completion": {
            "jobs": len(observations),
            "classifications": dict(sorted(classifications.items())),
            "terminal_success_or_oom": terminal,
            "modeling_usable": sum(row["outcome"].get("usable_for_modeling") is True for row in observations),
        },
        "frozen_prediction_binding": {
            "path": str(FROZEN_PREDICTIONS.resolve()),
            "sha256": sha256_file(FROZEN_PREDICTIONS),
            "report_sha256": frozen.get("report_sha256"),
        },
        "observations": {"path": str(OBSERVATIONS.resolve()), "sha256": sha256_file(OBSERVATIONS)},
        "ranking": ranking,
        "repeat_diagnostic": _repeat_diagnostic(observations),
        "acceptance_gates": gates,
        "accepted": all(gates.values()),
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    return report


def main() -> None:
    report = evaluate()
    print(json.dumps({"output": str(OUTPUT.resolve()), "completion": report["completion"], "ranking": report["ranking"], "acceptance_gates": report["acceptance_gates"], "accepted": report["accepted"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
