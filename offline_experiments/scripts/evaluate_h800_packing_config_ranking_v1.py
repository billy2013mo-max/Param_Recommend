#!/usr/bin/env python3
"""Collect model-ready Packing observations and evaluate frozen ranking scores.

Collection works on a partially completed campaign and preserves missing/OOM
rows.  Ranking metrics are emitted only when an exact, pre-frozen prediction
artifact is supplied.  Predictions are never fitted in this script.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Any

from common import (
    ARTIFACT_DIR,
    RESULTS_DIR,
    percentile,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from prepare_h800_packing_config_ranking_v1 import (
    CAMPAIGN_ID,
    EXPECTED_JOBS,
    HOLDOUT_PREDICTIONS,
    MEASURE_STEPS,
    PHASE_ID,
    QUEUE_ALL,
    QUEUE_HOLDOUT,
)


SCHEMA = "sft_h800_packing_config_ranking_results/v2"
OBSERVATION_SCHEMA = "sft_h800_packing_config_ranking_observation/v2"
PREDICTION_SCHEMA = "sft_h800_packing_config_ranking_frozen_predictions/v2"
OUTPUT = ARTIFACT_DIR / "h800_packing_config_ranking_results_v2.json"
OBSERVATIONS = ARTIFACT_DIR / "h800_packing_config_ranking_observations_v2.jsonl"
MATERIAL_GAP = 0.03
BLOCK_SIZE = 4


def _latest_attempt_root(job_id: str) -> tuple[dict[str, Any], Path] | None:
    root = RESULTS_DIR / job_id
    latest_path = root / "latest_attempt.json"
    if not latest_path.is_file():
        return None
    latest = read_json(latest_path)
    attempt_id = str(latest.get("execution_attempt_id") or "")
    attempt = root / "attempts" / attempt_id
    if not attempt.is_dir():
        return None
    return latest, attempt


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _block_throughputs(
    summaries: Sequence[Mapping[str, Any]], metrics_dir: Path
) -> list[float]:
    samples: dict[int, float] = {}
    seconds: dict[int, float] = {}
    for summary in summaries:
        rank = int(summary["rank"])
        evidence = summary.get("batch_shape_evidence") or {}
        for step in evidence.get("measured_steps") or []:
            index = int(step["global_step"])
            samples[index] = samples.get(index, 0.0) + float(
                step["logical_sample_count"]
            )
        for event in _events(metrics_dir / f"events.rank{rank}.jsonl"):
            if event.get("event") != "step_end" or event.get("is_warmup") is not False:
                continue
            index = int(event["global_step"])
            seconds[index] = max(
                seconds.get(index, 0.0), float(event["step_seconds"])
            )
    steps = sorted(set(samples) & set(seconds))
    if len(steps) != MEASURE_STEPS:
        return []
    result = []
    for start in range(0, len(steps), BLOCK_SIZE):
        block = steps[start : start + BLOCK_SIZE]
        if len(block) != BLOCK_SIZE:
            return []
        result.append(
            sum(samples[index] for index in block)
            / sum(seconds[index] for index in block)
        )
    return result


def _distribution(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    numbers = [float(value) for value in values]
    histogram = Counter(str(int(value)) for value in numbers)
    return {
        "count": len(numbers),
        "minimum": min(numbers),
        "mean": statistics.fmean(numbers),
        "standard_deviation": statistics.pstdev(numbers),
        "p50": percentile(numbers, 50),
        "p90": percentile(numbers, 90),
        "p99": percentile(numbers, 99),
        "maximum": max(numbers),
        "integer_histogram": dict(sorted(histogram.items(), key=lambda row: int(row[0]))),
    }


def _static_observation(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": OBSERVATION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "job_id": str(job["job_id"]),
        "measurement_role": str(job["measurement_role"]),
        "ranking_eligible": job["ranking_eligible"] is True,
        "repeat": int(job["repeat"]),
        "execution_wave_index": int(job["execution_wave_index"]),
        "split_role": str(job["split_role"]),
        "split_unit_id": str(job["split_unit_id"]),
        "workload_id": str(job["workload_id"]),
        "shape_role": str(job["shape_role"]),
        "source_dataset_id": str(job["source_dataset_id"]),
        "runtime_dataset_id": str(job["dataset_id"]),
        "ranking_group_id": str(job["ranking_group_id"]),
        "fixed_cutoff_mechanism_group_id": str(
            job["fixed_cutoff_mechanism_group_id"]
        ),
        "candidate": {
            "model_id": str(job["model_id"]),
            "train_type": str(job["train_type"]),
            "gpu_type": str(job["gpu_type"]),
            "gpu_count": int(job["gpu_count"]),
            "cutoff_len": int(job["cutoff_len"]),
            "zero_stage": int(job["zero_stage"]),
            "gradient_checkpointing": bool(job["gc"]),
            "packing": True,
            "physical_mbs": 1,
            "virtual_mbs": float(job["virtual_mbs"]),
            "gradient_accumulation_steps": int(
                job["gradient_accumulation_steps"]
            ),
            "target_gbs": int(job["target_gbs"]),
            "expected_sample_gbs": float(job["expected_sample_gbs"]),
            "expected_sample_gbs_relative_error": float(
                job["expected_sample_gbs_relative_error"]
            ),
        },
        "upload_time_packing_features": job["upload_time_packing_features"],
        "model_input_projection": job["model_input_projection"],
        "bindings": {
            "data_path": str(job["data_path"]),
            "data_sha256": str(job["data_sha256"]),
            "dataset_profile_path": str(job["dataset_profile_path"]),
            "dataset_profile_sha256": str(job["dataset_profile_sha256"]),
            "dataset_registry_sha256": str(job["dataset_registry_sha256"]),
        },
        "outcome": {
            "classification": "missing",
            "calibration_eligible": False,
            "complete": False,
        },
    }


def collect_job(job: Mapping[str, Any]) -> dict[str, Any]:
    row = _static_observation(job)
    resolved = _latest_attempt_root(str(job["job_id"]))
    if resolved is None:
        return row
    latest, attempt = resolved
    status_path = attempt / "status.json"
    if not status_path.is_file():
        status_path = RESULTS_DIR / str(job["job_id"]) / "status.json"
    if not status_path.is_file():
        return row
    status = read_json(status_path)
    outcome: dict[str, Any] = {
        "classification": str(status.get("classification") or "unknown"),
        "calibration_eligible": status.get("calibration_eligible") is True,
        "complete": latest.get("state") == "complete",
        "execution_attempt_id": str(latest.get("execution_attempt_id") or ""),
        "gpu_mask": status.get("gpu_mask"),
        "return_code": status.get("return_code"),
        "wall_seconds": status.get("wall_seconds"),
        "execution_fingerprint_sha256": status.get(
            "execution_fingerprint_sha256"
        ),
        "runtime_fingerprint_sha256": status.get("runtime_fingerprint_sha256"),
        "thermal_observation": status.get("thermal_observation"),
    }
    row["outcome"] = outcome
    if status.get("classification") != "success":
        return row

    metrics_dir = attempt / "metrics"
    summaries = [
        read_json(path) for path in sorted(metrics_dir.glob("summary.rank*.json"))
    ]
    gpu_count = int(job["gpu_count"])
    ranks_exact = (
        len(summaries) == gpu_count
        and {int(summary.get("rank", -1)) for summary in summaries}
        == set(range(gpu_count))
        and all(int(summary.get("world_size", 0)) == gpu_count for summary in summaries)
        and all(int(summary.get("measured_steps", 0)) == MEASURE_STEPS for summary in summaries)
    )
    authoritative = bool(summaries) and all(
        (summary.get("token_ledger_evidence") or {}).get("authoritative") is True
        and (summary.get("batch_shape_evidence") or {}).get("authoritative") is True
        for summary in summaries
    )
    packing_semantics = bool(summaries) and all(
        ((summary.get("runtime_batch_evidence") or {}).get("packing") or {}).get(
            "semantic_checks_passed"
        )
        is True
        for summary in summaries
    )
    measured_seconds = max(
        (float(summary.get("measured_seconds") or 0.0) for summary in summaries),
        default=0.0,
    )
    totals: dict[str, float] = {}
    for name in (
        "logical_samples",
        "effective_tokens",
        "computed_tokens",
        "label_tokens",
        "computed_attention_token_pairs",
        "effective_attention_token_pairs",
        "physical_batches",
    ):
        totals[name] = sum(
            float((summary.get("measured_totals") or {}).get(name) or 0.0)
            for summary in summaries
        )
    block_tps = _block_throughputs(summaries, metrics_dir) if ranks_exact else []
    actual_pack_counts = [
        float(batch["logical_sample_count"])
        for summary in summaries
        for batch in (
            (summary.get("batch_shape_evidence") or {}).get(
                "measured_microbatches"
            )
            or []
        )
    ]
    throughput = {
        "measured_seconds_conservative": measured_seconds,
        "measured_totals_global": totals,
        "global_logical_samples_per_second": (
            totals["logical_samples"] / measured_seconds if measured_seconds else None
        ),
        "global_effective_tokens_per_second": (
            totals["effective_tokens"] / measured_seconds if measured_seconds else None
        ),
        "global_computed_tokens_per_second": (
            totals["computed_tokens"] / measured_seconds if measured_seconds else None
        ),
        "global_label_tokens_per_second": (
            totals["label_tokens"] / measured_seconds if measured_seconds else None
        ),
        "block_logical_samples_per_second": block_tps,
        "block_cv": (
            statistics.pstdev(block_tps) / statistics.fmean(block_tps)
            if block_tps and statistics.fmean(block_tps) > 0
            else None
        ),
    }
    memory = {
        "max_allocated_bytes_across_ranks": max(
            (int(summary.get("max_allocated") or 0) for summary in summaries),
            default=0,
        ),
        "max_reserved_bytes_across_ranks": max(
            (int(summary.get("max_reserved") or 0) for summary in summaries),
            default=0,
        ),
        "per_rank": [
            {
                "rank": int(summary["rank"]),
                "max_allocated_bytes": int(summary.get("max_allocated") or 0),
                "max_reserved_bytes": int(summary.get("max_reserved") or 0),
            }
            for summary in summaries
        ],
    }
    static_mean = float(job["virtual_mbs"])
    actual_distribution = _distribution(actual_pack_counts)
    checks = {
        "rank_world_and_steps_exact": ranks_exact,
        "authoritative_token_and_batch_ledgers": authoritative,
        "packing_semantics_passed": packing_semantics,
        "positive_primary_throughput": (
            throughput["global_logical_samples_per_second"] is not None
            and throughput["global_logical_samples_per_second"] > 0
        ),
        "three_complete_four_step_blocks": len(block_tps) == 3,
    }
    outcome.update(
        {
            "runtime_checks": checks,
            "usable_for_modeling": (
                outcome["calibration_eligible"] and all(checks.values())
            ),
            "throughput": throughput,
            "memory": memory,
            "actual_measured_samples_per_pack": actual_distribution,
            "actual_to_static_virtual_mbs_ratio": (
                actual_distribution["mean"] / static_mean
                if actual_distribution is not None
                else None
            ),
        }
    )
    return row


def _load_predictions(path: Path, jobs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if payload.get("schema") != PREDICTION_SCHEMA:
        raise ValueError(f"predictions must use {PREDICTION_SCHEMA}")
    queue = payload.get("queue_binding") or {}
    if queue.get("path") != str(QUEUE_HOLDOUT.resolve()) or queue.get("sha256") != sha256_file(QUEUE_HOLDOUT):
        raise ValueError("frozen predictions do not bind the exact holdout queue")
    if payload.get("training_outcomes_read") is not False:
        raise ValueError("holdout prediction artifact must state training_outcomes_read=false")
    rows = payload.get("predictions")
    if not isinstance(rows, list):
        raise ValueError("predictions must be a list")
    expected = {str(job["job_id"]) for job in jobs if job["split_role"] == "prospective_holdout"}
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("every prediction must be an object")
        job_id = str(row.get("job_id") or "")
        score = row.get("predicted_ranking_score")
        if job_id in by_id or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError(f"invalid or duplicate prediction for {job_id}")
        if not isinstance(row.get("admitted"), bool):
            raise ValueError(f"prediction {job_id} lacks boolean admitted")
        by_id[job_id] = dict(row)
    if set(by_id) != expected:
        raise ValueError("predictions must cover every holdout job exactly once")
    return by_id


def _ranking_metrics(
    observations: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    *,
    group_field: str,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        if (
            row["split_role"] == "prospective_holdout"
            and row.get("ranking_eligible") is True
        ):
            groups[str(row[group_field])].append(row)

    pair_correct = pair_total = 0
    material_correct = material_total = 0
    exact_hits = hit90 = selection_oom = 0
    regrets: list[float] = []
    details: list[dict[str, Any]] = []
    dataset_group_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group_id, candidates in sorted(groups.items()):
        successful = [
            row
            for row in candidates
            if row["outcome"].get("usable_for_modeling") is True
            and (row["outcome"].get("throughput") or {}).get(
                "global_logical_samples_per_second"
            )
            is not None
        ]
        if len(successful) < 2:
            continue
        current_correct = current_total = 0
        current_material_correct = current_material_total = 0
        for left, right in itertools.combinations(successful, 2):
            left_truth = float(
                left["outcome"]["throughput"]["global_logical_samples_per_second"]
            )
            right_truth = float(
                right["outcome"]["throughput"]["global_logical_samples_per_second"]
            )
            if math.isclose(left_truth, right_truth, rel_tol=0.0, abs_tol=1.0e-12):
                continue
            left_score = float(predictions[str(left["job_id"])]["predicted_ranking_score"])
            right_score = float(predictions[str(right["job_id"])]["predicted_ranking_score"])
            correct = (left_truth > right_truth) == (left_score > right_score)
            current_correct += int(correct)
            current_total += 1
            gap = abs(left_truth - right_truth) / max(left_truth, right_truth)
            if gap >= MATERIAL_GAP:
                current_material_correct += int(correct)
                current_material_total += 1
        admitted = [
            row for row in candidates if predictions[str(row["job_id"])]["admitted"]
        ]
        selected = (
            max(
                admitted,
                key=lambda row: float(
                    predictions[str(row["job_id"])]["predicted_ranking_score"]
                ),
            )
            if admitted
            else None
        )
        oracle = max(
            successful,
            key=lambda row: float(
                row["outcome"]["throughput"]["global_logical_samples_per_second"]
            ),
        )
        oracle_tps = float(
            oracle["outcome"]["throughput"]["global_logical_samples_per_second"]
        )
        selected_success = selected is not None and selected["outcome"].get("usable_for_modeling") is True
        selected_tps = (
            float(
                selected["outcome"]["throughput"][
                    "global_logical_samples_per_second"
                ]
            )
            if selected_success
            else 0.0
        )
        exact = bool(selected_success and selected["job_id"] == oracle["job_id"])
        within90 = selected_tps >= 0.90 * oracle_tps
        regret = max(0.0, 1.0 - selected_tps / oracle_tps)
        selected_classification = (
            str(selected["outcome"]["classification"])
            if selected is not None
            else "no_admitted_candidate"
        )
        is_oom = selected_classification == "oom"
        detail = {
            "group_id": group_id,
            "source_dataset_id": str(oracle["source_dataset_id"]),
            "candidate_count": len(candidates),
            "successful_candidate_count": len(successful),
            "pairwise_correct": current_correct,
            "pairwise_total": current_total,
            "material_pairwise_correct": current_material_correct,
            "material_pairwise_total": current_material_total,
            "predicted_top1_job_id": selected["job_id"] if selected else None,
            "predicted_top1_classification": selected_classification,
            "observed_top1_job_id": oracle["job_id"],
            "exact_top1": exact,
            "hit90": within90,
            "top1_regret": regret,
        }
        details.append(detail)
        dataset_group_metrics[str(oracle["source_dataset_id"])].append(detail)
        pair_correct += current_correct
        pair_total += current_total
        material_correct += current_material_correct
        material_total += current_material_total
        exact_hits += int(exact)
        hit90 += int(within90)
        selection_oom += int(is_oom)
        regrets.append(regret)

    scenarios = len(details)
    by_dataset = {}
    for dataset_id, rows in sorted(dataset_group_metrics.items()):
        by_dataset[dataset_id] = {
            "groups": len(rows),
            "exact_top1_accuracy": statistics.fmean(
                float(row["exact_top1"]) for row in rows
            ),
            "hit90_accuracy": statistics.fmean(float(row["hit90"]) for row in rows),
            "mean_top1_regret": statistics.fmean(
                float(row["top1_regret"]) for row in rows
            ),
        }
    return {
        "group_field": group_field,
        "scenario_count": scenarios,
        "pairwise_correct": pair_correct,
        "pairwise_total": pair_total,
        "pairwise_accuracy": pair_correct / pair_total if pair_total else None,
        "material_gap": MATERIAL_GAP,
        "material_pairwise_correct": material_correct,
        "material_pairwise_total": material_total,
        "material_pairwise_accuracy": (
            material_correct / material_total if material_total else None
        ),
        "exact_top1_hits": exact_hits,
        "exact_top1_accuracy": exact_hits / scenarios if scenarios else None,
        "hit90_hits": hit90,
        "hit90_accuracy": hit90 / scenarios if scenarios else None,
        "selected_oom_count": selection_oom,
        "mean_top1_regret": statistics.fmean(regrets) if regrets else None,
        "worst_top1_regret": max(regrets) if regrets else None,
        "dataset_macro_exact_top1_accuracy": (
            statistics.fmean(
                row["exact_top1_accuracy"] for row in by_dataset.values()
            )
            if by_dataset
            else None
        ),
        "dataset_macro_hit90_accuracy": (
            statistics.fmean(row["hit90_accuracy"] for row in by_dataset.values())
            if by_dataset
            else None
        ),
        "by_dataset": by_dataset,
        "details": details,
    }


def evaluate(predictions_path: Path | None = None) -> dict[str, Any]:
    jobs = read_jsonl(QUEUE_ALL)
    if len(jobs) != EXPECTED_JOBS or any(
        row.get("campaign_id") != CAMPAIGN_ID for row in jobs
    ):
        raise ValueError("all-jobs queue drifted")
    observations = [collect_job(job) for job in jobs]
    write_jsonl(OBSERVATIONS, observations)
    classifications = Counter(
        str(row["outcome"]["classification"]) for row in observations
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completion": {
            "jobs": len(observations),
            "classifications": dict(sorted(classifications.items())),
            "modeling_usable": sum(
                row["outcome"].get("usable_for_modeling") is True
                for row in observations
            ),
            "fit_modeling_usable": sum(
                row["split_role"] == "fit"
                and row["outcome"].get("usable_for_modeling") is True
                for row in observations
            ),
            "holdout_modeling_usable": sum(
                row["split_role"] == "prospective_holdout"
                and row["outcome"].get("usable_for_modeling") is True
                for row in observations
            ),
        },
        "observations": {
            "path": str(OBSERVATIONS.resolve()),
            "sha256": sha256_file(OBSERVATIONS),
            "schema": OBSERVATION_SCHEMA,
        },
        "ranking": None,
    }
    path = predictions_path
    if path is None and HOLDOUT_PREDICTIONS.is_file():
        path = HOLDOUT_PREDICTIONS
    if path is not None:
        predictions = _load_predictions(path, jobs)
        report["ranking"] = {
            "prediction_artifact": {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            },
            "primary_fixed_gpu_dataset": _ranking_metrics(
                observations, predictions, group_field="ranking_group_id"
            ),
            "diagnostic_fixed_gpu_dataset_cutoff": _ranking_metrics(
                observations,
                predictions,
                group_field="fixed_cutoff_mechanism_group_id",
            ),
        }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate(args.predictions)
    print(
        json.dumps(
            {
                "output": str(OUTPUT.resolve()),
                "observations": report["observations"],
                "completion": report["completion"],
                "ranking": report["ranking"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
