#!/usr/bin/env python3
"""Validate the frozen throughput predictor on the generalization campaign.

The first invocation seals predictions for every queued job in an immutable
ledger.  Later invocations reuse those exact predictions and only attach newly
available outcomes.  This separates predictor accuracy from any later refit.

Throughput metrics are computed only for terminal successful runs.  OOM runs
are reported, but never assigned a fabricated throughput.  Ranking metrics are
therefore explicitly conditional on a memory-safety filter having removed OOM
candidates first.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any

from common import (
    ROOT,
    percentile,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)
from throughput_predictor import ThroughputPredictor


SCHEMA_LEDGER = "sft_throughput_prediction_ledger/v1"
SCHEMA_VALIDATION = "sft_throughput_generalization_validation/v1"
IMPLEMENTATION_VERSION = (
    "sft_throughput_generalization_validator/2026-07-28.v1"
)

DEFAULT_QUEUE = (
    ROOT / "runtime" / "pipeline" / "pending-generalization-v1.jsonl"
)
DEFAULT_LEDGER = (
    ROOT
    / "artifacts"
    / "throughput_predictor_generalization_ledger_2026-07-28.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "throughput_predictor_generalization_validation_2026-07-28.json"
)
EXECUTION_ROOTS = (
    ROOT,
    ROOT.with_name("offline_experiments_genrun"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unix_to_iso(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _iso_to_unix(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _seal(payload: dict[str, Any], field: str) -> dict[str, Any]:
    if field in payload:
        raise ValueError(f"{field} must be absent before sealing")
    result = dict(payload)
    result[field] = sha256_json(result)
    return result


def _validate_seal(payload: Mapping[str, Any], field: str) -> None:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    expected = sha256_json(unsigned)
    if observed != expected:
        raise ValueError(
            f"Invalid {field}: expected {expected}, observed {observed}"
        )


def _model_roles(
    predictor: ThroughputPredictor,
) -> dict[str, dict[str, Any]]:
    roles: dict[str, dict[str, Any]] = {}
    for model_id, model in predictor.models.items():
        roles[model_id] = {
            "family": predictor._model_family(model),
            "generalization_role": model.get("generalization_role"),
            "calibration_role": model.get("calibration_role"),
            "actual_parameters": model.get("actual_parameters"),
        }
    return roles


def _prediction_request(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **job,
        "request_id": str(job["job_id"]),
        "hardware_id": "h800",
        "dtype": "bf16",
    }


def create_ledger(
    *,
    queue_path: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    if ledger_path.exists():
        ledger = read_json(ledger_path)
        _validate_seal(ledger, "ledger_sha256")
        return ledger

    predictor = ThroughputPredictor()
    jobs = _read_jsonl(queue_path)
    request_ids = [str(job["job_id"]) for job in jobs]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Generalization queue contains duplicate job ids")

    prediction_report = predictor.predict_many(
        [_prediction_request(job) for job in jobs],
        explain=False,
    )
    predictions = {
        str(row["request_id"]): row
        for row in prediction_report["predictions"]
    }
    if set(predictions) != set(request_ids):
        raise ValueError("Prediction ledger does not cover the exact queue")

    ledger = _seal(
        {
            "schema": SCHEMA_LEDGER,
            "implementation_version": IMPLEMENTATION_VERSION,
            "generated_at_utc": _utc_now(),
            "purpose": (
                "Freeze predictions before the remaining generalization "
                "jobs complete; never refit from these outcomes."
            ),
            "queue_binding": {
                "path": str(queue_path.resolve()),
                "sha256": sha256_file(queue_path),
                "jobs": len(jobs),
            },
            "frozen_model_binding": {
                "path": str(
                    predictor.model_artifact_path.resolve()
                ),
                "sha256": sha256_file(
                    predictor.model_artifact_path
                ),
                "generated_at_utc": predictor.report[
                    "generated_at_utc"
                ],
                "fit_model_ids": sorted(
                    predictor.frozen_model["training_support"][
                        "model_ids"
                    ]
                ),
            },
            "predictor_binding": {
                "path": str(
                    Path(__file__).with_name(
                        "throughput_predictor.py"
                    ).resolve()
                ),
                "sha256": prediction_report["predictor_binding"][
                    "sha256"
                ],
                "prediction_report_sha256": prediction_report[
                    "report_sha256"
                ],
            },
            "safety_contract": {
                "gpu_experiments_launched": False,
                "queues_mutated": False,
                "memory_safety_checked": False,
                "requires_memory_filter_before_ranking": True,
            },
            "model_roles": _model_roles(predictor),
            "jobs": [
                {
                    "job": job,
                    "job_payload_sha256": sha256_json(job),
                    "prediction": predictions[str(job["job_id"])],
                }
                for job in jobs
            ],
        },
        "ledger_sha256",
    )
    write_json(ledger_path, ledger)
    return ledger


def _latest_state(
    job: Mapping[str, Any],
) -> tuple[str, dict[str, Any] | None, Path | None]:
    result_roots = [
        root / "results" / str(job["job_id"])
        for root in EXECUTION_ROOTS
    ]
    latest_candidates = [
        result_root / "latest_attempt.json"
        for result_root in result_roots
        if (result_root / "latest_attempt.json").is_file()
    ]
    if latest_candidates:
        latest_path = max(
            latest_candidates,
            key=lambda path: path.stat().st_mtime,
        )
        latest = read_json(latest_path)
        attempt_path = (
            latest_path.parent / str(latest["attempt_path"])
        )
        if str(latest.get("state")) != "complete":
            return "in_progress_or_incomplete", latest, attempt_path
        classification = str(
            latest.get("classification") or "unknown"
        )
        return classification, latest, attempt_path

    attempts = []
    for result_root in result_roots:
        attempts_root = result_root / "attempts"
        if attempts_root.is_dir():
            attempts.extend(
                path
                for path in attempts_root.iterdir()
                if path.is_dir()
            )
    if attempts:
        newest = max(attempts, key=lambda path: path.stat().st_mtime)
        return "in_progress_or_incomplete", None, newest
    return "not_started", None, None


def _aggregate_success(
    *,
    job: Mapping[str, Any],
    latest: Mapping[str, Any],
    attempt_path: Path,
    prediction: Mapping[str, Any],
    model_roles: Mapping[str, Mapping[str, Any]],
    frozen_generated_unix: float,
    ledger_generated_unix: float,
) -> dict[str, Any]:
    status = read_json(attempt_path / "status.json")
    summary_paths = sorted(
        (attempt_path / "metrics").glob("summary.rank*.json")
    )
    expected_ranks = int(job["gpu_count"])
    if len(summary_paths) != expected_ranks:
        raise ValueError(
            f"{job['job_id']} has {len(summary_paths)} summaries; "
            f"expected {expected_ranks}"
        )
    summaries = [read_json(path) for path in summary_paths]
    attempt_ids = {
        str(summary.get("execution_attempt_id"))
        for summary in summaries
    }
    if attempt_ids != {str(latest["execution_attempt_id"])}:
        raise ValueError(
            f"{job['job_id']} rank summaries have mixed attempt ids"
        )

    measured_steps_set = {
        int(summary["measured_steps"]) for summary in summaries
    }
    if len(measured_steps_set) != 1:
        raise ValueError(
            f"{job['job_id']} ranks have different measured steps"
        )
    measured_steps = measured_steps_set.pop()
    measured_seconds = max(
        float(summary["measured_seconds"]) for summary in summaries
    )
    effective_tokens = sum(
        int(summary["measured_totals"]["effective_tokens"])
        for summary in summaries
    )
    computed_tokens = sum(
        int(summary["measured_totals"]["computed_tokens"])
        for summary in summaries
    )
    logical_samples = sum(
        int(summary["measured_totals"]["logical_samples"])
        for summary in summaries
    )
    observed_rate = effective_tokens / measured_seconds
    observed_step_seconds = measured_seconds / measured_steps
    observed_work_per_step = effective_tokens / measured_steps
    predicted_rate = float(
        prediction["predicted_effective_tokens_per_second"]
    )
    predicted_step_seconds = float(
        prediction["predicted_step_seconds"]
    )
    predicted_work_per_step = float(
        prediction["work_per_step"]["effective_tokens"]
    )
    rate_error = 100.0 * (
        predicted_rate - observed_rate
    ) / observed_rate
    step_error = 100.0 * (
        predicted_step_seconds - observed_step_seconds
    ) / observed_step_seconds
    work_error = 100.0 * (
        predicted_work_per_step - observed_work_per_step
    ) / observed_work_per_step
    finished_unix = float(status["finished_unix"])
    model_id = str(job["model_id"])
    role = model_roles.get(model_id) or {}
    declared_holdout = any(
        "holdout" in str(role.get(field) or "").lower()
        for field in ("generalization_role", "calibration_role")
    )
    confidence = prediction["confidence"]

    return {
        "job_id": str(job["job_id"]),
        "execution_attempt_id": str(latest["execution_attempt_id"]),
        "finished_unix": finished_unix,
        "finished_at_utc": _unix_to_iso(finished_unix),
        "finished_after_frozen_model": (
            finished_unix > frozen_generated_unix
        ),
        "finished_after_prediction_ledger": (
            finished_unix > ledger_generated_unix
        ),
        "declared_model_holdout": declared_holdout,
        "configuration": {
            "model_id": model_id,
            "model_family": role.get("family"),
            "train_type": str(job["train_type"]),
            "dataset_id": str(job["dataset_id"]),
            "gpu_count": int(job["gpu_count"]),
            "target_gbs": int(job["target_gbs"]),
            "cutoff_len": int(job["cutoff_len"]),
            "mbs": int(job["mbs"]),
            "gradient_checkpointing": bool(job["gc"]),
            "zero_stage": str(job["zero"]),
            "packing": bool(job["packing"]),
            "repeat": int(job["repeat"]),
            "kernel_path": prediction["configuration"][
                "kernel_path"
            ],
        },
        "observed": {
            "measured_steps": measured_steps,
            "measured_seconds": measured_seconds,
            "effective_tokens": effective_tokens,
            "computed_tokens": computed_tokens,
            "logical_samples": logical_samples,
            "effective_tokens_per_second": observed_rate,
            "step_seconds": observed_step_seconds,
            "effective_tokens_per_step": observed_work_per_step,
        },
        "predicted": {
            "effective_tokens_per_second": predicted_rate,
            "step_seconds": predicted_step_seconds,
            "effective_tokens_per_step": predicted_work_per_step,
        },
        "error": {
            "throughput_signed_percentage": rate_error,
            "throughput_absolute_percentage": abs(rate_error),
            "step_time_signed_percentage": step_error,
            "step_time_absolute_percentage": abs(step_error),
            "static_work_signed_percentage": work_error,
            "static_work_absolute_percentage": abs(work_error),
            "predicted_to_observed_throughput_ratio": (
                predicted_rate / observed_rate
            ),
        },
        "confidence": {
            "label": confidence["label"],
            "inside_frozen_support": confidence[
                "inside_frozen_support"
            ],
            "reason_codes": [
                str(reason["code"])
                for reason in confidence["reasons"]
            ],
        },
        "evidence": {
            "attempt_path": str(attempt_path.resolve()),
            "status_sha256": sha256_file(
                attempt_path / "status.json"
            ),
            "summary_sha256": {
                path.name: sha256_file(path)
                for path in summary_paths
            },
            "calibration_eligible": latest.get(
                "calibration_eligible"
            ),
            "execution_fingerprint_quality": status.get(
                "execution_fingerprint_quality"
            ),
            "thermal_observation": status.get(
                "thermal_observation"
            ),
        },
    }


def _absolute_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {
            "successful_runs": 0,
            "status": "insufficient_data",
        }
    rate_apes = [
        float(row["error"]["throughput_absolute_percentage"])
        for row in rows
    ]
    rate_bias = [
        float(row["error"]["throughput_signed_percentage"])
        for row in rows
    ]
    step_apes = [
        float(row["error"]["step_time_absolute_percentage"])
        for row in rows
    ]
    work_apes = [
        float(row["error"]["static_work_absolute_percentage"])
        for row in rows
    ]
    log_errors = [
        math.log(
            float(row["predicted"][
                "effective_tokens_per_second"
            ])
            / float(row["observed"][
                "effective_tokens_per_second"
            ])
        )
        for row in rows
    ]
    return {
        "successful_runs": len(rows),
        "status": "evaluated",
        "throughput_mape_percent": statistics.fmean(rate_apes),
        "throughput_median_ape_percent": statistics.median(
            rate_apes
        ),
        "throughput_p90_ape_percent": percentile(rate_apes, 90),
        "throughput_mean_signed_error_percent": statistics.fmean(
            rate_bias
        ),
        "throughput_log_rmse": math.sqrt(
            statistics.fmean(value * value for value in log_errors)
        ),
        "within_10_percent_fraction": statistics.fmean(
            float(value <= 10.0) for value in rate_apes
        ),
        "within_20_percent_fraction": statistics.fmean(
            float(value <= 20.0) for value in rate_apes
        ),
        "within_30_percent_fraction": statistics.fmean(
            float(value <= 30.0) for value in rate_apes
        ),
        "step_time_mape_percent": statistics.fmean(step_apes),
        "static_work_mape_percent": statistics.fmean(work_apes),
    }


def _scenario_key(row: Mapping[str, Any]) -> str:
    config = row["configuration"]
    return json.dumps(
        {
            "hardware_id": "h800",
            "model_id": config["model_id"],
            "dataset_id": config["dataset_id"],
            "train_type": config["train_type"],
            "target_gbs": config["target_gbs"],
            "cutoff_len": config["cutoff_len"],
            "packing": config["packing"],
            "dtype": "bf16",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _candidate_key(row: Mapping[str, Any]) -> str:
    config = row["configuration"]
    return json.dumps(
        {
            "gpu_count": config["gpu_count"],
            "mbs": config["mbs"],
            "gradient_checkpointing": config[
                "gradient_checkpointing"
            ],
            "zero_stage": config["zero_stage"],
            "kernel_path": config["kernel_path"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _ranking_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        by_scenario[_scenario_key(row)].append(row)

    scenarios = []
    pooled_pairs = 0
    pooled_correct = 0.0
    for scenario_key, scenario_rows in sorted(
        by_scenario.items()
    ):
        by_candidate: dict[
            str, list[Mapping[str, Any]]
        ] = defaultdict(list)
        for row in scenario_rows:
            by_candidate[_candidate_key(row)].append(row)

        candidates = []
        for candidate_key, repeat_rows in sorted(
            by_candidate.items()
        ):
            predicted_values = [
                float(row["predicted"][
                    "effective_tokens_per_second"
                ])
                for row in repeat_rows
            ]
            if (
                max(predicted_values) - min(predicted_values)
                > 1.0e-8 * max(predicted_values)
            ):
                raise ValueError(
                    "Frozen predictions changed across exact repeats"
                )
            candidates.append(
                {
                    "candidate": json.loads(candidate_key),
                    "repeats": len(repeat_rows),
                    "job_ids": sorted(
                        str(row["job_id"]) for row in repeat_rows
                    ),
                    "observed_effective_tokens_per_second": (
                        statistics.fmean(
                            float(
                                row["observed"][
                                    "effective_tokens_per_second"
                                ]
                            )
                            for row in repeat_rows
                        )
                    ),
                    "predicted_effective_tokens_per_second": (
                        statistics.fmean(predicted_values)
                    ),
                }
            )

        pair_count = 0
        pair_correct = 0.0
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                observed_delta = (
                    left["observed_effective_tokens_per_second"]
                    - right["observed_effective_tokens_per_second"]
                )
                predicted_delta = (
                    left["predicted_effective_tokens_per_second"]
                    - right["predicted_effective_tokens_per_second"]
                )
                if abs(observed_delta) <= 1.0e-12:
                    continue
                pair_count += 1
                if abs(predicted_delta) <= 1.0e-12:
                    pair_correct += 0.5
                elif observed_delta * predicted_delta > 0:
                    pair_correct += 1.0

        scenario_material = json.loads(scenario_key)
        scenario: dict[str, Any] = {
            "scenario": scenario_material,
            "successful_runs": len(scenario_rows),
            "unique_successful_configs": len(candidates),
            "conditional_on_memory_safe_successes": True,
            "candidates": sorted(
                candidates,
                key=lambda row: row[
                    "observed_effective_tokens_per_second"
                ],
                reverse=True,
            ),
        }
        if len(candidates) < 2 or pair_count == 0:
            scenario.update(
                {
                    "status": "insufficient_configs_for_ranking",
                    "pairwise_comparisons": pair_count,
                    "pairwise_accuracy_percent": None,
                    "top1_regret_percent": None,
                }
            )
        else:
            observed_best = max(
                candidates,
                key=lambda row: row[
                    "observed_effective_tokens_per_second"
                ],
            )
            predicted_best = max(
                candidates,
                key=lambda row: row[
                    "predicted_effective_tokens_per_second"
                ],
            )
            regret = 100.0 * (
                observed_best[
                    "observed_effective_tokens_per_second"
                ]
                - predicted_best[
                    "observed_effective_tokens_per_second"
                ]
            ) / observed_best[
                "observed_effective_tokens_per_second"
            ]
            scenario.update(
                {
                    "status": "evaluated",
                    "pairwise_comparisons": pair_count,
                    "pairwise_correct": pair_correct,
                    "pairwise_accuracy_percent": (
                        100.0 * pair_correct / pair_count
                    ),
                    "observed_best": observed_best,
                    "predictor_selected": predicted_best,
                    "top1_regret_percent": regret,
                }
            )
            pooled_pairs += pair_count
            pooled_correct += pair_correct
        scenarios.append(scenario)

    evaluated = [
        scenario
        for scenario in scenarios
        if scenario["status"] == "evaluated"
    ]
    return {
        "successful_runs": len(rows),
        "unique_successful_configs": sum(
            int(scenario["unique_successful_configs"])
            for scenario in scenarios
        ),
        "scenarios_total": len(scenarios),
        "scenarios_evaluated": len(evaluated),
        "scenarios_insufficient": len(scenarios) - len(evaluated),
        "conditional_on_memory_filter": True,
        "pooled_pairwise_comparisons": pooled_pairs,
        "pooled_pairwise_accuracy_percent": (
            100.0 * pooled_correct / pooled_pairs
            if pooled_pairs
            else None
        ),
        "scenario_equal_pairwise_accuracy_percent": (
            statistics.fmean(
                float(scenario["pairwise_accuracy_percent"])
                for scenario in evaluated
            )
            if evaluated
            else None
        ),
        "mean_top1_regret_percent": (
            statistics.fmean(
                float(scenario["top1_regret_percent"])
                for scenario in evaluated
            )
            if evaluated
            else None
        ),
        "worst_top1_regret_percent": (
            max(
                float(scenario["top1_regret_percent"])
                for scenario in evaluated
            )
            if evaluated
            else None
        ),
        "scenarios": scenarios,
    }


def _repeat_stability(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        groups[(_scenario_key(row), _candidate_key(row))].append(
            row
        )
    repeated = []
    for (scenario_key, candidate_key), repeat_rows in sorted(
        groups.items()
    ):
        if len(repeat_rows) < 2:
            continue
        rates = [
            float(row["observed"][
                "effective_tokens_per_second"
            ])
            for row in repeat_rows
        ]
        mean_rate = statistics.fmean(rates)
        repeated.append(
            {
                "scenario": json.loads(scenario_key),
                "candidate": json.loads(candidate_key),
                "runs": len(rates),
                "mean_effective_tokens_per_second": mean_rate,
                "min_effective_tokens_per_second": min(rates),
                "max_effective_tokens_per_second": max(rates),
                "relative_range_percent": (
                    100.0 * (max(rates) - min(rates)) / mean_rate
                ),
            }
        )
    ranges = [
        float(row["relative_range_percent"]) for row in repeated
    ]
    return {
        "repeated_configs": len(repeated),
        "median_relative_range_percent": (
            statistics.median(ranges) if ranges else None
        ),
        "p90_relative_range_percent": (
            percentile(ranges, 90) if ranges else None
        ),
        "details": repeated,
    }


def _cohort_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "absolute": _absolute_metrics(rows),
        "ranking": _ranking_metrics(rows),
    }


def build_validation(
    *,
    ledger: Mapping[str, Any],
    ledger_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    _validate_seal(ledger, "ledger_sha256")
    queue_binding = ledger["queue_binding"]
    queue_path = Path(str(queue_binding["path"]))
    if sha256_file(queue_path) != queue_binding["sha256"]:
        raise ValueError(
            "Generalization queue changed after prediction sealing"
        )

    frozen_generated_unix = _iso_to_unix(
        str(ledger["frozen_model_binding"]["generated_at_utc"])
    )
    ledger_generated_unix = _iso_to_unix(
        str(ledger["generated_at_utc"])
    )
    trained_models = set(
        str(value)
        for value in ledger["frozen_model_binding"][
            "fit_model_ids"
        ]
    )
    roles = ledger["model_roles"]

    state_counts: dict[str, int] = defaultdict(int)
    successes: list[dict[str, Any]] = []
    excluded_terminal: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for item in ledger["jobs"]:
        job = item["job"]
        prediction = item["prediction"]
        state, latest, attempt_path = _latest_state(job)
        state_counts[state] += 1
        model_id = str(job["model_id"])
        if state == "success":
            if latest is None or attempt_path is None:
                raise ValueError(
                    f"{job['job_id']} success has no attempt evidence"
                )
            row = _aggregate_success(
                job=job,
                latest=latest,
                attempt_path=attempt_path,
                prediction=prediction,
                model_roles=roles,
                frozen_generated_unix=frozen_generated_unix,
                ledger_generated_unix=ledger_generated_unix,
            )
            row["model_was_in_frozen_fit"] = (
                model_id in trained_models
            )
            successes.append(row)
        elif state in {"not_started", "in_progress_or_incomplete"}:
            pending.append(
                {
                    "job_id": str(job["job_id"]),
                    "state": state,
                    "configuration": {
                        "model_id": model_id,
                        "train_type": job["train_type"],
                        "gpu_count": job["gpu_count"],
                        "mbs": job["mbs"],
                        "gradient_checkpointing": job["gc"],
                        "zero_stage": job["zero"],
                        "repeat": job["repeat"],
                    },
                    "sealed_predicted_effective_tokens_per_second": (
                        prediction[
                            "predicted_effective_tokens_per_second"
                        ]
                    ),
                    "sealed_confidence": prediction["confidence"][
                        "label"
                    ],
                }
            )
        else:
            excluded_terminal.append(
                {
                    "job_id": str(job["job_id"]),
                    "classification": state,
                    "configuration": {
                        "model_id": model_id,
                        "train_type": job["train_type"],
                        "gpu_count": job["gpu_count"],
                        "target_gbs": job["target_gbs"],
                        "mbs": job["mbs"],
                        "gradient_checkpointing": job["gc"],
                        "zero_stage": job["zero"],
                        "repeat": job["repeat"],
                    },
                    "sealed_predicted_effective_tokens_per_second": (
                        prediction[
                            "predicted_effective_tokens_per_second"
                        ]
                    ),
                    "throughput_metric_excluded_reason": (
                        "No successful throughput exists; apply the "
                        "memory-safety filter before throughput ranking."
                    ),
                }
            )

    evaluated = [
        row
        for row in successes
        if not row["model_was_in_frozen_fit"]
    ]
    accidentally_seen = [
        row["job_id"]
        for row in successes
        if row["model_was_in_frozen_fit"]
    ]
    strict_post_freeze = [
        row for row in evaluated if row["finished_after_frozen_model"]
    ]
    post_ledger = [
        row
        for row in evaluated
        if row["finished_after_prediction_ledger"]
    ]
    declared_holdout = [
        row for row in evaluated if row["declared_model_holdout"]
    ]

    cohorts: dict[str, Any] = {
        "all_unseen_model_successes": _cohort_report(evaluated),
        "strict_post_frozen_model": _cohort_report(
            strict_post_freeze
        ),
        "post_prediction_ledger": _cohort_report(post_ledger),
        "declared_model_holdout": _cohort_report(
            declared_holdout
        ),
    }
    by_model_mode: dict[
        tuple[str, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in evaluated:
        config = row["configuration"]
        by_model_mode[
            (str(config["model_id"]), str(config["train_type"]))
        ].append(row)
    cohorts["by_model_and_train_type"] = {
        f"{model_id}/{train_type}": _cohort_report(rows)
        for (model_id, train_type), rows in sorted(
            by_model_mode.items()
        )
    }

    confidence_counts: dict[str, int] = defaultdict(int)
    for row in evaluated:
        confidence_counts[str(row["confidence"]["label"])] += 1

    report = _seal(
        {
            "schema": SCHEMA_VALIDATION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "generated_at_utc": _utc_now(),
            "ledger_binding": {
                "path": str(ledger_path.resolve()),
                "file_sha256": sha256_file(ledger_path),
                "ledger_sha256": ledger["ledger_sha256"],
                "generated_at_utc": ledger[
                    "generated_at_utc"
                ],
            },
            "frozen_model_binding": ledger[
                "frozen_model_binding"
            ],
            "evaluation_contract": {
                "predictor_refit_from_evaluated_outcomes": False,
                "predictions_recomputed_after_ledger": False,
                "throughput_target": "effective_tokens_per_second",
                "rank_aggregation": (
                    "sum work counters across ranks divided by the "
                    "maximum measured seconds across ranks"
                ),
                "repeat_policy": (
                    "Absolute metrics use every successful run; "
                    "ranking averages repeats of identical configs."
                ),
                "ranking_is_conditional_on_memory_filter": True,
                "oom_throughput_is_never_imputed": True,
            },
            "queue_snapshot": {
                "jobs": len(ledger["jobs"]),
                "state_counts": dict(sorted(state_counts.items())),
                "successful_rows": len(successes),
                "terminal_excluded_rows": len(excluded_terminal),
                "pending_rows": len(pending),
            },
            "leakage_audit": {
                "frozen_fit_model_ids": sorted(trained_models),
                "evaluated_model_ids": sorted(
                    {
                        str(row["configuration"]["model_id"])
                        for row in evaluated
                    }
                ),
                "evaluated_models_disjoint_from_frozen_fit": (
                    not accidentally_seen
                ),
                "accidentally_seen_success_job_ids": accidentally_seen,
                "strict_post_frozen_model_successes": len(
                    strict_post_freeze
                ),
                "post_prediction_ledger_successes": len(post_ledger),
                "note": (
                    "Post-frozen-model rows are temporal holdouts of "
                    "the model artifact. Only post-ledger rows are also "
                    "pre-registered before their outcomes existed."
                ),
            },
            "confidence_counts": dict(
                sorted(confidence_counts.items())
            ),
            "cohorts": cohorts,
            "repeat_stability": _repeat_stability(evaluated),
            "successful_evaluations": evaluated,
            "excluded_terminal_outcomes": excluded_terminal,
            "sealed_pending_predictions": pending,
            "interpretation_guardrails": [
                (
                    "A single post-freeze Qwen3-32B result can test "
                    "absolute error, but cannot test ranking."
                ),
                (
                    "Qwen3-VL is excluded from the frozen fit and is a "
                    "declared holdout, but its current outcomes predate "
                    "the prediction ledger; treat it as model-held-out "
                    "rather than a fully prospective test."
                ),
                (
                    "Ranking metrics include only successful configs. "
                    "The throughput predictor alone must never recommend "
                    "across OOM candidates."
                ),
            ],
        },
        "report_sha256",
    )
    write_json(output_path, report)
    return report


def _summary(report: Mapping[str, Any]) -> str:
    queue = report["queue_snapshot"]
    strict = report["cohorts"]["strict_post_frozen_model"][
        "absolute"
    ]
    holdout = report["cohorts"]["declared_model_holdout"]
    ranking = holdout["ranking"]
    lines = [
        f"queue={queue['jobs']} states={queue['state_counts']}",
        (
            "strict_post_freeze: "
            f"n={strict['successful_runs']} "
            f"MAPE={strict.get('throughput_mape_percent')}"
        ),
        (
            "declared_holdout: "
            f"n={holdout['absolute']['successful_runs']} "
            f"MAPE={holdout['absolute'].get('throughput_mape_percent')} "
            f"pairwise={ranking.get('pooled_pairwise_accuracy_percent')} "
            f"top1_regret={ranking.get('mean_top1_regret_percent')}"
        ),
        f"report_sha256={report['report_sha256']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ledger = create_ledger(
        queue_path=args.queue.resolve(),
        ledger_path=args.ledger.resolve(),
    )
    report = build_validation(
        ledger=ledger,
        ledger_path=args.ledger.resolve(),
        output_path=args.output.resolve(),
    )
    print(_summary(report))


if __name__ == "__main__":
    main()
