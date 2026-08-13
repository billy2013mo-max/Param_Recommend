#!/usr/bin/env python3
"""Replay frozen memory/throughput models on the 2026-07-29 Qwen holdout.

This is an inference-only audit.  The Qwen2.5/Qwen3.5 outcomes are labels;
none of the evaluated outcomes is used to refit a model.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np

import compare_memory_model_generations as memory_replay
from common import read_json, sha256_file, sha256_json, write_json
from compare_throughput_model_generations import (
    _cohort_breakdown,
    _evaluation_row,
    _joint_candidate,
    _metrics as throughput_metrics,
    _static_record,
)
from joint_throughput_modeling import (
    _predict_log_throughput,
    _predict_two_head_entries,
)
from throughput_predictor import ThroughputPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "offline_experiments"
QGEN = PROJECT_ROOT / "offline_experiments_qgen_20260729"
Q35 = PROJECT_ROOT / "offline_experiments_qwen35_tilelang_20260729"

DEFAULT_OUTPUT = (
    BASE
    / "artifacts"
    / "qwen25_qwen35_frozen_model_replay_2026-07-29.json"
)
NEW_INVENTORY = Q35 / "artifacts" / "model_inventory.json"
PREDICTOR_FREEZE = Q35 / "artifacts" / "predictor_freeze_before_holdout.json"

MEMORY_MODEL_IDS = (
    memory_replay.MODEL_11D,
    memory_replay.MODEL_NATIVE,
    memory_replay.MODEL_PHYSICAL,
)
THROUGHPUT_MODEL_IDS = (
    "v4_full_factor_single_head",
    "v4b_full_factor_two_head_rejected",
    "v5_structured_cross_card_single_head",
)
GIB = float(1024**3)

COHORTS = (
    {
        "cohort": "qwen25_memory_screen",
        "queue": QGEN / "matrix" / "queue_memory_qwen25.jsonl",
        "results": QGEN / "results",
        "memory_evidence": True,
        "throughput_role": "excluded_memory_probe",
    },
    {
        "cohort": "qwen25_formal_throughput",
        "queue": QGEN / "matrix" / "queue_formal_qwen25.jsonl",
        "results": QGEN / "results",
        "memory_evidence": True,
        "throughput_role": "primary_formal",
    },
    {
        "cohort": "qwen25_packing_abba",
        "queue": QGEN / "matrix" / "queue_packing_abba.jsonl",
        "results": QGEN / "results",
        "memory_evidence": True,
        "throughput_role": "packing_diagnostic",
    },
    {
        "cohort": "qwen35_tilelang_canary",
        "queue": Q35 / "matrix" / "queue_canary.jsonl",
        "results": Q35 / "results",
        "memory_evidence": True,
        "throughput_role": "canary_diagnostic",
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unix_to_iso(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _latest_terminal_attempt(
    results_root: Path,
    job_id: str,
) -> tuple[dict[str, Any], Path]:
    status_paths = sorted(
        (results_root / job_id / "attempts").glob("*/status.json")
    )
    if not status_paths:
        raise FileNotFoundError(f"No terminal status for {job_id}")
    candidates = [(read_json(path), path.parent) for path in status_paths]
    status, attempt = max(
        candidates,
        key=lambda item: (
            float(item[0].get("finished_unix") or 0.0),
            item[1].stat().st_mtime,
        ),
    )
    if str(status.get("job_id")) != job_id:
        raise ValueError(f"Status/job mismatch for {job_id}")
    return status, attempt


def _aggregate_success(
    job: Mapping[str, Any],
    status: Mapping[str, Any],
    attempt: Path,
) -> dict[str, Any]:
    summary_paths = sorted(
        (attempt / "metrics").glob("summary.rank*.json")
    )
    if len(summary_paths) != int(job["gpu_count"]):
        raise ValueError(
            f"{job['job_id']} has {len(summary_paths)} rank summaries; "
            f"expected {job['gpu_count']}"
        )
    summaries = [read_json(path) for path in summary_paths]
    measured_steps_set = {
        int(summary["measured_steps"]) for summary in summaries
    }
    if len(measured_steps_set) != 1:
        raise ValueError(f"{job['job_id']} rank step counts differ")
    measured_steps = measured_steps_set.pop()
    measured_seconds = max(
        float(summary["measured_seconds"]) for summary in summaries
    )
    totals = {
        key: sum(
            int(summary["measured_totals"][key])
            for summary in summaries
        )
        for key in (
            "effective_tokens",
            "computed_tokens",
            "logical_samples",
        )
    }
    return {
        "measured_steps": measured_steps,
        "measured_seconds": measured_seconds,
        **totals,
        "effective_tokens_per_second": (
            totals["effective_tokens"] / measured_seconds
        ),
        "step_seconds": measured_seconds / measured_steps,
        "effective_tokens_per_step": (
            totals["effective_tokens"] / measured_steps
        ),
        "max_allocated_bytes": max(
            float(summary["max_allocated"]) for summary in summaries
        ),
        "max_reserved_bytes": max(
            float(summary["max_reserved"]) for summary in summaries
        ),
        "finished_unix": float(status["finished_unix"]),
        "summary_paths": [str(path.resolve()) for path in summary_paths],
    }


def _collect_outcomes() -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in COHORTS:
        for queue_index, job in enumerate(_read_jsonl(spec["queue"])):
            job_id = str(job["job_id"])
            if job_id in seen:
                raise ValueError(f"Duplicate job id across queues: {job_id}")
            seen.add(job_id)
            status, attempt = _latest_terminal_attempt(
                Path(spec["results"]),
                job_id,
            )
            state = str(status["classification"])
            if state not in {"success", "oom"}:
                raise ValueError(
                    f"Unexpected terminal state for {job_id}: {state}"
                )
            observed = (
                _aggregate_success(job, status, attempt)
                if state == "success"
                else None
            )
            outcomes.append(
                {
                    "cohort": spec["cohort"],
                    "queue_path": str(Path(spec["queue"]).resolve()),
                    "queue_index": queue_index,
                    "memory_evidence": bool(spec["memory_evidence"]),
                    "throughput_role": spec["throughput_role"],
                    "job": dict(job),
                    "job_id": job_id,
                    "state": state,
                    "status": dict(status),
                    "attempt_path": str(attempt.resolve()),
                    "observed": observed,
                }
            )
    return outcomes


def _normalized_zero_for_memory(job: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(job)
    if str(result.get("zero") or "").lower() in {"", "none", "zero0"}:
        result["zero"] = "zero0"
    return result


def _subgroup_memory_metrics(
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    *,
    p95_head: bool,
    key_by_observation: Mapping[str, str],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[key_by_observation[str(record["observation_id"])]].append(
            record
        )
    return {
        key: memory_replay._metrics(
            rows,
            predictions,
            p95_head=p95_head,
            include_details=False,
        )
        for key, rows in sorted(groups.items())
    }


def _memory_replay(
    outcomes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    old_models, fixed_lora, hardware = memory_replay._inventory()
    new_inventory = read_json(NEW_INVENTORY)
    new_models = {
        str(model["id"]): dict(model)
        for model in new_inventory["models"]
    }
    all_models = {**old_models, **new_models}

    historical = read_json(memory_replay.DEFAULT_OUTPUT)
    legacy_reconstruction = historical["legacy_11d_reconstruction"]
    if int(legacy_reconstruction["fit_configurations"]) != 617:
        raise ValueError(
            "Legacy 11D frozen fit population drifted from 617 rows"
        )
    beta = np.asarray(
        legacy_reconstruction["coefficients"],
        dtype=float,
    )
    if len(beta) != len(memory_replay.RESOURCE_FEATURE_NAMES):
        raise ValueError("Frozen legacy 11D coefficient count drifted")
    old_training_summary = {
        key: legacy_reconstruction[key]
        for key in (
            "raw_result_rows",
            "aggregated_configurations",
            "fit_configurations",
            "fit_scenarios",
            "fit_model_ids",
        )
    }

    records: list[dict[str, Any]] = []
    outcome_by_job: dict[str, Mapping[str, Any]] = {}
    for outcome in outcomes:
        if not outcome["memory_evidence"]:
            continue
        job = _normalized_zero_for_memory(outcome["job"])
        observed = outcome["observed"]
        record = memory_replay._new_record(
            job=job,
            outcome=str(outcome["state"]),
            observed_allocated=(
                float(observed["max_allocated_bytes"])
                if observed is not None
                else None
            ),
            observed_reserved=(
                float(observed["max_reserved_bytes"])
                if observed is not None
                else None
            ),
            models=all_models,
            fixed_lora=fixed_lora,
            hardware=hardware,
        )
        record["holdout_cohort"] = str(outcome["cohort"])
        records.append(record)
        outcome_by_job[str(outcome["job_id"])] = outcome

    native = read_json(memory_replay.NATIVE_ARTIFACT)
    challenger = read_json(memory_replay.CHALLENGER_ARTIFACT)
    predictions = {
        memory_replay.MODEL_11D: (
            memory_replay._old_11d_frozen_predictions(
                records,
                beta=beta,
                models=all_models,
            )
        ),
        memory_replay.MODEL_NATIVE: memory_replay._native_predictions(
            records,
            artifact=native,
        ),
        memory_replay.MODEL_PHYSICAL: memory_replay._physical_predictions(
            records,
            artifact=challenger,
        ),
    }
    p95 = {
        memory_replay.MODEL_11D: False,
        memory_replay.MODEL_NATIVE: True,
        memory_replay.MODEL_PHYSICAL: True,
    }

    cohort_by_observation = {
        str(record["observation_id"]): str(record["holdout_cohort"])
        for record in records
    }
    family_by_observation = {
        str(record["observation_id"]): str(
            new_models[str(record["scenario"]["model_id"])]["family"]
        )
        for record in records
    }
    model_train_by_observation = {
        str(record["observation_id"]): (
            f"{record['scenario']['model_id']}/"
            f"{record['selector']['training_mode']}"
        )
        for record in records
    }

    model_results: dict[str, Any] = {}
    for model_id in MEMORY_MODEL_IDS:
        model_results[model_id] = {
            "overall": memory_replay._metrics(
                records,
                predictions[model_id],
                p95_head=p95[model_id],
                include_details=True,
            ),
            "by_cohort": _subgroup_memory_metrics(
                records,
                predictions[model_id],
                p95_head=p95[model_id],
                key_by_observation=cohort_by_observation,
            ),
            "by_model_family": _subgroup_memory_metrics(
                records,
                predictions[model_id],
                p95_head=p95[model_id],
                key_by_observation=family_by_observation,
            ),
            "by_model_and_train_type": _subgroup_memory_metrics(
                records,
                predictions[model_id],
                p95_head=p95[model_id],
                key_by_observation=model_train_by_observation,
            ),
        }

    physical_admission: dict[str, dict[str, Any]] = {}
    physical_predictions = predictions[memory_replay.MODEL_PHYSICAL]
    for record in records:
        job_id = str(record["observation_id"])
        prediction = physical_predictions[job_id]
        safe_limit = memory_replay._safe_limit(record)
        if safe_limit is None:
            raise ValueError(f"No safe limit for {job_id}")
        available = prediction.get("available") is True
        upper = (
            float(prediction["decision_upper_bytes"])
            if available
            else None
        )
        center = (
            float(prediction["reserved_center_bytes"])
            if available
            else None
        )
        admitted = bool(available and upper <= safe_limit)
        observed_reserved = memory_replay._observed_reserved(record)
        physical_admission[job_id] = {
            "job_id": job_id,
            "cohort": str(record["holdout_cohort"]),
            "model_id": str(record["scenario"]["model_id"]),
            "model_family": family_by_observation[job_id],
            "train_type": str(record["selector"]["training_mode"]),
            "state": str(record["outcome"]),
            "prediction_available": available,
            "predicted_reserved_center_gib": (
                center / GIB if center is not None else None
            ),
            "decision_upper_gib": (
                upper / GIB if upper is not None else None
            ),
            "safe_limit_gib": safe_limit / GIB,
            "predicted_admit": admitted,
            "observed_reserved_gib": (
                float(observed_reserved) / GIB
                if observed_reserved is not None
                else None
            ),
            "actual_safe_success": (
                bool(observed_reserved <= safe_limit)
                if observed_reserved is not None
                else False
            ),
            "false_safe_oom": (
                str(record["outcome"]) == "oom" and admitted
            ),
        }

    return (
        {
            "evaluation_rows": len(records),
            "state_counts": dict(
                sorted(
                    Counter(
                        str(record["outcome"]) for record in records
                    ).items()
                )
            ),
            "model_catalog": {
                memory_replay.MODEL_11D: {
                    "formula": "log(reserved) = 11 static features @ beta",
                    "decision_head": "point estimate",
                },
                memory_replay.MODEL_NATIVE: {
                    "formula": (
                        "analytic additive memory decomposition plus "
                        "operational P95 tail"
                    ),
                    "decision_head": "operational P95",
                },
                memory_replay.MODEL_PHYSICAL: {
                    "formula": (
                        "analytic reference times a 28-feature residual "
                        "plus operational P95 tail"
                    ),
                    "decision_head": "operational P95",
                },
            },
            "legacy_fit_reconstruction": {
                **old_training_summary,
                "coefficients": np.asarray(beta).tolist(),
                "new_holdout_rows_in_fit": 0,
            },
            "models": model_results,
            "physical_shares_admission_by_job": physical_admission,
            "historical_reference": {
                "path": str(memory_replay.DEFAULT_OUTPUT.resolve()),
                "sha256": sha256_file(memory_replay.DEFAULT_OUTPUT),
                "old_native_holdout": historical["old_evidence"][
                    "common_native_holdout"
                ]["models"],
                "previous_qwen3_generalization": historical[
                    "new_generalization_evidence"
                ]["models"],
            },
        },
        physical_admission,
    )


def _scenario_material(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "hardware_id": "h800",
        "model_id": str(job["model_id"]),
        "dataset_id": str(job["dataset_id"]),
        "train_type": str(job["train_type"]),
        "target_gbs": int(job["target_gbs"]),
        "cutoff_len": int(job["cutoff_len"]),
        "packing": bool(job["packing"]),
        "dtype": "bf16",
    }


def _scenario_id(job: Mapping[str, Any]) -> str:
    return sha256_json(_scenario_material(job))


def _post_first_step_observed(attempt: Path) -> dict[str, Any] | None:
    rank_events = sorted((attempt / "metrics").glob("events.rank*.jsonl"))
    if not rank_events:
        return None
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in rank_events:
        for row in _read_jsonl(path):
            if row.get("event") != "step_end":
                continue
            step = int(row["global_step"])
            if step >= 2:
                by_step[step].append(row)
    if not by_step:
        return None
    seconds = 0.0
    effective = computed = logical = 0
    for step_rows in by_step.values():
        seconds += max(float(row["step_seconds"]) for row in step_rows)
        effective += sum(
            int(row["tokens"]["effective_tokens"]) for row in step_rows
        )
        computed += sum(
            int(row["tokens"]["computed_tokens"]) for row in step_rows
        )
        logical += sum(
            int(row["tokens"]["logical_samples"]) for row in step_rows
        )
    steps = len(by_step)
    return {
        "measured_steps": steps,
        "measured_seconds": seconds,
        "effective_tokens": effective,
        "computed_tokens": computed,
        "logical_samples": logical,
        "effective_tokens_per_second": effective / seconds,
        "step_seconds": seconds / steps,
        "effective_tokens_per_step": effective / steps,
        "exclusion": "global_step_1_excluded_for_first_kernel_compile",
    }


def _throughput_source(
    outcome: Mapping[str, Any],
    *,
    kernel_path: str,
    observed_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    job = outcome["job"]
    observed = dict(observed_override or outcome["observed"])
    finished = float(outcome["status"]["finished_unix"])
    return {
        "job_id": str(job["job_id"]),
        "finished_unix": finished,
        "finished_at_utc": _unix_to_iso(finished),
        "configuration": {
            "model_id": str(job["model_id"]),
            "model_family": str(job.get("model_family") or ""),
            "train_type": str(job["train_type"]),
            "dataset_id": str(job["dataset_id"]),
            "gpu_count": int(job["gpu_count"]),
            "target_gbs": int(job["target_gbs"]),
            "cutoff_len": int(job["cutoff_len"]),
            "mbs": int(job["mbs"]),
            "gradient_checkpointing": bool(job["gc"]),
            "zero_stage": str(job["zero"]),
            "packing": bool(job["packing"]),
            "repeat": int(job.get("repeat") or 0),
            "kernel_path": kernel_path,
        },
        "observed": {
            "measured_steps": int(observed["measured_steps"]),
            "measured_seconds": float(observed["measured_seconds"]),
            "effective_tokens": int(observed["effective_tokens"]),
            "computed_tokens": int(observed["computed_tokens"]),
            "logical_samples": int(observed["logical_samples"]),
            "effective_tokens_per_second": float(
                observed["effective_tokens_per_second"]
            ),
            "step_seconds": float(observed["step_seconds"]),
            "effective_tokens_per_step": float(
                observed["effective_tokens_per_step"]
            ),
        },
    }


def _throughput_entries(
    outcomes: Sequence[Mapping[str, Any]],
    predictor: ThroughputPredictor,
) -> dict[str, dict[str, Any]]:
    joint = read_json(BASE / "artifacts" / "joint_throughput_modeling.json")
    joint_model = joint["h800"]["frozen_model"]
    entries: dict[str, dict[str, Any]] = {}
    for input_index, outcome in enumerate(outcomes):
        if outcome["throughput_role"] == "excluded_memory_probe":
            continue
        job = outcome["job"]
        request = {
            **job,
            "request_id": str(job["job_id"]),
            "comparison_group": _scenario_id(job),
            "hardware_id": "h800",
            "dtype": "bf16",
        }
        normalized = predictor._normalized_request(
            request,
            input_index=input_index,
        )
        v5_prediction = predictor._predict_normalized(
            normalized,
            explain=False,
        )
        record, basis = _static_record(
            predictor,
            job,
            input_index=input_index,
        )
        candidate = _joint_candidate(
            record,
            scenario_id=_scenario_id(job),
        )
        v4_rate = math.exp(
            _predict_log_throughput(candidate, joint_model)
        )
        source = (
            _throughput_source(
                outcome,
                kernel_path=str(
                    v5_prediction["configuration"]["kernel_path"]
                ),
            )
            if outcome["state"] == "success"
            else None
        )
        entries[str(job["job_id"])] = {
            "job_id": str(job["job_id"]),
            "cohort": str(outcome["cohort"]),
            "throughput_role": str(outcome["throughput_role"]),
            "job": dict(job),
            "outcome": outcome,
            "state": str(outcome["state"]),
            "scenario_id": _scenario_id(job),
            "scenario": _scenario_material(job),
            "candidate": candidate,
            "basis": basis,
            "source": source,
            "v4_rate": v4_rate,
            "v5_rate": float(
                v5_prediction[
                    "predicted_effective_tokens_per_second"
                ]
            ),
            "v5_confidence": v5_prediction["confidence"],
        }
    return entries


def _v4b_rates(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    if not entries:
        return {}
    joint = read_json(BASE / "artifacts" / "joint_throughput_modeling.json")
    model = joint["h800"]["frozen_two_head_challenger"]
    result: dict[str, float] = {}
    for candidate, predicted_log in _predict_two_head_entries(
        [entry["candidate"] for entry in entries],
        model,
    ):
        result[str(candidate["comparison_job_id"])] = math.exp(
            float(predicted_log)
        )
    return result


def _rates_for_scope(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    return {
        "v4_full_factor_single_head": {
            str(entry["job_id"]): float(entry["v4_rate"])
            for entry in entries
        },
        "v4b_full_factor_two_head_rejected": _v4b_rates(entries),
        "v5_structured_cross_card_single_head": {
            str(entry["job_id"]): float(entry["v5_rate"])
            for entry in entries
        },
    }


def _evaluation_rows(
    entries: Sequence[Mapping[str, Any]],
    rates: Mapping[str, float],
    *,
    observed_override: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        if entry["state"] != "success":
            continue
        job_id = str(entry["job_id"])
        source = entry["source"]
        if observed_override and job_id in observed_override:
            source = _throughput_source(
                entry["outcome"],
                kernel_path=str(
                    source["configuration"]["kernel_path"]
                ),
                observed_override=observed_override[job_id],
            )
        row = _evaluation_row(
            source,
            predicted_rate=float(rates[job_id]),
            static_work=float(
                entry["basis"]["work_per_step"]["effective_tokens"]
            ),
        )
        row["cohort"] = str(entry["cohort"])
        row["fidelity"] = str(entry["job"].get("fidelity") or "")
        row["v5_confidence"] = {
            "label": entry["v5_confidence"]["label"],
            "inside_frozen_support": entry["v5_confidence"][
                "inside_frozen_support"
            ],
            "reason_codes": [
                str(reason["code"])
                for reason in entry["v5_confidence"]["reasons"]
            ],
        }
        rows.append(row)
    return rows


def _top_k_summary(ranking: Mapping[str, Any]) -> dict[str, Any]:
    scenarios = [
        scenario
        for scenario in ranking["scenarios"]
        if scenario["status"] == "evaluated"
    ]
    if not scenarios:
        return {
            "evaluated_scenarios": 0,
            "exact_top1_hit_fraction": None,
            "observed_best_in_predicted_top2_fraction": None,
            "observed_best_in_predicted_top3_fraction": None,
            "selected_at_least_90pct_of_best_fraction": None,
            "mean_selected_to_best_percent": None,
        }
    top1 = top2 = top3 = hit90 = 0
    ratios = []
    for scenario in scenarios:
        candidates = scenario["candidates"]
        observed_best = max(
            candidates,
            key=lambda row: row[
                "observed_effective_tokens_per_second"
            ],
        )
        predicted_order = sorted(
            candidates,
            key=lambda row: row[
                "predicted_effective_tokens_per_second"
            ],
            reverse=True,
        )
        best_key = json.dumps(
            observed_best["candidate"],
            sort_keys=True,
            separators=(",", ":"),
        )
        predicted_keys = [
            json.dumps(
                row["candidate"],
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in predicted_order
        ]
        ratio = (
            float(
                predicted_order[0][
                    "observed_effective_tokens_per_second"
                ]
            )
            / float(
                observed_best["observed_effective_tokens_per_second"]
            )
        )
        ratios.append(ratio)
        top1 += int(predicted_keys[0] == best_key)
        top2 += int(best_key in predicted_keys[:2])
        top3 += int(best_key in predicted_keys[:3])
        hit90 += int(ratio >= 0.9)
    count = len(scenarios)
    return {
        "evaluated_scenarios": count,
        "exact_top1_hit_fraction": top1 / count,
        "observed_best_in_predicted_top2_fraction": top2 / count,
        "observed_best_in_predicted_top3_fraction": top3 / count,
        "selected_at_least_90pct_of_best_fraction": hit90 / count,
        "mean_selected_to_best_percent": (
            100.0 * statistics.fmean(ratios)
        ),
        "worst_selected_to_best_percent": 100.0 * min(ratios),
    }


def _score_throughput_scope(
    entries: Sequence[Mapping[str, Any]],
    *,
    rates: Mapping[str, Mapping[str, float]] | None = None,
    observed_override: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    model_rates = dict(rates or _rates_for_scope(entries))
    result: dict[str, Any] = {}
    for model_id in THROUGHPUT_MODEL_IDS:
        rows = _evaluation_rows(
            entries,
            model_rates[model_id],
            observed_override=observed_override,
        )
        metrics = throughput_metrics(rows)
        metrics["ranking"]["top_k"] = _top_k_summary(
            metrics["ranking"]
        )
        result[model_id] = {
            "metrics": metrics,
            "by_model_train_dataset": _cohort_breakdown(rows),
            "detailed_predictions": rows,
        }
    return result


def _pipeline_metrics(
    formal_entries: Sequence[Mapping[str, Any]],
    physical_admission: Mapping[str, Mapping[str, Any]],
    rates: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in formal_entries:
        by_scenario[str(entry["scenario_id"])].append(entry)

    result: dict[str, Any] = {}
    for model_id in THROUGHPUT_MODEL_IDS:
        scenarios = []
        for scenario_id, entries in sorted(by_scenario.items()):
            actual_safe_successes = [
                entry
                for entry in entries
                if entry["state"] == "success"
                and physical_admission[str(entry["job_id"])][
                    "actual_safe_success"
                ]
            ]
            admitted = [
                entry
                for entry in entries
                if physical_admission[str(entry["job_id"])][
                    "predicted_admit"
                ]
            ]
            scenario = {
                "scenario_id": scenario_id,
                "scenario": entries[0]["scenario"],
                "executed_candidates": len(entries),
                "actual_safe_successes": len(actual_safe_successes),
                "physical_admitted_candidates": len(admitted),
                "physical_rejected_actual_safe_successes": sum(
                    entry not in admitted
                    for entry in actual_safe_successes
                ),
            }
            if len(actual_safe_successes) < 2:
                scenario["status"] = (
                    "insufficient_actual_safe_successes"
                )
                scenarios.append(scenario)
                continue
            observed_best = max(
                actual_safe_successes,
                key=lambda entry: float(
                    entry["source"]["observed"][
                        "effective_tokens_per_second"
                    ]
                ),
            )
            if not admitted:
                scenario.update(
                    {
                        "status": "no_candidate_admitted",
                        "selected_job_id": None,
                        "selected_outcome": None,
                        "top1_regret_percent": None,
                        "selected_at_least_90pct_of_best": False,
                    }
                )
                scenarios.append(scenario)
                continue
            selected = max(
                admitted,
                key=lambda entry: float(
                    rates[model_id][str(entry["job_id"])]
                ),
            )
            selected_outcome = str(selected["state"])
            selected_rate = (
                float(
                    selected["source"]["observed"][
                        "effective_tokens_per_second"
                    ]
                )
                if selected_outcome == "success"
                else None
            )
            best_rate = float(
                observed_best["source"]["observed"][
                    "effective_tokens_per_second"
                ]
            )
            regret = (
                100.0 * (best_rate - selected_rate) / best_rate
                if selected_rate is not None
                else None
            )
            scenario.update(
                {
                    "status": (
                        "selected_success"
                        if selected_outcome == "success"
                        else "selected_oom"
                    ),
                    "observed_best_job_id": str(
                        observed_best["job_id"]
                    ),
                    "observed_best_effective_tokens_per_second": (
                        best_rate
                    ),
                    "selected_job_id": str(selected["job_id"]),
                    "selected_outcome": selected_outcome,
                    "selected_effective_tokens_per_second": selected_rate,
                    "top1_regret_percent": regret,
                    "selected_at_least_90pct_of_best": bool(
                        selected_rate is not None
                        and selected_rate >= 0.9 * best_rate
                    ),
                }
            )
            scenarios.append(scenario)

        eligible = [
            row
            for row in scenarios
            if row["actual_safe_successes"] >= 2
        ]
        successful_selections = [
            row for row in eligible if row["status"] == "selected_success"
        ]
        result[model_id] = {
            "eligible_scenarios": len(eligible),
            "selected_success_scenarios": len(successful_selections),
            "selected_oom_scenarios": sum(
                row["status"] == "selected_oom" for row in eligible
            ),
            "no_candidate_admitted_scenarios": sum(
                row["status"] == "no_candidate_admitted"
                for row in eligible
            ),
            "exact_best_hit_fraction_given_successful_selection": (
                statistics.fmean(
                    float(
                        row["selected_job_id"]
                        == row["observed_best_job_id"]
                    )
                    for row in successful_selections
                )
                if successful_selections
                else None
            ),
            "end_to_end_exact_best_hit_fraction": (
                statistics.fmean(
                    float(
                        row["status"] == "selected_success"
                        and row["selected_job_id"]
                        == row["observed_best_job_id"]
                    )
                    for row in eligible
                )
                if eligible
                else None
            ),
            "selected_at_least_90pct_of_best_fraction": (
                statistics.fmean(
                    float(row["selected_at_least_90pct_of_best"])
                    for row in eligible
                )
                if eligible
                else None
            ),
            "mean_top1_regret_percent_on_successful_selection": (
                statistics.fmean(
                    float(row["top1_regret_percent"])
                    for row in successful_selections
                )
                if successful_selections
                else None
            ),
            "worst_top1_regret_percent_on_successful_selection": (
                max(
                    float(row["top1_regret_percent"])
                    for row in successful_selections
                )
                if successful_selections
                else None
            ),
            "scenarios": scenarios,
        }
    return result


def _historical_throughput_reference() -> dict[str, Any]:
    path = (
        BASE
        / "artifacts"
        / "throughput_model_generations_generalization_2026-07-28.json"
    )
    report = read_json(path)
    models = {
        str(row["model_id"]): row for row in report["h800_models"]
    }
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "models": {
            model_id: {
                "common_unseen_model_holdout": models[model_id][
                    "common_unseen_model_holdout"
                ],
                "strict_post_artifact": models[model_id][
                    "strict_post_artifact"
                ],
            }
            for model_id in THROUGHPUT_MODEL_IDS
        },
    }


def _throughput_replay(
    outcomes: Sequence[Mapping[str, Any]],
    physical_admission: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    predictor = ThroughputPredictor(
        model_inventory=NEW_INVENTORY,
        strict_bindings=False,
    )
    entries_by_job = _throughput_entries(outcomes, predictor)
    entries = list(entries_by_job.values())
    formal = [
        entry
        for entry in entries
        if entry["throughput_role"] == "primary_formal"
    ]
    formal_success = [
        entry for entry in formal if entry["state"] == "success"
    ]
    packing_success = [
        entry
        for entry in entries
        if entry["throughput_role"] == "packing_diagnostic"
        and entry["state"] == "success"
    ]
    canary_success = [
        entry
        for entry in entries
        if entry["throughput_role"] == "canary_diagnostic"
        and entry["state"] == "success"
    ]

    primary_rates = _rates_for_scope(formal_success)
    primary = _score_throughput_scope(
        formal_success,
        rates=primary_rates,
    )

    packing = _score_throughput_scope(packing_success)
    canary_raw = _score_throughput_scope(canary_success)
    post_first = {}
    for entry in canary_success:
        observed = _post_first_step_observed(
            Path(entry["outcome"]["attempt_path"])
        )
        if observed is not None:
            post_first[str(entry["job_id"])] = observed
    canary_post_first = _score_throughput_scope(
        canary_success,
        observed_override=post_first,
    )

    admitted_formal = [
        entry
        for entry in formal
        if physical_admission[str(entry["job_id"])][
            "predicted_admit"
        ]
    ]
    pipeline_rates = _rates_for_scope(admitted_formal)
    admitted_success = [
        entry for entry in admitted_formal if entry["state"] == "success"
    ]
    physical_filtered = _score_throughput_scope(
        admitted_success,
        rates=pipeline_rates,
    )
    pipeline = _pipeline_metrics(
        formal,
        physical_admission,
        pipeline_rates,
    )

    return {
        "evaluation_contract": {
            "new_outcomes_refit_any_model": False,
            "primary_ranking_cohort": "qwen25_formal_throughput",
            "primary_success_rows": len(formal_success),
            "primary_oom_rows_excluded_from_throughput_metrics": (
                sum(entry["state"] == "oom" for entry in formal)
            ),
            "ranking_unit": (
                "exact repeats averaged before pairwise and Top-k metrics"
            ),
            "packing_policy": (
                "ABBA rows are a separate diagnostic because packing "
                "changes workload semantics and used 2+8 steps"
            ),
            "qwen35_policy": (
                "5-step compatibility canary is reported separately; "
                "step 1 includes TileLang compilation, and a steps 2-5 "
                "diagnostic is also reported"
            ),
            "v4b_candidate_set_policy": (
                "raw formal replay uses observed-success candidates; "
                "physical-shares pipeline recomputes V4b after the "
                "predicted memory filter, including any admitted OOM"
            ),
        },
        "model_catalog": {
            "v4_full_factor_single_head": (
                "70-feature full-factor single-head model"
            ),
            "v4b_full_factor_two_head_rejected": (
                "70-feature absolute plus pairwise two-head model; "
                "candidate-set dependent"
            ),
            "v5_structured_cross_card_single_head": (
                "35-feature structured physical model with card adapter"
            ),
        },
        "predictor_binding_mismatches": predictor.binding_mismatches,
        "primary_qwen25_formal_success_only": primary,
        "physical_shares_filtered_qwen25_formal": {
            "admitted_candidates_including_terminal_oom": len(
                admitted_formal
            ),
            "admitted_success_rows": len(admitted_success),
            "admitted_oom_rows": sum(
                entry["state"] == "oom" for entry in admitted_formal
            ),
            "models": physical_filtered,
        },
        "end_to_end_physical_shares_plus_throughput": pipeline,
        "packing_abba_diagnostic": packing,
        "qwen35_canary_raw_five_step_diagnostic": canary_raw,
        "qwen35_canary_steps_2_to_5_diagnostic": canary_post_first,
        "qwen35_post_first_step_observed": post_first,
        "historical_reference": _historical_throughput_reference(),
    }


def _freeze_checks() -> dict[str, Any]:
    freeze = read_json(PREDICTOR_FREEZE)
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
        "path": str(PREDICTOR_FREEZE.resolve()),
        "sha256": sha256_file(PREDICTOR_FREEZE),
        "all_frozen_artifacts_unchanged": all(
            row["matches"] for row in checks
        ),
        "checks": checks,
    }


def build_report(output: Path) -> dict[str, Any]:
    outcomes = _collect_outcomes()
    memory, physical_admission = _memory_replay(outcomes)
    throughput = _throughput_replay(outcomes, physical_admission)
    queue_bindings = {
        str(spec["cohort"]): {
            "path": str(Path(spec["queue"]).resolve()),
            "sha256": sha256_file(Path(spec["queue"])),
            "jobs": len(_read_jsonl(Path(spec["queue"]))),
        }
        for spec in COHORTS
    }
    report = {
        "schema": "qwen25_qwen35_frozen_model_replay/v1",
        "implementation_version": (
            "qwen25_qwen35_frozen_model_replay/2026-07-29.v1"
        ),
        "generated_at_utc": _utc_now(),
        "evaluation_contract": {
            "gpu_experiments_launched": False,
            "queues_mutated": False,
            "new_outcomes_used_as_labels_only": True,
            "new_outcomes_used_to_refit_models": False,
            "memory_models_replayed": list(MEMORY_MODEL_IDS),
            "throughput_models_replayed": list(
                THROUGHPUT_MODEL_IDS
            ),
        },
        "source_bindings": {
            "queues": queue_bindings,
            "new_model_inventory": {
                "path": str(NEW_INVENTORY.resolve()),
                "sha256": sha256_file(NEW_INVENTORY),
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
        },
        "freeze_audit": _freeze_checks(),
        "outcome_snapshot": {
            "rows": len(outcomes),
            "state_counts": dict(
                sorted(
                    Counter(
                        str(outcome["state"]) for outcome in outcomes
                    ).items()
                )
            ),
            "by_cohort": {
                cohort: {
                    "rows": len(rows),
                    "state_counts": dict(
                        sorted(
                            Counter(
                                str(row["state"]) for row in rows
                            ).items()
                        )
                    ),
                }
                for cohort, rows in sorted(
                    (
                        (
                            cohort,
                            [
                                outcome
                                for outcome in outcomes
                                if outcome["cohort"] == cohort
                            ],
                        )
                        for cohort in {
                            str(outcome["cohort"])
                            for outcome in outcomes
                        }
                    )
                )
            },
        },
        "memory_replay": memory,
        "throughput_replay": throughput,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(output, report)
    return report


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def _print_summary(report: Mapping[str, Any]) -> None:
    print("outcomes", report["outcome_snapshot"]["state_counts"])
    for model_id in MEMORY_MODEL_IDS:
        metrics = report["memory_replay"]["models"][model_id]["overall"]
        center = metrics["center_accuracy"][
            "absolute_percentage_error_percent"
        ]
        safety = metrics["safety"]
        print(
            "memory",
            model_id,
            "MAPE=" + _fmt(center["mean"]),
            "false_safe="
            + f"{safety['false_safe_oom']}/{safety['oom_rows']}",
            "coverage="
            + _fmt(safety["success_upper_coverage_percent"]),
            "recall="
            + _fmt(
                safety["safe_success_admission_recall_percent"]
            ),
        )
    primary = report["throughput_replay"][
        "primary_qwen25_formal_success_only"
    ]
    for model_id in THROUGHPUT_MODEL_IDS:
        metrics = primary[model_id]["metrics"]
        print(
            "throughput",
            model_id,
            "MAPE="
            + _fmt(metrics["absolute"].get("throughput_mape_percent")),
            "pairwise="
            + _fmt(
                metrics["ranking"].get(
                    "pooled_pairwise_accuracy_percent"
                )
            ),
            "regret="
            + _fmt(metrics["ranking"].get("mean_top1_regret_percent")),
            "hit90="
            + _fmt(
                metrics["ranking"]["top_k"].get(
                    "selected_at_least_90pct_of_best_fraction"
                )
            ),
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
