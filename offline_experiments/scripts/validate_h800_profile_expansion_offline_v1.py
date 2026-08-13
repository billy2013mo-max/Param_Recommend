#!/usr/bin/env python3
"""Offline validation of profile-aware H800 reserved-memory expansion.

This CPU-only diagnostic follows five ordered steps:

1. Rebuild the frozen V5 fit/strict-holdout split and extract pre-run profile
   statistics that already exist in the queues and tokenizer profiles.
2. Reconstruct per-step allocated/reserved trajectories from retained raw event
   logs.  These post-run values are diagnostic only and never enter the model.
3. Fit Ridge heads for log(reserved / allocated) using training successes only.
   Feature set, regularization, and weighting are selected by training-source
   leave-one-source-out (LOSO) predictions.
4. Evaluate the frozen heads once on the five strict unused dataset sources and
   compare them with the analytic reference and current V5 M1 centers.
5. Audit exact feature collisions before and after adding profile statistics.

OOM observations remain right-censored and are excluded from center fitting and
MAPE calculations.  The script writes shadow diagnostics only; it never mutates
or publishes the production predictor.
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

from calibrate_h800_m1_safety_upper_v2 import _prepare_records
from common import RESULTS_DIR, ROOT, read_json, sha256_file, write_json, write_jsonl
from fit_h800_lora_source_disjoint_recalibration_v1 import (
    BASE_FEATURES,
    DEFAULT_FROZEN_BASELINE,
    DEFAULT_HARDWARE,
    _collapse_records,
    _feature_values,
    _inventory_models,
    _is_critical_lora,
    _label,
    _mechanism_key,
    _outcome,
    _predict_center,
    _read_jsonl,
    _source_id,
)
from fit_h800_profile_aware_memory_challenger_v1 import _export_exact_jobs
from migrate_refit_h800_historical_memory_v1 import (
    DEFAULT_CANONICAL,
    DEFAULT_CURRENT_OLD,
    DEFAULT_DATASET_ANALYSIS,
    DEFAULT_NEW_QUEUE,
    DEFAULT_THEORY_BASIS,
    _build_pairs,
    _current_data_paths,
    _dataset_registry,
)
from refit_h800_m1_all_unused_validation_v3 import (
    DEFAULT_STAGE2_QUEUE,
    DEFAULT_VALIDATION_INVENTORY,
    HOLDOUT_CAMPAIGNS,
    _build_validation_record,
    _content_audit,
    _export_validation_jobs,
    _paths_from_queue,
)


SCHEMA = "sft_h800_profile_expansion_offline_validation/v1"
DEFAULT_CURRENT_CANDIDATE = (
    ROOT
    / "diagnostics"
    / "h800_m1_lora_full_admission_v5_20260805"
    / "candidate_model_m1_lora_full_admission_v5.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "diagnostics" / "h800_profile_expansion_offline_v1_20260809"
)
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
WEIGHTINGS = ("source_balanced", "mechanism_source_balanced")
PROFILE_FEATURES = (
    "profile_mean_fraction",
    "profile_p90_fraction",
    "profile_p99_fraction",
    "profile_expected_batch_max_fraction",
    "profile_maximum_fraction",
    "profile_cv_bounded",
    "profile_truncation_fraction",
    "profile_rare_tail_gap",
)
LORA_PROFILE_INTERACTIONS = (
    "lora_x_profile_expected_batch_max_fraction",
    "lora_x_profile_p99_fraction",
    "lora_x_profile_truncation_fraction",
    "lora_x_profile_rare_tail_gap",
)
CRITICAL_PROFILE_INTERACTIONS = (
    "critical_x_profile_expected_batch_max_fraction",
    "critical_x_profile_p99_fraction",
    "critical_x_profile_truncation_fraction",
    "critical_x_profile_rare_tail_gap",
)
FEATURE_SETS = {
    "m1_expansion_baseline": tuple(BASE_FEATURES),
    "profile_main": (*BASE_FEATURES, *PROFILE_FEATURES),
    "profile_lora_interactions": (
        *BASE_FEATURES,
        *PROFILE_FEATURES,
        *LORA_PROFILE_INTERACTIONS,
    ),
    "profile_critical_interactions": (
        *BASE_FEATURES,
        *PROFILE_FEATURES,
        *CRITICAL_PROFILE_INTERACTIONS,
    ),
    "profile_lora_and_critical_interactions": (
        *BASE_FEATURES,
        *PROFILE_FEATURES,
        *LORA_PROFILE_INTERACTIONS,
        *CRITICAL_PROFILE_INTERACTIONS,
    ),
}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bounded_fraction(value: Any, denominator: float) -> float:
    raw = _finite(value)
    if raw is None or denominator <= 0:
        raise ValueError("profile fraction is unavailable")
    return min(1.0, max(0.0, raw / denominator))


def _profile_values(record: Mapping[str, Any]) -> dict[str, float]:
    scenario = record.get("scenario") or {}
    selector = record.get("selector") or {}
    padding = (record.get("recalibration") or {}).get("padding") or {}
    cutoff = float(scenario["cutoff_len"])
    expected_fraction = _finite(
        padding.get("expected_random_batch_max_fraction_of_cutoff")
    )
    if expected_fraction is None:
        expected_fraction = _bounded_fraction(
            padding.get("expected_random_batch_max_tokens"), cutoff
        )
    expected_fraction = min(1.0, max(0.0, expected_fraction))
    maximum_fraction = _bounded_fraction(
        padding.get("maximum_clipped_tokens"), cutoff
    )
    cv = _finite(padding.get("coefficient_of_variation"))
    truncation = _finite(padding.get("truncation_fraction"))
    if cv is None or truncation is None:
        raise ValueError("profile CV or truncation fraction is unavailable")
    cv_bounded = min(1.0, max(0.0, cv / 2.0))
    truncation = min(1.0, max(0.0, truncation))
    is_lora = float(selector.get("training_mode") == "lora")
    critical = float(_is_critical_lora(record))
    values = {
        "profile_mean_fraction": _bounded_fraction(
            padding.get("mean_clipped_tokens"), cutoff
        ),
        "profile_p90_fraction": _bounded_fraction(
            padding.get("p90_clipped_tokens"), cutoff
        ),
        "profile_p99_fraction": _bounded_fraction(
            padding.get("p99_clipped_tokens"), cutoff
        ),
        "profile_expected_batch_max_fraction": expected_fraction,
        "profile_maximum_fraction": maximum_fraction,
        "profile_cv_bounded": cv_bounded,
        "profile_truncation_fraction": truncation,
        "profile_rare_tail_gap": max(0.0, maximum_fraction - expected_fraction),
    }
    values.update(
        {
            "lora_x_profile_expected_batch_max_fraction": (
                is_lora * values["profile_expected_batch_max_fraction"]
            ),
            "lora_x_profile_p99_fraction": (
                is_lora * values["profile_p99_fraction"]
            ),
            "lora_x_profile_truncation_fraction": (
                is_lora * values["profile_truncation_fraction"]
            ),
            "lora_x_profile_rare_tail_gap": (
                is_lora * values["profile_rare_tail_gap"]
            ),
            "critical_x_profile_expected_batch_max_fraction": (
                critical * values["profile_expected_batch_max_fraction"]
            ),
            "critical_x_profile_p99_fraction": (
                critical * values["profile_p99_fraction"]
            ),
            "critical_x_profile_truncation_fraction": (
                critical * values["profile_truncation_fraction"]
            ),
            "critical_x_profile_rare_tail_gap": (
                critical * values["profile_rare_tail_gap"]
            ),
        }
    )
    return values


def _all_feature_values(record: Mapping[str, Any]) -> dict[str, float]:
    return {**_feature_values(record), **_profile_values(record)}


def _vector(record: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    values = _all_feature_values(record)
    return np.asarray([values[name] for name in names], dtype=float)


def _successful(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in records
        if _outcome(row) == "success"
        and _label(row, "allocated") is not None
        and _label(row, "reserved") is not None
    ]


def _weights(
    records: Sequence[Mapping[str, Any]], weighting: str
) -> np.ndarray:
    if weighting == "source_balanced":
        counts = Counter(_source_id(row) for row in records)
        raw = np.asarray(
            [1.0 / counts[_source_id(row)] for row in records], dtype=float
        )
    elif weighting == "mechanism_source_balanced":
        pair_counts = Counter(
            (_mechanism_key(row), _source_id(row)) for row in records
        )
        sources_by_mechanism: dict[str, set[str]] = defaultdict(set)
        for row in records:
            sources_by_mechanism[_mechanism_key(row)].add(_source_id(row))
        mechanism_count = len(sources_by_mechanism)
        raw = np.asarray(
            [
                1.0
                / mechanism_count
                / len(sources_by_mechanism[_mechanism_key(row)])
                / pair_counts[(_mechanism_key(row), _source_id(row))]
                for row in records
            ],
            dtype=float,
        )
    else:
        raise ValueError(f"unsupported weighting: {weighting}")
    return raw * (len(records) / float(raw.sum()))


def _fit_expansion_ridge(
    records: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str],
    alpha: float,
    weighting: str,
) -> dict[str, Any]:
    successes = _successful(records)
    if not successes:
        raise ValueError("no successful expansion labels")
    features = np.vstack([_vector(row, names) for row in successes])
    targets = np.asarray(
        [
            math.log(float(_label(row, "reserved")) / float(_label(row, "allocated")))
            for row in successes
        ],
        dtype=float,
    )
    weights = _weights(successes, weighting)
    means = np.average(features, axis=0, weights=weights)
    scales = np.sqrt(np.average((features - means) ** 2, axis=0, weights=weights))
    scales[scales < 1e-9] = 1.0
    standardized = (features - means) / scales
    design = np.column_stack((np.ones(len(successes)), standardized))
    penalty = np.diag([0.0, *([float(alpha)] * len(names))])
    coefficients = np.linalg.pinv(
        design.T @ (weights[:, None] * design) + penalty
    ) @ (design.T @ (weights * targets))
    raw_coefficients = coefficients[1:] / scales
    raw_intercept = float(coefficients[0] - means @ raw_coefficients)
    return {
        "model_family": "weighted_log_reserved_over_allocated_ridge",
        "target": "log(observed_reserved_bytes / observed_allocated_bytes)",
        "feature_names": list(names),
        "alpha": float(alpha),
        "weighting": weighting,
        "fit_success_rows": len(successes),
        "fit_independent_sources": len({_source_id(row) for row in successes}),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "raw_intercept": raw_intercept,
        "raw_coefficients": raw_coefficients.tolist(),
    }


def _predict_expansion(record: Mapping[str, Any], model: Mapping[str, Any]) -> float:
    names = list(model["feature_names"])
    vector = _vector(record, names)
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    log_ratio = float(model["intercept"]) + float(
        ((vector - means) / scales) @ coefficients
    )
    ratio = math.exp(log_ratio)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("expansion Ridge produced an invalid ratio")
    return ratio


def _source_equal_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _finite(row.get(key))
        if value is not None:
            grouped[str(row["source_id"])].append(value)
    if not grouped:
        return None
    return statistics.mean(statistics.mean(values) for values in grouped.values())


def _prediction_metrics(
    rows: Sequence[Mapping[str, Any]], prediction_key: str
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for row in rows:
        observed = _finite(row.get("observed_reserved_bytes"))
        predicted = _finite(row.get(prediction_key))
        if observed is None or predicted is None or observed <= 0:
            continue
        details.append(
            {
                "source_id": row["source_id"],
                "ape": abs(predicted / observed - 1.0),
                "signed_error": predicted / observed - 1.0,
            }
        )
    return {
        "success_rows": len(details),
        "independent_sources": len({row["source_id"] for row in details}),
        "row_mape": statistics.mean(row["ape"] for row in details) if details else None,
        "source_equal_mape": _source_equal_mean(details, "ape"),
        "row_signed_bias": (
            statistics.mean(row["signed_error"] for row in details)
            if details
            else None
        ),
        "source_equal_signed_bias": _source_equal_mean(details, "signed_error"),
        "max_ape": max((row["ape"] for row in details), default=None),
    }


def _ratio_metrics(rows: Sequence[Mapping[str, Any]], ratio_key: str) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for row in rows:
        observed = _finite(row.get("observed_expansion"))
        predicted = _finite(row.get(ratio_key))
        if observed is None or predicted is None or observed <= 0:
            continue
        details.append(
            {
                "source_id": row["source_id"],
                "ape": abs(predicted / observed - 1.0),
                "signed_error": predicted / observed - 1.0,
            }
        )
    return {
        "rows": len(details),
        "independent_sources": len({row["source_id"] for row in details}),
        "row_mape": statistics.mean(row["ape"] for row in details) if details else None,
        "source_equal_mape": _source_equal_mean(details, "ape"),
        "source_equal_signed_bias": _source_equal_mean(details, "signed_error"),
        "max_ape": max((row["ape"] for row in details), default=None),
    }


def _loso_details(
    records: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str],
    alpha: float,
    weighting: str,
) -> list[dict[str, Any]]:
    successes = _successful(records)
    sources = sorted({_source_id(row) for row in successes})
    details: list[dict[str, Any]] = []
    for held in sources:
        fit = [row for row in successes if _source_id(row) != held]
        evaluation = [row for row in successes if _source_id(row) == held]
        model = _fit_expansion_ridge(
            fit,
            names=names,
            alpha=alpha,
            weighting=weighting,
        )
        for row in evaluation:
            observed = float(_label(row, "reserved")) / float(
                _label(row, "allocated")
            )
            predicted = _predict_expansion(row, model)
            details.append(
                {
                    "source_id": held,
                    "observed_expansion": observed,
                    "predicted_expansion": predicted,
                    "ape": abs(predicted / observed - 1.0),
                    "signed_error": predicted / observed - 1.0,
                }
            )
    return details


def _loso_metrics(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(details),
        "independent_sources": len({str(row["source_id"]) for row in details}),
        "row_mape": statistics.mean(float(row["ape"]) for row in details),
        "source_equal_mape": _source_equal_mean(details, "ape"),
        "source_equal_signed_bias": _source_equal_mean(details, "signed_error"),
        "max_ape": max(float(row["ape"]) for row in details),
    }


def _select_candidates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for feature_set, names in FEATURE_SETS.items():
        for weighting in WEIGHTINGS:
            for alpha in ALPHAS:
                details = _loso_details(
                    records,
                    names=names,
                    alpha=alpha,
                    weighting=weighting,
                )
                metrics = _loso_metrics(details)
                candidates.append(
                    {
                        "feature_set": feature_set,
                        "feature_names": list(names),
                        "feature_count": len(names),
                        "weighting": weighting,
                        "alpha": alpha,
                        "training_loso_metrics": metrics,
                    }
                )
    ranked = sorted(
        candidates,
        key=lambda row: (
            float(row["training_loso_metrics"]["source_equal_mape"]),
            int(row["feature_count"]),
            float(row["alpha"]),
            str(row["weighting"]),
        ),
    )
    m1_ranked = [row for row in ranked if row["feature_set"] == "m1_expansion_baseline"]
    profile_ranked = [row for row in ranked if row["feature_set"] != "m1_expansion_baseline"]
    return {
        "selection_protocol": (
            "feature set, weighting, and alpha selected only by source-equal "
            "expansion MAPE from 40-source training LOSO; strict holdout unused"
        ),
        "candidates": ranked,
        "best_overall": ranked[0],
        "best_m1_only": m1_ranked[0],
        "best_profile": profile_ranked[0],
    }


def _fit_selected(
    records: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]
) -> dict[str, Any]:
    return _fit_expansion_ridge(
        records,
        names=selection["feature_names"],
        alpha=float(selection["alpha"]),
        weighting=str(selection["weighting"]),
    )


def _record_job_ids(record: Mapping[str, Any]) -> list[str]:
    recalibration = record.get("recalibration") or {}
    values = recalibration.get("job_ids")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return sorted({str(value) for value in values if value})
    value = recalibration.get("job_id") or record.get("job_id")
    return [str(value)] if value else []


def _event_paths(job_id: str) -> list[Path]:
    root = RESULTS_DIR / job_id
    latest_path = root / "latest_attempt.json"
    if not latest_path.is_file():
        return []
    latest = read_json(latest_path)
    attempt_path = latest.get("attempt_path")
    if not attempt_path:
        return []
    return sorted((root / str(attempt_path) / "metrics").glob("events.rank*.jsonl"))


def _trajectory_rows(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for split, records in records_by_split.items():
        jobs_requested: set[str] = set()
        jobs_with_events: set[str] = set()
        records_with_events = 0
        for record in records:
            record_had_event = False
            for job_id in _record_job_ids(record):
                jobs_requested.add(job_id)
                paths = _event_paths(job_id)
                if paths:
                    jobs_with_events.add(job_id)
                for path in paths:
                    with path.open(encoding="utf-8") as source:
                        for line in source:
                            if not line.strip():
                                continue
                            event = json.loads(line)
                            if event.get("event") != "step_end":
                                continue
                            memory = event.get("memory") or {}
                            tokens = event.get("tokens") or {}
                            max_allocated = _finite(memory.get("max_allocated"))
                            max_reserved = _finite(memory.get("max_reserved"))
                            physical_batches = _finite(tokens.get("physical_batches"))
                            computed_tokens = _finite(tokens.get("computed_tokens"))
                            record_had_event = True
                            steps.append(
                                {
                                    "split": split,
                                    "cluster_id": record.get("cluster_id"),
                                    "source_id": _source_id(record),
                                    "outcome": _outcome(record),
                                    "job_id": job_id,
                                    "event_path": str(path),
                                    "rank": event.get("rank"),
                                    "global_step": event.get("global_step"),
                                    "is_warmup": bool(event.get("is_warmup")),
                                    "allocated_bytes": _finite(memory.get("allocated")),
                                    "max_allocated_bytes": max_allocated,
                                    "reserved_bytes": _finite(memory.get("reserved")),
                                    "max_reserved_bytes": max_reserved,
                                    "step_peak_expansion": (
                                        max_reserved / max_allocated
                                        if max_reserved is not None
                                        and max_allocated is not None
                                        and max_allocated > 0
                                        else None
                                    ),
                                    "computed_tokens": computed_tokens,
                                    "effective_tokens": _finite(
                                        tokens.get("effective_tokens")
                                    ),
                                    "computed_attention_token_pairs": _finite(
                                        tokens.get("computed_attention_token_pairs")
                                    ),
                                    "physical_batches": physical_batches,
                                    "computed_tokens_per_physical_batch": (
                                        computed_tokens / physical_batches
                                        if computed_tokens is not None
                                        and physical_batches is not None
                                        and physical_batches > 0
                                        else None
                                    ),
                                    "actual_batch_max_sequence_recorded": False,
                                }
                            )
            records_with_events += int(record_had_event)
        split_steps = [row for row in steps if row["split"] == split]
        audit[split] = {
            "records": len(records),
            "records_with_step_events": records_with_events,
            "jobs_requested": len(jobs_requested),
            "jobs_with_event_files": len(jobs_with_events),
            "jobs_missing_event_files": sorted(jobs_requested - jobs_with_events),
            "step_event_rows": len(split_steps),
            "non_warmup_step_event_rows": sum(
                not bool(row["is_warmup"]) for row in split_steps
            ),
            "actual_batch_max_sequence_field_available": False,
        }
    return steps, audit


def _record_summary(record: Mapping[str, Any], split: str) -> dict[str, Any]:
    scenario = record.get("scenario") or {}
    selector = record.get("selector") or {}
    padding = (record.get("recalibration") or {}).get("padding") or {}
    allocated = _label(record, "allocated")
    reserved = _label(record, "reserved")
    return {
        "split": split,
        "cluster_id": record.get("cluster_id"),
        "observation_id": record.get("observation_id"),
        "job_ids": _record_job_ids(record),
        "source_id": _source_id(record),
        "outcome": _outcome(record),
        "origin": (record.get("recalibration") or {}).get("origin"),
        "model_id": scenario.get("model_id"),
        "training_mode": selector.get("training_mode"),
        "gpu_count": scenario.get("gpu_count"),
        "mbs": scenario.get("physical_mbs"),
        "cutoff_len": scenario.get("cutoff_len"),
        "zero_stage": selector.get("zero_stage"),
        "gradient_checkpointing": bool(selector.get("gradient_checkpointing")),
        "packing": bool(selector.get("packing")),
        "profile_path": padding.get("path"),
        "profile_sha256": padding.get("sha256"),
        "profile_statistics": dict(padding),
        "model_features": _all_feature_values(record),
        "observed_allocated_bytes": allocated,
        "observed_reserved_bytes": reserved,
        "observed_expansion": (
            float(reserved) / float(allocated)
            if allocated is not None and reserved is not None and float(allocated) > 0
            else None
        ),
    }


def _feature_key(record: Mapping[str, Any], names: Sequence[str]) -> tuple[float, ...]:
    values = _all_feature_values(record)
    return tuple(round(float(values[name]), 12) for name in names)


def _collision_audit(
    records: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> dict[str, Any]:
    successes = _successful(records)
    grouped: dict[tuple[float, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in successes:
        grouped[_feature_key(row, names)].append(row)
    collisions = [rows for rows in grouped.values() if len(rows) > 1]
    groups: list[dict[str, Any]] = []
    for rows in collisions:
        expansions = [
            float(_label(row, "reserved")) / float(_label(row, "allocated"))
            for row in rows
        ]
        reserved = [float(_label(row, "reserved")) for row in rows]
        groups.append(
            {
                "rows": len(rows),
                "source_ids": sorted({_source_id(row) for row in rows}),
                "model_ids": sorted(
                    {str((row.get("scenario") or {}).get("model_id")) for row in rows}
                ),
                "expansion_min": min(expansions),
                "expansion_max": max(expansions),
                "expansion_max_over_min": max(expansions) / min(expansions),
                "reserved_max_over_min": max(reserved) / min(reserved),
            }
        )
    return {
        "feature_count": len(names),
        "success_rows": len(successes),
        "unique_feature_vectors": len(grouped),
        "collision_groups": len(collisions),
        "collision_rows": sum(len(rows) for rows in collisions),
        "worst_expansion_max_over_min": max(
            (row["expansion_max_over_min"] for row in groups), default=1.0
        ),
        "groups": sorted(
            groups, key=lambda row: float(row["expansion_max_over_min"]), reverse=True
        ),
    }


def _is_supported_full(record: Mapping[str, Any]) -> bool:
    selector = record.get("selector") or {}
    scenario = record.get("scenario") or {}
    return bool(
        selector.get("training_mode") == "full"
        and int(selector.get("zero_stage") or 0) == 3
        and bool(selector.get("gradient_checkpointing"))
        and int(scenario.get("gpu_count") or 0) == 2
        and not bool(selector.get("packing"))
    )


def _scope(record: Mapping[str, Any], scope: str) -> bool:
    selector = record.get("selector") or {}
    if scope == "all_success":
        return True
    if scope == "critical_lora":
        return _is_critical_lora(record)
    if scope == "other_lora":
        return selector.get("training_mode") == "lora" and not _is_critical_lora(record)
    if scope == "supported_full":
        return _is_supported_full(record)
    raise ValueError(f"unknown scope: {scope}")


def _evaluate_strict(
    records: Sequence[Mapping[str, Any]],
    *,
    current_bundle: Mapping[str, Any],
    m1_expansion_model: Mapping[str, Any],
    profile_expansion_model: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for record in records:
        summary = _record_summary(record, "strict_unused_dataset")
        if _outcome(record) != "success":
            predictions.append(summary)
            continue
        observed_allocated = float(_label(record, "allocated"))
        observed_reserved = float(_label(record, "reserved"))
        allocated_center = _predict_center(record, current_bundle["allocated_model"])
        current_reserved = _predict_center(record, current_bundle["reserved_model"])
        m1_ratio = _predict_expansion(record, m1_expansion_model)
        profile_ratio = _predict_expansion(record, profile_expansion_model)
        predictions.append(
            {
                **summary,
                "critical_lora": _is_critical_lora(record),
                "supported_full": _is_supported_full(record),
                "analytic_reference_bytes": float(
                    (record.get("memory") or {})["analytic_reference_bytes"]
                ),
                "current_v5_allocated_center_bytes": allocated_center,
                "current_v5_reserved_center_bytes": current_reserved,
                "m1_expansion_predicted_ratio": m1_ratio,
                "m1_expansion_reserved_center_bytes": allocated_center * m1_ratio,
                "m1_expansion_oracle_allocated_reserved_bytes": (
                    observed_allocated * m1_ratio
                ),
                "profile_expansion_predicted_ratio": profile_ratio,
                "profile_expansion_reserved_center_bytes": (
                    allocated_center * profile_ratio
                ),
                "profile_expansion_oracle_allocated_reserved_bytes": (
                    observed_allocated * profile_ratio
                ),
                "observed_allocated_bytes": observed_allocated,
                "observed_reserved_bytes": observed_reserved,
                "observed_expansion": observed_reserved / observed_allocated,
            }
        )

    model_keys = {
        "analytic_reference": "analytic_reference_bytes",
        "current_v5_m1_reserved": "current_v5_reserved_center_bytes",
        "m1_expansion_end_to_end": "m1_expansion_reserved_center_bytes",
        "profile_expansion_end_to_end": "profile_expansion_reserved_center_bytes",
        "profile_expansion_oracle_allocated": (
            "profile_expansion_oracle_allocated_reserved_bytes"
        ),
    }
    metrics: dict[str, Any] = {}
    for scope in ("all_success", "critical_lora", "other_lora", "supported_full"):
        scoped_records = [row for row in records if _outcome(row) == "success" and _scope(row, scope)]
        cluster_ids = {str(row.get("cluster_id")) for row in scoped_records}
        scoped_predictions = [
            row for row in predictions if str(row.get("cluster_id")) in cluster_ids
        ]
        metrics[scope] = {
            name: _prediction_metrics(scoped_predictions, key)
            for name, key in model_keys.items()
        }
        metrics[scope]["m1_expansion_ratio"] = _ratio_metrics(
            scoped_predictions, "m1_expansion_predicted_ratio"
        )
        metrics[scope]["profile_expansion_ratio"] = _ratio_metrics(
            scoped_predictions, "profile_expansion_predicted_ratio"
        )
    return predictions, metrics


def _build_data(args: argparse.Namespace) -> dict[str, Any]:
    prepared = _prepare_records(args)
    base_training = list(prepared["training"])

    inventory = read_json(args.inventory)
    model_by_id, fixed_lora = _inventory_models(inventory)
    hardware = read_json(args.hardware)
    padding_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    raw_max_cache: dict[str, int] = {}

    stage2_queue = _read_jsonl(args.stage2_queue)
    stage2_observations = _export_exact_jobs(stage2_queue)
    stage2_cutoff, stage2_effective = _build_pairs(
        stage2_observations,
        model_by_id=model_by_id,
        fixed_lora=fixed_lora,
        hardware=hardware,
        origin="stage2_20_new_sources",
        padding_cache=padding_cache,
        raw_max_cache=raw_max_cache,
    )
    _, stage2_collapsed, stage2_collapse = _collapse_records(
        stage2_cutoff, stage2_effective
    )
    training = [*base_training, *stage2_collapsed]

    registry, _ = _dataset_registry(args.dataset_analysis)
    training_paths = {Path(row["data_path"]).resolve() for row in registry.values()}
    training_paths |= _paths_from_queue(_read_jsonl(args.new_queue))
    training_paths |= _paths_from_queue(stage2_queue)
    old_observations = _read_jsonl(args.current_old)
    training_paths |= set(_current_data_paths([], old_observations).values())

    campaigns: list[dict[str, Any]] = []
    for spec in HOLDOUT_CAMPAIGNS:
        queue_rows = _read_jsonl(Path(spec["queue"]))
        snapshot = read_json(Path(spec["snapshot"]))
        if snapshot.get("immutable") is not True:
            raise ValueError(f"holdout snapshot is not immutable: {spec['snapshot']}")
        if snapshot.get("source_queue_sha256") != sha256_file(Path(spec["queue"])):
            raise ValueError(f"holdout queue binding drifted: {spec['name']}")
        campaigns.append({**dict(spec), "queue_rows": queue_rows})
    content_audit = _content_audit(training_paths=training_paths, campaigns=campaigns)
    strict_source_ids = set(content_audit["strict_source_ids"])
    strict_validation: list[dict[str, Any]] = []
    for campaign in campaigns:
        observations = _export_validation_jobs(campaign["queue_rows"])
        for observation in observations:
            record = _build_validation_record(
                observation,
                model_by_id=model_by_id,
                fixed_lora=fixed_lora,
                hardware=hardware,
                origin=str(campaign["name"]),
                padding_cache=padding_cache,
                raw_max_cache=raw_max_cache,
            )
            if _source_id(record) in strict_source_ids:
                strict_validation.append(record)

    fit_sources = {_source_id(row) for row in training}
    strict_sources = {_source_id(row) for row in strict_validation}
    if fit_sources & strict_sources:
        raise ValueError("strict validation source ids overlap training")
    if len(training) != 273 or len(_successful(training)) != 230:
        raise ValueError("V5 fit row count drifted")
    if len(fit_sources) != 40:
        raise ValueError("V5 fit source count drifted")
    if len(strict_validation) != 31 or len(strict_sources) != 5:
        raise ValueError("strict validation count drifted")
    return {
        "training": training,
        "strict": strict_validation,
        "base_training_audit": prepared["audit"],
        "stage2_collapse": stage2_collapse,
        "content_audit": content_audit,
    }


def _percent(value: Any) -> str:
    number = _finite(value)
    return "—" if number is None else f"{100.0 * number:.2f}%"


def _markdown_report(report: Mapping[str, Any]) -> str:
    strict = report["strict_metrics"]
    lines = [
        "# H800 profile-aware expansion 离线验证",
        "",
        "> CPU-only shadow diagnostic；未修改或发布生产预测器。",
        "",
        "## 数据审计",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        f"| 训练配置 | {report['data_audit']['training_configurations']} |",
        f"| 训练成功标签 | {report['data_audit']['training_successes']} |",
        f"| 训练独立源 | {report['data_audit']['training_sources']} |",
        f"| 严格验证配置 | {report['data_audit']['strict_configurations']} |",
        f"| 严格验证成功标签 | {report['data_audit']['strict_successes']} |",
        f"| 严格验证独立源 | {report['data_audit']['strict_sources']} |",
        "",
        "## 训练 LOSO 选择",
        "",
        "| 候选 | 特征集 | 权重 | alpha | Source-equal expansion MAPE |",
        "|---|---|---|---:|---:|",
    ]
    for label, key in (("M1-only", "best_m1_only"), ("Profile", "best_profile")):
        row = report["training_loso_selection"][key]
        lines.append(
            f"| {label} | {row['feature_set']} | {row['weighting']} | "
            f"{row['alpha']} | "
            f"{_percent(row['training_loso_metrics']['source_equal_mape'])} |"
        )
    lines.extend(
        [
            "",
            "## 严格未使用集 reserved 中心",
            "",
            "| 范围 | 解析基准 | 当前 V5 M1 | M1 expansion | Profile expansion | Profile expansion（oracle allocated） |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scope, label in (
        ("all_success", "全部成功"),
        ("critical_lora", "Critical LoRA"),
        ("other_lora", "其他 LoRA"),
        ("supported_full", "Supported FULL"),
    ):
        row = strict[scope]
        lines.append(
            f"| {label} | "
            f"{_percent(row['analytic_reference']['source_equal_mape'])} | "
            f"{_percent(row['current_v5_m1_reserved']['source_equal_mape'])} | "
            f"{_percent(row['m1_expansion_end_to_end']['source_equal_mape'])} | "
            f"{_percent(row['profile_expansion_end_to_end']['source_equal_mape'])} | "
            f"{_percent(row['profile_expansion_oracle_allocated']['source_equal_mape'])} |"
        )
    collision = report["collision_audit"]
    lines.extend(
        [
            "",
            "## 特征碰撞",
            "",
            "| 特征口径 | 成功样本 | 唯一特征向量 | 碰撞组 | 碰撞行 | 最坏 expansion max/min |",
            "|---|---:|---:|---:|---:|---:|",
            (
                f"| M1 | {collision['m1']['success_rows']} | "
                f"{collision['m1']['unique_feature_vectors']} | "
                f"{collision['m1']['collision_groups']} | "
                f"{collision['m1']['collision_rows']} | "
                f"{collision['m1']['worst_expansion_max_over_min']:.3f} |"
            ),
            (
                f"| Profile | {collision['profile']['success_rows']} | "
                f"{collision['profile']['unique_feature_vectors']} | "
                f"{collision['profile']['collision_groups']} | "
                f"{collision['profile']['collision_rows']} | "
                f"{collision['profile']['worst_expansion_max_over_min']:.3f} |"
            ),
            "",
            "## 解释边界",
            "",
            "- Profile 特征全部来自运行前可获得的队列/profile；逐 step event 仅用于诊断。",
            "- OOM 是右删失数据，不作为中心回归标签。",
            "- 严格验证只有 5 个独立源，本结果用于决定下一轮实验，不构成生产发布证明。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--theory-basis", type=Path, default=DEFAULT_THEORY_BASIS)
    parser.add_argument(
        "--dataset-analysis", type=Path, default=DEFAULT_DATASET_ANALYSIS
    )
    parser.add_argument("--new-queue", type=Path, default=DEFAULT_NEW_QUEUE)
    parser.add_argument("--current-old", type=Path, default=DEFAULT_CURRENT_OLD)
    parser.add_argument("--stage2-queue", type=Path, default=DEFAULT_STAGE2_QUEUE)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_VALIDATION_INVENTORY)
    parser.add_argument("--hardware", type=Path, default=DEFAULT_HARDWARE)
    parser.add_argument(
        "--frozen-baseline", type=Path, default=DEFAULT_FROZEN_BASELINE
    )
    parser.add_argument(
        "--current-candidate", type=Path, default=DEFAULT_CURRENT_CANDIDATE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    print("step 1/5: rebuilding split and extracting profile features", flush=True)
    data = _build_data(args)
    training = list(data["training"])
    strict = list(data["strict"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_rows = [
        *[_record_summary(row, "training") for row in training],
        *[_record_summary(row, "strict_unused_dataset") for row in strict],
    ]
    write_jsonl(args.output_dir / "profile_features.jsonl", profile_rows)

    print("step 2/5: reconstructing retained per-step event trajectories", flush=True)
    trajectory_steps, trajectory_audit = _trajectory_rows(
        {"training": training, "strict_unused_dataset": strict}
    )
    write_jsonl(args.output_dir / "event_trajectory_steps.jsonl", trajectory_steps)
    write_json(args.output_dir / "event_trajectory_audit.json", trajectory_audit)

    print("step 3/5: selecting and fitting expansion heads by training LOSO", flush=True)
    selection = _select_candidates(training)
    best_m1 = _fit_selected(training, selection["best_m1_only"])
    best_profile = _fit_selected(training, selection["best_profile"])
    write_json(
        args.output_dir / "expansion_models.json",
        {"best_m1_only": best_m1, "best_profile": best_profile},
    )
    write_json(args.output_dir / "training_loso_selection.json", selection)

    print("step 4/5: scoring the untouched strict validation sources", flush=True)
    current_candidate = read_json(args.current_candidate)
    current_bundle = current_candidate["model"]
    strict_predictions, strict_metrics = _evaluate_strict(
        strict,
        current_bundle=current_bundle,
        m1_expansion_model=best_m1,
        profile_expansion_model=best_profile,
    )
    current_mape = strict_metrics["all_success"]["current_v5_m1_reserved"][
        "source_equal_mape"
    ]
    if not math.isclose(float(current_mape), 0.15432226675609423, abs_tol=1e-12):
        raise ValueError(
            f"current V5 strict metric drifted: {current_mape} != 0.15432226675609423"
        )
    write_jsonl(args.output_dir / "strict_predictions.jsonl", strict_predictions)

    print("step 5/5: auditing feature collisions and writing report", flush=True)
    collision = {
        "m1": _collision_audit(strict, FEATURE_SETS["m1_expansion_baseline"]),
        "profile": _collision_audit(strict, selection["best_profile"]["feature_names"]),
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "shadow_offline_validation_complete",
        "publishable": False,
        "production_model_mutated": False,
        "protocol": {
            "ordered_steps": [
                "extract_existing_profile_features",
                "reconstruct_retained_step_memory_trajectories",
                "fit_log_reserved_over_allocated_expansion_head",
                "compare_on_strict_unused_dataset_sources",
                "audit_feature_collisions",
            ],
            "model_selection": selection["selection_protocol"],
            "event_trajectory_use": "diagnostic_only_not_a_pre_run_model_feature",
            "oom_treatment": "right-censored; excluded from center fitting and MAPE",
        },
        "input_bindings": {
            "current_candidate": {
                "path": str(args.current_candidate),
                "sha256": sha256_file(args.current_candidate),
            },
            "stage2_queue": {
                "path": str(args.stage2_queue),
                "sha256": sha256_file(args.stage2_queue),
            },
        },
        "data_audit": {
            "training_configurations": len(training),
            "training_successes": len(_successful(training)),
            "training_oom": sum(_outcome(row) == "oom" for row in training),
            "training_sources": len({_source_id(row) for row in training}),
            "strict_configurations": len(strict),
            "strict_successes": len(_successful(strict)),
            "strict_oom": sum(_outcome(row) == "oom" for row in strict),
            "strict_sources": len({_source_id(row) for row in strict}),
            "trajectory": trajectory_audit,
            "base_training": data["base_training_audit"],
            "stage2_collapse": data["stage2_collapse"],
        },
        "feature_definitions": {
            "profile_features": list(PROFILE_FEATURES),
            "bounded_cv": "min(1, max(0, coefficient_of_variation / 2))",
            "rare_tail_gap": (
                "maximum_clipped_fraction - expected_random_batch_max_fraction"
            ),
            "pre_run_available_only": True,
        },
        "training_loso_selection": {
            "selection_protocol": selection["selection_protocol"],
            "best_overall": selection["best_overall"],
            "best_m1_only": selection["best_m1_only"],
            "best_profile": selection["best_profile"],
            "candidate_count": len(selection["candidates"]),
        },
        "fitted_models": {
            "best_m1_only": best_m1,
            "best_profile": best_profile,
        },
        "strict_metrics": strict_metrics,
        "collision_audit": collision,
        "limitations": [
            "strict validation contains only five independent dataset sources",
            "actual per-microbatch maximum sequence and sampler order were not recorded",
            "retained step trajectories are post-run diagnostics and cannot be used at admission time",
            "this diagnostic does not recalibrate a P95 upper bound or admission head",
        ],
    }
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "report.md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    print(json.dumps(report["training_loso_selection"], indent=2), flush=True)
    print(json.dumps(strict_metrics, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
