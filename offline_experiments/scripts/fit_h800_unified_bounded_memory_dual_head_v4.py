#!/usr/bin/env python3
"""Fit and validate a two-head successor to unified bounded memory v3.

The model remains one end-to-end predictor.  Its shared input features feed:

* an allocated-memory head for ``log(allocated / analytic_reference)``;
* a positive composition-residual head for
  ``log(reserved / predicted_allocated)``.

Inference composes the heads as ``reserved = allocated * exp(max(0, gap))``.
The positive composition guarantees that predicted reserved memory cannot be
smaller than predicted allocated memory.  The existing independent censored
OOM risk head remains the admission guard.

The frozen v3 artifact is never overwritten.  Model selection uses only the
existing development sources.  The four completed full-coverage business runs
from peak-replay stage one remain out of fit and are evaluated prospectively.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import benchmark_h800_memory_center_models_v1 as bm
import fit_h800_unified_resource_partial_v1 as base
import freeze_h800_unified_bounded_memory_v3 as v3_freeze
import h800_unified_bounded_memory_v3_data as v3_data
from common import ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json, write_jsonl
from h800_unified_bounded_memory_model import (
    ARTIFACT_SCHEMA_V4,
    load_artifact,
    predict_records,
)

SCHEMA = "sft_h800_unified_bounded_memory_dual_head_fit/v4"
IMPLEMENTATION_VERSION = (
    "sft_h800_unified_bounded_memory/"
    "2026-08-10.v4.1-allocated-positive-composition-residual-independent-risk"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "diagnostics" / "h800_unified_bounded_memory_v4_residual_20260810"
)
DEFAULT_ARTIFACT = (
    ROOT / "artifacts" / "h800_unified_bounded_memory_candidate_v4_residual.json"
)
V3_ARTIFACT = ROOT / "artifacts" / "h800_unified_bounded_memory_candidate_v3.json"
STAGE1_QUEUE = ROOT / "matrix" / "h800_final_memory_peak_replay_jobs_v1.jsonl"
STAGE1_RESULTS = ROOT / "artifacts" / "h800_final_memory_peak_replay_results_v1.json"
OUTER_FOLDS = 10
INNER_FOLDS = 5


def _candidate(
    candidate_id: str,
    *,
    basis_kind: str,
    alpha: float,
    shrinkage: float,
    source_weight_power: float = 0.25,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "basis_kind": basis_kind,
        "feature_variant": "anchor_scale_v3_mechanisms",
        "alpha": alpha,
        "correction_shrinkage": shrinkage,
        "huber_delta": 0.2,
        "source_weight_power": source_weight_power,
        "censored_constraint_weight": 1.0,
    }


ALLOCATED_CANDIDATES = (
    _candidate(
        "allocated_bounded_mechanism_a0p03_s0p8",
        basis_kind="bounded_mechanism",
        alpha=0.03,
        shrinkage=0.8,
    ),
    _candidate(
        "allocated_bounded_mechanism_a0p1_s0p8",
        basis_kind="bounded_mechanism",
        alpha=0.1,
        shrinkage=0.8,
    ),
    _candidate(
        "allocated_bounded_linear_a0p1_s0p8",
        basis_kind="bounded_linear",
        alpha=0.1,
        shrinkage=0.8,
    ),
)

GAP_CANDIDATES = (
    _candidate(
        "gap_bounded_mechanism_a0p03_s1",
        basis_kind="bounded_mechanism",
        alpha=0.03,
        shrinkage=1.0,
    ),
    _candidate(
        "gap_bounded_mechanism_a0p1_s1",
        basis_kind="bounded_mechanism",
        alpha=0.1,
        shrinkage=1.0,
    ),
    _candidate(
        "gap_bounded_mechanism_a0p3_s1",
        basis_kind="bounded_mechanism",
        alpha=0.3,
        shrinkage=1.0,
    ),
    _candidate(
        "gap_bounded_linear_a0p1_s1",
        basis_kind="bounded_linear",
        alpha=0.1,
        shrinkage=1.0,
    ),
    _candidate(
        "gap_bounded_linear_a0p3_s1",
        basis_kind="bounded_linear",
        alpha=0.3,
        shrinkage=1.0,
    ),
    _candidate(
        "gap_bounded_linear_a1_s1",
        basis_kind="bounded_linear",
        alpha=1.0,
        shrinkage=1.0,
    ),
)


def _exact_target_records(
    records: Sequence[Mapping[str, Any]],
    *,
    target: str,
    allocated_model: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    exact_rows = [row for row in records if row["state"] == "exact"]
    inferred_allocated: dict[str, float] = {}
    if target == "allocator_gap":
        if allocated_model is None:
            raise ValueError(
                "allocator-gap residual targets need an allocated model"
            )
        corrections = bm._predict_correction(exact_rows, allocated_model)
        inferred_allocated = {
            str(row["record_id"]): float(row["reference_bytes"])
            * math.exp(float(correction))
            for row, correction in zip(exact_rows, corrections)
        }
    transformed: list[dict[str, Any]] = []
    for source in records:
        if source["state"] != "exact":
            continue
        allocated = source.get("target_allocated_bytes")
        reserved = source.get("target_reserved_bytes")
        if reserved is None:
            raise ValueError(f"exact row lacks reserved target: {source['record_id']}")
        if target == "allocated" and allocated is None:
            continue
        reserved_value = float(reserved)
        if allocated is not None and not (0.0 < float(allocated) <= reserved_value):
            raise ValueError(
                f"invalid allocated/reserved order for {source['record_id']}: "
                f"{float(allocated)} > {reserved_value}"
            )
        row = copy.deepcopy(dict(source))
        row["state"] = "exact"
        row["censor_lower_bytes"] = None
        if target == "allocated":
            row["target_reserved_bytes"] = float(allocated)
        elif target == "allocator_gap":
            # bm._fit_model learns log(target/reference).  This transformed
            # target learns log(reserved/predicted_allocated).  Training the
            # second head on the first head's residual makes the composed
            # reserved center, rather than two isolated heads, the objective.
            allocated_prediction = inferred_allocated[str(source["record_id"])]
            row["target_reserved_bytes"] = float(row["reference_bytes"]) * (
                max(1.0, reserved_value / allocated_prediction)
            )
            row["allocator_gap_target_kind"] = (
                "composition_residual_with_allocated_label"
                if allocated is not None
                else "composition_residual_reserved_only_partial_label"
            )
        else:
            raise ValueError(f"unknown dual-head target: {target}")
        transformed.append(row)
    if len(transformed) < 2:
        raise ValueError("dual-head fit needs at least two exact rows")
    return transformed


def _fit_dual_model(
    records: Sequence[Mapping[str, Any]],
    allocated_candidate: Mapping[str, Any],
    gap_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    allocated_training = _exact_target_records(records, target="allocated")
    allocated_model = bm._fit_model(allocated_training, allocated_candidate)
    gap_training = _exact_target_records(
        records,
        target="allocator_gap",
        allocated_model=allocated_model,
    )
    gap_model = bm._fit_model(gap_training, gap_candidate)
    allocated_model["target"] = "log(max_allocated_bytes / analytic_reference_bytes)"
    gap_model["target"] = (
        "log(max_reserved_bytes / allocated_head_prediction_bytes)"
    )
    return {
        "kind": "allocated_plus_positive_allocator_gap",
        "model_family": "one_shared_input_two_center_heads",
        "composition": (
            "reserved=allocated*exp(max(0,predicted_log_reserved_over_allocated))"
        ),
        "allocated_candidate": dict(allocated_candidate),
        "allocator_gap_candidate": dict(gap_candidate),
        "allocated_model": allocated_model,
        "allocator_gap_model": gap_model,
        "partial_label_contract": {
            "allocated_observed_rows": len(allocated_training),
            "allocator_gap_rows": len(gap_training),
            "reserved_only_gap_rows": sum(
                row.get("allocator_gap_target_kind")
                == "composition_residual_reserved_only_partial_label"
                for row in gap_training
            ),
        },
    }


def _predict_dual_center(
    records: Sequence[Mapping[str, Any]], model: Mapping[str, Any]
) -> list[dict[str, float]]:
    allocated_corrections = bm._predict_correction(records, model["allocated_model"])
    gap_corrections = bm._predict_correction(records, model["allocator_gap_model"])
    predictions = []
    for row, allocated_correction, raw_gap in zip(
        records, allocated_corrections, gap_corrections
    ):
        allocated = float(row["reference_bytes"]) * math.exp(float(allocated_correction))
        gap_log_ratio = max(0.0, float(raw_gap))
        reserved = allocated * math.exp(gap_log_ratio)
        predictions.append(
            {
                "allocated_center_bytes": allocated,
                "allocator_gap_log_ratio": gap_log_ratio,
                "allocator_gap_bytes": reserved - allocated,
                "reserved_center_bytes": reserved,
            }
        )
    return predictions


def _safe_limit() -> float:
    return v3_freeze.SAFE_LIMIT_FRACTION * bm.DEVICE_CAPACITY_BYTES


def _score(
    records: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    *,
    fold_id: str,
    risk_model: Mapping[str, Any] | None = None,
    risk_multiplier: float | None = None,
) -> list[dict[str, Any]]:
    centers = _predict_dual_center(records, model)
    risks = (
        v3_freeze._model_bytes(records, risk_model)
        if risk_model is not None
        else [None] * len(records)
    )
    details = []
    for row, center, risk in zip(records, centers, risks):
        reserved_prediction = float(center["reserved_center_bytes"])
        upper = (
            max(reserved_prediction, float(risk) * float(risk_multiplier))
            if risk is not None and risk_multiplier is not None
            else None
        )
        detail: dict[str, Any] = {
            "record_id": str(row["record_id"]),
            "source_id": str(row["source_id"]),
            "origin": str(row["origin"]),
            "role": str(row["role"]),
            "state": str(row["state"]),
            "model_id": str(row["model_id"]),
            "train_type": str(row["train_type"]),
            "gpu_count": int(row["gpu_count"]),
            "zero_stage": int(row["zero_stage"]),
            "gc": bool(row["gc"]),
            "packing": bool(row["packing"]),
            "mbs": int(row["mbs"]),
            "cutoff_len": int(row["cutoff_len"]),
            "reference_bytes": float(row["reference_bytes"]),
            **center,
            "predicted_reserved_bytes": reserved_prediction,
            "predicted_reserved_gib": reserved_prediction / (1 << 30),
            "center_bytes": reserved_prediction,
            "risk_head_bytes": risk,
            "risk_upper_multiplier": risk_multiplier,
            "admission_upper_bytes": upper,
            "safe_limit_bytes": _safe_limit(),
            "admitted": upper <= _safe_limit() if upper is not None else None,
            "fold_id": fold_id,
        }
        if row["state"] == "exact":
            observed_reserved = float(row["target_reserved_bytes"])
            detail.update(
                {
                    "observed_reserved_bytes": observed_reserved,
                    "observed_reserved_gib": observed_reserved / (1 << 30),
                    "absolute_percentage_error": abs(
                        reserved_prediction / observed_reserved - 1.0
                    ),
                    "signed_percentage_error": (
                        reserved_prediction / observed_reserved - 1.0
                    ),
                }
            )
            if row.get("target_allocated_bytes") is not None:
                observed_allocated = float(row["target_allocated_bytes"])
                detail.update(
                    {
                        "observed_allocated_bytes": observed_allocated,
                        "allocated_absolute_percentage_error": abs(
                            float(center["allocated_center_bytes"])
                            / observed_allocated
                            - 1.0
                        ),
                        "allocated_signed_percentage_error": (
                            float(center["allocated_center_bytes"])
                            / observed_allocated
                            - 1.0
                        ),
                    }
                )
        else:
            lower = float(row["censor_lower_bytes"])
            detail.update(
                {
                    "censor_lower_bytes": lower,
                    "censor_satisfied": reserved_prediction >= lower,
                    "censor_shortfall_fraction": max(
                        0.0, 1.0 - reserved_prediction / lower
                    ),
                }
            )
        details.append(detail)
    return details


def _error_metrics(
    details: Sequence[Mapping[str, Any]], *, allocated: bool = False
) -> dict[str, Any]:
    exact = [
        row
        for row in details
        if row["state"] == "exact"
        and (not allocated or row.get("observed_allocated_bytes") is not None)
    ]
    error_key = (
        "allocated_absolute_percentage_error"
        if allocated
        else "absolute_percentage_error"
    )
    signed_key = (
        "allocated_signed_percentage_error"
        if allocated
        else "signed_percentage_error"
    )
    by_source: dict[str, list[float]] = defaultdict(list)
    for row in exact:
        by_source[str(row["source_id"])].append(float(row[error_key]))
    errors = [float(row[error_key]) for row in exact]
    signed = [float(row[signed_key]) for row in exact]
    return {
        "exact_centres": len(exact),
        "exact_sources": len(by_source),
        "row_mape": statistics.fmean(errors) if errors else None,
        "source_equal_mape": (
            statistics.fmean(statistics.fmean(values) for values in by_source.values())
            if by_source
            else None
        ),
        "median_ape": base._percentile(errors, 0.5),
        "p90_ape": base._percentile(errors, 0.9),
        "max_ape": max(errors) if errors else None,
        "signed_bias": statistics.fmean(signed) if signed else None,
    }


def _admission_metrics(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    safe = [
        row
        for row in details
        if row["state"] == "exact"
        and float(row["observed_reserved_bytes"]) <= _safe_limit()
    ]
    unsafe = [
        row
        for row in details
        if row["state"] == "exact"
        and float(row["observed_reserved_bytes"]) > _safe_limit()
    ]
    oom = [row for row in details if row["state"] == "censored"]
    admitted_safe = sum(bool(row["admitted"]) for row in safe)
    admitted_oom = sum(bool(row["admitted"]) for row in oom)
    return {
        "actual_safe_success_rows": len(safe),
        "admitted_safe_success_rows": admitted_safe,
        "safe_success_admission_rate": admitted_safe / len(safe) if safe else None,
        "actual_unsafe_success_rows": len(unsafe),
        "admitted_unsafe_success_rows": sum(bool(row["admitted"]) for row in unsafe),
        "oom_rows": len(oom),
        "admitted_oom_rows": admitted_oom,
        "oom_admission_rate": admitted_oom / len(oom) if oom else None,
    }


def _pair_key(
    allocated_candidate: Mapping[str, Any], gap_candidate: Mapping[str, Any]
) -> str:
    return f"{allocated_candidate['candidate_id']}__{gap_candidate['candidate_id']}"


def _rank_pairs(
    records: Sequence[Mapping[str, Any]], *, fold_count: int
) -> list[dict[str, Any]]:
    folds = bm._source_folds(records, fold_count)
    predictions: dict[str, list[dict[str, Any]]] = {
        _pair_key(allocated, gap): []
        for allocated in ALLOCATED_CANDIDATES
        for gap in GAP_CANDIDATES
    }
    for fold_index, held_sources in enumerate(folds):
        training = [row for row in records if str(row["source_id"]) not in held_sources]
        evaluation = [row for row in records if str(row["source_id"]) in held_sources]
        allocated_models = {
            str(candidate["candidate_id"]): bm._fit_model(
                _exact_target_records(training, target="allocated"), candidate
            )
            for candidate in ALLOCATED_CANDIDATES
        }
        for allocated_candidate in ALLOCATED_CANDIDATES:
            allocated_model = allocated_models[str(allocated_candidate["candidate_id"])]
            gap_training = _exact_target_records(
                training,
                target="allocator_gap",
                allocated_model=allocated_model,
            )
            gap_models = {
                str(candidate["candidate_id"]): bm._fit_model(gap_training, candidate)
                for candidate in GAP_CANDIDATES
            }
            for gap_candidate in GAP_CANDIDATES:
                model = {
                    "allocated_model": allocated_model,
                    "allocator_gap_model": gap_models[
                        str(gap_candidate["candidate_id"])
                    ],
                }
                key = _pair_key(allocated_candidate, gap_candidate)
                predictions[key].extend(
                    _score(evaluation, model, fold_id=f"pair_{fold_index:02d}")
                )
    ranking = []
    for allocated_candidate in ALLOCATED_CANDIDATES:
        for gap_candidate in GAP_CANDIDATES:
            key = _pair_key(allocated_candidate, gap_candidate)
            details = predictions[key]
            ranking.append(
                {
                    "pair_id": key,
                    "allocated_candidate": dict(allocated_candidate),
                    "allocator_gap_candidate": dict(gap_candidate),
                    "reserved_metrics": _error_metrics(details),
                    "allocated_metrics": _error_metrics(details, allocated=True),
                }
            )
    return sorted(
        ranking,
        key=lambda row: (
            float(row["reserved_metrics"]["source_equal_mape"]),
            float(row["reserved_metrics"]["p90_ape"]),
            float(row["allocated_metrics"]["source_equal_mape"]),
            str(row["pair_id"]),
        ),
    )


def _nested_predictions(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = []
    audits = []
    for outer_index, held_sources in enumerate(bm._source_folds(records, OUTER_FOLDS)):
        print(f"dual-head nested outer fold {outer_index + 1}/{OUTER_FOLDS}", flush=True)
        training = [row for row in records if str(row["source_id"]) not in held_sources]
        evaluation = [row for row in records if str(row["source_id"]) in held_sources]
        selected = _rank_pairs(training, fold_count=INNER_FOLDS)[0]
        model = _fit_dual_model(
            training,
            selected["allocated_candidate"],
            selected["allocator_gap_candidate"],
        )
        risk_model = bm._fit_model(training, v3_freeze.RISK_CANDIDATE)
        risk_multiplier = v3_freeze._calibrate_risk_multiplier(
            training, v3_freeze._model_bytes(training, risk_model)
        )
        outer = _score(
            evaluation,
            model,
            fold_id=f"outer_{outer_index:02d}",
            risk_model=risk_model,
            risk_multiplier=risk_multiplier,
        )
        predictions.extend(outer)
        audits.append(
            {
                "outer_fold": outer_index,
                "held_out_sources": sorted(held_sources),
                "selected_pair_id": selected["pair_id"],
                "inner_reserved_metrics": selected["reserved_metrics"],
                "outer_reserved_metrics": _error_metrics(outer),
                "outer_allocated_metrics": _error_metrics(outer, allocated=True),
                "outer_admission_metrics": _admission_metrics(outer),
                "risk_multiplier": risk_multiplier,
            }
        )
    return predictions, audits


def _stage1_full_coverage_records() -> list[dict[str, Any]]:
    jobs = [
        row
        for row in read_jsonl(STAGE1_QUEUE)
        if row.get("replay_mode") == "full_coverage"
    ]
    results = read_json(STAGE1_RESULTS)
    observations = {
        str(row["job_id"]): row for row in results.get("observations") or []
    }
    if len(jobs) != 4 or results.get("terminal_counts") != {"success": 8}:
        raise ValueError("stage-one full-coverage prospective evidence is incomplete")
    inventory, models, capacity = v3_data._inventory()
    corrected_models = copy.deepcopy(models)
    for model_id, parameters in v3_data.RUNTIME_BASE_PARAMETERS.items():
        corrected_models[model_id]["actual_parameters"] = parameters
    profile_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    records = []
    for job in jobs:
        job = copy.deepcopy(job)
        if str(job["model_id"]) in v3_data.RUNTIME_BASE_PARAMETERS:
            job["model_parameters"] = v3_data.RUNTIME_BASE_PARAMETERS[
                str(job["model_id"])
            ]
        reference, features = base._current_features(
            job,
            model_by_id=corrected_models,
            fixed_lora=inventory["fixed_lora"],
            capacity_bytes=capacity,
            profile_cache=profile_cache,
        )
        observed = observations[str(job["job_id"])]
        if observed.get("classification") != "success":
            raise ValueError(f"stage-one full job is not successful: {job['job_id']}")
        records.append(
            {
                "record_id": f"stage1_full::{job['job_id']}",
                "source_id": str(job["source_dataset_id"]),
                "origin": "final_memory_peak_replay_stage1_full_coverage",
                "role": "prospective_structure_challenger_validation_only",
                "state": "exact",
                "reference_bytes": reference,
                "target_reserved_bytes": float(observed["max_reserved_bytes"]),
                "target_allocated_bytes": float(observed["max_allocated_bytes"]),
                "censor_lower_bytes": None,
                "features": features,
                "model_id": str(job["model_id"]),
                "train_type": str(job["train_type"]),
                "gpu_count": int(job["gpu_count"]),
                "zero_stage": int(job["zero_stage"]),
                "gc": bool(job["gc"]),
                "mbs": int(job["mbs"]),
                "cutoff_len": int(job["cutoff_len"]),
                "packing": bool(job["packing"]),
                "profile_sha256": str(job["dataset_profile_sha256"]),
            }
        )
    return records


def _v3_prospective_details(
    records: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]
) -> list[dict[str, Any]]:
    predictions = predict_records(records, artifact)
    details = []
    for row, prediction in zip(records, predictions):
        observed = float(row["target_reserved_bytes"])
        predicted = float(prediction["center_bytes"])
        details.append(
            {
                "record_id": str(row["record_id"]),
                "source_id": str(row["source_id"]),
                "state": "exact",
                "observed_reserved_bytes": observed,
                "predicted_reserved_bytes": predicted,
                "absolute_percentage_error": abs(predicted / observed - 1.0),
                "signed_percentage_error": predicted / observed - 1.0,
                "admitted": bool(prediction["admitted"]),
            }
        )
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()

    print("loading frozen v3 development records", flush=True)
    records, data_audit = v3_data.development_records()
    exact = [row for row in records if row["state"] == "exact"]
    allocated_exact = [
        row for row in exact if row.get("target_allocated_bytes") is not None
    ]
    reserved_only_exact = [
        row for row in exact if row.get("target_allocated_bytes") is None
    ]
    if len(allocated_exact) < 2:
        raise ValueError("fewer than two development rows have allocated targets")
    print(
        "partial-label audit: "
        f"exact={len(exact)}, allocated+reserved={len(allocated_exact)}, "
        f"reserved-only={len(reserved_only_exact)}",
        flush=True,
    )

    print("ranking fixed dual-head candidate pairs on development source folds", flush=True)
    ranking = _rank_pairs(records, fold_count=OUTER_FOLDS)
    selected = ranking[0]
    print(f"selected dual-head pair: {selected['pair_id']}", flush=True)

    nested, nested_audit = _nested_predictions(records)
    baseline, baseline_fold_audit = v3_freeze._nested_predictions(records)
    baseline_reserved = bm._metrics(baseline)
    baseline_admission = v3_freeze._admission_metrics(baseline)
    nested_reserved = _error_metrics(nested)
    nested_allocated = _error_metrics(nested, allocated=True)
    nested_admission = _admission_metrics(nested)

    dual_model = _fit_dual_model(
        records,
        selected["allocated_candidate"],
        selected["allocator_gap_candidate"],
    )
    risk_model = bm._fit_model(records, v3_freeze.RISK_CANDIDATE)
    risk_multiplier = v3_freeze._calibrate_risk_multiplier(
        records, v3_freeze._model_bytes(records, risk_model)
    )
    prospective_records = _stage1_full_coverage_records()
    prospective = _score(
        prospective_records,
        dual_model,
        fold_id="prospective_stage1_full_coverage",
        risk_model=risk_model,
        risk_multiplier=risk_multiplier,
    )
    old_artifact = load_artifact(V3_ARTIFACT)
    old_prospective = _v3_prospective_details(prospective_records, old_artifact)
    prospective_new_metrics = _error_metrics(prospective)
    prospective_old_metrics = bm._metrics(old_prospective)
    prospective_admission = _admission_metrics(prospective)

    gates = {
        "nested_reserved_source_equal_mape_better_than_v3": float(
            nested_reserved["source_equal_mape"]
        )
        < float(baseline_reserved["source_equal_mape"]),
        "nested_safe_admission_at_least_v3": float(
            nested_admission["safe_success_admission_rate"]
        )
        >= float(baseline_admission["safe_success_admission_rate"]),
        "nested_zero_oom_admission": nested_admission["admitted_oom_rows"] == 0,
        "prospective_reserved_mape_better_than_v3": float(
            prospective_new_metrics["row_mape"]
        )
        < float(prospective_old_metrics["row_mape"]),
        "prospective_all_four_safe_jobs_admitted": (
            prospective_admission["admitted_safe_success_rows"] == 4
        ),
    }
    generated = datetime.now(timezone.utc).isoformat()
    artifact: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA_V4,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": generated,
        "status": "frozen_shadow_candidate_waiting_validation",
        "immutable": True,
        "publishable": False,
        "production_override_allowed": False,
        "production_model_mutated": False,
        "gpu_family": "H800",
        "hardware_domain": {
            "capacity_bytes": bm.DEVICE_CAPACITY_BYTES,
            "safe_limit_bytes": _safe_limit(),
        },
        "candidate": {
            "allocated": selected["allocated_candidate"],
            "allocator_gap": selected["allocator_gap_candidate"],
            "risk": dict(v3_freeze.RISK_CANDIDATE),
        },
        "model": dual_model,
        "admission": {
            "kind": "independent_shared_risk_head",
            "rule": (
                "admit iff max(reserved_center_bytes, risk_head_bytes * "
                "upper_multiplier) <= 0.95 * capacity_bytes"
            ),
            "risk_model": risk_model,
            "upper_multiplier": risk_multiplier,
            "safe_limit_fraction": v3_freeze.SAFE_LIMIT_FRACTION,
            "calibration": (
                "final-fit right-censored risk-head self-consistency; center heads "
                "fit exact rows only"
            ),
        },
        "development_acceptance": {
            "dual_nested_reserved_metrics": nested_reserved,
            "dual_nested_allocated_metrics": nested_allocated,
            "dual_nested_admission_metrics": nested_admission,
            "v3_nested_reserved_metrics": baseline_reserved,
            "v3_nested_admission_metrics": baseline_admission,
            "gates": gates,
            "all_passed": all(gates.values()),
        },
        "evidence_contract": {
            "development_rows": len(records),
            "development_exact": len(exact),
            "development_allocated_and_reserved_exact": len(allocated_exact),
            "development_reserved_only_exact": len(reserved_only_exact),
            "development_right_censored": len(records) - len(exact),
            "stage1_full_coverage_rows_used_for_fit": 0,
            "stage1_full_coverage_rows_used_for_prospective_comparison": 4,
            "final_acceptance_rows_used": 0,
        },
        "inputs": {
            "v3_artifact": {
                "path": str(V3_ARTIFACT.resolve()),
                "sha256": sha256_file(V3_ARTIFACT),
            },
            "stage1_queue": {
                "path": str(STAGE1_QUEUE.resolve()),
                "sha256": sha256_file(STAGE1_QUEUE),
            },
            "stage1_results": {
                "path": str(STAGE1_RESULTS.resolve()),
                "sha256": sha256_file(STAGE1_RESULTS),
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
        },
    }
    artifact["artifact_sha256"] = sha256_json(artifact)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": generated,
        "status": (
            "dual_head_challenger_passed_offline_gates"
            if all(gates.values())
            else "dual_head_challenger_failed_one_or_more_offline_gates"
        ),
        "production_model_mutated": False,
        "data_audit": data_audit,
        "development_rows": len(records),
        "development_sources": len({str(row["source_id"]) for row in records}),
        "development_exact": len(exact),
        "development_allocated_and_reserved_exact": len(allocated_exact),
        "development_reserved_only_exact": len(reserved_only_exact),
        "development_right_censored": len(records) - len(exact),
        "candidate_grid": {
            "allocated_candidates": len(ALLOCATED_CANDIDATES),
            "allocator_gap_candidates": len(GAP_CANDIDATES),
            "pairs": len(ALLOCATED_CANDIDATES) * len(GAP_CANDIDATES),
            "ranking": ranking,
            "selected_pair_id": selected["pair_id"],
        },
        "nested_source_cv": {
            "protocol": (
                "10 source-grouped outer folds; each outer training split selects "
                "the two-head pair on 5 inner source folds; risk calibrated only "
                "inside the outer training split"
            ),
            "dual_reserved_metrics": nested_reserved,
            "dual_allocated_metrics": nested_allocated,
            "dual_admission_metrics": nested_admission,
            "v3_reserved_metrics": baseline_reserved,
            "v3_admission_metrics": baseline_admission,
            "dual_folds": nested_audit,
            "v3_folds": baseline_fold_audit,
        },
        "prospective_stage1_full_coverage": {
            "fit_rows_from_this_evidence": 0,
            "rows": 4,
            "dual_reserved_metrics": prospective_new_metrics,
            "dual_allocated_metrics": _error_metrics(prospective, allocated=True),
            "dual_admission_metrics": prospective_admission,
            "v3_reserved_metrics": prospective_old_metrics,
            "oom_admission_rate_is_measurable": False,
            "dual_predictions": prospective,
            "v3_predictions": old_prospective,
        },
        "gates": gates,
        "all_offline_gates_passed": all(gates.values()),
        "artifact": {
            "path": str(args.artifact.resolve()),
            "artifact_sha256": artifact["artifact_sha256"],
        },
    }
    report["report_sha256"] = sha256_json(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "nested_dual_predictions.jsonl", nested)
    write_jsonl(args.output_dir / "prospective_dual_predictions.jsonl", prospective)
    write_json(args.artifact, artifact)
    write_json(args.output_dir / "fit_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
