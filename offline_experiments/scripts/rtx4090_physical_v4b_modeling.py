#!/usr/bin/env python3
"""Fit and replay RTX 4090 physical-shares memory plus v4b throughput.

The split is intentionally temporal and scenario-clean where possible:

* memory is fit only on the original ``memory_probe`` population;
* throughput is fit on pre-screen throughput plus the short screen population;
* formal long-window throughput and packing A/B jobs are never used for fit;
* old-data estimates use nested complete-scenario CV;
* complete-model holdout is reported separately from temporal replay.

Every throughput feature is reconstructed from information available before a
run starts.  Runtime token counters are labels/audit evidence only.
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

from common import ROOT, read_json, sha256_file, sha256_json, write_json
from h800_challenger_modeling import (
    MEMORY_ALPHA_GRID,
    _fit_memory_ridge,
    _fit_memory_tail,
    _memory_features,
    _predict_memory_center,
    _predict_memory_upper,
)
from h800_theory_basis import memory_basis
from h800_theory_calibration import (
    _candidate_key,
    _observed_reserved,
    _safe_limit,
    scenario_id,
    scenario_material,
)
from joint_throughput_modeling import (
    FEATURE_NAMES,
    FEATURE_SETS,
    _evaluate_entries,
    _fit_two_head_model,
    _generic_candidate_folds,
    _model_candidate_folds,
    _pool_candidates,
    _predict_two_head_entries,
    _record_features,
    _record_log_analytic_anchor,
    _select_generic,
    _select_model_generalization,
    _select_two_head,
)
from structured_throughput_modeling import _static_structured_basis
from throughput_predictor import ThroughputPredictor


SCHEMA = "sft_rtx4090_physical_shares_v4b/v1"
IMPLEMENTATION_VERSION = (
    "sft_rtx4090_physical_shares_v4b/"
    "2026-07-29.temporal-static-work-nested-validation"
)
DEFAULT_CAMPAIGN_ROOT = ROOT / "campaigns" / "rtx4090_20260717"
DEFAULT_OUTPUT = (
    DEFAULT_CAMPAIGN_ROOT
    / "artifacts"
    / "rtx4090_physical_shares_v4b_2026-07-29.json"
)
GIB = float(1024**3)
MEMORY_FEATURE_SET = "physical_shares"
THROUGHPUT_FEATURE_SET = "full_factor"
KERNEL_PATH = "fa2+liger_fused_ce+adamw_torch_fused"


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    usable = sorted(
        float(value)
        for value in values
        if math.isfinite(float(value))
    )
    if not usable:
        return None
    position = min(100.0, max(0.0, percentile)) / 100.0
    position *= len(usable) - 1
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return usable[lower]
    fraction = position - lower
    return usable[lower] * (1.0 - fraction) + usable[upper] * fraction


def _summary(values: Sequence[float]) -> dict[str, Any]:
    usable = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]
    return {
        "count": len(usable),
        "mean": statistics.fmean(usable) if usable else None,
        "median": statistics.median(usable) if usable else None,
        "p90": _percentile(usable, 90.0),
        "max": max(usable) if usable else None,
    }


def _read_rows(campaign_root: Path) -> list[dict[str, Any]]:
    payload = read_json(
        campaign_root / "artifacts" / "collected_results.json"
    )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("RTX 4090 collected_results has no rows")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _rendered_job(
    campaign_root: Path,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    path = (
        campaign_root
        / "results"
        / str(row["job_id"])
        / "rendered_run.json"
    )
    payload = read_json(path)
    job = payload.get("job")
    if not isinstance(job, Mapping):
        raise ValueError(f"{path} has no rendered job")
    return dict(job)


def _request(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(job),
        "request_id": str(job["job_id"]),
        "hardware_id": "rtx4090",
        "dtype": "bf16",
        "kernel_path": KERNEL_PATH,
    }


def _scenario_material(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": str(job["model_id"]),
        "train_type": str(job["train_type"]),
        "dataset_id": str(job["dataset_id"]),
        "target_gbs": int(job["target_gbs"]),
    }


def _base_record(
    *,
    predictor: ThroughputPredictor,
    campaign_root: Path,
    row: Mapping[str, Any],
    input_index: int,
) -> dict[str, Any]:
    job = _rendered_job(campaign_root, row)
    normalized = predictor._normalized_request(
        _request(job),
        input_index=input_index,
    )
    hardware = normalized["hardware"]
    capacity = int(hardware.memory_bytes)
    basis_job = dict(job)
    basis_job["model_parameters"] = int(
        normalized["model_geometry"]["base_parameters"]
    )
    memory = memory_basis(
        basis_job,
        normalized["model_geometry"],
        capacity,
    )
    outcome = str(row.get("classification") or "").lower()
    allocated = float(row.get("max_allocated_bytes") or 0.0)
    reserved = float(row.get("max_reserved_bytes") or 0.0)
    if outcome == "success":
        if reserved <= 0:
            raise ValueError(
                f"{job['job_id']} success has no reserved-memory label"
            )
        memory["observed"] = {
            "kind": "exact_success_peak",
            "peak_allocated_diagnostic_bytes": (
                allocated if allocated > 0 else None
            ),
            "peak_reserved_target_bytes": reserved,
            "right_censor_lower_bytes": None,
        }
    elif outcome == "oom":
        # OOM demand is censored.  The operational fact that it lies beyond
        # the 95% admission boundary is enough for the one-sided guard.
        lower = max(
            float(memory["safe_limit_bytes"]) + 1.0,
            reserved,
        )
        memory["observed"] = {
            "kind": "right_censored_oom",
            "peak_allocated_diagnostic_bytes": (
                allocated if allocated > 0 else None
            ),
            "peak_reserved_target_bytes": None,
            "right_censor_lower_bytes": lower,
        }
    else:
        raise ValueError(f"Unsupported outcome {outcome!r}")

    material = _scenario_material(job)
    scenario = {
        **material,
        "gpu_count": int(job["gpu_count"]),
        "physical_mbs": int(job["mbs"]),
        "cutoff_len": int(job["cutoff_len"]),
    }
    return {
        "schema": "sft_rtx4090_static_record/v1",
        "observation_id": str(job["job_id"]),
        "job_id": str(job["job_id"]),
        "evidence_tier": "rtx4090_native_campaign",
        "outcome": outcome,
        "scenario": scenario,
        "scenario_id": sha256_json(material),
        "selector": {
            "runtime_cohort_id": "rtx4090_20260717_fa2",
            "dtype": "bf16",
            "kernel_path": KERNEL_PATH,
            "training_mode": str(job["train_type"]),
            "zero_stage": int(normalized["zero_stage"]),
            "gradient_checkpointing": bool(job["gc"]),
            "packing": bool(job["packing"]),
        },
        "runtime": {
            "runtime_cohort_id": "rtx4090_20260717_fa2",
        },
        "model_basis": normalized["model_geometry"],
        "memory": memory,
        "performance": None,
        "analysis_metadata": {
            "kind": str(row.get("kind") or ""),
            "fidelity": row.get("fidelity"),
            "started_unix": float(row.get("started_unix") or 0.0),
            "finished_unix": float(row.get("finished_unix") or 0.0),
            "clock_status": row.get("clock_status"),
        },
        "_normalized": normalized,
    }


def _add_static_performance(
    record: dict[str, Any],
    *,
    predictor: ThroughputPredictor,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = record.pop("_normalized")
    performance: dict[str, Any] = {
        "physical_priors": normalized["hardware"].physical_priors(),
    }
    if normalized["gradient_accumulation_steps"] is not None:
        performance["gradient_accumulation_steps"] = normalized[
            "gradient_accumulation_steps"
        ]
    record["performance"] = performance
    basis = _static_structured_basis(
        record,
        predictor.profiles,
        hardware_memory_bytes=normalized["hardware"].memory_bytes,
    )
    traffic = basis["traffic"]
    limits = basis["component_seconds_at_physical_limits"]
    performance.update(
        {
            "gradient_accumulation_steps": basis["work_evidence"][
                "gradient_accumulation_steps"
            ],
            "work_per_step": basis["work_per_step"],
            "work_is_per_optimizer_step": True,
            "flops_per_step": basis["flops"],
            "traffic_bytes_per_rank_step": {
                "kernel_total": traffic["kernel"],
                "optimizer": traffic["optimizer"],
            },
            "communication": {
                "payload_bytes_per_rank_step": traffic["communication"],
                "collective_count": traffic["collective_count"],
            },
            "ideal_seconds": {
                "compute_at_dense_peak": limits["compute"],
                "kernel_hbm_at_physical_peak": limits["kernel_hbm"],
                "optimizer_hbm_at_physical_peak": limits["optimizer_hbm"],
                "collective_payload_at_link_peak": (
                    float(traffic["communication"])
                    / normalized[
                        "hardware"
                    ].intra_node_bandwidth_bytes_per_second
                ),
            },
        }
    )
    observed_rate = float(
        row.get("effective_tokens_per_second") or 0.0
    )
    if record["outcome"] == "success" and observed_rate > 0:
        static_work = float(
            basis["work_per_step"]["effective_tokens"]
        )
        performance["observed"] = {
            # This normalized step makes the training target exactly the
            # measured effective-token rate while all inputs remain pre-run.
            "mean_step_seconds": static_work / observed_rate,
            "effective_tokens_per_second": observed_rate,
            "source": (
                "static_effective_work_divided_by_observed_effective_rate"
            ),
        }
        record["analysis_metadata"][
            "observed_effective_tokens_per_second"
        ] = observed_rate
    record["analysis_metadata"]["static_work_evidence"] = {
        "source_is_pre_run_static_profile": True,
        "gradient_accumulation_steps": basis["work_evidence"][
            "gradient_accumulation_steps"
        ],
    }
    return record


def _build_records(
    *,
    predictor: ThroughputPredictor,
    campaign_root: Path,
    rows: Sequence[Mapping[str, Any]],
    require_throughput: bool,
) -> list[dict[str, Any]]:
    records = []
    for input_index, row in enumerate(rows):
        if str(row.get("classification") or "") not in {
            "success",
            "oom",
        }:
            continue
        record = _base_record(
            predictor=predictor,
            campaign_root=campaign_root,
            row=row,
            input_index=input_index,
        )
        if require_throughput:
            record = _add_static_performance(
                record,
                predictor=predictor,
                row=row,
            )
            if (
                record["outcome"] == "success"
                and not (
                    (record.get("performance") or {}).get("observed")
                )
            ):
                continue
        else:
            record.pop("_normalized", None)
        records.append(record)
    return records


def _balanced_record_folds(
    records: Sequence[Mapping[str, Any]],
    fold_count: int,
) -> list[set[str]]:
    counts = Counter(scenario_id(record) for record in records)
    count = min(int(fold_count), len(counts))
    if count < 2:
        raise ValueError("Memory CV needs at least two scenario folds")
    folds = [set() for _ in range(count)]
    loads = [0] * count
    for current, rows in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        index = min(range(count), key=lambda value: (loads[value], value))
        folds[index].add(current)
        loads[index] += rows
    return folds


def _select_memory_alpha(
    records: Sequence[Mapping[str, Any]],
    *,
    fold_count: int,
) -> dict[str, Any]:
    folds = _balanced_record_folds(records, fold_count)
    evaluated = []
    for alpha in MEMORY_ALPHA_GRID:
        by_scenario: dict[str, list[float]] = defaultdict(list)
        row_errors = []
        for held_out in folds:
            train = [
                record
                for record in records
                if scenario_id(record) not in held_out
            ]
            test = [
                record
                for record in records
                if scenario_id(record) in held_out
                and record["outcome"] == "success"
            ]
            model = _fit_memory_ridge(
                train,
                feature_set=MEMORY_FEATURE_SET,
                alpha=float(alpha),
                historical_weight=0.0,
            )
            for record in test:
                observed = _observed_reserved(record)
                if observed is None:
                    continue
                error = abs(
                    _predict_memory_center(record, model) - observed
                ) / observed
                row_errors.append(error)
                by_scenario[scenario_id(record)].append(error)
        scenario_mape = statistics.fmean(
            statistics.fmean(values)
            for values in by_scenario.values()
        )
        evaluated.append(
            {
                "alpha": float(alpha),
                "success_rows": len(row_errors),
                "scenario_rows": len(by_scenario),
                "row_mape": statistics.fmean(row_errors),
                "scenario_equal_mape": scenario_mape,
            }
        )
    selected = min(
        evaluated,
        key=lambda item: (
            float(item["scenario_equal_mape"]),
            float(item["row_mape"]),
            float(item["alpha"]),
        ),
    )
    return {
        "selection_objective": (
            "minimum complete-scenario CV scenario-equal reserved-center MAPE"
        ),
        "selected": selected,
        "candidates": evaluated,
    }


def _memory_metrics(
    entries: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any]]
    ],
    *,
    include_details: bool = False,
) -> dict[str, Any]:
    absolute = []
    signed = []
    absolute_gib = []
    scenario_errors: dict[str, list[float]] = defaultdict(list)
    success_rows = covered = 0
    oom_rows = false_safe = 0
    safe_success = admitted_safe = 0
    actual_unsafe_success = admitted_unsafe_success = 0
    unavailable = 0
    details = []
    for record, prediction in entries:
        outcome = str(record["outcome"])
        current: dict[str, Any] = {
            "observation_id": str(record["observation_id"]),
            "scenario_id": scenario_id(record),
            "outcome": outcome,
            "configuration": {
                **scenario_material(record),
                "gpu_count": int(record["scenario"]["gpu_count"]),
                "mbs": int(record["scenario"]["physical_mbs"]),
                "zero_stage": int(record["selector"]["zero_stage"]),
                "gradient_checkpointing": bool(
                    record["selector"]["gradient_checkpointing"]
                ),
                "packing": bool(record["selector"]["packing"]),
            },
        }
        if prediction.get("available") is not True:
            unavailable += 1
            current["prediction_available"] = False
            if include_details:
                details.append(current)
            continue
        center = float(prediction["reserved_center_bytes"])
        upper = float(prediction["operational_p95_reserved_bytes"])
        limit = float(_safe_limit(record) or 0.0)
        admitted = upper <= limit
        current.update(
            {
                "prediction_available": True,
                "reserved_center_bytes": center,
                "operational_p95_reserved_bytes": upper,
                "safe_limit_bytes": limit,
                "predicted_admit": admitted,
                "tail_source": prediction.get("success_tail_source"),
            }
        )
        if outcome == "success":
            observed = _observed_reserved(record)
            if observed is None:
                unavailable += 1
                continue
            error = (center - observed) / observed
            absolute.append(abs(error))
            signed.append(error)
            absolute_gib.append(abs(center - observed) / GIB)
            scenario_errors[scenario_id(record)].append(abs(error))
            success_rows += 1
            covered += int(observed <= upper)
            if observed <= limit:
                safe_success += 1
                admitted_safe += int(admitted)
            else:
                actual_unsafe_success += 1
                admitted_unsafe_success += int(admitted)
            current.update(
                {
                    "observed_reserved_bytes": observed,
                    "center_absolute_percentage_error": abs(error),
                    "covered": observed <= upper,
                    "actual_safe_success": observed <= limit,
                }
            )
        elif outcome == "oom":
            oom_rows += 1
            false_safe += int(admitted)
            current["false_safe_oom"] = admitted
        if include_details:
            details.append(current)
    return {
        "rows": len(entries),
        "prediction_unavailable_rows": unavailable,
        "reserved_center_absolute_percentage_error": _summary(absolute),
        "reserved_center_signed_percentage_error": _summary(signed),
        "reserved_center_absolute_error_gib": _summary(absolute_gib),
        "scenario_equal_reserved_center_mape": (
            statistics.fmean(
                statistics.fmean(values)
                for values in scenario_errors.values()
            )
            if scenario_errors
            else None
        ),
        "center_within_10_percent_fraction": (
            statistics.fmean(float(value <= 0.10) for value in absolute)
            if absolute
            else None
        ),
        "center_within_20_percent_fraction": (
            statistics.fmean(float(value <= 0.20) for value in absolute)
            if absolute
            else None
        ),
        "success_rows": success_rows,
        "success_p95_coverage": (
            covered / success_rows if success_rows else None
        ),
        "oom_rows": oom_rows,
        "false_safe_oom": false_safe,
        "false_safe_oom_rate": (
            false_safe / oom_rows if oom_rows else None
        ),
        "oom_rejection_recall": (
            1.0 - false_safe / oom_rows if oom_rows else None
        ),
        "actual_safe_success_rows": safe_success,
        "admitted_safe_success_rows": admitted_safe,
        "false_reject_safe_success": safe_success - admitted_safe,
        "safe_success_admission_recall": (
            admitted_safe / safe_success if safe_success else None
        ),
        "actual_unsafe_success_rows": actual_unsafe_success,
        "admitted_actual_unsafe_success_rows": admitted_unsafe_success,
        "details": details if include_details else None,
    }


def _memory_nested_cv(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entries = []
    fold_reports = []
    for fold_index, held_out in enumerate(
        _balanced_record_folds(records, 5)
    ):
        train = [
            record
            for record in records
            if scenario_id(record) not in held_out
        ]
        test = [
            record
            for record in records
            if scenario_id(record) in held_out
        ]
        selection = _select_memory_alpha(train, fold_count=4)
        alpha = float(selection["selected"]["alpha"])
        center = _fit_memory_ridge(
            train,
            feature_set=MEMORY_FEATURE_SET,
            alpha=alpha,
            historical_weight=0.0,
        )
        tail = _fit_memory_tail(
            train,
            feature_set=MEMORY_FEATURE_SET,
            alpha=alpha,
            historical_weight=0.0,
        )
        current = [
            (record, _predict_memory_upper(record, center, tail))
            for record in test
        ]
        entries.extend(current)
        fold_reports.append(
            {
                "fold": fold_index,
                "held_out_scenario_ids": sorted(held_out),
                "train_rows": len(train),
                "test_rows": len(test),
                "selected_alpha": alpha,
                "metrics": _memory_metrics(current),
            }
        )
    return {
        "method": (
            "nested complete-scenario 5-fold outer CV; alpha selected by "
            "complete-scenario 4-fold inner CV; P95 tail refit in every fold"
        ),
        "metrics": _memory_metrics(entries),
        "folds": fold_reports,
    }


def _full_factor_selection(
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = [
        dict(candidate)
        for candidate in selection["candidates"]
        if candidate["feature_set"] == THROUGHPUT_FEATURE_SET
    ]
    selected = min(
        candidates,
        key=lambda item: (
            float(item["objective"]),
            float(
                item["metrics"]["scenario_equal_absolute_log_rmse"]
            ),
            -float(
                item["metrics"].get(
                    "scenario_equal_pairwise_accuracy"
                )
                or 0.0
            ),
            str(item["target_mode"]),
            float(item["pair_weight"]),
            float(item["alpha"]),
        ),
    )
    return {
        "selection_objective": selection["selection_objective"],
        "constraint": (
            "v4b absolute head is fixed to the 70-dimensional full_factor "
            "contract; only target mode, alpha and pair weight are selected"
        ),
        "candidate_count": len(candidates),
        "selected": selected,
        "candidates": candidates,
    }


def _throughput_nested_scenario_cv(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entries = []
    fold_reports = []
    for fold in _generic_candidate_folds(candidates, 5):
        train = fold["train"]
        test = fold["test"]
        absolute = _full_factor_selection(
            _select_generic(
                train,
                historical_weight=0.0,
                fold_count=4,
            )
        )
        selected = absolute["selected"]
        rank = _select_two_head(
            _generic_candidate_folds(train, 4),
            absolute_settings=selected,
            historical_weight=0.0,
            protocol="inner complete-scenario 4-fold CV",
        )
        rank_selected = rank["selected"]
        model = _fit_two_head_model(
            train,
            absolute_settings=selected,
            rank_feature_set=str(
                rank_selected["rank_feature_set"]
            ),
            rank_alpha=float(rank_selected["rank_alpha"]),
            absolute_deviation_blend=float(
                rank_selected["absolute_deviation_blend"]
            ),
            historical_weight=0.0,
        )
        current = _predict_two_head_entries(test, model)
        entries.extend(current)
        fold_reports.append(
            {
                "fold_id": fold["fold_id"],
                "held_out_scenario_ids": fold[
                    "held_out_scenario_ids"
                ],
                "train_candidates": len(train),
                "test_candidates": len(test),
                "absolute_head": {
                    key: selected[key]
                    for key in (
                        "target_mode",
                        "feature_set",
                        "feature_dimension",
                        "alpha",
                        "pair_weight",
                    )
                },
                "rank_head": {
                    key: rank_selected[key]
                    for key in (
                        "rank_feature_set",
                        "rank_feature_dimension",
                        "rank_alpha",
                        "absolute_deviation_blend",
                    )
                },
                "metrics": _evaluate_entries(current),
            }
        )
    return {
        "method": (
            "nested complete-scenario 5-fold outer CV; full-factor absolute "
            "head and set-aware pairwise head are both selected without the "
            "held-out scenarios"
        ),
        "metrics": _evaluate_entries(entries),
        "folds": fold_reports,
    }


def _throughput_nested_model_holdout(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    model_ids = sorted(
        {
            str(candidate["record"]["scenario"]["model_id"])
            for candidate in candidates
        }
    )
    entries = []
    fold_reports = []
    for held_model in model_ids:
        train = [
            candidate
            for candidate in candidates
            if str(candidate["record"]["scenario"]["model_id"])
            != held_model
        ]
        test = [
            candidate
            for candidate in candidates
            if str(candidate["record"]["scenario"]["model_id"])
            == held_model
        ]
        absolute = _full_factor_selection(
            _select_model_generalization(
                train,
                historical_weight=0.0,
            )
        )
        selected = absolute["selected"]
        rank = _select_two_head(
            _model_candidate_folds(train),
            absolute_settings=selected,
            historical_weight=0.0,
            protocol="inner leave-one-complete-model-id-out",
        )
        rank_selected = rank["selected"]
        model = _fit_two_head_model(
            train,
            absolute_settings=selected,
            rank_feature_set=str(
                rank_selected["rank_feature_set"]
            ),
            rank_alpha=float(rank_selected["rank_alpha"]),
            absolute_deviation_blend=float(
                rank_selected["absolute_deviation_blend"]
            ),
            historical_weight=0.0,
        )
        current = _predict_two_head_entries(test, model)
        entries.extend(current)
        fold_reports.append(
            {
                "held_out_model_id": held_model,
                "train_candidates": len(train),
                "test_candidates": len(test),
                "absolute_head": {
                    key: selected[key]
                    for key in (
                        "target_mode",
                        "feature_set",
                        "feature_dimension",
                        "alpha",
                        "pair_weight",
                    )
                },
                "rank_head": {
                    key: rank_selected[key]
                    for key in (
                        "rank_feature_set",
                        "rank_feature_dimension",
                        "rank_alpha",
                        "absolute_deviation_blend",
                    )
                },
                "metrics": _evaluate_entries(current),
            }
        )
    return {
        "method": (
            "nested leave-one-complete-model-id-out; the held model is absent "
            "from absolute-head and rank-head selection and fitting"
        ),
        "metrics": _evaluate_entries(entries),
        "folds": fold_reports,
    }


def _fit_final_v4b(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    absolute = _full_factor_selection(
        _select_generic(
            candidates,
            historical_weight=0.0,
            fold_count=5,
        )
    )
    selected = absolute["selected"]
    rank = _select_two_head(
        _generic_candidate_folds(candidates, 5),
        absolute_settings=selected,
        historical_weight=0.0,
        protocol="complete-scenario 5-fold CV on the frozen old population",
    )
    rank_selected = rank["selected"]
    model = _fit_two_head_model(
        candidates,
        absolute_settings=selected,
        rank_feature_set=str(rank_selected["rank_feature_set"]),
        rank_alpha=float(rank_selected["rank_alpha"]),
        absolute_deviation_blend=float(
            rank_selected["absolute_deviation_blend"]
        ),
        historical_weight=0.0,
    )
    return model, absolute, rank


def _inference_candidates(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        str, dict[tuple[Any, ...], list[Mapping[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[scenario_id(record)][_candidate_key(record)].append(
            record
        )
    result = []
    for current_scenario, by_candidate in sorted(grouped.items()):
        for key, replicates in sorted(
            by_candidate.items(),
            key=lambda item: tuple(str(value) for value in item[0]),
        ):
            feature_matrix = np.vstack(
                [_record_features(record) for record in replicates]
            )
            work = [
                math.log(
                    float(
                        record["performance"]["work_per_step"][
                            "effective_tokens"
                        ]
                    )
                )
                for record in replicates
            ]
            anchors = [
                _record_log_analytic_anchor(record)
                for record in replicates
            ]
            success_rates = [
                float(
                    record["analysis_metadata"].get(
                        "observed_effective_tokens_per_second"
                    )
                    or 0.0
                )
                for record in replicates
                if record["outcome"] == "success"
            ]
            outcomes = {str(record["outcome"]) for record in replicates}
            result.append(
                {
                    "scenario_id": current_scenario,
                    "scenario": scenario_material(replicates[0]),
                    "candidate_key": list(key),
                    "record": replicates[0],
                    "replicate_records": list(replicates),
                    "replicates": len(replicates),
                    "features": np.median(feature_matrix, axis=0),
                    "effective_log_work": statistics.median(work),
                    "log_analytic_anchor": statistics.median(anchors),
                    "outcome": (
                        "oom" if "oom" in outcomes else "success"
                    ),
                    "observed_log_throughput": (
                        statistics.median(
                            math.log(rate)
                            for rate in success_rates
                            if rate > 0
                        )
                        if success_rates
                        else None
                    ),
                    "historical": False,
                    "observation_ids": sorted(
                        str(record["observation_id"])
                        for record in replicates
                    ),
                }
            )
    return result


def _pipeline_evaluation(
    candidates: Sequence[Mapping[str, Any]],
    *,
    memory_center: Mapping[str, Any],
    memory_tail: Mapping[str, Any],
    throughput_model: Mapping[str, Any],
) -> dict[str, Any]:
    admitted = []
    memory_by_identity: dict[int, Mapping[str, Any]] = {}
    oom_rows = oom_admitted = 0
    safe_success = safe_success_admitted = 0
    for candidate in candidates:
        record = candidate["record"]
        prediction = _predict_memory_upper(
            record,
            memory_center,
            memory_tail,
        )
        memory_by_identity[id(candidate)] = prediction
        available = prediction.get("available") is True
        admit = bool(
            available
            and float(prediction["operational_p95_reserved_bytes"])
            <= float(_safe_limit(record) or 0.0)
        )
        if candidate["outcome"] == "oom":
            oom_rows += 1
            oom_admitted += int(admit)
        else:
            observed = _observed_reserved(record)
            actual_safe = bool(
                observed is not None
                and observed <= float(_safe_limit(record) or 0.0)
            )
            if actual_safe:
                safe_success += 1
                safe_success_admitted += int(admit)
        if admit:
            admitted.append(candidate)

    predicted = {
        id(candidate): value
        for candidate, value in _predict_two_head_entries(
            admitted,
            throughput_model,
        )
    }
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_scenario[str(candidate["scenario_id"])].append(candidate)

    scenarios = 0
    no_admitted = 0
    selected_oom = 0
    selected_unsafe_success = 0
    exact_best = []
    hit90 = []
    regrets = []
    details = []
    for current_scenario, rows in sorted(by_scenario.items()):
        safe_rows = []
        for candidate in rows:
            if candidate["outcome"] != "success":
                continue
            observed_memory = _observed_reserved(candidate["record"])
            observed_rate = candidate.get("observed_log_throughput")
            if (
                observed_memory is not None
                and observed_memory
                <= float(_safe_limit(candidate["record"]) or 0.0)
                and observed_rate is not None
            ):
                safe_rows.append(candidate)
        if not safe_rows:
            continue
        scenarios += 1
        oracle = max(
            safe_rows,
            key=lambda candidate: float(
                candidate["observed_log_throughput"]
            ),
        )
        admitted_rows = [
            candidate
            for candidate in rows
            if id(candidate) in predicted
        ]
        if not admitted_rows:
            no_admitted += 1
            hit90.append(0.0)
            details.append(
                {
                    "scenario_id": current_scenario,
                    "status": "no_admitted_candidate",
                    "oracle_candidate_key": oracle["candidate_key"],
                }
            )
            continue
        selected = max(
            admitted_rows,
            key=lambda candidate: predicted[id(candidate)],
        )
        if selected["outcome"] == "oom":
            selected_oom += 1
            hit90.append(0.0)
            details.append(
                {
                    "scenario_id": current_scenario,
                    "status": "selected_oom",
                    "selected_candidate_key": selected["candidate_key"],
                    "oracle_candidate_key": oracle["candidate_key"],
                }
            )
            continue
        selected_memory = _observed_reserved(selected["record"])
        selected_is_safe = bool(
            selected_memory is not None
            and selected_memory
            <= float(_safe_limit(selected["record"]) or 0.0)
        )
        if not selected_is_safe:
            selected_unsafe_success += 1
            hit90.append(0.0)
            details.append(
                {
                    "scenario_id": current_scenario,
                    "status": "selected_operationally_unsafe_success",
                    "selected_candidate_key": selected["candidate_key"],
                    "oracle_candidate_key": oracle["candidate_key"],
                }
            )
            continue
        regret = max(
            0.0,
            1.0
            - math.exp(
                float(selected["observed_log_throughput"])
                - float(oracle["observed_log_throughput"])
            ),
        )
        regrets.append(regret)
        exact_best.append(
            float(selected["candidate_key"] == oracle["candidate_key"])
        )
        hit90.append(float(regret <= 0.10))
        details.append(
            {
                "scenario_id": current_scenario,
                "status": "selected_safe_success",
                "selected_candidate_key": selected["candidate_key"],
                "oracle_candidate_key": oracle["candidate_key"],
                "top1_regret": regret,
                "hit_at_10_percent": regret <= 0.10,
            }
        )
    return {
        "candidate_rows": len(candidates),
        "predicted_admitted_candidate_rows": len(admitted),
        "oom_candidate_rows": oom_rows,
        "false_safe_oom_candidates": oom_admitted,
        "false_safe_oom_rate": (
            oom_admitted / oom_rows if oom_rows else None
        ),
        "safe_success_candidate_rows": safe_success,
        "admitted_safe_success_candidate_rows": safe_success_admitted,
        "safe_success_admission_recall": (
            safe_success_admitted / safe_success
            if safe_success
            else None
        ),
        "evaluable_scenarios": scenarios,
        "no_admitted_candidate_scenarios": no_admitted,
        "selected_oom_scenarios": selected_oom,
        "selected_operationally_unsafe_success_scenarios": (
            selected_unsafe_success
        ),
        "exact_best_fraction": (
            statistics.fmean(exact_best) if exact_best else None
        ),
        "selected_at_least_90_percent_of_best_fraction": (
            statistics.fmean(hit90) if hit90 else None
        ),
        "scenario_equal_top1_regret": (
            statistics.fmean(regrets) if regrets else None
        ),
        "details": details,
    }


def _feature_audit(
    memory_records: Sequence[Mapping[str, Any]],
    throughput_candidates: Sequence[Mapping[str, Any]],
    throughput_model: Mapping[str, Any],
) -> dict[str, Any]:
    memory_matrix = np.vstack(
        [
            _memory_features(record, MEMORY_FEATURE_SET)
            for record in memory_records
            if record["outcome"] == "success"
        ]
    )
    throughput_matrix = np.vstack(
        [
            np.asarray(candidate["features"], dtype=float)
            for candidate in throughput_candidates
        ]
    )
    absolute_names = throughput_model["absolute_head"]["feature_names"]
    absolute_indexes = [
        FEATURE_NAMES.index(name) for name in absolute_names
    ]
    rank_names = throughput_model["rank_head"]["feature_names"]
    rank_indexes = [FEATURE_NAMES.index(name) for name in rank_names]

    def audit(matrix: np.ndarray) -> dict[str, Any]:
        varying = np.ptp(matrix, axis=0) > 1.0e-9
        centered = matrix[:, varying] - np.mean(
            matrix[:, varying],
            axis=0,
        )
        return {
            "nominal_dimensions": int(matrix.shape[1]),
            "varying_dimensions": int(np.sum(varying)),
            "centered_matrix_rank": int(np.linalg.matrix_rank(centered)),
            "rows": int(matrix.shape[0]),
        }

    return {
        "memory_physical_shares": audit(memory_matrix),
        "throughput_absolute_head": audit(
            throughput_matrix[:, absolute_indexes]
        ),
        "throughput_rank_head": audit(
            throughput_matrix[:, rank_indexes]
        ),
    }


def _cohort_counts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "success": sum(
            str(row.get("classification")) == "success"
            for row in rows
        ),
        "oom": sum(
            str(row.get("classification")) == "oom"
            for row in rows
        ),
        "models": dict(
            sorted(Counter(str(row.get("model_id")) for row in rows).items())
        ),
        "datasets": dict(
            sorted(
                Counter(str(row.get("dataset_id")) for row in rows).items()
            )
        ),
    }


def build_report(campaign_root: Path) -> dict[str, Any]:
    rows = _read_rows(campaign_root)
    predictor = ThroughputPredictor(strict_bindings=False)
    hardware_path = campaign_root / "config" / "hardware.json"
    collected_path = campaign_root / "artifacts" / "collected_results.json"
    hardware = read_json(hardware_path)
    if "4090" not in str(
        hardware.get("name_reported_by_driver") or ""
    ):
        raise ValueError("This fit is RTX 4090-only")

    screen_rows = [
        row for row in rows if row.get("kind") == "throughput_screen"
    ]
    if not screen_rows:
        raise ValueError("No RTX 4090 throughput screen rows")
    screen_start = min(
        float(row["started_unix"]) for row in screen_rows
    )

    memory_old_rows = [
        row for row in rows if row.get("kind") == "memory_probe"
    ]
    memory_screen_rows = screen_rows
    memory_formal_rows = [
        row
        for row in rows
        if row.get("kind") == "throughput"
        and row.get("fidelity") == "formal"
    ]
    memory_packing_rows = [
        row
        for row in rows
        if bool(row.get("packing"))
        and row.get("kind") in {
            "packing_memory_probe",
            "throughput",
        }
    ]

    memory_old = _build_records(
        predictor=predictor,
        campaign_root=campaign_root,
        rows=memory_old_rows,
        require_throughput=False,
    )
    memory_screen = _build_records(
        predictor=predictor,
        campaign_root=campaign_root,
        rows=memory_screen_rows,
        require_throughput=False,
    )
    memory_formal = _build_records(
        predictor=predictor,
        campaign_root=campaign_root,
        rows=memory_formal_rows,
        require_throughput=False,
    )
    memory_packing = _build_records(
        predictor=predictor,
        campaign_root=campaign_root,
        rows=memory_packing_rows,
        require_throughput=False,
    )

    memory_nested = _memory_nested_cv(memory_old)
    memory_selection = _select_memory_alpha(
        memory_old,
        fold_count=5,
    )
    memory_alpha = float(memory_selection["selected"]["alpha"])
    memory_center = _fit_memory_ridge(
        memory_old,
        feature_set=MEMORY_FEATURE_SET,
        alpha=memory_alpha,
        historical_weight=0.0,
    )
    memory_tail = _fit_memory_tail(
        memory_old,
        feature_set=MEMORY_FEATURE_SET,
        alpha=memory_alpha,
        historical_weight=0.0,
    )

    def memory_eval(
        population: Sequence[Mapping[str, Any]],
        *,
        details: bool = False,
    ) -> dict[str, Any]:
        return _memory_metrics(
            [
                (
                    record,
                    _predict_memory_upper(
                        record,
                        memory_center,
                        memory_tail,
                    ),
                )
                for record in population
            ],
            include_details=details,
        )

    old_scenarios = {scenario_id(record) for record in memory_old}
    memory_screen_unseen = [
        record
        for record in memory_screen
        if scenario_id(record) not in old_scenarios
    ]
    memory_formal_unseen = [
        record
        for record in memory_formal
        if scenario_id(record) not in old_scenarios
    ]

    throughput_old_rows = [
        row
        for row in rows
        if (
            row.get("kind") == "throughput_screen"
            or (
                row.get("kind") == "throughput"
                and float(row.get("finished_unix") or 0.0)
                < screen_start
                and not bool(row.get("packing"))
            )
        )
        and row.get("classification") == "success"
        and bool(row.get("metrics_available"))
        and float(row.get("effective_tokens_per_second") or 0.0) > 0
    ]
    throughput_formal_all_rows = memory_formal_rows
    throughput_formal_success_rows = [
        row
        for row in throughput_formal_all_rows
        if row.get("classification") == "success"
        and bool(row.get("metrics_available"))
        and float(row.get("effective_tokens_per_second") or 0.0) > 0
    ]
    throughput_packing_all_rows = [
        row
        for row in rows
        if row.get("kind") == "throughput"
        and str(row.get("job_id") or "").startswith(
            ("packon-", "packoff-")
        )
    ]
    throughput_packing_success_rows = [
        row
        for row in throughput_packing_all_rows
        if row.get("classification") == "success"
        and bool(row.get("metrics_available"))
        and float(row.get("effective_tokens_per_second") or 0.0) > 0
    ]

    throughput_old_records = _build_records(
        predictor=predictor,
        campaign_root=campaign_root,
        rows=throughput_old_rows,
        require_throughput=True,
    )
    throughput_formal_records = _build_records(
        predictor=predictor,
        campaign_root=campaign_root,
        rows=throughput_formal_success_rows,
        require_throughput=True,
    )
    throughput_formal_all_records = _build_records(
        predictor=predictor,
        campaign_root=campaign_root,
        rows=throughput_formal_all_rows,
        require_throughput=True,
    )
    throughput_packing_records = _build_records(
        predictor=predictor,
        campaign_root=campaign_root,
        rows=throughput_packing_success_rows,
        require_throughput=True,
    )
    throughput_packing_all_records = _build_records(
        predictor=predictor,
        campaign_root=campaign_root,
        rows=throughput_packing_all_rows,
        require_throughput=True,
    )

    old_candidates = _pool_candidates(throughput_old_records)
    formal_candidates = _pool_candidates(throughput_formal_records)
    packing_candidates = _pool_candidates(throughput_packing_records)
    throughput_nested = _throughput_nested_scenario_cv(old_candidates)
    throughput_model_holdout = _throughput_nested_model_holdout(
        old_candidates
    )
    v4b, v4b_absolute_selection, v4b_rank_selection = (
        _fit_final_v4b(old_candidates)
    )
    formal_entries = _predict_two_head_entries(
        formal_candidates,
        v4b,
    )
    packing_entries = _predict_two_head_entries(
        packing_candidates,
        v4b,
    )
    formal_inference = _inference_candidates(
        throughput_formal_all_records
    )
    packing_inference = _inference_candidates(
        throughput_packing_all_records
    )

    old_configurations = {
        (
            str(candidate["scenario_id"]),
            tuple(candidate["candidate_key"]),
        )
        for candidate in old_candidates
    }
    formal_configurations = {
        (
            str(candidate["scenario_id"]),
            tuple(candidate["candidate_key"]),
        )
        for candidate in formal_candidates
    }

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_family": "NVIDIA GeForce RTX 4090",
        "hardware_id": str(hardware["hardware_id"]),
        "analysis_only": True,
        "publishable": False,
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "split_contract": {
            "memory_fit": "kind == memory_probe only",
            "memory_temporal_tests": [
                "throughput_screen",
                "formal_throughput",
                "packing_true",
            ],
            "throughput_fit": (
                "successful pre-screen throughput plus throughput_screen"
            ),
            "throughput_temporal_tests": [
                "formal long-window throughput",
                "packing A/B",
            ],
            "screen_start_unix": screen_start,
            "formal_is_exact_configuration_remeasurement": True,
            "formal_exact_configuration_overlap": len(
                old_configurations & formal_configurations
            ),
            "formal_test_configurations": len(formal_configurations),
        },
        "data": {
            "all_collected": _cohort_counts(rows),
            "memory_old_fit_population": _cohort_counts(
                memory_old_rows
            ),
            "memory_screen_replay": _cohort_counts(
                memory_screen_rows
            ),
            "memory_formal_replay": _cohort_counts(
                memory_formal_rows
            ),
            "memory_packing_replay": _cohort_counts(
                memory_packing_rows
            ),
            "throughput_old_fit_rows": _cohort_counts(
                throughput_old_rows
            ),
            "throughput_formal_rows": _cohort_counts(
                throughput_formal_all_rows
            ),
            "throughput_packing_ab_rows": _cohort_counts(
                throughput_packing_all_rows
            ),
        },
        "memory": {
            "model_contract": {
                "name": "physical-shares",
                "center_formula": (
                    "reserved_center = analytic_reference * "
                    "exp(intercept + standardized_28_features @ beta)"
                ),
                "operational_formula": (
                    "reserved_p95 = reserved_center * exp(max("
                    "OOF_success_q95, exact_selector_OOM_guard, 0))"
                ),
                "feature_set": MEMORY_FEATURE_SET,
                "feature_dimension": 28,
                "safe_limit_fraction_of_capacity": 0.95,
            },
            "selection": memory_selection,
            "old_experiment_nested_cv": memory_nested,
            "frozen_model": {
                "center": memory_center,
                "tail": memory_tail,
                "training_rows": len(memory_old),
                "training_scenarios": len(old_scenarios),
                "training_observation_ids_sha256": sha256_json(
                    sorted(
                        str(record["observation_id"])
                        for record in memory_old
                    )
                ),
            },
            "temporal_generalization": {
                "screen_all": memory_eval(memory_screen),
                "screen_unseen_scenarios": memory_eval(
                    memory_screen_unseen
                ),
                "formal_all": memory_eval(memory_formal),
                "formal_unseen_scenarios": memory_eval(
                    memory_formal_unseen
                ),
                "packing_true_unseen_factor": memory_eval(
                    memory_packing
                ),
            },
        },
        "throughput": {
            "model_contract": {
                "name": "v4b",
                "family": "set_aware_two_head_log_throughput_model",
                "absolute_head_feature_dimension": 70,
                "formula": (
                    "a_i=absolute_log_throughput(x_i); "
                    "r_i=pairwise_rank_score(x_i); "
                    "prediction_i=mean(a)+lambda*(a_i-mean(a))"
                    "+(1-lambda)*(r_i-mean(r))"
                ),
                "inputs_are_pre_run_static": True,
                "candidate_set_dependent": True,
            },
            "old_fit_candidates": len(old_candidates),
            "old_fit_scenarios": len(
                {str(candidate["scenario_id"]) for candidate in old_candidates}
            ),
            "old_experiment_nested_scenario_cv": throughput_nested,
            "complete_model_holdout": throughput_model_holdout,
            "absolute_head_selection": v4b_absolute_selection,
            "rank_head_selection": v4b_rank_selection,
            "frozen_model": v4b,
            "in_sample_diagnostic": _evaluate_entries(
                _predict_two_head_entries(old_candidates, v4b)
            ),
            "temporal_generalization": {
                "formal_long_window": _evaluate_entries(
                    formal_entries
                ),
                "packing_ab_unseen_factor": _evaluate_entries(
                    packing_entries
                ),
            },
        },
        "combined_pipeline": {
            "policy": (
                "physical-shares P95 filters candidates first; v4b is then "
                "recentered and ranked on the admitted candidate set"
            ),
            "formal_long_window": _pipeline_evaluation(
                formal_inference,
                memory_center=memory_center,
                memory_tail=memory_tail,
                throughput_model=v4b,
            ),
            "packing_ab_unseen_factor": _pipeline_evaluation(
                packing_inference,
                memory_center=memory_center,
                memory_tail=memory_tail,
                throughput_model=v4b,
            ),
        },
        "effective_feature_audit": _feature_audit(
            memory_old,
            old_candidates,
            v4b,
        ),
        "source_bindings": {
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "collected_results": {
                "path": str(collected_path.resolve()),
                "sha256": sha256_file(collected_path),
            },
            "hardware": {
                "path": str(hardware_path.resolve()),
                "sha256": sha256_file(hardware_path),
            },
            "static_dataset_profiles": predictor.profiles.source_bindings(),
        },
        "limitations": [
            (
                "All model-family evidence is Qwen3 dense text SFT at "
                "0.6B, 1.7B and 4B; this is not proof for larger models, "
                "VL models or Qwen3.5."
            ),
            (
                "Formal throughput is a longer-window repeat of screen-selected "
                "configurations, so it tests temporal/fidelity stability rather "
                "than unseen-configuration generalization."
            ),
            (
                "Packing A/B is an unseen factor test, but only 18 scenarios "
                "have two successful candidates for direct ranking."
            ),
            (
                "Many RTX 4090 measurements are thermal or power limited; "
                "the model represents this host/runtime cohort."
            ),
            (
                "The PCIe collective bandwidth is a nominal prior rather than "
                "a dedicated NCCL calibration."
            ),
        ],
    }
    report["report_sha256"] = sha256_json(report)
    return report


def _metric_subset(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metrics.get(key)
        for key in (
            "candidate_rows",
            "scenario_rows",
            "comparable_scenario_rows",
            "pairwise_rows",
            "scenario_equal_pairwise_accuracy",
            "scenario_equal_top1_regret",
            "scenario_equal_hit_at_10_percent",
            "absolute_throughput_mape",
            "scenario_equal_absolute_throughput_mape",
            "absolute_throughput_ape_p90",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()
    report = build_report(args.campaign_root.resolve())
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "memory_old_nested": report["memory"][
                    "old_experiment_nested_cv"
                ]["metrics"],
                "memory_formal": report["memory"][
                    "temporal_generalization"
                ]["formal_all"],
                "memory_packing": report["memory"][
                    "temporal_generalization"
                ]["packing_true_unseen_factor"],
                "throughput_old_nested": _metric_subset(
                    report["throughput"][
                        "old_experiment_nested_scenario_cv"
                    ]["metrics"]
                ),
                "throughput_model_holdout": _metric_subset(
                    report["throughput"][
                        "complete_model_holdout"
                    ]["metrics"]
                ),
                "throughput_formal": _metric_subset(
                    report["throughput"][
                        "temporal_generalization"
                    ]["formal_long_window"]
                ),
                "throughput_packing": _metric_subset(
                    report["throughput"][
                        "temporal_generalization"
                    ]["packing_ab_unseen_factor"]
                ),
                "pipeline_formal": report["combined_pipeline"][
                    "formal_long_window"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
