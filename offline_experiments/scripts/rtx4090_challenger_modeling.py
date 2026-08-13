#!/usr/bin/env python3
"""Evaluate the H800 hybrid throughput formula on the isolated RTX 4090 campaign.

This is an offline, non-publishing analysis.  It reuses the H800 challenger
structure:

    score = log(physical_throughput) + intercept + standardized_features @ beta

The physical efficiency curve and pairwise residual coefficients are refit on
RTX 4090 observations.  Every reported cross-validation prediction leaves the
complete (model, train type, dataset, target GBS) scenario out of both fits.
No GPU work is launched and no runtime queue is read or mutated.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from common import ROOT, read_json, sha256_file, write_json
from audit_h800_calibration_readiness import SUPPORTED_MBS
from h800_challenger_modeling import (
    _baseline_ranker_model,
    _fit_pairwise_ranker,
    _ranking_evaluation,
    _throughput_candidates,
)
from h800_theory_basis import build_record as build_theory_record
from h800_theory_calibration import (
    _fit_compute_model,
    scenario_id,
    scenario_material,
)


SCHEMA = "sft_rtx4090_challenger_modeling/v1"
IMPLEMENTATION_VERSION = (
    "sft_rtx4090_challenger_modeling_impl/"
    "2026-07-28.h800-formula-scenario-loso"
)
DEFAULT_CAMPAIGN_ROOT = (
    ROOT / "campaigns" / "rtx4090_20260717"
)
DEFAULT_HBM_BANDWIDTH_BYTES_S = 1.008e12
DEFAULT_COLLECTIVE_BANDWIDTH_BYTES_S = 32e9
DEFAULT_HBM_EFFICIENCY = 0.60
DEFAULT_COLLECTIVE_EFFICIENCY = 0.70
DEFAULT_COLLECTIVE_LATENCY_SECONDS = 0.00001
DEFAULT_ALPHA = 0.01
RUNTIME_COHORT_ID = "rtx4090_20260717_fa2"
WORK_KEYS = (
    "computed_tokens",
    "effective_tokens",
    "computed_attention_token_pairs",
    "logical_samples",
)


def _matrix_jobs(path: Path) -> list[dict[str, Any]]:
    jobs = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("job_id"):
                raise ValueError(f"{path}:{line_number} has no job_id")
            jobs.append(row)
    if len({str(job["job_id"]) for job in jobs}) != len(jobs):
        raise ValueError(f"{path} contains duplicate job ids")
    return jobs


def _result_summaries(result_dir: Path) -> list[dict[str, Any]]:
    summaries = [
        read_json(path)
        for path in sorted((result_dir / "metrics").glob("summary.rank*.json"))
    ]
    if not summaries:
        raise ValueError(f"{result_dir} has no rank summaries")
    return summaries


def _canonical_success_row(
    aggregate: Mapping[str, Any],
    job: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    measured_steps = {int(summary["measured_steps"]) for summary in summaries}
    if len(measured_steps) != 1:
        raise ValueError(
            f"{job['job_id']} has inconsistent measured-step counts: "
            f"{sorted(measured_steps)}"
        )
    steps = next(iter(measured_steps))
    measured_seconds = float(aggregate["measured_seconds"])
    if steps <= 0 or measured_seconds <= 0:
        raise ValueError(f"{job['job_id']} has no positive measurement window")
    work = {
        key: sum(
            int((summary.get("measured_totals") or {})[key])
            for summary in summaries
        )
        for key in WORK_KEYS
    }
    return {
        "observation_id": str(job["job_id"]),
        "configuration": {"job": dict(job)},
        "outcome": {"class": "success"},
        "measurements": {
            "mean_step_seconds": measured_seconds / steps,
            "measured_step_count": steps,
            "work": work,
            "memory": {
                "max_allocated_bytes": int(
                    aggregate.get("max_allocated_bytes") or 0
                ),
                "max_reserved_bytes": int(
                    aggregate.get("max_reserved_bytes") or 0
                ),
                "values_are_observed_not_imputed": True,
            },
        },
    }


def _apply_4090_physical_constants(
    record: dict[str, Any],
    *,
    hbm_bandwidth_bytes_s: float,
    collective_bandwidth_bytes_s: float,
) -> None:
    performance = record["performance"]
    scenario = record["scenario"]
    peak = float(
        performance["physical_priors"][
            "dense_bf16_peak_flops_per_gpu"
        ]
    )
    gpu_count = int(scenario["gpu_count"])
    flops = float(performance["flops_per_step"]["total"])
    traffic = performance["traffic_bytes_per_rank_step"]
    communication = performance["communication"]
    kernel_bytes = float(traffic["kernel_total"])
    optimizer_bytes = float(traffic["optimizer"])
    collective_bytes = float(communication["payload_bytes_per_rank_step"])
    communication["ideal_payload_seconds"] = (
        collective_bytes / collective_bandwidth_bytes_s
    )
    performance["ideal_seconds"] = {
        "compute_at_dense_peak": flops / (gpu_count * peak),
        "kernel_hbm_at_physical_peak": (
            kernel_bytes / hbm_bandwidth_bytes_s
        ),
        "optimizer_hbm_at_physical_peak": (
            optimizer_bytes / hbm_bandwidth_bytes_s
        ),
        "collective_payload_at_link_peak": (
            collective_bytes / collective_bandwidth_bytes_s
        ),
    }
    performance["physical_priors"] = {
        "dense_bf16_peak_flops_per_gpu": peak,
        "hbm_bandwidth_bytes_per_second": hbm_bandwidth_bytes_s,
        "intra_node_bandwidth_bytes_per_second": (
            collective_bandwidth_bytes_s
        ),
        "hbm_efficiency": {
            "center": DEFAULT_HBM_EFFICIENCY,
            "lower": 0.50,
        },
        "collective_efficiency": {
            "center": DEFAULT_COLLECTIVE_EFFICIENCY,
            "lower": 0.60,
        },
        "communication_overlap": 0.0,
        "collective_latency_seconds": DEFAULT_COLLECTIVE_LATENCY_SECONDS,
    }


def _physical_priors(
    hardware: Mapping[str, Any],
    *,
    hbm_bandwidth_bytes_s: float,
    collective_bandwidth_bytes_s: float,
) -> dict[str, Any]:
    return {
        "dense_peak_flops_per_s": float(
            hardware["bf16_dense_peak_flops_per_second_for_mfu"]
        ),
        "memory_bandwidth_bytes_per_s": hbm_bandwidth_bytes_s,
        "collective_bandwidth_bytes_per_s": collective_bandwidth_bytes_s,
        "hbm_efficiency": DEFAULT_HBM_EFFICIENCY,
        "optimizer_hbm_efficiency": DEFAULT_HBM_EFFICIENCY,
        "collective_efficiency": DEFAULT_COLLECTIVE_EFFICIENCY,
        "collective_latency_seconds": DEFAULT_COLLECTIVE_LATENCY_SECONDS,
        "microstep_latency_seconds": 0.0,
        "framework_latency_seconds": 0.0,
        "communication_overlap_by_stage": {
            0: 0.0,
            1: 0.0,
            2: 0.0,
            3: 0.0,
        },
        "memory_capacity_bytes": float(
            hardware["memory_bytes_reported_by_torch"]
        ),
        "compute_efficiency": None,
    }


def _build_records(
    campaign_root: Path,
    *,
    matrix_name: str,
    hbm_bandwidth_bytes_s: float,
    collective_bandwidth_bytes_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix_path = campaign_root / "matrix" / matrix_name
    planned = _matrix_jobs(matrix_path)
    collected = read_json(
        campaign_root / "artifacts" / "collected_results.json"
    )
    rows = {
        str(row["job_id"]): row
        for row in collected.get("rows") or []
        if isinstance(row, Mapping) and row.get("job_id")
    }
    inventory = read_json(
        campaign_root / "artifacts" / "model_inventory.json"
    )
    models = {
        str(model["id"]): model for model in inventory.get("models") or []
    }
    fixed_lora = inventory["fixed_lora"]
    hardware = read_json(campaign_root / "config" / "hardware.json")
    records = []
    exclusion = Counter()
    status = Counter()
    clock_status = Counter()
    model_rows = Counter()
    for planned_job in planned:
        job_id = str(planned_job["job_id"])
        aggregate = rows.get(job_id)
        if aggregate is None:
            exclusion["result_missing"] += 1
            continue
        classification = str(aggregate.get("classification") or "unknown")
        status[classification] += 1
        if classification != "success":
            exclusion[f"outcome_{classification}"] += 1
            continue
        if aggregate.get("metrics_available") is not True:
            exclusion["metrics_unavailable"] += 1
            continue
        if aggregate.get("packing") is True:
            exclusion["packing_excluded"] += 1
            continue
        try:
            mbs = int(aggregate.get("mbs"))
        except (TypeError, ValueError):
            exclusion["mbs_missing_or_invalid"] += 1
            continue
        if mbs not in SUPPORTED_MBS:
            exclusion["mbs_outside_h800_challenger_domain"] += 1
            continue
        result_dir = campaign_root / "results" / job_id
        rendered = read_json(result_dir / "rendered_run.json")
        job = rendered.get("job")
        if not isinstance(job, Mapping):
            exclusion["rendered_job_missing"] += 1
            continue
        if any(job.get(key) != planned_job.get(key) for key in (
            "job_id",
            "kind",
            "model_id",
            "train_type",
            "dataset_id",
            "target_gbs",
            "gpu_count",
            "mbs",
            "zero",
            "gc",
            "packing",
        )):
            exclusion["matrix_identity_mismatch"] += 1
            continue
        try:
            summaries = _result_summaries(result_dir)
            canonical = _canonical_success_row(
                aggregate, job, summaries
            )
            recovery = {
                "runtime": {
                    "runtime_cohort_id": RUNTIME_COHORT_ID,
                    "runtime_cohort_material": {
                        "campaign_id": aggregate.get("campaign_id"),
                        "runtime_manifest": str(
                            campaign_root / "runtime" / "runtime_manifest.json"
                        ),
                    },
                },
                "measurement_eligibility": {
                    "throughput_primary": True
                },
                "source_observation_sha256": "campaign_collected_result",
                "recovery_id": None,
                "evidence_tier": "native_campaign",
            }
            record = build_theory_record(
                canonical,
                recovery,
                models[str(job["model_id"])],
                fixed_lora,
                hardware,
                campaign_root / "runtime",
            )
        except (KeyError, TypeError, ValueError) as error:
            exclusion[f"record_build_error:{type(error).__name__}"] += 1
            continue
        record["selector"]["kernel_path"] = (
            "fa2+liger_fused_ce+adamw_torch_fused"
        )
        record["selector"]["runtime_cohort_id"] = RUNTIME_COHORT_ID
        record["runtime"]["runtime_cohort_id"] = RUNTIME_COHORT_ID
        record["analysis_metadata"] = {
            "matrix": matrix_name,
            "fidelity": aggregate.get("fidelity"),
            "clock_status": aggregate.get("clock_status"),
            "downclock_detected": aggregate.get("downclock_detected"),
            "provenance_sha256": aggregate.get("provenance_sha256"),
        }
        _apply_4090_physical_constants(
            record,
            hbm_bandwidth_bytes_s=hbm_bandwidth_bytes_s,
            collective_bandwidth_bytes_s=collective_bandwidth_bytes_s,
        )
        records.append(record)
        clock_status[str(aggregate.get("clock_status") or "unknown")] += 1
        model_rows[str(job["model_id"])] += 1
    summary = {
        "matrix": matrix_name,
        "matrix_sha256": sha256_file(matrix_path),
        "planned_jobs": len(planned),
        "terminal_outcomes": dict(sorted(status.items())),
        "admitted_success_rows": len(records),
        "excluded": dict(sorted(exclusion.items())),
        "clock_status_of_admitted": dict(sorted(clock_status.items())),
        "model_rows": dict(sorted(model_rows.items())),
    }
    return records, summary


def _aggregate_fold_metrics(
    folds: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    metrics = [fold[field] for fold in folds]
    pair_rows = sum(int(item.get("pairwise_rows") or 0) for item in metrics)
    pair_correct = sum(
        float(item["pooled_pairwise_accuracy"])
        * int(item["pairwise_rows"])
        for item in metrics
        if item.get("pooled_pairwise_accuracy") is not None
    )
    comparable_scenarios = sum(
        int(fold.get("comparable_scenarios") or 0) for fold in folds
    )
    scenario_pair = sum(
        float(item["scenario_equal_pairwise_accuracy"])
        * int(fold.get("comparable_scenarios") or 0)
        for fold, item in zip(folds, metrics)
        if item.get("scenario_equal_pairwise_accuracy") is not None
    )
    top1 = sum(
        float(item["scenario_equal_top1_regret"])
        * int(fold.get("comparable_scenarios") or 0)
        for fold, item in zip(folds, metrics)
        if item.get("scenario_equal_top1_regret") is not None
    )
    hit10 = sum(
        float(item["scenario_equal_hit_at_10_percent"])
        * int(fold.get("comparable_scenarios") or 0)
        for fold, item in zip(folds, metrics)
        if item.get("scenario_equal_hit_at_10_percent") is not None
    )
    gpu_rows = sum(int(item.get("gpu_group_rows") or 0) for item in metrics)
    gpu_regret = sum(
        float(item["scenario_gpu_equal_top1_regret"])
        * int(item["gpu_group_rows"])
        for item in metrics
        if item.get("scenario_gpu_equal_top1_regret") is not None
    )
    gpu_hit = sum(
        float(item["scenario_gpu_equal_hit_at_10_percent"])
        * int(item["gpu_group_rows"])
        for item in metrics
        if item.get("scenario_gpu_equal_hit_at_10_percent") is not None
    )
    scenario_rows = sum(
        int(fold.get("test_scenarios") or 0) for fold in folds
    )
    scenario_mape = sum(
        float(item["scenario_equal_absolute_throughput_mape"])
        * int(fold.get("test_scenarios") or 0)
        for fold, item in zip(folds, metrics)
        if item.get("scenario_equal_absolute_throughput_mape") is not None
    )
    candidate_rows = sum(int(item.get("candidate_rows") or 0) for item in metrics)
    absolute_mean = sum(
        float(item["absolute_throughput_percentage_error"]["mean"])
        * int(item["candidate_rows"])
        for item in metrics
        if (item.get("absolute_throughput_percentage_error") or {}).get(
            "mean"
        )
        is not None
    )
    return {
        "candidate_rows": candidate_rows,
        "scenario_rows": scenario_rows,
        "comparable_scenarios": comparable_scenarios,
        "pairwise_rows": pair_rows,
        "pooled_pairwise_accuracy": (
            pair_correct / pair_rows if pair_rows else None
        ),
        "scenario_equal_pairwise_accuracy": (
            scenario_pair / comparable_scenarios
            if comparable_scenarios
            else None
        ),
        "scenario_equal_top1_regret": (
            top1 / comparable_scenarios if comparable_scenarios else None
        ),
        "scenario_equal_hit_at_10_percent": (
            hit10 / comparable_scenarios if comparable_scenarios else None
        ),
        "gpu_group_rows": gpu_rows,
        "scenario_gpu_equal_top1_regret": (
            gpu_regret / gpu_rows if gpu_rows else None
        ),
        "scenario_gpu_equal_hit_at_10_percent": (
            gpu_hit / gpu_rows if gpu_rows else None
        ),
        "absolute_throughput_mape": (
            absolute_mean / candidate_rows if candidate_rows else None
        ),
        "scenario_equal_absolute_throughput_mape": (
            scenario_mape / scenario_rows if scenario_rows else None
        ),
    }


def _candidate_group_counts(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    counts = Counter(str(candidate["scenario_id"]) for candidate in candidates)
    return len(counts), sum(count >= 2 for count in counts.values())


def _scenario_loso(
    records: Sequence[Mapping[str, Any]],
    *,
    priors: Mapping[str, Any],
    alpha: float,
) -> dict[str, Any]:
    scenario_ids = sorted({scenario_id(record) for record in records})
    folds = []
    for held_out in scenario_ids:
        train_records = [
            record for record in records if scenario_id(record) != held_out
        ]
        test_records = [
            record for record in records if scenario_id(record) == held_out
        ]
        if {
            scenario_id(record) for record in train_records
        }.intersection({scenario_id(record) for record in test_records}):
            raise AssertionError("scenario leakage detected")
        physical_model = _fit_compute_model(train_records, priors)
        if physical_model.get("available") is not True:
            raise ValueError(
                f"physical fit unavailable in fold {held_out}: "
                f"{physical_model.get('blockers')}"
            )
        train_candidates = _throughput_candidates(
            train_records,
            physical_model=physical_model,
            priors=priors,
        )
        test_candidates = _throughput_candidates(
            test_records,
            physical_model=physical_model,
            priors=priors,
        )
        challenger = _fit_pairwise_ranker(
            train_candidates,
            feature_set="physical",
            alpha=alpha,
            historical_weight=0.0,
            use_physics_base=True,
        )
        no_physics_ablation = _fit_pairwise_ranker(
            train_candidates,
            feature_set="basic",
            alpha=alpha,
            historical_weight=0.0,
            use_physics_base=False,
        )
        baseline_metrics = _ranking_evaluation(
            test_candidates,
            _baseline_ranker_model("physical"),
            include_details=False,
        )
        challenger_metrics = _ranking_evaluation(
            test_candidates,
            challenger,
            include_details=False,
        )
        no_physics_metrics = _ranking_evaluation(
            test_candidates,
            no_physics_ablation,
            include_details=False,
        )
        test_scenarios, comparable_scenarios = _candidate_group_counts(
            test_candidates
        )
        folds.append(
            {
                "held_out_scenario_id": held_out,
                "held_out_scenario": scenario_material(test_records[0]),
                "train_records": len(train_records),
                "test_records": len(test_records),
                "train_candidates": len(train_candidates),
                "test_candidates": len(test_candidates),
                "test_scenarios": test_scenarios,
                "comparable_scenarios": comparable_scenarios,
                "physical_solver_converged": bool(
                    physical_model.get("bounded_solver_converged")
                ),
                "physical_baseline": baseline_metrics,
                "pairwise_residual": challenger_metrics,
                "configuration_only_pairwise_ablation": no_physics_metrics,
            }
        )
    return {
        "method": (
            "leave-one-(model, train_type, dataset, target_gbs)-scenario-out; "
            "physical efficiency and pairwise residual are both refit without "
            "the held-out scenario"
        ),
        "scenario_folds": len(folds),
        "feature_set": "physical",
        "feature_dimension": 28,
        "ridge_alpha": alpha,
        "use_physics_base": True,
        "physical_solver_converged_folds": sum(
            fold["physical_solver_converged"] for fold in folds
        ),
        "physical_baseline": _aggregate_fold_metrics(
            folds, "physical_baseline"
        ),
        "pairwise_residual": _aggregate_fold_metrics(
            folds, "pairwise_residual"
        ),
        "configuration_only_pairwise_ablation": _aggregate_fold_metrics(
            folds, "configuration_only_pairwise_ablation"
        ),
        "folds": folds,
    }


def _model_holdout(
    records: Sequence[Mapping[str, Any]],
    *,
    priors: Mapping[str, Any],
    alpha: float,
) -> dict[str, Any]:
    model_ids = sorted(
        {str(record["scenario"]["model_id"]) for record in records}
    )
    folds = []
    for held_out in model_ids:
        train_records = [
            record
            for record in records
            if str(record["scenario"]["model_id"]) != held_out
        ]
        test_records = [
            record
            for record in records
            if str(record["scenario"]["model_id"]) == held_out
        ]
        physical_model = _fit_compute_model(train_records, priors)
        if physical_model.get("available") is not True:
            raise ValueError(
                f"physical fit unavailable for model holdout {held_out}: "
                f"{physical_model.get('blockers')}"
            )
        train_candidates = _throughput_candidates(
            train_records,
            physical_model=physical_model,
            priors=priors,
        )
        test_candidates = _throughput_candidates(
            test_records,
            physical_model=physical_model,
            priors=priors,
        )
        challenger = _fit_pairwise_ranker(
            train_candidates,
            feature_set="physical",
            alpha=alpha,
            historical_weight=0.0,
            use_physics_base=True,
        )
        no_physics_ablation = _fit_pairwise_ranker(
            train_candidates,
            feature_set="basic",
            alpha=alpha,
            historical_weight=0.0,
            use_physics_base=False,
        )
        baseline_metrics = _ranking_evaluation(
            test_candidates,
            _baseline_ranker_model("physical"),
            include_details=False,
        )
        challenger_metrics = _ranking_evaluation(
            test_candidates,
            challenger,
            include_details=False,
        )
        no_physics_metrics = _ranking_evaluation(
            test_candidates,
            no_physics_ablation,
            include_details=False,
        )
        test_scenarios, comparable_scenarios = _candidate_group_counts(
            test_candidates
        )
        folds.append(
            {
                "held_out_model_id": held_out,
                "train_records": len(train_records),
                "test_records": len(test_records),
                "train_candidates": len(train_candidates),
                "test_candidates": len(test_candidates),
                "test_scenarios": test_scenarios,
                "comparable_scenarios": comparable_scenarios,
                "physical_solver_converged": bool(
                    physical_model.get("bounded_solver_converged")
                ),
                "physical_baseline": baseline_metrics,
                "pairwise_residual": challenger_metrics,
                "configuration_only_pairwise_ablation": no_physics_metrics,
            }
        )
    return {
        "method": (
            "leave-one-complete-model-id-out; physical efficiency and "
            "pairwise residual are both refit without that model scale"
        ),
        "model_folds": len(folds),
        "feature_set": "physical",
        "feature_dimension": 28,
        "ridge_alpha": alpha,
        "use_physics_base": True,
        "physical_solver_converged_folds": sum(
            fold["physical_solver_converged"] for fold in folds
        ),
        "physical_baseline": _aggregate_fold_metrics(
            folds, "physical_baseline"
        ),
        "pairwise_residual": _aggregate_fold_metrics(
            folds, "pairwise_residual"
        ),
        "configuration_only_pairwise_ablation": _aggregate_fold_metrics(
            folds, "configuration_only_pairwise_ablation"
        ),
        "folds": folds,
    }


def _fit_final_model(
    records: Sequence[Mapping[str, Any]],
    *,
    priors: Mapping[str, Any],
    alpha: float,
) -> dict[str, Any]:
    physical_model = _fit_compute_model(records, priors)
    candidates = _throughput_candidates(
        records,
        physical_model=physical_model,
        priors=priors,
    )
    ranker = _fit_pairwise_ranker(
        candidates,
        feature_set="physical",
        alpha=alpha,
        historical_weight=0.0,
        use_physics_base=True,
    )
    return {
        "physical_model": physical_model,
        "ranker": ranker,
        "in_sample_diagnostic": _ranking_evaluation(
            candidates, ranker, include_details=False
        ),
    }


def build_report(
    campaign_root: Path,
    *,
    matrix_name: str,
    hbm_bandwidth_bytes_s: float,
    collective_bandwidth_bytes_s: float,
    alpha: float,
) -> dict[str, Any]:
    hardware_path = campaign_root / "config" / "hardware.json"
    collected_path = campaign_root / "artifacts" / "collected_results.json"
    inventory_path = campaign_root / "artifacts" / "model_inventory.json"
    runtime_manifest_path = campaign_root / "runtime" / "runtime_manifest.json"
    hardware = read_json(hardware_path)
    if "4090" not in str(hardware.get("name_reported_by_driver") or ""):
        raise ValueError("This analysis accepts only the isolated RTX 4090 campaign")
    records, admission = _build_records(
        campaign_root,
        matrix_name=matrix_name,
        hbm_bandwidth_bytes_s=hbm_bandwidth_bytes_s,
        collective_bandwidth_bytes_s=collective_bandwidth_bytes_s,
    )
    if len(records) < 30:
        raise ValueError("Too few admitted 4090 throughput records")
    priors = _physical_priors(
        hardware,
        hbm_bandwidth_bytes_s=hbm_bandwidth_bytes_s,
        collective_bandwidth_bytes_s=collective_bandwidth_bytes_s,
    )
    validation = _scenario_loso(records, priors=priors, alpha=alpha)
    validation["model_holdout"] = _model_holdout(
        records, priors=priors, alpha=alpha
    )
    final_model = _fit_final_model(records, priors=priors, alpha=alpha)
    blockers = [
        "retrospective_only_no_predeclared_prospective_holdout",
        "clock_or_power_limited_rows_are_not_excluded_because_the_clean_subset_is_too_sparse",
        "collective_bandwidth_is_a_nominal_PCIe_prior_not_a_measured_NCCL_calibration",
        "packing_effects_are_excluded",
        "only_qwen3_0p6b_1p7b_and_4b_are_covered",
        "candidate_set_was_selected_by_prior_screening_not_random_full_search",
        "held_out_workload_features_use_run_counters_instead_of_prospective_dataset_expectations",
    ]
    if (
        validation["physical_solver_converged_folds"]
        < validation["scenario_folds"]
        or final_model["physical_model"].get("bounded_solver_converged")
        is not True
    ):
        blockers.append(
            "bounded_physical_solver_did_not_converge_in_all_validation_folds"
        )
    report = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_root": str(campaign_root.resolve()),
        "publishable": False,
        "analysis_only": True,
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "formula": (
            "score = log(T_physical) + intercept + "
            "standardized_28_features @ beta"
        ),
        "target": "effective_tokens_per_second",
        "inputs": {
            "hardware": {
                "path": str(hardware_path),
                "sha256": sha256_file(hardware_path),
            },
            "collected_results": {
                "path": str(collected_path),
                "sha256": sha256_file(collected_path),
            },
            "model_inventory": {
                "path": str(inventory_path),
                "sha256": sha256_file(inventory_path),
            },
            "runtime_manifest": {
                "path": str(runtime_manifest_path),
                "sha256": sha256_file(runtime_manifest_path),
            },
        },
        "physical_priors": {
            **priors,
            "hbm_bandwidth_source": (
                "NVIDIA Ada architecture RTX 4090 nominal specification"
            ),
            "collective_bandwidth_source": (
                "explicit exploratory PCIe prior; sensitivity must be checked"
            ),
        },
        "admission": admission,
        "validation": validation,
        "final_model": final_model,
        "blockers": blockers,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT,
    )
    parser.add_argument(
        "--matrix-name",
        default="throughput_jobs.jsonl",
    )
    parser.add_argument(
        "--hbm-bandwidth-bytes-s",
        type=float,
        default=DEFAULT_HBM_BANDWIDTH_BYTES_S,
    )
    parser.add_argument(
        "--collective-bandwidth-bytes-s",
        type=float,
        default=DEFAULT_COLLECTIVE_BANDWIDTH_BYTES_S,
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    campaign_root = args.campaign_root.resolve()
    output = args.output or (
        campaign_root / "artifacts" / "rtx4090_challenger_modeling.json"
    )
    report = build_report(
        campaign_root,
        matrix_name=args.matrix_name,
        hbm_bandwidth_bytes_s=args.hbm_bandwidth_bytes_s,
        collective_bandwidth_bytes_s=args.collective_bandwidth_bytes_s,
        alpha=args.alpha,
    )
    write_json(output, report)
    baseline = report["validation"]["physical_baseline"]
    challenger = report["validation"]["pairwise_residual"]
    print(
        json.dumps(
            {
                "output": str(output),
                "admitted": report["admission"]["admitted_success_rows"],
                "scenarios": report["validation"]["scenario_folds"],
                "physical_baseline": baseline,
                "pairwise_residual": challenger,
                "model_holdout": report["validation"]["model_holdout"],
                "publishable": report["publishable"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
