#!/usr/bin/env python3
"""Evaluate the frozen H800 physical-shares + v4b fresh holdout.

This script is intentionally evaluation-only.  It joins the frozen prediction,
the pre-run design/queue and the post-run observations without refitting either
model or mutating the canonical observation/anchor libraries.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Mapping, Sequence

from common import ARTIFACT_DIR, MATRIX_DIR, ROOT, read_json, read_jsonl, sha256_file, write_json
from prospective_acceptance import evaluate_memory_acceptance, evaluate_ranking_acceptance


SCHEMA = "sft_h800_fresh_holdout_acceptance/v2"
DEFAULT_CAMPAIGN_ID = "h800_fresh_business_physical_v4b_holdout_20260802"
DEFAULT_QUEUE = MATRIX_DIR / "h800_fresh_holdout_jobs_v2.jsonl"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_fresh_holdout_design_v2.json"
DEFAULT_PREDICTIONS = ARTIFACT_DIR / "h800_frozen_predictions_before_fresh_holdout_v2.json"
DEFAULT_RESULTS = ARTIFACT_DIR / "collected_results.json"
DEFAULT_SPLIT = ARTIFACT_DIR / "fresh_holdout_v2" / "split_manifest.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_fresh_holdout_acceptance_v2.json"
DEFAULT_MARKDOWN = ARTIFACT_DIR / "h800_fresh_holdout_acceptance_v2.md"
DEFAULT_SNAPSHOT = ARTIFACT_DIR / "h800_fresh_holdout_observations_v2.json"
APPROVAL_RECEIPTS = ROOT / "runtime" / "approval_promotion_receipts"
GIB = float(1024**3)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return fmean(clean) if clean else None


def _percentile(values: Iterable[float | None], q: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not clean:
        return None
    position = (len(clean) - 1) * float(q) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def _rank_desc(rows: Sequence[Mapping[str, Any]], value_key: str) -> dict[str, float]:
    """Return average descending ranks, retaining deterministic tie handling."""

    ordered = sorted(
        ((str(row["candidate_id"]), float(row[value_key])) for row in rows),
        key=lambda item: (-item[1], item[0]),
    )
    result: dict[str, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and math.isclose(
            ordered[end][1], ordered[cursor][1], rel_tol=1e-12, abs_tol=1e-12
        ):
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for candidate_id, _ in ordered[cursor:end]:
            result[candidate_id] = average_rank
        cursor = end
    return result


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator > 0.0 else None


def _pairwise_accuracy(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    correct = 0
    total = 0
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            predicted_difference = float(left["predicted_throughput_proxy"]) - float(
                right["predicted_throughput_proxy"]
            )
            observed_difference = float(left["observed_effective_tokens_per_second"]) - float(
                right["observed_effective_tokens_per_second"]
            )
            if math.isclose(predicted_difference, 0.0, abs_tol=1e-12) or math.isclose(
                observed_difference, 0.0, abs_tol=1e-12
            ):
                continue
            total += 1
            correct += int(predicted_difference * observed_difference > 0.0)
    return correct, total


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-6)
    return left == right


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _format_number(value: Any, digits: int = 3) -> str:
    finite = _finite(value)
    return "—" if finite is None else f"{finite:.{digits}f}"


def _format_percent(value: Any, digits: int = 1) -> str:
    finite = _finite(value)
    return "—" if finite is None else f"{100.0 * finite:.{digits}f}%"


def _config_mismatches(job: Mapping[str, Any], prediction: Mapping[str, Any]) -> list[str]:
    configuration = prediction.get("configuration") or {}
    field_pairs = (
        ("scenario_id", job.get("scenario_id"), prediction.get("comparison_group")),
        ("model_id", job.get("model_id"), configuration.get("model_id")),
        ("train_type", job.get("train_type"), configuration.get("training_mode")),
        ("dataset_id", job.get("dataset_id"), configuration.get("dataset_id")),
        ("dataset_profile_sha256", job.get("dataset_profile_sha256"), configuration.get("dataset_profile_sha256")),
        ("cutoff_len", job.get("cutoff_len"), configuration.get("cutoff_len")),
        ("target_gbs", job.get("target_gbs"), configuration.get("target_gbs")),
        ("gpu_count", job.get("gpu_count"), configuration.get("gpu_count")),
        ("mbs", job.get("mbs"), configuration.get("physical_mbs")),
        (
            "gradient_accumulation_steps",
            job.get("gradient_accumulation_steps"),
            configuration.get("gradient_accumulation_steps"),
        ),
        ("zero_stage", job.get("zero_stage"), configuration.get("zero_stage")),
        ("gc", job.get("gc"), configuration.get("gradient_checkpointing")),
        ("packing", job.get("packing"), configuration.get("packing")),
        ("offload", job.get("offload"), configuration.get("offload")),
    )
    return [name for name, left, right in field_pairs if not _same(left, right)]


def _result_mismatches(job: Mapping[str, Any], result: Mapping[str, Any]) -> list[str]:
    fields = (
        "campaign_id",
        "candidate_slot_id",
        "scenario_id",
        "predictor_request_id",
        "model_id",
        "train_type",
        "dataset_id",
        "dataset_profile_sha256",
        "data_sha256",
        "cutoff_len",
        "target_gbs",
        "gpu_count",
        "mbs",
        "gradient_accumulation_steps",
        "zero_stage",
        "gc",
        "packing",
        "offload",
    )
    return [field for field in fields if not _same(job.get(field), result.get(field))]


def _load_receipt_hashes() -> set[str]:
    hashes: set[str] = set()
    if not APPROVAL_RECEIPTS.is_dir():
        return hashes
    for path in sorted(APPROVAL_RECEIPTS.glob("*.json")):
        payload = read_json(path)
        value = payload.get("candidate_sha256") if isinstance(payload, Mapping) else None
        if value:
            hashes.add(str(value))
    return hashes


def evaluate(
    *,
    campaign_id: str,
    queue_path: Path,
    design_path: Path,
    prediction_path: Path,
    result_path: Path,
    split_path: Path,
) -> dict[str, Any]:
    queue = read_jsonl(queue_path)
    design = read_json(design_path)
    frozen = read_json(prediction_path)
    collected = read_json(result_path)
    split = read_json(split_path)

    result_rows = [
        row for row in collected.get("rows", []) if row.get("campaign_id") == campaign_id
    ]
    predictions = frozen.get("predictions") or []
    slots = design.get("candidate_slots") or []
    scenarios = design.get("scenarios") or []

    queue_by_job = {str(row["job_id"]): row for row in queue}
    result_by_job = {str(row["job_id"]): row for row in result_rows}
    prediction_by_request = {str(row["request_id"]): row for row in predictions}
    slot_by_id = {str(row["candidate_slot_id"]): row for row in slots}
    scenario_by_id = {str(row["scenario_id"]): row for row in scenarios}

    errors: list[dict[str, Any]] = []
    duplicate_checks = {
        "queue_job_ids": _duplicates(str(row["job_id"]) for row in queue),
        "queue_candidate_slot_ids": _duplicates(str(row["candidate_slot_id"]) for row in queue),
        "queue_predictor_request_ids": _duplicates(str(row["predictor_request_id"]) for row in queue),
        "result_job_ids": _duplicates(str(row["job_id"]) for row in result_rows),
        "prediction_request_ids": _duplicates(str(row["request_id"]) for row in predictions),
        "design_candidate_slot_ids": _duplicates(str(row["candidate_slot_id"]) for row in slots),
    }
    for name, duplicates in duplicate_checks.items():
        if duplicates:
            errors.append({"check": name, "duplicates": duplicates})

    queue_ids = set(queue_by_job)
    result_ids = set(result_by_job)
    if queue_ids != result_ids:
        errors.append(
            {
                "check": "queue_result_job_id_set",
                "missing_results": sorted(queue_ids - result_ids),
                "unexpected_results": sorted(result_ids - queue_ids),
            }
        )
    if set(slot_by_id) != {str(row["candidate_slot_id"]) for row in queue}:
        errors.append({"check": "queue_design_candidate_slot_set"})
    if any(row.get("campaign_id") != campaign_id for row in queue):
        errors.append({"check": "queue_campaign_id"})
    if design.get("campaign_id") != campaign_id:
        errors.append({"check": "design_campaign_id", "actual": design.get("campaign_id")})

    bound_prediction = (design.get("frozen_bindings") or {}).get("frozen_predictions") or {}
    actual_prediction_sha = sha256_file(prediction_path)
    if bound_prediction.get("sha256") != actual_prediction_sha:
        errors.append(
            {
                "check": "frozen_prediction_sha256",
                "expected": bound_prediction.get("sha256"),
                "actual": actual_prediction_sha,
            }
        )

    receipt_hashes = _load_receipt_hashes()
    result_approval_hashes = sorted(
        {str(row.get("approval_design_sha256")) for row in result_rows if row.get("approval_design_sha256")}
    )
    unknown_approval_hashes = sorted(set(result_approval_hashes) - receipt_hashes)
    if unknown_approval_hashes:
        errors.append({"check": "approval_receipts", "unknown_hashes": unknown_approval_hashes})

    current_hashes: dict[str, str] = {}
    joined: list[dict[str, Any]] = []
    for job_id in sorted(queue_ids):
        job = queue_by_job[job_id]
        result = result_by_job.get(job_id)
        prediction = prediction_by_request.get(str(job["predictor_request_id"]))
        slot = slot_by_id.get(str(job["candidate_slot_id"]))
        if result is None or prediction is None or slot is None:
            errors.append(
                {
                    "check": "join",
                    "job_id": job_id,
                    "result_found": result is not None,
                    "prediction_found": prediction is not None,
                    "slot_found": slot is not None,
                }
            )
            continue

        config_mismatches = _config_mismatches(job, prediction)
        result_mismatches = _result_mismatches(job, result)
        if config_mismatches:
            errors.append({"check": "queue_prediction_configuration", "job_id": job_id, "fields": config_mismatches})
        if result_mismatches:
            errors.append({"check": "queue_result_configuration", "job_id": job_id, "fields": result_mismatches})
        if slot.get("request_id") != job.get("predictor_request_id"):
            errors.append({"check": "slot_request_id", "job_id": job_id})

        embedded = job.get("frozen_prediction") or {}
        embedded_pairs = (
            ("memory_upper_reserved_bytes", (prediction.get("memory") or {}).get("admission_upper_reserved_bytes")),
            ("safe_limit_bytes", (prediction.get("memory") or {}).get("safe_limit_bytes")),
            ("v4b_rank_within_gpu_count", prediction.get("rank_within_gpu_count")),
            (
                "v4b_throughput_proxy",
                (prediction.get("throughput") or {}).get("throughput_proxy_tokens_per_second"),
            ),
        )
        embedded_mismatches = [name for name, expected in embedded_pairs if not _same(embedded.get(name), expected)]
        if embedded_mismatches:
            errors.append({"check": "embedded_frozen_prediction", "job_id": job_id, "fields": embedded_mismatches})

        for path_field, sha_field in (
            ("data_path", "data_sha256"),
            ("dataset_profile_path", "dataset_profile_sha256"),
        ):
            path = Path(str(job[path_field]))
            cache_key = str(path)
            if cache_key not in current_hashes:
                current_hashes[cache_key] = sha256_file(path) if path.is_file() else "missing"
            if current_hashes[cache_key] != job.get(sha_field):
                errors.append(
                    {
                        "check": f"current_{sha_field}",
                        "job_id": job_id,
                        "expected": job.get(sha_field),
                        "actual": current_hashes[cache_key],
                    }
                )

        memory = prediction.get("memory") or {}
        throughput = prediction.get("throughput") or {}
        observed_reserved = _finite(result.get("max_reserved_bytes"))
        observed_throughput = _finite(result.get("effective_tokens_per_second"))
        joined.append(
            {
                "job_id": job_id,
                "candidate_id": str(job["predictor_request_id"]),
                "candidate_slot_id": job["candidate_slot_id"],
                "scenario_id": job["scenario_id"],
                "dataset_id": job["dataset_id"],
                "dataset_category": job["dataset_category"],
                "model_id": job["model_id"],
                "train_type": job["train_type"],
                "cutoff_len": int(job["cutoff_len"]),
                "gpu_count": int(job["gpu_count"]),
                "mbs": int(job["mbs"]),
                "gradient_accumulation_steps": int(job["gradient_accumulation_steps"]),
                "zero_stage": int(job["zero_stage"]),
                "gc": bool(job["gc"]),
                "packing": bool(job["packing"]),
                "offload": bool(job["offload"]),
                "outcome": result.get("classification"),
                "predicted_admit": memory.get("admitted") is True,
                "memory_admission_source": memory.get("admission_source"),
                "memory_anchor_override_applied": memory.get("anchor_override_applied") is True,
                "predicted_memory_center_bytes": _finite(memory.get("reserved_center_bytes")),
                "predicted_memory_upper_bytes": _finite(memory.get("admission_upper_reserved_bytes")),
                "safe_limit_bytes": _finite(memory.get("safe_limit_bytes")),
                "observed_allocated_bytes": _finite(result.get("max_allocated_bytes")),
                "observed_reserved_bytes": observed_reserved,
                "observed_nvidia_smi_peak_mib": _finite(result.get("nvidia_smi_peak_mib")),
                "predicted_throughput_proxy": _finite(throughput.get("throughput_proxy_tokens_per_second")),
                "frozen_rank_within_gpu_count": prediction.get("rank_within_gpu_count"),
                "observed_effective_tokens_per_second": observed_throughput,
                "observed_computed_tokens_per_second": _finite(result.get("computed_tokens_per_second")),
                "observed_samples_per_second": _finite(result.get("samples_per_second")),
                "measured_seconds": _finite(result.get("measured_seconds")),
                "mfu": _finite(result.get("mfu")),
                "gpu_mask": result.get("gpu_mask"),
                "clock_status": result.get("clock_status"),
                "execution_fingerprint_quality": result.get("execution_fingerprint_quality"),
                "calibration_eligible": result.get("calibration_eligible") is True,
                "rank_summaries": result.get("rank_summaries"),
                "metrics_available": result.get("metrics_available") is True,
                "approval_design_sha256": result.get("approval_design_sha256"),
            }
        )

    split_check = split.get("prior_modeling_overlap_check") or {}
    queue_dataset_ids = sorted({str(row["dataset_id"]) for row in queue})
    split_dataset_ids = sorted(str(value) for value in split_check.get("selected_dataset_ids") or [])
    freshness_passes = bool(
        split_check.get("passed") is True
        and not split_check.get("matched_existing_canonical_observation_dataset_ids")
        and queue_dataset_ids == split_dataset_ids
    )
    if not freshness_passes:
        errors.append(
            {
                "check": "fresh_scenario_split",
                "queue_dataset_ids": queue_dataset_ids,
                "split_dataset_ids": split_dataset_ids,
                "overlap_check": split_check,
            }
        )

    outcome_counts = Counter(str(row["outcome"]) for row in joined)
    execution_checks = {
        "all_metrics_available": all(row["metrics_available"] for row in joined),
        "all_execution_fingerprints_complete": all(
            row["execution_fingerprint_quality"] == "complete" for row in joined
        ),
        "all_calibration_eligible": all(row["calibration_eligible"] for row in joined),
        "all_rank_summary_counts_match_gpu_count": all(
            int(row["rank_summaries"] or -1) == int(row["gpu_count"]) for row in joined
        ),
        "all_frozen_predictions_admitted": all(row["predicted_admit"] for row in joined),
    }
    for name, passed in execution_checks.items():
        if not passed:
            errors.append({"check": name})

    integrity = {
        "passes": not errors,
        "campaign_id": campaign_id,
        "queue_jobs": len(queue),
        "design_candidate_slots": len(slots),
        "frozen_prediction_rows_total": len(predictions),
        "joined_frozen_prediction_rows": len(joined),
        "collected_campaign_rows": len(result_rows),
        "scenario_count": len(scenarios),
        "outcomes": dict(sorted(outcome_counts.items())),
        "duplicate_checks": duplicate_checks,
        "execution_checks": execution_checks,
        "fresh_scenario_split_passes": freshness_passes,
        "approval_design_sha256_counts": dict(
            sorted(Counter(str(row["approval_design_sha256"]) for row in joined).items())
        ),
        "approval_hashes_have_promotion_receipts": not unknown_approval_hashes,
        "frozen_prediction_sha256": actual_prediction_sha,
        "errors": errors,
    }

    valid_partition_roles = {"calibration", "holdout"}
    canonical_partition_rows = sum(
        isinstance(job.get("calibration_partition"), Mapping)
        and (job.get("calibration_partition") or {}).get("role") in valid_partition_roles
        and isinstance((job.get("calibration_partition") or {}).get("split_unit_id"), str)
        and bool((job.get("calibration_partition") or {}).get("split_unit_id"))
        and isinstance((job.get("calibration_partition") or {}).get("policy"), str)
        and bool((job.get("calibration_partition") or {}).get("policy"))
        for job in queue
    )
    ingestion_eligibility = {
        "prospective_acceptance_evidence_eligible": integrity["passes"],
        "canonical_export_partition_rows": canonical_partition_rows,
        "canonical_export_partition_required_rows": len(queue),
        "canonical_fit_or_anchor_ingestion_eligible": canonical_partition_rows == len(queue),
        "holdout_may_be_reused_to_validate_a_refit": False,
        "status": (
            "acceptance_only_missing_job_bound_calibration_partition"
            if canonical_partition_rows != len(queue)
            else "job_bound_holdout_partition_present"
        ),
        "reason": (
            "The pre-run campaign/design proves prospective acceptance use, but the v2 materialized job "
            "payload omitted calibration_partition.  The canonical exporter therefore correctly excludes "
            "these rows from coefficient fitting and anchor promotion."
        ),
    }

    memory_rows = []
    memory_details = []
    for row in joined:
        observed = row["observed_reserved_bytes"]
        center = row["predicted_memory_center_bytes"]
        upper = row["predicted_memory_upper_bytes"]
        safe_limit = row["safe_limit_bytes"]
        success = row["outcome"] == "success"
        memory_rows.append(
            {
                "scenario_id": row["scenario_id"],
                "outcome": row["outcome"],
                "predicted_admit": row["predicted_admit"],
                "observed_reserved_bytes": observed,
                "upper_reserved_bytes": upper,
                "safe_limit_bytes": safe_limit,
                "actual_safe_success": bool(
                    success and observed is not None and safe_limit is not None and observed <= safe_limit
                ),
            }
        )
        memory_details.append(
            {
                **{key: row[key] for key in (
                    "job_id",
                    "candidate_id",
                    "scenario_id",
                    "model_id",
                    "train_type",
                    "gpu_count",
                    "mbs",
                    "zero_stage",
                    "gc",
                    "outcome",
                    "predicted_admit",
                    "memory_admission_source",
                    "memory_anchor_override_applied",
                )},
                "predicted_center_gib": center / GIB if center is not None else None,
                "predicted_upper_gib": upper / GIB if upper is not None else None,
                "observed_reserved_gib": observed / GIB if observed is not None else None,
                "safe_limit_gib": safe_limit / GIB if safe_limit is not None else None,
                "center_signed_relative_error": (
                    (center - observed) / observed if center is not None and observed and observed > 0.0 else None
                ),
                "center_absolute_relative_error": (
                    abs(center - observed) / observed if center is not None and observed and observed > 0.0 else None
                ),
                "upper_headroom_gib": (
                    (upper - observed) / GIB if upper is not None and observed is not None else None
                ),
                "upper_covers_observed": bool(
                    success and upper is not None and observed is not None and upper >= observed
                ) if success else None,
                "observed_headroom_to_safe_limit_gib": (
                    (safe_limit - observed) / GIB
                    if safe_limit is not None and observed is not None
                    else None
                ),
            }
        )

    contract = design.get("acceptance_contract") or {}
    memory_standard = evaluate_memory_acceptance(
        memory_rows,
        minimum_coverage=float(contract.get("memory_p95_coverage_minimum", 0.95)),
        maximum_false_safe_oom=int(contract.get("memory_false_safe_oom", 0)),
    )
    memory = {
        **memory_standard,
        "operational_safety_passes": memory_standard["memory_safety_failures"] == 0,
        "center_mape": _mean(row["center_absolute_relative_error"] for row in memory_details),
        "center_median_absolute_relative_error": (
            median(
                row["center_absolute_relative_error"]
                for row in memory_details
                if row["center_absolute_relative_error"] is not None
            )
            if any(row["center_absolute_relative_error"] is not None for row in memory_details)
            else None
        ),
        "center_p90_absolute_relative_error": _percentile(
            (row["center_absolute_relative_error"] for row in memory_details), 90.0
        ),
        "center_mean_signed_relative_error": _mean(
            row["center_signed_relative_error"] for row in memory_details
        ),
        "upper_mean_headroom_gib": _mean(row["upper_headroom_gib"] for row in memory_details),
        "upper_p90_headroom_gib": _percentile(
            (row["upper_headroom_gib"] for row in memory_details), 90.0
        ),
        "minimum_observed_headroom_to_safe_limit_gib": min(
            row["observed_headroom_to_safe_limit_gib"]
            for row in memory_details
            if row["observed_headroom_to_safe_limit_gib"] is not None
        ) if memory_details else None,
        "detail_rows": memory_details,
        "interpretation": {
            "safety_gate_is_primary": True,
            "center_absolute_error_is_diagnostic_only": True,
            "selected_admitted_candidates_only": True,
            "filtered_candidate_false_negative_rate_is_not_measured": True,
            "oom_boundary_stress_evidence_present": any(row["outcome"] == "oom" for row in joined),
        },
    }

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        grouped[(str(row["scenario_id"]), int(row["gpu_count"]))].append(row)

    standard_groups = []
    ranking_details = []
    pooled_pairwise_correct = 0
    pooled_pairwise_total = 0
    for (scenario_id, gpu_count), rows in sorted(grouped.items()):
        endpoint_id = f"{scenario_id}__g{gpu_count}"
        standard_candidates = [
            {
                "candidate_id": row["candidate_id"],
                "outcome": row["outcome"],
                "predicted_admit": row["predicted_admit"],
                "actual_safe_success": bool(
                    row["outcome"] == "success"
                    and row["observed_reserved_bytes"] is not None
                    and row["safe_limit_bytes"] is not None
                    and row["observed_reserved_bytes"] <= row["safe_limit_bytes"]
                ),
                "predicted_throughput": row["predicted_throughput_proxy"],
                "observed_throughput": row["observed_effective_tokens_per_second"],
            }
            for row in rows
        ]
        standard_groups.append({"scenario_id": endpoint_id, "candidates": standard_candidates})

        eligible = [
            row
            for row in rows
            if row["predicted_throughput_proxy"] is not None
            and row["observed_effective_tokens_per_second"] is not None
            and row["outcome"] == "success"
        ]
        predicted_winner = max(eligible, key=lambda row: row["predicted_throughput_proxy"])
        observed_best = max(eligible, key=lambda row: row["observed_effective_tokens_per_second"])
        selected_rate = float(predicted_winner["observed_effective_tokens_per_second"])
        best_rate = float(observed_best["observed_effective_tokens_per_second"])
        regret = 1.0 - selected_rate / best_rate
        pairwise_correct, pairwise_total = _pairwise_accuracy(eligible)
        pooled_pairwise_correct += pairwise_correct
        pooled_pairwise_total += pairwise_total
        predicted_ranks = _rank_desc(eligible, "predicted_throughput_proxy")
        observed_ranks = _rank_desc(eligible, "observed_effective_tokens_per_second")
        ids = sorted(predicted_ranks)
        spearman = _pearson(
            [predicted_ranks[candidate_id] for candidate_id in ids],
            [observed_ranks[candidate_id] for candidate_id in ids],
        )
        candidates = []
        for row in sorted(eligible, key=lambda item: -float(item["predicted_throughput_proxy"])):
            predicted = float(row["predicted_throughput_proxy"])
            observed = float(row["observed_effective_tokens_per_second"])
            candidates.append(
                {
                    "candidate_id": row["candidate_id"],
                    "job_id": row["job_id"],
                    "mbs": row["mbs"],
                    "gradient_accumulation_steps": row["gradient_accumulation_steps"],
                    "zero_stage": row["zero_stage"],
                    "gc": row["gc"],
                    "predicted_throughput_proxy": predicted,
                    "observed_effective_tokens_per_second": observed,
                    "actual_to_proxy_ratio": observed / predicted if predicted > 0.0 else None,
                    "predicted_rank_in_measured_subset": predicted_ranks[row["candidate_id"]],
                    "observed_rank_in_measured_subset": observed_ranks[row["candidate_id"]],
                    "frozen_rank_within_full_admitted_gpu_group": row["frozen_rank_within_gpu_count"],
                }
            )
        ranking_details.append(
            {
                "endpoint_id": endpoint_id,
                "scenario_id": scenario_id,
                "gpu_count": gpu_count,
                "measured_candidate_count": len(eligible),
                "full_memory_admitted_candidate_count": next(
                    (
                        int(endpoint["memory_admitted_candidates"])
                        for endpoint in scenario_by_id[scenario_id].get("endpoint_summary", [])
                        if int(endpoint["gpu_count"]) == gpu_count
                    ),
                    None,
                ),
                "predicted_winner": predicted_winner["candidate_id"],
                "observed_best": observed_best["candidate_id"],
                "top1_exact": predicted_winner["candidate_id"] == observed_best["candidate_id"],
                "selected_observed_throughput": selected_rate,
                "best_observed_throughput": best_rate,
                "selected_fraction_of_best": selected_rate / best_rate,
                "top1_regret": regret,
                "top1_within_10_percent": regret <= float(contract.get("top1_regret_target", 0.10)) + 1e-12,
                "pairwise_correct": pairwise_correct,
                "pairwise_total": pairwise_total,
                "pairwise_accuracy": pairwise_correct / pairwise_total if pairwise_total else None,
                "spearman_rank_correlation": spearman,
                "candidates": candidates,
            }
        )

    ranking_standard = evaluate_ranking_acceptance(
        standard_groups,
        maximum_top1_regret=float(contract.get("top1_regret_target", 0.10)),
    )
    absolute_rows = [
        row
        for row in joined
        if row["predicted_throughput_proxy"] is not None
        and row["observed_effective_tokens_per_second"] is not None
        and row["predicted_throughput_proxy"] > 0.0
        and row["observed_effective_tokens_per_second"] > 0.0
    ]
    apes = [
        abs(row["predicted_throughput_proxy"] - row["observed_effective_tokens_per_second"])
        / row["observed_effective_tokens_per_second"]
        for row in absolute_rows
    ]
    log_errors = [
        math.log(row["predicted_throughput_proxy"] / row["observed_effective_tokens_per_second"])
        for row in absolute_rows
    ]
    regrets = [float(row["top1_regret"]) for row in ranking_details]
    ranking = {
        **ranking_standard,
        "endpoint_count": len(ranking_details),
        "exact_top1_count": sum(row["top1_exact"] for row in ranking_details),
        "exact_top1_accuracy": _mean(float(row["top1_exact"]) for row in ranking_details),
        "endpoints_within_10_percent_count": sum(row["top1_within_10_percent"] for row in ranking_details),
        "endpoints_within_10_percent_fraction": _mean(
            float(row["top1_within_10_percent"]) for row in ranking_details
        ),
        "median_top1_regret": median(regrets) if regrets else None,
        "p90_top1_regret": _percentile(regrets, 90.0),
        "pooled_pairwise_correct": pooled_pairwise_correct,
        "pooled_pairwise_total": pooled_pairwise_total,
        "pooled_pairwise_accuracy": (
            pooled_pairwise_correct / pooled_pairwise_total if pooled_pairwise_total else None
        ),
        "scenario_equal_pairwise_accuracy": _mean(
            row["pairwise_accuracy"] for row in ranking_details
        ),
        "scenario_equal_spearman_rank_correlation": _mean(
            row["spearman_rank_correlation"] for row in ranking_details
        ),
        "absolute_proxy_diagnostics": {
            "gating_metric": False,
            "absolute_scale_trusted_by_frozen_predictor": frozen.get("absolute_throughput_scale_trusted") is True,
            "rows": len(absolute_rows),
            "mape": fmean(apes) if apes else None,
            "median_absolute_percentage_error": median(apes) if apes else None,
            "p90_absolute_percentage_error": _percentile(apes, 90.0),
            "weighted_absolute_percentage_error": (
                sum(
                    abs(row["predicted_throughput_proxy"] - row["observed_effective_tokens_per_second"])
                    for row in absolute_rows
                )
                / sum(row["observed_effective_tokens_per_second"] for row in absolute_rows)
                if absolute_rows
                else None
            ),
            "log_rmse": math.sqrt(fmean(error**2 for error in log_errors)) if log_errors else None,
            "median_actual_to_proxy_ratio": (
                median(
                    row["observed_effective_tokens_per_second"] / row["predicted_throughput_proxy"]
                    for row in absolute_rows
                )
                if absolute_rows
                else None
            ),
        },
        "detail_groups": ranking_details,
        "interpretation": {
            "group_unit": "scenario_id + gpu_count",
            "primary_metric": "measured-subset top1 regret after frozen memory admission",
            "exact_top1_is_secondary": True,
            "absolute_proxy_error_is_non_gating": True,
            "candidate_subset_warning": (
                "Only the prospectively selected witness subset was measured; regret is relative to "
                "that subset, not every memory-admitted candidate."
            ),
        },
    }

    detail_by_candidate = {row["candidate_id"]: row for row in joined}
    scale_groups = []
    threshold = float((frozen.get("scale_out") or {}).get("minimum_ratio", 1.8))
    for group in (frozen.get("scale_out") or {}).get("groups") or []:
        for step in group.get("scaling_steps") or []:
            baseline_id = str(step["baseline_candidate"])
            expanded_id = str(step["expanded_candidate"])
            baseline = detail_by_candidate.get(baseline_id)
            expanded = detail_by_candidate.get(expanded_id)
            scenario_id = str(group["comparison_group"])
            baseline_gpu = int(step["from_gpu"])
            expanded_gpu = int(step["to_gpu"])
            baseline_group = next(
                row
                for row in ranking_details
                if row["scenario_id"] == scenario_id and row["gpu_count"] == baseline_gpu
            )
            expanded_group = next(
                row
                for row in ranking_details
                if row["scenario_id"] == scenario_id and row["gpu_count"] == expanded_gpu
            )
            realized_ratio = (
                expanded["observed_effective_tokens_per_second"]
                / baseline["observed_effective_tokens_per_second"]
                if baseline
                and expanded
                and baseline["observed_effective_tokens_per_second"]
                and expanded["observed_effective_tokens_per_second"]
                else None
            )
            oracle_ratio = (
                expanded_group["best_observed_throughput"]
                / baseline_group["best_observed_throughput"]
            )
            scale_groups.append(
                {
                    "scenario_id": scenario_id,
                    "from_gpu": baseline_gpu,
                    "to_gpu": expanded_gpu,
                    "baseline_candidate": baseline_id,
                    "expanded_candidate": expanded_id,
                    "predicted_point_ratio": _finite(step.get("predicted_ratio")),
                    "realized_ratio_for_frozen_endpoint_winners": realized_ratio,
                    "measured_subset_oracle_ratio": oracle_ratio,
                    "throughput_per_gpu_ratio_for_frozen_endpoint_winners": (
                        realized_ratio / (expanded_gpu / baseline_gpu)
                        if realized_ratio is not None
                        else None
                    ),
                    "point_ratio_meets_threshold": bool(
                        realized_ratio is not None and realized_ratio >= threshold
                    ),
                    "conservative_prediction_lower": step.get("conservative_ratio_lower"),
                    "fresh_measured_ratio_lower": None,
                    "strict_scale_gate_passes": False,
                    "strict_scale_gate_status": "insufficient_repeats_for_measured_lower_bound",
                }
            )
    scaling = {
        "enabled_by_frozen_predictor": (frozen.get("scale_out") or {}).get("enabled") is True,
        "automatic_execution_allowed": False,
        "minimum_ratio": threshold,
        "pair_count": len(scale_groups),
        "point_ratio_meeting_threshold_count": sum(
            row["point_ratio_meets_threshold"] for row in scale_groups
        ),
        "strict_gate_passes": False,
        "strict_gate_status": "conservative prediction bounds and repeated measured lower bounds are unavailable",
        "minimum_card_policy_remains_active": True,
        "groups": scale_groups,
    }

    evidence_scope = {
        "fresh_dataset_scenarios": len(scenarios),
        "ranking_endpoints": len(ranking_details),
        "measured_configurations": len(joined),
        "exact_repeats_per_configuration": sorted({int(row.get("repeat", 0)) + 1 for row in queue}),
        "hardware_ids": sorted({str(row["hardware_id"]) for row in queue}),
        "model_ids": sorted({str(row["model_id"]) for row in queue}),
        "train_types": sorted({str(row["train_type"]) for row in queue}),
        "cutoff_lengths": sorted({int(row["cutoff_len"]) for row in queue}),
        "gpu_counts": sorted({int(row["gpu_count"]) for row in queue}),
        "packing_values": sorted({bool(row["packing"]) for row in queue}),
        "offload_values": sorted({bool(row["offload"]) for row in queue}),
        "gpu_masks": dict(sorted(Counter(str(row["gpu_mask"]) for row in joined).items())),
        "clock_statuses": dict(sorted(Counter(str(row["clock_status"]) for row in joined).items())),
        "thermal_slowdown_failures": sum(
            bool(result_by_job[row["job_id"]].get("thermal_observation", {}).get("any_hw_thermal_slowdown"))
            or bool(result_by_job[row["job_id"]].get("thermal_observation", {}).get("any_sw_thermal_slowdown"))
            for row in joined
        ),
        "not_covered": [
            "unmeasured memory-rejected candidates and false-negative rate",
            "the full admitted candidate set at each endpoint",
            "repeat-level throughput uncertainty",
            "packing=true and neat_packing",
            "optimizer/parameter offload",
            "vision-language and multimodal SFT",
            "MoE models",
            "model scales other than dense Qwen3 8B/14B",
            "GPU types other than H800",
        ],
    }

    memory_ranking_passes = bool(integrity["passes"] and memory["passes"] and ranking["passes"])
    decision_blockers = []
    if not integrity["passes"]:
        decision_blockers.append("evidence_integrity_failed")
    if not memory["passes"]:
        decision_blockers.append("memory_acceptance_failed")
    if not ranking["passes"]:
        decision_blockers.append("ranking_acceptance_failed")
    strict_generalization_blockers = list(decision_blockers)
    strict_generalization_blockers.extend(
        [
            "scale_out_confidence_bounds_unavailable",
            "only_four_fresh_scenarios",
            "measured_candidate_subsets_not_full_admitted_sets",
            "no_exact_repeats",
            "coverage_limited_to_dense_text_qwen3_8b_14b_on_h800",
        ]
    )
    decisions = {
        "frozen_memory_and_ranking_gate_passes": memory_ranking_passes,
        "strict_generalization_claim_allowed": not strict_generalization_blockers,
        "automatic_scale_out_allowed": False,
        "minimum_card_policy_remains_active": True,
        "safe_to_store_as_versioned_acceptance_evidence": integrity["passes"],
        "safe_to_append_to_fit_or_anchor_library": ingestion_eligibility[
            "canonical_fit_or_anchor_ingestion_eligible"
        ],
        "safe_to_refit_automatically": False,
        "refit_requires_explicit_versioned_challenger_evaluation": True,
        "memory_ranking_blockers": decision_blockers,
        "strict_generalization_blockers": strict_generalization_blockers,
    }

    evidence_completed_at_utc = datetime.fromtimestamp(
        max(float(row["finished_unix"]) for row in result_rows),
        timezone.utc,
    ).isoformat()
    return {
        "schema": SCHEMA,
        "generated_at_utc": evidence_completed_at_utc,
        "timestamp_semantics": "deterministic maximum terminal result finished_unix",
        "campaign_id": campaign_id,
        "evaluation_mode": "frozen_predictions_no_refit",
        "source_files": {
            "queue": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
            "design": {"path": str(design_path), "sha256": sha256_file(design_path)},
            "frozen_predictions": {"path": str(prediction_path), "sha256": actual_prediction_sha},
            "collected_results": {"path": str(result_path), "sha256": sha256_file(result_path)},
            "fresh_split_manifest": {"path": str(split_path), "sha256": sha256_file(split_path)},
        },
        "integrity": integrity,
        "ingestion_eligibility": ingestion_eligibility,
        "memory": memory,
        "throughput_ranking": ranking,
        "cross_card_scaling": scaling,
        "evidence_scope": evidence_scope,
        "decisions": decisions,
        "joined_observations": joined,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    integrity = report["integrity"]
    ingestion = report["ingestion_eligibility"]
    memory = report["memory"]
    ranking = report["throughput_ranking"]
    scaling = report["cross_card_scaling"]
    decisions = report["decisions"]
    scope = report["evidence_scope"]

    lines = [
        "# H800 fresh holdout 冻结验收报告（2026-08-02）",
        "",
        "> 本报告只回放实验开始前冻结的 physical-shares 显存模型与 v4b 吞吐排序器。",
        "> 生成报告时没有重训、没有回灌 canonical observations，也没有修改历史锚点库。",
        "",
        "## 一句话结论",
        "",
    ]
    if decisions["frozen_memory_and_ranking_gate_passes"]:
        lines.append(
            "这批 fresh holdout 上，冻结的显存安全门和 v4b 的 measured-subset Top-1 regret 门槛同时通过；"
            "但证据仍不足以宣称严格泛化，也不足以自动扩卡。"
        )
    elif memory["operational_safety_passes"] and ranking["passes"]:
        lines.append(
            "v4b 排序门通过；24 个配置全部安全运行且 0 OOM，但 physical-shares 的 formal upper-coverage "
            "门因 1 个明显低估点失败。因此当前组合不能整体通过冻结验收。"
        )
    else:
        lines.append(
            "这批 fresh holdout 上，冻结模型没有同时通过显存安全门与 v4b Top-1 regret 门槛；"
            "不得直接更新生产推荐策略。"
        )
    lines.extend(
        [
            "",
            "## 1. 证据完整性",
            "",
            f"- 队列 / 设计槽位 / 实测结果：{integrity['queue_jobs']} / "
            f"{integrity['design_candidate_slots']} / {integrity['collected_campaign_rows']}。",
            f"- 终态：`{json.dumps(integrity['outcomes'], ensure_ascii=False)}`。",
            f"- 完整性验收：{'通过' if integrity['passes'] else '失败'}；错误数 {len(integrity['errors'])}。",
            f"- fresh 场景切分：{'通过' if integrity['fresh_scenario_split_passes'] else '失败'}；"
            "这 4 个 dataset_id 不在冻结建模用的 canonical H800 observations 中。",
            "- 24 条结果均要求：完整执行指纹、rank summary 数等于 GPU 数、metrics 可用、calibration eligible。",
            f"- canonical fit/anchor 入库资格：{ingestion['canonical_export_partition_rows']}/"
            f"{ingestion['canonical_export_partition_required_rows']} 条带 job-bound `calibration_partition`；"
            "因此本批只能保存为 acceptance evidence，不能直接用于重训或锚点晋升。",
            "",
            "## 2. 显存模型（physical-shares）",
            "",
            f"- OOM：{memory['oom_rows']}；false-safe OOM：{memory['false_safe_oom']}。",
            f"- admission upper 对实测 reserved peak 的覆盖率：{_format_percent(memory['upper_coverage_fraction'])}。",
            f"- 场景等权 P05 覆盖率：{_format_percent(memory['scenario_equal_p05_coverage'])}；"
            f"门槛 {_format_percent(memory['minimum_coverage'])}。",
            f"- 显存安全门：{'通过' if memory['passes'] else '失败'}。",
            f"- 实际运行安全结果：{'通过' if memory['operational_safety_passes'] else '失败'}"
            "（该项只表示本批没有 OOM/越过安全线，不等于 upper 校准通过）。",
            f"- center MAPE / 中位 APE / P90 APE：{_format_percent(memory['center_mape'])} / "
            f"{_format_percent(memory['center_median_absolute_relative_error'])} / "
            f"{_format_percent(memory['center_p90_absolute_relative_error'])}。",
            f"- upper 平均比实测多留 {_format_number(memory['upper_mean_headroom_gib'], 2)} GiB；"
            f"实测离安全线的最小余量为 {_format_number(memory['minimum_observed_headroom_to_safe_limit_gib'], 2)} GiB。",
            "",
            "这里的主要验收指标是 upper 是否覆盖实测和是否出现 false-safe OOM；center 的绝对误差只用于判断保守程度。"
            "由于只运行了模型判为安全的候选，本批不能测出被过滤候选中的 false negative，也没有形成 OOM 边界压力证据。",
            "",
            "### 2.1 逐配置显存结果",
            "",
            "| 场景 | GPU | MBS | Z | GC | center GiB | upper GiB | 实测 GiB | upper覆盖 |",
            "|---|---:|---:|---:|:---:|---:|---:|---:|:---:|",
        ]
    )
    for row in sorted(
        memory["detail_rows"],
        key=lambda item: (item["scenario_id"], item["gpu_count"], item["predicted_upper_gib"] or 0.0),
    ):
        lines.append(
            f"| {row['scenario_id']} | {row['gpu_count']} | {row['mbs']} | {row['zero_stage']} | "
            f"{'on' if row['gc'] else 'off'} | {_format_number(row['predicted_center_gib'], 2)} | "
            f"{_format_number(row['predicted_upper_gib'], 2)} | {_format_number(row['observed_reserved_gib'], 2)} | "
            f"{'是' if row['upper_covers_observed'] else '否'} |"
        )

    lines.extend(
        [
            "",
            "## 3. 吞吐排序模型（v4b）",
            "",
            "排序组严格定义为“同一个业务场景 + 同一卡数”，共 "
            f"{ranking['endpoint_count']} 组；不同卡数之间单独进入扩卡分析。",
            "",
            f"- Top-1 完全命中：{ranking['exact_top1_count']}/{ranking['endpoint_count']} "
            f"({_format_percent(ranking['exact_top1_accuracy'])})。",
            f"- 选中配置达到实测最优 90%：{ranking['endpoints_within_10_percent_count']}/"
            f"{ranking['endpoint_count']} ({_format_percent(ranking['endpoints_within_10_percent_fraction'])})。",
            f"- mean / median / P90 / worst Top-1 regret："
            f"{_format_percent(ranking['mean_top1_regret'])} / {_format_percent(ranking['median_top1_regret'])} / "
            f"{_format_percent(ranking['p90_top1_regret'])} / {_format_percent(ranking['worst_top1_regret'])}。",
            f"- pooled pairwise accuracy：{ranking['pooled_pairwise_correct']}/"
            f"{ranking['pooled_pairwise_total']} = {_format_percent(ranking['pooled_pairwise_accuracy'])}。",
            f"- 场景等权 Spearman：{_format_number(ranking['scenario_equal_spearman_rank_correlation'])}。",
            f"- 正式 regret 门：{'通过' if ranking['passes'] else '失败'}；"
            f"要求每组 worst regret ≤ {_format_percent(ranking['maximum_top1_regret'])}。",
            "",
            "注意：这里的 oracle 是每组被前瞻性挑中并实测的 2–4 个 witness candidates 中的最好值，"
            "不是该组所有 memory-admitted candidates 的完整 oracle。",
            "",
            "### 3.1 分组结果",
            "",
            "| 场景 | GPU | 已测/安全候选 | Top-1命中 | 选中/最好 tok/s | regret | pairwise | Spearman |",
            "|---|---:|---:|:---:|---:|---:|---:|---:|",
        ]
    )
    for row in ranking["detail_groups"]:
        lines.append(
            f"| {row['scenario_id']} | {row['gpu_count']} | {row['measured_candidate_count']}/"
            f"{row['full_memory_admitted_candidate_count']} | {'是' if row['top1_exact'] else '否'} | "
            f"{_format_number(row['selected_observed_throughput'], 1)} / "
            f"{_format_number(row['best_observed_throughput'], 1)} | {_format_percent(row['top1_regret'])} | "
            f"{row['pairwise_correct']}/{row['pairwise_total']} | "
            f"{_format_number(row['spearman_rank_correlation'])} |"
        )

    absolute = ranking["absolute_proxy_diagnostics"]
    lines.extend(
        [
            "",
            "### 3.2 吞吐绝对值（只作诊断）",
            "",
            f"- proxy MAPE / 中位 APE / P90 APE：{_format_percent(absolute['mape'])} / "
            f"{_format_percent(absolute['median_absolute_percentage_error'])} / "
            f"{_format_percent(absolute['p90_absolute_percentage_error'])}。",
            f"- WAPE：{_format_percent(absolute['weighted_absolute_percentage_error'])}；"
            f"log-RMSE：{_format_number(absolute['log_rmse'])}。",
            f"- 实测/proxy 的中位比例：{_format_number(absolute['median_actual_to_proxy_ratio'])}。",
            "",
            "冻结 predictor 已明确标记 `absolute_scale_trusted=false`，所以这些绝对误差不参与 v4b 的上线门槛；"
            "产品目标仍是排序与低 regret。",
            "",
            "### 3.3 逐候选吞吐结果",
            "",
            "| 场景 | GPU | MBS | Z | GC | 预测proxy | 实测tok/s | 预测名次→实测名次 |",
            "|---|---:|---:|---:|:---:|---:|---:|---:|",
        ]
    )
    for group in ranking["detail_groups"]:
        for row in group["candidates"]:
            lines.append(
                f"| {group['scenario_id']} | {group['gpu_count']} | {row['mbs']} | {row['zero_stage']} | "
                f"{'on' if row['gc'] else 'off'} | {_format_number(row['predicted_throughput_proxy'], 1)} | "
                f"{_format_number(row['observed_effective_tokens_per_second'], 1)} | "
                f"{_format_number(row['predicted_rank_in_measured_subset'], 1)}→"
                f"{_format_number(row['observed_rank_in_measured_subset'], 1)} |"
            )

    lines.extend(
        [
            "",
            "## 4. 跨卡收益与最小卡数原则",
            "",
            f"冻结扩卡阈值是每次卡数翻倍后总吞吐至少达到 {scaling['minimum_ratio']:.1f}×。"
            "当前没有 conservative prediction lower bound，也没有重复实验形成 measured lower bound，"
            "所以以下比例只能作为 point diagnostic。",
            "",
            "| 场景 | 卡数 | 预测点比值 | 冻结端点赢家实测比值 | witness oracle 比值 | 每卡效率比 | 达到1.8×点阈值 |",
            "|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in scaling["groups"]:
        lines.append(
            f"| {row['scenario_id']} | {row['from_gpu']}→{row['to_gpu']} | "
            f"{_format_number(row['predicted_point_ratio'])} | "
            f"{_format_number(row['realized_ratio_for_frozen_endpoint_winners'])} | "
            f"{_format_number(row['measured_subset_oracle_ratio'])} | "
            f"{_format_number(row['throughput_per_gpu_ratio_for_frozen_endpoint_winners'])} | "
            f"{'是' if row['point_ratio_meets_threshold'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "因此当前产品行为不变：先找能安全训练的最小卡数，再在该卡数内用 v4b 排序；"
            "不得因为 point estimate 看起来较高就自动扩卡。",
            "",
            "## 5. 验收决策与下一步",
            "",
            f"- 冻结显存 + 排序门：{'通过' if decisions['frozen_memory_and_ranking_gate_passes'] else '未通过'}。",
            f"- 严格泛化声明：{'允许' if decisions['strict_generalization_claim_allowed'] else '暂不允许'}。",
            "- 这 24 条可以作为版本化 `acceptance evidence` 保存，但因 job payload 缺少冻结的 "
            "`calibration_partition`，不得进入 fit/anchor library。",
            "- 若要重训，必须生成新版本 challenger，并用本报告中的冻结结果做对照；不能再把同一批数据称为 holdout。",
            "- 显存下一批优先补：预测临近安全线的 14B FULL 2 卡边界点，以及被模型过滤但历史/单调锚点认为可跑的候选。",
            "- 吞吐下一批优先补：每个 endpoint 的完整 admitted top-k 邻域和重复测量，才能把 witness regret 升级为完整候选集 regret。",
            "- 扩卡下一批优先补：端点赢家至少 3 次独立重复，建立吞吐比值下置信界，再决定是否启用 1.8× 阈值。",
            "",
            "## 6. 本次证据边界",
            "",
            f"覆盖：{scope['fresh_dataset_scenarios']} 个 fresh 数据场景、{scope['ranking_endpoints']} 个排序端点、"
            f"{scope['measured_configurations']} 个配置；模型 {', '.join(scope['model_ids'])}；"
            f"训练方式 {', '.join(scope['train_types'])}；cutoff {scope['cutoff_lengths']}；GPU 数 {scope['gpu_counts']}。",
            "",
            "未覆盖：",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in scope["not_covered"])
    lines.extend(
        [
            "",
            "## 7. 可复现入口",
            "",
            f"- 机器可读报告：`{DEFAULT_OUTPUT}`",
            f"- 冻结预测：`{report['source_files']['frozen_predictions']['path']}`",
            f"- 队列：`{report['source_files']['queue']['path']}`",
            f"- 汇总实测：`{report['source_files']['collected_results']['path']}`",
            f"- 不可覆盖的 campaign 观测快照：`{DEFAULT_SNAPSHOT}`",
            "- 重跑评估：`/fine-tuning-launcher/.venv/bin/python "
            "offline_experiments/scripts/evaluate_h800_fresh_holdout_v2.py`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()

    report = evaluate(
        campaign_id=args.campaign_id,
        queue_path=args.queue,
        design_path=args.design,
        prediction_path=args.predictions,
        result_path=args.results,
        split_path=args.split,
    )
    snapshot = {
        "schema": "sft_h800_fresh_holdout_observation_snapshot/v2",
        "campaign_id": args.campaign_id,
        "immutable": True,
        "evaluation_mode": report["evaluation_mode"],
        "source_queue_sha256": report["source_files"]["queue"]["sha256"],
        "source_frozen_prediction_sha256": report["source_files"]["frozen_predictions"][
            "sha256"
        ],
        "rows": report["joined_observations"],
    }
    if args.snapshot.exists():
        if read_json(args.snapshot) != snapshot:
            raise FileExistsError(
                f"refusing to overwrite changed immutable observation snapshot: {args.snapshot}"
            )
    else:
        write_json(args.snapshot, snapshot)
    report["source_files"]["campaign_observation_snapshot"] = {
        "path": str(args.snapshot.resolve()),
        "sha256": sha256_file(args.snapshot),
        "rows": len(snapshot["rows"]),
        "immutable": True,
    }
    write_json(args.output, report)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "markdown": str(args.markdown),
                "snapshot": str(args.snapshot),
                "integrity_passes": report["integrity"]["passes"],
                "memory_passes": report["memory"]["passes"],
                "ranking_passes": report["throughput_ranking"]["passes"],
                "strict_generalization_claim_allowed": report["decisions"][
                    "strict_generalization_claim_allowed"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
