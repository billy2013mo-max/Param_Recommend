#!/usr/bin/env python3
"""Evaluate the frozen bounded-memory v2 H800 fresh holdout.

This evaluator is post-run only.  It joins the exact 21-job queue, the frozen
pre-run predictions and the latest terminal attempt of each job.  It does not
refit memory or throughput models and it refuses to overwrite a changed
observation, report or Markdown artifact.
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

from common import (
    ARTIFACT_DIR,
    MATRIX_DIR,
    RESULTS_DIR,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from evaluate_h800_final_unseen_holdout_v1 import (
    _collect_terminal_result,
    _config_mismatches,
    _duplicates,
    _embedded_prediction_mismatches,
    _finite,
    _format_number,
    _format_percent,
    _load_receipt_hashes,
    _percentile,
    _prediction_mismatches,
    _write_immutable_json,
    _write_immutable_text,
)
from prospective_acceptance import evaluate_memory_acceptance, evaluate_ranking_acceptance


SCHEMA = "sft_h800_bounded_memory_v2_fresh_holdout_acceptance/v1"
CAMPAIGN_ID = "h800_bounded_memory_v2_fresh_holdout_20260803_v1"
DEFAULT_QUEUE = MATRIX_DIR / "h800_bounded_memory_v2_fresh_holdout_jobs_v1.jsonl"
DEFAULT_FORMAL_QUEUE = MATRIX_DIR / "h800_bounded_memory_v2_fresh_holdout_formal_v1.jsonl"
DEFAULT_CANARY_QUEUE = MATRIX_DIR / "h800_bounded_memory_v2_fresh_holdout_canary_v1.jsonl"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_bounded_memory_v2_fresh_holdout_design_v1.json"
DEFAULT_PREDICTIONS = ARTIFACT_DIR / "h800_frozen_predictions_before_bounded_memory_v2_fresh_holdout_v1.json"
DEFAULT_QUEUE_MANIFEST = ARTIFACT_DIR / "h800_bounded_memory_v2_fresh_holdout_queue_manifest_v1.json"
DEFAULT_SPLIT = ARTIFACT_DIR / "bounded_memory_v2_fresh_holdout_v1" / "split_manifest.json"
DEFAULT_RESULTS_ROOT = RESULTS_DIR
DEFAULT_SNAPSHOT = ARTIFACT_DIR / "h800_bounded_memory_v2_fresh_holdout_observations_v1.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_bounded_memory_v2_fresh_holdout_acceptance_v1.json"
DEFAULT_MARKDOWN = ARTIFACT_DIR / "h800_bounded_memory_v2_fresh_holdout_acceptance_v1.md"
GIB = float(1024**3)


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return fmean(clean) if clean else None


def _source(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _memory_slice(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["outcome"] == "success"]
    apes = [float(row["center_ape"]) for row in successes if row["center_ape"] is not None]
    actual_safe = [row for row in rows if row["actual_safe_success"]]
    return {
        "rows": len(rows),
        "success_rows": len(successes),
        "oom_rows": sum(row["outcome"] == "oom" for row in rows),
        "center_mape": fmean(apes) if apes else None,
        "center_p90_ape": _percentile(apes, 90.0),
        "upper_coverage": _mean(
            float(row["upper_covers_observed"])
            for row in successes
            if row["upper_covers_observed"] is not None
        ),
        "actual_safe_rows": len(actual_safe),
        "admitted_actual_safe_rows": sum(row["predicted_admit"] for row in actual_safe),
        "false_rejected_safe_rows": sum(row["actual_safe_success"] and not row["predicted_admit"] for row in rows),
    }


def evaluate(
    *,
    campaign_id: str,
    queue_path: Path,
    formal_queue_path: Path,
    canary_queue_path: Path,
    design_path: Path,
    prediction_path: Path,
    queue_manifest_path: Path,
    split_path: Path,
    results_root: Path,
) -> dict[str, Any]:
    queue = read_jsonl(queue_path)
    formal_queue = read_jsonl(formal_queue_path)
    canary_queue = read_jsonl(canary_queue_path)
    design = read_json(design_path)
    frozen = read_json(prediction_path)
    queue_manifest = read_json(queue_manifest_path)
    split = read_json(split_path)
    predictions = frozen.get("predictions") or []
    slots = design.get("candidate_slots") or []
    queue_by_job = {str(row["job_id"]): row for row in queue}
    prediction_by_request = {str(row["request_id"]): row for row in predictions}
    slot_by_id = {str(row["candidate_slot_id"]): row for row in slots}
    errors: list[dict[str, Any]] = []

    duplicates = {
        "queue_job_ids": _duplicates(str(row["job_id"]) for row in queue),
        "queue_slot_ids": _duplicates(str(row["candidate_slot_id"]) for row in queue),
        "queue_prediction_ids": _duplicates(str(row["predictor_request_id"]) for row in queue),
        "prediction_request_ids": _duplicates(str(row["request_id"]) for row in predictions),
        "design_slot_ids": _duplicates(str(row["candidate_slot_id"]) for row in slots),
    }
    for name, values in duplicates.items():
        if values:
            errors.append({"check": name, "duplicates": values})

    queue_ids = set(queue_by_job)
    formal_ids = {str(row["job_id"]) for row in formal_queue}
    canary_ids = {str(row["job_id"]) for row in canary_queue}
    set_checks = {
        "staged_queue_partition": queue_ids == formal_ids | canary_ids and not formal_ids & canary_ids,
        "queue_prediction_set": set(prediction_by_request)
        == {str(row["predictor_request_id"]) for row in queue},
        "queue_design_slot_set": set(slot_by_id) == {str(row["candidate_slot_id"]) for row in queue},
        "campaign_binding": design.get("campaign_id") == campaign_id
        and all(row.get("campaign_id") == campaign_id for row in queue),
        "queue_manifest_job_count": int(queue_manifest.get("job_count", -1)) == len(queue),
    }
    for name, passes in set_checks.items():
        if not passes:
            errors.append({"check": name})

    bound_prediction = design.get("frozen_prediction_binding") or {}
    prediction_sha = sha256_file(prediction_path)
    if bound_prediction.get("sha256") != prediction_sha:
        errors.append(
            {
                "check": "frozen_prediction_sha256",
                "expected": bound_prediction.get("sha256"),
                "actual": prediction_sha,
            }
        )
    manifest_queue = queue_manifest.get("queue") or {}
    if manifest_queue.get("sha256") != sha256_file(queue_path):
        errors.append({"check": "queue_manifest_queue_sha256"})
    for label, path in (("formal", formal_queue_path), ("canary", canary_queue_path)):
        expected = ((queue_manifest.get("staged_queues") or {}).get(label) or {}).get("sha256")
        if expected != sha256_file(path):
            errors.append({"check": f"queue_manifest_{label}_sha256"})

    selected_profiles = {str(row["profile_id"]) for row in split.get("splits") or []}
    queue_profiles = {str(row["profile_id"]) for row in queue}
    fit_overlap = split.get("fit_overlap") or {}
    freshness_passes = bool(
        queue_profiles == selected_profiles
        and fit_overlap.get("selected_sources_used_for_v2_fit") is False
        and fit_overlap.get("selected_sources_used_for_prior_holdout") is False
    )
    if not freshness_passes:
        errors.append({"check": "freshness_binding"})

    result_rows: dict[str, dict[str, Any]] = {}
    result_manifests: list[dict[str, Any]] = []
    for job_id, job in sorted(queue_by_job.items()):
        result, manifest, collection_errors = _collect_terminal_result(results_root, job)
        errors.extend(collection_errors)
        if result is not None:
            result_rows[job_id] = result
        if manifest is not None:
            result_manifests.append(manifest)

    receipt_hashes = _load_receipt_hashes()
    approval_hashes = sorted(
        {str(row.get("approval_design_sha256")) for row in result_rows.values() if row.get("approval_design_sha256")}
    )
    unknown_approvals = sorted(set(approval_hashes) - receipt_hashes)
    if unknown_approvals:
        errors.append({"check": "approval_receipts", "unknown": unknown_approvals})

    current_hashes: dict[str, str] = {}
    joined: list[dict[str, Any]] = []
    for job_id, job in sorted(queue_by_job.items()):
        result = result_rows.get(job_id)
        prediction = prediction_by_request.get(str(job["predictor_request_id"]))
        slot = slot_by_id.get(str(job["candidate_slot_id"]))
        if result is None or prediction is None or slot is None:
            errors.append(
                {
                    "check": "join",
                    "job_id": job_id,
                    "result": result is not None,
                    "prediction": prediction is not None,
                    "slot": slot is not None,
                }
            )
            continue
        mismatches = _config_mismatches(job, result.get("job") or {})
        if mismatches:
            errors.append({"check": "queue_result_configuration", "job_id": job_id, "fields": mismatches})
        mismatches = _prediction_mismatches(job, prediction)
        if mismatches:
            errors.append({"check": "queue_prediction_configuration", "job_id": job_id, "fields": mismatches})
        mismatches = _embedded_prediction_mismatches(job.get("frozen_prediction") or {}, prediction)
        if mismatches:
            errors.append({"check": "embedded_prediction", "job_id": job_id, "fields": mismatches})
        if sha256_json(slot.get("frozen_prediction") or {}) != sha256_json(job.get("frozen_prediction") or {}):
            errors.append({"check": "slot_queue_prediction", "job_id": job_id})

        for path_field, sha_field in (("data_path", "data_sha256"), ("dataset_profile_path", "dataset_profile_sha256")):
            path = Path(str(job[path_field]))
            key = str(path)
            if key not in current_hashes:
                current_hashes[key] = sha256_file(path) if path.is_file() else "missing"
            if current_hashes[key] != job.get(sha_field):
                errors.append({"check": f"current_{sha_field}", "job_id": job_id})

        memory = prediction.get("memory") or {}
        throughput = prediction.get("throughput") or {}
        outcome = str(result.get("classification"))
        observed_reserved = _finite(result.get("max_reserved_bytes"))
        center = _finite(memory.get("reserved_center_bytes"))
        upper = _finite(memory.get("admission_upper_reserved_bytes"))
        safe_limit = _finite(memory.get("safe_limit_bytes"))
        success = outcome == "success"
        actual_safe = bool(
            success
            and observed_reserved is not None
            and safe_limit is not None
            and observed_reserved <= safe_limit
        )
        joined.append(
            {
                "job_id": job_id,
                "candidate_id": str(job["predictor_request_id"]),
                "candidate_slot_id": str(job["candidate_slot_id"]),
                "scenario_id": str(job["scenario_id"]),
                "profile_id": str(job["profile_id"]),
                "candidate_role": str(job["candidate_role"]),
                "software_canary": bool(job.get("software_canary")),
                "model_id": str(job["model_id"]),
                "train_type": str(job["train_type"]),
                "gpu_count": int(job["gpu_count"]),
                "mbs": int(job["mbs"]),
                "gradient_accumulation_steps": int(job["gradient_accumulation_steps"]),
                "zero_stage": int(job["zero_stage"]),
                "gc": bool(job["gc"]),
                "packing": bool(job["packing"]),
                "offload": bool(job["offload"]),
                "outcome": outcome,
                "predicted_admit": memory.get("admitted") is True,
                "actual_safe_success": actual_safe,
                "false_rejected_safe": bool(actual_safe and memory.get("admitted") is not True),
                "predicted_center_bytes": center,
                "predicted_upper_bytes": upper,
                "safe_limit_bytes": safe_limit,
                "observed_reserved_bytes": observed_reserved,
                "center_ape": (
                    abs(center - observed_reserved) / observed_reserved
                    if success and center is not None and observed_reserved and observed_reserved > 0.0
                    else None
                ),
                "upper_covers_observed": (
                    bool(upper is not None and observed_reserved is not None and upper >= observed_reserved)
                    if success
                    else None
                ),
                "predicted_throughput_proxy": _finite(throughput.get("throughput_proxy_tokens_per_second")),
                "observed_effective_tokens_per_second": _finite(result.get("effective_tokens_per_second")),
                "observed_computed_tokens_per_second": _finite(result.get("computed_tokens_per_second")),
                "observed_samples_per_second": _finite(result.get("samples_per_second")),
                "execution_attempt_id": result.get("execution_attempt_id"),
                "approval_design_sha256": result.get("approval_design_sha256"),
                "execution_fingerprint_quality": result.get("execution_fingerprint_quality"),
                "calibration_eligible": result.get("calibration_eligible") is True,
                "rank_summaries": int(result.get("rank_summaries") or 0),
                "metrics_available": result.get("metrics_available") is True,
                "gpu_mask": result.get("gpu_mask"),
                "finished_unix": result.get("finished_unix"),
            }
        )

    execution_checks = {
        "all_21_results_joined": len(joined) == len(queue) == 21,
        "all_terminal_success_or_oom": all(row["outcome"] in {"success", "oom"} for row in joined),
        "all_success_metrics_available": all(row["outcome"] != "success" or row["metrics_available"] for row in joined),
        "all_execution_fingerprints_complete": all(row["execution_fingerprint_quality"] == "complete" for row in joined),
        "all_calibration_eligible": all(row["calibration_eligible"] for row in joined),
        "success_rank_summaries_complete": all(
            row["outcome"] != "success" or row["rank_summaries"] == row["gpu_count"] for row in joined
        ),
    }
    for name, passes in execution_checks.items():
        if not passes:
            errors.append({"check": name})
    outcomes = Counter(row["outcome"] for row in joined)
    integrity = {
        "passes": not errors,
        "queue_jobs": len(queue),
        "formal_jobs": len(formal_queue),
        "canary_jobs": len(canary_queue),
        "design_slots": len(slots),
        "frozen_predictions": len(predictions),
        "joined_rows": len(joined),
        "outcomes": dict(sorted(outcomes.items())),
        "freshness_passes": freshness_passes,
        "approval_design_sha256_values": approval_hashes,
        "approval_hashes_have_receipts": not unknown_approvals,
        "set_checks": set_checks,
        "execution_checks": execution_checks,
        "duplicate_checks": duplicates,
        "errors": errors,
    }

    contract = design.get("acceptance_contract") or {}
    memory_standard = evaluate_memory_acceptance(
        [
            {
                "scenario_id": row["scenario_id"],
                "outcome": row["outcome"],
                "predicted_admit": row["predicted_admit"],
                "actual_safe_success": row["actual_safe_success"],
                "upper_covers_observed": row["upper_covers_observed"],
            }
            for row in joined
        ],
        minimum_coverage=float(contract["scenario_equal_p05_upper_coverage"]),
        maximum_false_safe_oom=int(contract["false_safe_oom"]),
    )
    successes = [row for row in joined if row["outcome"] == "success"]
    apes = [float(row["center_ape"]) for row in successes if row["center_ape"] is not None]
    scenario_apes: dict[str, list[float]] = defaultdict(list)
    for row in successes:
        if row["center_ape"] is not None:
            scenario_apes[row["scenario_id"]].append(float(row["center_ape"]))
    scenario_mean_apes = {key: fmean(values) for key, values in sorted(scenario_apes.items())}
    center_scenario_equal_mape = fmean(scenario_mean_apes.values()) if scenario_mean_apes else None
    center_row_p90 = _percentile(apes, 90.0)
    candidate_rows = [row for row in joined if row["candidate_role"] != "tail_forced_safety"]
    actual_safe_candidates = [row for row in candidate_rows if row["actual_safe_success"]]
    admitted_actual_safe = sum(row["predicted_admit"] for row in actual_safe_candidates)
    admission_recall = admitted_actual_safe / len(actual_safe_candidates) if actual_safe_candidates else None
    tail_rows = [row for row in joined if row["candidate_role"] == "tail_forced_safety"]
    tail_successes = [row for row in tail_rows if row["outcome"] == "success"]
    tail_upper_coverage = _mean(float(row["upper_covers_observed"]) for row in tail_successes)
    cross_rows = [row for row in joined if row["candidate_role"] == "cross_scale_diagnostic"]
    cross_false_safe = sum(row["outcome"] == "oom" and row["predicted_admit"] for row in cross_rows)
    precision_passes = bool(
        center_scenario_equal_mape is not None
        and center_scenario_equal_mape <= float(contract["reserved_center_scenario_equal_mape_secondary"])
        and center_row_p90 is not None
        and center_row_p90 <= float(contract["reserved_center_row_p90_ape_secondary"])
    )
    admission_passes = bool(
        admission_recall is not None
        and admission_recall >= float(contract["actual_safe_candidate_admission_recall"])
    )
    tail_passes = bool(
        tail_upper_coverage is not None
        and tail_upper_coverage >= float(contract["tail_forced_upper_coverage"])
        and not any(row["outcome"] == "oom" and row["predicted_admit"] for row in tail_rows)
    )
    cross_passes = cross_false_safe <= int(contract["cross_scale_false_safe_oom"])
    safety_passes = bool(integrity["passes"] and memory_standard["passes"] and tail_passes and cross_passes)

    by_slice: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_slice[(row["profile_id"], row["model_id"], row["candidate_role"])].append(row)
    memory = {
        **memory_standard,
        "center_row_mape": fmean(apes) if apes else None,
        "center_row_median_ape": median(apes) if apes else None,
        "center_row_p90_ape": center_row_p90,
        "center_scenario_equal_mape": center_scenario_equal_mape,
        "center_scenario_mean_apes": scenario_mean_apes,
        "center_scenario_equal_mape_limit_secondary": float(contract["reserved_center_scenario_equal_mape_secondary"]),
        "center_row_p90_ape_limit_secondary": float(contract["reserved_center_row_p90_ape_secondary"]),
        "precision_secondary_passes": precision_passes,
        "candidate_rows": len(candidate_rows),
        "actual_safe_candidate_rows": len(actual_safe_candidates),
        "admitted_actual_safe_candidate_rows": admitted_actual_safe,
        "actual_safe_candidate_admission_recall": admission_recall,
        "actual_safe_candidate_admission_recall_limit": float(contract["actual_safe_candidate_admission_recall"]),
        "admission_recall_passes": admission_passes,
        "false_rejected_candidates": [
            {
                "job_id": row["job_id"],
                "scenario_id": row["scenario_id"],
                "observed_reserved_gib": row["observed_reserved_bytes"] / GIB,
                "predicted_center_gib": row["predicted_center_bytes"] / GIB,
                "predicted_upper_gib": row["predicted_upper_bytes"] / GIB,
                "safe_limit_gib": row["safe_limit_bytes"] / GIB,
            }
            for row in actual_safe_candidates
            if not row["predicted_admit"]
        ],
        "tail_forced_rows": len(tail_rows),
        "tail_forced_success_rows": len(tail_successes),
        "tail_forced_upper_coverage": tail_upper_coverage,
        "tail_forced_passes": tail_passes,
        "cross_scale_rows": len(cross_rows),
        "cross_scale_false_safe_oom": cross_false_safe,
        "cross_scale_passes": cross_passes,
        "safety_passes": safety_passes,
        "replacement_gate_passes": bool(safety_passes and precision_passes and admission_passes),
        "slices": [
            {
                "profile_id": key[0],
                "model_id": key[1],
                "candidate_role": key[2],
                **_memory_slice(rows),
            }
            for key, rows in sorted(by_slice.items())
        ],
        "detail_rows": [
            {
                **{key: row[key] for key in (
                    "job_id", "scenario_id", "profile_id", "candidate_role", "model_id", "train_type",
                    "gpu_count", "mbs", "zero_stage", "gc", "outcome", "predicted_admit",
                    "actual_safe_success", "false_rejected_safe", "center_ape", "upper_covers_observed",
                )},
                "predicted_center_gib": row["predicted_center_bytes"] / GIB,
                "predicted_upper_gib": row["predicted_upper_bytes"] / GIB,
                "observed_reserved_gib": row["observed_reserved_bytes"] / GIB,
                "safe_limit_gib": row["safe_limit_bytes"] / GIB,
            }
            for row in joined
        ],
    }

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        if row["candidate_role"] == "base_selector":
            grouped[(row["scenario_id"], row["gpu_count"])].append(row)
    standard_groups: list[dict[str, Any]] = []
    ranking_details: list[dict[str, Any]] = []
    for (scenario_id, gpu_count), rows in sorted(grouped.items()):
        endpoint = f"{scenario_id}__g{gpu_count}"
        candidates = [
            {
                "candidate_id": row["candidate_id"],
                "outcome": row["outcome"],
                "predicted_admit": row["predicted_admit"],
                "actual_safe_success": row["actual_safe_success"],
                "predicted_throughput": row["predicted_throughput_proxy"],
                "observed_throughput": row["observed_effective_tokens_per_second"],
            }
            for row in rows
        ]
        standard_groups.append({"scenario_id": endpoint, "candidates": candidates})
        safe = [row for row in rows if row["actual_safe_success"]]
        scored = [
            row for row in rows
            if row["predicted_admit"]
            and row["predicted_throughput_proxy"] is not None
            and row["observed_effective_tokens_per_second"] is not None
        ]
        predicted = max(scored, key=lambda row: row["predicted_throughput_proxy"], default=None)
        observed = max(safe, key=lambda row: row["observed_effective_tokens_per_second"], default=None)
        regret = (
            1.0 - predicted["observed_effective_tokens_per_second"] / observed["observed_effective_tokens_per_second"]
            if predicted is not None and observed is not None
            else None
        )
        eligible = len(safe) >= 2 and predicted is not None and observed is not None and regret is not None
        ranking_details.append(
            {
                "endpoint_id": endpoint,
                "scenario_id": scenario_id,
                "gpu_count": gpu_count,
                "candidate_count": len(rows),
                "actual_safe_success_count": len(safe),
                "predicted_admitted_count": sum(row["predicted_admit"] for row in rows),
                "status": "evaluated" if eligible else "insufficient_evidence",
                "predicted_winner": predicted["candidate_id"] if predicted else None,
                "observed_best": observed["candidate_id"] if observed else None,
                "top1_exact": predicted["candidate_id"] == observed["candidate_id"] if eligible else None,
                "selected_observed_tokens_per_second": predicted["observed_effective_tokens_per_second"] if predicted else None,
                "best_observed_tokens_per_second": observed["observed_effective_tokens_per_second"] if observed else None,
                "top1_regret": regret,
                "top1_within_10_percent": bool(regret is not None and regret <= 0.10 + 1e-12),
                "candidates": [
                    {
                        "job_id": row["job_id"],
                        "candidate_id": row["candidate_id"],
                        "mbs": row["mbs"],
                        "predicted_admit": row["predicted_admit"],
                        "actual_safe_success": row["actual_safe_success"],
                        "predicted_throughput_proxy": row["predicted_throughput_proxy"],
                        "observed_effective_tokens_per_second": row["observed_effective_tokens_per_second"],
                    }
                    for row in sorted(rows, key=lambda item: item["mbs"])
                ],
            }
        )
    ranking_standard = evaluate_ranking_acceptance(standard_groups, maximum_top1_regret=0.10)
    eligible_rankings = [row for row in ranking_details if row["status"] == "evaluated"]
    proxy_rows = [
        row for row in candidate_rows
        if row["outcome"] == "success"
        and row["predicted_throughput_proxy"] is not None
        and row["observed_effective_tokens_per_second"]
    ]
    proxy_apes = [
        abs(row["predicted_throughput_proxy"] - row["observed_effective_tokens_per_second"])
        / row["observed_effective_tokens_per_second"]
        for row in proxy_rows
    ]
    ranking = {
        **ranking_standard,
        "endpoint_count": len(ranking_details),
        "eligible_endpoint_count": len(eligible_rankings),
        "exact_top1_count": sum(row["top1_exact"] is True for row in eligible_rankings),
        "hit_at_90_count": sum(row["top1_within_10_percent"] for row in eligible_rankings),
        "hit_at_90_fraction": _mean(float(row["top1_within_10_percent"]) for row in eligible_rankings),
        "absolute_proxy_diagnostic": {
            "gating_metric": False,
            "absolute_scale_trusted": frozen.get("absolute_throughput_scale_trusted") is True,
            "rows": len(proxy_rows),
            "mape": fmean(proxy_apes) if proxy_apes else None,
            "p90_ape": _percentile(proxy_apes, 90.0),
        },
        "detail_groups": ranking_details,
    }

    blockers: list[str] = []
    if not integrity["passes"]:
        blockers.append("evidence_integrity_failed")
    if not safety_passes:
        blockers.append("memory_safety_gate_failed")
    if not precision_passes:
        blockers.append("memory_precision_secondary_failed")
    if not admission_passes:
        blockers.append("actual_safe_candidate_admission_recall_failed")
    if not ranking_standard["passes"]:
        blockers.append("v4b_top1_regret_failed")
    decisions = {
        "bounded_memory_v2_replacement_allowed": memory["replacement_gate_passes"],
        "v4b_strict_generalization_passes": ranking_standard["passes"],
        "physical_shares_plus_v4b_publication_allowed": bool(
            memory["replacement_gate_passes"] and ranking_standard["passes"]
        ),
        "automatic_execution_allowed": False,
        "current_production_artifacts_must_remain_unchanged": True,
        "holdout_is_consumed": True,
        "may_reuse_for_next_fresh_acceptance": False,
        "blockers": blockers,
        "next_action": (
            "Keep the current production artifacts unchanged. Diagnose the Qwen3-14B two-GPU broad-profile "
            "false rejection and the Qwen3-8B LoRA concentrated-profile v4b inversion. Any refit must mark "
            "these sources as seen and use a new source-disjoint holdout. Packing and VL canary/calibration "
            "campaigns may proceed as separate shadow tracks."
        ),
    }

    completed = [float(row["finished_unix"]) for row in joined if row.get("finished_unix")]
    completed_at = datetime.fromtimestamp(max(completed), timezone.utc).isoformat() if completed else None
    report = {
        "schema": SCHEMA,
        "generated_at_utc": completed_at,
        "timestamp_semantics": "deterministic maximum terminal finished_unix",
        "campaign_id": campaign_id,
        "evaluation_mode": "frozen_predictions_no_refit",
        "source_files": {
            "queue": _source(queue_path),
            "formal_queue": _source(formal_queue_path),
            "canary_queue": _source(canary_queue_path),
            "design": _source(design_path),
            "frozen_predictions": _source(prediction_path),
            "queue_manifest": _source(queue_manifest_path),
            "split_manifest": _source(split_path),
            "terminal_result_manifests_sha256": sha256_json(result_manifests),
            "terminal_result_manifests": result_manifests,
        },
        "integrity": integrity,
        "acceptance_contract": contract,
        "memory": memory,
        "throughput_ranking": ranking,
        "decisions": decisions,
        "evidence_scope": {
            "profiles": sorted(queue_profiles),
            "model_train_slices": sorted({f"{row['model_id']}:{row['train_type']}" for row in joined}),
            "gpu_counts": sorted({row["gpu_count"] for row in joined}),
            "candidate_roles": sorted({row["candidate_role"] for row in joined}),
            "packing_values": sorted({row["packing"] for row in joined}),
            "offload_values": sorted({row["offload"] for row in joined}),
            "exact_repeats_per_configuration": 1,
        },
        "joined_observations": joined,
    }
    report["report_sha256"] = sha256_json(report)
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    integrity = report["integrity"]
    memory = report["memory"]
    ranking = report["throughput_ranking"]
    decisions = report["decisions"]
    lines = [
        "# H800 bounded-memory v2 新鲜留出集验收（2026-08-03）",
        "",
        "> 21 个任务及预测在 GPU 启动前冻结。本报告没有重训、回灌 holdout、覆盖历史锚点或修改验收阈值。",
        "",
        "## 结论",
        "",
        "证据完整性通过，显存没有 false-safe OOM，upper coverage 为 100%，center 的两个次要精度指标也通过；",
        "但一个真实安全的 14B Full 两卡配置被误拒，候选 admission recall 只有 "
        f"{_format_percent(memory['actual_safe_candidate_admission_recall'])}，低于冻结门槛 95%。",
        "v4b 在 4 个可严格排序端点中命中 3 个，Hit@90 为 "
        f"{_format_percent(ranking['hit_at_90_fraction'])}，最差 regret 为 "
        f"{_format_percent(ranking['worst_top1_regret'])}。因此 bounded-memory v2 与 v4b 均不能按本批结果发布。",
        "",
        "## 1. 证据完整性",
        "",
        f"- 完整队列 / formal / canary：{integrity['queue_jobs']} / {integrity['formal_jobs']} / {integrity['canary_jobs']}。",
        f"- 终态：`{json.dumps(integrity['outcomes'], ensure_ascii=False)}`。",
        f"- 队列、预测、数据、profile、attempt、指纹、rank summary、approval receipt："
        f"{'全部通过' if integrity['passes'] else '存在失败'}。",
        "- 三个 S3 源与 v2 fit、历史 holdout 均无重叠；本批结束后已被消费，不能再宣称 fresh holdout。",
        "",
        "## 2. 显存验收",
        "",
        "| 指标 | 门槛 | 结果 | 判定 |",
        "|---|---:|---:|:---:|",
        f"| false-safe OOM | = 0 | {memory['false_safe_oom']} | {'通过' if memory['false_safe_oom'] == 0 else '失败'} |",
        f"| 场景等权 P05 upper coverage | ≥ 95% | {_format_percent(memory['scenario_equal_p05_coverage'])} | {'通过' if memory['passes'] else '失败'} |",
        f"| tail-forced upper coverage | = 100% | {_format_percent(memory['tail_forced_upper_coverage'])} | {'通过' if memory['tail_forced_passes'] else '失败'} |",
        f"| cross-scale false-safe OOM | = 0 | {memory['cross_scale_false_safe_oom']} | {'通过' if memory['cross_scale_passes'] else '失败'} |",
        f"| 实际安全候选 admission recall | ≥ 95% | {_format_percent(memory['actual_safe_candidate_admission_recall'])} | {'通过' if memory['admission_recall_passes'] else '失败'} |",
        f"| center 场景等权 MAPE（次要） | ≤ 20% | {_format_percent(memory['center_scenario_equal_mape'])} | {'通过' if memory['center_scenario_equal_mape'] <= memory['center_scenario_equal_mape_limit_secondary'] else '失败'} |",
        f"| center 行级 P90 APE（次要） | ≤ 35% | {_format_percent(memory['center_row_p90_ape'])} | {'通过' if memory['center_row_p90_ape'] <= memory['center_row_p90_ape_limit_secondary'] else '失败'} |",
        "",
        "### 2.1 误拒案例",
        "",
    ]
    for row in memory["false_rejected_candidates"]:
        lines.append(
            f"- `{row['job_id']}`：14B Full 两卡实际 {_format_number(row['observed_reserved_gib'])} GiB，"
            f"安全线 {_format_number(row['safe_limit_gib'])} GiB；模型中心/上界为 "
            f"{_format_number(row['predicted_center_gib'])}/{_format_number(row['predicted_upper_gib'])} GiB，因而误拒。"
        )
    lines.extend(
        [
            "",
            "### 2.2 逐任务显存",
            "",
            "| role | profile | model | GPU | MBS | outcome | center/upper/actual GiB | APE | admitted | actual safe |",
            "|---|---|---|---:|---:|---|---:|---:|:---:|:---:|",
        ]
    )
    for row in sorted(memory["detail_rows"], key=lambda item: (item["candidate_role"], item["profile_id"], item["model_id"], item["gpu_count"], item["mbs"])):
        lines.append(
            f"| {row['candidate_role']} | {row['profile_id']} | {row['model_id']} {row['train_type']} | "
            f"{row['gpu_count']} | {row['mbs']} | {row['outcome']} | "
            f"{_format_number(row['predicted_center_gib'])}/{_format_number(row['predicted_upper_gib'])}/"
            f"{_format_number(row['observed_reserved_gib'])} | {_format_percent(row['center_ape'])} | "
            f"{'是' if row['predicted_admit'] else '否'} | {'是' if row['actual_safe_success'] else '否'} |"
        )

    lines.extend(
        [
            "",
            "## 3. 显存过滤后的 v4b 排序",
            "",
            f"- 可严格比较端点：{ranking['eligible_endpoint_count']}。",
            f"- Top-1：{ranking['exact_top1_count']}/{ranking['eligible_endpoint_count']}。",
            f"- Hit@90：{ranking['hit_at_90_count']}/{ranking['eligible_endpoint_count']} "
            f"({_format_percent(ranking['hit_at_90_fraction'])})。",
            f"- mean / worst regret：{_format_percent(ranking['mean_top1_regret'])} / "
            f"{_format_percent(ranking['worst_top1_regret'])}。",
            "- 吞吐代理的绝对值不作为门槛；报告中的实测吞吐为所有 rank 的聚合有效 token/s。",
            "",
            "| 场景 | GPU | 安全候选 | Top-1 | selected/best token/s | regret |",
            "|---|---:|---:|:---:|---:|---:|",
        ]
    )
    for row in ranking["detail_groups"]:
        if row["status"] != "evaluated":
            continue
        lines.append(
            f"| {row['scenario_id']} | {row['gpu_count']} | {row['actual_safe_success_count']} | "
            f"{'命中' if row['top1_exact'] else '未命中'} | "
            f"{_format_number(row['selected_observed_tokens_per_second'], 1)}/"
            f"{_format_number(row['best_observed_tokens_per_second'], 1)} | "
            f"{_format_percent(row['top1_regret'])} |"
        )

    lines.extend(
        [
            "",
            "## 4. 决策与下一步",
            "",
            f"- bounded-memory v2 替换允许：`{str(decisions['bounded_memory_v2_replacement_allowed']).lower()}`。",
            f"- v4b 严格泛化通过：`{str(decisions['v4b_strict_generalization_passes']).lower()}`。",
            f"- physical-shares + v4b 发布允许：`{str(decisions['physical_shares_plus_v4b_publication_allowed']).lower()}`。",
            "- 当前生产 artifact、历史锚点和推荐策略保持不变。",
            "- 先诊断 14B 两卡 broad-profile 误拒和 8B LoRA concentrated-profile 排序反转；若用于拟合必须标为 seen，并另取新 S3 源做验收。",
            "- Packing 与 VL 作为独立 shadow 轨道，可继续进入 canary 与 calibration，不据此扩大当前产品支持域。",
            "",
            "## 5. 产物",
            "",
            f"- 机器可读报告：`{DEFAULT_OUTPUT}`",
            f"- 不可覆盖观察：`{DEFAULT_SNAPSHOT}`",
            f"- 冻结预测：`{report['source_files']['frozen_predictions']['path']}`",
            "- 重跑：`/fine-tuning-launcher/.venv/bin/python "
            "offline_experiments/scripts/evaluate_h800_bounded_memory_v2_fresh_holdout_v1.py`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=CAMPAIGN_ID)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--formal-queue", type=Path, default=DEFAULT_FORMAL_QUEUE)
    parser.add_argument("--canary-queue", type=Path, default=DEFAULT_CANARY_QUEUE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--queue-manifest", type=Path, default=DEFAULT_QUEUE_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    report = evaluate(
        campaign_id=args.campaign_id,
        queue_path=args.queue,
        formal_queue_path=args.formal_queue,
        canary_queue_path=args.canary_queue,
        design_path=args.design,
        prediction_path=args.predictions,
        queue_manifest_path=args.queue_manifest,
        split_path=args.split,
        results_root=args.results_root,
    )
    snapshot = {
        "schema": "sft_h800_bounded_memory_v2_fresh_holdout_observations/v1",
        "campaign_id": args.campaign_id,
        "immutable": True,
        "evaluation_mode": report["evaluation_mode"],
        "source_queue_sha256": report["source_files"]["queue"]["sha256"],
        "source_frozen_prediction_sha256": report["source_files"]["frozen_predictions"]["sha256"],
        "source_terminal_result_manifests_sha256": report["source_files"]["terminal_result_manifests_sha256"],
        "rows": report["joined_observations"],
    }
    _write_immutable_json(args.snapshot, snapshot)
    report["source_files"]["campaign_observation_snapshot"] = {
        "path": str(args.snapshot.resolve()),
        "sha256": sha256_file(args.snapshot),
        "rows": len(snapshot["rows"]),
        "immutable": True,
    }
    report["report_sha256"] = sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    _write_immutable_json(args.output, report)
    _write_immutable_text(args.markdown, render_markdown(report))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "markdown": str(args.markdown),
                "snapshot": str(args.snapshot),
                "integrity_passes": report["integrity"]["passes"],
                "memory_replacement_allowed": report["decisions"]["bounded_memory_v2_replacement_allowed"],
                "v4b_generalization_passes": report["decisions"]["v4b_strict_generalization_passes"],
                "admission_recall": report["memory"]["actual_safe_candidate_admission_recall"],
                "ranking_hit_at_90": report["throughput_ranking"]["hit_at_90_fraction"],
                "ranking_worst_regret": report["throughput_ranking"]["worst_top1_regret"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
