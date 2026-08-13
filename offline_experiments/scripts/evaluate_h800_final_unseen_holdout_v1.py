#!/usr/bin/env python3
"""Evaluate the frozen H800 profile-aware memory final unseen holdout.

The evaluator is deliberately post-run and read-only with respect to models,
anchors, queues and approvals.  It joins the pre-run design, frozen predictions
and exact queue to the terminal attempt of each job.  It never refits the
challenger.  The joined observation snapshot is immutable so a later result
collection cannot silently change this acceptance decision.
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
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from prospective_acceptance import evaluate_memory_acceptance, evaluate_ranking_acceptance


SCHEMA = "sft_h800_final_unseen_holdout_acceptance/v1"
DEFAULT_CAMPAIGN_ID = "h800_profile_aware_memory_final_holdout_20260802_v1"
DEFAULT_QUEUE = MATRIX_DIR / "h800_final_unseen_holdout_jobs_v1.jsonl"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_final_unseen_holdout_design_v1.json"
DEFAULT_PREDICTIONS = ARTIFACT_DIR / "h800_frozen_predictions_before_final_unseen_holdout_v1.json"
DEFAULT_SPLIT = ARTIFACT_DIR / "final_unseen_holdout_v1" / "split_manifest.json"
DEFAULT_RESULTS_ROOT = RESULTS_DIR
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_final_unseen_holdout_acceptance_v1.json"
DEFAULT_MARKDOWN = ARTIFACT_DIR / "h800_final_unseen_holdout_acceptance_v1.md"
DEFAULT_SNAPSHOT = ARTIFACT_DIR / "h800_final_unseen_holdout_observations_v1.json"
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


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-6)
    return left == right


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _format_number(value: Any, digits: int = 2) -> str:
    finite = _finite(value)
    return "—" if finite is None else f"{finite:.{digits}f}"


def _format_percent(value: Any, digits: int = 1) -> str:
    finite = _finite(value)
    return "—" if finite is None else f"{100.0 * finite:.{digits}f}%"


def _load_receipt_hashes() -> set[str]:
    hashes: set[str] = set()
    if not APPROVAL_RECEIPTS.is_dir():
        return hashes
    for path in sorted(APPROVAL_RECEIPTS.glob("*.json")):
        payload = read_json(path)
        if isinstance(payload, Mapping) and payload.get("candidate_sha256"):
            hashes.add(str(payload["candidate_sha256"]))
    return hashes


def _source_entry(path: Path, *, base: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(base)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _collect_terminal_result(
    result_root: Path,
    expected_job: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    """Collect one exact latest attempt without scanning unrelated campaigns."""

    job_id = str(expected_job["job_id"])
    result_dir = result_root / job_id
    errors: list[dict[str, Any]] = []
    latest_path = result_dir / "latest_attempt.json"
    if not latest_path.is_file():
        return None, None, [{"check": "latest_attempt_exists", "job_id": job_id}]
    latest = read_json(latest_path)
    attempt_relative = latest.get("attempt_path")
    if not isinstance(attempt_relative, str) or not attempt_relative.startswith("attempts/"):
        return None, None, [{"check": "latest_attempt_path", "job_id": job_id}]
    attempt_dir = result_dir / attempt_relative
    if not attempt_dir.is_dir():
        return None, None, [{"check": "attempt_directory_exists", "job_id": job_id}]

    required = {
        "status": attempt_dir / "status.json",
        "rendered_run": attempt_dir / "rendered_run.json",
        "execution_fingerprint": attempt_dir / "execution_fingerprint.json",
        "execution_inputs": attempt_dir / "execution_inputs.json",
    }
    missing = sorted(name for name, path in required.items() if not path.is_file())
    if missing:
        return None, None, [{"check": "required_result_files", "job_id": job_id, "missing": missing}]

    status = read_json(required["status"])
    rendered = read_json(required["rendered_run"])
    fingerprint = read_json(required["execution_fingerprint"])
    rendered_job = rendered.get("job") or {}
    summaries = [read_json(path) for path in sorted((attempt_dir / "metrics").glob("summary.rank*.json"))]
    summary_paths = sorted((attempt_dir / "metrics").glob("summary.rank*.json"))
    attempt_id = str(latest.get("execution_attempt_id") or "")

    if latest.get("job_id") != job_id:
        errors.append({"check": "latest_job_id", "job_id": job_id})
    for name, value in (
        ("status_attempt_id", status.get("execution_attempt_id")),
        ("rendered_attempt_id", rendered.get("execution_attempt_id")),
        ("fingerprint_attempt_id", fingerprint.get("execution_attempt_id")),
    ):
        if str(value or "") != attempt_id:
            errors.append({"check": name, "job_id": job_id, "expected": attempt_id, "actual": value})
    if status.get("job_id") != job_id or fingerprint.get("job_id") != job_id:
        errors.append({"check": "terminal_job_identity", "job_id": job_id})
    if latest.get("classification") != status.get("classification"):
        errors.append({"check": "latest_status_classification", "job_id": job_id})
    if latest.get("state") != "complete":
        errors.append({"check": "latest_state_complete", "job_id": job_id, "actual": latest.get("state")})
    if status.get("execution_fingerprint_quality") != "complete":
        errors.append(
            {
                "check": "execution_fingerprint_complete",
                "job_id": job_id,
                "actual": status.get("execution_fingerprint_quality"),
            }
        )
    # run_job binds the canonical JSON digest of the execution manifest, not
    # the byte-level file digest (indentation must not affect evidence identity).
    actual_fingerprint_sha = sha256_json(fingerprint)
    if status.get("execution_fingerprint_sha256") != actual_fingerprint_sha:
        errors.append({"check": "execution_fingerprint_sha256", "job_id": job_id})

    expected_ranks = int(expected_job["gpu_count"])
    summary_ranks = sorted(int(summary.get("rank", -1)) for summary in summaries)
    if status.get("classification") == "success" and summary_ranks != list(range(expected_ranks)):
        errors.append(
            {
                "check": "success_summary_rank_set",
                "job_id": job_id,
                "expected": list(range(expected_ranks)),
                "actual": summary_ranks,
            }
        )
    for summary in summaries:
        if summary.get("job_id") != job_id or str(summary.get("execution_attempt_id") or "") != attempt_id:
            errors.append({"check": "summary_identity", "job_id": job_id, "rank": summary.get("rank")})
        if int(summary.get("world_size", -1)) != expected_ranks:
            errors.append({"check": "summary_world_size", "job_id": job_id, "rank": summary.get("rank")})

    measured_seconds = max((_finite(summary.get("measured_seconds")) or 0.0 for summary in summaries), default=0.0)
    totals: dict[str, int] = defaultdict(int)
    for summary in summaries:
        for key, value in (summary.get("measured_totals") or {}).items():
            totals[str(key)] += int(value)
    metrics_available = bool(summaries and measured_seconds > 0.0 and totals.get("effective_tokens", 0) > 0)
    classification = str(status.get("classification"))
    if classification == "success" and not metrics_available:
        errors.append({"check": "success_metrics_available", "job_id": job_id})

    source_paths = {"latest_attempt": latest_path, **required}
    for index, path in enumerate(summary_paths):
        source_paths[f"summary_rank{index}"] = path
    optional_paths = {
        "runtime_identity": attempt_dir / "runtime_identity.json",
        "runtime_hardware": attempt_dir / "runtime_hardware.json",
        "runtime_mechanism": attempt_dir / "runtime_mechanism.json",
        "thermal_summary": attempt_dir / "thermal_summary.json",
        "nvidia_smi": attempt_dir / "nvidia_smi.csv",
        "train_log": attempt_dir / "train.log",
    }
    source_paths.update({name: path for name, path in optional_paths.items() if path.is_file()})
    source_manifest = {
        "job_id": job_id,
        "execution_attempt_id": attempt_id,
        "attempt_path": str(attempt_dir.relative_to(result_root)),
        "files": {
            name: _source_entry(path, base=result_root)
            for name, path in sorted(source_paths.items())
        },
    }
    source_manifest["manifest_sha256"] = sha256_json(source_manifest)

    result = {
        "job_id": job_id,
        "execution_attempt_id": attempt_id,
        "job": rendered_job,
        "classification": classification,
        "return_code": status.get("return_code"),
        "finished_unix": _finite(status.get("finished_unix")),
        "approval_design_sha256": status.get("approval_design_sha256"),
        "authorization_mode": status.get("authorization_mode"),
        "calibration_eligible": status.get("calibration_eligible") is True,
        "execution_fingerprint_quality": status.get("execution_fingerprint_quality"),
        "execution_fingerprint_sha256": actual_fingerprint_sha,
        "rank_summaries": len(summaries),
        "metrics_available": metrics_available,
        "measured_seconds": measured_seconds if measured_seconds > 0.0 else None,
        "computed_tokens_per_second": (
            totals.get("computed_tokens", 0) / measured_seconds if metrics_available else None
        ),
        "effective_tokens_per_second": (
            totals.get("effective_tokens", 0) / measured_seconds if metrics_available else None
        ),
        "samples_per_second": (
            totals.get("logical_samples", 0) / measured_seconds if metrics_available else None
        ),
        "max_allocated_bytes": max((int(summary.get("max_allocated") or 0) for summary in summaries), default=0),
        "max_reserved_bytes": max((int(summary.get("max_reserved") or 0) for summary in summaries), default=0),
        "gpu_mask": status.get("gpu_mask"),
        "thermal_observation": status.get("thermal_observation") or {},
    }
    return result, source_manifest, errors


def _config_mismatches(job: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
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
    return [field for field in fields if not _same(job.get(field), actual.get(field))]


def _prediction_mismatches(job: Mapping[str, Any], prediction: Mapping[str, Any]) -> list[str]:
    config = prediction.get("configuration") or {}
    pairs = (
        ("scenario_id", job.get("scenario_id"), prediction.get("comparison_group")),
        ("model_id", job.get("model_id"), config.get("model_id")),
        ("train_type", job.get("train_type"), config.get("training_mode")),
        ("dataset_id", job.get("dataset_id"), config.get("dataset_id")),
        ("dataset_profile_sha256", job.get("dataset_profile_sha256"), config.get("dataset_profile_sha256")),
        ("cutoff_len", job.get("cutoff_len"), config.get("cutoff_len")),
        ("target_gbs", job.get("target_gbs"), config.get("target_gbs")),
        ("gpu_count", job.get("gpu_count"), config.get("gpu_count")),
        ("mbs", job.get("mbs"), config.get("physical_mbs")),
        ("gradient_accumulation_steps", job.get("gradient_accumulation_steps"), config.get("gradient_accumulation_steps")),
        ("zero_stage", job.get("zero_stage"), config.get("zero_stage")),
        ("gc", job.get("gc"), config.get("gradient_checkpointing")),
        ("packing", job.get("packing"), config.get("packing")),
        ("offload", job.get("offload"), config.get("offload")),
    )
    return [name for name, left, right in pairs if not _same(left, right)]


def _embedded_prediction_mismatches(
    embedded: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> list[str]:
    memory = prediction.get("memory") or {}
    throughput = prediction.get("throughput") or {}
    pairs = (
        ("prediction_available", embedded.get("prediction_available"), memory.get("prediction_available")),
        ("allocated_anchor_bytes", embedded.get("allocated_anchor_bytes"), memory.get("allocated_anchor_bytes")),
        ("allocated_center_bytes", embedded.get("allocated_center_bytes"), memory.get("allocated_center_bytes")),
        ("reserved_center_bytes", embedded.get("reserved_center_bytes"), memory.get("reserved_center_bytes")),
        (
            "operational_upper_reserved_bytes",
            embedded.get("operational_upper_reserved_bytes"),
            memory.get("operational_p95_reserved_bytes"),
        ),
        ("safe_limit_bytes", embedded.get("safe_limit_bytes"), memory.get("safe_limit_bytes")),
        ("predicted_safe", embedded.get("predicted_safe"), memory.get("admitted")),
        (
            "v4b_prediction_available",
            embedded.get("v4b_prediction_available"),
            throughput.get("prediction_available"),
        ),
        (
            "v4b_throughput_proxy",
            embedded.get("v4b_throughput_proxy"),
            throughput.get("throughput_proxy_tokens_per_second"),
        ),
    )
    return [name for name, left, right in pairs if not _same(left, right)]


def _memory_slice(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["outcome"] == "success" and row["observed_reserved_bytes"]]
    center_apes = [float(row["center_absolute_relative_error"]) for row in successes]
    allocated_apes = [
        float(row["allocated_absolute_relative_error"])
        for row in successes
        if row["allocated_absolute_relative_error"] is not None
    ]
    return {
        "rows": len(rows),
        "success_rows": len(successes),
        "predicted_admitted_rows": sum(row["predicted_admit"] for row in rows),
        "actual_safe_success_rows": sum(row["actual_safe_success"] for row in rows),
        "false_rejected_safe_rows": sum(row["false_rejected_safe"] for row in rows),
        "upper_coverage_fraction": _mean(
            float(row["upper_covers_observed"])
            for row in successes
            if row["upper_covers_observed"] is not None
        ),
        "center_mape": fmean(center_apes) if center_apes else None,
        "center_p90_ape": _percentile(center_apes, 90.0),
        "allocated_mape": fmean(allocated_apes) if allocated_apes else None,
        "allocated_p90_ape": _percentile(allocated_apes, 90.0),
    }


def evaluate(
    *,
    campaign_id: str = DEFAULT_CAMPAIGN_ID,
    queue_path: Path = DEFAULT_QUEUE,
    design_path: Path = DEFAULT_DESIGN,
    prediction_path: Path = DEFAULT_PREDICTIONS,
    split_path: Path = DEFAULT_SPLIT,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> dict[str, Any]:
    queue = read_jsonl(queue_path)
    design = read_json(design_path)
    frozen = read_json(prediction_path)
    split = read_json(split_path)
    predictions = frozen.get("predictions") or []
    slots = design.get("candidate_slots") or []

    queue_by_job = {str(row["job_id"]): row for row in queue}
    prediction_by_request = {str(row["request_id"]): row for row in predictions}
    slot_by_id = {str(row["candidate_slot_id"]): row for row in slots}
    errors: list[dict[str, Any]] = []

    duplicate_checks = {
        "queue_job_ids": _duplicates(str(row["job_id"]) for row in queue),
        "queue_candidate_slot_ids": _duplicates(str(row["candidate_slot_id"]) for row in queue),
        "queue_predictor_request_ids": _duplicates(str(row["predictor_request_id"]) for row in queue),
        "prediction_request_ids": _duplicates(str(row["request_id"]) for row in predictions),
        "design_candidate_slot_ids": _duplicates(str(row["candidate_slot_id"]) for row in slots),
    }
    for name, duplicates in duplicate_checks.items():
        if duplicates:
            errors.append({"check": name, "duplicates": duplicates})
    if design.get("campaign_id") != campaign_id or any(row.get("campaign_id") != campaign_id for row in queue):
        errors.append({"check": "campaign_id_binding"})
    if set(slot_by_id) != {str(row["candidate_slot_id"]) for row in queue}:
        errors.append({"check": "queue_design_slot_set"})
    if set(prediction_by_request) != {str(row["predictor_request_id"]) for row in queue}:
        errors.append({"check": "queue_prediction_request_set"})

    prediction_sha = sha256_file(prediction_path)
    bound_prediction = design.get("frozen_prediction_binding") or {}
    if bound_prediction.get("sha256") != prediction_sha:
        errors.append(
            {
                "check": "frozen_prediction_sha256",
                "expected": bound_prediction.get("sha256"),
                "actual": prediction_sha,
            }
        )

    split_profiles = sorted(str(row["profile_id"]) for row in split.get("splits") or [])
    queue_profiles = sorted({str(row["dataset_id"]) for row in queue})
    fit_overlap = split.get("fit_overlap") or {}
    freshness_passes = bool(
        queue_profiles == split_profiles
        and fit_overlap.get("selected_source_rows_used_for_fit") is False
        and fit_overlap.get("selected_dataset_ids_used_for_fit") is False
    )
    if not freshness_passes:
        errors.append(
            {
                "check": "fresh_holdout_split",
                "queue_profiles": queue_profiles,
                "split_profiles": split_profiles,
                "fit_overlap": fit_overlap,
            }
        )

    receipt_hashes = _load_receipt_hashes()
    result_rows: dict[str, dict[str, Any]] = {}
    result_manifests: list[dict[str, Any]] = []
    for job_id, job in sorted(queue_by_job.items()):
        result, manifest, collection_errors = _collect_terminal_result(results_root, job)
        errors.extend(collection_errors)
        if result is not None:
            result_rows[job_id] = result
        if manifest is not None:
            result_manifests.append(manifest)

    approval_hashes = sorted(
        {str(row.get("approval_design_sha256")) for row in result_rows.values() if row.get("approval_design_sha256")}
    )
    unknown_approval_hashes = sorted(set(approval_hashes) - receipt_hashes)
    if unknown_approval_hashes:
        errors.append({"check": "approval_receipts", "unknown_hashes": unknown_approval_hashes})

    joined: list[dict[str, Any]] = []
    current_hashes: dict[str, str] = {}
    for job_id, job in sorted(queue_by_job.items()):
        result = result_rows.get(job_id)
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
        result_mismatches = _config_mismatches(job, result.get("job") or {})
        prediction_mismatches = _prediction_mismatches(job, prediction)
        embedded_mismatches = _embedded_prediction_mismatches(job.get("frozen_prediction") or {}, prediction)
        if result_mismatches:
            errors.append({"check": "queue_result_configuration", "job_id": job_id, "fields": result_mismatches})
        if prediction_mismatches:
            errors.append({"check": "queue_prediction_configuration", "job_id": job_id, "fields": prediction_mismatches})
        if embedded_mismatches:
            errors.append({"check": "embedded_frozen_prediction", "job_id": job_id, "fields": embedded_mismatches})
        if slot.get("predictor_request_id") != job.get("predictor_request_id"):
            errors.append({"check": "slot_predictor_request_id", "job_id": job_id})
        if sha256_json(slot.get("frozen_prediction") or {}) != sha256_json(job.get("frozen_prediction") or {}):
            errors.append({"check": "slot_queue_frozen_prediction", "job_id": job_id})
        if slot.get("calibration_partition") != job.get("calibration_partition"):
            errors.append({"check": "slot_queue_calibration_partition", "job_id": job_id})

        for path_field, sha_field in (("data_path", "data_sha256"), ("dataset_profile_path", "dataset_profile_sha256")):
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
        observed_allocated = _finite(result.get("max_allocated_bytes"))
        safe_limit = _finite(memory.get("safe_limit_bytes"))
        center = _finite(memory.get("reserved_center_bytes"))
        allocated_center = _finite(memory.get("allocated_center_bytes"))
        upper = _finite(memory.get("admission_upper_reserved_bytes"))
        outcome = str(result.get("classification"))
        actual_safe = bool(
            outcome == "success"
            and observed_reserved is not None
            and safe_limit is not None
            and observed_reserved <= safe_limit
        )
        predicted_admit = memory.get("admitted") is True
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
                "calibration_partition": job.get("calibration_partition"),
                "outcome": outcome,
                "predicted_admit": predicted_admit,
                "actual_safe_success": actual_safe,
                "false_rejected_safe": bool(actual_safe and not predicted_admit),
                "predicted_allocated_center_bytes": allocated_center,
                "predicted_memory_center_bytes": center,
                "predicted_memory_upper_bytes": upper,
                "safe_limit_bytes": safe_limit,
                "observed_allocated_bytes": observed_allocated,
                "observed_reserved_bytes": observed_reserved,
                "allocated_absolute_relative_error": (
                    abs(allocated_center - observed_allocated) / observed_allocated
                    if allocated_center is not None and observed_allocated and observed_allocated > 0.0
                    else None
                ),
                "center_signed_relative_error": (
                    (center - observed_reserved) / observed_reserved
                    if center is not None and observed_reserved and observed_reserved > 0.0
                    else None
                ),
                "center_absolute_relative_error": (
                    abs(center - observed_reserved) / observed_reserved
                    if center is not None and observed_reserved and observed_reserved > 0.0
                    else None
                ),
                "upper_covers_observed": (
                    bool(upper is not None and observed_reserved is not None and upper >= observed_reserved)
                    if outcome == "success"
                    else None
                ),
                "predicted_throughput_proxy": _finite(throughput.get("throughput_proxy_tokens_per_second")),
                "observed_effective_tokens_per_second": _finite(result.get("effective_tokens_per_second")),
                "observed_computed_tokens_per_second": _finite(result.get("computed_tokens_per_second")),
                "observed_samples_per_second": _finite(result.get("samples_per_second")),
                "measured_seconds": _finite(result.get("measured_seconds")),
                "execution_attempt_id": result.get("execution_attempt_id"),
                "approval_design_sha256": result.get("approval_design_sha256"),
                "execution_fingerprint_quality": result.get("execution_fingerprint_quality"),
                "calibration_eligible": result.get("calibration_eligible") is True,
                "rank_summaries": result.get("rank_summaries"),
                "metrics_available": result.get("metrics_available") is True,
                "gpu_mask": result.get("gpu_mask"),
            }
        )

    partition_rows = sum(
        isinstance(row.get("calibration_partition"), Mapping)
        and (row.get("calibration_partition") or {}).get("role") == "holdout"
        and bool((row.get("calibration_partition") or {}).get("split_unit_id"))
        and bool((row.get("calibration_partition") or {}).get("policy"))
        for row in queue
    )
    execution_checks = {
        "all_results_joined": len(joined) == len(queue),
        "all_terminal_success_or_oom": all(row["outcome"] in {"success", "oom"} for row in joined),
        "all_success_metrics_available": all(
            row["outcome"] != "success" or row["metrics_available"] for row in joined
        ),
        "all_execution_fingerprints_complete": all(
            row["execution_fingerprint_quality"] == "complete" for row in joined
        ),
        "all_calibration_eligible": all(row["calibration_eligible"] for row in joined),
        "all_rank_summary_counts_match_gpu_count": all(
            row["outcome"] != "success" or int(row["rank_summaries"] or -1) == int(row["gpu_count"])
            for row in joined
        ),
        "all_holdout_partitions_bound": partition_rows == len(queue),
    }
    for name, passes in execution_checks.items():
        if not passes:
            errors.append({"check": name})

    outcome_counts = Counter(row["outcome"] for row in joined)
    integrity = {
        "passes": not errors,
        "campaign_id": campaign_id,
        "queue_jobs": len(queue),
        "design_candidate_slots": len(slots),
        "frozen_prediction_rows": len(predictions),
        "joined_rows": len(joined),
        "outcomes": dict(sorted(outcome_counts.items())),
        "duplicate_checks": duplicate_checks,
        "execution_checks": execution_checks,
        "fresh_holdout_split_passes": freshness_passes,
        "approval_design_sha256_values": approval_hashes,
        "approval_hashes_have_promotion_receipts": not unknown_approval_hashes,
        "errors": errors,
    }

    contract = design.get("acceptance_contract") or {}
    minimum_coverage = float(contract.get("scenario_equal_p05_upper_coverage", 0.95))
    maximum_false_safe_oom = int(contract.get("false_safe_oom", 0))
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
        minimum_coverage=minimum_coverage,
        maximum_false_safe_oom=maximum_false_safe_oom,
    )
    success_rows = [row for row in joined if row["outcome"] == "success"]
    center_apes = [float(row["center_absolute_relative_error"]) for row in success_rows]
    allocated_apes = [
        float(row["allocated_absolute_relative_error"])
        for row in success_rows
        if row["allocated_absolute_relative_error"] is not None
    ]
    center_mape = fmean(center_apes) if center_apes else None
    center_p90 = _percentile(center_apes, 90.0)
    allocated_mape = fmean(allocated_apes) if allocated_apes else None
    allocated_p90 = _percentile(allocated_apes, 90.0)
    center_mape_limit = float(contract.get("reserved_center_mean_absolute_percentage_error", 0.06))
    center_p90_limit = float(contract.get("reserved_center_p90_absolute_percentage_error", 0.12))
    center_precision_passes = bool(
        center_mape is not None
        and center_p90 is not None
        and center_mape <= center_mape_limit
        and center_p90 <= center_p90_limit
    )

    by_model_train: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_profile_model_train: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_model_train[(row["model_id"], row["train_type"])].append(row)
        by_profile_model_train[(row["dataset_id"], row["model_id"], row["train_type"])].append(row)
    model_slices = [
        {"model_id": key[0], "train_type": key[1], **_memory_slice(rows)}
        for key, rows in sorted(by_model_train.items())
    ]
    profile_slices = [
        {
            "dataset_id": key[0],
            "model_id": key[1],
            "train_type": key[2],
            **_memory_slice(rows),
        }
        for key, rows in sorted(by_profile_model_train.items())
    ]
    actual_safe_rows = [row for row in joined if row["actual_safe_success"]]
    false_rejected = [row for row in actual_safe_rows if row["false_rejected_safe"]]
    no_slice_safety_regression = all(
        row["upper_coverage_fraction"] == 1.0 for row in model_slices if row["success_rows"]
    ) and memory_standard["false_safe_oom"] == 0
    memory = {
        **memory_standard,
        "center_mape": center_mape,
        "center_median_ape": median(center_apes) if center_apes else None,
        "center_p90_ape": center_p90,
        "center_mape_limit": center_mape_limit,
        "center_p90_ape_limit": center_p90_limit,
        "center_precision_passes": center_precision_passes,
        "allocated_mape": allocated_mape,
        "allocated_p90_ape": allocated_p90,
        "actual_safe_success_rows": len(actual_safe_rows),
        "predicted_admitted_actual_safe_rows": sum(row["predicted_admit"] for row in actual_safe_rows),
        "false_rejected_safe_rows": len(false_rejected),
        "false_rejection_fraction_of_actual_safe": (
            len(false_rejected) / len(actual_safe_rows) if actual_safe_rows else None
        ),
        "actual_safe_admission_recall": (
            sum(row["predicted_admit"] for row in actual_safe_rows) / len(actual_safe_rows)
            if actual_safe_rows
            else None
        ),
        "no_supported_slice_safety_regression": no_slice_safety_regression,
        "safety_gate_passes": memory_standard["passes"],
        "replacement_gate_passes": bool(
            integrity["passes"]
            and memory_standard["passes"]
            and center_precision_passes
            and no_slice_safety_regression
        ),
        "model_train_slices": model_slices,
        "profile_model_train_slices": profile_slices,
        "false_rejected_candidates": [
            {
                "job_id": row["job_id"],
                "candidate_id": row["candidate_id"],
                "scenario_id": row["scenario_id"],
                "gpu_count": row["gpu_count"],
                "mbs": row["mbs"],
                "zero_stage": row["zero_stage"],
                "observed_reserved_gib": row["observed_reserved_bytes"] / GIB,
                "safe_limit_gib": row["safe_limit_bytes"] / GIB,
                "predicted_center_gib": row["predicted_memory_center_bytes"] / GIB,
                "predicted_upper_gib": row["predicted_memory_upper_bytes"] / GIB,
                "observed_effective_tokens_per_second": row["observed_effective_tokens_per_second"],
            }
            for row in false_rejected
        ],
        "detail_rows": [
            {
                "job_id": row["job_id"],
                "candidate_id": row["candidate_id"],
                "scenario_id": row["scenario_id"],
                "model_id": row["model_id"],
                "train_type": row["train_type"],
                "gpu_count": row["gpu_count"],
                "mbs": row["mbs"],
                "zero_stage": row["zero_stage"],
                "gc": row["gc"],
                "outcome": row["outcome"],
                "predicted_admit": row["predicted_admit"],
                "actual_safe_success": row["actual_safe_success"],
                "false_rejected_safe": row["false_rejected_safe"],
                "predicted_allocated_center_gib": (
                    row["predicted_allocated_center_bytes"] / GIB
                    if row["predicted_allocated_center_bytes"] is not None
                    else None
                ),
                "observed_allocated_gib": (
                    row["observed_allocated_bytes"] / GIB if row["observed_allocated_bytes"] is not None else None
                ),
                "predicted_center_gib": row["predicted_memory_center_bytes"] / GIB,
                "predicted_upper_gib": row["predicted_memory_upper_bytes"] / GIB,
                "observed_reserved_gib": row["observed_reserved_bytes"] / GIB,
                "safe_limit_gib": row["safe_limit_bytes"] / GIB,
                "center_absolute_relative_error": row["center_absolute_relative_error"],
                "allocated_absolute_relative_error": row["allocated_absolute_relative_error"],
                "upper_covers_observed": row["upper_covers_observed"],
            }
            for row in joined
        ],
        "interpretation": {
            "all_ten_jobs_succeeded_so_oom_boundary_was_not_observed": True,
            "upper_coverage_does_not_measure_false_rejection": True,
            "frozen_center_thresholds_are_binding_and_cannot_be_relaxed_post_hoc": True,
        },
    }

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        grouped[(row["scenario_id"], row["gpu_count"])].append(row)
    standard_groups = []
    ranking_details = []
    for (scenario_id, gpu_count), rows in sorted(grouped.items()):
        endpoint_id = f"{scenario_id}__g{gpu_count}"
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
        standard_groups.append({"scenario_id": endpoint_id, "candidates": candidates})
        safe = [row for row in rows if row["actual_safe_success"] and row["observed_effective_tokens_per_second"]]
        admitted_scored = [
            row
            for row in rows
            if row["predicted_admit"]
            and row["predicted_throughput_proxy"] is not None
            and row["observed_effective_tokens_per_second"] is not None
        ]
        predicted_winner = max(admitted_scored, key=lambda row: row["predicted_throughput_proxy"], default=None)
        observed_best = max(safe, key=lambda row: row["observed_effective_tokens_per_second"], default=None)
        regret = (
            1.0
            - float(predicted_winner["observed_effective_tokens_per_second"])
            / float(observed_best["observed_effective_tokens_per_second"])
            if predicted_winner is not None and observed_best is not None
            else None
        )
        ranking_details.append(
            {
                "endpoint_id": endpoint_id,
                "scenario_id": scenario_id,
                "gpu_count": gpu_count,
                "candidate_count": len(rows),
                "actual_safe_success_count": len(safe),
                "predicted_admitted_count": sum(row["predicted_admit"] for row in rows),
                "v4b_scored_count": sum(row["predicted_throughput_proxy"] is not None for row in rows),
                "status": "evaluated" if len(safe) >= 2 and predicted_winner and observed_best else "insufficient_evidence",
                "predicted_winner": predicted_winner["candidate_id"] if predicted_winner else None,
                "observed_best": observed_best["candidate_id"] if observed_best else None,
                "top1_exact": (
                    predicted_winner["candidate_id"] == observed_best["candidate_id"]
                    if predicted_winner and observed_best
                    else None
                ),
                "selected_observed_tokens_per_second": (
                    predicted_winner["observed_effective_tokens_per_second"] if predicted_winner else None
                ),
                "best_observed_tokens_per_second": (
                    observed_best["observed_effective_tokens_per_second"] if observed_best else None
                ),
                "top1_regret": regret,
                "top1_within_10_percent": bool(regret is not None and regret <= 0.10 + 1e-12),
                "candidates": [
                    {
                        "candidate_id": row["candidate_id"],
                        "job_id": row["job_id"],
                        "mbs": row["mbs"],
                        "zero_stage": row["zero_stage"],
                        "gc": row["gc"],
                        "predicted_admit": row["predicted_admit"],
                        "predicted_throughput_proxy": row["predicted_throughput_proxy"],
                        "observed_effective_tokens_per_second": row["observed_effective_tokens_per_second"],
                    }
                    for row in sorted(rows, key=lambda item: item["mbs"])
                ],
            }
        )

    ranking_standard = evaluate_ranking_acceptance(standard_groups, maximum_top1_regret=0.10)
    eligible_ranking = [row for row in ranking_details if row["status"] == "evaluated"]
    scored_rows = [
        row
        for row in joined
        if row["predicted_throughput_proxy"] is not None
        and row["observed_effective_tokens_per_second"] is not None
        and row["predicted_throughput_proxy"] > 0.0
        and row["observed_effective_tokens_per_second"] > 0.0
    ]
    proxy_apes = [
        abs(row["predicted_throughput_proxy"] - row["observed_effective_tokens_per_second"])
        / row["observed_effective_tokens_per_second"]
        for row in scored_rows
    ]
    ranking = {
        **ranking_standard,
        "endpoint_count": len(ranking_details),
        "eligible_endpoint_count": len(eligible_ranking),
        "exact_top1_count": sum(row["top1_exact"] is True for row in eligible_ranking),
        "endpoints_within_10_percent_count": sum(row["top1_within_10_percent"] for row in eligible_ranking),
        "endpoints_within_10_percent_fraction": (
            _mean(float(row["top1_within_10_percent"]) for row in eligible_ranking)
            if eligible_ranking
            else None
        ),
        "scoreless_actual_safe_candidates": sum(
            row["actual_safe_success"] and row["predicted_throughput_proxy"] is None for row in joined
        ),
        "absolute_proxy_diagnostics": {
            "gating_metric": False,
            "absolute_scale_trusted": frozen.get("absolute_throughput_scale_trusted") is True,
            "rows": len(scored_rows),
            "mape": fmean(proxy_apes) if proxy_apes else None,
            "p90_ape": _percentile(proxy_apes, 90.0),
        },
        "detail_groups": ranking_details,
        "decision": {
            "diagnostic_regret_gate_passes": ranking_standard["passes"],
            "formal_v4b_promotion_claim_allowed": False,
            "reason": (
                "The final holdout contract targets memory replacement; endpoints have no exact repeats, "
                "and one actual-safe candidate was intentionally unscored after memory rejection."
            ),
        },
    }

    governance = design.get("governance") or {}
    ingestion = {
        "acceptance_evidence_eligible": integrity["passes"],
        "holdout_partition_rows": partition_rows,
        "required_rows": len(queue),
        "fit_or_anchor_ingestion_allowed": False,
        "holdout_may_be_reused_to_validate_a_refit": False,
        "reason": "The frozen design declares holdout_rows_may_enter_fit=false.",
        "design_binding": governance.get("holdout_rows_may_enter_fit") is False,
    }

    blockers = []
    if not integrity["passes"]:
        blockers.append("evidence_integrity_failed")
    if not memory["safety_gate_passes"]:
        blockers.append("memory_safety_gate_failed")
    if not memory["center_precision_passes"]:
        blockers.append("frozen_center_precision_gate_failed")
    if not memory["no_supported_slice_safety_regression"]:
        blockers.append("supported_slice_safety_regression")
    if memory["false_rejected_safe_rows"]:
        blockers.append("actual_safe_candidate_false_rejected")
    decisions = {
        "profile_aware_memory_challenger_replacement_allowed": memory["replacement_gate_passes"],
        "publication_allowed": False,
        "automatic_execution_allowed": False,
        "current_memory_artifact_must_remain_unchanged": True,
        "v4b_regret_regression_observed": bool(
            eligible_ranking and ranking_standard["worst_top1_regret"] > 0.10
        ),
        "final_holdout_is_now_consumed": True,
        "may_reuse_this_holdout_for_next_acceptance": False,
        "blockers": blockers,
        "next_action": (
            "Redesign the Qwen3-8B LoRA profile correction with bounded extrapolation; if these rows are "
            "used for diagnosis or a future fit, label them seen and freeze a completely new holdout."
        ),
    }

    completed_values = [row.get("finished_unix") for row in result_rows.values() if row.get("finished_unix")]
    completed_at = datetime.fromtimestamp(max(completed_values), timezone.utc).isoformat() if completed_values else None
    report = {
        "schema": SCHEMA,
        "generated_at_utc": completed_at,
        "timestamp_semantics": "deterministic maximum terminal result finished_unix",
        "campaign_id": campaign_id,
        "evaluation_mode": "frozen_predictions_no_refit",
        "source_files": {
            "queue": {"path": str(queue_path.resolve()), "sha256": sha256_file(queue_path)},
            "design": {"path": str(design_path.resolve()), "sha256": sha256_file(design_path)},
            "frozen_predictions": {"path": str(prediction_path.resolve()), "sha256": prediction_sha},
            "split_manifest": {"path": str(split_path.resolve()), "sha256": sha256_file(split_path)},
            "terminal_result_manifests_sha256": sha256_json(result_manifests),
            "terminal_result_manifests": result_manifests,
        },
        "integrity": integrity,
        "ingestion_eligibility": ingestion,
        "acceptance_contract": contract,
        "memory": memory,
        "throughput_ranking": ranking,
        "evidence_scope": {
            "profiles": queue_profiles,
            "model_train_slices": sorted({f"{row['model_id']}:{row['train_type']}" for row in joined}),
            "gpu_counts": sorted({row["gpu_count"] for row in joined}),
            "configurations": len(joined),
            "exact_repeats_per_configuration": 1,
            "all_jobs_succeeded": outcome_counts == Counter({"success": len(joined)}),
            "oom_boundary_evidence_present": any(row["outcome"] == "oom" for row in joined),
            "packing_values": sorted({row["packing"] for row in joined}),
            "offload_values": sorted({row["offload"] for row in joined}),
        },
        "decisions": decisions,
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
        "# H800 profile-aware 显存 final unseen 冻结验收报告（2026-08-03）",
        "",
        "> 本报告只评估 GPU 启动前冻结的 profile-aware 显存 challenger 与未修改的 v4b。",
        "> 没有重训、没有回灌 holdout、没有覆盖历史锚点，也没有修改候选或验收阈值。",
        "",
        "## 一句话结论",
        "",
        "10/10 配置均成功，显存 upper 覆盖 10/10、false-safe OOM 为 0；但 reserved center MAPE "
        f"为 {_format_percent(memory['center_mape'])}、P90 APE 为 {_format_percent(memory['center_p90_ape'])}，"
        "且 1 个实际安全的 Qwen3-8B LoRA 配置被误拒。因此 profile-aware challenger 未通过冻结的替换门，"
        "不得发布。14B Full 切片准确，问题集中在 8B LoRA 的数据画像外推。",
        "",
        "## 1. 证据完整性与治理",
        "",
        f"- 队列 / 设计槽位 / 冻结预测 / 实测：{integrity['queue_jobs']} / "
        f"{integrity['design_candidate_slots']} / {integrity['frozen_prediction_rows']} / {integrity['joined_rows']}。",
        f"- 终态：`{json.dumps(integrity['outcomes'], ensure_ascii=False)}`。",
        f"- SHA、approval receipt、配置 join、执行指纹、rank summary、数据/profile 校验："
        f"{'全部通过' if integrity['passes'] else '失败'}。",
        f"- Holdout partition：{report['ingestion_eligibility']['holdout_partition_rows']}/"
        f"{report['ingestion_eligibility']['required_rows']} 完整绑定。",
        "- 这 10 条只能保存为版本化 acceptance evidence；不得加入当前 fit/anchor，也不得再作为下一版 unseen holdout。",
        "",
        "## 2. 显存验收",
        "",
        "| 指标 | 冻结门槛 | 结果 | 判定 |",
        "|---|---:|---:|:---:|",
        f"| false-safe OOM | = {memory['maximum_false_safe_oom']} | {memory['false_safe_oom']} | "
        f"{'通过' if memory['false_safe_oom'] <= memory['maximum_false_safe_oom'] else '失败'} |",
        f"| 场景等权 P05 upper coverage | ≥ {_format_percent(memory['minimum_coverage'])} | "
        f"{_format_percent(memory['scenario_equal_p05_coverage'])} | {'通过' if memory['safety_gate_passes'] else '失败'} |",
        f"| reserved center MAPE | ≤ {_format_percent(memory['center_mape_limit'])} | "
        f"{_format_percent(memory['center_mape'])} | {'通过' if memory['center_mape'] <= memory['center_mape_limit'] else '失败'} |",
        f"| reserved center P90 APE | ≤ {_format_percent(memory['center_p90_ape_limit'])} | "
        f"{_format_percent(memory['center_p90_ape'])} | {'通过' if memory['center_p90_ape'] <= memory['center_p90_ape_limit'] else '失败'} |",
        f"| 实际安全候选 admission recall | 诊断 | {_format_percent(memory['actual_safe_admission_recall'])} | "
        f"误拒 {memory['false_rejected_safe_rows']} 个 |",
        f"| challenger 替换门 | 全部门槛通过 | — | {'通过' if memory['replacement_gate_passes'] else '失败'} |",
        "",
        "本批全部成功，因此 10/10 upper coverage 证明的是保守覆盖，不是 OOM 边界已经验收。"
        "本批最重要的新信息是误拒和切片外推误差。",
        "",
        "### 2.1 按模型/训练方式",
        "",
        "| 模型 | 训练 | 配置数 | center MAPE | center P90 | allocated MAPE | upper覆盖 | 误拒 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in memory["model_train_slices"]:
        lines.append(
            f"| {row['model_id']} | {row['train_type']} | {row['rows']} | "
            f"{_format_percent(row['center_mape'])} | {_format_percent(row['center_p90_ape'])} | "
            f"{_format_percent(row['allocated_mape'])} | {_format_percent(row['upper_coverage_fraction'])} | "
            f"{row['false_rejected_safe_rows']} |"
        )
    lines.extend(
        [
            "",
            "### 2.2 逐配置结果",
            "",
            "| 数据画像 | 模型 | GPU | MBS | Z | GC | center/upper/实测 GiB | APE | 准入 | 实际安全 |",
            "|---|---|---:|---:|---:|:---:|---:|---:|:---:|:---:|",
        ]
    )
    for row in sorted(
        memory["detail_rows"],
        key=lambda item: (item["scenario_id"], item["gpu_count"], item["mbs"]),
    ):
        lines.append(
            f"| {row['scenario_id'].split('__')[0]} | {row['model_id']} {row['train_type']} | "
            f"{row['gpu_count']} | {row['mbs']} | {row['zero_stage']} | {'on' if row['gc'] else 'off'} | "
            f"{_format_number(row['predicted_center_gib'])} / {_format_number(row['predicted_upper_gib'])} / "
            f"{_format_number(row['observed_reserved_gib'])} | {_format_percent(row['center_absolute_relative_error'])} | "
            f"{'是' if row['predicted_admit'] else '否'} | {'是' if row['actual_safe_success'] else '否'} |"
        )

    false_rejected = memory["false_rejected_candidates"]
    if false_rejected:
        row = false_rejected[0]
        lines.extend(
            [
                "",
                "### 2.3 误拒案例",
                "",
                f"`{row['scenario_id']}` 的 GPU={row['gpu_count']}、MBS={row['mbs']}、ZeRO-{row['zero_stage']} "
                f"实测 reserved 仅 {_format_number(row['observed_reserved_gib'])} GiB，安全线为 "
                f"{_format_number(row['safe_limit_gib'])} GiB；模型却给出 center/upper "
                f"{_format_number(row['predicted_center_gib'])}/{_format_number(row['predicted_upper_gib'])} GiB 并拒绝。",
                "",
                "这不是安全事故，而是明显的可用性问题：显存门删掉了一个真实可行候选。",
            ]
        )

    lines.extend(
        [
            "",
            "## 3. 显存门后的 v4b 诊断",
            "",
            f"- 共有 {ranking['endpoint_count']} 个“场景 + 卡数”端点，其中 "
            f"{ranking['eligible_endpoint_count']} 个端点有至少两个实测安全候选，可计算 regret。",
            f"- Top-1 完全命中：{ranking['exact_top1_count']}/{ranking['eligible_endpoint_count']}。",
            f"- Hit@90%：{ranking['endpoints_within_10_percent_count']}/{ranking['eligible_endpoint_count']} "
            f"({_format_percent(ranking['endpoints_within_10_percent_fraction'])})。",
            f"- mean / worst regret：{_format_percent(ranking['mean_top1_regret'])} / "
            f"{_format_percent(ranking['worst_top1_regret'])}。",
            f"- 有 {ranking['scoreless_actual_safe_candidates']} 个实际安全候选因显存拒绝而没有 v4b 分数。",
            "- 这说明 v4b 没有表现出明显回归，但该批没有重复运行，且验收合同的主目标是显存 challenger，"
            "所以不能据此扩大 v4b 的正式支持域。",
            "",
            "| 场景 | GPU | 实测安全/准入/有分 | Top-1 | 选中/最好 tok/s | regret |",
            "|---|---:|---:|:---:|---:|---:|",
        ]
    )
    for row in ranking["detail_groups"]:
        if row["status"] != "evaluated":
            continue
        lines.append(
            f"| {row['scenario_id']} | {row['gpu_count']} | {row['actual_safe_success_count']} / "
            f"{row['predicted_admitted_count']} / {row['v4b_scored_count']} | "
            f"{'命中' if row['top1_exact'] else '未命中'} | "
            f"{_format_number(row['selected_observed_tokens_per_second'], 1)} / "
            f"{_format_number(row['best_observed_tokens_per_second'], 1)} | "
            f"{_format_percent(row['top1_regret'])} |"
        )

    lines.extend(
        [
            "",
            "## 4. 冻结决策",
            "",
            f"- profile-aware challenger 替换当前模型：`{str(decisions['profile_aware_memory_challenger_replacement_allowed']).lower()}`。",
            "- 当前显存 artifact、历史锚点和生产策略保持不变。",
            "- 这两个 unseen profile 已被消费；下一版不能继续拿它们宣称 fresh acceptance。",
            "- 下一步先修 8B LoRA 的 profile correction 外推，不立即铺开新 GPU 矩阵。",
            "- 若未来把本批用于诊断或拟合，必须显式标为 seen/calibration，并另冻完全新的 holdout。",
            "",
            "## 5. 产物",
            "",
            f"- 机器可读报告：`{DEFAULT_OUTPUT}`",
            f"- 不可覆盖观测快照：`{DEFAULT_SNAPSHOT}`",
            f"- 冻结预测：`{report['source_files']['frozen_predictions']['path']}`",
            "- 重跑评估：`/fine-tuning-launcher/.venv/bin/python "
            "offline_experiments/scripts/evaluate_h800_final_unseen_holdout_v1.py`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if read_json(path) != payload:
            raise FileExistsError(f"refusing to overwrite changed immutable artifact: {path}")
        return
    write_json(path, payload)


def _write_immutable_text(path: Path, payload: str) -> None:
    normalized = payload.rstrip() + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != normalized:
            raise FileExistsError(f"refusing to overwrite changed immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalized, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()

    report = evaluate(
        campaign_id=args.campaign_id,
        queue_path=args.queue,
        design_path=args.design,
        prediction_path=args.predictions,
        split_path=args.split,
        results_root=args.results_root,
    )
    snapshot = {
        "schema": "sft_h800_final_unseen_holdout_observation_snapshot/v1",
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
                "memory_safety_gate_passes": report["memory"]["safety_gate_passes"],
                "memory_center_precision_passes": report["memory"]["center_precision_passes"],
                "challenger_replacement_allowed": report["decisions"][
                    "profile_aware_memory_challenger_replacement_allowed"
                ],
                "ranking_hit_at_90": report["throughput_ranking"]["endpoints_within_10_percent_fraction"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
