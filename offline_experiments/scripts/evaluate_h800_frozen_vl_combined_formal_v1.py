#!/usr/bin/env python3
"""Summarize formal paired image/video evidence without claiming acceptance."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    RESULTS_DIR,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from evaluate_h800_qwen35_vl_supplement_v1 import _result, _success_metrics
from prepare_h800_frozen_vl_combined_formal_v1 import CAMPAIGN_ID, PHASE_ID, QUEUE

OUTPUT = ARTIFACT_DIR / "h800_frozen_vl_combined_formal_results_v1.json"
HARDWARE = CONFIG_DIR / "hardware.json"


def _oom_terminal_semantics(
    job: dict[str, Any],
    result: dict[str, Any],
    *,
    results_root: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Validate an attempt-bound CUDA OOM without requiring success metrics.

    ``calibration_eligible`` means that an attempt has a complete positive-step
    measurement fingerprint.  A CUDA OOM before the first measured step cannot
    satisfy that success-only contract, but it is still valid right-censored
    memory evidence when the terminal classification is bound to the exact
    attempt and hashed log.
    """

    if result.get("classification") != "oom":
        return {"required": False, "all_passed": True, "checks": {}}

    job_id = str(job["job_id"])
    job_root = results_root / job_id
    status_path = Path(str(result.get("status_path") or ""))
    latest_path = job_root / "latest_attempt.json"
    status: dict[str, Any] = {}
    latest: dict[str, Any] = {}
    if status_path.is_file():
        status = read_json(status_path)
    if latest_path.is_file():
        latest = read_json(latest_path)
    attempt_id = str(status.get("execution_attempt_id") or "")
    evidence = status.get("classification_evidence") or {}
    log_relative = str(evidence.get("log_path") or "")
    log_path = status_path.parent / log_relative if log_relative else Path()
    log_is_attempt_local = bool(
        log_relative and log_path.resolve().parent == status_path.resolve().parent
    )
    log_hash_matches = bool(
        log_is_attempt_local
        and log_path.is_file()
        and str(evidence.get("log_sha256") or "") == sha256_file(log_path)
    )
    matched_patterns = evidence.get("matched_cuda_oom_patterns") or []
    checks = {
        "status_present": status_path.is_file(),
        "status_schema": status.get("schema") == "sft_execution_status/v2",
        "status_job_exact": status.get("job_id") == job_id,
        "status_classification_oom": status.get("classification") == "oom",
        "status_return_code_nonzero": isinstance(status.get("return_code"), int)
        and int(status["return_code"]) != 0,
        "attempt_id_present": bool(attempt_id),
        "latest_present": latest_path.is_file(),
        "latest_schema": latest.get("schema") == "sft_latest_attempt/v1",
        "latest_state_complete": latest.get("state") == "complete",
        "latest_job_exact": latest.get("job_id") == job_id,
        "latest_classification_oom": latest.get("classification") == "oom",
        "latest_attempt_exact": latest.get("execution_attempt_id") == attempt_id,
        "latest_attempt_path_exact": latest.get("attempt_path")
        == f"attempts/{attempt_id}",
        "classification_evidence_schema": evidence.get("schema")
        == "sft_terminal_classification/v1",
        "classification_evidence_job_exact": evidence.get("job_id") == job_id,
        "classification_evidence_attempt_exact": evidence.get("execution_attempt_id")
        == attempt_id,
        "classification_evidence_oom": evidence.get("classification") == "oom",
        "cuda_oom_confirmed": evidence.get("cuda_oom_confirmed") is True,
        "cuda_oom_pattern_present": isinstance(matched_patterns, list)
        and bool(matched_patterns),
        "evidence_return_code_matches": evidence.get("return_code")
        == status.get("return_code"),
        "summary_set_exact": evidence.get("summary_set_exact") is True,
        "log_is_attempt_local": log_is_attempt_local,
        "log_hash_matches": log_hash_matches,
    }
    return {
        "required": True,
        "all_passed": all(checks.values()),
        "checks": checks,
        "execution_attempt_id": attempt_id or None,
        "log_path": str(log_path.resolve()) if log_is_attempt_local else None,
        "log_sha256": evidence.get("log_sha256"),
    }


def _success_semantics(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    if result.get("classification") != "success":
        return {"required": False, "all_passed": True, "rank_checks": []}
    rank_checks = []
    for summary, structure in zip(
        result.get("summaries") or [], result.get("structures") or []
    ):
        media = (summary.get("runtime_batch_evidence") or {}).get("media") or {}
        phase = summary.get("vision_phase_memory_probe") or {}
        components = structure.get("components") or {}
        checks = {
            "freeze_declaration_matched": structure.get("declaration_status")
            == "matched",
            "vision_frozen": int(
                (components.get("vision_tower") or {}).get(
                    "trainable_parameter_elements"
                )
                or 0
            )
            == 0,
            "projector_frozen": int(
                (components.get("multimodal_projector") or {}).get(
                    "trainable_parameter_elements"
                )
                or 0
            )
            == 0,
            "language_adapter_trainable": int(
                (components.get("language_model") or {}).get(
                    "adapter_trainable_parameter_elements"
                )
                or 0
            )
            > 0,
        }
        if job["arm_id"] == "real_image":
            checks["real_media_path"] = media.get("real_image_path_observed") is True
            checks["vision_phase_observed"] = (
                phase.get("all_measured_steps_observed") is True
            )
        elif job["arm_id"] == "real_video":
            checks["real_media_path"] = media.get("real_video_path_observed") is True
            checks["vision_phase_observed"] = (
                phase.get("all_measured_steps_observed") is True
            )
        else:
            checks["no_runtime_media"] = (
                int(media.get("source_image_count") or 0) == 0
                and int(media.get("source_video_count") or 0) == 0
            )
        rank_checks.append(
            {
                "rank": summary.get("rank"),
                "checks": checks,
                "all_passed": all(checks.values()),
            }
        )
    return {
        "required": True,
        "all_passed": len(rank_checks) == int(job["gpu_count"])
        and all(row["all_passed"] for row in rank_checks),
        "rank_checks": rank_checks,
    }


def _repeat_cv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        metrics = row.get("metrics") or {}
        value = metrics.get("effective_tokens_per_second")
        if value is None or row["mechanism_id"] != "SAFE":
            continue
        key = (row["model_id"], row["arm_id"], str(row["media_tier"]))
        groups[key].append(float(value))
    outputs = []
    for key, values in sorted(groups.items()):
        if len(values) < 2:
            continue
        mean = statistics.fmean(values)
        outputs.append(
            {
                "model_id": key[0],
                "arm_id": key[1],
                "media_tier": key[2],
                "repeats": len(values),
                "effective_tokens_per_second": values,
                "coefficient_of_variation": statistics.stdev(values) / mean
                if mean > 0
                else None,
            }
        )
    return outputs


def evaluate() -> dict[str, Any]:
    jobs = read_jsonl(QUEUE)
    results = [_result(job) for job in jobs]
    device_capacity_bytes = int(read_json(HARDWARE)["memory_bytes_reported_by_torch"])
    rows = []
    for job, result in zip(jobs, results):
        semantics = _success_semantics(job, result)
        oom_semantics = _oom_terminal_semantics(job, result)
        classification = result["classification"]
        terminal_eligible = bool(
            result["terminal_eligible"]
            if classification == "success"
            else oom_semantics["all_passed"]
            if classification == "oom"
            else False
        )
        result_artifacts_complete = bool(
            result["result_artifacts_complete"]
            if classification == "success"
            else oom_semantics["all_passed"]
            if classification == "oom"
            else False
        )
        rows.append(
            {
                "job_id": job["job_id"],
                "track": job["track"],
                "model_id": job["model_id"],
                "model_family": job["model_family"],
                "arm_id": job["arm_id"],
                "media_tier": job.get("media_tier"),
                "mechanism_id": job["mechanism_id"],
                "repeat": job["repeat"],
                "classification": classification,
                "calibration_eligible": result.get("calibration_eligible"),
                "terminal_eligible": terminal_eligible,
                "result_artifacts_complete": result_artifacts_complete,
                "memory_observation_kind": (
                    "exact_success_center"
                    if classification == "success"
                    else "right_censored_oom"
                    if classification == "oom"
                    else "missing"
                ),
                "exact_center_target_bytes": (
                    (_success_metrics(result) or {}).get("max_reserved_bytes")
                    if classification == "success"
                    else None
                ),
                "right_censor_lower_bytes": (
                    device_capacity_bytes if classification == "oom" else None
                ),
                "oom_peak_is_unknown_not_imputed": classification == "oom",
                "metrics": _success_metrics(result),
                "success_semantics": semantics,
                "oom_terminal_semantics": oom_semantics,
                "status_path": result.get("status_path"),
                "status_sha256": result.get("status_sha256"),
            }
        )
    classifications = Counter(str(row["classification"]) for row in rows)
    successful_real_media = {
        (str(row["model_id"]), str(row["arm_id"]))
        for row in rows
        if row["classification"] == "success"
        and row["arm_id"] in {"real_image", "real_video"}
        and row["success_semantics"]["all_passed"]
    }
    expected_real_media = {
        (model_id, arm)
        for model_id in ("qwen2p5_vl_3b", "qwen3_vl_4b", "qwen3p5_4b")
        for arm in ("real_image", "real_video")
    }
    repeat_cv = _repeat_cv(rows)
    checks = {
        "queue_exact": len(jobs) == 87,
        "all_jobs_terminal_success_or_oom": all(
            row["terminal_eligible"] for row in rows
        ),
        "no_software_or_infrastructure_failure": set(classifications)
        <= {"success", "oom"},
        "all_success_artifacts_and_semantics_complete": all(
            row["classification"] != "success"
            or (
                row["result_artifacts_complete"]
                and row["success_semantics"]["all_passed"]
            )
            for row in rows
        ),
        "every_model_has_real_image_and_video_success": successful_real_media
        == expected_real_media,
        "all_repeat_cv_at_most_5_percent": bool(repeat_cv)
        and all(
            row["coefficient_of_variation"] is not None
            and row["coefficient_of_variation"] <= 0.05
            for row in repeat_cv
        ),
    }
    report: dict[str, Any] = {
        "schema": "sft_h800_frozen_vl_combined_formal_results/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "queue": {"path": str(QUEUE.resolve()), "sha256": sha256_file(QUEUE)},
        "hardware": {
            "path": str(HARDWARE.resolve()),
            "sha256": sha256_file(HARDWARE),
            "device_capacity_bytes": device_capacity_bytes,
        },
        "checks": checks,
        "all_passed": all(checks.values()),
        "classifications": dict(sorted(classifications.items())),
        "repeat_cv": repeat_cv,
        "rows": rows,
        "fit_allowed": all(checks.values()),
        "oom_policy": "right_censored_lower_bound_never_exact_peak",
        "repeat_policy": "collapse_physical_repeats_before_fit",
        "acceptance_allowed": False,
        "publication_allowed": False,
        "next_required_stage": "fit VL residual heads then run a source-disjoint prospective holdout",
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    return report


def main() -> None:
    report = evaluate()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["all_passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
