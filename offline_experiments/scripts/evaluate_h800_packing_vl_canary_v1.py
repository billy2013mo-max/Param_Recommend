#!/usr/bin/env python3
"""Evaluate staged Packing and real-image VL semantic canaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, RESULTS_DIR, read_json, read_jsonl, sha256_file, sha256_json, write_json
from model_structure_manifest import validate_model_structure_manifest


CAMPAIGN_ID = "h800_packing_vl_canary_20260803_v1"
PACKING_QUEUE = MATRIX_DIR / "h800_packing_semantic_canary_v1.jsonl"
VL_QUEUE = MATRIX_DIR / "h800_vl_media_canary_v1.jsonl"
PACKING_OUTPUT = ARTIFACT_DIR / "h800_packing_semantic_canary_acceptance_v1.json"
VL_OUTPUT = ARTIFACT_DIR / "h800_vl_media_canary_acceptance_v1.json"
COMBINED_OUTPUT = ARTIFACT_DIR / "h800_packing_vl_canary_acceptance_v1.json"


def _result(job: dict[str, Any]) -> dict[str, Any]:
    status_path = RESULTS_DIR / str(job["job_id"]) / "status.json"
    if not status_path.is_file():
        return {
            "job_id": job["job_id"],
            "status_present": False,
            "classification": None,
            "success": False,
            "reason": "status_absent",
        }
    status = read_json(status_path)
    attempt_id = status.get("execution_attempt_id")
    metrics_dir = RESULTS_DIR / str(job["job_id"]) / "attempts" / str(attempt_id) / "metrics"
    summaries = []
    structures = []
    for rank in range(int(job["gpu_count"])):
        summary_path = metrics_dir / f"summary.rank{rank}.json"
        structure_path = metrics_dir / f"model_structure_manifest.{attempt_id}.rank{rank}.json"
        if summary_path.is_file():
            summaries.append(read_json(summary_path))
        if structure_path.is_file():
            structures.append(validate_model_structure_manifest(read_json(structure_path)))
    success = bool(
        status.get("job_id") == job["job_id"]
        and status.get("classification") == "success"
        and status.get("calibration_eligible") is True
        and len(summaries) == int(job["gpu_count"])
        and len(structures) == int(job["gpu_count"])
    )
    return {
        "job_id": job["job_id"],
        "status_present": True,
        "status_path": str(status_path.resolve()),
        "status_sha256": sha256_file(status_path),
        "execution_attempt_id": attempt_id,
        "classification": status.get("classification"),
        "calibration_eligible": status.get("calibration_eligible"),
        "execution_fingerprint_quality": status.get("execution_fingerprint_quality"),
        "success": success,
        "summaries": summaries,
        "structures": structures,
    }


def evaluate_packing() -> dict[str, Any]:
    jobs = read_jsonl(PACKING_QUEUE)
    rows = [_result(job) for job in jobs]
    by_treatment = {
        str(job["packing_treatment"]): result for job, result in zip(jobs, rows)
    }
    checks: dict[str, bool] = {
        "queue_exact": len(jobs) == 2
        and {job.get("packing_treatment") for job in jobs} == {"unpacked", "neat_packed"}
        and all(job.get("campaign_id") == CAMPAIGN_ID for job in jobs),
        "both_jobs_success": all(row["success"] for row in rows),
    }
    packed = by_treatment.get("neat_packed") or {}
    unpacked = by_treatment.get("unpacked") or {}
    packed_summaries = packed.get("summaries") or []
    unpacked_summaries = unpacked.get("summaries") or []
    packed_evidence = (
        (packed_summaries[0].get("runtime_batch_evidence") or {}).get("packing")
        if packed_summaries
        else {}
    ) or {}
    unpacked_evidence = (
        (unpacked_summaries[0].get("runtime_batch_evidence") or {}).get("packing")
        if unpacked_summaries
        else {}
    ) or {}
    checks.update(
        {
            "packed_semantics_passed": packed_evidence.get("semantic_checks_passed") is True,
            "packed_contains_multi_sample_packs": int(packed_evidence.get("multi_sample_features") or 0) > 0,
            "packed_violation_free": all(
                int(value) == 0 for value in (packed_evidence.get("violations") or {"missing": 1}).values()
            ),
            "unpacked_has_no_pack_features": int(unpacked_evidence.get("features") or 0) == 0,
        }
    )
    sample_gbs = {}
    for job, result in zip(jobs, rows):
        summaries = result.get("summaries") or []
        measured_steps = sum(int(summary.get("measured_steps") or 0) for summary in summaries)
        logical_samples = sum(
            int((summary.get("measured_totals") or {}).get("logical_samples") or 0)
            for summary in summaries
        )
        observed = logical_samples / measured_steps if measured_steps else None
        target = float(job["target_gbs"])
        error = abs(observed - target) / target if observed is not None else None
        sample_gbs[str(job["packing_treatment"])] = {
            "observed_samples_per_optimizer_step": observed,
            "target_gbs": target,
            "relative_error": error,
            "passed": error is not None and error <= 0.05,
        }
    checks["sample_gbs_error_at_most_5pct"] = all(
        row["passed"] for row in sample_gbs.values()
    )
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_semantic_canary_acceptance/v1",
        "campaign_id": CAMPAIGN_ID,
        "stage": "packing",
        "queue": {"path": str(PACKING_QUEUE.resolve()), "sha256": sha256_file(PACKING_QUEUE)},
        "checks": checks,
        "all_passed": all(checks.values()),
        "sample_gbs": sample_gbs,
        "runtime_packing_evidence": {
            "unpacked": unpacked_evidence,
            "neat_packed": packed_evidence,
        },
        "results": rows,
        "fit_allowed": False,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(PACKING_OUTPUT, report)
    return report


def evaluate_vl() -> dict[str, Any]:
    jobs = read_jsonl(VL_QUEUE)
    rows = [_result(job) for job in jobs]
    media_checks = []
    for job, result in zip(jobs, rows):
        summaries = result.get("summaries") or []
        structures = result.get("structures") or []
        rank_checks = []
        for summary, structure in zip(summaries, structures):
            media = (summary.get("runtime_batch_evidence") or {}).get("media") or {}
            freeze = structure.get("freeze_flags") or {}
            declared_observed = structure.get("declaration_observed_flags") or {}
            rank_checks.append(
                {
                    "rank": summary.get("rank"),
                    "real_image_path_observed": media.get("real_image_path_observed") is True,
                    "source_images_positive": int(media.get("source_image_count") or 0) > 0,
                    "image_grid_rows_positive": int(media.get("image_grid_rows") or 0) > 0,
                    "pixel_values_positive": int(media.get("pixel_value_elements") or 0) > 0,
                    "final_structure_visual_path_observed": structure.get("visual_path_observed") is True,
                    "vision_parameters_observed": structure.get("vision_parameters_observed") is True,
                    "freeze_declaration_matched": structure.get("declaration_status") == "matched",
                    "vision_tower_frozen": (freeze.get("freeze_vision_tower") or {}).get("value") is True,
                    "projector_frozen": (freeze.get("freeze_multi_modal_projector") or {}).get("value") is True,
                    "language_side_lora_enabled": declared_observed.get("freeze_language_model") is False,
                    "no_visual_lora_target_hit": (structure.get("lora_target_hits") or {}).get("any_visual_component") is False,
                }
            )
        media_checks.append(
            {
                "job_id": job["job_id"],
                "model_id": job["model_id"],
                "rank_checks": rank_checks,
                "all_ranks_passed": len(rank_checks) == int(job["gpu_count"])
                and all(all(value for key, value in check.items() if key != "rank") for check in rank_checks),
            }
        )
    checks = {
        "queue_exact": len(jobs) == 2
        and {job.get("model_id") for job in jobs} == {"qwen2p5_vl_7b", "qwen3_vl_8b"}
        and all(job.get("campaign_id") == CAMPAIGN_ID for job in jobs),
        "both_jobs_success": all(row["success"] for row in rows),
        "both_models_all_ranks_media_and_freeze_passed": len(media_checks) == 2
        and all(row["all_ranks_passed"] for row in media_checks),
    }
    report: dict[str, Any] = {
        "schema": "sft_h800_vl_media_canary_acceptance/v1",
        "campaign_id": CAMPAIGN_ID,
        "stage": "vl",
        "queue": {"path": str(VL_QUEUE.resolve()), "sha256": sha256_file(VL_QUEUE)},
        "checks": checks,
        "all_passed": all(checks.values()),
        "media_and_freeze_checks": media_checks,
        "results": rows,
        "fit_allowed": False,
        "recommendation_release_allowed": False,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(VL_OUTPUT, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("packing", "vl", "all"), required=True)
    args = parser.parse_args()
    packing = evaluate_packing()
    vl = evaluate_vl() if args.stage in {"vl", "all"} else None
    combined: dict[str, Any] = {
        "schema": "sft_h800_packing_vl_canary_acceptance/v1",
        "campaign_id": CAMPAIGN_ID,
        "packing": {
            "path": str(PACKING_OUTPUT.resolve()),
            "sha256": sha256_file(PACKING_OUTPUT),
            "all_passed": packing["all_passed"],
        },
        "vl": (
            {
                "path": str(VL_OUTPUT.resolve()),
                "sha256": sha256_file(VL_OUTPUT),
                "all_passed": vl["all_passed"],
            }
            if vl is not None
            else None
        ),
        "all_passed": bool(packing["all_passed"] and vl is not None and vl["all_passed"]),
        "next_stage_allowed": "vl" if packing["all_passed"] and vl is None else "calibration_design" if packing["all_passed"] and vl and vl["all_passed"] else None,
    }
    combined["report_sha256"] = sha256_json(combined)
    write_json(COMBINED_OUTPUT, combined)
    print(__import__("json").dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
