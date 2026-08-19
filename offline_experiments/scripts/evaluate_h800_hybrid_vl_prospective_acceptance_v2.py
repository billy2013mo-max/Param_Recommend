#!/usr/bin/env python3
"""Evaluate frozen hybrid/VL safety bounds and ranking on prospective outcomes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, RESULTS_DIR, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_hybrid_vl_prospective_acceptance_v2 import (
    CAMPAIGN_ID,
    COMBINED_QUEUE,
    DESIGN,
    FROZEN_PREDICTIONS,
    HYBRID_QUEUE,
    PHASE_ID,
    VL_QUEUE,
)


OUTPUT = ARTIFACT_DIR / "h800_hybrid_vl_prospective_acceptance_report_v2.json"
EXPECTED_INFORMATIVE_RANKING_GROUPS = {"hybrid": 3, "vl_image": 6}


def _internal_hash_valid(payload: dict[str, Any]) -> bool:
    unsigned = dict(payload)
    expected = unsigned.pop("report_sha256", None)
    return expected == sha256_json(unsigned)


def _rank_summaries(job: dict[str, Any], status: dict[str, Any]) -> list[dict[str, Any]]:
    attempt_id = str(status.get("execution_attempt_id") or "")
    metrics = RESULTS_DIR / str(job["job_id"]) / "attempts" / attempt_id / "metrics"
    rows = []
    for rank in range(int(job["gpu_count"])):
        path = metrics / f"summary.rank{rank}.json"
        if path.is_file():
            rows.append(read_json(path))
    return rows


def _visual_semantics(summary: dict[str, Any]) -> bool:
    media = (summary.get("runtime_batch_evidence") or {}).get("media") or {}
    structure = summary.get("runtime_structure_manifest") or {}
    probe = summary.get("vision_phase_memory_probe") or {}
    return bool(
        media.get("real_image_path_observed") is True
        and int(media.get("source_image_count") or 0) > 0
        and int(media.get("pixel_value_elements") or 0) > 0
        and structure.get("visual_path_observed") is True
        and int(probe.get("measured_steps_with_visual_forward") or 0) > 0
    )


def _observed(job: dict[str, Any]) -> dict[str, Any]:
    status_path = RESULTS_DIR / str(job["job_id"]) / "status.json"
    if not status_path.is_file():
        return {
            "classification": "missing",
            "terminal_valid": False,
            "visual_semantics_passed": False,
        }
    status = read_json(status_path)
    classification = str(status.get("classification") or "unknown")
    terminal_valid = bool(
        status.get("job_id") == job["job_id"]
        and classification in {"success", "oom"}
        and status.get("calibration_eligible") is True
        and status.get("execution_fingerprint_quality") == "complete"
    )
    summaries = _rank_summaries(job, status)
    success_complete = classification != "success" or len(summaries) == int(job["gpu_count"])
    observed_reserved = None
    effective_tps = None
    if classification == "success" and success_complete:
        observed_reserved = max(float(row.get("max_reserved") or 0.0) for row in summaries)
        seconds = max(float(row.get("measured_seconds") or 0.0) for row in summaries)
        effective_tokens = sum(
            float((row.get("measured_totals") or {}).get("effective_tokens") or 0.0)
            for row in summaries
        )
        effective_tps = effective_tokens / seconds if seconds > 0.0 and effective_tokens > 0.0 else None
    return {
        "classification": classification,
        "terminal_valid": terminal_valid and success_complete,
        "status_path": str(status_path.resolve()),
        "status_sha256": sha256_file(status_path),
        "execution_attempt_id": status.get("execution_attempt_id"),
        "observed_reserved_bytes": observed_reserved,
        "effective_tokens_per_second": effective_tps,
        "rank_summary_count": len(summaries),
        "visual_semantics_passed": (
            len(summaries) == int(job["gpu_count"])
            and all(_visual_semantics(row) for row in summaries)
            if str(job.get("design_arm") or "").startswith("vl_image_prospective") and classification == "success"
            else not str(job.get("design_arm") or "").startswith("vl_image_prospective") or classification == "oom"
        ),
    }


def _frozen_maps() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    frozen = read_json(FROZEN_PREDICTIONS)
    predictions = {
        str(row["request_id"]): row
        for section in ("hybrid", "vl_image")
        for row in frozen[section]["predictions"]
    }
    hybrid_throughput = {
        str(row["request_id"]): row
        for row in frozen["hybrid"]["throughput_predictions"]
    }
    return predictions, hybrid_throughput


def _join_rows() -> list[dict[str, Any]]:
    predictions, hybrid_throughput = _frozen_maps()
    joined = []
    for job in read_jsonl(COMBINED_QUEUE):
        job_id = str(job["job_id"])
        prediction = predictions[job_id]
        observed = _observed(job)
        track = "vl_image" if str(job.get("design_arm") or "").startswith("vl_image_prospective") else "hybrid"
        if track == "vl_image":
            shadow = prediction.get("vl_shadow") or {}
            memory = shadow.get("memory") or {}
            throughput = shadow.get("throughput") or {}
            upper = memory.get("safety_upper_bytes")
            safe_limit = memory.get("safe_limit_bytes")
            predicted_tps = throughput.get("effective_tokens_per_second")
        else:
            memory = prediction.get("memory") or {}
            upper = memory.get("admission_upper_reserved_bytes")
            safe_limit = memory.get("safe_limit_bytes")
            predicted_tps = hybrid_throughput[job_id].get("predicted_effective_tokens_per_second")
        if upper is None or safe_limit is None or predicted_tps is None:
            raise ValueError(f"frozen safety/ranking prediction is incomplete: {job_id}")
        joined.append(
            {
                "job_id": job_id,
                "track": track,
                "model_id": job["model_id"],
                "comparison_group": prediction["comparison_group"],
                "mechanism": job.get("mechanism_id") or job.get("parallel_route_id"),
                "predicted_upper_bytes": float(upper),
                "safe_limit_bytes": float(safe_limit),
                "predicted_admitted": float(upper) <= float(safe_limit),
                "predicted_effective_tokens_per_second": float(predicted_tps),
                **observed,
            }
        )
    return joined


def _memory_metrics(rows: list[dict[str, Any]], threshold: dict[str, Any]) -> dict[str, Any]:
    success = [row for row in rows if row["classification"] == "success"]
    oom = [row for row in rows if row["classification"] == "oom"]
    covered = [
        row for row in success
        if row["observed_reserved_bytes"] is not None
        and float(row["observed_reserved_bytes"]) <= row["predicted_upper_bytes"]
    ]
    actual_safe = [
        row for row in success
        if row["observed_reserved_bytes"] is not None
        and float(row["observed_reserved_bytes"]) <= row["safe_limit_bytes"]
    ]
    admitted_safe = [row for row in actual_safe if row["predicted_admitted"]]
    false_safe = [row for row in oom if row["predicted_admitted"]]
    coverage = len(covered) / len(success) if success else None
    recall = len(admitted_safe) / len(actual_safe) if actual_safe else None
    checks = {
        "has_exact_success": bool(success),
        "false_safe_oom_zero": len(false_safe) <= int(threshold["false_safe_oom"]),
        "exact_upper_coverage": coverage is not None and coverage >= float(threshold["exact_upper_coverage_min"]),
        "admission_recall": recall is not None and recall >= float(threshold["admission_recall_min"]),
    }
    return {
        "success_rows": len(success),
        "oom_rows": len(oom),
        "covered_success_rows": len(covered),
        "exact_upper_coverage": coverage,
        "actual_safe_success_rows": len(actual_safe),
        "admitted_safe_success_rows": len(admitted_safe),
        "admission_recall": recall,
        "false_safe_oom": len(false_safe),
        "false_safe_oom_job_ids": [row["job_id"] for row in false_safe],
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def _ranking_metrics(track: str, rows: list[dict[str, Any]], threshold: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["comparison_group"])].append(row)
    details = []
    for group_id, candidates in sorted(groups.items()):
        actual_safe = [
            row for row in candidates
            if row["classification"] == "success"
            and row["observed_reserved_bytes"] is not None
            and float(row["observed_reserved_bytes"]) <= row["safe_limit_bytes"]
            and row["effective_tokens_per_second"] is not None
        ]
        predicted_pool = [row for row in candidates if row["predicted_admitted"]]
        selected = max(predicted_pool, key=lambda row: row["predicted_effective_tokens_per_second"]) if predicted_pool else None
        observed_best = max(actual_safe, key=lambda row: row["effective_tokens_per_second"]) if actual_safe else None
        selected_observed = (
            float(selected["effective_tokens_per_second"])
            if selected is not None and selected["classification"] == "success" and selected["effective_tokens_per_second"] is not None
            else 0.0
        )
        best_observed = float(observed_best["effective_tokens_per_second"]) if observed_best else 0.0
        regret = 1.0 - selected_observed / best_observed if best_observed > 0.0 else None
        informative = len(actual_safe) >= 2
        details.append(
            {
                "comparison_group": group_id,
                "model_id": candidates[0]["model_id"],
                "candidate_count": len(candidates),
                "actual_safe_candidate_count": len(actual_safe),
                "predicted_admitted_candidate_count": len(predicted_pool),
                "informative": informative,
                "predicted_winner": selected["job_id"] if selected else None,
                "observed_best": observed_best["job_id"] if observed_best else None,
                "top1_regret": regret,
                "top1_within_10_percent": bool(regret is not None and regret <= float(threshold["worst_top1_regret_max"])),
            }
        )
    informative = [row for row in details if row["informative"]]
    regrets = [float(row["top1_regret"]) for row in informative if row["top1_regret"] is not None]
    hit = sum(row["top1_within_10_percent"] for row in informative) / len(informative) if informative else None
    worst = max(regrets) if regrets else None
    checks = {
        "informative_group_count": len(informative) >= EXPECTED_INFORMATIVE_RANKING_GROUPS[track],
        "hit_at_10_percent": hit is not None and hit >= float(threshold["ranking_hit_at_10_percent"]),
        "worst_top1_regret": worst is not None and worst <= float(threshold["worst_top1_regret_max"]),
    }
    passing_models = sorted(
        model_id
        for model_id in {str(row["model_id"]) for row in details}
        if (model_details := [row for row in details if row["model_id"] == model_id])
        and all(row["informative"] and row["top1_within_10_percent"] for row in model_details)
    )
    return {
        "groups": details,
        "informative_groups": len(informative),
        "hit_at_10_percent": hit,
        "worst_top1_regret": worst,
        "passing_model_ids": passing_models,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def evaluate(*, allow_incomplete: bool = False) -> dict[str, Any]:
    design = read_json(DESIGN)
    frozen = read_json(FROZEN_PREDICTIONS)
    rows = _join_rows()
    thresholds = design["acceptance_thresholds"]
    classifications = Counter(str(row["classification"]) for row in rows)
    input_checks = {
        "queue_exact_27": len(rows) == 27 and read_jsonl(COMBINED_QUEUE) == [*read_jsonl(HYBRID_QUEUE), *read_jsonl(VL_QUEUE)],
        "design_identity": design.get("campaign_id") == CAMPAIGN_ID and design.get("phase_id") == PHASE_ID,
        "design_internal_hash": _internal_hash_valid(design),
        "frozen_predictions_internal_hash": _internal_hash_valid(frozen),
        "frozen_predictions_bound": design.get("frozen_predictions", {}).get("sha256") == sha256_file(FROZEN_PREDICTIONS),
        "all_terminal_valid": all(row["terminal_valid"] for row in rows),
        "only_success_or_oom": set(classifications) <= {"success", "oom"},
    }
    complete = input_checks["all_terminal_valid"] and input_checks["only_success_or_oom"]
    if not complete and not allow_incomplete:
        missing = [row["job_id"] for row in rows if not row["terminal_valid"]]
        raise RuntimeError(f"prospective acceptance is incomplete: {missing}")

    tracks = {}
    for track in ("hybrid", "vl_image"):
        subset = [row for row in rows if row["track"] == track]
        memory = _memory_metrics(subset, thresholds)
        ranking = _ranking_metrics(track, subset, thresholds)
        visual_passed = track != "vl_image" or all(row["visual_semantics_passed"] for row in subset)
        tracks[track] = {
            "jobs": len(subset),
            "classifications": dict(Counter(str(row["classification"]) for row in subset)),
            "memory": memory,
            "ranking": ranking,
            "visual_semantics_passed": visual_passed,
            "automatic_admission_allowed": complete and memory["all_passed"] and visual_passed,
            "automatic_ranking_allowed": complete and memory["all_passed"] and ranking["all_passed"] and visual_passed,
        }
    report: dict[str, Any] = {
        "schema": "sft_h800_hybrid_vl_prospective_acceptance_report/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "usage": "prospective_acceptance_only_not_refit",
        "input_checks": input_checks,
        "complete": complete,
        "classifications": dict(classifications),
        "thresholds": thresholds,
        "tracks": tracks,
        "release_scope": {
            "hybrid_non_packing_automatic_admission": tracks["hybrid"]["automatic_admission_allowed"],
            "hybrid_non_packing_automatic_ranking": tracks["hybrid"]["automatic_ranking_allowed"],
            "vl_image_non_packing_automatic_admission": tracks["vl_image"]["automatic_admission_allowed"],
            "vl_image_non_packing_automatic_ranking": tracks["vl_image"]["automatic_ranking_allowed"],
            "vl_video_automatic_recommendation": False,
            "packing_automatic_recommendation": False,
        },
        "rows": rows,
    }
    report["all_requested_gates_passed"] = all(
        value for key, value in report["release_scope"].items()
        if key not in {"vl_video_automatic_recommendation", "packing_automatic_recommendation"}
    )
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(allow_incomplete=args.allow_incomplete), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
