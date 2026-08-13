#!/usr/bin/env python3
"""Evaluate semantic canaries or summarize formal Qwen3.5/VL evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, RESULTS_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from model_structure_manifest import validate_model_structure_manifest
from prepare_h800_qwen35_vl_supplement_v1 import (
    CAMPAIGN_ID,
    CANARY_PHASE_ID,
    CANARY_QUEUE,
    FORMAL_PHASE_ID,
    FORMAL_QUEUE,
)


CANARY_OUTPUT = ARTIFACT_DIR / "h800_qwen35_vl_supplement_canary_acceptance_v1.json"
FORMAL_OUTPUT = ARTIFACT_DIR / "h800_qwen35_vl_supplement_results_v1.json"
FORMAL_MARKDOWN = ARTIFACT_DIR / "h800_qwen35_vl_supplement_results_v1.md"


def _rank_artifacts(job: dict[str, Any], status: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    attempt_id = str(status.get("execution_attempt_id") or "")
    metrics_dir = RESULTS_DIR / str(job["job_id"]) / "attempts" / attempt_id / "metrics"
    summaries = []
    structures = []
    bindings = []
    for rank in range(int(job["gpu_count"])):
        summary_path = metrics_dir / f"summary.rank{rank}.json"
        structure_path = metrics_dir / f"model_structure_manifest.{attempt_id}.rank{rank}.json"
        if summary_path.is_file():
            summaries.append(read_json(summary_path))
        if structure_path.is_file():
            structures.append(validate_model_structure_manifest(read_json(structure_path)))
        bindings.append(
            {
                "rank": rank,
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": sha256_file(summary_path) if summary_path.is_file() else None,
                "structure_path": str(structure_path.resolve()),
                "structure_sha256": sha256_file(structure_path) if structure_path.is_file() else None,
            }
        )
    return summaries, structures, bindings


def _result(job: dict[str, Any]) -> dict[str, Any]:
    status_path = RESULTS_DIR / str(job["job_id"]) / "status.json"
    if not status_path.is_file():
        return {
            "job_id": job["job_id"],
            "classification": "missing",
            "calibration_eligible": False,
            "status_present": False,
            "terminal_eligible": False,
            "result_artifacts_complete": False,
        }
    status = read_json(status_path)
    summaries, structures, bindings = _rank_artifacts(job, status)
    classification = status.get("classification")
    terminal = bool(
        status.get("job_id") == job["job_id"]
        and classification in {"success", "oom"}
        and status.get("calibration_eligible") is True
    )
    success_complete = bool(
        classification == "success"
        and len(summaries) == int(job["gpu_count"])
        and len(structures) == int(job["gpu_count"])
    )
    return {
        "job_id": job["job_id"],
        "status_present": True,
        "status_path": str(status_path.resolve()),
        "status_sha256": sha256_file(status_path),
        "execution_attempt_id": status.get("execution_attempt_id"),
        "classification": classification,
        "calibration_eligible": status.get("calibration_eligible"),
        "execution_fingerprint_quality": status.get("execution_fingerprint_quality"),
        "terminal_eligible": terminal,
        "result_artifacts_complete": success_complete if classification == "success" else terminal,
        "summaries": summaries,
        "structures": structures,
        "artifact_bindings": bindings,
    }


def _media_scope_checks(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    rank_checks = []
    declared_scope = str(job["train_scope_id"])
    expected = {
        "language_only": {"vision_trainable": False, "projector_trainable": False},
        "projector_plus_language": {"vision_trainable": False, "projector_trainable": True},
        "all_visual_plus_language": {"vision_trainable": True, "projector_trainable": True},
    }[declared_scope]
    for summary, structure in zip(result.get("summaries") or [], result.get("structures") or []):
        media = (summary.get("runtime_batch_evidence") or {}).get("media") or {}
        freeze = structure.get("freeze_flags") or {}
        components = structure.get("components") or {}
        declared_observed = structure.get("declaration_observed_flags") or {}
        vision_trainable = int((components.get("vision_tower") or {}).get("trainable_parameter_elements") or 0) > 0
        projector_trainable = int((components.get("multimodal_projector") or {}).get("trainable_parameter_elements") or 0) > 0
        checks = {
            "real_image_path_observed": media.get("real_image_path_observed") is True,
            "source_images_positive": int(media.get("source_image_count") or 0) > 0,
            "image_grid_rows_positive": int(media.get("image_grid_rows") or 0) > 0,
            "pixel_values_positive": int(media.get("pixel_value_elements") or 0) > 0,
            "visual_structure_observed": structure.get("visual_path_observed") is True,
            "vision_parameters_observed": structure.get("vision_parameters_observed") is True,
            "freeze_declaration_matched": structure.get("declaration_status") == "matched",
            "declared_vision_flag_exact": declared_observed.get("freeze_vision_tower")
            is job["freeze_vision_tower"],
            "declared_projector_flag_exact": declared_observed.get("freeze_multi_modal_projector")
            is job["freeze_multi_modal_projector"],
            "declared_language_flag_exact": declared_observed.get("freeze_language_model")
            is job["freeze_language_model"],
            "vision_trainability_exact": vision_trainable is expected["vision_trainable"],
            "projector_trainability_exact": projector_trainable is expected["projector_trainable"],
            "language_adapter_trainable": int((components.get("language_model") or {}).get("adapter_trainable_parameter_elements") or 0) > 0,
        }
        if declared_scope == "language_only":
            checks["no_visual_lora_target_hit"] = (
                (structure.get("lora_target_hits") or {}).get("any_visual_component") is False
            )
            checks["vision_freeze_marker_true"] = (
                (freeze.get("freeze_vision_tower") or {}).get("value") is True
            )
            checks["projector_freeze_marker_true"] = (
                (freeze.get("freeze_multi_modal_projector") or {}).get("value") is True
            )
        rank_checks.append(
            {
                "rank": summary.get("rank"),
                "checks": checks,
                "all_passed": all(checks.values()),
            }
        )
    return {
        "job_id": job["job_id"],
        "model_id": job["model_id"],
        "train_scope_id": declared_scope,
        "rank_checks": rank_checks,
        "all_ranks_passed": len(rank_checks) == int(job["gpu_count"])
        and all(row["all_passed"] for row in rank_checks),
    }


def evaluate_canary() -> dict[str, Any]:
    jobs = read_jsonl(CANARY_QUEUE)
    results = [_result(job) for job in jobs]
    semantic = [
        _media_scope_checks(job, result) for job, result in zip(jobs, results)
    ]
    model_ids = {str(job["model_id"]) for job in jobs}
    checks = {
        "queue_exact": len(jobs) == 15
        and all(job.get("campaign_id") == CAMPAIGN_ID for job in jobs)
        and all(job.get("phase_id") == CANARY_PHASE_ID for job in jobs),
        "eleven_model_endpoints_covered": len(model_ids) == 11,
        "four_train_scope_ablation_canaries_present": sum(
            job.get("train_scope_id") != "language_only" for job in jobs
        )
        == 4,
        "all_jobs_success_with_complete_artifacts": all(
            row["classification"] == "success"
            and row["terminal_eligible"]
            and row["result_artifacts_complete"]
            for row in results
        ),
        "all_real_image_and_scope_semantics_passed": len(semantic) == 15
        and all(row["all_ranks_passed"] for row in semantic),
    }
    report: dict[str, Any] = {
        "schema": "sft_h800_qwen35_vl_supplement_canary_acceptance/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": CANARY_PHASE_ID,
        "queue": {"path": str(CANARY_QUEUE.resolve()), "sha256": sha256_file(CANARY_QUEUE)},
        "checks": checks,
        "all_passed": all(checks.values()),
        "media_and_train_scope_checks": semantic,
        "results": results,
        "fit_allowed": False,
        "formal_stage_allowed": all(checks.values()),
        "recommendation_release_allowed": False,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(CANARY_OUTPUT, report)
    return report


def _success_metrics(result: dict[str, Any]) -> dict[str, Any] | None:
    summaries = result.get("summaries") or []
    if result.get("classification") != "success" or not summaries:
        return None
    max_reserved = max(int(row.get("max_reserved") or 0) for row in summaries)
    max_allocated = max(int(row.get("max_allocated") or 0) for row in summaries)
    measured_seconds = max(float(row.get("measured_seconds") or 0.0) for row in summaries)
    effective_tokens = sum(
        int((row.get("measured_totals") or {}).get("effective_tokens") or 0)
        for row in summaries
    )
    computed_tokens = sum(
        int((row.get("measured_totals") or {}).get("computed_tokens") or 0)
        for row in summaries
    )
    return {
        "max_reserved_bytes": max_reserved,
        "max_allocated_bytes": max_allocated,
        "measured_seconds": measured_seconds,
        "effective_tokens": effective_tokens,
        "computed_tokens": computed_tokens,
        "effective_tokens_per_second": (
            effective_tokens / measured_seconds if measured_seconds > 0 else None
        ),
        "computed_tokens_per_second": (
            computed_tokens / measured_seconds if measured_seconds > 0 else None
        ),
    }


def evaluate_formal(*, allow_incomplete: bool = False) -> dict[str, Any]:
    jobs = read_jsonl(FORMAL_QUEUE)
    results = [_result(job) for job in jobs]
    rows = []
    for job, result in zip(jobs, results):
        rows.append(
            {
                "job_id": job["job_id"],
                "track": job["track"],
                "evidence_role": job["evidence_role"],
                "scenario_id": job["scenario_id"],
                "model_id": job["model_id"],
                "model_family": job["model_family"],
                "train_type": job["train_type"],
                "train_scope_id": job["train_scope_id"],
                "media_tier": job.get("media_tier"),
                "mechanism_id": job["mechanism_id"],
                "gpu_count": job["gpu_count"],
                "zero": job["zero"],
                "gc": job["gc"],
                "mbs": job["mbs"],
                "repeat": job["repeat"],
                "classification": result["classification"],
                "calibration_eligible": result["calibration_eligible"],
                "terminal_eligible": result["terminal_eligible"],
                "result_artifacts_complete": result["result_artifacts_complete"],
                "metrics": _success_metrics(result),
                "status_path": result.get("status_path"),
                "status_sha256": result.get("status_sha256"),
                "artifact_bindings": result.get("artifact_bindings") or [],
            }
        )
    classifications = Counter(str(row["classification"]) for row in rows)
    by_track: dict[str, Counter[str]] = defaultdict(Counter)
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_track[str(row["track"])][str(row["classification"])] += 1
        by_model[str(row["model_id"])][str(row["classification"])] += 1
    text_success_models = {
        str(row["model_id"])
        for row in rows
        if row["track"] == "qwen35_text_cross_scale"
        and row["classification"] == "success"
    }
    required_text_models = {
        str(row["model_id"])
        for row in rows
        if row["track"] == "qwen35_text_cross_scale"
    }
    required_image_groups = {
        (str(row["model_id"]), str(row["media_tier"]))
        for row in rows
        if row["media_tier"] is not None
        and row["train_scope_id"] == "language_only"
        and row["track"] != "repeat_variance_diagnostic"
    }
    image_success_groups = {
        (str(row["model_id"]), str(row["media_tier"]))
        for row in rows
        if row["classification"] == "success"
        and row["media_tier"] is not None
        and row["train_scope_id"] == "language_only"
        and row["track"] != "repeat_variance_diagnostic"
    }
    required_scope_groups = {
        (str(row["model_id"]), str(row["train_scope_id"]))
        for row in rows
        if row["track"] == "visual_train_scope_ablation"
    }
    scope_success_groups = {
        (str(row["model_id"]), str(row["train_scope_id"]))
        for row in rows
        if row["track"] == "visual_train_scope_ablation"
        and row["classification"] == "success"
    }
    repeat_rows = [row for row in rows if row["track"] == "repeat_variance_diagnostic"]
    terminal = all(row["terminal_eligible"] for row in rows)
    complete_artifacts = all(row["result_artifacts_complete"] for row in rows)
    checks = {
        "queue_exact": len(jobs) == 86
        and all(job.get("campaign_id") == CAMPAIGN_ID for job in jobs)
        and all(job.get("phase_id") == FORMAL_PHASE_ID for job in jobs),
        "all_jobs_terminal_success_or_oom": terminal,
        "all_success_artifacts_complete": complete_artifacts,
        "no_software_or_infrastructure_failure": set(classifications)
        <= {"success", "oom"},
        "at_least_one_exact_success": classifications.get("success", 0) > 0,
        "every_qwen35_text_scale_has_success": text_success_models
        == required_text_models,
        "every_language_only_model_tier_has_success": image_success_groups
        == required_image_groups,
        "every_visual_scope_ablation_has_success": scope_success_groups
        == required_scope_groups,
        "all_six_repeat_diagnostics_success": len(repeat_rows) == 6
        and all(row["classification"] == "success" for row in repeat_rows),
    }
    all_passed = all(checks.values())
    report: dict[str, Any] = {
        "schema": "sft_h800_qwen35_vl_supplement_results/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": FORMAL_PHASE_ID,
        "queue": {"path": str(FORMAL_QUEUE.resolve()), "sha256": sha256_file(FORMAL_QUEUE)},
        "checks": checks,
        "all_passed": all_passed,
        "allow_incomplete": allow_incomplete,
        "classifications": dict(sorted(classifications.items())),
        "by_track": {key: dict(sorted(value.items())) for key, value in sorted(by_track.items())},
        "by_model": {key: dict(sorted(value.items())) for key, value in sorted(by_model.items())},
        "success_coverage": {
            "qwen35_text_models": {
                "required": sorted(required_text_models),
                "observed": sorted(text_success_models),
                "missing": sorted(required_text_models - text_success_models),
            },
            "language_only_model_tiers": {
                "required": sorted([list(value) for value in required_image_groups]),
                "observed": sorted([list(value) for value in image_success_groups]),
                "missing": sorted([list(value) for value in required_image_groups - image_success_groups]),
            },
            "visual_scope_groups": {
                "required": sorted([list(value) for value in required_scope_groups]),
                "observed": sorted([list(value) for value in scope_success_groups]),
                "missing": sorted([list(value) for value in required_scope_groups - scope_success_groups]),
            },
            "repeat_successes": sum(row["classification"] == "success" for row in repeat_rows),
        },
        "rows": rows,
        "fit_allowed": all_passed,
        "fit_policy": {
            "success": "exact point observation",
            "oom": "right-censored lower bound",
            "repeat": "collapse by physical arm before fitting the memory center",
            "split": "group by source/profile; do not split repeated rows independently",
        },
        "acceptance_allowed": False,
        "publication_allowed": False,
        "next_required_stage": (
            "fit media-aware memory and throughput heads, then freeze a source-disjoint prospective holdout"
            if all_passed
            else "repair or resume non-terminal/software-failed jobs without relabeling them as OOM"
        ),
    }
    report["report_sha256"] = sha256_json(report)
    write_json(FORMAL_OUTPUT, report)
    lines = [
        "# Qwen3.5 与 VL 补充实验结果",
        "",
        f"- 队列：{len(jobs)} 个正式作业",
        f"- success：{classifications.get('success', 0)}",
        f"- OOM（右删失下界）：{classifications.get('oom', 0)}",
        f"- 其他/未完成：{len(jobs) - classifications.get('success', 0) - classifications.get('oom', 0)}",
        f"- 可进入拟合：{'是' if all_passed else '否'}",
        "- 可作为前瞻验收或发布：否",
        "",
        "## 分轨终态",
        "",
    ]
    for track, counts in sorted(by_track.items()):
        lines.append(f"- {track}: {dict(sorted(counts.items()))}")
    lines.extend(
        [
            "",
            "## 解释约束",
            "",
            "success 是精确观测；确认的 CUDA OOM 只表示峰值显存超过可用上限，是右删失下界。",
            "重复测量用于估计运行方差，拟合前必须按同一物理 arm 折叠，不能当作独立样本。",
            "本批数据只用于拟合，模型发布仍需来源隔离的前瞻验收。",
            "",
        ]
    )
    FORMAL_MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    if not all_passed and not allow_incomplete:
        raise RuntimeError("formal supplement is incomplete or contains invalid outcomes")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("canary", "formal"), required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report = (
        evaluate_canary()
        if args.stage == "canary"
        else evaluate_formal(allow_incomplete=args.allow_incomplete)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.stage == "canary" and report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
