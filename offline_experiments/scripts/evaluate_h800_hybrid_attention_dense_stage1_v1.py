#!/usr/bin/env python3
"""Evaluate the dense hybrid-attention stage-1 canary or formal queue.

Successful jobs export exact allocator peaks and throughput counters.  OOM jobs
export only a right-censored device-capacity lower bound.  Software failures,
missing jobs and incomplete rank artifacts remain separate classifications.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    RESULTS_DIR,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from hybrid_attention_memory_features import build_dense_hybrid_features
from model_structure_manifest import validate_model_structure_manifest
from prepare_h800_hybrid_attention_dense_stage1_v1 import (
    CAMPAIGN_ID,
    CANARY_PHASE_ID,
    CANARY_QUEUE,
    FIXED_LORA,
    FORMAL_PHASE_ID,
    FORMAL_QUEUE,
    MODEL_INVENTORY,
)

CAPACITY_BYTES = 150_142_189_568
CANARY_OUTPUT = (
    ARTIFACT_DIR
    / "h800_hybrid_attention_dense_stage1_canary_acceptance_v1.json"
)
FORMAL_OUTPUT = (
    ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_results_v1.json"
)
FORMAL_RECORDS = (
    ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_observations_v1.jsonl"
)
FORMAL_MARKDOWN = (
    ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_results_v1.md"
)


def _inventory() -> dict[str, dict[str, Any]]:
    report = read_json(MODEL_INVENTORY)
    if report.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("stage-1 model inventory campaign drifted")
    return {str(row["id"]): row for row in report.get("models") or []}


def _rank_artifacts(
    job: dict[str, Any], status: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    attempt_id = str(status.get("execution_attempt_id") or "")
    metrics_dir = (
        RESULTS_DIR / str(job["job_id"]) / "attempts" / attempt_id / "metrics"
    )
    summaries: list[dict[str, Any]] = []
    structures: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for rank in range(int(job["gpu_count"])):
        summary_path = metrics_dir / f"summary.rank{rank}.json"
        structure_path = (
            metrics_dir
            / f"model_structure_manifest.{attempt_id}.rank{rank}.json"
        )
        if summary_path.is_file():
            summaries.append(read_json(summary_path))
        if structure_path.is_file():
            structures.append(
                validate_model_structure_manifest(read_json(structure_path))
            )
        bindings.append(
            {
                "rank": rank,
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": (
                    sha256_file(summary_path) if summary_path.is_file() else None
                ),
                "structure_path": str(structure_path.resolve()),
                "structure_sha256": (
                    sha256_file(structure_path)
                    if structure_path.is_file()
                    else None
                ),
            }
        )
    return summaries, structures, bindings


def _nvidia_smi_watermark_bytes(attempt_root: Path) -> int | None:
    """Highest observed device memory usage before an OOM crash.

    Every nvidia-smi sample is a real usage observation, so the maximum over
    the trace is a valid lower bound on the true pre-OOM peak.  Returns None
    when the trace is missing or contains no numeric memory readings.
    """

    smi_path = attempt_root / "nvidia_smi.csv"
    if not smi_path.is_file():
        return None
    best = 0
    try:
        with smi_path.open(encoding="utf-8") as source:
            for row in csv.DictReader(source):
                raw = str(row.get("memory.used") or "").strip()
                if not raw or not raw.replace(".", "", 1).isdigit():
                    continue
                used_mib = float(raw)
                if used_mib > 0:
                    best = max(best, int(used_mib * 1024 * 1024))
    except (OSError, csv.Error, ValueError):
        return None
    return best if best > 0 else None


def _result(job: dict[str, Any]) -> dict[str, Any]:
    status_path = RESULTS_DIR / str(job["job_id"]) / "status.json"
    if not status_path.is_file():
        return {
            "job_id": job["job_id"],
            "classification": "missing",
            "calibration_eligible": False,
            "terminal_eligible": False,
            "result_artifacts_complete": False,
            "summaries": [],
            "structures": [],
            "artifact_bindings": [],
        }
    status = read_json(status_path)
    summaries, structures, bindings = _rank_artifacts(job, status)
    classification = str(status.get("classification") or "unknown")
    attempt_root = (
        RESULTS_DIR
        / str(job["job_id"])
        / "attempts"
        / str(status.get("execution_attempt_id") or "")
    )
    oom_censor = None
    oom_censor_source = None
    if classification == "oom":
        watermark = _nvidia_smi_watermark_bytes(attempt_root)
        if watermark is not None and watermark <= CAPACITY_BYTES:
            oom_censor = watermark
            oom_censor_source = "nvidia_smi_watermark"
        else:
            oom_censor = CAPACITY_BYTES
            oom_censor_source = "device_capacity_fallback"
    # OOM rows are usable as right-censored lower bounds even when the first
    # forward crashed before any measured step (the calibration_eligible flag
    # requires measured steps and therefore cannot gate OOM rows).
    if classification == "success":
        terminal = bool(
            status.get("job_id") == job["job_id"]
            and status.get("calibration_eligible") is True
        )
    else:
        terminal = bool(
            classification == "oom"
            and status.get("job_id") == job["job_id"]
            and oom_censor is not None
        )
    success_complete = bool(
        classification == "success"
        and len(summaries) == int(job["gpu_count"])
        and len(structures) == int(job["gpu_count"])
    )
    return {
        "job_id": job["job_id"],
        "classification": classification,
        "calibration_eligible": status.get("calibration_eligible"),
        "terminal_eligible": terminal,
        "oom_right_censor_lower_bytes": oom_censor,
        "oom_censor_source": oom_censor_source,
        "result_artifacts_complete": (
            success_complete if classification == "success" else terminal
        ),
        "execution_attempt_id": status.get("execution_attempt_id"),
        "execution_fingerprint_quality": status.get(
            "execution_fingerprint_quality"
        ),
        "status_path": str(status_path.resolve()),
        "status_sha256": sha256_file(status_path),
        "summaries": summaries,
        "structures": structures,
        "artifact_bindings": bindings,
    }


def _success_metrics(result: dict[str, Any]) -> dict[str, Any] | None:
    summaries = result.get("summaries") or []
    if result.get("classification") != "success" or not summaries:
        return None
    max_reserved = max(int(row.get("max_reserved") or 0) for row in summaries)
    max_allocated = max(int(row.get("max_allocated") or 0) for row in summaries)
    measured_seconds = max(
        float(row.get("measured_seconds") or 0.0) for row in summaries
    )
    totals = [row.get("measured_totals") or {} for row in summaries]
    effective_tokens = sum(int(row.get("effective_tokens") or 0) for row in totals)
    computed_tokens = sum(int(row.get("computed_tokens") or 0) for row in totals)
    label_tokens = sum(int(row.get("label_tokens") or 0) for row in totals)
    logical_samples = sum(int(row.get("logical_samples") or 0) for row in totals)
    return {
        "max_reserved_bytes": max_reserved,
        "max_allocated_bytes": max_allocated,
        "allocator_reserved_minus_allocated_bytes": max_reserved - max_allocated,
        "measured_seconds": measured_seconds,
        "effective_tokens": effective_tokens,
        "computed_tokens": computed_tokens,
        "label_tokens": label_tokens,
        "logical_samples": logical_samples,
        "effective_tokens_per_second": (
            effective_tokens / measured_seconds if measured_seconds > 0 else None
        ),
        "computed_tokens_per_second": (
            computed_tokens / measured_seconds if measured_seconds > 0 else None
        ),
    }


def _structure_checks(
    job: dict[str, Any],
    result: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    features = build_dense_hybrid_features(job, model, FIXED_LORA, CAPACITY_BYTES)
    expected_adapter = int(features["geometry"]["adapter_parameters"])
    rank_checks = []
    for structure in result.get("structures") or []:
        components = structure.get("components") or {}
        language = components.get("language_model") or {}
        observed_adapter = int(
            language.get("adapter_trainable_parameter_elements") or 0
        )
        checks = {
            "adapter_parameter_count_exact": observed_adapter == expected_adapter,
            "language_adapter_trainable": observed_adapter > 0,
        }
        if job.get("train_scope_id") == "language_only":
            vision = components.get("vision_tower") or {}
            projector = components.get("multimodal_projector") or {}
            checks.update(
                {
                    "freeze_declaration_matched": structure.get(
                        "declaration_status"
                    )
                    == "matched",
                    "vision_trainable_zero": int(
                        vision.get("trainable_parameter_elements") or 0
                    )
                    == 0,
                    "projector_trainable_zero": int(
                        projector.get("trainable_parameter_elements") or 0
                    )
                    == 0,
                    "no_visual_lora_target_hit": (
                        structure.get("lora_target_hits") or {}
                    ).get("any_visual_component")
                    is False,
                }
            )
        rank_checks.append(
            {
                "rank": structure.get("rank"),
                "expected_adapter_parameter_elements": expected_adapter,
                "observed_adapter_parameter_elements": observed_adapter,
                "checks": checks,
                "all_passed": all(checks.values()),
            }
        )
    return {
        "job_id": job["job_id"],
        "rank_checks": rank_checks,
        "all_ranks_passed": len(rank_checks) == int(job["gpu_count"])
        and all(row["all_passed"] for row in rank_checks),
    }


def evaluate_canary() -> dict[str, Any]:
    jobs = read_jsonl(CANARY_QUEUE)
    models = _inventory()
    results = [_result(job) for job in jobs]
    structures = [
        _structure_checks(job, result, models[str(job["model_id"])])
        for job, result in zip(jobs, results)
    ]
    checks = {
        "queue_exact": len(jobs) == 4
        and all(job.get("campaign_id") == CAMPAIGN_ID for job in jobs)
        and all(job.get("phase_id") == CANARY_PHASE_ID for job in jobs),
        "four_models_covered": len({job["model_id"] for job in jobs}) == 4,
        "all_jobs_success_with_complete_artifacts": all(
            row["classification"] == "success"
            and row["terminal_eligible"]
            and row["result_artifacts_complete"]
            for row in results
        ),
        "all_structure_and_adapter_checks_passed": len(structures) == 4
        and all(row["all_ranks_passed"] for row in structures),
    }
    report: dict[str, Any] = {
        "schema": "sft_h800_hybrid_attention_dense_stage1_canary_acceptance/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": CANARY_PHASE_ID,
        "queue": {
            "path": str(CANARY_QUEUE.resolve()),
            "sha256": sha256_file(CANARY_QUEUE),
        },
        "checks": checks,
        "all_passed": all(checks.values()),
        "structure_checks": structures,
        "results": results,
        "fit_allowed": False,
        "formal_stage_allowed": all(checks.values()),
        "recommendation_release_allowed": False,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(CANARY_OUTPUT, report)
    return report


def _observation(
    job: dict[str, Any],
    result: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    classification = str(result["classification"])
    metrics = _success_metrics(result)
    features = build_dense_hybrid_features(job, model, FIXED_LORA, CAPACITY_BYTES)
    structure = _structure_checks(job, result, model)
    return {
        "schema": "sft_h800_hybrid_attention_dense_stage1_observation/v1",
        "job_id": job["job_id"],
        "scenario_id": job["scenario_id"],
        "model_id": job["model_id"],
        "release_family": job["release_family"],
        "architecture_route": job["architecture_route"],
        "architecture_signature_sha256": job["architecture_signature_sha256"],
        "num_full_attention_layers": job["num_full_attention_layers"],
        "num_linear_attention_layers": job["num_linear_attention_layers"],
        "configuration": {
            "design_arm": job["design_arm"],
            "parallel_route_id": job["parallel_route_id"],
            "gpu_count": job["gpu_count"],
            "zero": job["zero"],
            "gradient_checkpointing": job["gc"],
            "micro_batch_size": job["mbs"],
            "exact_total_tokens_per_sample": job[
                "exact_total_tokens_per_sample"
            ],
            "source_dataset_id": job["source_dataset_id"],
            "repeat": job["repeat"],
            "packing": job["packing"],
            "kernel_path": job["effective_kernel_path"],
        },
        "outcome": {
            "classification": classification,
            "terminal_eligible": result["terminal_eligible"],
            "calibration_eligible": result["calibration_eligible"],
            "peak_reserved_bytes": (
                metrics["max_reserved_bytes"] if metrics is not None else None
            ),
            "peak_allocated_bytes": (
                metrics["max_allocated_bytes"] if metrics is not None else None
            ),
            "oom_right_censor_lower_bytes": result.get(
                "oom_right_censor_lower_bytes"
            ),
            "oom_censor_source": result.get("oom_censor_source"),
            "oom_peak_is_unknown_not_imputed": classification == "oom",
        },
        "metrics": metrics,
        "feature_basis": features,
        "structure_validation": structure,
        "bindings": {
            "status_path": result.get("status_path"),
            "status_sha256": result.get("status_sha256"),
            "artifacts": result.get("artifact_bindings") or [],
            "data_sha256": job["data_sha256"],
            "dataset_profile_sha256": job["dataset_profile_sha256"],
            "model_manifest_sha256": job["declared_model_manifest_sha256"],
            "feature_basis_sha256": job["feature_basis_sha256"],
        },
    }


def _write_markdown(report: dict[str, Any]) -> None:
    classifications = report["classifications"]
    lines = [
        "# H800 dense 混合注意力第一批实验结果",
        "",
        f"- 队列：{report['jobs']} 个正式任务",
        f"- 成功：{classifications.get('success', 0)}",
        f"- OOM（只记右删失下界）：{classifications.get('oom', 0)}",
        f"- 其他或未完成：{report['jobs'] - classifications.get('success', 0) - classifications.get('oom', 0)}",
        f"- 可进入拟合：{'是' if report['fit_allowed'] else '否'}",
        "- 可作为前瞻验收或发布：否",
        "",
        "| 模型 | 成功 | OOM | 其他 |",
        "|---|---:|---:|---:|",
    ]
    for model_id, counts in report["by_model"].items():
        other = sum(counts.values()) - counts.get("success", 0) - counts.get("oom", 0)
        lines.append(
            f"| {model_id} | {counts.get('success', 0)} | "
            f"{counts.get('oom', 0)} | {other} |"
        )
    lines.extend(
        [
            "",
            "成功任务的 `max_reserved_bytes` 是精确观测；CUDA OOM 不补峰值，只记 OOM 前显卡监控水位下界（无监控记录时退到设备容量）。",
            "重复方差任务拟合前按同一物理配置折叠；两个业务来源按来源分组，不能随机拆开造成泄漏。",
            "",
        ]
    )
    FORMAL_MARKDOWN.write_text("\n".join(lines), encoding="utf-8")


def evaluate_formal(*, allow_incomplete: bool = False) -> dict[str, Any]:
    jobs = read_jsonl(FORMAL_QUEUE)
    models = _inventory()
    results = [_result(job) for job in jobs]
    observations = [
        _observation(job, result, models[str(job["model_id"])])
        for job, result in zip(jobs, results)
    ]
    write_jsonl(FORMAL_RECORDS, observations)
    classifications = Counter(
        row["outcome"]["classification"] for row in observations
    )
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    by_route: dict[str, Counter[str]] = defaultdict(Counter)
    for row in observations:
        classification = str(row["outcome"]["classification"])
        by_model[str(row["model_id"])][classification] += 1
        by_route[str(row["architecture_route"])][classification] += 1
    success_models = {
        row["model_id"]
        for row in observations
        if row["outcome"]["classification"] == "success"
    }
    checks = {
        "queue_exact": len(jobs) == 208
        and all(job.get("campaign_id") == CAMPAIGN_ID for job in jobs)
        and all(job.get("phase_id") == FORMAL_PHASE_ID for job in jobs),
        "all_jobs_terminal_success_or_oom": all(
            row["outcome"]["terminal_eligible"]
            and row["outcome"]["classification"] in {"success", "oom"}
            for row in observations
        ),
        "all_success_artifacts_complete": all(
            result["classification"] != "success"
            or result["result_artifacts_complete"]
            for result in results
        ),
        "no_software_or_infrastructure_failure": set(classifications)
        <= {"success", "oom"},
        "every_model_has_success": success_models
        == {str(job["model_id"]) for job in jobs},
        "all_success_structure_checks_passed": all(
            row["outcome"]["classification"] != "success"
            or row["structure_validation"]["all_ranks_passed"]
            for row in observations
        ),
        "all_oom_peaks_unknown": all(
            row["outcome"]["classification"] != "oom"
            or (
                row["outcome"]["peak_reserved_bytes"] is None
                and row["outcome"]["oom_right_censor_lower_bytes"] is not None
                and 0
                < row["outcome"]["oom_right_censor_lower_bytes"]
                <= CAPACITY_BYTES
                and row["outcome"]["oom_peak_is_unknown_not_imputed"] is True
            )
            for row in observations
        ),
    }
    all_passed = all(checks.values())
    report: dict[str, Any] = {
        "schema": "sft_h800_hybrid_attention_dense_stage1_results/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": FORMAL_PHASE_ID,
        "jobs": len(jobs),
        "queue": {
            "path": str(FORMAL_QUEUE.resolve()),
            "sha256": sha256_file(FORMAL_QUEUE),
        },
        "observations": {
            "path": str(FORMAL_RECORDS.resolve()),
            "sha256": sha256_file(FORMAL_RECORDS),
        },
        "checks": checks,
        "all_passed": all_passed,
        "allow_incomplete": allow_incomplete,
        "classifications": dict(sorted(classifications.items())),
        "by_model": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_model.items())
        },
        "by_architecture_route": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_route.items())
        },
        "fit_allowed": all_passed,
        "fit_policy": {
            "success": "exact peak-reserved point observation",
            "oom": "right-censored pre-OOM nvidia-smi watermark lower bound (capacity fallback)",
            "architecture": "separate full and linear attention activation/workspace coefficients",
            "repeat": "collapse by physical arm before fitting center; retain variance",
            "source": "group by source_dataset_id; never split repeats independently",
        },
        "acceptance_allowed": False,
        "publication_allowed": False,
        "rows": observations,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(FORMAL_OUTPUT, report)
    _write_markdown(report)
    if not all_passed and not allow_incomplete:
        raise RuntimeError(
            "formal dense hybrid queue is incomplete or contains invalid outcomes"
        )
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
