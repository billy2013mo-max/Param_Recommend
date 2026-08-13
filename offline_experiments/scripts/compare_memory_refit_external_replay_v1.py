#!/usr/bin/env python3
"""Compare frozen V3 and the fixed 468-row refit on external outcomes.

The external business-blind and canary outcomes are replayed only after both
models are fixed.  No coefficient, feature, threshold, or margin is selected
from these external outcomes.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import benchmark_h800_memory_center_models_v1 as memory_benchmark
import freeze_h800_unified_bounded_memory_v3 as memory_v3
import h800_unified_bounded_memory_v3_data as memory_data
import strict_train_validation_refit_v1 as strict_refit
from common import ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json, write_jsonl
from h800_unified_bounded_memory_model import load_artifact, predict_records
from prepare_h800_final_memory_business_blind_v2 import implementation as business_prep


SCHEMA = "sft_memory_refit_external_replay/v1"
IMPLEMENTATION_VERSION = "2026-08-12.frozen-v3-vs-fixed-468-refit"
DEFAULT_OUTPUT_DIR = ROOT / "diagnostics" / "memory_refit_external_replay_v1"
FROZEN_ARTIFACT = ROOT / "artifacts" / "h800_unified_bounded_memory_candidate_v3.json"
BUSINESS_RESULTS = ROOT / "artifacts" / "h800_final_memory_business_blind_results_v2.json"
CANARY_RESULTS = ROOT / "artifacts" / "h800_unified_bounded_canary_results_v3.json"
CANARY_PREDICTIONS = ROOT / "artifacts" / "h800_unified_bounded_canary_frozen_predictions_v3.json"


def _business_records(
    frozen_artifact: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    jobs = read_jsonl(business_prep.DEFAULT_QUEUE)
    results = read_json(BUSINESS_RESULTS)
    official_predictions = {
        str(row["job_id"]): row
        for row in read_json(business_prep.DEFAULT_PREDICTIONS)["rows"]
    }
    observations = {
        str(row["job_id"]): dict(row) for row in results["observations"]
    }

    inventory = read_json(business_prep.MODEL_INVENTORY)
    models = {str(row["id"]): dict(row) for row in inventory["models"]}
    models = copy.deepcopy(models)
    for model_id, parameters in memory_data.RUNTIME_BASE_PARAMETERS.items():
        if model_id in models:
            models[model_id]["actual_parameters"] = parameters
    cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    terminal_observations: list[dict[str, Any]] = []
    terminal_jobs: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job["job_id"])
        observation = observations.get(job_id)
        if observation is None:
            continue
        record = business_prep._prediction_record(
            job,
            inventory=inventory,
            models=models,
            capacity=int(frozen_artifact["hardware_domain"]["capacity_bytes"]),
            profile_cache=cache,
        )
        record["record_id"] = f"external_business::{job_id}"
        if observation["classification"] == "success":
            record["state"] = "exact"
            record["target_reserved_bytes"] = float(
                observation["max_reserved_bytes"]
            )
            record["censor_lower_bytes"] = None
        else:
            record["state"] = "censored"
            record["target_reserved_bytes"] = None
            record["censor_lower_bytes"] = float(
                frozen_artifact["hardware_domain"]["capacity_bytes"]
            )
        records.append(record)
        terminal_observations.append(observation)
        terminal_jobs.append(job)

    reconstructed = predict_records(records, frozen_artifact)
    center_relative_errors: list[float] = []
    upper_relative_errors: list[float] = []
    decision_mismatches = 0
    for job, prediction in zip(terminal_jobs, reconstructed):
        official = official_predictions[str(job["job_id"])]
        center_relative_errors.append(
            abs(float(prediction["center_bytes"]) / float(official["center_bytes"]) - 1.0)
        )
        upper_relative_errors.append(
            abs(
                float(prediction["admission_upper_bytes"])
                / float(official["admission_upper_bytes"])
                - 1.0
            )
        )
        decision_mismatches += int(
            bool(prediction["admitted"]) != bool(official["admitted"])
        )
    return records, terminal_observations, {
        "jobs_planned": len(jobs),
        "jobs_terminal": len(records),
        "success": sum(row["classification"] == "success" for row in terminal_observations),
        "oom": sum(row["classification"] == "oom" for row in terminal_observations),
        "source_datasets": len({str(row["source_id"]) for row in records}),
        "frozen_input_reconstruction": {
            "max_center_relative_error": max(center_relative_errors, default=0.0),
            "max_upper_relative_error": max(upper_relative_errors, default=0.0),
            "admission_decision_mismatches": decision_mismatches,
        },
    }


def _canary_records(
    frozen_artifact: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frozen = read_json(CANARY_PREDICTIONS)
    results = read_json(CANARY_RESULTS)
    observations = {
        str(row["job_id"]): dict(row) for row in results["observations"]
    }
    records: list[dict[str, Any]] = []
    terminal_observations: list[dict[str, Any]] = []
    official_by_job: dict[str, Mapping[str, Any]] = {}
    for row in frozen["rows"]:
        job_id = str(row["job_id"])
        observation = observations.get(job_id)
        if observation is None:
            continue
        record = copy.deepcopy(row["record"])
        if observation["classification"] == "success":
            record["state"] = "exact"
            record["target_reserved_bytes"] = float(
                observation["max_reserved_bytes"]
            )
            record["censor_lower_bytes"] = None
        else:
            record["state"] = "censored"
            record["target_reserved_bytes"] = None
            record["censor_lower_bytes"] = float(
                frozen_artifact["hardware_domain"]["capacity_bytes"]
            )
        records.append(record)
        terminal_observations.append(observation)
        official_by_job[job_id] = row["v3"]

    reconstructed = predict_records(records, frozen_artifact)
    center_relative_errors: list[float] = []
    upper_relative_errors: list[float] = []
    decision_mismatches = 0
    for observation, prediction in zip(terminal_observations, reconstructed):
        official = official_by_job[str(observation["job_id"])]
        center_relative_errors.append(
            abs(float(prediction["center_bytes"]) / float(official["center_bytes"]) - 1.0)
        )
        upper_relative_errors.append(
            abs(
                float(prediction["admission_upper_bytes"])
                / float(official["admission_upper_bytes"])
                - 1.0
            )
        )
        decision_mismatches += int(
            bool(prediction["admitted"]) != bool(official["admitted"])
        )
    return records, terminal_observations, {
        "jobs_planned": len(frozen["rows"]),
        "jobs_terminal": len(records),
        "success": sum(row["classification"] == "success" for row in terminal_observations),
        "oom": sum(row["classification"] == "oom" for row in terminal_observations),
        "source_datasets": len({str(row["source_id"]) for row in records}),
        "frozen_input_reconstruction": {
            "max_center_relative_error": max(center_relative_errors, default=0.0),
            "max_upper_relative_error": max(upper_relative_errors, default=0.0),
            "admission_decision_mismatches": decision_mismatches,
        },
    }


def _refit_models() -> tuple[
    dict[str, Any], dict[str, Any], float, dict[str, Any], set[str]
]:
    development, _audit = memory_data.development_records()
    train, validation, split = strict_refit._memory_split(development)
    center = memory_benchmark._fit_model(train, memory_v3.CENTER_CANDIDATE)
    risk = memory_benchmark._fit_model(train, memory_v3.RISK_CANDIDATE)
    multiplier = memory_v3._calibrate_risk_multiplier(
        train, memory_v3._model_bytes(train, risk)
    )
    return (
        center,
        risk,
        multiplier,
        {
            **split,
            "development_rows": len(development),
            "development_sources": len(
                {str(row["source_id"]) for row in development}
            ),
            "internal_validation_rows_not_used_for_fit": len(validation),
        },
        {str(row["source_id"]) for row in train},
    )


def _model_predictions(
    records: Sequence[Mapping[str, Any]],
    *,
    center_model: Mapping[str, Any],
    risk_model: Mapping[str, Any],
    multiplier: float,
) -> list[dict[str, Any]]:
    centers = memory_v3._model_bytes(records, center_model)
    risks = memory_v3._model_bytes(records, risk_model)
    safe_limit = memory_v3._safe_limit()
    return [
        {
            "center_bytes": center,
            "risk_head_bytes": risk,
            "upper_multiplier": multiplier,
            "admission_upper_bytes": max(center, risk * multiplier),
            "safe_limit_bytes": safe_limit,
            "admitted": max(center, risk * multiplier) <= safe_limit,
        }
        for center, risk in zip(centers, risks)
    ]


def _metrics(
    records: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    by_source_errors: dict[str, list[float]] = defaultdict(list)
    errors: list[float] = []
    signed_errors: list[float] = []
    safe_rows = unsafe_rows = oom_rows = 0
    safe_admitted = unsafe_admitted = oom_admitted = 0
    for record, observation, prediction in zip(records, observations, predictions):
        classification = str(observation["classification"])
        observed = (
            float(observation["max_reserved_bytes"])
            if classification == "success"
            else None
        )
        safe_limit = float(prediction["safe_limit_bytes"])
        if classification == "oom":
            actual_class = "oom"
            oom_rows += 1
            oom_admitted += int(bool(prediction["admitted"]))
            signed_error = None
        else:
            assert observed is not None
            signed_error = float(prediction["center_bytes"]) / observed - 1.0
            errors.append(abs(signed_error))
            signed_errors.append(signed_error)
            by_source_errors[str(record["source_id"])].append(abs(signed_error))
            if observed <= safe_limit:
                actual_class = "safe_success"
                safe_rows += 1
                safe_admitted += int(bool(prediction["admitted"]))
            else:
                actual_class = "unsafe_success"
                unsafe_rows += 1
                unsafe_admitted += int(bool(prediction["admitted"]))
        details.append(
            {
                "job_id": str(observation["job_id"]),
                "record_id": str(record["record_id"]),
                "source_id": str(record["source_id"]),
                "model_id": str(record["model_id"]),
                "train_type": str(record["train_type"]),
                "gpu_count": int(record["gpu_count"]),
                "zero_stage": int(record["zero_stage"]),
                "gc": bool(record["gc"]),
                "packing": bool(record["packing"]),
                "mbs": int(record["mbs"]),
                "cutoff_len": int(record["cutoff_len"]),
                "classification": classification,
                "actual_class": actual_class,
                "observed_reserved_bytes": observed,
                "center_bytes": float(prediction["center_bytes"]),
                "risk_head_bytes": float(prediction["risk_head_bytes"]),
                "upper_multiplier": float(prediction["upper_multiplier"]),
                "admission_upper_bytes": float(prediction["admission_upper_bytes"]),
                "safe_limit_bytes": safe_limit,
                "admitted": bool(prediction["admitted"]),
                "signed_percentage_error": signed_error,
                "absolute_percentage_error": (
                    abs(signed_error) if signed_error is not None else None
                ),
            }
        )
    return {
        "success_rows": len(errors),
        "source_rows": len(by_source_errors),
        "row_center_mape": statistics.fmean(errors) if errors else None,
        "source_equal_center_mape": (
            statistics.fmean(
                statistics.fmean(current) for current in by_source_errors.values()
            )
            if by_source_errors
            else None
        ),
        "center_signed_bias": (
            statistics.fmean(signed_errors) if signed_errors else None
        ),
        "safe_success_admitted": safe_admitted,
        "safe_success_rows": safe_rows,
        "safe_success_admission_rate": (
            safe_admitted / safe_rows if safe_rows else None
        ),
        "unsafe_success_admitted": unsafe_admitted,
        "unsafe_success_rows": unsafe_rows,
        "unsafe_success_admission_rate": (
            unsafe_admitted / unsafe_rows if unsafe_rows else None
        ),
        "oom_admitted": oom_admitted,
        "oom_rows": oom_rows,
        "oom_admission_rate": oom_admitted / oom_rows if oom_rows else None,
    }, details


def _score_dataset(
    records: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    *,
    frozen_artifact: Mapping[str, Any],
    refit_center: Mapping[str, Any],
    refit_risk: Mapping[str, Any],
    refit_multiplier: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    frozen_raw = predict_records(records, frozen_artifact)
    frozen_predictions = [
        {
            "center_bytes": row["center_bytes"],
            "risk_head_bytes": row["risk_guard_bytes"],
            "upper_multiplier": row["upper_multiplier"],
            "admission_upper_bytes": row["admission_upper_bytes"],
            "safe_limit_bytes": row["safe_limit_bytes"],
            "admitted": row["admitted"],
        }
        for row in frozen_raw
    ]
    refit_predictions = _model_predictions(
        records,
        center_model=refit_center,
        risk_model=refit_risk,
        multiplier=refit_multiplier,
    )
    frozen_metrics, frozen_details = _metrics(
        records, observations, frozen_predictions
    )
    refit_metrics, refit_details = _metrics(records, observations, refit_predictions)
    details = [
        {"predictor_id": "frozen_v3_531_fit", **row} for row in frozen_details
    ] + [{"predictor_id": "fixed_v3_468_refit", **row} for row in refit_details]
    changes = []
    for frozen, refit in zip(frozen_details, refit_details):
        if bool(frozen["admitted"]) == bool(refit["admitted"]):
            continue
        changes.append(
            {
                "job_id": frozen["job_id"],
                "actual_class": frozen["actual_class"],
                "model_id": frozen["model_id"],
                "train_type": frozen["train_type"],
                "gpu_count": frozen["gpu_count"],
                "zero_stage": frozen["zero_stage"],
                "gc": frozen["gc"],
                "mbs": frozen["mbs"],
                "cutoff_len": frozen["cutoff_len"],
                "observed_reserved_bytes": frozen["observed_reserved_bytes"],
                "frozen_admitted": frozen["admitted"],
                "frozen_center_bytes": frozen["center_bytes"],
                "frozen_upper_bytes": frozen["admission_upper_bytes"],
                "refit_admitted": refit["admitted"],
                "refit_center_bytes": refit["center_bytes"],
                "refit_upper_bytes": refit["admission_upper_bytes"],
                "safe_limit_bytes": frozen["safe_limit_bytes"],
            }
        )
    change_summary: dict[str, int] = defaultdict(int)
    for row in changes:
        transition = (
            "frozen_reject_to_refit_admit"
            if row["refit_admitted"]
            else "frozen_admit_to_refit_reject"
        )
        change_summary[f"{transition}::{row['actual_class']}"] += 1
    return {
        "frozen_v3_531_fit": {
            "model_label_zh": "冻结 V3（531条全量拟合）",
            "metrics": frozen_metrics,
        },
        "fixed_v3_468_refit": {
            "model_label_zh": "单切分 V3（468条重拟合）",
            "metrics": refit_metrics,
        },
    }, details, {
        "changed_jobs": len(changes),
        "counts": dict(sorted(change_summary.items())),
        "cases": changes,
    }


def _rate(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} = {numerator / denominator:.2%}" if denominator else "无样本"


def _decision_label(key: str) -> str:
    transition, actual_class = key.split("::", maxsplit=1)
    transition_zh = {
        "frozen_reject_to_refit_admit": "冻结拒绝→重拟合放行",
        "frozen_admit_to_refit_reject": "冻结放行→重拟合拒绝",
    }[transition]
    class_zh = {
        "safe_success": "安全成功",
        "unsafe_success": "危险成功",
        "oom": "OOM",
    }[actual_class]
    return f"{transition_zh}的{class_zh}"


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# 显存 V3 重拟合外部回放比较",
        "",
        "> 业务盲测和 Canary 均未参与两套模型的系数、阈值或风险倍数拟合。结果已被查看过，因此这里是外部回放，不重新称为新鲜盲测。",
        "",
        "## 数据隔离",
        "",
        f"- 重拟合模型：训练 {report['refit']['train_rows']} 条/{report['refit']['train_sources']} 个来源；内部留出 {report['refit']['validation_rows']} 条/{report['refit']['validation_sources']} 个来源。",
        f"- 业务盲测：{report['datasets']['business_blind']['data']['jobs_terminal']} 个终态任务，训练来源重叠 {len(report['datasets']['business_blind']['train_source_overlap'])}。",
        f"- Canary：{report['datasets']['canary']['data']['jobs_terminal']} 个终态任务，训练来源重叠 {len(report['datasets']['canary']['train_source_overlap'])}。",
        "",
    ]
    for dataset_id, title, center_key in (
        ("business_blind", "业务盲测（最新45个终态任务）", "source_equal_center_mape"),
        ("canary", "Canary（10个任务）", "row_center_mape"),
    ):
        dataset = report["datasets"][dataset_id]
        center_label = "数据集等权中心 MAPE" if dataset_id == "business_blind" else "行级中心 MAPE"
        lines.extend(
            [
                f"## {title}",
                "",
                f"| 模型 | {center_label} | 安全放行率 | 危险成功放行 | OOM 放行 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for model in dataset["models"].values():
            metrics = model["metrics"]
            lines.append(
                "| {label} | {center:.2%} | {safe} | {unsafe} | {oom} |".format(
                    label=model["model_label_zh"],
                    center=metrics[center_key],
                    safe=_rate(
                        metrics["safe_success_admitted"],
                        metrics["safe_success_rows"],
                    ),
                    unsafe=_rate(
                        metrics["unsafe_success_admitted"],
                        metrics["unsafe_success_rows"],
                    ),
                    oom=_rate(metrics["oom_admitted"], metrics["oom_rows"]),
                )
            )
        lines.extend(
            [
                "",
                "决策变化：" + "；".join(
                    f"{_decision_label(key)} {value} 条"
                    for key, value in dataset["decision_comparison"]["counts"].items()
                ),
                "",
            ]
        )
        dangerous = next(
            (
                row
                for row in dataset["decision_comparison"]["cases"]
                if row["actual_class"] == "unsafe_success"
                and row["refit_admitted"]
                and not row["frozen_admitted"]
            ),
            None,
        )
        if dangerous is not None:
            gib = float(1 << 30)
            lines.extend(
                [
                    "具体危险变化：{model}、{mode}、{gpu}卡、ZeRO-{zero}、GC {gc}、MBS={mbs}，实测 {observed:.2f} GiB，高于 {safe:.2f} GiB 安全线；冻结 V3 上界 {frozen:.2f} GiB，拒绝；重拟合上界 {refit:.2f} GiB，改为放行。".format(
                        model=dangerous["model_id"],
                        mode=dangerous["train_type"],
                        gpu=dangerous["gpu_count"],
                        zero=dangerous["zero_stage"],
                        gc="开" if dangerous["gc"] else "关",
                        mbs=dangerous["mbs"],
                        observed=float(dangerous["observed_reserved_bytes"]) / gib,
                        safe=float(dangerous["safe_limit_bytes"]) / gib,
                        frozen=float(dangerous["frozen_upper_bytes"]) / gib,
                        refit=float(dangerous["refit_upper_bytes"]) / gib,
                    ),
                    "",
                ]
            )
    lines.extend(
        [
            "## 解释边界",
            "",
            "- 数学定义：安全放行率的分母只包含实测不超过 95% 显存安全线的成功任务。",
            "- 当前实验观察：危险成功放行单独列出，不能被安全放行率或 OOM 放行率掩盖。",
            "- 工程假设：这些已完成外部任务对未来业务分布仍有代表性；该假设不是数学保证。",
            "",
        ]
    )
    return "\n".join(lines)


def build_report() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frozen_artifact = load_artifact(FROZEN_ARTIFACT)
    (
        refit_center,
        refit_risk,
        refit_multiplier,
        refit_split,
        train_sources,
    ) = _refit_models()

    business_records, business_observations, business_data = _business_records(
        frozen_artifact
    )
    canary_records, canary_observations, canary_data = _canary_records(
        frozen_artifact
    )
    business_models, business_details, business_decisions = _score_dataset(
        business_records,
        business_observations,
        frozen_artifact=frozen_artifact,
        refit_center=refit_center,
        refit_risk=refit_risk,
        refit_multiplier=refit_multiplier,
    )
    canary_models, canary_details, canary_decisions = _score_dataset(
        canary_records,
        canary_observations,
        frozen_artifact=frozen_artifact,
        refit_center=refit_center,
        refit_risk=refit_risk,
        refit_multiplier=refit_multiplier,
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "external_replay_complete",
        "analysis_only": True,
        "fresh_blind_claimed": False,
        "external_outcomes_used_for_refit": False,
        "frozen_artifact_mutated": False,
        "refit": {
            **refit_split,
            "center_candidate": dict(memory_v3.CENTER_CANDIDATE),
            "risk_candidate": dict(memory_v3.RISK_CANDIDATE),
            "risk_multiplier": refit_multiplier,
        },
        "datasets": {
            "business_blind": {
                "data": business_data,
                "train_source_overlap": sorted(
                    train_sources
                    & {str(row["source_id"]) for row in business_records}
                ),
                "models": business_models,
                "decision_comparison": business_decisions,
            },
            "canary": {
                "data": canary_data,
                "train_source_overlap": sorted(
                    train_sources & {str(row["source_id"]) for row in canary_records}
                ),
                "models": canary_models,
                "decision_comparison": canary_decisions,
            },
        },
        "inputs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                FROZEN_ARTIFACT,
                business_prep.DEFAULT_QUEUE,
                business_prep.DEFAULT_PREDICTIONS,
                BUSINESS_RESULTS,
                CANARY_PREDICTIONS,
                CANARY_RESULTS,
            )
        },
    }
    report["report_sha256"] = sha256_json(report)
    predictions = [
        {"external_dataset": "business_blind", **row}
        for row in business_details
    ] + [{"external_dataset": "canary", **row} for row in canary_details]
    return report, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report, predictions = build_report()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "report.json", report)
    write_jsonl(args.output_dir / "external_predictions.jsonl", predictions)
    (args.output_dir / "report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str((args.output_dir / "report.json").resolve()),
                "markdown": str((args.output_dir / "report.md").resolve()),
                "report_sha256": report["report_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
