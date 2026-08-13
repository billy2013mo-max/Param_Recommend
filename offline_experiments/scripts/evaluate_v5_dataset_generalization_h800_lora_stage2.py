#!/usr/bin/env python3
"""Replay the frozen V5 throughput model on new H800 text datasets.

The evaluation isolates dataset-distribution transfer as far as the available
campaign permits: the card, runtime, model family, training mode, GBS, ZeRO
stage, GC setting and packing setting are fixed to fitted-domain choices.  A
strict subset additionally requires the cutoff to have appeared in V5 fitting
data.  New measurements are labels only and are never used to refit V5.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Any

from common import ROOT, read_json, sha256_file, sha256_json, write_json
from throughput_predictor import ThroughputPredictor


SCHEMA = "sft_v5_dataset_generalization_h800_lora_stage2/v1"
IMPLEMENTATION_VERSION = (
    "sft_v5_dataset_generalization_h800_lora_stage2/2026-08-05.v1"
)
QUEUE = ROOT / "matrix" / "h800_lora_safety_stage2_jobs_v1.jsonl"
DESIGN = ROOT / "artifacts" / "h800_lora_safety_stage2_experiment_design_v1.json"
PROFILE_DIR = ROOT / "artifacts" / "h800_lora_safety_stage2_v1" / "profiles"
RESULTS = ROOT / "results"
MODEL_ARTIFACT = ROOT / "artifacts" / "structured_throughput_modeling.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "v5_dataset_generalization_h800_lora_stage2_20260805.json"
DEFAULT_MARKDOWN = ROOT / "artifacts" / "v5_dataset_generalization_h800_lora_stage2_20260805.md"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _latest_attempt(job_id: str) -> tuple[dict[str, Any], Path]:
    latest_path = RESULTS / job_id / "latest_attempt.json"
    if not latest_path.exists():
        raise ValueError(f"Missing latest attempt for {job_id}")
    latest = read_json(latest_path)
    if str(latest.get("job_id")) != job_id:
        raise ValueError(f"Latest-attempt job mismatch for {job_id}")
    attempt = RESULTS / job_id / str(latest["attempt_path"])
    status_path = attempt / "status.json"
    if not status_path.exists():
        raise ValueError(f"Missing status for {job_id}")
    status = read_json(status_path)
    if str(status.get("job_id")) != job_id:
        raise ValueError(f"Status job mismatch for {job_id}")
    return status, attempt


def _aggregate_success(
    job: Mapping[str, Any],
    status: Mapping[str, Any],
    attempt: Path,
) -> dict[str, Any]:
    summary_paths = sorted((attempt / "metrics").glob("summary.rank*.json"))
    expected_ranks = int(job["gpu_count"])
    if len(summary_paths) != expected_ranks:
        raise ValueError(
            f"{job['job_id']} has {len(summary_paths)} summaries; "
            f"expected {expected_ranks}"
        )
    summaries = [read_json(path) for path in summary_paths]
    measured_steps_set = {int(summary["measured_steps"]) for summary in summaries}
    if len(measured_steps_set) != 1:
        raise ValueError(f"Rank step counts differ for {job['job_id']}")
    measured_steps = measured_steps_set.pop()
    measured_seconds = max(float(summary["measured_seconds"]) for summary in summaries)
    counter_names = (
        "effective_tokens",
        "computed_tokens",
        "computed_attention_token_pairs",
        "logical_samples",
    )
    totals = {
        name: sum(int(summary["measured_totals"][name]) for summary in summaries)
        for name in counter_names
    }
    ledger_authoritative = all(
        bool((summary.get("token_ledger_evidence") or {}).get("authoritative"))
        for summary in summaries
    )
    return {
        "measured_steps": measured_steps,
        "measured_seconds": measured_seconds,
        "step_seconds": measured_seconds / measured_steps,
        "effective_tokens_per_second": totals["effective_tokens"] / measured_seconds,
        "work_per_step": {
            name: totals[name] / measured_steps for name in counter_names
        },
        "max_allocated_bytes": max(float(summary["max_allocated"]) for summary in summaries),
        "max_reserved_bytes": max(float(summary["max_reserved"]) for summary in summaries),
        "finished_unix": float(status["finished_unix"]),
        "summary_paths": [str(path.resolve()) for path in summary_paths],
        "all_rank_token_ledgers_authoritative": ledger_authoritative,
    }


def _prediction_request(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **job,
        "request_id": str(job["job_id"]),
        "comparison_group": str(job["scenario_id"]),
        "hardware_id": "h800",
        "dtype": "bf16",
    }


def _collect_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    jobs = _read_jsonl(QUEUE)
    predictor = ThroughputPredictor(
        model_artifact=MODEL_ARTIFACT,
        additional_dataset_profile_dir=PROFILE_DIR,
    )
    prediction_report = predictor.predict_many(
        [_prediction_request(job) for job in jobs],
        explain=True,
    )
    predictions = {
        str(row["request_id"]): row for row in prediction_report["predictions"]
    }
    rows: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job["job_id"])
        status, attempt = _latest_attempt(job_id)
        state = str(status.get("classification"))
        if state not in {"success", "oom"}:
            raise ValueError(f"Unexpected state {state!r} for {job_id}")
        prediction = predictions[job_id]
        raw_observed = (
            _aggregate_success(job, status, attempt)
            if state == "success"
            else None
        )
        if raw_observed is not None and not raw_observed[
            "all_rank_token_ledgers_authoritative"
        ]:
            state = "success_non_authoritative"
            observed = None
        elif raw_observed is not None:
            state = "success_authoritative"
            observed = raw_observed
        else:
            observed = None
        predicted_rate = float(prediction["predicted_effective_tokens_per_second"])
        row: dict[str, Any] = {
            "job_id": job_id,
            "scenario_id": str(job["scenario_id"]),
            "dataset_id": str(job["dataset_id"]),
            "model_id": str(job["model_id"]),
            "business_scene": str(job.get("business_scene") or ""),
            "dataset_category": str(job.get("dataset_category") or ""),
            "ratio_bin": str(job.get("ratio_bin") or ""),
            "cutoff_len": int(job["cutoff_len"]),
            "mbs": int(job["mbs"]),
            "gpu_count": int(job["gpu_count"]),
            "target_gbs": int(job["target_gbs"]),
            "zero_stage": int(job["zero_stage"]),
            "gradient_checkpointing": bool(job["gradient_checkpointing"]),
            "packing": bool(job["packing"]),
            "state": state,
            "status_path": str((attempt / "status.json").resolve()),
            "predicted_effective_tokens_per_second": predicted_rate,
            "predicted_step_seconds": float(prediction["predicted_step_seconds"]),
            "predicted_work_per_step": dict(prediction["work_per_step"]),
            "confidence": prediction["confidence"],
            "structured_feature_values": prediction["explanation"][
                "structured_feature_values"
            ],
            "observed": observed,
        }
        if observed is not None:
            observed_rate = float(observed["effective_tokens_per_second"])
            row["throughput_signed_error"] = predicted_rate / observed_rate - 1.0
            row["throughput_absolute_percentage_error"] = abs(
                row["throughput_signed_error"]
            )
            row["work_relative_errors"] = {
                name: (
                    float(prediction["work_per_step"][name])
                    / float(observed["work_per_step"][name])
                    - 1.0
                )
                for name in (
                    "effective_tokens",
                    "computed_tokens",
                    "computed_attention_token_pairs",
                    "logical_samples",
                )
            }
        rows.append(row)
    return rows, prediction_report


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    all_rows = list(rows)
    successful = [
        row for row in all_rows if row["state"] == "success_authoritative"
    ]
    errors = [float(row["throughput_absolute_percentage_error"]) for row in successful]
    signed = [float(row["throughput_signed_error"]) for row in successful]
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in successful:
        by_scenario[str(row["scenario_id"])].append(row)

    scenario_mapes: list[float] = []
    scenario_pair_accuracies: list[float] = []
    top1_regrets: list[float] = []
    hit_at_10: list[float] = []
    exact_top1: list[float] = []
    pair_correct = 0
    pair_rows = 0
    ranking_details: list[dict[str, Any]] = []
    for scenario_id, scenario_rows in sorted(by_scenario.items()):
        scenario_mapes.append(
            statistics.fmean(
                float(row["throughput_absolute_percentage_error"])
                for row in scenario_rows
            )
        )
        current_correct = 0
        current_pairs = 0
        for left, right in itertools.combinations(scenario_rows, 2):
            observed_delta = (
                float(left["observed"]["effective_tokens_per_second"])
                - float(right["observed"]["effective_tokens_per_second"])
            )
            if abs(observed_delta) <= 1.0e-12:
                continue
            predicted_delta = (
                float(left["predicted_effective_tokens_per_second"])
                - float(right["predicted_effective_tokens_per_second"])
            )
            correct = (observed_delta > 0.0) == (predicted_delta > 0.0)
            current_correct += int(correct)
            current_pairs += 1
        pair_correct += current_correct
        pair_rows += current_pairs
        if current_pairs:
            scenario_pair_accuracies.append(current_correct / current_pairs)
        if len(scenario_rows) < 2:
            continue
        selected = max(
            scenario_rows,
            key=lambda row: float(row["predicted_effective_tokens_per_second"]),
        )
        oracle = max(
            scenario_rows,
            key=lambda row: float(row["observed"]["effective_tokens_per_second"]),
        )
        selected_observed = float(selected["observed"]["effective_tokens_per_second"])
        oracle_observed = float(oracle["observed"]["effective_tokens_per_second"])
        regret = max(0.0, 1.0 - selected_observed / oracle_observed)
        top1_regrets.append(regret)
        hit_at_10.append(float(regret <= 0.10))
        exact_top1.append(float(selected["job_id"] == oracle["job_id"]))
        ranking_details.append(
            {
                "scenario_id": scenario_id,
                "dataset_id": str(selected["dataset_id"]),
                "successful_candidates": len(scenario_rows),
                "selected_mbs": int(selected["mbs"]),
                "oracle_mbs": int(oracle["mbs"]),
                "selected_observed_effective_tokens_per_second": selected_observed,
                "oracle_observed_effective_tokens_per_second": oracle_observed,
                "top1_regret": regret,
                "hit_at_10_percent": regret <= 0.10,
                "exact_top1": selected["job_id"] == oracle["job_id"],
            }
        )

    work_metrics: dict[str, Any] = {}
    for name in (
        "effective_tokens",
        "computed_tokens",
        "computed_attention_token_pairs",
        "logical_samples",
    ):
        values = [
            float(row["work_relative_errors"][name]) for row in successful
        ]
        work_metrics[name] = {
            "mape": _mean([abs(value) for value in values]),
            "mean_signed_error": _mean(values),
            "p90_absolute_percentage_error": _percentile(
                [abs(value) for value in values], 90
            ),
        }

    reason_counts: dict[str, int] = defaultdict(int)
    outside_feature_counts: dict[str, int] = defaultdict(int)
    for row in all_rows:
        for reason in row["confidence"]["reasons"]:
            reason_counts[str(reason["code"])] += 1
        for feature in row["confidence"]["outside_feature_support"]:
            outside_feature_counts[str(feature["feature"])] += 1

    return {
        "candidate_rows": len(all_rows),
        "dataset_rows": len({str(row["dataset_id"]) for row in all_rows}),
        "scenario_rows": len({str(row["scenario_id"]) for row in all_rows}),
        "state_counts": {
            "success_authoritative": len(successful),
            "success_non_authoritative": sum(
                row["state"] == "success_non_authoritative" for row in all_rows
            ),
            "oom": sum(row["state"] == "oom" for row in all_rows),
        },
        "absolute": {
            "throughput_mape": _mean(errors),
            "scenario_equal_throughput_mape": _mean(scenario_mapes),
            "throughput_median_ape": statistics.median(errors) if errors else None,
            "throughput_p90_ape": _percentile(errors, 90),
            "throughput_mean_signed_error": _mean(signed),
            "within_10_percent_fraction": _mean([float(value <= 0.10) for value in errors]),
            "within_20_percent_fraction": _mean([float(value <= 0.20) for value in errors]),
            "within_30_percent_fraction": _mean([float(value <= 0.30) for value in errors]),
        },
        "ranking": {
            "comparable_scenarios": sum(len(value) >= 2 for value in by_scenario.values()),
            "pooled_pairwise_comparisons": pair_rows,
            "pooled_pairwise_accuracy": pair_correct / pair_rows if pair_rows else None,
            "scenario_equal_pairwise_accuracy": _mean(scenario_pair_accuracies),
            "mean_top1_regret": _mean(top1_regrets),
            "worst_top1_regret": max(top1_regrets) if top1_regrets else None,
            "hit_at_10_percent_fraction": _mean(hit_at_10),
            "exact_top1_fraction": _mean(exact_top1),
            "details": ranking_details,
        },
        "static_work_reconstruction": work_metrics,
        "confidence_reason_candidate_counts": dict(sorted(reason_counts.items())),
        "outside_feature_candidate_counts": dict(sorted(outside_feature_counts.items())),
    }


def _per_dataset(rows: Sequence[Mapping[str, Any]], design: Mapping[str, Any]) -> list[dict[str, Any]]:
    scenario_design = {
        str(row["dataset_id"]): row for row in design["scenarios"]
    }
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset_id"])].append(row)
    result: list[dict[str, Any]] = []
    for dataset_id, current in sorted(by_dataset.items()):
        metric = _metrics(current)
        metadata = scenario_design[dataset_id]
        ranking = metric["ranking"]
        detail = ranking["details"][0] if ranking["details"] else None
        result.append(
            {
                "dataset_id": dataset_id,
                "business_scene": str(metadata.get("business_scene") or ""),
                "model_id": str(metadata["model_id"]),
                "dataset_category": str(metadata["dataset_category"]),
                "ratio_bin": str(metadata["ratio_bin"]),
                "cutoff_len": int(metadata["cutoff_len"]),
                "profile_statistics": dict(metadata["profile_statistics"]),
                "state_counts": metric["state_counts"],
                "throughput_mape": metric["absolute"]["throughput_mape"],
                "throughput_p90_ape": metric["absolute"]["throughput_p90_ape"],
                "throughput_mean_signed_error": metric["absolute"][
                    "throughput_mean_signed_error"
                ],
                "effective_work_mape": metric["static_work_reconstruction"][
                    "effective_tokens"
                ]["mape"],
                "computed_work_mape": metric["static_work_reconstruction"][
                    "computed_tokens"
                ]["mape"],
                "scenario_equal_pairwise_accuracy": ranking[
                    "scenario_equal_pairwise_accuracy"
                ],
                "top1_regret": ranking["mean_top1_regret"],
                "selected_mbs": detail["selected_mbs"] if detail else None,
                "oracle_mbs": detail["oracle_mbs"] if detail else None,
                "outside_features": sorted(
                    {
                        str(feature["feature"])
                        for row in current
                        for feature in row["confidence"]["outside_feature_support"]
                    }
                ),
            }
        )
    return result


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


def _markdown(report: Mapping[str, Any]) -> str:
    primary = report["evaluation"]["strict_dataset_only_primary"]
    all_data = report["evaluation"]["all_new_datasets"]
    lines = [
        "# V5 新数据集泛化验证（H800、域内 Qwen3）",
        "",
        "V5 参数保持冻结；新实验结果只作为标签，不参与重拟合。吞吐排序以显存安全成功的候选为条件。",
        "",
        "## 汇总",
        "",
        "| 范围 | 数据集 | 权威成功/非权威/OOM | MAPE | P90 APE | Pairwise | Top-1 regret | Hit@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metric in (
        ("严格数据集泛化主结果", primary),
        ("全部新数据集", all_data),
    ):
        lines.append(
            "| {label} | {datasets} | {success}/{non_authoritative}/{oom} | {mape} | {p90} | {pair} | {regret} | {hit} |".format(
                label=label,
                datasets=metric["dataset_rows"],
                success=metric["state_counts"]["success_authoritative"],
                non_authoritative=metric["state_counts"]["success_non_authoritative"],
                oom=metric["state_counts"]["oom"],
                mape=_percent(metric["absolute"]["scenario_equal_throughput_mape"]),
                p90=_percent(metric["absolute"]["throughput_p90_ape"]),
                pair=_percent(metric["ranking"]["scenario_equal_pairwise_accuracy"]),
                regret=_percent(metric["ranking"]["mean_top1_regret"]),
                hit=_percent(metric["ranking"]["hit_at_10_percent_fraction"]),
            )
        )
    lines.extend(
        [
            "",
            "## 逐数据集",
            "",
            "| 数据集 | 模型 | cutoff | 权威/非权威/OOM | MAPE | Pairwise | regret | 预测/真实最优 MBS |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["per_dataset"]:
        lines.append(
            "| {dataset} | {model} | {cutoff} | {success}/{non_authoritative}/{oom} | {mape} | {pair} | {regret} | {selected}/{oracle} |".format(
                dataset=row["dataset_id"],
                model=row["model_id"],
                cutoff=row["cutoff_len"],
                success=row["state_counts"]["success_authoritative"],
                non_authoritative=row["state_counts"]["success_non_authoritative"],
                oom=row["state_counts"]["oom"],
                mape=_percent(row["throughput_mape"]),
                pair=_percent(row["scenario_equal_pairwise_accuracy"]),
                regret=_percent(row["top1_regret"]),
                selected=row["selected_mbs"] if row["selected_mbs"] is not None else "—",
                oracle=row["oracle_mbs"] if row["oracle_mbs"] is not None else "—",
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_report() -> dict[str, Any]:
    rows, prediction_report = _collect_rows()
    design = read_json(DESIGN)
    frozen = read_json(MODEL_ARTIFACT)
    supported_cutoffs = set(
        int(value)
        for value in frozen["frozen_model"]["training_support"]["cutoff_lens"]
    )
    rows_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_dataset[str(row["dataset_id"])].append(row)
    complete_dataset_ids = {
        dataset_id
        for dataset_id, current in rows_by_dataset.items()
        if len(current) == 3
        and all(row["state"] == "success_authoritative" for row in current)
    }
    strict_complete_dataset_ids = {
        dataset_id
        for dataset_id in complete_dataset_ids
        if int(rows_by_dataset[dataset_id][0]["cutoff_len"]) in supported_cutoffs
    }
    strict_rows = [
        row for row in rows if str(row["dataset_id"]) in strict_complete_dataset_ids
    ]
    complete_rows = [
        row for row in rows if str(row["dataset_id"]) in complete_dataset_ids
    ]
    cutoff_ood_rows = [
        row
        for row in complete_rows
        if int(row["cutoff_len"]) not in supported_cutoffs
    ]

    by_ratio_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ratio_bin[str(row["ratio_bin"])].append(row)
        by_model[str(row["model_id"])].append(row)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_contract": {
            "target": "effective_tokens_per_second",
            "frozen_v5_refit_from_new_outcomes": False,
            "new_outcomes_used_as_labels_only": True,
            "all_queue_jobs_included_without_outcome_selection": True,
            "gpu_jobs_launched_by_evaluator": False,
            "queue_mutated_by_evaluator": False,
            "prediction_pre_registered_before_outcomes": False,
            "temporal_model_holdout": True,
            "ranking_conditional_on_memory_safe_successes": True,
            "oom_throughput_imputed": False,
            "primary_scope": (
                "New dataset ids with H800, dense in-domain Qwen3 models, LoRA, "
                "GBS64, 2 GPUs, ZeRO-2, GC=false, packing=false and a cutoff "
                "observed in frozen V5 fitting data"
            ),
        },
        "bindings": {
            "frozen_v5": {
                "path": str(MODEL_ARTIFACT.resolve()),
                "file_sha256": sha256_file(MODEL_ARTIFACT),
                "generated_at_utc": frozen["generated_at_utc"],
            },
            "queue": {
                "path": str(QUEUE.resolve()),
                "file_sha256": sha256_file(QUEUE),
                "rows": len(_read_jsonl(QUEUE)),
            },
            "design": {
                "path": str(DESIGN.resolve()),
                "file_sha256": sha256_file(DESIGN),
            },
            "additional_dataset_profiles": {
                "path": str(PROFILE_DIR.resolve()),
                "files": len(list(PROFILE_DIR.glob("*.qwen3_nothink.jsonl"))),
            },
            "prediction_report_sha256": prediction_report["report_sha256"],
        },
        "support_partition": {
            "frozen_supported_cutoffs": sorted(supported_cutoffs),
            "strict_primary_candidate_rows": len(strict_rows),
            "strict_primary_dataset_rows": len({row["dataset_id"] for row in strict_rows}),
            "complete_authoritative_candidate_rows": len(complete_rows),
            "complete_authoritative_dataset_rows": len(complete_dataset_ids),
            "cutoff_ood_candidate_rows": len(cutoff_ood_rows),
            "cutoff_ood_dataset_rows": len({row["dataset_id"] for row in cutoff_ood_rows}),
        },
        "evaluation": {
            "strict_dataset_only_primary": _metrics(strict_rows),
            "all_new_datasets": _metrics(rows),
            "complete_authoritative_datasets_all_cutoffs": _metrics(complete_rows),
            "complete_authoritative_cutoff_outside_frozen_support": _metrics(
                cutoff_ood_rows
            ),
            "by_ratio_bin": {
                name: _metrics(current) for name, current in sorted(by_ratio_bin.items())
            },
            "by_model": {
                name: _metrics(current) for name, current in sorted(by_model.items())
            },
        },
        "per_dataset": _per_dataset(rows, design),
        "rows": rows,
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_report()
    write_json(args.output, report)
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    primary = report["evaluation"]["strict_dataset_only_primary"]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "markdown": str(args.markdown.resolve()),
                "report_sha256": report["report_sha256"],
                "strict_primary": {
                    "candidate_rows": primary["candidate_rows"],
                    "dataset_rows": primary["dataset_rows"],
                    "state_counts": primary["state_counts"],
                    "absolute": primary["absolute"],
                    "ranking": {
                        key: value
                        for key, value in primary["ranking"].items()
                        if key != "details"
                    },
                    "static_work_reconstruction": primary[
                        "static_work_reconstruction"
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
