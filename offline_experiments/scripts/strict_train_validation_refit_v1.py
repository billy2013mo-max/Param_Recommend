#!/usr/bin/env python3
"""Run one source/scenario-disjoint refit for memory and throughput.

This is a retrospective CPU-only diagnostic.  It does not mutate the frozen
memory or throughput artifacts and it does not launch GPU jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import benchmark_h800_memory_center_models_v1 as memory_benchmark
import fit_rank_first_throughput_challenger_v1 as rank_first
import freeze_h800_unified_bounded_memory_v3 as memory_v3
import h800_unified_bounded_memory_v3_data as memory_data
import structured_throughput_modeling as throughput_v5
from common import ROOT, read_json, sha256_file, sha256_json, write_json, write_jsonl

SCHEMA = "sft_strict_train_validation_refit/v1"
IMPLEMENTATION_VERSION = "2026-08-11.single-source-scenario-holdout-v1"
SPLIT_SEED = "strict-refit-v1"
MEMORY_VALIDATION_FRACTION = 0.20
ZERO3_FACTOR_GRID = (1.0, 1.25, 1.35, 1.4, 1.45, 1.5, 1.55, 1.75, 2.0)
DEFAULT_OUTPUT_DIR = ROOT / "diagnostics" / "strict_train_validation_refit_v1"


MEMORY_CHALLENGERS: tuple[dict[str, Any], ...] = (
    dict(memory_v3.CENTER_CANDIDATE),
    {
        "candidate_id": "bounded_quadratic_anchor_scale_preselected_v1",
        "basis_kind": "bounded_quadratic",
        "feature_variant": "anchor_scale",
        "alpha": 0.3,
        "correction_shrinkage": 0.8,
        "huber_delta": 0.2,
        "source_weight_power": 1.0,
        "censored_constraint_weight": 1.0,
    },
)


def _hash_key(value: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}|{value}".encode()).hexdigest()


def _memory_split(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, Any]]:
    """Hold out 20% of sources separately in the large/small source strata.

    The stratum is determined only by source row count, not by target memory or
    success/OOM outcome.  This keeps the one 242-row connected component from
    either swallowing or disappearing from a plain row-balanced holdout.
    """

    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_source[str(row["source_id"])].append(row)
    strata = {
        "large_source": sorted(
            (source for source, rows in by_source.items() if len(rows) > 10),
            key=_hash_key,
        ),
        "small_source": sorted(
            (source for source, rows in by_source.items() if len(rows) <= 10),
            key=_hash_key,
        ),
    }
    held: set[str] = set()
    selected_by_stratum: dict[str, list[str]] = {}
    for name, sources in strata.items():
        count = max(1, math.ceil(len(sources) * MEMORY_VALIDATION_FRACTION))
        selected = sources[:count]
        selected_by_stratum[name] = selected
        held.update(selected)

    train = [row for row in records if str(row["source_id"]) not in held]
    validation = [row for row in records if str(row["source_id"]) in held]
    train_sources = {str(row["source_id"]) for row in train}
    validation_sources = {str(row["source_id"]) for row in validation}
    overlap = sorted(train_sources & validation_sources)
    if overlap:
        raise ValueError(f"memory source leakage: {overlap}")
    return train, validation, {
        "method": (
            "deterministic SHA-256 source holdout; 20% separately from sources "
            "with >10 and <=10 rows; stratum uses row count only"
        ),
        "seed": SPLIT_SEED,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_sources": len(train_sources),
        "validation_sources": len(validation_sources),
        "validation_sources_by_stratum": selected_by_stratum,
        "source_overlap": overlap,
        "train_exact": sum(row["state"] == "exact" for row in train),
        "train_oom": sum(row["state"] == "censored" for row in train),
        "validation_exact": sum(row["state"] == "exact" for row in validation),
        "validation_oom": sum(row["state"] == "censored" for row in validation),
    }


def _memory_experiment() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records, data_audit = memory_data.development_records()
    train, validation, split = _memory_split(records)

    risk_model = memory_benchmark._fit_model(train, memory_v3.RISK_CANDIDATE)
    risk_multiplier = memory_v3._calibrate_risk_multiplier(
        train, memory_v3._model_bytes(train, risk_model)
    )
    results: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for candidate in MEMORY_CHALLENGERS:
        candidate_id = str(candidate["candidate_id"])
        center = memory_benchmark._prediction_details(
            train,
            validation,
            candidate,
            fold_id="single_locked_source_validation",
        )
        details = memory_v3._decorate_admission(
            validation, center, risk_model, risk_multiplier
        )
        for row in details:
            prediction_rows.append({"predictor_id": candidate_id, **row})
        center_metrics = memory_benchmark._metrics(details)
        admission = memory_v3._admission_metrics(details)
        exact = [row for row in details if row["state"] == "exact"]
        worst = max(exact, key=lambda row: float(row["absolute_percentage_error"]))
        results[candidate_id] = {
            "model_label_zh": (
                "当前统一有界显存 V3 重拟合"
                if candidate_id == memory_v3.CENTER_CANDIDATE["candidate_id"]
                else "低复杂度二次残差重拟合"
            ),
            "candidate": candidate,
            "risk_head": "shared current V3 risk head refitted on training rows only",
            "risk_multiplier": risk_multiplier,
            "validation": {
                "center_source_equal_mape": center_metrics["source_equal_mape"],
                "center_exact_rows": center_metrics["exact_centres"],
                "center_exact_sources": center_metrics["exact_sources"],
                "safe_success_admitted": admission["admitted_safe_success_rows"],
                "safe_success_rows": admission["actual_safe_success_rows"],
                "safe_success_admission_rate": admission[
                    "safe_success_admission_rate"
                ],
                "oom_admitted": admission["admitted_oom_rows"],
                "oom_rows": admission["oom_rows"],
                "oom_admission_rate": admission["oom_admission_rate"],
                "unsafe_success_admitted": admission[
                    "admitted_unsafe_success_rows"
                ],
                "unsafe_success_rows": admission["actual_unsafe_success_rows"],
            },
            "worst_center_example": {
                "record_id": worst["record_id"],
                "source_id": worst["source_id"],
                "observed_reserved_gib": worst["observed_reserved_gib"],
                "predicted_reserved_gib": worst["predicted_reserved_gib"],
                "absolute_percentage_error": worst["absolute_percentage_error"],
            },
        }
    return {
        "data": {
            "rows": len(records),
            "sources": len({str(row["source_id"]) for row in records}),
            "audit": data_audit,
        },
        "split": split,
        "risk_candidate": dict(memory_v3.RISK_CANDIDATE),
        "models": results,
    }, prediction_rows


def _zero_stage(candidate: Mapping[str, Any]) -> int:
    return int((candidate["record"].get("selector") or {}).get("zero_stage") or 0)


def _gpu_count(candidate: Mapping[str, Any]) -> int:
    return int((candidate["record"].get("scenario") or {}).get("gpu_count") or 0)


def _throughput_metrics(
    entries: Sequence[tuple[Mapping[str, Any], float]],
) -> dict[str, Any]:
    by_group: dict[tuple[str, int], list[tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for candidate, prediction in entries:
        by_group[(str(candidate["scenario_id"]), _gpu_count(candidate))].append(
            (candidate, float(prediction))
        )

    pair_correct = 0
    pair_total = 0
    top1_hits = 0
    hit10 = 0
    comparable_groups = 0
    details: list[dict[str, Any]] = []
    for (scenario_id, gpu_count), rows in sorted(by_group.items()):
        if len(rows) < 2:
            continue
        comparable_groups += 1
        for left, right in itertools.combinations(rows, 2):
            observed_delta = float(left[0]["observed_log_throughput"]) - float(
                right[0]["observed_log_throughput"]
            )
            if abs(observed_delta) <= 1.0e-12:
                continue
            predicted_delta = left[1] - right[1]
            pair_correct += int((observed_delta > 0.0) == (predicted_delta > 0.0))
            pair_total += 1
        selected = max(rows, key=lambda item: item[1])
        oracle = max(rows, key=lambda item: float(item[0]["observed_log_throughput"]))
        exact = selected[0] is oracle[0]
        regret = max(
            0.0,
            1.0
            - math.exp(
                float(selected[0]["observed_log_throughput"])
                - float(oracle[0]["observed_log_throughput"])
            ),
        )
        top1_hits += int(exact)
        hit10 += int(regret <= 0.10 + 1.0e-12)
        details.append(
            {
                "scenario_id": scenario_id,
                "gpu_count": gpu_count,
                "candidate_rows": len(rows),
                "selected_candidate_key": selected[0]["candidate_key"],
                "oracle_candidate_key": oracle[0]["candidate_key"],
                "selected_observed_throughput": math.exp(
                    float(selected[0]["observed_log_throughput"])
                ),
                "oracle_observed_throughput": math.exp(
                    float(oracle[0]["observed_log_throughput"])
                ),
                "top1_regret": regret,
                "exact_top1": exact,
                "hit_at_10_percent": regret <= 0.10 + 1.0e-12,
            }
        )
    return {
        "pairwise_correct": pair_correct,
        "pairwise_pairs": pair_total,
        "pairwise_accuracy": pair_correct / pair_total if pair_total else None,
        "exact_top1_hits": top1_hits,
        "comparable_scenario_gpu_groups": comparable_groups,
        "exact_top1_fraction": (
            top1_hits / comparable_groups if comparable_groups else None
        ),
        "hit10_hits": hit10,
        "hit10_fraction": hit10 / comparable_groups if comparable_groups else None,
        "details": details,
    }


def _correct_rank_entries(
    entries: Sequence[tuple[Mapping[str, Any], float, float, float]], factor: float
) -> list[tuple[Mapping[str, Any], float]]:
    return [
        (
            candidate,
            float(final_log) - (math.log(factor) if _zero_stage(candidate) == 3 else 0.0),
        )
        for candidate, final_log, _absolute_log, _rank_score in entries
    ]


def _select_zero3_factor(
    oof_entries: Sequence[tuple[Mapping[str, Any], float, float, float]],
) -> dict[str, Any]:
    evaluated = []
    for factor in ZERO3_FACTOR_GRID:
        metrics = _throughput_metrics(_correct_rank_entries(oof_entries, factor))
        evaluated.append(
            {
                "factor": factor,
                "pairwise_accuracy": metrics["pairwise_accuracy"],
                "exact_top1_fraction": metrics["exact_top1_fraction"],
                "hit10_fraction": metrics["hit10_fraction"],
            }
        )
    selected = max(
        evaluated,
        key=lambda row: (
            float(row["pairwise_accuracy"] or 0.0),
            -abs(float(row["factor"]) - 1.0),
        ),
    )
    return {
        "protocol": "selected on training-only five-fold out-of-fold predictions",
        "objective": "maximum pooled within-scenario-and-GPU pairwise accuracy",
        "tie_break": "smallest correction away from one",
        "selected_factor": selected["factor"],
        "candidates": evaluated,
    }


def _throughput_experiment() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    h800 = throughput_v5._load_h800(
        observation_path=ROOT / "artifacts" / "canonical_h800_observations.jsonl",
        theory_basis_path=ROOT / "artifacts" / "h800_theory_basis.json",
        inventory_path=ROOT / "artifacts" / "model_inventory.json",
        hardware_path=ROOT / "config" / "hardware.json",
        runtime_root=ROOT / "runtime",
    )
    train_scenarios = {
        str(row["scenario_id"]) for row in h800["strict_fit_candidates"]
    }
    validation_scenarios = {
        str(row["scenario_id"]) for row in h800["holdout_candidates"]
    }
    overlap = sorted(train_scenarios & validation_scenarios)
    if overlap:
        raise ValueError(f"throughput scenario leakage: {overlap}")

    hardware = read_json(ROOT / "config" / "hardware.json")
    memory_bytes = float(hardware["memory_bytes_reported_by_torch"])
    v5_profiles = throughput_v5.StaticDatasetProfiles(
        ROOT / "artifacts" / "dataset_profiles"
    )
    v5_train = throughput_v5._structured_candidates(
        h800["strict_fit_candidates"],
        card_id="h800",
        profiles=v5_profiles,
        hardware_memory_bytes=memory_bytes,
        role="strict_refit_train",
    )
    v5_validation = throughput_v5._structured_candidates(
        h800["holdout_candidates"],
        card_id="h800",
        profiles=v5_profiles,
        hardware_memory_bytes=memory_bytes,
        role="strict_refit_validation",
    )
    v5_model = throughput_v5._fit_model(v5_train)
    v5_entries = [
        (candidate, throughput_v5._predict_log_throughput(candidate, v5_model))
        for candidate in v5_validation
    ]
    v5_metrics = _throughput_metrics(v5_entries)

    rank_profiles = rank_first.InvariantStaticProfiles(
        [
            ROOT / "artifacts" / "dataset_profiles",
            ROOT / "artifacts" / "h800_lora_safety_stage2_v1" / "profiles",
        ]
    )
    rank_train = [
        rank_first._candidate_from_pooled(
            row,
            rank_profiles,
            hardware_memory_bytes=memory_bytes,
            source_role="strict_refit_train",
        )
        for row in h800["strict_fit_candidates"]
    ]
    rank_validation = [
        rank_first._candidate_from_pooled(
            row,
            rank_profiles,
            hardware_memory_bytes=memory_bytes,
            source_role="strict_refit_validation",
        )
        for row in h800["holdout_candidates"]
    ]
    folds = rank_first._scenario_folds(rank_train)
    absolute_selection = rank_first._select_absolute_alpha(rank_train, folds)
    absolute_alpha = float(absolute_selection["selected"]["alpha"])
    rank_selection = rank_first._select_rank_alpha(
        rank_train, folds, absolute_alpha=absolute_alpha
    )
    rank_alpha = float(rank_selection["selected"]["alpha"])
    oof_entries = rank_first._cross_validated_entries(
        rank_train,
        absolute_alpha=absolute_alpha,
        rank_alpha=rank_alpha,
        folds=folds,
    )
    factor_selection = _select_zero3_factor(oof_entries)
    factor = float(factor_selection["selected_factor"])
    absolute_head = rank_first._fit_absolute_head(rank_train, alpha=absolute_alpha)
    rank_head = rank_first._fit_rank_head(rank_train, alpha=rank_alpha)
    rank_entries_raw = rank_first._prediction_entries(
        rank_validation, absolute_head, rank_head
    )
    rank_entries = _correct_rank_entries(rank_entries_raw, factor)
    rank_metrics = _throughput_metrics(rank_entries)

    predictions: list[dict[str, Any]] = []
    for model_id, metrics in (
        ("structured_throughput_v5_refit", v5_metrics),
        ("rank_first_two_head_refit", rank_metrics),
    ):
        predictions.extend(
            {"predictor_id": model_id, **row} for row in metrics.pop("details")
        )

    def worst_example(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return dict(max(rows, key=lambda row: float(row["top1_regret"])))

    v5_rows = [
        row
        for row in predictions
        if row["predictor_id"] == "structured_throughput_v5_refit"
    ]
    rank_rows = [
        row
        for row in predictions
        if row["predictor_id"] == "rank_first_two_head_refit"
    ]
    return {
        "split": {
            "method": "repository-locked complete scenario holdout",
            "train_candidates": len(v5_train),
            "train_scenarios": len(train_scenarios),
            "validation_candidates": len(v5_validation),
            "validation_scenarios": len(validation_scenarios),
            "scenario_overlap": overlap,
        },
        "models": {
            "structured_throughput_v5_refit": {
                "model_label_zh": "结构化吞吐 V5 重拟合",
                "validation": v5_metrics,
                "worst_selection_example": worst_example(v5_rows),
            },
            "rank_first_two_head_refit": {
                "model_label_zh": "排序优先双头重拟合",
                "training_selection": {
                    "absolute_alpha": absolute_alpha,
                    "rank_alpha": rank_alpha,
                    "zero3_factor": factor_selection,
                    "validation_used": False,
                },
                "validation": rank_metrics,
                "worst_selection_example": worst_example(rank_rows),
            },
        },
    }, predictions


def _ratio(value: int, total: int) -> str:
    return f"{value}/{total} = {value / total:.2%}" if total else "无样本"


def _render_markdown(report: Mapping[str, Any]) -> str:
    memory = report["memory"]
    throughput = report["throughput"]
    lines = [
        "# 严格训练集/验证集重新拟合结果",
        "",
        "> 这是已有数据上的回顾性单次验证，不是新的盲测。冻结产物没有改动。",
        "",
        "## 唯一切分",
        "",
        f"- 显存：训练 {memory['split']['train_rows']} 条/{memory['split']['train_sources']} 个来源；验证 {memory['split']['validation_rows']} 条/{memory['split']['validation_sources']} 个来源；来源重叠 {len(memory['split']['source_overlap'])}。",
        f"- 吞吐：训练 {throughput['split']['train_candidates']} 个候选/{throughput['split']['train_scenarios']} 个场景；验证 {throughput['split']['validation_candidates']} 个候选/{throughput['split']['validation_scenarios']} 个场景；场景重叠 {len(throughput['split']['scenario_overlap'])}。",
        "",
        "## 显存验证指标",
        "",
        "| 模型 | 中心来源等权 MAPE | 安全放行率 | OOM 放行率 |",
        "|---|---:|---:|---:|",
    ]
    for model in memory["models"].values():
        metrics = model["validation"]
        lines.append(
            "| {label} | {mape:.2%} | {safe} | {oom} |".format(
                label=model["model_label_zh"],
                mape=metrics["center_source_equal_mape"],
                safe=_ratio(
                    metrics["safe_success_admitted"], metrics["safe_success_rows"]
                ),
                oom=_ratio(metrics["oom_admitted"], metrics["oom_rows"]),
            )
        )
    current_memory = memory["models"][memory_v3.CENTER_CANDIDATE["candidate_id"]][
        "validation"
    ]
    lines.extend(
        [
            "",
            "安全补充：当前显存 V3 仍把 {admitted}/{rows} 条超过 95% 安全线的成功记录放行；这不计入安全样本放行率，也不能被 0 OOM 放行掩盖。".format(
                admitted=current_memory["unsafe_success_admitted"],
                rows=current_memory["unsafe_success_rows"],
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## 吞吐验证指标",
            "",
            "排序按同一场景、同一卡数内的候选对计算；Top1 和 10% 命中率按完整场景-卡数组计算。",
            "",
            "| 模型 | 排序准确率 | 精确 Top1 率 | 10% 命中率 |",
            "|---|---:|---:|---:|",
        ]
    )
    for model in throughput["models"].values():
        metrics = model["validation"]
        lines.append(
            "| {label} | {ranking} | {top1} | {hit10} |".format(
                label=model["model_label_zh"],
                ranking=_ratio(metrics["pairwise_correct"], metrics["pairwise_pairs"]),
                top1=_ratio(
                    metrics["exact_top1_hits"],
                    metrics["comparable_scenario_gpu_groups"],
                ),
                hit10=_ratio(
                    metrics["hit10_hits"],
                    metrics["comparable_scenario_gpu_groups"],
                ),
            )
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 数学定义：10% 命中表示所选配置实测吞吐不低于该组最优实测吞吐的 90%。",
            "- 当前实验观察：表中 Top1 率和 10% 命中率是验证集经验频率，不是已校准的线上概率。",
            "- 工程假设：未来请求与这次按来源、按场景留出的验证分布接近。",
            "",
        ]
    )
    return "\n".join(lines)


def build_report() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    memory, memory_predictions = _memory_experiment()
    throughput, throughput_predictions = _throughput_experiment()
    inputs = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (
            ROOT / "artifacts" / "canonical_h800_observations.jsonl",
            ROOT / "artifacts" / "h800_theory_basis.json",
            ROOT / "artifacts" / "model_inventory.json",
            ROOT / "config" / "hardware.json",
        )
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "retrospective_single_validation_complete",
        "analysis_only": True,
        "frozen_artifacts_mutated": False,
        "validation_used_for_model_or_threshold_selection": False,
        "memory": memory,
        "throughput": throughput,
        "inputs": inputs,
    }
    report["report_sha256"] = sha256_json(report)
    return report, memory_predictions, throughput_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    report, memory_predictions, throughput_predictions = build_report()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "report.json", report)
    write_jsonl(args.output_dir / "memory_validation_predictions.jsonl", memory_predictions)
    write_jsonl(
        args.output_dir / "throughput_validation_predictions.jsonl",
        throughput_predictions,
    )
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
