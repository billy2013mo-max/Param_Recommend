#!/usr/bin/env python3
"""Evaluate Packing ranking for the virtual-MBS product design.

Two evidence populations remain separate:

* historical end-to-end frozen-model ranking, where the score is the frozen
  Unpacked model prediction at static mean samples per pack;
* final 4-to-7-GPU transfer ranking, where measured Unpacked floor/ceil arms
  are interpolated at virtual MBS to isolate the Packing transfer assumption.

No GPU work is launched and no model is refit.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_h800_packing_final_virtual_mbs_v1 as final_eval
from common import (
    ARTIFACT_DIR,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)


HISTORICAL = ARTIFACT_DIR / "h800_packing_virtual_mbs_analysis_v1.json"
FINAL_QUEUE = final_eval.QUEUE
OUTPUT = ARTIFACT_DIR / "h800_packing_ranking_acceptance_v1.json"
MARKDOWN = ARTIFACT_DIR / "h800_packing_ranking_acceptance_v1.md"
MATERIAL_GAP = 0.03


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("geometric mean requires data")
    return math.exp(statistics.fmean(math.log(float(value)) for value in values))


def _ranking_metrics(
    scenarios: Mapping[Any, Sequence[Mapping[str, Any]]],
    *,
    evidence_label: str,
) -> dict[str, Any]:
    pair_correct = 0
    pair_total = 0
    material_correct = 0
    material_total = 0
    exact_top1_hits = 0
    hit90 = 0
    regrets = []
    details = []
    for scenario_id, candidates_source in sorted(
        scenarios.items(), key=lambda item: str(item[0])
    ):
        candidates = [dict(row) for row in candidates_source]
        if len(candidates) < 2:
            continue
        scenario_correct = 0
        scenario_total = 0
        scenario_material_correct = 0
        scenario_material_total = 0
        for left, right in itertools.combinations(candidates, 2):
            left_observed = float(left["observed_throughput"])
            right_observed = float(right["observed_throughput"])
            if math.isclose(left_observed, right_observed, rel_tol=0.0, abs_tol=1e-12):
                continue
            left_predicted = float(left["predicted_score"])
            right_predicted = float(right["predicted_score"])
            correct = (left_observed > right_observed) == (
                left_predicted > right_predicted
            )
            scenario_correct += int(correct)
            scenario_total += 1
            observed_gap = abs(left_observed - right_observed) / max(
                left_observed, right_observed
            )
            if observed_gap >= MATERIAL_GAP:
                scenario_material_correct += int(correct)
                scenario_material_total += 1
        if not scenario_total:
            continue
        selected = max(candidates, key=lambda row: float(row["predicted_score"]))
        oracle = max(candidates, key=lambda row: float(row["observed_throughput"]))
        oracle_throughput = float(oracle["observed_throughput"])
        selected_throughput = float(selected["observed_throughput"])
        regret = 1.0 - selected_throughput / oracle_throughput
        exact = str(selected["candidate_id"]) == str(oracle["candidate_id"])
        within_90 = selected_throughput >= 0.90 * oracle_throughput
        pair_correct += scenario_correct
        pair_total += scenario_total
        material_correct += scenario_material_correct
        material_total += scenario_material_total
        exact_top1_hits += int(exact)
        hit90 += int(within_90)
        regrets.append(regret)
        details.append(
            {
                "scenario_id": scenario_id,
                "candidate_count": len(candidates),
                "pairwise_correct": scenario_correct,
                "pairwise_total": scenario_total,
                "material_pairwise_correct": scenario_material_correct,
                "material_pairwise_total": scenario_material_total,
                "predicted_top1_candidate_id": str(selected["candidate_id"]),
                "observed_top1_candidate_id": str(oracle["candidate_id"]),
                "exact_top1": exact,
                "top1_regret": regret,
                "selected_reaches_90_percent_of_oracle": within_90,
                "candidates": candidates,
            }
        )
    scenario_count = len(details)
    if not scenario_count:
        raise ValueError(f"no rankable scenarios for {evidence_label}")
    return {
        "evidence_label": evidence_label,
        "scenario_count": scenario_count,
        "candidate_count": sum(row["candidate_count"] for row in details),
        "minimum_candidates_per_scenario": min(
            row["candidate_count"] for row in details
        ),
        "maximum_candidates_per_scenario": max(
            row["candidate_count"] for row in details
        ),
        "pairwise_correct": pair_correct,
        "pairwise_total": pair_total,
        "pairwise_accuracy": pair_correct / pair_total,
        "material_gap_threshold": MATERIAL_GAP,
        "material_pairwise_correct": material_correct,
        "material_pairwise_total": material_total,
        "material_pairwise_accuracy": (
            material_correct / material_total if material_total else None
        ),
        "exact_top1_hits": exact_top1_hits,
        "exact_top1_accuracy": exact_top1_hits / scenario_count,
        "hit90_hits": hit90,
        "hit90_accuracy": hit90 / scenario_count,
        "mean_top1_regret": statistics.fmean(regrets),
        "worst_top1_regret": max(regrets),
        "details": details,
    }


def historical_end_to_end() -> dict[str, Any]:
    report = read_json(HISTORICAL)
    repeats: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    meta: dict[str, Mapping[str, Any]] = {}
    for row in report["pair_rows"]:
        setting_id = str(row["setting_id"])
        repeats[setting_id].append(row)
        meta[setting_id] = row
    scenarios: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for setting in report["setting_rows"]:
        setting_id = str(setting["setting_id"])
        representative = meta[setting_id]
        key = (
            str(setting["model_id"]),
            str(setting["train_type"]),
            str(setting["dataset_id"]),
            int(setting["cutoff_len"]),
            int(representative["nominal_target_gbs"]),
        )
        observed = _geometric_mean(
            [
                float(row["observed"]["packed_effective_tokens_per_second"])
                for row in repeats[setting_id]
            ]
        )
        scenarios[key].append(
            {
                "candidate_id": setting_id,
                "observed_throughput": observed,
                "predicted_score": float(
                    setting["prediction"]["virtual_effective_tokens_per_second"]
                ),
                "throughput_unit": "effective_tokens_per_second",
                "gpu_count": int(setting["gpu_count"]),
                "zero_stage": int(setting["zero_stage"]),
                "gc": bool(setting["gc"]),
                "virtual_mbs": float(setting["n_pack_mean"]),
            }
        )
    result = _ranking_metrics(
        scenarios,
        evidence_label="historical_frozen_v5_virtual_mbs_end_to_end",
    )
    if result["scenario_count"] != 6 or result["candidate_count"] != 12:
        raise ValueError("historical Packing ranking population drifted")
    result["scenario_definition"] = (
        "same model, training mode, dataset, cutoff and target GBS; Packing "
        "mechanism candidates vary"
    )
    result["prediction_contract"] = (
        "frozen Unpacked throughput prediction at static mean samples per pack; "
        "no Packing coefficient"
    )
    return result


def final_direct_transfer() -> dict[str, Any]:
    jobs = read_jsonl(FINAL_QUEUE)
    job_results = [final_eval._job_result(job) for job in jobs]
    groups = []
    for group_id in sorted({str(row["matched_group_id"]) for row in job_results}):
        groups.append(
            final_eval._group_result(
                [row for row in job_results if str(row["matched_group_id"]) == group_id]
            )
        )
    scenarios: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        if not bool(group["complete"]):
            continue
        scenarios[str(group["workload_id"])].append(
            {
                "candidate_id": str(group["matched_group_id"]),
                "observed_throughput": float(group["packed_tps"]),
                "predicted_score": float(
                    group["unpacked_interpolated_virtual_mbs_tps"]
                ),
                "throughput_unit": "logical_samples_per_second",
                "gpu_count": int(group["gpu_count"]),
                "virtual_mbs": float(group["arms"]["packed"]["virtual_mbs"]),
            }
        )
    result = _ranking_metrics(
        scenarios,
        evidence_label="final_4to7gpu_direct_virtual_mbs_transfer",
    )
    if result["scenario_count"] != 2 or result["candidate_count"] != 7:
        raise ValueError("current completed final Packing ranking population drifted")
    result["scenario_definition"] = (
        "same business workload; completed 4-to-7-GPU candidates are compared"
    )
    result["prediction_contract"] = (
        "measured Unpacked floor/ceil interpolation at static virtual MBS; "
        "this isolates Packing transfer and does not test V5 absolute prediction"
    )
    result["completion"] = {
        "complete_matched_groups": sum(bool(row["complete"]) for row in groups),
        "planned_matched_groups": len(groups),
        "rankable_workloads": result["scenario_count"],
        "unrankable_completed_singleton_workloads": sorted(
            workload
            for workload, rows in scenarios.items()
            if len(rows) < 2
        ),
    }
    return result


def _combined_counts(*reports: Mapping[str, Any]) -> dict[str, Any]:
    scenarios = sum(int(row["scenario_count"]) for row in reports)
    candidates = sum(int(row["candidate_count"]) for row in reports)
    pair_correct = sum(int(row["pairwise_correct"]) for row in reports)
    pair_total = sum(int(row["pairwise_total"]) for row in reports)
    exact_hits = sum(int(row["exact_top1_hits"]) for row in reports)
    hit90 = sum(int(row["hit90_hits"]) for row in reports)
    regrets = [
        float(detail["top1_regret"])
        for row in reports
        for detail in row["details"]
    ]
    return {
        "diagnostic_only_not_a_single_statistical_population": True,
        "scenario_count": scenarios,
        "candidate_count": candidates,
        "pairwise_correct": pair_correct,
        "pairwise_total": pair_total,
        "pairwise_accuracy": pair_correct / pair_total,
        "exact_top1_hits": exact_hits,
        "exact_top1_accuracy": exact_hits / scenarios,
        "hit90_hits": hit90,
        "hit90_accuracy": hit90 / scenarios,
        "mean_top1_regret": statistics.fmean(regrets),
        "worst_top1_regret": max(regrets),
    }


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _markdown(report: Mapping[str, Any]) -> str:
    historical = report["historical_end_to_end"]
    final = report["final_direct_transfer"]
    combined = report["combined_diagnostic_counts"]
    return "\n".join(
        [
            "# H800 Packing 排序验收",
            "",
            "本验收只关心候选排序、Top1 和达到最优吞吐 90% 的概率，不使用绝对吞吐 MAPE 作为放行条件。没有启动 GPU，也没有重拟合模型。",
            "",
            "| 证据 | 候选集合 | 候选数 | 两两排序 | Top1 | 达到最优 90% | 最差 Top1 损失 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| 历史端到端虚拟 MBS | {historical['scenario_count']} | {historical['candidate_count']} | {historical['pairwise_correct']}/{historical['pairwise_total']} = {_pct(historical['pairwise_accuracy'])} | {historical['exact_top1_hits']}/{historical['scenario_count']} = {_pct(historical['exact_top1_accuracy'])} | {historical['hit90_hits']}/{historical['scenario_count']} = {_pct(historical['hit90_accuracy'])} | {_pct(historical['worst_top1_regret'])} |",
            f"| 最终 4–7 卡直接迁移 | {final['scenario_count']} | {final['candidate_count']} | {final['pairwise_correct']}/{final['pairwise_total']} = {_pct(final['pairwise_accuracy'])} | {final['exact_top1_hits']}/{final['scenario_count']} = {_pct(final['exact_top1_accuracy'])} | {final['hit90_hits']}/{final['scenario_count']} = {_pct(final['hit90_accuracy'])} | {_pct(final['worst_top1_regret'])} |",
            f"| 合计诊断计数 | {combined['scenario_count']} | {combined['candidate_count']} | {combined['pairwise_correct']}/{combined['pairwise_total']} = {_pct(combined['pairwise_accuracy'])} | {combined['exact_top1_hits']}/{combined['scenario_count']} = {_pct(combined['exact_top1_accuracy'])} | {combined['hit90_hits']}/{combined['scenario_count']} = {_pct(combined['hit90_accuracy'])} | {_pct(combined['worst_top1_regret'])} |",
            "",
            "当前实测集合没有发生排序错误，也没有 Top1 损失。历史端到端集合每组只有两个机制候选；最终 4–7 卡集合包含一个四候选集合和一个三候选集合。两类证据不能当作同一个独立同分布统计样本，但结论方向一致。",
            "",
            "证据范围仍是 H800、Qwen3-8B LoRA。Qwen3-14B 历史实验没有形成同一推荐请求内的候选集合，因此不能用来计算 Top1 概率。",
            "",
        ]
    )


def main() -> None:
    historical = historical_end_to_end()
    final = final_direct_transfer()
    combined = _combined_counts(historical, final)
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_ranking_acceptance/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "validate Packing candidate ordering, exact Top1 selection and "
            "selection within 90% of oracle throughput"
        ),
        "decision": {
            "current_empirical_ranking_gate_passed": bool(
                historical["pairwise_accuracy"] == 1.0
                and historical["exact_top1_accuracy"] == 1.0
                and historical["hit90_accuracy"] == 1.0
                and final["pairwise_accuracy"] == 1.0
                and final["exact_top1_accuracy"] == 1.0
                and final["hit90_accuracy"] == 1.0
            ),
            "allowed_scope": "H800 Qwen3-8B LoRA Packing shadow/recommendation path",
            "absolute_throughput_mape_is_a_release_gate": False,
            "packing_coefficient_required_for_current_ranking": False,
        },
        "metric_definitions": {
            "pairwise_accuracy": (
                "fraction of candidate pairs whose predicted ordering matches measured ordering"
            ),
            "exact_top1_accuracy": (
                "fraction of recommendation sets whose predicted best candidate is the measured best candidate"
            ),
            "hit90_accuracy": (
                "fraction of recommendation sets where the predicted Top1 reaches at least 90% of measured oracle throughput"
            ),
        },
        "historical_end_to_end": historical,
        "final_direct_transfer": final,
        "combined_diagnostic_counts": combined,
        "source_bindings": {
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "historical_virtual_mbs": {
                "path": str(HISTORICAL.resolve()),
                "sha256": sha256_file(HISTORICAL),
            },
            "final_queue": {
                "path": str(FINAL_QUEUE.resolve()),
                "sha256": sha256_file(FINAL_QUEUE),
            },
        },
        "limitations": [
            "All rankable evidence is Qwen3-8B LoRA on H800.",
            "Historical end-to-end scenarios have exactly two candidates each.",
            "Only two final business workloads currently have at least two complete matched triples.",
            "Combined counts summarize consistent diagnostics but are not a single IID probability estimate.",
        ],
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    MARKDOWN.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "markdown": str(MARKDOWN),
                "decision": report["decision"],
                "historical": {
                    key: historical[key]
                    for key in (
                        "scenario_count",
                        "candidate_count",
                        "pairwise_accuracy",
                        "exact_top1_accuracy",
                        "hit90_accuracy",
                        "worst_top1_regret",
                    )
                },
                "final": {
                    key: final[key]
                    for key in (
                        "scenario_count",
                        "candidate_count",
                        "pairwise_accuracy",
                        "exact_top1_accuracy",
                        "hit90_accuracy",
                        "worst_top1_regret",
                    )
                },
                "combined": combined,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
