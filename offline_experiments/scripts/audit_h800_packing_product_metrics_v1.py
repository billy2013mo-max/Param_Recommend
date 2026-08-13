#!/usr/bin/env python3
"""Audit the currently measurable H800 Packing product metrics.

The audit keeps three evidence populations separate:

* source-group-held-out Packing rows from the frozen unified memory v3 fit;
* a retrospective replay of the same frozen v3 model on Packing boundary jobs;
* profile-group-held-out throughput predictions for Packed-only candidate sets.

The populations must not be pooled: the first has no Packing OOM, while the
second has Packing OOM but is retrospective.  The throughput population has
only two candidates per scenario and therefore remains diagnostic.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fit_h800_unified_resource_partial_v1 as unified_fit
import h800_unified_bounded_memory_v3_data as memory_v3_data
from common import (
    ARTIFACT_DIR,
    MATRIX_DIR,
    ROOT,
    percentile,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from h800_unified_bounded_memory_model import load_artifact, predict_records

MEMORY_V3 = ARTIFACT_DIR / "h800_unified_bounded_memory_candidate_v3.json"
MEMORY_OOF = (
    ROOT
    / "diagnostics"
    / "h800_unified_bounded_memory_v3_20260810"
    / "nested_admission_predictions.jsonl"
)
BOUNDARY_REPORT = (
    ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_evaluation_v1.json"
)
BOUNDARY_PARENT_QUEUE = MATRIX_DIR / "h800_packing_memory_boundary_stage1_v1.jsonl"
BOUNDARY_RESUME_QUEUE = (
    MATRIX_DIR / "h800_packing_memory_boundary_stage1_resume_v2.jsonl"
)
THROUGHPUT_BASE = ARTIFACT_DIR / "h800_packing_virtual_mbs_analysis_v1.json"
THROUGHPUT_RESIDUAL = ARTIFACT_DIR / "h800_packing_shared_physical_throughput_v1.json"
OUTPUT = ARTIFACT_DIR / "h800_packing_product_metrics_v1.json"
MARKDOWN = ARTIFACT_DIR / "h800_packing_product_metrics_v1.md"


def _memory_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    evidence_label: str,
) -> dict[str, Any]:
    exact = [row for row in rows if str(row["state"]) == "exact"]
    oom = [row for row in rows if str(row["state"]) == "censored"]
    by_source: dict[str, list[float]] = defaultdict(list)
    for row in exact:
        by_source[str(row["source_id"])].append(float(row["absolute_percentage_error"]))
    safe = [
        row
        for row in exact
        if float(row["observed_reserved_bytes"]) <= float(row["safe_limit_bytes"])
    ]
    unsafe = [row for row in exact if row not in safe]
    errors = [float(row["absolute_percentage_error"]) for row in exact]
    safe_admitted = sum(bool(row["admitted"]) for row in safe)
    oom_admitted = sum(bool(row["admitted"]) for row in oom)
    return {
        "evidence_label": evidence_label,
        "rows": len(rows),
        "exact_success_rows": len(exact),
        "profile_or_source_groups": len(by_source),
        "center_source_equal_mape": (
            statistics.fmean(statistics.fmean(values) for values in by_source.values())
            if by_source
            else None
        ),
        "center_row_mape": statistics.fmean(errors) if errors else None,
        "center_p90_ape": percentile(errors, 90) if errors else None,
        "actual_safe_success_rows": len(safe),
        "admitted_safe_success_rows": safe_admitted,
        "safe_success_admission_rate": (safe_admitted / len(safe) if safe else None),
        "actual_unsafe_success_rows": len(unsafe),
        "admitted_unsafe_success_rows": sum(bool(row["admitted"]) for row in unsafe),
        "oom_rows": len(oom),
        "admitted_oom_rows": oom_admitted,
        "oom_admission_rate": oom_admitted / len(oom) if oom else None,
    }


def memory_source_group_oof() -> dict[str, Any]:
    rows = [row for row in read_jsonl(MEMORY_OOF) if bool(row.get("packing"))]
    if len(rows) != 6:
        raise ValueError(f"expected six Packing memory OOF rows, found {len(rows)}")
    if any(str(row["state"]) != "exact" for row in rows):
        raise ValueError("current Packing memory OOF population unexpectedly has OOM")
    result = _memory_metrics(
        rows,
        evidence_label="nested_leave_one_source_out_development",
    )
    result["rows_detail"] = rows
    result["interpretation"] = (
        "valid for center error and safe-success admission on the six held-source "
        "Packing successes; OOM admission is undefined because oom_rows is zero"
    )
    return result


def _boundary_unique_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = {
        str(row["job_id"]): row
        for path in (BOUNDARY_PARENT_QUEUE, BOUNDARY_RESUME_QUEUE)
        for row in read_jsonl(path)
    }
    report = read_json(BOUNDARY_REPORT)
    selected: list[dict[str, Any]] = []
    seen_chains: set[str] = set()
    for row in report["rows"]:
        if not bool(row["calibration_eligible"]) or not bool(row["packing"]):
            continue
        chain_id = str(row["chain_id"])
        if chain_id in seen_chains:
            continue
        seen_chains.add(chain_id)
        selected.append(dict(row))
    if len(selected) != 4:
        raise ValueError(
            f"expected four unique eligible Packing boundary units, found {len(selected)}"
        )

    inventory, model_by_id, capacity_bytes = memory_v3_data._inventory()
    profile_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    records = []
    for outcome in selected:
        job = copy.deepcopy(jobs[str(outcome["job_id"])])
        job["packing_contract"] = {
            "expected_samples_per_pack": float(job["n_pack_mean"])
        }
        reference, features = unified_fit._current_features(
            job,
            model_by_id=model_by_id,
            fixed_lora=inventory["fixed_lora"],
            capacity_bytes=capacity_bytes,
            profile_cache=profile_cache,
        )
        records.append(
            {
                "record_id": str(outcome["job_id"]),
                "reference_bytes": reference,
                "features": features,
                "model_id": str(job["model_id"]),
                "train_type": str(job["train_type"]),
                "gpu_count": int(job["gpu_count"]),
                "zero_stage": int(job["zero_stage"]),
                "gc": bool(job["gc"]),
                "mbs": int(job["mbs"]),
                "cutoff_len": int(job["cutoff_len"]),
                "packing": True,
            }
        )
    return selected, records


def memory_boundary_replay() -> dict[str, Any]:
    outcomes, records = _boundary_unique_records()
    predictions = predict_records(records, load_artifact(MEMORY_V3))
    rows: list[dict[str, Any]] = []
    for outcome, prediction in zip(outcomes, predictions):
        state = "exact" if str(outcome["outcome"]) == "success" else "censored"
        row = {
            "record_id": str(outcome["job_id"]),
            "chain_id": str(outcome["chain_id"]),
            "source_id": str(outcome["chain_id"]),
            "state": state,
            "center_bytes": float(prediction["center_bytes"]),
            "admission_upper_bytes": float(prediction["admission_upper_bytes"]),
            "safe_limit_bytes": float(prediction["safe_limit_bytes"]),
            "admitted": bool(prediction["admitted"]),
        }
        if state == "exact":
            observed = float(outcome["observed_max_reserved_gib"]) * (2**30)
            row.update(
                {
                    "observed_reserved_bytes": observed,
                    "absolute_percentage_error": abs(
                        float(prediction["center_bytes"]) / observed - 1.0
                    ),
                }
            )
        else:
            row["censor_lower_bytes"] = float(prediction["safe_limit_bytes"])
        rows.append(row)
    result = _memory_metrics(
        rows,
        evidence_label="retrospective_frozen_v3_boundary_replay",
    )
    result["unique_physical_units"] = True
    result["rows_detail"] = rows
    result["interpretation"] = (
        "contains Packing OOM and is useful as a guard audit, but it is retrospective "
        "and cannot be pooled with the source-group OOF metrics"
    )
    return result


def _geometric_mean(values: Sequence[float]) -> float:
    return math.exp(statistics.fmean(math.log(float(value)) for value in values))


def throughput_packed_ranking() -> dict[str, Any]:
    base = read_json(THROUGHPUT_BASE)
    residual = read_json(THROUGHPUT_RESIDUAL)
    validation = residual["validation"]["retrospective_absolute_effective"]
    family = str(validation["selected_exploratory_candidate"])
    correction_rows = validation["ridge_candidates"][family]["oof_rows"]
    correction_by_setting = {
        str(row["setting_id"]): float(row["predicted_multiplier"])
        for row in correction_rows
    }

    pair_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    pair_meta: dict[str, Mapping[str, Any]] = {}
    for row in base["pair_rows"]:
        setting_id = str(row["setting_id"])
        pair_rows[setting_id].append(row)
        pair_meta[setting_id] = row

    scenarios: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for setting in base["setting_rows"]:
        setting_id = str(setting["setting_id"])
        meta = pair_meta[setting_id]
        key = (
            str(setting["model_id"]),
            str(setting["train_type"]),
            str(setting["dataset_id"]),
            int(setting["cutoff_len"]),
            int(meta["nominal_target_gbs"]),
        )
        observed = _geometric_mean(
            [
                float(row["observed"]["packed_effective_tokens_per_second"])
                for row in pair_rows[setting_id]
            ]
        )
        shared_prediction = float(
            setting["prediction"]["packed_physical_effective_tokens_per_second"]
        )
        scenarios[key].append(
            {
                "setting_id": setting_id,
                "observed_packed_effective_tokens_per_second": observed,
                "shared_physical_prediction": shared_prediction,
                "oof_residual_multiplier": correction_by_setting[setting_id],
                "predicted_packed_effective_tokens_per_second": (
                    shared_prediction * correction_by_setting[setting_id]
                ),
            }
        )

    eligible = {key: rows for key, rows in scenarios.items() if len(rows) >= 2}
    pairwise_correct = 0
    pairwise_rows = 0
    scenario_pairwise: list[float] = []
    regrets: list[float] = []
    hits: list[bool] = []
    details = []
    for key, candidates in sorted(eligible.items()):
        scenario_correct = 0
        scenario_pairs = 0
        for left, right in itertools.combinations(candidates, 2):
            left_observed = float(left["observed_packed_effective_tokens_per_second"])
            right_observed = float(right["observed_packed_effective_tokens_per_second"])
            if left_observed == right_observed:
                continue
            left_predicted = float(left["predicted_packed_effective_tokens_per_second"])
            right_predicted = float(
                right["predicted_packed_effective_tokens_per_second"]
            )
            correct = (left_observed > right_observed) == (
                left_predicted > right_predicted
            )
            scenario_correct += int(correct)
            scenario_pairs += 1
        if not scenario_pairs:
            continue
        pairwise_correct += scenario_correct
        pairwise_rows += scenario_pairs
        scenario_pairwise.append(scenario_correct / scenario_pairs)
        selected = max(
            candidates,
            key=lambda row: float(row["predicted_packed_effective_tokens_per_second"]),
        )
        oracle = max(
            float(row["observed_packed_effective_tokens_per_second"])
            for row in candidates
        )
        regret = 1.0 - (
            float(selected["observed_packed_effective_tokens_per_second"]) / oracle
        )
        hit = regret <= 0.10
        regrets.append(regret)
        hits.append(hit)
        details.append(
            {
                "scenario": {
                    "model_id": key[0],
                    "train_type": key[1],
                    "dataset_id": key[2],
                    "cutoff_len": key[3],
                    "target_gbs": key[4],
                },
                "candidates": candidates,
                "pairwise_rows": scenario_pairs,
                "pairwise_correct": scenario_correct,
                "selected_setting_id": str(selected["setting_id"]),
                "top1_regret": regret,
                "hit_at_10_percent": hit,
            }
        )
    if len(details) != 6 or pairwise_rows != 6:
        raise ValueError(
            "expected six two-candidate Packed-only ranking scenarios; "
            f"found {len(details)} scenarios and {pairwise_rows} pairs"
        )
    return {
        "evidence_label": "profile_group_oof_residual_packed_only_diagnostic",
        "scenario_definition": (
            "same model, train type, dataset, cutoff_len and nominal target GBS; "
            "only Packed candidates are compared"
        ),
        "residual_family": family,
        "scenarios": len(details),
        "candidates": sum(len(row["candidates"]) for row in details),
        "minimum_candidates_per_scenario": min(
            len(row["candidates"]) for row in details
        ),
        "maximum_candidates_per_scenario": max(
            len(row["candidates"]) for row in details
        ),
        "pairwise_rows": pairwise_rows,
        "pairwise_correct_rows": pairwise_correct,
        "pooled_pairwise_accuracy": pairwise_correct / pairwise_rows,
        "scenario_equal_pairwise_accuracy": statistics.fmean(scenario_pairwise),
        "scenario_equal_top1_regret": statistics.fmean(regrets),
        "hit_at_10_percent_rows": sum(hits),
        "scenario_equal_hit_at_10_percent": statistics.fmean(hits),
        "details": details,
        "interpretation": (
            "all six scenarios contain exactly two candidates and all are Qwen3-8B "
            "LoRA; these metrics are diagnostic rather than a broad candidate-lattice holdout"
        ),
    }


def _source_bindings() -> dict[str, Any]:
    paths = {
        "implementation": Path(__file__),
        "memory_v3": MEMORY_V3,
        "memory_oof": MEMORY_OOF,
        "boundary_report": BOUNDARY_REPORT,
        "boundary_parent_queue": BOUNDARY_PARENT_QUEUE,
        "boundary_resume_queue": BOUNDARY_RESUME_QUEUE,
        "throughput_base": THROUGHPUT_BASE,
        "throughput_residual": THROUGHPUT_RESIDUAL,
    }
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _format_rate(numerator: int, denominator: int, rate: float | None) -> str:
    if rate is None:
        return f"{numerator}/{denominator}（未定义）"
    return f"{numerator}/{denominator} = {rate:.2%}"


def _render_markdown(report: Mapping[str, Any]) -> str:
    oof = report["memory"]["source_group_oof"]
    boundary = report["memory"]["retrospective_boundary_replay"]
    throughput = report["throughput"]["packed_only_ranking"]
    return "\n".join(
        [
            "# H800 Packing 产品指标专项审计",
            "",
            "## 1. 现在的问题",
            "",
            "现有 Packing 报告主要验证绝对误差和残差稳定性，没有把产品需要的显存放行指标与吞吐排序指标放在同一处。",
            "",
            "## 2. 当前可严格报告的结果",
            "",
            "### 2.1 显存",
            "",
            "| 证据范围 | Packing 中心 MAPE | 安全放行率 | OOM 放行率 |",
            "|---|---:|---:|---:|",
            (
                f"| 开发集按源折外 | {oof['center_source_equal_mape']:.2%} | "
                f"{_format_rate(oof['admitted_safe_success_rows'], oof['actual_safe_success_rows'], oof['safe_success_admission_rate'])} | "
                f"{_format_rate(oof['admitted_oom_rows'], oof['oom_rows'], oof['oom_admission_rate'])} |"
            ),
            (
                f"| 冻结 v3 回放 Packing 边界（回顾性） | {boundary['center_source_equal_mape']:.2%} | "
                f"{_format_rate(boundary['admitted_safe_success_rows'], boundary['actual_safe_success_rows'], boundary['safe_success_admission_rate'])} | "
                f"{_format_rate(boundary['admitted_oom_rows'], boundary['oom_rows'], boundary['oom_admission_rate'])} |"
            ),
            "",
            "两行不能合并：按源折外行能评价泛化误差，但没有 Packing OOM；边界行有 OOM，但模型设计发生在这些结果之后，只能作为回顾性 guard 检查。",
            "",
            "### 2.2 吞吐排序",
            "",
            "| 指标 | 结果 |",
            "|---|---:|",
            f"| 两两排序准确率 | {throughput['pairwise_correct_rows']}/{throughput['pairwise_rows']} = {throughput['scenario_equal_pairwise_accuracy']:.2%} |",
            f"| Top-1 regret | {throughput['scenario_equal_top1_regret']:.2%} |",
            f"| Hit@10% | {throughput['hit_at_10_percent_rows']}/{throughput['scenarios']} = {throughput['scenario_equal_hit_at_10_percent']:.2%} |",
            "",
            f"这里有 {throughput['scenarios']} 个 Packing-only 场景、{throughput['candidates']} 个候选；每个场景都只有 2 个候选，并且全部是 Qwen3-8B LoRA。因此 100% 只能说明当前 6 对机制对照没有排错，不能代表完整候选空间。",
            "",
            "## 3. 一个具体例子",
            "",
            "Packing 显存按源折外的 6 个成功配置中，模型安全放行 4 个，因此安全放行率是 66.67%；同一总体没有 OOM，所以 OOM 放行率不能写成 0%。",
            "",
            "## 4. 准备怎么改",
            "",
            "下一轮 Packing 验收必须在同一批冻结预测中同时包含安全 success、OOM 和每场景至少 3 个可排序候选。只有这样才能产出一行可用于上线判断的五项联合指标。",
            "",
            "数学事实、当前实验观察和工程限制已在 JSON 的 interpretation_contract 中分开记录。",
            "",
        ]
    )


def build_report() -> dict[str, Any]:
    memory_oof = memory_source_group_oof()
    memory_boundary = memory_boundary_replay()
    throughput = throughput_packed_ranking()
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_product_metrics_audit/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "partial_metrics_available_not_release_ready",
        "analysis_only": True,
        "publishable": False,
        "production_model_mutated": False,
        "gpu_experiments_launched": False,
        "problem": (
            "report Packing memory center/admission and throughput ranking metrics "
            "with explicit evidence populations"
        ),
        "metric_definitions": {
            "memory_center_mape": (
                "mean absolute percentage error of predicted reserved-memory center "
                "on successful runs; OOM is right-censored and excluded"
            ),
            "safe_success_admission_rate": (
                "actual safe successful configurations admitted by predicted upper "
                "guard divided by all actual safe successful configurations"
            ),
            "oom_admission_rate": (
                "actual OOM configurations admitted by predicted upper guard divided "
                "by all actual OOM configurations; lower is better"
            ),
            "pairwise_accuracy": (
                "fraction of non-tied candidate pairs whose predicted throughput order "
                "matches observed throughput order"
            ),
            "top1_regret": (
                "one minus observed throughput of the predicted winner divided by "
                "observed throughput of the oracle winner"
            ),
            "hit_at_10_percent": "fraction of scenarios whose Top-1 regret is at most 10%",
        },
        "memory": {
            "source_group_oof": memory_oof,
            "retrospective_boundary_replay": memory_boundary,
            "one_joint_prospective_packing_row_available": False,
        },
        "throughput": {"packed_only_ranking": throughput},
        "interpretation_contract": {
            "mathematical_fact": (
                "OOM rows do not reveal an exact memory peak and therefore cannot "
                "enter center MAPE; a zero OOM denominator makes OOM admission undefined"
            ),
            "empirical_observation": (
                "the six source-group-held-out Packing successes have 8.70% center "
                "MAPE and 4/6 safe-success admission; the six two-candidate throughput "
                "scenarios are all ranked correctly"
            ),
            "engineering_limit": (
                "no single prospective Packing population currently contains both "
                "safe successes, OOMs, and a broad throughput candidate lattice"
            ),
        },
        "decision": {
            "automatic_packing_recommendation_allowed": False,
            "merge_metrics_into_one_release_row": False,
            "reason": (
                "the OOM evidence is retrospective and the throughput ranking set has "
                "only six two-candidate Qwen3-8B LoRA scenarios"
            ),
        },
        "source_bindings": _source_bindings(),
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    report = build_report()
    write_json(OUTPUT, report)
    MARKDOWN.write_text(_render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "markdown": str(MARKDOWN),
                "memory_source_group_oof": {
                    key: report["memory"]["source_group_oof"][key]
                    for key in (
                        "center_source_equal_mape",
                        "safe_success_admission_rate",
                        "oom_admission_rate",
                    )
                },
                "memory_retrospective_boundary": {
                    key: report["memory"]["retrospective_boundary_replay"][key]
                    for key in (
                        "center_source_equal_mape",
                        "safe_success_admission_rate",
                        "oom_admission_rate",
                    )
                },
                "throughput": {
                    key: report["throughput"]["packed_only_ranking"][key]
                    for key in (
                        "scenario_equal_pairwise_accuracy",
                        "scenario_equal_top1_regret",
                        "scenario_equal_hit_at_10_percent",
                    )
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
