#!/usr/bin/env python3
"""Fit rank-first challenger v3: a ZeRO-3 mechanism-level intercept.

Why an intercept rather than an interaction term
------------------------------------------------
Two independent sweeps rejected ``zero3_x_log2_mbs`` --- the second one after
matched contrasts had removed the ZeRO/MBS confound, so the term was rejected
while fully identifiable.  The matched batch showed why: the frozen model's
error on ZeRO-3 is a near-constant multiplicative offset, not something that
scales with log2(MBS).  On the merged evidence challenger v1 predicts ZeRO-2
almost exactly (median ratio 0.98) but overestimates ZeRO-3 by 1.68--1.71x
regardless of micro-batch size.  A term of the wrong functional form cannot fix
a constant offset, which is precisely what both sweeps observed.

What this changes
-----------------
Exactly one scalar is added to the ranking head: a constant subtracted from the
log-throughput score of every ZeRO-3 candidate.  No new feature is introduced,
no existing coefficient is refitted, and the absolute head is untouched.  The
correction factor is selected by leave-one-scenario-out cross-validation over a
grid, never by inspecting the population it is scored on.

Scope limit carried from the matched batch: ZeRO-2 never exceeded 58% of the
product safety line in that design, so this establishes behaviour when ZeRO-2 is
memory-feasible.  It says nothing about the regime where ZeRO-2 does not fit,
which is where ZeRO-3 is actually used in production.

Offline and diagnostic: no GPU work, no queue mutation, no production profile,
and neither the frozen V5 artifact nor the v1 challenger artifact is modified.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sys

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import ARTIFACT_DIR, ROOT, read_json, sha256_file, sha256_json, write_json  # noqa: E402
import fit_rank_first_throughput_challenger_v1 as F  # noqa: E402

SCHEMA = "sft_rank_first_throughput_challenger/v3"
IMPLEMENTATION_VERSION = (
    "sft_rank_first_throughput_challenger_impl/"
    "2026-08-06.zero3-mechanism-intercept-v3"
)
FACTOR_GRID = tuple(round(1.0 + 0.05 * i, 2) for i in range(41))
MATERIAL_GAP = 0.05

DEFAULT_MERGED_EVAL = (
    ROOT / "diagnostics" / "zero_gc_matched_contrast_merge"
    / "merged_extension_evaluation_v1.json"
)
DEFAULT_MERGED_PRED = (
    ROOT / "diagnostics" / "zero_gc_matched_contrast_merge"
    / "merged_frozen_predictions_v1.json"
)
DEFAULT_V1 = ARTIFACT_DIR / "rank_first_throughput_challenger_v1.json"
OUT_DIR = ROOT / "diagnostics" / "rank_first_challenger_zero3_intercept_v3"


def _rows(
    merged_eval: Path, merged_pred: Path, v1_artifact: Mapping[str, Any]
) -> list[dict[str, Any]]:
    profiles = F.InvariantStaticProfiles(
        [
            ROOT / "artifacts" / "dataset_profiles",
            ROOT / "artifacts" / "h800_lora_safety_stage2_v1" / "profiles",
        ]
    )
    hardware = read_json(ROOT / "config" / "hardware.json")
    candidates = F._load_extension_candidates(
        merged_eval,
        merged_pred,
        profiles,
        hardware_memory_bytes=float(hardware["memory_bytes_reported_by_torch"]),
    )
    absolute = v1_artifact["models"]["final_absolute_head"]
    rank = v1_artifact["models"]["final_rank_head"]
    out: list[dict[str, Any]] = []
    for candidate, final_log, absolute_log, rank_score in F._prediction_entries(
        candidates, absolute, rank
    ):
        selector = candidate["record"]["selector"]
        scenario = candidate["record"]["scenario"]
        out.append(
            {
                "scenario_id": str(candidate["scenario_id"]),
                "job_id": candidate.get("job_id"),
                "dataset_id": str(scenario.get("dataset_id")),
                "mbs": int(scenario.get("physical_mbs") or 0),
                "zero_stage": int(selector.get("zero_stage") or 0),
                "gradient_checkpointing": bool(selector.get("gradient_checkpointing")),
                "v1_log_prediction": float(final_log),
                "observed_log_throughput": float(candidate["observed_log_throughput"]),
            }
        )
    return out


def _corrected(row: Mapping[str, Any], factor: float) -> float:
    if int(row["zero_stage"]) == 3:
        return float(row["v1_log_prediction"]) - math.log(factor)
    return float(row["v1_log_prediction"])


def _pair_metrics(
    rows: Sequence[Mapping[str, Any]], factor: float
) -> dict[str, Any]:
    by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario_id"]), []).append(row)
    total = correct = 0
    material_total = material_correct = 0
    regrets: list[float] = []
    exact: list[float] = []
    apes: list[float] = []
    for group in by_scenario.values():
        for row in group:
            apes.append(
                abs(math.exp(_corrected(row, factor) - row["observed_log_throughput"]) - 1.0)
            )
        for left, right in itertools.combinations(group, 2):
            delta = float(left["observed_log_throughput"]) - float(
                right["observed_log_throughput"]
            )
            if abs(delta) <= 1.0e-12:
                continue
            predicted = _corrected(left, factor) - _corrected(right, factor)
            ok = (delta > 0.0) == (predicted > 0.0)
            total += 1
            correct += ok
            if 1.0 - math.exp(-abs(delta)) >= MATERIAL_GAP:
                material_total += 1
                material_correct += ok
        if len(group) >= 2:
            chosen = max(group, key=lambda item: _corrected(item, factor))
            oracle = max(group, key=lambda item: float(item["observed_log_throughput"]))
            regrets.append(
                max(
                    0.0,
                    1.0
                    - math.exp(
                        float(chosen["observed_log_throughput"])
                        - float(oracle["observed_log_throughput"])
                    ),
                )
            )
            exact.append(float(chosen is oracle))
    return {
        "pairs": total,
        "pairwise_accuracy": correct / total if total else None,
        "material_pairs": material_total,
        "material_pairwise_accuracy": (
            material_correct / material_total if material_total else None
        ),
        "mean_top1_regret": statistics.fmean(regrets) if regrets else None,
        "worst_top1_regret": max(regrets) if regrets else None,
        "exact_top1_fraction": statistics.fmean(exact) if exact else None,
        "absolute_mape": statistics.fmean(apes) if apes else None,
    }


def _select_factor(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Leave-one-scenario-out: choose on N-1 scenarios, score on the held-out one."""

    scenarios = sorted({str(row["scenario_id"]) for row in rows})
    folds: list[dict[str, Any]] = []
    picks: list[float] = []
    for held in scenarios:
        train = [row for row in rows if str(row["scenario_id"]) != held]
        test = [row for row in rows if str(row["scenario_id"]) == held]
        scored = [
            (
                _pair_metrics(train, factor)["pairwise_accuracy"] or 0.0,
                -abs(factor - 1.0),
                factor,
            )
            for factor in FACTOR_GRID
        ]
        chosen = max(scored)[2]
        picks.append(chosen)
        folds.append(
            {
                "held_out_scenario": held,
                "factor_chosen_on_training_scenarios": chosen,
                "held_out_uncorrected": _pair_metrics(test, 1.0),
                "held_out_corrected": _pair_metrics(test, chosen),
            }
        )
    return {
        "protocol": "leave_one_scenario_out",
        "grid": list(FACTOR_GRID),
        "per_fold": folds,
        "factors_chosen": picks,
        "selected_factor": statistics.median(picks),
        "selection_never_saw_the_scored_fold": True,
    }


def build_report(
    *, merged_eval: Path, merged_pred: Path, v1_path: Path
) -> dict[str, Any]:
    v1 = read_json(v1_path)
    rows = _rows(merged_eval, merged_pred, v1)
    selection = _select_factor(rows)
    factor = float(selection["selected_factor"])

    before = _pair_metrics(rows, 1.0)
    after = _pair_metrics(rows, factor)

    by_mechanism: dict[str, list[float]] = {}
    for row in rows:
        key = "z{z}_gc{g}".format(
            z=int(row["zero_stage"]), g=int(bool(row["gradient_checkpointing"]))
        )
        by_mechanism.setdefault(key, []).append(
            math.exp(_corrected(row, factor) - row["observed_log_throughput"])
        )
    residual = {
        key: {
            "n": len(values),
            "median_prediction_over_observation": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        for key, values in sorted(by_mechanism.items())
    }

    sensitivity = [
        {"factor": f, **_pair_metrics(rows, f)}
        for f in (1.0, 1.25, 1.35, 1.4, 1.45, 1.5, 1.55, 1.75, 2.0, 2.5, 3.0)
    ]

    # The optimum is a plateau, not a point: several adjacent factors score
    # identically.  Record its extent so the selected value is not mistaken for a
    # sharp optimum, and so a reader comparing the coarse grid against the chosen
    # factor does not think the two disagree.
    best = max(
        row["pairwise_accuracy"] or 0.0 for row in sensitivity
    )
    plateau = [
        row["factor"]
        for row in sensitivity
        if (row["pairwise_accuracy"] or 0.0) >= best - 1.0e-12
    ]
    plateau_note = {
        "best_pairwise_accuracy": best,
        "factors_attaining_it": plateau,
        "selected_is_within_plateau": factor in plateau,
        "tie_break": "smallest factor wins, i.e. the most conservative correction",
    }

    zero2_rows = [row for row in rows if int(row["zero_stage"]) == 2]
    no_regression = {
        "zero2_only_pairwise_before": _pair_metrics(zero2_rows, 1.0)["pairwise_accuracy"],
        "zero2_only_pairwise_after": _pair_metrics(zero2_rows, factor)["pairwise_accuracy"],
        "zero2_predictions_unchanged_by_construction": True,
    }

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "zero3_mechanism_intercept_fitted_and_replayed",
        "analysis_only": True,
        "publishable": False,
        "production_profile_generated": False,
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "frozen_v1_artifact_modified": False,
        "model_contract": {
            "base_model": "rank_first_throughput_challenger/v1",
            "change": "single mechanism-level intercept on the ranking score",
            "form": "log_score(ZeRO-3) -= log(factor); all other candidates unchanged",
            "selected_factor": factor,
            "features_added": 0,
            "coefficients_refitted": 0,
            "absolute_head_modified": False,
            "why_not_an_interaction_term": (
                "two sweeps rejected zero3_x_log2_mbs, the second while it was "
                "fully identifiable; the observed error is a constant offset, so "
                "a log2(MBS)-scaled term has the wrong functional form"
            ),
        },
        "selection": selection,
        "evaluation": {
            "merged_population_before": before,
            "merged_population_after": after,
            "residual_ratio_by_mechanism": residual,
            "sensitivity": sensitivity,
            "optimum_plateau": plateau_note,
            "no_regression_on_zero2": no_regression,
        },
        "data": {
            "rows": len(rows),
            "scenarios": len({str(row["scenario_id"]) for row in rows}),
            "zero3_rows": sum(1 for row in rows if int(row["zero_stage"]) == 3),
            "zero3_mbs_levels": sorted(
                {int(row["mbs"]) for row in rows if int(row["zero_stage"]) == 3}
            ),
        },
        "source_bindings": {
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "challenger_v1": {"path": str(v1_path.resolve()), "sha256": sha256_file(v1_path)},
            "merged_evaluation": {
                "path": str(merged_eval.resolve()),
                "sha256": sha256_file(merged_eval),
            },
            "merged_predictions": {
                "path": str(merged_pred.resolve()),
                "sha256": sha256_file(merged_pred),
            },
        },
        "limitations": [
            "the correction was motivated by, and fitted on, evidence that is not "
            "a fresh release holdout; a prospective acceptance run is still required",
            "ZeRO-2 never exceeded 58% of the safety line in the matched design, so "
            "the regime where ZeRO-2 does not fit is untested",
            "residual cross-MBS ranking errors on src02 are not addressed here",
            "packing, VL, offload and non-H800 cards remain out of scope",
        ],
    }
    report["report_sha256"] = sha256_json(report)
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    def pct(value: Any) -> str:
        return "n/a" if value is None else f"{float(value) * 100:.2f}%"

    before = report["evaluation"]["merged_population_before"]
    after = report["evaluation"]["merged_population_after"]
    factor = report["model_contract"]["selected_factor"]
    lines = [
        "# 吞吐 challenger v3：ZeRO-3 机制级截距",
        "",
        f"- 生成时间：{report['generated_at_utc']}",
        f"- 选定修正因子：**÷{factor:.2f}**（仅作用于 ZeRO-3 候选）",
        f"- 新增特征 {report['model_contract']['features_added']} 个，"
        f"重拟合系数 {report['model_contract']['coefficients_refitted']} 个",
        "- 未发布，未启动 GPU 实验，未改动 V5 与 challenger v1 冻结产物。",
        "",
        "## 为何是截距而不是交互项",
        "",
        "两轮消融都否决了 `zero3_x_log2_mbs`，第二轮是在该项**完全可识别**"
        "（公共 MBS 已非空）的条件下否决的。配对对照显示误差是**常数级乘性偏移**，"
        "与 MBS 无关，因此按 log2(MBS) 缩放的项函数形式本就不对。",
        "",
        "## 合并证据上的效果",
        "",
        "| 指标 | 修正前 | 修正后 |",
        "|---|---:|---:|",
        f"| 两两排序正确率 | {pct(before['pairwise_accuracy'])} | {pct(after['pairwise_accuracy'])} |",
        f"| 重要差距对正确率 | {pct(before['material_pairwise_accuracy'])} | {pct(after['material_pairwise_accuracy'])} |",
        f"| 平均 Top-1 损失 | {pct(before['mean_top1_regret'])} | {pct(after['mean_top1_regret'])} |",
        f"| 最差 Top-1 损失 | {pct(before['worst_top1_regret'])} | {pct(after['worst_top1_regret'])} |",
        f"| Top-1 完全命中率 | {pct(before['exact_top1_fraction'])} | {pct(after['exact_top1_fraction'])} |",
        f"| 绝对吞吐平均相对误差 | {pct(before['absolute_mape'])} | {pct(after['absolute_mape'])} |",
        "",
        "## 留一场景交叉验证（因子从不在被评分的那折上选取）",
        "",
        "| 留出场景 | 训练折选出因子 | 留出集修正前 | 留出集修正后 |",
        "|---|---:|---:|---:|",
    ]
    for fold in report["selection"]["per_fold"]:
        lines.append(
            "| {s} | {f:.2f} | {a} | {b} |".format(
                s=fold["held_out_scenario"].split("__")[0].replace("lora_s2_", ""),
                f=fold["factor_chosen_on_training_scenarios"],
                a=pct(fold["held_out_uncorrected"]["pairwise_accuracy"]),
                b=pct(fold["held_out_corrected"]["pairwise_accuracy"]),
            )
        )
    lines += [
        "",
        "## 修正后各机制的残余比值（预测/实测，1.00 为无偏）",
        "",
        "| 机制 | n | 中位 | 最小 | 最大 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, value in report["evaluation"]["residual_ratio_by_mechanism"].items():
        lines.append(
            f"| `{key}` | {value['n']} | {value['median_prediction_over_observation']:.2f} "
            f"| {value['min']:.2f} | {value['max']:.2f} |"
        )
    lines += [
        "",
        "## 因子敏感性",
        "",
        "| 因子 | 两两排序 | 平均 Top-1 损失 | 最差 Top-1 损失 |",
        "|---:|---:|---:|---:|",
    ]
    plateau = report["evaluation"]["optimum_plateau"]
    for row in report["evaluation"]["sensitivity"]:
        mark = " ←最优区" if row["factor"] in plateau["factors_attaining_it"] else ""
        star = " **(选定)**" if abs(row["factor"] - factor) < 1e-9 else ""
        lines.append(
            f"| {row['factor']:.2f}{star} | {pct(row['pairwise_accuracy'])}{mark} "
            f"| {pct(row['mean_top1_regret'])} | {pct(row['worst_top1_regret'])} |"
        )
    lines += [
        "",
        f"最优是一个**平台区**而非单点：因子 {plateau['factors_attaining_it']} 给出完全相同的"
        f"排序正确率 {pct(plateau['best_pairwise_accuracy'])}。"
        "平手时取最小值，即最保守的修正幅度。",
    ]
    nr = report["evaluation"]["no_regression_on_zero2"]
    lines += [
        "",
        "## ZeRO-2 无回退检查",
        "",
        f"- 仅 ZeRO-2 候选的两两排序：修正前 {pct(nr['zero2_only_pairwise_before'])}，"
        f"修正后 {pct(nr['zero2_only_pairwise_after'])}",
        "- ZeRO-2 的预测按构造不受影响。",
        "",
        "## 局限",
        "",
    ]
    lines += [f"- {item}" for item in report["limitations"]]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-evaluation", type=Path, default=DEFAULT_MERGED_EVAL)
    parser.add_argument("--merged-predictions", type=Path, default=DEFAULT_MERGED_PRED)
    parser.add_argument("--v1-artifact", type=Path, default=DEFAULT_V1)
    parser.add_argument(
        "--output", type=Path, default=OUT_DIR / "rank_first_throughput_challenger_v3.json"
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=OUT_DIR / "rank_first_throughput_challenger_v3.md",
    )
    args = parser.parse_args()
    report = build_report(
        merged_eval=args.merged_evaluation.resolve(),
        merged_pred=args.merged_predictions.resolve(),
        v1_path=args.v1_artifact.resolve(),
    )
    write_json(args.output, report)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "markdown_output": str(args.markdown_output.resolve()),
                "selected_factor": report["model_contract"]["selected_factor"],
                "before": report["evaluation"]["merged_population_before"],
                "after": report["evaluation"]["merged_population_after"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
