#!/usr/bin/env python3
"""Fit an analysis-only Packing residual on the shared physical-work trunk.

The frozen H800 throughput model remains unchanged.  This script starts from
its existing Packed physical-work prediction and tests whether a small static
residual can make that route usable for Packing.  No runtime Packing statistic
is admitted as an input.  In particular, ``n_pack_mean`` is used upstream by
the physical-work/GBS reconstruction, but is not exposed to the residual fit.

Validation leaves one upstream dataset-profile group out at a time.  The
feature-family comparison is exploratory because all candidates are evaluated
on the current evidence population; therefore this artifact cannot publish a
production model even when an individual metric gate passes.
"""

from __future__ import annotations

import copy
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from common import (
    ARTIFACT_DIR,
    ROOT,
    percentile,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

INPUT = ARTIFACT_DIR / "h800_packing_virtual_mbs_analysis_v1.json"
MAIN_MODEL = (
    ROOT
    / "diagnostics"
    / "h800_unified_resource_partial_refit_20260809"
    / "throughput_model.json"
)
OUTPUT = ARTIFACT_DIR / "h800_packing_shared_physical_throughput_v1.json"
MARKDOWN = ARTIFACT_DIR / "h800_packing_shared_physical_throughput_v1.md"

RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
SHADOW_GATES = {
    "group_equal_mape": 0.15,
    "p90_ape": 0.30,
}

TARGETS = {
    "retrospective_absolute_effective": {
        "description": (
            "observed Packed effective throughput divided by the frozen shared "
            "physical-work prediction"
        ),
        "use": "Packed absolute-throughput/ETA correction",
    },
    "paired_effect_transfer_effective": {
        "description": (
            "observed Packed/Unpacked effective-throughput ratio divided by "
            "the frozen shared-model Packed/Unpacked ratio"
        ),
        "use": "Packing-versus-Unpacked recommendation correction",
    },
}

FEATURE_DEFINITIONS = {
    "mean_length_to_cutoff": (
        "upload-time mean sequence length divided by selected cutoff_len"
    ),
    "length_cv": "upload-time sequence-length coefficient of variation",
    "p99_length_to_cutoff": (
        "upload-time P99 sequence length divided by selected cutoff_len"
    ),
    "pack_fill_proxy": (
        "static mean_length * mean_samples_per_pack / cutoff_len, clipped at 1.25; "
        "this is physical-work geometry rather than a free residual n_pack input"
    ),
    "zero2": "one when ZeRO stage is 2",
    "zero3": "one when ZeRO stage is 3",
    "gc_off": "one when gradient checkpointing is disabled",
    "log2_gpu_count": "base-2 logarithm of requested GPU count",
}

# These families are deliberately small.  profile_shape_3 is the preferred
# product-compatible challenger: it does not use n_pack_mean, runtime fill, or
# any observation collected after training starts.
FEATURE_FAMILIES = {
    "profile_shape_3": (
        "mean_length_to_cutoff",
        "length_cv",
        "p99_length_to_cutoff",
    ),
    "profile_with_physical_fill_4": (
        "mean_length_to_cutoff",
        "length_cv",
        "p99_length_to_cutoff",
        "pack_fill_proxy",
    ),
    "mechanism_4": (
        "zero2",
        "zero3",
        "gc_off",
        "log2_gpu_count",
    ),
    "profile_and_mechanism_7": (
        "mean_length_to_cutoff",
        "length_cv",
        "p99_length_to_cutoff",
        "zero2",
        "zero3",
        "gc_off",
        "log2_gpu_count",
    ),
}


def _positive(value: Any, floor: float = 1.0e-12) -> float:
    return max(floor, float(value))


def _verify_input(report: Mapping[str, Any]) -> None:
    if report.get("schema") != "sft_h800_packing_virtual_mbs_analysis/v1":
        raise ValueError(f"unexpected input schema: {report.get('schema')}")
    body = copy.deepcopy(dict(report))
    stored = str(body.pop("report_sha256", ""))
    if not stored or sha256_json(body) != stored:
        raise ValueError("input report_sha256 does not match the input body")
    if int(report["population"]["settings"]) != 23:
        raise ValueError("expected the frozen 23-setting evidence population")
    if int(report["population"]["profile_groups"]) != 10:
        raise ValueError("expected the frozen 10-profile-group evidence population")


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("geometric mean requires at least one value")
    return math.exp(statistics.fmean(math.log(_positive(value)) for value in values))


def build_feature_rows(report: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Build one modeling row per setting from the frozen diagnostic artifact."""

    source = read_json(INPUT) if report is None else report
    _verify_input(source)
    repeat_absolute: dict[str, list[float]] = defaultdict(list)
    for row in source["pair_rows"]:
        observed = float(row["observed"]["packed_effective_tokens_per_second"])
        predicted = float(
            row["prediction"]["packed_physical_shared_model"][
                "predicted_effective_tokens_per_second"
            ]
        )
        repeat_absolute[str(row["setting_id"])].append(observed / predicted)

    rows: list[dict[str, Any]] = []
    for source_row in source["setting_rows"]:
        setting_id = str(source_row["setting_id"])
        profile = source_row["profile"]
        cutoff = int(source_row["cutoff_len"])
        features = {
            "mean_length_to_cutoff": float(profile["mean_length"]) / cutoff,
            "length_cv": float(profile["length_cv"]),
            "p99_length_to_cutoff": float(profile["p99_length_to_cutoff"]),
            "pack_fill_proxy": float(profile["pack_fill_proxy"]),
            "zero2": float(int(source_row["zero_stage"]) == 2),
            "zero3": float(int(source_row["zero_stage"]) == 3),
            "gc_off": float(not bool(source_row["gc"])),
            "log2_gpu_count": math.log2(int(source_row["gpu_count"])),
        }
        if "n_pack_mean" in features:
            raise AssertionError("raw n_pack_mean must not enter residual features")
        rows.append(
            {
                "setting_id": setting_id,
                "profile_group": str(source_row["profile_group"]),
                "source": str(source_row["source"]),
                "dataset_id": str(source_row["dataset_id"]),
                "cutoff_len": cutoff,
                "gpu_count": int(source_row["gpu_count"]),
                "zero_stage": int(source_row["zero_stage"]),
                "gc": bool(source_row["gc"]),
                "n_pack_mean_physical_only": float(source_row["n_pack_mean"]),
                "features": features,
                "targets": {
                    "retrospective_absolute_effective": _geometric_mean(
                        repeat_absolute[setting_id]
                    ),
                    "paired_effect_transfer_effective": float(
                        source_row["residual_coefficient"]["packed_physical_effective"]
                    ),
                },
                "observed_packed_over_unpacked": float(
                    source_row["observed_ratio"]["effective_tokens_per_second"]
                ),
                "shared_model_packed_over_unpacked": (
                    float(
                        source_row["prediction"][
                            "packed_physical_effective_tokens_per_second"
                        ]
                    )
                    / float(
                        source_row["prediction"][
                            "unpacked_actual_effective_tokens_per_second"
                        ]
                    )
                ),
            }
        )
    rows.sort(key=lambda row: str(row["setting_id"]))
    if len(rows) != 23 or len({str(row["profile_group"]) for row in rows}) != 10:
        raise ValueError("feature rows do not preserve the frozen population")
    return rows


def _group_equal_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts = Counter(str(row["profile_group"]) for row in rows)
    groups = len(counts)
    return np.asarray(
        [1.0 / (groups * counts[str(row["profile_group"])]) for row in rows],
        dtype=float,
    )


def _target_log(row: Mapping[str, Any], target: str) -> float:
    return math.log(_positive(row["targets"][target]))


def _fit_constant(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
) -> dict[str, Any]:
    y = np.asarray([_target_log(row, target) for row in rows], dtype=float)
    weights = _group_equal_weights(rows)
    intercept = float(np.sum(weights * y) / np.sum(weights))
    return {
        "intercept_log": intercept,
        "multiplier": math.exp(intercept),
    }


def _fit_ridge(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    features: Sequence[str],
    alpha: float,
) -> dict[str, Any]:
    x = np.asarray(
        [[float(row["features"][name]) for name in features] for row in rows],
        dtype=float,
    )
    y = np.asarray([_target_log(row, target) for row in rows], dtype=float)
    weights = _group_equal_weights(rows)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = Ridge(alpha=float(alpha)).fit(
        scaler.transform(x),
        y,
        sample_weight=weights,
    )
    return {
        "features": list(features),
        "alpha": float(alpha),
        "feature_means": {
            name: float(value) for name, value in zip(features, scaler.mean_)
        },
        "feature_scales": {
            name: float(value) for name, value in zip(features, scaler.scale_)
        },
        "intercept_log": float(model.intercept_),
        "standardized_coefficients": {
            name: float(value) for name, value in zip(features, model.coef_)
        },
    }


def _predict_ridge_log(model: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    prediction = float(model["intercept_log"])
    for name in model["features"]:
        scale = _positive(model["feature_scales"][name])
        standardized = (
            float(row["features"][name]) - float(model["feature_means"][name])
        ) / scale
        prediction += float(model["standardized_coefficients"][name]) * standardized
    return prediction


def _metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    predicted_logs: Sequence[float],
) -> dict[str, Any]:
    if len(rows) != len(predicted_logs):
        raise ValueError("prediction length does not match row length")
    signed_errors = [
        math.exp(predicted - _target_log(row, target)) - 1.0
        for row, predicted in zip(rows, predicted_logs)
    ]
    absolute_errors = [abs(value) for value in signed_errors]
    by_group: dict[str, list[float]] = defaultdict(list)
    for row, error in zip(rows, absolute_errors):
        by_group[str(row["profile_group"])].append(error)
    return {
        "settings": len(rows),
        "profile_groups": len(by_group),
        "group_equal_mape": statistics.fmean(
            statistics.fmean(values) for values in by_group.values()
        ),
        "row_mape": statistics.fmean(absolute_errors),
        "median_ape": percentile(absolute_errors, 50),
        "p90_ape": percentile(absolute_errors, 90),
        "maximum_ape": max(absolute_errors),
        "signed_bias": statistics.fmean(signed_errors),
        "by_profile_group_mape": {
            group: statistics.fmean(values)
            for group, values in sorted(by_group.items())
        },
    }


def _gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    results = {
        name: float(metrics[name]) <= threshold
        for name, threshold in SHADOW_GATES.items()
    }
    return {
        "thresholds": dict(SHADOW_GATES),
        "results": results,
        "all_passed": all(results.values()),
    }


def _raw_shared_trunk(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
) -> dict[str, Any]:
    predictions = [0.0] * len(rows)
    metrics = _metrics(rows, target=target, predicted_logs=predictions)
    return {
        "model_family": "shared_physical_work_without_residual",
        "validation": "direct evaluation on profile-collapsed settings",
        "metrics": metrics,
        "gate": _gate(metrics),
        "oof_rows": [
            {
                "setting_id": str(row["setting_id"]),
                "actual_multiplier": float(row["targets"][target]),
                "predicted_multiplier": 1.0,
                "absolute_percentage_error": abs(
                    1.0 / float(row["targets"][target]) - 1.0
                ),
            }
            for row in rows
        ],
    }


def _loso_constant(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
) -> dict[str, Any]:
    predictions = [0.0] * len(rows)
    groups = sorted({str(row["profile_group"]) for row in rows})
    for held_group in groups:
        train = [row for row in rows if str(row["profile_group"]) != held_group]
        fitted = _fit_constant(train, target=target)
        for index, row in enumerate(rows):
            if str(row["profile_group"]) == held_group:
                predictions[index] = float(fitted["intercept_log"])
    metrics = _metrics(rows, target=target, predicted_logs=predictions)
    return {
        "model_family": "profile_equal_global_scalar",
        "validation": "leave_one_profile_group_out",
        "metrics": metrics,
        "gate": _gate(metrics),
        "full_fit": _fit_constant(rows, target=target),
        "oof_rows": _oof_rows(rows, target=target, predicted_logs=predictions),
    }


def _inner_select_alpha(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    features: Sequence[str],
) -> float:
    groups = sorted({str(row["profile_group"]) for row in rows})
    if len(groups) < 3:
        return 10.0
    scores: list[tuple[float, float, float]] = []
    for alpha in RIDGE_ALPHAS:
        held_rows: list[Mapping[str, Any]] = []
        predictions: list[float] = []
        for held_group in groups:
            train = [row for row in rows if str(row["profile_group"]) != held_group]
            test = [row for row in rows if str(row["profile_group"]) == held_group]
            fitted = _fit_ridge(
                train,
                target=target,
                features=features,
                alpha=alpha,
            )
            held_rows.extend(test)
            predictions.extend(_predict_ridge_log(fitted, row) for row in test)
        metric = _metrics(held_rows, target=target, predicted_logs=predictions)
        scores.append((float(metric["group_equal_mape"]), -float(alpha), float(alpha)))
    return min(scores)[2]


def _oof_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    predicted_logs: Sequence[float],
) -> list[dict[str, Any]]:
    result = []
    for row, predicted_log in zip(rows, predicted_logs):
        actual = float(row["targets"][target])
        predicted = math.exp(float(predicted_log))
        result.append(
            {
                "setting_id": str(row["setting_id"]),
                "profile_group": str(row["profile_group"]),
                "actual_multiplier": actual,
                "predicted_multiplier": predicted,
                "absolute_percentage_error": abs(predicted / actual - 1.0),
            }
        )
    return result


def _loso_ridge(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    family: str,
    features: Sequence[str],
) -> dict[str, Any]:
    predictions = [0.0] * len(rows)
    selected_alphas: dict[str, float] = {}
    groups = sorted({str(row["profile_group"]) for row in rows})
    for held_group in groups:
        train = [row for row in rows if str(row["profile_group"]) != held_group]
        alpha = _inner_select_alpha(train, target=target, features=features)
        selected_alphas[held_group] = alpha
        fitted = _fit_ridge(
            train,
            target=target,
            features=features,
            alpha=alpha,
        )
        for index, row in enumerate(rows):
            if str(row["profile_group"]) == held_group:
                predictions[index] = _predict_ridge_log(fitted, row)
    final_alpha = _inner_select_alpha(rows, target=target, features=features)
    metrics = _metrics(rows, target=target, predicted_logs=predictions)
    return {
        "model_family": family,
        "features": list(features),
        "validation": "nested_leave_one_profile_group_out",
        "outer_fold_selected_alphas": selected_alphas,
        "metrics": metrics,
        "gate": _gate(metrics),
        "full_fit": _fit_ridge(
            rows,
            target=target,
            features=features,
            alpha=final_alpha,
        ),
        "oof_rows": _oof_rows(rows, target=target, predicted_logs=predictions),
    }


def _validate_target(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
) -> dict[str, Any]:
    candidates = {
        family: _loso_ridge(
            rows,
            target=target,
            family=family,
            features=features,
        )
        for family, features in FEATURE_FAMILIES.items()
    }
    selected_name, selected = min(
        candidates.items(),
        key=lambda item: (
            float(item[1]["metrics"]["group_equal_mape"]),
            float(item[1]["metrics"]["p90_ape"]),
            len(item[1]["features"]),
            item[0],
        ),
    )
    return {
        "target": target,
        "target_contract": dict(TARGETS[target]),
        "raw_shared_trunk": _raw_shared_trunk(rows, target=target),
        "global_scalar": _loso_constant(rows, target=target),
        "ridge_candidates": candidates,
        "selected_exploratory_candidate": selected_name,
        "selected_metrics": selected["metrics"],
        "selected_gate": selected["gate"],
        "selection_warning": (
            "feature-family selection used the current evidence population and "
            "must be frozen before prospective validation"
        ),
    }


def _main_model_reference() -> dict[str, Any]:
    report = read_json(MAIN_MODEL)
    metrics = report["validation"]["h800_native_holdout"]["native_only"]["metrics"]
    return {
        "scenario_equal_absolute_throughput_mape": float(
            metrics["scenario_equal_absolute_throughput_mape"]
        ),
        "absolute_throughput_ape_p90": float(metrics["absolute_throughput_ape_p90"]),
        "scenario_equal_pairwise_accuracy": float(
            metrics["scenario_equal_pairwise_accuracy"]
        ),
        "packing_primary_training_rows": int(
            report["model_contract"]["packing_primary_training_rows"]
        ),
    }


def _example(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    setting_id = "real:d4500edu:c32768"
    row = next(row for row in rows if str(row["setting_id"]) == setting_id)
    absolute = float(row["targets"]["retrospective_absolute_effective"])
    paired = float(row["targets"]["paired_effect_transfer_effective"])
    return {
        "setting_id": setting_id,
        "cutoff_len": int(row["cutoff_len"]),
        "absolute_observed_over_model": absolute,
        "absolute_raw_shared_trunk_ape": abs(1.0 / absolute - 1.0),
        "paired_effect_observed_over_model": paired,
        "paired_effect_raw_shared_trunk_ape": abs(1.0 / paired - 1.0),
        "interpretation": (
            "the Packed absolute prediction is close, but the model overstates "
            "the Packed-versus-Unpacked advantage for this setting"
        ),
    }


def _source_bindings() -> dict[str, Any]:
    return {
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "input_virtual_mbs_diagnostic": {
            "path": str(INPUT.resolve()),
            "sha256": sha256_file(INPUT),
        },
        "frozen_main_throughput_model": {
            "path": str(MAIN_MODEL.resolve()),
            "sha256": sha256_file(MAIN_MODEL),
        },
    }


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _render_markdown(report: Mapping[str, Any]) -> str:
    absolute = report["validation"]["retrospective_absolute_effective"]
    paired = report["validation"]["paired_effect_transfer_effective"]
    example = report["concrete_example"]

    def table_rows(validation: Mapping[str, Any]) -> list[str]:
        models = [
            ("Packed 物理主干，不加修正", validation["raw_shared_trunk"]),
            ("Packed 物理主干 + 一个全局系数", validation["global_scalar"]),
        ]
        labels = {
            "profile_shape_3": "+ 长度画像 3 参数",
            "profile_with_physical_fill_4": "+ 长度画像与填充 4 参数",
            "mechanism_4": "+ 训练机制 4 参数",
            "profile_and_mechanism_7": "+ 画像与机制 7 参数",
        }
        models.extend(
            (labels[name], validation["ridge_candidates"][name])
            for name in FEATURE_FAMILIES
        )
        result = []
        for label, model in models:
            metrics = model["metrics"]
            result.append(
                f"| {label} | {_pct(metrics['group_equal_mape'])} | "
                f"{_pct(metrics['p90_ape'])} | {_pct(metrics['maximum_ape'])} | "
                f"{'通过' if model['gate']['all_passed'] else '未通过'} |"
            )
        return result

    lines = [
        "# H800 Packing 共享物理主干 + 小残差离线拟合",
        "",
        "本产物只做影子验证；主显存模型、主吞吐模型和线上推荐均未修改。",
        "",
        "## 1. 现在的问题",
        "",
        "把 Packed 物理工作量送进现有 unpacked 吞吐主干之后，还需要多大的 Packing 专属修正；这个修正能否只依赖产品已经保存的静态字段。",
        "",
        "最终计算口径很简单：先由主模型按 Packed 物理工作量算吞吐，再乘一个小残差倍数。",
        "",
        "## 2. 当前实测结果",
        "",
        f"共 {report['population']['matched_repeat_pairs']} 个 matched repeat pairs，折叠成 {report['population']['settings']} 个 setting、{report['population']['profile_groups']} 个数据画像组。验证时每次完整留出一个数据画像组。",
        "",
        "### 2.1 Packed 绝对吞吐/ETA",
        "",
        "| 模型 | 数据画像等权 MAPE | P90 APE | 最大 APE | 15% / 30% 门槛 |",
        "|---|---:|---:|---:|---|",
        *table_rows(absolute),
        "",
        f"当前最好的探索性候选是 `{absolute['selected_exploratory_candidate']}`：数据画像等权 MAPE {_pct(absolute['selected_metrics']['group_equal_mape'])}，P90 APE {_pct(absolute['selected_metrics']['p90_ape'])}。",
        "",
        "### 2.2 Packing 与 unpacked 的相对收益",
        "",
        "| 模型 | 数据画像等权 MAPE | P90 APE | 最大 APE | 15% / 30% 门槛 |",
        "|---|---:|---:|---:|---|",
        *table_rows(paired),
        "",
        f"当前最好的探索性候选是 `{paired['selected_exploratory_candidate']}`：数据画像等权 MAPE {_pct(paired['selected_metrics']['group_equal_mape'])}，P90 APE {_pct(paired['selected_metrics']['p90_ape'])}。相对收益门槛{'通过' if paired['selected_gate']['all_passed'] else '未通过'}。",
        "",
        "## 3. 一个具体例子",
        "",
        f"教育会话中长数据集、cutoff={example['cutoff_len']}：Packed 绝对吞吐的实测/预测为 {example['absolute_observed_over_model']:.3f}，裸主干误差只有 {_pct(example['absolute_raw_shared_trunk_ape'])}；但 Packed/Unpacked 相对收益的实测/预测为 {example['paired_effect_observed_over_model']:.3f}，裸主干误差达到 {_pct(example['paired_effect_raw_shared_trunk_ape'])}。",
        "",
        "这说明“Packed 绝对吞吐准”不能直接推出“是否开 Packing 的排序也准”。产品推荐最终关心后者，所以暂时不能发布。",
        "",
        "## 4. 产品输入和新增参数",
        "",
        "- 不新增上传阶段字段：使用均值、长度变异系数（CV）、P99、cutoff 和已有训练配置。",
        "- 平均每 pack 样本数不进入残差；它只用于反推 GAS/GBS 和构造 Packed 物理工作量。",
        "- 绝对吞吐候选新增 1 个截距和 3 个画像系数；这是新模型系数，不是新产品参数。",
        "- 训练机制候选使用 ZeRO、梯度检查点和卡数；这些本来就是用户输入。",
        "",
        "## 5. 结论和下一步",
        "",
        "数学上能推出：共享主干不会改变 unpacked 预测；Packing 残差只作用于 Packed 路径。",
        "",
        "当前实验观察：3 个长度画像系数可以把 Packed 绝对吞吐做到可用量级，但相对收益最好的候选仍未通过门槛。",
        "",
        "工程决定：冻结本次候选，不并入主模型；下一轮只补平均每 pack 样本数附近的 unpacked MBS 对照，并用新的数据画像做前瞻验证。通过后再合并，产品侧无需增加字段。",
        "",
    ]
    return "\n".join(lines)


def build_report() -> dict[str, Any]:
    source = read_json(INPUT)
    rows = build_feature_rows(source)
    validation = {target: _validate_target(rows, target=target) for target in TARGETS}
    absolute_passed = bool(
        validation["retrospective_absolute_effective"]["selected_gate"]["all_passed"]
    )
    paired_passed = bool(
        validation["paired_effect_transfer_effective"]["selected_gate"]["all_passed"]
    )
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_shared_physical_throughput/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_complete_not_publishable",
        "analysis_only": True,
        "publishable": False,
        "gpu_experiments_launched": False,
        "main_memory_model_mutated": False,
        "main_throughput_model_mutated": False,
        "problem": (
            "fit a small product-compatible Packing residual on top of the "
            "frozen shared physical-work throughput trunk"
        ),
        "population": {
            "matched_repeat_pairs": int(source["population"]["matched_repeat_pairs"]),
            "settings": len(rows),
            "profile_groups": len({str(row["profile_group"]) for row in rows}),
            "sources": dict(Counter(str(row["source"]) for row in rows)),
        },
        "input_contract": {
            "product_fields_already_available": [
                "mean_length",
                "length_cv",
                "p99_length",
                "cutoff_len",
                "zero_stage",
                "gradient_checkpointing",
                "gpu_count",
            ],
            "raw_n_pack_mean_in_residual": False,
            "n_pack_mean_allowed_uses": [
                "derive expected Packed logical GBS",
                "derive gradient accumulation steps",
                "construct Packed physical work",
            ],
            "runtime_packing_statistics_used": False,
            "new_product_metadata_required": False,
            "feature_definitions": dict(FEATURE_DEFINITIONS),
        },
        "model_contract": {
            "unpacked_path": "frozen main throughput model, exactly unchanged",
            "packed_path": (
                "frozen main throughput model evaluated with Packed physical work, "
                "then multiplied by a small static residual"
            ),
            "new_fitted_parameters": (
                "one residual intercept plus one coefficient per selected feature"
            ),
        },
        "validation_contract": {
            "unit": "setting collapsed over repeats",
            "split": "leave one upstream dataset-profile group out",
            "weights": "each dataset-profile group has equal total weight",
            "ridge_alpha_selection": (
                "nested leave-one-profile-group-out inside each outer fold"
            ),
            "shadow_gates": dict(SHADOW_GATES),
            "feature_family_selection": "exploratory on current evidence",
            "prospective_validation_required_before_publication": True,
        },
        "main_unpacked_model_reference": _main_model_reference(),
        "validation": validation,
        "concrete_example": _example(rows),
        "feature_rows": rows,
        "gates": {
            "absolute_packed_throughput_shadow_gate_passed": absolute_passed,
            "packing_relative_effect_shadow_gate_passed": paired_passed,
            "all_modeling_gates_passed": absolute_passed and paired_passed,
            "prospective_profile_holdout_completed": False,
            "automatic_packing_recommendation_allowed": False,
            "automatic_publication_allowed": False,
        },
        "interpretation_contract": {
            "mathematical_fact": (
                "a residual applied only after the Packed shared-trunk prediction "
                "leaves every Unpacked prediction exactly unchanged"
            ),
            "empirical_observation": (
                "the reported errors apply to the current 23 settings and 10 "
                "upstream profile groups"
            ),
            "engineering_assumption": (
                "upload-time mean/CV/P99 plus selected cutoff capture enough profile "
                "shape for the Packing residual to transfer to a new dataset"
            ),
            "leakage_limitation": (
                "the frozen main trunk may have seen Unpacked observations related "
                "to the current experiments; therefore absolute Packed validation "
                "is retrospective and cannot replace prospective profile validation"
            ),
        },
        "decision": {
            "keep_shared_packed_physical_work_trunk": True,
            "freeze_exploratory_residual_candidate": True,
            "merge_into_main_model_now": False,
            "reason": (
                "absolute Packed throughput passes the shadow gate, but the "
                "Packing-versus-Unpacked transfer gate does not"
            ),
            "next_experiment": (
                "for new held-out profiles, measure Packed MBS=1 against Unpacked "
                "MBS values nearest static mean_samples_per_pack, while keeping the "
                "candidate feature family frozen"
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
    summary = {
        "output": str(OUTPUT),
        "markdown": str(MARKDOWN),
        "settings": report["population"]["settings"],
        "profile_groups": report["population"]["profile_groups"],
        "absolute_selected": report["validation"]["retrospective_absolute_effective"][
            "selected_exploratory_candidate"
        ],
        "absolute_metrics": report["validation"]["retrospective_absolute_effective"][
            "selected_metrics"
        ],
        "paired_selected": report["validation"]["paired_effect_transfer_effective"][
            "selected_exploratory_candidate"
        ],
        "paired_metrics": report["validation"]["paired_effect_transfer_effective"][
            "selected_metrics"
        ],
        "all_gates_passed": report["gates"]["all_modeling_gates_passed"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
