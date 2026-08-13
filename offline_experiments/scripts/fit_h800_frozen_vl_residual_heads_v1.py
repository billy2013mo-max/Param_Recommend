#!/usr/bin/env python3
"""Fit and formally evaluate the frozen-VL H800 residual overlay.

The campaign uses token-length-matched text controls.  Successful runs provide
exact max-reserved and step-time labels.  Confirmed CUDA OOM runs remain
right-censored constraints and are never converted into peak labels.

This is a development fit with grouped cross-validation.  It writes a
shadow-only model artifact because no source-disjoint prospective workload was
reserved before fitting.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from evaluate_h800_frozen_vl_combined_formal_v1 import OUTPUT as FORMAL_RESULTS
from h800_frozen_vl_overlay_v1 import ARTIFACT_SCHEMA
from prepare_h800_frozen_vl_combined_formal_v1 import QUEUE

GIB = float(1 << 30)
MODEL_INVENTORY = ARTIFACT_DIR / "h800_qwen35_vl_supplement_model_inventory_v1.json"
MODEL_OUTPUT = ARTIFACT_DIR / "h800_frozen_vl_residual_overlay_v1.json"
EVALUATION_OUTPUT = ARTIFACT_DIR / "h800_frozen_vl_modeling_formal_evaluation_v1.json"
MARKDOWN_OUTPUT = ARTIFACT_DIR / "h800_frozen_vl_modeling_formal_evaluation_v1.md"


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mean_stat(summary: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = summary.get(name)
    if value is None:
        return float(default)
    if isinstance(value, dict):
        value = value.get("mean", default)
    return _finite(value, name)


def _profile_features(job: dict[str, Any], model: dict[str, Any]) -> dict[str, float]:
    profile = read_json(Path(job["dataset_profile_path"]))
    summary = profile.get("summary") or {}
    geometry = model.get("vision_geometry") or {}
    components = model.get("component_parameter_estimates") or {}

    visual_tokens = _mean_stat(summary, "visual_tokens_total")
    images = _mean_stat(summary, "images_per_sample")
    videos = _mean_stat(summary, "videos_per_sample")
    media = _mean_stat(summary, "media_per_sample", images + videos)
    spatial_merge = _finite(geometry["spatial_merge_size"], "spatial_merge_size")
    raw_patches = _mean_stat(
        summary,
        "raw_patch_units_total",
        visual_tokens * spatial_merge * spatial_merge,
    )
    patch_size = _finite(geometry["patch_size"], "patch_size")
    temporal_patch = _finite(geometry["temporal_patch_size"], "temporal_patch_size")
    input_channels = _finite(geometry.get("in_channels") or 3, "in_channels")
    pixel_elements = _mean_stat(
        summary,
        "pixel_values_elements_total",
        raw_patches * input_channels * temporal_patch * patch_size * patch_size,
    )
    sampled_frames = _mean_stat(summary, "sampled_video_frames_total", images)
    depth = _finite(geometry["depth"], "vision_depth")
    hidden = _finite(geometry["hidden_size"], "vision_hidden")
    intermediate = _finite(geometry["intermediate_size"], "vision_intermediate")
    output_hidden = _finite(geometry["out_hidden_size"], "vision_output_hidden")
    vision_parameters = _finite(components["vision_tower"], "vision_tower_parameters")
    projector_parameters = _finite(
        components["projector_or_merger"], "projector_parameters"
    )
    dtype_bytes = 2.0
    physical_mbs = float(job["mbs"])

    vision_dynamic_bytes = physical_mbs * (
        pixel_elements * dtype_bytes
        + raw_patches * hidden * dtype_bytes
        + visual_tokens * output_hidden * dtype_bytes
    )
    per_patch_linear_flops = 8.0 * hidden * hidden + 4.0 * hidden * intermediate
    vision_forward_flops = (
        depth * raw_patches * per_patch_linear_flops
        + 2.0 * projector_parameters * visual_tokens
    )
    return {
        "mean_visual_tokens_per_sample": visual_tokens,
        "mean_raw_patch_units_per_sample": raw_patches,
        "mean_pixel_value_elements_per_sample": pixel_elements,
        "mean_sampled_frames_per_sample": sampled_frames,
        "mean_media_per_sample": media,
        "vision_depth": depth,
        "vision_hidden_size": hidden,
        "vision_parameter_elements": vision_parameters,
        "projector_parameter_elements": projector_parameters,
        "vision_dynamic_microbatch_bytes": vision_dynamic_bytes,
        "vision_forward_flops_per_sample": vision_forward_flops,
    }


def _logical_samples(status_path: str) -> float:
    attempt_dir = Path(status_path).resolve().parent
    summaries = [
        read_json(path)
        for path in sorted((attempt_dir / "metrics").glob("summary.rank*.json"))
    ]
    if not summaries:
        raise ValueError(f"success status has no summaries: {status_path}")
    value = sum(
        _finite(
            (row.get("measured_totals") or {}).get("logical_samples"), "logical_samples"
        )
        for row in summaries
    )
    if value <= 0.0:
        raise ValueError(
            f"success status has no positive logical samples: {status_path}"
        )
    return value


def _collapse_success_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row["formal_row"]["metrics"] for row in rows]
    if not metrics or any(value is None for value in metrics):
        raise ValueError("success arm has missing metrics")

    def mean(name: str) -> float:
        return statistics.fmean(_finite(value[name], name) for value in metrics)

    return {
        "repeat_count": len(rows),
        "job_ids": [row["job"]["job_id"] for row in rows],
        "max_reserved_bytes": mean("max_reserved_bytes"),
        "max_allocated_bytes": mean("max_allocated_bytes"),
        "measured_seconds": mean("measured_seconds"),
        "effective_tokens": mean("effective_tokens"),
        "logical_samples": statistics.fmean(
            _logical_samples(row["formal_row"]["status_path"]) for row in rows
        ),
    }


def collect_pair_units() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    formal = read_json(FORMAL_RESULTS)
    if formal.get("fit_allowed") is not True or formal.get("all_passed") is not True:
        raise RuntimeError("formal VL evidence gate has not passed")
    jobs = read_jsonl(QUEUE)
    formal_by_job = {str(row["job_id"]): row for row in formal["rows"]}
    if set(formal_by_job) != {str(job["job_id"]) for job in jobs}:
        raise RuntimeError("formal result and queue job sets differ")
    inventory = read_json(MODEL_INVENTORY)
    model_by_id = {str(model["id"]): model for model in inventory["models"]}

    grouped: dict[tuple[str, str, str, str], dict[str, list[dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for job in jobs:
        if job["arm_id"] not in {
            "text_length_matched",
            "real_image",
            "real_video",
        }:
            continue
        key = (
            str(job["track"]),
            str(job["model_id"]),
            str(job.get("media_tier")),
            str(job["mechanism_id"]),
        )
        grouped[key][str(job["arm_id"])].append(
            {"job": job, "formal_row": formal_by_job[str(job["job_id"])]}
        )

    pairs: list[dict[str, Any]] = []
    for key, arms in sorted(grouped.items()):
        real_arm = "real_image" if "real_image" in arms else "real_video"
        controls = arms.get("text_length_matched") or []
        real_rows = arms.get(real_arm) or []
        if not controls or not real_rows:
            raise RuntimeError(f"incomplete paired cell: {key}")
        all_rows = controls + real_rows
        classifications = Counter(
            str(row["formal_row"]["classification"]) for row in all_rows
        )
        real_job = real_rows[0]["job"]
        features = _profile_features(real_job, model_by_id[str(real_job["model_id"])])
        pair: dict[str, Any] = {
            "pair_unit_id": "::".join(key),
            "track": key[0],
            "model_id": key[1],
            "model_family": str(real_job["model_family"]),
            "media_tier": real_job.get("media_tier"),
            "mechanism_id": key[3],
            "physical_mbs": int(real_job["mbs"]),
            "gradient_checkpointing": bool(real_job["gradient_checkpointing"]),
            "classification_counts": dict(sorted(classifications.items())),
            "features": features,
            "memory_observation_kind": (
                "right_censored_base_and_real"
                if classifications.get("oom")
                else "paired_exact_success_centers"
            ),
        }
        if classifications.get("oom"):
            if classifications != Counter({"oom": len(all_rows)}):
                raise RuntimeError(f"mixed success/OOM pair is unsupported: {key}")
            lowers = [
                int(row["formal_row"]["right_censor_lower_bytes"]) for row in all_rows
            ]
            pair.update(
                {
                    "right_censor_lower_bytes": max(lowers),
                    "exact_peak_target_bytes": None,
                    "oom_peak_is_unknown_not_imputed": True,
                    "job_ids": [row["job"]["job_id"] for row in all_rows],
                }
            )
        else:
            if classifications != Counter({"success": len(all_rows)}):
                raise RuntimeError(f"non-terminal paired cell: {key}")
            control = _collapse_success_arm(controls)
            real = _collapse_success_arm(real_rows)
            signed_memory_delta = (
                real["max_reserved_bytes"] - control["max_reserved_bytes"]
            )
            signed_time_delta = real["measured_seconds"] - control["measured_seconds"]
            pair.update(
                {
                    "control": control,
                    "real_media": real,
                    "signed_memory_delta_bytes": signed_memory_delta,
                    "memory_fit_target_bytes": max(signed_memory_delta, 0.0),
                    "signed_visual_seconds": signed_time_delta,
                    "throughput_fit_target_seconds": max(signed_time_delta, 0.0),
                    "oom_peak_is_unknown_not_imputed": False,
                    "right_censor_lower_bytes": None,
                }
            )
        pairs.append(pair)
    if len(pairs) != 33:
        raise RuntimeError(f"expected 33 paired physical units, got {len(pairs)}")
    return pairs, formal


def _scalar_nnls(xs: Iterable[float], ys: Iterable[float]) -> float:
    clean = [(_finite(x, "x"), _finite(y, "y")) for x, y in zip(xs, ys)]
    denominator = sum(x * x for x, _ in clean)
    if denominator <= 0.0:
        raise ValueError("NNLS feature norm must be positive")
    return max(0.0, sum(x * y for x, y in clean) / denominator)


def _mape(predictions: Iterable[tuple[float, float]]) -> float:
    rows = list(predictions)
    if not rows or any(actual <= 0.0 for _, actual in rows):
        raise ValueError("MAPE requires positive observations")
    return statistics.fmean(abs(predicted / actual - 1.0) for predicted, actual in rows)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile needs values")
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def fit_memory_head(
    exact: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    families = sorted({str(row["model_family"]) for row in exact})
    predictions = []
    fold_coefficients = {}
    for family in families:
        train = [row for row in exact if row["model_family"] != family]
        test = [row for row in exact if row["model_family"] == family]
        coefficient = _scalar_nnls(
            (row["features"]["vision_dynamic_microbatch_bytes"] for row in train),
            (row["memory_fit_target_bytes"] for row in train),
        )
        fold_coefficients[family] = coefficient
        for row in test:
            predicted_delta = (
                coefficient * row["features"]["vision_dynamic_microbatch_bytes"]
            )
            predicted = row["control"]["max_reserved_bytes"] + predicted_delta
            actual = row["real_media"]["max_reserved_bytes"]
            predictions.append(
                {
                    "pair_unit_id": row["pair_unit_id"],
                    "held_out_family": family,
                    "predicted_visual_delta_bytes": predicted_delta,
                    "predicted_peak_bytes": predicted,
                    "observed_peak_bytes": actual,
                    "absolute_percentage_error": abs(predicted / actual - 1.0),
                    "signed_percentage_error": predicted / actual - 1.0,
                }
            )

    coefficient = _scalar_nnls(
        (row["features"]["vision_dynamic_microbatch_bytes"] for row in exact),
        (row["memory_fit_target_bytes"] for row in exact),
    )
    errors = [row["absolute_percentage_error"] for row in predictions]
    baseline_mape = _mape(
        (row["control"]["max_reserved_bytes"], row["real_media"]["max_reserved_bytes"])
        for row in exact
    )
    per_family = {}
    for family in families:
        family_rows = [row for row in predictions if row["held_out_family"] == family]
        per_family[family] = {
            "pair_units": len(family_rows),
            "mape": statistics.fmean(
                row["absolute_percentage_error"] for row in family_rows
            ),
            "max_absolute_percentage_error": max(
                row["absolute_percentage_error"] for row in family_rows
            ),
        }
    evaluation = {
        "validation": "leave_one_model_family_out",
        "fit_pair_units": len(exact),
        "fold_coefficients_bytes_per_proxy_byte": fold_coefficients,
        "baseline_text_only_mape": baseline_mape,
        "overlay_mape": statistics.fmean(errors),
        "overlay_p90_absolute_percentage_error": _percentile(errors, 90.0),
        "overlay_max_absolute_percentage_error": max(errors),
        "within_10_percent_rate": statistics.fmean(error <= 0.10 for error in errors),
        "underprediction_rate": statistics.fmean(
            row["signed_percentage_error"] < 0.0 for row in predictions
        ),
        "per_held_out_family": per_family,
        "target_mape_at_most_10_percent": statistics.fmean(errors) <= 0.10,
    }
    model = {
        "form": (
            "vl_peak_center_bytes = text_matched_peak_center_bytes + "
            "coefficient * vision_dynamic_microbatch_bytes"
        ),
        "coefficient_bytes_per_proxy_byte": coefficient,
        "coefficient_constraint": "nonnegative_scalar_nnls",
        "target": "max(real_media_peak - matched_text_peak, 0) bytes",
        "feature": "vision_dynamic_microbatch_bytes",
        "feature_definition": (
            "physical_mbs * 2 bytes * (pixel_elements + raw_patch_units * "
            "vision_hidden + visual_tokens * language_hidden)"
        ),
        "phase_semantics": (
            "the nonnegative residual approximates max(language_phase, vision_phase); "
            "the two phase peaks are never added independently"
        ),
    }
    return {"model": model, "evaluation": evaluation}, predictions


def _scenario_units(exact: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in exact:
        key = (
            str(row["model_family"]),
            str(row["model_id"]),
            str(row["track"]),
            str(row["media_tier"]),
        )
        grouped[key].append(row)
    scenarios = []
    for key, rows in sorted(grouped.items()):
        x_values = [
            row["features"]["vision_forward_flops_per_sample"]
            * row["real_media"]["logical_samples"]
            / 1.0e15
            for row in rows
        ]
        if max(x_values) - min(x_values) > max(x_values) * 1.0e-12:
            raise RuntimeError(f"scenario feature differs across mechanisms: {key}")
        scenarios.append(
            {
                "scenario_unit_id": "::".join(key),
                "model_family": key[0],
                "model_id": key[1],
                "track": key[2],
                "media_tier": key[3],
                "vision_forward_pflop": statistics.fmean(x_values),
                "visual_seconds_target": statistics.fmean(
                    row["throughput_fit_target_seconds"] for row in rows
                ),
                "mechanisms_collapsed": sorted(row["mechanism_id"] for row in rows),
            }
        )
    return scenarios


def _ranking_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        source = row["source"]
        grouped[
            (
                str(source["model_id"]),
                str(source["track"]),
                str(source["media_tier"]),
            )
        ].append(row)
    group_details = []
    pairwise_total = 0
    pairwise_correct = 0
    for key, rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        comparisons = 0
        correct = 0
        for left, right in itertools.combinations(rows, 2):
            actual_direction = (
                left["observed_effective_tokens_per_second"]
                - right["observed_effective_tokens_per_second"]
            )
            predicted_direction = (
                left["predicted_effective_tokens_per_second"]
                - right["predicted_effective_tokens_per_second"]
            )
            if actual_direction == 0.0:
                continue
            comparisons += 1
            correct += int(actual_direction * predicted_direction > 0.0)
        oracle = max(rows, key=lambda row: row["observed_effective_tokens_per_second"])
        chosen = max(rows, key=lambda row: row["predicted_effective_tokens_per_second"])
        regret = 1.0 - (
            chosen["observed_effective_tokens_per_second"]
            / oracle["observed_effective_tokens_per_second"]
        )
        pairwise_total += comparisons
        pairwise_correct += correct
        group_details.append(
            {
                "ranking_group": "::".join(key),
                "candidate_count": len(rows),
                "pairwise_comparisons": comparisons,
                "pairwise_correct": correct,
                "observed_best_mechanism": oracle["source"]["mechanism_id"],
                "predicted_best_mechanism": chosen["source"]["mechanism_id"],
                "top1_regret": regret,
            }
        )
    regrets = [row["top1_regret"] for row in group_details]
    return {
        "ranking_groups": len(group_details),
        "pairwise_comparisons": pairwise_total,
        "pairwise_order_accuracy": (
            pairwise_correct / pairwise_total if pairwise_total else None
        ),
        "top1_exact_rate": statistics.fmean(
            row["observed_best_mechanism"] == row["predicted_best_mechanism"]
            for row in group_details
        ),
        "mean_top1_regret": statistics.fmean(regrets),
        "max_top1_regret": max(regrets),
        "details": group_details,
    }


def fit_throughput_head(
    exact: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenarios = _scenario_units(exact)
    families = sorted({str(row["model_family"]) for row in scenarios})
    coefficient_by_holdout: dict[str, float] = {}
    for held_out in scenarios:
        train = [
            row
            for row in scenarios
            if row["model_family"] == held_out["model_family"]
            and row["scenario_unit_id"] != held_out["scenario_unit_id"]
        ]
        coefficient_by_holdout[held_out["scenario_unit_id"]] = _scalar_nnls(
            (row["vision_forward_pflop"] for row in train),
            (row["visual_seconds_target"] for row in train),
        )

    predictions = []
    for row in exact:
        scenario_id = "::".join(
            (
                str(row["model_family"]),
                str(row["model_id"]),
                str(row["track"]),
                str(row["media_tier"]),
            )
        )
        x_value = (
            row["features"]["vision_forward_flops_per_sample"]
            * row["real_media"]["logical_samples"]
            / 1.0e15
        )
        coefficient = coefficient_by_holdout[scenario_id]
        predicted_visual_seconds = coefficient * x_value
        predicted_seconds = (
            row["control"]["measured_seconds"] + predicted_visual_seconds
        )
        observed_seconds = row["real_media"]["measured_seconds"]
        tokens = row["real_media"]["effective_tokens"]
        predictions.append(
            {
                "pair_unit_id": row["pair_unit_id"],
                "held_out_scenario_unit_id": scenario_id,
                "coefficient_seconds_per_pflop": coefficient,
                "predicted_visual_seconds": predicted_visual_seconds,
                "predicted_step_seconds": predicted_seconds,
                "observed_step_seconds": observed_seconds,
                "absolute_percentage_error": abs(
                    predicted_seconds / observed_seconds - 1.0
                ),
                "predicted_effective_tokens_per_second": tokens / predicted_seconds,
                "observed_effective_tokens_per_second": tokens / observed_seconds,
                "source": {
                    "model_id": row["model_id"],
                    "model_family": row["model_family"],
                    "track": row["track"],
                    "media_tier": row["media_tier"],
                    "mechanism_id": row["mechanism_id"],
                },
            }
        )

    family_coefficients = {}
    for family in families:
        train = [row for row in scenarios if row["model_family"] == family]
        family_coefficients[family] = _scalar_nnls(
            (row["vision_forward_pflop"] for row in train),
            (row["visual_seconds_target"] for row in train),
        )
    errors = [row["absolute_percentage_error"] for row in predictions]
    baseline_mape = _mape(
        (row["control"]["measured_seconds"], row["real_media"]["measured_seconds"])
        for row in exact
    )
    per_family = {}
    for family in families:
        family_rows = [
            row for row in predictions if row["source"]["model_family"] == family
        ]
        per_family[family] = {
            "pair_units": len(family_rows),
            "mape": statistics.fmean(
                row["absolute_percentage_error"] for row in family_rows
            ),
            "max_absolute_percentage_error": max(
                row["absolute_percentage_error"] for row in family_rows
            ),
        }
    ranking = _ranking_metrics(predictions)
    evaluation = {
        "validation": (
            "leave_one_media_scenario_out_within_family; all mechanisms of the "
            "held-out model/modality/tier scenario are collapsed before fitting"
        ),
        "scenario_units": len(scenarios),
        "pair_units": len(exact),
        "baseline_text_only_step_time_mape": baseline_mape,
        "overlay_step_time_mape": statistics.fmean(errors),
        "overlay_p90_absolute_percentage_error": _percentile(errors, 90.0),
        "overlay_max_absolute_percentage_error": max(errors),
        "within_10_percent_rate": statistics.fmean(error <= 0.10 for error in errors),
        "per_family": per_family,
        "ranking": ranking,
        "target_step_time_mape_at_most_10_percent": statistics.fmean(errors) <= 0.10,
        "target_pairwise_order_accuracy_at_least_90_percent": (
            ranking["pairwise_order_accuracy"] is not None
            and ranking["pairwise_order_accuracy"] >= 0.90
        ),
        "target_max_top1_regret_below_10_percent": ranking["max_top1_regret"] < 0.10,
    }
    model = {
        "form": (
            "vl_step_seconds = text_matched_step_seconds + family_coefficient * "
            "vision_forward_pflop"
        ),
        "family_coefficients_seconds_per_pflop": family_coefficients,
        "coefficient_constraint": "nonnegative_scalar_nnls_per_model_family",
        "target": "max(real_media_seconds - matched_text_seconds, 0) seconds",
        "feature": "vision_forward_flops_per_sample * logical_samples / 1e15",
        "ranking_policy": "apply memory admission first, then rank predicted tokens/s",
    }
    return {
        "model": model,
        "evaluation": evaluation,
        "scenarios": scenarios,
    }, predictions


def _write_markdown(report: dict[str, Any], model_path: Path) -> None:
    memory = report["memory_evaluation"]
    throughput = report["throughput_evaluation"]
    ranking = throughput["ranking"]
    evidence = report["evidence"]
    checks = report["checks"]
    lines = [
        "# H800 冻结视觉塔 VL 汇总建模与正式评估 V1",
        "",
        "## 结论",
        "",
        (
            f"正式队列共 {evidence['jobs']} 个任务：{evidence['successful_jobs']} 个成功，"
            f"{evidence['oom_jobs']} 个 CUDA OOM。重复实验先合并为 "
            f"{evidence['paired_physical_units']} 个物理配对单元，其中 "
            f"{evidence['paired_exact_units']} 个是精确标签，"
            f"{evidence['paired_right_censored_units']} 个只作为右删失约束。"
        ),
        "",
        "| 指标 | 结果 | 门槛 | 是否通过 |",
        "|---|---:|---:|---:|",
        (
            f"| 显存峰值留一模型族 MAPE | {memory['overlay_mape']:.2%} | ≤10% | "
            f"{checks['memory_leave_family_out_mape']} |"
        ),
        (
            f"| 吞吐步时留一场景 MAPE | {throughput['overlay_step_time_mape']:.2%} | ≤10% | "
            f"{checks['throughput_leave_scenario_out_mape']} |"
        ),
        (
            f"| 配置两两排序准确率 | {ranking['pairwise_order_accuracy']:.2%} | ≥90% | "
            f"{checks['throughput_pairwise_order_accuracy']} |"
        ),
        (
            f"| Top-1 最大后悔率 | {ranking['max_top1_regret']:.2%} | <10% | "
            f"{checks['throughput_top1_regret']} |"
        ),
        "",
        "## 建模口径",
        "",
        "- 显存精确标签只取成功任务的 `max_reserved_bytes`。CUDA OOM 不补峰值，只记设备容量下界。",
        "- 视觉残差由真实图片/视频减去后处理 token 长度匹配的文本对照得到；负差值按 0 拟合，避免把测量抖动解释为负视觉显存或负视觉耗时。",
        "- 显存峰值按语言阶段与视觉阶段的 phase-max 解释，不能把两个阶段峰值直接相加。",
        "- 吞吐头按模型族分别拟合，因为实测中 Qwen2.5-VL 与 Qwen3-VL/Qwen3.5 的视觉前向效率不是一个系数。",
        "",
        "## 当前边界",
        "",
        "该模型只覆盖 H800 140GB、单卡、LoRA、ZeRO-0、不开 Packing、冻结视觉塔和多模态投影层。当前评估是开发集分组交叉验证，不是预先冻结的来源隔离前瞻测试，因此模型只进入 shadow，不允许自动准入或正式推荐。",
        "",
        f"模型产物：`{model_path}`",
        "",
    ]
    MARKDOWN_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def fit_and_evaluate() -> tuple[dict[str, Any], dict[str, Any]]:
    pairs, formal = collect_pair_units()
    exact = [
        row
        for row in pairs
        if row["memory_observation_kind"] == "paired_exact_success_centers"
    ]
    censored = [
        row
        for row in pairs
        if row["memory_observation_kind"] == "right_censored_base_and_real"
    ]
    if len(exact) != 31 or len(censored) != 2:
        raise RuntimeError(
            f"expected 31 exact and 2 censored pairs, got {len(exact)} and {len(censored)}"
        )

    memory, memory_predictions = fit_memory_head(exact)
    throughput, throughput_predictions = fit_throughput_head(exact)
    generated_at = datetime.now(timezone.utc).isoformat()
    model: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "generated_at_utc": generated_at,
        "hardware": "NVIDIA H800 140GB HBM3",
        "memory_center_head": memory["model"],
        "throughput_head": throughput["model"],
        "input_contract": {
            "required_base_predictions": [
                "text_memory_center_bytes",
                "text_step_seconds",
                "effective_tokens_per_step",
            ],
            "required_workload_aggregates": [
                "mean_visual_tokens_per_sample",
                "mean_raw_patch_units_per_sample",
                "mean_pixel_value_elements_per_sample",
                "mean_sampled_video_frames_per_sample",
                "mean_media_per_sample",
            ],
            "required_model_fields": [
                "model_family",
                "vision_depth",
                "vision_hidden_size",
                "vision_intermediate_size",
                "vision_output_hidden_size",
                "vision_parameter_elements",
                "projector_parameter_elements",
            ],
            "required_training_fields": [
                "physical_mbs",
                "logical_samples_per_step",
                "freeze_vision_tower=true",
                "freeze_multi_modal_projector=true",
                "packing=false",
            ],
            "raw_images_or_videos_required_at_recommendation_time": False,
            "aggregate_profile_only": True,
        },
        "fit_scope": {
            "model_ids": sorted({row["model_id"] for row in exact}),
            "model_families": sorted({row["model_family"] for row in exact}),
            "modalities": ["image", "video"],
            "train_type": "lora",
            "gpu_count": 1,
            "zero_stage": 0,
            "packing": False,
            "vision_tower_frozen": True,
            "multimodal_projector_frozen": True,
        },
        "release_contract": {
            "mode": "shadow_only",
            "automatic_admission_allowed": False,
            "automatic_ranking_allowed": False,
            "reason": "source-disjoint prospective acceptance has not run",
        },
        "source_bindings": {
            "queue_sha256": sha256_file(QUEUE),
            "formal_results_sha256": sha256_file(FORMAL_RESULTS),
            "model_inventory_sha256": sha256_file(MODEL_INVENTORY),
        },
    }
    model["artifact_sha256"] = sha256_json(model)
    write_json(MODEL_OUTPUT, model)

    checks = {
        "formal_evidence_gate": formal.get("fit_allowed") is True,
        "memory_leave_family_out_mape": memory["evaluation"][
            "target_mape_at_most_10_percent"
        ],
        "throughput_leave_scenario_out_mape": throughput["evaluation"][
            "target_step_time_mape_at_most_10_percent"
        ],
        "throughput_pairwise_order_accuracy": throughput["evaluation"][
            "target_pairwise_order_accuracy_at_least_90_percent"
        ],
        "throughput_top1_regret": throughput["evaluation"][
            "target_max_top1_regret_below_10_percent"
        ],
        "oom_never_used_as_exact_peak": all(
            row.get("exact_peak_target_bytes") is None
            and row.get("oom_peak_is_unknown_not_imputed") is True
            for row in censored
        ),
        "physical_repeats_collapsed_before_fit": any(
            row["control"]["repeat_count"] > 1 or row["real_media"]["repeat_count"] > 1
            for row in exact
        ),
    }
    evaluation: dict[str, Any] = {
        "schema": "sft_h800_frozen_vl_modeling_formal_evaluation/v1",
        "generated_at_utc": generated_at,
        "model_artifact": {
            "path": str(MODEL_OUTPUT.resolve()),
            "sha256": sha256_file(MODEL_OUTPUT),
        },
        "source_bindings": model["source_bindings"],
        "evidence": {
            "jobs": len(formal["rows"]),
            "successful_jobs": int(formal["classifications"].get("success") or 0),
            "oom_jobs": int(formal["classifications"].get("oom") or 0),
            "paired_physical_units": len(pairs),
            "paired_exact_units": len(exact),
            "paired_right_censored_units": len(censored),
            "negative_signed_memory_deltas_clipped_to_zero": sum(
                row["signed_memory_delta_bytes"] < 0.0 for row in exact
            ),
            "negative_signed_time_deltas_clipped_to_zero": sum(
                row["signed_visual_seconds"] < 0.0 for row in exact
            ),
        },
        "memory_evaluation": memory["evaluation"],
        "throughput_evaluation": throughput["evaluation"],
        "checks": checks,
        "all_development_gates_passed": all(checks.values()),
        "formal_campaign_fit_complete": all(checks.values()),
        "automatic_recommendation_release_allowed": False,
        "acceptance_status": "development_pass_shadow_only",
        "blocking_acceptance_gap": (
            "freeze this overlay, then run an unseen source-disjoint prospective "
            "image/video holdout before enabling admission or ranking"
        ),
        "right_censored_pairs": censored,
        "paired_units": pairs,
        "memory_cross_validation_predictions": memory_predictions,
        "throughput_cross_validation_predictions": throughput_predictions,
        "throughput_fit_scenarios": throughput["scenarios"],
    }
    evaluation["report_sha256"] = sha256_json(evaluation)
    write_json(EVALUATION_OUTPUT, evaluation)
    _write_markdown(evaluation, MODEL_OUTPUT.resolve())
    return model, evaluation


def main() -> None:
    model, evaluation = fit_and_evaluate()
    print(
        json.dumps(
            {
                "model_output": str(MODEL_OUTPUT.resolve()),
                "evaluation_output": str(EVALUATION_OUTPUT.resolve()),
                "markdown_output": str(MARKDOWN_OUTPUT.resolve()),
                "memory_coefficient": model["memory_center_head"][
                    "coefficient_bytes_per_proxy_byte"
                ],
                "throughput_coefficients": model["throughput_head"][
                    "family_coefficients_seconds_per_pflop"
                ],
                "checks": evaluation["checks"],
                "all_development_gates_passed": evaluation[
                    "all_development_gates_passed"
                ],
                "automatic_recommendation_release_allowed": evaluation[
                    "automatic_recommendation_release_allowed"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if evaluation["all_development_gates_passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
