#!/usr/bin/env python3
"""Compare fit-only Packing throughput challengers and diagnose memory residuals.

The script deliberately works at the matched-setting level.  Repeated GPU runs
are collapsed before model selection, and leave-one-profile-out folds keep all
cutoffs and repeats from the same upstream profile together.  The output is a
fit-only diagnostic; it cannot enable automatic Packing recommendations.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler

from common import ARTIFACT_DIR, percentile, read_json, sha256_file, sha256_json, write_json


PHASE_B = ARTIFACT_DIR / "h800_packing_profile_phase_b_results_v1.json"
REAL_BUSINESS = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_results_v1.json"
INTERACTIONS = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_combined_results_v1.json"
GBS_REPAIR = ARTIFACT_DIR / "h800_packing_gbs_repair_batch_results_v1.json"
MEMBERSHIP = ARTIFACT_DIR / "packing_fit_membership_v1.json"
REAL_STATIC = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_static_features_v1.json"
INTERACTION_STATIC = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_static_features_v1.json"
PREFLIGHT = ARTIFACT_DIR / "h800_packing_profile_phase_b_memory_predictions_v1.json"

OUTPUT = ARTIFACT_DIR / "h800_packing_phase_b_model_selection_v1.json"
MARKDOWN = ARTIFACT_DIR / "h800_packing_phase_b_model_selection_v1.md"

FEATURE_NAMES = (
    "log2_cutoff",
    "log2_mean_length",
    "length_cv",
    "p99_length_to_cutoff",
    "pack_fill_ratio",
    "log2_n_pack_mean",
    "log2_packed_ga",
    "zero2",
    "log2_gpu_count",
)
SPLINE_FEATURES = ("log2_cutoff", "p99_length_to_cutoff", "log2_n_pack_mean")
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
GIB = float(2**30)


def _metric(row: dict[str, Any], name: str) -> float:
    aliases = {
        "logical": ("global_logical_samples_per_second", "logical_samples_per_second"),
        "effective": ("global_effective_tokens_per_second", "effective_tokens_per_second"),
        "reserved": ("max_reserved_gib",),
    }
    for key in aliases[name]:
        value = row.get(key)
        if value is not None:
            return float(value)
    raise ValueError(f"{row.get('job_id')} has no {name} metric")


@lru_cache(maxsize=None)
def _profile_stats(path_text: str) -> dict[str, float]:
    path = Path(path_text)
    lengths: list[float] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            value = json.loads(line).get("total_tokens")
            if value is not None:
                lengths.append(float(value))
    if not lengths:
        raise ValueError(f"profile has no total_tokens: {path}")
    mean = statistics.fmean(lengths)
    return {
        "records": float(len(lengths)),
        "mean": mean,
        "std": statistics.pstdev(lengths),
        "cv": statistics.pstdev(lengths) / mean,
        "p99": percentile(lengths, 99),
        "maximum": max(lengths),
    }


def _source_rows() -> list[dict[str, Any]]:
    phase_b = read_json(PHASE_B)
    required_gates = phase_b.get("gates", {})
    if not (
        required_gates.get("packing_semantics_and_ledger_passed") is True
        and required_gates.get("all_treatment_cv_le_0p05") is True
        and required_gates.get("phase_b_complete_without_extra_repeats") is True
    ):
        raise ValueError("Phase B has not passed the frozen fit-only gates")

    membership = read_json(MEMBERSHIP)
    strict_ids = {
        str(row["job_id"])
        for row in membership["rows"]
        if row.get("evidence_role") == "strict_target_gbs_fit_only"
    }
    sources = (
        ("real_business", read_json(REAL_BUSINESS)["job_results"], None),
        ("strict_interactions", read_json(INTERACTIONS)["job_results"], strict_ids),
        ("strict_gbs_repair", read_json(GBS_REPAIR)["job_results"], strict_ids),
        ("phase_b", phase_b["job_results"], None),
    )
    rows: list[dict[str, Any]] = []
    for source_name, source_rows, allow_ids in sources:
        for original in source_rows:
            if original.get("classification") != "success":
                continue
            if original.get("calibration_eligible") is False:
                continue
            if allow_ids is not None and str(original.get("job_id")) not in allow_ids:
                continue
            if source_name == "real_business" and int(original.get("mbs", 0)) != 1:
                continue
            row = dict(original)
            row["fit_source"] = source_name
            rows.append(row)
    return rows


def _setting_identity(row: dict[str, Any]) -> tuple[str, str]:
    source = str(row["fit_source"])
    if source == "real_business":
        profile_group = f"real:{row['family_id']}"
        setting = f"{profile_group}:c{int(row['cutoff_len'])}"
    elif source == "phase_b":
        profile_group = f"phase_b:{str(row['workload_id']).upper()}"
        setting = f"phase_b:{row['family_id']}"
    else:
        family = str(row.get("profile_family_id") or row.get("dataset_id"))
        if family.startswith("w1_"):
            profile_group = "strict:W1"
        elif family.startswith("w4_"):
            profile_group = "strict:W4"
        else:
            profile_group = f"strict:{family}"
        setting = f"{source}:{row['setting_id']}"
    return setting, profile_group


def _static_feature_maps() -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, dict[str, Any]]]:
    real_rows = read_json(REAL_STATIC)["rows"]
    real = {
        (str(row["dataset_key"]), int(row["cutoff_len"])): row["decision"]["features"]
        for row in real_rows
    }
    interaction = {
        str(row["setting_id"]): row["platform_estimate"]
        for row in read_json(INTERACTION_STATIC)["rows"]
    }
    return real, interaction


def _n_pack_mean(
    packed: dict[str, Any],
    real_static: dict[tuple[str, int], dict[str, Any]],
    interaction_static: dict[str, dict[str, Any]],
) -> float:
    if packed.get("n_pack_mean") is not None:
        return float(packed["n_pack_mean"])
    if packed["fit_source"] == "real_business":
        key = (str(packed["family_id"]), int(packed["cutoff_len"]))
        return float(real_static[key]["mean_samples_per_pack"])
    if packed["fit_source"] == "strict_interactions":
        return float(interaction_static[str(packed["setting_id"])]["n_pack"]["center"])
    raise ValueError(f"no n_pack_mean for {packed.get('job_id')}")


def _modeling_role(row: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not bool(row.get("gc")):
        reasons.append("gc_off_has_only_one_independent_profile")
    if int(row.get("zero_stage", 0)) == 3:
        reasons.append("zero3_has_only_one_independent_profile")
    return ("interaction_diagnostic_only" if reasons else "primary_model_selection"), reasons


def build_setting_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    real_static, interaction_static = _static_feature_maps()
    grouped: dict[tuple[str, int], dict[bool, dict[str, Any]]] = defaultdict(dict)
    for row in _source_rows():
        setting, _ = _setting_identity(row)
        key = (setting, int(row["repeat"]))
        treatment = bool(row["packing"])
        if treatment in grouped[key]:
            raise ValueError(f"duplicate treatment in pair: {key}/{treatment}")
        grouped[key][treatment] = row

    pair_rows: list[dict[str, Any]] = []
    for (setting, repeat), treatments in sorted(grouped.items()):
        if set(treatments) != {False, True}:
            raise ValueError(f"incomplete matched pair: {setting}/{repeat}")
        unpacked, packed = treatments[False], treatments[True]
        _, profile_group = _setting_identity(packed)
        profile = _profile_stats(str(packed["dataset_profile_path"]))
        cutoff = float(packed["cutoff_len"])
        n_pack = _n_pack_mean(packed, real_static, interaction_static)
        role, reasons = _modeling_role(packed)
        features = {
            "log2_cutoff": math.log2(cutoff),
            "log2_mean_length": math.log2(profile["mean"]),
            "length_cv": profile["cv"],
            "p99_length_to_cutoff": profile["p99"] / cutoff,
            "pack_fill_ratio": min(1.25, profile["mean"] * n_pack / cutoff),
            "log2_n_pack_mean": math.log2(n_pack),
            "log2_packed_ga": math.log2(float(packed["gradient_accumulation_steps"])),
            "zero2": float(int(packed.get("zero_stage", 0)) == 2),
            "log2_gpu_count": math.log2(float(packed["gpu_count"])),
        }
        pair_rows.append(
            {
                "setting_id": setting,
                "profile_group": profile_group,
                "source": packed["fit_source"],
                "repeat": repeat,
                "role": role,
                "exclusion_reasons": reasons,
                "cutoff_len": int(cutoff),
                "gpu_count": int(packed["gpu_count"]),
                "zero_stage": int(packed.get("zero_stage", 0)),
                "gc": bool(packed.get("gc")),
                "n_pack_mean": n_pack,
                "packed_ga": int(packed["gradient_accumulation_steps"]),
                "features": features,
                "log_effective_ratio": math.log(
                    _metric(packed, "effective") / _metric(unpacked, "effective")
                ),
                "log_logical_ratio": math.log(
                    _metric(packed, "logical") / _metric(unpacked, "logical")
                ),
                "packed_reserved_gib": _metric(packed, "reserved"),
                "unpacked_reserved_gib": _metric(unpacked, "reserved"),
            }
        )

    by_setting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_setting[row["setting_id"]].append(row)
    setting_rows: list[dict[str, Any]] = []
    for setting, repeats in sorted(by_setting.items()):
        representative = repeats[0]
        for row in repeats[1:]:
            if row["features"] != representative["features"]:
                raise ValueError(f"features vary across repeats for {setting}")
        effective = [float(row["log_effective_ratio"]) for row in repeats]
        logical = [float(row["log_logical_ratio"]) for row in repeats]
        setting_rows.append(
            {
                key: representative[key]
                for key in (
                    "setting_id", "profile_group", "source", "role", "exclusion_reasons",
                    "cutoff_len", "gpu_count", "zero_stage", "gc", "n_pack_mean",
                    "packed_ga", "features",
                )
            }
            | {
                "repeat_count": len(repeats),
                "log_effective_ratio": statistics.fmean(effective),
                "log_effective_ratio_repeat_std": statistics.pstdev(effective),
                "effective_ratio": math.exp(statistics.fmean(effective)),
                "log_logical_ratio": statistics.fmean(logical),
                "log_logical_ratio_repeat_std": statistics.pstdev(logical),
                "logical_ratio": math.exp(statistics.fmean(logical)),
                "packed_reserved_gib_mean": statistics.fmean(
                    float(row["packed_reserved_gib"]) for row in repeats
                ),
                "unpacked_reserved_gib_mean": statistics.fmean(
                    float(row["unpacked_reserved_gib"]) for row in repeats
                ),
            }
        )
    return setting_rows, pair_rows


def _raw_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(row["features"][name]) for name in FEATURE_NAMES] for row in rows],
        dtype=float,
    )


def _design_builder(kind: str, train_raw: np.ndarray) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    if kind == "ridge_linear":
        def transform(values: np.ndarray) -> np.ndarray:
            return values
        return transform, {"kind": kind, "raw_feature_names": list(FEATURE_NAMES)}
    if kind != "gam_spline":
        raise ValueError(f"unknown model kind: {kind}")
    spline_indices = [FEATURE_NAMES.index(name) for name in SPLINE_FEATURES]
    linear_indices = [index for index in range(len(FEATURE_NAMES)) if index not in spline_indices]
    spline = SplineTransformer(
        n_knots=3,
        degree=2,
        include_bias=False,
        extrapolation="linear",
    ).fit(train_raw[:, spline_indices])

    def transform(values: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (spline.transform(values[:, spline_indices]), values[:, linear_indices]),
            axis=1,
        )

    return transform, {
        "kind": kind,
        "spline_feature_names": list(SPLINE_FEATURES),
        "linear_feature_names": [FEATURE_NAMES[index] for index in linear_indices],
        "n_knots": 3,
        "degree": 2,
        "extrapolation": "linear",
    }


def _fit_predict(
    kind: str,
    alpha: float,
    train_rows: list[dict[str, Any]],
    train_y: np.ndarray,
    test_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    train_raw = _raw_matrix(train_rows)
    test_raw = _raw_matrix(test_rows)
    transform, design = _design_builder(kind, train_raw)
    train_design = transform(train_raw)
    test_design = transform(test_raw)
    scaler = StandardScaler().fit(train_design)
    model = Ridge(alpha=alpha).fit(scaler.transform(train_design), train_y)
    prediction = model.predict(scaler.transform(test_design))
    return prediction, {
        "design": design,
        "standardizer_mean": scaler.mean_.tolist(),
        "standardizer_scale": scaler.scale_.tolist(),
        "intercept": float(model.intercept_),
        "coefficients": model.coef_.tolist(),
    }


def _group_equal_mae(rows: list[dict[str, Any]], y: np.ndarray, prediction: np.ndarray) -> float:
    errors: dict[str, list[float]] = defaultdict(list)
    for row, actual, predicted in zip(rows, y, prediction, strict=True):
        errors[str(row["profile_group"])].append(abs(float(actual) - float(predicted)))
    return statistics.fmean(statistics.fmean(values) for values in errors.values())


def _select_alpha(kind: str, rows: list[dict[str, Any]], y: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    groups = sorted({str(row["profile_group"]) for row in rows})
    scores: list[dict[str, float]] = []
    for alpha in ALPHAS:
        predictions = np.empty(len(rows), dtype=float)
        for group in groups:
            train_indices = [i for i, row in enumerate(rows) if row["profile_group"] != group]
            test_indices = [i for i, row in enumerate(rows) if row["profile_group"] == group]
            fold_prediction, _ = _fit_predict(
                kind,
                alpha,
                [rows[i] for i in train_indices],
                y[train_indices],
                [rows[i] for i in test_indices],
            )
            predictions[test_indices] = fold_prediction
        scores.append(
            {
                "alpha": alpha,
                "group_equal_mae_log_ratio": _group_equal_mae(rows, y, predictions),
            }
        )
    selected = min(scores, key=lambda row: (row["group_equal_mae_log_ratio"], -row["alpha"]))
    return float(selected["alpha"]), scores


def _prediction_metrics(rows: list[dict[str, Any]], y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(y - prediction)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["profile_group"])].append(index)
    group_mae = {
        group: float(np.mean(absolute[indices])) for group, indices in groups.items()
    }
    top1 = []
    for group, indices in groups.items():
        if len(indices) < 2:
            continue
        actual_best = max(indices, key=lambda i: y[i])
        selected = max(indices, key=lambda i: prediction[i])
        top1.append(
            {
                "profile_group": group,
                "candidates": len(indices),
                "selected_setting_id": rows[selected]["setting_id"],
                "actual_best_setting_id": rows[actual_best]["setting_id"],
                "exact": selected == actual_best,
                "regret_ratio": math.exp(float(y[actual_best] - y[selected])),
            }
        )
    group_residual = [max(absolute[indices]) for indices in groups.values()]
    conformal_q95 = max(group_residual)
    return {
        "rows": len(rows),
        "profile_groups": len(groups),
        "row_mae_log_ratio": float(np.mean(absolute)),
        "group_equal_mae_log_ratio": statistics.fmean(group_mae.values()),
        "rmse_log_ratio": float(np.sqrt(np.mean((y - prediction) ** 2))),
        "median_absolute_ratio_error": float(np.median(np.abs(np.exp(prediction - y) - 1.0))),
        "worst_profile_group_mae_log_ratio": max(group_mae.values()),
        "gain_sign_accuracy": float(np.mean((prediction > 0.0) == (y > 0.0))),
        "gain_gt_20pct_accuracy": float(
            np.mean((prediction >= math.log(1.2)) == (y >= math.log(1.2)))
        ),
        "top1": {
            "eligible_profile_groups": len(top1),
            "exact_accuracy": (
                statistics.fmean(float(row["exact"]) for row in top1) if top1 else None
            ),
            "mean_regret_ratio": (
                statistics.fmean(float(row["regret_ratio"]) for row in top1) if top1 else None
            ),
            "maximum_regret_ratio": max(
                (float(row["regret_ratio"]) for row in top1), default=None
            ),
            "rows": top1,
        },
        "diagnostic_lopo_absolute_residual_q95_log": float(conformal_q95),
        "diagnostic_lopo_multiplicative_interval": math.exp(float(conformal_q95)),
        "per_profile_group_mae_log_ratio": group_mae,
    }


def _cross_validate(kind: str, rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    y = np.asarray([float(row[target]) for row in rows], dtype=float)
    groups = sorted({str(row["profile_group"]) for row in rows})
    prediction = np.empty(len(rows), dtype=float)
    outer_folds = []
    for group in groups:
        train_indices = [i for i, row in enumerate(rows) if row["profile_group"] != group]
        test_indices = [i for i, row in enumerate(rows) if row["profile_group"] == group]
        train_rows = [rows[i] for i in train_indices]
        alpha, _ = _select_alpha(kind, train_rows, y[train_indices])
        fold_prediction, _ = _fit_predict(
            kind, alpha, train_rows, y[train_indices], [rows[i] for i in test_indices]
        )
        prediction[test_indices] = fold_prediction
        outer_folds.append(
            {
                "held_out_profile_group": group,
                "selected_alpha": alpha,
                "test_setting_ids": [rows[i]["setting_id"] for i in test_indices],
            }
        )
    final_alpha, final_alpha_scores = _select_alpha(kind, rows, y)
    _, final_fit = _fit_predict(kind, final_alpha, rows, y, rows)
    return {
        "kind": kind,
        "target": target,
        "nested_lopo": True,
        "metrics": _prediction_metrics(rows, y, prediction),
        "outer_folds": outer_folds,
        "oof_predictions": [
            {
                "setting_id": row["setting_id"],
                "profile_group": row["profile_group"],
                "actual_log_ratio": float(actual),
                "predicted_log_ratio": float(predicted),
                "actual_ratio": math.exp(float(actual)),
                "predicted_ratio": math.exp(float(predicted)),
            }
            for row, actual, predicted in zip(rows, y, prediction, strict=True)
        ],
        "final_fit_only_model": {
            "alpha": final_alpha,
            "alpha_selection_scores": final_alpha_scores,
            **final_fit,
        },
    }


def _select_challenger(ridge: dict[str, Any], gam: dict[str, Any]) -> dict[str, Any]:
    r = ridge["metrics"]
    g = gam["metrics"]
    improvement = 1.0 - float(g["group_equal_mae_log_ratio"]) / float(
        r["group_equal_mae_log_ratio"]
    )
    gam_passes = (
        improvement >= 0.10
        and float(g["worst_profile_group_mae_log_ratio"])
        <= 1.05 * float(r["worst_profile_group_mae_log_ratio"])
        and float(g["top1"]["maximum_regret_ratio"])
        <= float(r["top1"]["maximum_regret_ratio"]) + 1e-12
    )
    return {
        "selected": "gam_spline" if gam_passes else "ridge_linear",
        "selection_scope": "paired_effect_center_fit_only_not_production_ranker",
        "gam_group_equal_mae_relative_improvement": improvement,
        "rule": {
            "minimum_group_equal_mae_improvement": 0.10,
            "maximum_worst_group_mae_ratio": 1.05,
            "gam_top1_maximum_regret_must_not_exceed_ridge": True,
        },
        "gam_passed_rule": gam_passes,
    }


def _ridge_matrix_fit_predict(
    alpha: float,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    scaler = StandardScaler().fit(train_x)
    model = Ridge(alpha=alpha).fit(scaler.transform(train_x), train_y)
    return model.predict(scaler.transform(test_x))


def _ridge_matrix_cv(
    rows: list[dict[str, Any]],
    feature_names: tuple[str, ...],
    *,
    target: str,
) -> dict[str, Any]:
    x = np.asarray(
        [[float(row["features"][name]) for name in feature_names] for row in rows],
        dtype=float,
    )
    y = np.asarray([float(row[target]) for row in rows], dtype=float)
    groups = sorted({str(row["profile_group"]) for row in rows})
    prediction = np.empty(len(rows), dtype=float)
    fold_alphas = []
    for held_out in groups:
        train = np.asarray(
            [i for i, row in enumerate(rows) if row["profile_group"] != held_out],
            dtype=int,
        )
        test = np.asarray(
            [i for i, row in enumerate(rows) if row["profile_group"] == held_out],
            dtype=int,
        )
        inner_groups = sorted({str(rows[i]["profile_group"]) for i in train})
        alpha_scores = []
        for alpha in ALPHAS:
            inner_prediction = np.empty(len(train), dtype=float)
            train_rows = [rows[i] for i in train]
            for inner_held_out in inner_groups:
                inner_train_local = np.asarray(
                    [
                        j
                        for j, row in enumerate(train_rows)
                        if row["profile_group"] != inner_held_out
                    ],
                    dtype=int,
                )
                inner_test_local = np.asarray(
                    [
                        j
                        for j, row in enumerate(train_rows)
                        if row["profile_group"] == inner_held_out
                    ],
                    dtype=int,
                )
                inner_prediction[inner_test_local] = _ridge_matrix_fit_predict(
                    alpha,
                    x[train[inner_train_local]],
                    y[train[inner_train_local]],
                    x[train[inner_test_local]],
                )
            alpha_scores.append(
                (
                    _group_equal_mae(train_rows, y[train], inner_prediction),
                    -alpha,
                    alpha,
                )
            )
        selected_alpha = float(min(alpha_scores)[2])
        fold_alphas.append({"held_out_profile_group": held_out, "alpha": selected_alpha})
        prediction[test] = _ridge_matrix_fit_predict(
            selected_alpha, x[train], y[train], x[test]
        )
    metrics = _prediction_metrics(rows, y, prediction)
    return {
        "model_family": "nested_lopo_standardized_ridge",
        "feature_names": list(feature_names),
        "target": target,
        "metrics": metrics,
        "fold_alphas": fold_alphas,
        "oof_predictions": [
            {
                "arm_id": row["arm_id"],
                "profile_group": row["profile_group"],
                "actual": float(actual),
                "predicted": float(predicted),
            }
            for row, actual, predicted in zip(rows, y, prediction, strict=True)
        ],
    }


def _phase_b_memory_feature_comparison() -> dict[str, Any]:
    result = read_json(PHASE_B)
    job_rows = []
    for row in result["job_results"]:
        attempt = str(row["execution_attempt_id"])
        metrics = (
            Path(row.get("result_path", ""))
            if row.get("result_path")
            else Path(__file__).resolve().parents[1]
            / "results"
            / str(row["job_id"])
            / "attempts"
            / attempt
            / "metrics"
        )
        summaries = sorted(metrics.glob("summary.rank*.json"))
        if not summaries:
            raise ValueError(f"missing summaries for {row['job_id']}")
        rank_features = []
        for path in summaries:
            summary = read_json(path)
            totals = summary["measured_totals"]
            batches = float(totals["physical_batches"])
            rank_features.append(
                {
                    "computed_tokens_per_batch": float(totals["computed_tokens"]) / batches,
                    "attention_pairs_per_batch": float(
                        totals["computed_attention_token_pairs"]
                    ) / batches,
                    "logical_samples_per_batch": float(totals["logical_samples"]) / batches,
                }
            )
        job_rows.append(
            {
                "arm_id": f"{row['family_id']}:{'P' if row['packing'] else 'U'}",
                "profile_group": str(row["family_id"]),
                "packing": bool(row["packing"]),
                "features": {
                    "log2_configured_cutoff": math.log2(float(row["cutoff_len"])),
                    "log2_observed_computed_tokens_per_batch": math.log2(
                        statistics.fmean(
                            rank["computed_tokens_per_batch"] for rank in rank_features
                        )
                    ),
                    "log1p_observed_attention_pairs_per_batch": math.log1p(
                        statistics.fmean(
                            rank["attention_pairs_per_batch"] for rank in rank_features
                        )
                    ),
                    "log1p_observed_logical_samples_per_batch": math.log1p(
                        statistics.fmean(
                            rank["logical_samples_per_batch"] for rank in rank_features
                        )
                    ),
                    "packing": float(bool(row["packing"])),
                },
                "log_reserved_gib": math.log(_metric(row, "reserved")),
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in job_rows:
        grouped[row["arm_id"]].append(row)
    arms = []
    for arm_id, repeats in sorted(grouped.items()):
        representative = repeats[0]
        arms.append(
            {
                "arm_id": arm_id,
                "setting_id": arm_id,
                "profile_group": representative["profile_group"],
                "packing": representative["packing"],
                "features": {
                    name: statistics.fmean(float(row["features"][name]) for row in repeats)
                    for name in representative["features"]
                },
                "log_reserved_gib": statistics.fmean(
                    float(row["log_reserved_gib"]) for row in repeats
                ),
            }
        )
    candidates = {
        "configured_cutoff_only": ("log2_configured_cutoff",),
        "observed_workload": (
            "log2_observed_computed_tokens_per_batch",
            "log1p_observed_attention_pairs_per_batch",
        ),
        "observed_workload_plus_packing": (
            "log2_observed_computed_tokens_per_batch",
            "log1p_observed_attention_pairs_per_batch",
            "log1p_observed_logical_samples_per_batch",
            "packing",
        ),
    }
    models = {
        name: _ridge_matrix_cv(arms, features, target="log_reserved_gib")
        for name, features in candidates.items()
    }
    workload = models["observed_workload"]["metrics"]
    packing = models["observed_workload_plus_packing"]["metrics"]
    improvement = 1.0 - float(packing["group_equal_mae_log_ratio"]) / float(
        workload["group_equal_mae_log_ratio"]
    )
    return {
        "scope": "phase_b_observed_workload_diagnostic_not_recommendation_time_model",
        "arm_rows": len(arms),
        "profile_groups": len({row["profile_group"] for row in arms}),
        "models": models,
        "packing_incremental_group_equal_mae_improvement": improvement,
        "packing_specific_center_residual_supported_diagnostically": bool(
            improvement >= 0.10
            and float(packing["worst_profile_group_mae_log_ratio"])
            <= 1.05 * float(workload["worst_profile_group_mae_log_ratio"])
        ),
        "limitations": [
            "Observed token/attention features are post-run diagnostics, not recommendation-time inputs.",
            "Only six independent Phase B profile groups are available.",
            "No high-capacity boundary or OOM observation is present.",
        ],
    }


def _memory_diagnostic() -> dict[str, Any]:
    result = read_json(PHASE_B)
    prediction_rows = {
        str(row["request_id"]): row for row in read_json(PREFLIGHT)["predictions"]
    }
    observations: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in result["job_results"]:
        observations[(str(row["family_id"]), bool(row["packing"]))].append(row)
    rows = []
    for (family, packing), repeats in sorted(observations.items()):
        request_ids = {str(row["memory_preflight_request_id"]) for row in repeats}
        if len(request_ids) != 1:
            raise ValueError(f"memory request varies across repeats: {family}/{packing}")
        prediction = prediction_rows[next(iter(request_ids))]["memory"]
        observed = statistics.fmean(_metric(row, "reserved") for row in repeats)
        center = float(prediction["reserved_center_bytes"]) / GIB
        upper = float(prediction["operational_p95_reserved_bytes"]) / GIB
        rows.append(
            {
                "family_id": family,
                "packing": packing,
                "observed_reserved_gib_mean": observed,
                "frozen_center_gib": center,
                "frozen_operational_p95_gib": upper,
                "center_error_gib": observed - center,
                "center_log_residual": math.log(observed / center),
                "upper_covered": observed <= upper,
            }
        )
    by_treatment = {}
    for packing in (False, True):
        subset = [row for row in rows if row["packing"] is packing]
        by_treatment["packed" if packing else "unpacked"] = {
            "arms": len(subset),
            "center_mae_gib": statistics.fmean(abs(row["center_error_gib"]) for row in subset),
            "center_mean_signed_error_gib": statistics.fmean(row["center_error_gib"] for row in subset),
            "center_max_underprediction_gib": max(row["center_error_gib"] for row in subset),
            "operational_p95_coverage": statistics.fmean(float(row["upper_covered"]) for row in subset),
        }
    packed_residual = [row["center_log_residual"] for row in rows if row["packing"]]
    unpacked_residual = [row["center_log_residual"] for row in rows if not row["packing"]]
    return {
        "scope": "phase_b_frozen_preflight_replay_only",
        "arms": len(rows),
        "rows": rows,
        "by_treatment": by_treatment,
        "packing_minus_unpacked_mean_center_log_residual": (
            statistics.fmean(packed_residual) - statistics.fmean(unpacked_residual)
        ),
        "feature_comparison": _phase_b_memory_feature_comparison(),
        "center_refit_completed": False,
        "upper_guard_accepted": False,
        "interpretation": (
            "This replay tests the previously frozen shared model on Phase B. "
            "It is not a new Packing-aware memory fit and contains no boundary/OOM evidence."
        ),
    }


def fit() -> dict[str, Any]:
    setting_rows, pair_rows = build_setting_rows()
    primary = [row for row in setting_rows if row["role"] == "primary_model_selection"]
    diagnostic = [row for row in setting_rows if row["role"] != "primary_model_selection"]
    if len(setting_rows) != 19 or len(primary) != 17 or len(diagnostic) != 2:
        raise ValueError(
            f"unexpected evidence population: all={len(setting_rows)}, primary={len(primary)}, diagnostic={len(diagnostic)}"
        )
    models: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    for label, target in (
        ("effective_tokens", "log_effective_ratio"),
        ("logical_samples", "log_logical_ratio"),
    ):
        ridge = _cross_validate("ridge_linear", primary, target)
        gam = _cross_validate("gam_spline", primary, target)
        models[label] = {"ridge_linear": ridge, "gam_spline": gam}
        selections[label] = _select_challenger(ridge, gam)

    report: dict[str, Any] = {
        "schema": "sft_h800_packing_phase_b_model_selection/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fit_only_not_publishable",
        "inputs": {
            path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in (
                PHASE_B, REAL_BUSINESS, INTERACTIONS, GBS_REPAIR, MEMBERSHIP,
                REAL_STATIC, INTERACTION_STATIC, PREFLIGHT,
            )
        },
        "estimand": {
            "primary": "log(Packed effective tokens/s / Unpacked effective tokens/s)",
            "secondary": "log(Packed logical samples/s / Unpacked logical samples/s)",
            "matched_mbs1_mechanism_effect_only": True,
            "best_branch_route_effect_claim_allowed": False,
        },
        "evidence": {
            "matched_repeat_pairs": len(pair_rows),
            "setting_rows": len(setting_rows),
            "primary_setting_rows": len(primary),
            "primary_profile_groups": len({row["profile_group"] for row in primary}),
            "interaction_diagnostic_setting_rows": len(diagnostic),
            "feature_names": list(FEATURE_NAMES),
            "primary_rows": primary,
            "interaction_diagnostic_rows": diagnostic,
            "exclusion_policy": (
                "GC-off and ZeRO-3 each have only one independent upstream profile; "
                "they remain diagnostics until cross-profile Phase C evidence exists."
            ),
        },
        "validation": {
            "split": "nested_leave_one_upstream_profile_out",
            "repeat_rows_collapsed_before_fit": True,
            "all_cutoffs_and_repeats_for_profile_stay_in_one_fold": True,
            "alpha_grid": list(ALPHAS),
            "selection_uses_group_equal_error": True,
        },
        "models": models,
        "selection": selections,
        "memory_diagnostic": _memory_diagnostic(),
        "gates": {
            "phase_b_evaluator_passed": True,
            "throughput_fit_only_challenger_comparison_complete": True,
            "production_throughput_model_selected": False,
            "packing_memory_center_fit_complete": False,
            "packing_memory_boundary_guard_complete": False,
            "automatic_packing_recommendation_allowed": False,
            "automatic_publication_allowed": False,
            "automatic_next_gpu_batch_allowed": False,
        },
        "next_step": (
            "Review selected fit-only throughput form, then implement the Packing-aware "
            "shared-memory center refit. Phase C is required before fitting general GC/ZeRO-3 "
            "interactions; Phase D is required before accepting the memory upper guard."
        ),
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)

    effective = report["models"]["effective_tokens"]
    logical = report["models"]["logical_samples"]
    memory = report["memory_diagnostic"]["by_treatment"]
    memory_features = report["memory_diagnostic"]["feature_comparison"]
    lines = [
        "# H800 Packing Phase B：模型选型与显存残差诊断",
        "",
        "本报告为 fit-only；不发布系数、不自动开启 Packing，也不自动启动新 GPU 批次。",
        "",
        f"证据：{len(pair_rows)} 个 matched repeat pairs，折叠为 {len(setting_rows)} 个配置；"
        f"主选型使用 {len(primary)} 个配置、{report['evidence']['primary_profile_groups']} 个独立画像组。",
        "",
        "| target | Ridge group-equal log-MAE | GAM group-equal log-MAE | selected |",
        "|---|---:|---:|---|",
        f"| effective tokens/s ratio | {effective['ridge_linear']['metrics']['group_equal_mae_log_ratio']:.4f} | "
        f"{effective['gam_spline']['metrics']['group_equal_mae_log_ratio']:.4f} | {selections['effective_tokens']['selected']} |",
        f"| logical samples/s ratio | {logical['ridge_linear']['metrics']['group_equal_mae_log_ratio']:.4f} | "
        f"{logical['gam_spline']['metrics']['group_equal_mae_log_ratio']:.4f} | {selections['logical_samples']['selected']} |",
        "",
        "表中的 selected 只表示 paired-effect center 的 fit-only 选择；生产 absolute＋pairwise 吞吐模型尚未选定。",
        "",
        "GC-off 与 ZeRO-3 各只有一个独立画像，未进入通用系数选型；它们保留为交互诊断。",
        "",
        "| memory replay | center MAE GiB | mean signed error GiB | P95 coverage |",
        "|---|---:|---:|---:|",
        f"| Unpacked | {memory['unpacked']['center_mae_gib']:.3f} | "
        f"{memory['unpacked']['center_mean_signed_error_gib']:.3f} | {memory['unpacked']['operational_p95_coverage']:.1%} |",
        f"| Packed | {memory['packed']['center_mae_gib']:.3f} | "
        f"{memory['packed']['center_mean_signed_error_gib']:.3f} | {memory['packed']['operational_p95_coverage']:.1%} |",
        "",
        "| Phase B memory feature diagnostic | group-equal log-MAE |",
        "|---|---:|",
        f"| configured cutoff only | {memory_features['models']['configured_cutoff_only']['metrics']['group_equal_mae_log_ratio']:.4f} |",
        f"| observed workload | {memory_features['models']['observed_workload']['metrics']['group_equal_mae_log_ratio']:.4f} |",
        f"| observed workload + Packing residual | {memory_features['models']['observed_workload_plus_packing']['metrics']['group_equal_mae_log_ratio']:.4f} |",
        "",
        "显存部分只是冻结 preflight replay，尚未完成 Packing-aware center refit；当前数据也没有容量边界/OOM证据，不能验收 upper guard。",
        "",
    ]
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = fit()
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "markdown": str(MARKDOWN),
                "evidence": {
                    key: value
                    for key, value in result["evidence"].items()
                    if key not in {"primary_rows", "interaction_diagnostic_rows"}
                },
                "selection": result["selection"],
                "memory": result["memory_diagnostic"]["by_treatment"],
                "gates": result["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
