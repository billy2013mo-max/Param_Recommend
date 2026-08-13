#!/usr/bin/env python3
"""Refit fit-only H800 Packing effect and memory-center models after Phase C.

This script intentionally does not publish a production predictor.  It adds the
completed Phase-C GC/ZeRO interaction evidence to the matched Packing effect
head and fits a recommendation-time memory-center residual on top of the frozen
physical-share center.  Memory upper bounds, cross-card scaling, and the final
absolute+pairwise throughput ranker remain separate acceptance problems.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import fit_h800_packing_phase_b_challengers_v1 as phase_b_fit
from common import ARTIFACT_DIR, read_json, sha256_file, sha256_json, write_json


PHASE_B = ARTIFACT_DIR / "h800_packing_profile_phase_b_results_v1.json"
PHASE_C = ARTIFACT_DIR / "h800_packing_profile_phase_c_results_v1.json"
PHASE_B_PREFLIGHT = ARTIFACT_DIR / "h800_packing_profile_phase_b_memory_predictions_v1.json"
PHASE_C_PREFLIGHT = ARTIFACT_DIR / "h800_packing_profile_phase_c_memory_predictions_v1.json"
PHASE_B_MODEL = ARTIFACT_DIR / "h800_packing_phase_b_model_selection_v1.json"

OUTPUT = ARTIFACT_DIR / "h800_packing_phase_c_model_refit_v1.json"
MARKDOWN = ARTIFACT_DIR / "h800_packing_phase_c_model_refit_v1.md"

GIB = float(2**30)
ALPHAS = phase_b_fit.ALPHAS

THROUGHPUT_FEATURE_NAMES = (
    "log2_cutoff",
    "log2_mean_length",
    "length_cv",
    "p99_length_to_cutoff",
    "pack_fill_ratio",
    "log2_n_pack_mean",
    "log2_packed_ga",
    "zero2",
    "zero3",
    "gc_off",
    "log2_gpu_count",
)

MEMORY_SHARED_FEATURES = (
    "log2_physical_center_gib",
    "log2_cutoff",
    "log2_mean_length",
    "length_cv",
    "p99_length_to_cutoff",
    "log2_effective_samples_per_physical_row",
    "zero3",
    "gc_off",
)

MEMORY_PACKING_FEATURES = MEMORY_SHARED_FEATURES + (
    "packing",
    "packing_x_log2_n_pack_mean",
    "packing_x_zero3",
    "packing_x_gc_off",
)


def _require_phase_c() -> dict[str, Any]:
    report = read_json(PHASE_C)
    gates = report.get("gates", {})
    required = (
        "packing_semantics_and_ledger_passed",
        "all_treatment_cv_le_0p05",
        "interaction_contrasts_available",
        "phase_c_complete_without_extra_repeats",
    )
    if any(gates.get(name) is not True for name in required):
        raise ValueError("Phase C has not passed every frozen fit-only gate")
    completion = report.get("completion", {})
    if completion.get("success") != 24 or completion.get("oom") != 0:
        raise ValueError("Phase C is not the exact 24-success/0-OOM evidence set")
    return report


def _add_mechanism_features(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    updated["features"] = dict(row["features"])
    updated["features"]["zero3"] = float(int(row.get("zero_stage", 0)) == 3)
    updated["features"]["gc_off"] = float(not bool(row.get("gc")))
    updated["role"] = "primary_model_selection"
    updated["exclusion_reasons"] = []
    return updated


def _phase_c_throughput_rows(report: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    results = report["job_results"]
    setting_rows: list[dict[str, Any]] = []
    repeat_pairs = 0
    for setting in report["settings"]:
        setting_id = str(setting["setting_id"])
        subset = [row for row in results if str(row["setting_id"]) == setting_id]
        packed = [row for row in subset if bool(row["packing"])]
        unpacked = [row for row in subset if not bool(row["packing"])]
        if len(packed) != 3 or len(unpacked) != 3:
            raise ValueError(f"Phase-C setting is not U/P x3: {setting_id}")
        representative = packed[0]
        workload = str(representative["workload_id"]).upper()
        cutoff = float(representative["cutoff_len"])
        n_pack = float(representative["n_pack_mean"])
        profile = phase_b_fit._profile_stats(str(representative["dataset_profile_path"]))
        effective_logs = [
            math.log(float(pair["packed_over_unpacked_effective_tokens_per_second"]))
            for pair in setting["pairs"]
        ]
        logical_logs = [
            math.log(float(pair["packed_over_unpacked_logical_samples_per_second"]))
            for pair in setting["pairs"]
        ]
        features = {
            "log2_cutoff": math.log2(cutoff),
            "log2_mean_length": math.log2(profile["mean"]),
            "length_cv": profile["cv"],
            "p99_length_to_cutoff": profile["p99"] / cutoff,
            "pack_fill_ratio": min(1.25, profile["mean"] * n_pack / cutoff),
            "log2_n_pack_mean": math.log2(n_pack),
            "log2_packed_ga": math.log2(float(representative["gradient_accumulation_steps"])),
            "zero2": float(int(representative.get("zero_stage", 0)) == 2),
            "zero3": float(int(representative.get("zero_stage", 0)) == 3),
            "gc_off": float(not bool(representative.get("gc"))),
            "log2_gpu_count": math.log2(float(representative["gpu_count"])),
        }
        setting_rows.append(
            {
                "setting_id": f"phase_c:{setting_id}",
                "profile_group": f"phase_b:{workload}",
                "source": "phase_c",
                "role": "primary_model_selection",
                "exclusion_reasons": [],
                "cutoff_len": int(cutoff),
                "gpu_count": int(representative["gpu_count"]),
                "zero_stage": int(representative.get("zero_stage", 0)),
                "gc": bool(representative.get("gc")),
                "n_pack_mean": n_pack,
                "packed_ga": int(representative["gradient_accumulation_steps"]),
                "features": features,
                "repeat_count": 3,
                "log_effective_ratio": statistics.fmean(effective_logs),
                "log_effective_ratio_repeat_std": statistics.pstdev(effective_logs),
                "effective_ratio": math.exp(statistics.fmean(effective_logs)),
                "log_logical_ratio": statistics.fmean(logical_logs),
                "log_logical_ratio_repeat_std": statistics.pstdev(logical_logs),
                "logical_ratio": math.exp(statistics.fmean(logical_logs)),
                "packed_reserved_gib_mean": float(setting["reserved_memory"]["packed_gib_mean"]),
                "unpacked_reserved_gib_mean": float(setting["reserved_memory"]["unpacked_gib_mean"]),
            }
        )
        repeat_pairs += 3
    return setting_rows, repeat_pairs


def _throughput_refit(phase_c: dict[str, Any]) -> dict[str, Any]:
    baseline_rows, baseline_pairs = phase_b_fit.build_setting_rows()
    old_rows = [_add_mechanism_features(row) for row in baseline_rows]
    phase_c_rows, phase_c_pairs = _phase_c_throughput_rows(phase_c)
    rows = sorted([*old_rows, *phase_c_rows], key=lambda row: str(row["setting_id"]))
    if len(rows) != 23 or len(baseline_pairs) + phase_c_pairs != 61:
        raise ValueError(
            f"unexpected Phase-C throughput population: settings={len(rows)}, pairs={len(baseline_pairs) + phase_c_pairs}"
        )

    # Reuse the nested profile-group CV implementation with the expanded,
    # recommendation-time feature contract.
    phase_b_fit.FEATURE_NAMES = THROUGHPUT_FEATURE_NAMES
    models: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    for label, target in (
        ("effective_tokens", "log_effective_ratio"),
        ("logical_samples", "log_logical_ratio"),
    ):
        ridge = phase_b_fit._cross_validate("ridge_linear", rows, target)
        gam = phase_b_fit._cross_validate("gam_spline", rows, target)
        models[label] = {"ridge_linear": ridge, "gam_spline": gam}
        selections[label] = phase_b_fit._select_challenger(ridge, gam)

    return {
        "status": "paired_effect_center_refit_complete_fit_only",
        "estimand": "log(Packed throughput / Unpacked throughput) at matched MBS=1",
        "matched_repeat_pairs": len(baseline_pairs) + phase_c_pairs,
        "setting_rows": len(rows),
        "profile_groups": len({str(row["profile_group"]) for row in rows}),
        "feature_names": list(THROUGHPUT_FEATURE_NAMES),
        "phase_c_setting_rows": phase_c_rows,
        "models": models,
        "selection": selections,
        "production_absolute_pairwise_ranker_selected": False,
        "best_branch_route_effect_claim_allowed": False,
    }


def _prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["request_id"]): row
        for row in read_json(path)["predictions"]
    }


def _memory_rows() -> list[dict[str, Any]]:
    sources = (
        ("phase_b", read_json(PHASE_B), _prediction_map(PHASE_B_PREFLIGHT)),
        ("phase_c", read_json(PHASE_C), _prediction_map(PHASE_C_PREFLIGHT)),
    )
    arms: list[dict[str, Any]] = []
    for source_name, report, predictions in sources:
        grouped: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
        for row in report["job_results"]:
            setting = str(row["family_id"] if source_name == "phase_b" else row["setting_id"])
            grouped[(setting, bool(row["packing"]))].append(row)
        for (setting, packing), repeats in sorted(grouped.items()):
            if len(repeats) != 3:
                raise ValueError(f"memory arm is not repeated three times: {source_name}/{setting}/{packing}")
            representative = repeats[0]
            request_ids = {str(row["memory_preflight_request_id"]) for row in repeats}
            if len(request_ids) != 1:
                raise ValueError(f"memory request varies across repeats: {source_name}/{setting}/{packing}")
            prediction = predictions[next(iter(request_ids))]["memory"]
            center_gib = float(prediction["reserved_center_bytes"]) / GIB
            observed_gib = statistics.fmean(float(row["max_reserved_gib"]) for row in repeats)
            cutoff = float(representative["cutoff_len"])
            profile = phase_b_fit._profile_stats(str(representative["dataset_profile_path"]))
            n_pack = float(representative.get("n_pack_mean") or 1.0)
            effective_samples = n_pack if packing else 1.0
            zero3 = float(int(representative.get("zero_stage", 0)) == 3)
            gc_off = float(not bool(representative.get("gc")))
            packing_value = float(packing)
            features = {
                "log2_physical_center_gib": math.log2(center_gib),
                "log2_cutoff": math.log2(cutoff),
                "log2_mean_length": math.log2(profile["mean"]),
                "length_cv": profile["cv"],
                "p99_length_to_cutoff": profile["p99"] / cutoff,
                "log2_effective_samples_per_physical_row": math.log2(effective_samples),
                "zero3": zero3,
                "gc_off": gc_off,
                "packing": packing_value,
                "packing_x_log2_n_pack_mean": packing_value * math.log2(n_pack),
                "packing_x_zero3": packing_value * zero3,
                "packing_x_gc_off": packing_value * gc_off,
            }
            workload = str(representative.get("workload_id") or setting).upper()
            arms.append(
                {
                    "arm_id": f"{source_name}:{setting}:{'P' if packing else 'U'}",
                    "setting_id": f"{source_name}:{setting}",
                    "profile_group": workload,
                    "source": source_name,
                    "packing": packing,
                    "features": features,
                    "physical_center_gib": center_gib,
                    "observed_reserved_gib": observed_gib,
                    "target_log_residual": math.log(observed_gib / center_gib),
                }
            )
    if len(arms) != 20:
        raise ValueError(f"unexpected memory-center arm count: {len(arms)}")
    return arms


def _matrix(rows: list[dict[str, Any]], names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[float(row["features"][name]) for name in names] for row in rows],
        dtype=float,
    )


def _group_equal_mae(rows: list[dict[str, Any]], actual: np.ndarray, predicted: np.ndarray) -> float:
    errors: dict[str, list[float]] = defaultdict(list)
    for row, y, yhat in zip(rows, actual, predicted, strict=True):
        errors[str(row["profile_group"])].append(abs(float(y) - float(yhat)))
    return statistics.fmean(statistics.fmean(values) for values in errors.values())


def _fit_ridge(
    alpha: float,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, StandardScaler, Ridge]:
    scaler = StandardScaler().fit(train_x)
    model = Ridge(alpha=alpha).fit(scaler.transform(train_x), train_y)
    return model.predict(scaler.transform(test_x)), scaler, model


def _select_alpha(rows: list[dict[str, Any]], x: np.ndarray, y: np.ndarray) -> float:
    groups = sorted({str(row["profile_group"]) for row in rows})
    scores = []
    for alpha in ALPHAS:
        prediction = np.empty(len(rows), dtype=float)
        for held_out in groups:
            train = np.asarray([i for i, row in enumerate(rows) if row["profile_group"] != held_out])
            test = np.asarray([i for i, row in enumerate(rows) if row["profile_group"] == held_out])
            prediction[test], _, _ = _fit_ridge(alpha, x[train], y[train], x[test])
        scores.append((_group_equal_mae(rows, y, prediction), -alpha, alpha))
    return float(min(scores)[2])


def _memory_candidate(rows: list[dict[str, Any]], feature_names: tuple[str, ...]) -> dict[str, Any]:
    x = _matrix(rows, feature_names)
    y = np.asarray([float(row["target_log_residual"]) for row in rows], dtype=float)
    groups = sorted({str(row["profile_group"]) for row in rows})
    prediction = np.empty(len(rows), dtype=float)
    folds = []
    for held_out in groups:
        train_indices = [i for i, row in enumerate(rows) if row["profile_group"] != held_out]
        test_indices = [i for i, row in enumerate(rows) if row["profile_group"] == held_out]
        train_rows = [rows[i] for i in train_indices]
        train_x = x[train_indices]
        train_y = y[train_indices]
        alpha = _select_alpha(train_rows, train_x, train_y)
        prediction[test_indices], _, _ = _fit_ridge(alpha, train_x, train_y, x[test_indices])
        folds.append({"held_out_profile_group": held_out, "selected_alpha": alpha})

    final_alpha = _select_alpha(rows, x, y)
    _, scaler, model = _fit_ridge(final_alpha, x, y, x)
    predicted_gib = np.asarray(
        [float(row["physical_center_gib"]) for row in rows], dtype=float
    ) * np.exp(prediction)
    observed_gib = np.asarray(
        [float(row["observed_reserved_gib"]) for row in rows], dtype=float
    )
    signed = predicted_gib - observed_gib
    group_log_errors: dict[str, list[float]] = defaultdict(list)
    for row, actual, predicted in zip(rows, y, prediction, strict=True):
        group_log_errors[str(row["profile_group"])].append(
            abs(float(actual) - float(predicted))
        )
    per_group_log_mae = {
        group: statistics.fmean(values)
        for group, values in sorted(group_log_errors.items())
    }
    metrics = {
        "arms": len(rows),
        "profile_groups": len(groups),
        "group_equal_mae_log_residual": _group_equal_mae(rows, y, prediction),
        "row_mae_log_residual": float(np.mean(np.abs(y - prediction))),
        "reserved_mae_gib": float(np.mean(np.abs(signed))),
        "reserved_mean_signed_error_gib": float(np.mean(signed)),
        "maximum_underprediction_gib": float(np.max(observed_gib - predicted_gib)),
        "worst_profile_group_mae_log_residual": max(per_group_log_mae.values()),
        "per_profile_group_mae_log_residual": per_group_log_mae,
        "packing_reserved_mae_gib": float(
            np.mean([abs(signed[i]) for i, row in enumerate(rows) if row["packing"]])
        ),
        "unpacked_reserved_mae_gib": float(
            np.mean([abs(signed[i]) for i, row in enumerate(rows) if not row["packing"]])
        ),
    }
    return {
        "model_family": "physical_center_log_residual_standardized_ridge",
        "feature_names": list(feature_names),
        "target": "log(observed_reserved_gib / frozen_physical_center_gib)",
        "nested_leave_one_workload_profile_out": True,
        "metrics": metrics,
        "folds": folds,
        "oof_predictions": [
            {
                "arm_id": row["arm_id"],
                "profile_group": row["profile_group"],
                "packing": row["packing"],
                "observed_reserved_gib": row["observed_reserved_gib"],
                "physical_center_gib": row["physical_center_gib"],
                "predicted_log_residual": float(predicted),
                "predicted_reserved_gib": float(pred_gib),
            }
            for row, predicted, pred_gib in zip(rows, prediction, predicted_gib, strict=True)
        ],
        "final_fit_only_model": {
            "alpha": final_alpha,
            "standardizer_mean": scaler.mean_.tolist(),
            "standardizer_scale": scaler.scale_.tolist(),
            "intercept": float(model.intercept_),
            "coefficients": model.coef_.tolist(),
        },
    }


def _memory_refit() -> dict[str, Any]:
    rows = _memory_rows()
    shared = _memory_candidate(rows, MEMORY_SHARED_FEATURES)
    packing = _memory_candidate(rows, MEMORY_PACKING_FEATURES)
    shared_mae = float(shared["metrics"]["group_equal_mae_log_residual"])
    packing_mae = float(packing["metrics"]["group_equal_mae_log_residual"])
    improvement = 1.0 - packing_mae / shared_mae
    worst_group_ratio = float(
        packing["metrics"]["worst_profile_group_mae_log_residual"]
    ) / float(shared["metrics"]["worst_profile_group_mae_log_residual"])
    packing_passes = improvement >= 0.05 and worst_group_ratio <= 1.05
    selected = "shared_workload_plus_packing" if packing_passes else "shared_workload"
    return {
        "status": "memory_center_refit_complete_fit_only",
        "physical_baseline": "frozen physical-share reserved_center_bytes",
        "arm_rows": len(rows),
        "profile_groups": len({str(row["profile_group"]) for row in rows}),
        "candidates": {
            "shared_workload": shared,
            "shared_workload_plus_packing": packing,
        },
        "selection": {
            "selected": selected,
            "minimum_group_equal_log_mae_improvement": 0.05,
            "maximum_worst_profile_group_mae_ratio": 1.05,
            "observed_relative_improvement": improvement,
            "observed_worst_profile_group_mae_ratio": worst_group_ratio,
            "packing_candidate_passed_rule": packing_passes,
            "selection_scope": "center_fit_only",
        },
        "upper_guard_accepted": False,
        "boundary_or_oom_evidence_present": False,
    }


def fit() -> dict[str, Any]:
    phase_c = _require_phase_c()
    throughput = _throughput_refit(phase_c)
    memory = _memory_refit()
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_phase_c_model_refit/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fit_only_not_publishable",
        "inputs": {
            path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in (
                PHASE_B,
                PHASE_C,
                PHASE_B_PREFLIGHT,
                PHASE_C_PREFLIGHT,
                PHASE_B_MODEL,
            )
        },
        "throughput_effect_center": throughput,
        "memory_center": memory,
        "gates": {
            "phase_c_evaluator_passed": True,
            "paired_effect_center_refit_complete": True,
            "packing_memory_center_refit_complete": True,
            "production_absolute_pairwise_throughput_selected": False,
            "packing_memory_upper_guard_complete": False,
            "cross_card_scaling_complete": False,
            "prospective_holdout_complete": False,
            "automatic_packing_recommendation_allowed": False,
            "automatic_publication_allowed": False,
        },
        "next_step": (
            "Use the refit diagnostics to select high-information memory-boundary and "
            "cross-card experiments; then freeze the full absolute+pairwise model before "
            "a source-disjoint prospective holdout."
        ),
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)

    effective = throughput["models"]["effective_tokens"]
    logical = throughput["models"]["logical_samples"]
    mem_shared = memory["candidates"]["shared_workload"]["metrics"]
    mem_packing = memory["candidates"]["shared_workload_plus_packing"]["metrics"]
    lines = [
        "# H800 Packing Phase C 后模型重拟合",
        "",
        "本产物为 fit-only：已重拟合 paired-effect 吞吐中心和 physical-share 显存中心残差，但不发布生产推荐。",
        "",
        f"吞吐证据：{throughput['matched_repeat_pairs']} 个 matched repeat pairs，折叠为 "
        f"{throughput['setting_rows']} 个 setting、{throughput['profile_groups']} 个画像组。",
        "",
        "| target | Ridge group-equal log-MAE | GAM group-equal log-MAE | selected |",
        "|---|---:|---:|---|",
        f"| effective tokens/s ratio | {effective['ridge_linear']['metrics']['group_equal_mae_log_ratio']:.4f} | "
        f"{effective['gam_spline']['metrics']['group_equal_mae_log_ratio']:.4f} | "
        f"{throughput['selection']['effective_tokens']['selected']} |",
        f"| logical samples/s ratio | {logical['ridge_linear']['metrics']['group_equal_mae_log_ratio']:.4f} | "
        f"{logical['gam_spline']['metrics']['group_equal_mae_log_ratio']:.4f} | "
        f"{throughput['selection']['logical_samples']['selected']} |",
        "",
        "| memory center candidate | group-equal log-MAE | reserved MAE GiB | Packing MAE GiB |",
        "|---|---:|---:|---:|",
        f"| shared workload | {mem_shared['group_equal_mae_log_residual']:.4f} | "
        f"{mem_shared['reserved_mae_gib']:.3f} | {mem_shared['packing_reserved_mae_gib']:.3f} |",
        f"| shared workload + Packing residual | {mem_packing['group_equal_mae_log_residual']:.4f} | "
        f"{mem_packing['reserved_mae_gib']:.3f} | {mem_packing['packing_reserved_mae_gib']:.3f} |",
        "",
        f"显存中心选择：`{memory['selection']['selected']}`；相对 group-equal log-MAE 改善 "
        f"{memory['selection']['observed_relative_improvement']:.1%}。",
        "",
        "当前没有容量边界/OOM证据，显存 upper guard 仍未验收；生产 absolute＋pairwise 吞吐排序器也未冻结。",
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
                "throughput_evidence": {
                    key: result["throughput_effect_center"][key]
                    for key in ("matched_repeat_pairs", "setting_rows", "profile_groups")
                },
                "throughput_selection": result["throughput_effect_center"]["selection"],
                "memory_selection": result["memory_center"]["selection"],
                "gates": result["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
