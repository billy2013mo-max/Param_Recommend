#!/usr/bin/env python3
"""Training-only selection of route-specific H800 expansion heads.

The pooled profile-aware model from offline validation v1 is treated as a
shared prior.  For each mechanism route, a route-only Ridge model is fitted and
geometrically blended with the shared prediction:

    log(rho_hat) = (1 - lambda) * log(rho_shared)
                   + lambda * log(rho_route)

Feature set, route regularization, and lambda are selected solely from
leave-one-source-out predictions on the V5 fit sources.  The five strict unused
dataset sources are scored once after every route has been frozen.  OOM rows
remain right-censored and never become exact center labels.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any

from common import RESULTS_DIR, ROOT, read_json, sha256_file, write_json, write_jsonl
from fit_h800_lora_source_disjoint_recalibration_v1 import (
    DEFAULT_FROZEN_BASELINE,
    DEFAULT_HARDWARE,
    _is_critical_lora,
    _label,
    _outcome,
    _predict_center,
    _source_id,
)
from migrate_refit_h800_historical_memory_v1 import (
    DEFAULT_CANONICAL,
    DEFAULT_CURRENT_OLD,
    DEFAULT_DATASET_ANALYSIS,
    DEFAULT_NEW_QUEUE,
    DEFAULT_THEORY_BASIS,
)
from refit_h800_m1_all_unused_validation_v3 import (
    DEFAULT_STAGE2_QUEUE,
    DEFAULT_VALIDATION_INVENTORY,
)
from validate_h800_profile_expansion_offline_v1 import (
    ALPHAS,
    DEFAULT_CURRENT_CANDIDATE,
    FEATURE_SETS,
    _build_data,
    _fit_expansion_ridge,
    _finite,
    _predict_expansion,
    _prediction_metrics,
    _ratio_metrics,
    _record_summary,
    _successful,
)


SCHEMA = "sft_h800_route_specific_expansion_offline_validation/v2"
DEFAULT_V1_REPORT = (
    ROOT
    / "diagnostics"
    / "h800_profile_expansion_offline_v1_20260809"
    / "report.json"
)
DEFAULT_V5_REPORT = (
    ROOT
    / "diagnostics"
    / "h800_m1_lora_full_admission_v5_20260805"
    / "refit_and_all_unused_validation_report.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "diagnostics" / "h800_route_specific_expansion_offline_v2_20260809"
)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ROUTES = ("critical_lora", "other_lora", "supported_full", "other_full")


def _route(record: Mapping[str, Any]) -> str:
    selector = record.get("selector") or {}
    scenario = record.get("scenario") or {}
    mode = str(selector.get("training_mode") or "")
    if _is_critical_lora(record):
        return "critical_lora"
    if bool(
        mode == "full"
        and int(selector.get("zero_stage") or 0) == 3
        and bool(selector.get("gradient_checkpointing"))
        and int(scenario.get("gpu_count") or 0) == 2
        and not bool(selector.get("packing"))
    ):
        return "supported_full"
    if mode == "lora":
        return "other_lora"
    if mode == "full":
        return "other_full"
    return "other"


def _route_feature_sets(route: str) -> tuple[str, ...]:
    if route == "critical_lora":
        return (
            "m1_expansion_baseline",
            "profile_main",
            "profile_critical_interactions",
            "profile_lora_and_critical_interactions",
        )
    if route == "other_lora":
        return (
            "m1_expansion_baseline",
            "profile_main",
            "profile_lora_interactions",
            "profile_lora_and_critical_interactions",
        )
    return ("m1_expansion_baseline", "profile_main")


def _observed_expansion(record: Mapping[str, Any]) -> float:
    return float(_label(record, "reserved")) / float(_label(record, "allocated"))


def _record_id(record: Mapping[str, Any]) -> str:
    return str(record["cluster_id"])


def _shared_oof(
    all_training: Sequence[Mapping[str, Any]],
    route_records: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str],
    alpha: float,
    weighting: str,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for held in sorted({_source_id(row) for row in route_records}):
        fit = [row for row in all_training if _source_id(row) != held]
        evaluation = [row for row in route_records if _source_id(row) == held]
        model = _fit_expansion_ridge(
            fit,
            names=names,
            alpha=alpha,
            weighting=weighting,
        )
        for row in evaluation:
            results[_record_id(row)] = {
                "source_id": held,
                "observed_expansion": _observed_expansion(row),
                "shared_expansion": _predict_expansion(row, model),
            }
    return results


def _route_oof(
    route_records: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str],
    alpha: float,
) -> dict[str, float]:
    results: dict[str, float] = {}
    for held in sorted({_source_id(row) for row in route_records}):
        fit = [row for row in route_records if _source_id(row) != held]
        evaluation = [row for row in route_records if _source_id(row) == held]
        model = _fit_expansion_ridge(
            fit,
            names=names,
            alpha=alpha,
            weighting="source_balanced",
        )
        for row in evaluation:
            results[_record_id(row)] = _predict_expansion(row, model)
    return results


def _source_distribution(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    by_source: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _finite(row.get(key))
        if value is not None:
            by_source[str(row["source_id"])].append(value)
    source_means = sorted(statistics.mean(values) for values in by_source.values())
    return {
        "independent_sources": len(source_means),
        "source_mean_min": min(source_means) if source_means else None,
        "source_mean_median": statistics.median(source_means) if source_means else None,
        "source_mean_max": max(source_means) if source_means else None,
        "sources_above_0p20": sum(value > 0.20 for value in source_means),
    }


def _select_route(
    all_training: Sequence[Mapping[str, Any]],
    *,
    route: str,
    shared_spec: Mapping[str, Any],
) -> dict[str, Any]:
    route_records = [
        row for row in _successful(all_training) if _route(row) == route
    ]
    source_count = len({_source_id(row) for row in route_records})
    if source_count < 2:
        raise ValueError(f"route {route} has fewer than two successful sources")
    shared = _shared_oof(
        _successful(all_training),
        route_records,
        names=shared_spec["feature_names"],
        alpha=float(shared_spec["alpha"]),
        weighting=str(shared_spec["weighting"]),
    )
    candidates: list[dict[str, Any]] = []
    for feature_set in _route_feature_sets(route):
        names = FEATURE_SETS[feature_set]
        for alpha in ALPHAS:
            route_predictions = _route_oof(
                route_records,
                names=names,
                alpha=alpha,
            )
            for blend in LAMBDAS:
                details: list[dict[str, Any]] = []
                for record in route_records:
                    row_id = _record_id(record)
                    shared_row = shared[row_id]
                    shared_ratio = float(shared_row["shared_expansion"])
                    route_ratio = float(route_predictions[row_id])
                    prediction = math.exp(
                        (1.0 - blend) * math.log(shared_ratio)
                        + blend * math.log(route_ratio)
                    )
                    observed = float(shared_row["observed_expansion"])
                    details.append(
                        {
                            "source_id": shared_row["source_id"],
                            "observed_expansion": observed,
                            "predicted_expansion": prediction,
                            "ape": abs(prediction / observed - 1.0),
                            "signed_error": prediction / observed - 1.0,
                        }
                    )
                metrics = _ratio_metrics(details, "predicted_expansion")
                metrics["source_error_distribution"] = _source_distribution(
                    details, "ape"
                )
                candidates.append(
                    {
                        "route": route,
                        "fit_success_rows": len(route_records),
                        "fit_independent_sources": source_count,
                        "route_feature_set": feature_set,
                        "route_feature_names": list(names),
                        "route_alpha": alpha,
                        "shared_weight": 1.0 - blend,
                        "route_weight": blend,
                        "training_loso_metrics": metrics,
                    }
                )
    ranked = sorted(
        candidates,
        key=lambda row: (
            float(row["training_loso_metrics"]["source_equal_mape"]),
            int(row["training_loso_metrics"]["source_error_distribution"]["sources_above_0p20"]),
            -float(row["shared_weight"]),
            len(row["route_feature_names"]),
            float(row["route_alpha"]),
        ),
    )
    return {
        "selection_protocol": (
            "route feature set, alpha, and shared/route blend selected only by "
            "route-local source-equal expansion MAPE under training LOSO"
        ),
        "best": ranked[0],
        "candidate_count": len(candidates),
        "top_candidates": ranked[:10],
    }


def _fit_route_bundle(
    training: Sequence[Mapping[str, Any]],
    *,
    shared_spec: Mapping[str, Any],
    route_selections: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    shared_model = _fit_expansion_ridge(
        training,
        names=shared_spec["feature_names"],
        alpha=float(shared_spec["alpha"]),
        weighting=str(shared_spec["weighting"]),
    )
    routes: dict[str, Any] = {}
    for route, selection in route_selections.items():
        best = selection["best"]
        route_records = [
            row for row in _successful(training) if _route(row) == route
        ]
        route_model = _fit_expansion_ridge(
            route_records,
            names=best["route_feature_names"],
            alpha=float(best["route_alpha"]),
            weighting="source_balanced",
        )
        routes[route] = {
            "shared_weight": best["shared_weight"],
            "route_weight": best["route_weight"],
            "route_model": route_model,
            "training_loso_metrics": best["training_loso_metrics"],
        }
    return {
        "model_family": "hierarchical_geometric_blend_of_shared_and_route_expansion",
        "formula": (
            "log(rho_hat)=(1-lambda)*log(rho_shared)+lambda*log(rho_route)"
        ),
        "shared_model": shared_model,
        "routes": routes,
    }


def _predict_route_expansion(
    record: Mapping[str, Any], bundle: Mapping[str, Any]
) -> tuple[float, dict[str, Any]]:
    route = _route(record)
    shared_ratio = _predict_expansion(record, bundle["shared_model"])
    route_entry = (bundle.get("routes") or {}).get(route)
    if not isinstance(route_entry, Mapping):
        return shared_ratio, {
            "route": route,
            "path": "shared_fallback",
            "shared_ratio": shared_ratio,
            "route_ratio": None,
            "route_weight": 0.0,
        }
    route_ratio = _predict_expansion(record, route_entry["route_model"])
    route_weight = float(route_entry["route_weight"])
    prediction = math.exp(
        (1.0 - route_weight) * math.log(shared_ratio)
        + route_weight * math.log(route_ratio)
    )
    return prediction, {
        "route": route,
        "path": "hierarchical_route",
        "shared_ratio": shared_ratio,
        "route_ratio": route_ratio,
        "route_weight": route_weight,
    }


def _strict_predictions(
    strict: Sequence[Mapping[str, Any]],
    *,
    current_bundle: Mapping[str, Any],
    pooled_model: Mapping[str, Any],
    route_bundle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in strict:
        summary = _record_summary(record, "strict_unused_dataset")
        base = {**summary, "route": _route(record)}
        if _outcome(record) != "success":
            rows.append(base)
            continue
        observed_allocated = float(_label(record, "allocated"))
        observed_reserved = float(_label(record, "reserved"))
        allocated_center = _predict_center(record, current_bundle["allocated_model"])
        current_reserved = _predict_center(record, current_bundle["reserved_model"])
        pooled_ratio = _predict_expansion(record, pooled_model)
        route_ratio, route_detail = _predict_route_expansion(record, route_bundle)
        rows.append(
            {
                **base,
                "analytic_reference_bytes": float(
                    (record.get("memory") or {})["analytic_reference_bytes"]
                ),
                "observed_allocated_bytes": observed_allocated,
                "observed_reserved_bytes": observed_reserved,
                "observed_expansion": observed_reserved / observed_allocated,
                "current_v5_allocated_center_bytes": allocated_center,
                "current_v5_reserved_center_bytes": current_reserved,
                "pooled_profile_expansion": pooled_ratio,
                "pooled_profile_reserved_center_bytes": allocated_center * pooled_ratio,
                "route_expansion": route_ratio,
                "route_expansion_detail": route_detail,
                "route_reserved_center_bytes": allocated_center * route_ratio,
                "route_oracle_allocated_reserved_bytes": (
                    observed_allocated * route_ratio
                ),
            }
        )
    return rows


def _scope_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    success = [row for row in rows if row.get("observed_reserved_bytes") is not None]
    scopes: dict[str, list[Mapping[str, Any]]] = {
        "all_success": success,
        **{route: [row for row in success if row.get("route") == route] for route in ROUTES},
    }
    model_keys = {
        "analytic_reference": "analytic_reference_bytes",
        "current_v5_m1_reserved": "current_v5_reserved_center_bytes",
        "pooled_profile_expansion": "pooled_profile_reserved_center_bytes",
        "route_specific_expansion": "route_reserved_center_bytes",
        "route_specific_oracle_allocated": "route_oracle_allocated_reserved_bytes",
    }
    result: dict[str, Any] = {}
    for scope, scoped in scopes.items():
        result[scope] = {
            name: _prediction_metrics(scoped, key) for name, key in model_keys.items()
        }
        result[scope]["route_expansion_ratio"] = _ratio_metrics(
            scoped, "route_expansion"
        )
        result[scope]["strict_source_error_distribution"] = _source_distribution(
            [
                {
                    "source_id": row["source_id"],
                    "ape": abs(
                        float(row["route_reserved_center_bytes"])
                        / float(row["observed_reserved_bytes"])
                        - 1.0
                    ),
                }
                for row in scoped
            ],
            "ape",
        )
    return result


def _experiment_counts(v5_report: Mapping[str, Any]) -> dict[str, Any]:
    fit = v5_report["fit_set"]
    metrics = v5_report["validation_metrics"]
    strict = metrics["strict_unused_datasets"]["all_mechanisms"]
    historical = metrics["historical_unfitted_configurations"]["all_mechanisms"]
    replay = metrics["overlap_replay_not_validation"]["all_mechanisms"]
    result_jobs = len(list(RESULTS_DIR.glob("*/latest_attempt.json")))
    return {
        "v5_fit_raw_observations": int(fit["raw_observations"]),
        "v5_fit_unique_configurations": int(fit["unique_configuration_results"]),
        "v5_fit_success_center_labels": int(fit["outcomes"]["success"]),
        "v5_fit_oom_right_censored": int(fit["outcomes"]["oom"]),
        "v5_fit_independent_sources": int(fit["independent_sources"]),
        "strict_unused_configurations": int(strict["configurations"]),
        "historical_unfitted_configurations": int(historical["configurations"]),
        "overlap_replay_configurations_not_independent_validation": int(
            replay["configurations"]
        ),
        "v5_report_unique_fit_and_scored_records": (
            int(fit["unique_configuration_results"])
            + int(strict["configurations"])
            + int(historical["configurations"])
            + int(replay["configurations"])
        ),
        "workspace_result_jobs_all_offline_purposes": result_jobs,
        "workspace_result_jobs_note": (
            "includes throughput, thermal, canary, repeats, superseded runs, and "
            "other jobs; it is not the center-model training count"
        ),
    }


def _pct(value: Any) -> str:
    parsed = _finite(value)
    return "—" if parsed is None else f"{100.0 * parsed:.2f}%"


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# H800 按机制 expansion 离线验证 v2",
        "",
        "> 训练内 LOSO 选型；严格未使用源只评估一次；未修改生产模型。",
        "",
        "## 机制训练覆盖",
        "",
        "| 路由 | 成功标签 | 独立源 | Route 权重 | 特征集 | alpha | 训练 LOSO MAPE |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for route in ROUTES:
        selection = report["route_selections"][route]["best"]
        lines.append(
            f"| {route} | {selection['fit_success_rows']} | "
            f"{selection['fit_independent_sources']} | {selection['route_weight']:.2f} | "
            f"{selection['route_feature_set']} | {selection['route_alpha']} | "
            f"{_pct(selection['training_loso_metrics']['source_equal_mape'])} |"
        )
    lines.extend(
        [
            "",
            "## 严格未使用集 reserved 中心",
            "",
            "| 范围 | 解析基准 | 当前 V5 | Pooled profile | Route-specific | Route oracle allocated |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scope, label in (
        ("all_success", "全部成功"),
        ("critical_lora", "Critical LoRA"),
        ("other_lora", "其他 LoRA"),
        ("supported_full", "Supported FULL"),
    ):
        row = report["strict_metrics"][scope]
        lines.append(
            f"| {label} | {_pct(row['analytic_reference']['source_equal_mape'])} | "
            f"{_pct(row['current_v5_m1_reserved']['source_equal_mape'])} | "
            f"{_pct(row['pooled_profile_expansion']['source_equal_mape'])} | "
            f"{_pct(row['route_specific_expansion']['source_equal_mape'])} | "
            f"{_pct(row['route_specific_oracle_allocated']['source_equal_mape'])} |"
        )
    counts = report["experiment_counts"]
    lines.extend(
        [
            "",
            "## 显存实验数量口径",
            "",
            "| 口径 | 数量 |",
            "|---|---:|",
            f"| V5 原始拟合 observations | {counts['v5_fit_raw_observations']} |",
            f"| 折叠后拟合配置 | {counts['v5_fit_unique_configurations']} |",
            f"| 成功中心标签 | {counts['v5_fit_success_center_labels']} |",
            f"| OOM 右删失样本 | {counts['v5_fit_oom_right_censored']} |",
            f"| 严格未使用配置 | {counts['strict_unused_configurations']} |",
            f"| 历史未拟合配置 | {counts['historical_unfitted_configurations']} |",
            f"| V5 报告拟合与评分记录合计 | {counts['v5_report_unique_fit_and_scored_records']} |",
            f"| workspace 全部离线 result jobs | {counts['workspace_result_jobs_all_offline_purposes']} |",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--theory-basis", type=Path, default=DEFAULT_THEORY_BASIS)
    parser.add_argument(
        "--dataset-analysis", type=Path, default=DEFAULT_DATASET_ANALYSIS
    )
    parser.add_argument("--new-queue", type=Path, default=DEFAULT_NEW_QUEUE)
    parser.add_argument("--current-old", type=Path, default=DEFAULT_CURRENT_OLD)
    parser.add_argument("--stage2-queue", type=Path, default=DEFAULT_STAGE2_QUEUE)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_VALIDATION_INVENTORY)
    parser.add_argument("--hardware", type=Path, default=DEFAULT_HARDWARE)
    parser.add_argument(
        "--frozen-baseline", type=Path, default=DEFAULT_FROZEN_BASELINE
    )
    parser.add_argument(
        "--current-candidate", type=Path, default=DEFAULT_CURRENT_CANDIDATE
    )
    parser.add_argument("--v1-report", type=Path, default=DEFAULT_V1_REPORT)
    parser.add_argument("--v5-report", type=Path, default=DEFAULT_V5_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    print("rebuilding the frozen V5 fit and strict validation split", flush=True)
    data = _build_data(args)
    training = list(data["training"])
    strict = list(data["strict"])
    v1_report = read_json(args.v1_report)
    shared_spec = v1_report["training_loso_selection"]["best_profile"]

    print("selecting route heads from training LOSO only", flush=True)
    route_selections: dict[str, Any] = {}
    for route in ROUTES:
        route_selections[route] = _select_route(
            training,
            route=route,
            shared_spec=shared_spec,
        )
        print(
            route,
            json.dumps(route_selections[route]["best"], sort_keys=True),
            flush=True,
        )

    bundle = _fit_route_bundle(
        training,
        shared_spec=shared_spec,
        route_selections=route_selections,
    )
    current_candidate = read_json(args.current_candidate)
    current_bundle = current_candidate["model"]
    pooled_model = v1_report["fitted_models"]["best_profile"]

    print("scoring strict unused sources once", flush=True)
    predictions = _strict_predictions(
        strict,
        current_bundle=current_bundle,
        pooled_model=pooled_model,
        route_bundle=bundle,
    )
    strict_metrics = _scope_metrics(predictions)
    current_mape = strict_metrics["all_success"]["current_v5_m1_reserved"][
        "source_equal_mape"
    ]
    pooled_mape = strict_metrics["all_success"]["pooled_profile_expansion"][
        "source_equal_mape"
    ]
    if not math.isclose(float(current_mape), 0.15432226675609423, abs_tol=1e-12):
        raise ValueError("current V5 strict metric drifted")
    if not math.isclose(float(pooled_mape), 0.09611206189080983, abs_tol=1e-12):
        raise ValueError("pooled v1 strict metric drifted")

    v5_report = read_json(args.v5_report)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "shadow_route_specific_validation_complete",
        "publishable": False,
        "production_model_mutated": False,
        "protocol": {
            "selection": (
                "per-route feature set, alpha, and hierarchical blend selected "
                "from training-source LOSO only"
            ),
            "strict_holdout_use": "single final evaluation after route freeze",
            "target": "log(observed_reserved / observed_allocated)",
            "oom_treatment": "right-censored; excluded from center fitting",
        },
        "input_bindings": {
            "v1_report": {"path": str(args.v1_report), "sha256": sha256_file(args.v1_report)},
            "v5_report": {"path": str(args.v5_report), "sha256": sha256_file(args.v5_report)},
            "current_candidate": {
                "path": str(args.current_candidate),
                "sha256": sha256_file(args.current_candidate),
            },
        },
        "shared_model_spec": shared_spec,
        "route_selections": route_selections,
        "fitted_bundle": bundle,
        "strict_metrics": strict_metrics,
        "experiment_counts": _experiment_counts(v5_report),
        "limitations": [
            "strict validation contains five independent sources",
            "other LoRA has eight successful fit sources and remains a partial-pooling route",
            "other FULL has no strict unused configurations in this report",
            "actual per-microbatch maximum sequence and sampler order remain unrecorded",
            "the upper bound and admission head are not recalibrated in this center-only diagnostic",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "report.json", report)
    write_json(args.output_dir / "route_expansion_bundle.json", bundle)
    write_jsonl(args.output_dir / "strict_predictions.jsonl", predictions)
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(strict_metrics, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
