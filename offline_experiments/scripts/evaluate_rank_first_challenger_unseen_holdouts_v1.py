#!/usr/bin/env python3
"""Replay the frozen rank-first challenger on fit-disjoint H800 holdouts.

The evaluator is deliberately read-only with respect to model fitting: it
loads the already fitted challenger heads, joins three historical prospective
campaign snapshots to their frozen request descriptions, reconstructs the
non-Packing workload from the saved token profiles, and evaluates predictions
without changing a coefficient or hyperparameter.

These campaigns are fit-disjoint but not temporally fresh for the challenger:
their outcomes existed before the challenger implementation was frozen.  The
result is therefore an expanded retrospective validation, not release
acceptance evidence.
"""

from __future__ import annotations

import argparse
import itertools
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ROOT, read_json, sha256_file, sha256_json, write_json
from fit_rank_first_throughput_challenger_v1 import (
    MATERIAL_GAP,
    InvariantStaticProfiles,
    _configuration,
    _evaluate_entries,
    _invariant_basis,
    _linear_score,
    _mechanism,
    _prediction_entries,
)
from h800_theory_basis import _model_geometry


SCHEMA = "sft_rank_first_challenger_fit_disjoint_replay/v1"
IMPLEMENTATION_VERSION = (
    "sft_rank_first_challenger_fit_disjoint_replay/"
    "2026-08-05.three-h800-holdouts-v1"
)


DEFAULT_COHORTS = (
    {
        "cohort_id": "fresh_holdout_v2",
        "observations": ROOT / "artifacts" / "h800_fresh_holdout_observations_v2.json",
        "frozen_requests": ROOT / "artifacts" / "h800_frozen_predictions_before_fresh_holdout_v2.json",
        "profiles": ROOT / "artifacts" / "fresh_holdout_v2" / "profiles",
    },
    {
        "cohort_id": "final_unseen_holdout_v1",
        "observations": ROOT / "artifacts" / "h800_final_unseen_holdout_observations_v1.json",
        "frozen_requests": ROOT / "artifacts" / "h800_frozen_predictions_before_final_unseen_holdout_v1.json",
        "profiles": ROOT / "artifacts" / "final_unseen_holdout_v1" / "profiles",
    },
    {
        "cohort_id": "bounded_memory_v2_fresh_holdout_v1",
        "observations": ROOT / "artifacts" / "h800_bounded_memory_v2_fresh_holdout_observations_v1.json",
        "frozen_requests": ROOT / "artifacts" / "h800_frozen_predictions_before_bounded_memory_v2_fresh_holdout_v1.json",
        "profiles": ROOT / "artifacts" / "bounded_memory_v2_fresh_holdout_v1" / "profiles",
    },
)


def _prediction_map(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    predictions = report.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("Frozen request report has no prediction list")
    result: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            continue
        request_id = str(prediction.get("request_id") or "")
        if not request_id or request_id in result:
            raise ValueError("Frozen request ids must be present and unique")
        result[request_id] = dict(prediction)
    return result


def _observation_request_id(row: Mapping[str, Any]) -> str:
    return str(row.get("candidate_id") or row.get("job_id") or "")


def _hardware_contract(
    hardware: Mapping[str, Any], theory: Mapping[str, Any]
) -> tuple[float, dict[str, float]]:
    priors = theory["physical_priors"]["values"]
    memory_bytes = float(hardware["memory_bytes_reported_by_torch"])
    return memory_bytes, {
        "dense_bf16_peak_flops_per_gpu": float(
            hardware["bf16_dense_peak_flops_per_second_for_mfu"]
        ),
        "hbm_bandwidth_bytes_per_second": float(
            priors["memory_bandwidth_bytes_per_s"]
        ),
        "intra_node_bandwidth_bytes_per_second": float(
            priors["collective_bandwidth_bytes_per_s"]
        ),
        "collective_latency_seconds": float(
            priors["collective_latency_seconds"]
        ),
    }


def _record_from_prediction(
    prediction: Mapping[str, Any],
    *,
    model_by_id: Mapping[str, Mapping[str, Any]],
    fixed_lora: Mapping[str, Any],
    physical_priors: Mapping[str, float],
) -> dict[str, Any]:
    configuration = prediction.get("configuration") or {}
    model_id = str(configuration.get("model_id") or "")
    if model_id not in model_by_id:
        raise KeyError(f"Model {model_id!r} is outside the in-domain catalog")
    training_mode = str(configuration.get("training_mode") or "").lower()
    model = dict(model_by_id[model_id])
    lora_contract = {
        **dict(fixed_lora),
        "rank": int(configuration.get("lora_rank") or fixed_lora["rank"]),
    }
    geometry = _model_geometry(
        {
            "model_id": model_id,
            "model_parameters": int(model["actual_parameters"]),
            "train_type": training_mode,
        },
        model,
        lora_contract,
    )
    return {
        "scenario": {
            "model_id": model_id,
            "dataset_id": str(configuration["dataset_id"]),
            "train_type": training_mode,
            "gpu_count": int(configuration["gpu_count"]),
            "physical_mbs": int(configuration["physical_mbs"]),
            "target_gbs": int(configuration["target_gbs"]),
            "cutoff_len": int(configuration["cutoff_len"]),
        },
        "selector": {
            "training_mode": training_mode,
            "zero_stage": int(configuration["zero_stage"]),
            "gradient_checkpointing": bool(
                configuration["gradient_checkpointing"]
            ),
            "packing": bool(configuration["packing"]),
            "offload": bool(configuration["offload"]),
            "kernel_path": str(configuration["kernel_path"]),
            "dtype": str(configuration["dtype"]),
        },
        "model_basis": geometry,
        "performance": {
            "gradient_accumulation_steps": int(
                configuration["gradient_accumulation_steps"]
            ),
            "physical_priors": dict(physical_priors),
        },
    }


def _load_candidates(
    cohorts: Sequence[Mapping[str, Any]],
    *,
    profiles: InvariantStaticProfiles,
    hardware_memory_bytes: float,
    physical_priors: Mapping[str, float],
    inventory: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    models = inventory.get("models")
    fixed_lora = inventory.get("fixed_lora")
    if not isinstance(models, list) or not isinstance(fixed_lora, Mapping):
        raise ValueError("Model inventory is incomplete")
    model_by_id = {
        str(model["id"]): dict(model)
        for model in models
        if isinstance(model, Mapping)
    }
    successful: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    bindings: dict[str, Any] = {}
    seen_request_ids: set[str] = set()

    for cohort in cohorts:
        cohort_id = str(cohort["cohort_id"])
        observation_path = Path(cohort["observations"])
        request_path = Path(cohort["frozen_requests"])
        profile_dir = Path(cohort["profiles"])
        observations = read_json(observation_path)
        request_report = read_json(request_path)
        requests = _prediction_map(request_report)
        rows = observations.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"Observation report {cohort_id} has no rows")
        bindings[cohort_id] = {
            "observations": {
                "path": str(observation_path.resolve()),
                "sha256": sha256_file(observation_path),
                "rows": len(rows),
            },
            "frozen_requests": {
                "path": str(request_path.resolve()),
                "sha256": sha256_file(request_path),
                "predictions": len(requests),
            },
            "profiles": {
                "path": str(profile_dir.resolve()),
                "files": len(list(profile_dir.glob("*.jsonl"))),
            },
        }
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            request_id = _observation_request_id(row)
            if not request_id or request_id in seen_request_ids:
                raise ValueError("Observation request ids must be present and unique")
            seen_request_ids.add(request_id)
            prediction = requests.get(request_id)
            if prediction is None:
                raise KeyError(f"No frozen request description for {request_id}")
            configuration = prediction.get("configuration") or {}
            reason = None
            if str(row.get("outcome") or "").lower() != "success":
                reason = "outcome_not_success"
            elif str(configuration.get("hardware_id") or "").lower() != "h800":
                reason = "card_outside_scope"
            elif bool(configuration.get("packing")):
                reason = "packing_outside_scope"
            elif bool(configuration.get("offload")):
                reason = "offload_outside_scope"
            elif str(configuration.get("model_id") or "") not in model_by_id:
                reason = "model_outside_in_domain_catalog"
            observed = row.get("observed_effective_tokens_per_second")
            if reason is None and (observed is None or float(observed) <= 0.0):
                reason = "throughput_label_missing"
            if reason is not None:
                excluded.append(
                    {
                        "cohort_id": cohort_id,
                        "scenario_id": str(row.get("scenario_id") or ""),
                        "request_id": request_id,
                        "model_id": str(configuration.get("model_id") or ""),
                        "outcome": str(row.get("outcome") or ""),
                        "reason": reason,
                    }
                )
                continue
            record = _record_from_prediction(
                prediction,
                model_by_id=model_by_id,
                fixed_lora=fixed_lora,
                physical_priors=physical_priors,
            )
            basis = _invariant_basis(
                record,
                profiles,
                hardware_memory_bytes=hardware_memory_bytes,
            )
            successful.append(
                {
                    "scenario_id": str(row["scenario_id"]),
                    "cohort_id": cohort_id,
                    "job_id": str(row.get("job_id") or request_id),
                    "request_id": request_id,
                    "record": record,
                    "candidate_key": [
                        int(configuration["gpu_count"]),
                        int(configuration["physical_mbs"]),
                        int(configuration["zero_stage"]),
                        bool(configuration["gradient_checkpointing"]),
                    ],
                    "source_role": "fit_disjoint_historical_holdout",
                    "historical": False,
                    "features": basis["invariant_features"],
                    "basis": basis,
                    "observed_log_throughput": math.log(float(observed)),
                }
            )
    return successful, excluded, bindings


def _partition_multi_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    counts = Counter(str(row["scenario_id"]) for row in candidates)
    primary = [row for row in candidates if counts[str(row["scenario_id"])] >= 2]
    diagnostic = [row for row in candidates if counts[str(row["scenario_id"])] < 2]
    return primary, diagnostic


def _pair_scope_metrics(
    entries: Sequence[tuple[Mapping[str, Any], float, float, float]],
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[Mapping[str, Any], float, float, float]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry[0]["scenario_id"])].append(entry)
    counters: dict[str, dict[str, int]] = defaultdict(
        lambda: {"comparisons": 0, "correct": 0, "material_comparisons": 0, "material_correct": 0}
    )
    for scope in (
        "all",
        "same_gpu_count",
        "cross_gpu_count",
        "cross_mechanism",
        "same_mbs_cross_mechanism",
        "same_gpu_and_mbs_cross_mechanism",
    ):
        counters[scope]
    for rows in grouped.values():
        for left, right in itertools.combinations(rows, 2):
            left_record = left[0]["record"]
            right_record = right[0]["record"]
            left_scenario = left_record.get("scenario") or {}
            right_scenario = right_record.get("scenario") or {}
            observed_delta = float(left[0]["observed_log_throughput"]) - float(
                right[0]["observed_log_throughput"]
            )
            if abs(observed_delta) <= 1.0e-12:
                continue
            predicted_delta = float(left[1]) - float(right[1])
            correct = (observed_delta > 0.0) == (predicted_delta > 0.0)
            material = 1.0 - math.exp(-abs(observed_delta)) >= MATERIAL_GAP
            scopes = ["all"]
            if int(left_scenario["gpu_count"]) == int(right_scenario["gpu_count"]):
                scopes.append("same_gpu_count")
            else:
                scopes.append("cross_gpu_count")
            if _mechanism(left[0]) != _mechanism(right[0]):
                scopes.append("cross_mechanism")
            if (
                int(left_scenario["physical_mbs"])
                == int(right_scenario["physical_mbs"])
                and _mechanism(left[0]) != _mechanism(right[0])
            ):
                scopes.append("same_mbs_cross_mechanism")
                if int(left_scenario["gpu_count"]) == int(
                    right_scenario["gpu_count"]
                ):
                    scopes.append("same_gpu_and_mbs_cross_mechanism")
            for scope in scopes:
                counters[scope]["comparisons"] += 1
                counters[scope]["correct"] += int(correct)
                counters[scope]["material_comparisons"] += int(material)
                counters[scope]["material_correct"] += int(material and correct)
    return {
        scope: {
            **counts,
            "accuracy": (
                counts["correct"] / counts["comparisons"]
                if counts["comparisons"]
                else None
            ),
            "material_accuracy": (
                counts["material_correct"] / counts["material_comparisons"]
                if counts["material_comparisons"]
                else None
            ),
        }
        for scope, counts in sorted(counters.items())
    }


def _candidate_predictions(
    entries: Sequence[tuple[Mapping[str, Any], float, float, float]],
) -> list[dict[str, Any]]:
    by_scenario: dict[str, list[tuple[Mapping[str, Any], float, float, float]]] = defaultdict(list)
    for entry in entries:
        by_scenario[str(entry[0]["scenario_id"])].append(entry)
    result: list[dict[str, Any]] = []
    for scenario_id, rows in sorted(by_scenario.items()):
        ordered = sorted(rows, key=lambda item: float(item[1]), reverse=True)
        observed_order = sorted(
            rows,
            key=lambda item: float(item[0]["observed_log_throughput"]),
            reverse=True,
        )
        observed_rank = {id(item[0]): rank for rank, item in enumerate(observed_order, 1)}
        for predicted_rank, (candidate, final_log, absolute_log, rank_score) in enumerate(ordered, 1):
            record = candidate["record"]
            scenario = record.get("scenario") or {}
            selector = record.get("selector") or {}
            observed = math.exp(float(candidate["observed_log_throughput"]))
            predicted = math.exp(float(final_log))
            result.append(
                {
                    "cohort_id": str(candidate["cohort_id"]),
                    "scenario_id": scenario_id,
                    "job_id": str(candidate["job_id"]),
                    "request_id": str(candidate["request_id"]),
                    "model_id": str(scenario["model_id"]),
                    "dataset_id": str(scenario["dataset_id"]),
                    "training_mode": str(selector["training_mode"]),
                    "gpu_count": int(scenario["gpu_count"]),
                    "mbs": int(scenario["physical_mbs"]),
                    "zero_stage": int(selector["zero_stage"]),
                    "gradient_checkpointing": bool(selector["gradient_checkpointing"]),
                    "configuration": (
                        f"G{int(scenario['gpu_count'])}/"
                        + _configuration(candidate)
                    ),
                    "observed_effective_tokens_per_second": observed,
                    "predicted_effective_tokens_per_second": predicted,
                    "absolute_head_effective_tokens_per_second": math.exp(float(absolute_log)),
                    "rank_score": float(rank_score),
                    "absolute_percentage_error": abs(predicted / observed - 1.0),
                    "predicted_rank": predicted_rank,
                    "observed_rank": observed_rank[id(candidate)],
                }
            )
    return result


def _evaluate_slice(
    candidates: Sequence[Mapping[str, Any]],
    absolute_head: Mapping[str, Any],
    rank_head: Mapping[str, Any],
) -> tuple[dict[str, Any], list[tuple[Mapping[str, Any], float, float, float]]]:
    entries = _prediction_entries(candidates, absolute_head, rank_head)
    metrics = _evaluate_entries(entries)
    metrics["pair_scope_breakdown"] = _pair_scope_metrics(entries)
    return metrics, entries


def _fit_dataset_ids(challenger: Mapping[str, Any]) -> set[str]:
    frozen_v5_path = Path(challenger["source_bindings"]["frozen_v5"]["path"])
    extension_path = Path(
        challenger["source_bindings"]["extension_evaluation"]["path"]
    )
    frozen_v5 = read_json(frozen_v5_path)
    extension = read_json(extension_path)
    result = {str(value) for value in frozen_v5["data"]["datasets"]}
    for row in extension.get("rows") or []:
        if str(row.get("state") or "") == "success_authoritative":
            result.add(str(row["dataset_id"]))
    return result


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


def _render_markdown(report: Mapping[str, Any]) -> str:
    overall = report["evaluation"]["primary_aggregate"]
    lines = [
        "# Rank-first Challenger 扩大范围的拟合隔离回放",
        "",
        "> 本报告冻结 challenger 参数，不重拟合、不调参。三批数据没有进入 challenger 系数拟合，",
        "> 但实验结果早于 challenger 冻结，因此属于拟合隔离的回溯验证，不是全新的前瞻发布验收。",
        "",
        "## 汇总结论",
        "",
        "| 候选 | 场景 | Pairwise | 跨机制 Pairwise | 重要差距跨机制 | Exact Top-1 | 平均 regret | 最坏 regret | 绝对 MAPE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {candidate_rows} | {scenario_rows} | {pairwise} | {cross} | {material} | {top1} | {mean_regret} | {worst_regret} | {mape} |".format(
            candidate_rows=overall["candidate_rows"],
            scenario_rows=overall["scenario_rows"],
            pairwise=_pct(overall["all_pairwise_accuracy"]),
            cross=_pct(overall["cross_mechanism_pairwise_accuracy"]),
            material=_pct(overall["material_cross_mechanism_pairwise_accuracy"]),
            top1=_pct(overall["exact_top1_fraction"]),
            mean_regret=_pct(overall["mean_top1_regret"]),
            worst_regret=_pct(overall["worst_top1_regret"]),
            mape=_pct(overall["set_aware_absolute_mape"]),
        ),
        "",
        "## 分批结果",
        "",
        "| 验证批次 | 候选 | 场景 | Pairwise | 跨机制 Pairwise | Exact Top-1 | 平均 regret | 绝对 MAPE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort_id, metrics in report["evaluation"]["by_cohort"].items():
        lines.append(
            "| {cohort} | {rows} | {scenarios} | {pairwise} | {cross} | {top1} | {regret} | {mape} |".format(
                cohort=cohort_id,
                rows=metrics["candidate_rows"],
                scenarios=metrics["scenario_rows"],
                pairwise=_pct(metrics["all_pairwise_accuracy"]),
                cross=_pct(metrics["cross_mechanism_pairwise_accuracy"]),
                top1=_pct(metrics["exact_top1_fraction"]),
                regret=_pct(metrics["mean_top1_regret"]),
                mape=_pct(metrics["set_aware_absolute_mape"]),
            )
        )
    lines.extend(
        [
            "",
            "## 分训练方式结果",
            "",
            "| 训练方式 | 候选 | 场景 | Pairwise | 跨机制 Pairwise | 重要差距跨机制 | Exact Top-1 | 平均 regret | 绝对 MAPE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, metrics in report["evaluation"]["by_training_mode"].items():
        lines.append(
            "| {mode} | {rows} | {scenarios} | {pairwise} | {cross} | {material} | {top1} | {regret} | {mape} |".format(
                mode=mode,
                rows=metrics["candidate_rows"],
                scenarios=metrics["scenario_rows"],
                pairwise=_pct(metrics["all_pairwise_accuracy"]),
                cross=_pct(metrics["cross_mechanism_pairwise_accuracy"]),
                material=_pct(metrics["material_cross_mechanism_pairwise_accuracy"]),
                top1=_pct(metrics["exact_top1_fraction"]),
                regret=_pct(metrics["mean_top1_regret"]),
                mape=_pct(metrics["set_aware_absolute_mape"]),
            )
        )
    lines.extend(
        [
            "",
            "## 配置对分层",
            "",
            "| 配置对范围 | 比较数 | 准确率 | 重要差距比较数 | 重要差距准确率 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    pair_labels = {
        "same_gpu_count": "相同 GPU 数",
        "cross_gpu_count": "不同 GPU 数",
        "cross_mechanism": "跨机制",
        "same_mbs_cross_mechanism": "相同 MBS 的跨机制（GPU 数可不同）",
        "same_gpu_and_mbs_cross_mechanism": "相同 GPU 数和 MBS 的跨机制",
    }
    pair_scopes = overall["pair_scope_breakdown"]
    for scope in pair_labels:
        if scope not in pair_scopes:
            continue
        metrics = pair_scopes[scope]
        lines.append(
            "| {label} | {count} | {accuracy} | {material_count} | {material_accuracy} |".format(
                label=pair_labels[scope],
                count=metrics["comparisons"],
                accuracy=_pct(metrics["accuracy"]),
                material_count=metrics["material_comparisons"],
                material_accuracy=_pct(metrics["material_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "## 逐场景选择",
            "",
            "| 批次 | 场景 | 候选 | Challenger 选择 | Oracle | Regret | 跨机制 Pairwise |",
            "|---|---|---:|---|---|---:|---:|",
        ]
    )
    predictions = report["predictions"]
    by_request = {
        key: row
        for row in predictions
        for key in (row["request_id"], row["job_id"])
    }
    for detail in overall["details"]:
        selected = by_request[detail["selected_job_id"]]
        oracle = by_request[detail["oracle_job_id"]]
        lines.append(
            "| {cohort} | {scenario} | {count} | {selected} | {oracle} | {regret} | {cross} |".format(
                cohort=selected["cohort_id"],
                scenario=detail["scenario_id"],
                count=detail["candidates"],
                selected=selected["configuration"],
                oracle=oracle["configuration"],
                regret=_pct(detail["top1_regret"]),
                cross=_pct(detail["cross_mechanism_pairwise_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            f"- 拟合数据集与本次验证数据集重叠数：{report['disjointness_audit']['dataset_id_overlap_count']}。",
            f"- 多候选主评估包含 {overall['scenario_rows']} 个场景；单候选诊断点不进入排序指标。",
            "- OOM 不具有完整吞吐标签，不虚构吞吐，也不进入吞吐排序。",
            "- 当前卡型仍固定为 H800，且只评估纯文本、Packing=false、Offload=false。",
            "- 由于这些实验结果早于 challenger 冻结，本报告不能单独将 challenger 标记为可发布。",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    *,
    challenger_path: Path,
    inventory_path: Path,
    hardware_path: Path,
    theory_path: Path,
    cohorts: Sequence[Mapping[str, Any]] = DEFAULT_COHORTS,
) -> dict[str, Any]:
    challenger = read_json(challenger_path)
    inventory = read_json(inventory_path)
    hardware = read_json(hardware_path)
    theory = read_json(theory_path)
    if challenger.get("schema") != "sft_rank_first_throughput_challenger/v1":
        raise ValueError("Unexpected challenger schema")
    profile_dirs = [Path(cohort["profiles"]) for cohort in cohorts]
    profiles = InvariantStaticProfiles(profile_dirs)
    memory_bytes, physical_priors = _hardware_contract(hardware, theory)
    candidates, excluded, bindings = _load_candidates(
        cohorts,
        profiles=profiles,
        hardware_memory_bytes=memory_bytes,
        physical_priors=physical_priors,
        inventory=inventory,
    )
    primary, single_candidate = _partition_multi_candidate(candidates)
    absolute_head = challenger["models"]["final_absolute_head"]
    rank_head = challenger["models"]["final_rank_head"]
    overall, overall_entries = _evaluate_slice(primary, absolute_head, rank_head)
    by_cohort: dict[str, Any] = {}
    for cohort_id in sorted({str(row["cohort_id"]) for row in primary}):
        current = [row for row in primary if str(row["cohort_id"]) == cohort_id]
        by_cohort[cohort_id] = _evaluate_slice(
            current, absolute_head, rank_head
        )[0]
    by_training_mode: dict[str, Any] = {}
    for mode in sorted(
        {
            str((row["record"].get("selector") or {})["training_mode"])
            for row in primary
        }
    ):
        current = [
            row
            for row in primary
            if str((row["record"].get("selector") or {})["training_mode"])
            == mode
        ]
        by_training_mode[mode] = _evaluate_slice(
            current, absolute_head, rank_head
        )[0]

    diagnostic_entries = _prediction_entries(
        single_candidate, absolute_head, rank_head
    ) if single_candidate else []
    fit_dataset_ids = _fit_dataset_ids(challenger)
    validation_dataset_ids = {
        str((row["record"].get("scenario") or {})["dataset_id"])
        for row in candidates
    }
    overlap = sorted(fit_dataset_ids & validation_dataset_ids)
    model_ids = sorted(
        {
            str((row["record"].get("scenario") or {})["model_id"])
            for row in candidates
        }
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fit_disjoint_retrospective_replay_complete",
        "analysis_only": True,
        "publishable": False,
        "model_refit": False,
        "hyperparameters_changed": False,
        "gpu_experiments_launched": False,
        "evaluation_contract": {
            "target": "effective_tokens_per_second",
            "primary_scope": "H800 dense Qwen3 text SFT, packing=false, offload=false",
            "ordering_head": "frozen final_rank_head",
            "absolute_center_head": "frozen final_absolute_head",
            "success_only_throughput_ranking": True,
            "oom_throughput_imputed": False,
            "fit_disjoint": not overlap,
            "temporally_fresh_for_challenger": False,
            "temporal_interpretation": (
                "holdout outcomes predate challenger freeze; use as expanded "
                "retrospective evidence, not prospective release acceptance"
            ),
        },
        "source_bindings": {
            "challenger": {
                "path": str(challenger_path.resolve()),
                "sha256": sha256_file(challenger_path),
                "report_sha256": challenger.get("report_sha256"),
            },
            "model_inventory": {
                "path": str(inventory_path.resolve()),
                "sha256": sha256_file(inventory_path),
            },
            "hardware": {
                "path": str(hardware_path.resolve()),
                "sha256": sha256_file(hardware_path),
            },
            "theory_basis": {
                "path": str(theory_path.resolve()),
                "sha256": sha256_file(theory_path),
            },
            "cohorts": bindings,
        },
        "disjointness_audit": {
            "fit_dataset_ids": sorted(fit_dataset_ids),
            "validation_dataset_ids": sorted(validation_dataset_ids),
            "dataset_id_overlap": overlap,
            "dataset_id_overlap_count": len(overlap),
            "validation_model_ids": model_ids,
            "model_ids_intentionally_in_domain": True,
        },
        "population": {
            "successful_in_scope_candidates": len(candidates),
            "primary_multi_candidate_rows": len(primary),
            "primary_scenarios": len({str(row["scenario_id"]) for row in primary}),
            "single_candidate_diagnostics": len(single_candidate),
            "excluded_rows": len(excluded),
            "excluded_reason_counts": dict(
                sorted(Counter(str(row["reason"]) for row in excluded).items())
            ),
        },
        "evaluation": {
            "primary_aggregate": overall,
            "by_cohort": by_cohort,
            "by_training_mode": by_training_mode,
            "single_candidate_diagnostic": (
                _evaluate_entries(diagnostic_entries)
                if diagnostic_entries
                else None
            ),
        },
        "predictions": _candidate_predictions(overall_entries),
        "single_candidate_predictions": _candidate_predictions(diagnostic_entries),
        "excluded": excluded,
        "limitations": [
            "validation outcomes predate challenger freeze",
            "no coefficient or hyperparameter was selected on these rows, but prior human inspection cannot be excluded",
            "single-candidate scenarios cannot evaluate ranking",
            "OOM rows are excluded from throughput ranking",
            "packing, VL, offload and non-H800 cards are outside this replay",
        ],
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--challenger",
        type=Path,
        default=ROOT / "artifacts" / "rank_first_throughput_challenger_v1.json",
    )
    parser.add_argument(
        "--model-inventory",
        type=Path,
        default=ROOT / "artifacts" / "model_inventory.json",
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=ROOT / "config" / "hardware.json",
    )
    parser.add_argument(
        "--theory-basis",
        type=Path,
        default=ROOT / "artifacts" / "h800_theory_basis.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "rank_first_challenger_fit_disjoint_replay_v1.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "artifacts" / "rank_first_challenger_fit_disjoint_replay_v1.md",
    )
    args = parser.parse_args()
    report = build_report(
        challenger_path=args.challenger,
        inventory_path=args.model_inventory,
        hardware_path=args.hardware,
        theory_path=args.theory_basis,
    )
    write_json(args.output, report)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_render_markdown(report), encoding="utf-8")
    print(
        {
            "output": str(args.output.resolve()),
            "markdown_output": str(args.markdown_output.resolve()),
            "report_sha256": report["report_sha256"],
            "population": report["population"],
            "primary_aggregate": report["evaluation"]["primary_aggregate"],
        }
    )


if __name__ == "__main__":
    main()
