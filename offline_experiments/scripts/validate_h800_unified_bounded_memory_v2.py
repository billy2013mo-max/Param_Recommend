#!/usr/bin/env python3
"""Replay the frozen unified bounded-memory v2 artifact without refitting.

The 31-row strict set is dataset-disjoint but has been inspected historically,
so it is an external locked replay rather than prospective evidence.  The
106-row historical set is configuration-disjoint but shares a connected
dataset component with historical fit rows.  The two scopes are never pooled
under an "independent validation" label.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import benchmark_h800_memory_center_models_v1 as bm
import calibrate_h800_m1_safety_upper_v2 as historical_builder
import fit_h800_unified_resource_partial_v1 as base
import migrate_refit_h800_historical_memory_v1 as historical_inputs
import refit_h800_m1_all_unused_validation_v3 as historical_defaults
import validate_h800_profile_expansion_offline_v1 as profile_validation
from common import (
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from h800_unified_bounded_memory_model import load_artifact, predict_records

SCHEMA = "sft_h800_unified_bounded_memory_validation/v2"
IMPLEMENTATION_VERSION = (
    "sft_h800_unified_bounded_memory_validation/"
    "2026-08-10.frozen-v2-strict31-historical106"
)
DEFAULT_ARTIFACT = ROOT / "artifacts" / "h800_unified_bounded_memory_candidate_v2.json"
DEFAULT_STRICT = (
    ROOT
    / "diagnostics"
    / "h800_profile_expansion_offline_v1_20260809"
    / "strict_predictions.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "diagnostics" / "h800_unified_bounded_memory_validation_v2_20260810"
)


def _strict_records(path: Path) -> list[dict[str, Any]]:
    inventory = read_json(base.DEFAULT_INVENTORY)
    hardware = read_json(base.DEFAULT_HARDWARE)
    model_by_id = {str(row["id"]): row for row in inventory["models"]}
    capacity = int(hardware["memory_bytes_reported_by_torch"])
    records = []
    for row in read_jsonl(path):
        outcome = str(row["outcome"])
        reference = base._reference_from_summary_row(
            row,
            model_by_id=model_by_id,
            fixed_lora=inventory["fixed_lora"],
            capacity_bytes=capacity,
        )
        records.append(
            {
                "record_id": f"strict::{row['observation_id']}",
                "cluster_id": str(row["cluster_id"]),
                "source_id": str(row["source_id"]),
                "origin": str(row.get("origin") or "strict"),
                "campaign": str(row.get("origin") or "strict"),
                "role": "strict_unused_dataset",
                "state": "exact" if outcome == "success" else "censored",
                "reference_bytes": reference,
                "target_reserved_bytes": (
                    float(row["observed_reserved_bytes"])
                    if outcome == "success"
                    else None
                ),
                "censor_lower_bytes": float(capacity) if outcome == "oom" else None,
                "features": base._complete_feature_values(
                    row["model_features"],
                    packing=bool(row.get("packing")),
                    samples_per_pack=1.0,
                ),
                "model_id": str(row["model_id"]),
                "train_type": str(row["training_mode"]),
                "gpu_count": int(row["gpu_count"]),
                "zero_stage": int(row["zero_stage"]),
                "gc": bool(row["gradient_checkpointing"]),
                "mbs": int(row["mbs"]),
                "cutoff_len": int(row["cutoff_len"]),
                "packing": bool(row.get("packing")),
                "profile_sha256": str(row.get("profile_sha256") or ""),
            }
        )
    if len(records) != 31 or Counter(row["state"] for row in records) != {
        "exact": 29,
        "censored": 2,
    }:
        raise ValueError("strict31 composition drifted")
    return records


def _historical_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    args = argparse.Namespace(
        canonical=historical_inputs.DEFAULT_CANONICAL,
        theory_basis=historical_inputs.DEFAULT_THEORY_BASIS,
        dataset_analysis=historical_inputs.DEFAULT_DATASET_ANALYSIS,
        new_queue=historical_inputs.DEFAULT_NEW_QUEUE,
        current_old=historical_inputs.DEFAULT_CURRENT_OLD,
        inventory=historical_defaults.DEFAULT_VALIDATION_INVENTORY,
        hardware=historical_defaults.DEFAULT_HARDWARE,
    )
    prepared = historical_builder._prepare_records(args)
    capacity = int(read_json(args.hardware)["memory_bytes_reported_by_torch"])
    records = []
    for raw in prepared["holdout"]:
        row = profile_validation._record_summary(
            raw, "historical_unfitted_configuration"
        )
        outcome = str(row["outcome"])
        memory = raw["memory"]
        records.append(
            {
                "record_id": f"historical_holdout::{row['cluster_id']}",
                "cluster_id": str(row["cluster_id"]),
                "source_id": str(row["source_id"]),
                "origin": "historical_fixed_holdout_167",
                "campaign": "historical_fixed_holdout_167",
                "role": "historical_configuration_holdout",
                "state": "exact" if outcome == "success" else "censored",
                "reference_bytes": float(memory["analytic_reference_bytes"]),
                "target_reserved_bytes": (
                    float(row["observed_reserved_bytes"])
                    if outcome == "success"
                    else None
                ),
                "censor_lower_bytes": float(capacity) if outcome == "oom" else None,
                "features": base._complete_feature_values(
                    row["model_features"],
                    packing=bool(row.get("packing")),
                    samples_per_pack=1.0,
                ),
                "model_id": str(row["model_id"]),
                "train_type": str(row["training_mode"]),
                "gpu_count": int(row["gpu_count"]),
                "zero_stage": int(row["zero_stage"]),
                "gc": bool(row["gradient_checkpointing"]),
                "mbs": int(row["mbs"]),
                "cutoff_len": int(row["cutoff_len"]),
                "packing": bool(row.get("packing")),
                "profile_sha256": str(row.get("profile_sha256") or ""),
            }
        )
    if len(records) != 106 or Counter(row["state"] for row in records) != {
        "exact": 74,
        "censored": 32,
    }:
        raise ValueError("historical106 composition drifted")
    return records, dict(prepared["audit"])


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _score(
    records: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any], *, scope: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = predict_records(records, artifact)
    safe_limit = float(artifact["hardware_domain"]["safe_limit_bytes"])
    details = []
    by_source: dict[str, list[float]] = defaultdict(list)
    errors: list[float] = []
    signed: list[float] = []
    for record, prediction in zip(records, predictions):
        detail = {
            **dict(prediction),
            "validation_scope": scope,
            "cluster_id": str(record["cluster_id"]),
            "source_id": str(record["source_id"]),
            "origin": str(record["origin"]),
            "state": str(record["state"]),
            "model_id": str(record["model_id"]),
            "train_type": str(record["train_type"]),
            "gpu_count": int(record["gpu_count"]),
            "zero_stage": int(record["zero_stage"]),
            "gc": bool(record["gc"]),
            "mbs": int(record["mbs"]),
            "cutoff_len": int(record["cutoff_len"]),
            "packing": bool(record["packing"]),
        }
        if record["state"] == "exact":
            observed = float(record["target_reserved_bytes"])
            error = abs(float(prediction["center_bytes"]) / observed - 1.0)
            signed_error = float(prediction["center_bytes"]) / observed - 1.0
            detail.update(
                {
                    "observed_reserved_bytes": observed,
                    "actually_safe_success": observed <= safe_limit,
                    "absolute_percentage_error": error,
                    "signed_percentage_error": signed_error,
                }
            )
            errors.append(error)
            signed.append(signed_error)
            by_source[str(record["source_id"])].append(error)
        else:
            detail["censor_lower_bytes"] = float(record["censor_lower_bytes"])
        details.append(detail)
    exact = [row for row in details if row["state"] == "exact"]
    safe = [row for row in exact if row["actually_safe_success"]]
    unsafe = [row for row in exact if not row["actually_safe_success"]]
    oom = [row for row in details if row["state"] == "censored"]
    safe_admitted = sum(bool(row["admitted"]) for row in safe)
    oom_admitted = sum(bool(row["admitted"]) for row in oom)
    metrics = {
        "configurations": len(details),
        "independent_sources": len({str(row["source_id"]) for row in details}),
        "success_configurations": len(exact),
        "oom_configurations": len(oom),
        "center_row_mape": statistics.fmean(errors) if errors else None,
        "center_source_equal_mape": (
            statistics.fmean(statistics.fmean(values) for values in by_source.values())
            if by_source
            else None
        ),
        "center_signed_bias": statistics.fmean(signed) if signed else None,
        "center_p90_ape": _percentile(errors, 0.9),
        "actual_safe_success_configurations": len(safe),
        "admitted_safe_success_configurations": safe_admitted,
        "safe_success_admission_rate": safe_admitted / len(safe) if safe else None,
        "actual_unsafe_success_configurations": len(unsafe),
        "admitted_unsafe_success_configurations": sum(
            bool(row["admitted"]) for row in unsafe
        ),
        "admitted_oom_configurations": oom_admitted,
        "oom_admission_rate": oom_admitted / len(oom) if oom else None,
    }
    return details, metrics


def _markdown(report: Mapping[str, Any]) -> str:
    strict = report["results"]["strict31_locked_replay"]["metrics"]
    historical = report["results"]["historical106_configuration_holdout"]["metrics"]
    lines = [
        "# H800 统一有界显存模型 v2 冻结重放",
        "",
        "模型系数与准入余量均来自已冻结 artifact；本次运行未重拟合、未调参。",
        "",
        "| 范围 | 中心 MAPE | 中心偏差 | P90 APE | 安全放行率 | OOM 放行率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in (("严格 31", strict), ("历史 106", historical)):
        lines.append(
            f"| {label} | {metrics['center_source_equal_mape']:.2%} | "
            f"{metrics['center_signed_bias']:+.2%} | {metrics['center_p90_ape']:.2%} | "
            f"{metrics['admitted_safe_success_configurations']}/"
            f"{metrics['actual_safe_success_configurations']} = "
            f"{metrics['safe_success_admission_rate']:.2%} | "
            f"{metrics['admitted_oom_configurations']}/"
            f"{metrics['oom_configurations']} = {metrics['oom_admission_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "证据解释：严格 31 条是数据集不相交但历史上已查看过的锁定重放；历史 106 条只保证配置不相交，不保证数据集独立。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--strict", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    artifact = load_artifact(args.artifact)
    strict_records = _strict_records(args.strict)
    historical_records, historical_audit = _historical_records()
    fit_records, _unused_strict, _fit_audit = bm._load_records()
    fit_cluster_ids = {
        str(row["record_id"]).split("::", 1)[1]
        for row in fit_records
        if "::" in str(row["record_id"])
    }
    strict_clusters = {str(row["cluster_id"]) for row in strict_records}
    historical_clusters = {str(row["cluster_id"]) for row in historical_records}
    if strict_clusters & historical_clusters:
        raise ValueError("strict31 and historical106 cluster IDs overlap")
    if historical_clusters & fit_cluster_ids:
        raise ValueError("historical106 configuration IDs entered the v2 fit")

    strict_details, strict_metrics = _score(
        strict_records, artifact, scope="strict31_locked_replay"
    )
    historical_details, historical_metrics = _score(
        historical_records,
        artifact,
        scope="historical106_configuration_holdout",
    )
    checks = {
        "strict31_zero_oom_admission": strict_metrics["admitted_oom_configurations"]
        == 0,
        "historical106_zero_oom_admission": historical_metrics[
            "admitted_oom_configurations"
        ]
        == 0,
        "strict31_safe_admission_at_least_0p95": float(
            strict_metrics["safe_success_admission_rate"]
        )
        >= 0.95,
        "historical106_safe_admission_at_least_0p95": float(
            historical_metrics["safe_success_admission_rate"]
        )
        >= 0.95,
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_artifact_replayed_no_refit",
        "production_model_mutated": False,
        "artifact": {
            "path": str(args.artifact.resolve()),
            "sha256": sha256_file(args.artifact),
            "artifact_sha256": artifact["artifact_sha256"],
        },
        "evidence_contract": {
            "strict31_locked_replay": (
                "five dataset-disjoint sources; historically inspected; not prospective"
            ),
            "historical106_configuration_holdout": (
                "configuration-disjoint; one connected historical source; not dataset-independent"
            ),
            "model_or_margin_refit": False,
        },
        "overlap_audit": {
            "strict31_historical106_cluster_overlap": 0,
            "historical106_v2_fit_cluster_overlap": 0,
            "historical_builder": historical_audit,
        },
        "results": {
            "strict31_locked_replay": {
                "metrics": strict_metrics,
                "input": {
                    "path": str(args.strict.resolve()),
                    "sha256": sha256_file(args.strict),
                },
            },
            "historical106_configuration_holdout": {
                "metrics": historical_metrics,
                "inputs": {
                    "canonical": {
                        "path": str(historical_inputs.DEFAULT_CANONICAL.resolve()),
                        "sha256": sha256_file(historical_inputs.DEFAULT_CANONICAL),
                    },
                    "theory_basis": {
                        "path": str(historical_inputs.DEFAULT_THEORY_BASIS.resolve()),
                        "sha256": sha256_file(historical_inputs.DEFAULT_THEORY_BASIS),
                    },
                },
            },
        },
        "checks": checks,
        "all_admission_checks_passed": all(checks.values()),
    }
    report["report_sha256"] = sha256_json(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = [*strict_details, *historical_details]
    predictions_path = args.output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
