#!/usr/bin/env python3
"""Replay v2 control and frozen v3 shadow on one identical request corpus."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h800_unified_bounded_memory_v3_data as v3_data
from common import ROOT, sha256_file, sha256_json, write_json, write_jsonl
from h800_unified_bounded_memory_model import load_artifact, predict_records

SCHEMA = "sft_h800_unified_bounded_memory_shadow/v3"
DEFAULT_V2 = ROOT / "artifacts" / "h800_unified_bounded_memory_candidate_v2.json"
DEFAULT_V3 = ROOT / "artifacts" / "h800_unified_bounded_memory_candidate_v3.json"
DEFAULT_OUTPUT_DIR = (
    ROOT / "diagnostics" / "h800_unified_bounded_memory_shadow_v3_20260810"
)


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metrics(
    records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    safe_limit = float(predictions[0]["safe_limit_bytes"])
    errors: list[float] = []
    signed: list[float] = []
    by_source: dict[str, list[float]] = defaultdict(list)
    safe = []
    unsafe = []
    oom = []
    for record, prediction in zip(records, predictions):
        if record["state"] == "censored":
            oom.append(prediction)
            continue
        observed = float(record["target_reserved_bytes"])
        error = abs(float(prediction["center_bytes"]) / observed - 1.0)
        signed_error = float(prediction["center_bytes"]) / observed - 1.0
        errors.append(error)
        signed.append(signed_error)
        by_source[str(record["source_id"])].append(error)
        (safe if observed <= safe_limit else unsafe).append(prediction)
    safe_admitted = sum(bool(row["admitted"]) for row in safe)
    oom_admitted = sum(bool(row["admitted"]) for row in oom)
    return {
        "rows": len(records),
        "exact_success_rows": len(errors),
        "oom_rows": len(oom),
        "center_source_equal_mape": statistics.fmean(
            statistics.fmean(values) for values in by_source.values()
        ),
        "center_row_mape": statistics.fmean(errors),
        "center_signed_bias": statistics.fmean(signed),
        "center_p90_ape": _percentile(errors, 0.9),
        "actual_safe_success_rows": len(safe),
        "admitted_safe_success_rows": safe_admitted,
        "safe_success_admission_rate": safe_admitted / len(safe),
        "actual_unsafe_success_rows": len(unsafe),
        "admitted_unsafe_success_rows": sum(bool(row["admitted"]) for row in unsafe),
        "admitted_oom_rows": oom_admitted,
        "oom_admission_rate": oom_admitted / len(oom),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    control = report["metrics"]["v2_control"]
    shadow = report["metrics"]["v3_shadow"]
    return "\n".join(
        [
            "# H800 统一有界显存 v3 影子重放",
            "",
            "v2 作为控制决策，v3 只记录影子结果；本次未改变任何线上或调度决策。",
            "",
            "| 模型 | 中心 MAPE | 中心偏差 | P90 APE | 安全放行率 | OOM 放行率 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| v2 控制 | {control['center_source_equal_mape']:.2%} | {control['center_signed_bias']:+.2%} | {control['center_p90_ape']:.2%} | {control['admitted_safe_success_rows']}/{control['actual_safe_success_rows']} = {control['safe_success_admission_rate']:.2%} | {control['admitted_oom_rows']}/{control['oom_rows']} = {control['oom_admission_rate']:.2%} |",
            f"| v3 影子 | {shadow['center_source_equal_mape']:.2%} | {shadow['center_signed_bias']:+.2%} | {shadow['center_p90_ape']:.2%} | {shadow['admitted_safe_success_rows']}/{shadow['actual_safe_success_rows']} = {shadow['safe_success_admission_rate']:.2%} | {shadow['admitted_oom_rows']}/{shadow['oom_rows']} = {shadow['oom_admission_rate']:.2%} |",
            "",
            f"决策分歧：{report['decision_delta']['disagreements']}/{report['decision_delta']['rows']}。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, default=DEFAULT_V2)
    parser.add_argument("--v3", type=Path, default=DEFAULT_V3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    control = load_artifact(args.v2)
    shadow = load_artifact(args.v3)
    records, data_audit = v3_data.development_records()
    control_predictions = predict_records(records, control)
    shadow_predictions = predict_records(records, shadow)
    control_metrics = _metrics(records, control_predictions)
    shadow_metrics = _metrics(records, shadow_predictions)
    rows = []
    decision_counts = defaultdict(int)
    for record, old, new in zip(records, control_predictions, shadow_predictions):
        key = f"v2_{'admit' if old['admitted'] else 'reject'}__v3_{'admit' if new['admitted'] else 'reject'}"
        decision_counts[key] += 1
        rows.append(
            {
                "record_id": str(record["record_id"]),
                "state": str(record["state"]),
                "model_id": str(record["model_id"]),
                "train_type": str(record["train_type"]),
                "gpu_count": int(record["gpu_count"]),
                "zero_stage": int(record["zero_stage"]),
                "gc": bool(record["gc"]),
                "mbs": int(record["mbs"]),
                "cutoff_len": int(record["cutoff_len"]),
                "v2_center_bytes": float(old["center_bytes"]),
                "v2_upper_bytes": float(old["admission_upper_bytes"]),
                "v2_admitted_control_decision": bool(old["admitted"]),
                "v3_center_bytes": float(new["center_bytes"]),
                "v3_upper_bytes": float(new["admission_upper_bytes"]),
                "v3_admitted_shadow_only": bool(new["admitted"]),
                "decision_cell": key,
            }
        )
    disagreements = sum(
        old["admitted"] != new["admitted"]
        for old, new in zip(control_predictions, shadow_predictions)
    )
    gates = {
        "production_or_control_decision_not_mutated": True,
        "v3_zero_known_oom_admission": shadow_metrics["admitted_oom_rows"] == 0,
        "v3_safe_admission_at_least_0p95": shadow_metrics["safe_success_admission_rate"]
        >= 0.95,
        "v3_center_source_mape_better_than_v2": shadow_metrics[
            "center_source_equal_mape"
        ]
        < control_metrics["center_source_equal_mape"],
        "v3_oom_admission_not_worse_than_v2": shadow_metrics["admitted_oom_rows"]
        <= control_metrics["admitted_oom_rows"],
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "shadow_replay_complete_waiting_prospective_canary",
        "production_model_mutated": False,
        "queues_mutated": False,
        "gpu_experiments_launched": False,
        "control_decision_source": "frozen_v2_artifact",
        "shadow_decision_source": "frozen_v3_artifact",
        "evidence_label": "retrospective shadow integration; not prospective acceptance",
        "inputs": {
            "v2": {"path": str(args.v2.resolve()), "sha256": sha256_file(args.v2)},
            "v3": {"path": str(args.v3.resolve()), "sha256": sha256_file(args.v3)},
        },
        "data_audit": {
            "rows": len(records),
            "sources": len({str(row["source_id"]) for row in records}),
            "prospective_rows": data_audit["prospective_acceptance_rows_used"],
        },
        "metrics": {"v2_control": control_metrics, "v3_shadow": shadow_metrics},
        "decision_delta": {
            "rows": len(records),
            "disagreements": disagreements,
            "disagreement_rate": disagreements / len(records),
            "cells": dict(sorted(decision_counts.items())),
        },
        "gates": gates,
        "all_shadow_gates_passed": all(gates.values()),
    }
    report["report_sha256"] = sha256_json(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "shadow_predictions.jsonl", rows)
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
