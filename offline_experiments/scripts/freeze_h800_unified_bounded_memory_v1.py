#!/usr/bin/env python3
"""Freeze the unified H800 bounded center and one-layer admission margin.

The candidate is fixed before any prospective experiment.  Every development
prediction is source-disjoint.  The final admission multiplier is calibrated
from grouped out-of-fold OOM evidence and is never fitted on an OOM as though
it were an exact peak-memory target.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import benchmark_h800_memory_center_models_v1 as bm
from common import ROOT, sha256_file, sha256_json, write_json, write_jsonl
from h800_unified_bounded_memory_model import ARTIFACT_SCHEMA

SCHEMA = "sft_h800_unified_bounded_memory_freeze_report/v1"
IMPLEMENTATION_VERSION = (
    "sft_h800_unified_bounded_memory/"
    "2026-08-10.bounded-mechanism-shrunk-center-unshrunk-risk-guard"
)
DEFAULT_OUTPUT_DIR = ROOT / "diagnostics" / "h800_unified_bounded_memory_v2_20260810"
DEFAULT_ARTIFACT = ROOT / "artifacts" / "h800_unified_bounded_memory_candidate_v2.json"
SAFE_LIMIT_FRACTION = 0.95
INNER_FOLDS = 5
OUTER_FOLDS = 10
CANDIDATE = {
    "candidate_id": (
        "bounded_mechanism__anchor_scale_lora_zero3_nogc_activation__a1__s0.8__h0.2"
    ),
    "basis_kind": "bounded_mechanism",
    "feature_variant": "anchor_scale_lora_zero3_nogc_activation",
    "alpha": 1.0,
    "correction_shrinkage": 0.8,
    "huber_delta": 0.2,
}


def _safe_limit() -> float:
    return SAFE_LIMIT_FRACTION * bm.DEVICE_CAPACITY_BYTES


def _calibrate_multiplier(
    details: Sequence[Mapping[str, Any]], *, safe_limit: float
) -> float:
    required = [1.0]
    for row in details:
        if row["state"] != "censored":
            continue
        predicted = float(row["predicted_reserved_bytes"])
        if predicted <= safe_limit:
            required.append(safe_limit / predicted)
    return math.nextafter(max(required), math.inf)


def _risk_guard_bytes(row: Mapping[str, Any]) -> float:
    center = float(row["predicted_reserved_bytes"])
    reference = float(row["reference_bytes"])
    shrinkage = float(CANDIDATE["correction_shrinkage"])
    return reference * math.exp(math.log(center / reference) / shrinkage)


def _calibrate_risk_guard_multiplier(
    details: Sequence[Mapping[str, Any]], *, safe_limit: float
) -> float:
    required = [1.0]
    for row in details:
        if row["state"] != "censored":
            continue
        risk_guard = _risk_guard_bytes(row)
        if risk_guard <= safe_limit:
            required.append(safe_limit / risk_guard)
    return math.nextafter(max(required), math.inf)


def _admission_metrics(
    details: Sequence[Mapping[str, Any]], *, safe_limit: float
) -> dict[str, Any]:
    safe = [
        row
        for row in details
        if row["state"] == "exact"
        and float(row["observed_reserved_bytes"]) <= safe_limit
    ]
    unsafe_success = [
        row
        for row in details
        if row["state"] == "exact"
        and float(row["observed_reserved_bytes"]) > safe_limit
    ]
    oom = [row for row in details if row["state"] == "censored"]
    admitted_safe = sum(bool(row["admitted"]) for row in safe)
    admitted_unsafe_success = sum(bool(row["admitted"]) for row in unsafe_success)
    admitted_oom = sum(bool(row["admitted"]) for row in oom)
    return {
        "actual_safe_success_rows": len(safe),
        "admitted_safe_success_rows": admitted_safe,
        "safe_success_admission_rate": (admitted_safe / len(safe) if safe else None),
        "actual_unsafe_success_rows": len(unsafe_success),
        "admitted_unsafe_success_rows": admitted_unsafe_success,
        "oom_rows": len(oom),
        "admitted_oom_rows": admitted_oom,
        "oom_admission_rate": admitted_oom / len(oom) if oom else None,
    }


def _nested_admission_predictions(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    safe_limit = _safe_limit()
    predictions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for index, held_sources in enumerate(bm._source_folds(records, OUTER_FOLDS)):
        training = [row for row in records if str(row["source_id"]) not in held_sources]
        evaluation = [row for row in records if str(row["source_id"]) in held_sources]
        inner = bm._grouped_predictions(
            training,
            CANDIDATE,
            folds=bm._source_folds(training, INNER_FOLDS),
            prefix=f"outer_{index:02d}_inner_",
        )
        multiplier = _calibrate_multiplier(inner, safe_limit=safe_limit)
        training_fit = bm._prediction_details(
            training,
            training,
            CANDIDATE,
            fold_id=f"outer_{index:02d}_training_self_consistency",
        )
        risk_guard_multiplier = _calibrate_risk_guard_multiplier(
            training_fit, safe_limit=safe_limit
        )
        outer = bm._prediction_details(
            training,
            evaluation,
            CANDIDATE,
            fold_id=f"outer_{index:02d}",
        )
        for row in outer:
            row["upper_multiplier"] = multiplier
            row["risk_guard_bytes"] = _risk_guard_bytes(row)
            row["risk_guard_multiplier"] = risk_guard_multiplier
            row["admission_upper_bytes"] = max(
                float(row["predicted_reserved_bytes"]) * multiplier,
                float(row["risk_guard_bytes"]) * risk_guard_multiplier,
            )
            row["safe_limit_bytes"] = safe_limit
            row["admitted"] = row["admission_upper_bytes"] <= safe_limit
        predictions.extend(outer)
        audits.append(
            {
                "outer_fold": index,
                "held_out_sources": sorted(held_sources),
                "train_rows": len(training),
                "test_rows": len(evaluation),
                "inner_calibrated_upper_multiplier": multiplier,
                "training_self_consistency_risk_guard_multiplier": risk_guard_multiplier,
                "outer_center_metrics": bm._metrics(outer),
                "outer_admission_metrics": _admission_metrics(
                    outer, safe_limit=safe_limit
                ),
            }
        )
    return predictions, audits


def _markdown(report: Mapping[str, Any]) -> str:
    center = report["primary_nested_source_cv"]["center_metrics"]
    admission = report["primary_nested_source_cv"]["admission_metrics"]
    final = report["final_artifact_calibration"]
    return "\n".join(
        [
            "# H800 统一有界显存模型冻结记录",
            "",
            "该候选在后续验证前冻结；验证数据不得用于重拟合或调参。",
            "",
            "| 指标 | 结果 |",
            "|---|---:|",
            f"| 中心 source-equal MAPE | {center['source_equal_mape']:.2%} |",
            f"| 中心 signed bias | {center['signed_bias']:+.2%} |",
            f"| 中心 P90 APE | {center['p90_ape']:.2%} |",
            f"| 安全配置放行率 | {admission['admitted_safe_success_rows']}/{admission['actual_safe_success_rows']} = {admission['safe_success_admission_rate']:.2%} |",
            f"| OOM 放行率 | {admission['admitted_oom_rows']}/{admission['oom_rows']} = {admission['oom_admission_rate']:.2%} |",
            f"| 最终中心安全余量 | {(final['upper_multiplier'] - 1.0):.4%} |",
            f"| 最终未收缩风险护栏余量 | {(final['risk_guard_multiplier'] - 1.0):.4%} |",
            "",
            "模型状态：冻结 shadow 候选，尚未获准替换线上模型。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()

    records, _strict, data_audit = bm._load_records()
    nested_predictions, nested_audit = _nested_admission_predictions(records)
    center_metrics = bm._metrics(nested_predictions)
    admission_metrics = _admission_metrics(nested_predictions, safe_limit=_safe_limit())
    grouped = bm._grouped_predictions(
        records,
        CANDIDATE,
        folds=bm._source_folds(records, OUTER_FOLDS),
        prefix="final_calibration_",
    )
    final_multiplier = _calibrate_multiplier(grouped, safe_limit=_safe_limit())
    final_model = bm._fit_model(records, CANDIDATE)
    final_training_fit = bm._prediction_details(
        records, records, CANDIDATE, fold_id="final_training_self_consistency"
    )
    final_risk_guard_multiplier = _calibrate_risk_guard_multiplier(
        final_training_fit, safe_limit=_safe_limit()
    )
    gates = {
        "center_source_equal_mape_at_most_0p13": float(
            center_metrics["source_equal_mape"]
        )
        <= 0.13,
        "absolute_signed_bias_at_most_0p03": abs(float(center_metrics["signed_bias"]))
        <= 0.03,
        "center_p90_ape_at_most_0p31": float(center_metrics["p90_ape"]) <= 0.31,
        "safe_admission_at_least_0p95": float(
            admission_metrics["safe_success_admission_rate"]
        )
        >= 0.95,
        "zero_oom_admission": admission_metrics["admitted_oom_rows"] == 0,
    }
    if not all(gates.values()):
        raise RuntimeError(f"development freeze gates failed: {gates}")

    generated = datetime.now(timezone.utc).isoformat()
    artifact: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": generated,
        "status": "frozen_shadow_candidate_waiting_validation",
        "immutable": True,
        "publishable": False,
        "production_override_allowed": False,
        "production_model_mutated": False,
        "gpu_family": "H800",
        "hardware_domain": {
            "capacity_bytes": bm.DEVICE_CAPACITY_BYTES,
            "safe_limit_bytes": _safe_limit(),
        },
        "candidate": dict(CANDIDATE),
        "model": final_model,
        "admission": {
            "rule": "admit iff max(center_bytes * upper_multiplier, unshrunk_risk_guard_bytes * risk_guard_multiplier) <= 0.95 * capacity_bytes",
            "upper_multiplier": final_multiplier,
            "risk_guard_multiplier": final_risk_guard_multiplier,
            "safe_limit_fraction": SAFE_LIMIT_FRACTION,
            "calibration": "center margin from ten-fold source-grouped OOF; risk guard from final-fit censored self-consistency",
        },
        "development_acceptance": {
            "center_metrics": center_metrics,
            "admission_metrics": admission_metrics,
            "gates": gates,
            "all_passed": all(gates.values()),
        },
        "inputs": {
            "queue": {
                "path": str(bm.base.DEFAULT_QUEUE.resolve()),
                "sha256": sha256_file(bm.base.DEFAULT_QUEUE),
            },
            "audit": {
                "path": str(bm.base.DEFAULT_AUDIT.resolve()),
                "sha256": sha256_file(bm.base.DEFAULT_AUDIT),
            },
            "prior_rows": {
                "path": str(bm.base.DEFAULT_PRIOR_ROWS.resolve()),
                "sha256": sha256_file(bm.base.DEFAULT_PRIOR_ROWS),
            },
            "inventory": {
                "path": str(bm.base.DEFAULT_INVENTORY.resolve()),
                "sha256": sha256_file(bm.base.DEFAULT_INVENTORY),
            },
            "hardware": {
                "path": str(bm.base.DEFAULT_HARDWARE.resolve()),
                "sha256": sha256_file(bm.base.DEFAULT_HARDWARE),
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "model_math": {
                "path": str(Path(bm.__file__).resolve()),
                "sha256": sha256_file(Path(bm.__file__)),
            },
        },
    }
    artifact["artifact_sha256"] = sha256_json(artifact)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": generated,
        "status": "frozen_shadow_candidate_waiting_validation",
        "production_model_mutated": False,
        "data_audit": data_audit,
        "candidate": dict(CANDIDATE),
        "primary_nested_source_cv": {
            "protocol": "ten outer source folds; five inner source folds calibrate the center margin; each outer training split calibrates final-fit censored risk self-consistency",
            "center_metrics": center_metrics,
            "admission_metrics": admission_metrics,
            "folds": nested_audit,
        },
        "final_artifact_calibration": {
            "protocol": "ten-fold source-grouped OOF center margin plus final-fit censored self-consistency risk guard",
            "upper_multiplier": final_multiplier,
            "margin_fraction": final_multiplier - 1.0,
            "risk_guard_multiplier": final_risk_guard_multiplier,
            "risk_guard_margin_fraction": final_risk_guard_multiplier - 1.0,
        },
        "gates": gates,
        "all_development_gates_passed": all(gates.values()),
        "artifact": {
            "path": str(args.artifact.resolve()),
            "artifact_sha256": artifact["artifact_sha256"],
        },
    }
    report["report_sha256"] = sha256_json(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.output_dir / "nested_admission_predictions.jsonl",
        nested_predictions,
    )
    write_jsonl(args.output_dir / "final_calibration_predictions.jsonl", grouped)
    write_json(args.artifact, artifact)
    write_json(args.output_dir / "freeze_report.json", report)
    (args.output_dir / "freeze_report.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
