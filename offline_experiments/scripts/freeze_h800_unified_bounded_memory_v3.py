#!/usr/bin/env python3
"""Freeze the unified H800 memory v3 center and shared risk heads.

All previously inspected rows, including historical106 and the targeted
mechanism campaign, are development data.  The artifact is therefore a shadow
candidate only; acceptance evidence must be generated prospectively after the
artifact checksum is frozen.
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
import h800_unified_bounded_memory_v3_data as v3_data
import validate_h800_unified_bounded_memory_v2 as retrospective
from common import ROOT, sha256_file, sha256_json, write_json, write_jsonl
from h800_unified_bounded_memory_model import ARTIFACT_SCHEMA_V3

SCHEMA = "sft_h800_unified_bounded_memory_freeze_report/v3"
IMPLEMENTATION_VERSION = (
    "sft_h800_unified_bounded_memory/"
    "2026-08-10.v3-shared-center-risk-mechanism-features"
)
DEFAULT_OUTPUT_DIR = ROOT / "diagnostics" / "h800_unified_bounded_memory_v3_20260810"
DEFAULT_ARTIFACT = ROOT / "artifacts" / "h800_unified_bounded_memory_candidate_v3.json"
SAFE_LIMIT_FRACTION = 0.95
OUTER_FOLDS = 10
CENTER_CANDIDATE = {
    "candidate_id": "v3_center_bounded_mechanism_p0p25_a0p03_s0p8_h0p2",
    "basis_kind": "bounded_mechanism",
    "feature_variant": "anchor_scale_v3_mechanisms",
    "alpha": 0.03,
    "correction_shrinkage": 0.8,
    "huber_delta": 0.2,
    "source_weight_power": 0.25,
    "censored_constraint_weight": 1.0,
}
RISK_CANDIDATE = {
    "candidate_id": "v3_risk_bounded_linear_p0_a0p3_s1_h0p2_cw10",
    "basis_kind": "bounded_linear",
    "feature_variant": "anchor_scale_v3_mechanisms",
    "alpha": 0.3,
    "correction_shrinkage": 1.0,
    "huber_delta": 0.2,
    "source_weight_power": 0.0,
    "censored_constraint_weight": 10.0,
}


def _safe_limit() -> float:
    return SAFE_LIMIT_FRACTION * bm.DEVICE_CAPACITY_BYTES


def _model_bytes(
    records: Sequence[Mapping[str, Any]], model: Mapping[str, Any]
) -> list[float]:
    corrections = bm._predict_correction(records, model)
    return [
        float(row["reference_bytes"]) * math.exp(float(correction))
        for row, correction in zip(records, corrections)
    ]


def _calibrate_risk_multiplier(
    records: Sequence[Mapping[str, Any]], risk_bytes: Sequence[float]
) -> float:
    safe_limit = _safe_limit()
    required = [1.0]
    for row, predicted in zip(records, risk_bytes):
        if row["state"] == "censored" and predicted <= safe_limit:
            required.append(safe_limit / predicted)
    return math.nextafter(max(required), math.inf)


def _decorate_admission(
    records: Sequence[Mapping[str, Any]],
    center_details: list[dict[str, Any]],
    risk_model: Mapping[str, Any],
    risk_multiplier: float,
) -> list[dict[str, Any]]:
    risk_bytes = _model_bytes(records, risk_model)
    safe_limit = _safe_limit()
    for record, detail, risk in zip(records, center_details, risk_bytes):
        center = float(detail["predicted_reserved_bytes"])
        upper = max(center, risk * risk_multiplier)
        detail.update(
            {
                "center_bytes": center,
                "risk_head_bytes": risk,
                "risk_upper_multiplier": risk_multiplier,
                "admission_upper_bytes": upper,
                "safe_limit_bytes": safe_limit,
                "admitted": upper <= safe_limit,
                "mbs": int(record["mbs"]),
                "cutoff_len": int(record["cutoff_len"]),
            }
        )
    return center_details


def _admission_metrics(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    safe_limit = _safe_limit()
    safe = [
        row
        for row in details
        if row["state"] == "exact"
        and float(row["observed_reserved_bytes"]) <= safe_limit
    ]
    unsafe = [
        row
        for row in details
        if row["state"] == "exact"
        and float(row["observed_reserved_bytes"]) > safe_limit
    ]
    oom = [row for row in details if row["state"] == "censored"]
    admitted_safe = sum(bool(row["admitted"]) for row in safe)
    admitted_oom = sum(bool(row["admitted"]) for row in oom)
    return {
        "actual_safe_success_rows": len(safe),
        "admitted_safe_success_rows": admitted_safe,
        "safe_success_admission_rate": admitted_safe / len(safe) if safe else None,
        "actual_unsafe_success_rows": len(unsafe),
        "admitted_unsafe_success_rows": sum(bool(row["admitted"]) for row in unsafe),
        "oom_rows": len(oom),
        "admitted_oom_rows": admitted_oom,
        "oom_admission_rate": admitted_oom / len(oom) if oom else None,
    }


def _nested_predictions(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for index, held_sources in enumerate(bm._source_folds(records, OUTER_FOLDS)):
        training = [row for row in records if str(row["source_id"]) not in held_sources]
        evaluation = [row for row in records if str(row["source_id"]) in held_sources]
        risk_model = bm._fit_model(training, RISK_CANDIDATE)
        multiplier = _calibrate_risk_multiplier(
            training, _model_bytes(training, risk_model)
        )
        center = bm._prediction_details(
            training,
            evaluation,
            CENTER_CANDIDATE,
            fold_id=f"outer_{index:02d}",
        )
        outer = _decorate_admission(evaluation, center, risk_model, multiplier)
        predictions.extend(outer)
        audits.append(
            {
                "outer_fold": index,
                "held_out_sources": sorted(held_sources),
                "train_rows": len(training),
                "test_rows": len(evaluation),
                "risk_multiplier": multiplier,
                "center_metrics": bm._metrics(outer),
                "admission_metrics": _admission_metrics(outer),
            }
        )
    return predictions, audits


def _score_with_models(
    records: Sequence[Mapping[str, Any]],
    center_model: Mapping[str, Any],
    risk_model: Mapping[str, Any],
    multiplier: float,
    *,
    fold_id: str,
) -> list[dict[str, Any]]:
    center_bytes = _model_bytes(records, center_model)
    risk_bytes = _model_bytes(records, risk_model)
    details: list[dict[str, Any]] = []
    for record, center, risk in zip(records, center_bytes, risk_bytes):
        observed = (
            float(record["target_reserved_bytes"])
            if record["state"] == "exact"
            else None
        )
        upper = max(center, risk * multiplier)
        detail: dict[str, Any] = {
            "record_id": str(record["record_id"]),
            "source_id": str(record["source_id"]),
            "origin": str(record["origin"]),
            "role": str(record["role"]),
            "state": str(record["state"]),
            "model_id": str(record["model_id"]),
            "train_type": str(record["train_type"]),
            "gpu_count": int(record["gpu_count"]),
            "zero_stage": int(record["zero_stage"]),
            "gc": bool(record["gc"]),
            "packing": bool(record["packing"]),
            "mbs": int(record["mbs"]),
            "cutoff_len": int(record["cutoff_len"]),
            "reference_bytes": float(record["reference_bytes"]),
            "predicted_reserved_bytes": center,
            "predicted_reserved_gib": center / (1 << 30),
            "center_bytes": center,
            "risk_head_bytes": risk,
            "risk_upper_multiplier": multiplier,
            "admission_upper_bytes": upper,
            "safe_limit_bytes": _safe_limit(),
            "admitted": upper <= _safe_limit(),
            "fold_id": fold_id,
            "candidate_id": str(CENTER_CANDIDATE["candidate_id"]),
        }
        if observed is not None:
            detail.update(
                {
                    "observed_reserved_bytes": observed,
                    "observed_reserved_gib": observed / (1 << 30),
                    "absolute_percentage_error": abs(center / observed - 1.0),
                    "signed_percentage_error": center / observed - 1.0,
                }
            )
        else:
            lower = float(record["censor_lower_bytes"])
            detail.update(
                {
                    "censor_lower_bytes": lower,
                    "censor_satisfied": center >= lower,
                    "censor_shortfall_fraction": max(0.0, 1.0 - center / lower),
                }
            )
        details.append(detail)
    return details


def _markdown(report: Mapping[str, Any]) -> str:
    center = report["primary_nested_source_cv"]["center_metrics"]
    admission = report["primary_nested_source_cv"]["admission_metrics"]
    strict_center = report["retrospective_diagnostics"]["strict31"]["center_metrics"]
    strict_admission = report["retrospective_diagnostics"]["strict31"][
        "admission_metrics"
    ]
    return "\n".join(
        [
            "# H800 统一有界显存模型 v3 冻结记录",
            "",
            "历史重放和定向实验均已进入开发集；只有冻结后的 shadow/canary 可以作为前瞻验收证据。",
            "",
            "| 范围 | 中心 MAPE | 中心偏差 | P90 APE | 安全放行率 | OOM 放行率 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| 开发集按源折外 | {center['source_equal_mape']:.2%} | {center['signed_bias']:+.2%} | {center['p90_ape']:.2%} | {admission['admitted_safe_success_rows']}/{admission['actual_safe_success_rows']} = {admission['safe_success_admission_rate']:.2%} | {admission['admitted_oom_rows']}/{admission['oom_rows']} = {admission['oom_admission_rate']:.2%} |",
            f"| 严格 31（回顾性） | {strict_center['source_equal_mape']:.2%} | {strict_center['signed_bias']:+.2%} | {strict_center['p90_ape']:.2%} | {strict_admission['admitted_safe_success_rows']}/{strict_admission['actual_safe_success_rows']} = {strict_admission['safe_success_admission_rate']:.2%} | {strict_admission['admitted_oom_rows']}/{strict_admission['oom_rows']} = {strict_admission['oom_admission_rate']:.2%} |",
            "",
            "状态：冻结 shadow 候选；未修改线上推荐结果。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()

    records, data_audit = v3_data.development_records()
    nested, fold_audit = _nested_predictions(records)
    center_metrics = bm._metrics(nested)
    admission_metrics = _admission_metrics(nested)
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
        raise RuntimeError(f"v3 development freeze gates failed: {gates}")

    center_model = bm._fit_model(records, CENTER_CANDIDATE)
    risk_model = bm._fit_model(records, RISK_CANDIDATE)
    risk_multiplier = _calibrate_risk_multiplier(
        records, _model_bytes(records, risk_model)
    )
    full_fit = _score_with_models(
        records,
        center_model,
        risk_model,
        risk_multiplier,
        fold_id="final_training_self_consistency",
    )
    strict = v3_data.correct_records(
        retrospective._strict_records(retrospective.DEFAULT_STRICT)
    )
    strict_details = _score_with_models(
        strict,
        center_model,
        risk_model,
        risk_multiplier,
        fold_id="retrospective_strict31",
    )
    generated = datetime.now(timezone.utc).isoformat()
    artifact: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA_V3,
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
        "candidate": {
            "center": dict(CENTER_CANDIDATE),
            "risk": dict(RISK_CANDIDATE),
        },
        "model": center_model,
        "admission": {
            "kind": "independent_shared_risk_head",
            "rule": "admit iff max(center_bytes, risk_head_bytes * upper_multiplier) <= 0.95 * capacity_bytes",
            "risk_model": risk_model,
            "upper_multiplier": risk_multiplier,
            "safe_limit_fraction": SAFE_LIMIT_FRACTION,
            "calibration": "final-fit right-censored risk-head self-consistency; nested source folds used only for development acceptance",
        },
        "development_acceptance": {
            "center_metrics": center_metrics,
            "admission_metrics": admission_metrics,
            "gates": gates,
            "all_passed": all(gates.values()),
        },
        "evidence_contract": {
            "historical106_role": "development_after_v2_replay_failure",
            "targeted15_role": "development_mechanism_fit",
            "strict31_role": "retrospective_diagnostic_only",
            "prospective_acceptance_rows_used": 0,
        },
        "inputs": {
            "targeted_queue": {
                "path": str(v3_data.TARGETED_QUEUE.resolve()),
                "sha256": sha256_file(v3_data.TARGETED_QUEUE),
            },
            "targeted_results": {
                "path": str(v3_data.TARGETED_RESULTS.resolve()),
                "sha256": sha256_file(v3_data.TARGETED_RESULTS),
            },
            "strict31": {
                "path": str(retrospective.DEFAULT_STRICT.resolve()),
                "sha256": sha256_file(retrospective.DEFAULT_STRICT),
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "model_math": {
                "path": str(Path(bm.__file__).resolve()),
                "sha256": sha256_file(Path(bm.__file__)),
            },
            "data_builder": {
                "path": str(Path(v3_data.__file__).resolve()),
                "sha256": sha256_file(Path(v3_data.__file__)),
            },
        },
    }
    artifact["artifact_sha256"] = sha256_json(artifact)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": generated,
        "status": "frozen_shadow_candidate_waiting_prospective_validation",
        "production_model_mutated": False,
        "data_audit": data_audit,
        "development_rows": len(records),
        "development_sources": len({str(row["source_id"]) for row in records}),
        "development_exact": sum(row["state"] == "exact" for row in records),
        "development_right_censored": sum(
            row["state"] == "censored" for row in records
        ),
        "candidate": artifact["candidate"],
        "primary_nested_source_cv": {
            "protocol": "ten source-grouped outer folds; risk multiplier calibrated only on each outer training split",
            "center_metrics": center_metrics,
            "admission_metrics": admission_metrics,
            "folds": fold_audit,
        },
        "final_artifact_calibration": {
            "risk_multiplier": risk_multiplier,
            "self_center_metrics": bm._metrics(full_fit),
            "self_admission_metrics": _admission_metrics(full_fit),
        },
        "retrospective_diagnostics": {
            "strict31": {
                "label": "historically inspected; not prospective acceptance evidence",
                "center_metrics": bm._metrics(strict_details),
                "admission_metrics": _admission_metrics(strict_details),
            }
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
    write_jsonl(args.output_dir / "nested_admission_predictions.jsonl", nested)
    write_jsonl(args.output_dir / "final_fit_predictions.jsonl", full_fit)
    write_jsonl(
        args.output_dir / "strict31_retrospective_predictions.jsonl", strict_details
    )
    write_json(args.artifact, artifact)
    write_json(args.output_dir / "freeze_report.json", report)
    (args.output_dir / "freeze_report.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
