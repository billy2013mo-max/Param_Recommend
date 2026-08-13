#!/usr/bin/env python3
"""Leave-one-mechanism-out test of a GENERIC admission head.

Today's design fits one coefficient set per mechanism (training mode x ZeRO stage
x gradient checkpointing x GPU count), because the head's only features are the
two pressure ratios.  With 20 unpacked mechanisms that means 20 lookups, and any
mechanism without both safe and failing samples gets no coefficients at all --
so it is refused unconditionally.  Only 2 of 20 currently have a head.

The center model does not work this way: it feeds the mechanism in as features
and covers all 20 with a single coefficient set.  This script asks whether the
admission head can do the same, and whether that stays safe.

The test is deliberately harsh: withhold ONE MECHANISM ENTIRELY, fit on the
rest, then score the withheld mechanism.  A generic head is only worth adopting
if it admits zero failing configurations on mechanisms it has never seen.

Analysis only -- fits nothing into a frozen artifact, launches no GPU work.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import fit_h800_m1_new_mechanism_admission_heads_v1 as M
from common import ARTIFACT_DIR, sha256_json, write_json
from fit_h800_lora_source_disjoint_recalibration_v1 import (
    VARIANT_FEATURES,
    VARIANT_M1,
    _fit_bundle,
    _predict_center,
    _source_id,
)


SCHEMA = "sft_h800_generic_admission_head_leave_mechanism_out/v1"
SAFE_LIMIT_BYTES = 142635080089.6
COVERAGE = 0.95
ALPHA = 0.1
DEFAULT_OUTPUT = (
    ARTIFACT_DIR / "h800_generic_admission_head_leave_mechanism_out_v1.json"
)

FEATURE_NAMES = (
    "log_reserved_center_over_safe_limit",
    "log_allocated_center_over_safe_limit",
    "is_lora",
    "gradient_checkpointing",
    "zero2",
    "zero3",
    "log2_gpu_count",
    "log2_mbs",
)


def _mechanism(record: Mapping[str, Any]) -> tuple[str, int, bool, int]:
    selector = record["selector"]
    scenario = record["scenario"]
    return (
        str(selector["training_mode"]),
        int(selector.get("zero_stage") or 0),
        bool(selector.get("gradient_checkpointing")),
        int(scenario.get("gpu_count") or 0),
    )


def _label(mechanism: tuple[str, int, bool, int]) -> str:
    mode, zero, gc, gpus = mechanism
    return (
        f"{mode.upper()} {gpus}卡 "
        f"{'不切分' if zero == 0 else f'ZeRO-{zero}'} "
        f"检查点{'开' if gc else '关'}"
    )


def _features(
    record: Mapping[str, Any], allocated: float, reserved: float, limit: float
) -> list[float]:
    """Pressure ratios plus the mechanism itself, mirroring the center model."""
    selector = record["selector"]
    scenario = record["scenario"]
    zero = int(selector.get("zero_stage") or 0)
    return [
        math.log(reserved / limit),
        math.log(allocated / limit),
        1.0 if selector["training_mode"] == "lora" else 0.0,
        1.0 if selector.get("gradient_checkpointing") else 0.0,
        1.0 if zero == 2 else 0.0,
        1.0 if zero == 3 else 0.0,
        math.log2(max(1, int(scenario.get("gpu_count") or 1))),
        math.log2(max(1, int(scenario.get("physical_mbs") or 1))),
    ]


def _fit_logistic(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Source-balanced, class-balanced ridge logistic.

    Same shape as the per-mechanism fitter, except the ridge penalty is sized
    from the data instead of a fixed two-feature constant -- that constant is why
    the production fitter cannot accept mechanism features at all.
    """
    labels = np.asarray([int(s["unsafe_label"]) for s in samples], dtype=float)
    if set(labels.tolist()) != {0.0, 1.0}:
        return None
    matrix = np.asarray([s["features"] for s in samples], dtype=float)
    counts = collections.Counter(str(s["source_id"]) for s in samples)
    weights = np.asarray(
        [1.0 / counts[str(s["source_id"])] for s in samples], dtype=float
    )
    for label in (0.0, 1.0):
        mask = labels == label
        total = float(weights[mask].sum())
        if total > 0:
            weights[mask] *= 0.5 / total
    means = np.average(matrix, axis=0, weights=weights)
    scales = np.sqrt(np.average((matrix - means) ** 2, axis=0, weights=weights))
    scales[scales < 1e-9] = 1.0
    design = np.column_stack((np.ones(len(samples)), (matrix - means) / scales))
    beta = np.zeros(design.shape[1], dtype=float)
    penalty = np.diag([0.0, *([ALPHA] * matrix.shape[1])])
    for _ in range(100):
        linear = np.clip(design @ beta, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-linear))
        variance = np.maximum(probs * (1.0 - probs), 1e-6)
        gradient = design.T @ (weights * (probs - labels)) + penalty @ beta
        hessian = design.T @ ((weights * variance)[:, None] * design) + penalty
        step = np.linalg.pinv(hessian) @ gradient
        beta -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return {"means": means, "scales": scales, "beta": beta}


def _risk(features: Sequence[float], model: Mapping[str, Any]) -> float:
    row = (np.asarray(features, dtype=float) - model["means"]) / model["scales"]
    linear = float(np.concatenate(([1.0], row)) @ model["beta"])
    return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, linear)))))


def _samples(records: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any]):
    out = []
    for record in records:
        try:
            allocated = _predict_center(record, bundle["allocated_model"])
            reserved = _predict_center(record, bundle["reserved_model"])
        except Exception:  # noqa: BLE001
            continue
        limit = float(
            (record.get("memory") or {}).get("safe_limit_bytes") or SAFE_LIMIT_BYTES
        )
        out.append(
            {
                "features": _features(record, allocated, reserved, limit),
                "unsafe_label": 1 if M._is_negative(record) else 0,
                "source_id": _source_id(record),
                "allocated": allocated,
                "reserved": reserved,
                "limit": limit,
            }
        )
    return out


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    mechanisms = sorted({_mechanism(r) for r in records})
    results = []
    for held in mechanisms:
        evaluation = [r for r in records if _mechanism(r) == held]
        training = [r for r in records if _mechanism(r) != held]
        entry: dict[str, Any] = {
            "withheld_mechanism": {
                "training_mode": held[0],
                "zero_stage": held[1],
                "gradient_checkpointing": held[2],
                "gpu_count": held[3],
                "label_cn": _label(held),
            },
            "withheld_observations": len(evaluation),
            "withheld_safe": sum(1 for r in evaluation if not M._is_negative(r)),
            "withheld_failing": sum(1 for r in evaluation if M._is_negative(r)),
        }
        if len(evaluation) < 3:
            entry.update({"runnable": False, "reason": "fewer than three observations"})
            results.append(entry)
            continue
        bundle = _fit_bundle(
            training,
            names=VARIANT_FEATURES[VARIANT_M1],
            coverage=COVERAGE,
            diagnostic_max_fallback=True,
        )
        train_samples = _samples(training, bundle)
        model = _fit_logistic(train_samples)
        if model is None:
            entry.update({"runnable": False, "reason": "training set is single-class"})
            results.append(entry)
            continue
        # Threshold from leave-one-source-out on the TRAINING mechanisms only;
        # the withheld mechanism never informs it.
        sources = sorted({s["source_id"] for s in train_samples})
        scored: list[tuple[float, int]] = []
        for source in sources:
            inner = [s for s in train_samples if s["source_id"] != source]
            inner_model = _fit_logistic(inner)
            if inner_model is None:
                continue
            scored.extend(
                (_risk(s["features"], inner_model), s["unsafe_label"])
                for s in train_samples
                if s["source_id"] == source
            )
        unsafe = [risk for risk, label in scored if label == 1]
        if not unsafe:
            entry.update(
                {"runnable": False, "reason": "no out-of-fold failing sample"}
            )
            results.append(entry)
            continue
        threshold = min(unsafe)
        admitted_safe = admitted_failing = safe_total = failing_total = 0
        for s in _samples(evaluation, bundle):
            admit = bool(
                _risk(s["features"], model) < threshold
                and s["allocated"] <= s["limit"]
                and s["reserved"] <= s["limit"]
            )
            if s["unsafe_label"]:
                failing_total += 1
                admitted_failing += int(admit)
            else:
                safe_total += 1
                admitted_safe += int(admit)
        entry.update(
            {
                "runnable": True,
                "threshold": threshold,
                "training_mechanisms": len({_mechanism(r) for r in training}),
                "safe_total": safe_total,
                "admitted_safe": admitted_safe,
                "safe_admission_recall": (
                    admitted_safe / safe_total if safe_total else None
                ),
                "failing_total": failing_total,
                "admitted_failing": admitted_failing,
                "passed": admitted_failing == 0,
            }
        )
        results.append(entry)
    runnable = [r for r in results if r.get("runnable")]
    return {
        "schema": SCHEMA,
        "analysis_only": True,
        "model_refit": False,
        "publishable": False,
        "question": (
            "can one generic admission head with mechanism features replace the "
            "current per-mechanism coefficient lookup, without admitting failures "
            "on mechanisms it has never seen?"
        ),
        "feature_names": list(FEATURE_NAMES),
        "protocol": (
            "withhold one mechanism entirely; fit centers, generic head and "
            "threshold on the remaining mechanisms; score the withheld one"
        ),
        "fit_records": len(records),
        "mechanisms_tested": len(runnable),
        "mechanisms_passed": sum(1 for r in runnable if r["passed"]),
        "total_admitted_failing": sum(r["admitted_failing"] for r in runnable),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    records, provenance = M.load_records(
        M.DEFAULT_OBSERVATIONS, M.DEFAULT_PROFILE_DIR
    )
    records = [r for r in records if not r["selector"].get("packing")]
    report = evaluate(records)
    report["provenance"] = provenance
    report["report_sha256"] = sha256_json(
        {k: v for k, v in report.items() if k != "report_sha256"}
    )
    print(f"记录 {report['fit_records']}  测试机制 {report['mechanisms_tested']}")
    print(
        f"{'留出的机制':26s}{'样本':>5s}{'安全放行':>10s}{'失败放行':>9s}  判定"
    )
    for row in report["results"]:
        label = row["withheld_mechanism"]["label_cn"]
        if not row.get("runnable"):
            print(f"{label:26s}{row['withheld_observations']:>5d}  跳过:{row['reason']}")
            continue
        recall = (
            f"{row['admitted_safe']}/{row['safe_total']}"
            if row["safe_total"]
            else "无安全样本"
        )
        print(
            f"{label:26s}{row['withheld_observations']:>5d}{recall:>10s}"
            f"{row['admitted_failing']:>9d}  {'通过' if row['passed'] else '错放!'}"
        )
    print(
        f"\n通过 {report['mechanisms_passed']}/{report['mechanisms_tested']}，"
        f"累计错误放行 {report['total_admitted_failing']}"
    )
    write_json(args.output, report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
