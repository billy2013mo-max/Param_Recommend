#!/usr/bin/env python3
"""Fit a separate Critical-LoRA admission head for H800 M1.

The memory upper bound and the binary admission decision answer different
questions.  This module keeps the direct reserved-memory upper bound for
capacity reporting, but does not multiply its tail by a second allocator
expansion tail.  Admission is instead learned as a source-balanced binary risk
problem from two normalized point estimates:

* log(reserved_center / safe_limit)
* log(allocated_center / safe_limit)

Successes above the safe limit and OOMs are unsafe labels.  OOM remains a
classification/censoring constraint and is never converted to an exact peak
memory target.  The admission threshold is calibrated only from nested
leave-source-out risk scores and is chosen to maximize safe-success recall
subject to zero admitted unsafe observations in that calibration evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from calibrate_h800_m1_safety_upper_v2 import (
    S0_STACKED,
    S2_RESERVED_OOM,
    _detail,
    _predict_safety,
)
from fit_h800_lora_source_disjoint_recalibration_v1 import (
    VARIANT_FEATURES,
    VARIANT_M1,
    _fit_bundle,
    _fit_ridge,
    _is_critical_lora,
    _observed_reserved,
    _outcome,
    _predict_center,
    _safe_limit,
    _source_id,
)


S4_SEPARATE_ADMISSION = "S4_direct_reserved_upper_separate_admission_head"
ADMISSION_FEATURE_NAMES = (
    "log_reserved_center_over_safe_limit",
    "log_allocated_center_over_safe_limit",
)
DEFAULT_LOGISTIC_ALPHA = 0.1


def _is_bounded_full(record: Mapping[str, Any]) -> bool:
    selector = record.get("selector") or {}
    scenario = record.get("scenario") or {}
    return bool(
        selector.get("training_mode") == "full"
        and int(selector.get("zero_stage") or 0) == 3
        and bool(selector.get("gradient_checkpointing"))
        and int(scenario.get("gpu_count") or 0) == 2
        and not bool(selector.get("packing"))
    )


def _is_lora_zero2_no_gc_four_gpu(record: Mapping[str, Any]) -> bool:
    """LoRA, ZeRO-2, gradient checkpointing off, four GPUs, no packing.

    The second LoRA mechanism with enough independent negative-boundary
    evidence to carry its own admission head.  Only consumed by V5 and later;
    V4 never requests this scope, so its behaviour is unchanged.
    """
    selector = record.get("selector") or {}
    scenario = record.get("scenario") or {}
    return bool(
        selector.get("training_mode") == "lora"
        and int(selector.get("zero_stage") or 0) == 2
        and not bool(selector.get("gradient_checkpointing"))
        and int(scenario.get("gpu_count") or 0) == 4
        and not bool(selector.get("packing"))
    )


def _mechanism_predicate(
    *, training_mode: str, zero_stage: int, gradient_checkpointing: bool, gpu_count: int
):
    """Build an exact unpacked-mechanism predicate.

    Every admission head is scoped to one (training mode, ZeRO stage, gradient
    checkpointing, GPU count) combination with packing off.  Writing these by
    hand once per mechanism invited copy-paste drift, so newer scopes are
    generated instead.  ``packing`` is always excluded: the V5 length policy
    only holds when packing is off.
    """

    def predicate(record: Mapping[str, Any]) -> bool:
        selector = record.get("selector") or {}
        scenario = record.get("scenario") or {}
        return bool(
            selector.get("training_mode") == training_mode
            and int(selector.get("zero_stage") or 0) == zero_stage
            and bool(selector.get("gradient_checkpointing")) is gradient_checkpointing
            and int(scenario.get("gpu_count") or 0) == gpu_count
            and not bool(selector.get("packing"))
        )

    return predicate


# Mechanisms whose independent-source and negative-boundary evidence already
# meets the planning thresholds (>=5 sources, >=2 of them carrying an OOM or an
# over-the-line success), so each can carry its own head with no new GPU work.
GENERATED_SCOPES: dict[str, dict[str, Any]] = {
    "lora_zero0_no_gc_one_gpu": {
        "training_mode": "lora",
        "zero_stage": 0,
        "gradient_checkpointing": False,
        "gpu_count": 1,
    },
    "full_zero0_gc_one_gpu": {
        "training_mode": "full",
        "zero_stage": 0,
        "gradient_checkpointing": True,
        "gpu_count": 1,
    },
    "full_zero3_gc_four_gpu": {
        "training_mode": "full",
        "zero_stage": 3,
        "gradient_checkpointing": True,
        "gpu_count": 4,
    },
}
_GENERATED_PREDICATES = {
    name: _mechanism_predicate(**spec) for name, spec in GENERATED_SCOPES.items()
}


def _scope_predicate(scope: str):
    if scope == "critical_lora":
        return _is_critical_lora
    if scope == "bounded_full":
        return _is_bounded_full
    if scope == "lora_zero2_no_gc_four_gpu":
        return _is_lora_zero2_no_gc_four_gpu
    if scope in _GENERATED_PREDICATES:
        return _GENERATED_PREDICATES[scope]
    raise ValueError(f"unsupported admission-head scope: {scope}")


def _unsafe_label(record: Mapping[str, Any]) -> int:
    outcome = _outcome(record)
    if outcome == "oom":
        return 1
    if outcome != "success":
        raise ValueError(f"unsupported admission outcome: {outcome}")
    reserved = _observed_reserved(record)
    safe = _safe_limit(record)
    if reserved is None or safe is None:
        raise ValueError("success admission label requires reserved peak and safe limit")
    return int(float(reserved) > float(safe))


def _admission_features(
    *, allocated_center: float, reserved_center: float, safe_limit: float
) -> list[float]:
    values = (allocated_center, reserved_center, safe_limit)
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in values):
        raise ValueError("admission features require positive finite byte values")
    return [
        math.log(float(reserved_center) / float(safe_limit)),
        math.log(float(allocated_center) / float(safe_limit)),
    ]


def _fit_logistic(
    samples: Sequence[Mapping[str, Any]],
    *,
    alpha: float = DEFAULT_LOGISTIC_ALPHA,
) -> dict[str, Any]:
    if alpha <= 0:
        raise ValueError("logistic alpha must be positive")
    if not samples:
        raise ValueError("admission logistic has no samples")
    labels = np.asarray([int(row["unsafe_label"]) for row in samples], dtype=float)
    if set(labels.tolist()) != {0.0, 1.0}:
        raise ValueError("admission logistic requires safe and unsafe observations")
    features = np.asarray([row["features"] for row in samples], dtype=float)
    counts = Counter(str(row["source_id"]) for row in samples)
    weights = np.asarray(
        [1.0 / counts[str(row["source_id"])] for row in samples], dtype=float
    )
    # Give safe and unsafe classes equal total influence after source balancing.
    for label in (0.0, 1.0):
        mask = labels == label
        weights[mask] *= 0.5 / float(weights[mask].sum())
    means = np.average(features, axis=0, weights=weights)
    scales = np.sqrt(
        np.average((features - means) ** 2, axis=0, weights=weights)
    )
    scales[scales < 1e-9] = 1.0
    design = np.column_stack((np.ones(len(samples)), (features - means) / scales))
    coefficients = np.zeros(design.shape[1], dtype=float)
    penalty = np.diag([0.0, *([float(alpha)] * len(ADMISSION_FEATURE_NAMES))])
    converged = False
    for iteration in range(1, 101):
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-linear))
        variance = np.maximum(probabilities * (1.0 - probabilities), 1e-6)
        gradient = design.T @ (weights * (probabilities - labels))
        gradient += penalty @ coefficients
        hessian = design.T @ ((weights * variance)[:, None] * design)
        hessian += penalty
        step = np.linalg.pinv(hessian) @ gradient
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-9:
            converged = True
            break
    if not converged:
        raise ValueError("admission logistic did not converge")
    return {
        "model_family": "source_balanced_class_balanced_logistic_ridge",
        "feature_names": list(ADMISSION_FEATURE_NAMES),
        "alpha": float(alpha),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "fit_rows": len(samples),
        "fit_independent_sources": len(counts),
        "fit_safe_rows": int(sum(label == 0.0 for label in labels)),
        "fit_unsafe_rows": int(sum(label == 1.0 for label in labels)),
        "weighting": "source-balanced, then safe/unsafe class-balanced",
        "iterations": iteration,
    }


def _predict_risk(features: Sequence[float], model: Mapping[str, Any]) -> float:
    vector = np.asarray(features, dtype=float)
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    linear = float(model["intercept"]) + float(
        ((vector - means) / scales) @ coefficients
    )
    linear = min(30.0, max(-30.0, linear))
    return 1.0 / (1.0 + math.exp(-linear))


def _center_oof_samples(
    records: Sequence[Mapping[str, Any]],
    base_bundle: Mapping[str, Any],
    *,
    scope: str = "critical_lora",
) -> list[dict[str, Any]]:
    names = list(base_bundle["feature_names"])
    allocated_alpha = float(base_bundle["allocated_model"]["alpha"])
    reserved_alpha = float(base_bundle["reserved_model"]["alpha"])
    predicate = _scope_predicate(scope)
    scoped = [row for row in records if predicate(row)]
    sources = sorted({_source_id(row) for row in scoped})
    samples: list[dict[str, Any]] = []
    for held in sources:
        training = [row for row in records if _source_id(row) != held]
        evaluation = [row for row in scoped if _source_id(row) == held]
        allocated_model = _fit_ridge(
            training,
            names=names,
            alpha=allocated_alpha,
            target="allocated",
        )
        reserved_model = _fit_ridge(
            training,
            names=names,
            alpha=reserved_alpha,
            target="reserved",
        )
        for row in evaluation:
            allocated_center = _predict_center(row, allocated_model)
            reserved_center = _predict_center(row, reserved_model)
            samples.append(
                {
                    "source_id": held,
                    "cluster_id": row.get("cluster_id"),
                    "unsafe_label": _unsafe_label(row),
                    "outcome": _outcome(row),
                    "features": _admission_features(
                        allocated_center=allocated_center,
                        reserved_center=reserved_center,
                        safe_limit=float(_safe_limit(row)),
                    ),
                }
            )
    return samples


def _fit_admission_head(
    records: Sequence[Mapping[str, Any]],
    base_bundle: Mapping[str, Any],
    *,
    alpha: float = DEFAULT_LOGISTIC_ALPHA,
    scope: str = "critical_lora",
    classifier_oof_threshold: bool = True,
) -> dict[str, Any]:
    samples = _center_oof_samples(records, base_bundle, scope=scope)
    sources = sorted({str(row["source_id"]) for row in samples})
    risk_oof: list[dict[str, Any]] = []
    if classifier_oof_threshold:
        for held in sources:
            training = [row for row in samples if row["source_id"] != held]
            evaluation = [row for row in samples if row["source_id"] == held]
            model = _fit_logistic(training, alpha=alpha)
            for row in evaluation:
                risk_oof.append(
                    {
                        **dict(row),
                        "unsafe_risk": _predict_risk(row["features"], model),
                    }
                )
    else:
        calibration_model = _fit_logistic(samples, alpha=alpha)
        risk_oof = [
            {
                **dict(row),
                "unsafe_risk": _predict_risk(row["features"], calibration_model),
            }
            for row in samples
        ]
    unsafe_risks = [
        float(row["unsafe_risk"])
        for row in risk_oof
        if int(row["unsafe_label"]) == 1
    ]
    if not unsafe_risks:
        raise ValueError("admission threshold has no unsafe OOF constraint")
    threshold = min(unsafe_risks)
    safe_rows = [row for row in risk_oof if int(row["unsafe_label"]) == 0]
    unsafe_rows = [row for row in risk_oof if int(row["unsafe_label"]) == 1]
    admitted_safe = sum(float(row["unsafe_risk"]) < threshold for row in safe_rows)
    admitted_unsafe = sum(
        float(row["unsafe_risk"]) < threshold for row in unsafe_rows
    )
    if admitted_unsafe:
        raise ValueError("strict admission threshold admitted an unsafe OOF row")
    return {
        "schema": "sft_h800_m1_separate_admission_head/v4",
        "scope": scope,
        "definition": (
            "binary unsafe-risk head, separate from the direct reserved-memory "
            "upper; OOM is an unsafe classification constraint, never an exact peak"
        ),
        "model": _fit_logistic(samples, alpha=alpha),
        "threshold": float(threshold),
        "admission_rule": "admit iff unsafe_risk < threshold",
        "threshold_calibration": (
            (
                "minimum unsafe risk among leave-source-out classifier predictions"
                if classifier_oof_threshold
                else "minimum unsafe risk among classifier-fit calibration scores"
            )
            + "; maximizes safe recall subject to zero unsafe admissions"
        ),
        "classifier_oof_threshold": classifier_oof_threshold,
        "center_feature_protocol": (
            "allocated and reserved center features are leave-source-out within "
            "the head fit; the final head is trained on those OOF center features"
        ),
        "calibration_oof": {
            "rows": len(risk_oof),
            "independent_sources": len(sources),
            "safe_rows": len(safe_rows),
            "unsafe_rows": len(unsafe_rows),
            "admitted_safe_rows": admitted_safe,
            "safe_recall": admitted_safe / len(safe_rows) if safe_rows else None,
            "admitted_unsafe_rows": admitted_unsafe,
            "risk_min": min(float(row["unsafe_risk"]) for row in risk_oof),
            "risk_max": max(float(row["unsafe_risk"]) for row in risk_oof),
        },
    }


def _predict_separate_admission(
    record: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    # Reporting upper: one direct reserved tail plus the OOM lower-bound guard.
    # The old allocated-tail * allocator-expansion-tail product is absent.
    prediction = dict(
        _predict_safety(record, bundle, safety_variant=S2_RESERVED_OOM)
    )
    prediction["safety_variant"] = S4_SEPARATE_ADMISSION
    head = bundle.get("separate_admission_head") or {}
    if _is_critical_lora(record):
        if prediction.get("available") is not True or not head:
            prediction["admission_available"] = False
            prediction["admission_issues"] = ["separate_admission_head_unavailable"]
            return prediction
        features = _admission_features(
            allocated_center=float(prediction["allocated_center_bytes"]),
            reserved_center=float(prediction["reserved_center_bytes"]),
            safe_limit=float(_safe_limit(record)),
        )
        risk = _predict_risk(features, head["model"])
        threshold = float(head["threshold"])
        prediction.update(
            {
                "admission_available": True,
                "admission_scope": "critical_lora",
                "admission_features": dict(zip(ADMISSION_FEATURE_NAMES, features)),
                "unsafe_risk": risk,
                "admission_threshold": threshold,
                "admitted_by_separate_head": risk < threshold,
                "admission_issues": [],
            }
        )
        return prediction

    # Outside the new head's declared scope, retain the prior S0 behavior.
    fallback = dict(_predict_safety(record, bundle, safety_variant=S0_STACKED))
    fallback.update(
        {
            "safety_variant": S4_SEPARATE_ADMISSION,
            "admission_available": fallback.get("available") is True,
            "admission_scope": "outside_scope_keep_current_s0",
            "unsafe_risk": None,
            "admission_threshold": None,
            "admitted_by_separate_head": None,
            "admission_issues": ["outside_critical_lora_scope_kept_s0"],
        }
    )
    return fallback


def _separate_admission_detail(
    record: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    prediction = _predict_separate_admission(record, bundle)
    detail = _detail(
        record,
        prediction,
        safety_variant=S4_SEPARATE_ADMISSION,
    )
    detail["memory_upper_would_admit"] = detail["admitted"]
    if _is_critical_lora(record):
        detail["admitted"] = bool(
            prediction.get("admission_available") is True
            and prediction.get("admitted_by_separate_head") is True
        )
    detail["admission_scope"] = prediction.get("admission_scope")
    detail["unsafe_risk"] = prediction.get("unsafe_risk")
    detail["admission_threshold"] = prediction.get("admission_threshold")
    detail["admission_features"] = dict(
        prediction.get("admission_features") or {}
    )
    detail["admission_issues"] = list(prediction.get("admission_issues") or [])
    selector = record.get("selector") or {}
    detail["zero_stage"] = selector.get("zero_stage")
    detail["gradient_checkpointing"] = bool(
        selector.get("gradient_checkpointing")
    )
    detail["packing"] = bool(selector.get("packing"))
    return detail


def _nested_separate_admission(
    records: Sequence[Mapping[str, Any]],
    *,
    coverage: float,
) -> dict[str, Any]:
    sources = sorted({_source_id(row) for row in records})
    details: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for index, held in enumerate(sources, start=1):
        print(
            f"separate admission outer fold {index}/{len(sources)} holdout={held}",
            flush=True,
        )
        training = [row for row in records if _source_id(row) != held]
        evaluation = [row for row in records if _source_id(row) == held]
        bundle = _fit_bundle(
            training,
            names=VARIANT_FEATURES[VARIANT_M1],
            coverage=coverage,
            diagnostic_max_fallback=True,
        )
        bundle["separate_admission_head"] = _fit_admission_head(training, bundle)
        details.extend(_separate_admission_detail(row, bundle) for row in evaluation)
        folds.append(
            {
                "held_source_id": held,
                "training_sources": len({_source_id(row) for row in training}),
                "evaluation_configurations": len(evaluation),
            }
        )
    return {
        "protocol": (
            "outer leave split_unit_id out; inside every outer training fold, M1 "
            "centers, direct reserved tail, OOM guard, admission-head center "
            "features, classifier and strict threshold are regenerated without "
            "the outer held source"
        ),
        "folds": folds,
        "details": details,
    }
