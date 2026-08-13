#!/usr/bin/env python3
"""Extend separate admission to the supported H800 FULL mechanism.

V5 keeps the V4 Critical-LoRA head and adds a distinct FULL head for the
mechanism covered by the strict holdouts: FULL, ZeRO-3, gradient checkpointing,
two GPUs, and no packing.  Other FULL mechanisms fail closed instead of using
the known-broken stacked S0 admission rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from architecture_domain_guard import observation_architecture_refusals
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
    _is_critical_lora,
    _mechanism_key,
    _safe_limit,
    _source_id,
)
from fit_h800_m1_separate_admission_v4 import (
    ADMISSION_FEATURE_NAMES,
    _admission_features,
    _fit_admission_head,
    _is_bounded_full,
    _is_lora_zero2_no_gc_four_gpu,
    _predict_risk,
)


S5_LORA_FULL_ADMISSION = "S5_lora_full_separate_admission_heads"


def _stacked_expansion_available(
    record: Mapping[str, Any], bundle: Mapping[str, Any]
) -> bool:
    """True when the S0 allocated-tail x expansion-tail product would apply.

    The stacked upper is only inert because most mechanisms have no fitted
    ``reservation_expansion`` entry.  That is a property of the current data,
    not a structural guarantee, so callers must fail closed rather than admit
    silently once an entry becomes available.
    """
    entry = (
        (bundle.get("reservation_expansion") or {}).get("entries") or {}
    ).get(_mechanism_key(record))
    return bool(isinstance(entry, Mapping) and entry.get("available") is True)


def _is_full(record: Mapping[str, Any]) -> bool:
    return (record.get("selector") or {}).get("training_mode") == "full"


def _head_prediction(
    record: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    head_key: str,
    scope: str,
    hard_center_guard: bool = False,
) -> dict[str, Any]:
    prediction = dict(
        _predict_safety(record, bundle, safety_variant=S2_RESERVED_OOM)
    )
    prediction["safety_variant"] = S5_LORA_FULL_ADMISSION
    head = bundle.get(head_key) or {}
    if prediction.get("available") is not True or not head:
        prediction.update(
            {
                "admission_available": False,
                "admission_scope": scope,
                "admission_issues": [f"{head_key}_unavailable"],
            }
        )
        return prediction
    features = _admission_features(
        allocated_center=float(prediction["allocated_center_bytes"]),
        reserved_center=float(prediction["reserved_center_bytes"]),
        safe_limit=float(_safe_limit(record)),
    )
    risk = _predict_risk(features, head["model"])
    threshold = float(head["threshold"])
    safe_limit = float(_safe_limit(record))
    center_within_safe_limit = bool(
        float(prediction["allocated_center_bytes"]) <= safe_limit
        and float(prediction["reserved_center_bytes"]) <= safe_limit
    )
    prediction.update(
        {
            "admission_available": True,
            "admission_scope": scope,
            "admission_features": dict(zip(ADMISSION_FEATURE_NAMES, features)),
            "unsafe_risk": risk,
            "admission_threshold": threshold,
            "hard_center_guard": hard_center_guard,
            "center_within_safe_limit": center_within_safe_limit,
            "admitted_by_separate_head": bool(
                risk < threshold
                and (center_within_safe_limit or not hard_center_guard)
            ),
            "admission_issues": [],
        }
    )
    return prediction


def _is_architecture_outside_v5_scope(record: Mapping[str, Any]) -> bool:
    """True when the checkpoint's architecture is outside V5's calibrated domain.

    V5 routes purely on *mechanism* (training mode, ZeRO stage, gradient
    checkpointing, GPU count, packing) and has no architecture dimension at all.
    Every one of its heads is fitted on dense, text-only, uniform-softmax-
    attention Qwen3 checkpoints, so a vision-language or hybrid-attention
    request is priced by a skeleton that does not describe it -- and the S0
    fallback below would then admit it on the known-defective stacked upper.

    Refusing here is a strictly monotone reduction in admissions: it can only
    turn admit into refuse, never the reverse, so it cannot manufacture a false
    admission.
    """
    return bool(observation_architecture_refusals(record))


def _architecture_refusal_prediction(
    record: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    """Non-stacked capacity upper for diagnostics, with admission refused.

    Mirrors the ``_is_full`` out-of-mechanism branch: keep a usable number for
    reporting, but never admit on it.  S2 rather than S0 deliberately -- the
    stacked product is the defect V5 exists to remove.
    """
    prediction = dict(_predict_safety(record, bundle, safety_variant=S2_RESERVED_OOM))
    prediction.update(
        {
            "safety_variant": S5_LORA_FULL_ADMISSION,
            "admission_available": True,
            "admission_scope": "architecture_outside_v5_scope_fail_closed",
            "unsafe_risk": None,
            "admission_threshold": None,
            "admitted_by_separate_head": False,
            "admission_issues": [
                code for code, _ in observation_architecture_refusals(record)
            ],
        }
    )
    return prediction


def _predict_v5(record: Mapping[str, Any], bundle: Mapping[str, Any]) -> dict[str, Any]:
    if _is_architecture_outside_v5_scope(record):
        # Must precede every mechanism branch: a VL LoRA job at ZeRO-2 with GC
        # off on two GPUs satisfies _is_critical_lora exactly, so a
        # mechanism-first order would admit it on a text-only head.
        return _architecture_refusal_prediction(record, bundle)
    if _is_critical_lora(record):
        return _head_prediction(
            record,
            bundle,
            head_key="separate_admission_head",
            scope="critical_lora",
        )
    if _is_lora_zero2_no_gc_four_gpu(record):
        return _head_prediction(
            record,
            bundle,
            head_key="lora_zero2_no_gc_four_gpu_admission_head",
            scope="lora_zero2_no_gc_four_gpu",
            hard_center_guard=True,
        )
    if _is_bounded_full(record):
        return _head_prediction(
            record,
            bundle,
            head_key="full_admission_head",
            scope="full_zero3_gc_two_gpu",
            hard_center_guard=True,
        )
    if _is_full(record):
        # Preserve a direct, non-stacked capacity upper for diagnostics, but do
        # not admit outside the supported FULL mechanism.
        prediction = dict(
            _predict_safety(record, bundle, safety_variant=S2_RESERVED_OOM)
        )
        prediction.update(
            {
                "safety_variant": S5_LORA_FULL_ADMISSION,
                "admission_available": True,
                "admission_scope": "full_outside_supported_mechanism_fail_closed",
                "unsafe_risk": None,
                "admission_threshold": None,
                "admitted_by_separate_head": False,
                "admission_issues": ["full_mechanism_outside_v5_scope"],
            }
        )
        return prediction
    fallback = dict(_predict_safety(record, bundle, safety_variant=S0_STACKED))
    # Mechanisms without their own head still route through S0.  That rule
    # multiplies the allocated tail by the reservation-expansion tail, which is
    # the defect V5 removed on the head-backed paths.  It is currently inert
    # only because these mechanisms have no fitted expansion entry, so refuse
    # to admit if one ever appears instead of reviving the stacked product.
    stacked_live = _stacked_expansion_available(record, bundle)
    issues = ["outside_v5_scope_kept_s0"]
    if stacked_live:
        issues.append("stacked_expansion_upper_active_fail_closed")
    fallback.update(
        {
            "safety_variant": S5_LORA_FULL_ADMISSION,
            "admission_available": fallback.get("available") is True,
            "admission_scope": "outside_lora_and_full_keep_current_s0",
            "unsafe_risk": None,
            "admission_threshold": None,
            "stacked_expansion_available": stacked_live,
            "admitted_by_separate_head": False if stacked_live else None,
            "admission_issues": issues,
        }
    )
    return fallback


def _v5_detail(
    record: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    prediction = _predict_v5(record, bundle)
    detail = _detail(
        record,
        prediction,
        safety_variant=S5_LORA_FULL_ADMISSION,
    )
    detail["memory_upper_would_admit"] = detail["admitted"]
    if _is_architecture_outside_v5_scope(record):
        # Load-bearing: without this branch the architecture refusal in
        # _predict_v5 would be cosmetic.  ``head_backed`` below is false for an
        # out-of-domain architecture, so ``detail["admitted"]`` would keep
        # whatever the memory upper decided and the row would be admitted anyway.
        detail["admitted"] = False
    else:
        head_backed = (
            _is_critical_lora(record)
            or _is_lora_zero2_no_gc_four_gpu(record)
            or _is_full(record)
        )
        if head_backed:
            detail["admitted"] = bool(
                prediction.get("admission_available") is True
                and prediction.get("admitted_by_separate_head") is True
            )
        elif prediction.get("stacked_expansion_available"):
            # S0 path with a live stacked product: refuse instead of admitting on
            # the known-defective upper.
            detail["admitted"] = False
    selector = record.get("selector") or {}
    detail.update(
        {
            "admission_scope": prediction.get("admission_scope"),
            "unsafe_risk": prediction.get("unsafe_risk"),
            "admission_threshold": prediction.get("admission_threshold"),
            "hard_center_guard": prediction.get("hard_center_guard"),
            "center_within_safe_limit": prediction.get(
                "center_within_safe_limit"
            ),
            "stacked_expansion_available": prediction.get(
                "stacked_expansion_available"
            ),
            "admission_features": dict(
                prediction.get("admission_features") or {}
            ),
            "admission_issues": list(prediction.get("admission_issues") or []),
            "zero_stage": selector.get("zero_stage"),
            "gradient_checkpointing": bool(
                selector.get("gradient_checkpointing")
            ),
            "packing": bool(selector.get("packing")),
            "supported_full": _is_bounded_full(record),
            "supported_lora_four_gpu": _is_lora_zero2_no_gc_four_gpu(record),
        }
    )
    return detail


def _fit_v5_heads(
    records: Sequence[Mapping[str, Any]], bundle: dict[str, Any]
) -> None:
    bundle["separate_admission_head"] = _fit_admission_head(
        records,
        bundle,
        scope="critical_lora",
        classifier_oof_threshold=True,
    )
    bundle["lora_zero2_no_gc_four_gpu_admission_head"] = _fit_admission_head(
        records,
        bundle,
        scope="lora_zero2_no_gc_four_gpu",
        classifier_oof_threshold=True,
    )
    bundle["full_admission_head"] = _fit_admission_head(
        records,
        bundle,
        scope="bounded_full",
        classifier_oof_threshold=False,
    )


def _nested_v5(
    records: Sequence[Mapping[str, Any]],
    *,
    coverage: float,
) -> dict[str, Any]:
    sources = sorted({_source_id(row) for row in records})
    details: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for index, held in enumerate(sources, start=1):
        print(
            f"V5 outer fold {index}/{len(sources)} holdout={held}",
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
        _fit_v5_heads(training, bundle)
        details.extend(_v5_detail(row, bundle) for row in evaluation)
        folds.append(
            {
                "held_source_id": held,
                "training_sources": len({_source_id(row) for row in training}),
                "evaluation_configurations": len(evaluation),
            }
        )
    return {
        "protocol": (
            "outer leave split_unit_id out; LoRA and supported FULL heads are "
            "refitted inside every outer training fold; other FULL mechanisms "
            "fail closed"
        ),
        "folds": folds,
        "details": details,
    }
