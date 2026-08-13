#!/usr/bin/env python3
"""Frozen inference contract for the unified H800 bounded memory candidate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark_h800_memory_center_models_v1 import _predict_correction
from common import read_json, sha256_json

ARTIFACT_SCHEMA_V2 = "sft_h800_unified_bounded_memory_shadow_candidate/v2"
ARTIFACT_SCHEMA_V3 = "sft_h800_unified_bounded_memory_shadow_candidate/v3"
ARTIFACT_SCHEMA_V4 = "sft_h800_unified_bounded_memory_shadow_candidate/v4"
# Backward-compatible alias for the already-frozen v2 builder.
ARTIFACT_SCHEMA = ARTIFACT_SCHEMA_V2


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    schema = artifact.get("schema")
    if schema not in {ARTIFACT_SCHEMA_V2, ARTIFACT_SCHEMA_V3, ARTIFACT_SCHEMA_V4}:
        raise ValueError("unified bounded memory artifact schema mismatch")
    unsigned = dict(artifact)
    expected = unsigned.pop("artifact_sha256", None)
    if not isinstance(expected, str) or expected != sha256_json(unsigned):
        raise ValueError("unified bounded memory artifact checksum mismatch")
    if (
        artifact.get("immutable") is not True
        or artifact.get("publishable") is not False
        or artifact.get("production_override_allowed") is not False
        or artifact.get("status") != "frozen_shadow_candidate_waiting_validation"
    ):
        raise ValueError("unified bounded memory shadow release contract drifted")
    admission = artifact.get("admission") or {}
    multiplier = float(admission.get("upper_multiplier"))
    if not math.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError("unified bounded memory upper multiplier is invalid")
    if schema == ARTIFACT_SCHEMA_V2:
        risk_multiplier = float(admission.get("risk_guard_multiplier"))
        if not math.isfinite(risk_multiplier) or risk_multiplier < 1.0:
            raise ValueError("unified bounded memory risk guard multiplier is invalid")
    else:
        if admission.get("kind") != "independent_shared_risk_head":
            raise ValueError("unified bounded memory shared risk-head contract drifted")
        if not isinstance(admission.get("risk_model"), Mapping):
            raise ValueError("unified bounded memory risk model is unavailable")
    if schema == ARTIFACT_SCHEMA_V4:
        model = artifact.get("model") or {}
        if (
            model.get("kind") != "allocated_plus_positive_allocator_gap"
            or not isinstance(model.get("allocated_model"), Mapping)
            or not isinstance(model.get("allocator_gap_model"), Mapping)
            or model.get("composition")
            != "reserved=allocated*exp(max(0,predicted_log_reserved_over_allocated))"
        ):
            raise ValueError("unified bounded memory v4 dual-head contract drifted")


def load_artifact(path: Path) -> dict[str, Any]:
    artifact = read_json(path)
    validate_artifact(artifact)
    return artifact


def predict_records(
    records: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]
) -> list[dict[str, Any]]:
    validate_artifact(artifact)
    schema = str(artifact["schema"])
    if schema == ARTIFACT_SCHEMA_V4:
        dual_model = artifact["model"]
        allocated_corrections = _predict_correction(
            records, dual_model["allocated_model"]
        )
        gap_corrections = _predict_correction(
            records, dual_model["allocator_gap_model"]
        )
        corrections = None
    else:
        corrections = _predict_correction(records, artifact["model"])
        allocated_corrections = None
        gap_corrections = None
    capacity = float(artifact["hardware_domain"]["capacity_bytes"])
    safe_fraction = float(artifact["admission"]["safe_limit_fraction"])
    multiplier = float(artifact["admission"]["upper_multiplier"])
    if schema in {ARTIFACT_SCHEMA_V3, ARTIFACT_SCHEMA_V4}:
        risk_corrections = _predict_correction(
            records, artifact["admission"]["risk_model"]
        )
    else:
        risk_corrections = None
        risk_multiplier = float(artifact["admission"]["risk_guard_multiplier"])
        shrinkage = float(artifact["model"]["correction_shrinkage"])
        if not 0.0 < shrinkage <= 1.0:
            raise ValueError("unified bounded memory correction shrinkage is invalid")
    safe_limit = capacity * safe_fraction
    predictions = []
    for index, record in enumerate(records):
        if schema == ARTIFACT_SCHEMA_V4:
            assert allocated_corrections is not None
            assert gap_corrections is not None
            allocated_correction = float(allocated_corrections[index])
            predicted_log_ratio = max(0.0, float(gap_corrections[index]))
            allocated_center = float(record["reference_bytes"]) * math.exp(
                allocated_correction
            )
            center = allocated_center * math.exp(predicted_log_ratio)
            allocator_gap = center - allocated_center
        else:
            assert corrections is not None
            correction = float(corrections[index])
            allocated_correction = None
            predicted_log_ratio = None
            allocated_center = None
            allocator_gap = None
            center = float(record["reference_bytes"]) * math.exp(correction)
        if risk_corrections is None:
            risk_guard = float(record["reference_bytes"]) * math.exp(
                float(correction) / shrinkage
            )
            risk_guard_multiplier = risk_multiplier
            upper = max(center * multiplier, risk_guard * risk_guard_multiplier)
        else:
            risk_correction = float(risk_corrections[index])
            risk_guard = float(record["reference_bytes"]) * math.exp(risk_correction)
            risk_guard_multiplier = multiplier
            upper = max(center, risk_guard * risk_guard_multiplier)
        prediction = {
            "available": True,
            "record_id": str(record["record_id"]),
            "reference_bytes": float(record["reference_bytes"]),
            "log_center_correction": (
                math.log(center / float(record["reference_bytes"]))
            ),
            "center_bytes": center,
            "upper_multiplier": multiplier,
            "risk_guard_bytes": risk_guard,
            "risk_guard_multiplier": risk_guard_multiplier,
            "admission_upper_bytes": upper,
            "safe_limit_bytes": safe_limit,
            "admitted": upper <= safe_limit,
            "issues": [],
        }
        if schema == ARTIFACT_SCHEMA_V4:
            prediction.update(
                {
                    "allocated_log_correction": allocated_correction,
                    "allocated_center_bytes": allocated_center,
                    "allocator_gap_log_ratio": predicted_log_ratio,
                    "allocator_gap_bytes": allocator_gap,
                    "reserved_center_bytes": center,
                }
            )
        predictions.append(prediction)
    return predictions
