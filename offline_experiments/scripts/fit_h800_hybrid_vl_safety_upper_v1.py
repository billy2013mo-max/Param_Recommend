#!/usr/bin/env python3
"""Fit one-sided H800 safety multipliers for hybrid memory and frozen VL.

The center models stay unchanged.  Exact rows contribute out-of-fold log
underprediction residuals.  OOM rows remain right-censored lower bounds and
can only increase the guard.  This script does not grant production release;
that requires a separately frozen prospective acceptance report.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from fit_h800_hybrid_attention_dense_stage1_v1 import (
    _collapsed_rows,
    _fit_route,
    _predict,
)
from h800_resource_predictor import H800ResourcePredictor


SCHEMA = "sft_h800_hybrid_vl_safety_upper/v1"
TARGET_COVERAGE = 0.95
CAPACITY_BYTES = 150142189568
SAFE_LIMIT_FRACTION = 0.95
SAFE_LIMIT_BYTES = CAPACITY_BYTES * SAFE_LIMIT_FRACTION

HYBRID_OBSERVATIONS = (
    ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_observations_v1.jsonl"
)
HYBRID_CENTER_ARTIFACT = ARTIFACT_DIR / "h800_hybrid_memory_artifact_v1.json"
VL_EVALUATION = ARTIFACT_DIR / "h800_frozen_vl_modeling_formal_evaluation_v1.json"
VL_CENTER_ARTIFACT = ARTIFACT_DIR / "h800_frozen_vl_residual_overlay_v1.json"
VL_PROFILE_DIR = ARTIFACT_DIR / "h800_vl_business_workload_profiles_v2"
OUTPUT = ARTIFACT_DIR / "h800_hybrid_vl_safety_upper_v1.json"


def _finite_sample_upper(values: Sequence[float], coverage: float) -> dict[str, Any]:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        raise ValueError("one-sided calibration requires exact residuals")
    rank = math.ceil((len(clean) + 1) * coverage)
    selected_rank = min(rank, len(clean))
    quantile = clean[selected_rank - 1]
    return {
        "exact_rows": len(clean),
        "target_coverage": coverage,
        "finite_sample_rank": rank,
        "selected_rank": selected_rank,
        "rank_capped_at_sample_maximum": rank > len(clean),
        "exact_log_residual_quantile": quantile,
    }


def _guard(
    exact_log_residuals: Sequence[float],
    censored_lower_log_residuals: Sequence[float],
) -> dict[str, Any]:
    result = _finite_sample_upper(exact_log_residuals, TARGET_COVERAGE)
    censor_guard = max(
        (float(value) for value in censored_lower_log_residuals),
        default=-math.inf,
    )
    selected = max(0.0, float(result["exact_log_residual_quantile"]), censor_guard)
    result.update(
        {
            "censored_rows": len(censored_lower_log_residuals),
            "censored_lower_log_residual_max": (
                censor_guard if math.isfinite(censor_guard) else None
            ),
            "selected_log_guard": selected,
            "upper_multiplier": math.exp(selected),
            "form": "upper_bytes = center_bytes * exp(selected_log_guard)",
        }
    )
    return result


def _hybrid_calibration() -> dict[str, Any]:
    observations = read_jsonl(HYBRID_OBSERVATIONS)
    terminal = [
        row
        for row in observations
        if row.get("architecture_route") == "dense_hybrid_attention"
        and (row.get("outcome") or {}).get("classification") in {"success", "oom"}
        and (row.get("outcome") or {}).get("terminal_eligible")
    ]
    rows = _collapsed_rows(terminal)
    sources = sorted({str(row["source_dataset_id"]) for row in rows})
    exact_residuals: list[float] = []
    censor_residuals: list[float] = []
    predictions: list[dict[str, Any]] = []
    for held_source in sources:
        train = [row for row in rows if str(row["source_dataset_id"]) != held_source]
        test = [row for row in rows if str(row["source_dataset_id"]) == held_source]
        if not train or not test:
            raise ValueError(f"hybrid source fold is empty: {held_source}")
        coefficients = np.asarray(_fit_route(train)["coefficients"], dtype=float)
        for row in test:
            center = max(_predict(coefficients, row["design"]), 1.0)
            if row["state"] == "exact":
                target = float(row["peak_reserved_bytes"])
                residual = math.log(target / center)
                exact_residuals.append(residual)
                target_name = "peak_reserved_bytes"
            else:
                target = float(row["censor_lower_bytes"])
                residual = math.log(target / center)
                censor_residuals.append(residual)
                target_name = "censor_lower_bytes"
            predictions.append(
                {
                    "held_out_source": held_source,
                    "model_id": row["model_id"],
                    "state": row["state"],
                    "center_bytes": center,
                    target_name: target,
                    "log_residual_or_lower_bound": residual,
                }
            )
    guard = _guard(exact_residuals, censor_residuals)
    multiplier = float(guard["upper_multiplier"])
    exact_covered = sum(
        float(row["center_bytes"]) * multiplier
        >= float(row["peak_reserved_bytes"])
        for row in predictions
        if row["state"] == "exact"
    )
    censor_satisfied = sum(
        float(row["center_bytes"]) * multiplier
        >= float(row["censor_lower_bytes"])
        for row in predictions
        if row["state"] == "censored"
    )
    guard["oof_exact_coverage"] = exact_covered / len(exact_residuals)
    guard["oof_censor_lower_bound_satisfaction"] = (
        censor_satisfied / len(censor_residuals) if censor_residuals else None
    )
    guard["oof_predictions"] = predictions
    return guard


def _vl_profile_path(pair: Mapping[str, Any]) -> Path:
    source = "pzfj38" if pair["track"] == "frozen_vision_decomposition" else "zltbjg"
    return VL_PROFILE_DIR / f"{pair['model_id']}.{source}.{pair['media_tier']}.json"


def _vl_request(pair: Mapping[str, Any], profile_path: Path) -> dict[str, Any]:
    return {
        "request_id": str(pair["pair_unit_id"]),
        "comparison_group": str(pair["pair_unit_id"]),
        "hardware_id": "h800",
        "model_id": str(pair["model_id"]),
        "training_mode": "lora",
        "lora_rank": 32,
        "target_gbs": 64,
        "cutoff_len": 8192,
        "gpu_count": 1,
        "physical_mbs": int(pair["physical_mbs"]),
        "zero_stage": 0,
        "gradient_checkpointing": bool(pair["gradient_checkpointing"]),
        "packing": False,
        "dtype": "bf16",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "vl_workload_profile_path": str(profile_path.resolve()),
    }


def _vl_calibration() -> dict[str, Any]:
    evaluation = read_json(VL_EVALUATION)
    exact_pairs = [
        row
        for row in evaluation["paired_units"]
        if row["memory_observation_kind"] == "paired_exact_success_centers"
    ]
    censored_pairs = list(evaluation["right_censored_pairs"])
    cv_by_id = {
        str(row["pair_unit_id"]): row
        for row in evaluation["memory_cross_validation_predictions"]
    }
    predictor = H800ResourcePredictor()
    requests = [
        _vl_request(pair, _vl_profile_path(pair))
        for pair in [*exact_pairs, *censored_pairs]
    ]
    report = predictor.predict(requests)
    deployed_by_id = {str(row["request_id"]): row for row in report["predictions"]}

    raw_text_centers = {
        pair_id: float(row["vl_shadow"]["memory"]["text_center_bytes"])
        for pair_id, row in deployed_by_id.items()
    }
    base_ratios: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for pair in exact_pairs:
        pair_id = str(pair["pair_unit_id"])
        mechanism = str(pair["mechanism_id"])
        key = (str(pair["model_id"]), mechanism)
        ratio = float(pair["control"]["max_reserved_bytes"]) / raw_text_centers[pair_id]
        base_ratios.setdefault(key, []).append((pair_id, ratio))
    final_base_scales = {
        f"{model_id}::{mechanism}": statistics.median(
            ratio for _, ratio in values
        )
        for (model_id, mechanism), values in sorted(base_ratios.items())
    }

    def base_scale(pair: Mapping[str, Any], *, exclude_pair_id: str | None) -> float:
        key = (str(pair["model_id"]), str(pair["mechanism_id"]))
        same_mechanism = [
            ratio
            for pair_id, ratio in base_ratios.get(key, [])
            if pair_id != exclude_pair_id
        ]
        if same_mechanism:
            return statistics.median(same_mechanism)
        same_model = [
            ratio
            for (model_id, _), values in base_ratios.items()
            if model_id == str(pair["model_id"])
            for pair_id, ratio in values
            if pair_id != exclude_pair_id
        ]
        if not same_model:
            raise ValueError(f"VL base scale has no training rows: {pair['pair_unit_id']}")
        return statistics.median(same_model)

    by_modality: dict[str, dict[str, list[float] | list[dict[str, Any]]]] = {
        "image": {"exact": [], "censored": [], "rows": []},
        "video": {"exact": [], "censored": [], "rows": []},
    }
    for pair in exact_pairs:
        pair_id = str(pair["pair_unit_id"])
        modality = "image" if pair["track"] == "frozen_vision_decomposition" else "video"
        deployed = deployed_by_id[pair_id]
        raw_text_center = float(
            deployed["vl_shadow"]["memory"]["text_center_bytes"]
        )
        scale = base_scale(pair, exclude_pair_id=pair_id)
        text_center = raw_text_center * scale
        visual_oof = float(cv_by_id[pair_id]["predicted_visual_delta_bytes"])
        center = text_center + visual_oof
        actual = float(pair["real_media"]["max_reserved_bytes"])
        residual = math.log(actual / center)
        by_modality[modality]["exact"].append(residual)
        by_modality[modality]["rows"].append(
            {
                "pair_unit_id": pair_id,
                "model_id": pair["model_id"],
                "state": "exact",
                "raw_text_center_bytes": raw_text_center,
                "text_base_scale_oof": scale,
                "text_center_bytes": text_center,
                "oof_visual_residual_bytes": visual_oof,
                "center_bytes": center,
                "peak_reserved_bytes": actual,
                "log_residual": residual,
            }
        )
    for pair in censored_pairs:
        pair_id = str(pair["pair_unit_id"])
        modality = "image" if pair["track"] == "frozen_vision_decomposition" else "video"
        deployed = deployed_by_id[pair_id]
        raw_text_center = float(
            deployed["vl_shadow"]["memory"]["text_center_bytes"]
        )
        scale = base_scale(pair, exclude_pair_id=None)
        visual_delta = float(deployed["vl_shadow"]["memory"]["visual_residual_bytes"])
        center = raw_text_center * scale + visual_delta
        lower = float(pair["right_censor_lower_bytes"])
        residual = math.log(lower / center)
        by_modality[modality]["censored"].append(residual)
        by_modality[modality]["rows"].append(
            {
                "pair_unit_id": pair_id,
                "model_id": pair["model_id"],
                "state": "right_censored",
                "raw_text_center_bytes": raw_text_center,
                "text_base_scale": scale,
                "visual_residual_bytes": visual_delta,
                "center_bytes": center,
                "censor_lower_bytes": lower,
                "log_residual_lower_bound": residual,
            }
        )

    result: dict[str, Any] = {
        "text_base_scale_by_model_mechanism": final_base_scales,
        "text_base_scale_form": (
            "corrected_text_center_bytes = raw_text_center_bytes * "
            "median(control_peak / raw_text_center) by model and mechanism"
        ),
        "text_base_scale_oof_policy": (
            "leave-one-pair-out within model and mechanism; model-level median fallback"
        ),
    }
    for modality, values in by_modality.items():
        exact = list(values["exact"])
        censored = list(values["censored"])
        guard = _guard(exact, censored)
        multiplier = float(guard["upper_multiplier"])
        rows = list(values["rows"])
        exact_rows = [row for row in rows if row["state"] == "exact"]
        censored_rows = [row for row in rows if row["state"] == "right_censored"]
        guard["oof_exact_coverage"] = sum(
            float(row["center_bytes"]) * multiplier
            >= float(row["peak_reserved_bytes"])
            for row in exact_rows
        ) / len(exact_rows)
        guard["oof_censor_lower_bound_satisfaction"] = (
            sum(
                float(row["center_bytes"]) * multiplier
                >= float(row["censor_lower_bytes"])
                for row in censored_rows
            )
            / len(censored_rows)
            if censored_rows
            else None
        )
        guard["oof_predictions"] = rows
        result[modality] = guard
    return result


def fit() -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_coverage": TARGET_COVERAGE,
        "safe_limit_fraction": SAFE_LIMIT_FRACTION,
        "safe_limit_bytes": SAFE_LIMIT_BYTES,
        "hybrid_memory": _hybrid_calibration(),
        "vl_memory_by_modality": _vl_calibration(),
        "source_bindings": {
            "hybrid_observations": {
                "path": str(HYBRID_OBSERVATIONS.resolve()),
                "sha256": sha256_file(HYBRID_OBSERVATIONS),
            },
            "hybrid_center_artifact": {
                "path": str(HYBRID_CENTER_ARTIFACT.resolve()),
                "sha256": sha256_file(HYBRID_CENTER_ARTIFACT),
            },
            "vl_evaluation": {
                "path": str(VL_EVALUATION.resolve()),
                "sha256": sha256_file(VL_EVALUATION),
            },
            "vl_center_artifact": {
                "path": str(VL_CENTER_ARTIFACT.resolve()),
                "sha256": sha256_file(VL_CENTER_ARTIFACT),
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
        },
        "release_contract": {
            "mode": "candidate_pending_prospective_acceptance",
            "automatic_admission_allowed": False,
            "automatic_ranking_allowed": False,
        },
    }
    artifact["report_sha256"] = sha256_json(artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    artifact = fit()
    write_json(args.output, artifact)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
