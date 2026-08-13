"""Canonical data preparation for unified-bounded memory v3 diagnostics."""

from __future__ import annotations

import copy
import math
from typing import Any

import benchmark_h800_memory_center_models_v1 as bm
import fit_h800_unified_resource_partial_v1 as base
from common import ARTIFACT_DIR, MATRIX_DIR, read_json, read_jsonl
from h800_theory_basis import _model_geometry, memory_basis

RUNTIME_BASE_PARAMETERS = {
    "qwen3_1p7b": 1_720_574_976,
}
TARGETED_QUEUE = MATRIX_DIR / "h800_unified_bounded_targeted_jobs_v1.jsonl"
TARGETED_RESULTS = ARTIFACT_DIR / "h800_unified_bounded_targeted_results_v1.json"
HISTORICAL_CONNECTED_SOURCE = "historical_public_connected_component_01"


def _inventory() -> tuple[dict[str, Any], dict[str, dict[str, Any]], int]:
    inventory = read_json(base.DEFAULT_INVENTORY)
    models = {str(row["id"]): dict(row) for row in inventory["models"]}
    capacity = int(read_json(base.DEFAULT_HARDWARE)["memory_bytes_reported_by_torch"])
    return inventory, models, capacity


def correct_runtime_parameter_convention(
    record: dict[str, Any],
    *,
    inventory: dict[str, Any],
    models: dict[str, dict[str, Any]],
    capacity_bytes: int,
) -> dict[str, Any]:
    row = copy.deepcopy(record)
    model_id = str(row["model_id"])
    runtime_parameters = RUNTIME_BASE_PARAMETERS.get(model_id)
    if runtime_parameters is None:
        row["parameter_convention"] = "inventory_matches_runtime"
        return row
    model = dict(models[model_id])
    model["actual_parameters"] = runtime_parameters
    cutoff = int(row["cutoff_len"])
    effective_sequence = max(
        1,
        round(cutoff * math.exp(float(row["features"]["log_effective_fraction"]))),
    )
    job = {
        "model_parameters": runtime_parameters,
        "train_type": str(row["train_type"]),
        "gpu_count": int(row["gpu_count"]),
        "mbs": int(row["mbs"]),
        "cutoff_len": effective_sequence,
        "zero": ("none" if int(row["zero_stage"]) == 0 else f"zero{row['zero_stage']}"),
        "zero_stage": int(row["zero_stage"]),
        "gc": bool(row["gc"]),
    }
    basis = memory_basis(
        job,
        _model_geometry(job, model, inventory["fixed_lora"]),
        capacity_bytes,
    )
    reference = float(basis["analytic_reference_bytes"])
    activation_share = float(basis["structural_activation_bytes"]) / reference
    is_lora = float(row["train_type"] == "lora")
    log_parameters = math.log2(runtime_parameters / 4_022_468_096.0)
    log_mbs = math.log2(float(row["mbs"]))
    features = dict(row["features"])
    features.update(
        {
            "log2_parameters_over_4b": log_parameters,
            "activation_share": activation_share,
            "is_lora_x_activation_share": is_lora * activation_share,
            "is_lora_x_log2_parameters": is_lora * log_parameters,
            "activation_share_x_log2_mbs": activation_share * log_mbs,
        }
    )
    row["reference_bytes"] = reference
    row["features"] = base._complete_feature_values(
        features,
        packing=bool(row["packing"]),
        samples_per_pack=1.0,
    )
    row["parameter_convention"] = "runtime_tied_embedding"
    row["runtime_base_parameters"] = runtime_parameters
    return row


def targeted_records() -> list[dict[str, Any]]:
    inventory, models, capacity = _inventory()
    corrected_models = copy.deepcopy(models)
    for model_id, parameters in RUNTIME_BASE_PARAMETERS.items():
        corrected_models[model_id]["actual_parameters"] = parameters
    jobs = {str(row["job_id"]): row for row in read_jsonl(TARGETED_QUEUE)}
    report = read_json(TARGETED_RESULTS)
    if (
        report.get("status") != "complete"
        or len(report.get("observations") or []) != 15
    ):
        raise ValueError("targeted results are incomplete or drifted")
    profile_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    records = []
    for observation in report["observations"]:
        job = copy.deepcopy(jobs[str(observation["job_id"])])
        if str(job["model_id"]) in RUNTIME_BASE_PARAMETERS:
            job["model_parameters"] = RUNTIME_BASE_PARAMETERS[str(job["model_id"])]
        reference, features = base._current_features(
            job,
            model_by_id=corrected_models,
            fixed_lora=inventory["fixed_lora"],
            capacity_bytes=capacity,
            profile_cache=profile_cache,
        )
        success = observation["classification"] == "success"
        records.append(
            {
                "record_id": f"targeted::{observation['job_id']}",
                # All three datasets belong to the historical content-connected
                # component; retaining that identity avoids pseudo-replication.
                "source_id": HISTORICAL_CONNECTED_SOURCE,
                "origin": "targeted_runtime_20260810",
                "campaign": "h800_unified_bounded_targeted_20260810_v1",
                "role": "targeted_mechanism_fit",
                "state": "exact" if success else "censored",
                "reference_bytes": reference,
                "target_reserved_bytes": (
                    float(observation["max_reserved_bytes"]) if success else None
                ),
                "target_allocated_bytes": (
                    float(observation["max_allocated_bytes"]) if success else None
                ),
                "censor_lower_bytes": float(capacity) if not success else None,
                "features": features,
                "model_id": str(job["model_id"]),
                "train_type": str(job["train_type"]),
                "gpu_count": int(job["gpu_count"]),
                "zero_stage": int(job["zero_stage"]),
                "gc": bool(job["gc"]),
                "mbs": int(job["mbs"]),
                "cutoff_len": int(job["cutoff_len"]),
                "packing": False,
                "profile_sha256": str(job["dataset_profile_sha256"]),
                "parameter_convention": (
                    "runtime_tied_embedding"
                    if str(job["model_id"]) in RUNTIME_BASE_PARAMETERS
                    else "inventory_matches_runtime"
                ),
            }
        )
    return records


def development_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records, _strict, audit = bm._load_records()
    inventory, models, capacity = _inventory()
    corrected = [
        correct_runtime_parameter_convention(
            row,
            inventory=inventory,
            models=models,
            capacity_bytes=capacity,
        )
        for row in records
    ]
    # The v2 freeze was replayed against historical106 before v3 model design.
    # Because that replay failed, these rows are no longer validation evidence:
    # v3 promotes them into development and reserves prospective shadow/canary
    # outcomes for acceptance.
    import validate_h800_unified_bounded_memory_v2 as validation

    historical, historical_audit = validation._historical_records()
    corrected_historical = correct_records(historical)
    for row in corrected_historical:
        row["role"] = "historical_replay_promoted_after_v2_failure"
        row["promotion_reason"] = "v2_locked_replay_failed_before_v3_design"
    targeted = targeted_records()
    combined, collapse = base._collapse_combined(
        [*corrected, *targeted, *corrected_historical]
    )
    return combined, {
        "original": audit,
        "targeted_records": len(targeted),
        "historical_replay_promoted_records": len(corrected_historical),
        "historical_builder": historical_audit,
        "v3_combined": collapse,
        "prospective_acceptance_rows_used": 0,
    }


def correct_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inventory, models, capacity = _inventory()
    return [
        correct_runtime_parameter_convention(
            row,
            inventory=inventory,
            models=models,
            capacity_bytes=capacity,
        )
        for row in records
    ]
