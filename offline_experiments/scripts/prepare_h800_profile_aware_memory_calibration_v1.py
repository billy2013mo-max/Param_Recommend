#!/usr/bin/env python3
"""Prepare, but never execute, the profile-aware H800 memory calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analyze_h800_fresh_memory_residual_v2 import profile_padding_statistics
from common import ARTIFACT_DIR, sha256_file, sha256_json, stable_id, write_json


DEFAULT_DESIGN = ARTIFACT_DIR / "h800_fresh_holdout_design_v2.json"
DEFAULT_ACCEPTANCE = ARTIFACT_DIR / "h800_fresh_holdout_acceptance_v2.json"
DEFAULT_DIAGNOSIS = ARTIFACT_DIR / "h800_fresh_memory_residual_diagnosis_v2.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_profile_aware_memory_calibration_design_v1.json"
SCHEMA = "sft_h800_profile_aware_memory_calibration_design/v1"
CAMPAIGN_ID = "h800_profile_aware_memory_calibration_20260802_v1"
TARGET_GBS = 64
MODEL_ID = "qwen3_8b"
TRAIN_TYPE = "lora"
CUTOFF_LEN = 4096


def _configuration_grid() -> list[dict[str, Any]]:
    grid = []
    # The exact failing selector gets three independent attempts per profile.
    for repeat in range(3):
        grid.append(
            {
                "purpose": "main_reservation_risk_slice_repeat",
                "gpu_count": 2,
                "zero_stage": 2,
                "mbs": 2,
                "gc": False,
                "repeat": repeat,
            }
        )
    # One MBS control and one single-GPU control isolate the interactions.
    grid.extend(
        (
            {
                "purpose": "mbs_control",
                "gpu_count": 2,
                "zero_stage": 2,
                "mbs": 1,
                "gc": False,
                "repeat": 0,
            },
            {
                "purpose": "gpu_zero_control",
                "gpu_count": 1,
                "zero_stage": 0,
                "mbs": 2,
                "gc": False,
                "repeat": 0,
            },
        )
    )
    return grid


def build_design(
    *,
    source_design_path: Path,
    acceptance_path: Path,
    diagnosis_path: Path,
) -> dict[str, Any]:
    source_design = json.loads(source_design_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    if acceptance["memory"]["passes"] is not False:
        raise ValueError("targeted calibration is only justified after the frozen memory gate fails")
    if diagnosis["summary"]["upper_miss_count"] != 1:
        raise ValueError("the v1 design is bound to exactly one diagnosed upper miss")

    scenarios = []
    slots = []
    for source in source_design["scenarios"]:
        profile_path = Path(source["profile_path"])
        data_path = Path(source["data_path"])
        if sha256_file(profile_path) != source["profile_sha256"]:
            raise ValueError(f"profile changed: {profile_path}")
        if sha256_file(data_path) != source["data_sha256"]:
            raise ValueError(f"data changed: {data_path}")
        profile_1 = profile_padding_statistics(
            profile_path,
            cutoff_len=CUTOFF_LEN,
            physical_mbs=1,
        )
        profile_2 = profile_padding_statistics(
            profile_path,
            cutoff_len=CUTOFF_LEN,
            physical_mbs=2,
        )
        calibration_scenario_id = (
            f"{source['dataset_id']}__{MODEL_ID}_{TRAIN_TYPE}_cutoff{CUTOFF_LEN}"
        )
        scenarios.append(
            {
                "scenario_id": calibration_scenario_id,
                "split_unit_id": source["dataset_id"],
                "business_scene": source["business_scene"],
                "dataset_id": source["dataset_id"],
                "dataset_category": source["dataset_category"],
                "data_path": str(data_path.resolve()),
                "data_sha256": source["data_sha256"],
                "profile_path": str(profile_path.resolve()),
                "profile_sha256": source["profile_sha256"],
                "model_id": MODEL_ID,
                "train_type": TRAIN_TYPE,
                "cutoff_len": CUTOFF_LEN,
                "target_gbs": TARGET_GBS,
                "profile_padding_statistics": {
                    "mbs1": profile_1,
                    "mbs2": profile_2,
                },
            }
        )
        for configuration in _configuration_grid():
            gpu_count = int(configuration["gpu_count"])
            mbs = int(configuration["mbs"])
            denominator = gpu_count * mbs
            if TARGET_GBS % denominator:
                raise ValueError("target GBS is not divisible by gpu_count * mbs")
            material = {
                "campaign_id": CAMPAIGN_ID,
                "scenario_id": calibration_scenario_id,
                "gpu_count": gpu_count,
                "zero_stage": int(configuration["zero_stage"]),
                "mbs": mbs,
                "gc": bool(configuration["gc"]),
                "repeat": int(configuration["repeat"]),
            }
            slots.append(
                {
                    "slot_id": stable_id("h800memprof", material),
                    **material,
                    "purpose": configuration["purpose"],
                    "model_id": MODEL_ID,
                    "train_type": TRAIN_TYPE,
                    "dataset_id": source["dataset_id"],
                    "dataset_category": source["dataset_category"],
                    "data_path": str(data_path.resolve()),
                    "data_sha256": source["data_sha256"],
                    "dataset_profile_path": str(profile_path.resolve()),
                    "dataset_profile_sha256": source["profile_sha256"],
                    "cutoff_len": CUTOFF_LEN,
                    "target_gbs": TARGET_GBS,
                    "gradient_accumulation_steps": TARGET_GBS // denominator,
                    "zero": (
                        "none"
                        if int(configuration["zero_stage"]) == 0
                        else f"zero{int(configuration['zero_stage'])}"
                    ),
                    "gradient_checkpointing": bool(configuration["gc"]),
                    "packing": False,
                    "offload": False,
                    "fidelity": "formal_3plus10",
                    "warmup_steps": 3,
                    "measure_steps": 10,
                    "calibration_partition": {
                        "role": "calibration",
                        "split_unit_id": source["dataset_id"],
                        "policy": "profile_padding_stratified_scenario_disjoint_v1",
                    },
                    "expected_padding_pressure": (
                        profile_1 if mbs == 1 else profile_2
                    ),
                }
            )

    if len(scenarios) != 4 or len(slots) != 20:
        raise ValueError(
            f"expected four profile scenarios and 20 slots, got {len(scenarios)} and {len(slots)}"
        )
    if len({slot["slot_id"] for slot in slots}) != len(slots):
        raise ValueError("calibration slot ids are not unique")
    ordered = sorted(
        scenarios,
        key=lambda row: row["profile_padding_statistics"]["mbs2"][
            "expected_random_batch_max_tokens"
        ],
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": diagnosis["generated_at_utc"],
        "timestamp_semantics": "inherits deterministic trigger-evidence completion time",
        "status": "design_only_waiting_for_exact_approval",
        "gpu_training_started": False,
        "queues_mutated": False,
        "execution_authorized": False,
        "publication_allowed": False,
        "objective": (
            "Calibrate the selector- and profile-dependent CUDA reservation multiplier while "
            "preserving the already accurate physical allocated-memory head."
        ),
        "trigger": {
            "frozen_memory_gate_passed": acceptance["memory"]["passes"],
            "upper_miss_count": diagnosis["summary"]["upper_miss_count"],
            "allocated_center_mape": diagnosis["summary"]["allocated_center"]["mape"],
            "reserved_center_mape": diagnosis["summary"]["reserved_center"]["mape"],
            "maximum_reserved_to_allocated_ratio": diagnosis["summary"][
                "observed_reserved_to_allocated"
            ]["maximum"],
        },
        "source_bindings": {
            "fresh_holdout_design": {
                "path": str(source_design_path.resolve()),
                "sha256": sha256_file(source_design_path),
            },
            "acceptance_report": {
                "path": str(acceptance_path.resolve()),
                "sha256": sha256_file(acceptance_path),
            },
            "residual_diagnosis": {
                "path": str(diagnosis_path.resolve()),
                "sha256": sha256_file(diagnosis_path),
            },
        },
        "design": {
            "scenario_count": len(scenarios),
            "job_count": len(slots),
            "total_gpu_job_equivalents": sum(int(slot["gpu_count"]) for slot in slots),
            "main_slice_repeats_per_profile": 3,
            "profiles_ordered_by_expected_mbs2_padding_pressure": [
                {
                    "dataset_id": row["dataset_id"],
                    "expected_batch_max_tokens": row["profile_padding_statistics"]["mbs2"][
                        "expected_random_batch_max_tokens"
                    ],
                    "fraction_of_cutoff": row["profile_padding_statistics"]["mbs2"][
                        "expected_random_batch_max_fraction_of_cutoff"
                    ],
                }
                for row in ordered
            ],
            "factor_isolation": [
                "four data length distributions at the same cutoff/model/selector",
                "MBS 1 versus 2 at two GPUs and ZeRO-2",
                "one GPU ZeRO-0 versus two GPU ZeRO-2 at MBS 2",
                "three exact repeats of the diagnosed risk slice per profile",
            ],
        },
        "scenarios": scenarios,
        "candidate_slots": slots,
        "candidate_model_contract": {
            "name": "allocated_physical_plus_profile_aware_reservation_guard_v1",
            "allocated_center_source": "existing frozen physical-shares allocated diagnostic head",
            "reservation_target": "log(max_reserved_bytes / max_allocated_bytes)",
            "profile_feature_budget": [
                "expected_random_batch_max_fraction_of_cutoff",
                "p99_clipped_tokens / cutoff_len",
                "coefficient_of_variation",
                "truncation_fraction",
            ],
            "selector_interaction_budget": [
                "lora x zero2 x gc_off",
                "log2(mbs)",
                "log2(gpu_count)",
            ],
            "maximum_new_feature_terms": 7,
            "selection_protocol": (
                "scenario-disjoint nested CV; compare no-profile, padding-pressure-only and "
                "bounded selector-interaction variants"
            ),
            "upper_rule": (
                "allocated_center * exp(reservation_center + scenario-disjoint conformal "
                "log-residual upper); retain exact-selector OOM right-censor guard"
            ),
        },
        "governance": {
            "existing_memory_artifact_may_be_overwritten": False,
            "new_versioned_challenger_required": True,
            "current_24_row_holdout_may_enter_training": False,
            "current_24_row_holdout_may_validate_the_challenger": False,
            "candidate_slots_must_carry_job_bound_calibration_partition": True,
            "all_exact_attempts_must_have_complete_execution_fingerprints": True,
            "ooms_are_right_censored_not_dropped": True,
            "promotion_requires_a_different_unseen_profile_holdout": True,
            "anchor_registry_may_not_be_used_to_patch_a_base_admitted_underprediction": True,
        },
        "post_calibration_holdout": {
            "required": True,
            "minimum_unseen_profiles": 2,
            "estimated_jobs": "10-12",
            "profile_ids": "must be selected and frozen before challenger coefficients are inspected",
            "minimum_acceptance": {
                "false_safe_oom": 0,
                "scenario_equal_p05_upper_coverage": 0.95,
                "reserved_center_mean_absolute_percentage_error": 0.06,
                "reserved_center_p90_absolute_percentage_error": 0.12,
                "no_supported_slice_safety_regression": True,
            },
        },
        "next_step": (
            "review this 20-job design, select an idle exact-H800 pool, then materialize a new "
            "queue and exact approval; do not reuse the completed holdout approval"
        ),
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    design = build_design(
        source_design_path=DEFAULT_DESIGN,
        acceptance_path=DEFAULT_ACCEPTANCE,
        diagnosis_path=DEFAULT_DIAGNOSIS,
    )
    write_json(DEFAULT_OUTPUT, design)
    print(
        json.dumps(
            {
                "output": str(DEFAULT_OUTPUT),
                "jobs": design["design"]["job_count"],
                "gpu_training_started": design["gpu_training_started"],
                "execution_authorized": design["execution_authorized"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
