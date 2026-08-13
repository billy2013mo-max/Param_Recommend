#!/usr/bin/env python3
"""Freeze predictions and the exact ten-job unseen-profile H800 holdout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from common import (
    ARTIFACT_DIR,
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
)
from h800_profile_aware_v4b_predictor import H800ProfileAwareV4BPredictor
from prepare_h800_prospective_holdout import probe_hardware


SCHEMA = "sft_h800_final_unseen_holdout_design/v1"
CAMPAIGN_ID = "h800_profile_aware_memory_final_holdout_20260802_v1"
PHASE_ID = "h800_profile_aware_memory_final_holdout_v1"
DEFAULT_SELECTION = ARTIFACT_DIR / "h800_final_unseen_holdout_selection_v1.json"
DEFAULT_BUNDLE = ARTIFACT_DIR / "h800_final_unseen_business_data_v1.json"
DEFAULT_CHALLENGER = ARTIFACT_DIR / "h800_profile_aware_memory_challenger_v1.json"
DEFAULT_PREDICTIONS = (
    ARTIFACT_DIR / "h800_frozen_predictions_before_final_unseen_holdout_v1.json"
)
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_final_unseen_holdout_design_v1.json"
PROFILE_DIR = ARTIFACT_DIR / "final_unseen_holdout_v1" / "profiles"
REQUIRED_GPU_IDS = (4, 5)


def _binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _model_map() -> dict[str, Mapping[str, Any]]:
    inventory = read_json(ARTIFACT_DIR / "model_inventory.json")
    return {str(row["id"]): row for row in inventory["models"]}


def _category(profile: Mapping[str, Any]) -> str:
    return (
        "short"
        if profile["distribution_stratum"] == "short concentrated"
        else "longcontext"
    )


def _request(
    profile: Mapping[str, Any],
    configuration: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    gpu_count = int(configuration["gpu_count"])
    mbs = int(configuration["mbs"])
    target_gbs = 64
    if target_gbs % (gpu_count * mbs):
        raise ValueError("target GBS is not divisible by gpu_count * mbs")
    material = {
        "campaign_id": CAMPAIGN_ID,
        "profile_id": profile["profile_id"],
        **dict(configuration),
    }
    return {
        "request_id": stable_id("finalunseenpred", material),
        "comparison_group": (
            f"{profile['profile_id']}__{configuration['model_id']}_"
            f"{configuration['train_type']}"
        ),
        "model_id": configuration["model_id"],
        "training_mode": configuration["train_type"],
        "dataset_id": profile["dataset_id"],
        "dataset_category": _category(profile),
        "target_gbs": target_gbs,
        "cutoff_len": int(configuration["cutoff_len"]),
        "actual_parameters": int(model["actual_parameters"]),
        "gpu_count": gpu_count,
        "physical_mbs": mbs,
        "gradient_accumulation_steps": target_gbs // (gpu_count * mbs),
        "zero_stage": int(configuration["zero_stage"]),
        "gradient_checkpointing": bool(configuration["gc"]),
        "packing": False,
        "offload": False,
        "dtype": "bf16",
        "kernel_path": "fa3_orig+liger_fused_ce+adamw_torch_fused",
        "lora_rank": 32,
        "profile_tokenizer_id": "qwen3_shared_local",
        "profile_template_id": "qwen3_nothink",
    }


def build_design(
    *,
    selection_path: Path,
    bundle_path: Path,
    challenger_path: Path,
    prediction_path: Path,
) -> dict[str, Any]:
    selection = read_json(selection_path)
    bundle = read_json(bundle_path)
    challenger = read_json(challenger_path)
    if (
        selection.get("schema") != "sft_h800_final_unseen_holdout_selection/v1"
        or selection.get("campaign_id") != CAMPAIGN_ID
        or selection.get("frozen_before_challenger_coefficient_inspection") is not True
    ):
        raise ValueError("frozen unseen selection contract mismatch")
    if (
        bundle.get("schema") != "sft_h800_final_unseen_business_data/v1"
        or bundle.get("campaign_id") != CAMPAIGN_ID
        or bundle.get("model_fitting_performed") is not False
    ):
        raise ValueError("unseen business data bundle contract mismatch")
    if (
        challenger.get("schema") != "sft_h800_profile_aware_memory_challenger/v1"
        or challenger.get("status")
        != "frozen_shadow_candidate_requires_unseen_profile_holdout"
    ):
        raise ValueError("profile-aware challenger is not frozen for holdout")

    models = _model_map()
    requests: list[dict[str, Any]] = []
    for profile in bundle["profiles"]:
        for configuration in selection["frozen_candidate_policy"]["configurations"]:
            requests.append(
                _request(
                    profile,
                    configuration,
                    models[str(configuration["model_id"])],
                )
            )
    if len(requests) != 10 or len({row["request_id"] for row in requests}) != 10:
        raise ValueError("the frozen candidate policy must yield ten unique requests")
    predictor = H800ProfileAwareV4BPredictor(
        challenger_artifact=challenger_path,
        additional_dataset_profile_dir=PROFILE_DIR,
    )
    predictions = predictor.predict(requests)
    write_json(prediction_path, predictions)
    prediction_by_request = {
        str(row["request_id"]): row for row in predictions["predictions"]
    }
    bundle_by_profile = {
        str(row["profile_id"]): row for row in bundle["profiles"]
    }

    slots: list[dict[str, Any]] = []
    for request in requests:
        prediction = prediction_by_request[str(request["request_id"])]
        profile = bundle_by_profile[str(request["dataset_id"])]
        slot_material = {
            "campaign_id": CAMPAIGN_ID,
            "request_id": request["request_id"],
        }
        memory = prediction["memory"]
        slots.append(
            {
                "candidate_slot_id": stable_id("finalunseenslot", slot_material),
                "predictor_request_id": request["request_id"],
                "comparison_group": request["comparison_group"],
                "profile_id": profile["profile_id"],
                "dataset_id": profile["dataset_id"],
                "data_path": profile["data_path"],
                "data_sha256": profile["data_sha256"],
                "dataset_profile_path": profile["profile_path"],
                "dataset_profile_sha256": profile["profile_sha256"],
                "dataset_category": request["dataset_category"],
                "model_id": request["model_id"],
                "train_type": request["training_mode"],
                "cutoff_len": request["cutoff_len"],
                "target_gbs": request["target_gbs"],
                "gpu_count": request["gpu_count"],
                "zero_stage": request["zero_stage"],
                "gc": request["gradient_checkpointing"],
                "mbs": request["physical_mbs"],
                "gradient_accumulation_steps": request[
                    "gradient_accumulation_steps"
                ],
                "packing": False,
                "offload": False,
                "calibration_partition": {
                    "role": "holdout",
                    "split_unit_id": profile["profile_id"],
                    "policy": "prospective_unseen_dense_profiles_v1",
                },
                "frozen_prediction": {
                    "prediction_available": memory["prediction_available"],
                    "allocated_anchor_bytes": memory["allocated_anchor_bytes"],
                    "allocated_center_bytes": memory["allocated_center_bytes"],
                    "reserved_center_bytes": memory["reserved_center_bytes"],
                    "operational_upper_reserved_bytes": memory[
                        "operational_p95_reserved_bytes"
                    ],
                    "safe_limit_bytes": memory["safe_limit_bytes"],
                    "predicted_safe": memory["base_physical_model_admitted"],
                    "tail_bucket": memory["tail_bucket"],
                    "padding_statistics": memory["padding_statistics"],
                    "v4b_prediction_available": prediction["throughput"][
                        "prediction_available"
                    ],
                    "v4b_rank_within_gpu_count": prediction[
                        "rank_within_gpu_count"
                    ],
                    "v4b_throughput_proxy": prediction["throughput"].get(
                        "throughput_proxy_tokens_per_second"
                    ),
                },
            }
        )
    if len(slots) != 10 or len({row["candidate_slot_id"] for row in slots}) != 10:
        raise ValueError("holdout design did not produce ten unique slots")

    source_files = {
        "selection": selection_path,
        "business_data_bundle": bundle_path,
        "challenger": challenger_path,
        "frozen_predictions": prediction_path,
        "experiment_config": ROOT / "config" / "experiment.json",
        "hardware_config": ROOT / "config" / "hardware.json",
        "dataset_registry": ROOT / "data" / "dataset_info.json",
        "model_inventory": ARTIFACT_DIR / "model_inventory.json",
        "old_memory_anchor": ARTIFACT_DIR / "h800_challenger_modeling.json",
        "throughput_model": ARTIFACT_DIR / "joint_throughput_modeling.json",
        "profile_memory_math": ROOT / "scripts" / "h800_profile_aware_memory_model.py",
        "profile_predictor": ROOT / "scripts" / "h800_profile_aware_v4b_predictor.py",
        "data_preparer": ROOT / "scripts" / "prepare_h800_final_unseen_data_v1.py",
        "design_preparer": Path(__file__).resolve(),
        "materializer": ROOT / "scripts" / "materialize_h800_final_unseen_holdout_v1.py",
        "approval_freezer": ROOT / "scripts" / "freeze_h800_final_unseen_holdout_v1.py",
        "run_job": ROOT / "scripts" / "run_job.py",
        "scheduler": ROOT / "scripts" / "scheduler.py",
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "predictions_frozen_waiting_for_queue_and_exact_approval",
        "gpu_training_started": False,
        "queues_mutated": False,
        "execution_authorized": False,
        "publication_allowed": False,
        "objective": (
            "Prospectively evaluate the frozen profile-aware H800 memory challenger "
            "on two unseen Dense SFT business profiles."
        ),
        "required_gpu_pool": {
            "gpu_ids": list(REQUIRED_GPU_IDS),
            "expected_name": "NVIDIA H800",
            "max_gpu_count": 2,
            "preemption_allowed": False,
            "join_busy_pool_allowed": True,
            "hardware_probe_at_design": probe_hardware(
                required_gpu_ids=REQUIRED_GPU_IDS
            ),
        },
        "source_bindings": {
            name: _binding(path) for name, path in sorted(source_files.items())
        },
        "frozen_prediction_binding": {
            "path": str(prediction_path.resolve()),
            "sha256": sha256_file(prediction_path),
            "report_sha256": predictions["report_sha256"],
            "generated_before_gpu": True,
        },
        "profiles": bundle["profiles"],
        "candidate_slots": slots,
        "design_summary": {
            "profile_count": 2,
            "job_count": 10,
            "one_gpu_jobs": sum(row["gpu_count"] == 1 for row in slots),
            "two_gpu_jobs": sum(row["gpu_count"] == 2 for row in slots),
            "predicted_safe_jobs": sum(
                row["frozen_prediction"]["predicted_safe"] is True for row in slots
            ),
            "predicted_unsafe_jobs": sum(
                row["frozen_prediction"]["predicted_safe"] is False for row in slots
            ),
            "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in slots),
        },
        "acceptance_contract": selection["minimum_acceptance"],
        "governance": {
            "all_frozen_configurations_run_even_if_predicted_unsafe": True,
            "holdout_rows_may_enter_fit": False,
            "challenger_coefficients_may_change_after_gpu_start": False,
            "old_memory_artifact_may_be_overwritten": False,
            "historical_anchor_override_allowed": False,
            "new_exact_approval_required": True,
        },
        "next_step": (
            "materialize the exact ten-job queue, capture provenance after all "
            "source/config changes, freeze and promote a new GPU-4,5 approval"
        ),
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--challenger", type=Path, default=DEFAULT_CHALLENGER)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_design(
        selection_path=args.selection,
        bundle_path=args.bundle,
        challenger_path=args.challenger,
        prediction_path=args.predictions,
    )
    write_json(args.output, report)
    print(
        {
            "design": str(args.output.resolve()),
            "report_sha256": report["report_sha256"],
            **report["design_summary"],
        }
    )


if __name__ == "__main__":
    main()
