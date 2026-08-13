#!/usr/bin/env python3
"""Freeze predictions for the 21-job bounded-memory v2 fresh holdout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from common import ARTIFACT_DIR, ROOT, read_json, sha256_file, sha256_json, stable_id, write_json
from h800_bounded_v4b_predictor import H800BoundedV4BPredictor
from prepare_h800_prospective_holdout import probe_hardware


SCHEMA = "sft_h800_bounded_memory_v2_fresh_holdout_design/v1"
CAMPAIGN_ID = "h800_bounded_memory_v2_fresh_holdout_20260803_v1"
PHASE_ID = "h800_bounded_memory_v2_fresh_holdout_v1"
DEFAULT_SELECTION = ARTIFACT_DIR / "h800_bounded_memory_v2_fresh_selection_v1.json"
DEFAULT_BUNDLE = ARTIFACT_DIR / "h800_bounded_memory_v2_fresh_data_v1.json"
DEFAULT_Q35_PROFILES = ARTIFACT_DIR / "h800_bounded_memory_v2_qwen35_profiles_v1.json"
DEFAULT_CHALLENGER = ARTIFACT_DIR / "h800_bounded_memory_challenger_v2.json"
DEFAULT_INVENTORY = ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_v1.json"
DEFAULT_Q35_RUNTIME = ARTIFACT_DIR / "h800_qwen35_runtime_contract_v1.json"
DEFAULT_PREDICTIONS = ARTIFACT_DIR / "h800_frozen_predictions_before_bounded_memory_v2_fresh_holdout_v1.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_bounded_memory_v2_fresh_holdout_design_v1.json"
PROFILE_DIR = ARTIFACT_DIR / "bounded_memory_v2_fresh_holdout_v1" / "profiles"
REQUIRED_GPU_IDS = (0, 1)


def _binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _category(profile: Mapping[str, Any]) -> str:
    stratum = str(profile["distribution_stratum"])
    if stratum == "rare-tail short":
        return "short"
    if stratum == "broad non-truncated longtail":
        return "longcontext"
    return "longtail"


def _request(
    *,
    profile: Mapping[str, Any],
    dataset_id: str,
    configuration: Mapping[str, Any],
    model: Mapping[str, Any],
    role: str,
    profile_tokenizer_id: str,
    profile_template_id: str,
) -> dict[str, Any]:
    gpu_count = int(configuration["gpu_count"])
    mbs = int(configuration["mbs"])
    target_gbs = 64
    if target_gbs % (gpu_count * mbs):
        raise ValueError("target GBS is not divisible by gpu_count * mbs")
    material = {
        "campaign_id": CAMPAIGN_ID,
        "profile_id": profile["profile_id"],
        "dataset_id": dataset_id,
        "candidate_role": role,
        **dict(configuration),
    }
    return {
        "request_id": stable_id("boundedv2freshpred", material),
        "comparison_group": f"{profile['profile_id']}__{configuration['model_id']}__{role}",
        "model_id": configuration["model_id"],
        "training_mode": configuration["train_type"],
        "dataset_id": dataset_id,
        "dataset_category": _category(profile),
        "target_gbs": target_gbs,
        "cutoff_len": int(profile["cutoff_len"]),
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
        "profile_tokenizer_id": profile_tokenizer_id,
        "profile_template_id": profile_template_id,
    }


def build_design(
    *,
    selection_path: Path,
    bundle_path: Path,
    q35_profiles_path: Path,
    challenger_path: Path,
    inventory_path: Path,
    q35_runtime_path: Path,
    prediction_path: Path,
    reuse_frozen_predictions: bool = False,
) -> dict[str, Any]:
    selection = read_json(selection_path)
    bundle = read_json(bundle_path)
    q35_profiles = read_json(q35_profiles_path)
    challenger = read_json(challenger_path)
    inventory = read_json(inventory_path)
    q35_runtime = read_json(q35_runtime_path)
    if (
        selection.get("schema") != "sft_h800_bounded_memory_v2_fresh_selection/v1"
        or selection.get("campaign_id") != CAMPAIGN_ID
        or selection.get("selection_frozen_before_gpu_results") is not True
    ):
        raise ValueError("bounded v2 fresh selection contract mismatch")
    if (
        bundle.get("schema") != "sft_h800_bounded_memory_v2_fresh_data/v1"
        or bundle.get("campaign_id") != CAMPAIGN_ID
        or bundle.get("model_fitting_performed") is not False
    ):
        raise ValueError("bounded v2 fresh data bundle contract mismatch")
    if q35_profiles.get("schema") != "sft_h800_bounded_memory_v2_qwen35_profiles/v1":
        raise ValueError("Qwen3.5 profile sidecar contract mismatch")
    unsigned_q35_runtime = dict(q35_runtime)
    expected_q35_runtime_sha = unsigned_q35_runtime.pop("report_sha256", None)
    if (
        q35_runtime.get("schema") != "sft_h800_qwen35_runtime_contract/v1"
        or expected_q35_runtime_sha != sha256_json(unsigned_q35_runtime)
    ):
        raise ValueError("Qwen3.5 runtime contract mismatch")
    if (
        challenger.get("schema") != "sft_h800_bounded_memory_challenger/v2"
        or not str(challenger.get("status") or "").startswith("frozen_post_holdout_repair_candidate")
        or challenger.get("publishable") is not False
    ):
        raise ValueError("bounded memory challenger v2 is not frozen")

    models = {str(row["id"]): row for row in inventory["models"]}
    q35_by_source = {str(row["source_profile_id"]): row for row in q35_profiles["profiles"]}
    policy = selection["frozen_candidate_policy"]
    requests: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for profile in bundle["profiles"]:
        for configuration in policy["base_configurations"]:
            request = _request(
                profile=profile,
                dataset_id=str(profile["dataset_id"]),
                configuration=configuration,
                model=models[str(configuration["model_id"])],
                role="base_selector",
                profile_tokenizer_id="qwen3_shared_local",
                profile_template_id="qwen3_nothink",
            )
            requests.append(request)
            metadata[request["request_id"]] = {"profile": profile, "role": "base_selector"}

        configuration = policy["cross_scale_configuration"]
        q35 = q35_by_source[str(profile["profile_id"])]
        request = _request(
            profile=profile,
            dataset_id=str(q35["dataset_id"]),
            configuration=configuration,
            model=models[str(configuration["model_id"])],
            role="cross_scale_diagnostic",
            profile_tokenizer_id=str(q35["profile_tokenizer_id"]),
            profile_template_id=str(q35["profile_template_id"]),
        )
        requests.append(request)
        metadata[request["request_id"]] = {
            "profile": profile,
            "role": "cross_scale_diagnostic",
            "q35_profile": q35,
        }

        configuration = policy["tail_forced_configuration"]
        request = _request(
            profile=profile,
            dataset_id=str(profile["tail_forced_dataset_id"]),
            configuration=configuration,
            model=models[str(configuration["model_id"])],
            role="tail_forced_safety",
            profile_tokenizer_id="qwen3_shared_local",
            profile_template_id="qwen3_nothink",
        )
        requests.append(request)
        metadata[request["request_id"]] = {"profile": profile, "role": "tail_forced_safety"}

    if len(requests) != 21 or len({row["request_id"] for row in requests}) != 21:
        raise ValueError("frozen candidate policy must yield 21 unique requests")

    if reuse_frozen_predictions:
        predictions = read_json(prediction_path)
        predicted_ids = [str(row["request_id"]) for row in predictions["predictions"]]
        request_ids = [str(row["request_id"]) for row in requests]
        if predicted_ids != request_ids:
            raise ValueError("existing frozen predictions do not exactly match the rebuilt request order")
        unsigned_predictions = dict(predictions)
        expected_predictions_sha = unsigned_predictions.pop("report_sha256", None)
        if expected_predictions_sha != sha256_json(unsigned_predictions):
            raise ValueError("existing frozen prediction checksum is invalid")
    else:
        predictor = H800BoundedV4BPredictor(
            challenger_artifact=challenger_path,
            model_inventory=inventory_path,
            strict_model_inventory_binding=False,
            additional_dataset_profile_dir=PROFILE_DIR,
        )
        predictions = predictor.predict(requests)
        write_json(prediction_path, predictions)
    prediction_by_request = {str(row["request_id"]): row for row in predictions["predictions"]}

    slots: list[dict[str, Any]] = []
    for request in requests:
        meta = metadata[request["request_id"]]
        profile = meta["profile"]
        role = str(meta["role"])
        prediction = prediction_by_request[request["request_id"]]
        if role == "tail_forced_safety":
            data_path = profile["tail_forced_data_path"]
            data_sha256 = profile["tail_forced_data_sha256"]
            profile_path = profile["tail_forced_profile_path"]
            profile_sha256 = profile["tail_forced_profile_sha256"]
        elif role == "cross_scale_diagnostic":
            q35 = meta["q35_profile"]
            data_path = q35["data_path"]
            data_sha256 = q35["data_sha256"]
            profile_path = q35["profile_path"]
            profile_sha256 = q35["profile_sha256"]
        else:
            data_path = profile["data_path"]
            data_sha256 = profile["data_sha256"]
            profile_path = profile["profile_path"]
            profile_sha256 = profile["profile_sha256"]
        memory = prediction["memory"]
        throughput = prediction["throughput"]
        slots.append(
            {
                "candidate_slot_id": stable_id(
                    "boundedv2freshslot",
                    {"campaign_id": CAMPAIGN_ID, "request_id": request["request_id"]},
                ),
                "predictor_request_id": request["request_id"],
                "candidate_role": role,
                "comparison_group": request["comparison_group"],
                "profile_id": profile["profile_id"],
                "dataset_id": request["dataset_id"],
                "data_path": data_path,
                "data_sha256": data_sha256,
                "dataset_profile_path": profile_path,
                "dataset_profile_sha256": profile_sha256,
                "profile_tokenizer_id": request["profile_tokenizer_id"],
                "profile_template_id": request["profile_template_id"],
                "dataset_category": request["dataset_category"],
                "model_id": request["model_id"],
                "train_type": request["training_mode"],
                "cutoff_len": request["cutoff_len"],
                "target_gbs": request["target_gbs"],
                "gpu_count": request["gpu_count"],
                "zero_stage": request["zero_stage"],
                "gc": request["gradient_checkpointing"],
                "mbs": request["physical_mbs"],
                "gradient_accumulation_steps": request["gradient_accumulation_steps"],
                "packing": False,
                "offload": False,
                "calibration_partition": {
                    "role": "holdout",
                    "split_unit_id": profile["profile_id"],
                    "policy": "prospective_complete_s3_source_disjoint_v1",
                },
                "frozen_prediction": {
                    "prediction_available": memory["prediction_available"],
                    "allocated_anchor_bytes": memory["allocated_anchor_bytes"],
                    "allocated_center_bytes": memory["allocated_center_bytes"],
                    "reserved_center_bytes": memory["reserved_center_bytes"],
                    "operational_upper_reserved_bytes": memory["operational_p95_reserved_bytes"],
                    "safe_limit_bytes": memory["safe_limit_bytes"],
                    "predicted_safe": memory["base_physical_model_admitted"],
                    "selector_bucket": memory.get("selector_bucket"),
                    "issues": memory.get("issues") or [],
                    "padding_statistics": memory["padding_statistics"],
                    "v4b_prediction_available": throughput["prediction_available"],
                    "v4b_rank_within_gpu_count": prediction["rank_within_gpu_count"],
                    "v4b_throughput_proxy": throughput.get("throughput_proxy_tokens_per_second"),
                },
            }
        )
    if len(slots) != 21 or len({row["candidate_slot_id"] for row in slots}) != 21:
        raise ValueError("holdout design did not produce 21 unique slots")

    source_files = {
        "selection": selection_path,
        "fresh_data_bundle": bundle_path,
        "qwen35_profiles": q35_profiles_path,
        "challenger": challenger_path,
        "frozen_predictions": prediction_path,
        "experiment_config": ROOT / "config" / "experiment.json",
        "hardware_config": ROOT / "config" / "hardware.json",
        "dataset_registry": ROOT / "data" / "dataset_info.json",
        "transfer_model_inventory": inventory_path,
        "qwen35_runtime_contract": q35_runtime_path,
        "frozen_model_inventory": ARTIFACT_DIR / "model_inventory.json",
        "physical_anchor": ARTIFACT_DIR / "h800_challenger_modeling.json",
        "throughput_model": ARTIFACT_DIR / "joint_throughput_modeling.json",
        "bounded_memory_math": ROOT / "scripts" / "h800_bounded_memory_model.py",
        "bounded_predictor": ROOT / "scripts" / "h800_bounded_v4b_predictor.py",
        "base_predictor": ROOT / "scripts" / "h800_physical_v4b_predictor.py",
        "data_preparer": ROOT / "scripts" / "prepare_h800_bounded_memory_v2_fresh_data_v1.py",
        "qwen35_profile_preparer": ROOT / "scripts" / "prepare_h800_bounded_memory_v2_qwen35_profiles_v1.py",
        "inventory_builder": ROOT / "scripts" / "build_h800_bounded_memory_v2_model_inventory.py",
        "qwen35_runtime_builder": ROOT / "scripts" / "build_h800_qwen35_runtime_contract_v1.py",
        "design_preparer": Path(__file__).resolve(),
        "queue_materializer": ROOT / "scripts" / "materialize_h800_bounded_memory_v2_fresh_holdout_v1.py",
        "approval_freezer": ROOT / "scripts" / "freeze_h800_bounded_memory_v2_fresh_holdout_v1.py",
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
        "prior_software_canary_attempts": [
            {
                "execution_attempt_id": "2b9061c26667558afa7b",
                "job_id": "h800boundedv2fresh-559e6c1f9a95d7fe",
                "classification": "failed",
                "training_steps_observed": 0,
                "failure_stage": "model_configuration_load",
                "failure_reason": "Transformers 4.57.1 did not recognize qwen3_5",
                "memory_or_throughput_measurement_observed": False,
            },
            {
                "execution_attempt_id": "11ce17fb5906feaefd84",
                "job_id": "h800boundedv2fresh-559e6c1f9a95d7fe",
                "classification": "software_failure_then_operator_terminated",
                "training_steps_observed": 0,
                "failure_stage": "first_backward",
                "failure_reason": "TileLang TypeAttr __ffi_repr__ duplicate registration",
                "partial_gpu_telemetry_observed": True,
                "eligible_for_acceptance_or_fit": False,
            },
            {
                "execution_attempt_id": "d80246686730b1a4c74d",
                "job_id": "h800boundedv2fresh-559e6c1f9a95d7fe",
                "classification": "failed",
                "training_steps_observed": 0,
                "failure_stage": "first_backward",
                "failure_reason": "FLA rejected Triton 3.4 on Hopper because it can produce incorrect gated backward results",
                "partial_gpu_telemetry_observed": True,
                "eligible_for_acceptance_or_fit": False,
            },
        ],
        "prediction_file_regenerated_after_attempts": False,
        "queues_mutated": False,
        "execution_authorized": False,
        "publication_allowed": False,
        "objective": "Prospectively evaluate bounded H800 memory challenger v2 on three source-disjoint S3 business profiles.",
        "required_gpu_pool": {
            "gpu_ids": list(REQUIRED_GPU_IDS),
            "expected_name": "NVIDIA H800",
            "max_gpu_count": 2,
            "preemption_allowed": False,
            "join_busy_pool_allowed": True,
            "hardware_probe_at_design": probe_hardware(required_gpu_ids=REQUIRED_GPU_IDS),
        },
        "source_bindings": {name: _binding(path) for name, path in sorted(source_files.items())},
        "frozen_prediction_binding": {
            "path": str(prediction_path.resolve()),
            "sha256": sha256_file(prediction_path),
            "report_sha256": predictions["report_sha256"],
            "generated_before_gpu": True,
        },
        "profiles": bundle["profiles"],
        "qwen35_profiles": q35_profiles["profiles"],
        "candidate_slots": slots,
        "design_summary": {
            "profile_count": 3,
            "job_count": 21,
            "base_selector_jobs": sum(row["candidate_role"] == "base_selector" for row in slots),
            "cross_scale_jobs": sum(row["candidate_role"] == "cross_scale_diagnostic" for row in slots),
            "tail_forced_jobs": sum(row["candidate_role"] == "tail_forced_safety" for row in slots),
            "one_gpu_jobs": sum(row["gpu_count"] == 1 for row in slots),
            "two_gpu_jobs": sum(row["gpu_count"] == 2 for row in slots),
            "predicted_safe_jobs": sum(row["frozen_prediction"]["predicted_safe"] is True for row in slots),
            "predicted_unsafe_jobs": sum(row["frozen_prediction"]["predicted_safe"] is False for row in slots),
            "prediction_unavailable_jobs": sum(row["frozen_prediction"]["prediction_available"] is not True for row in slots),
            "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in slots),
        },
        "acceptance_contract": selection["minimum_acceptance"],
        "governance": {
            "all_frozen_base_and_tail_configurations_run_even_if_predicted_unsafe": True,
            "qwen35_cross_scale_requires_software_canary": True,
            "holdout_sources_may_enter_fit": False,
            "challenger_coefficients_may_change_after_gpu_start": False,
            "current_memory_artifact_may_be_overwritten": False,
            "historical_anchor_override_allowed": False,
            "new_exact_approval_required": True,
        },
        "next_step": "materialize the unchanged 21-job queue, reuse the passed H800 Qwen3.5 software canary, then wait for GPU 0,1 to become idle before running the remaining formal jobs",
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--qwen35-profiles", type=Path, default=DEFAULT_Q35_PROFILES)
    parser.add_argument("--challenger", type=Path, default=DEFAULT_CHALLENGER)
    parser.add_argument("--model-inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--qwen35-runtime", type=Path, default=DEFAULT_Q35_RUNTIME)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--reuse-frozen-predictions",
        action="store_true",
        help="Rebuild runtime/source bindings without rewriting the pre-GPU prediction file.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_design(
        selection_path=args.selection,
        bundle_path=args.bundle,
        q35_profiles_path=args.qwen35_profiles,
        challenger_path=args.challenger,
        inventory_path=args.model_inventory,
        q35_runtime_path=args.qwen35_runtime,
        prediction_path=args.predictions,
        reuse_frozen_predictions=args.reuse_frozen_predictions,
    )
    write_json(args.output, report)
    print({"design": str(args.output.resolve()), "report_sha256": report["report_sha256"], **report["design_summary"]})


if __name__ == "__main__":
    main()
