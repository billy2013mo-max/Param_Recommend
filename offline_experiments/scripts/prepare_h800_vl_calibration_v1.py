#!/usr/bin/env python3
"""Materialize the real-image H800 VL calibration matrix."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    DATA_DIR,
    MATRIX_DIR,
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from inventory_models import inventory_model


SCHEMA = "sft_h800_vl_calibration_design/v1"
JOB_SCHEMA = "sft_h800_vl_calibration_job/v1"
CAMPAIGN_ID = "h800_vl_calibration_20260803_v1"
PHASE_ID = "h800_vl_calibration_v1"
DATA = DATA_DIR / "vl_calibration_v1" / "vl_pzfj38_calibration_v1.jsonl"
MEDIA_MANIFEST = ARTIFACT_DIR / "h800_vl_calibration_media_manifest_v1.json"
PROFILE_MANIFEST = ARTIFACT_DIR / "h800_vl_calibration_processor_profiles_manifest_v1.json"
PROFILE_DIR = ARTIFACT_DIR / "h800_vl_calibration_profiles_v1"
QUEUE = MATRIX_DIR / "h800_vl_calibration_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_vl_calibration_design_v1.json"
PREDICTIONS = ARTIFACT_DIR / "h800_vl_calibration_frozen_predictions_v1.json"
INVENTORY = ARTIFACT_DIR / "h800_vl_calibration_model_inventory_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_vl_calibration_queue_manifest_v1.json"


MODELS = (
    {
        "id": "qwen2p5_vl_7b",
        "nominal_scale_b": 7,
        "path": "/wanqing-models/Qwen2.5-VL-7B-Instruct",
        "tokenizer_path": "/wanqing-models/Qwen2.5-VL-7B-Instruct",
        "family": "qwen2p5_vl",
        "template": "qwen2_vl",
        "train_types": ["lora"],
        "image_min_pixels": 56 * 56,
        "enable_liger_kernel": True,
        "effective_kernel_path": "fa3_orig+liger_fused_ce+adamw_torch_fused",
    },
    {
        "id": "qwen3_vl_8b",
        "nominal_scale_b": 8,
        "path": "/wanqing-models/Qwen3-VL-8B-Instruct",
        "tokenizer_path": "/wanqing-models/Qwen3-VL-8B-Instruct",
        "family": "qwen3_vl",
        "template": "qwen3_vl",
        "train_types": ["lora"],
        "image_min_pixels": 64 * 64,
        "enable_liger_kernel": False,
        "effective_kernel_path": "fa3_orig+native_ce+adamw_torch_fused",
    },
)


CANDIDATES = {
    "C1": {"gpu_count": 1, "zero": "none", "gc": True, "mbs": 1, "ga": 64},
    "C3": {"gpu_count": 2, "zero": "zero3", "gc": True, "mbs": 1, "ga": 32},
    "C4": {"gpu_count": 2, "zero": "zero3", "gc": True, "mbs": 2, "ga": 16},
    "C5": {"gpu_count": 2, "zero": "zero2", "gc": False, "mbs": 1, "ga": 32},
}


def _binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _inventory() -> dict[str, Any]:
    rows = [inventory_model(dict(model)) for model in MODELS]
    report: dict[str, Any] = {
        "schema": "sft_h800_vl_calibration_model_inventory/v1",
        "campaign_id": CAMPAIGN_ID,
        "fixed_lora": {"rank": 32, "alpha": 32, "dropout": 0.0, "target": "all"},
        "models": rows,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(INVENTORY, report)
    return report


def _profiles() -> dict[tuple[str, str], dict[str, Any]]:
    manifest = read_json(PROFILE_MANIFEST)
    if (
        manifest.get("schema") != "sft_h800_vl_calibration_processor_profiles/v1"
        or manifest.get("all_actual_processor_checks_passed") is not True
        or len(manifest.get("profiles") or []) != 4
    ):
        raise ValueError("VL processor profiles are not complete")
    rows = {}
    for binding in manifest["profiles"]:
        path = Path(binding["path"])
        if binding.get("sha256") != sha256_file(path):
            raise ValueError(f"VL profile checksum drifted: {path}")
        profile = read_json(path)
        if (
            profile.get("report_sha256") != binding.get("report_sha256")
            or profile.get("actual_processor_validation", {}).get("all_exact") is not True
            or profile.get("summary", {}).get("records") != 1000
        ):
            raise ValueError(f"VL profile contract drifted: {path}")
        rows[(str(binding["model_id"]), str(binding["tier"]))] = binding
    return rows


def _frozen_unavailable_predictions() -> dict[str, Any]:
    rows = []
    for model in MODELS:
        for tier in ("low", "high"):
            for candidate_id in CANDIDATES:
                rows.append(
                    {
                        "scenario_id": f"pzfj38-{model['id']}-{tier}",
                        "model_id": model["id"],
                        "media_tier": tier,
                        "candidate_id": candidate_id,
                        "memory_prediction": {
                            "available": False,
                            "reason": "real_image_vl_memory_head_not_calibrated",
                        },
                        "throughput_prediction": {
                            "available": False,
                            "reason": "real_image_vl_ranker_not_calibrated",
                        },
                        "automatic_execution_from_prediction": False,
                    }
                )
    report: dict[str, Any] = {
        "schema": "sft_h800_vl_calibration_frozen_predictions/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_before_gpu": True,
        "prediction_rows": rows,
        "all_predictions_unavailable": True,
        "execution_basis": "preregistered_calibration_matrix_not_model_recommendation",
        "publication_allowed": False,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(PREDICTIONS, report)
    return report


def _jobs(
    inventory: dict[str, Any], profile_bindings: dict[tuple[str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    models = {str(row["id"]): row for row in inventory["models"]}
    model_specs = {str(row["id"]): row for row in MODELS}
    scenarios = [
        ("qwen2p5_vl_7b", "low", ("C1", "C3", "C4", "C5")),
        ("qwen2p5_vl_7b", "high", ("C3", "C4", "C5", "C1")),
        ("qwen3_vl_8b", "low", ("C4", "C5", "C1", "C3")),
        ("qwen3_vl_8b", "high", ("C5", "C1", "C3", "C4")),
    ]
    jobs = []
    # Four interleaved Latin-square rounds distribute candidate mechanisms
    # across the campaign clock while keeping execution strictly sequential.
    for round_index in range(4):
        for model_id, tier, order in scenarios:
            candidate_id = order[round_index]
            candidate = CANDIDATES[candidate_id]
            model = models[model_id]
            spec = model_specs[model_id]
            profile_binding = profile_bindings[(model_id, tier)]
            profile_path = Path(profile_binding["path"])
            row: dict[str, Any] = {
                "schema": JOB_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "phase_id": PHASE_ID,
                "candidate_role": f"vl_calibration_{model_id}_{tier}_{candidate_id.lower()}",
                "scenario_id": f"pzfj38-{model_id}-{tier}",
                "candidate_id": candidate_id,
                "latin_square_round": round_index,
                "model_id": model_id,
                "model_family": model["family"],
                "model_path": model["path"],
                "tokenizer_path": model["tokenizer_path"],
                "template": model["template"],
                "model_parameters": int(model["actual_parameters"]),
                "train_type": "lora",
                "dataset_id": "vl_pzfj38_calibration_v1",
                "profile_id": f"pzfj38-first1000-{model_id}-{tier}",
                "dataset_category": "vl_full_reference_image_quality",
                "data_path": str(DATA.resolve()),
                "data_sha256": sha256_file(DATA),
                "dataset_profile_path": str(profile_path.resolve()),
                "dataset_profile_sha256": sha256_file(profile_path),
                "media_manifest_path": str(MEDIA_MANIFEST.resolve()),
                "media_manifest_sha256": sha256_file(MEDIA_MANIFEST),
                "media_tier": tier,
                "cutoff_len": 8192,
                "target_gbs": 64,
                "gpu_count": int(candidate["gpu_count"]),
                "zero": candidate["zero"],
                "zero_stage": 0 if candidate["zero"] == "none" else int(str(candidate["zero"])[-1]),
                "gc": bool(candidate["gc"]),
                "gradient_checkpointing": bool(candidate["gc"]),
                "mbs": int(candidate["mbs"]),
                "gradient_accumulation_steps": int(candidate["ga"]),
                "packing": False,
                "offload": False,
                "freeze_vision_tower": True,
                "freeze_multi_modal_projector": True,
                "freeze_language_model": False,
                "image_min_pixels": int(spec["image_min_pixels"]),
                "image_max_pixels": 448 * 448 if tier == "low" else 768 * 768,
                "expected_images_per_sample": 2,
                "visual_runtime_evidence_required": True,
                "enable_liger_kernel": bool(spec["enable_liger_kernel"]),
                "effective_kernel_path": spec["effective_kernel_path"],
                "gpu_type": "NVIDIA H800 140GB HBM3",
                "required_runtime_gpu_name": "NVIDIA H800",
                "hardware_id": "local_h800_140g",
                "kind": "throughput",
                "warmup_steps": 2,
                "measure_steps": 8,
                "fidelity": "formal_vl_calibration_2plus8",
                "parallel_class": "exclusive_node",
                "requires_external_node_idle": True,
                "repeat": 0,
                "calibration_partition": {
                    "role": "calibration",
                    "split_unit_id": f"pzfj38-first1000-{model_id}-{tier}",
                    "policy": "fit_only_never_acceptance_v1",
                },
                "declared_model_manifest_path": str(INVENTORY.resolve()),
                "declared_model_manifest_sha256": sha256_file(INVENTORY),
                "frozen_prediction_available": False,
            }
            row["job_id"] = stable_id("h800vlcal", row)
            jobs.append(row)
    if len(jobs) != 16 or len({job["job_id"] for job in jobs}) != 16:
        raise ValueError("VL calibration must contain exactly 16 unique jobs")
    return jobs


def prepare() -> dict[str, Any]:
    for path in (DATA, MEDIA_MANIFEST, PROFILE_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    media = read_json(MEDIA_MANIFEST)
    if (
        media.get("derived", {}).get("rows") != 1000
        or media.get("derived", {}).get("images_per_row") != 2
        or media.get("derived", {}).get("all_decoded") is not True
        or media.get("derived", {}).get("sha256") != sha256_file(DATA)
    ):
        raise ValueError("VL calibration media is not complete")
    inventory = _inventory()
    profiles = _profiles()
    predictions = _frozen_unavailable_predictions()
    jobs = _jobs(inventory, profiles)
    write_jsonl(QUEUE, jobs)
    jobs_dir = ARTIFACT_DIR / "h800_vl_calibration_jobs_v1"
    for job in jobs:
        write_json(jobs_dir / f"{job['job_id']}.json", job)

    source_files = {
        "strict_design": ARTIFACT_DIR / "h800_vl_image_strict_acceptance_design_v1.md",
        "canary_closeout": ARTIFACT_DIR / "h800_packing_vl_canary_closeout_v2.json",
        "media_manifest": MEDIA_MANIFEST,
        "profile_manifest": PROFILE_MANIFEST,
        "dataset": DATA,
        "dataset_registry": DATA_DIR / "dataset_info.json",
        "experiment_config": ROOT / "config" / "experiment.json",
        "inventory": INVENTORY,
        "frozen_predictions": PREDICTIONS,
        "media_preparer": ROOT / "scripts" / "prepare_h800_vl_calibration_media_v1.py",
        "profile_preparer": ROOT / "scripts" / "prepare_h800_vl_calibration_profiles_v1.py",
        "queue_preparer": Path(__file__).resolve(),
        "run_job": ROOT / "scripts" / "run_job.py",
        "scheduler": ROOT / "scripts" / "scheduler.py",
        "train_entry": ROOT / "scripts" / "train_entry.py",
        "metrics_callback": ROOT / "scripts" / "metrics_callback.py",
        "model_structure_manifest": ROOT / "scripts" / "model_structure_manifest.py",
    }
    for (model_id, tier), binding in profiles.items():
        source_files[f"profile_{model_id}_{tier}"] = Path(binding["path"])
    design: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_gpu_waiting_for_exact_approval",
        "gpu_training_started": False,
        "fit_allowed_after_complete_results": True,
        "acceptance_allowed": False,
        "publication_allowed": False,
        "required_gpu_pool": {"gpu_ids": [0, 1], "max_gpu_count": 2, "preemption_allowed": False},
        "queue": {**_binding(QUEUE), "job_count": len(jobs), "ordered_job_ids": [job["job_id"] for job in jobs]},
        "model_inventory": _binding(INVENTORY),
        "frozen_predictions": _binding(PREDICTIONS),
        "media_manifest": _binding(MEDIA_MANIFEST),
        "profile_manifest": _binding(PROFILE_MANIFEST),
        "source_bindings": {name: _binding(path) for name, path in sorted(source_files.items())},
        "mechanism_contract": {
            "qwen2p5_vl_7b": {"enable_liger_kernel": True},
            "qwen3_vl_8b": {
                "enable_liger_kernel": False,
                "reason": "installed Liger runtime does not support Qwen3-VL",
            },
            "all_jobs_real_image_path_required": True,
            "all_jobs_language_only_lora": True,
            "packing": False,
        },
        "measurement_contract": {
            "warmup_steps": 2,
            "measure_steps": 8,
            "token_source": "consumed_token_ledger/v1",
            "memory_and_throughput_fit_only": True,
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    queue_manifest: dict[str, Any] = {
        "schema": "sft_h800_vl_calibration_queue_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "design": _binding(DESIGN),
        "queue": {**_binding(QUEUE), "job_count": len(jobs), "ordered_job_ids": [job["job_id"] for job in jobs]},
    }
    queue_manifest["report_sha256"] = sha256_json(queue_manifest)
    write_json(QUEUE_MANIFEST, queue_manifest)
    return {"design": _binding(DESIGN), "queue": _binding(QUEUE), "jobs": len(jobs), "predictions_all_unavailable": predictions["all_predictions_unavailable"]}


def main() -> None:
    import json

    print(json.dumps(prepare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
