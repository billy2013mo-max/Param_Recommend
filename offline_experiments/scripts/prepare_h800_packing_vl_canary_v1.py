#!/usr/bin/env python3
"""Freeze the Packing-then-VL semantic canary design and staged queues."""

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
from static_packing_predictor import build_decision, load_policy


SCHEMA = "sft_h800_packing_vl_canary_design/v1"
JOB_SCHEMA = "sft_h800_packing_vl_canary_job/v1"
CAMPAIGN_ID = "h800_packing_vl_canary_20260803_v1"
PHASE_ID = "h800_packing_vl_canary_v1"
DESIGN_PATH = ARTIFACT_DIR / "h800_packing_vl_canary_design_v1.json"
DECISIONS_PATH = ARTIFACT_DIR / "h800_packing_vl_canary_frozen_decisions_v1.json"
INVENTORY_PATH = ARTIFACT_DIR / "h800_packing_vl_canary_model_inventory_v1.json"
QUEUE_MANIFEST_PATH = ARTIFACT_DIR / "h800_packing_vl_canary_queue_manifest_v1.json"
PACKING_QUEUE = MATRIX_DIR / "h800_packing_semantic_canary_v1.jsonl"
VL_QUEUE = MATRIX_DIR / "h800_vl_media_canary_v1.jsonl"
MEDIA_MANIFEST = ARTIFACT_DIR / "h800_vl_canary_media_manifest_v1.json"
POLICY_PATH = ARTIFACT_DIR / "static_packing_policy_v1.json"
PACKING_PROFILE = (
    ARTIFACT_DIR
    / "bounded_memory_v2_fresh_holdout_v1"
    / "profiles"
    / "fresh_s3_rare_tail_short_v1.qwen3_nothink.jsonl"
)
PACKING_DATA = (
    DATA_DIR
    / "bounded_memory_v2_fresh_holdout_v1"
    / "fresh_s3_rare_tail_short_v1.jsonl"
)
VL_DATA = DATA_DIR / "packing_vl_canary_v1" / "vl_pzfj38_canary_v1.jsonl"


MODEL_ENTRIES = (
    {
        "id": "qwen3_8b",
        "nominal_scale_b": 8,
        "path": "/wanqing-models/Qwen3-8B",
        "tokenizer_path": "/wanqing-models/Qwen3-8B",
        "family": "qwen3",
        "template": "qwen3_nothink",
        "train_types": ["lora"],
        "canary_role": "text_neat_packing_semantics",
    },
    {
        "id": "qwen2p5_vl_7b",
        "nominal_scale_b": 7,
        "path": "/wanqing-models/Qwen2.5-VL-7B-Instruct",
        "tokenizer_path": "/wanqing-models/Qwen2.5-VL-7B-Instruct",
        "family": "qwen2p5_vl",
        "template": "qwen2_vl",
        "train_types": ["lora"],
        "canary_role": "real_image_media_path_generation_2p5",
    },
    {
        "id": "qwen3_vl_8b",
        "nominal_scale_b": 8,
        "path": "/wanqing-models/Qwen3-VL-8B-Instruct",
        "tokenizer_path": "/wanqing-models/Qwen3-VL-8B-Instruct",
        "family": "qwen3_vl",
        "template": "qwen3_vl",
        "train_types": ["lora"],
        "canary_role": "real_image_media_path_generation_3",
    },
)


def _binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _inventory() -> dict[str, Any]:
    rows = []
    for entry in MODEL_ENTRIES:
        print(f"inventory {entry['id']}", flush=True)
        rows.append(inventory_model(dict(entry)))
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_vl_canary_model_inventory/v1",
        "campaign_id": CAMPAIGN_ID,
        "fixed_lora": {"rank": 32, "alpha": 32, "dropout": 0.0, "target": "all"},
        "models": rows,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(INVENTORY_PATH, report)
    return report


def _packing_decision() -> dict[str, Any]:
    policy = load_policy(POLICY_PATH.resolve())
    request = {
        "request_id": "h800-packing-vl-canary-static-v1",
        "gpu_family": "H800",
        "modality": "text",
        "stage": "sft",
        "dtype": "bf16",
        "model_id": "qwen3_8b",
        "train_type": "lora",
        "profile_path": str(PACKING_PROFILE.resolve()),
        "cutoff_len": 4096,
        "no_packing_mbs": 1,
        "gpu_count": 1,
        "data_parallel": 1,
        "target_gbs": 64,
        "preprocessing_num_workers": 8,
        "packing_algorithm_id": policy["packing_algorithm"]["id"],
    }
    decision = build_decision(
        request,
        policy=policy,
        policy_path=POLICY_PATH.resolve(),
        request_base=ROOT.parent,
    )
    geometry = (decision.get("features") or {}).get("packed_batch_geometry") or {}
    if (
        decision.get("recommendation", {}).get("decision") != "on"
        or geometry.get("gradient_accumulation_steps") != 10
        or float(geometry.get("relative_error") or 1.0) > 0.05
    ):
        raise RuntimeError(f"static Packing decision is not the frozen ON canary: {decision}")
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_vl_canary_frozen_decisions/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_before_gpu": True,
        "packing_static_decision": decision,
        "vl_prediction": {
            "status": "not_available_semantic_canary_only",
            "automatic_recommendation_allowed": False,
            "reason": "real-image VL memory and throughput heads are not calibrated",
        },
    }
    report["report_sha256"] = sha256_json(report)
    write_json(DECISIONS_PATH, report)
    return report


def _base_job(model: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "schema": JOB_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "candidate_role": role,
        "model_id": model["id"],
        "model_family": model["family"],
        "model_path": model["path"],
        "tokenizer_path": model["tokenizer_path"],
        "template": model["template"],
        "model_parameters": int(model["actual_parameters"]),
        "train_type": "lora",
        "target_gbs": 64,
        "gpu_type": "NVIDIA H800 140GB HBM3",
        "required_runtime_gpu_name": "NVIDIA H800",
        "hardware_id": "local_h800_140g",
        "offload": False,
        "repeat": 0,
        "kind": "throughput",
        "warmup_steps": 0,
        "measure_steps": 2,
        "fidelity": "semantic_canary_0plus2",
        "software_canary": True,
        "requires_external_node_idle": False,
        "calibration_partition": {
            "role": "canary_excluded",
            "policy": "software_and_runtime_semantics_only_not_fit_or_acceptance",
        },
        "declared_model_manifest_path": str(INVENTORY_PATH.resolve()),
        "declared_model_manifest_sha256": sha256_file(INVENTORY_PATH),
    }


def _with_job_id(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    result = dict(row)
    result["job_id"] = stable_id(prefix, result)
    return result


def _jobs(inventory: dict[str, Any], decisions: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models = {str(row["id"]): row for row in inventory["models"]}
    decision = decisions["packing_static_decision"]
    geometry = decision["features"]["packed_batch_geometry"]
    pair_id = stable_id(
        "packingcanarypair",
        {"campaign_id": CAMPAIGN_ID, "profile_sha256": sha256_file(PACKING_PROFILE)},
    )
    packing_jobs = []
    for treatment, packing, ga in (("unpacked", False, 64), ("neat_packed", True, 10)):
        row = _base_job(models["qwen3_8b"], f"packing_semantic_{treatment}")
        row.update(
            {
                "scenario_id": pair_id,
                "packing_pair_id": pair_id,
                "packing_treatment": treatment,
                "dataset_id": "fresh_s3_rare_tail_short_v1",
                "profile_id": "fresh_s3_rare_tail_short_v1",
                "dataset_category": "rare_tail_short",
                "data_path": str(PACKING_DATA.resolve()),
                "data_sha256": sha256_file(PACKING_DATA),
                "dataset_profile_path": str(PACKING_PROFILE.resolve()),
                "dataset_profile_sha256": sha256_file(PACKING_PROFILE),
                "cutoff_len": 4096,
                "gpu_count": 1,
                "zero": "none",
                "zero_stage": 0,
                "gc": False,
                "gradient_checkpointing": False,
                "mbs": 1,
                "gradient_accumulation_steps": ga,
                "packing": packing,
                "expected_sample_gbs": (
                    float(geometry["expected_sample_gbs"]) if packing else 64.0
                ),
                "expected_sample_gbs_relative_error": (
                    float(geometry["relative_error"]) if packing else 0.0
                ),
                "static_packing_decision_id": decision["decision_id"],
                "static_packing_decision_report_sha256": decision["report_sha256"],
                "parallel_class": "gpu_partitionable",
            }
        )
        packing_jobs.append(_with_job_id(row, "h800packcanary"))

    vl_jobs = []
    for model_id, minimum_pixels in (("qwen2p5_vl_7b", 56 * 56), ("qwen3_vl_8b", 64 * 64)):
        row = _base_job(models[model_id], "vl_real_image_software_media_canary")
        row.update(
            {
                "scenario_id": f"pzfj38_high_media__{model_id}__c3",
                "dataset_id": "vl_pzfj38_canary_v1",
                "profile_id": "pzfj38_first128_two_image_high_media_v1",
                "dataset_category": "vl_full_reference_image_quality",
                "data_path": str(VL_DATA.resolve()),
                "data_sha256": sha256_file(VL_DATA),
                "dataset_profile_path": str(MEDIA_MANIFEST.resolve()),
                "dataset_profile_sha256": sha256_file(MEDIA_MANIFEST),
                "cutoff_len": 8192,
                "gpu_count": 2,
                "zero": "zero3",
                "zero_stage": 3,
                "gc": True,
                "gradient_checkpointing": True,
                "mbs": 1,
                "gradient_accumulation_steps": 32,
                "packing": False,
                "freeze_vision_tower": True,
                "freeze_multi_modal_projector": True,
                "freeze_language_model": False,
                "image_min_pixels": minimum_pixels,
                "image_max_pixels": 768 * 768,
                "expected_images_per_sample": 2,
                "media_profile": "high_media_768_square_cap",
                "visual_runtime_evidence_required": True,
                "parallel_class": "gpu_partitionable",
            }
        )
        vl_jobs.append(_with_job_id(row, "h800vlcanary"))
    return packing_jobs, vl_jobs


def prepare() -> dict[str, Any]:
    for path in (POLICY_PATH, PACKING_PROFILE, PACKING_DATA, VL_DATA, MEDIA_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    media = read_json(MEDIA_MANIFEST)
    if (
        media.get("schema") != "sft_h800_vl_canary_media/v1"
        or media.get("derived", {}).get("rows") != 128
        or media.get("derived", {}).get("images_per_row") != 2
        or media.get("derived", {}).get("all_decoded") is not True
        or media.get("derived", {}).get("sha256") != sha256_file(VL_DATA)
    ):
        raise ValueError("VL media manifest is not the exact decoded 128-row canary")
    inventory = _inventory()
    decisions = _packing_decision()
    packing_jobs, vl_jobs = _jobs(inventory, decisions)
    write_jsonl(PACKING_QUEUE, packing_jobs)
    write_jsonl(VL_QUEUE, vl_jobs)
    jobs_dir = ARTIFACT_DIR / "h800_packing_vl_canary_jobs_v1"
    for row in packing_jobs + vl_jobs:
        write_json(jobs_dir / f"{row['job_id']}.json", row)

    source_files = {
        "packing_acceptance_design": ARTIFACT_DIR / "h800_text_neat_packing_strict_acceptance_design_v1.md",
        "vl_acceptance_design": ARTIFACT_DIR / "h800_vl_image_strict_acceptance_design_v1.md",
        "packing_policy": POLICY_PATH,
        "packing_profile": PACKING_PROFILE,
        "packing_data": PACKING_DATA,
        "vl_media_manifest": MEDIA_MANIFEST,
        "vl_data": VL_DATA,
        "dataset_registry": DATA_DIR / "dataset_info.json",
        "experiment_config": ROOT / "config" / "experiment.json",
        "model_inventory": INVENTORY_PATH,
        "frozen_decisions": DECISIONS_PATH,
        "media_preparer": ROOT / "scripts" / "prepare_h800_vl_canary_media_v1.py",
        "canary_preparer": Path(__file__).resolve(),
        "approval_freezer": ROOT / "scripts" / "freeze_h800_packing_vl_canary_v1.py",
        "evaluator": ROOT / "scripts" / "evaluate_h800_packing_vl_canary_v1.py",
        "run_job": ROOT / "scripts" / "run_job.py",
        "scheduler": ROOT / "scripts" / "scheduler.py",
        "train_entry": ROOT / "scripts" / "train_entry.py",
        "metrics_callback": ROOT / "scripts" / "metrics_callback.py",
        "model_structure_manifest": ROOT / "scripts" / "model_structure_manifest.py",
    }
    missing_sources = [str(path) for path in source_files.values() if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"canary implementation is incomplete: {missing_sources}")
    design: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_gpu_waiting_for_exact_staged_approval",
        "gpu_training_started": False,
        "execution_authorized": False,
        "publication_allowed": False,
        "execution_order": ["packing", "vl"],
        "required_gpu_pool": {
            "gpu_ids": [0, 1],
            "max_gpu_count": 2,
            "preemption_allowed": False,
        },
        "stages": {
            "packing": {
                "purpose": "U/P semantic and sample-GBS canary only",
                "job_count": len(packing_jobs),
                "queue": _binding(PACKING_QUEUE),
                "job_ids": [row["job_id"] for row in packing_jobs],
                "fit_allowed": False,
            },
            "vl": {
                "purpose": "two-generation real-image media-path and freeze canary only",
                "job_count": len(vl_jobs),
                "queue": _binding(VL_QUEUE),
                "job_ids": [row["job_id"] for row in vl_jobs],
                "fit_allowed": False,
                "prerequisite": "packing stage evaluated semantic_pass before a new approval",
            },
        },
        "frozen_decisions": _binding(DECISIONS_PATH),
        "model_inventory": _binding(INVENTORY_PATH),
        "source_bindings": {
            name: _binding(path) for name, path in sorted(source_files.items())
        },
        "semantic_gates": {
            "packing": {
                "both_jobs_success": True,
                "packed_runtime_semantic_checks_passed": True,
                "sample_gbs_relative_error_max": 0.05,
            },
            "vl": {
                "both_models_success": True,
                "all_ranks_real_image_path_observed": True,
                "all_ranks_freeze_declarations_matched": True,
                "all_ranks_visual_lora_target_hits": False,
            },
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN_PATH, design)
    queue_manifest: dict[str, Any] = {
        "schema": "sft_h800_packing_vl_canary_queue_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "design": _binding(DESIGN_PATH),
        "staged_queues": {
            "packing": {**_binding(PACKING_QUEUE), "job_count": len(packing_jobs)},
            "vl": {**_binding(VL_QUEUE), "job_count": len(vl_jobs)},
        },
    }
    queue_manifest["report_sha256"] = sha256_json(queue_manifest)
    write_json(QUEUE_MANIFEST_PATH, queue_manifest)
    return {
        "design": str(DESIGN_PATH),
        "design_sha256": sha256_file(DESIGN_PATH),
        "packing_queue": str(PACKING_QUEUE),
        "packing_jobs": len(packing_jobs),
        "vl_queue": str(VL_QUEUE),
        "vl_jobs": len(vl_jobs),
        "inventory": str(INVENTORY_PATH),
        "decisions": str(DECISIONS_PATH),
    }


def main() -> None:
    import json

    print(json.dumps(prepare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
