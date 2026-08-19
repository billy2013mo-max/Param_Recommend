#!/usr/bin/env python3
"""Freeze source-disjoint hybrid-memory and image-VL acceptance queues."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, DATA_DIR, MATRIX_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json, write_jsonl
from h800_resource_predictor import H800ResourcePredictor
from run_job import resolve_dataset_dir, validate_job


SCHEMA = "sft_h800_hybrid_vl_prospective_acceptance_design/v1"
CAMPAIGN_ID = "h800_hybrid_vl_prospective_acceptance_20260817_v1"
PHASE_ID = "h800_hybrid_vl_prospective_acceptance_v1"
HYBRID_QUEUE = MATRIX_DIR / "h800_hybrid_memory_prospective_acceptance_v1.jsonl"
VL_QUEUE = MATRIX_DIR / "h800_vl_image_prospective_acceptance_v1.jsonl"
COMBINED_QUEUE = MATRIX_DIR / "h800_hybrid_vl_prospective_acceptance_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_hybrid_vl_prospective_acceptance_design_v1.json"
FROZEN_PREDICTIONS = ARTIFACT_DIR / "h800_hybrid_vl_prospective_frozen_predictions_v1.json"
SAFETY_ARTIFACT = ARTIFACT_DIR / "h800_hybrid_vl_safety_upper_v1.json"

QYPE_DATA = DATA_DIR / "vl_media_business_v1/derived/qype19_v7_images_local.jsonl"
VL_PROFILE_DIR = ARTIFACT_DIR / "h800_vl_business_workload_profiles_v2"
VL_MANIFEST = ARTIFACT_DIR / "h800_vl_media_business_download_manifest_v1.json"
VL_REGISTRY_DIR = DATA_DIR / "vl_prospective_acceptance_v1/registry"
VL_REGISTRY = VL_REGISTRY_DIR / "dataset_info.json"
VL_DATASET_ID = "vl_qype19_prospective_v1"
FORMAL_VL_QUEUE = MATRIX_DIR / "h800_frozen_vl_combined_formal_v1.jsonl"

HYBRID_SOURCE = ROOT / "blind_data/text/0105_inference.sft.jsonl"
HYBRID_PROFILE = ROOT / "blind_data/text/0105_inference.profile.json"
HYBRID_MODELS = {"qwen3p5_4b", "qwen3p5_9b", "qwen3_6_27b"}


def _job_id(material: str, prefix: str) -> str:
    return f"{prefix}-{hashlib.sha256(material.encode()).hexdigest()[:16]}"


def _write_vl_registry() -> None:
    payload = {
        VL_DATASET_ID: {
            "file_name": str(QYPE_DATA.resolve()),
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    write_json(VL_REGISTRY, payload)


def _hybrid_jobs() -> list[dict[str, Any]]:
    import prepare_h800_business_blind_v1 as prep

    prep.TEXT_SOURCE = HYBRID_SOURCE
    prep.TEXT_PROFILE = HYBRID_PROFILE
    prep.DATASET_ID = "business_0105_inference"
    prep._build_matrix()
    jobs = [
        deepcopy(row)
        for row in prep._build_jobs()
        if row["model_id"] in HYBRID_MODELS and row["packing"] is False
    ]
    if len(jobs) != 9:
        raise ValueError(f"expected 9 hybrid prospective jobs, got {len(jobs)}")
    for row in jobs:
        material = "::".join(
            str(row[key])
            for key in (
                "model_id",
                "gpu_count",
                "zero",
                "gc",
                "cutoff_len",
            )
        )
        row.update(
            {
                "campaign_id": CAMPAIGN_ID,
                "phase_id": PHASE_ID,
                "track": "hybrid_memory_source_disjoint_prospective",
                "evidence_role": "prospective_acceptance_frozen_before_outcomes",
                "design_arm": "hybrid_memory_prospective",
                "source_dataset_id": "业务前瞻0105-inference",
                "source_dataset_sha256": sha256_file(HYBRID_SOURCE),
                "job_id": _job_id(material, "h800hyacc"),
                "scenario_id": f"hybrid::{material}",
                "requested_gpu_pool": list(range(7)),
            }
        )
        validate_job(row)
        resolve_dataset_dir(row)
    return jobs


def _vl_templates() -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_jsonl(FORMAL_VL_QUEUE)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("arm_id") != "real_image":
            continue
        key = (str(row["model_id"]), str(row["mechanism_id"]))
        result.setdefault(key, row)
    expected = {
        (model_id, mechanism)
        for model_id in ("qwen2p5_vl_3b", "qwen3_vl_4b", "qwen3p5_4b")
        for mechanism in ("SAFE", "NOGC", "PRESSURE")
    }
    if set(result) != expected:
        raise ValueError(f"VL formal templates are incomplete: {expected - set(result)}")
    return result


def _vl_jobs() -> list[dict[str, Any]]:
    _write_vl_registry()
    templates = _vl_templates()
    rows: list[dict[str, Any]] = []
    for model_id in ("qwen2p5_vl_3b", "qwen3_vl_4b", "qwen3p5_4b"):
        for tier in ("low", "high"):
            profile_path = VL_PROFILE_DIR / f"{model_id}.qype19.{tier}.json"
            profile = read_json(profile_path)
            processor = profile["processor_binding"]
            for mechanism in ("SAFE", "NOGC", "PRESSURE"):
                row = deepcopy(templates[(model_id, mechanism)])
                material = f"{model_id}::{tier}::{mechanism}"
                row.update(
                    {
                        "campaign_id": CAMPAIGN_ID,
                        "phase_id": PHASE_ID,
                        "track": "vl_image_source_disjoint_prospective",
                        "evidence_role": "prospective_acceptance_frozen_before_outcomes",
                        "design_arm": "vl_image_prospective",
                        "dataset_id": VL_DATASET_ID,
                        "dataset_category": "vl_ecommerce_image_quality",
                        "dataset_dir": str(VL_REGISTRY_DIR.resolve()),
                        "dataset_registry_sha256": sha256_file(VL_REGISTRY),
                        "data_path": str(QYPE_DATA.resolve()),
                        "data_sha256": sha256_file(QYPE_DATA),
                        "dataset_profile_path": str(profile_path.resolve()),
                        "dataset_profile_sha256": sha256_file(profile_path),
                        "media_manifest_path": str(VL_MANIFEST.resolve()),
                        "media_manifest_sha256": sha256_file(VL_MANIFEST),
                        "media_tier": tier,
                        "image_min_pixels": int(processor["image_min_pixels"]),
                        "image_max_pixels": int(processor["image_max_pixels"]),
                        "max_samples": 512,
                        "job_id": _job_id(material, "h800vlacc"),
                        "pair_id": material,
                        "scenario_id": material,
                        "requested_gpu_pool": list(range(7)),
                    }
                )
                validate_job(row)
                resolve_dataset_dir(row)
                rows.append(row)
    if len(rows) != 18:
        raise ValueError(f"expected 18 VL prospective jobs, got {len(rows)}")
    return rows


def _hybrid_request(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": job["job_id"],
        "comparison_group": f"hybrid::{job['model_id']}::{job['cutoff_len']}",
        "hardware_id": "h800",
        "model_id": job["model_id"],
        "training_mode": "lora",
        "lora_rank": 32,
        "dataset_id": "longtail_8192",
        "dataset_category": "longtail",
        "target_gbs": job["target_gbs"],
        "cutoff_len": job["cutoff_len"],
        "gpu_count": job["gpu_count"],
        "physical_mbs": job["mbs"],
        "zero_stage": job["zero_stage"],
        "gradient_checkpointing": job["gc"],
        "packing": False,
        "dtype": "bf16",
    }


def _vl_request(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": job["job_id"],
        "comparison_group": f"vl::{job['model_id']}::{job['media_tier']}",
        "hardware_id": "h800",
        "model_id": job["model_id"],
        "training_mode": "lora",
        "lora_rank": 32,
        "target_gbs": job["target_gbs"],
        "cutoff_len": job["cutoff_len"],
        "gpu_count": 1,
        "physical_mbs": job["mbs"],
        "zero_stage": 0,
        "gradient_checkpointing": job["gc"],
        "packing": False,
        "dtype": "bf16",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "vl_workload_profile_path": job["dataset_profile_path"],
        "vl_workload_profile_sha256": job["dataset_profile_sha256"],
    }


def _freeze_predictions(
    hybrid_jobs: list[dict[str, Any]],
    vl_jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    predictor = H800ResourcePredictor()
    hybrid_requests = [_hybrid_request(job) for job in hybrid_jobs]
    vl_requests = [_vl_request(job) for job in vl_jobs]
    hybrid_report = predictor.predict(hybrid_requests)
    hybrid_throughput = predictor.predictor.base.predict_many(hybrid_requests)
    vl_report = predictor.predict(vl_requests)
    return {
        "schema": "sft_h800_hybrid_vl_prospective_frozen_predictions/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "safety_artifact": {
            "path": str(SAFETY_ARTIFACT.resolve()),
            "sha256": sha256_file(SAFETY_ARTIFACT),
        },
        "hybrid": {
            "predictions": hybrid_report["predictions"],
            "throughput_predictions": hybrid_throughput["predictions"],
        },
        "vl_image": {"predictions": vl_report["predictions"]},
    }


def prepare() -> dict[str, Any]:
    hybrid_jobs = _hybrid_jobs()
    vl_jobs = _vl_jobs()
    all_ids = [str(row["job_id"]) for row in [*hybrid_jobs, *vl_jobs]]
    existing = [
        job_id
        for job_id in all_ids
        if (ROOT / "results" / job_id / "status.json").is_file()
    ]
    if existing:
        raise RuntimeError(f"prospective outcomes already exist: {existing}")
    write_jsonl(HYBRID_QUEUE, hybrid_jobs)
    write_jsonl(VL_QUEUE, vl_jobs)
    write_jsonl(COMBINED_QUEUE, [*hybrid_jobs, *vl_jobs])
    predictions = _freeze_predictions(hybrid_jobs, vl_jobs)
    predictions["report_sha256"] = sha256_json(predictions)
    write_json(FROZEN_PREDICTIONS, predictions)
    design: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "usage": "source_disjoint_prospective_acceptance_only",
        "gpu_training_started": False,
        "queues": {
            "combined": {
                "path": str(COMBINED_QUEUE.resolve()),
                "sha256": sha256_file(COMBINED_QUEUE),
                "jobs": len(hybrid_jobs) + len(vl_jobs),
            },
            "hybrid": {
                "path": str(HYBRID_QUEUE.resolve()),
                "sha256": sha256_file(HYBRID_QUEUE),
                "jobs": len(hybrid_jobs),
            },
            "vl_image": {
                "path": str(VL_QUEUE.resolve()),
                "sha256": sha256_file(VL_QUEUE),
                "jobs": len(vl_jobs),
            },
        },
        "frozen_predictions": {
            "path": str(FROZEN_PREDICTIONS.resolve()),
            "sha256": sha256_file(FROZEN_PREDICTIONS),
        },
        "source_disjointness": {
            "hybrid_fit_sources_exclude_0105_inference": True,
            "hybrid_source_sha256": sha256_file(HYBRID_SOURCE),
            "vl_fit_sources": ["pzfj38_v113", "zltbjg_v2"],
            "vl_prospective_source": "qype19_v7",
            "vl_source_sha256": sha256_file(QYPE_DATA),
        },
        "acceptance_thresholds": {
            "false_safe_oom": 0,
            "exact_upper_coverage_min": 0.95,
            "admission_recall_min": 0.95,
            "ranking_hit_at_10_percent": 1.0,
            "worst_top1_regret_max": 0.10,
        },
        "scope": {
            "hybrid_models": sorted(HYBRID_MODELS),
            "hybrid_packing": False,
            "vl_models": ["qwen2p5_vl_3b", "qwen3_vl_4b", "qwen3p5_4b"],
            "vl_modality": "image",
            "vl_packing": False,
            "automatic_execution_allowed": False,
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    return design


if __name__ == "__main__":
    import json

    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
