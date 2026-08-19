#!/usr/bin/env python3
"""Prepare the V2 source-disjoint hybrid/VL prospective acceptance campaign.

This V2 campaign uses brand-new, never-old data as the source-disjoint set:
  * hybrid text  : blind_data/text/0105_inference2.sft.jsonl (never consumed)
  * VL images    : blind_data/vl real business frames + synthetic prompt
                   (built by build_h800_blind_vl_sft_v1.py)

Compared with the V1 campaign the acceptance evidence is:
  * a stage-2 hybrid center (h800_hybrid_memory_artifact_v2.json) that fixes
    the large-model ZeRO-3 under-prediction,
  * a V3 safety upper (h800_hybrid_vl_safety_upper_v3.json) fitted on the
    stage-1 development set only, so this campaign's rows stay untouched for
    acceptance,
  * an explicitly larger hybrid matrix (36 jobs) so a single point cannot sink
    the 0.95 coverage gate.

All predictions are frozen BEFORE any training job starts; the prepare asserts
that no result directory exists yet for any queued job.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, DATA_DIR, MATRIX_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json, write_jsonl
from run_job import resolve_dataset_dir, validate_job

BLIND_DIR = ROOT / "blind_data"

SCHEMA = "sft_h800_hybrid_vl_prospective_acceptance_design/v2"
CAMPAIGN_ID = "h800_hybrid_vl_prospective_acceptance_20260818_v2"
PHASE_ID = "h800_hybrid_vl_prospective_acceptance_v2"
HYBRID_QUEUE = MATRIX_DIR / "h800_hybrid_memory_prospective_acceptance_v2.jsonl"
VL_QUEUE = MATRIX_DIR / "h800_vl_image_prospective_acceptance_v2.jsonl"
COMBINED_QUEUE = MATRIX_DIR / "h800_hybrid_vl_prospective_acceptance_v2.jsonl"
DESIGN = ARTIFACT_DIR / "h800_hybrid_vl_prospective_acceptance_design_v2.json"
FROZEN_PREDICTIONS = ARTIFACT_DIR / "h800_hybrid_vl_prospective_frozen_predictions_v2.json"

# --- V2 campaign sources -----------------------------------------------------
TEXT_SOURCE = BLIND_DIR / "text" / "0105_inference2.sft.jsonl"
TEXT_PROFILE = BLIND_DIR / "text" / "0105_inference2.profile.json"
TEXT_REGISTRY = BLIND_DIR / "text" / "dataset_info.json"
TEXT_DATASET_ID = "business_0105_inference2"

VL_DATASET_ID = "vl_blind_prospective_v1"
VL_REGISTRY = DATA_DIR / "blind_vl_prospective_v1" / "registry" / "dataset_info.json"
VL_JSONL = DATA_DIR / "blind_vl_prospective_v1" / "blind_vl_images.jsonl"
VL_PROFILE_DIR = ARTIFACT_DIR / "h800_blind_vl_workload_profiles_v1"

MODEL_INVENTORY = ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_model_inventory_v1.json"
UPPER_V3 = ARTIFACT_DIR / "h800_hybrid_vl_safety_upper_v3.json"
FORMAL_VL_QUEUE = MATRIX_DIR / "h800_frozen_vl_combined_formal_v1.jsonl"

JOB_SCHEMA = "sft_h800_hybrid_attention_dense_stage1_job/v1"
HYBRID_MODELS = ("qwen3p5_4b", "qwen3p5_9b", "qwen3p6_27b")
VL_MODELS = ("qwen2p5_vl_3b", "qwen3_vl_4b", "qwen3p5_4b")
VL_MECHANISMS = ("SAFE", "NOGC", "PRESSURE")
VL_TIERS = ("low", "high")
TARGET_GBS = 64

# Hybrid configuration matrix: (model_id, gpu_count, zero, gc, cutoff_len).
# 1-card must be ZeRO-0 and multi-card ZeRO-2/3 (validate_job enforces this).
# 9b/27b multi-GPU ZeRO-3 is the coverage-critical cell (V1 missed it).
HYBRID_CONFIGS: tuple[tuple[str, int, str, bool, int], ...] = (
    # qwen3p5_4b
    ("qwen3p5_4b", 1, "none", True, 2048),
    ("qwen3p5_4b", 1, "none", False, 2048),
    ("qwen3p5_4b", 1, "none", False, 4096),
    ("qwen3p5_4b", 1, "none", True, 4096),
    ("qwen3p5_4b", 2, "zero2", True, 2048),
    ("qwen3p5_4b", 2, "zero2", False, 2048),
    ("qwen3p5_4b", 2, "zero2", False, 4096),
    ("qwen3p5_4b", 2, "zero3", True, 2048),
    ("qwen3p5_4b", 2, "zero3", False, 2048),
    ("qwen3p5_4b", 2, "zero3", False, 4096),
    ("qwen3p5_4b", 4, "zero2", True, 2048),
    ("qwen3p5_4b", 4, "zero2", False, 2048),
    ("qwen3p5_4b", 4, "zero2", False, 4096),
    ("qwen3p5_4b", 4, "zero3", True, 2048),
    ("qwen3p5_4b", 4, "zero3", False, 2048),
    ("qwen3p5_4b", 4, "zero3", False, 4096),
    # qwen3p5_9b (ZeRO-3 stress)
    ("qwen3p5_9b", 2, "zero2", True, 2048),
    ("qwen3p5_9b", 2, "zero2", False, 2048),
    ("qwen3p5_9b", 2, "zero3", True, 2048),
    ("qwen3p5_9b", 2, "zero3", False, 2048),
    ("qwen3p5_9b", 2, "zero3", False, 4096),
    ("qwen3p5_9b", 2, "zero2", False, 4096),
    ("qwen3p5_9b", 4, "zero2", True, 2048),
    ("qwen3p5_9b", 4, "zero2", False, 2048),
    ("qwen3p5_9b", 4, "zero3", True, 2048),
    ("qwen3p5_9b", 4, "zero3", False, 2048),
    ("qwen3p5_9b", 4, "zero3", False, 4096),
    ("qwen3p5_9b", 4, "zero2", False, 4096),
    # qwen3p6_27b (ZeRO-3 stress)
    ("qwen3p6_27b", 2, "zero2", True, 2048),
    ("qwen3p6_27b", 2, "zero2", False, 2048),
    ("qwen3p6_27b", 2, "zero3", True, 2048),
    ("qwen3p6_27b", 2, "zero3", False, 2048),
    ("qwen3p6_27b", 2, "zero3", False, 4096),
    ("qwen3p6_27b", 4, "zero3", True, 2048),
    ("qwen3p6_27b", 4, "zero3", False, 2048),
    ("qwen3p6_27b", 4, "zero3", False, 4096),
)


def _zero_stage(zero: str) -> int:
    return {"none": 0, "zero0": 0, "zero1": 1, "zero2": 2, "zero3": 3}[zero]


def _job_id(prefix: str, material: str) -> str:
    return f"{prefix}-{hashlib.sha256(material.encode()).hexdigest()[:16]}"


def _hybrid_overlay() -> dict:
    from prepare_h800_business_blind_v1 import QWEN36_VENV, QWEN35_CONTRACT, QWEN35_TILELANG_OVERLAY

    return {
        "PYTHONPATH_prepend": [QWEN35_TILELANG_OVERLAY, str(QWEN36_VENV.resolve())],
        "contract_path": str(QWEN35_CONTRACT.resolve()),
        "contract_sha256": sha256_file(QWEN35_CONTRACT),
        "variables": {"FLA_TILELANG": "1", "TILELANG_CACHE_DIR": "/tmp/qwen35_tilelang_cache_h800_precedence_v2"},
    }


def _hybrid_jobs() -> list[dict[str, Any]]:
    from prepare_h800_business_blind_v1 import _architecture_signature

    inventory = read_json(MODEL_INVENTORY)
    models = {str(m["id"]): m for m in inventory["models"]}
    data_sha256 = sha256_file(TEXT_SOURCE)
    profile_sha256 = sha256_file(TEXT_PROFILE)
    registry_sha256 = sha256_file(TEXT_REGISTRY)
    overlay = _hybrid_overlay()

    jobs: list[dict[str, Any]] = []
    for idx, (model_id, gpu, zero, gc, cutoff) in enumerate(HYBRID_CONFIGS):
        model = models[model_id]
        sig = _architecture_signature(model)
        mbs = 1
        material = f"{model_id}::{gpu}::{zero}::{gc}::{cutoff}"
        job: dict[str, Any] = {
            "schema": JOB_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "phase_id": PHASE_ID,
            "track": "hybrid_memory_source_disjoint_prospective_v2",
            "evidence_role": "prospective_acceptance_frozen_before_outcomes",
            "design_arm": "hybrid_memory_prospective_v2",
            "model_id": model_id,
            "model_family": model["family"],
            "release_family": model.get("release_family", model["family"]),
            "model_path": model["path"],
            "tokenizer_path": model["tokenizer_path"],
            "model_parameters": int(model["actual_parameters"]),
            "template": model["template"],
            "train_type": "lora",
            "target_gbs": TARGET_GBS,
            "gpu_count": gpu,
            "zero": zero,
            "zero_stage": _zero_stage(zero),
            "gc": gc,
            "gradient_checkpointing": gc,
            "mbs": mbs,
            "gradient_accumulation_steps": max(1, TARGET_GBS // (gpu * mbs)),
            "packing": False,
            "offload": False,
            "kind": "throughput",
            "warmup_steps": 3,
            "measure_steps": 10,
            "fidelity": "formal_3plus10",
            "repeat": 0,
            "parallel_route_id": f"D{gpu}_Z{_zero_stage(zero)}",
            "architecture_route": sig["architecture_route"],
            "architecture_signature_sha256": sig["signature_sha256"],
            "num_full_attention_layers": int(sig["num_full_attention_layers"]),
            "num_linear_attention_layers": int(sig["num_linear_attention_layers"]),
            "dataset_id": TEXT_DATASET_ID,
            "dataset_dir": str(TEXT_REGISTRY.parent.resolve()),
            "dataset_registry_sha256": registry_sha256,
            "data_path": str(TEXT_SOURCE.resolve()),
            "data_sha256": data_sha256,
            "dataset_profile_path": str(TEXT_PROFILE.resolve()),
            "dataset_profile_sha256": profile_sha256,
            "source_dataset_id": "业务前瞻0105-inference2",
            "source_dataset_sha256": data_sha256,
            "cutoff_len": cutoff,
            "exact_total_tokens_per_sample": cutoff,
            "all_total_tokens_exact": False,
            "max_samples": 64,
            "enable_liger_kernel": bool(model.get("enable_liger_kernel", True)),
            "effective_kernel_path": "fa3_orig+liger_fused_ce+adamw_torch_fused",
            "environment_overlay": overlay,
            "required_runtime_gpu_name": "NVIDIA H800",
            "required_gpu_type": "NVIDIA H800",
            "gpu_type": "NVIDIA H800 140GB HBM3",
            "hardware_id": "local_h800_140g",
            "requested_gpu_pool": list(range(8)),
            "requires_external_node_idle": False,
            "visual_runtime_evidence_required": False,
            "job_id": _job_id("h800hyv2", material),
            "scenario_id": f"hybrid::{material}",
            "comparison_group": f"hybrid::{model_id}::{cutoff}",
            "target_gbs_override": TARGET_GBS,
        }
        validate_job(job)
        resolve_dataset_dir(job)
        jobs.append(job)
    if len(jobs) != len(HYBRID_CONFIGS):
        raise ValueError(f"expected {len(HYBRID_CONFIGS)} hybrid jobs, got {len(jobs)}")
    return jobs


def _vl_jobs() -> list[dict[str, Any]]:
    templates = {
        (str(row["model_id"]), str(row["mechanism_id"])): row
        for row in read_jsonl(FORMAL_VL_QUEUE)
        if row.get("arm_id") == "real_image"
    }
    registry_sha256 = sha256_file(VL_REGISTRY)
    jobs: list[dict[str, Any]] = []
    for model_id in VL_MODELS:
        for tier in VL_TIERS:
            profile_path = VL_PROFILE_DIR / f"{model_id}.blindvl.{tier}.json"
            profile = read_json(profile_path)
            for mechanism in VL_MECHANISMS:
                row = dict(templates[(model_id, mechanism)])
                material = f"{model_id}::{tier}::{mechanism}"
                row.update(
                    {
                        "campaign_id": CAMPAIGN_ID,
                        "phase_id": PHASE_ID,
                        "track": "vl_image_source_disjoint_prospective_v2",
                        "evidence_role": "prospective_acceptance_frozen_before_outcomes",
                        "design_arm": "vl_image_prospective_v2",
                        "dataset_id": VL_DATASET_ID,
                        "dataset_directory": str(VL_REGISTRY.parent.resolve()),
                        "dataset_dir": str(VL_REGISTRY.parent.resolve()),
                        "dataset_registry_sha256": registry_sha256,
                        "data_path": str(VL_JSONL.resolve()),
                        "data_sha256": sha256_file(VL_JSONL),
                        "dataset_profile_path": str(profile_path.resolve()),
                        "dataset_profile_sha256": sha256_file(profile_path),
                        "dataset_category": "blind_business_frames_synthetic_prompt",
                        "media_tier": tier,
                        "job_id": _job_id("h800vlv2", material),
                        "pair_id": material,
                        "scenario_id": material,
                        "requested_gpu_pool": list(range(8)),
                    }
                )
                validate_job(row)
                resolve_dataset_dir(row)
                jobs.append(row)
    if len(jobs) != len(VL_MODELS) * len(VL_TIERS) * len(VL_MECHANISMS):
        raise ValueError(f"expected {len(VL_MODELS) * len(VL_TIERS) * len(VL_MECHANISMS)} VL jobs")
    return jobs


def _hybrid_request(job: dict[str, Any]) -> dict[str, Any]:
    # The production throughput predictor resolves model ids from its own
    # inventory, which names the 27B hybrid "qwen3_6_27b" (stage-1 naming).
    model_id = "qwen3_6_27b" if job["model_id"] == "qwen3p6_27b" else job["model_id"]
    return {
        "request_id": job["job_id"],
        "comparison_group": f"hybrid::{job['model_id']}::{job['cutoff_len']}",
        "hardware_id": "h800",
        "model_id": model_id,
        "training_mode": "lora",
        "lora_rank": 32,
        "dataset_id": "longtail_8192",
        "dataset_category": "longtail",
        "target_gbs": TARGET_GBS,
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
        "target_gbs": TARGET_GBS,
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


def _frozen_hybrid_prediction(job: dict[str, Any]) -> dict[str, Any]:
    """Frozen V2 hybrid prediction: stage-2 center x V3 safety multiplier.

    The V2 center comes from the stage-2 fit (fixed ZeRO-3 under-prediction);
    the safety multiplier comes from the V3 upper fitted on the stage-1
    development rows only, so this campaign's outcomes stay clean.
    """
    from hybrid_memory_bridge import predict_memory_center_v2

    upper_v3 = read_json(UPPER_V3)
    multipliers = upper_v3["hybrid_memory"]["conditional_by_model_mechanism"]
    mechanism = f"{job['model_id']}::g{int(job['gpu_count'])}_z{job['zero_stage']}_gc{int(bool(job['gc']))}"
    if mechanism not in multipliers:
        raise ValueError(f"no V3 upper multiplier for {mechanism}")
    multiplier = float(multipliers[mechanism]["upper_multiplier"])
    center_v2 = float(
        predict_memory_center_v2(
            {"path": job["model_path"]},
            job,
            fixed_lora={"rank": 32, "target": "all"},
            capacity_bytes=150_142_189_568,
            coefficients_by_name=read_json(
                ARTIFACT_DIR / "h800_hybrid_memory_artifact_v2.json"
            )["coefficients_by_name"],
        )["center_bytes"]
    )
    upper = center_v2 * multiplier
    return {
        "request_id": job["job_id"],
        "comparison_group": f"hybrid::{job['model_id']}::{job['cutoff_len']}",
        "memory": {
            "prediction_available": True,
            "hybrid_model": True,
            "reserved_center_bytes": float(center_v2),
            "risk_guard_multiplier": multiplier,
            "safety_upper_source": UPPER_V3.name,
            "admission_upper_reserved_bytes": float(upper),
            "safe_limit_bytes": float(0.95 * 150_142_189_568),
        },
        "throughput": None,
        "hybrid_shadow": {"center_version": "v2", "upper_version": "v3"},
    }


def _freeze_predictions(
    hybrid_jobs: list[dict[str, Any]],
    vl_jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    from h800_resource_predictor import H800ResourcePredictor

    # Hybrid: local V2/V3 chain independent of the production V3 predictor.
    hybrid_predictions = [_frozen_hybrid_prediction(job) for job in hybrid_jobs]

    # Hybrid throughput ranking uses the production V5 throughput predictor
    # (the stage-2 fix is memory-side only; V1 froze hybrid throughput the
    # same way).
    hybrid_requests = [_hybrid_request(job) for job in hybrid_jobs]
    predictor = H800ResourcePredictor()
    hybrid_throughput = predictor.predictor.base.predict_many(hybrid_requests)

    # VL: reuse the frozen-VL shadow predictor, but force the blindvl dataset
    # profile binding so the prediction is frozen against the new source.
    vl_requests = [_vl_request(job) for job in vl_jobs]
    vl_report = predictor.predict(vl_requests)
    return {
        "schema": "sft_h800_hybrid_vl_prospective_frozen_predictions/v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "safety_upper": {"path": str(UPPER_V3.resolve()), "sha256": sha256_file(UPPER_V3)},
        "hybrid": {"predictions": hybrid_predictions, "throughput_predictions": hybrid_throughput["predictions"]},
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
        "usage": "source_disjoint_prospective_acceptance_only_v2",
        "gpu_training_started": False,
        "queues": {
            "combined": {"path": str(COMBINED_QUEUE.resolve()), "sha256": sha256_file(COMBINED_QUEUE), "jobs": len(hybrid_jobs) + len(vl_jobs)},
            "hybrid": {"path": str(HYBRID_QUEUE.resolve()), "sha256": sha256_file(HYBRID_QUEUE), "jobs": len(hybrid_jobs)},
            "vl_image": {"path": str(VL_QUEUE.resolve()), "sha256": sha256_file(VL_QUEUE), "jobs": len(vl_jobs)},
        },
        "frozen_predictions": {"path": str(FROZEN_PREDICTIONS.resolve()), "sha256": sha256_file(FROZEN_PREDICTIONS)},
        "source_disjointness": {
            "hybrid_fit_sources_exclude_0105_inference2": True,
            "hybrid_source_sha256": sha256_file(TEXT_SOURCE),
            "vl_fit_sources": ["pzfj38_v113", "zltbjg_v2", "qype19_v7"],
            "vl_prospective_source": "blind_data/vl synthetic-prompt frames",
            "vl_source_sha256": sha256_file(VL_JSONL),
            "note": "blind_data/vl groups were never used by any fit or acceptance before this campaign.",
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
            "vl_models": list(VL_MODELS),
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
