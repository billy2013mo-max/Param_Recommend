#!/usr/bin/env python3
"""V3 prospective queue: VL frame grid + hybrid GC-off/Z3 boundary + OOM.

VL grid (36 jobs): 3 models x {frames 1,2,4} x {gc on/off} x {mbs 1,4} so the
vision-tower peak's patch (frames), saved-activation (gc) and batch (mbs)
dependencies are all identifiable.  Profiles come from the f1/f2/f4 blindvl
datasets.

hybrid (18 jobs): re-covers the GC-off / multi-GPU ZeRO-3 cells that V2 missed
and adds OOM-boundary rows (27b/9b GC-off larger cutoffs) to validate OOM
admission rate.

Frozen predictions use the V3 hybrid center x V4 safety upper, and the
production VL shadow for image rows (source-disjoint blindvl profiles).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, DATA_DIR, MATRIX_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json, write_jsonl
from prepare_h800_hybrid_vl_prospective_acceptance_v2 import _zero_stage

SCHEMA = "sft_h800_hybrid_vl_prospective_acceptance_design/v3"
CAMPAIGN_ID = "h800_hybrid_vl_prospective_acceptance_20260818_v3"
PHASE_ID = "h800_hybrid_vl_prospective_acceptance_v3"
HYBRID_QUEUE = MATRIX_DIR / "h800_hybrid_memory_prospective_acceptance_v3.jsonl"
VL_QUEUE = MATRIX_DIR / "h800_vl_image_prospective_acceptance_v3.jsonl"
COMBINED_QUEUE = MATRIX_DIR / "h800_hybrid_vl_prospective_acceptance_v3.jsonl"
DESIGN = ARTIFACT_DIR / "h800_hybrid_vl_prospective_acceptance_design_v3.json"
FROZEN_PREDICTIONS = ARTIFACT_DIR / "h800_hybrid_vl_prospective_frozen_predictions_v3.json"

UPPER_V4 = ARTIFACT_DIR / "h800_hybrid_vl_safety_upper_v4.json"
MODEL_INVENTORY = ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_model_inventory_v1.json"
JOB_SCHEMA = "sft_h800_hybrid_attention_dense_stage1_job/v1"
TARGET_GBS = 64


def _job_id(prefix: str, material: str) -> str:
    return f"{prefix}-{hashlib.sha256(material.encode()).hexdigest()[:16]}"


def _hybrid_jobs() -> list[dict[str, Any]]:
    from prepare_h800_business_blind_v1 import _architecture_signature, QWEN36_VENV, QWEN35_CONTRACT, QWEN35_TILELANG_OVERLAY

    from common import ROOT as _r
    text_source = _r / "blind_data" / "text" / "0105_inference2.sft.jsonl"
    text_profile = _r / "blind_data" / "text" / "0105_inference2.profile.json"
    text_registry = _r / "blind_data" / "text" / "dataset_info.json"
    inventory = read_json(MODEL_INVENTORY)
    models = {str(m["id"]): m for m in inventory["models"]}
    overlay = {
        "PYTHONPATH_prepend": [QWEN35_TILELANG_OVERLAY, str(QWEN36_VENV.resolve())],
        "contract_path": str(QWEN35_CONTRACT.resolve()),
        "contract_sha256": sha256_file(QWEN35_CONTRACT),
        "variables": {"FLA_TILELANG": "1", "TILELANG_CACHE_DIR": "/tmp/qwen35_tilelang_cache_h800_precedence_v3"},
    }
    # model, gpu, zero, gc, cutoff   (GC-off/Z3 boundary + OOM rows)
    configs = (
        ("qwen3p5_9b", 2, "zero3", False, 4096),
        ("qwen3p5_9b", 4, "zero3", False, 4096),
        ("qwen3p5_9b", 4, "zero2", False, 4096),
        ("qwen3p6_27b", 2, "zero2", False, 4096),
        ("qwen3p6_27b", 2, "zero3", False, 4096),
        ("qwen3p6_27b", 4, "zero3", False, 4096),
        ("qwen3p6_27b", 2, "zero3", True, 4096),
        ("qwen3p6_27b", 4, "zero3", True, 4096),
        ("qwen3p5_9b", 2, "zero3", True, 4096),
        ("qwen3p5_9b", 4, "zero3", True, 4096),
        ("qwen3p6_27b", 4, "zero3", False, 8192),  # OOM boundary
        ("qwen3p6_27b", 2, "zero2", False, 8192),  # OOM boundary
        ("qwen3p5_9b", 4, "zero3", False, 8192),   # OOM boundary
        ("qwen3p5_9b", 4, "zero2", False, 8192),   # OOM boundary
        ("qwen3p5_4b", 4, "zero2", True, 4096),
        ("qwen3p5_4b", 4, "zero3", False, 4096),
        ("qwen3p5_4b", 2, "zero2", False, 4096),
        ("qwen3p5_4b", 2, "zero3", False, 4096),
    )
    data_sha256 = sha256_file(text_source)
    jobs = []
    for idx, (model_id, gpu, zero, gc, cutoff) in enumerate(configs):
        model = models[model_id]
        sig = _architecture_signature(model)
        mbs = 1
        material = f"{model_id}::{gpu}::{zero}::{gc}::{cutoff}"
        job = {
            "schema": JOB_SCHEMA, "campaign_id": CAMPAIGN_ID, "phase_id": PHASE_ID,
            "track": "hybrid_memory_source_disjoint_prospective_v3",
            "evidence_role": "prospective_acceptance_frozen_before_outcomes",
            "design_arm": "hybrid_memory_prospective_v3",
            "model_id": model_id, "model_family": model["family"],
            "release_family": model.get("release_family", model["family"]),
            "model_path": model["path"], "tokenizer_path": model["tokenizer_path"],
            "model_parameters": int(model["actual_parameters"]), "template": model["template"],
            "train_type": "lora", "target_gbs": TARGET_GBS,
            "gpu_count": gpu, "zero": zero, "zero_stage": _zero_stage(zero),
            "gc": gc, "gradient_checkpointing": gc, "mbs": mbs,
            "gradient_accumulation_steps": max(1, TARGET_GBS // (gpu * mbs)),
            "packing": False, "offload": False, "kind": "throughput",
            "warmup_steps": 3, "measure_steps": 10, "fidelity": "formal_3plus10", "repeat": 0,
            "parallel_route_id": f"D{gpu}_Z{_zero_stage(zero)}",
            "architecture_route": sig["architecture_route"],
            "architecture_signature_sha256": sig["signature_sha256"],
            "num_full_attention_layers": int(sig["num_full_attention_layers"]),
            "num_linear_attention_layers": int(sig["num_linear_attention_layers"]),
            "dataset_id": "business_0105_inference2",
            "dataset_dir": str(text_registry.parent.resolve()),
            "dataset_registry_sha256": sha256_file(text_registry),
            "data_path": str(text_source.resolve()), "data_sha256": data_sha256,
            "dataset_profile_path": str(text_profile.resolve()),
            "dataset_profile_sha256": sha256_file(text_profile),
            "source_dataset_id": "业务前瞻0105-inference2",
            "source_dataset_sha256": data_sha256,
            "cutoff_len": cutoff, "exact_total_tokens_per_sample": cutoff,
            "all_total_tokens_exact": False, "max_samples": 64,
            "enable_liger_kernel": bool(model.get("enable_liger_kernel", True)),
            "effective_kernel_path": "fa3_orig+liger_fused_ce+adamw_torch_fused",
            "environment_overlay": overlay,
            "required_runtime_gpu_name": "NVIDIA H800", "gpu_type": "NVIDIA H800 140GB HBM3",
            "hardware_id": "local_h800_140g", "requested_gpu_pool": list(range(8)),
            "requires_external_node_idle": False, "visual_runtime_evidence_required": False,
            "job_id": _job_id("h800hyv3", material),
            "scenario_id": f"hybrid::{material}", "comparison_group": f"hybrid::{model_id}::{cutoff}",
            "target_gbs_override": TARGET_GBS,
        }
        from run_job import validate_job, resolve_dataset_dir
        validate_job(job)
        resolve_dataset_dir(job)
        jobs.append(job)
    if len(jobs) != len(configs):
        raise ValueError("hybrid V3 queue size mismatch")
    return jobs


def _vl_jobs() -> list[dict[str, Any]]:
    from prepare_h800_hybrid_vl_prospective_acceptance_v2 import VL_MECHANISMS
    from run_job import validate_job, resolve_dataset_dir

    frame_map = {}  # (model,frames,gc,mbs) -> template
    formal = read_jsonl(MATRIX_DIR / "h800_frozen_vl_combined_formal_v1.jsonl")
    templates = {
        (str(row["model_id"]), str(row["mechanism_id"])): row
        for row in formal
        if row.get("arm_id") == "real_image"
    }
    jobs = []
    for model_id in ("qwen2p5_vl_3b", "qwen3_vl_4b", "qwen3p5_4b"):
        for frames in (1, 2, 4):
            out_dir = DATA_DIR / "blind_vl_prospective_v1" / f"f{frames}"
            jsonl = out_dir / f"blind_vl_images_f{frames}.jsonl"
            registry = out_dir / "registry" / "dataset_info.json"
            profile_dir = ARTIFACT_DIR / f"h800_blind_vl_workload_profiles_v1_f{frames}"
            # Production VL shadow supports exactly three mechanism tuples.
            mechanisms = (
                ("SAFE", True, 1),
                ("NOGC", False, 1),
                ("PRESSURE", False, 4),
            )
            for gc, mbs in ((m[1], m[2]) for m in mechanisms):
                mechanism = [m[0] for m in mechanisms if m[1] == gc and m[2] == mbs][0]
                profile_path = profile_dir / f"{model_id}.blindvl.low.json"
                row = dict(templates[(model_id, mechanism)])
                material = f"{model_id}::f{frames}::gc{int(gc)}::mbs{mbs}"
                row.update({
                    "campaign_id": CAMPAIGN_ID, "phase_id": PHASE_ID,
                    "track": "vl_image_source_disjoint_prospective_v3",
                    "evidence_role": "prospective_acceptance_frozen_before_outcomes",
                    "design_arm": "vl_image_prospective_v3",
                    "dataset_id": "vl_blind_prospective_v1",
                    "dataset_dir": str(registry.parent.resolve()),
                    "dataset_registry_sha256": sha256_file(registry),
                    "data_path": str(jsonl.resolve()), "data_sha256": sha256_file(jsonl),
                    "dataset_profile_path": str(profile_path.resolve()),
                    "dataset_profile_sha256": sha256_file(profile_path),
                    "dataset_category": "blind_business_frames_synthetic_prompt",
                    "gc": gc, "gradient_checkpointing": gc, "mbs": mbs,
                    "media_tier": "low",
                    "job_id": _job_id("h800vlv3", material),
                    "pair_id": material, "scenario_id": material,
                    "requested_gpu_pool": list(range(8)),
                    "comparison_group": f"vl::{model_id}::f{frames}::gc{int(gc)}::mbs{mbs}",
                })
                validate_job(row)
                resolve_dataset_dir(row)
                jobs.append(row)
    if len(jobs) != 3 * 3 * 3:
        raise ValueError("VL V3 queue size mismatch")
    return jobs


def _hybrid_request(job: dict[str, Any]) -> dict[str, Any]:
    model_id = "qwen3_6_27b" if job["model_id"] == "qwen3p6_27b" else job["model_id"]
    return {
        "request_id": job["job_id"],
        "comparison_group": job["comparison_group"],
        "hardware_id": "h800", "model_id": model_id, "training_mode": "lora",
        "lora_rank": 32, "dataset_id": "longtail_8192", "dataset_category": "longtail",
        "target_gbs": TARGET_GBS, "cutoff_len": job["cutoff_len"],
        "gpu_count": job["gpu_count"], "physical_mbs": job["mbs"],
        "zero_stage": job["zero_stage"], "gradient_checkpointing": job["gc"],
        "packing": False, "dtype": "bf16",
    }


def _vl_request(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": job["job_id"], "comparison_group": job["comparison_group"],
        "hardware_id": "h800", "model_id": job["model_id"], "training_mode": "lora",
        "lora_rank": 32, "target_gbs": TARGET_GBS, "cutoff_len": job["cutoff_len"],
        "gpu_count": 1, "physical_mbs": job["mbs"], "zero_stage": 0,
        "gradient_checkpointing": job["gc"], "packing": False, "dtype": "bf16",
        "freeze_vision_tower": True, "freeze_multi_modal_projector": True,
        "vl_workload_profile_path": job["dataset_profile_path"],
        "vl_workload_profile_sha256": job["dataset_profile_sha256"],
    }


def _effective_sequence_tokens(job: dict[str, Any]) -> int:
    """Align prediction sequence length with the fit basis.

    The stage-1 fit learned coefficients against the real per-sample token count
    (exact_total_tokens_per_sample), not against cutoff_len.  When the dataset's
    longest sample is shorter than cutoff, no sample can fill the cutoff, so
    using cutoff over-estimates the center.  When the dataset has samples at or
    beyond cutoff, cutoff is the correct ceiling.
    """
    from pathlib import Path as _P

    cutoff = int(job.get("cutoff_len") or 0)
    profile_path = _P(str(job.get("dataset_profile_path") or ""))
    if not profile_path.is_file():
        return cutoff
    try:
        profile = read_json(profile_path)
    except Exception:
        return cutoff
    max_tokens = None
    for key in ("max_tokens_per_sample", "maximum_clipped_tokens"):
        value = profile.get(key)
        if isinstance(value, (int, float)) and value > 0:
            max_tokens = int(value)
            break
    if max_tokens is None:
        return cutoff
    return min(cutoff, max_tokens)


def _frozen_hybrid(job: dict[str, Any]) -> dict[str, Any]:
    from hybrid_memory_bridge import predict_memory_center_v3
    from common import ROOT as _r

    coefficients = read_json(ARTIFACT_DIR / "h800_hybrid_memory_artifact_v3.json")["coefficients_by_name"]
    multipliers = read_json(UPPER_V4)["hybrid_memory"]["conditional_by_model_mechanism"]
    zero = job["zero"]
    z = "0" if zero == "none" else zero.replace("zero", "")
    mechanism = f"{job['model_id']}::g{int(job['gpu_count'])}_z{z}_gc{int(bool(job['gc']))}"
    multiplier = float(multipliers[mechanism]["upper_multiplier"])
    # Align sequence length with the fit basis (real token count, not cutoff).
    effective_seq = _effective_sequence_tokens(job)
    feature_job = dict(job)
    feature_job["cutoff_len"] = effective_seq
    center = float(predict_memory_center_v3(
        {"path": job["model_path"]}, feature_job, coefficients,
        {"rank": 32, "target": "all"}, capacity_bytes=150_142_189_568,
    )["center_bytes"])
    return {
        "request_id": job["job_id"],
        "comparison_group": job["comparison_group"],
        "memory": {
            "prediction_available": True, "hybrid_model": True,
            "reserved_center_bytes": float(center),
            "risk_guard_multiplier": multiplier,
            "safety_upper_source": UPPER_V4.name,
            "admission_upper_reserved_bytes": float(center * multiplier),
            "safe_limit_bytes": float(0.95 * 150_142_189_568),
            "effective_sequence_tokens": effective_seq,
        },
        "throughput": None,
        "hybrid_shadow": {"center_version": "v3", "upper_version": "v4"},
    }


def _freeze_predictions(hybrid_jobs, vl_jobs) -> dict[str, Any]:
    from h800_resource_predictor import H800ResourcePredictor

    hybrid_predictions = [_frozen_hybrid(job) for job in hybrid_jobs]
    pilot = H800ResourcePredictor()
    hybrid_throughput = pilot.predictor.base.predict_many([_hybrid_request(j) for j in hybrid_jobs])
    vl_report = pilot.predict([_vl_request(j) for j in vl_jobs])
    return {
        "schema": "sft_h800_hybrid_vl_prospective_frozen_predictions/v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "safety_upper": {"path": str(UPPER_V4.resolve()), "sha256": sha256_file(UPPER_V4)},
        "hybrid": {"predictions": hybrid_predictions, "throughput_predictions": hybrid_throughput["predictions"]},
        "vl_image": {"predictions": vl_report["predictions"]},
    }


def prepare() -> dict[str, Any]:
    hybrid_jobs = _hybrid_jobs()
    vl_jobs = _vl_jobs()
    all_ids = [str(row["job_id"]) for row in [*hybrid_jobs, *vl_jobs]]
    existing = [jid for jid in all_ids if (ROOT / "results" / jid / "status.json").is_file()]
    if existing:
        raise RuntimeError(f"prospective outcomes already exist: {existing}")
    write_jsonl(HYBRID_QUEUE, hybrid_jobs)
    write_jsonl(VL_QUEUE, vl_jobs)
    write_jsonl(COMBINED_QUEUE, [*hybrid_jobs, *vl_jobs])
    predictions = _freeze_predictions(hybrid_jobs, vl_jobs)
    predictions["report_sha256"] = sha256_json(predictions)
    write_json(FROZEN_PREDICTIONS, predictions)
    design = {
        "schema": SCHEMA, "campaign_id": CAMPAIGN_ID, "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "usage": "source_disjoint_prospective_acceptance_only_v3",
        "gpu_training_started": False,
        "queues": {
            "combined": {"path": str(COMBINED_QUEUE.resolve()), "sha256": sha256_file(COMBINED_QUEUE), "jobs": len(hybrid_jobs) + len(vl_jobs)},
            "hybrid": {"path": str(HYBRID_QUEUE.resolve()), "sha256": sha256_file(HYBRID_QUEUE), "jobs": len(hybrid_jobs)},
            "vl_image": {"path": str(VL_QUEUE.resolve()), "sha256": sha256_file(VL_QUEUE), "jobs": len(vl_jobs)},
        },
        "frozen_predictions": {"path": str(FROZEN_PREDICTIONS.resolve()), "sha256": sha256_file(FROZEN_PREDICTIONS)},
        "source_disjointness": {
            "hybrid_fit_sources_exclude_0105_inference2": True,
            "vl_fit_sources": ["pzfj38_v113", "zltbjg_v2", "qype19_v7"],
            "vl_prospective_source": "blind_data/vl frames grid (1/2/4) synthetic prompt",
        },
        "acceptance_thresholds": {
            "false_safe_oom": 0, "exact_upper_coverage_min": 0.95,
            "admission_recall_min": 0.95, "ranking_hit_at_10_percent": 1.0,
            "worst_top1_regret_max": 0.10,
        },
        "scope": {
            "hybrid_models": ["qwen3p5_4b", "qwen3p5_9b", "qwen3p6_27b"],
            "hybrid_packing": False,
            "vl_models": ["qwen2p5_vl_3b", "qwen3_vl_4b", "qwen3p5_4b"],
            "vl_modality": "image", "vl_packing": False,
            "automatic_execution_allowed": False,
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    return design


if __name__ == "__main__":
    import json

    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
