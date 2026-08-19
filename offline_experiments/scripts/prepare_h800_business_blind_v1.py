#!/usr/bin/env python3
"""Prepare a business-data blind-test queue for the H800 memory/throughput
predictors.

Uses the downloaded 0105 short-video ASR/OCR business text as a fresh data
source and builds a queue that varies model family (qwen3/qwen3.5/qwen3.6),
packing on/off, ZeRO stage, GPU count, GC and cutoff.  The design freezes the
queue + model + data bindings so the run is reproducible and auditable.

This is a fresh generalization blind test: none of these rows were used to
fit the predictors.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import re
from pathlib import Path

from common import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    MATRIX_DIR,
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from prepare_h800_hybrid_attention_dense_stage1_v1 import (
    AUTHORIZED_GPU_IDS,
    JOB_SCHEMA,
    TARGET_GBS,
    _zero_stage,
)
from run_job import validate_job

CAMPAIGN_ID = "h800_business_blind_20260816_v1"
BLIND_DATA_DIR = ROOT / "blind_data"
SOURCES = {
    "datatest": {
        "sft": "0105_datatest.sft.jsonl",
        "profile": "0105_datatest.profile.json",
        "dataset_id": "business_0105_datatest",
    },
    "inference": {
        "sft": "0105_inference.sft.jsonl",
        "profile": "0105_inference.profile.json",
        "dataset_id": "business_0105_inference",
    },
    "inference2": {
        "sft": "0105_inference2.sft.jsonl",
        "profile": "0105_inference2.profile.json",
        "dataset_id": "business_0105_inference2",
    },
}
TEXT_SOURCE = BLIND_DATA_DIR / "text" / SOURCES["datatest"]["sft"]
TEXT_PROFILE = BLIND_DATA_DIR / "text" / SOURCES["datatest"]["profile"]
DATASET_ID = SOURCES["datatest"]["dataset_id"]

DESIGN_SCHEMA = "sft_h800_business_blind_design/v1"
QUEUE_PATH = MATRIX_DIR / "h800_business_datatest_v1.jsonl"
DESIGN_PATH = ARTIFACT_DIR / "h800_business_datatest_design_v1.json"
MODEL_INVENTORY = (
    ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_with_hybrid_v1.json"
)

# (model_id, gpu_count, zero, gc, mbs, cutoff, packing)
BLIND_MATRIX: list[tuple[str, int, str, bool, int, int, bool]] = []


def _build_matrix() -> None:
    """Model x packing x configuration matrix (the initial blind batch).

    Single-card jobs must use ZeRO-0 (no DeepSpeed); 2-card jobs use ZeRO-2/3.
    """
    BLIND_MATRIX.clear()
    models = [
        # model_id, single-gpu zeros, two-gpu zeros, gc options, cutoffs
        ("qwen3_4b", ["none"], ["zero2", "zero3"], [True, False], [2048, 4096]),
        ("qwen3_8b", ["none"], ["zero2", "zero3"], [True, False], [2048, 4096]),
        ("qwen3_14b", [], ["zero2", "zero3"], [True], [2048]),
        ("qwen3p5_4b", ["none"], ["zero2", "zero3"], [True, False], [2048, 4096]),
        ("qwen3p5_9b", [], ["zero2", "zero3"], [True], [2048]),
        ("qwen3_6_27b", [], ["zero3"], [True], [2048]),
    ]
    for model_id, z1, z2, gcs, cutoffs in models:
        rows: list[tuple] = []
        for zero in z1:
            for gc in gcs:
                for cutoff in cutoffs:
                    rows.append((1, zero, gc, cutoff))
        for zero in z2:
            for gc in gcs:
                for cutoff in cutoffs:
                    rows.append((2, zero, gc, cutoff))
        # keep the batch bounded: at most 6 configs per model
        if len(rows) > 6:
            rows = rows[:6]
        for gpu, zero, gc, cutoff in rows:
            for packing in (True, False):
                BLIND_MATRIX.append((model_id, gpu, zero, gc, 1, cutoff, packing))


def _job_id(model_id: str, gpu: int, zero: str, gc: bool, cutoff: int, packing: bool, idx: int) -> str:
    tag = f"{DATASET_ID}_{model_id}_g{gpu}_{zero}_gc{int(gc)}_c{cutoff}_p{int(packing)}"
    digest = hashlib.sha256(tag.encode()).hexdigest()[:12]
    return f"h800blind-{digest}"


HYBRID_MODELS = frozenset({"qwen3p5_4b", "qwen3p5_9b", "qwen3_6_27b"})
QWEN36_VENV = ROOT.parent / "qwen36_venv" / "lib" / "python3.11" / "site-packages"
QWEN35_CONTRACT = ARTIFACT_DIR / "h800_qwen35_runtime_contract_v1.json"
QWEN35_TILELANG_OVERLAY = "/tmp/qwen35_tilelang_overlay_0.1.12"


def _hybrid_overlay() -> dict:
    return {
        "PYTHONPATH_prepend": [
            QWEN35_TILELANG_OVERLAY,
            str(QWEN36_VENV.resolve()),
        ],
        "contract_path": str(QWEN35_CONTRACT.resolve()),
        "contract_sha256": sha256_file(QWEN35_CONTRACT),
        "variables": {
            "FLA_TILELANG": "1",
            "TILELANG_CACHE_DIR": "/tmp/qwen35_tilelang_cache_h800_precedence_v2",
        },
    }


def _build_jobs() -> list[dict]:
    inventory = read_json(MODEL_INVENTORY)
    models = {str(m["id"]): m for m in inventory["models"]}
    rows = []
    for idx, (model_id, gpu, zero, gc, mbs, cutoff, packing) in enumerate(BLIND_MATRIX):
        model = models[model_id]
        sig = _architecture_signature(model)
        job = {
            "schema": JOB_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "phase_id": "h800_business_blind_v1",
            "track": "business_blind_generalization_v1",
            "evidence_role": "blind_holdout_never_calibration",
            "design_arm": "business_blind",
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
            "packing": packing,
            "offload": False,
            "kind": "throughput",
            "warmup_steps": 3,
            "measure_steps": 10,
            "fidelity": "formal_3plus10",
            "repeat": 0,
            "parallel_route_id": f"D{gpu}_{'Z' + str(_zero_stage(zero)) if zero.startswith('zero') else 'Z0'}",
            "dataset_id": DATASET_ID,
            "dataset_dir": str(TEXT_SOURCE.parent.resolve()),
            "dataset_registry_sha256": sha256_file(
                TEXT_SOURCE.parent / "dataset_info.json"
            ),
            "data_path": str(TEXT_SOURCE.resolve()),
            "data_sha256": sha256_file(TEXT_SOURCE),
            "dataset_profile_path": str(TEXT_PROFILE.resolve()),
            "dataset_profile_sha256": sha256_file(TEXT_PROFILE),
            "source_dataset_id": "业务盲测0105",
            "source_dataset_sha256": sha256_file(TEXT_SOURCE),
            "cutoff_len": cutoff,
            "exact_total_tokens_per_sample": cutoff,
            "all_total_tokens_exact": False,
            "max_samples": 64,
            "enable_liger_kernel": bool(model.get("enable_liger_kernel", True)),
            "effective_kernel_path": "fa3_orig+liger_fused_ce+adamw_torch_fused",
            "gpu_type": "NVIDIA H800 140GB HBM3",
            "required_runtime_gpu_name": "NVIDIA H800",
            "hardware_id": "local_h800_140g",
            "requested_gpu_pool": list(AUTHORIZED_GPU_IDS),
            "architecture_route": sig["architecture_route"],
            "architecture_signature_sha256": sig["signature_sha256"],
            "num_full_attention_layers": sig["num_full_attention_layers"],
            "num_linear_attention_layers": sig["num_linear_attention_layers"],
            "model_family": model["family"],
            "release_family": model.get("release_family", model["family"]),
            "visual_runtime_evidence_required": False,
            "requires_external_node_idle": False,
            "job_id": _job_id(model_id, gpu, zero, gc, cutoff, packing, idx),
            "scenario_id": f"{model_id}__0105__c{cutoff}__g{gpu}__{zero}__gc{int(gc)}__p{int(packing)}",
            "environment_overlay": _hybrid_overlay() if model_id in HYBRID_MODELS else None,
        }
        rows.append(job)
    return rows


def _architecture_signature(model: dict) -> dict:
    # Reuse the real-checkpoint architecture signature used by the hybrid features.
    from hybrid_attention_memory_features import architecture_signature

    sig = architecture_signature(model)
    return sig


def prepare() -> dict:
    _build_matrix()
    jobs = _build_jobs()
    for job in jobs:
        validate_job(job)
    write_jsonl(QUEUE_PATH, jobs)

    design = {
        "schema": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "gpu_training_started": False,
        "approval_promoted": False,
        "scope": {
            "included": [
                "business short-video ASR/OCR text (0105)",
                "qwen3 / qwen3.5 / qwen3.6 LoRA SFT",
                "packing on and off",
                "ZeRO-2 / ZeRO-3, 1-2 GPU, GC on/off, cutoff 2048/4096",
            ],
            "excluded": ["MoE", "real media", "full fine-tune", "publication acceptance"],
        },
        "models": [
            {
                "model_id": m["id"],
                "actual_parameters": m["actual_parameters"],
                "path": m["path"],
            }
            for m in read_json(MODEL_INVENTORY)["models"]
            if m["id"] in {t[0] for t in BLIND_MATRIX}
        ],
        "queue": {
            "path": str(QUEUE_PATH.resolve()),
            "sha256": sha256_file(QUEUE_PATH),
            "jobs": len(jobs),
        },
        "data": {
            "text_source": str(TEXT_SOURCE.resolve()),
            "text_sha256": sha256_file(TEXT_SOURCE),
            "profile_sha256": sha256_file(TEXT_PROFILE),
        },
        "required_gpu_pool": {
            "gpu_ids": list(AUTHORIZED_GPU_IDS),
            "max_gpu_count_per_job": 2,
            "allow_gpu_ids_outside_pool": False,
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN_PATH, design)
    return {"queue": str(QUEUE_PATH), "jobs": len(jobs), "design": str(DESIGN_PATH)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write queue + design")
    parser.add_argument(
        "--source",
        choices=tuple(SOURCES),
        default="datatest",
        help="business data source (datatest=validate, inference*=train)",
    )
    args = parser.parse_args()
    global TEXT_SOURCE, TEXT_PROFILE, DATASET_ID, QUEUE_PATH, DESIGN_PATH
    spec = SOURCES[args.source]
    TEXT_SOURCE = BLIND_DATA_DIR / "text" / spec["sft"]
    TEXT_PROFILE = BLIND_DATA_DIR / "text" / spec["profile"]
    DATASET_ID = spec["dataset_id"]
    QUEUE_PATH = MATRIX_DIR / f"h800_business_{args.source}_v1.jsonl"
    DESIGN_PATH = ARTIFACT_DIR / f"h800_business_{args.source}_design_v1.json"
    _build_matrix()
    jobs = _build_jobs()
    print(json.dumps({"jobs": len(jobs), "preview": jobs[:2]}, ensure_ascii=False, indent=1)[:1500])
    if args.write:
        result = prepare()
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
