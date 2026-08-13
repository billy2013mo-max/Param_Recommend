#!/usr/bin/env python3
"""Materialize the staged Qwen3.5 and real-image VL supplement campaign.

The campaign contains a fail-closed semantic canary followed by a formal
calibration queue.  It writes frozen artifacts only; it neither promotes an
approval nor starts GPU work.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
from typing import Any, Iterable

from common import (
    ARTIFACT_DIR,
    DATA_DIR,
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from inventory_models import inventory_model


CAMPAIGN_ID = "h800_qwen35_vl_supplement_20260809_v1"
CANARY_PHASE_ID = "h800_qwen35_vl_supplement_canary_v1"
FORMAL_PHASE_ID = "h800_qwen35_vl_supplement_formal_v1"
JOB_SCHEMA = "sft_h800_qwen35_vl_supplement_job/v1"
DESIGN_SCHEMA = "sft_h800_qwen35_vl_supplement_design/v1"
GENERATED_AT_UTC = "2026-08-09T08:15:00+00:00"
TARGET_GBS = 64

PROFILE_MANIFEST = (
    ARTIFACT_DIR / "h800_qwen35_vl_supplement_processor_profiles_manifest_v1.json"
)
TEXT_PROFILE_MANIFEST = ARTIFACT_DIR / "h800_bounded_memory_v2_qwen35_profiles_v1.json"
MEDIA_MANIFEST = ARTIFACT_DIR / "h800_vl_calibration_media_manifest_v1.json"
VL_DATA = DATA_DIR / "vl_calibration_v1" / "vl_pzfj38_calibration_v1.jsonl"
RUNTIME_CONTRACT = ARTIFACT_DIR / "h800_qwen35_runtime_contract_v1.json"
INVENTORY = ARTIFACT_DIR / "h800_qwen35_vl_supplement_model_inventory_v1.json"
FROZEN_SELECTION = (
    ARTIFACT_DIR / "h800_qwen35_vl_supplement_frozen_selection_v1.json"
)
CANARY_QUEUE = ROOT / "matrix" / "h800_qwen35_vl_supplement_canary_v1.jsonl"
FORMAL_QUEUE = ROOT / "matrix" / "h800_qwen35_vl_supplement_formal_v1.jsonl"
CANARY_DESIGN = ARTIFACT_DIR / "h800_qwen35_vl_supplement_canary_design_v1.json"
FORMAL_DESIGN = ARTIFACT_DIR / "h800_qwen35_vl_supplement_formal_design_v1.json"
CANARY_MANIFEST = (
    ARTIFACT_DIR / "h800_qwen35_vl_supplement_canary_queue_manifest_v1.json"
)
FORMAL_MANIFEST = (
    ARTIFACT_DIR / "h800_qwen35_vl_supplement_formal_queue_manifest_v1.json"
)
STAGING_CONFIG = (
    ROOT
    / "qwen35_vl_staging"
    / "experiment.h800_qwen35_vl_supplement_v1.json"
)

MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "qwen3p5_0p8b",
        "nominal_scale_b": 0.8,
        "path": "/wanqing-models/Qwen3.5-0.8B",
        "tokenizer_path": "/wanqing-models/Qwen3.5-0.8B",
        "family": "qwen3_5",
        "template": "qwen3_5_nothink",
        "train_types": ["lora", "full"],
        "image_min_pixels": 64 * 64,
        "enable_liger_kernel": True,
    },
    {
        "id": "qwen3p5_4b",
        "nominal_scale_b": 4,
        "path": "/wanqing-models/Qwen3.5-4B",
        "tokenizer_path": "/wanqing-models/Qwen3.5-4B",
        "family": "qwen3_5",
        "template": "qwen3_5_nothink",
        "train_types": ["lora", "full"],
        "image_min_pixels": 64 * 64,
        "enable_liger_kernel": True,
    },
    {
        "id": "qwen3p5_9b",
        "nominal_scale_b": 9,
        "path": "/wanqing-models/Qwen3.5-9B",
        "tokenizer_path": "/wanqing-models/Qwen3.5-9B",
        "family": "qwen3_5",
        "template": "qwen3_5_nothink",
        "train_types": ["lora", "full"],
        "image_min_pixels": 64 * 64,
        "enable_liger_kernel": True,
    },
    {
        "id": "qwen3p5_27b",
        "nominal_scale_b": 27,
        "path": "/wanqing-models/Qwen3.5-27B",
        "tokenizer_path": "/wanqing-models/Qwen3.5-27B",
        "family": "qwen3_5",
        "template": "qwen3_5_nothink",
        "train_types": ["lora", "full"],
        "image_min_pixels": 64 * 64,
        "enable_liger_kernel": True,
    },
    {
        "id": "qwen2p5_vl_3b",
        "nominal_scale_b": 3,
        "path": "/wanqing-models/Qwen2.5-VL-3B-Instruct",
        "tokenizer_path": "/wanqing-models/Qwen2.5-VL-3B-Instruct",
        "family": "qwen2p5_vl",
        "template": "qwen2_vl",
        "train_types": ["lora"],
        "image_min_pixels": 56 * 56,
        "enable_liger_kernel": True,
    },
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
    },
    {
        "id": "qwen3_vl_2b",
        "nominal_scale_b": 2,
        "path": "/wanqing-models/Qwen3-VL-2B-Instruct",
        "tokenizer_path": "/wanqing-models/Qwen3-VL-2B-Instruct",
        "family": "qwen3_vl",
        "template": "qwen3_vl",
        "train_types": ["lora"],
        "image_min_pixels": 64 * 64,
        "enable_liger_kernel": False,
    },
    {
        "id": "qwen3_vl_4b",
        "nominal_scale_b": 4,
        "path": "/wanqing-models/Qwen3-VL-4B-Instruct",
        "tokenizer_path": "/wanqing-models/Qwen3-VL-4B-Instruct",
        "family": "qwen3_vl",
        "template": "qwen3_vl",
        "train_types": ["lora"],
        "image_min_pixels": 64 * 64,
        "enable_liger_kernel": False,
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
    },
    {
        "id": "qwen3_vl_32b",
        "nominal_scale_b": 32,
        "path": "/wanqing-models/Qwen3-VL-32B-Instruct",
        "tokenizer_path": "/wanqing-models/Qwen3-VL-32B-Instruct",
        "family": "qwen3_vl",
        "template": "qwen3_vl",
        "train_types": ["lora"],
        "image_min_pixels": 64 * 64,
        "enable_liger_kernel": False,
    },
    {
        "id": "qwen3_vl_30b_a3b",
        "nominal_scale_b": 30,
        "path": "/wanqing-models/Qwen3-VL-30B-A3B-Instruct",
        "tokenizer_path": "/wanqing-models/Qwen3-VL-30B-A3B-Instruct",
        "family": "qwen3_vl_moe",
        "template": "qwen3_vl",
        "train_types": ["lora"],
        "image_min_pixels": 64 * 64,
        "enable_liger_kernel": False,
    },
)

LANGUAGE_ONLY = {
    "freeze_vision_tower": True,
    "freeze_multi_modal_projector": True,
    "freeze_language_model": False,
}
TRAIN_SCOPES = {
    "language_only": LANGUAGE_ONLY,
    "projector_plus_language": {
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": False,
        "freeze_language_model": False,
    },
    "all_visual_plus_language": {
        "freeze_vision_tower": False,
        "freeze_multi_modal_projector": False,
        "freeze_language_model": False,
    },
}

MECHANISMS = {
    "C1": {"gpu_count": 1, "zero": "none", "gc": True, "mbs": 1},
    "C3": {"gpu_count": 2, "zero": "zero3", "gc": True, "mbs": 1},
    "C4": {"gpu_count": 2, "zero": "zero3", "gc": True, "mbs": 2},
    "C5": {"gpu_count": 2, "zero": "zero2", "gc": False, "mbs": 1},
}


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _environment_overlay() -> dict[str, Any]:
    contract = read_json(RUNTIME_CONTRACT)
    if contract.get("schema") != "sft_h800_qwen35_runtime_contract/v1":
        raise ValueError("Qwen3.5 runtime contract schema drifted")
    return {
        "PYTHONPATH_prepend": list(contract["environment"]["PYTHONPATH_prepend"]),
        "variables": dict(contract["environment"]["variables"]),
        "contract_path": str(RUNTIME_CONTRACT.resolve()),
        "contract_sha256": sha256_file(RUNTIME_CONTRACT),
    }


def _inventory() -> dict[str, Any]:
    rows = [inventory_model(dict(spec)) for spec in MODEL_SPECS]
    report: dict[str, Any] = {
        "schema": "sft_h800_qwen35_vl_supplement_model_inventory/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": GENERATED_AT_UTC,
        "gpu_training_started": False,
        "fixed_lora": {"rank": 32, "alpha": 32, "dropout": 0.0, "target": "all"},
        "models": rows,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(INVENTORY, report)
    return report


def _profiles() -> dict[tuple[str, str], dict[str, Any]]:
    manifest = read_json(PROFILE_MANIFEST)
    if (
        manifest.get("schema")
        != "sft_h800_qwen35_vl_supplement_processor_profiles/v1"
        or manifest.get("all_actual_processor_checks_passed") is not True
        or manifest.get("all_alias_equivalence_checks_passed") is not True
    ):
        raise ValueError("supplement processor profiles are not complete")
    rows = {}
    for row in manifest["profiles"]:
        path = Path(row["path"])
        if row["sha256"] != sha256_file(path):
            raise ValueError(f"processor profile drifted: {path}")
        rows[(str(row["model_id"]), str(row["tier"]))] = row
    expected = {(spec["id"], tier) for spec in MODEL_SPECS for tier in ("low", "high")}
    if set(rows) != expected:
        raise ValueError("processor profile model/tier domain is incomplete")
    return rows


def _text_profiles() -> list[dict[str, Any]]:
    report = read_json(TEXT_PROFILE_MANIFEST)
    if (
        report.get("schema") != "sft_h800_bounded_memory_v2_qwen35_profiles/v1"
        or report.get("gpu_training_started") is not False
        or len(report.get("profiles") or []) != 3
    ):
        raise ValueError("Qwen3.5 text profile binding is incomplete")
    for row in report["profiles"]:
        if row["profile_sha256"] != sha256_file(Path(row["profile_path"])):
            raise ValueError(f"Qwen3.5 text profile drifted: {row['profile_path']}")
        if row["data_sha256"] != sha256_file(Path(row["data_path"])):
            raise ValueError(f"Qwen3.5 text data drifted: {row['data_path']}")
    return list(report["profiles"])


def _zero_stage(zero: str) -> int:
    return 0 if zero == "none" else int(zero[-1])


def _base_job(
    *,
    model: dict[str, Any],
    phase_id: str,
    track: str,
    evidence_role: str,
    scenario_id: str,
    train_type: str,
    mechanism_id: str,
    gpu_count: int,
    zero: str,
    gc: bool,
    mbs: int,
    train_scope_id: str,
    warmup_steps: int,
    measure_steps: int,
    repeat: int,
) -> dict[str, Any]:
    denominator = gpu_count * mbs
    if TARGET_GBS % denominator:
        raise ValueError(f"GBS {TARGET_GBS} is not divisible by {denominator}")
    spec = next(row for row in MODEL_SPECS if row["id"] == model["id"])
    row: dict[str, Any] = {
        "schema": JOB_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": phase_id,
        "track": track,
        "evidence_role": evidence_role,
        "scenario_id": scenario_id,
        "mechanism_id": mechanism_id,
        "model_id": model["id"],
        "model_family": model["family"],
        "model_path": model["path"],
        "tokenizer_path": model["tokenizer_path"],
        "model_parameters": int(model["actual_parameters"]),
        "template": model["template"],
        "train_type": train_type,
        "target_gbs": TARGET_GBS,
        "gpu_count": gpu_count,
        "zero": zero,
        "zero_stage": _zero_stage(zero),
        "gc": gc,
        "gradient_checkpointing": gc,
        "mbs": mbs,
        "gradient_accumulation_steps": TARGET_GBS // denominator,
        "packing": False,
        "offload": False,
        "kind": "throughput",
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "fidelity": f"formal_{warmup_steps}plus{measure_steps}",
        "repeat": repeat,
        "train_scope_id": train_scope_id,
        **TRAIN_SCOPES[train_scope_id],
        "enable_liger_kernel": bool(spec["enable_liger_kernel"]),
        "effective_kernel_path": (
            "fa3_orig+liger_fused_ce+adamw_torch_fused"
            if spec["enable_liger_kernel"]
            else "fa3_orig+native_ce+adamw_torch_fused"
        ),
        "gpu_type": "NVIDIA H800 140GB HBM3",
        "required_runtime_gpu_name": "NVIDIA H800",
        "hardware_id": "local_h800_140g",
        "parallel_class": "gpu_partitionable",
        "requires_external_node_idle": False,
        "declared_model_manifest_path": str(INVENTORY.resolve()),
        "declared_model_manifest_sha256": sha256_file(INVENTORY),
    }
    if model["family"] == "qwen3_5":
        row["environment_overlay"] = _environment_overlay()
    return row


def _image_job(
    *,
    model: dict[str, Any],
    profiles: dict[tuple[str, str], dict[str, Any]],
    phase_id: str,
    track: str,
    evidence_role: str,
    tier: str,
    mechanism_id: str,
    mechanism: dict[str, Any],
    train_scope_id: str = "language_only",
    warmup_steps: int = 2,
    measure_steps: int = 8,
    repeat: int = 0,
    max_samples: int = 1000,
) -> dict[str, Any]:
    scenario_id = (
        f"pzfj38__{model['id']}__{tier}__{train_scope_id}__{mechanism_id}"
    )
    row = _base_job(
        model=model,
        phase_id=phase_id,
        track=track,
        evidence_role=evidence_role,
        scenario_id=scenario_id,
        train_type="lora",
        mechanism_id=mechanism_id,
        gpu_count=int(mechanism["gpu_count"]),
        zero=str(mechanism["zero"]),
        gc=bool(mechanism["gc"]),
        mbs=int(mechanism["mbs"]),
        train_scope_id=train_scope_id,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        repeat=repeat,
    )
    profile = profiles[(str(model["id"]), tier)]
    row.update(
        {
            "dataset_id": "vl_pzfj38_calibration_v1",
            "profile_id": f"pzfj38-first1000-{model['id']}-{tier}",
            "dataset_category": "vl_full_reference_image_quality",
            "data_path": str(VL_DATA.resolve()),
            "data_sha256": sha256_file(VL_DATA),
            "dataset_profile_path": str(Path(profile["path"]).resolve()),
            "dataset_profile_sha256": profile["sha256"],
            "media_manifest_path": str(MEDIA_MANIFEST.resolve()),
            "media_manifest_sha256": sha256_file(MEDIA_MANIFEST),
            "media_tier": tier,
            "cutoff_len": 8192,
            "max_samples": max_samples,
            "image_min_pixels": int(profile["image_min_pixels"]),
            "image_max_pixels": int(profile["image_max_pixels"]),
            "expected_images_per_sample": 2,
            "visual_runtime_evidence_required": True,
            "calibration_partition": {
                "role": "canary_excluded" if "canary" in evidence_role else "calibration",
                "split_unit_id": f"pzfj38-{model['id']}-{tier}",
                "policy": (
                    "software_and_runtime_semantics_only_not_fit_or_acceptance"
                    if "canary" in evidence_role
                    else "fit_only_never_acceptance_v1"
                ),
            },
        }
    )
    row["job_id"] = stable_id("h800q35vl", row)
    return row


def _text_job(
    *,
    model: dict[str, Any],
    profile: dict[str, Any],
    train_type: str,
    mechanism_id: str,
    mechanism: dict[str, Any],
) -> dict[str, Any]:
    scenario_id = f"{profile['source_profile_id']}__{model['id']}__{train_type}__{mechanism_id}"
    row = _base_job(
        model=model,
        phase_id=FORMAL_PHASE_ID,
        track="qwen35_text_cross_scale",
        evidence_role="text_cross_scale_calibration",
        scenario_id=scenario_id,
        train_type=train_type,
        mechanism_id=mechanism_id,
        gpu_count=int(mechanism["gpu_count"]),
        zero=str(mechanism["zero"]),
        gc=bool(mechanism["gc"]),
        mbs=int(mechanism["mbs"]),
        train_scope_id="language_only",
        warmup_steps=3,
        measure_steps=10,
        repeat=0,
    )
    statistics = profile["statistics"]
    row.update(
        {
            "dataset_id": profile["dataset_id"],
            "profile_id": profile["source_profile_id"],
            "dataset_category": (
                "short"
                if "rare_tail_short" in profile["source_profile_id"]
                else "longcontext"
                if "broad_nontruncated" in profile["source_profile_id"]
                else "longtail"
            ),
            "data_path": profile["data_path"],
            "data_sha256": profile["data_sha256"],
            "dataset_profile_path": profile["profile_path"],
            "dataset_profile_sha256": profile["profile_sha256"],
            "cutoff_len": (
                4096
                if "rare_tail_short" in profile["source_profile_id"]
                else 32768
                if "broad_nontruncated" in profile["source_profile_id"]
                else 2048
            ),
            "raw_profile_max": int(statistics["maximum_total_tokens"]),
            "max_samples": int(statistics["rows"]),
            "calibration_partition": {
                "role": "calibration",
                "split_unit_id": profile["source_profile_id"],
                "policy": "qwen35_cross_scale_source_grouped_fit_v1",
            },
        }
    )
    row["job_id"] = stable_id("h800q35vl", row)
    return row


def _canary_jobs(
    models: dict[str, dict[str, Any]],
    profiles: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    safe = {
        "qwen3p5_0p8b": MECHANISMS["C1"],
        "qwen3p5_4b": MECHANISMS["C1"],
        "qwen3p5_9b": MECHANISMS["C3"],
        "qwen3p5_27b": {"gpu_count": 4, "zero": "zero3", "gc": True, "mbs": 1},
        "qwen2p5_vl_3b": MECHANISMS["C1"],
        "qwen2p5_vl_7b": MECHANISMS["C3"],
        "qwen3_vl_2b": MECHANISMS["C1"],
        "qwen3_vl_4b": MECHANISMS["C1"],
        "qwen3_vl_8b": MECHANISMS["C3"],
        "qwen3_vl_32b": {"gpu_count": 4, "zero": "zero3", "gc": True, "mbs": 1},
        "qwen3_vl_30b_a3b": {"gpu_count": 4, "zero": "zero3", "gc": True, "mbs": 1},
    }
    jobs = [
        _image_job(
            model=models[model_id],
            profiles=profiles,
            phase_id=CANARY_PHASE_ID,
            track="compatibility_canary",
            evidence_role="real_image_language_only_canary",
            tier="high",
            mechanism_id="SAFE",
            mechanism=mechanism,
            warmup_steps=0,
            measure_steps=2,
            max_samples=64,
        )
        for model_id, mechanism in safe.items()
    ]
    for model_id in ("qwen3p5_4b", "qwen3_vl_4b"):
        for scope in ("projector_plus_language", "all_visual_plus_language"):
            jobs.append(
                _image_job(
                    model=models[model_id],
                    profiles=profiles,
                    phase_id=CANARY_PHASE_ID,
                    track="train_scope_canary",
                    evidence_role="real_image_train_scope_canary",
                    tier="high",
                    mechanism_id="SAFE",
                    mechanism=MECHANISMS["C1"],
                    train_scope_id=scope,
                    warmup_steps=0,
                    measure_steps=2,
                    max_samples=64,
                )
            )
    if len(jobs) != 15 or len({row["job_id"] for row in jobs}) != 15:
        raise ValueError("semantic canary must contain exactly 15 unique jobs")
    return jobs


def _formal_jobs(
    models: dict[str, dict[str, Any]],
    profiles: dict[tuple[str, str], dict[str, Any]],
    text_profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    text_mechanisms = {
        "qwen3p5_0p8b": (
            ("T_SAFE", {"gpu_count": 1, "zero": "none", "gc": False, "mbs": 2}),
            ("T_PRESSURE", {"gpu_count": 1, "zero": "none", "gc": False, "mbs": 8}),
        ),
        "qwen3p5_4b": (
            ("T_SAFE", {"gpu_count": 1, "zero": "none", "gc": True, "mbs": 1}),
            ("T_PRESSURE", {"gpu_count": 1, "zero": "none", "gc": True, "mbs": 4}),
        ),
        "qwen3p5_9b": (
            ("T_SAFE", {"gpu_count": 2, "zero": "zero3", "gc": True, "mbs": 1}),
            ("T_PRESSURE", {"gpu_count": 2, "zero": "zero3", "gc": True, "mbs": 2}),
        ),
        "qwen3p5_27b": (
            ("T_SAFE", {"gpu_count": 4, "zero": "zero3", "gc": True, "mbs": 1}),
            ("T_PRESSURE", {"gpu_count": 4, "zero": "zero3", "gc": True, "mbs": 2}),
        ),
    }
    for model_id, mechanism_rows in text_mechanisms.items():
        for profile in text_profiles:
            for mechanism_id, mechanism in mechanism_rows:
                jobs.append(
                    _text_job(
                        model=models[model_id],
                        profile=profile,
                        train_type="lora",
                        mechanism_id=mechanism_id,
                        mechanism=mechanism,
                    )
                )
    short_profile = next(
        row for row in text_profiles if "rare_tail_short" in row["source_profile_id"]
    )
    full_mechanisms = {
        "qwen3p5_0p8b": {"gpu_count": 1, "zero": "none", "gc": True, "mbs": 1},
        "qwen3p5_4b": {"gpu_count": 2, "zero": "zero3", "gc": True, "mbs": 1},
        "qwen3p5_9b": {"gpu_count": 4, "zero": "zero3", "gc": True, "mbs": 1},
        "qwen3p5_27b": {"gpu_count": 4, "zero": "zero3", "gc": True, "mbs": 1},
    }
    for model_id, mechanism in full_mechanisms.items():
        jobs.append(
            _text_job(
                model=models[model_id],
                profile=short_profile,
                train_type="full",
                mechanism_id="FULL_ANCHOR",
                mechanism=mechanism,
            )
        )

    for model_id in ("qwen2p5_vl_7b", "qwen3_vl_8b"):
        for tier in ("low", "high"):
            for mechanism_id in ("C1", "C3", "C4", "C5"):
                jobs.append(
                    _image_job(
                        model=models[model_id],
                        profiles=profiles,
                        phase_id=FORMAL_PHASE_ID,
                        track="vl_anchor_mechanism_calibration",
                        evidence_role="real_image_anchor_mechanism_fit",
                        tier=tier,
                        mechanism_id=mechanism_id,
                        mechanism=MECHANISMS[mechanism_id],
                    )
                )

    q35_image_mechanisms = {
        "qwen3p5_0p8b": (
            ("I_SAFE", MECHANISMS["C1"]),
            ("I_PRESSURE", {"gpu_count": 1, "zero": "none", "gc": False, "mbs": 4}),
        ),
        "qwen3p5_4b": (
            ("I_SAFE", MECHANISMS["C1"]),
            ("I_PRESSURE", {"gpu_count": 1, "zero": "none", "gc": False, "mbs": 4}),
        ),
        "qwen3p5_9b": (
            ("I_SAFE", MECHANISMS["C3"]),
            ("I_PRESSURE", {"gpu_count": 2, "zero": "zero2", "gc": False, "mbs": 2}),
        ),
        "qwen3p5_27b": (
            ("I_SAFE", {"gpu_count": 4, "zero": "zero3", "gc": True, "mbs": 1}),
            ("I_PRESSURE", {"gpu_count": 4, "zero": "zero2", "gc": False, "mbs": 2}),
        ),
    }
    for model_id, mechanism_rows in q35_image_mechanisms.items():
        for tier in ("low", "high"):
            for mechanism_id, mechanism in mechanism_rows:
                jobs.append(
                    _image_job(
                        model=models[model_id],
                        profiles=profiles,
                        phase_id=FORMAL_PHASE_ID,
                        track="qwen35_real_image_cross_scale",
                        evidence_role="real_image_qwen35_cross_scale_fit",
                        tier=tier,
                        mechanism_id=mechanism_id,
                        mechanism=mechanism,
                    )
                )

    for model_id in ("qwen2p5_vl_3b", "qwen3_vl_2b", "qwen3_vl_4b"):
        for tier in ("low", "high"):
            for mechanism_id, mechanism in (
                ("S_SAFE", MECHANISMS["C1"]),
                ("S_PRESSURE", {"gpu_count": 1, "zero": "none", "gc": False, "mbs": 4}),
            ):
                jobs.append(
                    _image_job(
                        model=models[model_id],
                        profiles=profiles,
                        phase_id=FORMAL_PHASE_ID,
                        track="vl_dense_scale_transfer",
                        evidence_role="real_image_dense_scale_transfer_fit",
                        tier=tier,
                        mechanism_id=mechanism_id,
                        mechanism=mechanism,
                    )
                )
    for model_id in ("qwen3_vl_32b", "qwen3_vl_30b_a3b"):
        for tier in ("low", "high"):
            jobs.append(
                _image_job(
                    model=models[model_id],
                    profiles=profiles,
                    phase_id=FORMAL_PHASE_ID,
                    track=(
                        "vl_moe_transfer"
                        if model_id == "qwen3_vl_30b_a3b"
                        else "vl_dense_scale_transfer"
                    ),
                    evidence_role="real_image_large_model_transfer_fit",
                    tier=tier,
                    mechanism_id="L_SAFE",
                    mechanism={"gpu_count": 4, "zero": "zero3", "gc": True, "mbs": 1},
                )
            )

    for model_id in ("qwen3p5_4b", "qwen3_vl_4b"):
        for scope in ("projector_plus_language", "all_visual_plus_language"):
            jobs.append(
                _image_job(
                    model=models[model_id],
                    profiles=profiles,
                    phase_id=FORMAL_PHASE_ID,
                    track="visual_train_scope_ablation",
                    evidence_role="real_image_train_scope_fit",
                    tier="high",
                    mechanism_id="SCOPE_SAFE",
                    mechanism=MECHANISMS["C1"],
                    train_scope_id=scope,
                )
            )

    repeat_specs = (
        ("qwen3p5_4b", MECHANISMS["C1"]),
        ("qwen2p5_vl_7b", MECHANISMS["C3"]),
        ("qwen3_vl_8b", MECHANISMS["C3"]),
    )
    for model_id, mechanism in repeat_specs:
        for repeat in (1, 2):
            jobs.append(
                _image_job(
                    model=models[model_id],
                    profiles=profiles,
                    phase_id=FORMAL_PHASE_ID,
                    track="repeat_variance_diagnostic",
                    evidence_role="real_image_repeat_diagnostic",
                    tier="high",
                    mechanism_id="REPEAT_SAFE",
                    mechanism=mechanism,
                    repeat=repeat,
                )
            )
    if len(jobs) != 86 or len({row["job_id"] for row in jobs}) != 86:
        raise ValueError(f"formal supplement must contain 86 unique jobs, got {len(jobs)}")
    return jobs


def _frozen_selection(canary: list[dict[str, Any]], formal: list[dict[str, Any]]) -> dict[str, Any]:
    by_track = Counter(str(row["track"]) for row in formal)
    report: dict[str, Any] = {
        "schema": "sft_h800_qwen35_vl_supplement_frozen_selection/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": GENERATED_AT_UTC,
        "generated_before_gpu": True,
        "gpu_training_started": False,
        "selection_basis": "preregistered coverage matrix, not model recommendation",
        "prediction_availability": {
            "qwen35_4b_text_historical_prediction": "diagnostic_only_not_reused_as_acceptance",
            "qwen35_other_scales": "unavailable",
            "real_image_memory_head": "unavailable",
            "real_image_throughput_ranker": "unavailable",
            "visual_train_scope_head": "unavailable",
        },
        "automatic_execution_from_prediction": False,
        "right_censoring_policy": "success is exact; confirmed CUDA OOM is a lower bound, never a point target",
        "repeat_policy": "repeats estimate run variance and collapse to one physical arm before model fitting",
        "canary_jobs": len(canary),
        "formal_jobs": len(formal),
        "formal_track_counts": dict(sorted(by_track.items())),
        "excluded_from_this_batch": {
            "video": "no frozen real-video dataset and no bound video workload profile",
            "multimodal_packing": "packing semantics have no accepted real-image training path",
            "qwen3p5_moe": "no local Qwen3.5 MoE checkpoint",
            "qwen3p5_27b_fp8": "FP8 checkpoint is outside the BF16 SFT calibration domain",
        },
    }
    report["report_sha256"] = sha256_json(report)
    write_json(FROZEN_SELECTION, report)
    return report


def _write_stage(
    *, phase_id: str, queue_path: Path, design_path: Path, manifest_path: Path, jobs: list[dict[str, Any]], prerequisite: dict[str, Any] | None
) -> None:
    write_jsonl(queue_path, jobs)
    jobs_dir = ARTIFACT_DIR / f"{phase_id}_jobs"
    for row in jobs:
        write_json(jobs_dir / f"{row['job_id']}.json", row)
    tracks = Counter(str(row["track"]) for row in jobs)
    classifications = {
        "success": "exact observed memory/throughput evidence",
        "oom": "right-censored memory lower bound; valid only in formal phase",
        "software_or_infrastructure_failure": "not calibration evidence and blocks the next gate",
    }
    design: dict[str, Any] = {
        "schema": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": phase_id,
        "generated_at_utc": GENERATED_AT_UTC,
        "status": "frozen_before_gpu_waiting_for_exact_approval",
        "gpu_training_started": False,
        "execution_authorized": False,
        "publication_allowed": False,
        "fit_allowed": phase_id == FORMAL_PHASE_ID,
        "acceptance_allowed": False,
        "prerequisite": prerequisite,
        "required_gpu_pool": {
            "gpu_ids": list(range(8)),
            "max_gpu_count": 4,
            "preemption_allowed": False,
        },
        "queue": {
            **_binding(queue_path),
            "job_count": len(jobs),
            "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in jobs),
            "ordered_job_ids": [str(row["job_id"]) for row in jobs],
            "ordered_job_payload_sha256": sha256_json(jobs),
        },
        "track_counts": dict(sorted(tracks.items())),
        "model_inventory": _binding(INVENTORY),
        "processor_profiles": _binding(PROFILE_MANIFEST),
        "frozen_selection": _binding(FROZEN_SELECTION),
        "outcome_semantics": classifications,
        "measurement_contract": {
            "canary": "0 warmup + 2 measured optimizer steps; semantics only",
            "formal_text": "3 warmup + 10 measured optimizer steps",
            "formal_image": "2 warmup + 8 measured optimizer steps",
            "token_source": "consumed_token_ledger/v1",
            "memory_source": "max_memory_reserved over all ranks plus telemetry",
        },
        "interpretation_contract": {
            "model_size_and_media_features_are_inputs_not_model_routes": True,
            "mechanism_fields_are_inputs_not_separate_models": True,
            "oom_is_right_censored": True,
            "repeats_are_not_independent_samples": True,
            "canary_rows_are_excluded_from_fit": True,
            "formal_rows_are_fit_only_not_acceptance": True,
            "publication_requires_source_disjoint_prospective_acceptance": True,
        },
        "source_bindings": {
            "profile_manifest": _binding(PROFILE_MANIFEST),
            "text_profile_manifest": _binding(TEXT_PROFILE_MANIFEST),
            "media_manifest": _binding(MEDIA_MANIFEST),
            "vl_data": _binding(VL_DATA),
            "runtime_contract": _binding(RUNTIME_CONTRACT),
            "dataset_registry": _binding(DATA_DIR / "dataset_info.json"),
            "preparer": _binding(Path(__file__).resolve()),
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(design_path, design)
    manifest: dict[str, Any] = {
        "schema": "sft_h800_qwen35_vl_supplement_queue_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": phase_id,
        "generated_at_utc": GENERATED_AT_UTC,
        "gpu_training_started": False,
        "design": _binding(design_path),
        "queue": {
            **_binding(queue_path),
            "job_count": len(jobs),
            "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in jobs),
            "ordered_job_ids": [str(row["job_id"]) for row in jobs],
            "ordered_job_payload_sha256": sha256_json(jobs),
        },
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(manifest_path, manifest)


def _experiment_config() -> dict[str, Any]:
    current = read_json(ROOT / "config" / "experiment.json")
    config = copy.deepcopy(current)
    config["training_scope"] = {
        "phase_id": "h800_qwen35_vl_supplement_v1",
        "model_ids": sorted(spec["id"] for spec in MODEL_SPECS),
        "gpu_ids": list(range(8)),
        "max_gpu_count": 4,
        "exclusive_node_gpu_ids": list(range(8)),
        "deferred_model_ids": [],
        "stage": "sft",
        "precision": "bf16",
        "gpu_type": "NVIDIA H800 140GB HBM3",
        "gpu_counts": [1, 2, 4],
        "global_batch_sizes": [64],
        "gradient_checkpointing": [False, True],
        "zero_by_gpu_count": {
            "1": ["none"],
            "2": ["zero2", "zero3"],
            "4": ["zero2", "zero3"],
        },
        "hardware_followups": {
            "memory": "fit one resource model with model/media/scope/mechanism inputs",
            "followup": "formal rows are fit only; publication needs a new source-disjoint acceptance campaign",
        },
    }
    config["measurement"]["performance_parallelism"] = "disjoint_gpu_masks"
    config["measurement"]["scheduler_order_policy"] = "parallel_queue"
    config["datasets"] = [
        {"id": "vl_pzfj38_calibration_v1", "category": "real_image", "target_cutoffs": [8192]},
        {"id": "fresh_s3_rare_tail_short_v1__qwen35", "category": "short", "target_cutoffs": [4096]},
        {"id": "fresh_s3_broad_nontruncated_longtail_v1__qwen35", "category": "longcontext", "target_cutoffs": [32768]},
        {"id": "fresh_s3_concentrated_truncated_long_v1__qwen35", "category": "longtail", "target_cutoffs": [2048]},
    ]
    config["supplement_policy"] = {
        "campaign_id": CAMPAIGN_ID,
        "canary_phase_id": CANARY_PHASE_ID,
        "formal_phase_id": FORMAL_PHASE_ID,
        "packing": False,
        "offload": False,
        "formal_requires_canary_acceptance": True,
    }
    write_json(STAGING_CONFIG, config)
    return config


def prepare() -> dict[str, Any]:
    for path in (
        PROFILE_MANIFEST,
        TEXT_PROFILE_MANIFEST,
        MEDIA_MANIFEST,
        VL_DATA,
        RUNTIME_CONTRACT,
        DATA_DIR / "dataset_info.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    profile_rows = _profiles()
    text_profiles = _text_profiles()
    inventory = _inventory()
    models = {str(row["id"]): row for row in inventory["models"]}
    canary = _canary_jobs(models, profile_rows)
    formal = _formal_jobs(models, profile_rows, text_profiles)
    selection = _frozen_selection(canary, formal)
    _write_stage(
        phase_id=CANARY_PHASE_ID,
        queue_path=CANARY_QUEUE,
        design_path=CANARY_DESIGN,
        manifest_path=CANARY_MANIFEST,
        jobs=canary,
        prerequisite={
            "upstream_queue": "matrix/h800_unified_resource_evidence_jobs_v1.jsonl",
            "required_terminal_eligible": 220,
            "required_active_upstream_processes": 0,
            "required_idle_grace_polls": 3,
        },
    )
    _write_stage(
        phase_id=FORMAL_PHASE_ID,
        queue_path=FORMAL_QUEUE,
        design_path=FORMAL_DESIGN,
        manifest_path=FORMAL_MANIFEST,
        jobs=formal,
        prerequisite={
            "acceptance_path": "artifacts/h800_qwen35_vl_supplement_canary_acceptance_v1.json",
            "required_all_passed": True,
        },
    )
    config = _experiment_config()
    return {
        "campaign_id": CAMPAIGN_ID,
        "inventory": _binding(INVENTORY),
        "frozen_selection": {**_binding(FROZEN_SELECTION), "track_counts": selection["formal_track_counts"]},
        "canary": {"design": _binding(CANARY_DESIGN), "queue": _binding(CANARY_QUEUE), "jobs": len(canary)},
        "formal": {"design": _binding(FORMAL_DESIGN), "queue": _binding(FORMAL_QUEUE), "jobs": len(formal)},
        "staging_config": {"path": str(STAGING_CONFIG.resolve()), "sha256": sha256_file(STAGING_CONFIG), "model_ids": config["training_scope"]["model_ids"]},
        "gpu_training_started": False,
        "execution_authorized": False,
    }


def main() -> None:
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
