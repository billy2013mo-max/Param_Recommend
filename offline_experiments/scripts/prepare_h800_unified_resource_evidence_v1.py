#!/usr/bin/env python3
"""Prepare the unified H800 resource-model evidence campaign.

This campaign is deliberately *not* organized as one model per mechanism.
Every job becomes a row for one shared model whose inputs contain training mode,
ZeRO, gradient checkpointing, GPU count, workload profile and packing.

The queue has three evidence roles:

* 128 matched non-packing rows: four workload sources x sixteen under-covered
  mechanisms x (one shared anchor + one boundary probe).  The shared anchor is
  identical inside a source and identifies mechanism effects; the boundary row
  supplies high-pressure success or a right-censored OOM lower bound.
* 20 Critical-LoRA order runs: four controlled profiles with the same exact
  maximum token length x five data-order seeds.  They estimate aleatoric
  allocator/order variance and must be collapsed to four profile scenarios
  before center fitting.
* 72 formal Packing rows: three profiles x four missing mode/scale mechanisms x
  unpacked/packed x three repeats.  Repeats estimate throughput noise and must be
  collapsed by physical arm before memory-center fitting.

The script is CPU-only.  It materializes the four controlled length profiles,
updates the local dataset registry, writes a queue/design/manifest and emits a
staged experiment configuration for all eight H800s.  It never authorizes or
launches GPU work and refuses to overwrite drifted artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    DATA_DIR,
    MATRIX_DIR,
    ROOT,
    percentile,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from prepare_packing_dataprofile_v2 import _curve

CAMPAIGN_ID = "h800_unified_resource_evidence_20260809_v1"
PHASE_ID = "h800_unified_resource_evidence_v1"
JOB_SCHEMA = "sft_h800_unified_resource_evidence_job/v1"
DESIGN_SCHEMA = "sft_h800_unified_resource_evidence_design/v1"
MANIFEST_SCHEMA = "sft_h800_unified_resource_evidence_queue_manifest/v1"
CONTROLLED_SCHEMA = "sft_controlled_runtime_profile_manifest/v1"

GPU_IDS = tuple(range(8))
SAFE_LIMIT_GIB = 132.83927001953126
NONPACKING_TARGET_GBS = 64
PACKING_TARGET_GBS = 128
WARMUP_STEPS = 3
MEASURE_STEPS = 10
ORDER_SEEDS = (2026080901, 2026080902, 2026080903, 2026080904, 2026080905)

QUEUE = MATRIX_DIR / "h800_unified_resource_evidence_jobs_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_unified_resource_evidence_design_v1.json"
QUEUE_MANIFEST = (
    ARTIFACT_DIR / "h800_unified_resource_evidence_queue_manifest_v1.json"
)
STAGING_DIR = ROOT / "unified_resource_staging"
EXPERIMENT_CONFIG = STAGING_DIR / "experiment.h800_unified_resource_evidence_v1.json"
CONTROLLED_DATA_DIR = DATA_DIR / "unified_resource_evidence_v1"
CONTROLLED_PROFILE_DIR = ARTIFACT_DIR / "unified_resource_evidence_v1" / "profiles"
CONTROLLED_MANIFEST = (
    ARTIFACT_DIR / "unified_resource_evidence_v1" / "controlled_profiles_manifest.json"
)
MODEL_INVENTORY = ARTIFACT_DIR / "model_inventory.json"
DATASET_REGISTRY = DATA_DIR / "dataset_info.json"


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _models() -> dict[str, Mapping[str, Any]]:
    return {
        str(row["id"]): row for row in read_json(MODEL_INVENTORY)["models"]
    }


FIT_SOURCES: dict[str, dict[str, Any]] = {
    "lora_s2_src04_live_punishment": {
        "profile_family": "short_concentrated",
        "dataset_category": "short",
        "campaign_dir": "h800_lora_safety_stage2_v1",
    },
    "lora_src07_flood_event_classify": {
        "profile_family": "broad_mid_length",
        "dataset_category": "multiturn",
        "campaign_dir": "h800_lora_source_disjoint_v1",
    },
    "lora_src15_live_highlight_long": {
        "profile_family": "rare_extreme_tail",
        "dataset_category": "longtail",
        "campaign_dir": "h800_lora_source_disjoint_v1",
    },
    "lora_s2_src17_content_risk": {
        "profile_family": "long_concentrated",
        "dataset_category": "longcontext",
        "campaign_dir": "h800_lora_safety_stage2_v1",
    },
}


def _source_data_path(source_id: str) -> Path:
    spec = FIT_SOURCES[source_id]
    return DATA_DIR / spec["campaign_dir"] / f"{source_id}.jsonl"


def _source_profile_path(source_id: str) -> Path:
    spec = FIT_SOURCES[source_id]
    return (
        ARTIFACT_DIR
        / spec["campaign_dir"]
        / "profiles"
        / f"{source_id}.qwen3_nothink.jsonl"
    )


# These are precisely the 16 non-packing cells with fewer than five independent
# success sources in the frozen 273-row V5 fit table.  The boundary probes are
# inherited from historical safe/OOM transitions or the existing negative-
# evidence plan; the common anchor is added separately and is identical across
# mechanisms inside each data source.
MECHANISMS: tuple[dict[str, Any], ...] = (
    {
        "mechanism_id": "lora_zero0_gc1_1gpu_pack0",
        "training_mode": "lora", "zero_stage": 0, "gc": True, "gpu_count": 1,
        "boundary": {"model_id": "qwen3_8b", "mbs": 16, "cutoff_len": 16384},
    },
    {
        "mechanism_id": "lora_zero2_gc0_4gpu_pack0",
        "training_mode": "lora", "zero_stage": 2, "gc": False, "gpu_count": 4,
        "boundary": {"model_id": "qwen3_14b", "mbs": 4, "cutoff_len": 8192},
    },
    {
        "mechanism_id": "lora_zero2_gc1_2gpu_pack0",
        "training_mode": "lora", "zero_stage": 2, "gc": True, "gpu_count": 2,
        "boundary": {"model_id": "qwen3_8b", "mbs": 16, "cutoff_len": 16384},
    },
    {
        "mechanism_id": "lora_zero2_gc1_4gpu_pack0",
        "training_mode": "lora", "zero_stage": 2, "gc": True, "gpu_count": 4,
        "boundary": {"model_id": "qwen3_8b", "mbs": 16, "cutoff_len": 16384},
    },
    {
        "mechanism_id": "lora_zero3_gc0_2gpu_pack0",
        "training_mode": "lora", "zero_stage": 3, "gc": False, "gpu_count": 2,
        "boundary": {"model_id": "qwen3_14b", "mbs": 2, "cutoff_len": 16384},
    },
    {
        "mechanism_id": "lora_zero3_gc0_4gpu_pack0",
        "training_mode": "lora", "zero_stage": 3, "gc": False, "gpu_count": 4,
        "boundary": {"model_id": "qwen3_14b", "mbs": 2, "cutoff_len": 16384},
    },
    {
        "mechanism_id": "lora_zero3_gc1_2gpu_pack0",
        "training_mode": "lora", "zero_stage": 3, "gc": True, "gpu_count": 2,
        "boundary": {"model_id": "qwen3_32b", "mbs": 8, "cutoff_len": 16384},
    },
    {
        "mechanism_id": "lora_zero3_gc1_4gpu_pack0",
        "training_mode": "lora", "zero_stage": 3, "gc": True, "gpu_count": 4,
        "boundary": {"model_id": "qwen3_32b", "mbs": 4, "cutoff_len": 16384},
    },
    {
        "mechanism_id": "full_zero0_gc0_1gpu_pack0",
        "training_mode": "full", "zero_stage": 0, "gc": False, "gpu_count": 1,
        "boundary": {"model_id": "qwen3_4b", "mbs": 2, "cutoff_len": 4096},
    },
    {
        "mechanism_id": "full_zero0_gc1_1gpu_pack0",
        "training_mode": "full", "zero_stage": 0, "gc": True, "gpu_count": 1,
        "boundary": {"model_id": "qwen3_8b", "mbs": 1, "cutoff_len": 4096},
    },
    {
        "mechanism_id": "full_zero2_gc0_2gpu_pack0",
        "training_mode": "full", "zero_stage": 2, "gc": False, "gpu_count": 2,
        "boundary": {"model_id": "qwen3_8b", "mbs": 4, "cutoff_len": 4096},
    },
    {
        "mechanism_id": "full_zero2_gc0_4gpu_pack0",
        "training_mode": "full", "zero_stage": 2, "gc": False, "gpu_count": 4,
        "boundary": {"model_id": "qwen3_14b", "mbs": 2, "cutoff_len": 4096},
    },
    {
        "mechanism_id": "full_zero2_gc1_2gpu_pack0",
        "training_mode": "full", "zero_stage": 2, "gc": True, "gpu_count": 2,
        "boundary": {"model_id": "qwen3_14b", "mbs": 1, "cutoff_len": 2048},
    },
    {
        "mechanism_id": "full_zero2_gc1_4gpu_pack0",
        "training_mode": "full", "zero_stage": 2, "gc": True, "gpu_count": 4,
        "boundary": {"model_id": "qwen3_14b", "mbs": 4, "cutoff_len": 8192},
    },
    {
        "mechanism_id": "full_zero3_gc0_2gpu_pack0",
        "training_mode": "full", "zero_stage": 3, "gc": False, "gpu_count": 2,
        "boundary": {"model_id": "qwen3_8b", "mbs": 4, "cutoff_len": 4096},
    },
    {
        "mechanism_id": "full_zero3_gc0_4gpu_pack0",
        "training_mode": "full", "zero_stage": 3, "gc": False, "gpu_count": 4,
        "boundary": {"model_id": "qwen3_8b", "mbs": 8, "cutoff_len": 4096},
    },
)

COMMON_ANCHOR = {"model_id": "qwen3_4b", "mbs": 1, "cutoff_len": 2048}


CONTROLLED_INPUTS: dict[str, tuple[Path, Path]] = {
    "concentrated": (
        DATA_DIR
        / "bounded_memory_v2_fresh_holdout_v1"
        / "fresh_s3_concentrated_truncated_long_v1.jsonl",
        ARTIFACT_DIR
        / "bounded_memory_v2_fresh_holdout_v1"
        / "profiles"
        / "fresh_s3_concentrated_truncated_long_v1.qwen3_nothink.jsonl",
    ),
    "broad": (
        DATA_DIR
        / "bounded_memory_v2_fresh_holdout_v1"
        / "fresh_s3_broad_nontruncated_longtail_v1.jsonl",
        ARTIFACT_DIR
        / "bounded_memory_v2_fresh_holdout_v1"
        / "profiles"
        / "fresh_s3_broad_nontruncated_longtail_v1.qwen3_nothink.jsonl",
    ),
    "rare_tail": (
        DATA_DIR
        / "bounded_memory_v2_fresh_holdout_v1"
        / "fresh_s3_rare_tail_short_v1.jsonl",
        ARTIFACT_DIR
        / "bounded_memory_v2_fresh_holdout_v1"
        / "profiles"
        / "fresh_s3_rare_tail_short_v1.qwen3_nothink.jsonl",
    ),
}
CONTROLLED_IDS = (
    "controlled_concentrated_same_max_v1",
    "controlled_broad_same_max_v1",
    "controlled_rare_tail_same_max_v1",
    "controlled_bimodal_same_max_v1",
)


def _paired_rows(data_path: Path, profile_path: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    data = read_jsonl(data_path)
    profiles = read_jsonl(profile_path)
    if len(data) != len(profiles):
        raise ValueError(f"data/profile row count mismatch: {data_path}, {profile_path}")
    pairs = []
    for row, profile in zip(data, profiles):
        if str(row.get("sample_id")) != str(profile.get("sample_id")):
            raise ValueError(f"data/profile order mismatch at {data_path}")
        pairs.append((row, profile))
    return pairs


def _rename_pair(
    pair: tuple[dict[str, Any], dict[str, Any]],
    *,
    dataset_id: str,
    index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data, profile = copy.deepcopy(pair[0]), copy.deepcopy(pair[1])
    original = str(profile["sample_id"])
    sample_id = f"{dataset_id}:controlled:{index:04d}"
    data["derived_source_sample_id"] = original
    data["sample_id"] = sample_id
    profile["derived_source_sample_id"] = original
    profile["controlled_profile_id"] = dataset_id
    profile["sample_id"] = sample_id
    return data, profile


def _statistics(lengths: Sequence[int]) -> dict[str, Any]:
    return {
        "rows": len(lengths),
        "mean": sum(lengths) / len(lengths),
        "p50": percentile(list(lengths), 50),
        "p90": percentile(list(lengths), 90),
        "p95": percentile(list(lengths), 95),
        "p99": percentile(list(lengths), 99),
        "maximum": max(lengths),
    }


def materialize_controlled_profiles() -> dict[str, Any]:
    """Create four real-row profiles with one shared exact maximum token row."""

    if CONTROLLED_MANIFEST.is_file():
        manifest = read_json(CONTROLLED_MANIFEST)
        if manifest.get("schema") != CONTROLLED_SCHEMA:
            raise ValueError("controlled profile manifest schema drifted")
        unsigned = dict(manifest)
        report_sha256 = unsigned.pop("report_sha256", None)
        if report_sha256 != sha256_json(unsigned):
            raise ValueError("controlled profile manifest internal hash drifted")
        for binding in manifest.get("outputs") or []:
            path = Path(binding["path"])
            if not path.is_file() or sha256_file(path) != binding["sha256"]:
                raise ValueError(f"controlled profile output drifted: {path}")
        registry = read_json(DATASET_REGISTRY)
        missing = [dataset_id for dataset_id in CONTROLLED_IDS if dataset_id not in registry]
        if missing:
            raise ValueError(f"controlled datasets missing from registry: {missing}")
        return manifest

    partials = [
        path
        for dataset_id in CONTROLLED_IDS
        for path in (
            CONTROLLED_DATA_DIR / f"{dataset_id}.jsonl",
            CONTROLLED_PROFILE_DIR / f"{dataset_id}.qwen3_nothink.jsonl",
        )
        if path.exists()
    ]
    if partials:
        raise FileExistsError(
            f"controlled profile outputs exist without a sealed manifest: {partials}"
        )
    inputs = {name: _paired_rows(*paths) for name, paths in CONTROLLED_INPUTS.items()}
    broad = inputs["broad"]
    common_max = max(broad, key=lambda pair: int(pair[1]["total_tokens"]))
    common_max_tokens = int(common_max[1]["total_tokens"])

    concentrated = sorted(inputs["concentrated"], key=lambda pair: str(pair[1]["sample_id"]))[:999]
    common_max_sample_id = str(common_max[1]["sample_id"])
    broad_without_max = [
        pair
        for pair in broad
        if str(pair[1]["sample_id"]) != common_max_sample_id
    ][:999]
    rare = sorted(inputs["rare_tail"], key=lambda pair: str(pair[1]["sample_id"]))[:999]
    bimodal = (
        sorted(inputs["rare_tail"], key=lambda pair: int(pair[1]["total_tokens"]))[:500]
        + sorted(
            inputs["concentrated"],
            key=lambda pair: int(pair[1]["total_tokens"]),
            reverse=True,
        )[:499]
    )
    selected = {
        CONTROLLED_IDS[0]: [*concentrated, common_max],
        CONTROLLED_IDS[1]: [*broad_without_max, common_max],
        CONTROLLED_IDS[2]: [*rare, common_max],
        CONTROLLED_IDS[3]: [*bimodal, common_max],
    }
    outputs: list[dict[str, Any]] = []
    profile_stats: dict[str, Any] = {}
    CONTROLLED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONTROLLED_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_id, pairs in selected.items():
        if len(pairs) != 1000:
            raise ValueError(f"{dataset_id}: expected 1000 controlled rows")
        renamed = [
            _rename_pair(pair, dataset_id=dataset_id, index=index)
            for index, pair in enumerate(pairs)
        ]
        data_path = CONTROLLED_DATA_DIR / f"{dataset_id}.jsonl"
        profile_path = CONTROLLED_PROFILE_DIR / f"{dataset_id}.qwen3_nothink.jsonl"
        write_jsonl(data_path, (row[0] for row in renamed))
        write_jsonl(profile_path, (row[1] for row in renamed))
        lengths = [int(row[1]["total_tokens"]) for row in renamed]
        stats = _statistics(lengths)
        if stats["maximum"] != common_max_tokens:
            raise ValueError(f"{dataset_id}: controlled maximum is not shared")
        profile_stats[dataset_id] = stats
        outputs.extend((_binding(data_path), _binding(profile_path)))

    registry = read_json(DATASET_REGISTRY)
    for dataset_id in CONTROLLED_IDS:
        expected = {
            "file_name": f"unified_resource_evidence_v1/{dataset_id}.jsonl",
            "columns": {"prompt": "prompt", "response": "response", "system": "system"},
        }
        existing = registry.get(dataset_id)
        if existing is not None and existing != expected:
            raise ValueError(f"dataset registry collision for {dataset_id}")
        registry[dataset_id] = expected
    write_json(DATASET_REGISTRY, registry)
    manifest = {
        "schema": CONTROLLED_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "hold global maximum fixed while changing the empirical sequence-length "
            "distribution; repeated seeds estimate order/allocator variance"
        ),
        "fit_policy": "collapse_five_seeds_to_one_profile_scenario_before_center_fit",
        "shared_maximum_tokens": common_max_tokens,
        "shared_maximum_source_sample_id": str(common_max[1]["sample_id"]),
        "source_bindings": [
            _binding(path)
            for paths in CONTROLLED_INPUTS.values()
            for path in paths
        ],
        "outputs": outputs,
        "profile_statistics": profile_stats,
        "dataset_registry": _binding(DATASET_REGISTRY),
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(CONTROLLED_MANIFEST, manifest)
    return manifest


PACKING_PROFILES: dict[str, dict[str, Any]] = {
    "W5": {
        "dataset_id": "packing_w5_longcontext_probe_v2",
        "dataset_category": "longcontext",
        "data_file": "packing_w5_longcontext_probe_v2.jsonl",
        "profile_file": "packing_w5_longcontext_probe_v2.qwen3_nothink.jsonl",
    },
    "W7": {
        "dataset_id": "packing_w7_bimodal_v1",
        "dataset_category": "longtail",
        "data_file": "packing_w7_bimodal_v1.jsonl",
        "profile_file": "packing_w7_bimodal_v1.qwen3_nothink.jsonl",
    },
    "W8": {
        "dataset_id": "packing_w8_code_structured_v1",
        "dataset_category": "longtail",
        "data_file": "packing_w8_code_structured_v1.jsonl",
        "profile_file": "packing_w8_code_structured_v1.qwen3_nothink.jsonl",
    },
}
PACKING_DATA_ROOT = DATA_DIR / "packing_profile_phase_b_v1"
PACKING_PROFILE_ROOT = ARTIFACT_DIR / "packing_profile_phase_b_v1" / "profiles"
PACKING_SETTINGS: tuple[dict[str, Any], ...] = (
    {
        "mechanism_id": "lora_zero3_gc1_4gpu",
        "training_mode": "lora", "zero_stage": 3, "gc": True, "gpu_count": 4,
        "model_id": "qwen3_32b", "cutoff_len": 32768,
    },
    {
        "mechanism_id": "full_zero2_gc0_2gpu",
        "training_mode": "full", "zero_stage": 2, "gc": False, "gpu_count": 2,
        "model_id": "qwen3_8b", "cutoff_len": 4096,
    },
    {
        "mechanism_id": "full_zero3_gc1_2gpu",
        "training_mode": "full", "zero_stage": 3, "gc": True, "gpu_count": 2,
        "model_id": "qwen3_14b", "cutoff_len": 4096,
    },
    {
        "mechanism_id": "full_zero3_gc0_4gpu",
        "training_mode": "full", "zero_stage": 3, "gc": False, "gpu_count": 4,
        "model_id": "qwen3_8b", "cutoff_len": 8192,
    },
)


def _profile_max(path: Path) -> int:
    return max(int(row["total_tokens"]) for row in read_jsonl(path))


def _base_job(
    *,
    identity: Mapping[str, Any],
    evidence_role: str,
    mechanism_id: str,
    training_mode: str,
    zero_stage: int,
    gc: bool,
    gpu_count: int,
    model_id: str,
    mbs: int,
    cutoff_len: int,
    target_gbs: int,
    gradient_accumulation_steps: int,
    packing: bool,
    dataset_id: str,
    dataset_category: str,
    data_path: Path,
    profile_path: Path,
    models: Mapping[str, Mapping[str, Any]],
    repeat: int = 0,
    seed: int = 20260716,
) -> dict[str, Any]:
    model = models[model_id]
    if not packing and target_gbs != gpu_count * mbs * gradient_accumulation_steps:
        raise ValueError("unpacked GBS contract is not exact")
    raw_max = _profile_max(profile_path)
    job_identity = {"campaign_id": CAMPAIGN_ID, **dict(identity)}
    return {
        "schema": JOB_SCHEMA,
        "job_id": stable_id("h800unires", job_identity),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "evidence_role": evidence_role,
        "mechanism_id": mechanism_id,
        "scenario_id": str(identity["scenario_id"]),
        "split_unit_id": dataset_id,
        "hardware_id": "local_h800_140g",
        "gpu_type": "NVIDIA H800 140GB HBM3",
        "required_runtime_gpu_name": "NVIDIA H800",
        "model_id": model_id,
        "model_path": model["path"],
        "tokenizer_path": model["tokenizer_path"],
        "model_family": model["family"],
        "model_parameters": model["actual_parameters"],
        "template": model["template"],
        "train_type": training_mode,
        "dataset_id": dataset_id,
        "dataset_category": dataset_category,
        "dataset_profile_path": str(profile_path.resolve()),
        "dataset_profile_sha256": sha256_file(profile_path),
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "cutoff_len": cutoff_len,
        "raw_profile_max": raw_max,
        "aligned_effective_sequence": 8 * ((min(raw_max, cutoff_len) + 7) // 8),
        "target_gbs": target_gbs,
        "gpu_count": gpu_count,
        "zero_stage": zero_stage,
        "zero": "none" if gpu_count == 1 else f"zero{zero_stage}",
        "gc": gc,
        "gradient_checkpointing": gc,
        "mbs": mbs,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "packing": packing,
        "offload": False,
        "kind": "throughput",
        "fidelity": "formal_3plus10",
        "warmup_steps": WARMUP_STEPS,
        "measure_steps": MEASURE_STEPS,
        "repeat": repeat,
        "seed": seed,
        "data_seed": seed,
        "max_samples": len(read_jsonl(data_path)),
        "parallel_class": "gpu_partitionable",
        "requires_external_node_idle": False,
        "calibration_partition": {
            "role": "fit" if "diagnostic" not in evidence_role else "diagnostic",
            "split_unit_id": dataset_id,
            "policy": "unified_model_source_grouped_v1",
        },
    }


def build_nonpacking_jobs(models: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for source_id, source in FIT_SOURCES.items():
        data_path = _source_data_path(source_id)
        profile_path = _source_profile_path(source_id)
        for mechanism in MECHANISMS:
            gpu_count = int(mechanism["gpu_count"])
            for role, probe in (("matched_anchor", COMMON_ANCHOR), ("boundary_probe", mechanism["boundary"])):
                mbs = int(probe["mbs"])
                denominator = gpu_count * mbs
                if NONPACKING_TARGET_GBS % denominator:
                    raise ValueError(
                        f"{mechanism['mechanism_id']} {role}: target GBS is not divisible"
                    )
                scenario_id = f"{source_id}__{mechanism['mechanism_id']}__{role}"
                jobs.append(
                    _base_job(
                        identity={
                            "scenario_id": scenario_id,
                            "role": role,
                            "source_id": source_id,
                            "mechanism_id": mechanism["mechanism_id"],
                            "probe": dict(probe),
                        },
                        evidence_role=f"unified_nonpacking_{role}_fit",
                        mechanism_id=str(mechanism["mechanism_id"]),
                        training_mode=str(mechanism["training_mode"]),
                        zero_stage=int(mechanism["zero_stage"]),
                        gc=bool(mechanism["gc"]),
                        gpu_count=gpu_count,
                        model_id=str(probe["model_id"]),
                        mbs=mbs,
                        cutoff_len=int(probe["cutoff_len"]),
                        target_gbs=NONPACKING_TARGET_GBS,
                        gradient_accumulation_steps=NONPACKING_TARGET_GBS // denominator,
                        packing=False,
                        dataset_id=source_id,
                        dataset_category=str(source["dataset_category"]),
                        data_path=data_path,
                        profile_path=profile_path,
                        models=models,
                    )
                )
    return jobs


def build_critical_variance_jobs(models: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for dataset_id in CONTROLLED_IDS:
        data_path = CONTROLLED_DATA_DIR / f"{dataset_id}.jsonl"
        profile_path = CONTROLLED_PROFILE_DIR / f"{dataset_id}.qwen3_nothink.jsonl"
        for repeat, seed in enumerate(ORDER_SEEDS):
            scenario_id = f"{dataset_id}__critical_lora__seed{seed}"
            jobs.append(
                _base_job(
                    identity={
                        "scenario_id": scenario_id,
                        "dataset_id": dataset_id,
                        "seed": seed,
                    },
                    evidence_role="critical_same_max_order_variance_diagnostic",
                    mechanism_id="lora_zero2_gc0_2gpu_pack0",
                    training_mode="lora",
                    zero_stage=2,
                    gc=False,
                    gpu_count=2,
                    model_id="qwen3_8b",
                    mbs=2,
                    cutoff_len=32768,
                    target_gbs=NONPACKING_TARGET_GBS,
                    gradient_accumulation_steps=16,
                    packing=False,
                    dataset_id=dataset_id,
                    dataset_category="longtail",
                    data_path=data_path,
                    profile_path=profile_path,
                    models=models,
                    repeat=repeat,
                    seed=seed,
                )
            )
    return jobs


def build_packing_jobs(models: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for profile_id, profile in PACKING_PROFILES.items():
        data_path = PACKING_DATA_ROOT / profile["data_file"]
        profile_path = PACKING_PROFILE_ROOT / profile["profile_file"]
        lengths = [int(row["total_tokens"]) for row in read_jsonl(profile_path)]
        for setting in PACKING_SETTINGS:
            gpu_count = int(setting["gpu_count"])
            cutoff_len = int(setting["cutoff_len"])
            curve = _curve(lengths, [cutoff_len])[0]
            samples_per_pack = float(curve["samples_per_pack"]["mean"])
            # A packed physical microbatch can already contain more than 128
            # logical samples (W8 at cutoff=32768 is the concrete case).  GA
            # cannot fall below one, so raise the *matched pair's* target by
            # powers of two until an attainable packed arm lies within 10%.
            # Both packed and unpacked arms use the same resulting target.
            pair_target_gbs = PACKING_TARGET_GBS
            minimum_packed_gbs = gpu_count * samples_per_pack
            while minimum_packed_gbs > pair_target_gbs * 1.10:
                pair_target_gbs *= 2
            raw_ga = pair_target_gbs / (gpu_count * samples_per_pack)
            packed_candidates = sorted(
                {max(1, math.floor(raw_ga)), max(1, round(raw_ga)), max(1, math.ceil(raw_ga))}
            )
            packed_ga = min(
                packed_candidates,
                key=lambda value: (
                    abs(gpu_count * value * samples_per_pack - PACKING_TARGET_GBS),
                    value,
                ),
            )
            expected_packed_gbs = gpu_count * packed_ga * samples_per_pack
            packed_gbs_error = (
                abs(expected_packed_gbs - pair_target_gbs) / pair_target_gbs
            )
            if packed_gbs_error > 0.10:
                raise ValueError(
                    f"{profile_id}/{setting['mechanism_id']}: packed GBS error {packed_gbs_error:.3f}"
                )
            for packing in (False, True):
                ga = packed_ga if packing else pair_target_gbs // gpu_count
                for repeat in range(3):
                    branch = "packed" if packing else "unpacked"
                    scenario_id = (
                        f"{profile_id}__{setting['mechanism_id']}__{branch}__repeat{repeat}"
                    )
                    job = _base_job(
                        identity={
                            "scenario_id": scenario_id,
                            "profile_id": profile_id,
                            "mechanism_id": setting["mechanism_id"],
                            "packing": packing,
                            "repeat": repeat,
                        },
                        evidence_role="unified_packing_matched_formal_fit",
                        mechanism_id=str(setting["mechanism_id"]),
                        training_mode=str(setting["training_mode"]),
                        zero_stage=int(setting["zero_stage"]),
                        gc=bool(setting["gc"]),
                        gpu_count=gpu_count,
                        model_id=str(setting["model_id"]),
                        mbs=1,
                        cutoff_len=cutoff_len,
                        target_gbs=pair_target_gbs,
                        gradient_accumulation_steps=ga,
                        packing=packing,
                        dataset_id=str(profile["dataset_id"]),
                        dataset_category=str(profile["dataset_category"]),
                        data_path=data_path,
                        profile_path=profile_path,
                        models=models,
                        repeat=repeat,
                        seed=2026080910 + repeat,
                    )
                    job["packing_contract"] = {
                        "expected_samples_per_pack": samples_per_pack,
                        "expected_sample_gbs": (
                            expected_packed_gbs if packing else pair_target_gbs
                        ),
                        "expected_sample_gbs_relative_error": (
                            packed_gbs_error if packing else 0.0
                        ),
                        "comparison_policy": (
                            "same source/model/cutoff/mechanism; GA changes only to "
                            "match logical sample GBS"
                        ),
                    }
                    jobs.append(job)
    return jobs


def build_jobs(models: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    jobs = [
        *build_nonpacking_jobs(models),
        *build_critical_variance_jobs(models),
        *build_packing_jobs(models),
    ]
    counts = Counter(job["evidence_role"] for job in jobs)
    expected = {
        "unified_nonpacking_matched_anchor_fit": 64,
        "unified_nonpacking_boundary_probe_fit": 64,
        "critical_same_max_order_variance_diagnostic": 20,
        "unified_packing_matched_formal_fit": 72,
    }
    if dict(counts) != expected:
        raise ValueError(f"evidence-role counts drifted: {dict(counts)}")
    ids = [job["job_id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("queue contains duplicate job IDs")
    if any(job["offload"] for job in jobs):
        raise ValueError("offload is outside this campaign")
    return jobs


def _experiment_config(jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    live = read_json(CONFIG_DIR / "experiment.json")
    zero_by_gpu: dict[str, set[str]] = {}
    for job in jobs:
        zero_by_gpu.setdefault(str(job["gpu_count"]), set()).add(str(job["zero"]))
    datasets = []
    for dataset_id in sorted({str(job["dataset_id"]) for job in jobs}):
        rows = [job for job in jobs if job["dataset_id"] == dataset_id]
        datasets.append(
            {
                "id": dataset_id,
                "category": rows[0]["dataset_category"],
                "target_cutoffs": sorted({int(row["cutoff_len"]) for row in rows}),
            }
        )
    return {
        "schema_version": 1,
        "training_scope": {
            "phase_id": PHASE_ID,
            "model_ids": sorted({str(job["model_id"]) for job in jobs}),
            "gpu_ids": list(GPU_IDS),
            "max_gpu_count": 4,
            "exclusive_node_gpu_ids": list(GPU_IDS),
            "deferred_model_ids": [],
            "hardware_followups": {
                "memory": (
                    "one shared H800 resource model; mechanism fields are inputs, "
                    "not separately routed models"
                ),
                "followup": (
                    "fit rows and repeat diagnostics only; publication requires a "
                    "separate source-disjoint prospective acceptance campaign"
                ),
            },
            "stage": "sft",
            "precision": "bf16",
            "gpu_type": "NVIDIA H800 140GB HBM3",
            "gpu_counts": [1, 2, 4],
            "global_batch_sizes": sorted({int(job["target_gbs"]) for job in jobs}),
            "gradient_checkpointing": [False, True],
            "zero_by_gpu_count": {
                key: sorted(values) for key, values in sorted(zero_by_gpu.items())
            },
        },
        "fixed_runtime": live["fixed_runtime"],
        "measurement": {
            **live["measurement"],
            "throughput_warmup_steps": WARMUP_STEPS,
            "throughput_measure_steps": MEASURE_STEPS,
            "throughput_repeats": 3,
            "performance_parallelism": "disjoint_gpu_masks",
            "scheduler_order_policy": "parallel_queue",
            "formal_throughput_requires_exclusive_node": False,
        },
        "datasets": datasets,
        "packing_static_gate": {
            "minimum_pack_utilization": 0.0,
            "minimum_sequence_reduction": 0.0,
            "minimum_mean_samples_per_pack": 1.0,
            "maximum_expected_gbs_error": 0.10,
            "note": "packed GA is derived from each exact frozen profile curve",
        },
        "scaling_rule": live.get("scaling_rule", {}),
        "matrix_policy": {
            "nonpacking_undercovered_mechanisms": len(MECHANISMS),
            "fit_sources": len(FIT_SOURCES),
            "probes_per_source_mechanism": 2,
            "critical_profile_shapes": len(CONTROLLED_IDS),
            "critical_order_seeds": len(ORDER_SEEDS),
            "packing_profiles": len(PACKING_PROFILES),
            "packing_mechanisms": len(PACKING_SETTINGS),
            "packing_branches": 2,
            "packing_repeats": 3,
        },
    }


def build_design(jobs: Sequence[Mapping[str, Any]], controlled: Mapping[str, Any]) -> dict[str, Any]:
    counts = Counter(job["evidence_role"] for job in jobs)
    generated_at_utc = (
        str(read_json(DESIGN)["generated_at_utc"])
        if DESIGN.is_file()
        else datetime.now(timezone.utc).isoformat()
    )
    design = {
        "schema": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": generated_at_utc,
        "status": "fit_and_diagnostic_queue_materialized_waiting_for_promotion",
        "gpu_training_started": False,
        "execution_authorized": False,
        "publication_allowed": False,
        "objective": (
            "Acquire balanced evidence for one unified, physics-informed H800 "
            "memory/throughput recommender without mechanism-specific product routes."
        ),
        "declared_domain": {
            "hardware": "8x local NVIDIA H800 140GB pool",
            "model_family": "dense Qwen3 text models represented by 4B/8B/14B/32B",
            "training_modes": ["lora", "full"],
            "gpu_counts": [1, 2, 4],
            "zero_stages": [0, 2, 3],
            "gradient_checkpointing": [False, True],
            "packing": [False, True],
            "offload": False,
        },
        "job_count": len(jobs),
        "gpu_job_equivalents": sum(int(job["gpu_count"]) for job in jobs),
        "evidence_role_counts": dict(counts),
        "fit_contract": {
            "single_shared_model": True,
            "mechanism_fields_are_features_not_routes": True,
            "success_rows_are_exact_center_labels": True,
            "oom_rows_are_right_censored_lower_bounds": True,
            "critical_seed_rows_collapse_to_four_profile_scenarios": True,
            "packing_repeats_collapse_by_physical_arm_for_memory_center": True,
            "packing_repeats_remain_repeats_for_throughput_noise": True,
            "source_grouped_cross_validation_required": True,
            "campaign_grouped_sensitivity_required": True,
        },
        "identifiability_review": {
            "matched_anchor": (
                "within each source, all 16 mechanisms share Qwen3-4B/MBS1/cutoff2048"
            ),
            "boundary_probe": (
                "same source/mechanism receives a historically bracketed high-pressure point"
            ),
            "critical_distribution_test": (
                "four profiles share exact maximum token length; only distribution/order changes"
            ),
            "packing_test": (
                "each profile/mechanism has matched unpacked/packed arms and three repeats"
            ),
        },
        "not_in_this_queue": {
            "prospective_acceptance": (
                "must be generated only after the unified model, features and candidate "
                "selection policy are frozen"
            ),
            "publication": False,
        },
        "controlled_profile_manifest": {
            "path": str(CONTROLLED_MANIFEST.resolve()),
            "sha256": sha256_file(CONTROLLED_MANIFEST),
            "report_sha256": controlled["report_sha256"],
        },
        "ordered_job_ids": [str(job["job_id"]) for job in jobs],
        "ordered_job_payload_sha256": sha256_json(list(jobs)),
    }
    design["report_sha256"] = sha256_json(design)
    return design


def _write_or_validate(path: Path, value: Any, *, jsonl: bool = False) -> None:
    if path.exists():
        existing = read_jsonl(path) if jsonl else read_json(path)
        if existing != value:
            raise FileExistsError(f"refusing to overwrite drifted artifact: {path}")
        return
    if jsonl:
        write_jsonl(path, value)
    else:
        write_json(path, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for source_id in FIT_SOURCES:
        for path in (_source_data_path(source_id), _source_profile_path(source_id)):
            if not path.is_file():
                raise FileNotFoundError(path)
    controlled = materialize_controlled_profiles()
    jobs = build_jobs(_models())
    experiment = _experiment_config(jobs)
    design = build_design(jobs, controlled)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    _write_or_validate(EXPERIMENT_CONFIG, experiment)
    _write_or_validate(QUEUE, jobs, jsonl=True)
    _write_or_validate(DESIGN, design)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "gpu_training_started": False,
        "execution_authorized": False,
        "publication_allowed": False,
        "queue": {**_binding(QUEUE), "job_count": len(jobs)},
        "design": _binding(DESIGN),
        "experiment_config": _binding(EXPERIMENT_CONFIG),
        "controlled_profiles": _binding(CONTROLLED_MANIFEST),
        "dataset_registry": _binding(DATASET_REGISTRY),
        "next_step": (
            "run unit tests and scheduler preview; promotion/launch is a separate explicit action"
        ),
    }
    manifest["report_sha256"] = sha256_json(manifest)
    _write_or_validate(QUEUE_MANIFEST, manifest)
    print(
        json.dumps(
            {
                "queue": _binding(QUEUE),
                "design": _binding(DESIGN),
                "manifest": _binding(QUEUE_MANIFEST),
                "experiment_config": _binding(EXPERIMENT_CONFIG),
                "jobs": len(jobs),
                "gpu_job_equivalents": sum(int(job["gpu_count"]) for job in jobs),
                "evidence_roles": dict(Counter(job["evidence_role"] for job in jobs)),
                "authorized_gpu_ids": list(GPU_IDS),
                "execution_authorized": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
