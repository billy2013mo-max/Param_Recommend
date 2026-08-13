#!/usr/bin/env python3
"""Verify the retired H800 calibration candidate without authorizing execution.

The v1 109-row plan is retained only for deterministic historical audit.  New
files may no longer be written from this schema; ``--verify-only`` reconstructs
and validates it in memory.  It never launches training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from approval_gate import reject_retired_approval_plan
from common import ROOT, read_json, sha256_file, sha256_json, stable_id


SCHEMA = "sft_h800_calibration_candidate/v1"
JOB_SCHEMA = "sft_h800_calibration_job/v1"
SPLIT_POLICY = "model_length_disjoint_h800_v1"
HARDWARE_NAME = "NVIDIA H800"
HARDWARE_TYPE = "NVIDIA H800 140GB HBM3"
HARDWARE_ID = "local_h800_140g"
MBS_DOMAIN = (1, 2, 4, 8, 16)
TARGET_GBS = 64
PACKING_TARGET_GBS = 64
BOUNDARY_WARMUP_STEPS = 1
BOUNDARY_MEASURE_STEPS = 2
PACKING_WARMUP_STEPS = 1
PACKING_MEASURE_STEPS = 3
RUN_TIMEOUT_SECONDS = 45 * 60
HARD_WALL_TIME_HOURS = 72.0
MAX_CANDIDATE_RUNS = 110
FORBIDDEN_GPU_PATTERNS = ("4090",)

DESIGN_NAME = "h800_calibration_candidate.design.json"
JOBS_NAME = "h800_calibration_candidate.jobs.jsonl"

SOURCE_FILES = (
    "config/experiment.json",
    "config/hardware.json",
    "config/models.json",
    "config/deepspeed/ds_z3.json",
    "artifacts/model_inventory.json",
    "artifacts/dataset_analysis.json",
    "artifacts/canonical_h800_observations.jsonl",
    "artifacts/h800_calibration_readiness.json",
    "scripts/run_job.py",
    "scripts/export_h800_observations.py",
    "scripts/audit_h800_calibration_readiness.py",
    "scripts/prepare_h800_calibration_candidate.py",
    "scripts/freeze_h800_calibration_approval_candidate.py",
    "scripts/scheduler.py",
)

DATASETS = {
    512: "short_512",
    2048: "multiturn_2048",
    4096: "multiturn_4096",
    8192: "longtail_8192",
    32768: "longcontext_32768",
}

# The split unit is the model/sequence pair.  A unit is never permitted to
# change role for another training mode, ZeRO stage, GPU count, or packing arm.
SPLIT_UNITS = {
    "calibration": (
        ("qwen3_1p7b", 512),
        ("qwen3_4b", 8192),
        ("qwen3_8b", 4096),
        ("qwen3_14b", 32768),
        ("qwen3_14b", 512),
    ),
    "holdout": (
        ("qwen3_1p7b", 32768),
        ("qwen3_4b", 2048),
        ("qwen3_8b", 32768),
        ("qwen3_14b", 4096),
    ),
}

# Each tuple is role, model, length, GPU count, anchor MBS.  The candidate
# includes anchor, one conditional halving fallback, and up to two doublings.
# Execution stops as soon as the family is bracketed by success/OOM evidence.
STAGE0_FULL_FAMILIES = (
    ("calibration", "qwen3_1p7b", 512, 1, 8),
    ("calibration", "qwen3_4b", 8192, 1, 4),
    ("calibration", "qwen3_14b", 512, 1, 1),
    ("holdout", "qwen3_1p7b", 32768, 1, 2),
    ("holdout", "qwen3_4b", 2048, 1, 8),
)

STAGE0_LORA_FAMILIES = (
    ("calibration", "qwen3_1p7b", 512, 1, 8),
    ("calibration", "qwen3_4b", 8192, 1, 4),
    ("calibration", "qwen3_14b", 512, 1, 8),
    ("holdout", "qwen3_1p7b", 32768, 1, 2),
    ("holdout", "qwen3_4b", 2048, 1, 8),
)

STAGE3_FULL_FAMILIES = (
    ("calibration", "qwen3_4b", 8192, 2, 4),
    ("calibration", "qwen3_8b", 4096, 4, 8),
    ("calibration", "qwen3_14b", 32768, 2, 1),
    ("holdout", "qwen3_8b", 32768, 2, 2),
    ("holdout", "qwen3_14b", 4096, 4, 8),
)

STAGE3_LORA_FAMILIES = (
    ("calibration", "qwen3_4b", 8192, 2, 8),
    ("calibration", "qwen3_8b", 4096, 4, 8),
    ("calibration", "qwen3_14b", 32768, 2, 2),
    ("holdout", "qwen3_8b", 32768, 2, 2),
    ("holdout", "qwen3_14b", 4096, 4, 8),
)

# role, model, length, GPU count, safe unpacked MBS.  There are two calibration
# and two holdout pairs per training mode.  AB/BA gives two effective repeats
# per arm while preserving a compact, independently auditable packing study.
PACKING_PAIRS = {
    "full": (
        ("calibration", "qwen3_4b", 8192, 1, 4),
        ("calibration", "qwen3_8b", 4096, 2, 8),
        ("holdout", "qwen3_4b", 2048, 1, 16),
        ("holdout", "qwen3_14b", 4096, 4, 8),
    ),
    "lora": (
        ("calibration", "qwen3_4b", 8192, 1, 4),
        ("calibration", "qwen3_8b", 4096, 4, 8),
        ("holdout", "qwen3_4b", 2048, 1, 16),
        ("holdout", "qwen3_14b", 4096, 2, 8),
    ),
}


def _canonical_jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _split_unit_id(model_id: str, cutoff_len: int) -> str:
    return f"{model_id}@{cutoff_len}"


def _role_map() -> dict[tuple[str, int], str]:
    mapping: dict[tuple[str, int], str] = {}
    for role, units in SPLIT_UNITS.items():
        for unit in units:
            if unit in mapping:
                raise ValueError(f"Calibration split unit is duplicated: {unit}")
            mapping[unit] = role
    return mapping


def _source_manifest(project_root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required candidate source is missing: {path}")
        manifest[relative] = sha256_file(path)
    return manifest


def _load_inputs(
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    experiment = read_json(project_root / "config" / "experiment.json")
    hardware = read_json(project_root / "config" / "hardware.json")
    inventory = read_json(project_root / "artifacts" / "model_inventory.json")
    analysis = read_json(project_root / "artifacts" / "dataset_analysis.json")
    return experiment, hardware, inventory, analysis


def _validate_h800_sources(
    experiment: dict[str, Any], hardware: dict[str, Any]
) -> None:
    scope = experiment.get("training_scope") or {}
    identity = " ".join(
        str(value)
        for value in (
            scope.get("gpu_type"),
            hardware.get("gpu_id"),
            hardware.get("name_reported_by_driver"),
        )
    ).lower()
    if (
        scope.get("gpu_type") != HARDWARE_TYPE
        or hardware.get("gpu_id") != HARDWARE_ID
        or hardware.get("name_reported_by_driver") != HARDWARE_NAME
        or "h800" not in identity
        or any(pattern in identity for pattern in FORBIDDEN_GPU_PATTERNS)
    ):
        raise ValueError("Candidate generation requires the exact H800 140GB profile")
    if set(scope.get("gpu_counts") or ()) != {1, 2, 4}:
        raise ValueError("H800 candidate source must expose exactly 1/2/4 GPU counts")
    if not set(scope.get("gpu_ids") or ()) >= {1, 2, 3, 4}:
        raise ValueError("H800 candidate source does not expose four assigned GPUs")


def _model_map(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected = {model for units in SPLIT_UNITS.values() for model, _ in units}
    rows = {
        str(row.get("id")): row
        for row in inventory.get("models") or ()
        if isinstance(row, dict) and row.get("id") in selected
    }
    if set(rows) != selected:
        raise ValueError(f"Model inventory is missing H800 split models: {selected - set(rows)}")
    return rows


def _partition(role: str, model_id: str, cutoff_len: int) -> dict[str, str]:
    expected = _role_map().get((model_id, cutoff_len))
    if expected != role:
        raise ValueError(
            f"Split unit {model_id}@{cutoff_len} is {expected!r}, not {role!r}"
        )
    return {
        "role": role,
        "split_unit_id": _split_unit_id(model_id, cutoff_len),
        "policy": SPLIT_POLICY,
    }


def _base_job(
    *,
    experiment: dict[str, Any],
    model: dict[str, Any],
    role: str,
    cutoff_len: int,
    train_type: str,
    gpu_count: int,
    zero: str,
    mbs: int,
    target_gbs: int,
    packing: bool,
) -> dict[str, Any]:
    dataset_id = DATASETS[cutoff_len]
    scope = experiment["training_scope"]
    return {
        "schema": JOB_SCHEMA,
        "campaign_id": "h800_calibration_candidate_v1",
        "phase_id": scope["phase_id"],
        "hardware_id": HARDWARE_ID,
        "gpu_type": HARDWARE_TYPE,
        "required_runtime_gpu_name": HARDWARE_NAME,
        "required_memory_bytes": 150142189568,
        "forbid_mig": True,
        "model_id": model["id"],
        "model_path": model["path"],
        "tokenizer_path": model["tokenizer_path"],
        "model_family": model["family"],
        "model_parameters": model["actual_parameters"],
        "template": model["template"],
        "train_type": train_type,
        "dataset_id": dataset_id,
        "cutoff_len": cutoff_len,
        "gpu_count": gpu_count,
        "zero": zero,
        "zero_stage": 0 if zero == "none" else 3,
        "gc": True,
        "precision": "bf16",
        "mbs": mbs,
        "target_gbs": target_gbs,
        "packing": packing,
        "repeat": 0,
        "parallel_class": "exclusive_pool" if gpu_count == 4 else "gpu_partitionable",
        "requires_external_node_idle": False,
        "calibration_partition": _partition(role, model["id"], cutoff_len),
        "execution_authorized": False,
    }


def _probe_plan(anchor: int) -> list[tuple[str, int, dict[str, Any]]]:
    if anchor not in MBS_DOMAIN:
        raise ValueError(f"Boundary anchor is outside the planner MBS domain: {anchor}")
    rows: list[tuple[str, int, dict[str, Any]]] = [
        ("anchor", anchor, {"type": "always"})
    ]
    if anchor > 1:
        rows.append(
            (
                "fallback_half",
                anchor // 2,
                {"type": "if_probe_outcome", "probe": "anchor", "outcome": "oom"},
            )
        )
    prior_probe = "anchor"
    upward_index = 1
    candidate = anchor * 2
    while candidate <= MBS_DOMAIN[-1]:
        probe_name = f"upward_{upward_index}"
        rows.append(
            (
                probe_name,
                candidate,
                {
                    "type": "if_probe_outcome",
                    "probe": prior_probe,
                    "outcome": "success",
                },
            )
        )
        prior_probe = probe_name
        upward_index += 1
        candidate *= 2
    return rows


def _boundary_jobs(
    experiment: dict[str, Any], models: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    selector_specs = (
        ("stage0_full_unpacked", "full", STAGE0_FULL_FAMILIES),
        ("stage0_lora_unpacked", "lora", STAGE0_LORA_FAMILIES),
        ("stage3_full_unpacked", "full", STAGE3_FULL_FAMILIES),
        ("stage3_lora_unpacked", "lora", STAGE3_LORA_FAMILIES),
    )
    for selector_id, train_type, specifications in selector_specs:
        for role, model_id, cutoff_len, gpu_count, anchor in specifications:
            zero = "none" if gpu_count == 1 else "zero3"
            family_identity = {
                "candidate": "h800_calibration_candidate_v1",
                "selector_id": selector_id,
                "role": role,
                "model_id": model_id,
                "cutoff_len": cutoff_len,
                "gpu_count": gpu_count,
                "zero": zero,
                "train_type": train_type,
            }
            family_id = stable_id("h800calfam", family_identity)
            family_job_ids: list[str] = []
            probes = _probe_plan(anchor)
            for sequence_index, (probe_name, mbs, condition) in enumerate(probes):
                if TARGET_GBS % (gpu_count * mbs) != 0:
                    raise ValueError(
                        f"Boundary MBS is incompatible with GBS: {family_id}/{mbs}"
                    )
                job = _base_job(
                    experiment=experiment,
                    model=models[model_id],
                    role=role,
                    cutoff_len=cutoff_len,
                    train_type=train_type,
                    gpu_count=gpu_count,
                    zero=zero,
                    mbs=mbs,
                    target_gbs=TARGET_GBS,
                    packing=False,
                )
                job.update(
                    {
                        "request_id": family_id,
                        "kind": "throughput_screen",
                        "fidelity": "calibration_boundary_short",
                        "warmup_steps": BOUNDARY_WARMUP_STEPS,
                        "measure_steps": BOUNDARY_MEASURE_STEPS,
                        "selector_id": selector_id,
                        "calibration_evidence_class": "unpacked_boundary_and_efficiency",
                        "boundary_probe": {
                            "family_id": family_id,
                            "probe": probe_name,
                            "sequence_index": sequence_index,
                            "condition": condition,
                            "stop_after_first_oom_on_upward_path": True,
                            "first_probe_oom_allows_only_halving_fallback": True,
                            "mbs16_success_means_covered_domain_fully_feasible": True,
                        },
                    }
                )
                job["job_id"] = stable_id(
                    "h800cal",
                    {**family_identity, "probe": probe_name, "mbs": mbs},
                )
                family_job_ids.append(job["job_id"])
                jobs.append(job)
            families.append(
                {
                    "family_id": family_id,
                    "selector_id": selector_id,
                    "partition": _partition(role, model_id, cutoff_len),
                    "model_id": model_id,
                    "cutoff_len": cutoff_len,
                    "gpu_count": gpu_count,
                    "zero_stage": 0 if zero == "none" else 3,
                    "training_mode": train_type,
                    "anchor_mbs": anchor,
                    "probe_order": [name for name, _, _ in probes],
                    "ordered_job_ids": family_job_ids,
                    "maximum_runs": len(probes),
                    "minimum_expected_runs_to_bracket": 2,
                }
            )
    return jobs, families


def _packing_ga(
    analysis: dict[str, Any], dataset_id: str, gpu_count: int
) -> tuple[int, int, float, float]:
    packing = analysis["datasets"][dataset_id]["profiles"]["qwen3_nothink"][
        "packing"
    ]
    if packing.get("packing_eligible_for_paired_test") is not True:
        raise ValueError(f"Dataset is not eligible for paired packing: {dataset_id}")
    matches = [
        row
        for row in packing.get("ga_table") or ()
        if row.get("data_parallel") == gpu_count
        and row.get("target_gbs") in {PACKING_TARGET_GBS, 256}
        and float(row.get("relative_error", 1.0)) <= 0.05
    ]
    if not matches:
        raise ValueError(
            f"No <=5% packing GBS match for {dataset_id}/{gpu_count} GPU"
        )
    row = min(matches, key=lambda value: int(value["target_gbs"]))
    return (
        int(row["target_gbs"]),
        int(row["gradient_accumulation_steps"]),
        float(row["expected_sample_gbs"]),
        float(row["relative_error"]),
    )


def _packing_jobs(
    experiment: dict[str, Any],
    models: dict[str, dict[str, Any]],
    analysis: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    treatment_order = ("unpacked", "packed", "packed", "unpacked")
    for train_type, specifications in PACKING_PAIRS.items():
        for role, model_id, cutoff_len, gpu_count, unpacked_mbs in specifications:
            zero = "none" if gpu_count == 1 else "zero3"
            dataset_id = DATASETS[cutoff_len]
            pair_target_gbs, packed_ga, expected_gbs, relative_error = _packing_ga(
                analysis, dataset_id, gpu_count
            )
            pair_identity = {
                "candidate": "h800_calibration_candidate_v1",
                "role": role,
                "model_id": model_id,
                "cutoff_len": cutoff_len,
                "gpu_count": gpu_count,
                "zero": zero,
                "train_type": train_type,
            }
            pair_id = stable_id("h800packpair", pair_identity)
            pair_job_ids: list[str] = []
            prior_success_ids: list[str] = []
            for sequence_index, treatment in enumerate(treatment_order):
                packing = treatment == "packed"
                mbs = 1 if packing else unpacked_mbs
                job = _base_job(
                    experiment=experiment,
                    model=models[model_id],
                    role=role,
                    cutoff_len=cutoff_len,
                    train_type=train_type,
                    gpu_count=gpu_count,
                    zero=zero,
                    mbs=mbs,
                    target_gbs=pair_target_gbs,
                    packing=packing,
                )
                repeat = 0 if sequence_index < 2 else 1
                job.update(
                    {
                        "request_id": pair_id,
                        "kind": "throughput",
                        "fidelity": "packing_paired_short",
                        "repeat": repeat,
                        "warmup_steps": PACKING_WARMUP_STEPS,
                        "measure_steps": PACKING_MEASURE_STEPS,
                        "selector_id": f"packing_{train_type}_paired",
                        "calibration_evidence_class": "packing_paired_only",
                        "packing_pair": {
                            "pair_id": pair_id,
                            "order": "ABBA",
                            "sequence_index": sequence_index,
                            "treatment": treatment,
                            "condition": (
                                {"type": "always"}
                                if not prior_success_ids
                                else {
                                    "type": "all_jobs_succeeded",
                                    "job_ids": list(prior_success_ids),
                                }
                            ),
                            "expected_sample_gbs": (
                                expected_gbs if packing else pair_target_gbs
                            ),
                            "expected_gbs_relative_error": (
                                relative_error if packing else 0.0
                            ),
                        },
                    }
                )
                if packing:
                    job["gradient_accumulation_steps"] = packed_ga
                job["job_id"] = stable_id(
                    "h800pack",
                    {
                        **pair_identity,
                        "sequence_index": sequence_index,
                        "treatment": treatment,
                        "repeat": repeat,
                    },
                )
                pair_job_ids.append(job["job_id"])
                prior_success_ids.append(job["job_id"])
                jobs.append(job)
            pairs.append(
                {
                    "pair_id": pair_id,
                    "partition": _partition(role, model_id, cutoff_len),
                    "model_id": model_id,
                    "cutoff_len": cutoff_len,
                    "gpu_count": gpu_count,
                    "zero_stage": 0 if zero == "none" else 3,
                    "training_mode": train_type,
                    "order": "ABBA",
                    "unpacked_mbs": unpacked_mbs,
                    "packed_physical_mbs": 1,
                    "packed_gradient_accumulation_steps": packed_ga,
                    "target_sample_gbs": pair_target_gbs,
                    "ordered_job_ids": pair_job_ids,
                }
            )
    return jobs, pairs


def _budget(jobs: list[dict[str, Any]], boundary_jobs: int) -> dict[str, Any]:
    gpu_run_units = sum(int(job["gpu_count"]) for job in jobs)
    counts_by_gpu = Counter(int(job["gpu_count"]) for job in jobs)
    ideal_slot_hours = sum(
        count * RUN_TIMEOUT_SECONDS / 3600.0 / (4 // gpu_count)
        for gpu_count, count in counts_by_gpu.items()
    )
    return {
        "maximum_candidate_runs": len(jobs),
        "maximum_boundary_runs_before_early_stop": boundary_jobs,
        "packing_pair_runs": len(jobs) - boundary_jobs,
        "gpu_run_units": gpu_run_units,
        "runs_by_gpu_count": {str(key): counts_by_gpu[key] for key in sorted(counts_by_gpu)},
        "per_run_timeout_seconds": RUN_TIMEOUT_SECONDS,
        "ideal_resource_slot_upper_bound_hours": ideal_slot_hours,
        "planning_overhead_multiplier": 1.5,
        "planned_wall_time_upper_bound_hours": ideal_slot_hours * 1.5,
        "hard_wall_time_stop_hours": HARD_WALL_TIME_HOURS,
        "hard_gpu_hour_stop": HARD_WALL_TIME_HOURS * 4,
        "stop_is_fail_closed": True,
        "note": (
            "The static JSONL is the maximum authorization envelope. Conditional "
            "boundary and packing rules may skip rows after a bracket "
            "or failed prerequisite is observed."
        ),
    }


def build_candidate(
    project_root: Path = ROOT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Purely construct and validate the deterministic candidate payload."""

    root = project_root.resolve()
    experiment, hardware, inventory, analysis = _load_inputs(root)
    _validate_h800_sources(experiment, hardware)
    models = _model_map(inventory)
    boundary_jobs, families = _boundary_jobs(experiment, models)
    packing_jobs, pairs = _packing_jobs(experiment, models, analysis)
    jobs = [*boundary_jobs, *packing_jobs]
    job_bytes = _canonical_jsonl_bytes(jobs)
    job_payload_hashes = [sha256_json(job) for job in jobs]
    source_manifest = _source_manifest(root)
    budget = _budget(jobs, len(boundary_jobs))
    design: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate_state": "unapproved_offline_only",
        "training_started": False,
        "execution_authorized": False,
        "compatible_with_live_promotion": False,
        "requires_fresh_explicit_human_approval": True,
        "safety": {
            "writes_live_approval": False,
            "writes_live_queue": False,
            "launches_gpu_process": False,
            "allowed_output_root": "artifacts/candidates",
        },
        "scope": {
            "gpu_family": "H800",
            "gpu_type": HARDWARE_TYPE,
            "required_runtime_gpu_name": HARDWARE_NAME,
            "required_memory_bytes": int(hardware["memory_bytes_reported_by_torch"]),
            "allowed_gpu_counts": [1, 2, 4],
            "allowed_micro_batch_sizes": list(MBS_DOMAIN),
            "placement_policy": {
                "1": {"zero_stage": 0, "deepspeed": False},
                "2": {"zero_stage": 3, "deepspeed": True},
                "4": {"zero_stage": 3, "deepspeed": True},
            },
            "training_modes": ["full", "lora"],
            "gradient_checkpointing": True,
            "precision": "bf16",
            "packing": [False, True],
            "forbidden_gpu_identity_patterns": list(FORBIDDEN_GPU_PATTERNS),
            "partial_other_gpu_results_may_enter_fit": False,
        },
        "calibration_partition": {
            "policy": SPLIT_POLICY,
            "frozen_before_execution": True,
            "split_unit": "model_id@cutoff_len",
            "calibration": [
                _split_unit_id(model_id, length)
                for model_id, length in SPLIT_UNITS["calibration"]
            ],
            "holdout": [
                _split_unit_id(model_id, length)
                for model_id, length in SPLIT_UNITS["holdout"]
            ],
        },
        "runtime_contract": {
            "flash_attention": "fa3:orig",
            "liger_kernel": True,
            "chunked_cross_entropy": False,
            "optimizer": "adamw_torch_fused",
            "torch_compile": False,
            "lora_rank": 32,
            "mechanism_fingerprint_must_be_identical_across_rows": True,
            "actual_hardware_attestation_required": True,
            "attempt_scoped_rank_evidence_required": True,
        },
        "measurement": {
            "boundary": {
                "warmup_steps": BOUNDARY_WARMUP_STEPS,
                "measure_steps": BOUNDARY_MEASURE_STEPS,
                "target_gbs": TARGET_GBS,
                "per_run_timeout_seconds": RUN_TIMEOUT_SECONDS,
            },
            "packing_pair": {
                "order": "ABBA",
                "warmup_steps": PACKING_WARMUP_STEPS,
                "measure_steps": PACKING_MEASURE_STEPS,
                "target_sample_gbs_options": [64, 256],
                "maximum_gbs_relative_error": 0.05,
                "per_run_timeout_seconds": RUN_TIMEOUT_SECONDS,
            },
        },
        "boundary_early_stop": {
            "policy": "anchor_then_one_directional_bracket_v1",
            "run_anchor_first": True,
            "if_anchor_oom": "run_exactly_one_predeclared_half-anchor fallback then stop",
            "if_anchor_success": "double until first OOM or MBS=16, then stop",
            "if_mbs16_success": "record planner coverage as fully feasible; never probe MBS>16",
            "software_or_infrastructure_failure": "repair and rerun the same job; never label infeasible",
            "unexecuted_conditional_rows_are_not_failures": True,
            "legacy_anchor_policy": (
                "legacy-incomplete H800 rows select only the first probe; every "
                "calibration label is produced by a new approved attempt"
            ),
            "families": families,
        },
        "packing_study": {
            "evidence_class": "packing_paired_only",
            "does_not_replace_unpacked_oom_boundary_evidence": True,
            "analysis": "paired log throughput ratio with mode/stage covariates",
            "pairs": pairs,
        },
        "budget": budget,
        "acceptance": {
            "unsafe_holdout_oom_predicted_safe": 0,
            "successful_holdout_peak_below_p95_fraction_minimum": 0.95,
            "p95_capacity_ceiling_fraction": 0.95,
            "maximum_holdout_throughput_regret_fraction": 0.10,
            "minimum_measured_doubling_speedup_to_recommend": 1.8,
            "minimum_unpacked_calibration_feasibility_rows_per_selector": 6,
            "minimum_unpacked_calibration_throughput_rows_per_selector": 4,
            "minimum_unpacked_holdout_feasibility_rows_per_selector": 4,
            "minimum_unpacked_holdout_throughput_rows_per_selector": 2,
            "boundary_completion_rule": (
                "success/OOM bracket, or verified success at MBS=16 proving the "
                "entire supported MBS domain is feasible"
            ),
            "out_of_domain_oom_is_not_required": True,
            "packing_minimum_calibration_pairs_per_training_mode": 2,
            "packing_minimum_holdout_pairs_per_training_mode": 2,
            "packing_recommendation_requires_paired_p10_speedup": 1.05,
            "packing_requires_zero_new_oom": True,
            "publication_requires_separate_approved_holdout_report": True,
        },
        "legacy_evidence_policy": {
            "canonical_observations_source": "artifacts/canonical_h800_observations.jsonl",
            "historical_rows_are_navigation_only": True,
            "legacy_incomplete_rows_may_calibrate": False,
            "readiness_source": "artifacts/h800_calibration_readiness.json",
        },
        "source_file_sha256": source_manifest,
        "jobs_binding": {
            "path": f"artifacts/candidates/{JOBS_NAME}",
            "format": "canonical-jsonl",
            "rows": len(jobs),
            "sha256": _sha256_bytes(job_bytes),
            "ordered_job_ids": [job["job_id"] for job in jobs],
            "ordered_job_payload_sha256": job_payload_hashes,
            "ordered_payload_manifest_sha256": sha256_json(job_payload_hashes),
        },
    }
    design["canonical_identity"] = {
        "algorithm": "sha256(canonical-json(design-without-canonical_identity))",
        "sha256": sha256_json(design),
    }
    validation = validate_candidate(design, jobs, root)
    if not validation["all_passed"]:
        raise ValueError(f"Generated H800 candidate failed validation: {validation}")
    return design, jobs


def validate_candidate(
    design: dict[str, Any],
    jobs: list[dict[str, Any]],
    project_root: Path = ROOT,
) -> dict[str, Any]:
    """Recompute every static safety, split, shape, budget, and hash invariant."""

    root = project_root.resolve()
    errors: list[str] = []
    checks: dict[str, bool] = {}

    def check(name: str, passed: bool, detail: str) -> None:
        checks[name] = bool(passed)
        if not passed:
            errors.append(detail)

    canonical = design.get("canonical_identity") or {}
    material = dict(design)
    material.pop("canonical_identity", None)
    check("schema", design.get("schema") == SCHEMA, "candidate schema is invalid")
    check(
        "canonical_hash",
        canonical.get("sha256") == sha256_json(material),
        "candidate canonical hash is invalid",
    )
    check(
        "unapproved",
        design.get("execution_authorized") is False
        and design.get("training_started") is False
        and design.get("compatible_with_live_promotion") is False,
        "candidate is not fail-closed",
    )
    manifest = design.get("source_file_sha256") or {}
    manifest_current = set(manifest) == set(SOURCE_FILES) and all(
        (root / relative).is_file()
        and sha256_file(root / relative) == expected
        for relative, expected in manifest.items()
    )
    check("sources_current", manifest_current, "candidate source manifest is stale")

    binding = design.get("jobs_binding") or {}
    job_bytes = _canonical_jsonl_bytes(jobs)
    payload_hashes = [sha256_json(job) for job in jobs]
    ids = [job.get("job_id") for job in jobs]
    check("jobs_nonempty_unique", bool(ids) and len(ids) == len(set(ids)), "job IDs are empty or duplicated")
    check("jobs_count", binding.get("rows") == len(jobs), "bound job count differs")
    check("jobs_sha256", binding.get("sha256") == _sha256_bytes(job_bytes), "jobs JSONL hash differs")
    check("job_order", binding.get("ordered_job_ids") == ids, "ordered job IDs differ")
    check(
        "job_payload_hashes",
        binding.get("ordered_job_payload_sha256") == payload_hashes
        and binding.get("ordered_payload_manifest_sha256") == sha256_json(payload_hashes),
        "ordered job payload hashes differ",
    )
    check("run_cap", 0 < len(jobs) <= MAX_CANDIDATE_RUNS, "candidate exceeds the run cap")

    role_mapping = _role_map()
    selector_roles: dict[str, Counter[str]] = defaultdict(Counter)
    packing_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_errors: list[str] = []
    for index, job in enumerate(jobs):
        identity = json.dumps(job, ensure_ascii=False, sort_keys=True).lower()
        partition = job.get("calibration_partition") or {}
        unit = (str(job.get("model_id")), int(job.get("cutoff_len") or 0))
        role = role_mapping.get(unit)
        gpu_count = job.get("gpu_count")
        zero = job.get("zero")
        if job.get("schema") != JOB_SCHEMA:
            row_errors.append(f"row {index}: schema")
        if (
            job.get("gpu_type") != HARDWARE_TYPE
            or job.get("required_runtime_gpu_name") != HARDWARE_NAME
            or job.get("hardware_id") != HARDWARE_ID
            or job.get("required_memory_bytes") != 150142189568
            or job.get("forbid_mig") is not True
            or any(pattern in identity for pattern in FORBIDDEN_GPU_PATTERNS)
        ):
            row_errors.append(f"row {index}: non-H800 identity")
        if gpu_count not in {1, 2, 4}:
            row_errors.append(f"row {index}: GPU count")
        if (gpu_count == 1 and zero != "none") or (
            gpu_count in {2, 4} and zero != "zero3"
        ):
            row_errors.append(f"row {index}: stage placement")
        if job.get("gc") is not True or job.get("precision") != "bf16":
            row_errors.append(f"row {index}: fixed runtime selector")
        if role is None or partition != _partition(role, *unit):
            row_errors.append(f"row {index}: partition")
        if job.get("execution_authorized") is not False:
            row_errors.append(f"row {index}: authorization")
        mbs = job.get("mbs")
        target_gbs = job.get("target_gbs")
        if type(mbs) is not int or mbs not in MBS_DOMAIN:
            row_errors.append(f"row {index}: MBS")
        if not job.get("packing") and (
            type(target_gbs) is not int
            or type(gpu_count) is not int
            or type(mbs) is not int
            or target_gbs % (gpu_count * mbs) != 0
        ):
            row_errors.append(f"row {index}: GBS divisibility")
        if job.get("kind") == "throughput_screen":
            if (
                job.get("warmup_steps") != BOUNDARY_WARMUP_STEPS
                or job.get("measure_steps") != BOUNDARY_MEASURE_STEPS
                or job.get("packing") is not False
            ):
                row_errors.append(f"row {index}: boundary measurement")
            selector_roles[str(job.get("selector_id"))][role] += 1
        elif job.get("kind") == "throughput":
            pair = job.get("packing_pair") or {}
            packing_pairs[str(pair.get("pair_id"))].append(job)
            if (
                job.get("warmup_steps") != PACKING_WARMUP_STEPS
                or job.get("measure_steps") != PACKING_MEASURE_STEPS
                or job.get("calibration_evidence_class") != "packing_paired_only"
            ):
                row_errors.append(f"row {index}: packing measurement")
            if job.get("packing") and (
                mbs != 1
                or type(job.get("gradient_accumulation_steps")) is not int
                or job["gradient_accumulation_steps"] <= 0
            ):
                row_errors.append(f"row {index}: packed physical batch")
        else:
            row_errors.append(f"row {index}: unsupported kind")
    check("job_rows", not row_errors, "; ".join(row_errors))

    split = design.get("calibration_partition") or {}
    calibration_units = set(split.get("calibration") or ())
    holdout_units = set(split.get("holdout") or ())
    check(
        "split_disjoint",
        bool(calibration_units)
        and bool(holdout_units)
        and calibration_units.isdisjoint(holdout_units),
        "calibration and holdout split units overlap",
    )
    expected_selectors = {
        "stage0_full_unpacked",
        "stage0_lora_unpacked",
        "stage3_full_unpacked",
        "stage3_lora_unpacked",
    }
    check(
        "unpacked_selector_coverage",
        set(selector_roles) == expected_selectors
        and all(
            roles["calibration"] >= 6 and roles["holdout"] >= 4
            for roles in selector_roles.values()
        ),
        "unpacked selector calibration/holdout coverage is insufficient",
    )
    pair_errors = []
    pair_role_mode: Counter[tuple[str, str]] = Counter()
    for pair_id, rows in packing_pairs.items():
        rows.sort(key=lambda row: int((row.get("packing_pair") or {})["sequence_index"]))
        treatments = [(row.get("packing_pair") or {}).get("treatment") for row in rows]
        if len(rows) != 4 or treatments != ["unpacked", "packed", "packed", "unpacked"]:
            pair_errors.append(pair_id)
            continue
        partition = rows[0]["calibration_partition"]
        pair_role_mode[(str(rows[0]["train_type"]), str(partition["role"]))] += 1
    check("packing_abba", not pair_errors and len(packing_pairs) == 8, f"invalid packing pairs: {pair_errors}")
    check(
        "packing_partition_coverage",
        all(
            pair_role_mode[(mode, role)] >= 2
            for mode in ("full", "lora")
            for role in ("calibration", "holdout")
        ),
        "packing pairs lack two independent units per mode/role",
    )

    budget = design.get("budget") or {}
    check(
        "budget",
        budget.get("maximum_candidate_runs") == len(jobs)
        and float(budget.get("planned_wall_time_upper_bound_hours") or 999)
        <= HARD_WALL_TIME_HOURS
        and budget.get("hard_wall_time_stop_hours") == HARD_WALL_TIME_HOURS,
        "candidate budget exceeds 72 hours",
    )
    return {"checks": checks, "errors": errors, "all_passed": all(checks.values())}


def _safe_output_dir(project_root: Path, output_dir: Path | None) -> Path:
    root = project_root.resolve()
    allowed = (root / "artifacts" / "candidates").resolve()
    output = (output_dir or allowed).resolve()
    try:
        output.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"Candidate output must stay under {allowed}") from error
    live_paths = {
        (root / "config" / "APPROVED_TO_RUN.json").resolve(),
        (root / "runtime" / "approval_design.json").resolve(),
        (root / "runtime" / "queue.json").resolve(),
    }
    if output in live_paths or any(parent in live_paths for parent in output.parents):
        raise ValueError("Candidate output must not target live approval/queue state")
    return output


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_candidate(
    design: dict[str, Any],
    jobs: list[dict[str, Any]],
    *,
    project_root: Path = ROOT,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    reject_retired_approval_plan(
        design,
        operation="write a new candidate",
    )
    output = _safe_output_dir(project_root, output_dir)
    design_path = output / DESIGN_NAME
    jobs_path = output / JOBS_NAME
    jobs_payload = _canonical_jsonl_bytes(jobs)
    design_payload = (
        json.dumps(design, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    _atomic_write(jobs_path, jobs_payload)
    _atomic_write(design_path, design_payload)
    return {
        "candidate_path": str(design_path),
        "candidate_file_sha256": sha256_file(design_path),
        "candidate_canonical_sha256": design["canonical_identity"]["sha256"],
        "jobs_path": str(jobs_path),
        "jobs_sha256": sha256_file(jobs_path),
        "jobs": len(jobs),
        "execution_authorized": False,
        "training_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an unapproved, offline-only H800 calibration candidate."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "candidates",
        help="Must remain under artifacts/candidates.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Build and validate in memory without writing any file.",
    )
    args = parser.parse_args()
    if not args.verify_only:
        reject_retired_approval_plan(
            {"schema": SCHEMA},
            operation="write a new candidate",
        )
    design, jobs = build_candidate(ROOT)
    if args.verify_only:
        result = {
            "candidate_canonical_sha256": design["canonical_identity"]["sha256"],
            "jobs_sha256": design["jobs_binding"]["sha256"],
            "jobs": len(jobs),
            "execution_authorized": False,
            "training_started": False,
            "retired": True,
            "retirement_policy": "historical_audit_only_no_write_no_promotion_no_execution",
            "written": False,
        }
    else:
        result = write_candidate(
            design, jobs, project_root=ROOT, output_dir=args.output_dir
        )
        result["written"] = True
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
