#!/usr/bin/env python3
"""Materialize the approved low-cutoff half of the Packing memory-boundary DOE.

This script is CPU-only.  It consumes the frozen eight-setting selection, keeps
only execution_order_within_chain=1, verifies the already-tokenized execution
profiles and writes four counterbalanced U/P pairs with two repeats per arm.
The high-cutoff half is deliberately absent from the queue.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    MATRIX_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from prepare_packing_dataprofile_v2 import _curve


SCHEMA = "sft_h800_packing_memory_boundary_stage1_design/v1"
JOB_SCHEMA = "sft_h800_packing_memory_boundary_stage1_job/v1"
CAMPAIGN_ID = "h800_packing_memory_boundary_20260805_v1"
PHASE_ID = "h800_packing_memory_boundary_stage1_v1"
# GPU 1/2 were occupied together at launch time.  The H800 pool is fully
# NVLinked, so order the scheduler slots as the idle pair [0,3] and the busy
# pair [1,2] without changing the authorized physical device set.
GPU_IDS = (0, 3, 1, 2)
GPU_COUNT = 2
MBS = 1
WARMUP_STEPS = 1
MEASURE_STEPS = 4
TOTAL_STEPS = WARMUP_STEPS + MEASURE_STEPS
EPSILON_GBS = 0.10

SELECTION = ARTIFACT_DIR / "h800_packing_memory_boundary_selection_v1.json"
MODEL_REFIT = ARTIFACT_DIR / "h800_packing_phase_c_model_refit_v1.json"
PHASE_C = ARTIFACT_DIR / "h800_packing_profile_phase_c_results_v1.json"
MODEL_INVENTORY = ARTIFACT_DIR / "model_inventory.json"
HARDWARE = ROOT / "config/hardware.json"
QUEUE = MATRIX_DIR / "h800_packing_memory_boundary_stage1_v1.jsonl"
STATIC = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_static_v1.json"
DESIGN = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_design_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_queue_manifest_v1.json"
JOB_DIR = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_jobs_v1"

SEQUENCES = (
    (False, True, True, False),
    (True, False, False, True),
)

DATASET_CATEGORIES = {
    "W3": "multiturn",
    "W7": "longtail",
    # The predictor has no code/structured release category.  Longtail is the
    # same explicit execution-only transfer label used by Phase B.
    "W8": "longtail",
}


def _binding(path: Path, **extra: Any) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), **extra}


def _profile_curve(profile_path: Path, cutoff_len: int) -> dict[str, Any]:
    rows = read_jsonl(profile_path)
    if not rows or any("total_tokens" not in row for row in rows):
        raise ValueError(f"invalid cached token profile: {profile_path}")
    return _curve([int(row["total_tokens"]) for row in rows], [cutoff_len])[0]


def _models() -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in read_json(MODEL_INVENTORY)["models"]}


def _lower_settings() -> list[dict[str, Any]]:
    selection = read_json(SELECTION)
    if (
        selection.get("schema") != "sft_h800_packing_memory_boundary_selection/v1"
        or selection.get("campaign_id") != CAMPAIGN_ID
        or selection.get("design", {}).get("selected_settings") != 8
        or selection.get("design", {}).get("all_selected_gbs_contracts_passed") is not True
        or selection.get("execution_contract", {}).get("first_run_lower_target_per_chain") is not True
        or selection.get("gates", {}).get("automatic_gpu_launch_allowed") is not False
    ):
        raise ValueError("frozen boundary selection drifted")
    rows = [
        row
        for row in selection["selected_settings"]
        if int(row["execution_order_within_chain"]) == 1
    ]
    if len(rows) != 4 or len({str(row["chain_id"]) for row in rows}) != 4:
        raise ValueError("selection does not contain exactly four lower settings")
    return rows


def prepare() -> dict[str, Any]:
    settings = _lower_settings()
    models = _models()
    static_settings: list[dict[str, Any]] = []
    runtime_by_setting: dict[str, dict[str, Any]] = {}

    for setting in settings:
        arms = list(setting["arms"])
        if len(arms) != 2 or {bool(row["packing"]) for row in arms} != {False, True}:
            raise ValueError(f"{setting['boundary_setting_id']}: not a U/P pair")
        unpacked = next(row for row in arms if not row["packing"])
        packed = next(row for row in arms if row["packing"])
        profile_path = Path(packed["dataset_profile_path"])
        data_path = Path(packed["data_path"])
        dataprofile_path = (
            ARTIFACT_DIR
            / "packing_dataprofile_v2"
            / f"{str(packed['workload_id']).lower()}_packing_dataprofile_v2.json"
        )
        for path in (profile_path, data_path, dataprofile_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        profile_rows = read_jsonl(profile_path)
        data_rows = read_jsonl(data_path)
        if len(profile_rows) != len(data_rows):
            raise ValueError(f"{setting['boundary_setting_id']}: data/profile rows differ")
        curve = _profile_curve(profile_path, int(setting["selected_cutoff_len"]))
        n_pack_mean = float(curve["samples_per_pack"]["mean"])
        # Re-evaluate the integer GA on the exact repeated execution snapshot.
        # This repairs W3@14336 from GA=6 (10.29% error) to GA=5 (8.09%).
        raw_ga = float(packed["target_gbs"]) / (GPU_COUNT * n_pack_mean)
        packed_ga = max(1, round(raw_ga))
        alternatives = sorted({max(1, math.floor(raw_ga)), max(1, math.ceil(raw_ga)), packed_ga})
        packed_ga = min(
            alternatives,
            key=lambda value: (
                abs(GPU_COUNT * value * n_pack_mean - float(packed["target_gbs"])),
                value,
            ),
        )
        expected_packed_gbs = GPU_COUNT * packed_ga * n_pack_mean
        expected_error = abs(expected_packed_gbs - float(packed["target_gbs"])) / float(
            packed["target_gbs"]
        )
        global_step_p99 = GPU_COUNT * float(curve["samples_per_pack"]["p99"])
        if expected_error > EPSILON_GBS or global_step_p99 > float(packed["target_gbs"]) * (1 + EPSILON_GBS):
            raise ValueError(f"{setting['boundary_setting_id']}: exact runtime GBS contract failed")
        first_epoch_capacity = {
            "total_probe_optimizer_steps": TOTAL_STEPS,
            "unpacked": {
                "available_examples": len(data_rows),
                "required_examples": GPU_COUNT * int(unpacked["gradient_accumulation_steps"]) * TOTAL_STEPS,
            },
            "packed": {
                "available_packs": int(curve["packs"]),
                "required_packs": GPU_COUNT * packed_ga * TOTAL_STEPS,
            },
        }
        for branch in ("unpacked", "packed"):
            item = first_epoch_capacity[branch]
            item["passed"] = item[f"available_{'examples' if branch == 'unpacked' else 'packs'}"] >= item[f"required_{'examples' if branch == 'unpacked' else 'packs'}"]
        if not all(first_epoch_capacity[branch]["passed"] for branch in ("unpacked", "packed")):
            raise ValueError(f"{setting['boundary_setting_id']}: probe crosses epoch boundary")
        runtime = {
            "setting": setting,
            "unpacked": unpacked,
            "packed": packed,
            "profile_path": profile_path,
            "data_path": data_path,
            "dataprofile_path": dataprofile_path,
            "curve": curve,
            "packed_ga": packed_ga,
            "expected_packed_gbs": expected_packed_gbs,
            "expected_error": expected_error,
            "global_step_p99": global_step_p99,
            "first_epoch_capacity": first_epoch_capacity,
        }
        runtime_by_setting[str(setting["boundary_setting_id"])] = runtime
        static_settings.append(
            {
                "boundary_setting_id": setting["boundary_setting_id"],
                "chain_id": setting["chain_id"],
                "cutoff_len": setting["selected_cutoff_len"],
                "target_capacity_fraction": setting["target_capacity_fraction"],
                "predicted_pair_center_capacity_fraction": setting[
                    "pair_refit_center_capacity_fraction"
                ],
                "target_gbs": packed["target_gbs"],
                "selection_packed_ga": packed["gradient_accumulation_steps"],
                "execution_snapshot_packed_ga": packed_ga,
                "execution_snapshot_expected_sample_gbs": expected_packed_gbs,
                "execution_snapshot_expected_sample_gbs_relative_error": expected_error,
                "execution_snapshot_global_microstep_sample_p99": global_step_p99,
                "execution_curve": curve,
                "first_epoch_capacity": first_epoch_capacity,
                "refit_domain_relation": packed["refit_domain_relation"],
            }
        )

    static: dict[str, Any] = {
        "schema": "sft_h800_packing_memory_boundary_stage1_static/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_before_gpu": True,
        "recommendation_path_reads_raw_data": False,
        "recommendation_path_runs_full_packer": False,
        "materialization_reads_cached_token_profiles": True,
        "fixed_probe_optimizer_steps": TOTAL_STEPS,
        "gbs_epsilon": EPSILON_GBS,
        "settings": static_settings,
        "all_exact_runtime_gbs_contracts_passed": True,
        "all_first_epoch_capacity_checks_passed": True,
        "physical_p95_is_execution_admission_guard": False,
        "oom_is_right_censored_memory_evidence": True,
        "automatic_publication_allowed": False,
    }
    static["report_sha256"] = sha256_json(static)
    write_json(STATIC, static)

    counters = {
        str(setting["boundary_setting_id"]): {False: 0, True: 0}
        for setting in settings
    }
    jobs: list[dict[str, Any]] = []
    for block in range(4):
        for setting_index, setting in enumerate(settings):
            setting_id = str(setting["boundary_setting_id"])
            runtime = runtime_by_setting[setting_id]
            packed = SEQUENCES[setting_index % 2][block]
            repeat = counters[setting_id][packed]
            counters[setting_id][packed] += 1
            arm = runtime["packed"] if packed else runtime["unpacked"]
            model = models[str(arm["model_id"])]
            ga = runtime["packed_ga"] if packed else int(arm["gradient_accumulation_steps"])
            expected_gbs = runtime["expected_packed_gbs"] if packed else float(arm["target_gbs"])
            expected_error = runtime["expected_error"] if packed else 0.0
            pair_id = stable_id(
                "h800packboundarypair",
                {"campaign_id": CAMPAIGN_ID, "setting_id": setting_id, "repeat": repeat},
            )
            job: dict[str, Any] = {
                "schema": JOB_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "phase_id": PHASE_ID,
                "candidate_role": "packing_memory_boundary_low_anchor_fit_only",
                "boundary_stage": 1,
                "boundary_setting_id": setting_id,
                "chain_id": setting["chain_id"],
                "execution_order_within_chain": 1,
                "setting_id": setting_id,
                "family_id": setting_id,
                "scenario_id": f"{setting_id}-{arm['model_id']}-{arm['training_mode']}",
                "workload_id": arm["workload_id"],
                "packing_pair_id": pair_id,
                "packing_treatment": "packed" if packed else "unpacked",
                "counterbalance_block": block,
                "repeat": repeat,
                "model_id": arm["model_id"],
                "model_family": "qwen3",
                "model_path": str(Path(model["path"]).resolve()),
                "tokenizer_path": str(Path(model["tokenizer_path"]).resolve()),
                "model_parameters": int(model["actual_parameters"]),
                "train_type": arm["training_mode"],
                "dataset_id": arm["dataset_id"],
                "dataset_category": DATASET_CATEGORIES[str(arm["workload_id"])],
                "data_path": str(runtime["data_path"].resolve()),
                "data_sha256": sha256_file(runtime["data_path"]),
                "dataset_profile_path": str(runtime["profile_path"].resolve()),
                "dataset_profile_sha256": sha256_file(runtime["profile_path"]),
                "packing_dataprofile_path": str(runtime["dataprofile_path"].resolve()),
                "packing_dataprofile_sha256": sha256_file(runtime["dataprofile_path"]),
                "template": "qwen3_nothink",
                "cutoff_len": int(setting["selected_cutoff_len"]),
                "target_gbs": int(arm["target_gbs"]),
                "gpu_count": GPU_COUNT,
                "zero": f"zero{int(arm['zero_stage'])}",
                "zero_stage": int(arm["zero_stage"]),
                "gc": bool(arm["gc"]),
                "gradient_checkpointing": bool(arm["gc"]),
                "mbs": MBS,
                "gradient_accumulation_steps": ga,
                "packing": packed,
                "expected_sample_gbs": expected_gbs,
                "expected_sample_gbs_relative_error": expected_error,
                "n_pack_mean": float(runtime["curve"]["samples_per_pack"]["mean"]),
                "n_pack_step_p99": float(runtime["curve"]["samples_per_pack"]["p99"]),
                "global_microstep_sample_p99": runtime["global_step_p99"] if packed else float(GPU_COUNT),
                "first_epoch_capacity": runtime["first_epoch_capacity"],
                "offload": False,
                "gpu_type": "NVIDIA H800 140GB HBM3",
                "required_runtime_gpu_name": "NVIDIA H800",
                "hardware_id": "local_h800_140g",
                "kind": "throughput",
                "warmup_steps": WARMUP_STEPS,
                "measure_steps": MEASURE_STEPS,
                "max_samples": len(read_jsonl(runtime["data_path"])),
                "fidelity": "memory_boundary_1plus4",
                "parallel_class": "gpu_partitionable",
                "requires_external_node_idle": False,
                "execution_sequence_index": len(jobs),
                "target_capacity_fraction": setting["target_capacity_fraction"],
                "predicted_pair_center_capacity_fraction": setting[
                    "pair_refit_center_capacity_fraction"
                ],
                "predicted_arm_center_gib": arm["refit_center_gib"],
                "physical_operational_p95_gib_diagnostic_only": arm[
                    "physical_operational_p95_gib"
                ],
                "refit_domain_relation": arm["refit_domain_relation"],
                "declared_model_manifest_path": str(MODEL_INVENTORY.resolve()),
                "declared_model_manifest_sha256": sha256_file(MODEL_INVENTORY),
                "selection_path": str(SELECTION.resolve()),
                "selection_sha256": sha256_file(SELECTION),
                "static_features_path": str(STATIC.resolve()),
                "static_features_sha256": sha256_file(STATIC),
                "calibration_partition": {
                    "role": "calibration",
                    "split_unit_id": setting_id,
                    "policy": "packing_memory_boundary_low_anchor_never_acceptance_v1",
                },
                "oom_role": "right_censored_lower_bound",
                "high_cutoff_auto_release_allowed": False,
                "automatic_packing_recommendation_allowed": False,
                "publication_allowed": False,
            }
            job["job_id"] = stable_id("h800packboundary1", job)
            jobs.append(job)

    if len(jobs) != 16 or len({str(row["job_id"]) for row in jobs}) != 16:
        raise ValueError("stage-1 queue must contain 16 unique jobs")
    for setting in settings:
        setting_id = str(setting["boundary_setting_id"])
        subset = [row for row in jobs if row["boundary_setting_id"] == setting_id]
        seen = Counter((bool(row["packing"]), int(row["repeat"])) for row in subset)
        if len(subset) != 4 or len(seen) != 4 or set(seen.values()) != {1}:
            raise ValueError(f"{setting_id}: not a balanced two-repeat U/P pair")
    write_jsonl(QUEUE, jobs)
    for job in jobs:
        write_json(JOB_DIR / f"{job['job_id']}.json", job)

    source_files = {
        "selection": SELECTION,
        "model_refit": MODEL_REFIT,
        "phase_c_results": PHASE_C,
        "model_inventory": MODEL_INVENTORY,
        "hardware": HARDWARE,
        "experiment_config": ROOT / "config/experiment.json",
        "dataset_registry": ROOT / "data/dataset_info.json",
        "preparer": Path(__file__).resolve(),
        "freezer": ROOT / "scripts/freeze_h800_packing_memory_boundary_stage1_v1.py",
        "scheduler": ROOT / "scripts/scheduler.py",
        "run_job": ROOT / "scripts/run_job.py",
        "train_entry": ROOT / "scripts/train_entry.py",
        "metrics_callback": ROOT / "scripts/metrics_callback.py",
    }
    for setting in settings:
        runtime = runtime_by_setting[str(setting["boundary_setting_id"])]
        key = str(setting["chain_id"]).lower()
        source_files[f"data_{key}"] = runtime["data_path"]
        source_files[f"profile_{key}"] = runtime["profile_path"]
        source_files[f"dataprofile_{key}"] = runtime["dataprofile_path"]
    source_bindings = {name: _binding(path) for name, path in source_files.items()}
    design: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "stage1_frozen_before_gpu_waiting_for_exact_approval",
        "gpu_training_started": False,
        "objective": "Calibrate Packing U/P memory and throughput at the four lower boundary anchors.",
        "stage_policy": {
            "included_execution_order_within_chain": [1],
            "high_cutoff_jobs_in_this_queue": 0,
            "stage2_requires_all_repeats_success_by_chain": True,
            "stage2_automatic_release_allowed": False,
            "oom_role": "right_censored_lower_bound_not_regression_label",
            "software_failure_role": "repair_and_rerun_same_job",
        },
        "required_gpu_pool": {
            "gpu_ids": list(GPU_IDS),
            "max_gpu_count_per_job": 2,
            "maximum_parallel_jobs_when_all_idle": 2,
            "preview_two_gpu_masks_when_all_idle": [[0, 3], [1, 2]],
            "join_busy_pool": True,
            "preemption_allowed": False,
        },
        "measurement_contract": {
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "total_optimizer_steps": TOTAL_STEPS,
            "packing_treatment_order": "counterbalanced U-P-P-U / P-U-U-P",
            "repeats_per_treatment": 2,
        },
        "queue": {
            **_binding(QUEUE),
            "jobs": len(jobs),
            "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in jobs),
            "ordered_job_ids": [str(row["job_id"]) for row in jobs],
        },
        "static": _binding(STATIC),
        "source_bindings": source_bindings,
        "automatic_next_batch_allowed": False,
        "automatic_packing_recommendation_allowed": False,
        "publication_allowed": False,
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    manifest: dict[str, Any] = {
        "schema": "sft_h800_packing_memory_boundary_stage1_queue_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": _binding(DESIGN),
        "static": _binding(STATIC),
        "queue": _binding(
            QUEUE,
            jobs=len(jobs),
            ordered_job_ids=[str(row["job_id"]) for row in jobs],
        ),
        "stage2_materialized": False,
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(QUEUE_MANIFEST, manifest)
    return {
        "design": _binding(DESIGN),
        "queue": _binding(QUEUE),
        "static": _binding(STATIC),
        "jobs": len(jobs),
        "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in jobs),
        "gpu_ids": list(GPU_IDS),
        "high_cutoff_jobs_materialized": 0,
    }


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
