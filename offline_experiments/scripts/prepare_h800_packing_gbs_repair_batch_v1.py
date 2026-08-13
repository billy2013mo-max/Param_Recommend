#!/usr/bin/env python3
"""Materialize the 18-job W4 GBS-contract repair batch without launching GPUs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR, MATRIX_DIR, ROOT, read_json, sha256_file, sha256_json,
    stable_id, write_json, write_jsonl,
)
from packing_gbs_contract import derive_packing_gbs_contract


SCHEMA = "sft_h800_packing_gbs_repair_batch_design/v1"
JOB_SCHEMA = "sft_h800_packing_gbs_repair_batch_job/v1"
CAMPAIGN_ID = "h800_packing_gbs_repair_batch_20260804_v1"
PHASE_ID = "h800_packing_gbs_repair_batch_v1"
GPU_POOL = (0, 1, 5, 6, 7)
MODEL_ID = "qwen3_8b"
MODEL_PATH = Path("/wanqing-models/Qwen3-8B")
TEMPLATE = "qwen3_nothink"
CUTOFF = 32_768
WARMUP_STEPS = 2
MEASURE_STEPS = 8

BASE = ARTIFACT_DIR / "real_business_packing_cutoff_mbs_v1"
SUMMARY = BASE / "profile_summaries/real_4500_content_longtail_qwen3_v1.json"
PROFILE = ARTIFACT_DIR / "packing_dataprofile_v2/w4_packing_dataprofile_v2.json"
PROFILE_MANIFEST = ARTIFACT_DIR / "packing_data_profiles_w1_w9_manifest_v2.json"
SCREEN = ARTIFACT_DIR / "packing_cutoff_dp_gbs_screen_w1_w9_v2.json"
CANARY_RESULTS = ARTIFACT_DIR / "h800_packing_platform_v4_semantic_canary_results_v1.json"
INVENTORY = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_model_inventory_v1.json"
STATIC = ARTIFACT_DIR / "h800_packing_gbs_repair_batch_static_v1.json"
QUEUE = MATRIX_DIR / "h800_packing_gbs_repair_batch_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_packing_gbs_repair_batch_design_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_gbs_repair_batch_queue_manifest_v1.json"

UPPUUP = (False, True, True, False, False, True)
PUUPPU = (True, False, False, True, True, False)


@dataclass(frozen=True)
class Setting:
    setting_id: str
    display_name: str
    gpu_count: int
    target_gbs: int
    zero: str
    sequence: tuple[bool, ...]


SETTINGS = (
    Setting("w4_dp1_g64", "W4/DP1/GBS64", 1, 64, "none", UPPUUP),
    Setting("w4_dp2_g128_zero2", "W4/DP2/GBS128/ZeRO-2", 2, 128, "zero2", PUUPPU),
    Setting("w4_dp2_g128_zero3", "W4/DP2/GBS128/ZeRO-3", 2, 128, "zero3", UPPUUP),
)


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _curve_point() -> tuple[dict[str, Any], dict[str, Any]]:
    profile = read_json(PROFILE)
    if (
        profile.get("schema") != "sft_packing_data_profile/v2"
        or profile.get("workload_id") != "W4"
        or profile.get("recommendation_contract", {}).get("raw_lengths_read") is not False
    ):
        raise ValueError("W4 DataProfile v2 is absent or malformed")
    points = [row for row in profile["packing_curve"] if int(row["cutoff_len"]) == CUTOFF]
    if len(points) != 1:
        raise ValueError("W4 DataProfile lacks the exact cutoff=32768 curve point")
    return profile, points[0]


def prepare() -> dict[str, Any]:
    canary = read_json(CANARY_RESULTS)
    if canary.get("gates", {}).get("packing_semantics_and_ledger_passed") is not True:
        raise PermissionError("Packing semantic canary is not healthy")
    profile, point = _curve_point()
    contracts = {
        setting.setting_id: derive_packing_gbs_contract(
            target_gbs=setting.target_gbs,
            data_parallel=setting.gpu_count,
            samples_per_pack=point["samples_per_pack"],
            epsilon_gbs=0.10,
            maximum_center_relative_error=0.05,
        )
        for setting in SETTINGS
    }
    if not all(value["gates"]["candidate_admissible"] for value in contracts.values()):
        raise ValueError(f"a repair setting is not GBS-admissible: {contracts}")

    static: dict[str, Any] = {
        "schema": "sft_h800_packing_gbs_repair_batch_static/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_before_gpu": True,
        "recommendation_path_reads_raw_data": False,
        "recommendation_path_reads_raw_lengths": False,
        "recommendation_path_runs_full_packer": False,
        "profile": _binding(PROFILE),
        "profile_manifest": _binding(PROFILE_MANIFEST),
        "cpu_screen": _binding(SCREEN),
        "cutoff_curve_point": point,
        "contracts": contracts,
    }
    static["report_sha256"] = sha256_json(static)
    write_json(STATIC, static)

    summary = read_json(SUMMARY)
    model_parameters = int(read_json(INVENTORY)["models"][0]["actual_parameters"])
    counters = {setting.setting_id: {False: 0, True: 0} for setting in SETTINGS}
    jobs: list[dict[str, Any]] = []
    for block in range(6):
        for setting in SETTINGS:
            packing = setting.sequence[block]
            repeat = counters[setting.setting_id][packing]
            counters[setting.setting_id][packing] += 1
            if packing:
                contract = contracts[setting.setting_id]
                ga = int(contract["gradient_accumulation_steps"])
                expected = float(contract["expected_epoch_sample_gbs"])
                error = float(contract["expected_epoch_sample_gbs_relative_error"])
                arm_id = "P-C-1-gP"
            else:
                ga = setting.target_gbs // setting.gpu_count
                expected = float(setting.target_gbs)
                error = 0.0
                arm_id = "N-C-1-gN"
            row: dict[str, Any] = {
                "schema": JOB_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "phase_id": PHASE_ID,
                "candidate_role": "formal_packing_gbs_contract_repair",
                "setting_id": setting.setting_id,
                "profile_family_id": "w4_broad_long_tail",
                "family_id": setting.setting_id,
                "display_name": setting.display_name,
                "scenario_id": f"{setting.setting_id}-qwen3_8b-lora",
                "interaction_axis": "packing_x_dp_target_gbs_zero",
                "counterbalance_block": block,
                "arm_id": arm_id,
                "repeat": repeat,
                "model_id": MODEL_ID,
                "model_family": "qwen3",
                "model_path": str(MODEL_PATH),
                "tokenizer_path": str(MODEL_PATH),
                "template": TEMPLATE,
                "model_parameters": model_parameters,
                "train_type": "lora",
                "dataset_id": "real_4500_content_longtail_qwen3_v1",
                "profile_id": "real_4500_content_longtail_qwen3_v1",
                "dataset_category": "content_broad_long_tail",
                "source_dataset_records": int(summary["source"]["records"]),
                "frozen_slice_records": int(summary["slice"]["records"]),
                "data_path": summary["slice"]["data_path"],
                "data_sha256": summary["slice"]["data_sha256"],
                "dataset_profile_path": summary["profile"]["path"],
                "dataset_profile_sha256": summary["profile"]["sha256"],
                "packing_dataprofile_path": str(PROFILE.resolve()),
                "packing_dataprofile_sha256": sha256_file(PROFILE),
                "cutoff_label": "gbs_repair_event",
                "base_cutoff_len": CUTOFF,
                "cutoff_scale": 1,
                "cutoff_len": CUTOFF,
                "target_gbs": setting.target_gbs,
                "gpu_count": setting.gpu_count,
                "zero": setting.zero,
                "zero_stage": 0 if setting.zero == "none" else int(setting.zero[-1]),
                "gc": True,
                "gradient_checkpointing": True,
                "mbs": 1,
                "gradient_accumulation_steps": ga,
                "packing": packing,
                "expected_sample_gbs": expected,
                "expected_epoch_sample_gbs": expected,
                "expected_sample_gbs_relative_error": error,
                "n_pack_mean": point["samples_per_pack"]["mean"],
                "n_pack_step_p99": point["samples_per_pack"]["p99"],
                "gbs_contract_v2": contracts[setting.setting_id],
                "offload": False,
                "gpu_type": "NVIDIA H800 140GB HBM3",
                "required_runtime_gpu_name": "NVIDIA H800",
                "hardware_id": "local_h800_140g",
                "kind": "throughput",
                "warmup_steps": WARMUP_STEPS,
                "measure_steps": MEASURE_STEPS,
                "max_samples": int(summary["slice"]["records"]),
                "fidelity": "formal_gbs_repair_2plus8",
                "parallel_class": "gpu_partitionable",
                "requires_external_node_idle": False,
                "execution_sequence_index": len(jobs),
                "calibration_partition": {
                    "role": "calibration",
                    "split_unit_id": setting.setting_id,
                    "policy": "packing_gbs_contract_repair_fit_only_v1",
                },
                "declared_model_manifest_path": str(INVENTORY.resolve()),
                "declared_model_manifest_sha256": sha256_file(INVENTORY),
                "static_features_path": str(STATIC.resolve()),
                "static_features_sha256": sha256_file(STATIC),
                "matched_interaction_pair": True,
                "publication_allowed": False,
            }
            row["job_id"] = stable_id("h800packgbsfix", row)
            jobs.append(row)
    if len(jobs) != 18 or len({row["job_id"] for row in jobs}) != 18:
        raise ValueError("repair batch must contain 18 unique jobs")
    for setting in SETTINGS:
        subset = [row for row in jobs if row["setting_id"] == setting.setting_id]
        if {value: sum(row["packing"] is value for row in subset) for value in (False, True)} != {False: 3, True: 3}:
            raise ValueError(f"unbalanced setting: {setting.setting_id}")
    write_jsonl(QUEUE, jobs)
    job_dir = ARTIFACT_DIR / "h800_packing_gbs_repair_batch_jobs_v1"
    for job in jobs:
        write_json(job_dir / f"{job['job_id']}.json", job)

    source_files = {
        "plan": ROOT.parent / "Neat_Packing联合搜索_数据实验与建模完整计划_2026-08-04.md",
        "packing_decision": ROOT.parent / "Packing 决策逻辑 v2.md",
        "experiment_config": ROOT / "config/experiment.json",
        "hardware_config": ROOT / "config/hardware.json",
        "dataset_registry": ROOT / "data/dataset_info.json",
        "profile_schema": ARTIFACT_DIR / "packing_data_profile_schema_v2.json",
        "profile_manifest": PROFILE_MANIFEST,
        "w4_profile": PROFILE,
        "cpu_screen": SCREEN,
        "canary_results": CANARY_RESULTS,
        "summary": SUMMARY,
        "training_slice": Path(summary["slice"]["data_path"]),
        "training_token_profile": Path(summary["profile"]["path"]),
        "model_inventory": INVENTORY,
        "preparer": Path(__file__).resolve(),
        "freezer": ROOT / "scripts/freeze_h800_packing_gbs_repair_batch_v1.py",
        "evaluator": ROOT / "scripts/evaluate_h800_packing_gbs_repair_batch_v1.py",
        "run_job": ROOT / "scripts/run_job.py",
        "scheduler": ROOT / "scripts/scheduler.py",
        "train_entry": ROOT / "scripts/train_entry.py",
        "metrics_callback": ROOT / "scripts/metrics_callback.py",
    }
    design: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_gpu_waiting_for_exact_approval",
        "gpu_training_started": False,
        "fit_allowed_after_complete_results": True,
        "automatic_next_batch_allowed": False,
        "publication_allowed": False,
        "objective": "Repair W4 target-GBS calibration with only v2-admissible DP/GBS configurations.",
        "required_gpu_pool": {
            "gpu_ids": list(GPU_POOL),
            "max_gpu_count_per_job": 2,
            "maximum_parallel_gpu_slots": len(GPU_POOL),
            "two_gpu_masks": [[0, 1], [5, 6]],
            "single_gpu_fallback": [7],
            "preemption_allowed": False,
            "excluded_busy_gpu_ids": [2, 3, 4],
            "join_busy_pool": True,
        },
        "queue": {**_binding(QUEUE), "job_count": 18, "gpu_job_equivalents": 30, "ordered_job_ids": [row["job_id"] for row in jobs]},
        "matrix": {
            "settings": [
                {
                    "setting_id": setting.setting_id, "gpu_count": setting.gpu_count,
                    "target_gbs": setting.target_gbs, "zero": setting.zero,
                    "cutoff_len": CUTOFF,
                    "sequence": ["P" if value else "U" for value in setting.sequence],
                    "treatment_repeats": 3,
                    "contract": contracts[setting.setting_id],
                }
                for setting in SETTINGS
            ],
            "jobs": 18,
            "gpu_job_equivalents": 30,
        },
        "measurement_contract": {
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "token_source": "consumed_token_ledger/v1",
            "expected_epoch_gbs_separate_from_probe_window_gbs": True,
            "packed_semantics_required_on_every_rank": True,
            "authoritative_ledger_required_on_every_rank": True,
            "global_metrics_sum_across_ranks": True,
            "counterbalance": "U-P-P-U-U-P_or_mirror",
        },
        "static_features": _binding(STATIC),
        "model_inventory": _binding(INVENTORY),
        "source_bindings": {name: _binding(path) for name, path in sorted(source_files.items())},
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    manifest: dict[str, Any] = {
        "schema": "sft_h800_packing_gbs_repair_batch_queue_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "design": _binding(DESIGN),
        "queue": {**_binding(QUEUE), "job_count": 18, "ordered_job_ids": [row["job_id"] for row in jobs]},
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(QUEUE_MANIFEST, manifest)
    return {"design": _binding(DESIGN), "queue": _binding(QUEUE), "jobs": 18, "gpu_job_equivalents": 30}


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
