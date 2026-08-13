#!/usr/bin/env python3
"""Materialize formal Packing×GC/ZeRO interactions batch 1 (24 jobs)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, ROOT, read_json, sha256_file, sha256_json, stable_id, write_json, write_jsonl


SCHEMA = "sft_h800_packing_platform_v4_interactions_batch1_design/v1"
JOB_SCHEMA = "sft_h800_packing_platform_v4_interactions_batch1_job/v1"
CAMPAIGN_ID = "h800_packing_platform_v4_interactions_batch1_20260804_v1"
PHASE_ID = "h800_packing_platform_v4_interactions_batch1_v1"
GPU_POOL = (0, 1, 4, 5, 6, 7)
MODEL_ID = "qwen3_8b"
MODEL_PATH = Path("/wanqing-models/Qwen3-8B")
TEMPLATE = "qwen3_nothink"
TARGET_GBS = 64
WARMUP_STEPS = 2
MEASURE_STEPS = 8

BASE = ARTIFACT_DIR / "real_business_packing_cutoff_mbs_v1"
CANARY_STATIC = ARTIFACT_DIR / "h800_packing_platform_v4_semantic_canary_static_features_v1.json"
CANARY_RESULTS = ARTIFACT_DIR / "h800_packing_platform_v4_semantic_canary_results_v1.json"
INVENTORY = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_model_inventory_v1.json"
QUEUE = MATRIX_DIR / "h800_packing_platform_v4_interactions_batch1_v1.jsonl"
STATIC = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_static_features_v1.json"
DESIGN = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_design_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_queue_manifest_v1.json"


@dataclass(frozen=True)
class Setting:
    setting_id: str
    profile_family_id: str
    display_name: str
    dataset_id: str
    category: str
    cutoff_len: int
    gpu_count: int
    zero: str
    gc: bool
    interaction_axis: str
    sequence: tuple[bool, ...]


UPPUUP = (False, True, True, False, False, True)
PUUPPU = (True, False, False, True, True, False)
SETTINGS = (
    Setting("w1_gc_on", "w1_high_samples_per_pack", "W1/GC-on", "real_177870_short_qwen3_v1", "very_short_concentrated", 7168, 1, "none", True, "packing_x_gc", UPPUUP),
    Setting("w1_gc_off", "w1_high_samples_per_pack", "W1/GC-off", "real_177870_short_qwen3_v1", "very_short_concentrated", 7168, 1, "none", False, "packing_x_gc", PUUPPU),
    Setting("w4_zero2", "w4_broad_long_tail", "W4/ZeRO-2", "real_4500_content_longtail_qwen3_v1", "content_broad_long_tail", 32768, 2, "zero2", True, "packing_x_zero", UPPUUP),
    Setting("w4_zero3", "w4_broad_long_tail", "W4/ZeRO-3", "real_4500_content_longtail_qwen3_v1", "content_broad_long_tail", 32768, 2, "zero3", True, "packing_x_zero", PUUPPU),
)


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _summary_path(setting: Setting) -> Path:
    return BASE / "profile_summaries" / f"{setting.dataset_id}.json"


def prepare() -> dict[str, Any]:
    canary_results = read_json(CANARY_RESULTS)
    gates = canary_results.get("gates") or {}
    if (
        gates.get("packing_semantics_and_ledger_passed") is not True
        or gates.get("dataprofile_n_pack_center_error_le_10pct") is not True
        or gates.get("packed_successor_expansion_allowed") is not True
    ):
        raise PermissionError("phase-B canary does not permit a Packed successor")
    canary_static = read_json(CANARY_STATIC)
    estimates = {
        str(row["family_id"]): row["platform_estimate"]
        for row in canary_static["rows"]
    }
    static: dict[str, Any] = {
        "schema": "sft_h800_packing_platform_v4_interactions_batch1_static/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_before_gpu": True,
        "recommendation_path_reads_raw_data": False,
        "recommendation_path_reads_full_profile": False,
        "canary_gate": _binding(CANARY_RESULTS),
        "canary_static": _binding(CANARY_STATIC),
        "rows": [
            {
                "setting_id": setting.setting_id,
                "profile_family_id": setting.profile_family_id,
                "interaction_axis": setting.interaction_axis,
                "gc": setting.gc,
                "zero": setting.zero,
                "gpu_count": setting.gpu_count,
                "cutoff_len": setting.cutoff_len,
                "platform_estimate": estimates[setting.profile_family_id],
            }
            for setting in SETTINGS
        ],
    }
    static["report_sha256"] = sha256_json(static)
    write_json(STATIC, static)

    model_parameters = int(read_json(INVENTORY)["models"][0]["actual_parameters"])
    treatment_counters = {
        setting.setting_id: {False: 0, True: 0} for setting in SETTINGS
    }
    jobs: list[dict[str, Any]] = []
    for block_index in range(6):
        for setting in SETTINGS:
            packing = setting.sequence[block_index]
            repeat = treatment_counters[setting.setting_id][packing]
            treatment_counters[setting.setting_id][packing] += 1
            estimate = estimates[setting.profile_family_id]
            if packing:
                ga = int(estimate["gradient_accumulation_steps"])
                expected_gbs = float(estimate["expected_sample_gbs"])
                expected_error = float(estimate["expected_sample_gbs_relative_error"])
                arm_id = "P-C-1-gP"
            else:
                denominator = setting.gpu_count
                if TARGET_GBS % denominator:
                    raise ValueError(f"{setting.setting_id}: unpacked GBS cannot be exact")
                ga = TARGET_GBS // denominator
                expected_gbs = float(TARGET_GBS)
                expected_error = 0.0
                arm_id = "N-C-1-gN"
            summary = read_json(_summary_path(setting))
            row: dict[str, Any] = {
                "schema": JOB_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "phase_id": PHASE_ID,
                "candidate_role": "formal_packing_execution_interaction",
                "setting_id": setting.setting_id,
                "profile_family_id": setting.profile_family_id,
                "family_id": setting.setting_id,
                "display_name": setting.display_name,
                "scenario_id": f"{setting.setting_id}-qwen3_8b-lora",
                "interaction_axis": setting.interaction_axis,
                "counterbalance_block": block_index,
                "arm_id": arm_id,
                "repeat": repeat,
                "model_id": MODEL_ID,
                "model_family": "qwen3",
                "model_path": str(MODEL_PATH),
                "tokenizer_path": str(MODEL_PATH),
                "template": TEMPLATE,
                "model_parameters": model_parameters,
                "train_type": "lora",
                "dataset_id": setting.dataset_id,
                "profile_id": setting.dataset_id,
                "dataset_category": setting.category,
                "source_dataset_records": int(summary["source"]["records"]),
                "frozen_slice_records": int(summary["slice"]["records"]),
                "data_path": summary["slice"]["data_path"],
                "data_sha256": summary["slice"]["data_sha256"],
                "dataset_profile_path": summary["profile"]["path"],
                "dataset_profile_sha256": summary["profile"]["sha256"],
                "cutoff_label": "event",
                "base_cutoff_len": setting.cutoff_len,
                "cutoff_scale": 1,
                "cutoff_len": setting.cutoff_len,
                "target_gbs": TARGET_GBS,
                "gpu_count": setting.gpu_count,
                "zero": setting.zero,
                "zero_stage": 0 if setting.zero == "none" else int(setting.zero[-1]),
                "gc": setting.gc,
                "gradient_checkpointing": setting.gc,
                "mbs": 1,
                "gradient_accumulation_steps": ga,
                "packing": packing,
                "expected_sample_gbs": expected_gbs,
                "expected_sample_gbs_relative_error": expected_error,
                "offload": False,
                "gpu_type": "NVIDIA H800 140GB HBM3",
                "required_runtime_gpu_name": "NVIDIA H800",
                "hardware_id": "local_h800_140g",
                "kind": "throughput",
                "warmup_steps": WARMUP_STEPS,
                "measure_steps": MEASURE_STEPS,
                "max_samples": int(summary["slice"]["records"]),
                "fidelity": "formal_interaction_2plus8",
                "parallel_class": "gpu_partitionable",
                "requires_external_node_idle": False,
                "execution_sequence_index": len(jobs),
                "calibration_partition": {
                    "role": "calibration",
                    "split_unit_id": setting.setting_id,
                    "policy": "platform_v4_interactions_batch1_fit_only",
                },
                "declared_model_manifest_path": str(INVENTORY.resolve()),
                "declared_model_manifest_sha256": sha256_file(INVENTORY),
                "static_features_path": str(STATIC.resolve()),
                "static_features_sha256": sha256_file(STATIC),
                "platform_estimate": estimate,
                "matched_interaction_pair": True,
                "publication_allowed": False,
            }
            row["job_id"] = stable_id("h800packv4int1", row)
            jobs.append(row)
    if len(jobs) != 24 or len({str(row["job_id"]) for row in jobs}) != 24:
        raise ValueError("interaction batch must contain 24 unique jobs")
    for setting in SETTINGS:
        subset = [row for row in jobs if row["setting_id"] == setting.setting_id]
        counts = {value: sum(bool(row["packing"]) is value for row in subset) for value in (False, True)}
        if counts != {False: 3, True: 3}:
            raise ValueError(f"{setting.setting_id}: unbalanced treatments {counts}")
    write_jsonl(QUEUE, jobs)
    jobs_dir = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_jobs_v1"
    for job in jobs:
        write_json(jobs_dir / f"{job['job_id']}.json", job)

    source_files = {
        "plan": ROOT.parent / "Neat_Packing联合搜索_数据实验与建模完整计划_2026-08-04.md",
        "experiment_config": ROOT / "config" / "experiment.json",
        "hardware_config": ROOT / "config" / "hardware.json",
        "dataset_registry": ROOT / "data" / "dataset_info.json",
        "canary_static": CANARY_STATIC,
        "canary_results": CANARY_RESULTS,
        "model_inventory": INVENTORY,
        "preparer": Path(__file__).resolve(),
        "freezer": ROOT / "scripts" / "freeze_h800_packing_platform_v4_interactions_batch1_v1.py",
        "evaluator": ROOT / "scripts" / "evaluate_h800_packing_platform_v4_interactions_batch1_v1.py",
        "run_job": ROOT / "scripts" / "run_job.py",
        "scheduler": ROOT / "scripts" / "scheduler.py",
        "train_entry": ROOT / "scripts" / "train_entry.py",
        "metrics_callback": ROOT / "scripts" / "metrics_callback.py",
    }
    for setting in SETTINGS:
        summary = read_json(_summary_path(setting))
        source_files[f"summary_{setting.setting_id}"] = _summary_path(setting)
        source_files[f"slice_{setting.setting_id}"] = Path(summary["slice"]["data_path"])
        source_files[f"profile_{setting.setting_id}"] = Path(summary["profile"]["path"])

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
        "objective": "Estimate Packing×GC and Packing×ZeRO interactions with three counterbalanced repeats per treatment.",
        "required_gpu_pool": {
            "gpu_ids": list(GPU_POOL),
            "max_gpu_count_per_job": 2,
            "maximum_parallel_gpu_slots": 6,
            "preemption_allowed": False,
            "excluded_busy_gpu_ids": [2, 3],
            "join_busy_pool": True,
        },
        "queue": {**_binding(QUEUE), "job_count": 24, "ordered_job_ids": [str(row["job_id"]) for row in jobs]},
        "matrix": {
            "settings": [
                {
                    "setting_id": setting.setting_id,
                    "interaction_axis": setting.interaction_axis,
                    "gpu_count": setting.gpu_count,
                    "zero": setting.zero,
                    "gc": setting.gc,
                    "cutoff_len": setting.cutoff_len,
                    "sequence": ["P" if value else "U" for value in setting.sequence],
                    "treatment_repeats": 3,
                }
                for setting in SETTINGS
            ],
            "jobs": 24,
            "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in jobs),
        },
        "measurement_contract": {
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "token_source": "consumed_token_ledger/v1",
            "packed_semantics_required_on_every_rank": True,
            "authoritative_ledger_required_on_every_rank": True,
            "global_metrics_sum_across_ranks": True,
            "counterbalance": "U-P-P-U-U-P_or_mirror",
            "stop_on_semantic_or_ledger_failure_before_next_batch": True,
        },
        "static_features": _binding(STATIC),
        "model_inventory": _binding(INVENTORY),
        "source_bindings": {name: _binding(path) for name, path in sorted(source_files.items())},
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    manifest: dict[str, Any] = {
        "schema": "sft_h800_packing_platform_v4_interactions_batch1_queue_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "design": _binding(DESIGN),
        "queue": {**_binding(QUEUE), "job_count": 24, "ordered_job_ids": [str(row["job_id"]) for row in jobs]},
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(QUEUE_MANIFEST, manifest)
    return {
        "design": _binding(DESIGN),
        "queue": _binding(QUEUE),
        "jobs": len(jobs),
        "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in jobs),
        "settings": [setting.setting_id for setting in SETTINGS],
    }


def main() -> None:
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
