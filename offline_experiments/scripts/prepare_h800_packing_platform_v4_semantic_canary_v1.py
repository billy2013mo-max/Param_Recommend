#!/usr/bin/env python3
"""Materialize the four-job Packing platform-v4 semantic canary.

The execution queue derives Packing GA only from cached aggregate DataProfile
features.  A full-profile packer is run separately to create calibration labels;
those labels are never consumed by the queue's recommendation geometry.
This script never launches GPU training.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    MATRIX_DIR,
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from static_packing_predictor import build_decision, load_policy


SCHEMA = "sft_h800_packing_platform_v4_semantic_canary_design/v1"
JOB_SCHEMA = "sft_h800_packing_platform_v4_semantic_canary_job/v1"
STATIC_SCHEMA = "sft_h800_packing_platform_v4_static_features/v1"
CAMPAIGN_ID = "h800_packing_platform_v4_semantic_canary_20260804_v1"
PHASE_ID = "h800_packing_platform_v4_semantic_canary_v1"
GPU_POOL = (0, 1, 4, 5, 6, 7)
MODEL_ID = "qwen3_8b"
MODEL_PATH = Path("/wanqing-models/Qwen3-8B")
TEMPLATE = "qwen3_nothink"
TARGET_GBS = 64
EPSILON_GBS_SHADOW = 0.10
K_MIN_STEPS_SHADOW = 20
EPOCHS_FOR_GATE = 1.0
WARMUP_STEPS = 1
MEASURE_STEPS = 3

BASE = ARTIFACT_DIR / "real_business_packing_cutoff_mbs_v1"
OLD_STATIC = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_static_features_v1.json"
POLICY = ARTIFACT_DIR / "static_packing_policy_v1.json"
INVENTORY = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_model_inventory_v1.json"
QUEUE = MATRIX_DIR / "h800_packing_platform_v4_semantic_canary_v1.jsonl"
STATIC = ARTIFACT_DIR / "h800_packing_platform_v4_semantic_canary_static_features_v1.json"
DESIGN = ARTIFACT_DIR / "h800_packing_platform_v4_semantic_canary_design_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_platform_v4_semantic_canary_queue_manifest_v1.json"


@dataclass(frozen=True)
class Family:
    key: str
    display_name: str
    dataset_id: str
    category: str
    cutoff_len: int
    gpu_count: int
    zero: str
    cached_curve_cutoff: int
    utilization_lower_margin: float
    utilization_upper_margin: float
    utilization_source: str


FAMILIES = (
    Family(
        key="w1_high_samples_per_pack",
        display_name="W1 极短集中/高 samples-per-pack",
        dataset_id="real_177870_short_qwen3_v1",
        category="very_short_concentrated",
        cutoff_len=7_168,
        gpu_count=1,
        zero="none",
        cached_curve_cutoff=4_096,
        utilization_lower_margin=0.02,
        utilization_upper_margin=0.01,
        utilization_source="cached_curve_last_point_extrapolation",
    ),
    Family(
        key="w4_broad_long_tail",
        display_name="W4 宽长尾/长上下文",
        dataset_id="real_4500_content_longtail_qwen3_v1",
        category="content_broad_long_tail",
        cutoff_len=32_768,
        gpu_count=2,
        zero="zero2",
        cached_curve_cutoff=32_768,
        utilization_lower_margin=0.01,
        utilization_upper_margin=0.01,
        utilization_source="cached_exact_cutoff_curve",
    ),
)


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _summary_path(family: Family) -> Path:
    return BASE / "profile_summaries" / f"{family.dataset_id}.json"


def _cached_feature_rows() -> dict[tuple[str, int], dict[str, Any]]:
    dataset_key = {
        "real_177870_short_qwen3_v1": "d177870",
        "real_4500_content_longtail_qwen3_v1": "d4500content",
    }
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_json(OLD_STATIC)["rows"]:
        rows[(str(row["dataset_key"]), int(row["cutoff_len"]))] = row
    return {
        (family.key, family.cached_curve_cutoff): rows[
            (dataset_key[family.dataset_id], family.cached_curve_cutoff)
        ]
        for family in FAMILIES
    }


def _platform_and_oracle(family: Family, cached_row: dict[str, Any]) -> dict[str, Any]:
    summary = read_json(_summary_path(family))
    lengths = summary["profile"]["length_tokens"]
    records = int(summary["slice"]["records"])
    mean_length = float(lengths["mean"])
    cached_utilization = float(
        cached_row["decision"]["features"]["pack_utilization"]
    )
    u_center = cached_utilization
    u_lower = max(0.0, u_center - family.utilization_lower_margin)
    u_upper = min(0.995, u_center + family.utilization_upper_margin)
    n_lower = family.cutoff_len * u_lower / mean_length
    n_center = family.cutoff_len * u_center / mean_length
    n_upper = family.cutoff_len * u_upper / mean_length
    ga = max(1, round(TARGET_GBS / (n_center * family.gpu_count)))
    expected_gbs = n_center * family.gpu_count * ga
    expected_error = abs(expected_gbs - TARGET_GBS) / TARGET_GBS
    packs_center = records / n_center
    opt_steps_lower = (
        records
        / n_upper
        * EPOCHS_FOR_GATE
        / (family.gpu_count * ga)
    )
    gbs_controllable = (
        n_upper * family.gpu_count
        <= TARGET_GBS * (1.0 + EPSILON_GBS_SHADOW)
    )
    cutoff_covers_max = family.cutoff_len >= int(lengths["maximum"])
    if expected_error > 0.05:
        raise ValueError(f"{family.key}: center expected GBS error exceeds 5%")
    if not gbs_controllable or not cutoff_covers_max:
        raise ValueError(f"{family.key}: platform hard gate failed")
    if opt_steps_lower < K_MIN_STEPS_SHADOW:
        raise ValueError(f"{family.key}: optimizer-step shadow gate failed")

    policy = load_policy(POLICY.resolve())
    oracle = build_decision(
        {
            "request_id": f"{CAMPAIGN_ID}-{family.key}-calibration-oracle",
            "gpu_family": "H800",
            "modality": "text",
            "stage": "sft",
            "dtype": "bf16",
            "model_id": MODEL_ID,
            "train_type": "lora",
            "profile_path": summary["profile"]["path"],
            "cutoff_len": family.cutoff_len,
            "no_packing_mbs": 1,
            "gpu_count": family.gpu_count,
            "data_parallel": family.gpu_count,
            "target_gbs": TARGET_GBS,
            "preprocessing_num_workers": 8,
            "packing_algorithm_id": policy["packing_algorithm"]["id"],
        },
        policy=policy,
        policy_path=POLICY.resolve(),
        request_base=ROOT.parent,
    )
    oracle_features = oracle["features"]
    oracle_n_pack = float(oracle_features["mean_samples_per_pack"])
    return {
        "family_id": family.key,
        "display_name": family.display_name,
        "dataset_id": family.dataset_id,
        "dataset_profile_summary": _binding(_summary_path(family)),
        "candidate": {
            "cutoff_len": family.cutoff_len,
            "gpu_count": family.gpu_count,
            "target_gbs": TARGET_GBS,
            "epochs_for_gate": EPOCHS_FOR_GATE,
        },
        "platform_estimate": {
            "input_contract": "cached_aggregate_DataProfile_only",
            "raw_dataset_read": False,
            "full_profile_read": False,
            "mean_length_tokens": mean_length,
            "records": records,
            "maximum_length_tokens": int(lengths["maximum"]),
            "utilization_source": family.utilization_source,
            "cached_curve_source_cutoff": family.cached_curve_cutoff,
            "cached_curve_source_sha256": sha256_file(OLD_STATIC),
            "utilization": {
                "lower": u_lower,
                "center": u_center,
                "upper": u_upper,
            },
            "n_pack": {
                "lower": n_lower,
                "center": n_center,
                "upper": n_upper,
            },
            "gradient_accumulation_steps": ga,
            "expected_sample_gbs": expected_gbs,
            "expected_sample_gbs_relative_error": expected_error,
            "packs_center": packs_center,
            "opt_steps_lower": opt_steps_lower,
            "gates": {
                "epsilon_gbs_shadow": EPSILON_GBS_SHADOW,
                "k_min_steps_shadow": K_MIN_STEPS_SHADOW,
                "cutoff_covers_profile_max": cutoff_covers_max,
                "gbs_controllable": gbs_controllable,
                "optimizer_steps_sufficient": opt_steps_lower >= K_MIN_STEPS_SHADOW,
            },
        },
        "calibration_oracle": {
            "usage": "offline_label_only_not_consumed_by_queue_geometry",
            "decision": oracle,
            "actual_pack_utilization": float(oracle_features["pack_utilization"]),
            "actual_mean_samples_per_pack": oracle_n_pack,
            "n_pack_center_relative_error": abs(n_center - oracle_n_pack) / oracle_n_pack,
        },
    }


def prepare() -> dict[str, Any]:
    cached = _cached_feature_rows()
    feature_rows = [
        _platform_and_oracle(
            family,
            cached[(family.key, family.cached_curve_cutoff)],
        )
        for family in FAMILIES
    ]
    static: dict[str, Any] = {
        "schema": STATIC_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "generated_before_gpu": True,
        "recommendation_path_reads_raw_data": False,
        "recommendation_path_reads_full_profile": False,
        "oracle_is_calibration_only": True,
        "policy": _binding(POLICY),
        "shadow_contract": {
            "target_gbs": TARGET_GBS,
            "epsilon_gbs": EPSILON_GBS_SHADOW,
            "k_min_steps": K_MIN_STEPS_SHADOW,
            "epochs": EPOCHS_FOR_GATE,
            "automatic_publication": False,
        },
        "rows": feature_rows,
    }
    static["report_sha256"] = sha256_json(static)
    write_json(STATIC, static)

    model_parameters = int(read_json(INVENTORY)["models"][0]["actual_parameters"])
    feature_by_family = {str(row["family_id"]): row for row in feature_rows}
    jobs: list[dict[str, Any]] = []
    for family in FAMILIES:
        summary = read_json(_summary_path(family))
        estimate = feature_by_family[family.key]["platform_estimate"]
        for packing in (False, True):
            if packing:
                ga = int(estimate["gradient_accumulation_steps"])
                expected_gbs = float(estimate["expected_sample_gbs"])
                expected_error = float(estimate["expected_sample_gbs_relative_error"])
                arm_id = "P-C-1-gP"
            else:
                denominator = family.gpu_count
                if TARGET_GBS % denominator:
                    raise ValueError(f"{family.key}: unpacked GBS not divisible by DP")
                ga = TARGET_GBS // denominator
                expected_gbs = float(TARGET_GBS)
                expected_error = 0.0
                arm_id = "N-C-1-gN"
            row: dict[str, Any] = {
                "schema": JOB_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "phase_id": PHASE_ID,
                "candidate_role": "packing_semantic_and_instrumentation_canary",
                "family_id": family.key,
                "display_name": family.display_name,
                "scenario_id": f"{family.key}-qwen3_8b-lora-dp{family.gpu_count}",
                "arm_id": arm_id,
                "repeat": 0,
                "model_id": MODEL_ID,
                "model_family": "qwen3",
                "model_path": str(MODEL_PATH),
                "tokenizer_path": str(MODEL_PATH),
                "template": TEMPLATE,
                "model_parameters": model_parameters,
                "train_type": "lora",
                "dataset_id": family.dataset_id,
                "profile_id": family.dataset_id,
                "dataset_category": family.category,
                "source_dataset_records": int(summary["source"]["records"]),
                "frozen_slice_records": int(summary["slice"]["records"]),
                "data_path": summary["slice"]["data_path"],
                "data_sha256": summary["slice"]["data_sha256"],
                "dataset_profile_path": summary["profile"]["path"],
                "dataset_profile_sha256": summary["profile"]["sha256"],
                "cutoff_label": "semantic_event",
                "base_cutoff_len": family.cutoff_len,
                "cutoff_scale": 1,
                "cutoff_len": family.cutoff_len,
                "target_gbs": TARGET_GBS,
                "gpu_count": family.gpu_count,
                "zero": family.zero,
                "zero_stage": 0 if family.zero == "none" else 2,
                "gc": True,
                "gradient_checkpointing": True,
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
                "fidelity": "semantic_canary_1plus3",
                "parallel_class": "gpu_partitionable",
                "requires_external_node_idle": False,
                "execution_sequence_index": len(jobs),
                "calibration_partition": {
                    "role": "calibration",
                    "split_unit_id": family.key,
                    "policy": "platform_v4_semantic_canary_only",
                },
                "declared_model_manifest_path": str(INVENTORY.resolve()),
                "declared_model_manifest_sha256": sha256_file(INVENTORY),
                "static_features_path": str(STATIC.resolve()),
                "static_features_sha256": sha256_file(STATIC),
                "platform_estimate": estimate,
                "matched_semantic_pair": True,
                "publication_allowed": False,
            }
            row["job_id"] = stable_id("h800packv4canary", row)
            jobs.append(row)
    if len(jobs) != 4 or len({str(row["job_id"]) for row in jobs}) != 4:
        raise ValueError("semantic canary must contain four unique jobs")
    write_jsonl(QUEUE, jobs)
    jobs_dir = ARTIFACT_DIR / "h800_packing_platform_v4_semantic_canary_jobs_v1"
    for job in jobs:
        write_json(jobs_dir / f"{job['job_id']}.json", job)

    source_files = {
        "plan": ROOT.parent / "Neat_Packing联合搜索_数据实验与建模完整计划_2026-08-04.md",
        "experiment_config": ROOT / "config" / "experiment.json",
        "hardware_config": ROOT / "config" / "hardware.json",
        "dataset_registry": ROOT / "data" / "dataset_info.json",
        "packing_policy": POLICY,
        "prior_cached_curve": OLD_STATIC,
        "model_inventory": INVENTORY,
        "preparer": Path(__file__).resolve(),
        "freezer": ROOT / "scripts" / "freeze_h800_packing_platform_v4_semantic_canary_v1.py",
        "evaluator": ROOT / "scripts" / "evaluate_h800_packing_platform_v4_semantic_canary_v1.py",
        "run_job": ROOT / "scripts" / "run_job.py",
        "scheduler": ROOT / "scripts" / "scheduler.py",
        "train_entry": ROOT / "scripts" / "train_entry.py",
        "metrics_callback": ROOT / "scripts" / "metrics_callback.py",
    }
    for family in FAMILIES:
        summary = read_json(_summary_path(family))
        source_files[f"summary_{family.key}"] = _summary_path(family)
        source_files[f"slice_{family.key}"] = Path(summary["slice"]["data_path"])
        source_files[f"profile_{family.key}"] = Path(summary["profile"]["path"])

    design: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_gpu_waiting_for_exact_approval",
        "gpu_training_started": False,
        "fit_allowed_after_complete_results": True,
        "automatic_expansion_allowed": False,
        "publication_allowed": False,
        "objective": (
            "Close Packing semantics/instrumentation on high-samples-per-pack and "
            "broad-long-tail profiles at DP=1/2 before any larger platform-v4 campaign."
        ),
        "required_gpu_pool": {
            "gpu_ids": list(GPU_POOL),
            "max_gpu_count_per_job": 2,
            "max_parallel_gpu_slots": len(GPU_POOL),
            "maximum_parallel_gpu_slots": 6,
            "live_launch_policy": "start on currently idle authorized GPUs and join remaining pairs only after they become idle",
            "preemption_allowed": False,
            "excluded_busy_gpu_ids": [2, 3],
        },
        "queue": {
            **_binding(QUEUE),
            "job_count": len(jobs),
            "ordered_job_ids": [str(row["job_id"]) for row in jobs],
        },
        "matrix": {
            "families": [
                {
                    "family_id": family.key,
                    "dataset_id": family.dataset_id,
                    "cutoff_len": family.cutoff_len,
                    "gpu_count": family.gpu_count,
                    "zero": family.zero,
                    "arms": ["N-C-1-gN", "P-C-1-gP"],
                }
                for family in FAMILIES
            ],
            "jobs": len(jobs),
            "total_gpu_slots": sum(int(row["gpu_count"]) for row in jobs),
        },
        "measurement_contract": {
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "token_source": "consumed_token_ledger/v1",
            "packed_semantics_required_on_every_rank": True,
            "authoritative_ledger_required_on_every_rank": True,
            "platform_estimate_vs_oracle_report_required": True,
            "stop_all_packed_successors_on_failure": True,
            "publication_allowed": False,
        },
        "static_features": _binding(STATIC),
        "model_inventory": _binding(INVENTORY),
        "source_bindings": {name: _binding(path) for name, path in sorted(source_files.items())},
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    manifest: dict[str, Any] = {
        "schema": "sft_h800_packing_platform_v4_semantic_canary_queue_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "design": _binding(DESIGN),
        "queue": {
            **_binding(QUEUE),
            "job_count": len(jobs),
            "ordered_job_ids": [str(row["job_id"]) for row in jobs],
        },
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(QUEUE_MANIFEST, manifest)
    return {
        "design": _binding(DESIGN),
        "queue": _binding(QUEUE),
        "jobs": len(jobs),
        "gpu_slots": sum(int(row["gpu_count"]) for row in jobs),
        "platform_geometry": {
            row["family_id"]: row["platform_estimate"] for row in feature_rows
        },
        "oracle_errors": {
            row["family_id"]: row["calibration_oracle"]["n_pack_center_relative_error"]
            for row in feature_rows
        },
    }


def main() -> None:
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
