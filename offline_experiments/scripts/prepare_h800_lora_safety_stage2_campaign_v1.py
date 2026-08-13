#!/usr/bin/env python3
"""Materialize the 20-source, 60-job H800 critical-LoRA safety campaign.

This step writes an immutable design, JSONL queue, and queue manifest.  It does
not authorize or launch GPU training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

from common import (
    ARTIFACT_DIR,
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)


CAMPAIGN_ID = "h800_lora_safety_stage2_20260805_v1"
PHASE_ID = "h800_lora_safety_stage2_v1"
DESIGN_SCHEMA = "sft_h800_lora_safety_stage2_experiment_design/v1"
JOB_SCHEMA = "sft_h800_lora_safety_stage2_job/v1"
TARGET_GBS = 64
DEFAULT_BUNDLE = ARTIFACT_DIR / "h800_lora_safety_stage2_bundle_v1.json"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_lora_safety_stage2_experiment_design_v1.json"
DEFAULT_QUEUE = ROOT / "matrix" / "h800_lora_safety_stage2_jobs_v1.jsonl"
DEFAULT_MANIFEST = ARTIFACT_DIR / "h800_lora_safety_stage2_queue_manifest_v1.json"


def _binding(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _models() -> dict[str, Mapping[str, Any]]:
    inventory = read_json(ARTIFACT_DIR / "model_inventory.json")
    return {str(row["id"]): row for row in inventory["models"]}


def _job(
    scenario: Mapping[str, Any], model: Mapping[str, Any], *, mbs: int
) -> dict[str, Any]:
    gpu_count = 2
    if TARGET_GBS % (gpu_count * mbs):
        raise ValueError("target GBS must be divisible by GPU count times MBS")
    identity = {
        "campaign_id": CAMPAIGN_ID,
        "scenario_id": scenario["scenario_id"],
        "mbs": mbs,
    }
    return {
        "schema": JOB_SCHEMA,
        "job_id": stable_id("h800loras2mem", identity),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "scenario_id": scenario["scenario_id"],
        "split_unit_id": scenario["split_unit_id"],
        "experiment_group": "S2",
        "calibration_purpose": f"stage2_critical_lora_mbs{mbs}",
        "calibration_partition": {
            "role": "calibration",
            "split_unit_id": scenario["split_unit_id"],
            "policy": "remote_source_dataset_id_disjoint_stage2_v1",
        },
        "hardware_id": "local_h800_140g",
        "gpu_type": "NVIDIA H800 140GB HBM3",
        "required_runtime_gpu_name": "NVIDIA H800",
        "model_id": scenario["model_id"],
        "model_path": model["path"],
        "tokenizer_path": model["tokenizer_path"],
        "model_family": model["family"],
        "model_parameters": model["actual_parameters"],
        "template": model["template"],
        "train_type": "lora",
        "dataset_id": scenario["dataset_id"],
        "dataset_category": scenario["dataset_category"],
        "dataset_profile_path": scenario["profile_path"],
        "dataset_profile_sha256": scenario["profile_sha256"],
        "data_path": scenario["data_path"],
        "data_sha256": scenario["data_sha256"],
        "cutoff_len": int(scenario["cutoff_len"]),
        "raw_profile_max": int(scenario["raw_profile_max"]),
        "aligned_effective_sequence": int(scenario["aligned_effective_sequence"]),
        "raw_profile_max_over_cutoff": float(scenario["raw_profile_max_over_cutoff"]),
        "ratio_bin": scenario["ratio_bin"],
        "target_gbs": TARGET_GBS,
        "gpu_count": gpu_count,
        "zero_stage": 2,
        "zero": "zero2",
        "gc": False,
        "gradient_checkpointing": False,
        "mbs": mbs,
        "gradient_accumulation_steps": TARGET_GBS // (gpu_count * mbs),
        "packing": False,
        "offload": False,
        "kind": "throughput",
        "fidelity": "formal_3plus10",
        "warmup_steps": 3,
        "measure_steps": 10,
        "repeat": 0,
        "max_samples": int(scenario["profile_statistics"]["rows"]),
        "parallel_class": "gpu_partitionable",
        "requires_external_node_idle": False,
    }


def build_jobs(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    if bundle.get("schema") != "sft_h800_lora_safety_stage2_data_bundle/v1":
        raise ValueError("stage-two source bundle schema mismatch")
    unsigned = dict(bundle)
    expected_hash = unsigned.pop("report_sha256", None)
    if expected_hash != sha256_json(unsigned):
        raise ValueError("stage-two source bundle internal hash mismatch")
    if bundle.get("gpu_training_started") is not False:
        raise ValueError("source bundle must remain pre-GPU")
    if bundle.get("counts", {}).get("scenarios") != 20:
        raise ValueError("source bundle must bind exactly 20 scenarios")

    models = _models()
    scenarios = list(bundle["scenarios"])
    jobs = [
        _job(scenario, models[str(scenario["model_id"])], mbs=mbs)
        for mbs in (1, 2, 4)
        for scenario in scenarios
    ]
    if len(jobs) != 60 or len({job["job_id"] for job in jobs}) != 60:
        raise ValueError(f"expected 60 unique jobs, got {len(jobs)}")
    if sum(int(job["gpu_count"]) for job in jobs) != 120:
        raise ValueError("expected 120 GPU-job equivalents")
    if Counter(job["mbs"] for job in jobs) != {1: 20, 2: 20, 4: 20}:
        raise ValueError("MBS design is not exactly 20 jobs at each of 1, 2, and 4")
    if len({job["split_unit_id"] for job in jobs}) != 20:
        raise ValueError("queue does not contain exactly 20 independent sources")
    return jobs


def build_design(
    bundle_path: Path, bundle: Mapping[str, Any], jobs: list[dict[str, Any]]
) -> dict[str, Any]:
    outcome = {
        "schema": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": bundle["generated_at_utc"],
        "timestamp_semantics": "inherits the frozen data-bundle completion time",
        "status": "design_and_queue_materialized_waiting_for_exact_approval",
        "gpu_training_started": False,
        "queues_mutated": True,
        "execution_authorized": False,
        "publication_allowed": False,
        "objective": (
            "Recalibrate the H800 critical-LoRA safety upper with 20 additional "
            "source-disjoint profiles, using the physically achievable profile maximum."
        ),
        "source_bundle": {
            **_binding(bundle_path),
            "report_sha256": bundle["report_sha256"],
        },
        "design": {
            "new_independent_split_units": 20,
            "job_count": len(jobs),
            "gpu_job_equivalents": sum(int(job["gpu_count"]) for job in jobs),
            "job_groups": dict(Counter(job["experiment_group"] for job in jobs)),
            "ratio_bins": bundle["counts"]["ratio_bins"],
            "models": bundle["counts"]["models"],
            "mbs_points": [1, 2, 4],
            "target_gbs": TARGET_GBS,
            "fidelity": "warmup 3 plus measure 10 optimizer steps",
            "fixed_mechanism": {
                "train_type": "lora",
                "gpu_count": 2,
                "zero_stage": 2,
                "gradient_checkpointing": False,
                "packing": False,
                "offload": False,
            },
        },
        "interpretation": {
            "success_and_oom_are_both_valid_safety_evidence": True,
            "infra_failure_is_not_oom": True,
            "source_disjoint_outer_validation_required": True,
            "historical_holdout_remains_diagnostic_only": True,
            "publication_requires_a_separate_gate": True,
        },
        "scenarios": bundle["scenarios"],
        "ordered_job_ids": [job["job_id"] for job in jobs],
        "ordered_job_payload_sha256": sha256_json(jobs),
    }
    outcome["report_sha256"] = sha256_json(outcome)
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    existing = [path for path in (args.design, args.queue, args.manifest) if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing campaign artifacts: {existing}")
    bundle = read_json(args.bundle)
    jobs = build_jobs(bundle)
    design = build_design(args.bundle, bundle, jobs)
    write_json(args.design, design)
    write_jsonl(args.queue, jobs)
    manifest = {
        "schema": "sft_h800_lora_safety_stage2_queue_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": bundle["generated_at_utc"],
        "gpu_training_started": False,
        "execution_authorized": False,
        "source_bundle": _binding(args.bundle),
        "design": _binding(args.design),
        "queue": {
            **_binding(args.queue),
            "job_count": len(jobs),
            "gpu_job_equivalents": sum(int(job["gpu_count"]) for job in jobs),
            "ordered_job_ids": [job["job_id"] for job in jobs],
            "ordered_job_payload_sha256": sha256_json(jobs),
        },
        "next_step": (
            "validate installed preprocessing, capture fresh H800 occupancy, and "
            "promote the exact non-preemptive approval before execution"
        ),
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(args.manifest, manifest)
    print(
        json.dumps(
            {
                "design": _binding(args.design),
                "queue": _binding(args.queue),
                "manifest": _binding(args.manifest),
                "jobs": len(jobs),
                "gpu_job_equivalents": sum(int(job["gpu_count"]) for job in jobs),
                "mbs_counts": dict(Counter(job["mbs"] for job in jobs)),
                "execution_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
