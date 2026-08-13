#!/usr/bin/env python3
"""Materialize the frozen H800 v2 design into an execution-ready queue."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from common import ARTIFACT_DIR, ROOT, read_json, sha256_file, sha256_json, stable_id, write_json, write_jsonl


DESIGN_SCHEMA = "sft_h800_prospective_holdout_design/v2"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_fresh_holdout_design_v2.json"
DEFAULT_REQUIREMENTS = ARTIFACT_DIR / "h800_fresh_profile_requirements_v2.json"
DEFAULT_QUEUE = ROOT / "matrix" / "h800_fresh_holdout_jobs_v2.jsonl"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_prospective_queue_manifest_v2.json"


def _models() -> dict[str, Mapping[str, Any]]:
    return {str(row["id"]): row for row in read_json(ARTIFACT_DIR / "model_inventory.json")["models"]}


def build_jobs(design: Mapping[str, Any], requirements: Mapping[str, Any]) -> list[dict[str, Any]]:
    if design.get("schema") != DESIGN_SCHEMA or design.get("materialization_allowed") is not True:
        raise ValueError("v2 design is absent, mismatched, or not materializable")
    if requirements.get("ready_for_materialization") is not True:
        raise ValueError("fresh profile requirements are not ready")
    scenarios = {str(row["scenario_id"]): row for row in design["scenarios"]}
    requirement_rows = {str(row["scenario_id"]): row for row in requirements["scenarios"]}
    models = _models()
    jobs: list[dict[str, Any]] = []
    for slot in design["candidate_slots"]:
        if slot.get("memory_admitted") is not True:
            raise ValueError(f"selected slot is not admitted: {slot.get('candidate_slot_id')}")
        scenario = scenarios[str(slot["scenario_id"])]
        requirement = requirement_rows[str(slot["scenario_id"])]
        bindings = requirement["required_bindings"]
        for field in ("profile", "data"):
            path = Path(bindings[f"{field}_path"])
            if not path.is_file() or sha256_file(path) != bindings[f"{field}_sha256"]:
                raise ValueError(f"{slot['scenario_id']} {field} binding changed")
        model = models[str(slot["model_id"])]
        zero_stage = int(slot["zero_stage"])
        identity = {
            "campaign_id": design["campaign_id"],
            "candidate_slot_id": slot["candidate_slot_id"],
        }
        jobs.append(
            {
                "schema": "sft_h800_fresh_holdout_job/v2",
                "job_id": stable_id("h800freshv2", identity),
                "campaign_id": design["campaign_id"],
                "scenario_id": slot["scenario_id"],
                "candidate_slot_id": slot["candidate_slot_id"],
                "predictor_request_id": slot["request_id"],
                "phase_id": "h800_fresh_business_holdout_v2",
                "calibration_partition": {
                    "role": "holdout",
                    "split_unit_id": str(slot["scenario_id"]),
                    "policy": "prospective_fresh_scenario_disjoint_v2",
                },
                "hardware_id": "local_h800_140g",
                "gpu_type": "NVIDIA H800 140GB HBM3",
                "required_runtime_gpu_name": "NVIDIA H800",
                "model_id": slot["model_id"],
                "model_path": model["path"],
                "tokenizer_path": model["tokenizer_path"],
                "model_family": model["family"],
                "model_parameters": model["actual_parameters"],
                "template": model["template"],
                "train_type": slot["training_mode"],
                "dataset_id": slot["dataset_id"],
                "dataset_profile_path": bindings["profile_path"],
                "dataset_profile_sha256": bindings["profile_sha256"],
                "data_path": bindings["data_path"],
                "data_sha256": bindings["data_sha256"],
                "dataset_category": scenario["dataset_category"],
                "cutoff_len": int(slot["cutoff_len"]),
                "target_gbs": int(slot["target_gbs"]),
                "gpu_count": int(slot["gpu_count"]),
                "zero_stage": zero_stage,
                "zero": "none" if zero_stage == 0 else f"zero{zero_stage}",
                "gc": bool(slot["gradient_checkpointing"]),
                "gradient_checkpointing": bool(slot["gradient_checkpointing"]),
                "mbs": int(slot["physical_mbs"]),
                "gradient_accumulation_steps": int(slot["gradient_accumulation_steps"]),
                "packing": False,
                "offload": False,
                "kind": "throughput",
                "fidelity": "formal_3plus10",
                "warmup_steps": 3,
                "measure_steps": 10,
                "repeat": 0,
                "parallel_class": "exclusive_pool" if int(slot["gpu_count"]) == 4 else "gpu_partitionable",
                "requires_external_node_idle": False,
                "frozen_prediction": {
                    "memory_admission_source": slot["memory_admission_source"],
                    "memory_upper_reserved_bytes": slot["memory_upper_reserved_bytes"],
                    "safe_limit_bytes": slot["safe_limit_bytes"],
                    "v4b_rank_within_gpu_count": slot["v4b_rank_within_gpu_count"],
                    "v4b_throughput_proxy": slot["v4b_throughput_proxy"],
                },
            }
        )
    if len(jobs) != 24 or len({job["job_id"] for job in jobs}) != 24:
        raise ValueError(f"expected 24 unique jobs, got {len(jobs)}")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.queue.exists():
        raise SystemExit(f"refusing to overwrite existing queue: {args.queue}")
    design = read_json(args.design)
    requirements = read_json(args.requirements)
    jobs = build_jobs(design, requirements)
    write_jsonl(args.queue, jobs)
    manifest = {
        "schema": "sft_h800_prospective_queue_manifest/v2",
        "campaign_id": design["campaign_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": {"path": str(args.design.resolve()), "sha256": sha256_file(args.design)},
        "requirements": {"path": str(args.requirements.resolve()), "sha256": sha256_file(args.requirements)},
        "queue": {"path": str(args.queue.resolve()), "sha256": sha256_file(args.queue)},
        "queue_binding_sha256": sha256_json(jobs),
        "candidate_count": len(jobs),
        "gpu_training_started": False,
        "execution_authorized": False,
        "next_step": "capture fresh provenance and promote an exact queue approval before scheduler execution",
    }
    write_json(args.output, manifest)
    print(f"wrote {args.queue}; jobs={len(jobs)}; execution_authorized=False")


if __name__ == "__main__":
    main()
