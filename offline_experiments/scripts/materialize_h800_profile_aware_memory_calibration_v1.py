#!/usr/bin/env python3
"""Materialize the approved 20-slot profile-aware H800 calibration design."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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


DESIGN_SCHEMA = "sft_h800_profile_aware_memory_calibration_design/v1"
JOB_SCHEMA = "sft_h800_profile_aware_memory_calibration_job/v1"
PHASE_ID = "h800_profile_aware_memory_calibration_v1"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_profile_aware_memory_calibration_design_v1.json"
DEFAULT_QUEUE = ROOT / "matrix" / "h800_profile_aware_memory_calibration_jobs_v1.jsonl"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_profile_aware_memory_calibration_queue_manifest_v1.json"


def _validate_design(design: Mapping[str, Any]) -> None:
    if design.get("schema") != DESIGN_SCHEMA:
        raise ValueError("profile-aware calibration design schema mismatch")
    unsigned = dict(design)
    expected_report_sha256 = unsigned.pop("report_sha256", None)
    if expected_report_sha256 != sha256_json(unsigned):
        raise ValueError("profile-aware calibration design internal hash mismatch")
    if (
        design.get("gpu_training_started") is not False
        or design.get("queues_mutated") is not False
        or design.get("execution_authorized") is not False
    ):
        raise ValueError("source design is not an unexecuted immutable design")
    if design.get("design", {}).get("job_count") != 20:
        raise ValueError("source design must declare exactly 20 jobs")


def _models() -> dict[str, Mapping[str, Any]]:
    inventory = read_json(ARTIFACT_DIR / "model_inventory.json")
    return {str(row["id"]): row for row in inventory["models"]}


def build_jobs(design: Mapping[str, Any]) -> list[dict[str, Any]]:
    _validate_design(design)
    models = _models()
    jobs: list[dict[str, Any]] = []
    for slot in design["candidate_slots"]:
        if slot.get("campaign_id") != design["campaign_id"]:
            raise ValueError(f"slot campaign mismatch: {slot.get('slot_id')}")
        data_path = Path(str(slot["data_path"]))
        profile_path = Path(str(slot["dataset_profile_path"]))
        for label, path, expected in (
            ("data", data_path, slot["data_sha256"]),
            ("profile", profile_path, slot["dataset_profile_sha256"]),
        ):
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"{label} binding changed for {slot['slot_id']}: {path}")
        partition = slot.get("calibration_partition") or {}
        if partition != {
            "role": "calibration",
            "split_unit_id": slot["dataset_id"],
            "policy": "profile_padding_stratified_scenario_disjoint_v1",
        }:
            raise ValueError(f"invalid calibration partition: {slot['slot_id']}")
        gpu_count = int(slot["gpu_count"])
        mbs = int(slot["mbs"])
        expected_ga = int(slot["target_gbs"]) // (gpu_count * mbs)
        if (
            gpu_count not in {1, 2}
            or int(slot["target_gbs"]) % (gpu_count * mbs)
            or int(slot["gradient_accumulation_steps"]) != expected_ga
        ):
            raise ValueError(f"invalid batch decomposition: {slot['slot_id']}")
        zero_stage = int(slot["zero_stage"])
        if (gpu_count, zero_stage) not in {(1, 0), (2, 2)}:
            raise ValueError(f"invalid GPU/ZeRO selector: {slot['slot_id']}")
        model = models[str(slot["model_id"])]
        identity = {
            "campaign_id": design["campaign_id"],
            "slot_id": slot["slot_id"],
        }
        jobs.append(
            {
                "schema": JOB_SCHEMA,
                "job_id": stable_id("h800memprofjob", identity),
                "campaign_id": design["campaign_id"],
                "phase_id": PHASE_ID,
                "scenario_id": slot["scenario_id"],
                "candidate_slot_id": slot["slot_id"],
                "calibration_purpose": slot["purpose"],
                "calibration_partition": dict(partition),
                "hardware_id": "local_h800_140g",
                "gpu_type": "NVIDIA H800 140GB HBM3",
                "required_runtime_gpu_name": "NVIDIA H800",
                "model_id": slot["model_id"],
                "model_path": model["path"],
                "tokenizer_path": model["tokenizer_path"],
                "model_family": model["family"],
                "model_parameters": model["actual_parameters"],
                "template": model["template"],
                "train_type": slot["train_type"],
                "dataset_id": slot["dataset_id"],
                "dataset_category": slot["dataset_category"],
                "dataset_profile_path": str(profile_path.resolve()),
                "dataset_profile_sha256": slot["dataset_profile_sha256"],
                "data_path": str(data_path.resolve()),
                "data_sha256": slot["data_sha256"],
                "cutoff_len": int(slot["cutoff_len"]),
                "target_gbs": int(slot["target_gbs"]),
                "gpu_count": gpu_count,
                "zero_stage": zero_stage,
                "zero": slot["zero"],
                "gc": bool(slot["gc"]),
                "gradient_checkpointing": bool(slot["gradient_checkpointing"]),
                "mbs": mbs,
                "gradient_accumulation_steps": expected_ga,
                "packing": bool(slot["packing"]),
                "offload": bool(slot["offload"]),
                "kind": "throughput",
                "fidelity": slot["fidelity"],
                "warmup_steps": int(slot["warmup_steps"]),
                "measure_steps": int(slot["measure_steps"]),
                "repeat": int(slot["repeat"]),
                "parallel_class": "gpu_partitionable",
                "requires_external_node_idle": False,
                "expected_padding_pressure": dict(slot["expected_padding_pressure"]),
            }
        )
    if len(jobs) != 20 or len({job["job_id"] for job in jobs}) != 20:
        raise ValueError(f"expected 20 unique jobs, got {len(jobs)}")
    if sum(int(job["gpu_count"]) for job in jobs) != 36:
        raise ValueError("expected exactly 36 GPU-job equivalents")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.queue.exists() or args.output.exists():
        raise SystemExit("refusing to overwrite an existing queue or queue manifest")
    design = read_json(args.design)
    jobs = build_jobs(design)
    write_jsonl(args.queue, jobs)
    manifest = {
        "schema": "sft_h800_profile_aware_memory_calibration_queue_manifest/v1",
        "campaign_id": design["campaign_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_design": {
            "path": str(args.design.resolve()),
            "sha256": sha256_file(args.design),
        },
        "queue": {
            "path": str(args.queue.resolve()),
            "sha256": sha256_file(args.queue),
            "canonical_rows_sha256": sha256_json(jobs),
        },
        "job_count": len(jobs),
        "gpu_job_equivalents": sum(int(job["gpu_count"]) for job in jobs),
        "gpu_training_started": False,
        "execution_authorized": False,
        "next_step": "capture provenance and promote an exact GPU-4,5 approval before execution",
    }
    write_json(args.output, manifest)
    print(
        f"wrote {args.queue}; jobs={len(jobs)}; "
        f"gpu_job_equivalents={manifest['gpu_job_equivalents']}; execution_authorized=False"
    )


if __name__ == "__main__":
    main()
