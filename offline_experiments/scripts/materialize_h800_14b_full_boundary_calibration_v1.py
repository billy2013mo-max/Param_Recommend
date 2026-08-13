#!/usr/bin/env python3
"""Materialize the frozen eight-job Qwen3-14B Full boundary calibration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from common import ARTIFACT_DIR, ROOT, read_json, sha256_file, sha256_json, stable_id, write_json, write_jsonl


DESIGN_SCHEMA = "sft_h800_qwen3_14b_full_boundary_calibration_design/v1"
JOB_SCHEMA = "sft_h800_qwen3_14b_full_boundary_calibration_job/v1"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_14b_full_boundary_calibration_design_v1.json"
DEFAULT_QUEUE = ROOT / "matrix" / "h800_14b_full_boundary_calibration_jobs_v1.jsonl"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_14b_full_boundary_calibration_queue_manifest_v1.json"


def main() -> None:
    if DEFAULT_QUEUE.exists() or DEFAULT_OUTPUT.exists():
        raise SystemExit("refusing to overwrite an existing queue or queue manifest")
    design = read_json(DEFAULT_DESIGN)
    unsigned = dict(design)
    expected = unsigned.pop("report_sha256", None)
    if design.get("schema") != DESIGN_SCHEMA or expected != sha256_json(unsigned):
        raise ValueError("14B boundary design schema or internal hash mismatch")
    inventory = read_json(ARTIFACT_DIR / "model_inventory.json")
    model = next(row for row in inventory["models"] if row["id"] == "qwen3_14b")
    jobs = []
    for slot in design["candidate_slots"]:
        for path_key, hash_key in (
            ("data_path", "data_sha256"),
            ("dataset_profile_path", "dataset_profile_sha256"),
        ):
            path = Path(slot[path_key])
            if not path.is_file() or sha256_file(path) != slot[hash_key]:
                raise ValueError(f"slot binding changed: {slot['slot_id']} {path_key}")
        partition = slot["calibration_partition"]
        if partition.get("role") != "calibration" or partition.get("split_unit_id") != slot["dataset_id"]:
            raise ValueError(f"invalid calibration partition: {slot['slot_id']}")
        jobs.append(
            {
                "schema": JOB_SCHEMA,
                "job_id": stable_id(
                    "h80014bfulljob",
                    {"campaign_id": design["campaign_id"], "slot_id": slot["slot_id"]},
                ),
                "campaign_id": design["campaign_id"],
                "phase_id": design["phase_id"],
                "scenario_id": slot["scenario_id"],
                "candidate_slot_id": slot["slot_id"],
                "calibration_purpose": slot["purpose"],
                "calibration_partition": partition,
                "hardware_id": "local_h800_140g",
                "gpu_type": "NVIDIA H800 140GB HBM3",
                "required_runtime_gpu_name": "NVIDIA H800",
                "model_id": "qwen3_14b",
                "model_path": model["path"],
                "tokenizer_path": model["tokenizer_path"],
                "model_family": model["family"],
                "model_parameters": model["actual_parameters"],
                "template": model["template"],
                "train_type": "full",
                "dataset_id": slot["dataset_id"],
                "dataset_profile_path": slot["dataset_profile_path"],
                "dataset_profile_sha256": slot["dataset_profile_sha256"],
                "data_path": slot["data_path"],
                "data_sha256": slot["data_sha256"],
                "cutoff_len": 2048,
                "target_gbs": 64,
                "gpu_count": 2,
                "zero_stage": 3,
                "zero": "zero3",
                "gc": True,
                "gradient_checkpointing": True,
                "mbs": int(slot["mbs"]),
                "gradient_accumulation_steps": int(slot["gradient_accumulation_steps"]),
                "packing": False,
                "offload": False,
                "kind": "throughput",
                "fidelity": "formal_3plus10",
                "warmup_steps": 3,
                "measure_steps": 10,
                "repeat": int(slot["repeat"]),
                "parallel_class": "gpu_partitionable",
                "requires_external_node_idle": False,
                "expected_padding_pressure": slot["expected_padding_pressure"],
            }
        )
    if len(jobs) != 8 or len({job["job_id"] for job in jobs}) != 8:
        raise ValueError(f"expected eight unique jobs, got {len(jobs)}")
    write_jsonl(DEFAULT_QUEUE, jobs)
    manifest = {
        "schema": "sft_h800_qwen3_14b_full_boundary_queue_manifest/v1",
        "campaign_id": design["campaign_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": {"path": str(DEFAULT_DESIGN.resolve()), "sha256": sha256_file(DEFAULT_DESIGN)},
        "queue": {
            "path": str(DEFAULT_QUEUE.resolve()),
            "sha256": sha256_file(DEFAULT_QUEUE),
            "canonical_rows_sha256": sha256_json(jobs),
        },
        "job_count": len(jobs),
        "gpu_job_equivalents": 16,
        "gpu_training_started": False,
        "execution_authorized": False,
    }
    write_json(DEFAULT_OUTPUT, manifest)
    print(f"wrote {DEFAULT_QUEUE}; jobs={len(jobs)}; execution_authorized=False")


if __name__ == "__main__":
    main()
