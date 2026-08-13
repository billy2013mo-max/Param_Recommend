#!/usr/bin/env python3
"""Materialize the frozen ten-job final unseen-profile H800 holdout."""

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


DESIGN_SCHEMA = "sft_h800_final_unseen_holdout_design/v1"
JOB_SCHEMA = "sft_h800_final_unseen_holdout_job/v1"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_final_unseen_holdout_design_v1.json"
DEFAULT_QUEUE = ROOT / "matrix" / "h800_final_unseen_holdout_jobs_v1.jsonl"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_final_unseen_holdout_queue_manifest_v1.json"


def _models() -> dict[str, Mapping[str, Any]]:
    inventory = read_json(ARTIFACT_DIR / "model_inventory.json")
    return {str(row["id"]): row for row in inventory["models"]}


def build_jobs(design: Mapping[str, Any]) -> list[dict[str, Any]]:
    unsigned = dict(design)
    expected = unsigned.pop("report_sha256", None)
    if design.get("schema") != DESIGN_SCHEMA or expected != sha256_json(unsigned):
        raise ValueError("final unseen holdout design schema or checksum mismatch")
    if (
        design.get("gpu_training_started") is not False
        or design.get("execution_authorized") is not False
        or design.get("status")
        != "predictions_frozen_waiting_for_queue_and_exact_approval"
    ):
        raise ValueError("final unseen holdout design is not materializable")
    models = _models()
    jobs: list[dict[str, Any]] = []
    for slot in design["candidate_slots"]:
        for path_key, hash_key in (
            ("data_path", "data_sha256"),
            ("dataset_profile_path", "dataset_profile_sha256"),
        ):
            path = Path(slot[path_key])
            if not path.is_file() or sha256_file(path) != slot[hash_key]:
                raise ValueError(
                    f"slot binding changed: {slot['candidate_slot_id']} {path_key}"
                )
        partition = slot["calibration_partition"]
        if (
            partition.get("role") != "holdout"
            or partition.get("split_unit_id") != slot["profile_id"]
            or partition.get("policy") != "prospective_unseen_dense_profiles_v1"
        ):
            raise ValueError(
                f"invalid holdout partition: {slot['candidate_slot_id']}"
            )
        model = models[str(slot["model_id"])]
        zero_stage = int(slot["zero_stage"])
        jobs.append(
            {
                "schema": JOB_SCHEMA,
                "job_id": stable_id(
                    "h800finalunseen",
                    {
                        "campaign_id": design["campaign_id"],
                        "candidate_slot_id": slot["candidate_slot_id"],
                    },
                ),
                "campaign_id": design["campaign_id"],
                "phase_id": design["phase_id"],
                "scenario_id": slot["comparison_group"],
                "candidate_slot_id": slot["candidate_slot_id"],
                "predictor_request_id": slot["predictor_request_id"],
                "calibration_partition": partition,
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
                "dataset_profile_path": slot["dataset_profile_path"],
                "dataset_profile_sha256": slot["dataset_profile_sha256"],
                "data_path": slot["data_path"],
                "data_sha256": slot["data_sha256"],
                "cutoff_len": int(slot["cutoff_len"]),
                "target_gbs": int(slot["target_gbs"]),
                "gpu_count": int(slot["gpu_count"]),
                "zero_stage": zero_stage,
                "zero": "none" if zero_stage == 0 else f"zero{zero_stage}",
                "gc": bool(slot["gc"]),
                "gradient_checkpointing": bool(slot["gc"]),
                "mbs": int(slot["mbs"]),
                "gradient_accumulation_steps": int(
                    slot["gradient_accumulation_steps"]
                ),
                "packing": False,
                "offload": False,
                "kind": "throughput",
                "fidelity": "formal_3plus10",
                "warmup_steps": 3,
                "measure_steps": 10,
                "repeat": 0,
                "parallel_class": "gpu_partitionable",
                "requires_external_node_idle": False,
                "expected_padding_pressure": slot["frozen_prediction"][
                    "padding_statistics"
                ],
                "frozen_prediction": slot["frozen_prediction"],
            }
        )
    if len(jobs) != 10 or len({job["job_id"] for job in jobs}) != 10:
        raise ValueError(f"expected ten unique jobs, got {len(jobs)}")
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
        "schema": "sft_h800_final_unseen_holdout_queue_manifest/v1",
        "campaign_id": design["campaign_id"],
        "phase_id": design["phase_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "path": str(args.design.resolve()),
            "sha256": sha256_file(args.design),
            "report_sha256": design["report_sha256"],
        },
        "queue": {
            "path": str(args.queue.resolve()),
            "sha256": sha256_file(args.queue),
            "canonical_rows_sha256": sha256_json(jobs),
        },
        "job_count": len(jobs),
        "one_gpu_jobs": sum(job["gpu_count"] == 1 for job in jobs),
        "two_gpu_jobs": sum(job["gpu_count"] == 2 for job in jobs),
        "gpu_job_equivalents": sum(int(job["gpu_count"]) for job in jobs),
        "gpu_training_started": False,
        "execution_authorized": False,
        "next_step": "capture provenance, freeze and promote exact GPU-4,5 approval",
    }
    write_json(args.output, manifest)
    print(
        f"wrote {args.queue}; jobs={len(jobs)}; execution_authorized=False"
    )


if __name__ == "__main__":
    main()
