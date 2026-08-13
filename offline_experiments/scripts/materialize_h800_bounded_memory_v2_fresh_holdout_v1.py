#!/usr/bin/env python3
"""Materialize the frozen 21-job bounded-memory v2 fresh holdout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from common import ARTIFACT_DIR, ROOT, read_json, sha256_file, sha256_json, stable_id, write_json, write_jsonl


DESIGN_SCHEMA = "sft_h800_bounded_memory_v2_fresh_holdout_design/v1"
JOB_SCHEMA = "sft_h800_bounded_memory_v2_fresh_holdout_job/v1"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_bounded_memory_v2_fresh_holdout_design_v1.json"
DEFAULT_INVENTORY = ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_v1.json"
DEFAULT_Q35_RUNTIME = ARTIFACT_DIR / "h800_qwen35_runtime_contract_v1.json"
DEFAULT_QUEUE = ROOT / "matrix" / "h800_bounded_memory_v2_fresh_holdout_jobs_v1.jsonl"
DEFAULT_CANARY_QUEUE = ROOT / "matrix" / "h800_bounded_memory_v2_fresh_holdout_canary_v1.jsonl"
DEFAULT_FORMAL_QUEUE = ROOT / "matrix" / "h800_bounded_memory_v2_fresh_holdout_formal_v1.jsonl"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_bounded_memory_v2_fresh_holdout_queue_manifest_v1.json"


def _models(inventory_path: Path) -> dict[str, Mapping[str, Any]]:
    inventory = read_json(inventory_path)
    return {str(row["id"]): row for row in inventory["models"]}


def build_jobs(
    design: Mapping[str, Any],
    *,
    inventory_path: Path = DEFAULT_INVENTORY,
) -> list[dict[str, Any]]:
    unsigned = dict(design)
    expected = unsigned.pop("report_sha256", None)
    if design.get("schema") != DESIGN_SCHEMA or expected != sha256_json(unsigned):
        raise ValueError("bounded v2 fresh design schema or checksum mismatch")
    if (
        design.get("gpu_training_started") is not False
        or design.get("execution_authorized") is not False
        or design.get("status") != "predictions_frozen_waiting_for_queue_and_exact_approval"
    ):
        raise ValueError("bounded v2 fresh design is not materializable")
    inventory_binding = design["source_bindings"]["transfer_model_inventory"]
    if (
        Path(inventory_binding["path"]).resolve() != inventory_path.resolve()
        or inventory_binding["sha256"] != sha256_file(inventory_path)
    ):
        raise ValueError("transfer model inventory binding drifted")
    models = _models(inventory_path)
    q35_runtime_binding = design["source_bindings"]["qwen35_runtime_contract"]
    q35_runtime_path = Path(q35_runtime_binding["path"])
    if (
        q35_runtime_path.resolve() != DEFAULT_Q35_RUNTIME.resolve()
        or not q35_runtime_path.is_file()
        or q35_runtime_binding["sha256"] != sha256_file(q35_runtime_path)
    ):
        raise ValueError("Qwen3.5 runtime contract binding drifted")
    q35_runtime = read_json(q35_runtime_path)
    jobs: list[dict[str, Any]] = []
    q35_canary_assigned = False
    for slot in design["candidate_slots"]:
        for path_key, hash_key in (
            ("data_path", "data_sha256"),
            ("dataset_profile_path", "dataset_profile_sha256"),
        ):
            path = Path(slot[path_key])
            if not path.is_file() or sha256_file(path) != slot[hash_key]:
                raise ValueError(f"slot binding changed: {slot['candidate_slot_id']} {path_key}")
        partition = slot["calibration_partition"]
        if (
            partition.get("role") != "holdout"
            or partition.get("split_unit_id") != slot["profile_id"]
            or partition.get("policy") != "prospective_complete_s3_source_disjoint_v1"
        ):
            raise ValueError(f"invalid holdout partition: {slot['candidate_slot_id']}")
        model = models[str(slot["model_id"])]
        zero_stage = int(slot["zero_stage"])
        is_q35 = slot["candidate_role"] == "cross_scale_diagnostic"
        is_canary = bool(is_q35 and not q35_canary_assigned)
        q35_canary_assigned = q35_canary_assigned or is_canary
        job = {
                "schema": JOB_SCHEMA,
                "job_id": stable_id(
                    "h800boundedv2fresh",
                    {"campaign_id": design["campaign_id"], "candidate_slot_id": slot["candidate_slot_id"]},
                ),
                "campaign_id": design["campaign_id"],
                "phase_id": design["phase_id"],
                "scenario_id": slot["comparison_group"],
                "candidate_slot_id": slot["candidate_slot_id"],
                "predictor_request_id": slot["predictor_request_id"],
                "candidate_role": slot["candidate_role"],
                "software_canary": is_canary,
                "profile_id": slot["profile_id"],
                "calibration_partition": partition,
                "hardware_id": "local_h800_140g",
                "gpu_type": "NVIDIA H800 140GB HBM3",
                "required_runtime_gpu_name": "NVIDIA H800",
                "model_id": slot["model_id"],
                "model_path": model["path"],
                "tokenizer_path": model["tokenizer_path"],
                "model_family": model["family"],
                "model_parameters": model["actual_parameters"],
                "declared_model_manifest_path": str(inventory_path.resolve()),
                "declared_model_manifest_sha256": sha256_file(inventory_path),
                "template": model["template"],
                "train_type": slot["train_type"],
                "dataset_id": slot["dataset_id"],
                "dataset_category": slot["dataset_category"],
                "dataset_profile_path": slot["dataset_profile_path"],
                "dataset_profile_sha256": slot["dataset_profile_sha256"],
                "profile_tokenizer_id": slot["profile_tokenizer_id"],
                "profile_template_id": slot["profile_template_id"],
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
                "gradient_accumulation_steps": int(slot["gradient_accumulation_steps"]),
                "packing": False,
                "offload": False,
                "kind": "throughput",
                "fidelity": "formal_3plus10",
                "warmup_steps": 3,
                "measure_steps": 10,
                "repeat": 0,
                "parallel_class": "gpu_partitionable",
                "requires_external_node_idle": False,
                "expected_padding_pressure": slot["frozen_prediction"]["padding_statistics"],
                "frozen_prediction": slot["frozen_prediction"],
            }
        if is_q35:
            job["environment_overlay"] = {
                "contract_path": str(q35_runtime_path.resolve()),
                "contract_sha256": sha256_file(q35_runtime_path),
                "PYTHONPATH_prepend": q35_runtime["environment"]["PYTHONPATH_prepend"],
                "variables": q35_runtime["environment"]["variables"],
            }
        jobs.append(job)
    if len(jobs) != 21 or len({job["job_id"] for job in jobs}) != 21:
        raise ValueError(f"expected 21 unique jobs, got {len(jobs)}")
    if sum(job["software_canary"] is True for job in jobs) != 1:
        raise ValueError("exactly one Qwen3.5 software canary is required")
    role_order = {"cross_scale_diagnostic": 0, "base_selector": 1, "tail_forced_safety": 2}
    jobs.sort(
        key=lambda job: (
            0 if job["software_canary"] else 1,
            role_order[job["candidate_role"]],
            job["profile_id"] if "profile_id" in job else job["scenario_id"],
            job["job_id"],
        )
    )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--model-inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--canary-queue", type=Path, default=DEFAULT_CANARY_QUEUE)
    parser.add_argument("--formal-queue", type=Path, default=DEFAULT_FORMAL_QUEUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace only these generated queue artifacts after a frozen runtime/scope repair.",
    )
    args = parser.parse_args()
    outputs = (args.queue, args.canary_queue, args.formal_queue, args.output)
    if any(path.exists() for path in outputs) and not args.replace:
        raise SystemExit(f"refusing to overwrite existing queue material: {[str(path) for path in outputs if path.exists()]}")
    design = read_json(args.design)
    jobs = build_jobs(design, inventory_path=args.model_inventory)
    canary_jobs = [job for job in jobs if job["software_canary"]]
    formal_jobs = [job for job in jobs if not job["software_canary"]]
    if len(canary_jobs) != 1 or len(formal_jobs) != 20:
        raise ValueError("staged queue split must contain exactly one canary and 20 formal jobs")
    write_jsonl(args.queue, jobs)
    write_jsonl(args.canary_queue, canary_jobs)
    write_jsonl(args.formal_queue, formal_jobs)
    manifest = {
        "schema": "sft_h800_bounded_memory_v2_fresh_holdout_queue_manifest/v1",
        "campaign_id": design["campaign_id"],
        "phase_id": design["phase_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "path": str(args.design.resolve()),
            "sha256": sha256_file(args.design),
            "report_sha256": design["report_sha256"],
        },
        "model_inventory": {"path": str(args.model_inventory.resolve()), "sha256": sha256_file(args.model_inventory)},
        "queue": {
            "path": str(args.queue.resolve()),
            "sha256": sha256_file(args.queue),
            "canonical_rows_sha256": sha256_json(jobs),
        },
        "staged_queues": {
            "canary": {
                "path": str(args.canary_queue.resolve()),
                "sha256": sha256_file(args.canary_queue),
                "canonical_rows_sha256": sha256_json(canary_jobs),
                "job_count": len(canary_jobs),
            },
            "formal": {
                "path": str(args.formal_queue.resolve()),
                "sha256": sha256_file(args.formal_queue),
                "canonical_rows_sha256": sha256_json(formal_jobs),
                "job_count": len(formal_jobs),
            },
        },
        "job_count": len(jobs),
        "software_canary_job_id": next(job["job_id"] for job in jobs if job["software_canary"]),
        "one_gpu_jobs": sum(job["gpu_count"] == 1 for job in jobs),
        "two_gpu_jobs": sum(job["gpu_count"] == 2 for job in jobs),
        "gpu_job_equivalents": sum(int(job["gpu_count"]) for job in jobs),
        "gpu_training_started": False,
        "execution_authorized": False,
        "next_step": "freeze exact canary approval; run the one-job canary queue; freeze the 20-job formal approval only after canary success",
    }
    write_json(args.output, manifest)
    print(
        f"wrote aggregate={args.queue}, canary={args.canary_queue}, "
        f"formal={args.formal_queue}; jobs={len(jobs)}; execution_authorized=False"
    )


if __name__ == "__main__":
    main()
