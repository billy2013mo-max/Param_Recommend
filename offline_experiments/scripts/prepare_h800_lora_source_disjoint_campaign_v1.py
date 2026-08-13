#!/usr/bin/env python3
"""Materialize the 15-source, 77-job H800 LoRA memory campaign.

The campaign implements groups A1--A4 from the Chinese experiment plan.  It
only writes an immutable design, JSONL queue, and queue manifest; it does not
authorize or launch GPU training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

from common import ARTIFACT_DIR, ROOT, read_json, sha256_file, sha256_json, stable_id, write_json, write_jsonl


CAMPAIGN_ID = "h800_lora_source_disjoint_recalibration_20260804_v1"
PHASE_ID = "h800_lora_source_disjoint_recalibration_v1"
DESIGN_SCHEMA = "sft_h800_lora_source_disjoint_experiment_design/v1"
JOB_SCHEMA = "sft_h800_lora_source_disjoint_job/v1"
TARGET_GBS = 64
DEFAULT_BUNDLE = ARTIFACT_DIR / "h800_lora_source_disjoint_bundle_v1.json"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_lora_source_disjoint_experiment_design_v1.json"
DEFAULT_QUEUE = ROOT / "matrix" / "h800_lora_source_disjoint_jobs_v1.jsonl"
DEFAULT_MANIFEST = ARTIFACT_DIR / "h800_lora_source_disjoint_queue_manifest_v1.json"


def _binding(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _models() -> dict[str, Mapping[str, Any]]:
    inventory = read_json(ARTIFACT_DIR / "model_inventory.json")
    return {str(row["id"]): row for row in inventory["models"]}


def _base_job(
    scenario: Mapping[str, Any],
    model: Mapping[str, Any],
    *, group: str,
    purpose: str,
    train_type: str,
    gpu_count: int,
    zero_stage: int,
    gc: bool,
    mbs: int,
    repeat: int,
) -> dict[str, Any]:
    if TARGET_GBS % (gpu_count * mbs):
        raise ValueError("target GBS must be divisible by GPU count times MBS")
    if train_type == "lora" and zero_stage != 2:
        raise ValueError("critical LoRA jobs must use ZeRO-2")
    if train_type == "full" and zero_stage != 3:
        raise ValueError("Full controls must use ZeRO-3")
    identity = {
        "campaign_id": CAMPAIGN_ID,
        "scenario_id": scenario["scenario_id"],
        "group": group,
        "purpose": purpose,
        "train_type": train_type,
        "gpu_count": gpu_count,
        "zero_stage": zero_stage,
        "gc": gc,
        "mbs": mbs,
        "repeat": repeat,
    }
    return {
        "schema": JOB_SCHEMA,
        "job_id": stable_id("h800loramem", identity),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "scenario_id": scenario["scenario_id"],
        "split_unit_id": scenario["split_unit_id"],
        "experiment_group": group,
        "calibration_purpose": purpose,
        "calibration_partition": {
            "role": "calibration",
            "split_unit_id": scenario["split_unit_id"],
            "policy": "remote_source_dataset_id_disjoint_v1",
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
        "train_type": train_type,
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
        "zero_stage": zero_stage,
        "zero": f"zero{zero_stage}",
        "gc": gc,
        "gradient_checkpointing": gc,
        "mbs": mbs,
        "gradient_accumulation_steps": TARGET_GBS // (gpu_count * mbs),
        "packing": False,
        "offload": False,
        "kind": "throughput",
        "fidelity": "formal_3plus10",
        "warmup_steps": 3,
        "measure_steps": 10,
        "repeat": repeat,
        "max_samples": int(scenario["profile_statistics"]["rows"]),
        "parallel_class": "gpu_partitionable",
        "requires_external_node_idle": False,
    }


def build_jobs(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    if bundle.get("schema") != "sft_h800_lora_source_disjoint_data_bundle/v1":
        raise ValueError("source-disjoint data bundle schema mismatch")
    unsigned = dict(bundle)
    expected_hash = unsigned.pop("report_sha256", None)
    if expected_hash != sha256_json(unsigned):
        raise ValueError("source-disjoint data bundle internal hash mismatch")
    if bundle.get("gpu_training_started") is not False:
        raise ValueError("source bundle must remain pre-GPU")
    if bundle.get("counts", {}).get("scenarios") != 15:
        raise ValueError("source bundle must bind exactly 15 scenarios")

    models = _models()
    scenarios = list(bundle["scenarios"])
    jobs_by_group: dict[str, list[dict[str, Any]]] = {key: [] for key in ("A1", "A2", "A3", "A4")}

    # A1: all 15 scenarios at MBS 1/2/4 on the critical LoRA selector.
    for mbs, purpose in (
        (1, "a1_low_pressure_mbs1"),
        (2, "a1_boundary_candidate_mbs2"),
        (4, "a1_high_pressure_mbs4"),
    ):
        for scenario in scenarios:
            model = models[str(scenario["model_id"])]
            jobs_by_group["A1"].append(
                _base_job(
                    scenario, model, group="A1", purpose=purpose,
                    train_type="lora", gpu_count=2, zero_stage=2,
                    gc=False, mbs=mbs, repeat=0,
                )
            )

    # A2: two Full/ZeRO-3/GC-on MBS controls in two scenarios per ratio bin.
    a2 = [row for row in scenarios if row["matrix_roles"]["a2_full_control"]]
    for mbs in (1, 2):
        for scenario in a2:
            model = models[str(scenario["model_id"])]
            jobs_by_group["A2"].append(
                _base_job(
                    scenario, model, group="A2", purpose=f"a2_full_effective_sequence_mbs{mbs}",
                    train_type="full", gpu_count=2, zero_stage=3,
                    gc=True, mbs=mbs, repeat=0,
                )
            )

    # A3: one scenario per ratio bin, with matched LoRA and Full 4-GPU MBS=2.
    a3 = [row for row in scenarios if row["matrix_roles"]["a3_four_gpu_pair"]]
    for train_type, zero_stage, gc in (("lora", 2, False), ("full", 3, True)):
        for scenario in a3:
            model = models[str(scenario["model_id"])]
            jobs_by_group["A3"].append(
                _base_job(
                    scenario, model, group="A3",
                    purpose=f"a3_four_gpu_{train_type}_mbs2",
                    train_type=train_type, gpu_count=4, zero_stage=zero_stage,
                    gc=gc, mbs=2, repeat=0,
                )
            )

    # A4: two exact extra attempts of the MBS=2 LoRA boundary candidate.
    a4 = [row for row in scenarios if row["matrix_roles"]["a4_repeat_control"]]
    for repeat in (1, 2):
        for scenario in a4:
            model = models[str(scenario["model_id"])]
            jobs_by_group["A4"].append(
                _base_job(
                    scenario, model, group="A4",
                    purpose="a4_allocator_repeat_boundary_mbs2",
                    train_type="lora", gpu_count=2, zero_stage=2,
                    gc=False, mbs=2, repeat=repeat,
                )
            )

    jobs = jobs_by_group["A1"] + jobs_by_group["A2"] + jobs_by_group["A4"] + jobs_by_group["A3"]
    expected_groups = {"A1": 45, "A2": 16, "A3": 8, "A4": 8}
    actual_groups = Counter(job["experiment_group"] for job in jobs)
    if dict(actual_groups) != expected_groups:
        raise ValueError(f"job group count mismatch: {dict(actual_groups)}")
    if len(jobs) != 77 or len({job["job_id"] for job in jobs}) != 77:
        raise ValueError(f"expected 77 unique jobs, got {len(jobs)}")
    if sum(int(job["gpu_count"]) for job in jobs) != 170:
        raise ValueError("expected 170 GPU-job equivalents")
    for role_rows, expected in ((a2, 8), (a3, 4), (a4, 4)):
        if len(role_rows) != expected:
            raise ValueError(f"scenario role count mismatch: expected {expected}, got {len(role_rows)}")
        bins = Counter(row["ratio_bin"] for row in role_rows)
        expected_per_bin = 2 if expected == 8 else 1
        if set(bins.values()) != {expected_per_bin} or len(bins) != 4:
            raise ValueError(f"scenario role is not ratio-bin balanced: {dict(bins)}")
    return jobs


def build_design(bundle_path: Path, bundle: Mapping[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    groups = Counter(job["experiment_group"] for job in jobs)
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
            "Recalibrate H800 non-packing memory on the physically achievable effective "
            "sequence using 15 new source-disjoint profiles across four max/cutoff bins."
        ),
        "source_bundle": {**_binding(bundle_path), "report_sha256": bundle["report_sha256"]},
        "design": {
            "new_independent_split_units": 15,
            "job_count": len(jobs),
            "gpu_job_equivalents": sum(int(job["gpu_count"]) for job in jobs),
            "job_groups": dict(groups),
            "ratio_bins": bundle["counts"]["ratio_bins"],
            "models": bundle["counts"]["models"],
            "a1_mbs_points": [1, 2, 4],
            "a2_full_mbs_points": [1, 2],
            "a3_matched_mbs": 2,
            "a4_extra_repeats": [1, 2],
            "target_gbs": TARGET_GBS,
            "fidelity": "warmup 3 plus measure 10 optimizer steps",
        },
        "adaptive_interpretation": {
            "a1_mbs2_is_boundary_candidate_not_assumed_success": True,
            "minimum_two_successful_points_per_split_unit_required_for_fit": True,
            "if_mbs2_oom": "do not treat MBS4 as a boundary success; prepare a separately approved repair point",
            "infra_failure_is_not_oom": True,
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
        "schema": "sft_h800_lora_source_disjoint_queue_manifest/v1",
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
        "next_step": "capture fresh provenance and promote an exact six-GPU approval before execution",
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
                "groups": dict(Counter(job["experiment_group"] for job in jobs)),
                "execution_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
