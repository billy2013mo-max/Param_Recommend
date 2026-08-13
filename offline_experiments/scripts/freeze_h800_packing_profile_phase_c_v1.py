#!/usr/bin/env python3
"""Freeze, validate, and optionally promote the 24-job Packing Phase-C batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from approval_gate import build_provenance_binding, build_queue_binding
from common import (
    ARTIFACT_DIR,
    ROOT,
    gpu_process_snapshot,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch


AUTHORIZATION = (
    "用户于2026-08-05在获知下一步是Packing Phase C交互标定后明确要求继续推进。"
    "本approval仅允许在H800 GPU 0至7的真实空闲卡上执行冻结的24个DP2/GBS128/MBS1作业："
    "W7@20480与W3@40960的Packing×ZeRO-3/GC-on，以及W3@4096与W8@10240的"
    "Packing×ZeRO-2/GC-off。长cutoff GC-off因冻结显存P95超过90%容量而明确禁止。"
    "不得抢占、自动扩展候选、自动发布或外推长cutoff GC交互。"
)
CAMPAIGN_ID = "h800_packing_profile_phase_c_20260805_v1"
PHASE_ID = "h800_packing_profile_phase_c_v1"
JOB_SCHEMA = "sft_h800_packing_profile_phase_c_job/v1"
GPU_IDS = list(range(8))
EXPECTED_MASKS = [[0, 1], [2, 3], [4, 5], [6, 7]]
QUEUE = ROOT / "matrix/h800_packing_profile_phase_c_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_packing_profile_phase_c_design_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_profile_phase_c_queue_manifest_v1.json"
STATIC = ARTIFACT_DIR / "h800_packing_profile_phase_c_static_v1.json"
PREDICTIONS = ARTIFACT_DIR / "h800_packing_profile_phase_c_memory_predictions_v1.json"
INVENTORY = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_model_inventory_v1.json"

EXPECTED_SETTINGS = {
    "w7-c20480-z3-gcon": ("packing_x_zero", "zero3", True, 20_480, 16),
    "w3-c40960-z3-gcon": ("packing_x_zero", "zero3", True, 40_960, 2),
    "w3-c4096-z2-gcoff": ("packing_x_gc", "zero2", False, 4_096, 19),
    "w8-c10240-z2-gcoff": ("packing_x_gc", "zero2", False, 10_240, 3),
}


def _validate_queue(rows: list[dict[str, Any]]) -> list[str]:
    ids = [str(row.get("job_id") or "") for row in rows]
    if (
        len(rows) != 24
        or len(set(ids)) != 24
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
        or any(int(row.get("gpu_count", 0)) != 2 for row in rows)
        or any(int(row.get("target_gbs", 0)) != 128 for row in rows)
        or any(int(row.get("mbs", 0)) != 1 for row in rows)
        or any(row.get("parallel_class") != "gpu_partitionable" for row in rows)
        or any(row.get("requires_external_node_idle") is not False for row in rows)
        or any(row.get("publication_allowed") is not False for row in rows)
        or any(row.get("final_route_effect_claim_allowed") is not False for row in rows)
        or [row.get("execution_sequence_index") for row in rows] != list(range(24))
    ):
        raise ValueError("queue is not the exact frozen 24-job Phase-C batch")
    if {str(row["setting_id"]) for row in rows} != set(EXPECTED_SETTINGS):
        raise ValueError("Phase-C setting set drifted")
    for setting_id, (axis, zero, gc, cutoff, packed_ga) in EXPECTED_SETTINGS.items():
        subset = [row for row in rows if row["setting_id"] == setting_id]
        seen = {(packing, repeat): 0 for packing in (False, True) for repeat in range(3)}
        for row in subset:
            key = (bool(row["packing"]), int(row["repeat"]))
            if key not in seen:
                raise ValueError(f"{setting_id}: invalid treatment/repeat {key}")
            seen[key] += 1
            expected_ga = packed_ga if key[0] else 64
            if (
                row["interaction_axis"] != axis
                or row["zero"] != zero
                or row["gc"] is not gc
                or row["gradient_checkpointing"] is not gc
                or int(row["cutoff_len"]) != cutoff
                or int(row["gradient_accumulation_steps"]) != expected_ga
                or float(row["expected_sample_gbs_relative_error"]) > 0.05
                or float(row["memory_execution_guarded_upper_bytes"])
                > float(row["memory_execution_limit_bytes"])
            ):
                raise ValueError(f"{setting_id}: frozen contract drifted")
        if len(subset) != 6 or any(value != 1 for value in seen.values()):
            raise ValueError(f"{setting_id}: unbalanced U/P repeats")
    return ids


def freeze() -> Path:
    rows = read_jsonl(QUEUE)
    ids = _validate_queue(rows)
    experiment = read_json(ROOT / "config/experiment.json")
    scope = experiment["training_scope"]
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("model_ids") != ["qwen3_8b"]
        or scope.get("gpu_ids") != GPU_IDS
        or scope.get("exclusive_node_gpu_ids") != GPU_IDS
        or scope.get("max_gpu_count") != 2
        or scope.get("gpu_counts") != [2]
        or scope.get("global_batch_sizes") != [128]
        or scope.get("gradient_checkpointing") != [True, False]
        or scope.get("zero_by_gpu_count") != {"2": ["zero2", "zero3"]}
        or experiment.get("measurement", {}).get("performance_parallelism")
        != "disjoint_gpu_masks"
    ):
        raise ValueError("experiment config is not the exact Phase-C scope")
    design = read_json(DESIGN)
    manifest = read_json(QUEUE_MANIFEST)
    static = read_json(STATIC)
    if (
        design.get("schema") != "sft_h800_packing_profile_phase_c_design/v1"
        or design.get("campaign_id") != CAMPAIGN_ID
        or design.get("gpu_training_started") is not False
        or design.get("automatic_next_batch_allowed") is not False
        or design.get("publication_allowed") is not False
        or design.get("required_gpu_pool", {}).get("gpu_ids") != GPU_IDS
        or design.get("required_gpu_pool", {}).get("preview_two_gpu_masks_when_all_idle")
        != EXPECTED_MASKS
        or design.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or design.get("queue", {}).get("ordered_job_ids") != ids
        or design.get("safety_amendment", {}).get("original_long_cutoff_gc_off_admitted")
        is not False
        or design.get("inference_contract", {}).get(
            "gc_interaction_at_long_cutoff_claim_allowed"
        )
        is not False
        or manifest.get("design", {}).get("sha256") != sha256_file(DESIGN)
        or manifest.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or static.get("memory_preflight", {}).get("selected_all_passed") is not True
        or static.get("memory_preflight", {}).get("excluded_all_failed") is not True
        or static.get("memory_preflight", {}).get(
            "automatic_packing_admission_allowed"
        )
        is not False
    ):
        raise ValueError("design/static/manifest drifted")

    hardware = probe_hardware(required_gpu_ids=GPU_IDS)
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"selected GPUs are not an exact H800 pool: {hardware}")
    live_processes = gpu_process_snapshot(GPU_IDS).get("processes") or []
    blocked = sorted({int(row["gpu_index"]) for row in live_processes})
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(QUEUE, rows, ROOT)
    provenance_binding = build_provenance_binding(ROOT)

    bound_files = {
        QUEUE,
        DESIGN,
        QUEUE_MANIFEST,
        STATIC,
        PREDICTIONS,
        INVENTORY,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ARTIFACT_DIR / "dataset_analysis.json",
        ROOT / "config/experiment.json",
        ROOT / "config/hardware.json",
        ROOT / "data/dataset_info.json",
    }
    for binding in (design.get("source_bindings") or {}).values():
        bound_files.add(Path(binding["path"]))
        if sha256_file(Path(binding["path"])) != binding.get("sha256"):
            raise ValueError(f"campaign source binding drifted: {binding['path']}")
    for row in rows:
        for key in ("data_path", "dataset_profile_path", "packing_dataprofile_path"):
            bound_files.add(Path(row[key]))
    missing = [str(path) for path in bound_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files missing: {missing}")
    root = ROOT.resolve()
    file_manifest = {
        str(path.resolve().relative_to(root)): sha256_file(path)
        for path in sorted(bound_files, key=str)
        if path.resolve().is_relative_to(root)
    }
    stage = {
        "allowed_job_ids": ids,
        "queue_path": queue_binding["path"],
        "queue_sha256": queue_binding["sha256"],
        "ordered_job_ids": queue_binding["ordered_job_ids"],
        "ordered_job_payload_sha256": queue_binding["ordered_job_payload_sha256"],
        "job_payload_sha256": queue_binding["job_payload_sha256"],
    }
    approval: dict[str, Any] = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}",
        "campaign_design": {
            "path": str(DESIGN.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(DESIGN),
            "report_sha256": design["report_sha256"],
        },
        "file_sha256": file_manifest,
        "execution_order": [PHASE_ID],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": GPU_IDS,
        "max_gpu_count": 2,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": "use only idle disjoint H800 pairs; wait rather than preempt",
        },
        "hardware_preflight": hardware,
        "idle_process_preflight": {
            "gpu_ids": GPU_IDS,
            "processes": live_processes,
            "initial_blocked_gpu_ids": blocked,
            "all_idle": not blocked,
            "join_busy_pool": True,
            "preemption_allowed": False,
        },
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": stage,
    }
    candidate = ARTIFACT_DIR / "approval_design_h800_packing_profile_phase_c_v1_candidate.json"
    write_json(candidate, approval)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = freeze()
    digest = sha256_file(candidate)
    report = promote_candidate(
        candidate_path=candidate,
        expected_candidate_sha256=digest,
        project_root=ROOT,
        authorization=AUTHORIZATION,
        approved_by="user",
        promote=args.promote,
    )
    print(json.dumps({"candidate": str(candidate), "sha256": digest, "promotion": report}, ensure_ascii=False, indent=2))
    if report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
