#!/usr/bin/env python3
"""Freeze, validate, and promote the exact 36-job Packing Phase-B batch."""

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
    "用户于2026-08-05在确认Packing后续补实验计划后明确要求继续推进，并授权文档完成后"
    "直接启动实验，并使用所有空闲卡。资源状态在准备期间发生变化，GPU 0随后被外部单卡"
    "任务占用；本approval允许在H800 GPU 0至7上执行冻结的36个W3/W5/W7/W8 "
    "DP2/GBS128/ZeRO-2/GC-on/MBS1 U/P作业。不得抢占、扩展候选、自动启动Phase C-H、自动发布或把本批"
    "解释为两个分支各自调优后的最终route effect。启动时仅使用真实空闲卡，忙卡释放后"
    "才可通过join-busy-pool加入。"
)
CAMPAIGN_ID = "h800_packing_profile_phase_b_20260805_v1"
PHASE_ID = "h800_packing_profile_phase_b_v1"
JOB_SCHEMA = "sft_h800_packing_profile_phase_b_job/v1"
GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
EXPECTED_MASKS = [[0, 1], [2, 3], [4, 5], [6, 7]]
QUEUE = ROOT / "matrix/h800_packing_profile_phase_b_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_packing_profile_phase_b_design_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_profile_phase_b_queue_manifest_v1.json"
STATIC = ARTIFACT_DIR / "h800_packing_profile_phase_b_static_v1.json"
PREDICTIONS = ARTIFACT_DIR / "h800_packing_profile_phase_b_memory_predictions_v1.json"
INVENTORY = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_model_inventory_v1.json"


def _validate_queue(rows: list[dict[str, Any]]) -> list[str]:
    ids = [str(row.get("job_id") or "") for row in rows]
    if (
        len(rows) != 36
        or len(set(ids)) != 36
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
        or any(int(row.get("gpu_count", 0)) != 2 for row in rows)
        or any(row.get("zero") != "zero2" for row in rows)
        or any(row.get("gc") is not True for row in rows)
        or any(int(row.get("mbs", 0)) != 1 for row in rows)
        or any(int(row.get("target_gbs", 0)) != 128 for row in rows)
        or any(row.get("parallel_class") != "gpu_partitionable" for row in rows)
        or any(row.get("requires_external_node_idle") is not False for row in rows)
        or any(row.get("publication_allowed") is not False for row in rows)
        or any(row.get("final_route_effect_claim_allowed") is not False for row in rows)
        or any(
            row.get("production_optimizer_step_gate")
            != "not_applicable_fit_only_probe"
            for row in rows
        )
        or [row.get("execution_sequence_index") for row in rows] != list(range(36))
    ):
        raise ValueError("queue is not the exact frozen 36-job Phase-B batch")

    expected_families = {
        "w3-c4096-dp2-g128": (4_096, 19),
        "w3-c40960-dp2-g128": (40_960, 2),
        "w5-c16384-dp2-g128": (16_384, 35),
        "w7-c20480-dp2-g128": (20_480, 16),
        "w8-c2048-dp2-g128": (2_048, 15),
        "w8-c10240-dp2-g128": (10_240, 3),
    }
    if {str(row["family_id"]) for row in rows} != set(expected_families):
        raise ValueError("Phase-B family set drifted")
    for family_id, (cutoff_len, packed_ga) in expected_families.items():
        subset = [row for row in rows if row["family_id"] == family_id]
        if len(subset) != 6:
            raise ValueError(f"{family_id}: expected six jobs")
        seen = {(packing, repeat): 0 for packing in (False, True) for repeat in range(3)}
        for row in subset:
            packing = bool(row["packing"])
            repeat = int(row["repeat"])
            seen[(packing, repeat)] += 1
            expected_ga = packed_ga if packing else 64
            if (
                int(row["cutoff_len"]) != cutoff_len
                or int(row["gradient_accumulation_steps"]) != expected_ga
            ):
                raise ValueError(f"{family_id}: cutoff/GA drifted")
        if any(value != 1 for value in seen.values()):
            raise ValueError(f"{family_id}: unbalanced U/P repeats")
    packed = [row for row in rows if row["packing"] is True]
    if (
        len(packed) != 18
        or any(
            row.get("gbs_contract_v2", {}).get("gates", {}).get(
                "candidate_admissible"
            )
            is not True
            for row in packed
        )
        or any(float(row["expected_sample_gbs_relative_error"]) > 0.05 for row in packed)
        or any(
            float(row["memory_execution_guarded_upper_bytes"])
            > float(row["memory_execution_limit_bytes"])
            for row in rows
        )
    ):
        raise ValueError("a Packed job violates the GBS or memory execution gate")
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
        or scope.get("gradient_checkpointing") != [True]
        or scope.get("zero_by_gpu_count") != {"2": ["zero2"]}
        or experiment.get("measurement", {}).get("performance_parallelism")
        != "disjoint_gpu_masks"
    ):
        raise ValueError("experiment config is not the exact Phase-B scope")

    design = read_json(DESIGN)
    manifest = read_json(QUEUE_MANIFEST)
    static = read_json(STATIC)
    if (
        design.get("schema") != "sft_h800_packing_profile_phase_b_design/v1"
        or design.get("campaign_id") != CAMPAIGN_ID
        or design.get("gpu_training_started") is not False
        or design.get("automatic_next_batch_allowed") is not False
        or design.get("publication_allowed") is not False
        or design.get("required_gpu_pool", {}).get("gpu_ids") != GPU_IDS
        or design.get("required_gpu_pool", {}).get(
            "preview_two_gpu_masks_when_all_idle"
        )
        != EXPECTED_MASKS
        or design.get("required_gpu_pool", {}).get("two_gpu_mask_policy")
        != "any_disjoint_pair_within_fully_nvlinked_approved_pool"
        or design.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or design.get("queue", {}).get("ordered_job_ids") != ids
        or design.get("inference_contract", {}).get(
            "best_branch_route_effect_claim_allowed"
        )
        is not False
        or manifest.get("design", {}).get("sha256") != sha256_file(DESIGN)
        or manifest.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or static.get("recommendation_path_reads_raw_data") is not False
        or static.get("recommendation_path_reads_raw_lengths") is not False
        or static.get("recommendation_path_runs_full_packer") is not False
        or static.get("memory_preflight", {}).get("all_passed") is not True
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
    initial_blocked_gpu_ids = sorted(
        {int(row["gpu_index"]) for row in live_processes}
    )
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
    source_bindings = list((design.get("source_bindings") or {}).values())
    for binding in source_bindings:
        bound_files.add(Path(binding["path"]))
    for row in rows:
        bound_files.add(Path(row["data_path"]))
        bound_files.add(Path(row["dataset_profile_path"]))
        bound_files.add(Path(row["packing_dataprofile_path"]))
    missing = [str(path) for path in bound_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files missing: {missing}")
    mismatched = [
        str(binding["path"])
        for binding in source_bindings
        if sha256_file(Path(binding["path"])) != binding.get("sha256")
    ]
    if mismatched:
        raise ValueError(f"campaign source bindings drifted: {mismatched}")
    root = ROOT.resolve()
    file_manifest = {
        str(path.resolve().relative_to(root)): sha256_file(path)
        for path in sorted(bound_files, key=lambda value: str(value))
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
            "policy": (
                "use only idle disjoint masks [0,1], [2,3], [4,5], [6,7]; "
                "wait rather than preempt"
            ),
        },
        "hardware_preflight": hardware,
        "idle_process_preflight": {
            "gpu_ids": GPU_IDS,
            "processes": live_processes,
            "initial_blocked_gpu_ids": initial_blocked_gpu_ids,
            "all_idle": not initial_blocked_gpu_ids,
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
    candidate = (
        ARTIFACT_DIR
        / "approval_design_h800_packing_profile_phase_b_v1_candidate.json"
    )
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
    print(
        json.dumps(
            {
                "candidate": str(candidate),
                "sha256": digest,
                "promotion": report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
