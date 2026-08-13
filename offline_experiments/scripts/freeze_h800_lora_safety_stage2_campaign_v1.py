#!/usr/bin/env python3
"""Freeze and optionally promote the exact 20-source, 60-job LoRA campaign."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch


AUTHORIZATION = (
    "用户在2026-08-05明确要求按照安全上界纠正的落地顺序推进第二阶段实验；"
    "本次只允许执行20个新独立来源、60个critical LoRA作业，并且只使用原授权GPU池"
    "0、1、4、5、6、7中实时空闲的成对GPU。不得使用GPU 2、3，不得终止或抢占外部"
    "进程，不得自动发布模型。"
)
CAMPAIGN_ID = "h800_lora_safety_stage2_20260805_v1"
PHASE_ID = "h800_lora_safety_stage2_v1"
JOB_SCHEMA = "sft_h800_lora_safety_stage2_job/v1"
AUTHORIZED_GPU_IDS = (0, 1, 4, 5, 6, 7)
AUTHORIZED_PAIRS = ((0, 1), (4, 5), (6, 7))
DEFAULT_QUEUE = ROOT / "matrix" / "h800_lora_safety_stage2_jobs_v1.jsonl"
DEFAULT_CAMPAIGN_DESIGN = ARTIFACT_DIR / "h800_lora_safety_stage2_experiment_design_v1.json"
DEFAULT_QUEUE_MANIFEST = ARTIFACT_DIR / "h800_lora_safety_stage2_queue_manifest_v1.json"
DEFAULT_BUNDLE = ARTIFACT_DIR / "h800_lora_safety_stage2_bundle_v1.json"
DEFAULT_PREPROCESSING = ARTIFACT_DIR / "h800_lora_safety_stage2_preprocessing_validation_v1.json"


def _hardware_for_join_busy_pool() -> dict[str, object]:
    hardware = probe_hardware(required_gpu_ids=AUTHORIZED_GPU_IDS)
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"GPU pool is not the exact authorized H800 pool: {hardware}")
    uuid_to_index = {
        str(row["uuid"]): int(row["index"])
        for row in hardware["selected_gpu_rows"]
    }
    busy_ids = sorted(
        {
            uuid_to_index[str(row["gpu_uuid"])]
            for row in hardware["selected_gpu_compute_processes"]
            if str(row["gpu_uuid"]) in uuid_to_index
        }
    )
    idle_pairs = [list(pair) for pair in AUTHORIZED_PAIRS if not set(pair) & set(busy_ids)]
    if not idle_pairs:
        raise RuntimeError(
            "No authorized two-GPU pair is currently idle; refusing to promote a campaign "
            "that cannot make immediate progress"
        )
    hardware.update(
        {
            "occupancy_is_advisory": True,
            "busy_gpu_ids_at_freeze": busy_ids,
            "idle_authorized_pairs_at_freeze": idle_pairs,
            "join_busy_pool_required": True,
            "preemption_allowed": False,
        }
    )
    return hardware


def freeze(
    queue: Path,
    campaign_design: Path,
    queue_manifest: Path,
    bundle_path: Path,
    preprocessing_path: Path,
) -> Path:
    rows = read_jsonl(queue)
    if (
        len(rows) != 60
        or len({str(row.get("job_id")) for row in rows}) != 60
        or Counter(int(row.get("mbs", 0)) for row in rows) != {1: 20, 2: 20, 4: 20}
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
        or any(row.get("experiment_group") != "S2" for row in rows)
        or any(row.get("train_type") != "lora" for row in rows)
        or any(int(row.get("gpu_count", 0)) != 2 for row in rows)
        or any(int(row.get("zero_stage", 0)) != 2 for row in rows)
        or any(row.get("gradient_checkpointing") is not False for row in rows)
        or any(row.get("packing") is not False for row in rows)
    ):
        raise ValueError("queue is not the exact 60-job stage-two critical-LoRA campaign")
    if sum(int(row["gpu_count"]) for row in rows) != 120:
        raise ValueError("queue must contain exactly 120 GPU-job equivalents")
    partitions = [row.get("calibration_partition") or {} for row in rows]
    if any(
        part.get("role") != "calibration"
        or part.get("policy") != "remote_source_dataset_id_disjoint_stage2_v1"
        or not part.get("split_unit_id")
        for part in partitions
    ):
        raise ValueError("every queue row must carry the stage-two source partition")
    if len({part["split_unit_id"] for part in partitions}) != 20:
        raise ValueError("queue must contain exactly 20 independent split-unit IDs")

    design = read_json(campaign_design)
    queue_meta = read_json(queue_manifest)
    bundle = read_json(bundle_path)
    preprocessing = read_json(preprocessing_path)
    if (
        design.get("schema") != "sft_h800_lora_safety_stage2_experiment_design/v1"
        or design.get("campaign_id") != CAMPAIGN_ID
        or design.get("ordered_job_ids") != [row["job_id"] for row in rows]
        or design.get("ordered_job_payload_sha256") != sha256_json(rows)
        or queue_meta.get("queue", {}).get("sha256") != sha256_file(queue)
        or queue_meta.get("design", {}).get("sha256") != sha256_file(campaign_design)
        or queue_meta.get("queue", {}).get("ordered_job_payload_sha256") != sha256_json(rows)
    ):
        raise ValueError("campaign design or queue manifest binding drifted")
    if (
        bundle.get("counts", {}).get("scenarios") != 20
        or bundle.get("gpu_training_started") is not False
        or design.get("source_bundle", {}).get("sha256") != sha256_file(bundle_path)
    ):
        raise ValueError("20-source frozen bundle binding drifted")
    if (
        preprocessing.get("all_passed") is not True
        or len(preprocessing.get("checks") or []) != 20
        or preprocessing.get("bundle", {}).get("sha256") != sha256_file(bundle_path)
        or preprocessing.get("dataset_info", {}).get("sha256")
        != sha256_file(ROOT / "data" / "dataset_info.json")
    ):
        raise ValueError("installed LLaMA-Factory preprocessing validation is not current")

    experiment = read_json(ROOT / "config" / "experiment.json")
    scope = experiment["training_scope"]
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("model_ids") != ["qwen3_4b", "qwen3_8b", "qwen3_14b"]
        or scope.get("gpu_ids") != list(AUTHORIZED_GPU_IDS)
        or scope.get("exclusive_node_gpu_ids") != list(AUTHORIZED_GPU_IDS)
        or scope.get("max_gpu_count") != 2
        or scope.get("gpu_counts") != [2]
        or scope.get("global_batch_sizes") != [64]
        or scope.get("gradient_checkpointing") != [False]
        or scope.get("zero_by_gpu_count") != {"2": ["zero2"]}
    ):
        raise ValueError("experiment config is not the exact stage-two six-GPU scope")

    hardware = _hardware_for_join_busy_pool()
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(queue, rows, ROOT)
    provenance_binding = build_provenance_binding(ROOT)

    bound_files = {
        queue,
        campaign_design,
        queue_manifest,
        bundle_path,
        preprocessing_path,
        ARTIFACT_DIR / "h800_lora_remote_candidate_audit_v1.json",
        ARTIFACT_DIR / "h800_lora_source_disjoint_bundle_v1.json",
        ARTIFACT_DIR / "h800_lora_source_disjoint_v1" / "split_manifest.json",
        ARTIFACT_DIR / "h800_profile_aware_memory_calibration_design_v1.json",
        ARTIFACT_DIR / "dataset_analysis.json",
        ARTIFACT_DIR / "h800_lora_safety_stage2_v1" / "processor_contract.json",
        ARTIFACT_DIR / "h800_lora_safety_stage2_v1" / "split_manifest.json",
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "data" / "dataset_info.json",
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "models.json",
    }
    for row in rows:
        bound_files.add(Path(str(row["data_path"])))
        bound_files.add(Path(str(row["dataset_profile_path"])))
    missing = [str(path) for path in bound_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files are absent: {missing}")
    manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=lambda value: str(value))
    }
    ids = [str(row["job_id"]) for row in rows]
    stage_binding = {
        "allowed_job_ids": ids,
        "queue_path": queue_binding["path"],
        "queue_sha256": queue_binding["sha256"],
        "ordered_job_ids": queue_binding["ordered_job_ids"],
        "ordered_job_payload_sha256": queue_binding["ordered_job_payload_sha256"],
        "job_payload_sha256": queue_binding["job_payload_sha256"],
    }
    approval_design = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}",
        "campaign_design": {
            "path": str(campaign_design.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(campaign_design),
        },
        "file_sha256": manifest,
        "execution_order": [PHASE_ID],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": list(AUTHORIZED_GPU_IDS),
        "max_gpu_count": 2,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": (
                "start only on idle pairs (0,1), (4,5), or (6,7); never use GPU 2,3; "
                "never terminate external processes"
            ),
        },
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": stage_binding,
    }
    candidate = ARTIFACT_DIR / "approval_design_h800_lora_safety_stage2_v1_candidate.json"
    write_json(candidate, approval_design)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--campaign-design", type=Path, default=DEFAULT_CAMPAIGN_DESIGN)
    parser.add_argument("--queue-manifest", type=Path, default=DEFAULT_QUEUE_MANIFEST)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--preprocessing", type=Path, default=DEFAULT_PREPROCESSING)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = freeze(
        args.queue.resolve(),
        args.campaign_design.resolve(),
        args.queue_manifest.resolve(),
        args.bundle.resolve(),
        args.preprocessing.resolve(),
    )
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
            {"candidate": str(candidate), "sha256": digest, "promotion": report},
            ensure_ascii=False,
            indent=2,
        )
    )
    if report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
