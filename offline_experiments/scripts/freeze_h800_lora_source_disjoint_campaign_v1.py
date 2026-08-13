#!/usr/bin/env python3
"""Freeze and optionally promote the exact six-GPU, 77-job LoRA campaign."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch


AUTHORIZATION = (
    "用户明确要求使用当前空闲的六张 H800 GPU 0、1、4、5、6、7 运行 LoRA 显存补充实验，"
    "并指定从 infra-ai-infra-storage 的 datasets 目录选择新数据源；本次只允许执行已冻结的 "
    "15 个 source-disjoint 场景、77 个 A1-A4 作业，不得占用 GPU 2、3，不得终止或抢占外部"
    "进程，不得扩展到第二阶段或自动发布模型。"
)
CAMPAIGN_ID = "h800_lora_source_disjoint_recalibration_20260804_v1"
PHASE_ID = "h800_lora_source_disjoint_recalibration_v1"
JOB_SCHEMA = "sft_h800_lora_source_disjoint_job/v1"
AUTHORIZED_GPU_IDS = (0, 1, 4, 5, 6, 7)
DEFAULT_QUEUE = ROOT / "matrix" / "h800_lora_source_disjoint_jobs_v1.jsonl"
DEFAULT_CAMPAIGN_DESIGN = ARTIFACT_DIR / "h800_lora_source_disjoint_experiment_design_v1.json"
DEFAULT_QUEUE_MANIFEST = ARTIFACT_DIR / "h800_lora_source_disjoint_queue_manifest_v1.json"
DEFAULT_BUNDLE = ARTIFACT_DIR / "h800_lora_source_disjoint_bundle_v1.json"
DEFAULT_PREPROCESSING = ARTIFACT_DIR / "h800_lora_source_disjoint_preprocessing_validation_v1.json"
GPU_SNAPSHOT = ARTIFACT_DIR / "h800_lora_source_disjoint_gpu_snapshot_v1.csv"
COMPUTE_SNAPSHOT = ARTIFACT_DIR / "h800_lora_source_disjoint_compute_snapshot_v1.csv"


def _hardware_from_direct_nvidia_smi_snapshots() -> dict[str, object]:
    """Validate fresh outputs captured by separately approved nvidia-smi calls."""
    now = time.time()
    for path in (GPU_SNAPSHOT, COMPUTE_SNAPSHOT):
        if not path.is_file():
            raise FileNotFoundError(f"missing direct nvidia-smi snapshot: {path}")
        age = now - path.stat().st_mtime
        if age < 0 or age > 300:
            raise RuntimeError(f"stale direct nvidia-smi snapshot ({age:.1f}s): {path}")
    gpu_rows = []
    with GPU_SNAPSHOT.open(encoding="utf-8", newline="") as source:
        for row in csv.reader(source):
            if len(row) != 7:
                raise ValueError(f"malformed GPU snapshot row: {row}")
            gpu_rows.append(
                {
                    "index": int(row[0].strip()),
                    "uuid": row[1].strip(),
                    "name": row[2].strip(),
                    "memory_total_mib": int(row[3].strip()),
                    "memory_used_mib": int(row[4].strip()),
                    "utilization_gpu_percent": int(row[5].strip()),
                    "temperature_c": int(row[6].strip()),
                }
            )
    selected = [row for row in gpu_rows if row["index"] in AUTHORIZED_GPU_IDS]
    if (
        [row["index"] for row in selected] != list(AUTHORIZED_GPU_IDS)
        or any(row["name"] != "NVIDIA H800" for row in selected)
        or any(row["memory_total_mib"] < 140000 for row in selected)
        or any(row["memory_used_mib"] > 100 for row in selected)
        or any(row["utilization_gpu_percent"] != 0 for row in selected)
    ):
        raise RuntimeError(f"authorized GPU snapshot is not six idle H800 cards: {selected}")
    compute_rows = []
    with COMPUTE_SNAPSHOT.open(encoding="utf-8", newline="") as source:
        for row in csv.reader(source):
            if not row:
                continue
            if len(row) != 4:
                raise ValueError(f"malformed compute snapshot row: {row}")
            compute_rows.append(
                {
                    "gpu_uuid": row[0].strip(),
                    "pid": int(row[1].strip()),
                    "process_name": row[2].strip(),
                    "used_memory_mib": int(row[3].strip()),
                }
            )
    selected_uuids = {row["uuid"] for row in selected}
    selected_processes = [row for row in compute_rows if row["gpu_uuid"] in selected_uuids]
    if selected_processes:
        raise RuntimeError(f"authorized GPUs have compute processes: {selected_processes}")
    return {
        "required_gpu_ids": list(AUTHORIZED_GPU_IDS),
        "expected_gpu_name": "H800",
        "capture_method": "direct separately approved nvidia-smi output",
        "gpu_snapshot": {
            "path": str(GPU_SNAPSHOT.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(GPU_SNAPSHOT),
            "age_seconds": now - GPU_SNAPSHOT.stat().st_mtime,
        },
        "compute_snapshot": {
            "path": str(COMPUTE_SNAPSHOT.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(COMPUTE_SNAPSHOT),
            "age_seconds": now - COMPUTE_SNAPSHOT.stat().st_mtime,
        },
        "selected_gpu_rows": selected,
        "selected_gpu_compute_processes": selected_processes,
        "exact_h800_pool": True,
        "selected_pool_idle": True,
    }


def freeze(
    queue: Path,
    campaign_design: Path,
    queue_manifest: Path,
    bundle_path: Path,
    preprocessing_path: Path,
) -> Path:
    rows = read_jsonl(queue)
    groups = Counter(str(row.get("experiment_group")) for row in rows)
    if (
        len(rows) != 77
        or len({str(row.get("job_id")) for row in rows}) != 77
        or dict(groups) != {"A1": 45, "A2": 16, "A4": 8, "A3": 8}
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
    ):
        raise ValueError("queue is not the exact 77-job source-disjoint campaign")
    if sum(int(row["gpu_count"]) for row in rows) != 170:
        raise ValueError("queue must contain exactly 170 GPU-job equivalents")
    partitions = [row.get("calibration_partition") or {} for row in rows]
    if any(
        part.get("role") != "calibration"
        or part.get("policy") != "remote_source_dataset_id_disjoint_v1"
        or not part.get("split_unit_id")
        for part in partitions
    ):
        raise ValueError("every queue row must carry the source-disjoint partition")
    if len({part["split_unit_id"] for part in partitions}) != 15:
        raise ValueError("queue must contain exactly 15 independent split-unit IDs")

    design_manifest = read_json(campaign_design)
    queue_meta = read_json(queue_manifest)
    bundle = read_json(bundle_path)
    preprocessing = read_json(preprocessing_path)
    if (
        design_manifest.get("schema") != "sft_h800_lora_source_disjoint_experiment_design/v1"
        or design_manifest.get("campaign_id") != CAMPAIGN_ID
        or design_manifest.get("ordered_job_ids") != [row["job_id"] for row in rows]
        or design_manifest.get("ordered_job_payload_sha256") != sha256_json(rows)
        or queue_meta.get("queue", {}).get("sha256") != sha256_file(queue)
        or queue_meta.get("design", {}).get("sha256") != sha256_file(campaign_design)
        or queue_meta.get("queue", {}).get("ordered_job_payload_sha256") != sha256_json(rows)
    ):
        raise ValueError("campaign design or queue manifest binding drifted")
    if (
        bundle.get("counts", {}).get("scenarios") != 15
        or bundle.get("gpu_training_started") is not False
        or design_manifest.get("source_bundle", {}).get("sha256") != sha256_file(bundle_path)
    ):
        raise ValueError("15-source frozen bundle binding drifted")
    if (
        preprocessing.get("all_passed") is not True
        or len(preprocessing.get("checks") or []) != 15
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
        or scope.get("max_gpu_count") != 4
        or scope.get("gpu_counts") != [2, 4]
        or scope.get("global_batch_sizes") != [64]
    ):
        raise ValueError("experiment config is not the exact six-GPU 77-job scope")

    hardware = probe_hardware(required_gpu_ids=AUTHORIZED_GPU_IDS)
    if hardware.get("exact_h800_pool") is not True:
        hardware = _hardware_from_direct_nvidia_smi_snapshots()
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"GPU pool is not the exact six-card H800 pool: {hardware}")
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
        ARTIFACT_DIR / "h800_lora_source_disjoint_v1" / "processor_contract.json",
        ARTIFACT_DIR / "h800_lora_source_disjoint_v1" / "split_manifest.json",
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "data" / "dataset_info.json",
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "models.json",
        GPU_SNAPSHOT,
        COMPUTE_SNAPSHOT,
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
        "max_gpu_count": 4,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": (
                "start only on idle members of GPU 0,1,4,5,6,7; never use GPU 2,3; "
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
    candidate = ARTIFACT_DIR / "approval_design_h800_lora_source_disjoint_six_gpu_v1_candidate.json"
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
