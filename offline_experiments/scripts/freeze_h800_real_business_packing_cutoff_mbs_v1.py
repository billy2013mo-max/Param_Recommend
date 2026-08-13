#!/usr/bin/env python3
"""Freeze and promote the exact 40-job real-business interaction campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch


AUTHORIZATION = (
    "用户在 2026-08-04 明确要求把现在空闲的卡都用上，跑这40个实验，然后给出结论。"
    "本 approval 仅允许在当时空闲的 H800 GPU 0、1、4、5、6、7 上执行冻结的40项"
    "真实业务 Packing×cutoff×MBS 单卡实验；明确排除正被外部推理占用的 GPU 2、3，"
    "不得抢占外部任务、自动增加任务、把校准结果当成最终泛化验收或自动发布模型。"
)
CAMPAIGN_ID = "h800_real_business_packing_cutoff_mbs_20260804_v1"
PHASE_ID = "h800_real_business_packing_cutoff_mbs_v1"
JOB_SCHEMA = "sft_h800_real_business_packing_cutoff_mbs_job/v1"
GPU_IDS = [0, 1, 4, 5, 6, 7]
QUEUE = ROOT / "matrix" / "h800_real_business_packing_cutoff_mbs_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_design_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_queue_manifest_v1.json"
STATIC_FEATURES = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_static_features_v1.json"
INVENTORY = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_model_inventory_v1.json"


def freeze() -> Path:
    rows = read_jsonl(QUEUE)
    ids = [str(row.get("job_id") or "") for row in rows]
    if (
        len(rows) != 40
        or len(set(ids)) != 40
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
        or any(int(row.get("gpu_count", 0)) != 1 for row in rows)
        or any(row.get("parallel_class") != "gpu_partitionable" for row in rows)
        or any(row.get("requires_external_node_idle") is not False for row in rows)
        or any(row.get("publication_allowed") is not False for row in rows)
        or [row.get("execution_sequence_index") for row in rows] != list(range(40))
    ):
        raise ValueError("queue is not the exact frozen 40-job interaction matrix")
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["family_id"]), str(row["arm_id"]))
        counts[key] = counts.get(key, 0) + 1
    if len(counts) != 20 or any(value != 2 for value in counts.values()):
        raise ValueError(f"the 4×5×2 matrix is unbalanced: {counts}")
    if any(float(row["expected_sample_gbs_relative_error"]) > 0.05 for row in rows):
        raise ValueError("a job exceeds the frozen 5% logical-sample GBS tolerance")

    experiment = read_json(ROOT / "config" / "experiment.json")
    scope = experiment["training_scope"]
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("model_ids") != ["qwen3_8b"]
        or scope.get("gpu_ids") != GPU_IDS
        or scope.get("exclusive_node_gpu_ids") != GPU_IDS
        or scope.get("max_gpu_count") != 1
        or scope.get("gpu_counts") != [1]
        or experiment.get("measurement", {}).get("performance_parallelism")
        != "disjoint_gpu_masks"
    ):
        raise ValueError("experiment config is not the exact six-GPU single-card scope")

    design = read_json(DESIGN)
    queue_manifest = read_json(QUEUE_MANIFEST)
    static = read_json(STATIC_FEATURES)
    if (
        design.get("schema") != "sft_h800_real_business_packing_cutoff_mbs_design/v1"
        or design.get("campaign_id") != CAMPAIGN_ID
        or design.get("gpu_training_started") is not False
        or design.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or design.get("queue", {}).get("ordered_job_ids") != ids
        or queue_manifest.get("design", {}).get("sha256") != sha256_file(DESIGN)
        or queue_manifest.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or static.get("generated_before_gpu") is not True
        or len(static.get("rows") or []) != 8
    ):
        raise ValueError("design, static gates or queue manifest drifted")

    hardware = probe_hardware(required_gpu_ids=GPU_IDS)
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"selected GPUs are not an exact H800 pool: {hardware}")
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
        STATIC_FEATURES,
        INVENTORY,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ARTIFACT_DIR / "dataset_analysis.json",
        ROOT / "data" / "dataset_info.json",
    }
    source_bindings = list((design.get("source_bindings") or {}).values())
    for binding in source_bindings:
        bound_files.add(Path(binding["path"]))
    for row in rows:
        bound_files.add(Path(row["data_path"]))
        bound_files.add(Path(row["dataset_profile_path"]))
    missing = [str(path) for path in bound_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files are absent: {missing}")
    mismatched_sources = [
        str(binding["path"])
        for binding in source_bindings
        if sha256_file(Path(binding["path"])) != binding.get("sha256")
    ]
    if mismatched_sources:
        raise ValueError(
            f"campaign source bindings drifted: {mismatched_sources}"
        )
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
    approval = {
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
        "max_gpu_count": 1,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": (
                "run up to six disjoint single-card jobs; wait rather than preempt if a selected "
                "GPU becomes externally occupied; GPU 2 and 3 are outside scope"
            ),
        },
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": stage,
    }
    candidate = ARTIFACT_DIR / "approval_design_h800_real_business_packing_cutoff_mbs_v1_candidate.json"
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
            {"candidate": str(candidate), "sha256": digest, "promotion": report},
            ensure_ascii=False,
            indent=2,
        )
    )
    if report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
