#!/usr/bin/env python3
"""Freeze and optionally promote the exact VL-only canary continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch


AUTHORIZATION = (
    "用户在 2026-08-03 要求按顺序继续推进 Packing 与 VL，并限定 H800 GPU 0、1。"
    "Packing 两个 job 已成功；v1 只因一个未消费 collator prefetch 计数失败，v2 在不改阈值、"
    "模型或队列的情况下按冻结 sampler 修正。本 approval 仅允许执行两个 VL real-image canary。"
)
CAMPAIGN_ID = "h800_packing_vl_canary_20260803_v1"
PHASE_ID = "h800_packing_vl_canary_v1"
JOB_SCHEMA = "sft_h800_packing_vl_canary_job/v1"
DESIGN = ARTIFACT_DIR / "h800_packing_vl_canary_continuation_design_v2.json"
CORRECTION = ARTIFACT_DIR / "h800_packing_semantic_canary_acceptance_v2_prefetch_corrected.json"
QUEUE = ROOT / "matrix" / "h800_vl_media_canary_v1.jsonl"
INVENTORY = ARTIFACT_DIR / "h800_packing_vl_canary_model_inventory_v1.json"


def freeze() -> Path:
    rows = read_jsonl(QUEUE)
    design = read_json(DESIGN)
    correction = read_json(CORRECTION)
    if (
        len(rows) != 2
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
        or any(int(row.get("gpu_count", 0)) != 2 for row in rows)
        or any(row.get("candidate_role") != "vl_real_image_software_media_canary" for row in rows)
        or any(row.get("software_canary") is not True for row in rows)
        or any(Path(row.get("declared_model_manifest_path", "")).resolve() != INVENTORY.resolve() for row in rows)
        or any(row.get("declared_model_manifest_sha256") != sha256_file(INVENTORY) for row in rows)
        or design.get("schema") != "sft_h800_packing_vl_canary_continuation_design/v2"
        or design.get("campaign_id") != CAMPAIGN_ID
        or design.get("packing_gpu_training_started") is not True
        or design.get("vl_gpu_training_started") is not False
        or design.get("remaining_execution_order") != ["vl"]
        or design.get("vl_queue", {}).get("sha256") != sha256_file(QUEUE)
        or correction.get("all_passed") is not True
        or design.get("v2_prefetch_correction", {}).get("sha256") != sha256_file(CORRECTION)
    ):
        raise ValueError("VL continuation is not the exact frozen v2 transition")
    scope = read_json(ROOT / "config" / "experiment.json")["training_scope"]
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("model_ids") != ["qwen3_8b", "qwen2p5_vl_7b", "qwen3_vl_8b"]
        or scope.get("gpu_ids") != [0, 1]
        or scope.get("exclusive_node_gpu_ids") != [0, 1]
        or scope.get("max_gpu_count") != 2
    ):
        raise ValueError("experiment scope drifted from the GPU-0,1 continuation")
    hardware = probe_hardware(required_gpu_ids=(0, 1))
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"GPU 0,1 are not an exact H800 pool: {hardware}")
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(QUEUE, rows, ROOT)
    provenance_binding = build_provenance_binding(ROOT)
    bound_files = {
        QUEUE,
        DESIGN,
        CORRECTION,
        INVENTORY,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "data" / "dataset_info.json",
    }
    for binding in design["source_bindings"].values():
        bound_files.add(Path(binding["path"]))
    for packing in design["packing_results"]:
        bound_files.add(Path(packing["status"]["path"]))
    for row in rows:
        bound_files.add(Path(row["data_path"]))
        bound_files.add(Path(row["dataset_profile_path"]))
    file_manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=lambda value: str(value))
    }
    ids = [row["job_id"] for row in rows]
    queue_stage = {
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
        "design_purpose": f"{CAMPAIGN_ID}:vl_continuation_v2",
        "campaign_design": {
            "path": str(DESIGN.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(DESIGN),
            "report_sha256": design["report_sha256"],
        },
        "stage": "vl",
        "stage_prerequisite": {
            "path": str(CORRECTION.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(CORRECTION),
            "report_sha256": correction["report_sha256"],
            "all_passed": True,
        },
        "file_sha256": file_manifest,
        "execution_order": [f"{PHASE_ID}:vl_continuation_v2"],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": [0, 1],
        "max_gpu_count": 2,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": "run the two approved 2-GPU VL canaries sequentially on idle GPU 0,1",
        },
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": queue_stage,
    }
    candidate = ARTIFACT_DIR / "approval_design_h800_packing_vl_continuation_v2_candidate.json"
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
