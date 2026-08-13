#!/usr/bin/env python3
"""Freeze and optionally promote one exact Packing/VL canary stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, RESULTS_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch


AUTHORIZATION = (
    "用户在 2026-08-03 明确要求继续按冻结顺序推进，并沿用只使用 H800 GPU 0、1 的范围。"
    "本 approval 仅允许先执行 Packing U/P semantic canary，再在 Packing 机器验收通过后以新 approval "
    "执行两代 VL real-image media canary；不得占用其他 GPU、抢占外部进程、拟合模型或扩展队列。"
)
CAMPAIGN_ID = "h800_packing_vl_canary_20260803_v1"
PHASE_ID = "h800_packing_vl_canary_v1"
JOB_SCHEMA = "sft_h800_packing_vl_canary_job/v1"
DESIGN = ARTIFACT_DIR / "h800_packing_vl_canary_design_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_vl_canary_queue_manifest_v1.json"
DECISIONS = ARTIFACT_DIR / "h800_packing_vl_canary_frozen_decisions_v1.json"
INVENTORY = ARTIFACT_DIR / "h800_packing_vl_canary_model_inventory_v1.json"
PACKING_ACCEPTANCE = ARTIFACT_DIR / "h800_packing_semantic_canary_acceptance_v1.json"
STAGES = {
    "packing": {
        "queue": ROOT / "matrix" / "h800_packing_semantic_canary_v1.jsonl",
        "gpu_count": 1,
        "candidate_role_prefix": "packing_semantic_",
    },
    "vl": {
        "queue": ROOT / "matrix" / "h800_vl_media_canary_v1.jsonl",
        "gpu_count": 2,
        "candidate_role_prefix": "vl_real_image_",
    },
}


def _packing_prerequisite() -> dict[str, Any]:
    if not PACKING_ACCEPTANCE.is_file():
        raise RuntimeError("VL stage remains locked: Packing acceptance report is absent")
    report = read_json(PACKING_ACCEPTANCE)
    if (
        report.get("schema") != "sft_h800_packing_semantic_canary_acceptance/v1"
        or report.get("campaign_id") != CAMPAIGN_ID
        or report.get("all_passed") is not True
    ):
        raise RuntimeError(f"VL stage remains locked: Packing canary did not pass: {report}")
    return {
        "path": str(PACKING_ACCEPTANCE.resolve().relative_to(ROOT.resolve())),
        "sha256": sha256_file(PACKING_ACCEPTANCE),
        "report_sha256": report.get("report_sha256"),
        "all_passed": True,
    }


def freeze(stage_name: str) -> Path:
    spec = STAGES[stage_name]
    queue_path = Path(spec["queue"])
    rows = read_jsonl(queue_path)
    if (
        len(rows) != 2
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != PHASE_ID for row in rows)
        or any(int(row.get("gpu_count", 0)) != int(spec["gpu_count"]) for row in rows)
        or any(not str(row.get("candidate_role") or "").startswith(str(spec["candidate_role_prefix"])) for row in rows)
        or any(row.get("software_canary") is not True for row in rows)
        or any((row.get("calibration_partition") or {}).get("role") != "canary_excluded" for row in rows)
        or any(Path(row.get("declared_model_manifest_path", "")).resolve() != INVENTORY.resolve() for row in rows)
        or any(row.get("declared_model_manifest_sha256") != sha256_file(INVENTORY) for row in rows)
    ):
        raise ValueError(f"queue is not the exact frozen {stage_name} canary")

    design = read_json(DESIGN)
    queue_manifest = read_json(QUEUE_MANIFEST)
    decisions = read_json(DECISIONS)
    staged_design = (design.get("stages") or {}).get(stage_name) or {}
    staged_manifest = (queue_manifest.get("staged_queues") or {}).get(stage_name) or {}
    if (
        design.get("schema") != "sft_h800_packing_vl_canary_design/v1"
        or design.get("campaign_id") != CAMPAIGN_ID
        or design.get("phase_id") != PHASE_ID
        or design.get("gpu_training_started") is not False
        or design.get("execution_order") != ["packing", "vl"]
        or design.get("frozen_decisions", {}).get("sha256") != sha256_file(DECISIONS)
        or decisions.get("generated_before_gpu") is not True
        or queue_manifest.get("design", {}).get("sha256") != sha256_file(DESIGN)
        or staged_design.get("queue", {}).get("sha256") != sha256_file(queue_path)
        or staged_manifest.get("sha256") != sha256_file(queue_path)
        or staged_manifest.get("job_count") != 2
    ):
        raise ValueError("Packing/VL design, decisions, or staged queue drifted")

    experiment = read_json(ROOT / "config" / "experiment.json")
    scope = experiment["training_scope"]
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("model_ids") != ["qwen3_8b", "qwen2p5_vl_7b", "qwen3_vl_8b"]
        or scope.get("gpu_ids") != [0, 1]
        or scope.get("exclusive_node_gpu_ids") != [0, 1]
        or scope.get("max_gpu_count") != 2
        or scope.get("gpu_counts") != [1, 2]
    ):
        raise ValueError("experiment config is not the exact GPU-0,1 Packing/VL canary scope")

    prerequisite = _packing_prerequisite() if stage_name == "vl" else None
    hardware = probe_hardware(required_gpu_ids=(0, 1))
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"GPU 0,1 are not an exact H800 pool: {hardware}")
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(queue_path, rows, ROOT)
    provenance_binding = build_provenance_binding(ROOT)

    bound_files = {
        queue_path,
        DESIGN,
        QUEUE_MANIFEST,
        DECISIONS,
        INVENTORY,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "data" / "dataset_info.json",
    }
    for binding in (design.get("source_bindings") or {}).values():
        bound_files.add(Path(binding["path"]))
    for row in rows:
        bound_files.add(Path(row["data_path"]))
        bound_files.add(Path(row["dataset_profile_path"]))
    if prerequisite is not None:
        bound_files.add(PACKING_ACCEPTANCE)
    missing = [str(path) for path in bound_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files are absent: {missing}")
    file_manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=lambda value: str(value))
    }
    ids = [str(row["job_id"]) for row in rows]
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
        "design_purpose": f"{CAMPAIGN_ID}:{stage_name}",
        "campaign_design": {
            "path": str(DESIGN.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(DESIGN),
            "report_sha256": design["report_sha256"],
        },
        "frozen_decisions": {
            "path": str(DECISIONS.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(DECISIONS),
            "report_sha256": decisions["report_sha256"],
            "generated_before_gpu": True,
        },
        "stage": stage_name,
        "stage_prerequisite": prerequisite,
        "file_sha256": file_manifest,
        "execution_order": [f"{PHASE_ID}:{stage_name}"],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": [0, 1],
        "max_gpu_count": 2,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": "use only idle GPU 0,1; Packing runs on disjoint single-card masks, VL runs sequentially on the pair",
        },
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": queue_stage,
    }
    candidate = ARTIFACT_DIR / f"approval_design_h800_packing_vl_canary_{stage_name}_v1_candidate.json"
    write_json(candidate, approval)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = freeze(args.stage)
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
