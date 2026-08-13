#!/usr/bin/env python3
"""Freeze and optionally promote one exact Qwen3.5/VL supplement stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, DATA_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch, validate_job
from prepare_h800_qwen35_vl_supplement_v1 import (
    CAMPAIGN_ID,
    CANARY_DESIGN,
    CANARY_MANIFEST,
    CANARY_PHASE_ID,
    CANARY_QUEUE,
    FORMAL_DESIGN,
    FORMAL_MANIFEST,
    FORMAL_PHASE_ID,
    FORMAL_QUEUE,
    FROZEN_SELECTION,
    INVENTORY,
    JOB_SCHEMA,
    MEDIA_MANIFEST,
    PROFILE_MANIFEST,
    RUNTIME_CONTRACT,
    STAGING_CONFIG,
    TEXT_PROFILE_MANIFEST,
    VL_DATA,
)
from prepare_h800_qwen35_vl_canary_projector_repair_v1 import (
    REPAIR_DESIGN,
    REPAIR_MANIFEST,
    REPAIR_QUEUE,
    SOURCE_ACCEPTANCE as REPAIR_SOURCE_ACCEPTANCE,
)
from validate_h800_qwen35_vl_supplement_v1 import validate as validate_campaign


AUTHORIZATION = (
    "用户在 2026-08-09 明确要求全面分析 Qwen3.5 与 VL 缺口、写好实验脚本、完整记录，"
    "并要求在当前 8 卡任务全部结束后自动开启新实验。本 approval 只授权 GPU 0-7 上已冻结的"
    "Qwen3.5/VL 补充队列，不允许抢占现有进程、扩展模型/数据/作业、启用 offload 或 packing。"
    "canary 只验证软件和真实图像语义，正式作业只用于标定；OOM 只能作为右删失下界，"
    "重复测量不得冒充独立样本，任何发表或业务放量仍需新的来源隔离前瞻验收。"
    "用户随后在 2026-08-09 明确说明已手动暂时中止上游任务，并要求先运行本批实验；"
    "该后续授权只允许在上游控制进程已退出且 GPU 0-7 通过原空闲门禁时，显式绕过"
    "上游 220/220 终态要求，不放宽其他硬件、队列、canary 或正式阶段门禁。"
    "用户在 2026-08-09 进一步明确要求按已分析步骤修复并真正启动实验，若无法运行必须即时说明。"
    "本授权因此允许用不变的 job ID 和 payload 只重跑两个 projector+language 语义失败 Canary；"
    "只有完整 15/15 Canary 最新结果通过，才允许开启原冻结的 86 个正式作业。"
)
AUTHORIZED_GPU_IDS = list(range(8))
MAX_GPU_COUNT = 4
CANARY_ACCEPTANCE = (
    ARTIFACT_DIR / "h800_qwen35_vl_supplement_canary_acceptance_v1.json"
)
STAGES = {
    "canary": {
        "phase_id": CANARY_PHASE_ID,
        "queue": CANARY_QUEUE,
        "design": CANARY_DESIGN,
        "manifest": CANARY_MANIFEST,
        "jobs": 15,
    },
    "formal": {
        "phase_id": FORMAL_PHASE_ID,
        "queue": FORMAL_QUEUE,
        "design": FORMAL_DESIGN,
        "manifest": FORMAL_MANIFEST,
        "jobs": 86,
    },
    "canary_repair": {
        "phase_id": CANARY_PHASE_ID,
        "queue": REPAIR_QUEUE,
        "design": REPAIR_DESIGN,
        "manifest": REPAIR_MANIFEST,
        "jobs": 1,
    },
}

SCOPED_SOURCE_RELATIVE_PATHS = (
    "config/experiment.json",
    "config/hardware.json",
    "config/models.json",
    "config/deepspeed/ds_z2.json",
    "config/deepspeed/ds_z3.json",
    "scripts/approval_gate.py",
    "scripts/common.py",
    "scripts/promote_approval_candidate.py",
    "scripts/prepare_h800_qwen35_vl_supplement_profiles_v1.py",
    "scripts/prepare_h800_qwen35_vl_supplement_v1.py",
    "scripts/prepare_h800_qwen35_vl_canary_projector_repair_v1.py",
    "scripts/validate_h800_qwen35_vl_supplement_v1.py",
    "scripts/freeze_h800_qwen35_vl_supplement_v1.py",
    "scripts/evaluate_h800_qwen35_vl_supplement_v1.py",
    "scripts/run_job.py",
    "scripts/scheduler.py",
    "scripts/train_entry.py",
    "scripts/metrics_callback.py",
    "scripts/model_structure_manifest.py",
    "scripts/runtime_evidence.py",
    "scripts/gpu_telemetry.py",
    "scripts/collect_results.py",
    "qwen35_vl_staging/launch_h800_qwen35_vl_after_unified_v1.py",
)


def _prerequisite(stage: str) -> dict[str, Any] | None:
    if stage == "canary":
        return None
    if stage == "canary_repair":
        report = read_json(REPAIR_SOURCE_ACCEPTANCE)
        if (
            report.get("schema")
            != "sft_h800_qwen35_vl_supplement_canary_acceptance/v1"
            or report.get("campaign_id") != CAMPAIGN_ID
            or report.get("all_passed") is not False
        ):
            raise RuntimeError("repair stage requires the bound failed Canary report")
        return {
            "path": str(
                REPAIR_SOURCE_ACCEPTANCE.resolve().relative_to(ROOT.resolve())
            ),
            "sha256": sha256_file(REPAIR_SOURCE_ACCEPTANCE),
            "report_sha256": report.get("report_sha256"),
            "all_passed": False,
            "repair_only": True,
        }
    if not CANARY_ACCEPTANCE.is_file():
        raise RuntimeError("formal stage remains locked: canary acceptance is absent")
    report = read_json(CANARY_ACCEPTANCE)
    if (
        report.get("schema")
        != "sft_h800_qwen35_vl_supplement_canary_acceptance/v1"
        or report.get("campaign_id") != CAMPAIGN_ID
        or report.get("all_passed") is not True
    ):
        raise RuntimeError(f"formal stage remains locked: canary failed: {report}")
    return {
        "path": str(CANARY_ACCEPTANCE.resolve().relative_to(ROOT.resolve())),
        "sha256": sha256_file(CANARY_ACCEPTANCE),
        "report_sha256": report.get("report_sha256"),
        "all_passed": True,
    }


def freeze(stage: str, *, manual_upstream_takeover: bool = False) -> Path:
    static = validate_campaign(allow_results=stage in {"formal", "canary_repair"})
    if static.get("all_passed") is not True:
        raise RuntimeError("static supplement validation failed")
    spec = STAGES[stage]
    queue_path = Path(spec["queue"])
    rows = read_jsonl(queue_path)
    design = read_json(Path(spec["design"]))
    manifest = read_json(Path(spec["manifest"]))
    if (
        len(rows) != int(spec["jobs"])
        or len({str(row.get("job_id") or "") for row in rows}) != len(rows)
        or any(row.get("schema") != JOB_SCHEMA for row in rows)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in rows)
        or any(row.get("phase_id") != spec["phase_id"] for row in rows)
    ):
        raise ValueError(f"{stage} queue identity/count drifted")
    for row in rows:
        validate_job(row)
    unsigned = dict(design)
    expected_design_hash = unsigned.pop("report_sha256", None)
    if (
        expected_design_hash != sha256_json(unsigned)
        or design.get("gpu_training_started") is not False
        or design.get("execution_authorized") is not False
        or design.get("queue", {}).get("sha256") != sha256_file(queue_path)
        or design.get("queue", {}).get("ordered_job_ids")
        != [str(row["job_id"]) for row in rows]
        or manifest.get("design", {}).get("sha256") != sha256_file(Path(spec["design"]))
        or manifest.get("queue", {}).get("sha256") != sha256_file(queue_path)
    ):
        raise ValueError(f"{stage} design/manifest drifted")
    live_config = ROOT / "config" / "experiment.json"
    if sha256_file(live_config) != sha256_file(STAGING_CONFIG):
        raise ValueError("live experiment config is not the exact supplement config")
    scope = read_json(live_config)["training_scope"]
    if (
        scope.get("gpu_ids") != AUTHORIZED_GPU_IDS
        or scope.get("exclusive_node_gpu_ids") != AUTHORIZED_GPU_IDS
        or scope.get("max_gpu_count") != MAX_GPU_COUNT
        or scope.get("gpu_counts") != [1, 2, 4]
    ):
        raise ValueError("live supplement GPU scope drifted")

    prerequisite = _prerequisite(stage)
    hardware = probe_hardware(required_gpu_ids=tuple(AUTHORIZED_GPU_IDS))
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"GPU 0-7 are not an exact H800 pool: {hardware}")
    if hardware.get("selected_gpu_compute_processes"):
        raise RuntimeError("refusing to freeze while GPU 0-7 have compute processes")
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")
    queue_binding = build_queue_binding(queue_path, rows, ROOT)
    scoped_paths = [ROOT / relative for relative in SCOPED_SOURCE_RELATIVE_PATHS]
    missing_sources = [str(path) for path in scoped_paths if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"approval source files are absent: {missing_sources}")
    provenance_binding = build_provenance_binding(ROOT, source_paths=scoped_paths)

    bound_files = {
        queue_path,
        Path(spec["design"]),
        Path(spec["manifest"]),
        INVENTORY,
        PROFILE_MANIFEST,
        TEXT_PROFILE_MANIFEST,
        MEDIA_MANIFEST,
        VL_DATA,
        RUNTIME_CONTRACT,
        FROZEN_SELECTION,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        DATA_DIR / "dataset_info.json",
        *scoped_paths,
    }
    for row in rows:
        bound_files.add(Path(row["data_path"]))
        bound_files.add(Path(row["dataset_profile_path"]))
        if row.get("media_manifest_path"):
            bound_files.add(Path(row["media_manifest_path"]))
        overlay = row.get("environment_overlay") or {}
        if overlay:
            bound_files.add(Path(overlay["contract_path"]))
    if prerequisite is not None:
        bound_files.add(
            REPAIR_SOURCE_ACCEPTANCE
            if stage == "canary_repair"
            else CANARY_ACCEPTANCE
        )
    missing = [str(path) for path in bound_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files are absent: {missing}")
    file_manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=str)
    }
    ids = [str(row["job_id"]) for row in rows]
    queue_stage = {
        "allowed_job_ids": ids,
        "queue_path": queue_binding["path"],
        "queue_sha256": queue_binding["sha256"],
        "ordered_job_ids": queue_binding["ordered_job_ids"],
        "ordered_job_payload_sha256": queue_binding[
            "ordered_job_payload_sha256"
        ],
        "job_payload_sha256": queue_binding["job_payload_sha256"],
    }
    approval_design: dict[str, Any] = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{spec['phase_id']}",
        "campaign_design": {
            "path": str(Path(spec["design"]).resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(Path(spec["design"])),
            "report_sha256": design["report_sha256"],
        },
        "stage": stage,
        "stage_prerequisite": prerequisite,
        "file_sha256": file_manifest,
        "execution_order": [str(spec["phase_id"])],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": AUTHORIZED_GPU_IDS,
        "max_gpu_count": MAX_GPU_COUNT,
        "scheduler_execution": {
            "join_busy_pool": False,
            "preemption_allowed": False,
            "manual_upstream_takeover": manual_upstream_takeover,
            "policy": (
                (
                    "the user explicitly stopped the incomplete upstream campaign; "
                    "start only after its controllers are gone and GPU 0-7 pass the "
                    "unchanged idle grace gate; "
                )
                if manual_upstream_takeover
                else "start only after the upstream 220-job campaign is terminal and GPU 0-7 pass the idle grace gate; "
            )
            + (
                "use disjoint 1/2-card masks and the repository-conservative "
                "pool-exclusive 4-card policy"
            ),
        },
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": queue_stage,
    }
    candidate = (
        ARTIFACT_DIR
        / f"approval_design_h800_qwen35_vl_supplement_{stage}_v1_candidate.json"
    )
    write_json(candidate, approval_design)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--manual-upstream-takeover", action="store_true")
    args = parser.parse_args()
    candidate = freeze(
        args.stage,
        manual_upstream_takeover=args.manual_upstream_takeover,
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
            {
                "candidate": str(candidate),
                "sha256": digest,
                "stage": args.stage,
                "promoted": bool(args.promote),
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
