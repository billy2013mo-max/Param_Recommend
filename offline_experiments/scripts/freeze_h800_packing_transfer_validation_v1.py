#!/usr/bin/env python3
"""Freeze and optionally promote the 24-job Packing transfer approval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from approval_gate import build_provenance_binding, build_queue_binding
from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_packing_transfer_validation_v1 import (
    CAMPAIGN_ID,
    DESIGN,
    EXPERIMENT,
    FROZEN_PREDICTIONS,
    GPU_IDS,
    MODEL_INVENTORY,
    PACKING_RANKING_ACCEPTANCE,
    PHASE_ID,
    QUEUE,
)
from prepare_h800_prospective_holdout import probe_hardware
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch, validate_job


AUTHORIZATION = (
    "用户在 2026-08-17 明确要求继续下一步，审批并启动当前已冻结纯文本 Packing 排序规则的"
    "小规模跨场景验证。范围固定为 H800、5 个稠密 Qwen3 模型、6 个模型/训练/数据场景、"
    "1/2/4 卡、24 个事前冻结任务；不支持 VL+Packing，不抢占其它任务，不扩展任务，"
    "验证结果不得回填调参，OOM 必须保留为失败证据。"
)
CANDIDATE = ARTIFACT_DIR / "approval_design_h800_packing_transfer_validation_v1_candidate.json"
SCRIPTS = ROOT / "scripts"
STAGING = ROOT / "packing_config_ranking_staging"


def _validate(rows: list[dict[str, Any]]) -> None:
    if (
        len(rows) != 24
        or len({str(row.get("job_id")) for row in rows}) != 24
        or [int(row.get("execution_sequence_index", -1)) for row in rows] != list(range(24))
    ):
        raise ValueError("queue is not the exact ordered 24-job design")
    for row in rows:
        validate_job(row)
        if (
            row.get("campaign_id") != CAMPAIGN_ID
            or row.get("phase_id") != PHASE_ID
            or row.get("split_role") != "prospective_holdout"
            or row.get("packing") is not True
            or row.get("publication_allowed") is not False
            or int(row.get("gpu_count", 0)) not in {1, 2, 4}
        ):
            raise ValueError(f"job left the approved validation scope: {row.get('job_id')}")
        for path_key, digest_key in (("data_path", "data_sha256"), ("dataset_profile_path", "dataset_profile_sha256")):
            path = Path(str(row[path_key]))
            if not path.is_file() or sha256_file(path) != row[digest_key]:
                raise ValueError(f"job data binding drifted: {row['job_id']}:{path_key}")


def freeze(*, authorization: str) -> Path:
    rows = read_jsonl(QUEUE)
    _validate(rows)
    design = read_json(DESIGN)
    predictions = read_json(FROZEN_PREDICTIONS)
    if (
        design.get("campaign_id") != CAMPAIGN_ID
        or design.get("phase_id") != PHASE_ID
        or design.get("gpu_training_started") is not False
        or design.get("bindings", {}).get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or design.get("bindings", {}).get("experiment", {}).get("sha256") != sha256_file(EXPERIMENT)
    ):
        raise ValueError("campaign design bindings drifted")
    if (
        predictions.get("schema") != "sft_h800_packing_transfer_validation_frozen_predictions/v1"
        or predictions.get("validation_outcomes_read") is not False
        or predictions.get("coefficients_fitted") is not False
        or predictions.get("queue_binding", {}).get("sha256") != sha256_file(QUEUE)
        or len(predictions.get("predictions") or []) != 24
    ):
        raise ValueError("prospective frozen predictions are absent or drifted")
    existing = [
        str(row["job_id"])
        for row in rows
        if (ROOT / "results" / str(row["job_id"]) / "latest_attempt.json").is_file()
    ]
    if existing:
        raise ValueError(f"results already exist before approval: {existing[:5]}")

    live = read_json(ROOT / "config" / "experiment.json")
    scope = live.get("training_scope") or {}
    if (
        scope.get("phase_id") != PHASE_ID
        or scope.get("gpu_ids") != list(GPU_IDS)
        or scope.get("exclusive_node_gpu_ids") != list(GPU_IDS)
        or scope.get("gpu_counts") != [1, 2, 4]
        or int(scope.get("max_gpu_count", 0)) != 4
        or live.get("measurement", {}).get("performance_parallelism") != "disjoint_gpu_masks"
    ):
        raise ValueError("live experiment config is not the exact transfer-validation scope")

    hardware = probe_hardware(required_gpu_ids=list(GPU_IDS))
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError("GPU 0-7 are not the expected H800 pool")
    if hardware.get("selected_gpu_compute_processes"):
        raise RuntimeError("refusing to approve while GPU 0-7 are busy")
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")

    queue_binding = build_queue_binding(QUEUE, rows, ROOT)
    scoped_sources = [
        SCRIPTS / name
        for name in (
            "approval_gate.py",
            "common.py",
            "metrics_callback.py",
            "promote_approval_candidate.py",
            "run_job.py",
            "runtime_evidence.py",
            "scheduler.py",
            "train_entry.py",
            "prepare_h800_packing_transfer_validation_v1.py",
            "freeze_h800_packing_transfer_validation_v1.py",
            "evaluate_h800_packing_transfer_validation_v1.py",
        )
    ]
    launcher = STAGING / "launch_h800_packing_transfer_validation_v1.py"
    scoped_sources.append(launcher)
    missing = [str(path) for path in scoped_sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"approval source files are absent: {missing}")
    provenance_binding = build_provenance_binding(ROOT, source_paths=scoped_sources)

    bound_files: set[Path] = {
        QUEUE,
        DESIGN,
        EXPERIMENT,
        PACKING_RANKING_ACCEPTANCE,
        FROZEN_PREDICTIONS,
        MODEL_INVENTORY,
        ROOT / "artifacts" / "provenance.json",
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "hardware.json",
        ROOT / "config" / "models.json",
        *scoped_sources,
    }
    for row in rows:
        bound_files.add(Path(str(row["data_path"])))
        bound_files.add(Path(str(row["dataset_profile_path"])))
    missing = [str(path) for path in bound_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files are absent: {missing}")
    manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=str)
        if path.resolve().is_relative_to(ROOT.resolve())
    }
    approval_design = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:prospective_transfer_validation",
        "authorization": authorization,
        "campaign_design": {"path": str(DESIGN.resolve().relative_to(ROOT.resolve())), "sha256": sha256_file(DESIGN), "report_sha256": design["report_sha256"]},
        "frozen_predictions": {"path": str(FROZEN_PREDICTIONS.resolve().relative_to(ROOT.resolve())), "sha256": sha256_file(FROZEN_PREDICTIONS), "report_sha256": predictions["report_sha256"]},
        "file_sha256": manifest,
        "execution_order": [PHASE_ID],
        "allowed_job_ids": queue_binding["ordered_job_ids"],
        "authorized_gpu_ids": list(GPU_IDS),
        "max_gpu_count": 4,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": "strict 1/2/4-card homogeneous full-node waves with capacities 8/4/2",
        },
        "oom_policy": "record CUDA OOM as failed transfer evidence; no automatic retry",
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": {
            "allowed_job_ids": queue_binding["ordered_job_ids"],
            "queue_path": queue_binding["path"],
            "queue_sha256": queue_binding["sha256"],
            "ordered_job_ids": queue_binding["ordered_job_ids"],
            "ordered_job_payload_sha256": queue_binding["ordered_job_payload_sha256"],
            "job_payload_sha256": queue_binding["job_payload_sha256"],
        },
    }
    write_json(CANDIDATE, approval_design)
    return CANDIDATE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", default=AUTHORIZATION)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = freeze(authorization=args.authorization)
    digest = sha256_file(candidate)
    promotion = promote_candidate(
        candidate_path=candidate,
        expected_candidate_sha256=digest,
        project_root=ROOT,
        authorization=args.authorization,
        approved_by="user",
        promote=args.promote,
    )
    print(json.dumps({"candidate": str(candidate.resolve()), "sha256": digest, "jobs": 24, "promotion": promotion}, ensure_ascii=False, indent=2))
    if promotion.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
