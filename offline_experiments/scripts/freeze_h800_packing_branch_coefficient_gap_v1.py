#!/usr/bin/env python3
"""Freeze and optionally promote the four-job packed coefficient-gap retry (batch 2).

Without ``--promote`` this is a read-only dry run: it validates the design, the
queue and the live runtime, writes an approval *candidate*, and reports what a
promotion would change.  Only ``--promote`` performs the atomic approval swap.

Batch 1 closed the single-arm W5 profile but its two ZeRO-3/GC-off settings
OOMed at cutoff 20480; this batch retries that cell at cutoff 12288, sized
against the model centre rather than the analytic reference.  CUDA OOM is a right-censored
lower bound, never a regression label; a software failure must be repaired and
re-run under the same job id and payload rather than re-materialized.
"""

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
    "用户于2026-08-08查看第一批（G1至G3）结果后，选择方案A并同意继续在GPU 0至3上跑第二批。"
    "第一批的ZeRO-3/GC-off两设置在cutoff 20480下全部OOM——原因是当时按解析值110 GiB定容量，"
    "而模型真正的中心预测是160.3 GiB，按实测比值0.862推得约138 GiB，超过132.84安全线。"
    "本批把cutoff降到12288（中心预测123.0 GiB×0.862≈106 GiB）重试，"
    "仅允许在H800 GPU 0至3上执行冻结的4个双卡packed作业（2设置×2重复）；"
    "目的是补齐ZeRO-3/GC-off空缺机制格，使packed分支系数可被报告。"
    "不得抢占任何外部任务；CUDA OOM仅作为右删失显存下界，不作为回归标签；"
    "软件故障必须以同一job_id与payload原地重跑；"
    "本批不得自动发布、不得开启Packing自动推荐、不得据此宣称验收通过。"
)

CAMPAIGN_ID = "h800_packing_branch_coefficient_gap_b2_20260808_v1"
PHASE_ID = "h800_packing_branch_coefficient_gap_b2_v1"
JOB_SCHEMA = "sft_h800_packing_branch_coefficient_gap_job/v2"
DESIGN_SCHEMA = "sft_h800_packing_branch_coefficient_gap_design/v2"
GPU_IDS = [0, 1, 2, 3]
TWO_GPU_MASKS = [[0, 1], [2, 3]]
MAX_GPU_COUNT = 2
EXPECTED_JOBS = 4

QUEUE = ROOT / "matrix" / "h800_packing_branch_coefficient_gap_b2_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_packing_branch_coefficient_gap_b2_design_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_branch_coefficient_gap_b2_queue_manifest_v1.json"
CHALLENGER = (
    ROOT
    / "diagnostics"
    / "packing_branch_specific_memory_centre_20260807"
    / "packing_branch_specific_memory_centre_challenger_v1.json"
)
CANDIDATE = ARTIFACT_DIR / "approval_design_h800_packing_branch_coefficient_gap_b2_v1_candidate.json"

# Every setting is pinned here as well as in the design.  A drift between the two
# is a hard error: it means the queue was regenerated after review.
EXPECTED_SETTINGS: dict[str, tuple[str, str, int, str, bool, bool, int]] = {
    # setting_id: (model, workload, cutoff, zero, gc, packing, repeats)
    "G4_8B_LoRA_W8_Z3_GCoff-c12288": ("qwen3_8b", "W8", 12_288, "zero3", False, True, 2),
    "G5_8B_LoRA_W3_Z3_GCoff-c12288": ("qwen3_8b", "W3", 12_288, "zero3", False, True, 2),
}


def _validate_queue(rows: list[dict[str, Any]]) -> list[str]:
    """Check every job against the pinned settings and return the ordered ids."""
    if len(rows) != EXPECTED_JOBS:
        raise ValueError(f"queue has {len(rows)} jobs, expected {EXPECTED_JOBS}")
    seen: dict[str, int] = {}
    for row in rows:
        setting = str(row["setting_id"])
        if setting not in EXPECTED_SETTINGS:
            raise ValueError(f"unexpected setting in queue: {setting}")
        model, workload, cutoff, zero, gc, packing, _ = EXPECTED_SETTINGS[setting]
        actual = (
            str(row["model_id"]),
            str(row["workload_id"]),
            int(row["cutoff_len"]),
            str(row["zero"]),
            bool(row["gc"]),
            bool(row["packing"]),
        )
        if actual != (model, workload, cutoff, zero, gc, packing):
            raise ValueError(f"{setting}: queue row drifted from the pinned setting: {actual}")
        if str(row["schema"]) != JOB_SCHEMA:
            raise ValueError(f"{setting}: unexpected job schema")
        if int(row["gpu_count"]) != MAX_GPU_COUNT or int(row["mbs"]) != 1:
            raise ValueError(f"{setting}: gpu_count/mbs are not the frozen two-GPU mbs=1 shape")
        if row.get("neat_packing") is not True:
            raise ValueError(f"{setting}: packed arms must set neat_packing")
        if row.get("publication_allowed") is not False:
            raise ValueError(f"{setting}: publication_allowed must be false")
        if row.get("automatic_packing_recommendation_allowed") is not False:
            raise ValueError(f"{setting}: automatic recommendation must stay disabled")
        if str(row.get("oom_role")) != "right_censored_lower_bound":
            raise ValueError(f"{setting}: OOM must be recorded as a right-censored bound")
        seen[setting] = seen.get(setting, 0) + 1
    for setting, (*_, repeats) in EXPECTED_SETTINGS.items():
        if seen.get(setting) != repeats:
            raise ValueError(f"{setting}: expected {repeats} repeats, found {seen.get(setting)}")
    ids = [str(row["job_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate job ids in queue")
    return ids


def freeze() -> Path:
    rows = read_jsonl(QUEUE)
    ids = _validate_queue(rows)
    design = read_json(DESIGN)
    manifest = read_json(QUEUE_MANIFEST)
    challenger = read_json(CHALLENGER)

    if (
        design.get("schema") != DESIGN_SCHEMA
        or design.get("campaign_id") != CAMPAIGN_ID
        or design.get("phase_id") != PHASE_ID
        or design.get("gpu_training_started") is not False
        or design.get("publication_allowed") is not False
        or design.get("all_gbs_contracts_passed") is not True
        or design.get("all_first_epoch_capacity_checks_passed") is not True
        or design.get("all_memory_bands_within_safe_limit") is not True
        or design.get("required_gpu_pool", {}).get("gpu_ids") != GPU_IDS
        or design.get("required_gpu_pool", {}).get("two_gpu_masks") != TWO_GPU_MASKS
        or design.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or design.get("queue", {}).get("ordered_job_ids") != ids
        or design.get("safety_flags", {}).get("automatic_packing_recommendation_allowed") is not False
        or design.get("execution_contract", {}).get("automatic_gpu_launch_allowed") is not False
        or manifest.get("design", {}).get("sha256") != sha256_file(DESIGN)
        or manifest.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or manifest.get("gpu_training_started") is not False
    ):
        raise ValueError("design/manifest drifted from the queue under review")

    # The motivating challenger must still be the artifact the design cites, so a
    # later refit cannot silently change what this campaign is for.
    if challenger.get("report_sha256") != design["motivating_evidence"]["challenger"]["report_sha256"]:
        raise ValueError("motivating challenger changed since the design was written")
    if challenger.get("publishable") is not False:
        raise ValueError("motivating challenger is unexpectedly publishable")

    hardware = probe_hardware(required_gpu_ids=GPU_IDS)
    if hardware.get("exact_h800_pool") is not True:
        raise RuntimeError(f"selected GPUs are not an exact four-H800 pool: {hardware}")
    live_processes = gpu_process_snapshot(GPU_IDS).get("processes") or []
    blocked = sorted({int(row["gpu_index"]) for row in live_processes})
    if blocked:
        # Never preempt: refuse to freeze while another task holds any of 0-3.
        raise RuntimeError(
            f"GPUs {blocked} are busy; this campaign must not preempt external work"
        )
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError(f"live runtime patch is unhealthy: {runtime_patch}")

    queue_binding = build_queue_binding(QUEUE, rows, ROOT)
    provenance_binding = build_provenance_binding(ROOT)

    # Scoped provenance: bind only the files this campaign actually reads, so a
    # parallel session adding an unrelated script cannot fail-closed the queue.
    bound_files = {
        QUEUE,
        DESIGN,
        QUEUE_MANIFEST,
        CHALLENGER,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "hardware.json",
        ROOT / "data" / "dataset_info.json",
    }
    for binding in (design.get("source_bindings") or {}).values():
        path = Path(binding["path"])
        bound_files.add(path)
        if sha256_file(path) != binding.get("sha256"):
            raise ValueError(f"campaign source binding drifted: {path}")
    for row in rows:
        for key in (
            "data_path",
            "dataset_profile_path",
            "packing_dataprofile_path",
            "declared_model_manifest_path",
        ):
            bound_files.add(Path(row[key]))
    missing = [str(p) for p in bound_files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"approval-bound files missing: {missing}")
    root = ROOT.resolve()
    file_manifest = {
        str(p.resolve().relative_to(root)): sha256_file(p)
        for p in sorted(bound_files, key=str)
        if p.resolve().is_relative_to(root)
    }

    approval: dict[str, Any] = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}",
        "campaign_design": {
            "path": str(DESIGN.resolve().relative_to(root)),
            "sha256": sha256_file(DESIGN),
            "report_sha256": design["report_sha256"],
        },
        "file_sha256": file_manifest,
        "execution_order": [PHASE_ID],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": GPU_IDS,
        "max_gpu_count": MAX_GPU_COUNT,
        "scheduler_execution": {
            "join_busy_pool": True,
            "preemption_allowed": False,
            "policy": (
                "use only idle fully-NVLinked two-GPU masks [0,1] and [2,3]; wait "
                "rather than preempt"
            ),
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
        "throughput_screen_delta": {
            "allowed_job_ids": ids,
            "queue_path": queue_binding["path"],
            "queue_sha256": queue_binding["sha256"],
            "ordered_job_ids": queue_binding["ordered_job_ids"],
            "ordered_job_payload_sha256": queue_binding["ordered_job_payload_sha256"],
            "job_payload_sha256": queue_binding["job_payload_sha256"],
        },
        "evidence_role": {
            "candidate_role": "packing_branch_coefficient_gap_fit_only",
            "oom_role": "right_censored_lower_bound_not_regression_label",
            "software_failure_role": "repair_and_rerun_same_job_id_and_payload",
            "publication_allowed": False,
            "automatic_packing_recommendation_allowed": False,
            "acceptance_claim_allowed": False,
        },
    }
    write_json(CANDIDATE, approval)
    return CANDIDATE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="perform the atomic approval swap (without this flag the run is a dry run)",
    )
    args = parser.parse_args()
    candidate = freeze()
    digest = sha256_file(candidate)
    live_before = read_json(ROOT / "runtime" / "approval_design.json")
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
                "mode": "promote" if args.promote else "dry_run",
                "candidate": str(candidate),
                "candidate_sha256": digest,
                "would_replace": {
                    "design_purpose": live_before.get("design_purpose"),
                    "authorized_gpu_ids": live_before.get("authorized_gpu_ids"),
                    "allowed_job_ids": len(live_before.get("allowed_job_ids") or []),
                },
                "new_approval": {
                    "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}",
                    "authorized_gpu_ids": GPU_IDS,
                    "allowed_job_ids": EXPECTED_JOBS,
                },
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
