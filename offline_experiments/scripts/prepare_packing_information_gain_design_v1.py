#!/usr/bin/env python3
"""Prepare, but never materialize, the next Packing information-gain batch."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import ARTIFACT_DIR, sha256_file, sha256_json, write_json
from fit_packing_profile_estimators_v1 import FEATURE_NAMES, feature_vector
from packing_gbs_contract import derive_packing_gbs_contract


SCHEMA = "sft_packing_information_gain_design/v1"
PROFILE_DIR = ARTIFACT_DIR / "packing_dataprofile_v2"
ESTIMATOR = ARTIFACT_DIR / "packing_profile_estimators_v1.json"
ACCEPTANCE = ARTIFACT_DIR / "packing_profile_estimators_acceptance_v1.json"
MEMBERSHIP = ARTIFACT_DIR / "packing_fit_membership_v1.json"
OUTPUT = ARTIFACT_DIR / "packing_information_gain_design_v1.json"
MARKDOWN = ARTIFACT_DIR / "packing_information_gain_design_v1.md"

TARGET_GBS = 128
DATA_PARALLEL = 2
GPU_COUNT = 2
ZERO = "zero2"
GC = True
REPEATS_PER_TREATMENT = 3
JOBS_PER_FAMILY = 2 * REPEATS_PER_TREATMENT
SELECTED_FAMILIES = 6

# These exact-cache endpoints cover natural multi-turn, near-cutoff saturation,
# bimodality, and code/structured data while keeping one common DP/GBS contract.
CANDIDATE_CUTOFFS = {
    "W3": (4_096, 40_960),
    "W5": (16_384, 32_768),
    "W7": (20_480,),
    "W8": (2_048, 10_240),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile(workload_id: str) -> tuple[Path, dict[str, Any]]:
    path = PROFILE_DIR / f"{workload_id.lower()}_packing_dataprofile_v2.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, _read_json(path)


def _candidate(workload_id: str, cutoff_len: int) -> dict[str, Any]:
    path, profile = _profile(workload_id)
    points = {int(point["cutoff_len"]): point for point in profile["packing_curve"]}
    if cutoff_len not in points:
        raise ValueError(f"{workload_id} has no exact cached cutoff {cutoff_len}")
    point = points[cutoff_len]
    if float(point["sample_truncation_rate"]) != 0.0:
        raise ValueError(f"{workload_id}@{cutoff_len} truncates the profile")
    contract = derive_packing_gbs_contract(
        target_gbs=TARGET_GBS,
        data_parallel=DATA_PARALLEL,
        samples_per_pack=point["samples_per_pack"],
        epsilon_gbs=0.10,
        maximum_center_relative_error=0.05,
    )
    if contract["gates"]["candidate_admissible"] is not True:
        raise ValueError(f"{workload_id}@{cutoff_len} violates the strict GBS contract")
    return {
        "candidate_id": f"{workload_id.lower()}-c{cutoff_len}-dp{DATA_PARALLEL}-g{TARGET_GBS}",
        "workload_id": workload_id,
        "profile_id": profile["profile_id"],
        "profile_role": profile["profile_role"],
        "profile_path": str(path.resolve()),
        "profile_sha256": sha256_file(path),
        "cutoff_len": cutoff_len,
        "gpu_count": GPU_COUNT,
        "data_parallel": DATA_PARALLEL,
        "target_gbs": TARGET_GBS,
        "zero": ZERO,
        "gc": GC,
        "mbs": 1,
        "packing_contract": contract,
        "exact_cached_curve": {
            "pack_utilization": float(point["pack_utilization"]),
            "samples_per_pack": point["samples_per_pack"],
            "packs": int(point["packs"]),
            "sample_truncation_rate": float(point["sample_truncation_rate"]),
        },
        "design_features": dict(zip(FEATURE_NAMES, feature_vector(profile, point), strict=True)),
    }


def _logdet_score(rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> float:
    all_x = np.asarray([[row["design_features"][name] for name in FEATURE_NAMES] for row in all_rows])
    mean = np.mean(all_x, axis=0)
    scale = np.std(all_x, axis=0)
    scale[scale == 0] = 1.0
    x = np.asarray([[row["design_features"][name] for name in FEATURE_NAMES] for row in rows])
    x = (x - mean) / scale
    x = np.column_stack((np.ones(len(x)), x))
    sign, value = np.linalg.slogdet(np.eye(x.shape[1]) + x.T @ x)
    if sign <= 0:
        raise ValueError("invalid information matrix")
    return float(value)


def _select(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trials = []
    for indices in combinations(range(len(candidates)), SELECTED_FAMILIES):
        rows = [candidates[index] for index in indices]
        if {row["workload_id"] for row in rows} != set(CANDIDATE_CUTOFFS):
            continue
        score = _logdet_score(rows, candidates)
        trials.append({
            "candidate_ids": [row["candidate_id"] for row in rows],
            "ridge_logdet_information_score": score,
        })
    if not trials:
        raise ValueError("no feasible constrained information-gain selection")
    winner = max(trials, key=lambda row: (row["ridge_logdet_information_score"], row["candidate_ids"]))
    selected_ids = set(winner["candidate_ids"])
    selected = [row for row in candidates if row["candidate_id"] in selected_ids]
    return selected, sorted(trials, key=lambda row: row["ridge_logdet_information_score"], reverse=True)


def build_design() -> dict[str, Any]:
    estimator = _read_json(ESTIMATOR)
    acceptance = _read_json(ACCEPTANCE)
    membership = _read_json(MEMBERSHIP)
    if acceptance["gates"]["fit_only_estimator_ready"] is not True:
        raise ValueError("fit-only estimator safety gates have not passed")
    if acceptance["gates"]["automatic_gbs_candidate_pruning_allowed"] is not False:
        raise ValueError("this design expects exact-cache selection, not automatic estimator pruning")
    candidates = [
        _candidate(workload_id, cutoff)
        for workload_id, cutoffs in CANDIDATE_CUTOFFS.items()
        for cutoff in cutoffs
    ]
    selected, trials = _select(candidates)
    for index, row in enumerate(selected):
        row["counterbalanced_treatment_order"] = (
            ["unpacked", "packed", "packed", "unpacked", "unpacked", "packed"]
            if index % 2 == 0
            else ["packed", "unpacked", "unpacked", "packed", "packed", "unpacked"]
        )
        row["planned_jobs"] = JOBS_PER_FAMILY
        row["candidate_role"] = "prospective_gpu_information_gain"
        row["memory_gate_status"] = "requires_frozen_prediction_and_preflight_before_materialization"
    counts = Counter(row["workload_id"] for row in selected)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_design_only_not_materialized",
        "objective": "calibrate Packing route effect across W3/W5/W7/W8 exact-cache profiles",
        "source_bindings": {
            "estimator": {"path": str(ESTIMATOR.resolve()), "sha256": sha256_file(ESTIMATOR), "model_sha256": estimator["model_sha256"]},
            "acceptance": {"path": str(ACCEPTANCE.resolve()), "sha256": sha256_file(ACCEPTANCE), "report_sha256": acceptance["report_sha256"]},
            "fit_membership": {"path": str(MEMBERSHIP.resolve()), "sha256": sha256_file(MEMBERSHIP), "report_sha256": membership["report_sha256"]},
        },
        "selection": {
            "policy": "constrained_ridge_logdet_information_gain_v1",
            "candidate_families": len(candidates),
            "selected_families": len(selected),
            "workload_coverage": dict(sorted(counts.items())),
            "all_trial_scores": trials,
        },
        "execution_contract": {
            "model_id": "qwen3_8b",
            "train_type": "lora",
            "hardware": "NVIDIA H800 140GB HBM3",
            "gpu_count_per_job": GPU_COUNT,
            "target_gbs": TARGET_GBS,
            "zero": ZERO,
            "gc": GC,
            "mbs": 1,
            "unpacked_gradient_accumulation_steps": 64,
            "warmup_steps": 2,
            "measure_steps": 8,
            "repeats_per_treatment": REPEATS_PER_TREATMENT,
        },
        "budget": {
            "families": len(selected),
            "jobs": len(selected) * JOBS_PER_FAMILY,
            "gpu_job_equivalents": len(selected) * JOBS_PER_FAMILY * GPU_COUNT,
        },
        "selected_families": selected,
        "authorization": {
            "queue_materialized": False,
            "gpu_execution_allowed": False,
            "automatic_next_batch_allowed": False,
            "requires_explicit_gpu_envelope_approval": True,
            "reason": "candidate design must pass frozen memory prediction, exact budget review, provenance capture, and explicit GPU approval",
        },
        "known_modeling_gap": {
            "p99_center_family_loo_relative_mae": acceptance["calibration"]["n_pack_step_p99_center"]["scenario_equal_relative_mae"],
            "p99_upper_multiplier": estimator["safety_guard"]["p99_upper_multiplier"],
            "consequence": "exact cached pack-count curves are mandatory for this GPU batch; estimator-only automatic GBS pruning remains disabled",
        },
    }
    if len(selected) != SELECTED_FAMILIES or set(counts) != set(CANDIDATE_CUTOFFS):
        raise ValueError("selection coverage invariant failed")
    if report["budget"]["jobs"] != 36 or report["budget"]["gpu_job_equivalents"] != 72:
        raise ValueError("unexpected candidate budget")
    report["report_sha256"] = sha256_json(report)
    return report


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Packing W3/W5/W7/W8 信息增益候选设计 v1", "",
        "状态：`candidate_design_only_not_materialized`；未生成 queue，未授权 GPU 执行。", "",
        "| Family | profile | cutoff | DP | target GBS | Packed GA | step-P99×DP | jobs |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["selected_families"]:
        contract = row["packing_contract"]
        lines.append(
            f"| {row['candidate_id']} | {row['workload_id']} | {row['cutoff_len']} | "
            f"{row['data_parallel']} | {row['target_gbs']} | {contract['gradient_accumulation_steps']} | "
            f"{contract['global_microstep_sample_gbs']['p99']:.1f} | {row['planned_jobs']} |"
        )
    lines.extend((
        "",
        f"总预算：{report['budget']['families']} families，{report['budget']['jobs']} jobs，{report['budget']['gpu_job_equivalents']} GPU-job equivalents。",
        "",
        "所有点使用 exact cached curve；当前 P99 estimator 上界过宽，禁止仅凭 estimator 自动剪枝。执行前还需冻结显存预测、GPU 范围、queue/provenance 并获得明确批准。",
        "",
    ))
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    report = build_design()
    write_json(OUTPUT, report)
    write_markdown(report)
    return {
        "output": str(OUTPUT), "markdown": str(MARKDOWN),
        "status": report["status"], "budget": report["budget"],
        "selected": [row["candidate_id"] for row in report["selected_families"]],
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
