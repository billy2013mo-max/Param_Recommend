#!/usr/bin/env python3
"""Evaluate the LoRA allocator-prefix repair and the combined four-pair gate."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from evaluate_h800_final_memory_peak_replay_stage1_v1 import (
    RUNTIME_PARAMETERS,
    _observed_max_padded_sequence,
)
from evaluate_h800_unified_bounded_canary_v3 import _attempt_dir, _structure_audit, _summaries
from prepare_h800_final_memory_allocator_prefix_replay_v2 import (
    CAMPAIGN_ID,
    DEFAULT_DATA_BUNDLE,
    DEFAULT_DESIGN,
    DEFAULT_QUEUE,
    EXPECTED_JOBS,
    PHASE_ID,
    V1_RESULTS,
)

SCHEMA = "sft_h800_final_memory_allocator_prefix_replay_results/v2"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_final_memory_allocator_prefix_replay_results_v2.json"
DEFAULT_MARKDOWN = ARTIFACT_DIR / "h800_final_memory_allocator_prefix_replay_results_v2.md"


def _terminal_observation(job: dict[str, Any]) -> dict[str, Any] | None:
    job_id = str(job["job_id"])
    status_path = ROOT / "results" / job_id / "status.json"
    attempt = _attempt_dir(job_id)
    if not status_path.is_file() or attempt is None:
        return None
    status = read_json(status_path)
    classification = str(status.get("classification") or "")
    if classification not in {"success", "oom"}:
        return None
    structure = _structure_audit(
        attempt,
        expected_parameters=RUNTIME_PARAMETERS[str(job["model_id"])],
        expected_world_size=int(job["gpu_count"]),
    )
    if not structure["all_passed"]:
        raise RuntimeError(f"runtime model identity mismatch: {job_id}: {structure}")
    summaries = _summaries(attempt)
    max_reserved = max((float(row["max_reserved"]) for row in summaries), default=None)
    max_allocated = max((float(row["max_allocated"]) for row in summaries), default=None)
    if classification == "success" and max_reserved is None:
        raise RuntimeError(f"successful repair job has no memory summary: {job_id}")
    contract = job["allocator_prefix_contract"]
    expected_padded = max(
        int(row["rank_max_padded_sequence_length"])
        for row in contract["rank_contracts"]
    )
    observed_padded = _observed_max_padded_sequence(summaries)
    completed_steps = min(int(row.get("total_steps") or 0) for row in summaries)
    return {
        "job_id": job_id,
        "v1_full_baseline_job_id": str(job["v1_full_baseline_job_id"]),
        "v1_pair_id": str(job["v1_pair_id"]),
        "source_dataset_id": str(job["source_dataset_id"]),
        "model_id": str(job["model_id"]),
        "train_type": str(job["train_type"]),
        "gpu_count": int(job["gpu_count"]),
        "zero_stage": int(job["zero_stage"]),
        "gc": bool(job["gc"]),
        "mbs": int(job["mbs"]),
        "cutoff_len": int(job["cutoff_len"]),
        "classification": classification,
        "wall_seconds": float(status.get("wall_seconds") or 0.0),
        "max_reserved_bytes": max_reserved,
        "max_allocated_bytes": max_allocated,
        "expected_optimizer_steps": int(contract["prefix_optimizer_steps"]),
        "completed_optimizer_steps": completed_steps,
        "expected_max_padded_sequence_length": expected_padded,
        "observed_max_padded_sequence_length": observed_padded,
        "expected_prefix_shape_observed": (
            observed_padded is not None and observed_padded >= expected_padded
        ),
        "runtime_model_identity": structure,
        "rank_summaries": len(summaries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--data-bundle", type=Path, default=DEFAULT_DATA_BUNDLE)
    parser.add_argument("--v1-results", type=Path, default=V1_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    jobs = read_jsonl(args.queue)
    design = read_json(args.design)
    data_bundle = read_json(args.data_bundle)
    v1 = read_json(args.v1_results)
    bindings = design.get("bindings") or {}
    if (
        len(jobs) != EXPECTED_JOBS
        or design.get("ordered_job_ids") != [str(row["job_id"]) for row in jobs]
        or design.get("ordered_job_payload_sha256") != sha256_json(jobs)
        or bindings.get("queue", {}).get("sha256") != sha256_file(args.queue)
        or bindings.get("data_bundle", {}).get("sha256") != sha256_file(args.data_bundle)
        or bindings.get("v1_results", {}).get("sha256") != sha256_file(args.v1_results)
        or data_bundle.get("final_acceptance_sources_read") != 0
        or v1.get("final_acceptance_outcomes_observed") != 0
    ):
        raise ValueError("allocator-prefix repair binding drifted")

    observations = []
    incomplete = []
    for job in jobs:
        row = _terminal_observation(job)
        if row is None:
            incomplete.append(str(job["job_id"]))
        else:
            observations.append(row)
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(f"allocator-prefix repair is incomplete: {incomplete}")

    v1_pairs = {str(row["pair_id"]): row for row in v1["pairs"]}
    combined_pairs = [dict(row) for row in v1["pairs"] if row["train_type"] == "full"]
    repair_pairs = []
    for observation in observations:
        baseline = v1_pairs[str(observation["v1_pair_id"])]
        if (
            baseline.get("train_type") != "lora"
            or baseline.get("full_job_id") != observation["v1_full_baseline_job_id"]
        ):
            raise ValueError("v1 LoRA baseline binding mismatch")
        complete_success = observation["classification"] == "success"
        relative_error = None
        if complete_success:
            relative_error = (
                float(observation["max_reserved_bytes"])
                - float(baseline["full_max_reserved_bytes"])
            ) / float(baseline["full_max_reserved_bytes"])
        pair = {
            **dict(baseline),
            "complete_success": complete_success,
            "replay_job_id": observation["job_id"],
            "replay_classification": observation["classification"],
            "replay_max_reserved_bytes": observation["max_reserved_bytes"],
            "relative_replay_error": relative_error,
            "absolute_replay_error": abs(relative_error) if relative_error is not None else None,
            "replay_peak_shape_observed": observation["expected_prefix_shape_observed"],
            "replay_mode": "allocator_prefix_replay",
            "prefix_optimizer_steps": observation["expected_optimizer_steps"],
            "full_plan_optimizer_steps": next(
                int(job["allocator_prefix_contract"]["full_plan_optimizer_steps"])
                for job in jobs
                if job["job_id"] == observation["job_id"]
            ),
        }
        repair_pairs.append(pair)
        combined_pairs.append(pair)
    combined_pairs.sort(key=lambda row: str(row["pair_id"]))
    complete_pairs = [row for row in combined_pairs if row.get("complete_success")]
    errors = [float(row["relative_replay_error"]) for row in complete_pairs]
    mean_absolute_error = statistics.fmean(abs(value) for value in errors) if errors else None
    directional = {}
    for train_type in ("full", "lora"):
        values = [
            float(row["relative_replay_error"])
            for row in complete_pairs
            if row["train_type"] == train_type
        ]
        directional[train_type] = {
            "pairs": len(values),
            "errors": values,
            "all_underpredict": bool(values) and all(value < 0 for value in values),
        }
    thresholds = design["acceptance_thresholds"]
    checks = {
        "all_repair_jobs_terminal": not incomplete and len(observations) == EXPECTED_JOBS,
        "all_repair_jobs_success": len(observations) == EXPECTED_JOBS
        and all(row["classification"] == "success" for row in observations),
        "all_runtime_model_identities_passed": all(
            row["runtime_model_identity"]["all_passed"] for row in observations
        ),
        "all_prefix_steps_completed": all(
            row["completed_optimizer_steps"] == row["expected_optimizer_steps"]
            for row in observations
        ),
        "all_expected_prefix_shapes_observed": all(
            row["expected_prefix_shape_observed"] for row in observations
        ),
        "combined_four_pairs_complete": len(complete_pairs) == 4,
        "combined_mean_absolute_replay_error": mean_absolute_error is not None
        and mean_absolute_error
        <= float(thresholds["combined_four_pair_mean_absolute_replay_error_max"]),
        "combined_no_underprediction_over_limit": bool(errors)
        and min(errors)
        >= -float(thresholds["combined_four_pair_maximum_replay_underprediction"]),
        "full_no_directionally_consistent_underprediction": not directional["full"]["all_underpredict"],
        "lora_no_directionally_consistent_underprediction": not directional["lora"]["all_underpredict"],
    }
    complete = not incomplete and len(combined_pairs) == 4
    passed = complete and all(checks.values())
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "measurement_equivalence_repaired"
            if passed
            else "measurement_equivalence_repair_failed"
            if complete
            else "incomplete"
        ),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "production_model_mutated": False,
        "final_acceptance_outcomes_observed": 0,
        "terminal_counts": dict(Counter(row["classification"] for row in observations)),
        "incomplete_job_ids": incomplete,
        "acceptance_thresholds": thresholds,
        "metrics": {
            "combined_complete_pairs": len(complete_pairs),
            "combined_mean_absolute_replay_error": mean_absolute_error,
            "combined_maximum_underprediction": min(errors) if errors else None,
            "directional_underprediction": directional,
            "repair_wall_seconds": sum(row["wall_seconds"] for row in observations),
        },
        "checks": checks,
        "all_measurement_equivalence_checks_passed": passed,
        "repair_pairs": repair_pairs,
        "combined_pairs": combined_pairs,
        "repair_observations": observations,
        "bindings": {
            "queue": {"path": str(args.queue.resolve()), "sha256": sha256_file(args.queue)},
            "design": {"path": str(args.design.resolve()), "sha256": sha256_file(args.design)},
            "data_bundle": {"path": str(args.data_bundle.resolve()), "sha256": sha256_file(args.data_bundle)},
            "v1_results": {"path": str(args.v1_results.resolve()), "sha256": sha256_file(args.v1_results)},
        },
    }
    report["report_sha256"] = sha256_json(report)
    write_json(args.output, report)

    def pct(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):+.2%}"

    def gib(value: Any) -> str:
        return "n/a" if value is None else f"{float(value) / (1 << 30):.2f}"

    lines = [
        "# H800 allocator 前缀回放修复验证",
        "",
        f"状态：`{report['status']}`。最终业务盲测结果观察数仍为 0。",
        "",
        "| 开发数据集 | 训练机制 | 完整覆盖 GiB | 修复后回放 GiB | 相对误差 | 前缀步数/完整步数 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in combined_pairs:
        prefix = (
            f"{row.get('prefix_optimizer_steps')}/{row.get('full_plan_optimizer_steps')}"
            if row["train_type"] == "lora"
            else "沿用 v1"
        )
        lines.append(
            f"| `{row['source_dataset_id']}` | {row['model_id']} / {row['train_type']} | "
            f"{gib(row['full_max_reserved_bytes'])} | {gib(row['replay_max_reserved_bytes'])} | "
            f"{pct(row['relative_replay_error'])} | {prefix} |"
        )
    lines.extend(["", f"四对平均绝对误差：{pct(mean_absolute_error)}。", ""])
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
