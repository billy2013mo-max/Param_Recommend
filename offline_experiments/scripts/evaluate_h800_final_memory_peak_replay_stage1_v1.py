#!/usr/bin/env python3
"""Evaluate full-coverage versus exact-peak-replay CUDA reserved memory."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from evaluate_h800_unified_bounded_canary_v3 import (
    _attempt_dir,
    _structure_audit,
    _summaries,
)
from prepare_h800_final_memory_peak_replay_stage1_v1 import (
    ACCEPTANCE_THRESHOLDS,
    CAMPAIGN_ID,
    DEFAULT_DATA_BUNDLE,
    DEFAULT_DESIGN,
    DEFAULT_PREDICTIONS,
    DEFAULT_QUEUE,
    EXPECTED_JOBS,
    EXPECTED_PAIRS,
    PHASE_ID,
    V3_ARTIFACT,
)

SCHEMA = "sft_h800_final_memory_peak_replay_results/v1"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_final_memory_peak_replay_results_v1.json"
DEFAULT_MARKDOWN = ARTIFACT_DIR / "h800_final_memory_peak_replay_results_v1.md"
RUNTIME_PARAMETERS = {
    "qwen3_4b": 4_022_468_096,
    "qwen3_8b": 8_190_735_360,
}


def _observed_max_padded_sequence(summaries: list[dict[str, Any]]) -> int | None:
    values = []
    for summary in summaries:
        evidence = summary.get("batch_shape_evidence") or {}
        batches = [
            *(evidence.get("warmup_microbatches") or []),
            *(evidence.get("measured_microbatches") or []),
        ]
        values.extend(
            int(row["padded_sequence_length"])
            for row in batches
            if row.get("padded_sequence_length") is not None
        )
    return max(values) if values else None


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
    max_allocated = max(
        (float(row["max_allocated"]) for row in summaries), default=None
    )
    if classification == "success" and max_reserved is None:
        raise RuntimeError(f"successful job has no max_reserved: {job_id}")
    expected_peak = int(job["peak_batch_contract"]["padded_sequence_length"])
    observed_peak = _observed_max_padded_sequence(summaries)
    return {
        "job_id": job_id,
        "pair_id": str(job["pair_id"]),
        "replay_mode": str(job["replay_mode"]),
        "source_dataset_id": str(job["source_dataset_id"]),
        "profile_role": str(job["profile_role"]),
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
        "expected_peak_padded_sequence_length": expected_peak,
        "observed_max_padded_sequence_length": observed_peak,
        "expected_peak_shape_observed": (
            observed_peak is not None and observed_peak >= expected_peak
        ),
        "runtime_model_identity": structure,
        "rank_summaries": len(summaries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--data-bundle", type=Path, default=DEFAULT_DATA_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    jobs = read_jsonl(args.queue)
    design = read_json(args.design)
    predictions = read_json(args.predictions)
    data_bundle = read_json(args.data_bundle)
    if (
        len(jobs) != EXPECTED_JOBS
        or len(predictions.get("rows") or []) != EXPECTED_JOBS
    ):
        raise ValueError("stage-one input size drifted")
    bindings = design.get("bindings") or {}
    if (
        bindings.get("queue", {}).get("sha256") != sha256_file(args.queue)
        or bindings.get("frozen_predictions", {}).get("sha256")
        != sha256_file(args.predictions)
        or bindings.get("data_bundle", {}).get("sha256")
        != sha256_file(args.data_bundle)
        or predictions.get("v3_artifact", {}).get("sha256") != sha256_file(V3_ARTIFACT)
        or predictions.get("ordered_job_payload_sha256") != sha256_json(jobs)
        or predictions.get("outcomes_observed") != 0
        or predictions.get("status") != "frozen_before_any_stage1_gpu_outcome"
        or data_bundle.get("final_acceptance_sources_read") != 0
        or design.get("acceptance_thresholds") != ACCEPTANCE_THRESHOLDS
    ):
        raise ValueError("stage-one frozen binding drifted")

    observations = []
    incomplete = []
    for job in jobs:
        observation = _terminal_observation(job)
        if observation is None:
            incomplete.append(str(job["job_id"]))
        else:
            observations.append(observation)
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(f"stage-one campaign is incomplete: {incomplete}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[str(row["pair_id"])].append(row)
    pairs = []
    for pair_id, rows in sorted(grouped.items()):
        by_mode = {str(row["replay_mode"]): row for row in rows}
        full = by_mode.get("full_coverage")
        replay = by_mode.get("peak_replay")
        complete_success = bool(
            full
            and replay
            and full["classification"] == "success"
            and replay["classification"] == "success"
        )
        relative_error = None
        if complete_success:
            relative_error = (
                float(replay["max_reserved_bytes"]) - float(full["max_reserved_bytes"])
            ) / float(full["max_reserved_bytes"])
        anchor = full or replay or {}
        pairs.append(
            {
                "pair_id": pair_id,
                "source_dataset_id": anchor.get("source_dataset_id"),
                "profile_role": anchor.get("profile_role"),
                "model_id": anchor.get("model_id"),
                "train_type": anchor.get("train_type"),
                "gpu_count": anchor.get("gpu_count"),
                "zero_stage": anchor.get("zero_stage"),
                "gc": anchor.get("gc"),
                "mbs": anchor.get("mbs"),
                "cutoff_len": anchor.get("cutoff_len"),
                "complete_success": complete_success,
                "full_job_id": full.get("job_id") if full else None,
                "replay_job_id": replay.get("job_id") if replay else None,
                "full_classification": full.get("classification") if full else None,
                "replay_classification": replay.get("classification")
                if replay
                else None,
                "full_max_reserved_bytes": full.get("max_reserved_bytes")
                if full
                else None,
                "replay_max_reserved_bytes": replay.get("max_reserved_bytes")
                if replay
                else None,
                "relative_replay_error": relative_error,
                "absolute_replay_error": (
                    abs(relative_error) if relative_error is not None else None
                ),
                "full_peak_shape_observed": (
                    full.get("expected_peak_shape_observed") if full else False
                ),
                "replay_peak_shape_observed": (
                    replay.get("expected_peak_shape_observed") if replay else False
                ),
            }
        )

    complete_pairs = [row for row in pairs if row["complete_success"]]
    errors = [float(row["relative_replay_error"]) for row in complete_pairs]
    mean_absolute_error = (
        statistics.fmean(abs(value) for value in errors) if errors else None
    )
    directional_underprediction = {}
    for train_type in ("full", "lora"):
        values = [
            float(row["relative_replay_error"])
            for row in complete_pairs
            if row["train_type"] == train_type
        ]
        directional_underprediction[train_type] = {
            "pairs": len(values),
            "errors": values,
            "all_underpredict": bool(values) and all(value < 0.0 for value in values),
        }
    thresholds = ACCEPTANCE_THRESHOLDS
    checks = {
        "all_jobs_terminal": not incomplete and len(observations) == EXPECTED_JOBS,
        "all_jobs_success": len(observations) == EXPECTED_JOBS
        and all(row["classification"] == "success" for row in observations),
        "all_runtime_model_identities_passed": all(
            row["runtime_model_identity"]["all_passed"] for row in observations
        ),
        "all_expected_peak_shapes_observed": all(
            row["expected_peak_shape_observed"] for row in observations
        ),
        "minimum_complete_pairs": len(complete_pairs)
        >= int(thresholds["minimum_complete_pairs"]),
        "mean_absolute_replay_error": mean_absolute_error is not None
        and mean_absolute_error <= float(thresholds["mean_absolute_replay_error_max"]),
        "no_replay_underprediction_over_limit": bool(errors)
        and min(errors) >= -float(thresholds["maximum_replay_underprediction"]),
        "full_no_directionally_consistent_underprediction": not directional_underprediction[
            "full"
        ]["all_underpredict"],
        "lora_no_directionally_consistent_underprediction": not directional_underprediction[
            "lora"
        ]["all_underpredict"],
    }
    complete = not incomplete and len(pairs) == EXPECTED_PAIRS
    passed = complete and all(checks.values())
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "measurement_equivalence_passed"
            if passed
            else "measurement_equivalence_failed"
            if complete
            else "incomplete"
        ),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "production_model_mutated": False,
        "final_acceptance_outcomes_observed": 0,
        "acceptance_thresholds": thresholds,
        "terminal_counts": dict(Counter(row["classification"] for row in observations)),
        "incomplete_job_ids": incomplete,
        "metrics": {
            "complete_pairs": len(complete_pairs),
            "mean_absolute_replay_error": mean_absolute_error,
            "maximum_underprediction": min(errors) if errors else None,
            "directional_underprediction": directional_underprediction,
        },
        "checks": checks,
        "all_measurement_equivalence_checks_passed": passed,
        "pairs": pairs,
        "observations": observations,
        "bindings": {
            "queue": {
                "path": str(args.queue.resolve()),
                "sha256": sha256_file(args.queue),
            },
            "design": {
                "path": str(args.design.resolve()),
                "sha256": sha256_file(args.design),
            },
            "frozen_predictions": {
                "path": str(args.predictions.resolve()),
                "sha256": sha256_file(args.predictions),
            },
            "data_bundle": {
                "path": str(args.data_bundle.resolve()),
                "sha256": sha256_file(args.data_bundle),
            },
        },
    }
    report["report_sha256"] = sha256_json(report)
    write_json(args.output, report)

    def pct(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):+.2%}"

    def gib(value: Any) -> str:
        return "n/a" if value is None else f"{float(value) / (1 << 30):.2f}"

    lines = [
        "# H800 完整运行与峰值回放等价性校准",
        "",
        f"状态：`{report['status']}`。本阶段只使用四个开发数据集，最终六个业务盲测集结果仍未观察。",
        "",
        "| 开发数据集 | 机制 | 完整运行 GiB | 峰值回放 GiB | 相对误差 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in pairs:
        mechanism = (
            f"{row['model_id']} / {row['train_type']} / {row['gpu_count']}卡 / "
            f"ZeRO-{row['zero_stage']} / GC={'开' if row['gc'] else '关'} / MBS={row['mbs']}"
        )
        lines.append(
            f"| `{row['source_dataset_id']}` | {mechanism} | "
            f"{gib(row['full_max_reserved_bytes'])} | "
            f"{gib(row['replay_max_reserved_bytes'])} | "
            f"{pct(row['relative_replay_error'])} |"
        )
    lines.extend(
        [
            "",
            f"四对平均绝对误差：{pct(mean_absolute_error)}。",
            "",
        ]
    )
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
