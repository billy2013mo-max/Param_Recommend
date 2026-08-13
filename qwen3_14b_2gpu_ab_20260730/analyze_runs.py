#!/usr/bin/env python3
"""Summarize aligned LlamaFactory throughput runs from step logs and GPU samples."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


STEP_DICT = re.compile(r"(\{[^{}\n]*'num_input_tokens_seen'[^{}\n]*\})")
FINAL_DICT = re.compile(r"(\{[^{}\n]*'train_loss'[^{}\n]*\})")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def parse_step_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(errors="replace").splitlines():
        match = STEP_DICT.search(line)
        if not match:
            continue
        try:
            item = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        if "train_tokens_per_second" not in item:
            continue
        records.append(
            {
                "step": len(records) + 1,
                "loss": float(item["loss"]),
                "tokens": int(item["num_input_tokens_seen"]),
                "runtime_s": float(item["train_runtime"]),
                "cumulative_tokens_per_s": float(item["train_tokens_per_second"]),
            }
        )
    return records


def parse_gpu_samples(path: Path) -> dict[str, Any]:
    by_gpu: dict[str, list[dict[str, float]]] = {}
    if not path.exists():
        return {"sample_count": 0, "by_gpu": {}}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                gpu = row["index"].strip()
                sample = {
                    "memory_used_mib": float(row["memory_used_mib"]),
                    "utilization_gpu_percent": float(row["utilization_gpu_percent"]),
                    "power_draw_w": float(row["power_draw_w"]),
                }
            except (KeyError, ValueError):
                continue
            by_gpu.setdefault(gpu, []).append(sample)
    result: dict[str, Any] = {}
    for gpu, samples in sorted(by_gpu.items(), key=lambda item: int(item[0])):
        active = [sample for sample in samples if sample["utilization_gpu_percent"] > 0]
        result[gpu] = {
            "samples": len(samples),
            "peak_memory_mib": max(sample["memory_used_mib"] for sample in samples),
            "peak_memory_gib": max(sample["memory_used_mib"] for sample in samples) / 1024,
            "active_mean_utilization_percent": (
                statistics.fmean(sample["utilization_gpu_percent"] for sample in active) if active else None
            ),
            "active_mean_power_w": (
                statistics.fmean(sample["power_draw_w"] for sample in active) if active else None
            ),
        }
    return {"sample_count": sum(len(samples) for samples in by_gpu.values()), "by_gpu": result}


def read_exit_code(path: Path) -> int | None:
    if not path.exists():
        return None
    for line in reversed(path.read_text(errors="replace").splitlines()):
        if line.startswith("exit_code="):
            try:
                return int(line.split("=", 1)[1])
            except ValueError:
                return None
    return None


def summarize_run(root: Path, tag: str, stable_from_step: int) -> dict[str, Any]:
    log_path = root / "logs" / f"{tag}.log"
    records = parse_step_records(log_path)
    intervals: list[dict[str, float | int]] = []
    for previous, current in zip(records, records[1:]):
        elapsed = current["runtime_s"] - previous["runtime_s"]
        tokens = current["tokens"] - previous["tokens"]
        if elapsed <= 0 or tokens <= 0:
            continue
        intervals.append(
            {
                "step": current["step"],
                "elapsed_s": elapsed,
                "tokens": tokens,
                "tokens_per_s": tokens / elapsed,
            }
        )
    stable = [item for item in intervals if item["step"] >= stable_from_step]
    nominal_tokens = max((int(item["tokens"]) for item in intervals), default=0)
    stable_full_batch = [item for item in stable if int(item["tokens"]) == nominal_tokens]
    stable_elapsed = sum(float(item["elapsed_s"]) for item in stable)
    stable_tokens = sum(int(item["tokens"]) for item in stable)
    full_batch_elapsed = sum(float(item["elapsed_s"]) for item in stable_full_batch)
    full_batch_tokens = sum(int(item["tokens"]) for item in stable_full_batch)
    stable_step_times = [float(item["elapsed_s"]) for item in stable]
    stable_rates = [float(item["tokens_per_s"]) for item in stable]
    full_batch_step_times = [float(item["elapsed_s"]) for item in stable_full_batch]
    full_batch_rates = [float(item["tokens_per_s"]) for item in stable_full_batch]
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    final_metrics = None
    for match in FINAL_DICT.finditer(log_text):
        try:
            parsed = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        if "train_runtime" in parsed:
            final_metrics = parsed
    checkpoint_match = re.search(r"checkpoint_save:\s*([0-9.]+)s\s*x([0-9]+)", log_text)
    final_save_match = re.search(r"final_save_model:\s*([0-9.]+)s", log_text)
    completed_successfully = (
        len(records) == 45
        and "Training completed." in log_text
        and final_save_match is not None
        and not re.search(r"OutOfMemory|CUDA out of memory", log_text, re.IGNORECASE)
    )
    return {
        "tag": tag,
        "exit_code": read_exit_code(root / "logs" / f"{tag}_meta.txt"),
        "completed_successfully": completed_successfully,
        "completed_steps": len(records),
        "expected_steps": 45,
        "oom_detected": bool(re.search(r"OutOfMemory|CUDA out of memory", log_text, re.IGNORECASE)),
        "allocator_cache_flush_warning_count": log_text.count("allocator cache flushes since last step"),
        "stable_definition": f"optimizer-step intervals ending at step {stable_from_step} or later",
        "stable_interval_count": len(stable),
        "stable_aggregate_tokens_per_s": stable_tokens / stable_elapsed if stable_elapsed else None,
        "nominal_tokens_per_optimizer_step": nominal_tokens or None,
        "stable_full_batch_interval_count": len(stable_full_batch),
        "stable_full_batch_aggregate_tokens_per_s": (
            full_batch_tokens / full_batch_elapsed if full_batch_elapsed else None
        ),
        "stable_full_batch_step_time_s": {
            "mean": statistics.fmean(full_batch_step_times) if full_batch_step_times else None,
            "median": statistics.median(full_batch_step_times) if full_batch_step_times else None,
            "p10": percentile(full_batch_step_times, 0.10),
            "p90": percentile(full_batch_step_times, 0.90),
        },
        "stable_full_batch_interval_tokens_per_s": {
            "mean": statistics.fmean(full_batch_rates) if full_batch_rates else None,
            "median": statistics.median(full_batch_rates) if full_batch_rates else None,
            "p10": percentile(full_batch_rates, 0.10),
            "p90": percentile(full_batch_rates, 0.90),
        },
        "stable_step_time_s": {
            "mean": statistics.fmean(stable_step_times) if stable_step_times else None,
            "median": statistics.median(stable_step_times) if stable_step_times else None,
            "p10": percentile(stable_step_times, 0.10),
            "p90": percentile(stable_step_times, 0.90),
        },
        "stable_interval_tokens_per_s": {
            "mean": statistics.fmean(stable_rates) if stable_rates else None,
            "median": statistics.median(stable_rates) if stable_rates else None,
            "p10": percentile(stable_rates, 0.10),
            "p90": percentile(stable_rates, 0.90),
        },
        "last_step": records[-1] if records else None,
        "final_metrics_including_checkpoint": final_metrics,
        "checkpoint_save_s": float(checkpoint_match.group(1)) if checkpoint_match else None,
        "checkpoint_save_count": int(checkpoint_match.group(2)) if checkpoint_match else None,
        "final_save_model_s": float(final_save_match.group(1)) if final_save_match else None,
        "step_records": records,
        "gpu": parse_gpu_samples(root / "logs" / f"{tag}_gpu.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--stable-from-step", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("tags", nargs="+")
    args = parser.parse_args()

    report = {
        "schema": "qwen3_14b_aligned_throughput_ab/v1",
        "stable_from_step": args.stable_from_step,
        "runs": [summarize_run(args.root, tag, args.stable_from_step) for tag in args.tags],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
