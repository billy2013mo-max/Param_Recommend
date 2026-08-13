#!/usr/bin/env python3
"""Aggregate rank metrics, external GPU peaks, throughput and MFU labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from common import ARTIFACT_DIR, CONFIG_DIR, RESULTS_DIR, read_json, write_json
from flops import result_flops
from gpu_telemetry import (
    add_clock_adjusted_mfu,
    aggregate_gpu_telemetry,
    read_nvidia_smi_rows,
)


def aggregate_result(result_dir: Path, models: dict[str, dict[str, Any]], hardware: dict[str, Any]) -> dict[str, Any] | None:
    status_path = result_dir / "status.json"
    rendered_path = result_dir / "rendered_run.json"
    if not status_path.is_file() or not rendered_path.is_file():
        return None
    status = read_json(status_path)
    rendered = read_json(rendered_path)
    job = rendered["job"]
    summaries = [read_json(path) for path in sorted((result_dir / "metrics").glob("summary.rank*.json"))]
    telemetry = aggregate_gpu_telemetry(
        read_nvidia_smi_rows(result_dir / "nvidia_smi.csv"),
        hardware,
    )
    row: dict[str, Any] = {
        **{key: value for key, value in job.items() if not isinstance(value, (dict, list))},
        **status,
        "rank_summaries": len(summaries),
        **telemetry,
    }
    if not summaries:
        return row

    measured_seconds = max(
        float(summary.get("measured_seconds") or 0)
        for summary in summaries
    )
    peak_allocated = max(
        int(summary.get("max_allocated") or 0)
        for summary in summaries
    )
    peak_reserved = max(
        int(summary.get("max_reserved") or 0)
        for summary in summaries
    )
    if measured_seconds <= 0:
        # OOM probes can emit a valid rank summary before completing a measured
        # step.  Preserve their terminal classification and memory evidence,
        # but do not manufacture rates or fail collection by dividing by zero.
        row.update(
            {
                "measured_seconds": measured_seconds,
                "max_allocated_bytes": peak_allocated,
                "max_reserved_bytes": peak_reserved,
                "metrics_available": False,
                "metrics_unavailable_reason": "non_positive_measured_seconds",
            }
        )
        return row

    totals: dict[str, int] = {}
    for key in summaries[0]["measured_totals"]:
        totals[key] = sum(int(summary["measured_totals"][key]) for summary in summaries)
    model = models[job["model_id"]]
    flops = result_flops(model, job["train_type"], totals)
    gpu_count = int(job["gpu_count"])
    device_peak = hardware["bf16_dense_peak_flops_per_second_for_mfu"]
    public_reference_peak = hardware["bf16_public_reference_dense_peak_flops_per_second"]
    row.update(
        {
            "measured_seconds": measured_seconds,
            "metrics_available": True,
            "computed_tokens": totals["computed_tokens"],
            "effective_tokens": totals["effective_tokens"],
            "logical_samples": totals["logical_samples"],
            "computed_tokens_per_second": totals["computed_tokens"] / measured_seconds,
            "effective_tokens_per_second": totals["effective_tokens"] / measured_seconds,
            "samples_per_second": totals["logical_samples"] / measured_seconds,
            "max_allocated_bytes": peak_allocated,
            "max_reserved_bytes": peak_reserved,
            **flops,
            "mfu": flops["computed_useful_flops"] / (measured_seconds * gpu_count * device_peak),
            "effective_mfu": flops["effective_useful_flops"] / (measured_seconds * gpu_count * device_peak),
            "public_reference_mfu": flops["computed_useful_flops"]
            / (measured_seconds * gpu_count * public_reference_peak),
        }
    )
    add_clock_adjusted_mfu(row)
    return row


def main() -> None:
    inventory = read_json(ARTIFACT_DIR / "model_inventory.json")
    models = {model["id"]: model for model in inventory["models"]}
    hardware = read_json(CONFIG_DIR / "hardware.json")
    rows = []
    if RESULTS_DIR.exists():
        for result_dir in sorted(path for path in RESULTS_DIR.iterdir() if path.is_dir()):
            row = aggregate_result(result_dir, models, hardware)
            if row is not None:
                rows.append(row)
    write_json(ARTIFACT_DIR / "collected_results.json", {"schema_version": 2, "rows": rows})
    if rows:
        serializable_rows = [
            {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()}
            for row in rows
        ]
        pd.DataFrame(serializable_rows).to_parquet(ARTIFACT_DIR / "collected_results.parquet", index=False)
    print(f"Collected {len(rows)} result rows")


if __name__ == "__main__":
    main()
