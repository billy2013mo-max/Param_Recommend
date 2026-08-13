#!/usr/bin/env python3
"""Evaluate exact rank, device, ledger and Packing semantics for Phase 0."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, RESULTS_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_packing_final_executor_canary_v1 import (
    CAMPAIGN_ID,
    CUTOFF_LEN,
    EXPECTED_JOBS,
    GPU_COUNTS,
    MEASURE_STEPS,
    PHASE_ID,
    QUEUE,
)


OUTPUT = ARTIFACT_DIR / "h800_packing_final_executor_canary_results_v1.json"
APPROVAL_DESIGN = ROOT / "runtime" / "approval_design.json"


def _relative_error(observed: float, expected: float) -> float:
    return abs(observed - expected) / expected if expected else float("inf")


def _expected_uuid_by_index() -> dict[int, str]:
    design = read_json(APPROVAL_DESIGN)
    if design.get("design_purpose") != f"{CAMPAIGN_ID}:{PHASE_ID}":
        raise ValueError("active approval is not the Phase 0 canary approval")
    hardware = design.get("hardware_preflight") or {}
    inventory = (
        hardware.get("selected_gpu_rows")
        or hardware.get("selected_gpu_inventory")
        or hardware.get("selected_inventory")
        or []
    )
    mapping = {
        int(row["index"]): str(row["uuid"])
        for row in inventory
        if row.get("index") is not None and row.get("uuid")
    }
    if set(mapping) != set(range(8)):
        raise ValueError(f"approval did not freeze all eight physical UUIDs: {mapping}")
    return mapping


def _evaluate_job(job: dict[str, Any], uuid_by_index: dict[int, str]) -> dict[str, Any]:
    job_root = RESULTS_DIR / str(job["job_id"])
    status_path = job_root / "status.json"
    row: dict[str, Any] = {
        "job_id": job["job_id"],
        "gpu_count": int(job["gpu_count"]),
        "packing": bool(job["packing"]),
        "classification": "missing",
        "checks": {},
    }
    if not status_path.is_file():
        row["all_passed"] = False
        return row
    status = read_json(status_path)
    row["classification"] = status.get("classification")
    latest = read_json(job_root / "latest_attempt.json")
    attempt_id = str(latest.get("execution_attempt_id") or "")
    attempt_root = job_root / "attempts" / attempt_id
    summaries = [
        read_json(path)
        for path in sorted((attempt_root / "metrics").glob("summary.rank*.json"))
    ]
    gpu_count = int(job["gpu_count"])
    expected_mask = list(range(gpu_count))
    hardware_path = attempt_root / "runtime_hardware.json"
    hardware = read_json(hardware_path) if hardware_path.is_file() else {}
    devices = hardware.get("devices") or []
    observed_device_map = {
        int(device["physical_index"]): str(device["uuid"])
        for device in devices
        if device.get("physical_index") is not None and device.get("uuid")
    }
    expected_device_map = {index: uuid_by_index[index] for index in expected_mask}

    ranks_exact = (
        len(summaries) == gpu_count
        and {int(summary.get("rank", -1)) for summary in summaries} == set(range(gpu_count))
        and all(int(summary.get("world_size", 0)) == gpu_count for summary in summaries)
        and all(int(summary.get("local_rank", -1)) == int(summary.get("rank", -2)) for summary in summaries)
    )
    ledgers_authoritative = bool(summaries) and all(
        (summary.get("token_ledger_evidence") or {}).get("authoritative") is True
        and (summary.get("batch_shape_evidence") or {}).get("authoritative") is True
        for summary in summaries
    )
    measured_batches = [
        batch
        for summary in summaries
        for batch in (summary.get("batch_shape_evidence") or {}).get("measured_microbatches", [])
    ]
    expected_microbatches = gpu_count * int(job["gradient_accumulation_steps"]) * MEASURE_STEPS
    physical_mbs_exact = (
        len(measured_batches) == expected_microbatches
        and all(int(batch.get("physical_batch_size", 0)) == 1 for batch in measured_batches)
    )
    packing_semantics = (
        all(
            ((summary.get("runtime_batch_evidence") or {}).get("packing") or {}).get("semantic_checks_passed") is True
            for summary in summaries
        )
        if bool(job["packing"])
        else all(not bool(batch.get("packing")) for batch in measured_batches)
    )
    observed_samples = sum(
        float((summary.get("measured_totals") or {}).get("logical_samples") or 0)
        for summary in summaries
    )
    expected_samples = float(job["expected_sample_gbs"]) * MEASURE_STEPS
    sample_error = _relative_error(observed_samples, expected_samples)
    sample_contract = sample_error <= (0.20 if bool(job["packing"]) else 0.0)

    observed_effective_tokens = sum(
        float((summary.get("measured_totals") or {}).get("effective_tokens") or 0)
        for summary in summaries
    )
    static = job["static_packing_contract"]
    expected_effective_tokens = (
        expected_microbatches
        * (CUTOFF_LEN - 1)
        * float(static["pack_utilization"])
    )
    effective_token_error = (
        _relative_error(observed_effective_tokens, expected_effective_tokens)
        if bool(job["packing"])
        else None
    )
    token_contract = (
        effective_token_error is not None and effective_token_error <= 0.15
        if bool(job["packing"])
        else observed_effective_tokens > 0
    )
    checks = {
        "success": status.get("classification") == "success",
        "calibration_eligible": status.get("calibration_eligible") is True,
        "status_mask_exact": status.get("gpu_mask") == ",".join(map(str, expected_mask)),
        "rank_and_world_size_exact": ranks_exact,
        "runtime_hardware_attested": hardware.get("all_passed") is True,
        "runtime_requested_mask_exact": hardware.get("requested_physical_gpu_ids") == expected_mask,
        "runtime_uuid_set_exact": observed_device_map == expected_device_map,
        "authoritative_ledgers_all_ranks": ledgers_authoritative,
        "physical_mbs_one_exact": physical_mbs_exact,
        "packing_semantics_exact": packing_semantics,
        "raw_sample_contract": sample_contract,
        "effective_token_contract": token_contract,
    }
    row.update(
        {
            "execution_attempt_id": attempt_id,
            "expected_gpu_mask": expected_mask,
            "expected_device_map": expected_device_map,
            "observed_device_map": observed_device_map,
            "summary_count": len(summaries),
            "observed_measured_logical_samples": observed_samples,
            "expected_measured_logical_samples": expected_samples,
            "raw_sample_relative_error": sample_error,
            "observed_measured_effective_tokens": observed_effective_tokens,
            "expected_measured_effective_tokens": expected_effective_tokens if bool(job["packing"]) else None,
            "effective_token_relative_error": effective_token_error,
            "checks": checks,
            "all_passed": all(checks.values()),
        }
    )
    return row


def evaluate() -> dict[str, Any]:
    jobs = read_jsonl(QUEUE)
    if (
        len(jobs) != EXPECTED_JOBS
        or {int(row["gpu_count"]) for row in jobs} != set(GPU_COUNTS)
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in jobs)
    ):
        raise ValueError("queue is not the exact Phase 0 canary")
    uuid_by_index = _expected_uuid_by_index()
    results = [_evaluate_job(job, uuid_by_index) for job in jobs]
    passed = [row for row in results if row["all_passed"]]
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_final_executor_canary_results/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue": {"path": str(QUEUE.resolve()), "sha256": sha256_file(QUEUE)},
        "completion": {
            "jobs": len(results),
            "passed": len(passed),
            "classifications": dict(Counter(str(row["classification"]) for row in results)),
        },
        "gates": {
            "all_eight_executor_canaries_passed": len(passed) == EXPECTED_JOBS,
            "phase_1_materialization_allowed": len(passed) == EXPECTED_JOBS,
            "automatic_phase_1_launch_allowed": False,
            "model_publication_allowed": False,
        },
        "jobs": results,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    return report


def main() -> None:
    report = evaluate()
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "completion": report["completion"],
                "gates": report["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["gates"]["all_eight_executor_canaries_passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
