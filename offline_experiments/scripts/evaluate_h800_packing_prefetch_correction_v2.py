#!/usr/bin/env python3
"""Correct only the frozen Packing canary's one-batch collator prefetch bias."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from common import ARTIFACT_DIR, MATRIX_DIR, RESULTS_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from static_packing_predictor import _load_lengths, _pack_in_worker_shards


CAMPAIGN_ID = "h800_packing_vl_canary_20260803_v1"
QUEUE = MATRIX_DIR / "h800_packing_semantic_canary_v1.jsonl"
V1_REPORT = ARTIFACT_DIR / "h800_packing_semantic_canary_acceptance_v1.json"
OUTPUT = ARTIFACT_DIR / "h800_packing_semantic_canary_acceptance_v2_prefetch_corrected.json"


def _attempt(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    status_path = RESULTS_DIR / job["job_id"] / "status.json"
    status = read_json(status_path)
    attempt = RESULTS_DIR / job["job_id"] / "attempts" / status["execution_attempt_id"]
    summary = read_json(attempt / "metrics" / "summary.rank0.json")
    config = yaml.safe_load((attempt / "runtime_config.yaml").read_text(encoding="utf-8"))
    return status, summary, config


def main() -> None:
    jobs = read_jsonl(QUEUE)
    v1 = read_json(V1_REPORT)
    if (
        len(jobs) != 2
        or v1.get("schema") != "sft_h800_packing_semantic_canary_acceptance/v1"
        or v1.get("campaign_id") != CAMPAIGN_ID
        or v1.get("all_passed") is not False
    ):
        raise ValueError("v1 Packing canary/report is not the exact correction source")
    failed_checks = {key for key, passed in v1["checks"].items() if not passed}
    if failed_checks != {"sample_gbs_error_at_most_5pct"}:
        raise ValueError(f"v2 cannot correct non-prefetch failures: {failed_checks}")

    corrections: list[dict[str, Any]] = []
    for job in jobs:
        status, summary, config = _attempt(job)
        if status.get("classification") != "success" or status.get("calibration_eligible") is not True:
            raise ValueError(f"job is not a successful complete canary: {status}")
        steps = int(summary["measured_steps"])
        ga = int(config["gradient_accumulation_steps"])
        consumed_physical_batches = steps * ga
        collated_physical_batches = int(
            (summary["runtime_batch_evidence"]["media"] or {}).get("collator_batches") or 0
        )
        if collated_physical_batches != consumed_physical_batches + 1:
            raise ValueError("correction only supports the observed one-batch prefetch bias")

        if job["packing"]:
            lengths = [
                min(length, int(job["cutoff_len"]) - 1)
                for length in _load_lengths(Path(job["dataset_profile_path"]), length_field="total_tokens")
            ]
            packs = _pack_in_worker_shards(
                lengths,
                capacity=int(job["cutoff_len"]) - 1,
                workers=int(config["preprocessing_num_workers"]),
            )
            generator = torch.Generator()
            generator.manual_seed(int(config["data_seed"]))
            order = torch.randperm(len(packs), generator=generator).tolist()
            logical_sizes = [len(packs[index]) for index in order]
            consumed_logical_samples = sum(logical_sizes[:consumed_physical_batches])
            prefetched_logical_samples = logical_sizes[consumed_physical_batches]
            runtime_packing = summary["runtime_batch_evidence"]["packing"]
            observed_examples = runtime_packing["examples"]
            expected_example_sizes = logical_sizes[: len(observed_examples)]
            if [int(row["logical_samples"]) for row in observed_examples] != expected_example_sizes:
                raise ValueError("frozen sampler reconstruction does not match runtime pack evidence")
            if (
                int(runtime_packing["features"]) != collated_physical_batches
                or int(runtime_packing["logical_samples"])
                != consumed_logical_samples + prefetched_logical_samples
            ):
                raise ValueError("runtime counters do not equal consumed plus one reconstructed prefetch")
        else:
            consumed_logical_samples = consumed_physical_batches
            prefetched_logical_samples = 1
            measured = summary["measured_totals"]
            if (
                int(measured["physical_batches"]) != collated_physical_batches
                or int(measured["logical_samples"])
                != consumed_logical_samples + prefetched_logical_samples
            ):
                raise ValueError("unpacked runtime counters do not show exactly one prefetched sample")

        observed_sample_gbs = consumed_logical_samples / steps
        target = float(job["target_gbs"])
        relative_error = abs(observed_sample_gbs - target) / target
        corrections.append(
            {
                "job_id": job["job_id"],
                "packing_treatment": job["packing_treatment"],
                "execution_attempt_id": status["execution_attempt_id"],
                "measured_steps": steps,
                "gradient_accumulation_steps": ga,
                "consumed_physical_batches": consumed_physical_batches,
                "collated_physical_batches": collated_physical_batches,
                "prefetched_not_consumed_physical_batches": 1,
                "consumed_logical_samples": consumed_logical_samples,
                "prefetched_not_consumed_logical_samples": prefetched_logical_samples,
                "observed_sample_gbs": observed_sample_gbs,
                "target_gbs": target,
                "relative_error": relative_error,
                "passed": relative_error <= 0.05,
            }
        )

    checks = {
        "v1_only_failed_prefetch_biased_sample_gbs_check": True,
        "both_jobs_success_and_semantics_passed_in_v1": all(
            passed for key, passed in v1["checks"].items() if key != "sample_gbs_error_at_most_5pct"
        ),
        "exactly_one_unconsumed_collator_prefetch_per_job": all(
            row["prefetched_not_consumed_physical_batches"] == 1 for row in corrections
        ),
        "frozen_sampler_reconstruction_bound_to_runtime_evidence": True,
        "corrected_sample_gbs_error_at_most_5pct": all(row["passed"] for row in corrections),
        "threshold_unchanged": True,
        "gpu_rerun_not_performed": True,
    }
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_semantic_canary_acceptance/v2_prefetch_corrected",
        "campaign_id": CAMPAIGN_ID,
        "correction_scope": "measurement_only_remove_exactly_one_collated_but_unconsumed_batch",
        "v1_report": {"path": str(V1_REPORT.resolve()), "sha256": sha256_file(V1_REPORT), "report_sha256": v1["report_sha256"]},
        "queue": {"path": str(QUEUE.resolve()), "sha256": sha256_file(QUEUE)},
        "checks": checks,
        "all_passed": all(checks.values()),
        "corrections": corrections,
        "scientific_status": "transparent_post_run_instrumentation_defect_correction_not_model_or_threshold_change",
        "fit_allowed": False,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
