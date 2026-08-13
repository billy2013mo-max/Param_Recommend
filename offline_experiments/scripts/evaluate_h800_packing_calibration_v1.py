#!/usr/bin/env python3
"""Evaluate the frozen H800 neat-Packing calibration without promoting it."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from common import ARTIFACT_DIR, RESULTS_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json


CAMPAIGN_ID = "h800_packing_calibration_20260803_v1"
QUEUE = ROOT / "matrix" / "h800_packing_calibration_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_packing_calibration_design_v1.json"
DECISIONS = ARTIFACT_DIR / "h800_packing_calibration_frozen_decisions_v1.json"
OUTPUT = ARTIFACT_DIR / "h800_packing_calibration_evaluation_v1.json"
T_CRITICAL_95_DF2 = 4.302652729749


def _summaries(job_id: str) -> list[dict[str, Any]]:
    return [
        read_json(path)
        for path in sorted((RESULTS_DIR / job_id / "metrics").glob("summary.rank*.json"))
    ]


def _packing_semantics(summary: dict[str, Any]) -> bool:
    evidence = ((summary.get("runtime_batch_evidence") or {}).get("packing") or {})
    violations = evidence.get("violations") or {}
    return bool(
        evidence.get("semantic_checks_passed") is True
        and int(evidence.get("multi_sample_features") or 0) > 0
        and all(int(value) == 0 for value in violations.values())
    )


def _ledger_valid(summary: dict[str, Any]) -> bool:
    evidence = summary.get("token_ledger_evidence") or {}
    return bool(
        evidence.get("schema") == "consumed_token_ledger/v1"
        and evidence.get("authoritative") is True
        and int(evidence.get("measured_batch_count") or 0)
        == int(summary["measured_steps"])
        * int(summary["metadata"]["gradient_accumulation_steps"])
    )


def _successful_measurement(
    job: dict[str, Any], status: dict[str, Any], summaries: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if status.get("classification") != "success" or len(summaries) != int(job["gpu_count"]):
        return None
    if any(summary.get("failure") is not None for summary in summaries):
        return None
    measured_steps = {int(summary.get("measured_steps") or 0) for summary in summaries}
    if measured_steps != {int(job["measure_steps"])}:
        return None
    seconds = max(float(summary["measured_seconds"]) for summary in summaries)
    logical_samples = sum(
        int(summary["measured_totals"]["logical_samples"]) for summary in summaries
    )
    effective_tokens = sum(
        int(summary["measured_totals"]["effective_tokens"]) for summary in summaries
    )
    computed_tokens = sum(
        int(summary["measured_totals"]["computed_tokens"]) for summary in summaries
    )
    optimizer_steps = next(iter(measured_steps))
    observed_sample_gbs = logical_samples / optimizer_steps
    return {
        "measured_seconds": seconds,
        "observed_sample_gbs": observed_sample_gbs,
        "sample_gbs_relative_error": abs(observed_sample_gbs - 64.0) / 64.0,
        "logical_samples_per_second": logical_samples / seconds,
        "effective_tokens_per_second": effective_tokens / seconds,
        "computed_tokens_per_second": computed_tokens / seconds,
        "max_reserved_bytes": max(
            int(summary.get("max_reserved") or 0) for summary in summaries
        ),
        "all_ranks_token_ledger_authoritative": all(
            _ledger_valid(summary) for summary in summaries
        ),
        "all_ranks_packing_semantics_passed": (
            all(_packing_semantics(summary) for summary in summaries)
            if job["packing"]
            else True
        ),
    }


def _paired_gain(values: list[float]) -> dict[str, Any]:
    logs = [math.log(value) for value in values]
    mean = statistics.fmean(logs)
    if len(logs) >= 2:
        half_width = T_CRITICAL_95_DF2 * statistics.stdev(logs) / math.sqrt(len(logs))
        interval = [math.exp(mean - half_width) - 1.0, math.exp(mean + half_width) - 1.0]
    else:
        interval = [None, None]
    return {
        "paired_repeats": len(values),
        "geometric_mean_gain": math.exp(mean) - 1.0,
        "gain_95_interval": interval,
        "individual_gains": [value - 1.0 for value in values],
    }


def _cv(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else None


def evaluate() -> dict[str, Any]:
    design = read_json(DESIGN)
    decisions = read_json(DECISIONS)
    jobs = read_jsonl(QUEUE)
    if (
        design.get("campaign_id") != CAMPAIGN_ID
        or design.get("queue", {}).get("sha256") != sha256_file(QUEUE)
        or len(jobs) != 24
    ):
        raise ValueError("Packing calibration frozen inputs drifted")
    decision_map = {
        row["family_id"]: row["decision"]["recommendation"]["decision"]
        for row in decisions["families"]
    }
    rows = []
    terminal = {"success", "oom", "failed", "timeout", "invalid_measurement"}
    for job in jobs:
        status_path = RESULTS_DIR / job["job_id"] / "status.json"
        status = read_json(status_path) if status_path.is_file() else {"classification": "missing"}
        summaries = _summaries(job["job_id"])
        measurement = _successful_measurement(job, status, summaries)
        rows.append(
            {
                "job_id": job["job_id"],
                "family_id": job["family_id"],
                "packing_pair_id": job["packing_pair_id"],
                "treatment": job["packing_treatment"],
                "repeat": job["repeat"],
                "classification": status.get("classification"),
                "terminal": status.get("classification") in terminal,
                "calibration_eligible": status.get("calibration_eligible"),
                "execution_attempt_id": status.get("execution_attempt_id"),
                "execution_fingerprint_sha256": status.get("execution_fingerprint_sha256"),
                "measurement": measurement,
            }
        )

    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    by_family_treatment: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        measurement = row["measurement"]
        if measurement is None:
            continue
        by_pair[row["packing_pair_id"]][row["treatment"]] = row
        by_family_treatment[(row["family_id"], row["treatment"])].append(
            float(measurement["logical_samples_per_second"])
        )
    family_reports = []
    for family_id in ("C1", "C2", "C3", "C4"):
        ratios = []
        pair_rows = []
        for pair_id, treatments in by_pair.items():
            if set(treatments) != {"unpacked", "packed"}:
                continue
            unpacked = treatments["unpacked"]
            if unpacked["family_id"] != family_id:
                continue
            packed = treatments["packed"]
            ratio = (
                packed["measurement"]["logical_samples_per_second"]
                / unpacked["measurement"]["logical_samples_per_second"]
            )
            ratios.append(ratio)
            pair_rows.append(
                {
                    "packing_pair_id": pair_id,
                    "repeat": unpacked["repeat"],
                    "packed_over_unpacked_samples_per_second": ratio,
                }
            )
        gain = _paired_gain(ratios) if ratios else None
        unpacked_mbs = next(
            int(job["mbs"])
            for job in jobs
            if job["family_id"] == family_id and not job["packing"]
        )
        required_gain = 0.10 if unpacked_mbs == 1 else 0.20
        lower = gain["gain_95_interval"][0] if gain else None
        static_on = decision_map[family_id] == "on"
        family_reports.append(
            {
                "family_id": family_id,
                "static_decision": decision_map[family_id],
                "unpacked_mbs": unpacked_mbs,
                "required_conservative_gain": required_gain,
                "pairs": pair_rows,
                "paired_gain": gain,
                "unpacked_samples_per_second_cv": _cv(
                    by_family_treatment[(family_id, "unpacked")]
                ),
                "packed_samples_per_second_cv": _cv(
                    by_family_treatment[(family_id, "packed")]
                ),
                "measured_benefit_gate_passed": lower is not None and lower >= required_gain,
                "static_false_enable": static_on
                and not (lower is not None and lower >= required_gain),
                "static_false_negative": (not static_on)
                and lower is not None
                and lower >= required_gain,
            }
        )

    successful = [row for row in rows if row["measurement"] is not None]
    packed_success = [row for row in successful if row["treatment"] == "packed"]
    all_terminal = all(row["terminal"] for row in rows)
    all_success = len(successful) == len(rows)
    semantic_pass = bool(packed_success) and all(
        row["measurement"]["all_ranks_packing_semantics_passed"]
        for row in packed_success
    )
    ledger_pass = bool(successful) and all(
        row["measurement"]["all_ranks_token_ledger_authoritative"]
        for row in successful
    )
    gbs_pass = bool(successful) and all(
        row["measurement"]["sample_gbs_relative_error"] <= 0.05
        for row in successful
    )
    complete_pairs = sum(len(report["pairs"]) for report in family_reports)
    repeat_noise_requires_extension = any(
        value is not None and value > 0.05
        for family in family_reports
        for value in (
            family["unpacked_samples_per_second_cv"],
            family["packed_samples_per_second_cv"],
        )
    )
    fit_ready = bool(
        all_terminal
        and all_success
        and semantic_pass
        and ledger_pass
        and gbs_pass
        and complete_pairs == 12
    )
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_calibration_evaluation/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_inputs": {
            "design": {
                "path": str(DESIGN.resolve()),
                "sha256": sha256_file(DESIGN),
                "report_sha256": design["report_sha256"],
            },
            "queue": {"path": str(QUEUE.resolve()), "sha256": sha256_file(QUEUE)},
            "decisions": {
                "path": str(DECISIONS.resolve()),
                "sha256": sha256_file(DECISIONS),
                "report_sha256": decisions["report_sha256"],
            },
        },
        "outcomes": {
            "jobs": len(rows),
            "terminal_jobs": sum(row["terminal"] for row in rows),
            "successful_measurements": len(successful),
            "packed_oom": sum(
                row["treatment"] == "packed" and row["classification"] == "oom"
                for row in rows
            ),
            "complete_pairs": complete_pairs,
        },
        "semantic_gates": {
            "all_packed_successes_semantic_pass": semantic_pass,
            "all_successes_consumed_ledger_authoritative": ledger_pass,
            "all_successes_sample_gbs_error_at_most_5pct": gbs_pass,
        },
        "families": family_reports,
        "rows": rows,
        "calibration_complete_for_joint_fit": fit_ready,
        "conditional_repeat_extension_required": repeat_noise_requires_extension,
        "prospective_acceptance_passed": False,
        "publication_allowed": False,
        "next_step": (
            "fit_shadow_packing_memory_and_throughput_heads"
            if fit_ready
            else "repair_or_complete_frozen_calibration_before_fit"
        ),
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
                "report_sha256": report["report_sha256"],
                "outcomes": report["outcomes"],
                "semantic_gates": report["semantic_gates"],
                "calibration_complete_for_joint_fit": report[
                    "calibration_complete_for_joint_fit"
                ],
                "conditional_repeat_extension_required": report[
                    "conditional_repeat_extension_required"
                ],
                "next_step": report["next_step"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
