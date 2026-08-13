#!/usr/bin/env python3
"""Evaluate the four-job Packing platform-v4 semantic canary."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, RESULTS_DIR, read_json, read_jsonl, sha256_file, sha256_json, write_json


CAMPAIGN_ID = "h800_packing_platform_v4_semantic_canary_20260804_v1"
QUEUE = MATRIX_DIR / "h800_packing_platform_v4_semantic_canary_v1.jsonl"
STATIC = ARTIFACT_DIR / "h800_packing_platform_v4_semantic_canary_static_features_v1.json"
OUTPUT = ARTIFACT_DIR / "h800_packing_platform_v4_semantic_canary_results_v1.json"
MARKDOWN = ARTIFACT_DIR / "h800_packing_platform_v4_semantic_canary_results_v1.md"


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _job_result(job: dict[str, Any]) -> dict[str, Any]:
    root = RESULTS_DIR / str(job["job_id"])
    status_path = root / "status.json"
    row: dict[str, Any] = {**job, "classification": "missing"}
    if not status_path.is_file():
        return row
    status = read_json(status_path)
    row.update(
        {
            "classification": status.get("classification"),
            "calibration_eligible": status.get("calibration_eligible"),
            "gpu_mask": status.get("gpu_mask"),
            "wall_seconds": status.get("wall_seconds"),
            "execution_attempt_id": status.get("execution_attempt_id"),
        }
    )
    if status.get("classification") != "success":
        return row
    summaries = sorted((root / "metrics").glob("summary.rank*.json"))
    parsed = [read_json(path) for path in summaries]
    if len(parsed) != int(job["gpu_count"]):
        row["metrics_rank_count_exact"] = False
        return row
    row["metrics_rank_count_exact"] = True
    rank0 = parsed[0]
    totals = rank0.get("measured_totals") or {}
    measured_steps = int(rank0.get("measured_steps") or 0)
    logical_samples = float(totals.get("logical_samples") or 0)
    row.update(
        {
            "measured_steps": measured_steps,
            "logical_samples": logical_samples,
            "observed_logical_samples_per_step": (
                logical_samples / measured_steps if measured_steps else None
            ),
            "logical_samples_per_second": rank0.get("logical_samples_per_second"),
            "effective_tokens_per_second": rank0.get("effective_tokens_per_second"),
            "computed_tokens_per_second": rank0.get("computed_tokens_per_second"),
            "max_allocated_gib": max(float(item.get("max_allocated") or 0) for item in parsed) / 2**30,
            "max_reserved_gib": max(float(item.get("max_reserved") or 0) for item in parsed) / 2**30,
            "authoritative_ledger_all_ranks": all(
                (item.get("token_ledger_evidence") or {}).get("authoritative") is True
                for item in parsed
            ),
            "packing_semantics_all_ranks": (
                all(
                    (((item.get("runtime_batch_evidence") or {}).get("packing") or {}).get("semantic_checks_passed"))
                    is True
                    for item in parsed
                )
                if bool(job["packing"])
                else None
            ),
            "packing_violations_by_rank": [
                (((item.get("runtime_batch_evidence") or {}).get("packing") or {}).get("violations"))
                for item in parsed
            ],
        }
    )
    return row


def evaluate() -> dict[str, Any]:
    jobs = read_jsonl(QUEUE)
    if len(jobs) != 4 or any(row.get("campaign_id") != CAMPAIGN_ID for row in jobs):
        raise ValueError("queue is not the frozen four-job canary")
    results = [_job_result(job) for job in jobs]
    pairs = []
    for family_id in sorted({str(row["family_id"]) for row in results}):
        family = [row for row in results if row["family_id"] == family_id]
        by_packing = {bool(row["packing"]): row for row in family}
        unpacked = by_packing[False]
        packed = by_packing[True]
        pairs.append(
            {
                "family_id": family_id,
                "display_name": packed["display_name"],
                "gpu_count": packed["gpu_count"],
                "cutoff_len": packed["cutoff_len"],
                "expected_sample_gbs": packed["expected_sample_gbs"],
                "expected_sample_gbs_relative_error": packed["expected_sample_gbs_relative_error"],
                "packing_over_unpacked": {
                    "logical_samples_per_second": _ratio(
                        packed.get("logical_samples_per_second"),
                        unpacked.get("logical_samples_per_second"),
                    ),
                    "effective_tokens_per_second": _ratio(
                        packed.get("effective_tokens_per_second"),
                        unpacked.get("effective_tokens_per_second"),
                    ),
                    "max_allocated": _ratio(
                        packed.get("max_allocated_gib"),
                        unpacked.get("max_allocated_gib"),
                    ),
                    "max_reserved": _ratio(
                        packed.get("max_reserved_gib"),
                        unpacked.get("max_reserved_gib"),
                    ),
                },
                "unpacked": unpacked,
                "packed": packed,
            }
        )
    terminal = [row for row in results if row["classification"] != "missing"]
    success = [row for row in results if row["classification"] == "success"]
    packed_success = [row for row in success if bool(row["packing"])]
    semantic_gate = (
        len(success) == 4
        and all(row.get("metrics_rank_count_exact") is True for row in success)
        and all(row.get("authoritative_ledger_all_ranks") is True for row in success)
        and len(packed_success) == 2
        and all(row.get("packing_semantics_all_ranks") is True for row in packed_success)
    )
    static = read_json(STATIC)
    estimator_gate = all(
        float(row["calibration_oracle"]["n_pack_center_relative_error"]) <= 0.10
        for row in static["rows"]
    )
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_platform_v4_semantic_canary_results/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue": {"path": str(QUEUE.resolve()), "sha256": sha256_file(QUEUE), "jobs": 4},
        "completion": {
            "terminal": len(terminal),
            "success": len(success),
            "classifications": dict(Counter(str(row["classification"]) for row in results)),
        },
        "gates": {
            "packing_semantics_and_ledger_passed": semantic_gate,
            "dataprofile_n_pack_center_error_le_10pct": estimator_gate,
            "packed_successor_expansion_allowed": semantic_gate and estimator_gate,
            "automatic_publication_allowed": False,
        },
        "pairs": pairs,
        "job_results": results,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    lines = [
        "# Packing platform-v4 semantic canary",
        "",
        f"完成 {len(terminal)}/4，成功 {len(success)}/4，语义与 ledger 门禁：`{semantic_gate}`，DataProfile n_pack 门禁：`{estimator_gate}`。",
        "",
        "| 数据分布 | DP | cutoff | 期望 GBS | Packing samples/s 倍数 | effective tok/s 倍数 | allocated 倍数 | reserved 倍数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in pairs:
        ratios = pair["packing_over_unpacked"]
        fmt = lambda value: "—" if value is None else f"{float(value):.3f}"
        lines.append(
            f"| {pair['display_name']} | {pair['gpu_count']} | {pair['cutoff_len']} | "
            f"{pair['expected_sample_gbs']:.2f} | {fmt(ratios['logical_samples_per_second'])} | "
            f"{fmt(ratios['effective_tokens_per_second'])} | {fmt(ratios['max_allocated'])} | "
            f"{fmt(ratios['max_reserved'])} |"
        )
    lines.extend(("", "本 canary 只决定是否允许扩展后续 Packed 实验，不构成自动发布验收。", ""))
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    report = evaluate()
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "markdown": str(MARKDOWN),
                "completion": report["completion"],
                "gates": report["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
