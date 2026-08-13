#!/usr/bin/env python3
"""Evaluate formal Packing×GC/ZeRO interactions batch 1."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, RESULTS_DIR, read_json, read_jsonl, sha256_file, sha256_json, write_json


CAMPAIGN_ID = "h800_packing_platform_v4_interactions_batch1_20260804_v1"
QUEUE = MATRIX_DIR / "h800_packing_platform_v4_interactions_batch1_v1.jsonl"
OUTPUT = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_results_v1.json"
MARKDOWN = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_results_v1.md"


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _cv(values: list[float]) -> float | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if len(values) > 1 and mean else 0.0


def _geomean(values: list[float]) -> float | None:
    return math.exp(statistics.fmean(math.log(value) for value in values)) if values and all(value > 0 for value in values) else None


def _job_result(job: dict[str, Any]) -> dict[str, Any]:
    root = RESULTS_DIR / str(job["job_id"])
    status_path = root / "status.json"
    row: dict[str, Any] = {**job, "classification": "missing"}
    if not status_path.is_file():
        return row
    status = read_json(status_path)
    row.update({
        "classification": status.get("classification"),
        "calibration_eligible": status.get("calibration_eligible"),
        "gpu_mask": status.get("gpu_mask"),
        "wall_seconds": status.get("wall_seconds"),
        "execution_attempt_id": status.get("execution_attempt_id"),
    })
    if status.get("classification") != "success":
        return row
    summaries = [read_json(path) for path in sorted((root / "metrics").glob("summary.rank*.json"))]
    if len(summaries) != int(job["gpu_count"]):
        row["metrics_rank_count_exact"] = False
        return row
    measured_steps = {int(item.get("measured_steps") or 0) for item in summaries}
    if len(measured_steps) != 1 or next(iter(measured_steps)) <= 0:
        row["metrics_rank_count_exact"] = True
        row["measured_steps_consistent"] = False
        return row
    steps = next(iter(measured_steps))
    logical_samples = sum(float((item.get("measured_totals") or {}).get("logical_samples") or 0) for item in summaries)
    observed_gbs = logical_samples / steps
    row.update({
        "metrics_rank_count_exact": True,
        "measured_steps_consistent": True,
        "measured_steps": steps,
        "global_logical_samples": logical_samples,
        "observed_sample_gbs": observed_gbs,
        "observed_sample_gbs_relative_error": abs(observed_gbs - float(job["target_gbs"])) / float(job["target_gbs"]),
        "global_logical_samples_per_second": sum(float(item.get("logical_samples_per_second") or 0) for item in summaries),
        "global_effective_tokens_per_second": sum(float(item.get("effective_tokens_per_second") or 0) for item in summaries),
        "global_computed_tokens_per_second": sum(float(item.get("computed_tokens_per_second") or 0) for item in summaries),
        "max_allocated_gib": max(float(item.get("max_allocated") or 0) for item in summaries) / 2**30,
        "max_reserved_gib": max(float(item.get("max_reserved") or 0) for item in summaries) / 2**30,
        "authoritative_ledger_all_ranks": all((item.get("token_ledger_evidence") or {}).get("authoritative") is True for item in summaries),
        "packing_semantics_all_ranks": (
            all((((item.get("runtime_batch_evidence") or {}).get("packing") or {}).get("semantic_checks_passed")) is True for item in summaries)
            if bool(job["packing"]) else None
        ),
        "packing_violations_by_rank": [(((item.get("runtime_batch_evidence") or {}).get("packing") or {}).get("violations")) for item in summaries],
    })
    return row


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = [row for row in rows if row.get("classification") == "success"]
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in success if row.get(key) is not None]
    rates = values("global_logical_samples_per_second")
    return {
        "jobs": len(rows),
        "classifications": dict(Counter(str(row.get("classification")) for row in rows)),
        "successful_repeats": len(success),
        "samples_per_second_mean": _mean(rates),
        "samples_per_second_cv": _cv(rates),
        "effective_tokens_per_second_mean": _mean(values("global_effective_tokens_per_second")),
        "max_allocated_gib_mean": _mean(values("max_allocated_gib")),
        "max_reserved_gib_mean": _mean(values("max_reserved_gib")),
        "observed_sample_gbs_mean": _mean(values("observed_sample_gbs")),
        "observed_sample_gbs_error_max": max(values("observed_sample_gbs_relative_error"), default=None),
        "all_ledgers_authoritative": all(row.get("authoritative_ledger_all_ranks") is True for row in success) if success else False,
        "packing_semantics_all_passed": all(row.get("packing_semantics_all_ranks") is True for row in success) if success and bool(rows[0]["packing"]) else None,
    }


def evaluate() -> dict[str, Any]:
    jobs = read_jsonl(QUEUE)
    if len(jobs) != 24 or any(row.get("campaign_id") != CAMPAIGN_ID for row in jobs):
        raise ValueError("queue is not the frozen interaction batch")
    results = [_job_result(job) for job in jobs]
    settings = []
    ratio_by_setting: dict[str, float | None] = {}
    for setting_id in sorted({str(row["setting_id"]) for row in results}):
        subset = [row for row in results if row["setting_id"] == setting_id]
        arms = {
            "unpacked": _arm_summary([row for row in subset if not bool(row["packing"])]),
            "packed": _arm_summary([row for row in subset if bool(row["packing"])]),
        }
        pairs = []
        ratios = []
        for repeat in range(3):
            by_treatment = {bool(row["packing"]): row for row in subset if int(row["repeat"]) == repeat}
            unpacked = by_treatment.get(False)
            packed = by_treatment.get(True)
            ratio = None
            if unpacked and packed and unpacked.get("global_logical_samples_per_second") and packed.get("global_logical_samples_per_second"):
                ratio = float(packed["global_logical_samples_per_second"]) / float(unpacked["global_logical_samples_per_second"])
                ratios.append(ratio)
            pairs.append({"repeat": repeat, "packed_over_unpacked_samples_per_second": ratio})
        geometric_ratio = _geomean(ratios)
        ratio_by_setting[setting_id] = geometric_ratio
        representative = subset[0]
        settings.append({
            "setting_id": setting_id,
            "display_name": representative["display_name"],
            "interaction_axis": representative["interaction_axis"],
            "gpu_count": representative["gpu_count"],
            "zero": representative["zero"],
            "gc": representative["gc"],
            "cutoff_len": representative["cutoff_len"],
            "arms": arms,
            "pairs": pairs,
            "packing_over_unpacked_geometric_mean": geometric_ratio,
            "packing_over_unpacked_min": min(ratios, default=None),
            "packing_over_unpacked_max": max(ratios, default=None),
        })
    terminal = [row for row in results if row["classification"] != "missing"]
    success = [row for row in results if row["classification"] == "success"]
    packed_success = [row for row in success if bool(row["packing"])]
    semantic_gate = (
        len(success) == 24
        and all(row.get("metrics_rank_count_exact") is True and row.get("measured_steps_consistent") is True for row in success)
        and all(row.get("authoritative_ledger_all_ranks") is True for row in success)
        and len(packed_success) == 12
        and all(row.get("packing_semantics_all_ranks") is True for row in packed_success)
    )
    gc_on = ratio_by_setting.get("w1_gc_on")
    gc_off = ratio_by_setting.get("w1_gc_off")
    z2 = ratio_by_setting.get("w4_zero2")
    z3 = ratio_by_setting.get("w4_zero3")
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_platform_v4_interactions_batch1_results/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue": {"path": str(QUEUE.resolve()), "sha256": sha256_file(QUEUE), "jobs": 24},
        "completion": {
            "terminal": len(terminal),
            "success": len(success),
            "oom": sum(row["classification"] == "oom" for row in results),
            "classifications": dict(Counter(str(row["classification"]) for row in results)),
        },
        "gates": {
            "packing_semantics_and_ledger_passed": semantic_gate,
            "next_batch_allowed": semantic_gate,
            "automatic_publication_allowed": False,
        },
        "interaction_contrasts": {
            "packing_x_gc_log_ratio_difference_off_minus_on": math.log(gc_off) - math.log(gc_on) if gc_off and gc_on else None,
            "packing_x_zero_log_ratio_difference_z3_minus_z2": math.log(z3) - math.log(z2) if z3 and z2 else None,
            "interpretation": "positive means Packing's relative throughput advantage is larger at GC-off or ZeRO-3 respectively",
        },
        "settings": settings,
        "job_results": results,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    lines = [
        "# Packing platform-v4 交互标定第一批",
        "",
        f"完成 {len(terminal)}/24，成功 {len(success)}/24，语义与 ledger 门禁：`{semantic_gate}`。",
        "",
        "| setting | DP | ZeRO | GC | cutoff | Packing samples/s 几何倍数 | 最小～最大 | U/P reserved GiB |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    fmt = lambda value, digits=3: "—" if value is None else f"{float(value):.{digits}f}"
    for setting in settings:
        u = setting["arms"]["unpacked"]["max_reserved_gib_mean"]
        p = setting["arms"]["packed"]["max_reserved_gib_mean"]
        lines.append(
            f"| {setting['display_name']} | {setting['gpu_count']} | {setting['zero']} | {setting['gc']} | {setting['cutoff_len']} | "
            f"{fmt(setting['packing_over_unpacked_geometric_mean'])} | {fmt(setting['packing_over_unpacked_min'])}～{fmt(setting['packing_over_unpacked_max'])} | {fmt(u, 2)}/{fmt(p, 2)} |"
        )
    lines.extend(("", f"`Packing×GC` log-ratio contrast：{fmt(report['interaction_contrasts']['packing_x_gc_log_ratio_difference_off_minus_on'])}。", "", f"`Packing×ZeRO` log-ratio contrast：{fmt(report['interaction_contrasts']['packing_x_zero_log_ratio_difference_z3_minus_z2'])}。", "", "本批仅用于交互系数标定，不构成自动发布验收。", ""))
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    report = evaluate()
    print(json.dumps({"output": str(OUTPUT), "markdown": str(MARKDOWN), "completion": report["completion"], "gates": report["gates"], "interaction_contrasts": report["interaction_contrasts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
