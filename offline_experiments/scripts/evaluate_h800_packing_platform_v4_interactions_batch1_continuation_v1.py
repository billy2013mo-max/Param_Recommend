#!/usr/bin/env python3
"""Evaluate the combined 14 parent + 10 continuation Packing interaction runs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, read_json, read_jsonl, sha256_file, sha256_json, write_json
from evaluate_h800_packing_platform_v4_interactions_batch1_v1 import _arm_summary, _geomean, _job_result


PARENT_CAMPAIGN_ID = "h800_packing_platform_v4_interactions_batch1_20260804_v1"
CAMPAIGN_ID = "h800_packing_platform_v4_interactions_batch1_continuation_20260804_v1"
PARENT_QUEUE = MATRIX_DIR / "h800_packing_platform_v4_interactions_batch1_v1.jsonl"
QUEUE = MATRIX_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_v1.jsonl"
SELECTION = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_selection_v1.json"
OUTPUT = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_combined_results_v1.json"
MARKDOWN = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_combined_results_v1.md"


def evaluate() -> dict[str, Any]:
    parent_jobs = read_jsonl(PARENT_QUEUE)
    continuation_jobs = read_jsonl(QUEUE)
    selection = read_json(SELECTION)
    if (
        len(parent_jobs) != 24
        or any(row.get("campaign_id") != PARENT_CAMPAIGN_ID for row in parent_jobs)
        or len(continuation_jobs) != 10
        or any(row.get("campaign_id") != CAMPAIGN_ID for row in continuation_jobs)
        or selection.get("completed_success_count") != 14
        or selection.get("remaining_count") != 10
    ):
        raise ValueError("combined evaluator inputs do not match the frozen 14+10 continuation")
    completed_ids = {str(row["source_job_id"]) for row in selection["completed"]}
    continuation_by_source = {str(row["source_job_id"]): row for row in continuation_jobs}
    if len(completed_ids) != 14 or len(continuation_by_source) != 10 or completed_ids & set(continuation_by_source):
        raise ValueError("selection does not form a disjoint 14+10 partition")
    if completed_ids | set(continuation_by_source) != {str(row["job_id"]) for row in parent_jobs}:
        raise ValueError("selection does not cover the complete parent queue")

    results: list[dict[str, Any]] = []
    continuation_results: list[dict[str, Any]] = []
    for parent in parent_jobs:
        source_id = str(parent["job_id"])
        if source_id in completed_ids:
            row = _job_result(parent)
            row["execution_source"] = "parent"
            row["source_job_id"] = source_id
            row["effective_job_id"] = source_id
        else:
            continuation = continuation_by_source[source_id]
            row = _job_result(continuation)
            row["execution_source"] = "continuation"
            row["source_job_id"] = source_id
            row["effective_job_id"] = str(continuation["job_id"])
            continuation_results.append(row)
        row["original_design_job_id"] = source_id
        results.append(row)

    settings: list[dict[str, Any]] = []
    ratio_by_setting: dict[str, float | None] = {}
    for setting_id in sorted({str(row["setting_id"]) for row in results}):
        subset = [row for row in results if row["setting_id"] == setting_id]
        arms = {
            "unpacked": _arm_summary([row for row in subset if not bool(row["packing"])]),
            "packed": _arm_summary([row for row in subset if bool(row["packing"])]),
        }
        pairs: list[dict[str, Any]] = []
        ratios: list[float] = []
        for repeat in range(3):
            by_treatment = {
                bool(row["packing"]): row
                for row in subset
                if int(row["repeat"]) == repeat
            }
            unpacked = by_treatment.get(False)
            packed = by_treatment.get(True)
            ratio = None
            if (
                unpacked
                and packed
                and unpacked.get("global_logical_samples_per_second")
                and packed.get("global_logical_samples_per_second")
            ):
                ratio = float(packed["global_logical_samples_per_second"]) / float(
                    unpacked["global_logical_samples_per_second"]
                )
                ratios.append(ratio)
            pairs.append({"repeat": repeat, "packed_over_unpacked_samples_per_second": ratio})
        geometric_ratio = _geomean(ratios)
        ratio_by_setting[setting_id] = geometric_ratio
        representative = subset[0]
        settings.append(
            {
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
            }
        )

    terminal = [row for row in results if row["classification"] != "missing"]
    success = [row for row in results if row["classification"] == "success"]
    packed_success = [row for row in success if bool(row["packing"])]
    semantic_gate = (
        len(success) == 24
        and all(
            row.get("metrics_rank_count_exact") is True
            and row.get("measured_steps_consistent") is True
            for row in success
        )
        and all(row.get("authoritative_ledger_all_ranks") is True for row in success)
        and len(packed_success) == 12
        and all(row.get("packing_semantics_all_ranks") is True for row in packed_success)
    )
    gc_on = ratio_by_setting.get("w1_gc_on")
    gc_off = ratio_by_setting.get("w1_gc_off")
    z2 = ratio_by_setting.get("w4_zero2")
    z3 = ratio_by_setting.get("w4_zero3")
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_platform_v4_interactions_batch1_continuation_combined_results/v1",
        "campaign_id": CAMPAIGN_ID,
        "parent_campaign_id": PARENT_CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queues": {
            "parent": {"path": str(PARENT_QUEUE.resolve()), "sha256": sha256_file(PARENT_QUEUE), "jobs": 24},
            "continuation": {"path": str(QUEUE.resolve()), "sha256": sha256_file(QUEUE), "jobs": 10},
        },
        "selection": {"path": str(SELECTION.resolve()), "sha256": sha256_file(SELECTION)},
        "completion": {
            "combined_terminal": len(terminal),
            "combined_success": len(success),
            "combined_oom": sum(row["classification"] == "oom" for row in results),
            "combined_classifications": dict(Counter(str(row["classification"]) for row in results)),
            "continuation_terminal": sum(row["classification"] != "missing" for row in continuation_results),
            "continuation_success": sum(row["classification"] == "success" for row in continuation_results),
            "continuation_classifications": dict(Counter(str(row["classification"]) for row in continuation_results)),
        },
        "gates": {
            "packing_semantics_and_ledger_passed": semantic_gate,
            "interaction_batch_complete": semantic_gate,
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

    fmt = lambda value, digits=3: "—" if value is None else f"{float(value):.{digits}f}"
    lines = [
        "# Packing platform-v4 交互标定第一批（断点合并）",
        "",
        f"合并完成 {len(terminal)}/24，成功 {len(success)}/24；continuation 成功 {report['completion']['continuation_success']}/10；语义与 ledger 门禁：`{semantic_gate}`。",
        "",
        "| setting | DP | ZeRO | GC | cutoff | Packing samples/s 几何倍数 | 最小～最大 | U/P reserved GiB |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for setting in settings:
        unpacked = setting["arms"]["unpacked"]["max_reserved_gib_mean"]
        packed = setting["arms"]["packed"]["max_reserved_gib_mean"]
        lines.append(
            f"| {setting['display_name']} | {setting['gpu_count']} | {setting['zero']} | {setting['gc']} | {setting['cutoff_len']} | "
            f"{fmt(setting['packing_over_unpacked_geometric_mean'])} | {fmt(setting['packing_over_unpacked_min'])}～{fmt(setting['packing_over_unpacked_max'])} | {fmt(unpacked, 2)}/{fmt(packed, 2)} |"
        )
    lines.extend(
        (
            "",
            f"`Packing×GC` log-ratio contrast：{fmt(report['interaction_contrasts']['packing_x_gc_log_ratio_difference_off_minus_on'])}。",
            "",
            f"`Packing×ZeRO` log-ratio contrast：{fmt(report['interaction_contrasts']['packing_x_zero_log_ratio_difference_z3_minus_z2'])}。",
            "",
            "本批仅用于交互系数标定，不构成自动发布验收。",
            "",
        )
    )
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = evaluate()
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "markdown": str(MARKDOWN),
                "completion": report["completion"],
                "gates": report["gates"],
                "interaction_contrasts": report["interaction_contrasts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
