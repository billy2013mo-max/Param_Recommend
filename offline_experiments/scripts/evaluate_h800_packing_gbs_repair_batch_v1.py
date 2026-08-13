#!/usr/bin/env python3
"""Evaluate the 18-job W4 Packing GBS-contract repair batch."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
import statistics
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, read_jsonl, sha256_file, sha256_json, write_json
from evaluate_h800_packing_platform_v4_interactions_batch1_v1 import _arm_summary, _geomean, _job_result


CAMPAIGN_ID = "h800_packing_gbs_repair_batch_20260804_v1"
QUEUE = MATRIX_DIR / "h800_packing_gbs_repair_batch_v1.jsonl"
OUTPUT = ARTIFACT_DIR / "h800_packing_gbs_repair_batch_results_v1.json"
MARKDOWN = ARTIFACT_DIR / "h800_packing_gbs_repair_batch_results_v1.md"


def evaluate() -> dict[str, Any]:
    jobs = read_jsonl(QUEUE)
    if len(jobs) != 18 or any(row.get("campaign_id") != CAMPAIGN_ID for row in jobs):
        raise ValueError("queue is not the frozen repair batch")
    results = [_job_result(job) for job in jobs]
    settings = []
    for setting_id in sorted({str(row["setting_id"]) for row in results}):
        subset = [row for row in results if row["setting_id"] == setting_id]
        arms = {
            "unpacked": _arm_summary([row for row in subset if not bool(row["packing"])]),
            "packed": _arm_summary([row for row in subset if bool(row["packing"])]),
        }
        pairs = []
        ratios = []
        for repeat in range(3):
            treatment = {bool(row["packing"]): row for row in subset if int(row["repeat"]) == repeat}
            unpacked, packed = treatment.get(False), treatment.get(True)
            ratio = None
            if unpacked and packed and unpacked.get("global_logical_samples_per_second") and packed.get("global_logical_samples_per_second"):
                ratio = float(packed["global_logical_samples_per_second"]) / float(unpacked["global_logical_samples_per_second"])
                ratios.append(ratio)
            pairs.append({"repeat": repeat, "packed_over_unpacked_samples_per_second": ratio})
        representative = subset[0]
        packed_probe = [float(row["observed_sample_gbs"]) for row in subset if bool(row["packing"]) and row.get("observed_sample_gbs") is not None]
        settings.append({
            "setting_id": setting_id,
            "display_name": representative["display_name"],
            "gpu_count": representative["gpu_count"],
            "target_gbs": representative["target_gbs"],
            "zero": representative["zero"],
            "cutoff_len": representative["cutoff_len"],
            "gradient_accumulation_steps_packed": next(int(row["gradient_accumulation_steps"]) for row in subset if bool(row["packing"])),
            "expected_epoch_sample_gbs_packed": next(float(row["expected_epoch_sample_gbs"]) for row in subset if bool(row["packing"])),
            "observed_probe_window_sample_gbs_packed_mean": statistics.fmean(packed_probe) if packed_probe else None,
            "n_pack_step_p99": representative["n_pack_step_p99"],
            "arms": arms,
            "pairs": pairs,
            "packing_over_unpacked_geometric_mean": _geomean(ratios),
            "packing_over_unpacked_min": min(ratios, default=None),
            "packing_over_unpacked_max": max(ratios, default=None),
        })
    terminal = [row for row in results if row["classification"] != "missing"]
    success = [row for row in results if row["classification"] == "success"]
    packed_success = [row for row in success if bool(row["packing"])]
    semantic_gate = (
        len(success) == 18
        and all(row.get("metrics_rank_count_exact") is True and row.get("measured_steps_consistent") is True for row in success)
        and all(row.get("authoritative_ledger_all_ranks") is True for row in success)
        and len(packed_success) == 9
        and all(row.get("packing_semantics_all_ranks") is True for row in packed_success)
    )
    z2 = next((row for row in settings if row["setting_id"] == "w4_dp2_g128_zero2"), None)
    z3 = next((row for row in settings if row["setting_id"] == "w4_dp2_g128_zero3"), None)
    z2_ratio = z2 and z2["packing_over_unpacked_geometric_mean"]
    z3_ratio = z3 and z3["packing_over_unpacked_geometric_mean"]
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_gbs_repair_batch_results/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue": {"path": str(QUEUE.resolve()), "sha256": sha256_file(QUEUE), "jobs": 18},
        "completion": {
            "terminal": len(terminal), "success": len(success),
            "oom": sum(row["classification"] == "oom" for row in results),
            "classifications": dict(Counter(str(row["classification"]) for row in results)),
        },
        "gates": {
            "packing_semantics_and_ledger_passed": semantic_gate,
            "repair_batch_complete": semantic_gate,
            "strict_target_gbs_fit_only_allowed": semantic_gate,
            "automatic_publication_allowed": False,
            "automatic_next_batch_allowed": False,
        },
        "interaction_contrasts": {
            "packing_x_zero_log_ratio_difference_z3_minus_z2": math.log(z3_ratio) - math.log(z2_ratio) if z2_ratio and z3_ratio else None,
        },
        "settings": settings,
        "job_results": results,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    fmt = lambda value, digits=3: "—" if value is None else f"{float(value):.{digits}f}"
    lines = [
        "# W4 Packing GBS 合同修复补测", "",
        f"完成 {len(terminal)}/18，成功 {len(success)}/18；语义与 ledger：`{semantic_gate}`。", "",
        "| setting | DP | target GBS | ZeRO | GA(P) | expected epoch GBS(P) | probe window GBS(P) | Packing samples/s 倍数 | U/P reserved GiB |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for setting in settings:
        u = setting["arms"]["unpacked"]["max_reserved_gib_mean"]
        p = setting["arms"]["packed"]["max_reserved_gib_mean"]
        lines.append(
            f"| {setting['display_name']} | {setting['gpu_count']} | {setting['target_gbs']} | {setting['zero']} | "
            f"{setting['gradient_accumulation_steps_packed']} | {fmt(setting['expected_epoch_sample_gbs_packed'])} | "
            f"{fmt(setting['observed_probe_window_sample_gbs_packed_mean'])} | {fmt(setting['packing_over_unpacked_geometric_mean'])} | "
            f"{fmt(u,2)}/{fmt(p,2)} |"
        )
    lines.extend(("", "本批只作为严格 target-GBS 的 fit-only 修复证据，不自动发布或启动下一批。", ""))
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps({"output": str(OUTPUT), "markdown": str(MARKDOWN), "completion": result["completion"], "gates": result["gates"]}, ensure_ascii=False, indent=2))
