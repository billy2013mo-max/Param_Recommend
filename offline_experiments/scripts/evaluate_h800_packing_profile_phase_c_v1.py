#!/usr/bin/env python3
"""Evaluate the 24-job Packing Phase-C interaction batch."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, read_json, read_jsonl, sha256_file, sha256_json, write_json
from evaluate_h800_packing_platform_v4_interactions_batch1_v1 import (
    _arm_summary,
    _geomean,
    _job_result,
)


CAMPAIGN_ID = "h800_packing_profile_phase_c_20260805_v1"
QUEUE = MATRIX_DIR / "h800_packing_profile_phase_c_v1.jsonl"
BASELINE = ARTIFACT_DIR / "h800_packing_profile_phase_b_results_v1.json"
OUTPUT = ARTIFACT_DIR / "h800_packing_profile_phase_c_results_v1.json"
MARKDOWN = ARTIFACT_DIR / "h800_packing_profile_phase_c_results_v1.md"


def _ratio(unpacked: dict[str, Any] | None, packed: dict[str, Any] | None, key: str) -> float | None:
    if not unpacked or not packed:
        return None
    denominator = unpacked.get(key)
    numerator = packed.get(key)
    if denominator is None or numerator is None or float(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)


def _baseline_by_family() -> dict[str, dict[str, Any]]:
    report = read_json(BASELINE)
    if report.get("gates", {}).get("phase_b_complete_without_extra_repeats") is not True:
        raise PermissionError("Phase-B baseline is not complete")
    return {str(row["family_id"]): row for row in report["families"]}


def evaluate() -> dict[str, Any]:
    jobs = read_jsonl(QUEUE)
    if len(jobs) != 24 or any(row.get("campaign_id") != CAMPAIGN_ID for row in jobs):
        raise ValueError("queue is not the frozen Phase-C batch")
    baseline = _baseline_by_family()
    results = [_job_result(job) for job in jobs]
    settings: list[dict[str, Any]] = []
    for setting_id in sorted({str(row["setting_id"]) for row in results}):
        subset = [row for row in results if row["setting_id"] == setting_id]
        unpacked_rows = [row for row in subset if not bool(row["packing"])]
        packed_rows = [row for row in subset if bool(row["packing"])]
        arms = {"unpacked": _arm_summary(unpacked_rows), "packed": _arm_summary(packed_rows)}
        logical_ratios: list[float] = []
        effective_ratios: list[float] = []
        computed_ratios: list[float] = []
        pairs: list[dict[str, Any]] = []
        for repeat in range(3):
            by_treatment = {
                bool(row["packing"]): row
                for row in subset
                if int(row["repeat"]) == repeat
            }
            logical = _ratio(by_treatment.get(False), by_treatment.get(True), "global_logical_samples_per_second")
            effective = _ratio(by_treatment.get(False), by_treatment.get(True), "global_effective_tokens_per_second")
            computed = _ratio(by_treatment.get(False), by_treatment.get(True), "global_computed_tokens_per_second")
            if logical is not None:
                logical_ratios.append(logical)
            if effective is not None:
                effective_ratios.append(effective)
            if computed is not None:
                computed_ratios.append(computed)
            pairs.append(
                {
                    "repeat": repeat,
                    "packed_over_unpacked_logical_samples_per_second": logical,
                    "packed_over_unpacked_effective_tokens_per_second": effective,
                    "packed_over_unpacked_computed_tokens_per_second": computed,
                }
            )
        representative = subset[0]
        baseline_family = baseline[str(representative["baseline_family_id"])]
        baseline_effect = baseline_family["matched_mechanism_effect"]
        new_logical = _geomean(logical_ratios)
        new_effective = _geomean(effective_ratios)
        new_computed = _geomean(computed_ratios)
        base_logical = baseline_effect["logical_samples_per_second_geometric_mean_ratio"]
        base_effective = baseline_effect["effective_tokens_per_second_geometric_mean_ratio"]
        base_computed = baseline_effect["computed_tokens_per_second_geometric_mean_ratio"]
        def contrast(new: float | None, old: float | None) -> dict[str, float | None]:
            if new is None or old is None or float(old) <= 0:
                return {"ratio_of_ratios": None, "log_ratio_of_ratios": None}
            value = float(new) / float(old)
            return {"ratio_of_ratios": value, "log_ratio_of_ratios": math.log(value)}
        settings.append(
            {
                "setting_id": setting_id,
                "baseline_family_id": representative["baseline_family_id"],
                "workload_id": representative["workload_id"],
                "cutoff_len": representative["cutoff_len"],
                "interaction_axis": representative["interaction_axis"],
                "zero": representative["zero"],
                "gc": representative["gc"],
                "arms": arms,
                "pairs": pairs,
                "packing_ratio": {
                    "logical_samples_per_second": new_logical,
                    "effective_tokens_per_second": new_effective,
                    "computed_tokens_per_second": new_computed,
                },
                "phase_b_baseline_packing_ratio": {
                    "logical_samples_per_second": base_logical,
                    "effective_tokens_per_second": base_effective,
                    "computed_tokens_per_second": base_computed,
                },
                "interaction_contrast": {
                    "logical_samples_per_second": contrast(new_logical, base_logical),
                    "effective_tokens_per_second": contrast(new_effective, base_effective),
                    "computed_tokens_per_second": contrast(new_computed, base_computed),
                },
                "reserved_memory": {
                    "unpacked_gib_mean": arms["unpacked"]["max_reserved_gib_mean"],
                    "packed_gib_mean": arms["packed"]["max_reserved_gib_mean"],
                },
            }
        )

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
    stable = all(
        arm["samples_per_second_cv"] is not None
        and float(arm["samples_per_second_cv"]) <= 0.05
        for setting in settings
        for arm in setting["arms"].values()
    )
    contrasts_available = all(
        setting["interaction_contrast"][metric]["ratio_of_ratios"] is not None
        for setting in settings
        for metric in ("logical_samples_per_second", "effective_tokens_per_second")
    )
    complete = bool(semantic_gate and stable and contrasts_available)
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_profile_phase_c_results/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue": {"path": str(QUEUE.resolve()), "sha256": sha256_file(QUEUE), "jobs": 24},
        "baseline": {"path": str(BASELINE.resolve()), "sha256": sha256_file(BASELINE)},
        "estimand": {
            "name": "packing_interaction_ratio_of_ratios",
            "definition": "(Packed/Unpacked under Phase-C arm) / (Packed/Unpacked under same-family Phase-B ZeRO-2/GC-on baseline).",
            "gc_long_cutoff_generalization_allowed": False,
            "best_branch_route_effect_claim_allowed": False,
        },
        "completion": {
            "terminal": len(terminal),
            "success": len(success),
            "oom": sum(row["classification"] == "oom" for row in results),
            "classifications": dict(Counter(str(row["classification"]) for row in results)),
        },
        "gates": {
            "packing_semantics_and_ledger_passed": semantic_gate,
            "all_treatment_cv_le_0p05": stable,
            "interaction_contrasts_available": contrasts_available,
            "phase_c_complete_without_extra_repeats": complete,
            "automatic_publication_allowed": False,
            "automatic_next_batch_allowed": False,
            "gc_long_cutoff_generalization_allowed": False,
        },
        "settings": settings,
        "job_results": results,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    fmt = lambda value, digits=3: "—" if value is None else f"{float(value):.{digits}f}"
    lines = [
        "# Packing Phase C：GC / ZeRO 交互标定",
        "",
        f"完成 {len(terminal)}/24，成功 {len(success)}/24；语义与 ledger：`{semantic_gate}`；全部 treatment CV≤5%：`{stable}`。",
        "",
        "交互量定义为 `(Phase-C 的 P/U) / (同画像 Phase-B ZeRO-2/GC-on 的 P/U)`；1 表示未观察到交互偏移。",
        "",
        "| setting | axis | cutoff | P/U effective | baseline P/U | ratio-of-ratios | U/P reserved GiB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for setting in settings:
        lines.append(
            f"| {setting['setting_id']} | {setting['interaction_axis']} | {setting['cutoff_len']} | "
            f"{fmt(setting['packing_ratio']['effective_tokens_per_second'])} | "
            f"{fmt(setting['phase_b_baseline_packing_ratio']['effective_tokens_per_second'])} | "
            f"{fmt(setting['interaction_contrast']['effective_tokens_per_second']['ratio_of_ratios'])} | "
            f"{fmt(setting['reserved_memory']['unpacked_gib_mean'], 2)}/{fmt(setting['reserved_memory']['packed_gib_mean'], 2)} |"
        )
    lines.extend(
        (
            "",
            "长 cutoff GC-off 组合未通过冻结显存预检，因此本批 GC 交互不得外推到 W7@20480 或 W3@40960。",
            "",
            "若任一 treatment CV>5%，只补该 treatment 的第4次；本批不自动发布或启动下一阶段。",
            "",
        )
    )
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = evaluate()
    print(json.dumps({"output": str(OUTPUT), "markdown": str(MARKDOWN), "completion": report["completion"], "gates": report["gates"]}, ensure_ascii=False, indent=2))
