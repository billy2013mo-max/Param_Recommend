#!/usr/bin/env python3
"""Correct the batch-1 GBS interpretation using pack-count distributions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any

from common import ARTIFACT_DIR, read_json, sha256_file, sha256_json, write_json
from packing_gbs_contract import derive_from_cached_packing_features


INPUT = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_combined_results_v1.json"
STATIC = ARTIFACT_DIR / "h800_packing_platform_v4_semantic_canary_static_features_v1.json"
OUTPUT = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_gbs_contract_correction_v2.json"
MARKDOWN = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_gbs_contract_correction_v2.md"
SETTING_TO_FAMILY = {
    "w1_gc_on": "w1_high_samples_per_pack",
    "w1_gc_off": "w1_high_samples_per_pack",
    "w4_zero2": "w4_broad_long_tail",
    "w4_zero3": "w4_broad_long_tail",
}


def evaluate() -> dict[str, Any]:
    combined = read_json(INPUT)
    static = read_json(STATIC)
    if (
        combined.get("completion", {}).get("combined_success") != 24
        or combined.get("gates", {}).get("packing_semantics_and_ledger_passed") is not True
        or len(static.get("rows") or []) != 2
    ):
        raise ValueError("GBS correction requires the complete semantic-valid 24-job batch")
    oracle_by_family = {
        str(row["family_id"]): row["calibration_oracle"]["decision"]["features"]
        for row in static["rows"]
    }
    jobs_by_setting: dict[str, list[dict[str, Any]]] = {}
    for row in combined["job_results"]:
        jobs_by_setting.setdefault(str(row["setting_id"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for setting in combined["settings"]:
        setting_id = str(setting["setting_id"])
        family_id = SETTING_TO_FAMILY[setting_id]
        features = oracle_by_family[family_id]
        contract = derive_from_cached_packing_features(
            features,
            target_gbs=64,
            data_parallel=int(setting["gpu_count"]),
            epsilon_gbs=0.10,
            maximum_center_relative_error=0.05,
        )
        packed_jobs = [row for row in jobs_by_setting[setting_id] if bool(row["packing"])]
        probe_gbs = [float(row["observed_sample_gbs"]) for row in packed_jobs]
        observed = statistics.fmean(probe_gbs)
        expected = float(contract["expected_epoch_sample_gbs"])
        rows.append(
            {
                "setting_id": setting_id,
                "family_id": family_id,
                "gpu_count": setting["gpu_count"],
                "zero": setting["zero"],
                "gc": setting["gc"],
                "cutoff_len": setting["cutoff_len"],
                "contract": contract,
                "probe_window": {
                    "warmup_steps": 2,
                    "measured_steps": 8,
                    "observed_sample_gbs_by_repeat": probe_gbs,
                    "observed_sample_gbs_mean": observed,
                    "relative_deviation_from_epoch_center": abs(observed - expected) / expected,
                    "interpretation": "deterministic short-window observation; not an estimator error against the full-epoch mean",
                },
                "packing_throughput_ratio": setting["packing_over_unpacked_geometric_mean"],
            }
        )

    center_pass = all(row["contract"]["gates"]["center_integer_representable"] for row in rows)
    controllable = all(row["contract"]["gates"]["gbs_controllable_at_ga_floor"] for row in rows)
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_platform_v4_gbs_contract_correction/v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "combined_results": {"path": str(INPUT.resolve()), "sha256": sha256_file(INPUT)},
            "static_packing_oracle": {"path": str(STATIC.resolve()), "sha256": sha256_file(STATIC)},
        },
        "correction": {
            "incorrect_interpretation": "treating an 8-step probe-window GBS as the full-epoch n_pack center",
            "correct_interpretation": "derive GA from the epoch mean; gate cutoff/DP safety with the cached empirical P99 pack count; report short-window GBS separately",
            "mean_estimator_was_wrong": False,
            "upper_tail_gate_was_missing": True,
        },
        "gates": {
            "execution_semantics_and_ledger_passed": True,
            "expected_epoch_center_within_5pct": center_pass,
            "all_candidates_gbs_controllable_at_ga_floor": controllable,
            "strict_target_gbs_coefficient_fit_allowed": center_pass and controllable,
            "distribution_labeled_shadow_calibration_allowed": True,
            "automatic_next_batch_allowed": False,
            "automatic_publication_allowed": False,
        },
        "rows": rows,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)

    fmt = lambda value, digits=2: f"{float(value):.{digits}f}"
    lines = [
        "# Packing batch-1 GBS 合同修正 v2",
        "",
        "修正结论：61.2 是 W4 完整 pack 集合的 epoch 平均 GBS；50.25 是固定 8-step 窗口观测。真正缺失的是 pack-count P99 的 GA-floor 安全门禁。",
        "",
        "| setting | mean n_pack | P99 n_pack | GA | expected epoch GBS | probe-window GBS | P99 global microstep | GBS可控 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        contract = row["contract"]
        lines.append(
            f"| {row['setting_id']} | {fmt(contract['samples_per_pack']['mean'])} | "
            f"{fmt(contract['samples_per_pack']['p99'])} | {contract['gradient_accumulation_steps']} | "
            f"{fmt(contract['expected_epoch_sample_gbs'])} | {fmt(row['probe_window']['observed_sample_gbs_mean'])} | "
            f"{fmt(contract['global_microstep_sample_gbs']['p99'])} | "
            f"{contract['gates']['gbs_controllable_at_ga_floor']} |"
        )
    lines.extend(
        (
            "",
            "W1 可继续作为严格目标-GBS交互标定；W4 保留为带分布标签的 shadow calibration，但在重新选择 cutoff/DP 前不得进入严格目标-GBS系数拟合。",
            "",
        )
    )
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = evaluate()
    print(json.dumps({"output": str(OUTPUT), "markdown": str(MARKDOWN), "gates": report["gates"]}, ensure_ascii=False, indent=2))
