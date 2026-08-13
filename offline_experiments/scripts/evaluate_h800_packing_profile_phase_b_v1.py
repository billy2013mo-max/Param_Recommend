#!/usr/bin/env python3
"""Evaluate the 36-job W3/W5/W7/W8 Packing profile Phase-B batch."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import statistics
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, read_jsonl, sha256_file, sha256_json, write_json
from evaluate_h800_packing_platform_v4_interactions_batch1_v1 import (
    _arm_summary,
    _geomean,
    _job_result,
)


CAMPAIGN_ID = "h800_packing_profile_phase_b_20260805_v1"
QUEUE = MATRIX_DIR / "h800_packing_profile_phase_b_v1.jsonl"
OUTPUT = ARTIFACT_DIR / "h800_packing_profile_phase_b_results_v1.json"
MARKDOWN = ARTIFACT_DIR / "h800_packing_profile_phase_b_results_v1.md"


def _ratio(
    unpacked: dict[str, Any] | None,
    packed: dict[str, Any] | None,
    key: str,
) -> float | None:
    if not unpacked or not packed:
        return None
    denominator = unpacked.get(key)
    numerator = packed.get(key)
    if denominator is None or numerator is None or float(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)


def evaluate() -> dict[str, Any]:
    jobs = read_jsonl(QUEUE)
    if len(jobs) != 36 or any(row.get("campaign_id") != CAMPAIGN_ID for row in jobs):
        raise ValueError("queue is not the frozen Phase-B batch")
    results = [_job_result(job) for job in jobs]
    families = []
    for family_id in sorted({str(row["family_id"]) for row in results}):
        subset = [row for row in results if row["family_id"] == family_id]
        unpacked_rows = [row for row in subset if not bool(row["packing"])]
        packed_rows = [row for row in subset if bool(row["packing"])]
        arms = {
            "unpacked": _arm_summary(unpacked_rows),
            "packed": _arm_summary(packed_rows),
        }
        pairs = []
        sample_ratios = []
        effective_token_ratios = []
        computed_token_ratios = []
        for repeat in range(3):
            by_treatment = {
                bool(row["packing"]): row
                for row in subset
                if int(row["repeat"]) == repeat
            }
            unpacked = by_treatment.get(False)
            packed = by_treatment.get(True)
            sample_ratio = _ratio(
                unpacked, packed, "global_logical_samples_per_second"
            )
            effective_ratio = _ratio(
                unpacked, packed, "global_effective_tokens_per_second"
            )
            computed_ratio = _ratio(
                unpacked, packed, "global_computed_tokens_per_second"
            )
            if sample_ratio is not None:
                sample_ratios.append(sample_ratio)
            if effective_ratio is not None:
                effective_token_ratios.append(effective_ratio)
            if computed_ratio is not None:
                computed_token_ratios.append(computed_ratio)
            pairs.append(
                {
                    "repeat": repeat,
                    "packed_over_unpacked_logical_samples_per_second": sample_ratio,
                    "packed_over_unpacked_effective_tokens_per_second": effective_ratio,
                    "packed_over_unpacked_computed_tokens_per_second": computed_ratio,
                }
            )
        representative = subset[0]
        packed_probe_gbs = [
            float(row["observed_sample_gbs"])
            for row in packed_rows
            if row.get("observed_sample_gbs") is not None
        ]
        u_memory = arms["unpacked"]["max_reserved_gib_mean"]
        p_memory = arms["packed"]["max_reserved_gib_mean"]
        families.append(
            {
                "family_id": family_id,
                "workload_id": representative["workload_id"],
                "profile_role": representative["profile_family_id"],
                "cutoff_len": representative["cutoff_len"],
                "packed_gradient_accumulation_steps": next(
                    int(row["gradient_accumulation_steps"]) for row in packed_rows
                ),
                "expected_epoch_sample_gbs_packed": next(
                    float(row["expected_epoch_sample_gbs"]) for row in packed_rows
                ),
                "observed_probe_window_sample_gbs_packed_mean": (
                    statistics.fmean(packed_probe_gbs) if packed_probe_gbs else None
                ),
                "n_pack_step_p99": representative["n_pack_step_p99"],
                "arms": arms,
                "pairs": pairs,
                "matched_mechanism_effect": {
                    "logical_samples_per_second_geometric_mean_ratio": _geomean(
                        sample_ratios
                    ),
                    "effective_tokens_per_second_geometric_mean_ratio": _geomean(
                        effective_token_ratios
                    ),
                    "computed_tokens_per_second_geometric_mean_ratio": _geomean(
                        computed_token_ratios
                    ),
                    "logical_samples_per_second_min_ratio": min(
                        sample_ratios, default=None
                    ),
                    "logical_samples_per_second_max_ratio": max(
                        sample_ratios, default=None
                    ),
                },
                "reserved_memory": {
                    "unpacked_gib_mean": u_memory,
                    "packed_gib_mean": p_memory,
                    "packed_minus_unpacked_gib": (
                        float(p_memory) - float(u_memory)
                        if p_memory is not None and u_memory is not None
                        else None
                    ),
                },
            }
        )

    terminal = [row for row in results if row["classification"] != "missing"]
    success = [row for row in results if row["classification"] == "success"]
    packed_success = [row for row in success if bool(row["packing"])]
    semantic_gate = (
        len(success) == 36
        and all(
            row.get("metrics_rank_count_exact") is True
            and row.get("measured_steps_consistent") is True
            for row in success
        )
        and all(
            row.get("authoritative_ledger_all_ranks") is True for row in success
        )
        and len(packed_success) == 18
        and all(
            row.get("packing_semantics_all_ranks") is True
            for row in packed_success
        )
    )
    stable_treatments = all(
        arm["samples_per_second_cv"] is not None
        and float(arm["samples_per_second_cv"]) <= 0.05
        for family in families
        for arm in family["arms"].values()
    )
    expected_gbs_fields_present = all(
        family["expected_epoch_sample_gbs_packed"] is not None
        and family["observed_probe_window_sample_gbs_packed_mean"] is not None
        for family in families
    )
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_profile_phase_b_results/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue": {
            "path": str(QUEUE.resolve()),
            "sha256": sha256_file(QUEUE),
            "jobs": 36,
        },
        "estimand": {
            "name": "matched_packing_mechanism_effect_at_mbs1",
            "definition": (
                "Packing over Unpacked with common model/data/cutoff/DP/ZeRO/GC/"
                "target-GBS/MBS and Packing-specific GA from the exact pack-count mean."
            ),
            "best_branch_route_effect_claim_allowed": False,
        },
        "completion": {
            "terminal": len(terminal),
            "success": len(success),
            "oom": sum(row["classification"] == "oom" for row in results),
            "classifications": dict(
                Counter(str(row["classification"]) for row in results)
            ),
        },
        "gates": {
            "packing_semantics_and_ledger_passed": semantic_gate,
            "all_treatment_cv_le_0p05": stable_treatments,
            "expected_epoch_and_probe_window_gbs_reported_separately": expected_gbs_fields_present,
            "phase_b_complete_without_extra_repeats": bool(
                semantic_gate and stable_treatments and expected_gbs_fields_present
            ),
            "automatic_publication_allowed": False,
            "automatic_next_batch_allowed": False,
            "best_branch_route_effect_claim_allowed": False,
        },
        "families": families,
        "job_results": results,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)

    fmt = lambda value, digits=3: "—" if value is None else f"{float(value):.{digits}f}"
    lines = [
        "# Packing Phase B：跨画像 matched 机制效应",
        "",
        f"完成 {len(terminal)}/36，成功 {len(success)}/36；语义与 ledger：`{semantic_gate}`；全部 treatment CV≤5%：`{stable_treatments}`。",
        "",
        "本批固定 MBS=1，估计 Packing 机制效应；不能单独解释为两个分支各自调优后的最终 route-effect。",
        "",
        "| family | cutoff | GA(P) | expected/probe GBS(P) | logical samples/s P/U | effective tokens/s P/U | U/P reserved GiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in families:
        effect = family["matched_mechanism_effect"]
        memory = family["reserved_memory"]
        lines.append(
            f"| {family['family_id']} | {family['cutoff_len']} | "
            f"{family['packed_gradient_accumulation_steps']} | "
            f"{fmt(family['expected_epoch_sample_gbs_packed'])}/{fmt(family['observed_probe_window_sample_gbs_packed_mean'])} | "
            f"{fmt(effect['logical_samples_per_second_geometric_mean_ratio'])} | "
            f"{fmt(effect['effective_tokens_per_second_geometric_mean_ratio'])} | "
            f"{fmt(memory['unpacked_gib_mean'], 2)}/{fmt(memory['packed_gib_mean'], 2)} |"
        )
    lines.extend(
        (
            "",
            "若任一 treatment CV>5%，只补该 treatment 的第4次；不允许单侧挑选最好结果。",
            "",
            "本批为 fit-only，不自动发布，也不自动启动 Phase C。",
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
            },
            ensure_ascii=False,
            indent=2,
        )
    )
