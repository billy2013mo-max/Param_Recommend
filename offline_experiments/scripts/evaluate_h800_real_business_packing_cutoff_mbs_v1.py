#!/usr/bin/env python3
"""Evaluate the frozen 40-job real-business Packing×cutoff×MBS campaign."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, RESULTS_DIR, read_json, read_jsonl, sha256_file, sha256_json, write_json


CAMPAIGN_ID = "h800_real_business_packing_cutoff_mbs_20260804_v1"
QUEUE = MATRIX_DIR / "h800_real_business_packing_cutoff_mbs_v1.jsonl"
OUTPUT = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_results_v1.json"
MARKDOWN = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_results_v1.md"
ARMS = ("N-C-1", "N-C-k", "P-C-1", "N-kC-1", "P-kC-1")


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _cv(values: list[float]) -> float | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if len(values) > 1 and mean else 0.0


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _job_result(job: dict[str, Any]) -> dict[str, Any]:
    root = RESULTS_DIR / str(job["job_id"])
    status_path = root / "status.json"
    if not status_path.is_file():
        return {**job, "result_state": "missing", "classification": "missing"}
    status = read_json(status_path)
    row: dict[str, Any] = {
        **job,
        "result_state": "terminal",
        "classification": status.get("classification"),
        "calibration_eligible": status.get("calibration_eligible"),
        "gpu_mask": status.get("gpu_mask"),
        "wall_seconds": status.get("wall_seconds"),
        "execution_attempt_id": status.get("execution_attempt_id"),
    }
    summary_path = root / "metrics" / "summary.rank0.json"
    if status.get("classification") != "success" or not summary_path.is_file():
        return row
    summary = read_json(summary_path)
    totals = summary.get("measured_totals") or {}
    packing = ((summary.get("runtime_batch_evidence") or {}).get("packing") or {})
    ledger = summary.get("token_ledger_evidence") or {}
    measured_steps = int(summary.get("measured_steps") or 0)
    logical_samples = float(totals.get("logical_samples") or 0)
    row.update(
        {
            "measured_seconds": summary.get("measured_seconds"),
            "measured_steps": measured_steps,
            "logical_samples": logical_samples,
            "observed_logical_samples_per_step": (
                logical_samples / measured_steps if measured_steps else None
            ),
            "logical_samples_per_second": summary.get("logical_samples_per_second"),
            "effective_tokens_per_second": summary.get("effective_tokens_per_second"),
            "computed_tokens_per_second": summary.get("computed_tokens_per_second"),
            "computed_attention_pairs_per_second": (
                float(totals.get("computed_attention_token_pairs") or 0)
                / float(summary["measured_seconds"])
                if summary.get("measured_seconds")
                else None
            ),
            "effective_attention_pairs_per_second": (
                float(totals.get("effective_attention_token_pairs") or 0)
                / float(summary["measured_seconds"])
                if summary.get("measured_seconds")
                else None
            ),
            "max_allocated_gib": float(summary.get("max_allocated") or 0) / 2**30,
            "max_reserved_gib": float(summary.get("max_reserved") or 0) / 2**30,
            "token_ledger_authoritative": ledger.get("authoritative"),
            "prefetched_not_consumed_batches": ledger.get(
                "prefetched_not_consumed_batch_count"
            ),
            "packing_features": packing.get("features"),
            "packing_multi_sample_features": packing.get("multi_sample_features"),
            "packing_semantic_checks_passed": packing.get("semantic_checks_passed"),
            "packing_violations": packing.get("violations"),
        }
    )
    return row


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = [row for row in rows if row.get("classification") == "success"]
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in success if row.get(name) is not None]
    samples = values("logical_samples_per_second")
    effective = values("effective_tokens_per_second")
    computed = values("computed_tokens_per_second")
    allocated = values("max_allocated_gib")
    reserved = values("max_reserved_gib")
    return {
        "jobs": len(rows),
        "classifications": dict(Counter(str(row.get("classification")) for row in rows)),
        "successful_repeats": len(success),
        "gpu_masks": [row.get("gpu_mask") for row in rows],
        "logical_samples_per_second_mean": _mean(samples),
        "logical_samples_per_second_cv": _cv(samples),
        "effective_tokens_per_second_mean": _mean(effective),
        "effective_tokens_per_second_cv": _cv(effective),
        "computed_tokens_per_second_mean": _mean(computed),
        "max_allocated_gib_mean": _mean(allocated),
        "max_reserved_gib_mean": _mean(reserved),
        "authoritative_ledgers": sum(row.get("token_ledger_authoritative") is True for row in success),
        "packing_semantics_all_passed": (
            all(row.get("packing_semantic_checks_passed") is True for row in success)
            if success and bool(rows[0].get("packing"))
            else None
        ),
    }


def evaluate() -> dict[str, Any]:
    jobs = read_jsonl(QUEUE)
    if len(jobs) != 40 or any(row.get("campaign_id") != CAMPAIGN_ID for row in jobs):
        raise ValueError("queue is not the frozen 40-job campaign")
    results = [_job_result(job) for job in jobs]
    families = []
    for family_id in sorted({str(row["family_id"]) for row in jobs}):
        family_rows = [row for row in results if row["family_id"] == family_id]
        arm_summaries = {
            arm: _arm_summary([row for row in family_rows if row["arm_id"] == arm])
            for arm in ARMS
        }
        sample_rate = {
            arm: arm_summaries[arm]["logical_samples_per_second_mean"] for arm in ARMS
        }
        primary = _ratio(sample_rate["P-kC-1"], sample_rate["N-C-k"])
        ranking = sorted(
            (
                {"arm_id": arm, "logical_samples_per_second": rate}
                for arm, rate in sample_rate.items()
                if rate is not None
            ),
            key=lambda row: float(row["logical_samples_per_second"]),
            reverse=True,
        )
        representative = family_rows[0]
        families.append(
            {
                "family_id": family_id,
                "display_name": representative["display_name"],
                "base_cutoff_len": representative["base_cutoff_len"],
                "cutoff_scale": representative["cutoff_scale"],
                "target_gbs": representative["target_gbs"],
                "arms": arm_summaries,
                "ranking_by_logical_samples_per_second": ranking,
                "contrasts": {
                    "primary_packed_expanded_vs_unpacked_larger_mbs": primary,
                    "packing_at_fixed_C": _ratio(sample_rate["P-C-1"], sample_rate["N-C-1"]),
                    "larger_mbs_without_packing": _ratio(sample_rate["N-C-k"], sample_rate["N-C-1"]),
                    "larger_cutoff_with_packing": _ratio(sample_rate["P-kC-1"], sample_rate["P-C-1"]),
                    "larger_cutoff_without_packing": _ratio(sample_rate["N-kC-1"], sample_rate["N-C-1"]),
                },
                "primary_winner": (
                    "P-kC-1" if primary is not None and primary > 1 else
                    "N-C-k" if primary is not None else "incomplete"
                ),
            }
        )

    completed = sum(row.get("classification") != "missing" for row in results)
    successful = sum(row.get("classification") == "success" for row in results)
    eligible = sum(row.get("calibration_eligible") is True for row in results)
    all_complete = completed == 40
    all_terminal_usable = all_complete and all(
        row.get("classification") in {"success", "oom"} for row in results
    )
    winner_counts = Counter(row["primary_winner"] for row in families)
    report: dict[str, Any] = {
        "schema": "sft_h800_real_business_packing_cutoff_mbs_results/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue": {"path": str(QUEUE.resolve()), "sha256": sha256_file(QUEUE), "jobs": len(jobs)},
        "completion": {
            "terminal": completed,
            "success": successful,
            "oom": sum(row.get("classification") == "oom" for row in results),
            "failed": sum(row.get("classification") not in {"missing", "success", "oom"} for row in results),
            "missing": 40 - completed,
            "calibration_eligible": eligible,
            "all_complete": all_complete,
            "all_terminal_usable": all_terminal_usable,
        },
        "families": families,
        "winner_counts": dict(winner_counts),
        "interpretation_contract": {
            "primary_metric": "logical_samples_per_second",
            "primary_contrast": "P-kC-1 / N-C-k",
            "ratio_above_one": "neat Packing MBS1 with expanded cutoff is faster per logical sample",
            "memory_metric": "max_reserved_gib_mean is the safety-facing measure; max_allocated is diagnostic",
            "token_metrics": "effective TPS measures useful non-padding tokens; computed TPS measures model-facing work",
            "parallelism_caveat": (
                "Jobs ran concurrently on disjoint GPUs. GPU assignment and repeat CV are reported; "
                "small differences can include shared CPU/I/O contention."
            ),
            "publication_allowed": False,
        },
        "job_results": results,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    _write_markdown(report)
    return report


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _write_markdown(report: dict[str, Any]) -> None:
    completion = report["completion"]
    lines = [
        "# 真实业务 Packing × cutoff × MBS 实测结果",
        "",
        f"完成：{completion['terminal']}/40；成功：{completion['success']}；OOM：{completion['oom']}；失败：{completion['failed']}。",
        "",
        "主比较是 `P-kC-1 / N-C-k`：前者为 neat Packing、MBS=1、扩大 cutoff；后者为不开 Packing、保持 cutoff、把 MBS 放大 k 倍。比值大于 1 代表 Packing 路径更快。",
        "",
        "| 数据分布 | 最快 arm | 主比较倍数 | Packing@C | MBS 放大收益 | Packing 扩 cutoff | 非 Packing 扩 cutoff |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in report["families"]:
        ranking = family["ranking_by_logical_samples_per_second"]
        fastest = ranking[0]["arm_id"] if ranking else "—"
        contrasts = family["contrasts"]
        lines.append(
            "| {name} | {fastest} | {primary} | {pc} | {mbs} | {pkc} | {nkc} |".format(
                name=family["display_name"],
                fastest=fastest,
                primary=_fmt(contrasts["primary_packed_expanded_vs_unpacked_larger_mbs"]),
                pc=_fmt(contrasts["packing_at_fixed_C"]),
                mbs=_fmt(contrasts["larger_mbs_without_packing"]),
                pkc=_fmt(contrasts["larger_cutoff_with_packing"]),
                nkc=_fmt(contrasts["larger_cutoff_without_packing"]),
            )
        )
    lines.extend(
        (
            "",
            "## 各 arm 明细",
            "",
            "| 数据分布 | arm | 状态 | logical samples/s | CV | effective tok/s | reserved GiB | GPU |",
            "|---|---|---|---:|---:|---:|---:|---|",
        )
    )
    for family in report["families"]:
        for arm in ARMS:
            summary = family["arms"][arm]
            lines.append(
                "| {name} | {arm} | {states} | {sps} | {cv} | {tps} | {mem} | {gpus} |".format(
                    name=family["display_name"],
                    arm=arm,
                    states=json.dumps(summary["classifications"], ensure_ascii=False),
                    sps=_fmt(summary["logical_samples_per_second_mean"]),
                    cv=_fmt(summary["logical_samples_per_second_cv"]),
                    tps=_fmt(summary["effective_tokens_per_second_mean"], 1),
                    mem=_fmt(summary["max_reserved_gib_mean"], 2),
                    gpus=", ".join(str(value) for value in summary["gpu_masks"]),
                )
            )
    lines.extend(
        (
            "",
            "说明：logical samples/s 决定同一数据集跑完一轮的时间；effective tok/s 排除了 padding；computed tok/s 表示模型实际处理的 token。两次重复的 CV 超过 5% 时，细小排序不应视为稳定结论。并发运行会引入共享 CPU/I/O 争用，因此本结果用于路线取舍与校准，不直接作为最终发布验收。",
            "",
        )
    )
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    report = evaluate()
    print(json.dumps({"output": str(OUTPUT), "markdown": str(MARKDOWN), "completion": report["completion"], "winner_counts": report["winner_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
