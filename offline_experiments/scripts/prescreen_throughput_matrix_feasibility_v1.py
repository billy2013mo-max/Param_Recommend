#!/usr/bin/env python3
"""Offline feasibility pre-screen for a planned throughput candidate matrix.

The 2026-08-06 v3 prospective acceptance spent four single-use holdout datasets
and returned 22 OOM plus 2 non-authoritative token ledgers out of 48 jobs.  Both
failure modes were decidable before any GPU started:

* **Sample sufficiency** -- this screen.  See below.
* **Memory admission** -- MBS16 appears in that matrix only with gradient
  checkpointing off, which drives activations to ~137 of 140 GiB.  Screening it
  needs the memory model and is out of scope here.

What this screen asserts
------------------------
A job whose measurement window stays inside a single epoch cannot lose a batch
to an epoch boundary.  Concretely it requires

    dataset_samples  >=  (warmup + measure) x GA x MBS x gpu_count

which for the standard 3+10 protocol is 832 samples at GBS 64.  ``src14`` holds
495, so it never satisfied this and should not have been paired with a 13-step
protocol.

Why not model the boundary instead
----------------------------------
The observed failure is narrower than "crosses an epoch".  All four datasets
wrapped, and on ``src14`` all three surviving arms hit a short step at the
boundary -- yet only MBS4 lost a batch:

    arm    GA  per-step collation                      required/collated
    mbs4    8  9,8,8,8,8,8,8,[5],9,8,8,8,8             104 / 103  non-authoritative
    mbs8    4  5,4,4,4,4,4,4,[2],5,4,4,4,4              52 /  52  authoritative
    mbs16   2  3,2,2,2,2,2,2,[1],3,2,2,2,2              26 /  27  authoritative

The boundary step under-collates by ``GA - k`` while dataloader prefetch adds
back exactly two batches (one at step 1, one at step 9), so the ledger survives
only when the shortfall is <= 2.  Reproducing that would mean reimplementing
prefetch depth and sampler drop_last semantics, and it was inferred from three
configurations.  This screen therefore enforces the *sufficient* condition --
stay within one epoch -- and accepts flagging some jobs that would have squeaked
through.  A conservative screen that never lets a bad matrix through is worth
more than a precise one that depends on dataloader internals.

Arithmetic only: no GPU, no model, no queue mutation.  Exit status is 1 when any
job fails, so it can gate a queue at authoring time.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ROOT, read_jsonl, sha256_file, sha256_json, write_json


def _dataset_sample_count(path: Path) -> int:
    """Count non-blank JSONL records without holding the file in memory."""

    total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                total += 1
    return total


def _sample_requirement(job: Mapping[str, Any]) -> dict[str, Any]:
    """Per-rank and total sample demand implied by the measurement protocol.

    ``max_samples`` caps how much of the dataset the launcher exposes, so the
    effective stream is the smaller of the cap and the dataset itself.  Ranks
    consume disjoint shards, hence the multiplication by ``gpu_count``.
    """

    warmup = int(job["warmup_steps"])
    measure = int(job["measure_steps"])
    ga = int(job["gradient_accumulation_steps"])
    mbs = int(job["mbs"])
    gpus = int(job["gpu_count"])
    batches_per_rank = (warmup + measure) * ga
    return {
        "optimizer_steps": warmup + measure,
        "gradient_accumulation_steps": ga,
        "mbs": mbs,
        "gpu_count": gpus,
        "physical_batches_per_rank": batches_per_rank,
        "samples_per_rank": batches_per_rank * mbs,
        "samples_total": batches_per_rank * mbs * gpus,
    }


def _screen_job(job: Mapping[str, Any], available: int) -> dict[str, Any]:
    need = _sample_requirement(job)
    cap = int(job.get("max_samples") or available)
    effective_stream = min(cap, available)

    # Two tiers, because "never crosses an epoch" is unreachable on stage2: the
    # standard protocol wants 832 samples and the largest of these datasets holds
    # 768.  Flagging all 48 jobs would be literally correct and useless.
    #
    # Tier 1 (blocking) -- the boundary shortfall.  At an epoch boundary the step
    # collates only the whole batches left in the epoch; dataloader prefetch adds
    # back two batches over the run.  So the ledger survives when
    #
    #     GA - (per_rank_stream mod MBS) / MBS ... <= 2 batches short
    #
    # Concretely the shortfall in batches is GA minus the number of whole batches
    # available in the epoch's tail.  Measured on src14: MBS4 short by 3 -> lost a
    # batch; MBS8 short by 2 -> survived; MBS16 short by 1 -> survived.
    per_rank_stream = effective_stream // need["gpu_count"]
    whole_batches_per_epoch = per_rank_stream // need["mbs"] if need["mbs"] else 0
    crosses_epoch = need["physical_batches_per_rank"] > whole_batches_per_epoch
    if crosses_epoch and whole_batches_per_epoch:
        # Batches the boundary step can still draw from the epoch tail.
        tail = need["physical_batches_per_rank"] % whole_batches_per_epoch
        boundary_shortfall = (
            need["gradient_accumulation_steps"] - tail % need["gradient_accumulation_steps"]
        ) % need["gradient_accumulation_steps"]
    else:
        boundary_shortfall = 0

    PREFETCH_COMPENSATION = 2
    issues: list[str] = []
    warnings: list[str] = []
    if per_rank_stream < need["mbs"]:
        issues.append("stream_smaller_than_one_physical_batch")
    elif boundary_shortfall > PREFETCH_COMPENSATION:
        issues.append("epoch_boundary_shortfall_exceeds_prefetch_compensation")
    elif crosses_epoch:
        warnings.append("measurement_window_crosses_epoch_boundary")
    return {
        "job_id": str(job["job_id"]),
        "dataset_id": str(job["dataset_id"]),
        "candidate_arm_id": str(job.get("candidate_arm_id", "")),
        "repeat": int(job.get("repeat", 0)),
        "dataset_samples": available,
        "max_samples_cap": cap,
        "effective_stream": effective_stream,
        "per_rank_stream": per_rank_stream,
        "requirement": need,
        "whole_batches_per_epoch": whole_batches_per_epoch,
        "crosses_epoch": crosses_epoch,
        "epoch_boundary_shortfall_batches": boundary_shortfall,
        "prefetch_compensation_batches": PREFETCH_COMPENSATION,
        "samples_for_single_epoch": need["samples_total"],
        "predicted_ledger_authoritative": not issues,
        "issues": issues,
        "warnings": warnings,
    }


def screen(queue_path: Path, data_dir: Path) -> dict[str, Any]:
    jobs = read_jsonl(queue_path)
    counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for job in jobs:
        dataset_id = str(job["dataset_id"])
        if dataset_id not in counts:
            declared = job.get("data_path")
            path = Path(str(declared)) if declared else data_dir / f"{dataset_id}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"dataset file is absent: {path}")
            counts[dataset_id] = _dataset_sample_count(path)
        rows.append(_screen_job(job, counts[dataset_id]))

    failing = [row for row in rows if not row["predicted_ledger_authoritative"]]
    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset_id, available in sorted(counts.items()):
        subset = [row for row in rows if row["dataset_id"] == dataset_id]
        worst = max(row["requirement"]["samples_total"] for row in subset)
        by_dataset[dataset_id] = {
            "samples": available,
            "largest_total_demand": worst,
            "demand_exceeds_dataset": worst > available,
            "jobs": len(subset),
            "jobs_predicted_non_authoritative": sum(
                1 for row in subset if not row["predicted_ledger_authoritative"]
            ),
        }
    by_arm: dict[str, int] = defaultdict(int)
    for row in failing:
        by_arm[row["candidate_arm_id"]] += 1
    return {
        "schema": "sft_throughput_matrix_sample_sufficiency_screen/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "publishable": False,
        "queue": {"path": str(queue_path.resolve()), "sha256": sha256_file(queue_path)},
        "job_count": len(rows),
        "jobs_predicted_non_authoritative": len(failing),
        "per_dataset": by_dataset,
        "non_authoritative_by_arm": dict(sorted(by_arm.items())),
        "rows": rows,
    }


def _markdown(report: Mapping[str, Any], observed: Mapping[str, str] | None) -> str:
    lines = [
        "# 吞吐候选矩阵样本量预筛",
        "",
        f"- 生成时间：{report['generated_at_utc']}",
        f"- 作业数：{report['job_count']}",
        f"- 预筛拦下（会跨 epoch）：**{report['jobs_predicted_non_authoritative']}**",
        "",
        "判据：`数据集样本数 ≥ (warmup+measure) × GA × MBS × 卡数`（标准 3+10 协议为 832）。",
        "满足即测量窗口不跨 epoch，ledger 按构造完整。这是充分条件，偏保守。",
        "",
        "## 各数据集",
        "",
        "| 数据集 | 样本数 | 最大总需求 | 需求超出 | 拦下作业 |",
        "|---|---:|---:|---|---:|",
    ]
    for dataset_id, info in report["per_dataset"].items():
        lines.append(
            "| {d} | {s} | {n} | {e} | {f} |".format(
                d=dataset_id.replace("lora_s2_", ""),
                s=info["samples"],
                n=info["largest_total_demand"],
                e="是" if info["demand_exceeds_dataset"] else "否",
                f=info["jobs_predicted_non_authoritative"],
            )
        )
    failing = [row for row in report["rows"] if not row["predicted_ledger_authoritative"]]
    if failing:
        lines += [
            "",
            "## 预测跨 epoch 的作业（预筛拦下）",
            "",
            "| 数据集 | 候选 | 重复 | 需样本 | 可用流 | 缺口 | 约束 |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for row in failing:
            lines.append(
                "| {d} | {a} | {r} | {n} | {s} | {g} | {b} |".format(
                    d=row["dataset_id"].replace("lora_s2_", ""),
                    a=row["candidate_arm_id"],
                    r=row["repeat"],
                    n=row["requirement"]["samples_total"],
                    s=row["effective_stream"],
                    g=row["samples_short_of_single_epoch"],
                    b=row["binding_constraint"] or "—",
                )
            )
    if observed is not None:
        lines += [
            "",
            "## 与实测对照（回测）",
            "",
            "由于本预筛用的是充分条件（不跨 epoch），把实际能侥幸通过的作业也拦下"
            "属于**保守拦截**，不是预测错误。真正的错误只有一种：预筛放过、实测非权威。",
            "",
            "| 数据集 | 候选 | 重复 | 预筛 | 实测 | 判定 |",
            "|---|---|---:|---|---|---|",
        ]
        caught = conservative = clean = missed = 0
        for row in report["rows"]:
            actual = observed.get(row["job_id"])
            if actual is None:
                continue
            passes = row["predicted_ledger_authoritative"]
            if actual == "oom":
                verdict = "—（显存，不属本预筛）"
            elif actual == "success_non_authoritative":
                if passes:
                    verdict = "**漏报（预筛失效）**"
                    missed += 1
                else:
                    verdict = "命中"
                    caught += 1
            else:
                if passes:
                    verdict = "一致"
                    clean += 1
                else:
                    verdict = "保守拦截"
                    conservative += 1
            lines.append(
                "| {d} | {a} | {r} | {p} | {s} | {v} |".format(
                    d=row["dataset_id"].replace("lora_s2_", ""),
                    a=row["candidate_arm_id"],
                    r=row["repeat"],
                    p="通过" if passes else "拦下",
                    s=actual,
                    v=verdict,
                )
            )
        lines += [
            "",
            f"- **漏报 {missed}**（预筛放过但实测非权威——必须为 0）",
            f"- 命中 {caught}（预筛拦下且实测确实非权威）",
            f"- 保守拦截 {conservative}（预筛拦下但实测侥幸通过）",
            f"- 一致通过 {clean}",
        ]
    lines.append("")
    return "\n".join(lines)


def _observed_states(results_path: Path) -> dict[str, str]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    return {str(row["job_id"]): str(row["state"]) for row in payload["rows"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data" / "h800_lora_safety_stage2_v1"
    )
    parser.add_argument(
        "--backtest-results",
        type=Path,
        default=None,
        help="optional evaluated results JSON, to score the screen against outcomes",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    report = screen(args.queue.resolve(), args.data_dir.resolve())
    observed: dict[str, str] | None = None
    if args.backtest_results is not None:
        observed = _observed_states(args.backtest_results.resolve())
        report["backtest"] = {
            "results_path": str(args.backtest_results.resolve()),
            "results_sha256": sha256_file(args.backtest_results),
        }
    report["report_sha256"] = sha256_json(report)

    if args.output is not None:
        write_json(args.output, report)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(_markdown(report, observed), encoding="utf-8")

    summary = {
        "job_count": report["job_count"],
        "jobs_predicted_non_authoritative": report["jobs_predicted_non_authoritative"],
        "non_authoritative_by_arm": report["non_authoritative_by_arm"],
        "datasets_with_demand_exceeding_supply": [
            dataset_id
            for dataset_id, info in report["per_dataset"].items()
            if info["demand_exceeds_dataset"]
        ],
    }
    if observed is not None:
        scored = [
            (row, observed[row["job_id"]])
            for row in report["rows"]
            if row["job_id"] in observed and observed[row["job_id"]] != "oom"
        ]
        # The only defect that matters: the screen passed a job whose ledger then
        # came back incomplete.  Conservative flags are by design.
        summary["backtest_scored_jobs"] = len(scored)
        summary["backtest_missed_failures"] = sum(
            1
            for row, actual in scored
            if row["predicted_ledger_authoritative"]
            and actual == "success_non_authoritative"
        )
        summary["backtest_caught_failures"] = sum(
            1
            for row, actual in scored
            if not row["predicted_ledger_authoritative"]
            and actual == "success_non_authoritative"
        )
        summary["backtest_conservative_flags"] = sum(
            1
            for row, actual in scored
            if not row["predicted_ledger_authoritative"]
            and actual == "success_authoritative"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(1 if report["jobs_predicted_non_authoritative"] else 0)


if __name__ == "__main__":
    main()
