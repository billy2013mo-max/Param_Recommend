#!/usr/bin/env python3
"""Derive an empirically safe operating region for lora / 2-GPU / ZeRO-3 / no-GC.

Why not use the memory model
----------------------------
V5 has no admission head for this mechanism, and
``fit_h800_m1_new_mechanism_admission_heads_v1.py`` cannot fit one: with the
held-out business sources removed the mechanism no longer clears the planning
thresholds.  A throughput acceptance still has to launch ZeRO-3 candidates, and
launching them blind is what cost 14 of 16 ZeRO-3 jobs and four single-use
holdout datasets on 2026-08-06.

So instead of predicting the boundary, this script bounds it from measurements:
which (cutoff, MBS) cells have been observed to stay clear of the safety line on
*every* source tested, and which have not.  A cell is only called safe when no
observation of it has ever come close, which makes the region a floor rather than
an estimate.

Three rules the classification obeys
------------------------------------
1. **Crossing the line counts as unsafe even without an OOM.**  ``src13`` at MBS2
   completed while reserving 102% of the safety limit.  Calling that safe because
   the process survived would be reading luck as headroom.
2. **A crash's reported peak understates the truth.**  ``src13`` at MBS4 OOMed
   with ``max_reserved`` at 80% of the line, because the allocation that failed is
   never recorded.  The peak of an OOM row is therefore a lower bound and is never
   used to argue safety.
3. **The worst rank decides.**  ZeRO-3 shards unevenly -- ``src13`` MBS2 measured
   116.5 GiB on rank 0 and 135.2 GiB on rank 1.  Averaging would have hidden the
   crossing entirely.

Analysis only: reads terminal results, writes a diagnostic report, touches no
frozen artifact and no GPU.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_json, write_json

RESULTS = ROOT / "results"
GIB = float(1 << 30)
SAFE_LIMIT_GIB = 132.839

# A cell must sit at or below this fraction of the safety line on every source
# before it is called safe.  0.85 leaves room for the source-to-source spread the
# data actually shows: at 14B / cutoff 4096 / MBS4 one source measured 89% and
# another OOMed outright, so a threshold at 0.95 would have called MBS4 safe.
SAFE_FRACTION_CEILING = 0.85

# Queues carrying evidence for this mechanism.  Boundary probes come first because
# their MBS ladder is what makes the region resolvable at all.
EVIDENCE_QUEUES = (
    ROOT / "matrix" / "h800_lora_2gpu_zero3_boundary_jobs_v1.jsonl",
    ROOT / "matrix" / "h800_v3_prospective_acceptance_jobs_v1.jsonl",
    ROOT / "matrix" / "h800_zero_gc_matched_contrast_jobs_v1.jsonl",
)


def _matches_mechanism(job: Mapping[str, Any]) -> bool:
    return (
        job.get("train_type") == "lora"
        and int(job.get("zero_stage") or 0) == 3
        and not bool(job.get("gradient_checkpointing"))
        and int(job.get("gpu_count") or 0) == 2
        and not bool(job.get("packing"))
        and not bool(job.get("offload"))
    )


def _terminal_observation(job_id: str) -> dict[str, Any] | None:
    directory = RESULTS / job_id
    status_path = directory / "status.json"
    if not status_path.is_file():
        return None
    status = read_json(status_path)
    classification = str(status.get("classification") or "")
    if classification not in {"success", "oom"}:
        return None
    peak_gib: float | None = None
    per_rank: list[float] = []
    latest = directory / "latest_attempt.json"
    if latest.is_file():
        attempt = directory / str(read_json(latest)["attempt_path"])
        for path in sorted((attempt / "metrics").glob("summary.rank*.json")):
            summary = read_json(path)
            reserved = summary.get("max_reserved")
            if reserved:
                per_rank.append(float(reserved) / GIB)
        # Worst rank, never the mean: ZeRO-3 shards unevenly and the mean hides
        # a single rank crossing the line.
        peak_gib = max(per_rank) if per_rank else None
    return {
        "classification": classification,
        "peak_gib": peak_gib,
        "per_rank_peak_gib": per_rank,
        "wall_seconds": status.get("wall_seconds"),
    }


def _collect() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for queue in EVIDENCE_QUEUES:
        if not queue.is_file():
            continue
        for job in read_jsonl(queue):
            job_id = str(job.get("job_id") or "")
            if not job_id or job_id in seen or not _matches_mechanism(job):
                continue
            observation = _terminal_observation(job_id)
            if observation is None:
                continue
            seen.add(job_id)
            peak = observation["peak_gib"]
            fraction = (peak / SAFE_LIMIT_GIB) if peak else None
            crossed = bool(fraction and fraction > 1.0)
            oom = observation["classification"] == "oom"
            rows.append(
                {
                    "job_id": job_id,
                    "queue": queue.name,
                    "model_id": job.get("model_id"),
                    "dataset_id": job.get("dataset_id"),
                    "cutoff_len": int(job.get("cutoff_len") or 0),
                    "mbs": int(job.get("mbs") or 0),
                    "classification": observation["classification"],
                    "peak_gib": peak,
                    "per_rank_peak_gib": observation["per_rank_peak_gib"],
                    "fraction_of_safe_limit": fraction,
                    "crossed_safe_limit": crossed,
                    # An OOM peak is a lower bound: the failing allocation is not
                    # in the counter, so it can read lower than a survivor's.
                    "peak_is_lower_bound_only": oom,
                    "is_unsafe_evidence": oom or crossed,
                }
            )
    return rows


def _cells(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_id"]), int(row["cutoff_len"]), int(row["mbs"]))].append(row)

    out: list[dict[str, Any]] = []
    for (model_id, cutoff, mbs), group in sorted(grouped.items()):
        sources = sorted({str(r["dataset_id"]) for r in group})
        unsafe = [r for r in group if r["is_unsafe_evidence"]]
        unsafe_sources = sorted({str(r["dataset_id"]) for r in unsafe})
        # Only survivors carry a trustworthy peak, so the headroom claim rests on
        # those alone.
        survivor_fractions = [
            float(r["fraction_of_safe_limit"])
            for r in group
            if r["fraction_of_safe_limit"] is not None and not r["peak_is_lower_bound_only"]
        ]
        worst = max(survivor_fractions) if survivor_fractions else None
        if unsafe:
            verdict = "unsafe"
        elif worst is None:
            verdict = "unknown_no_measured_peak"
        elif worst <= SAFE_FRACTION_CEILING:
            verdict = "safe"
        else:
            verdict = "marginal"
        out.append(
            {
                "model_id": model_id,
                "cutoff_len": cutoff,
                "mbs": mbs,
                "observations": len(group),
                "sources": sources,
                "source_count": len(sources),
                "unsafe_observations": len(unsafe),
                "unsafe_sources": unsafe_sources,
                "oom_observations": sum(
                    1 for r in group if r["classification"] == "oom"
                ),
                "crossed_without_oom": sum(
                    1 for r in group if r["crossed_safe_limit"] and r["classification"] != "oom"
                ),
                "worst_survivor_fraction": worst,
                "verdict": verdict,
            }
        )
    return out


def _region(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Largest MBS callable safe for each (model, cutoff), plus what bounds it."""

    by_scenario: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_scenario[(str(cell["model_id"]), int(cell["cutoff_len"]))].append(cell)

    recommendations: list[dict[str, Any]] = []
    for (model_id, cutoff), group in sorted(by_scenario.items()):
        safe = [c for c in group if c["verdict"] == "safe"]
        unsafe = [c for c in group if c["verdict"] == "unsafe"]
        marginal = [c for c in group if c["verdict"] == "marginal"]
        max_safe = max((int(c["mbs"]) for c in safe), default=None)
        min_unsafe = min((int(c["mbs"]) for c in unsafe), default=None)
        # A safe cell at or above the smallest unsafe MBS would mean the boundary
        # is not monotone in MBS; report it rather than smoothing it away.
        inconsistent = bool(
            max_safe is not None and min_unsafe is not None and max_safe >= min_unsafe
        )
        recommendations.append(
            {
                "model_id": model_id,
                "cutoff_len": cutoff,
                "max_safe_mbs": max_safe,
                "min_unsafe_mbs": min_unsafe,
                "marginal_mbs": sorted(int(c["mbs"]) for c in marginal),
                "safe_cell_source_counts": {
                    str(c["mbs"]): c["source_count"] for c in safe
                },
                "monotonicity_violated": inconsistent,
                "recommended_ceiling_mbs": max_safe,
            }
        )
    return {
        "safe_fraction_ceiling": SAFE_FRACTION_CEILING,
        "safety_limit_gib": SAFE_LIMIT_GIB,
        "per_scenario": recommendations,
        "any_monotonicity_violation": any(
            r["monotonicity_violated"] for r in recommendations
        ),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# ZeRO-3（LoRA 2卡 检查点关）实测安全区",
        "",
        f"- 生成时间：{report['generated_at_utc']}",
        f"- 安全线：{SAFE_LIMIT_GIB:.1f} GiB；判安全要求存活样本峰值 ≤ "
        f"**{SAFE_FRACTION_CEILING*100:.0f}%** 安全线",
        f"- 证据：{report['observation_count']} 条终结观测，"
        f"{report['source_count']} 个数据源",
        "",
        "## 为什么不用显存模型",
        "",
        "该机制拟合不出准入头（留出业务源后不达标）。但吞吐验收必须跑 ZeRO-3，"
        "盲跑的代价已经付过一次：8/6 那批 16 个 ZeRO-3 作业爆了 14 个。",
        "所以这里不预测边界，而是用实测把边界**框住**。",
        "",
        "## 三条判定规则",
        "",
        "1. **超线即不安全，哪怕没崩**。`src13` MBS2 跑通了，但峰值是安全线的 102%。",
        "2. **崩溃时报告的峰值偏低**。`src13` MBS4 崩了，峰值却只有 80% —— "
        "失败的那笔分配不计入计数器。所以 OOM 行的峰值只当下界，不用来论证安全。",
        "3. **看最差的那张卡**。ZeRO-3 分片不均：`src13` MBS2 两卡是 116.5 / 135.2 GiB，"
        "取平均会把越线完全掩盖。",
        "",
        "## 各配置格判定",
        "",
        "| 模型 | cutoff | MBS | 观测 | 源数 | OOM | 超线未崩 | 存活最差占比 | 判定 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    label = {
        "safe": "**安全**",
        "unsafe": "不安全",
        "marginal": "临界",
        "unknown_no_measured_peak": "未知",
    }
    for cell in report["cells"]:
        worst = cell["worst_survivor_fraction"]
        lines.append(
            "| {m} | {c} | {b} | {n} | {s} | {o} | {x} | {w} | {v} |".format(
                m=str(cell["model_id"]).replace("qwen3_", ""),
                c=cell["cutoff_len"],
                b=cell["mbs"],
                n=cell["observations"],
                s=cell["source_count"],
                o=cell["oom_observations"],
                x=cell["crossed_without_oom"],
                w="—" if worst is None else f"{worst*100:.0f}%",
                v=label[str(cell["verdict"])],
            )
        )
    lines += [
        "",
        "## 可用的安全上限",
        "",
        "| 模型 | cutoff | 最大安全 MBS | 最小不安全 MBS | 临界 MBS | 单调性 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for entry in report["region"]["per_scenario"]:
        lines.append(
            "| {m} | {c} | {s} | {u} | {g} | {v} |".format(
                m=str(entry["model_id"]).replace("qwen3_", ""),
                c=entry["cutoff_len"],
                s="—" if entry["max_safe_mbs"] is None else entry["max_safe_mbs"],
                u="—" if entry["min_unsafe_mbs"] is None else entry["min_unsafe_mbs"],
                g=", ".join(str(v) for v in entry["marginal_mbs"]) or "—",
                v="**违反**" if entry["monotonicity_violated"] else "正常",
            )
        )
    lines += [
        "",
        "## 怎么用",
        "",
        "设计吞吐验收矩阵时，ZeRO-3 候选只取上表「最大安全 MBS」及以下的格子。"
        "这样不需要显存模型点头，也不会再废掉一批一次性数据集。",
        "",
        "## 局限",
        "",
    ]
    lines += [f"- {item}" for item in report["limitations"]]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "diagnostics" / "lora_2gpu_zero3_safe_region" / "region_v1.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "diagnostics" / "lora_2gpu_zero3_safe_region" / "region_v1.md",
    )
    args = parser.parse_args()

    rows = _collect()
    if not rows:
        raise SystemExit("no terminal observations matched the mechanism")
    cells = _cells(rows)
    region = _region(cells)
    report = {
        "schema": "sft_lora_2gpu_zero3_safe_region/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "model_refit": False,
        "publishable": False,
        "mechanism": {
            "training_mode": "lora",
            "zero_stage": 3,
            "gradient_checkpointing": False,
            "gpu_count": 2,
            "packing": False,
            "label_cn": "LORA 2卡 ZeRO-3 检查点关",
        },
        "why_not_a_fitted_head": (
            "fit_h800_m1_new_mechanism_admission_heads_v1.py reports this "
            "mechanism as not standing once held-out business sources are removed, "
            "so no admission head is available to gate ZeRO-3 candidates"
        ),
        "observation_count": len(rows),
        "source_count": len({str(r["dataset_id"]) for r in rows}),
        "cells": cells,
        "region": region,
        "rows": rows,
        "limitations": [
            "the region is bounded by what has been measured; a cell absent from "
            "the table is unknown, not safe",
            "an OOM row's peak is a lower bound because the failing allocation is "
            "never counted, so unsafe cells carry no usable magnitude",
            "cells resting on a single source can flip once a second source is "
            "measured -- source_count is reported per cell for that reason",
            "packing, offload, VL, gradient checkpointing on, other GPU counts "
            "and non-H800 cards are all out of scope",
        ],
    }
    report["report_sha256"] = sha256_json(report)
    write_json(args.output, report)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "markdown_output": str(args.markdown_output.resolve()),
                "observations": len(rows),
                "cells": len(cells),
                "safe_cells": sum(1 for c in cells if c["verdict"] == "safe"),
                "unsafe_cells": sum(1 for c in cells if c["verdict"] == "unsafe"),
                "region": region["per_scenario"],
                "any_monotonicity_violation": region["any_monotonicity_violation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
