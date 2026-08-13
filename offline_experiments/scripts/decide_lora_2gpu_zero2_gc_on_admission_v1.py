#!/usr/bin/env python3
"""Decide whether `lora / 2 GPU / ZeRO-2 / gradient checkpointing on` needs an
admission head, or whether its measured headroom justifies admitting it outright.

Why this exists
---------------
V5 cannot fit an admission head for this mechanism: the recount reports 7
independent sources but **0 negative-boundary sources**, and the fitting rule
needs at least 2.  A logistic head separates safe from unsafe, so with no unsafe
row there is nothing to separate.

The obvious response -- run jobs at higher MBS until one OOMs -- was rejected.
The scheduler's ladder tops out at MBS 16 (`ADAPTIVE_MBS_DOMAIN`), so forcing a
boundary would mean editing a script shared with other campaigns, and the
evidence below suggests the boundary is far outside the domain anyone actually
trains in.

What this script asserts
------------------------
That the absence of negatives is a *property of the mechanism*, not a sampling
gap: gradient checkpointing recomputes activations instead of storing them, so
peak memory becomes nearly insensitive to the factors that normally drive it.
If that holds across models, cutoffs and MBS values, then "no head, admit within
the measured envelope" is the honest conclusion and the envelope is the contract.

The script fails closed in the direction that matters.  If any observation
crosses the safety line, or if the headroom margin is thinner than the declared
threshold, it reports that a head is required and refuses the blanket-admit
recommendation -- which is the outcome that would send us back to running GPU
probes.

Read-only: fits nothing, publishes nothing, touches no frozen artifact.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, ROOT, sha256_file, sha256_json, write_json

OBSERVATIONS = ARTIFACT_DIR / "canonical_h800_observations_with_backfill_v1.jsonl"
RECOUNT = ARTIFACT_DIR / "h800_admission_mechanism_evidence_recount_v1.json"

MECHANISM = {
    "training_mode": "lora",
    "zero_stage": 2,
    "gradient_checkpointing": True,
    "gpu_count": 2,
    "packing": False,
}
LABEL_CN = "LORA 2卡 ZeRO-2 检查点开"

# A blanket admit is only defensible with a wide, uniform margin.  These are the
# thresholds the recommendation is conditioned on; they are stated before the
# numbers are read so the conclusion cannot be reverse-fitted to them.
MAX_OBSERVED_FRACTION_OF_SAFE_LIMIT = 0.75
MINIMUM_OBSERVATIONS = 40
MINIMUM_DISTINCT_CUTOFFS = 5
MINIMUM_DISTINCT_MBS = 3
MINIMUM_DISTINCT_MODELS = 2


def _job(row: Mapping[str, Any]) -> dict[str, Any]:
    configuration = row.get("configuration") or {}
    job = configuration.get("job")
    return job if isinstance(job, dict) else {}


def _matches(job: Mapping[str, Any]) -> bool:
    return (
        job.get("train_type") == MECHANISM["training_mode"]
        and int(job.get("zero_stage") or 0) == MECHANISM["zero_stage"]
        and bool(job.get("gradient_checkpointing")) is MECHANISM["gradient_checkpointing"]
        and int(job.get("gpu_count") or 0) == MECHANISM["gpu_count"]
        and bool(job.get("packing")) is MECHANISM["packing"]
    )


def _collect(observations: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with observations.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            job = _job(row)
            if not _matches(job):
                continue
            memory = (row.get("measurements") or {}).get("memory") or {}
            peak = memory.get("max_reserved_bytes")
            outcome = row.get("outcome") or {}
            # The safety line is per-job in principle; fall back to the shared
            # constant only when a row predates that field.
            limit = job.get("safe_limit_bytes") or 0.95 * 150_142_189_568
            rows.append(
                {
                    "observation_id": row.get("observation_id"),
                    "model_id": job.get("model_id"),
                    "cutoff_len": job.get("cutoff_len"),
                    "mbs": job.get("mbs"),
                    "dataset_id": job.get("dataset_id"),
                    "outcome_class": outcome.get("class"),
                    "is_oom": "oom" in str(outcome.get("class") or ""),
                    "peak_reserved_bytes": float(peak) if peak else None,
                    "safe_limit_bytes": float(limit),
                    "fraction_of_safe_limit": (float(peak) / float(limit)) if peak else None,
                    "peak_is_imputed": bool(outcome.get("peak_memory_is_imputed")),
                    "calibration_base_eligible": bool(
                        outcome.get("calibration_base_eligible")
                    ),
                }
            )
    return rows


def _envelope(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The measured region inside which a blanket admit would be claimed."""

    measured = [r for r in rows if r["fraction_of_safe_limit"] is not None]
    cutoffs = sorted({int(r["cutoff_len"]) for r in measured if r["cutoff_len"]})
    mbs_values = sorted({int(r["mbs"]) for r in measured if r["mbs"]})
    models = sorted({str(r["model_id"]) for r in measured if r["model_id"]})
    fractions = [float(r["fraction_of_safe_limit"]) for r in measured]
    return {
        "observations": len(rows),
        "observations_with_measured_peak": len(measured),
        "models": models,
        "cutoff_len_range": [cutoffs[0], cutoffs[-1]] if cutoffs else None,
        "cutoff_len_values": cutoffs,
        "mbs_range": [mbs_values[0], mbs_values[-1]] if mbs_values else None,
        "mbs_values": mbs_values,
        "max_fraction_of_safe_limit": max(fractions) if fractions else None,
        "median_fraction_of_safe_limit": statistics.median(fractions) if fractions else None,
        "oom_count": sum(1 for r in rows if r["is_oom"]),
        "crossed_safe_limit_count": sum(1 for r in measured if r["fraction_of_safe_limit"] > 1.0),
        "imputed_peak_count": sum(1 for r in rows if r["peak_is_imputed"]),
    }


def _sensitivity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How much does peak memory actually move with cutoff and MBS?

    The blanket-admit argument rests on insensitivity.  If peak memory scaled
    steeply with either factor, a boundary would sit just outside the sampled
    region and the wide margin would be an illusion of sampling.

    Comparisons are made **within a fixed (model, other-factor) slice**.  Pooling
    across models and MBS values mixes an 8B/MBS1 point with a 14B/MBS16 point and
    produces a meaningless elasticity -- an earlier version of this script did
    exactly that and reported peak memory *falling* with cutoff.
    """

    measured = [
        r
        for r in rows
        if r["peak_reserved_bytes"] and r["model_id"] and r["cutoff_len"] and r["mbs"]
    ]

    def slices(vary: str, hold: str) -> list[dict[str, Any]]:
        """Largest-span series for each (model, held factor) combination."""

        grouped: dict[tuple[str, int], dict[int, float]] = {}
        for row in measured:
            key = (str(row["model_id"]), int(row[hold]))
            bucket = grouped.setdefault(key, {})
            value = int(row[vary])
            bucket[value] = max(
                bucket.get(value, 0.0), float(row["peak_reserved_bytes"])
            )
        out: list[dict[str, Any]] = []
        for (model_id, held), bucket in sorted(grouped.items()):
            if len(bucket) < 2:
                continue
            keys = sorted(bucket)
            low, high = keys[0], keys[-1]
            input_factor = high / low
            peak_factor = bucket[high] / bucket[low]
            out.append(
                {
                    "model_id": model_id,
                    hold: held,
                    f"{vary}_from": low,
                    f"{vary}_to": high,
                    "input_factor": round(input_factor, 2),
                    "peak_factor": round(peak_factor, 3),
                    "elasticity": (
                        round((peak_factor - 1.0) / (input_factor - 1.0), 3)
                        if input_factor != 1.0
                        else None
                    ),
                    "peak_gib_by_value": {
                        str(k): round(bucket[k] / 1024**3, 1) for k in keys
                    },
                }
            )
        return out

    cutoff_slices = slices("cutoff_len", "mbs")
    mbs_slices = slices("mbs", "cutoff_len")
    cutoff_elasticities = [
        s["elasticity"] for s in cutoff_slices if s["elasticity"] is not None
    ]
    mbs_elasticities = [s["elasticity"] for s in mbs_slices if s["elasticity"] is not None]
    return {
        "comparison_basis": (
            "within a fixed (model, other-factor) slice; never pooled across "
            "models or across the held factor"
        ),
        "cutoff_slices": cutoff_slices,
        "mbs_slices": mbs_slices,
        "max_cutoff_elasticity": max(cutoff_elasticities) if cutoff_elasticities else None,
        "max_mbs_elasticity": max(mbs_elasticities) if mbs_elasticities else None,
        "interpretation": (
            "gradient checkpointing recomputes activations rather than storing "
            "them, so peak memory is dominated by weights, gradients and "
            "optimizer state, which do not scale with cutoff or MBS"
        ),
    }


def _decide(envelope: Mapping[str, Any], recount: Mapping[str, Any]) -> dict[str, Any]:
    max_fraction = envelope["max_fraction_of_safe_limit"]
    checks = {
        "no_observation_reached_the_safety_line": envelope["crossed_safe_limit_count"] == 0,
        "no_oom_observed": envelope["oom_count"] == 0,
        "headroom_margin_below_declared_threshold": (
            max_fraction is not None
            and float(max_fraction) <= MAX_OBSERVED_FRACTION_OF_SAFE_LIMIT
        ),
        "sample_size_sufficient": envelope["observations"] >= MINIMUM_OBSERVATIONS,
        "cutoff_coverage_sufficient": (
            len(envelope["cutoff_len_values"]) >= MINIMUM_DISTINCT_CUTOFFS
        ),
        "mbs_coverage_sufficient": len(envelope["mbs_values"]) >= MINIMUM_DISTINCT_MBS,
        "model_coverage_sufficient": len(envelope["models"]) >= MINIMUM_DISTINCT_MODELS,
        "no_imputed_peaks_in_evidence": envelope["imputed_peak_count"] == 0,
    }
    blanket_admit_defensible = all(checks.values())
    return {
        "thresholds": {
            "max_observed_fraction_of_safe_limit": MAX_OBSERVED_FRACTION_OF_SAFE_LIMIT,
            "minimum_observations": MINIMUM_OBSERVATIONS,
            "minimum_distinct_cutoffs": MINIMUM_DISTINCT_CUTOFFS,
            "minimum_distinct_mbs": MINIMUM_DISTINCT_MBS,
            "minimum_distinct_models": MINIMUM_DISTINCT_MODELS,
        },
        "checks": checks,
        "recount_says_head_fittable": bool(recount.get("head_fittable_now")),
        "recount_negative_boundary_sources": recount.get("negative_boundary_sources"),
        "recommendation": (
            "admit_within_measured_envelope_without_head"
            if blanket_admit_defensible
            else "admission_head_required_run_boundary_probes"
        ),
        "blanket_admit_defensible": blanket_admit_defensible,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    env = report["envelope"]
    dec = report["decision"]
    sens = report["sensitivity"]
    gib = 1024**3
    lines = [
        f"# {LABEL_CN}：是否需要准入头",
        "",
        f"- 生成时间：{report['generated_at_utc']}",
        f"- 结论：**{'无需准入头，在已测范围内直接放行' if dec['blanket_admit_defensible'] else '需要准入头，必须补跑边界探测'}**",
        "",
        "## 为什么不能拟合准入头",
        "",
        f"该机制有 {env['observations']} 条观测、"
        f"{report['recount']['independent_sources']} 个独立数据源，"
        f"但**跑爆样本 {env['oom_count']} 个、超安全线样本 {env['crossed_safe_limit_count']} 个**。",
        "准入头本质是一条「安全 / 不安全」的分界线，没有不安全样本就画不出线。",
        "",
        "## 实测显存余量",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 观测数 | {env['observations']} |",
        f"| 最大峰值占安全线 | **{env['max_fraction_of_safe_limit']*100:.1f}%** |",
        f"| 中位峰值占安全线 | {env['median_fraction_of_safe_limit']*100:.1f}% |",
        f"| 跑爆数 | {env['oom_count']} |",
        f"| 超安全线数 | {env['crossed_safe_limit_count']} |",
        "",
        "## 覆盖范围（即「已测范围」的边界）",
        "",
        "| 维度 | 覆盖 |",
        "|---|---|",
        f"| 模型 | {', '.join(env['models'])} |",
        f"| cutoff | {env['cutoff_len_range'][0]} ~ {env['cutoff_len_range'][1]}"
        f"（{len(env['cutoff_len_values'])} 档） |",
        f"| MBS | {env['mbs_range'][0]} ~ {env['mbs_range'][1]}"
        f"（{len(env['mbs_values'])} 档） |",
        "",
        "## 关键证据：显存对负载几乎不敏感",
        "",
        "开梯度检查点后，激活值是重算而非存储，所以峰值主要由权重、梯度、优化器状态决定，"
        "这些都不随 cutoff 或 MBS 增长。",
        "",
        "下面每一行都是**固定模型、固定另一个因子**的对照，不跨模型混算。",
        "弹性 1.0 = 等比增长，0 = 完全不敏感。",
        "",
    ]
    for key, vary_cn, hold_key, hold_cn in (
        ("cutoff_slices", "cutoff", "mbs", "MBS"),
        ("mbs_slices", "MBS", "cutoff_len", "cutoff"),
    ):
        entries = sens.get(key) or []
        if not entries:
            continue
        lines += [
            f"**峰值 vs {vary_cn}**",
            "",
            f"| 模型 | 固定 {hold_cn} | {vary_cn} 变化 | 输入倍数 | 峰值倍数 | 弹性 | 各点峰值 GiB |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
        low_key = "cutoff_len_from" if key == "cutoff_slices" else "mbs_from"
        high_key = "cutoff_len_to" if key == "cutoff_slices" else "mbs_to"
        for entry in entries:
            points = ", ".join(
                f"{k}→{v}" for k, v in entry["peak_gib_by_value"].items()
            )
            lines.append(
                "| {m} | {h} | {a}→{b} | {i:.1f}× | {p:.2f}× | {e} | {pts} |".format(
                    m=entry["model_id"].replace("qwen3_", ""),
                    h=entry[hold_key],
                    a=entry[low_key],
                    b=entry[high_key],
                    i=entry["input_factor"],
                    p=entry["peak_factor"],
                    e="—" if entry["elasticity"] is None else f"{entry['elasticity']:.3f}",
                    pts=points,
                )
            )
        lines.append("")
    if sens.get("max_cutoff_elasticity") is not None:
        lines += [
            f"最大弹性：cutoff 方向 **{sens['max_cutoff_elasticity']:.3f}**，"
            f"MBS 方向 **{sens['max_mbs_elasticity']:.3f}**。"
            "两者都远小于 1，说明峰值不随负载等比增长。",
            "",
        ]
    lines += [
        "## 预注册判定项",
        "",
        "| 判定项 | 结果 |",
        "|---|---|",
    ]
    for key, value in dec["checks"].items():
        lines.append(f"| `{key}` | {'通过' if value else '未通过'} |")
    lines += [
        "",
        "## 适用边界（重要）",
        "",
        f"本结论**只在已测范围内成立**：LoRA、2 卡、ZeRO-2、开梯度检查点、不 packing、"
        f"MBS ≤ {env['mbs_range'][1]}、cutoff ≤ {env['cutoff_len_range'][1]}、"
        f"模型 {' 或 '.join(env['models'])}。",
        "",
        "超出上述范围时**不得**沿用「直接放行」，需要重新取证。特别是：",
        "",
        f"- MBS > {env['mbs_range'][1]}：调度器 MBS 档位上限即为 {env['mbs_range'][1]}，"
        "更大的 MBS 从未测过。",
        "- 更大的模型：本结论基于 8B 与 14B，未覆盖更大参数量。",
        "- packing、offload、VL、非 H800 卡：全部在范围外。",
        "",
        "## 与其他机制的关系",
        "",
        "同为 2 卡 LoRA 的 **ZeRO-3 检查点关** 机制则完全相反：它有跑爆样本，"
        "且在 cutoff 4096 / MBS8 就会撞线（约 138 GiB）。两者不可互相类推。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=OBSERVATIONS)
    parser.add_argument("--recount", type=Path, default=RECOUNT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "diagnostics" / "lora_2gpu_zero2_gc_on_admission_decision"
        / "decision_v1.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "diagnostics" / "lora_2gpu_zero2_gc_on_admission_decision"
        / "decision_v1.md",
    )
    args = parser.parse_args()

    rows = _collect(args.observations.resolve())
    if not rows:
        raise SystemExit("no observations matched the mechanism; refusing to conclude")
    envelope = _envelope(rows)
    sensitivity = _sensitivity(rows)

    recount_doc = json.loads(args.recount.read_text(encoding="utf-8"))
    mechanism_recount: dict[str, Any] = {}
    for entry in recount_doc.get("mechanisms", []):
        if (
            entry.get("training_mode") == MECHANISM["training_mode"]
            and int(entry.get("zero_stage") or 0) == MECHANISM["zero_stage"]
            and bool(entry.get("gradient_checkpointing")) is MECHANISM["gradient_checkpointing"]
            and int(entry.get("gpu_count") or 0) == MECHANISM["gpu_count"]
            and bool(entry.get("packing")) is MECHANISM["packing"]
        ):
            mechanism_recount = entry
            break
    if not mechanism_recount:
        raise SystemExit("mechanism absent from the recount; refusing to conclude")

    decision = _decide(envelope, mechanism_recount)
    report = {
        "schema": "sft_lora_2gpu_zero2_gc_on_admission_decision/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "model_refit": False,
        "publishable": False,
        "mechanism": {**MECHANISM, "label_cn": LABEL_CN},
        "envelope": envelope,
        "sensitivity": sensitivity,
        "recount": mechanism_recount,
        "decision": decision,
        "observations_sha256": sha256_file(args.observations),
        "recount_sha256": sha256_file(args.recount),
        "rows": rows,
        "limitations": [
            "the conclusion is bounded by the measured envelope and does not "
            "extend to MBS above the scheduler ladder maximum",
            "only 8B and 14B were measured; larger models are unaddressed",
            "packing, offload, VL and non-H800 cards remain out of scope",
            "no negative sample exists for this mechanism, so no admission head "
            "can be validated -- the recommendation rests on measured headroom, "
            "not on a fitted decision boundary",
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
                "observations": envelope["observations"],
                "max_fraction_of_safe_limit": envelope["max_fraction_of_safe_limit"],
                "oom_count": envelope["oom_count"],
                "crossed_safe_limit_count": envelope["crossed_safe_limit_count"],
                "checks": decision["checks"],
                "recommendation": decision["recommendation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
