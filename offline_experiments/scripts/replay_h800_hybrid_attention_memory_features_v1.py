#!/usr/bin/env python3
"""Replay H800 text history through the dense hybrid-attention feature basis.

The command is CPU-only.  It combines the canonical Qwen3 text archive with
the text-only arm of the Qwen3.5/VL supplement, exports coefficient-ready
records, verifies analytic LoRA counts against runtime structure manifests,
and writes a Chinese audit summary.  It does not fit production coefficients
or treat OOM rows as exact peak-memory measurements.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    MATRIX_DIR,
    percentile,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from h800_theory_basis import _model_geometry as old_model_geometry
from h800_theory_basis import memory_basis as old_memory_basis
from hybrid_attention_memory_features import build_dense_hybrid_features

SCHEMA = "sft_h800_hybrid_attention_history_replay/v1"
RECORD_SCHEMA = "sft_h800_hybrid_attention_history_replay_record/v1"
CAPACITY_BYTES = 150_142_189_568

THEORY_BASIS = ARTIFACT_DIR / "h800_theory_basis.json"
QWEN3_INVENTORY = ARTIFACT_DIR / "model_inventory.json"
SUPPLEMENT_INVENTORY = (
    ARTIFACT_DIR / "h800_qwen35_vl_supplement_model_inventory_v1.json"
)
SUPPLEMENT_QUEUE = MATRIX_DIR / "h800_qwen35_vl_supplement_formal_v1.jsonl"
SUPPLEMENT_RESULTS = (
    ARTIFACT_DIR / "h800_qwen35_vl_supplement_results_v1.json"
)
OUTPUT_RECORDS = ARTIFACT_DIR / "h800_hybrid_attention_history_replay_v1.jsonl"
OUTPUT_REPORT = ARTIFACT_DIR / "h800_hybrid_attention_history_replay_v1.json"
OUTPUT_MARKDOWN = ARTIFACT_DIR / "h800_hybrid_attention_history_replay_v1.md"


def _inventory(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    report = read_json(path)
    models = {str(row["id"]): row for row in report.get("models") or []}
    fixed_lora = dict(report.get("fixed_lora") or {})
    if not models or fixed_lora.get("target") != "all":
        raise ValueError(f"inventory is incomplete: {path}")
    return models, fixed_lora


def _zero_name(stage: Any) -> str:
    value = int(stage)
    if value == 0:
        return "none"
    if value in {1, 2, 3}:
        return f"zero{value}"
    raise ValueError(f"unsupported ZeRO stage {stage!r}")


def _flat_feature_vector(features: dict[str, Any]) -> dict[str, float | int]:
    memory = features["memory"]
    activation = memory["activation_components"]
    workspace = memory["workspace_candidates"]
    compute = features["compute"]
    return {
        "state_bytes": memory["state_bytes"],
        "saved_full_attention_activations_bytes": activation[
            "saved_full_attention_activations_bytes"
        ],
        "saved_linear_attention_activations_bytes": activation[
            "saved_linear_attention_activations_bytes"
        ],
        "recompute_workspace_bytes": activation["recompute_workspace_bytes"],
        "full_attention_workspace_bytes": workspace[
            "full_attention_workspace_bytes"
        ],
        "linear_attention_workspace_bytes": workspace[
            "linear_attention_workspace_bytes"
        ],
        "linear_recurrent_state_bytes": workspace[
            "linear_recurrent_state_bytes"
        ],
        "logits_workspace_bytes": workspace["logits_workspace_bytes"],
        "zero_collective_workspace_bytes": workspace[
            "zero_collective_workspace_bytes"
        ],
        "analytic_reference_bytes": memory["analytic_reference_bytes"],
        "projection_macs_per_forward": compute["projection_macs_per_forward"],
        "full_attention_mixing_macs_per_forward": compute[
            "full_attention_mixing_macs_per_forward"
        ],
        "linear_recurrence_macs_per_forward": compute[
            "linear_recurrence_macs_per_forward"
        ],
    }


def _record(
    *,
    source: str,
    job: dict[str, Any],
    outcome: str,
    observed_reserved: int | None,
    right_censor_lower: int | None,
    model: dict[str, Any],
    fixed_lora: dict[str, Any],
    old_reference: float,
    evidence_id: str,
) -> dict[str, Any]:
    features = build_dense_hybrid_features(
        job, model, fixed_lora, CAPACITY_BYTES
    )
    signature = features["architecture_signature"]
    return {
        "schema": RECORD_SCHEMA,
        "source": source,
        "evidence_id": evidence_id,
        "job_id": str(job["job_id"]),
        "model_id": str(job["model_id"]),
        "outcome": outcome,
        "observation": {
            "peak_reserved_bytes": observed_reserved,
            "right_censor_lower_bytes": right_censor_lower,
            "semantics": (
                "exact_success_peak"
                if outcome == "success"
                else "oom_right_censored_lower_bound"
            ),
        },
        "configuration": {
            "train_type": job["train_type"],
            "gpu_count": int(job["gpu_count"]),
            "zero": job["zero"],
            "gradient_checkpointing": bool(job.get("gc")),
            "micro_batch_size": int(job["mbs"]),
            "cutoff_len": int(job["cutoff_len"]),
            "packing": bool(job.get("packing", False)),
            "kernel_path": job.get("effective_kernel_path"),
            "dataset_id": job.get("dataset_id"),
        },
        "architecture": {
            "route": signature["architecture_route"],
            "signature_sha256": signature["signature_sha256"],
            "config_sha256": signature["config_sha256"],
            "num_hidden_layers": signature["num_hidden_layers"],
            "num_full_attention_layers": signature[
                "num_full_attention_layers"
            ],
            "num_linear_attention_layers": signature[
                "num_linear_attention_layers"
            ],
            "layer_pattern_sha256": signature["layer_pattern_sha256"],
        },
        "feature_vector": _flat_feature_vector(features),
        "old_analytic_reference_bytes": float(old_reference),
        "new_minus_old_reference_bytes": float(
            features["memory"]["analytic_reference_bytes"] - old_reference
        ),
        "full_features": features,
    }


def _canonical_records() -> list[dict[str, Any]]:
    theory = read_json(THEORY_BASIS)
    models, fixed_lora = _inventory(QWEN3_INVENTORY)
    rows: list[dict[str, Any]] = []
    for old in theory.get("records") or []:
        scenario = old["scenario"]
        selector = old["selector"]
        model_id = str(scenario["model_id"])
        model = models[model_id]
        job = {
            "job_id": old["job_id"],
            "model_id": model_id,
            "model_parameters": old["model_basis"]["base_parameters"],
            "train_type": scenario["train_type"],
            "gpu_count": scenario["gpu_count"],
            "zero": _zero_name(selector["zero_stage"]),
            "gc": selector["gradient_checkpointing"],
            "mbs": scenario["physical_mbs"],
            "cutoff_len": scenario["cutoff_len"],
            "packing": selector["packing"],
            "effective_kernel_path": selector["kernel_path"],
            "dataset_id": scenario["dataset_id"],
        }
        observed = old["memory"]["observed"]
        outcome = str(old["outcome"])
        rows.append(
            _record(
                source="canonical_qwen3_theory_history",
                job=job,
                outcome=outcome,
                observed_reserved=(
                    int(observed["peak_reserved_target_bytes"])
                    if observed.get("peak_reserved_target_bytes") is not None
                    else None
                ),
                right_censor_lower=(
                    int(observed["right_censor_lower_bytes"])
                    if observed.get("right_censor_lower_bytes") is not None
                    else None
                ),
                model=model,
                fixed_lora=fixed_lora,
                old_reference=float(
                    old["memory"]["analytic_reference_bytes"]
                ),
                evidence_id=str(old["observation_id"]),
            )
        )
    return rows


def _supplement_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models, fixed_lora = _inventory(SUPPLEMENT_INVENTORY)
    queue = {
        str(row["job_id"]): row
        for row in read_jsonl(SUPPLEMENT_QUEUE)
        if row.get("track") == "qwen35_text_cross_scale"
    }
    results = read_json(SUPPLEMENT_RESULTS)
    result_rows = {
        str(row["job_id"]): row
        for row in results.get("rows") or []
        if row.get("track") == "qwen35_text_cross_scale"
    }
    if set(queue) != set(result_rows):
        raise ValueError("Qwen3.5 text queue/results do not match exactly")

    replay: list[dict[str, Any]] = []
    adapter_checks: list[dict[str, Any]] = []
    for job_id, job in queue.items():
        result = result_rows[job_id]
        model = models[str(job["model_id"])]
        old_geometry = old_model_geometry(job, model, fixed_lora)
        old_memory = old_memory_basis(job, old_geometry, CAPACITY_BYTES)
        outcome = str(result["classification"])
        metrics = result.get("metrics") or {}
        replay.append(
            _record(
                source="qwen35_text_supplement_history",
                job=job,
                outcome=outcome,
                observed_reserved=(
                    int(metrics["max_reserved_bytes"])
                    if metrics.get("max_reserved_bytes") is not None
                    else None
                ),
                right_censor_lower=(CAPACITY_BYTES if outcome == "oom" else None),
                model=model,
                fixed_lora=fixed_lora,
                old_reference=float(old_memory["analytic_reference_bytes"]),
                evidence_id=job_id,
            )
        )

        if outcome != "success" or job.get("train_type") != "lora":
            continue
        expected = int(replay[-1]["full_features"]["geometry"]["adapter_parameters"])
        for binding in result.get("artifact_bindings") or []:
            if int(binding.get("rank", -1)) != 0:
                continue
            structure_path = Path(binding["structure_path"])
            if binding["structure_sha256"] != sha256_file(structure_path):
                raise ValueError(f"structure manifest drifted: {structure_path}")
            structure = read_json(structure_path)
            observed = int(
                structure["components"]["language_model"][
                    "adapter_trainable_parameter_elements"
                ]
            )
            adapter_checks.append(
                {
                    "job_id": job_id,
                    "model_id": job["model_id"],
                    "expected_elements": expected,
                    "observed_elements": observed,
                    "exact_match": expected == observed,
                    "structure_path": str(structure_path),
                    "structure_sha256": binding["structure_sha256"],
                }
            )
            break
    return replay, adapter_checks


def _error_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    errors: list[float] = []
    ratios: list[float] = []
    for row in rows:
        observed = row["observation"]["peak_reserved_bytes"]
        if observed is None or observed <= 0:
            continue
        predicted = (
            row["feature_vector"]["analytic_reference_bytes"]
            if field == "new"
            else row["old_analytic_reference_bytes"]
        )
        ratios.append(predicted / observed)
        errors.append(abs(predicted - observed) / observed)
    return {
        "success_rows": len(errors),
        "median_absolute_percentage_error": percentile(errors, 50),
        "p90_absolute_percentage_error": percentile(errors, 90),
        "mean_predicted_to_observed_ratio": sum(ratios) / len(ratios),
        "median_predicted_to_observed_ratio": percentile(ratios, 50),
        "status": "uncalibrated_diagnostic_only",
    }


def _group_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["architecture"]["route"]].append(row)
    output: dict[str, Any] = {}
    for route, route_rows in sorted(grouped.items()):
        outcomes = Counter(row["outcome"] for row in route_rows)
        output[route] = {
            "records": len(route_rows),
            "models": sorted({row["model_id"] for row in route_rows}),
            "outcomes": dict(sorted(outcomes.items())),
            "old_uncalibrated": _error_summary(route_rows, "old"),
            "new_uncalibrated": _error_summary(route_rows, "new"),
        }
    return output


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    groups = report["coverage"]["by_architecture_route"]
    lines = [
        "# H800 混合注意力新特征历史重放",
        "",
        "本文件是 CPU 历史重放诊断，不是已经校准的线上显存预测器。OOM 只记显存下界。",
        "",
        "| 架构路线 | 记录数 | 成功 | OOM | 模型 |",
        "|---|---:|---:|---:|---|",
    ]
    for route, group in groups.items():
        outcomes = group["outcomes"]
        lines.append(
            f"| {route} | {group['records']} | {outcomes.get('success', 0)} | "
            f"{outcomes.get('oom', 0)} | {', '.join(group['models'])} |"
        )
    lines.extend(
        [
            "",
            "| 架构路线 | 旧基线未校准中位绝对百分比误差 | 新基线未校准中位绝对百分比误差 |",
            "|---|---:|---:|",
        ]
    )
    for route, group in groups.items():
        old = group["old_uncalibrated"]["median_absolute_percentage_error"]
        new = group["new_uncalibrated"]["median_absolute_percentage_error"]
        lines.append(f"| {route} | {old:.2%} | {new:.2%} |")
    validation = report["adapter_parameter_validation"]
    lines.extend(
        [
            "",
            "## 核对结果",
            "",
            (
                f"- 语言侧 LoRA 参数计数：{validation['exact_matches']}/"
                f"{validation['checks']} 个 rank-0 运行清单完全一致。"
            ),
            "- 全注意力历史只能验证新特征在普通 dense Qwen 上的特化；不能识别线性注意力系数。",
            "- Qwen3.5 历史只有 27 个成功样本和 1 个 OOM，且并行机制与长度覆盖不完整，因此只作诊断。",
            "- 新基线把同一时刻不共存的算子工作区取最大值，而不是全部相加；这属于物理结构修正，仍需新实验拟合存活系数。",
            "",
            "## 下一步",
            "",
            "第一批实验应补 Qwen3-8B 全注意力对照、Qwen3.5-4B/9B 和 Qwen3.6-27B，覆盖 1/2/4 卡、ZeRO-2/3、重算开关、长度和数据源复验。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def replay(
    *, output_records: Path, output_report: Path, output_markdown: Path
) -> dict[str, Any]:
    canonical = _canonical_records()
    supplement, adapter_checks = _supplement_records()
    records = canonical + supplement
    if not records:
        raise ValueError("history replay produced no records")
    if not adapter_checks or not all(row["exact_match"] for row in adapter_checks):
        raise ValueError("architecture-aware LoRA parameter validation failed")
    if any(
        row["outcome"] == "oom"
        and row["observation"]["peak_reserved_bytes"] is not None
        for row in records
    ):
        raise ValueError("OOM was incorrectly materialized as an exact peak")

    write_jsonl(output_records, records)
    by_route = _group_summaries(records)
    report_core = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "coefficients_were_fit": False,
        "publishable": False,
        "inputs": {
            "theory_basis": {
                "path": str(THEORY_BASIS.resolve()),
                "sha256": sha256_file(THEORY_BASIS),
            },
            "qwen3_inventory": {
                "path": str(QWEN3_INVENTORY.resolve()),
                "sha256": sha256_file(QWEN3_INVENTORY),
            },
            "supplement_queue": {
                "path": str(SUPPLEMENT_QUEUE.resolve()),
                "sha256": sha256_file(SUPPLEMENT_QUEUE),
            },
            "supplement_results": {
                "path": str(SUPPLEMENT_RESULTS.resolve()),
                "sha256": sha256_file(SUPPLEMENT_RESULTS),
            },
        },
        "coverage": {
            "records": len(records),
            "sources": dict(sorted(Counter(row["source"] for row in records).items())),
            "outcomes": dict(sorted(Counter(row["outcome"] for row in records).items())),
            "by_architecture_route": by_route,
        },
        "adapter_parameter_validation": {
            "checks": len(adapter_checks),
            "exact_matches": sum(row["exact_match"] for row in adapter_checks),
            "all_exact": all(row["exact_match"] for row in adapter_checks),
            "rows": adapter_checks,
        },
        "checks": {
            "all_records_have_architecture_hash": all(
                bool(row["architecture"]["signature_sha256"]) for row in records
            ),
            "all_oom_are_right_censored": all(
                row["outcome"] != "oom"
                or (
                    row["observation"]["peak_reserved_bytes"] is None
                    and row["observation"]["right_censor_lower_bytes"] is not None
                )
                for row in records
            ),
            "all_adapter_counts_exact": all(
                row["exact_match"] for row in adapter_checks
            ),
            "full_and_hybrid_routes_present": set(by_route)
            == {"dense_full_attention", "dense_hybrid_attention"},
        },
        "limitations": [
            "uncalibrated analytic reference is diagnostic and not an acceptance metric",
            "canonical Qwen3 archive contains no linear-attention layer",
            "Qwen3.5 text supplement is too sparse to identify all architecture-by-runtime interactions",
            "historical cutoff length is a configured bound rather than an exact per-step token shape",
            "MoE and real-media vision rows are intentionally excluded from this dense text replay",
        ],
        "outputs": {
            "records_path": str(output_records.resolve()),
            "records_sha256": sha256_file(output_records),
            "markdown_path": str(output_markdown.resolve()),
        },
    }
    report = {**report_core, "report_sha256": sha256_json(report_core)}
    _write_markdown(report, output_markdown)
    report["outputs"]["markdown_sha256"] = sha256_file(output_markdown)
    report["report_sha256"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    write_json(output_report, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-records", type=Path, default=OUTPUT_RECORDS)
    parser.add_argument("--output-report", type=Path, default=OUTPUT_REPORT)
    parser.add_argument("--output-markdown", type=Path, default=OUTPUT_MARKDOWN)
    args = parser.parse_args()
    report = replay(
        output_records=args.output_records,
        output_report=args.output_report,
        output_markdown=args.output_markdown,
    )
    print(json.dumps(report["coverage"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
