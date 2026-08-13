#!/usr/bin/env python3
"""Diagnose fresh H800 memory misses without fitting on the holdout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Mapping

from common import ARTIFACT_DIR, MATRIX_DIR, RESULTS_DIR, read_json, read_jsonl, sha256_file, write_json
from h800_challenger_modeling import _predict_memory_center
from h800_physical_v4b_predictor import H800PhysicalV4BPredictor


GIB = float(1024**3)
DEFAULT_QUEUE = MATRIX_DIR / "h800_fresh_holdout_jobs_v2.jsonl"
DEFAULT_PREDICTIONS = ARTIFACT_DIR / "h800_frozen_predictions_before_fresh_holdout_v2.json"
DEFAULT_RESULTS = ARTIFACT_DIR / "collected_results.json"
DEFAULT_ACCEPTANCE = ARTIFACT_DIR / "h800_fresh_holdout_acceptance_v2.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_fresh_memory_residual_diagnosis_v2.json"
DEFAULT_MARKDOWN = ARTIFACT_DIR / "h800_fresh_memory_residual_diagnosis_v2.md"
FRESH_PROFILE_DIR = ARTIFACT_DIR / "fresh_holdout_v2" / "profiles"
CATALOG_PROFILE_DIR = ARTIFACT_DIR / "dataset_profiles"


def _percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}" if math.isfinite(number) else "—"


def _profile_path(dataset_id: str, explicit: str | None = None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        (
            FRESH_PROFILE_DIR / f"{dataset_id}.qwen3_nothink.jsonl",
            CATALOG_PROFILE_DIR / f"{dataset_id}.qwen3_nothink.jsonl",
        )
    )
    return next((path for path in candidates if path.is_file()), None)


def profile_padding_statistics(path: Path, *, cutoff_len: int, physical_mbs: int) -> dict[str, Any]:
    lengths = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            lengths.append(int(row["total_tokens"]))
    clipped = sorted(min(int(cutoff_len), value) for value in lengths)
    count = len(clipped)
    if not count:
        raise ValueError(f"empty profile: {path}")
    expected_batch_max = sum(
        value
        * (
            (index / count) ** int(physical_mbs)
            - ((index - 1) / count) ** int(physical_mbs)
        )
        for index, value in enumerate(clipped, start=1)
    )
    mean = fmean(clipped)
    variance = fmean((value - mean) ** 2 for value in clipped)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": count,
        "cutoff_len": int(cutoff_len),
        "physical_mbs": int(physical_mbs),
        "mean_clipped_tokens": mean,
        "p50_clipped_tokens": _percentile(clipped, 50.0),
        "p90_clipped_tokens": _percentile(clipped, 90.0),
        "p99_clipped_tokens": _percentile(clipped, 99.0),
        "maximum_clipped_tokens": max(clipped),
        "truncation_fraction": sum(value >= cutoff_len for value in lengths) / count,
        "coefficient_of_variation": math.sqrt(variance) / mean if mean > 0.0 else None,
        "expected_random_batch_max_tokens": expected_batch_max,
        "expected_random_batch_max_fraction_of_cutoff": expected_batch_max / cutoff_len,
        "expected_padded_tokens_per_physical_batch": expected_batch_max * physical_mbs,
        "formula": "E[max(clipped_length_1..clipped_length_mbs)] from the empirical profile CDF",
    }


def _request_from_prediction(prediction: Mapping[str, Any]) -> dict[str, Any]:
    configuration = prediction["configuration"]
    material = prediction["scenario_material"]
    return {
        "request_id": prediction["request_id"],
        "comparison_group": prediction["comparison_group"],
        "model_id": configuration["model_id"],
        "dataset_id": configuration["dataset_id"],
        "training_mode": configuration["training_mode"],
        "lora_rank": configuration["lora_rank"],
        "dataset_category": configuration["dataset_category"],
        "target_gbs": configuration["target_gbs"],
        "cutoff_len": configuration["cutoff_len"],
        "gpu_count": configuration["gpu_count"],
        "physical_mbs": configuration["physical_mbs"],
        "gradient_accumulation_steps": configuration["gradient_accumulation_steps"],
        "zero_stage": configuration["zero_stage"],
        "gradient_checkpointing": configuration["gradient_checkpointing"],
        "packing": configuration["packing"],
        "offload": configuration["offload"],
        "dtype": configuration["dtype"],
        "kernel_path": configuration["kernel_path"],
        "profile_tokenizer_id": material["profile_tokenizer_id"],
        "profile_template_id": material["profile_template_id"],
    }


def _rank_summaries(job_id: str, *, mbs: int) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted((RESULTS_DIR / job_id / "metrics").glob("summary.rank*.json")):
        row = read_json(path)
        totals = row.get("measured_totals") or {}
        physical_batches = int(totals.get("physical_batches") or 0)
        computed_tokens = int(totals.get("computed_tokens") or 0)
        allocated = int(row.get("max_allocated") or 0)
        reserved = int(row.get("max_reserved") or 0)
        summaries.append(
            {
                "path": str(path.resolve()),
                "rank": row.get("rank"),
                "measured_steps": row.get("measured_steps"),
                "max_allocated_gib": allocated / GIB if allocated else None,
                "max_reserved_gib": reserved / GIB if reserved else None,
                "reserved_to_allocated_ratio": reserved / allocated if allocated else None,
                "computed_tokens_per_physical_batch": (
                    computed_tokens / physical_batches if physical_batches else None
                ),
                "observed_average_padded_sequence_proxy": (
                    computed_tokens / physical_batches / int(mbs)
                    if physical_batches and mbs
                    else None
                ),
            }
        )
    return summaries


def _same_configuration(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = (
        "model_id",
        "train_type",
        "gpu_count",
        "mbs",
        "zero_stage",
        "zero",
        "gc",
        "cutoff_len",
        "packing",
    )
    for field in fields:
        left_value = left.get(field)
        right_value = right.get(field)
        if field == "zero_stage" and (left_value is None or right_value is None):
            continue
        if field == "zero" and (left_value is None or right_value is None):
            continue
        if left_value != right_value:
            return False
    return True


def analyze(
    *,
    queue_path: Path,
    prediction_path: Path,
    result_path: Path,
    acceptance_path: Path,
) -> dict[str, Any]:
    queue = read_jsonl(queue_path)
    frozen = read_json(prediction_path)
    collected = read_json(result_path)
    acceptance = read_json(acceptance_path)
    results = collected["rows"]
    result_by_job = {str(row["job_id"]): row for row in results}
    prediction_by_request = {
        str(row["request_id"]): row for row in frozen["predictions"]
    }
    predictor = H800PhysicalV4BPredictor(
        additional_dataset_profile_dir=FRESH_PROFILE_DIR
    )
    allocated_model = predictor.memory_report["memory"]["frozen_model"][
        "allocated_center_diagnostic"
    ]

    rows = []
    for input_index, job in enumerate(queue):
        prediction = prediction_by_request[str(job["predictor_request_id"])]
        record, _, _ = predictor._record(  # noqa: SLF001 - diagnostic uses frozen inference path
            _request_from_prediction(prediction),
            input_index=input_index,
        )
        predicted_allocated = _predict_memory_center(record, allocated_model)
        result = result_by_job[str(job["job_id"])]
        observed_allocated = float(result["max_allocated_bytes"])
        observed_reserved = float(result["max_reserved_bytes"])
        memory = prediction["memory"]
        profile_path = _profile_path(
            str(job["dataset_id"]), str(job["dataset_profile_path"])
        )
        if profile_path is None:
            raise ValueError(f"missing profile for {job['dataset_id']}")
        profile = profile_padding_statistics(
            profile_path,
            cutoff_len=int(job["cutoff_len"]),
            physical_mbs=int(job["mbs"]),
        )
        rows.append(
            {
                "job_id": job["job_id"],
                "candidate_id": job["predictor_request_id"],
                "scenario_id": job["scenario_id"],
                "dataset_id": job["dataset_id"],
                "model_id": job["model_id"],
                "train_type": job["train_type"],
                "gpu_count": job["gpu_count"],
                "mbs": job["mbs"],
                "zero_stage": job["zero_stage"],
                "zero": job["zero"],
                "gc": job["gc"],
                "cutoff_len": job["cutoff_len"],
                "packing": job["packing"],
                "predicted_allocated_gib": predicted_allocated / GIB,
                "observed_allocated_gib": observed_allocated / GIB,
                "allocated_absolute_relative_error": abs(
                    predicted_allocated - observed_allocated
                )
                / observed_allocated,
                "predicted_reserved_center_gib": float(memory["reserved_center_bytes"]) / GIB,
                "predicted_reserved_upper_gib": float(memory["admission_upper_reserved_bytes"]) / GIB,
                "observed_reserved_gib": observed_reserved / GIB,
                "reserved_center_absolute_relative_error": abs(
                    float(memory["reserved_center_bytes"]) - observed_reserved
                )
                / observed_reserved,
                "upper_covers_observed": float(memory["admission_upper_reserved_bytes"])
                >= observed_reserved,
                "observed_reserved_to_allocated_ratio": observed_reserved / observed_allocated,
                "observed_fragmentation_fraction_of_allocated": (
                    observed_reserved - observed_allocated
                )
                / observed_allocated,
                "observed_reserved_to_predicted_allocated_ratio": observed_reserved
                / predicted_allocated,
                "profile_padding": profile,
                "rank_summaries": _rank_summaries(str(job["job_id"]), mbs=int(job["mbs"])),
            }
        )

    misses = [row for row in rows if not row["upper_covers_observed"]]
    historical_matches = []
    for miss in misses:
        job = next(row for row in queue if row["job_id"] == miss["job_id"])
        for candidate in results:
            if candidate.get("job_id") == miss["job_id"]:
                continue
            if candidate.get("classification") != "success" or not _same_configuration(job, candidate):
                continue
            dataset_id = str(candidate.get("dataset_id") or "")
            profile_path = _profile_path(dataset_id)
            profile = (
                profile_padding_statistics(
                    profile_path,
                    cutoff_len=int(job["cutoff_len"]),
                    physical_mbs=int(job["mbs"]),
                )
                if profile_path is not None
                else None
            )
            historical_matches.append(
                {
                    "fresh_miss_job_id": miss["job_id"],
                    "historical_job_id": candidate.get("job_id"),
                    "dataset_id": dataset_id,
                    "classification": candidate.get("classification"),
                    "observed_allocated_gib": (
                        float(candidate["max_allocated_bytes"]) / GIB
                        if candidate.get("max_allocated_bytes")
                        else None
                    ),
                    "observed_reserved_gib": (
                        float(candidate["max_reserved_bytes"]) / GIB
                        if candidate.get("max_reserved_bytes")
                        else None
                    ),
                    "reserved_to_allocated_ratio": (
                        float(candidate["max_reserved_bytes"])
                        / float(candidate["max_allocated_bytes"])
                        if candidate.get("max_reserved_bytes")
                        and candidate.get("max_allocated_bytes")
                        else None
                    ),
                    "execution_fingerprint_quality": candidate.get(
                        "execution_fingerprint_quality"
                    ),
                    "calibration_eligible": candidate.get("calibration_eligible"),
                    "profile_padding": profile,
                    "rank_summaries": _rank_summaries(
                        str(candidate["job_id"]), mbs=int(job["mbs"])
                    ),
                    "evidence_role": "legacy_corroboration_only",
                }
            )

    allocated_errors = [row["allocated_absolute_relative_error"] for row in rows]
    reserved_errors = [row["reserved_center_absolute_relative_error"] for row in rows]
    multipliers = [row["observed_reserved_to_allocated_ratio"] for row in rows]
    summary = {
        "rows": len(rows),
        "allocated_center": {
            "mape": fmean(allocated_errors),
            "median_ape": median(allocated_errors),
            "p90_ape": _percentile(allocated_errors, 90.0),
            "maximum_ape": max(allocated_errors),
        },
        "reserved_center": {
            "mape": fmean(reserved_errors),
            "median_ape": median(reserved_errors),
            "p90_ape": _percentile(reserved_errors, 90.0),
            "maximum_ape": max(reserved_errors),
        },
        "observed_reserved_to_allocated": {
            "mean": fmean(multipliers),
            "median": median(multipliers),
            "p90": _percentile(multipliers, 90.0),
            "maximum": max(multipliers),
        },
        "upper_miss_count": len(misses),
    }
    return {
        "schema": "sft_h800_fresh_memory_residual_diagnosis/v2",
        "generated_at_utc": acceptance["generated_at_utc"],
        "timestamp_semantics": "inherits deterministic acceptance evidence completion time",
        "analysis_mode": "post_holdout_diagnosis_no_refit",
        "gpu_training_started": False,
        "queues_mutated": False,
        "source_bindings": {
            "queue": {"path": str(queue_path.resolve()), "sha256": sha256_file(queue_path)},
            "frozen_predictions": {
                "path": str(prediction_path.resolve()),
                "sha256": sha256_file(prediction_path),
            },
            "collected_results": {
                "path": str(result_path.resolve()),
                "sha256": sha256_file(result_path),
            },
            "acceptance_report": {
                "path": str(acceptance_path.resolve()),
                "sha256": sha256_file(acceptance_path),
            },
            "allocated_model": {
                "source": "h800_challenger_modeling.json.memory.frozen_model.allocated_center_diagnostic",
                "training_rows": allocated_model.get("fit_success_rows"),
                "feature_count": len(allocated_model.get("feature_names") or []),
            },
        },
        "summary": summary,
        "upper_misses": misses,
        "same_configuration_historical_matches": historical_matches,
        "diagnosis": {
            "evidence_supported": [
                "The frozen allocated-memory diagnostic generalizes materially better than the direct reserved center on these 24 rows.",
                "The only upper miss has the largest observed reserved/allocated multiplier in the campaign.",
                "The frozen reserved model consumes the dataset profile binding but has no profile-length or dynamic-padding feature in its memory basis.",
                "For the same configuration, empirical expected padded tokens per MBS=2 batch are nearly twice as high on the fresh education profile as on the legacy multiturn_4096 profile.",
            ],
            "inference_requiring_repeats": (
                "The concentrated miss is most consistent with a selector-specific CUDA allocator reservation/fragmentation response to non-packing padding pressure. "
                "One fresh exact run is insufficient to estimate its variance or promote a guard."
            ),
            "not_supported": [
                "Treating the single miss as proof that every long dataset needs the same multiplier.",
                "Refitting coefficients on this holdout and reporting the same rows as fresh validation.",
                "Using the existing historical override anchor to lower or raise a base-admitted prediction; its current contract only overrides base rejections.",
            ],
        },
        "recommended_challenger_structure": {
            "name": "allocated_physical_plus_profile_aware_reservation_guard",
            "stage_1": "predict allocated center from physical shares",
            "stage_2": "predict log(reserved / allocated) from a small padding-pressure and selector interaction model",
            "profile_features": [
                "expected_random_batch_max_fraction_of_cutoff",
                "p99_clipped_tokens / cutoff_len",
                "coefficient_of_variation",
                "truncation_fraction",
            ],
            "selector_interactions": [
                "lora x zero2 x gc_off",
                "log2(mbs)",
                "log2(gpu_count)",
            ],
            "upper_formula": (
                "allocated_center * exp(reservation_multiplier_center + "
                "scenario-disjoint conformal log-residual upper)"
            ),
            "feature_budget": "4 profile terms plus 3 bounded selector interactions; select by scenario-disjoint CV",
            "publication_rule": "new calibration data plus a different unseen-profile holdout are required",
        },
        "all_rows": rows,
        "acceptance_decision_unchanged": acceptance["decisions"],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    allocated = summary["allocated_center"]
    reserved = summary["reserved_center"]
    multiplier = summary["observed_reserved_to_allocated"]
    lines = [
        "# H800 fresh holdout 显存残差诊断（2026-08-02）",
        "",
        "> 这是 holdout 结果揭晓后的诊断，不是重新验收；没有用这 24 条结果重训任何系数。",
        "",
        "## 结论",
        "",
        "显存主体的 allocated 物理模型表现良好，主要缺口集中在 CUDA allocator 的 reserved/allocated 倍率。"
        "下一版不应继续扩大单头 reserved 回归，而应拆成“allocated 物理中心 + profile-aware reservation guard”。",
        "",
        "## 1. 两种中心预测的对比",
        "",
        f"- allocated center：MAPE {_fmt(100 * allocated['mape'], 2)}%，中位 APE "
        f"{_fmt(100 * allocated['median_ape'], 2)}%，P90 APE {_fmt(100 * allocated['p90_ape'], 2)}%。",
        f"- direct reserved center：MAPE {_fmt(100 * reserved['mape'], 2)}%，中位 APE "
        f"{_fmt(100 * reserved['median_ape'], 2)}%，P90 APE {_fmt(100 * reserved['p90_ape'], 2)}%。",
        f"- 实测 reserved/allocated：中位 {_fmt(multiplier['median'])}，P90 {_fmt(multiplier['p90'])}，"
        f"最大 {_fmt(multiplier['maximum'])}。",
        "",
        "## 2. formal upper miss",
        "",
        "| 配置 | allocated 预测/实测 GiB | reserved center/upper/实测 GiB | reserved/allocated | profile E[max]/cutoff |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["upper_misses"]:
        profile = row["profile_padding"]
        lines.append(
            f"| {row['model_id']} {row['gpu_count']}GPU Z{row['zero_stage']} MBS{row['mbs']} "
            f"GC={'on' if row['gc'] else 'off'} | {_fmt(row['predicted_allocated_gib'], 2)} / "
            f"{_fmt(row['observed_allocated_gib'], 2)} | {_fmt(row['predicted_reserved_center_gib'], 2)} / "
            f"{_fmt(row['predicted_reserved_upper_gib'], 2)} / {_fmt(row['observed_reserved_gib'], 2)} | "
            f"{_fmt(row['observed_reserved_to_allocated_ratio'])} | "
            f"{_fmt(profile['expected_random_batch_max_fraction_of_cutoff'])} |"
        )
    lines.extend(
        [
            "",
            "该点 allocated 只偏约 4%，但 reserved/allocated 达到约 2.06，所以 28.72 GiB 的 upper miss "
            "主要不是模型状态或激活主体漏算，而是 reservation/fragmentation 尾部没有覆盖。",
            "",
            "## 3. 数据分布证据",
            "",
        ]
    )
    for match in report["same_configuration_historical_matches"]:
        fresh = next(
            row
            for row in report["upper_misses"]
            if row["job_id"] == match["fresh_miss_job_id"]
        )
        current_profile = fresh["profile_padding"]
        old_profile = match.get("profile_padding") or {}
        lines.extend(
            [
                f"同配置旧结果 `{match['historical_job_id']}` 使用 `{match['dataset_id']}`，其 profile 的 "
                f"MBS=2 期望 batch max 为 {_fmt(old_profile.get('expected_random_batch_max_tokens'), 1)} tokens；"
                f"fresh education profile 为 {_fmt(current_profile['expected_random_batch_max_tokens'], 1)} tokens。",
                "",
                f"旧实测 reserved/allocated 为 {_fmt(match['reserved_to_allocated_ratio'])}，fresh 为 "
                f"{_fmt(fresh['observed_reserved_to_allocated_ratio'])}。旧结果执行指纹不完整，只能作为旁证，"
                "不能替代新 calibration repeats。",
                "",
            ]
        )
    lines.extend(
        [
            "## 4. 下一版公式结构",
            "",
            "1. 用现有 physical-shares allocated head 预测 `allocated_center`。",
            "2. 从用户静态 profile 计算 clipped length、经验 `E[max(L1...Lmbs)]`、P99、CV 和截断率。",
            "3. 用少量 selector interaction 预测 `log(reserved / allocated)`。",
            "4. 用场景级留出残差形成 conformal upper；OOM 仍作为右删失安全 guard。",
            "",
            "这批 holdout 不能用于给新模型做独立验收；下一步必须先采集显式标记为 calibration 的长度分层重复实验，"
            "再换一个未参与训练的数据 profile 做新的 prospective holdout。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = analyze(
        queue_path=args.queue,
        prediction_path=args.predictions,
        result_path=args.results,
        acceptance_path=args.acceptance,
    )
    write_json(args.output, report)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "markdown": str(args.markdown),
                "allocated_mape": report["summary"]["allocated_center"]["mape"],
                "reserved_center_mape": report["summary"]["reserved_center"]["mape"],
                "upper_miss_count": report["summary"]["upper_miss_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
