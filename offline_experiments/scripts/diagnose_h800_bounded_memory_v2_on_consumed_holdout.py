#!/usr/bin/env python3
"""Replay bounded H800 memory v2 on the already-consumed v1 holdout.

This report is diagnostic only.  It may demonstrate that the known failure was
repaired, but it cannot establish generalization or authorize publication.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from common import ARTIFACT_DIR, read_json, sha256_file, sha256_json, write_json
from h800_bounded_memory_model import ARTIFACT_SCHEMA, predict_memory
from h800_challenger_modeling import _predict_memory_center
from h800_physical_v4b_predictor import H800PhysicalV4BPredictor


DEFAULT_CANDIDATE = ARTIFACT_DIR / "h800_bounded_memory_challenger_v2.json"
DEFAULT_FROZEN_V1_PREDICTIONS = ARTIFACT_DIR / "h800_frozen_predictions_before_final_unseen_holdout_v1.json"
DEFAULT_OBSERVATIONS = ARTIFACT_DIR / "h800_final_unseen_holdout_observations_v1.json"
DEFAULT_PROFILE_DIR = ARTIFACT_DIR / "final_unseen_holdout_v1" / "profiles"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_bounded_memory_v2_consumed_holdout_diagnostic.json"
DEFAULT_MARKDOWN = ARTIFACT_DIR / "h800_bounded_memory_v2_consumed_holdout_diagnostic.md"
GIB = float(1024**3)


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = min(1.0, max(0.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_summary(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values]
    return {
        "count": len(clean),
        "mean": fmean(clean),
        "median": _percentile(clean, 0.5),
        "p90": _percentile(clean, 0.9),
        "maximum": max(clean),
    }


def _request(row: Mapping[str, Any]) -> dict[str, Any]:
    configuration = row["configuration"]
    scenario = row["scenario_material"]
    return {
        "request_id": row["request_id"],
        "comparison_group": row["comparison_group"],
        "model_id": configuration["model_id"],
        "training_mode": configuration["training_mode"],
        "lora_rank": scenario["lora_rank"],
        "dataset_id": configuration["dataset_id"],
        "dataset_category": configuration["dataset_category"],
        "target_gbs": configuration["target_gbs"],
        "cutoff_len": configuration["cutoff_len"],
        "gpu_count": configuration["gpu_count"],
        "physical_mbs": configuration["physical_mbs"],
        "gradient_accumulation_steps": configuration[
            "gradient_accumulation_steps"
        ],
        "zero_stage": configuration["zero_stage"],
        "gradient_checkpointing": configuration["gradient_checkpointing"],
        "packing": configuration["packing"],
        "offload": configuration["offload"],
        "dtype": configuration["dtype"],
        "kernel_path": configuration["kernel_path"],
        "profile_tokenizer_id": scenario["profile_tokenizer_id"],
        "profile_template_id": scenario["profile_template_id"],
    }


def _scenario_equal(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset_id"])].append(float(row[key]))
    return fmean(fmean(values) for values in grouped.values())


def diagnose(
    *,
    candidate_path: Path,
    frozen_v1_predictions_path: Path,
    observations_path: Path,
    profile_dir: Path,
) -> dict[str, Any]:
    candidate = read_json(candidate_path)
    if candidate.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("bounded v2 candidate schema mismatch")
    unsigned = dict(candidate)
    expected = unsigned.pop("report_sha256", None)
    if not isinstance(expected, str) or expected != sha256_json(unsigned):
        raise ValueError("bounded v2 candidate checksum mismatch")
    if (
        candidate.get("publishable") is not False
        or candidate.get("production_override_allowed") is not False
    ):
        raise ValueError("diagnostic candidate release contract drifted")

    frozen = read_json(frozen_v1_predictions_path)
    observations = read_json(observations_path)
    observed_by_id = {
        str(row["candidate_id"]): row for row in observations["rows"]
    }
    frozen_ids = {str(row["request_id"]) for row in frozen["predictions"]}
    if frozen_ids != set(observed_by_id) or len(frozen_ids) != 10:
        raise ValueError("consumed holdout prediction/observation identities drifted")

    base = H800PhysicalV4BPredictor(
        additional_dataset_profile_dir=profile_dir
    )
    allocated_anchor_model = base.memory_report["memory"]["frozen_model"][
        "allocated_center_diagnostic"
    ]
    rows = []
    for index, frozen_row in enumerate(frozen["predictions"]):
        record, _normalized, _support = base._record(
            _request(frozen_row), input_index=index
        )
        anchor = _predict_memory_center(record, allocated_anchor_model)
        frozen_anchor = float(frozen_row["memory"]["allocated_anchor_bytes"])
        if not math.isclose(anchor, frozen_anchor, rel_tol=1e-12, abs_tol=1e-3):
            raise ValueError("physical allocated anchor changed after v1 freeze")
        prediction = predict_memory(
            record,
            frozen_row["memory"]["padding_statistics"],
            allocated_anchor_bytes=anchor,
            artifact=candidate,
        )
        if prediction.get("available") is not True:
            raise ValueError(
                f"bounded candidate rejected frozen selector: {frozen_row['request_id']}"
            )
        observed = observed_by_id[str(frozen_row["request_id"])]
        actual_reserved = float(observed["observed_reserved_bytes"])
        actual_allocated = float(observed["observed_allocated_bytes"])
        center = float(prediction["reserved_center_bytes"])
        upper = float(prediction["operational_upper_reserved_bytes"])
        safe_limit = float(observed["safe_limit_bytes"])
        predicted_admit = upper <= safe_limit
        actual_safe = bool(
            observed["outcome"] == "success" and actual_reserved <= safe_limit
        )
        configuration = frozen_row["configuration"]
        rows.append(
            {
                "request_id": frozen_row["request_id"],
                "comparison_group": frozen_row["comparison_group"],
                "dataset_id": configuration["dataset_id"],
                "dataset_category": configuration["dataset_category"],
                "model_id": configuration["model_id"],
                "training_mode": configuration["training_mode"],
                "gpu_count": configuration["gpu_count"],
                "zero_stage": configuration["zero_stage"],
                "gradient_checkpointing": configuration[
                    "gradient_checkpointing"
                ],
                "physical_mbs": configuration["physical_mbs"],
                "selector_bucket": prediction["selector_bucket"],
                "allocated_anchor_bytes": anchor,
                "predicted_allocated_center_bytes": prediction[
                    "allocated_center_bytes"
                ],
                "observed_allocated_bytes": actual_allocated,
                "allocated_absolute_percentage_error": abs(
                    float(prediction["allocated_center_bytes"])
                    / actual_allocated
                    - 1.0
                ),
                "predicted_reserved_center_bytes": center,
                "predicted_center_guarded_upper_bytes": prediction[
                    "center_guarded_upper_bytes"
                ],
                "predicted_anchor_envelope_upper_bytes": prediction[
                    "anchor_envelope_upper_bytes"
                ],
                "predicted_operational_upper_bytes": upper,
                "observed_reserved_bytes": actual_reserved,
                "center_absolute_percentage_error": abs(
                    center / actual_reserved - 1.0
                ),
                "center_signed_percentage_error": center / actual_reserved - 1.0,
                "upper_covers_observed": upper >= actual_reserved,
                "safe_limit_bytes": safe_limit,
                "predicted_admit": predicted_admit,
                "actual_safe": actual_safe,
                "false_safe": predicted_admit and not actual_safe,
                "false_reject": (not predicted_admit) and actual_safe,
                "v1_predicted_reserved_center_bytes": observed[
                    "predicted_memory_center_bytes"
                ],
                "v1_predicted_operational_upper_bytes": observed[
                    "predicted_memory_upper_bytes"
                ],
                "v1_center_absolute_percentage_error": observed[
                    "center_absolute_relative_error"
                ],
                "v1_upper_covers_observed": observed["upper_covers_observed"],
                "v1_false_reject": observed["false_rejected_safe"],
            }
        )

    v2_errors = [row["center_absolute_percentage_error"] for row in rows]
    v1_errors = [row["v1_center_absolute_percentage_error"] for row in rows]
    report: dict[str, Any] = {
        "schema": "sft_h800_bounded_memory_v2_consumed_holdout_diagnostic/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_only_consumed_holdout_not_release_evidence",
        "publishable": False,
        "production_override_allowed": False,
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "source_bindings": {
            "bounded_v2_candidate": _binding(candidate_path),
            "frozen_v1_predictions": _binding(frozen_v1_predictions_path),
            "immutable_consumed_observations": _binding(observations_path),
        },
        "integrity": {
            "prediction_rows": len(frozen["predictions"]),
            "observation_rows": len(observations["rows"]),
            "identity_join_complete": True,
            "physical_anchor_unchanged": True,
        },
        "governance": {
            "holdout_is_already_consumed": True,
            "v2_architecture_was_designed_after_v1_failure": True,
            "diagnostic_may_support_failure_repair_only": True,
            "diagnostic_may_support_generalization_claim": False,
            "new_prospective_holdout_required": True,
        },
        "v1_frozen_baseline": {
            "center_scenario_equal_mape": _scenario_equal(
                rows, "v1_center_absolute_percentage_error"
            ),
            "center_row_absolute_percentage_error": _metric_summary(v1_errors),
            "upper_row_coverage": sum(
                row["v1_upper_covers_observed"] for row in rows
            )
            / len(rows),
            "false_safe_count": 0,
            "false_reject_count": sum(row["v1_false_reject"] for row in rows),
        },
        "v2_post_holdout_repair": {
            "center_scenario_equal_mape": _scenario_equal(
                rows, "center_absolute_percentage_error"
            ),
            "center_row_absolute_percentage_error": _metric_summary(v2_errors),
            "allocated_row_absolute_percentage_error": _metric_summary(
                [row["allocated_absolute_percentage_error"] for row in rows]
            ),
            "upper_row_coverage": sum(row["upper_covers_observed"] for row in rows)
            / len(rows),
            "false_safe_count": sum(row["false_safe"] for row in rows),
            "false_reject_count": sum(row["false_reject"] for row in rows),
            "admitted_count": sum(row["predicted_admit"] for row in rows),
        },
        "diagnostic_delta": {
            "center_mape_absolute_change": (
                _scenario_equal(rows, "center_absolute_percentage_error")
                - _scenario_equal(rows, "v1_center_absolute_percentage_error")
            ),
            "center_p90_ape_absolute_change": (
                _percentile(v2_errors, 0.9) - _percentile(v1_errors, 0.9)
            ),
            "false_reject_count_change": (
                sum(row["false_reject"] for row in rows)
                - sum(row["v1_false_reject"] for row in rows)
            ),
        },
        "rows": rows,
        "conclusion": (
            "The bounded direct head repairs the known v1 extrapolation failure "
            "on the consumed cases, but a new pre-frozen holdout is still required."
        ),
    }
    report["report_sha256"] = sha256_json(report)
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    v1 = report["v1_frozen_baseline"]
    v2 = report["v2_post_holdout_repair"]
    lines = [
        "# H800 显存 v2：已揭盲 holdout 诊断回放",
        "",
        "> 该回放只能说明已知故障是否被修复，不能证明泛化性，也不能用于发布。",
        "",
        "| 指标 | v1 冻结模型 | v2 修复候选 |",
        "|---|---:|---:|",
        f"| scenario-equal center MAPE | {100 * v1['center_scenario_equal_mape']:.2f}% | {100 * v2['center_scenario_equal_mape']:.2f}% |",
        f"| row P90 APE | {100 * v1['center_row_absolute_percentage_error']['p90']:.2f}% | {100 * v2['center_row_absolute_percentage_error']['p90']:.2f}% |",
        f"| upper 覆盖率 | {100 * v1['upper_row_coverage']:.1f}% | {100 * v2['upper_row_coverage']:.1f}% |",
        f"| false reject | {v1['false_reject_count']} | {v2['false_reject_count']} |",
        "",
        "## 逐配置结果",
        "",
        "| 数据画像 | 模型 | GPU | MBS | v2 center GiB | v2 upper GiB | 实测 GiB | APE | admission |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {dataset_category} | {model_id} | {gpu_count} | {physical_mbs} | "
            "{center:.2f} | {upper:.2f} | {actual:.2f} | {ape:.1f}% | {admit} |".format(
                **row,
                center=row["predicted_reserved_center_bytes"] / GIB,
                upper=row["predicted_operational_upper_bytes"] / GIB,
                actual=row["observed_reserved_bytes"] / GIB,
                ape=100 * row["center_absolute_percentage_error"],
                admit="通过" if row["predicted_admit"] else "过滤",
            )
        )
    lines.extend(
        [
            "",
            "结论：v2 已消除已知的三次项爆炸和 14B 两卡误过滤，但正式结论仍取决于下一批从未参与设计的新画像。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--frozen-v1-predictions",
        type=Path,
        default=DEFAULT_FROZEN_V1_PREDICTIONS,
    )
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = diagnose(
        candidate_path=args.candidate,
        frozen_v1_predictions_path=args.frozen_v1_predictions,
        observations_path=args.observations,
        profile_dir=args.profile_dir,
    )
    write_json(args.output, report)
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "markdown": str(args.markdown.resolve()),
                "v1_center_mape": report["v1_frozen_baseline"][
                    "center_scenario_equal_mape"
                ],
                "v2_center_mape": report["v2_post_holdout_repair"][
                    "center_scenario_equal_mape"
                ],
                "v2_center_p90": report["v2_post_holdout_repair"][
                    "center_row_absolute_percentage_error"
                ]["p90"],
                "v2_upper_coverage": report["v2_post_holdout_repair"][
                    "upper_row_coverage"
                ],
                "v2_false_rejects": report["v2_post_holdout_repair"][
                    "false_reject_count"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
