#!/usr/bin/env python3
"""Fit V2 hybrid/VL safety heads after the V1 prospective acceptance failure.

The 27 prospective rows are no longer treated as acceptance evidence once they
are used here.  They become development observations for V2.  Hybrid centers
stay frozen and receive sparse model/mechanism-specific upper multipliers.
Image-VL centers receive model/mechanism-specific total-center reanchoring and
cross-tier one-sided guards.  Automatic release remains disabled until a new,
source-disjoint prospective campaign passes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, read_json, read_jsonl, sha256_file, sha256_json, write_json


SCHEMA = "sft_h800_hybrid_vl_safety_upper/v2"
TARGET_COVERAGE = 0.95
SPARSE_HYBRID_MEASUREMENT_MARGIN = 0.02
V1_SAFETY = ARTIFACT_DIR / "h800_hybrid_vl_safety_upper_v1.json"
V1_ACCEPTANCE = ARTIFACT_DIR / "h800_hybrid_vl_prospective_acceptance_report_v1.json"
FROZEN_PREDICTIONS = ARTIFACT_DIR / "h800_hybrid_vl_prospective_frozen_predictions_v1.json"
QUEUE = MATRIX_DIR / "h800_hybrid_vl_prospective_acceptance_v1.jsonl"
OUTPUT = ARTIFACT_DIR / "h800_hybrid_vl_safety_upper_v2.json"


def _finite_sample_upper(values: list[float]) -> dict[str, Any]:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        raise ValueError("one-sided V2 calibration requires exact residuals")
    rank = math.ceil((len(clean) + 1) * TARGET_COVERAGE)
    selected_rank = min(rank, len(clean))
    selected = max(0.0, clean[selected_rank - 1])
    return {
        "exact_rows": len(clean),
        "target_coverage": TARGET_COVERAGE,
        "finite_sample_rank": rank,
        "selected_rank": selected_rank,
        "rank_capped_at_sample_maximum": rank > len(clean),
        "selected_log_guard": selected,
        "upper_multiplier": math.exp(selected),
    }


def _load_development_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    acceptance = read_json(V1_ACCEPTANCE)
    if (
        acceptance.get("complete") is not True
        or acceptance.get("classifications") != {"success": 27}
        or acceptance.get("all_requested_gates_passed") is not False
    ):
        raise ValueError("V1 acceptance must be the complete failed 27-success campaign")
    queue = read_jsonl(QUEUE)
    if len(queue) != 27:
        raise ValueError("V2 development queue must contain exactly 27 rows")
    queue_by_id = {str(row["job_id"]): row for row in queue}
    observed_by_id = {str(row["job_id"]): row for row in acceptance["rows"]}
    if set(queue_by_id) != set(observed_by_id):
        raise ValueError("V1 acceptance rows no longer match the frozen queue")

    frozen = read_json(FROZEN_PREDICTIONS)
    frozen_by_id = {
        str(row["request_id"]): row
        for section in ("hybrid", "vl_image")
        for row in frozen[section]["predictions"]
    }
    if set(frozen_by_id) != set(queue_by_id):
        raise ValueError("frozen prediction rows no longer match the V2 development queue")

    rows = []
    for job_id, job in queue_by_id.items():
        observed = observed_by_id[job_id]
        frozen_row = frozen_by_id[job_id]
        if observed.get("classification") != "success":
            raise ValueError("V2 development currently accepts exact successes only")
        if job.get("design_arm") == "hybrid_memory_prospective":
            center = float(frozen_row["memory"]["reserved_center_bytes"])
            mechanism_key = (
                f"{job['model_id']}::g{int(job['gpu_count'])}_"
                f"z{int(job['zero_stage'])}_gc{int(bool(job['gc']))}"
            )
            track = "hybrid"
        elif job.get("design_arm") == "vl_image_prospective":
            center = float(frozen_row["vl_shadow"]["memory"]["predicted_center_bytes"])
            mechanism_key = f"{job['model_id']}::{job['mechanism_id']}"
            track = "vl_image"
        else:
            raise ValueError(f"unexpected V2 development arm: {job_id}")
        actual = float(observed["observed_reserved_bytes"])
        rows.append(
            {
                "job_id": job_id,
                "track": track,
                "model_id": str(job["model_id"]),
                "mechanism_key": mechanism_key,
                "media_tier": job.get("media_tier"),
                "comparison_group": str(observed["comparison_group"]),
                "center_bytes_v1": center,
                "actual_reserved_bytes": actual,
                "safe_limit_bytes": float(observed["safe_limit_bytes"]),
                "predicted_effective_tokens_per_second": float(
                    observed["predicted_effective_tokens_per_second"]
                ),
                "observed_effective_tokens_per_second": float(
                    observed["effective_tokens_per_second"]
                ),
                "log_actual_to_v1_center": math.log(actual / center),
            }
        )
    return rows, frozen


def _fit_hybrid(rows: list[dict[str, Any]], v1: dict[str, Any]) -> dict[str, Any]:
    subset = [row for row in rows if row["track"] == "hybrid"]
    fallback = float(v1["hybrid_memory"]["upper_multiplier"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset:
        grouped[row["mechanism_key"]].append(row)
    conditional = {}
    for key, values in sorted(grouped.items()):
        maximum_ratio = max(
            row["actual_reserved_bytes"] / row["center_bytes_v1"] for row in values
        )
        selected = max(fallback, maximum_ratio * (1.0 + SPARSE_HYBRID_MEASUREMENT_MARGIN))
        conditional[key] = {
            "development_exact_rows": len(values),
            "maximum_actual_to_center_ratio": maximum_ratio,
            "fallback_v1_multiplier": fallback,
            "sparse_measurement_margin_fraction": SPARSE_HYBRID_MEASUREMENT_MARGIN,
            "upper_multiplier": selected,
            "selected_source": (
                "development_max_plus_sparse_margin"
                if selected > fallback
                else "v1_global_guard_floor"
            ),
            "job_ids": [row["job_id"] for row in values],
        }
    replay = []
    for row in subset:
        multiplier = float(conditional[row["mechanism_key"]]["upper_multiplier"])
        upper = row["center_bytes_v1"] * multiplier
        replay.append(
            {
                **row,
                "center_bytes_v2": row["center_bytes_v1"],
                "upper_multiplier_v2": multiplier,
                "upper_bytes_v2": upper,
                "covered_v2": upper >= row["actual_reserved_bytes"],
                "admitted_v2": upper <= row["safe_limit_bytes"],
            }
        )
    return {
        "center_policy": "frozen_hybrid_center_unchanged",
        "fallback_v1_upper_multiplier": fallback,
        "conditional_key": "model_id::gpu_count_zero_stage_gradient_checkpointing",
        "conditional_by_model_mechanism": conditional,
        "development_replay": {
            "exact_rows": len(replay),
            "covered_rows": sum(row["covered_v2"] for row in replay),
            "exact_upper_coverage": sum(row["covered_v2"] for row in replay) / len(replay),
            "safe_success_rows": sum(
                row["actual_reserved_bytes"] <= row["safe_limit_bytes"] for row in replay
            ),
            "admitted_safe_success_rows": sum(
                row["actual_reserved_bytes"] <= row["safe_limit_bytes"]
                and row["admitted_v2"]
                for row in replay
            ),
            "rows": replay,
        },
    }


def _fit_vl_image(rows: list[dict[str, Any]]) -> dict[str, Any]:
    subset = [row for row in rows if row["track"] == "vl_image"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset:
        grouped[row["mechanism_key"]].append(row)
    conditional = {}
    for key, values in sorted(grouped.items()):
        if len(values) != 2 or {row["media_tier"] for row in values} != {"low", "high"}:
            raise ValueError(f"VL V2 requires one low/high pair for {key}")
        log_ratios = [row["log_actual_to_v1_center"] for row in values]
        final_log_scale = statistics.median(log_ratios)
        center_scale = math.exp(final_log_scale)
        oof_residuals = []
        oof_rows = []
        for held in values:
            training = [
                row["log_actual_to_v1_center"]
                for row in values
                if row["job_id"] != held["job_id"]
            ]
            oof_log_scale = statistics.median(training)
            adjusted_center = held["center_bytes_v1"] * math.exp(oof_log_scale)
            residual = math.log(held["actual_reserved_bytes"] / adjusted_center)
            oof_residuals.append(residual)
            oof_rows.append(
                {
                    "job_id": held["job_id"],
                    "held_out_tier": held["media_tier"],
                    "oof_total_center_scale": math.exp(oof_log_scale),
                    "oof_adjusted_center_bytes": adjusted_center,
                    "oof_log_residual": residual,
                }
            )
        guard = _finite_sample_upper(oof_residuals)
        conditional[key] = {
            "development_exact_rows": len(values),
            "total_center_scale": center_scale,
            "total_center_scale_form": (
                "exp(median(log(actual_reserved_bytes / v1_overlay_center_bytes)))"
            ),
            "upper_multiplier": float(guard["upper_multiplier"]),
            "one_sided_guard": guard,
            "cross_tier_oof_rows": oof_rows,
            "job_ids": [row["job_id"] for row in values],
        }
    replay = []
    for row in subset:
        spec = conditional[row["mechanism_key"]]
        center = row["center_bytes_v1"] * float(spec["total_center_scale"])
        upper = center * float(spec["upper_multiplier"])
        replay.append(
            {
                **row,
                "total_center_scale_v2": float(spec["total_center_scale"]),
                "center_bytes_v2": center,
                "upper_multiplier_v2": float(spec["upper_multiplier"]),
                "upper_bytes_v2": upper,
                "covered_v2": upper >= row["actual_reserved_bytes"],
                "admitted_v2": upper <= row["safe_limit_bytes"],
            }
        )
    safe = [
        row for row in replay
        if row["actual_reserved_bytes"] <= row["safe_limit_bytes"]
    ]
    return {
        "center_policy": "model_mechanism_total_center_reanchor_from_v1_overlay_center",
        "conditional_key": "model_id::SAFE_NOGC_PRESSURE",
        "conditional_by_model_mechanism": conditional,
        "development_replay": {
            "exact_rows": len(replay),
            "covered_rows": sum(row["covered_v2"] for row in replay),
            "exact_upper_coverage": sum(row["covered_v2"] for row in replay) / len(replay),
            "safe_success_rows": len(safe),
            "admitted_safe_success_rows": sum(row["admitted_v2"] for row in safe),
            "admission_recall": sum(row["admitted_v2"] for row in safe) / len(safe),
            "rows": replay,
        },
    }


def _ranking_replay(vl: dict[str, Any]) -> dict[str, Any]:
    rows = vl["development_replay"]["rows"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["comparison_group"]].append(row)
    details = []
    for group_id, candidates in sorted(groups.items()):
        actual_safe = [
            row for row in candidates
            if row["actual_reserved_bytes"] <= row["safe_limit_bytes"]
        ]
        predicted_pool = [row for row in candidates if row["admitted_v2"]]
        selected = max(
            predicted_pool,
            key=lambda row: row["predicted_effective_tokens_per_second"],
        )
        observed_best = max(
            actual_safe,
            key=lambda row: row["observed_effective_tokens_per_second"],
        )
        selected_rate = selected["observed_effective_tokens_per_second"]
        best_rate = observed_best["observed_effective_tokens_per_second"]
        regret = 1.0 - selected_rate / best_rate
        details.append(
            {
                "comparison_group": group_id,
                "predicted_winner": selected["job_id"],
                "observed_best": observed_best["job_id"],
                "top1_regret": regret,
                "top1_within_10_percent": regret <= 0.10,
            }
        )
    return {
        "groups": len(details),
        "hit_at_10_percent": sum(row["top1_within_10_percent"] for row in details) / len(details),
        "worst_top1_regret": max(row["top1_regret"] for row in details),
        "details": details,
    }


def fit() -> dict[str, Any]:
    v1 = read_json(V1_SAFETY)
    rows, _ = _load_development_rows()
    hybrid = _fit_hybrid(rows, v1)
    vl_image = _fit_vl_image(rows)
    vl_image["development_ranking_replay"] = _ranking_replay(vl_image)
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_coverage": TARGET_COVERAGE,
        "data_role": {
            "v1_campaign_original_role": "source_disjoint_prospective_acceptance",
            "v2_role_after_inspection": "development_only",
            "may_be_used_as_future_v2_acceptance": False,
            "reason": "outcomes were inspected and used to select V2 corrections",
        },
        "hybrid_memory": hybrid,
        "vl_memory_by_modality": {
            "image": vl_image,
            "video": {
                "policy": "unchanged_v1_shadow_fallback",
                "v1_spec": v1["vl_memory_by_modality"]["video"],
                "automatic_admission_allowed": False,
            },
            "v1_text_base_scale_by_model_mechanism": v1["vl_memory_by_modality"][
                "text_base_scale_by_model_mechanism"
            ],
        },
        "source_bindings": {
            "v1_safety": {"path": str(V1_SAFETY.resolve()), "sha256": sha256_file(V1_SAFETY)},
            "v1_acceptance": {"path": str(V1_ACCEPTANCE.resolve()), "sha256": sha256_file(V1_ACCEPTANCE)},
            "frozen_predictions": {"path": str(FROZEN_PREDICTIONS.resolve()), "sha256": sha256_file(FROZEN_PREDICTIONS)},
            "queue": {"path": str(QUEUE.resolve()), "sha256": sha256_file(QUEUE)},
            "implementation": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        },
        "release_contract": {
            "mode": "candidate_pending_new_source_acceptance_v2",
            "automatic_admission_allowed": False,
            "automatic_ranking_allowed": False,
            "vl_packing_allowed": False,
            "pure_text_packing_governed_by_main_predictor": True,
            "video_allowed": False,
        },
    }
    artifact["report_sha256"] = sha256_json(artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    artifact = fit()
    write_json(args.output, artifact)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
