#!/usr/bin/env python3
"""Stage-2-center hybrid memory safety upper bound (development-set only).

This is the V3 safety head that pairs with the stage-2 hybrid center
(`h800_hybrid_memory_artifact_v2.json`).  It exists to avoid the V1 mistake
where the prospective acceptance rows were inspected and then used to pick the
upper multiplier: here the multipliers are derived ONLY from the stage-1
development observations (the same 208 rows the stage-2 center was fit on).

The Part C prospective acceptance data (0105_inference2 text, blind_data/vl
images) is NEVER read by this script; it is reserved strictly for evaluation.

Method (per mechanism key model_id::g{gpu}_z{zero}_gc{gc}):
  * compute the stage-2 center for each exact success development row;
  * take the worst actual/center ratio in the group and add a sparse-measurement
    margin, floored by a global finite-sample guard over all residuals;
  * the multiplier is one-sided (memory upper), never below the global guard.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from common import ARTIFACT_DIR, read_json, read_jsonl, sha256_file, sha256_json, write_json
from fit_h800_hybrid_attention_dense_stage2_v1 import (
    COEFFICIENTS,
    FORMAL_RECORDS,
    _design_row,
)

SCHEMA = "sft_h800_hybrid_vl_safety_upper/v3"
OUTPUT = ARTIFACT_DIR / "h800_hybrid_vl_safety_upper_v3.json"
CENTER_ARTIFACT = ARTIFACT_DIR / "h800_hybrid_memory_artifact_v2.json"
TARGET_COVERAGE = 0.95
SPARSE_HYBRID_MEASUREMENT_MARGIN = 0.02

_ZERO_STAGE = {"none": 0, "zero0": 0, "zero1": 1, "zero2": 2, "zero3": 3}


def _finite_sample_upper(values: list[float]) -> dict[str, Any]:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        raise ValueError("one-sided calibration requires residuals")
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


def _mechanism_key(observation: dict[str, Any]) -> str:
    configuration = observation["configuration"]
    zero = _ZERO_STAGE[str(configuration.get("zero") or "none").lower()]
    gc = int(bool(configuration.get("gradient_checkpointing")))
    return f"{observation['model_id']}::g{int(configuration['gpu_count'])}_z{zero}_gc{gc}"


def _v2_center(coef_vector: list[float], observation: dict[str, Any]) -> float:
    design = _design_row(observation)
    return max(sum(c * d for c, d in zip(coef_vector, design)), 1.0)


def fit() -> dict[str, Any]:
    artifact = read_json(CENTER_ARTIFACT)
    if artifact.get("schema") != "sft_h800_hybrid_memory_artifact/v2":
        raise ValueError("stage-2 center artifact schema mismatch")
    coefficients_by_name = artifact["coefficients_by_name"]
    coef_vector = [float(coefficients_by_name[name]) for name, _ in COEFFICIENTS]

    observations = read_jsonl(FORMAL_RECORDS)
    development = [
        row
        for row in observations
        if row["architecture_route"] == "dense_hybrid_attention"
        and row["outcome"].get("terminal_eligible")
        and row["outcome"]["classification"] == "success"
        and row["outcome"].get("peak_reserved_bytes")
    ]
    if not development:
        raise ValueError("no hybrid development successes available")

    rows = []
    all_log_ratios: list[float] = []
    for observation in development:
        center = _v2_center(coef_vector, observation)
        actual = float(observation["outcome"]["peak_reserved_bytes"])
        log_ratio = math.log(actual / center)
        all_log_ratios.append(log_ratio)
        rows.append(
            {
                "job_id": observation["job_id"],
                "model_id": observation["model_id"],
                "mechanism_key": _mechanism_key(observation),
                "center_bytes_v2": center,
                "actual_reserved_bytes": actual,
                "actual_to_center_ratio": actual / center,
                "log_actual_to_center": log_ratio,
            }
        )

    global_guard = _finite_sample_upper(all_log_ratios)
    fallback = float(global_guard["upper_multiplier"])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["mechanism_key"]].append(row)

    conditional = {}
    for key, values in sorted(grouped.items()):
        maximum_ratio = max(row["actual_to_center_ratio"] for row in values)
        selected = max(fallback, maximum_ratio * (1.0 + SPARSE_HYBRID_MEASUREMENT_MARGIN))
        conditional[key] = {
            "development_exact_rows": len(values),
            "maximum_actual_to_center_ratio": maximum_ratio,
            "fallback_global_multiplier": fallback,
            "sparse_measurement_margin_fraction": SPARSE_HYBRID_MEASUREMENT_MARGIN,
            "upper_multiplier": selected,
            "selected_source": (
                "development_max_plus_sparse_margin"
                if selected > fallback
                else "global_guard_floor"
            ),
            "job_ids": [row["job_id"] for row in values],
        }

    # In-sample development coverage: upper must dominate every development actual.
    covered = 0
    for row in rows:
        upper = row["center_bytes_v2"] * conditional[row["mechanism_key"]]["upper_multiplier"]
        if upper >= row["actual_reserved_bytes"]:
            covered += 1

    payload = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_coverage": TARGET_COVERAGE,
        "center_source": {
            "path": str(CENTER_ARTIFACT.resolve()),
            "sha256": sha256_file(CENTER_ARTIFACT),
            "policy": "stage2_hybrid_center_v2",
        },
        "development_source": {
            "path": str(FORMAL_RECORDS.resolve()),
            "sha256": sha256_file(FORMAL_RECORDS),
            "role": "stage1_development_only",
            "note": "Part C prospective data (0105_inference2, blind_data/vl) is NOT used here.",
        },
        "data_role": {
            "prospective_acceptance_data_used": False,
            "may_be_used_as_future_acceptance": True,
            "reason": "multipliers derived solely from stage-1 development observations",
        },
        "hybrid_memory": {
            "center_policy": "stage2_center_v2_unchanged",
            "fallback_global_upper_multiplier": fallback,
            "global_guard": global_guard,
            "conditional_key": "model_id::gpu_count_zero_stage_gradient_checkpointing",
            "conditional_by_model_mechanism": conditional,
            "development_coverage": {
                "exact_rows": len(rows),
                "covered_rows": covered,
                "exact_upper_coverage": covered / len(rows),
            },
        },
        "release": {"mode": "shadow_only", "automatic_admission_allowed": False},
    }
    payload["report_sha256"] = sha256_json(payload)
    write_json(OUTPUT, payload)
    return payload


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(fit(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
