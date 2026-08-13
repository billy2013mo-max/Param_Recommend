#!/usr/bin/env python3
"""Run the active H800 memory-V3 plus structured-throughput-V5 predictor."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
PREDICTOR_SCRIPTS = PROJECT_ROOT / "offline_experiments" / "scripts"
if str(PREDICTOR_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PREDICTOR_SCRIPTS))

from h800_resource_predictor import (
    H800ResourcePredictor,
    validate_prediction_report,
)

TARGET_GBS = 64
GPU_COUNTS = (1, 2, 4)
MBS_VALUES = (1, 2, 4, 8, 16)
CUTOFFS = (2048, 4096, 8192)
SCENARIOS = (
    {
        "scenario_name": "qwen3-8b-lora-business-yt0jrk",
        "dataset_id": "business_yt0jrk_v2",
        "model_id": "qwen3_8b",
        "training_mode": "lora",
        "lora_rank": 32,
        "profile_tokenizer_id": "Qwen3-8B@local",
    },
    {
        "scenario_name": "qwen3-14b-full-business-flieht",
        "dataset_id": "business_flieht_v4",
        "model_id": "qwen3_14b",
        "training_mode": "full",
        "lora_rank": 32,
        "profile_tokenizer_id": "Qwen3-14B@local",
    },
)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def requests() -> list[dict[str, Any]]:
    result = []
    for scenario in SCENARIOS:
        for cutoff in CUTOFFS:
            group = f"{scenario['scenario_name']}-cutoff{cutoff}-gbs64"
            for gpu_count in GPU_COUNTS:
                zero_stages = (0,) if gpu_count == 1 else (2, 3)
                for zero_stage in zero_stages:
                    for mbs in MBS_VALUES:
                        if TARGET_GBS % (gpu_count * mbs):
                            continue
                        for gc in (False, True):
                            result.append(
                                {
                                    "request_id": (
                                        f"{group}-gpu{gpu_count}-z"
                                        f"{zero_stage}-mbs{mbs}-gc{int(gc)}"
                                    ),
                                    "comparison_group": group,
                                    "hardware_id": "h800",
                                    "model_id": scenario["model_id"],
                                    "training_mode": scenario["training_mode"],
                                    "lora_rank": scenario["lora_rank"],
                                    "dataset_id": scenario["dataset_id"],
                                    "dataset_category": "longtail",
                                    "profile_tokenizer_id": scenario[
                                        "profile_tokenizer_id"
                                    ],
                                    "profile_template_id": (
                                        "qwen3_nothink@llamafactory"
                                    ),
                                    "target_gbs": TARGET_GBS,
                                    "cutoff_len": cutoff,
                                    "gpu_count": gpu_count,
                                    "physical_mbs": mbs,
                                    "zero_stage": zero_stage,
                                    "gradient_checkpointing": gc,
                                    "packing": False,
                                    "dtype": "bf16",
                                }
                            )
    return result


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    rows = {row["request_id"]: row for row in report["predictions"]}
    groups = []
    for group in report["ranking_groups"]:
        ranked = [rows[value] for value in group["ranked_request_ids"]]
        minimum_gpu = min(
            (row["configuration"]["gpu_count"] for row in ranked),
            default=None,
        )
        minimum_gpu_rows = [
            row for row in ranked if row["configuration"]["gpu_count"] == minimum_gpu
        ]
        economical = max(
            ranked,
            key=lambda row: (
                row["throughput"]["predicted_effective_tokens_per_second"]
                / row["configuration"]["gpu_count"],
                -row["input_index"],
            ),
            default=None,
        )
        groups.append(
            {
                **group,
                "selected_configuration": (
                    rows[group["selected_request_id"]]["configuration"]
                    if group["selected_request_id"]
                    else None
                ),
                "selected_memory_operational_p95_gib": (
                    rows[group["selected_request_id"]]["memory"]["gib"][
                        "operational_p95"
                    ]
                    if group["selected_request_id"]
                    else None
                ),
                "selected_memory_operational_p95_is_compatibility_alias": (
                    True if group["selected_request_id"] else None
                ),
                "selected_memory_center_gib": (
                    rows[group["selected_request_id"]]["memory"]["gib"][
                        "reserved_center"
                    ]
                    if group["selected_request_id"]
                    else None
                ),
                "selected_memory_risk_guard_gib": (
                    rows[group["selected_request_id"]]["memory"]["gib"]["risk_guard"]
                    if group["selected_request_id"]
                    else None
                ),
                "selected_memory_admission_upper_gib": (
                    rows[group["selected_request_id"]]["memory"]["gib"][
                        "admission_upper"
                    ]
                    if group["selected_request_id"]
                    else None
                ),
                "selected_memory_admission_source": (
                    rows[group["selected_request_id"]]["memory"]["admission_source"]
                    if group["selected_request_id"]
                    else None
                ),
                "throughput_top_configuration": (
                    rows[group["throughput_top_request_id"]]["configuration"]
                    if group["throughput_top_request_id"]
                    else None
                ),
                "top3": [
                    {
                        "request_id": row["request_id"],
                        "configuration": row["configuration"],
                        "memory_operational_p95_gib": row["memory"]["gib"][
                            "operational_p95"
                        ],
                        "memory_operational_p95_is_compatibility_alias": True,
                        "memory_center_gib": row["memory"]["gib"]["reserved_center"],
                        "memory_risk_guard_gib": row["memory"]["gib"]["risk_guard"],
                        "memory_admission_upper_gib": row["memory"]["gib"][
                            "admission_upper"
                        ],
                        "memory_admission_source": row["memory"]["admission_source"],
                        "predicted_effective_tokens_per_second": row["throughput"][
                            "predicted_effective_tokens_per_second"
                        ],
                    }
                    for row in ranked[:3]
                ],
                "minimum_gpu_safe_request_id": (
                    minimum_gpu_rows[0]["request_id"] if minimum_gpu_rows else None
                ),
                "effective_tokens_per_second_per_gpu_request_id": (
                    economical["request_id"] if economical else None
                ),
            }
        )
    return {
        "schema": "business_text_recommendation_summary/v5",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictor_schema": report["schema"],
        "release": report["release"],
        "selection_policy": report["selection_policy"],
        "throughput_model": report["throughput_model"],
        "single_output_used_for_absolute_and_ranking": report[
            "single_output_used_for_absolute_and_ranking"
        ],
        "absolute_throughput_scale_trusted": report[
            "absolute_throughput_scale_trusted"
        ],
        "memory_upper_semantics": (
            "admission_upper=max(center,risk_guard*risk_guard_multiplier); "
            "operational_p95 is a compatibility alias, not a P95"
        ),
        "groups": groups,
        "gpu_training_started": False,
    }


def main() -> None:
    candidate_requests = requests()
    request_path = ROOT / "requests" / "text_candidates.json"
    prediction_path = ROOT / "predictions" / "text_predictions.json"
    summary_path = ROOT / "predictions" / "text_recommendation_summary.json"
    write_json_atomic(
        request_path,
        {
            "schema": "business_text_candidate_request/v1",
            "candidates": candidate_requests,
        },
    )
    predictor = H800ResourcePredictor(additional_dataset_profile_dir=ROOT / "profiles")
    report = predictor.predict(candidate_requests)
    validate_prediction_report(report)
    write_json_atomic(prediction_path, report)
    summary = summarize(report)
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
