#!/usr/bin/env python3
"""V4 hybrid memory safety upper: mechanism-aware, wider guard, old-source only.

V2 acceptance showed the global guard (x1.0988) was too tight outside the
stage-1 source: GC-off and large-model multi-GPU ZeRO-3 rows reached 1.17-1.28
x center on the novel source.  The V4 upper keeps one multiplier per mechanism
(model::gpu::zero::gc), floored by a global guard, and applies an explicit
source-drift margin on TOP of the old-source maximum ratio of that mechanism.

The margin is a fixed engineering safety factor (not tuned on any acceptance
row); all ratio inputs come from the old stage-1 development set.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, read_json, read_jsonl, sha256_file, sha256_json, write_json
from fit_h800_hybrid_attention_dense_stage3_v1 import COEFFICIENTS, FORMAL_RECORDS, _design_row

SCHEMA = "sft_h800_hybrid_vl_safety_upper/v4"
OUTPUT = ARTIFACT_DIR / "h800_hybrid_vl_safety_upper_v4.json"
CENTER_ARTIFACT = ARTIFACT_DIR / "h800_hybrid_memory_artifact_v3.json"
SOURCE_DRIFT_MARGIN = 1.15  # fixed engineering safety factor over old-source max ratio


def _v3_center(coef_vector: list[float], observation: dict[str, Any]) -> float:
    design = _design_row(observation)
    return max(sum(c * d for c, d in zip(coef_vector, design)), 1.0)


def fit() -> dict[str, Any]:
    artifact = read_json(CENTER_ARTIFACT)
    if artifact.get("schema") != "sft_h800_hybrid_memory_artifact/v3":
        raise ValueError("stage-3 center artifact schema mismatch")
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
    ratios: list[tuple[dict[str, Any], float]] = []
    all_log_ratios: list[float] = []
    for observation in development:
        center = _v3_center(coef_vector, observation)
        actual = float(observation["outcome"]["peak_reserved_bytes"])
        ratio = actual / center
        ratios.append((observation, ratio))
        all_log_ratios.append(math.log(ratio))

    def pct_log(values: list[float], q: float) -> float:
        ordered = sorted(values)
        idx = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
        return math.exp(ordered[idx])

    global_guard = pct_log(all_log_ratios, 1.0) * SOURCE_DRIFT_MARGIN

    grouped: dict[str, list[float]] = defaultdict(list)
    config_index: dict[str, dict[str, Any]] = {}
    for observation, ratio in ratios:
        cfg = observation["configuration"]
        zero = str(cfg.get("zero") or "none")
        key = (
            f"{observation['model_id']}::g{int(cfg['gpu_count'])}_"
            f"z{'0' if zero == 'none' else zero.replace('zero', '')}_"
            f"gc{int(bool(cfg.get('gradient_checkpointing')))}"
        )
        grouped[key].append(ratio)
        config_index[key] = observation

    conditional = {}
    for key, values in sorted(grouped.items()):
        mechanism_max = max(values)
        selected = max(global_guard, mechanism_max * SOURCE_DRIFT_MARGIN)
        observation = config_index[key]
        cfg = observation["configuration"]
        conditional[key] = {
            "development_exact_rows": len(values),
            "development_max_ratio": mechanism_max,
            "source_drift_margin": SOURCE_DRIFT_MARGIN,
            "upper_multiplier": selected,
            "selected_source": (
                "mechanism_max_times_drift_margin"
                if selected > global_guard
                else "global_guard_floor"
            ),
            "model_id": observation["model_id"],
            "gpu_count": int(cfg["gpu_count"]),
            "zero": str(cfg.get("zero") or "none"),
            "gradient_checkpointing": bool(cfg.get("gradient_checkpointing")),
        }

    payload = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "center_source": {"path": str(CENTER_ARTIFACT.resolve()), "sha256": sha256_file(CENTER_ARTIFACT)},
        "development_source": {
            "path": str(FORMAL_RECORDS.resolve()),
            "sha256": sha256_file(FORMAL_RECORDS),
            "role": "stage1_development_only",
        },
        "data_role": {
            "prospective_acceptance_data_used": False,
            "may_be_used_as_future_acceptance": True,
            "reason": "all multipliers derive from old stage-1 rows plus a fixed margin",
        },
        "hybrid_memory": {
            "center_policy": "stage3_center_v3_unchanged",
            "source_drift_margin": SOURCE_DRIFT_MARGIN,
            "global_guard_multiplier": global_guard,
            "conditional_by_model_mechanism": conditional,
            "development_coverage": {
                "exact_rows": len(development),
                "covered_rows": sum(
                    1 for observation, ratio in ratios if ratio <= global_guard
                ),
            },
        },
        "release": {"mode": "shadow_only", "automatic_admission_allowed": False},
    }
    payload["report_sha256"] = sha256_json(payload)
    write_json(OUTPUT, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(fit(), ensure_ascii=False, indent=2))
