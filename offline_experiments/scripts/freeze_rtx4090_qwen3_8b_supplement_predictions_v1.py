#!/usr/bin/env python3
"""Frozen admission predictions for the proposed RTX 4090 Qwen3-8B LoRA supplement.

The repo's protocol is predict-then-measure: freeze what the model says BEFORE
occupying cards, so the run is a test of the model rather than a fishing trip.

This enumerates the proposed supplement grid, runs it through the joint two-card
V3 memory model (``artifacts/joint_card_v3_memory_v1.json``) at RTX 4090
capacity, and writes the frozen prediction for every cell.

Occupies no GPU.  Writes one artifact.  Reads everything else.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import ROOT, read_json, sha256_json, write_json

import benchmark_h800_memory_center_models_v1 as bm
import fit_h800_unified_resource_partial_v1 as base
import fit_joint_card_v3_memory_v1 as joint

SCHEMA = "sft_rtx4090_qwen3_8b_lora_supplement_frozen_predictions/v1"
JOINT_ARTIFACT = ROOT / "artifacts" / "joint_card_v3_memory_v1.json"
V3_INVENTORY = ROOT / "artifacts" / "h800_bounded_memory_v2_model_inventory_v1.json"
PROFILE_DIR = ROOT / "campaigns" / "rtx4090_20260717" / "artifacts" / "dataset_profiles"
RTX4090_CAPACITY_BYTES = 25386352640
SAFE_LIMIT_FRACTION = 0.95
MODEL_ID = "qwen3_8b"

# Why this grid: the 4090 side currently has 0.6B / 1.7B / 4B only, so the
# cross-card anchors sit at 1.7B and 4B.  8B is the smallest scale that both
# extends the 4090 range upward AND overlaps H800 (which has 8B).  Everything
# else is held to values the existing 4090 campaign already exercises, so the
# supplement adds one axis rather than a new regime.
GRID = {
    "gpu_count": (1, 2, 4),
    "zero_stage": (0, 2, 3),
    "gradient_checkpointing": (True, False),
    "mbs": (1, 2, 4),
    "dataset_id": ("short_512", "multiturn_4096", "longtail_8192"),
}
CUTOFF_BY_DATASET = {
    "short_512": 512,
    "multiturn_4096": 4096,
    "longtail_8192": 8192,
}


def valid(combo: dict) -> str | None:
    """Configuration-level exclusions, independent of memory."""
    if combo["gpu_count"] == 1 and combo["zero_stage"] in (2, 3):
        return "deepspeed_does_not_support_single_gpu_zero2_3"
    if combo["gpu_count"] > 1 and combo["zero_stage"] == 0:
        return "multi_gpu_requires_zero2_or_zero3_in_this_campaign"
    return None


def build_record(combo: dict, model_by_id, fixed_lora, profile_cache) -> dict:
    cutoff = CUTOFF_BY_DATASET[combo["dataset_id"]]
    profile_path = PROFILE_DIR / f"{combo['dataset_id']}.qwen3_nothink.jsonl"
    from analyze_h800_fresh_memory_residual_v2 import profile_padding_statistics

    key = (str(profile_path.resolve()), cutoff, int(combo["mbs"]))
    if key not in profile_cache:
        profile_cache[key] = profile_padding_statistics(
            profile_path, cutoff_len=cutoff, physical_mbs=int(combo["mbs"])
        )
    raw_max = int(profile_cache[key]["maximum_clipped_tokens"])
    aligned = 8 * ((min(cutoff, raw_max) + 7) // 8)
    job = {
        "model_id": MODEL_ID,
        "model_parameters": int(model_by_id[MODEL_ID]["actual_parameters"]),
        "train_type": "lora",
        "gpu_count": int(combo["gpu_count"]),
        "mbs": int(combo["mbs"]),
        "cutoff_len": cutoff,
        "aligned_effective_sequence": int(aligned),
        "zero": "none" if combo["zero_stage"] == 0 else f"zero{combo['zero_stage']}",
        "zero_stage": int(combo["zero_stage"]),
        "gc": bool(combo["gradient_checkpointing"]),
        "packing": False,
        "dataset_id": combo["dataset_id"],
        "dataset_profile_path": str(profile_path),
    }
    reference, values = base._current_features(
        job,
        model_by_id=model_by_id,
        fixed_lora=fixed_lora,
        capacity_bytes=RTX4090_CAPACITY_BYTES,
        profile_cache={},
    )
    values.update(
        joint.capacity_features(
            values,
            reference_bytes=reference,
            capacity_bytes=RTX4090_CAPACITY_BYTES,
            model_id=MODEL_ID,
        )
    )
    record = {
        "features": values,
        "reference_bytes": reference,
        "model_id": MODEL_ID,
        "_card": "rtx4090",
        "_capacity_bytes": RTX4090_CAPACITY_BYTES,
    }
    record["features"].update(joint.card_feature_values(record))
    record["_job"] = job
    return record


def predict(record, model) -> float:
    names = [str(n) for n in model["raw_feature_names"]]
    raw = bm._raw_matrix([record], names)
    expanded = bm._expand_basis(
        raw,
        kind=str(model["basis_kind"]),
        raw_means=np.asarray(model["raw_means"], dtype=float),
        raw_scales=np.asarray(model["raw_scales"], dtype=float),
        nonlinear_indexes=[int(v) for v in model["nonlinear_indexes"]],
        feature_names=names,
    )
    standardized = (
        expanded - np.asarray(model["expanded_means"], dtype=float)
    ) / np.asarray(model["expanded_scales"], dtype=float)
    correction = float(model["correction_shrinkage"]) * (
        float(model["intercept"])
        + float(standardized[0] @ np.asarray(model["coefficients"], dtype=float))
    )
    return float(record["reference_bytes"]) * math.exp(correction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts"
        / "rtx4090_qwen3_8b_lora_supplement_frozen_predictions_v1.json",
    )
    args = parser.parse_args()

    report = read_json(JOINT_ARTIFACT)
    centre_model = report["frozen_model"]["centre"]
    risk_model = report["frozen_model"]["risk"]
    multiplier = float(
        report["evaluation"]["upper_multiplier_calibration"][
            "recalibrated_per_card"
        ]["rtx4090"]
    )
    inventory = read_json(V3_INVENTORY)
    model_by_id = {str(m["id"]): dict(m) for m in inventory["models"]}
    fixed_lora = dict(inventory["fixed_lora"])
    safe_limit = RTX4090_CAPACITY_BYTES * SAFE_LIMIT_FRACTION

    profile_cache: dict = {}
    cells = []
    excluded = Counter()
    keys = sorted(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        combo = dict(zip(keys, values))
        reason = valid(combo)
        if reason:
            excluded[reason] += 1
            continue
        record = build_record(combo, model_by_id, fixed_lora, profile_cache)
        centre = predict(record, centre_model)
        risk = predict(record, risk_model)
        upper = max(centre, risk * multiplier)
        cells.append(
            {
                **combo,
                "cutoff_len": CUTOFF_BY_DATASET[combo["dataset_id"]],
                "aligned_effective_sequence": record["_job"][
                    "aligned_effective_sequence"
                ],
                "analytic_reference_bytes": record["reference_bytes"],
                "predicted_centre_bytes": centre,
                "predicted_risk_bytes": risk,
                "predicted_upper_bytes": upper,
                "predicted_upper_gib": upper / float(1 << 30),
                "safe_limit_bytes": safe_limit,
                "predicted_admitted": bool(upper <= safe_limit),
                "headroom_gib": (safe_limit - upper) / float(1 << 30),
            }
        )

    admitted = [c for c in cells if c["predicted_admitted"]]
    rejected = [c for c in cells if not c["predicted_admitted"]]
    # Boundary probes: the rejected cells closest to the limit are the most
    # informative to actually run, because that is where the model is least
    # sure and where a wrong answer costs a user a crashed job.
    rejected_sorted = sorted(rejected, key=lambda c: -c["headroom_gib"])

    payload = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_measurement",
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "purpose": (
            "Freeze what the joint two-card V3 memory model predicts for a "
            "proposed RTX 4090 Qwen3-8B LoRA supplement, BEFORE any card is "
            "occupied, so the run tests the model instead of exploring."
        ),
        "why_8b": (
            "The 4090 side has 0.6B/1.7B/4B only, so cross-card anchors sit at "
            "1.7B and 4B.  8B is the smallest scale that both extends the 4090 "
            "range and overlaps H800.  Adding it is the fix for the 24.3% "
            "false-reject rate, which is a data-coverage problem, not a "
            "hyper-parameter one."
        ),
        "model": {
            "id": MODEL_ID,
            "path": model_by_id[MODEL_ID].get("path"),
            "actual_parameters": model_by_id[MODEL_ID].get("actual_parameters"),
        },
        "card": {
            "card_id": "rtx4090",
            "capacity_bytes": RTX4090_CAPACITY_BYTES,
            "safe_limit_fraction": SAFE_LIMIT_FRACTION,
            "safe_limit_bytes": safe_limit,
        },
        "grid": {k: list(v) for k, v in GRID.items()},
        "cutoff_by_dataset": CUTOFF_BY_DATASET,
        "configuration_exclusions": dict(excluded),
        "summary": {
            "cells_total": len(cells),
            "predicted_admitted": len(admitted),
            "predicted_rejected": len(rejected),
            "admitted_by_gpu_count": dict(
                Counter(c["gpu_count"] for c in admitted)
            ),
            "admitted_by_dataset": dict(
                Counter(c["dataset_id"] for c in admitted)
            ),
        },
        "upper_multiplier_used": multiplier,
        "source_bindings": {
            "joint_memory_artifact": {
                "path": str(JOINT_ARTIFACT),
                "report_sha256": report["report_sha256"],
            },
            "v3_inventory": {"path": str(V3_INVENTORY)},
        },
        "recommended_boundary_probes": rejected_sorted[:8],
        "cells": cells,
    }
    payload["report_sha256"] = sha256_json(payload)
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cells_total": len(cells),
                "predicted_admitted": len(admitted),
                "predicted_rejected": len(rejected),
                "configuration_exclusions": dict(excluded),
                "admitted_by_gpu_count": payload["summary"][
                    "admitted_by_gpu_count"
                ],
                "admitted_by_dataset": payload["summary"]["admitted_by_dataset"],
            },
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
