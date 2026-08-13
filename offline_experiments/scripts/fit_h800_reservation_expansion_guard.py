#!/usr/bin/env python3
"""Model reserved/allocated explicitly instead of letting a loose anchor hide it.

The analytic basis sums tensor bytes, so it predicts *allocated*.  Admission is
decided by *reserved* -- what the caching allocator asks the driver for.  The gap
is real and mechanism-dependent: across 1367 successful H800 rows the median is
1.098, but ``lora + zero2 + gc_off`` sits at 1.412 and reaches 2.181.

The frozen pipeline never modelled that gap.  It survived because a loose
``cutoff_len`` anchor inflated every prediction enough to cover it by accident.
Tighten the anchor (see fit_h800_effective_sequence_challenger) and the gap turns
into a false-safe admission -- an 8B LoRA row observed at 137.56 GiB was admitted
against a 132.84 GiB limit.

So the two changes have to ship together.  This module supplies the missing
piece: a per-selector expansion guard applied to an allocated-semantics center.

Two properties keep it from becoming another unbounded extrapolation:

* The guard is a per-selector empirical quantile, never a fitted polynomial, so
  it cannot extrapolate outside the calibrated range.
* Floors are reported uncapped.  An earlier version clamped them at device
  capacity on the theory that ``reserved`` cannot exceed it.  That was wrong: the
  clamp replaced an over-capacity floor with a constant 139.83 GiB that still
  exceeds the 132.84 GiB safe limit, so every affected candidate was rejected
  regardless of evidence.  A floor above capacity is a genuine infeasibility
  signal and is surfaced as one.

Unseen selectors fail closed rather than falling back to a pooled ratio: pooling
would apply the benign 1.098 median to the very bucket that reaches 2.181.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import statistics
from typing import Any

from common import ARTIFACT_DIR, read_json

GIB = float(1 << 30)
SCHEMA = "sft_h800_reservation_expansion_guard/v1"
IMPLEMENTATION_VERSION = (
    "sft_h800_reservation_expansion_guard/2026-08-04.per-selector-quantile-v1"
)

# Structural bound declared before fitting.  A selector whose empirical quantile
# lands outside this range means the mechanism is not understood well enough to
# guard, not that the guard should stretch.
MINIMUM_PLAUSIBLE_RATIO = 1.0
MAXIMUM_PLAUSIBLE_RATIO = 3.0

# Minimum independent rows before a selector earns its own guard.
MINIMUM_SELECTOR_ROWS = 12


def selector_key(
    *,
    training_mode: str,
    zero_stage: int,
    gradient_checkpointing: bool,
    gpu_count: int,
) -> str:
    """Mechanism key for the reservation guard.

    Includes gpu_count: under ZeRO the per-rank working set changes with the shard
    count, and the expansion ratio follows.  For full/zero3/gc_on the 2-GPU rows
    sit at q95 1.112 while the 4-GPU rows reach 1.359 -- pooling them charges every
    2-GPU candidate a tail risk it does not carry.

    Deliberately excludes model id, dataset id, MBS and cutoff.  Allocator
    behaviour is a property of the execution mechanism; keying on model or dataset
    would make this a lookup table rather than a model.  The empirical spread
    within each selector is carried by the quantile, not by more keys.
    """

    return json.dumps(
        [
            str(training_mode),
            int(zero_stage),
            bool(gradient_checkpointing),
            int(gpu_count),
        ],
        separators=(",", ":"),
    )


def _zero_stage(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value or "").lower()
    if text in {"", "none", "zero0", "0"}:
        return 0
    for stage in (1, 2, 3):
        if text in {f"zero{stage}", str(stage)}:
            return stage
    raise ValueError(f"unrecognized zero stage {value!r}")


def _quantile(values: Sequence[float], probability: float) -> float:
    """Conservative order-statistic quantile: index up, never interpolate down."""

    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    rank = min(count - 1, math.ceil(probability * count) - 1)
    return ordered[max(0, rank)]


def collect_observations(path: Path) -> list[dict[str, Any]]:
    """Successful rows that report both allocated and reserved peaks."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            outcome = row.get("outcome")
            if isinstance(outcome, Mapping):
                outcome = outcome.get("class")
            if str(outcome or "").lower() != "success":
                continue
            memory = (row.get("measurements") or {}).get("memory") or {}
            allocated = memory.get("max_allocated_bytes") or memory.get("max_allocated")
            reserved = memory.get("max_reserved_bytes") or memory.get("max_reserved")
            if not allocated or not reserved:
                continue
            job = row["configuration"]["job"]
            rows.append(
                {
                    "observation_id": row["observation_id"],
                    "training_mode": str(job["train_type"]),
                    "zero_stage": _zero_stage(job.get("zero")),
                    "gradient_checkpointing": bool(job.get("gc")),
                    "model_id": str(job.get("model_id")),
                    "dataset_id": str(job.get("dataset_id")),
                    "gpu_count": int(job["gpu_count"]),
                    "physical_mbs": int(job["mbs"]),
                    "cutoff_len": int(job["cutoff_len"]),
                    "allocated_bytes": float(allocated),
                    "reserved_bytes": float(reserved),
                    "ratio": float(reserved) / float(allocated),
                }
            )
    return rows


def fit_guard(
    rows: Sequence[Mapping[str, Any]], *, coverage: float = 0.95
) -> dict[str, Any]:
    """Per-selector expansion quantiles, weighted so no scenario dominates."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = selector_key(
            training_mode=row["training_mode"],
            zero_stage=row["zero_stage"],
            gradient_checkpointing=row["gradient_checkpointing"],
            gpu_count=row["gpu_count"],
        )
        grouped.setdefault(key, []).append(row)

    guards: dict[str, Any] = {}
    rejected: dict[str, Any] = {}
    for key, group in sorted(grouped.items()):
        ratios = [float(row["ratio"]) for row in group]
        if len(group) < MINIMUM_SELECTOR_ROWS:
            rejected[key] = {
                "reason": "insufficient_rows",
                "rows": len(group),
                "minimum_rows": MINIMUM_SELECTOR_ROWS,
            }
            continue
        guard = _quantile(ratios, coverage)
        if not MINIMUM_PLAUSIBLE_RATIO <= guard <= MAXIMUM_PLAUSIBLE_RATIO:
            rejected[key] = {
                "reason": "outside_declared_structural_range",
                "quantile": guard,
                "declared_range": [MINIMUM_PLAUSIBLE_RATIO, MAXIMUM_PLAUSIBLE_RATIO],
            }
            continue
        guards[key] = {
            "expansion_quantile": guard,
            "coverage": coverage,
            "rows": len(group),
            "independent_scenarios": len(
                {(row["model_id"], row["dataset_id"]) for row in group}
            ),
            "median": statistics.median(ratios),
            "maximum": max(ratios),
            "source": "per_selector_empirical_order_statistic",
        }
    return {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "coverage": coverage,
        "selector_key_fields": [
            "training_mode",
            "zero_stage",
            "gradient_checkpointing",
            "gpu_count",
        ],
        "declared_structural_range": [MINIMUM_PLAUSIBLE_RATIO, MAXIMUM_PLAUSIBLE_RATIO],
        "minimum_selector_rows": MINIMUM_SELECTOR_ROWS,
        "fit_rows": len(rows),
        "guards": guards,
        "rejected_selectors": rejected,
        "unseen_selector_policy": "fail_closed_no_pooled_fallback",
        "capacity_policy": (
            "floors are reported uncapped; a floor above capacity is a real "
            "infeasibility signal, not something to clamp away"
        ),
    }


def predict_reserved(
    *,
    allocated_center_bytes: float,
    training_mode: str,
    zero_stage: int,
    gradient_checkpointing: bool,
    gpu_count: int,
    capacity_bytes: float,
    guard: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand an allocated-semantics center into a reserved-semantics bound."""

    key = selector_key(
        training_mode=training_mode,
        zero_stage=zero_stage,
        gradient_checkpointing=gradient_checkpointing,
        gpu_count=gpu_count,
    )
    entry = (guard.get("guards") or {}).get(key)
    if not isinstance(entry, Mapping):
        return {
            "available": False,
            "selector": key,
            "issues": ["unseen_reservation_selector"],
        }
    expansion = float(entry["expansion_quantile"])
    raw = float(allocated_center_bytes) * expansion
    # Deliberately NOT capped at capacity.  Clamping to capacity would replace an
    # over-capacity floor with a constant that still exceeds the safe limit, which
    # rejects the candidate just as surely while hiding how far over it was.  A
    # floor above capacity is a real signal: the allocator would have to ask for
    # more than the device has.
    return {
        "available": True,
        "selector": key,
        "expansion_quantile": expansion,
        "reserved_bytes": raw,
        "exceeds_capacity": raw > float(capacity_bytes),
        "issues": [],
    }


def evaluate(
    rows: Sequence[Mapping[str, Any]], guard: Mapping[str, Any], *, capacity_bytes: float
) -> dict[str, Any]:
    """Leave-one-scenario-out coverage of the guard against observed reserved."""

    scenarios = sorted({(row["model_id"], row["dataset_id"]) for row in rows})
    covered = 0
    scored = 0
    unavailable = 0
    overshoot: list[float] = []
    for held in scenarios:
        training = [
            row
            for row in rows
            if (row["model_id"], row["dataset_id"]) != held
        ]
        evaluation = [
            row for row in rows if (row["model_id"], row["dataset_id"]) == held
        ]
        refit = fit_guard(training, coverage=float(guard["coverage"]))
        for row in evaluation:
            prediction = predict_reserved(
                allocated_center_bytes=row["allocated_bytes"],
                training_mode=row["training_mode"],
                zero_stage=row["zero_stage"],
                gradient_checkpointing=row["gradient_checkpointing"],
                gpu_count=row["gpu_count"],
                capacity_bytes=capacity_bytes,
                guard=refit,
            )
            if not prediction["available"]:
                unavailable += 1
                continue
            scored += 1
            predicted = float(prediction["reserved_bytes"])
            if predicted >= float(row["reserved_bytes"]):
                covered += 1
            overshoot.append(predicted / float(row["reserved_bytes"]))
    return {
        "protocol": "leave_one_model_dataset_scenario_out",
        "scenarios": len(scenarios),
        "scored_rows": scored,
        "unavailable_rows": unavailable,
        "reserved_coverage": covered / scored if scored else None,
        "predicted_over_observed": {
            "median": statistics.median(overshoot) if overshoot else None,
            "p90": _quantile(overshoot, 0.9) if overshoot else None,
            "max": max(overshoot) if overshoot else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        type=Path,
        default=ARTIFACT_DIR / "canonical_h800_observations.jsonl",
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "hardware.json",
    )
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    hardware = read_json(args.hardware)
    capacity = float(hardware["memory_bytes_reported_by_torch"])
    rows = collect_observations(args.observations)
    guard = fit_guard(rows, coverage=args.coverage)
    guard["capacity_bytes"] = capacity
    guard["evaluation"] = evaluate(rows, guard, capacity_bytes=capacity)
    guard["status"] = "diagnostic_candidate_requires_prospective_holdout"
    guard["publishable"] = False

    print(f"fit rows: {len(rows)}   capacity {capacity / GIB:.2f} GiB")
    print(f"guards: {len(guard['guards'])}   rejected: {len(guard['rejected_selectors'])}")
    for key, entry in sorted(guard["guards"].items()):
        print(
            f"  {key:34s} n={entry['rows']:4d} "
            f"med={entry['median']:.3f} P{int(args.coverage * 100)}="
            f"{entry['expansion_quantile']:.3f} max={entry['maximum']:.3f}"
        )
    for key, entry in sorted(guard["rejected_selectors"].items()):
        print(f"  REJECTED {key:30s} {entry['reason']}")
    evaluation = guard["evaluation"]
    print(
        f"\nLOSO reserved coverage: {evaluation['reserved_coverage']:.2%} "
        f"({evaluation['scored_rows']} rows, "
        f"{evaluation['unavailable_rows']} unavailable)"
    )
    print(
        f"predicted/observed: median "
        f"{evaluation['predicted_over_observed']['median']:.3f} "
        f"p90 {evaluation['predicted_over_observed']['p90']:.3f} "
        f"max {evaluation['predicted_over_observed']['max']:.3f}"
    )

    if args.output is not None:
        args.output.write_text(
            json.dumps(guard, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
