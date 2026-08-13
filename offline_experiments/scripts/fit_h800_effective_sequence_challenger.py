#!/usr/bin/env python3
"""Challenger: anchor memory on the achievable padded length, not on cutoff_len.

The frozen basis sets ``sequence = cutoff_len`` for every activation, attention
and logits term.  Runtime evidence says that budget is never spent: dynamic
padding pads each micro-batch to its own longest member, so a job whose longest
sample is far below the cutoff is charged for tokens it cannot materialize.  On
the 2026-08-03 fresh holdout this mis-rejected a genuinely safe 14B Full run --
predicted upper 143.21 GiB against an observed 127.40 GiB.

This challenger substitutes ``sequence = min(cutoff_len, profile_max)``.  That is
still a hard upper bound (no micro-batch can exceed the longest clipped sample,
whatever the MBS), so it can only ever tighten the anchor -- never loosen it.
Quantile rules such as p99 are deliberately NOT offered: discarding the tail
under-predicts the peak, which is a safety regression rather than a tightening.

Nothing frozen is mutated.  ``h800_theory_basis.memory_basis`` is called with a
substituted job dict, so the production implementation hash is untouched and the
frozen predictors keep validating.

Evaluation is a diagnostic.  Both replayed holdouts were consumed by earlier
campaigns, so a pass here is evidence for building a prospective design, not an
acceptance result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from common import ARTIFACT_DIR, read_json
from experiment_effective_sequence_memory_basis import (
    ProfileLengths,
    fit_reserved_ridge,
    fit_tail,
    leave_scenario_out,
    load_profile_lengths,
    predict_center,
    predict_upper,
)
from fit_h800_reservation_expansion_guard import (
    collect_observations,
    fit_guard,
    predict_reserved,
)
from h800_native_memory_calibration import (
    build_native_record,
    native_admission_reason,
)
from h800_theory_basis import memory_basis

GIB = float(1 << 30)
SEQUENCE_POLICY = "min(cutoff_len, max_clipped_profile_length)"

# Holdout campaigns replayed as diagnostics, with the profile directory each one
# was bound to.  Both were consumed when their own acceptance ran.
HOLDOUTS = (
    {
        "name": "2026-08-02_fresh_holdout_v2",
        "observations": "h800_fresh_holdout_observations_v2.json",
        "profiles": "fresh_holdout_v2/profiles",
        "frozen_predictions": None,
    },
    {
        "name": "2026-08-03_bounded_v2_fresh_holdout_v1",
        "observations": "h800_bounded_memory_v2_fresh_holdout_observations_v1.json",
        "profiles": "bounded_memory_v2_fresh_holdout_v1/profiles",
        # This campaign's observation rows omit cutoff_len and target_gbs; both
        # live in the frozen prediction bound to the same candidate id.
        "frozen_predictions": (
            "h800_frozen_predictions_before_bounded_memory_v2_fresh_holdout_v1.json"
        ),
    },
)


def _configuration_index(path: Path) -> dict[str, dict[str, Any]]:
    """Map candidate id -> frozen request configuration."""

    report = read_json(path)
    index: dict[str, dict[str, Any]] = {}
    for prediction in report.get("predictions") or []:
        request_id = str(prediction.get("request_id") or "")
        configuration = prediction.get("configuration")
        if request_id and isinstance(configuration, Mapping):
            index[request_id] = dict(configuration)
    return index


def _profile_max(profiles: ProfileLengths, dataset_id: str, cutoff_len: int) -> int:
    """Longest sample this dataset can present once clipped to ``cutoff_len``."""

    return profiles.sequence_for(
        dataset_id, cutoff_len=cutoff_len, mbs=1, rule="max"
    )


def reanchor(
    record: dict[str, Any],
    *,
    profiles: ProfileLengths,
    hardware: Mapping[str, Any],
    enabled: bool,
) -> dict[str, Any] | None:
    """Rebuild the analytic anchor under the challenger sequence policy."""

    if not enabled:
        return record
    scenario = record["scenario"]
    dataset_id = str(scenario.get("dataset_id") or "")
    if not profiles.has(dataset_id):
        return None
    cutoff = int(scenario["cutoff_len"])
    sequence = _profile_max(profiles, dataset_id, cutoff)
    substituted = {
        "gpu_count": scenario["gpu_count"],
        "mbs": int(scenario["physical_mbs"]),
        "cutoff_len": int(sequence),
        "zero": f"zero{int(record['selector'].get('zero_stage') or 0)}",
        "gc": bool(record["selector"].get("gradient_checkpointing")),
    }
    rebuilt = memory_basis(
        substituted,
        record["model_basis"],
        int(hardware["memory_bytes_reported_by_torch"]),
    )
    rebuilt["observed"] = record["memory"]["observed"]
    record = dict(record)
    record["memory"] = rebuilt
    record["effective_sequence"] = {
        "policy": SEQUENCE_POLICY,
        "sequence_tokens": int(sequence),
        "cutoff_len": cutoff,
        "fraction_of_cutoff": sequence / float(cutoff),
    }
    return record


def build_training_records(
    *,
    observations: Path,
    theory_basis: Path,
    inventory: Mapping[str, Any],
    hardware: Mapping[str, Any],
    profiles: ProfileLengths,
    enabled: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Native calibration rows plus the legacy theory-basis rows, re-anchored."""

    model_by_id = {str(entry["id"]): dict(entry) for entry in inventory.get("models") or []}
    fixed_lora = dict(inventory.get("fixed_lora") or {})
    records: list[dict[str, Any]] = []
    skipped = 0
    with observations.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if native_admission_reason(row) != "admitted":
                continue
            record = build_native_record(
                row,
                model_by_id=model_by_id,
                fixed_lora=fixed_lora,
                hardware=hardware,
            )
            rebuilt = reanchor(
                record, profiles=profiles, hardware=hardware, enabled=enabled
            )
            if rebuilt is None:
                skipped += 1
                continue
            records.append(rebuilt)
    for legacy in read_json(theory_basis).get("records") or []:
        if not isinstance(legacy, Mapping):
            continue
        if (legacy.get("route") or {}).get("memory_boundary") is not True:
            continue
        outcome = legacy.get("outcome")
        if isinstance(outcome, Mapping):
            outcome = outcome.get("class")
        if str(outcome or "").lower() not in {"success", "oom"}:
            continue
        rebuilt = reanchor(
            dict(legacy), profiles=profiles, hardware=hardware, enabled=enabled
        )
        if rebuilt is None:
            skipped += 1
            continue
        records.append(rebuilt)
    return records, skipped


def _holdout_record(
    row: Mapping[str, Any],
    *,
    inventory: Mapping[str, Any],
    hardware: Mapping[str, Any],
    profiles: ProfileLengths,
    enabled: bool,
    configurations: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Assemble a scoring record from a frozen holdout observation row."""

    from h800_theory_basis import _model_geometry

    merged = dict(row)
    if configurations:
        configuration = configurations.get(str(row.get("candidate_id") or ""))
        if configuration:
            for key, value in configuration.items():
                merged.setdefault(key, value)
    model_by_id = {str(entry["id"]): dict(entry) for entry in inventory.get("models") or []}
    model_id = str(merged.get("model_id") or "")
    if model_id not in model_by_id:
        return None
    dataset_id = str(merged.get("dataset_id") or merged.get("profile_id") or "")
    if not profiles.has(dataset_id):
        return None
    if merged.get("cutoff_len") is None:
        return None
    cutoff = int(merged["cutoff_len"])
    mbs = int(merged.get("mbs") or merged.get("physical_mbs"))
    train_type = str(merged.get("train_type") or merged.get("training_mode"))
    zero_stage = int(merged.get("zero_stage") or 0)
    gc = bool(
        merged["gc"] if "gc" in merged else merged.get("gradient_checkpointing")
    )
    geometry = _model_geometry(
        {
            "model_id": model_id,
            "train_type": train_type,
            "model_parameters": model_by_id[model_id]["actual_parameters"],
        },
        model_by_id[model_id],
        dict(inventory.get("fixed_lora") or {}),
    )
    sequence = (
        _profile_max(profiles, dataset_id, cutoff) if enabled else cutoff
    )
    memory = memory_basis(
        {
            "gpu_count": int(merged["gpu_count"]),
            "mbs": mbs,
            "cutoff_len": int(sequence),
            "zero": f"zero{zero_stage}",
            "gc": gc,
        },
        geometry,
        int(hardware["memory_bytes_reported_by_torch"]),
    )
    return {
        "observation_id": str(merged["job_id"]),
        "outcome": str(merged["outcome"]).lower(),
        "evidence_tier": "holdout",
        "scenario": {
            "model_id": model_id,
            "train_type": train_type,
            "dataset_id": dataset_id,
            "target_gbs": merged.get("target_gbs"),
            "gpu_count": int(merged["gpu_count"]),
            "physical_mbs": mbs,
            "cutoff_len": cutoff,
        },
        "selector": {
            "training_mode": train_type,
            "zero_stage": zero_stage,
            "gradient_checkpointing": gc,
            "packing": bool(row.get("packing")),
        },
        "model_basis": geometry,
        "memory": memory,
        "observed_reserved_bytes": merged.get("observed_reserved_bytes"),
        "sequence_tokens": int(sequence),
        "frozen_predicted_upper_bytes": merged.get("predicted_memory_upper_bytes")
        or merged.get("predicted_upper_bytes"),
        "frozen_predicted_center_bytes": merged.get("predicted_memory_center_bytes")
        or merged.get("predicted_center_bytes"),
        "frozen_predicted_admit": merged.get("predicted_admit"),
    }


def score_holdout(
    rows: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    tail: Mapping[str, Any],
    *,
    reservation_guard: Mapping[str, Any] | None = None,
    capacity_bytes: float | None = None,
    stratified_tail: bool = False,
) -> dict[str, Any]:
    """Score a holdout replay on safety first, then recall, then accuracy."""

    errors: list[float] = []
    covered = 0
    success_rows = 0
    safe_success = 0
    admitted_safe = 0
    false_safe_oom = 0
    unsafe_admitted = 0
    detail: list[dict[str, Any]] = []
    for row in rows:
        limit = float(row["memory"]["safe_limit_bytes"])
        center = predict_center(row, model)
        upper = predict_upper(row, model, tail, stratified=stratified_tail)
        guard_applied = None
        guard_floor = None
        if reservation_guard is not None and capacity_bytes is not None:
            # The ridge target is log(observed_reserved / anchor), so center and
            # upper are already reserved-semantics -- multiplying them by the
            # expansion ratio would double-count it.  What the tightened anchor
            # loses is the accidental slack that used to cover allocator
            # expansion, so the guard contributes an independent FLOOR instead:
            # anchor is allocated-semantics, and the allocator will ask for at
            # least anchor * expansion_quantile.
            selector = row["selector"]
            floor = predict_reserved(
                allocated_center_bytes=float(
                    row["memory"]["analytic_reference_bytes"]
                ),
                training_mode=str(selector["training_mode"]),
                zero_stage=int(selector.get("zero_stage") or 0),
                gradient_checkpointing=bool(selector.get("gradient_checkpointing")),
                gpu_count=int(row["scenario"]["gpu_count"]),
                capacity_bytes=capacity_bytes,
                guard=reservation_guard,
            )
            if not floor["available"]:
                # Fail closed: an unguarded selector is not admissible.
                guard_applied = False
                upper = float("inf")
            else:
                guard_applied = True
                guard_floor = float(floor["reserved_bytes"])
                upper = max(upper, guard_floor)
        admitted = upper <= limit
        observed = row.get("observed_reserved_bytes")
        entry: dict[str, Any] = {
            "observation_id": row["observation_id"],
            "dataset_id": row["scenario"]["dataset_id"],
            "model_id": row["scenario"]["model_id"],
            "training_mode": row["selector"]["training_mode"],
            "gpu_count": row["scenario"]["gpu_count"],
            "physical_mbs": row["scenario"]["physical_mbs"],
            "cutoff_len": row["scenario"]["cutoff_len"],
            "sequence_tokens": row["sequence_tokens"],
            "outcome": row["outcome"],
            "center_gib": center / GIB,
            "upper_gib": upper / GIB,
            "safe_limit_gib": limit / GIB,
            "admitted": admitted,
            "frozen_upper_gib": (
                float(row["frozen_predicted_upper_bytes"]) / GIB
                if row.get("frozen_predicted_upper_bytes")
                else None
            ),
            "frozen_admitted": row.get("frozen_predicted_admit"),
            "reservation_guard_applied": guard_applied,
            "reservation_guard_floor_gib": (
                guard_floor / GIB if guard_floor is not None else None
            ),
        }
        if row["outcome"] == "oom":
            if admitted:
                false_safe_oom += 1
        elif observed:
            observed = float(observed)
            success_rows += 1
            error = abs(center - observed) / observed
            errors.append(error)
            if upper >= observed:
                covered += 1
            entry["observed_gib"] = observed / GIB
            entry["absolute_percentage_error"] = error
            entry["upper_covers_observed"] = upper >= observed
            actually_safe = observed <= limit
            entry["actually_safe"] = actually_safe
            if actually_safe:
                safe_success += 1
                if admitted:
                    admitted_safe += 1
            elif admitted:
                unsafe_admitted += 1
        detail.append(entry)
    return {
        "rows": len(rows),
        "success_rows": success_rows,
        "false_safe_oom": false_safe_oom,
        "unsafe_success_admitted": unsafe_admitted,
        "success_upper_coverage": covered / success_rows if success_rows else None,
        "actual_safe_success_rows": safe_success,
        "admitted_safe_success_rows": admitted_safe,
        "admission_recall": admitted_safe / safe_success if safe_success else None,
        "center_absolute_percentage_error": {
            "count": len(errors),
            "mean": statistics.fmean(errors) if errors else None,
            "median": statistics.median(errors) if errors else None,
            "p90": (
                sorted(errors)[min(len(errors) - 1, math.ceil(0.9 * len(errors)) - 1)]
                if errors
                else None
            ),
            "max": max(errors) if errors else None,
        },
        "detail": detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        type=Path,
        default=ARTIFACT_DIR / "canonical_h800_observations.jsonl",
    )
    parser.add_argument(
        "--theory-basis", type=Path, default=ARTIFACT_DIR / "h800_theory_basis.json"
    )
    parser.add_argument(
        "--profiles", type=Path, default=ARTIFACT_DIR / "dataset_profiles"
    )
    parser.add_argument(
        "--inventory", type=Path, default=ARTIFACT_DIR / "model_inventory.json"
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "hardware.json",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--historical-weight", type=float, default=0.25)
    parser.add_argument("--feature-set", default="physical_shares")
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    hardware = read_json(args.hardware)
    inventory = read_json(args.inventory)
    training_profiles = load_profile_lengths(args.profiles)

    report: dict[str, Any] = {
        "schema": "sft_h800_effective_sequence_challenger_diagnostic/v1",
        "sequence_policy": SEQUENCE_POLICY,
        "evaluation_kind": "cpu_only_diagnostic_on_consumed_holdouts_not_acceptance",
        "frozen_files_mutated": False,
        "variants": {},
    }

    fitted: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    reservation_rows = collect_observations(args.observations)
    reservation_guard = fit_guard(reservation_rows, coverage=args.coverage)
    capacity = float(hardware["memory_bytes_reported_by_torch"])
    report["reservation_guard"] = {
        "fit_rows": reservation_guard["fit_rows"],
        "guards": len(reservation_guard["guards"]),
        "coverage": reservation_guard["coverage"],
        "selector_key_fields": reservation_guard["selector_key_fields"],
    }

    variants = (
        ("baseline_cutoff", False, False, False),
        ("challenger_profile_max", True, False, False),
        ("challenger_profile_max_plus_reservation_guard", True, True, False),
        ("challenger_full_stack_stratified_tail", True, True, True),
    )
    for label, enabled, use_guard, stratified in variants:
        records, skipped = build_training_records(
            observations=args.observations,
            theory_basis=args.theory_basis,
            inventory=inventory,
            hardware=hardware,
            profiles=training_profiles,
            enabled=enabled,
        )
        model = fit_reserved_ridge(
            records,
            feature_set=args.feature_set,
            alpha=args.alpha,
            historical_weight=args.historical_weight,
        )
        tail = fit_tail(records, model)
        fitted[label] = (model, tail)
        cross_validated = leave_scenario_out(
            records,
            feature_set=args.feature_set,
            alpha=args.alpha,
            historical_weight=args.historical_weight,
            stratified=stratified,
        )
        cross_validated.pop("rows", None)
        report["variants"][label] = {
            "training_rows": len(records),
            "skipped_missing_profile": skipped,
            "fit": {
                "intercept": model["intercept"],
                "fit_success_rows": model["fit_success_rows"],
                "fit_scenarios": model["fit_scenarios"],
            },
            "leave_native_scenario_out": cross_validated,
            "holdouts": {},
        }
        print(f"[{label}] training rows={len(records)} skipped={skipped}")
        error = cross_validated["center_absolute_percentage_error"]
        print(
            f"  LOSO   MAPE={error['mean']:.2%} P90={error['p90']:.2%} "
            f"coverage={cross_validated['success_upper_coverage']:.2%} "
            f"recall={cross_validated['admission_recall']:.2%} "
            f"false_safe_oom={cross_validated['false_safe_oom']}"
        )

        for holdout in HOLDOUTS:
            observations = read_json(ARTIFACT_DIR / holdout["observations"])
            profiles = load_profile_lengths(ARTIFACT_DIR / holdout["profiles"])
            configurations = (
                _configuration_index(ARTIFACT_DIR / holdout["frozen_predictions"])
                if holdout.get("frozen_predictions")
                else None
            )
            rows = []
            for row in observations.get("rows") or []:
                built = _holdout_record(
                    row,
                    inventory=inventory,
                    hardware=hardware,
                    profiles=profiles,
                    enabled=enabled,
                    configurations=configurations,
                )
                if built is not None:
                    rows.append(built)
            if not rows:
                continue
            scored = score_holdout(
                rows,
                model,
                tail,
                reservation_guard=reservation_guard if use_guard else None,
                capacity_bytes=capacity if use_guard else None,
                stratified_tail=stratified,
            )
            report["variants"][label]["holdouts"][holdout["name"]] = scored
            error = scored["center_absolute_percentage_error"]
            print(
                f"  {holdout['name'][:34]:34s} "
                f"MAPE={error['mean']:.2%} coverage={scored['success_upper_coverage']:.2%} "
                f"recall={scored['admission_recall']:.2%} "
                f"fsoom={scored['false_safe_oom']} "
                f"unsafe_admitted={scored['unsafe_success_admitted']}"
            )
        print()

    if args.output is not None:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
