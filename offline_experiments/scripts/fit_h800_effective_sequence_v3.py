#!/usr/bin/env python3
"""CPU-only H800 memory v3 diagnostic with an effective-sequence anchor.

This challenger separates four questions that were previously entangled:

1. What happens when the physical basis uses ``cutoff_len`` directly?
2. What happens when non-packing jobs use the largest achievable padded length?
3. How much error remains after calibrating *allocated* memory explicitly?
4. Can a direct-reserved upper plus an allocated-to-reserved expansion path guard
   admission without hiding unsupported high-variance mechanisms?

All tail scores are generated out of scenario, every outer validation fold
refits the center models and their inner calibration, and repeated rows inside a
scenario contribute only one (worst) calibration score.  Consumed holdouts are
replayed only as diagnostics.  The script never writes to ``artifacts/`` unless
the caller explicitly supplies such a path (which is not a supported workflow).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import statistics
from typing import Any

from common import ARTIFACT_DIR, read_json, sha256_file
from experiment_effective_sequence_memory_basis import (
    ProfileLengths,
    load_profile_lengths,
)
from fit_h800_effective_sequence_challenger import (
    HOLDOUTS,
    _configuration_index,
    _holdout_record,
    build_training_records,
)
from h800_challenger_modeling import (
    _fit_memory_ridge,
    _observed_allocated,
    _observed_reserved,
    _predict_memory_center,
    scenario_id,
)
from h800_theory_basis import memory_basis
from h800_theory_calibration import _oom_lower, _safe_limit


GIB = float(1 << 30)
SCHEMA = "sft_h800_effective_sequence_memory_v3_diagnostic/v1"
IMPLEMENTATION_VERSION = (
    "sft_h800_effective_sequence_memory_v3/"
    "2026-08-04.allocated-then-reserved-nested-scenario-v1"
)
DEFAULT_PAD_MULTIPLE = 8
DEFAULT_COVERAGE = 0.95

VARIANT_CUTOFF = "cutoff_physical_only"
VARIANT_EFFECTIVE = "effective_sequence_physical_only"
VARIANT_ALLOCATED = "effective_sequence_plus_allocated_calibration"
VARIANT_FULL = "effective_sequence_full_reserved_stack"
VARIANTS = (
    VARIANT_CUTOFF,
    VARIANT_EFFECTIVE,
    VARIANT_ALLOCATED,
    VARIANT_FULL,
)

# This bucket produced an unsafe admission on the consumed 2026-08-03 replay.
# It stays unavailable until its allocated-to-reserved expansion quantile is
# identifiable from independent scenario scores.
FAIL_CLOSED_EXPANSION_KEYS = {
    json.dumps(["lora", 2, False, 2, False], separators=(",", ":"))
}


def _outcome(record: Mapping[str, Any]) -> str:
    value = record.get("outcome")
    if isinstance(value, Mapping):
        value = value.get("class")
    return str(value or "").lower()


def round_up(value: int, multiple: int) -> int:
    """Round a positive integer up to a positive alignment multiple."""

    value = int(value)
    multiple = int(multiple)
    if value <= 0 or multiple <= 0:
        raise ValueError("value and multiple must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def effective_sequence_tokens(
    profiles: ProfileLengths,
    *,
    dataset_id: str,
    cutoff_len: int,
    packing: bool,
    pad_multiple: int = DEFAULT_PAD_MULTIPLE,
) -> dict[str, Any] | None:
    """Return the maximum model-facing sequence under the runtime contract.

    Non-packing SFT truncates individual samples at ``cutoff_len`` and the
    collator then pads the batch to a multiple of eight.  The order matters:
    round_up(min(cutoff, profile_max), 8) can exceed a non-aligned cutoff.

    Neat packing concatenates samples into packs near the configured cutoff, so
    a single-sample maximum is not an upper bound.  Packing therefore keeps the
    conservative cutoff assumption in this version.
    """

    cutoff = int(cutoff_len)
    if cutoff <= 0:
        raise ValueError("cutoff_len must be positive")
    if packing:
        return {
            "available": True,
            "tokens": cutoff,
            "raw_clipped_profile_max": None,
            "padding_multiple": None,
            "policy": "packing_uses_cutoff",
        }
    if not profiles.has(str(dataset_id)):
        return None
    raw = profiles.sequence_for(
        str(dataset_id), cutoff_len=cutoff, mbs=1, rule="max"
    )
    padded = round_up(min(cutoff, int(raw)), int(pad_multiple))
    return {
        "available": True,
        "tokens": padded,
        "raw_clipped_profile_max": int(raw),
        "padding_multiple": int(pad_multiple),
        "policy": "round_up(min(cutoff_len, profile_max), padding_multiple)",
    }


def reanchor_record(
    record: Mapping[str, Any],
    *,
    profiles: ProfileLengths,
    hardware: Mapping[str, Any],
    use_effective_sequence: bool,
    pad_multiple: int = DEFAULT_PAD_MULTIPLE,
) -> dict[str, Any] | None:
    """Rebuild a record's physical basis without mutating its evidence."""

    copied = dict(record)
    scenario = copied["scenario"]
    selector = copied["selector"]
    cutoff = int(scenario["cutoff_len"])
    packing = bool(selector.get("packing"))
    if use_effective_sequence:
        contract = effective_sequence_tokens(
            profiles,
            dataset_id=str(scenario.get("dataset_id") or ""),
            cutoff_len=cutoff,
            packing=packing,
            pad_multiple=pad_multiple,
        )
        if contract is None:
            return None
        sequence = int(contract["tokens"])
    else:
        sequence = cutoff
        contract = {
            "available": True,
            "tokens": cutoff,
            "raw_clipped_profile_max": None,
            "padding_multiple": None,
            "policy": "cutoff_len",
        }

    rebuilt = memory_basis(
        {
            "gpu_count": int(scenario["gpu_count"]),
            "mbs": int(scenario["physical_mbs"]),
            "cutoff_len": sequence,
            "zero": f"zero{int(selector.get('zero_stage') or 0)}",
            "gc": bool(selector.get("gradient_checkpointing")),
        },
        copied["model_basis"],
        int(hardware["memory_bytes_reported_by_torch"]),
    )
    old_memory = copied.get("memory") or {}
    if "observed" in old_memory:
        rebuilt["observed"] = old_memory["observed"]
    copied["memory"] = rebuilt
    copied["effective_sequence"] = {
        **contract,
        "cutoff_len": cutoff,
        "fraction_of_cutoff": sequence / float(cutoff),
        "packing": packing,
    }
    # Holdout records keep the model-facing token count at top level too.
    if "sequence_tokens" in copied:
        copied["sequence_tokens"] = sequence
    return copied


def mechanism_key(record: Mapping[str, Any]) -> str:
    selector = record.get("selector") or {}
    scenario = record.get("scenario") or {}
    return json.dumps(
        [
            str(selector.get("training_mode") or "unknown"),
            int(selector.get("zero_stage") or 0),
            bool(selector.get("gradient_checkpointing")),
            int(scenario.get("gpu_count") or 0),
            bool(selector.get("packing")),
        ],
        separators=(",", ":"),
    )


def mode_gc_key(record: Mapping[str, Any]) -> str:
    selector = record.get("selector") or {}
    return json.dumps(
        [
            str(selector.get("training_mode") or "unknown"),
            bool(selector.get("gradient_checkpointing")),
        ],
        separators=(",", ":"),
    )


def _label(record: Mapping[str, Any], label: str) -> float | None:
    if label == "allocated":
        return _observed_allocated(record)
    if label == "reserved":
        return _observed_reserved(record)
    raise ValueError(f"unknown memory label {label!r}")


def _fit_center(
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
    feature_set: str,
    alpha: float,
    historical_weight: float,
) -> dict[str, Any]:
    return _fit_memory_ridge(
        records,
        feature_set=feature_set,
        alpha=alpha,
        historical_weight=historical_weight,
        label=label,
    )


def _inner_oof_scores(
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
    feature_set: str,
    alpha: float,
    historical_weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate success and OOM scores without fitting on their scenario."""

    success_scores: list[dict[str, Any]] = []
    oom_scores: list[dict[str, Any]] = []
    scenarios = sorted({scenario_id(record) for record in records})
    for held in scenarios:
        training = [record for record in records if scenario_id(record) != held]
        evaluation = [record for record in records if scenario_id(record) == held]
        model = _fit_center(
            training,
            label=label,
            feature_set=feature_set,
            alpha=alpha,
            historical_weight=historical_weight,
        )
        for record in evaluation:
            center = _predict_memory_center(record, model)
            base = {
                "scenario_id": held,
                "mechanism_key": mechanism_key(record),
                "mode_gc_key": mode_gc_key(record),
            }
            if _outcome(record) == "success":
                observed = _label(record, label)
                if observed is not None:
                    success_scores.append(
                        {**base, "score": math.log(float(observed) / center)}
                    )
            elif label == "reserved" and _outcome(record) == "oom":
                safe = _safe_limit(record)
                lower = _oom_lower(record)
                floors = [
                    float(value)
                    for value in (lower, (safe + 1.0) if safe is not None else None)
                    if value is not None
                ]
                if floors:
                    oom_scores.append(
                        {**base, "score": math.log(max(floors) / center)}
                    )
    return success_scores, oom_scores


def conformal_upper(
    scenario_scores: Mapping[str, float], *, coverage: float
) -> dict[str, Any]:
    """Finite-sample one-sided conformal quantile over independent scenarios."""

    ordered = sorted(float(value) for value in scenario_scores.values())
    rank = math.ceil((len(ordered) + 1) * float(coverage))
    available = bool(ordered) and rank <= len(ordered)
    return {
        "available": available,
        "coverage": float(coverage),
        "independent_scenarios": len(ordered),
        "rank": rank,
        "log_upper": ordered[rank - 1] if available else None,
        "empirical_max": max(ordered) if ordered else None,
    }


def _collapse_scores(
    scores: Sequence[Mapping[str, Any]], *, key_field: str | None
) -> dict[str, dict[str, float]]:
    """Collapse repeated rows to the worst score in each scenario and bucket."""

    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in scores:
        bucket = str(row[key_field]) if key_field is not None else "pooled"
        scenario = str(row["scenario_id"])
        score = float(row["score"])
        grouped[bucket][scenario] = max(
            score, grouped[bucket].get(scenario, -math.inf)
        )
    return dict(grouped)


def fit_residual_hierarchy(
    scores: Sequence[Mapping[str, Any]], *, coverage: float
) -> dict[str, Any]:
    exact = _collapse_scores(scores, key_field="mechanism_key")
    mode_gc = _collapse_scores(scores, key_field="mode_gc_key")
    pooled = _collapse_scores(scores, key_field=None).get("pooled", {})
    return {
        "coverage": float(coverage),
        "score_unit": "worst_row_per_independent_scenario",
        "exact": {
            key: conformal_upper(values, coverage=coverage)
            for key, values in sorted(exact.items())
        },
        "mode_gc": {
            key: conformal_upper(values, coverage=coverage)
            for key, values in sorted(mode_gc.items())
        },
        "pooled": conformal_upper(pooled, coverage=coverage),
        "hierarchy": ["exact_mechanism", "mode_x_gc", "pooled"],
    }


def select_residual_upper(
    record: Mapping[str, Any], hierarchy: Mapping[str, Any]
) -> dict[str, Any]:
    exact = (hierarchy.get("exact") or {}).get(mechanism_key(record))
    if isinstance(exact, Mapping) and exact.get("available") is True:
        return {"available": True, "source": "exact_mechanism", **dict(exact)}
    coarse = (hierarchy.get("mode_gc") or {}).get(mode_gc_key(record))
    if isinstance(coarse, Mapping) and coarse.get("available") is True:
        return {"available": True, "source": "mode_x_gc", **dict(coarse)}
    pooled = hierarchy.get("pooled") or {}
    if isinstance(pooled, Mapping) and pooled.get("available") is True:
        return {"available": True, "source": "pooled", **dict(pooled)}
    return {"available": False, "source": None, "log_upper": None}


def fit_expansion_guard(
    records: Sequence[Mapping[str, Any]], *, coverage: float
) -> dict[str, Any]:
    """Calibrate reserved/allocated using independent scenario maxima."""

    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    row_counts: dict[str, int] = defaultdict(int)
    for record in records:
        if _outcome(record) != "success":
            continue
        allocated = _observed_allocated(record)
        reserved = _observed_reserved(record)
        if allocated is None or reserved is None:
            continue
        key = mechanism_key(record)
        scenario = scenario_id(record)
        log_ratio = math.log(float(reserved) / float(allocated))
        grouped[key][scenario] = max(
            log_ratio, grouped[key].get(scenario, -math.inf)
        )
        row_counts[key] += 1
    entries = {}
    for key, values in sorted(grouped.items()):
        entry = conformal_upper(values, coverage=coverage)
        entry["rows"] = row_counts[key]
        entry["expansion_upper"] = (
            math.exp(float(entry["log_upper"]))
            if entry["available"]
            else None
        )
        entry["empirical_max_expansion"] = math.exp(
            float(entry["empirical_max"])
        )
        entries[key] = entry
    return {
        "coverage": float(coverage),
        "score_unit": "worst_ratio_per_independent_scenario",
        "entries": entries,
        "no_pooled_fallback": True,
        "fail_closed_keys": sorted(FAIL_CLOSED_EXPANSION_KEYS),
    }


def fit_oom_guards(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = _collapse_scores(scores, key_field="mechanism_key")
    return {
        key: {
            "independent_scenarios": len(values),
            "log_lower": max(values.values()),
        }
        for key, values in sorted(grouped.items())
        if values
    }


def fit_bundle(
    records: Sequence[Mapping[str, Any]],
    *,
    feature_set: str,
    alpha: float,
    historical_weight: float,
    coverage: float,
) -> dict[str, Any]:
    """Fit allocated and direct-reserved centers plus OOF safety calibration."""

    allocated_model = _fit_center(
        records,
        label="allocated",
        feature_set=feature_set,
        alpha=alpha,
        historical_weight=historical_weight,
    )
    allocated_scores, _ = _inner_oof_scores(
        records,
        label="allocated",
        feature_set=feature_set,
        alpha=alpha,
        historical_weight=historical_weight,
    )
    reserved_model = _fit_center(
        records,
        label="reserved",
        feature_set=feature_set,
        alpha=alpha,
        historical_weight=historical_weight,
    )
    reserved_scores, oom_scores = _inner_oof_scores(
        records,
        label="reserved",
        feature_set=feature_set,
        alpha=alpha,
        historical_weight=historical_weight,
    )
    return {
        "feature_set": feature_set,
        "alpha": float(alpha),
        "historical_weight": float(historical_weight),
        "coverage": float(coverage),
        "allocated_model": allocated_model,
        "allocated_residual_upper": fit_residual_hierarchy(
            allocated_scores, coverage=coverage
        ),
        "reserved_model": reserved_model,
        "reserved_residual_upper": fit_residual_hierarchy(
            reserved_scores, coverage=coverage
        ),
        "reservation_expansion": fit_expansion_guard(
            records, coverage=coverage
        ),
        "oom_guards": fit_oom_guards(oom_scores),
        "inner_oof": {
            "allocated_success_scores": len(allocated_scores),
            "reserved_success_scores": len(reserved_scores),
            "reserved_oom_scores": len(oom_scores),
        },
    }


def _calibrated_upper(
    record: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    residuals: Mapping[str, Any],
) -> dict[str, Any]:
    center = _predict_memory_center(record, model)
    tail = select_residual_upper(record, residuals)
    if tail.get("available") is not True:
        return {
            "available": False,
            "center_bytes": center,
            "upper_bytes": None,
            "tail_source": None,
        }
    log_upper = max(0.0, float(tail["log_upper"]))
    return {
        "available": True,
        "center_bytes": center,
        "upper_bytes": center * math.exp(log_upper),
        "tail_source": tail["source"],
        "tail_independent_scenarios": tail.get("independent_scenarios"),
    }


def predict_variant(
    record: Mapping[str, Any],
    *,
    variant: str,
    bundle: Mapping[str, Any] | None,
) -> dict[str, Any]:
    anchor = float(record["memory"]["analytic_reference_bytes"])
    if variant in {VARIANT_CUTOFF, VARIANT_EFFECTIVE}:
        return {
            "available": True,
            "center_semantics": "analytic_allocated_anchor",
            "center_bytes": anchor,
            "upper_bytes": anchor,
            "components": {"analytic_anchor_bytes": anchor},
            "issues": [],
        }
    if bundle is None:
        raise ValueError("calibrated variants require a fitted bundle")

    allocated = _calibrated_upper(
        record,
        model=bundle["allocated_model"],
        residuals=bundle["allocated_residual_upper"],
    )
    if allocated["available"] is not True:
        return {
            "available": False,
            "center_semantics": "allocated",
            "center_bytes": allocated["center_bytes"],
            "upper_bytes": None,
            "components": {"allocated": allocated},
            "issues": ["allocated_residual_upper_unavailable"],
        }
    if variant == VARIANT_ALLOCATED:
        return {
            "available": True,
            "center_semantics": "allocated",
            "center_bytes": allocated["center_bytes"],
            # This is intentionally evaluated against reserved to expose the
            # size of the allocator gap before the fourth layer is added.
            "upper_bytes": allocated["upper_bytes"],
            "components": {"allocated": allocated},
            "issues": [],
        }
    if variant != VARIANT_FULL:
        raise ValueError(f"unknown variant {variant!r}")

    reserved = _calibrated_upper(
        record,
        model=bundle["reserved_model"],
        residuals=bundle["reserved_residual_upper"],
    )
    if reserved["available"] is not True:
        return {
            "available": False,
            "center_semantics": "reserved",
            "center_bytes": reserved["center_bytes"],
            "upper_bytes": None,
            "components": {"allocated": allocated, "direct_reserved": reserved},
            "issues": ["direct_reserved_residual_upper_unavailable"],
        }

    key = mechanism_key(record)
    expansion_entry = (
        (bundle.get("reservation_expansion") or {}).get("entries") or {}
    ).get(key)
    expansion_upper = None
    expansion_available = bool(
        isinstance(expansion_entry, Mapping)
        and expansion_entry.get("available") is True
    )
    if expansion_available:
        expansion_upper = float(allocated["upper_bytes"]) * float(
            expansion_entry["expansion_upper"]
        )
    if key in FAIL_CLOSED_EXPANSION_KEYS and not expansion_available:
        return {
            "available": False,
            "center_semantics": "reserved",
            "center_bytes": reserved["center_bytes"],
            "upper_bytes": None,
            "components": {
                "allocated": allocated,
                "direct_reserved": reserved,
                "expansion": dict(expansion_entry or {}),
            },
            "issues": ["critical_expansion_guard_unavailable_fail_closed"],
        }

    oom_entry = (bundle.get("oom_guards") or {}).get(key)
    oom_upper = None
    if isinstance(oom_entry, Mapping):
        oom_upper = float(reserved["center_bytes"]) * math.exp(
            max(0.0, float(oom_entry["log_lower"]))
        )
    candidates = [float(reserved["upper_bytes"])]
    if expansion_upper is not None:
        candidates.append(expansion_upper)
    if oom_upper is not None:
        candidates.append(oom_upper)
    upper = max(candidates)
    issues = []
    if not expansion_available:
        issues.append("expansion_guard_unavailable_direct_reserved_path_only")
    return {
        "available": True,
        "center_semantics": "reserved",
        "center_bytes": reserved["center_bytes"],
        "upper_bytes": upper,
        "components": {
            "analytic_anchor_bytes": anchor,
            "allocated": allocated,
            "direct_reserved": reserved,
            "expansion_upper_bytes": expansion_upper,
            "expansion_entry": dict(expansion_entry or {}),
            "oom_upper_bytes": oom_upper,
        },
        "issues": issues,
    }


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return ordered[max(0, rank)]


def evaluate_predictions(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
) -> dict[str, Any]:
    allocated_errors: list[float] = []
    reserved_errors: list[float] = []
    success_rows = success_scored = success_covered = 0
    safe_success = admitted_safe = unsafe_admitted = 0
    oom_rows = false_safe_oom = unavailable = 0
    detail: list[dict[str, Any]] = []
    for record, prediction in rows:
        available = prediction.get("available") is True
        upper = (
            float(prediction["upper_bytes"])
            if available and prediction.get("upper_bytes") is not None
            else None
        )
        limit = float(record["memory"]["safe_limit_bytes"])
        admitted = bool(available and upper is not None and upper <= limit)
        entry: dict[str, Any] = {
            "observation_id": record.get("observation_id"),
            "scenario_id": scenario_id(record),
            "model_id": (record.get("scenario") or {}).get("model_id"),
            "dataset_id": (record.get("scenario") or {}).get("dataset_id"),
            "training_mode": (record.get("selector") or {}).get("training_mode"),
            "gpu_count": (record.get("scenario") or {}).get("gpu_count"),
            "physical_mbs": (record.get("scenario") or {}).get("physical_mbs"),
            "cutoff_len": (record.get("scenario") or {}).get("cutoff_len"),
            "sequence_tokens": (
                (record.get("effective_sequence") or {}).get("tokens")
                or record.get("sequence_tokens")
            ),
            "outcome": _outcome(record),
            "prediction_available": available,
            "admitted": admitted,
            "center_semantics": prediction.get("center_semantics"),
            "center_gib": (
                float(prediction["center_bytes"]) / GIB
                if prediction.get("center_bytes") is not None
                else None
            ),
            "upper_gib": upper / GIB if upper is not None else None,
            "safe_limit_gib": limit / GIB,
            "issues": list(prediction.get("issues") or []),
        }
        if not available:
            unavailable += 1
        if _outcome(record) == "success":
            observed_allocated = _observed_allocated(record)
            observed_reserved = _observed_reserved(record)
            # Frozen holdout records carry reserved at top level.
            if observed_reserved is None and record.get("observed_reserved_bytes"):
                observed_reserved = float(record["observed_reserved_bytes"])
            if observed_allocated is not None and prediction.get("center_bytes"):
                if prediction.get("center_semantics") in {
                    "analytic_allocated_anchor",
                    "allocated",
                }:
                    allocated_errors.append(
                        abs(float(prediction["center_bytes"]) - observed_allocated)
                        / observed_allocated
                    )
                    entry["observed_allocated_gib"] = observed_allocated / GIB
            if observed_reserved is None:
                detail.append(entry)
                continue
            success_rows += 1
            entry["observed_reserved_gib"] = observed_reserved / GIB
            if prediction.get("center_semantics") == "reserved" and prediction.get(
                "center_bytes"
            ):
                reserved_errors.append(
                    abs(float(prediction["center_bytes"]) - observed_reserved)
                    / observed_reserved
                )
            if available and upper is not None:
                success_scored += 1
                covered = upper >= observed_reserved
                success_covered += int(covered)
                entry["upper_covers_reserved"] = covered
            actually_safe = observed_reserved <= limit
            entry["actually_safe"] = actually_safe
            if actually_safe:
                safe_success += 1
                admitted_safe += int(admitted)
            elif admitted:
                unsafe_admitted += 1
        elif _outcome(record) == "oom":
            oom_rows += 1
            false_safe_oom += int(admitted)
        detail.append(entry)
    return {
        "rows": len(rows),
        "prediction_unavailable_rows": unavailable,
        "success_rows": success_rows,
        "success_scored_rows": success_scored,
        "reserved_upper_coverage_scored": (
            success_covered / success_scored if success_scored else None
        ),
        "reserved_upper_coverage_with_fail_closed": (
            (success_covered + success_rows - success_scored) / success_rows
            if success_rows
            else None
        ),
        "actual_safe_success_rows": safe_success,
        "admitted_safe_success_rows": admitted_safe,
        "admission_recall": admitted_safe / safe_success if safe_success else None,
        "unsafe_success_admitted": unsafe_admitted,
        "oom_rows": oom_rows,
        "false_safe_oom": false_safe_oom,
        "allocated_center_mape": (
            statistics.fmean(allocated_errors) if allocated_errors else None
        ),
        "allocated_center_p90_ape": _percentile(allocated_errors, 0.90),
        "reserved_center_mape": (
            statistics.fmean(reserved_errors) if reserved_errors else None
        ),
        "reserved_center_p90_ape": _percentile(reserved_errors, 0.90),
        "detail": detail,
    }


def _native(record: Mapping[str, Any]) -> bool:
    return not str(record.get("evidence_tier") or "").startswith("legacy")


def nested_leave_scenario_out(
    cutoff_records: Sequence[Mapping[str, Any]],
    effective_records: Sequence[Mapping[str, Any]],
    *,
    feature_set: str,
    alpha: float,
    historical_weight: float,
    coverage: float,
) -> dict[str, Any]:
    cutoff_by_id = {str(row["observation_id"]): row for row in cutoff_records}
    native_effective = [row for row in effective_records if _native(row)]
    held_scenarios = sorted({scenario_id(row) for row in native_effective})
    predictions: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {
        variant: [] for variant in VARIANTS
    }
    folds: list[dict[str, Any]] = []
    for index, held in enumerate(held_scenarios, start=1):
        print(
            f"nested_loso: fitting fold {index}/{len(held_scenarios)} "
            f"holdout={held}",
            flush=True,
        )
        training = [row for row in effective_records if scenario_id(row) != held]
        evaluation = [row for row in native_effective if scenario_id(row) == held]
        bundle = fit_bundle(
            training,
            feature_set=feature_set,
            alpha=alpha,
            historical_weight=historical_weight,
            coverage=coverage,
        )
        for record in evaluation:
            observation_id = str(record["observation_id"])
            cutoff = cutoff_by_id[observation_id]
            predictions[VARIANT_CUTOFF].append(
                (
                    cutoff,
                    predict_variant(cutoff, variant=VARIANT_CUTOFF, bundle=None),
                )
            )
            for variant in VARIANTS[1:]:
                predictions[variant].append(
                    (
                        record,
                        predict_variant(record, variant=variant, bundle=bundle),
                    )
                )
        folds.append(
            {
                "index": index,
                "held_scenario_id": held,
                "training_rows": len(training),
                "evaluation_rows": len(evaluation),
            }
        )
    return {
        "protocol": (
            "outer_leave_native_scenario_out; every outer fold refits allocated "
            "and reserved centers; tails use inner leave-scenario-out scores; "
            "calibration scores collapse to scenario maxima"
        ),
        "native_scenarios": len(held_scenarios),
        "folds": folds,
        "variants": {
            variant: evaluate_predictions(rows)
            for variant, rows in predictions.items()
        },
    }


def _profile_manifest(profile_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(profile_dir.glob("*.jsonl")):
        files.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "profile_dir": str(profile_dir.resolve()),
        "files": files,
        "profile_binding_required_at_inference": True,
    }


def build_holdout_records(
    *,
    holdout: Mapping[str, Any],
    inventory: Mapping[str, Any],
    hardware: Mapping[str, Any],
    use_effective_sequence: bool,
    pad_multiple: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations_path = ARTIFACT_DIR / str(holdout["observations"])
    profile_dir = ARTIFACT_DIR / str(holdout["profiles"])
    observations = read_json(observations_path)
    profiles = load_profile_lengths(profile_dir)
    configurations = (
        _configuration_index(ARTIFACT_DIR / str(holdout["frozen_predictions"]))
        if holdout.get("frozen_predictions")
        else None
    )
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for source in observations.get("rows") or []:
        built = _holdout_record(
            source,
            inventory=inventory,
            hardware=hardware,
            profiles=profiles,
            enabled=False,
            configurations=configurations,
        )
        if built is None:
            skipped.append(str(source.get("job_id") or source.get("candidate_id")))
            continue
        built["candidate_role"] = source.get("candidate_role")
        rebuilt = reanchor_record(
            built,
            profiles=profiles,
            hardware=hardware,
            use_effective_sequence=use_effective_sequence,
            pad_multiple=pad_multiple,
        )
        if rebuilt is None:
            skipped.append(str(source.get("job_id") or source.get("candidate_id")))
            continue
        rows.append(rebuilt)
    return rows, {
        "source_rows": len(observations.get("rows") or []),
        "built_rows": len(rows),
        "skipped_rows": len(skipped),
        "skipped_ids": skipped,
        "profiles": _profile_manifest(profile_dir),
    }


def _bundle_summary(bundle: Mapping[str, Any]) -> dict[str, Any]:
    expansion = bundle.get("reservation_expansion") or {}
    entries = expansion.get("entries") or {}
    return {
        "feature_set": bundle["feature_set"],
        "alpha": bundle["alpha"],
        "historical_weight": bundle["historical_weight"],
        "coverage": bundle["coverage"],
        "inner_oof": bundle["inner_oof"],
        "allocated_fit_success_rows": bundle["allocated_model"]["fit_success_rows"],
        "reserved_fit_success_rows": bundle["reserved_model"]["fit_success_rows"],
        "expansion_entries": len(entries),
        "expansion_identifiable_entries": sum(
            entry.get("available") is True for entry in entries.values()
        ),
        "critical_expansion_entries": {
            key: entries.get(key) for key in sorted(FAIL_CLOSED_EXPANSION_KEYS)
        },
        "oom_guard_entries": len(bundle.get("oom_guards") or {}),
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
        "--inventory",
        type=Path,
        default=ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_v1.json",
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "hardware.json",
    )
    parser.add_argument("--feature-set", default="physical_shares")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--historical-weight", type=float, default=0.25)
    parser.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    parser.add_argument("--pad-multiple", type=int, default=DEFAULT_PAD_MULTIPLE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not 0.5 < args.coverage < 1.0:
        raise ValueError("coverage must lie strictly between 0.5 and 1")
    if args.output is not None and ARTIFACT_DIR.resolve() in args.output.resolve().parents:
        raise ValueError("v3 diagnostics must not be written under artifacts/")

    hardware = read_json(args.hardware)
    inventory = read_json(args.inventory)
    profiles = load_profile_lengths(args.profiles)
    base_records, base_skipped = build_training_records(
        observations=args.observations,
        theory_basis=args.theory_basis,
        inventory=inventory,
        hardware=hardware,
        profiles=profiles,
        enabled=False,
    )
    cutoff_records: list[dict[str, Any]] = []
    effective_records: list[dict[str, Any]] = []
    effective_skipped = 0
    for record in base_records:
        cutoff = reanchor_record(
            record,
            profiles=profiles,
            hardware=hardware,
            use_effective_sequence=False,
            pad_multiple=args.pad_multiple,
        )
        effective = reanchor_record(
            record,
            profiles=profiles,
            hardware=hardware,
            use_effective_sequence=True,
            pad_multiple=args.pad_multiple,
        )
        assert cutoff is not None
        cutoff_records.append(cutoff)
        if effective is None:
            effective_skipped += 1
        else:
            effective_records.append(effective)

    if len(cutoff_records) != len(effective_records):
        raise ValueError(
            "effective-sequence training profiles are incomplete; fail closed"
        )

    print(
        f"training records={len(effective_records)} "
        f"base_skipped={base_skipped} effective_skipped={effective_skipped}"
    )
    nested = nested_leave_scenario_out(
        cutoff_records,
        effective_records,
        feature_set=args.feature_set,
        alpha=args.alpha,
        historical_weight=args.historical_weight,
        coverage=args.coverage,
    )
    for variant, result in nested["variants"].items():
        print(
            f"{variant:48s} "
            f"alloc_mape={result['allocated_center_mape']} "
            f"reserved_cov={result['reserved_upper_coverage_scored']} "
            f"recall={result['admission_recall']} "
            f"unsafe={result['unsafe_success_admitted']} "
            f"false_oom={result['false_safe_oom']} "
            f"unavailable={result['prediction_unavailable_rows']}"
        )

    full_bundle = fit_bundle(
        effective_records,
        feature_set=args.feature_set,
        alpha=args.alpha,
        historical_weight=args.historical_weight,
        coverage=args.coverage,
    )
    holdout_reports: dict[str, Any] = {}
    for holdout in HOLDOUTS:
        cutoff_holdout, cutoff_meta = build_holdout_records(
            holdout=holdout,
            inventory=inventory,
            hardware=hardware,
            use_effective_sequence=False,
            pad_multiple=args.pad_multiple,
        )
        effective_holdout, effective_meta = build_holdout_records(
            holdout=holdout,
            inventory=inventory,
            hardware=hardware,
            use_effective_sequence=True,
            pad_multiple=args.pad_multiple,
        )
        if len(cutoff_holdout) != len(effective_holdout):
            raise ValueError(f"holdout {holdout['name']} profile coverage drifted")
        by_id = {str(row["observation_id"]): row for row in cutoff_holdout}
        variants: dict[str, Any] = {}
        for variant in VARIANTS:
            scored = []
            for effective in effective_holdout:
                record = (
                    by_id[str(effective["observation_id"])]
                    if variant == VARIANT_CUTOFF
                    else effective
                )
                scored.append(
                    (
                        record,
                        predict_variant(
                            record,
                            variant=variant,
                            bundle=full_bundle if variant not in {VARIANT_CUTOFF, VARIANT_EFFECTIVE} else None,
                        ),
                    )
                )
            variants[variant] = evaluate_predictions(scored)
        holdout_reports[str(holdout["name"])] = {
            "evaluation_kind": "diagnostic_replay_of_consumed_holdout",
            "cutoff_build": cutoff_meta,
            "effective_build": effective_meta,
            "variants": variants,
        }
        print(f"holdout {holdout['name']} rows={len(effective_holdout)}")
        for variant, result in variants.items():
            print(
                f"  {variant:46s} cov={result['reserved_upper_coverage_scored']} "
                f"recall={result['admission_recall']} "
                f"unsafe={result['unsafe_success_admitted']} "
                f"false_oom={result['false_safe_oom']} "
                f"unavailable={result['prediction_unavailable_rows']}"
            )

    fractions = [
        float(row["effective_sequence"]["fraction_of_cutoff"])
        for row in effective_records
    ]
    report = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "diagnostic_candidate_requires_new_calibration_and_prospective_holdout",
        "publishable": False,
        "evaluation_kind": "cpu_only_nested_cv_plus_consumed_holdout_diagnostics",
        "contracts": {
            "effective_sequence": (
                "packing ? cutoff_len : "
                "round_up(min(cutoff_len, profile_max), pad_multiple)"
            ),
            "pad_multiple": args.pad_multiple,
            "missing_profile_policy": "fail_closed",
            "profile_binding": (
                "dataset, tokenizer, template, preprocessing implementation and "
                "profile SHA-256 must match at inference"
            ),
            "expansion": (
                "allocated_upper * conformal_upper(reserved/allocated); no "
                "statistical-independence claim"
            ),
            "critical_risk_policy": "fail_closed_when_expansion_unidentifiable",
        },
        "inputs": {
            "observations": {
                "path": str(args.observations.resolve()),
                "sha256": sha256_file(args.observations),
            },
            "theory_basis": {
                "path": str(args.theory_basis.resolve()),
                "sha256": sha256_file(args.theory_basis),
            },
            "inventory": {
                "path": str(args.inventory.resolve()),
                "sha256": sha256_file(args.inventory),
            },
            "hardware": {
                "path": str(args.hardware.resolve()),
                "sha256": sha256_file(args.hardware),
            },
            "profiles": _profile_manifest(args.profiles),
        },
        "training": {
            "records": len(effective_records),
            "base_skipped": base_skipped,
            "effective_skipped": effective_skipped,
            "sequence_fraction_of_cutoff": {
                "min": min(fractions),
                "median": statistics.median(fractions),
                "max": max(fractions),
            },
        },
        "nested_leave_scenario_out": nested,
        "full_fit_summary": _bundle_summary(full_bundle),
        "consumed_holdout_diagnostics": holdout_reports,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
