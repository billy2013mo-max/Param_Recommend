#!/usr/bin/env python3
"""Step 1: joint card-aware memory-centre refit (H800 + RTX 4090).

Why this exists
---------------
H800 and RTX 4090 already share one memory contract in code: both call
``h800_challenger_modeling._memory_features(record, "physical_shares")`` and
both fit ``analytic_reference_log_residual_ridge``.  What they do NOT share is
a fitted model - each card has its own frozen coefficient vector, so the Go
side would have to carry two.  This script fits ONE coefficient vector over
both cards by appending explicit card features, so the platform reads a single
contract.

What it does
------------
1. Builds memory records for both cards (shared 28-dim physical_shares design).
2. Fits candidate joint models that differ only in which card features they add.
3. Selects among candidates by CROSS-CARD generalisation, not in-sample fit.
4. Reports, for the selected model and for the two per-card baselines:
     * per-card leave-scenario-out CV (does joint hurt either card?)
     * cross-card holdout (train one card, predict the other)
     * leave-one-model-out across both cards
     * operational upper bound: OOM recall and false-reject rate
5. Writes an analysis artifact.  Sets publishable=False - this is a fit +
   evaluation report, promoting it to production is a separate decision.

Honest caveats are recorded in the artifact's ``limitations``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import ROOT, read_json, sha256_file, sha256_json, write_json

SCHEMA = "sft_joint_card_memory_centre/v1"
GIB = 1024.0 ** 3

BASE_FEATURE_SET = "physical_shares"

# Candidate card-feature sets.  Each entry maps a name to the list of card
# feature names it adds on top of the shared 28 physical_shares features.
CARD_FEATURE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "card_intercept": ("is_rtx4090",),
    "card_intercept_plus_gpu": (
        "is_rtx4090",
        "rtx4090_x_log2_gpu_count",
    ),
    "card_intercept_plus_gpu_zero": (
        "is_rtx4090",
        "rtx4090_x_log2_gpu_count",
        "rtx4090_x_zero2",
        "rtx4090_x_zero3",
    ),
    "card_full": (
        "is_rtx4090",
        "rtx4090_x_log2_gpu_count",
        "rtx4090_x_zero2",
        "rtx4090_x_zero3",
        "rtx4090_x_log2_parameters",
        "rtx4090_x_gradient_checkpointing",
    ),
}

ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


# --------------------------------------------------------------- record build
def _import_shared():
    import h800_challenger_modeling as H

    return H


def build_h800_records() -> tuple[list[dict[str, Any]], dict[str, int]]:
    H = _import_shared()
    from h800_native_memory_calibration import (
        build_native_record,
        native_admission_reason,
    )

    observations = H._read_observations(
        ROOT / "artifacts" / "canonical_h800_observations.jsonl"
    )
    inventory = read_json(ROOT / "artifacts" / "model_inventory.json")
    hardware = read_json(ROOT / "config" / "hardware.json")
    model_by_id, fixed_lora = H._inventory_models(inventory)

    records: list[dict[str, Any]] = []
    reasons: Counter = Counter()
    for row in observations:
        reason = native_admission_reason(row)
        reasons[reason] += 1
        if reason != "admitted":
            continue
        record = build_native_record(
            row,
            model_by_id=model_by_id,
            fixed_lora=fixed_lora,
            hardware=hardware,
        )
        record["_card"] = "h800"
        records.append(record)
    return records, dict(reasons)


def build_rtx4090_records() -> list[dict[str, Any]]:
    """Build 4090 memory records.

    The 4090 record builder needs ``ThroughputPredictor._normalized_request``
    for hardware + model geometry + zero-stage resolution.  That constructor
    hard-fails on an implementation-SHA gate which NO commit in this repo can
    satisfy: ``artifacts/structured_throughput_modeling.json`` binds
    ``5bb89087..`` while the file has only ever been ``cfaf6a54..`` (b1bd5c7)
    or ``11963c59..`` (381ebaf, a try/except around the scipy import - the
    only delta).  The gate guards the frozen *throughput* head, which memory
    work never reads, so we bypass it here and record that we did.
    """
    import throughput_predictor as TP
    import rtx4090_physical_v4b_modeling as R

    original = TP.ThroughputPredictor._validate_bindings

    def _skip(self) -> None:  # noqa: ANN001
        self.binding_mismatches = [
            "implementation-sha gate bypassed: memory-only fit, "
            "frozen throughput head never read"
        ]

    TP.ThroughputPredictor._validate_bindings = _skip
    try:
        predictor = TP.ThroughputPredictor(strict_bindings=False)
        campaign_root = ROOT / "campaigns" / "rtx4090_20260717"
        rows = R._read_rows(campaign_root)
        records = R._build_records(
            predictor=predictor,
            campaign_root=campaign_root,
            rows=rows,
            require_throughput=False,
        )
    finally:
        TP.ThroughputPredictor._validate_bindings = original
    for record in records:
        record["_card"] = "rtx4090"
    return records


# ------------------------------------------------------------------ features
def card_features(record: Mapping[str, Any], names: Sequence[str]) -> list[float]:
    selector = record.get("selector") or {}
    scenario = record.get("scenario") or {}
    basis = record.get("model_basis") or {}
    is_4090 = 1.0 if record.get("_card") == "rtx4090" else 0.0
    zero = int(selector.get("zero_stage") or 0)
    log_gpu = math.log2(float(scenario["gpu_count"]))
    log_parameters = math.log2(
        float(basis["base_parameters"]) / 2_031_739_904.0
    )
    gc = float(bool(selector.get("gradient_checkpointing")))
    table = {
        "is_rtx4090": is_4090,
        "rtx4090_x_log2_gpu_count": is_4090 * log_gpu,
        "rtx4090_x_zero2": is_4090 * float(zero == 2),
        "rtx4090_x_zero3": is_4090 * float(zero == 3),
        "rtx4090_x_log2_parameters": is_4090 * log_parameters,
        "rtx4090_x_gradient_checkpointing": is_4090 * gc,
    }
    missing = [name for name in names if name not in table]
    if missing:
        raise KeyError(f"Unknown card feature(s): {missing}")
    return [table[name] for name in names]


def joint_features(
    record: Mapping[str, Any], names: Sequence[str]
) -> np.ndarray:
    H = _import_shared()
    base = _base_feature_cache(record, H)
    return np.concatenate((base, np.asarray(card_features(record, names), float)))


_BASE_CACHE: dict[int, np.ndarray] = {}


def _base_feature_cache(record: Mapping[str, Any], H) -> np.ndarray:  # noqa: ANN001
    """Cache the shared 28-dim vector; it is identical across candidates."""
    key = id(record)
    cached = _BASE_CACHE.get(key)
    if cached is None:
        cached = H._memory_features(record, BASE_FEATURE_SET)
        _BASE_CACHE[key] = cached
    return cached


def joint_feature_names(names: Sequence[str]) -> list[str]:
    H = _import_shared()
    return [*H._memory_feature_names(BASE_FEATURE_SET), *names]


def namespaced_scenario_id(record: Mapping[str, Any]) -> str:
    H = _import_shared()
    return f"{record['_card']}:{H.scenario_id(record)}"


def observed_reserved(record: Mapping[str, Any]) -> float | None:
    H = _import_shared()
    return H._memory_label(record, "reserved")


def outcome(record: Mapping[str, Any]) -> str:
    H = _import_shared()
    return H._outcome(record)


# ------------------------------------------------------------------- fitting
def fit_joint(
    records: Sequence[Mapping[str, Any]],
    *,
    card_names: Sequence[str],
    alpha: float,
    card_equal_weight: bool,
) -> dict[str, Any]:
    successes = [
        record
        for record in records
        if outcome(record) == "success" and observed_reserved(record) is not None
    ]
    if not successes:
        raise ValueError("No success rows")
    features = np.vstack([joint_features(r, card_names) for r in successes])
    targets = np.asarray(
        [
            math.log(
                float(observed_reserved(r))
                / float(r["memory"]["analytic_reference_bytes"])
            )
            for r in successes
        ],
        dtype=float,
    )
    scenario_counts = Counter(namespaced_scenario_id(r) for r in successes)
    raw = np.asarray(
        [1.0 / scenario_counts[namespaced_scenario_id(r)] for r in successes],
        dtype=float,
    )
    if card_equal_weight:
        per_card = defaultdict(float)
        for record, weight in zip(successes, raw):
            per_card[record["_card"]] += weight
        cards = len(per_card)
        raw = np.asarray(
            [
                weight / per_card[record["_card"]] / cards
                for record, weight in zip(successes, raw)
            ],
            dtype=float,
        )
    weights = raw
    means = np.average(features, axis=0, weights=weights)
    scales = np.sqrt(np.average((features - means) ** 2, axis=0, weights=weights))
    scales[scales < 1e-9] = 1.0
    standardized = (features - means) / scales
    design = np.column_stack((np.ones(len(standardized)), standardized))
    penalty = np.diag([0.0, *([float(alpha)] * standardized.shape[1])])
    normal = design.T @ (weights[:, None] * design) + penalty
    target = design.T @ (weights * targets)
    coefficients = np.linalg.pinv(normal) @ target
    card_rows = Counter(r["_card"] for r in successes)
    return {
        "available": True,
        "model_family": "analytic_reference_log_residual_ridge",
        "label": "reserved",
        "feature_set": f"{BASE_FEATURE_SET}_plus_card",
        "base_feature_set": BASE_FEATURE_SET,
        "card_feature_names": list(card_names),
        "feature_names": joint_feature_names(card_names),
        "alpha": float(alpha),
        "card_equal_weight": bool(card_equal_weight),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "fit_success_rows": len(successes),
        "fit_scenarios": len(scenario_counts),
        "fit_rows_by_card": dict(card_rows),
    }


def predict_joint(record: Mapping[str, Any], model: Mapping[str, Any]) -> float:
    features = joint_features(record, model["card_feature_names"])
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    residual = float(model["intercept"]) + float(
        ((features - means) / scales) @ coefficients
    )
    prediction = float(record["memory"]["analytic_reference_bytes"]) * math.exp(
        residual
    )
    if not math.isfinite(prediction) or prediction <= 0:
        raise ValueError("Joint memory model produced a non-positive prediction")
    return prediction


# ---------------------------------------------------------------- evaluation
def centre_metrics(
    records: Sequence[Mapping[str, Any]],
    predict,
) -> dict[str, Any]:
    """Scenario-equal weighted absolute percentage error on success rows."""
    rows = [
        r
        for r in records
        if outcome(r) == "success" and observed_reserved(r) is not None
    ]
    if not rows:
        return {"rows": 0}
    by_scenario: dict[str, list[float]] = defaultdict(list)
    apes: list[float] = []
    signed: list[float] = []
    for record in rows:
        observed = float(observed_reserved(record))
        predicted = predict(record)
        ape = abs(predicted - observed) / observed
        apes.append(ape)
        signed.append((predicted - observed) / observed)
        by_scenario[namespaced_scenario_id(record)].append(ape)
    scenario_equal = float(
        np.mean([float(np.mean(v)) for v in by_scenario.values()])
    )
    return {
        "rows": len(rows),
        "scenarios": len(by_scenario),
        "row_mape": float(np.mean(apes)),
        "scenario_equal_mape": scenario_equal,
        "ape_median": float(np.median(apes)),
        "ape_p90": float(np.percentile(apes, 90)),
        "signed_mean": float(np.mean(signed)),
        "signed_p10": float(np.percentile(signed, 10)),
        "signed_p90": float(np.percentile(signed, 90)),
    }


def scenario_folds(
    records: Sequence[Mapping[str, Any]], folds: int
) -> list[list[str]]:
    ids = sorted({namespaced_scenario_id(r) for r in records})
    buckets: list[list[str]] = [[] for _ in range(folds)]
    for index, scenario in enumerate(ids):
        buckets[index % folds].append(scenario)
    return buckets


def leave_scenario_out_per_card(
    records: Sequence[Mapping[str, Any]],
    *,
    card_names: Sequence[str],
    alpha: float,
    card_equal_weight: bool,
    folds: int,
) -> dict[str, dict[str, Any]]:
    """One pass of scenario-fold CV, scored separately per card.

    Fits each fold ONCE and scores every card from it, so adding a card does
    not multiply the fitting cost.
    """
    buckets = scenario_folds(records, folds)
    collected: dict[str, list[tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for held in buckets:
        held_set = set(held)
        train = [r for r in records if namespaced_scenario_id(r) not in held_set]
        test = [r for r in records if namespaced_scenario_id(r) in held_set]
        if not test or not train:
            continue
        try:
            model = fit_joint(
                train,
                card_names=card_names,
                alpha=alpha,
                card_equal_weight=card_equal_weight,
            )
        except ValueError:
            continue
        for record in test:
            if outcome(record) != "success":
                continue
            if observed_reserved(record) is None:
                continue
            collected[record["_card"]].append(
                (record, predict_joint(record, model))
            )
    out: dict[str, dict[str, Any]] = {}
    for card, pairs in collected.items():
        lookup = {id(record): value for record, value in pairs}
        out[card] = centre_metrics(
            [record for record, _ in pairs], lambda r: lookup[id(r)]
        )
    return out


def leave_scenario_out(
    records: Sequence[Mapping[str, Any]],
    *,
    card_names: Sequence[str],
    alpha: float,
    card_equal_weight: bool,
    folds: int,
    evaluate_card: str | None,
) -> dict[str, Any]:
    per_card = leave_scenario_out_per_card(
        records,
        card_names=card_names,
        alpha=alpha,
        card_equal_weight=card_equal_weight,
        folds=folds,
    )
    if evaluate_card is None:
        merged = [m for m in per_card.values()]
        if not merged:
            return {"rows": 0}
        return {
            "rows": sum(m["rows"] for m in merged),
            "scenario_equal_mape": float(
                np.mean([m["scenario_equal_mape"] for m in merged])
            ),
        }
    return per_card.get(evaluate_card, {"rows": 0})


def cross_card(
    records: Sequence[Mapping[str, Any]],
    *,
    card_names: Sequence[str],
    alpha: float,
    train_card: str,
    test_card: str,
) -> dict[str, Any]:
    train = [r for r in records if r["_card"] == train_card]
    test = [r for r in records if r["_card"] == test_card]
    model = fit_joint(
        train,
        card_names=card_names,
        alpha=alpha,
        card_equal_weight=False,
    )
    metrics = centre_metrics(test, lambda r: predict_joint(r, model))
    metrics["train_card"] = train_card
    metrics["test_card"] = test_card
    metrics["train_rows"] = model["fit_success_rows"]
    metrics["note"] = (
        "card features are identically zero within a single-card training set, "
        "so they cannot be estimated; this measures the shared 28-feature core"
    )
    return metrics


def leave_model_out(
    records: Sequence[Mapping[str, Any]],
    *,
    card_names: Sequence[str],
    alpha: float,
    card_equal_weight: bool,
) -> dict[str, Any]:
    def model_id(record: Mapping[str, Any]) -> str:
        return str((record.get("scenario") or {}).get("model_id") or "?")

    ids = sorted({model_id(r) for r in records})
    folds = []
    for held in ids:
        train = [r for r in records if model_id(r) != held]
        test = [r for r in records if model_id(r) == held]
        cards_in_train = {r["_card"] for r in train}
        try:
            model = fit_joint(
                train,
                card_names=card_names,
                alpha=alpha,
                card_equal_weight=card_equal_weight,
            )
        except ValueError:
            continue
        entry: dict[str, Any] = {
            "held_out_model_id": held,
            "train_cards": sorted(cards_in_train),
            "train_rows": model["fit_success_rows"],
        }
        for card in sorted({r["_card"] for r in test}):
            per = [r for r in test if r["_card"] == card]
            entry[card] = centre_metrics(per, lambda r: predict_joint(r, model))
        folds.append(entry)
    return {"folds": folds}


# ------------------------------------------------------------------ upper bound
def fit_upper_bound(
    records: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    *,
    coverage: float,
    min_samples: int,
    centre_override: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    """One-sided conformal log-residual upper bound, per card x selector.

    Mirrors the existing per-card tail contract: the operational prediction is
    ``centre * exp(max(success_q, oom_guard, 0))``.  Keyed per card because the
    safe limit and allocator behaviour differ.

    Pass ``centre_override`` (record id -> out-of-fold centre) to build the
    tail from out-of-fold residuals, which is the honest version - an in-sample
    tail understates how much inflation real unseen rows need.
    """
    residuals: dict[str, list[float]] = defaultdict(list)
    oom_guard: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if centre_override is not None:
            centre = centre_override.get(id(record))
            if centre is None:
                continue
        else:
            centre = predict_joint(record, model)
        key = _tail_key(record)
        if outcome(record) == "success":
            observed = observed_reserved(record)
            if observed is None:
                continue
            residuals[key].append(math.log(float(observed) / centre))
        elif outcome(record) == "oom":
            lower = (record.get("memory") or {}).get("observed", {}).get(
                "right_censor_lower_bytes"
            )
            if lower:
                oom_guard[key].append(math.log(float(lower) / centre))
    out: dict[str, Any] = {
        "coverage": coverage,
        "min_samples": min_samples,
        "keyed_by": "card|[training_mode,zero_stage,gc,packing]",
        "selectors": {},
    }
    for key in sorted(set(residuals) | set(oom_guard)):
        success = sorted(residuals.get(key, []))
        ooms = oom_guard.get(key, [])
        entry: dict[str, Any] = {
            "success_samples": len(success),
            "oom_samples": len(ooms),
        }
        if len(success) >= min_samples:
            rank = min(
                len(success) - 1,
                max(0, math.ceil(coverage * (len(success) + 1)) - 1),
            )
            entry["success_log_residual_upper"] = float(success[rank])
            entry["available"] = True
        else:
            entry["success_log_residual_upper"] = None
            entry["available"] = False
        entry["oom_log_residual_lower"] = (
            float(max(ooms)) if ooms else None
        )
        out["selectors"][key] = entry
    global_success = sorted(
        value for values in residuals.values() for value in values
    )
    if global_success:
        rank = min(
            len(global_success) - 1,
            max(0, math.ceil(coverage * (len(global_success) + 1)) - 1),
        )
        out["global_fallback_log_residual_upper"] = float(global_success[rank])
    return out


def out_of_fold_centres(
    records: Sequence[Mapping[str, Any]],
    *,
    card_names: Sequence[str],
    alpha: float,
    card_equal_weight: bool,
    folds: int,
) -> dict[int, float]:
    """Out-of-fold centre prediction for EVERY row, success and OOM alike.

    OOM rows are needed too: the tail's OOM guard is measured against the
    centre, so it has to use the same out-of-fold centre.
    """
    buckets = scenario_folds(records, folds)
    out: dict[int, float] = {}
    for held in buckets:
        held_set = set(held)
        train = [r for r in records if namespaced_scenario_id(r) not in held_set]
        test = [r for r in records if namespaced_scenario_id(r) in held_set]
        if not train or not test:
            continue
        try:
            model = fit_joint(
                train,
                card_names=card_names,
                alpha=alpha,
                card_equal_weight=card_equal_weight,
            )
        except ValueError:
            continue
        for record in test:
            out[id(record)] = predict_joint(record, model)
    return out


def _tail_key(record: Mapping[str, Any]) -> str:
    selector = record.get("selector") or {}
    return "{}|{}".format(
        record["_card"],
        json.dumps(
            [
                selector.get("training_mode"),
                int(selector.get("zero_stage") or 0),
                bool(selector.get("gradient_checkpointing")),
                bool(selector.get("packing")),
            ]
        ),
    )


def _inflate(
    record: Mapping[str, Any], centre: float, tail: Mapping[str, Any]
) -> float:
    entry = (tail.get("selectors") or {}).get(_tail_key(record)) or {}
    margin = entry.get("success_log_residual_upper")
    if margin is None:
        margin = tail.get("global_fallback_log_residual_upper") or 0.0
    guard = entry.get("oom_log_residual_lower")
    return centre * math.exp(max(float(margin), float(guard or 0.0), 0.0))


def upper_bound_value(
    record: Mapping[str, Any],
    model: Mapping[str, Any],
    tail: Mapping[str, Any],
) -> float:
    return _inflate(record, predict_joint(record, model), tail)


def admission_metrics(
    records: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    tail: Mapping[str, Any],
    *,
    centre_override: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    """Admission quality under the established per-card contract.

    Denominators match the existing frozen 4090 report so the numbers are
    comparable:

      * ``safe_success``   - ran AND observed peak <= safe limit.  Rejecting one
                             of these is a genuine false reject.
      * ``unsafe_success`` - ran but observed peak exceeded the safe limit.
                             Rejecting these is CORRECT, not a false reject.
      * ``oom``            - admitting one of these is a false-safe, the only
                             failure mode that actually breaks a user's job.
    """
    per_card: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "oom_total": 0,
            "oom_admitted_false_safe": 0,
            "safe_success_total": 0,
            "safe_success_rejected": 0,
            "unsafe_success_total": 0,
            "unsafe_success_admitted": 0,
        }
    )
    for record in records:
        card = record["_card"]
        limit = float(record["memory"]["safe_limit_bytes"])
        if centre_override is not None:
            centre = centre_override.get(id(record))
            if centre is None:
                continue
            upper = _inflate(record, centre, tail)
        else:
            upper = upper_bound_value(record, model, tail)
        admit = upper <= limit
        state = outcome(record)
        if state == "oom":
            per_card[card]["oom_total"] += 1
            if admit:
                per_card[card]["oom_admitted_false_safe"] += 1
        elif state == "success":
            observed = observed_reserved(record)
            if observed is None:
                continue
            if float(observed) <= limit:
                per_card[card]["safe_success_total"] += 1
                if not admit:
                    per_card[card]["safe_success_rejected"] += 1
            else:
                per_card[card]["unsafe_success_total"] += 1
                if admit:
                    per_card[card]["unsafe_success_admitted"] += 1
    out = {}
    for card, counts in sorted(per_card.items()):
        oom_total = counts["oom_total"]
        safe_total = counts["safe_success_total"]
        out[card] = {
            **counts,
            "oom_rejection_recall": (
                (oom_total - counts["oom_admitted_false_safe"]) / oom_total
                if oom_total
                else None
            ),
            "false_safe_oom_rate": (
                counts["oom_admitted_false_safe"] / oom_total
                if oom_total
                else None
            ),
            "safe_success_admission_recall": (
                (safe_total - counts["safe_success_rejected"]) / safe_total
                if safe_total
                else None
            ),
            "false_reject_rate": (
                counts["safe_success_rejected"] / safe_total
                if safe_total
                else None
            ),
        }
    return out


# ----------------------------------------------------------------- baselines
def per_card_baseline(
    records: Sequence[Mapping[str, Any]], *, folds: int
) -> dict[str, Any]:
    """Existing contract: one model per card, no card features."""
    H = _import_shared()
    out: dict[str, Any] = {}
    for card in sorted({r["_card"] for r in records}):
        card_records = [r for r in records if r["_card"] == card]
        best = None
        for alpha in ALPHA_GRID:
            metrics = leave_scenario_out(
                card_records,
                card_names=(),
                alpha=alpha,
                card_equal_weight=False,
                folds=folds,
                evaluate_card=card,
            )
            if metrics.get("rows", 0) == 0:
                continue
            if best is None or metrics["scenario_equal_mape"] < best[1][
                "scenario_equal_mape"
            ]:
                best = (alpha, metrics)
        if best is None:
            continue
        out[card] = {
            "selected_alpha": best[0],
            "leave_scenario_out": best[1],
            "rows": len(card_records),
        }
    return out


# ---------------------------------------------------------------------- main
def build_report(*, folds: int, coverage: float, min_samples: int) -> dict[str, Any]:
    h800, h800_reasons = build_h800_records()
    rtx = build_rtx4090_records()
    records = [*h800, *rtx]

    inventory_counts = {
        card: {
            "records": sum(1 for r in records if r["_card"] == card),
            "success": sum(
                1
                for r in records
                if r["_card"] == card and outcome(r) == "success"
            ),
            "oom": sum(
                1 for r in records if r["_card"] == card and outcome(r) == "oom"
            ),
            "models": dict(
                Counter(
                    str((r.get("scenario") or {}).get("model_id") or "?")
                    for r in records
                    if r["_card"] == card
                )
            ),
        }
        for card in sorted({r["_card"] for r in records})
    }

    baselines = per_card_baseline(records, folds=folds)

    candidates: list[dict[str, Any]] = []
    for name, card_names in CARD_FEATURE_CANDIDATES.items():
        for card_equal in (False, True):
            for alpha in ALPHA_GRID:
                entry: dict[str, Any] = {
                    "candidate": name,
                    "card_feature_names": list(card_names),
                    "alpha": alpha,
                    "card_equal_weight": card_equal,
                }
                per_card_cv = leave_scenario_out_per_card(
                    records,
                    card_names=card_names,
                    alpha=alpha,
                    card_equal_weight=card_equal,
                    folds=folds,
                )
                cards = sorted({r["_card"] for r in records})
                if any(
                    per_card_cv.get(card, {}).get("rows", 0) == 0
                    for card in cards
                ):
                    continue
                worst = max(
                    per_card_cv[card]["scenario_equal_mape"] for card in cards
                )
                entry["per_card_leave_scenario_out"] = per_card_cv
                entry["worst_card_scenario_equal_mape"] = worst
                # regression guard against the per-card baseline
                entry["vs_baseline"] = {
                    card: {
                        "joint": per_card_cv[card]["scenario_equal_mape"],
                        "baseline": baselines[card]["leave_scenario_out"][
                            "scenario_equal_mape"
                        ],
                        "delta": per_card_cv[card]["scenario_equal_mape"]
                        - baselines[card]["leave_scenario_out"][
                            "scenario_equal_mape"
                        ],
                    }
                    for card in per_card_cv
                    if card in baselines
                }
                candidates.append(entry)

    if not candidates:
        raise RuntimeError("No usable joint candidate")

    # Selection rule, fixed before looking at the numbers:
    #   minimise the WORST per-card scenario-equal MAPE.  A joint model that
    #   helps one card by hurting the other is not acceptable for a platform
    #   that must serve both.
    selected = min(candidates, key=lambda e: e["worst_card_scenario_equal_mape"])

    card_names = tuple(selected["card_feature_names"])
    alpha = float(selected["alpha"])
    card_equal = bool(selected["card_equal_weight"])

    final = fit_joint(
        records,
        card_names=card_names,
        alpha=alpha,
        card_equal_weight=card_equal,
    )
    tail = fit_upper_bound(
        records, final, coverage=coverage, min_samples=min_samples
    )

    # Honest version: build the tail from out-of-fold centres and score
    # admission with those same out-of-fold centres.  This is what an unseen
    # scenario would actually see.
    oof = out_of_fold_centres(
        records,
        card_names=card_names,
        alpha=alpha,
        card_equal_weight=card_equal,
        folds=folds,
    )
    tail_oof = fit_upper_bound(
        records,
        final,
        coverage=coverage,
        min_samples=min_samples,
        centre_override=oof,
    )

    cross = {
        "h800_to_rtx4090": cross_card(
            records,
            card_names=card_names,
            alpha=alpha,
            train_card="h800",
            test_card="rtx4090",
        ),
        "rtx4090_to_h800": cross_card(
            records,
            card_names=card_names,
            alpha=alpha,
            train_card="rtx4090",
            test_card="h800",
        ),
    }

    model_holdout = leave_model_out(
        records,
        card_names=card_names,
        alpha=alpha,
        card_equal_weight=card_equal,
    )

    in_sample = {
        card: centre_metrics(
            [r for r in records if r["_card"] == card],
            lambda r: predict_joint(r, final),
        )
        for card in sorted({r["_card"] for r in records})
    }

    admission = admission_metrics(records, final, tail)
    admission_oof = admission_metrics(
        records, final, tail_oof, centre_override=oof
    )

    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "joint_card_memory_centre_fitted_and_validated",
        "analysis_only": True,
        "publishable": False,
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "purpose": (
            "One memory-centre coefficient vector covering H800 and RTX 4090, "
            "so the platform carries a single contract instead of two."
        ),
        "contract": {
            "formula": (
                "reserved_centre = analytic_reference_bytes * "
                "exp(intercept + standardised([physical_shares_28, card_features]) @ beta)"
            ),
            "operational_formula": (
                "reserved_upper = reserved_centre * "
                "exp(max(success_log_residual_upper, oom_log_residual_lower, 0))"
            ),
            "base_feature_set": BASE_FEATURE_SET,
            "base_feature_source": (
                "h800_challenger_modeling._memory_features - the SAME function "
                "both per-card models already use; unchanged by this work"
            ),
            "card_feature_names": list(card_names),
            "unknown_card_policy": (
                "set every card feature to zero; the prediction falls back to "
                "the shared 28-feature core, which is what the H800-trained "
                "core does on an unseen card"
            ),
        },
        "data": {
            "by_card": inventory_counts,
            "h800_admission_reasons": h800_reasons,
            "model_overlap": sorted(
                {
                    str((r.get("scenario") or {}).get("model_id") or "?")
                    for r in h800
                }
                & {
                    str((r.get("scenario") or {}).get("model_id") or "?")
                    for r in rtx
                }
            ),
        },
        "selection": {
            "rule": (
                "minimise the worst per-card leave-scenario-out "
                "scenario-equal MAPE; fixed before inspecting results"
            ),
            "selected": {
                "candidate": selected["candidate"],
                "card_feature_names": selected["card_feature_names"],
                "alpha": selected["alpha"],
                "card_equal_weight": selected["card_equal_weight"],
                "worst_card_scenario_equal_mape": selected[
                    "worst_card_scenario_equal_mape"
                ],
                "vs_baseline": selected["vs_baseline"],
            },
            "candidates": candidates,
        },
        "per_card_baseline": baselines,
        "frozen_model": {
            "centre": final,
            "tail": tail,
            "tail_out_of_fold": tail_oof,
        },
        "evaluation": {
            "in_sample_by_card": in_sample,
            "cross_card_holdout": cross,
            "complete_model_holdout": model_holdout,
            "admission_in_sample_tail": admission,
            "admission_out_of_fold": admission_oof,
            "admission_comparison_note": (
                "admission_out_of_fold is the honest number: both the centre "
                "and the conformal margin come from folds that never saw the "
                "row.  admission_in_sample_tail is optimistic and is kept only "
                "to show the size of the gap."
            ),
        },
        "limitations": [
            "Model coverage differs per card: H800 has 1.7B/4B/8B/14B, RTX 4090 "
            "has 0.6B/1.7B/4B.  Only 1.7B and 4B are shared, so the card "
            "features are anchored on two model scales.",
            "RTX 4090 rows are largely thermal or power limited.  That affects "
            "throughput far more than peak memory, but it is not a clean-clock "
            "population.",
            "The RTX 4090 record builder was run with the ThroughputPredictor "
            "implementation-SHA gate bypassed.  That gate guards the frozen "
            "throughput head, which this memory fit never reads.  The only "
            "delta between the current implementation file and the last "
            "committed one is a try/except around the scipy import.",
            "This report is analysis_only.  Replacing the two per-card frozen "
            "models on the platform is a separate decision and needs the Go "
            "side to read the card features.",
            "The upper bound is conformal on the fitted population, not on a "
            "fresh holdout.  Admission numbers here are in-sample for the "
            "tail; treat them as an upper bound on real performance.",
        ],
        "source_bindings": {
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "h800_observations": {
                "path": str(ROOT / "artifacts" / "canonical_h800_observations.jsonl"),
                "sha256": sha256_file(
                    ROOT / "artifacts" / "canonical_h800_observations.jsonl"
                ),
            },
            "rtx4090_collected_results": {
                "path": str(
                    ROOT
                    / "campaigns"
                    / "rtx4090_20260717"
                    / "artifacts"
                    / "collected_results.json"
                ),
                "sha256": sha256_file(
                    ROOT
                    / "campaigns"
                    / "rtx4090_20260717"
                    / "artifacts"
                    / "collected_results.json"
                ),
            },
            "model_inventory": {
                "path": str(ROOT / "artifacts" / "model_inventory.json"),
                "sha256": sha256_file(ROOT / "artifacts" / "model_inventory.json"),
            },
            "hardware": {
                "path": str(ROOT / "config" / "hardware.json"),
                "sha256": sha256_file(ROOT / "config" / "hardware.json"),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "joint_card_memory_centre_v1.json",
    )
    args = parser.parse_args()
    report = build_report(
        folds=args.folds,
        coverage=args.coverage,
        min_samples=args.min_samples,
    )
    report["report_sha256"] = sha256_json(report)
    write_json(args.output, report)
    summary = {
        "output": str(args.output),
        "selected": report["selection"]["selected"],
        "in_sample_by_card": {
            card: round(value["scenario_equal_mape"], 4)
            for card, value in report["evaluation"]["in_sample_by_card"].items()
        },
        "admission": report["evaluation"]["admission_out_of_fold"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
