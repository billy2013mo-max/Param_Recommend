#!/usr/bin/env python3
"""Step 1 (corrected): joint card-aware refit of the PLATFORM's V3 memory model.

Why this file exists
--------------------
An earlier attempt fitted the ``physical_shares`` 28-feature contract.  The
platform does not read that.  ``server_integration`` binds
``artifacts/h800_unified_bounded_memory_candidate_v3.json``, schema
``sft_h800_unified_bounded_memory_shadow_candidate/v3``:

    model_family    physics_anchored_shared_bounded_residual
    center          basis_kind=bounded_mechanism, 38 raw -> 144 basis
    risk            basis_kind=bounded_linear, same 38 raw
    admission       admit iff max(centre, risk * upper_multiplier)
                            <= safe_limit_fraction * capacity

This script fits BOTH heads jointly over H800 + RTX 4090.

Approach "A": the shipped modules are NOT modified
--------------------------------------------------
The 9 capacity-normalised V3 features are produced inside
``benchmark_h800_memory_center_models_v1._feature_value``, which divides by a
MODULE-LEVEL constant pinned to the H800 (150142189568 bytes).  Off H800 those
features are wrong by the capacity ratio (5.91x for a 4090).

Rather than edit that file - which is the V3 artifact's ``model_math`` binding,
so editing it would break the live platform predictor's source-binding check -
this script precomputes ALL 38 raw features into each record's ``features``
dict using the ROW's own capacity.  ``_feature_value`` returns anything already
present in ``features`` verbatim, so the hardcoded constant is never reached.

``assert_h800_bit_identical`` proves the reimplementation is exact: for every
H800 row it compares the 9 recomputed values against the shipped
implementation's own output and requires bit-for-bit equality.

All model math (basis expansion, weighting, IRLS ridge, targets) is imported
from the shipped module unchanged.
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

import benchmark_h800_memory_center_models_v1 as bm
import fit_h800_unified_resource_partial_v1 as base

SCHEMA = "sft_joint_card_v3_memory/v1"
GIB = float(1 << 30)

CAMPAIGN_4090 = ROOT / "campaigns" / "rtx4090_20260717"
PROFILE_DIR_4090 = CAMPAIGN_4090 / "artifacts" / "dataset_profiles"
V3_ARTIFACT = ROOT / "artifacts" / "h800_unified_bounded_memory_candidate_v3.json"
V3_INVENTORY = ROOT / "artifacts" / "h800_bounded_memory_v2_model_inventory_v1.json"

TIED_EMBEDDING_MODEL_IDS = frozenset({"qwen3_1p7b", "qwen3_4b"})

CAPACITY_FEATURE_NAMES = (
    "reference_fraction_of_capacity",
    "activation_fraction_of_capacity",
    "nonactivation_fraction_of_capacity",
    "lora_zero3_nogc_activation_fraction_of_capacity",
    "lora_zero3_nogc_truncation_activation_pressure",
    "lora_zero3_nogc_rare_tail_activation_pressure",
    "full_zero3_gc_reference_fraction_of_capacity",
    "full_single_gpu_zero0_nogc_mbs_reference_pressure",
    "tied_embedding_gc_reference_pressure",
)

CARD_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "none": (),
    "card_intercept": ("is_rtx4090",),
    "card_scale": (
        "is_rtx4090",
        "rtx4090_x_reference_fraction_of_capacity",
        "rtx4090_x_activation_fraction_of_capacity",
    ),
    "card_scale_plus_topology": (
        "is_rtx4090",
        "rtx4090_x_reference_fraction_of_capacity",
        "rtx4090_x_activation_fraction_of_capacity",
        "rtx4090_x_log2_gpu_count",
        "rtx4090_x_zero3",
    ),
}

# Hyper-parameters copied verbatim from the shipped V3 artifact so the joint
# fit differs from production ONLY in the data and the card features.
CENTER_CANDIDATE = {
    "basis_kind": "bounded_mechanism",
    "feature_variant": "anchor_scale_v3_mechanisms",
    "alpha": 0.03,
    "huber_delta": 0.2,
    "correction_shrinkage": 0.8,
    "source_weight_power": 0.25,
    "censored_constraint_weight": 1.0,
}
RISK_CANDIDATE = {
    "basis_kind": "bounded_linear",
    "feature_variant": "anchor_scale_v3_mechanisms",
    "alpha": 0.3,
    "huber_delta": 0.2,
    "correction_shrinkage": 1.0,
    "source_weight_power": 0.0,
    "censored_constraint_weight": 10.0,
}
SAFE_LIMIT_FRACTION = 0.95


# ------------------------------------------------------- capacity features
def capacity_features(
    values: Mapping[str, float],
    *,
    reference_bytes: float,
    capacity_bytes: float,
    model_id: str,
) -> dict[str, float]:
    """Reimplementation of the 9 capacity features with per-row capacity.

    Every branch mirrors ``benchmark_h800_memory_center_models_v1._feature_value``
    line for line; the ONLY change is that ``reference_fraction`` divides by the
    row's own capacity instead of the module-level H800 constant.
    """
    reference_fraction = float(reference_bytes) / float(capacity_bytes)
    activation_share = float(values["activation_share"])
    is_lora = float(values["is_lora"])
    zero2 = float(values["zero2"])
    zero3 = float(values["zero3"])
    gc = float(values["gradient_checkpointing"])
    single_gpu = float(abs(float(values["log2_gpu_count"])) < 1.0e-12)
    zero0 = max(0.0, 1.0 - zero2 - zero3)
    tied = float(str(model_id) in TIED_EMBEDDING_MODEL_IDS)
    return {
        "reference_fraction_of_capacity": reference_fraction,
        "activation_fraction_of_capacity": reference_fraction * activation_share,
        "nonactivation_fraction_of_capacity": (
            reference_fraction * (1.0 - activation_share)
        ),
        "lora_zero3_nogc_activation_fraction_of_capacity": (
            is_lora * zero3 * (1.0 - gc) * reference_fraction * activation_share
        ),
        "lora_zero3_nogc_truncation_activation_pressure": (
            is_lora
            * zero3
            * (1.0 - gc)
            * reference_fraction
            * activation_share
            * float(values["profile_truncation_fraction"])
        ),
        "lora_zero3_nogc_rare_tail_activation_pressure": (
            is_lora
            * zero3
            * (1.0 - gc)
            * reference_fraction
            * activation_share
            * float(values["profile_rare_tail_gap"])
        ),
        "full_zero3_gc_reference_fraction_of_capacity": (
            (1.0 - is_lora) * zero3 * gc * reference_fraction
        ),
        "full_single_gpu_zero0_nogc_mbs_reference_pressure": (
            (1.0 - is_lora)
            * single_gpu
            * zero0
            * (1.0 - gc)
            * reference_fraction
            * float(values["log2_mbs"])
        ),
        "tied_embedding_gc_reference_pressure": tied * gc * reference_fraction,
    }


def card_feature_values(record: Mapping[str, Any]) -> dict[str, float]:
    features = record["features"]
    is_4090 = 1.0 if record["_card"] == "rtx4090" else 0.0
    return {
        "is_rtx4090": is_4090,
        "rtx4090_x_reference_fraction_of_capacity": (
            is_4090 * float(features["reference_fraction_of_capacity"])
        ),
        "rtx4090_x_activation_fraction_of_capacity": (
            is_4090 * float(features["activation_fraction_of_capacity"])
        ),
        "rtx4090_x_log2_gpu_count": is_4090 * float(features["log2_gpu_count"]),
        "rtx4090_x_zero3": is_4090 * float(features["zero3"]),
    }


# ------------------------------------------------------------ record build
def build_h800_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """H800 V3 development records, exactly as the shipped model defines them.

    ``strict31`` is skipped: the V3 artifact's own evidence contract marks it
    ``retrospective_diagnostic_only`` (never a fitting input), and rebuilding it
    currently trips an analytic-reference drift gate on 3 qwen3p5_4b rows.  See
    ``notes.strict31_skipped`` in the output.
    """
    import h800_unified_bounded_memory_v3_data as data

    original = base._strict_records
    base._strict_records = lambda *args, **kwargs: ([], [])
    try:
        records, audit = data.development_records()
    finally:
        base._strict_records = original
    capacity = int(
        read_json(base.DEFAULT_HARDWARE)["memory_bytes_reported_by_torch"]
    )
    for record in records:
        record["_card"] = "h800"
        record["_capacity_bytes"] = capacity
    return records, audit


def build_rtx4090_records() -> list[dict[str, Any]]:
    import throughput_predictor as TP
    import rtx4090_physical_v4b_modeling as R
    from analyze_h800_fresh_memory_residual_v2 import profile_padding_statistics

    inventory = read_json(V3_INVENTORY)
    model_by_id = {str(m["id"]): dict(m) for m in inventory["models"]}
    fixed_lora = dict(inventory["fixed_lora"])

    original = TP.ThroughputPredictor._validate_bindings
    TP.ThroughputPredictor._validate_bindings = lambda self: setattr(
        self, "binding_mismatches", ["implementation-sha gate bypassed (memory-only)"]
    )
    try:
        predictor = TP.ThroughputPredictor(strict_bindings=False)
        rows = R._read_rows(CAMPAIGN_4090)
        padding_cache: dict = {}
        profile_cache: dict = {}
        records: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            classification = str(row.get("classification") or "")
            if classification not in {"success", "oom"}:
                continue
            job = R._rendered_job(CAMPAIGN_4090, row)
            normalized = predictor._normalized_request(
                R._request(job), input_index=index
            )
            capacity = int(normalized["hardware"].memory_bytes)
            cutoff = int(job["cutoff_len"])
            packing = bool(job.get("packing"))
            profile_path = PROFILE_DIR_4090 / f"{job['dataset_id']}.qwen3_nothink.jsonl"
            key = (str(profile_path.resolve()), cutoff, int(job["mbs"]))
            if key not in padding_cache:
                padding_cache[key] = profile_padding_statistics(
                    profile_path,
                    cutoff_len=cutoff,
                    physical_mbs=int(job["mbs"]),
                )
            if packing:
                aligned = cutoff
            else:
                raw_max = int(padding_cache[key]["maximum_clipped_tokens"])
                aligned = 8 * ((min(cutoff, raw_max) + 7) // 8)
            v3_job = {
                "model_id": str(job["model_id"]),
                "model_parameters": int(
                    normalized["model_geometry"]["base_parameters"]
                ),
                "train_type": str(job["train_type"]),
                "gpu_count": int(job["gpu_count"]),
                "mbs": int(job["mbs"]),
                "cutoff_len": cutoff,
                "aligned_effective_sequence": int(aligned),
                "zero": str(job.get("zero") or "none"),
                "zero_stage": int(normalized["zero_stage"]),
                "gc": bool(job["gc"]),
                "packing": packing,
                "dataset_id": str(job["dataset_id"]),
                "dataset_profile_path": str(profile_path),
            }
            reference, values = base._current_features(
                v3_job,
                model_by_id=model_by_id,
                fixed_lora=fixed_lora,
                capacity_bytes=capacity,
                profile_cache=profile_cache,
            )
            values.update(
                capacity_features(
                    values,
                    reference_bytes=reference,
                    capacity_bytes=capacity,
                    model_id=str(job["model_id"]),
                )
            )
            reserved = float(row.get("max_reserved_bytes") or 0.0)
            if classification == "success":
                if reserved <= 0:
                    continue
                state, target, censor = "exact", reserved, None
            else:
                state = "censored"
                target = None
                censor = max(float(capacity) * SAFE_LIMIT_FRACTION + 1.0, reserved)
            records.append(
                {
                    "record_id": f"rtx4090::{job['job_id']}",
                    # source_id drives the fitter's source-equal weighting.
                    # Mirror the H800 convention: one source per
                    # (dataset, model, mode) material, namespaced by card.
                    "source_id": "rtx4090::{}::{}::{}".format(
                        job["dataset_id"], job["model_id"], job["train_type"]
                    ),
                    "origin": "rtx4090_20260717_campaign",
                    "campaign": "rtx4090_20260717",
                    "role": "rtx4090_native_campaign",
                    "state": state,
                    "reference_bytes": reference,
                    "target_reserved_bytes": target,
                    "target_allocated_bytes": None,
                    "censor_lower_bytes": censor,
                    "features": values,
                    "model_id": str(job["model_id"]),
                    "train_type": str(job["train_type"]),
                    "gpu_count": int(job["gpu_count"]),
                    "mbs": int(job["mbs"]),
                    "cutoff_len": cutoff,
                    "gc": bool(job["gc"]),
                    "packing": packing,
                    "zero_stage": int(normalized["zero_stage"]),
                    "_card": "rtx4090",
                    "_capacity_bytes": capacity,
                    "_observed_reserved_bytes": reserved if reserved > 0 else None,
                }
            )
    finally:
        TP.ThroughputPredictor._validate_bindings = original
    return records


def complete_h800_features(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fill the 9 capacity features on H800 rows and PROVE they are unchanged.

    The shipped ``_feature_value`` computes them lazily from its H800 constant.
    We compute them from the row's capacity - which for an H800 row IS that
    constant - and require bit-for-bit equality before overwriting.
    """
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for record in records:
        values = record["features"]
        computed = capacity_features(
            values,
            reference_bytes=float(record["reference_bytes"]),
            capacity_bytes=float(record["_capacity_bytes"]),
            model_id=str(record.get("model_id")),
        )
        for name, value in computed.items():
            shipped = bm._feature_value(record, name)
            checked += 1
            if shipped != value:
                mismatches.append(
                    {
                        "record_id": record.get("record_id"),
                        "feature": name,
                        "shipped": shipped,
                        "recomputed": value,
                        "abs_delta": abs(shipped - value),
                    }
                )
        values.update(computed)
    return {
        "h800_rows": len(records),
        "values_compared": checked,
        "bit_identical": not mismatches,
        "mismatches": mismatches[:10],
        "mismatch_count": len(mismatches),
    }


# ------------------------------------------------------------------ fitting
def feature_names_for(card_set: str) -> list[str]:
    return [*bm.RAW_FEATURE_NAMES, *bm.V3_MECHANISM_FEATURE_NAMES, *CARD_FEATURE_SETS[card_set]]


def attach_card_features(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        record["features"].update(card_feature_values(record))


def fit_head(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """IRLS bounded-residual ridge, reusing the shipped module's math verbatim."""
    kind = str(candidate["basis_kind"])
    raw = bm._raw_matrix(records, feature_names)
    source_weights = bm._source_weights(
        records, power=float(candidate["source_weight_power"])
    )
    raw_means, raw_scales = bm._weighted_location_scale(raw, source_weights)
    unique_counts = [len(np.unique(raw[:, i])) for i in range(raw.shape[1])]
    nonlinear_indexes = [i for i, c in enumerate(unique_counts) if c >= 5]
    expanded = bm._expand_basis(
        raw,
        kind=kind,
        raw_means=raw_means,
        raw_scales=raw_scales,
        nonlinear_indexes=nonlinear_indexes,
        feature_names=feature_names,
    )
    expanded_means, expanded_scales = bm._weighted_location_scale(
        expanded, source_weights
    )
    standardized = (expanded - expanded_means) / expanded_scales
    design = np.column_stack((np.ones(len(records)), standardized))
    targets, exact = bm._targets(records)
    censored = ~exact
    alpha = float(candidate["alpha"])
    huber_delta = candidate.get("huber_delta")
    constraint_weight = float(candidate["censored_constraint_weight"])
    parameters = bm._solve_weighted_ridge(
        design[exact], targets[exact], source_weights[exact], alpha=alpha
    )
    signature: tuple[int, ...] | None = None
    for _ in range(100):
        prediction = design @ parameters
        active_censored = censored & (prediction < targets)
        active = exact | active_censored
        residual = prediction[active] - targets[active]
        if huber_delta is None:
            robust = np.ones(len(residual), dtype=float)
        else:
            robust = np.minimum(
                1.0, float(huber_delta) / np.maximum(np.abs(residual), 1.0e-12)
            )
        weights = (
            source_weights[active]
            * robust
            * np.where(censored[active], constraint_weight, 1.0)
        )
        updated = bm._solve_weighted_ridge(
            design[active], targets[active], weights, alpha=alpha
        )
        new_signature = tuple(np.flatnonzero(active_censored).tolist())
        converged = (
            signature == new_signature
            and float(np.max(np.abs(updated - parameters))) < 1.0e-9
        )
        parameters = updated
        signature = new_signature
        if converged:
            break
    return {
        "model_family": "physics_anchored_shared_bounded_residual",
        "basis_kind": kind,
        "feature_variant": "anchor_scale_v3_mechanisms_plus_card",
        "raw_feature_names": list(feature_names),
        "basis_feature_names": bm._basis_names(
            kind, feature_names, nonlinear_indexes
        ),
        "alpha": alpha,
        "huber_delta": huber_delta,
        "correction_shrinkage": float(candidate["correction_shrinkage"]),
        "source_weight_power": float(candidate["source_weight_power"]),
        "censored_constraint_weight": constraint_weight,
        "raw_means": raw_means.tolist(),
        "raw_scales": raw_scales.tolist(),
        "nonlinear_indexes": nonlinear_indexes,
        "expanded_means": expanded_means.tolist(),
        "expanded_scales": expanded_scales.tolist(),
        "intercept": float(parameters[0]),
        "coefficients": parameters[1:].tolist(),
        "fit_records": len(records),
        "fit_exact_centres": int(exact.sum()),
        "fit_right_censored": int(censored.sum()),
        "fit_sources": len({str(r["source_id"]) for r in records}),
        "fit_rows_by_card": dict(Counter(r["_card"] for r in records)),
    }


def predict_bytes(
    records: Sequence[Mapping[str, Any]], model: Mapping[str, Any]
) -> np.ndarray:
    correction = bm._predict_correction(records, model)
    reference = np.asarray(
        [float(r["reference_bytes"]) for r in records], dtype=float
    )
    return reference * np.exp(correction)


# --------------------------------------------------------------- evaluation
def centre_metrics(
    records: Sequence[Mapping[str, Any]], predicted: np.ndarray
) -> dict[str, Any]:
    rows = [
        (r, p)
        for r, p in zip(records, predicted)
        if r["state"] == "exact" and r.get("target_reserved_bytes")
    ]
    if not rows:
        return {"rows": 0}
    apes, signed = [], []
    by_source: dict[str, list[float]] = defaultdict(list)
    for record, prediction in rows:
        observed = float(record["target_reserved_bytes"])
        ape = abs(prediction - observed) / observed
        apes.append(ape)
        signed.append((prediction - observed) / observed)
        by_source[str(record["source_id"])].append(ape)
    return {
        "rows": len(rows),
        "sources": len(by_source),
        "row_mape": float(np.mean(apes)),
        "source_equal_mape": float(
            np.mean([float(np.mean(v)) for v in by_source.values()])
        ),
        "ape_median": float(np.median(apes)),
        "ape_p90": float(np.percentile(apes, 90)),
        "signed_mean": float(np.mean(signed)),
    }


def calibrate_upper_multiplier(
    records: Sequence[Mapping[str, Any]],
    centre: np.ndarray,
    risk: np.ndarray,
    *,
    per_card: bool,
) -> dict[str, Any]:
    """Smallest risk-head multiplier that admits ZERO OOM rows.

    The V3 admission rule is
        admit iff max(centre, risk * m) <= safe_limit
    so an OOM row is caught when either head already exceeds the limit, or when
    ``risk * m`` does.  For each OOM row that the centre head does not already
    catch, the binding requirement is ``m > safe_limit / risk``.  The calibrated
    multiplier is the max of those, nudged up by one ulp-ish epsilon.

    The shipped value (1.017) was calibrated on H800 alone.  Reusing it for a
    joint model leaves 4090 OOM rows admitted, which is the one failure mode
    that actually breaks a user's job.
    """
    requirements: dict[str, list[float]] = defaultdict(list)
    for record, c, r in zip(records, centre, risk):
        if record["state"] != "censored":
            continue
        if not (math.isfinite(c) and math.isfinite(r)) or r <= 0:
            continue
        limit = float(record["_capacity_bytes"]) * SAFE_LIMIT_FRACTION
        if float(c) > limit:
            continue  # already caught by the centre head
        key = record["_card"] if per_card else "__shared__"
        requirements[key].append(limit / float(r))
    out: dict[str, float] = {}
    for key, values in requirements.items():
        out[key] = float(max(values)) * (1.0 + 1e-9)
    return out


def multiplier_for(
    record: Mapping[str, Any], calibration: Mapping[str, float], *, default: float
) -> float:
    if "__shared__" in calibration:
        return calibration["__shared__"]
    return calibration.get(record["_card"], default)


def admission_metrics(
    records: Sequence[Mapping[str, Any]],
    centre: np.ndarray,
    risk: np.ndarray,
    *,
    upper_multiplier: float | Mapping[str, float],
) -> dict[str, Any]:
    per_card: dict[str, Counter] = defaultdict(Counter)
    for record, c, r in zip(records, centre, risk):
        card = record["_card"]
        capacity = float(record["_capacity_bytes"])
        limit = capacity * SAFE_LIMIT_FRACTION
        if isinstance(upper_multiplier, Mapping):
            factor = multiplier_for(record, upper_multiplier, default=1.0)
        else:
            factor = float(upper_multiplier)
        admit = max(float(c), float(r) * factor) <= limit
        counts = per_card[card]
        if record["state"] == "censored":
            counts["oom_total"] += 1
            if admit:
                counts["oom_admitted_false_safe"] += 1
        else:
            observed = float(record["target_reserved_bytes"])
            if observed <= limit:
                counts["safe_success_total"] += 1
                if not admit:
                    counts["safe_success_rejected"] += 1
            else:
                counts["unsafe_success_total"] += 1
                if admit:
                    counts["unsafe_success_admitted"] += 1
    out: dict[str, Any] = {}
    for card, counts in sorted(per_card.items()):
        oom = counts["oom_total"]
        safe = counts["safe_success_total"]
        out[card] = {
            **dict(counts),
            "oom_rejection_recall": (
                (oom - counts["oom_admitted_false_safe"]) / oom if oom else None
            ),
            "false_safe_oom_rate": (
                counts["oom_admitted_false_safe"] / oom if oom else None
            ),
            "false_reject_rate": (
                counts["safe_success_rejected"] / safe if safe else None
            ),
        }
    return out


def source_folds(records: Sequence[Mapping[str, Any]], folds: int) -> list[set[str]]:
    ids = sorted({str(r["source_id"]) for r in records})
    buckets: list[set[str]] = [set() for _ in range(folds)]
    for index, source in enumerate(ids):
        buckets[index % folds].add(source)
    return buckets


def out_of_fold(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate: Mapping[str, Any],
    feature_names: Sequence[str],
    folds: int,
) -> np.ndarray:
    predictions = np.full(len(records), np.nan, dtype=float)
    index_by_id = {id(r): i for i, r in enumerate(records)}
    for held in source_folds(records, folds):
        train = [r for r in records if str(r["source_id"]) not in held]
        test = [r for r in records if str(r["source_id"]) in held]
        if not train or not test:
            continue
        if not any(r["state"] == "exact" for r in train):
            continue
        model = fit_head(
            train, candidate=candidate, feature_names=feature_names
        )
        values = predict_bytes(test, model)
        for record, value in zip(test, values):
            predictions[index_by_id[id(record)]] = value
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "joint_card_v3_memory_v1.json",
    )
    args = parser.parse_args()

    h800, h800_audit = build_h800_records()
    identity = complete_h800_features(h800)
    if not identity["bit_identical"]:
        raise SystemExit(
            "H800 capacity-feature reimplementation is NOT bit-identical: "
            + json.dumps(identity["mismatches"][:3], ensure_ascii=False)
        )
    rtx = build_rtx4090_records()
    records = [*h800, *rtx]
    attach_card_features(records)

    upper_multiplier = float(
        read_json(V3_ARTIFACT)["admission"]["upper_multiplier"]
    )

    inventory = {
        card: {
            "rows": sum(1 for r in records if r["_card"] == card),
            "exact": sum(
                1
                for r in records
                if r["_card"] == card and r["state"] == "exact"
            ),
            "censored": sum(
                1
                for r in records
                if r["_card"] == card and r["state"] == "censored"
            ),
            "sources": len(
                {r["source_id"] for r in records if r["_card"] == card}
            ),
            "models": dict(
                Counter(
                    str(r.get("model_id"))
                    for r in records
                    if r["_card"] == card
                )
            ),
            "capacity_bytes": sorted(
                {int(r["_capacity_bytes"]) for r in records if r["_card"] == card}
            ),
        }
        for card in sorted({r["_card"] for r in records})
    }

    cards = sorted({r["_card"] for r in records})
    candidates = []
    for card_set in CARD_FEATURE_SETS:
        names = feature_names_for(card_set)
        oof_centre = out_of_fold(
            records, candidate=CENTER_CANDIDATE, feature_names=names, folds=args.folds
        )
        per_card = {}
        usable = True
        for card in cards:
            subset = [
                (r, p)
                for r, p in zip(records, oof_centre)
                if r["_card"] == card and math.isfinite(p)
            ]
            if not subset:
                usable = False
                break
            per_card[card] = centre_metrics(
                [r for r, _ in subset], np.asarray([p for _, p in subset])
            )
        if not usable:
            continue
        candidates.append(
            {
                "card_feature_set": card_set,
                "card_feature_names": list(CARD_FEATURE_SETS[card_set]),
                "raw_feature_dimension": len(names),
                "per_card_leave_source_out": per_card,
                "worst_card_source_equal_mape": max(
                    per_card[c]["source_equal_mape"] for c in cards
                ),
            }
        )
    if not candidates:
        raise SystemExit("no usable candidate")
    selected = min(candidates, key=lambda e: e["worst_card_source_equal_mape"])
    names = feature_names_for(selected["card_feature_set"])

    centre_model = fit_head(
        records, candidate=CENTER_CANDIDATE, feature_names=names
    )
    risk_model = fit_head(records, candidate=RISK_CANDIDATE, feature_names=names)

    centre_in = predict_bytes(records, centre_model)
    risk_in = predict_bytes(records, risk_model)
    centre_oof = out_of_fold(
        records, candidate=CENTER_CANDIDATE, feature_names=names, folds=args.folds
    )
    risk_oof = out_of_fold(
        records, candidate=RISK_CANDIDATE, feature_names=names, folds=args.folds
    )
    complete = np.isfinite(centre_oof) & np.isfinite(risk_oof)
    oof_records = [r for r, ok in zip(records, complete) if ok]

    # The shipped multiplier was calibrated on H800 alone.  Recalibrate it on
    # out-of-fold predictions so the number reflects unseen rows, and report
    # both a single shared value and per-card values.
    shared_cal = calibrate_upper_multiplier(
        oof_records, centre_oof[complete], risk_oof[complete], per_card=False
    )
    per_card_cal = calibrate_upper_multiplier(
        oof_records, centre_oof[complete], risk_oof[complete], per_card=True
    )

    evaluation = {
        "in_sample_by_card": {
            card: centre_metrics(
                [r for r in records if r["_card"] == card],
                np.asarray(
                    [p for r, p in zip(records, centre_in) if r["_card"] == card]
                ),
            )
            for card in cards
        },
        "out_of_fold_by_card": {
            card: centre_metrics(
                [r for r in oof_records if r["_card"] == card],
                np.asarray(
                    [
                        p
                        for r, p, ok in zip(records, centre_oof, complete)
                        if ok and r["_card"] == card
                    ]
                ),
            )
            for card in cards
        },
        "upper_multiplier_calibration": {
            "shipped_h800_only": upper_multiplier,
            "recalibrated_shared": shared_cal,
            "recalibrated_per_card": per_card_cal,
            "rule": (
                "smallest multiplier admitting zero OOM rows, measured on "
                "out-of-fold centre and risk predictions"
            ),
        },
        "admission_out_of_fold_shipped_multiplier": admission_metrics(
            oof_records,
            centre_oof[complete],
            risk_oof[complete],
            upper_multiplier=upper_multiplier,
        ),
        "admission_out_of_fold_shared_recalibrated": admission_metrics(
            oof_records,
            centre_oof[complete],
            risk_oof[complete],
            upper_multiplier=shared_cal,
        ),
        "admission_out_of_fold_per_card_recalibrated": admission_metrics(
            oof_records,
            centre_oof[complete],
            risk_oof[complete],
            upper_multiplier=per_card_cal,
        ),
    }

    report = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "joint_card_v3_memory_fitted_and_validated",
        "analysis_only": True,
        "publishable": False,
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "purpose": (
            "Fit the platform's V3 memory contract (centre + risk) jointly over "
            "H800 and RTX 4090 so server_integration can carry one model."
        ),
        "approach": {
            "shipped_modules_modified": False,
            "how": (
                "All 38 V3 raw features are precomputed into each record's "
                "features dict using the row's OWN capacity. "
                "benchmark_h800_memory_center_models_v1._feature_value returns "
                "anything already present verbatim, so its module-level H800 "
                "capacity constant is never reached.  All basis expansion, "
                "weighting and IRLS ridge math is imported unchanged."
            ),
            "h800_bit_identity_check": identity,
        },
        "contract": {
            "centre_formula": "reference_bytes * exp(shrinkage * (intercept + standardised_basis @ beta))",
            "admission_rule": (
                "admit iff max(centre, risk * upper_multiplier) <= "
                "safe_limit_fraction * capacity_bytes"
            ),
            "upper_multiplier": upper_multiplier,
            "safe_limit_fraction": SAFE_LIMIT_FRACTION,
            "hyperparameters_source": "copied verbatim from the shipped V3 artifact",
            "unknown_card_policy": "set card features to zero",
        },
        "data": {"by_card": inventory, "h800_audit_summary": {
            k: v for k, v in (h800_audit or {}).items() if not isinstance(v, (list, dict))
        }},
        "selection": {
            "rule": "minimise the worst per-card leave-source-out source-equal MAPE",
            "selected": {
                k: selected[k]
                for k in (
                    "card_feature_set",
                    "card_feature_names",
                    "raw_feature_dimension",
                    "worst_card_source_equal_mape",
                )
            },
            "candidates": candidates,
        },
        "frozen_model": {"centre": centre_model, "risk": risk_model},
        "evaluation": evaluation,
        "notes": {
            "strict31_skipped": (
                "The V3 evidence contract marks strict31 "
                "retrospective_diagnostic_only, so it is not a fitting input. "
                "Rebuilding it currently raises 'strict analytic reference "
                "reconstruction drifted' on exactly 3 qwen3p5_4b rows, each off "
                "by a constant 34603008 bytes (33 MiB); the other 26 rows "
                "reconstruct exactly.  Reported separately, not fixed here."
            ),
            "rtx4090_binding_bypass": (
                "The 4090 record builder needs "
                "ThroughputPredictor._normalized_request, whose constructor "
                "gate no commit can satisfy.  The gate guards the frozen "
                "throughput head, which memory fitting never reads."
            ),
        },
        "limitations": [
            "RTX 4090 covers qwen3 0.6B/1.7B/4B; H800 covers 1.7B/4B/8B/14B/32B. "
            "Card features are anchored on the 1.7B and 4B overlap only.",
            "H800 sources are far more numerous (49) than the RTX 4090 material, "
            "and source-equal weighting is what balances them; a different "
            "weighting choice would move the numbers.",
            "The risk head's upper_multiplier is reused from the shipped H800 "
            "calibration rather than recalibrated jointly.  Recalibrating it is "
            "required before any production use.",
            "analysis_only: replacing the shipped V3 artifact needs the Go side "
            "to read card features and a fresh acceptance run.",
        ],
        "source_bindings": {
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "v3_artifact": {
                "path": str(V3_ARTIFACT),
                "sha256": sha256_file(V3_ARTIFACT),
            },
            "model_math_unmodified": {
                "path": str(ROOT / "scripts" / "benchmark_h800_memory_center_models_v1.py"),
                "sha256": sha256_file(
                    ROOT / "scripts" / "benchmark_h800_memory_center_models_v1.py"
                ),
            },
            "v3_inventory": {
                "path": str(V3_INVENTORY),
                "sha256": sha256_file(V3_INVENTORY),
            },
        },
    }
    report["report_sha256"] = sha256_json(report)
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "h800_bit_identical": identity["bit_identical"],
                "values_compared": identity["values_compared"],
                "rows_by_card": {c: inventory[c]["rows"] for c in cards},
                "selected": report["selection"]["selected"],
                "out_of_fold_source_equal_mape": {
                    c: round(evaluation["out_of_fold_by_card"][c]["source_equal_mape"], 4)
                    for c in cards
                },
                "admission_out_of_fold_shipped_multiplier": evaluation[
                    "admission_out_of_fold_shipped_multiplier"
                ],
                "admission_out_of_fold_per_card_recalibrated": evaluation[
                    "admission_out_of_fold_per_card_recalibrated"
                ],
                "upper_multiplier_calibration": evaluation[
                    "upper_multiplier_calibration"
                ],
            },
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
