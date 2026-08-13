#!/usr/bin/env python3
"""CPU-only diagnostic: does a profile-derived sequence length beat cutoff_len?

The frozen memory basis sets ``sequence = cutoff_len`` for every activation and
logits term.  ``cutoff_len`` is a user-supplied ceiling, not an observed length,
so a dataset whose longest sample is far below the ceiling gets an activation and
logits budget it can never spend.  The 2026-08-03 fresh holdout mis-rejected a
genuinely safe 14B Full configuration on exactly that profile.

This script rebuilds the analytic anchor with ``sequence`` taken from the frozen
length profile instead, refits the reserved-center ridge on each variant, and
scores every variant with the same leave-native-scenario-out protocol the frozen
model used.  It writes nothing into the artifact directory and mutates no frozen
file: the production basis is called with a substituted job dict.

It is a diagnostic, not an acceptance run.  The holdout replays it reports were
already consumed by earlier campaigns.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any

from collections.abc import Mapping, Sequence

import numpy as np

from common import ARTIFACT_DIR, read_json
from h800_challenger_modeling import (
    _memory_features,
    _memory_feature_names,
    _observed_reserved,
    scenario_id,
)
from h800_native_memory_calibration import (
    build_native_record,
    native_admission_reason,
)
from h800_theory_basis import memory_basis

GIB = float(1 << 30)

# Sequence-length rules under test.  "cutoff" reproduces the frozen behaviour and
# is the control; every other rule reads the frozen length profile.
SEQUENCE_RULES = ("cutoff", "max", "p99", "batch_max")

# A (selector, stratum) cell needs this many rows before it earns its own
# quantile; otherwise the plain selector quantile is used.
MINIMUM_STRATUM_ROWS = 12


def _job(row: Mapping[str, Any]) -> dict[str, Any]:
    return dict(row["configuration"]["job"])


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (probability / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _expected_batch_max(clipped: Sequence[int], mbs: int) -> float:
    """E[max of ``mbs`` iid draws] from the empirical CDF of clipped lengths."""

    ordered = sorted(clipped)
    count = len(ordered)
    return sum(
        value * ((index / count) ** mbs - ((index - 1) / count) ** mbs)
        for index, value in enumerate(ordered, start=1)
    )


class ProfileLengths:
    """Frozen per-dataset token lengths, clipped on demand to a cutoff."""

    def __init__(self, lengths_by_dataset: Mapping[str, list[int]]) -> None:
        self._lengths = {key: list(value) for key, value in lengths_by_dataset.items()}
        self._cache: dict[tuple[str, int, int, str], float] = {}

    def datasets(self) -> list[str]:
        return sorted(self._lengths)

    def raw_lengths(self, dataset_id: str) -> list[int]:
        """Frozen, unclipped token lengths for one dataset.

        Exposed so callers can merge several campaigns' profiles into a single
        lookup without reaching into the private mapping.
        """
        return list(self._lengths[dataset_id])

    def has(self, dataset_id: str) -> bool:
        return dataset_id in self._lengths

    def sequence_for(
        self, dataset_id: str, *, cutoff_len: int, mbs: int, rule: str
    ) -> int:
        """Return the sequence length this rule assigns, never above cutoff."""

        if rule == "cutoff":
            return int(cutoff_len)
        key = (dataset_id, int(cutoff_len), int(mbs), rule)
        if key not in self._cache:
            clipped = [min(int(cutoff_len), value) for value in self._lengths[dataset_id]]
            if rule == "max":
                value = float(max(clipped))
            elif rule == "p99":
                value = _percentile(clipped, 99.0)
            elif rule == "batch_max":
                value = _expected_batch_max(clipped, int(mbs))
            else:
                raise ValueError(f"unknown sequence rule {rule!r}")
            self._cache[key] = value
        # Round up so the budget is never below an achievable padded length, and
        # clamp to cutoff so a rule can only ever reduce the frozen assumption.
        return int(min(int(cutoff_len), max(1, math.ceil(self._cache[key]))))


def load_profile_lengths(profile_dir: Path) -> ProfileLengths:
    """Read the per-dataset length profiles used by the frozen campaigns."""

    lengths: dict[str, list[int]] = {}
    for path in sorted(profile_dir.glob("*.jsonl")):
        dataset_id = path.name.split(".")[0]
        values: list[int] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    values.append(int(json.loads(line)["total_tokens"]))
        if values:
            lengths[dataset_id] = values
    if not lengths:
        raise ValueError(f"no length profiles under {profile_dir}")
    return ProfileLengths(lengths)


def _reanchor(
    record: dict[str, Any],
    *,
    rule: str,
    profiles: ProfileLengths,
    hardware: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Recompute a record's analytic anchor with ``rule`` choosing the sequence."""

    if rule == "cutoff":
        return record
    scenario = record["scenario"]
    dataset_id = str(scenario.get("dataset_id") or "")
    if not profiles.has(dataset_id):
        return None
    cutoff = int(scenario["cutoff_len"])
    mbs = int(scenario["physical_mbs"])
    sequence = profiles.sequence_for(
        dataset_id, cutoff_len=cutoff, mbs=mbs, rule=rule
    )
    # memory_basis reads the sequence length out of job["cutoff_len"], so a
    # substituted job dict changes the anchor without touching frozen code.
    substituted = {
        "gpu_count": scenario["gpu_count"],
        "mbs": mbs,
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
    # The 28-dim feature vector keeps the *requested* cutoff: the ridge must still
    # know the user asked for a wide ceiling, since padding policy depends on it.
    record["effective_sequence"] = {
        "rule": rule,
        "sequence_tokens": int(sequence),
        "cutoff_len": cutoff,
        "fraction_of_cutoff": sequence / float(cutoff),
    }
    return record


def rebuild_record_with_rule(
    row: Mapping[str, Any],
    *,
    rule: str,
    profiles: ProfileLengths,
    model_by_id: Mapping[str, dict[str, Any]],
    fixed_lora: dict[str, Any],
    hardware: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a native record whose analytic anchor uses ``rule`` for sequence."""

    record = build_native_record(
        row,
        model_by_id=model_by_id,
        fixed_lora=fixed_lora,
        hardware=hardware,
    )
    return _reanchor(record, rule=rule, profiles=profiles, hardware=hardware)


def fit_reserved_ridge(
    records: Sequence[Mapping[str, Any]],
    *,
    feature_set: str,
    alpha: float,
    historical_weight: float,
) -> dict[str, Any]:
    """Weighted ridge on log(observed / anchor); mirrors the frozen objective."""

    successes = [
        record
        for record in records
        if str(record.get("outcome") or "").lower() == "success"
        and _observed_reserved(record) is not None
    ]
    if not successes:
        raise ValueError("no success rows to fit")
    features = np.vstack(
        [_memory_features(record, feature_set) for record in successes]
    )
    targets = np.asarray(
        [
            math.log(
                float(_observed_reserved(record))
                / float(record["memory"]["analytic_reference_bytes"])
            )
            for record in successes
        ],
        dtype=float,
    )
    counts = Counter(scenario_id(record) for record in successes)
    weights = np.asarray(
        [
            (
                historical_weight
                if str(record.get("evidence_tier") or "").startswith("legacy")
                else 1.0
            )
            / counts[scenario_id(record)]
            for record in successes
        ],
        dtype=float,
    )
    means = np.average(features, axis=0, weights=weights)
    scales = np.sqrt(np.average((features - means) ** 2, axis=0, weights=weights))
    scales[scales < 1e-9] = 1.0
    standardized = (features - means) / scales
    design = np.column_stack((np.ones(len(standardized)), standardized))
    penalty = np.diag([0.0, *([float(alpha)] * standardized.shape[1])])
    normal = design.T @ (weights[:, None] * design) + penalty
    coefficients = np.linalg.pinv(normal) @ (design.T @ (weights * targets))
    return {
        "feature_set": feature_set,
        "feature_names": list(_memory_feature_names(feature_set)),
        "alpha": float(alpha),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "fit_success_rows": len(successes),
        "fit_scenarios": len(counts),
    }


def predict_center(record: Mapping[str, Any], model: Mapping[str, Any]) -> float:
    features = _memory_features(record, str(model["feature_set"]))
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    residual = float(model["intercept"]) + float(
        ((features - means) / scales) @ coefficients
    )
    return float(record["memory"]["analytic_reference_bytes"]) * math.exp(residual)


# Fraction of the analytic anchor that is fixed model state (parameters,
# gradients, optimizer, ZeRO-3 residency, collective buckets).  These terms do not
# move with cutoff or MBS, so a high share means the prediction is dominated by
# quantities the basis computes almost exactly, while a low share means activation
# and logits terms -- the uncertain part -- dominate.  Strictly in [0, 1].
FIXED_STATE_COMPONENTS = (
    "parameters_bytes",
    "gradients_bytes",
    "optimizer_bytes",
    "stage3_live_parameters_bytes",
    "zero_collective_workspace_bytes",
)

# Stratum edges declared before fitting.  Coarse on purpose: three strata keep
# every (selector, stratum) cell populated instead of slicing into noise.
FIXED_STATE_STRATA = ((0.85, "high"), (0.50, "mid"), (0.0, "low"))


def fixed_state_share(record: Mapping[str, Any]) -> float:
    """Share of the anchor made of cutoff/MBS-independent model state."""

    memory = record["memory"]
    components = memory["components"]
    reference = float(memory["analytic_reference_bytes"])
    total = sum(float(components[name]) for name in FIXED_STATE_COMPONENTS)
    share = total / reference
    return min(1.0, max(0.0, share))


def fixed_state_stratum(record: Mapping[str, Any]) -> str:
    share = fixed_state_share(record)
    for threshold, label in FIXED_STATE_STRATA:
        if share >= threshold:
            return label
    return "low"


def _selector_key(record: Mapping[str, Any]) -> str:
    selector = record["selector"]
    return json.dumps(
        [
            str(selector.get("training_mode")),
            int(selector.get("zero_stage") or 0),
            bool(selector.get("gradient_checkpointing")),
            bool(selector.get("packing")),
        ],
        separators=(",", ":"),
    )


def _stratified_key(record: Mapping[str, Any]) -> str:
    """Tail key including gpu_count.

    The plain selector key omits gpu_count, but per-rank memory behaviour differs
    sharply between 2 and 4 GPUs under ZeRO-3: for full/zero3/gc_on the 2-GPU rows
    sit at q95 x1.025 while the 4-GPU rows reach x1.264.  Pooling them charges every
    2-GPU candidate the 4-GPU tail risk, which is what mis-rejects genuinely safe
    2-GPU configurations.  Fixed-state stratum is kept as a secondary split.
    """

    scenario = record["scenario"]
    return (
        f"{_selector_key(record)}|g{int(scenario['gpu_count'])}"
        f"|{fixed_state_stratum(record)}"
    )


def fit_tail(
    records: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    *,
    coverage: float = 0.95,
) -> dict[str, Any]:
    """Conformal success-tail quantiles plus right-censored OOM lower guards."""

    success_residuals: dict[str, list[float]] = {}
    stratified_residuals: dict[str, list[float]] = {}
    pooled: list[float] = []
    for record in records:
        if str(record.get("outcome") or "").lower() != "success":
            continue
        observed = _observed_reserved(record)
        if observed is None:
            continue
        residual = math.log(float(observed) / predict_center(record, model))
        success_residuals.setdefault(_selector_key(record), []).append(residual)
        stratified_residuals.setdefault(
            _stratified_key(record), []
        ).append(residual)
        pooled.append(residual)

    def quantile(values: Sequence[float]) -> float:
        ordered = sorted(values)
        if not ordered:
            raise ValueError("empty residuals")
        rank = min(len(ordered) - 1, math.ceil(coverage * len(ordered)) - 1)
        return ordered[max(0, rank)]

    oom_guards: dict[str, list[float]] = {}
    for record in records:
        if str(record.get("outcome") or "").lower() != "oom":
            continue
        limit = float(record["memory"]["safe_limit_bytes"])
        residual = math.log(limit / predict_center(record, model))
        oom_guards.setdefault(_selector_key(record), []).append(residual)

    return {
        "coverage": coverage,
        "pooled_log_upper": quantile(pooled),
        "selector_log_upper": {
            key: quantile(values) for key, values in success_residuals.items()
        },
        "stratified_log_upper": {
            key: quantile(values)
            for key, values in stratified_residuals.items()
            if len(values) >= MINIMUM_STRATUM_ROWS
        },
        "stratified_rows": {
            key: len(values) for key, values in stratified_residuals.items()
        },
        "selector_oom_log_lower": {
            key: max(values) for key, values in oom_guards.items()
        },
    }


def predict_upper(
    record: Mapping[str, Any],
    model: Mapping[str, Any],
    tail: Mapping[str, Any],
    *,
    stratified: bool = False,
) -> float:
    center = predict_center(record, model)
    key = _selector_key(record)
    success_guard = None
    if stratified:
        # Prefer the (selector, fixed-state stratum) quantile.  Fall back to the
        # plain selector quantile rather than to pooled: a missing stratum means
        # too few rows to trust, not that the mechanism is benign.
        success_guard = tail.get("stratified_log_upper", {}).get(
            _stratified_key(record)
        )
    if success_guard is None:
        success_guard = tail["selector_log_upper"].get(key)
    if success_guard is None:
        success_guard = tail["pooled_log_upper"]
    oom_guard = tail["selector_oom_log_lower"].get(key, 0.0)
    return center * math.exp(max(0.0, float(success_guard), float(oom_guard)))


def leave_scenario_out(
    records: Sequence[Mapping[str, Any]],
    *,
    feature_set: str,
    alpha: float,
    historical_weight: float,
    stratified: bool = False,
) -> dict[str, Any]:
    """Hold out one native scenario at a time, dropping its legacy rows too."""

    native_scenarios = sorted(
        {
            scenario_id(record)
            for record in records
            if not str(record.get("evidence_tier") or "").startswith("legacy")
        }
    )
    absolute_percentage_errors: list[float] = []
    covered = 0
    success_rows = 0
    safe_success_rows = 0
    admitted_safe_success = 0
    false_safe_oom = 0
    oom_rows = 0
    per_row: list[dict[str, Any]] = []

    for held in native_scenarios:
        training = [record for record in records if scenario_id(record) != held]
        evaluation = [
            record
            for record in records
            if scenario_id(record) == held
            and not str(record.get("evidence_tier") or "").startswith("legacy")
        ]
        if not evaluation:
            continue
        model = fit_reserved_ridge(
            training,
            feature_set=feature_set,
            alpha=alpha,
            historical_weight=historical_weight,
        )
        tail = fit_tail(training, model)
        for record in evaluation:
            outcome = str(record.get("outcome") or "").lower()
            upper = predict_upper(record, model, tail, stratified=stratified)
            limit = float(record["memory"]["safe_limit_bytes"])
            admitted = upper <= limit
            row: dict[str, Any] = {
                "observation_id": record["observation_id"],
                "dataset_id": record["scenario"].get("dataset_id"),
                "model_id": record["scenario"].get("model_id"),
                "training_mode": record["selector"].get("training_mode"),
                "gpu_count": record["scenario"].get("gpu_count"),
                "physical_mbs": record["scenario"].get("physical_mbs"),
                "cutoff_len": record["scenario"].get("cutoff_len"),
                "outcome": outcome,
                "admitted": admitted,
                "upper_gib": upper / GIB,
                "safe_limit_gib": limit / GIB,
            }
            if "effective_sequence" in record:
                row["sequence_tokens"] = record["effective_sequence"]["sequence_tokens"]
            if outcome == "success":
                observed = _observed_reserved(record)
                if observed is None:
                    continue
                success_rows += 1
                center = predict_center(record, model)
                error = abs(center - float(observed)) / float(observed)
                absolute_percentage_errors.append(error)
                if upper >= float(observed):
                    covered += 1
                row["observed_gib"] = float(observed) / GIB
                row["center_gib"] = center / GIB
                row["absolute_percentage_error"] = error
                row["upper_covers_observed"] = upper >= float(observed)
                if float(observed) <= limit:
                    safe_success_rows += 1
                    if admitted:
                        admitted_safe_success += 1
                    row["actually_safe"] = True
                else:
                    row["actually_safe"] = False
            elif outcome == "oom":
                oom_rows += 1
                if admitted:
                    false_safe_oom += 1
            per_row.append(row)

    return {
        "native_scenarios": len(native_scenarios),
        "success_rows": success_rows,
        "oom_rows": oom_rows,
        "false_safe_oom": false_safe_oom,
        "success_upper_coverage": covered / success_rows if success_rows else None,
        "actual_safe_success_rows": safe_success_rows,
        "admitted_safe_success_rows": admitted_safe_success,
        "admission_recall": (
            admitted_safe_success / safe_success_rows if safe_success_rows else None
        ),
        "center_absolute_percentage_error": {
            "count": len(absolute_percentage_errors),
            "mean": statistics.fmean(absolute_percentage_errors)
            if absolute_percentage_errors
            else None,
            "median": statistics.median(absolute_percentage_errors)
            if absolute_percentage_errors
            else None,
            "p90": _percentile(absolute_percentage_errors, 90.0)
            if absolute_percentage_errors
            else None,
            "max": max(absolute_percentage_errors)
            if absolute_percentage_errors
            else None,
        },
        "rows": per_row,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        type=Path,
        default=ARTIFACT_DIR / "canonical_h800_observations.jsonl",
    )
    parser.add_argument(
        "--profiles", type=Path, default=ARTIFACT_DIR / "dataset_profiles"
    )
    parser.add_argument(
        "--inventory", type=Path, default=ARTIFACT_DIR / "model_inventory.json"
    )
    parser.add_argument(
        "--theory-basis", type=Path, default=ARTIFACT_DIR / "h800_theory_basis.json"
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "hardware.json",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--historical-weight", type=float, default=0.25)
    parser.add_argument("--feature-set", default="physical_shares")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    hardware = read_json(args.hardware)
    inventory = read_json(args.inventory)
    models = inventory.get("models") or []
    model_by_id = {str(entry["id"]): dict(entry) for entry in models}
    fixed_lora = dict(inventory.get("fixed_lora") or {})
    profiles = load_profile_lengths(args.profiles)

    rows: list[dict[str, Any]] = []
    with args.observations.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    admitted = [row for row in rows if native_admission_reason(row) == "admitted"]

    # The frozen fit augments native calibration rows with legacy theory-basis
    # rows at weight 0.25.  Reproducing that mix is what makes the "cutoff"
    # control comparable to the frozen leave-scenario-out numbers.
    theory_basis = read_json(args.theory_basis)
    legacy_rows = [
        record
        for record in theory_basis.get("records") or []
        if isinstance(record, Mapping)
        and (record.get("route") or {}).get("memory_boundary") is True
        and str(
            record["outcome"].get("class")
            if isinstance(record.get("outcome"), Mapping)
            else record.get("outcome")
        ).lower()
        in {"success", "oom"}
    ]

    summary: dict[str, Any] = {
        "observation_rows": len(rows),
        "admitted_rows": len(admitted),
        "legacy_memory_rows": len(legacy_rows),
        "profile_datasets": profiles.datasets(),
        "evaluation_kind": "cpu_only_diagnostic_not_acceptance",
        "variants": {},
    }

    for rule in SEQUENCE_RULES:
        records: list[dict[str, Any]] = []
        skipped = 0
        for row in admitted:
            record = rebuild_record_with_rule(
                row,
                rule=rule,
                profiles=profiles,
                model_by_id=model_by_id,
                fixed_lora=fixed_lora,
                hardware=hardware,
            )
            if record is None:
                skipped += 1
                continue
            records.append(record)
        for legacy in legacy_rows:
            record = _reanchor(
                dict(legacy), rule=rule, profiles=profiles, hardware=hardware
            )
            if record is None:
                skipped += 1
                continue
            records.append(record)
        fractions = [
            record["effective_sequence"]["fraction_of_cutoff"]
            for record in records
            if "effective_sequence" in record
        ]
        result = leave_scenario_out(
            records,
            feature_set=args.feature_set,
            alpha=args.alpha,
            historical_weight=args.historical_weight,
        )
        result["records"] = len(records)
        result["skipped_missing_profile"] = skipped
        result["sequence_fraction_of_cutoff"] = (
            {
                "min": min(fractions),
                "median": statistics.median(fractions),
                "max": max(fractions),
            }
            if fractions
            else None
        )
        summary["variants"][rule] = result
        coverage = result["success_upper_coverage"]
        recall = result["admission_recall"]
        mape = result["center_absolute_percentage_error"]["mean"]
        print(
            f"{rule:>10}  rows={result['records']:4d}  "
            f"MAPE={mape:6.2%}  coverage={coverage:6.2%}  "
            f"recall={recall:6.2%}  false_safe_oom={result['false_safe_oom']}"
        )

    if args.output is not None:
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
