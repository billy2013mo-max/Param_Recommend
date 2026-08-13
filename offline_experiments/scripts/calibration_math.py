#!/usr/bin/env python3
"""Deterministic numerical primitives for physical calibration.

The helpers in this module deliberately expose only bounded linear fitting and
distribution-free/censor-aware tail calculations.  They do not depend on
SciPy, sklearn, CUDA, model identifiers, or dataset identifiers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class BoundedFitResult:
    coefficients: tuple[float, ...]
    iterations: int
    converged: bool
    weighted_huber_loss: float
    residual_mad: float


@dataclass(frozen=True)
class QuantileResult:
    value: float | None
    rank: int
    observations: int
    identifiable: bool


def _finite_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return array


def scenario_equal_weights(scenario_ids: Sequence[str]) -> np.ndarray:
    """Return weights giving every scenario total mass one.

    The returned weights are normalized to mean one so solver tolerances do not
    change merely because a caller groups the same observations differently.
    """

    if not scenario_ids:
        raise ValueError("At least one scenario id is required")
    counts: dict[str, int] = {}
    for scenario_id in scenario_ids:
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("Scenario ids must be non-empty strings")
        counts[scenario_id] = counts.get(scenario_id, 0) + 1
    weights = np.asarray([1.0 / counts[value] for value in scenario_ids])
    return weights / float(np.mean(weights))


def bounded_huber_fit(
    matrix: Any,
    target: Any,
    lower: Any,
    upper: Any,
    *,
    sample_weight: Any | None = None,
    initial: Any | None = None,
    ridge: Any | None = None,
    ridge_prior: Any | None = None,
    huber_delta: float = 1.345,
    max_iterations: int = 80,
    tolerance: float = 1e-9,
) -> BoundedFitResult:
    """Fit a box-constrained robust linear model by deterministic IRLS/CD.

    Every coefficient has an explicit finite physical bound.  ``ridge`` is an
    optional per-coefficient non-negative penalty toward ``ridge_prior`` and is
    useful for shrinking runtime-cohort nuisance effects without turning them
    into reusable planner coefficients.
    """

    design = np.asarray(matrix, dtype=np.float64)
    if design.ndim != 2 or not np.all(np.isfinite(design)):
        raise ValueError("matrix must be a finite two-dimensional array")
    response = _finite_vector(target, "target")
    if design.shape[0] != response.size or response.size == 0:
        raise ValueError("matrix and target must have the same non-zero row count")
    coefficient_count = design.shape[1]
    if coefficient_count == 0:
        raise ValueError("matrix must contain at least one coefficient column")
    lower_bound = _finite_vector(lower, "lower")
    upper_bound = _finite_vector(upper, "upper")
    if lower_bound.size != coefficient_count or upper_bound.size != coefficient_count:
        raise ValueError("coefficient bounds do not match matrix columns")
    if np.any(lower_bound > upper_bound):
        raise ValueError("lower coefficient bounds cannot exceed upper bounds")
    if not math.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("huber_delta must be finite and positive")
    if max_iterations <= 0 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("invalid solver iteration controls")

    weights = (
        np.ones(response.size, dtype=np.float64)
        if sample_weight is None
        else _finite_vector(sample_weight, "sample_weight")
    )
    if weights.size != response.size or np.any(weights <= 0):
        raise ValueError("sample weights must be positive and match target rows")
    weights = weights / float(np.mean(weights))
    penalties = (
        np.zeros(coefficient_count, dtype=np.float64)
        if ridge is None
        else _finite_vector(ridge, "ridge")
    )
    if penalties.size != coefficient_count or np.any(penalties < 0):
        raise ValueError("ridge penalties must be non-negative and match columns")
    priors = (
        np.zeros(coefficient_count, dtype=np.float64)
        if ridge_prior is None
        else _finite_vector(ridge_prior, "ridge_prior")
    )
    if priors.size != coefficient_count:
        raise ValueError("ridge priors must match matrix columns")
    priors = np.clip(priors, lower_bound, upper_bound)
    coefficients = (
        np.clip(priors, lower_bound, upper_bound)
        if initial is None
        else np.clip(_finite_vector(initial, "initial"), lower_bound, upper_bound)
    )
    if coefficients.size != coefficient_count:
        raise ValueError("initial coefficients must match matrix columns")

    converged = False
    scale = 0.0
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        previous = coefficients.copy()
        residual = response - design @ coefficients
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        numeric_floor = max(1e-12, float(np.max(np.abs(response))) * 1e-12)
        scale = max(1.4826 * mad, numeric_floor)
        standardized = np.abs(residual) / scale
        robust = np.ones_like(standardized)
        outliers = standardized > huber_delta
        robust[outliers] = huber_delta / standardized[outliers]
        effective_weight = weights * robust

        for column_index in range(coefficient_count):
            column = design[:, column_index]
            old_value = coefficients[column_index]
            partial = response - design @ coefficients + column * old_value
            denominator = float(
                np.sum(effective_weight * column * column)
                + penalties[column_index]
            )
            if denominator <= 0:
                candidate = priors[column_index]
            else:
                numerator = float(
                    np.sum(effective_weight * column * partial)
                    + penalties[column_index] * priors[column_index]
                )
                candidate = numerator / denominator
            coefficients[column_index] = float(
                np.clip(candidate, lower_bound[column_index], upper_bound[column_index])
            )

        maximum_change = float(np.max(np.abs(coefficients - previous)))
        scale_of_solution = max(1.0, float(np.max(np.abs(previous))))
        if maximum_change <= tolerance * scale_of_solution:
            converged = True
            break

    residual = response - design @ coefficients
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    loss_scale = max(scale, 1e-12)
    absolute = np.abs(residual) / loss_scale
    huber = np.where(
        absolute <= huber_delta,
        0.5 * absolute * absolute,
        huber_delta * (absolute - 0.5 * huber_delta),
    )
    loss = float(
        np.sum(weights * huber)
        + 0.5 * np.sum(penalties * (coefficients - priors) ** 2)
    )
    if not np.all(np.isfinite(coefficients)) or not math.isfinite(loss):
        raise ValueError("bounded fit produced non-finite output")
    return BoundedFitResult(
        coefficients=tuple(float(value) for value in coefficients),
        iterations=iteration,
        converged=converged,
        weighted_huber_loss=loss,
        residual_mad=mad,
    )


def finite_sample_upper_quantile(
    values: Iterable[float], coverage: float = 0.95
) -> QuantileResult:
    """Return the one-sided split-conformal order statistic.

    For coverage ``p``, the rank is ``ceil((n+1)*p)``.  If that rank exceeds
    ``n`` there is no finite distribution-free upper bound, which is reported
    explicitly instead of silently returning the sample maximum.
    """

    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("conformal values must be finite")
    if not math.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("coverage must be in (0,1)")
    rank = math.ceil((len(ordered) + 1) * coverage)
    identifiable = bool(ordered) and rank <= len(ordered)
    return QuantileResult(
        value=ordered[rank - 1] if identifiable else None,
        rank=rank,
        observations=len(ordered),
        identifiable=identifiable,
    )


def finite_sample_lower_quantile(
    values: Iterable[float], coverage: float = 0.95
) -> QuantileResult:
    """Return the finite-sample one-sided lower order statistic.

    A 95% lower bound uses rank ``floor((n+1)*0.05)``.  When the rank is zero
    the distribution-free lower endpoint is unbounded, so the function reports
    it as non-identifiable rather than substituting the sample minimum.
    """

    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("conformal values must be finite")
    if not math.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("coverage must be in (0,1)")
    rank = math.floor((len(ordered) + 1) * (1 - coverage))
    identifiable = bool(ordered) and rank >= 1
    return QuantileResult(
        value=ordered[rank - 1] if identifiable else None,
        rank=rank,
        observations=len(ordered),
        identifiable=identifiable,
    )


def bounded_isotonic_fit(
    values: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    lower: float = 0.0,
    upper: float = 1.0,
    increasing: bool = True,
) -> tuple[float, ...]:
    """Weighted PAVA with a shared finite box constraint.

    Inputs must already be ordered by the physical control variable (for
    example increasing MBS).  The result is deterministic and contains no
    interpolation or model/dataset-dependent feature.
    """

    observations = _finite_vector(values, "isotonic values")
    if observations.size == 0:
        raise ValueError("isotonic fit requires at least one value")
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("isotonic bounds must be finite and ordered")
    masses = (
        np.ones(observations.size, dtype=np.float64)
        if weights is None
        else _finite_vector(weights, "isotonic weights")
    )
    if masses.size != observations.size or np.any(masses <= 0):
        raise ValueError("isotonic weights must be positive and match values")
    if not increasing:
        reversed_fit = bounded_isotonic_fit(
            observations[::-1],
            weights=masses[::-1],
            lower=lower,
            upper=upper,
            increasing=True,
        )
        return tuple(reversed(reversed_fit))

    blocks: list[dict[str, float | int]] = []
    for index, (value, mass) in enumerate(zip(observations, masses)):
        blocks.append(
            {
                "start": index,
                "end": index + 1,
                "mass": float(mass),
                "weighted_sum": float(value * mass),
                "level": float(np.clip(value, lower, upper)),
            }
        )
        while len(blocks) >= 2 and float(blocks[-2]["level"]) > float(
            blocks[-1]["level"]
        ):
            right = blocks.pop()
            left = blocks.pop()
            merged_mass = float(left["mass"]) + float(right["mass"])
            merged_sum = float(left["weighted_sum"]) + float(
                right["weighted_sum"]
            )
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "mass": merged_mass,
                    "weighted_sum": merged_sum,
                    "level": float(
                        np.clip(merged_sum / merged_mass, lower, upper)
                    ),
                }
            )
    fitted = np.empty(observations.size, dtype=np.float64)
    for block in blocks:
        fitted[int(block["start"]) : int(block["end"])] = float(block["level"])
    return tuple(float(value) for value in fitted)


def kaplan_meier_quantile(
    values: Sequence[float],
    events: Sequence[bool],
    probability: float = 0.95,
) -> QuantileResult:
    """Estimate a quantile with exact events and right-censored observations.

    Exact events are processed before censoring at the same value.  If the
    estimated survival curve never reaches ``1-probability``, the requested
    tail is not identifiable and ``value`` remains ``None``.
    """

    if len(values) != len(events) or not values:
        raise ValueError("KM values/events must have the same non-zero length")
    if not math.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("probability must be in (0,1)")
    observations: list[tuple[float, bool]] = []
    for value, event in zip(values, events):
        number = float(value)
        if not math.isfinite(number) or not isinstance(event, (bool, np.bool_)):
            raise ValueError("KM inputs must contain finite values and booleans")
        observations.append((number, bool(event)))
    grouped: dict[float, list[int]] = {}
    for value, event in observations:
        counts = grouped.setdefault(value, [0, 0])
        counts[0 if event else 1] += 1

    at_risk = len(observations)
    survival = 1.0
    event_rank = 0
    threshold = 1 - probability
    for value in sorted(grouped):
        event_count, censor_count = grouped[value]
        if event_count:
            survival *= 1 - event_count / at_risk
            event_rank += event_count
            if survival <= threshold + 1e-15:
                return QuantileResult(
                    value=value,
                    rank=event_rank,
                    observations=len(observations),
                    identifiable=True,
                )
        at_risk -= event_count + censor_count
    return QuantileResult(
        value=None,
        rank=event_rank,
        observations=len(observations),
        identifiable=False,
    )


def global_scenario_folds(
    rows: Sequence[dict[str, Any]],
    *,
    scenario_field: str = "scenario_id",
    observation_field: str = "observation_id",
) -> list[dict[str, Any]]:
    """Build deterministic global leave-one-scenario-out memberships."""

    if not rows:
        raise ValueError("Cannot build folds without rows")
    by_scenario: dict[str, list[str]] = {}
    all_ids: set[str] = set()
    for row in rows:
        scenario_id = row.get(scenario_field)
        observation_id = row.get(observation_field)
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("Every row must have a non-empty scenario id")
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("Every row must have a non-empty observation id")
        if observation_id in all_ids:
            raise ValueError(f"Duplicate observation id {observation_id}")
        all_ids.add(observation_id)
        by_scenario.setdefault(scenario_id, []).append(observation_id)
    folds = []
    test_counts: dict[str, int] = {}
    for scenario_id in sorted(by_scenario):
        test_ids = sorted(by_scenario[scenario_id])
        train_ids = sorted(all_ids - set(test_ids))
        if set(train_ids).intersection(test_ids):
            raise ValueError("Global scenario fold membership overlaps")
        for observation_id in test_ids:
            test_counts[observation_id] = test_counts.get(observation_id, 0) + 1
        material = {
            "held_out_scenario_id": scenario_id,
            "train_observation_ids_sha256": _sha256_strings(train_ids),
            "test_observation_ids_sha256": _sha256_strings(test_ids),
        }
        folds.append(
            {
                "fold_id": _sha256_mapping(material),
                **material,
                "train_observation_ids": train_ids,
                "test_observation_ids": test_ids,
            }
        )
    if set(test_counts) != all_ids or any(count != 1 for count in test_counts.values()):
        raise ValueError("Every observation must be test data exactly once")
    return folds


def _sha256_strings(values: Sequence[str]) -> str:
    import hashlib
    import json

    payload = json.dumps(
        list(values), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256_mapping(value: dict[str, Any]) -> str:
    import hashlib
    import json

    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()
