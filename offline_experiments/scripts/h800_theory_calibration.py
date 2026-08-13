#!/usr/bin/env python3
"""CPU-only, non-publishable calibration checks for the H800 theory basis.

This module deliberately does *not* produce a planner profile.  It exercises a
small set of physically bounded parameters against historical observations and
reports grouped, out-of-scenario validation metrics.  In particular:

* the scientific split unit is ``(model, mode, dataset, target GBS)`` globally,
  not separately inside each runtime cohort;
* screen-only throughput runs are diagnostic and never enter a fit;
* only compute efficiency and activation liveness are fitted physical terms;
* bandwidth, collective and latency terms must be supplied as explicit priors;
* OOM observations are lower inequalities, never imputed peak measurements;
* every result is marked ``bootstrap/theory_only/nonpublishable``.

The implementation uses only the Python standard library so it can run in the
CPU-only recommendation service and in the offline experiment test suite.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

try:  # Package import in application code.
    from .calibration_math import (
        bounded_isotonic_fit,
        bounded_huber_fit,
        finite_sample_lower_quantile,
        finite_sample_upper_quantile,
        global_scenario_folds,
        kaplan_meier_quantile,
        scenario_equal_weights,
    )
except ImportError:  # Direct scripts/ import in the offline test suite.
    from calibration_math import (  # type: ignore
        bounded_isotonic_fit,
        bounded_huber_fit,
        finite_sample_lower_quantile,
        finite_sample_upper_quantile,
        global_scenario_folds,
        kaplan_meier_quantile,
        scenario_equal_weights,
    )


SCHEMA = "sft_h800_theory_calibration/v1"
STATUS = "theory_only"
CONFIDENCE = "bootstrap"
PUBLICATION = "nonpublishable"

# Bumped whenever the fitting/validation semantics change, so a stored report
# can be tied to the exact implementation that produced it.  Reported as an
# implementation hash rather than trusted prose.
IMPLEMENTATION_VERSION = "sft_h800_theory_calibration_impl/2026-07-22.inner-oof-joint-gate"
EXPECTED_BASIS_SCHEMA = "sft_h800_theory_basis/v1"

THROUGHPUT_PRIMARY = "throughput_primary"
THROUGHPUT_SCREEN = "throughput_screen_only"
MEMORY_BOUNDARY = "memory_boundary"
FEASIBILITY = "feasibility"
SPECIAL_PURPOSE_ROUTES = (
    "profiler",
    "packing_pair",
    "packing_memory_safety",
)

MEMORY_SELECTOR_MIN_SUCCESS = 19
DEFAULT_ALPHA = 0.05
COMPUTE_EFFICIENCY_BOUNDS = (1e-6, 1.0)
ACTIVATION_LIVENESS_BOUNDS = (0.25, 4.0)

# Inner cross-validation depth for out-of-fold conformal residuals.
#
# The memory-safety tail uses strict leave-one-scenario-out (``None``): under the
# "avoid OOM first" policy it is the safety-critical path, it carries no free
# grouping parameter, and it was measured to yield *fewer* false-safe held-out
# OOMs than grouped folds on the real H800 data (2 vs 3).  The throughput tail
# is not safety-critical (it only drives regret/scaling) and its center fit is
# ~4x more expensive, so it is capped at ``THROUGHPUT_INNER_OOF_FOLDS`` grouped
# scenario folds to keep the offline run tractable.
THROUGHPUT_INNER_OOF_FOLDS = 5
MEMORY_INNER_OOF_FOLDS = None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _positive(value: Any) -> float | None:
    result = _finite(value)
    return result if result is not None and result > 0 else None


def _nonnegative(value: Any) -> float | None:
    result = _finite(value)
    return result if result is not None and result >= 0 else None


def _integer(value: Any) -> int | None:
    result = _finite(value)
    if result is None or not result.is_integer():
        return None
    return int(result)


def _median(values: Iterable[float]) -> float | None:
    usable = [value for value in values if _finite(value) is not None]
    return float(statistics.median(usable)) if usable else None


def _mean(values: Iterable[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.fmean(usable) if usable else None


def _percentile(values: Iterable[float], probability: float) -> float | None:
    usable = sorted(float(value) for value in values if math.isfinite(value))
    if not usable:
        return None
    probability = min(1.0, max(0.0, probability))
    position = probability * (len(usable) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return usable[lower]
    fraction = position - lower
    return usable[lower] * (1 - fraction) + usable[upper] * fraction


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_basis_integrity(
    basis_report: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Independently recompute the basis digest and check its schema.

    The calibrator does not trust the ``report_sha256`` string the basis carries;
    it recomputes the canonical digest over the basis content and compares.  A
    schema or digest mismatch is a fail-closed blocker recorded in the report,
    not a silent acceptance.
    """

    schema = basis_report.get("schema")
    bound = basis_report.get("report_sha256")
    content = {key: value for key, value in basis_report.items() if key != "report_sha256"}
    recomputed = _canonical_sha256(content)
    schema_ok = schema == EXPECTED_BASIS_SCHEMA
    digest_ok = isinstance(bound, str) and bound == recomputed
    blockers: list[str] = []
    if not schema_ok:
        blockers.append("basis_schema_unexpected")
    if not digest_ok:
        blockers.append("basis_report_sha256_mismatch")
    return {
        "expected_schema": EXPECTED_BASIS_SCHEMA,
        "observed_schema": schema,
        "schema_ok": schema_ok,
        "bound_sha256": bound,
        "recomputed_sha256": recomputed,
        "digest_ok": digest_ok,
        "validated_before_fit": True,
    }, blockers


def _contains_non_finite(value: Any) -> bool:
    """Recursively detect NaN/Inf floats anywhere in a report payload."""

    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_contains_non_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_non_finite(item) for item in value)
    return False


def _observation_id(record: Mapping[str, Any], index: int | None = None) -> str:
    value = record.get("observation_id") or record.get("source_observation_id")
    if isinstance(value, str) and value:
        return value
    material = {
        "job_id": record.get("job_id"),
        "scenario": record.get("scenario"),
        "selector": record.get("selector"),
        "index": index,
    }
    return "anonymous-" + _canonical_sha256(material)


def _route_names(record: Mapping[str, Any]) -> set[str]:
    routes: set[str] = set()
    raw = record.get("route")
    if isinstance(raw, str) and raw:
        routes.add(raw)
    elif isinstance(raw, Mapping):
        routes.update(str(key) for key, enabled in raw.items() if enabled is True)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        routes.update(str(value) for value in raw if isinstance(value, str))
    eligibility = record.get("measurement_eligibility")
    if isinstance(eligibility, Mapping):
        routes.update(
            str(key) for key, enabled in eligibility.items() if enabled is True
        )
    return routes


def _has_route(record: Mapping[str, Any], route: str) -> bool:
    return route in _route_names(record)


def _outcome(record: Mapping[str, Any]) -> str:
    raw = record.get("outcome")
    if isinstance(raw, Mapping):
        raw = raw.get("class")
    return str(raw or "").strip().lower()


def _selector(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = record.get("selector")
    return dict(raw) if isinstance(raw, Mapping) else {}


def _scenario(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = record.get("scenario")
    return dict(raw) if isinstance(raw, Mapping) else {}


def scenario_material(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the global scientific split unit for one theory-basis record."""

    scenario = _scenario(record)
    selector = _selector(record)
    return {
        "model_id": scenario.get("model_id"),
        "training_mode": selector.get("training_mode")
        or scenario.get("training_mode")
        or scenario.get("train_type"),
        "dataset_id": scenario.get("dataset_id"),
        "target_gbs": scenario.get("target_gbs"),
    }


def scenario_id(record: Mapping[str, Any]) -> str:
    return _canonical_sha256(scenario_material(record))


def _runtime_cohort(record: Mapping[str, Any]) -> str:
    selector = _selector(record)
    value = selector.get("runtime_cohort_id")
    if not value and isinstance(record.get("runtime"), Mapping):
        value = record["runtime"].get("runtime_cohort_id")
    return str(value or "__missing_runtime_cohort__")


def _evidence_tier(record: Mapping[str, Any]) -> str:
    value = record.get("evidence_tier")
    return str(value or "__missing_evidence_tier__")


def _is_verified_anchor_tier(tier: str) -> bool:
    """A verified publication anchor is any non-legacy, present evidence tier.

    Every ``legacy_*`` tier (including ``legacy_verified``) is explicitly *not*
    a publication anchor per the phased data-admission policy, so a dataset made
    only of legacy tiers has no verified anchor and must stay blocked.
    """

    return (
        isinstance(tier, str)
        and bool(tier)
        and tier != "__missing_evidence_tier__"
        and not tier.startswith("legacy")
    )


def _zero_stage(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    text = str(value or "").strip().lower().replace("zero", "")
    try:
        return int(text)
    except ValueError:
        return 0


def _mode_gc_key(record: Mapping[str, Any]) -> tuple[str, bool]:
    selector = _selector(record)
    return (
        str(selector.get("training_mode") or "unknown").lower(),
        bool(selector.get("gradient_checkpointing")),
    )


def _memory_selector_key(record: Mapping[str, Any]) -> tuple[str, int, bool, bool]:
    selector = _selector(record)
    return (
        str(selector.get("training_mode") or "unknown").lower(),
        _zero_stage(selector.get("zero_stage")),
        bool(selector.get("gradient_checkpointing")),
        bool(selector.get("packing")),
    )


def _selector_json(key: tuple[str, int, bool, bool]) -> dict[str, Any]:
    return {
        "training_mode": key[0],
        "zero_stage": key[1],
        "gradient_checkpointing": key[2],
        "packing": key[3],
    }


def _gpu_count(record: Mapping[str, Any]) -> int | None:
    scenario = _scenario(record)
    selector = _selector(record)
    for value in (
        scenario.get("gpu_count"),
        selector.get("gpu_count"),
        record.get("gpu_count"),
    ):
        result = _integer(value)
        if result is not None and result > 0:
            return result
    return None


def _physical_mbs(record: Mapping[str, Any]) -> int | None:
    scenario = _scenario(record)
    performance = record.get("performance")
    performance = performance if isinstance(performance, Mapping) else {}
    for value in (
        scenario.get("physical_mbs"),
        scenario.get("mbs"),
        performance.get("physical_mbs"),
    ):
        result = _integer(value)
        if result is not None and result > 0:
            return result
    return None


def build_global_scenario_folds(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build globally held-out scenario folds across every runtime cohort.

    The returned membership lists make the no-leakage property auditable in
    tests and downstream reports.  Route-specific fitting happens later; the
    fold itself is deliberately route-agnostic.
    """

    indexed = [
        {
            "observation_id": _observation_id(record, index),
            "scenario_id": scenario_id(record),
            "record": record,
        }
        for index, record in enumerate(records)
    ]
    materials = {
        item["scenario_id"]: scenario_material(item["record"]) for item in indexed
    }
    by_id = {item["observation_id"]: item["record"] for item in indexed}
    folds = global_scenario_folds(indexed)
    for fold in folds:
        fold["held_out_scenario"] = materials[fold["held_out_scenario_id"]]
        fold["train_runtime_cohorts"] = sorted(
            {_runtime_cohort(by_id[value]) for value in fold["train_observation_ids"]}
        )
        fold["test_runtime_cohorts"] = sorted(
            {_runtime_cohort(by_id[value]) for value in fold["test_observation_ids"]}
        )
    return folds


def bounded_isotonic_by_mbs(
    values: Mapping[int, float],
    *,
    weights: Mapping[int, float] | None = None,
    lower: float = COMPUTE_EFFICIENCY_BOUNDS[0],
    upper: float = COMPUTE_EFFICIENCY_BOUNDS[1],
) -> dict[int, float]:
    """Weighted PAVA projection, clipped to physical bounds.

    Extrapolation is handled by :func:`_efficiency_for_mbs`: the largest fitted
    MBS value is held constant, which is the explicit saturation constraint.
    """

    if not (0 < lower <= upper <= 1):
        raise ValueError("compute efficiency bounds must satisfy 0 < lower <= upper <= 1")
    points = sorted((int(mbs), float(value)) for mbs, value in values.items())
    if not points:
        return {}
    fitted = bounded_isotonic_fit(
        [value for _mbs, value in points],
        weights=[float((weights or {}).get(mbs, 1.0)) for mbs, _value in points],
        lower=lower,
        upper=upper,
        increasing=True,
    )
    return {mbs: value for (mbs, _raw), value in zip(points, fitted)}


def one_sided_conformal(
    scores: Iterable[float],
    *,
    alpha: float = DEFAULT_ALPHA,
    side: str = "upper",
) -> dict[str, Any]:
    """Finite-sample one-sided conformal order statistic.

    For an upper bound this uses rank ``ceil((n+1)*(1-alpha))``.  For a lower
    bound it uses ``floor((n+1)*alpha)``.  Ranks are clipped to the available
    sample range, making sparse bootstrap behavior explicit rather than silently
    interpolating a population quantile.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    usable = sorted(float(score) for score in scores if _finite(score) is not None)
    if side == "upper":
        result = finite_sample_upper_quantile(usable, coverage=1 - alpha)
    elif side == "lower":
        result = finite_sample_lower_quantile(usable, coverage=1 - alpha)
    else:
        raise ValueError("side must be 'upper' or 'lower'")
    return {
        "available": result.identifiable,
        "side": side,
        "alpha": alpha,
        "sample_count": result.observations,
        "rank": result.rank,
        "quantile": result.value,
    }


def _assign_inner_folds(
    scenario_ids: Sequence[str], folds: int | None
) -> list[frozenset[str]]:
    """Partition scenarios into deterministic inner cross-validation folds.

    ``folds is None`` (or a bound no smaller than the scenario count) yields
    leave-one-scenario-out.  Otherwise scenarios are distributed round-robin in
    sorted order so every inner training set omits whole scenarios and the
    partition never depends on record ordering.
    """

    unique = sorted(set(scenario_ids))
    if not unique:
        return []
    if folds is None or folds >= len(unique):
        return [frozenset((scenario,)) for scenario in unique]
    if folds < 2:
        raise ValueError("inner folds must be None or at least 2")
    buckets: list[set[str]] = [set() for _ in range(folds)]
    for index, scenario in enumerate(unique):
        buckets[index % folds].add(scenario)
    return [frozenset(bucket) for bucket in buckets if bucket]


def _inner_scenario_oof(
    route_records: Sequence[Mapping[str, Any]],
    outer_excluded_scenarios: frozenset[str],
    *,
    fit_fn: Any,
    collect_fn: Any,
    fit_cache: dict[frozenset[str], Any],
    folds: int | None,
) -> list[Any]:
    """Collect inner leave-scenario-out out-of-fold residuals.

    ``route_records`` is the full route population (for example every primary
    throughput row).  ``outer_excluded_scenarios`` are the scenarios already
    held out by the enclosing outer fold.  Each inner training set additionally
    removes a held-out scenario group, so no residual is ever produced by a
    center that trained on that residual's own scenario or on the outer test
    scenario.  Centers are memoized by the frozenset of excluded scenarios so
    the shared two-out fits are computed only once across the whole run.
    """

    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in route_records:
        scenario = scenario_id(record)
        if scenario in outer_excluded_scenarios:
            continue
        by_scenario[scenario].append(record)
    collected: list[Any] = []
    for group in _assign_inner_folds(sorted(by_scenario), folds):
        excluded = outer_excluded_scenarios | group
        model = fit_cache.get(excluded)
        if model is None:
            train = [
                record
                for record in route_records
                if scenario_id(record) not in excluded
            ]
            model = fit_fn(train)
            fit_cache[excluded] = model
        test = [record for scenario in group for record in by_scenario[scenario]]
        collected.extend(collect_fn(model, test))
    return collected


_PRIOR_FIELDS = (
    "dense_peak_flops_per_s",
    "memory_bandwidth_bytes_per_s",
    "collective_bandwidth_bytes_per_s",
    "hbm_efficiency",
    "optimizer_hbm_efficiency",
    "collective_efficiency",
    "collective_latency_seconds",
    "microstep_latency_seconds",
    "framework_latency_seconds",
    "communication_overlap_by_stage",
)


def resolve_physical_priors(
    basis_report: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve explicit priors without replacing ``null`` by fitted values."""

    raw = basis_report.get("physical_priors")
    if isinstance(raw, Mapping) and isinstance(raw.get("values"), Mapping):
        raw = raw["values"]
    values = dict(raw) if isinstance(raw, Mapping) else {}
    sources = {key: "theory_basis" for key, value in values.items() if value is not None}

    # sft_h800_theory_basis/v1 binds identical hardware/efficiency priors on
    # every performance record rather than at report top level.  Promote only
    # values that are internally consistent; disagreement remains unresolved.
    derived: dict[str, set[str]] = defaultdict(set)
    derived_values: dict[str, Any] = {}
    overlaps: dict[int, set[float]] = defaultdict(set)
    for record in basis_report.get("records") or []:
        if not isinstance(record, Mapping):
            continue
        performance = record.get("performance")
        performance = performance if isinstance(performance, Mapping) else {}
        physical = performance.get("physical_priors")
        physical = physical if isinstance(physical, Mapping) else {}
        translations = {
            "dense_peak_flops_per_s": physical.get("dense_bf16_peak_flops_per_gpu"),
            "memory_bandwidth_bytes_per_s": physical.get("hbm_bandwidth_bytes_per_second"),
            "collective_bandwidth_bytes_per_s": physical.get("intra_node_bandwidth_bytes_per_second"),
            "collective_latency_seconds": physical.get("collective_latency_seconds"),
        }
        hbm = physical.get("hbm_efficiency")
        hbm = hbm.get("center") if isinstance(hbm, Mapping) else hbm
        collective = physical.get("collective_efficiency")
        collective = collective.get("center") if isinstance(collective, Mapping) else collective
        translations["hbm_efficiency"] = hbm
        translations["optimizer_hbm_efficiency"] = hbm
        translations["collective_efficiency"] = collective
        for key, value in translations.items():
            finite = _finite(value)
            if finite is not None:
                derived[key].add(repr(finite))
                derived_values[key] = finite
        overlap = _finite(physical.get("communication_overlap"))
        if overlap is not None:
            overlaps[_zero_stage(_selector(record).get("zero_stage"))].add(overlap)
    for key, representations in derived.items():
        if len(representations) == 1 and values.get(key) is None:
            values[key] = derived_values[key]
            sources[key] = "theory_basis_per_record_consensus"
    if overlaps and values.get("communication_overlap_by_stage") is None:
        if all(len(stage_values) == 1 for stage_values in overlaps.values()):
            values["communication_overlap_by_stage"] = {
                stage: next(iter(stage_values)) for stage, stage_values in overlaps.items()
            }
            sources["communication_overlap_by_stage"] = "theory_basis_per_record_consensus"
    for key, value in (overrides or {}).items():
        if key not in _PRIOR_FIELDS and key not in {
            "memory_capacity_bytes",
            "compute_efficiency",
        }:
            raise ValueError(f"unknown physical prior override {key!r}")
        values[key] = value
        sources[key] = "explicit_override"
    normalized = {key: values.get(key) for key in _PRIOR_FIELDS}
    normalized["memory_capacity_bytes"] = values.get("memory_capacity_bytes")
    normalized["compute_efficiency"] = values.get("compute_efficiency")

    positive = {
        "dense_peak_flops_per_s",
        "memory_bandwidth_bytes_per_s",
        "collective_bandwidth_bytes_per_s",
    }
    unit_interval = {
        "hbm_efficiency",
        "optimizer_hbm_efficiency",
        "collective_efficiency",
        "compute_efficiency",
    }
    nonnegative = {
        "collective_latency_seconds",
        "microstep_latency_seconds",
        "framework_latency_seconds",
        "memory_capacity_bytes",
    }
    invalid: list[str] = []
    for key in positive:
        if normalized.get(key) is not None and _positive(normalized[key]) is None:
            invalid.append(key)
    for key in unit_interval:
        value = normalized.get(key)
        if value is not None and (_positive(value) is None or float(value) > 1):
            invalid.append(key)
    for key in nonnegative:
        if normalized.get(key) is not None and _nonnegative(normalized[key]) is None:
            invalid.append(key)
    overlap = normalized.get("communication_overlap_by_stage")
    if overlap is not None:
        if not isinstance(overlap, Mapping):
            invalid.append("communication_overlap_by_stage")
        else:
            for stage, value in overlap.items():
                if _zero_stage(stage) not in {0, 1, 2, 3} or _nonnegative(value) is None or float(value) > 1:
                    invalid.append("communication_overlap_by_stage")
                    break
    if invalid:
        raise ValueError("invalid physical priors: " + ", ".join(sorted(set(invalid))))

    return {
        "values": normalized,
        "sources": {key: sources.get(key) for key in normalized},
        "unresolved": sorted(key for key in _PRIOR_FIELDS if normalized.get(key) is None),
        "policy": "explicit_only_null_is_never_imputed_or_fitted",
    }


def _performance(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("performance")
    return dict(value) if isinstance(value, Mapping) else {}


def _observed_step_seconds(record: Mapping[str, Any]) -> float | None:
    performance = _performance(record)
    observed = performance.get("observed")
    observed = observed if isinstance(observed, Mapping) else {}
    return _positive(performance.get("measured_step_seconds")) or _positive(
        observed.get("mean_step_seconds")
    )


def _work_per_step(record: Mapping[str, Any], key: str) -> float | None:
    performance = _performance(record)
    work = performance.get("work_per_step") or performance.get("work")
    work = work if isinstance(work, Mapping) else {}
    direct = _positive(work.get(f"{key}_per_step"))
    if direct is not None:
        return direct
    value = _positive(work.get(key))
    if value is None:
        return None
    if performance.get("work_is_per_optimizer_step") is True:
        return value
    measured_steps = _positive(performance.get("measured_steps"))
    return value / measured_steps if measured_steps else value


def _flops_total(performance: Mapping[str, Any]) -> float | None:
    flops = performance.get("flops_per_step") or performance.get("flops")
    if not isinstance(flops, Mapping):
        return None
    direct = _positive(flops.get("total"))
    if direct is not None:
        return direct
    values = [_nonnegative(value) for value in flops.values()]
    if not values or any(value is None for value in values):
        return None
    total = sum(float(value) for value in values if value is not None)
    return total if total > 0 else None


def _operator_work(record: Mapping[str, Any]) -> list[tuple[float, float]]:
    """Return ``(FLOPs, traffic bytes/rank)`` operator classes.

    Newer basis versions may expose explicit classes.  The v1 aggregate basis
    remains accepted as one class and is flagged in fit identifiability.
    """

    performance = _performance(record)
    raw_classes = performance.get("operator_classes") or performance.get("operators")
    classes: list[tuple[float, float]] = []
    if isinstance(raw_classes, Sequence) and not isinstance(raw_classes, (str, bytes)):
        for raw in raw_classes:
            if not isinstance(raw, Mapping):
                continue
            flops = _positive(raw.get("flops"))
            traffic = _nonnegative(raw.get("traffic_bytes_per_rank") or raw.get("traffic_bytes"))
            if flops is not None and traffic is not None:
                classes.append((flops, traffic))
    if classes:
        return classes
    total_flops = _flops_total(performance)
    traffic_basis = performance.get("operator_traffic_bytes") or performance.get(
        "traffic_bytes_per_rank_step"
    )
    traffic_basis = traffic_basis if isinstance(traffic_basis, Mapping) else {}
    traffic = _nonnegative(
        traffic_basis.get("kernel_per_rank_per_step")
        if "kernel_per_rank_per_step" in traffic_basis
        else traffic_basis.get("kernel_total")
        if "kernel_total" in traffic_basis
        else traffic_basis.get("total_per_rank_per_step")
    )
    if total_flops is None or traffic is None:
        return []
    return [(total_flops, traffic)]


def _overlap_for_stage(priors: Mapping[str, Any], stage: int) -> float | None:
    raw = priors.get("communication_overlap_by_stage")
    if not isinstance(raw, Mapping):
        return None
    for key in (stage, str(stage), f"zero{stage}"):
        if key in raw:
            value = _nonnegative(raw[key])
            return value if value is not None and value <= 1 else None
    return None


def _roofline_basis(
    record: Mapping[str, Any], priors: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    """Build a piecewise-linear reciprocal-efficiency step-time basis."""

    issues: list[str] = []
    gpu_count = _gpu_count(record)
    mbs = _physical_mbs(record)
    peak = _positive(priors.get("dense_peak_flops_per_s"))
    if gpu_count is None:
        issues.append("gpu_count_missing")
    if mbs is None:
        issues.append("physical_mbs_missing")
    if peak is None:
        issues.append("dense_peak_flops_per_s_prior_missing")
    operators = _operator_work(record)
    if not operators:
        issues.append("operator_flops_or_traffic_missing")

    traffic_present = any(traffic > 0 for _flops, traffic in operators)
    memory_bandwidth = _positive(priors.get("memory_bandwidth_bytes_per_s"))
    hbm_efficiency = _positive(priors.get("hbm_efficiency"))
    if traffic_present and memory_bandwidth is None:
        issues.append("memory_bandwidth_bytes_per_s_prior_missing")
    if traffic_present and hbm_efficiency is None:
        issues.append("hbm_efficiency_prior_missing")

    performance = _performance(record)
    traffic = performance.get("traffic_bytes_per_rank_step")
    traffic = traffic if isinstance(traffic, Mapping) else {}
    optimizer_bytes = _nonnegative(performance.get("optimizer_bytes_per_rank_per_step"))
    if optimizer_bytes is None:
        optimizer_bytes = _nonnegative(traffic.get("optimizer"))
    optimizer_bytes = 0.0 if optimizer_bytes is None else optimizer_bytes
    optimizer_efficiency = _positive(priors.get("optimizer_hbm_efficiency"))
    if optimizer_bytes > 0 and optimizer_efficiency is None:
        issues.append("optimizer_hbm_efficiency_prior_missing")
    if optimizer_bytes > 0 and memory_bandwidth is None:
        issues.append("memory_bandwidth_bytes_per_s_prior_missing")

    communication = performance.get("communication")
    communication = communication if isinstance(communication, Mapping) else {}
    payload = _nonnegative(
        communication.get("payload_bytes_per_rank")
        if "payload_bytes_per_rank" in communication
        else communication.get("payload_bytes_per_rank_step")
    ) or 0.0
    collective_count = _integer(communication.get("collective_count")) or 0
    collective_bandwidth = _positive(priors.get("collective_bandwidth_bytes_per_s"))
    collective_efficiency = _positive(priors.get("collective_efficiency"))
    collective_latency = _nonnegative(priors.get("collective_latency_seconds"))
    stage = _zero_stage(_selector(record).get("zero_stage"))
    overlap = _overlap_for_stage(priors, stage)
    if payload > 0 and collective_bandwidth is None:
        issues.append("collective_bandwidth_bytes_per_s_prior_missing")
    if payload > 0 and collective_efficiency is None:
        issues.append("collective_efficiency_prior_missing")
    if collective_count > 0 and collective_latency is None:
        issues.append("collective_latency_seconds_prior_missing")
    if (payload > 0 or collective_count > 0) and overlap is None:
        issues.append(f"communication_overlap_stage_{stage}_prior_missing")

    latency_basis = performance.get("latency_basis")
    latency_basis = latency_basis if isinstance(latency_basis, Mapping) else {}
    microsteps = _nonnegative(latency_basis.get("microsteps")) or 0.0
    framework_steps = _nonnegative(latency_basis.get("framework_steps")) or 0.0
    micro_latency = _nonnegative(priors.get("microstep_latency_seconds"))
    framework_latency = _nonnegative(priors.get("framework_latency_seconds"))
    if microsteps > 0 and micro_latency is None:
        issues.append("microstep_latency_seconds_prior_missing")
    if framework_steps > 0 and framework_latency is None:
        issues.append("framework_latency_seconds_prior_missing")
    if issues:
        return None, sorted(set(issues))

    assert gpu_count is not None and mbs is not None and peak is not None
    operator_terms = []
    for flops, traffic in operators:
        compute_coefficient = flops / (gpu_count * peak)
        memory_floor = 0.0
        if traffic > 0:
            assert memory_bandwidth is not None and hbm_efficiency is not None
            memory_floor = traffic / (memory_bandwidth * hbm_efficiency)
        operator_terms.append(
            {
                "compute_coefficient_seconds_at_theta_1": compute_coefficient,
                "memory_floor_seconds": memory_floor,
            }
        )

    optimizer_seconds = 0.0
    if optimizer_bytes > 0:
        assert memory_bandwidth is not None and optimizer_efficiency is not None
        optimizer_seconds = optimizer_bytes / (memory_bandwidth * optimizer_efficiency)
    communication_seconds = 0.0
    if payload > 0 or collective_count > 0:
        raw_seconds = 0.0
        if payload > 0:
            assert collective_bandwidth is not None and collective_efficiency is not None
            raw_seconds += payload / (collective_bandwidth * collective_efficiency)
        if collective_count > 0:
            assert collective_latency is not None
            raw_seconds += collective_count * collective_latency
        assert overlap is not None
        communication_seconds = raw_seconds * (1 - overlap)
    latency_seconds = 0.0
    if microsteps > 0:
        assert micro_latency is not None
        latency_seconds += microsteps * micro_latency
    if framework_steps > 0:
        assert framework_latency is not None
        latency_seconds += framework_steps * framework_latency

    return {
        "mbs": mbs,
        "runtime_cohort": _runtime_cohort(record),
        "operator_terms": operator_terms,
        "fixed_seconds": optimizer_seconds + communication_seconds + latency_seconds,
        "optimizer_seconds": optimizer_seconds,
        "communication_seconds": communication_seconds,
        "latency_seconds": latency_seconds,
        "aggregate_operator_basis": not bool(
            performance.get("operator_classes") or performance.get("operators")
        ),
    }, []


def _efficiency_for_mbs(model: Mapping[str, Any], mbs: int) -> float | None:
    raw = model.get("compute_efficiency_by_mbs")
    if not isinstance(raw, Mapping) or not raw:
        return None
    points = sorted(
        (int(key), float(value))
        for key, value in raw.items()
        if _integer(key) is not None and _positive(value) is not None
    )
    if not points:
        return None
    if mbs <= points[0][0]:
        return points[0][1]
    if mbs >= points[-1][0]:
        return points[-1][1]  # explicit saturation above observed MBS
    for (left_mbs, left), (right_mbs, right) in zip(points, points[1:]):
        if left_mbs <= mbs <= right_mbs:
            fraction = (mbs - left_mbs) / (right_mbs - left_mbs)
            return left + fraction * (right - left)
    return None


def _roofline_seconds(basis: Mapping[str, Any], efficiency: float) -> float:
    theta = 1.0 / efficiency
    return sum(
        max(
            float(term["compute_coefficient_seconds_at_theta_1"]) * theta,
            float(term["memory_floor_seconds"]),
        )
        for term in basis["operator_terms"]
    )


def _prepare_throughput_rows(
    records: Sequence[Mapping[str, Any]], priors: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if _outcome(record) != "success":
            continue
        observed = _observed_step_seconds(record)
        if observed is None:
            excluded.append(
                {
                    "observation_id": _observation_id(record, index),
                    "issues": ["measured_step_seconds_missing"],
                }
            )
            continue
        basis, issues = _roofline_basis(record, priors)
        if basis is None:
            excluded.append(
                {
                    "observation_id": _observation_id(record, index),
                    "issues": issues,
                }
            )
            continue
        prepared.append(
            {
                "observation_id": _observation_id(record, index),
                "scenario_id": scenario_id(record),
                "record": record,
                "observed_step_seconds": observed,
                "basis": basis,
            }
        )
    return prepared, excluded


def _fit_compute_model(
    records: Sequence[Mapping[str, Any]],
    priors: Mapping[str, Any],
    *,
    efficiency_bounds: tuple[float, float] = COMPUTE_EFFICIENCY_BOUNDS,
) -> dict[str, Any]:
    prepared, excluded = _prepare_throughput_rows(records, priors)
    if not prepared:
        issues = sorted(
            {
                issue
                for item in excluded
                for issue in item.get("issues", [])
            }
        )
        return {
            "available": False,
            "blockers": issues or ["no_successful_primary_throughput_rows"],
            "excluded": excluded,
        }
    lower_eta, upper_eta = efficiency_bounds
    if not (0 < lower_eta <= upper_eta <= 1):
        raise ValueError("efficiency bounds must satisfy 0 < lower <= upper <= 1")
    mbs_values = sorted({int(item["basis"]["mbs"]) for item in prepared})
    mbs_index = {value: index for index, value in enumerate(mbs_values)}
    counts = {
        value: sum(int(item["basis"]["mbs"]) == value for item in prepared)
        for value in mbs_values
    }
    initial_eta = _positive(priors.get("compute_efficiency")) or 0.35
    initial_eta = min(upper_eta, max(lower_eta, initial_eta))
    efficiency = {value: initial_eta for value in mbs_values}
    nuisance = {
        cohort: 0.0
        for cohort in sorted({str(item["basis"]["runtime_cohort"]) for item in prepared})
    }
    weights = scenario_equal_weights([item["scenario_id"] for item in prepared])
    last_fit = None
    iterations = 0
    for iterations in range(1, 13):
        matrix: list[list[float]] = []
        target: list[float] = []
        for item in prepared:
            basis = item["basis"]
            current_theta = 1.0 / efficiency[int(basis["mbs"])]
            slope = 0.0
            memory_offset = 0.0
            for term in basis["operator_terms"]:
                compute = float(term["compute_coefficient_seconds_at_theta_1"])
                memory = float(term["memory_floor_seconds"])
                if compute * current_theta >= memory:
                    slope += compute
                else:
                    memory_offset += memory
            row = [0.0] * len(mbs_values)
            row[mbs_index[int(basis["mbs"])]] = slope
            matrix.append(row)
            target.append(
                float(item["observed_step_seconds"])
                - float(basis["fixed_seconds"])
                - memory_offset
                - nuisance[str(basis["runtime_cohort"])]
            )
        last_fit = bounded_huber_fit(
            matrix,
            target,
            lower=[1.0 / upper_eta] * len(mbs_values),
            upper=[1.0 / lower_eta] * len(mbs_values),
            sample_weight=weights,
            initial=[1.0 / efficiency[value] for value in mbs_values],
        )
        raw_efficiency = {
            value: 1.0 / last_fit.coefficients[mbs_index[value]]
            for value in mbs_values
        }
        projected = bounded_isotonic_by_mbs(
            raw_efficiency,
            weights=counts,
            lower=lower_eta,
            upper=upper_eta,
        )
        residuals_by_cohort: dict[str, list[float]] = defaultdict(list)
        for item in prepared:
            basis = item["basis"]
            predicted_without_nuisance = (
                _roofline_seconds(basis, projected[int(basis["mbs"])])
                + float(basis["fixed_seconds"])
            )
            residuals_by_cohort[str(basis["runtime_cohort"])].append(
                float(item["observed_step_seconds"]) - predicted_without_nuisance
            )
        updated_nuisance = {
            cohort: max(0.0, float(statistics.median(values)))
            for cohort, values in residuals_by_cohort.items()
        }
        maximum_change = max(
            [abs(projected[value] - efficiency[value]) for value in mbs_values]
            + [
                abs(updated_nuisance.get(cohort, 0.0) - nuisance.get(cohort, 0.0))
                for cohort in set(updated_nuisance) | set(nuisance)
            ]
        )
        efficiency = projected
        nuisance = updated_nuisance
        if maximum_change <= 1e-9:
            break

    assert last_fit is not None
    fitted_predictions = []
    for item in prepared:
        basis = item["basis"]
        fitted_predictions.append(
            _roofline_seconds(basis, efficiency[int(basis["mbs"])])
            + float(basis["fixed_seconds"])
            + nuisance.get(str(basis["runtime_cohort"]), 0.0)
        )
    residuals = [
        math.log(float(item["observed_step_seconds"]) / predicted)
        for item, predicted in zip(prepared, fitted_predictions)
        if predicted > 0
    ]
    return {
        "available": True,
        "compute_efficiency_by_mbs": {
            str(value): efficiency[value] for value in mbs_values
        },
        "saturation_policy": "linear_between_observed_mbs_plateau_above_max",
        "runtime_cohort_nuisance_seconds": dict(sorted(nuisance.items())),
        "runtime_cohort_is_nuisance_not_publication_selector": True,
        "fit_rows": len(prepared),
        "fit_scenarios": len({item["scenario_id"] for item in prepared}),
        "fit_runtime_cohorts": len(nuisance),
        "iterations": iterations,
        "bounded_solver_converged": bool(last_fit.converged),
        "weighted_huber_loss": last_fit.weighted_huber_loss,
        "residual_log_median": _median(residuals),
        "residual_log_p90": _percentile(residuals, 0.90),
        "aggregate_operator_basis_used": any(
            bool(item["basis"]["aggregate_operator_basis"]) for item in prepared
        ),
        "excluded": excluded,
        "blockers": [],
    }


def _predict_step_seconds(
    record: Mapping[str, Any], model: Mapping[str, Any], priors: Mapping[str, Any]
) -> tuple[float | None, list[str]]:
    if model.get("available") is not True:
        return None, list(model.get("blockers") or ["throughput_model_unavailable"])
    basis, issues = _roofline_basis(record, priors)
    if basis is None:
        return None, issues
    efficiency = _efficiency_for_mbs(model, int(basis["mbs"]))
    if efficiency is None:
        return None, ["compute_efficiency_unavailable"]
    nuisance_raw = model.get("runtime_cohort_nuisance_seconds")
    nuisance_raw = nuisance_raw if isinstance(nuisance_raw, Mapping) else {}
    nuisance = _nonnegative(nuisance_raw.get(str(basis["runtime_cohort"]))) or 0.0
    prediction = _roofline_seconds(basis, efficiency) + float(basis["fixed_seconds"]) + nuisance
    if prediction <= 0 or not math.isfinite(prediction):
        return None, ["nonpositive_step_prediction"]
    return prediction, []


def _candidate_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    selector = _selector(record)
    return (
        _gpu_count(record),
        _physical_mbs(record),
        _zero_stage(selector.get("zero_stage")),
        bool(selector.get("gradient_checkpointing")),
        bool(selector.get("packing")),
        str(selector.get("dtype") or "unknown"),
        str(selector.get("kernel_path") or selector.get("kernel") or "unknown"),
    )


def _candidate_material(key: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "gpu_count": key[0],
        "physical_mbs": key[1],
        "zero_stage": key[2],
        "gradient_checkpointing": key[3],
        "packing": key[4],
        "dtype": key[5],
        "kernel": key[6],
    }


def _step_log_residual(
    record: Mapping[str, Any],
    model: Mapping[str, Any],
    priors: Mapping[str, Any],
) -> float | None:
    observed = _observed_step_seconds(record)
    predicted, _issues = _predict_step_seconds(record, model, priors)
    if observed is None or predicted is None or predicted <= 0:
        return None
    return math.log(observed / predicted)


def _make_step_residual_collector(priors: Mapping[str, Any]) -> Any:
    """Return a collector that maps an inner center to its step log residuals."""

    def collect(model: Mapping[str, Any], test: Sequence[Mapping[str, Any]]) -> list[float]:
        residuals: list[float] = []
        for record in test:
            value = _step_log_residual(record, model, priors)
            if value is not None:
                residuals.append(value)
        return residuals

    return collect


def _step_conformal_from_residuals(
    residuals: Sequence[float],
    *,
    alpha: float,
    residual_source: str,
    excluded_rows: int = 0,
) -> dict[str, Any]:
    """Build the one-sided step-time upper conformal from precomputed scores."""

    result = one_sided_conformal(residuals, alpha=alpha, side="upper")
    quantile = result.get("quantile")
    result.update(
        {
            "score": "log(observed_step_seconds/predicted_center_seconds)",
            "residual_source": residual_source,
            "excluded_rows": excluded_rows,
            "log_residual_upper": max(0.0, float(quantile))
            if quantile is not None
            else None,
            "upper_multiplier": math.exp(max(0.0, float(quantile)))
            if quantile is not None
            else None,
        }
    )
    return result


def _step_conformal(
    records: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    priors: Mapping[str, Any],
    *,
    alpha: float,
) -> dict[str, Any]:
    residuals: list[float] = []
    excluded = 0
    for record in records:
        value = _step_log_residual(record, model, priors)
        if value is None:
            excluded += 1
            continue
        residuals.append(value)
    return _step_conformal_from_residuals(
        residuals,
        alpha=alpha,
        residual_source="in_sample_outer_training_fold",
        excluded_rows=excluded,
    )


def _candidate_summaries(
    records: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    priors: Mapping[str, Any],
    conformal: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    multiplier = _positive(conformal.get("upper_multiplier"))
    grouped: dict[tuple[Any, ...], list[dict[str, float]]] = defaultdict(list)
    excluded: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        observed_step = _observed_step_seconds(record)
        predicted_step, issues = _predict_step_seconds(record, model, priors)
        effective_tokens = _work_per_step(record, "effective_tokens")
        if observed_step is None or predicted_step is None or effective_tokens is None:
            excluded.append(
                {
                    "observation_id": _observation_id(record, index),
                    "issues": issues
                    or [
                        "effective_tokens_per_step_missing"
                        if effective_tokens is None
                        else "throughput_label_missing"
                    ],
                }
            )
            continue
        grouped[_candidate_key(record)].append(
            {
                "observed_step_seconds": observed_step,
                "predicted_step_seconds": predicted_step,
                "observed_throughput": effective_tokens / observed_step,
                "predicted_throughput_center": effective_tokens / predicted_step,
                "predicted_throughput_lower": effective_tokens / (predicted_step * multiplier)
                if multiplier is not None
                else math.nan,
            }
        )
    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        rows = grouped[key]
        summaries.append(
            {
                "candidate": _candidate_material(key),
                "replicates": len(rows),
                "observed_step_seconds": _median(
                    row["observed_step_seconds"] for row in rows
                ),
                "predicted_step_seconds": _median(
                    row["predicted_step_seconds"] for row in rows
                ),
                "observed_throughput": _median(
                    row["observed_throughput"] for row in rows
                ),
                "predicted_throughput_center": _median(
                    row["predicted_throughput_center"] for row in rows
                ),
                "predicted_throughput_lower": _median(
                    row["predicted_throughput_lower"]
                    for row in rows
                    if math.isfinite(row["predicted_throughput_lower"])
                ),
            }
        )
    return summaries, excluded


def _joint_candidate_summaries(
    records: Sequence[Mapping[str, Any]],
    throughput_model: Mapping[str, Any],
    priors: Mapping[str, Any],
    conformal: Mapping[str, Any],
    memory_center: Mapping[str, Any],
    memory_tail: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Join every held-out config's throughput and predicted memory admission.

    Each candidate carries its predicted P95 memory admission decision, its
    observed feasibility (did it actually stay under the 0.95 line, or OOM), and
    its throughput bounds.  Ranking and scaling both consume this table so a
    config is never recommended on throughput alone.
    """

    multiplier = _positive(conformal.get("upper_multiplier"))
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        entry: dict[str, Any] = {"outcome": _outcome(record)}
        memory = _predict_memory(record, memory_center, memory_tail)
        entry["predicted_p95_reserved_bytes"] = (
            _positive(memory.get("p95_reserved_bytes"))
            if memory.get("available") is True
            else None
        )
        entry["safe_limit_bytes"] = _safe_limit(record)
        entry["observed_reserved_bytes"] = _observed_reserved(record)
        if _has_route(record, THROUGHPUT_PRIMARY):
            observed_step = _observed_step_seconds(record)
            predicted_step, _issues = _predict_step_seconds(record, throughput_model, priors)
            effective_tokens = _work_per_step(record, "effective_tokens")
            if (
                observed_step is not None
                and predicted_step is not None
                and effective_tokens is not None
            ):
                entry["observed_throughput"] = effective_tokens / observed_step
                entry["predicted_throughput_center"] = effective_tokens / predicted_step
                entry["predicted_throughput_lower"] = (
                    effective_tokens / (predicted_step * multiplier)
                    if multiplier is not None
                    else None
                )
        grouped[_candidate_key(record)].append(entry)

    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        rows = grouped[key]
        predicted_p95 = _median(
            row["predicted_p95_reserved_bytes"]
            for row in rows
            if row.get("predicted_p95_reserved_bytes") is not None
        )
        safe_limit = _median(
            row["safe_limit_bytes"]
            for row in rows
            if row.get("safe_limit_bytes") is not None
        )
        observed_reserved = _median(
            row["observed_reserved_bytes"]
            for row in rows
            if row.get("observed_reserved_bytes") is not None
        )
        outcomes = {row["outcome"] for row in rows}
        # A config that OOMs in any replicate is treated as unsafe (conservative).
        actually_oom = "oom" in outcomes
        predicted_admit = (
            predicted_p95 is not None
            and safe_limit is not None
            and predicted_p95 <= safe_limit
        )
        actually_safe = (
            not actually_oom
            and "success" in outcomes
            and observed_reserved is not None
            and safe_limit is not None
            and observed_reserved <= safe_limit
        )
        summaries.append(
            {
                "candidate": _candidate_material(key),
                "replicates": len(rows),
                "predicted_p95_reserved_bytes": predicted_p95,
                "safe_limit_bytes": safe_limit,
                "observed_reserved_bytes": observed_reserved,
                "predicted_memory_admit": predicted_admit,
                "actually_safe": actually_safe,
                "actually_oom": actually_oom,
                "observed_throughput": _median(
                    row["observed_throughput"]
                    for row in rows
                    if row.get("observed_throughput") is not None
                ),
                "predicted_throughput_center": _median(
                    row["predicted_throughput_center"]
                    for row in rows
                    if row.get("predicted_throughput_center") is not None
                ),
                "predicted_throughput_lower": _median(
                    row["predicted_throughput_lower"]
                    for row in rows
                    if row.get("predicted_throughput_lower") is not None
                ),
            }
        )
    return summaries


def _joint_ranking_metrics(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Memory-gated per-card ranking regret and safety-failure accounting.

    A candidate is rankable only if its predicted P95 clears the 0.95 line.  The
    regret oracle is the best *measured* successful candidate whose observed
    peak also clears the 0.95 line.  Recommending nothing when a real safe
    option exists is regret 1; a predicted-safe candidate that actually OOMs (or
    ran above the line) is a safety failure counted separately, never masked as
    throughput regret.
    """

    by_gpu: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        gpu_count = _integer((candidate.get("candidate") or {}).get("gpu_count"))
        if gpu_count is not None:
            by_gpu[gpu_count].append(candidate)
    groups: list[dict[str, Any]] = []
    for gpu_count in sorted(by_gpu):
        rows = by_gpu[gpu_count]
        real_safe = [
            row
            for row in rows
            if row.get("actually_safe") is True
            and _positive(row.get("observed_throughput")) is not None
        ]
        rankable = [
            row
            for row in rows
            if row.get("predicted_memory_admit") is True
            and _positive(row.get("predicted_throughput_lower")) is not None
            and _positive(row.get("observed_throughput")) is not None
        ]
        safety_failures = [
            row
            for row in rows
            if row.get("predicted_memory_admit") is True
            and (
                row.get("actually_oom") is True
                or (
                    row.get("observed_reserved_bytes") is not None
                    and row.get("safe_limit_bytes") is not None
                    and float(row["observed_reserved_bytes"])
                    > float(row["safe_limit_bytes"])
                )
            )
        ]
        oracle = max(
            (float(row["observed_throughput"]) for row in real_safe), default=None
        )
        selected = (
            max(rankable, key=lambda row: float(row["predicted_throughput_lower"]))
            if rankable
            else None
        )
        if oracle is None:
            top1_regret: float | None = None
            hit: bool | None = None
        elif selected is None:
            top1_regret = 1.0
            hit = False
        else:
            selected_value = float(selected["observed_throughput"])
            top1_regret = max(0.0, (oracle - selected_value) / oracle)
            hit = selected_value >= 0.9 * oracle
        groups.append(
            {
                "gpu_count": gpu_count,
                "candidates": len(rows),
                "real_safe_candidates": len(real_safe),
                "predicted_admitted_rankable": len(rankable),
                "memory_gated_safety_failures": len(safety_failures),
                "selected": selected["candidate"] if selected is not None else None,
                "oracle_safe_best_throughput": oracle,
                "top1_regret": top1_regret,
                "hit_at_10_percent": hit,
            }
        )
    return {
        "gpu_groups": groups,
        "policy": "admit_iff_predicted_p95<=0.95xcapacity_then_rank_by_throughput_lower",
        "scenario_equal_top1_regret": _mean(
            group["top1_regret"] for group in groups if group["top1_regret"] is not None
        ),
        "scenario_equal_hit_at_10_percent": _mean(
            (1.0 if group["hit_at_10_percent"] else 0.0)
            for group in groups
            if group["hit_at_10_percent"] is not None
        ),
        "memory_gated_safety_failures": sum(
            int(group["memory_gated_safety_failures"]) for group in groups
        ),
    }


def _step_metrics(
    records: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    priors: Mapping[str, Any],
    conformal: Mapping[str, Any],
) -> dict[str, Any]:
    multiplier = _positive(conformal.get("upper_multiplier"))
    absolute_percent_errors: list[float] = []
    log_errors: list[float] = []
    covered: list[bool] = []
    for record in records:
        observed = _observed_step_seconds(record)
        predicted, _issues = _predict_step_seconds(record, model, priors)
        if observed is None or predicted is None:
            continue
        absolute_percent_errors.append(abs(predicted - observed) / observed)
        log_errors.append(math.log(predicted / observed))
        if multiplier is not None:
            covered.append(observed <= predicted * multiplier)
    return {
        "rows": len(absolute_percent_errors),
        "median_absolute_percentage_error": _median(absolute_percent_errors),
        "p90_absolute_percentage_error": _percentile(absolute_percent_errors, 0.90),
        "median_log_error": _median(log_errors),
        "upper_step_time_coverage": _mean(1.0 if value else 0.0 for value in covered),
        "coverage_rows": len(covered),
    }


def _scaling_pairs(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_gpu: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        gpu_count = _integer((candidate.get("candidate") or {}).get("gpu_count"))
        if (
            gpu_count is not None
            # Both scaling endpoints must clear the predicted memory admission
            # line; a doubling claim over an inadmissible config is meaningless.
            and candidate.get("predicted_memory_admit") is True
            and _positive(candidate.get("predicted_throughput_lower")) is not None
            and _positive(candidate.get("predicted_throughput_center")) is not None
            and _positive(candidate.get("observed_throughput")) is not None
        ):
            by_gpu[gpu_count].append(candidate)
    selected = {
        gpu_count: max(
            rows, key=lambda row: float(row["predicted_throughput_lower"])
        )
        for gpu_count, rows in by_gpu.items()
    }
    pairs: list[dict[str, Any]] = []
    for gpu_count in sorted(selected):
        if 2 * gpu_count not in selected:
            continue
        current = selected[gpu_count]
        doubled = selected[2 * gpu_count]
        predicted_ratio = float(doubled["predicted_throughput_center"]) / float(
            current["predicted_throughput_center"]
        )
        observed_ratio = float(doubled["observed_throughput"]) / float(
            current["observed_throughput"]
        )
        pairs.append(
            {
                "from_gpus": gpu_count,
                "to_gpus": 2 * gpu_count,
                "both_endpoints_memory_admitted": True,
                "predicted_center_ratio": predicted_ratio,
                "observed_ratio": observed_ratio,
                "log_ratio_residual": math.log(observed_ratio / predicted_ratio),
                "from_candidate": current["candidate"],
                "to_candidate": doubled["candidate"],
            }
        )
    return pairs


def _evaluate_scaling(
    train_candidates_by_scenario: Sequence[Sequence[Mapping[str, Any]]],
    test_candidates: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
) -> dict[str, Any]:
    training_pairs = [
        pair
        for candidates in train_candidates_by_scenario
        for pair in _scaling_pairs(candidates)
    ]
    lower = one_sided_conformal(
        (float(pair["log_ratio_residual"]) for pair in training_pairs),
        alpha=alpha,
        side="lower",
    )
    test_pairs = _scaling_pairs(test_candidates)
    quantile = _finite(lower.get("quantile"))
    claims = 0
    valid_claims = 0
    for pair in test_pairs:
        conservative = (
            float(pair["predicted_center_ratio"]) * math.exp(quantile)
            if quantile is not None
            else None
        )
        pair["conservative_ratio_lower"] = conservative
        pair["recommended_1_8x"] = bool(conservative is not None and conservative >= 1.8)
        pair["measured_clears_1_8x"] = float(pair["observed_ratio"]) >= 1.8
        if pair["recommended_1_8x"]:
            claims += 1
            valid_claims += int(pair["measured_clears_1_8x"])
    return {
        "calibration": lower,
        "training_pairs": len(training_pairs),
        "test_pairs": test_pairs,
        "claims": claims,
        "valid_claims": valid_claims,
        "claim_precision": valid_claims / claims if claims else None,
        "false_claims": claims - valid_claims,
        "policy": (
            "recommend_2N_only_if_conservative_lower>=1.8_and_measured>=1.8_"
            "and_both_endpoints_pass_predicted_memory_admission"
        ),
    }


def _memory(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("memory")
    return dict(value) if isinstance(value, Mapping) else {}


def _memory_basis(record: Mapping[str, Any]) -> tuple[dict[str, float] | None, list[str]]:
    memory = _memory(record)
    activation = _nonnegative(memory.get("structural_activation_bytes"))
    non_activation = _nonnegative(
        memory.get("non_activation_bytes")
        if "non_activation_bytes" in memory
        else memory.get("analytic_non_activation_bytes")
    )
    theory = _nonnegative(
        memory.get("theory_bytes")
        if "theory_bytes" in memory
        else memory.get("analytic_reference_bytes")
    )
    if non_activation is None and theory is not None and activation is not None:
        non_activation = theory - activation
        if non_activation < 0:
            non_activation = None
    issues = []
    if activation is None:
        issues.append("structural_activation_bytes_missing")
    if non_activation is None:
        issues.append("non_activation_bytes_missing")
    if issues:
        return None, issues
    return {"activation": activation, "non_activation": non_activation}, []


def _observed_allocated(record: Mapping[str, Any]) -> float | None:
    memory = _memory(record)
    observed = memory.get("observed")
    observed = observed if isinstance(observed, Mapping) else {}
    return _positive(memory.get("observed_allocated_bytes")) or _positive(
        observed.get("peak_allocated_diagnostic_bytes")
    )


def _observed_reserved(record: Mapping[str, Any]) -> float | None:
    memory = _memory(record)
    observed = memory.get("observed")
    observed = observed if isinstance(observed, Mapping) else {}
    return _positive(memory.get("observed_reserved_bytes")) or _positive(
        observed.get("peak_reserved_target_bytes")
    )


def _oom_lower(record: Mapping[str, Any]) -> float | None:
    memory = _memory(record)
    observed = memory.get("observed")
    observed = observed if isinstance(observed, Mapping) else {}
    return _positive(memory.get("right_censor_lower_bytes")) or _positive(
        observed.get("right_censor_lower_bytes")
    )


def _safe_limit(record: Mapping[str, Any]) -> float | None:
    memory = _memory(record)
    return _positive(memory.get("safe_limit_bytes")) or _oom_lower(record)


def _mode_gc_token(value: tuple[str, bool]) -> str:
    return json.dumps(value, separators=(",", ":"))


def _memory_selector_token(value: tuple[str, int, bool, bool]) -> str:
    return json.dumps(value, separators=(",", ":"))


def _prepare_memory_success(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if _outcome(record) != "success":
            continue
        basis, issues = _memory_basis(record)
        allocated = _observed_allocated(record)
        if allocated is None:
            issues = [*issues, "observed_allocated_bytes_missing"]
        if basis is None or allocated is None:
            excluded.append(
                {
                    "observation_id": _observation_id(record, index),
                    "issues": sorted(set(issues)),
                }
            )
            continue
        prepared.append(
            {
                "observation_id": _observation_id(record, index),
                "scenario_id": scenario_id(record),
                "record": record,
                "basis": basis,
                "allocated": allocated,
                "reserved": _observed_reserved(record),
                "mode_gc": _mode_gc_key(record),
                "selector": _memory_selector_key(record),
                "cohort": _runtime_cohort(record),
            }
        )
    return prepared, excluded


def _fit_memory_center(
    records: Sequence[Mapping[str, Any]],
    *,
    liveness_bounds: tuple[float, float] = ACTIVATION_LIVENESS_BOUNDS,
) -> dict[str, Any]:
    prepared, excluded = _prepare_memory_success(records)
    if not prepared:
        return {
            "available": False,
            "blockers": ["no_memory_boundary_success_with_allocated_label"],
            "excluded": excluded,
        }
    lower, upper = liveness_bounds
    if not (0 < lower <= upper):
        raise ValueError("activation liveness bounds must be finite and positive")
    groups = sorted({item["mode_gc"] for item in prepared})
    cohorts = sorted({item["cohort"] for item in prepared})
    group_index = {value: index for index, value in enumerate(groups)}
    cohort_index = {
        value: len(groups) + index for index, value in enumerate(cohorts)
    }
    matrix: list[list[float]] = []
    target: list[float] = []
    for item in prepared:
        row = [0.0] * (len(groups) + len(cohorts))
        row[group_index[item["mode_gc"]]] = float(item["basis"]["activation"])
        row[cohort_index[item["cohort"]]] = 1.0
        matrix.append(row)
        target.append(float(item["allocated"]) - float(item["basis"]["non_activation"]))
    maximum_label = max(float(item["allocated"]) for item in prepared)
    fit = bounded_huber_fit(
        matrix,
        target,
        lower=[lower] * len(groups) + [0.0] * len(cohorts),
        upper=[upper] * len(groups) + [maximum_label] * len(cohorts),
        sample_weight=scenario_equal_weights([item["scenario_id"] for item in prepared]),
        initial=[1.0] * len(groups) + [0.0] * len(cohorts),
    )
    liveness = {
        _mode_gc_token(group): fit.coefficients[group_index[group]] for group in groups
    }
    nuisance = {
        cohort: fit.coefficients[cohort_index[cohort]] for cohort in cohorts
    }
    return {
        "available": True,
        "activation_liveness_by_mode_gc": dict(sorted(liveness.items())),
        "activation_liveness_material": {
            _mode_gc_token(group): {
                "training_mode": group[0],
                "gradient_checkpointing": group[1],
            }
            for group in groups
        },
        "runtime_cohort_nuisance_allocated_bytes": dict(sorted(nuisance.items())),
        "runtime_cohort_is_nuisance_not_publication_selector": True,
        "fit_rows": len(prepared),
        "fit_scenarios": len({item["scenario_id"] for item in prepared}),
        "fit_runtime_cohorts": len(cohorts),
        "liveness_bounds": [lower, upper],
        "bounded_solver_converged": fit.converged,
        "weighted_huber_loss": fit.weighted_huber_loss,
        "center_identifiability": (
            "mode_x_gc liveness and nonnegative cohort nuisance are bootstrap-only; "
            "cohort nuisance is excluded from publication selectors"
        ),
        "excluded": excluded,
        "blockers": [],
    }


def _predict_allocated_center(
    record: Mapping[str, Any], model: Mapping[str, Any]
) -> tuple[float | None, list[str]]:
    if model.get("available") is not True:
        return None, list(model.get("blockers") or ["memory_center_unavailable"])
    basis, issues = _memory_basis(record)
    if basis is None:
        return None, issues
    liveness_raw = model.get("activation_liveness_by_mode_gc")
    liveness_raw = liveness_raw if isinstance(liveness_raw, Mapping) else {}
    liveness = _positive(liveness_raw.get(_mode_gc_token(_mode_gc_key(record))))
    if liveness is None:
        return None, ["activation_liveness_mode_gc_unseen"]
    nuisance_raw = model.get("runtime_cohort_nuisance_allocated_bytes")
    nuisance_raw = nuisance_raw if isinstance(nuisance_raw, Mapping) else {}
    nuisance = _nonnegative(nuisance_raw.get(_runtime_cohort(record))) or 0.0
    return (
        float(basis["non_activation"])
        + liveness * float(basis["activation"])
        + nuisance,
        [],
    )


def _summary_bucket(values: Sequence[float]) -> dict[str, Any]:
    return {"count": len(values), "median": _median(values)}


def _hierarchical_bucket(
    record: Mapping[str, Any],
    buckets: Mapping[str, Any],
    *,
    minimum_exact: int = MEMORY_SELECTOR_MIN_SUCCESS,
) -> tuple[Mapping[str, Any] | None, str]:
    exact = buckets.get("selector")
    exact = exact if isinstance(exact, Mapping) else {}
    mode_gc = buckets.get("mode_gc")
    mode_gc = mode_gc if isinstance(mode_gc, Mapping) else {}
    pooled = buckets.get("pooled")
    pooled = pooled if isinstance(pooled, Mapping) else None
    exact_bucket = exact.get(_memory_selector_token(_memory_selector_key(record)))
    if isinstance(exact_bucket, Mapping) and int(exact_bucket.get("count") or 0) >= minimum_exact:
        return exact_bucket, "selector"
    mode_bucket = mode_gc.get(_mode_gc_token(_mode_gc_key(record)))
    if isinstance(mode_bucket, Mapping) and int(mode_bucket.get("count") or 0) >= minimum_exact:
        return mode_bucket, "mode_x_gc_fallback"
    return pooled, "pooled_fallback"


def _gap_buckets(reserved_successes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Hierarchical ``reserved - allocated`` center medians (never fitted)."""

    gaps_exact: dict[str, list[float]] = defaultdict(list)
    gaps_mode: dict[str, list[float]] = defaultdict(list)
    gaps_pooled: list[float] = []
    for item in reserved_successes:
        gap = max(0.0, float(item["reserved"]) - float(item["allocated"]))
        gaps_exact[_memory_selector_token(item["selector"])].append(gap)
        gaps_mode[_mode_gc_token(item["mode_gc"])].append(gap)
        gaps_pooled.append(gap)
    return {
        "selector": {key: _summary_bucket(value) for key, value in gaps_exact.items()},
        "mode_gc": {key: _summary_bucket(value) for key, value in gaps_mode.items()},
        "pooled": _summary_bucket(gaps_pooled),
    }


def _fit_reserved_model(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit the allocated center and reserved-gap buckets on one training set."""

    prepared, _excluded = _prepare_memory_success(records)
    reserved_successes = [item for item in prepared if item["reserved"] is not None]
    return {
        "center": _fit_memory_center(records),
        "gap_buckets": _gap_buckets(reserved_successes),
    }


def _predicted_reserved_center(
    record: Mapping[str, Any], reserved_model: Mapping[str, Any]
) -> float | None:
    allocated, _issues = _predict_allocated_center(record, reserved_model["center"])
    if allocated is None:
        return None
    gap_buckets = reserved_model.get("gap_buckets")
    gap_buckets = gap_buckets if isinstance(gap_buckets, Mapping) else {}
    gap_bucket, _source = _hierarchical_bucket(record, gap_buckets)
    gap = _nonnegative((gap_bucket or {}).get("median")) or 0.0
    return float(allocated) + gap


def _oom_guard_floor_bytes(record: Mapping[str, Any]) -> float | None:
    """Lower inequality on true reserved demand for a right-censored OOM.

    Never invents a peak: the demand is at least the recovered censor lower
    bound and strictly above the safe limit that was already exceeded, hence
    ``max(censor_lower, safe_limit + 1)``.
    """

    lower = _oom_lower(record)
    safe = _safe_limit(record)
    floors = [value for value in (lower, (safe + 1.0) if safe is not None else None) if value is not None]
    return max(floors) if floors else None


def _memory_residual_observations(
    reserved_model: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Tagged reserved-residual observations against a reserved-center model.

    Successful runs contribute exact events (``reserved - reserved_center``).
    Right-censored OOMs contribute a lower inequality on the residual using the
    ``max(censor_lower, safe_limit + 1)`` demand floor.
    """

    observations: list[dict[str, Any]] = []
    for record in records:
        outcome = _outcome(record)
        if outcome not in {"success", "oom"}:
            continue
        reserved_center = _predicted_reserved_center(record, reserved_model)
        if reserved_center is None:
            continue
        base = {
            "selector": _memory_selector_key(record),
            "mode_gc": _mode_gc_key(record),
        }
        if outcome == "success":
            reserved = _observed_reserved(record)
            if reserved is None:
                continue
            observations.append(
                {**base, "value": float(reserved) - reserved_center, "event": True}
            )
        else:
            floor = _oom_guard_floor_bytes(record)
            if floor is None:
                continue
            observations.append(
                {**base, "value": floor - reserved_center, "event": False}
            )
    return observations


def _make_memory_residual_collector() -> Any:
    def collect(
        reserved_model: Mapping[str, Any], test: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        return _memory_residual_observations(reserved_model, test)

    return collect


def _cohort_evidence_inflation(
    records: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    base_residual_p95_bytes: float | None,
) -> dict[str, Any]:
    """Non-negative leave-one-cohort-out and evidence-tier uncertainty.

    The inflation is the excess of cross-cohort predictive spread over the
    scenario-OOF spread already priced into the tail -- identifiable only with at
    least two runtime cohorts.  A missing verified publication anchor (every
    tier is ``legacy_*``) stays a blocker; it is never converted into a
    multiplier or a silent default.
    """

    prepared, _excluded = _prepare_memory_success(records)
    reserved = [item for item in prepared if item["reserved"] is not None]
    tier_counts: dict[str, int] = {}
    for item in reserved:
        tier = _evidence_tier(item["record"])
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    verified_anchor = any(_is_verified_anchor_tier(tier) for tier in tier_counts)
    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in reserved:
        by_cohort[item["cohort"]].append(item)
    cohorts = sorted(by_cohort)

    blockers: list[str] = []
    if not verified_anchor:
        blockers.append("verified_calibration_anchor_missing")

    loco_residuals: list[float] = []
    if len(cohorts) >= 2:
        for cohort in cohorts:
            rest = [
                item["record"]
                for other, items in by_cohort.items()
                if other != cohort
                for item in items
            ]
            model = _fit_reserved_model(rest)
            for item in by_cohort[cohort]:
                predicted = _predicted_reserved_center(item["record"], model)
                if predicted is not None:
                    loco_residuals.append(float(item["reserved"]) - predicted)
    loco = (
        one_sided_conformal(loco_residuals, alpha=alpha, side="upper")
        if loco_residuals
        else {"available": False, "quantile": None, "sample_count": 0}
    )
    loco_q = _finite(loco.get("quantile"))
    loco_identifiable = bool(loco.get("available")) and len(cohorts) >= 2
    base = _nonnegative(base_residual_p95_bytes) or 0.0
    if loco_identifiable and loco_q is not None:
        inflation = max(0.0, max(0.0, loco_q) - base)
    else:
        inflation = 0.0
        if len(cohorts) < 2:
            blockers.append("leave_one_cohort_out_not_identifiable_single_cohort")
        else:
            blockers.append("leave_one_cohort_out_not_identifiable_insufficient_scores")
    return {
        "runtime_cohorts": len(cohorts),
        "evidence_tier_counts": dict(sorted(tier_counts.items())),
        "verified_publication_anchor_present": verified_anchor,
        "leave_one_cohort_out_identifiable": loco_identifiable,
        "leave_one_cohort_out_residual_p95_bytes": (
            max(0.0, loco_q) if loco_q is not None else None
        ),
        "base_success_residual_p95_bytes": base,
        "inflation_bytes": inflation,
        "policy": (
            "nonnegative_loco_excess_over_scenario_oof; "
            "missing_verified_anchor_is_a_blocker_not_a_multiplier"
        ),
        "blockers": sorted(set(blockers)),
    }


def _fit_memory_tail(
    records: Sequence[Mapping[str, Any]],
    center: Mapping[str, Any],
    *,
    alpha: float,
    residual_observations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Right-censored reserved-memory tail from out-of-fold residuals.

    ``residual_observations`` are inner-scenario OOF observations produced by
    :func:`_memory_residual_observations`.  When omitted the residuals are
    recomputed in-sample against ``center`` so the function stays usable as a
    unit; the calibration driver always passes OOF observations.
    """

    prepared, success_excluded = _prepare_memory_success(records)
    reserved_successes = [item for item in prepared if item["reserved"] is not None]
    gap_buckets = _gap_buckets(reserved_successes)

    if residual_observations is None:
        reserved_model = {"center": center, "gap_buckets": gap_buckets}
        residual_observations = _memory_residual_observations(reserved_model, records)
        residual_source = "in_sample_outer_training_fold"
    else:
        residual_source = "inner_scenario_kfold_oof"

    success_residual_exact: dict[str, list[float]] = defaultdict(list)
    success_residual_mode: dict[str, list[float]] = defaultdict(list)
    success_residual_pooled: list[float] = []
    censored_exact: dict[str, list[float]] = defaultdict(list)
    censored_pooled: list[float] = []
    for observation in residual_observations:
        selector_token = _memory_selector_token(tuple(observation["selector"]))
        mode_token = _mode_gc_token(tuple(observation["mode_gc"]))
        value = float(observation["value"])
        if observation.get("event") is True:
            success_residual_exact[selector_token].append(value)
            success_residual_mode[mode_token].append(value)
            success_residual_pooled.append(value)
        else:
            censored_exact[selector_token].append(value)
            censored_pooled.append(value)

    def conformal_bucket(values: Sequence[float]) -> dict[str, Any]:
        result = one_sided_conformal(values, alpha=alpha, side="upper")
        value = _finite(result.get("quantile"))
        result["residual_upper_bytes"] = max(0.0, value) if value is not None else None
        return result

    def km_bucket(events: Sequence[float], censored: Sequence[float]) -> dict[str, Any]:
        values = [float(value) for value in events] + [float(value) for value in censored]
        flags = [True] * len(events) + [False] * len(censored)
        if not values:
            return {
                "events": 0,
                "censored": len(censored),
                "identifiable": False,
                "statistical_residual_p95_bytes": None,
            }
        result = kaplan_meier_quantile(values, flags, probability=1 - alpha)
        quantile = _finite(result.value)
        return {
            "events": len(events),
            "censored": len(censored),
            "identifiable": bool(result.identifiable),
            "statistical_residual_p95_bytes": max(0.0, quantile)
            if quantile is not None
            else None,
        }

    success_buckets = {
        "selector": {
            key: conformal_bucket(value) for key, value in success_residual_exact.items()
        },
        "mode_gc": {
            key: conformal_bucket(value) for key, value in success_residual_mode.items()
        },
        "pooled": conformal_bucket(success_residual_pooled),
    }
    statistical_km = {
        "selector": {
            key: km_bucket(success_residual_exact.get(key, []), censored_exact.get(key, []))
            for key in sorted(set(success_residual_exact) | set(censored_exact))
        },
        "pooled": km_bucket(success_residual_pooled, censored_pooled),
    }
    # Operational safety upper bound: never call a seen-OOM selector safe.  The
    # censored floors are already right-inequalities on the residual, so the
    # per-selector maximum is the binding guard.
    oom_bounds = {
        key: {
            "count": len(values),
            "residual_lower_bytes": max(0.0, max(values)),
        }
        for key, values in censored_exact.items()
    }
    pooled_base = _nonnegative(success_buckets["pooled"].get("residual_upper_bytes"))
    cohort_evidence = _cohort_evidence_inflation(
        records, alpha=alpha, base_residual_p95_bytes=pooled_base
    )
    return {
        "available": center.get("available") is True,
        "residual_source": residual_source,
        "reserved_minus_allocated_center": gap_buckets,
        "success_residual_conformal": success_buckets,
        "statistical_residual_p95_km": statistical_km,
        "oom_residual_lower_by_exact_selector": oom_bounds,
        "oom_guard_policy": "max(censor_lower, safe_limit+1) - reserved_center",
        "cohort_evidence_inflation": cohort_evidence,
        "selector_minimum_success_for_own_tail": MEMORY_SELECTOR_MIN_SUCCESS,
        "tail_hierarchy": ["selector", "mode_x_gc", "pooled"],
        "oom_hierarchy": "exact_selector_only_never_pooled",
        "success_rows_with_reserved": len(success_residual_pooled),
        "oom_rows_with_lower_constraint": len(censored_pooled),
        "success_excluded": success_excluded,
    }


def _predict_memory(
    record: Mapping[str, Any],
    center: Mapping[str, Any],
    tail: Mapping[str, Any],
) -> dict[str, Any]:
    allocated, issues = _predict_allocated_center(record, center)
    if allocated is None:
        return {"available": False, "issues": issues}
    gap_buckets = tail.get("reserved_minus_allocated_center")
    gap_buckets = gap_buckets if isinstance(gap_buckets, Mapping) else {}
    gap_bucket, gap_source = _hierarchical_bucket(record, gap_buckets)
    gap = _nonnegative((gap_bucket or {}).get("median")) or 0.0
    reserved_center = allocated + gap

    success_buckets = tail.get("success_residual_conformal")
    success_buckets = success_buckets if isinstance(success_buckets, Mapping) else {}
    success_bucket, success_source = _hierarchical_bucket(record, success_buckets)
    success_q = _nonnegative((success_bucket or {}).get("residual_upper_bytes"))
    success_available = bool((success_bucket or {}).get("available")) and success_q is not None

    oom_raw = tail.get("oom_residual_lower_by_exact_selector")
    oom_raw = oom_raw if isinstance(oom_raw, Mapping) else {}
    selector_token = _memory_selector_token(_memory_selector_key(record))
    oom_bucket = oom_raw.get(selector_token)
    oom_q = _nonnegative((oom_bucket or {}).get("residual_lower_bytes")) or 0.0

    # Statistical P95 (Kaplan-Meier over success events + censored OOM floors)
    # is reported separately from the operational safety upper bound; it may be
    # non-identifiable under heavy censoring and never lowers the guard.
    km_raw = tail.get("statistical_residual_p95_km")
    km_raw = km_raw if isinstance(km_raw, Mapping) else {}
    km_selector = km_raw.get("selector") if isinstance(km_raw.get("selector"), Mapping) else {}
    km_bucket = km_selector.get(selector_token)
    if not isinstance(km_bucket, Mapping) or km_bucket.get("statistical_residual_p95_bytes") is None:
        km_bucket = km_raw.get("pooled") if isinstance(km_raw.get("pooled"), Mapping) else {}
        km_source = "pooled"
    else:
        km_source = "selector"
    km_identifiable = bool((km_bucket or {}).get("identifiable"))
    km_residual = _nonnegative((km_bucket or {}).get("statistical_residual_p95_bytes"))
    statistical_p95_reserved = (
        reserved_center + km_residual if km_identifiable and km_residual is not None else None
    )

    if not success_available:
        return {
            "available": False,
            "allocated_center_bytes": allocated,
            "reserved_center_bytes": reserved_center,
            "reserved_gap_source": gap_source,
            "success_tail_source": success_source,
            "oom_exact_selector_residual_lower_bytes": oom_q,
            "statistical_p95_identifiable": km_identifiable,
            "statistical_p95_reserved_bytes": statistical_p95_reserved,
            "issues": ["finite_sample_95_percent_success_tail_not_identifiable"],
        }
    inflation = _nonnegative(
        (tail.get("cohort_evidence_inflation") or {}).get("inflation_bytes")
    ) or 0.0
    residual = max(0.0, success_q, oom_q) + inflation
    return {
        "available": True,
        "allocated_center_bytes": allocated,
        "reserved_center_bytes": reserved_center,
        "reserved_gap_source": gap_source,
        "success_tail_source": success_source,
        "success_residual_upper_bytes": success_q,
        "oom_exact_selector_residual_lower_bytes": oom_q,
        "cohort_evidence_inflation_bytes": inflation,
        "operational_safety_upper_is_max_of_success_conformal_and_oom_guard": True,
        "statistical_p95_source": km_source,
        "statistical_p95_identifiable": km_identifiable,
        "statistical_p95_reserved_bytes": statistical_p95_reserved,
        "p95_reserved_bytes": reserved_center + residual,
        "issues": [],
    }


def _evaluate_memory(
    records: Sequence[Mapping[str, Any]],
    center: Mapping[str, Any],
    tail: Mapping[str, Any],
) -> dict[str, Any]:
    success_rows = 0
    success_covered = 0
    oom_rows = 0
    false_safe = 0
    unavailable = 0
    false_safe_by_selector: dict[str, int] = defaultdict(int)
    details: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        outcome = _outcome(record)
        if outcome not in {"success", "oom"}:
            continue
        prediction = _predict_memory(record, center, tail)
        detail = {
            "observation_id": _observation_id(record, index),
            "outcome": outcome,
            "prediction_available": prediction.get("available") is True,
        }
        if prediction.get("available") is not True:
            unavailable += 1
            detail["issues"] = prediction.get("issues") or []
            details.append(detail)
            continue
        p95 = float(prediction["p95_reserved_bytes"])
        detail["p95_reserved_bytes"] = p95
        if outcome == "success":
            observed = _observed_reserved(record)
            if observed is None:
                unavailable += 1
                detail["issues"] = ["observed_reserved_bytes_missing"]
            else:
                success_rows += 1
                covered = observed <= p95
                success_covered += int(covered)
                detail.update({"observed_reserved_bytes": observed, "covered": covered})
        else:
            safe_limit = _safe_limit(record)
            lower = _oom_lower(record)
            if safe_limit is None or lower is None:
                unavailable += 1
                detail["issues"] = ["oom_capacity_or_censor_lower_missing"]
            else:
                oom_rows += 1
                is_false_safe = p95 <= safe_limit
                false_safe += int(is_false_safe)
                if is_false_safe:
                    false_safe_by_selector[
                        _memory_selector_token(_memory_selector_key(record))
                    ] += 1
                detail.update(
                    {
                        "right_censor_lower_bytes": lower,
                        "safe_limit_bytes": safe_limit,
                        "false_safe": is_false_safe,
                    }
                )
        details.append(detail)
    return {
        "success_rows": success_rows,
        "success_p95_coverage": success_covered / success_rows if success_rows else None,
        "success_covered": success_covered,
        "oom_rows": oom_rows,
        "false_safe_oom": false_safe,
        "false_safe_oom_rate": false_safe / oom_rows if oom_rows else None,
        "false_safe_oom_by_selector": dict(sorted(false_safe_by_selector.items())),
        "prediction_unavailable_rows": unavailable,
        "details": details,
    }


def _aggregate_validation(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    throughput_primary = [
        fold["throughput"]["primary"]
        for fold in folds
        if isinstance(fold.get("throughput"), Mapping)
        and isinstance(fold["throughput"].get("primary"), Mapping)
        and int(fold["throughput"]["primary"].get("test_rows") or 0) > 0
    ]
    screen = [
        fold["throughput"]["screen_diagnostic"]
        for fold in folds
        if isinstance(fold.get("throughput"), Mapping)
        and isinstance(fold["throughput"].get("screen_diagnostic"), Mapping)
        and int(fold["throughput"]["screen_diagnostic"].get("test_rows") or 0) > 0
    ]
    memory = [
        fold["memory"]["feasibility_validation"]
        for fold in folds
        if isinstance(fold.get("memory"), Mapping)
        and isinstance(fold["memory"].get("feasibility_validation"), Mapping)
        and int(fold["memory"]["feasibility_validation"].get("test_rows") or 0) > 0
    ]
    scaling_claims = sum(
        int(item.get("scaling", {}).get("claims") or 0) for item in throughput_primary
    )
    scaling_valid = sum(
        int(item.get("scaling", {}).get("valid_claims") or 0)
        for item in throughput_primary
    )
    memory_gated_safety_failures = sum(
        int(item.get("joint_memory_gated_ranking", {}).get("memory_gated_safety_failures") or 0)
        for item in throughput_primary
    )
    success_rows = sum(
        int(item.get("metrics", {}).get("success_rows") or 0) for item in memory
    )
    success_covered = sum(
        int(item.get("metrics", {}).get("success_covered") or 0) for item in memory
    )
    oom_rows = sum(int(item.get("metrics", {}).get("oom_rows") or 0) for item in memory)
    false_safe = sum(
        int(item.get("metrics", {}).get("false_safe_oom") or 0) for item in memory
    )
    false_safe_by_selector: dict[str, int] = defaultdict(int)
    for item in memory:
        for selector, count in (
            (item.get("metrics", {}).get("false_safe_oom_by_selector") or {}).items()
        ):
            false_safe_by_selector[selector] += int(count)
    return {
        "throughput_primary": {
            "scenario_folds": len(throughput_primary),
            "scenario_equal_median_step_mape": _mean(
                item.get("step", {}).get("median_absolute_percentage_error")
                for item in throughput_primary
            ),
            "scenario_equal_p90_step_mape": _mean(
                item.get("step", {}).get("p90_absolute_percentage_error")
                for item in throughput_primary
            ),
            "scenario_equal_upper_step_time_coverage": _mean(
                item.get("step", {}).get("upper_step_time_coverage")
                for item in throughput_primary
            ),
            "scenario_equal_top1_regret": _mean(
                item.get("joint_memory_gated_ranking", {}).get("scenario_equal_top1_regret")
                for item in throughput_primary
            ),
            "scenario_equal_hit_at_10_percent": _mean(
                item.get("joint_memory_gated_ranking", {}).get(
                    "scenario_equal_hit_at_10_percent"
                )
                for item in throughput_primary
            ),
            "memory_gated_safety_failures": memory_gated_safety_failures,
            "regret_and_ranking_are_memory_gated": True,
            "scaling_1_8_claims": scaling_claims,
            "scaling_1_8_valid_claims": scaling_valid,
            "scaling_1_8_claim_precision": scaling_valid / scaling_claims
            if scaling_claims
            else None,
            "scaling_1_8_false_claims": scaling_claims - scaling_valid,
            "route_policy": "fit_and_primary_validation",
        },
        "throughput_screen_only": {
            "scenario_folds": len(screen),
            "scenario_equal_median_step_mape": _mean(
                item.get("step", {}).get("median_absolute_percentage_error")
                for item in screen
            ),
            "scenario_equal_p90_step_mape": _mean(
                item.get("step", {}).get("p90_absolute_percentage_error")
                for item in screen
            ),
            "route_policy": "diagnostic_only_never_fit_or_acceptance",
        },
        "memory_feasibility": {
            "scenario_folds": len(memory),
            "success_rows": success_rows,
            "success_p95_coverage": success_covered / success_rows
            if success_rows
            else None,
            "scenario_equal_success_p95_coverage": _mean(
                item.get("metrics", {}).get("success_p95_coverage") for item in memory
            ),
            "oom_rows": oom_rows,
            "false_safe_oom": false_safe,
            "false_safe_oom_rate": false_safe / oom_rows if oom_rows else None,
            "false_safe_oom_by_selector": dict(sorted(false_safe_by_selector.items())),
            "scenario_equal_false_safe_oom_rate": _mean(
                item.get("metrics", {}).get("false_safe_oom_rate") for item in memory
            ),
            "train_route": MEMORY_BOUNDARY,
            "validation_route": FEASIBILITY,
        },
    }


def calibrate_h800_theory(
    basis_report: Mapping[str, Any],
    *,
    physical_priors_override: Mapping[str, Any] | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Run global scenario LOOCV over a theory-basis report.

    This is intentionally a bootstrap audit.  Even a numerically excellent
    report is never promoted to a production planner profile by this function.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    basis_integrity, basis_integrity_blockers = _validate_basis_integrity(basis_report)
    raw_records = basis_report.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("basis report must contain a records array")
    records: list[Mapping[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []
    special_purpose_records: list[dict[str, Any]] = []
    for index, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            excluded_records.append(
                {"observation_id": f"invalid-{index}", "issues": ["record_not_mapping"]}
            )
            continue
        routes = _route_names(record)
        special_routes = sorted(routes.intersection(SPECIAL_PURPOSE_ROUTES))
        if special_routes:
            special_purpose_records.append(
                {
                    "observation_id": _observation_id(record, index),
                    "outcome": _outcome(record),
                    "special_routes": special_routes,
                    "also_marked_feasibility": FEASIBILITY in routes,
                }
            )
            continue
        material = scenario_material(record)
        if any(value is None or value == "" for value in material.values()):
            excluded_records.append(
                {
                    "observation_id": _observation_id(record, index),
                    "issues": ["global_scenario_material_incomplete"],
                }
            )
            continue
        if not routes.intersection(
            {THROUGHPUT_PRIMARY, THROUGHPUT_SCREEN, MEMORY_BOUNDARY, FEASIBILITY}
        ):
            continue
        records.append(record)
    priors_report = resolve_physical_priors(basis_report, physical_priors_override)
    priors = priors_report["values"]

    if not records:
        folds_membership: list[dict[str, Any]] = []
    else:
        folds_membership = build_global_scenario_folds(records)
    by_id = {
        _observation_id(record, index): record for index, record in enumerate(records)
    }
    full_primary = [record for record in records if _has_route(record, THROUGHPUT_PRIMARY)]
    full_memory_rows = [record for record in records if _has_route(record, MEMORY_BOUNDARY)]
    throughput_fit_cache: dict[frozenset[str], Any] = {}
    memory_fit_cache: dict[frozenset[str], Any] = {}
    step_residual_collector = _make_step_residual_collector(priors)
    memory_residual_collector = _make_memory_residual_collector()

    def fit_primary(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return _fit_compute_model(subset, priors)

    fold_reports: list[dict[str, Any]] = []
    fold_blockers: list[str] = []
    for fold in folds_membership:
        train = [by_id[value] for value in fold["train_observation_ids"]]
        test = [by_id[value] for value in fold["test_observation_ids"]]
        train_primary = [record for record in train if _has_route(record, THROUGHPUT_PRIMARY)]
        test_primary = [record for record in test if _has_route(record, THROUGHPUT_PRIMARY)]
        test_screen = [record for record in test if _has_route(record, THROUGHPUT_SCREEN)]

        throughput_model = _fit_compute_model(train_primary, priors)
        oof_step_residuals = _inner_scenario_oof(
            full_primary,
            frozenset({fold["held_out_scenario_id"]}),
            fit_fn=fit_primary,
            collect_fn=step_residual_collector,
            fit_cache=throughput_fit_cache,
            folds=THROUGHPUT_INNER_OOF_FOLDS,
        )
        step_tail = _step_conformal_from_residuals(
            oof_step_residuals,
            alpha=alpha,
            residual_source="inner_scenario_kfold_oof",
        )

        train_memory = [record for record in train if _has_route(record, MEMORY_BOUNDARY)]
        test_feasibility = [record for record in test if _has_route(record, FEASIBILITY)]
        memory_center = _fit_memory_center(train_memory)
        oof_memory_observations = _inner_scenario_oof(
            full_memory_rows,
            frozenset({fold["held_out_scenario_id"]}),
            fit_fn=_fit_reserved_model,
            collect_fn=memory_residual_collector,
            fit_cache=memory_fit_cache,
            folds=MEMORY_INNER_OOF_FOLDS,
        )
        memory_tail = _fit_memory_tail(
            train_memory,
            memory_center,
            alpha=alpha,
            residual_observations=oof_memory_observations,
        )
        memory_metrics = _evaluate_memory(
            test_feasibility, memory_center, memory_tail
        )

        # Joint memory-gated throughput candidates: rank/scale only over configs
        # whose predicted P95 clears the 0.95 admission line, so throughput is
        # never recommended for a config the memory model rejects.
        test_joint_candidates = _joint_candidate_summaries(
            test_feasibility, throughput_model, priors, step_tail, memory_center, memory_tail
        )
        ranking = _joint_ranking_metrics(test_joint_candidates)
        train_joint_by_scenario: list[list[dict[str, Any]]] = []
        train_by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in train:
            if _has_route(record, FEASIBILITY):
                train_by_scenario[scenario_id(record)].append(record)
        for scenario_records in train_by_scenario.values():
            train_joint_by_scenario.append(
                _joint_candidate_summaries(
                    scenario_records,
                    throughput_model,
                    priors,
                    step_tail,
                    memory_center,
                    memory_tail,
                )
            )
        scaling = _evaluate_scaling(
            train_joint_by_scenario, test_joint_candidates, alpha=alpha
        )

        # Screen route stays throughput-only and strictly diagnostic.
        _primary_candidates, primary_excluded = _candidate_summaries(
            test_primary, throughput_model, priors, step_tail
        )
        screen_candidates, screen_excluded = _candidate_summaries(
            test_screen, throughput_model, priors, step_tail
        )

        blockers = list(throughput_model.get("blockers") or [])
        if test_primary and step_tail.get("available") is not True:
            blockers.append("throughput_95_percent_log_conformal_not_identifiable")
        if test_feasibility and memory_metrics["prediction_unavailable_rows"]:
            blockers.append("memory_95_percent_tail_unavailable_for_some_feasibility_rows")
        fold_blockers.extend(blockers)
        fold_reports.append(
            {
                "fold_id": fold["fold_id"],
                "held_out_scenario_id": fold["held_out_scenario_id"],
                "held_out_scenario": fold["held_out_scenario"],
                "train_observations": len(train),
                "test_observations": len(test),
                "train_observation_ids_sha256": fold[
                    "train_observation_ids_sha256"
                ],
                "test_observation_ids_sha256": fold[
                    "test_observation_ids_sha256"
                ],
                "train_runtime_cohorts": fold["train_runtime_cohorts"],
                "test_runtime_cohorts": fold["test_runtime_cohorts"],
                "throughput": {
                    "fit": throughput_model,
                    "step_time_upper_conformal": step_tail,
                    "primary": {
                        "test_rows": len(test_primary),
                        "excluded": primary_excluded,
                        "step": _step_metrics(
                            test_primary, throughput_model, priors, step_tail
                        ),
                        "joint_memory_gated_ranking": ranking,
                        "scaling": scaling,
                    },
                    "screen_diagnostic": {
                        "test_rows": len(test_screen),
                        "excluded": screen_excluded,
                        "step": _step_metrics(
                            test_screen, throughput_model, priors, step_tail
                        ),
                        "candidate_count": len(screen_candidates),
                        "used_for_fit": False,
                        "used_for_acceptance": False,
                    },
                },
                "memory": {
                    "center_fit": memory_center,
                    "tail_fit": memory_tail,
                    "feasibility_validation": {
                        "test_rows": len(test_feasibility),
                        "metrics": memory_metrics,
                    },
                },
                "blockers": sorted(set(blockers)),
            }
        )

    full_throughput = _fit_compute_model(full_primary, priors)
    full_oof_step_residuals = _inner_scenario_oof(
        full_primary,
        frozenset(),
        fit_fn=fit_primary,
        collect_fn=step_residual_collector,
        fit_cache=throughput_fit_cache,
        folds=THROUGHPUT_INNER_OOF_FOLDS,
    )
    full_step_tail = _step_conformal_from_residuals(
        full_oof_step_residuals,
        alpha=alpha,
        residual_source="inner_scenario_kfold_oof",
    )
    full_memory_center = _fit_memory_center(full_memory_rows)
    full_oof_memory_observations = _inner_scenario_oof(
        full_memory_rows,
        frozenset(),
        fit_fn=_fit_reserved_model,
        collect_fn=memory_residual_collector,
        fit_cache=memory_fit_cache,
        folds=MEMORY_INNER_OOF_FOLDS,
    )
    full_memory_tail = _fit_memory_tail(
        full_memory_rows,
        full_memory_center,
        alpha=alpha,
        residual_observations=full_oof_memory_observations,
    )
    aggregate = _aggregate_validation(fold_reports)

    blockers = sorted(
        set(
            [
                "historical_bootstrap_never_generates_production_profile",
                *basis_integrity_blockers,
                *list(
                    (full_memory_tail.get("cohort_evidence_inflation") or {}).get(
                        "blockers"
                    )
                    or []
                ),
                *list(basis_report.get("publication_blockers") or []),
                *fold_blockers,
            ]
        )
    )
    fold_membership_sha256 = _canonical_sha256(
        [
            {
                "fold_id": fold["fold_id"],
                "held_out_scenario_id": fold["held_out_scenario_id"],
                "train_observation_ids_sha256": fold["train_observation_ids_sha256"],
                "test_observation_ids_sha256": fold["test_observation_ids_sha256"],
            }
            for fold in folds_membership
        ]
    )
    implementation_sha256 = _canonical_sha256(
        {
            "implementation_version": IMPLEMENTATION_VERSION,
            "schema": SCHEMA,
            "alpha": alpha,
            "throughput_inner_oof_folds": THROUGHPUT_INNER_OOF_FOLDS,
            "memory_inner_oof_folds": MEMORY_INNER_OOF_FOLDS,
            "memory_selector_min_success": MEMORY_SELECTOR_MIN_SUCCESS,
            "compute_efficiency_bounds": list(COMPUTE_EFFICIENCY_BOUNDS),
            "activation_liveness_bounds": list(ACTIVATION_LIVENESS_BOUNDS),
            "special_purpose_routes": list(SPECIAL_PURPOSE_ROUTES),
        }
    )
    identifiability = {
        "fitted_physical_parameters": [
            "compute_efficiency_by_physical_mbs_bounded_(0,1]_monotone_saturating",
            "activation_liveness_by_training_mode_x_gc_bounded_[0.25,4]",
        ],
        "nuisance_only": [
            "runtime_cohort_additive_step_seconds_nonnegative",
            "runtime_cohort_allocated_bytes_nonnegative",
            "reserved_minus_allocated_bytes_separate_hierarchical_center",
        ],
        "fixed_explicit_physical_priors": sorted(
            key for key, value in priors.items() if value is not None
        ),
        "unresolved_physical_priors": priors_report["unresolved"],
        "aggregate_operator_traffic_limits_identifiability": bool(
            full_throughput.get("aggregate_operator_basis_used")
        ),
        "forbidden_fit_features": ["model_id", "dataset_id", "target_gbs"],
        "runtime_cohort_in_publication_selector": False,
        "tail_residual_policy": "inner_global_scenario_out_of_fold_nested_in_every_outer_fold",
        "tail_residual_scheme": (
            "memory_tail=strict_leave_one_scenario_out (safety-critical, no free "
            "grouping parameter, measured fewer false-safe OOMs than grouped folds); "
            f"throughput_tail=grouped_scenario_kfold(folds<={THROUGHPUT_INNER_OOF_FOLDS}) "
            "(not safety-critical, bounds an otherwise ~quadratic fit cost)"
        ),
        "memory_tail_censoring": (
            "success_residuals_are_exact_events; oom_lower_inequalities_are_right_censored; "
            "statistical_p95_via_kaplan_meier_reported_separately_from_operational_safety_upper_bound"
        ),
        "tail_residual_is_inner_scenario_oof": True,
    }
    route_counts = {
        route: sum(_has_route(record, route) for record in records)
        for route in (THROUGHPUT_PRIMARY, THROUGHPUT_SCREEN, MEMORY_BOUNDARY, FEASIBILITY)
    }
    special_route_counts = {
        route: sum(route in item["special_routes"] for item in special_purpose_records)
        for route in SPECIAL_PURPOSE_ROUTES
    }
    special_outcome_counts = {
        outcome: sum(item["outcome"] == outcome for item in special_purpose_records)
        for outcome in sorted({item["outcome"] for item in special_purpose_records})
    }
    special_ids = sorted(item["observation_id"] for item in special_purpose_records)
    report = {
        "schema": SCHEMA,
        "status": STATUS,
        "confidence": CONFIDENCE,
        "publication": PUBLICATION,
        "publishable": False,
        "production_profile_generated": False,
        "basis_schema": basis_report.get("schema"),
        "basis_report_sha256": basis_report.get("report_sha256"),
        "integrity": {
            "implementation_version": IMPLEMENTATION_VERSION,
            "implementation_sha256": implementation_sha256,
            "basis": basis_integrity,
            "fold_membership_sha256": fold_membership_sha256,
            "production_profile_from_historical_recovery": "forbidden",
        },
        "alpha": alpha,
        "input": {
            "basis_records": len(raw_records),
            "records": len(records),
            "excluded_records": excluded_records,
            "special_purpose_excluded": {
                "records": len(special_purpose_records),
                "route_counts": special_route_counts,
                "outcome_counts": special_outcome_counts,
                "all_were_also_marked_feasibility": all(
                    item["also_marked_feasibility"] for item in special_purpose_records
                ),
                "observation_ids_sha256": _canonical_sha256(special_ids),
                "policy": "profiler_and_packing_specific_routes_are_never_resource_loocv_rows",
            },
            "global_scenarios": len(folds_membership),
            "runtime_cohorts": len({_runtime_cohort(record) for record in records}),
            "route_counts": route_counts,
        },
        "physical_priors": priors_report,
        "identifiability": identifiability,
        "global_scenario_loocv": {
            "policy": "leave-one-(model,training_mode,dataset,target_gbs)-out_globally_across_runtime_cohorts",
            "folds": fold_reports,
        },
        "aggregate_validation": aggregate,
        "full_historical_bootstrap_fit": {
            "throughput": {
                "center": full_throughput,
                "step_time_upper_conformal": full_step_tail,
            },
            "memory": {
                "center": full_memory_center,
                "tail": full_memory_tail,
            },
            "publication_use": "forbidden",
        },
        "blockers": blockers,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


# Short aliases keep this module consistent with other offline audit scripts.
analyze = calibrate_h800_theory
build_report = calibrate_h800_theory


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Validate fail-closed and fold-leakage invariants of a built report."""

    issues: list[str] = []
    if report.get("schema") != SCHEMA:
        issues.append("schema_mismatch")
    if report.get("status") != STATUS:
        issues.append("status_is_not_theory_only")
    if report.get("confidence") != CONFIDENCE:
        issues.append("confidence_is_not_bootstrap")
    if report.get("publication") != PUBLICATION or report.get("publishable") is not False:
        issues.append("report_is_not_fail_closed_nonpublishable")
    if report.get("production_profile_generated") is not False:
        issues.append("production_profile_generation_must_be_false")
    identifiability = report.get("identifiability")
    identifiability = identifiability if isinstance(identifiability, Mapping) else {}
    if identifiability.get("tail_residual_is_inner_scenario_oof") is not True:
        issues.append("tail_residual_is_not_inner_scenario_oof")
    if (
        "conformal_tails_use_same_outer_training_fold_residuals_not_inner_scenario_oof"
        in (report.get("blockers") or [])
    ):
        issues.append("stale_non_oof_tail_blocker_present_after_inner_oof_fix")
    expected_digest = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if expected_digest != _canonical_sha256(unsigned):
        issues.append("report_sha256_mismatch")
    if _contains_non_finite(unsigned):
        issues.append("report_contains_non_finite_values")

    integrity = report.get("integrity")
    integrity = integrity if isinstance(integrity, Mapping) else {}
    if not integrity.get("implementation_sha256") or not integrity.get(
        "fold_membership_sha256"
    ):
        issues.append("integrity_hashes_missing")
    basis_integrity = integrity.get("basis")
    basis_integrity = basis_integrity if isinstance(basis_integrity, Mapping) else {}
    if basis_integrity.get("validated_before_fit") is not True:
        issues.append("basis_integrity_not_validated_before_fit")
    # A recorded basis schema/digest failure must be carried as a blocker, never
    # silently accepted.
    if basis_integrity.get("schema_ok") is not True and (
        "basis_schema_unexpected" not in (report.get("blockers") or [])
    ):
        issues.append("basis_schema_failure_not_blocked")
    if basis_integrity.get("digest_ok") is not True and (
        "basis_report_sha256_mismatch" not in (report.get("blockers") or [])
    ):
        issues.append("basis_digest_failure_not_blocked")

    loocv = report.get("global_scenario_loocv")
    loocv = loocv if isinstance(loocv, Mapping) else {}
    folds = loocv.get("folds")
    input_summary = report.get("input")
    input_summary = input_summary if isinstance(input_summary, Mapping) else {}
    special = input_summary.get("special_purpose_excluded")
    if not isinstance(special, Mapping) or special.get("policy") != (
        "profiler_and_packing_specific_routes_are_never_resource_loocv_rows"
    ):
        issues.append("special_purpose_resource_exclusion_audit_missing")
    if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)):
        issues.append("global_scenario_folds_missing")
        folds = []
    test_counts: dict[str, int] = defaultdict(int)
    for fold in folds:
        if not isinstance(fold, Mapping):
            issues.append("fold_not_mapping")
            continue
        throughput = fold.get("throughput")
        throughput = throughput if isinstance(throughput, Mapping) else {}
        step_tail = throughput.get("step_time_upper_conformal")
        if isinstance(step_tail, Mapping) and step_tail.get("residual_source") != (
            "inner_scenario_kfold_oof"
        ):
            issues.append("step_tail_residual_is_not_inner_scenario_oof")
        screen = throughput.get("screen_diagnostic")
        if isinstance(screen, Mapping) and (
            screen.get("used_for_fit") is not False
            or screen.get("used_for_acceptance") is not False
        ):
            issues.append("screen_route_used_for_fit_or_acceptance")
        # Membership lists intentionally live in calibration_math's fold builder;
        # this report stores their hashes.  Reconstruct exact-once test membership
        # from the held-out scenario IDs, which must be unique globally.
        scenario = fold.get("held_out_scenario_id")
        if not isinstance(scenario, str) or not scenario:
            issues.append("fold_held_out_scenario_missing")
        else:
            test_counts[scenario] += 1
        center = throughput.get("fit")
        if isinstance(center, Mapping) and center.get("available") is True:
            raw = center.get("compute_efficiency_by_mbs")
            if not isinstance(raw, Mapping) or not raw:
                issues.append("compute_efficiency_map_missing")
            else:
                ordered = sorted((int(key), float(value)) for key, value in raw.items())
                if any(not 0 < value <= 1 for _mbs, value in ordered):
                    issues.append("compute_efficiency_out_of_bounds")
                if any(left[1] > right[1] + 1e-12 for left, right in zip(ordered, ordered[1:])):
                    issues.append("compute_efficiency_not_monotone_by_mbs")
        memory = fold.get("memory")
        memory = memory if isinstance(memory, Mapping) else {}
        memory_center = memory.get("center_fit")
        if isinstance(memory_center, Mapping) and memory_center.get("available") is True:
            liveness = memory_center.get("activation_liveness_by_mode_gc")
            if not isinstance(liveness, Mapping) or any(
                not ACTIVATION_LIVENESS_BOUNDS[0]
                <= float(value)
                <= ACTIVATION_LIVENESS_BOUNDS[1]
                for value in (liveness or {}).values()
            ):
                issues.append("activation_liveness_out_of_bounds")
        memory_tail = memory.get("tail_fit")
        if isinstance(memory_tail, Mapping):
            if memory_tail.get("oom_hierarchy") != "exact_selector_only_never_pooled":
                issues.append("oom_tail_is_not_exact_selector_only")
            if memory_tail.get("residual_source") != "inner_scenario_kfold_oof":
                issues.append("memory_tail_residual_is_not_inner_scenario_oof")
    if any(count != 1 for count in test_counts.values()):
        issues.append("held_out_scenario_not_tested_exactly_once")
    return sorted(set(issues))


def write_report(report: Mapping[str, Any], path: Path) -> None:
    """Atomically write a report so a partial file can never be observed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    import os

    os.replace(temporary, path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--basis",
        type=Path,
        default=root / "artifacts" / "h800_theory_basis.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts" / "h800_theory_calibration.json",
    )
    parser.add_argument(
        "--physical-priors",
        type=Path,
        help="Optional JSON object of explicit prior overrides; null is never imputed.",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()
    basis = json.loads(args.basis.read_text(encoding="utf-8"))
    overrides = (
        json.loads(args.physical_priors.read_text(encoding="utf-8"))
        if args.physical_priors
        else None
    )
    report = calibrate_h800_theory(
        basis,
        physical_priors_override=overrides,
        alpha=args.alpha,
    )
    validation_issues = validate_report(report)
    if validation_issues:
        raise SystemExit("invalid calibration report: " + ", ".join(validation_issues))
    write_report(report, args.output)
    summary = report["aggregate_validation"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": report["report_sha256"],
                "status": report["status"],
                "publication": report["publication"],
                "global_scenarios": report["input"]["global_scenarios"],
                "throughput": summary["throughput_primary"],
                "memory": summary["memory_feasibility"],
                "blockers": report["blockers"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
