#!/usr/bin/env python3
"""Deterministic post-stage decisions for the staged SFT experiments.

This module is intentionally separate from the frozen training implementation so
it can be prepared while an approved memory-boundary run is still in progress.
It never launches training.  Its inputs are completed result directories and the
frozen request files; its outputs are derived JSON reports under ``artifacts``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from collect_results import aggregate_result
from common import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    MATRIX_DIR,
    RESULTS_DIR,
    read_json,
    read_jsonl,
    write_json,
)
from flops import training_flops


RATE_FIELD = "samples_per_second"
CLOCK_STATUS_PRIORITY = {
    "insufficient_data": 0,
    "normal": 1,
    "downclocked": 2,
    "power_limited": 3,
    "thermal_limited": 4,
}
CONFIG_FIELDS = (
    "request_id",
    "model_id",
    "train_type",
    "dataset_id",
    "cutoff_len",
    "gpu_count",
    "zero",
    "gc",
    "mbs",
    "target_gbs",
    "packing",
)
RUN_IDENTITY_FIELDS = (*CONFIG_FIELDS, "repeat")


def median(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.median(clean) if clean else None


def coefficient_of_variation(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2:
        return None
    mean = statistics.fmean(clean)
    return statistics.stdev(clean) / mean if mean else None


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def expected_repeats(
    request: dict[str, Any] | None, rows: Sequence[dict[str, Any]]
) -> int:
    if request is not None and request.get("repeats") is not None:
        return int(request["repeats"])
    indices = [int(row.get("repeat", 0)) for row in rows]
    return max(indices, default=0) + 1


def configuration_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in CONFIG_FIELDS)


def run_identity_matches(result: dict[str, Any], planned: dict[str, Any]) -> bool:
    """Match the immutable physical/run identity while allowing longer old runs.

    Warmup/measurement windows and scheduler hints have changed during this
    campaign, while a stable job ID intentionally allows a longer healthy
    measurement to satisfy a shorter current plan.  The physical configuration
    and repeat index must nevertheless match exactly.
    """

    return all(result.get(field) == planned.get(field) for field in RUN_IDENTITY_FIELDS)


def active_result_rows(
    rows: Sequence[dict[str, Any]],
    planned_jobs: Sequence[dict[str, Any]],
    *,
    expected_kind: str,
) -> list[dict[str, Any]]:
    """Return only exact current-matrix jobs with matching kind and identity."""

    planned_counts: dict[str, int] = defaultdict(int)
    planned_by_id: dict[str, dict[str, Any]] = {}
    for planned in planned_jobs:
        job_id = str(planned.get("job_id") or "")
        planned_counts[job_id] += 1
        planned_by_id[job_id] = planned
    unique_plans = {
        job_id: planned
        for job_id, planned in planned_by_id.items()
        if job_id and planned_counts[job_id] == 1
    }
    return [
        row
        for row in rows
        if (
            (planned := unique_plans.get(str(row.get("job_id") or ""))) is not None
            and planned.get("kind") == expected_kind
            and row.get("kind") == expected_kind
            and run_identity_matches(row, planned)
        )
    ]


def aggregate_configurations(
    rows: Sequence[dict[str, Any]],
    requests: dict[str, dict[str, Any]] | None = None,
    planned_rows: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate planned runs and require a complete run set.

    The current campaigns plan one run per configuration.  The aggregation
    remains generic for historical or explicitly repeated designs.  Duplicate
    run indices are treated as incomplete/invalid rather than silently averaged.
    """

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[configuration_key(row)].append(row)
    planned_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] | None = None
    if planned_rows is not None:
        planned_grouped = defaultdict(list)
        for row in planned_rows:
            planned_grouped[configuration_key(row)].append(row)
    aggregated = []
    keys = planned_grouped if planned_grouped is not None else grouped
    for key, plan_config_rows in keys.items():
        config_rows = grouped.get(key, [])
        base = plan_config_rows[0]
        request = (requests or {}).get(str(base.get("request_id")))
        if planned_grouped is None:
            required = expected_repeats(request, config_rows)
            planned_job_ids: list[str] | None = None
            planned_repeat_indices = list(range(required))
        else:
            required = len(plan_config_rows)
            planned_job_ids = [str(row.get("job_id")) for row in plan_config_rows]
            planned_repeat_indices = [
                int(row.get("repeat", 0)) for row in plan_config_rows
            ]
        successful = [
            row
            for row in config_rows
            if row.get("classification") == "success" and row.get(RATE_FIELD)
        ]
        successful_indices = [int(row.get("repeat", 0)) for row in successful]
        if planned_job_ids is None:
            complete = (
                len(successful) == required
                and len(set(successful_indices)) == required
                and set(successful_indices) == set(planned_repeat_indices)
            )
        else:
            successful_job_ids = [str(row.get("job_id")) for row in successful]
            complete = (
                required > 0
                and len(planned_job_ids) == len(set(planned_job_ids))
                and len(planned_repeat_indices) == len(set(planned_repeat_indices))
                and len(successful) == required
                and set(successful_job_ids) == set(planned_job_ids)
                and set(successful_indices) == set(planned_repeat_indices)
            )
        result = {field: base.get(field) for field in CONFIG_FIELDS}
        result.update(
            {
                "kind": base.get("kind"),
                "planned_repeats": required,
                "planned_repeat_indices": sorted(planned_repeat_indices),
                "planned_job_ids": (
                    sorted(planned_job_ids) if planned_job_ids is not None else None
                ),
                "observed_runs": len(config_rows),
                "successful_runs": len(successful),
                "complete": complete,
                "job_ids": sorted(str(row.get("job_id")) for row in config_rows),
                RATE_FIELD: median(row.get(RATE_FIELD) for row in successful),
                "samples_per_second_cv": coefficient_of_variation(
                    row.get(RATE_FIELD) for row in successful
                ),
                "effective_tokens_per_second": median(
                    row.get("effective_tokens_per_second") for row in successful
                ),
                "computed_tokens_per_second": median(
                    row.get("computed_tokens_per_second") for row in successful
                ),
                "measured_seconds": median(
                    row.get("measured_seconds") for row in successful
                ),
                "max_allocated_bytes": median(
                    row.get("max_allocated_bytes") for row in successful
                ),
                "max_reserved_bytes": median(
                    row.get("max_reserved_bytes") for row in successful
                ),
                "nvidia_smi_peak_mib": median(
                    row.get("nvidia_smi_peak_mib") for row in successful
                ),
                "mfu": median(row.get("mfu") for row in successful),
                "effective_mfu": median(row.get("effective_mfu") for row in successful),
                "clock_adjusted_mfu": median(
                    row.get("clock_adjusted_mfu") for row in successful
                ),
                "busy_clock_p5_mhz": median(
                    row.get("busy_clock_p5_mhz") for row in successful
                ),
                "busy_clock_p50_mhz": median(
                    row.get("busy_clock_p50_mhz") for row in successful
                ),
                "busy_clock_p50_ratio": median(
                    row.get("busy_clock_p50_ratio") for row in successful
                ),
                "busy_clock_below_90pct_fraction": median(
                    row.get("busy_clock_below_90pct_fraction") for row in successful
                ),
                "power_limit_busy_fraction": median(
                    row.get("power_limit_busy_fraction") for row in successful
                ),
                "busy_temperature_p95_c": median(
                    row.get("busy_temperature_p95_c") for row in successful
                ),
                "sw_power_cap_busy_fraction": median(
                    row.get("sw_power_cap_busy_fraction") for row in successful
                ),
                "sw_thermal_slowdown_busy_fraction": median(
                    row.get("sw_thermal_slowdown_busy_fraction") for row in successful
                ),
                "hw_thermal_slowdown_busy_fraction": median(
                    row.get("hw_thermal_slowdown_busy_fraction") for row in successful
                ),
            }
        )
        clock_statuses = [
            str(row.get("clock_status"))
            for row in successful
            if row.get("clock_status")
        ]
        result["clock_status_counts"] = {
            status: clock_statuses.count(status)
            for status in sorted(set(clock_statuses))
        }
        result["clock_status"] = max(
            clock_statuses,
            key=lambda status: CLOCK_STATUS_PRIORITY.get(status, -1),
            default="insufficient_data",
        )
        rate = result[RATE_FIELD]
        result["seconds_per_1000_samples"] = 1000.0 / rate if rate else None
        aggregated.append(result)
    return sorted(
        aggregated,
        key=lambda row: tuple(str(row.get(field)) for field in CONFIG_FIELDS),
    )


def select_fastest(configurations: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        row for row in configurations if row.get("complete") and row.get(RATE_FIELD)
    ]
    if not eligible:
        return None
    # The final tuple makes ties deterministic and favors GC-off, then a larger
    # physical MBS.  The measured rate remains the primary and decisive metric.
    return max(
        eligible,
        key=lambda row: (
            float(row[RATE_FIELD]),
            not bool(row.get("gc")),
            int(row.get("mbs") or 0),
            str(row.get("zero") or ""),
        ),
    )


def throughput_decisions(
    rows: Sequence[dict[str, Any]],
    requests: Sequence[dict[str, Any]],
    planned_jobs: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    request_map = {str(row["request_id"]): row for row in requests}
    if planned_jobs is None:
        active_requests = list(requests)
        request_ids = set(request_map)
        stage_rows = [
            row
            for row in rows
            if str(row.get("request_id")) in request_ids
            and row.get("kind") == "throughput"
        ]
        configs = aggregate_configurations(stage_rows, request_map)
    else:
        stage_plans = [
            job
            for job in planned_jobs
            if str(job.get("request_id")) in request_map
            and job.get("kind") == "throughput"
        ]
        active_request_ids = {str(job.get("request_id")) for job in stage_plans}
        active_requests = [
            request
            for request in requests
            if str(request.get("request_id")) in active_request_ids
        ]
        stage_rows = active_result_rows(
            rows,
            stage_plans,
            expected_kind="throughput",
        )
        configs = aggregate_configurations(
            stage_rows,
            request_map,
            stage_plans,
        )
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in configs:
        by_request[str(row.get("request_id"))].append(row)
    decisions = []
    for request in active_requests:
        request_id = str(request["request_id"])
        choices = by_request.get(request_id, [])
        best = select_fastest(choices)
        decisions.append(
            {
                "request_id": request_id,
                "status": "selected" if best else "pending_or_incomplete",
                "candidate_configurations": len(choices),
                "complete_configurations": sum(
                    bool(row.get("complete")) for row in choices
                ),
                "selected": best,
            }
        )
    return {
        "schema_version": 1,
        "metric": RATE_FIELD,
        "selection": "highest samples/s among configurations with every planned run successful; median if a design explicitly has multiple runs",
        "requests": len(active_requests),
        "selected": sum(row["status"] == "selected" for row in decisions),
        "decisions": decisions,
        "configurations": configs,
    }


def canonical_formal_observation(
    rows: Sequence[dict[str, Any]],
    active_job_ids: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
    """Choose one reproducible formal substitute for a screen candidate.

    An exact current-matrix formal result wins over historical evidence. If
    only history exists, the lowest repeat and then lexical job ID are used;
    this avoids changing an aggregate median as the result directory grows.
    """

    eligible = [
        row
        for row in rows
        if row.get("kind") == "throughput"
        and row.get("classification") == "success"
        and row.get(RATE_FIELD)
        and row.get("job_id")
    ]
    if not eligible:
        return None, None
    selected = min(
        eligible,
        key=lambda row: (
            str(row.get("job_id")) not in active_job_ids,
            int(row.get("repeat", 0)),
            str(row.get("job_id")),
        ),
    )
    scope = "active" if str(selected.get("job_id")) in active_job_ids else "historical"
    return selected, scope


def throughput_screening_decisions(
    rows: Sequence[dict[str, Any]],
    requests: Sequence[dict[str, Any]],
    planned_jobs: Sequence[dict[str, Any]],
    top_k: int,
    active_formal_jobs: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rank short measurements while preserving resource-diverse finalists.

    A completed formal measurement is a stronger substitute for a screening
    measurement of the same physical configuration. Short measurements are
    only used to choose finalists and are never mixed into final throughput
    aggregates.
    """

    request_map = {str(row["request_id"]): row for row in requests}
    request_ids = set(request_map)
    screen_planned: dict[tuple[Any, ...], dict[str, Any]] = {}
    for job in planned_jobs:
        if str(job.get("request_id")) in request_ids:
            screen_planned[configuration_key(job)] = job

    active_screen_rows = active_result_rows(
        rows,
        list(screen_planned.values()),
        expected_kind="throughput_screen",
    )
    screens_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in active_screen_rows:
        screens_by_key[configuration_key(row)].append(row)

    active_formal_rows = active_result_rows(
        rows,
        active_formal_jobs or (),
        expected_kind="throughput",
    )
    successful_active_formal_keys = {
        configuration_key(row)
        for row in active_formal_rows
        if row.get("classification") == "success" and row.get(RATE_FIELD)
    }
    # A healthy formal measurement is stronger than a short screen and must
    # remain eligible even when the bounded screen matrix no longer contains
    # that physical configuration.  Otherwise rematerializing a screen delta
    # can silently collapse a two-candidate shortlist to one despite already
    # having trustworthy formal evidence for the omitted alternative.
    planned = dict(screen_planned)
    for job in active_formal_jobs or ():
        key = configuration_key(job)
        if (
            str(job.get("request_id")) in request_ids
            and key in successful_active_formal_keys
        ):
            planned.setdefault(key, job)
    planned_formal_job_ids = {
        str(job.get("job_id")) for job in active_formal_jobs or () if job.get("job_id")
    }
    active_formal_job_ids = {str(row.get("job_id")) for row in active_formal_rows}
    formal_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            str(row.get("request_id")) in request_ids
            and row.get("kind") == "throughput"
            and row.get("classification") == "success"
            and row.get(RATE_FIELD)
            and (
                str(row.get("job_id")) not in planned_formal_job_ids
                or str(row.get("job_id")) in active_formal_job_ids
            )
        ):
            formal_by_key[configuration_key(row)].append(row)

    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    configurations = []
    for key, job in sorted(
        planned.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        active_screens = screens_by_key.get(key, [])
        formal, formal_scope = canonical_formal_observation(
            formal_by_key.get(key, []),
            active_formal_job_ids,
        )
        successful_screens = [
            row
            for row in active_screens
            if row.get("classification") == "success" and row.get(RATE_FIELD)
        ]
        screen = min(
            successful_screens,
            key=lambda row: (
                int(row.get("repeat", 0)),
                str(row.get("job_id")),
            ),
            default=None,
        )
        selected_rows = (
            [formal] if formal is not None else ([screen] if screen is not None else [])
        )
        terminal = next(
            (
                str(row.get("classification"))
                for row in sorted(
                    active_screens,
                    key=lambda item: str(item.get("job_id")),
                )
                if row.get("classification") in {"oom", "failed", "incomplete_metrics"}
            ),
            None,
        )
        candidate = {field: job.get(field) for field in CONFIG_FIELDS}
        candidate.update(
            {
                "candidate_plan_source": (
                    "screen" if key in screen_planned else "active_formal"
                ),
                "status": "measured" if selected_rows else (terminal or "pending"),
                "measurement_source": "formal"
                if formal
                else ("screen" if selected_rows else None),
                "formal_substitute_scope": formal_scope,
                "canonical_selection": (
                    "prefer_active_formal_then_lowest_repeat_and_job_id"
                    if formal is not None
                    else ("exact_active_screen_job" if screen is not None else None)
                ),
                "job_ids": sorted(str(row.get("job_id")) for row in selected_rows),
                RATE_FIELD: median(row.get(RATE_FIELD) for row in selected_rows),
                "samples_per_second_cv": coefficient_of_variation(
                    row.get(RATE_FIELD) for row in selected_rows
                ),
                "mfu": median(row.get("mfu") for row in selected_rows),
                "sw_power_cap_busy_fraction": median(
                    row.get("sw_power_cap_busy_fraction") for row in selected_rows
                ),
                "sw_thermal_slowdown_busy_fraction": median(
                    row.get("sw_thermal_slowdown_busy_fraction")
                    for row in selected_rows
                ),
                "hw_thermal_slowdown_busy_fraction": median(
                    row.get("hw_thermal_slowdown_busy_fraction")
                    for row in selected_rows
                ),
            }
        )
        clock_statuses = [
            str(row.get("clock_status"))
            for row in selected_rows
            if row.get("clock_status")
        ]
        candidate["clock_status_counts"] = {
            status: clock_statuses.count(status)
            for status in sorted(set(clock_statuses))
        }
        candidate["clock_status"] = max(
            clock_statuses,
            key=lambda status: CLOCK_STATUS_PRIORITY.get(status, -1),
            default="insufficient_data",
        )
        rate = candidate[RATE_FIELD]
        gpu_count = int(candidate.get("gpu_count") or 1)
        candidate["seconds_per_1000_samples"] = 1000.0 / rate if rate else None
        candidate["gpu_hours_per_1000_samples"] = (
            1000.0 * gpu_count / rate / 3600.0 if rate else None
        )
        configurations.append(candidate)
        by_request[str(candidate["request_id"])].append(candidate)

    decisions = []
    for request in requests:
        request_id = str(request["request_id"])
        candidates = by_request.get(request_id, [])
        measured = [row for row in candidates if row.get(RATE_FIELD)]
        resolved = [row for row in candidates if row.get("status") != "pending"]
        shortlisted: list[dict[str, Any]] = []
        selected_keys: set[tuple[Any, ...]] = set()

        # A two-slot shortlist must retain both ends needed by the final
        # recommendation: the fastest low-resource option and the fastest
        # option overall. Additional slots then add intermediate GPU counts.
        gpu_winners = [
            max(
                (
                    row
                    for row in measured
                    if int(row.get("gpu_count") or 1) == gpu_count
                ),
                key=lambda row: (
                    float(row[RATE_FIELD]),
                    not bool(row.get("gc")),
                    int(row.get("mbs") or 0),
                ),
            )
            for gpu_count in sorted(
                {int(row.get("gpu_count") or 1) for row in measured}
            )
        ]
        preferred = []
        if gpu_winners:
            preferred.append(gpu_winners[0])
        if measured:
            preferred.append(
                max(
                    measured,
                    key=lambda row: (
                        float(row[RATE_FIELD]),
                        -int(row.get("gpu_count") or 1),
                        not bool(row.get("gc")),
                        int(row.get("mbs") or 0),
                    ),
                )
            )
        preferred.extend(
            sorted(
                gpu_winners[1:],
                key=lambda row: (
                    -float(row[RATE_FIELD]),
                    int(row.get("gpu_count") or 1),
                ),
            )
        )
        for best in preferred:
            key = configuration_key(best)
            if key not in selected_keys and len(shortlisted) < top_k:
                shortlisted.append(best)
                selected_keys.add(key)

        for candidate in sorted(
            measured,
            key=lambda row: (
                -float(row[RATE_FIELD]),
                int(row.get("gpu_count") or 1),
                not bool(row.get("gc")),
                -int(row.get("mbs") or 0),
            ),
        ):
            key = configuration_key(candidate)
            if key not in selected_keys and len(shortlisted) < top_k:
                shortlisted.append(candidate)
                selected_keys.add(key)

        decisions.append(
            {
                "request_id": request_id,
                "status": (
                    "shortlisted"
                    if shortlisted and len(resolved) == len(candidates)
                    else (
                        "no_viable_candidate"
                        if not shortlisted and len(resolved) == len(candidates)
                        else "screening_in_progress"
                    )
                ),
                "candidate_configurations": len(candidates),
                "measured_configurations": len(measured),
                "resolved_configurations": len(resolved),
                "top_k": top_k,
                "shortlisted": shortlisted,
            }
        )
    return {
        "schema_version": 1,
        "metric": RATE_FIELD,
        "selection": "fastest lowest-resource configuration and fastest overall, then intermediate GPU-count winners up to Top-K",
        "requests": len(requests),
        "complete": sum(row["status"] == "shortlisted" for row in decisions),
        "decisions": decisions,
        "configurations": configurations,
    }


def best_configuration_decisions(
    rows: Sequence[dict[str, Any]],
    requests: Sequence[dict[str, Any]],
    planned_jobs: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select the measured ZeRO/GC/MBS winner for each comparable scenario."""

    request_map = {str(row["request_id"]): row for row in requests}
    request_ids = set(request_map)
    if planned_jobs is None:
        stage_rows = [
            row
            for row in rows
            if str(row.get("request_id")) in request_ids
            and row.get("kind") == "throughput"
        ]
        configs = aggregate_configurations(stage_rows, request_map)
    else:
        stage_plans = [
            job
            for job in planned_jobs
            if str(job.get("request_id")) in request_ids
            and job.get("kind") == "throughput"
        ]
        stage_rows = active_result_rows(
            rows,
            stage_plans,
            expected_kind="throughput",
        )
        configs = aggregate_configurations(
            stage_rows,
            request_map,
            stage_plans,
        )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    fields = (
        "model_id",
        "train_type",
        "dataset_id",
        "cutoff_len",
        "target_gbs",
        "gpu_count",
        "packing",
    )
    for row in configs:
        grouped[tuple(row.get(field) for field in fields)].append(row)
    decisions = []
    for key, candidates in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        winner = select_fastest(candidates)
        decisions.append(
            {
                **dict(zip(fields, key)),
                "status": "selected" if winner else "pending_or_incomplete",
                "candidate_configurations": len(candidates),
                "complete_configurations": sum(
                    bool(row.get("complete")) for row in candidates
                ),
                "selected": winner,
            }
        )
    return {
        "schema_version": 1,
        "comparison_key": list(fields),
        "metric": RATE_FIELD,
        "decisions": decisions,
        "selected": sum(row["status"] == "selected" for row in decisions),
    }


def scaling_decisions(
    rows: Sequence[dict[str, Any]],
    requests: Sequence[dict[str, Any]],
    minimum_gain: float | None = None,
    *,
    minimum_ratio: float | None = None,
) -> dict[str, Any]:
    """Summarize strong-scaling evidence with an explicit ratio threshold.

    ``minimum_gain`` is retained only as a compatibility shim for historical
    callers and old unit fixtures.  New callers must pass ``minimum_ratio``;
    the report always emits ratio semantics.  This function still uses
    measured point rates for the historical diagnostic report.  A publishable
    scale-out claim additionally requires conservative bounds, as enforced by
    :mod:`cross_card_scaling`.
    """

    if minimum_gain is not None and minimum_ratio is not None:
        raise ValueError("pass minimum_ratio or legacy minimum_gain, not both")
    if minimum_ratio is None:
        minimum_ratio = 1.0 + (
            float(minimum_gain) if minimum_gain is not None else 0.80
        )
    try:
        minimum_ratio = float(minimum_ratio)
    except (TypeError, ValueError) as error:
        raise ValueError("minimum_ratio must be a finite number") from error
    if not math.isfinite(minimum_ratio) or minimum_ratio < 1.0:
        raise ValueError("minimum_ratio must be finite and >= 1.0")
    minimum_gain = minimum_ratio - 1.0

    request_map = {str(row["request_id"]): row for row in requests}
    request_ids = set(request_map)
    stage_rows = [row for row in rows if str(row.get("request_id")) in request_ids]
    configs = aggregate_configurations(stage_rows, request_map)
    by_request_gpu: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in configs:
        by_request_gpu[str(row.get("request_id"))][
            int(row.get("gpu_count") or 0)
        ].append(row)

    families = []
    for request in requests:
        request_id = str(request["request_id"])
        sequence = [int(value) for value in request.get("gpu_sequence", (1, 2, 4))]
        points = []
        previous: dict[str, Any] | None = None
        stopped = False
        for gpu_count in sequence:
            best = select_fastest(by_request_gpu.get(request_id, {}).get(gpu_count, []))
            if stopped:
                points.append(
                    {
                        "gpu_count": gpu_count,
                        "status": "skipped_by_early_stop",
                        "selected": None,
                        "gain_from_previous": None,
                        "ratio_from_previous": None,
                        "passes_threshold": False,
                    }
                )
                continue
            if best is None:
                points.append(
                    {
                        "gpu_count": gpu_count,
                        "status": "pending_or_infeasible",
                        "selected": None,
                        "gain_from_previous": None,
                        "ratio_from_previous": None,
                        "passes_threshold": None,
                    }
                )
                continue
            gain = None
            ratio = None
            passes = None
            if previous is not None:
                old_rate = float(previous[RATE_FIELD])
                ratio = float(best[RATE_FIELD]) / old_rate if old_rate else None
                gain = (
                    (float(best[RATE_FIELD]) - old_rate) / old_rate
                    if old_rate
                    else None
                )
                passes = ratio is not None and ratio >= minimum_ratio
                if not passes:
                    stopped = True
            points.append(
                {
                    "gpu_count": gpu_count,
                    "status": "measured",
                    "selected": best,
                    "gain_from_previous": gain,
                    "ratio_from_previous": ratio,
                    "passes_threshold": passes,
                }
            )
            previous = best
        measured = [point for point in points if point["selected"]]
        recommended = (
            measured[-2]["selected"]
            if stopped and len(measured) >= 2
            else (measured[-1]["selected"] if measured else None)
        )
        # When the current doubling fails, keep the previous card count as the
        # default recommendation; the failed point is retained as evidence.
        families.append(
            {
                "request_id": request_id,
                "minimum_ratio": minimum_ratio,
                "minimum_gain": minimum_gain,
                "threshold_semantics": "throughput_ratio",
                "conservative_bound_status": (
                    "not_available_in_legacy_point_rate_report"
                ),
                "stopped_early": stopped,
                "recommended": recommended,
                "points": points,
            }
        )
    return {
        "schema_version": 1,
        "metric": RATE_FIELD,
        "minimum_ratio_per_doubling": minimum_ratio,
        "minimum_gain_per_doubling": minimum_gain,
        "threshold_semantics": "throughput_ratio",
        "diagnostic_point_estimates_only": True,
        "families": families,
        "configurations": configs,
    }


def scaling_eligible_request_ids(
    report: dict[str, Any],
    next_gpu_count: int,
) -> set[str]:
    eligible = set()
    for family in report["families"]:
        earlier = [
            point
            for point in family["points"]
            if int(point["gpu_count"]) < next_gpu_count
        ]
        measured = [point for point in earlier if point.get("selected")]
        if not measured:
            eligible.add(str(family["request_id"]))
            continue
        latest = measured[-1]
        if latest.get("passes_threshold") is not False:
            eligible.add(str(family["request_id"]))
    return eligible


def packing_decisions(
    rows: Sequence[dict[str, Any]],
    requests: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    request_map = {str(row["request_id"]): row for row in requests}
    request_ids = set(request_map)
    stage_rows = [row for row in rows if str(row.get("request_id")) in request_ids]
    by_pair: dict[tuple[str, int], dict[bool, dict[str, Any]]] = defaultdict(dict)
    for row in stage_rows:
        if row.get("classification") != "success" or not row.get(RATE_FIELD):
            continue
        key = (str(row.get("request_id")), int(row.get("repeat", 0)))
        by_pair[key][bool(row.get("packing"))] = row

    decisions = []
    for request in requests:
        request_id = str(request["request_id"])
        planned = int(request.get("repeats", 1))
        gains = []
        baseline_mbs = None
        paired_job_ids = []
        for repeat in range(planned):
            sides = by_pair.get((request_id, repeat), {})
            off, on = sides.get(False), sides.get(True)
            if off is None or on is None:
                continue
            off_rate = float(off[RATE_FIELD])
            on_rate = float(on[RATE_FIELD])
            if off_rate <= 0 or on_rate <= 0:
                continue
            off_time = 1000.0 / off_rate
            on_time = 1000.0 / on_rate
            gains.append((off_time - on_time) / off_time)
            baseline_mbs = int(off.get("mbs") or 1)
            paired_job_ids.append([off.get("job_id"), on.get("job_id")])
        threshold = 0.10 if (baseline_mbs or 1) == 1 else 0.20
        conservative = min(gains) if len(gains) == planned else None
        static_error = float(request.get("expected_gbs_relative_error", 0.0))
        complete = len(gains) == planned
        enabled = (
            complete
            and static_error <= 0.05
            and conservative is not None
            and conservative >= threshold
        )
        decisions.append(
            {
                "request_id": request_id,
                "status": "complete" if complete else "pending_or_incomplete",
                "planned_pairs": planned,
                "complete_pairs": len(gains),
                "paired_job_ids": paired_job_ids,
                "baseline_mbs": baseline_mbs,
                "paired_time_gains": gains,
                "median_time_gain": median(gains),
                "conservative_time_gain_lower_bound": conservative,
                "lower_bound_method": (
                    "single paired time gain"
                    if planned == 1
                    else "minimum of the independent paired-run time gains"
                ),
                "decision_threshold": threshold,
                "expected_gbs_relative_error": static_error,
                "decision": "on" if enabled else "off",
            }
        )
    return {
        "schema_version": 1,
        "pairing": "same request_id and repeat index; compare seconds per 1000 logical samples",
        "decisions": decisions,
        "packing_on": sum(row["decision"] == "on" for row in decisions),
    }


def profiler_tokens(result_dir: Path, active_steps: int = 3) -> dict[str, int]:
    totals = {
        "computed_tokens": 0,
        "effective_tokens": 0,
        "computed_attention_token_pairs": 0,
        "effective_attention_token_pairs": 0,
    }
    for events_path in sorted((result_dir / "metrics").glob("events.rank*.jsonl")):
        step_rows = []
        with events_path.open(encoding="utf-8") as source:
            for line in source:
                event = json.loads(line)
                if event.get("event") == "step_end":
                    step_rows.append(event)
        for event in step_rows[-active_steps:]:
            tokens = event.get("tokens") or {}
            for field in totals:
                totals[field] += int(tokens.get(field, 0))
    return totals


def profiler_observed_flops(result_dir: Path) -> float:
    total = 0.0
    for operators_path in sorted(
        (result_dir / "metrics" / "profiler").glob("operators.rank*.json")
    ):
        for event in read_json(operators_path):
            total += float(event.get("flops") or 0.0)
    return total


def profiler_points(
    rows: Sequence[dict[str, Any]],
    models: dict[str, dict[str, Any]],
    results_dir: Path = RESULTS_DIR,
) -> list[dict[str, Any]]:
    points = []
    for row in rows:
        if row.get("classification") != "success" or not row.get("enable_profiler"):
            continue
        result_dir = results_dir / str(row["job_id"])
        observed = profiler_observed_flops(result_dir)
        tokens = profiler_tokens(result_dir, int(row.get("profiler_active_steps", 3)))
        if observed <= 0 or tokens["computed_tokens"] <= 0:
            continue
        analytic = training_flops(
            models[str(row["model_id"])],
            str(row["train_type"]),
            tokens["computed_tokens"],
            tokens["computed_attention_token_pairs"],
        )["useful_flops"]
        role = row.get("profiler_role") or (
            "holdout" if str(row["job_id"]).startswith("profhold-") else "calibration"
        )
        points.append(
            {
                "job_id": row["job_id"],
                "role": role,
                "model_id": row["model_id"],
                "train_type": row["train_type"],
                "dataset_id": row["dataset_id"],
                "target_gbs": row["target_gbs"],
                "gc": bool(row["gc"]),
                "computed_tokens": tokens["computed_tokens"],
                "computed_attention_token_pairs": tokens[
                    "computed_attention_token_pairs"
                ],
                "analytic_flops": float(analytic),
                "observed_profiler_flops": observed,
                "observed_to_analytic_ratio": observed / float(analytic),
            }
        )
    return points


def calibration_features(point: dict[str, Any]) -> list[float]:
    return [
        1.0,
        1.0 if point.get("train_type") == "lora" else 0.0,
        1.0 if point.get("gc") else 0.0,
    ]


def ridge_coefficients(
    features: Sequence[Sequence[float]], targets: Sequence[float], ridge: float = 1.0e-6
) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    penalty = np.eye(x.shape[1], dtype=float) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.pinv(x.T @ x + penalty) @ x.T @ y


def fit_profiler_calibration(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    calibration = [point for point in points if point.get("role") == "calibration"]
    holdout = [point for point in points if point.get("role") == "holdout"]
    promoted = []
    # Some training modes can be absent from the dedicated calibration matrix
    # after the memory gate (for example a known Full/ZeRO incompatibility).
    # Promote the minimum number of already independent profiler holdouts,
    # preferring points that increase feature-matrix rank, and retain all
    # remaining holdouts for an honest out-of-sample evaluation.
    while len(calibration) < 3 and holdout:
        selected = max(
            holdout,
            key=lambda point: (
                np.linalg.matrix_rank(
                    np.asarray(
                        [calibration_features(item) for item in calibration + [point]],
                        dtype=float,
                    )
                ),
                str(point.get("job_id") or ""),
            ),
        )
        holdout.remove(selected)
        promoted_point = {
            **selected,
            "role": "fallback_calibration",
            "original_role": "holdout",
        }
        calibration.append(promoted_point)
        promoted.append(promoted_point)
    if len(calibration) < 3:
        return {
            "status": "pending",
            "reason": "at least three successful profiler calibration points are required",
            "calibration_points": calibration,
            "holdout_points": holdout,
            "fallback_calibration_job_ids": [point["job_id"] for point in promoted],
        }
    coefficients = ridge_coefficients(
        [calibration_features(point) for point in calibration],
        [math.log(point["observed_to_analytic_ratio"]) for point in calibration],
    )

    def predict(point: dict[str, Any], values: np.ndarray = coefficients) -> float:
        log_ratio = float(np.asarray(calibration_features(point)) @ values)
        return float(point["analytic_flops"]) * math.exp(log_ratio)

    fitted = []
    for point in calibration:
        predicted = predict(point)
        fitted.append(
            {
                **point,
                "predicted_flops": predicted,
                "absolute_percentage_error": abs(
                    predicted - point["observed_profiler_flops"]
                )
                / point["observed_profiler_flops"],
            }
        )

    evaluated = []
    evaluation_method = "independent_holdout"
    if holdout:
        for point in holdout:
            predicted = predict(point)
            evaluated.append(
                {
                    **point,
                    "predicted_flops": predicted,
                    "absolute_percentage_error": abs(
                        predicted - point["observed_profiler_flops"]
                    )
                    / point["observed_profiler_flops"],
                }
            )
    else:
        evaluation_method = "leave_one_calibration_point_out_provisional"
        for index, point in enumerate(calibration):
            train = calibration[:index] + calibration[index + 1 :]
            loo = ridge_coefficients(
                [calibration_features(item) for item in train],
                [math.log(item["observed_to_analytic_ratio"]) for item in train],
            )
            predicted = predict(point, loo)
            evaluated.append(
                {
                    **point,
                    "predicted_flops": predicted,
                    "absolute_percentage_error": abs(
                        predicted - point["observed_profiler_flops"]
                    )
                    / point["observed_profiler_flops"],
                }
            )
    errors = [float(point["absolute_percentage_error"]) for point in evaluated]
    return {
        "status": "evaluated" if holdout else "provisional_without_independent_holdout",
        "model": "log(observed/analytic) = intercept + beta_lora + beta_gc",
        "coefficient_names": ["intercept", "is_lora", "gc_enabled"],
        "coefficients": coefficients.tolist(),
        "multipliers": {
            "full_gc_off": math.exp(float(coefficients[0])),
            "full_gc_on": math.exp(float(coefficients[0] + coefficients[2])),
            "lora_gc_off": math.exp(float(coefficients[0] + coefficients[1])),
            "lora_gc_on": math.exp(
                float(coefficients[0] + coefficients[1] + coefficients[2])
            ),
        },
        "calibration_points": fitted,
        "fallback_calibration_job_ids": [point["job_id"] for point in promoted],
        "evaluation_method": evaluation_method,
        "evaluation_points": evaluated,
        "evaluation_mape": statistics.fmean(errors) if errors else None,
        "evaluation_median_ape": median(errors),
        "evaluation_p90_ape": percentile(errors, 90),
    }


RESOURCE_FEATURE_NAMES = (
    "intercept",
    "log_parameters",
    "log_cutoff",
    "log_target_gbs",
    "log_gpu_count",
    "log_mbs",
    "is_full",
    "gc_enabled",
    "zero2",
    "zero3",
    "packing",
)


def resource_features(row: dict[str, Any], model: dict[str, Any]) -> list[float]:
    return [
        1.0,
        math.log(max(float(model["actual_parameters"]), 1.0)),
        math.log(max(float(row.get("cutoff_len") or 1), 1.0)),
        math.log(max(float(row.get("target_gbs") or 1), 1.0)),
        math.log(max(float(row.get("gpu_count") or 1), 1.0)),
        math.log(max(float(row.get("mbs") or 1), 1.0)),
        1.0 if row.get("train_type") == "full" else 0.0,
        1.0 if row.get("gc") else 0.0,
        1.0 if row.get("zero") == "zero2" else 0.0,
        1.0 if row.get("zero") == "zero3" else 0.0,
        1.0 if row.get("packing") else 0.0,
    ]


def scenario_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row.get("model_id")),
        str(row.get("train_type")),
        str(row.get("dataset_id")),
        int(row.get("target_gbs") or 0),
    )


def group_holdout_resource_evaluation(
    rows: Sequence[dict[str, Any]],
    models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Leave one complete model/train/data/GBS scenario out at a time."""

    configs = [
        row
        for row in aggregate_configurations(rows)
        if row.get("complete") and row.get(RATE_FIELD) and row.get("max_reserved_bytes")
    ]
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in configs:
        groups[scenario_key(row)].append(row)
    if len(groups) < 3 or len(configs) < len(RESOURCE_FEATURE_NAMES) + 2:
        return {
            "status": "pending",
            "reason": "at least three scenarios and enough completed configurations are required",
            "configurations": len(configs),
            "scenarios": len(groups),
        }

    throughput_errors = []
    memory_errors = []
    ranking_hits = []
    pairwise_hits = []
    fold_rows = []
    all_groups = set(groups)
    for held_key, held_rows in sorted(groups.items()):
        train_rows = [row for key in all_groups - {held_key} for row in groups[key]]
        if len(train_rows) < len(RESOURCE_FEATURE_NAMES):
            continue
        x_train = [
            resource_features(row, models[str(row["model_id"])]) for row in train_rows
        ]
        throughput_beta = ridge_coefficients(
            x_train,
            [math.log(float(row[RATE_FIELD])) for row in train_rows],
            ridge=1.0e-3,
        )
        memory_beta = ridge_coefficients(
            x_train,
            [math.log(float(row["max_reserved_bytes"])) for row in train_rows],
            ridge=1.0e-3,
        )
        predictions = []
        for row in held_rows:
            features = np.asarray(resource_features(row, models[str(row["model_id"])]))
            predicted_rate = math.exp(float(features @ throughput_beta))
            predicted_memory = math.exp(float(features @ memory_beta))
            rate_error = abs(predicted_rate - float(row[RATE_FIELD])) / float(
                row[RATE_FIELD]
            )
            memory_error = abs(
                predicted_memory - float(row["max_reserved_bytes"])
            ) / float(row["max_reserved_bytes"])
            throughput_errors.append(rate_error)
            memory_errors.append(memory_error)
            predictions.append((row, predicted_rate))
        if len(predictions) >= 2:
            observed_best = max(
                predictions, key=lambda item: float(item[0][RATE_FIELD])
            )[0]
            predicted_best = max(predictions, key=lambda item: item[1])[0]
            ranking_hits.append(
                configuration_key(observed_best) == configuration_key(predicted_best)
            )
            for left_index, (left, left_prediction) in enumerate(predictions):
                for right, right_prediction in predictions[left_index + 1 :]:
                    observed_order = float(left[RATE_FIELD]) >= float(right[RATE_FIELD])
                    predicted_order = left_prediction >= right_prediction
                    pairwise_hits.append(observed_order == predicted_order)
        fold_rows.append(
            {
                "held_out_scenario": list(held_key),
                "train_configurations": len(train_rows),
                "test_configurations": len(held_rows),
            }
        )
    return {
        "status": "evaluated" if fold_rows else "pending",
        "method": "leave-one-(model, train_type, dataset, target_gbs)-scenario-out log-linear ridge regression",
        "feature_names": list(RESOURCE_FEATURE_NAMES),
        "folds": fold_rows,
        "throughput_mape": statistics.fmean(throughput_errors)
        if throughput_errors
        else None,
        "throughput_median_ape": median(throughput_errors),
        "throughput_p90_ape": percentile(throughput_errors, 90),
        "memory_mape": statistics.fmean(memory_errors) if memory_errors else None,
        "memory_median_ape": median(memory_errors),
        "memory_p90_ape": percentile(memory_errors, 90),
        "candidate_top1_accuracy": statistics.fmean(ranking_hits)
        if ranking_hits
        else None,
        "candidate_pairwise_accuracy": statistics.fmean(pairwise_hits)
        if pairwise_hits
        else None,
    }


def load_result_rows(
    results_dir: Path = RESULTS_DIR,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    inventory = read_json(ARTIFACT_DIR / "model_inventory.json")
    models = {str(model["id"]): model for model in inventory["models"]}
    hardware = read_json(CONFIG_DIR / "hardware.json")
    rows = []
    if results_dir.exists():
        for result_dir in sorted(
            path for path in results_dir.iterdir() if path.is_dir()
        ):
            try:
                row = aggregate_result(result_dir, models, hardware)
            except (KeyError, TypeError, ZeroDivisionError, ValueError):
                # An OOM can occur before the first measured step while still
                # leaving a rank summary with zero measured seconds.  Such a
                # result is a valid negative status but has no rate label.
                status_path = result_dir / "status.json"
                rendered_path = result_dir / "rendered_run.json"
                if not status_path.is_file() or not rendered_path.is_file():
                    row = None
                else:
                    status = read_json(status_path)
                    job = read_json(rendered_path)["job"]
                    row = {
                        **{
                            key: value
                            for key, value in job.items()
                            if not isinstance(value, (dict, list))
                        },
                        **status,
                        "rank_summaries": 0,
                        "nvidia_smi_peak_mib": None,
                    }
            if row is not None:
                rows.append(row)
    return rows, models


def load_requests(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def build_reports(
    results_dir: Path = RESULTS_DIR,
    throughput_jobs_path: Path | None = None,
) -> dict[str, Any]:
    rows, models = load_result_rows(results_dir)
    throughput_requests = load_requests(MATRIX_DIR / "throughput_requests.jsonl")
    throughput_screen_jobs = load_requests(MATRIX_DIR / "throughput_screen_jobs.jsonl")
    throughput_jobs = load_requests(
        throughput_jobs_path or MATRIX_DIR / "throughput_jobs.jsonl"
    )
    scaling_requests = load_requests(MATRIX_DIR / "strong_scaling_requests.jsonl")
    scaling_jobs = load_requests(MATRIX_DIR / "scaling_candidate_jobs.jsonl")
    packing_requests = load_requests(MATRIX_DIR / "packing_pair_requests.jsonl")
    all_selection_requests = throughput_requests + scaling_requests
    experiment = read_json(CONFIG_DIR / "experiment.json")
    screening_top_k = int(
        experiment.get("matrix_policy", {}).get("throughput_shortlist_top_k", 3)
    )
    scaling_rule = experiment.get("scaling_rule") or {}
    has_ratio = "minimum_throughput_ratio_per_doubling" in scaling_rule
    has_legacy_gain = "minimum_throughput_gain_per_doubling" in scaling_rule
    if has_ratio and has_legacy_gain:
        raise ValueError(
            "scaling_rule must not define both throughput ratio and gain thresholds"
        )
    if has_ratio:
        scaling_ratio = float(
            scaling_rule["minimum_throughput_ratio_per_doubling"]
        )
    elif has_legacy_gain:
        # Read old report fixtures without allowing the old config key to
        # remain in the canonical experiment.json.
        scaling_ratio = 1.0 + float(
            scaling_rule["minimum_throughput_gain_per_doubling"]
        )
    else:
        scaling_ratio = 1.8
    final_rows = [row for row in rows if row.get("kind") != "throughput_screen"]
    active_scaling_rows = active_result_rows(
        rows,
        scaling_jobs,
        expected_kind="throughput",
    )
    formal_rows = [
        row
        for row in rows
        if str(row.get("job_id", "")).startswith(
            ("tput-", "scale-", "packoff-", "packon-")
        )
    ]
    profiler = fit_profiler_calibration(profiler_points(rows, models, results_dir))
    return {
        "schema_version": 1,
        "inputs": {"result_rows": len(rows), "results_dir": str(results_dir)},
        "throughput_screening": throughput_screening_decisions(
            rows,
            throughput_requests,
            throughput_screen_jobs,
            screening_top_k,
            active_formal_jobs=throughput_jobs,
        ),
        "throughput": throughput_decisions(
            rows,
            throughput_requests,
            throughput_jobs,
        ),
        "best_configurations": best_configuration_decisions(
            rows,
            all_selection_requests,
            throughput_jobs + scaling_jobs,
        ),
        "scaling": scaling_decisions(
            active_scaling_rows,
            scaling_requests,
            minimum_ratio=scaling_ratio,
        ),
        "packing": packing_decisions(final_rows, packing_requests),
        "profiler_calibration": profiler,
        "resource_holdout": group_holdout_resource_evaluation(formal_rows, models),
    }


def stable_holdout_score(row: dict[str, Any]) -> str:
    identity = "|".join(
        str(row.get(field))
        for field in (
            "model_id",
            "train_type",
            "dataset_id",
            "target_gbs",
            "gpu_count",
            "zero",
            "gc",
            "mbs",
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--output", type=Path, default=ARTIFACT_DIR / "stage_decisions.json"
    )
    parser.add_argument(
        "--throughput-jobs",
        type=Path,
        help=(
            "Explicit active formal plan used as trusted screening evidence; "
            "defaults to matrix/throughput_jobs.jsonl"
        ),
    )
    args = parser.parse_args()
    if args.throughput_jobs is not None and not args.throughput_jobs.is_file():
        raise SystemExit(f"Explicit formal plan is absent: {args.throughput_jobs}")
    report = build_reports(args.results_dir, args.throughput_jobs)
    write_json(args.output, report)
    write_json(
        ARTIFACT_DIR / "profiler_calibration.json", report["profiler_calibration"]
    )
    print(
        json.dumps(
            {"output": str(args.output), "inputs": report["inputs"]}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
