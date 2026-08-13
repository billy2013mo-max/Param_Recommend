#!/usr/bin/env python3
"""Fail-closed policy for thresholded cross-card scale-out.

This module contains only the decision contract.  It does not estimate
throughput and it never launches a job.  A caller must provide memory-admitted
candidate counts and conservative throughput bounds for both endpoints.  A
point estimate without those bounds is deliberately insufficient for an
automatic scale-out claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


SCHEMA = "sft_cross_card_scale_out_policy/v1"
POLICY_ID = "thresholded_doubling_then_v4b"
DEFAULT_MINIMUM_THROUGHPUT_RATIO = 1.8
DEFAULT_GPU_ORDER = (1, 2, 4)
MIN_ADMITTED_CANDIDATES = 2


def _positive_finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) and converted > 0.0 else None


def _checked_ratio(value: Any) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("minimum throughput ratio must be a finite number") from error
    if not math.isfinite(ratio) or ratio < 1.0:
        raise ValueError("minimum throughput ratio must be finite and >= 1.0")
    return ratio


def _candidate_count(summary: Mapping[str, Any]) -> int:
    raw = summary.get("admitted_candidate_count")
    if raw is None:
        raw = summary.get("candidate_count")
    if type(raw) is not int or raw < 0:
        raise ValueError(
            "card summary must contain a non-negative integer "
            "admitted_candidate_count"
        )
    return raw


def _scenario_contract(summary: Mapping[str, Any]) -> str | None:
    """Return an explicit shared-scenario/runtime binding for an endpoint.

    A ratio is only meaningful when both endpoints describe the same model,
    workload profile, GBS/cutoff, execution mechanism and runtime cohort.  The
    predictor normally supplies the combined hash; accepting the two
    component hashes keeps the evaluator usable for independently materialized
    experiment summaries.  Missing binding is deliberately not inferred from
    a matching request id or dataset name.
    """

    direct = summary.get("scenario_contract_sha256")
    if isinstance(direct, str) and direct:
        return direct
    material = summary.get("scenario_material_sha256")
    runtime = summary.get("runtime_mechanism_component_sha256") or summary.get(
        "runtime_mechanism_sha256"
    )
    if isinstance(material, str) and material and isinstance(runtime, str) and runtime:
        return f"{material}:{runtime}"
    return None


def evaluate_doubling(
    baseline: Mapping[str, Any],
    expanded: Mapping[str, Any],
    *,
    minimum_ratio: float = DEFAULT_MINIMUM_THROUGHPUT_RATIO,
    minimum_candidates: int = MIN_ADMITTED_CANDIDATES,
) -> dict[str, Any]:
    """Evaluate one N -> 2N step using a conservative throughput ratio.

    ``baseline`` and ``expanded`` are endpoint summaries.  Required fields are
    ``gpu_count``, ``memory_gate_passed``, ``admitted_candidate_count`` and
    ``best_candidate_request_id``.  The ratio is

        lower_throughput(2N) / upper_throughput(N)

    using ``conservative_lower_throughput`` and
    ``conservative_upper_throughput``.  Missing bounds, unsafe endpoints or
    fewer than two candidates all stop the decision rather than silently using
    a point estimate.
    """

    threshold = _checked_ratio(minimum_ratio)
    if type(minimum_candidates) is not int or minimum_candidates < 1:
        raise ValueError("minimum_candidates must be a positive integer")

    try:
        from_gpu = int(baseline["gpu_count"])
        to_gpu = int(expanded["gpu_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("card summaries must contain integer gpu_count") from error
    if from_gpu <= 0 or to_gpu <= 0:
        raise ValueError("gpu_count must be positive")

    result: dict[str, Any] = {
        "from_gpu": from_gpu,
        "to_gpu": to_gpu,
        "baseline_candidate_count": _candidate_count(baseline),
        "expanded_candidate_count": _candidate_count(expanded),
        "baseline_candidate": baseline.get("best_candidate_request_id"),
        "expanded_candidate": expanded.get("best_candidate_request_id"),
        "scenario_contract": None,
        "predicted_ratio": None,
        "conservative_ratio_lower": None,
        "minimum_ratio": threshold,
        "passes": False,
        "status": None,
        "reason": None,
    }

    if to_gpu != 2 * from_gpu:
        result.update(
            status="unsupported_transition",
            reason="scale-out policy only evaluates adjacent doublings",
        )
        return result

    baseline_contract = _scenario_contract(baseline)
    expanded_contract = _scenario_contract(expanded)
    if baseline_contract is None or expanded_contract is None:
        result.update(
            status="scenario_contract_unavailable",
            reason=(
                "both endpoints require an explicit shared scenario and runtime "
                "fingerprint before a cross-card ratio can be evaluated"
            ),
        )
        return result
    if baseline_contract != expanded_contract:
        result.update(
            status="scenario_contract_mismatch",
            reason="cross-card endpoints are not bound to the same scenario/runtime contract",
        )
        return result
    result["scenario_contract"] = baseline_contract

    if baseline.get("memory_gate_passed") is not True:
        result.update(
            status="baseline_not_memory_admitted",
            reason="baseline endpoint did not pass the memory gate",
        )
        return result
    if expanded.get("memory_gate_passed") is not True:
        result.update(
            status="expanded_not_memory_admitted",
            reason="expanded endpoint did not pass the memory gate",
        )
        return result
    if result["baseline_candidate_count"] < minimum_candidates:
        result.update(
            status="insufficient_candidate_evidence",
            reason="baseline has fewer than the required safe candidates",
        )
        return result
    if result["expanded_candidate_count"] < minimum_candidates:
        result.update(
            status="insufficient_candidate_evidence",
            reason="expanded endpoint has fewer than the required safe candidates",
        )
        return result
    if not baseline.get("best_candidate_request_id"):
        result.update(
            status="endpoint_selection_unavailable",
            reason="baseline best safe candidate is missing",
        )
        return result
    if not expanded.get("best_candidate_request_id"):
        result.update(
            status="endpoint_selection_unavailable",
            reason="expanded best safe candidate is missing",
        )
        return result

    baseline_point = _positive_finite(baseline.get("predicted_throughput"))
    expanded_point = _positive_finite(expanded.get("predicted_throughput"))
    if baseline_point is not None and expanded_point is not None:
        result["predicted_ratio"] = expanded_point / baseline_point

    baseline_upper = _positive_finite(
        baseline.get("conservative_upper_throughput")
    )
    expanded_lower = _positive_finite(
        expanded.get("conservative_lower_throughput")
    )
    if baseline_upper is None or expanded_lower is None:
        result.update(
            status="conservative_bound_unavailable",
            reason=(
                "both endpoints require finite positive conservative throughput "
                "bounds; point estimates are diagnostic only"
            ),
        )
        return result

    conservative_ratio = expanded_lower / baseline_upper
    if not math.isfinite(conservative_ratio) or conservative_ratio <= 0.0:
        result.update(
            status="conservative_bound_invalid",
            reason="computed conservative ratio is not finite and positive",
        )
        return result
    result["conservative_ratio_lower"] = conservative_ratio
    if conservative_ratio >= threshold:
        result.update(
            status="passed",
            reason="conservative lower ratio cleared the minimum threshold",
            passes=True,
        )
    else:
        result.update(
            status="threshold_not_cleared",
            reason="conservative lower ratio is below the minimum threshold",
        )
    return result


def evaluate_scale_out_sequence(
    summaries: Sequence[Mapping[str, Any]],
    *,
    gpu_order: Sequence[int] = DEFAULT_GPU_ORDER,
    minimum_ratio: float = DEFAULT_MINIMUM_THROUGHPUT_RATIO,
    minimum_candidates: int = MIN_ADMITTED_CANDIDATES,
) -> dict[str, Any]:
    """Evaluate repeated adjacent doublings and stop at the first failure."""

    threshold = _checked_ratio(minimum_ratio)
    order = tuple(int(value) for value in gpu_order)
    if not order or any(value <= 0 for value in order):
        raise ValueError("gpu_order must contain positive card counts")
    if len(set(order)) != len(order):
        raise ValueError("gpu_order must not contain duplicates")

    by_gpu: dict[int, Mapping[str, Any]] = {}
    for summary in summaries:
        if not isinstance(summary, Mapping):
            raise ValueError("every card summary must be an object")
        gpu_count = summary.get("gpu_count")
        if type(gpu_count) is not int or gpu_count <= 0:
            raise ValueError("card summary gpu_count must be a positive integer")
        if gpu_count in by_gpu:
            raise ValueError(f"duplicate card summary for gpu_count={gpu_count}")
        _candidate_count(summary)
        by_gpu[gpu_count] = summary

    minimum_gpu: int | None = None
    for gpu_count in order:
        summary = by_gpu.get(gpu_count)
        if summary is None:
            continue
        if (
            summary.get("memory_gate_passed") is True
            and _candidate_count(summary) >= minimum_candidates
            and summary.get("best_candidate_request_id")
        ):
            minimum_gpu = gpu_count
            break

    if minimum_gpu is None:
        return {
            "schema": SCHEMA,
            "selection_policy": POLICY_ID,
            "minimum_ratio": threshold,
            "minimum_admitted_gpu_count": None,
            "recommended_gpu_count": None,
            "scaling_steps": [],
            "scaling_stop_reason": "no_admitted_candidate_evidence",
            "automatic_execution_allowed": False,
        }

    recommended = minimum_gpu
    steps: list[dict[str, Any]] = []
    stop_reason = "no_next_doubling_in_gpu_order"
    current_index = order.index(minimum_gpu)
    while current_index + 1 < len(order):
        next_gpu = order[current_index + 1]
        baseline = by_gpu.get(recommended)
        expanded = by_gpu.get(next_gpu)
        if baseline is None or expanded is None:
            stop_reason = "endpoint_evidence_unavailable"
            break
        step = evaluate_doubling(
            baseline,
            expanded,
            minimum_ratio=threshold,
            minimum_candidates=minimum_candidates,
        )
        steps.append(step)
        if not step["passes"]:
            stop_reason = str(step["status"])
            break
        recommended = next_gpu
        current_index += 1

    if current_index + 1 >= len(order) and not steps:
        stop_reason = "no_next_doubling_in_gpu_order"
    elif current_index + 1 >= len(order) and steps and steps[-1]["passes"]:
        stop_reason = "all_available_doublings_passed"

    return {
        "schema": SCHEMA,
        "selection_policy": POLICY_ID,
        "minimum_ratio": threshold,
        "minimum_admitted_gpu_count": minimum_gpu,
        "recommended_gpu_count": recommended,
        "scaling_steps": steps,
        "scaling_stop_reason": stop_reason,
        "automatic_execution_allowed": False,
    }
