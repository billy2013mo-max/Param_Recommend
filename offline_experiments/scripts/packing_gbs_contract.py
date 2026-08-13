#!/usr/bin/env python3
"""Packed sample-GBS contract derived from a cached pack-count distribution.

The center of ``samples_per_pack`` determines integer GA and epoch-average
sample GBS.  The upper tail is a separate safety quantity: if one global
microstep can already exceed the target tolerance while GA is at its floor,
no integer GA can repair that cutoff.  Keeping these two quantities separate
prevents a utilization confidence interval from being mistaken for the
within-dataset pack-count distribution.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


SCHEMA = "sft_packing_sample_gbs_contract/v2"


def _positive(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _positive(value, name)
    if int(result) != result:
        raise ValueError(f"{name} must be a positive integer")
    return int(result)


def derive_packing_gbs_contract(
    *,
    target_gbs: float,
    data_parallel: int,
    samples_per_pack: Mapping[str, Any],
    epsilon_gbs: float = 0.10,
    maximum_center_relative_error: float = 0.05,
) -> dict[str, Any]:
    """Derive center geometry and the GA-floor upper-tail hard gate.

    ``samples_per_pack`` is a compact upload-time cache produced by the exact
    tokenizer/template/packer fingerprint.  It must contain the epoch mean and
    an empirical P99; recommendation-time raw lengths are not required.
    """

    target = _positive(target_gbs, "target_gbs")
    dp = _positive_int(data_parallel, "data_parallel")
    epsilon = float(epsilon_gbs)
    center_tolerance = float(maximum_center_relative_error)
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon_gbs must be finite and nonnegative")
    if not math.isfinite(center_tolerance) or center_tolerance < 0:
        raise ValueError("maximum_center_relative_error must be finite and nonnegative")

    center = _positive(samples_per_pack.get("mean"), "samples_per_pack.mean")
    p99 = _positive(samples_per_pack.get("p99"), "samples_per_pack.p99")
    maximum = _positive(
        samples_per_pack.get("maximum", p99),
        "samples_per_pack.maximum",
    )
    # A mean is not mathematically required to be below P99: fewer than one
    # percent of very large packs can pull the mean above the empirical P99.
    # The maximum, however, must dominate both statistics.
    if maximum < p99 or maximum < center:
        raise ValueError("samples_per_pack.maximum must be >= mean and p99")

    raw_ga = target / (dp * center)
    candidates = sorted({max(1, math.floor(raw_ga)), max(1, math.ceil(raw_ga))})
    ga = min(
        candidates,
        key=lambda value: (abs(dp * value * center - target), value),
    )
    expected = dp * ga * center
    center_error = abs(expected - target) / target

    # GA cannot be smaller than one.  This bound is intentionally per
    # microstep and therefore does not multiply by the selected GA.
    global_microstep_p99 = dp * p99
    global_microstep_maximum = dp * maximum
    allowed_microstep_upper = target * (1.0 + epsilon)
    controllable = global_microstep_p99 <= allowed_microstep_upper
    center_representable = center_error <= center_tolerance
    reason_codes: list[str] = []
    if not center_representable:
        reason_codes.append("expected_gbs_center_not_integer_representable")
    if not controllable:
        reason_codes.append("p99_global_microstep_exceeds_target_tolerance_at_ga_floor")

    return {
        "schema": SCHEMA,
        "input_contract": "cached_upload_time_pack_count_distribution",
        "recommendation_time_raw_lengths_read": False,
        "target_gbs": target,
        "data_parallel": dp,
        "samples_per_pack": {
            "mean": center,
            "p99": p99,
            "maximum": maximum,
        },
        "raw_gradient_accumulation_steps": raw_ga,
        "gradient_accumulation_steps": ga,
        "expected_epoch_sample_gbs": expected,
        "expected_epoch_sample_gbs_relative_error": center_error,
        "global_microstep_sample_gbs": {
            "center": dp * center,
            "p99": global_microstep_p99,
            "maximum": global_microstep_maximum,
            "allowed_p99_upper": allowed_microstep_upper,
        },
        "gates": {
            "epsilon_gbs": epsilon,
            "maximum_center_relative_error": center_tolerance,
            "center_integer_representable": center_representable,
            "gbs_controllable_at_ga_floor": controllable,
            "candidate_admissible": center_representable and controllable,
            "reason_codes": reason_codes,
        },
    }


def derive_from_cached_packing_features(
    features: Mapping[str, Any],
    *,
    target_gbs: float,
    data_parallel: int,
    epsilon_gbs: float = 0.10,
    maximum_center_relative_error: float = 0.05,
) -> dict[str, Any]:
    """Build the v2 contract from compact features cached at upload time."""

    distribution = features.get("samples_per_pack")
    if not isinstance(distribution, Mapping):
        raise ValueError("cached packing features lack samples_per_pack distribution")
    if distribution.get("mean") is None:
        distribution = {**distribution, "mean": features.get("mean_samples_per_pack")}
    return derive_packing_gbs_contract(
        target_gbs=target_gbs,
        data_parallel=data_parallel,
        samples_per_pack=distribution,
        epsilon_gbs=epsilon_gbs,
        maximum_center_relative_error=maximum_center_relative_error,
    )
