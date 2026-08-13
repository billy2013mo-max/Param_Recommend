#!/usr/bin/env python3
"""Parse and summarize backward-compatible NVIDIA GPU telemetry."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


BUSY_UTILIZATION_PERCENT = 90.0
POWER_LIMIT_THRESHOLD_RATIO = 0.98

CSV_FIELD_TO_KEY = {
    "memory.used": "memory_used_mib",
    "utilization.gpu": "utilization_gpu",
    "power.draw": "power_draw_w",
    "clocks.sm": "clock_sm_mhz",
    "temperature.gpu": "temperature_gpu_c",
    "fan.speed": "fan_speed_percent",
    "clocks_event_reasons.sw_thermal_slowdown": "sw_thermal_slowdown_active",
    "clocks_event_reasons.hw_thermal_slowdown": "hw_thermal_slowdown_active",
    "clocks_event_reasons.sw_power_cap": "sw_power_cap_active",
}
FLAG_KEYS = {
    "sw_thermal_slowdown_active",
    "hw_thermal_slowdown_active",
    "sw_power_cap_active",
}
NUMERIC_KEYS = set(CSV_FIELD_TO_KEY.values()) - FLAG_KEYS
MISSING_VALUES = {"", "n/a", "[n/a]", "[not supported]", "not supported"}


def optional_float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in MISSING_VALUES:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_active_flag(value: Any) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in MISSING_VALUES:
        return None
    if normalized in {"active", "true", "yes", "1"}:
        return 1
    if normalized in {"not active", "false", "no", "0"}:
        return 0
    return None


def parse_nvidia_timestamp(value: str, fallback: float) -> float:
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return fallback


def normalize_nvidia_smi_row(
    row: Mapping[str, Any],
    fallback_time: float,
) -> dict[str, Any] | None:
    """Normalize both the historical six-column and extended CSV formats."""

    timestamp = str(row.get("timestamp") or "").strip()
    try:
        gpu_index = int(str(row.get("index") or "").strip())
    except ValueError:
        return None
    normalized: dict[str, Any] = {
        "timestamp_text": timestamp,
        "time_unix": parse_nvidia_timestamp(timestamp, fallback_time),
        "gpu_index": gpu_index,
    }
    for csv_field, key in CSV_FIELD_TO_KEY.items():
        value = row.get(csv_field)
        normalized[key] = (
            optional_active_flag(value) if key in FLAG_KEYS else optional_float(value)
        )
    return normalized


def read_nvidia_smi_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    fallback = path.stat().st_mtime
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            normalized = normalize_nvidia_smi_row(row, fallback)
            if normalized is not None:
                rows.append(normalized)
    return rows


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    left, right = math.floor(position), math.ceil(position)
    if left == right:
        return ordered[left]
    return ordered[left] * (right - position) + ordered[right] * (position - left)


def _fraction(rows: list[dict[str, Any]], key: str) -> float | None:
    known = [int(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(known) if known else None


def _hardware_float(hardware: Mapping[str, Any] | None, key: str) -> float | None:
    value = optional_float((hardware or {}).get(key))
    return value if value is not None and value > 0 else None


def aggregate_gpu_telemetry(
    rows: list[dict[str, Any]],
    hardware: Mapping[str, Any] | None = None,
    busy_utilization_percent: float = BUSY_UTILIZATION_PERCENT,
) -> dict[str, Any]:
    """Aggregate telemetry and classify clock behavior without failing a job."""

    if not rows:
        return {}
    memory = [
        float(row["memory_used_mib"])
        for row in rows
        if row.get("memory_used_mib") is not None
    ]
    utilization = [
        float(row["utilization_gpu"])
        for row in rows
        if row.get("utilization_gpu") is not None
    ]
    power = [
        float(row["power_draw_w"])
        for row in rows
        if row.get("power_draw_w") is not None
    ]
    clocks = [
        float(row["clock_sm_mhz"])
        for row in rows
        if row.get("clock_sm_mhz") is not None
    ]
    temperatures = [
        float(row["temperature_gpu_c"])
        for row in rows
        if row.get("temperature_gpu_c") is not None
    ]
    fans = [
        float(row["fan_speed_percent"])
        for row in rows
        if row.get("fan_speed_percent") is not None
    ]
    busy_rows = [
        row
        for row in rows
        if float(row.get("utilization_gpu") or 0) >= busy_utilization_percent
    ]
    busy_clocks = [
        float(row["clock_sm_mhz"])
        for row in busy_rows
        if row.get("clock_sm_mhz") is not None
    ]
    busy_power = [
        float(row["power_draw_w"])
        for row in busy_rows
        if row.get("power_draw_w") is not None
    ]
    busy_temperatures = [
        float(row["temperature_gpu_c"])
        for row in busy_rows
        if row.get("temperature_gpu_c") is not None
    ]

    spec_max_clock = _hardware_float(
        hardware, "max_sm_clock_mhz_reported_by_nvidia_smi"
    )
    reference_clock = _hardware_float(hardware, "healthy_busy_sm_clock_mhz")
    reference_source = (
        "configured_healthy_busy_clock" if reference_clock is not None else None
    )
    power_limit = _hardware_float(hardware, "power_limit_w")
    p5 = _percentile(busy_clocks, 0.05)
    p50 = _percentile(busy_clocks, 0.50)
    p95 = _percentile(busy_clocks, 0.95)
    # NVIDIA's reported maximum SM clock is a hardware ceiling, not the
    # workload's expected sustained clock.  For example, an RTX 4090 may report
    # 3105 MHz while a healthy, fully utilized training workload sustains about
    # 2640 MHz.  Keep the ceiling as raw context, but never infer downclocking
    # from it.  Clock-only inference and adjusted MFU require an independently
    # calibrated healthy-load reference; explicit NVIDIA limiter flags remain
    # authoritative even when only one busy sample is available.
    p50_ratio = p50 / reference_clock if p50 is not None and reference_clock else None
    p50_spec_max_ratio = (
        p50 / spec_max_clock
        if p50 is not None and spec_max_clock
        else None
    )
    below_95_fraction = (
        statistics.fmean(value < reference_clock * 0.95 for value in busy_clocks)
        if busy_clocks and reference_clock
        else None
    )
    below_90_fraction = (
        statistics.fmean(value < reference_clock * 0.90 for value in busy_clocks)
        if busy_clocks and reference_clock
        else None
    )
    power_limit_fraction = (
        statistics.fmean(
            value >= power_limit * POWER_LIMIT_THRESHOLD_RATIO for value in busy_power
        )
        if busy_power and power_limit
        else None
    )
    sw_power_fraction = _fraction(busy_rows, "sw_power_cap_active")
    sw_thermal_fraction = _fraction(busy_rows, "sw_thermal_slowdown_active")
    hw_thermal_fraction = _fraction(busy_rows, "hw_thermal_slowdown_active")
    reason_support = {
        key: any(row.get(key) is not None for row in busy_rows)
        for key in sorted(FLAG_KEYS)
    }
    reason_data_available = any(reason_support.values())
    reason_data_complete = all(reason_support.values())

    enough_clock_data = len(busy_clocks) >= 3 and reference_clock is not None
    thermal_observed = bool((sw_thermal_fraction or 0) > 0 or (hw_thermal_fraction or 0) > 0)
    power_reason_observed = bool((sw_power_fraction or 0) > 0)
    inferred_power_limit = bool(
        power_limit_fraction is not None
        and power_limit_fraction >= 0.20
        and below_95_fraction is not None
        and below_95_fraction >= 0.20
    )
    clock_only_drop = bool(
        enough_clock_data
        and (
            (p50_ratio is not None and p50_ratio < 0.95)
            or (below_90_fraction is not None and below_90_fraction >= 0.10)
        )
    )
    if thermal_observed:
        clock_status, clock_status_source = "thermal_limited", "observed_throttle_reason"
    elif power_reason_observed:
        clock_status, clock_status_source = "power_limited", "observed_throttle_reason"
    elif enough_clock_data and inferred_power_limit:
        clock_status, clock_status_source = "power_limited", "inferred_power_cap"
    elif enough_clock_data and clock_only_drop:
        clock_status, clock_status_source = "downclocked", "clock_only"
    elif reason_data_complete:
        clock_status, clock_status_source = "normal", "no_observed_queried_limiter"
    elif reason_data_available:
        clock_status, clock_status_source = "insufficient_data", "partial_throttle_reason_support"
    else:
        clock_status, clock_status_source = "insufficient_data", "missing_throttle_reason"

    by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_gpu[int(row["gpu_index"])].append(row)
    energy_wh = 0.0
    for samples in by_gpu.values():
        samples.sort(key=lambda row: row["time_unix"])
        for left, right in zip(samples, samples[1:]):
            dt = max(
                0.0,
                min(10.0, float(right["time_unix"]) - float(left["time_unix"])),
            )
            if left.get("power_draw_w") is not None and right.get("power_draw_w") is not None:
                energy_wh += (
                    float(left["power_draw_w"]) + float(right["power_draw_w"])
                ) * 0.5 * dt / 3600.0

    return {
        "telemetry_schema_version": 3,
        "nvidia_smi_peak_mib": max(memory, default=None),
        "average_gpu_utilization": statistics.fmean(utilization) if utilization else None,
        "average_power_w": statistics.fmean(power) if power else None,
        "average_clock_mhz": statistics.fmean(clocks) if clocks else None,
        "energy_wh": energy_wh,
        "gpu_samples": len(rows),
        "busy_gpu_samples": len(busy_rows),
        "busy_utilization_threshold_percent": busy_utilization_percent,
        "sm_clock_reference_mhz": reference_clock,
        "sm_clock_reference_source": reference_source,
        "sm_clock_spec_max_mhz": spec_max_clock,
        "power_limit_w": power_limit,
        "busy_clock_p5_mhz": p5,
        "busy_clock_p50_mhz": p50,
        "busy_clock_p95_mhz": p95,
        "busy_clock_p50_ratio": p50_ratio,
        "busy_clock_p50_to_spec_max_ratio": p50_spec_max_ratio,
        "busy_clock_below_95pct_fraction": below_95_fraction,
        "busy_clock_below_90pct_fraction": below_90_fraction,
        "busy_power_p50_w": _percentile(busy_power, 0.50),
        "busy_power_p95_w": _percentile(busy_power, 0.95),
        "power_limit_busy_fraction": power_limit_fraction,
        "temperature_max_c": max(temperatures, default=None),
        "busy_temperature_p95_c": _percentile(busy_temperatures, 0.95),
        "fan_speed_max_percent": max(fans, default=None),
        "sw_power_cap_busy_fraction": sw_power_fraction,
        "sw_thermal_slowdown_busy_fraction": sw_thermal_fraction,
        "hw_thermal_slowdown_busy_fraction": hw_thermal_fraction,
        "throttle_reason_data_available": reason_data_available,
        "throttle_reason_data_complete": reason_data_complete,
        "throttle_reason_support": reason_support,
        "clock_status": clock_status,
        "clock_status_source": clock_status_source,
        "downclock_detected": clock_status
        in {"power_limited", "thermal_limited", "downclocked"},
    }


def add_clock_adjusted_mfu(metrics: dict[str, Any]) -> None:
    """Add diagnostic MFU estimates relative to a calibrated healthy clock."""

    ratio = optional_float(metrics.get("busy_clock_p50_ratio"))
    if ratio is None or ratio <= 0:
        return
    for source, target in (
        ("mfu", "clock_adjusted_mfu"),
        ("effective_mfu", "clock_adjusted_effective_mfu"),
    ):
        value = optional_float(metrics.get(source))
        if value is not None:
            metrics[target] = value / ratio
