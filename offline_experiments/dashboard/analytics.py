"""Derived metrics and cross-run comparisons for the live dashboard."""

from __future__ import annotations

import math
import re
import statistics
import time
from collections import defaultdict
from typing import Any, Iterable

from scripts.flops import result_flops
from scripts.gpu_telemetry import aggregate_gpu_telemetry

from .store import SKIPPED_STATUSES, DashboardStore


PHASE_LABELS = {
    "memory": "显存边界",
    "throughput": "吞吐与 MFU",
    "scaling": "多卡扩展",
    "packing": "Packing 对照",
    "profiler": "Profiler 校准",
    "validation": "验证实验",
    "preflight": "静态检查",
}
PHASE_ORDER = ("memory", "throughput", "scaling", "packing", "profiler")
PROGRESS_TERMINAL_STATUSES = {
    "success",
    "oom",
    "failed",
    "incomplete_metrics",
    *SKIPPED_STATUSES,
}
FAILED_RUN_STATUSES = {
    "failed",
    "incomplete_metrics",
    "approval_rejected",
    "launcher_failed",
    "scheduler_interrupted",
}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    return ordered[left] * (right - position) + ordered[right] * (position - left)


def safe_rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_step_rows(
    rows: list[dict[str, Any]],
    job: dict[str, Any],
    model: dict[str, Any] | None,
    hardware: dict[str, Any],
    rolling_steps: int = 20,
) -> dict[str, Any]:
    """Aggregate distributed rank events by optimizer step.

    Rank token counts are summed while elapsed time and memory are max-reduced,
    matching the completed-result collector's global interpretation.
    """
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[int(row["global_step"])].append(row)
    step_rows = []
    for step, ranks in sorted(by_step.items()):
        tokens = {
            key: sum(int(row.get(key, 0) or 0) for row in ranks)
            for key in (
                "computed_tokens", "effective_tokens", "label_tokens", "logical_samples",
                "physical_batches", "computed_attention_pairs", "effective_attention_pairs",
            )
        }
        step_rows.append(
            {
                "global_step": step,
                "time_unix": max(float(row.get("time_unix") or 0) for row in ranks),
                "step_seconds": max(float(row.get("step_seconds") or 0) for row in ranks),
                "is_warmup": all(bool(row.get("is_warmup")) for row in ranks),
                "allocated_bytes": max(int(row.get("allocated_bytes") or 0) for row in ranks),
                "reserved_bytes": max(int(row.get("reserved_bytes") or 0) for row in ranks),
                "max_allocated_bytes": max(int(row.get("max_allocated_bytes") or 0) for row in ranks),
                "max_reserved_bytes": max(int(row.get("max_reserved_bytes") or 0) for row in ranks),
                **tokens,
            }
        )
    measured = [row for row in step_rows if not row["is_warmup"]]
    sample = (measured or step_rows)[-rolling_steps:]
    seconds = sum(row["step_seconds"] for row in sample)
    totals = {
        key: sum(row[key] for row in sample)
        for key in (
            "computed_tokens", "effective_tokens", "label_tokens", "logical_samples",
            "physical_batches", "computed_attention_pairs", "effective_attention_pairs",
        )
    }
    times = [row["step_seconds"] for row in sample]
    metrics: dict[str, Any] = {
        "current_step": max(by_step, default=0),
        "observed_steps": len(step_rows),
        "measured_steps": len(measured),
        "window_steps": len(sample),
        "window_seconds": seconds,
        "step_p50_seconds": percentile(times, 0.5),
        "step_p90_seconds": percentile(times, 0.9),
        "computed_tokens_per_second": safe_rate(totals["computed_tokens"], seconds),
        "effective_tokens_per_second": safe_rate(totals["effective_tokens"], seconds),
        "samples_per_second": safe_rate(totals["logical_samples"], seconds),
        "padding_efficiency": safe_rate(totals["effective_tokens"], totals["computed_tokens"]),
        "label_tokens_per_second": safe_rate(totals["label_tokens"], seconds),
        "max_allocated_bytes": max((row["max_allocated_bytes"] for row in step_rows), default=0),
        "max_reserved_bytes": max((row["max_reserved_bytes"] for row in step_rows), default=0),
        "window_totals": totals,
    }
    if model and seconds and totals["computed_tokens"]:
        counters = {
            "computed_tokens": totals["computed_tokens"],
            "effective_tokens": totals["effective_tokens"],
            "computed_attention_token_pairs": totals["computed_attention_pairs"],
            "effective_attention_token_pairs": totals["effective_attention_pairs"],
        }
        flops = result_flops(model, job.get("train_type") or "lora", counters)
        gpu_count = max(1, int(job.get("gpu_count") or 1))
        device_peak = float(
            hardware.get("bf16_dense_peak_flops_per_second_for_mfu")
            or hardware.get("bf16_dense_peak_flops_per_second_for_standard_mfu")
            or 989.5e12
        )
        public_reference_peak = float(
            hardware.get("bf16_public_reference_dense_peak_flops_per_second")
            or hardware.get("clock_normalized_dense_peak_flops_per_second")
            or device_peak
        )
        metrics.update(
            {
                "mfu": flops["computed_useful_flops"] / (seconds * gpu_count * device_peak),
                "effective_mfu": flops["effective_useful_flops"] / (seconds * gpu_count * device_peak),
                "public_reference_mfu": flops["computed_useful_flops"]
                / (seconds * gpu_count * public_reference_peak),
                "flops_calibration": flops["calibration_status"],
            }
        )
    return metrics


_MEMORY_UNITS = {"kib": 1024, "mib": 1024**2, "gib": 1024**3}


def _oom_bytes(error: str, prefix: str) -> int | None:
    match = re.search(
        rf"{prefix}\s+([0-9]+(?:\.[0-9]+)?)\s*(KiB|MiB|GiB)", error, flags=re.IGNORECASE
    )
    if not match:
        return None
    return round(float(match.group(1)) * _MEMORY_UNITS[match.group(2).lower()])


def aggregate_failure_rows(
    rows: list[dict[str, Any]], current: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Merge failure-time allocator state with peaks from completed steps."""
    if not rows:
        return {}
    current = current or {}
    failure_max_allocated = max(int(row.get("max_allocated_bytes") or 0) for row in rows)
    failure_max_reserved = max(int(row.get("max_reserved_bytes") or 0) for row in rows)
    parsed = []
    for row in rows:
        error = str(row.get("error") or "")
        values = {
            "oom_requested_bytes": _oom_bytes(error, r"Tried to allocate"),
            "oom_total_capacity_bytes": _oom_bytes(error, r"total capacity of"),
            "oom_free_bytes": _oom_bytes(error, r"of which"),
            "oom_pytorch_allocated_bytes": _oom_bytes(error, r"Of the allocated memory"),
            "oom_reserved_unallocated_bytes": _oom_bytes(error, r"and"),
        }
        if values["oom_requested_bytes"] is not None:
            parsed.append(values)
    oom = max(
        parsed,
        key=lambda row: (row.get("oom_requested_bytes") or 0) - (row.get("oom_free_bytes") or 0),
        default={},
    )
    return {
        "failure_event_count": len(rows),
        "failure_allocated_bytes": max(int(row.get("allocated_bytes") or 0) for row in rows),
        "failure_reserved_bytes": max(int(row.get("reserved_bytes") or 0) for row in rows),
        "failure_max_allocated_bytes": failure_max_allocated,
        "failure_max_reserved_bytes": failure_max_reserved,
        "max_allocated_bytes": max(int(current.get("max_allocated_bytes") or 0), failure_max_allocated),
        "max_reserved_bytes": max(int(current.get("max_reserved_bytes") or 0), failure_max_reserved),
        **oom,
    }


def aggregate_gpu_rows(
    rows: list[dict[str, Any]],
    hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return aggregate_gpu_telemetry(rows, hardware)


def _median(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.median(clean) if clean else None


def _variation(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2:
        return None
    mean = statistics.fmean(clean)
    return statistics.stdev(clean) / mean if mean else None


def _config_key(job: dict[str, Any], include_request_id: bool = True) -> tuple[Any, ...]:
    fields = [
        "campaign_id", "hardware_id", "model_id", "train_type", "dataset_id", "cutoff_len",
        "gpu_count", "zero_name", "gc", "mbs", "target_gbs", "packing",
    ]
    if include_request_id:
        fields.insert(2, "request_id")
    return tuple(job.get(field) for field in fields)


def _scenario_key(job: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        job.get(field)
        for field in (
            "campaign_id",
            "hardware_id",
            "model_id",
            "train_type",
            "dataset_id",
            "target_gbs",
        )
    )


def throughput_stage_progress(
    jobs: list[dict[str, Any]],
    plans: dict[str, dict[str, Any]],
    active_screen_job_ids: set[str] | None = None,
    active_formal_job_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Report screening coverage and formal Top-K work with matching units."""
    terminal_statuses = PROGRESS_TERMINAL_STATUSES
    throughput_jobs = [row for row in jobs if row.get("phase") == "throughput"]
    screen_jobs = [
        row
        for row in throughput_jobs
        if row.get("kind") == "throughput_screen"
        and (
            active_screen_job_ids is None
            or str(row.get("job_id")) in active_screen_job_ids
        )
    ]
    # Historical formal measurements may satisfy an equivalent current screen
    # candidate, but they must not enlarge or complete the current formal
    # stage.  Keep the reuse and active universes separate for that reason.
    historical_formal_jobs = [
        row
        for row in throughput_jobs
        if row.get("kind") == "throughput"
    ]
    formal_jobs = [
        row
        for row in historical_formal_jobs
        if (
            active_formal_job_ids is None
            or str(row.get("job_id")) in active_formal_job_ids
        )
    ]

    def grouped(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
        result: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            result[_config_key(row, include_request_id=False)].append(row)
        return result

    screen_groups = grouped(screen_jobs)
    historical_formal_groups = grouped(historical_formal_jobs)
    formal_groups = grouped(formal_jobs)
    screen_keys = set(screen_groups)
    screen_terminal = {
        key
        for key, rows in screen_groups.items()
        if any(str(row.get("status") or "planned") in terminal_statuses for row in rows)
    }
    historical_formal_healthy = {
        key
        for key, rows in historical_formal_groups.items()
        if any(row.get("status") == "success" for row in rows)
    }
    formal_healthy = {
        key
        for key, rows in formal_groups.items()
        # A terminal success is enough for stage progress. Detailed metrics can
        # still be backfilled asynchronously after a cold dashboard restart.
        if any(row.get("status") == "success" for row in rows)
    }
    formal_terminal = {
        key
        for key, rows in formal_groups.items()
        if any(str(row.get("status") or "planned") in terminal_statuses for row in rows)
    }
    screen_completed = screen_terminal | (screen_keys & historical_formal_healthy)
    screen_running = {
        key
        for key, rows in screen_groups.items()
        if key not in screen_completed
        and any(row.get("status") == "running" for row in rows)
    }
    reused_formal = (screen_keys & historical_formal_healthy) - screen_terminal

    screen_by_scenario: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    formal_by_scenario: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    for key, rows in screen_groups.items():
        screen_by_scenario[_scenario_key(rows[0])].add(key)
    for key, rows in formal_groups.items():
        formal_by_scenario[_scenario_key(rows[0])].add(key)

    if active_formal_job_ids is not None:
        # A published formal matrix is the exact current plan.  Count physical
        # configurations (repeat runs share a key), not an inferred Top-K and
        # not terminal rows retained from a previous materialization.
        eligible_formal_keys = set(formal_groups)
        formal_total = len(eligible_formal_keys)
        formal_completed = len(eligible_formal_keys & formal_terminal)
        formal_successful = len(eligible_formal_keys & formal_healthy)
        formal_running = sum(
            key not in formal_terminal
            and any(row.get("status") == "running" for row in formal_groups[key])
            for key in eligible_formal_keys
        )
    else:
        # Compatibility for indexes created before active formal manifests
        # existed: retain the former inferred Top-K behavior.
        formal_total = 0
        formal_completed = 0
        formal_successful = 0
        formal_running = 0
        eligible_formal_keys: set[tuple[Any, ...]] = set()
        scenarios = (
            set(screen_by_scenario)
            if active_screen_job_ids is not None and screen_by_scenario
            else set(screen_by_scenario) | set(formal_by_scenario)
        )
        for scenario in scenarios:
            screen_candidates = screen_by_scenario.get(scenario, set())
            formal_candidates = formal_by_scenario.get(scenario, set())
            if screen_candidates:
                campaign_id = str(scenario[0] or "")
                top_k = max(
                    1,
                    int(
                        (plans.get(campaign_id) or {}).get("throughput_shortlist_top_k")
                        or 3
                    ),
                )
                target = min(top_k, len(screen_candidates))
                eligible = formal_candidates & screen_candidates
            else:
                target = len(formal_candidates)
                eligible = formal_candidates
            formal_total += target
            eligible_formal_keys.update(eligible)
            formal_completed += min(target, len(eligible & formal_terminal))
            formal_successful += min(target, len(eligible & formal_healthy))
            unresolved_running = {
                key
                for key in eligible
                if key not in formal_terminal
                and any(row.get("status") == "running" for row in formal_groups[key])
            }
            formal_running += min(
                max(target - len(eligible & formal_terminal), 0),
                len(unresolved_running),
            )

    screen_total = len(screen_keys)
    screen_done = len(screen_completed)
    current_stage = (
        "screening"
        if screen_total and (screen_running or screen_done < screen_total)
        else "formal"
    )
    return {
        "current_stage": current_stage,
        "screening": {
            "completed": screen_done,
            "total": screen_total,
            "executed": len(screen_terminal),
            "reused_formal": len(reused_formal),
            "running": len(screen_running),
            "remaining": max(screen_total - screen_done - len(screen_running), 0),
        },
        "formal": {
            "completed": formal_completed,
            "total": formal_total,
            "successful": formal_successful,
            "running": formal_running,
            "remaining": max(formal_total - formal_completed - formal_running, 0),
            "materialized": len(eligible_formal_keys),
            "waiting_for_screening": bool(screen_total and screen_done < screen_total),
            "total_is_top_k_upper_bound": bool(
                screen_total and active_formal_job_ids is None
            ),
        },
    }


def aggregate_repeats(
    jobs: list[dict[str, Any]], include_request_id: bool = True
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        groups[_config_key(job, include_request_id)].append(job)
    result = []
    metric_fields = (
        "samples_per_second", "effective_tokens_per_second", "computed_tokens_per_second",
        "padding_efficiency", "step_p50_seconds", "step_p90_seconds", "mfu", "effective_mfu",
        "clock_adjusted_mfu", "clock_adjusted_effective_mfu",
        "max_allocated_bytes", "max_reserved_bytes", "nvidia_smi_peak_mib",
        "average_power_w", "energy_wh", "busy_clock_p5_mhz", "busy_clock_p50_mhz",
        "busy_clock_p95_mhz", "sm_clock_reference_mhz", "sm_clock_spec_max_mhz",
        "busy_clock_p50_ratio", "busy_clock_p50_to_spec_max_ratio",
        "busy_clock_below_95pct_fraction", "busy_clock_below_90pct_fraction",
        "power_limit_busy_fraction", "busy_power_p50_w", "busy_power_p95_w",
        "temperature_max_c", "busy_temperature_p95_c",
        "sw_power_cap_busy_fraction", "sw_thermal_slowdown_busy_fraction",
        "hw_thermal_slowdown_busy_fraction",
    )
    clock_status_priority = {
        "insufficient_data": 0,
        "normal": 1,
        "downclocked": 2,
        "power_limited": 3,
        "thermal_limited": 4,
    }
    for rows in groups.values():
        successful = [row for row in rows if row.get("status") == "success" and row.get("metrics")]
        base = rows[0]
        item = {
            key: base.get(key)
            for key in (
                "campaign_id", "hardware_id", "gpu_type", "request_id", "model_id", "train_type", "dataset_id", "cutoff_len", "gpu_count",
                "zero_name", "gc", "mbs", "target_gbs", "packing",
            )
        }
        status_counts = dict(
            sorted(
                (
                    status,
                    sum(
                        1
                        for row in rows
                        if str(row.get("status") or "planned") == status
                    ),
                )
                for status in {
                    str(row.get("status") or "planned") for row in rows
                }
            )
        )
        item.update(
            {
                "runs": len(rows),
                "successful_runs": len(successful),
                "failed_runs": sum(
                    count
                    for status, count in status_counts.items()
                    if status in FAILED_RUN_STATUSES
                ),
                "skipped_runs": sum(
                    count
                    for status, count in status_counts.items()
                    if status in SKIPPED_STATUSES
                ),
                "job_ids": [row["job_id"] for row in rows],
                "request_ids": sorted({str(row["request_id"]) for row in rows if row.get("request_id")}),
                "phases": sorted({str(row["phase"]) for row in rows if row.get("phase")}),
                "kinds": sorted({str(row["kind"]) for row in rows if row.get("kind")}),
                "status_counts": status_counts,
            }
        )
        for field in metric_fields:
            values = [(row.get("metrics") or {}).get(field) for row in successful]
            item[field] = _median(values)
            if field in {"samples_per_second", "effective_tokens_per_second", "step_p50_seconds"}:
                clean = [float(value) for value in values if value is not None]
                item[f"{field}_min"] = min(clean) if clean else None
                item[f"{field}_max"] = max(clean) if clean else None
                item[f"{field}_cv"] = _variation(clean)
        clock_statuses = [
            str((row.get("metrics") or {}).get("clock_status"))
            for row in successful
            if (row.get("metrics") or {}).get("clock_status")
        ]
        item["clock_status_counts"] = {
            status: clock_statuses.count(status) for status in sorted(set(clock_statuses))
        }
        item["clock_status"] = max(
            clock_statuses,
            key=lambda status: clock_status_priority.get(status, -1),
            default="insufficient_data",
        )
        samples_per_second = item.get("samples_per_second")
        item["estimated_epoch_seconds_1000_samples"] = 1000.0 / samples_per_second if samples_per_second else None
        epoch_seconds = item["estimated_epoch_seconds_1000_samples"]
        item["gpu_hours_per_1000_samples"] = (
            epoch_seconds * int(item.get("gpu_count") or 1) / 3600.0 if epoch_seconds else None
        )
        result.append(item)
    return sorted(result, key=lambda row: (row.get("model_id") or "", row.get("dataset_id") or "", row.get("gpu_count") or 0))


class DashboardAnalytics:
    def __init__(self, store: DashboardStore):
        self.store = store

    def _active_throughput_screen_job_ids(
        self, campaigns: list[dict[str, Any]]
    ) -> set[str] | None:
        active: set[str] = set()
        registry_found = False
        for campaign in campaigns:
            campaign_id = str(campaign.get("campaign_id") or "")
            job_ids = self.store.get_meta(
                f"campaign:{campaign_id}:active_throughput_screen_job_ids",
                None,
            )
            if job_ids is None:
                continue
            registry_found = True
            active.update(str(job_id) for job_id in job_ids)
        # A present-but-empty registry means no active screening batch is scoping
        # progress; treat that like an absent registry ("count all") rather than
        # "restrict to nothing", which would zero the screening progress.
        return active if (registry_found and active) else None

    def _active_throughput_formal_job_ids(
        self, campaigns: list[dict[str, Any]]
    ) -> set[str] | None:
        active: set[str] = set()
        registry_found = False
        for campaign in campaigns:
            campaign_id = str(campaign.get("campaign_id") or "")
            job_ids = self.store.get_meta(
                f"campaign:{campaign_id}:active_throughput_formal_job_ids",
                None,
            )
            if job_ids is None:
                continue
            registry_found = True
            active.update(str(job_id) for job_id in job_ids)
        # Empty formal registry -> "count all", not "restrict to nothing".
        return active if (registry_found and active) else None

    def _active_manifest_job_ids(
        self, campaigns: list[dict[str, Any]]
    ) -> dict[str, set[str]]:
        active: dict[str, set[str]] = {}
        for campaign in campaigns:
            campaign_id = str(campaign.get("campaign_id") or "")
            job_ids = self.store.get_meta(
                f"campaign:{campaign_id}:active_manifest_job_ids",
                None,
            )
            if isinstance(job_ids, list):
                active[campaign_id] = {str(job_id) for job_id in job_ids}
        return active

    def overview(self, campaign_id: str | None = None) -> dict[str, Any]:
        all_campaigns = self.store.get_meta("campaigns", [])
        campaigns = [
            row for row in all_campaigns if not campaign_id or row.get("campaign_id") == campaign_id
        ]
        if campaign_id and not campaigns:
            campaigns = [{"campaign_id": campaign_id}]
        plans = {
            row["campaign_id"]: self.store.get_meta(f"campaign:{row['campaign_id']}:design_summary", {})
            for row in campaigns
        }
        pipeline_states = {
            row["campaign_id"]: self.store.get_meta(
                f"campaign:{row['campaign_id']}:pipeline_state", {}
            )
            for row in campaigns
        }
        active_manifest_job_ids = self._active_manifest_job_ids(campaigns)
        indexed_jobs = self.store.all_jobs(campaign_id=campaign_id)
        progress_jobs = [
            job
            for job in indexed_jobs
            # A campaign with no active manifest (absent key OR an empty set) is
            # not currently scoped to a running batch, so all of its indexed jobs
            # count toward progress.  Only restrict to the manifest membership
            # when that manifest is non-empty.
            if not active_manifest_job_ids.get(str(job.get("campaign_id") or ""))
            or str(job.get("job_id") or "")
            in active_manifest_job_ids[str(job.get("campaign_id") or "")]
        ]
        phase_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        runs_by_status: dict[str, int] = defaultdict(int)
        for job in progress_jobs:
            phase = str(job.get("phase") or "other")
            status = str(job.get("status") or "planned")
            record_type = str(job.get("record_type") or "run")
            phase_counts[phase][status] += 1
            phase_counts[phase]["indexed"] += 1
            phase_counts[phase][f"{record_type}_indexed"] += 1
            phase_counts[phase][f"{record_type}_{status}"] += 1
            if record_type == "run":
                runs_by_status[status] += 1
        run_jobs = [
            job for job in progress_jobs if job.get("record_type") == "run"
        ]
        running = [
            job
            for job in run_jobs
            if job["status"] == "running"
        ]
        active_screen_job_ids = self._active_throughput_screen_job_ids(campaigns)
        active_formal_job_ids = self._active_throughput_formal_job_ids(campaigns)
        throughput_progress = throughput_stage_progress(
            run_jobs, plans, active_screen_job_ids, active_formal_job_ids
        )
        pipeline_currents = [
            state.get("current") or {}
            for state in pipeline_states.values()
            if state
        ]
        all_pipelines_complete = bool(pipeline_currents) and all(
            current.get("stage") == "analysis"
            and current.get("status") == "complete"
            for current in pipeline_currents
        ) and not active_manifest_job_ids
        active_manifest_size = sum(
            len(job_ids) for job_ids in active_manifest_job_ids.values()
        )
        active_manifest_jobs = [
            job
            for job in run_jobs
            if str(job.get("campaign_id") or "") in active_manifest_job_ids
        ]
        active_manifest_complete = (
            bool(active_manifest_size)
            and len(active_manifest_jobs) == active_manifest_size
            and all(
                str(job.get("status") or "planned")
                in PROGRESS_TERMINAL_STATUSES
                for job in active_manifest_jobs
            )
        )
        if all_pipelines_complete:
            # The controller's completed analysis state is authoritative even
            # while a cold dashboard index is still backfilling historical
            # metrics used to identify reused formal measurements.
            screening = throughput_progress["screening"]
            screening["completed"] = screening["total"]
            screening["reused_formal"] = max(
                screening["total"] - screening["executed"],
                screening["reused_formal"],
            )
            screening["running"] = 0
            screening["remaining"] = 0
            formal = throughput_progress["formal"]
            formal["completed"] = formal["total"]
            formal["running"] = 0
            formal["remaining"] = 0
            formal["waiting_for_screening"] = False
            throughput_progress["current_stage"] = "formal"
        current_phase = next((job["phase"] for job in running), None)
        if current_phase is None and not all_pipelines_complete:
            for phase in PHASE_ORDER:
                values = phase_counts.get(phase, {})
                if values.get("planned", 0) or values.get("running", 0):
                    current_phase = phase
                    break
        latest_gpus = self.store.latest_gpus(campaign_id)
        now = time.time()
        for gpu in latest_gpus:
            gpu["stale_seconds"] = max(0.0, now - float(gpu["time_unix"]))
            gpu["live"] = gpu["stale_seconds"] <= 10 and gpu.get("status") == "running"
        configured_gpu_ids = sorted(
            {
                int(gpu_id)
                for campaign in campaigns
                for gpu_id in campaign.get("gpu_ids", [])
            }
        )
        gpu_ids = [int(gpu_id) for gpu_id in configured_gpu_ids]
        if not gpu_ids:
            gpu_ids = sorted({int(gpu["gpu_index"]) for gpu in latest_gpus}) or [0, 1, 2, 3]
        scheduler_key = f"campaign:{campaign_id}:scheduler_state" if campaign_id else "scheduler_state"
        scheduler = self.store.get_meta(scheduler_key, {})
        phase_label = PHASE_LABELS.get(
            current_phase or "",
            "全部实验完成"
            if all_pipelines_complete or active_manifest_complete
            else "等待实验",
        )
        if current_phase == "throughput":
            phase_label = (
                "吞吐初筛"
                if throughput_progress["current_stage"] == "screening"
                else "正式复验"
            )
        if not running and scheduler.get("status") != "running" and current_phase:
            phase_label = f"待执行：{phase_label}"
        queued = next(
            (
                state.get("current") or {}
                for state in pipeline_states.values()
                if (state.get("current") or {}).get("status")
                == "queued_waiting_approval"
            ),
            None,
        ) if not active_manifest_job_ids else None
        if not running and queued:
            current_phase = "throughput"
            phase_label = "待审批：吞吐初筛"
        phase_plan_fields = {
            "memory": "memory_boundary_families",
            "scaling": "strong_scaling_families",
            "packing": "packing_pair_families",
            "profiler": "profiler_calibration_configurations",
        }
        phase_progress: dict[str, dict[str, int]] = {
            phase: {"planned": 0, "completed": 0}
            for phase in phase_plan_fields
        }
        campaign_ids = {
            str(row.get("campaign_id") or "") for row in campaigns
        } | {str(job.get("campaign_id") or "") for job in progress_jobs}
        for current_campaign_id in campaign_ids:
            campaign_rows = [
                job
                for job in progress_jobs
                if str(job.get("campaign_id") or "") == current_campaign_id
            ]
            manifest_is_active = current_campaign_id in active_manifest_job_ids
            design = plans.get(current_campaign_id) or {}
            for phase, plan_field in phase_plan_fields.items():
                rows = [job for job in campaign_rows if job.get("phase") == phase]
                if manifest_is_active:
                    planned = len(rows)
                    completed = sum(
                        str(job.get("status") or "planned")
                        in PROGRESS_TERMINAL_STATUSES
                        for job in rows
                    )
                else:
                    planned = int(design.get(plan_field) or len(rows))
                    if phase == "memory":
                        completed = sum(
                            job.get("record_type") == "family"
                            and job.get("status")
                            in {"boundary_found", "infeasible"}
                            for job in rows
                        )
                    else:
                        completed = sum(
                            job.get("record_type") == "run"
                            and str(job.get("status") or "planned")
                            in PROGRESS_TERMINAL_STATUSES
                            for job in rows
                        )
                phase_progress[phase]["planned"] += planned
                phase_progress[phase]["completed"] += completed

        return {
            "revision": self.store.revision(),
            "selected_campaign_id": campaign_id,
            "generated_unix": now,
            "current_phase": current_phase,
            "current_phase_label": phase_label,
            "runs_by_status": dict(runs_by_status),
            "phase_counts": {phase: dict(values) for phase, values in phase_counts.items()},
            "phase_progress": phase_progress,
            "active_manifest_job_ids": {
                key: sorted(value) for key, value in active_manifest_job_ids.items()
            },
            "active_manifest_complete": active_manifest_complete,
            "throughput_progress": throughput_progress,
            "plans": plans,
            "pipeline_states": pipeline_states,
            "campaigns": campaigns,
            "running_jobs": running,
            "gpu_ids": gpu_ids,
            "gpus": latest_gpus,
            "recent_events": self.store.recent_scheduler_events(12, campaign_id),
            "scheduler": scheduler,
            "last_scan_unix": self.store.get_meta("last_scan_unix", 0),
        }

    def memory(self, campaign_id: str | None = None) -> dict[str, Any]:
        families = self.store.all_jobs(phase="memory", record_type="family", campaign_id=campaign_id)
        campaign_memory = {
            row.get("campaign_id"): row.get("memory_total_bytes")
            for row in self.store.get_meta("campaigns", [])
        }
        rows = []
        for family in families:
            boundary = family.get("boundary") or {}
            if not boundary and family.get("status") == "planned":
                continue
            max_mbs = boundary.get("max_feasible_mbs")
            trial_id = f"{family['job_id']}-mbs{max_mbs}" if max_mbs is not None else None
            trial = self.store.get_job(trial_id) if trial_id else None
            metrics = (trial or {}).get("metrics") or {}
            usable_memory_gib = safe_rate(
                (family.get("raw_job") or {}).get("memory_total_bytes_per_gpu")
                or campaign_memory.get(family.get("campaign_id"))
                or 0,
                1024**3,
            )
            rows.append(
                {
                    "family_job_id": family["job_id"],
                    "campaign_id": family.get("campaign_id"),
                    "hardware_id": family.get("hardware_id"),
                    "gpu_type": family.get("gpu_type"),
                    "memory_total_gib": round(usable_memory_gib or 0),
                    "memory_usable_gib": usable_memory_gib,
                    "model_id": family.get("model_id"),
                    "train_type": family.get("train_type"),
                    "dataset_id": family.get("dataset_id"),
                    "cutoff_len": family.get("cutoff_len"),
                    "gpu_count": family.get("gpu_count"),
                    "zero": family.get("zero_name"),
                    "gc": family.get("gc"),
                    "status": family.get("status"),
                    "max_feasible_mbs": max_mbs,
                    "first_failed_mbs": boundary.get("first_failed_mbs"),
                    "mfu": metrics.get("mfu"),
                    "samples_per_second": metrics.get("samples_per_second"),
                    "max_allocated_gib": safe_rate(metrics.get("max_allocated_bytes", 0), 1024**3),
                    "max_reserved_gib": safe_rate(metrics.get("max_reserved_bytes", 0), 1024**3),
                    "nvidia_smi_peak_gib": safe_rate(metrics.get("nvidia_smi_peak_mib", 0), 1024),
                }
            )
        return {"rows": rows, "families_indexed": len(families), "boundaries_found": len(rows)}

    def throughput(self, phase: str = "throughput", campaign_id: str | None = None) -> dict[str, Any]:
        jobs = self.store.all_jobs(phase=phase, record_type="run", campaign_id=campaign_id)
        screening_runs = 0
        if phase == "throughput":
            screening_runs = sum(job.get("kind") == "throughput_screen" for job in jobs)
            jobs = [job for job in jobs if job.get("kind") != "throughput_screen"]
        configurations = aggregate_repeats(jobs)
        return {
            "phase": phase,
            "runs": len(jobs),
            "screening_runs": screening_runs,
            "configurations": configurations,
        }

    def scaling(self, campaign_id: str | None = None) -> dict[str, Any]:
        configs = self.throughput("scaling", campaign_id)["configurations"]
        by_request: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for row in configs:
            if row.get("successful_runs"):
                by_request[row.get("request_id") or "unknown"][int(row.get("gpu_count") or 0)].append(row)
        families = []
        for request_id, cards in by_request.items():
            best = {
                gpu_count: max(rows, key=lambda row: row.get("samples_per_second") or 0)
                for gpu_count, rows in cards.items()
            }
            baseline = (best.get(1) or next(iter(best.values()), {})).get("samples_per_second")
            points = []
            previous = None
            for gpu_count, row in sorted(best.items()):
                throughput = row.get("samples_per_second")
                speedup = safe_rate(throughput or 0, baseline or 0)
                gain = safe_rate((throughput or 0) - (previous or 0), previous or 0) if previous else None
                points.append(
                    {
                        **row,
                        "speedup": speedup,
                        "parallel_efficiency": safe_rate(speedup or 0, gpu_count),
                        "gain_from_previous": gain,
                        "passes_70_percent_rule": gain is None or gain >= 0.70,
                    }
                )
                previous = throughput
            families.append({"request_id": request_id, "points": points})
        return {"families": families, "configurations": configs}

    def packing(self, campaign_id: str | None = None) -> dict[str, Any]:
        jobs = self.store.all_jobs(phase="packing", record_type="run", campaign_id=campaign_id)
        configs = aggregate_repeats(jobs)
        repeat_pairs: dict[tuple[str, int], dict[bool, dict[str, Any]]] = defaultdict(dict)
        for job in jobs:
            if job.get("status") == "success" and job.get("metrics"):
                key = (str(job.get("request_id") or "unknown"), int(job.get("repeat_index") or 0))
                repeat_pairs[key][bool(job.get("packing"))] = job
        paired_gains: dict[str, list[float]] = defaultdict(list)
        for (request_id, _), sides in repeat_pairs.items():
            off, on = sides.get(False), sides.get(True)
            if not off or not on:
                continue
            off_rate = (off.get("metrics") or {}).get("samples_per_second")
            on_rate = (on.get("metrics") or {}).get("samples_per_second")
            if off_rate and on_rate:
                off_time, on_time = 1000.0 / off_rate, 1000.0 / on_rate
                paired_gains[request_id].append((off_time - on_time) / off_time)
        by_request: dict[str, dict[bool, dict[str, Any]]] = defaultdict(dict)
        for row in configs:
            by_request[row.get("request_id") or "unknown"][bool(row.get("packing"))] = row
        pairs = []
        for request_id, sides in by_request.items():
            off, on = sides.get(False), sides.get(True)
            if not off or not on:
                continue
            off_time, on_time = off.get("estimated_epoch_seconds_1000_samples"), on.get("estimated_epoch_seconds_1000_samples")
            gain = safe_rate((off_time or 0) - (on_time or 0), off_time or 0)
            threshold = 0.10 if int(off.get("mbs") or 1) == 1 else 0.20
            gains = paired_gains.get(request_id) or ([] if gain is None else [gain])
            conservative_gain = min(gains) if gains else None
            pairs.append(
                {
                    "request_id": request_id,
                    "no_packing": off,
                    "packing": on,
                    "time_gain": gain,
                    "paired_time_gains": gains,
                    "median_paired_time_gain": statistics.median(gains) if gains else None,
                    "worst_paired_time_gain": conservative_gain,
                    "best_paired_time_gain": max(gains) if gains else None,
                    "decision_threshold": threshold,
                    "decision": "on" if conservative_gain is not None and conservative_gain >= threshold else "off",
                }
            )
        return {"pairs": pairs, "unpaired_configurations": len(configs) - len(pairs) * 2}

    def recommendations(self, campaign_id: str | None = None) -> dict[str, Any]:
        throughput_jobs = self.store.all_jobs(
            phase="throughput", record_type="run", campaign_id=campaign_id
        )
        all_campaigns = self.store.get_meta("campaigns", [])
        campaigns = [
            row
            for row in all_campaigns
            if not campaign_id or row.get("campaign_id") == campaign_id
        ]
        active_screen_job_ids = self._active_throughput_screen_job_ids(campaigns)
        active_formal_job_ids = self._active_throughput_formal_job_ids(campaigns)
        screen_jobs = [
            row
            for row in throughput_jobs
            if row.get("kind") == "throughput_screen"
            and (
                active_screen_job_ids is None
                or str(row.get("job_id")) in active_screen_job_ids
            )
        ]
        formal_jobs = [
            row
            for row in throughput_jobs
            if row.get("kind") == "throughput"
            and (
                active_formal_job_ids is None
                or str(row.get("job_id")) in active_formal_job_ids
            )
        ]
        screens = aggregate_repeats(screen_jobs, include_request_id=False)
        formals = aggregate_repeats(
            formal_jobs,
            include_request_id=False,
        )
        screen_map = {_config_key(row, include_request_id=False): row for row in screens}
        formal_map = {_config_key(row, include_request_id=False): row for row in formals}
        candidates = []
        for config_key in sorted(
            set(screen_map) | set(formal_map),
            key=lambda values: tuple(str(value) for value in values),
        ):
            screen = screen_map.get(config_key)
            formal = formal_map.get(config_key)
            row = dict(formal or screen or {})
            screen_counts = (screen or {}).get("status_counts") or {}
            formal_counts = (formal or {}).get("status_counts") or {}
            screen_successes = int((screen or {}).get("successful_runs") or 0)
            formal_successes = int((formal or {}).get("successful_runs") or 0)
            screen_resolved = bool(
                screen_successes
                or sum(
                    int(screen_counts.get(status) or 0)
                    for status in PROGRESS_TERMINAL_STATUSES
                )
            )
            formal_resolved = bool(
                formal_successes
                or sum(
                    int(formal_counts.get(status) or 0)
                    for status in PROGRESS_TERMINAL_STATUSES
                )
            )
            row.update(
                {
                    "screen_successful_runs": screen_successes,
                    "formal_successful_runs": formal_successes,
                    "screen_status_counts": screen_counts,
                    "formal_status_counts": formal_counts,
                    "screen_resolved": screen_resolved or formal_successes > 0,
                    "has_screen_job": screen is not None,
                    "is_shortlisted": formal is not None,
                    "formal_resolved": formal_resolved,
                    "screen_metrics": (
                        {
                            key: screen.get(key)
                            for key in (
                                "samples_per_second",
                                "mfu",
                                "clock_adjusted_mfu",
                                "clock_status",
                                "busy_clock_p50_mhz",
                                "sm_clock_reference_mhz",
                                "sm_clock_spec_max_mhz",
                                "busy_clock_p50_to_spec_max_ratio",
                                "power_limit_busy_fraction",
                                "busy_temperature_p95_c",
                                "estimated_epoch_seconds_1000_samples",
                                "gpu_hours_per_1000_samples",
                            )
                        }
                        if screen
                        else None
                    ),
                }
            )
            candidates.append(row)
        scenarios: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            key = (
                row.get("campaign_id"),
                row.get("hardware_id"),
                row.get("model_id"),
                row.get("train_type"),
                row.get("dataset_id"),
                row.get("target_gbs"),
            )
            scenarios[key].append(row)
        result = []
        for key, rows in scenarios.items():
            rows.sort(
                key=lambda row: (
                    int(row.get("gpu_count") or 1),
                    str(row.get("zero_name") or ""),
                    bool(row.get("gc")),
                    int(row.get("mbs") or 0),
                )
            )
            measured = [row for row in rows if row.get("formal_successful_runs")]
            has_screening = any(row["has_screen_job"] for row in rows)
            lowest_resource = None
            fastest = None
            default = None
            if measured:
                lowest_gpu = min(int(row.get("gpu_count") or 1) for row in measured)
                lowest_resource = min(
                    (row for row in measured if int(row.get("gpu_count") or 1) == lowest_gpu),
                    key=lambda row: row.get("estimated_epoch_seconds_1000_samples") or float("inf"),
                )
                fastest = min(
                    measured,
                    key=lambda row: row.get("estimated_epoch_seconds_1000_samples") or float("inf"),
                )
                default = lowest_resource
                higher_gpu_counts = sorted(
                    {
                        int(row.get("gpu_count") or 1)
                        for row in measured
                        if int(row.get("gpu_count") or 1) > lowest_gpu
                    }
                )
                for gpu_count in higher_gpu_counts:
                    contender = min(
                        (row for row in measured if int(row.get("gpu_count") or 1) == gpu_count),
                        key=lambda row: row.get("estimated_epoch_seconds_1000_samples") or float("inf"),
                    )
                    gain = safe_rate(
                        (contender.get("samples_per_second") or 0)
                        - (default.get("samples_per_second") or 0),
                        default.get("samples_per_second") or 0,
                    )
                    if gain is not None and gain >= 0.70:
                        default = contender
                    else:
                        break
            result.append(
                {
                    "campaign_id": key[0], "hardware_id": key[1], "model_id": key[2],
                    "train_type": key[3], "dataset_id": key[4], "target_gbs": key[5],
                    "default": default, "lowest_resource": lowest_resource, "fastest": fastest,
                    "candidates": rows,
                    "candidate_count": len(rows),
                    "measured_candidate_count": len(measured),
                    "screened_candidate_count": (
                        sum(bool(row["screen_resolved"]) for row in rows)
                        if has_screening
                        else len(rows)
                    ),
                    "shortlisted_candidate_count": sum(bool(row["is_shortlisted"]) for row in rows),
                    "pending_candidate_count": sum(
                        bool(row["is_shortlisted"] and not row["formal_resolved"])
                        for row in rows
                    ),
                    "status": (
                        "screening_in_progress"
                        if has_screening and not all(row["screen_resolved"] for row in rows)
                        else (
                            "formal_in_progress"
                            if any(
                                row["is_shortlisted"] and not row["formal_resolved"]
                                for row in rows
                            )
                            else "comparison_complete"
                        )
                    ),
                }
            )
        return {
            "scenarios": result,
            "note": (
                "Selections use completed candidates from the current formal manifest; "
                "pending configurations are included for progress visibility."
            ),
        }
