#!/usr/bin/env python3
"""Prepare a fail-closed formal-throughput delta after a validated screen delta.

The command is deliberately offline-only.  It compares an explicitly preserved
formal baseline with a newly materialized formal matrix, proves that only
requests affected by a validated screen delta changed, and writes only formal
runs that still need execution.  A healthy historical formal run is recorded as
an explicit reuse; any existing but unhealthy formal attempt blocks the delta.

This tool never launches training and refuses to write either live approval
file or any input matrix.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from common import (
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)


CONFIG_KEY_FIELDS = (
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
RUN_KEY_FIELDS = (*CONFIG_KEY_FIELDS, "repeat")
FAMILY_KEY_FIELDS = (
    "model_id",
    "train_type",
    "dataset_id",
    "cutoff_len",
    "gpu_count",
    "zero",
    "gc",
    "packing",
)
REQUEST_SHAPE_FIELDS = (
    "model_id",
    "train_type",
    "dataset_id",
    "cutoff_len",
    "target_gbs",
)
ABSOLUTE_MAX_DELTA_RUNS = 36


def config_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in CONFIG_KEY_FIELDS)


def run_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in RUN_KEY_FIELDS)


def family_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in FAMILY_KEY_FIELDS)


def key_record(fields: Sequence[str], key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(fields, key))


def config_key_record(key: tuple[Any, ...]) -> dict[str, Any]:
    return key_record(CONFIG_KEY_FIELDS, key)


def run_key_record(key: tuple[Any, ...]) -> dict[str, Any]:
    return key_record(RUN_KEY_FIELDS, key)


def duplicate_values(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def duplicate_key_rows(
    rows: Iterable[dict[str, Any]],
    key_fn: Any,
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(str(row.get("job_id")))
    return [
        {"key": key_record(fields, key), "job_ids": job_ids}
        for key, job_ids in grouped.items()
        if len(job_ids) > 1
    ]


def valid_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def all_true(mapping: Any) -> bool:
    return (
        isinstance(mapping, dict)
        and bool(mapping)
        and all(value is True for value in mapping.values())
    )


def all_empty(mapping: Any) -> bool:
    return isinstance(mapping, dict) and all(not value for value in mapping.values())


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def expected_formal_job_id(request: dict[str, Any], job: dict[str, Any]) -> str:
    identity: dict[str, Any] = {
        "request_id": request["request_id"],
        "gpu_count": job["gpu_count"],
        "zero": job["zero"],
        "gc": job["gc"],
        "mbs": job["mbs"],
        "target_gbs": request["target_gbs"],
        "packing": job["packing"],
        "repeat": job["repeat"],
    }
    if request.get("campaign_id") is not None:
        identity = {
            "campaign_id": request["campaign_id"],
            "hardware_id": request.get("hardware_id"),
            **identity,
        }
    return stable_id("tput", identity)


def formal_job_shape_errors(
    job: dict[str, Any],
    request: dict[str, Any] | None,
    experiment: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    missing = [
        field
        for field in (*RUN_KEY_FIELDS, "job_id", "kind", "fidelity")
        if field not in job
    ]
    if missing:
        errors.append(f"missing fields: {missing}")
    if job.get("kind") != "throughput":
        errors.append("kind must be throughput")
    if job.get("fidelity") != "formal":
        errors.append("fidelity must be formal")
    if not isinstance(job.get("job_id"), str) or not str(job.get("job_id")).startswith(
        "tput-"
    ):
        errors.append("job_id must use the tput- prefix")
    if not isinstance(job.get("request_id"), str) or not job.get("request_id"):
        errors.append("request_id must be a non-empty string")
    for field in ("cutoff_len", "gpu_count", "mbs", "target_gbs"):
        if not valid_positive_int(job.get(field)):
            errors.append(f"{field} must be a positive integer")
    if not isinstance(job.get("gc"), bool):
        errors.append("gc must be boolean")
    if job.get("packing") is not False:
        errors.append("formal throughput delta must be unpacked")
    if not isinstance(job.get("repeat"), int) or isinstance(job.get("repeat"), bool):
        errors.append("repeat must be an integer")

    if request is None:
        errors.append("request_id is absent from throughput requests")
        return errors

    for field in REQUEST_SHAPE_FIELDS:
        if job.get(field) != request.get(field):
            errors.append(f"{field} does not match the request")
    for field in ("campaign_id", "phase_id", "hardware_id", "gpu_type"):
        if request.get(field) is not None and job.get(field) != request.get(field):
            errors.append(f"{field} does not match the request scope")

    repeats = request.get("repeats")
    if not valid_positive_int(repeats):
        errors.append("request repeats must be a positive integer")
    elif not isinstance(job.get("repeat"), int) or not 0 <= int(job["repeat"]) < int(
        repeats
    ):
        errors.append("repeat is outside the request repeat range")

    measurement = experiment.get("measurement") or {}
    request_warmup = request.get("warmup_steps")
    request_measure = request.get("measure_steps")
    if request_warmup != measurement.get("throughput_warmup_steps"):
        errors.append("request warmup_steps differs from formal measurement policy")
    if request_measure != measurement.get("throughput_measure_steps"):
        errors.append("request measure_steps differs from formal measurement policy")
    if job.get("warmup_steps") != request_warmup:
        errors.append("warmup_steps does not match the formal request")
    if job.get("measure_steps") != request_measure:
        errors.append("measure_steps does not match the formal request")

    gpu_count = job.get("gpu_count")
    mbs = job.get("mbs")
    target_gbs = job.get("target_gbs")
    if (
        valid_positive_int(gpu_count)
        and valid_positive_int(mbs)
        and valid_positive_int(target_gbs)
        and int(target_gbs) % (int(gpu_count) * int(mbs)) != 0
    ):
        errors.append("target_gbs must be divisible by gpu_count * mbs")

    scope = experiment.get("training_scope") or {}
    if valid_positive_int(gpu_count) and int(gpu_count) not in set(
        scope.get("gpu_counts") or ()
    ):
        errors.append("gpu_count is outside the configured scope")
    zero_options = (scope.get("zero_by_gpu_count") or {}).get(str(gpu_count)) or ()
    if job.get("zero") not in zero_options:
        errors.append("zero strategy is outside the configured gpu_count policy")
    expected_parallel = request.get(
        "parallel_class", "exclusive_pool" if gpu_count == 4 else "gpu_partitionable"
    )
    if job.get("parallel_class") != expected_parallel:
        errors.append("parallel_class does not match the request")
    if job.get("requires_external_node_idle") is not False:
        errors.append("formal delta jobs must not require external-node idleness")

    try:
        expected_id = expected_formal_job_id(request, job)
    except (KeyError, TypeError, ValueError):
        expected_id = None
    if expected_id is None or job.get("job_id") != expected_id:
        errors.append("job_id is not the deterministic formal run ID")
    return errors


def screen_delta_scope(report: dict[str, Any]) -> tuple[list[str], dict[str, bool]]:
    candidates = list(report.get("delta_candidates") or ())
    candidate_ids = [str(row.get("job_id")) for row in candidates]
    allowed_ids = list(
        (report.get("freeze_preparation") or {}).get("allowed_job_ids") or ()
    )
    request_ids: list[str] = []
    physical_keys_present = True
    for row in candidates:
        physical = row.get("physical_key")
        request_id = physical.get("request_id") if isinstance(physical, dict) else None
        if not isinstance(request_id, str) or not request_id:
            physical_keys_present = False
            continue
        if request_id not in request_ids:
            request_ids.append(request_id)
    checks = {
        "screen_delta_report_passed": report.get("all_passed") is True,
        "screen_delta_checks_recomputed": all_true(report.get("checks")),
        "screen_delta_has_no_violations": all_empty(report.get("violations")),
        "screen_delta_candidates_present": bool(candidates),
        "screen_delta_candidate_rows_passed": bool(candidates)
        and all(
            row.get("all_passed") is True and all_true(row.get("checks"))
            for row in candidates
        ),
        "screen_delta_allowed_ids_exact": bool(candidate_ids)
        and candidate_ids == allowed_ids,
        "screen_delta_physical_request_ids_present": physical_keys_present
        and bool(request_ids),
    }
    return request_ids, checks


def validate_boundary_summary(
    family: dict[str, Any], summary: dict[str, Any]
) -> dict[str, bool]:
    candidates = [int(value) for value in family.get("mbs_candidates") or ()]
    trials = list(summary.get("trials") or ())
    trial_mbs = [int(row.get("mbs") or 0) for row in trials]
    classifications = [str(row.get("classification")) for row in trials]
    first_oom_index = next(
        (
            index
            for index, classification in enumerate(classifications)
            if classification == "oom"
        ),
        None,
    )
    expected_prefix = (
        candidates if first_oom_index is None else candidates[: first_oom_index + 1]
    )
    successful = [
        mbs
        for mbs, classification in zip(trial_mbs, classifications)
        if classification == "success"
    ]
    expected_max = max(successful) if successful else None
    expected_failed = (
        trial_mbs[first_oom_index] if first_oom_index is not None else None
    )
    return {
        "family_id_matches": summary.get("family_job_id") == family.get("job_id"),
        "trials_present": bool(trials),
        "classifications_are_success_or_oom": bool(classifications)
        and all(value in {"success", "oom"} for value in classifications),
        "trials_are_candidate_prefix": trial_mbs == expected_prefix,
        "stops_at_first_oom_or_exhausts_candidates": first_oom_index is not None
        or trial_mbs == candidates,
        "max_feasible_matches_trials": summary.get("max_feasible_mbs") == expected_max,
        "first_failed_matches_trials": summary.get("first_failed_mbs")
        == expected_failed,
    }


def load_formal_history(
    results_dir: Path,
) -> tuple[dict[tuple[Any, ...], list[dict[str, Any]]], list[dict[str, str]]]:
    history: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    errors: list[dict[str, str]] = []
    if not results_dir.is_dir():
        return history, [
            {"path": str(results_dir), "error": "results directory missing"}
        ]
    for rendered_path in sorted(results_dir.glob("*/rendered_run.json")):
        try:
            rendered = read_json(rendered_path)
            job = dict(rendered.get("job") or {})
        except (OSError, ValueError, TypeError) as exc:
            errors.append({"path": str(rendered_path), "error": str(exc)})
            continue
        job_id = str(job.get("job_id") or "")
        if job.get("kind") != "throughput" or not job_id.startswith("tput-"):
            continue
        missing = [field for field in RUN_KEY_FIELDS if field not in job]
        if missing:
            errors.append(
                {
                    "path": str(rendered_path),
                    "error": f"missing run-key fields: {missing}",
                }
            )
            continue
        history[run_key(job)].append(
            {
                "job_id": job_id,
                "job": job,
                "rendered_run_path": str(rendered_path),
                "status_path": str(rendered_path.parent / "status.json"),
                "metrics_dir": str(rendered_path.parent / "metrics"),
            }
        )
    return history, errors


def historical_result_health(
    entry: dict[str, Any], expected_job: dict[str, Any]
) -> dict[str, Any]:
    rendered_run_path = Path(entry["rendered_run_path"])
    status_path = Path(entry["status_path"])
    metrics_dir = Path(entry["metrics_dir"])
    status: dict[str, Any] = {}
    status_readable = False
    if status_path.is_file():
        try:
            status = read_json(status_path)
            status_readable = isinstance(status, dict)
        except (OSError, ValueError, TypeError):
            status = {}
    summaries: list[dict[str, Any]] = []
    summary_paths = sorted(metrics_dir.glob("summary.rank*.json"))
    summaries_readable = True
    try:
        summaries = [read_json(path) for path in summary_paths]
    except (OSError, ValueError, TypeError):
        summaries_readable = False
        summaries = []

    expected_ranks = int(expected_job.get("gpu_count") or 0)
    expected_steps = int(expected_job.get("measure_steps") or 0)
    ranks = [summary.get("rank") for summary in summaries]
    checks = {
        "status_present_and_readable": status_readable,
        "classification_success": status.get("classification") == "success",
        "run_key_matches": run_key(entry["job"]) == run_key(expected_job),
        "summary_files_readable": summaries_readable,
        "rank_count_complete": expected_ranks > 0 and len(summaries) == expected_ranks,
        "rank_set_complete": ranks == list(range(expected_ranks)),
        "measurement_window_complete": expected_steps > 0
        and bool(summaries)
        and all(
            summary.get("failure") is None
            and int(summary.get("measured_steps") or 0) >= expected_steps
            and float(summary.get("measured_seconds") or 0.0) > 0.0
            and int((summary.get("measured_totals") or {}).get("logical_samples") or 0)
            > 0
            for summary in summaries
        ),
    }
    return {
        "job_id": entry["job_id"],
        "rendered_run_path": entry["rendered_run_path"],
        "rendered_run_sha256": sha256_file(rendered_run_path),
        "status_path": entry["status_path"],
        "status_sha256": sha256_file(status_path) if status_path.is_file() else None,
        "summary_sha256": {
            path.name: sha256_file(path) for path in summary_paths if path.is_file()
        },
        "classification": status.get("classification"),
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def validate_formal_delta(
    *,
    baseline_path: Path,
    new_matrix_path: Path,
    screen_delta_report_path: Path,
    stage_decisions_path: Path,
    screen_matrix_path: Path,
    throughput_requests_path: Path,
    experiment_path: Path,
    memory_families_path: Path,
    boundary_dir: Path,
    results_dir: Path,
    max_delta_runs: int = ABSOLUTE_MAX_DELTA_RUNS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = read_jsonl(baseline_path)
    new_matrix = read_jsonl(new_matrix_path)
    screen_delta_report = read_json(screen_delta_report_path)
    stage_decisions = read_json(stage_decisions_path)
    screen_matrix = read_jsonl(screen_matrix_path)
    requests = read_jsonl(throughput_requests_path)
    experiment = read_json(experiment_path)
    families = read_jsonl(memory_families_path)

    affected_request_ids, screen_delta_checks = screen_delta_scope(screen_delta_report)
    affected = set(affected_request_ids)
    request_ids = [str(row.get("request_id")) for row in requests]
    request_duplicates = duplicate_values(request_ids)
    request_map = {str(row.get("request_id")): row for row in requests}

    decisions = list(
        (stage_decisions.get("throughput_screening") or {}).get("decisions") or ()
    )
    decision_ids = [str(row.get("request_id")) for row in decisions]
    decision_duplicates = duplicate_values(decision_ids)
    decision_map = {str(row.get("request_id")): row for row in decisions}

    baseline_run_duplicates = duplicate_key_rows(baseline, run_key, RUN_KEY_FIELDS)
    new_run_duplicates = duplicate_key_rows(new_matrix, run_key, RUN_KEY_FIELDS)
    baseline_job_id_duplicates = duplicate_values(
        str(row.get("job_id")) for row in baseline
    )
    new_job_id_duplicates = duplicate_values(
        str(row.get("job_id")) for row in new_matrix
    )
    screen_config_duplicates = duplicate_key_rows(
        screen_matrix, config_key, CONFIG_KEY_FIELDS
    )

    baseline_by_run = {run_key(row): row for row in baseline}
    new_by_run = {run_key(row): row for row in new_matrix}
    baseline_by_config: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in baseline:
        baseline_by_config[config_key(row)].append(row)
    baseline_run_keys = set(baseline_by_run)
    new_run_keys = set(new_by_run)
    added_rows = [row for row in new_matrix if run_key(row) not in baseline_run_keys]
    removed_rows = [row for row in baseline if run_key(row) not in new_run_keys]
    retained_changes = [
        {
            "run_key": run_key_record(key),
            "baseline_job_id": baseline_by_run[key].get("job_id"),
            "new_job_id": new_by_run[key].get("job_id"),
        }
        for key in sorted(
            baseline_run_keys & new_run_keys,
            key=lambda value: tuple(str(part) for part in value),
        )
        if sha256_json(baseline_by_run[key]) != sha256_json(new_by_run[key])
    ]

    baseline_unaffected = [
        row for row in baseline if str(row.get("request_id")) not in affected
    ]
    new_unaffected = [
        row for row in new_matrix if str(row.get("request_id")) not in affected
    ]
    unaffected_rows_unchanged = [sha256_json(row) for row in baseline_unaffected] == [
        sha256_json(row) for row in new_unaffected
    ]

    screen_by_config = {config_key(row): row for row in screen_matrix}
    shortlist_by_request: dict[str, list[dict[str, Any]]] = {}
    shortlist_config_keys: set[tuple[Any, ...]] = set()
    affected_request_rows: list[dict[str, Any]] = []
    expected_affected_run_keys: list[tuple[Any, ...]] = []
    shortlist_violations: list[dict[str, Any]] = []
    dynamic_max_delta_runs = 0
    affected_order = [
        request_id for request_id in request_ids if request_id in affected
    ]
    for request_id in affected_order:
        request = request_map[request_id]
        decision = decision_map.get(request_id)
        request_errors: list[str] = []
        shortlist: list[dict[str, Any]] = []
        if decision is None:
            request_errors.append("latest screening decision missing")
        else:
            shortlist = list(decision.get("shortlisted") or ())
            if decision.get("status") != "shortlisted":
                request_errors.append("latest screening decision is not shortlisted")
        request_top_k = request.get(
            "shortlist_top_k",
            (experiment.get("matrix_policy") or {}).get("throughput_shortlist_top_k"),
        )
        repeats = request.get("repeats")
        if not valid_positive_int(request_top_k):
            request_errors.append("request shortlist_top_k is invalid")
        if not valid_positive_int(repeats):
            request_errors.append("request repeats is invalid")
        if decision is not None and decision.get("top_k") != request_top_k:
            request_errors.append("decision top_k differs from request policy")
        if not shortlist:
            request_errors.append("shortlist is empty")
        if valid_positive_int(request_top_k) and len(shortlist) > int(request_top_k):
            request_errors.append("shortlist exceeds top_k")
        if valid_positive_int(request_top_k) and valid_positive_int(repeats):
            dynamic_max_delta_runs += int(request_top_k) * int(repeats)

        local_config_keys: list[tuple[Any, ...]] = []
        for selected in shortlist:
            selected_key = config_key(selected)
            local_config_keys.append(selected_key)
            shortlist_config_keys.add(selected_key)
            if str(selected.get("request_id")) != request_id:
                request_errors.append("shortlist row has the wrong request_id")
            if any(
                selected.get(field) != request.get(field)
                for field in REQUEST_SHAPE_FIELDS
            ):
                request_errors.append("shortlist row differs from request shape")
            if selected_key not in screen_by_config:
                formal_matches = baseline_by_config.get(selected_key, [])
                selected_job_ids = list(selected.get("job_ids") or ())
                formal_evidence_valid = (
                    selected.get("candidate_plan_source") == "active_formal"
                    and selected.get("measurement_source") == "formal"
                    and selected.get("formal_substitute_scope") == "active"
                    and selected.get("status") == "measured"
                    and isinstance(selected.get("samples_per_second"), (int, float))
                    and not isinstance(selected.get("samples_per_second"), bool)
                    and float(selected["samples_per_second"]) > 0.0
                    and len(selected_job_ids) == 1
                    and selected_job_ids[0]
                    in {str(row.get("job_id")) for row in formal_matches}
                )
                if not formal_evidence_valid:
                    request_errors.append(
                        "shortlist configuration is absent from screen matrix "
                        "and lacks exact active-formal evidence"
                    )
            elif (
                screen_by_config[selected_key].get("kind") != "throughput_screen"
                or screen_by_config[selected_key].get("fidelity") != "screen"
                or screen_by_config[selected_key].get("repeat") != 0
                or not str(
                    screen_by_config[selected_key].get("job_id") or ""
                ).startswith("tputscreen-")
            ):
                request_errors.append("matching screen row has an invalid screen shape")
        if len(local_config_keys) != len(set(local_config_keys)):
            request_errors.append("shortlist contains duplicate configurations")
        if valid_positive_int(repeats):
            for selected in shortlist:
                for repeat in range(int(repeats)):
                    expected_affected_run_keys.append((*config_key(selected), repeat))
        shortlist_by_request[request_id] = shortlist
        if request_errors:
            shortlist_violations.append(
                {"request_id": request_id, "errors": sorted(set(request_errors))}
            )
        affected_request_rows.append(
            {
                "request_id": request_id,
                "baseline_runs": sum(
                    str(row.get("request_id")) == request_id for row in baseline
                ),
                "new_runs": sum(
                    str(row.get("request_id")) == request_id for row in new_matrix
                ),
                "shortlisted_configurations": [
                    config_key_record(value) for value in local_config_keys
                ],
                "errors": sorted(set(request_errors)),
                "all_passed": not request_errors,
            }
        )

    actual_affected_run_keys = [
        run_key(row) for row in new_matrix if str(row.get("request_id")) in affected
    ]
    expected_affected_by_request: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for key in expected_affected_run_keys:
        expected_affected_by_request[str(key[0])].append(key)
    expected_full_run_order: list[tuple[Any, ...]] = []
    for request_id in request_ids:
        if request_id in affected:
            expected_full_run_order.extend(expected_affected_by_request[request_id])
        else:
            expected_full_run_order.extend(
                run_key(row)
                for row in baseline
                if str(row.get("request_id")) == request_id
            )
    actual_full_run_order = [run_key(row) for row in new_matrix]

    new_shape_errors = []
    for row in new_matrix:
        errors = formal_job_shape_errors(
            row, request_map.get(str(row.get("request_id"))), experiment
        )
        if errors:
            new_shape_errors.append(
                {"job_id": str(row.get("job_id")), "errors": errors}
            )

    added_scope_errors = [
        {
            "job_id": str(row.get("job_id")),
            "request_id": str(row.get("request_id")),
            "reason": "added formal run is outside the validated screen-delta requests",
        }
        for row in added_rows
        if str(row.get("request_id")) not in affected
    ]
    removed_scope_errors = [
        {
            "job_id": str(row.get("job_id")),
            "request_id": str(row.get("request_id")),
            "reason": "removed formal run is outside the validated screen-delta requests",
        }
        for row in removed_rows
        if str(row.get("request_id")) not in affected
    ]
    added_selection_errors = []
    for row in added_rows:
        errors = []
        if config_key(row) not in shortlist_config_keys:
            errors.append("configuration is absent from the latest shortlist")
        if config_key(row) not in screen_by_config:
            errors.append("configuration is absent from the latest screen matrix")
        if errors:
            added_selection_errors.append(
                {"job_id": str(row.get("job_id")), "errors": errors}
            )

    family_index: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for family in families:
        family_index[family_key(family)].append(family)
    family_signature_duplicates = [
        {
            "family_key": key_record(FAMILY_KEY_FIELDS, key),
            "family_job_ids": [str(row.get("job_id")) for row in rows],
        }
        for key, rows in family_index.items()
        if len(rows) > 1
    ]

    boundary_by_config: dict[tuple[Any, ...], dict[str, Any]] = {}
    boundary_errors: list[dict[str, Any]] = []
    for row in added_rows:
        key = config_key(row)
        if key in boundary_by_config:
            continue
        matches = family_index.get(family_key(row), [])
        evidence: dict[str, Any] = {
            "configuration": config_key_record(key),
            "matching_family_job_ids": [str(value.get("job_id")) for value in matches],
        }
        checks: dict[str, bool] = {
            "maps_to_exactly_one_memory_family": len(matches) == 1,
        }
        if len(matches) == 1:
            family = matches[0]
            summary_path = boundary_dir / f"{family['job_id']}.json"
            evidence["family_job_id"] = family["job_id"]
            evidence["summary_path"] = str(summary_path)
            checks["boundary_summary_present"] = summary_path.is_file()
            if summary_path.is_file():
                try:
                    summary = read_json(summary_path)
                    summary_checks = validate_boundary_summary(family, summary)
                    checks.update(summary_checks)
                    max_feasible = summary.get("max_feasible_mbs")
                    checks["mbs_within_boundary"] = valid_positive_int(
                        max_feasible
                    ) and int(row["mbs"]) <= int(max_feasible)
                    evidence["max_feasible_mbs"] = max_feasible
                    evidence["summary_sha256"] = sha256_file(summary_path)
                except (OSError, ValueError, TypeError) as exc:
                    checks["boundary_summary_readable"] = False
                    evidence["read_error"] = str(exc)
        evidence["checks"] = checks
        evidence["all_passed"] = all(checks.values())
        boundary_by_config[key] = evidence
        if not evidence["all_passed"]:
            boundary_errors.append(evidence)

    history, history_read_errors = load_formal_history(results_dir)
    added_run_rows: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    reused_runs: list[dict[str, Any]] = []
    unhealthy_history_conflicts: list[dict[str, Any]] = []
    orphan_result_collisions: list[dict[str, Any]] = []
    for row in added_rows:
        histories = history.get(run_key(row), [])
        assessments = [historical_result_health(entry, row) for entry in histories]
        healthy = [entry for entry in assessments if entry["all_passed"]]
        result_dir = results_dir / str(row.get("job_id"))
        disposition = "queue"
        if healthy:
            disposition = "reused"
            reused_runs.append(
                {
                    "job_id": str(row.get("job_id")),
                    "run_key": run_key_record(run_key(row)),
                    "healthy_history": healthy,
                }
            )
        elif histories:
            disposition = "rejected_unhealthy_history"
            unhealthy_history_conflicts.append(
                {
                    "job_id": str(row.get("job_id")),
                    "run_key": run_key_record(run_key(row)),
                    "history": assessments,
                }
            )
        elif result_dir.exists() and any(result_dir.iterdir()):
            disposition = "rejected_orphan_result_collision"
            orphan_result_collisions.append(
                {
                    "job_id": str(row.get("job_id")),
                    "result_dir": str(result_dir),
                    "reason": "result directory exists without indexed formal history",
                }
            )
        else:
            queue.append(row)
        added_run_rows.append(
            {
                "job_id": str(row.get("job_id")),
                "run_key": run_key_record(run_key(row)),
                "disposition": disposition,
                "history": assessments,
                "boundary": boundary_by_config.get(config_key(row)),
            }
        )

    queue_expected = [
        row
        for row, audit in zip(added_rows, added_run_rows)
        if audit["disposition"] == "queue"
    ]
    queue_order_exact = [sha256_json(row) for row in queue] == [
        sha256_json(row) for row in queue_expected
    ]

    unknown_matrix_request_ids = sorted(
        {
            str(row.get("request_id"))
            for row in (*baseline, *new_matrix)
            if str(row.get("request_id")) not in request_map
        }
    )
    checks = {
        **screen_delta_checks,
        "screen_delta_requests_unique": len(affected_request_ids)
        == len(set(affected_request_ids)),
        "screen_delta_requests_known": bool(affected) and affected <= set(request_map),
        "throughput_request_ids_unique": not request_duplicates,
        "screening_decision_ids_unique": not decision_duplicates,
        "matrix_request_ids_known": not unknown_matrix_request_ids,
        "baseline_run_keys_unique": not baseline_run_duplicates,
        "new_run_keys_unique": not new_run_duplicates,
        "baseline_job_ids_unique": not baseline_job_id_duplicates,
        "new_job_ids_unique": not new_job_id_duplicates,
        "screen_config_keys_unique": not screen_config_duplicates,
        "memory_family_signatures_unique": not family_signature_duplicates,
        "unaffected_request_rows_unchanged": unaffected_rows_unchanged,
        "retained_run_rows_unchanged": not retained_changes,
        "only_affected_requests_added": not added_scope_errors,
        "only_affected_requests_removed": not removed_scope_errors,
        "affected_shortlists_valid": bool(affected_request_rows)
        and not shortlist_violations,
        "affected_formal_order_matches_shortlist": actual_affected_run_keys
        == expected_affected_run_keys,
        "new_matrix_order_matches_requests_and_shortlists": actual_full_run_order
        == expected_full_run_order,
        "new_formal_rows_valid": not new_shape_errors,
        "added_configurations_selected_and_screened": not added_selection_errors,
        "added_configurations_within_memory_boundaries": not boundary_errors,
        "historical_formal_results_readable": not history_read_errors,
        "existing_formal_attempts_are_healthy_reuses": not unhealthy_history_conflicts,
        "no_orphan_result_collisions": not orphan_result_collisions,
        "dynamic_delta_cap_positive": dynamic_max_delta_runs > 0,
        "dynamic_delta_cap_within_absolute_limit": 0
        < dynamic_max_delta_runs
        <= ABSOLUTE_MAX_DELTA_RUNS,
        "requested_delta_limit_valid": 0 < max_delta_runs <= ABSOLUTE_MAX_DELTA_RUNS,
        "added_runs_within_dynamic_limit": len(added_rows) <= dynamic_max_delta_runs,
        "added_runs_within_absolute_limit": len(added_rows) <= ABSOLUTE_MAX_DELTA_RUNS,
        "added_runs_within_requested_limit": len(added_rows) <= max_delta_runs,
        "queued_runs_within_all_limits": len(queue)
        <= min(
            dynamic_max_delta_runs or ABSOLUTE_MAX_DELTA_RUNS,
            max_delta_runs,
            ABSOLUTE_MAX_DELTA_RUNS,
        ),
        "delta_queue_preserves_new_matrix_order": queue_order_exact,
    }
    violations = {
        "throughput_request_id_duplicates": request_duplicates,
        "screening_decision_id_duplicates": decision_duplicates,
        "unknown_matrix_request_ids": unknown_matrix_request_ids,
        "baseline_duplicate_run_keys": baseline_run_duplicates,
        "new_duplicate_run_keys": new_run_duplicates,
        "baseline_duplicate_job_ids": baseline_job_id_duplicates,
        "new_duplicate_job_ids": new_job_id_duplicates,
        "screen_duplicate_config_keys": screen_config_duplicates,
        "memory_family_signature_duplicates": family_signature_duplicates,
        "retained_run_row_changes": retained_changes,
        "shortlist_violations": shortlist_violations,
        "new_formal_shape_errors": new_shape_errors,
        "added_scope_errors": added_scope_errors,
        "removed_scope_errors": removed_scope_errors,
        "added_selection_errors": added_selection_errors,
        "boundary_errors": boundary_errors,
        "historical_result_read_errors": history_read_errors,
        "unhealthy_history_conflicts": unhealthy_history_conflicts,
        "orphan_result_collisions": orphan_result_collisions,
    }
    report = {
        "schema_version": 1,
        "purpose": (
            "Prepare only newly shortlisted formal-throughput runs caused by the "
            "validated H800 LoRA + ZeRO-3 screen delta"
        ),
        "config_key_fields": list(CONFIG_KEY_FIELDS),
        "run_key_fields": list(RUN_KEY_FIELDS),
        "inputs": {
            "baseline": {
                "path": str(baseline_path),
                "sha256": sha256_file(baseline_path),
            },
            "new_matrix": {
                "path": str(new_matrix_path),
                "sha256": sha256_file(new_matrix_path),
            },
            "screen_delta_report": {
                "path": str(screen_delta_report_path),
                "sha256": sha256_file(screen_delta_report_path),
            },
            "stage_decisions": {
                "path": str(stage_decisions_path),
                "sha256": sha256_file(stage_decisions_path),
            },
            "screen_matrix": {
                "path": str(screen_matrix_path),
                "sha256": sha256_file(screen_matrix_path),
            },
            "throughput_requests": {
                "path": str(throughput_requests_path),
                "sha256": sha256_file(throughput_requests_path),
            },
            "experiment": {
                "path": str(experiment_path),
                "sha256": sha256_file(experiment_path),
            },
            "memory_families": {
                "path": str(memory_families_path),
                "sha256": sha256_file(memory_families_path),
            },
            "boundary_dir": str(boundary_dir),
            "results_dir": str(results_dir),
        },
        "counts": {
            "baseline_runs": len(baseline),
            "baseline_configurations": len({config_key(row) for row in baseline}),
            "new_runs": len(new_matrix),
            "new_configurations": len({config_key(row) for row in new_matrix}),
            "affected_requests": len(affected),
            "retained_runs": len(baseline_run_keys & new_run_keys),
            "added_runs": len(added_rows),
            "added_configurations": len({config_key(row) for row in added_rows}),
            "removed_runs": len(removed_rows),
            "reused_healthy_runs": len(reused_runs),
            "queued_runs": len(queue),
            "dynamic_max_delta_runs": dynamic_max_delta_runs,
            "requested_max_delta_runs": max_delta_runs,
            "absolute_max_delta_runs": ABSOLUTE_MAX_DELTA_RUNS,
        },
        "affected_request_ids": affected_request_ids,
        "affected_requests": affected_request_rows,
        "added_runs": added_run_rows,
        "reused_runs": reused_runs,
        "removed_runs": [
            {"job_id": str(row.get("job_id")), "run_key": run_key_record(run_key(row))}
            for row in removed_rows
        ],
        "checks": checks,
        "violations": violations,
        "freeze_preparation": {
            "allowed_job_ids": [str(row["job_id"]) for row in queue],
            "reused_job_ids": [str(row["job_id"]) for row in reused_runs],
            "policy": (
                "Run only queued formal delta rows in new-matrix order; explicitly reuse "
                "healthy historical formal runs; reject every unhealthy prior attempt."
            ),
        },
        "all_passed": all(checks.values()) and all_empty(violations),
    }
    return report, queue


def output_paths_are_safe(
    report_path: Path,
    output_queue_path: Path,
    input_paths: Iterable[Path],
    project_root: Path,
) -> bool:
    outputs = {report_path.resolve(), output_queue_path.resolve()}
    protected = {path.resolve() for path in input_paths}
    protected.update(
        {
            (project_root / "config" / "APPROVED_TO_RUN.json").resolve(),
            (project_root / "runtime" / "approval_design.json").resolve(),
        }
    )
    return (
        len(outputs) == 2
        and outputs.isdisjoint(protected)
        and all(is_within(path, project_root) for path in outputs)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--new-matrix", type=Path, required=True)
    parser.add_argument("--screen-delta-report", type=Path, required=True)
    parser.add_argument("--stage-decisions", type=Path, required=True)
    parser.add_argument("--screen-matrix", type=Path, required=True)
    parser.add_argument("--throughput-requests", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--memory-families", type=Path, required=True)
    parser.add_argument("--boundary-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-queue", type=Path, required=True)
    parser.add_argument("--max-delta-runs", type=int, default=ABSOLUTE_MAX_DELTA_RUNS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_inputs = (
        args.baseline,
        args.new_matrix,
        args.screen_delta_report,
        args.stage_decisions,
        args.screen_matrix,
        args.throughput_requests,
        args.experiment,
        args.memory_families,
    )
    directory_inputs = (args.boundary_dir, args.results_dir)
    missing = [str(path) for path in file_inputs if not path.is_file()]
    missing.extend(str(path) for path in directory_inputs if not path.is_dir())
    if missing:
        raise SystemExit(f"Required formal-delta inputs are absent: {missing}")
    if not 0 < args.max_delta_runs <= ABSOLUTE_MAX_DELTA_RUNS:
        raise SystemExit(
            f"--max-delta-runs must be between 1 and {ABSOLUTE_MAX_DELTA_RUNS}"
        )
    if not output_paths_are_safe(
        args.report,
        args.output_queue,
        (*file_inputs, *directory_inputs),
        args.project_root,
    ):
        raise SystemExit(
            "Refusing unsafe output paths: outputs must be distinct, inside the project, "
            "and separate from all inputs and live approvals"
        )
    existing = [path for path in (args.report, args.output_queue) if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing formal-delta outputs without --overwrite: {existing}"
        )

    report, queue = validate_formal_delta(
        baseline_path=args.baseline,
        new_matrix_path=args.new_matrix,
        screen_delta_report_path=args.screen_delta_report,
        stage_decisions_path=args.stage_decisions,
        screen_matrix_path=args.screen_matrix,
        throughput_requests_path=args.throughput_requests,
        experiment_path=args.experiment,
        memory_families_path=args.memory_families,
        boundary_dir=args.boundary_dir,
        results_dir=args.results_dir,
        max_delta_runs=args.max_delta_runs,
    )
    if report["all_passed"]:
        write_jsonl(args.output_queue, queue)
        report["output_queue"] = {
            "path": str(args.output_queue),
            "sha256": sha256_file(args.output_queue),
            "jobs": len(queue),
            "written": True,
        }
    else:
        if args.output_queue.exists():
            args.output_queue.unlink()
        report["output_queue"] = {
            "path": str(args.output_queue),
            "jobs": 0,
            "written": False,
        }
    write_json(args.report, report)
    print(
        json.dumps(
            {
                "all_passed": report["all_passed"],
                "added_runs": report["counts"]["added_runs"],
                "reused_healthy_runs": report["counts"]["reused_healthy_runs"],
                "queued_runs": report["counts"]["queued_runs"],
                "report": str(args.report),
                "output_queue": report["output_queue"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
