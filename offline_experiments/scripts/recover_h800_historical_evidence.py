#!/usr/bin/env python3
"""Recover auditable evidence for the historical flat-layout H800 results.

This module is deliberately separate from the native v2 execution-fingerprint
path.  It never upgrades a historical result to native evidence.  Instead it
builds a source-bound sidecar that records three independent facts:

* whether the terminal attempt can be isolated;
* how much of the original runtime/approval/provenance archive still exists;
* which pre-declared experimental use, if any, the measurement may serve.

No GPU work is performed and the source result tree is never modified.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from common import ROOT, read_json, sha256_file, sha256_json
from export_h800_observations import (
    SCHEMA as OBSERVATION_SCHEMA,
    export_observations,
    validate_canonical_observation,
)


SCHEMA = "sft_h800_historical_recovery/v1"
RECORD_SCHEMA = "sft_historical_recovered_evidence/v1"
LOOCV_POLICY = (
    "leave-one-(model,train_type,dataset,target_gbs)-scenario-out/v1"
)
PACKING_POLICY = "paired-once-per-side/v1"
PROFILER_POLICY = "declared-profiler-calibration-holdout/v1"
EVIDENCE_TIERS = {"native_v2", "legacy_verified", "legacy_consistent"}
USAGE_CLASSES = {"calibration_candidate", "diagnostic_only", "rejected_terminal"}
DIAGNOSTIC_KINDS = {"smoke", "runtime_canary", "thermal_validation"}
RESOURCE_KINDS = {"memory_probe", "throughput_screen", "throughput"}
OOM_SIGNATURE = re.compile(
    r"(?:cuda\s+out\s+of\s+memory|torch\.outofmemoryerror|"
    r"outofmemoryerror:\s*cuda|cuda\s+error:\s*out\s+of\s+memory)",
    re.IGNORECASE,
)
JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
EVENT_FILE = re.compile(r"^events\.rank([0-9]+)\.jsonl$")
SUMMARY_FILE = re.compile(r"^summary\.rank([0-9]+)\.json$")


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _safe_project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _snapshot(path: Path, root: Path) -> dict[str, Any] | None:
    try:
        path = _safe_project_path(root, path)
    except (OSError, ValueError):
        return None
    if not path.is_file():
        return None
    return {
        "path": _relative(path, root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _read_observations(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid observation JSON on line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict) or row.get("schema") != OBSERVATION_SCHEMA:
                raise ValueError(
                    f"Observation line {line_number} is not {OBSERVATION_SCHEMA}"
                )
            reasons = validate_canonical_observation(row)
            if reasons:
                raise ValueError(
                    f"Observation line {line_number} failed canonical validation: "
                    f"{reasons}"
                )
            rows.append(row)
    return rows


def _json_file_index(
    paths: Iterable[Path], root: Path
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    safe_paths: set[Path] = set()
    for candidate in paths:
        try:
            path = _safe_project_path(root, candidate)
        except (OSError, ValueError):
            continue
        if path.is_file():
            safe_paths.add(path)
    for path in sorted(safe_paths):
        try:
            payload = read_json(path)
            if not isinstance(payload, dict):
                continue
            index[sha256_file(path)] = {"path": path, "payload": payload}
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return index


def _approval_design_index(root: Path) -> dict[str, dict[str, Any]]:
    candidates: set[Path] = set()
    for base in (root / "runtime", root / "artifacts"):
        if not base.is_dir():
            continue
        for pattern in (
            "**/*approval*design*.json",
            "**/*approval*candidate*.json",
            "**/candidate_approval_design.json",
            "**/previous_approval_design.json",
        ):
            candidates.update(base.glob(pattern))
    candidates.add(root / "runtime" / "approval_design.json")
    return _json_file_index(candidates, root)


def _provenance_index(root: Path) -> dict[str, dict[str, Any]]:
    candidates = [root / "artifacts" / "provenance.json"]
    candidates.extend((root / "artifacts" / "provenance_history").glob("*.json"))
    return _json_file_index(candidates, root)


def _parse_terminal_payload(text: Any) -> list[dict[str, Any]]:
    if not isinstance(text, str):
        return []
    payloads: list[dict[str, Any]] = []
    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{") or not candidate.endswith("}"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _scheduler_index(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    path = root / "runtime" / "scheduler_events.jsonl"
    by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
    malformed = 0
    lines = 0
    if path.is_file():
        with path.open(encoding="utf-8", errors="replace") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                lines += 1
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(event, dict):
                    malformed += 1
                    continue
                job_id = event.get("job_id")
                if isinstance(job_id, str):
                    by_job[job_id].append({**event, "line_number": line_number})
    return by_job, {
        "source": _snapshot(path, root),
        "lines": lines,
        "malformed_lines": malformed,
    }


def _normalized_mask(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, list):
        values = value
    elif isinstance(value, str) and value.strip():
        values = value.split(",")
    else:
        return None
    parsed: list[int] = []
    for item in values:
        if isinstance(item, bool):
            return None
        try:
            parsed.append(int(str(item).strip()))
        except ValueError:
            return None
    if not parsed or len(set(parsed)) != len(parsed):
        return None
    return tuple(parsed)


def _same_mask(left: Any, right: Any) -> bool:
    left_mask = _normalized_mask(left)
    right_mask = _normalized_mask(right)
    return left_mask is not None and left_mask == right_mask


def _scheduler_closure(
    status: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, Any]:
    started = _finite_float(status.get("started_unix"))
    finished = _finite_float(status.get("finished_unix"))
    classification = status.get("classification")
    expected_wrapper_return_code = 0 if classification == "success" else 2
    ordered = sorted(
        events,
        key=lambda event: (
            _integer(event.get("line_number")) or 0,
            _finite_float(event.get("time_unix")) or 0.0,
        ),
    )
    candidates: list[dict[str, Any]] = []
    exact: list[dict[str, Any]] = []
    for end_index, end_event in enumerate(ordered):
        if end_event.get("event") != "trial_end":
            continue
        matching_payloads = [
            payload
            for payload in _parse_terminal_payload(
                end_event.get("launcher_output_tail")
            )
            if payload.get("job_id") == status.get("job_id")
            and payload.get("started_unix") == status.get("started_unix")
            and payload.get("finished_unix") == status.get("finished_unix")
        ]
        if not matching_payloads:
            continue
        payload = matching_payloads[0]
        end_time = _finite_float(end_event.get("time_unix"))
        prior_starts = [
            (index, event)
            for index, event in enumerate(ordered[:end_index])
            if event.get("event") == "trial_start"
            and _same_mask(event.get("gpu_mask"), status.get("gpu_mask"))
            and started is not None
            and _finite_float(event.get("time_unix")) is not None
            and _finite_float(event.get("time_unix")) <= started
        ]
        start_index, start_event = (
            max(prior_starts, key=lambda item: item[0])
            if prior_starts
            else (None, None)
        )
        intervening_boundaries = (
            [
                event
                for event in ordered[start_index + 1 : end_index]
                if event.get("event") in {"trial_start", "trial_end"}
            ]
            if start_index is not None
            else []
        )
        checks = {
            "terminal_payload_unique": len(matching_payloads) == 1,
            "start_present": start_event is not None,
            "time_envelope_matches": bool(
                start_event is not None
                and started is not None
                and finished is not None
                and _finite_float(start_event.get("time_unix")) is not None
                and _finite_float(start_event.get("time_unix")) <= started
                <= finished
                and end_time is not None
                and finished <= end_time
            ),
            "no_intervening_job_boundary": not intervening_boundaries,
            "classification_matches": payload.get("classification")
            == classification
            and end_event.get("classification") == classification,
            "payload_return_code_matches": payload.get("return_code")
            == status.get("return_code"),
            "wrapper_return_code_matches": end_event.get("return_code")
            == expected_wrapper_return_code,
            "start_gpu_mask_matches": bool(
                start_event is not None
                and _same_mask(start_event.get("gpu_mask"), status.get("gpu_mask"))
            ),
            "end_gpu_mask_matches": _same_mask(
                end_event.get("gpu_mask"), status.get("gpu_mask")
            ),
            "payload_gpu_mask_matches": _same_mask(
                payload.get("gpu_mask"), status.get("gpu_mask")
            ),
            "approval_reference_matches": payload.get(
                "approval_design_sha256"
            )
            == status.get("approval_design_sha256"),
            "provenance_reference_matches": payload.get("provenance_sha256")
            == status.get("provenance_sha256"),
            "runtime_reference_matches": payload.get(
                "runtime_fingerprint_sha256"
            )
            == status.get("runtime_fingerprint_sha256"),
        }
        candidate = {
            "start_line_number": (
                start_event.get("line_number") if start_event else None
            ),
            "end_line_number": end_event.get("line_number"),
            "start_time_unix": (
                start_event.get("time_unix") if start_event else None
            ),
            "end_time_unix": end_event.get("time_unix"),
            "start_event_sha256": (
                sha256_json(start_event) if start_event else None
            ),
            "end_event_sha256": sha256_json(end_event),
            "payload_sha256": sha256_json(payload),
            **checks,
        }
        candidates.append(candidate)
        if all(checks.values()):
            exact.append(candidate)
    return {
        "strength": "scheduler_verified" if len(exact) == 1 else "self_closed",
        "candidate_terminal_payloads": len(candidates),
        "exact_closures": len(exact),
        "closure": exact[0] if len(exact) == 1 else None,
    }


def _event_evidence(
    result_dir: Path,
    row: dict[str, Any],
    job: dict[str, Any],
    root: Path,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    attempt = row.get("attempt") or {}
    started = _finite_float(attempt.get("started_unix"))
    finished = _finite_float(attempt.get("finished_unix"))
    expected_ranks = _integer(job.get("gpu_count")) or 0
    expected_files = {
        Path(str(item.get("path") or "")).name: item.get("sha256")
        for item in ((row.get("source") or {}).get("event_filter") or {}).get(
            "event_files", []
        )
        if isinstance(item, dict)
    }
    artifacts: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    malformed = 0
    missing_time = 0
    outside = 0
    hashes_match = True
    file_rank_mismatches = 0
    for path in sorted((result_dir / "metrics").glob("events.rank*.jsonl")):
        file_match = EVENT_FILE.fullmatch(path.name)
        file_rank = int(file_match.group(1)) if file_match else None
        snapshot = _snapshot(path, root)
        if snapshot is not None:
            artifacts.append(snapshot)
            hashes_match = hashes_match and expected_files.get(path.name) == snapshot["sha256"]
        with path.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(event, dict):
                    malformed += 1
                    continue
                timestamp = _finite_float(event.get("time_unix"))
                if timestamp is None:
                    missing_time += 1
                    continue
                if started is None or finished is None or not started <= timestamp <= finished:
                    outside += 1
                    continue
                event_rank = _integer(event.get("rank"))
                if file_rank is None or event_rank != file_rank:
                    file_rank_mismatches += 1
                selected.append(event)

    ranks = {
        rank
        for event in selected
        if (rank := _integer(event.get("rank"))) is not None
    }
    train_begin = [event for event in selected if event.get("event") == "train_begin"]
    begin_ranks = {
        rank
        for event in train_begin
        if (rank := _integer(event.get("rank"))) is not None
    }
    begin_counts = Counter(
        _integer(event.get("rank")) for event in train_begin
    )
    begin_metadata_matches = all(
        isinstance(event.get("metadata"), dict) and event["metadata"] == job
        for event in train_begin
    )
    begin_world_size_matches = all(
        _integer(event.get("world_size")) == expected_ranks
        for event in train_begin
    )
    begin_once_per_rank = bool(
        expected_ranks > 0
        and begin_counts
        == Counter({rank: 1 for rank in range(expected_ranks)})
    )
    failure_errors = [
        str(event.get("error"))
        for event in selected
        if event.get("event") == "failure" and event.get("error")
    ]
    evidence = {
        "method": "inclusive_status_window_with_rank_metadata/v1",
        "started_unix": started,
        "finished_unix": finished,
        "event_files": len(artifacts),
        "event_file_hashes_match_canonical": hashes_match
        and len(artifacts) == len(expected_files),
        "events_in_window": len(selected),
        "events_outside_window": outside,
        "malformed_lines": malformed,
        "events_missing_time": missing_time,
        "ranks": sorted(ranks),
        "expected_ranks": list(range(expected_ranks)),
        "rank_set_complete": ranks == set(range(expected_ranks)),
        "event_file_rank_mismatches": file_rank_mismatches,
        "event_file_rank_binding_complete": file_rank_mismatches == 0,
        "train_begin_ranks": sorted(begin_ranks),
        "train_begin_rank_set_complete": begin_ranks == set(range(expected_ranks)),
        "train_begin_once_per_rank": begin_once_per_rank,
        "train_begin_world_size_matches": begin_world_size_matches,
        "train_begin_metadata_matches_runtime_job": begin_metadata_matches,
        "failure_event_count": len(failure_errors),
        "failure_signatures_sha256": sorted(
            set(sha256_json(error) for error in failure_errors)
        ),
    }
    return evidence, failure_errors, artifacts


def _summary_evidence(
    result_dir: Path, job: dict[str, Any], root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_ranks = _integer(job.get("gpu_count")) or 0
    ranks: set[int] = set()
    valid = True
    file_rank_mismatches = 0
    artifacts: list[dict[str, Any]] = []
    summaries = []
    for path in sorted((result_dir / "metrics").glob("summary.rank*.json")):
        file_match = SUMMARY_FILE.fullmatch(path.name)
        file_rank = int(file_match.group(1)) if file_match else None
        snapshot = _snapshot(path, root)
        if snapshot is not None:
            artifacts.append(snapshot)
        try:
            payload = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            valid = False
            continue
        if not isinstance(payload, dict):
            valid = False
            continue
        rank = _integer(payload.get("rank"))
        if file_rank is None or rank != file_rank:
            file_rank_mismatches += 1
        if rank is not None:
            ranks.add(rank)
        metadata = payload.get("metadata")
        row_valid = bool(
            rank is not None
            and rank == file_rank
            and payload.get("world_size") == expected_ranks
            and payload.get("failure") is None
            and metadata == job
            and (_integer(payload.get("measured_steps")) or 0) > 0
        )
        valid = valid and row_valid
        summaries.append(
            {
                "rank": rank,
                "payload_sha256": sha256_json(payload),
                "valid": row_valid,
            }
        )
    complete = bool(
        valid
        and ranks == set(range(expected_ranks))
        and len(summaries) == expected_ranks
    )
    return {
        "method": "complete_rank_summaries/v1",
        "expected_ranks": list(range(expected_ranks)),
        "ranks": sorted(ranks),
        "summaries": summaries,
        "file_rank_mismatches": file_rank_mismatches,
        "file_rank_binding_complete": file_rank_mismatches == 0,
        "complete": complete,
    }, artifacts


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _deepspeed_config_evidence(
    root: Path, job: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    zero = str(job.get("zero") or "none").lower().replace("-", "")
    expected_stage = {
        "none": None,
        "zero0": None,
        "0": None,
        "zero2": 2,
        "zero3": 3,
    }.get(zero, "invalid")
    deepspeed = config.get("deepspeed")
    if expected_stage == "invalid":
        return {
            "expected_stage": None,
            "actual_stage": None,
            "path": None,
            "matches": False,
        }, None
    if zero in {"none", "zero0", "0"}:
        return {
            "expected_stage": None,
            "actual_stage": None,
            "path": None,
            "matches": not deepspeed,
        }, None
    if not isinstance(deepspeed, str) or not deepspeed:
        return {
            "expected_stage": expected_stage,
            "actual_stage": None,
            "path": None,
            "matches": False,
        }, None
    try:
        path = _safe_project_path(root, deepspeed)
    except (OSError, ValueError):
        return {
            "expected_stage": expected_stage,
            "actual_stage": None,
            "path": None,
            "matches": False,
        }, None
    snapshot = _snapshot(path, root)
    try:
        payload = read_json(path) if path.is_file() else None
    except (OSError, ValueError, json.JSONDecodeError):
        payload = None
    optimization = (
        payload.get("zero_optimization") if isinstance(payload, dict) else None
    )
    actual_stage = (
        _integer(optimization.get("stage"))
        if isinstance(optimization, dict)
        else None
    )
    return {
        "expected_stage": expected_stage,
        "actual_stage": actual_stage,
        "path": _relative(path, root) if path.is_file() else None,
        "matches": actual_stage == expected_stage,
    }, snapshot


def _telemetry_indices(path: Path) -> tuple[set[int], int, int]:
    indices: set[int] = set()
    rows = 0
    invalid_rows = 0
    if not path.is_file():
        return indices, rows, invalid_rows
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as source:
            reader = csv.DictReader(source)
            fields = set(reader.fieldnames or [])
            index_field = "index" if "index" in fields else "gpu" if "gpu" in fields else None
            if index_field is None:
                return indices, rows, 1
            for row in reader:
                rows += 1
                index = _integer(row.get(index_field))
                if index is None:
                    invalid_rows += 1
                else:
                    indices.add(index)
    except OSError:
        invalid_rows += 1
    return indices, rows, invalid_rows


def _hardware_evidence(
    root: Path,
    result_dir: Path,
    row: dict[str, Any],
    job: dict[str, Any],
    status: dict[str, Any],
    rendered: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hardware_path = root / "config" / "hardware.json"
    experiment_path = root / "config" / "experiment.json"
    try:
        hardware = read_json(hardware_path) if hardware_path.is_file() else None
    except (OSError, ValueError, json.JSONDecodeError):
        hardware = None
    try:
        experiment = read_json(experiment_path) if experiment_path.is_file() else None
    except (OSError, ValueError, json.JSONDecodeError):
        experiment = None
    hardware = hardware if isinstance(hardware, dict) else {}
    experiment = experiment if isinstance(experiment, dict) else {}
    scope = experiment.get("training_scope") or {}
    scope = scope if isinstance(scope, dict) else {}
    identity = " ".join(
        str(value)
        for value in (
            hardware.get("gpu_id"),
            hardware.get("name_reported_by_driver"),
            scope.get("gpu_type"),
        )
        if value is not None
    ).lower()
    status_mask = _normalized_mask(status.get("gpu_mask"))
    rendered_mask = _normalized_mask(rendered.get("gpu_mask"))
    row_hardware = row.get("hardware") or {}
    row_mask = _normalized_mask(row_hardware.get("gpu_mask"))
    expected_count = _integer(job.get("gpu_count")) or 0
    allowed_values = scope.get("gpu_ids")
    allowed_ids = {
        value
        for value in (
            _integer(item) for item in allowed_values or []
        )
        if value is not None
    }
    telemetry_path = result_dir / "nvidia_smi.csv"
    telemetry_indices, telemetry_rows, telemetry_invalid = _telemetry_indices(
        telemetry_path
    )
    mask_set = set(status_mask or ())
    evidence = {
        "campaign_identified_as_h800": "h800" in identity and "4090" not in identity,
        "canonical_profile_matches_campaign": row_hardware.get("profile") == hardware,
        "canonical_family_is_h800": row_hardware.get("gpu_family")
        in {"H800", "declared_H800_unverified"},
        "mask": list(status_mask or ()),
        "mask_present_unique_and_sized": bool(
            status_mask is not None
            and expected_count > 0
            and len(status_mask) == expected_count
        ),
        "status_rendered_canonical_masks_match": bool(
            status_mask is not None
            and status_mask == rendered_mask == row_mask
        ),
        "mask_within_declared_pool": not allowed_ids or mask_set.issubset(allowed_ids),
        "declared_pool": sorted(allowed_ids),
        "telemetry_present": telemetry_rows > 0,
        "telemetry_rows": telemetry_rows,
        "telemetry_invalid_rows": telemetry_invalid,
        "telemetry_gpu_indices": sorted(telemetry_indices),
        "telemetry_matches_mask": bool(
            status_mask is not None
            and telemetry_invalid == 0
            and telemetry_indices == mask_set
        ),
    }
    evidence["complete"] = all(
        evidence[key]
        for key in (
            "campaign_identified_as_h800",
            "canonical_profile_matches_campaign",
            "canonical_family_is_h800",
            "mask_present_unique_and_sized",
            "status_rendered_canonical_masks_match",
            "mask_within_declared_pool",
            "telemetry_present",
            "telemetry_matches_mask",
        )
    )
    artifacts = [
        artifact
        for artifact in (
            _snapshot(hardware_path, root),
            _snapshot(experiment_path, root),
            _snapshot(telemetry_path, root),
        )
        if artifact is not None
    ]
    return evidence, artifacts


def _configuration_evidence(
    root: Path,
    result_dir: Path,
    row: dict[str, Any],
    job: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = row.get("source") or {}
    rendered = (source.get("rendered_run") or {}).get("payload") or {}
    status = (source.get("status") or {}).get("payload") or {}
    job_id = str(job.get("job_id") or "")
    job_id_valid = bool(JOB_ID.fullmatch(job_id))
    jobs_root = root / "runtime" / "jobs"
    runtime_job_path = jobs_root / (f"{job_id}.json" if job_id_valid else ".invalid")
    input_job_path = jobs_root / (
        f"input-{job_id}.json" if job_id_valid else ".invalid-input"
    )
    try:
        runtime_job = read_json(runtime_job_path) if runtime_job_path.is_file() else None
    except (OSError, ValueError, json.JSONDecodeError):
        runtime_job = None
    try:
        input_job = read_json(input_job_path) if input_job_path.is_file() else None
    except (OSError, ValueError, json.JSONDecodeError):
        input_job = None
    config_value = rendered.get("config_path")
    try:
        config_path = (
            _safe_project_path(root, str(config_value))
            if config_value
            else result_dir / "missing"
        )
    except (OSError, ValueError):
        config_path = result_dir / "invalid-outside-project-config"
    config = _load_yaml(config_path)
    deepspeed_evidence, deepspeed_snapshot = (
        _deepspeed_config_evidence(root, job, config)
        if isinstance(config, dict)
        else (
            {
                "expected_stage": None,
                "actual_stage": None,
                "path": None,
                "matches": False,
            },
            None,
        )
    )
    environment = rendered.get("environment") or {}
    gpu_mask_matches = _same_mask(status.get("gpu_mask"), rendered.get("gpu_mask"))
    input_subset = bool(
        isinstance(input_job, dict)
        and all(runtime_job.get(key) == value for key, value in input_job.items())
        if isinstance(runtime_job, dict)
        else False
    )
    semantic_checks: dict[str, bool] = {}
    if isinstance(config, dict):
        semantic_checks = {
            "dataset": config.get("dataset") == job.get("dataset_id"),
            "cutoff_len": config.get("cutoff_len") == job.get("cutoff_len"),
            "micro_batch_size": config.get("per_device_train_batch_size")
            == job.get("mbs"),
            "training_mode": config.get("finetuning_type") == job.get("train_type"),
            "packing": config.get("packing") is bool(job.get("packing")),
            "gradient_checkpointing": config.get("gradient_checkpointing")
            is bool(job.get("gc")),
            "zero": deepspeed_evidence["matches"],
        }
    fixed_runtime_checks = {
        "bf16": isinstance(config, dict) and config.get("bf16") is True,
        "fp16_disabled": isinstance(config, dict) and config.get("fp16") is False,
        "fa3": isinstance(config, dict) and config.get("flash_attn") == "fa3",
        "fa3_orig": str(environment.get("FA3_VARIANT") or "").lower() == "orig",
        "liger": isinstance(config, dict)
        and config.get("enable_liger_kernel") is True,
        "fused_adamw": isinstance(config, dict)
        and config.get("optim") == "adamw_torch_fused",
        "cce_disabled": str(environment.get("ENABLE_CCE") or "0") == "0",
    }
    artifacts = [
        item
        for item in (
            _snapshot(runtime_job_path, root),
            _snapshot(input_job_path, root),
            _snapshot(config_path, root),
            deepspeed_snapshot,
            _snapshot(result_dir / "train.log", root),
            _snapshot(result_dir / "nvidia_smi.csv", root),
        )
        if item is not None
    ]
    job_mtime = runtime_job_path.stat().st_mtime if runtime_job_path.is_file() else None
    config_mtime = config_path.stat().st_mtime if config_path.is_file() else None
    started = _finite_float(status.get("started_unix"))
    status_runtime = status.get("runtime_fingerprint_sha256")
    rendered_runtime = rendered.get("runtime_fingerprint_sha256")
    return {
        "job_id_path_safe": job_id_valid,
        "rendered_job_matches_canonical_job": rendered.get("job") == job,
        "runtime_job_present": isinstance(runtime_job, dict),
        "runtime_job_matches_rendered_job": runtime_job == job,
        "input_job_present": isinstance(input_job, dict),
        "input_job_is_runtime_job_subset": input_subset,
        "runtime_config_present": isinstance(config, dict),
        "runtime_config_semantic_checks": semantic_checks,
        "runtime_config_semantics_complete": bool(semantic_checks)
        and all(semantic_checks.values()),
        "deepspeed_config": deepspeed_evidence,
        "fixed_design_runtime_checks": fixed_runtime_checks,
        "fixed_design_runtime_complete": all(fixed_runtime_checks.values()),
        "status_rendered_gpu_mask_matches": gpu_mask_matches,
        "status_rendered_provenance_matches": bool(status.get("provenance_sha256"))
        and status.get("provenance_sha256") == rendered.get("provenance_sha256"),
        "status_rendered_runtime_reference_consistent": bool(
            (status_runtime is None and rendered_runtime is None)
            or (
                isinstance(status_runtime, str)
                and status_runtime
                and status_runtime == rendered_runtime
            )
        ),
        "runtime_job_materialized_before_start": bool(
            started is not None and job_mtime is not None and job_mtime <= started
        ),
        "runtime_config_materialized_before_start": bool(
            started is not None and config_mtime is not None and config_mtime <= started
        ),
    }, artifacts


def _profiler_effective_roles(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = root / "artifacts" / "profiler_calibration.json"
    roles: dict[str, dict[str, Any]] = {}
    payload = read_json(path) if path.is_file() else {}
    if not isinstance(payload, dict):
        payload = {}
    calibration_job_ids = sorted({
        str(point.get("job_id"))
        for point in payload.get("calibration_points") or []
        if isinstance(point, dict) and isinstance(point.get("job_id"), str)
    })
    evaluation_job_ids = sorted({
        str(point.get("job_id"))
        for point in payload.get("evaluation_points") or []
        if isinstance(point, dict) and isinstance(point.get("job_id"), str)
    })
    for point in payload.get("calibration_points") or []:
        if not isinstance(point, dict) or not isinstance(point.get("job_id"), str):
            continue
        roles[point["job_id"]] = {
            "effective_role": point.get("role"),
            "original_role": point.get("original_role") or point.get("role"),
            "effective_role_source": "profiler_calibration_artifact",
        }
    for point in payload.get("evaluation_points") or []:
        if not isinstance(point, dict) or not isinstance(point.get("job_id"), str):
            continue
        roles[point["job_id"]] = {
            "effective_role": point.get("role"),
            "original_role": point.get("role"),
            "effective_role_source": "profiler_calibration_artifact",
        }
    return roles, {
        "source": _snapshot(path, root),
        "status": payload.get("status"),
        "evaluation_method": payload.get("evaluation_method"),
        "evaluation_mape": payload.get("evaluation_mape"),
        "artifact_calibration_job_ids": calibration_job_ids,
        "artifact_calibration_job_ids_sha256": sha256_json(
            calibration_job_ids
        ),
        "evaluation_job_ids": evaluation_job_ids,
        "evaluation_job_ids_sha256": sha256_json(evaluation_job_ids),
        "declared_calibration_points_used": sum(
            point.get("role") == "calibration"
            for point in payload.get("calibration_points") or []
            if isinstance(point, dict)
        ),
        "fallback_promotions": [
            {
                "job_id": point.get("job_id"),
                "original_role": point.get("original_role"),
                "effective_role": point.get("role"),
                "selection_policy": "recorded_fallback_calibration",
            }
            for point in payload.get("calibration_points") or []
            if isinstance(point, dict) and point.get("role") == "fallback_calibration"
        ],
        "remaining_evaluation_points": len(payload.get("evaluation_points") or []),
    }


def _validation_context(root: Path, profiler_context: dict[str, Any]) -> dict[str, Any]:
    stage_path = root / "artifacts" / "stage_decisions.json"
    stage = read_json(stage_path) if stage_path.is_file() else {}
    holdout = stage.get("resource_holdout") if isinstance(stage, dict) else {}
    holdout = holdout if isinstance(holdout, dict) else {}
    folds = holdout.get("folds") or []
    configuration_counts = [
        (_integer(fold.get("train_configurations")) or 0)
        + (_integer(fold.get("test_configurations")) or 0)
        for fold in folds
        if isinstance(fold, dict)
    ]
    return {
        "resource_loocv": {
            "source": _snapshot(stage_path, root),
            "status": holdout.get("status"),
            "method": holdout.get("method"),
            "policy": LOOCV_POLICY,
            "folds": len(folds),
            "formal_configurations": max(configuration_counts, default=0),
            "throughput_mape": holdout.get("throughput_mape"),
            "memory_mape": holdout.get("memory_mape"),
            "candidate_top1_accuracy": holdout.get("candidate_top1_accuracy"),
            "candidate_pairwise_accuracy": holdout.get(
                "candidate_pairwise_accuracy"
            ),
        },
        "profiler": profiler_context,
        "design_sources": [
            item
            for item in (
                _snapshot(root / "EXPERIMENT_DESIGN.md", root),
                _snapshot(root / "PIPELINE.md", root),
            )
            if item is not None
        ],
    }


def _is_packing_pair(job: dict[str, Any]) -> bool:
    return str(job.get("job_id") or "").startswith(("packon-", "packoff-")) or str(
        job.get("request_id") or ""
    ).startswith("pack-")


def _scenario(job: dict[str, Any]) -> dict[str, Any]:
    material = {
        "model_id": job.get("model_id"),
        "train_type": job.get("train_type"),
        "dataset_id": job.get("dataset_id"),
        "target_gbs": job.get("target_gbs"),
    }
    return {"material": material, "scenario_id": sha256_json(material)}


def _validation_design(
    job: dict[str, Any], profiler_roles: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    kind = str(job.get("kind") or "")
    if kind == "profiler":
        declared = job.get("profiler_role")
        effective = profiler_roles.get(str(job.get("job_id") or ""), {})
        return {
            "mode": "fixed_partition",
            "policy": PROFILER_POLICY,
            "declared_role": declared,
            "effective_role": effective.get("effective_role") or declared,
            "original_role": effective.get("original_role") or declared,
            "declared_role_source": "authorized_job_payload",
            "effective_role_source": effective.get("effective_role_source")
            or "authorized_job_payload_fallback",
        }
    if kind == "packing_memory_probe" or _is_packing_pair(job):
        return {
            "mode": "paired_comparison",
            "policy": PACKING_POLICY,
            "pair_id": job.get("request_id") or job.get("family_job_id"),
            "treatment": "packed" if job.get("packing") is True else "unpacked",
            "abba": False,
        }
    if kind in RESOURCE_KINDS:
        scenario = _scenario(job)
        return {
            "mode": "fold_dependent_loocv",
            "policy": LOOCV_POLICY,
            "explicit_role": "fold_dependent",
            **scenario,
        }
    return {"mode": "not_for_model_fit", "policy": None}


def _usage(job: dict[str, Any], outcome: str, measured_complete: bool) -> dict[str, Any]:
    kind = str(job.get("kind") or "")
    diagnostic = kind in DIAGNOSTIC_KINDS
    calibratable_terminal = outcome in {"success", "oom"}
    usage_class = (
        "diagnostic_only"
        if diagnostic
        else "calibration_candidate"
        if calibratable_terminal
        else "rejected_terminal"
    )
    packing_pair = kind == "throughput" and _is_packing_pair(job)
    success = outcome == "success" and measured_complete
    return {
        "class": usage_class,
        "feasibility": usage_class == "calibration_candidate"
        and calibratable_terminal,
        "memory_boundary": usage_class == "calibration_candidate"
        and kind == "memory_probe",
        "packing_memory_safety": usage_class == "calibration_candidate"
        and kind == "packing_memory_probe",
        "throughput_primary": usage_class == "calibration_candidate"
        and kind == "throughput"
        and not packing_pair
        and success,
        "throughput_formal_all": usage_class == "calibration_candidate"
        and kind == "throughput"
        and success,
        "throughput_screen_only": usage_class == "calibration_candidate"
        and kind == "throughput_screen"
        and success,
        "profiler": usage_class == "calibration_candidate"
        and kind == "profiler"
        and success,
        "packing_pair": usage_class == "calibration_candidate"
        and packing_pair
        and success,
        "prospective_acceptance_holdout": False,
    }


def _observation_for_raw_comparison(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize only the known relocatable provenance display path."""

    normalized = copy.deepcopy(row)
    provenance = normalized.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("source_path", None)
    return normalized


def _recover_record(
    row: dict[str, Any],
    *,
    raw_rebuilt_row: dict[str, Any] | None,
    root: Path,
    scheduler: dict[str, list[dict[str, Any]]],
    approval_designs: dict[str, dict[str, Any]],
    provenances: dict[str, dict[str, Any]],
    profiler_roles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = row.get("source") or {}
    status = (source.get("status") or {}).get("payload") or {}
    rendered = (source.get("rendered_run") or {}).get("payload") or {}
    job = (row.get("configuration") or {}).get("job") or {}
    job_id = str(job.get("job_id") or "")
    result_dir = _safe_project_path(root, str(source.get("result_dir") or ""))
    status_snapshot = _snapshot(result_dir / "status.json", root)
    rendered_snapshot = _snapshot(result_dir / "rendered_run.json", root)
    source_artifacts = [
        item
        for item in (status_snapshot, rendered_snapshot)
        if item is not None
    ]

    event_evidence, failure_errors, event_artifacts = _event_evidence(
        result_dir, row, job, root
    )
    source_artifacts.extend(event_artifacts)
    summaries, summary_artifacts = _summary_evidence(result_dir, job, root)
    source_artifacts.extend(summary_artifacts)
    configuration, configuration_artifacts = _configuration_evidence(
        root, result_dir, row, job
    )
    source_artifacts.extend(configuration_artifacts)
    hardware_evidence, hardware_artifacts = _hardware_evidence(
        root, result_dir, row, job, status, rendered
    )
    source_artifacts.extend(hardware_artifacts)
    raw_rebuild_matches = bool(
        isinstance(raw_rebuilt_row, dict)
        and _observation_for_raw_comparison(row)
        == _observation_for_raw_comparison(raw_rebuilt_row)
    )

    outcome = str((row.get("outcome") or {}).get("class") or "")
    return_code = _integer(status.get("return_code"))
    oom_confirmed = bool(
        outcome == "oom"
        and return_code not in (None, 0)
        and any(OOM_SIGNATURE.search(error) for error in failure_errors)
    )
    success_confirmed = bool(
        outcome == "success"
        and return_code == 0
        and summaries["complete"]
        and (row.get("quality") or {}).get("measured_ranks_complete") is True
    )
    terminal_verified = success_confirmed or oom_confirmed
    terminal = {
        "class": outcome,
        "return_code": return_code,
        "verified": terminal_verified,
        "method": (
            "complete_rank_summaries/v1"
            if success_confirmed
            else "cuda_oom_failure_event_in_status_window/v1"
            if oom_confirmed
            else "non_calibratable_terminal"
        ),
        "success_confirmed": success_confirmed,
        "cuda_oom_confirmed": oom_confirmed,
        "rank_summaries": summaries,
    }

    closure = _scheduler_closure(status, scheduler.get(job_id, []))
    status_provenance = status.get("provenance_sha256")
    approval_sha = status.get("approval_design_sha256")
    provenance_resolved = provenances.get(str(status_provenance))
    approval_resolved = approval_designs.get(str(approval_sha))
    fingerprint = row.get("fingerprint") or {}
    runtime_identity = fingerprint.get("runtime_identity")
    computed_runtime_identity = (
        sha256_json(runtime_identity) if isinstance(runtime_identity, dict) else None
    )
    runtime_exact = bool(
        computed_runtime_identity
        and computed_runtime_identity
        == fingerprint.get("computed_runtime_identity_sha256")
        == fingerprint.get("status_runtime_fingerprint_sha256")
        == fingerprint.get("rendered_runtime_fingerprint_sha256")
    )
    runtime_identity_path = result_dir / "runtime_identity.json"
    runtime_snapshot = _snapshot(runtime_identity_path, root)
    if runtime_snapshot is not None:
        source_artifacts.append(runtime_snapshot)
    if provenance_resolved is not None:
        snapshot = _snapshot(provenance_resolved["path"], root)
        if snapshot is not None:
            source_artifacts.append(snapshot)
    if approval_resolved is not None:
        snapshot = _snapshot(approval_resolved["path"], root)
        if snapshot is not None:
            source_artifacts.append(snapshot)

    provenance_reference_matches = bool(status_provenance) and status_provenance == rendered.get(
        "provenance_sha256"
    )
    archive_full = bool(
        provenance_resolved is not None and approval_resolved is not None and runtime_exact
    )
    evidence_tier = (
        "native_v2"
        if fingerprint.get("quality") == "complete"
        else "legacy_verified"
        if archive_full and closure["strength"] == "scheduler_verified"
        else "legacy_consistent"
    )
    runtime_cohort_material = {
        "provenance_sha256": status_provenance,
        "runtime_fingerprint_sha256": status.get("runtime_fingerprint_sha256")
        or rendered.get("runtime_fingerprint_sha256"),
    }
    evidence_cohort_material = {
        **runtime_cohort_material,
        "approval_design_sha256": approval_sha,
    }
    archive = {
        "provenance_reference_matches": provenance_reference_matches,
        "provenance_payload_resolved": provenance_resolved is not None,
        "runtime_identity_payload_resolved_and_matching": runtime_exact,
        "approval_design_reference_present": isinstance(approval_sha, str),
        "approval_design_payload_resolved": approval_resolved is not None,
        "full_payload_set": archive_full,
    }
    core_checks = {
        "canonical_observation_matches_raw_rebuild": raw_rebuild_matches,
        "status_render_source_hashes_match_canonical": bool(
            status_snapshot is not None
            and rendered_snapshot is not None
            and status_snapshot["sha256"]
            == (source.get("status") or {}).get("sha256")
            and rendered_snapshot["sha256"]
            == (source.get("rendered_run") or {}).get("sha256")
        ),
        "rendered_job_matches_canonical_job": configuration[
            "rendered_job_matches_canonical_job"
        ],
        "job_id_path_safe": configuration["job_id_path_safe"],
        "runtime_job_matches": configuration["runtime_job_matches_rendered_job"],
        "input_job_is_runtime_job_subset": configuration[
            "input_job_is_runtime_job_subset"
        ],
        "runtime_job_materialized_before_start": configuration[
            "runtime_job_materialized_before_start"
        ],
        "runtime_config_materialized_before_start": configuration[
            "runtime_config_materialized_before_start"
        ],
        "runtime_config_semantics_complete": configuration[
            "runtime_config_semantics_complete"
        ],
        "fixed_design_runtime_complete": configuration[
            "fixed_design_runtime_complete"
        ],
        "status_rendered_gpu_mask_matches": configuration[
            "status_rendered_gpu_mask_matches"
        ],
        "status_rendered_provenance_matches": configuration[
            "status_rendered_provenance_matches"
        ],
        "status_rendered_runtime_reference_consistent": configuration[
            "status_rendered_runtime_reference_consistent"
        ],
        "hardware_campaign_mask_and_telemetry_complete": hardware_evidence[
            "complete"
        ],
        "event_files_match": event_evidence[
            "event_file_hashes_match_canonical"
        ],
        "events_well_formed": event_evidence["malformed_lines"] == 0
        and event_evidence["events_missing_time"] == 0,
        "rank_events_complete": event_evidence["rank_set_complete"],
        "event_file_rank_binding_complete": event_evidence[
            "event_file_rank_binding_complete"
        ],
        "train_begin_binding_complete": event_evidence[
            "train_begin_rank_set_complete"
        ]
        and event_evidence["train_begin_once_per_rank"]
        and event_evidence["train_begin_world_size_matches"]
        and event_evidence["train_begin_metadata_matches_runtime_job"],
        "terminal_verified": terminal_verified,
    }
    limitations = [
        "retrospective_recovery_not_native_run_bound_v2",
        "gpu_sku_uuid_and_topology_not_attested_per_attempt",
    ]
    if closure["strength"] != "scheduler_verified":
        limitations.append("scheduler_terminal_closure_missing")
    if provenance_resolved is None:
        limitations.append("provenance_payload_not_archived")
    if not runtime_exact:
        limitations.append("runtime_identity_payload_not_archived_or_mismatched")
    if approval_resolved is None:
        limitations.append("approval_design_payload_not_archived")
    if not terminal_verified:
        limitations.append("terminal_not_a_calibration_label")
    if not all(core_checks.values()) and terminal_verified:
        limitations.append("historical_core_evidence_incomplete")

    measured_complete = bool(
        raw_rebuild_matches
        and (raw_rebuilt_row.get("quality") or {}).get(
            "measured_ranks_complete"
        )
        is True
    )
    usage = _usage(job, outcome, measured_complete)
    if usage["class"] == "calibration_candidate" and not all(core_checks.values()):
        usage = {key: False for key in usage}
        usage["class"] = "rejected_terminal"
        usage["prospective_acceptance_holdout"] = False

    source_artifacts = sorted(
        {item["path"]: item for item in source_artifacts}.values(),
        key=lambda item: item["path"],
    )
    source_manifest_sha256 = sha256_json(source_artifacts)
    record = {
        "schema": RECORD_SCHEMA,
        "source_observation_id": row.get("observation_id"),
        "source_observation_sha256": sha256_json(row),
        "job_id": job_id,
        "legacy_attempt_sha256": (row.get("attempt") or {}).get(
            "legacy_attempt_sha256"
        ),
        "evidence_tier": evidence_tier,
        "attempt_integrity": {
            **closure,
            "event_window": event_evidence,
        },
        "raw_rebuild": {
            "present": isinstance(raw_rebuilt_row, dict),
            "observation_sha256": (
                sha256_json(raw_rebuilt_row)
                if isinstance(raw_rebuilt_row, dict)
                else None
            ),
            "matches_canonical_except_provenance_source_path": raw_rebuild_matches,
        },
        "archive_completeness": archive,
        "configuration_evidence": configuration,
        "hardware_evidence": hardware_evidence,
        "terminal_evidence": terminal,
        "runtime": {
            "runtime_cohort_id": sha256_json(runtime_cohort_material),
            "runtime_cohort_material": runtime_cohort_material,
            "evidence_cohort_id": sha256_json(evidence_cohort_material),
            "evidence_cohort_material": evidence_cohort_material,
        },
        "validation_design": _validation_design(job, profiler_roles),
        "measurement_eligibility": usage,
        "core_checks": core_checks,
        "limitations": sorted(set(limitations)),
        "source_artifacts": source_artifacts,
        "source_manifest_sha256": source_manifest_sha256,
    }
    record["recovery_id"] = sha256_json(
        {
            "schema": RECORD_SCHEMA,
            "source_observation_id": record["source_observation_id"],
            "source_observation_sha256": record["source_observation_sha256"],
            "legacy_attempt_sha256": record["legacy_attempt_sha256"],
            "raw_rebuild_observation_sha256": record["raw_rebuild"][
                "observation_sha256"
            ],
            "source_manifest_sha256": source_manifest_sha256,
        }
    )
    record["evidence_sha256"] = sha256_json(record)
    return record


def _summary_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_tiers = Counter(record["evidence_tier"] for record in records)
    usage = Counter(record["measurement_eligibility"]["class"] for record in records)
    attempt = Counter(record["attempt_integrity"]["strength"] for record in records)
    archive = Counter(
        "full_payload" if record["archive_completeness"]["full_payload_set"] else "hash_only"
        for record in records
    )
    routes = Counter()
    for record in records:
        for route, allowed in record["measurement_eligibility"].items():
            if route != "class" and allowed is True:
                routes[route] += 1
    display = Counter()
    for record in records:
        usage_class = record["measurement_eligibility"]["class"]
        if usage_class == "diagnostic_only":
            display["diagnostic_only"] += 1
        elif usage_class == "rejected_terminal":
            display["rejected"] += 1
        else:
            display[record["evidence_tier"]] += 1
    return {
        "records": len(records),
        "evidence_tiers": dict(sorted(evidence_tiers.items())),
        "attempt_integrity": dict(sorted(attempt.items())),
        "archive_completeness": dict(sorted(archive.items())),
        "usage_classes": dict(sorted(usage.items())),
        "measurement_routes": dict(sorted(routes.items())),
        "display_buckets": dict(sorted(display.items())),
        "runtime_cohorts": len(
            {record["runtime"]["runtime_cohort_id"] for record in records}
        ),
        "evidence_cohorts": len(
            {record["runtime"]["evidence_cohort_id"] for record in records}
        ),
    }


def recover(
    observations_path: Path,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    observations_path = observations_path.resolve()
    rows = _read_observations(observations_path)
    raw_rebuilt_rows = export_observations(project_root)
    raw_rebuilt_by_id: dict[str, dict[str, Any]] = {}
    for raw_row in raw_rebuilt_rows:
        observation_id = raw_row.get("observation_id")
        if not isinstance(observation_id, str) or observation_id in raw_rebuilt_by_id:
            raise ValueError(
                "Raw H800 observation rebuild produced a missing or duplicate "
                "observation_id"
            )
        raw_rebuilt_by_id[observation_id] = raw_row
    scheduler, scheduler_context = _scheduler_index(project_root)
    approval_designs = _approval_design_index(project_root)
    provenances = _provenance_index(project_root)
    profiler_roles, profiler_context = _profiler_effective_roles(project_root)
    records = [
        _recover_record(
            row,
            raw_rebuilt_row=raw_rebuilt_by_id.get(str(row.get("observation_id"))),
            root=project_root,
            scheduler=scheduler,
            approval_designs=approval_designs,
            provenances=provenances,
            profiler_roles=profiler_roles,
        )
        for row in rows
    ]
    report = {
        "schema": SCHEMA,
        "project_root": str(project_root),
        "source_observations": {
            "path": str(observations_path),
            "sha256": sha256_file(observations_path),
            "schema": OBSERVATION_SCHEMA,
        },
        "raw_rebuild": {
            "records": len(raw_rebuilt_rows),
            "observation_ids_sha256": sha256_json(
                sorted(raw_rebuilt_by_id)
            ),
            "canonical_observation_ids_sha256": sha256_json(
                sorted(str(row.get("observation_id")) for row in rows)
            ),
            "matching_records": sum(
                record["raw_rebuild"][
                    "matches_canonical_except_provenance_source_path"
                ]
                for record in records
            ),
        },
        "policy": {
            "native_v2_semantics_unchanged": True,
            "historical_evidence_is_retrospective": True,
            "evidence_tier_is_independent_of_measurement_usage": True,
            "legacy_verified_requires": [
                "unique_scheduler_terminal_closure",
                "resolved_provenance_payload",
                "resolved_runtime_identity_payload",
                "resolved_approval_design_payload",
            ],
            "resource_partition": LOOCV_POLICY,
            "packing_partition": PACKING_POLICY,
            "profiler_partition": PROFILER_POLICY,
        },
        "scheduler": scheduler_context,
        "validation_context": _validation_context(project_root, profiler_context),
        "implementation_sources": [
            item
            for item in (
                _snapshot(
                    project_root / "scripts" / "recover_h800_historical_evidence.py",
                    project_root,
                ),
                _snapshot(
                    project_root / "scripts" / "export_h800_observations.py",
                    project_root,
                ),
                _snapshot(project_root / "scripts" / "common.py", project_root),
            )
            if item is not None
        ],
        "counts": _summary_counts(records),
        "records": records,
    }
    material = dict(report)
    report["report_sha256"] = sha256_json(material)
    return report


def _record_digest(record: dict[str, Any]) -> str:
    material = dict(record)
    material.pop("evidence_sha256", None)
    return sha256_json(material)


def validate_recovery_report(
    report: dict[str, Any],
    observations_path: Path,
    *,
    verify_source_files: bool = False,
    project_root: Path | None = None,
) -> list[str]:
    reasons: list[str] = []
    observations_path = observations_path.resolve()
    if project_root is None:
        project_root = (
            observations_path.parent.parent
            if observations_path.parent.name == "artifacts"
            else ROOT
        )
    trusted_project_root = project_root.resolve()
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        return ["historical_recovery_schema_invalid"]
    material = dict(report)
    claimed_report_sha256 = material.pop("report_sha256", None)
    if claimed_report_sha256 != sha256_json(material):
        reasons.append("historical_recovery_report_hash_mismatch")
    source = report.get("source_observations") or {}
    if source.get("sha256") != sha256_file(observations_path):
        reasons.append("historical_recovery_source_observation_hash_mismatch")
    if Path(str(source.get("path") or "")).resolve() != observations_path:
        reasons.append("historical_recovery_source_observation_path_mismatch")
    try:
        reported_project_root = Path(str(report.get("project_root") or "")).resolve()
    except (OSError, ValueError):
        reported_project_root = Path("/__invalid_historical_project_root__")
    if reported_project_root != trusted_project_root:
        reasons.append("historical_recovery_project_root_mismatch")
    canonical_rows = _read_observations(observations_path)
    canonical_by_id: dict[str, dict[str, Any]] = {}
    for row in canonical_rows:
        observation_id = row.get("observation_id")
        if not isinstance(observation_id, str) or observation_id in canonical_by_id:
            reasons.append(
                "historical_recovery_canonical_observation_id_missing_or_duplicate"
            )
            continue
        canonical_by_id[observation_id] = row
    raw_rebuilt_by_id: dict[str, dict[str, Any]] = {}
    if verify_source_files:
        for raw_row in export_observations(trusted_project_root):
            observation_id = raw_row.get("observation_id")
            if not isinstance(observation_id, str) or observation_id in raw_rebuilt_by_id:
                reasons.append(
                    "historical_recovery_raw_observation_id_missing_or_duplicate"
                )
                continue
            raw_rebuilt_by_id[observation_id] = raw_row
    records = report.get("records")
    if not isinstance(records, list):
        return sorted(set(reasons + ["historical_recovery_records_missing"]))
    validation_context = report.get("validation_context") or {}
    profiler_context = validation_context.get("profiler") or {}
    profiler_membership: set[str] = set()
    for list_field, digest_field in (
        (
            "artifact_calibration_job_ids",
            "artifact_calibration_job_ids_sha256",
        ),
        ("evaluation_job_ids", "evaluation_job_ids_sha256"),
    ):
        values = profiler_context.get(list_field)
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            or values != sorted(set(values))
        ):
            reasons.append(f"historical_recovery_profiler_{list_field}_invalid")
            continue
        profiler_membership.update(values)
        if profiler_context.get(digest_field) != sha256_json(values):
            reasons.append(f"historical_recovery_profiler_{digest_field}_mismatch")
    expected_profiler_roles: dict[str, dict[str, Any]] = {}
    if verify_source_files:
        expected_profiler_roles, expected_profiler_context = (
            _profiler_effective_roles(trusted_project_root)
        )
        for field in (
            "artifact_calibration_job_ids",
            "artifact_calibration_job_ids_sha256",
            "evaluation_job_ids",
            "evaluation_job_ids_sha256",
        ):
            if profiler_context.get(field) != expected_profiler_context.get(field):
                reasons.append(
                    f"historical_recovery_profiler_{field}_source_mismatch"
                )
    observation_ids: set[str] = set()
    recovery_ids: set[str] = set()
    if verify_source_files:
        resource_context = validation_context.get("resource_loocv") or {}
        top_level_artifacts = [
            (report.get("scheduler") or {}).get("source"),
            resource_context.get("source"),
            profiler_context.get("source"),
            *(validation_context.get("design_sources") or []),
            *(report.get("implementation_sources") or []),
        ]
        for index, artifact in enumerate(top_level_artifacts):
            if artifact is None:
                continue
            if not isinstance(artifact, dict):
                reasons.append(f"context_source_{index}_invalid")
                continue
            try:
                path = _safe_project_path(
                    trusted_project_root, artifact.get("path") or ""
                )
            except (ValueError, OSError):
                reasons.append(f"context_source_{index}_outside_project")
                continue
            if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
                reasons.append(f"context_source_{index}_missing_or_changed")
    for index, record in enumerate(records):
        prefix = f"record_{index}"
        if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
            reasons.append(f"{prefix}_schema_invalid")
            continue
        observation_id = record.get("source_observation_id")
        recovery_id = record.get("recovery_id")
        if not isinstance(observation_id, str) or observation_id in observation_ids:
            reasons.append(f"{prefix}_source_observation_id_missing_or_duplicate")
        else:
            observation_ids.add(observation_id)
        canonical_row = canonical_by_id.get(str(observation_id))
        if canonical_row is None:
            reasons.append(f"{prefix}_source_observation_not_found")
        else:
            if record.get("source_observation_sha256") != sha256_json(canonical_row):
                reasons.append(f"{prefix}_source_observation_hash_mismatch")
            expected_job = (
                (canonical_row.get("configuration") or {}).get("job") or {}
            )
            if record.get("job_id") != expected_job.get("job_id"):
                reasons.append(f"{prefix}_job_id_mismatch")
            expected_attempt = (canonical_row.get("attempt") or {}).get(
                "legacy_attempt_sha256"
            )
            if record.get("legacy_attempt_sha256") != expected_attempt:
                reasons.append(f"{prefix}_legacy_attempt_mismatch")
            if expected_job.get("kind") == "profiler":
                design = record.get("validation_design") or {}
                if design.get("declared_role") != expected_job.get("profiler_role"):
                    reasons.append(f"{prefix}_profiler_declared_role_mismatch")
                if design.get("declared_role_source") != "authorized_job_payload":
                    reasons.append(
                        f"{prefix}_profiler_declared_role_source_invalid"
                    )
                expected_effective_source = (
                    "profiler_calibration_artifact"
                    if record.get("job_id") in profiler_membership
                    else "authorized_job_payload_fallback"
                )
                if design.get("effective_role_source") != expected_effective_source:
                    reasons.append(
                        f"{prefix}_profiler_effective_role_source_mismatch"
                    )
                if verify_source_files and design != _validation_design(
                    expected_job, expected_profiler_roles
                ):
                    reasons.append(f"{prefix}_profiler_validation_design_mismatch")
        if not isinstance(recovery_id, str) or recovery_id in recovery_ids:
            reasons.append(f"{prefix}_recovery_id_missing_or_duplicate")
        else:
            recovery_ids.add(recovery_id)
        if record.get("evidence_tier") not in EVIDENCE_TIERS:
            reasons.append(f"{prefix}_evidence_tier_invalid")
        usage = record.get("measurement_eligibility") or {}
        if usage.get("class") not in USAGE_CLASSES:
            reasons.append(f"{prefix}_usage_class_invalid")
        if record.get("evidence_sha256") != _record_digest(record):
            reasons.append(f"{prefix}_evidence_hash_mismatch")
        artifacts = record.get("source_artifacts")
        if not isinstance(artifacts, list) or record.get(
            "source_manifest_sha256"
        ) != sha256_json(artifacts):
            reasons.append(f"{prefix}_source_manifest_hash_mismatch")
            continue
        raw_rebuild = record.get("raw_rebuild") or {}
        expected_recovery_id = sha256_json(
            {
                "schema": RECORD_SCHEMA,
                "source_observation_id": record.get("source_observation_id"),
                "source_observation_sha256": record.get(
                    "source_observation_sha256"
                ),
                "legacy_attempt_sha256": record.get("legacy_attempt_sha256"),
                "raw_rebuild_observation_sha256": raw_rebuild.get(
                    "observation_sha256"
                ),
                "source_manifest_sha256": record.get("source_manifest_sha256"),
            }
        )
        if recovery_id != expected_recovery_id:
            reasons.append(f"{prefix}_recovery_id_mismatch")
        core_checks = record.get("core_checks") or {}
        raw_matches = raw_rebuild.get(
            "matches_canonical_except_provenance_source_path"
        )
        if core_checks.get("canonical_observation_matches_raw_rebuild") is not raw_matches:
            reasons.append(f"{prefix}_raw_rebuild_core_check_mismatch")
        if usage.get("class") == "calibration_candidate" and (
            not isinstance(core_checks, dict)
            or not core_checks
            or not all(value is True for value in core_checks.values())
        ):
            reasons.append(f"{prefix}_calibration_usage_has_incomplete_core_evidence")
        if verify_source_files and canonical_row is not None:
            raw_row = raw_rebuilt_by_id.get(str(observation_id))
            actual_raw_match = bool(
                raw_row is not None
                and _observation_for_raw_comparison(raw_row)
                == _observation_for_raw_comparison(canonical_row)
            )
            if raw_matches is not actual_raw_match:
                reasons.append(f"{prefix}_raw_rebuild_claim_mismatch")
            expected_raw_sha = sha256_json(raw_row) if raw_row is not None else None
            if raw_rebuild.get("observation_sha256") != expected_raw_sha:
                reasons.append(f"{prefix}_raw_rebuild_hash_mismatch")
        if verify_source_files:
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    reasons.append(f"{prefix}_source_artifact_invalid")
                    continue
                try:
                    path = _safe_project_path(
                        trusted_project_root, artifact.get("path") or ""
                    )
                except (ValueError, OSError):
                    reasons.append(f"{prefix}_source_artifact_outside_project")
                    continue
                if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
                    reasons.append(f"{prefix}_source_artifact_missing_or_changed")
    if observation_ids != set(canonical_by_id):
        reasons.append("historical_recovery_record_observation_set_mismatch")
    raw_context = report.get("raw_rebuild") or {}
    if raw_context.get("records") != len(canonical_by_id):
        reasons.append("historical_recovery_raw_record_count_mismatch")
    if raw_context.get("matching_records") != sum(
        (record.get("raw_rebuild") or {}).get(
            "matches_canonical_except_provenance_source_path"
        )
        is True
        for record in records
        if isinstance(record, dict)
    ):
        reasons.append("historical_recovery_raw_matching_count_mismatch")
    if raw_context.get("canonical_observation_ids_sha256") != sha256_json(
        sorted(canonical_by_id)
    ):
        reasons.append("historical_recovery_canonical_id_set_hash_mismatch")
    if verify_source_files:
        if set(raw_rebuilt_by_id) != set(canonical_by_id):
            reasons.append("historical_recovery_raw_observation_set_mismatch")
        if raw_context.get("observation_ids_sha256") != sha256_json(
            sorted(raw_rebuilt_by_id)
        ):
            reasons.append("historical_recovery_raw_id_set_hash_mismatch")
    if report.get("counts") != _summary_counts(
        [record for record in records if isinstance(record, dict)]
    ):
        reasons.append("historical_recovery_counts_mismatch")
    return sorted(set(reasons))


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(
                report,
                output,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts" / "canonical_h800_observations.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "historical_h800_recovery.json",
    )
    parser.add_argument("--verify-source-files", action="store_true")
    args = parser.parse_args()
    report = recover(args.input, args.project_root)
    validation = validate_recovery_report(
        report,
        args.input,
        verify_source_files=args.verify_source_files,
        project_root=args.project_root,
    )
    if validation:
        raise SystemExit(f"Historical recovery validation failed: {validation}")
    write_report(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": report["report_sha256"],
                "counts": report["counts"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
