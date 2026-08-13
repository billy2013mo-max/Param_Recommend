#!/usr/bin/env python3
"""Export terminal H800 runs as one canonical JSON object per attempt.

The result tree is treated as immutable input.  In particular, metric event
files are append-only across retries, so observations are rebuilt only from
events whose ``time_unix`` falls inside the terminal status attempt window.
This exporter deliberately rejects 4090 campaign inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from common import ROOT, read_json, sha256_file, sha256_json
from runtime_evidence import (
    RuntimeEvidenceError,
    validate_runtime_model_manifest,
)


SCHEMA = "sft_efficiency_observation/v2"
EXECUTION_INPUTS_SCHEMA = "sft_execution_inputs/v2"
EXECUTION_FINGERPRINT_SCHEMA = "sft_execution_fingerprint/v2"
RUNTIME_HARDWARE_SCHEMA = "sft_runtime_hardware/v2"
RUNTIME_MECHANISM_SCHEMA = "sft_runtime_mechanism/v2"
EXECUTION_AUTHORIZATION_SCHEMA = "sft_execution_authorization/v2"
SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
STATIC_EXECUTION_COMPONENTS = (
    "runtime_identity_sha256",
    "runtime_config_sha256",
    "runtime_metadata_sha256",
    "deepspeed_config_sha256",
    "command_sha256",
    "environment_sha256",
    "runtime_hardware_manifest_sha256",
    "live_topology_sha256",
    "declared_hardware_sha256",
    "declared_model_manifest_sha256",
    "dataset_manifest_sha256",
    "job_payload_sha256",
    "authorization_sha256",
    "approval_evidence_sha256",
    "patch_set_sha256",
    "runtime_mechanism_manifest_sha256",
    "runtime_mechanism_source_manifest_sha256",
    "runtime_mechanism_fingerprint_sha256",
    "provenance_sha256",
)
RUNTIME_EXECUTION_COMPONENTS = (
    "execution_inputs_sha256",
    "runtime_model_inventory_sha256",
    "runtime_model_manifest_set_sha256",
)
REQUIRED_EXECUTION_COMPONENTS = (
    *STATIC_EXECUTION_COMPONENTS,
    *RUNTIME_EXECUTION_COMPONENTS,
)
TERMINAL_OUTCOME_CLASSES = {
    "success": "success",
    "oom": "oom",
    "failed": "software_failure",
    "software_failure": "software_failure",
    "incomplete_metrics": "infrastructure_failure",
    "infrastructure_failure": "infrastructure_failure",
}
WORK_COUNTERS = (
    "computed_tokens",
    "effective_tokens",
    "label_tokens",
    "logical_samples",
    "physical_batches",
    "computed_attention_token_pairs",
    "effective_attention_token_pairs",
)
OOM_ALLOCATION = re.compile(
    r"(?:tried|trying) to allocate\s+([0-9]+(?:\.[0-9]+)?)\s*"
    r"(bytes?|kib|mib|gib|kb|mb|gb)",
    re.IGNORECASE,
)
OOM_CAPACITY = re.compile(
    r"total capacity of\s+([0-9]+(?:\.[0-9]+)?)\s*"
    r"(bytes?|kib|mib|gib|kb|mb|gb)",
    re.IGNORECASE,
)
OOM_FREE = re.compile(
    r"(?:of which|with)\s+([0-9]+(?:\.[0-9]+)?)\s*"
    r"(bytes?|kib|mib|gib|kb|mb|gb)\s+(?:is\s+)?free",
    re.IGNORECASE,
)


class UnsupportedHardwareError(ValueError):
    """Raised rather than silently mixing a non-H800 campaign into output."""


def _normalized_terminal_outcome(value: Any) -> str | None:
    return TERMINAL_OUTCOME_CLASSES.get(str(value or ""))


def _terminal_evidence_is_calibratable(evidence: Any) -> bool:
    """Mirror the runner's fail-closed success/OOM eligibility policy."""

    if not isinstance(evidence, dict):
        return False
    classification = evidence.get("classification")
    return bool(
        (
            classification == "success"
            and evidence.get("return_code") == 0
            and evidence.get("summaries_complete") is True
        )
        or (
            classification == "oom"
            and evidence.get("cuda_oom_confirmed") is True
        )
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _bytes(value: str, unit: str) -> int:
    multipliers = {
        "byte": 1,
        "bytes": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
    }
    return int(float(value) * multipliers[unit.lower()])


def _largest_parsed(errors: Iterable[str], pattern: re.Pattern[str]) -> int | None:
    values = [
        _bytes(match.group(1), match.group(2))
        for error in errors
        for match in pattern.finditer(error)
    ]
    return max(values) if values else None


def _validate_h800_source(
    project_root: Path,
    results_dir: Path,
    hardware: dict[str, Any],
    experiment: dict[str, Any],
) -> None:
    source_identity = " ".join(
        str(value)
        for value in (
            hardware.get("gpu_id"),
            hardware.get("name_reported_by_driver"),
            (experiment.get("training_scope") or {}).get("gpu_type"),
            results_dir,
        )
        if value is not None
    ).lower()
    if "4090" in source_identity or "rtx4090" in source_identity:
        raise UnsupportedHardwareError(
            f"4090 campaign input is forbidden for the H800 exporter: {results_dir}"
        )
    if "h800" not in source_identity:
        raise UnsupportedHardwareError(
            "The configured campaign is not identifiable as H800; refusing export"
        )
    if results_dir.resolve() != (project_root / "results").resolve():
        # Tests and explicitly relocated copies are allowed, but never a path
        # nested under the known 4090 campaign tree (checked above).
        return


def _validate_h800_job(job: dict[str, Any]) -> None:
    identity = " ".join(
        str(job.get(key) or "")
        for key in ("gpu_type", "hardware_id", "campaign_id", "phase_id")
    ).lower()
    if "4090" in identity or "rtx4090" in identity:
        raise UnsupportedHardwareError(
            f"Job {job.get('job_id')} carries 4090 identity and cannot be exported"
        )
    if job.get("gpu_type") and "h800" not in str(job["gpu_type"]).lower():
        raise UnsupportedHardwareError(
            f"Job {job.get('job_id')} is explicitly assigned to non-H800 hardware"
        )


def _provenance_index(project_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    artifacts = project_root / "artifacts"
    candidates = [artifacts / "provenance.json"]
    candidates.extend(sorted((artifacts / "provenance_history").glob("*.json")))
    index: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            index[sha256_file(path)] = (path, _load_json(path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return index


def _read_events(
    result_dir: Path,
    started: float,
    finished: float,
    *,
    expected_job_id: str | None = None,
    expected_execution_attempt_id: str | None = None,
    require_attempt_binding: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    total_lines = 0
    malformed_lines = 0
    missing_time = 0
    outside_attempt = 0
    missing_attempt_binding = 0
    mismatched_attempt_binding = 0
    files = sorted((result_dir / "metrics").glob("events.rank*.jsonl"))
    file_records = []
    for path in files:
        kept = 0
        lines = 0
        with path.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                if not line.strip():
                    continue
                total_lines += 1
                lines += 1
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue
                if not isinstance(event, dict):
                    malformed_lines += 1
                    continue
                timestamp = _finite_float(event.get("time_unix"))
                if timestamp is None:
                    missing_time += 1
                    continue
                if timestamp < started or timestamp > finished:
                    outside_attempt += 1
                    continue
                if require_attempt_binding:
                    if not event.get("job_id") or not event.get(
                        "execution_attempt_id"
                    ):
                        missing_attempt_binding += 1
                        continue
                    if (
                        event.get("job_id") != expected_job_id
                        or event.get("execution_attempt_id")
                        != expected_execution_attempt_id
                    ):
                        mismatched_attempt_binding += 1
                        continue
                selected.append(event)
                kept += 1
        file_records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "lines": lines,
                "events_in_attempt": kept,
            }
        )
    return selected, {
        "policy": (
            "exact_job_attempt_binding_and_inclusive_status_window"
            if require_attempt_binding
            else "legacy_inclusive_status_started_unix_finished_unix"
        ),
        "started_unix": started,
        "finished_unix": finished,
        "event_files": file_records,
        "total_nonblank_lines": total_lines,
        "events_in_attempt": len(selected),
        "events_outside_attempt": outside_attempt,
        "events_missing_time": missing_time,
        "events_missing_attempt_binding": missing_attempt_binding,
        "events_mismatched_attempt_binding": mismatched_attempt_binding,
        "attempt_binding_required": require_attempt_binding,
        "attempt_binding_complete": bool(
            require_attempt_binding
            and missing_attempt_binding == 0
            and mismatched_attempt_binding == 0
        ),
        "malformed_lines": malformed_lines,
    }


def _aggregate_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    measured_seconds_by_rank: dict[int, float] = defaultdict(float)
    measured_steps_by_rank: dict[int, int] = defaultdict(int)
    work = {key: 0 for key in WORK_COUNTERS}
    work_present = {key: False for key in WORK_COUNTERS}
    allocated_values: list[int] = []
    reserved_values: list[int] = []
    errors: list[str] = []
    ranks: set[int] = set()

    for event in events:
        rank = _integer(event.get("rank"))
        if rank is not None:
            ranks.add(rank)
        memory = event.get("memory") or {}
        if isinstance(memory, dict):
            for key in ("allocated", "max_allocated"):
                value = _integer(memory.get(key))
                if value is not None and value >= 0:
                    allocated_values.append(value)
            for key in ("reserved", "max_reserved"):
                value = _integer(memory.get(key))
                if value is not None and value >= 0:
                    reserved_values.append(value)
        if event.get("event") == "failure" and event.get("error"):
            errors.append(str(event["error"]))
        if event.get("event") != "step_end" or bool(event.get("is_warmup")):
            continue
        if rank is None:
            continue
        seconds = _finite_float(event.get("step_seconds"))
        if seconds is not None and seconds >= 0:
            measured_seconds_by_rank[rank] += seconds
        measured_steps_by_rank[rank] += 1
        tokens = event.get("tokens") or {}
        if not isinstance(tokens, dict):
            continue
        for key in WORK_COUNTERS:
            value = _integer(tokens.get(key))
            if value is not None:
                work[key] += value
                work_present[key] = True

    measured_seconds = (
        max(measured_seconds_by_rank.values())
        if measured_seconds_by_rank
        else None
    )
    measured_steps = (
        max(measured_steps_by_rank.values()) if measured_steps_by_rank else 0
    )
    aggregate_work = {
        key: work[key] if work_present[key] else None for key in WORK_COUNTERS
    }
    rates = {}
    for key in ("computed_tokens", "effective_tokens", "logical_samples"):
        value = aggregate_work[key]
        rates[f"{key}_per_second"] = (
            value / measured_seconds
            if value is not None and measured_seconds and measured_seconds > 0
            else None
        )
    return {
        "rank_aggregation": {
            "elapsed_time": "max_of_per_rank_measured_step_sums",
            "memory": "max_across_all_in_attempt_rank_events",
            "work_counters": "sum_across_all_measured_non_warmup_rank_events",
        },
        "ranks_with_events": sorted(ranks),
        "measured_seconds": measured_seconds,
        "measured_step_count": measured_steps,
        "measured_seconds_by_rank": {
            str(rank): value
            for rank, value in sorted(measured_seconds_by_rank.items())
        },
        "measured_steps_by_rank": {
            str(rank): value
            for rank, value in sorted(measured_steps_by_rank.items())
        },
        "mean_step_seconds": (
            measured_seconds / measured_steps
            if measured_seconds is not None and measured_steps > 0
            else None
        ),
        "work": aggregate_work,
        "rates": rates,
        "memory": {
            "max_allocated_bytes": max(allocated_values) if allocated_values else None,
            "max_reserved_bytes": max(reserved_values) if reserved_values else None,
            "values_are_observed_not_imputed": True,
        },
        "failure_errors": errors,
    }


def _load_yaml_snapshot(path_value: Any) -> tuple[Path | None, dict[str, Any] | None]:
    if not path_value:
        return None, None
    path = Path(str(path_value))
    if not path.is_file():
        return path, None
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return path, value if isinstance(value, dict) else None


def _snapshot_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "payload": _load_json(path),
    }


def _execution_manifest_reasons(
    manifest: dict[str, Any],
    job_id: str,
    execution_attempt_id: str,
    expected_components: dict[str, str | None],
) -> list[str]:
    reasons = []
    if manifest.get("schema") != EXECUTION_FINGERPRINT_SCHEMA:
        reasons.append("execution_fingerprint_schema_invalid")
    if manifest.get("job_id") != job_id:
        reasons.append("execution_fingerprint_job_binding_mismatch")
    if manifest.get("execution_attempt_id") != execution_attempt_id:
        reasons.append("execution_fingerprint_attempt_binding_mismatch")
    components = manifest.get("components")
    if not isinstance(components, dict):
        return reasons + ["execution_fingerprint_components_missing"]
    for name in REQUIRED_EXECUTION_COMPONENTS:
        value = components.get(name)
        if not isinstance(value, str) or not SHA256.fullmatch(value):
            reasons.append(f"execution_component_{name}_missing_or_invalid")
            continue
        expected = expected_components.get(name)
        if expected is not None and value.lower() != expected.lower():
            reasons.append(f"execution_component_{name}_mismatch")
    return reasons


def _result_child_path(result_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = result_dir / candidate
    try:
        candidate.resolve().relative_to(result_dir.resolve())
    except ValueError:
        return None
    # Attempt-scoped runs write per-attempt evidence under ``attempts/<id>/`` and
    # record bare filenames (relative to the attempt dir) inside the fingerprint
    # manifest.  If the root-relative path is absent, fall back to a unique
    # matching file under a single attempt directory, still constrained to stay
    # inside the result directory.
    if not candidate.exists():
        attempts_root = result_dir / "attempts"
        if attempts_root.is_dir():
            matches = sorted(attempts_root.glob(f"*/{value}"))
            inside = [
                match
                for match in matches
                if _is_inside(match, result_dir) and match.is_file()
            ]
            if len(inside) == 1:
                return inside[0]
    return candidate


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _execution_inputs_evidence(
    result_dir: Path,
    manifest: dict[str, Any],
    job_id: str,
    execution_attempt_id: str,
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    reasons: list[str] = []
    path = _result_child_path(result_dir, manifest.get("execution_inputs_path"))
    if path is None or not path.is_file():
        return None, None, ["execution_inputs_manifest_missing_or_outside_result"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None, None, ["execution_inputs_manifest_unreadable"]
    if payload.get("schema") != EXECUTION_INPUTS_SCHEMA:
        reasons.append("execution_inputs_schema_invalid")
    if payload.get("job_id") != job_id:
        reasons.append("execution_inputs_job_binding_mismatch")
    if payload.get("execution_attempt_id") != execution_attempt_id:
        reasons.append("execution_inputs_attempt_binding_mismatch")
    job_snapshot = payload.get("job_snapshot")
    if not isinstance(job_snapshot, dict):
        reasons.append("execution_inputs_job_snapshot_missing")
    else:
        job_sha256 = sha256_json(job_snapshot)
        if payload.get("job_payload_sha256") != job_sha256:
            reasons.append("execution_inputs_job_payload_hash_mismatch")
        if job_snapshot.get("job_id") != job_id:
            reasons.append("execution_inputs_job_snapshot_binding_mismatch")
    authorization = payload.get("authorization")
    if not isinstance(authorization, dict):
        reasons.append("execution_authorization_missing")
    else:
        if authorization.get("schema") != EXECUTION_AUTHORIZATION_SCHEMA:
            reasons.append("execution_authorization_schema_invalid")
        mode = authorization.get("mode")
        if mode not in {"approved", "smoke", "render"}:
            reasons.append("execution_authorization_mode_invalid")
        if authorization.get("calibration_eligible") is not (mode == "approved"):
            reasons.append("execution_authorization_calibration_policy_invalid")
        if isinstance(job_snapshot, dict) and authorization.get(
            "job_payload_sha256"
        ) != sha256_json(job_snapshot):
            reasons.append("execution_authorization_job_hash_mismatch")
        evidence = authorization.get("evidence")
        if not isinstance(evidence, dict) or authorization.get(
            "evidence_sha256"
        ) != sha256_json(evidence):
            reasons.append("execution_authorization_evidence_hash_mismatch")
        elif mode == "approved":
            for name in (
                "approval_sha256",
                "approval_design_sha256",
                "queue_sha256",
                "ordered_job_ids_sha256",
                "job_payload_sha256",
            ):
                if not isinstance(evidence.get(name), str) or not SHA256.fullmatch(
                    evidence[name]
                ):
                    reasons.append(
                        f"execution_authorization_{name}_missing_or_invalid"
                    )
            if isinstance(job_snapshot, dict) and evidence.get(
                "job_payload_sha256"
            ) != sha256_json(job_snapshot):
                reasons.append("execution_authorization_approved_job_hash_mismatch")
            ordered_ids = evidence.get("ordered_job_ids")
            if (
                not isinstance(ordered_ids, list)
                or job_id not in ordered_ids
                or evidence.get("ordered_job_ids_sha256")
                != sha256_json(ordered_ids)
            ):
                reasons.append("execution_authorization_queue_order_binding_invalid")
    components = payload.get("components")
    if not isinstance(components, dict) or set(components) != set(
        STATIC_EXECUTION_COMPONENTS
    ):
        reasons.append("execution_inputs_components_incomplete")
    return payload, sha256_file(path), reasons


def _bound_input_artifacts(
    result_dir: Path,
    execution_inputs: dict[str, Any] | None,
    *,
    job_id: str,
    execution_attempt_id: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Reload every attempt-local input whose digest was bound before launch."""

    if not isinstance(execution_inputs, dict):
        return {}, ["execution_inputs_unavailable_for_artifact_validation"]
    paths = execution_inputs.get("evidence_paths")
    components = execution_inputs.get("components")
    if not isinstance(paths, dict) or not isinstance(components, dict):
        return {}, ["execution_input_evidence_paths_missing"]
    component_by_name = {
        "runtime_identity": "runtime_identity_sha256",
        "runtime_config": "runtime_config_sha256",
        "runtime_metadata": "runtime_metadata_sha256",
        "runtime_hardware": "runtime_hardware_manifest_sha256",
        "live_topology": "live_topology_sha256",
        "declared_hardware": "declared_hardware_sha256",
        "declared_model_manifest": "declared_model_manifest_sha256",
        "dataset_manifest": "dataset_manifest_sha256",
        "deepspeed_config": "deepspeed_config_sha256",
        "runtime_mechanism": "runtime_mechanism_manifest_sha256",
        "provenance": "provenance_sha256",
    }
    required = {
        "runtime_identity",
        "runtime_config",
        "runtime_metadata",
        "runtime_hardware",
        "live_topology",
        "declared_hardware",
        "declared_model_manifest",
        "dataset_manifest",
        "runtime_mechanism",
        "provenance",
    }
    reasons: list[str] = []
    if not required.issubset(paths):
        reasons.append(
            "execution_input_evidence_paths_incomplete:"
            + ",".join(sorted(required - set(paths)))
        )
    artifacts: dict[str, dict[str, Any]] = {}
    for name, component_name in component_by_name.items():
        path_value = paths.get(name)
        if path_value in (None, "") and name == "deepspeed_config":
            continue
        path = _result_child_path(result_dir, path_value)
        if path is None or not path.is_file():
            if name in required or path_value not in (None, ""):
                reasons.append(f"execution_input_{name}_missing_or_outside_attempt")
            continue
        file_sha256 = sha256_file(path)
        expected_sha256 = components.get(component_name)
        if name == "runtime_identity":
            try:
                payload = _load_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                reasons.append("execution_input_runtime_identity_unreadable")
                continue
            actual_component_sha256 = sha256_json(payload)
        else:
            actual_component_sha256 = file_sha256
            if path.suffix in {".json", ".yaml", ".yml"}:
                try:
                    if path.suffix == ".json":
                        payload = _load_json(path)
                    else:
                        value = yaml.safe_load(path.read_text(encoding="utf-8"))
                        payload = value if isinstance(value, dict) else None
                except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError):
                    payload = None
            else:
                payload = path.read_text(encoding="utf-8", errors="replace")
        if expected_sha256 != actual_component_sha256:
            reasons.append(f"execution_input_{name}_component_hash_mismatch")
        artifacts[name] = {
            "path": str(path_value),
            "file_sha256": file_sha256,
            "component_sha256": actual_component_sha256,
            "payload_sha256": sha256_json(payload),
            "payload": payload,
        }

    hardware = (artifacts.get("runtime_hardware") or {}).get("payload")
    if not isinstance(hardware, dict):
        reasons.append("runtime_hardware_manifest_unavailable")
    else:
        if hardware.get("schema") != RUNTIME_HARDWARE_SCHEMA:
            reasons.append("runtime_hardware_schema_invalid")
        if hardware.get("job_id") != job_id:
            reasons.append("runtime_hardware_job_binding_mismatch")
        if hardware.get("execution_attempt_id") != execution_attempt_id:
            reasons.append("runtime_hardware_attempt_binding_mismatch")
        if (
            hardware.get("capture_mode") != "live_pre_execution"
            or hardware.get("all_passed") is not True
            or hardware.get("calibration_hardware_eligible") is not True
        ):
            reasons.append("runtime_hardware_not_calibration_eligible")
        devices = hardware.get("devices")
        declared = hardware.get("declared_hardware")
        if not isinstance(devices, list) or not devices:
            reasons.append("runtime_hardware_devices_missing")
        else:
            for device in devices:
                if not isinstance(device, dict):
                    reasons.append("runtime_hardware_device_invalid")
                    continue
                if device.get("name") != "NVIDIA H800":
                    reasons.append("runtime_hardware_live_sku_not_exact_h800")
                if (_integer(device.get("memory_total_bytes")) or 0) < 140 * 10**9:
                    reasons.append("runtime_hardware_live_memory_below_h800_140g")
                identity = " ".join(str(value) for value in device.values()).lower()
                if "4090" in identity:
                    reasons.append("runtime_hardware_contains_4090_identity")
        if not isinstance(declared, dict) or declared.get("gpu_id") != "local_h800_140g":
            reasons.append("runtime_hardware_declared_profile_not_h800_140g")

    mechanism = (artifacts.get("runtime_mechanism") or {}).get("payload")
    if not isinstance(mechanism, dict):
        reasons.append("runtime_mechanism_manifest_unavailable")
    else:
        if mechanism.get("schema") != RUNTIME_MECHANISM_SCHEMA:
            reasons.append("runtime_mechanism_schema_invalid")
        fingerprint = mechanism.get("fingerprint_sha256")
        material = dict(mechanism)
        material.pop("fingerprint_sha256", None)
        if fingerprint != sha256_json(material):
            reasons.append("runtime_mechanism_fingerprint_invalid")
        source_manifest = mechanism.get("source_manifest")
        if not isinstance(source_manifest, dict) or mechanism.get(
            "source_manifest_sha256"
        ) != sha256_json(source_manifest):
            reasons.append("runtime_mechanism_source_manifest_hash_mismatch")
        forbidden = {"model", "dataset", "micro_batch_size", "global_batch_size", "gpu_count"}
        excluded = set(mechanism.get("excluded_dimensions") or [])
        if not forbidden.issubset(excluded):
            reasons.append("runtime_mechanism_excluded_dimensions_incomplete")

    metadata = (artifacts.get("runtime_metadata") or {}).get("payload")
    snapshot = execution_inputs.get("job_snapshot")
    if not isinstance(metadata, dict) or not isinstance(snapshot, dict):
        reasons.append("runtime_metadata_or_job_snapshot_unavailable")
    else:
        expected_metadata = {
            **snapshot,
            "_execution_attempt_id": execution_attempt_id,
        }
        if metadata != expected_metadata:
            reasons.append("runtime_metadata_job_attempt_binding_mismatch")
    return artifacts, reasons


def _calibration_partition(job: dict[str, Any]) -> tuple[dict[str, str] | None, list[str]]:
    """Read only a partition frozen inside the authorized job payload."""

    raw = job.get("calibration_partition")
    if isinstance(raw, dict):
        role = raw.get("role")
        split_unit_id = raw.get("split_unit_id")
        policy = raw.get("policy")
    else:
        role = job.get("calibration_role") or job.get("profiler_role")
        split_unit_id = job.get("calibration_split_unit_id")
        policy = job.get("calibration_split_policy")
    # The sealed unified-resource campaign uses ``fit`` to distinguish model
    # fitting from its later, separately frozen prospective acceptance queue.
    # Canonical observations expose the older two-way vocabulary expected by
    # downstream fitters, so normalize only this exact bound policy.  Do not
    # accept ``fit`` generically: an arbitrary queue must not acquire
    # calibration authority by choosing a convenient role string.
    effective_role = (
        "calibration"
        if role == "fit" and policy == "unified_model_source_grouped_v1"
        else role
    )
    reasons = []
    if effective_role not in {"calibration", "holdout"}:
        reasons.append("authorized_job_calibration_role_missing")
    if not isinstance(split_unit_id, str) or not split_unit_id.strip():
        reasons.append("authorized_job_calibration_split_unit_missing")
    if not isinstance(policy, str) or not policy.strip():
        reasons.append("authorized_job_calibration_split_policy_missing")
    if reasons:
        return None, reasons
    return {
        "role": str(effective_role),
        "split_unit_id": str(split_unit_id),
        "policy": str(policy),
    }, []


def _runtime_model_evidence(
    result_dir: Path,
    manifest: dict[str, Any],
    job_id: str,
    expected_ranks: int,
    *,
    training_mode: str,
    runtime_hardware: dict[str, Any] | None,
) -> tuple[
    dict[str, Any] | None,
    str | None,
    str | None,
    list[dict[str, Any]],
    list[str],
]:
    reasons: list[str] = []
    references = manifest.get("runtime_model_manifests")
    attempt_id = manifest.get("execution_attempt_id")
    if not isinstance(references, list):
        return None, None, None, [], ["runtime_model_manifest_references_missing"]
    if not isinstance(attempt_id, str) or not re.fullmatch(r"[0-9a-f]{20}", attempt_id):
        reasons.append("execution_attempt_id_invalid")
    canonical_references: list[dict[str, Any]] = []
    common_inventory: dict[str, Any] | None = None
    common_inventory_sha256: str | None = None
    seen_ranks: set[int] = set()
    embedded_rank_evidence: list[dict[str, Any]] = []
    hardware_devices = (
        runtime_hardware.get("devices")
        if isinstance(runtime_hardware, dict)
        and isinstance(runtime_hardware.get("devices"), list)
        else []
    )
    for reference in references:
        if not isinstance(reference, dict):
            reasons.append("runtime_model_manifest_reference_invalid")
            continue
        rank = _integer(reference.get("rank"))
        path_value = reference.get("path")
        path = _result_child_path(result_dir, path_value)
        if rank is None or rank < 0 or rank in seen_ranks:
            reasons.append("runtime_model_manifest_rank_invalid")
            continue
        seen_ranks.add(rank)
        if path is None or not path.is_file():
            reasons.append(f"runtime_model_manifest_rank_{rank}_missing")
            continue
        file_sha256 = sha256_file(path)
        if reference.get("file_sha256") != file_sha256:
            reasons.append(f"runtime_model_manifest_rank_{rank}_file_hash_mismatch")
        try:
            rank_manifest = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            reasons.append(f"runtime_model_manifest_rank_{rank}_unreadable")
            continue
        try:
            validate_runtime_model_manifest(
                rank_manifest,
                expected_job_id=job_id,
                expected_execution_attempt_id=attempt_id,
                expected_rank=rank,
                expected_world_size=expected_ranks,
                expected_training_mode=training_mode,
                allow_unavailable_device=False,
            )
        except RuntimeEvidenceError as error:
            reasons.append(
                f"runtime_model_manifest_rank_{rank}_invalid:{error}"
            )
        inventory = rank_manifest.get("inventory")
        if not isinstance(inventory, dict):
            reasons.append(f"runtime_model_manifest_rank_{rank}_inventory_missing")
            continue
        inventory_sha256 = sha256_json(inventory)
        if (
            rank_manifest.get("inventory_sha256") != inventory_sha256
            or reference.get("inventory_sha256") != inventory_sha256
        ):
            reasons.append(f"runtime_model_manifest_rank_{rank}_inventory_hash_mismatch")
        if common_inventory_sha256 is None:
            common_inventory = inventory
            common_inventory_sha256 = inventory_sha256
        elif common_inventory_sha256 != inventory_sha256:
            reasons.append("runtime_model_inventories_differ_across_ranks")
        device = rank_manifest.get("device_attestation")
        local_rank = _integer(rank_manifest.get("local_rank"))
        if local_rank is None or local_rank >= len(hardware_devices):
            reasons.append(f"runtime_model_manifest_rank_{rank}_device_unmapped")
        elif isinstance(device, dict):
            parent = hardware_devices[local_rank]
            # torch's device properties report usable memory, which differs from
            # nvidia-smi's total by a small driver-reserved margin.  run_job's own
            # live attestation already tolerates 2%; the exporter must use the
            # same tolerance rather than demand exact equality, or every real run
            # is rejected over a ~0.4% driver-reserved difference.
            expected_memory = int(parent.get("memory_total_bytes") or 0)
            memory_tolerance = max(1024**3, int(expected_memory * 0.02))
            for field, expected in {
                "name": parent.get("name"),
                "uuid": parent.get("uuid"),
            }.items():
                actual = device.get(field)
                if field == "uuid" and actual is None:
                    continue
                if actual != expected:
                    reasons.append(
                        f"runtime_model_manifest_rank_{rank}_device_{field}_mismatch"
                    )
            actual_memory = int(device.get("total_memory_bytes") or 0)
            if abs(actual_memory - expected_memory) > memory_tolerance:
                reasons.append(
                    f"runtime_model_manifest_rank_{rank}_device_total_memory_bytes_mismatch"
                )
            capability = device.get("compute_capability")
            if isinstance(capability, dict):
                actual_capability = (
                    f"{capability.get('major')}.{capability.get('minor')}"
                )
                if actual_capability != parent.get("compute_capability"):
                    reasons.append(
                        f"runtime_model_manifest_rank_{rank}_device_compute_capability_mismatch"
                    )
        canonical_references.append(
            {
                "rank": rank,
                "path": str(path_value),
                "file_sha256": file_sha256,
                "inventory_sha256": inventory_sha256,
            }
        )
        embedded_rank_evidence.append(
            {
                "reference": reference,
                "manifest_payload_sha256": sha256_json(rank_manifest),
                "manifest_without_inventory": {
                    key: value
                    for key, value in rank_manifest.items()
                    if key != "inventory"
                },
            }
        )
    if seen_ranks != set(range(expected_ranks)):
        reasons.append("runtime_model_manifest_rank_set_incomplete")
    canonical_references.sort(key=lambda item: item["rank"])
    if references != sorted(
        references,
        key=lambda item: item.get("rank", -1) if isinstance(item, dict) else -1,
    ):
        reasons.append("runtime_model_manifest_references_not_rank_ordered")
    reference_set_sha256 = sha256_json(references) if references else None
    return (
        common_inventory,
        common_inventory_sha256,
        reference_set_sha256,
        embedded_rank_evidence,
        reasons,
    )


def _fingerprint(
    result_dir: Path,
    status: dict[str, Any],
    rendered: dict[str, Any],
    runtime_config_path: Path | None,
    runtime_config: dict[str, Any] | None,
    hardware: dict[str, Any],
    experiment: dict[str, Any],
    provenance: dict[str, Any] | None,
    project_root: Path,
) -> dict[str, Any]:
    exact_manifest_path = _result_child_path(
        result_dir, rendered.get("execution_fingerprint_path")
    )
    if exact_manifest_path is None and not rendered.get("execution_fingerprint_path"):
        legacy_path = result_dir / "execution_fingerprint.json"
        exact_manifest_path = legacy_path if legacy_path.is_file() else None
    exact_manifest = None
    if exact_manifest_path is not None and exact_manifest_path.is_file():
        try:
            exact_manifest = _load_json(exact_manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            exact_manifest = None
    computed_exact = sha256_json(exact_manifest) if exact_manifest else None
    status_exact = status.get("execution_fingerprint_sha256")
    rendered_exact = rendered.get("execution_fingerprint_sha256")
    if exact_manifest and status_exact == rendered_exact == computed_exact:
        quality = "complete"
        reasons: list[str] = []
    elif status_exact or rendered_exact or exact_manifest:
        quality = "incomplete"
        reasons = ["execution_fingerprint_manifest_missing_or_mismatched"]
    else:
        quality = "legacy_incomplete"
        reasons = ["no_run_bound_execution_fingerprint"]
    job_id = str((rendered.get("job") or {}).get("job_id") or "")
    execution_attempt_id = str(status.get("execution_attempt_id") or "")
    execution_inputs = None
    execution_inputs_sha256 = None
    bound_artifacts: dict[str, dict[str, Any]] = {}
    bound_job = rendered.get("job") or {}
    runtime_model_inventory = None
    runtime_model_inventory_sha256 = None
    runtime_model_manifest_set_sha256 = None
    runtime_rank_evidence: list[dict[str, Any]] = []
    if exact_manifest is not None:
        if rendered.get("execution_fingerprint_quality") != "complete":
            reasons.append("rendered_execution_fingerprint_quality_not_complete")
        if status.get("execution_fingerprint_quality") != "complete":
            reasons.append("status_execution_fingerprint_quality_not_complete")
        if not re.fullmatch(r"[0-9a-f]{20}", execution_attempt_id):
            reasons.append("execution_attempt_id_invalid")
        if rendered.get("execution_attempt_id") != execution_attempt_id:
            reasons.append("rendered_status_attempt_binding_mismatch")
        (
            execution_inputs,
            execution_inputs_sha256,
            input_reasons,
        ) = _execution_inputs_evidence(
            result_dir,
            exact_manifest,
            job_id,
            execution_attempt_id,
        )
        reasons.extend(input_reasons)
        bound_artifacts, artifact_reasons = _bound_input_artifacts(
            result_dir,
            execution_inputs,
            job_id=job_id,
            execution_attempt_id=execution_attempt_id,
        )
        reasons.extend(artifact_reasons)
        if isinstance(execution_inputs, dict) and isinstance(
            execution_inputs.get("job_snapshot"), dict
        ):
            bound_job = execution_inputs["job_snapshot"]
        if rendered.get("job") != bound_job:
            reasons.append("rendered_job_differs_from_authorized_job")
        expected_ranks = _integer(bound_job.get("gpu_count")) or 0
        runtime_hardware = (bound_artifacts.get("runtime_hardware") or {}).get(
            "payload"
        )
        (
            runtime_model_inventory,
            runtime_model_inventory_sha256,
            runtime_model_manifest_set_sha256,
            runtime_rank_evidence,
            runtime_model_reasons,
        ) = _runtime_model_evidence(
            result_dir,
            exact_manifest,
            job_id,
            expected_ranks,
            training_mode=str(bound_job.get("train_type") or ""),
            runtime_hardware=(
                runtime_hardware if isinstance(runtime_hardware, dict) else None
            ),
        )
        reasons.extend(runtime_model_reasons)
        final_job = exact_manifest.get("job_snapshot")
        if final_job != bound_job:
            reasons.append("execution_fingerprint_job_snapshot_mismatch")
        final_authorization = exact_manifest.get("authorization")
        input_authorization = (
            execution_inputs.get("authorization")
            if isinstance(execution_inputs, dict)
            else None
        )
        if final_authorization != input_authorization:
            reasons.append("execution_fingerprint_authorization_mismatch")
        final_outcome = exact_manifest.get("outcome")
        status_outcome = status.get("classification_evidence")
        if (
            not isinstance(final_outcome, dict)
            or final_outcome != status_outcome
            or exact_manifest.get("outcome_sha256") != sha256_json(final_outcome)
        ):
            reasons.append("execution_fingerprint_terminal_outcome_mismatch")
        expected_calibration_eligible = bool(
            isinstance(input_authorization, dict)
            and input_authorization.get("calibration_eligible") is True
            and _terminal_evidence_is_calibratable(final_outcome)
        )
        if (
            exact_manifest.get("calibration_eligible")
            is not expected_calibration_eligible
        ):
            reasons.append("execution_fingerprint_calibration_policy_mismatch")

    input_components = (
        execution_inputs.get("components")
        if isinstance(execution_inputs, dict)
        else None
    )
    expected_execution_components = dict(input_components or {})
    expected_execution_components.update(
        {
            "execution_inputs_sha256": execution_inputs_sha256,
            "runtime_model_inventory_sha256": runtime_model_inventory_sha256,
            "runtime_model_manifest_set_sha256": runtime_model_manifest_set_sha256,
        }
    )
    if exact_manifest is not None:
        final_components = exact_manifest.get("components")
        if isinstance(input_components, dict) and isinstance(final_components, dict):
            for name in STATIC_EXECUTION_COMPONENTS:
                if final_components.get(name) != input_components.get(name):
                    reasons.append(f"execution_component_{name}_input_mismatch")
        reasons.extend(
            _execution_manifest_reasons(
                exact_manifest,
                job_id,
                execution_attempt_id,
                expected_execution_components,
            )
        )

    runtime_identity = (bound_artifacts.get("runtime_identity") or {}).get("payload")
    computed_runtime = sha256_json(runtime_identity) if runtime_identity else None
    status_runtime = status.get("runtime_fingerprint_sha256")
    rendered_runtime = rendered.get("runtime_fingerprint_sha256")
    if quality == "legacy_incomplete":
        legacy_runtime_path = result_dir / "runtime_identity.json"
        if legacy_runtime_path.is_file():
            try:
                runtime_identity = _load_json(legacy_runtime_path)
                computed_runtime = sha256_json(runtime_identity)
            except (OSError, ValueError, json.JSONDecodeError):
                runtime_identity = None
        if runtime_identity is None and not status_runtime and not rendered_runtime:
            runtime_quality = "legacy_missing"
        else:
            runtime_quality = "legacy_unverified"
        if provenance is None:
            reasons.append("run_provenance_unresolved")
    elif (
        runtime_identity is not None
        and computed_runtime == status_runtime == rendered_runtime
    ):
        runtime_quality = "exact"
    else:
        runtime_quality = "incomplete_or_mismatched"
        reasons.append("runtime_identity_incomplete_or_mismatched")

    if isinstance(input_components, dict):
        if input_components.get("command_sha256") != sha256_json(
            rendered.get("command") or []
        ):
            reasons.append("rendered_command_not_bound_to_execution_inputs")
        # ``environment_sha256`` deliberately binds the full effective worker
        # environment without persisting secrets.  ``rendered.environment`` is
        # only the non-sensitive job overlay, so it cannot be used to recompute
        # that component.  Input -> final component equality plus the
        # allowlisted runtime-mechanism manifest is the verifiable closure.
        mechanism = (bound_artifacts.get("runtime_mechanism") or {}).get("payload")
        if isinstance(mechanism, dict) and input_components.get(
            "runtime_mechanism_fingerprint_sha256"
        ) != mechanism.get("fingerprint_sha256"):
            reasons.append("runtime_mechanism_component_fingerprint_mismatch")
        if (
            status.get("provenance_sha256")
            != input_components.get("provenance_sha256")
            or rendered.get("provenance_sha256")
            != input_components.get("provenance_sha256")
        ):
            reasons.append("provenance_status_render_binding_mismatch")

    authorization = (
        execution_inputs.get("authorization")
        if isinstance(execution_inputs, dict)
        else None
    )
    approval_design_sha256 = (
        (authorization.get("evidence") or {}).get("approval_design_sha256")
        if isinstance(authorization, dict)
        else None
    )
    if quality != "legacy_incomplete" and (
        status.get("approval_design_sha256") != approval_design_sha256
        or rendered.get("approval_design_sha256") != approval_design_sha256
        or exact_manifest.get("approval_design_sha256") != approval_design_sha256
    ):
        reasons.append("approval_design_binding_mismatch")
    if quality != "legacy_incomplete" and (
        status.get("execution_inputs_sha256") != execution_inputs_sha256
        or rendered.get("execution_inputs_sha256") != execution_inputs_sha256
    ):
        reasons.append("execution_inputs_status_render_binding_mismatch")

    if quality == "complete" and reasons:
        quality = "incomplete"
    hardware_manifest = (bound_artifacts.get("runtime_hardware") or {}).get("payload")
    authorization_eligible = bool(
        quality == "complete"
        and isinstance(authorization, dict)
        and authorization.get("mode") == "approved"
        and authorization.get("calibration_eligible") is True
    )
    hardware_eligible = bool(
        quality == "complete"
        and isinstance(hardware_manifest, dict)
        and hardware_manifest.get("calibration_hardware_eligible") is True
    )
    evidence_verified = quality == "complete"
    calibration_evidence_eligible = bool(
        evidence_verified
        and authorization_eligible
        and hardware_eligible
        and isinstance(exact_manifest, dict)
        and exact_manifest.get("calibration_eligible") is True
    )
    if calibration_evidence_eligible:
        calibration_policy = "eligible_after_bound_partition_and_holdout"
    elif quality == "legacy_incomplete":
        calibration_policy = "diagnostic_only_legacy_never_calibration"
    else:
        calibration_policy = "ineligible_until_evidence_is_repaired"

    verification_material = {
        "execution_manifest": exact_manifest,
        "execution_inputs_manifest": execution_inputs,
        "bound_artifacts": bound_artifacts,
        "runtime_model_inventory": runtime_model_inventory,
        "runtime_rank_evidence": runtime_rank_evidence,
    }
    topology_path = project_root / "artifacts" / "nvidia_topology.txt"
    model_inventory = _snapshot_json(project_root / "artifacts" / "model_inventory.json")
    dataset_analysis = _snapshot_json(project_root / "artifacts" / "dataset_analysis.json")
    retrospective_components = {
        "job": rendered.get("job") or {},
        "runtime_config": runtime_config,
        "hardware": hardware,
        "training_scope": experiment.get("training_scope") or {},
        "topology_sha256": sha256_file(topology_path) if topology_path.is_file() else None,
        "model_inventory_sha256": model_inventory.get("sha256") if model_inventory else None,
        "dataset_analysis_sha256": dataset_analysis.get("sha256") if dataset_analysis else None,
    }
    return {
        "quality": quality,
        "quality_reasons": sorted(set(reasons)),
        "calibration_policy": calibration_policy,
        "evidence_verified": evidence_verified,
        "calibration_evidence_eligible": calibration_evidence_eligible,
        "authorization_eligible": authorization_eligible,
        "hardware_eligible": hardware_eligible,
        "runtime_identity_quality": runtime_quality,
        "status_runtime_fingerprint_sha256": status_runtime,
        "rendered_runtime_fingerprint_sha256": rendered_runtime,
        "computed_runtime_identity_sha256": computed_runtime,
        "runtime_identity": runtime_identity,
        "status_execution_fingerprint_sha256": status_exact,
        "rendered_execution_fingerprint_sha256": rendered_exact,
        "computed_execution_manifest_sha256": computed_exact,
        "execution_manifest_path": (
            str(exact_manifest_path) if exact_manifest_path is not None else None
        ),
        "execution_manifest": exact_manifest,
        "execution_inputs_manifest": execution_inputs,
        "bound_job": bound_job,
        "authorization": authorization,
        "bound_artifacts": bound_artifacts,
        "runtime_model_inventory": runtime_model_inventory,
        "runtime_model_inventory_sha256": runtime_model_inventory_sha256,
        "runtime_rank_evidence": runtime_rank_evidence,
        "runtime_mechanism_fingerprint_sha256": (
            (exact_manifest.get("components") or {}).get(
                "runtime_mechanism_fingerprint_sha256"
            )
            if isinstance(exact_manifest, dict)
            else None
        ),
        "evidence_verification_sha256": sha256_json(verification_material),
        "retrospective_fingerprint": {
            "sha256": sha256_json(retrospective_components),
            "is_run_bound": False,
            "must_not_upgrade_confidence": True,
            "components": retrospective_components,
        },
        "runtime_config": {
            **(bound_artifacts.get("runtime_config") or {
                "path": str(runtime_config_path) if runtime_config_path else None,
                "file_sha256": (
                    sha256_file(runtime_config_path)
                    if runtime_config_path and runtime_config_path.is_file()
                    else None
                ),
                "payload": runtime_config,
            }),
            "hash_was_bound_at_run": evidence_verified,
        },
        "model_inventory": model_inventory,
        "dataset_analysis": dataset_analysis,
    }


def _terminal_outcome_quality(
    status: dict[str, Any],
    *,
    job_id: str,
    execution_attempt_id: str | None,
    expected_ranks: int,
) -> tuple[bool, list[str]]:
    """Require structured success/OOM truth rather than a broad log substring."""

    evidence = status.get("classification_evidence")
    if not isinstance(evidence, dict):
        return False, ["structured_terminal_classification_evidence_missing"]
    reasons = []
    if evidence.get("schema") != "sft_terminal_classification/v1":
        reasons.append("terminal_classification_schema_invalid")
    if evidence.get("job_id") != job_id:
        reasons.append("terminal_classification_job_binding_mismatch")
    if evidence.get("execution_attempt_id") != execution_attempt_id:
        reasons.append("terminal_classification_attempt_binding_mismatch")
    if evidence.get("classification") != status.get("classification"):
        reasons.append("terminal_classification_status_mismatch")
    if evidence.get("return_code") != status.get("return_code"):
        reasons.append("terminal_classification_return_code_mismatch")
    classification = status.get("classification")
    if classification == "success":
        if status.get("return_code") != 0:
            reasons.append("success_has_nonzero_return_code")
        if evidence.get("summaries_complete") is not True:
            reasons.append("success_rank_summaries_incomplete")
        summaries = evidence.get("summary_files")
        if not isinstance(summaries, list) or {
            _integer(item.get("rank"))
            for item in summaries
            if isinstance(item, dict)
        } != set(range(expected_ranks)):
            reasons.append("success_rank_summary_set_invalid")
    elif classification == "oom":
        if evidence.get("cuda_oom_confirmed") is not True:
            reasons.append("oom_is_not_structurally_confirmed_cuda_oom")
    else:
        reasons.append("failure_outcome_is_not_a_calibration_label")
    return not reasons, reasons


def _one_observation(
    result_dir: Path,
    project_root: Path,
    hardware: dict[str, Any],
    experiment: dict[str, Any],
    provenance_index: dict[str, tuple[Path, dict[str, Any]]],
    *,
    job_id_hint: str,
    attempt_scoped: bool,
) -> dict[str, Any] | None:
    status_path = result_dir / "status.json"
    rendered_path = result_dir / "rendered_run.json"
    if not status_path.is_file() or not rendered_path.is_file():
        return None
    status = _load_json(status_path)
    classification = _normalized_terminal_outcome(status.get("classification"))
    if classification is None:
        return None
    started = _finite_float(status.get("started_unix"))
    finished = _finite_float(status.get("finished_unix"))
    if started is None or finished is None or finished < started:
        return None
    rendered = _load_json(rendered_path)
    job = rendered.get("job")
    if not isinstance(job, dict):
        return None
    _validate_h800_job(job)
    if status.get("job_id") != job.get("job_id") or job_id_hint != job.get("job_id"):
        return None

    execution_attempt_id = status.get("execution_attempt_id")
    if attempt_scoped and (
        result_dir.name != execution_attempt_id
        or rendered.get("execution_attempt_id") != execution_attempt_id
    ):
        return None

    events, event_filter = _read_events(
        result_dir,
        started,
        finished,
        expected_job_id=str(job["job_id"]),
        expected_execution_attempt_id=(
            str(execution_attempt_id) if execution_attempt_id else None
        ),
        require_attempt_binding=attempt_scoped,
    )
    aggregate = _aggregate_events(events)
    errors = aggregate.pop("failure_errors")
    requested = _largest_parsed(errors, OOM_ALLOCATION)
    capacity = _largest_parsed(errors, OOM_CAPACITY)
    free = _largest_parsed(errors, OOM_FREE)
    expected_ranks = _integer(job.get("gpu_count")) or 0
    ranks_complete = len(aggregate["ranks_with_events"]) == expected_ranks
    measured_rank_count = len(aggregate["measured_steps_by_rank"])
    measured_ranks_complete = bool(
        measured_rank_count == expected_ranks
        and all(
            steps > 0 for steps in aggregate["measured_steps_by_rank"].values()
        )
    )
    raw_throughput_measurement_available = bool(
        classification == "success"
        and aggregate["measured_seconds"]
        and aggregate["measured_seconds"] > 0
        and measured_ranks_complete
    )

    runtime_config_path, runtime_config = _load_yaml_snapshot(rendered.get("config_path"))
    expected_provenance = status.get("provenance_sha256")
    rendered_provenance = rendered.get("provenance_sha256")
    resolved = provenance_index.get(str(expected_provenance)) if expected_provenance else None
    provenance_path, provenance_payload = resolved if resolved else (None, None)
    fingerprint = _fingerprint(
        result_dir,
        status,
        rendered,
        runtime_config_path,
        runtime_config,
        hardware,
        experiment,
        provenance_payload,
        project_root,
    )

    bound_job = fingerprint.get("bound_job")
    if not isinstance(bound_job, dict):
        bound_job = job
    _validate_h800_job(bound_job)
    expected_ranks = _integer(bound_job.get("gpu_count")) or 0
    ranks_complete = set(aggregate["ranks_with_events"]) == set(
        range(expected_ranks)
    )
    measured_rank_count = len(aggregate["measured_steps_by_rank"])
    measured_ranks_complete = bool(
        set(int(rank) for rank in aggregate["measured_steps_by_rank"])
        == set(range(expected_ranks))
        and all(
            steps > 0 for steps in aggregate["measured_steps_by_rank"].values()
        )
    )
    raw_throughput_measurement_available = bool(
        classification == "success"
        and aggregate["measured_seconds"]
        and aggregate["measured_seconds"] > 0
        and measured_ranks_complete
    )
    partition, partition_reasons = _calibration_partition(bound_job)
    terminal_label_verified, terminal_reasons = _terminal_outcome_quality(
        status,
        job_id=str(bound_job.get("job_id") or ""),
        execution_attempt_id=(
            str(execution_attempt_id) if execution_attempt_id else None
        ),
        expected_ranks=expected_ranks,
    )
    event_binding_verified = bool(
        attempt_scoped
        and event_filter["attempt_binding_complete"]
        and event_filter["malformed_lines"] == 0
        and event_filter["events_missing_time"] == 0
    )
    calibration_base_eligible = bool(
        fingerprint["calibration_evidence_eligible"]
        and partition is not None
        and terminal_label_verified
        and event_binding_verified
        and ranks_complete
    )
    feasibility_usable = bool(
        calibration_base_eligible and classification in {"success", "oom"}
    )
    throughput_usable = bool(
        feasibility_usable
        and classification == "success"
        and raw_throughput_measurement_available
    )

    runtime_hardware = (
        (fingerprint.get("bound_artifacts") or {}).get("runtime_hardware") or {}
    ).get("payload")
    if isinstance(runtime_hardware, dict):
        devices = runtime_hardware.get("devices") or []
        live_identity = " ".join(
            str(value)
            for device in devices
            if isinstance(device, dict)
            for value in device.values()
        ).lower()
        if "4090" in live_identity or any(
            not isinstance(device, dict) or device.get("name") != "NVIDIA H800"
            for device in devices
        ):
            raise UnsupportedHardwareError(
                f"Attempt {execution_attempt_id} is not exact H800 hardware"
            )

    status_sha = sha256_file(status_path)
    rendered_sha = sha256_file(rendered_path)
    legacy_attempt_sha256 = sha256_json(
        {
            "job_id": bound_job["job_id"],
            "started_unix": started,
            "finished_unix": finished,
            "status_sha256": status_sha,
        }
    )
    observation_identity = {
        "schema": SCHEMA,
        "job_id": bound_job["job_id"],
        "execution_attempt_id": (
            str(execution_attempt_id) if execution_attempt_id else None
        ),
        "legacy_attempt_sha256": legacy_attempt_sha256,
        "status_sha256": status_sha,
        "rendered_run_sha256": rendered_sha,
        "execution_manifest_sha256": fingerprint[
            "computed_execution_manifest_sha256"
        ],
    }
    observation_id = sha256_json(observation_identity)
    gpu_mask = [
        int(value)
        for value in str(status.get("gpu_mask") or rendered.get("gpu_mask") or "").split(",")
        if value.strip()
    ]
    return {
        "schema": SCHEMA,
        "observation_id": observation_id,
        "observation_identity": observation_identity,
        "attempt": {
            "execution_attempt_id": (
                str(execution_attempt_id) if execution_attempt_id else None
            ),
            "legacy_attempt_sha256": legacy_attempt_sha256,
            "job_id": bound_job["job_id"],
            "started_unix": started,
            "finished_unix": finished,
            "wall_seconds": _finite_float(status.get("wall_seconds")),
            "attempt_scoped": attempt_scoped,
            "event_binding_policy": event_filter["policy"],
        },
        "hardware": {
            "gpu_family": (
                "H800"
                if isinstance(runtime_hardware, dict)
                and runtime_hardware.get("calibration_hardware_eligible") is True
                else "declared_H800_unverified"
            ),
            "gpu_type": (
                ((runtime_hardware or {}).get("declared_hardware") or {}).get(
                    "campaign_gpu_type"
                )
                if isinstance(runtime_hardware, dict)
                else (experiment.get("training_scope") or {}).get("gpu_type")
            ),
            "gpu_id": (
                ((runtime_hardware or {}).get("declared_hardware") or {}).get(
                    "gpu_id"
                )
                if isinstance(runtime_hardware, dict)
                else hardware.get("gpu_id")
            ),
            "gpu_mask": gpu_mask,
            "num_gpus": expected_ranks,
            "profile": hardware,
            "runtime_attestation": runtime_hardware,
            "identity_was_run_bound": fingerprint["hardware_eligible"],
        },
        "configuration": {
            "job": bound_job,
            "job_was_authorized_and_bound": fingerprint["authorization_eligible"],
            "calibration_partition": partition,
            "environment": rendered.get("environment") or {},
        },
        "outcome": {
            "class": classification,
            "return_code": _integer(status.get("return_code")),
            "is_terminal_completed_attempt": True,
            "is_memory_right_censored": classification == "oom",
            "peak_memory_is_imputed": False,
            "raw_feasibility_measurement_available": classification
            in {"success", "oom"},
            "raw_throughput_measurement_available": raw_throughput_measurement_available,
            "terminal_label_verified": terminal_label_verified,
            "calibration_base_eligible": calibration_base_eligible,
            "usable_for_feasibility_calibration": feasibility_usable,
            "usable_for_throughput_calibration": throughput_usable,
            "calibration_exclusion_reasons": sorted(
                set(
                    partition_reasons
                    + terminal_reasons
                    + ([] if event_binding_verified else ["event_attempt_binding_incomplete"])
                    + (
                        []
                        if fingerprint["calibration_evidence_eligible"]
                        else ["execution_evidence_not_calibration_eligible"]
                    )
                    + ([] if ranks_complete else ["rank_events_incomplete"])
                )
            ),
            "failure_signatures_sha256": sorted(set(sha256_json(error) for error in errors)),
        },
        "measurements": aggregate,
        "censoring": (
            {
                "kind": "right_censored_memory_demand",
                "constraint": "required_memory_exceeded_available_capacity_at_failure",
                "requested_allocation_bytes": requested,
                "device_capacity_bytes_reported_in_error": capacity,
                "free_bytes_reported_in_error": free,
                "demand_peak_bytes": None,
                "demand_peak_is_unknown_not_imputed": True,
                "raw_failure_errors": errors,
            }
            if classification == "oom"
            else None
        ),
        "provenance": {
            "status_provenance_sha256": expected_provenance,
            "rendered_provenance_sha256": rendered_provenance,
            "references_match": bool(
                expected_provenance and expected_provenance == rendered_provenance
            ),
            "resolved": provenance_payload is not None,
            "source_path": str(provenance_path) if provenance_path else None,
            "resolved_file_sha256": sha256_file(provenance_path) if provenance_path else None,
            "payload": provenance_payload,
            "approval_design_sha256": status.get("approval_design_sha256"),
            "attempt_snapshot": (
                (fingerprint.get("bound_artifacts") or {}).get("provenance")
            ),
        },
        "fingerprint": fingerprint,
        "source": {
            "result_dir": _relative(result_dir, project_root),
            "status": {
                "path": _relative(status_path, project_root),
                "sha256": status_sha,
                "payload_sha256": sha256_json(status),
                "payload": status,
            },
            "rendered_run": {
                "path": _relative(rendered_path, project_root),
                "sha256": rendered_sha,
                "payload_sha256": sha256_json(rendered),
                "payload": rendered,
            },
            "event_filter": event_filter,
        },
        "quality": {
            "expected_rank_count": expected_ranks,
            "rank_events_complete": ranks_complete,
            "measured_rank_count": measured_rank_count,
            "measured_ranks_complete": measured_ranks_complete,
            "malformed_event_lines": event_filter["malformed_lines"],
            "fingerprint_quality": fingerprint["quality"],
            "evidence_verified": fingerprint["evidence_verified"],
            "event_attempt_binding_complete": event_binding_verified,
            "terminal_label_verified": terminal_label_verified,
            "calibration_partition_bound": partition is not None,
            "requires_legacy_compatibility_review": fingerprint["quality"]
            == "legacy_incomplete",
        },
    }


def validate_canonical_observation(row: dict[str, Any]) -> list[str]:
    """Recompute the security-relevant invariants embedded in one JSONL row."""

    reasons: list[str] = []
    if not isinstance(row, dict) or row.get("schema") != SCHEMA:
        return ["observation_schema_invalid"]
    identity = row.get("observation_identity")
    if not isinstance(identity, dict) or row.get("observation_id") != sha256_json(
        identity
    ):
        reasons.append("observation_identity_hash_mismatch")
    source = row.get("source")
    source = source if isinstance(source, dict) else {}
    status_record = source.get("status")
    render_record = source.get("rendered_run")
    status_record = status_record if isinstance(status_record, dict) else {}
    render_record = render_record if isinstance(render_record, dict) else {}
    status = status_record.get("payload")
    rendered = render_record.get("payload")
    status = status if isinstance(status, dict) else {}
    rendered = rendered if isinstance(rendered, dict) else {}
    if status_record.get("payload_sha256") != sha256_json(status):
        reasons.append("embedded_status_payload_hash_mismatch")
    if render_record.get("payload_sha256") != sha256_json(rendered):
        reasons.append("embedded_render_payload_hash_mismatch")
    attempt = row.get("attempt")
    attempt = attempt if isinstance(attempt, dict) else {}
    configuration = row.get("configuration")
    configuration = configuration if isinstance(configuration, dict) else {}
    job = configuration.get("job")
    job = job if isinstance(job, dict) else {}
    execution_attempt_id = attempt.get("execution_attempt_id")
    if (
        status.get("job_id") != job.get("job_id")
        or attempt.get("job_id") != job.get("job_id")
        or status.get("execution_attempt_id") != execution_attempt_id
    ):
        reasons.append("observation_job_attempt_binding_mismatch")
    if isinstance(identity, dict):
        expected_identity = {
            "schema": SCHEMA,
            "job_id": job.get("job_id"),
            "execution_attempt_id": execution_attempt_id,
            "legacy_attempt_sha256": attempt.get("legacy_attempt_sha256"),
            "status_sha256": status_record.get("sha256"),
            "rendered_run_sha256": render_record.get("sha256"),
            "execution_manifest_sha256": (row.get("fingerprint") or {}).get(
                "computed_execution_manifest_sha256"
            ),
        }
        if identity != expected_identity:
            reasons.append("observation_identity_material_mismatch")

    fingerprint = row.get("fingerprint")
    fingerprint = fingerprint if isinstance(fingerprint, dict) else {}
    quality = fingerprint.get("quality")
    outcome = row.get("outcome")
    outcome = outcome if isinstance(outcome, dict) else {}
    normalized_outcome = _normalized_terminal_outcome(status.get("classification"))
    if normalized_outcome is None or outcome.get("class") != normalized_outcome:
        reasons.append("embedded_terminal_outcome_normalization_mismatch")
    if quality != "complete":
        if fingerprint.get("evidence_verified") is True:
            reasons.append("incomplete_fingerprint_claims_verified_evidence")
        if outcome.get("usable_for_feasibility_calibration") is True or outcome.get(
            "usable_for_throughput_calibration"
        ) is True:
            reasons.append("incomplete_fingerprint_claims_calibration_usability")
        return sorted(set(reasons))

    if fingerprint.get("evidence_verified") is not True or fingerprint.get(
        "quality_reasons"
    ) not in ([], None):
        reasons.append("complete_fingerprint_verification_claim_invalid")
    execution_manifest = fingerprint.get("execution_manifest")
    execution_inputs = fingerprint.get("execution_inputs_manifest")
    artifacts = fingerprint.get("bound_artifacts")
    rank_evidence = fingerprint.get("runtime_rank_evidence")
    inventory = fingerprint.get("runtime_model_inventory")
    if not all(
        (
            isinstance(execution_manifest, dict),
            isinstance(execution_inputs, dict),
            isinstance(artifacts, dict),
            isinstance(rank_evidence, list),
            isinstance(inventory, dict),
        )
    ):
        return sorted(set(reasons + ["complete_fingerprint_embedded_evidence_missing"]))
    verification_material = {
        "execution_manifest": execution_manifest,
        "execution_inputs_manifest": execution_inputs,
        "bound_artifacts": artifacts,
        "runtime_model_inventory": inventory,
        "runtime_rank_evidence": rank_evidence,
    }
    if fingerprint.get("evidence_verification_sha256") != sha256_json(
        verification_material
    ):
        reasons.append("evidence_verification_digest_mismatch")
    execution_sha256 = sha256_json(execution_manifest)
    if (
        execution_sha256
        != fingerprint.get("computed_execution_manifest_sha256")
        or execution_sha256 != status.get("execution_fingerprint_sha256")
        or execution_sha256 != rendered.get("execution_fingerprint_sha256")
    ):
        reasons.append("execution_manifest_embedded_hash_mismatch")
    if execution_manifest.get("schema") != EXECUTION_FINGERPRINT_SCHEMA:
        reasons.append("execution_manifest_schema_invalid")
    if (
        execution_inputs.get("schema") != EXECUTION_INPUTS_SCHEMA
        or execution_inputs.get("job_id") != job.get("job_id")
        or execution_inputs.get("execution_attempt_id") != execution_attempt_id
        or execution_manifest.get("job_id") != job.get("job_id")
        or execution_manifest.get("execution_attempt_id") != execution_attempt_id
    ):
        reasons.append("execution_inputs_final_job_attempt_binding_mismatch")
    if execution_inputs.get("job_snapshot") != job or execution_manifest.get(
        "job_snapshot"
    ) != job:
        reasons.append("authorized_job_snapshot_mismatch")
    if execution_inputs.get("job_payload_sha256") != sha256_json(job):
        reasons.append("authorized_job_payload_hash_mismatch")

    authorization = execution_inputs.get("authorization")
    if not isinstance(authorization, dict) or execution_manifest.get(
        "authorization"
    ) != authorization:
        reasons.append("embedded_authorization_mismatch")
    else:
        evidence = authorization.get("evidence")
        if (
            authorization.get("schema") != EXECUTION_AUTHORIZATION_SCHEMA
            or not isinstance(evidence, dict)
            or authorization.get("evidence_sha256") != sha256_json(evidence)
            or authorization.get("job_payload_sha256") != sha256_json(job)
        ):
            reasons.append("embedded_authorization_invalid")
        if authorization.get("calibration_eligible") is not (
            authorization.get("mode") == "approved"
        ):
            reasons.append("embedded_authorization_policy_invalid")

    terminal_evidence = status.get("classification_evidence")
    if (
        not isinstance(terminal_evidence, dict)
        or execution_manifest.get("outcome") != terminal_evidence
        or execution_manifest.get("outcome_sha256")
        != sha256_json(terminal_evidence)
    ):
        reasons.append("embedded_terminal_evidence_mismatch")
    expected_final_eligibility = bool(
        isinstance(authorization, dict)
        and authorization.get("calibration_eligible") is True
        and _terminal_evidence_is_calibratable(terminal_evidence)
    )
    if (
        execution_manifest.get("calibration_eligible")
        is not expected_final_eligibility
    ):
        reasons.append("embedded_final_calibration_policy_invalid")

    input_components = execution_inputs.get("components")
    final_components = execution_manifest.get("components")
    if not isinstance(input_components, dict) or set(input_components) != set(
        STATIC_EXECUTION_COMPONENTS
    ):
        reasons.append("embedded_execution_input_components_invalid")
        input_components = {}
    if not isinstance(final_components, dict) or set(final_components) != set(
        REQUIRED_EXECUTION_COMPONENTS
    ):
        reasons.append("embedded_final_components_invalid")
        final_components = {}
    for name in STATIC_EXECUTION_COMPONENTS:
        if input_components.get(name) != final_components.get(name):
            reasons.append(f"embedded_static_component_{name}_mismatch")

    for name, artifact in artifacts.items():
        if not isinstance(artifact, dict) or artifact.get(
            "payload_sha256"
        ) != sha256_json(artifact.get("payload")):
            reasons.append(f"embedded_artifact_{name}_payload_hash_mismatch")
    mechanism = (artifacts.get("runtime_mechanism") or {}).get("payload")
    if isinstance(mechanism, dict):
        mechanism_material = dict(mechanism)
        claimed_mechanism = mechanism_material.pop("fingerprint_sha256", None)
        if claimed_mechanism != sha256_json(mechanism_material) or claimed_mechanism != (
            final_components.get("runtime_mechanism_fingerprint_sha256")
        ):
            reasons.append("embedded_runtime_mechanism_fingerprint_mismatch")
    else:
        reasons.append("embedded_runtime_mechanism_missing")
    runtime_hardware = (artifacts.get("runtime_hardware") or {}).get("payload")
    devices = (
        runtime_hardware.get("devices")
        if isinstance(runtime_hardware, dict)
        else None
    )
    if (
        not isinstance(runtime_hardware, dict)
        or runtime_hardware.get("schema") != RUNTIME_HARDWARE_SCHEMA
        or runtime_hardware.get("job_id") != job.get("job_id")
        or runtime_hardware.get("execution_attempt_id") != execution_attempt_id
        or runtime_hardware.get("calibration_hardware_eligible") is not True
        or not isinstance(devices, list)
        or len(devices) != _integer(job.get("gpu_count"))
        or any(
            not isinstance(device, dict) or device.get("name") != "NVIDIA H800"
            for device in (devices or [])
        )
    ):
        reasons.append("embedded_runtime_hardware_attestation_invalid")
    expected_fingerprint_eligibility = bool(
        expected_final_eligibility
        and isinstance(runtime_hardware, dict)
        and runtime_hardware.get("calibration_hardware_eligible") is True
    )
    if (
        fingerprint.get("calibration_evidence_eligible")
        is not expected_fingerprint_eligibility
    ):
        reasons.append("embedded_fingerprint_calibration_policy_invalid")

    training_mode = str(job.get("train_type") or "")
    inventory_sha256 = sha256_json(inventory)
    if inventory_sha256 != final_components.get("runtime_model_inventory_sha256"):
        reasons.append("embedded_runtime_inventory_component_mismatch")
    references = []
    for item in rank_evidence:
        if not isinstance(item, dict):
            reasons.append("embedded_rank_evidence_invalid")
            continue
        header = item.get("manifest_without_inventory")
        reference = item.get("reference")
        if not isinstance(header, dict) or not isinstance(reference, dict):
            reasons.append("embedded_rank_evidence_invalid")
            continue
        rank_manifest = {**header, "inventory": inventory}
        if item.get("manifest_payload_sha256") != sha256_json(rank_manifest):
            reasons.append("embedded_rank_manifest_payload_hash_mismatch")
        try:
            validate_runtime_model_manifest(
                rank_manifest,
                expected_job_id=str(job.get("job_id") or ""),
                expected_execution_attempt_id=str(execution_attempt_id or ""),
                expected_rank=_integer(reference.get("rank")),
                expected_world_size=_integer(job.get("gpu_count")),
                expected_training_mode=training_mode,
                allow_unavailable_device=False,
            )
        except (RuntimeEvidenceError, TypeError) as error:
            reasons.append(f"embedded_runtime_rank_manifest_invalid:{error}")
        references.append(reference)
    if {
        _integer(reference.get("rank"))
        for reference in references
        if isinstance(reference, dict)
    } != set(range(_integer(job.get("gpu_count")) or 0)):
        reasons.append("embedded_runtime_rank_set_incomplete")
    if sha256_json(references) != final_components.get(
        "runtime_model_manifest_set_sha256"
    ):
        reasons.append("embedded_runtime_rank_reference_set_mismatch")

    partition = configuration.get("calibration_partition")
    partition_value, partition_reasons = _calibration_partition(job)
    if partition != partition_value:
        reasons.append("embedded_calibration_partition_mismatch")
    quality_record = row.get("quality")
    quality_record = quality_record if isinstance(quality_record, dict) else {}
    if outcome.get("usable_for_feasibility_calibration") is True:
        if (
            fingerprint.get("calibration_evidence_eligible") is not True
            or partition_reasons
            or outcome.get("terminal_label_verified") is not True
            or quality_record.get("event_attempt_binding_complete") is not True
        ):
            reasons.append("feasibility_calibration_usability_claim_invalid")
    if outcome.get("usable_for_throughput_calibration") is True and (
        outcome.get("class") != "success"
        or outcome.get("usable_for_feasibility_calibration") is not True
        or quality_record.get("measured_ranks_complete") is not True
    ):
        reasons.append("throughput_calibration_usability_claim_invalid")
    return sorted(set(reasons))


def export_observations(
    project_root: Path,
    results_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Read terminal success/OOM H800 attempts without mutating source state."""

    project_root = project_root.resolve()
    results_dir = (results_dir or project_root / "results").resolve()
    hardware = _load_json(project_root / "config" / "hardware.json")
    experiment = _load_json(project_root / "config" / "experiment.json")
    _validate_h800_source(project_root, results_dir, hardware, experiment)
    provenance = _provenance_index(project_root)
    rows = []
    if not results_dir.is_dir():
        return rows
    for job_root in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        attempts_root = job_root / "attempts"
        attempt_roots = (
            sorted(
                path
                for path in attempts_root.iterdir()
                if path.is_dir() and re.fullmatch(r"[0-9a-f]{20}", path.name)
            )
            if attempts_root.is_dir()
            else []
        )
        for result_dir in attempt_roots:
            row = _one_observation(
                result_dir,
                project_root,
                hardware,
                experiment,
                provenance,
                job_id_hint=job_root.name,
                attempt_scoped=True,
            )
            if row is not None:
                rows.append(row)
        # A top-level status/render pair is the historical flat format. New
        # launchers expose symlinked latest views; skip those to avoid exporting
        # the newest attempt twice.
        status_path = job_root / "status.json"
        rendered_path = job_root / "rendered_run.json"
        if (
            status_path.is_file()
            and rendered_path.is_file()
            and not status_path.is_symlink()
            and not rendered_path.is_symlink()
        ):
            row = _one_observation(
                job_root,
                project_root,
                hardware,
                experiment,
                provenance,
                job_id_hint=job_root.name,
                attempt_scoped=False,
            )
            if row is not None:
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Atomically replace only the requested export artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            count += 1
    temporary.replace(path)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "canonical_h800_observations.jsonl",
    )
    args = parser.parse_args()
    rows = export_observations(args.project_root, args.results_dir)
    count = write_jsonl(args.output, rows)
    print(f"Exported {count} terminal H800 observations to {args.output}")


if __name__ == "__main__":
    main()
