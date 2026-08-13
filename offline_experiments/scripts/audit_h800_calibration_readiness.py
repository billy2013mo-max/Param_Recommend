#!/usr/bin/env python3
"""Audit whether canonical H800 observations are ready for calibration.

This command is intentionally read-only with respect to observations and never
fits or publishes coefficients.  It fails closed on any 4090 identity and
requires run-bound complete execution fingerprints before counting a row as a
calibration candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from common import ROOT, read_json, sha256_file, sha256_json
from export_h800_observations import (
    SCHEMA as OBSERVATION_SCHEMA,
    validate_canonical_observation,
)
from historical_h800_readiness import analyze as analyze_historical_recovery
from recover_h800_historical_evidence import validate_recovery_report


SCHEMA = "sft_h800_calibration_readiness/v6"
ALLOWED_OUTCOMES = {
    "success",
    "oom",
    "software_failure",
    "infrastructure_failure",
}
CALIBRATION_ROLES = {"calibration", "holdout"}
MIN_CALIBRATION_FEASIBILITY = 6
MIN_CALIBRATION_THROUGHPUT = 4
MIN_HOLDOUT_FEASIBILITY = 4
MIN_HOLDOUT_THROUGHPUT = 2
SUPPORTED_MBS = {1, 2, 4, 8, 16}
MIN_PACKING_CALIBRATION_PAIRS = 2
MIN_PACKING_HOLDOUT_PAIRS = 2
PACKING_EVIDENCE_CLASS = "packing_paired_only"


class NonH800ObservationError(ValueError):
    """Raised rather than allowing cross-GPU calibration contamination."""


def _read_rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on observation line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"Observation line {line_number} must contain a JSON object"
                )
            if row.get("schema") != OBSERVATION_SCHEMA:
                raise ValueError(
                    f"Observation line {line_number} has unsupported schema "
                    f"{row.get('schema')!r}"
                )
            yield line_number, row


def _hardware_identity(row: dict[str, Any]) -> str:
    hardware = row.get("hardware")
    if not isinstance(hardware, dict):
        return ""
    profile = hardware.get("profile")
    profile = profile if isinstance(profile, dict) else {}
    parts = (
        hardware.get("gpu_family"),
        hardware.get("gpu_type"),
        hardware.get("gpu_id"),
        profile.get("gpu_id"),
        profile.get("name_reported_by_driver"),
    )
    return " ".join(str(part) for part in parts if part is not None).lower()


def _reject_non_h800(row: dict[str, Any], line_number: int) -> None:
    hardware = row.get("hardware")
    family = hardware.get("gpu_family") if isinstance(hardware, dict) else None
    identity = _hardware_identity(row)
    configuration = row.get("configuration")
    job = configuration.get("job") if isinstance(configuration, dict) else None
    job_hardware_identity = " ".join(
        str((job or {}).get(key) or "")
        for key in ("gpu_type", "hardware_id", "campaign_id", "phase_id")
    ).lower()
    if (
        family not in {"H800", "declared_H800_unverified"}
        or "h800" not in identity
        or "4090" in identity
        or "4090" in job_hardware_identity
    ):
        raise NonH800ObservationError(
            f"Observation line {line_number} is not exclusively H800: "
            f"family={family!r}, identity={identity!r}, "
            f"job_hardware={job_hardware_identity!r}"
        )
    fingerprint = row.get("fingerprint")
    if (
        isinstance(fingerprint, dict)
        and fingerprint.get("quality") == "complete"
        and family != "H800"
    ):
        raise NonH800ObservationError(
            f"Observation line {line_number} claims complete evidence without "
            "an exact run-bound H800 attestation"
        )


def _dtype(runtime_config: dict[str, Any]) -> str:
    if runtime_config.get("bf16") is True:
        return "bf16"
    if runtime_config.get("fp16") is True:
        return "fp16"
    return "unknown"


def _kernel_path(
    runtime_config: dict[str, Any], environment: dict[str, Any]
) -> str:
    attention = runtime_config.get("flash_attn")
    if isinstance(attention, str) and attention.lower() in {"fa2", "fa3"}:
        variant = str(environment.get("FA3_VARIANT") or "default").lower()
        attention_path = (
            f"{attention.lower()}:{variant}"
            if attention.lower() == "fa3"
            else attention.lower()
        )
    elif attention in (False, None, "disabled", "none"):
        attention_path = "standard_attention"
    else:
        attention_path = "unknown_attention"
    if str(environment.get("ENABLE_CCE") or "0") == "1":
        cross_entropy = "chunked_ce"
    elif runtime_config.get("enable_liger_kernel") is True:
        cross_entropy = "liger_fused_ce"
    else:
        cross_entropy = "full_ce"
    compile_path = f"compile={bool(runtime_config.get('torch_compile'))}"
    optimizer = f"optim={runtime_config.get('optim') or 'unknown'}"
    reentrant = f"gc_reentrant={runtime_config.get('use_reentrant_gc')!r}"
    return "+".join(
        (attention_path, cross_entropy, compile_path, optimizer, reentrant)
    )


def _zero_stage(value: Any) -> int:
    normalized = str(value or "").lower().replace("-", "")
    if normalized in {"none", "0", "zero0"}:
        return 0
    if normalized.startswith("zero") and normalized[4:].isdigit():
        return int(normalized[4:])
    return -1


def _selector(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    configuration = row.get("configuration") or {}
    job = configuration.get("job") or {}
    environment = configuration.get("environment") or {}
    fingerprint = row.get("fingerprint") or {}
    runtime = fingerprint.get("runtime_config") or {}
    runtime_config = runtime.get("payload") or {}
    runtime_fingerprint = fingerprint.get(
        "runtime_mechanism_fingerprint_sha256"
    )
    gc = runtime_config.get("gradient_checkpointing")
    if not isinstance(gc, bool):
        gc = job.get("gradient_checkpointing")
    selector = {
        "runtime_fingerprint": runtime_fingerprint,
        "dtype": _dtype(runtime_config),
        "kernel_path": _kernel_path(runtime_config, environment),
        "training_mode": str(job.get("train_type") or "unknown").lower(),
        "zero_stage": _zero_stage(job.get("zero")),
        "gradient_checkpointing": gc if isinstance(gc, bool) else None,
        "packing": job.get("packing") if isinstance(job.get("packing"), bool) else False,
    }
    key = "|".join(
        (
            str(selector["runtime_fingerprint"]),
            selector["dtype"],
            selector["kernel_path"],
            selector["training_mode"],
            f"zero{selector['zero_stage']}",
            f"gc={selector['gradient_checkpointing']}",
            f"packing={selector['packing']}",
        )
    )
    return key, selector


def _partition(row: dict[str, Any]) -> tuple[str, str | None, str | None]:
    configuration = row.get("configuration")
    partition = (
        configuration.get("calibration_partition")
        if isinstance(configuration, dict)
        else None
    )
    if not isinstance(partition, dict):
        return "unspecified", None, None
    role = str(partition.get("role") or "").lower()
    if role not in CALIBRATION_ROLES:
        role = "unspecified"
    unit = partition.get("split_unit_id")
    policy = partition.get("policy")
    return (
        role,
        str(unit) if isinstance(unit, str) and unit else None,
        str(policy) if isinstance(policy, str) and policy else None,
    )


def _required_calibration_variation(selector: dict[str, Any]) -> tuple[str, ...]:
    if selector.get("packing") is True:
        return ()
    return ("parameter_counts", "sequence_lengths", "micro_batch_sizes")


def _dependency_selector(selector: dict[str, Any]) -> dict[str, Any]:
    """Selector dimensions shared by packing and its unpacked safety basis."""

    return {
        key: selector.get(key)
        for key in (
            "runtime_fingerprint",
            "dtype",
            "kernel_path",
            "training_mode",
            "zero_stage",
            "gradient_checkpointing",
        )
    }


def _dependency_key(selector: dict[str, Any]) -> str:
    return json.dumps(
        _dependency_selector(selector),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _packing_effect_selector(selector: dict[str, Any]) -> dict[str, Any]:
    """Pool paired effects across stage while retaining stage as a covariate."""

    return {
        key: selector.get(key)
        for key in (
            "runtime_fingerprint",
            "dtype",
            "kernel_path",
            "training_mode",
            "gradient_checkpointing",
        )
    }


def _packing_effect_key(selector: dict[str, Any]) -> str:
    return json.dumps(
        _packing_effect_selector(selector),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _boundary_evidence(
    *, success: int, oom: int, mbs16_success: int, outside_domain: set[int]
) -> dict[str, Any]:
    complete = bool(
        success > 0
        and not outside_domain
        and (oom > 0 or mbs16_success > 0)
    )
    if not complete:
        completion_mode = None
    elif oom > 0:
        completion_mode = "success_oom_bracket"
    else:
        completion_mode = "supported_domain_fully_feasible_at_mbs16"
    return {
        "success_rows": success,
        "oom_rows": oom,
        "mbs16_success_rows": mbs16_success,
        "outside_supported_mbs": sorted(outside_domain),
        "supported_mbs": sorted(SUPPORTED_MBS),
        "complete": complete,
        "completion_mode": completion_mode,
    }


def audit(
    path: Path,
    historical_recovery_path: Path | None = None,
    *,
    verify_historical_sources: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Canonical H800 observation file does not exist: {path}")

    historical_report: dict[str, Any] | None = None
    if historical_recovery_path is not None:
        if not historical_recovery_path.is_file():
            raise FileNotFoundError(
                "Historical H800 recovery report does not exist: "
                f"{historical_recovery_path}"
            )
        historical_report = read_json(historical_recovery_path)
        if not isinstance(historical_report, dict):
            raise ValueError("Historical H800 recovery report must be a JSON object")
        historical_reasons = validate_recovery_report(
            historical_report,
            path,
            verify_source_files=verify_historical_sources,
        )
        if historical_reasons:
            raise ValueError(
                "Historical H800 recovery report failed validation: "
                f"{historical_reasons}"
            )

    outcomes: Counter[str] = Counter()
    fingerprint_qualities: Counter[str] = Counter()
    selectors: dict[str, dict[str, Any]] = {}
    selector_sets: dict[str, dict[str, dict[str, set[Any]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    selector_units: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    selector_policies: dict[str, set[str]] = defaultdict(set)
    packing_records: list[dict[str, Any]] = []
    global_split_units: dict[str, set[str]] = defaultdict(set)
    global_split_policies: set[str] = set()
    seen_observation_ids: set[str] = set()
    seen_execution_attempt_ids: set[str] = set()
    seen_execution_fingerprints: set[str] = set()
    observations_by_id: dict[str, dict[str, Any]] = {}
    total = 0
    complete = 0
    complete_authorized_h800 = 0
    complete_feasibility = 0
    complete_throughput = 0
    for line_number, row in _read_rows(path):
        _reject_non_h800(row, line_number)
        validation_reasons = validate_canonical_observation(row)
        if validation_reasons:
            raise ValueError(
                f"Observation line {line_number} failed embedded evidence "
                f"validation: {validation_reasons}"
            )
        observation_id = row.get("observation_id")
        if not isinstance(observation_id, str) or observation_id in seen_observation_ids:
            raise ValueError(
                f"Observation line {line_number} has a missing or duplicate observation_id"
            )
        seen_observation_ids.add(observation_id)
        observations_by_id[observation_id] = row
        attempt = row.get("attempt") or {}
        execution_attempt_id = attempt.get("execution_attempt_id")
        if execution_attempt_id is not None:
            if (
                not isinstance(execution_attempt_id, str)
                or execution_attempt_id in seen_execution_attempt_ids
            ):
                raise ValueError(
                    f"Observation line {line_number} has a duplicate execution attempt"
                )
            seen_execution_attempt_ids.add(execution_attempt_id)
        total += 1
        outcome = str((row.get("outcome") or {}).get("class") or "unknown")
        outcomes[outcome] += 1
        if outcome not in ALLOWED_OUTCOMES:
            raise ValueError(
                f"Observation line {line_number} has non-terminal outcome {outcome!r}"
            )
        fingerprint = row.get("fingerprint") or {}
        quality = str(fingerprint.get("quality") or "missing")
        fingerprint_qualities[quality] += 1
        if quality != "complete":
            continue
        execution_fingerprint = fingerprint.get(
            "computed_execution_manifest_sha256"
        )
        if (
            not isinstance(execution_fingerprint, str)
            or execution_fingerprint in seen_execution_fingerprints
        ):
            raise ValueError(
                f"Observation line {line_number} has a missing or duplicate "
                "execution fingerprint"
            )
        seen_execution_fingerprints.add(execution_fingerprint)
        complete += 1
        if fingerprint.get("calibration_evidence_eligible") is not True:
            continue
        complete_authorized_h800 += 1
        outcome_payload = row.get("outcome") or {}
        feasibility_usable = bool(
            outcome_payload.get("usable_for_feasibility_calibration")
        )
        throughput_usable = bool(
            outcome_payload.get("usable_for_throughput_calibration")
        )
        complete_feasibility += int(feasibility_usable)
        complete_throughput += int(throughput_usable)
        key, selector = _selector(row)
        job = ((row.get("configuration") or {}).get("job") or {})
        role, split_unit_id, split_policy = _partition(row)
        if role in CALIBRATION_ROLES and split_unit_id is not None:
            global_split_units[role].add(split_unit_id)
        if split_policy is not None:
            global_split_policies.add(split_policy)
        evidence_class = str(job.get("calibration_evidence_class") or "")
        if evidence_class == PACKING_EVIDENCE_CLASS or selector["packing"] is True:
            packing_records.append(
                {
                    "row": row,
                    "job": job,
                    "selector": selector,
                    "effect_key": _packing_effect_key(selector),
                    "effect_selector": _packing_effect_selector(selector),
                    "dependency_key": _dependency_key(selector),
                    "dependency_selector": _dependency_selector(selector),
                    "role": role,
                    "split_unit_id": split_unit_id,
                    "split_policy": split_policy,
                    "outcome": outcome,
                    "feasibility_usable": feasibility_usable,
                    "throughput_usable": throughput_usable,
                }
            )
            continue
        bucket = selectors.setdefault(
            key,
            {
                "selector": selector,
                "observations": 0,
                "success": 0,
                "oom": 0,
                "software_failure": 0,
                "infrastructure_failure": 0,
                "feasibility_usable": 0,
                "throughput_usable": 0,
                "roles": {"calibration": 0, "holdout": 0, "unspecified": 0},
                "role_usable": {
                    "calibration": {"feasibility": 0, "throughput": 0},
                    "holdout": {"feasibility": 0, "throughput": 0},
                    "unspecified": {"feasibility": 0, "throughput": 0},
                },
                "role_outcomes": {
                    role_name: {
                        "success": 0,
                        "oom": 0,
                        "software_failure": 0,
                        "infrastructure_failure": 0,
                    }
                    for role_name in ("calibration", "holdout", "unspecified")
                },
                "role_feasibility_outcomes": {
                    role_name: {"success": 0, "oom": 0}
                    for role_name in ("calibration", "holdout", "unspecified")
                },
                "role_mbs16_success": {
                    role_name: 0
                    for role_name in ("calibration", "holdout", "unspecified")
                },
                "role_outside_supported_mbs": {
                    role_name: set()
                    for role_name in ("calibration", "holdout", "unspecified")
                },
            },
        )
        bucket["observations"] += 1
        bucket[outcome] += 1
        bucket["feasibility_usable"] += int(feasibility_usable)
        bucket["throughput_usable"] += int(throughput_usable)
        bucket["roles"][role] += 1
        bucket["role_usable"][role]["feasibility"] += int(feasibility_usable)
        bucket["role_usable"][role]["throughput"] += int(throughput_usable)
        bucket["role_outcomes"][role][outcome] += 1
        if feasibility_usable and outcome in {"success", "oom"}:
            bucket["role_feasibility_outcomes"][role][outcome] += 1
            mbs = job.get("mbs")
            if type(mbs) is int:
                if mbs not in SUPPORTED_MBS:
                    bucket["role_outside_supported_mbs"][role].add(mbs)
                if outcome == "success" and mbs == 16:
                    bucket["role_mbs16_success"][role] += 1
        if split_unit_id is not None:
            selector_units[key][role].add(split_unit_id)
        if split_policy is not None:
            selector_policies[key].add(split_policy)
        if feasibility_usable:
            inventory = fingerprint.get("runtime_model_inventory") or {}
            for name, value in (
                ("parameter_counts", inventory.get("logical_parameter_elements")),
                ("gpu_counts", job.get("gpu_count")),
                ("micro_batch_sizes", job.get("mbs")),
                ("sequence_lengths", job.get("cutoff_len")),
            ):
                if value is not None:
                    selector_sets[key][role][name].add(value)

    selector_rows = []
    for key in sorted(selectors):
        bucket = selectors[key]
        coverage = {
            role: {
                name: sorted(values, key=str)
                for name, values in sorted(selector_sets[key][role].items())
            }
            for role in ("calibration", "holdout", "unspecified")
        }
        roles = bucket["roles"]
        bucket["coverage_observed"] = coverage
        bucket["split_units"] = {
            role: sorted(selector_units[key][role])
            for role in ("calibration", "holdout", "unspecified")
        }
        bucket["split_policies"] = sorted(selector_policies[key])
        overlap = selector_units[key]["calibration"].intersection(
            selector_units[key]["holdout"]
        )
        role_boundaries = {
            role: _boundary_evidence(
                success=bucket["role_feasibility_outcomes"][role]["success"],
                oom=bucket["role_feasibility_outcomes"][role]["oom"],
                mbs16_success=bucket["role_mbs16_success"][role],
                outside_domain=bucket["role_outside_supported_mbs"][role],
            )
            for role in ("calibration", "holdout", "unspecified")
        }
        bucket["role_boundary_evidence"] = role_boundaries
        del bucket["role_outside_supported_mbs"]
        identifiability_blockers: list[str] = []
        calibration_usable = bucket["role_usable"]["calibration"]
        calibration_coverage = selector_sets[key]["calibration"]
        if calibration_usable["feasibility"] < MIN_CALIBRATION_FEASIBILITY:
            identifiability_blockers.append("insufficient_calibration_feasibility_rows")
        if calibration_usable["throughput"] < MIN_CALIBRATION_THROUGHPUT:
            identifiability_blockers.append("insufficient_calibration_throughput_rows")
        if not role_boundaries["calibration"]["complete"]:
            identifiability_blockers.append(
                "calibration_supported_mbs_boundary_incomplete"
            )
        for name in _required_calibration_variation(bucket["selector"]):
            if len(calibration_coverage[name]) < 2:
                identifiability_blockers.append(f"calibration_{name}_not_varied")
        if bucket["selector"]["zero_stage"] == 3 and len(
            calibration_coverage["gpu_counts"]
        ) < 2:
            identifiability_blockers.append("stage3_calibration_gpu_count_not_varied")

        holdout_blockers: list[str] = []
        holdout_usable = bucket["role_usable"]["holdout"]
        if holdout_usable["feasibility"] < MIN_HOLDOUT_FEASIBILITY:
            holdout_blockers.append("insufficient_holdout_feasibility_rows")
        if holdout_usable["throughput"] < MIN_HOLDOUT_THROUGHPUT:
            holdout_blockers.append("insufficient_holdout_throughput_rows")
        if not role_boundaries["holdout"]["complete"]:
            holdout_blockers.append("holdout_supported_mbs_boundary_incomplete")
        if len(selector_units[key]["holdout"]) < 2:
            holdout_blockers.append("insufficient_independent_holdout_split_units")
        if overlap:
            holdout_blockers.append("calibration_holdout_split_units_overlap")
        if len(selector_policies[key]) != 1:
            holdout_blockers.append("split_policy_missing_or_inconsistent")
        bucket["has_explicit_calibration_partition"] = roles["calibration"] > 0
        bucket["has_independent_holdout_partition"] = roles["holdout"] > 0
        bucket["identifiability_blockers"] = identifiability_blockers
        bucket["holdout_blockers"] = holdout_blockers
        bucket["ready_for_bounded_fit"] = not identifiability_blockers
        bucket["ready_for_holdout_validation"] = not holdout_blockers
        bucket["ready_for_profile_candidate"] = bool(
            not identifiability_blockers and not holdout_blockers
        )
        selector_rows.append(bucket)

    unpacked_dependencies = {
        _dependency_key(row["selector"]): row for row in selector_rows
    }
    packing_by_effect: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in packing_records:
        packing_by_effect[record["effect_key"]].append(record)

    packing_selector_rows: list[dict[str, Any]] = []
    for effect_key in sorted(packing_by_effect):
        records = packing_by_effect[effect_key]
        pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            metadata = record["job"].get("packing_pair")
            pair_id = metadata.get("pair_id") if isinstance(metadata, dict) else None
            if not isinstance(pair_id, str) or not pair_id:
                observation_id = str(record["row"].get("observation_id") or "unknown")
                pair_id = f"missing:{observation_id}"
            pair_groups[pair_id].append(record)

        pair_rows: list[dict[str, Any]] = []
        complete_pair_units: dict[str, set[str]] = defaultdict(set)
        complete_pair_counts: Counter[str] = Counter()
        split_policies: set[str] = set()
        dependency_selectors: dict[str, dict[str, Any]] = {}
        observed_zero_stages: set[int] = set()
        packed_oom_rows = 0
        for record in records:
            if record["outcome"] == "oom" and record["selector"]["packing"] is True:
                packed_oom_rows += 1
            dependency_selectors[record["dependency_key"]] = record[
                "dependency_selector"
            ]
            zero_stage = record["selector"].get("zero_stage")
            if isinstance(zero_stage, int):
                observed_zero_stages.add(zero_stage)

        for pair_id in sorted(pair_groups):
            pair_records = pair_groups[pair_id]
            issues: list[str] = []
            indexed: list[tuple[int, dict[str, Any]]] = []
            for record in pair_records:
                metadata = record["job"].get("packing_pair")
                sequence_index = (
                    metadata.get("sequence_index")
                    if isinstance(metadata, dict)
                    else None
                )
                if type(sequence_index) is not int:
                    issues.append("sequence_index_missing")
                    continue
                indexed.append((sequence_index, record))
            indexed.sort(key=lambda item: item[0])
            ordered = [record for _, record in indexed]
            if len(pair_records) != 4 or [index for index, _ in indexed] != [0, 1, 2, 3]:
                issues.append("abba_rows_incomplete")

            metadata_rows = [record["job"].get("packing_pair") for record in ordered]
            treatments = [
                metadata.get("treatment") if isinstance(metadata, dict) else None
                for metadata in metadata_rows
            ]
            if treatments != ["unpacked", "packed", "packed", "unpacked"]:
                issues.append("abba_treatment_order_invalid")
            if any(
                not isinstance(metadata, dict) or metadata.get("order") != "ABBA"
                for metadata in metadata_rows
            ):
                issues.append("abba_order_binding_missing")
            if [record["job"].get("packing") for record in ordered] != [
                False,
                True,
                True,
                False,
            ]:
                issues.append("packing_treatment_flag_mismatch")
            if [record["job"].get("repeat") for record in ordered] != [0, 0, 1, 1]:
                issues.append("abba_repeat_order_invalid")
            if any(
                record["job"].get("calibration_evidence_class")
                != PACKING_EVIDENCE_CLASS
                for record in ordered
            ):
                issues.append("packing_evidence_class_missing")
            if any(
                record["job"].get("mbs") != 1
                for record in ordered
                if record["selector"]["packing"] is True
            ):
                issues.append("packed_physical_mbs_not_one")
            if any(
                record["outcome"] != "success" or not record["throughput_usable"]
                for record in ordered
            ):
                issues.append("pair_has_non_success_or_unusable_row")
            job_ids = [record["job"].get("job_id") for record in ordered]
            if len(job_ids) != 4 or len(set(job_ids)) != 4:
                issues.append("pair_job_ids_not_unique")
            roles = {record["role"] for record in pair_records}
            units = {record["split_unit_id"] for record in pair_records}
            policies = {record["split_policy"] for record in pair_records}
            dependencies = {record["dependency_key"] for record in pair_records}
            if len(roles) != 1 or next(iter(roles), None) not in CALIBRATION_ROLES:
                issues.append("pair_role_missing_or_inconsistent")
            if len(units) != 1 or next(iter(units), None) is None:
                issues.append("pair_split_unit_missing_or_inconsistent")
            if len(policies) != 1 or next(iter(policies), None) is None:
                issues.append("pair_split_policy_missing_or_inconsistent")
            if len(dependencies) != 1:
                issues.append("pair_mode_stage_dependency_inconsistent")

            role = next(iter(roles), "unspecified")
            split_unit_id = next(iter(units), None)
            split_policy = next(iter(policies), None)
            dependency = next(iter(dependencies), None)
            complete_pair = not issues
            if complete_pair:
                complete_pair_counts[role] += 1
                complete_pair_units[role].add(str(split_unit_id))
                split_policies.add(str(split_policy))
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "role": role,
                    "split_unit_id": split_unit_id,
                    "split_policy": split_policy,
                    "dependency_key": dependency,
                    "job_ids": job_ids,
                    "complete": complete_pair,
                    "issues": sorted(set(issues)),
                }
            )

        dependency_rows = []
        dependencies_complete = True
        for dependency_key in sorted(dependency_selectors):
            unpacked = unpacked_dependencies.get(dependency_key)
            role_boundaries = (
                unpacked.get("role_boundary_evidence")
                if isinstance(unpacked, dict)
                else None
            )
            calibration_complete = bool(
                isinstance(role_boundaries, dict)
                and (role_boundaries.get("calibration") or {}).get("complete")
                is True
            )
            holdout_complete = bool(
                isinstance(role_boundaries, dict)
                and (role_boundaries.get("holdout") or {}).get("complete") is True
            )
            dependency_complete = calibration_complete and holdout_complete
            dependencies_complete = dependencies_complete and dependency_complete
            dependency_rows.append(
                {
                    "dependency_key": dependency_key,
                    "selector": dependency_selectors[dependency_key],
                    "unpacked_selector_found": unpacked is not None,
                    "calibration_boundary_complete": calibration_complete,
                    "holdout_boundary_complete": holdout_complete,
                    "complete": dependency_complete,
                }
            )

        calibration_units = complete_pair_units["calibration"]
        holdout_units = complete_pair_units["holdout"]
        overlap = calibration_units.intersection(holdout_units)
        incomplete_pairs = [row["pair_id"] for row in pair_rows if not row["complete"]]
        calibration_blockers: list[str] = []
        if complete_pair_counts["calibration"] < MIN_PACKING_CALIBRATION_PAIRS:
            calibration_blockers.append("insufficient_complete_calibration_abba_pairs")
        if len(calibration_units) < MIN_PACKING_CALIBRATION_PAIRS:
            calibration_blockers.append("insufficient_independent_calibration_pair_units")
        if packed_oom_rows:
            calibration_blockers.append("packing_introduced_oom")
        if incomplete_pairs:
            calibration_blockers.append("incomplete_or_invalid_abba_pairs")
        if not dependencies_complete:
            calibration_blockers.append("corresponding_unpacked_boundary_incomplete")

        holdout_blockers: list[str] = []
        if complete_pair_counts["holdout"] < MIN_PACKING_HOLDOUT_PAIRS:
            holdout_blockers.append("insufficient_complete_holdout_abba_pairs")
        if len(holdout_units) < MIN_PACKING_HOLDOUT_PAIRS:
            holdout_blockers.append("insufficient_independent_holdout_pair_units")
        if packed_oom_rows:
            holdout_blockers.append("packing_introduced_oom")
        if incomplete_pairs:
            holdout_blockers.append("incomplete_or_invalid_abba_pairs")
        if overlap:
            holdout_blockers.append("calibration_holdout_pair_units_overlap")
        if len(split_policies) != 1:
            holdout_blockers.append("packing_split_policy_missing_or_inconsistent")
        if not dependencies_complete:
            holdout_blockers.append("corresponding_unpacked_boundary_incomplete")

        packing_selector_rows.append(
            {
                "selector": records[0]["effect_selector"],
                "stage_handling": "pooled_effect_with_explicit_zero_stage_covariate",
                "observed_zero_stages": sorted(observed_zero_stages),
                "observations": len(records),
                "outcomes": dict(sorted(Counter(record["outcome"] for record in records).items())),
                "packed_oom_rows": packed_oom_rows,
                "pairs": pair_rows,
                "complete_pairs": dict(sorted(complete_pair_counts.items())),
                "complete_pair_split_units": {
                    role: sorted(complete_pair_units[role])
                    for role in ("calibration", "holdout")
                },
                "split_policies": sorted(split_policies),
                "corresponding_unpacked_boundaries": dependency_rows,
                "calibration_blockers": calibration_blockers,
                "holdout_blockers": holdout_blockers,
                "ready_for_bounded_fit": not calibration_blockers,
                "ready_for_holdout_validation": not holdout_blockers,
                "ready_for_profile_candidate": not calibration_blockers
                and not holdout_blockers,
            }
        )

    global_split_overlap = global_split_units["calibration"].intersection(
        global_split_units["holdout"]
    )
    global_partition_ready = bool(
        not global_split_overlap and len(global_split_policies) == 1
    )

    historical_readiness = (
        analyze_historical_recovery(historical_report, observations_by_id)
        if historical_report is not None
        else None
    )

    native_bounded_fit_ready = any(
        row["ready_for_bounded_fit"] for row in selector_rows
    )
    native_full_bundle_ready = bool(
        selector_rows
        and all(row["ready_for_profile_candidate"] for row in selector_rows)
        and packing_selector_rows
        and all(
            row["ready_for_profile_candidate"] for row in packing_selector_rows
        )
        and global_partition_ready
    )
    historical_bounded_fit_ready = bool(
        isinstance(historical_readiness, dict)
        and historical_readiness.get("any_fit_ready") is True
    )
    historical_full_bundle_ready = bool(
        isinstance(historical_readiness, dict)
        and historical_readiness.get("full_bundle_ready") is True
    )
    bounded_fit_ready = native_bounded_fit_ready or historical_bounded_fit_ready
    full_bundle_ready = native_full_bundle_ready or historical_full_bundle_ready
    bounded_fit_blockers = [] if bounded_fit_ready else [
        "neither_native_v2_nor_validated_historical_evidence_is_fit_ready"
    ]
    full_bundle_blockers = [] if full_bundle_ready else [
        "no_single_validated_evidence_source_has_a_complete_fit_bundle"
    ]
    historical_components = (
        historical_readiness.get("component_readiness") or {}
        if isinstance(historical_readiness, dict)
        else {}
    )
    historical_requirements = (
        historical_readiness.get("requirements") or {}
        if isinstance(historical_readiness, dict)
        else {}
    )
    native_packing_effect_fit_ready = any(
        row["ready_for_bounded_fit"] for row in packing_selector_rows
    )
    historical_packing_effect_fit_ready = bool(
        historical_components.get("packing_low_confidence_effect_fit") is True
    )

    blockers = []
    if complete == 0:
        blockers.append("no_complete_run_bound_execution_fingerprints")
    if complete_authorized_h800 == 0:
        blockers.append("no_approved_exact_h800_execution_evidence")
    if complete_feasibility == 0:
        blockers.append("no_complete_feasibility_observations")
    if complete_throughput == 0:
        blockers.append("no_complete_throughput_observations")
    if not selector_rows or not all(
        row["ready_for_profile_candidate"] for row in selector_rows
    ):
        blockers.append("selector_identifiability_or_holdout_incomplete")
    if not packing_selector_rows:
        blockers.append("packing_pair_evidence_missing")
    elif not all(
        row["ready_for_profile_candidate"] for row in packing_selector_rows
    ):
        blockers.append("packing_pair_evidence_incomplete")
    if not global_partition_ready:
        blockers.append("global_calibration_holdout_partition_inconsistent")
    if historical_readiness is not None:
        blockers.append("historical_recovery_is_not_native_v2_publication_evidence")
    # This audit deliberately does not calculate acceptance metrics. Publication
    # remains blocked until a separate, versioned holdout report demonstrates
    # zero unsafe-as-safe errors, >=95% P95 coverage and <=10% throughput regret.
    blockers.append("approved_holdout_acceptance_report_not_supplied")
    effective_publication_blockers = [
        "bounded_coefficients_not_fit_or_versioned",
        "historical_recovery_is_not_publication_evidence",
        "prospective_acceptance_report_not_supplied",
    ]
    if not bounded_fit_ready:
        effective_publication_blockers.append(
            "no_validated_evidence_source_is_ready_for_bounded_fit"
        )
    report = {
        "schema": SCHEMA,
        "gpu_family": "H800",
        "source": {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        },
        "counts": {
            "observations": total,
            "outcomes": dict(sorted(outcomes.items())),
            "fingerprint_quality": dict(sorted(fingerprint_qualities.items())),
            "complete_fingerprint": complete,
            "complete_approved_exact_h800": complete_authorized_h800,
            "complete_feasibility_usable": complete_feasibility,
            "complete_throughput_usable": complete_throughput,
        },
        "selectors": selector_rows,
        "packing_selectors": packing_selector_rows,
        "partition_audit": {
            "calibration_split_units": sorted(global_split_units["calibration"]),
            "holdout_split_units": sorted(global_split_units["holdout"]),
            "overlap": sorted(global_split_overlap),
            "split_policies": sorted(global_split_policies),
            "ready": global_partition_ready,
        },
        "historical_recovery": historical_readiness,
        "ready_for_any_bounded_fit": bounded_fit_ready,
        "ready_for_full_bounded_fit_bundle": full_bundle_ready,
        "bounded_fit_blockers": bounded_fit_blockers,
        "full_bundle_blockers": full_bundle_blockers,
        "fit_decision": (
            "historical_component_scoped_fit_allowed"
            if historical_bounded_fit_ready
            else "native_v2_fit_allowed"
            if native_bounded_fit_ready
            else "blocked"
        ),
        "fit_scope": {
            "native_v2_any_component_ready": native_bounded_fit_ready,
            "native_v2_full_bundle_ready": native_full_bundle_ready,
            "historical_any_component_ready": historical_bounded_fit_ready,
            "historical_full_bundle_ready": historical_full_bundle_ready,
            "historical_components": historical_components,
            "historical_requirements": historical_requirements,
            "packing_is_automatic_enablement_evidence": False,
        },
        "all_selectors_ready_for_profile_candidate": native_full_bundle_ready,
        "packing_ready_for_any_effect_fit": bool(
            native_packing_effect_fit_ready
            or historical_packing_effect_fit_ready
        ),
        "native_v2_packing_ready_for_effect_fit": native_packing_effect_fit_ready,
        "historical_packing_ready_for_low_confidence_effect_fit": (
            historical_packing_effect_fit_ready
        ),
        "calibration_publishable": False,
        # Kept for native-v2 audit compatibility.  Consumers deciding whether
        # the current historical path may publish should use the effective list.
        "publication_blocker_scope": "native_v2_plus_global_compatibility",
        "publication_blockers": blockers,
        "native_v2_publication_blockers": blockers,
        "effective_publication_blockers": sorted(
            set(effective_publication_blockers)
        ),
        "feature_gates": {
            "packing_automatic_enablement": {
                "ready": False,
                "blockers": ["historical_packing_pairs_are_not_abba"],
            }
        },
        "coefficients_were_fit": False,
        "implementation_sources": {
            "scripts/audit_h800_calibration_readiness.py": sha256_file(
                ROOT / "scripts" / "audit_h800_calibration_readiness.py"
            ),
            "scripts/historical_h800_readiness.py": sha256_file(
                ROOT / "scripts" / "historical_h800_readiness.py"
            ),
            "scripts/recover_h800_historical_evidence.py": sha256_file(
                ROOT / "scripts" / "recover_h800_historical_evidence.py"
            ),
        },
        "policy": {
            "partial_4090_results_allowed": False,
            "legacy_incomplete_allowed_for_calibration": False,
            "historical_recovered_sidecar_allowed_for_bounded_fit": True,
            "historical_recovered_sidecar_allowed_for_publication": False,
            "historical_fold_dependent_loocv_supported": True,
            "duplicate_observations_allowed": False,
            "calibration_holdout_split_units_must_be_disjoint": True,
            "minimum_calibration_feasibility_rows_per_selector": MIN_CALIBRATION_FEASIBILITY,
            "minimum_calibration_throughput_rows_per_selector": MIN_CALIBRATION_THROUGHPUT,
            "minimum_holdout_feasibility_rows_per_selector": MIN_HOLDOUT_FEASIBILITY,
            "minimum_holdout_throughput_rows_per_selector": MIN_HOLDOUT_THROUGHPUT,
            "supported_micro_batch_sizes": sorted(SUPPORTED_MBS),
            "unpacked_boundary_completion": (
                "success and (OOM or complete-v2 success at MBS=16)"
            ),
            "out_of_domain_mbs_required": False,
            "minimum_packing_calibration_abba_pairs_per_training_mode": MIN_PACKING_CALIBRATION_PAIRS,
            "minimum_packing_holdout_abba_pairs_per_training_mode": MIN_PACKING_HOLDOUT_PAIRS,
            "packing_requires_zero_new_oom": True,
            "packing_requires_corresponding_unpacked_mode_stage_boundaries": True,
            "publication_requires_separate_approved_holdout_report": True,
        },
    }
    report["report_sha256"] = sha256_json(report)
    return report


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
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts" / "canonical_h800_observations.jsonl",
    )
    parser.add_argument(
        "--historical-recovery",
        type=Path,
        help=(
            "Source-bound historical recovery sidecar. If omitted, the default "
            "artifact is used when present."
        ),
    )
    parser.add_argument("--verify-historical-sources", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    historical_path = args.historical_recovery
    default_historical_path = ROOT / "artifacts" / "historical_h800_recovery.json"
    if historical_path is None and default_historical_path.is_file():
        historical_path = default_historical_path
    report = audit(
        args.input,
        historical_path,
        verify_historical_sources=args.verify_historical_sources,
    )
    if args.output is not None:
        write_report(args.output, report)
        historical = report.get("historical_recovery") or {}
        print(
            json.dumps(
                {
                    "schema": report["schema"],
                    "output": str(args.output),
                    "source": report["source"],
                    "native_v2_counts": report["counts"],
                    "historical_counts": historical.get("counts"),
                    "ready_for_any_bounded_fit": report[
                        "ready_for_any_bounded_fit"
                    ],
                    "ready_for_full_bounded_fit_bundle": report[
                        "ready_for_full_bounded_fit_bundle"
                    ],
                    "fit_decision": report["fit_decision"],
                    "historical_component_readiness": (
                        historical.get("component_readiness") or {}
                    ),
                    "historical_fit_requirements": (
                        historical.get("requirements") or {}
                    ),
                    "calibration_publishable": report["calibration_publishable"],
                    "effective_publication_blockers": report[
                        "effective_publication_blockers"
                    ],
                    "report_sha256": report["report_sha256"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
