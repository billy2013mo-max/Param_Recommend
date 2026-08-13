#!/usr/bin/env python3
"""Build, validate, and evaluate immutable historical memory anchors.

Anchors do not refit or overwrite the physical memory model.  They may only
replace an over-broad global tail bound for a tightly matched configuration
when repeated exact evidence plus explicit safety margins still fit below the
hardware safety limit.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from common import sha256_file, sha256_json


CANDIDATE_SCHEMA = "sft_memory_anchor_candidates/v1"
REGISTRY_SCHEMA = "sft_historical_memory_anchor_registry/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_observations(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Observation line {line_number} is not an object")
            rows.append(row)
    return rows


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_registry_immutable(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        existing = _read_json(path)
        if existing == value:
            return
        raise FileExistsError(
            f"Refusing to overwrite immutable registry {path}; "
            "create a new versioned output path"
        )
    _write_json_atomic(path, value)


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _job(row: Mapping[str, Any]) -> dict[str, Any]:
    configuration = row.get("configuration")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    job = configuration.get("job")
    return dict(job) if isinstance(job, Mapping) else {}


def _matches_job(row: Mapping[str, Any], match: Mapping[str, Any]) -> bool:
    job = _job(row)
    return all(job.get(key) == value for key, value in match.items())


def _outcome(row: Mapping[str, Any]) -> str:
    outcome = row.get("outcome")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    return str(outcome.get("class") or "").lower()


def _reserved_bytes(row: Mapping[str, Any]) -> int | None:
    measurements = row.get("measurements")
    measurements = measurements if isinstance(measurements, Mapping) else {}
    memory = measurements.get("memory")
    memory = memory if isinstance(memory, Mapping) else {}
    value = memory.get("max_reserved_bytes")
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _complete_eligible(row: Mapping[str, Any], gpu_family: str) -> bool:
    fingerprint = row.get("fingerprint")
    fingerprint = fingerprint if isinstance(fingerprint, Mapping) else {}
    hardware = row.get("hardware")
    hardware = hardware if isinstance(hardware, Mapping) else {}
    outcome = row.get("outcome")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    return bool(
        fingerprint.get("quality") == "complete"
        and fingerprint.get("calibration_evidence_eligible") is True
        and hardware.get("gpu_family") == gpu_family
        and outcome.get("usable_for_feasibility_calibration") is True
    )


def _runtime_mechanism_sha(row: Mapping[str, Any]) -> str | None:
    fingerprint = row.get("fingerprint")
    fingerprint = fingerprint if isinstance(fingerprint, Mapping) else {}
    artifacts = fingerprint.get("bound_artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    mechanism = artifacts.get("runtime_mechanism")
    mechanism = mechanism if isinstance(mechanism, Mapping) else {}
    value = mechanism.get("component_sha256")
    return str(value) if isinstance(value, str) and value else None


def _profile_max_tokens(path: Path) -> tuple[int, int]:
    rows = 0
    maximum = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            value = row.get("total_tokens")
            if isinstance(value, bool):
                raise ValueError(f"{path}:{line_number}.total_tokens is invalid")
            try:
                tokens = int(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line_number}.total_tokens is invalid"
                ) from error
            if tokens <= 0:
                raise ValueError(f"{path}:{line_number}.total_tokens is invalid")
            rows += 1
            maximum = max(maximum, tokens)
    if not rows:
        raise ValueError(f"Empty source profile {path}")
    return rows, maximum


def _anchor_from_candidate(
    candidate: Mapping[str, Any],
    *,
    observations: list[dict[str, Any]],
    governance: Mapping[str, Any],
    physical_capacity_bytes: int,
) -> dict[str, Any]:
    job_match = candidate["job_match"]
    runtime_match = candidate["runtime_match"]
    exact = [row for row in observations if _matches_job(row, job_match)]
    successes = [row for row in exact if _outcome(row) == "success"]
    ooms = [row for row in exact if _outcome(row) == "oom"]
    eligible = [
        row
        for row in successes
        if _complete_eligible(row, str(runtime_match["gpu_family"]))
    ]
    corroborating = [row for row in successes if row not in eligible]
    complete_job_ids = sorted(
        {str(_job(row).get("job_id")) for row in eligible if _job(row).get("job_id")}
    )
    eligible_runtime_shas = sorted(
        {
            value
            for row in eligible
            if (value := _runtime_mechanism_sha(row)) is not None
        }
    )
    observed = [
        value for row in successes if (value := _reserved_bytes(row)) is not None
    ]
    safe_limit = physical_capacity_bytes * float(
        governance["safe_limit_fraction_of_physical_capacity"]
    )
    observed_guard = (
        max(observed) * (1.0 + float(governance["observed_peak_relative_guard"]))
        if observed
        else None
    )

    profile = candidate["source_dataset_profile"]
    profile_path = _resolve_project_path(str(profile["path"]))
    if sha256_file(profile_path) != profile["sha256"]:
        raise ValueError(
            f"Anchor {candidate['anchor_id']} source profile checksum mismatch"
        )
    profile_rows, profile_max = _profile_max_tokens(profile_path)
    model_binding = candidate["model_binding"]
    model_config_path = _resolve_project_path(str(model_binding["model_config_path"]))
    if sha256_file(model_config_path) != model_binding["model_config_sha256"]:
        raise ValueError(
            f"Anchor {candidate['anchor_id']} model config checksum mismatch"
        )

    checks = [
        {
            "name": "minimum_complete_calibration_eligible_successes",
            "actual": len(eligible),
            "required": governance["minimum_complete_calibration_eligible_successes"],
            "passed": len(eligible)
            >= governance["minimum_complete_calibration_eligible_successes"],
        },
        {
            "name": "minimum_total_successes_with_corroboration",
            "actual": len(successes),
            "required": governance["minimum_total_successes_with_corroboration"],
            "passed": len(successes)
            >= governance["minimum_total_successes_with_corroboration"],
        },
        {
            "name": "maximum_exact_configuration_ooms",
            "actual": len(ooms),
            "required": governance["maximum_exact_configuration_ooms"],
            "passed": len(ooms) <= governance["maximum_exact_configuration_ooms"],
        },
        {
            "name": "distinct_complete_job_ids",
            "actual": len(complete_job_ids),
            "required": governance["minimum_complete_calibration_eligible_successes"],
            "passed": (
                not governance["require_distinct_complete_job_ids"]
                or len(complete_job_ids)
                >= governance["minimum_complete_calibration_eligible_successes"]
            ),
        },
        {
            "name": "runtime_mechanism_binding",
            "actual": eligible_runtime_shas,
            "required": [runtime_match["runtime_mechanism_component_sha256"]],
            "passed": eligible_runtime_shas
            == [runtime_match["runtime_mechanism_component_sha256"]],
        },
        {
            "name": "observed_guard_below_safe_limit",
            "actual": observed_guard,
            "required": safe_limit,
            "passed": observed_guard is not None and observed_guard <= safe_limit,
        },
    ]
    active = all(check["passed"] for check in checks)
    return {
        "anchor_id": candidate["anchor_id"],
        "status": "active_provisional" if active else "rejected",
        "memory_gate_override_allowed": active,
        "job_match": job_match,
        "runtime_match": runtime_match,
        "model_binding": model_binding,
        "source_dataset_profile": {
            **profile,
            "path": str(profile_path.resolve()),
            "records": profile_rows,
            "maximum_total_tokens": profile_max,
        },
        "transfer_contract": candidate["transfer_contract"],
        "evidence": {
            "exact_rows": len(exact),
            "successes": len(successes),
            "ooms": len(ooms),
            "complete_calibration_eligible_successes": len(eligible),
            "corroborating_successes": len(corroborating),
            "complete_job_ids": complete_job_ids,
            "eligible_observation_ids": sorted(
                str(row["observation_id"]) for row in eligible
            ),
            "corroborating_observation_ids": sorted(
                str(row["observation_id"]) for row in corroborating
            ),
            "oom_observation_ids": sorted(str(row["observation_id"]) for row in ooms),
            "observed_reserved_bytes": sorted(observed),
            "maximum_observed_reserved_bytes": (max(observed) if observed else None),
            "empirical_guard_bytes": observed_guard,
            "empirical_guard_formula": (
                "max_observed_reserved_bytes * "
                f"(1 + {governance['observed_peak_relative_guard']})"
            ),
            "safe_limit_bytes": safe_limit,
            "headroom_after_empirical_guard_bytes": (
                safe_limit - observed_guard if observed_guard is not None else None
            ),
        },
        "promotion_checks": checks,
    }


def build_registry(
    candidate_path: Path,
    observation_path: Path,
    hardware_path: Path,
) -> dict[str, Any]:
    candidates = _read_json(candidate_path)
    if not isinstance(candidates, dict) or candidates.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError(f"Candidates must use schema {CANDIDATE_SCHEMA}")
    governance = candidates["governance"]
    if governance.get("online_gpu_execution_allowed") is not False:
        raise ValueError("Anchor governance must prohibit online GPU execution")
    observations = _read_observations(observation_path)
    hardware = _read_json(hardware_path)
    capacity = int(hardware["memory_bytes_reported_by_torch"])
    anchors = [
        _anchor_from_candidate(
            candidate,
            observations=observations,
            governance=governance,
            physical_capacity_bytes=capacity,
        )
        for candidate in candidates["candidates"]
    ]
    report: dict[str, Any] = {
        "schema": REGISTRY_SCHEMA,
        "registry_id": candidates["registry_id"],
        "version": candidates["version"],
        "generated_at_utc": candidates["frozen_at_utc"],
        "immutable": True,
        "previous_registry_sha256": candidates["previous_registry_sha256"],
        "online_gpu_execution_performed": False,
        "governance": governance,
        "source_bindings": {
            "candidate_specification": {
                "path": str(candidate_path.resolve()),
                "sha256": sha256_file(candidate_path),
            },
            "canonical_observations": {
                "path": str(observation_path.resolve()),
                "sha256": sha256_file(observation_path),
                "rows": len(observations),
            },
            "hardware": {
                "path": str(hardware_path.resolve()),
                "sha256": sha256_file(hardware_path),
                "physical_capacity_bytes": capacity,
            },
        },
        "model_replacement_policy": {
            "overwrite_existing_artifact_allowed": False,
            "new_version_required": True,
            "candidate_must_be_frozen_before_holdout": True,
            "holdout_must_be_unseen": True,
            "anchor_evidence_may_not_double_as_replacement_holdout": True,
            "minimum_acceptance": {
                "false_safe_oom": 0,
                "success_p95_coverage": 0.95,
                "safe_success_admission_recall": 0.868421052631579,
                "reserved_center_mean_absolute_percentage_error": 0.06,
                "reserved_center_p90_absolute_percentage_error": 0.12,
                "targeted_boundary_false_rejects_must_decrease": True,
                "no_supported_slice_may_regress_safety": True,
            },
            "required_artifacts": [
                "immutable candidate model",
                "predeclared calibration/holdout split",
                "before/after replay on the same frozen holdout",
                "slice metrics by model scale, train type, ZeRO, GC, MBS, and cutoff",
                "signed approval record and rollback pointer",
            ],
        },
        "anchors": anchors,
        "active_anchor_count": sum(
            anchor["memory_gate_override_allowed"] for anchor in anchors
        ),
    }
    unsigned = dict(report)
    report["report_sha256"] = sha256_json(unsigned)
    validate_registry(report)
    return report


def validate_registry(report: Mapping[str, Any]) -> None:
    if report.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(f"Registry must use schema {REGISTRY_SCHEMA}")
    unsigned = dict(report)
    digest = unsigned.pop("report_sha256", None)
    if not isinstance(digest, str) or digest != sha256_json(unsigned):
        raise ValueError("Memory anchor registry checksum mismatch")
    if report.get("immutable") is not True:
        raise ValueError("Memory anchor registry must be immutable")
    if report.get("online_gpu_execution_performed") is not False:
        raise ValueError("Memory anchor registry cannot execute online GPU work")
    governance = report.get("governance") or {}
    if governance.get("online_gpu_execution_allowed") is not False:
        raise ValueError("Memory anchor governance drifted")
    active = 0
    seen: set[str] = set()
    for anchor in report.get("anchors") or []:
        anchor_id = anchor.get("anchor_id")
        if not isinstance(anchor_id, str) or not anchor_id or anchor_id in seen:
            raise ValueError("Memory anchor ids must be present and unique")
        seen.add(anchor_id)
        allowed = anchor.get("memory_gate_override_allowed") is True
        if allowed:
            active += 1
            if anchor.get("status") != "active_provisional":
                raise ValueError("Active memory anchor status drifted")
            if not all(
                check.get("passed") is True
                for check in anchor.get("promotion_checks") or []
            ):
                raise ValueError("Active memory anchor has a failed check")
            evidence = anchor.get("evidence") or {}
            if evidence.get("ooms") != 0:
                raise ValueError("Active memory anchor contains an OOM")
            if not (
                float(evidence["empirical_guard_bytes"])
                <= float(evidence["safe_limit_bytes"])
            ):
                raise ValueError("Active memory anchor guard exceeds safe limit")
    if report.get("active_anchor_count") != active:
        raise ValueError("Memory anchor active count drifted")
    replacement = report.get("model_replacement_policy") or {}
    if (
        replacement.get("overwrite_existing_artifact_allowed") is not False
        or replacement.get("new_version_required") is not True
    ):
        raise ValueError("Memory model replacement policy drifted")


def load_registry(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    if not isinstance(report, dict):
        raise ValueError("Memory anchor registry is not an object")
    validate_registry(report)
    return report


def _record_job(record: Mapping[str, Any]) -> dict[str, Any]:
    scenario = record.get("scenario") or {}
    selector = record.get("selector") or {}
    zero_stage = int(selector.get("zero_stage") or 0)
    return {
        "model_id": scenario.get("model_id"),
        "train_type": selector.get("training_mode"),
        "target_gbs": scenario.get("target_gbs"),
        "gpu_count": scenario.get("gpu_count"),
        "zero": "none" if zero_stage == 0 else f"zero{zero_stage}",
        "gc": selector.get("gradient_checkpointing"),
        "mbs": scenario.get("physical_mbs"),
        "cutoff_len": scenario.get("cutoff_len"),
        "packing": selector.get("packing"),
    }


def _match_anchor(
    record: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    candidate_job = _record_job(record)
    anchor_job = anchor["job_match"]
    transfer = anchor["transfer_contract"]
    for key, expected in anchor_job.items():
        actual = candidate_job.get(key)
        if (
            key == "cutoff_len"
            and transfer["candidate_cutoff_len_must_not_exceed_anchor"]
        ):
            if not isinstance(actual, int) or actual > expected:
                issues.append("cutoff_len_exceeds_anchor")
        elif key == "mbs" and transfer["candidate_physical_mbs_must_not_exceed_anchor"]:
            if not isinstance(actual, int) or actual > expected:
                issues.append("physical_mbs_exceeds_anchor")
        elif actual != expected:
            issues.append(f"{key}_mismatch")
    selector = record.get("selector") or {}
    runtime = anchor["runtime_match"]
    record_runtime = record.get("runtime") or {}
    if record_runtime.get("gpu_family") != runtime["gpu_family"]:
        issues.append("gpu_family_mismatch")
    if (
        record_runtime.get("runtime_mechanism_component_sha256")
        != runtime["runtime_mechanism_component_sha256"]
    ):
        issues.append("runtime_mechanism_mismatch")
    if selector.get("dtype") != runtime["dtype"]:
        issues.append("dtype_mismatch")
    if selector.get("kernel_path") != runtime["kernel_path"]:
        issues.append("kernel_path_mismatch")
    model = anchor["model_binding"]
    basis = record.get("model_basis") or {}
    if basis.get("base_parameters") != model["model_parameters"]:
        issues.append("model_parameters_mismatch")
    if sha256_json(basis) != model["model_geometry_sha256"]:
        issues.append("model_geometry_mismatch")
    return issues


def evaluate_anchor_admission(
    record: Mapping[str, Any],
    *,
    base_prediction: Mapping[str, Any],
    support: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a local guarded upper bound, without mutating the base model."""
    validate_registry(registry)
    matches: list[tuple[Mapping[str, Any], list[str]]] = []
    diagnostics: list[dict[str, Any]] = []
    for anchor in registry["anchors"]:
        issues = _match_anchor(record, anchor)
        diagnostics.append(
            {
                "anchor_id": anchor["anchor_id"],
                "matched": not issues,
                "issues": issues,
            }
        )
        if not issues and anchor.get("memory_gate_override_allowed") is True:
            matches.append((anchor, issues))
    if len(matches) > 1:
        raise ValueError("More than one active memory anchor matches a request")
    if not matches:
        return {
            "matched": False,
            "override_allowed": False,
            "issues": ["no_active_exact_or_monotonic_anchor_match"],
            "match_diagnostics": diagnostics,
        }

    anchor = matches[0][0]
    center = base_prediction.get("reserved_center_bytes")
    safe_limit = (record.get("memory") or {}).get("safe_limit_bytes")
    if center is None or safe_limit is None:
        return {
            "matched": True,
            "anchor_id": anchor["anchor_id"],
            "override_allowed": False,
            "issues": ["base_center_or_safe_limit_unavailable"],
            "match_diagnostics": diagnostics,
        }
    center = float(center)
    safe_limit = float(safe_limit)
    governance = registry["governance"]
    empirical_guard = float(anchor["evidence"]["empirical_guard_bytes"])
    center_guard = center * (1.0 + float(governance["model_center_relative_guard"]))
    local_upper = max(empirical_guard, center_guard)
    issues = []
    if support.get("label") == "unsupported":
        issues.append("support_domain_is_unsupported")
    if center > safe_limit:
        issues.append("base_center_exceeds_safe_limit")
    if local_upper > safe_limit:
        issues.append("local_anchor_upper_exceeds_safe_limit")
    return {
        "matched": True,
        "anchor_id": anchor["anchor_id"],
        "anchor_status": anchor["status"],
        "override_allowed": not issues,
        "local_guarded_upper_bytes": local_upper,
        "safe_limit_bytes": safe_limit,
        "headroom_to_safe_limit_bytes": safe_limit - local_upper,
        "guard_components": {
            "empirical_guard_bytes": empirical_guard,
            "model_center_guard_bytes": center_guard,
            "model_center_relative_guard": governance["model_center_relative_guard"],
        },
        "evidence": {
            "complete_calibration_eligible_successes": anchor["evidence"][
                "complete_calibration_eligible_successes"
            ],
            "corroborating_successes": anchor["evidence"]["corroborating_successes"],
            "ooms": anchor["evidence"]["ooms"],
            "eligible_observation_ids": anchor["evidence"]["eligible_observation_ids"],
        },
        "issues": issues,
        "match_diagnostics": diagnostics,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument(
        "--candidates",
        type=Path,
        default=PROJECT_ROOT
        / "offline_experiments"
        / "config"
        / "memory_anchor_candidates_v1.json",
    )
    build.add_argument(
        "--observations",
        type=Path,
        default=PROJECT_ROOT
        / "offline_experiments"
        / "artifacts"
        / "canonical_h800_observations.jsonl",
    )
    build.add_argument(
        "--hardware",
        type=Path,
        default=PROJECT_ROOT / "offline_experiments" / "config" / "hardware.json",
    )
    build.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--registry", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        report = build_registry(
            args.candidates.expanduser().resolve(),
            args.observations.expanduser().resolve(),
            args.hardware.expanduser().resolve(),
        )
        _write_registry_immutable(args.output.expanduser().resolve(), report)
    else:
        report = load_registry(args.registry.expanduser().resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
