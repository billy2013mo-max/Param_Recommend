#!/usr/bin/env python3
"""Read-only gate audit for a new H800 campaign design.

The design builders intentionally stop before queue materialization.  This
module is the small hand-off point used after the external prerequisites have
arrived: it re-probes the exact GPU pool, checks that fresh profiles (when the
campaign needs them) are present, and verifies that a newly promoted approval
binds the exact design bytes.  It never writes a queue and never launches a
process.  The only write is an optional JSON readiness report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from common import ARTIFACT_DIR, ROOT, read_json, sha256_file, write_json
from prepare_h800_prospective_holdout import probe_hardware


SCHEMA = "sft_h800_campaign_gate_report/v1"
PROFILE_REQUIREMENTS_SCHEMA = "sft_h800_fresh_profile_requirements/v1"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_fresh_holdout_design_v2.json"
DEFAULT_APPROVAL = ROOT / "config" / "APPROVED_TO_RUN.json"
DEFAULT_APPROVAL_DESIGN = ROOT / "runtime" / "approval_design.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_campaign_gate_report_v2.json"
DEFAULT_PROFILE_REQUIREMENTS = ARTIFACT_DIR / "h800_fresh_profile_requirements_v2.json"


def _check(name: str, passed: bool, reason: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "reason": reason, **details}


def _profile_check(
    design: Mapping[str, Any],
    root: Path,
    requirements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require an explicit, existing profile path for prospective holdouts."""

    schema = str(design.get("schema") or "")
    if "prospective_holdout" not in schema:
        return _check(
            "fresh_profiles",
            True,
            "campaign is not a prospective holdout; no dataset profile gate applies",
            required=False,
            missing=[],
            present=[],
        )
    missing: list[str] = []
    present: list[str] = []
    requirements_by_id = {
        str(row.get("scenario_id")): row
        for row in (requirements or {}).get("scenarios") or []
        if isinstance(row, Mapping) and row.get("scenario_id")
    }
    for scenario in design.get("scenarios") or []:
        scenario_id = str(scenario.get("scenario_id") or "<unknown>")
        freshness = scenario.get("freshness") or {}
        requirement = requirements_by_id.get(scenario_id) or {}
        requirement_bindings = requirement.get("required_bindings") or {}
        raw_path = (
            scenario.get("profile_path")
            or freshness.get("profile_path")
            or requirement_bindings.get("profile_path")
        )
        if not raw_path:
            missing.append(f"{scenario_id}:profile_path")
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            missing.append(f"{scenario_id}:{path}")
        else:
            present.append(str(path.resolve()))
    return _check(
        "fresh_profiles",
        not missing and bool(present),
        "all scenarios have an existing processor-bound profile"
        if not missing and present
        else "fresh processor-bound profiles are missing or not bound in the design",
        required=True,
        missing=missing,
        present=present,
    )


def _sha256_text(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _profile_content_check(
    path: Path,
    *,
    required_fields: Sequence[str],
) -> dict[str, Any]:
    """Validate the sample-level shape of a processor-bound profile.

    A file hash proves identity, not that the file is a usable workload
    profile.  Fresh intake accepts either JSONL sample rows or a JSON object
    carrying ``rows``/``records``/``samples``.  It deliberately does not
    infer processor metadata from the rows; that remains bound by the explicit
    processor contract and its SHA in the requirements artifact.
    """

    rows: list[Mapping[str, Any]] = []
    malformed = 0
    parse_error: str | None = None
    try:
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    malformed += 1
                else:
                    rows.append(value)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                values = payload
            elif isinstance(payload, Mapping):
                values = (
                    payload.get("rows")
                    or payload.get("records")
                    or payload.get("samples")
                )
                if values is None and all(field in payload for field in required_fields):
                    values = [payload]
            else:
                values = None
            if not isinstance(values, list):
                malformed += 1
                values = []
            for value in values:
                if not isinstance(value, Mapping):
                    malformed += 1
                else:
                    rows.append(value)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        parse_error = type(error).__name__

    required = [str(field) for field in required_fields]
    missing_fields = sorted(
        {
            field
            for row in rows
            for field in required
            if field not in row
        }
    )
    invalid_numeric = 0
    duplicate_sample_ids = 0
    sample_ids: set[str] = set()
    for row in rows:
        numeric_fields = ("total_tokens", "label_tokens", "turns", "assistant_turns")
        for field in numeric_fields:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
                invalid_numeric += 1
        total = row.get("total_tokens")
        label = row.get("label_tokens")
        if isinstance(total, (int, float)) and isinstance(label, (int, float)) and label > total:
            invalid_numeric += 1
        sample_id = row.get("sample_id")
        if sample_id is not None:
            key = str(sample_id)
            if key in sample_ids:
                duplicate_sample_ids += 1
            sample_ids.add(key)
    passed = bool(
        rows
        and not parse_error
        and malformed == 0
        and not missing_fields
        and invalid_numeric == 0
        and duplicate_sample_ids == 0
    )
    return {
        "path": str(path.resolve()),
        "passed": passed,
        "rows": len(rows),
        "parse_error": parse_error,
        "malformed_rows": malformed,
        "missing_fields": missing_fields,
        "invalid_numeric_rows": invalid_numeric,
        "duplicate_sample_ids": duplicate_sample_ids,
        "required_fields": required,
    }


def _requirements_check(
    design: Mapping[str, Any],
    design_path: Path,
    requirements: Mapping[str, Any] | None,
    requirements_path: Path | None,
    root: Path,
) -> dict[str, Any]:
    """Validate the externally filled fresh-profile/data intake contract."""

    if "prospective_holdout" not in str(design.get("schema") or ""):
        return _check(
            "fresh_profile_requirements",
            True,
            "campaign is not a prospective holdout; no fresh intake contract applies",
            required=False,
            missing=[],
            mismatched=[],
        )
    if requirements is None:
        return _check(
            "fresh_profile_requirements",
            False,
            "fresh profile requirements artifact is absent or unreadable",
            required=True,
            path=str(requirements_path.resolve()) if requirements_path else None,
            missing=["requirements_artifact"],
            mismatched=[],
        )
    missing: list[str] = []
    mismatched: list[str] = []
    profile_content: dict[str, dict[str, Any]] = {}
    if requirements.get("schema") != PROFILE_REQUIREMENTS_SCHEMA:
        mismatched.append("schema")
    if requirements.get("campaign_id") != design.get("campaign_id"):
        mismatched.append("campaign_id")
    expected_design_sha = sha256_file(design_path)
    binding = requirements.get("design_binding") or {}
    if binding.get("path"):
        bound_path = Path(str(binding["path"]))
        if not bound_path.is_absolute():
            bound_path = root / bound_path
        if bound_path.resolve() != design_path.resolve():
            mismatched.append("design_binding.path")
    if binding.get("sha256") != expected_design_sha:
        mismatched.append("design_binding.sha256")
    rows = requirements.get("scenarios")
    if not isinstance(rows, list):
        missing.append("scenarios")
        rows = []
    design_ids = {
        str(row.get("scenario_id"))
        for row in design.get("scenarios") or []
        if isinstance(row, Mapping)
    }
    requirement_ids = {
        str(row.get("scenario_id"))
        for row in rows
        if isinstance(row, Mapping)
    }
    if design_ids != requirement_ids:
        mismatched.append("scenario_ids")
    for row in rows:
        if not isinstance(row, Mapping):
            missing.append("scenario_row")
            continue
        scenario_id = str(row.get("scenario_id") or "<unknown>")
        design_row = next(
            (
                candidate
                for candidate in design.get("scenarios") or []
                if isinstance(candidate, Mapping)
                and str(candidate.get("scenario_id")) == scenario_id
            ),
            None,
        )
        if design_row is None:
            mismatched.append(f"{scenario_id}:unknown_scenario")
            continue
        for field in ("dataset_profile_id", "model_id", "cutoff_len"):
            if row.get(field) != design_row.get(field):
                mismatched.append(f"{scenario_id}:{field}")
        bindings = row.get("required_bindings") or {}
        if not isinstance(bindings, Mapping):
            missing.append(f"{scenario_id}:required_bindings")
            continue
        for field in ("profile_path", "data_path"):
            raw_path = bindings.get(field)
            if not raw_path:
                missing.append(f"{scenario_id}:{field}")
                continue
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                missing.append(f"{scenario_id}:{field}:file")
                continue
            expected = bindings.get(field.replace("_path", "_sha256"))
            if not _sha256_text(expected):
                missing.append(f"{scenario_id}:{field.replace('_path', '_sha256')}")
            elif sha256_file(path) != expected:
                mismatched.append(f"{scenario_id}:{field}_sha256")
            if field == "profile_path":
                content = _profile_content_check(
                    path,
                    required_fields=row.get("required_profile_row_fields") or (
                        "sample_id",
                        "total_tokens",
                        "label_tokens",
                        "turns",
                        "assistant_turns",
                    ),
                )
                profile_content[scenario_id] = content
                if content["passed"] is not True:
                    missing.append(f"{scenario_id}:profile_content")
        if bindings.get("runtime_dataset_registered") is not True:
            missing.append(f"{scenario_id}:runtime_dataset_registered")
        for field in ("processor_contract_sha256", "split_manifest_sha256"):
            if not _sha256_text(bindings.get(field)):
                missing.append(f"{scenario_id}:{field}")
    if requirements.get("ready_for_materialization") is not True:
        missing.append("ready_for_materialization")
    passed = not missing and not mismatched
    return _check(
        "fresh_profile_requirements",
        passed,
        "fresh profile/data intake contract is complete and hash-verified"
        if passed
        else "fresh profile/data intake contract is incomplete or mismatched",
        required=True,
        path=str(requirements_path.resolve()) if requirements_path else None,
        missing=sorted(set(missing)),
        mismatched=sorted(set(mismatched)),
        profile_content=profile_content,
        scenario_count=len(rows),
    )


def _approval_check(
    design: Mapping[str, Any],
    design_path: Path,
    approval: Mapping[str, Any] | None,
    required_gpu_ids: Sequence[int],
    approval_design: Mapping[str, Any] | None = None,
    approval_design_path: Path | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Check exact approval binding without changing the live approval file."""

    if approval is None:
        return _check(
            "approval_binding",
            False,
            "approval file is absent or unreadable",
            expected_design_sha256=sha256_file(design_path),
            actual_design_sha256=None,
        )
    expected_campaign_design_sha = sha256_file(design_path)
    actual_design_sha = approval.get("design_sha256")
    scope = approval.get("resource_scope") or {}
    actual_gpu_ids = sorted(int(value) for value in scope.get("gpu_ids") or [])
    expected_gpu_ids = sorted(int(value) for value in required_gpu_ids)
    queue_binding = approval.get("queue_binding_sha256")
    runtime_fingerprint = approval.get("runtime_fingerprint_sha256")
    # The live approval authorizes an executable approval-design, which in
    # turn binds the non-executable campaign design.  V1 campaigns used the
    # campaign design directly; retain that fail-closed compatibility path.
    approval_design_sha = (
        sha256_file(approval_design_path)
        if approval_design is not None
        and approval_design_path is not None
        and approval_design_path.is_file()
        else None
    )
    campaign_binding = (approval_design or {}).get("campaign_design") or {}
    expected_join_busy_pool = (
        ((design.get("required_gpu_pool") or {}).get("join_busy_pool_allowed") is True)
    )
    approval_join_busy_pool = (
        ((approval_design or {}).get("scheduler_execution") or {}).get("join_busy_pool")
        is True
    )
    bound_path = campaign_binding.get("path")
    if bound_path:
        resolved_bound_path = Path(str(bound_path))
        if not resolved_bound_path.is_absolute():
            resolved_bound_path = root / resolved_bound_path
        campaign_path_matches = resolved_bound_path.resolve() == design_path.resolve()
    else:
        campaign_path_matches = False
    executable_design_chain_passed = bool(
        approval_design_sha
        and actual_design_sha == approval_design_sha
        and campaign_binding.get("sha256") == expected_campaign_design_sha
        and campaign_path_matches
        and approval_join_busy_pool == expected_join_busy_pool
    )
    legacy_direct_binding_passed = bool(
        approval_design is None and actual_design_sha == expected_campaign_design_sha
    )
    passed = (
        approval.get("approved") is True
        and (executable_design_chain_passed or legacy_direct_binding_passed)
        and actual_gpu_ids == expected_gpu_ids
        and _sha256_text(queue_binding)
        and _sha256_text(runtime_fingerprint)
    )
    if passed:
        reason = "approval binds exact design bytes, GPU scope, queue and runtime fingerprints"
    else:
        reason = "live approval does not bind this design, exact GPU scope, queue, and runtime"
    return _check(
        "approval_binding",
        passed,
        reason,
        expected_design_sha256=expected_campaign_design_sha,
        actual_design_sha256=actual_design_sha,
        approval_design_sha256=approval_design_sha,
        executable_design_chain_passed=executable_design_chain_passed,
        campaign_design_path_matches=campaign_path_matches,
        expected_join_busy_pool=expected_join_busy_pool,
        approval_join_busy_pool=approval_join_busy_pool,
        expected_gpu_ids=expected_gpu_ids,
        actual_gpu_ids=actual_gpu_ids,
        queue_binding_sha256_valid=_sha256_text(queue_binding),
        runtime_fingerprint_sha256_valid=_sha256_text(runtime_fingerprint),
        approval_present=True,
    )


def _fingerprint_check(design: Mapping[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """Verify that frozen source fingerprints still match live files.

    Presence of a recorded SHA alone is not enough: a source file can change
    after the design was generated.  Fresh campaigns therefore fail closed on
    missing paths or mismatched bytes.  A legacy non-file binding is reported
    as unverified only when it has no path at all; current holdout designs do
    not use that compatibility form.
    """

    bindings = design.get("frozen_bindings") or {}
    if not bindings:
        implementation = design.get("design_implementation") or {}
        bindings = {"design_implementation": implementation} if implementation else {}
    missing: list[str] = []
    mismatched: list[dict[str, Any]] = []
    unverified: list[str] = []
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping) or not binding.get("sha256"):
            missing.append(str(name))
            continue
        raw_path = binding.get("path")
        if not raw_path:
            # Keep compatibility for old abstract design implementations;
            # new campaign builders always provide a concrete path.
            unverified.append(str(name))
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            missing.append(f"{name}:path_missing")
            continue
        expected = str(binding.get("sha256"))
        actual = sha256_file(path)
        if actual != expected:
            mismatched.append(
                {
                    "name": str(name),
                    "path": str(path.resolve()),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                }
            )
    passed = not missing and not mismatched
    return _check(
        "source_fingerprints",
        passed,
        "all design source bindings carry SHA-256 fingerprints"
        if passed and not unverified
        else "all resolvable design source bindings match their SHA-256 fingerprints"
        if passed
        else "one or more design source files are missing or changed",
        binding_count=len(bindings),
        missing=missing,
        mismatched=mismatched,
        unverified=unverified,
    )


def build_gate_report(
    *,
    design: Mapping[str, Any],
    design_path: Path,
    approval: Mapping[str, Any] | None,
    hardware_probe: Mapping[str, Any],
    fresh_profile_requirements: Mapping[str, Any] | None = None,
    fresh_profile_requirements_path: Path | None = None,
    approval_design: Mapping[str, Any] | None = None,
    approval_design_path: Path | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Build a deterministic, non-executable readiness report."""

    required = design.get("required_gpu_pool") or design.get("hardware_gate") or {}
    required_gpu_ids = tuple(int(value) for value in required.get("gpu_ids") or required.get("required_gpu_ids") or ())
    expected_name = str(
        required.get("expected_name_contains")
        or required.get("expected_gpu_name_contains")
        or "H800"
    )
    hardware_passed = (
        hardware_probe.get("exact_h800_pool") is True
        and (
            hardware_probe.get("selected_pool_idle") is True
            or required.get("join_busy_pool_allowed") is True
        )
        and sorted(int(value) for value in hardware_probe.get("required_gpu_ids") or ())
        == sorted(required_gpu_ids)
    )
    checks = [
        _check(
            "design_integrity",
            design.get("gpu_training_started") is False
            and design.get("queues_mutated") is False
            and bool(design.get("campaign_id")),
            "design is a named, non-executable manifest"
            if design.get("gpu_training_started") is False
            and design.get("queues_mutated") is False
            and design.get("campaign_id")
            else "design is missing non-executable campaign invariants",
        ),
        _check(
            "hardware_pool",
            hardware_passed,
            (
                "exact idle H800 pool is present"
                if hardware_probe.get("selected_pool_idle") is True
                else "exact H800 pool is present; busy devices remain blocked until idle"
            )
            if hardware_passed
            else "required idle H800 pool is unavailable or does not match the design",
            required_gpu_ids=list(required_gpu_ids),
            expected_name_contains=expected_name,
            join_busy_pool_allowed=required.get("join_busy_pool_allowed") is True,
            probe=dict(hardware_probe),
        ),
        _requirements_check(
            design,
            design_path,
            fresh_profile_requirements,
            fresh_profile_requirements_path,
            root,
        ),
        _profile_check(design, root, fresh_profile_requirements),
        _fingerprint_check(design, root),
        _approval_check(
            design,
            design_path,
            approval,
            required_gpu_ids,
            approval_design=approval_design,
            approval_design_path=approval_design_path,
            root=root,
        ),
    ]
    all_passed = all(check["passed"] for check in checks)
    return {
        "schema": SCHEMA,
        "campaign_id": design.get("campaign_id"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": {"path": str(design_path.resolve()), "sha256": sha256_file(design_path)},
        "fresh_profile_requirements": (
            {
                "path": str(fresh_profile_requirements_path.resolve()),
                "sha256": sha256_file(fresh_profile_requirements_path),
            }
            if fresh_profile_requirements_path
            and fresh_profile_requirements_path.is_file()
            else None
        ),
        "checks": checks,
        "launch_allowed": False,
        "materialization_allowed": False,
        "all_prerequisites_passed": all_passed,
        "next_action": (
            "create a separately approved queue and run existing materializer"
            if all_passed
            else "resolve failed checks; no queue may be materialized"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument(
        "--approval-design", type=Path, default=DEFAULT_APPROVAL_DESIGN
    )
    parser.add_argument(
        "--profile-requirements",
        type=Path,
        default=DEFAULT_PROFILE_REQUIREMENTS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true", help="return exit code 2 when any gate fails")
    args = parser.parse_args()
    design = read_json(args.design)
    approval = read_json(args.approval) if args.approval.is_file() else None
    approval_design = (
        read_json(args.approval_design)
        if args.approval_design.is_file()
        else None
    )
    profile_requirements = (
        read_json(args.profile_requirements)
        if args.profile_requirements.is_file()
        else None
    )
    required = design.get("required_gpu_pool") or design.get("hardware_gate") or {}
    required_ids = tuple(int(value) for value in required.get("gpu_ids") or required.get("required_gpu_ids") or (4, 5, 6, 7))
    expected_name = str(required.get("expected_name_contains") or required.get("expected_gpu_name_contains") or "H800")
    report = build_gate_report(
        design=design,
        design_path=args.design,
        approval=approval,
        approval_design=approval_design,
        approval_design_path=args.approval_design,
        hardware_probe=probe_hardware(required_gpu_ids=required_ids, expected_gpu_name=expected_name),
        fresh_profile_requirements=profile_requirements,
        fresh_profile_requirements_path=args.profile_requirements,
    )
    write_json(args.output, report)
    print(f"wrote {args.output}; all_prerequisites_passed={report['all_prerequisites_passed']}")
    if args.strict and not report["all_prerequisites_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
