#!/usr/bin/env python3
"""Shared fail-closed primitives for approval verification and promotion.

The execution side holds a shared lock for the lifetime of a scheduler or job.
Approval promotion holds the same lock exclusively, so an approval/design pair
cannot be switched underneath a participating launcher.
"""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, IO, Iterator

from common import read_json, read_jsonl, sha256_file, sha256_json, verify_file_manifest


LOCK_FILE_NAME = "approval_execution_gate.lock"
RETIRED_APPROVAL_PLAN_SCHEMAS = frozenset(
    {
        "sft_h800_calibration_candidate/v1",
        "sft_h800_calibration_approval_freeze/v1",
    }
)


def retired_approval_plan_schemas(document: Any) -> list[str]:
    """Return retired H800 plan schemas declared by a candidate/design.

    The original source candidate declares its schema at the top level.  The
    approval-design freezer wrapped its schema under ``h800_calibration``.
    Keeping both locations in one fail-closed helper prevents the historical
    109-row plan from being regenerated, promoted, or executed under a changed
    file hash.
    """

    if not isinstance(document, dict):
        return []
    declared = {document.get("schema")}
    h800_calibration = document.get("h800_calibration")
    if isinstance(h800_calibration, dict):
        declared.add(h800_calibration.get("schema"))
    return sorted(
        schema
        for schema in declared
        if isinstance(schema, str) and schema in RETIRED_APPROVAL_PLAN_SCHEMAS
    )


def reject_retired_approval_plan(document: Any, *, operation: str) -> None:
    """Permanently reject an operation on the retired 109-row H800 plan."""

    retired = retired_approval_plan_schemas(document)
    if retired:
        raise PermissionError(
            "Retired H800 calibration plan schema(s) "
            f"{retired} may not be used to {operation}. These files are "
            "historical audit artifacts only; any future gap-specific experiment "
            "must use a new schema and a new explicit approval."
        )


def canonical_job_sha256(job: dict[str, Any]) -> str:
    """Return the authorization hash for one canonical JSON job payload."""

    return sha256_json(job)


def strict_id_list(value: Any) -> list[str] | None:
    """Return a non-empty, unique list of non-empty string IDs, or ``None``."""

    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(item, str) and item for item in value):
        return None
    return value if len(value) == len(set(value)) else None


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def resolve_project_path(value: Any, project_root: Path) -> Path | None:
    """Resolve a declared project-relative path without accepting escapes."""

    if not isinstance(value, str) or not value:
        return None
    declared = Path(value)
    candidate = (
        declared.resolve()
        if declared.is_absolute()
        else (project_root / declared).resolve()
    )
    return candidate if is_within(candidate, project_root) else None


def project_relative(path: Path, project_root: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve()))


def expected_provenance_source_paths(project_root: Path) -> list[Path]:
    """Mirror ``capture_provenance.source_manifest``'s complete path set."""

    root = project_root.resolve()
    paths = list((root / "scripts").glob("*.py"))
    paths.extend(
        path
        for path in (root / "config").rglob("*")
        if path.is_file() and "APPROVED" not in path.name
    )
    paths.extend((root / "README.md", root / "EXPERIMENT_DESIGN.md"))
    return sorted(set(path.resolve() for path in paths))


def validate_provenance_binding(
    design: dict[str, Any], project_root: Path
) -> dict[str, Any]:
    """Revalidate provenance plus either the full or approval-scoped source set.

    The legacy/default policy continues to bind every project script and config.
    Long-running campaigns may opt into ``approval_file_manifest_scoped_v1``;
    that policy is still fail-closed, but only for the explicit execution-source
    subset that is also hash-bound by the approval's file manifest.  This keeps
    unrelated offline modeling edits from invalidating an active GPU queue.
    """

    root = project_root.resolve()
    binding = design.get("provenance_binding") or {}
    path = resolve_project_path(binding.get("path"), root)
    policy = binding.get(
        "source_validation_policy", "complete_project_source_snapshot_v1"
    )
    provenance: dict[str, Any] = {}
    read_error: str | None = None
    if path is not None and path.is_file():
        try:
            value = read_json(path)
            if isinstance(value, dict):
                provenance = value
            else:
                read_error = "provenance JSON is not an object"
        except (OSError, ValueError, json.JSONDecodeError) as error:
            read_error = repr(error)
    else:
        read_error = "bound provenance path is absent or outside the project"

    global_source_manifest = provenance.get("project_source_manifest") or {}
    scoped_policy = policy == "approval_file_manifest_scoped_v1"
    if scoped_policy:
        declared_scoped_manifest = binding.get("scoped_source_manifest") or {}
        source_manifest = (
            declared_scoped_manifest
            if isinstance(declared_scoped_manifest, dict)
            else {}
        )
        expected_keys = sorted(source_manifest)
        expected_paths = [resolve_project_path(key, root) for key in expected_keys]
        scoped_paths_inside_project = all(path is not None for path in expected_paths)
        current_manifest = {
            key: sha256_file(path)
            for key, path in zip(expected_keys, expected_paths, strict=True)
            if path is not None and path.is_file()
        }
    else:
        source_manifest = global_source_manifest
        expected_paths = expected_provenance_source_paths(root)
        expected_keys = [project_relative(item, root) for item in expected_paths]
        scoped_paths_inside_project = True
        current_manifest = {
            project_relative(item, root): sha256_file(item)
            for item in expected_paths
            if item.is_file()
        }
    runtime_identity = provenance.get("runtime_identity") or {}
    design_manifest = design.get("file_sha256") or {}
    relative = project_relative(path, root) if path is not None else None
    actual_sha256 = sha256_file(path) if path is not None and path.is_file() else None
    manifest_check = (
        verify_file_manifest(root, source_manifest)
        if isinstance(source_manifest, dict) and source_manifest
        else {"entries": 0, "missing": [], "mismatched": [], "all_passed": False}
    )
    checks = {
        "binding_present": isinstance(binding, dict) and bool(binding),
        "source_validation_policy_supported": policy
        in {
            "complete_project_source_snapshot_v1",
            "approval_file_manifest_scoped_v1",
        },
        "path_inside_project": path is not None,
        "provenance_readable": read_error is None and bool(provenance),
        "file_sha256_matches": actual_sha256 is not None
        and binding.get("sha256") == actual_sha256,
        "design_manifest_binds_provenance": relative is not None
        and design_manifest.get(relative) == actual_sha256,
        "source_path_set_exact": bool(source_manifest)
        and scoped_paths_inside_project
        and sorted(source_manifest) == expected_keys
        and binding.get("project_source_paths") == expected_keys,
        "source_manifest_current": source_manifest == current_manifest
        and manifest_check.get("all_passed") is True,
        "scoped_source_manifest_design_bound": not scoped_policy
        or all(design_manifest.get(key) == value for key, value in source_manifest.items()),
        "source_manifest_hash_bound": bool(source_manifest)
        and binding.get("project_source_manifest_sha256")
        == sha256_json(source_manifest),
        "runtime_fingerprint_internal": bool(runtime_identity)
        and provenance.get("runtime_fingerprint_sha256")
        == sha256_json(runtime_identity),
        "runtime_fingerprint_bound": bool(runtime_identity)
        and binding.get("runtime_fingerprint_sha256")
        == provenance.get("runtime_fingerprint_sha256"),
        "source_snapshot_internal": bool(global_source_manifest)
        and runtime_identity.get("project_source_snapshot_sha256")
        == sha256_json(global_source_manifest),
        "reproducibility_identity_complete": provenance.get(
            "reproducibility_identity_complete"
        )
        is True,
    }
    return {
        "path": str(path) if path else None,
        "actual_sha256": actual_sha256,
        "expected_source_paths": expected_keys,
        "read_error": read_error,
        "manifest_check": manifest_check,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def build_provenance_binding(
    project_root: Path, *, source_paths: list[Path] | None = None
) -> dict[str, Any]:
    """Build authorization metadata from a fresh full or scoped snapshot."""

    root = project_root.resolve()
    path = root / "artifacts" / "provenance.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing provenance snapshot: {path}")
    provenance = read_json(path)
    global_source_manifest = provenance.get("project_source_manifest") or {}
    if source_paths is None:
        source_manifest = global_source_manifest
    else:
        resolved = sorted(set(path.resolve() for path in source_paths), key=str)
        outside = [str(path) for path in resolved if not is_within(path, root)]
        missing = [str(path) for path in resolved if not path.is_file()]
        if outside or missing or not resolved:
            raise ValueError(
                "Scoped provenance paths must be non-empty existing project files: "
                f"outside={outside}, missing={missing}"
            )
        source_manifest = {
            project_relative(path, root): sha256_file(path) for path in resolved
        }
    binding = {
        "path": project_relative(path, root),
        "sha256": sha256_file(path),
        "project_source_paths": sorted(source_manifest),
        "project_source_manifest_sha256": sha256_json(source_manifest),
        "runtime_fingerprint_sha256": provenance.get("runtime_fingerprint_sha256"),
    }
    probe_manifest = {binding["path"]: binding["sha256"]}
    if source_paths is not None:
        binding["source_validation_policy"] = "approval_file_manifest_scoped_v1"
        binding["scoped_source_manifest"] = source_manifest
        probe_manifest.update(source_manifest)
    probe = dict(provenance_binding=binding, file_sha256=probe_manifest)
    validation = validate_provenance_binding(probe, root)
    if validation.get("all_passed") is not True:
        raise ValueError(f"Provenance is stale or incomplete: {validation}")
    return binding


def build_queue_binding(
    queue_path: Path, rows: list[dict[str, Any]], project_root: Path
) -> dict[str, Any]:
    """Build a complete ordered queue authorization binding."""

    root = project_root.resolve()
    path = queue_path.resolve()
    if not is_within(path, root) or not path.is_file():
        raise ValueError(f"Queue must be an existing file inside {root}: {path}")
    job_ids = [str(row.get("job_id") or "") for row in rows]
    if strict_id_list(job_ids) is None:
        raise ValueError("Queue job IDs must be non-empty and unique")
    payload_hashes = [canonical_job_sha256(row) for row in rows]
    return {
        "schema_version": 1,
        "path": project_relative(path, root),
        "sha256": sha256_file(path),
        "ordered_job_ids": job_ids,
        "ordered_job_payload_sha256": payload_hashes,
        "job_payload_sha256": dict(zip(job_ids, payload_hashes, strict=True)),
    }


def validate_queue_binding(
    design: dict[str, Any],
    project_root: Path,
    *,
    queue_path: Path | None = None,
    queue_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate the bound queue bytes, order and complete canonical payload set."""

    root = project_root.resolve()
    binding = design.get("queue_binding") or {}
    bound_path = resolve_project_path(binding.get("path"), root)
    rows: list[dict[str, Any]] = []
    read_error: str | None = None
    if bound_path is not None and bound_path.is_file():
        try:
            rows = read_jsonl(bound_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            read_error = repr(error)
    else:
        read_error = "bound queue path is absent or outside the project"

    ids = [str(row.get("job_id") or "") for row in rows]
    hashes = [canonical_job_sha256(row) for row in rows]
    hash_by_id = (
        dict(zip(ids, hashes, strict=True)) if strict_id_list(ids) is not None else {}
    )
    allowed = strict_id_list(design.get("allowed_job_ids"))
    design_manifest = design.get("file_sha256") or {}
    relative = project_relative(bound_path, root) if bound_path is not None else None
    actual_sha256 = (
        sha256_file(bound_path)
        if bound_path is not None and bound_path.is_file()
        else None
    )
    supplied_hashes = (
        [canonical_job_sha256(row) for row in queue_rows]
        if queue_rows is not None
        else None
    )
    checks = {
        "binding_present": isinstance(binding, dict) and bool(binding),
        "schema_supported": binding.get("schema_version") == 1,
        "path_inside_project": bound_path is not None,
        "queue_readable_and_nonempty": read_error is None and bool(rows),
        "queue_sha256_matches": actual_sha256 is not None
        and binding.get("sha256") == actual_sha256,
        "design_manifest_binds_queue": relative is not None
        and design_manifest.get(relative) == actual_sha256,
        "queue_job_ids_unique": strict_id_list(ids) is not None,
        "allowed_job_ids_exact": allowed is not None and allowed == ids,
        "ordered_job_ids_exact": binding.get("ordered_job_ids") == ids,
        "ordered_payload_hashes_exact": binding.get("ordered_job_payload_sha256")
        == hashes,
        "payload_hash_map_exact": binding.get("job_payload_sha256") == hash_by_id,
        "requested_queue_path_exact": queue_path is None
        or (bound_path is not None and queue_path.resolve() == bound_path),
        "requested_queue_rows_exact": queue_rows is None or supplied_hashes == hashes,
    }
    return {
        "path": str(bound_path) if bound_path else None,
        "actual_sha256": actual_sha256,
        "job_ids": ids,
        "job_payload_sha256": hash_by_id,
        "read_error": read_error,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def acquire_execution_lock(
    runtime_dir: Path,
    *,
    exclusive: bool,
    blocking: bool = False,
) -> IO[str]:
    """Acquire and return the process-lifetime approval execution lock."""

    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / LOCK_FILE_NAME
    handle = path.open("a+", encoding="utf-8")
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if not blocking:
        operation |= fcntl.LOCK_NB
    try:
        fcntl.flock(handle.fileno(), operation)
    except BlockingIOError as error:
        handle.close()
        actor = "approval promotion" if exclusive else "experiment execution"
        raise PermissionError(
            f"Cannot enter {actor}: the approval execution gate is busy ({path})"
        ) from error
    return handle


@contextmanager
def execution_lock(
    runtime_dir: Path,
    *,
    exclusive: bool,
    blocking: bool = False,
) -> Iterator[IO[str]]:
    handle = acquire_execution_lock(
        runtime_dir,
        exclusive=exclusive,
        blocking=blocking,
    )
    try:
        yield handle
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
