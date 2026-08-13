#!/usr/bin/env python3
"""Historical transformer for the retired H800 conditional approval candidate.

The pure transformation helpers remain for audit and regression tests.  The
v1 109-row plan was retired on 2026-07-22, so ``freeze_candidate`` now fails
before writing either the offline queue or approval-design candidate.  It never
starts a GPU process.

The source rows deliberately contain ``execution_authorized=false``.  Copying
that flag into an execution queue would be semantically misleading, while
flipping it to true before approval would be unsafe.  The sole transformation
therefore removes the flag and adds a fail-closed ``execution_gate`` binding;
the existing approval gate remains the only authority that can launch a row.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from approval_gate import (
    build_provenance_binding,
    expected_provenance_source_paths,
    project_relative,
    reject_retired_approval_plan,
    validate_provenance_binding,
    validate_queue_binding,
)
from common import ROOT, read_json, read_jsonl, sha256_file, sha256_json
from freeze_lora_zero3_fix import validate_patch
from prepare_h800_calibration_candidate import (
    DESIGN_NAME as SOURCE_DESIGN_NAME,
    JOBS_NAME as SOURCE_JOBS_NAME,
    validate_candidate as validate_source_candidate,
)
from run_job import live_runtime_identity
from scheduler import AdaptiveQueueError, validate_adaptive_queue


SCHEMA = "sft_h800_calibration_approval_freeze/v1"
EXECUTION_GATE_SCHEMA = "sft_h800_calibration_execution_gate/v1"
TRANSFORM_POLICY = "remove_false_authorization_add_exact_source_binding/v1"
DEFAULT_SOURCE_DESIGN = ROOT / "artifacts" / "candidates" / SOURCE_DESIGN_NAME
DEFAULT_SOURCE_JOBS = ROOT / "artifacts" / "candidates" / SOURCE_JOBS_NAME
DEFAULT_QUEUE = (
    ROOT / "artifacts" / "candidates" / "h800_calibration_execution_queue.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "candidates"
    / "h800_calibration_approval_design.candidate.json"
)
LIVE_PATHS = (
    "config/APPROVED_TO_RUN.json",
    "runtime/approval_design.json",
    "runtime/queue.json",
)
EXPECTED_GPU_IDS = [1, 2, 3, 4]
EXPECTED_GPU_TYPE = "NVIDIA H800 140GB HBM3"
PER_RUN_TIMEOUT_SECONDS = 45 * 60
CAMPAIGN_HARD_WALL_TIME_SECONDS = 72 * 60 * 60


def _canonical_jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _safe_offline_path(
    path: Path,
    project_root: Path,
    *,
    protected: Iterable[Path] = (),
) -> Path:
    root = project_root.resolve()
    output_root = (root / "artifacts" / "candidates").resolve()
    resolved = path.resolve()
    forbidden = {(root / relative).resolve() for relative in LIVE_PATHS}
    forbidden.update(item.resolve() for item in protected)
    if (
        not _is_within(resolved, output_root)
        or resolved in forbidden
        or resolved.suffix not in {".json", ".jsonl"}
    ):
        raise ValueError(
            f"Offline H800 outputs must be distinct JSON/JSONL files under {output_root}"
        )
    return resolved


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_execution_jobs(
    source_design: dict[str, Any], source_jobs: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Apply the one declared source-to-execution transformation."""

    canonical = (source_design.get("canonical_identity") or {}).get("sha256")
    source_jobs_sha256 = (source_design.get("jobs_binding") or {}).get("sha256")
    if not isinstance(canonical, str) or not isinstance(source_jobs_sha256, str):
        raise ValueError("Source candidate lacks canonical jobs/design bindings")
    boundary_timeout = (
        ((source_design.get("measurement") or {}).get("boundary") or {}).get(
            "per_run_timeout_seconds"
        )
    )
    packing_timeout = (
        ((source_design.get("measurement") or {}).get("packing_pair") or {}).get(
            "per_run_timeout_seconds"
        )
    )
    hard_wall_hours = (source_design.get("budget") or {}).get(
        "hard_wall_time_stop_hours"
    )
    if (
        boundary_timeout != PER_RUN_TIMEOUT_SECONDS
        or packing_timeout != PER_RUN_TIMEOUT_SECONDS
        or hard_wall_hours != 72.0
    ):
        raise ValueError("Source candidate runtime/campaign budgets are not exact")
    execution_rows: list[dict[str, Any]] = []
    mapping: list[dict[str, str]] = []
    for source in source_jobs:
        if source.get("execution_authorized") is not False:
            raise ValueError(
                f"Source job {source.get('job_id')} is not explicitly unapproved"
            )
        source_sha256 = sha256_json(source)
        row = copy.deepcopy(source)
        del row["execution_authorized"]
        row["execution_gate"] = {
            "schema": EXECUTION_GATE_SCHEMA,
            "authorization_state": "requires_exact_promoted_approval_design",
            "source_candidate_canonical_sha256": canonical,
            "source_jobs_sha256": source_jobs_sha256,
            "source_job_payload_sha256": source_sha256,
            "transform_policy": TRANSFORM_POLICY,
            "per_run_timeout_seconds": PER_RUN_TIMEOUT_SECONDS,
            "campaign_hard_wall_time_seconds": CAMPAIGN_HARD_WALL_TIME_SECONDS,
        }
        execution_sha256 = sha256_json(row)
        execution_rows.append(row)
        mapping.append(
            {
                "job_id": str(row["job_id"]),
                "source_job_payload_sha256": source_sha256,
                "execution_job_payload_sha256": execution_sha256,
            }
        )
    return execution_rows, mapping


def build_queue_binding_offline(
    queue_path: Path,
    rows: list[dict[str, Any]],
    project_root: Path,
) -> dict[str, Any]:
    """Build the existing queue-binding schema from not-yet-written bytes."""

    root = project_root.resolve()
    path = queue_path.resolve()
    if not _is_within(path, root):
        raise ValueError("Execution queue path escapes the campaign root")
    ids = [str(row.get("job_id") or "") for row in rows]
    if not ids or any(not job_id for job_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Execution queue IDs must be non-empty and unique")
    payload_hashes = [sha256_json(row) for row in rows]
    return {
        "schema_version": 1,
        "path": project_relative(path, root),
        "sha256": _sha256_bytes(_canonical_jsonl_bytes(rows)),
        "ordered_job_ids": ids,
        "ordered_job_payload_sha256": payload_hashes,
        "job_payload_sha256": dict(zip(ids, payload_hashes, strict=True)),
    }


def _approval_manifest(
    project_root: Path,
    paths: Iterable[Path],
) -> dict[str, str]:
    root = project_root.resolve()
    required = set(expected_provenance_source_paths(root))
    required.update(path.resolve() for path in paths)
    outside = [str(path) for path in required if not _is_within(path, root)]
    missing = [str(path) for path in required if not path.is_file()]
    if outside or missing:
        raise ValueError(
            f"Approval manifest has outside/missing inputs: outside={outside}, missing={missing}"
        )
    return {
        project_relative(path, root): sha256_file(path)
        for path in sorted(required)
    }


def validate_materialization(
    *,
    source_design: dict[str, Any],
    source_jobs: list[dict[str, Any]],
    execution_jobs: list[dict[str, Any]],
    transform_mapping: list[dict[str, str]],
    project_root: Path,
) -> dict[str, Any]:
    """Recompute every source, transformation, H800 and DAG invariant."""

    checks: dict[str, bool] = {}
    errors: list[str] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks[name] = bool(passed)
        if not passed:
            errors.append(detail)

    source_validation = validate_source_candidate(
        source_design, source_jobs, project_root
    )
    check(
        "source_candidate_valid",
        source_validation.get("all_passed") is True,
        f"source candidate validation failed: {source_validation}",
    )
    source_ids = [row.get("job_id") for row in source_jobs]
    execution_ids = [row.get("job_id") for row in execution_jobs]
    check(
        "order_and_ids_preserved",
        source_ids == execution_ids
        and len(execution_ids) == len(set(execution_ids)),
        "source/execution order or IDs differ",
    )
    expected_rows, expected_mapping = materialize_execution_jobs(
        source_design, source_jobs
    )
    check(
        "transform_exact",
        execution_jobs == expected_rows and transform_mapping == expected_mapping,
        "execution rows do not match the declared one-step transform",
    )
    check(
        "false_authorization_removed",
        all("execution_authorized" not in row for row in execution_jobs),
        "execution queue retained the source false authorization flag",
    )
    check(
        "execution_gate_present",
        all(
            (row.get("execution_gate") or {}).get("authorization_state")
            == "requires_exact_promoted_approval_design"
            for row in execution_jobs
        ),
        "execution queue lacks its fail-closed approval requirement",
    )
    identity = json.dumps(execution_jobs, ensure_ascii=False, sort_keys=True).lower()
    check(
        "h800_only",
        "4090" not in identity
        and all(row.get("gpu_type") == EXPECTED_GPU_TYPE for row in execution_jobs),
        "execution queue contains a non-H800 identity",
    )
    adaptive_error: str | None = None
    try:
        adaptive_plan = validate_adaptive_queue(
            execution_jobs, require_execution_gate=True
        )
    except (AdaptiveQueueError, ValueError, TypeError, KeyError) as error:
        adaptive_plan = None
        adaptive_error = repr(error)
    check(
        "conditional_dag_valid",
        adaptive_plan is not None and adaptive_error is None,
        f"adaptive scheduler rejected the queue: {adaptive_error}",
    )
    return {
        "schema": SCHEMA,
        "checks": checks,
        "errors": errors,
        "source_validation": source_validation,
        "all_passed": all(checks.values()),
    }


def build_approval_design(
    *,
    project_root: Path,
    source_design_path: Path,
    source_jobs_path: Path,
    source_design: dict[str, Any],
    source_jobs: list[dict[str, Any]],
    execution_queue_path: Path,
    execution_jobs: list[dict[str, Any]],
    transform_mapping: list[dict[str, str]],
    runtime_identity: dict[str, Any],
    runtime_patch: dict[str, Any],
    provenance_binding: dict[str, Any],
) -> dict[str, Any]:
    root = project_root.resolve()
    experiment = read_json(root / "config" / "experiment.json")
    scope = experiment.get("training_scope") or {}
    if (
        scope.get("gpu_ids") != EXPECTED_GPU_IDS
        or scope.get("max_gpu_count") != 4
        or scope.get("gpu_type") != EXPECTED_GPU_TYPE
        or set(scope.get("gpu_counts") or ()) != {1, 2, 4}
    ):
        raise ValueError("Live campaign scope is not the exact four-card H800 scope")
    if not runtime_identity or runtime_patch.get("all_passed") is not True:
        raise ValueError("Runtime identity/patch evidence is absent or unhealthy")

    materialization = validate_materialization(
        source_design=source_design,
        source_jobs=source_jobs,
        execution_jobs=execution_jobs,
        transform_mapping=transform_mapping,
        project_root=root,
    )
    if materialization.get("all_passed") is not True:
        raise ValueError(f"H800 materialization validation failed: {materialization}")
    queue_binding = build_queue_binding_offline(
        execution_queue_path, execution_jobs, root
    )
    allowed_ids = list(queue_binding["ordered_job_ids"])
    mapping_hash = sha256_json(transform_mapping)
    provenance_path = root / str(provenance_binding.get("path") or "")
    manifest = _approval_manifest(
        root,
        (
            source_design_path,
            source_jobs_path,
            execution_queue_path,
            provenance_path,
            root / "artifacts" / "model_inventory.json",
            root / "artifacts" / "dataset_analysis.json",
            root / "artifacts" / "h800_calibration_readiness.json",
        ),
    )
    runtime_fingerprint = sha256_json(runtime_identity)
    source_canonical = source_design["canonical_identity"]["sha256"]
    source_binding = source_design["jobs_binding"]
    stage = {
        "jobs": len(allowed_ids),
        "allowed_job_ids": allowed_ids,
        "queue_path": queue_binding["path"],
        "queue_sha256": queue_binding["sha256"],
        "ordered_job_ids": queue_binding["ordered_job_ids"],
        "ordered_job_payload_sha256": queue_binding[
            "ordered_job_payload_sha256"
        ],
        "job_payload_sha256": queue_binding["job_payload_sha256"],
        "policy": (
            "Execute only through scheduler.py's validated H800 conditional DAG; "
            "success/OOM are terminal probe outcomes, conditional_skipped never "
            "becomes a training observation, and every other outcome blocks retryably."
        ),
    }
    return {
        "schema_version": 1,
        "training_started": False,
        "file_sha256": manifest,
        "approval_instruction": (
            "Approval must bind the SHA256 bytes of this exact file. Promotion remains "
            "a separate explicit transaction and must revalidate every binding."
        ),
        "design_purpose": "H800-only theory-planner calibration and holdout rerun",
        "execution_order": ["h800_calibration_conditional"],
        "allowed_job_ids": allowed_ids,
        "authorized_gpu_ids": EXPECTED_GPU_IDS,
        "max_gpu_count": 4,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": runtime_fingerprint,
        "runtime_patch": runtime_patch,
        "queue_binding": queue_binding,
        "provenance_binding": provenance_binding,
        # Existing promotion code validates this compatibility field exactly.
        "throughput_screen_delta": stage,
        "h800_calibration": {
            "schema": SCHEMA,
            "gpu_family": "H800",
            "gpu_type": EXPECTED_GPU_TYPE,
            "forbidden_gpu_identity_patterns": ["4090"],
            "partial_other_gpu_results_allowed": False,
            "source_candidate": {
                "path": project_relative(source_design_path, root),
                "file_sha256": sha256_file(source_design_path),
                "canonical_sha256": source_canonical,
            },
            "source_jobs": {
                "path": project_relative(source_jobs_path, root),
                "sha256": sha256_file(source_jobs_path),
                "canonical_jsonl_sha256": source_binding["sha256"],
                "ordered_job_ids": source_binding["ordered_job_ids"],
                "ordered_job_payload_sha256": source_binding[
                    "ordered_job_payload_sha256"
                ],
            },
            "transformation": {
                "policy": TRANSFORM_POLICY,
                "rows": len(transform_mapping),
                "mapping_sha256": mapping_hash,
                "ordered_mapping": transform_mapping,
                "source_false_authorization_was_removed": True,
                "authorization_was_not_granted_by_materialization": True,
            },
            "conditional_runtime": {
                "scheduler_schema": "h800_adaptive_scheduler_plan/v1",
                "scheduler_source_sha256": manifest["scripts/scheduler.py"],
                "per_run_timeout_seconds": PER_RUN_TIMEOUT_SECONDS,
                "process_group_timeout_termination": "SIGTERM_then_SIGKILL",
                "campaign_hard_wall_time_seconds": CAMPAIGN_HARD_WALL_TIME_SECONDS,
                "campaign_deadline_anchor": (
                    "config/APPROVED_TO_RUN.json approved_unix plus the exact hard-wall budget; "
                    "scheduler restarts do not reset it"
                ),
                "campaign_deadline_stops_new_launches": True,
                "campaign_deadline_cancels_running_process_groups": True,
                "first_upward_oom_stops_family": True,
                "mbs16_success_caps_supported_domain": True,
                "packing_order": "ABBA",
                "packing_rows_require_all_prior_successes": True,
                "software_or_infrastructure_failure_is_non_terminal": True,
                "conditional_skip_is_not_calibration_evidence": True,
            },
            "materialization_validation": materialization,
        },
    }


def freeze_candidate(
    *,
    project_root: Path,
    source_design_path: Path,
    source_jobs_path: Path,
    queue_path: Path,
    output_path: Path,
    runtime_identity: dict[str, Any] | None = None,
    runtime_patch: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    # This freezer can only emit SCHEMA.  Refuse before reading inputs, probing
    # the runtime, refreshing provenance, or writing the offline queue.
    reject_retired_approval_plan(
        {"h800_calibration": {"schema": SCHEMA}},
        operation="freeze an execution queue or approval candidate",
    )
    root = project_root.resolve()
    source_design_path = source_design_path.resolve()
    source_jobs_path = source_jobs_path.resolve()
    if not source_design_path.is_file() or not source_jobs_path.is_file():
        raise FileNotFoundError("H800 source candidate design/jobs are absent")
    queue_path = _safe_offline_path(
        queue_path, root, protected=(source_design_path, source_jobs_path)
    )
    output_path = _safe_offline_path(
        output_path,
        root,
        protected=(source_design_path, source_jobs_path, queue_path),
    )
    if not overwrite and (queue_path.exists() or output_path.exists()):
        raise FileExistsError(
            "Refusing to overwrite an offline queue/design without --overwrite"
        )

    source_design = read_json(source_design_path)
    source_jobs = read_jsonl(source_jobs_path)
    execution_jobs, mapping = materialize_execution_jobs(source_design, source_jobs)
    materialization = validate_materialization(
        source_design=source_design,
        source_jobs=source_jobs,
        execution_jobs=execution_jobs,
        transform_mapping=mapping,
        project_root=root,
    )
    if materialization.get("all_passed") is not True:
        raise ValueError(f"Source candidate/materialization failed: {materialization}")

    # Provenance is checked before any output write.  Source edits require a new
    # explicit capture; this freezer never silently refreshes reproducibility state.
    provenance_binding = build_provenance_binding(root)
    live_identity = (
        live_runtime_identity() if runtime_identity is None else runtime_identity
    )
    live_patch = validate_patch() if runtime_patch is None else runtime_patch

    queue_bytes = _canonical_jsonl_bytes(execution_jobs)
    _atomic_write(queue_path, queue_bytes)
    try:
        design = build_approval_design(
            project_root=root,
            source_design_path=source_design_path,
            source_jobs_path=source_jobs_path,
            source_design=source_design,
            source_jobs=source_jobs,
            execution_queue_path=queue_path,
            execution_jobs=execution_jobs,
            transform_mapping=mapping,
            runtime_identity=live_identity,
            runtime_patch=live_patch,
            provenance_binding=provenance_binding,
        )
        queue_validation = validate_queue_binding(design, root)
        provenance_validation = validate_provenance_binding(design, root)
        if (
            queue_validation.get("all_passed") is not True
            or provenance_validation.get("all_passed") is not True
        ):
            raise ValueError(
                "Frozen design failed queue/provenance revalidation: "
                f"queue={queue_validation}, provenance={provenance_validation}"
            )
        design_bytes = (
            json.dumps(design, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        _atomic_write(output_path, design_bytes)
    except BaseException:
        # Never leave a newly materialized queue that appears paired with a
        # missing candidate.  Existing outputs are only replaced with --overwrite;
        # in that mode both are offline and the exact next invocation can rebuild.
        queue_path.unlink(missing_ok=True)
        raise

    return {
        "all_passed": True,
        "training_started": False,
        "approval_file_written": False,
        "live_approval_design_written": False,
        "live_queue_written": False,
        "gpu_process_started": False,
        "source_candidate_canonical_sha256": source_design["canonical_identity"][
            "sha256"
        ],
        "source_jobs_sha256": source_design["jobs_binding"]["sha256"],
        "execution_queue": {
            "path": str(queue_path),
            "sha256": sha256_file(queue_path),
            "rows": len(execution_jobs),
        },
        "approval_design_candidate": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-design", type=Path, default=DEFAULT_SOURCE_DESIGN)
    parser.add_argument("--source-jobs", type=Path, default=DEFAULT_SOURCE_JOBS)
    parser.add_argument("--queue-output", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = freeze_candidate(
        project_root=ROOT,
        source_design_path=args.source_design,
        source_jobs_path=args.source_jobs,
        queue_path=args.queue_output,
        output_path=args.output,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
