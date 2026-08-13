#!/usr/bin/env python3
"""Safely promote an exact approval-design candidate; dry-run by default.

Promotion is an explicit transaction.  It acquires the execution gate
exclusively, confirms that no legacy launcher is active, revalidates every
candidate binding, snapshots history, closes the old approval, installs the
candidate's original bytes, and enables the new approval as the final atomic
write.  Any failure after closing the gate rewrites a locked approval.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from approval_gate import (
    execution_lock,
    is_within,
    reject_retired_approval_plan,
    retired_approval_plan_schemas,
    strict_id_list,
    validate_provenance_binding,
    validate_queue_binding,
)
from common import (
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    verify_file_manifest,
)
from run_job import live_runtime_identity, live_runtime_patch, verify_approval


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes, mode: int = 0o600) -> None:
    """Durably replace one file using a temporary file in the same directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def active_execution_processes(proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    """Find legacy/participating launchers that could predate the gate lock."""

    active = []
    own_pid = os.getpid()
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or int(directory.name) == own_pid:
            continue
        try:
            command = (
                (directory / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        is_scheduler = "scripts/scheduler.py" in command and "--execute" in command
        is_job = "scripts/run_job.py" in command and (
            "--execute" in command or "--execute-smoke" in command
        )
        is_pipeline = "scripts/run_pipeline.py" in command
        if is_scheduler or is_job or is_pipeline:
            active.append({"pid": int(directory.name), "command": command.strip()})
    return sorted(active, key=lambda row: row["pid"])


def validate_candidate(
    *,
    candidate_path: Path,
    expected_candidate_sha256: str,
    project_root: Path,
    current_runtime_identity: dict[str, Any] | None = None,
    current_runtime_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Revalidate all authorization-critical candidate and live evidence."""

    root = project_root.resolve()
    path = candidate_path.resolve()
    expected_digest_valid = len(expected_candidate_sha256) == 64 and all(
        character in "0123456789abcdef" for character in expected_candidate_sha256
    )
    if not is_within(path, root):
        raise ValueError("Approval-design candidate must be inside the project root")
    if path in {
        (root / "runtime" / "approval_design.json").resolve(),
        (root / "config" / "APPROVED_TO_RUN.json").resolve(),
    }:
        raise ValueError("Candidate must be distinct from both live approval files")
    if not path.is_file():
        raise FileNotFoundError(path)
    candidate_bytes = path.read_bytes()
    actual_sha256 = sha256_bytes(candidate_bytes)
    try:
        design = json.loads(candidate_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"Candidate is not valid JSON: {error}") from error
    if not isinstance(design, dict):
        raise ValueError("Candidate JSON must be an object")

    manifest = design.get("file_sha256") or {}
    manifest_check = (
        verify_file_manifest(root, manifest)
        if isinstance(manifest, dict) and manifest
        else {"entries": 0, "missing": [], "mismatched": [], "all_passed": False}
    )
    allowed_ids = strict_id_list(design.get("allowed_job_ids"))
    execution_order = strict_id_list(design.get("execution_order"))
    queue_check = validate_queue_binding(design, root)
    provenance_check = validate_provenance_binding(design, root)
    expected_identity = design.get("runtime_identity") or {}
    expected_fingerprint = design.get("runtime_fingerprint_sha256")
    live_identity = current_runtime_identity or live_runtime_identity()
    expected_patch = design.get("runtime_patch") or {}
    live_patch = current_runtime_patch or live_runtime_patch()
    experiment = read_json(root / "config" / "experiment.json")
    scope = experiment.get("training_scope") or {}
    measurement = experiment.get("measurement") or {}
    stage = design.get("throughput_screen_delta") or {}
    queue_binding = design.get("queue_binding") or {}
    retired_schemas = retired_approval_plan_schemas(design)
    checks = {
        "expected_candidate_sha256_valid": expected_digest_valid,
        "candidate_sha256_exact": expected_digest_valid
        and actual_sha256 == expected_candidate_sha256,
        "candidate_not_retired": not retired_schemas,
        "schema_supported": design.get("schema_version") == 1,
        "training_not_started": design.get("training_started") is False,
        "manifest_current": manifest_check.get("all_passed") is True,
        "allowed_job_ids_nonempty_unique": allowed_ids is not None,
        "execution_order_nonempty_unique": execution_order is not None,
        "queue_binding_current_and_complete": queue_check.get("all_passed") is True,
        "provenance_current_and_complete": provenance_check.get("all_passed") is True,
        "runtime_identity_fingerprint_internal": bool(expected_identity)
        and expected_fingerprint == sha256_json(expected_identity),
        "live_runtime_identity_exact": bool(expected_identity)
        and live_identity == expected_identity,
        "runtime_patch_healthy_and_exact": isinstance(expected_patch, dict)
        and expected_patch.get("all_passed") is True
        and live_patch.get("all_passed") is True
        and live_patch == expected_patch,
        "gpu_scope_exact": design.get("authorized_gpu_ids") == scope.get("gpu_ids")
        and design.get("max_gpu_count") == scope.get("max_gpu_count"),
        "throughput_stage_metadata_exact": stage.get("allowed_job_ids") == allowed_ids
        and stage.get("queue_path") == queue_binding.get("path")
        and stage.get("queue_sha256") == queue_binding.get("sha256")
        and stage.get("ordered_job_ids") == queue_binding.get("ordered_job_ids")
        and stage.get("ordered_job_payload_sha256")
        == queue_binding.get("ordered_job_payload_sha256")
        and stage.get("job_payload_sha256") == queue_binding.get("job_payload_sha256"),
        "parallelism_declared": measurement.get("performance_parallelism")
        in {"disjoint_gpu_masks", "exclusive_pool"},
    }
    return {
        "schema_version": 1,
        "candidate_path": str(path),
        "candidate_sha256": actual_sha256,
        "candidate_bytes": candidate_bytes,
        "design": design,
        "manifest": manifest_check,
        "queue": queue_check,
        "provenance": provenance_check,
        "retired_plan_schemas": retired_schemas,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def build_approval_payload(
    *,
    design: dict[str, Any],
    candidate_sha256: str,
    project_root: Path,
    authorization: str,
    approved_by: str,
    transaction_id: str,
    receipt_path: Path,
    created_unix: float,
) -> dict[str, Any]:
    reject_retired_approval_plan(
        design,
        operation="build or promote an approval payload",
    )
    if not authorization.strip() or not approved_by.strip():
        raise ValueError("authorization and approved_by must both be non-empty")
    root = project_root.resolve()
    experiment = read_json(root / "config" / "experiment.json")
    scope = experiment["training_scope"]
    measurement = experiment["measurement"]
    queue_path = root / design["queue_binding"]["path"]
    queue_rows = [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model_ids = sorted({str(row["model_id"]) for row in queue_rows})
    return {
        "schema_version": 2,
        "approved": True,
        "design_sha256": candidate_sha256,
        "phase_id": design["execution_order"][0],
        "approved_model_ids": model_ids,
        "resource_scope": {
            "gpu_ids": list(scope["gpu_ids"]),
            "max_gpu_count": int(scope["max_gpu_count"]),
            "allow_gpu_ids_outside_pool": False,
            "performance_parallelism": measurement["performance_parallelism"],
        },
        "execution_order": list(design["execution_order"]),
        "allowed_job_ids": list(design["allowed_job_ids"]),
        "queue_binding_sha256": sha256_json(design["queue_binding"]),
        "runtime_fingerprint_sha256": design["runtime_fingerprint_sha256"],
        "runtime_patch_sha256": sha256_json(design["runtime_patch"]),
        "provenance_sha256": design["provenance_binding"]["sha256"],
        "runtime_fix": design.get("runtime_fix"),
        "authorization": authorization.strip(),
        "approved_by": approved_by.strip(),
        "approved_unix": created_unix,
        "promotion": {
            "transaction_id": transaction_id,
            "candidate_sha256": candidate_sha256,
            "receipt_path": str(receipt_path.resolve().relative_to(root)),
        },
    }


def closed_approval_payload(
    *, transaction_id: str, candidate_sha256: str, reason: str
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "approved": False,
        "locked": True,
        "transaction_id": transaction_id,
        "candidate_sha256": candidate_sha256,
        "reason": reason,
    }


def promote_candidate(
    *,
    candidate_path: Path,
    expected_candidate_sha256: str,
    project_root: Path,
    authorization: str,
    approved_by: str,
    promote: bool = False,
    process_probe: Callable[[], list[dict[str, Any]]] = active_execution_processes,
    current_runtime_identity: dict[str, Any] | None = None,
    current_runtime_patch: dict[str, Any] | None = None,
    created_unix: float | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    runtime_dir = root / "runtime"
    approval_path = root / "config" / "APPROVED_TO_RUN.json"
    design_path = runtime_dir / "approval_design.json"
    now = time.time() if created_unix is None else created_unix
    transaction_id = f"approval-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(now))}-{expected_candidate_sha256[:12]}"
    history_dir = runtime_dir / "approval_history" / transaction_id
    receipt_path = (
        runtime_dir / "approval_promotion_receipts" / f"{transaction_id}.json"
    )

    with execution_lock(runtime_dir, exclusive=True):
        active = process_probe()
        live_identity = (
            live_runtime_identity()
            if current_runtime_identity is None
            else current_runtime_identity
        )
        live_patch = (
            live_runtime_patch()
            if current_runtime_patch is None
            else current_runtime_patch
        )
        validation = validate_candidate(
            candidate_path=candidate_path,
            expected_candidate_sha256=expected_candidate_sha256,
            project_root=root,
            current_runtime_identity=live_identity,
            current_runtime_patch=live_patch,
        )
        report = {
            key: value
            for key, value in validation.items()
            if key not in {"candidate_bytes", "design"}
        }
        report["active_execution_processes"] = active
        report["no_execution_processes"] = not active
        report["mode"] = "promote" if promote else "dry-run"
        report["transaction_id"] = transaction_id
        if active or validation.get("all_passed") is not True:
            report["all_passed"] = False
            return report

        approval = build_approval_payload(
            design=validation["design"],
            candidate_sha256=validation["candidate_sha256"],
            project_root=root,
            authorization=authorization,
            approved_by=approved_by,
            transaction_id=transaction_id,
            receipt_path=receipt_path,
            created_unix=now,
        )
        approval_bytes = json_bytes(approval)
        report.update(
            {
                "all_passed": True,
                "would_install_design": str(design_path),
                "would_write_approval": str(approval_path),
                "new_approval_sha256": sha256_bytes(approval_bytes),
                "history_dir": str(history_dir),
                "receipt_path": str(receipt_path),
                "promoted": False,
            }
        )
        if not promote:
            return report

        if history_dir.exists() or receipt_path.exists():
            raise FileExistsError(
                f"Promotion transaction already exists: {history_dir} / {receipt_path}"
            )
        old_approval = approval_path.read_bytes() if approval_path.is_file() else None
        old_design = design_path.read_bytes() if design_path.is_file() else None
        history_dir.mkdir(parents=True, exist_ok=False)
        if old_approval is not None:
            atomic_write_bytes(history_dir / "previous_approval.json", old_approval)
        if old_design is not None:
            atomic_write_bytes(
                history_dir / "previous_approval_design.json", old_design
            )
        atomic_write_bytes(
            history_dir / "candidate_approval_design.json",
            validation["candidate_bytes"],
        )
        atomic_write_bytes(history_dir / "new_approval.json", approval_bytes)
        receipt = {
            "schema_version": 1,
            "transaction_id": transaction_id,
            "prepared_unix": now,
            "activation_protocol": (
                "This immutable receipt was prepared before activation. The new approval "
                "is the transaction's final atomic write; compare the live hashes below."
            ),
            "candidate_source_path": str(candidate_path.resolve()),
            "candidate_sha256": validation["candidate_sha256"],
            "previous_approval_sha256": sha256_bytes(old_approval)
            if old_approval is not None
            else None,
            "previous_design_sha256": sha256_bytes(old_design)
            if old_design is not None
            else None,
            "new_approval_sha256": sha256_bytes(approval_bytes),
            "live_design_path": str(design_path),
            "live_approval_path": str(approval_path),
            "history_dir": str(history_dir),
            "approved_by": approved_by.strip(),
            "authorization": authorization.strip(),
        }
        atomic_write_bytes(receipt_path, json_bytes(receipt))

        closed = closed_approval_payload(
            transaction_id=transaction_id,
            candidate_sha256=validation["candidate_sha256"],
            reason="Approval promotion in progress; fail closed until final atomic approval write.",
        )
        gate_closed = False
        try:
            atomic_write_bytes(approval_path, json_bytes(closed))
            gate_closed = True
            atomic_write_bytes(design_path, validation["candidate_bytes"])
            if sha256_file(design_path) != validation["candidate_sha256"]:
                raise OSError(
                    "Installed design bytes do not match the candidate SHA256"
                )
            # This is deliberately the final mutating step of a successful
            # promotion transaction.
            atomic_write_bytes(approval_path, approval_bytes)
            if (
                sha256_file(approval_path) != sha256_bytes(approval_bytes)
                or sha256_file(design_path) != validation["candidate_sha256"]
            ):
                raise OSError("Post-promotion live hash verification failed")
            bound_queue_path = root / validation["design"]["queue_binding"]["path"]
            bound_rows = read_jsonl(bound_queue_path)
            for row in bound_rows:
                verify_approval(
                    row,
                    queue_path=bound_queue_path,
                    queue_rows=bound_rows,
                    acquire_lock=False,
                    project_root=root,
                    approval_file=approval_path,
                    approval_design_path=design_path,
                    config_dir=root / "config",
                    runtime_dir=runtime_dir,
                    current_runtime_identity=live_identity,
                    current_runtime_patch=live_patch,
                )
        except BaseException as error:
            if gate_closed:
                try:
                    atomic_write_bytes(
                        approval_path,
                        json_bytes(
                            closed_approval_payload(
                                transaction_id=transaction_id,
                                candidate_sha256=validation["candidate_sha256"],
                                reason=f"Promotion failed closed: {error!r}",
                            )
                        ),
                    )
                except BaseException as close_error:
                    raise RuntimeError(
                        "Promotion failed and the locked approval could not be rewritten"
                    ) from close_error
            raise

        report["promoted"] = True
        report["installed_design_sha256"] = sha256_file(design_path)
        report["installed_approval_sha256"] = sha256_file(approval_path)
        report["receipt_sha256"] = sha256_file(receipt_path)
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Perform the locked atomic promotion; without this flag only validate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = promote_candidate(
            candidate_path=args.candidate,
            expected_candidate_sha256=args.expected_candidate_sha256,
            project_root=args.project_root,
            authorization=args.authorization,
            approved_by=args.approved_by,
            promote=args.promote,
        )
    except (OSError, ValueError, PermissionError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {"all_passed": False, "error": repr(error)},
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(2) from error
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
