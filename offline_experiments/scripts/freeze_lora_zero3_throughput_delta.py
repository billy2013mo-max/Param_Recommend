#!/usr/bin/env python3
"""Fail-closed freeze of the post-recovery H800 LoRA ZeRO-3 screen delta.

The command writes only the explicitly requested approval-design candidate.  It
never creates or edits ``config/APPROVED_TO_RUN.json`` and refuses to overwrite
the live ``runtime/approval_design.json``.  Promotion and the human approval
binding remain separate, explicit steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from approval_gate import build_provenance_binding, build_queue_binding
from common import ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from freeze_lora_zero3_fix import validate_patch
from prepare_lora_zero3_throughput_delta import (
    DEFAULT_MAX_DELTA_JOBS,
    duplicate_job_ids,
    duplicate_key_rows,
    job_shape_errors,
    key_record,
    physical_key,
)
from run_job import live_runtime_identity


EXPECTED_GPU_IDS = [1, 2, 3, 4]
EXPECTED_MODELS = {"qwen3_8b", "qwen3_14b"}
EXPECTED_UPSTREAM_FIX = {
    "pull_request": "https://github.com/deepspeedai/DeepSpeed/pull/8073",
    "commit": "b5b3fded4049d5e623aecdd0720d4b6b96a947af",
}
REPORT_FILE_INPUTS = (
    "baseline",
    "new_matrix",
    "pending_queue",
    "recovery_plan",
    "recovery_families",
    "boundary_evidence",
)
MATRIX_MANIFEST_FILES = (
    "design_summary.json",
    "memory_boundary_families.jsonl",
    "packing_pair_requests.jsonl",
    "profiler_requests.jsonl",
    "strong_scaling_requests.jsonl",
    "throughput_requests.jsonl",
)
ARTIFACT_MANIFEST_FILES = (
    "dataset_analysis.json",
    "model_inventory.json",
    "preprocessing_validation.json",
    "provenance.json",
    "smoke_validation.json",
)


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


def resolve_declared_path(value: Any, project_root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    declared = Path(value)
    if declared.is_absolute():
        return declared.resolve()
    candidates = [
        project_root / declared,
        project_root.parent / declared,
        Path.cwd() / declared,
    ]
    resolved = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in resolved:
            resolved.append(candidate)
    existing = [candidate for candidate in resolved if candidate.exists()]
    return existing[0] if len(existing) == 1 else None


def audit_report_inputs(report: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Re-hash every file input named by the preparation report."""

    rows = []
    bound_files: list[Path] = []
    inputs = report.get("inputs") or {}
    for name in REPORT_FILE_INPUTS:
        binding = inputs.get(name) or {}
        path = resolve_declared_path(binding.get("path"), project_root)
        expected = binding.get("sha256")
        checks = {
            "path_resolves_unambiguously": path is not None,
            "path_inside_project": path is not None and is_within(path, project_root),
            "file_exists": path is not None and path.is_file(),
            "sha256_declared": isinstance(expected, str) and len(expected) == 64,
            "sha256_matches": path is not None
            and path.is_file()
            and isinstance(expected, str)
            and sha256_file(path) == expected,
        }
        if checks["file_exists"] and checks["path_inside_project"]:
            bound_files.append(path)  # type: ignore[arg-type]
        rows.append(
            {
                "name": name,
                "declared_path": binding.get("path"),
                "resolved_path": str(path) if path else None,
                "expected_sha256": expected,
                "actual_sha256": sha256_file(path)
                if path is not None and path.is_file()
                else None,
                "checks": checks,
                "all_passed": all(checks.values()),
            }
        )
    return {
        "rows": rows,
        "bound_files": bound_files,
        "all_passed": len(rows) == len(REPORT_FILE_INPUTS)
        and all(row["all_passed"] for row in rows),
    }


def queue_row_checks(
    job: dict[str, Any], experiment: dict[str, Any]
) -> dict[str, bool]:
    scope = experiment.get("training_scope") or {}
    measurement = experiment.get("measurement") or {}
    gpu_count = job.get("gpu_count")
    mbs = job.get("mbs")
    target_gbs = job.get("target_gbs")
    return {
        "base_shape_valid": not job_shape_errors(job),
        "h800_recovered_model": job.get("model_id") in EXPECTED_MODELS,
        "lora_zero3_screen": job.get("kind") == "throughput_screen"
        and job.get("fidelity") == "screen"
        and job.get("train_type") == "lora"
        and job.get("zero") == "zero3",
        "gpu_count_allowed": isinstance(gpu_count, int)
        and not isinstance(gpu_count, bool)
        and gpu_count in {2, 4},
        "gpu_count_within_scope": isinstance(gpu_count, int)
        and not isinstance(gpu_count, bool)
        and gpu_count in set(scope.get("gpu_counts") or ()),
        "unpacked_recovery_shape": job.get("packing") is False,
        "screen_steps_match_policy": job.get("warmup_steps")
        == measurement.get("throughput_screen_warmup_steps")
        and job.get("measure_steps")
        == measurement.get("throughput_screen_measure_steps"),
        "single_screen_repeat": job.get("repeat") == 0,
        "partitionable_without_external_idle": job.get("parallel_class")
        == "gpu_partitionable"
        and job.get("requires_external_node_idle") is False,
        "qwen3_metadata_present": job.get("model_family") == "qwen3"
        and job.get("template") == "qwen3_nothink"
        and isinstance(job.get("model_path"), str)
        and bool(job.get("model_path"))
        and isinstance(job.get("tokenizer_path"), str)
        and bool(job.get("tokenizer_path"))
        and isinstance(job.get("model_parameters"), int)
        and int(job.get("model_parameters") or 0) > 0,
        "stable_id_shape": isinstance(job.get("job_id"), str)
        and str(job.get("job_id")).startswith("tputscreen-")
        and isinstance(job.get("request_id"), str)
        and str(job.get("request_id")).startswith("tput-"),
        "target_gbs_divisible": isinstance(target_gbs, int)
        and not isinstance(target_gbs, bool)
        and isinstance(gpu_count, int)
        and not isinstance(gpu_count, bool)
        and isinstance(mbs, int)
        and not isinstance(mbs, bool)
        and gpu_count > 0
        and mbs > 0
        and target_gbs % (gpu_count * mbs) == 0,
    }


def validate_freeze_inputs(
    *,
    report: dict[str, Any],
    report_sha256: str,
    input_queue_path: Path,
    input_queue_sha256: str,
    input_rows: list[dict[str, Any]],
    output_queue_path: Path,
    output_queue_sha256: str,
    output_rows: list[dict[str, Any]],
    recovery_results_path: Path,
    recovery_results_sha256: str,
    recovery_results: dict[str, Any],
    canary_validation: dict[str, Any],
    experiment: dict[str, Any],
    current_runtime_identity: dict[str, Any],
    current_patch: dict[str, Any],
    report_input_audit: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    """Pure validation of all authorization-critical values and ordering."""

    report_inputs = report.get("inputs") or {}
    report_pending = report_inputs.get("pending_queue") or {}
    report_boundary = report_inputs.get("boundary_evidence") or {}
    report_output = report.get("output_queue") or {}
    report_candidates = list(report.get("delta_candidates") or ())
    report_ids = list(
        (report.get("freeze_preparation") or {}).get("allowed_job_ids") or ()
    )
    input_ids = [str(row.get("job_id")) for row in input_rows]
    output_ids = [str(row.get("job_id")) for row in output_rows]
    candidate_ids = [str(row.get("job_id")) for row in report_candidates]
    candidate_physical_keys = [row.get("physical_key") for row in report_candidates]
    expected_physical_keys = [key_record(physical_key(row)) for row in output_rows]
    row_validations = [
        {
            "job_id": str(row.get("job_id")),
            "checks": queue_row_checks(row, experiment),
        }
        for row in output_rows
    ]
    for row in row_validations:
        row["all_passed"] = all(row["checks"].values())

    scope = experiment.get("training_scope") or {}
    zero_by_gpu_count = scope.get("zero_by_gpu_count") or {}
    expected_runtime_fingerprint = sha256_json(
        canary_validation.get("runtime_identity") or {}
    )
    recovery_checks = recovery_results.get("checks") or {}
    recovery_families = list(recovery_results.get("families") or ())
    canary_patch = canary_validation.get("patch") or {}
    recovery_patch = recovery_results.get("patch") or {}
    report_counts = report.get("counts") or {}
    queue_binding: dict[str, Any] = {}
    queue_binding_error: str | None = None
    try:
        queue_binding = build_queue_binding(
            output_queue_path,
            output_rows,
            project_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        queue_binding_error = repr(error)
    provenance_binding: dict[str, Any] = {}
    provenance_binding_error: str | None = None
    try:
        provenance_binding = build_provenance_binding(project_root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        provenance_binding_error = repr(error)

    checks = {
        "delta_report_sha256_present": isinstance(report_sha256, str)
        and len(report_sha256) == 64,
        "delta_report_passed": report.get("all_passed") is True,
        "delta_report_checks_recomputed": all_true(report.get("checks")),
        "delta_report_has_no_violations": all_empty(report.get("violations")),
        "delta_report_file_inputs_current": report_input_audit.get("all_passed")
        is True,
        "input_queue_path_bound": resolve_declared_path(
            report_pending.get("path"), project_root
        )
        == input_queue_path.resolve(),
        "input_queue_sha_bound": report_pending.get("sha256") == input_queue_sha256,
        "output_queue_was_written": report_output.get("written") is True,
        "output_queue_path_bound": resolve_declared_path(
            report_output.get("path"), project_root
        )
        == output_queue_path.resolve(),
        "output_queue_sha_bound": report_output.get("sha256") == output_queue_sha256,
        "input_and_output_queue_rows_identical": [
            sha256_json(row) for row in input_rows
        ]
        == [sha256_json(row) for row in output_rows],
        "ordered_job_ids_exact": bool(output_ids)
        and input_ids == output_ids == report_ids == candidate_ids,
        "queue_job_ids_unique": not duplicate_job_ids(output_rows),
        "queue_physical_keys_unique": not duplicate_key_rows(output_rows),
        "queue_rows_valid": bool(row_validations)
        and all(row["all_passed"] for row in row_validations),
        "queue_authorization_binding_complete": queue_binding_error is None
        and queue_binding.get("ordered_job_ids") == output_ids
        and queue_binding.get("sha256") == output_queue_sha256,
        "candidate_rows_revalidated": len(report_candidates) == len(output_rows)
        and all(
            row.get("all_passed") is True and all_true(row.get("checks"))
            for row in report_candidates
        )
        and candidate_physical_keys == expected_physical_keys,
        "delta_counts_exact": report_output.get("jobs") == len(output_rows)
        and report_counts.get("pending_jobs") == len(input_rows)
        and report_counts.get("delta_jobs") == len(output_rows)
        and 0 < len(output_rows) <= DEFAULT_MAX_DELTA_JOBS,
        "recovery_result_path_bound": report_boundary.get("mode")
        == "validated_manifest"
        and resolve_declared_path(report_boundary.get("path"), project_root)
        == recovery_results_path.resolve(),
        "recovery_result_sha_bound": report_boundary.get("sha256")
        == recovery_results_sha256,
        "recovery_result_passed": recovery_results.get("all_passed") is True
        and all_true(recovery_checks),
        "twenty_recovery_families_passed": len(recovery_families) == 20
        and all(row.get("all_passed") is True for row in recovery_families),
        "canary_validation_passed": canary_validation.get("all_passed") is True,
        "upstream_fix_exact": canary_validation.get("upstream_fix")
        == EXPECTED_UPSTREAM_FIX,
        "runtime_identity_present": bool(canary_validation.get("runtime_identity")),
        "runtime_identity_matches_recovery": recovery_results.get(
            "expected_runtime_fingerprint_sha256"
        )
        == expected_runtime_fingerprint,
        "live_runtime_identity_matches": current_runtime_identity
        == canary_validation.get("runtime_identity"),
        "canary_patch_passed": canary_patch.get("all_passed") is True,
        "recovery_patch_passed": recovery_patch.get("all_passed") is True,
        "live_patch_passed": current_patch.get("all_passed") is True,
        "patch_identity_matches": current_patch.get("installed_deepspeed_source_sha256")
        == recovery_patch.get("installed_deepspeed_source_sha256")
        == canary_patch.get("installed_deepspeed_source_sha256")
        == current_runtime_identity.get("framework_source_sha256", {}).get(
            "deepspeed_zero_partition_parameters"
        )
        and current_patch.get("runtime_patcher_sha256")
        == recovery_patch.get("runtime_patcher_sha256")
        == canary_patch.get("runtime_patcher_sha256")
        == current_runtime_identity.get("launcher_patch_sha256"),
        "fresh_provenance_path_set_bound": provenance_binding_error is None
        and bool(provenance_binding.get("project_source_paths"))
        and isinstance(provenance_binding.get("sha256"), str),
        "h800_scope_exact": "H800" in str(scope.get("gpu_type") or "")
        and scope.get("gpu_ids") == EXPECTED_GPU_IDS
        and scope.get("max_gpu_count") == 4,
        "zero3_supported_for_delta_gpu_counts": all(
            "zero3" in (zero_by_gpu_count.get(str(gpu_count)) or ())
            for gpu_count in {int(row["gpu_count"]) for row in output_rows}
        ),
    }
    return {
        "schema_version": 1,
        "purpose": "Freeze only the validated H800 LoRA + ZeRO-3 throughput-screen delta",
        "checks": checks,
        "queue_rows": row_validations,
        "allowed_job_ids": output_ids,
        "delta_report_sha256": report_sha256,
        "input_queue_sha256": input_queue_sha256,
        "output_queue_sha256": output_queue_sha256,
        "recovery_results_validation_sha256": recovery_results_sha256,
        "runtime_identity": canary_validation.get("runtime_identity"),
        "runtime_fingerprint_sha256": expected_runtime_fingerprint,
        "runtime_fix": canary_validation.get("upstream_fix"),
        "runtime_patch": current_patch,
        "queue_binding": queue_binding,
        "queue_binding_error": queue_binding_error,
        "provenance_binding": provenance_binding,
        "provenance_binding_error": provenance_binding_error,
        "all_passed": all(checks.values()),
    }


def build_freeze_metadata(validation: dict[str, Any]) -> dict[str, Any]:
    if validation.get("all_passed") is not True:
        raise ValueError("Cannot build approval metadata from a failed validation")
    allowed_job_ids = list(validation["allowed_job_ids"])
    return {
        "design_purpose": validation["purpose"],
        "execution_order": ["throughput_screen_delta"],
        "allowed_job_ids": allowed_job_ids,
        "authorized_gpu_ids": EXPECTED_GPU_IDS,
        "max_gpu_count": 4,
        "runtime_fix": validation["runtime_fix"],
        "runtime_identity": validation["runtime_identity"],
        "runtime_fingerprint_sha256": validation["runtime_fingerprint_sha256"],
        "runtime_patch": validation["runtime_patch"],
        "queue_binding": validation["queue_binding"],
        "provenance_binding": validation["provenance_binding"],
        "throughput_screen_delta": {
            "jobs": len(allowed_job_ids),
            "allowed_job_ids": allowed_job_ids,
            "delta_report_sha256": validation["delta_report_sha256"],
            "input_queue_sha256": validation["input_queue_sha256"],
            "output_queue_sha256": validation["output_queue_sha256"],
            "queue_path": validation["queue_binding"]["path"],
            "queue_sha256": validation["queue_binding"]["sha256"],
            "ordered_job_ids": validation["queue_binding"]["ordered_job_ids"],
            "ordered_job_payload_sha256": validation["queue_binding"][
                "ordered_job_payload_sha256"
            ],
            "job_payload_sha256": validation["queue_binding"]["job_payload_sha256"],
            "recovery_results_validation_sha256": validation[
                "recovery_results_validation_sha256"
            ],
            "policy": (
                "Run exactly this ordered narrow queue; accept success/OOM; "
                "any other failure blocks; no baseline or historical key may be rerun."
            ),
        },
    }


def approval_manifest_paths(
    project_root: Path,
    extra_files: Iterable[Path] = (),
) -> list[Path]:
    """Return the complete, deterministic approval manifest input set.

    This deliberately mirrors ``validate_setup.freeze_approval_design`` without
    importing or invoking it: that function owns the live approval-design path.
    Every file must exist and resolve inside the campaign root before a candidate
    design can be constructed.
    """

    root = project_root.resolve()
    config_dir = root / "config"
    matrix_dir = root / "matrix"
    artifact_dir = root / "artifacts"
    data_dir = root / "data"
    approval_file = config_dir / "APPROVED_TO_RUN.json"

    files: list[Path] = []
    files.extend(
        path
        for path in config_dir.rglob("*")
        if path.is_file() and path != approval_file
    )
    files.extend(path for path in (root / "scripts").glob("*.py") if path.is_file())
    files.extend(matrix_dir / name for name in MATRIX_MANIFEST_FILES)
    files.extend(artifact_dir / name for name in ARTIFACT_MANIFEST_FILES)
    files.extend(
        (
            data_dir / "dataset_info.json",
            root / "README.md",
            root / "EXPERIMENT_DESIGN.md",
        )
    )
    files.extend((data_dir / "derived").glob("*.jsonl"))
    files.extend(extra_files)

    normalized: list[Path] = []
    outside: list[str] = []
    for path in files:
        candidate = path if path.is_absolute() else root / path
        candidate = candidate.resolve()
        if not is_within(candidate, root):
            outside.append(str(candidate))
        else:
            normalized.append(candidate)
    if outside:
        raise ValueError(
            "Cannot freeze approval-design candidate; manifest inputs escape "
            f"the campaign root: {sorted(set(outside))}"
        )

    unique_files = sorted(set(normalized))
    absent = [str(path) for path in unique_files if not path.is_file()]
    if absent:
        raise FileNotFoundError(
            "Cannot freeze approval-design candidate; required files are absent: "
            f"{absent}"
        )
    return unique_files


def build_approval_design_candidate(
    *,
    project_root: Path,
    extra_files: Iterable[Path] = (),
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Purely construct a non-live approval design with the canonical schema."""

    root = project_root.resolve()
    files = approval_manifest_paths(root, extra_files)
    manifest = {str(path.relative_to(root)): sha256_file(path) for path in files}
    design: dict[str, Any] = {
        "schema_version": 1,
        "training_started": False,
        "file_sha256": manifest,
        "matrix_summary": read_json(root / "matrix" / "design_summary.json"),
        "approval_instruction": "Approval must bind to the SHA256 of this exact file.",
    }
    metadata = extra_metadata or {}
    reserved = sorted(set(design).intersection(metadata))
    if reserved:
        raise ValueError(
            "Refusing approval metadata that overrides canonical design fields: "
            f"{reserved}"
        )
    design.update(metadata)
    return design


def output_path_is_safe(
    output_path: Path, protected_paths: Iterable[Path], project_root: Path
) -> bool:
    output = output_path.resolve()
    root = project_root.resolve()
    forbidden = {
        (root / "config" / "APPROVED_TO_RUN.json").resolve(),
        (root / "runtime" / "approval_design.json").resolve(),
        *(
            path.resolve()
            for path in (root / "matrix" / name for name in MATRIX_MANIFEST_FILES)
        ),
        *(
            path.resolve()
            for path in (root / "artifacts" / name for name in ARTIFACT_MANIFEST_FILES)
        ),
        (root / "data" / "dataset_info.json").resolve(),
        *(path.resolve() for path in protected_paths),
    }
    # Any newly created JSON below config/ would silently join the canonical
    # recursive manifest after construction, creating a self-referential and
    # immediately stale candidate on a later overwrite.
    inside_config = is_within(output, root / "config")
    return (
        is_within(output, root)
        and not inside_config
        and output.suffix == ".json"
        and output not in forbidden
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-report", type=Path, required=True)
    parser.add_argument("--input-queue", type=Path, required=True)
    parser.add_argument("--output-queue", type=Path, required=True)
    parser.add_argument("--recovery-results-validation", type=Path, required=True)
    parser.add_argument(
        "--canary-validation",
        type=Path,
        default=ROOT / "artifacts" / "h800_lora_zero3_fix_validation.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_paths = (
        args.delta_report,
        args.input_queue,
        args.output_queue,
        args.recovery_results_validation,
        args.canary_validation,
        ROOT / "config" / "experiment.json",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Required freeze inputs are absent: {missing}")
    if not output_path_is_safe(args.output, required_paths, ROOT):
        raise SystemExit(
            "Refusing unsafe --output: use a distinct .json inside the campaign root; "
            "live approval files are forbidden"
        )
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {args.output} without --overwrite")

    report = read_json(args.delta_report)
    input_rows = read_jsonl(args.input_queue)
    output_rows = read_jsonl(args.output_queue)
    recovery_results = read_json(args.recovery_results_validation)
    canary = read_json(args.canary_validation)
    experiment = read_json(ROOT / "config" / "experiment.json")
    input_audit = audit_report_inputs(report, ROOT)
    report_files = list(input_audit["bound_files"])
    if not output_path_is_safe(
        args.output,
        (*required_paths, *report_files),
        ROOT,
    ):
        raise SystemExit(
            "Refusing unsafe --output: it aliases a file bound by the delta report"
        )
    validation = validate_freeze_inputs(
        report=report,
        report_sha256=sha256_file(args.delta_report),
        input_queue_path=args.input_queue,
        input_queue_sha256=sha256_file(args.input_queue),
        input_rows=input_rows,
        output_queue_path=args.output_queue,
        output_queue_sha256=sha256_file(args.output_queue),
        output_rows=output_rows,
        recovery_results_path=args.recovery_results_validation,
        recovery_results_sha256=sha256_file(args.recovery_results_validation),
        recovery_results=recovery_results,
        canary_validation=canary,
        experiment=experiment,
        current_runtime_identity=live_runtime_identity(),
        current_patch=validate_patch(),
        report_input_audit=input_audit,
        project_root=ROOT,
    )
    if not validation["all_passed"]:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    extra_files = [
        args.delta_report.resolve(),
        args.input_queue.resolve(),
        args.output_queue.resolve(),
        args.recovery_results_validation.resolve(),
        args.canary_validation.resolve(),
        *report_files,
    ]
    candidate = build_approval_design_candidate(
        project_root=ROOT,
        extra_files=extra_files,
        extra_metadata=build_freeze_metadata(validation),
    )
    write_json(args.output, candidate)
    frozen = {
        "path": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "manifest_entries": len(candidate["file_sha256"]),
    }
    print(
        json.dumps(
            {
                "all_passed": True,
                "jobs": len(output_rows),
                "approval_design_candidate": frozen,
                "approval_file_written": False,
                "live_approval_design_overwritten": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
