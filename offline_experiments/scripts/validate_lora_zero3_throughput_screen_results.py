#!/usr/bin/env python3
"""Fail-closed post-run validation for the H800 LoRA ZeRO-3 screen delta.

The validator is deliberately separate from the scheduler.  It never launches
training and never edits the live approval files.  It re-hashes the preparation
and approval inputs, validates every result in the exact frozen queue, and emits
self-contained approval snapshots plus a validation report for the next stage.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from common import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    RESULTS_DIR,
    ROOT,
    RUNTIME_DIR,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    verify_file_manifest,
    write_json,
)
from freeze_lora_zero3_fix import validate_patch
from freeze_lora_zero3_throughput_delta import (
    all_empty,
    all_true,
    audit_report_inputs,
    queue_row_checks,
    resolve_declared_path,
)
from prepare_lora_zero3_throughput_delta import (
    duplicate_job_ids,
    duplicate_key_rows,
    key_record,
    physical_key,
)
from run_job import OOM_PATTERNS, live_runtime_identity


DELTA_REPORT_PATH = ARTIFACT_DIR / "h800_lora_zero3_throughput_delta_validation.json"
INPUT_QUEUE_PATH = RUNTIME_DIR / "pipeline" / "pending-throughput-screen.jsonl"
OUTPUT_QUEUE_PATH = (
    RUNTIME_DIR / "pipeline" / "pending-h800-lora-zero3-throughput-delta.jsonl"
)
APPROVAL_PATH = CONFIG_DIR / "APPROVED_TO_RUN.json"
APPROVAL_DESIGN_PATH = RUNTIME_DIR / "approval_design.json"
PROVENANCE_PATH = ARTIFACT_DIR / "provenance.json"
VALIDATION_PATH = (
    ARTIFACT_DIR / "h800_lora_zero3_throughput_screen_results_validation.json"
)
APPROVAL_SNAPSHOT_PATH = (
    ARTIFACT_DIR / "h800_lora_zero3_throughput_screen_approval.json"
)
DESIGN_SNAPSHOT_PATH = (
    ARTIFACT_DIR / "h800_lora_zero3_throughput_screen_approval_design.json"
)
TERMINAL_CLASSIFICATIONS = {"success", "oom"}
FRESHNESS_TOLERANCE_SECONDS = 2.0
RUNTIME_RENDERED_FIELDS = {"runtime_model_path", "max_steps"}
FP32_LORA_MESSAGE = "DeepSpeed ZeRO3 detected, remaining trainable params in float32"
DTYPE_ERROR = "output tensor must have the same type as input tensor"


def parse_gpu_mask(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        raw = value
    elif isinstance(value, str):
        raw = [part.strip() for part in value.split(",") if part.strip()]
    else:
        return []
    try:
        return [int(part) for part in raw]
    except (TypeError, ValueError):
        return []


def positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def positive_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def artifact_hashes(paths: dict[str, Path]) -> dict[str, str | None]:
    return {
        name: sha256_file(path) if path.is_file() else None
        for name, path in paths.items()
    }


def read_json_dict(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError) as error:
        return {}, repr(error)
    if not isinstance(value, dict):
        return {}, f"expected JSON object, got {type(value).__name__}"
    return value, None


def rendered_payload_matches(
    queue_job: dict[str, Any], rendered_job: dict[str, Any]
) -> bool:
    expected_fields = set(queue_job)
    actual_fields = set(rendered_job)
    warmup_steps = positive_int(queue_job.get("warmup_steps"))
    measure_steps = positive_int(queue_job.get("measure_steps"))
    return (
        warmup_steps is not None
        and measure_steps is not None
        and expected_fields <= actual_fields
        and actual_fields - expected_fields == RUNTIME_RENDERED_FIELDS
        and all(rendered_job.get(field) == value for field, value in queue_job.items())
        and isinstance(rendered_job.get("runtime_model_path"), str)
        and bool(rendered_job.get("runtime_model_path"))
        and rendered_job.get("max_steps") == warmup_steps + measure_steps
    )


def metric_metadata_matches(
    queue_job: dict[str, Any],
    rendered_job: dict[str, Any],
    metadata: dict[str, Any],
) -> bool:
    return set(rendered_job) <= set(metadata) and all(
        metadata.get(field) == value for field, value in rendered_job.items()
    )


def validate_job_result(
    queue_job: dict[str, Any],
    *,
    results_dir: Path,
    authorized_gpu_ids: set[int],
    expected_approval_design_sha256: str,
    expected_runtime_identity: dict[str, Any],
    expected_runtime_fingerprint: str,
    expected_provenance_sha256: str,
) -> dict[str, Any]:
    """Validate one exact queue row against its immutable run artifacts."""

    job_id = str(queue_job.get("job_id") or "")
    result_dir = results_dir / job_id
    paths = {
        "status": result_dir / "status.json",
        "rendered_run": result_dir / "rendered_run.json",
        "runtime_identity": result_dir / "runtime_identity.json",
        "train_log": result_dir / "train.log",
    }
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        return {
            "job_id": job_id,
            "classification": "missing",
            "missing": missing,
            "artifact_sha256": artifact_hashes(paths),
            "checks": {
                "required_artifacts_present": False,
                "terminal_classification_present": False,
            },
            "all_passed": False,
        }

    status, status_error = read_json_dict(paths["status"])
    rendered, rendered_error = read_json_dict(paths["rendered_run"])
    runtime_identity, runtime_error = read_json_dict(paths["runtime_identity"])
    try:
        log = paths["train_log"].read_text(encoding="utf-8", errors="replace")
        log_error = None
    except OSError as error:
        log = ""
        log_error = repr(error)
    json_errors = {
        name: error
        for name, error in (
            ("status", status_error),
            ("rendered_run", rendered_error),
            ("runtime_identity", runtime_error),
            ("train_log", log_error),
        )
        if error is not None
    }
    rendered_job = dict(rendered.get("job") or {})
    classification = str(status.get("classification") or "missing")
    gpu_count = positive_int(queue_job.get("gpu_count")) or 0
    warmup_steps = positive_int(queue_job.get("warmup_steps")) or 0
    measure_steps = positive_int(queue_job.get("measure_steps")) or 0
    status_gpu_mask = parse_gpu_mask(status.get("gpu_mask"))
    rendered_gpu_mask = parse_gpu_mask(rendered.get("gpu_mask"))
    started = status.get("started_unix")
    finished = status.get("finished_unix")
    wall_seconds = status.get("wall_seconds")
    timestamps_numeric = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (started, finished, wall_seconds)
    )
    wall_consistent = bool(
        timestamps_numeric
        and float(finished) >= float(started)
        and math.isclose(
            float(finished) - float(started),
            float(wall_seconds),
            abs_tol=max(2.0, abs(float(wall_seconds)) * 0.02),
        )
    )
    computed_runtime_fingerprint = (
        sha256_json(runtime_identity) if runtime_identity else ""
    )
    runtime_identity_path = rendered.get("runtime_identity_path")
    runtime_identity_path_bound = False
    if isinstance(runtime_identity_path, str) and runtime_identity_path:
        runtime_identity_path_bound = (
            Path(runtime_identity_path).resolve() == paths["runtime_identity"].resolve()
        )

    checks: dict[str, bool] = {
        "required_artifacts_present": True,
        "required_artifacts_readable": not json_errors,
        "status_job_id_matches": status.get("job_id") == job_id,
        "terminal_classification_present": classification in TERMINAL_CLASSIFICATIONS,
        "rendered_payload_matches_exact_queue_row": rendered_payload_matches(
            queue_job, rendered_job
        ),
        "gpu_mask_matches_status_and_render": bool(status_gpu_mask)
        and status_gpu_mask == rendered_gpu_mask,
        "gpu_mask_matches_job_and_scope": len(status_gpu_mask) == gpu_count
        and len(status_gpu_mask) == len(set(status_gpu_mask))
        and set(status_gpu_mask) <= authorized_gpu_ids,
        "timestamps_present_and_consistent": wall_consistent,
        "core_artifacts_fresh_for_attempt": bool(
            timestamps_numeric
            and all(
                path.stat().st_mtime + FRESHNESS_TOLERANCE_SECONDS >= float(started)
                for path in paths.values()
            )
        ),
        "approval_design_bound": status.get("approval_design_sha256")
        == expected_approval_design_sha256,
        "provenance_bound": status.get("provenance_sha256")
        == rendered.get("provenance_sha256")
        == expected_provenance_sha256,
        "runtime_identity_path_bound": runtime_identity_path_bound,
        "runtime_identity_matches_approval_design": runtime_identity
        == expected_runtime_identity,
        "runtime_fingerprint_recomputed": computed_runtime_fingerprint
        == expected_runtime_fingerprint,
        "runtime_fingerprint_bound": status.get("runtime_fingerprint_sha256")
        == rendered.get("runtime_fingerprint_sha256")
        == computed_runtime_fingerprint,
        "dtype_failure_absent": DTYPE_ERROR not in log,
        "keyboard_interrupt_absent": "KeyboardInterrupt" not in log,
    }

    extra_paths: dict[str, Path] = {}
    if classification == "success":
        summary_paths = sorted((result_dir / "metrics").glob("summary.rank*.json"))
        summaries: list[dict[str, Any]] = []
        summary_read_errors = []
        for path in summary_paths:
            summary, error = read_json_dict(path)
            summaries.append(summary)
            if error is not None:
                summary_read_errors.append({"path": str(path), "error": error})
        train_results_path = result_dir / "trainer_output" / "train_results.json"
        train_results, train_results_error = read_json_dict(train_results_path)
        expected_ranks = list(range(gpu_count))
        expected_total_steps = warmup_steps + measure_steps
        expected_measured_steps = measure_steps
        ranks = [summary.get("rank") for summary in summaries]
        expected_summary_names = [f"summary.rank{rank}.json" for rank in expected_ranks]
        metadata_checks = []
        metric_checks = []
        step_checks = []
        for summary in summaries:
            metadata = dict(summary.get("metadata") or {})
            metadata_checks.append(
                metric_metadata_matches(queue_job, rendered_job, metadata)
            )
            totals = dict(summary.get("measured_totals") or {})
            metric_checks.append(
                all(
                    positive_finite(value)
                    for value in (
                        summary.get("computed_tokens_per_second"),
                        summary.get("effective_tokens_per_second"),
                        summary.get("logical_samples_per_second"),
                        summary.get("measured_seconds"),
                        summary.get("max_allocated"),
                        summary.get("max_reserved"),
                        totals.get("computed_tokens"),
                        totals.get("logical_samples"),
                    )
                )
            )
            step_checks.append(
                summary.get("failure") is None
                and summary.get("total_steps") == expected_total_steps
                and summary.get("measured_steps") == expected_measured_steps
            )
        extra_paths = {
            **{f"summary_rank_{rank}": path for rank, path in enumerate(summary_paths)},
            "train_results": train_results_path,
        }
        checks.update(
            {
                "return_code_zero": status.get("return_code") == 0,
                "oom_marker_absent": not any(
                    pattern.lower() in log.lower() for pattern in OOM_PATTERNS
                ),
                "fp32_lora_preserved": FP32_LORA_MESSAGE in log,
                "rank_summary_files_exact": [path.name for path in summary_paths]
                == expected_summary_names,
                "rank_summaries_readable": not summary_read_errors,
                "rank_set_complete": ranks == expected_ranks,
                "world_size_and_local_rank_match": len(summaries) == len(expected_ranks)
                and all(
                    summary.get("world_size") == len(expected_ranks)
                    and summary.get("local_rank") == rank
                    for rank, summary in enumerate(summaries)
                ),
                "metric_metadata_matches_rendered_job": bool(metadata_checks)
                and all(metadata_checks),
                "screen_steps_exact": bool(step_checks) and all(step_checks),
                "summary_metrics_finite_positive": bool(metric_checks)
                and all(metric_checks),
                "train_results_readable": train_results_path.is_file()
                and train_results_error is None,
                "finite_train_loss": finite_number(train_results.get("train_loss")),
                "success_artifacts_fresh": bool(
                    timestamps_numeric
                    and train_results_path.is_file()
                    and summary_paths
                    and all(
                        path.stat().st_mtime + FRESHNESS_TOLERANCE_SECONDS
                        >= float(started)
                        for path in (*summary_paths, train_results_path)
                    )
                ),
            }
        )
    elif classification == "oom":
        checks.update(
            {
                "return_code_nonzero": status.get("return_code") not in {None, 0},
                "oom_evidence_present": any(
                    pattern.lower() in log.lower() for pattern in OOM_PATTERNS
                ),
            }
        )
    else:
        checks["classification_allowed"] = False

    return {
        "job_id": job_id,
        "classification": classification,
        "gpu_mask": status_gpu_mask,
        "missing": [],
        "json_read_errors": json_errors,
        "artifact_sha256": artifact_hashes({**paths, **extra_paths}),
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def relative_manifest_key(path: Path, project_root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return None


def patch_identity_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    fields = (
        "installed_deepspeed_source_sha256",
        "runtime_patcher_sha256",
    )
    return (
        expected.get("all_passed") is True
        and actual.get("all_passed") is True
        and all(expected.get(field) == actual.get(field) for field in fields)
    )


def validate_screen_delta_results(
    *,
    delta_report_path: Path,
    input_queue_path: Path,
    output_queue_path: Path,
    approval_path: Path,
    approval_design_path: Path,
    experiment_path: Path,
    provenance_path: Path,
    results_dir: Path,
    project_root: Path = ROOT,
    current_runtime_identity: dict[str, Any] | None = None,
    current_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute every authorization and result check for the frozen delta."""

    delta_report = read_json(delta_report_path)
    input_rows = read_jsonl(input_queue_path)
    output_rows = read_jsonl(output_queue_path)
    approval = read_json(approval_path)
    approval_design = read_json(approval_design_path)
    experiment = read_json(experiment_path)
    report_input_audit = audit_report_inputs(delta_report, project_root)
    report_inputs = dict(delta_report.get("inputs") or {})
    report_pending = dict(report_inputs.get("pending_queue") or {})
    report_output = dict(delta_report.get("output_queue") or {})
    report_candidates = list(delta_report.get("delta_candidates") or ())
    report_ids = list(
        (delta_report.get("freeze_preparation") or {}).get("allowed_job_ids") or ()
    )
    input_ids = [str(row.get("job_id")) for row in input_rows]
    output_ids = [str(row.get("job_id")) for row in output_rows]
    candidate_ids = [str(row.get("job_id")) for row in report_candidates]
    output_physical_records = [key_record(physical_key(row)) for row in output_rows]
    candidate_physical_records = [row.get("physical_key") for row in report_candidates]

    new_matrix_binding = report_inputs.get("new_matrix") or {}
    new_matrix_path = resolve_declared_path(
        new_matrix_binding.get("path"), project_root
    )
    new_matrix_rows = (
        read_jsonl(new_matrix_path)
        if new_matrix_path is not None and new_matrix_path.is_file()
        else []
    )
    new_matrix_by_id = {str(row.get("job_id")): row for row in new_matrix_rows}
    matrix_delta_rows = [
        new_matrix_by_id[job_id] for job_id in output_ids if job_id in new_matrix_by_id
    ]

    queue_row_validation = []
    for row in output_rows:
        row_checks = queue_row_checks(row, experiment)
        queue_row_validation.append(
            {
                "job_id": str(row.get("job_id")),
                "checks": row_checks,
                "all_passed": all(row_checks.values()),
            }
        )

    actual_design_sha256 = sha256_file(approval_design_path)
    actual_report_sha256 = sha256_file(delta_report_path)
    actual_input_queue_sha256 = sha256_file(input_queue_path)
    actual_output_queue_sha256 = sha256_file(output_queue_path)
    actual_provenance_sha256 = sha256_file(provenance_path)
    design_manifest = dict(approval_design.get("file_sha256") or {})
    manifest_check = verify_file_manifest(project_root, design_manifest)
    provenance_key = relative_manifest_key(provenance_path, project_root)
    screen_metadata = dict(approval_design.get("throughput_screen_delta") or {})
    expected_runtime_identity = dict(approval_design.get("runtime_identity") or {})
    expected_runtime_fingerprint = (
        sha256_json(expected_runtime_identity) if expected_runtime_identity else ""
    )
    live_identity = current_runtime_identity or {}
    live_patch = current_patch or {}
    authorized_gpu_ids = {
        int(value)
        for value in (approval.get("resource_scope") or {}).get("gpu_ids") or ()
    }
    configured_scope = dict(experiment.get("training_scope") or {})
    configured_measurement = dict(experiment.get("measurement") or {})

    result_rows = [
        validate_job_result(
            row,
            results_dir=results_dir,
            authorized_gpu_ids=authorized_gpu_ids,
            expected_approval_design_sha256=actual_design_sha256,
            expected_runtime_identity=expected_runtime_identity,
            expected_runtime_fingerprint=expected_runtime_fingerprint,
            expected_provenance_sha256=actual_provenance_sha256,
        )
        for row in output_rows
    ]
    classifications = Counter(row["classification"] for row in result_rows)

    report_audit_public = {
        "rows": report_input_audit.get("rows") or [],
        "all_passed": report_input_audit.get("all_passed") is True,
    }
    checks = {
        "delta_report_passed": delta_report.get("all_passed") is True,
        "delta_report_checks_recomputed": all_true(delta_report.get("checks")),
        "delta_report_has_no_violations": all_empty(delta_report.get("violations")),
        "delta_report_inputs_still_hash_bound": report_input_audit.get("all_passed")
        is True,
        "input_queue_path_bound": resolve_declared_path(
            report_pending.get("path"), project_root
        )
        == input_queue_path.resolve(),
        "input_queue_sha_bound": report_pending.get("sha256")
        == actual_input_queue_sha256,
        "output_queue_was_written": report_output.get("written") is True,
        "output_queue_path_bound": resolve_declared_path(
            report_output.get("path"), project_root
        )
        == output_queue_path.resolve(),
        "output_queue_sha_bound": report_output.get("sha256")
        == actual_output_queue_sha256,
        "input_and_output_queue_payloads_identical": [
            sha256_json(row) for row in input_rows
        ]
        == [sha256_json(row) for row in output_rows],
        "queue_order_and_report_ids_exact": bool(output_ids)
        and input_ids == output_ids == report_ids == candidate_ids,
        "queue_job_ids_unique": not duplicate_job_ids(output_rows),
        "queue_physical_keys_unique": not duplicate_key_rows(output_rows),
        "queue_rows_match_report_physical_keys": candidate_physical_records
        == output_physical_records,
        "queue_rows_pass_screen_shape": bool(queue_row_validation)
        and all(row["all_passed"] for row in queue_row_validation),
        "queue_rows_match_bound_new_matrix": len(matrix_delta_rows) == len(output_rows)
        and [sha256_json(row) for row in matrix_delta_rows]
        == [sha256_json(row) for row in output_rows],
        "report_candidate_checks_passed": len(report_candidates) == len(output_rows)
        and all(
            row.get("all_passed") is True and all_true(row.get("checks"))
            for row in report_candidates
        ),
        "report_counts_exact": report_output.get("jobs") == len(output_rows)
        and (delta_report.get("counts") or {}).get("pending_jobs") == len(input_rows)
        and (delta_report.get("counts") or {}).get("delta_jobs") == len(output_rows),
        "approval_design_file_bound": approval.get("design_sha256")
        == actual_design_sha256,
        "approval_manifest_valid": manifest_check.get("all_passed") is True,
        "approval_manifest_binds_provenance": provenance_key is not None
        and design_manifest.get(provenance_key) == actual_provenance_sha256,
        "approval_scope_exact": approval.get("approved") is True
        and approval.get("execution_order") == ["throughput_screen_delta"]
        and approval.get("allowed_job_ids") == output_ids
        and approval_design.get("execution_order") == ["throughput_screen_delta"]
        and approval_design.get("allowed_job_ids") == output_ids
        and screen_metadata.get("allowed_job_ids") == output_ids
        and screen_metadata.get("jobs") == len(output_rows),
        "approval_binds_delta_inputs": screen_metadata.get("delta_report_sha256")
        == actual_report_sha256
        and screen_metadata.get("input_queue_sha256") == actual_input_queue_sha256
        and screen_metadata.get("output_queue_sha256") == actual_output_queue_sha256,
        "approval_gpu_scope_matches_experiment": sorted(authorized_gpu_ids)
        == [int(value) for value in configured_scope.get("gpu_ids") or ()]
        == [1, 2, 3, 4]
        and (approval.get("resource_scope") or {}).get("max_gpu_count")
        == configured_scope.get("max_gpu_count")
        == 4
        and (approval.get("resource_scope") or {}).get("allow_gpu_ids_outside_pool")
        is False
        and (approval.get("resource_scope") or {}).get("performance_parallelism")
        == configured_measurement.get("performance_parallelism"),
        "runtime_identity_present_and_fingerprint_bound": bool(
            expected_runtime_identity
        )
        and approval_design.get("runtime_fingerprint_sha256")
        == expected_runtime_fingerprint,
        "live_runtime_identity_matches_approval": live_identity
        == expected_runtime_identity,
        "live_runtime_patch_matches_approval": patch_identity_matches(
            dict(approval_design.get("runtime_patch") or {}), live_patch
        ),
        "every_queue_job_has_one_valid_terminal_result": len(result_rows)
        == len(output_rows)
        and bool(result_rows)
        and all(row["all_passed"] for row in result_rows)
        and sum(classifications.values()) == len(output_rows)
        and set(classifications) <= TERMINAL_CLASSIFICATIONS,
        "no_non_oom_failure_or_incomplete_result": not (
            set(classifications) - TERMINAL_CLASSIFICATIONS
        ),
    }
    return {
        "schema_version": 1,
        "purpose": (
            "Post-run validation for the exact H800 LoRA + ZeRO-3 "
            "throughput-screen delta"
        ),
        "inputs": {
            "delta_report": {
                "path": str(delta_report_path),
                "sha256": actual_report_sha256,
            },
            "input_queue": {
                "path": str(input_queue_path),
                "sha256": actual_input_queue_sha256,
            },
            "output_queue": {
                "path": str(output_queue_path),
                "sha256": actual_output_queue_sha256,
            },
            "approval": {
                "path": str(approval_path),
                "sha256": sha256_file(approval_path),
            },
            "approval_design": {
                "path": str(approval_design_path),
                "sha256": actual_design_sha256,
            },
            "experiment": {
                "path": str(experiment_path),
                "sha256": sha256_file(experiment_path),
            },
            "provenance": {
                "path": str(provenance_path),
                "sha256": actual_provenance_sha256,
            },
            "results_dir": str(results_dir),
        },
        "expected_runtime_identity": expected_runtime_identity,
        "expected_runtime_fingerprint_sha256": expected_runtime_fingerprint,
        "expected_approval_design_sha256": actual_design_sha256,
        "expected_provenance_sha256": actual_provenance_sha256,
        "report_input_audit": report_audit_public,
        "approval_manifest": manifest_check,
        "counts": {
            "queue_jobs": len(output_rows),
            "success": classifications.get("success", 0),
            "oom": classifications.get("oom", 0),
            "other_or_missing": sum(
                count
                for name, count in classifications.items()
                if name not in TERMINAL_CLASSIFICATIONS
            ),
        },
        "queue_row_validation": queue_row_validation,
        "results": result_rows,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def output_paths_are_safe(
    outputs: Iterable[Path],
    protected_inputs: Iterable[Path],
    project_root: Path,
) -> bool:
    resolved_outputs = [path.resolve() for path in outputs]
    protected = {path.resolve() for path in protected_inputs}
    artifact_root = (project_root / "artifacts").resolve()
    return (
        len(resolved_outputs) == len(set(resolved_outputs))
        and all(path.suffix == ".json" for path in resolved_outputs)
        and all(is_within(path, artifact_root) for path in resolved_outputs)
        and not set(resolved_outputs).intersection(protected)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-report", type=Path, default=DELTA_REPORT_PATH)
    parser.add_argument("--input-queue", type=Path, default=INPUT_QUEUE_PATH)
    parser.add_argument("--output-queue", type=Path, default=OUTPUT_QUEUE_PATH)
    parser.add_argument("--approval", type=Path, default=APPROVAL_PATH)
    parser.add_argument("--approval-design", type=Path, default=APPROVAL_DESIGN_PATH)
    parser.add_argument(
        "--experiment", type=Path, default=CONFIG_DIR / "experiment.json"
    )
    parser.add_argument("--provenance", type=Path, default=PROVENANCE_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=VALIDATION_PATH)
    parser.add_argument(
        "--approval-snapshot", type=Path, default=APPROVAL_SNAPSHOT_PATH
    )
    parser.add_argument(
        "--approval-design-snapshot", type=Path, default=DESIGN_SNAPSHOT_PATH
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_files = (
        args.delta_report,
        args.input_queue,
        args.output_queue,
        args.approval,
        args.approval_design,
        args.experiment,
        args.provenance,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise SystemExit(f"Required post-run validation inputs are absent: {missing}")
    outputs = (args.output, args.approval_snapshot, args.approval_design_snapshot)
    if not output_paths_are_safe(outputs, required_files, args.project_root):
        raise SystemExit(
            "Refusing unsafe output paths: reports and snapshots must be distinct "
            "JSON files under the campaign artifacts directory"
        )
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite post-run validation outputs without --overwrite: {existing}"
        )

    validation = validate_screen_delta_results(
        delta_report_path=args.delta_report,
        input_queue_path=args.input_queue,
        output_queue_path=args.output_queue,
        approval_path=args.approval,
        approval_design_path=args.approval_design,
        experiment_path=args.experiment,
        provenance_path=args.provenance,
        results_dir=args.results_dir,
        project_root=args.project_root,
        current_runtime_identity=live_runtime_identity(),
        current_patch=validate_patch(),
    )
    approval = read_json(args.approval)
    approval_design = read_json(args.approval_design)
    write_json(args.approval_snapshot, approval)
    write_json(args.approval_design_snapshot, approval_design)
    validation["approval_snapshot"] = {
        "path": str(args.approval_snapshot),
        "sha256": sha256_file(args.approval_snapshot),
    }
    validation["approval_design_snapshot"] = {
        "path": str(args.approval_design_snapshot),
        "sha256": sha256_file(args.approval_design_snapshot),
    }
    write_json(args.output, validation)
    print(
        json.dumps(
            {
                "all_passed": validation["all_passed"],
                "counts": validation["counts"],
                "output": str(args.output),
                "approval_snapshot": validation["approval_snapshot"],
                "approval_design_snapshot": validation["approval_design_snapshot"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not validation["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
