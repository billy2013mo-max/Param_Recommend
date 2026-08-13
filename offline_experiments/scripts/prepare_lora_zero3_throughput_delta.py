#!/usr/bin/env python3
"""Fail-closed preparation of the post-recovery H800 LoRA ZeRO-3 screen delta.

This tool never launches training and never edits the current approval files or
materialized matrices.  It compares an explicitly saved pre-recovery baseline
with a newly materialized screen matrix, proves that the pending queue contains
exactly the new, recovery-enabled jobs, and writes a narrow queue plus an audit
report suitable for a later approval freeze.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from common import ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json, write_jsonl


PHYSICAL_KEY_FIELDS = (
    "request_id",
    "model_id",
    "train_type",
    "dataset_id",
    "cutoff_len",
    "gpu_count",
    "zero",
    "gc",
    "mbs",
    "target_gbs",
    "packing",
)
FAMILY_MATCH_FIELDS = (
    "model_id",
    "train_type",
    "dataset_id",
    "cutoff_len",
    "gpu_count",
    "zero",
    "gc",
    "packing",
)
EXPECTED_MODELS = {"qwen3_8b", "qwen3_14b"}
TERMINAL_CLASSIFICATIONS = {"success", "oom"}
DEFAULT_MAX_DELTA_JOBS = 18


def physical_key(job: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(job.get(field) for field in PHYSICAL_KEY_FIELDS)


def family_match_key(job: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(job.get(field) for field in FAMILY_MATCH_FIELDS)


def key_record(key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(PHYSICAL_KEY_FIELDS, key))


def duplicate_key_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for row in rows:
        grouped[physical_key(row)].append(str(row.get("job_id")))
    return [
        {"physical_key": key_record(key), "job_ids": job_ids}
        for key, job_ids in grouped.items()
        if len(job_ids) > 1
    ]


def duplicate_job_ids(rows: Iterable[dict[str, Any]]) -> list[str]:
    counts = Counter(str(row.get("job_id")) for row in rows)
    return sorted(job_id for job_id, count in counts.items() if count > 1)


def valid_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def job_shape_errors(job: dict[str, Any]) -> list[str]:
    errors = []
    missing = [field for field in (*PHYSICAL_KEY_FIELDS, "job_id", "kind") if field not in job]
    if missing:
        errors.append(f"missing fields: {missing}")
    if job.get("kind") != "throughput_screen":
        errors.append("kind must be throughput_screen")
    if job.get("fidelity") != "screen":
        errors.append("fidelity must be screen")
    if job.get("model_id") not in EXPECTED_MODELS:
        errors.append("model_id must be qwen3_8b or qwen3_14b")
    if job.get("train_type") != "lora":
        errors.append("train_type must be lora")
    if job.get("zero") != "zero3":
        errors.append("zero must be zero3")
    if not isinstance(job.get("gc"), bool):
        errors.append("gc must be boolean")
    if not isinstance(job.get("packing"), bool):
        errors.append("packing must be boolean")
    for field in ("cutoff_len", "gpu_count", "mbs", "target_gbs"):
        if not valid_positive_int(job.get(field)):
            errors.append(f"{field} must be a positive integer")
    if not isinstance(job.get("request_id"), str) or not job.get("request_id"):
        errors.append("request_id must be a non-empty string")
    if not isinstance(job.get("job_id"), str) or not job.get("job_id"):
        errors.append("job_id must be a non-empty string")
    return errors


def family_shape_errors(family: dict[str, Any]) -> list[str]:
    errors = []
    if not isinstance(family.get("job_id"), str) or not family.get("job_id"):
        errors.append("job_id must be a non-empty string")
    if family.get("kind") != "memory_boundary":
        errors.append("kind must be memory_boundary")
    if family.get("model_id") not in EXPECTED_MODELS:
        errors.append("model_id must be qwen3_8b or qwen3_14b")
    if family.get("train_type") != "lora":
        errors.append("train_type must be lora")
    if family.get("zero") != "zero3":
        errors.append("zero must be zero3")
    if family.get("gpu_count") not in {2, 4}:
        errors.append("gpu_count must be 2 or 4")
    if family.get("packing") is not False:
        errors.append("packing must be false")
    if not isinstance(family.get("gc"), bool):
        errors.append("gc must be boolean")
    if not isinstance(family.get("dataset_id"), str) or not family.get("dataset_id"):
        errors.append("dataset_id must be a non-empty string")
    for field in ("cutoff_len", "target_gbs"):
        if not valid_positive_int(family.get(field)):
            errors.append(f"{field} must be a positive integer")
    candidates = family.get("mbs_candidates") or []
    if (
        not candidates
        or not all(valid_positive_int(value) for value in candidates)
        or candidates != sorted(set(candidates))
    ):
        errors.append("mbs_candidates must be unique, positive, and strictly increasing")
    return errors


def validate_boundary(family: dict[str, Any], summary: dict[str, Any]) -> dict[str, bool]:
    candidates = [int(value) for value in family.get("mbs_candidates") or ()]
    trials = list(summary.get("trials") or ())
    trial_mbs = [int(trial.get("mbs") or 0) for trial in trials]
    classifications = [str(trial.get("classification")) for trial in trials]
    first_oom_index = next(
        (index for index, classification in enumerate(classifications) if classification == "oom"),
        None,
    )
    expected_prefix = candidates if first_oom_index is None else candidates[: first_oom_index + 1]
    successful = [
        mbs
        for mbs, classification in zip(trial_mbs, classifications)
        if classification == "success"
    ]
    expected_max = max(successful) if successful else None
    expected_failed = trial_mbs[first_oom_index] if first_oom_index is not None else None
    return {
        "family_id_matches": summary.get("family_job_id") == family.get("job_id"),
        "trials_present": bool(trials),
        "classifications_valid": all(
            classification in TERMINAL_CLASSIFICATIONS for classification in classifications
        ),
        "trial_mbs_is_candidate_prefix": trial_mbs == expected_prefix,
        "stopped_at_first_oom_or_exhausted": first_oom_index is not None
        or trial_mbs == candidates,
        "max_feasible_matches_trials": summary.get("max_feasible_mbs") == expected_max,
        "first_failed_matches_trials": summary.get("first_failed_mbs") == expected_failed,
    }


def resolve_bound_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_recovery_inputs(
    recovery_plan_path: Path,
    project_root: Path,
    recovery_families_path: Path | None = None,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]], dict[str, bool]]:
    plan = read_json(recovery_plan_path)
    declared_path = resolve_bound_path(str(plan.get("queue_path") or ""), project_root)
    families_path = recovery_families_path or declared_path
    families = read_jsonl(families_path)
    family_ids = [str(family.get("job_id")) for family in families]
    expected_ids = [str(value) for value in plan.get("family_job_ids") or ()]
    purpose = str(plan.get("purpose") or "").lower()
    checks = {
        "recovery_plan_passed": plan.get("all_passed") is True,
        "recovery_plan_is_h800_lora_zero3": "h800" in purpose
        and "lora" in purpose
        and ("zero-3" in purpose or "zero3" in purpose),
        "exactly_twenty_recovery_families": len(expected_ids) == 20,
        "recovery_plan_family_ids_unique": len(expected_ids) == len(set(expected_ids)),
        "recovery_queue_matches_plan_order": family_ids == expected_ids,
        "recovery_queue_sha_matches_plan": sha256_file(families_path)
        == plan.get("queue_sha256"),
    }
    return plan, families_path, families, checks


def load_boundary_evidence(
    evidence_path: Path,
    families: list[dict[str, Any]],
    project_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, bool]]:
    family_ids = [str(family["job_id"]) for family in families]
    summaries: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str | None] = {}
    manifest_checks = {
        "boundary_evidence_validation_passed": True,
        "boundary_evidence_rows_passed": True,
        "boundary_evidence_hashes_match": True,
    }
    mode = "directory"
    evidence_document_sha256 = None

    if evidence_path.is_dir():
        for family_id in family_ids:
            summary_path = evidence_path / f"{family_id}.json"
            if summary_path.is_file():
                summaries[family_id] = read_json(summary_path)
                hashes[family_id] = sha256_file(summary_path)
            else:
                hashes[family_id] = None
    else:
        mode = "validated_manifest"
        document = read_json(evidence_path)
        evidence_document_sha256 = sha256_file(evidence_path)
        rows = list(document.get("families") or ())
        row_ids = [str(row.get("family_job_id")) for row in rows]
        manifest_checks["boundary_evidence_validation_passed"] = document.get("all_passed") is True
        manifest_checks["boundary_evidence_rows_passed"] = (
            row_ids == family_ids and all(row.get("all_passed") is True for row in rows)
        )
        for row in rows:
            family_id = str(row.get("family_job_id"))
            summary_value = str(row.get("summary_path") or "")
            summary_path = resolve_bound_path(summary_value, project_root)
            actual_hash = sha256_file(summary_path) if summary_path.is_file() else None
            expected_hash = row.get("summary_sha256")
            hashes[family_id] = actual_hash
            if actual_hash is None or actual_hash != expected_hash:
                manifest_checks["boundary_evidence_hashes_match"] = False
                continue
            summaries[family_id] = read_json(summary_path)

    audit = {
        "mode": mode,
        "path": str(evidence_path),
        "sha256": evidence_document_sha256,
        "summary_sha256": hashes,
    }
    return summaries, audit, manifest_checks


def load_historical_throughput(
    results_dir: Path,
) -> tuple[dict[tuple[Any, ...], list[dict[str, Any]]], list[dict[str, str]]]:
    history: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    errors = []
    for rendered_path in sorted(results_dir.glob("*/rendered_run.json")):
        try:
            rendered = read_json(rendered_path)
            job = dict(rendered.get("job") or {})
        except (OSError, ValueError, TypeError) as exc:
            errors.append({"path": str(rendered_path), "error": str(exc)})
            continue
        if job.get("kind") not in {"throughput_screen", "throughput"}:
            continue
        missing = [field for field in PHYSICAL_KEY_FIELDS if field not in job]
        if missing:
            errors.append({"path": str(rendered_path), "error": f"missing fields: {missing}"})
            continue
        status_path = rendered_path.parent / "status.json"
        classification = None
        if status_path.is_file():
            try:
                classification = read_json(status_path).get("classification")
            except (OSError, ValueError, TypeError) as exc:
                errors.append({"path": str(status_path), "error": str(exc)})
                continue
        history[physical_key(job)].append(
            {
                "job_id": str(job.get("job_id")),
                "kind": str(job.get("kind")),
                "classification": classification,
                "rendered_run_path": str(rendered_path),
            }
        )
    return history, errors


def validate_delta(
    *,
    baseline_path: Path,
    new_matrix_path: Path,
    pending_queue_path: Path,
    recovery_plan_path: Path,
    boundary_evidence_path: Path,
    results_dir: Path,
    project_root: Path = ROOT,
    recovery_families_path: Path | None = None,
    max_delta_jobs: int = DEFAULT_MAX_DELTA_JOBS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = read_jsonl(baseline_path)
    new_matrix = read_jsonl(new_matrix_path)
    pending = read_jsonl(pending_queue_path)
    plan, families_path, families, recovery_checks = load_recovery_inputs(
        recovery_plan_path,
        project_root,
        recovery_families_path,
    )
    boundaries, boundary_audit, evidence_checks = load_boundary_evidence(
        boundary_evidence_path,
        families,
        project_root,
    )
    history, history_errors = load_historical_throughput(results_dir)

    baseline_duplicates = duplicate_key_rows(baseline)
    new_duplicates = duplicate_key_rows(new_matrix)
    pending_duplicates = duplicate_key_rows(pending)
    baseline_job_id_duplicates = duplicate_job_ids(baseline)
    new_job_id_duplicates = duplicate_job_ids(new_matrix)
    pending_job_id_duplicates = duplicate_job_ids(pending)

    baseline_keys = {physical_key(row) for row in baseline}
    new_by_key = {physical_key(row): row for row in new_matrix}
    delta_keys = [physical_key(row) for row in new_matrix if physical_key(row) not in baseline_keys]
    delta_rows = [new_by_key[key] for key in delta_keys]
    pending_keys = [physical_key(row) for row in pending]
    pending_by_key = {physical_key(row): row for row in pending}

    expected_family_ids = [str(value) for value in plan.get("family_job_ids") or ()]
    family_ids = [str(family.get("job_id")) for family in families]
    family_errors = {
        str(family.get("job_id")): family_shape_errors(family)
        for family in families
        if family_shape_errors(family)
    }
    family_signature_index: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for family in families:
        family_signature_index[family_match_key(family)].append(family)

    boundary_rows = {}
    for family in families:
        family_id = str(family["job_id"])
        summary = boundaries.get(family_id)
        checks = (
            validate_boundary(family, summary)
            if isinstance(summary, dict)
            else {"boundary_summary_present": False}
        )
        boundary_rows[family_id] = {
            "checks": checks,
            "max_feasible_mbs": summary.get("max_feasible_mbs") if summary else None,
            "all_passed": all(checks.values()),
        }

    candidate_rows = []
    shape_error_rows = []
    history_hits = []
    terminal_status_hits = []
    terminal_status_read_errors = []
    for job in delta_rows:
        job_id = str(job.get("job_id"))
        shape_errors = job_shape_errors(job)
        if shape_errors:
            shape_error_rows.append({"job_id": job_id, "errors": shape_errors})
        sources = family_signature_index.get(family_match_key(job), [])
        source = sources[0] if len(sources) == 1 else None
        source_id = str(source["job_id"]) if source else None
        boundary = boundaries.get(source_id or "")
        max_feasible = boundary.get("max_feasible_mbs") if boundary else None
        divisibility_valid = bool(
            valid_positive_int(job.get("target_gbs"))
            and valid_positive_int(job.get("gpu_count"))
            and valid_positive_int(job.get("mbs"))
            and int(job["target_gbs"]) % (int(job["gpu_count"]) * int(job["mbs"])) == 0
        )
        mbs_within_boundary = bool(
            valid_positive_int(job.get("mbs"))
            and valid_positive_int(max_feasible)
            and int(job["mbs"]) <= int(max_feasible)
        )
        prior = history.get(physical_key(job), [])
        if prior:
            history_hits.append(
                {"job_id": job_id, "physical_key": key_record(physical_key(job)), "history": prior}
            )
        status_path = results_dir / job_id / "status.json"
        classification = None
        if status_path.is_file():
            try:
                classification = read_json(status_path).get("classification")
            except (OSError, ValueError, TypeError) as exc:
                terminal_status_read_errors.append(
                    {"job_id": job_id, "path": str(status_path), "error": str(exc)}
                )
        if classification in TERMINAL_CLASSIFICATIONS:
            terminal_status_hits.append(
                {
                    "job_id": job_id,
                    "classification": classification,
                    "status_path": str(status_path),
                    "status_sha256": sha256_file(status_path),
                }
            )
        candidate_checks = {
            "shape_valid": not shape_errors,
            "not_in_pre_recovery_baseline": physical_key(job) not in baseline_keys,
            "maps_to_exactly_one_recovery_family": len(sources) == 1,
            "source_is_in_recovery_plan": source_id in set(expected_family_ids),
            "source_boundary_valid": bool(source_id)
            and boundary_rows.get(source_id, {}).get("all_passed") is True,
            "mbs_within_recovered_boundary": mbs_within_boundary,
            "target_gbs_divisible": divisibility_valid,
            "not_in_historical_screen_or_formal": not prior,
            "no_existing_success_or_oom": classification not in TERMINAL_CLASSIFICATIONS,
        }
        candidate_rows.append(
            {
                "job_id": job_id,
                "physical_key": key_record(physical_key(job)),
                "source_family_job_id": source_id,
                "max_feasible_mbs": max_feasible,
                "gradient_accumulation_steps": (
                    int(job["target_gbs"]) // (int(job["gpu_count"]) * int(job["mbs"]))
                    if divisibility_valid
                    else None
                ),
                "checks": candidate_checks,
                "all_passed": all(candidate_checks.values()),
            }
        )

    pending_row_mismatches = []
    for key in delta_keys:
        matrix_row = new_by_key[key]
        pending_row = pending_by_key.get(key)
        if pending_row is None or sha256_json(pending_row) != sha256_json(matrix_row):
            pending_row_mismatches.append(
                {
                    "physical_key": key_record(key),
                    "matrix_job_id": matrix_row.get("job_id"),
                    "pending_job_id": pending_row.get("job_id") if pending_row else None,
                }
            )

    checks = {
        **recovery_checks,
        **evidence_checks,
        "recovery_queue_has_expected_ids": family_ids == expected_family_ids,
        "all_recovery_family_shapes_valid": not family_errors,
        "recovery_family_signatures_unique": all(
            len(rows) == 1 for rows in family_signature_index.values()
        ),
        "all_twenty_boundaries_present_and_valid": len(boundary_rows) == 20
        and all(row["all_passed"] for row in boundary_rows.values()),
        "baseline_physical_keys_unique": not baseline_duplicates,
        "baseline_job_ids_unique": not baseline_job_id_duplicates,
        "new_matrix_physical_keys_unique": not new_duplicates,
        "new_matrix_job_ids_unique": not new_job_id_duplicates,
        "pending_physical_keys_unique": not pending_duplicates,
        "pending_job_ids_unique": not pending_job_id_duplicates,
        "delta_is_nonempty": bool(delta_rows),
        "delta_within_safety_cap": 0 < len(delta_rows) <= max_delta_jobs,
        "pending_keys_exactly_match_delta_in_order": pending_keys == delta_keys,
        "pending_rows_exactly_match_new_matrix": not pending_row_mismatches,
        "all_delta_candidates_pass": bool(candidate_rows)
        and all(row["all_passed"] for row in candidate_rows),
        "historical_results_readable": not history_errors,
        "candidate_statuses_readable": not terminal_status_read_errors,
        "no_historical_screen_or_formal_keys": not history_hits,
        "no_existing_success_or_oom": not terminal_status_hits,
    }
    source_family_ids = sorted(
        {str(row["source_family_job_id"]) for row in candidate_rows if row["source_family_job_id"]}
    )
    report = {
        "schema_version": 1,
        "purpose": "Prepare only newly enabled H800 LoRA + ZeRO-3 throughput-screen jobs after memory recovery",
        "physical_key_fields": list(PHYSICAL_KEY_FIELDS),
        "inputs": {
            "baseline": {"path": str(baseline_path), "sha256": sha256_file(baseline_path)},
            "new_matrix": {"path": str(new_matrix_path), "sha256": sha256_file(new_matrix_path)},
            "pending_queue": {
                "path": str(pending_queue_path),
                "sha256": sha256_file(pending_queue_path),
            },
            "recovery_plan": {
                "path": str(recovery_plan_path),
                "sha256": sha256_file(recovery_plan_path),
            },
            "recovery_families": {
                "path": str(families_path),
                "sha256": sha256_file(families_path),
            },
            "boundary_evidence": boundary_audit,
            "results_dir": str(results_dir),
        },
        "counts": {
            "baseline_jobs": len(baseline),
            "new_matrix_jobs": len(new_matrix),
            "removed_baseline_physical_keys": len(baseline_keys - set(new_by_key)),
            "pending_jobs": len(pending),
            "delta_jobs": len(delta_rows),
            "recovery_families": len(families),
            "source_recovery_families": len(source_family_ids),
            "historical_screen_or_formal_keys": len(history),
            "max_delta_jobs": max_delta_jobs,
        },
        "checks": checks,
        "violations": {
            "baseline_duplicate_physical_keys": baseline_duplicates,
            "baseline_duplicate_job_ids": baseline_job_id_duplicates,
            "new_matrix_duplicate_physical_keys": new_duplicates,
            "new_matrix_duplicate_job_ids": new_job_id_duplicates,
            "pending_duplicate_physical_keys": pending_duplicates,
            "pending_duplicate_job_ids": pending_job_id_duplicates,
            "recovery_family_shape_errors": family_errors,
            "pending_row_mismatches": pending_row_mismatches,
            "delta_shape_errors": shape_error_rows,
            "historical_result_read_errors": history_errors,
            "terminal_status_read_errors": terminal_status_read_errors,
            "historical_key_hits": history_hits,
            "terminal_status_hits": terminal_status_hits,
        },
        "boundary_validation": boundary_rows,
        "delta_candidates": candidate_rows,
        "freeze_preparation": {
            "allowed_job_ids": [str(row["job_id"]) for row in delta_rows],
            "source_family_job_ids": source_family_ids,
            "policy": (
                "Run exactly this delta queue; accept success/OOM; any other failure blocks; "
                "do not rerun any baseline or historical physical key."
            ),
        },
        "all_passed": all(checks.values()),
    }
    return report, delta_rows


def output_paths_are_safe(
    report_path: Path,
    output_queue_path: Path,
    input_paths: Iterable[Path],
    project_root: Path,
) -> bool:
    outputs = {report_path.resolve(), output_queue_path.resolve()}
    protected = {path.resolve() for path in input_paths}
    protected.update(
        {
            (project_root / "config" / "APPROVED_TO_RUN.json").resolve(),
            (project_root / "runtime" / "approval_design.json").resolve(),
        }
    )
    return len(outputs) == 2 and outputs.isdisjoint(protected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--new-matrix", type=Path, required=True)
    parser.add_argument("--pending-queue", type=Path, required=True)
    parser.add_argument("--recovery-plan", type=Path, required=True)
    parser.add_argument("--recovery-families", type=Path)
    parser.add_argument("--boundary-evidence", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-queue", type=Path, required=True)
    parser.add_argument("--max-delta-jobs", type=int, default=DEFAULT_MAX_DELTA_JOBS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [
        args.baseline,
        args.new_matrix,
        args.pending_queue,
        args.recovery_plan,
        args.boundary_evidence,
        args.results_dir,
    ]
    if args.recovery_families:
        input_paths.append(args.recovery_families)
    if not output_paths_are_safe(args.report, args.output_queue, input_paths, args.project_root):
        raise SystemExit("Refusing unsafe output paths: outputs must be distinct from all inputs and approvals")
    existing = [path for path in (args.report, args.output_queue) if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing outputs without --overwrite: {existing}")
    if args.max_delta_jobs <= 0:
        raise SystemExit("--max-delta-jobs must be positive")

    report, delta_rows = validate_delta(
        baseline_path=args.baseline,
        new_matrix_path=args.new_matrix,
        pending_queue_path=args.pending_queue,
        recovery_plan_path=args.recovery_plan,
        recovery_families_path=args.recovery_families,
        boundary_evidence_path=args.boundary_evidence,
        results_dir=args.results_dir,
        project_root=args.project_root,
        max_delta_jobs=args.max_delta_jobs,
    )
    if report["all_passed"]:
        write_jsonl(args.output_queue, delta_rows)
        report["output_queue"] = {
            "path": str(args.output_queue),
            "sha256": sha256_file(args.output_queue),
            "jobs": len(delta_rows),
            "written": True,
        }
    else:
        report["output_queue"] = {
            "path": str(args.output_queue),
            "jobs": 0,
            "written": False,
        }
    write_json(args.report, report)
    print(
        json.dumps(
            {
                "all_passed": report["all_passed"],
                "delta_jobs": len(delta_rows),
                "report": str(args.report),
                "output_queue": report["output_queue"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
