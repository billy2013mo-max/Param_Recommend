#!/usr/bin/env python3
"""Validate the completed H800 LoRA + ZeRO-3 memory-boundary recovery."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
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
from freeze_lora_zero3_fix import VALIDATION_PATH as CANARY_VALIDATION_PATH
from freeze_lora_zero3_fix import validate_patch
from freeze_lora_zero3_recovery import (
    QUEUE_PATH,
    VALIDATION_PATH as RECOVERY_PLAN_PATH,
    validate_family_shape,
)
from run_job import OOM_PATTERNS
from run_pipeline import memory_progress


APPROVAL_PATH = ROOT / "config" / "APPROVED_TO_RUN.json"
APPROVAL_DESIGN_PATH = RUNTIME_DIR / "approval_design.json"
APPROVAL_SNAPSHOT_PATH = ARTIFACT_DIR / "h800_lora_zero3_recovery_approval.json"
DESIGN_SNAPSHOT_PATH = ARTIFACT_DIR / "h800_lora_zero3_recovery_approval_design.json"
VALIDATION_PATH = ARTIFACT_DIR / "h800_lora_zero3_recovery_results_validation.json"
DTYPE_ERROR = "output tensor must have the same type as input tensor"
FP32_LORA_MESSAGE = "DeepSpeed ZeRO3 detected, remaining trainable params in float32"
EXPECTED_STEPS = 5
FRESHNESS_TOLERANCE_SECONDS = 2.0
TRIAL_SHAPE_FIELDS = (
    "model_id",
    "dataset_id",
    "cutoff_len",
    "train_type",
    "zero",
    "gc",
    "gpu_count",
    "target_gbs",
    "packing",
)


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


def validate_boundary(family: dict[str, Any], summary: dict[str, Any]) -> dict[str, bool]:
    candidates = [int(value) for value in family.get("mbs_candidates") or ()]
    trials = list(summary.get("trials") or ())
    trial_mbs = [int(trial.get("mbs") or 0) for trial in trials]
    classifications = [str(trial.get("classification")) for trial in trials]
    allowed = {"success", "oom"}
    first_oom_index = next(
        (index for index, value in enumerate(classifications) if value == "oom"),
        None,
    )
    expected_mbs = candidates if first_oom_index is None else candidates[: first_oom_index + 1]
    successful_mbs = [
        mbs
        for mbs, classification in zip(trial_mbs, classifications)
        if classification == "success"
    ]
    expected_max = max(successful_mbs) if successful_mbs else None
    expected_first_failed = trial_mbs[first_oom_index] if first_oom_index is not None else None
    return {
        "family_id_matches": summary.get("family_job_id") == family.get("job_id"),
        "trials_present": bool(trials),
        "classifications_are_success_or_oom": all(value in allowed for value in classifications),
        "candidate_prefix_is_contiguous": trial_mbs == expected_mbs,
        "stops_at_first_oom_or_exhausts_candidates": (
            first_oom_index is not None or trial_mbs == candidates
        ),
        "max_feasible_mbs_matches_trials": summary.get("max_feasible_mbs") == expected_max,
        "first_failed_mbs_matches_trials": summary.get("first_failed_mbs") == expected_first_failed,
    }


def validate_family_definition(family: dict[str, Any]) -> dict[str, bool]:
    candidates = [int(value) for value in family.get("mbs_candidates") or ()]
    return {
        "expected_family_shape": validate_family_shape(family),
        "mbs_candidates_positive": bool(candidates) and all(value > 0 for value in candidates),
        "mbs_candidates_unique": len(candidates) == len(set(candidates)),
        "mbs_candidates_strictly_increasing": candidates == sorted(candidates)
        and all(left < right for left, right in zip(candidates, candidates[1:])),
    }


def validate_runtime_chain(canary: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, bool]]:
    identity = dict(canary.get("runtime_identity") or {})
    computed_fingerprint = sha256_json(identity) if identity else ""
    row_fingerprints = {
        str(row.get("runtime_fingerprint_sha256"))
        for row in canary.get("results", {}).get("rows", ())
        if row.get("runtime_fingerprint_sha256")
    }
    checks = {
        "runtime_identity_present": bool(identity),
        "three_canary_rows_present": len(canary.get("results", {}).get("rows", ())) == 3,
        "canary_fingerprints_unique": row_fingerprints == {computed_fingerprint},
    }
    return identity, computed_fingerprint, checks


def expected_rendered_shape(family: dict[str, Any], trial_job_id: str, mbs: int) -> dict[str, Any]:
    return {
        "job_id": trial_job_id,
        "family_job_id": family["job_id"],
        "kind": "memory_probe",
        "mbs": mbs,
        "mbs_candidates": family["mbs_candidates"],
        "warmup_steps": 0,
        "max_steps": EXPECTED_STEPS,
        **{field: family.get(field) for field in TRIAL_SHAPE_FIELDS},
    }


def artifact_hashes(paths: dict[str, Path]) -> dict[str, str | None]:
    return {name: sha256_file(path) if path.is_file() else None for name, path in paths.items()}


def validate_trial(
    family: dict[str, Any],
    trial: dict[str, Any],
    *,
    summary_gpu_mask: Any,
    authorized_gpu_ids: set[int],
    expected_approval_sha256: str,
    expected_runtime_fingerprint: str,
    expected_runtime_identity: dict[str, Any],
    expected_provenance_sha256: str,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    mbs = int(trial["mbs"])
    trial_job_id = f"{family['job_id']}-mbs{mbs}"
    result_dir = results_dir / trial_job_id
    paths = {
        "status": result_dir / "status.json",
        "rendered_run": result_dir / "rendered_run.json",
        "runtime_identity": result_dir / "runtime_identity.json",
        "train_log": result_dir / "train.log",
    }
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        return {
            "job_id": trial_job_id,
            "classification": trial.get("classification"),
            "missing": missing,
            "artifact_sha256": artifact_hashes(paths),
            "checks": {"required_artifacts_present": False},
            "all_passed": False,
        }

    status = read_json(paths["status"])
    rendered = read_json(paths["rendered_run"])
    rendered_job = dict(rendered.get("job") or {})
    runtime_identity = read_json(paths["runtime_identity"])
    log = paths["train_log"].read_text(encoding="utf-8", errors="replace")
    expected_classification = str(trial.get("classification"))
    expected_shape = expected_rendered_shape(family, trial_job_id, mbs)
    status_gpu_mask = parse_gpu_mask(status.get("gpu_mask"))
    rendered_gpu_mask = parse_gpu_mask(rendered.get("gpu_mask"))
    boundary_gpu_mask = parse_gpu_mask(summary_gpu_mask)
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
            abs_tol=max(2.0, float(wall_seconds) * 0.02),
        )
    )
    computed_runtime_fingerprint = sha256_json(runtime_identity)
    checks: dict[str, bool] = {
        "required_artifacts_present": True,
        "status_job_id_matches": status.get("job_id") == trial_job_id,
        "classification_matches_boundary": status.get("classification") == expected_classification,
        "rendered_job_shape_matches": all(
            rendered_job.get(field) == value for field, value in expected_shape.items()
        ),
        "gpu_mask_matches_all_artifacts": bool(boundary_gpu_mask)
        and status_gpu_mask == rendered_gpu_mask == boundary_gpu_mask,
        "gpu_mask_matches_scope": len(boundary_gpu_mask) == int(family["gpu_count"])
        and len(boundary_gpu_mask) == len(set(boundary_gpu_mask))
        and set(boundary_gpu_mask) <= authorized_gpu_ids,
        "timestamps_present_and_consistent": wall_consistent,
        "artifacts_fresh_for_attempt": bool(
            timestamps_numeric
            and all(
                path.stat().st_mtime + FRESHNESS_TOLERANCE_SECONDS >= float(started)
                for path in paths.values()
            )
        ),
        "approval_design_bound": status.get("approval_design_sha256")
        == expected_approval_sha256,
        "provenance_bound": status.get("provenance_sha256")
        == rendered.get("provenance_sha256")
        == expected_provenance_sha256,
        "runtime_fingerprint_recomputed": computed_runtime_fingerprint
        == expected_runtime_fingerprint,
        "runtime_fingerprint_bound": status.get("runtime_fingerprint_sha256")
        == rendered.get("runtime_fingerprint_sha256")
        == computed_runtime_fingerprint,
        "runtime_identity_matches_canary": runtime_identity == expected_runtime_identity,
        "dtype_failure_absent": DTYPE_ERROR not in log,
        "keyboard_interrupt_absent": "KeyboardInterrupt" not in log,
    }

    train_loss: float | int | None = None
    extra_paths: dict[str, Path] = {}
    if expected_classification == "success":
        summary_paths = sorted((result_dir / "metrics").glob("summary.rank*.json"))
        summaries = [read_json(path) for path in summary_paths]
        train_results_path = result_dir / "trainer_output" / "train_results.json"
        extra_paths = {
            **{f"summary_rank_{index}": path for index, path in enumerate(summary_paths)},
            "train_results": train_results_path,
        }
        train_results = read_json(train_results_path) if train_results_path.is_file() else {}
        train_loss = train_results.get("train_loss")
        ranks = [summary.get("rank") for summary in summaries]
        expected_ranks = list(range(int(family["gpu_count"])))
        metadata_checks = []
        metric_values_valid = []
        for summary in summaries:
            metadata = dict(summary.get("metadata") or {})
            metadata_checks.append(
                metadata.get("job_id") == trial_job_id
                and metadata.get("family_job_id") == family["job_id"]
                and metadata.get("mbs") == mbs
                and all(
                    metadata.get(field) == family.get(field)
                    for field in TRIAL_SHAPE_FIELDS
                )
            )
            values = (
                summary.get("computed_tokens_per_second"),
                summary.get("logical_samples_per_second"),
                summary.get("max_allocated"),
                summary.get("max_reserved"),
            )
            metric_values_valid.append(
                all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) > 0
                    for value in values
                )
            )
        checks.update(
            {
                "return_code_zero": status.get("return_code") == 0,
                "fp32_lora_preserved": FP32_LORA_MESSAGE in log,
                "rank_set_complete": ranks == expected_ranks,
                "world_size_and_local_rank_match": len(summaries) == len(expected_ranks)
                and all(
                    summary.get("world_size") == int(family["gpu_count"])
                    and summary.get("local_rank") == rank
                    for rank, summary in enumerate(summaries)
                ),
                "metric_metadata_matches_job": bool(metadata_checks) and all(metadata_checks),
                "exactly_five_optimizer_steps": bool(summaries)
                and all(
                    summary.get("failure") is None
                    and summary.get("total_steps") == EXPECTED_STEPS
                    and summary.get("measured_steps") == EXPECTED_STEPS
                    for summary in summaries
                ),
                "summary_metrics_finite_positive": bool(metric_values_valid)
                and all(metric_values_valid),
                "finite_loss": isinstance(train_loss, (int, float))
                and not isinstance(train_loss, bool)
                and math.isfinite(float(train_loss)),
                "success_artifacts_fresh": bool(
                    timestamps_numeric
                    and train_results_path.is_file()
                    and summary_paths
                    and all(
                        path.stat().st_mtime + FRESHNESS_TOLERANCE_SECONDS >= float(started)
                        for path in (*summary_paths, train_results_path)
                    )
                ),
            }
        )
    elif expected_classification == "oom":
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

    all_paths = {**paths, **extra_paths}
    return {
        "job_id": trial_job_id,
        "classification": expected_classification,
        "gpu_mask": boundary_gpu_mask,
        "train_loss": train_loss,
        "missing": [],
        "artifact_sha256": artifact_hashes(all_paths),
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def main() -> None:
    recovery_plan = read_json(RECOVERY_PLAN_PATH)
    canary = read_json(CANARY_VALIDATION_PATH)
    approval = read_json(APPROVAL_PATH)
    approval_design = read_json(APPROVAL_DESIGN_PATH)
    families = read_jsonl(QUEUE_PATH)
    expected_family_ids = list(recovery_plan.get("family_job_ids") or ())
    family_ids = [str(family.get("job_id")) for family in families]
    concrete_job_ids = [
        f"{family['job_id']}-mbs{int(mbs)}"
        for family in families
        for mbs in family.get("mbs_candidates") or ()
    ]
    runtime_identity, expected_runtime_fingerprint, runtime_checks = validate_runtime_chain(canary)
    actual_approval_sha256 = sha256_file(APPROVAL_DESIGN_PATH)
    design_manifest = dict(approval_design.get("file_sha256") or {})
    manifest_check = verify_file_manifest(ROOT, design_manifest)
    authorized_gpu_ids = {
        int(value) for value in approval.get("resource_scope", {}).get("gpu_ids") or ()
    }
    expected_provenance_sha256 = str(design_manifest.get("artifacts/provenance.json") or "")
    family_definition_rows = {
        str(family["job_id"]): validate_family_definition(family) for family in families
    }

    family_rows = []
    for family in families:
        family_id = str(family["job_id"])
        summary_path = RESULTS_DIR / "boundary_summaries" / f"{family_id}.json"
        if not summary_path.is_file():
            family_rows.append(
                {
                    "family_job_id": family_id,
                    "summary_path": str(summary_path.relative_to(ROOT)),
                    "checks": {"boundary_summary_present": False},
                    "trials": [],
                    "all_passed": False,
                }
            )
            continue
        summary = read_json(summary_path)
        boundary_checks = {"boundary_summary_present": True, **validate_boundary(family, summary)}
        trial_rows = [
            validate_trial(
                family,
                trial,
                summary_gpu_mask=summary.get("gpu_mask"),
                authorized_gpu_ids=authorized_gpu_ids,
                expected_approval_sha256=actual_approval_sha256,
                expected_runtime_fingerprint=expected_runtime_fingerprint,
                expected_runtime_identity=runtime_identity,
                expected_provenance_sha256=expected_provenance_sha256,
            )
            for trial in summary.get("trials") or ()
        ]
        family_rows.append(
            {
                "family_job_id": family_id,
                "summary_path": str(summary_path.relative_to(ROOT)),
                "summary_sha256": sha256_file(summary_path),
                "checks": boundary_checks,
                "trials": trial_rows,
                "all_passed": all(boundary_checks.values())
                and bool(trial_rows)
                and all(row["all_passed"] for row in trial_rows),
            }
        )

    progress = memory_progress()
    patch = validate_patch()
    expected_upstream_fix = canary.get("upstream_fix")
    checks = {
        "recovery_plan_was_valid": recovery_plan.get("all_passed") is True,
        "canaries_were_valid": canary.get("all_passed") is True,
        **runtime_checks,
        "runtime_patch_still_valid": patch.get("all_passed") is True,
        "exactly_twenty_expected_families": len(expected_family_ids) == 20,
        "family_ids_unique": len(family_ids) == len(set(family_ids)),
        "queue_matches_frozen_family_ids": family_ids == expected_family_ids,
        "all_family_definitions_valid": all(
            all(row.values()) for row in family_definition_rows.values()
        ),
        "queue_sha_matches_recovery_plan": sha256_file(QUEUE_PATH)
        == recovery_plan.get("queue_sha256"),
        "concrete_ids_match_recovery_plan": concrete_job_ids
        == recovery_plan.get("potential_concrete_job_ids"),
        "approval_design_file_bound": approval.get("design_sha256")
        == actual_approval_sha256,
        "approval_manifest_valid": manifest_check.get("all_passed") is True,
        "approval_scope_matches_recovery": approval.get("approved") is True
        and approval.get("execution_order") == ["memory_boundary_recovery"]
        and approval.get("allowed_job_ids") == concrete_job_ids
        and approval_design.get("allowed_job_ids") == concrete_job_ids
        and approval_design.get("allowed_family_job_ids") == expected_family_ids,
        "approval_metadata_matches_recovery": approval_design.get("design_purpose")
        == recovery_plan.get("purpose")
        and approval_design.get("runtime_fix") == expected_upstream_fix
        and approval.get("runtime_fix") == expected_upstream_fix
        and set(approval_design.get("authorized_gpu_ids") or ()) == authorized_gpu_ids
        == {1, 2, 3, 4},
        "approval_manifest_binds_recovery_evidence": design_manifest.get(
            str(QUEUE_PATH.relative_to(ROOT))
        )
        == sha256_file(QUEUE_PATH)
        and design_manifest.get(str(CANARY_VALIDATION_PATH.relative_to(ROOT)))
        == sha256_file(CANARY_VALIDATION_PATH)
        and design_manifest.get(str(RECOVERY_PLAN_PATH.relative_to(ROOT)))
        == sha256_file(RECOVERY_PLAN_PATH),
        "all_memory_families_summarized": progress.get("summarized")
        == progress.get("total")
        == 186,
        "no_missing_memory_families": not progress.get("missing"),
        "no_excluded_memory_families": not progress.get("excluded"),
        "all_recovery_families_valid": len(family_rows) == 20
        and all(row["all_passed"] for row in family_rows),
    }

    write_json(APPROVAL_SNAPSHOT_PATH, approval)
    write_json(DESIGN_SNAPSHOT_PATH, approval_design)
    validation = {
        "schema_version": 2,
        "purpose": "Post-run validation for the 20 H800 LoRA + ZeRO-3 recovered memory families",
        "expected_approval_design_sha256": actual_approval_sha256,
        "expected_runtime_fingerprint_sha256": expected_runtime_fingerprint,
        "approval_snapshot_path": str(APPROVAL_SNAPSHOT_PATH.relative_to(ROOT)),
        "approval_snapshot_sha256": sha256_file(APPROVAL_SNAPSHOT_PATH),
        "approval_design_snapshot_path": str(DESIGN_SNAPSHOT_PATH.relative_to(ROOT)),
        "approval_design_snapshot_sha256": sha256_file(DESIGN_SNAPSHOT_PATH),
        "approval_manifest": manifest_check,
        "checks": checks,
        "family_definition_checks": family_definition_rows,
        "memory_progress": {
            "total": progress.get("total"),
            "summarized": progress.get("summarized"),
            "missing": len(progress.get("missing") or ()),
            "excluded": len(progress.get("excluded") or ()),
        },
        "patch": patch,
        "families": family_rows,
        "all_passed": all(checks.values()),
    }
    write_json(VALIDATION_PATH, validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not validation["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
