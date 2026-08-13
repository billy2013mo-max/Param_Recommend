#!/usr/bin/env python3
"""Evaluate the frozen Phase D Stage 1 Packing memory boundary without promoting it.

Stage 1 measured four boundary chains at their lower cutoff anchor only, each with
an unpacked/packed treatment pair and two repeats.  This script joins the frozen
queues to the recorded terminal evidence, scores the Phase D 7.3 acceptance
thresholds, and decides which chains may be materialized into Stage 2.  It writes
a report and mutates nothing: no queue, no approval, no frozen model artifact.

CUDA OOM is a right-censored lower bound on memory, never a regression label, so
OOM rows never contribute to center error or upper coverage.  Only rows whose
execution fingerprint is complete (``calibration_eligible``) carry evidence; the
one ineligible parent OOM is reported as excluded and its retry is scored in its
place.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from common import ARTIFACT_DIR, RESULTS_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json


CAMPAIGN_ID = "h800_packing_memory_boundary_20260805_v1"
PHASE_ID = "h800_packing_memory_boundary_stage1_v1"
PARENT_QUEUE = ROOT / "matrix" / "h800_packing_memory_boundary_stage1_v1.jsonl"
RESUME_QUEUE = ROOT / "matrix" / "h800_packing_memory_boundary_stage1_resume_v2.jsonl"
PARENT_DESIGN = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_design_v1.json"
RESUME_DESIGN = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_resume_design_v2.json"
SELECTION = ARTIFACT_DIR / "h800_packing_memory_boundary_selection_v1.json"
RESUME_SELECTION = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_resume_selection_v2.json"
PREDICTIONS = ARTIFACT_DIR / "h800_packing_memory_boundary_candidate_predictions_v1.json"
OUTPUT = ARTIFACT_DIR / "h800_packing_memory_boundary_stage1_evaluation_v1.json"

GIB = float(1024**3)
# Phase D 7.3: an automatically recommended configuration must retain at least
# 5% of card capacity as headroom.  The safe limit is the memory model's own
# admission ceiling and is stricter than raw capacity.
MINIMUM_CAPACITY_SAFETY_MARGIN = 0.05
SUCCESSFUL_UPPER_COVERAGE_FLOOR = 0.95
SCENARIO_EQUAL_UPPER_COVERAGE_FLOOR = 0.95
TERMINAL = {"success", "oom", "failed", "timeout", "invalid_measurement"}


def _request_id(job: dict[str, Any]) -> str:
    treatment = "p" if job["packing"] else "u"
    return f"boundary-{job['chain_id'].lower()}-c{job['cutoff_len']}-{treatment}"


def _summaries(job_id: str) -> list[dict[str, Any]]:
    return [
        read_json(path)
        for path in sorted((RESULTS_DIR / job_id / "metrics").glob("summary.rank*.json"))
    ]


def _ledger_authoritative(summary: dict[str, Any]) -> bool:
    evidence = summary.get("token_ledger_evidence") or {}
    return bool(
        evidence.get("schema") == "consumed_token_ledger/v1"
        and evidence.get("authoritative") is True
    )


def _packing_semantics_passed(summary: dict[str, Any]) -> bool:
    evidence = ((summary.get("runtime_batch_evidence") or {}).get("packing") or {})
    violations = evidence.get("violations") or {}
    return bool(
        evidence.get("semantic_checks_passed") is True
        and int(evidence.get("multi_sample_features") or 0) > 0
        and all(int(value) == 0 for value in violations.values())
    )


def _observed_reserved_bytes(summaries: list[dict[str, Any]]) -> int | None:
    """Peak reserved bytes across every rank.

    Admission safety is a per-card property, so the maximum over ranks is the
    only defensible reduction: rank 0 alone understates an imbalanced job.
    """
    values = [int(summary.get("max_reserved") or 0) for summary in summaries]
    values = [value for value in values if value > 0]
    return max(values) if values else None


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _load_queue_rows() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Merge parent and resume queues, keeping one row per job id.

    The resume queue re-lists the twelve parent rows that never ran plus one
    retry, so a plain concatenation would double-count.  Parent rows win on
    identity; provenance records which queue actually executed each row.
    """
    rows: dict[str, dict[str, Any]] = {}
    origin: dict[str, str] = {}
    for job in read_jsonl(PARENT_QUEUE):
        rows[job["job_id"]] = job
        origin[job["job_id"]] = "stage1_v1"
    for job in read_jsonl(RESUME_QUEUE):
        if job["job_id"] not in rows:
            rows[job["job_id"]] = job
            origin[job["job_id"]] = "stage1_resume_v2"
        else:
            origin[job["job_id"]] = "stage1_v1+resume_v2"
    return list(rows.values()), origin


def _verify_frozen_inputs() -> dict[str, Any]:
    parent_design = read_json(PARENT_DESIGN)
    resume_design = read_json(RESUME_DESIGN)
    selection = read_json(SELECTION)
    resume_selection = read_json(RESUME_SELECTION)
    errors: list[dict[str, Any]] = []
    if parent_design.get("campaign_id") != CAMPAIGN_ID or parent_design.get("phase_id") != PHASE_ID:
        errors.append({"check": "parent_design_identity"})
    if parent_design.get("queue", {}).get("sha256") != sha256_file(PARENT_QUEUE):
        errors.append({"check": "parent_queue_sha256"})
    if resume_design.get("queue", {}).get("sha256") != sha256_file(RESUME_QUEUE):
        errors.append({"check": "resume_queue_sha256"})
    if resume_design.get("resume_contract", {}).get("ineligible_oom_retries") != 1:
        errors.append({"check": "resume_contract_retry_count"})
    if selection.get("report_sha256") is None or resume_selection.get("report_sha256") is None:
        errors.append({"check": "selection_report_sha256"})
    for artifact, path in (
        (parent_design, PARENT_DESIGN),
        (resume_design, RESUME_DESIGN),
    ):
        if artifact.get("publication_allowed") is not False:
            errors.append({"check": "publication_allowed_must_be_false", "path": str(path)})
    return {
        "parent_design": parent_design,
        "resume_design": resume_design,
        "selection": selection,
        "resume_selection": resume_selection,
        "errors": errors,
    }


def _join_rows(
    queue: list[dict[str, Any]],
    origin: dict[str, str],
    predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for job in sorted(queue, key=lambda row: (row["chain_id"], row["packing"], row.get("repeat", 0))):
        job_id = job["job_id"]
        status_path = RESULTS_DIR / job_id / "status.json"
        status = read_json(status_path) if status_path.is_file() else {"classification": "missing"}
        summaries = _summaries(job_id)
        outcome = str(status.get("classification"))
        eligible = status.get("calibration_eligible") is True
        prediction = predictions.get(_request_id(job)) or {}
        memory = prediction.get("memory") or {}

        # The queue's own refit center is the quantity Stage 1 was designed to
        # test; the shared physical model's center is carried alongside as a
        # diagnostic so the two can be compared without conflating them.
        refit_center_gib = float(job["predicted_arm_center_gib"])
        physical_center_gib = memory.get("gib", {}).get("reserved_center")
        admission_upper_gib = memory.get("gib", {}).get("admission_upper")
        safe_limit_gib = memory.get("gib", {}).get("safe_limit")

        observed_bytes = _observed_reserved_bytes(summaries)
        observed_gib = observed_bytes / GIB if observed_bytes is not None else None
        success = outcome == "success"

        # Memory evidence is only meaningful on eligible rows; an incomplete
        # fingerprint means we cannot attribute the reading to this config.
        scored = eligible and observed_gib is not None
        center_ape = (
            abs(refit_center_gib - observed_gib) / observed_gib
            if scored and success and observed_gib > 0.0
            else None
        )
        upper_covers = (
            bool(admission_upper_gib is not None and observed_gib <= admission_upper_gib)
            if scored and success
            else None
        )
        headroom_fraction = (
            (safe_limit_gib - observed_gib) / safe_limit_gib
            if scored and success and safe_limit_gib
            else None
        )
        joined.append(
            {
                "job_id": job_id,
                "queue_origin": origin.get(job_id),
                "request_id": _request_id(job),
                "chain_id": job["chain_id"],
                "boundary_setting_id": job["boundary_setting_id"],
                "model_id": job["model_id"],
                "train_type": job["train_type"],
                "workload_id": job["workload_id"],
                "cutoff_len": int(job["cutoff_len"]),
                "zero_stage": int(job["zero_stage"]),
                "gc": bool(job["gc"]),
                "packing": bool(job["packing"]),
                "gpu_count": int(job["gpu_count"]),
                "repeat": int(job.get("repeat", 0)),
                "outcome": outcome,
                "terminal": outcome in TERMINAL,
                "calibration_eligible": eligible,
                "execution_fingerprint_quality": status.get("execution_fingerprint_quality"),
                "execution_fingerprint_errors": status.get("execution_fingerprint_errors"),
                "rank_summaries": len(summaries),
                "rank_summaries_complete": len(summaries) == int(job["gpu_count"]),
                "all_ranks_ledger_authoritative": bool(
                    summaries and all(_ledger_authoritative(row) for row in summaries)
                ),
                "packing_semantics_passed": (
                    bool(summaries and all(_packing_semantics_passed(row) for row in summaries))
                    if job["packing"] and success
                    else None
                ),
                "predicted_refit_center_gib": refit_center_gib,
                "predicted_physical_center_gib": physical_center_gib,
                "predicted_admission_upper_gib": admission_upper_gib,
                "safe_limit_gib": safe_limit_gib,
                "predicted_admitted": memory.get("admitted") is True,
                "observed_max_reserved_gib": observed_gib,
                "observed_scored": scored,
                "center_absolute_percentage_error": center_ape,
                "upper_covers_observed": upper_covers,
                "capacity_safety_margin_fraction": headroom_fraction,
                "capacity_safety_margin_passed": (
                    bool(headroom_fraction is not None and headroom_fraction >= MINIMUM_CAPACITY_SAFETY_MARGIN)
                    if headroom_fraction is not None
                    else None
                ),
                # A false-safe OOM means the model would have let the config
                # through and it OOMed.  Stage 1 deliberately queued only
                # rejected candidates (every row has admitted=false), so this
                # gate is vacuous here and must not be read as evidence that
                # admission is safe.  `admission_upper_would_admit` is the
                # non-vacuous companion: it asks whether the published upper
                # guard sits above the safe limit, i.e. whether the guard would
                # have called this OOM configuration safe.
                "false_safe_oom": bool(scored and outcome == "oom" and memory.get("admitted") is True),
                "admission_upper_would_admit": (
                    bool(
                        admission_upper_gib is not None
                        and safe_limit_gib is not None
                        and admission_upper_gib <= safe_limit_gib
                    )
                ),
                "upper_guard_exceeds_safe_limit": (
                    bool(
                        admission_upper_gib is not None
                        and safe_limit_gib is not None
                        and admission_upper_gib > safe_limit_gib
                    )
                ),
                # A success whose observed peak exceeded the admission ceiling is
                # a near-miss: the run happened to fit, but the model's own
                # safety line was breached.
                "success_exceeding_safe_limit": bool(
                    scored and success and safe_limit_gib is not None and observed_gib > safe_limit_gib
                ),
                "gpu_mask": status.get("gpu_mask"),
                "started_unix": status.get("started_unix"),
                "finished_unix": status.get("finished_unix"),
            }
        )
    return joined


def _chain_reports(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chains: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        chains.setdefault(row["chain_id"], []).append(row)
    reports: list[dict[str, Any]] = []
    for chain_id, chain_rows in sorted(chains.items()):
        eligible = [row for row in chain_rows if row["calibration_eligible"]]
        scored = [row for row in chain_rows if row["observed_scored"]]
        successes = [row for row in scored if row["outcome"] == "success"]
        # An OOM counts as evidence whether or not a rank summary survived to
        # record a reserved reading: the terminal classification is the signal.
        # Requiring memory evidence here would silently report "0 OOM" for a
        # chain that OOMed on every repeat before emitting a summary.
        ooms = [row for row in eligible if row["outcome"] == "oom"]
        ooms_without_memory_evidence = [row for row in ooms if not row["observed_scored"]]
        margins = [
            row["capacity_safety_margin_fraction"]
            for row in successes
            if row["capacity_safety_margin_fraction"] is not None
        ]
        # Stage 2 releases the higher cutoff of the same chain, so every
        # unpacked/packed repeat at the low anchor must have succeeded AND kept
        # the 5% margin.  A chain that only barely fit at the low anchor gives
        # no license to extrapolate upward.  `all_repeats_success` requires at
        # least one eligible row, otherwise an all-OOM chain would vacuously
        # pass an `all(...)` over an empty list.
        all_repeats_success = bool(eligible) and all(row["outcome"] == "success" for row in eligible)
        margin_ok = bool(margins) and all(
            value >= MINIMUM_CAPACITY_SAFETY_MARGIN for value in margins
        )
        packing_ratio = None
        packed = [row for row in successes if row["packing"]]
        unpacked = [row for row in successes if not row["packing"]]
        if packed and unpacked:
            packing_ratio = max(row["observed_max_reserved_gib"] for row in packed) / max(
                row["observed_max_reserved_gib"] for row in unpacked
            )
        predicted_ratio = None
        packed_pred = {row["predicted_refit_center_gib"] for row in chain_rows if row["packing"]}
        unpacked_pred = {row["predicted_refit_center_gib"] for row in chain_rows if not row["packing"]}
        if packed_pred and unpacked_pred:
            predicted_ratio = max(packed_pred) / max(unpacked_pred)
        reports.append(
            {
                "chain_id": chain_id,
                "cutoff_len": chain_rows[0]["cutoff_len"],
                "model_id": chain_rows[0]["model_id"],
                "train_type": chain_rows[0]["train_type"],
                "jobs": len(chain_rows),
                "eligible_jobs": len(eligible),
                "scored_jobs": len(scored),
                "successes": len(successes),
                "ooms": len(ooms),
                "ooms_without_memory_evidence": len(ooms_without_memory_evidence),
                "minimum_capacity_safety_margin_fraction": min(margins) if margins else None,
                "all_eligible_repeats_success": all_repeats_success,
                "capacity_safety_margin_passed": margin_ok,
                "observed_packed_over_unpacked_reserved_ratio": packing_ratio,
                "predicted_packed_over_unpacked_center_ratio": predicted_ratio,
                "packing_memory_ratio_underpredicted": (
                    bool(packing_ratio is not None and predicted_ratio is not None and packing_ratio > predicted_ratio)
                ),
                "stage2_release_allowed": bool(all_repeats_success and margin_ok),
                "stage2_block_reasons": [
                    reason
                    for reason, blocked in (
                        ("eligible_repeat_not_all_success", not all_repeats_success),
                        ("capacity_safety_margin_below_5pct", not margin_ok),
                    )
                    if blocked
                ],
            }
        )
    return reports


def evaluate() -> dict[str, Any]:
    frozen = _verify_frozen_inputs()
    queue, origin = _load_queue_rows()
    prediction_artifact = read_json(PREDICTIONS)
    predictions = {row["request_id"]: row for row in prediction_artifact["predictions"]}

    rows = _join_rows(queue, origin, predictions)
    chains = _chain_reports(rows)

    scored = [row for row in rows if row["observed_scored"]]
    successes = [row for row in scored if row["outcome"] == "success"]
    excluded = [row for row in rows if not row["calibration_eligible"]]

    upper_covered = [row for row in successes if row["upper_covers_observed"] is True]
    successful_upper_coverage = _fraction(len(upper_covered), len(successes))

    # Scenario-equal coverage weights each boundary setting once, so a chain with
    # more repeats cannot mask another chain's miss.
    setting_coverage: dict[str, list[bool]] = {}
    for row in successes:
        setting_coverage.setdefault(row["boundary_setting_id"], []).append(
            row["upper_covers_observed"] is True
        )
    scenario_equal_upper_coverage = _fraction(
        sum(all(values) for values in setting_coverage.values()), len(setting_coverage)
    )

    false_safe_oom = [row for row in scored if row["false_safe_oom"]]
    admitted_rows = [row for row in rows if row["predicted_admitted"]]
    near_miss = [row for row in successes if row["success_exceeding_safe_limit"]]
    margin_failures = [
        row for row in successes if row["capacity_safety_margin_passed"] is False
    ]
    # Every Stage 1 row was a rejected candidate, so "0 false-safe OOM" is
    # vacuously true and carries no safety information.  Report the vacuity
    # explicitly and score the upper guard directly instead.
    oom_rows = [row for row in rows if row["calibration_eligible"] and row["outcome"] == "oom"]
    guard_would_admit_oom = [row for row in oom_rows if row["upper_guard_exceeds_safe_limit"]]

    # Monotonicity: a lighter configuration must not OOM while a strictly heavier
    # one on the same chain is judged safe.  Within Stage 1 the only within-chain
    # weight axis is packing, which strictly increases resident memory.
    monotonicity_violations = []
    for chain in chains:
        chain_rows = [row for row in scored if row["chain_id"] == chain["chain_id"]]
        packed_success = any(row["packing"] and row["outcome"] == "success" for row in chain_rows)
        unpacked_oom = any(not row["packing"] and row["outcome"] == "oom" for row in chain_rows)
        if packed_success and unpacked_oom:
            monotonicity_violations.append(
                {"chain_id": chain["chain_id"], "pattern": "unpacked_oom_with_packed_success"}
            )

    acceptance = {
        "false_safe_oom_count": len(false_safe_oom),
        "false_safe_oom_passes": len(false_safe_oom) == 0,
        "false_safe_oom_gate_is_vacuous": len(admitted_rows) == 0,
        "admitted_candidate_count": len(admitted_rows),
        "oom_rows_whose_upper_guard_exceeds_safe_limit": len(guard_would_admit_oom),
        "successful_p95_upper_coverage": successful_upper_coverage,
        "successful_p95_upper_coverage_passes": bool(
            successful_upper_coverage is not None
            and successful_upper_coverage >= SUCCESSFUL_UPPER_COVERAGE_FLOOR
        ),
        "scenario_equal_upper_coverage": scenario_equal_upper_coverage,
        "scenario_equal_upper_coverage_passes": bool(
            scenario_equal_upper_coverage is not None
            and scenario_equal_upper_coverage >= SCENARIO_EQUAL_UPPER_COVERAGE_FLOOR
        ),
        "capacity_safety_margin_floor": MINIMUM_CAPACITY_SAFETY_MARGIN,
        "capacity_safety_margin_failure_count": len(margin_failures),
        "capacity_safety_margin_passes": len(margin_failures) == 0,
        "successes_exceeding_safe_limit_count": len(near_miss),
        # An upper guard that sits above the safe limit trivially covers any
        # observation that fit on the card, so 100% coverage is not evidence of
        # a usable guard.  Count how many successes were covered by a guard that
        # was itself inside the safe limit.
        "successes_covered_by_guard_inside_safe_limit": sum(
            row["upper_covers_observed"] is True and row["admission_upper_would_admit"]
            for row in successes
        ),
        "upper_coverage_gate_is_vacuous": all(
            row["upper_guard_exceeds_safe_limit"] for row in successes
        )
        if successes
        else None,
        "ordering_monotonicity_violations": monotonicity_violations,
        "ordering_monotonicity_passes": not monotonicity_violations,
    }
    acceptance["all_thresholds_passed"] = bool(
        acceptance["false_safe_oom_passes"]
        and acceptance["successful_p95_upper_coverage_passes"]
        and acceptance["scenario_equal_upper_coverage_passes"]
        and acceptance["capacity_safety_margin_passes"]
        and acceptance["ordering_monotonicity_passes"]
    )

    execution = {
        "unique_jobs": len(rows),
        "parent_queue_jobs": len(read_jsonl(PARENT_QUEUE)),
        "resume_queue_jobs": len(read_jsonl(RESUME_QUEUE)),
        "all_terminal": all(row["terminal"] for row in rows),
        "calibration_eligible_jobs": len(rows) - len(excluded),
        "excluded_jobs": [
            {
                "job_id": row["job_id"],
                "chain_id": row["chain_id"],
                "outcome": row["outcome"],
                "execution_fingerprint_quality": row["execution_fingerprint_quality"],
                "execution_fingerprint_errors": row["execution_fingerprint_errors"],
            }
            for row in excluded
        ],
        "all_successes_rank_summaries_complete": all(
            row["rank_summaries_complete"] for row in successes
        ),
        "all_successes_ledger_authoritative": all(
            row["all_ranks_ledger_authoritative"] for row in successes
        ),
        "all_packed_successes_semantics_passed": all(
            row["packing_semantics_passed"] is not False for row in successes
        ),
        "self_gpu_mask_collisions": 0,
    }

    stage2 = {
        "automatic_release_allowed": False,
        "released_chains": [row["chain_id"] for row in chains if row["stage2_release_allowed"]],
        "blocked_chains": [
            {"chain_id": row["chain_id"], "reasons": row["stage2_block_reasons"]}
            for row in chains
            if not row["stage2_release_allowed"]
        ],
        "release_rule": (
            "release the higher cutoff of a chain only when every eligible low-anchor "
            "repeat succeeded and retained at least 5% capacity headroom"
        ),
    }

    blockers: list[str] = []
    if not execution["all_terminal"]:
        blockers.append("non_terminal_jobs_present")
    if frozen["errors"]:
        blockers.append("frozen_input_drift")
    if not acceptance["capacity_safety_margin_passes"]:
        blockers.append("capacity_safety_margin_below_floor")
    if not acceptance["successful_p95_upper_coverage_passes"]:
        blockers.append("successful_upper_coverage_below_floor")
    if not acceptance["scenario_equal_upper_coverage_passes"]:
        blockers.append("scenario_equal_upper_coverage_below_floor")
    if not acceptance["false_safe_oom_passes"]:
        blockers.append("false_safe_oom_present")
    if acceptance["oom_rows_whose_upper_guard_exceeds_safe_limit"]:
        blockers.append("upper_guard_above_safe_limit_on_oom_rows")

    report: dict[str, Any] = {
        "schema": "sft_h800_packing_memory_boundary_stage1_evaluation/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_inputs": {
            key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for key, path in (
                ("parent_queue", PARENT_QUEUE),
                ("resume_queue", RESUME_QUEUE),
                ("parent_design", PARENT_DESIGN),
                ("resume_design", RESUME_DESIGN),
                ("selection", SELECTION),
                ("resume_selection", RESUME_SELECTION),
                ("candidate_predictions", PREDICTIONS),
            )
        },
        "frozen_input_errors": frozen["errors"],
        "execution": execution,
        "acceptance": acceptance,
        "chains": chains,
        "stage2": stage2,
        "rows": rows,
        "memory_evidence_policy": {
            "oom_role": "right_censored_lower_bound_not_regression_label",
            "reserved_reduction": "maximum_over_all_rank_summaries",
            "ineligible_rows_excluded_from_all_metrics": True,
        },
        "stage1_complete_for_joint_fit": bool(
            execution["all_terminal"] and not frozen["errors"] and not blockers
        ),
        "prospective_acceptance_passed": False,
        "publication_allowed": False,
        "automatic_packing_recommendation_allowed": False,
        "blockers": blockers,
        "next_step": (
            "materialize_stage2_for_released_chains"
            if not blockers
            else "repair_memory_upper_guard_before_stage2"
        ),
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    return report


def main() -> None:
    report = evaluate()
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "report_sha256": report["report_sha256"],
                "execution": {
                    key: report["execution"][key]
                    for key in (
                        "unique_jobs",
                        "all_terminal",
                        "calibration_eligible_jobs",
                    )
                },
                "acceptance": {
                    key: report["acceptance"][key]
                    for key in (
                        "false_safe_oom_count",
                        "false_safe_oom_gate_is_vacuous",
                        "oom_rows_whose_upper_guard_exceeds_safe_limit",
                        "successful_p95_upper_coverage",
                        "upper_coverage_gate_is_vacuous",
                        "scenario_equal_upper_coverage",
                        "capacity_safety_margin_failure_count",
                        "all_thresholds_passed",
                    )
                },
                "chains": [
                    {
                        "chain_id": row["chain_id"],
                        "successes": row["successes"],
                        "ooms": row["ooms"],
                        "minimum_capacity_safety_margin_fraction": row[
                            "minimum_capacity_safety_margin_fraction"
                        ],
                        "observed_packed_over_unpacked_reserved_ratio": row[
                            "observed_packed_over_unpacked_reserved_ratio"
                        ],
                        "predicted_packed_over_unpacked_center_ratio": row[
                            "predicted_packed_over_unpacked_center_ratio"
                        ],
                        "stage2_release_allowed": row["stage2_release_allowed"],
                    }
                    for row in report["chains"]
                ],
                "stage2_released_chains": report["stage2"]["released_chains"],
                "blockers": report["blockers"],
                "next_step": report["next_step"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
