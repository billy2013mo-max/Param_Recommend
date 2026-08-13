#!/usr/bin/env python3
"""Derive a small, machine-readable evidence-gap report from the H800 theory
calibration audit.

This module never launches GPU work, never creates or edits a queue, and never
proposes a concrete campaign.  It reads a *finished* calibration report, and for
each acceptance criterion the historical bootstrap fails, it names the specific
mechanism (and selector where the data localizes it) whose evidence is missing,
plus the prospective evidence that would close the gap.  Any targeted experiment
that follows must be a separately approved design with its own schema.

Everything is derived from the bound calibration report; nothing is invented.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "sft_h800_evidence_gap/v1"
STATUS = "gap_only"

# Grey-scale acceptance thresholds mirror the planner design's minimum bar.
MIN_SUCCESS_P95_COVERAGE = 0.95
MAX_FALSE_SAFE_OOM = 0
MAX_TOP1_REGRET = 0.10
MIN_SCALING_RATIO = 1.8


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _zero_stage_int(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    text = str(value or "").strip().lower().replace("zero", "")
    try:
        return int(text)
    except ValueError:
        return 0


def _memory_selector_token(selector: Mapping[str, Any]) -> str:
    """Match the calibration module's tuple-JSON selector token exactly, so the
    conflict set intersects ``false_safe_oom_by_selector`` keys."""

    return json.dumps(
        [
            str(selector.get("training_mode") or "unknown").lower(),
            _zero_stage_int(selector.get("zero_stage")),
            bool(selector.get("gradient_checkpointing")),
            bool(selector.get("packing")),
        ],
        separators=(",", ":"),
    )


def _same_config_label_conflicts(basis_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Find memory selectors whose identical (mbs, seq, gpu) cell both succeeds and OOMs.

    Such a cell proves the OOM is driven by something outside every calibration
    key (suspected runtime defect or shared-host interference), so no
    deterministic memory model can both admit it (it usually fits) and reject it
    (it sometimes crashes).  Read directly from plain basis fields so this module
    stays dependency-light.
    """

    records = basis_report.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return []
    # selector_token -> cell -> {"success": n, "oom": n}
    by_selector: dict[str, dict[tuple, dict[str, int]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        route = record.get("route")
        if not (isinstance(route, Mapping) and route.get("memory_boundary") is True):
            continue
        outcome = record.get("outcome")
        outcome = outcome.get("class") if isinstance(outcome, Mapping) else outcome
        outcome = str(outcome or "").strip().lower()
        if outcome not in {"success", "oom"}:
            continue
        selector = record.get("selector") if isinstance(record.get("selector"), Mapping) else {}
        scenario = record.get("scenario") if isinstance(record.get("scenario"), Mapping) else {}
        selector_key = _memory_selector_token(selector)
        cell = (
            scenario.get("physical_mbs"),
            scenario.get("cutoff_len"),
            scenario.get("gpu_count"),
        )
        bucket = by_selector.setdefault(selector_key, {}).setdefault(
            cell, {"success": 0, "oom": 0}
        )
        bucket[outcome] += 1
    conflicts: list[dict[str, Any]] = []
    for selector_key, cells in sorted(by_selector.items()):
        mixed = {
            f"mbs={c[0]},seq={c[1]},gpu={c[2]}": counts
            for c, counts in sorted(cells.items(), key=lambda kv: str(kv[0]))
            if counts["success"] > 0 and counts["oom"] > 0
        }
        if mixed:
            conflicts.append({"selector": selector_key, "conflicting_cells": mixed})
    return conflicts


def _memory_gaps(
    memory: Mapping[str, Any], label_conflicts: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    coverage = _finite(memory.get("scenario_equal_success_p95_coverage"))
    pooled_coverage = _finite(memory.get("success_p95_coverage"))
    false_safe = int(memory.get("false_safe_oom") or 0)
    by_selector = memory.get("false_safe_oom_by_selector")
    by_selector = by_selector if isinstance(by_selector, Mapping) else {}
    conflicted_selectors = {c["selector"] for c in label_conflicts}
    if coverage is not None and coverage < MIN_SUCCESS_P95_COVERAGE:
        gaps.append(
            {
                "gap_id": "memory_p95_coverage_below_bar",
                "mechanism": "reserved_memory_p95_upper_bound",
                "observed": {
                    "scenario_equal_success_p95_coverage": coverage,
                    "pooled_success_p95_coverage": pooled_coverage,
                },
                "why_insufficient": (
                    "held-out reserved peaks exceed the predicted P95 more often than "
                    "the 0.95 admission bar allows"
                ),
                "prospective_acceptance": {
                    "scenario_equal_success_p95_coverage_at_least": MIN_SUCCESS_P95_COVERAGE,
                    "evaluated_on": "independent_prospective_holdout",
                },
            }
        )
    if false_safe > MAX_FALSE_SAFE_OOM:
        # Distinguish the two possible causes.  A false-safe whose selector has an
        # identical (mbs,seq,gpu) cell that BOTH succeeds and OOMs is a label
        # conflict driven outside every calibration key -- no deterministic model
        # can fix it; it must first be triaged as software/infra failure.  Only a
        # false-safe without such a conflict points at model under-prediction.
        false_safe_selectors = set(by_selector)
        conflict_driven = sorted(false_safe_selectors & conflicted_selectors)
        model_driven = sorted(false_safe_selectors - conflicted_selectors)
        if conflict_driven:
            gaps.append(
                {
                    "gap_id": "false_safe_is_label_conflict_not_model_error",
                    "mechanism": "oom_driven_by_factors_outside_every_calibration_key",
                    "observed": {
                        "false_safe_oom": false_safe,
                        "conflicted_selectors": conflict_driven,
                        "conflicting_cells": [
                            c for c in label_conflicts if c["selector"] in conflict_driven
                        ],
                    },
                    "why_insufficient": (
                        "identical (mbs,seq,gpu) configs both succeed and OOM for these "
                        "selectors, so the OOM is driven outside any calibration key "
                        "(suspected mixed-dtype all-gather runtime defect or shared-host "
                        "interference); a deterministic memory model cannot both admit "
                        "(usually fits) and reject (sometimes crashes) them"
                    ),
                    "prospective_acceptance": {
                        "triage_these_ooms_as_software_or_infrastructure_failure_first": True,
                        "if_real_boundary_require_noncapped_peak_reserved_upload": True,
                        "never_exclude_lora_zero3_from_search_on_this_alone": True,
                    },
                }
            )
        if model_driven:
            gaps.append(
                {
                    "gap_id": "false_safe_oom_present",
                    "mechanism": "reserved_memory_center_underprediction_for_specific_selectors",
                    "observed": {
                        "false_safe_oom": false_safe,
                        "model_driven_selectors": model_driven,
                    },
                    "why_insufficient": (
                        "the memory center under-predicts these selectors and no "
                        "same-config success/OOM conflict explains it; the OOM boundary "
                        "for these mechanisms is under-sampled"
                    ),
                    "prospective_acceptance": {
                        "false_safe_oom_at_most": MAX_FALSE_SAFE_OOM,
                        "needs_success_and_oom_boundary_for_selectors": model_driven,
                    },
                }
            )
    return gaps


def _ranking_evidence_thinness(calibration_report: Mapping[str, Any]) -> dict[str, Any]:
    """Count per-card ranking groups that have fewer than two rankable candidates.

    A group with <2 admitted candidates cannot even be ordered, so its ranking
    concordance is undefined -- the throughput ranking is largely unproven not
    because it ranks badly but because most cells have nothing to rank.
    """

    loocv = calibration_report.get("global_scenario_loocv")
    folds = loocv.get("folds") if isinstance(loocv, Mapping) else None
    if not isinstance(folds, Sequence):
        return {"groups": 0, "rankable_ge2": 0, "rankable_le1": 0}
    total = ge2 = 0
    for fold in folds:
        if not isinstance(fold, Mapping):
            continue
        primary = ((fold.get("throughput") or {}).get("primary") or {})
        ranking = primary.get("joint_memory_gated_ranking") or {}
        for group in ranking.get("gpu_groups") or []:
            if not isinstance(group, Mapping):
                continue
            total += 1
            if int(group.get("predicted_admitted_rankable") or 0) >= 2:
                ge2 += 1
    return {"groups": total, "rankable_ge2": ge2, "rankable_le1": total - ge2}


def _throughput_gaps(
    throughput: Mapping[str, Any], ranking_thinness: Mapping[str, Any]
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    regret = _finite(throughput.get("scenario_equal_top1_regret"))
    safety_failures = int(throughput.get("memory_gated_safety_failures") or 0)
    scaling_claims = int(throughput.get("scaling_1_8_claims") or 0)
    scaling_valid = int(throughput.get("scaling_1_8_valid_claims") or 0)
    groups = int(ranking_thinness.get("groups") or 0)
    rankable_le1 = int(ranking_thinness.get("rankable_le1") or 0)
    if groups and rankable_le1 * 2 >= groups:
        gaps.append(
            {
                "gap_id": "throughput_ranking_evidence_too_thin",
                "mechanism": "per_card_candidate_coverage_for_throughput_ranking",
                "observed": {
                    "ranking_groups": groups,
                    "groups_with_fewer_than_two_rankable_candidates": rankable_le1,
                    "groups_with_two_or_more_rankable_candidates": groups - rankable_le1,
                },
                "why_insufficient": (
                    "most per-card groups have fewer than two memory-admitted candidates, "
                    "so their throughput order is undefined; the ranking cannot be judged "
                    "reliable on this evidence regardless of pairwise concordance"
                ),
                "prospective_acceptance": {
                    "each_recommendable_card_count_needs_at_least_two_admitted_candidates": True,
                    "evaluated_on": "independent_prospective_holdout",
                },
            }
        )
    if regret is not None and regret > MAX_TOP1_REGRET:
        gaps.append(
            {
                "gap_id": "throughput_top1_regret_above_bar",
                "mechanism": "memory_gated_throughput_lower_bound_ranking",
                "observed": {"scenario_equal_top1_regret": regret},
                "why_insufficient": (
                    "memory-admitted ranking loses more than the 10% throughput bar "
                    "against the best measured safe candidate"
                ),
                "prospective_acceptance": {
                    "scenario_equal_top1_regret_at_most": MAX_TOP1_REGRET,
                    "evaluated_on": "independent_prospective_holdout",
                },
            }
        )
    if safety_failures > 0:
        gaps.append(
            {
                "gap_id": "memory_gated_ranking_safety_failures",
                "mechanism": "predicted_safe_configs_that_are_actually_unsafe",
                "observed": {"memory_gated_safety_failures": safety_failures},
                "why_insufficient": (
                    "configs admitted as memory-safe were measured to OOM or exceed the "
                    "0.95 line; the admission boundary is not yet trustworthy"
                ),
                "prospective_acceptance": {"memory_gated_safety_failures_at_most": 0},
            }
        )
    # A doubling recommendation needs both endpoints memory-admitted and a
    # conservative and measured ratio >= 1.8; with none surviving there is simply
    # no evidence for a scaling rule (never assert 1.8x on this alone).
    gaps.append(
        {
            "gap_id": "scaling_1_8x_unproven",
            "mechanism": "minimum_card_to_double_card_conservative_throughput_ratio",
            "observed": {
                "surviving_scaling_1_8_claims": scaling_claims,
                "validated_scaling_1_8_claims": scaling_valid,
            },
            "why_insufficient": (
                "no memory-admitted minimum-card vs double-card pair clears a "
                "conservative 1.8x with the measured ratio also >= 1.8"
            ),
            "prospective_acceptance": {
                "matched_N_and_2N_pairs_both_memory_admitted": True,
                "conservative_and_measured_ratio_at_least": MIN_SCALING_RATIO,
            },
        }
    )
    return gaps


# Publication blockers that map to a specific missing-evidence mechanism.
_BLOCKER_GAPS = {
    "verified_calibration_anchor_missing": {
        "gap_id": "no_verified_publication_anchor",
        "mechanism": "native_v2_execution_fingerprint_anchor",
        "why_insufficient": (
            "every admitted row is legacy-tier recovered evidence; no native "
            "sft_execution_fingerprint/v2 attempt anchors the calibration"
        ),
        "prospective_acceptance": {
            "at_least_one_native_v2_fingerprint_attempt_per_calibration_selector": True
        },
    },
    "historical_recovery_is_not_native_v2_publication_evidence": {
        "gap_id": "recovery_not_publication_grade",
        "mechanism": "phased_data_admission_historical_bounded_is_theory_only",
        "why_insufficient": (
            "historical_bounded recovery is bootstrap-only and cannot publish a "
            "calibrated profile by policy"
        ),
        "prospective_acceptance": {"prospective_native_v2_campaign_separately_approved": True},
    },
    "planner_trusted_exact_operator_manifest_missing": {
        "gap_id": "no_trusted_exact_operator_manifest",
        "mechanism": "zero3_persistence_and_max_current_module_upper_bound",
        "why_insufficient": (
            "without a trusted per-operator manifest, ZeRO-3 persistence uses the full "
            "loaded-parameter upper bound and cannot reach the exact tier"
        ),
        "prospective_acceptance": {
            "cpu_checkpoint_scanner_operator_manifest_with_trusted_source_bound": True
        },
    },
    "runtime_mechanism_fingerprint_missing": {
        "gap_id": "no_runtime_mechanism_fingerprint",
        "mechanism": "sft_runtime_mechanism_v2_calibration_key",
        "why_insufficient": (
            "no reusable runtime-mechanism fingerprint, so calibration cannot be keyed "
            "to a specific build/kernel/ZeRO path"
        ),
        "prospective_acceptance": {"bound_sft_runtime_mechanism_v2_per_calibration_row": True},
    },
    "prospective_acceptance_required": {
        "gap_id": "prospective_acceptance_required",
        "mechanism": "independent_prospective_holdout_acceptance",
        "why_insufficient": (
            "historical cross-validation cannot substitute for prospective acceptance on "
            "an independently collected holdout"
        ),
        "prospective_acceptance": {"prospective_holdout_meets_all_bars_separately_approved": True},
    },
}


def build_gap_report(
    calibration_report: Mapping[str, Any],
    basis_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive the evidence-gap report from a finished calibration report.

    ``basis_report`` is optional; when supplied it is used only to detect
    same-config success/OOM label conflicts, which distinguish a model
    under-prediction from an out-of-key label conflict for any false-safe OOM.
    """

    if calibration_report.get("schema") != "sft_h800_theory_calibration/v1":
        raise ValueError("gap report requires an sft_h800_theory_calibration/v1 report")
    if calibration_report.get("publishable") is not False:
        raise ValueError("gap report is only defined for a nonpublishable calibration")

    aggregate = calibration_report.get("aggregate_validation")
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    memory = aggregate.get("memory_feasibility")
    memory = memory if isinstance(memory, Mapping) else {}
    throughput = aggregate.get("throughput_primary")
    throughput = throughput if isinstance(throughput, Mapping) else {}

    cohort = (
        (calibration_report.get("full_historical_bootstrap_fit") or {})
        .get("memory", {})
        .get("tail", {})
        .get("cohort_evidence_inflation")
    )
    cohort = cohort if isinstance(cohort, Mapping) else {}

    label_conflicts = (
        _same_config_label_conflicts(basis_report) if isinstance(basis_report, Mapping) else []
    )

    gaps: list[dict[str, Any]] = []
    gaps.extend(_memory_gaps(memory, label_conflicts))
    gaps.extend(_throughput_gaps(throughput, _ranking_evidence_thinness(calibration_report)))

    blockers = calibration_report.get("blockers") or []
    for blocker in blockers:
        template = _BLOCKER_GAPS.get(blocker)
        if template is not None:
            gaps.append({**template, "observed": {"calibration_blocker": blocker}})

    if int(cohort.get("runtime_cohorts") or 0) < 2:
        gaps.append(
            {
                "gap_id": "single_runtime_cohort",
                "mechanism": "runtime_cohort_nuisance_identifiability",
                "observed": {"runtime_cohorts": cohort.get("runtime_cohorts")},
                "why_insufficient": (
                    "with fewer than two runtime cohorts the cohort nuisance and its "
                    "uncertainty inflation are not separately identifiable"
                ),
                "prospective_acceptance": {"at_least_two_independent_runtime_cohorts": True},
            }
        )

    gaps.sort(key=lambda item: item["gap_id"])
    report = {
        "schema": SCHEMA,
        "status": STATUS,
        "source_calibration_report_sha256": calibration_report.get("report_sha256"),
        "source_calibration_status": calibration_report.get("status"),
        "creates_gpu_queue": False,
        "proposes_campaign": False,
        "requires_separately_approved_design": True,
        "acceptance_bars": {
            "scenario_equal_success_p95_coverage_at_least": MIN_SUCCESS_P95_COVERAGE,
            "false_safe_oom_at_most": MAX_FALSE_SAFE_OOM,
            "scenario_equal_top1_regret_at_most": MAX_TOP1_REGRET,
            "scaling_ratio_at_least": MIN_SCALING_RATIO,
        },
        "gap_count": len(gaps),
        "gaps": gaps,
        "policy": (
            "names only mechanism-specific missing evidence and prospective acceptance; "
            "never launches GPU work, never creates a queue, never proposes a campaign"
        ),
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def validate_gap_report(report: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if report.get("schema") != SCHEMA:
        issues.append("schema_mismatch")
    if report.get("status") != STATUS:
        issues.append("status_is_not_gap_only")
    if report.get("creates_gpu_queue") is not False:
        issues.append("gap_report_must_not_create_a_queue")
    if report.get("proposes_campaign") is not False:
        issues.append("gap_report_must_not_propose_a_campaign")
    if report.get("requires_separately_approved_design") is not True:
        issues.append("gap_report_must_require_separately_approved_design")
    if not isinstance(report.get("source_calibration_report_sha256"), str) or not report.get(
        "source_calibration_report_sha256"
    ):
        issues.append("gap_report_not_bound_to_calibration_report")
    gaps = report.get("gaps")
    if not isinstance(gaps, Sequence) or isinstance(gaps, (str, bytes)):
        issues.append("gaps_missing")
        gaps = []
    for gap in gaps:
        if not isinstance(gap, Mapping) or not gap.get("mechanism") or not gap.get(
            "prospective_acceptance"
        ):
            issues.append("gap_missing_mechanism_or_prospective_acceptance")
            break
    expected = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if expected != _canonical_sha256(unsigned):
        issues.append("report_sha256_mismatch")
    return sorted(set(issues))


def write_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    import os

    os.replace(temporary, path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=root / "artifacts" / "h800_theory_calibration.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts" / "h800_evidence_gap.json",
    )
    parser.add_argument(
        "--basis",
        type=Path,
        default=root / "artifacts" / "h800_theory_basis.json",
        help="Optional theory basis; used only to detect same-config label conflicts.",
    )
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    basis = (
        json.loads(args.basis.read_text(encoding="utf-8"))
        if args.basis and args.basis.is_file()
        else None
    )
    report = build_gap_report(calibration, basis)
    issues = validate_gap_report(report)
    if issues:
        raise SystemExit("invalid evidence-gap report: " + ", ".join(issues))
    write_report(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": report["report_sha256"],
                "gap_count": report["gap_count"],
                "gap_ids": [gap["gap_id"] for gap in report["gaps"]],
                "creates_gpu_queue": report["creates_gpu_queue"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
