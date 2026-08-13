#!/usr/bin/env python3
"""Build fold-aware readiness summaries from historical H800 recovery evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from common import sha256_json
from recover_h800_historical_evidence import (
    LOOCV_POLICY,
    PACKING_POLICY,
    PROFILER_POLICY,
    SCHEMA as RECOVERY_SCHEMA,
)


SCHEMA = "sft_h800_historical_readiness/v2"

RESOURCE_COMPONENTS = (
    "feasibility",
    "memory_boundary",
    "throughput_primary",
    "throughput_screen_only",
)
PRIMARY_RESOURCE_COMPONENTS = (
    "feasibility",
    "memory_boundary",
    "throughput_primary",
)
PROFILER_DECLARED_ROLES = {"calibration", "holdout"}
PROFILER_EFFECTIVE_ROLES = {
    "calibration",
    "holdout",
    "fallback_calibration",
}
MIN_LOOCV_SCENARIOS = 3


def _matrix_rank(matrix: list[list[float]], tolerance: float = 1e-12) -> int:
    if not matrix:
        return 0
    values = [row[:] for row in matrix]
    rows = len(values)
    columns = len(values[0])
    rank = 0
    for column in range(columns):
        pivot = next(
            (row for row in range(rank, rows) if abs(values[row][column]) > tolerance),
            None,
        )
        if pivot is None:
            continue
        values[rank], values[pivot] = values[pivot], values[rank]
        scale = values[rank][column]
        values[rank] = [value / scale for value in values[rank]]
        for row in range(rows):
            if row == rank:
                continue
            factor = values[row][column]
            if abs(factor) <= tolerance:
                continue
            values[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(values[row], values[rank])
            ]
        rank += 1
        if rank == rows:
            break
    return rank


def _scenario_from_job(job: dict[str, Any]) -> dict[str, Any]:
    material = {
        "model_id": job.get("model_id"),
        "train_type": job.get("train_type"),
        "dataset_id": job.get("dataset_id"),
        "target_gbs": job.get("target_gbs"),
    }
    return {"material": material, "scenario_id": sha256_json(material)}


def _runtime_identity(
    record: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None, list[str]]:
    runtime = record.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    cohort_id = runtime.get("runtime_cohort_id")
    material = runtime.get("runtime_cohort_material")
    issues: list[str] = []
    if not isinstance(cohort_id, str) or not cohort_id:
        issues.append("runtime_cohort_id_missing")
    if not isinstance(material, dict):
        issues.append("runtime_cohort_material_missing")
        material = None
    if (
        isinstance(cohort_id, str)
        and cohort_id
        and isinstance(material, dict)
        and cohort_id != sha256_json(material)
    ):
        issues.append("runtime_cohort_id_material_hash_mismatch")
    return (
        cohort_id if isinstance(cohort_id, str) and cohort_id else None,
        material,
        issues,
    )


def _route_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for candidate in candidates:
        eligibility = candidate["record"].get("measurement_eligibility") or {}
        for route in RESOURCE_COMPONENTS:
            if eligibility.get(route) is True:
                counts[route] += 1
    return {route: counts[route] for route in RESOURCE_COMPONENTS}


def _tier_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(candidate["record"].get("evidence_tier") or "unknown")
                for candidate in candidates
            ).items()
        )
    )


def _job_id_set(
    context: dict[str, Any], field: str, digest_field: str
) -> tuple[set[str] | None, list[str]]:
    raw = context.get(field)
    if raw is None:
        return None, [f"{field}_missing"]
    if (
        not isinstance(raw, list)
        or any(not isinstance(item, str) or not item for item in raw)
        or len(raw) != len(set(raw))
    ):
        return None, [f"{field}_invalid"]
    normalized = sorted(raw)
    issues = []
    if context.get(digest_field) != sha256_json(normalized):
        issues.append(f"{digest_field}_mismatch")
    return set(normalized), issues


def _resource_loocv(
    records: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for record in records:
        eligibility = record.get("measurement_eligibility") or {}
        design = record.get("validation_design") or {}
        if eligibility.get("class") != "calibration_candidate":
            continue
        if design.get("mode") != "fold_dependent_loocv":
            continue
        row = rows[record["source_observation_id"]]
        job = (row.get("configuration") or {}).get("job") or {}
        expected = _scenario_from_job(job)
        cohort_id, cohort_material, runtime_issues = _runtime_identity(record)
        issues = list(runtime_issues)
        if design.get("policy") != LOOCV_POLICY:
            issues.append("loocv_policy_mismatch")
        if design.get("explicit_role") != "fold_dependent":
            issues.append("loocv_static_role_is_forbidden")
        if design.get("material") != expected["material"]:
            issues.append("loocv_scenario_material_mismatch")
        if design.get("scenario_id") != expected["scenario_id"]:
            issues.append("loocv_scenario_id_mismatch")
        item = {
            "record": record,
            "row": row,
            "job": job,
            "scenario": expected,
            "runtime_cohort_id": cohort_id,
            "runtime_cohort_material": cohort_material,
            "issues": sorted(set(issues)),
        }
        if issues:
            invalid.append(
                {
                    "source_observation_id": record["source_observation_id"],
                    "issues": sorted(set(issues)),
                }
            )
        else:
            candidates.append(item)

    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_cohort[str(candidate["runtime_cohort_id"])].append(candidate)

    # A scenario can occur under more than one runtime cohort.  Holding it out
    # only inside one cohort would leave the same model/mode/dataset/GBS
    # scenario in the training set through another cohort, which leaks the
    # scientific unit being evaluated.  Publication-style validation therefore
    # groups the scenario globally; runtime cohorts remain nuisance/fixed-effect
    # groups inside each training fold.
    global_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        global_by_scenario[candidate["scenario"]["scenario_id"]].append(candidate)

    global_folds: list[dict[str, Any]] = []
    global_fold_ids: list[str] = []
    global_test_counts: Counter[str] = Counter()
    for scenario_id in sorted(global_by_scenario):
        test = global_by_scenario[scenario_id]
        train = [
            candidate
            for other_id, group in global_by_scenario.items()
            if other_id != scenario_id
            for candidate in group
        ]
        train_ids = sorted(
            candidate["record"]["source_observation_id"] for candidate in train
        )
        test_ids = sorted(
            candidate["record"]["source_observation_id"] for candidate in test
        )
        for observation_id in test_ids:
            global_test_counts[observation_id] += 1
        train_cohorts = sorted(
            {str(candidate["runtime_cohort_id"]) for candidate in train}
        )
        test_cohorts = sorted(
            {str(candidate["runtime_cohort_id"]) for candidate in test}
        )
        fold_material = {
            "held_out_scenario_id": scenario_id,
            "train_observation_ids_sha256": sha256_json(train_ids),
            "test_observation_ids_sha256": sha256_json(test_ids),
        }
        fold_id = sha256_json(fold_material)
        global_fold_ids.append(fold_id)
        train_set = set(train_ids)
        test_set = set(test_ids)
        blockers = []
        if not train_ids:
            blockers.append("no_other_global_scenario")
        if train_set.intersection(test_set):
            blockers.append("train_test_membership_overlap")
        global_folds.append(
            {
                "fold_id": fold_id,
                "held_out_scenario_id": scenario_id,
                "held_out_scenario": test[0]["scenario"]["material"],
                "train_observations": len(train_ids),
                "test_observations": len(test_ids),
                "train_observation_ids_sha256": fold_material[
                    "train_observation_ids_sha256"
                ],
                "test_observation_ids_sha256": fold_material[
                    "test_observation_ids_sha256"
                ],
                "train_runtime_cohort_ids": train_cohorts,
                "test_runtime_cohort_ids": test_cohorts,
                "unseen_test_runtime_cohort_ids": sorted(
                    set(test_cohorts) - set(train_cohorts)
                ),
                "train_measurement_routes": _route_counts(train),
                "test_measurement_routes": _route_counts(test),
                "test_outcomes": dict(
                    sorted(
                        Counter(
                            str(
                                (candidate["row"].get("outcome") or {}).get(
                                    "class"
                                )
                                or ""
                            )
                            for candidate in test
                        ).items()
                    )
                ),
                "membership_disjoint": not train_set.intersection(test_set),
                "blockers": sorted(set(blockers)),
            }
        )

    cohort_rows: list[dict[str, Any]] = []
    all_fold_ids: list[str] = []
    all_test_counts: Counter[str] = Counter()
    component_cohorts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cohort_id in sorted(by_cohort):
        cohort = by_cohort[cohort_id]
        cohort_materials = {
            sha256_json(candidate["runtime_cohort_material"]) for candidate in cohort
        }
        cohort_integrity_blockers = []
        if len(cohort_materials) != 1:
            cohort_integrity_blockers.append(
                "runtime_cohort_material_inconsistent_within_id"
            )
        by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in cohort:
            by_scenario[candidate["scenario"]["scenario_id"]].append(candidate)

        folds = []
        for scenario_id in sorted(by_scenario):
            test = by_scenario[scenario_id]
            train = [
                candidate
                for other_id, group in by_scenario.items()
                if other_id != scenario_id
                for candidate in group
            ]
            train_ids = sorted(
                candidate["record"]["source_observation_id"] for candidate in train
            )
            test_ids = sorted(
                candidate["record"]["source_observation_id"] for candidate in test
            )
            for observation_id in test_ids:
                all_test_counts[observation_id] += 1
            train_set = set(train_ids)
            test_set = set(test_ids)
            outcomes = Counter(
                str((candidate["row"].get("outcome") or {}).get("class") or "")
                for candidate in test
            )
            fold_material = {
                "runtime_cohort_id": cohort_id,
                "held_out_scenario_id": scenario_id,
                "train_observation_ids_sha256": sha256_json(train_ids),
                "test_observation_ids_sha256": sha256_json(test_ids),
            }
            fold_id = sha256_json(fold_material)
            all_fold_ids.append(fold_id)
            blockers = list(cohort_integrity_blockers)
            if not train_ids:
                blockers.append("no_other_scenario_in_runtime_cohort")
            if train_set.intersection(test_set):
                blockers.append("train_test_membership_overlap")
            folds.append(
                {
                    "fold_id": fold_id,
                    "held_out_scenario_id": scenario_id,
                    "held_out_scenario": test[0]["scenario"]["material"],
                    "train_observations": len(train_ids),
                    "test_observations": len(test_ids),
                    "train_observation_ids_sha256": fold_material[
                        "train_observation_ids_sha256"
                    ],
                    "test_observation_ids_sha256": fold_material[
                        "test_observation_ids_sha256"
                    ],
                    "train_measurement_routes": _route_counts(train),
                    "test_measurement_routes": _route_counts(test),
                    "test_outcomes": dict(sorted(outcomes.items())),
                    "membership_disjoint": not train_set.intersection(test_set),
                    "blockers": sorted(set(blockers)),
                }
            )

        component_status: dict[str, dict[str, Any]] = {}
        for route in RESOURCE_COMPONENTS:
            route_by_scenario = {
                scenario_id: [
                    candidate
                    for candidate in group
                    if (candidate["record"].get("measurement_eligibility") or {}).get(
                        route
                    )
                    is True
                ]
                for scenario_id, group in by_scenario.items()
            }
            route_by_scenario = {
                scenario_id: group
                for scenario_id, group in route_by_scenario.items()
                if group
            }
            route_candidates = [
                candidate for group in route_by_scenario.values() for candidate in group
            ]
            component_folds = []
            component_blockers = list(cohort_integrity_blockers)
            if len(route_by_scenario) < MIN_LOOCV_SCENARIOS:
                component_blockers.append(
                    f"fewer_than_{MIN_LOOCV_SCENARIOS}_eligible_scenarios"
                )
            for scenario_id in sorted(route_by_scenario):
                test_route = route_by_scenario[scenario_id]
                train_route = [
                    candidate
                    for other_id, group in route_by_scenario.items()
                    if other_id != scenario_id
                    for candidate in group
                ]
                fold_blockers = []
                if not train_route:
                    fold_blockers.append("no_route_eligible_training_observations")
                if not test_route:
                    fold_blockers.append("no_route_eligible_test_observations")
                if fold_blockers:
                    component_blockers.append("one_or_more_route_folds_incomplete")
                component_folds.append(
                    {
                        "held_out_scenario_id": scenario_id,
                        "train_observations": len(train_route),
                        "test_observations": len(test_route),
                        "train_observation_ids_sha256": sha256_json(
                            sorted(
                                candidate["record"]["source_observation_id"]
                                for candidate in train_route
                            )
                        ),
                        "test_observation_ids_sha256": sha256_json(
                            sorted(
                                candidate["record"]["source_observation_id"]
                                for candidate in test_route
                            )
                        ),
                        "blockers": fold_blockers,
                    }
                )
            component_status[route] = {
                "observations": len(route_candidates),
                "scenarios": len(route_by_scenario),
                "folds": len(component_folds),
                "fold_set_sha256": sha256_json(component_folds),
                "tier_counts": _tier_counts(route_candidates),
                "blockers": sorted(set(component_blockers)),
                "ready": bool(route_candidates and not component_blockers),
            }
            component_cohorts[route].append(
                {
                    "runtime_cohort_id": cohort_id,
                    **component_status[route],
                }
            )

        cohort_blockers = list(cohort_integrity_blockers)
        if len(by_scenario) < MIN_LOOCV_SCENARIOS:
            cohort_blockers.append(f"fewer_than_{MIN_LOOCV_SCENARIOS}_scenarios")
        if any(fold["blockers"] for fold in folds):
            cohort_blockers.append("one_or_more_folds_incomplete")
        any_primary_component_ready = any(
            component_status[route]["ready"] for route in PRIMARY_RESOURCE_COMPONENTS
        )
        cohort_rows.append(
            {
                "runtime_cohort_id": cohort_id,
                "runtime_cohort_material": cohort[0]["runtime_cohort_material"],
                "observations": len(cohort),
                "scenarios": len(by_scenario),
                "folds": folds,
                "fold_membership_sha256": sha256_json(
                    [
                        {
                            "fold_id": fold["fold_id"],
                            "train": fold["train_observation_ids_sha256"],
                            "test": fold["test_observation_ids_sha256"],
                        }
                        for fold in folds
                    ]
                ),
                "components": component_status,
                "blockers": sorted(set(cohort_blockers)),
                "ready": bool(any_primary_component_ready and not cohort_blockers),
            }
        )

    every_candidate_tests_once = bool(candidates) and all(
        global_test_counts[candidate["record"]["source_observation_id"]] == 1
        for candidate in candidates
    )
    integrity_ready = not invalid and every_candidate_tests_once
    components = {}
    for route in RESOURCE_COMPONENTS:
        statuses = component_cohorts[route]
        eligible_ids = sorted(
            status["runtime_cohort_id"]
            for status in statuses
            if status["ready"] and integrity_ready
        )
        dropped = [
            {
                "runtime_cohort_id": status["runtime_cohort_id"],
                "blockers": status["blockers"]
                or (
                    ["global_resource_integrity_failed"] if not integrity_ready else []
                ),
            }
            for status in statuses
            if status["runtime_cohort_id"] not in eligible_ids
        ]
        route_candidates = [
            candidate
            for candidate in candidates
            if (candidate["record"].get("measurement_eligibility") or {}).get(route)
            is True
        ]
        global_route_by_scenario = {
            scenario_id: [
                candidate
                for candidate in group
                if (candidate["record"].get("measurement_eligibility") or {}).get(
                    route
                )
                is True
            ]
            for scenario_id, group in global_by_scenario.items()
        }
        global_route_by_scenario = {
            scenario_id: group
            for scenario_id, group in global_route_by_scenario.items()
            if group
        }
        global_component_folds = []
        global_component_blockers = []
        if len(global_route_by_scenario) < MIN_LOOCV_SCENARIOS:
            global_component_blockers.append(
                f"fewer_than_{MIN_LOOCV_SCENARIOS}_eligible_scenarios"
            )
        for scenario_id in sorted(global_route_by_scenario):
            test_route = global_route_by_scenario[scenario_id]
            train_route = [
                candidate
                for other_id, group in global_route_by_scenario.items()
                if other_id != scenario_id
                for candidate in group
            ]
            train_ids = sorted(
                candidate["record"]["source_observation_id"]
                for candidate in train_route
            )
            test_ids = sorted(
                candidate["record"]["source_observation_id"]
                for candidate in test_route
            )
            fold_blockers = []
            if not train_ids:
                fold_blockers.append("no_route_eligible_training_observations")
            if not test_ids:
                fold_blockers.append("no_route_eligible_test_observations")
            if set(train_ids).intersection(test_ids):
                fold_blockers.append("train_test_membership_overlap")
            if fold_blockers:
                global_component_blockers.append(
                    "one_or_more_global_route_folds_incomplete"
                )
            train_cohorts = {
                str(candidate["runtime_cohort_id"]) for candidate in train_route
            }
            test_cohorts = {
                str(candidate["runtime_cohort_id"]) for candidate in test_route
            }
            global_component_folds.append(
                {
                    "held_out_scenario_id": scenario_id,
                    "train_observations": len(train_ids),
                    "test_observations": len(test_ids),
                    "train_observation_ids_sha256": sha256_json(train_ids),
                    "test_observation_ids_sha256": sha256_json(test_ids),
                    "unseen_test_runtime_cohort_ids": sorted(
                        test_cohorts - train_cohorts
                    ),
                    "blockers": sorted(set(fold_blockers)),
                }
            )
        global_component_ready = bool(
            route_candidates
            and integrity_ready
            and not global_component_blockers
        )
        components[route] = {
            "observations": len(route_candidates),
            "tier_counts": _tier_counts(route_candidates),
            "eligible_runtime_cohort_ids": eligible_ids,
            "dropped_runtime_cohort_ids": sorted(
                item["runtime_cohort_id"] for item in dropped
            ),
            "dropped_cohorts": dropped,
            "cohorts": statuses,
            "global_scenarios": len(global_route_by_scenario),
            "global_folds": global_component_folds,
            "global_fold_set_sha256": sha256_json(global_component_folds),
            "global_blockers": sorted(set(global_component_blockers)),
            "ready_for_bounded_fit": global_component_ready,
            "is_primary_throughput_evidence": route == "throughput_primary",
            "is_low_fidelity_screening_only": route == "throughput_screen_only",
        }

    current_fold_set_sha256 = sha256_json(sorted(global_fold_ids))
    prior = (context.get("validation_context") or {}).get("resource_loocv") or {}
    prior_fold_digest = prior.get("fold_set_sha256") or prior.get(
        "fold_membership_sha256"
    )
    prior_membership_matches = bool(
        isinstance(prior_fold_digest, str)
        and prior_fold_digest == current_fold_set_sha256
    )
    prior_comparison_status = (
        "membership_matched"
        if prior_membership_matches
        else "membership_mismatch"
        if isinstance(prior_fold_digest, str)
        else "historical_reference_unbound"
    )
    any_component_ready = any(
        components[route]["ready_for_bounded_fit"]
        for route in PRIMARY_RESOURCE_COMPONENTS
    )
    full_resource_bundle_ready = all(
        components[route]["ready_for_bounded_fit"]
        for route in PRIMARY_RESOURCE_COMPONENTS
    )
    return {
        "policy": LOOCV_POLICY,
        "role_semantics": "fold_dependent_not_static_per_observation",
        "fold_scope": "global_scenario_across_all_runtime_cohorts",
        "scenario_dimensions": [
            "model_id",
            "train_type",
            "dataset_id",
            "target_gbs",
        ],
        "minimum_scenarios_per_component_cohort": MIN_LOOCV_SCENARIOS,
        "sampling_roles_are_partitions": False,
        "candidate_observations": len(candidates),
        "invalid_observations": invalid,
        "runtime_cohorts": len(cohort_rows),
        "ready_runtime_cohorts": sum(row["ready"] for row in cohort_rows),
        "folds": len(global_fold_ids),
        "global_folds": global_folds,
        "cohort_local_folds_diagnostic_only": len(all_fold_ids),
        "fold_set_sha256": current_fold_set_sha256,
        "every_observation_is_test_exactly_once": every_candidate_tests_once,
        "cohorts": cohort_rows,
        "components": components,
        "prior_validation_comparison": {
            "status": prior_comparison_status,
            "current_fold_set_sha256": current_fold_set_sha256,
            "prior_fold_set_sha256": prior_fold_digest,
            "current_folds": len(all_fold_ids),
            "prior_folds": prior.get("folds"),
            "membership_matches_current": prior_membership_matches,
            "metrics_usable_for_current_fold_set": prior_membership_matches,
        },
        "any_component_fit_ready": any_component_ready,
        "full_resource_bundle_ready": full_resource_bundle_ready,
        "ready_for_bounded_fit": any_component_ready,
    }


def _profiler_partition(
    records: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    profiler = [
        record
        for record in records
        if (record.get("validation_design") or {}).get("mode") == "fixed_partition"
    ]
    artifact = (context.get("validation_context") or {}).get("profiler") or {}
    fallback_rows = artifact.get("fallback_promotions") or []
    fallback_by_job = {
        str(item.get("job_id")): item
        for item in fallback_rows
        if isinstance(item, dict) and isinstance(item.get("job_id"), str)
    }
    declared = Counter()
    effective = Counter()
    eligible_items: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    observed_fallback_jobs: set[str] = set()
    for record in profiler:
        design = record["validation_design"]
        row = rows[record["source_observation_id"]]
        job = (row.get("configuration") or {}).get("job") or {}
        cohort_id, cohort_material, runtime_issues = _runtime_identity(record)
        issues = list(runtime_issues)
        if design.get("policy") != PROFILER_POLICY:
            issues.append("profiler_partition_policy_mismatch")
        if job.get("kind") != "profiler":
            issues.append("fixed_partition_source_job_is_not_profiler")
        declared_role = design.get("declared_role")
        original_role = design.get("original_role")
        effective_role = design.get("effective_role")
        declared_role_source = design.get("declared_role_source")
        effective_role_source = design.get("effective_role_source")
        declared[str(declared_role)] += 1
        effective[str(effective_role)] += 1
        if declared_role not in PROFILER_DECLARED_ROLES:
            issues.append("profiler_declared_role_invalid")
        if effective_role not in PROFILER_EFFECTIVE_ROLES:
            issues.append("profiler_effective_role_invalid")
        if job.get("profiler_role") != declared_role:
            issues.append("profiler_declared_role_source_job_mismatch")
        if declared_role_source != "authorized_job_payload":
            issues.append("profiler_declared_role_source_invalid")
        if effective_role_source not in {
            "profiler_calibration_artifact",
            "authorized_job_payload_fallback",
        }:
            issues.append("profiler_effective_role_source_invalid")
        if original_role != declared_role:
            issues.append("profiler_original_role_mismatch")
        if declared_role == "calibration" and effective_role != "calibration":
            issues.append("profiler_calibration_role_transition_invalid")
        if declared_role == "holdout" and effective_role not in {
            "holdout",
            "fallback_calibration",
        }:
            issues.append("profiler_holdout_role_transition_invalid")
        if effective_role == "fallback_calibration":
            if effective_role_source != "profiler_calibration_artifact":
                issues.append("profiler_fallback_effective_role_source_invalid")
            observed_fallback_jobs.add(str(record.get("job_id")))
            promotion = fallback_by_job.get(str(record.get("job_id")))
            if not isinstance(promotion, dict):
                issues.append("profiler_fallback_promotion_not_source_bound")
            elif (
                promotion.get("original_role") != "holdout"
                or promotion.get("effective_role") != "fallback_calibration"
            ):
                issues.append("profiler_fallback_promotion_binding_mismatch")
        if issues:
            invalid.append(
                {
                    "job_id": record.get("job_id"),
                    "issues": sorted(set(issues)),
                }
            )
            continue
        if record["measurement_eligibility"].get("profiler") is not True:
            continue
        item = {
            "job_id": record["job_id"],
            "features": [
                1.0,
                1.0 if str(job.get("train_type") or "").lower() == "lora" else 0.0,
                1.0 if job.get("gc") is True else 0.0,
            ],
            "declared_role": declared_role,
            "original_role": original_role,
            "effective_role": effective_role,
            "runtime_cohort_id": cohort_id,
            "runtime_cohort_material": cohort_material,
            "evidence_tier": str(record.get("evidence_tier") or "unknown"),
        }
        eligible_items.append(item)

    unknown_fallbacks = sorted(set(fallback_by_job) - observed_fallback_jobs)
    if unknown_fallbacks:
        invalid.append(
            {
                "job_id": None,
                "issues": ["profiler_fallback_context_contains_unmatched_jobs"],
                "unmatched_job_ids": unknown_fallbacks,
            }
        )

    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in eligible_items:
        by_cohort[str(item["runtime_cohort_id"])].append(item)
    required_rank = 3
    cohort_rows = []
    all_calibration: list[dict[str, Any]] = []
    all_holdout: list[dict[str, Any]] = []
    for cohort_id in sorted(by_cohort):
        cohort = by_cohort[cohort_id]
        material_hashes = {
            sha256_json(item["runtime_cohort_material"]) for item in cohort
        }
        calibration = [
            item
            for item in cohort
            if item["effective_role"] in {"calibration", "fallback_calibration"}
        ]
        holdout = [item for item in cohort if item["effective_role"] == "holdout"]
        all_calibration.extend(calibration)
        all_holdout.extend(holdout)
        feature_rank = _matrix_rank([item["features"] for item in calibration])
        fit_blockers = []
        if len(material_hashes) != 1:
            fit_blockers.append("runtime_cohort_material_inconsistent_within_id")
        if feature_rank < required_rank:
            fit_blockers.append("profiler_calibration_feature_matrix_rank_deficient")
        holdout_blockers = list(fit_blockers)
        if not holdout:
            holdout_blockers.append("independent_profiler_holdout_missing")
        cohort_rows.append(
            {
                "runtime_cohort_id": cohort_id,
                "runtime_cohort_material": cohort[0]["runtime_cohort_material"],
                "measurement_eligible": len(cohort),
                "effective_calibration_job_ids": sorted(
                    item["job_id"] for item in calibration
                ),
                "remaining_holdout_job_ids": sorted(item["job_id"] for item in holdout),
                "feature_matrix_rank": feature_rank,
                "required_feature_matrix_rank": required_rank,
                "tier_counts": dict(
                    sorted(Counter(item["evidence_tier"] for item in cohort).items())
                ),
                "fit_blockers": fit_blockers,
                "holdout_blockers": holdout_blockers,
                "ready_for_bounded_fit": not fit_blockers,
                "ready_for_independent_holdout_validation": not holdout_blockers,
            }
        )

    actual_calibration_ids = {item["job_id"] for item in all_calibration}
    actual_holdout_ids = {item["job_id"] for item in all_holdout}
    artifact_calibration_ids, calibration_binding_issues = _job_id_set(
        artifact,
        "artifact_calibration_job_ids",
        "artifact_calibration_job_ids_sha256",
    )
    evaluation_ids, evaluation_binding_issues = _job_id_set(
        artifact,
        "evaluation_job_ids",
        "evaluation_job_ids_sha256",
    )
    artifact_binding_issues = calibration_binding_issues + evaluation_binding_issues
    if artifact_calibration_ids is not None and (
        artifact_calibration_ids != actual_calibration_ids
    ):
        artifact_binding_issues.append("artifact_calibration_job_set_mismatch")
    if evaluation_ids is not None:
        if not evaluation_ids:
            artifact_binding_issues.append("artifact_evaluation_job_set_empty")
        if not evaluation_ids.issubset(actual_holdout_ids):
            artifact_binding_issues.append(
                "artifact_evaluation_jobs_not_current_holdout"
            )
        if evaluation_ids.intersection(actual_calibration_ids):
            artifact_binding_issues.append("artifact_calibration_evaluation_overlap")
        artifact_count = int(artifact.get("remaining_evaluation_points") or 0)
        if artifact_count != len(evaluation_ids):
            artifact_binding_issues.append("artifact_evaluation_point_count_mismatch")
    artifact_binding_issues = sorted(set(artifact_binding_issues))
    artifact_membership_bound = not artifact_binding_issues and not invalid
    fit_ready_cohorts = sorted(
        row["runtime_cohort_id"]
        for row in cohort_rows
        if row["ready_for_bounded_fit"] and not invalid
    )
    holdout_ready_cohorts = sorted(
        row["runtime_cohort_id"]
        for row in cohort_rows
        if row["ready_for_independent_holdout_validation"] and not invalid
    )
    aggregate_feature_rank = _matrix_rank(
        [item["features"] for item in all_calibration]
    )
    blockers = []
    if invalid:
        blockers.append("invalid_profiler_partition_records")
    if aggregate_feature_rank < required_rank:
        blockers.append("profiler_calibration_feature_matrix_rank_deficient")
    if not fit_ready_cohorts:
        blockers.append("no_runtime_cohort_has_full_rank_profiler_calibration")
    return {
        "policy": PROFILER_POLICY,
        "records": len(profiler),
        "measurement_eligible": sum(
            record["measurement_eligibility"].get("profiler") is True
            for record in profiler
        ),
        "declared_roles": dict(sorted(declared.items())),
        "effective_roles": dict(sorted(effective.items())),
        "effective_calibration_job_ids": sorted(actual_calibration_ids),
        "remaining_holdout_job_ids": sorted(actual_holdout_ids),
        "feature_names": ["intercept", "is_lora", "gc_enabled"],
        "feature_matrix_rank": aggregate_feature_rank,
        "required_feature_matrix_rank": required_rank,
        "runtime_cohorts": cohort_rows,
        "eligible_fit_runtime_cohort_ids": fit_ready_cohorts,
        "eligible_holdout_runtime_cohort_ids": holdout_ready_cohorts,
        "dropped_fit_runtime_cohort_ids": sorted(
            set(by_cohort) - set(fit_ready_cohorts)
        ),
        "actual_remaining_holdout_points": len(actual_holdout_ids),
        "fallback_promotions": artifact.get("fallback_promotions") or [],
        "prior_evaluation": {
            "status": (
                "membership_bound"
                if artifact_membership_bound
                else "historical_reference_unbound"
                if any(issue.endswith("_missing") for issue in artifact_binding_issues)
                else "membership_mismatch"
            ),
            "artifact_calibration_job_ids": sorted(artifact_calibration_ids or []),
            "evaluation_job_ids": sorted(evaluation_ids or []),
            "binding_issues": artifact_binding_issues,
            "membership_matches_current": artifact_membership_bound,
            "metrics_usable_for_current_partition": artifact_membership_bound,
            "evaluation_mape": artifact.get("evaluation_mape"),
        },
        "invalid_records": invalid,
        "blockers": blockers,
        "ready_for_bounded_fit": bool(fit_ready_cohorts and not invalid),
        "ready_for_independent_holdout_validation": bool(
            holdout_ready_cohorts and not invalid
        ),
        "independent_holdout_validation_completed": artifact_membership_bound,
    }


def _packing_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    pair_records = [
        record
        for record in records
        if record["measurement_eligibility"].get("packing_pair") is True
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_records = []
    for record in pair_records:
        design = record.get("validation_design") or {}
        pair_id = design.get("pair_id")
        treatment = design.get("treatment")
        _, _, runtime_issues = _runtime_identity(record)
        issues = list(runtime_issues)
        if design.get("mode") != "paired_comparison":
            issues.append("packing_pair_mode_mismatch")
        if design.get("policy") != PACKING_POLICY:
            issues.append("packing_pair_policy_mismatch")
        if not isinstance(pair_id, str) or not pair_id:
            issues.append("packing_pair_id_missing")
        if treatment not in {"packed", "unpacked"}:
            issues.append("packing_pair_treatment_invalid")
        if issues:
            invalid_records.append(
                {
                    "job_id": record.get("job_id"),
                    "issues": sorted(set(issues)),
                }
            )
            continue
        grouped[pair_id].append(record)
    pairs = []
    for pair_id in sorted(grouped):
        group = grouped[pair_id]
        treatments = Counter(
            str(record["validation_design"].get("treatment")) for record in group
        )
        job_ids = [str(record.get("job_id") or "") for record in group]
        cohort_ids = {
            str((record.get("runtime") or {}).get("runtime_cohort_id") or "")
            for record in group
        }
        issues = []
        if len(group) != 2:
            issues.append("packing_pair_row_count_not_two")
        if treatments != Counter({"packed": 1, "unpacked": 1}):
            issues.append("packing_pair_treatments_not_exactly_one_per_side")
        if len(job_ids) != len(set(job_ids)) or any(not job_id for job_id in job_ids):
            issues.append("packing_pair_job_ids_missing_or_duplicate")
        if len(cohort_ids) != 1:
            issues.append("packing_pair_crosses_runtime_cohorts")
        complete = not issues
        pairs.append(
            {
                "pair_id": pair_id,
                "job_ids": sorted(job_ids),
                "observation_rows": len(group),
                "treatments": dict(sorted(treatments.items())),
                "complete": complete,
                "issues": issues,
            }
        )
    complete_pairs = sum(pair["complete"] for pair in pairs)
    complete_pair_ids = sorted(pair["pair_id"] for pair in pairs if pair["complete"])
    incomplete_pair_ids = sorted(
        pair["pair_id"] for pair in pairs if not pair["complete"]
    )
    fit_observation_rows = sum(
        pair["observation_rows"] for pair in pairs if pair["complete"]
    )
    return {
        "policy": PACKING_POLICY,
        "candidate_observation_rows": len(pair_records),
        "valid_observation_rows": sum(len(group) for group in grouped.values()),
        "fit_observation_rows": fit_observation_rows,
        "observations": len(pair_records),
        "pairs": pairs,
        "complete_pairs": complete_pairs,
        "complete_pair_ids": complete_pair_ids,
        "incomplete_pair_ids": incomplete_pair_ids,
        "invalid_records": invalid_records,
        "packing_memory_safety_rows": sum(
            record["measurement_eligibility"].get("packing_memory_safety") is True
            for record in records
        ),
        "evidence_strength": "historical_single_pair_per_side",
        "is_abba": False,
        "ready_for_low_confidence_effect_fit": complete_pairs > 0,
        "fit_uses_only_complete_pairs": True,
        "ready_for_automatic_packing_enablement": False,
        "automatic_enablement_blockers": ["historical_pairs_are_not_abba"],
    }


def analyze(
    recovery_report: dict[str, Any], rows: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if recovery_report.get("schema") != RECOVERY_SCHEMA:
        raise ValueError("Historical recovery report schema is invalid")
    records = recovery_report.get("records") or []
    recovery_ids = {record.get("source_observation_id") for record in records}
    if recovery_ids != set(rows):
        missing = sorted(set(rows) - recovery_ids)
        extra = sorted(recovery_ids - set(rows))
        raise ValueError(
            "Historical recovery must bind one-to-one with observations: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    for record in records:
        row = rows[record["source_observation_id"]]
        job = (row.get("configuration") or {}).get("job") or {}
        if record.get("job_id") != job.get("job_id"):
            raise ValueError(
                f"Historical recovery job mismatch: {record.get('source_observation_id')}"
            )

    resource = _resource_loocv(records, rows, recovery_report)
    profiler = _profiler_partition(records, rows, recovery_report)
    packing = _packing_pairs(records)
    counts = recovery_report.get("counts") or {}
    warnings = [
        "historical_recovery_is_not_native_v2_execution_evidence",
        "resource_validation_uses_global_scenario_holdout_across_runtime_cohorts",
        "hash_only_runtime_cohorts_require_uncertainty_inflation_or_fixed_effects",
        "throughput_screen_rows_are_low_fidelity_and_not_primary_throughput_labels",
        "historical_packing_pairs_are_not_abba",
        "recorded_profiler_holdout_validates_only_the_auxiliary_historical_fit_not_the_theory_planner",
        "publication_still_requires_current_runtime_prospective_acceptance",
    ]
    component_readiness = {
        "resource_feasibility": resource["components"]["feasibility"][
            "ready_for_bounded_fit"
        ],
        "resource_memory_boundary": resource["components"]["memory_boundary"][
            "ready_for_bounded_fit"
        ],
        "resource_throughput_primary": resource["components"]["throughput_primary"][
            "ready_for_bounded_fit"
        ],
        "resource_throughput_screening_only": resource["components"][
            "throughput_screen_only"
        ]["ready_for_bounded_fit"],
        "profiler_bounded_fit": profiler["ready_for_bounded_fit"],
        "profiler_independent_holdout_available": profiler[
            "ready_for_independent_holdout_validation"
        ],
        "packing_low_confidence_effect_fit": packing[
            "ready_for_low_confidence_effect_fit"
        ],
    }
    model_fit_components = (
        "resource_feasibility",
        "resource_memory_boundary",
        "resource_throughput_primary",
        "profiler_bounded_fit",
        "packing_low_confidence_effect_fit",
    )
    any_fit_ready = any(component_readiness[name] for name in model_fit_components)
    full_bundle_ready = bool(
        component_readiness["resource_feasibility"]
        and component_readiness["resource_memory_boundary"]
        and component_readiness["resource_throughput_primary"]
        and component_readiness["profiler_bounded_fit"]
        and component_readiness["profiler_independent_holdout_available"]
    )
    evidence_tiers = Counter(
        str(record.get("evidence_tier") or "unknown") for record in records
    )
    requirements = {
        "uncertainty_inflation_required": bool(
            any_fit_ready and int(evidence_tiers.get("legacy_consistent") or 0) > 0
        ),
        "runtime_cohort_fixed_effects_required": bool(
            any_fit_ready and resource.get("runtime_cohorts", 0) > 1
        ),
        "legacy_consistent_is_acceptance_evidence": False,
        "recorded_profiler_mape_is_planner_acceptance_evidence": False,
        "prospective_acceptance_holdout_required": True,
        "historical_evidence_is_publication_evidence": False,
    }
    return {
        "schema": SCHEMA,
        "source_recovery_sha256": recovery_report.get("report_sha256"),
        "counts": counts,
        "resource_loocv": resource,
        "profiler_fixed_partition": profiler,
        "packing_paired_evidence": packing,
        "prior_validation_context": recovery_report.get("validation_context"),
        "component_readiness": component_readiness,
        "ready_components": sorted(
            name for name, ready in component_readiness.items() if ready
        ),
        "blocked_components": sorted(
            name for name, ready in component_readiness.items() if not ready
        ),
        "any_fit_ready": any_fit_ready,
        "full_bundle_ready": full_bundle_ready,
        "ready_for_any_bounded_fit": any_fit_ready,
        "requirements": requirements,
        "ready_for_prospective_acceptance": False,
        "calibration_publishable": False,
        "warnings": warnings,
    }
