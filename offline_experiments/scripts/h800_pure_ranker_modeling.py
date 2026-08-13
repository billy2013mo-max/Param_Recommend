#!/usr/bin/env python3
"""Fit and evaluate an H800 configuration-only pairwise throughput ranker.

This analysis reuses the canonical H800 admission and split protocol, but
forces the throughput model to use only the 21 basic configuration/mechanism
features:

    score = intercept + standardized_21_configuration_features @ beta

Hyperparameters are selected only by leave-one-scenario-out cross-validation
on the native calibration partition.  The selected model is then frozen using
historical plus native calibration candidates and compared with the existing
physics baseline and frozen hybrid ranker on the same native holdout rows.

The script is offline and diagnostic: it does not launch GPU work, mutate a
queue, publish a production profile, or treat the previously inspected holdout
as fresh publication acceptance.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from common import ROOT, read_json, sha256_file, sha256_json, write_json
from h800_challenger_modeling import (
    SCHEMA as HYBRID_SCHEMA,
    THROUGHPUT_ALPHA_GRID,
    THROUGHPUT_BASIC_FEATURES,
    THROUGHPUT_HISTORICAL_WEIGHT_GRID,
    _baseline_ranker_model,
    _build_native_throughput_record,
    _fit_pairwise_ranker,
    _inventory_models,
    _is_legacy,
    _observation_id,
    _outcome,
    _ranking_evaluation,
    _read_observations,
    _same_split_audit,
    _throughput_candidates,
    _throughput_cv,
    _verify_bound_report,
    throughput_admission_reason,
)
from h800_theory_calibration import scenario_id


SCHEMA = "sft_h800_pure_ranker_modeling/v1"
IMPLEMENTATION_VERSION = (
    "sft_h800_pure_ranker_modeling_impl/"
    "2026-07-28.configuration-only-pairwise-ridge"
)
FEATURE_SET = "basic"
USE_PHYSICS_BASE = False


def _finite(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite")
    return number


def _selection_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    """Apply the original H800 ranking-model selection policy."""

    cv = candidate["cv"]
    return (
        _finite(
            cv["scenario_gpu_equal_top1_regret"],
            name="scenario_gpu_equal_top1_regret",
        ),
        -_finite(
            cv["scenario_equal_pairwise_accuracy"],
            name="scenario_equal_pairwise_accuracy",
        ),
        _finite(
            cv["scenario_equal_top1_regret"],
            name="scenario_equal_top1_regret",
        ),
        _finite(
            candidate["historical_scenario_weight"],
            name="historical_scenario_weight",
        ),
        _finite(candidate["alpha"], name="alpha"),
    )


def _select_pure_ranker(
    historical: Sequence[Mapping[str, Any]],
    native_calibration: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for historical_weight in THROUGHPUT_HISTORICAL_WEIGHT_GRID:
        for alpha in THROUGHPUT_ALPHA_GRID:
            candidates.append(
                {
                    "feature_set": FEATURE_SET,
                    "feature_dimension": len(THROUGHPUT_BASIC_FEATURES),
                    "historical_scenario_weight": historical_weight,
                    "alpha": alpha,
                    "use_physics_base": USE_PHYSICS_BASE,
                    "cv": _throughput_cv(
                        historical,
                        native_calibration,
                        feature_set=FEATURE_SET,
                        alpha=alpha,
                        historical_weight=historical_weight,
                        use_physics_base=USE_PHYSICS_BASE,
                    ),
                }
            )
    selected = min(candidates, key=_selection_key)
    return {
        "selection_policy": (
            "minimum calibration-LOSO scenario×GPU top1 regret; "
            "scenario-equal pairwise accuracy then overall top1 regret, "
            "historical weight, and ridge alpha are tie-breakers"
        ),
        "holdout_used_for_selection": False,
        "candidate_count": len(candidates),
        "selected": selected,
        "candidates": candidates,
    }


def _metric_delta(
    pure: Mapping[str, Any],
    hybrid: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "pooled_pairwise_accuracy_pure_minus_hybrid": (
            float(pure["pooled_pairwise_accuracy"])
            - float(hybrid["pooled_pairwise_accuracy"])
        ),
        "scenario_equal_pairwise_accuracy_pure_minus_hybrid": (
            float(pure["scenario_equal_pairwise_accuracy"])
            - float(hybrid["scenario_equal_pairwise_accuracy"])
        ),
        "scenario_equal_top1_regret_pure_minus_hybrid": (
            float(pure["scenario_equal_top1_regret"])
            - float(hybrid["scenario_equal_top1_regret"])
        ),
        "scenario_equal_hit_at_10_percent_pure_minus_hybrid": (
            float(pure["scenario_equal_hit_at_10_percent"])
            - float(hybrid["scenario_equal_hit_at_10_percent"])
        ),
        "scenario_gpu_equal_top1_regret_pure_minus_hybrid": (
            float(pure["scenario_gpu_equal_top1_regret"])
            - float(hybrid["scenario_gpu_equal_top1_regret"])
        ),
        "scenario_gpu_equal_hit_at_10_percent_pure_minus_hybrid": (
            float(pure["scenario_gpu_equal_hit_at_10_percent"])
            - float(hybrid["scenario_gpu_equal_hit_at_10_percent"])
        ),
    }


def _enrich_ranking_metrics(
    metrics: dict[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_scenario[str(candidate["scenario_id"])].append(candidate)
    comparable_scenarios = sum(
        len(rows) >= 2 for rows in by_scenario.values()
    )
    pairwise_rows = int(metrics["pairwise_rows"])
    correct_float = (
        pairwise_rows * float(metrics["pooled_pairwise_accuracy"])
    )
    pairwise_correct = int(round(correct_float))
    if not math.isclose(
        correct_float, pairwise_correct, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("Pairwise correct count is not integral")
    return {
        **metrics,
        "comparable_scenario_rows": comparable_scenarios,
        "pairwise_correct_rows": pairwise_correct,
    }


def _assert_same_metrics(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    exact_fields = ("candidate_rows", "scenario_rows", "pairwise_rows")
    float_fields = (
        "pooled_pairwise_accuracy",
        "scenario_equal_pairwise_accuracy",
        "scenario_equal_top1_regret",
        "scenario_equal_hit_at_10_percent",
        "scenario_gpu_equal_top1_regret",
        "scenario_gpu_equal_hit_at_10_percent",
    )
    for field in exact_fields:
        if observed.get(field) != expected.get(field):
            raise ValueError(
                f"{label} {field} drifted: "
                f"{observed.get(field)!r} != {expected.get(field)!r}"
            )
    for field in float_fields:
        left = observed.get(field)
        right = expected.get(field)
        if left is None or right is None:
            if left != right:
                raise ValueError(
                    f"{label} {field} availability drifted: "
                    f"{left!r} != {right!r}"
                )
            continue
        if not math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"{label} {field} drifted: {left!r} != {right!r}"
            )


def _scenario_overlap(
    historical: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    holdout: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    historical_ids = {str(candidate["scenario_id"]) for candidate in historical}
    calibration_ids = {
        str(candidate["scenario_id"]) for candidate in calibration
    }
    holdout_ids = {str(candidate["scenario_id"]) for candidate in holdout}
    return {
        "historical_scenarios": len(historical_ids),
        "native_calibration_scenarios": len(calibration_ids),
        "native_holdout_scenarios": len(holdout_ids),
        "historical_and_native_calibration_overlap": sorted(
            historical_ids & calibration_ids
        ),
        "historical_and_native_holdout_overlap": sorted(
            historical_ids & holdout_ids
        ),
        "native_calibration_and_holdout_overlap": sorted(
            calibration_ids & holdout_ids
        ),
    }


def _freeze_pure_ranker(
    candidates: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    model = _fit_pairwise_ranker(
        candidates,
        feature_set=FEATURE_SET,
        alpha=float(selected["alpha"]),
        historical_weight=float(selected["historical_scenario_weight"]),
        use_physics_base=USE_PHYSICS_BASE,
    )
    return {
        **model,
        "training_observation_ids_sha256": sha256_json(
            sorted(
                observation_id
                for candidate in candidates
                for observation_id in candidate["observation_ids"]
                if (
                    float(selected["historical_scenario_weight"]) > 0
                    or candidate.get("historical") is not True
                )
            )
        ),
        "holdout_rows_in_training": 0,
    }


def build_report(
    *,
    observation_path: Path,
    theory_basis_path: Path,
    theory_calibration_path: Path,
    inventory_path: Path,
    hardware_path: Path,
    runtime_root: Path,
    hybrid_report_path: Path,
) -> dict[str, Any]:
    observations = _read_observations(observation_path)
    theory_basis = read_json(theory_basis_path)
    theory_calibration = read_json(theory_calibration_path)
    inventory = read_json(inventory_path)
    hardware = read_json(hardware_path)
    hybrid_report = read_json(hybrid_report_path)
    _verify_bound_report(
        theory_basis,
        expected_schema="sft_h800_theory_basis/v1",
        name="theory basis",
    )
    _verify_bound_report(
        theory_calibration,
        expected_schema="sft_h800_theory_calibration/v1",
        name="theory calibration",
    )
    _verify_bound_report(
        hybrid_report,
        expected_schema=HYBRID_SCHEMA,
        name="H800 hybrid challenger",
    )
    if "h800" not in str(
        hardware.get("name_reported_by_driver") or ""
    ).lower():
        raise ValueError("Pure-ranker modeling is H800-only")

    model_by_id, fixed_lora = _inventory_models(inventory)
    admission = Counter()
    native_records: list[dict[str, Any]] = []
    for row in observations:
        reason = throughput_admission_reason(row)
        admission[reason] += 1
        if reason == "admitted":
            native_records.append(
                _build_native_throughput_record(
                    row,
                    model_by_id=model_by_id,
                    fixed_lora=fixed_lora,
                    hardware=hardware,
                    runtime_root=runtime_root,
                )
            )
    native_records.sort(key=_observation_id)
    split = _same_split_audit(native_records)
    if split["disjoint"] is not True:
        raise ValueError("Native calibration and holdout split units overlap")
    calibration_records = [
        record
        for record in native_records
        if record["calibration_partition"]["role"] == "calibration"
    ]
    holdout_records = [
        record
        for record in native_records
        if record["calibration_partition"]["role"] == "holdout"
    ]

    basis_records = theory_basis.get("records")
    if not isinstance(basis_records, list):
        raise ValueError("Theory basis has no records")
    historical_records = [
        record
        for record in basis_records
        if isinstance(record, Mapping)
        and (record.get("route") or {}).get("throughput_primary") is True
        and _outcome(record) == "success"
    ]
    physical_model = (
        theory_calibration["full_historical_bootstrap_fit"]["throughput"][
            "center"
        ]
    )
    physical_priors = theory_calibration["physical_priors"]["values"]
    historical_candidates = _throughput_candidates(
        historical_records,
        physical_model=physical_model,
        priors=physical_priors,
    )
    calibration_candidates = _throughput_candidates(
        calibration_records,
        physical_model=physical_model,
        priors=physical_priors,
    )
    holdout_candidates = _throughput_candidates(
        holdout_records,
        physical_model=physical_model,
        priors=physical_priors,
    )

    holdout_scenario_ids = {
        str(candidate["scenario_id"]) for candidate in holdout_candidates
    }
    scenario_clean_historical_candidates = [
        candidate
        for candidate in historical_candidates
        if str(candidate["scenario_id"]) not in holdout_scenario_ids
    ]

    # Protocol-matched model: exactly the same historical/calibration training
    # population used by the existing hybrid challenger.  This is the fair
    # architecture ablation, but historical records overlap native holdout
    # scenario IDs.
    selection = _select_pure_ranker(
        historical_candidates, calibration_candidates
    )
    selected = selection["selected"]
    fit_candidates = [*historical_candidates, *calibration_candidates]
    pure_ranker = _freeze_pure_ranker(fit_candidates, selected)

    # Scenario-clean diagnostic: remove every historical candidate whose
    # complete scenario ID appears in the native holdout before hyperparameter
    # selection and fitting.  This is a stricter generalization check, although
    # it no longer has the same training population as the frozen hybrid model.
    scenario_clean_selection = _select_pure_ranker(
        scenario_clean_historical_candidates, calibration_candidates
    )
    scenario_clean_selected = scenario_clean_selection["selected"]
    scenario_clean_fit_candidates = [
        *scenario_clean_historical_candidates,
        *calibration_candidates,
    ]
    scenario_clean_ranker = _freeze_pure_ranker(
        scenario_clean_fit_candidates, scenario_clean_selected
    )

    pure_holdout = _enrich_ranking_metrics(
        _ranking_evaluation(
            holdout_candidates, pure_ranker, include_details=True
        ),
        holdout_candidates,
    )
    scenario_clean_holdout = _enrich_ranking_metrics(
        _ranking_evaluation(
            holdout_candidates,
            scenario_clean_ranker,
            include_details=True,
        ),
        holdout_candidates,
    )
    physics_holdout = _enrich_ranking_metrics(
        _ranking_evaluation(
            holdout_candidates,
            _baseline_ranker_model(FEATURE_SET),
            include_details=True,
        ),
        holdout_candidates,
    )
    hybrid_model = hybrid_report["throughput"]["frozen_model"]
    hybrid_holdout = _enrich_ranking_metrics(
        _ranking_evaluation(
            holdout_candidates, hybrid_model, include_details=True
        ),
        holdout_candidates,
    )
    stored_comparison = hybrid_report["throughput"][
        "same_native_holdout_comparison"
    ]
    _assert_same_metrics(
        physics_holdout,
        stored_comparison["physics_baseline"],
        label="physics baseline",
    )
    _assert_same_metrics(
        hybrid_holdout,
        stored_comparison["pairwise_challenger"],
        label="hybrid challenger",
    )

    overlap = _scenario_overlap(
        historical_candidates,
        calibration_candidates,
        holdout_candidates,
    )
    blockers = [
        "analysis_is_retrospective_and_does_not_auto_publish",
        "native_holdout_was_previously_inspected_and_is_not_fresh_acceptance",
        "pure_ranker_has_no_gpu_hardware_features_and_is_h800_calibrated",
        "absolute_throughput_is_only_a_diagnostic_for_a_pairwise_objective",
        "packing_effect_evidence_is_excluded",
        "prospective_unseen_scenario_acceptance_is_still_required",
    ]
    if overlap["historical_and_native_holdout_overlap"]:
        blockers.append(
            "historical_training_data_share_scenario_ids_with_native_holdout"
        )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "h800_configuration_only_pairwise_ranker_fitted_and_compared",
        "gpu_family": "H800",
        "analysis_only": True,
        "publishable": False,
        "production_profile_generated": False,
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "model_contract": {
            "model_family": "scenario_equal_pairwise_ridge_ranker",
            "formula": (
                "score = global diagnostic intercept + "
                "standardized_21_configuration_features @ beta"
            ),
            "pairwise_objective": (
                "sum over scenarios of mean squared error between observed "
                "and predicted log-throughput differences + alpha*||beta||^2"
            ),
            "target": "effective_tokens_per_second",
            "primary_use": "ordering_existing_memory-safe_configurations",
            "absolute_prediction_is_diagnostic": True,
            "feature_set": FEATURE_SET,
            "feature_dimension": len(THROUGHPUT_BASIC_FEATURES),
            "feature_names": list(THROUGHPUT_BASIC_FEATURES),
            "use_physics_base": USE_PHYSICS_BASE,
            "hardware_features_present": False,
        },
        "protocol": {
            "model_and_hyperparameter_selection": (
                "native calibration leave-complete-scenario-out only"
            ),
            "holdout_used_for_fit_or_selection": False,
            "holdout_touched_after_pure_ranker_frozen": True,
            "holdout_is_fresh_publication_acceptance": False,
            "same_native_holdout_as_hybrid": True,
            "packing_used": False,
        },
        "source_bindings": {
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "canonical_observations": {
                "path": str(observation_path.resolve()),
                "rows": len(observations),
                "sha256": sha256_file(observation_path),
            },
            "theory_basis": {
                "path": str(theory_basis_path.resolve()),
                "sha256": sha256_file(theory_basis_path),
                "report_sha256": theory_basis["report_sha256"],
            },
            "theory_calibration": {
                "path": str(theory_calibration_path.resolve()),
                "sha256": sha256_file(theory_calibration_path),
                "report_sha256": theory_calibration["report_sha256"],
            },
            "model_inventory": {
                "path": str(inventory_path.resolve()),
                "sha256": sha256_file(inventory_path),
            },
            "hardware": {
                "path": str(hardware_path.resolve()),
                "sha256": sha256_file(hardware_path),
            },
            "hybrid_comparison": {
                "path": str(hybrid_report_path.resolve()),
                "sha256": sha256_file(hybrid_report_path),
                "report_sha256": hybrid_report["report_sha256"],
            },
        },
        "data_admission": {
            "counts": dict(sorted(admission.items())),
            "historical_records": len(historical_records),
            "historical_candidates_after_replicate_pooling": len(
                historical_candidates
            ),
            "native_calibration_records": len(calibration_records),
            "native_calibration_candidates_after_replicate_pooling": len(
                calibration_candidates
            ),
            "native_holdout_records": len(holdout_records),
            "native_holdout_candidates_after_replicate_pooling": len(
                holdout_candidates
            ),
            "scenario_clean_historical_candidates": len(
                scenario_clean_historical_candidates
            ),
            "scenario_clean_fit_candidates": len(
                scenario_clean_fit_candidates
            ),
            "split": split,
            "scenario_overlap": overlap,
        },
        "selection": selection,
        "frozen_model": pure_ranker,
        "scenario_clean_diagnostic": {
            "definition": (
                "all historical candidates sharing a complete scenario ID "
                "with native holdout are removed before calibration-LOSO "
                "hyperparameter selection and final fitting"
            ),
            "fit_scenario_overlap_with_native_holdout": sorted(
                {
                    str(candidate["scenario_id"])
                    for candidate in scenario_clean_fit_candidates
                }
                & holdout_scenario_ids
            ),
            "selection": scenario_clean_selection,
            "frozen_model": scenario_clean_ranker,
        },
        "evaluation": {
            "calibration_loso_selected_candidate": selected["cv"],
            "same_native_holdout_comparison": {
                "same_rows": True,
                "physics_baseline": physics_holdout,
                "existing_hybrid_ranker": hybrid_holdout,
                "pure_configuration_ranker": pure_holdout,
                "pure_scenario_clean_ranker": scenario_clean_holdout,
                "pure_minus_hybrid": _metric_delta(
                    pure_holdout, hybrid_holdout
                ),
                "scenario_clean_pure_minus_hybrid": _metric_delta(
                    scenario_clean_holdout, hybrid_holdout
                ),
            },
        },
        "publication_blockers": blockers,
    }
    report["report_sha256"] = sha256_json(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema") != SCHEMA:
        raise ValueError("Pure-ranker report schema mismatch")
    unsigned = dict(report)
    digest = unsigned.pop("report_sha256", None)
    if digest != sha256_json(unsigned):
        raise ValueError("Pure-ranker report SHA-256 mismatch")
    if (
        report.get("gpu_experiments_launched") is not False
        or report.get("queues_mutated") is not False
    ):
        raise ValueError("Pure-ranker analysis cannot launch or mutate GPU work")
    if (
        report.get("publishable") is not False
        or report.get("production_profile_generated") is not False
    ):
        raise ValueError("Pure-ranker diagnostic cannot auto-publish")
    contract = report.get("model_contract") or {}
    if (
        contract.get("feature_set") != FEATURE_SET
        or contract.get("feature_dimension") != len(THROUGHPUT_BASIC_FEATURES)
        or contract.get("use_physics_base") is not False
        or contract.get("hardware_features_present") is not False
    ):
        raise ValueError("Model is not the required configuration-only ranker")
    protocol = report.get("protocol") or {}
    if (
        protocol.get("holdout_used_for_fit_or_selection") is not False
        or protocol.get("holdout_touched_after_pure_ranker_frozen") is not True
    ):
        raise ValueError("Holdout separation protocol is missing")
    split = (report.get("data_admission") or {}).get("split") or {}
    if split.get("disjoint") is not True or split.get("overlap"):
        raise ValueError("Native calibration/holdout split overlaps")
    frozen = report.get("frozen_model") or {}
    if (
        frozen.get("use_physics_base") is not False
        or frozen.get("feature_set") != FEATURE_SET
        or frozen.get("holdout_rows_in_training") != 0
        or len(frozen.get("coefficients") or [])
        != len(THROUGHPUT_BASIC_FEATURES)
    ):
        raise ValueError("Frozen pure ranker contract is invalid")
    scenario_clean_frozen = (
        (report.get("scenario_clean_diagnostic") or {}).get("frozen_model")
        or {}
    )
    scenario_clean_overlap = (
        (report.get("scenario_clean_diagnostic") or {}).get(
            "fit_scenario_overlap_with_native_holdout"
        )
        or []
    )
    if (
        scenario_clean_frozen.get("use_physics_base") is not False
        or scenario_clean_frozen.get("feature_set") != FEATURE_SET
        or scenario_clean_frozen.get("holdout_rows_in_training") != 0
        or len(scenario_clean_frozen.get("coefficients") or [])
        != len(THROUGHPUT_BASIC_FEATURES)
        or scenario_clean_overlap
    ):
        raise ValueError("Scenario-clean pure ranker contract is invalid")
    comparison = (
        (report.get("evaluation") or {}).get(
            "same_native_holdout_comparison"
        )
        or {}
    )
    row_signatures = {
        (
            metrics.get("candidate_rows"),
            metrics.get("scenario_rows"),
            metrics.get("pairwise_rows"),
        )
        for metrics in (
            comparison.get("physics_baseline") or {},
            comparison.get("existing_hybrid_ranker") or {},
            comparison.get("pure_configuration_ranker") or {},
            comparison.get("pure_scenario_clean_ranker") or {},
        )
    }
    if comparison.get("same_rows") is not True or len(row_signatures) != 1:
        raise ValueError("Holdout comparisons do not use the same rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        type=Path,
        default=ROOT / "artifacts" / "canonical_h800_observations.jsonl",
    )
    parser.add_argument(
        "--theory-basis",
        type=Path,
        default=ROOT / "artifacts" / "h800_theory_basis.json",
    )
    parser.add_argument(
        "--theory-calibration",
        type=Path,
        default=ROOT / "artifacts" / "h800_theory_calibration.json",
    )
    parser.add_argument(
        "--model-inventory",
        type=Path,
        default=ROOT / "artifacts" / "model_inventory.json",
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=ROOT / "config" / "hardware.json",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=ROOT / "runtime",
    )
    parser.add_argument(
        "--hybrid-report",
        type=Path,
        default=ROOT / "artifacts" / "h800_challenger_modeling.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "h800_pure_ranker_modeling.json",
    )
    args = parser.parse_args()
    report = build_report(
        observation_path=args.observations,
        theory_basis_path=args.theory_basis,
        theory_calibration_path=args.theory_calibration,
        inventory_path=args.model_inventory,
        hardware_path=args.hardware,
        runtime_root=args.runtime_root,
        hybrid_report_path=args.hybrid_report,
    )
    write_json(args.output, report)
    comparison = report["evaluation"]["same_native_holdout_comparison"]
    metric_fields = (
        "candidate_rows",
        "scenario_rows",
        "gpu_group_rows",
        "pairwise_rows",
        "pairwise_correct_rows",
        "comparable_scenario_rows",
        "pooled_pairwise_accuracy",
        "scenario_equal_pairwise_accuracy",
        "scenario_equal_top1_regret",
        "scenario_equal_hit_at_10_percent",
        "scenario_gpu_equal_top1_regret",
        "scenario_gpu_equal_hit_at_10_percent",
        "scenario_equal_absolute_throughput_mape",
    )

    def summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
        return {field: metrics.get(field) for field in metric_fields}

    print(
        json.dumps(
            {
                "schema": report["schema"],
                "output": str(args.output),
                "selected": report["selection"]["selected"],
                "holdout": {
                    "physics_baseline": summary(
                        comparison["physics_baseline"]
                    ),
                    "existing_hybrid_ranker": summary(
                        comparison["existing_hybrid_ranker"]
                    ),
                    "pure_configuration_ranker": summary(
                        comparison["pure_configuration_ranker"]
                    ),
                    "pure_scenario_clean_ranker": summary(
                        comparison["pure_scenario_clean_ranker"]
                    ),
                    "pure_minus_hybrid": comparison["pure_minus_hybrid"],
                    "scenario_clean_pure_minus_hybrid": comparison[
                        "scenario_clean_pure_minus_hybrid"
                    ],
                },
                "publishable": report["publishable"],
                "gpu_experiments_launched": report[
                    "gpu_experiments_launched"
                ],
                "report_sha256": report["report_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
