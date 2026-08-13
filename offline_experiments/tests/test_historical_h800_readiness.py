from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import sha256_json  # noqa: E402
from historical_h800_readiness import analyze  # noqa: E402
from recover_h800_historical_evidence import (  # noqa: E402
    LOOCV_POLICY,
    PACKING_POLICY,
    PROFILER_POLICY,
    SCHEMA as RECOVERY_SCHEMA,
)


def resource_row(
    index: int,
    *,
    model: str,
    mbs: int,
    zero: str,
    routes: dict[str, bool] | None = None,
    cohort_material: dict | None = None,
    evidence_tier: str = "legacy_consistent",
) -> tuple[dict, dict]:
    observation_id = f"observation-{index}"
    job = {
        "job_id": f"job-{index}",
        "kind": "throughput",
        "model_id": model,
        "train_type": "full",
        "dataset_id": "data",
        "target_gbs": 64,
        "mbs": mbs,
        "zero": zero,
        "gpu_count": 2,
        "sampling_roles": ["must-not-become-a-partition"],
    }
    scenario = {
        "model_id": model,
        "train_type": "full",
        "dataset_id": "data",
        "target_gbs": 64,
    }
    row = {
        "observation_id": observation_id,
        "configuration": {"job": job},
        "outcome": {"class": "success"},
    }
    runtime_material = cohort_material or {"provenance_sha256": "a" * 64}
    eligibility = {
        "class": "calibration_candidate",
        **(routes or {"feasibility": True, "throughput_primary": True}),
    }
    record = {
        "source_observation_id": observation_id,
        "job_id": job["job_id"],
        "evidence_tier": evidence_tier,
        "runtime": {
            "runtime_cohort_id": sha256_json(runtime_material),
            "runtime_cohort_material": runtime_material,
        },
        "validation_design": {
            "mode": "fold_dependent_loocv",
            "policy": LOOCV_POLICY,
            "explicit_role": "fold_dependent",
            "material": scenario,
            "scenario_id": sha256_json(scenario),
        },
        "measurement_eligibility": eligibility,
    }
    return row, record


def profiler_row(
    index: int,
    *,
    train_type: str,
    gc: bool,
    declared_role: str,
    effective_role: str,
    cohort_material: dict | None = None,
) -> tuple[dict, dict]:
    observation_id = f"profiler-observation-{index}"
    job = {
        "job_id": f"profiler-job-{index}",
        "kind": "profiler",
        "train_type": train_type,
        "gc": gc,
        "profiler_role": declared_role,
    }
    row = {
        "observation_id": observation_id,
        "configuration": {"job": job},
        "outcome": {"class": "success"},
    }
    runtime_material = cohort_material or {"provenance_sha256": "a" * 64}
    record = {
        "source_observation_id": observation_id,
        "job_id": job["job_id"],
        "evidence_tier": "legacy_consistent",
        "runtime": {
            "runtime_cohort_id": sha256_json(runtime_material),
            "runtime_cohort_material": runtime_material,
        },
        "validation_design": {
            "mode": "fixed_partition",
            "policy": PROFILER_POLICY,
            "declared_role": declared_role,
            "original_role": declared_role,
            "effective_role": effective_role,
            "declared_role_source": "authorized_job_payload",
            "effective_role_source": (
                "profiler_calibration_artifact"
                if effective_role == "fallback_calibration"
                else "authorized_job_payload_fallback"
            ),
        },
        "measurement_eligibility": {
            "class": "calibration_candidate",
            "profiler": True,
        },
    }
    return row, record


def packing_row(
    index: int, *, pair_id: str | None, treatment: str
) -> tuple[dict, dict]:
    observation_id = f"packing-observation-{index}"
    job = {
        "job_id": f"packing-job-{index}",
        "kind": "throughput",
        "packing": treatment == "packed",
    }
    runtime_material = {"provenance_sha256": "packing"}
    row = {
        "observation_id": observation_id,
        "configuration": {"job": job},
        "outcome": {"class": "success"},
    }
    record = {
        "source_observation_id": observation_id,
        "job_id": job["job_id"],
        "evidence_tier": "legacy_consistent",
        "runtime": {
            "runtime_cohort_id": sha256_json(runtime_material),
            "runtime_cohort_material": runtime_material,
        },
        "validation_design": {
            "mode": "paired_comparison",
            "policy": PACKING_POLICY,
            "pair_id": pair_id,
            "treatment": treatment,
        },
        "measurement_eligibility": {
            "class": "calibration_candidate",
            "packing_pair": True,
        },
    }
    return row, record


def report(
    records: list[dict],
    *,
    evaluation_points: int = 0,
    artifact_calibration_job_ids: list[str] | None = None,
    evaluation_job_ids: list[str] | None = None,
    resource_context: dict | None = None,
) -> dict:
    fallback_promotions = [
        {
            "job_id": record["job_id"],
            "original_role": "holdout",
            "effective_role": "fallback_calibration",
            "selection_policy": "recorded_fallback_calibration",
        }
        for record in records
        if (record.get("validation_design") or {}).get("effective_role")
        == "fallback_calibration"
    ]
    profiler_context = {
        "remaining_evaluation_points": evaluation_points,
        "evaluation_mape": 0.05,
        "fallback_promotions": fallback_promotions,
    }
    if artifact_calibration_job_ids is not None:
        profiler_context["artifact_calibration_job_ids"] = sorted(
            artifact_calibration_job_ids
        )
        profiler_context["artifact_calibration_job_ids_sha256"] = sha256_json(
            sorted(artifact_calibration_job_ids)
        )
    if evaluation_job_ids is not None:
        profiler_context["evaluation_job_ids"] = sorted(evaluation_job_ids)
        profiler_context["evaluation_job_ids_sha256"] = sha256_json(
            sorted(evaluation_job_ids)
        )
    return {
        "schema": RECOVERY_SCHEMA,
        "report_sha256": "recovery",
        "counts": {},
        "validation_context": {
            "profiler": profiler_context,
            "resource_loocv": resource_context or {},
        },
        "records": records,
    }


class HistoricalReadinessTests(unittest.TestCase):
    def test_loocv_holds_all_configuration_variants_of_scenario_together(self) -> None:
        rows = {}
        records = []
        for scenario_index, model in enumerate(("1b", "4b", "8b")):
            for variant, (mbs, zero) in enumerate(((1, "zero2"), (2, "zero3"))):
                row, record = resource_row(
                    scenario_index * 2 + variant,
                    model=model,
                    mbs=mbs,
                    zero=zero,
                )
                rows[row["observation_id"]] = row
                records.append(record)

        readiness = analyze(report(records), rows)["resource_loocv"]

        self.assertEqual(readiness["folds"], 3)
        self.assertTrue(readiness["every_observation_is_test_exactly_once"])
        self.assertTrue(readiness["ready_for_bounded_fit"])
        folds = readiness["cohorts"][0]["folds"]
        self.assertEqual({fold["test_observations"] for fold in folds}, {2})
        self.assertTrue(all(fold["membership_disjoint"] for fold in folds))
        self.assertFalse(readiness["sampling_roles_are_partitions"])
        self.assertTrue(
            readiness["components"]["throughput_primary"]["ready_for_bounded_fit"]
        )
        self.assertEqual(
            readiness["components"]["throughput_primary"]["tier_counts"],
            {"legacy_consistent": 6},
        )

    def test_loocv_holds_scenario_out_across_runtime_cohorts(self) -> None:
        rows = {}
        records = []
        for cohort_name in ("runtime-a", "runtime-b"):
            cohort_material = {"provenance_sha256": cohort_name}
            for scenario_index, model in enumerate(("1b", "4b", "8b")):
                row, record = resource_row(
                    len(records),
                    model=model,
                    mbs=1,
                    zero="zero2",
                    cohort_material=cohort_material,
                )
                rows[row["observation_id"]] = row
                records.append(record)

        readiness = analyze(report(records), rows)["resource_loocv"]

        self.assertEqual(
            readiness["fold_scope"],
            "global_scenario_across_all_runtime_cohorts",
        )
        self.assertEqual(readiness["folds"], 3)
        self.assertEqual(readiness["cohort_local_folds_diagnostic_only"], 6)
        self.assertTrue(readiness["every_observation_is_test_exactly_once"])
        self.assertEqual(
            {fold["test_observations"] for fold in readiness["global_folds"]},
            {2},
        )
        self.assertEqual(
            {fold["train_observations"] for fold in readiness["global_folds"]},
            {4},
        )
        self.assertTrue(
            all(not fold["unseen_test_runtime_cohort_ids"] for fold in readiness["global_folds"])
        )

    def test_loocv_static_role_or_scenario_tamper_is_rejected(self) -> None:
        row, record = resource_row(1, model="8b", mbs=1, zero="zero2")
        record["validation_design"]["explicit_role"] = "calibration"
        record["validation_design"]["scenario_id"] = "tampered"

        readiness = analyze(report([record]), {row["observation_id"]: row})[
            "resource_loocv"
        ]

        self.assertEqual(readiness["candidate_observations"], 0)
        self.assertEqual(len(readiness["invalid_observations"]), 1)
        self.assertIn(
            "loocv_static_role_is_forbidden",
            readiness["invalid_observations"][0]["issues"],
        )
        self.assertIn(
            "loocv_scenario_id_mismatch",
            readiness["invalid_observations"][0]["issues"],
        )

    def test_resource_components_do_not_promote_screen_to_primary_throughput(
        self,
    ) -> None:
        rows = {}
        records = []
        for index, model in enumerate(("1b", "4b", "8b")):
            row, record = resource_row(
                index,
                model=model,
                mbs=1,
                zero="zero2",
                routes={"feasibility": True, "throughput_screen_only": True},
            )
            rows[row["observation_id"]] = row
            records.append(record)

        readiness = analyze(report(records), rows)["resource_loocv"]

        self.assertTrue(
            readiness["components"]["throughput_screen_only"]["ready_for_bounded_fit"]
        )
        self.assertFalse(
            readiness["components"]["throughput_primary"]["ready_for_bounded_fit"]
        )
        self.assertFalse(readiness["full_resource_bundle_ready"])
        fold = readiness["cohorts"][0]["folds"][0]
        self.assertEqual(fold["train_measurement_routes"]["throughput_primary"], 0)
        self.assertEqual(fold["test_measurement_routes"]["throughput_screen_only"], 1)

    def test_resource_component_requires_three_scenarios(self) -> None:
        rows = {}
        records = []
        for index, model in enumerate(("1b", "8b")):
            row, record = resource_row(index, model=model, mbs=1, zero="zero2")
            rows[row["observation_id"]] = row
            records.append(record)

        component = analyze(report(records), rows)["resource_loocv"]["components"][
            "throughput_primary"
        ]

        self.assertFalse(component["ready_for_bounded_fit"])
        self.assertEqual(component["eligible_runtime_cohort_ids"], [])
        self.assertIn(
            "fewer_than_3_eligible_scenarios",
            component["dropped_cohorts"][0]["blockers"],
        )

    def test_runtime_cohort_id_must_hash_its_material(self) -> None:
        row, record = resource_row(1, model="8b", mbs=1, zero="zero2")
        record["runtime"]["runtime_cohort_id"] = "forged-shared-cohort"

        readiness = analyze(report([record]), {row["observation_id"]: row})[
            "resource_loocv"
        ]

        self.assertEqual(readiness["candidate_observations"], 0)
        self.assertIn(
            "runtime_cohort_id_material_hash_mismatch",
            readiness["invalid_observations"][0]["issues"],
        )

    def test_prior_resource_metrics_are_not_current_without_membership_digest(
        self,
    ) -> None:
        rows = {}
        records = []
        for index, model in enumerate(("1b", "4b", "8b")):
            row, record = resource_row(index, model=model, mbs=1, zero="zero2")
            rows[row["observation_id"]] = row
            records.append(record)

        comparison = analyze(report(records, resource_context={"folds": 3}), rows)[
            "resource_loocv"
        ]["prior_validation_comparison"]

        self.assertEqual(comparison["status"], "historical_reference_unbound")
        self.assertFalse(comparison["membership_matches_current"])
        self.assertFalse(comparison["metrics_usable_for_current_fold_set"])

    def test_profiler_requires_feature_rank_not_only_point_count(self) -> None:
        rows = {}
        records = []
        for index in range(3):
            row, record = profiler_row(
                index,
                train_type="lora",
                gc=True,
                declared_role="calibration",
                effective_role="calibration",
            )
            rows[row["observation_id"]] = row
            records.append(record)
        row, record = profiler_row(
            4,
            train_type="full",
            gc=False,
            declared_role="holdout",
            effective_role="holdout",
        )
        rows[row["observation_id"]] = row
        records.append(record)

        profiler = analyze(report(records, evaluation_points=1), rows)[
            "profiler_fixed_partition"
        ]

        self.assertEqual(profiler["feature_matrix_rank"], 1)
        self.assertIn(
            "profiler_calibration_feature_matrix_rank_deficient",
            profiler["blockers"],
        )
        self.assertFalse(profiler["ready_for_bounded_fit"])

    def test_profiler_fallback_preserves_declared_role_and_reaches_rank_three(
        self,
    ) -> None:
        specifications = [
            ("lora", False, "calibration", "calibration"),
            ("lora", True, "calibration", "calibration"),
            ("full", False, "holdout", "fallback_calibration"),
            ("full", True, "holdout", "holdout"),
        ]
        rows = {}
        records = []
        for index, specification in enumerate(specifications):
            row, record = profiler_row(
                index,
                train_type=specification[0],
                gc=specification[1],
                declared_role=specification[2],
                effective_role=specification[3],
            )
            rows[row["observation_id"]] = row
            records.append(record)

        profiler = analyze(
            report(
                records,
                evaluation_points=1,
                artifact_calibration_job_ids=[
                    "profiler-job-0",
                    "profiler-job-1",
                    "profiler-job-2",
                ],
                evaluation_job_ids=["profiler-job-3"],
            ),
            rows,
        )["profiler_fixed_partition"]

        self.assertEqual(profiler["feature_matrix_rank"], 3)
        self.assertEqual(profiler["declared_roles"], {"calibration": 2, "holdout": 2})
        self.assertEqual(
            profiler["effective_roles"],
            {"calibration": 2, "fallback_calibration": 1, "holdout": 1},
        )
        self.assertTrue(profiler["ready_for_bounded_fit"])
        self.assertTrue(profiler["ready_for_independent_holdout_validation"])
        self.assertTrue(profiler["independent_holdout_validation_completed"])
        self.assertEqual(profiler["prior_evaluation"]["status"], "membership_bound")

    def test_profiler_does_not_trust_artifact_count_without_actual_holdout(
        self,
    ) -> None:
        rows = {}
        records = []
        for index, (train_type, gc) in enumerate(
            (("full", False), ("lora", False), ("lora", True))
        ):
            row, record = profiler_row(
                index,
                train_type=train_type,
                gc=gc,
                declared_role="calibration",
                effective_role="calibration",
            )
            rows[row["observation_id"]] = row
            records.append(record)

        profiler = analyze(
            report(
                records,
                evaluation_points=1,
                artifact_calibration_job_ids=[record["job_id"] for record in records],
                evaluation_job_ids=["nonexistent-holdout"],
            ),
            rows,
        )["profiler_fixed_partition"]

        self.assertTrue(profiler["ready_for_bounded_fit"])
        self.assertFalse(profiler["ready_for_independent_holdout_validation"])
        self.assertFalse(profiler["independent_holdout_validation_completed"])
        self.assertEqual(profiler["actual_remaining_holdout_points"], 0)

    def test_profiler_fallback_must_be_source_bound(self) -> None:
        specifications = [
            ("lora", False, "calibration", "calibration"),
            ("lora", True, "calibration", "calibration"),
            ("full", False, "holdout", "fallback_calibration"),
            ("full", True, "holdout", "holdout"),
        ]
        rows = {}
        records = []
        for index, specification in enumerate(specifications):
            row, record = profiler_row(
                index,
                train_type=specification[0],
                gc=specification[1],
                declared_role=specification[2],
                effective_role=specification[3],
            )
            rows[row["observation_id"]] = row
            records.append(record)
        recovery = report(records)
        recovery["validation_context"]["profiler"]["fallback_promotions"] = []

        profiler = analyze(recovery, rows)["profiler_fixed_partition"]

        self.assertFalse(profiler["ready_for_bounded_fit"])
        self.assertIn("invalid_profiler_partition_records", profiler["blockers"])
        self.assertIn(
            "profiler_fallback_promotion_not_source_bound",
            profiler["invalid_records"][0]["issues"],
        )

    def test_profiler_rejects_forged_role_sources(self) -> None:
        row, record = profiler_row(
            1,
            train_type="full",
            gc=False,
            declared_role="holdout",
            effective_role="fallback_calibration",
        )
        record["validation_design"]["declared_role_source"] = "self_reported"
        record["validation_design"]["effective_role_source"] = (
            "authorized_job_payload_fallback"
        )

        profiler = analyze(report([record]), {row["observation_id"]: row})[
            "profiler_fixed_partition"
        ]

        issues = profiler["invalid_records"][0]["issues"]
        self.assertIn("profiler_declared_role_source_invalid", issues)
        self.assertIn("profiler_fallback_effective_role_source_invalid", issues)
        self.assertFalse(profiler["independent_holdout_validation_completed"])

    def test_profiler_does_not_mix_runtime_cohorts_to_reach_rank(self) -> None:
        rows = {}
        records = []
        for index, (train_type, gc) in enumerate(
            (("full", False), ("lora", False), ("lora", True))
        ):
            row, record = profiler_row(
                index,
                train_type=train_type,
                gc=gc,
                declared_role="calibration",
                effective_role="calibration",
                cohort_material={"runtime": index},
            )
            rows[row["observation_id"]] = row
            records.append(record)

        profiler = analyze(report(records), rows)["profiler_fixed_partition"]

        self.assertEqual(profiler["feature_matrix_rank"], 3)
        self.assertFalse(profiler["ready_for_bounded_fit"])
        self.assertEqual(profiler["eligible_fit_runtime_cohort_ids"], [])

    def test_packing_fit_uses_only_complete_pairs(self) -> None:
        rows = {}
        records = []
        specifications = [
            ("pair-good", "unpacked"),
            ("pair-good", "packed"),
            ("pair-bad", "packed"),
            ("pair-bad", "packed"),
            (None, "unpacked"),
        ]
        for index, (pair_id, treatment) in enumerate(specifications):
            row, record = packing_row(index, pair_id=pair_id, treatment=treatment)
            rows[row["observation_id"]] = row
            records.append(record)

        packing = analyze(report(records), rows)["packing_paired_evidence"]

        self.assertEqual(packing["candidate_observation_rows"], 5)
        self.assertEqual(packing["complete_pairs"], 1)
        self.assertEqual(packing["fit_observation_rows"], 2)
        self.assertEqual(packing["complete_pair_ids"], ["pair-good"])
        self.assertEqual(packing["incomplete_pair_ids"], ["pair-bad"])
        self.assertEqual(len(packing["invalid_records"]), 1)
        self.assertTrue(packing["ready_for_low_confidence_effect_fit"])
        self.assertFalse(packing["ready_for_automatic_packing_enablement"])

    def test_any_fit_and_full_bundle_are_separate(self) -> None:
        rows = {}
        records = []
        for index, model in enumerate(("1b", "4b", "8b")):
            row, record = resource_row(index, model=model, mbs=1, zero="zero2")
            rows[row["observation_id"]] = row
            records.append(record)

        readiness = analyze(report(records), rows)

        self.assertTrue(readiness["any_fit_ready"])
        self.assertFalse(readiness["full_bundle_ready"])
        self.assertTrue(readiness["requirements"]["uncertainty_inflation_required"])
        self.assertFalse(readiness["calibration_publishable"])


if __name__ == "__main__":
    unittest.main()
