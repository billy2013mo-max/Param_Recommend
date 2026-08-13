from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import h800_theory_calibration as calibration  # noqa: E402


def physical_priors() -> dict:
    return {
        "policy": "unit-test-explicit",
        "values": {
            "dense_peak_flops_per_s": 1_000_000_000_000.0,
            "memory_bandwidth_bytes_per_s": 1_000_000_000_000.0,
            "collective_bandwidth_bytes_per_s": 1_000_000_000_000.0,
            "compute_efficiency": None,
            "hbm_efficiency": 0.5,
            "optimizer_hbm_efficiency": 0.5,
            "collective_efficiency": 0.5,
            "collective_latency_seconds": 0.0,
            "microstep_latency_seconds": None,
            "framework_latency_seconds": None,
            "communication_overlap_by_stage": {"0": 0.0, "2": 0.0, "3": 0.0},
            "memory_capacity_bytes": 1_000.0,
        },
    }


def basis_record(
    observation_id: str,
    *,
    model_id: str,
    mbs: int = 1,
    gpu_count: int = 1,
    zero_stage: int = 0,
    cohort: str = "runtime-a",
    route: dict[str, bool] | None = None,
    outcome: str = "success",
    measured_step_seconds: float = 4.0,
    allocated: float | None = 200.0,
    reserved: float | None = 210.0,
    oom_lower: float | None = None,
) -> dict:
    observed_memory = {
        "kind": "exact_success_peak" if outcome == "success" else "right_censored_oom",
        "peak_allocated_diagnostic_bytes": allocated,
        "peak_reserved_target_bytes": reserved,
        "right_censor_lower_bytes": oom_lower,
    }
    return {
        "observation_id": observation_id,
        "job_id": observation_id,
        "route": route
        or {
            "throughput_primary": True,
            "memory_boundary": True,
            "feasibility": True,
        },
        "scenario": {
            "model_id": model_id,
            "dataset_id": "dataset",
            "target_gbs": 64,
            "train_type": "full",
            "gpu_count": gpu_count,
            "physical_mbs": mbs,
            "cutoff_len": 512,
        },
        "selector": {
            "gpu_family": "h800_140g",
            "runtime_cohort_id": cohort,
            "dtype": "bf16",
            "kernel_path": "flash_attention",
            "training_mode": "full",
            "zero_stage": zero_stage,
            "gradient_checkpointing": False,
            "packing": False,
        },
        "outcome": outcome,
        "performance": {
            "physical_mbs": mbs,
            "observed": {"mean_step_seconds": measured_step_seconds}
            if outcome == "success"
            else {},
            "work_per_step": {
                "computed_tokens": 100.0,
                "effective_tokens": 100.0,
                "logical_samples": 10.0,
            },
            "flops_per_step": {"linear": 1_000_000_000_000.0, "total": 1_000_000_000_000.0},
            "traffic_bytes_per_rank_step": {"kernel_total": 0.0, "optimizer": 0.0},
            "communication": {
                "payload_bytes_per_rank_step": 0.0,
                "collective_count": 0,
            },
        },
        "memory": {
            "analytic_non_activation_bytes": 100.0,
            "analytic_reference_bytes": 200.0,
            "structural_activation_bytes": 100.0,
            "safe_limit_bytes": 950.0,
            "observed": observed_memory,
        },
    }


def basis_report(records: list[dict], *, with_priors: bool = True) -> dict:
    report = {
        "schema": "sft_h800_theory_basis/v1",
        "status": "bootstrap_theory_only",
        "physical_priors": physical_priors() if with_priors else {"values": {}},
        "publication_blockers": ["fixture_is_not_publishable"],
        "records": records,
    }
    # A correct self-consistent digest so the calibrator's independent basis
    # integrity check passes for the fixture (tampering tests recompute it).
    report["report_sha256"] = calibration._canonical_sha256(report)  # noqa: SLF001
    return report


class H800TheoryCalibrationTests(unittest.TestCase):
    def test_global_fold_holds_scenario_across_cohorts_and_routes(self) -> None:
        records = [
            basis_record("a-primary", model_id="model-a", cohort="runtime-a"),
            basis_record(
                "a-screen",
                model_id="model-a",
                cohort="runtime-b",
                route={"throughput_screen_only": True},
            ),
            basis_record("b-primary", model_id="model-b", cohort="runtime-a"),
        ]
        folds = calibration.build_global_scenario_folds(records)
        a_id = calibration.scenario_id(records[0])
        a_fold = next(fold for fold in folds if fold["held_out_scenario_id"] == a_id)
        self.assertEqual(a_fold["test_observation_ids"], ["a-primary", "a-screen"])
        self.assertNotIn("a-primary", a_fold["train_observation_ids"])
        self.assertNotIn("a-screen", a_fold["train_observation_ids"])
        self.assertEqual(a_fold["test_runtime_cohorts"], ["runtime-a", "runtime-b"])

    def test_finite_conformal_does_not_substitute_max_below_nineteen(self) -> None:
        insufficient = calibration.one_sided_conformal(range(18))
        sufficient = calibration.one_sided_conformal(range(19))
        self.assertFalse(insufficient["available"])
        self.assertIsNone(insufficient["quantile"])
        self.assertTrue(sufficient["available"])
        self.assertEqual(sufficient["quantile"], 18)

    def test_inner_folds_are_loso_or_a_scenario_partition(self) -> None:
        scenarios = [f"s{index}" for index in range(7)]
        loso = calibration._assign_inner_folds(scenarios, None)  # noqa: SLF001
        self.assertEqual(len(loso), 7)
        self.assertTrue(all(len(group) == 1 for group in loso))
        kfold = calibration._assign_inner_folds(scenarios, 3)  # noqa: SLF001
        self.assertEqual(len(kfold), 3)
        union: set[str] = set()
        for group in kfold:
            self.assertFalse(union.intersection(group))
            union |= set(group)
        self.assertEqual(union, set(scenarios))
        # A bound at or above the scenario count collapses to leave-one-out.
        self.assertEqual(
            calibration._assign_inner_folds(scenarios, 99),  # noqa: SLF001
            loso,
        )

    def test_inner_scenario_oof_never_trains_on_its_own_or_outer_scenario(self) -> None:
        records = [
            basis_record(f"m{index}-r0", model_id=f"model-{index}")
            for index in range(6)
        ]
        # Two rows per scenario so a residual row and a fit are distinguishable.
        records += [
            basis_record(f"m{index}-r1", model_id=f"model-{index}")
            for index in range(6)
        ]
        outer = calibration.scenario_id(records[0])

        def fit_fn(subset: list[dict]) -> frozenset[str]:
            return frozenset(calibration.scenario_id(record) for record in subset)

        def collect_fn(model: frozenset[str], test: list[dict]) -> list[tuple[str, frozenset[str]]]:
            return [(calibration.scenario_id(record), model) for record in test]

        cache: dict = {}
        collected = calibration._inner_scenario_oof(  # noqa: SLF001
            records,
            frozenset({outer}),
            fit_fn=fit_fn,
            collect_fn=collect_fn,
            fit_cache=cache,
            folds=3,
        )
        self.assertTrue(collected)
        for scenario, trained_on in collected:
            self.assertNotEqual(scenario, outer)
            self.assertNotIn(scenario, trained_on)  # own scenario never in its center
            self.assertNotIn(outer, trained_on)  # outer test scenario never leaks in
        # Determinism: identical inputs reproduce identical residual order.
        again = calibration._inner_scenario_oof(  # noqa: SLF001
            records,
            frozenset({outer}),
            fit_fn=fit_fn,
            collect_fn=collect_fn,
            fit_cache={},
            folds=3,
        )
        self.assertEqual(collected, again)

    def test_joint_ranking_gates_on_memory_and_counts_safety_failures(self) -> None:
        candidates = [
            {  # admitted, safe, but slower -- this is the correct pick
                "candidate": {"gpu_count": 1},
                "predicted_memory_admit": True,
                "actually_safe": True,
                "actually_oom": False,
                "observed_throughput": 100.0,
                "predicted_throughput_lower": 90.0,
                "observed_reserved_bytes": 100.0,
                "safe_limit_bytes": 200.0,
            },
            {  # fastest, but predicted over the line -> must never be selected
                "candidate": {"gpu_count": 1},
                "predicted_memory_admit": False,
                "actually_safe": True,
                "actually_oom": False,
                "observed_throughput": 200.0,
                "predicted_throughput_lower": 180.0,
                "observed_reserved_bytes": 150.0,
                "safe_limit_bytes": 200.0,
            },
            {  # predicted safe but actually OOMs -> safety failure, not regret
                "candidate": {"gpu_count": 1},
                "predicted_memory_admit": True,
                "actually_safe": False,
                "actually_oom": True,
                "observed_throughput": None,
                "predicted_throughput_lower": None,
                "observed_reserved_bytes": None,
                "safe_limit_bytes": 200.0,
            },
            {  # a card count where a real safe option exists but none is admitted
                "candidate": {"gpu_count": 2},
                "predicted_memory_admit": False,
                "actually_safe": True,
                "actually_oom": False,
                "observed_throughput": 300.0,
                "predicted_throughput_lower": 280.0,
                "observed_reserved_bytes": 190.0,
                "safe_limit_bytes": 200.0,
            },
        ]
        metrics = calibration._joint_ranking_metrics(candidates)  # noqa: SLF001
        groups = {group["gpu_count"]: group for group in metrics["gpu_groups"]}
        # Oracle is the best measured safe throughput (200); only the admitted
        # candidate (100) is selectable, so regret is 0.5 -- not the 0 a
        # throughput-only ranker reports by picking the inadmissible fast one.
        self.assertEqual(groups[1]["oracle_safe_best_throughput"], 200.0)
        self.assertEqual(groups[1]["top1_regret"], 0.5)
        self.assertEqual(groups[1]["memory_gated_safety_failures"], 1)
        # A real safe option with nothing admissible is full regret, never 0.
        self.assertEqual(groups[2]["top1_regret"], 1.0)
        self.assertEqual(metrics["memory_gated_safety_failures"], 1)

    def test_cohort_evidence_inflation_nonnegative_and_missing_anchor_blocks(self) -> None:
        # Single cohort: LOCO not identifiable; legacy tier -> anchor-missing blocker.
        single = [basis_record(f"s{index}", model_id=f"m{index}", cohort="A") for index in range(19)]
        for record in single:
            record["evidence_tier"] = "legacy_consistent"
        out = calibration._cohort_evidence_inflation(  # noqa: SLF001
            single, alpha=0.05, base_residual_p95_bytes=100.0
        )
        self.assertFalse(out["leave_one_cohort_out_identifiable"])
        self.assertEqual(out["inflation_bytes"], 0.0)
        self.assertFalse(out["verified_publication_anchor_present"])
        self.assertIn("verified_calibration_anchor_missing", out["blockers"])
        self.assertIn(
            "leave_one_cohort_out_not_identifiable_single_cohort", out["blockers"]
        )
        # Two cohorts + a non-legacy tier: identifiable, non-negative, anchor present.
        pair = [
            basis_record(f"a{index}", model_id=f"m{index}", cohort="A")
            for index in range(19)
        ] + [
            basis_record(f"b{index}", model_id=f"n{index}", cohort="B")
            for index in range(19)
        ]
        for record in pair:
            record["evidence_tier"] = "native_v2_verified"
        out2 = calibration._cohort_evidence_inflation(  # noqa: SLF001
            pair, alpha=0.05, base_residual_p95_bytes=0.0
        )
        self.assertTrue(out2["leave_one_cohort_out_identifiable"])
        self.assertGreaterEqual(out2["inflation_bytes"], 0.0)
        self.assertTrue(out2["verified_publication_anchor_present"])
        self.assertNotIn("verified_calibration_anchor_missing", out2["blockers"])

    def test_memory_oom_guard_uses_safe_limit_floor_and_reports_statistical_km(self) -> None:
        successes = [
            basis_record(f"s-{index}", model_id=f"model-{index}", zero_stage=2)
            for index in range(19)
        ]
        oom = basis_record(
            "oom-zero2",
            model_id="oom-model",
            zero_stage=2,
            outcome="oom",
            allocated=999_999.0,
            reserved=999_999.0,
            oom_lower=900.0,  # below safe_limit + 1, so the safe-limit floor must bind
            route={"memory_boundary": True, "feasibility": True},
        )
        train = [*successes, oom]
        center = calibration._fit_memory_center(train)  # noqa: SLF001
        tail = calibration._fit_memory_tail(train, center, alpha=0.05)  # noqa: SLF001
        prediction = calibration._predict_memory(successes[0], center, tail)  # noqa: SLF001
        # safe_limit is 950 in the fixture; the guard must push p95 strictly above it.
        self.assertGreater(prediction["p95_reserved_bytes"], 950.0)
        self.assertIn("statistical_p95_identifiable", prediction)
        km = tail["statistical_residual_p95_km"]["pooled"]
        self.assertEqual(km["events"], 19)
        self.assertEqual(km["censored"], 1)

    def test_compute_efficiency_projection_is_bounded_monotone_and_saturating(self) -> None:
        fitted = calibration.bounded_isotonic_by_mbs(
            {1: 0.6, 2: 0.3, 4: 1.4}, lower=0.1, upper=1.0
        )
        self.assertGreaterEqual(fitted[2], fitted[1])
        self.assertGreaterEqual(fitted[4], fitted[2])
        self.assertLessEqual(fitted[4], 1.0)
        model = {"compute_efficiency_by_mbs": {str(key): value for key, value in fitted.items()}}
        self.assertEqual(
            calibration._efficiency_for_mbs(model, 64),  # noqa: SLF001
            fitted[4],
        )

    def test_report_uses_primary_for_fit_and_screen_only_for_diagnostics(self) -> None:
        records: list[dict] = []
        for scenario in range(20):
            model = f"model-{scenario}"
            records.extend(
                [
                    basis_record(
                        f"{model}-mbs1", model_id=model, mbs=1, measured_step_seconds=4.0
                    ),
                    basis_record(
                        f"{model}-mbs2", model_id=model, mbs=2, measured_step_seconds=2.0
                    ),
                    basis_record(
                        f"{model}-screen",
                        model_id=model,
                        mbs=1,
                        measured_step_seconds=100.0,
                        route={"throughput_screen_only": True},
                    ),
                ]
            )
        report = calibration.calibrate_h800_theory(basis_report(records))
        self.assertEqual(report["status"], "theory_only")
        self.assertEqual(report["confidence"], "bootstrap")
        self.assertEqual(report["publication"], "nonpublishable")
        self.assertFalse(report["publishable"])
        self.assertFalse(report["production_profile_generated"])
        self.assertEqual(
            report["full_historical_bootstrap_fit"]["throughput"]["center"]["fit_rows"],
            40,
        )
        self.assertEqual(report["input"]["route_counts"]["throughput_screen_only"], 20)
        self.assertEqual(calibration.validate_report(report), [])

    def test_oom_constraint_is_exact_selector_only_and_never_center_label(self) -> None:
        successes = [
            basis_record(f"success-{index}", model_id=f"model-{index}", zero_stage=2)
            for index in range(19)
        ]
        oom = basis_record(
            "oom-zero2",
            model_id="oom-model",
            zero_stage=2,
            outcome="oom",
            allocated=999_999.0,  # must not be treated as an exact center label
            reserved=999_999.0,
            oom_lower=900.0,
            route={"memory_boundary": True, "feasibility": True},
        )
        train = [*successes, oom]
        center = calibration._fit_memory_center(train)  # noqa: SLF001
        tail = calibration._fit_memory_tail(train, center, alpha=0.05)  # noqa: SLF001
        zero2 = calibration._predict_memory(successes[0], center, tail)  # noqa: SLF001
        zero3_record = copy.deepcopy(successes[0])
        zero3_record["observation_id"] = "unseen-zero3"
        zero3_record["selector"]["zero_stage"] = 3
        zero3 = calibration._predict_memory(zero3_record, center, tail)  # noqa: SLF001
        self.assertTrue(zero2["available"])
        self.assertTrue(zero3["available"])
        self.assertGreaterEqual(zero2["p95_reserved_bytes"], 900.0)
        self.assertLess(zero3["p95_reserved_bytes"], zero2["p95_reserved_bytes"])
        self.assertEqual(tail["oom_hierarchy"], "exact_selector_only_never_pooled")
        liveness = next(iter(center["activation_liveness_by_mode_gc"].values()))
        self.assertLessEqual(liveness, 4.0)

    def test_missing_physical_priors_blocks_throughput_without_publishing(self) -> None:
        records = [
            basis_record("a", model_id="model-a"),
            basis_record("b", model_id="model-b"),
        ]
        report = calibration.calibrate_h800_theory(
            basis_report(records, with_priors=False)
        )
        full = report["full_historical_bootstrap_fit"]["throughput"]["center"]
        self.assertFalse(full["available"])
        self.assertIn("dense_peak_flops_per_s_prior_missing", full["blockers"])
        self.assertFalse(report["publishable"])
        self.assertFalse(report["production_profile_generated"])

    def test_basis_integrity_recomputes_digest_and_rejects_tamper(self) -> None:
        records = [
            basis_record(f"row-{index}", model_id=f"model-{index}")
            for index in range(20)
        ]
        report = basis_report(records)
        good, blockers = calibration._validate_basis_integrity(report)  # noqa: SLF001
        self.assertTrue(good["schema_ok"])
        self.assertTrue(good["digest_ok"])
        self.assertEqual(blockers, [])
        # Tampering a record without re-signing must be caught independently.
        tampered = copy.deepcopy(report)
        tampered["records"][0]["outcome"] = "oom"
        bad, bad_blockers = calibration._validate_basis_integrity(tampered)  # noqa: SLF001
        self.assertFalse(bad["digest_ok"])
        self.assertIn("basis_report_sha256_mismatch", bad_blockers)
        # A calibration built over the tampered basis carries the blocker and
        # still validates as a fail-closed, blocked report.
        built = calibration.calibrate_h800_theory(tampered)
        self.assertIn("basis_report_sha256_mismatch", built["blockers"])
        self.assertEqual(calibration.validate_report(built), [])

    def test_report_integrity_binds_hashes_and_rejects_non_finite(self) -> None:
        records = [
            basis_record(f"row-{index}", model_id=f"model-{index}")
            for index in range(20)
        ]
        report = calibration.build_report(basis_report(records))
        integrity = report["integrity"]
        self.assertTrue(integrity["implementation_sha256"])
        self.assertTrue(integrity["fold_membership_sha256"])
        self.assertTrue(integrity["basis"]["validated_before_fit"])
        self.assertFalse(
            calibration._contains_non_finite({"a": [1.0, 2, True]})  # noqa: SLF001
        )
        self.assertTrue(
            calibration._contains_non_finite({"a": [1.0, float("inf")]})  # noqa: SLF001
        )
        # An injected non-finite value is rejected by validate_report.
        poisoned = copy.deepcopy(report)
        poisoned["aggregate_validation"]["memory_feasibility"][
            "success_p95_coverage"
        ] = float("nan")
        self.assertIn(
            "report_contains_non_finite_values", calibration.validate_report(poisoned)
        )

    def test_validation_detects_mutation(self) -> None:
        records = [
            basis_record(f"row-{index}", model_id=f"model-{index}")
            for index in range(20)
        ]
        report = calibration.build_report(basis_report(records))
        self.assertEqual(calibration.validate_report(report), [])
        report["publishable"] = True
        issues = calibration.validate_report(report)
        self.assertIn("report_is_not_fail_closed_nonpublishable", issues)
        self.assertIn("report_sha256_mismatch", issues)

    def test_real_artifact_resource_scope_excludes_special_purpose_routes(self) -> None:
        path = ROOT / "artifacts" / "h800_theory_basis.json"
        real_basis = json.loads(path.read_text(encoding="utf-8"))
        report = calibration.calibrate_h800_theory(real_basis)
        self.assertEqual(report["input"]["basis_records"], 1111)
        self.assertEqual(report["input"]["records"], 1062)
        self.assertEqual(report["input"]["global_scenarios"], 84)
        special = report["input"]["special_purpose_excluded"]
        self.assertEqual(special["records"], 49)
        self.assertEqual(
            special["route_counts"],
            {
                "packing_memory_safety": 10,
                "packing_pair": 20,
                "profiler": 19,
            },
        )
        self.assertTrue(special["all_were_also_marked_feasibility"])
        memory = report["aggregate_validation"]["memory_feasibility"]
        self.assertEqual(memory["success_rows"], 956)
        self.assertEqual(memory["oom_rows"], 106)
        self.assertNotIn(
            "conformal_tails_use_same_outer_training_fold_residuals_not_inner_scenario_oof",
            report["blockers"],
        )
        self.assertTrue(
            report["identifiability"]["tail_residual_is_inner_scenario_oof"]
        )
        first_fold = report["global_scenario_loocv"]["folds"][0]
        self.assertEqual(
            first_fold["throughput"]["step_time_upper_conformal"]["residual_source"],
            "inner_scenario_kfold_oof",
        )
        self.assertEqual(
            first_fold["memory"]["tail_fit"]["residual_source"],
            "inner_scenario_kfold_oof",
        )
        self.assertEqual(calibration.validate_report(report), [])


if __name__ == "__main__":
    unittest.main()
