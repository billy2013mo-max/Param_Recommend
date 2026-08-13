"""Tests for the scenario candidate generator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import candidate_generator as cg  # noqa: E402

CAPACITY = 150_142_189_568
QWEN3_8B = 8_190_735_360
QWEN3_14B = 14_768_307_200


def _scenario(**overrides):
    scenario = {
        "model_id": "qwen3_8b",
        "training_mode": "lora",
        "dataset_id": "short_512",
        "dataset_category": "short",
        "target_gbs": 64,
        "cutoff_len": 512,
        "actual_parameters": QWEN3_8B,
        "packing": False,
    }
    scenario.update(overrides)
    return scenario


class TestGbsContract(unittest.TestCase):
    def test_every_unpacked_candidate_reproduces_the_target_gbs(self) -> None:
        out = cg.generate_candidates(_scenario(), capacity_bytes=CAPACITY)
        self.assertTrue(out["candidates"])
        for candidate in out["candidates"]:
            product = (
                candidate["gpu_count"]
                * candidate["physical_mbs"]
                * candidate["gradient_accumulation_steps"]
            )
            self.assertEqual(product, candidate["target_gbs"])

    def test_indivisible_combinations_are_rejected_with_a_reason(self) -> None:
        out = cg.generate_candidates(
            _scenario(target_gbs=32), capacity_bytes=CAPACITY
        )
        reasons = {row["reason"] for row in out["statically_rejected"]}
        self.assertIn("gbs_not_divisible_by_gpu_count_times_mbs", reasons)
        for candidate in out["candidates"]:
            self.assertEqual(
                candidate["target_gbs"]
                % (candidate["gpu_count"] * candidate["physical_mbs"]),
                0,
            )

    def test_gradient_accumulation_is_derived_not_searched(self) -> None:
        out = cg.generate_candidates(_scenario(), capacity_bytes=CAPACITY)
        self.assertNotIn(
            "gradient_accumulation_steps", out["generation_policy"]["searched_fields"]
        )
        for candidate in out["candidates"]:
            self.assertGreaterEqual(candidate["gradient_accumulation_steps"], 1)


class TestGeometryRules(unittest.TestCase):
    def test_single_gpu_uses_stage_zero_only(self) -> None:
        out = cg.generate_candidates(_scenario(), capacity_bytes=CAPACITY)
        stages = {
            candidate["zero_stage"]
            for candidate in out["candidates"]
            if candidate["gpu_count"] == 1
        }
        self.assertEqual(stages, {0})

    def test_multi_gpu_never_uses_stage_zero(self) -> None:
        out = cg.generate_candidates(_scenario(), capacity_bytes=CAPACITY)
        for candidate in out["candidates"]:
            if candidate["gpu_count"] > 1:
                self.assertIn(candidate["zero_stage"], {1, 2, 3})

    def test_zero1_is_opt_in(self) -> None:
        without = cg.generate_candidates(_scenario(), capacity_bytes=CAPACITY)
        self.assertNotIn(
            1, {candidate["zero_stage"] for candidate in without["candidates"]}
        )
        with_zero1 = cg.generate_candidates(
            _scenario(), capacity_bytes=CAPACITY, include_zero1=True
        )
        self.assertIn(
            1, {candidate["zero_stage"] for candidate in with_zero1["candidates"]}
        )

    def test_long_cutoff_shortens_the_mbs_ladder(self) -> None:
        short = cg.generate_candidates(
            _scenario(cutoff_len=512), capacity_bytes=CAPACITY
        )
        long = cg.generate_candidates(
            _scenario(cutoff_len=32768, target_gbs=64), capacity_bytes=CAPACITY
        )
        short_mbs = {c["physical_mbs"] for c in short["candidates"]}
        long_mbs = {c["physical_mbs"] for c in long["candidates"]}
        self.assertGreater(max(short_mbs), max(long_mbs))

    def test_mbs_stays_inside_the_calibrated_grid_by_default(self) -> None:
        out = cg.generate_candidates(_scenario(), capacity_bytes=CAPACITY)
        for candidate in out["candidates"]:
            self.assertIn(candidate["physical_mbs"], cg.SUPPORTED_MBS)
            self.assertTrue(candidate["inside_predictor_supported_grid"])
        reasons = {row["reason"] for row in out["statically_rejected"]}
        self.assertIn("mbs_outside_calibrated_support_grid", reasons)

    def test_uncalibrated_mbs_can_be_requested_for_experiment_design(self) -> None:
        out = cg.generate_candidates(
            _scenario(), capacity_bytes=CAPACITY, restrict_to_supported_mbs=False
        )
        self.assertIn(32, {c["physical_mbs"] for c in out["candidates"]})
        outside = [
            c for c in out["candidates"] if not c["inside_predictor_supported_grid"]
        ]
        self.assertTrue(outside)


class TestStaticPruning(unittest.TestCase):
    def test_full_finetune_of_14b_is_pruned_on_one_gpu(self) -> None:
        out = cg.generate_candidates(
            _scenario(
                model_id="qwen3_14b",
                training_mode="full",
                dataset_id="multiturn_4096",
                dataset_category="multiturn",
                cutoff_len=4096,
                actual_parameters=QWEN3_14B,
            ),
            capacity_bytes=CAPACITY,
        )
        self.assertNotIn(1, {c["gpu_count"] for c in out["candidates"]})
        pruned = [
            row
            for row in out["statically_rejected"]
            if row["reason"] == "analytic_model_state_lower_bound_exceeds_capacity"
        ]
        self.assertTrue(pruned)

    def test_pruning_only_removes_arithmetically_impossible_candidates(self) -> None:
        out = cg.generate_candidates(_scenario(), capacity_bytes=CAPACITY)
        self.assertTrue(out["guarantees"]["pruning_is_static_arithmetic_only"])
        for candidate in out["candidates"]:
            self.assertLess(
                candidate["static_model_state_lower_bound_bytes"], CAPACITY
            )

    def test_zero2_does_not_shard_weights_so_extra_cards_barely_help_lora(
        self,
    ) -> None:
        # This is the mechanism behind the observed success(1 GPU) -> OOM(2 GPU)
        # direction violation: under ZeRO-2 the base weights stay replicated.
        two = cg.optimizer_state_lower_bound_bytes(
            total_parameters=QWEN3_14B,
            train_type="lora",
            gpu_count=2,
            zero_stage=2,
            lora_fraction=0.01,
        )
        four = cg.optimizer_state_lower_bound_bytes(
            total_parameters=QWEN3_14B,
            train_type="lora",
            gpu_count=4,
            zero_stage=2,
            lora_fraction=0.01,
        )
        weights = QWEN3_14B * cg.BYTES_PER_BF16
        self.assertGreater(two, weights)
        self.assertLess((two - four) / weights, 0.02)

    def test_zero3_shards_weights_so_extra_cards_help_materially(self) -> None:
        two = cg.optimizer_state_lower_bound_bytes(
            total_parameters=QWEN3_14B,
            train_type="full",
            gpu_count=2,
            zero_stage=3,
            lora_fraction=0.01,
        )
        four = cg.optimizer_state_lower_bound_bytes(
            total_parameters=QWEN3_14B,
            train_type="full",
            gpu_count=4,
            zero_stage=3,
            lora_fraction=0.01,
        )
        self.assertAlmostEqual(two / four, 2.0, places=3)


class TestPacking(unittest.TestCase):
    def test_packing_fixes_physical_mbs_to_one(self) -> None:
        out = cg.generate_candidates(
            _scenario(packing=True), capacity_bytes=CAPACITY
        )
        self.assertTrue(out["candidates"])
        self.assertEqual({c["physical_mbs"] for c in out["candidates"]}, {1})
        self.assertTrue(
            out["generation_policy"]["packed_physical_mbs_fixed_to_one"]
        )

    def test_packed_candidates_do_not_derive_unpacked_ga(self) -> None:
        out = cg.generate_candidates(
            _scenario(packing=True), capacity_bytes=CAPACITY
        )
        for candidate in out["candidates"]:
            self.assertNotIn("gradient_accumulation_steps", candidate)

    def test_generator_never_enables_packing_on_its_own(self) -> None:
        out = cg.generate_candidates(_scenario(), capacity_bytes=CAPACITY)
        self.assertFalse(out["guarantees"]["enables_packing"])
        self.assertEqual({c["packing"] for c in out["candidates"]}, {False})


class TestContract(unittest.TestCase):
    def test_generator_makes_no_safety_or_ranking_claim(self) -> None:
        out = cg.generate_candidates(_scenario(), capacity_bytes=CAPACITY)
        guarantees = out["guarantees"]
        self.assertFalse(guarantees["predicts_memory"])
        self.assertFalse(guarantees["admits_candidates"])
        self.assertFalse(guarantees["ranks_candidates"])
        self.assertFalse(guarantees["selects_gpu_count"])
        self.assertFalse(guarantees["creates_gpu_queue"])

    def test_scenario_fields_are_fixed_not_searched(self) -> None:
        out = cg.generate_candidates(_scenario(), capacity_bytes=CAPACITY)
        fixed = set(out["generation_policy"]["fixed_fields"])
        searched = set(out["generation_policy"]["searched_fields"])
        self.assertFalse(fixed & searched)
        for field in ("target_gbs", "cutoff_len", "packing", "model_id"):
            self.assertIn(field, fixed)
        for candidate in out["candidates"]:
            self.assertEqual(candidate["target_gbs"], 64)
            self.assertEqual(candidate["cutoff_len"], 512)

    def test_request_ids_are_unique_and_group_is_shared(self) -> None:
        out = cg.generate_candidates(_scenario(), capacity_bytes=CAPACITY)
        ids = [candidate["request_id"] for candidate in out["candidates"]]
        self.assertEqual(len(ids), len(set(ids)))
        groups = {candidate["comparison_group"] for candidate in out["candidates"]}
        self.assertEqual(len(groups), 1)

    def test_reports_gpu_counts_meeting_the_two_candidate_bar(self) -> None:
        out = cg.generate_candidates(_scenario(), capacity_bytes=CAPACITY)
        self.assertEqual(
            out["gpu_counts_with_at_least_two_candidates"], [1, 2, 4]
        )

    def test_invalid_training_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cg.generate_candidates(
                _scenario(training_mode="dpo"), capacity_bytes=CAPACITY
            )

    def test_missing_parameter_count_is_rejected(self) -> None:
        scenario = _scenario()
        del scenario["actual_parameters"]
        with self.assertRaises(ValueError):
            cg.generate_candidates(scenario, capacity_bytes=CAPACITY)

    def test_report_is_checksummed(self) -> None:
        report = cg.build_report([_scenario()], capacity_bytes=CAPACITY)
        self.assertEqual(report["schema"], cg.SCHEMA)
        self.assertIn("report_sha256", report)
        self.assertEqual(report["group_count"], 1)


class TestPredictorHandoff(unittest.TestCase):
    def test_generated_candidates_are_accepted_by_the_active_predictor(self) -> None:
        try:
            from h800_resource_predictor import H800ResourcePredictor
        except Exception:  # pragma: no cover - artifacts may be absent
            self.skipTest("active predictor unavailable")
        out = cg.generate_candidates(_scenario(), capacity_bytes=CAPACITY)
        report = H800ResourcePredictor().predict(out["candidates"])
        group = report["ranking_groups"][0]
        # The handoff is only useful if nothing is thrown out as out-of-domain.
        self.assertEqual(group["rejected_by_support_domain"], 0)
        self.assertEqual(group["requested_candidates"], out["candidate_count"])
        self.assertIsNotNone(group["selected_request_id"])
        self.assertFalse(group["automatic_execution_allowed"])


if __name__ == "__main__":
    unittest.main()
