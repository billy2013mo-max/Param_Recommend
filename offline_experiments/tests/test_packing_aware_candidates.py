"""Tests for the packing-aware candidate space."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import packing_aware_candidates as pac  # noqa: E402
from common import ARTIFACT_DIR  # noqa: E402

CAPACITY = 150_142_189_568
QWEN3_8B = 8_190_735_360
QWEN3_14B = 14_768_307_200

PROFILE_DIR = ARTIFACT_DIR / "dataset_profiles"
SHORT_PROFILE = PROFILE_DIR / "short_512.qwen3_nothink.jsonl"
MULTITURN_PROFILE = PROFILE_DIR / "multiturn_4096.qwen3_nothink.jsonl"
POLICY = ARTIFACT_DIR / "static_packing_policy_v1.json"


def _scenario(**overrides):
    scenario = {
        "model_id": "qwen3_8b",
        "training_mode": "lora",
        "dataset_id": "multiturn_4096",
        "dataset_category": "multiturn",
        "target_gbs": 64,
        "cutoff_len": 4096,
        "actual_parameters": QWEN3_8B,
    }
    scenario.update(overrides)
    return scenario


def _requires_artifacts(test):
    if not (POLICY.is_file() and MULTITURN_PROFILE.is_file()):
        test.skipTest("frozen packing policy or dataset profile unavailable")


class TestBranchContract(unittest.TestCase):
    def setUp(self) -> None:
        _requires_artifacts(self)
        self.report = pac.build_candidate_space(
            _scenario(),
            capacity_bytes=CAPACITY,
            profile_path=str(MULTITURN_PROFILE),
            policy_path=POLICY,
        )

    def test_branches_are_separate_predictor_groups_under_one_scenario(self) -> None:
        # The frozen predictor puts ``packing`` inside ``scenario_material`` and
        # rejects a comparison_group that mixes different user scenarios, so the
        # two branches must NOT share a predictor group id.  The single candidate
        # space is expressed at the report level (one scenario, one report) while
        # each branch stays independently priceable.
        rankable_groups = {
            candidate["comparison_group"]
            for candidate in self.report["rankable_candidates"]
        }
        shadow_groups = {
            candidate["comparison_group"]
            for candidate in self.report["shadow_candidates"]
        }
        self.assertEqual(len(rankable_groups), 2)
        self.assertEqual(len(shadow_groups), 0)
        self.assertFalse(rankable_groups & shadow_groups)
        # Both branches still describe one scenario, and the report says so.
        self.assertIn(self.report["comparison_group"], rankable_groups)
        self.assertEqual(
            self.report["scenario"]["dataset_id"],
            _scenario()["dataset_id"],
        )

    def test_unpacked_candidates_are_rankable(self) -> None:
        unpacked = [
            candidate
            for candidate in self.report["rankable_candidates"]
            if candidate["candidate_branch"] == pac.BRANCH_UNPACKED
        ]
        self.assertTrue(unpacked)
        for candidate in unpacked:
            self.assertEqual(candidate["candidate_branch"], pac.BRANCH_UNPACKED)
            self.assertEqual(candidate["eligibility"], "rankable")
            self.assertFalse(candidate["packing"])

    def test_policy_positive_packed_candidates_are_released(self) -> None:
        self.assertTrue(self.report["released_packing_candidates"])
        for candidate in self.report["released_packing_candidates"]:
            self.assertEqual(candidate["candidate_branch"], pac.BRANCH_PACKED_RELEASED)
            self.assertEqual(candidate["eligibility"], "rankable")
            self.assertTrue(
                candidate["automatic_execution_allowed_after_runtime_gates"]
            )
            self.assertTrue(candidate["packing"])

    def test_packed_candidates_fix_physical_mbs_to_one(self) -> None:
        self.assertEqual(
            {
                c["physical_mbs"]
                for c in self.report["released_packing_candidates"]
            },
            {1},
        )

    def test_packed_candidates_record_the_baseline_they_replace(self) -> None:
        for candidate in self.report["released_packing_candidates"]:
            self.assertTrue(candidate["replaces_no_packing_mbs"])
            self.assertIsInstance(candidate["replaces_no_packing_mbs"], list)

    def test_module_claims_no_authority_it_does_not_have(self) -> None:
        guarantees = self.report["guarantees"]
        self.assertFalse(guarantees["derives_packing_decision_itself"])
        self.assertFalse(guarantees["overrides_policy_off"])
        self.assertFalse(guarantees["predicts_memory"])
        self.assertFalse(guarantees["ranks_candidates"])
        self.assertTrue(guarantees["releases_packed_candidates_for_runtime_admission"])
        self.assertTrue(guarantees["final_memory_admission_still_required"])
        self.assertFalse(guarantees["creates_gpu_queue"])

    def test_policy_is_bound_by_sha(self) -> None:
        binding = self.report["policy_binding"]
        self.assertEqual(len(binding["file_sha256"]), 64)
        self.assertEqual(binding["packed_physical_mbs"], 1)
        self.assertIn("report_sha256", self.report)


class TestPolicyObedience(unittest.TestCase):
    def test_short_cutoff_yields_no_packed_branch(self) -> None:
        if not (POLICY.is_file() and SHORT_PROFILE.is_file()):
            self.skipTest("artifacts unavailable")
        report = pac.build_candidate_space(
            _scenario(
                dataset_id="short_512",
                dataset_category="short",
                cutoff_len=512,
            ),
            capacity_bytes=CAPACITY,
            profile_path=str(SHORT_PROFILE),
            policy_path=POLICY,
        )
        # The frozen gate requires cutoff/mbs >= 1024, so a 512 cutoff can never
        # enable packing.  The module must obey that, not invent an exception.
        self.assertEqual(report["branch_counts"][pac.BRANCH_PACKED_SHADOW], 0)
        self.assertEqual(report["shadow_candidates"], [])
        self.assertIn("off", report["decision_summary"]["distinct_decisions"])

    def test_longer_cutoff_can_enable_packing(self) -> None:
        _requires_artifacts(self)
        report = pac.build_candidate_space(
            _scenario(), capacity_bytes=CAPACITY, profile_path=str(MULTITURN_PROFILE), policy_path=POLICY
        )
        self.assertIn("on", report["decision_summary"]["distinct_decisions"])
        self.assertTrue(report["decision_summary"]["shapes_with_packing_on"])
        self.assertTrue(report["released_packing_candidates"])

    def test_full_sft_stays_shadow_only_outside_release_scope(self) -> None:
        _requires_artifacts(self)
        report = pac.build_candidate_space(
            _scenario(training_mode="full"),
            capacity_bytes=CAPACITY,
            profile_path=str(MULTITURN_PROFILE),
            policy_path=POLICY,
        )
        self.assertEqual(report["released_packing_candidates"], [])
        self.assertTrue(report["shadow_candidates"])
        self.assertTrue(
            all(
                candidate["eligibility"] == "shadow_only"
                for candidate in report["shadow_candidates"]
            )
        )

    def test_decision_is_queried_per_baseline_shape(self) -> None:
        _requires_artifacts(self)
        report = pac.build_candidate_space(
            _scenario(), capacity_bytes=CAPACITY, profile_path=str(MULTITURN_PROFILE), policy_path=POLICY
        )
        # Packing benefit depends on the mbs it replaces, so one scenario must
        # produce more than a single decision.
        self.assertGreater(report["decision_summary"]["queried_baseline_shapes"], 1)
        shapes = {
            (row["gpu_count"], row["no_packing_mbs"])
            for row in report["packing_decisions"]
        }
        self.assertEqual(
            len(shapes), report["decision_summary"]["queried_baseline_shapes"]
        )

    def test_reason_codes_are_carried_through(self) -> None:
        _requires_artifacts(self)
        report = pac.build_candidate_space(
            _scenario(), capacity_bytes=CAPACITY, profile_path=str(MULTITURN_PROFILE), policy_path=POLICY
        )
        self.assertTrue(report["decision_summary"]["distinct_reason_codes"])
        for row in report["packing_decisions"]:
            self.assertIn("reason_codes", row)

    def test_preset_packing_scenario_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pac.build_candidate_space(
                _scenario(packing=True),
                capacity_bytes=CAPACITY,
                profile_path=str(MULTITURN_PROFILE),
                policy_path=POLICY,
            )

    def test_missing_profile_degrades_to_off_not_crash(self) -> None:
        if not POLICY.is_file():
            self.skipTest("policy unavailable")
        report = pac.build_candidate_space(
            _scenario(),
            capacity_bytes=CAPACITY,
            profile_path="/nonexistent/profile.jsonl",
            policy_path=POLICY,
        )
        self.assertEqual(report["shadow_candidates"], [])
        self.assertTrue(
            all(row["packing"] is False for row in report["packing_decisions"])
        )
        self.assertIn(
            "static_policy_query_failed",
            report["decision_summary"]["distinct_reason_codes"],
        )


class TestPredictorHandoff(unittest.TestCase):
    def setUp(self) -> None:
        _requires_artifacts(self)
        try:
            from h800_resource_predictor import H800ResourcePredictor
        except Exception:  # pragma: no cover
            self.skipTest("active predictor unavailable")
        self.predictor_cls = H800ResourcePredictor
        self.report = pac.build_candidate_space(
            _scenario(),
            capacity_bytes=CAPACITY,
            profile_path=str(MULTITURN_PROFILE),
            policy_path=POLICY,
        )

    def test_rankable_branch_is_fully_in_domain(self) -> None:
        group = self.predictor_cls().predict(
            self.report["rankable_candidates"]
        )["ranking_groups"][0]
        self.assertEqual(group["rejected_by_support_domain"], 0)
        self.assertIsNotNone(group["selected_request_id"])

    def test_released_packing_branch_can_pass_active_admission(self) -> None:
        report = self.predictor_cls().predict(
            self.report["released_packing_candidates"]
        )
        group = report["ranking_groups"][0]
        self.assertEqual(group["rejected_by_support_domain"], 0)
        self.assertTrue(group["ranked_request_ids"])
        self.assertIsNotNone(group["selected_request_id"])
        self.assertTrue(group["automatic_execution_allowed"])
        for prediction in report["predictions"]:
            self.assertTrue(
                prediction["support"]["packing_production_admission"]["verified"]
            )


if __name__ == "__main__":
    unittest.main()
