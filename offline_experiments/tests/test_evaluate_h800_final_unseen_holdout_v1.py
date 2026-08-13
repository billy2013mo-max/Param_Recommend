from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_h800_final_unseen_holdout_v1 import evaluate, render_markdown  # noqa: E402


class FinalUnseenHoldoutAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = evaluate()

    def test_exact_campaign_evidence_is_complete_and_fresh(self) -> None:
        integrity = self.report["integrity"]
        self.assertTrue(integrity["passes"], integrity["errors"])
        self.assertEqual(integrity["queue_jobs"], 10)
        self.assertEqual(integrity["joined_rows"], 10)
        self.assertEqual(integrity["outcomes"], {"success": 10})
        self.assertTrue(integrity["fresh_holdout_split_passes"])
        self.assertEqual(
            len(self.report["source_files"]["terminal_result_manifests"]),
            10,
        )

    def test_safety_upper_passes_but_frozen_precision_contract_fails(self) -> None:
        memory = self.report["memory"]
        self.assertTrue(memory["safety_gate_passes"])
        self.assertEqual(memory["false_safe_oom"], 0)
        self.assertEqual(memory["scenario_equal_p05_coverage"], 1.0)
        self.assertAlmostEqual(memory["center_mape"], 0.6599698059, places=8)
        self.assertAlmostEqual(memory["center_p90_ape"], 1.8145850319, places=8)
        self.assertFalse(memory["center_precision_passes"])
        self.assertFalse(memory["replacement_gate_passes"])

    def test_failure_is_localized_to_8b_lora_and_one_false_rejection(self) -> None:
        memory = self.report["memory"]
        slices = {
            (row["model_id"], row["train_type"]): row
            for row in memory["model_train_slices"]
        }
        self.assertAlmostEqual(slices[("qwen3_14b", "full")]["center_mape"], 0.0163150637, places=8)
        self.assertAlmostEqual(slices[("qwen3_8b", "lora")]["center_mape"], 1.0890729674, places=8)
        self.assertEqual(memory["false_rejected_safe_rows"], 1)
        rejected = memory["false_rejected_candidates"][0]
        self.assertEqual(rejected["scenario_id"], "unseen_business_user_long_v1__qwen3_8b_lora")
        self.assertEqual(rejected["mbs"], 2)
        self.assertLess(rejected["observed_reserved_gib"], rejected["safe_limit_gib"])

    def test_memory_gated_v4b_still_meets_hit_at_90_diagnostically(self) -> None:
        ranking = self.report["throughput_ranking"]
        self.assertEqual(ranking["eligible_endpoint_count"], 4)
        self.assertEqual(ranking["exact_top1_count"], 3)
        self.assertEqual(ranking["endpoints_within_10_percent_count"], 4)
        self.assertEqual(ranking["endpoints_within_10_percent_fraction"], 1.0)
        self.assertAlmostEqual(ranking["worst_top1_regret"], 0.0821120443, places=8)
        self.assertEqual(ranking["scoreless_actual_safe_candidates"], 1)
        self.assertFalse(ranking["decision"]["formal_v4b_promotion_claim_allowed"])

    def test_holdout_cannot_enter_fit_or_validate_the_next_refit(self) -> None:
        ingestion = self.report["ingestion_eligibility"]
        decisions = self.report["decisions"]
        self.assertEqual(ingestion["holdout_partition_rows"], 10)
        self.assertFalse(ingestion["fit_or_anchor_ingestion_allowed"])
        self.assertFalse(ingestion["holdout_may_be_reused_to_validate_a_refit"])
        self.assertTrue(decisions["final_holdout_is_now_consumed"])
        self.assertFalse(decisions["may_reuse_this_holdout_for_next_acceptance"])
        self.assertFalse(decisions["profile_aware_memory_challenger_replacement_allowed"])

    def test_human_report_states_the_non_promotion_decision(self) -> None:
        markdown = render_markdown(self.report)
        self.assertIn("reserved center MAPE 为 66.0%", markdown)
        self.assertIn("profile-aware challenger 替换当前模型：`false`", markdown)
        self.assertIn("Hit@90%：4/4", markdown)


if __name__ == "__main__":
    unittest.main()
