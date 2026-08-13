from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_h800_packing_memory_boundary_stage1_v1 import (  # noqa: E402
    MINIMUM_CAPACITY_SAFETY_MARGIN,
    evaluate,
)


class PackingMemoryBoundaryStage1EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = evaluate()
        cls.chains = {row["chain_id"]: row for row in cls.report["chains"]}

    def test_frozen_inputs_are_bound_and_unpublished(self) -> None:
        self.assertEqual(self.report["frozen_input_errors"], [])
        self.assertFalse(self.report["publication_allowed"])
        self.assertFalse(self.report["automatic_packing_recommendation_allowed"])
        self.assertFalse(self.report["prospective_acceptance_passed"])
        self.assertFalse(self.report["stage2"]["automatic_release_allowed"])

    def test_parent_and_resume_queues_deduplicate_to_seventeen_jobs(self) -> None:
        execution = self.report["execution"]
        # 16 parent rows + 1 retry; the 12 resurfaced parent rows must not double-count.
        self.assertEqual(execution["unique_jobs"], 17)
        self.assertEqual(execution["parent_queue_jobs"], 16)
        self.assertEqual(execution["resume_queue_jobs"], 13)
        self.assertTrue(execution["all_terminal"])

    def test_the_one_incomplete_fingerprint_is_excluded_not_scored(self) -> None:
        execution = self.report["execution"]
        self.assertEqual(execution["calibration_eligible_jobs"], 16)
        self.assertEqual(len(execution["excluded_jobs"]), 1)
        excluded = execution["excluded_jobs"][0]
        self.assertEqual(excluded["job_id"], "h800packboundary1-17c90fc6ae2bc6bc")
        self.assertEqual(excluded["outcome"], "oom")
        self.assertEqual(excluded["execution_fingerprint_quality"], "incomplete")
        for row in self.report["rows"]:
            if row["job_id"] == excluded["job_id"]:
                self.assertFalse(row["observed_scored"])
                self.assertIsNone(row["center_absolute_percentage_error"])

    def test_oom_chains_report_every_oom_even_without_memory_evidence(self) -> None:
        # B4 OOMed on all four repeats and emitted no usable reserved reading.
        # Counting OOMs only from scored rows would report zero here.
        b4 = self.chains["B4_14B_Full_W3_Z2_GCon_G2"]
        self.assertEqual(b4["successes"], 0)
        self.assertEqual(b4["ooms"], 4)
        self.assertEqual(b4["ooms_without_memory_evidence"], 4)
        b3 = self.chains["B3_14B_Full_W3_Z3_GCoff_G2"]
        self.assertEqual(b3["ooms"], 4)
        self.assertEqual(b3["ooms_without_memory_evidence"], 1)

    def test_all_oom_chain_does_not_vacuously_release_stage2(self) -> None:
        for chain_id in ("B3_14B_Full_W3_Z3_GCoff_G2", "B4_14B_Full_W3_Z2_GCon_G2"):
            chain = self.chains[chain_id]
            self.assertFalse(chain["all_eligible_repeats_success"])
            self.assertFalse(chain["stage2_release_allowed"])
            self.assertIn("eligible_repeat_not_all_success", chain["stage2_block_reasons"])

    def test_b1_packed_success_breaches_the_five_percent_margin(self) -> None:
        b1 = self.chains["B1_8B_LoRA_W8_Z2_GCoff"]
        self.assertEqual(b1["successes"], 4)
        self.assertEqual(b1["ooms"], 0)
        # Every repeat succeeded, yet the packed arm ran past the safe limit, so
        # succeeding is not sufficient to release the higher cutoff.
        self.assertTrue(b1["all_eligible_repeats_success"])
        self.assertLess(
            b1["minimum_capacity_safety_margin_fraction"], MINIMUM_CAPACITY_SAFETY_MARGIN
        )
        self.assertFalse(b1["capacity_safety_margin_passed"])
        self.assertFalse(b1["stage2_release_allowed"])
        self.assertIn("capacity_safety_margin_below_5pct", b1["stage2_block_reasons"])

    def test_only_b2_is_released_to_stage2(self) -> None:
        self.assertEqual(self.report["stage2"]["released_chains"], ["B2_14B_LoRA_W7_Z2_GCoff"])
        b2 = self.chains["B2_14B_LoRA_W7_Z2_GCoff"]
        self.assertTrue(b2["stage2_release_allowed"])
        self.assertGreaterEqual(
            b2["minimum_capacity_safety_margin_fraction"], MINIMUM_CAPACITY_SAFETY_MARGIN
        )

    def test_packing_memory_ratio_is_underpredicted_on_b1(self) -> None:
        b1 = self.chains["B1_8B_LoRA_W8_Z2_GCoff"]
        self.assertTrue(b1["packing_memory_ratio_underpredicted"])
        self.assertGreater(b1["observed_packed_over_unpacked_reserved_ratio"], 5.0)
        self.assertLess(b1["predicted_packed_over_unpacked_center_ratio"], 1.6)

    def test_vacuous_gates_are_reported_as_vacuous(self) -> None:
        acceptance = self.report["acceptance"]
        # Stage 1 queued only rejected candidates, so a zero false-safe count is
        # not evidence of admission safety and must be flagged.
        self.assertEqual(acceptance["admitted_candidate_count"], 0)
        self.assertTrue(acceptance["false_safe_oom_gate_is_vacuous"])
        self.assertEqual(acceptance["successful_p95_upper_coverage"], 1.0)
        self.assertTrue(acceptance["upper_coverage_gate_is_vacuous"])
        self.assertEqual(acceptance["successes_covered_by_guard_inside_safe_limit"], 0)
        self.assertEqual(acceptance["oom_rows_whose_upper_guard_exceeds_safe_limit"], 8)

    def test_acceptance_fails_and_blocks_stage2_materialization(self) -> None:
        self.assertFalse(self.report["acceptance"]["all_thresholds_passed"])
        self.assertFalse(self.report["stage1_complete_for_joint_fit"])
        self.assertIn("capacity_safety_margin_below_floor", self.report["blockers"])
        self.assertIn("upper_guard_above_safe_limit_on_oom_rows", self.report["blockers"])
        self.assertEqual(self.report["next_step"], "repair_memory_upper_guard_before_stage2")

    def test_reserved_reduction_is_max_over_ranks(self) -> None:
        self.assertEqual(
            self.report["memory_evidence_policy"]["reserved_reduction"],
            "maximum_over_all_rank_summaries",
        )
        # B1 packed ranks read 113.6 and 137.6 GiB; taking rank 0 alone would
        # hide the breach entirely.
        packed = [
            row
            for row in self.report["rows"]
            if row["chain_id"] == "B1_8B_LoRA_W8_Z2_GCoff" and row["packing"]
        ]
        self.assertTrue(packed)
        for row in packed:
            self.assertGreater(row["observed_max_reserved_gib"], row["safe_limit_gib"])
            self.assertTrue(row["success_exceeding_safe_limit"])


if __name__ == "__main__":
    unittest.main()
