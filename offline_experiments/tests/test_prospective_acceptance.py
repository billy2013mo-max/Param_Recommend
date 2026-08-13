from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prospective_acceptance import (  # noqa: E402
    evaluate_memory_acceptance,
    evaluate_prospective_acceptance,
    evaluate_ranking_acceptance,
    evaluate_scale_out_acceptance,
    build_scale_out_acceptance_report,
)


def _summary(gpu_count: int, *, lower: float | None = None, upper: float | None = None) -> dict:
    return {
        "gpu_count": gpu_count,
        "scenario_contract_sha256": "same-scenario-runtime",
        "memory_gate_passed": True,
        "admitted_candidate_count": 2,
        "best_candidate_request_id": f"best-{gpu_count}",
        "predicted_throughput": 100.0 * gpu_count,
        "conservative_lower_throughput": lower,
        "conservative_upper_throughput": upper,
    }


class ProspectiveAcceptanceTests(unittest.TestCase):
    def test_memory_oom_is_right_censored_and_false_safe_is_counted(self) -> None:
        report = evaluate_memory_acceptance(
            [
                {
                    "scenario_id": "a",
                    "outcome": "success",
                    "predicted_admit": True,
                    "actual_safe_success": True,
                    "upper_covers_observed": True,
                },
                {
                    "scenario_id": "a",
                    "outcome": "oom",
                    "predicted_admit": False,
                },
                {
                    "scenario_id": "b",
                    "outcome": "oom",
                    "predicted_admit": True,
                },
            ]
        )
        self.assertEqual(report["false_safe_oom"], 1)
        self.assertEqual(report["memory_safety_failures"], 1)
        self.assertFalse(report["passes"])

    def test_ranking_requires_two_actual_safe_successes(self) -> None:
        group = {
            "scenario_id": "s",
            "candidates": [
                {
                    "candidate_id": "a",
                    "predicted_admit": True,
                    "predicted_throughput": 10,
                    "outcome": "success",
                    "actual_safe_success": True,
                    "observed_throughput": 100,
                },
                {
                    "candidate_id": "b",
                    "predicted_admit": True,
                    "predicted_throughput": 9,
                    "outcome": "success",
                    "actual_safe_success": True,
                    "observed_throughput": 90,
                },
            ],
        }
        report = evaluate_ranking_acceptance([group], maximum_top1_regret=0.1)
        self.assertTrue(report["passes"])
        self.assertEqual(report["worst_top1_regret"], 0.0)

    def test_scale_out_requires_fresh_measured_lower_bound(self) -> None:
        pair = {"pair_id": "s-1-2", "baseline": _summary(1, upper=110), "expanded": _summary(2, lower=200)}
        missing = evaluate_scale_out_acceptance([pair])
        self.assertFalse(missing["passes"])
        self.assertEqual(missing["false_positive_scale_out_claims"], 1)
        passed = evaluate_scale_out_acceptance([{**pair, "measured_ratio_lower": 1.9}])
        self.assertTrue(passed["passes"])

    def test_combined_report_is_non_executable_and_reports_split_blockers(self) -> None:
        report = evaluate_prospective_acceptance(
            memory_rows=[
                {
                    "scenario_id": "s",
                    "outcome": "success",
                    "predicted_admit": True,
                    "actual_safe_success": True,
                    "upper_covers_observed": True,
                },
            ],
            ranking_groups=[],
            fresh_split=False,
            scenario_level_split=False,
        )
        self.assertFalse(report["publication_allowed"])
        self.assertFalse(report["gpu_training_started"])
        self.assertFalse(report["queues_mutated"])
        self.assertIn("holdout_is_not_fresh", report["publication_blockers"])
        self.assertIn("scenario_level_split_missing", report["publication_blockers"])

    def test_scale_projection_is_separate_and_non_executable(self) -> None:
        report = evaluate_prospective_acceptance(
            memory_rows=[],
            ranking_groups=[],
            scale_pairs=[],
            fresh_split=False,
            scenario_level_split=True,
        )
        scale = build_scale_out_acceptance_report(report)
        self.assertEqual(scale["schema"], "sft_scale_out_acceptance_report/v1")
        self.assertFalse(scale["publication_allowed"])
        self.assertFalse(scale["automatic_execution_allowed"])
        self.assertIn("scale_out_evidence_missing", scale["publication_blockers"])


if __name__ == "__main__":
    unittest.main()
