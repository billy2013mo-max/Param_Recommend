from __future__ import annotations

import unittest
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cross_card_scaling import (
    DEFAULT_MINIMUM_THROUGHPUT_RATIO,
    POLICY_ID,
    evaluate_doubling,
    evaluate_scale_out_sequence,
)


def summary(
    gpu_count: int,
    *,
    lower: float | None = None,
    upper: float | None = None,
    point: float | None = None,
    candidates: int = 2,
    candidate_id: str | None = None,
    memory_gate: bool = True,
) -> dict:
    return {
        "gpu_count": gpu_count,
        "scenario_contract_sha256": "shared-scenario-contract",
        "memory_gate_passed": memory_gate,
        "admitted_candidate_count": candidates,
        "best_candidate_request_id": candidate_id or f"best-{gpu_count}",
        "predicted_throughput": point,
        "conservative_lower_throughput": lower,
        "conservative_upper_throughput": upper,
    }


class CrossCardScalingTests(unittest.TestCase):
    def test_passes_only_when_conservative_ratio_clears_threshold(self) -> None:
        decision = evaluate_doubling(
            summary(1, point=100, upper=110),
            summary(2, point=200, lower=200),
        )
        self.assertEqual(decision["status"], "passed")
        self.assertTrue(decision["passes"])
        self.assertAlmostEqual(decision["predicted_ratio"], 2.0)
        self.assertAlmostEqual(decision["conservative_ratio_lower"], 200 / 110)
        self.assertEqual(decision["minimum_ratio"], DEFAULT_MINIMUM_THROUGHPUT_RATIO)

    def test_missing_bounds_fails_closed_even_with_good_point_ratio(self) -> None:
        decision = evaluate_doubling(
            summary(1, point=100),
            summary(2, point=200),
        )
        self.assertEqual(decision["status"], "conservative_bound_unavailable")
        self.assertFalse(decision["passes"])
        self.assertEqual(decision["predicted_ratio"], 2.0)

    def test_ratio_equal_to_1_8_is_inclusive(self) -> None:
        decision = evaluate_doubling(
            summary(1, upper=100.0),
            summary(2, lower=180.0),
        )
        self.assertEqual(decision["status"], "passed")
        self.assertTrue(decision["passes"])
        self.assertAlmostEqual(decision["conservative_ratio_lower"], 1.8)

    def test_insufficient_candidates_stops_scale_out(self) -> None:
        decision = evaluate_doubling(
            summary(1, lower=100, upper=110, candidates=1),
            summary(2, lower=200, upper=220),
        )
        self.assertEqual(decision["status"], "insufficient_candidate_evidence")
        self.assertFalse(decision["passes"])

    def test_missing_shared_scenario_contract_fails_closed(self) -> None:
        baseline = summary(1, upper=100)
        expanded = summary(2, lower=200)
        baseline.pop("scenario_contract_sha256")
        decision = evaluate_doubling(baseline, expanded)
        self.assertEqual(decision["status"], "scenario_contract_unavailable")
        self.assertFalse(decision["passes"])

    def test_mismatched_shared_scenario_contract_fails_closed(self) -> None:
        baseline = summary(1, upper=100)
        expanded = summary(2, lower=200)
        expanded["scenario_contract_sha256"] = "different-scenario"
        decision = evaluate_doubling(baseline, expanded)
        self.assertEqual(decision["status"], "scenario_contract_mismatch")
        self.assertFalse(decision["passes"])

    def test_sequence_stops_at_first_failed_doubling(self) -> None:
        report = evaluate_scale_out_sequence(
            [
                summary(1, lower=95, upper=110, point=100),
                summary(2, lower=200, upper=210, point=180),
                summary(4, lower=250, upper=300, point=300),
            ]
        )
        self.assertEqual(report["selection_policy"], POLICY_ID)
        self.assertEqual(report["minimum_admitted_gpu_count"], 1)
        self.assertEqual(report["recommended_gpu_count"], 2)
        self.assertEqual(report["scaling_stop_reason"], "threshold_not_cleared")
        self.assertEqual(len(report["scaling_steps"]), 2)
        self.assertTrue(report["scaling_steps"][0]["passes"])
        self.assertFalse(report["scaling_steps"][1]["passes"])

    def test_sequence_recommends_four_when_both_doublings_clear(self) -> None:
        report = evaluate_scale_out_sequence(
            [
                summary(1, lower=95, upper=100, point=100),
                summary(2, lower=180, upper=190, point=180),
                summary(4, lower=342, upper=350, point=342),
            ]
        )
        self.assertEqual(report["recommended_gpu_count"], 4)
        self.assertEqual(report["scaling_stop_reason"], "all_available_doublings_passed")
        self.assertTrue(all(step["passes"] for step in report["scaling_steps"]))

    def test_no_candidate_evidence_is_not_a_zero_throughput_claim(self) -> None:
        report = evaluate_scale_out_sequence(
            [summary(1, candidates=0, memory_gate=True)]
        )
        self.assertIsNone(report["recommended_gpu_count"])
        self.assertEqual(
            report["scaling_stop_reason"],
            "no_admitted_candidate_evidence",
        )
        self.assertFalse(report["automatic_execution_allowed"])


if __name__ == "__main__":
    unittest.main()
