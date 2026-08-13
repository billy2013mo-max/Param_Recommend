from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = EXPERIMENT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_rank_first_challenger_unseen_holdouts_v1 import build_report


class RankFirstFitDisjointReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report(
            challenger_path=(
                EXPERIMENT_ROOT
                / "artifacts"
                / "rank_first_throughput_challenger_v1.json"
            ),
            inventory_path=EXPERIMENT_ROOT / "artifacts" / "model_inventory.json",
            hardware_path=EXPERIMENT_ROOT / "config" / "hardware.json",
            theory_path=EXPERIMENT_ROOT / "artifacts" / "h800_theory_basis.json",
        )

    def test_replay_is_fit_disjoint_and_does_not_refit(self) -> None:
        self.assertTrue(self.report["evaluation_contract"]["fit_disjoint"])
        self.assertEqual(
            self.report["disjointness_audit"]["dataset_id_overlap"], []
        )
        self.assertFalse(self.report["model_refit"])
        self.assertFalse(self.report["hyperparameters_changed"])
        self.assertFalse(
            self.report["evaluation_contract"][
                "temporally_fresh_for_challenger"
            ]
        )

    def test_primary_population_and_metrics_are_stable(self) -> None:
        population = self.report["population"]
        metrics = self.report["evaluation"]["primary_aggregate"]
        self.assertEqual(population["primary_multi_candidate_rows"], 48)
        self.assertEqual(population["primary_scenarios"], 14)
        self.assertEqual(metrics["all_pairwise_comparisons"], 78)
        self.assertAlmostEqual(metrics["all_pairwise_accuracy"], 71.0 / 78.0)
        self.assertEqual(metrics["cross_mechanism_pairwise_comparisons"], 57)
        self.assertAlmostEqual(
            metrics["cross_mechanism_pairwise_accuracy"], 51.0 / 57.0
        )
        self.assertAlmostEqual(metrics["exact_top1_fraction"], 13.0 / 14.0)
        self.assertLess(metrics["worst_top1_regret"], 0.02)

    def test_no_strict_same_gpu_same_mbs_mechanism_pair_exists(self) -> None:
        scopes = self.report["evaluation"]["primary_aggregate"][
            "pair_scope_breakdown"
        ]
        self.assertEqual(
            scopes["same_gpu_and_mbs_cross_mechanism"]["comparisons"], 0
        )
        self.assertIsNone(
            scopes["same_gpu_and_mbs_cross_mechanism"]["accuracy"]
        )

    def test_oom_and_out_of_catalog_models_are_not_ranked(self) -> None:
        reasons = self.report["population"]["excluded_reason_counts"]
        self.assertEqual(reasons["outcome_not_success"], 2)
        self.assertEqual(reasons["model_outside_in_domain_catalog"], 3)


if __name__ == "__main__":
    unittest.main()
