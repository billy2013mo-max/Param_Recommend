from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_h800_packing_virtual_mbs_v1 import (
    OUTPUT,
    build_matched_pairs,
    virtual_mbs_bracket,
)
from common import read_json


class PackingVirtualMbsAnalysisTest(unittest.TestCase):
    def test_matched_population_is_exact_and_uses_static_n_pack(self) -> None:
        rows = build_matched_pairs()
        self.assertEqual(61, len(rows))
        self.assertEqual(23, len({row["setting_id"] for row in rows}))
        self.assertEqual(10, len({row["profile_group"] for row in rows}))
        self.assertEqual(
            {
                "phase_b",
                "phase_c",
                "real_business",
                "strict_gbs_repair",
                "strict_interactions",
            },
            {row["source"] for row in rows},
        )
        for row in rows:
            self.assertGreater(row["n_pack_mean"], 1.0)
            self.assertEqual(1, row["unpacked_mbs"])
            self.assertAlmostEqual(
                row["expected_packed_gbs"],
                row["gpu_count"] * row["packed_ga"] * row["n_pack_mean"],
            )

    def test_fractional_virtual_mbs_uses_log2_interpolation(self) -> None:
        exact = virtual_mbs_bracket(4.0)
        self.assertEqual(4, exact["lower_mbs"])
        self.assertEqual(4, exact["upper_mbs"])
        self.assertEqual(0.0, exact["upper_log_weight"])
        self.assertFalse(exact["is_grid_extrapolation"])

        midpoint = virtual_mbs_bracket(math.sqrt(32.0))
        self.assertEqual(4, midpoint["lower_mbs"])
        self.assertEqual(8, midpoint["upper_mbs"])
        self.assertAlmostEqual(0.5, midpoint["upper_log_weight"])

        above = virtual_mbs_bracket(33.0)
        self.assertEqual(16, above["lower_mbs"])
        self.assertEqual(32, above["upper_mbs"])
        self.assertTrue(above["is_grid_extrapolation"])

    def test_artifact_is_analysis_only_and_coefficient_gate_fails(self) -> None:
        report = read_json(OUTPUT)
        self.assertEqual(61, report["population"]["matched_repeat_pairs"])
        self.assertEqual(23, report["population"]["settings"])
        self.assertFalse(report["publishable"])
        self.assertFalse(report["gates"]["global_virtual_mbs_coefficient_stable"])
        self.assertFalse(report["gates"]["automatic_packing_recommendation_allowed"])
        self.assertEqual(
            0,
            report["frozen_main_model_contract"]["packing_primary_training_rows"],
        )
        self.assertEqual(
            0.0,
            report["frozen_main_model_contract"]["packing_correction_coefficient"],
        )

        first = report["pair_rows"][0]
        observed = first["observed_ratio"]["effective_tokens_per_second"]
        predicted = first["prediction"]["virtual_over_unpacked_ratio"]
        self.assertAlmostEqual(
            observed / predicted,
            first["residual_coefficient"]["virtual_effective"],
        )


if __name__ == "__main__":
    unittest.main()
