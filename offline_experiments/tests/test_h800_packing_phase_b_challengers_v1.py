from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fit_h800_packing_phase_b_challengers_v1 import (  # noqa: E402
    FEATURE_NAMES,
    _memory_diagnostic,
    build_setting_rows,
)


class PackingPhaseBChallengerTest(unittest.TestCase):
    def test_fit_membership_and_pairing(self) -> None:
        settings, pairs = build_setting_rows()
        primary = [row for row in settings if row["role"] == "primary_model_selection"]
        diagnostic = [row for row in settings if row["role"] != "primary_model_selection"]
        self.assertEqual(len(settings), 19)
        self.assertEqual(len(pairs), 49)
        self.assertEqual(len(primary), 17)
        self.assertEqual(len(diagnostic), 2)
        self.assertEqual(len({row["profile_group"] for row in primary}), 10)
        self.assertTrue(all(set(row["features"]) == set(FEATURE_NAMES) for row in settings))
        self.assertEqual(
            {reason for row in diagnostic for reason in row["exclusion_reasons"]},
            {
                "gc_off_has_only_one_independent_profile",
                "zero3_has_only_one_independent_profile",
            },
        )

    def test_memory_replay_has_exact_phase_b_arms(self) -> None:
        diagnostic = _memory_diagnostic()
        self.assertEqual(diagnostic["arms"], 12)
        self.assertEqual(diagnostic["by_treatment"]["packed"]["arms"], 6)
        self.assertEqual(diagnostic["by_treatment"]["unpacked"]["arms"], 6)
        self.assertFalse(diagnostic["center_refit_completed"])
        self.assertFalse(diagnostic["upper_guard_accepted"])
        comparison = diagnostic["feature_comparison"]
        self.assertEqual(comparison["arm_rows"], 12)
        self.assertEqual(comparison["profile_groups"], 6)
        self.assertEqual(
            set(comparison["models"]),
            {
                "configured_cutoff_only",
                "observed_workload",
                "observed_workload_plus_packing",
            },
        )


if __name__ == "__main__":
    unittest.main()
