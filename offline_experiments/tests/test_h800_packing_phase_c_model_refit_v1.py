from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fit_h800_packing_phase_c_models_v1 import (  # noqa: E402
    OUTPUT,
    _memory_rows,
    _phase_c_throughput_rows,
    _require_phase_c,
)
from common import read_json  # noqa: E402


class PackingPhaseCModelRefitTest(unittest.TestCase):
    def test_phase_c_population_is_exact(self) -> None:
        report = _require_phase_c()
        settings, pairs = _phase_c_throughput_rows(report)
        self.assertEqual(4, len(settings))
        self.assertEqual(12, pairs)
        self.assertEqual({2, 3}, {row["zero_stage"] for row in settings})
        self.assertEqual({True, False}, {row["gc"] for row in settings})

    def test_memory_rows_are_collapsed_before_fit(self) -> None:
        rows = _memory_rows()
        self.assertEqual(20, len(rows))
        self.assertEqual(10, sum(bool(row["packing"]) for row in rows))
        self.assertEqual(4, len({row["profile_group"] for row in rows}))

    def test_refit_artifact_stays_fit_only(self) -> None:
        report = read_json(OUTPUT)
        self.assertEqual(61, report["throughput_effect_center"]["matched_repeat_pairs"])
        self.assertEqual(23, report["throughput_effect_center"]["setting_rows"])
        self.assertTrue(report["gates"]["paired_effect_center_refit_complete"])
        self.assertTrue(report["gates"]["packing_memory_center_refit_complete"])
        self.assertFalse(report["gates"]["packing_memory_upper_guard_complete"])
        self.assertFalse(report["gates"]["automatic_packing_recommendation_allowed"])


if __name__ == "__main__":
    unittest.main()
