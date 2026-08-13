from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_packing_information_gain_design_v1 import build_design  # noqa: E402


class PackingInformationGainDesignV1Tests(unittest.TestCase):
    def test_design_is_exact_cache_strict_gbs_and_not_authorized(self) -> None:
        design = build_design()
        self.assertEqual(design["status"], "candidate_design_only_not_materialized")
        self.assertEqual(design["budget"]["jobs"], 36)
        self.assertEqual(design["budget"]["gpu_job_equivalents"], 72)
        self.assertEqual(len(design["selected_families"]), 6)
        self.assertEqual(
            {row["workload_id"] for row in design["selected_families"]},
            {"W3", "W5", "W7", "W8"},
        )
        self.assertTrue(all(row["packing_contract"]["gates"]["candidate_admissible"] for row in design["selected_families"]))
        self.assertTrue(all(row["exact_cached_curve"]["sample_truncation_rate"] == 0 for row in design["selected_families"]))
        self.assertFalse(design["authorization"]["queue_materialized"])
        self.assertFalse(design["authorization"]["gpu_execution_allowed"])


if __name__ == "__main__":
    unittest.main()
