from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import read_json  # noqa: E402
from select_h800_packing_memory_boundary_v1 import OUTPUT  # noqa: E402


class PackingMemoryBoundarySelectionTest(unittest.TestCase):
    def test_selected_design_is_balanced_and_fail_closed(self) -> None:
        report = read_json(OUTPUT)
        selected = report["selected_settings"]
        self.assertEqual(8, len(selected))
        self.assertEqual(4, len({row["chain_id"] for row in selected}))
        self.assertEqual(32, report["design"]["planned_jobs"])
        self.assertTrue(
            all(value >= 1 for value in report["design"]["required_capacity_bin_coverage"].values())
        )
        self.assertTrue(all(len(row["arms"]) == 2 for row in selected))
        self.assertTrue(
            all({bool(arm["packing"]) for arm in row["arms"]} == {False, True} for row in selected)
        )
        self.assertTrue(report["design"]["all_selected_gbs_contracts_passed"])
        self.assertTrue(all(arm["gbs_contract_passed"] for row in selected for arm in row["arms"]))
        self.assertTrue(
            all(
                abs(row["pair_refit_center_capacity_fraction"] - row["target_capacity_fraction"]) <= 0.01
                for row in selected
            )
        )
        self.assertFalse(report["gates"]["memory_upper_guard_complete"])
        self.assertFalse(report["gates"]["automatic_gpu_launch_allowed"])
        self.assertFalse(report["execution_contract"]["queue_materialized"])


if __name__ == "__main__":
    unittest.main()
