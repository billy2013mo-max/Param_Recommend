from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from packing_gbs_contract import derive_packing_gbs_contract  # noqa: E402


class PackingGbsContractTests(unittest.TestCase):
    def test_w1_like_distribution_is_controllable_at_ga_floor(self) -> None:
        contract = derive_packing_gbs_contract(
            target_gbs=64,
            data_parallel=1,
            samples_per_pack={"mean": 31.875486, "p99": 41, "maximum": 42},
        )
        self.assertEqual(contract["gradient_accumulation_steps"], 2)
        self.assertTrue(contract["gates"]["center_integer_representable"])
        self.assertTrue(contract["gates"]["gbs_controllable_at_ga_floor"])
        self.assertTrue(contract["gates"]["candidate_admissible"])

    def test_w4_like_mean_passes_but_upper_tail_fails(self) -> None:
        contract = derive_packing_gbs_contract(
            target_gbs=64,
            data_parallel=2,
            samples_per_pack={"mean": 30.612245, "p99": 60, "maximum": 63},
        )
        self.assertEqual(contract["gradient_accumulation_steps"], 1)
        self.assertAlmostEqual(contract["expected_epoch_sample_gbs"], 61.22449, places=5)
        self.assertTrue(contract["gates"]["center_integer_representable"])
        self.assertFalse(contract["gates"]["gbs_controllable_at_ga_floor"])
        self.assertFalse(contract["gates"]["candidate_admissible"])
        self.assertEqual(contract["global_microstep_sample_gbs"]["p99"], 120)
        self.assertIn(
            "p99_global_microstep_exceeds_target_tolerance_at_ga_floor",
            contract["gates"]["reason_codes"],
        )

    def test_distribution_order_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "maximum must be >= mean and p99"):
            derive_packing_gbs_contract(
                target_gbs=64,
                data_parallel=2,
                samples_per_pack={"mean": 30, "p99": 41, "maximum": 40},
            )

    def test_empirical_p99_may_be_below_mean_for_rare_extreme_tail(self) -> None:
        contract = derive_packing_gbs_contract(
            target_gbs=64,
            data_parallel=1,
            samples_per_pack={"mean": 2.1, "p99": 2.0, "maximum": 20},
        )
        self.assertEqual(contract["samples_per_pack"]["p99"], 2.0)
        self.assertTrue(contract["gates"]["gbs_controllable_at_ga_floor"])


if __name__ == "__main__":
    unittest.main()
