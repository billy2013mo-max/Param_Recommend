from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_h800_prospective_holdout import (  # noqa: E402
    DEFAULT_SCENARIOS,
    build_design,
)


class H800ProspectiveHoldoutTests(unittest.TestCase):
    def test_default_design_is_bounded_and_never_executable(self) -> None:
        design = build_design(
            hardware_probe={
                "selected_pool_idle": False,
                "missing_gpu_ids": [4, 5, 6, 7],
                "selected_gpu_rows": [],
            }
        )
        self.assertEqual(design["schema"], "sft_h800_prospective_holdout_design/v1")
        self.assertEqual(len(design["scenarios"]), 6)
        self.assertEqual(len(design["candidate_slots"]), 24)
        for scenario in design["scenarios"]:
            transition = scenario["scale_out_transition"]
            left, right = (1, 2) if transition == "1_to_2" else (2, 4)
            counts = {
                gpu: sum(
                    slot["scenario_id"] == scenario["scenario_id"]
                    and slot["gpu_count"] == gpu
                    for slot in design["candidate_slots"]
                )
                for gpu in (left, right)
            }
            self.assertEqual(counts, {left: 2, right: 2})
        self.assertFalse(design["gpu_training_started"])
        self.assertFalse(design["queues_mutated"])
        self.assertFalse(design["materialization_allowed"])
        self.assertEqual(
            design["acceptance_contract"]["minimum_throughput_ratio_per_doubling"],
            1.8,
        )
        self.assertEqual(
            design["acceptance_contract"]["minimum_safe_candidates_per_declared_endpoint"],
            2,
        )

    def test_reused_profile_requires_explicit_opt_in(self) -> None:
        scenarios = [dict(row) for row in DEFAULT_SCENARIOS]
        scenarios[1]["dataset_profile_id"] = scenarios[0]["dataset_profile_id"]
        with self.assertRaisesRegex(ValueError, "reused without explicit opt-in"):
            build_design(scenarios, hardware_probe={"selected_pool_idle": False})

    def test_offload_is_not_silently_added_to_dense_holdout(self) -> None:
        scenarios = [dict(row) for row in DEFAULT_SCENARIOS]
        scenarios[0]["offload"] = True
        with self.assertRaisesRegex(ValueError, "packing/offload"):
            build_design(scenarios, hardware_probe={"selected_pool_idle": False})

    def test_duplicate_scenario_id_is_rejected(self) -> None:
        scenarios = [dict(row) for row in DEFAULT_SCENARIOS]
        scenarios[1]["scenario_id"] = scenarios[0]["scenario_id"]
        with self.assertRaisesRegex(ValueError, "scenario_id"):
            build_design(scenarios, hardware_probe={"selected_pool_idle": False})

    def test_custom_design_can_use_the_planned_20_slot_lower_bound(self) -> None:
        design = build_design(
            [dict(row) for row in DEFAULT_SCENARIOS[:5]],
            hardware_probe={"selected_pool_idle": False},
        )
        self.assertEqual(len(design["candidate_slots"]), 20)


if __name__ == "__main__":
    unittest.main()
