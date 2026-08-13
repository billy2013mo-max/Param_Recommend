from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_h800_rank_closeout import (  # noqa: E402
    DEFAULT_VALIDATION_ROOT,
    build_closeout_design,
    summarize_validation,
)


class H800RankCloseoutTests(unittest.TestCase):
    def test_existing_attempts_require_rank1_and_reverse_pair_retests(self) -> None:
        summary = summarize_validation(DEFAULT_VALIDATION_ROOT)
        self.assertEqual(summary["closeout_status"], "pending")
        self.assertEqual(len(summary["required_reruns"]), 3)
        self.assertEqual(summary["candidates"][0]["measured_steps"], 35)
        self.assertLess(summary["rank2_rank3_relative_gap"], 0.03)

    def test_closeout_design_never_launches_on_missing_hardware(self) -> None:
        design = build_closeout_design(
            hardware_probe={
                "selected_pool_idle": False,
                "missing_gpu_ids": [4, 5, 6, 7],
            }
        )
        self.assertFalse(design["gpu_training_started"])
        self.assertFalse(design["queues_mutated"])
        self.assertFalse(design["automatic_execution_allowed"])
        self.assertFalse(design["hardware_gate"]["passed"])
        self.assertEqual(design["launch_status"], "blocked_by_hardware_or_approval")


if __name__ == "__main__":
    unittest.main()
