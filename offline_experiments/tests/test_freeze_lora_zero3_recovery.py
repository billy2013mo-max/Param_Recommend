from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from freeze_lora_zero3_recovery import validate_family_shape  # noqa: E402


class FreezeLoraZero3RecoveryTests(unittest.TestCase):
    def test_only_expected_memory_family_shape_is_accepted(self) -> None:
        family = {
            "kind": "memory_boundary",
            "model_id": "qwen3_8b",
            "train_type": "lora",
            "zero": "zero3",
            "gpu_count": 4,
            "mbs_candidates": [1, 2],
        }
        self.assertTrue(validate_family_shape(family))

        for field, value in (
            ("kind", "throughput"),
            ("model_id", "qwen3_4b"),
            ("train_type", "full"),
            ("zero", "zero2"),
            ("gpu_count", 1),
            ("mbs_candidates", []),
        ):
            changed = dict(family)
            changed[field] = value
            self.assertFalse(validate_family_shape(changed), field)


if __name__ == "__main__":
    unittest.main()
