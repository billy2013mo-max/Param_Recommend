from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import fit_h800_effective_sequence_v3 as v3
from experiment_effective_sequence_memory_basis import ProfileLengths


class EffectiveSequenceTests(unittest.TestCase):
    def test_round_up_matches_collator_multiple(self) -> None:
        self.assertEqual(v3.round_up(12_513, 8), 12_520)
        self.assertEqual(v3.round_up(4_096, 8), 4_096)

    def test_nonpacking_uses_profile_max_before_padding(self) -> None:
        profile = ProfileLengths({"toy": [4, 12_513, 7]})
        self.assertEqual(
            v3.effective_sequence_tokens(
                profile,
                dataset_id="toy",
                cutoff_len=32_768,
                packing=False,
                pad_multiple=8,
            )["tokens"],
            12_520,
        )

    def test_cutoff_caps_profile_before_padding(self) -> None:
        profile = ProfileLengths({"toy": [40_000]})
        self.assertEqual(
            v3.effective_sequence_tokens(
                profile,
                dataset_id="toy",
                cutoff_len=4_096,
                packing=False,
                pad_multiple=8,
            )["tokens"],
            4_096,
        )

    def test_packing_uses_cutoff_even_without_profile_lengths(self) -> None:
        profile = ProfileLengths({})
        self.assertEqual(
            v3.effective_sequence_tokens(
                profile,
                dataset_id="missing",
                cutoff_len=32_768,
                packing=True,
                pad_multiple=8,
            )["tokens"],
            32_768,
        )

    def test_nonpacking_missing_profile_is_unavailable(self) -> None:
        profile = ProfileLengths({})
        self.assertIsNone(
            v3.effective_sequence_tokens(
                profile,
                dataset_id="missing",
                cutoff_len=32_768,
                packing=False,
                pad_multiple=8,
            )
        )


class CalibrationTests(unittest.TestCase):
    def test_conformal_requires_independent_scenarios(self) -> None:
        unavailable = v3.conformal_upper(
            {str(i): float(i) for i in range(18)}, coverage=0.95
        )
        available = v3.conformal_upper(
            {str(i): float(i) for i in range(19)}, coverage=0.95
        )

        self.assertFalse(unavailable["available"])
        self.assertEqual(unavailable["independent_scenarios"], 18)
        self.assertTrue(available["available"])
        self.assertEqual(available["log_upper"], 18.0)

    def test_duplicate_rows_collapse_to_one_scenario_score(self) -> None:
        scores = [
            {"scenario_id": "same", "score": 1.0},
            {"scenario_id": "same", "score": 3.0},
            {"scenario_id": "other", "score": 2.0},
        ]

        collapsed = v3._collapse_scores(scores, key_field=None)

        self.assertEqual(collapsed, {"pooled": {"same": 3.0, "other": 2.0}})

    def test_critical_bucket_fails_closed_without_expansion_evidence(self) -> None:
        record = {
            "selector": {
                "training_mode": "lora",
                "zero_stage": 2,
                "gradient_checkpointing": False,
                "packing": False,
            },
            "scenario": {"gpu_count": 2},
            "memory": {"analytic_reference_bytes": 100.0},
        }
        key = v3.mechanism_key(record)
        bundle = {
            "allocated_model": object(),
            "reserved_model": object(),
            "allocated_residual_upper": {"pooled": {"available": True}},
            "reserved_residual_upper": {"pooled": {"available": True}},
            "reservation_expansion": {
                "entries": {key: {"available": False, "reason": "insufficient"}}
            },
            "oom_guards": {},
        }
        calibrated = {
            "available": True,
            "center_bytes": 110.0,
            "upper_bytes": 120.0,
            "calibration_level": "pooled",
            "calibration": {"available": True},
        }

        with patch.object(v3, "_calibrated_upper", return_value=calibrated):
            result = v3.predict_variant(
                record,
                bundle=bundle,
                variant=v3.VARIANT_FULL,
            )

        self.assertFalse(result["available"])
        self.assertIn(
            "critical_expansion_guard_unavailable_fail_closed", result["issues"]
        )


if __name__ == "__main__":
    unittest.main()
