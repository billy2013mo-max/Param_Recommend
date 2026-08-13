from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = EXPERIMENT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fit_rank_first_throughput_challenger_v1 import (
    FEATURE_NAMES,
    InvariantStaticProfiles,
    invariant_profile_signature,
    set_aware_log_predictions,
)


class RankFirstThroughputChallengerTests(unittest.TestCase):
    def test_nonpacking_uses_batch_max_rounded_to_multiple_of_eight(self) -> None:
        profiles = InvariantStaticProfiles.from_rows(
            {
                "tiny": [
                    {"total_tokens": 5, "label_tokens": 2, "turns": 1},
                    {"total_tokens": 9, "label_tokens": 3, "turns": 1},
                ]
            }
        )
        profile = profiles.profile(
            "tiny",
            cutoff_len=32,
            physical_mbs=2,
            packing=False,
        )
        self.assertEqual(profile["pad_to_multiple_of"], 8)
        self.assertEqual(profile["expected_padded_batch_length"], 16.0)
        self.assertEqual(
            profile["work_per_physical_sequence"]["computed_tokens"],
            16.0,
        )
        self.assertEqual(
            profile["work_per_physical_sequence"][
                "computed_attention_token_pairs"
            ],
            256.0,
        )
        self.assertAlmostEqual(profile["padding_utilization"], 7.0 / 16.0)

    def test_cutoff_above_raw_maximum_is_profile_invariant(self) -> None:
        profiles = InvariantStaticProfiles.from_rows(
            {
                "tiny": [
                    {"total_tokens": 17, "label_tokens": 5, "turns": 1},
                    {"total_tokens": 31, "label_tokens": 7, "turns": 2},
                    {"total_tokens": 9, "label_tokens": 3, "turns": 1},
                ]
            }
        )
        low = profiles.profile(
            "tiny", cutoff_len=32, physical_mbs=2, packing=False
        )
        high = profiles.profile(
            "tiny", cutoff_len=4096, physical_mbs=2, packing=False
        )
        self.assertEqual(
            invariant_profile_signature(low),
            invariant_profile_signature(high),
        )
        self.assertEqual(low["truncated_sample_fraction"], 0.0)
        self.assertEqual(low["truncated_token_fraction"], 0.0)

    def test_cutoff_remains_a_real_truncation_control(self) -> None:
        profiles = InvariantStaticProfiles.from_rows(
            {
                "tiny": [
                    {"total_tokens": 17, "label_tokens": 5, "turns": 1},
                    {"total_tokens": 41, "label_tokens": 7, "turns": 2},
                ]
            }
        )
        profile = profiles.profile(
            "tiny", cutoff_len=32, physical_mbs=1, packing=False
        )
        self.assertEqual(profile["length"]["maximum"], 32)
        self.assertEqual(profile["truncated_sample_fraction"], 0.5)
        self.assertAlmostEqual(profile["truncated_token_fraction"], 9.0 / 58.0)

    def test_feature_contract_has_no_direct_cutoff_calibration(self) -> None:
        self.assertNotIn("log2_cutoff_over_512", FEATURE_NAMES)
        self.assertNotIn("mean_length_to_cutoff", FEATURE_NAMES)
        self.assertFalse(any("cutoff" in name for name in FEATURE_NAMES))
        self.assertIn(
            "log2_expected_padded_batch_length_over_512",
            FEATURE_NAMES,
        )
        self.assertIn("truncated_sample_fraction", FEATURE_NAMES)

    def test_set_aware_output_order_is_exclusively_rank_head_order(self) -> None:
        absolute_logs = [math.log(1000.0), math.log(5000.0), math.log(3000.0)]
        rank_scores = [2.0, -1.0, 0.5]
        final = set_aware_log_predictions(absolute_logs, rank_scores)
        self.assertEqual(max(range(3), key=final.__getitem__), 0)
        self.assertEqual(min(range(3), key=final.__getitem__), 1)
        self.assertAlmostEqual(
            sum(final) / len(final),
            sum(absolute_logs) / len(absolute_logs),
        )

    def test_packing_is_explicitly_out_of_scope(self) -> None:
        profiles = InvariantStaticProfiles.from_rows(
            {"tiny": [{"total_tokens": 8, "label_tokens": 2, "turns": 1}]}
        )
        with self.assertRaisesRegex(ValueError, "packing=false only"):
            profiles.profile(
                "tiny", cutoff_len=32, physical_mbs=1, packing=True
            )


if __name__ == "__main__":
    unittest.main()
