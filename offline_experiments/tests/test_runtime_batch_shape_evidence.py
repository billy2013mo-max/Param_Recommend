from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from metrics_callback import slice_consumed_batch_shape_ledger


class RuntimeBatchShapeEvidenceTests(unittest.TestCase):
    def test_prefetch_is_excluded_and_measured_steps_are_grouped(self) -> None:
        records = [
            {
                "physical_batch_size": 2,
                "padded_sequence_length": padded,
                "logical_sequence_lengths": lengths,
                "padding_tokens": padding,
                "packing": False,
            }
            for padded, lengths, padding in (
                (128, [80, 128], 48),
                (256, [200, 256], 56),
                (512, [400, 500], 124),
                (384, [300, 384], 84),
                (1024, [900, 1000], 148),  # prefetched, never consumed
            )
        ]
        evidence = slice_consumed_batch_shape_ledger(
            records,
            gradient_accumulation_steps=2,
            completed_warmup_steps=1,
            completed_measured_steps=1,
        )

        self.assertTrue(evidence["authoritative"])
        self.assertEqual(evidence["prefetched_not_consumed_batch_count"], 1)
        self.assertEqual(len(evidence["measured_microbatches"]), 2)
        step = evidence["measured_steps"][0]
        self.assertEqual(step["padded_sequence_length_max"], 512)
        self.assertEqual(step["logical_sequence_length_max"], 500)
        self.assertEqual(step["logical_sample_count"], 4)
        self.assertEqual(step["padding_tokens"], 208)

    def test_lengths_only_contract_rejects_empty_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "logical lengths"):
            slice_consumed_batch_shape_ledger(
                [
                    {
                        "physical_batch_size": 1,
                        "padded_sequence_length": 128,
                        "logical_sequence_lengths": [],
                        "padding_tokens": 0,
                        "packing": False,
                    }
                ],
                gradient_accumulation_steps=1,
                completed_warmup_steps=0,
                completed_measured_steps=1,
            )


if __name__ == "__main__":
    unittest.main()
