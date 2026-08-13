from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from metrics_callback import TOKEN_COUNTER_KEYS, slice_consumed_token_ledger  # noqa: E402


def record(logical_samples: int, computed_tokens: int = 4096) -> dict[str, int]:
    value = {
        "computed_tokens": computed_tokens,
        "effective_tokens": computed_tokens,
        "label_tokens": computed_tokens // 2,
        "logical_samples": logical_samples,
        "physical_batches": 1,
        "computed_attention_token_pairs": computed_tokens * computed_tokens,
        "effective_attention_token_pairs": computed_tokens * computed_tokens,
    }
    assert set(value) == set(TOKEN_COUNTER_KEYS)
    return value


class ConsumedTokenLedgerTests(unittest.TestCase):
    def test_excludes_one_materialized_but_unconsumed_prefetch_batch(self) -> None:
        records = [record(6) for _ in range(21)]

        evidence = slice_consumed_token_ledger(
            records,
            gradient_accumulation_steps=10,
            completed_warmup_steps=0,
            completed_measured_steps=2,
        )

        self.assertTrue(evidence["authoritative"])
        self.assertEqual(evidence["collated_batch_count"], 21)
        self.assertEqual(evidence["consumed_batch_count_observed"], 20)
        self.assertEqual(evidence["measured_batch_count"], 20)
        self.assertEqual(evidence["prefetched_not_consumed_batch_count"], 1)
        self.assertEqual(evidence["measured_totals"]["logical_samples"], 120)
        self.assertEqual(
            evidence["prefetched_not_consumed_totals"]["logical_samples"], 6
        )

    def test_warmup_is_removed_from_measured_prefix(self) -> None:
        records = [record(index + 1) for index in range(7)]

        evidence = slice_consumed_token_ledger(
            records,
            gradient_accumulation_steps=2,
            completed_warmup_steps=1,
            completed_measured_steps=2,
        )

        self.assertEqual(evidence["warmup_totals"]["logical_samples"], 3)
        self.assertEqual(evidence["measured_totals"]["logical_samples"], 18)
        self.assertEqual(evidence["prefetched_not_consumed_totals"]["logical_samples"], 7)

    def test_incomplete_ledger_is_not_authoritative(self) -> None:
        evidence = slice_consumed_token_ledger(
            [record(1) for _ in range(3)],
            gradient_accumulation_steps=2,
            completed_warmup_steps=0,
            completed_measured_steps=2,
        )

        self.assertFalse(evidence["authoritative"])
        self.assertEqual(evidence["consumed_batch_count_required"], 4)
        self.assertEqual(evidence["consumed_batch_count_observed"], 3)


if __name__ == "__main__":
    unittest.main()
