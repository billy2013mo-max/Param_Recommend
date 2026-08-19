from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import freeze_h800_packing_transfer_validation_predictions_v1 as freeze_module  # noqa: E402
from common import read_jsonl, sha256_file  # noqa: E402
import prepare_h800_packing_transfer_validation_v1 as campaign  # noqa: E402


class PackingTransferPredictionFreezeTests(unittest.TestCase):
    def test_freeze_is_exact_and_prospective(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.json"
            report = freeze_module.freeze(output)
        jobs = read_jsonl(campaign.QUEUE)
        self.assertEqual(len(report["predictions"]), 24)
        self.assertEqual(len(report["ranking_freeze"]), 6)
        self.assertFalse(report["validation_outcomes_read"])
        self.assertFalse(report["coefficients_fitted"])
        self.assertEqual(report["queue_binding"]["sha256"], sha256_file(campaign.QUEUE))
        self.assertEqual(
            {row["job_id"] for row in report["predictions"]},
            {row["job_id"] for row in jobs},
        )
        self.assertTrue(all(row["admitted"] is True for row in report["predictions"]))
        self.assertTrue(
            all(
                math.isfinite(float(row["predicted_ranking_score"]))
                and float(row["predicted_ranking_score"]) > 0.0
                for row in report["predictions"]
            )
        )


if __name__ == "__main__":
    unittest.main()
