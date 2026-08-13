#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rtx4090_physical_v4b_predictor import (  # noqa: E402
    RTX4090PhysicalV4BPredictor,
)


class RTX4090PhysicalV4BPredictorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = RTX4090PhysicalV4BPredictor()
        example = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "rtx4090_physical_v4b_request.json"
        )
        cls.requests = json.loads(example.read_text())["candidates"]

    def test_safe_candidates_are_ranked(self) -> None:
        report = self.predictor.predict(self.requests)
        self.assertFalse(report["gpu_experiments_launched"])
        self.assertFalse(report["queues_mutated"])
        self.assertEqual(
            report["ranking_groups"][0]["status"],
            "ranked",
        )
        self.assertEqual(
            report["ranking_groups"][0]["ranked_request_ids"],
            [
                "qwen3-0p6b-full-gpu1-mbs2",
                "qwen3-0p6b-full-gpu1-mbs1",
            ],
        )
        self.assertTrue(
            all(
                row["memory"]["admitted"]
                and row["throughput"]["prediction_available"]
                for row in report["predictions"]
            )
        )

    def test_memory_rejection_prevents_throughput_ranking(self) -> None:
        request = {
            **self.requests[0],
            "request_id": "memory-reject",
            "mbs": 16,
        }
        report = self.predictor.predict([request])
        row = report["predictions"][0]
        self.assertFalse(row["memory"]["admitted"])
        self.assertFalse(row["throughput"]["prediction_available"])
        self.assertEqual(
            report["ranking_groups"][0]["status"],
            "no_memory_safe_candidate",
        )


if __name__ == "__main__":
    unittest.main()
