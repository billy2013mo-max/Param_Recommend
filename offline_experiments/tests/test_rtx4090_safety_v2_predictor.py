#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import sha256_json  # noqa: E402
from rtx4090_safety_v2_predictor import (  # noqa: E402
    RTX4090SafetyV2Predictor,
)


class RTX4090SafetyV2PredictorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = RTX4090SafetyV2Predictor()
        example = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "rtx4090_physical_v4b_request.json"
        )
        cls.requests = json.loads(example.read_text())["candidates"]

    def test_conditional_gate_and_v4b_ranking(self) -> None:
        report = self.predictor.predict(self.requests)
        unsigned = dict(report)
        digest = unsigned.pop("report_sha256")
        self.assertEqual(digest, sha256_json(unsigned))
        self.assertEqual(
            report["ranking_groups"][0]["ranked_request_ids"],
            [
                "qwen3-0p6b-full-gpu1-mbs2",
                "qwen3-0p6b-full-gpu1-mbs1",
            ],
        )
        for row in report["predictions"]:
            memory = row["memory"]
            self.assertEqual(
                memory["admission_policy"],
                "conditional_unsafe_score",
            )
            self.assertLess(
                memory["unsafe_score"],
                memory["unsafe_score_threshold"],
            )
            self.assertTrue(row["throughput"]["prediction_available"])

    def test_large_mbs_is_rejected_before_ranking(self) -> None:
        report = self.predictor.predict(
            [
                {
                    **self.requests[0],
                    "request_id": "safety-v2-reject",
                    "mbs": 16,
                }
            ]
        )
        row = report["predictions"][0]
        self.assertFalse(row["memory"]["admitted"])
        self.assertFalse(row["throughput"]["prediction_available"])
        self.assertEqual(
            report["ranking_groups"][0]["status"],
            "no_memory_safe_candidate",
        )


if __name__ == "__main__":
    unittest.main()
