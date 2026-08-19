"""Regression tests for the 0105 business blind-test tooling."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import freeze_h800_business_blind_v1 as freeze
import prepare_h800_business_blind_v1 as prepare


class BusinessBlindPreparationTests(unittest.TestCase):
    def test_job_ids_are_isolated_by_data_source(self) -> None:
        prepare._build_matrix()
        with patch.object(prepare, "DATASET_ID", "business_0105_datatest"):
            datatest_ids = {row["job_id"] for row in prepare._build_jobs()}
        with patch.object(prepare, "DATASET_ID", "business_0105_inference"):
            inference_ids = {row["job_id"] for row in prepare._build_jobs()}

        self.assertEqual(len(datatest_ids), 46)
        self.assertEqual(len(inference_ids), 46)
        self.assertTrue(datatest_ids.isdisjoint(inference_ids))

    def test_freeze_matches_global_approval_scope(self) -> None:
        self.assertEqual(freeze.MAX_GPU_COUNT, 4)

    def test_each_job_uses_at_most_two_gpus(self) -> None:
        prepare._build_matrix()
        jobs = prepare._build_jobs()
        self.assertEqual(max(row["gpu_count"] for row in jobs), 2)


if __name__ == "__main__":
    unittest.main()
