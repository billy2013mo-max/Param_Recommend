from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import freeze_recovery  # noqa: E402


class FreezeRecoveryTests(unittest.TestCase):
    def test_recovery_queue_must_be_unique_single_gpu_throughput_subset(self) -> None:
        source = [
            {"job_id": "one", "kind": "throughput", "gpu_count": 1},
            {"job_id": "two", "kind": "throughput", "gpu_count": 2},
        ]
        valid = freeze_recovery.validate_recovery_jobs(source, [source[0]])
        invalid = freeze_recovery.validate_recovery_jobs(source, [source[1], source[1]])
        self.assertTrue(valid["all_passed"])
        self.assertFalse(invalid["all_passed"])
        self.assertEqual(invalid["duplicate_job_ids"], ["two"])
        self.assertEqual(invalid["invalid_non_single_gpu_throughput_job_ids"], ["two", "two"])


if __name__ == "__main__":
    unittest.main()
