from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import freeze_lora_zero3_fix  # noqa: E402


class FreezeLoraZero3FixTests(unittest.TestCase):
    def test_exact_canary_matrix_is_accepted(self) -> None:
        jobs = [
            {
                "job_id": job_id,
                "kind": "runtime_canary",
                "model_id": model_id,
                "gpu_count": gpu_count,
                "train_type": "lora",
                "zero": "zero3",
                "dataset_id": "short_512",
                "cutoff_len": 512,
                "mbs": 1,
                "target_gbs": 16,
                "gc": False,
                "warmup_steps": 0,
                "measure_steps": 3,
            }
            for job_id, model_id, gpu_count in (
                ("canary-z3fix-qwen3-8b-2g-20260721", "qwen3_8b", 2),
                ("canary-z3fix-qwen3-8b-4g-20260721", "qwen3_8b", 4),
                ("canary-z3fix-qwen3-14b-4g-20260721", "qwen3_14b", 4),
            )
        ]

        result = freeze_lora_zero3_fix.validate_jobs(jobs)

        self.assertEqual(result["duplicate_job_ids"], [])
        self.assertEqual(result["missing_job_ids"], [])
        self.assertEqual(result["unexpected_job_ids"], [])
        self.assertEqual(result["shape_errors"], [])

    def test_changed_precision_or_strategy_is_rejected(self) -> None:
        jobs = [
            {
                "job_id": "canary-z3fix-qwen3-8b-2g-20260721",
                "kind": "runtime_canary",
                "model_id": "qwen3_8b",
                "gpu_count": 2,
                "train_type": "lora",
                "zero": "zero2",
                "dataset_id": "short_512",
                "cutoff_len": 512,
                "mbs": 1,
                "target_gbs": 16,
                "gc": False,
                "warmup_steps": 0,
                "measure_steps": 3,
            }
        ]

        result = freeze_lora_zero3_fix.validate_jobs(jobs)

        self.assertEqual(result["shape_errors"], ["canary-z3fix-qwen3-8b-2g-20260721"])


if __name__ == "__main__":
    unittest.main()
