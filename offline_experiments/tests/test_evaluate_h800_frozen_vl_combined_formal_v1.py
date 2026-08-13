from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import sha256_file
from evaluate_h800_frozen_vl_combined_formal_v1 import _oom_terminal_semantics


class FrozenVLCombinedFormalOOMTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[dict, dict, Path]:
        job_id = "oom-job"
        attempt_id = "attempt-1"
        attempt = root / job_id / "attempts" / attempt_id
        attempt.mkdir(parents=True)
        log_path = attempt / "train.log"
        log_path.write_text(
            "torch.OutOfMemoryError: CUDA out of memory\n", encoding="utf-8"
        )
        status = {
            "schema": "sft_execution_status/v2",
            "job_id": job_id,
            "classification": "oom",
            "calibration_eligible": False,
            "execution_attempt_id": attempt_id,
            "return_code": 1,
            "classification_evidence": {
                "schema": "sft_terminal_classification/v1",
                "job_id": job_id,
                "execution_attempt_id": attempt_id,
                "classification": "oom",
                "cuda_oom_confirmed": True,
                "matched_cuda_oom_patterns": ["CUDA out of memory"],
                "return_code": 1,
                "summary_set_exact": True,
                "log_path": "train.log",
                "log_sha256": sha256_file(log_path),
            },
        }
        status_path = attempt / "status.json"
        status_path.write_text(json.dumps(status), encoding="utf-8")
        latest = {
            "schema": "sft_latest_attempt/v1",
            "state": "complete",
            "job_id": job_id,
            "classification": "oom",
            "execution_attempt_id": attempt_id,
            "attempt_path": f"attempts/{attempt_id}",
            "calibration_eligible": False,
        }
        (root / job_id / "latest_attempt.json").write_text(
            json.dumps(latest), encoding="utf-8"
        )
        return (
            {"job_id": job_id},
            {
                "classification": "oom",
                "status_path": str(status_path),
                "calibration_eligible": False,
            },
            log_path,
        )

    def test_confirmed_attempt_bound_oom_is_valid_right_censored_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, result, _ = self._fixture(root)
            semantics = _oom_terminal_semantics(job, result, results_root=root)
            self.assertTrue(semantics["required"])
            self.assertTrue(semantics["all_passed"])
            self.assertTrue(semantics["checks"]["log_hash_matches"])

    def test_log_mutation_invalidates_oom_terminal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, result, log_path = self._fixture(root)
            log_path.write_text("mutated\n", encoding="utf-8")
            semantics = _oom_terminal_semantics(job, result, results_root=root)
            self.assertFalse(semantics["all_passed"])
            self.assertFalse(semantics["checks"]["log_hash_matches"])


if __name__ == "__main__":
    unittest.main()
