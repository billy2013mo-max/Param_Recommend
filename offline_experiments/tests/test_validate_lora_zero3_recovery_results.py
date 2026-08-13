from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import sha256_json  # noqa: E402
from validate_lora_zero3_recovery_results import (  # noqa: E402
    FP32_LORA_MESSAGE,
    expected_rendered_shape,
    validate_boundary,
    validate_runtime_chain,
    validate_trial,
)


class ValidateLoraZero3RecoveryResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.family = {
            "job_id": "mem-example",
            "kind": "memory_boundary",
            "model_id": "qwen3_8b",
            "dataset_id": "short_512",
            "cutoff_len": 512,
            "train_type": "lora",
            "zero": "zero3",
            "gc": False,
            "gpu_count": 2,
            "target_gbs": 64,
            "packing": False,
            "mbs_candidates": [1, 2, 4, 8],
        }

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def create_base_trial(
        self,
        results_dir: Path,
        classification: str,
    ) -> tuple[Path, dict[str, object], str]:
        trial_job_id = "mem-example-mbs1"
        result_dir = results_dir / trial_job_id
        started = time.time() - 10
        runtime_identity = {"schema_version": 1, "python_executable": "/venv/bin/python"}
        runtime_fingerprint = sha256_json(runtime_identity)
        status = {
            "job_id": trial_job_id,
            "classification": classification,
            "return_code": 0 if classification == "success" else 1,
            "started_unix": started,
            "finished_unix": started + 5,
            "wall_seconds": 5,
            "gpu_mask": "1,2",
            "approval_design_sha256": "approval",
            "provenance_sha256": "provenance",
            "runtime_fingerprint_sha256": runtime_fingerprint,
        }
        rendered = {
            "job": expected_rendered_shape(self.family, trial_job_id, 1),
            "gpu_mask": "1,2",
            "provenance_sha256": "provenance",
            "runtime_fingerprint_sha256": runtime_fingerprint,
        }
        self.write_json(result_dir / "status.json", status)
        self.write_json(result_dir / "rendered_run.json", rendered)
        self.write_json(result_dir / "runtime_identity.json", runtime_identity)
        return result_dir, runtime_identity, runtime_fingerprint

    def validate_created_trial(
        self,
        results_dir: Path,
        classification: str,
        runtime_identity: dict[str, object],
        runtime_fingerprint: str,
    ) -> dict[str, object]:
        return validate_trial(
            self.family,
            {"mbs": 1, "classification": classification},
            summary_gpu_mask=[1, 2],
            authorized_gpu_ids={1, 2, 3, 4},
            expected_approval_sha256="approval",
            expected_runtime_fingerprint=runtime_fingerprint,
            expected_runtime_identity=runtime_identity,
            expected_provenance_sha256="provenance",
            results_dir=results_dir,
        )

    def test_accepts_contiguous_boundary_stopped_at_first_oom(self) -> None:
        summary = {
            "family_job_id": "mem-example",
            "trials": [
                {"mbs": 1, "classification": "success"},
                {"mbs": 2, "classification": "success"},
                {"mbs": 4, "classification": "oom"},
            ],
            "max_feasible_mbs": 2,
            "first_failed_mbs": 4,
        }
        self.assertTrue(all(validate_boundary(self.family, summary).values()))

    def test_accepts_exhausted_candidates_without_oom(self) -> None:
        summary = {
            "family_job_id": "mem-example",
            "trials": [
                {"mbs": 1, "classification": "success"},
                {"mbs": 2, "classification": "success"},
                {"mbs": 4, "classification": "success"},
                {"mbs": 8, "classification": "success"},
            ],
            "max_feasible_mbs": 8,
            "first_failed_mbs": None,
        }
        self.assertTrue(all(validate_boundary(self.family, summary).values()))

    def test_rejects_skipped_candidate_and_non_oom_failure(self) -> None:
        summary = {
            "family_job_id": "mem-example",
            "trials": [
                {"mbs": 1, "classification": "success"},
                {"mbs": 4, "classification": "failed"},
            ],
            "max_feasible_mbs": 1,
            "first_failed_mbs": 4,
        }
        checks = validate_boundary(self.family, summary)
        self.assertFalse(checks["classifications_are_success_or_oom"])
        self.assertFalse(checks["candidate_prefix_is_contiguous"])

    def test_runtime_fingerprint_is_recomputed_and_unique_across_canaries(self) -> None:
        identity = {"schema_version": 1, "packages": {"deepspeed": "0.19.2"}}
        fingerprint = sha256_json(identity)
        canary = {
            "runtime_identity": identity,
            "results": {
                "rows": [
                    {"runtime_fingerprint_sha256": fingerprint},
                    {"runtime_fingerprint_sha256": fingerprint},
                    {"runtime_fingerprint_sha256": fingerprint},
                ]
            },
        }
        _, computed, checks = validate_runtime_chain(canary)
        self.assertEqual(computed, fingerprint)
        self.assertTrue(all(checks.values()))

        canary["results"]["rows"][2]["runtime_fingerprint_sha256"] = "forged"
        self.assertFalse(validate_runtime_chain(canary)[2]["canary_fingerprints_unique"])

    def test_oom_requires_real_oom_log_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            result_dir, identity, fingerprint = self.create_base_trial(results_dir, "oom")
            (result_dir / "train.log").write_text("unrelated runtime failure\n", encoding="utf-8")
            invalid = self.validate_created_trial(results_dir, "oom", identity, fingerprint)
            self.assertFalse(invalid["checks"]["oom_evidence_present"])
            self.assertFalse(invalid["all_passed"])

            (result_dir / "train.log").write_text(
                "torch.cuda.OutOfMemoryError: CUDA out of memory\n",
                encoding="utf-8",
            )
            valid = self.validate_created_trial(results_dir, "oom", identity, fingerprint)
            self.assertTrue(valid["all_passed"])

            status_path = result_dir / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["job_id"] = "copied-from-another-job"
            self.write_json(status_path, status)
            wrong_identity = self.validate_created_trial(results_dir, "oom", identity, fingerprint)
            self.assertFalse(wrong_identity["checks"]["status_job_id_matches"])

    def test_success_binds_exact_ranks_steps_metadata_and_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            result_dir, identity, fingerprint = self.create_base_trial(results_dir, "success")
            (result_dir / "train.log").write_text(FP32_LORA_MESSAGE + "\n", encoding="utf-8")
            metadata = {
                **{field: self.family[field] for field in (
                    "model_id",
                    "dataset_id",
                    "cutoff_len",
                    "train_type",
                    "zero",
                    "gc",
                    "gpu_count",
                    "target_gbs",
                    "packing",
                )},
                "job_id": "mem-example-mbs1",
                "family_job_id": "mem-example",
                "mbs": 1,
            }
            for rank in (0, 1):
                self.write_json(
                    result_dir / f"metrics/summary.rank{rank}.json",
                    {
                        "rank": rank,
                        "local_rank": rank,
                        "world_size": 2,
                        "failure": None,
                        "total_steps": 5,
                        "measured_steps": 5,
                        "computed_tokens_per_second": 1.0,
                        "logical_samples_per_second": 1.0,
                        "max_allocated": 1,
                        "max_reserved": 2,
                        "metadata": metadata,
                    },
                )
            self.write_json(
                result_dir / "trainer_output/train_results.json",
                {"train_loss": 1.25},
            )
            valid = self.validate_created_trial(results_dir, "success", identity, fingerprint)
            self.assertTrue(valid["all_passed"])

            summary_path = result_dir / "metrics/summary.rank1.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["metadata"]["job_id"] = "stale-job"
            self.write_json(summary_path, summary)
            stale = self.validate_created_trial(results_dir, "success", identity, fingerprint)
            self.assertFalse(stale["checks"]["metric_metadata_matches_job"])


if __name__ == "__main__":
    unittest.main()
