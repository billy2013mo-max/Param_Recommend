from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import materialize_jobs  # noqa: E402


class ThroughputMaterializationTests(unittest.TestCase):
    def test_screen_candidates_are_bounded_but_keep_gpu_and_strategy_contrasts(self) -> None:
        request = {
            "request_id": "request-1",
            "model_id": "model-1",
            "train_type": "lora",
            "dataset_id": "dataset-1",
            "cutoff_len": 4096,
            "gpu_count": 1,
            "zero": "none",
            "target_gbs": 16,
            "repeats": 2,
            "mbs_points_per_strategy": 2,
            "screen_max_candidates": 4,
        }
        models = {
            "model-1": {
                "id": "model-1",
                "path": "/models/model-1",
                "tokenizer_path": "/models/model-1",
                "family": "test",
                "actual_parameters": 1,
                "template": "test",
            }
        }
        families = {
            "family-a": {
                **request,
                "job_id": "family-a",
                "gpu_count": 1,
                "zero": "none",
                "gc": False,
            },
            "family-b": {
                **request,
                "job_id": "family-b",
                "gpu_count": 1,
                "zero": "none",
                "gc": True,
            },
            "family-c": {
                **request,
                "job_id": "family-c",
                "gpu_count": 2,
                "zero": "zero2",
                "gc": False,
            },
        }
        summaries = {
            "family-a": {"max_feasible_mbs": 8},
            "family-b": {"max_feasible_mbs": 4},
            "family-c": {"max_feasible_mbs": 4},
        }

        with tempfile.TemporaryDirectory() as directory:
            matrix_dir = Path(directory)
            (matrix_dir / "throughput_requests.jsonl").write_text(json.dumps(request) + "\n")
            with mock.patch.object(materialize_jobs, "MATRIX_DIR", matrix_dir):
                jobs, skipped = materialize_jobs.materialize_throughput_screen(
                    models, families, summaries
                )

        self.assertEqual(skipped, [])
        self.assertEqual(len(jobs), 4)
        configurations = {
            (job["gpu_count"], job["zero"], job["gc"], job["mbs"])
            for job in jobs
        }
        self.assertEqual(
            configurations,
            {
                (1, "none", False, 4),
                (1, "none", False, 8),
                (1, "none", True, 4),
                (2, "zero2", False, 4),
            },
        )
        self.assertEqual({job["repeat"] for job in jobs}, {0})
        self.assertEqual({job["kind"] for job in jobs}, {"throughput_screen"})
        self.assertEqual({job["fidelity"] for job in jobs}, {"screen"})
        self.assertEqual({job["warmup_steps"] for job in jobs}, {10})
        self.assertEqual({job["measure_steps"] for job in jobs}, {20})
        self.assertEqual(len({job["job_id"] for job in jobs}), len(jobs))

    def test_formal_jobs_only_include_completed_screening_shortlist(self) -> None:
        request = {
            "request_id": "request-1",
            "model_id": "model-1",
            "train_type": "lora",
            "dataset_id": "dataset-1",
            "cutoff_len": 4096,
            "target_gbs": 16,
            "repeats": 2,
            "mbs_points_per_strategy": 2,
        }
        models = {
            "model-1": {
                "id": "model-1",
                "path": "/models/model-1",
                "tokenizer_path": "/models/model-1",
                "family": "test",
                "actual_parameters": 1,
                "template": "test",
            }
        }
        families = {
            "family-a": {
                **request,
                "job_id": "family-a",
                "gpu_count": 1,
                "zero": "none",
                "gc": False,
            }
        }
        summaries = {"family-a": {"max_feasible_mbs": 8}}
        shortlist = {
            **request,
            "gpu_count": 1,
            "zero": "none",
            "gc": False,
            "mbs": 8,
            "packing": False,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_dir = root / "matrix"
            artifact_dir = root / "artifacts"
            matrix_dir.mkdir()
            artifact_dir.mkdir()
            (matrix_dir / "throughput_requests.jsonl").write_text(json.dumps(request) + "\n")
            (artifact_dir / "stage_decisions.json").write_text(
                json.dumps(
                    {
                        "throughput_screening": {
                            "decisions": [
                                {
                                    "request_id": "request-1",
                                    "status": "shortlisted",
                                    "shortlisted": [shortlist],
                                }
                            ]
                        }
                    }
                )
            )
            with (
                mock.patch.object(materialize_jobs, "MATRIX_DIR", matrix_dir),
                mock.patch.object(materialize_jobs, "ARTIFACT_DIR", artifact_dir),
            ):
                jobs, skipped = materialize_jobs.materialize_throughput(
                    models, families, summaries
                )

        self.assertEqual(skipped, [])
        self.assertEqual(len(jobs), 2)
        self.assertEqual({job["mbs"] for job in jobs}, {8})
        self.assertEqual({job["repeat"] for job in jobs}, {0, 1})
        self.assertEqual({job["fidelity"] for job in jobs}, {"formal"})
        self.assertEqual({job["warmup_steps"] for job in jobs}, {20})
        self.assertEqual({job["measure_steps"] for job in jobs}, {100})

    def test_scaling_uses_one_preferred_strategy_per_gpu_count(self) -> None:
        request = {
            "request_id": "scaling-1",
            "model_id": "model-1",
            "train_type": "lora",
            "dataset_id": "dataset-1",
            "cutoff_len": 4096,
            "target_gbs": 16,
            "gpu_sequence": [1, 2],
            "repeats": 1,
            "warmup_steps": 2,
            "measure_steps": 8,
        }
        models = {
            "model-1": {
                "id": "model-1",
                "path": "/models/model-1",
                "tokenizer_path": "/models/model-1",
                "family": "test",
                "actual_parameters": 1,
                "template": "test",
            }
        }
        families = {
            "one-fast": {
                **request, "job_id": "one-fast", "gpu_count": 1,
                "zero": "none", "gc": False,
            },
            "one-gc": {
                **request, "job_id": "one-gc", "gpu_count": 1,
                "zero": "none", "gc": True,
            },
            "two-zero2": {
                **request, "job_id": "two-zero2", "gpu_count": 2,
                "zero": "zero2", "gc": False,
            },
            "two-zero3": {
                **request, "job_id": "two-zero3", "gpu_count": 2,
                "zero": "zero3", "gc": False,
            },
        }
        summaries = {
            key: {"max_feasible_mbs": 4}
            for key in families
        }

        with tempfile.TemporaryDirectory() as directory:
            matrix_dir = Path(directory)
            (matrix_dir / "strong_scaling_requests.jsonl").write_text(
                json.dumps(request) + "\n"
            )
            with mock.patch.object(materialize_jobs, "MATRIX_DIR", matrix_dir):
                jobs, skipped = materialize_jobs.materialize_scaling(
                    models, families, summaries
                )

        self.assertEqual(skipped, [])
        self.assertEqual(len(jobs), 2)
        self.assertEqual(
            {(job["gpu_count"], job["zero"], job["gc"]) for job in jobs},
            {(1, "none", False), (2, "zero2", False)},
        )
        self.assertEqual({job["warmup_steps"] for job in jobs}, {2})
        self.assertEqual({job["measure_steps"] for job in jobs}, {8})


if __name__ == "__main__":
    unittest.main()
