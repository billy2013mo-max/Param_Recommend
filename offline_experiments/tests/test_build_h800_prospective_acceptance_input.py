from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_h800_prospective_acceptance_input import (  # noqa: E402
    build_acceptance_input,
    _blocked_missing_sources,
)
from common import sha256_json  # noqa: E402
from prospective_acceptance import evaluate_prospective_acceptance  # noqa: E402


def _job(job_id: str, gpu_count: int, mbs: int) -> dict:
    return {
        "job_id": job_id,
        "campaign_id": "fresh-campaign",
        "scenario_id": "scenario-a",
        "candidate_slot_id": f"slot-{job_id}",
        "scale_out_transition": "1_to_2",
        "model_id": "qwen3_8b",
        "dataset_id": "fresh-profile",
        "gpu_count": gpu_count,
        "cutoff_len": 512,
        "mbs": mbs,
        "packing": False,
        "offload": False,
    }


def _prediction(job: dict, point: float) -> dict:
    return {
        "request_id": job["job_id"],
        "comparison_group": "scenario-a",
        "scenario_material": {"scenario_id": "scenario-a", "cutoff_len": 512},
        "runtime_mechanism_component_sha256": "r" * 64,
        "configuration": {"gpu_count": job["gpu_count"], "physical_mbs": job["mbs"]},
        "memory": {
            "admitted": True,
            "admission_upper_reserved_bytes": 200,
            "operational_p95_reserved_bytes": 200,
            "safe_limit_bytes": 200,
        },
        "throughput": {
            "prediction_available": True,
            "throughput_proxy_tokens_per_second": point,
            "conservative_lower_throughput": point * 0.95,
            "conservative_upper_throughput": point * 1.05,
        },
    }


def _observation(job: dict, rate: float) -> dict:
    return {
        "schema": "sft_efficiency_observation/v2",
        "configuration": {"job": job},
        "hardware": {"gpu_type": "NVIDIA H800 140GB HBM3"},
        "quality": {
            "evidence_verified": True,
            "event_attempt_binding_complete": True,
            "terminal_label_verified": True,
        },
        "fingerprint": {"runtime_mechanism_fingerprint_sha256": "f" * 64},
        "outcome": {"class": "success", "terminal_label_verified": True},
        "measurements": {
            "memory": {"max_reserved_bytes": 150},
            "rates": {"effective_tokens_per_second": rate},
        },
    }


class BuildProspectiveAcceptanceInputTests(unittest.TestCase):
    def test_missing_sources_produce_blocked_input_instead_of_fabrication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = _blocked_missing_sources(
                queue_path=root / "queue.json",
                prediction_path=root / "prediction.json",
                observations_path=root / "observations.jsonl",
            )
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["publication_allowed"])
        self.assertIn("source_file_missing:prediction_report", result["blockers"])
        self.assertEqual(result["memory_rows"], [])

    def test_frozen_join_builds_ranking_and_scale_inputs(self) -> None:
        jobs = [
            _job("a-1", 1, 8),
            _job("a-2", 1, 16),
            _job("a-3", 2, 4),
            _job("a-4", 2, 8),
        ]
        predictions = [_prediction(job, 100.0 if job["gpu_count"] == 1 else 200.0) for job in jobs]
        report = {
            "schema": "sft_h800_physical_shares_v4b_prediction/v3",
            "predictions": predictions,
        }
        report["report_sha256"] = sha256_json(report)
        result = build_acceptance_input(
            queue_manifest={
                "schema": "sft_h800_prospective_queue_manifest/v1",
                "campaign_id": "fresh-campaign",
                "gpu_training_started": False,
                "queues_mutated": True,
                "jobs": jobs,
            },
            prediction_report=report,
            observations=[
                _observation(job, 90.0 if job["gpu_count"] == 1 else 190.0)
                for job in jobs
            ],
            scale_evidence={
                "pairs": [
                    {
                        "pair_id": "scenario-a:1_to_2",
                        "measured_ratio_lower": 1.9,
                    }
                ]
            },
        )
        self.assertEqual(result["status"], "ready_for_evaluator")
        self.assertTrue(result["fresh_split"])
        self.assertTrue(result["scenario_level_split"])
        self.assertEqual(len(result["memory_rows"]), 4)
        self.assertEqual(len(result["ranking_groups"]), 1)
        self.assertEqual(len(result["scale_pairs"]), 1)
        self.assertEqual(result["scale_pairs"][0]["measured_ratio_lower"], 1.9)
        evaluated = evaluate_prospective_acceptance(
            memory_rows=result["memory_rows"],
            ranking_groups=result["ranking_groups"],
            scale_pairs=result["scale_pairs"],
            fresh_split=result["fresh_split"],
            scenario_level_split=result["scenario_level_split"],
        )
        self.assertTrue(evaluated["publication_allowed"])
        self.assertTrue(evaluated["scale_out"]["passes"])

    def test_missing_runtime_quality_is_fail_closed(self) -> None:
        job = _job("a-1", 1, 8)
        prediction = _prediction(job, 100.0)
        result = build_acceptance_input(
            queue_manifest={
                "schema": "sft_h800_prospective_queue_manifest/v1",
                "campaign_id": "fresh-campaign",
                "gpu_training_started": False,
                "queues_mutated": False,
                "jobs": [job],
            },
            prediction_report={
                "schema": "sft_h800_physical_shares_v4b_prediction/v3",
                "predictions": [prediction],
            },
            observations=[
                {
                    "schema": "sft_efficiency_observation/v2",
                    "configuration": {"job": job},
                    "outcome": {"class": "success"},
                    "measurements": {"memory": {"max_reserved_bytes": 150}},
                }
            ],
        )
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(any(item.startswith("observation_quality:a-1") for item in result["blockers"]))
        self.assertIn("scale_endpoint_observation_missing:scenario-a:1_to_2", result["blockers"])


if __name__ == "__main__":
    unittest.main()
