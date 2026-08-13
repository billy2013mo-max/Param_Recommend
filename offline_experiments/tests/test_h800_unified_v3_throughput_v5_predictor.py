from __future__ import annotations

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = EXPERIMENT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from h800_resource_predictor import (
    DEFAULT_MEMORY_GATE,
    H800ResourcePredictor,
    validate_prediction_report,
)
from h800_unified_v3_throughput_v5_predictor import (
    ADMISSION_SOURCE,
    SCHEMA,
    THROUGHPUT_MODEL_ID,
)


class H800UnifiedV3ThroughputV5PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        example = (
            EXPERIMENT_ROOT / "examples" / "h800_unified_v3_throughput_v5_request.json"
        )
        cls.requests = json.loads(example.read_text(encoding="utf-8"))["candidates"]

    def test_default_entrypoint_uses_memory_v3_and_throughput_v5(self) -> None:
        self.assertEqual(DEFAULT_MEMORY_GATE, "unified_v3")
        report = H800ResourcePredictor().predict(self.requests)
        validate_prediction_report(report)

        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["release"]["mode"], "active_recommendation")
        self.assertFalse(report["release"]["automatic_execution_allowed"])
        self.assertEqual(
            report["memory_gate"]["id"],
            "unified_v3_shared_center_independent_risk",
        )
        self.assertTrue(
            report["model_artifacts"]["memory"]["path"].endswith(
                "h800_unified_bounded_memory_candidate_v3.json"
            )
        )
        self.assertTrue(
            report["model_artifacts"]["throughput"]["path"].endswith(
                "structured_throughput_modeling.json"
            )
        )
        self.assertNotIn("throughput_legacy_memory_binding", report["model_artifacts"])
        self.assertEqual(report["throughput_model"]["id"], THROUGHPUT_MODEL_ID)
        self.assertFalse(
            report["throughput_model"]["dataset_id_used_as_fitted_identity_feature"]
        )
        self.assertFalse(report["throughput_model"]["dataset_id_used_as_model_feature"])
        self.assertTrue(
            report["throughput_model"]["dataset_id_resolves_static_profile"]
        )
        self.assertTrue(report["single_output_used_for_absolute_and_ranking"])
        self.assertTrue(report["absolute_throughput_scale_trusted"])
        self.assertTrue(
            all(
                row["memory"]["admission_source"] == ADMISSION_SOURCE
                and row["memory"]["anchor_override_applied"] is False
                for row in report["predictions"]
                if row["memory"]["prediction_available"]
            )
        )
        self.assertTrue(
            all(
                group["status"]
                in {"ranked_active_recommendation", "no_admitted_candidate"}
                for group in report["ranking_groups"]
            )
        )
        group = report["ranking_groups"][0]
        self.assertEqual(
            group["ranked_request_ids"],
            [
                "qwen3-14b-full-gpu4-z2-mbs8-gc",
                "qwen3-14b-full-gpu4-z3-mbs8",
                "qwen3-14b-full-gpu4-z2-mbs4-memory-reject",
                "qwen3-14b-full-gpu4-z3-mbs4",
            ],
        )
        self.assertEqual(
            group["selected_request_id"],
            "qwen3-14b-full-gpu4-z2-mbs8-gc",
        )

    def test_unified_output_matches_direct_throughput_v5(self) -> None:
        predictor = H800ResourcePredictor()
        report = predictor.predict(self.requests)
        direct = predictor.predictor.base.predict_many(self.requests)
        direct_by_id = {row["request_id"]: row for row in direct["predictions"]}
        for row in report["predictions"]:
            throughput = row["throughput"]
            self.assertTrue(throughput["prediction_available"])
            source = direct_by_id[row["request_id"]]
            self.assertEqual(
                throughput["predicted_effective_tokens_per_second"],
                source["predicted_effective_tokens_per_second"],
            )
            self.assertEqual(
                throughput["predicted_step_seconds"],
                source["predicted_step_seconds"],
            )

    def test_v3_upper_is_center_risk_max_and_drives_throughput_availability(
        self,
    ) -> None:
        report = H800ResourcePredictor().predict(self.requests)
        for row in report["predictions"]:
            memory = row["memory"]
            expected_upper = max(
                memory["reserved_center_bytes"],
                memory["risk_guard_bytes"] * memory["risk_guard_multiplier"],
            )
            self.assertAlmostEqual(
                memory["admission_upper_reserved_bytes"], expected_upper, places=3
            )
            self.assertEqual(
                memory["admitted"],
                row["throughput"]["prediction_available"],
            )

    def test_runtime_feature_adapter_matches_frozen_v3_canary(self) -> None:
        frozen_path = (
            EXPERIMENT_ROOT
            / "artifacts"
            / "h800_unified_bounded_canary_frozen_predictions_v3.json"
        )
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        expected = next(
            row["v3"]
            for row in frozen["rows"]
            if row["job_id"] == "h800ubc3-b09d1d1caf98822c"
        )
        request = {
            "request_id": "frozen-canary-adapter-check",
            "comparison_group": "frozen-canary-adapter-check",
            "hardware_id": "h800",
            "model_id": "qwen3_8b",
            "training_mode": "lora",
            "lora_rank": 32,
            "dataset_id": "longtail_8192",
            "dataset_category": "longtail",
            "target_gbs": 32,
            "cutoff_len": 4096,
            "gpu_count": 2,
            "physical_mbs": 4,
            "zero_stage": 2,
            "gradient_checkpointing": False,
            "packing": False,
            "dtype": "bf16",
        }
        memory = H800ResourcePredictor().predict([request])["predictions"][0]["memory"]
        pairs = (
            ("analytic_reference_bytes", "reference_bytes"),
            ("reserved_center_bytes", "center_bytes"),
            ("risk_guard_bytes", "risk_guard_bytes"),
            ("admission_upper_reserved_bytes", "admission_upper_bytes"),
        )
        for actual_name, expected_name in pairs:
            self.assertAlmostEqual(
                memory[actual_name], expected[expected_name], places=3
            )

    def test_report_checksum_detects_mutation(self) -> None:
        report = H800ResourcePredictor().predict(self.requests)
        mutated = deepcopy(report)
        mutated["predictions"][0]["memory"]["reserved_center_bytes"] += 1.0
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_prediction_report(mutated)

    def test_explicit_legacy_rollback_remains_available(self) -> None:
        report = H800ResourcePredictor(memory_gate="legacy_physical_v1").predict(
            self.requests
        )
        validate_prediction_report(report)
        self.assertEqual(report["schema"], "sft_h800_physical_shares_v4b_prediction/v3")
        self.assertEqual(report["release"]["mode"], "shadow_only")


if __name__ == "__main__":
    unittest.main()
