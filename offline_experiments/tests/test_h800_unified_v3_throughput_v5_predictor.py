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
    HYBRID_VL_V2_RELEASE_MODE,
    SCHEMA,
    THROUGHPUT_MODEL_ID,
)
from prepare_h800_hybrid_vl_prospective_acceptance_v1 import (
    _hybrid_request as prospective_hybrid_request,
)
from prepare_h800_hybrid_vl_prospective_acceptance_v1 import (
    _vl_request as prospective_vl_request,
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

    @staticmethod
    def _vl_request(model_id: str, profile: Path) -> dict[str, object]:
        return {
            "request_id": f"{model_id}-vl-shadow",
            "comparison_group": f"{model_id}-vl-shadow",
            "hardware_id": "h800",
            "model_id": model_id,
            "training_mode": "lora",
            "lora_rank": 32,
            "target_gbs": 64,
            "cutoff_len": 8192,
            "gpu_count": 1,
            "physical_mbs": 1,
            "zero_stage": 0,
            "gradient_checkpointing": True,
            "packing": False,
            "dtype": "bf16",
            "freeze_vision_tower": True,
            "freeze_multi_modal_projector": True,
            "vl_workload_profile_path": str(profile.resolve()),
        }

    def test_vl_overlay_is_integrated_for_all_fitted_models_but_stays_shadow_only(
        self,
    ) -> None:
        profile_dir = (
            EXPERIMENT_ROOT / "artifacts" / "h800_vl_business_workload_profiles_v2"
        )
        requests = [
            self._vl_request(model_id, profile_dir / f"{model_id}.pzfj38.low.json")
            for model_id in ("qwen2p5_vl_3b", "qwen3_vl_4b", "qwen3p5_4b")
        ]
        report = H800ResourcePredictor().predict(requests)
        validate_prediction_report(report)

        self.assertEqual(report["vl_overlay"]["mode"], "shadow_only")
        self.assertFalse(report["vl_overlay"]["automatic_admission_allowed"])
        self.assertFalse(report["vl_overlay"]["automatic_ranking_allowed"])
        self.assertEqual(
            report["vl_overlay"]["model_ids"],
            ["qwen2p5_vl_3b", "qwen3_vl_4b", "qwen3p5_4b"],
        )
        for row in report["predictions"]:
            shadow = row["vl_shadow"]
            self.assertEqual(row["support"]["label"], "unsupported")
            self.assertFalse(row["memory"]["admitted"])
            self.assertFalse(row["throughput"]["prediction_available"])
            self.assertEqual(shadow["mode"], "shadow_only")
            self.assertGreaterEqual(
                shadow["memory"]["safety_upper_bytes"],
                shadow["memory"]["predicted_center_bytes"],
            )
            self.assertGreater(
                shadow["memory"]["v1_overlay_center_bytes"],
                shadow["memory"]["text_center_bytes"],
            )
            self.assertGreater(shadow["memory"]["total_center_scale"], 0.0)
            self.assertEqual(
                shadow["recommendation_status"], HYBRID_VL_V2_RELEASE_MODE
            )
            self.assertGreater(shadow["throughput"]["predicted_step_seconds"], 0.0)
            self.assertGreater(
                shadow["throughput"]["effective_tokens_per_second"], 0.0
            )
        self.assertTrue(
            all(group["selected_request_id"] is None for group in report["ranking_groups"])
        )

    def test_vl_overlay_rejects_packing_outside_fitted_scope(self) -> None:
        profile = (
            EXPERIMENT_ROOT
            / "artifacts"
            / "h800_vl_business_workload_profiles_v2"
            / "qwen3p5_4b.pzfj38.low.json"
        )
        request = self._vl_request("qwen3p5_4b", profile)
        request["packing"] = True
        with self.assertRaisesRegex(ValueError, "does not support packing"):
            H800ResourcePredictor().predict([request])

    def test_pure_text_packing_is_a_separate_candidate_path(self) -> None:
        request = deepcopy(self.requests[0])
        request["request_id"] = "pure-text-packing-candidate"
        request["comparison_group"] = "pure-text-packing-candidate"
        request["packing"] = True
        report = H800ResourcePredictor().predict([request])
        validate_prediction_report(report)
        row = report["predictions"][0]

        self.assertTrue(row["configuration"]["packing"])
        self.assertIsNone(row.get("vl_shadow"))
        self.assertEqual(row["support"]["label"], "caution")
        self.assertIn(
            "packing_limited_evidence",
            {reason["code"] for reason in row["support"]["reasons"]},
        )
        self.assertTrue(row["memory"]["prediction_available"])
        self.assertTrue(row["memory"]["admitted"])
        self.assertTrue(row["throughput"]["prediction_available"])
        self.assertEqual(
            report["ranking_groups"][0]["selected_request_id"],
            "pure-text-packing-candidate",
        )
        self.assertFalse(report["vl_overlay"]["packing_allowed"])
        self.assertTrue(
            report["pure_text_packing"]["candidate_prediction_available"]
        )
        self.assertTrue(
            report["pure_text_packing"]["candidate_ranking_available"]
        )
        self.assertTrue(
            report["pure_text_packing"]["automatic_execution_allowed"]
        )
        # Packing fixes physical MBS to one.  This inherited MBS=8 request can
        # still be diagnosed, but must remain outside automatic execution.
        self.assertIn(
            "physical_mbs",
            row["support"]["packing_production_admission"]["mismatches"],
        )
        self.assertFalse(row["support"]["automatic_execution_allowed"])
        self.assertFalse(
            report["ranking_groups"][0]["automatic_execution_allowed"]
        )
        self.assertFalse(report["release"]["automatic_execution_allowed"])

    def test_explicit_full_qwen4_packing_request_is_in_limited_production(self) -> None:
        request = {
            "request_id": "qwen4-full-packing-production",
            "comparison_group": "qwen4-full-packing-production",
            "hardware_id": "h800",
            "model_id": "qwen3_4b",
            "training_mode": "full",
            "lora_rank": 32,
            "dataset_id": "multiturn_4096",
            "dataset_category": "multiturn",
            "target_gbs": 64,
            "cutoff_len": 4096,
            "gpu_count": 2,
            "physical_mbs": 1,
            "zero_stage": 2,
            "gradient_checkpointing": True,
            "packing": True,
            "offload": False,
            "dtype": "bf16",
            "kernel_path": "fa3_orig+liger_fused_ce+adamw_torch_fused",
        }
        report = H800ResourcePredictor().predict([request])
        validate_prediction_report(report)
        row = report["predictions"][0]

        self.assertEqual(
            row["support"]["packing_production_admission"]["admission_mode"],
            "explicit_packing_request",
        )
        self.assertTrue(row["support"]["automatic_execution_allowed"])
        self.assertTrue(row["memory"]["admitted"])
        self.assertTrue(row["throughput"]["prediction_available"])
        self.assertTrue(report["ranking_groups"][0]["automatic_execution_allowed"])
        self.assertTrue(report["release"]["automatic_execution_allowed"])

    def test_packing_release_does_not_bypass_memory_upper(self) -> None:
        request = {
            "request_id": "qwen32-full-packing-memory-reject",
            "comparison_group": "qwen32-full-packing-memory-reject",
            "hardware_id": "h800",
            "model_id": "qwen3_32b",
            "training_mode": "full",
            "lora_rank": 32,
            "dataset_id": "longcontext_16384",
            "dataset_category": "longcontext",
            "target_gbs": 128,
            "cutoff_len": 16384,
            "gpu_count": 1,
            "physical_mbs": 1,
            "zero_stage": 0,
            "gradient_checkpointing": False,
            "packing": True,
            "offload": False,
            "dtype": "bf16",
            "kernel_path": "fa3_orig+liger_fused_ce+adamw_torch_fused",
        }
        report = H800ResourcePredictor().predict([request])
        validate_prediction_report(report)
        row = report["predictions"][0]

        self.assertTrue(row["support"]["automatic_execution_allowed"])
        self.assertFalse(row["memory"]["admitted"])
        self.assertEqual(
            row["memory"]["rejection_reason"],
            "memory_upper_exceeds_safe_limit",
        )
        self.assertFalse(report["ranking_groups"][0]["automatic_execution_allowed"])
        self.assertFalse(report["release"]["automatic_execution_allowed"])

    def test_hybrid_memory_artifact_is_integrated_but_cannot_admit(self) -> None:
        requests = []
        for model_id in ("qwen3p5_4b", "qwen3p5_9b", "qwen3_6_27b"):
            requests.append(
                {
                    "request_id": f"{model_id}-hybrid-shadow",
                    "comparison_group": f"{model_id}-hybrid-shadow",
                    "hardware_id": "h800",
                    "model_id": model_id,
                    "training_mode": "lora",
                    "lora_rank": 32,
                    "dataset_id": "longtail_8192",
                    "dataset_category": "longtail",
                    "target_gbs": 64,
                    "cutoff_len": 8192,
                    "gpu_count": 1,
                    "physical_mbs": 1,
                    "zero_stage": 0,
                    "gradient_checkpointing": True,
                    "packing": False,
                    "dtype": "bf16",
                }
            )
        report = H800ResourcePredictor().predict(requests)
        validate_prediction_report(report)

        self.assertEqual(report["hybrid_memory"]["mode"], "shadow_only")
        self.assertEqual(
            report["hybrid_memory"]["model_ids"],
            ["qwen3_6_27b", "qwen3p5_4b", "qwen3p5_9b"],
        )
        for row in report["predictions"]:
            shadow = row["hybrid_shadow"]
            self.assertGreater(shadow["predicted_center_bytes"], 0.0)
            self.assertGreaterEqual(
                shadow["safety_upper_bytes"], shadow["predicted_center_bytes"]
            )
            self.assertTrue(row["memory"]["safety_upper_calibrated"])
            self.assertFalse(row["memory"]["admitted"])
            self.assertFalse(row["throughput"]["prediction_available"])

    def test_v2_runtime_matches_all_27_development_replay_rows(self) -> None:
        artifact_path = (
            EXPERIMENT_ROOT / "artifacts" / "h800_hybrid_vl_safety_upper_v2.json"
        )
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        queue_path = (
            EXPERIMENT_ROOT
            / "matrix"
            / "h800_hybrid_vl_prospective_acceptance_v1.jsonl"
        )
        jobs = [
            json.loads(line)
            for line in queue_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        requests = [
            prospective_hybrid_request(job)
            if job["design_arm"] == "hybrid_memory_prospective"
            else prospective_vl_request(job)
            for job in jobs
        ]
        report = H800ResourcePredictor().predict(requests)
        validate_prediction_report(report)
        actual_by_id = {row["request_id"]: row for row in report["predictions"]}
        expected_rows = [
            *artifact["hybrid_memory"]["development_replay"]["rows"],
            *artifact["vl_memory_by_modality"]["image"]["development_replay"][
                "rows"
            ],
        ]

        self.assertEqual(len(expected_rows), 27)
        self.assertEqual(set(actual_by_id), {row["job_id"] for row in expected_rows})
        for expected in expected_rows:
            output = actual_by_id[expected["job_id"]]
            shadow_memory = (
                output["hybrid_shadow"]
                if expected["track"] == "hybrid"
                else output["vl_shadow"]["memory"]
            )
            self.assertAlmostEqual(
                shadow_memory["predicted_center_bytes"],
                expected["center_bytes_v2"],
                places=3,
            )
            self.assertAlmostEqual(
                shadow_memory["safety_upper_bytes"],
                expected["upper_bytes_v2"],
                places=3,
            )

        jobs_by_id = {job["job_id"]: job for job in jobs}
        qwen35_pressure = [
            actual_by_id[job_id]
            for job_id, job in jobs_by_id.items()
            if job.get("model_id") == "qwen3p5_4b"
            and job.get("mechanism_id") == "PRESSURE"
        ]
        self.assertEqual(len(qwen35_pressure), 2)
        self.assertTrue(
            all(
                row["vl_shadow"]["memory"]["candidate_admitted_by_upper"]
                for row in qwen35_pressure
            )
        )
        self.assertTrue(
            all(
                row["vl_shadow"]["memory"]["legacy_v1_base_guard_upper_bytes"]
                > row["vl_shadow"]["memory"]["safety_upper_bytes"]
                for row in qwen35_pressure
            )
        )
        qwen3_pressure_high = next(
            actual_by_id[job_id]
            for job_id, job in jobs_by_id.items()
            if job.get("model_id") == "qwen3_vl_4b"
            and job.get("mechanism_id") == "PRESSURE"
            and job.get("media_tier") == "high"
        )
        self.assertFalse(
            qwen3_pressure_high["vl_shadow"]["memory"][
                "candidate_admitted_by_upper"
            ]
        )


if __name__ == "__main__":
    unittest.main()
