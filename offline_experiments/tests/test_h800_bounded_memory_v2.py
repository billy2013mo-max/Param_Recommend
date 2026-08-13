from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest
from copy import deepcopy


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import read_json, sha256_json  # noqa: E402
from h800_bounded_memory_model import (  # noqa: E402
    ARTIFACT_SCHEMA,
    bounded_feature_values,
    predict_memory,
    selector_bucket_key,
)
from h800_bounded_v4b_predictor import (  # noqa: E402
    H800BoundedV4BPredictor,
    SCHEMA as PREDICTION_SCHEMA,
)


ARTIFACT = ROOT / "artifacts" / "h800_bounded_memory_challenger_v2.json"
DIAGNOSTIC = (
    ROOT
    / "artifacts"
    / "h800_bounded_memory_v2_consumed_holdout_diagnostic.json"
)


def _record(*, gpu_count: int = 2, mbs: int = 2, zero_stage: int = 2) -> dict:
    return {
        "scenario": {
            "cutoff_len": 4096,
            "physical_mbs": mbs,
            "gpu_count": gpu_count,
        },
        "selector": {
            "training_mode": "lora",
            "zero_stage": zero_stage,
            "gradient_checkpointing": False,
            "packing": False,
        },
    }


def _padding() -> dict:
    return {
        "maximum_clipped_tokens": 4096,
        "p99_clipped_tokens": 4096,
        "expected_random_batch_max_fraction_of_cutoff": 0.8,
        "truncation_fraction": 0.75,
    }


class H800BoundedMemoryV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = read_json(ARTIFACT)
        cls.diagnostic = read_json(DIAGNOSTIC)
        cls.predictor = H800BoundedV4BPredictor()
        example = ROOT / "examples" / "h800_physical_v4b_request.json"
        cls.example_request = json.loads(example.read_text(encoding="utf-8"))[
            "candidates"
        ][0]

    def test_artifact_is_checksum_sealed_and_non_publishable(self) -> None:
        self.assertEqual(self.artifact["schema"], ARTIFACT_SCHEMA)
        unsigned = dict(self.artifact)
        expected = unsigned.pop("report_sha256")
        self.assertEqual(expected, sha256_json(unsigned))
        self.assertFalse(self.artifact["publishable"])
        self.assertFalse(self.artifact["production_override_allowed"])
        self.assertEqual(
            self.artifact["governance"][
                "consumed_holdout_measurement_rows_used_in_numerical_fit"
            ],
            0,
        )
        self.assertTrue(
            self.artifact["governance"]["new_prospective_holdout_required"]
        )

    def test_selected_head_is_bounded_and_low_dimensional(self) -> None:
        model = self.artifact["model"]["direct_reserved_center"]
        self.assertEqual(
            model["candidate_name"],
            "selector_plus_lora_maximum_and_fragmentation",
        )
        self.assertEqual(len(model["feature_names"]), 7)
        self.assertFalse(
            any("squared" in name or "cubed" in name for name in model["feature_names"])
        )
        stress = self.artifact["selection"]["direct_reserved_center"][
            "selected"
        ]["stress_test"]
        self.assertTrue(stress["passed"])
        self.assertGreaterEqual(stress["observed_factor_bounds"]["minimum"], 0.20)
        self.assertLessEqual(stress["observed_factor_bounds"]["maximum"], 3.00)

    def test_shape_features_are_bounded_and_truncation_reduces_fragmentation(self) -> None:
        record = _record()
        bucket = selector_bucket_key(record)
        values = bounded_feature_values(
            record, _padding(), supported_selector_keys=[bucket]
        )
        self.assertEqual(values["lora_maximum_fraction"], 1.0)
        self.assertEqual(values["lora_p99_fraction"], 1.0)
        self.assertAlmostEqual(values["lora_fragmentation_pressure"], 0.2)
        self.assertAlmostEqual(values["risk_fragmentation_pressure"], 0.2)
        self.assertAlmostEqual(
            values["risk_fragmentation_pressure_x_log2_mbs"], 0.2
        )
        self.assertTrue(all(math.isfinite(value) for value in values.values()))

    def test_unknown_selector_fails_closed(self) -> None:
        prediction = predict_memory(
            _record(gpu_count=4, mbs=1, zero_stage=2),
            _padding(),
            allocated_anchor_bytes=50 * 1024**3,
            artifact=self.artifact,
        )
        self.assertFalse(prediction["available"])
        self.assertIn("unseen_selector_bucket", prediction["issues"])

    def test_calibration_and_consumed_holdout_diagnostics_are_explicit(self) -> None:
        fixed = self.artifact["evaluation"][
            "fixed_selected_reserved_leave_profile_out"
        ]
        nested = self.artifact["evaluation"][
            "nested_reserved_model_selection_leave_profile_out"
        ]
        guarded = self.artifact["evaluation"][
            "guarded_fixed_reserved_leave_profile_out"
        ]
        self.assertLess(fixed["scenario_equal_mape"], 0.08)
        self.assertLess(nested["scenario_equal_mape"], 0.10)
        self.assertEqual(guarded["row_coverage"], 1.0)

        self.assertEqual(
            self.diagnostic["status"],
            "diagnostic_only_consumed_holdout_not_release_evidence",
        )
        self.assertFalse(
            self.diagnostic["governance"][
                "diagnostic_may_support_generalization_claim"
            ]
        )
        v2 = self.diagnostic["v2_post_holdout_repair"]
        self.assertLess(v2["center_scenario_equal_mape"], 0.10)
        self.assertEqual(v2["upper_row_coverage"], 1.0)
        self.assertEqual(v2["false_safe_count"], 0)
        self.assertEqual(v2["false_reject_count"], 0)

    def test_shadow_wrapper_admits_14b_two_gpu_without_anchor_override(self) -> None:
        request = deepcopy(self.example_request)
        request.update(
            {
                "request_id": "bounded-v2-14b-full-2gpu",
                "comparison_group": "bounded-v2-14b-full-2gpu",
                "model_id": "qwen3_14b",
                "training_mode": "full",
                "dataset_id": "multiturn_4096",
                "dataset_category": "multiturn",
                "target_gbs": 64,
                "cutoff_len": 4096,
                "gpu_count": 2,
                "physical_mbs": 2,
                "gradient_accumulation_steps": 16,
                "zero_stage": 3,
                "gradient_checkpointing": True,
                "packing": False,
            }
        )
        report = self.predictor.predict([request])
        row = report["predictions"][0]
        self.assertEqual(report["schema"], PREDICTION_SCHEMA)
        self.assertEqual(report["release"]["mode"], "shadow_only")
        self.assertFalse(report["release"]["automatic_execution_allowed"])
        self.assertTrue(row["memory"]["admitted"])
        self.assertFalse(row["memory"]["anchor_override_applied"])
        self.assertLess(
            row["memory"]["operational_p95_reserved_bytes"],
            row["memory"]["safe_limit_bytes"],
        )
        self.assertTrue(row["throughput"]["prediction_available"])

    def test_shadow_wrapper_rejects_unseen_four_gpu_bucket(self) -> None:
        request = deepcopy(self.example_request)
        request.update(
            {
                "request_id": "bounded-v2-8b-lora-4gpu",
                "comparison_group": "bounded-v2-8b-lora-4gpu",
                "model_id": "qwen3_8b",
                "training_mode": "lora",
                "dataset_id": "multiturn_4096",
                "dataset_category": "multiturn",
                "target_gbs": 64,
                "cutoff_len": 4096,
                "gpu_count": 4,
                "physical_mbs": 2,
                "gradient_accumulation_steps": 8,
                "zero_stage": 2,
                "gradient_checkpointing": False,
                "packing": False,
            }
        )
        report = self.predictor.predict([request])
        row = report["predictions"][0]
        self.assertFalse(row["memory"]["prediction_available"])
        self.assertEqual(
            row["memory"]["rejection_reason"], "memory_prediction_unavailable"
        )
        self.assertIn("unseen_selector_bucket", row["memory"]["issues"])
        self.assertFalse(row["throughput"]["prediction_available"])


if __name__ == "__main__":
    unittest.main()
