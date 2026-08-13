from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fit_h800_m1_separate_admission_v4 import (  # noqa: E402
    _admission_features,
    _fit_logistic,
    _predict_risk,
    _predict_separate_admission,
)


class SeparateAdmissionV4Tests(unittest.TestCase):
    def test_features_are_normalized_center_pressures(self) -> None:
        values = _admission_features(
            allocated_center=25.0,
            reserved_center=50.0,
            safe_limit=100.0,
        )
        self.assertAlmostEqual(values[0], -0.6931471805599453)
        self.assertAlmostEqual(values[1], -1.3862943611198906)

    def test_source_balanced_logistic_orders_separated_risks(self) -> None:
        samples = [
            {
                "source_id": "safe-a",
                "unsafe_label": 0,
                "features": [-1.2, -1.5],
            },
            {
                "source_id": "safe-b",
                "unsafe_label": 0,
                "features": [-0.9, -1.3],
            },
            {
                "source_id": "unsafe-a",
                "unsafe_label": 1,
                "features": [0.1, -0.2],
            },
            {
                "source_id": "unsafe-b",
                "unsafe_label": 1,
                "features": [0.3, 0.0],
            },
        ]
        model = _fit_logistic(samples, alpha=0.1)
        self.assertLess(
            _predict_risk([-1.0, -1.4], model),
            _predict_risk([0.2, -0.1], model),
        )

    def test_critical_prediction_has_no_stacked_expansion_component(self) -> None:
        record = {
            "selector": {
                "training_mode": "lora",
                "zero_stage": 2,
                "gradient_checkpointing": False,
                "packing": False,
            },
            "scenario": {"gpu_count": 2},
            "memory": {"safe_limit_bytes": 100.0},
        }
        bundle = {
            "separate_admission_head": {
                "threshold": 0.5,
                "model": {
                    "feature_means": [0.0, 0.0],
                    "feature_scales": [1.0, 1.0],
                    "intercept": -2.0,
                    "coefficients": [0.0, 0.0],
                },
            }
        }
        direct = {
            "available": True,
            "allocated_center_bytes": 40.0,
            "reserved_center_bytes": 50.0,
            "upper_bytes": 80.0,
            "upper_components": {
                "direct_reserved_upper": 80.0,
                "oom_upper": 60.0,
            },
        }
        with patch(
            "fit_h800_m1_separate_admission_v4._predict_safety",
            return_value=direct,
        ):
            prediction = _predict_separate_admission(record, bundle)
        self.assertTrue(prediction["admitted_by_separate_head"])
        self.assertNotIn(
            "allocated_expansion_upper", prediction["upper_components"]
        )
        self.assertEqual(
            set(prediction["upper_components"]),
            {"direct_reserved_upper", "oom_upper"},
        )


if __name__ == "__main__":
    unittest.main()

