from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from h800_frozen_vl_overlay_v1 import ARTIFACT_SCHEMA, predict_shadow_overlay


class H800FrozenVLOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifact = {
            "schema": ARTIFACT_SCHEMA,
            "memory_center_head": {"coefficient_bytes_per_proxy_byte": 2.0},
            "throughput_head": {
                "family_coefficients_seconds_per_pflop": {"qwen3_vl": 4.0}
            },
            "release_contract": {"mode": "shadow_only"},
        }
        self.features = {
            "memory_proxies_bytes": {"vision_dynamic_microbatch": 100.0},
            "throughput_proxies_per_sample": {"vision_forward_flops": 5.0e12},
        }

    def test_overlay_adds_nonnegative_visual_residuals(self) -> None:
        prediction = predict_shadow_overlay(
            text_memory_center_bytes=1000.0,
            text_step_seconds=10.0,
            effective_tokens_per_step=1000.0,
            logical_samples_per_step=50.0,
            vl_features=self.features,
            model_family="qwen3_vl",
            artifact=self.artifact,
        )
        self.assertEqual(prediction["memory"]["visual_residual_bytes"], 200.0)
        self.assertEqual(prediction["memory"]["predicted_center_bytes"], 1200.0)
        self.assertEqual(prediction["throughput"]["visual_forward_pflop"], 0.25)
        self.assertEqual(prediction["throughput"]["visual_seconds"], 1.0)
        self.assertEqual(prediction["throughput"]["predicted_step_seconds"], 11.0)
        self.assertFalse(prediction["automatic_admission_allowed"])
        self.assertFalse(prediction["automatic_ranking_allowed"])

    def test_right_censored_text_base_never_gets_an_imputed_center(self) -> None:
        prediction = predict_shadow_overlay(
            text_memory_center_bytes=None,
            text_step_seconds=None,
            effective_tokens_per_step=None,
            logical_samples_per_step=50.0,
            vl_features=self.features,
            model_family="qwen3_vl",
            artifact=self.artifact,
            text_memory_right_censored=True,
        )
        self.assertEqual(prediction["memory"]["state"], "right_censored_base_unsafe")
        self.assertIsNone(prediction["memory"]["predicted_center_bytes"])
        self.assertIsNone(prediction["memory"]["safety_upper_bytes"])


if __name__ == "__main__":
    unittest.main()
