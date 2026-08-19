"""Tests for the stage-2 dense hybrid-attention memory fit (ZeRO-3 fix)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from common import ARTIFACT_DIR  # noqa: E402
import hybrid_attention_memory_features_v2 as feats_v2  # noqa: E402

ARTIFACT_V2 = ARTIFACT_DIR / "h800_hybrid_memory_artifact_v2.json"


class Zero3WorkspaceFeatureTest(unittest.TestCase):
    def test_zero3_multi_gpu_is_positive(self):
        value = feats_v2.zero3_param_workspace_bytes(
            loaded_parameters=9_000_000_000,
            max_module_parameter_elements=600_000_000,
            gpu_count=2,
            zero_stage=3,
        )
        self.assertGreater(value, 0.0)

    def test_single_gpu_and_non_zero3_are_zero(self):
        common = dict(loaded_parameters=9_000_000_000, max_module_parameter_elements=600_000_000)
        self.assertEqual(
            feats_v2.zero3_param_workspace_bytes(gpu_count=1, zero_stage=3, **common), 0.0
        )
        self.assertEqual(
            feats_v2.zero3_param_workspace_bytes(gpu_count=2, zero_stage=2, **common), 0.0
        )
        self.assertEqual(
            feats_v2.zero3_param_workspace_bytes(gpu_count=4, zero_stage=0, **common), 0.0
        )

    def test_monotonic_in_module_size(self):
        small = feats_v2.zero3_param_workspace_bytes(
            loaded_parameters=9e9, max_module_parameter_elements=1e8, gpu_count=2, zero_stage=3
        )
        large = feats_v2.zero3_param_workspace_bytes(
            loaded_parameters=9e9, max_module_parameter_elements=6e8, gpu_count=2, zero_stage=3
        )
        self.assertGreater(large, small)


class ArtifactV2Test(unittest.TestCase):
    def setUp(self):
        if not ARTIFACT_V2.is_file():
            self.skipTest("stage-2 artifact not generated yet")
        from common import read_json

        self.artifact = read_json(ARTIFACT_V2)

    def test_shadow_only_not_admissible(self):
        self.assertEqual(self.artifact["status"], "hybrid_shadow_candidate")
        self.assertIs(self.artifact["production_admission_allowed"], False)

    def test_zero3_param_workspace_coefficient_is_positive(self):
        coef = self.artifact["coefficients_by_name"]
        self.assertIn("zero3_param_workspace", coef)
        self.assertGreater(coef["zero3_param_workspace"], 0.0)

    def test_state_coefficient_in_range(self):
        self.assertTrue(0.8 <= self.artifact["coefficients_by_name"]["state"] <= 1.25)


class FitGatesTest(unittest.TestCase):
    """Heavy: re-runs the stage-2 fit. Requires scipy (venv python)."""

    def test_fit_passes_all_pre_registered_gates(self):
        try:
            import fit_h800_hybrid_attention_dense_stage2_v1 as fit
        except ModuleNotFoundError as error:
            self.skipTest(f"scipy unavailable: {error}")
        if not fit.PROTOCOL_PATH.is_file():
            self.skipTest("stage-2 protocol not frozen")
        report = fit.fit(allow_incomplete=False)
        self.assertTrue(report["data_complete"])
        self.assertTrue(report["all_gates_passed"], report["gates"])
        pooled = report["cross_validation"]["dense_hybrid_attention"]["pooled_held_out"]
        self.assertLessEqual(pooled["zero3_exact_mape"], 0.10)

    def test_large_model_zero3_no_longer_systematically_underpredicted(self):
        try:
            import fit_h800_hybrid_attention_dense_stage2_v1 as fit
        except ModuleNotFoundError as error:
            self.skipTest(f"scipy unavailable: {error}")
        report = fit.fit(allow_incomplete=True)
        replay = report["large_model_zero3_replay"]["dense_hybrid_attention"]
        self.assertTrue(replay, "expected large-model ZeRO-3 replay rows")
        ratios = [row["actual_over_predicted"] for row in replay]
        mean_ratio = sum(ratios) / len(ratios)
        # Stage-1 systematically under-predicted (actual/predicted ~1.15-1.18).
        # Stage-2 should center the ratio near 1.0, not high.
        self.assertLess(mean_ratio, 1.08, f"mean actual/predicted still high: {mean_ratio}")
        self.assertGreater(mean_ratio, 0.92, f"mean actual/predicted too low: {mean_ratio}")


if __name__ == "__main__":
    unittest.main()
