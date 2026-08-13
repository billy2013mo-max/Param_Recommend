from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_h800_memory_center_models_v1 as benchmark  # noqa: E402


class TestAnchorScaleFeatures(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "reference_bytes": 0.5 * benchmark.DEVICE_CAPACITY_BYTES,
            "features": {
                name: 0.0 for name in benchmark.LEGACY_FEATURE_NAMES
            },
        }
        self.row["features"]["activation_share"] = 0.4
        self.row["features"]["is_lora"] = 1.0

    def test_absolute_activation_fraction_has_correct_units(self) -> None:
        self.assertAlmostEqual(
            benchmark._feature_value(
                self.row, "activation_fraction_of_capacity"
            ),
            0.2,
        )

    def test_mechanism_interaction_uses_no_runtime_label(self) -> None:
        self.assertAlmostEqual(
            benchmark._feature_value(
                self.row, "is_lora_x_activation_fraction_of_capacity"
            ),
            0.2,
        )
        self.assertAlmostEqual(
            benchmark._feature_value(
                self.row,
                "gradient_checkpointing_x_activation_fraction_of_capacity",
            ),
            0.0,
        )


class TestCandidateContract(unittest.TestCase):
    def test_candidate_ids_are_unique_and_include_legacy_baseline(self) -> None:
        candidates = benchmark._candidate_grid()
        identifiers = [str(row["candidate_id"]) for row in candidates]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertIn(
            "legacy_linear__legacy__a0.1__s1__hsquared", identifiers
        )

    def test_source_identity_is_not_a_model_feature(self) -> None:
        forbidden = {"source_id", "dataset_id", "campaign", "origin"}
        self.assertTrue(forbidden.isdisjoint(benchmark.RAW_FEATURE_NAMES))
        self.assertTrue(forbidden.isdisjoint(benchmark.LEGACY_FEATURE_NAMES))

    def test_anchor_quadratic_basis_shape_is_deterministic(self) -> None:
        raw = np.asarray([[0.0, 0.5], [1.0, 1.5]], dtype=float)
        expanded = benchmark._expand_basis(
            raw,
            kind="anchor_quadratic",
            raw_means=np.zeros(2),
            raw_scales=np.ones(2),
            nonlinear_indexes=[1],
            feature_names=["is_lora", "reference_fraction_of_capacity"],
        )
        self.assertEqual(expanded.shape, (2, 3))
        np.testing.assert_allclose(expanded[:, -1], raw[:, 1] ** 2)


if __name__ == "__main__":
    unittest.main()
