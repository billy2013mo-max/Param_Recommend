"""Unit tests for the pre-registered stage-1 fit script."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from fit_h800_hybrid_attention_dense_stage1_v1 import (
    CEIL,
    COEFFICIENTS,
    ROUTES,
    THRESHOLDS,
    _collapsed_rows,
    _fit_route,
    _predict,
    _verify_protocol,
    protocol_payload,
)


def _observation(job_id: str, peak: int | None, classification: str) -> dict:
    memory = {
        "state_bytes": 10_000_000_000.0,
        "activation_components": {
            "saved_full_attention_activations_bytes": 1_000_000_000.0,
            "saved_linear_attention_activations_bytes": 0.0,
            "recompute_workspace_bytes": 500_000_000.0,
        },
        "workspace_candidates": {
            "full_attention_workspace_bytes": 200_000_000.0,
            "linear_attention_workspace_bytes": 0.0,
            "linear_recurrent_state_bytes": 0.0,
            "logits_workspace_bytes": 100_000_000.0,
            "zero_collective_workspace_bytes": 50_000_000.0,
        },
    }
    return {
        "job_id": job_id,
        "model_id": "qwen3_8b",
        "architecture_route": "dense_full_attention",
        "configuration": {
            "design_arm": "核心正交",
            "gpu_count": 1,
            "zero": "none",
            "gradient_checkpointing": False,
            "micro_batch_size": 1,
            "exact_total_tokens_per_sample": 8192,
            "source_dataset_id": "source_a",
            "repeat": 0,
            "packing": False,
        },
        "outcome": {
            "classification": classification,
            "terminal_eligible": True,
            "peak_reserved_bytes": peak,
            "oom_right_censor_lower_bytes": (
                149_000_000_000 if classification == "oom" else None
            ),
            "oom_censor_source": (
                "nvidia_smi_watermark" if classification == "oom" else None
            ),
        },
        "feature_basis": {"memory": memory},
        "metrics": None,
    }


class CollapseTests(unittest.TestCase):
    def test_repeat_folding_takes_median(self) -> None:
        rows = [
            _observation("a", peak=120_000_000_000, classification="success"),
            _observation("b", peak=130_000_000_000, classification="success"),
        ]
        collapsed = _collapsed_rows(rows)
        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed[0]["peak_reserved_bytes"], 125_000_000_000)
        self.assertEqual(collapsed[0]["repeat_count"], 2)

    def test_mixed_success_and_oom_collapses_to_censored(self) -> None:
        rows = [
            _observation("a", peak=120_000_000_000, classification="success"),
            _observation("b", peak=None, classification="oom"),
        ]
        collapsed = _collapsed_rows(rows)
        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed[0]["state"], "censored")
        self.assertEqual(
            collapsed[0]["censor_lower_bytes"], 149_000_000_000
        )
        self.assertEqual(
            collapsed[0]["discarded_exact_peaks"], [120_000_000_000]
        )


class FitRouteTests(unittest.TestCase):
    def _rows(self, peaks: list[int], censored_bounds: list[int]) -> list:
        rows = [
            _observation(f"e{i}", peak=peak, classification="success")
            for i, peak in enumerate(peaks)
        ]
        rows.extend(
            _observation(f"c{i}", peak=None, classification="oom")
            for i, bound in enumerate(censored_bounds)
        )
        return _collapsed_rows(rows)

    def test_zero_feature_columns_are_fixed_at_zero(self) -> None:
        rows = self._rows([120_000_000_000], [])
        fit = _fit_route(rows)
        names = [name for name, _ in COEFFICIENTS]
        for name in fit["fixed_zero_coefficients"]:
            self.assertEqual(
                fit["coefficients_by_name"][name], 0.0
            )
        self.assertIn("saved_linear", fit["fixed_zero_coefficients"])
        self.assertTrue(fit["solver_success"])

    def test_censored_bound_pulls_prediction_up(self) -> None:
        exact = self._rows([120_000_000_000, 122_000_000_000], [])
        censored = self._rows(
            [120_000_000_000, 122_000_000_000],
            [200_000_000_000],  # impossible bound: must push prediction up
        )
        exact_fit = _fit_route(exact)
        censored_fit = _fit_route(censored)
        exact_pred = _predict(
            np.asarray(exact_fit["coefficients"]), exact[0]["design"]
        )
        censored_pred = _predict(
            np.asarray(censored_fit["coefficients"]), censored[0]["design"]
        )
        self.assertGreater(censored_pred, exact_pred * 1.05)

    def test_exact_fit_reproduces_peaks(self) -> None:
        rows = self._rows(
            [120_000_000_000, 122_000_000_000, 118_000_000_000,
             125_000_000_000, 121_000_000_000, 123_000_000_000],
            [],
        )
        fit = _fit_route(rows)
        for row in rows:
            predicted = _predict(
                np.asarray(fit["coefficients"]), row["design"]
            )
            ape = abs(predicted - row["peak_reserved_bytes"]) / row[
                "peak_reserved_bytes"
            ]
            self.assertLess(ape, 0.05)


class ProtocolTests(unittest.TestCase):
    def test_verify_protocol_accepts_fresh_payload(self) -> None:
        payload = protocol_payload()
        _verify_protocol(payload)  # must not raise

    def test_verify_protocol_rejects_tampered_thresholds(self) -> None:
        payload = protocol_payload()
        payload["thresholds"] = dict(THRESHOLDS)
        payload["thresholds"]["exact_mape_max"] = 0.5
        with self.assertRaises(RuntimeError):
            _verify_protocol(payload)

    def test_verify_protocol_rejects_tampered_release_mode(self) -> None:
        payload = protocol_payload()
        payload["release"] = {
            "mode": "automatic",
            "automatic_admission_allowed": True,
        }
        with self.assertRaises(RuntimeError):
            _verify_protocol(payload)


if __name__ == "__main__":
    unittest.main()
