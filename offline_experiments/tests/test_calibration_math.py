from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from calibration_math import (  # noqa: E402
    bounded_isotonic_fit,
    bounded_huber_fit,
    finite_sample_lower_quantile,
    finite_sample_upper_quantile,
    global_scenario_folds,
    kaplan_meier_quantile,
    scenario_equal_weights,
)


class CalibrationMathTests(unittest.TestCase):
    def test_bounded_fit_recovers_coefficients_and_clips(self) -> None:
        matrix = np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]])
        target = matrix @ np.asarray([2.0, 3.0])
        result = bounded_huber_fit(
            matrix,
            target,
            lower=[0.0, 0.0],
            upper=[2.0, 2.5],
            initial=[1.0, 1.0],
        )
        self.assertAlmostEqual(result.coefficients[0], 2.0, places=7)
        self.assertAlmostEqual(result.coefficients[1], 2.5, places=7)
        self.assertTrue(result.converged)

    def test_bounded_fit_is_deterministic_and_robust_to_outlier(self) -> None:
        matrix = np.arange(1, 22, dtype=np.float64).reshape(-1, 1)
        target = 1.5 * matrix[:, 0]
        target[-1] = 10_000
        first = bounded_huber_fit(matrix, target, [0.0], [4.0], initial=[1.0])
        second = bounded_huber_fit(matrix, target, [0.0], [4.0], initial=[1.0])
        self.assertEqual(first, second)
        self.assertLess(abs(first.coefficients[0] - 1.5), 0.1)

    def test_scenario_weights_give_each_scenario_equal_mass(self) -> None:
        weights = scenario_equal_weights(["a", "a", "a", "b"])
        self.assertAlmostEqual(float(sum(weights[:3])), float(weights[3]))
        self.assertAlmostEqual(float(np.mean(weights)), 1.0)

    def test_conformal_rank_requires_nineteen_for_95_percent(self) -> None:
        insufficient = finite_sample_upper_quantile(range(18))
        sufficient = finite_sample_upper_quantile(range(19))
        self.assertFalse(insufficient.identifiable)
        self.assertEqual(insufficient.rank, 19)
        self.assertTrue(sufficient.identifiable)
        self.assertEqual(sufficient.rank, 19)
        self.assertEqual(sufficient.value, 18)

    def test_lower_conformal_rank_requires_nineteen_for_95_percent(self) -> None:
        insufficient = finite_sample_lower_quantile(range(18))
        sufficient = finite_sample_lower_quantile(range(19))
        self.assertFalse(insufficient.identifiable)
        self.assertEqual(insufficient.rank, 0)
        self.assertTrue(sufficient.identifiable)
        self.assertEqual(sufficient.rank, 1)
        self.assertEqual(sufficient.value, 0)

    def test_bounded_isotonic_fit_pools_violations(self) -> None:
        increasing = bounded_isotonic_fit(
            [0.2, 0.6, 0.4, 1.2], lower=0.1, upper=1.0
        )
        decreasing = bounded_isotonic_fit(
            [0.8, 0.2, 0.4], lower=0.0, upper=1.0, increasing=False
        )
        self.assertEqual(increasing, (0.2, 0.5, 0.5, 1.0))
        self.assertEqual(decreasing[0], 0.8)
        self.assertAlmostEqual(decreasing[1], 0.3)
        self.assertAlmostEqual(decreasing[2], 0.3)

    def test_km_exact_sample_and_tail_censoring(self) -> None:
        exact = kaplan_meier_quantile(list(range(1, 21)), [True] * 20)
        censored = kaplan_meier_quantile(
            list(range(1, 19)) + [19, 20],
            [True] * 18 + [False, False],
        )
        self.assertTrue(exact.identifiable)
        self.assertEqual(exact.value, 19)
        self.assertFalse(censored.identifiable)
        self.assertIsNone(censored.value)

    def test_km_processes_event_before_censor_at_tie(self) -> None:
        result = kaplan_meier_quantile([1, 1], [True, False], probability=0.5)
        self.assertTrue(result.identifiable)
        self.assertEqual(result.value, 1)

    def test_global_folds_hold_scenario_across_all_rows(self) -> None:
        rows = [
            {"observation_id": "a1", "scenario_id": "a", "cohort": "x"},
            {"observation_id": "a2", "scenario_id": "a", "cohort": "y"},
            {"observation_id": "b1", "scenario_id": "b", "cohort": "x"},
        ]
        folds = global_scenario_folds(rows)
        self.assertEqual(len(folds), 2)
        a_fold = next(fold for fold in folds if fold["held_out_scenario_id"] == "a")
        self.assertEqual(a_fold["test_observation_ids"], ["a1", "a2"])
        self.assertEqual(a_fold["train_observation_ids"], ["b1"])

    def test_rejects_non_finite_fit_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            bounded_huber_fit([[1.0], [float("nan")]], [1.0, 2.0], [0.0], [1.0])


if __name__ == "__main__":
    unittest.main()
