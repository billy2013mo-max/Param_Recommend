from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import read_json  # noqa: E402
from fit_packing_profile_estimators_v1 import (  # noqa: E402
    CALIBRATION_WORKLOADS,
    FEATURE_NAMES,
    PROFILE_MANIFEST,
    PROSPECTIVE_WORKLOADS,
    _fit_ridge,
    build_fit_membership,
    build_label_rows,
    feature_vector,
    predict_serialized,
)


class PackingProfileEstimatorV1Tests(unittest.TestCase):
    def test_serialized_ridge_round_trip(self) -> None:
        x = np.asarray([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
        y = np.asarray([0.7, 0.8, 0.9, 0.95])
        model = _fit_ridge(x, y, alpha=1.0, transform="logit")
        prediction = predict_serialized(model, x)
        self.assertEqual(prediction.shape, y.shape)
        self.assertTrue(np.all(prediction > 0))
        self.assertTrue(np.all(prediction < 1))

    def test_features_do_not_read_curve_targets(self) -> None:
        manifest = read_json(PROFILE_MANIFEST)
        profile = read_json(Path(manifest["profiles"][0]["path"]))
        point = copy.deepcopy(profile["packing_curve"][0])
        for key in ("pack_utilization", "packs", "samples_per_pack"):
            point.pop(key)
        values = feature_vector(profile, point)
        self.assertEqual(len(values), len(FEATURE_NAMES))

    def test_family_membership_is_disjoint_and_labels_are_exact(self) -> None:
        rows = build_label_rows(read_json(PROFILE_MANIFEST))
        calibration = {row["workload_id"] for row in rows if row["fit_eligible"]}
        prospective = {
            row["workload_id"] for row in rows
            if row["profile_partition"] == "prospective_holdout" and row["nontruncating"]
        }
        self.assertEqual(calibration, set(CALIBRATION_WORKLOADS))
        self.assertEqual(prospective, set(PROSPECTIVE_WORKLOADS))
        self.assertTrue(calibration.isdisjoint(prospective))
        for row in rows:
            if not row["nontruncating"]:
                continue
            expected = (
                row["packing_capacity"]
                * row["targets"]["pack_utilization"]
                / row["physics"]["length_mean"]
            )
            self.assertAlmostEqual(expected, row["targets"]["n_pack_mean"], places=9)

    def test_gpu_membership_keeps_invalid_old_w4_in_shadow(self) -> None:
        membership = build_fit_membership()
        self.assertEqual(membership["counts"]["jobs"], 42)
        self.assertEqual(membership["counts"]["strict_target_gbs_fit_only"], 30)
        self.assertEqual(membership["counts"]["distribution_labeled_shadow_only"], 12)
        old_w4 = [
            row for row in membership["rows"]
            if row["source"] == "platform_v4_interactions_batch1_combined"
            and row["setting_id"].startswith("w4_")
        ]
        self.assertEqual(len(old_w4), 12)
        self.assertTrue(all(row["evidence_role"] == "distribution_labeled_shadow_only" for row in old_w4))


if __name__ == "__main__":
    unittest.main()
