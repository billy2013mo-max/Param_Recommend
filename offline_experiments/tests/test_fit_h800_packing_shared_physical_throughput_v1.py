from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import read_json, sha256_json
from fit_h800_packing_shared_physical_throughput_v1 import (
    FEATURE_FAMILIES,
    INPUT,
    OUTPUT,
    build_feature_rows,
)


class PackingSharedPhysicalThroughputTest(unittest.TestCase):
    def test_feature_rows_preserve_population_and_exclude_raw_n_pack(self) -> None:
        rows = build_feature_rows()
        self.assertEqual(23, len(rows))
        self.assertEqual(10, len({row["profile_group"] for row in rows}))
        for row in rows:
            self.assertNotIn("n_pack_mean", row["features"])
            self.assertGreater(row["n_pack_mean_physical_only"], 1.0)
        for features in FEATURE_FAMILIES.values():
            self.assertNotIn("n_pack_mean", features)

    def test_absolute_target_is_geometric_mean_of_repeat_ratios(self) -> None:
        source = read_json(INPUT)
        rows = {row["setting_id"]: row for row in build_feature_rows(source)}
        setting_id = "real:d4500edu:c32768"
        ratios = [
            row["observed"]["packed_effective_tokens_per_second"]
            / row["prediction"]["packed_physical_shared_model"][
                "predicted_effective_tokens_per_second"
            ]
            for row in source["pair_rows"]
            if row["setting_id"] == setting_id
        ]
        expected = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
        self.assertAlmostEqual(
            expected,
            rows[setting_id]["targets"]["retrospective_absolute_effective"],
        )

    def test_artifact_freezes_candidate_without_publishing(self) -> None:
        report = read_json(OUTPUT)
        body = dict(report)
        stored_hash = body.pop("report_sha256")
        self.assertEqual(stored_hash, sha256_json(body))
        self.assertEqual(
            "sft_h800_packing_shared_physical_throughput/v1",
            report["schema"],
        )
        self.assertFalse(report["publishable"])
        self.assertFalse(report["main_memory_model_mutated"])
        self.assertFalse(report["main_throughput_model_mutated"])
        self.assertFalse(report["input_contract"]["raw_n_pack_mean_in_residual"])
        self.assertFalse(report["input_contract"]["new_product_metadata_required"])

        absolute = report["validation"]["retrospective_absolute_effective"]
        paired = report["validation"]["paired_effect_transfer_effective"]
        self.assertEqual("profile_shape_3", absolute["selected_exploratory_candidate"])
        self.assertEqual("mechanism_4", paired["selected_exploratory_candidate"])
        self.assertTrue(absolute["selected_gate"]["all_passed"])
        self.assertFalse(paired["selected_gate"]["all_passed"])
        self.assertFalse(report["gates"]["all_modeling_gates_passed"])
        self.assertFalse(report["decision"]["merge_into_main_model_now"])

        selected_absolute = absolute["ridge_candidates"]["profile_shape_3"]
        self.assertEqual(3, len(selected_absolute["full_fit"]["features"]))
        self.assertLess(
            selected_absolute["metrics"]["group_equal_mape"],
            absolute["raw_shared_trunk"]["metrics"]["group_equal_mape"],
        )


if __name__ == "__main__":
    unittest.main()
