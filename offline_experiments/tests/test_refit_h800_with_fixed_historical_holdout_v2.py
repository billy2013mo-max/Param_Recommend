from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import read_json, sha256_file, sha256_json  # noqa: E402
from fit_h800_lora_source_disjoint_recalibration_v1 import (  # noqa: E402
    VARIANT_M1,
)


OUTPUT = ROOT / "diagnostics" / "h800_fixed_historical_holdout_20260805"


class FixedHistoricalHoldoutV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = read_json(OUTPUT / "fixed_historical_holdout_report.json")
        cls.candidate = read_json(OUTPUT / "candidate_model_strict_holdout.json")
        cls.manifest = read_json(OUTPUT / "output_manifest.json")

    def test_outputs_are_checksum_sealed(self) -> None:
        unsigned_report = dict(self.report)
        report_sha = unsigned_report.pop("report_sha256")
        self.assertEqual(report_sha, sha256_json(unsigned_report))

        unsigned_candidate = dict(self.candidate)
        candidate_sha = unsigned_candidate.pop("candidate_sha256")
        self.assertEqual(candidate_sha, sha256_json(unsigned_candidate))
        self.assertEqual(
            self.candidate["source_recalibration_report_sha256"], report_sha
        )

        unsigned_manifest = dict(self.manifest)
        manifest_sha = unsigned_manifest.pop("manifest_sha256")
        self.assertEqual(manifest_sha, sha256_json(unsigned_manifest))
        for binding in self.manifest["files"].values():
            self.assertEqual(binding["sha256"], sha256_file(Path(binding["path"])))

    def test_original_holdout_never_enters_fit(self) -> None:
        contract = self.report["data_contract"]
        self.assertEqual(contract["current_observations"], 105)
        self.assertEqual(
            contract["historical_original_calibration_observations"], 158
        )
        self.assertEqual(contract["historical_fixed_holdout_observations"], 167)
        self.assertEqual(contract["training_raw_observations"], 263)
        self.assertEqual(contract["training_unique_configuration_results"], 213)
        self.assertEqual(contract["holdout_unique_configuration_results"], 106)
        self.assertEqual(contract["training_holdout_observation_id_overlap"], 0)
        self.assertEqual(
            contract["historical_training_holdout_configuration_result_overlap"],
            0,
        )
        for key in (
            "holdout_used_for_variant_selection",
            "holdout_used_for_center_fit",
            "holdout_used_for_residual_or_expansion_calibration",
            "holdout_used_for_oom_guard_fit",
        ):
            self.assertFalse(contract[key])

    def test_fixed_holdout_safety_improves_with_historical_calibration(self) -> None:
        metrics = self.report["metrics"]["historical_fixed_holdout"]
        baseline = metrics["current_only_training"]
        augmented = metrics["augmented_training"]
        self.assertEqual(augmented["configurations"], 106)
        self.assertEqual(augmented["success_configurations"], 74)
        self.assertEqual(augmented["oom_configurations"], 32)
        self.assertEqual(baseline["unsafe_success_admitted"], 2)
        self.assertEqual(baseline["false_safe_oom"], 4)
        self.assertEqual(augmented["unsafe_success_admitted"], 0)
        self.assertEqual(augmented["false_safe_oom"], 0)
        self.assertAlmostEqual(
            augmented["allocated_center_source_equal_mape"],
            0.1970107684931578,
        )
        self.assertAlmostEqual(
            augmented["reserved_center_source_equal_mape"],
            0.23076186380127042,
        )
        self.assertEqual(augmented["reserved_upper_source_equal_coverage"], 1.0)

    def test_pooled_summary_and_scope_limits_are_explicit(self) -> None:
        pooled = self.report["metrics"][
            "pooled_current_plus_historical_holdout"
        ]["augmented_training"]
        self.assertEqual(pooled["configurations"], 193)
        self.assertEqual(pooled["independent_sources"], 20)
        self.assertEqual(pooled["false_safe_oom"], 0)
        self.assertEqual(pooled["unsafe_success_admitted"], 0)
        self.assertEqual(pooled["reserved_upper_source_equal_coverage"], 1.0)
        self.assertTrue(
            self.report["data_contract"][
                "historical_train_and_holdout_share_dataset_source_group"
            ]
        )

    def test_candidate_remains_shadow_and_stage_two_is_required(self) -> None:
        self.assertEqual(self.candidate["selected_variant"], VARIANT_M1)
        self.assertFalse(self.candidate["publishable"])
        self.assertFalse(self.candidate["production_override_allowed"])
        self.assertTrue(self.candidate["stage_two_calibration_required"])
        gate = self.report["release_gate"]
        self.assertTrue(gate["stage_two_calibration_required"])
        self.assertFalse(gate["formal_new_source_prospective_acceptance_passed"])
        self.assertFalse(gate["production_replacement_allowed"])


if __name__ == "__main__":
    unittest.main()
