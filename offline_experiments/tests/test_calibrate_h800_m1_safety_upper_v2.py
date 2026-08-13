from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from calibrate_h800_m1_safety_upper_v2 import (  # noqa: E402
    S0_STACKED,
    S1_RESERVED_ONLY,
    S2_RESERVED_OOM,
    S3_JOINT,
)
from common import read_json, sha256_file, sha256_json  # noqa: E402
from fit_h800_lora_source_disjoint_recalibration_v1 import (  # noqa: E402
    VARIANT_M1,
)


OUTPUT = ROOT / "diagnostics" / "h800_m1_safety_upper_v2_20260805"


class H800M1SafetyUpperV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = read_json(OUTPUT / "safety_upper_ablation_report.json")
        cls.candidate = read_json(OUTPUT / "candidate_model_safety_upper_v2.json")
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

    def test_data_and_holdout_isolation_contract(self) -> None:
        audit = self.report["data_audit"]
        self.assertEqual(audit["current_raw_observations"], 105)
        self.assertEqual(audit["historical_calibration_raw_observations"], 158)
        self.assertEqual(audit["training_raw_observations"], 263)
        self.assertEqual(audit["training_unique_configuration_results"], 213)
        self.assertEqual(audit["historical_holdout_raw_observations"], 167)
        self.assertEqual(
            audit["historical_holdout_unique_configuration_results"], 106
        )
        self.assertEqual(audit["training_sources"], 20)
        self.assertEqual(audit["training_holdout_observation_id_overlap"], 0)
        self.assertEqual(
            audit["training_holdout_configuration_result_overlap"], 0
        )
        self.assertFalse(
            self.report["nested_protocol"][
                "historical_fixed_holdout_used_for_selection"
            ]
        )

    def test_center_model_is_fixed_across_safety_ablation(self) -> None:
        contract = self.report["center_model_contract"]
        self.assertEqual(contract["variant"], VARIANT_M1)
        self.assertTrue(
            contract["same_fold_specific_centers_across_all_safety_variants"]
        )
        self.assertTrue(contract["center_refit_is_not_the_ablation_variable"])

    def test_s2_is_selected_by_training_oof_safety_gates(self) -> None:
        selection = self.report["selection"]
        self.assertEqual(selection["selected_variant"], S2_RESERVED_OOM)
        self.assertIn(S0_STACKED, selection["eligible_variants"])
        self.assertIn(S2_RESERVED_OOM, selection["eligible_variants"])
        self.assertIn(S1_RESERVED_ONLY, selection["diagnostic_only_variants"])
        self.assertFalse(selection["gates"][S3_JOINT]["all_passed"])

        scopes = self.report["training_nested_metrics"]
        baseline = scopes[S0_STACKED]["current_19_sources_critical_lora"]
        selected = scopes[S2_RESERVED_OOM]["current_19_sources_critical_lora"]
        self.assertEqual(baseline["admitted_safe_success_configurations"], 32)
        self.assertEqual(selected["admitted_safe_success_configurations"], 33)
        self.assertAlmostEqual(baseline["admission_recall"], 32 / 41)
        self.assertAlmostEqual(selected["admission_recall"], 33 / 41)
        self.assertGreaterEqual(
            selected["reserved_upper_source_equal_coverage"], 0.95
        )
        self.assertEqual(selected["unsafe_success_admitted"], 0)
        self.assertEqual(selected["false_safe_oom"], 0)
        self.assertEqual(selected["prediction_availability"], 1.0)

    def test_joint_variant_is_rejected_for_unsafe_admission(self) -> None:
        metrics = self.report["training_nested_metrics"][S3_JOINT][
            "selection_scope_combined_critical_lora"
        ]
        self.assertEqual(metrics["unsafe_success_admitted"], 1)
        self.assertEqual(metrics["false_safe_oom"], 0)
        self.assertEqual(metrics["prediction_availability"], 1.0)
        joint = self.report["selected_full_fit"]["bundle"]["joint_safety"]
        self.assertEqual(joint["inner_oof_scores"]["rows"], 213)
        self.assertEqual(joint["inner_oof_scores"]["success"], 173)
        self.assertEqual(joint["inner_oof_scores"]["oom"], 40)

    def test_historical_holdout_is_diagnostic_only(self) -> None:
        holdout = self.report["historical_fixed_holdout_diagnostics"]
        self.assertFalse(holdout["used_for_selection"])
        self.assertTrue(holdout["not_valid_for_future_formal_acceptance"])
        selected = holdout["metrics"][S2_RESERVED_OOM]
        critical = selected["critical_lora_diagnostic"]
        self.assertEqual(critical["safe_success_configurations"], 2)
        self.assertEqual(critical["admitted_safe_success_configurations"], 1)
        self.assertEqual(critical["unsafe_success_admitted"], 0)
        self.assertEqual(critical["false_safe_oom"], 0)
        all_mechanisms = selected["all_mechanisms_diagnostic"]
        self.assertEqual(all_mechanisms["safe_success_configurations"], 71)
        self.assertEqual(
            all_mechanisms["admitted_safe_success_configurations"], 24
        )

    def test_stage_two_and_new_prospective_holdout_remain_required(self) -> None:
        gate = self.report["release_gate"]
        self.assertTrue(gate["stage_two_calibration_required"])
        self.assertTrue(
            gate["stage_two_triggers"][
                "current_critical_admission_recall_below_0p90"
            ]
        )
        self.assertTrue(
            gate["stage_two_triggers"]["critical_tail_rank_is_empirical_max"]
        )
        self.assertTrue(gate["new_post_freeze_prospective_holdout_required"])
        self.assertFalse(gate["production_replacement_allowed"])
        self.assertEqual(self.candidate["selected_safety_variant"], S2_RESERVED_OOM)
        self.assertFalse(self.candidate["publishable"])
        self.assertFalse(self.candidate["production_override_allowed"])


if __name__ == "__main__":
    unittest.main()
