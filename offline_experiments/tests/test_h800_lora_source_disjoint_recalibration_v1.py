from __future__ import annotations

import math
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
    VARIANT_M2,
    VARIANT_M3,
    _round_up,
    _upper_quantile,
)


OUTPUT = ROOT / "diagnostics" / "h800_lora_memory_recalibration_20260804"


class H800LoraSourceDisjointRecalibrationV1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = read_json(OUTPUT / "nested_ablation_results.json")
        cls.candidate = read_json(OUTPUT / "candidate_model.json")
        cls.manifest = read_json(OUTPUT / "output_manifest.json")

    def test_effective_sequence_rounding_contract(self) -> None:
        self.assertEqual(_round_up(1, 8), 8)
        self.assertEqual(_round_up(2560, 8), 2560)
        self.assertEqual(_round_up(2561, 8), 2568)

    def test_finite_sample_95_percent_boundary(self) -> None:
        eighteen = _upper_quantile(
            {str(i): float(i) for i in range(18)},
            coverage=0.95,
            diagnostic_max_fallback=True,
        )
        self.assertFalse(eighteen["formal_finite_sample"])
        self.assertTrue(eighteen["diagnostic_empirical_max_fallback"])
        self.assertEqual(eighteen["rank"], 19)
        self.assertAlmostEqual(
            eighteen["maximum_guaranteed_coverage_if_fallback"], 18 / 19
        )

        nineteen = _upper_quantile(
            {str(i): float(i) for i in range(19)},
            coverage=0.95,
            diagnostic_max_fallback=False,
        )
        self.assertTrue(nineteen["formal_finite_sample"])
        self.assertFalse(nineteen["diagnostic_empirical_max_fallback"])
        self.assertEqual(nineteen["rank"], 19)
        self.assertEqual(nineteen["log_upper"], 18.0)

    def test_report_and_candidate_are_checksum_sealed(self) -> None:
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

    def test_data_audit_and_repeat_collapse_are_explicit(self) -> None:
        audit = self.report["data_audit"]
        self.assertEqual(audit["new_observations"], 77)
        self.assertEqual(audit["old_observations"], 28)
        self.assertEqual(audit["new_outcomes"], {"success": 68, "oom": 9})
        self.assertEqual(audit["combined_independent_sources"], 19)
        self.assertEqual(audit["source_overlap"], [])
        self.assertEqual(audit["collapse"]["unique_configurations"], 87)
        self.assertEqual(audit["collapsed_outcomes"], {"success": 80, "oom": 7})
        for group in audit["collapse"]["repeat_groups"]:
            if group["outcome"] == "oom":
                self.assertTrue(group["oom_reproduced"])
            else:
                self.assertEqual(group["allocated_relative_range"], 0.0)
                self.assertEqual(group["reserved_relative_range"], 0.0)

    def test_effective_anchor_fixes_cutoff_bias(self) -> None:
        anchor = self.report["anchor_diagnostics"]
        self.assertGreater(anchor["cutoff_anchor_source_equal_mape"], 0.25)
        self.assertLess(anchor["effective_anchor_source_equal_mape"], 0.07)
        lora = anchor["by_training_mode"]["lora"]
        self.assertGreater(lora["cutoff_anchor_mape"], 0.40)
        self.assertLess(lora["effective_anchor_mape"], 0.07)

    def test_m1_selected_and_ratio_terms_rejected_as_unstable(self) -> None:
        selection = self.report["selection"]
        self.assertEqual(selection["selected_variant"], VARIANT_M1)
        self.assertTrue(selection["ratio_stability"][VARIANT_M1])
        self.assertFalse(selection["ratio_stability"][VARIANT_M2])
        self.assertFalse(selection["ratio_stability"][VARIANT_M3])
        for variant in (VARIANT_M2, VARIANT_M3):
            entries = self.report["nested_M1_M3"][variant][
                "coefficient_stability"
            ]
            self.assertTrue(
                any(entry["sign_changes_across_range"] for entry in entries.values())
            )

    def test_safety_gates_pass_but_stage_two_is_required(self) -> None:
        metrics = self.report["nested_M1_M3"][VARIANT_M1]["metrics"]
        self.assertEqual(metrics["false_safe_oom"], 0)
        self.assertEqual(metrics["unsafe_success_admitted"], 0)
        self.assertGreaterEqual(metrics["reserved_upper_source_equal_coverage"], 0.95)
        self.assertLess(metrics["allocated_center_source_equal_mape"], 0.10)
        self.assertLess(metrics["admission_recall"], 0.90)

        gate = self.report["release_gate"]
        self.assertTrue(gate["stage_two_calibration_required"])
        self.assertTrue(all(gate["stage_two_triggers"].values()))
        self.assertEqual(gate["stage_two_plan"]["jobs"], 60)
        self.assertFalse(gate["formal_prospective_acceptance_passed"])
        self.assertFalse(gate["production_replacement_allowed"])
        self.assertFalse(self.candidate["publishable"])
        self.assertFalse(self.candidate["production_override_allowed"])
        self.assertTrue(self.candidate["stage_two_calibration_required"])

        expansion = self.report["selected_full_fit"]["critical_expansion"]
        self.assertTrue(expansion["formal_finite_sample"])
        self.assertEqual(expansion["independent_sources"], 19)
        self.assertTrue(
            math.isclose(
                expansion["expansion_upper"],
                expansion["empirical_max_expansion"],
            )
        )


if __name__ == "__main__":
    unittest.main()
