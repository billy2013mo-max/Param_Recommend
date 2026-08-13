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


OUTPUT = ROOT / "diagnostics" / "h800_historical_memory_migration_20260805"


class HistoricalMemoryMigrationRefitV1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = read_json(OUTPUT / "historical_migration_refit_report.json")
        cls.candidate = read_json(OUTPUT / "candidate_model.json")
        cls.manifest = read_json(OUTPUT / "output_manifest.json")

    def test_reports_are_checksum_sealed(self) -> None:
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

    def test_historical_exact_and_legacy_exclusion_contract(self) -> None:
        audit = self.report["historical_migration_audit"]
        self.assertEqual(audit["strict_exact_observations_admitted"], 325)
        self.assertEqual(
            audit["strict_exact_outcomes"], {"success": 232, "oom": 93}
        )
        self.assertEqual(
            audit["original_partition_roles"],
            {"calibration": 158, "holdout": 167},
        )
        self.assertEqual(
            audit["former_holdout_rows_reclassified_as_consumed_calibration"],
            167,
        )
        legacy = audit["legacy_non_attempt_scoped_observations_excluded"]
        self.assertEqual(legacy["rows"], 594)
        reasons = audit["canonical_audit"][
            "legacy_calibration_exclusion_reasons"
        ]
        self.assertEqual(reasons["event_attempt_binding_incomplete"], 594)
        self.assertEqual(
            reasons["structured_terminal_classification_evidence_missing"], 594
        )

    def test_profiles_and_content_independence_are_bound(self) -> None:
        audit = self.report["historical_migration_audit"]
        self.assertEqual(len(audit["dataset_registry"]), 5)
        for binding in audit["dataset_registry"].values():
            self.assertEqual(binding["data_rows"], 1000)
            self.assertEqual(binding["profile_rows"], 1000)
            self.assertEqual(
                binding["data_sha256"], sha256_file(Path(binding["data_path"]))
            )
            self.assertEqual(
                binding["profile_sha256"],
                sha256_file(Path(binding["profile_path"])),
            )
        self.assertEqual(audit["historical_current_content_overlap_total"], 0)
        self.assertEqual(len(audit["historical_connected_components"]), 1)
        self.assertEqual(
            len(audit["historical_connected_components"][0]), 5
        )
        self.assertEqual(
            audit["historical_mixed_outcome_configuration_group_count"], 2
        )

    def test_combined_data_counts_and_source_weighting(self) -> None:
        audit = self.report["combined_data_audit"]
        self.assertEqual(audit["current_raw_observations"], 105)
        self.assertEqual(audit["historical_raw_observations"], 325)
        self.assertEqual(audit["combined_raw_observations"], 430)
        self.assertEqual(audit["combined_independent_sources"], 20)
        self.assertEqual(audit["collapse"]["unique_configurations"], 319)
        self.assertEqual(audit["collapse"]["collapsed_repeat_rows"], 111)
        self.assertEqual(
            audit["collapsed_outcomes"], {"success": 247, "oom": 72}
        )

    def test_candidate_scope_selects_m1_but_does_not_expand_scope(self) -> None:
        selection = self.report["selection"]
        self.assertEqual(
            selection["candidate_scope_selection"]["selected_variant"],
            VARIANT_M1,
        )
        self.assertIsNone(
            selection["all_mechanism_diagnostic_selection"]["selected_variant"]
        )
        metrics = self.report["selected_scope_diagnostics"][
            "combined_critical_lora"
        ]
        self.assertEqual(metrics["false_safe_oom"], 0)
        self.assertEqual(metrics["unsafe_success_admitted"], 0)
        self.assertEqual(metrics["reserved_upper_source_equal_coverage"], 1.0)
        self.assertFalse(self.candidate["publishable"])
        self.assertFalse(self.candidate["production_override_allowed"])
        self.assertTrue(self.candidate["stage_two_calibration_required"])

    def test_migrated_history_does_not_remove_stage_two_requirement(self) -> None:
        gate = self.report["release_gate"]
        self.assertTrue(gate["stage_two_calibration_required"])
        self.assertTrue(all(gate["stage_two_triggers"].values()))
        self.assertFalse(gate["formal_prospective_acceptance_passed"])
        self.assertFalse(gate["production_replacement_allowed"])
        expansion = self.report["selected_full_fit"]["critical_expansion"]
        self.assertEqual(expansion["independent_sources"], 20)
        self.assertEqual(expansion["rank"], 20)
        self.assertEqual(
            expansion["expansion_upper"], expansion["empirical_max_expansion"]
        )


if __name__ == "__main__":
    unittest.main()
