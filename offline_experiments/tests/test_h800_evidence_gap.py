from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import h800_evidence_gap as gap  # noqa: E402


def failing_calibration() -> dict:
    return {
        "schema": "sft_h800_theory_calibration/v1",
        "status": "theory_only",
        "publishable": False,
        "report_sha256": "calibration-sha",
        "blockers": [
            "verified_calibration_anchor_missing",
            "planner_trusted_exact_operator_manifest_missing",
            "prospective_acceptance_required",
        ],
        "aggregate_validation": {
            "memory_feasibility": {
                "success_p95_coverage": 0.94,
                "scenario_equal_success_p95_coverage": 0.90,
                "oom_rows": 106,
                "false_safe_oom": 2,
                "false_safe_oom_by_selector": {'["lora",3,false,false]': 2},
            },
            "throughput_primary": {
                "scenario_equal_top1_regret": 0.20,
                "memory_gated_safety_failures": 10,
                "scaling_1_8_claims": 0,
                "scaling_1_8_valid_claims": 0,
            },
        },
        "full_historical_bootstrap_fit": {
            "memory": {"tail": {"cohort_evidence_inflation": {"runtime_cohorts": 2}}}
        },
    }


def basis_with_conflict() -> dict:
    """A basis where LoRA+ZeRO-3 has one (mbs,seq,gpu) cell that both succeeds and OOMs."""
    def mrow(oid, outcome):
        return {
            "observation_id": oid,
            "route": {"memory_boundary": True, "feasibility": True},
            "outcome": outcome,
            "selector": {"training_mode": "lora", "zero_stage": 3,
                         "gradient_checkpointing": False, "packing": False},
            "scenario": {"physical_mbs": 4, "cutoff_len": 4096, "gpu_count": 4},
        }
    return {"schema": "sft_h800_theory_basis/v1", "records": [
        mrow("s1", "success"), mrow("o1", "oom")]}


class EvidenceGapTests(unittest.TestCase):
    def test_derives_named_mechanism_gaps_without_a_queue(self) -> None:
        report = gap.build_gap_report(failing_calibration())
        self.assertEqual(gap.validate_gap_report(report), [])
        self.assertFalse(report["creates_gpu_queue"])
        self.assertFalse(report["proposes_campaign"])
        self.assertTrue(report["requires_separately_approved_design"])
        ids = {item["gap_id"] for item in report["gaps"]}
        self.assertIn("memory_p95_coverage_below_bar", ids)
        self.assertIn("throughput_top1_regret_above_bar", ids)
        self.assertIn("scaling_1_8x_unproven", ids)
        self.assertIn("no_verified_publication_anchor", ids)
        # Without a basis, a false-safe cannot be proven a conflict -> model-driven.
        self.assertIn("false_safe_oom_present", ids)
        false_safe = next(g for g in report["gaps"] if g["gap_id"] == "false_safe_oom_present")
        self.assertEqual(
            false_safe["prospective_acceptance"]["needs_success_and_oom_boundary_for_selectors"],
            ['["lora",3,false,false]'],
        )
        for item in report["gaps"]:
            self.assertTrue(item["mechanism"])
            self.assertTrue(item["prospective_acceptance"])

    def test_false_safe_reclassified_as_label_conflict_with_basis(self) -> None:
        report = gap.build_gap_report(failing_calibration(), basis_with_conflict())
        self.assertEqual(gap.validate_gap_report(report), [])
        ids = {item["gap_id"] for item in report["gaps"]}
        # The same-config success+OOM cell reclassifies it away from model error.
        self.assertIn("false_safe_is_label_conflict_not_model_error", ids)
        self.assertNotIn("false_safe_oom_present", ids)
        conflict = next(
            g for g in report["gaps"]
            if g["gap_id"] == "false_safe_is_label_conflict_not_model_error"
        )
        self.assertEqual(conflict["observed"]["conflicted_selectors"], ['["lora",3,false,false]'])
        self.assertTrue(
            conflict["prospective_acceptance"][
                "triage_these_ooms_as_software_or_infrastructure_failure_first"
            ]
        )
        self.assertTrue(
            conflict["prospective_acceptance"]["never_exclude_lora_zero3_from_search_on_this_alone"]
        )

    def test_passing_memory_and_throughput_drop_those_gaps(self) -> None:
        clean = failing_calibration()
        clean["aggregate_validation"]["memory_feasibility"].update(
            {
                "success_p95_coverage": 0.99,
                "scenario_equal_success_p95_coverage": 0.99,
                "false_safe_oom": 0,
                "false_safe_oom_by_selector": {},
            }
        )
        clean["aggregate_validation"]["throughput_primary"].update(
            {"scenario_equal_top1_regret": 0.02, "memory_gated_safety_failures": 0}
        )
        report = gap.build_gap_report(clean)
        ids = {item["gap_id"] for item in report["gaps"]}
        self.assertNotIn("memory_p95_coverage_below_bar", ids)
        self.assertNotIn("false_safe_oom_present", ids)
        self.assertNotIn("throughput_top1_regret_above_bar", ids)
        self.assertNotIn("memory_gated_ranking_safety_failures", ids)

    def test_rejects_publishable_source_and_queue_tamper(self) -> None:
        publishable = failing_calibration()
        publishable["publishable"] = True
        with self.assertRaisesRegex(ValueError, "nonpublishable"):
            gap.build_gap_report(publishable)
        report = gap.build_gap_report(failing_calibration())
        tampered = copy.deepcopy(report)
        tampered["creates_gpu_queue"] = True
        self.assertIn(
            "gap_report_must_not_create_a_queue", gap.validate_gap_report(tampered)
        )


if __name__ == "__main__":
    unittest.main()
