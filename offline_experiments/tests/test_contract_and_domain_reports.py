"""Tests for the contract registry, domain reconciliation and VL replay."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import reconcile_support_domains as recon  # noqa: E402
import replay_vl_endpoints as vlreplay  # noqa: E402
import unified_contract_registry as registry  # noqa: E402


class TestContractRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.report = registry.build_registry()

    def test_all_three_plan_contracts_are_present(self) -> None:
        names = {item["plan_name"] for item in self.report["contracts"]}
        self.assertEqual(
            names,
            {
                "ModelStructureManifest/v1",
                "WorkloadProfile/v1",
                "RuntimeMechanism/v1",
            },
        )

    def test_registry_aliases_rather_than_forking_schemas(self) -> None:
        # Redeclaring these would fork three live formats that frozen artefacts
        # already bind by SHA.
        policy = self.report["policy"]
        self.assertFalse(policy["authors_new_schemas"])
        self.assertFalse(policy["renames_live_schemas"])
        self.assertFalse(policy["forks_existing_formats"])

    def test_plan_names_map_onto_live_schema_ids(self) -> None:
        self.assertEqual(
            registry.resolve_schema("WorkloadProfile/v1"),
            "sft_static_workload_profiles/v1",
        )
        self.assertEqual(
            registry.resolve_schema("ModelStructureManifest/v1"),
            "sft_model_structure_manifest/v1",
        )
        # The plan says v1 but the live mechanism format is already v2; the
        # higher live version must win.
        self.assertEqual(
            registry.resolve_schema("RuntimeMechanism/v1"),
            "sft_runtime_mechanism/v2",
        )

    def test_unknown_contract_name_raises(self) -> None:
        with self.assertRaises(KeyError):
            registry.resolve_schema("NoSuchContract/v9")

    def test_each_contract_records_limitations(self) -> None:
        for contract in self.report["contracts"]:
            self.assertTrue(contract["known_limitations"])
            self.assertTrue(contract["guaranteed_fields"])

    def test_realising_artifacts_are_sha_bound_when_present(self) -> None:
        for contract in self.report["contracts"]:
            for item in contract["realising_artifacts"]:
                if item["exists"]:
                    self.assertEqual(len(item["sha256"]), 64)

    def test_registry_claims_no_modelling_authority(self) -> None:
        guarantees = self.report["guarantees"]
        self.assertFalse(guarantees["fits_or_publishes_coefficients"])
        self.assertFalse(guarantees["mutates_frozen_artifacts"])
        self.assertFalse(guarantees["creates_gpu_queue"])
        self.assertEqual(len(self.report["registry_sha256"]), 64)


class TestDomainReconciliation(unittest.TestCase):
    def setUp(self) -> None:
        if not recon.DEFAULT_POLICY.is_file():
            self.skipTest("frozen packing policy unavailable")
        self.report = recon.build_reconciliation()

    def test_reconciliation_changes_neither_domain(self) -> None:
        guarantees = self.report["guarantees"]
        self.assertFalse(guarantees["modifies_predictor_domain"])
        self.assertFalse(guarantees["modifies_packing_domain"])
        self.assertFalse(guarantees["widens_any_support_domain"])

    def test_the_three_known_conflicts_are_detected(self) -> None:
        self.assertEqual(
            set(self.report["conflicting_dimensions"]),
            {"cutoff_len", "model_id", "train_type"},
        )

    def test_16384_is_in_the_packing_range_but_off_the_predictor_grid(self) -> None:
        cutoff = next(
            item
            for item in self.report["dimensions"]
            if item["dimension"] == "cutoff_len"
        )
        self.assertEqual(cutoff["packing"]["maximum"], 16384)
        self.assertNotIn(16384, cutoff["predictor"])
        self.assertEqual(
            cutoff["packing_boundary_absent_from_predictor_grid"], [16384]
        )

    def test_32768_is_the_mirror_case(self) -> None:
        cutoff = next(
            item
            for item in self.report["dimensions"]
            if item["dimension"] == "cutoff_len"
        )
        # The predictor ranks 32768 but packing's released range stops at 16384.
        self.assertIn(32768, cutoff["predictor"])
        self.assertIn(32768, cutoff["predictor_only"])

    def test_joint_domain_is_the_intersection(self) -> None:
        joint = self.report["joint_supported_domain"]
        self.assertEqual(joint["cutoff_len"], [512, 2048, 4096, 8192])
        self.assertEqual(joint["train_type"], ["lora"])
        self.assertEqual(set(joint["model_id"]), {"qwen3_8b", "qwen3_14b"})

    def test_recommended_option_requires_no_new_evidence(self) -> None:
        recommended = [
            option
            for option in self.report["resolution_options"]
            if option.get("recommended")
        ]
        self.assertEqual(len(recommended), 1)
        self.assertFalse(recommended[0]["changes_evidence_requirements"])

    def test_widening_options_name_the_evidence_they_need(self) -> None:
        for option in self.report["resolution_options"]:
            if option["changes_evidence_requirements"]:
                self.assertIn("required_evidence", option)
                self.assertFalse(option["recommended"])

    def test_request_classification_matches_the_domains(self) -> None:
        inside = recon.classify_request(
            cutoff_len=4096,
            model_id="qwen3_8b",
            train_type="lora",
            reconciliation=self.report,
        )
        self.assertTrue(inside["jointly_supported"])
        self.assertEqual(inside["reasons"], [])

        full = recon.classify_request(
            cutoff_len=4096,
            model_id="qwen3_8b",
            train_type="full",
            reconciliation=self.report,
        )
        # Full may be ranked but never packed.
        self.assertTrue(full["memory_and_ranking_available"])
        self.assertFalse(full["packing_decision_available"])
        self.assertIn("outside_packing_release_domain", full["reasons"])

        long = recon.classify_request(
            cutoff_len=16384,
            model_id="qwen3_8b",
            train_type="lora",
            reconciliation=self.report,
        )
        self.assertFalse(long["memory_and_ranking_available"])
        self.assertFalse(long["jointly_supported"])


class TestVlEndpointReplay(unittest.TestCase):
    def setUp(self) -> None:
        if not vlreplay.DEFAULT_REPLAY.is_file():
            self.skipTest("memory replay artifact unavailable")
        self.report = vlreplay.build_report()

    def test_replay_is_explicitly_not_an_acceptance_set(self) -> None:
        role = self.report["evidence_role"]
        self.assertFalse(role["is_fresh_acceptance_set"])
        self.assertFalse(role["may_refit_coefficients"])
        self.assertFalse(role["may_publish"])

    def test_scope_records_that_input_was_text_only(self) -> None:
        scope = self.report["endpoint_scope"]
        self.assertEqual(scope["input_modality"], "text_only")
        self.assertEqual(
            scope["vision_tower_state"], "frozen_in_both_lora_and_full"
        )
        self.assertIn("floor", scope["consequence"])

    def test_endpoint_counts_match_the_known_evidence(self) -> None:
        totals = self.report["memory_heads"][vlreplay.SERVING_HEAD]["totals"]
        self.assertEqual(totals["success_rows"], 14)
        self.assertEqual(totals["oom_rows"], 4)

    def test_serving_head_admitted_configurations_that_oomed(self) -> None:
        totals = self.report["memory_heads"][vlreplay.SERVING_HEAD]["totals"]
        self.assertEqual(totals["false_safe_oom"], 2)
        self.assertEqual(totals["memory_admission_safety_failures"], 6)

    def test_under_prediction_is_one_sided_even_at_best_case(self) -> None:
        # Nested stat blocks made a naive sign check silently pass; the guard is
        # that even the least-negative endpoint is still negative.
        summary = self.report["serving_head_summary"]
        self.assertTrue(summary["all_slices_under_predict_even_at_best_case"])
        for value in summary["signed_best_case_percentage_errors"]:
            self.assertLess(value, 0.0)
        for value in summary["signed_mean_percentage_errors"]:
            self.assertLess(value, 0.0)

    def test_all_three_memory_heads_are_compared(self) -> None:
        self.assertEqual(len(self.report["memory_heads"]), 3)
        self.assertIn(vlreplay.SERVING_HEAD, self.report["memory_heads"])

    def test_findings_flag_the_visual_path_gap(self) -> None:
        self.assertIn(
            "text_only_endpoints_cannot_validate_the_visual_path",
            self.report["findings"],
        )
        self.assertIn(
            "vl_memory_is_systematically_under_predicted", self.report["findings"]
        )

    def test_report_claims_no_real_image_coverage(self) -> None:
        self.assertFalse(self.report["guarantees"]["claims_real_image_coverage"])
        self.assertFalse(
            self.report["guarantees"]["fits_or_publishes_coefficients"]
        )


if __name__ == "__main__":
    unittest.main()
