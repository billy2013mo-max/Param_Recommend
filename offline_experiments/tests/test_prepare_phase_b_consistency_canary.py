"""Tests for the Phase-B consistency canary design."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prepare_phase_b_consistency_canary as phase_b  # noqa: E402


def _probe(
    *,
    names=("NVIDIA H800",),
    missing=(),
    processes=(),
    exact_h800=True,
):
    rows = [
        {
            "index": index,
            "uuid": f"GPU-{index:08d}-0000-0000-0000-000000000000",
            "name": names[index % len(names)],
            "memory_total_mib": 143771.0,
        }
        for index in range(4)
    ]
    return {
        "required_gpu_ids": [0, 1, 2, 3],
        "expected_gpu_name": "H800",
        "selected_gpu_rows": rows,
        "missing_gpu_ids": list(missing),
        "selected_gpu_compute_processes": list(processes),
        "exact_h800_pool": exact_h800,
        "selected_pool_idle": exact_h800 and not processes,
    }


class TestHardwarePolicy(unittest.TestCase):
    def test_consistency_verdicts_stay_valid_on_a_mismatched_pool(self) -> None:
        design = phase_b.build_design(
            hardware_probe=_probe(names=("NVIDIA H200",), exact_h800=False)
        )
        policy = design["hardware_policy"]
        self.assertFalse(policy["pool_matches_frozen_family"])
        # The entire point of this phase: a mismatched SKU does not invalidate a
        # prediction-vs-observation comparison made on the same run.
        self.assertTrue(policy["consistency_verdicts_valid_on_this_pool"])

    def test_mismatched_pool_still_forbids_calibration_entry(self) -> None:
        design = phase_b.build_design(
            hardware_probe=_probe(names=("NVIDIA H200",), exact_h800=False)
        )
        policy = design["hardware_policy"]
        self.assertFalse(policy["runs_may_enter_memory_calibration"])
        self.assertFalse(policy["runs_may_enter_throughput_calibration"])
        self.assertFalse(policy["runs_may_close_out_rank_validation"])

    def test_occupancy_is_assessed_separately_from_sku(self) -> None:
        # probe_hardware folds "exact H800" into selected_pool_idle; Phase B must
        # not inherit that, or an idle non-H800 pool looks busy.
        design = phase_b.build_design(
            hardware_probe=_probe(names=("NVIDIA H200",), exact_h800=False)
        )
        policy = design["hardware_policy"]
        self.assertTrue(policy["occupancy_and_sku_assessed_separately"])
        self.assertTrue(policy["pool_idle"])
        self.assertNotIn("gpu_pool_not_idle", design["blockers"])

    def test_busy_pool_is_a_blocker(self) -> None:
        design = phase_b.build_design(
            hardware_probe=_probe(processes=[{"pid": 1, "gpu_uuid": "x"}])
        )
        self.assertFalse(design["hardware_policy"]["pool_idle"])
        self.assertIn("gpu_pool_not_idle", design["blockers"])

    def test_missing_gpu_is_a_blocker(self) -> None:
        design = phase_b.build_design(hardware_probe=_probe(missing=(2, 3)))
        self.assertFalse(design["hardware_policy"]["pool_complete"])
        self.assertIn("required_gpu_indices_missing", design["blockers"])

    def test_observed_names_are_reported_verbatim(self) -> None:
        design = phase_b.build_design(
            hardware_probe=_probe(names=("NVIDIA H200",), exact_h800=False)
        )
        self.assertEqual(
            design["hardware_policy"]["observed_gpu_names"], ["NVIDIA H200"]
        )


class TestCheckClassification(unittest.TestCase):
    def setUp(self) -> None:
        self.design = phase_b.build_design(hardware_probe=_probe())

    def test_every_check_is_classified(self) -> None:
        for check in self.design["checks"]:
            self.assertIn(
                check["classification"],
                {phase_b.CLASS_HARDWARE_INDEPENDENT, phase_b.CLASS_HARDWARE_BOUND},
            )
            self.assertIn("compares", check)
            self.assertIn("rationale", check)

    def test_memory_and_timing_are_hardware_bound_and_not_evidence(self) -> None:
        bound = {
            check["check_id"]: check
            for check in self.design["checks"]
            if check["classification"] == phase_b.CLASS_HARDWARE_BOUND
        }
        self.assertIn("memory_peak_recorded", bound)
        self.assertIn("step_timing_recorded", bound)
        for check in bound.values():
            self.assertFalse(check["evidence_eligible"])

    def test_semantic_checks_are_hardware_independent(self) -> None:
        independent = {
            check["check_id"]
            for check in self.design["checks"]
            if check["classification"] == phase_b.CLASS_HARDWARE_INDEPENDENT
        }
        for expected in (
            "packed_batch_loss_mask_correct",
            "cross_sample_attention_isolated",
            "computed_tokens_match_event_log",
            "per_rank_inventory_consistent",
            "expected_gbs_reproduced",
        ):
            self.assertIn(expected, independent)

    def test_cpu_only_checks_are_identified(self) -> None:
        cpu_only = set(self.design["cpu_only_checks_already_passing"])
        self.assertIn("tokenized_length_matches_profile", cpu_only)
        self.assertIn("static_packing_matches_processor", cpu_only)
        for check in self.design["checks"]:
            if check["check_id"] in cpu_only:
                self.assertFalse(check["requires_training_run"])

    def test_freeze_flag_check_exists_because_train_type_is_insufficient(self) -> None:
        checks = {check["check_id"]: check for check in self.design["checks"]}
        self.assertIn("freeze_flags_match_declaration", checks)
        self.assertIn("vision tower", checks["freeze_flags_match_declaration"]["rationale"])


class TestCells(unittest.TestCase):
    def setUp(self) -> None:
        self.design = phase_b.build_design(hardware_probe=_probe())

    def test_cells_cover_packed_and_unpacked_and_multi_gpu(self) -> None:
        cells = self.design["cells"]
        self.assertTrue(any(cell["packing"] for cell in cells))
        self.assertTrue(any(not cell["packing"] for cell in cells))
        self.assertTrue(any(cell["gpu_count"] > 1 for cell in cells))

    def test_packed_cell_fixes_mbs_to_one(self) -> None:
        for cell in self.design["cells"]:
            if cell["packing"]:
                self.assertEqual(cell["mbs"], 1)

    def test_every_cell_satisfies_the_gbs_contract(self) -> None:
        for cell in self.design["cells"]:
            if cell["packing"]:
                continue
            per_step = cell["gpu_count"] * cell["mbs"]
            self.assertEqual(cell["target_gbs"] % per_step, 0)

    def test_cells_are_bounded_probes_not_a_matrix(self) -> None:
        self.assertLessEqual(len(self.design["cells"]), 6)
        self.assertLessEqual(
            self.design["approval_request"]["maximum_steps_per_job"], 3
        )

    def test_vl_cell_does_not_claim_an_observed_visual_path(self) -> None:
        vl_cells = [
            cell for cell in self.design["cells"] if "vl" in cell["model_id"]
        ]
        self.assertTrue(vl_cells)
        for cell in vl_cells:
            self.assertFalse(cell["expects_visual_path_observed"])

    def test_vl_can_be_excluded(self) -> None:
        design = phase_b.build_design(
            hardware_probe=_probe(), include_vl=False
        )
        self.assertFalse(
            any("vl" in cell["model_id"] for cell in design["cells"])
        )
        self.assertFalse(
            any(check.get("vl_only") for check in design["checks"])
        )


class TestApprovalRequest(unittest.TestCase):
    def setUp(self) -> None:
        self.design = phase_b.build_design(hardware_probe=_probe())

    def test_design_does_not_grant_its_own_approval(self) -> None:
        self.assertFalse(self.design["launch_allowed"])
        self.assertFalse(self.design["guarantees"]["grants_own_approval"])
        self.assertFalse(self.design["guarantees"]["creates_gpu_queue"])
        self.assertFalse(self.design["guarantees"]["writes_job_jsonl"])

    def test_existing_approval_does_not_cover_the_canary(self) -> None:
        request = self.design["approval_request"]
        current = request["current_approval"]
        if not current.get("present"):
            self.skipTest("approval file unavailable")
        # The live approval enumerates finished jobs from another phase, so it
        # cannot authorise new canary cells.
        self.assertFalse(current["covers_canary"])
        self.assertTrue(request["approval_action_required"])
        self.assertIn("canary_not_covered_by_current_approval", self.design["blockers"])

    def test_request_states_it_never_enters_calibration(self) -> None:
        request = self.design["approval_request"]
        self.assertFalse(request["enters_calibration"])
        self.assertFalse(request["writes_frozen_artifacts"])
        self.assertFalse(request["enables_scale_out"])

    def test_request_enumerates_exactly_the_designed_cells(self) -> None:
        request = self.design["approval_request"]
        self.assertEqual(
            sorted(request["requested_job_ids"]),
            sorted(cell["cell_id"] for cell in self.design["cells"]),
        )
        self.assertEqual(request["total_jobs"], len(self.design["cells"]))

    def test_design_is_checksummed_and_binds_sources(self) -> None:
        self.assertEqual(self.design["schema"], phase_b.SCHEMA)
        self.assertEqual(len(self.design["design_sha256"]), 64)
        self.assertTrue(self.design["source_bindings"])


class TestRealProbe(unittest.TestCase):
    def test_design_builds_against_the_live_pool(self) -> None:
        design = phase_b.build_design()
        self.assertEqual(design["status"], "design_pending_approval")
        self.assertFalse(design["launch_allowed"])
        # Whatever the live pool is, calibration entry must stay closed.
        self.assertFalse(
            design["hardware_policy"]["runs_may_enter_memory_calibration"]
        )


if __name__ == "__main__":
    unittest.main()
