"""Tests for the Phase-B canary materializer and approval diff."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import materialize_phase_b_canary as mat  # noqa: E402


def _cell(**overrides):
    cell = {
        "cell_id": "text-unpacked-1gpu",
        "model_id": "qwen3_1p7b",
        "train_type": "lora",
        "dataset_id": "short_512",
        "cutoff_len": 512,
        "gpu_count": 1,
        "mbs": 2,
        "zero": "none",
        "gc": False,
        "packing": False,
        "target_gbs": 32,
        "purpose": "baseline",
    }
    cell.update(overrides)
    return cell


def _inventory():
    return {
        "models": [
            {
                "id": "qwen3_1p7b",
                "family": "qwen3",
                "path": "/models/Qwen3-1.7B",
                "tokenizer_path": "/models/Qwen3-1.7B",
                "template": "qwen3_nothink",
                "actual_parameters": 1_720_574_976,
            }
        ]
    }


class TestJobConstruction(unittest.TestCase):
    def test_unpacked_job_derives_ga_satisfying_the_gbs_contract(self) -> None:
        jobs, blockers = mat.build_jobs([_cell()], inventory=_inventory())
        self.assertEqual(blockers, [])
        job = jobs[0]
        self.assertEqual(
            job["gpu_count"] * job["mbs"] * job["gradient_accumulation_steps"],
            job["target_gbs"],
        )

    def test_packed_job_has_no_unpacked_ga(self) -> None:
        jobs, _ = mat.build_jobs(
            [_cell(cell_id="packed", dataset_id="multiturn_4096", packing=True, mbs=1)],
            inventory=_inventory(),
        )
        if not jobs:
            self.skipTest("dataset unavailable in this tree")
        self.assertNotIn("gradient_accumulation_steps", jobs[0])
        self.assertTrue(jobs[0]["packing"])

    def test_indivisible_gbs_is_blocked(self) -> None:
        jobs, blockers = mat.build_jobs(
            [_cell(cell_id="bad", mbs=3)], inventory=_inventory()
        )
        self.assertEqual(jobs, [])
        self.assertTrue(any(b.startswith("gbs_contract_violated") for b in blockers))

    def test_unknown_model_is_blocked_not_guessed(self) -> None:
        jobs, blockers = mat.build_jobs(
            [_cell(model_id="not_a_model")], inventory=_inventory()
        )
        self.assertEqual(jobs, [])
        self.assertIn("model_not_in_inventory:not_a_model", blockers)

    def test_every_job_is_marked_ineligible_for_calibration(self) -> None:
        jobs, _ = mat.build_jobs([_cell()], inventory=_inventory())
        for job in jobs:
            self.assertFalse(job["calibration_evidence_eligible"])
            self.assertEqual(job["evidence_role"], "consistency_only")
            self.assertTrue(job["hardware_bound_outputs_are_diagnostic"])

    def test_steps_stay_small(self) -> None:
        jobs, _ = mat.build_jobs([_cell()], inventory=_inventory())
        self.assertLessEqual(jobs[0]["measure_steps"], 3)
        self.assertEqual(jobs[0]["warmup_steps"], 0)


class TestProposedApproval(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs, _ = mat.build_jobs([_cell()], inventory=_inventory())
        self.current = {
            "schema_version": 1,
            "approved": True,
            "phase_id": "developer_h800_1p7b_to_14b",
            "allowed_job_ids": ["gen-a", "gen-b"],
            "provenance_sha256": "a" * 64,
            "runtime_fingerprint_sha256": "b" * 64,
            "runtime_patch_sha256": "c" * 64,
        }

    def test_tamper_check_fields_are_never_silently_dropped(self) -> None:
        # An approval that omits these would be weaker than the one it replaces.
        # Each must be present and must name where its value comes from, rather
        # than being null or a plausible-looking but unverifiable hash.
        result = mat.build_proposed_approval(
            self.jobs,
            queue_path=Path("/tmp/queue.jsonl"),
            current=self.current,
            provenance={"binding_available": False},
        )
        document = result["document"]
        for field in (
            "provenance_sha256",
            "runtime_fingerprint_sha256",
            "runtime_patch_sha256",
            "design_sha256",
        ):
            self.assertIn(field, document)
            value = str(document[field])
            self.assertTrue(value.startswith("<") and value.endswith(">"), field)
            self.assertNotEqual(value, "None", field)

    def test_valid_provenance_is_carried_into_the_proposal(self) -> None:
        result = mat.build_proposed_approval(
            self.jobs,
            queue_path=Path("/tmp/queue.jsonl"),
            current=self.current,
            provenance={
                "binding_available": True,
                "sha256": "d" * 64,
                "runtime_fingerprint_sha256": "e" * 64,
            },
        )
        document = result["document"]
        self.assertEqual(document["provenance_sha256"], "d" * 64)
        self.assertEqual(document["runtime_fingerprint_sha256"], "e" * 64)

    def test_delta_shows_job_id_replacement_explicitly(self) -> None:
        result = mat.build_proposed_approval(
            self.jobs,
            queue_path=Path("/tmp/queue.jsonl"),
            current=self.current,
            provenance={"binding_available": False},
        )
        entry = next(
            item
            for item in result["delta_vs_current"]
            if item["field"] == "allowed_job_ids"
        )
        self.assertEqual(entry["before_count"], 2)
        self.assertEqual(entry["after_count"], len(self.jobs))
        self.assertIn("not re-authorised", entry["note"])

    def test_proposal_records_that_runs_cannot_calibrate(self) -> None:
        result = mat.build_proposed_approval(
            self.jobs,
            queue_path=Path("/tmp/queue.jsonl"),
            current=self.current,
            provenance={"binding_available": False},
        )
        policy = result["document"]["evidence_policy"]
        self.assertFalse(policy["calibration_evidence_eligible"])
        self.assertFalse(policy["may_enter_memory_calibration"])
        self.assertFalse(policy["may_close_out_rank_validation"])

    def test_queue_binding_lists_ordered_ids_and_payload_hashes(self) -> None:
        result = mat.build_proposed_approval(
            self.jobs,
            queue_path=Path("/tmp/queue.jsonl"),
            current=self.current,
            provenance={"binding_available": False},
        )
        binding = result["document"]["queue_binding"]
        ids = [job["job_id"] for job in self.jobs]
        self.assertEqual(binding["ordered_job_ids"], ids)
        self.assertEqual(len(binding["ordered_job_payload_sha256"]), len(ids))
        self.assertEqual(sorted(binding["job_payload_sha256"]), sorted(ids))


class TestManifestContract(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = mat.build_manifest()

    def test_manifest_never_installs_or_launches(self) -> None:
        guarantees = self.manifest["guarantees"]
        self.assertFalse(guarantees["installs_approval"])
        self.assertFalse(guarantees["launches_training"])
        self.assertFalse(guarantees["marks_runs_as_calibration_evidence"])
        self.assertFalse(guarantees["refreshes_provenance"])
        self.assertFalse(guarantees["weakens_existing_tamper_checks"])
        self.assertFalse(self.manifest["launch_allowed"])

    def test_stale_provenance_is_surfaced_as_a_blocker(self) -> None:
        state = self.manifest["provenance_state"]
        if state.get("binding_available"):
            self.skipTest("provenance is currently valid in this tree")
        self.assertIn("provenance_binding_unavailable", self.manifest["blockers"])
        self.assertIn("refresh_command", state)

    def test_current_approval_does_not_cover_this_campaign(self) -> None:
        self.assertFalse(self.manifest["current_approval"]["covers_this_campaign"])
        self.assertIn(
            "canary_not_covered_by_current_approval", self.manifest["blockers"]
        )

    def test_hardware_bound_checks_are_listed_as_non_evidence(self) -> None:
        listed = self.manifest["hardware_bound_checks_not_evidence"]
        self.assertIn("memory_peak_recorded", listed)
        self.assertIn("step_timing_recorded", listed)

    def test_manifest_is_checksummed(self) -> None:
        self.assertEqual(self.manifest["schema"], mat.SCHEMA)
        self.assertEqual(len(self.manifest["manifest_sha256"]), 64)

    def test_queue_file_is_not_created_by_building_the_manifest(self) -> None:
        self.assertFalse(self.manifest["queue"]["exists"])


if __name__ == "__main__":
    unittest.main()
