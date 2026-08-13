from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from materialize_h800_prospective_holdout import (  # noqa: E402
    approval_binds_queue,
    build_queue_manifest,
    queue_binding_sha256,
)
from common import sha256_file  # noqa: E402


def _design(profile: Path, data: Path, *, registered: bool) -> dict:
    return {
        "schema": "sft_h800_prospective_holdout_design/v1",
        "campaign_id": "test-campaign",
        "gpu_training_started": False,
        "queues_mutated": False,
        "materialization_allowed": True,
        "scenarios": [
            {
                "scenario_id": "s1",
                "model_id": "m1",
                "dataset_profile_id": "p1",
                "cutoff_len": 512,
                "freshness": {
                    "profile_path": str(profile),
                    "data_path": str(data),
                    "runtime_dataset_registered": registered,
                },
            }
        ],
        "candidate_slots": [
            {
                "candidate_slot_id": "slot-1",
                "scenario_id": "s1",
                "model_id": "m1",
                "dataset_profile_id": "p1",
                "target_gbs": 64,
                "cutoff_len": 512,
                "gpu_count": 1,
                "zero_stage": 0,
                "physical_mbs": 16,
                "gradient_checkpointing": True,
                "packing": False,
                "offload": False,
            }
        ],
    }


class MaterializeH800ProspectiveTests(unittest.TestCase):
    def test_queue_binding_requires_exact_promoted_approval(self) -> None:
        jobs = [{"job_id": "j1", "scenario_id": "s1"}]
        binding = queue_binding_sha256(jobs)
        self.assertEqual(
            approval_binds_queue(
                {"approved": True, "queue_binding_sha256": binding},
                binding,
            ),
            (True, "approval_queue_binding_matches"),
        )
        self.assertEqual(
            approval_binds_queue(
                {"approved": True, "queue_binding_sha256": "a" * 64},
                binding,
            ),
            (False, "approval_queue_binding_mismatch"),
        )
        self.assertEqual(
            approval_binds_queue(
                {"approved": True, "queue_binding_sha256": "not-a-hash"},
                binding,
            ),
            (False, "approval_queue_binding_invalid"),
        )

    def test_current_gate_refuses_materialization(self) -> None:
        import json

        root = Path(__file__).resolve().parents[1]
        design = json.loads((root / "artifacts/h800_fresh_holdout_design_v1.json").read_text())
        gate = json.loads((root / "artifacts/h800_campaign_gate_report_v1.json").read_text())
        inventory = json.loads((root / "artifacts/model_inventory.json").read_text())
        manifest = build_queue_manifest(
            design=design,
            gate_report=gate,
            model_inventory=inventory,
            root=root,
        )
        self.assertFalse(manifest["materialization_allowed"])
        self.assertFalse(manifest["launch_allowed"])
        self.assertIn("campaign_gate_not_passed", manifest["blockers"])

    def test_profile_and_data_are_not_enough_without_runtime_registration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.jsonl"
            data = root / "data.jsonl"
            profile.write_text("{}\n", encoding="utf-8")
            data.write_text("{}\n", encoding="utf-8")
            manifest = build_queue_manifest(
                design=_design(profile, data, registered=False),
                gate_report={
                    "schema": "sft_h800_campaign_gate_report/v1",
                    "all_prerequisites_passed": True,
                },
                model_inventory={"models": [{"id": "m1", "path": "/model", "tokenizer_path": "/tok"}]},
                root=root,
            )
            self.assertFalse(manifest["materialization_allowed"])
            self.assertIn("runtime_dataset_registration_missing:s1", manifest["blockers"])

    def test_registered_bindings_produce_non_launching_job_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.jsonl"
            data = root / "data.jsonl"
            profile.write_text("{}\n", encoding="utf-8")
            data.write_text("{}\n", encoding="utf-8")
            manifest = build_queue_manifest(
                design=_design(profile, data, registered=True),
                gate_report={
                    "schema": "sft_h800_campaign_gate_report/v1",
                    "all_prerequisites_passed": True,
                },
                model_inventory={"models": [{"id": "m1", "path": "/model", "tokenizer_path": "/tok"}]},
                root=root,
            )
            self.assertTrue(manifest["materialization_allowed"])
            self.assertFalse(manifest["launch_allowed"])
            self.assertFalse(manifest["gpu_training_started"])
            self.assertEqual(manifest["candidate_count"], 1)
            self.assertEqual(manifest["jobs"][0]["execution_state"], "awaiting_exact_promoted_approval")

    def test_requirements_contract_supplies_bindings_without_design_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.jsonl"
            data = root / "data.jsonl"
            split = root / "split.json"
            profile.write_text("{}\n", encoding="utf-8")
            data.write_text("{}\n", encoding="utf-8")
            split.write_text("{}\n", encoding="utf-8")
            design = _design(profile, data, registered=True)
            design["scenarios"][0]["freshness"] = {}
            requirements = {
                "schema": "sft_h800_fresh_profile_requirements/v1",
                "campaign_id": design["campaign_id"],
                "ready_for_materialization": True,
                "scenarios": [
                    {
                        "scenario_id": "s1",
                        "dataset_profile_id": "p1",
                        "model_id": "m1",
                        "cutoff_len": 512,
                        "required_bindings": {
                            "profile_path": str(profile),
                            "profile_sha256": sha256_file(profile),
                            "data_path": str(data),
                            "data_sha256": sha256_file(data),
                            "runtime_dataset_registered": True,
                            "processor_contract_sha256": "a" * 64,
                            "split_manifest_sha256": sha256_file(split),
                        },
                    }
                ],
            }
            manifest = build_queue_manifest(
                design=design,
                gate_report={
                    "schema": "sft_h800_campaign_gate_report/v1",
                    "all_prerequisites_passed": True,
                },
                model_inventory={"models": [{"id": "m1", "path": "/model", "tokenizer_path": "/tok"}]},
                fresh_profile_requirements=requirements,
                root=root,
            )
            self.assertTrue(manifest["materialization_allowed"])
            self.assertEqual(manifest["jobs"][0]["dataset_profile_path"]["path"], str(profile.resolve()))


if __name__ == "__main__":
    unittest.main()
