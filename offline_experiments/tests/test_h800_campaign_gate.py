from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from check_h800_campaign_gate import build_gate_report, _profile_content_check  # noqa: E402
from common import sha256_file
from prepare_h800_prospective_holdout import build_design  # noqa: E402


class H800CampaignGateTests(unittest.TestCase):
    def test_profile_content_check_rejects_malformed_or_incomplete_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "malformed.jsonl"
            malformed.write_text("not-json\n", encoding="utf-8")
            incomplete = root / "incomplete.jsonl"
            incomplete.write_text('{"sample_id":"s0"}\n', encoding="utf-8")
            valid = root / "valid.jsonl"
            valid.write_text(
                '{"sample_id":"s0","total_tokens":8,"label_tokens":2,"turns":2,"assistant_turns":1}\n',
                encoding="utf-8",
            )
            self.assertFalse(_profile_content_check(malformed, required_fields=("sample_id",))["passed"])
            self.assertFalse(
                _profile_content_check(
                    incomplete,
                    required_fields=("sample_id", "total_tokens", "label_tokens", "turns", "assistant_turns"),
                )["passed"]
            )
            self.assertTrue(
                _profile_content_check(
                    valid,
                    required_fields=("sample_id", "total_tokens", "label_tokens", "turns", "assistant_turns"),
                )["passed"]
            )

    def _design(self, root: Path) -> tuple[dict, Path]:
        design = build_design(
            hardware_probe={
                "selected_pool_idle": True,
                "exact_h800_pool": True,
                "required_gpu_ids": [4, 5, 6, 7],
                "missing_gpu_ids": [],
                "selected_gpu_rows": [],
            }
        )
        path = root / "design.json"
        path.write_text("{}\n", encoding="utf-8")
        # The gate only needs a stable design file hash; use a serialized copy
        # for the synthetic approval below.
        import json

        path.write_text(json.dumps(design, sort_keys=True) + "\n", encoding="utf-8")
        return design, path

    def test_current_environment_fails_closed_without_queue_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design, path = self._design(root)
            report = build_gate_report(
                design=design,
                design_path=path,
                approval={"approved": True, "resource_scope": {"gpu_ids": [0, 1, 2, 3]}},
                hardware_probe={
                    "exact_h800_pool": False,
                    "selected_pool_idle": False,
                    "required_gpu_ids": [4, 5, 6, 7],
                    "missing_gpu_ids": [4, 5, 6, 7],
                },
                root=root,
            )
            self.assertFalse(report["all_prerequisites_passed"])
            self.assertFalse(report["launch_allowed"])
            self.assertFalse(report["materialization_allowed"])
            self.assertTrue(any(not check["passed"] for check in report["checks"]))

    def test_exact_approval_still_requires_fresh_profile_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design, path = self._design(root)
            approval = {
                "approved": True,
                "design_sha256": sha256_file(path),
                "queue_binding_sha256": "a" * 64,
                "runtime_fingerprint_sha256": "b" * 64,
                "resource_scope": {"gpu_ids": [4, 5, 6, 7]},
            }
            report = build_gate_report(
                design=design,
                design_path=path,
                approval=approval,
                hardware_probe={
                    "exact_h800_pool": True,
                    "selected_pool_idle": True,
                    "required_gpu_ids": [4, 5, 6, 7],
                },
                root=root,
            )
            self.assertFalse(report["all_prerequisites_passed"])
            names = {check["name"]: check for check in report["checks"]}
            self.assertTrue(names["approval_binding"]["passed"])
            self.assertFalse(names["fresh_profiles"]["passed"])

    def test_approval_fingerprint_fields_must_be_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design, path = self._design(root)
            approval = {
                "approved": True,
                "design_sha256": sha256_file(path),
                "queue_binding_sha256": "not-a-hash",
                "runtime_fingerprint_sha256": "also-not-a-hash",
                "resource_scope": {"gpu_ids": [4, 5, 6, 7]},
            }
            report = build_gate_report(
                design=design,
                design_path=path,
                approval=approval,
                hardware_probe={
                    "exact_h800_pool": True,
                    "selected_pool_idle": True,
                    "required_gpu_ids": [4, 5, 6, 7],
                },
                root=root,
            )
            check = next(item for item in report["checks"] if item["name"] == "approval_binding")
            self.assertFalse(check["passed"])
            self.assertFalse(check["queue_binding_sha256_valid"])
            self.assertFalse(check["runtime_fingerprint_sha256_valid"])

    def test_filled_profile_requirements_can_bind_profiles_without_editing_design(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design, path = self._design(root)
            profile = root / "profile.jsonl"
            data = root / "data.jsonl"
            split = root / "split.json"
            profile.write_text(
                '{"sample_id":"s0","total_tokens":1,"label_tokens":1,"turns":1,"assistant_turns":1}\n',
                encoding="utf-8",
            )
            data.write_text("data\n", encoding="utf-8")
            split.write_text("split\n", encoding="utf-8")
            import json

            requirements = {
                "schema": "sft_h800_fresh_profile_requirements/v1",
                "campaign_id": design["campaign_id"],
                "design_binding": {"sha256": sha256_file(path)},
                "ready_for_materialization": True,
                "scenarios": [],
            }
            for scenario in design["scenarios"]:
                requirements["scenarios"].append(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "dataset_profile_id": scenario["dataset_profile_id"],
                        "model_id": scenario["model_id"],
                        "cutoff_len": scenario["cutoff_len"],
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
                )
            req_path = root / "requirements.json"
            req_path.write_text(json.dumps(requirements, sort_keys=True) + "\n", encoding="utf-8")
            approval = {
                "approved": True,
                "design_sha256": sha256_file(path),
                "queue_binding_sha256": "a" * 64,
                "runtime_fingerprint_sha256": "b" * 64,
                "resource_scope": {"gpu_ids": [4, 5, 6, 7]},
            }
            report = build_gate_report(
                design=design,
                design_path=path,
                approval=approval,
                hardware_probe={
                    "exact_h800_pool": True,
                    "selected_pool_idle": True,
                    "required_gpu_ids": [4, 5, 6, 7],
                },
                fresh_profile_requirements=requirements,
                fresh_profile_requirements_path=req_path,
                root=root,
            )
            names = {check["name"]: check for check in report["checks"]}
            self.assertTrue(names["fresh_profile_requirements"]["passed"])
            self.assertTrue(names["fresh_profiles"]["passed"])
            self.assertTrue(names["approval_binding"]["passed"])
            self.assertTrue(report["all_prerequisites_passed"])

    def test_rank_closeout_does_not_require_dataset_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            implementation = root / "implementation.py"
            implementation.write_text("# synthetic\n", encoding="utf-8")
            design = {
                "schema": "sft_h800_four_card_rank_closeout_design/v1",
                "campaign_id": "rank-closeout",
                "gpu_training_started": False,
                "queues_mutated": False,
                "required_gpu_pool": {"gpu_ids": [4, 5, 6, 7], "expected_name_contains": "H800"},
                "design_implementation": {
                    "path": str(implementation),
                    "sha256": sha256_file(implementation),
                },
            }
            path = root / "rank.json"
            import json

            path.write_text(json.dumps(design, sort_keys=True) + "\n", encoding="utf-8")
            report = build_gate_report(
                design=design,
                design_path=path,
                approval=None,
                hardware_probe={
                    "exact_h800_pool": False,
                    "selected_pool_idle": False,
                    "required_gpu_ids": [4, 5, 6, 7],
                },
                root=root,
            )
            names = {check["name"]: check for check in report["checks"]}
            self.assertTrue(names["fresh_profiles"]["passed"])
            self.assertTrue(names["source_fingerprints"]["passed"])

    def test_changed_source_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_text("before\n", encoding="utf-8")
            design = {
                "schema": "sft_h800_four_card_rank_closeout_design/v1",
                "campaign_id": "rank-closeout",
                "gpu_training_started": False,
                "queues_mutated": False,
                "required_gpu_pool": {"gpu_ids": [4, 5, 6, 7], "expected_name_contains": "H800"},
                "design_implementation": {
                    "path": str(source),
                    "sha256": sha256_file(source),
                },
            }
            source.write_text("after\n", encoding="utf-8")
            path = root / "rank.json"
            import json

            path.write_text(json.dumps(design, sort_keys=True) + "\n", encoding="utf-8")
            report = build_gate_report(
                design=design,
                design_path=path,
                approval=None,
                hardware_probe={
                    "exact_h800_pool": False,
                    "selected_pool_idle": False,
                    "required_gpu_ids": [4, 5, 6, 7],
                },
                root=root,
            )
            fingerprint = next(check for check in report["checks"] if check["name"] == "source_fingerprints")
            self.assertFalse(fingerprint["passed"])
            self.assertEqual(len(fingerprint["mismatched"]), 1)


if __name__ == "__main__":
    unittest.main()
