"""Tests for the runtime cohort restore checklist."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import runtime_cohort_restore_checklist as restore  # noqa: E402


def _provenance(**identity_overrides):
    identity = {
        "container_runtime_id": "a" * 64,
        "nvidia_driver": "560.35.03",
        "launcher_patch_sha256": "b" * 64,
        "launcher_commits": {"finetuning_launcher": "c" * 40},
        "packages": {"deepspeed": "0.19.2", "torch": "2.8.0+cu126"},
        "framework_source_sha256": {
            "deepspeed_zero_partition_parameters": "d" * 64,
            "llamafactory_adapter": "e" * 64,
        },
        "project_source_snapshot_sha256": "f" * 64,
        "python": "3.11.11",
    }
    identity.update(identity_overrides)
    return {"runtime_identity": identity}


def _write(payload) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(payload, handle)
    handle.close()
    return Path(handle.name)


class TestSeverityClassification(unittest.TestCase):
    def test_missing_deepspeed_patch_is_blocking(self) -> None:
        baseline = _write(_provenance())
        current = _write(_provenance(launcher_patch_sha256=None))
        report = restore.build_checklist(
            baseline_path=baseline, current_path=current
        )
        entry = next(
            item
            for item in report["differences"]
            if item["field"] == "launcher_patch_sha256"
        )
        self.assertEqual(entry["severity"], restore.SEVERITY_BLOCKING)
        # The reason must cite the measured evidence, not just assert importance.
        self.assertIn("20/20 OOM", entry["why_it_matters"])

    def test_changed_deepspeed_source_hash_is_blocking(self) -> None:
        baseline = _write(_provenance())
        current = _write(
            _provenance(
                framework_source_sha256={
                    "deepspeed_zero_partition_parameters": "9" * 64,
                    "llamafactory_adapter": "e" * 64,
                }
            )
        )
        report = restore.build_checklist(
            baseline_path=baseline, current_path=current
        )
        entry = next(
            item
            for item in report["differences"]
            if item["field"] == "framework_source_sha256"
        )
        self.assertEqual(entry["severity"], restore.SEVERITY_BLOCKING)
        self.assertIn(
            "deepspeed_zero_partition_parameters", entry["sub_differences"]
        )

    def test_container_id_alone_is_benign(self) -> None:
        # It rotates on every pod restart, so treating it as blocking would make
        # the checklist cry wolf.
        baseline = _write(_provenance())
        current = _write(_provenance(container_runtime_id="9" * 64))
        report = restore.build_checklist(
            baseline_path=baseline, current_path=current
        )
        entry = next(
            item
            for item in report["differences"]
            if item["field"] == "container_runtime_id"
        )
        self.assertEqual(entry["severity"], restore.SEVERITY_BENIGN)
        self.assertEqual(report["difference_counts"][restore.SEVERITY_BLOCKING], 0)

    def test_project_source_drift_is_benign(self) -> None:
        # This repo's own sources changed because analysis modules were added on
        # purpose; that is not an execution-environment change.
        baseline = _write(_provenance())
        current = _write(_provenance(project_source_snapshot_sha256="9" * 64))
        report = restore.build_checklist(
            baseline_path=baseline, current_path=current
        )
        entry = next(
            item
            for item in report["differences"]
            if item["field"] == "project_source_snapshot_sha256"
        )
        self.assertEqual(entry["severity"], restore.SEVERITY_BENIGN)

    def test_driver_change_needs_review(self) -> None:
        baseline = _write(_provenance())
        current = _write(_provenance(nvidia_driver="580.82.07"))
        report = restore.build_checklist(
            baseline_path=baseline, current_path=current
        )
        entry = next(
            item
            for item in report["differences"]
            if item["field"] == "nvidia_driver"
        )
        self.assertEqual(entry["severity"], restore.SEVERITY_REVIEW)

    def test_unknown_field_defaults_to_review_not_benign(self) -> None:
        baseline = _write(_provenance(some_new_field="x"))
        current = _write(_provenance(some_new_field="y"))
        report = restore.build_checklist(
            baseline_path=baseline, current_path=current
        )
        entry = next(
            item for item in report["differences"] if item["field"] == "some_new_field"
        )
        self.assertEqual(entry["severity"], restore.SEVERITY_REVIEW)


class TestChecklistContract(unittest.TestCase):
    def setUp(self) -> None:
        baseline = _write(_provenance())
        current = _write(
            _provenance(launcher_patch_sha256=None, nvidia_driver="580.82.07")
        )
        self.report = restore.build_checklist(
            baseline_path=baseline, current_path=current
        )

    def test_identical_environment_yields_no_differences(self) -> None:
        same = _write(_provenance())
        report = restore.build_checklist(baseline_path=same, current_path=same)
        self.assertEqual(report["differences"], [])
        self.assertEqual(
            report["difference_counts"][restore.SEVERITY_BLOCKING], 0
        )

    def test_unchanged_fields_are_reported_too(self) -> None:
        # A short restore list is only credible if the matching surface is stated.
        self.assertTrue(self.report["unchanged_identity_fields"])
        self.assertIn("packages", self.report["unchanged_identity_fields"])

    def test_decision_is_restore_not_new_cohort(self) -> None:
        self.assertEqual(
            self.report["decision"],
            "restore_original_cohort_rather_than_open_a_new_one",
        )

    def test_checklist_restores_nothing_itself(self) -> None:
        guarantees = self.report["guarantees"]
        self.assertFalse(guarantees["restores_anything"])
        self.assertFalse(guarantees["modifies_runtime"])
        self.assertFalse(guarantees["fits_or_publishes_coefficients"])
        self.assertFalse(guarantees["creates_gpu_queue"])

    def test_phase_b_is_blocked_until_restore(self) -> None:
        until = self.report["until_restored"]
        self.assertFalse(until["phase_b_canary_may_run"])
        self.assertFalse(until["calibration_or_acceptance_may_run"])
        # Phase B is hardware-independent but not mechanism-independent.
        self.assertIn("execution semantics", until["reason"])

    def test_acceptance_puts_capture_provenance_last(self) -> None:
        acceptance = self.report["acceptance_of_restore"]
        self.assertIn("last step", acceptance["note"])
        self.assertTrue(
            any("live_runtime_patch" in check for check in acceptance["checks"])
        )

    def test_restore_targets_are_all_blocking_and_actionable(self) -> None:
        targets = self.report["restore_targets"]
        self.assertTrue(targets)
        for target in targets:
            self.assertTrue(target["blocking"])
            self.assertIn("item", target)
        names = {target["item"] for target in targets}
        self.assertIn("deepspeed_mixed_dtype_patch_file", names)
        self.assertIn("gpu_pool_identity", names)

    def test_checklist_is_checksummed(self) -> None:
        self.assertEqual(self.report["schema"], restore.SCHEMA)
        self.assertEqual(len(self.report["checklist_sha256"]), 64)


class TestRealEnvironment(unittest.TestCase):
    def test_live_tree_reports_the_actual_patch_state_and_hash(self) -> None:
        if not restore.DEFAULT_BASELINE.is_file():
            self.skipTest("pre-drift baseline unavailable")
        report = restore.build_checklist()
        patch = next(
            target
            for target in report["restore_targets"]
            if target["item"] == "deepspeed_mixed_dtype_patch_file"
        )
        self.assertEqual(patch["present_now"], restore.DEEPSPEED_PATCH.is_file())
        self.assertEqual(
            patch["current_sha256"],
            (
                restore.sha256_file(restore.DEEPSPEED_PATCH)
                if restore.DEEPSPEED_PATCH.is_file()
                else None
            ),
        )
        self.assertGreaterEqual(
            report["difference_counts"][restore.SEVERITY_BLOCKING], 1
        )


if __name__ == "__main__":
    unittest.main()
