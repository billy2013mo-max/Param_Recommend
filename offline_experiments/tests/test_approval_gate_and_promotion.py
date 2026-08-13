from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import promote_approval_candidate as promotion  # noqa: E402
import run_job  # noqa: E402
from approval_gate import (  # noqa: E402
    acquire_execution_lock,
    build_provenance_binding,
    build_queue_binding,
    expected_provenance_source_paths,
    retired_approval_plan_schemas,
)
from common import sha256_file, sha256_json, write_json, write_jsonl  # noqa: E402


class ApprovalFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.runtime_identity = {
            "schema_version": 1,
            "python_executable": "/test/venv/bin/python",
            "framework_source_sha256": {"deepspeed_zero_partition_parameters": "ds"},
            "launcher_patch_sha256": "patcher",
        }
        self.runtime_patch = {
            "installed_deepspeed_source_sha256": "ds",
            "runtime_patcher_sha256": "patcher",
            "all_passed": True,
        }
        self.job = {
            "job_id": "tputscreen-test",
            "request_id": "tput-request",
            "kind": "throughput_screen",
            "fidelity": "screen",
            "model_id": "qwen3_8b",
            "gpu_count": 2,
            "mbs": 1,
            "target_gbs": 16,
            "zero": "zero3",
            "gc": False,
            "packing": False,
            "repeat": 0,
            "warmup_steps": 1,
            "measure_steps": 2,
        }
        self.experiment = {
            "training_scope": {
                "gpu_ids": [1, 2, 3, 4],
                "gpu_counts": [1, 2, 4],
                "max_gpu_count": 4,
                "exclusive_node_gpu_ids": [1, 2, 3, 4],
            },
            "measurement": {"performance_parallelism": "disjoint_gpu_masks"},
        }
        write_json(root / "config" / "experiment.json", self.experiment)
        (root / "scripts").mkdir(parents=True)
        (root / "scripts" / "launcher.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / "README.md").write_text("readme\n", encoding="utf-8")
        (root / "EXPERIMENT_DESIGN.md").write_text("design\n", encoding="utf-8")
        self.queue_path = root / "audit" / "delta.jsonl"
        write_jsonl(self.queue_path, [self.job])
        self._write_fresh_provenance()
        self.queue_binding = build_queue_binding(self.queue_path, [self.job], root)
        self.provenance_binding = build_provenance_binding(root)
        self.design = self._build_design()
        self.candidate_path = root / "audit" / "approval-design-candidate.json"
        write_json(self.candidate_path, self.design)
        self.candidate_bytes = self.candidate_path.read_bytes()
        self.candidate_sha256 = sha256_file(self.candidate_path)
        self.design_path = root / "runtime" / "approval_design.json"
        self.design_path.parent.mkdir(parents=True, exist_ok=True)
        self.design_path.write_bytes(self.candidate_bytes)
        self.approval_path = root / "config" / "APPROVED_TO_RUN.json"
        self.receipt_path = root / "runtime" / "fixture-receipt.json"
        self.approval = promotion.build_approval_payload(
            design=self.design,
            candidate_sha256=self.candidate_sha256,
            project_root=root,
            authorization="unit-test authorization",
            approved_by="unit-test",
            transaction_id="fixture",
            receipt_path=self.receipt_path,
            created_unix=1.0,
        )
        write_json(self.approval_path, self.approval)

    def _write_fresh_provenance(self) -> None:
        paths = expected_provenance_source_paths(self.root)
        manifest = {
            str(path.relative_to(self.root)): sha256_file(path) for path in paths
        }
        identity = {
            "schema_version": 1,
            "project_source_snapshot_sha256": sha256_json(manifest),
            "environment": "test",
        }
        write_json(
            self.root / "artifacts" / "provenance.json",
            {
                "schema_version": 1,
                "runtime_identity": identity,
                "runtime_fingerprint_sha256": sha256_json(identity),
                "project_source_manifest": manifest,
                "reproducibility_identity_complete": True,
            },
        )

    def _build_design(self) -> dict:
        manifest_paths = [
            *expected_provenance_source_paths(self.root),
            self.queue_path,
            self.root / "artifacts" / "provenance.json",
        ]
        manifest = {
            str(path.relative_to(self.root)): sha256_file(path)
            for path in sorted(set(manifest_paths))
        }
        stage = {
            "jobs": 1,
            "allowed_job_ids": [self.job["job_id"]],
            "queue_path": self.queue_binding["path"],
            "queue_sha256": self.queue_binding["sha256"],
            "ordered_job_ids": self.queue_binding["ordered_job_ids"],
            "ordered_job_payload_sha256": self.queue_binding[
                "ordered_job_payload_sha256"
            ],
            "job_payload_sha256": self.queue_binding["job_payload_sha256"],
            "policy": "exact test queue",
        }
        return {
            "schema_version": 1,
            "training_started": False,
            "file_sha256": manifest,
            "execution_order": ["throughput_screen_delta"],
            "allowed_job_ids": [self.job["job_id"]],
            "authorized_gpu_ids": [1, 2, 3, 4],
            "max_gpu_count": 4,
            "runtime_identity": self.runtime_identity,
            "runtime_fingerprint_sha256": sha256_json(self.runtime_identity),
            "runtime_patch": self.runtime_patch,
            "runtime_fix": {"commit": "test"},
            "queue_binding": self.queue_binding,
            "provenance_binding": self.provenance_binding,
            "throughput_screen_delta": stage,
        }


class ApprovalGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = ApprovalFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def verify(self, job: dict | None = None, **kwargs):
        fixture = self.fixture
        with (
            patch.object(run_job, "ROOT", fixture.root),
            patch.object(run_job, "CONFIG_DIR", fixture.root / "config"),
            patch.object(run_job, "RUNTIME_DIR", fixture.root / "runtime"),
            patch.object(run_job, "APPROVAL_FILE", fixture.approval_path),
            patch.object(
                run_job,
                "live_runtime_identity",
                return_value=fixture.runtime_identity,
            ),
            patch.object(
                run_job,
                "live_runtime_patch",
                return_value=fixture.runtime_patch,
            ),
        ):
            return run_job.verify_approval(job, **kwargs)

    def test_exact_job_and_queue_are_approved(self) -> None:
        result = self.verify(
            self.fixture.job,
            queue_path=self.fixture.queue_path,
            queue_rows=[self.fixture.job],
        )
        self.assertTrue(result["queue"]["all_passed"])
        self.assertTrue(result["provenance"]["all_passed"])

    def test_same_job_id_with_changed_payload_is_rejected(self) -> None:
        changed = {**self.fixture.job, "mbs": 2}
        with self.assertRaisesRegex(PermissionError, "canonical payload"):
            self.verify(changed)

    def test_retired_frozen_design_is_rejected_before_execution(self) -> None:
        design = {
            **self.fixture.design,
            "h800_calibration": {
                "schema": "sft_h800_calibration_approval_freeze/v1"
            },
        }
        write_json(self.fixture.design_path, design)
        approval = {
            **self.fixture.approval,
            "design_sha256": sha256_file(self.fixture.design_path),
        }
        write_json(self.fixture.approval_path, approval)

        with self.assertRaisesRegex(PermissionError, "Retired H800 calibration"):
            self.verify(self.fixture.job)

    def test_empty_or_mismatched_approval_ids_are_rejected(self) -> None:
        approval = {**self.fixture.approval, "allowed_job_ids": []}
        write_json(self.fixture.approval_path, approval)
        with self.assertRaisesRegex(PermissionError, "allowed_job_ids"):
            self.verify(self.fixture.job)

    def test_alternate_path_reorder_and_subset_are_rejected(self) -> None:
        second = {**self.fixture.job, "job_id": "other"}
        alternate = self.fixture.root / "audit" / "alternate.jsonl"
        write_jsonl(alternate, [self.fixture.job])
        with self.assertRaisesRegex(PermissionError, "bound queue"):
            self.verify(queue_path=alternate, queue_rows=[self.fixture.job])
        with self.assertRaisesRegex(PermissionError, "bound queue"):
            self.verify(queue_path=self.fixture.queue_path, queue_rows=[second])

    def test_live_runtime_patch_and_provenance_drift_are_rejected(self) -> None:
        with self.assertRaisesRegex(PermissionError, "runtime identity"):
            fixture = self.fixture
            with (
                patch.object(run_job, "ROOT", fixture.root),
                patch.object(run_job, "CONFIG_DIR", fixture.root / "config"),
                patch.object(run_job, "RUNTIME_DIR", fixture.root / "runtime"),
                patch.object(run_job, "APPROVAL_FILE", fixture.approval_path),
                patch.object(
                    run_job, "live_runtime_identity", return_value={"stale": True}
                ),
            ):
                run_job.verify_approval(fixture.job)

        with self.assertRaisesRegex(PermissionError, "runtime patch"):
            fixture = self.fixture
            with (
                patch.object(run_job, "ROOT", fixture.root),
                patch.object(run_job, "CONFIG_DIR", fixture.root / "config"),
                patch.object(run_job, "RUNTIME_DIR", fixture.root / "runtime"),
                patch.object(run_job, "APPROVAL_FILE", fixture.approval_path),
                patch.object(
                    run_job,
                    "live_runtime_identity",
                    return_value=fixture.runtime_identity,
                ),
                patch.object(
                    run_job,
                    "live_runtime_patch",
                    return_value={"all_passed": False},
                ),
            ):
                run_job.verify_approval(fixture.job)

        (self.fixture.root / "scripts" / "new_unrecorded.py").write_text(
            "VALUE = 2\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(PermissionError, "provenance"):
            self.verify(self.fixture.job)

    def test_execute_rejects_before_render_or_result_artifacts(self) -> None:
        fixture = self.fixture
        job_path = fixture.root / "job.json"
        write_json(job_path, fixture.job)
        result_root = fixture.root / "results-before-gate"

        def reject_while_shared_lock_is_held(*args, **kwargs):
            with self.assertRaisesRegex(PermissionError, "gate is busy"):
                acquire_execution_lock(fixture.root / "runtime", exclusive=True)
            raise PermissionError("gate rejected")

        with (
            patch.object(
                sys,
                "argv",
                [
                    "run_job.py",
                    "--job-file",
                    str(job_path),
                    "--gpu-mask",
                    "1,2",
                    "--execute",
                ],
            ),
            patch.object(run_job, "CONFIG_DIR", fixture.root / "config"),
            patch.object(run_job, "RUNTIME_DIR", fixture.root / "runtime"),
            patch.object(run_job, "RESULTS_DIR", result_root),
            patch.object(run_job, "validate_gpu_assignment"),
            patch.object(
                run_job,
                "verify_approval",
                side_effect=reject_while_shared_lock_is_held,
            ),
            patch.object(run_job, "render_config") as render,
        ):
            with self.assertRaises(PermissionError):
                run_job.main()
        render.assert_not_called()
        self.assertFalse(result_root.exists())
        released = acquire_execution_lock(fixture.root / "runtime", exclusive=True)
        released.close()

    def test_shared_execution_lock_blocks_exclusive_promotion_lock(self) -> None:
        handle = acquire_execution_lock(self.fixture.root / "runtime", exclusive=False)
        try:
            with self.assertRaisesRegex(PermissionError, "gate is busy"):
                acquire_execution_lock(self.fixture.root / "runtime", exclusive=True)
        finally:
            handle.close()


class PromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = ApprovalFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def promote(self, *, promote: bool, process_probe=lambda: [], **kwargs):
        fixture = self.fixture
        return promotion.promote_candidate(
            candidate_path=fixture.candidate_path,
            expected_candidate_sha256=kwargs.pop(
                "expected_candidate_sha256", fixture.candidate_sha256
            ),
            project_root=fixture.root,
            authorization="explicit unit-test approval",
            approved_by="unit-test",
            promote=promote,
            process_probe=process_probe,
            current_runtime_identity=fixture.runtime_identity,
            current_runtime_patch=fixture.runtime_patch,
            created_unix=2.0,
            **kwargs,
        )

    def test_default_dry_run_does_not_change_live_files(self) -> None:
        old_design = self.fixture.design_path.read_bytes()
        old_approval = self.fixture.approval_path.read_bytes()
        report = self.promote(promote=False)
        self.assertTrue(report["all_passed"])
        self.assertTrue(report["checks"]["candidate_not_retired"])
        self.assertFalse(report["promoted"])
        self.assertEqual(self.fixture.design_path.read_bytes(), old_design)
        self.assertEqual(self.fixture.approval_path.read_bytes(), old_approval)
        self.assertFalse(Path(report["receipt_path"]).exists())

    def test_both_retired_h800_schema_locations_block_promotion(self) -> None:
        declarations = (
            {"schema": "sft_h800_calibration_candidate/v1"},
            {
                "h800_calibration": {
                    "schema": "sft_h800_calibration_approval_freeze/v1"
                }
            },
        )
        for declaration in declarations:
            with self.subTest(declaration=declaration):
                design = {**self.fixture.design, **declaration}
                write_json(self.fixture.candidate_path, design)
                digest = sha256_file(self.fixture.candidate_path)
                report = self.promote(
                    promote=False,
                    expected_candidate_sha256=digest,
                )
                self.assertFalse(report["all_passed"])
                self.assertFalse(report["checks"]["candidate_not_retired"])
                self.assertEqual(
                    report["retired_plan_schemas"],
                    retired_approval_plan_schemas(design),
                )

    def test_expected_sha_and_active_process_fail_closed_without_mutation(self) -> None:
        old_design = self.fixture.design_path.read_bytes()
        old_approval = self.fixture.approval_path.read_bytes()
        report = self.promote(
            promote=True,
            expected_candidate_sha256="0" * 64,
        )
        self.assertFalse(report["all_passed"])
        active = self.promote(
            promote=True,
            process_probe=lambda: [{"pid": 7, "command": "scheduler --execute"}],
        )
        self.assertFalse(active["all_passed"])
        self.assertEqual(self.fixture.design_path.read_bytes(), old_design)
        self.assertEqual(self.fixture.approval_path.read_bytes(), old_approval)

    def test_success_installs_original_bytes_with_history_and_receipt(self) -> None:
        with patch.object(
            promotion,
            "verify_approval",
            wraps=run_job.verify_approval,
        ) as verify:
            report = self.promote(promote=True)
        self.assertTrue(report["promoted"])
        self.assertEqual(verify.call_count, 1)
        self.assertEqual(
            self.fixture.design_path.read_bytes(), self.fixture.candidate_bytes
        )
        approval = json.loads(self.fixture.approval_path.read_text())
        self.assertIs(approval["approved"], True)
        self.assertEqual(approval["design_sha256"], self.fixture.candidate_sha256)
        receipt = Path(report["receipt_path"])
        history = Path(report["history_dir"])
        self.assertTrue(receipt.is_file())
        self.assertTrue((history / "previous_approval.json").is_file())
        self.assertTrue((history / "previous_approval_design.json").is_file())
        self.assertEqual(
            (history / "candidate_approval_design.json").read_bytes(),
            self.fixture.candidate_bytes,
        )

    def test_failure_after_design_install_leaves_approval_locked(self) -> None:
        original = promotion.atomic_write_bytes

        def fail_final_approval(path: Path, payload: bytes, mode: int = 0o600):
            parsed = json.loads(payload) if path == self.fixture.approval_path else {}
            if parsed.get("approved") is True:
                raise OSError("simulated final approval failure")
            return original(path, payload, mode)

        with patch.object(
            promotion, "atomic_write_bytes", side_effect=fail_final_approval
        ):
            with self.assertRaisesRegex(OSError, "simulated"):
                self.promote(promote=True)
        approval = json.loads(self.fixture.approval_path.read_text())
        self.assertIs(approval["approved"], False)
        self.assertIs(approval["locked"], True)
        self.assertEqual(
            self.fixture.design_path.read_bytes(), self.fixture.candidate_bytes
        )


if __name__ == "__main__":
    unittest.main()
