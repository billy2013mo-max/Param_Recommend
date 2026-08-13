from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from approval_gate import expected_provenance_source_paths  # noqa: E402
from common import sha256_file, sha256_json  # noqa: E402
from freeze_h800_calibration_approval_candidate import (  # noqa: E402
    CAMPAIGN_HARD_WALL_TIME_SECONDS,
    PER_RUN_TIMEOUT_SECONDS,
    freeze_candidate,
    materialize_execution_jobs,
    validate_materialization,
)
from prepare_h800_calibration_candidate import (  # noqa: E402
    DESIGN_NAME,
    JOBS_NAME,
    SOURCE_FILES,
    build_candidate,
)
from scheduler import validate_adaptive_queue  # noqa: E402


RUNTIME_IDENTITY = {
    "schema_version": 1,
    "python_executable": "/fixed/python",
    "packages": {"deepspeed": "0.19.2"},
}
RUNTIME_PATCH = {"all_passed": True, "patch": "test"}


class H800CalibrationApprovalFreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for relative in SOURCE_FILES:
            source = PROJECT / relative
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for name in ("README.md", "EXPERIMENT_DESIGN.md"):
            shutil.copy2(PROJECT / name, self.root / name)

        self.live_paths = [
            self.root / "config" / "APPROVED_TO_RUN.json",
            self.root / "runtime" / "approval_design.json",
            self.root / "runtime" / "queue.json",
        ]
        for path in self.live_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"sentinel":"untouched"}\n', encoding="utf-8")
        self.live_before = {path: path.read_bytes() for path in self.live_paths}

        design, jobs = build_candidate(self.root)
        self.source_design_path = (
            self.root / "artifacts" / "candidates" / DESIGN_NAME
        )
        self.source_jobs_path = self.root / "artifacts" / "candidates" / JOBS_NAME
        self.source_design_path.parent.mkdir(parents=True, exist_ok=True)
        self.source_design_path.write_text(
            json.dumps(design, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        self.source_jobs_path.write_text(
            "".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for row in jobs
            ),
            encoding="utf-8",
        )
        self.source_design = design
        self.source_jobs = jobs
        self._write_fresh_provenance()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_fresh_provenance(self) -> None:
        paths = expected_provenance_source_paths(self.root)
        manifest = {
            str(path.relative_to(self.root)): sha256_file(path) for path in paths
        }
        identity = {"project_source_snapshot_sha256": sha256_json(manifest)}
        report = {
            "schema_version": 1,
            "project_source_manifest": manifest,
            "runtime_identity": identity,
            "runtime_fingerprint_sha256": sha256_json(identity),
            "reproducibility_identity_complete": True,
        }
        path = self.root / "artifacts" / "provenance.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    def assert_live_untouched(self) -> None:
        for path, before in self.live_before.items():
            self.assertEqual(path.read_bytes(), before)

    def test_materialization_removes_false_flag_and_binds_exact_source_and_budgets(
        self,
    ) -> None:
        rows, mapping = materialize_execution_jobs(
            self.source_design, self.source_jobs
        )

        self.assertEqual(len(rows), 109)
        self.assertTrue(all("execution_authorized" not in row for row in rows))
        self.assertEqual(
            [row["job_id"] for row in rows],
            [row["job_id"] for row in self.source_jobs],
        )
        for source, execution, binding in zip(
            self.source_jobs, rows, mapping, strict=True
        ):
            gate = execution["execution_gate"]
            self.assertEqual(gate["source_job_payload_sha256"], sha256_json(source))
            self.assertEqual(
                binding["execution_job_payload_sha256"], sha256_json(execution)
            )
            self.assertEqual(
                gate["per_run_timeout_seconds"], PER_RUN_TIMEOUT_SECONDS
            )
            self.assertEqual(
                gate["campaign_hard_wall_time_seconds"],
                CAMPAIGN_HARD_WALL_TIME_SECONDS,
            )
        self.assertIsNotNone(
            validate_adaptive_queue(rows, require_execution_gate=True)
        )
        validation = validate_materialization(
            source_design=self.source_design,
            source_jobs=self.source_jobs,
            execution_jobs=rows,
            transform_mapping=mapping,
            project_root=self.root,
        )
        self.assertTrue(validation["all_passed"])
        self.assert_live_untouched()

    def test_tampering_cross_candidate_budget_or_4090_fails_closed(self) -> None:
        rows, mapping = materialize_execution_jobs(
            self.source_design, self.source_jobs
        )
        cases = []
        other_candidate = copy.deepcopy(rows)
        other_candidate[0]["execution_gate"][
            "source_candidate_canonical_sha256"
        ] = "0" * 64
        cases.append(other_candidate)
        changed_timeout = copy.deepcopy(rows)
        changed_timeout[0]["execution_gate"]["per_run_timeout_seconds"] = 999999
        cases.append(changed_timeout)
        rtx = copy.deepcopy(rows)
        rtx[0]["gpu_type"] = "NVIDIA GeForce RTX 4090"
        cases.append(rtx)
        missing_dependency = copy.deepcopy(rows)
        boundary = next(row for row in missing_dependency if "boundary_probe" in row)
        boundary["boundary_probe"]["condition"] = {
            "type": "if_probe_outcome",
            "probe": "absent",
            "outcome": "success",
        }
        cases.append(missing_dependency)

        for tampered in cases:
            with self.subTest(case=cases.index(tampered)):
                report = validate_materialization(
                    source_design=self.source_design,
                    source_jobs=self.source_jobs,
                    execution_jobs=tampered,
                    transform_mapping=mapping,
                    project_root=self.root,
                )
                self.assertFalse(report["all_passed"])
        self.assert_live_untouched()

    def test_retired_freezer_rejects_before_any_offline_write(self) -> None:
        queue = (
            self.root
            / "artifacts"
            / "candidates"
            / "h800_calibration_execution_queue.jsonl"
        )
        output = (
            self.root
            / "artifacts"
            / "candidates"
            / "h800_calibration_approval_design.candidate.json"
        )
        with self.assertRaisesRegex(PermissionError, "Retired H800 calibration"):
            freeze_candidate(
                project_root=self.root,
                source_design_path=self.source_design_path,
                source_jobs_path=self.source_jobs_path,
                queue_path=queue,
                output_path=output,
                runtime_identity=RUNTIME_IDENTITY,
                runtime_patch=RUNTIME_PATCH,
            )

        self.assertFalse(queue.exists())
        self.assertFalse(output.exists())
        self.assert_live_untouched()

    def test_retirement_precedes_even_live_output_validation(self) -> None:
        with self.assertRaisesRegex(PermissionError, "Retired H800 calibration"):
            freeze_candidate(
                project_root=self.root,
                source_design_path=self.source_design_path,
                source_jobs_path=self.source_jobs_path,
                queue_path=self.root / "runtime" / "queue.json",
                output_path=(
                    self.root / "artifacts" / "candidates" / "candidate.json"
                ),
                runtime_identity=RUNTIME_IDENTITY,
                runtime_patch=RUNTIME_PATCH,
            )
        self.assert_live_untouched()


if __name__ == "__main__":
    unittest.main()
