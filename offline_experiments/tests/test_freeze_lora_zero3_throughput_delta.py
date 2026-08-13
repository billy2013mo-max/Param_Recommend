from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import sha256_file, write_json  # noqa: E402
from freeze_lora_zero3_throughput_delta import (  # noqa: E402
    ARTIFACT_MANIFEST_FILES,
    MATRIX_MANIFEST_FILES,
    approval_manifest_paths,
    build_approval_design_candidate,
    build_freeze_metadata,
    output_path_is_safe,
)


class FreezeLoraZero3ThroughputDeltaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        write_json(self.root / "config" / "experiment.json", {"campaign": "test"})
        write_json(
            self.root / "config" / "APPROVED_TO_RUN.json",
            {"approved": True, "must_not_be_manifested": True},
        )
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "scripts" / "runner.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        for name in MATRIX_MANIFEST_FILES:
            path = self.root / "matrix" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "design_summary.json":
                write_json(path, {"schema_version": 2, "campaign_id": "test"})
            else:
                path.write_text("{}\n", encoding="utf-8")
        for name in ARTIFACT_MANIFEST_FILES:
            write_json(self.root / "artifacts" / name, {"name": name})
        write_json(self.root / "data" / "dataset_info.json", {"dataset": "test"})
        derived = self.root / "data" / "derived" / "short.jsonl"
        derived.parent.mkdir(parents=True)
        derived.write_text('{"text":"test"}\n', encoding="utf-8")
        (self.root / "README.md").write_text("test\n", encoding="utf-8")
        (self.root / "EXPERIMENT_DESIGN.md").write_text("test\n", encoding="utf-8")
        self.extra = self.root / "audit" / "delta-report.json"
        write_json(self.extra, {"all_passed": True})

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_builds_canonical_full_manifest_and_preserves_metadata(self) -> None:
        metadata = {
            "design_purpose": "validated delta",
            "allowed_job_ids": ["tputscreen-1"],
            "authorized_gpu_ids": [1, 2, 3, 4],
            "throughput_screen_delta": {"jobs": 1},
        }

        design = build_approval_design_candidate(
            project_root=self.root,
            extra_files=(self.extra,),
            extra_metadata=metadata,
        )

        self.assertEqual(design["schema_version"], 1)
        self.assertIs(design["training_started"], False)
        self.assertEqual(
            design["matrix_summary"],
            {"schema_version": 2, "campaign_id": "test"},
        )
        self.assertEqual(design["design_purpose"], "validated delta")
        self.assertEqual(design["allowed_job_ids"], ["tputscreen-1"])
        manifest = design["file_sha256"]
        expected_paths = {
            "config/experiment.json",
            "scripts/runner.py",
            *(f"matrix/{name}" for name in MATRIX_MANIFEST_FILES),
            *(f"artifacts/{name}" for name in ARTIFACT_MANIFEST_FILES),
            "data/dataset_info.json",
            "data/derived/short.jsonl",
            "README.md",
            "EXPERIMENT_DESIGN.md",
            "audit/delta-report.json",
        }
        self.assertEqual(set(manifest), expected_paths)
        self.assertNotIn("config/APPROVED_TO_RUN.json", manifest)
        for relative, digest in manifest.items():
            self.assertEqual(digest, sha256_file(self.root / relative))
        self.assertFalse((self.root / "runtime" / "approval_design.json").exists())

    def test_missing_required_input_fails_closed(self) -> None:
        (self.root / "artifacts" / ARTIFACT_MANIFEST_FILES[0]).unlink()

        with self.assertRaises(FileNotFoundError):
            build_approval_design_candidate(project_root=self.root)

    def test_manifest_input_outside_root_fails_closed(self) -> None:
        outside = self.root.parent / f"{self.root.name}-outside-freeze-input.json"
        outside.write_text("{}\n", encoding="utf-8")
        self.addCleanup(outside.unlink, missing_ok=True)

        with self.assertRaises(ValueError):
            approval_manifest_paths(self.root, (outside,))

    def test_metadata_cannot_override_canonical_fields(self) -> None:
        with self.assertRaises(ValueError):
            build_approval_design_candidate(
                project_root=self.root,
                extra_metadata={"training_started": True},
            )

    def test_only_distinct_non_live_json_output_is_safe(self) -> None:
        protected_input = self.root / "audit" / "delta-report.json"
        protected = [protected_input]
        self.assertTrue(
            output_path_is_safe(
                self.root / "audit" / "approval-design-candidate.json",
                protected,
                self.root,
            )
        )
        for unsafe in (
            protected_input,
            self.root / "config" / "APPROVED_TO_RUN.json",
            self.root / "config" / "candidate.json",
            self.root / "runtime" / "approval_design.json",
            self.root / "matrix" / "design_summary.json",
            self.root / "artifacts" / "provenance.json",
            self.root / "audit" / "candidate.txt",
            self.root.parent / "outside.json",
        ):
            self.assertFalse(
                output_path_is_safe(unsafe, protected, self.root),
                str(unsafe),
            )

    def test_freeze_metadata_binds_queue_order_payloads_and_provenance(self) -> None:
        queue_binding = {
            "schema_version": 1,
            "path": "audit/delta.jsonl",
            "sha256": "a" * 64,
            "ordered_job_ids": ["tputscreen-1"],
            "ordered_job_payload_sha256": ["b" * 64],
            "job_payload_sha256": {"tputscreen-1": "b" * 64},
        }
        provenance_binding = {
            "path": "artifacts/provenance.json",
            "sha256": "c" * 64,
            "project_source_paths": ["scripts/runner.py"],
        }
        metadata = build_freeze_metadata(
            {
                "all_passed": True,
                "purpose": "validated delta",
                "allowed_job_ids": ["tputscreen-1"],
                "runtime_fix": {"commit": "test"},
                "runtime_identity": {"schema_version": 1},
                "runtime_fingerprint_sha256": "d" * 64,
                "runtime_patch": {"all_passed": True},
                "queue_binding": queue_binding,
                "provenance_binding": provenance_binding,
                "delta_report_sha256": "e" * 64,
                "input_queue_sha256": "f" * 64,
                "output_queue_sha256": "a" * 64,
                "recovery_results_validation_sha256": "1" * 64,
            }
        )

        self.assertEqual(metadata["queue_binding"], queue_binding)
        self.assertEqual(metadata["provenance_binding"], provenance_binding)
        stage = metadata["throughput_screen_delta"]
        self.assertEqual(stage["queue_path"], queue_binding["path"])
        self.assertEqual(stage["ordered_job_ids"], ["tputscreen-1"])
        self.assertEqual(
            stage["job_payload_sha256"], queue_binding["job_payload_sha256"]
        )


if __name__ == "__main__":
    unittest.main()
