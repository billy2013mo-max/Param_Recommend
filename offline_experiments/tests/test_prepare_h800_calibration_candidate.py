from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_h800_calibration_candidate import (  # noqa: E402
    MBS_DOMAIN,
    SOURCE_FILES,
    build_candidate,
    validate_candidate,
    write_candidate,
)
import prepare_h800_calibration_candidate as prepare  # noqa: E402


class H800CalibrationCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for relative in SOURCE_FILES:
            source = PROJECT / relative
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        self.approval = self.root / "config" / "APPROVED_TO_RUN.json"
        self.queue = self.root / "runtime" / "queue.json"
        self.live_design = self.root / "runtime" / "approval_design.json"
        self.approval.write_text('{"sentinel":"approval"}\n', encoding="utf-8")
        self.queue.parent.mkdir(parents=True, exist_ok=True)
        self.queue.write_text('{"sentinel":"queue"}\n', encoding="utf-8")
        self.live_design.write_text('{"sentinel":"design"}\n', encoding="utf-8")
        self.live_before = {
            path: path.read_bytes()
            for path in (self.approval, self.queue, self.live_design)
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_live_untouched(self) -> None:
        for path, expected in self.live_before.items():
            self.assertEqual(path.read_bytes(), expected)

    def test_matrix_is_h800_only_bounded_and_identifiable(self) -> None:
        design, jobs = build_candidate(self.root)

        self.assertEqual(len(jobs), 109)
        self.assertLess(len(jobs), 125)
        self.assertEqual({job["mbs"] for job in jobs}, set(MBS_DOMAIN))
        self.assertTrue(all(job["mbs"] <= 16 for job in jobs))
        self.assertTrue(
            all(job["mbs"] == 1 for job in jobs if job.get("packing") is True)
        )
        self.assertEqual({job["gpu_count"] for job in jobs}, {1, 2, 4})
        self.assertEqual({job["train_type"] for job in jobs}, {"full", "lora"})
        for job in jobs:
            self.assertEqual(job["gpu_type"], "NVIDIA H800 140GB HBM3")
            self.assertNotIn("4090", json.dumps(job, sort_keys=True).lower())
            expected_zero = "none" if job["gpu_count"] == 1 else "zero3"
            self.assertEqual(job["zero"], expected_zero)
            self.assertTrue(job["gc"])
            self.assertFalse(job["execution_authorized"])

        split = design["calibration_partition"]
        self.assertTrue(
            set(split["calibration"]).isdisjoint(set(split["holdout"]))
        )
        self.assertEqual(design["jobs_binding"]["rows"], len(jobs))
        self.assertLessEqual(
            design["budget"]["planned_wall_time_upper_bound_hours"], 72
        )
        self.assertEqual(
            design["acceptance"]["minimum_measured_doubling_speedup_to_recommend"],
            1.8,
        )
        self.assertEqual(
            design["acceptance"]["p95_capacity_ceiling_fraction"], 0.95
        )
        self.assertTrue(validate_candidate(design, jobs, self.root)["all_passed"])
        self.assert_live_untouched()

    def test_boundary_stops_at_mbs16_without_out_of_domain_probe(self) -> None:
        design, jobs = build_candidate(self.root)
        boundary = [job for job in jobs if job["kind"] == "throughput_screen"]

        self.assertEqual(len(boundary), 77)
        self.assertTrue(all(job["mbs"] in MBS_DOMAIN for job in boundary))
        self.assertTrue(
            all(
                job["boundary_probe"][
                    "mbs16_success_means_covered_domain_fully_feasible"
                ]
                is True
                for job in boundary
            )
        )
        stop = design["boundary_early_stop"]
        self.assertIn("MBS=16", stop["if_anchor_success"])
        self.assertIn("never probe MBS>16", stop["if_mbs16_success"])

    def test_packing_is_abba_with_independent_holdout_units(self) -> None:
        design, jobs = build_candidate(self.root)
        packed = [
            job
            for job in jobs
            if job.get("calibration_evidence_class") == "packing_paired_only"
        ]
        grouped: dict[str, list[dict]] = {}
        for job in packed:
            grouped.setdefault(job["packing_pair"]["pair_id"], []).append(job)

        self.assertEqual(len(grouped), 8)
        for rows in grouped.values():
            rows.sort(key=lambda row: row["packing_pair"]["sequence_index"])
            self.assertEqual(
                [row["packing_pair"]["treatment"] for row in rows],
                ["unpacked", "packed", "packed", "unpacked"],
            )
            self.assertEqual([row["repeat"] for row in rows], [0, 0, 1, 1])
        pairs = design["packing_study"]["pairs"]
        for mode in ("full", "lora"):
            for role in ("calibration", "holdout"):
                units = {
                    row["partition"]["split_unit_id"]
                    for row in pairs
                    if row["training_mode"] == mode
                    and row["partition"]["role"] == role
                }
                self.assertEqual(len(units), 2)

    def test_build_is_deterministic_but_retired_schema_cannot_write(self) -> None:
        first_design, first_jobs = build_candidate(self.root)
        second_design, second_jobs = build_candidate(self.root)
        self.assertEqual(first_design, second_design)
        self.assertEqual(first_jobs, second_jobs)

        output = self.root / "artifacts" / "candidates" / "review"
        with self.assertRaisesRegex(PermissionError, "Retired H800 calibration"):
            write_candidate(
                first_design,
                first_jobs,
                project_root=self.root,
                output_dir=output,
            )

        self.assertFalse(output.exists())
        self.assert_live_untouched()

    def test_cli_without_verify_only_is_retired_before_build(self) -> None:
        with (
            patch.object(sys, "argv", ["prepare_h800_calibration_candidate.py"]),
            patch.object(prepare, "build_candidate") as build,
        ):
            with self.assertRaisesRegex(PermissionError, "Retired H800 calibration"):
                prepare.main()
        build.assert_not_called()
        self.assert_live_untouched()

    def test_retirement_precedes_output_validation_and_tampering_is_detected(self) -> None:
        design, jobs = build_candidate(self.root)
        with self.assertRaisesRegex(PermissionError, "Retired H800 calibration"):
            write_candidate(
                design,
                jobs,
                project_root=self.root,
                output_dir=self.root / "runtime",
            )

        tampered = copy.deepcopy(jobs)
        tampered[0]["gpu_type"] = "NVIDIA GeForce RTX 4090"
        report = validate_candidate(design, tampered, self.root)
        self.assertFalse(report["all_passed"])
        self.assertIn("jobs_sha256", report["checks"])
        self.assertFalse(report["checks"]["job_rows"])

        out_of_domain = copy.deepcopy(jobs)
        out_of_domain[0]["mbs"] = 32
        report = validate_candidate(design, out_of_domain, self.root)
        self.assertFalse(report["all_passed"])
        self.assertFalse(report["checks"]["job_rows"])
        self.assert_live_untouched()


if __name__ == "__main__":
    unittest.main()
