from __future__ import annotations

import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import sha256_file, write_json, write_jsonl  # noqa: E402
from prepare_lora_zero3_throughput_delta import (  # noqa: E402
    output_paths_are_safe,
    validate_delta,
)


class PrepareLoraZero3ThroughputDeltaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.baseline_path = self.root / "baseline.jsonl"
        self.new_matrix_path = self.root / "new.jsonl"
        self.pending_path = self.root / "pending.jsonl"
        self.plan_path = self.root / "recovery-plan.json"
        self.families_path = self.root / "runtime" / "families.jsonl"
        self.boundary_dir = self.root / "results" / "boundary_summaries"
        self.results_dir = self.root / "results"
        self.families = self.make_families()
        write_jsonl(self.families_path, self.families)
        write_json(
            self.plan_path,
            {
                "purpose": "Retry only the 20 H800 LoRA + ZeRO-3 memory families",
                "all_passed": True,
                "family_job_ids": [family["job_id"] for family in self.families],
                "queue_path": "runtime/families.jsonl",
                "queue_sha256": sha256_file(self.families_path),
            },
        )
        for family in self.families:
            write_json(
                self.boundary_dir / f"{family['job_id']}.json",
                {
                    "family_job_id": family["job_id"],
                    "trials": [
                        {"mbs": 1, "classification": "success"},
                        {"mbs": 2, "classification": "success"},
                        {"mbs": 4, "classification": "oom"},
                    ],
                    "max_feasible_mbs": 2,
                    "first_failed_mbs": 4,
                },
            )
        self.old_job = self.make_job(
            self.families[0],
            job_id="old-screen",
            request_id="request-old",
            zero="zero2",
        )
        self.delta_job = self.make_job(
            self.families[0],
            job_id="new-zero3-screen",
            request_id="request-new",
        )
        # Memory feasibility is a per-microbatch boundary and is intentionally
        # reusable by throughput requests with a different target GBS.
        self.delta_job["target_gbs"] = 256
        write_jsonl(self.baseline_path, [self.old_job])
        write_jsonl(self.new_matrix_path, [self.old_job, self.delta_job])
        write_jsonl(self.pending_path, [self.delta_job])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def make_families() -> list[dict[str, object]]:
        datasets = (
            ("short_512", 512),
            ("multiturn_4096", 4096),
            ("longcontext_32768", 32768),
        )
        combinations = itertools.product(
            ("qwen3_8b", "qwen3_14b"),
            datasets,
            (2, 4),
            (False, True),
        )
        families = []
        for index, (model_id, (dataset_id, cutoff_len), gpu_count, gc) in enumerate(combinations):
            if index == 20:
                break
            families.append(
                {
                    "job_id": f"mem-{index:02d}",
                    "kind": "memory_boundary",
                    "model_id": model_id,
                    "train_type": "lora",
                    "dataset_id": dataset_id,
                    "cutoff_len": cutoff_len,
                    "gpu_count": gpu_count,
                    "zero": "zero3",
                    "gc": gc,
                    "target_gbs": 64,
                    "packing": False,
                    "mbs_candidates": [1, 2, 4],
                }
            )
        return families

    @staticmethod
    def make_job(
        family: dict[str, object],
        *,
        job_id: str,
        request_id: str,
        zero: str = "zero3",
    ) -> dict[str, object]:
        return {
            "job_id": job_id,
            "kind": "throughput_screen",
            "fidelity": "screen",
            "request_id": request_id,
            "model_id": family["model_id"],
            "train_type": family["train_type"],
            "dataset_id": family["dataset_id"],
            "cutoff_len": family["cutoff_len"],
            "gpu_count": family["gpu_count"],
            "zero": zero,
            "gc": family["gc"],
            "mbs": 2,
            "target_gbs": family["target_gbs"],
            "packing": family["packing"],
        }

    def validate(self, boundary_evidence: Path | None = None) -> tuple[dict, list[dict]]:
        return validate_delta(
            baseline_path=self.baseline_path,
            new_matrix_path=self.new_matrix_path,
            pending_queue_path=self.pending_path,
            recovery_plan_path=self.plan_path,
            boundary_evidence_path=boundary_evidence or self.boundary_dir,
            results_dir=self.results_dir,
            project_root=self.root,
        )

    def test_accepts_exact_new_recovery_enabled_delta(self) -> None:
        report, rows = self.validate()

        self.assertTrue(report["all_passed"])
        self.assertEqual(rows, [self.delta_job])
        self.assertEqual(report["counts"]["delta_jobs"], 1)
        self.assertEqual(
            report["delta_candidates"][0]["source_family_job_id"],
            self.families[0]["job_id"],
        )
        self.assertEqual(
            report["delta_candidates"][0]["gradient_accumulation_steps"],
            64,
        )

    def test_rejects_pending_queue_that_is_not_exact_delta(self) -> None:
        write_jsonl(self.pending_path, [self.delta_job, self.delta_job])

        report, _ = self.validate()

        self.assertFalse(report["all_passed"])
        self.assertFalse(report["checks"]["pending_physical_keys_unique"])
        self.assertFalse(report["checks"]["pending_keys_exactly_match_delta_in_order"])

    def test_rejects_boundary_overflow_and_nondivisible_gbs(self) -> None:
        changed = {**self.delta_job, "mbs": 3}
        write_jsonl(self.new_matrix_path, [self.old_job, changed])
        write_jsonl(self.pending_path, [changed])

        report, _ = self.validate()

        candidate = report["delta_candidates"][0]
        self.assertFalse(report["all_passed"])
        self.assertFalse(candidate["checks"]["mbs_within_recovered_boundary"])
        self.assertFalse(candidate["checks"]["target_gbs_divisible"])

    def test_rejects_historical_physical_key_and_existing_terminal_status(self) -> None:
        historical_job = {**self.delta_job, "job_id": "previous-attempt"}
        write_json(
            self.results_dir / "previous-attempt" / "rendered_run.json",
            {"job": historical_job},
        )
        write_json(
            self.results_dir / "previous-attempt" / "status.json",
            {"classification": "failed"},
        )
        write_json(
            self.results_dir / str(self.delta_job["job_id"]) / "status.json",
            {"classification": "success"},
        )

        report, _ = self.validate()

        self.assertFalse(report["checks"]["no_historical_screen_or_formal_keys"])
        self.assertFalse(report["checks"]["no_existing_success_or_oom"])
        self.assertEqual(len(report["violations"]["historical_key_hits"]), 1)
        self.assertEqual(len(report["violations"]["terminal_status_hits"]), 1)

    def test_validated_boundary_manifest_is_hash_bound(self) -> None:
        evidence_path = self.root / "recovery-results-validation.json"
        evidence_rows = []
        for family in self.families:
            summary_path = self.boundary_dir / f"{family['job_id']}.json"
            evidence_rows.append(
                {
                    "family_job_id": family["job_id"],
                    "summary_path": str(summary_path.relative_to(self.root)),
                    "summary_sha256": sha256_file(summary_path),
                    "all_passed": True,
                }
            )
        write_json(evidence_path, {"all_passed": True, "families": evidence_rows})
        report, _ = self.validate(evidence_path)
        self.assertTrue(report["all_passed"])

        summary_path = self.boundary_dir / f"{self.families[0]['job_id']}.json"
        summary = json.loads(summary_path.read_text())
        summary["max_feasible_mbs"] = 1
        write_json(summary_path, summary)
        report, _ = self.validate(evidence_path)
        self.assertFalse(report["checks"]["boundary_evidence_hashes_match"])
        self.assertFalse(report["all_passed"])

    def test_outputs_cannot_alias_inputs_or_approval(self) -> None:
        self.assertFalse(
            output_paths_are_safe(
                self.baseline_path,
                self.root / "delta.jsonl",
                [self.baseline_path],
                self.root,
            )
        )
        self.assertFalse(
            output_paths_are_safe(
                self.root / "report.json",
                self.root / "config" / "APPROVED_TO_RUN.json",
                [self.baseline_path],
                self.root,
            )
        )
        self.assertTrue(
            output_paths_are_safe(
                self.root / "report.json",
                self.root / "delta.jsonl",
                [self.baseline_path],
                self.root,
            )
        )

    def test_cli_writes_only_explicit_report_and_delta_queue(self) -> None:
        report_path = self.root / "audit" / "delta-report.json"
        output_queue_path = self.root / "audit" / "delta.jsonl"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "prepare_lora_zero3_throughput_delta.py"),
                "--baseline",
                str(self.baseline_path),
                "--new-matrix",
                str(self.new_matrix_path),
                "--pending-queue",
                str(self.pending_path),
                "--recovery-plan",
                str(self.plan_path),
                "--boundary-evidence",
                str(self.boundary_dir),
                "--results-dir",
                str(self.results_dir),
                "--project-root",
                str(self.root),
                "--report",
                str(report_path),
                "--output-queue",
                str(output_queue_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(report_path.read_text())
        self.assertTrue(report["all_passed"])
        self.assertTrue(report["output_queue"]["written"])
        self.assertEqual(report["output_queue"]["sha256"], sha256_file(output_queue_path))
        self.assertEqual(
            [json.loads(line) for line in output_queue_path.read_text().splitlines()],
            [self.delta_job],
        )


if __name__ == "__main__":
    unittest.main()
