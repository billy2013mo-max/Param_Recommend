from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
from prepare_throughput_formal_delta import (  # noqa: E402
    ABSOLUTE_MAX_DELTA_RUNS,
    config_key_record,
    expected_formal_job_id,
    output_paths_are_safe,
    validate_formal_delta,
)


class PrepareThroughputFormalDeltaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.baseline_path = self.root / "baseline-formal.jsonl"
        self.new_matrix_path = self.root / "new-formal.jsonl"
        self.screen_delta_report_path = self.root / "screen-delta-report.json"
        self.stage_decisions_path = self.root / "stage-decisions.json"
        self.screen_matrix_path = self.root / "screen-matrix.jsonl"
        self.requests_path = self.root / "throughput-requests.jsonl"
        self.experiment_path = self.root / "experiment.json"
        self.families_path = self.root / "memory-families.jsonl"
        self.boundary_dir = self.root / "results" / "boundary_summaries"
        self.results_dir = self.root / "results"
        self.results_dir.mkdir(parents=True)

        self.request_a = self.make_request("request-a", "qwen3_8b")
        self.request_b = self.make_request("request-b", "qwen3_4b")
        self.old_config_a = self.make_config(
            self.request_a, gpu_count=2, zero="zero2", gc=False, mbs=2
        )
        self.removed_config_a = self.make_config(
            self.request_a, gpu_count=4, zero="zero2", gc=False, mbs=1
        )
        self.new_config_a = self.make_config(
            self.request_a, gpu_count=2, zero="zero3", gc=False, mbs=2
        )
        self.config_b = self.make_config(
            self.request_b, gpu_count=1, zero="none", gc=False, mbs=2
        )

        self.old_job_a = self.make_formal(self.request_a, self.old_config_a)
        self.removed_job_a = self.make_formal(self.request_a, self.removed_config_a)
        self.new_job_a = self.make_formal(self.request_a, self.new_config_a)
        self.job_b = self.make_formal(self.request_b, self.config_b)
        self.new_screen = self.make_screen(self.new_config_a, "new-zero3-screen")

        write_jsonl(
            self.baseline_path,
            [self.old_job_a, self.removed_job_a, self.job_b],
        )
        write_jsonl(
            self.new_matrix_path,
            [self.old_job_a, self.new_job_a, self.job_b],
        )
        write_jsonl(
            self.screen_matrix_path,
            [
                self.make_screen(self.old_config_a, "old-a-screen"),
                self.new_screen,
                self.make_screen(self.config_b, "b-screen"),
            ],
        )
        write_jsonl(self.requests_path, [self.request_a, self.request_b])
        write_json(self.experiment_path, self.make_experiment())
        write_json(self.screen_delta_report_path, self.make_screen_delta_report())
        write_json(self.stage_decisions_path, self.make_stage_decisions())

        self.family = {
            "job_id": "mem-new-zero3",
            "kind": "memory_boundary",
            "model_id": self.new_config_a["model_id"],
            "train_type": self.new_config_a["train_type"],
            "dataset_id": self.new_config_a["dataset_id"],
            "cutoff_len": self.new_config_a["cutoff_len"],
            "gpu_count": self.new_config_a["gpu_count"],
            "zero": self.new_config_a["zero"],
            "gc": self.new_config_a["gc"],
            "packing": self.new_config_a["packing"],
            "target_gbs": 64,
            "mbs_candidates": [1, 2, 4],
        }
        write_jsonl(self.families_path, [self.family])
        self.write_boundary(
            trials=[
                {"mbs": 1, "classification": "success"},
                {"mbs": 2, "classification": "success"},
                {"mbs": 4, "classification": "oom"},
            ],
            max_feasible=2,
            first_failed=4,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def make_request(request_id: str, model_id: str) -> dict[str, object]:
        return {
            "request_id": request_id,
            "kind": "throughput_after_memory",
            "model_id": model_id,
            "train_type": "lora",
            "dataset_id": "short_512",
            "cutoff_len": 512,
            "target_gbs": 16,
            "repeats": 1,
            "warmup_steps": 3,
            "measure_steps": 10,
            "screen_warmup_steps": 2,
            "screen_measure_steps": 4,
            "shortlist_top_k": 2,
            "parallel_class": "gpu_partitionable",
        }

    @staticmethod
    def make_experiment() -> dict[str, object]:
        return {
            "measurement": {
                "throughput_warmup_steps": 3,
                "throughput_measure_steps": 10,
            },
            "matrix_policy": {"throughput_shortlist_top_k": 2},
            "training_scope": {
                "gpu_counts": [1, 2, 4],
                "zero_by_gpu_count": {
                    "1": ["none"],
                    "2": ["zero2", "zero3"],
                    "4": ["zero2", "zero3"],
                },
            },
        }

    @staticmethod
    def make_config(
        request: dict[str, object],
        *,
        gpu_count: int,
        zero: str,
        gc: bool,
        mbs: int,
    ) -> dict[str, object]:
        return {
            "request_id": request["request_id"],
            "model_id": request["model_id"],
            "train_type": request["train_type"],
            "dataset_id": request["dataset_id"],
            "cutoff_len": request["cutoff_len"],
            "gpu_count": gpu_count,
            "zero": zero,
            "gc": gc,
            "mbs": mbs,
            "target_gbs": request["target_gbs"],
            "packing": False,
        }

    @staticmethod
    def make_formal(
        request: dict[str, object], config: dict[str, object], repeat: int = 0
    ) -> dict[str, object]:
        job: dict[str, object] = {
            **config,
            "job_id": "pending",
            "kind": "throughput",
            "fidelity": "formal",
            "repeat": repeat,
            "warmup_steps": request["warmup_steps"],
            "measure_steps": request["measure_steps"],
            "parallel_class": request["parallel_class"],
            "requires_external_node_idle": False,
            "model_family": "qwen3",
            "model_parameters": 8_000_000_000,
            "model_path": f"/models/{config['model_id']}",
            "tokenizer_path": f"/models/{config['model_id']}",
            "template": "qwen3_nothink",
        }
        job["job_id"] = expected_formal_job_id(request, job)
        return job

    @staticmethod
    def make_screen(config: dict[str, object], job_id: str) -> dict[str, object]:
        return {
            **config,
            "job_id": f"tputscreen-{job_id}",
            "kind": "throughput_screen",
            "fidelity": "screen",
            "repeat": 0,
            "warmup_steps": 2,
            "measure_steps": 4,
        }

    def make_screen_delta_report(self) -> dict[str, object]:
        return {
            "all_passed": True,
            "checks": {"validated": True},
            "violations": {"errors": []},
            "delta_candidates": [
                {
                    "job_id": self.new_screen["job_id"],
                    "physical_key": config_key_record(
                        tuple(self.new_config_a.values())
                    ),
                    "checks": {"validated": True},
                    "all_passed": True,
                }
            ],
            "freeze_preparation": {"allowed_job_ids": [self.new_screen["job_id"]]},
        }

    def make_stage_decisions(self) -> dict[str, object]:
        return {
            "throughput_screening": {
                "decisions": [
                    {
                        "request_id": self.request_a["request_id"],
                        "status": "shortlisted",
                        "top_k": 2,
                        "shortlisted": [
                            {**self.old_config_a, "status": "measured"},
                            {**self.new_config_a, "status": "measured"},
                        ],
                    }
                ]
            }
        }

    def write_boundary(
        self,
        *,
        trials: list[dict[str, object]],
        max_feasible: int | None,
        first_failed: int | None,
    ) -> None:
        write_json(
            self.boundary_dir / f"{self.family['job_id']}.json",
            {
                "family_job_id": self.family["job_id"],
                "trials": trials,
                "max_feasible_mbs": max_feasible,
                "first_failed_mbs": first_failed,
            },
        )

    def validate(self) -> tuple[dict, list[dict]]:
        return validate_formal_delta(
            baseline_path=self.baseline_path,
            new_matrix_path=self.new_matrix_path,
            screen_delta_report_path=self.screen_delta_report_path,
            stage_decisions_path=self.stage_decisions_path,
            screen_matrix_path=self.screen_matrix_path,
            throughput_requests_path=self.requests_path,
            experiment_path=self.experiment_path,
            memory_families_path=self.families_path,
            boundary_dir=self.boundary_dir,
            results_dir=self.results_dir,
        )

    def write_history(self, *, healthy: bool) -> None:
        result_dir = self.results_dir / str(self.new_job_a["job_id"])
        historical_job = {
            **self.new_job_a,
            "warmup_steps": 20,
            "measure_steps": 100,
        }
        write_json(result_dir / "rendered_run.json", {"job": historical_job})
        write_json(
            result_dir / "status.json",
            {"classification": "success" if healthy else "failed"},
        )
        if healthy:
            for rank in range(2):
                write_json(
                    result_dir / "metrics" / f"summary.rank{rank}.json",
                    {
                        "rank": rank,
                        "failure": None,
                        "measured_steps": 100,
                        "measured_seconds": 10.0,
                        "measured_totals": {"logical_samples": 160},
                    },
                )

    def cli_command(self, report: Path, queue: Path) -> list[str]:
        return [
            sys.executable,
            str(SCRIPTS / "prepare_throughput_formal_delta.py"),
            "--baseline",
            str(self.baseline_path),
            "--new-matrix",
            str(self.new_matrix_path),
            "--screen-delta-report",
            str(self.screen_delta_report_path),
            "--stage-decisions",
            str(self.stage_decisions_path),
            "--screen-matrix",
            str(self.screen_matrix_path),
            "--throughput-requests",
            str(self.requests_path),
            "--experiment",
            str(self.experiment_path),
            "--memory-families",
            str(self.families_path),
            "--boundary-dir",
            str(self.boundary_dir),
            "--results-dir",
            str(self.results_dir),
            "--project-root",
            str(self.root),
            "--report",
            str(report),
            "--output-queue",
            str(queue),
        ]

    def test_accepts_exact_formal_delta_and_preserves_new_matrix_order(self) -> None:
        report, queue = self.validate()

        self.assertTrue(report["all_passed"])
        self.assertEqual(queue, [self.new_job_a])
        self.assertEqual(report["counts"]["affected_requests"], 1)
        self.assertEqual(report["counts"]["added_runs"], 1)
        self.assertEqual(report["counts"]["removed_runs"], 1)
        self.assertEqual(report["counts"]["dynamic_max_delta_runs"], 2)
        self.assertEqual(report["added_runs"][0]["disposition"], "queue")

    def test_healthy_historical_formal_run_is_explicitly_reused(self) -> None:
        self.write_history(healthy=True)

        report, queue = self.validate()

        self.assertTrue(report["all_passed"])
        self.assertEqual(queue, [])
        self.assertEqual(report["counts"]["reused_healthy_runs"], 1)
        self.assertEqual(report["added_runs"][0]["disposition"], "reused")
        self.assertTrue(report["reused_runs"][0]["healthy_history"][0]["all_passed"])

    def test_existing_unhealthy_formal_attempt_blocks_instead_of_rerunning(
        self,
    ) -> None:
        self.write_history(healthy=False)

        report, queue = self.validate()

        self.assertFalse(report["all_passed"])
        self.assertEqual(queue, [])
        self.assertFalse(
            report["checks"]["existing_formal_attempts_are_healthy_reuses"]
        )
        self.assertEqual(
            report["added_runs"][0]["disposition"],
            "rejected_unhealthy_history",
        )

    def test_rejects_any_change_to_an_unaffected_request(self) -> None:
        changed_b = {**self.job_b, "model_path": "/tampered/model"}
        write_jsonl(
            self.new_matrix_path,
            [self.old_job_a, self.new_job_a, changed_b],
        )

        report, _ = self.validate()

        self.assertFalse(report["all_passed"])
        self.assertFalse(report["checks"]["unaffected_request_rows_unchanged"])
        self.assertFalse(report["checks"]["retained_run_rows_unchanged"])

    def test_rejects_formal_rows_out_of_latest_shortlist_order(self) -> None:
        write_jsonl(
            self.new_matrix_path,
            [self.new_job_a, self.old_job_a, self.job_b],
        )

        report, _ = self.validate()

        self.assertFalse(report["all_passed"])
        self.assertFalse(report["checks"]["affected_formal_order_matches_shortlist"])
        self.assertFalse(
            report["checks"]["new_matrix_order_matches_requests_and_shortlists"]
        )

    def test_rejects_added_configuration_missing_from_screen_matrix(self) -> None:
        write_jsonl(
            self.screen_matrix_path,
            [
                self.make_screen(self.old_config_a, "old-a-screen"),
                self.make_screen(self.config_b, "b-screen"),
            ],
        )

        report, _ = self.validate()

        self.assertFalse(report["all_passed"])
        self.assertFalse(report["checks"]["affected_shortlists_valid"])
        self.assertFalse(report["checks"]["added_configurations_selected_and_screened"])

    def test_accepts_retained_shortlist_with_exact_active_formal_evidence(
        self,
    ) -> None:
        write_jsonl(
            self.screen_matrix_path,
            [self.new_screen, self.make_screen(self.config_b, "b-screen")],
        )
        decisions = self.make_stage_decisions()
        retained = decisions["throughput_screening"]["decisions"][0]["shortlisted"][0]
        retained.update(
            {
                "candidate_plan_source": "active_formal",
                "measurement_source": "formal",
                "formal_substitute_scope": "active",
                "status": "measured",
                "samples_per_second": 100.0,
                "job_ids": [self.old_job_a["job_id"]],
            }
        )
        write_json(self.stage_decisions_path, decisions)

        report, queue = self.validate()

        self.assertTrue(report["all_passed"])
        self.assertEqual(queue, [self.new_job_a])

    def test_rejects_unscreened_shortlist_without_exact_active_formal_evidence(
        self,
    ) -> None:
        write_jsonl(
            self.screen_matrix_path,
            [self.new_screen, self.make_screen(self.config_b, "b-screen")],
        )

        report, _ = self.validate()

        self.assertFalse(report["all_passed"])
        self.assertFalse(report["checks"]["affected_shortlists_valid"])

    def test_rejects_wrong_formal_steps_and_boundary_overflow(self) -> None:
        changed = {**self.new_job_a, "measure_steps": 9}
        write_jsonl(
            self.new_matrix_path,
            [self.old_job_a, changed, self.job_b],
        )
        self.write_boundary(
            trials=[
                {"mbs": 1, "classification": "success"},
                {"mbs": 2, "classification": "oom"},
            ],
            max_feasible=1,
            first_failed=2,
        )

        report, _ = self.validate()

        self.assertFalse(report["all_passed"])
        self.assertFalse(report["checks"]["new_formal_rows_valid"])
        self.assertFalse(
            report["checks"]["added_configurations_within_memory_boundaries"]
        )

    def test_dynamic_limit_cannot_exceed_absolute_limit(self) -> None:
        self.request_a["shortlist_top_k"] = ABSOLUTE_MAX_DELTA_RUNS + 1
        write_jsonl(self.requests_path, [self.request_a, self.request_b])
        decisions = self.make_stage_decisions()
        decisions["throughput_screening"]["decisions"][0]["top_k"] = (
            ABSOLUTE_MAX_DELTA_RUNS + 1
        )
        write_json(self.stage_decisions_path, decisions)

        report, _ = self.validate()

        self.assertFalse(report["all_passed"])
        self.assertEqual(
            report["counts"]["dynamic_max_delta_runs"],
            ABSOLUTE_MAX_DELTA_RUNS + 1,
        )
        self.assertFalse(report["checks"]["dynamic_delta_cap_within_absolute_limit"])

    def test_cli_writes_only_delta_and_removes_stale_queue_on_failure(self) -> None:
        report_path = self.root / "audit" / "formal-delta-report.json"
        queue_path = self.root / "audit" / "formal-delta.jsonl"
        command = self.cli_command(report_path, queue_path)
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_jsonl(queue_path), [self.new_job_a])
        report = json.loads(report_path.read_text())
        self.assertTrue(report["output_queue"]["written"])
        self.assertEqual(report["output_queue"]["sha256"], sha256_file(queue_path))

        write_jsonl(
            self.new_matrix_path,
            [self.new_job_a, self.old_job_a, self.job_b],
        )
        result = subprocess.run(
            [*command, "--overwrite"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(queue_path.exists())
        failed_report = json.loads(report_path.read_text())
        self.assertFalse(failed_report["all_passed"])
        self.assertFalse(failed_report["output_queue"]["written"])

    def test_outputs_cannot_alias_inputs_or_live_approvals(self) -> None:
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
                self.root / "audit" / "report.json",
                self.root / "audit" / "delta.jsonl",
                [self.baseline_path],
                self.root,
            )
        )


if __name__ == "__main__":
    unittest.main()
