from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_pipeline  # noqa: E402


class PipelineTests(unittest.TestCase):
    @staticmethod
    def write_healthy_summary(results: Path, job_id: str, measured_steps: int = 100) -> None:
        metrics = results / job_id / "metrics"
        metrics.mkdir(parents=True, exist_ok=True)
        (metrics / "summary.rank0.json").write_text(
            json.dumps(
                {
                    "failure": None,
                    "measured_steps": measured_steps,
                    "measured_seconds": 10.0,
                    "measured_totals": {"logical_samples": 64},
                }
            )
        )

    def test_scheduler_active_tracks_execution_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(json.dumps({"event": "scheduler_start", "execution_id": "one"}) + "\n")
            self.assertTrue(run_pipeline.scheduler_active(path))
            with path.open("a") as output:
                output.write(json.dumps({"event": "scheduler_complete", "execution_id": "one"}) + "\n")
            self.assertFalse(run_pipeline.scheduler_active(path))

    def test_completed_jobs_are_removed_from_resume_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results / "done").mkdir()
            (results / "done" / "status.json").write_text(json.dumps({"classification": "success"}))
            self.write_healthy_summary(results, "done")
            jobs = [
                {"job_id": "done", "kind": "throughput", "gpu_count": 1, "measure_steps": 100},
                {"job_id": "pending", "kind": "throughput", "gpu_count": 1, "measure_steps": 100},
            ]
            experiment = {"measurement": {"rerun_on_unhealthy_result": True}}
            with mock.patch.object(run_pipeline, "RESULTS_DIR", results), mock.patch.object(
                run_pipeline, "read_json", side_effect=lambda path: (
                    experiment if Path(path).name == "experiment.json" else json.loads(Path(path).read_text())
                )
            ):
                pending = run_pipeline.pending_jobs(jobs)
            self.assertEqual(pending, [jobs[1]])

    def test_success_with_truncated_measurement_is_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results / "truncated").mkdir()
            (results / "truncated" / "status.json").write_text(json.dumps({"classification": "success"}))
            self.write_healthy_summary(results, "truncated", measured_steps=99)
            job = {"job_id": "truncated", "kind": "throughput", "gpu_count": 1, "measure_steps": 100}
            experiment = {"measurement": {"rerun_on_unhealthy_result": True}}
            with mock.patch.object(run_pipeline, "RESULTS_DIR", results), mock.patch.object(
                run_pipeline, "read_json", side_effect=lambda path: (
                    experiment if Path(path).name == "experiment.json" else json.loads(Path(path).read_text())
                )
            ):
                self.assertEqual(run_pipeline.pending_jobs([job]), [job])
                self.assertEqual(run_pipeline.job_outcomes([job]), {"unhealthy_success": 1})

    def test_longer_completed_measurement_satisfies_reduced_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results / "longer").mkdir()
            (results / "longer" / "status.json").write_text(
                json.dumps({"classification": "success"})
            )
            self.write_healthy_summary(results, "longer", measured_steps=20)
            job = {
                "job_id": "longer",
                "kind": "throughput_screen",
                "gpu_count": 1,
                "measure_steps": 4,
            }
            with mock.patch.object(run_pipeline, "RESULTS_DIR", results):
                self.assertTrue(run_pipeline.successful_result_is_healthy(job))
                self.assertEqual(run_pipeline.job_outcomes([job]), {"success": 1})

    def test_profiler_success_requires_positive_operator_flops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            job_id = "profiler"
            (results / job_id).mkdir()
            (results / job_id / "status.json").write_text(
                json.dumps({"classification": "success"})
            )
            self.write_healthy_summary(results, job_id, measured_steps=3)
            profiler_dir = results / job_id / "metrics" / "profiler"
            profiler_dir.mkdir()
            operators = profiler_dir / "operators.rank0.json"
            operators.write_text(json.dumps([{"key": "matmul", "flops": 0}]))
            job = {
                "job_id": job_id,
                "kind": "profiler",
                "gpu_count": 1,
                "measure_steps": 3,
                "enable_profiler": True,
            }
            with mock.patch.object(run_pipeline, "RESULTS_DIR", results):
                self.assertFalse(run_pipeline.successful_result_is_healthy(job))
                operators.write_text(json.dumps([{"key": "matmul", "flops": 42}]))
                self.assertTrue(run_pipeline.successful_result_is_healthy(job))

    def test_oom_can_be_an_accepted_terminal_probe_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results / "oom").mkdir()
            (results / "oom" / "status.json").write_text(json.dumps({"classification": "oom"}))
            experiment = {"measurement": {"rerun_on_unhealthy_result": True}}
            with mock.patch.object(run_pipeline, "RESULTS_DIR", results), mock.patch.object(
                run_pipeline, "read_json", side_effect=lambda path: (
                    experiment if Path(path).name == "experiment.json" else json.loads(Path(path).read_text())
                )
            ):
                pending = run_pipeline.pending_jobs([{"job_id": "oom"}], {"success", "oom"})
            self.assertEqual(pending, [])

    def test_formal_oom_is_allowed_when_same_scenario_has_success(self) -> None:
        jobs = [
            {"job_id": "oom", "request_id": "scenario"},
            {"job_id": "success", "request_id": "scenario"},
        ]
        with mock.patch.object(
            run_pipeline,
            "job_outcome",
            side_effect=lambda job: "success" if job["job_id"] == "success" else "oom",
        ):
            self.assertEqual(run_pipeline.throughput_scenarios_without_success(jobs), [])

    def test_formal_scenario_without_success_still_blocks(self) -> None:
        jobs = [
            {
                "job_id": "oom",
                "request_id": "scenario",
                "model_id": "model",
                "train_type": "full",
                "dataset_id": "data",
                "target_gbs": 64,
            }
        ]
        with mock.patch.object(run_pipeline, "job_outcome", return_value="oom"):
            unavailable = run_pipeline.throughput_scenarios_without_success(jobs)
        self.assertEqual(
            unavailable,
            [
                {
                    "request_id": "scenario",
                    "model_id": "model",
                    "train_type": "full",
                    "dataset_id": "data",
                    "target_gbs": 64,
                    "outcomes": {"oom": 1},
                }
            ],
        )

    def test_scaling_oom_is_recorded_and_later_card_counts_continue(self) -> None:
        jobs = [
            {"job_id": f"scale-{gpu_count}", "request_id": "scenario", "gpu_count": gpu_count}
            for gpu_count in (1, 2, 4)
        ]
        report = {"scaling": {"families": []}}
        with mock.patch.object(run_pipeline, "materialize", return_value=jobs), mock.patch.object(
            run_pipeline, "read_jsonl", return_value=[{"request_id": "scenario"}]
        ), mock.patch.object(
            run_pipeline, "write_reports", return_value=report
        ), mock.patch.object(
            run_pipeline, "scaling_eligible_request_ids", return_value={"scenario"}
        ), mock.patch.object(
            run_pipeline, "run_scheduler_jobs"
        ) as run_jobs, mock.patch.object(
            run_pipeline, "now_state"
        ):
            run_pipeline.run_scaling()

        self.assertEqual(run_jobs.call_count, 3)
        for call in run_jobs.call_args_list:
            self.assertEqual(call.kwargs["accepted"], {"success", "oom"})

    def test_pipeline_scheduler_waits_for_busy_approved_gpus(self) -> None:
        experiment = {"fixed_runtime": {"python": "/python"}}
        with mock.patch.object(run_pipeline, "read_json", return_value=experiment):
            command = run_pipeline.scheduler_command(Path("/queue.jsonl"))
        self.assertEqual(command[-1], "--join-busy-pool")

    def test_profiler_holdouts_are_partitionable_and_cap_gradient_accumulation(self) -> None:
        selected = [
            {
                "job_id": "source-one",
                "campaign_id": "campaign",
                "hardware_id": "hardware",
                "request_id": "request-one",
                "model_id": "qwen3_14b",
                "train_type": "lora",
                "dataset_id": "longcontext_16384",
                "cutoff_len": 16384,
                "gpu_count": 1,
                "zero": "none",
                "gc": True,
                "mbs": 1,
                "target_gbs": 256,
                "packing": False,
            },
            {
                "job_id": "source-one-already-small",
                "campaign_id": "campaign",
                "hardware_id": "hardware",
                "request_id": "request-one-small",
                "model_id": "qwen3_14b",
                "train_type": "lora",
                "dataset_id": "longcontext_16384",
                "cutoff_len": 16384,
                "gpu_count": 1,
                "zero": "none",
                "gc": True,
                "mbs": 1,
                "target_gbs": 16,
                "packing": False,
            },
            {
                "job_id": "source-four",
                "campaign_id": "campaign",
                "hardware_id": "hardware",
                "request_id": "request-four",
                "model_id": "qwen3_14b",
                "train_type": "lora",
                "dataset_id": "longcontext_32768",
                "cutoff_len": 32768,
                "gpu_count": 4,
                "zero": "zero2",
                "gc": True,
                "mbs": 2,
                "target_gbs": 256,
                "packing": False,
            },
        ]
        report = {
            "throughput": {
                "decisions": [{"selected": row} for row in selected],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            throughput_path = Path(directory) / "throughput.jsonl"
            throughput_path.write_text("{}\n")
            experiment = {
                "measurement": {
                    "profiler_warmup_steps": 1,
                    "profiler_measure_steps": 3,
                }
            }
            with mock.patch.dict(
                run_pipeline.MATERIALIZED_PATHS,
                {"throughput": throughput_path},
            ), mock.patch.object(
                run_pipeline,
                "read_json",
                return_value=experiment,
            ), mock.patch.object(
                run_pipeline,
                "read_jsonl",
                return_value=selected,
            ), mock.patch.object(
                run_pipeline,
                "matching_job",
                side_effect=lambda row, jobs: row,
            ):
                jobs = run_pipeline.profiler_holdout_jobs(report)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["gpu_count"], 1)
        self.assertEqual(jobs[0]["source_target_gbs"], 256)
        self.assertEqual(jobs[0]["target_gbs"], 8)
        self.assertEqual(
            jobs[0]["target_gbs"] // (jobs[0]["gpu_count"] * jobs[0]["mbs"]),
            run_pipeline.PROFILER_HOLDOUT_MAX_GRADIENT_ACCUMULATION,
        )

    def test_controller_must_be_in_frozen_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            design = Path(directory) / "approval_design.json"
            design.write_text(json.dumps({"file_sha256": {}}))
            with self.assertRaises(PermissionError):
                run_pipeline.verify_controller_frozen(design)

    def test_prepare_screen_queue_reuses_healthy_formal_measurement(self) -> None:
        jobs = [
            {
                "job_id": "screen-reused",
                "request_id": "request",
                "model_id": "model",
                "train_type": "lora",
                "dataset_id": "data",
                "cutoff_len": 512,
                "gpu_count": 1,
                "zero": "none",
                "gc": False,
                "mbs": 1,
                "target_gbs": 16,
                "packing": False,
            },
            {
                "job_id": "screen-pending",
                "request_id": "request",
                "model_id": "model",
                "train_type": "lora",
                "dataset_id": "data",
                "cutoff_len": 512,
                "gpu_count": 1,
                "zero": "none",
                "gc": True,
                "mbs": 1,
                "target_gbs": 16,
                "packing": False,
            },
        ]
        reused_key = run_pipeline.throughput_physical_key(jobs[0])
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            run_pipeline, "PIPELINE_DIR", Path(directory)
        ), mock.patch.object(
            run_pipeline, "materialize", return_value=jobs
        ), mock.patch.object(
            run_pipeline, "completed_formal_throughput_keys", return_value={reused_key}
        ), mock.patch.object(
            run_pipeline, "pending_jobs", side_effect=lambda rows, accepted: list(rows)
        ), mock.patch.object(
            run_pipeline, "now_state", return_value={"current": {"status": "queued_waiting_approval"}}
        ):
            report = run_pipeline.prepare_throughput_screen_queue()

        self.assertEqual(report["candidate_jobs"], 2)
        self.assertEqual(report["substituted_by_healthy_formal"], 1)
        self.assertEqual(report["queued_jobs"], 1)

    def test_lora_zero3_dtype_failure_remains_retryable(self) -> None:
        family = {
            "job_id": "mem-known",
            "train_type": "lora",
            "zero": "zero3",
            "mbs_candidates": [1, 2],
        }
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            trial = results / "mem-known-mbs1"
            (trial / "metrics").mkdir(parents=True)
            (trial / "status.json").write_text(json.dumps({"classification": "failed"}))
            (trial / "metrics" / "events.rank0.jsonl").write_text(
                json.dumps(
                    {
                        "event": "failure",
                        "error": "TypeError: output tensor must have the same type as input tensor",
                    }
                )
                + "\n"
            )
            with mock.patch.object(run_pipeline, "RESULTS_DIR", results), mock.patch.object(
                run_pipeline, "read_jsonl", return_value=[family]
            ):
                progress = run_pipeline.memory_progress()
            self.assertEqual(progress["complete"], 0)
            self.assertEqual(progress["excluded"], [])
            self.assertEqual(progress["missing"], [family])


if __name__ == "__main__":
    unittest.main()
