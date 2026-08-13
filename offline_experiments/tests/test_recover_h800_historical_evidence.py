from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_h800_calibration_readiness import audit  # noqa: E402
from common import sha256_file, sha256_json  # noqa: E402
from export_h800_observations import export_observations  # noqa: E402
from recover_h800_historical_evidence import (  # noqa: E402
    _summary_counts,
    recover,
    validate_recovery_report,
    write_report,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


class HistoricalProject:
    def __init__(self, root: Path) -> None:
        self.root = root
        write_json(
            root / "config" / "hardware.json",
            {
                "gpu_id": "local_h800_140g",
                "name_reported_by_driver": "NVIDIA H800",
                "memory_bytes_reported_by_torch": 150_142_189_568,
            },
        )
        write_json(
            root / "config" / "experiment.json",
            {"training_scope": {"gpu_type": "NVIDIA H800 140GB HBM3"}},
        )
        write_json(root / "artifacts" / "model_inventory.json", {"models": []})
        write_json(root / "artifacts" / "dataset_analysis.json", {"datasets": {}})
        (root / "artifacts" / "nvidia_topology.txt").write_text(
            "GPU0 X\n", encoding="utf-8"
        )
        (root / "EXPERIMENT_DESIGN.md").write_text(
            "fixed runtime and holdout\n", encoding="utf-8"
        )
        (root / "PIPELINE.md").write_text(
            "leave one scenario out\n", encoding="utf-8"
        )
        write_json(
            root / "artifacts" / "stage_decisions.json",
            {
                "resource_holdout": {
                    "status": "evaluated",
                    "method": "leave-one scenario out",
                    "folds": [
                        {
                            "held_out_scenario": ["model", "full", "data", 16],
                            "train_configurations": 2,
                            "test_configurations": 1,
                        }
                    ],
                }
            },
        )
        write_json(
            root / "artifacts" / "profiler_calibration.json",
            {
                "status": "pending",
                "calibration_points": [],
                "evaluation_points": [],
            },
        )
        self.provenance_path = root / "artifacts" / "provenance.json"
        write_json(
            self.provenance_path,
            {"runtime_identity": {"packages": {"torch": "test"}}},
        )
        self.approval_path = root / "artifacts" / "test_approval_design.json"
        write_json(
            self.approval_path,
            {"allowed_job_ids": ["legacy-job"], "runtime": "test"},
        )

    def add_result(
        self,
        *,
        job_id: str = "legacy-job",
        classification: str = "success",
        scheduler_closed: bool = True,
        stale_failure_outside_window: bool = False,
        kind: str = "throughput",
        profiler_role: str | None = None,
        zero: str = "none",
        deepspeed_stage: int | None = None,
    ) -> None:
        job = {
            "job_id": job_id,
            "kind": kind,
            "gpu_count": 1,
            "model_id": "model",
            "dataset_id": "data",
            "zero": zero,
            "train_type": "full",
            "gc": False,
            "mbs": 1,
            "target_gbs": 16,
            "cutoff_len": 512,
            "packing": False,
        }
        if profiler_role is not None:
            job["profiler_role"] = profiler_role
        runtime_job = {**job, "warmup_steps": 1, "measure_steps": 2, "max_steps": 3}
        input_job = dict(job)
        write_json(self.root / "runtime" / "jobs" / f"input-{job_id}.json", input_job)
        write_json(self.root / "runtime" / "jobs" / f"{job_id}.json", runtime_job)
        config_path = self.root / "runtime" / "configs" / f"{job_id}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
                    "bf16": True,
                    "fp16": False,
                    "flash_attn": "fa3",
                    "enable_liger_kernel": True,
                    "optim": "adamw_torch_fused",
                    "dataset": "data",
                    "cutoff_len": 512,
                    "per_device_train_batch_size": 1,
                    "finetuning_type": "full",
                    "packing": False,
                    "gradient_checkpointing": False,
                }
        if zero != "none":
            deepspeed_path = self.root / "runtime" / "configs" / f"{job_id}.ds.json"
            write_json(
                deepspeed_path,
                {"zero_optimization": {"stage": deepspeed_stage}},
            )
            config["deepspeed"] = str(deepspeed_path)
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=True),
            encoding="utf-8",
        )
        result = self.root / "results" / job_id
        result.mkdir(parents=True)
        runtime_identity = {
            "python_executable": "/venv/bin/python",
            "packages": {"torch": "test"},
        }
        write_json(result / "runtime_identity.json", runtime_identity)
        runtime_sha = sha256_json(runtime_identity)
        provenance_sha = sha256_file(self.provenance_path)
        approval_sha = sha256_file(self.approval_path)
        rendered = {
            "job": runtime_job,
            "gpu_mask": "1",
            "config_path": str(config_path),
            "environment": {"FA3_VARIANT": "orig", "ENABLE_CCE": "0"},
            "provenance_sha256": provenance_sha,
            "runtime_fingerprint_sha256": runtime_sha,
        }
        return_code = 0 if classification == "success" else 1
        status = {
            "job_id": job_id,
            "classification": classification,
            "return_code": return_code,
            "started_unix": 2_000_000_000.0,
            "finished_unix": 2_000_000_100.0,
            "wall_seconds": 100.0,
            "gpu_mask": "1",
            "approval_design_sha256": approval_sha,
            "provenance_sha256": provenance_sha,
            "runtime_fingerprint_sha256": runtime_sha,
        }
        write_json(result / "rendered_run.json", rendered)
        write_json(result / "status.json", status)
        events = []
        if stale_failure_outside_window:
            events.append(
                {
                    "event": "failure",
                    "rank": 0,
                    "time_unix": 1_999_999_000.0,
                    "error": "CUDA out of memory from abandoned attempt",
                }
            )
        events.append(
            {
                "event": "train_begin",
                "rank": 0,
                "time_unix": 2_000_000_010.0,
                "world_size": 1,
                "metadata": runtime_job,
            }
        )
        if classification == "success":
            events.append(
                {
                    "event": "step_end",
                    "rank": 0,
                    "time_unix": 2_000_000_020.0,
                    "is_warmup": False,
                    "step_seconds": 1.0,
                    "tokens": {
                        "computed_tokens": 10,
                        "effective_tokens": 9,
                        "logical_samples": 1,
                    },
                    "memory": {"max_allocated": 100, "max_reserved": 120},
                }
            )
            write_json(
                result / "metrics" / "summary.rank0.json",
                {
                    "rank": 0,
                    "world_size": 1,
                    "failure": None,
                    "measured_steps": 1,
                    "metadata": runtime_job,
                },
            )
            log = "training complete\n"
        else:
            events.append(
                {
                    "event": "failure",
                    "rank": 0,
                    "time_unix": 2_000_000_020.0,
                    "error": "torch.OutOfMemoryError: CUDA out of memory",
                }
            )
            log = "torch.OutOfMemoryError: CUDA out of memory\n"
        write_jsonl(result / "metrics" / "events.rank0.jsonl", events)
        (result / "train.log").write_text(log, encoding="utf-8")
        (result / "nvidia_smi.csv").write_text("time,gpu\n1,1\n", encoding="utf-8")

        if scheduler_closed:
            write_jsonl(
                self.root / "runtime" / "scheduler_events.jsonl",
                [
                    {
                        "event": "trial_start",
                        "job_id": job_id,
                        "gpu_mask": [1],
                        "time_unix": 1_999_999_999.0,
                    },
                    {
                        "event": "trial_end",
                        "job_id": job_id,
                        "gpu_mask": [1],
                        "classification": classification,
                        "return_code": 0 if classification == "success" else 2,
                        "launcher_output_tail": json.dumps(status, sort_keys=True),
                        "time_unix": 2_000_000_100.1,
                    },
                ],
            )
        else:
            write_jsonl(self.root / "runtime" / "scheduler_events.jsonl", [])

    def canonical_path(self) -> Path:
        path = self.root / "artifacts" / "canonical_h800_observations.jsonl"
        write_jsonl(path, export_observations(self.root))
        return path


class HistoricalRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_complete_archive_and_scheduler_are_legacy_verified(self) -> None:
        project = HistoricalProject(self.root)
        project.add_result()
        observations = project.canonical_path()

        report = recover(observations, self.root)
        record = report["records"][0]

        self.assertEqual(record["evidence_tier"], "legacy_verified")
        self.assertEqual(
            record["attempt_integrity"]["strength"], "scheduler_verified"
        )
        self.assertTrue(record["archive_completeness"]["full_payload_set"])
        self.assertEqual(
            record["measurement_eligibility"]["class"], "calibration_candidate"
        )
        self.assertTrue(record["measurement_eligibility"]["throughput_primary"])
        self.assertEqual(validate_recovery_report(report, observations), [])

    def test_missing_scheduler_is_consistent_not_rejected(self) -> None:
        project = HistoricalProject(self.root)
        project.add_result(scheduler_closed=False)
        observations = project.canonical_path()

        record = recover(observations, self.root)["records"][0]

        self.assertEqual(record["evidence_tier"], "legacy_consistent")
        self.assertEqual(record["attempt_integrity"]["strength"], "self_closed")
        self.assertTrue(record["measurement_eligibility"]["feasibility"])

    def test_orphan_or_mismatched_scheduler_end_cannot_verify_attempt(self) -> None:
        project = HistoricalProject(self.root)
        project.add_result()
        observations = project.canonical_path()
        status = json.loads(
            (self.root / "results" / "legacy-job" / "status.json").read_text(
                encoding="utf-8"
            )
        )
        write_jsonl(
            self.root / "runtime" / "scheduler_events.jsonl",
            [
                {
                    "event": "trial_end",
                    "job_id": "legacy-job",
                    "gpu_mask": [9],
                    "classification": "success",
                    "return_code": 999,
                    "launcher_output_tail": json.dumps(status, sort_keys=True),
                    "time_unix": status["started_unix"] - 1,
                }
            ],
        )

        record = recover(observations, self.root)["records"][0]

        self.assertEqual(record["attempt_integrity"]["strength"], "self_closed")
        self.assertEqual(record["attempt_integrity"]["exact_closures"], 0)

    def test_stale_oom_outside_window_cannot_label_final_attempt(self) -> None:
        project = HistoricalProject(self.root)
        project.add_result(
            classification="success", stale_failure_outside_window=True
        )
        observations = project.canonical_path()

        record = recover(observations, self.root)["records"][0]

        self.assertEqual(record["terminal_evidence"]["class"], "success")
        self.assertFalse(record["terminal_evidence"]["cuda_oom_confirmed"])
        self.assertEqual(
            record["attempt_integrity"]["event_window"]["events_outside_window"],
            1,
        )

    def test_sidecar_and_current_source_tampering_fail_closed(self) -> None:
        project = HistoricalProject(self.root)
        project.add_result()
        observations = project.canonical_path()
        report = recover(observations, self.root)

        report["records"][0]["evidence_tier"] = "legacy_consistent"
        reasons = validate_recovery_report(report, observations)
        self.assertIn("historical_recovery_report_hash_mismatch", reasons)
        self.assertIn("record_0_evidence_hash_mismatch", reasons)

        report = recover(observations, self.root)
        (self.root / "results" / "legacy-job" / "train.log").write_text(
            "changed\n", encoding="utf-8"
        )
        reasons = validate_recovery_report(
            report, observations, verify_source_files=True
        )
        self.assertIn("record_0_source_artifact_missing_or_changed", reasons)

        report = recover(observations, self.root)
        (self.root / "runtime" / "scheduler_events.jsonl").write_text(
            "", encoding="utf-8"
        )
        reasons = validate_recovery_report(
            report, observations, verify_source_files=True
        )
        self.assertIn("context_source_0_missing_or_changed", reasons)

    def test_validator_binds_record_set_recovery_id_and_trusted_root(self) -> None:
        project = HistoricalProject(self.root)
        project.add_result()
        observations = project.canonical_path()

        report = recover(observations, self.root)
        report["records"] = []
        report["counts"] = _summary_counts([])
        report.pop("report_sha256")
        report["report_sha256"] = sha256_json(report)
        reasons = validate_recovery_report(
            report, observations, project_root=self.root
        )
        self.assertIn("historical_recovery_record_observation_set_mismatch", reasons)

        report = recover(observations, self.root)
        record = report["records"][0]
        record["recovery_id"] = "f" * 64
        record.pop("evidence_sha256")
        record["evidence_sha256"] = sha256_json(record)
        report.pop("report_sha256")
        report["report_sha256"] = sha256_json(report)
        reasons = validate_recovery_report(
            report, observations, project_root=self.root
        )
        self.assertIn("record_0_recovery_id_mismatch", reasons)

        report = recover(observations, self.root)
        report["project_root"] = "/"
        report.pop("report_sha256")
        report["report_sha256"] = sha256_json(report)
        reasons = validate_recovery_report(
            report, observations, project_root=self.root
        )
        self.assertIn("historical_recovery_project_root_mismatch", reasons)

    def test_canonical_measurement_tamper_is_rejected_by_raw_rebuild(self) -> None:
        project = HistoricalProject(self.root)
        project.add_result()
        observations = project.canonical_path()
        row = json.loads(observations.read_text(encoding="utf-8"))
        row["measurements"]["rates"]["computed_tokens_per_second"] = 9.99e99
        write_jsonl(observations, [row])

        record = recover(observations, self.root)["records"][0]

        self.assertFalse(
            record["core_checks"]["canonical_observation_matches_raw_rebuild"]
        )
        self.assertEqual(
            record["measurement_eligibility"]["class"], "rejected_terminal"
        )

    def test_missing_status_after_export_rejects_record_without_crashing(self) -> None:
        project = HistoricalProject(self.root)
        project.add_result()
        observations = project.canonical_path()
        (self.root / "results" / "legacy-job" / "status.json").unlink()

        report = recover(observations, self.root)
        record = report["records"][0]

        self.assertFalse(
            record["core_checks"][
                "status_render_source_hashes_match_canonical"
            ]
        )
        self.assertEqual(
            record["measurement_eligibility"]["class"], "rejected_terminal"
        )

    def test_readiness_keeps_native_zero_and_surfaces_recovery(self) -> None:
        project = HistoricalProject(self.root)
        project.add_result()
        observations = project.canonical_path()
        recovery_path = self.root / "artifacts" / "historical.json"
        write_report(recovery_path, recover(observations, self.root))

        readiness = audit(observations, recovery_path)

        self.assertEqual(readiness["counts"]["complete_fingerprint"], 0)
        self.assertEqual(
            readiness["historical_recovery"]["counts"]["display_buckets"][
                "legacy_verified"
            ],
            1,
        )
        self.assertFalse(readiness["historical_recovery"]["calibration_publishable"])


if __name__ == "__main__":
    unittest.main()
