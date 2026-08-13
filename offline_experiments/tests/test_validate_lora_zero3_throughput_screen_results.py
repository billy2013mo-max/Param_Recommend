from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import sha256_file, sha256_json, write_json, write_jsonl  # noqa: E402
from prepare_lora_zero3_throughput_delta import (  # noqa: E402
    key_record,
    physical_key,
)
from validate_lora_zero3_throughput_screen_results import (  # noqa: E402
    FP32_LORA_MESSAGE,
    output_paths_are_safe,
    validate_screen_delta_results,
)


class ValidateLoraZero3ThroughputScreenResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "artifacts"
        self.results = self.root / "results"
        self.delta_report_path = self.artifacts / "delta-report.json"
        self.input_queue_path = self.root / "runtime" / "pending-screen.jsonl"
        self.output_queue_path = self.root / "runtime" / "delta-screen.jsonl"
        self.approval_path = self.root / "config" / "APPROVED_TO_RUN.json"
        self.design_path = self.root / "runtime" / "approval_design.json"
        self.experiment_path = self.root / "config" / "experiment.json"
        self.provenance_path = self.artifacts / "provenance.json"
        self.job = self.make_job()
        self.runtime_identity = {
            "schema_version": 1,
            "python_executable": "/venv/bin/python",
            "packages": {"deepspeed": "0.19.2"},
        }
        self.runtime_fingerprint = sha256_json(self.runtime_identity)
        self.patch = {
            "all_passed": True,
            "installed_deepspeed_source_sha256": "deepspeed-source",
            "runtime_patcher_sha256": "runtime-patcher",
        }
        self.bound_inputs = self.create_bound_report_inputs()
        write_jsonl(self.input_queue_path, [self.job])
        write_jsonl(self.output_queue_path, [self.job])
        write_json(self.provenance_path, {"schema_version": 1, "commit": "test"})
        write_json(
            self.experiment_path,
            {
                "training_scope": {
                    "gpu_type": "NVIDIA H800 140GB HBM3",
                    "gpu_ids": [1, 2, 3, 4],
                    "gpu_counts": [1, 2, 4],
                    "max_gpu_count": 4,
                    "zero_by_gpu_count": {
                        "1": ["none"],
                        "2": ["zero2", "zero3"],
                        "4": ["zero2", "zero3"],
                    },
                },
                "measurement": {
                    "throughput_screen_warmup_steps": 2,
                    "throughput_screen_measure_steps": 4,
                    "performance_parallelism": "disjoint_gpu_masks",
                },
            },
        )
        self.write_delta_report()
        self.refresh_approval_binding()
        self.write_success_result()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def make_job() -> dict[str, object]:
        return {
            "job_id": "tputscreen-0123456789abcdef",
            "kind": "throughput_screen",
            "fidelity": "screen",
            "request_id": "tput-0123456789abcdef",
            "model_id": "qwen3_8b",
            "model_path": "/models/Qwen3-8B",
            "tokenizer_path": "/models/Qwen3-8B",
            "model_family": "qwen3",
            "model_parameters": 8_000_000_000,
            "template": "qwen3_nothink",
            "train_type": "lora",
            "dataset_id": "multiturn_4096",
            "cutoff_len": 4096,
            "gpu_count": 2,
            "zero": "zero3",
            "gc": False,
            "mbs": 2,
            "target_gbs": 64,
            "packing": False,
            "repeat": 0,
            "warmup_steps": 2,
            "measure_steps": 4,
            "parallel_class": "gpu_partitionable",
            "requires_external_node_idle": False,
        }

    def create_bound_report_inputs(self) -> dict[str, Path]:
        paths = {
            "baseline": self.root / "matrix" / "baseline.jsonl",
            "new_matrix": self.root / "matrix" / "new.jsonl",
            "pending_queue": self.input_queue_path,
            "recovery_plan": self.artifacts / "recovery-plan.json",
            "recovery_families": self.root / "runtime" / "families.jsonl",
            "boundary_evidence": self.artifacts / "recovery-results.json",
        }
        write_jsonl(paths["baseline"], [])
        write_jsonl(paths["new_matrix"], [self.job])
        write_json(paths["recovery_plan"], {"all_passed": True})
        write_jsonl(paths["recovery_families"], [{"job_id": "mem-one"}])
        write_json(paths["boundary_evidence"], {"all_passed": True})
        return paths

    def write_delta_report(self) -> None:
        inputs = {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in self.bound_inputs.items()
        }
        report = {
            "schema_version": 1,
            "purpose": "Prepare H800 LoRA + ZeRO-3 throughput-screen delta",
            "all_passed": True,
            "inputs": inputs,
            "counts": {
                "pending_jobs": 1,
                "delta_jobs": 1,
            },
            "checks": {"validated": True},
            "violations": {
                "duplicate_job_ids": [],
                "historical_key_hits": [],
            },
            "delta_candidates": [
                {
                    "job_id": self.job["job_id"],
                    "physical_key": key_record(physical_key(self.job)),
                    "checks": {"validated": True},
                    "all_passed": True,
                }
            ],
            "freeze_preparation": {"allowed_job_ids": [self.job["job_id"]]},
            "output_queue": {
                "path": str(self.output_queue_path),
                "sha256": sha256_file(self.output_queue_path),
                "jobs": 1,
                "written": True,
            },
        }
        write_json(self.delta_report_path, report)

    def refresh_approval_binding(self) -> None:
        files = {
            *self.bound_inputs.values(),
            self.delta_report_path,
            self.input_queue_path,
            self.output_queue_path,
            self.experiment_path,
            self.provenance_path,
        }
        manifest = {
            str(path.relative_to(self.root)): sha256_file(path)
            for path in sorted(files)
        }
        output_ids = [
            str(row["job_id"])
            for row in (
                json.loads(line)
                for line in self.output_queue_path.read_text().splitlines()
                if line.strip()
            )
        ]
        design = {
            "schema_version": 1,
            "file_sha256": manifest,
            "execution_order": ["throughput_screen_delta"],
            "allowed_job_ids": output_ids,
            "authorized_gpu_ids": [1, 2, 3, 4],
            "runtime_identity": self.runtime_identity,
            "runtime_fingerprint_sha256": self.runtime_fingerprint,
            "runtime_patch": self.patch,
            "throughput_screen_delta": {
                "jobs": len(output_ids),
                "allowed_job_ids": output_ids,
                "delta_report_sha256": sha256_file(self.delta_report_path),
                "input_queue_sha256": sha256_file(self.input_queue_path),
                "output_queue_sha256": sha256_file(self.output_queue_path),
                "recovery_results_validation_sha256": "recovery-results",
            },
        }
        write_json(self.design_path, design)
        write_json(
            self.approval_path,
            {
                "approved": True,
                "design_sha256": sha256_file(self.design_path),
                "execution_order": ["throughput_screen_delta"],
                "allowed_job_ids": output_ids,
                "resource_scope": {
                    "gpu_ids": [1, 2, 3, 4],
                    "max_gpu_count": 4,
                    "allow_gpu_ids_outside_pool": False,
                    "performance_parallelism": "disjoint_gpu_masks",
                },
            },
        )

    def rendered_job(self) -> dict[str, object]:
        return {
            **self.job,
            "runtime_model_path": "/runtime/Qwen3-8B",
            "max_steps": 6,
        }

    def base_status(self, classification: str) -> dict[str, object]:
        started = time.time() - 30.0
        return {
            "job_id": self.job["job_id"],
            "return_code": 0 if classification == "success" else 1,
            "classification": classification,
            "started_unix": started,
            "finished_unix": started + 20.0,
            "wall_seconds": 20.0,
            "gpu_mask": "1,2",
            "approval_design_sha256": sha256_file(self.design_path),
            "provenance_sha256": sha256_file(self.provenance_path),
            "runtime_fingerprint_sha256": self.runtime_fingerprint,
        }

    def write_core_result(self, classification: str) -> Path:
        result_dir = self.results / str(self.job["job_id"])
        runtime_path = result_dir / "runtime_identity.json"
        write_json(runtime_path, self.runtime_identity)
        write_json(
            result_dir / "rendered_run.json",
            {
                "job": self.rendered_job(),
                "gpu_mask": "1,2",
                "provenance_sha256": sha256_file(self.provenance_path),
                "runtime_identity_path": str(runtime_path),
                "runtime_fingerprint_sha256": self.runtime_fingerprint,
            },
        )
        write_json(result_dir / "status.json", self.base_status(classification))
        return result_dir

    def write_success_result(self) -> None:
        result_dir = self.write_core_result("success")
        (result_dir / "train.log").write_text(
            FP32_LORA_MESSAGE + ".\ntraining complete\n", encoding="utf-8"
        )
        metadata = self.rendered_job()
        for rank in (0, 1):
            write_json(
                result_dir / "metrics" / f"summary.rank{rank}.json",
                {
                    "rank": rank,
                    "local_rank": rank,
                    "world_size": 2,
                    "failure": None,
                    "total_steps": 6,
                    "measured_steps": 4,
                    "measured_seconds": 4.0,
                    "computed_tokens_per_second": 10.0,
                    "effective_tokens_per_second": 8.0,
                    "logical_samples_per_second": 2.0,
                    "max_allocated": 100,
                    "max_reserved": 120,
                    "measured_totals": {
                        "computed_tokens": 1000,
                        "logical_samples": 64,
                    },
                    "metadata": metadata,
                },
            )
        write_json(
            result_dir / "trainer_output" / "train_results.json",
            {"train_loss": 1.25},
        )

    def write_oom_result(
        self,
        log: str = "torch.cuda.OutOfMemoryError: CUDA out of memory\n",
    ) -> None:
        result_dir = self.write_core_result("oom")
        shutil.rmtree(result_dir / "metrics", ignore_errors=True)
        shutil.rmtree(result_dir / "trainer_output", ignore_errors=True)
        (result_dir / "train.log").write_text(log, encoding="utf-8")

    def validate(self) -> dict[str, object]:
        return validate_screen_delta_results(
            delta_report_path=self.delta_report_path,
            input_queue_path=self.input_queue_path,
            output_queue_path=self.output_queue_path,
            approval_path=self.approval_path,
            approval_design_path=self.design_path,
            experiment_path=self.experiment_path,
            provenance_path=self.provenance_path,
            results_dir=self.results,
            project_root=self.root,
            current_runtime_identity=self.runtime_identity,
            current_patch=self.patch,
        )

    def test_accepts_exact_success_with_bound_runtime_and_rank_metrics(self) -> None:
        report = self.validate()

        self.assertTrue(report["all_passed"])
        self.assertEqual(
            report["counts"],
            {
                "queue_jobs": 1,
                "success": 1,
                "oom": 0,
                "other_or_missing": 0,
            },
        )
        self.assertTrue(report["results"][0]["all_passed"])

    def test_accepts_oom_only_with_real_oom_evidence(self) -> None:
        self.write_oom_result()
        valid = self.validate()
        self.assertTrue(valid["all_passed"])
        self.assertEqual(valid["counts"]["oom"], 1)

        self.write_oom_result("unrelated launcher failure\n")
        invalid = self.validate()
        self.assertFalse(invalid["all_passed"])
        self.assertFalse(invalid["results"][0]["checks"]["oom_evidence_present"])

    def test_rejects_failed_incomplete_and_missing_terminal_status(self) -> None:
        status_path = self.results / str(self.job["job_id"]) / "status.json"
        for classification in ("failed", "incomplete_metrics"):
            status = self.base_status(classification)
            write_json(status_path, status)
            report = self.validate()
            self.assertFalse(report["all_passed"])
            self.assertFalse(
                report["checks"]["no_non_oom_failure_or_incomplete_result"]
            )

        status_path.unlink()
        missing = self.validate()
        self.assertFalse(missing["all_passed"])
        self.assertEqual(missing["results"][0]["classification"], "missing")

    def test_rejects_rendered_payload_drift_and_queue_duplication(self) -> None:
        rendered_path = self.results / str(self.job["job_id"]) / "rendered_run.json"
        rendered = json.loads(rendered_path.read_text())
        rendered["job"]["mbs"] = 4
        write_json(rendered_path, rendered)
        payload_drift = self.validate()
        self.assertFalse(payload_drift["all_passed"])
        self.assertFalse(
            payload_drift["results"][0]["checks"][
                "rendered_payload_matches_exact_queue_row"
            ]
        )

        write_jsonl(self.output_queue_path, [self.job, self.job])
        duplicate = self.validate()
        self.assertFalse(duplicate["checks"]["queue_job_ids_unique"])
        self.assertFalse(
            duplicate["checks"]["input_and_output_queue_payloads_identical"]
        )

    def test_rejects_runtime_approval_provenance_and_metric_tampering(self) -> None:
        result_dir = self.results / str(self.job["job_id"])
        status_path = result_dir / "status.json"
        status = json.loads(status_path.read_text())
        status["approval_design_sha256"] = "stale-approval"
        status["provenance_sha256"] = "stale-provenance"
        status["runtime_fingerprint_sha256"] = "stale-runtime"
        write_json(status_path, status)
        bindings = self.validate()["results"][0]["checks"]
        self.assertFalse(bindings["approval_design_bound"])
        self.assertFalse(bindings["provenance_bound"])
        self.assertFalse(bindings["runtime_fingerprint_bound"])

        self.write_success_result()
        summary_path = result_dir / "metrics" / "summary.rank1.json"
        summary = json.loads(summary_path.read_text())
        summary["measured_steps"] = 3
        summary["metadata"]["job_id"] = "copied-result"
        write_json(summary_path, summary)
        metrics = self.validate()["results"][0]["checks"]
        self.assertFalse(metrics["screen_steps_exact"])
        self.assertFalse(metrics["metric_metadata_matches_rendered_job"])

    def test_rejects_report_or_matrix_binding_drift(self) -> None:
        write_jsonl(self.bound_inputs["new_matrix"], [{**self.job, "mbs": 1}])
        report = self.validate()
        self.assertFalse(report["checks"]["delta_report_inputs_still_hash_bound"])
        self.assertFalse(report["checks"]["queue_rows_match_bound_new_matrix"])

    def test_outputs_are_restricted_to_distinct_artifact_json_files(self) -> None:
        outputs = [
            self.artifacts / "validation.json",
            self.artifacts / "approval-snapshot.json",
            self.artifacts / "design-snapshot.json",
        ]
        self.assertTrue(
            output_paths_are_safe(
                outputs,
                [self.approval_path, self.design_path],
                self.root,
            )
        )
        self.assertFalse(
            output_paths_are_safe(
                [outputs[0], outputs[0], outputs[2]],
                [self.approval_path],
                self.root,
            )
        )
        self.assertFalse(
            output_paths_are_safe(
                [self.approval_path, outputs[1], outputs[2]],
                [self.approval_path],
                self.root,
            )
        )


if __name__ == "__main__":
    unittest.main()
