from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_job  # noqa: E402
from common import sha256_file, sha256_json  # noqa: E402
from export_h800_observations import (  # noqa: E402
    UnsupportedHardwareError,
    export_observations,
    validate_canonical_observation,
    write_jsonl,
)


H800_MEMORY_BYTES = 150_142_189_568
RTX4090_MEMORY_BYTES = 24 * 1024**3


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )


def runtime_inventory(logical_elements: int = 6) -> dict:
    return {
        "model_class": "tests.TinyModel",
        "unique_tensor_count": 1,
        "logical_parameter_elements": logical_elements,
        "trainable_parameter_elements": logical_elements,
        "frozen_parameter_elements": 0,
        "tensors": [
            {
                "tensor_id": "tensor-000000",
                "canonical_name": "weight",
                "aliases": ["weight"],
                "logical_numel": logical_elements,
                "logical_shape": [logical_elements],
                "dtype": "torch.bfloat16",
                "requires_grad": True,
            }
        ],
        "module_groups": [
            {
                "module_name": "<root>",
                "module_class": "tests.TinyModel",
                "tensor_ids": ["tensor-000000"],
                "logical_parameter_elements": logical_elements,
                "trainable_parameter_elements": logical_elements,
            }
        ],
        "largest_module_group": {
            "module_name": "<root>",
            "logical_parameter_elements": logical_elements,
        },
    }


def device_attestation(
    rank: int,
    *,
    gpu_name: str = "NVIDIA H800",
    memory_bytes: int = H800_MEMORY_BYTES,
) -> dict:
    capability = {"major": 9, "minor": 0}
    if "4090" in gpu_name:
        capability = {"major": 8, "minor": 9}
    return {
        "schema": "sft_runtime_device_attestation",
        "schema_version": 1,
        "availability": "available",
        "source": "torch.cuda.get_device_properties",
        "local_rank": rank,
        "visible_device_index": rank,
        "name": gpu_name,
        "total_memory_bytes": memory_bytes,
        "compute_capability": capability,
        "uuid": f"GPU-{rank:032x}",
        "uuid_unavailable_reason": None,
        "unavailable_reason": None,
    }


class EvidenceProject:
    """Build immutable v2 attempts using the launcher's production helpers."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.hardware = {
            "schema_version": 1,
            "gpu_id": "local_h800_140g",
            "name_reported_by_driver": "NVIDIA H800",
            "memory_bytes_reported_by_torch": H800_MEMORY_BYTES,
        }
        self.experiment = {
            "schema_version": 1,
            "training_scope": {"gpu_type": "NVIDIA H800 140GB HBM3"},
            "fixed_runtime": {
                "bf16": True,
                "flash_attn": "fa3",
                "gradient_checkpointing": True,
            },
        }
        self.provenance = {
            "schema_version": 1,
            "runtime_identity": {"packages": {"torch": "test"}},
            "runtime_fingerprint_sha256": "a" * 64,
        }
        write_json(root / "config" / "hardware.json", self.hardware)
        write_json(root / "config" / "experiment.json", self.experiment)
        write_json(
            root / "artifacts" / "model_inventory.json",
            {"schema_version": 1, "models": [{"id": "model"}]},
        )
        write_json(
            root / "artifacts" / "dataset_analysis.json",
            {"schema_version": 1, "datasets": {"data": {"sha256": "data"}}},
        )
        write_json(root / "artifacts" / "provenance.json", self.provenance)
        (root / "artifacts" / "nvidia_topology.txt").write_text(
            "GPU0 GPU1 NV8\n", encoding="utf-8"
        )

    @staticmethod
    def attempt_id(seed: str) -> str:
        return sha256_json({"attempt": seed})[:20]

    def _job(
        self,
        job_id: str,
        *,
        role: str,
        split_unit_id: str,
        split_policy: str,
        model_id: str,
        cutoff_len: int,
        mbs: int,
    ) -> dict:
        return {
            "job_id": job_id,
            "gpu_count": 2,
            "gpu_type": "NVIDIA H800 140GB HBM3",
            "model_id": model_id,
            "dataset_id": "data",
            "zero": "zero3",
            "train_type": "full",
            "mbs": mbs,
            "target_gbs": 16,
            "cutoff_len": cutoff_len,
            "packing": False,
            "calibration_partition": {
                "role": role,
                "split_unit_id": split_unit_id,
                "policy": split_policy,
            },
        }

    @staticmethod
    def _approval_check(job: dict) -> dict:
        job_sha256 = run_job.canonical_job_sha256(job)
        return {
            "approval_path": "config/APPROVED_TO_RUN.json",
            "approval_sha256": "1" * 64,
            "approval_design_path": "artifacts/approved_design.json",
            "design_sha256": "2" * 64,
            "queue": {
                "path": "runtime/queue.json",
                "actual_sha256": "3" * 64,
                "job_ids": [job["job_id"]],
                "job_payload_sha256": {job["job_id"]: job_sha256},
            },
        }

    def add_attempt(
        self,
        job_id: str,
        *,
        attempt_id: str | None = None,
        authorization_mode: str = "approved",
        role: str = "calibration",
        split_unit_id: str = "unit-a",
        split_policy: str = "model-length-disjoint-v1",
        model_id: str = "qwen3_8b",
        cutoff_len: int = 4096,
        mbs: int = 2,
        gpu_name: str = "NVIDIA H800",
        memory_bytes: int | None = None,
    ) -> Path:
        attempt_id = attempt_id or self.attempt_id(job_id)
        memory_bytes = memory_bytes or (
            RTX4090_MEMORY_BYTES if "4090" in gpu_name else H800_MEMORY_BYTES
        )
        job = self._job(
            job_id,
            role=role,
            split_unit_id=split_unit_id,
            split_policy=split_policy,
            model_id=model_id,
            cutoff_len=cutoff_len,
            mbs=mbs,
        )
        attempt = self.root / "results" / job_id / "attempts" / attempt_id
        attempt.mkdir(parents=True)
        snapshots = attempt / "input_snapshots"
        snapshots.mkdir()
        snapshot_paths = {
            "declared_hardware": snapshots / "declared_hardware.json",
            "declared_model_manifest": snapshots / "declared_model_manifest.json",
            "dataset_manifest": snapshots / "dataset_manifest.json",
            "provenance": snapshots / "provenance.json",
            "deepspeed_config": snapshots / "deepspeed_config.json",
        }
        snapshot_payloads = {
            "declared_hardware": self.hardware,
            "declared_model_manifest": {
                "schema_version": 1,
                "models": [{"id": "model"}],
            },
            "dataset_manifest": {
                "schema_version": 1,
                "datasets": {"data": {"sha256": "data"}},
            },
            "provenance": self.provenance,
            "deepspeed_config": {"zero_optimization": {"stage": 3}},
        }
        for name, payload in snapshot_payloads.items():
            write_json(snapshot_paths[name], payload)

        runtime_identity = {
            "python_executable": "/venv/bin/python",
            "python_prefix": "/venv",
            "python_version": "3.11",
            "packages": {"torch": "2.8.0", "deepspeed": "0.19.2"},
            "launcher_patch_sha256": "4" * 64,
        }
        runtime_identity_path = attempt / "runtime_identity.json"
        write_json(runtime_identity_path, runtime_identity)
        runtime_config_path = attempt / "runtime_config.yaml"
        runtime_config = {
            "bf16": True,
            "flash_attn": "fa3",
            "gradient_checkpointing": True,
            "torch_compile": False,
            "optim": "adamw_torch",
            "use_reentrant_gc": False,
            "deepspeed": str(snapshot_paths["deepspeed_config"]),
        }
        runtime_config_path.write_text(
            yaml.safe_dump(runtime_config, sort_keys=True), encoding="utf-8"
        )
        runtime_metadata_path = attempt / "job_metadata.json"
        write_json(
            runtime_metadata_path,
            {**job, "_execution_attempt_id": attempt_id},
        )
        topology_path = attempt / "nvidia_topology.txt"
        topology_path.write_text(
            "        GPU0 GPU1\nGPU0    X    NV8\nGPU1    NV8  X\n",
            encoding="utf-8",
        )
        capability = "8.9" if "4090" in gpu_name else "9.0"
        runtime_hardware = {
            "schema": run_job.RUNTIME_HARDWARE_SCHEMA,
            "job_id": job_id,
            "execution_attempt_id": attempt_id,
            "capture_mode": "live_pre_execution",
            "captured_unix": 90.0,
            "requested_physical_gpu_ids": [1, 2],
            "topology_sha256": sha256_file(topology_path),
            "devices": [
                {
                    "physical_index": rank + 1,
                    "uuid": f"GPU-{rank:032x}",
                    "name": gpu_name,
                    "memory_total_bytes": memory_bytes,
                    "compute_capability": capability,
                }
                for rank in range(2)
            ],
            "declared_hardware": {
                "gpu_id": "local_h800_140g",
                "campaign_gpu_type": "NVIDIA H800 140GB HBM3",
            },
            "checks": {"test_fixture": True},
            "all_passed": True,
            "calibration_hardware_eligible": True,
        }
        runtime_hardware_path = attempt / "runtime_hardware.json"
        write_json(runtime_hardware_path, runtime_hardware)
        environment = {
            "CUDA_VISIBLE_DEVICES": "1,2",
            "ENABLE_CCE": "1",
            "FA3_VARIANT": "default",
        }
        runtime_mechanism = run_job.runtime_mechanism_manifest(
            experiment=self.experiment,
            runtime_identity=runtime_identity,
            runtime_hardware=runtime_hardware,
            environment=environment,
            project_root=ROOT,
        )
        runtime_mechanism_path = attempt / "runtime_mechanism.json"
        write_json(runtime_mechanism_path, runtime_mechanism)
        authorization = run_job.execution_authorization(
            mode=authorization_mode,
            job=job,
            approval_check=(
                self._approval_check(job)
                if authorization_mode == "approved"
                else None
            ),
        )
        command = ["python", "scripts/train_entry.py"]
        execution_inputs = run_job.execution_inputs_manifest(
            job=job,
            execution_attempt_id=attempt_id,
            attempt_root=attempt,
            runtime_identity=runtime_identity,
            runtime_identity_path=runtime_identity_path,
            runtime_config=runtime_config_path,
            runtime_metadata=runtime_metadata_path,
            config=runtime_config,
            command=command,
            environment=environment,
            provenance=self.provenance,
            authorization=authorization,
            runtime_hardware=runtime_hardware,
            runtime_hardware_path=runtime_hardware_path,
            live_topology_path=topology_path,
            runtime_mechanism=runtime_mechanism,
            runtime_mechanism_path=runtime_mechanism_path,
            input_snapshots=snapshot_paths,
        )
        execution_inputs_path = attempt / "execution_inputs.json"
        write_json(execution_inputs_path, execution_inputs)

        metrics = attempt / "metrics"
        for rank in range(2):
            inventory = runtime_inventory()
            attestation = device_attestation(
                rank,
                gpu_name=gpu_name,
                memory_bytes=memory_bytes,
            )
            write_json(
                run_job.runtime_model_manifest_path(attempt, attempt_id, rank),
                {
                    "schema": "sft_runtime_model_manifest",
                    "schema_version": 2,
                    "job_id": job_id,
                    "execution_attempt_id": attempt_id,
                    "rank": rank,
                    "local_rank": rank,
                    "world_size": 2,
                    "training_mode": "full",
                    "inventory": inventory,
                    "inventory_sha256": sha256_json(inventory),
                    "device_attestation": attestation,
                    "device_attestation_sha256": sha256_json(attestation),
                },
            )
            write_json(
                metrics / f"summary.rank{rank}.json",
                {
                    "job_id": job_id,
                    "execution_attempt_id": attempt_id,
                    "rank": rank,
                    "world_size": 2,
                },
            )
            write_events(
                metrics / f"events.rank{rank}.jsonl",
                [
                    {
                        "event": "step_end",
                        "job_id": job_id,
                        "execution_attempt_id": attempt_id,
                        "rank": rank,
                        "local_rank": rank,
                        "world_size": 2,
                        "time_unix": 110.0 + rank,
                        "global_step": 1,
                        "is_warmup": False,
                        "step_seconds": 2.0 + rank,
                        "tokens": {
                            "computed_tokens": 10 * (rank + 1),
                            "effective_tokens": 9 * (rank + 1),
                            "label_tokens": 8 * (rank + 1),
                            "logical_samples": 2,
                            "physical_batches": 1,
                            "computed_attention_token_pairs": 100,
                            "effective_attention_token_pairs": 90,
                        },
                        "memory": {
                            "allocated": 100 + rank,
                            "max_allocated": 200 + rank,
                            "reserved": 300 + rank,
                            "max_reserved": 400 + rank,
                        },
                    }
                ],
            )
        log_path = attempt / "train.log"
        log_path.write_text("training complete\n", encoding="utf-8")
        classification = run_job.classify_execution(
            0,
            log_path,
            2,
            job_id=job_id,
            execution_attempt_id=attempt_id,
        )
        terminal = classification["evidence"]
        execution_manifest = run_job.finalize_execution_fingerprint(
            job=job,
            execution_attempt_id=attempt_id,
            execution_inputs=execution_inputs,
            execution_inputs_path=execution_inputs_path,
            result_dir=attempt,
            expected_ranks=2,
            terminal_classification=terminal,
            return_code=0,
        )
        execution_manifest_path = attempt / "execution_fingerprint.json"
        write_json(execution_manifest_path, execution_manifest)
        execution_fingerprint_sha256 = sha256_json(execution_manifest)
        runtime_fingerprint_sha256 = sha256_json(runtime_identity)
        provenance_sha256 = sha256_file(snapshot_paths["provenance"])
        approval_design_sha256 = (authorization.get("evidence") or {}).get(
            "approval_design_sha256"
        )
        rendered = {
            "schema": "sft_rendered_run/v2",
            "job": job,
            "execution_attempt_id": attempt_id,
            "authorization": authorization,
            "approval_design_sha256": approval_design_sha256,
            "calibration_eligible": execution_manifest["calibration_eligible"],
            "gpu_mask": "1,2",
            "config_path": str(runtime_config_path),
            "command": command,
            "environment": environment,
            "provenance_sha256": provenance_sha256,
            "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
            "execution_inputs_sha256": sha256_file(execution_inputs_path),
            "execution_fingerprint_path": "execution_fingerprint.json",
            "execution_fingerprint_sha256": execution_fingerprint_sha256,
            "execution_fingerprint_quality": "complete",
            "execution_fingerprint_errors": [],
        }
        status = {
            "schema": "sft_execution_status/v2",
            "job_id": job_id,
            "return_code": 0,
            "classification": classification["classification"],
            "classification_evidence": terminal,
            "started_unix": 100.0,
            "finished_unix": 200.0,
            "wall_seconds": 100.0,
            "gpu_mask": "1,2",
            "authorization_mode": authorization_mode,
            "calibration_eligible": execution_manifest["calibration_eligible"],
            "approval_design_sha256": approval_design_sha256,
            "provenance_sha256": provenance_sha256,
            "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
            "execution_attempt_id": attempt_id,
            "execution_inputs_sha256": sha256_file(execution_inputs_path),
            "execution_fingerprint_sha256": execution_fingerprint_sha256,
            "execution_fingerprint_quality": "complete",
            "execution_fingerprint_errors": [],
        }
        write_json(attempt / "rendered_run.json", rendered)
        write_json(attempt / "status.json", status)
        return attempt

    def add_legacy_result(self, job_id: str = "legacy-h800") -> Path:
        result = self.root / "results" / job_id
        result.mkdir(parents=True)
        config_path = result / "runtime_config.yaml"
        config_path.write_text("bf16: true\n", encoding="utf-8")
        job = {
            "job_id": job_id,
            "gpu_count": 1,
            "model_id": "model",
            "dataset_id": "data",
            "zero": "none",
            "train_type": "full",
            "mbs": 1,
            "cutoff_len": 512,
        }
        rendered = {
            "job": job,
            "gpu_mask": "1",
            "config_path": str(config_path),
            "environment": {},
        }
        status = {
            "job_id": job_id,
            "classification": "success",
            "return_code": 0,
            "started_unix": 100.0,
            "finished_unix": 200.0,
            "wall_seconds": 100.0,
            "gpu_mask": "1",
        }
        write_json(result / "rendered_run.json", rendered)
        write_json(result / "status.json", status)
        write_events(
            result / "metrics" / "events.rank0.jsonl",
            [
                {
                    "event": "step_end",
                    "rank": 0,
                    "time_unix": 110.0,
                    "is_warmup": False,
                    "step_seconds": 1.0,
                    "tokens": {"computed_tokens": 10},
                    "memory": {"max_allocated": 100},
                }
            ],
        )
        return result


class H800ObservationExporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def project(self, name: str) -> EvidenceProject:
        return EvidenceProject(self.base / name)

    def test_approved_h800_attempt_is_complete_and_calibration_usable(self) -> None:
        project = self.project("approved")
        project.add_attempt("approved-h800")

        rows = export_observations(project.root)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["schema"], "sft_efficiency_observation/v2")
        self.assertEqual(row["hardware"]["gpu_family"], "H800")
        self.assertEqual(row["fingerprint"]["quality"], "complete")
        self.assertTrue(row["fingerprint"]["evidence_verified"])
        self.assertTrue(row["fingerprint"]["calibration_evidence_eligible"])
        self.assertTrue(row["outcome"]["usable_for_feasibility_calibration"])
        self.assertTrue(row["outcome"]["usable_for_throughput_calibration"])
        self.assertTrue(row["quality"]["event_attempt_binding_complete"])
        self.assertEqual(row["measurements"]["work"]["computed_tokens"], 30)
        self.assertEqual(validate_canonical_observation(row), [])

    def test_unified_fit_role_is_narrowly_normalized_to_calibration(self) -> None:
        project = self.project("unified-fit")
        project.add_attempt(
            "unified-fit-h800",
            role="fit",
            split_policy="unified_model_source_grouped_v1",
        )

        row = export_observations(project.root)[0]

        self.assertEqual(
            row["configuration"]["calibration_partition"],
            {
                "role": "calibration",
                "split_unit_id": "unit-a",
                "policy": "unified_model_source_grouped_v1",
            },
        )
        self.assertTrue(row["outcome"]["usable_for_feasibility_calibration"])
        self.assertTrue(row["outcome"]["usable_for_throughput_calibration"])
        self.assertEqual(validate_canonical_observation(row), [])

    def test_fit_role_is_not_accepted_for_unrelated_partition_policy(self) -> None:
        project = self.project("unrelated-fit")
        project.add_attempt(
            "unrelated-fit-h800",
            role="fit",
            split_policy="some_other_policy",
        )

        row = export_observations(project.root)[0]

        self.assertIsNone(row["configuration"]["calibration_partition"])
        self.assertFalse(row["outcome"]["usable_for_feasibility_calibration"])
        self.assertFalse(row["outcome"]["usable_for_throughput_calibration"])
        self.assertIn(
            "authorized_job_calibration_role_missing",
            row["outcome"]["calibration_exclusion_reasons"],
        )
        self.assertEqual(validate_canonical_observation(row), [])

    def test_smoke_attempt_can_be_complete_but_never_calibrates(self) -> None:
        project = self.project("smoke")
        project.add_attempt("smoke-h800", authorization_mode="smoke")

        row = export_observations(project.root)[0]

        self.assertEqual(row["fingerprint"]["quality"], "complete")
        self.assertFalse(row["fingerprint"]["calibration_evidence_eligible"])
        self.assertFalse(row["outcome"]["usable_for_feasibility_calibration"])
        self.assertFalse(row["outcome"]["usable_for_throughput_calibration"])
        self.assertIn(
            "execution_evidence_not_calibration_eligible",
            row["outcome"]["calibration_exclusion_reasons"],
        )
        self.assertEqual(validate_canonical_observation(row), [])

    def test_legacy_attempt_is_diagnostic_only_even_with_measured_steps(self) -> None:
        project = self.project("legacy")
        project.add_legacy_result()

        row = export_observations(project.root)[0]

        self.assertEqual(row["fingerprint"]["quality"], "legacy_incomplete")
        self.assertTrue(row["outcome"]["raw_throughput_measurement_available"])
        self.assertFalse(row["outcome"]["usable_for_feasibility_calibration"])
        self.assertFalse(row["outcome"]["usable_for_throughput_calibration"])
        self.assertTrue(row["quality"]["requires_legacy_compatibility_review"])
        self.assertEqual(validate_canonical_observation(row), [])

    def test_actual_4090_attestation_is_rejected_from_h800_export(self) -> None:
        project = self.project("actual-4090")
        project.add_attempt(
            "wrong-live-device",
            gpu_name="NVIDIA GeForce RTX 4090",
            memory_bytes=RTX4090_MEMORY_BYTES,
        )

        with self.assertRaises(UnsupportedHardwareError):
            export_observations(project.root)

    def test_bound_evidence_tampering_fails_closed(self) -> None:
        def tamper_rendered_job(attempt: Path) -> None:
            path = attempt / "rendered_run.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["job"]["mbs"] += 1
            write_json(path, value)

        def tamper_authorization(attempt: Path) -> None:
            path = attempt / "execution_inputs.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["authorization"]["evidence"]["approval_sha256"] = "0" * 64
            write_json(path, value)

        def tamper_hardware(attempt: Path) -> None:
            path = attempt / "runtime_hardware.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["devices"][0]["memory_total_bytes"] -= 1
            write_json(path, value)

        def tamper_rank_binding(attempt: Path) -> None:
            path = next((attempt / "metrics").glob("runtime_model_manifest.*.rank0.json"))
            value = json.loads(path.read_text(encoding="utf-8"))
            value["job_id"] = "substituted-job"
            write_json(path, value)

        def tamper_inventory(attempt: Path) -> None:
            path = next((attempt / "metrics").glob("runtime_model_manifest.*.rank0.json"))
            value = json.loads(path.read_text(encoding="utf-8"))
            value["inventory"]["logical_parameter_elements"] += 1
            write_json(path, value)

        cases: dict[str, Callable[[Path], None]] = {
            "job": tamper_rendered_job,
            "authorization": tamper_authorization,
            "hardware": tamper_hardware,
            "rank": tamper_rank_binding,
            "inventory": tamper_inventory,
        }
        for name, tamper in cases.items():
            with self.subTest(evidence=name):
                project = self.project(f"tamper-{name}")
                attempt = project.add_attempt(f"tamper-{name}")
                tamper(attempt)

                row = export_observations(project.root)[0]

                self.assertEqual(row["fingerprint"]["quality"], "incomplete")
                self.assertFalse(row["fingerprint"]["evidence_verified"])
                self.assertFalse(
                    row["outcome"]["usable_for_feasibility_calibration"]
                )
                self.assertFalse(
                    row["outcome"]["usable_for_throughput_calibration"]
                )
                self.assertTrue(row["fingerprint"]["quality_reasons"])
                self.assertEqual(validate_canonical_observation(row), [])

    def test_jsonl_writer_is_deterministic_and_replaces_only_output(self) -> None:
        project = self.project("writer")
        project.add_attempt("writer-h800")
        rows = export_observations(project.root)
        output = project.root / "exports" / "observations.jsonl"

        self.assertEqual(write_jsonl(output, rows), 1)
        first = output.read_bytes()
        self.assertEqual(write_jsonl(output, copy.deepcopy(rows)), 1)
        self.assertEqual(output.read_bytes(), first)
        self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
