from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_job  # noqa: E402
from common import sha256_file, sha256_json  # noqa: E402


ATTEMPT_ID = "a" * 20
MEMORY_BYTES = 150_142_189_568


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def inventory(logical_elements: int = 6) -> dict:
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


def device_attestation(rank: int, *, uuid: str | None = None) -> dict:
    value = {
        "schema": "sft_runtime_device_attestation",
        "schema_version": 1,
        "availability": "available",
        "source": "torch.cuda.get_device_properties",
        "local_rank": rank,
        "visible_device_index": rank,
        "name": "NVIDIA H800",
        "total_memory_bytes": MEMORY_BYTES,
        "compute_capability": {"major": 9, "minor": 0},
        "uuid": uuid or f"GPU-{rank:032x}",
        "uuid_unavailable_reason": None,
        "unavailable_reason": None,
    }
    return value


class ExecutionFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.attempt = self.root / "results" / "h800-test" / "attempts" / ATTEMPT_ID
        self.attempt.mkdir(parents=True)
        self.job = {
            "job_id": "h800-test",
            "train_type": "full",
            "gpu_count": 2,
        }
        self.runtime_identity = {
            "packages": {"deepspeed": "0.19.2"},
            "launcher_patch_sha256": "a" * 64,
        }
        self.runtime_hardware = self._runtime_hardware()
        self.runtime_mechanism = {
            "schema": run_job.RUNTIME_MECHANISM_SCHEMA,
            "source_manifest": {},
            "source_manifest_sha256": sha256_json({}),
        }
        self.runtime_mechanism["fingerprint_sha256"] = sha256_json(
            self.runtime_mechanism
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _runtime_hardware(self, count: int = 2) -> dict:
        return {
            "schema": run_job.RUNTIME_HARDWARE_SCHEMA,
            "job_id": "h800-test",
            "execution_attempt_id": ATTEMPT_ID,
            "capture_mode": "live_pre_execution",
            "topology_sha256": "d" * 64,
            "devices": [
                {
                    "physical_index": rank + 1,
                    "uuid": f"GPU-{rank:032x}",
                    "name": "NVIDIA H800",
                    "memory_total_bytes": MEMORY_BYTES,
                    "compute_capability": "9.0",
                }
                for rank in range(count)
            ],
            "all_passed": True,
            "calibration_hardware_eligible": True,
        }

    def _authorization(self, mode: str = "approved") -> dict:
        evidence = (
            {
                "approval_design_sha256": "e" * 64,
                "job_payload_sha256": run_job.canonical_job_sha256(self.job),
            }
            if mode == "approved"
            else {"policy": "validate_scoped_smoke"}
        )
        return {
            "schema": run_job.EXECUTION_AUTHORIZATION_SCHEMA,
            "mode": mode,
            "execution_permitted": True,
            "calibration_eligible": mode == "approved",
            "job_payload_sha256": run_job.canonical_job_sha256(self.job),
            "evidence": evidence,
            "evidence_sha256": sha256_json(evidence),
        }

    def build_inputs(self, *, authorization_mode: str = "approved") -> dict:
        snapshots = self.attempt / "input_snapshots"
        snapshots.mkdir(exist_ok=True)
        paths = {
            "declared_hardware": snapshots / "declared_hardware.json",
            "declared_model_manifest": snapshots / "declared_model_manifest.json",
            "dataset_manifest": snapshots / "dataset_manifest.json",
            "provenance": snapshots / "provenance.json",
            "deepspeed_config": snapshots / "deepspeed_config.json",
        }
        provenance = {"schema": "test-provenance"}
        for name, value in {
            "declared_hardware": {"gpu_id": "local_h800_140g"},
            "declared_model_manifest": {"models": []},
            "dataset_manifest": {"datasets": {}},
            "provenance": provenance,
            "deepspeed_config": {"zero_optimization": {"stage": 3}},
        }.items():
            write_json(paths[name], value)
        runtime_config = self.attempt / "runtime_config.yaml"
        runtime_config.write_text(
            f"deepspeed: {paths['deepspeed_config']}\n", encoding="utf-8"
        )
        runtime_metadata = self.attempt / "job_metadata.json"
        write_json(
            runtime_metadata,
            {**self.job, "_execution_attempt_id": ATTEMPT_ID},
        )
        runtime_identity_path = self.attempt / "runtime_identity.json"
        write_json(runtime_identity_path, self.runtime_identity)
        topology_path = self.attempt / "nvidia_topology.txt"
        topology_path.write_text("GPU0 GPU1\n", encoding="utf-8")
        self.runtime_hardware["topology_sha256"] = sha256_file(topology_path)
        hardware_path = self.attempt / "runtime_hardware.json"
        write_json(hardware_path, self.runtime_hardware)
        mechanism_path = self.attempt / "runtime_mechanism.json"
        write_json(mechanism_path, self.runtime_mechanism)
        return run_job.execution_inputs_manifest(
            job=self.job,
            execution_attempt_id=ATTEMPT_ID,
            attempt_root=self.attempt,
            runtime_identity=self.runtime_identity,
            runtime_identity_path=runtime_identity_path,
            runtime_config=runtime_config,
            runtime_metadata=runtime_metadata,
            config={"deepspeed": str(paths["deepspeed_config"])},
            command=["python", "train.py"],
            environment={"CUDA_VISIBLE_DEVICES": "1,2"},
            provenance=provenance,
            authorization=self._authorization(authorization_mode),
            runtime_hardware=self.runtime_hardware,
            runtime_hardware_path=hardware_path,
            live_topology_path=topology_path,
            runtime_mechanism=self.runtime_mechanism,
            runtime_mechanism_path=mechanism_path,
            input_snapshots=paths,
        )

    def _write_runtime_manifest(
        self,
        rank: int,
        *,
        logical_elements: int = 6,
        uuid: str | None = None,
    ) -> None:
        model_inventory = inventory(logical_elements)
        attestation = device_attestation(rank, uuid=uuid)
        write_json(
            run_job.runtime_model_manifest_path(self.attempt, ATTEMPT_ID, rank),
            {
                "schema": "sft_runtime_model_manifest",
                "schema_version": 2,
                "job_id": "h800-test",
                "execution_attempt_id": ATTEMPT_ID,
                "rank": rank,
                "local_rank": rank,
                "world_size": 2,
                "training_mode": "full",
                "inventory": model_inventory,
                "inventory_sha256": sha256_json(model_inventory),
                "device_attestation": attestation,
                "device_attestation_sha256": sha256_json(attestation),
            },
        )

    def _terminal_evidence(
        self, classification: str = "success", return_code: int = 0
    ) -> dict:
        summary_files = []
        if classification == "success":
            for rank in range(2):
                path = self.attempt / "metrics" / f"summary.rank{rank}.json"
                if not path.is_file():
                    write_json(path, {"rank": rank})
                summary_files.append(
                    {
                        "rank": rank,
                        "path": f"metrics/summary.rank{rank}.json",
                        "file_sha256": sha256_file(path),
                    }
                )
        return {
            "schema": "sft_terminal_classification/v1",
            "classification": classification,
            "return_code": return_code,
            "cuda_oom_confirmed": classification == "oom",
            "matched_cuda_oom_patterns": (
                [run_job.OOM_PATTERNS[0]] if classification == "oom" else []
            ),
            "summaries_complete": classification == "success",
            "summary_files": summary_files,
            "job_id": "h800-test",
            "execution_attempt_id": ATTEMPT_ID,
        }

    def test_input_manifest_binds_attempt_snapshots_and_authorization(self) -> None:
        manifest = self.build_inputs()

        self.assertEqual(manifest["schema"], "sft_execution_inputs/v2")
        self.assertEqual(manifest["job_snapshot"], self.job)
        self.assertEqual(
            manifest["job_payload_sha256"], run_job.canonical_job_sha256(self.job)
        )
        self.assertEqual(
            set(manifest["components"]), set(run_job.STATIC_EXECUTION_COMPONENTS)
        )
        self.assertEqual(manifest["authorization"]["mode"], "approved")
        self.assertTrue(
            all(
                (self.attempt / relative).is_file()
                for relative in manifest["evidence_paths"].values()
            )
        )

    def test_input_manifest_rejects_non_attempt_evidence(self) -> None:
        inputs = self.build_inputs()
        outside = self.root / "outside.json"
        write_json(outside, self.runtime_identity)
        with self.assertRaisesRegex(RuntimeError, "escapes attempt root"):
            run_job.execution_inputs_manifest(
                job=self.job,
                execution_attempt_id=ATTEMPT_ID,
                attempt_root=self.attempt,
                runtime_identity=self.runtime_identity,
                runtime_identity_path=outside,
                runtime_config=self.attempt / "runtime_config.yaml",
                runtime_metadata=self.attempt / "job_metadata.json",
                config={
                    "deepspeed": str(
                        self.attempt / "input_snapshots/deepspeed_config.json"
                    )
                },
                command=["python"],
                environment={},
                provenance={"schema": "test-provenance"},
                authorization=inputs["authorization"],
                runtime_hardware=self.runtime_hardware,
                runtime_hardware_path=self.attempt / "runtime_hardware.json",
                live_topology_path=self.attempt / "nvidia_topology.txt",
                runtime_mechanism=self.runtime_mechanism,
                runtime_mechanism_path=self.attempt / "runtime_mechanism.json",
                input_snapshots={
                    name: self.attempt / relative
                    for name, relative in inputs["evidence_paths"].items()
                    if name
                    in {
                        "declared_hardware",
                        "declared_model_manifest",
                        "dataset_manifest",
                        "provenance",
                        "deepspeed_config",
                    }
                },
            )

    def test_final_manifest_validates_ranks_and_cross_attests_hardware(self) -> None:
        inputs = self.build_inputs()
        inputs_path = self.attempt / "execution_inputs.json"
        write_json(inputs_path, inputs)
        self._write_runtime_manifest(0)
        self._write_runtime_manifest(1)

        manifest = run_job.finalize_execution_fingerprint(
            job=self.job,
            execution_attempt_id=ATTEMPT_ID,
            execution_inputs=inputs,
            execution_inputs_path=inputs_path,
            result_dir=self.attempt,
            expected_ranks=2,
            terminal_classification=self._terminal_evidence(),
            return_code=0,
        )

        self.assertEqual(manifest["schema"], "sft_execution_fingerprint/v2")
        self.assertTrue(manifest["calibration_eligible"])
        self.assertEqual(
            set(manifest["components"]), set(run_job.REQUIRED_EXECUTION_COMPONENTS)
        )
        self.assertTrue(
            all(
                all(row["device_cross_attestation"].values())
                for row in manifest["runtime_model_manifests"]
            )
        )
        self.assertEqual(manifest["outcome"], self._terminal_evidence())

    def test_final_manifest_rejects_rank_inventory_drift(self) -> None:
        inputs = self.build_inputs()
        inputs_path = self.attempt / "execution_inputs.json"
        write_json(inputs_path, inputs)
        self._write_runtime_manifest(0, logical_elements=6)
        self._write_runtime_manifest(1, logical_elements=7)

        with self.assertRaisesRegex(RuntimeError, "differ across ranks"):
            run_job.finalize_execution_fingerprint(
                job=self.job,
                execution_attempt_id=ATTEMPT_ID,
                execution_inputs=inputs,
                execution_inputs_path=inputs_path,
                result_dir=self.attempt,
                expected_ranks=2,
                terminal_classification=self._terminal_evidence(),
                return_code=0,
            )

    def test_final_manifest_rejects_rank_device_substitution(self) -> None:
        inputs = self.build_inputs()
        inputs_path = self.attempt / "execution_inputs.json"
        write_json(inputs_path, inputs)
        self._write_runtime_manifest(0)
        self._write_runtime_manifest(1, uuid="GPU-deadbeef")

        with self.assertRaisesRegex(RuntimeError, "does not match pre-execution"):
            run_job.finalize_execution_fingerprint(
                job=self.job,
                execution_attempt_id=ATTEMPT_ID,
                execution_inputs=inputs,
                execution_inputs_path=inputs_path,
                result_dir=self.attempt,
                expected_ranks=2,
                terminal_classification=self._terminal_evidence(),
                return_code=0,
            )

    def test_smoke_execution_is_never_calibration_eligible(self) -> None:
        inputs = self.build_inputs(authorization_mode="smoke")
        inputs_path = self.attempt / "execution_inputs.json"
        write_json(inputs_path, inputs)
        self._write_runtime_manifest(0)
        self._write_runtime_manifest(1)

        manifest = run_job.finalize_execution_fingerprint(
            job=self.job,
            execution_attempt_id=ATTEMPT_ID,
            execution_inputs=inputs,
            execution_inputs_path=inputs_path,
            result_dir=self.attempt,
            expected_ranks=2,
            terminal_classification=self._terminal_evidence(),
            return_code=0,
        )

        self.assertFalse(manifest["calibration_eligible"])


class RuntimeHardwareAndClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.declared = {
            "gpu_id": "local_h800_140g",
            "name_reported_by_driver": "NVIDIA H800",
            "memory_bytes_reported_by_torch": MEMORY_BYTES,
        }
        self.experiment = {
            "training_scope": {"gpu_type": "NVIDIA H800 140GB HBM3"}
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _command(self, command: list[str]) -> str:
        if command[-2:] == ["topo", "-m"]:
            return (
                "        GPU0 GPU1 GPU2 GPU3 CPU Affinity\n"
                "GPU0    X    NV8  NV8  NV8  0-31\n"
                "GPU1    NV8  X    NV8  NV8  0-31\n"
                "GPU2    NV8  NV8  X    NV8  0-31\n"
                "GPU3    NV8  NV8  NV8  X    0-31\n"
            )
        memory_mib = MEMORY_BYTES // (1024 * 1024)
        return "\n".join(
            f"{index}, GPU-{index:032x}, NVIDIA H800, {memory_mib}, "
            f"00000000:{index + 1:02x}:00.0, 570.00, 9.0, 700.0, Disabled"
            for index in range(4)
        )

    def test_live_hardware_manifest_binds_exact_allocated_h800_set(self) -> None:
        manifest, topology = run_job.capture_runtime_hardware_manifest(
            job={"job_id": "h800-test", "gpu_count": 2},
            execution_attempt_id=ATTEMPT_ID,
            gpu_ids=[1, 2],
            experiment=self.experiment,
            declared_hardware=self.declared,
            command_text=self._command,
            captured_unix=1.0,
        )

        self.assertTrue(manifest["all_passed"])
        self.assertEqual(
            [row["physical_index"] for row in manifest["devices"]], [1, 2]
        )
        self.assertEqual(manifest["assigned_topology"]["matrix"]["GPU1"]["GPU2"], "NV8")
        self.assertIn("GPU0", topology)

    def test_live_hardware_manifest_rejects_wrong_sku(self) -> None:
        def wrong_sku(command: list[str]) -> str:
            return self._command(command).replace("NVIDIA H800", "NVIDIA RTX 4090")

        with self.assertRaisesRegex(RuntimeError, "attestation failed"):
            run_job.capture_runtime_hardware_manifest(
                job={"job_id": "h800-test", "gpu_count": 2},
                execution_attempt_id=ATTEMPT_ID,
                gpu_ids=[1, 2],
                experiment=self.experiment,
                declared_hardware=self.declared,
                command_text=wrong_sku,
            )

    def _write_summary(
        self, attempt: Path, rank: int, *, attempt_id: str = ATTEMPT_ID
    ) -> None:
        write_json(
            attempt / "metrics" / f"summary.rank{rank}.json",
            {
                "job_id": "h800-test",
                "execution_attempt_id": attempt_id,
                "rank": rank,
                "world_size": 2,
                "measured_steps": 1,
                "measured_seconds": 1.0,
                "measured_totals": {"effective_tokens": 1},
            },
        )

    def test_terminal_rejects_zero_step_success(self) -> None:
        log = self.root / "attempt" / "train.log"
        log.parent.mkdir()
        log.write_text("trainer returned zero without an optimizer step\n", encoding="utf-8")
        self._write_summary(log.parent, 0)
        self._write_summary(log.parent, 1)
        for rank in (0, 1):
            path = log.parent / "metrics" / f"summary.rank{rank}.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["measured_steps"] = 0
            summary["measured_seconds"] = 0.0
            summary["measured_totals"] = {"effective_tokens": 0}
            write_json(path, summary)

        result = run_job.classify_execution(
            0,
            log,
            2,
            job_id="h800-test",
            execution_attempt_id=ATTEMPT_ID,
        )

        self.assertEqual(result["classification"], "incomplete_metrics")
        self.assertFalse(result["evidence"]["summaries_complete"])

    def test_terminal_success_requires_bound_complete_summaries(self) -> None:
        log = self.root / "attempt" / "train.log"
        log.parent.mkdir()
        log.write_text("finished\n", encoding="utf-8")
        self._write_summary(log.parent, 0)
        self._write_summary(log.parent, 1)

        result = run_job.classify_execution(
            0,
            log,
            2,
            job_id="h800-test",
            execution_attempt_id=ATTEMPT_ID,
        )

        evidence = result["evidence"]
        self.assertEqual(result["classification"], "success")
        self.assertEqual(evidence["schema"], "sft_terminal_classification/v1")
        self.assertTrue(evidence["summaries_complete"])
        self.assertEqual(len(evidence["summary_files"]), 2)
        self.assertTrue(all(row["file_sha256"] for row in evidence["summary_files"]))

    def test_terminal_rejects_stale_summary_binding(self) -> None:
        log = self.root / "attempt" / "train.log"
        log.parent.mkdir()
        log.write_text("finished\n", encoding="utf-8")
        self._write_summary(log.parent, 0)
        self._write_summary(log.parent, 1, attempt_id="b" * 20)

        result = run_job.classify_execution(
            0,
            log,
            2,
            job_id="h800-test",
            execution_attempt_id=ATTEMPT_ID,
        )

        self.assertEqual(result["classification"], "incomplete_metrics")
        self.assertFalse(result["evidence"]["summaries_complete"])

    def test_host_oomkill_is_not_cuda_oom(self) -> None:
        log = self.root / "train.log"
        log.write_text("systemd: OOMKilled process; host out of memory\n", encoding="utf-8")

        result = run_job.classify_execution(
            137,
            log,
            1,
            job_id="h800-test",
            execution_attempt_id=ATTEMPT_ID,
        )

        self.assertEqual(result["classification"], "failed")
        self.assertFalse(result["evidence"]["cuda_oom_confirmed"])

    def test_cuda_allocator_oom_is_structurally_confirmed(self) -> None:
        log = self.root / "train.log"
        log.write_text(
            "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate\n",
            encoding="utf-8",
        )

        result = run_job.classify_execution(
            1,
            log,
            1,
            job_id="h800-test",
            execution_attempt_id=ATTEMPT_ID,
        )

        self.assertEqual(result["classification"], "oom")
        self.assertTrue(result["evidence"]["cuda_oom_confirmed"])


class RuntimeMechanismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for relative in run_job.RUNTIME_MECHANISM_SOURCE_FILES:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"mechanism:{relative}\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_mechanism_fingerprint_excludes_scenario_and_gpu_assignment(self) -> None:
        identity = {
            "python_executable": "/venv/bin/python",
            "packages": {"torch": "2.8", "flash-attn": "3.0"},
        }
        hardware = {"devices": [{"driver_version": "570.00"}]}
        first = run_job.runtime_mechanism_manifest(
            experiment={
                "fixed_runtime": {"flash_attn": "fa3"},
                "training_scope": {"model_ids": ["small"]},
            },
            runtime_identity=identity,
            runtime_hardware=hardware,
            environment={"CUDA_VISIBLE_DEVICES": "1,2", "NCCL_DEBUG": "WARN"},
            project_root=self.root,
        )
        second = run_job.runtime_mechanism_manifest(
            experiment={
                "fixed_runtime": {"flash_attn": "fa3"},
                "training_scope": {"model_ids": ["large"]},
            },
            runtime_identity=identity,
            runtime_hardware=hardware,
            environment={"CUDA_VISIBLE_DEVICES": "3,4", "NCCL_DEBUG": "WARN"},
            project_root=self.root,
        )

        self.assertEqual(first["fingerprint_sha256"], second["fingerprint_sha256"])
        self.assertNotIn("CUDA_VISIBLE_DEVICES", first["mechanism_environment"])

    def test_mechanism_rejects_unknown_influential_environment(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-allowlisted"):
            run_job.runtime_mechanism_manifest(
                experiment={"fixed_runtime": {}},
                runtime_identity={},
                runtime_hardware={"devices": []},
                environment={"TORCH_LOGS": "+dynamo"},
                project_root=self.root,
            )


class AttemptCompatibilityTests(unittest.TestCase):
    def test_latest_symlink_archives_legacy_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest = root / "metrics"
            latest.mkdir()
            (latest / "old.json").write_text("{}\n", encoding="utf-8")
            target = root / "attempts" / ATTEMPT_ID / "metrics"
            target.mkdir(parents=True)

            run_job._replace_latest_symlink(latest, target, ATTEMPT_ID)

            self.assertTrue(latest.is_symlink())
            self.assertEqual(latest.resolve(), target.resolve())
            self.assertTrue(
                (root / "legacy_flat" / f"metrics.before-{ATTEMPT_ID}" / "old.json").is_file()
            )

    def test_latest_render_and_status_are_symlinks_not_duplicate_regular_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job_root = Path(directory) / "job"
            attempt = job_root / "attempts" / ATTEMPT_ID
            attempt.mkdir(parents=True)
            rendered_target = attempt / "rendered_run.json"
            status_target = attempt / "status.json"
            write_json(rendered_target, {"execution_attempt_id": ATTEMPT_ID})
            write_json(status_target, {"execution_attempt_id": ATTEMPT_ID})
            write_json(job_root / "rendered_run.json", {"legacy": True})
            write_json(job_root / "status.json", {"legacy": True})

            run_job._replace_latest_symlink(
                job_root / "rendered_run.json", rendered_target, ATTEMPT_ID
            )
            run_job._replace_latest_symlink(
                job_root / "status.json", status_target, ATTEMPT_ID
            )

            self.assertTrue((job_root / "rendered_run.json").is_symlink())
            self.assertTrue((job_root / "status.json").is_symlink())
            self.assertEqual(
                (job_root / "rendered_run.json").resolve(), rendered_target.resolve()
            )
            self.assertEqual(
                (job_root / "status.json").resolve(), status_target.resolve()
            )
            self.assertTrue(
                (
                    job_root
                    / "legacy_flat"
                    / f"rendered_run.json.before-{ATTEMPT_ID}"
                ).is_file()
            )
            self.assertTrue(
                (
                    job_root
                    / "legacy_flat"
                    / f"status.json.before-{ATTEMPT_ID}"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
