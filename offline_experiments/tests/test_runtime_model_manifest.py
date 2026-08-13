from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from metrics_callback import (  # noqa: E402
    ExperimentCallback,
    build_runtime_model_inventory,
    capture_runtime_device_attestation,
    write_runtime_model_manifest,
)
from runtime_evidence import validate_runtime_model_manifest  # noqa: E402


class TiedWeightModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(5, 3)
        self.lm_head = torch.nn.Linear(3, 5, bias=False)
        self.lm_head.weight = self.embed.weight


class LoraStyleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(4, 3, bias=True)
        self.base.requires_grad_(False)
        self.lora_a = torch.nn.Parameter(torch.zeros(2, 4))
        self.lora_b = torch.nn.Parameter(torch.zeros(3, 2))


class PartitionedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.partitioned = torch.nn.Parameter(torch.zeros(2))
        self.partitioned.ds_shape = torch.Size((3, 4))
        self.partitioned.ds_numel = 12


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RuntimeModelInventoryTests(unittest.TestCase):
    def test_tied_weight_aliases_are_counted_once_globally(self) -> None:
        inventory = build_runtime_model_inventory(TiedWeightModel())

        self.assertEqual(inventory["unique_tensor_count"], 1)
        self.assertEqual(inventory["logical_parameter_elements"], 15)
        self.assertEqual(inventory["trainable_parameter_elements"], 15)
        self.assertEqual(
            inventory["tensors"][0]["aliases"],
            ["embed.weight", "lm_head.weight"],
        )
        groups = {row["module_name"]: row for row in inventory["module_groups"]}
        self.assertEqual(groups["embed"]["logical_parameter_elements"], 15)
        self.assertEqual(groups["lm_head"]["logical_parameter_elements"], 15)
        self.assertEqual(groups["embed"]["tensor_ids"], groups["lm_head"]["tensor_ids"])
        self.assertEqual(
            inventory["largest_module_group"],
            {"module_name": "embed", "logical_parameter_elements": 15},
        )

    def test_lora_style_freezing_has_exact_trainable_total(self) -> None:
        inventory = build_runtime_model_inventory(LoraStyleModel())

        self.assertEqual(inventory["unique_tensor_count"], 4)
        self.assertEqual(inventory["logical_parameter_elements"], 29)
        self.assertEqual(inventory["trainable_parameter_elements"], 14)
        self.assertEqual(inventory["frozen_parameter_elements"], 15)
        trainable_names = {
            row["canonical_name"]
            for row in inventory["tensors"]
            if row["requires_grad"]
        }
        self.assertEqual(trainable_names, {"lora_a", "lora_b"})

    def test_deepspeed_logical_shape_overrides_local_partition(self) -> None:
        inventory = build_runtime_model_inventory(PartitionedModel())

        tensor = inventory["tensors"][0]
        self.assertEqual(tensor["logical_shape"], [3, 4])
        self.assertEqual(tensor["logical_numel"], 12)
        self.assertEqual(inventory["logical_parameter_elements"], 12)

    def test_manifest_inventory_hash_is_rank_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                rank_zero_device = capture_runtime_device_attestation(0)
                rank_one_device = capture_runtime_device_attestation(1)
            rank_zero = write_runtime_model_manifest(
                TiedWeightModel(),
                root / "rank0.json",
                rank=0,
                local_rank=0,
                world_size=2,
                job_id="job-1",
                execution_attempt_id="a" * 20,
                training_mode="full",
                device_attestation=rank_zero_device,
            )
            rank_one = write_runtime_model_manifest(
                TiedWeightModel(),
                root / "rank1.json",
                rank=1,
                local_rank=1,
                world_size=2,
                job_id="job-1",
                execution_attempt_id="a" * 20,
                training_mode="full",
                device_attestation=rank_one_device,
            )

            self.assertEqual(rank_zero["inventory"], rank_one["inventory"])
            self.assertEqual(rank_zero["inventory_sha256"], rank_one["inventory_sha256"])
            self.assertEqual(
                rank_zero["inventory_sha256"],
                sha256_json(rank_zero["inventory"]),
            )
            self.assertNotEqual(rank_zero["rank"], rank_one["rank"])
            self.assertEqual(rank_zero["schema_version"], 2)
            self.assertEqual(
                validate_runtime_model_manifest(
                    rank_zero,
                    expected_job_id="job-1",
                    expected_execution_attempt_id="a" * 20,
                    expected_rank=0,
                    expected_world_size=2,
                    expected_training_mode="full",
                ),
                rank_zero,
            )
            self.assertFalse(list(root.glob("*.tmp")))
            self.assertEqual(
                json.loads((root / "rank0.json").read_text(encoding="utf-8")),
                rank_zero,
            )

    def test_callback_binds_attempt_id_and_writes_rank_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metrics_dir = Path(temporary)
            environment = {"RANK": "2", "LOCAL_RANK": "0", "WORLD_SIZE": "4"}
            with mock.patch.dict(os.environ, environment, clear=False):
                callback = ExperimentCallback(
                    metrics_dir,
                    warmup_steps=0,
                    job_metadata={
                        "job_id": "job-7",
                        "_execution_attempt_id": "b" * 20,
                        "train_type": "lora",
                    },
                )
            with (
                mock.patch.object(callback, "_sync"),
                mock.patch.object(callback, "_memory", return_value={}),
                mock.patch.object(torch.cuda, "is_available", return_value=False),
            ):
                callback.on_train_begin(None, None, None, model=TiedWeightModel())

            expected = metrics_dir / (
                f"runtime_model_manifest.{'b' * 20}.rank2.json"
            )
            manifest = json.loads(expected.read_text(encoding="utf-8"))
            self.assertEqual(manifest["job_id"], "job-7")
            self.assertEqual(manifest["execution_attempt_id"], "b" * 20)
            self.assertEqual(manifest["rank"], 2)
            self.assertEqual(manifest["world_size"], 4)
            self.assertEqual(manifest["training_mode"], "lora")
            self.assertEqual(
                manifest["device_attestation"]["availability"], "unavailable"
            )
            structure_path = metrics_dir / (
                f"model_structure_manifest.{'b' * 20}.rank2.json"
            )
            structure = json.loads(structure_path.read_text(encoding="utf-8"))
            self.assertEqual(structure["schema"], "sft_model_structure_manifest/v1")
            self.assertEqual(structure["declaration_status"], "observed_only")
            self.assertFalse(structure["visual_path_observed"])
            event = json.loads(callback.events_path.read_text(encoding="utf-8"))
            self.assertEqual(event["job_id"], "job-7")
            self.assertEqual(event["execution_attempt_id"], "b" * 20)
            self.assertEqual(event["rank"], 2)
            self.assertEqual(event["local_rank"], 0)
            self.assertEqual(event["world_size"], 4)
            self.assertEqual(event["runtime_model_manifest"]["path"], expected.name)
            self.assertEqual(
                event["runtime_model_manifest"]["inventory_sha256"],
                manifest["inventory_sha256"],
            )
            self.assertEqual(
                event["runtime_structure_manifest"]["path"], structure_path.name
            )
            callback.finalize_after_failure()
            summary = json.loads(callback.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["job_id"], "job-7")
            self.assertEqual(summary["execution_attempt_id"], "b" * 20)
            self.assertEqual(summary["rank"], 2)
            self.assertEqual(summary["local_rank"], 0)
            self.assertEqual(summary["world_size"], 4)

    def test_callback_rejects_unsafe_attempt_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "20 lowercase hex"):
                ExperimentCallback(
                    Path(temporary),
                    warmup_steps=0,
                    job_metadata={
                        "job_id": "job-unsafe",
                        "_execution_attempt_id": "../escape",
                        "train_type": "full",
                    },
                )

    def test_callback_requires_job_attempt_and_training_mode(self) -> None:
        cases = (
            ({"_execution_attempt_id": "c" * 20, "train_type": "full"}, "job_id"),
            ({"job_id": "job-1", "train_type": "full"}, "execution_attempt_id"),
            (
                {"job_id": "job-1", "_execution_attempt_id": "c" * 20},
                "training_mode",
            ),
        )
        for metadata, message in cases:
            with self.subTest(metadata=metadata):
                with tempfile.TemporaryDirectory() as temporary:
                    with self.assertRaisesRegex(ValueError, message):
                        ExperimentCallback(
                            Path(temporary),
                            warmup_steps=0,
                            job_metadata=metadata,
                        )

    def test_cuda_attestation_records_properties_without_running_a_workload(self) -> None:
        properties = mock.Mock(
            name="NVIDIA H800",
            total_memory=150_142_189_568,
            major=9,
            minor=0,
            uuid="GPU-test-uuid",
        )
        properties.name = "NVIDIA H800"
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "current_device", return_value=1),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=properties,
            ),
        ):
            attestation = capture_runtime_device_attestation(1)

        self.assertEqual(attestation["availability"], "available")
        self.assertEqual(attestation["visible_device_index"], 1)
        self.assertEqual(attestation["name"], "NVIDIA H800")
        self.assertEqual(attestation["total_memory_bytes"], 150_142_189_568)
        self.assertEqual(attestation["compute_capability"], {"major": 9, "minor": 0})
        self.assertEqual(attestation["uuid"], "GPU-test-uuid")


if __name__ == "__main__":
    unittest.main()
