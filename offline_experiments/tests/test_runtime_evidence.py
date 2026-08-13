from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from runtime_evidence import (  # noqa: E402
    RUNTIME_DEVICE_ATTESTATION_SCHEMA,
    RUNTIME_DEVICE_ATTESTATION_SCHEMA_VERSION,
    RUNTIME_MODEL_MANIFEST_SCHEMA,
    RUNTIME_MODEL_MANIFEST_SCHEMA_VERSION,
    RuntimeEvidenceError,
    sha256_json,
    validate_runtime_device_attestation,
    validate_runtime_model_inventory,
    validate_runtime_model_manifest,
)


def inventory() -> dict:
    return {
        "model_class": "tests.Model",
        "unique_tensor_count": 2,
        "logical_parameter_elements": 8,
        "trainable_parameter_elements": 2,
        "frozen_parameter_elements": 6,
        "tensors": [
            {
                "tensor_id": "tensor-000000",
                "canonical_name": "block.bias",
                "aliases": ["block.bias"],
                "logical_numel": 2,
                "logical_shape": [2],
                "dtype": "torch.bfloat16",
                "requires_grad": True,
            },
            {
                "tensor_id": "tensor-000001",
                "canonical_name": "block.weight",
                "aliases": ["block.weight"],
                "logical_numel": 6,
                "logical_shape": [2, 3],
                "dtype": "torch.bfloat16",
                "requires_grad": False,
            },
        ],
        "module_groups": [
            {
                "module_name": "block",
                "module_class": "torch.nn.Linear",
                "tensor_ids": ["tensor-000000", "tensor-000001"],
                "logical_parameter_elements": 8,
                "trainable_parameter_elements": 2,
            }
        ],
        "largest_module_group": {
            "module_name": "block",
            "logical_parameter_elements": 8,
        },
    }


def unavailable_device(local_rank: int = 0) -> dict:
    return {
        "schema": RUNTIME_DEVICE_ATTESTATION_SCHEMA,
        "schema_version": RUNTIME_DEVICE_ATTESTATION_SCHEMA_VERSION,
        "availability": "unavailable",
        "source": "torch.cuda.get_device_properties",
        "local_rank": local_rank,
        "visible_device_index": None,
        "name": None,
        "total_memory_bytes": None,
        "compute_capability": None,
        "uuid": None,
        "uuid_unavailable_reason": None,
        "unavailable_reason": "torch_cuda_is_not_available",
    }


def manifest() -> dict:
    model_inventory = inventory()
    device = unavailable_device()
    return {
        "schema": RUNTIME_MODEL_MANIFEST_SCHEMA,
        "schema_version": RUNTIME_MODEL_MANIFEST_SCHEMA_VERSION,
        "job_id": "job-1",
        "execution_attempt_id": "a" * 20,
        "rank": 0,
        "local_rank": 0,
        "world_size": 2,
        "training_mode": "lora",
        "inventory": model_inventory,
        "inventory_sha256": sha256_json(model_inventory),
        "device_attestation": device,
        "device_attestation_sha256": sha256_json(device),
    }


class RuntimeInventoryValidationTests(unittest.TestCase):
    def test_valid_inventory_and_manifest_are_returned_unchanged(self) -> None:
        model_inventory = inventory()
        rank_manifest = manifest()

        self.assertIs(
            validate_runtime_model_inventory(
                model_inventory,
                training_mode="lora",
            ),
            model_inventory,
        )
        self.assertIs(
            validate_runtime_model_manifest(
                rank_manifest,
                expected_job_id="job-1",
                expected_execution_attempt_id="a" * 20,
                expected_rank=0,
                expected_world_size=2,
                expected_training_mode="lora",
            ),
            rank_manifest,
        )

    def test_empty_or_duplicate_tensor_evidence_is_rejected(self) -> None:
        empty = inventory()
        empty["tensors"] = []
        empty["unique_tensor_count"] = 0
        with self.assertRaisesRegex(RuntimeEvidenceError, "non-empty"):
            validate_runtime_model_inventory(empty, training_mode="full")

        duplicate_alias = inventory()
        duplicate_alias["tensors"][1]["canonical_name"] = "block.bias"
        duplicate_alias["tensors"][1]["aliases"] = ["block.bias"]
        with self.assertRaisesRegex(RuntimeEvidenceError, "shared"):
            validate_runtime_model_inventory(duplicate_alias, training_mode="full")

        duplicate_id = inventory()
        duplicate_id["tensors"][1]["tensor_id"] = "tensor-000000"
        with self.assertRaisesRegex(RuntimeEvidenceError, "Duplicate tensor_id"):
            validate_runtime_model_inventory(duplicate_id, training_mode="full")

    def test_tensor_shape_dtype_and_requires_grad_are_strict(self) -> None:
        wrong_shape = inventory()
        wrong_shape["tensors"][1]["logical_shape"] = [3, 3]
        with self.assertRaisesRegex(RuntimeEvidenceError, "shape product"):
            validate_runtime_model_inventory(wrong_shape, training_mode="full")

        empty_dtype = inventory()
        empty_dtype["tensors"][0]["dtype"] = ""
        with self.assertRaisesRegex(RuntimeEvidenceError, "dtype"):
            validate_runtime_model_inventory(empty_dtype, training_mode="full")

        integer_grad_flag = inventory()
        integer_grad_flag["tensors"][0]["requires_grad"] = 1
        with self.assertRaisesRegex(RuntimeEvidenceError, "must be a bool"):
            validate_runtime_model_inventory(integer_grad_flag, training_mode="full")

    def test_global_totals_and_lora_trainability_are_enforced(self) -> None:
        wrong_total = inventory()
        wrong_total["logical_parameter_elements"] = 9
        with self.assertRaisesRegex(RuntimeEvidenceError, "logical_parameter_elements"):
            validate_runtime_model_inventory(wrong_total, training_mode="full")

        no_trainable = inventory()
        no_trainable["tensors"][0]["requires_grad"] = False
        no_trainable["trainable_parameter_elements"] = 0
        no_trainable["frozen_parameter_elements"] = 8
        no_trainable["module_groups"][0]["trainable_parameter_elements"] = 0
        with self.assertRaisesRegex(RuntimeEvidenceError, "LoRA.*trainable"):
            validate_runtime_model_inventory(no_trainable, training_mode="lora")
        self.assertIs(
            validate_runtime_model_inventory(no_trainable, training_mode="full"),
            no_trainable,
        )

    def test_module_ownership_totals_and_largest_group_are_enforced(self) -> None:
        wrong_owner = inventory()
        wrong_owner["module_groups"][0]["module_name"] = "other"
        with self.assertRaisesRegex(RuntimeEvidenceError, "alias ownership"):
            validate_runtime_model_inventory(wrong_owner, training_mode="full")

        wrong_group_total = inventory()
        wrong_group_total["module_groups"][0]["logical_parameter_elements"] = 7
        with self.assertRaisesRegex(RuntimeEvidenceError, "is inconsistent"):
            validate_runtime_model_inventory(wrong_group_total, training_mode="full")

        wrong_largest = inventory()
        wrong_largest["largest_module_group"]["logical_parameter_elements"] = 7
        with self.assertRaisesRegex(RuntimeEvidenceError, "largest_module_group"):
            validate_runtime_model_inventory(wrong_largest, training_mode="full")

    def test_unknown_fields_are_rejected_instead_of_silently_ignored(self) -> None:
        model_inventory = inventory()
        model_inventory["unbound_hint"] = "not evidence"
        with self.assertRaisesRegex(RuntimeEvidenceError, "unknown"):
            validate_runtime_model_inventory(model_inventory, training_mode="full")


class RuntimeManifestValidationTests(unittest.TestCase):
    def test_manifest_rejects_schema_hash_and_external_binding_drift(self) -> None:
        wrong_version = manifest()
        wrong_version["schema_version"] = 1
        with self.assertRaisesRegex(RuntimeEvidenceError, "schema_version"):
            validate_runtime_model_manifest(wrong_version)

        wrong_hash = manifest()
        wrong_hash["inventory_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeEvidenceError, "inventory hash"):
            validate_runtime_model_manifest(wrong_hash)

        with self.assertRaisesRegex(RuntimeEvidenceError, "attempt_id binding"):
            validate_runtime_model_manifest(
                manifest(),
                expected_execution_attempt_id="b" * 20,
            )
        with self.assertRaisesRegex(RuntimeEvidenceError, "world_size binding"):
            validate_runtime_model_manifest(manifest(), expected_world_size=4)

    def test_device_attestation_can_be_structural_or_gpu_required(self) -> None:
        device = unavailable_device()
        self.assertIs(validate_runtime_device_attestation(device), device)
        with self.assertRaisesRegex(RuntimeEvidenceError, "is unavailable"):
            validate_runtime_device_attestation(device, allow_unavailable=False)
        with self.assertRaisesRegex(RuntimeEvidenceError, "is unavailable"):
            validate_runtime_model_manifest(
                manifest(),
                allow_unavailable_device=False,
            )

    def test_device_attestation_hash_and_local_rank_are_bound(self) -> None:
        wrong_device_hash = manifest()
        wrong_device_hash["device_attestation_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeEvidenceError, "device attestation hash"):
            validate_runtime_model_manifest(wrong_device_hash)

        wrong_local_rank = manifest()
        wrong_local_rank["device_attestation"] = unavailable_device(local_rank=1)
        wrong_local_rank["device_attestation_sha256"] = sha256_json(
            wrong_local_rank["device_attestation"]
        )
        with self.assertRaisesRegex(RuntimeEvidenceError, "local_rank binding"):
            validate_runtime_model_manifest(wrong_local_rank)

    def test_mutating_inventory_requires_recomputing_hash_but_still_fails_invariants(
        self,
    ) -> None:
        rank_manifest = manifest()
        mutated_inventory = copy.deepcopy(rank_manifest["inventory"])
        mutated_inventory["largest_module_group"]["logical_parameter_elements"] = 7
        rank_manifest["inventory"] = mutated_inventory
        rank_manifest["inventory_sha256"] = sha256_json(mutated_inventory)

        with self.assertRaisesRegex(RuntimeEvidenceError, "largest_module_group"):
            validate_runtime_model_manifest(rank_manifest)


if __name__ == "__main__":
    unittest.main()
