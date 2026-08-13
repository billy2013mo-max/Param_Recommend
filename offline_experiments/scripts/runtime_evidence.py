#!/usr/bin/env python3
"""Strict, dependency-free validators for per-rank runtime evidence."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


RUNTIME_MODEL_MANIFEST_SCHEMA = "sft_runtime_model_manifest"
RUNTIME_MODEL_MANIFEST_SCHEMA_VERSION = 2
RUNTIME_DEVICE_ATTESTATION_SCHEMA = "sft_runtime_device_attestation"
RUNTIME_DEVICE_ATTESTATION_SCHEMA_VERSION = 1

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_EXECUTION_ATTEMPT_ID = re.compile(r"[0-9a-f]{20}\Z")
_TRAINING_MODES = {"full", "lora"}

_INVENTORY_FIELDS = {
    "model_class",
    "unique_tensor_count",
    "logical_parameter_elements",
    "trainable_parameter_elements",
    "frozen_parameter_elements",
    "tensors",
    "module_groups",
    "largest_module_group",
}
_TENSOR_FIELDS = {
    "tensor_id",
    "canonical_name",
    "aliases",
    "logical_numel",
    "logical_shape",
    "dtype",
    "requires_grad",
}
_MODULE_GROUP_FIELDS = {
    "module_name",
    "module_class",
    "tensor_ids",
    "logical_parameter_elements",
    "trainable_parameter_elements",
}
_LARGEST_MODULE_GROUP_FIELDS = {
    "module_name",
    "logical_parameter_elements",
}
_DEVICE_ATTESTATION_FIELDS = {
    "schema",
    "schema_version",
    "availability",
    "source",
    "local_rank",
    "visible_device_index",
    "name",
    "total_memory_bytes",
    "compute_capability",
    "uuid",
    "uuid_unavailable_reason",
    "unavailable_reason",
}
_MANIFEST_FIELDS = {
    "schema",
    "schema_version",
    "job_id",
    "execution_attempt_id",
    "rank",
    "local_rank",
    "world_size",
    "training_mode",
    "inventory",
    "inventory_sha256",
    "device_attestation",
    "device_attestation_sha256",
}


class RuntimeEvidenceError(ValueError):
    """Raised when runtime evidence is incomplete, ambiguous, or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode strict canonical JSON for content-addressed evidence."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeEvidenceError(f"Evidence is not strict JSON: {error}") from error


def sha256_json(value: Any) -> str:
    """Return the SHA256 of strict canonical JSON."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeEvidenceError(f"{path} must be an object")
    return value


def _require_exact_fields(value: dict[str, Any], fields: set[str], path: str) -> None:
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise RuntimeEvidenceError(
            f"{path} fields are not exact: missing={missing}, unknown={unknown}"
        )


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeEvidenceError(f"{path} must be a non-empty string")
    return value


def _require_int(value: Any, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeEvidenceError(f"{path} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeEvidenceError(f"{path} must be a lowercase SHA256")
    return value


def validate_job_id(value: Any) -> str:
    """Validate the path-safe canonical job identifier used by evidence files."""

    if not isinstance(value, str) or _SAFE_JOB_ID.fullmatch(value) is None:
        raise RuntimeEvidenceError(
            "job_id must be a path-safe token of at most 256 characters"
        )
    return value


def validate_execution_attempt_id(value: Any) -> str:
    """Validate the launcher-generated, collision-resistant attempt identifier."""

    if not isinstance(value, str) or _EXECUTION_ATTEMPT_ID.fullmatch(value) is None:
        raise RuntimeEvidenceError(
            "execution_attempt_id must contain exactly 20 lowercase hex characters"
        )
    return value


def validate_training_mode(value: Any) -> str:
    """Return the canonical training mode used by calibration selectors."""

    if not isinstance(value, str):
        raise RuntimeEvidenceError("training_mode must be a string")
    normalized = value.strip().lower()
    if normalized not in _TRAINING_MODES:
        raise RuntimeEvidenceError(
            f"training_mode must be one of {sorted(_TRAINING_MODES)}, got {value!r}"
        )
    return normalized


def validate_rank_binding(rank: Any, local_rank: Any, world_size: Any) -> tuple[int, int, int]:
    """Validate a single-node rank tuple without coercing bools/floats/strings."""

    checked_world_size = _require_int(world_size, "world_size", minimum=1)
    checked_rank = _require_int(rank, "rank")
    checked_local_rank = _require_int(local_rank, "local_rank")
    if checked_rank >= checked_world_size:
        raise RuntimeEvidenceError("rank must be smaller than world_size")
    if checked_local_rank >= checked_world_size:
        raise RuntimeEvidenceError("local_rank must be smaller than world_size")
    return checked_rank, checked_local_rank, checked_world_size


def validate_runtime_model_inventory(
    inventory: Any,
    *,
    training_mode: str,
) -> dict[str, Any]:
    """Validate the exact logical tensor/module inventory and all derived totals."""

    checked_mode = validate_training_mode(training_mode)
    value = _require_object(inventory, "inventory")
    _require_exact_fields(value, _INVENTORY_FIELDS, "inventory")
    _require_string(value["model_class"], "inventory.model_class")

    tensors = value["tensors"]
    if not isinstance(tensors, list) or not tensors:
        raise RuntimeEvidenceError("inventory.tensors must be a non-empty list")
    unique_tensor_count = _require_int(
        value["unique_tensor_count"], "inventory.unique_tensor_count", minimum=1
    )
    if unique_tensor_count != len(tensors):
        raise RuntimeEvidenceError(
            "inventory.unique_tensor_count does not equal len(inventory.tensors)"
        )

    tensor_by_id: dict[str, dict[str, Any]] = {}
    globally_seen_aliases: set[str] = set()
    logical_total = 0
    trainable_total = 0
    expected_module_tensors: dict[str, set[str]] = {}
    canonical_names: list[str] = []
    for index, raw_tensor in enumerate(tensors):
        path = f"inventory.tensors[{index}]"
        tensor = _require_object(raw_tensor, path)
        _require_exact_fields(tensor, _TENSOR_FIELDS, path)
        tensor_id = _require_string(tensor["tensor_id"], f"{path}.tensor_id")
        if tensor_id in tensor_by_id:
            raise RuntimeEvidenceError(f"Duplicate tensor_id {tensor_id!r}")

        aliases = tensor["aliases"]
        if not isinstance(aliases, list) or not aliases:
            raise RuntimeEvidenceError(f"{path}.aliases must be a non-empty list")
        if not all(isinstance(alias, str) and alias for alias in aliases):
            raise RuntimeEvidenceError(f"{path}.aliases must contain non-empty strings")
        if aliases != sorted(aliases) or len(set(aliases)) != len(aliases):
            raise RuntimeEvidenceError(f"{path}.aliases must be sorted and unique")
        overlapping_aliases = globally_seen_aliases.intersection(aliases)
        if overlapping_aliases:
            raise RuntimeEvidenceError(
                f"Aliases are shared by different tensors: {sorted(overlapping_aliases)}"
            )
        globally_seen_aliases.update(aliases)

        canonical_name = _require_string(
            tensor["canonical_name"], f"{path}.canonical_name"
        )
        if canonical_name != aliases[0]:
            raise RuntimeEvidenceError(
                f"{path}.canonical_name must equal the first sorted alias"
            )
        canonical_names.append(canonical_name)

        logical_shape = tensor["logical_shape"]
        if not isinstance(logical_shape, list):
            raise RuntimeEvidenceError(f"{path}.logical_shape must be a list")
        shape_numel = 1
        for dimension_index, dimension in enumerate(logical_shape):
            checked_dimension = _require_int(
                dimension,
                f"{path}.logical_shape[{dimension_index}]",
            )
            shape_numel *= checked_dimension
        logical_numel = _require_int(tensor["logical_numel"], f"{path}.logical_numel")
        if shape_numel != logical_numel:
            raise RuntimeEvidenceError(
                f"{path} shape product {shape_numel} != logical_numel {logical_numel}"
            )
        _require_string(tensor["dtype"], f"{path}.dtype")
        if type(tensor["requires_grad"]) is not bool:
            raise RuntimeEvidenceError(f"{path}.requires_grad must be a bool")

        logical_total += logical_numel
        if tensor["requires_grad"]:
            trainable_total += logical_numel
        tensor_by_id[tensor_id] = tensor
        for alias in aliases:
            owner = alias.rpartition(".")[0] or "<root>"
            expected_module_tensors.setdefault(owner, set()).add(tensor_id)

    if canonical_names != sorted(canonical_names):
        raise RuntimeEvidenceError(
            "inventory.tensors must be ordered by canonical parameter name"
        )
    if logical_total <= 0:
        raise RuntimeEvidenceError("inventory must contain at least one parameter element")

    declared_logical = _require_int(
        value["logical_parameter_elements"],
        "inventory.logical_parameter_elements",
        minimum=1,
    )
    declared_trainable = _require_int(
        value["trainable_parameter_elements"],
        "inventory.trainable_parameter_elements",
    )
    declared_frozen = _require_int(
        value["frozen_parameter_elements"],
        "inventory.frozen_parameter_elements",
    )
    if declared_logical != logical_total:
        raise RuntimeEvidenceError("inventory.logical_parameter_elements is inconsistent")
    if declared_trainable != trainable_total:
        raise RuntimeEvidenceError("inventory.trainable_parameter_elements is inconsistent")
    if declared_frozen != logical_total - trainable_total:
        raise RuntimeEvidenceError("inventory.frozen_parameter_elements is inconsistent")
    if checked_mode == "lora" and trainable_total <= 0:
        raise RuntimeEvidenceError("LoRA runtime inventory must contain trainable tensors")

    module_groups = value["module_groups"]
    if not isinstance(module_groups, list) or not module_groups:
        raise RuntimeEvidenceError("inventory.module_groups must be a non-empty list")
    module_by_name: dict[str, dict[str, Any]] = {}
    for index, raw_group in enumerate(module_groups):
        path = f"inventory.module_groups[{index}]"
        group = _require_object(raw_group, path)
        _require_exact_fields(group, _MODULE_GROUP_FIELDS, path)
        module_name = _require_string(group["module_name"], f"{path}.module_name")
        if module_name in module_by_name:
            raise RuntimeEvidenceError(f"Duplicate module group {module_name!r}")
        _require_string(group["module_class"], f"{path}.module_class")
        tensor_ids = group["tensor_ids"]
        if not isinstance(tensor_ids, list) or not tensor_ids:
            raise RuntimeEvidenceError(f"{path}.tensor_ids must be a non-empty list")
        if (
            not all(isinstance(tensor_id, str) and tensor_id for tensor_id in tensor_ids)
            or tensor_ids != sorted(tensor_ids)
            or len(set(tensor_ids)) != len(tensor_ids)
        ):
            raise RuntimeEvidenceError(
                f"{path}.tensor_ids must contain sorted unique strings"
            )
        missing_tensor_ids = sorted(set(tensor_ids) - set(tensor_by_id))
        if missing_tensor_ids:
            raise RuntimeEvidenceError(
                f"{path} references unknown tensor IDs {missing_tensor_ids}"
            )
        if set(tensor_ids) != expected_module_tensors.get(module_name, set()):
            raise RuntimeEvidenceError(
                f"{path}.tensor_ids do not match parameter alias ownership"
            )
        group_logical = sum(tensor_by_id[item]["logical_numel"] for item in tensor_ids)
        group_trainable = sum(
            tensor_by_id[item]["logical_numel"]
            for item in tensor_ids
            if tensor_by_id[item]["requires_grad"]
        )
        if (
            _require_int(
                group["logical_parameter_elements"],
                f"{path}.logical_parameter_elements",
            )
            != group_logical
        ):
            raise RuntimeEvidenceError(
                f"{path}.logical_parameter_elements is inconsistent"
            )
        if (
            _require_int(
                group["trainable_parameter_elements"],
                f"{path}.trainable_parameter_elements",
            )
            != group_trainable
        ):
            raise RuntimeEvidenceError(
                f"{path}.trainable_parameter_elements is inconsistent"
            )
        module_by_name[module_name] = group

    if list(module_by_name) != sorted(module_by_name):
        raise RuntimeEvidenceError("inventory.module_groups must be sorted by module_name")
    if set(module_by_name) != set(expected_module_tensors):
        missing = sorted(set(expected_module_tensors) - set(module_by_name))
        unknown = sorted(set(module_by_name) - set(expected_module_tensors))
        raise RuntimeEvidenceError(
            f"Module groups do not exactly cover alias owners: missing={missing}, "
            f"unknown={unknown}"
        )

    largest = _require_object(
        value["largest_module_group"], "inventory.largest_module_group"
    )
    _require_exact_fields(
        largest,
        _LARGEST_MODULE_GROUP_FIELDS,
        "inventory.largest_module_group",
    )
    expected_largest = min(
        module_groups,
        key=lambda group: (
            -group["logical_parameter_elements"],
            group["module_name"],
        ),
    )
    if largest != {
        "module_name": expected_largest["module_name"],
        "logical_parameter_elements": expected_largest[
            "logical_parameter_elements"
        ],
    }:
        raise RuntimeEvidenceError("inventory.largest_module_group is inconsistent")

    canonical_json_bytes(value)
    return value


def validate_runtime_device_attestation(
    attestation: Any,
    *,
    expected_local_rank: int | None = None,
    allow_unavailable: bool = True,
) -> dict[str, Any]:
    """Validate one rank's torch-level CUDA identity without probing hardware."""

    value = _require_object(attestation, "device_attestation")
    _require_exact_fields(value, _DEVICE_ATTESTATION_FIELDS, "device_attestation")
    if value["schema"] != RUNTIME_DEVICE_ATTESTATION_SCHEMA:
        raise RuntimeEvidenceError("device_attestation.schema is unsupported")
    if value["schema_version"] != RUNTIME_DEVICE_ATTESTATION_SCHEMA_VERSION:
        raise RuntimeEvidenceError("device_attestation.schema_version is unsupported")
    source = _require_string(value["source"], "device_attestation.source")
    local_rank = _require_int(value["local_rank"], "device_attestation.local_rank")
    if expected_local_rank is not None and local_rank != expected_local_rank:
        raise RuntimeEvidenceError("device_attestation.local_rank binding mismatch")

    availability = value["availability"]
    if availability == "available":
        _require_int(
            value["visible_device_index"],
            "device_attestation.visible_device_index",
        )
        _require_string(value["name"], "device_attestation.name")
        _require_int(
            value["total_memory_bytes"],
            "device_attestation.total_memory_bytes",
            minimum=1,
        )
        capability = _require_object(
            value["compute_capability"], "device_attestation.compute_capability"
        )
        _require_exact_fields(
            capability,
            {"major", "minor"},
            "device_attestation.compute_capability",
        )
        _require_int(capability["major"], "device_attestation.compute_capability.major")
        _require_int(capability["minor"], "device_attestation.compute_capability.minor")
        uuid = value["uuid"]
        uuid_reason = value["uuid_unavailable_reason"]
        if uuid is None:
            _require_string(
                uuid_reason,
                "device_attestation.uuid_unavailable_reason",
            )
        else:
            _require_string(uuid, "device_attestation.uuid")
            if uuid_reason is not None:
                raise RuntimeEvidenceError(
                    "device_attestation.uuid_unavailable_reason must be null when UUID is present"
                )
        if value["unavailable_reason"] is not None:
            raise RuntimeEvidenceError(
                "device_attestation.unavailable_reason must be null when available"
            )
    elif availability == "unavailable":
        if not allow_unavailable:
            raise RuntimeEvidenceError(
                f"CUDA device attestation from {source!r} is unavailable"
            )
        for name in (
            "visible_device_index",
            "name",
            "total_memory_bytes",
            "compute_capability",
            "uuid",
            "uuid_unavailable_reason",
        ):
            if value[name] is not None:
                raise RuntimeEvidenceError(
                    f"device_attestation.{name} must be null when unavailable"
                )
        _require_string(
            value["unavailable_reason"],
            "device_attestation.unavailable_reason",
        )
    else:
        raise RuntimeEvidenceError(
            "device_attestation.availability must be 'available' or 'unavailable'"
        )

    canonical_json_bytes(value)
    return value


def validate_runtime_model_manifest(
    manifest: Any,
    *,
    expected_job_id: str | None = None,
    expected_execution_attempt_id: str | None = None,
    expected_rank: int | None = None,
    expected_world_size: int | None = None,
    expected_training_mode: str | None = None,
    allow_unavailable_device: bool = True,
) -> dict[str, Any]:
    """Validate one complete v2 rank manifest and optional external bindings."""

    value = _require_object(manifest, "runtime_model_manifest")
    _require_exact_fields(value, _MANIFEST_FIELDS, "runtime_model_manifest")
    if value["schema"] != RUNTIME_MODEL_MANIFEST_SCHEMA:
        raise RuntimeEvidenceError("runtime_model_manifest.schema is unsupported")
    if value["schema_version"] != RUNTIME_MODEL_MANIFEST_SCHEMA_VERSION:
        raise RuntimeEvidenceError("runtime_model_manifest.schema_version is unsupported")

    job_id = validate_job_id(value["job_id"])
    attempt_id = validate_execution_attempt_id(value["execution_attempt_id"])
    rank, local_rank, world_size = validate_rank_binding(
        value["rank"], value["local_rank"], value["world_size"]
    )
    training_mode = validate_training_mode(value["training_mode"])
    if expected_job_id is not None and job_id != validate_job_id(expected_job_id):
        raise RuntimeEvidenceError("runtime_model_manifest.job_id binding mismatch")
    if (
        expected_execution_attempt_id is not None
        and attempt_id
        != validate_execution_attempt_id(expected_execution_attempt_id)
    ):
        raise RuntimeEvidenceError(
            "runtime_model_manifest.execution_attempt_id binding mismatch"
        )
    if expected_rank is not None and rank != expected_rank:
        raise RuntimeEvidenceError("runtime_model_manifest.rank binding mismatch")
    if expected_world_size is not None and world_size != expected_world_size:
        raise RuntimeEvidenceError("runtime_model_manifest.world_size binding mismatch")
    if expected_training_mode is not None and training_mode != validate_training_mode(
        expected_training_mode
    ):
        raise RuntimeEvidenceError(
            "runtime_model_manifest.training_mode binding mismatch"
        )

    inventory = validate_runtime_model_inventory(
        value["inventory"], training_mode=training_mode
    )
    inventory_sha256 = _require_sha256(
        value["inventory_sha256"], "runtime_model_manifest.inventory_sha256"
    )
    if inventory_sha256 != sha256_json(inventory):
        raise RuntimeEvidenceError("runtime_model_manifest inventory hash mismatch")

    device_attestation = validate_runtime_device_attestation(
        value["device_attestation"],
        expected_local_rank=local_rank,
        allow_unavailable=allow_unavailable_device,
    )
    device_sha256 = _require_sha256(
        value["device_attestation_sha256"],
        "runtime_model_manifest.device_attestation_sha256",
    )
    if device_sha256 != sha256_json(device_attestation):
        raise RuntimeEvidenceError(
            "runtime_model_manifest device attestation hash mismatch"
        )

    canonical_json_bytes(value)
    return value
