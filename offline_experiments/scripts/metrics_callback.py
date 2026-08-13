#!/usr/bin/env python3
"""Per-rank timing, token and CUDA-memory instrumentation for LLaMA-Factory."""

from __future__ import annotations

import json
import os
import statistics
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

import torch
from runtime_evidence import (
    RUNTIME_DEVICE_ATTESTATION_SCHEMA,
    RUNTIME_DEVICE_ATTESTATION_SCHEMA_VERSION,
    RUNTIME_MODEL_MANIFEST_SCHEMA,
    RUNTIME_MODEL_MANIFEST_SCHEMA_VERSION,
    sha256_json,
    validate_execution_attempt_id,
    validate_job_id,
    validate_rank_binding,
    validate_runtime_device_attestation,
    validate_runtime_model_inventory,
    validate_runtime_model_manifest,
    validate_training_mode,
)
from transformers import TrainerCallback


def _atomic_write_json(path: Path, value: Any) -> None:
    """Durably replace one strict-JSON artifact without exposing partial data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(
                value,
                output,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _qualified_class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _unavailable_device_attestation(
    local_rank: int,
    *,
    source: str,
    reason: str,
) -> dict[str, Any]:
    attestation = {
        "schema": RUNTIME_DEVICE_ATTESTATION_SCHEMA,
        "schema_version": RUNTIME_DEVICE_ATTESTATION_SCHEMA_VERSION,
        "availability": "unavailable",
        "source": source,
        "local_rank": local_rank,
        "visible_device_index": None,
        "name": None,
        "total_memory_bytes": None,
        "compute_capability": None,
        "uuid": None,
        "uuid_unavailable_reason": None,
        "unavailable_reason": reason,
    }
    return validate_runtime_device_attestation(
        attestation,
        expected_local_rank=local_rank,
    )


def _device_uuid(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            decoded = value.decode("ascii").strip()
        except UnicodeDecodeError:
            decoded = ""
        normalized = decoded or value.hex()
    else:
        normalized = str(value).strip()
    if not normalized:
        return None
    # torch's _CUuuid stringifies to the bare hex form (``047b4c52-...``) while
    # nvidia-smi reports the canonical ``GPU-047b4c52-...`` form.  Normalize to
    # the ``GPU-`` prefixed canonical form so the runtime attestation matches the
    # pre-execution hardware evidence for the same physical device.
    if not normalized.startswith("GPU-") and not normalized.startswith("MIG-"):
        normalized = f"GPU-{normalized}"
    return normalized


def capture_runtime_device_attestation(local_rank: int) -> dict[str, Any]:
    """Capture the current rank's torch-visible CUDA identity without workloads."""

    if type(local_rank) is not int or local_rank < 0:
        raise ValueError("local_rank must be a non-negative integer")
    source = "torch.cuda.get_device_properties"
    if not torch.cuda.is_available():
        return _unavailable_device_attestation(
            local_rank,
            source=source,
            reason="torch_cuda_is_not_available",
        )
    try:
        visible_device_index = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(visible_device_index)
        uuid = _device_uuid(getattr(properties, "uuid", None))
        attestation = {
            "schema": RUNTIME_DEVICE_ATTESTATION_SCHEMA,
            "schema_version": RUNTIME_DEVICE_ATTESTATION_SCHEMA_VERSION,
            "availability": "available",
            "source": source,
            "local_rank": local_rank,
            "visible_device_index": visible_device_index,
            "name": str(properties.name),
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": {
                "major": int(properties.major),
                "minor": int(properties.minor),
            },
            "uuid": uuid,
            "uuid_unavailable_reason": (
                None if uuid is not None else "torch_device_properties_has_no_uuid"
            ),
            "unavailable_reason": None,
        }
        return validate_runtime_device_attestation(
            attestation,
            expected_local_rank=local_rank,
        )
    except (AssertionError, RuntimeError, TypeError, ValueError) as error:
        return _unavailable_device_attestation(
            local_rank,
            source=source,
            reason=f"torch_cuda_probe_failed:{type(error).__name__}",
        )


def _logical_shape(parameter: torch.nn.Parameter) -> list[int]:
    candidate = getattr(parameter, "ds_shape", None)
    shape = parameter.shape if candidate is None else candidate
    dimensions = [int(dimension) for dimension in shape]
    if any(dimension < 0 for dimension in dimensions):
        raise ValueError(f"Negative logical parameter shape is invalid: {dimensions}")
    return dimensions


def _logical_numel(parameter: torch.nn.Parameter, shape: list[int]) -> int:
    candidate = getattr(parameter, "ds_numel", None)
    logical_numel = int(parameter.numel() if candidate is None else candidate)
    if logical_numel < 0:
        raise ValueError(f"Negative logical parameter numel is invalid: {logical_numel}")
    shape_numel = 1
    for dimension in shape:
        shape_numel *= dimension
    if shape_numel != logical_numel:
        raise ValueError(
            "Logical parameter shape and numel disagree: "
            f"shape={shape}, shape_numel={shape_numel}, numel={logical_numel}"
        )
    return logical_numel


def _named_parameters_with_aliases(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    try:
        parameters = model.named_parameters(recurse=True, remove_duplicate=False)
        return list(parameters)
    except TypeError:
        # Compatibility for older torch releases whose public iterator does not
        # expose remove_duplicate. Direct registration is required here because
        # the default iterator intentionally hides tied-weight aliases.
        named: list[tuple[str, torch.nn.Parameter]] = []
        for module_name, module in model.named_modules():
            for local_name, parameter in module._parameters.items():
                if parameter is None:
                    continue
                name = f"{module_name}.{local_name}" if module_name else local_name
                named.append((name, parameter))
        return named


def build_runtime_model_inventory(model: torch.nn.Module) -> dict[str, Any]:
    """Build a deterministic logical inventory from the initialized model.

    Parameter object identity is the allocation identity. This preserves every
    registered alias while counting tied weights exactly once globally. A
    DeepSpeed-partitioned parameter uses ``ds_shape``/``ds_numel`` rather than
    the process-local shard shape.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module, got {type(model).__name__}")

    entries_by_identity: dict[int, dict[str, Any]] = {}
    for name, parameter in _named_parameters_with_aliases(model):
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(f"Named parameter {name!r} is not torch.nn.Parameter")
        if not name:
            raise ValueError("Runtime parameter names must be non-empty")
        identity = id(parameter)
        entry = entries_by_identity.get(identity)
        if entry is None:
            shape = _logical_shape(parameter)
            entry = {
                "aliases": [],
                "logical_numel": _logical_numel(parameter, shape),
                "logical_shape": shape,
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            }
            entries_by_identity[identity] = entry
        elif (
            entry["logical_shape"] != _logical_shape(parameter)
            or entry["logical_numel"]
            != _logical_numel(parameter, entry["logical_shape"])
            or entry["dtype"] != str(parameter.dtype)
            or entry["requires_grad"] != bool(parameter.requires_grad)
        ):
            raise ValueError(f"Inconsistent metadata across aliases of {name!r}")
        if name not in entry["aliases"]:
            entry["aliases"].append(name)

    ordered_entries = sorted(
        entries_by_identity.values(),
        key=lambda entry: tuple(sorted(entry["aliases"])),
    )
    tensors: list[dict[str, Any]] = []
    module_tensor_ids: dict[str, set[str]] = {}
    for index, entry in enumerate(ordered_entries):
        aliases = sorted(entry["aliases"])
        tensor_id = f"tensor-{index:06d}"
        tensor = {
            "tensor_id": tensor_id,
            "canonical_name": aliases[0],
            "aliases": aliases,
            "logical_numel": entry["logical_numel"],
            "logical_shape": entry["logical_shape"],
            "dtype": entry["dtype"],
            "requires_grad": entry["requires_grad"],
        }
        tensors.append(tensor)
        for alias in aliases:
            owner = alias.rpartition(".")[0] or "<root>"
            module_tensor_ids.setdefault(owner, set()).add(tensor_id)

    tensor_by_id = {tensor["tensor_id"]: tensor for tensor in tensors}
    module_classes = {
        name or "<root>": _qualified_class_name(module)
        for name, module in model.named_modules()
    }
    module_groups: list[dict[str, Any]] = []
    for module_name, tensor_ids in sorted(module_tensor_ids.items()):
        ordered_tensor_ids = sorted(tensor_ids)
        group_tensors = [tensor_by_id[tensor_id] for tensor_id in ordered_tensor_ids]
        module_groups.append(
            {
                "module_name": module_name,
                "module_class": module_classes.get(module_name),
                "tensor_ids": ordered_tensor_ids,
                "logical_parameter_elements": sum(
                    tensor["logical_numel"] for tensor in group_tensors
                ),
                "trainable_parameter_elements": sum(
                    tensor["logical_numel"]
                    for tensor in group_tensors
                    if tensor["requires_grad"]
                ),
            }
        )

    largest_module_group = None
    if module_groups:
        largest = min(
            module_groups,
            key=lambda group: (
                -group["logical_parameter_elements"],
                group["module_name"],
            ),
        )
        largest_module_group = {
            "module_name": largest["module_name"],
            "logical_parameter_elements": largest["logical_parameter_elements"],
        }

    logical_elements = sum(tensor["logical_numel"] for tensor in tensors)
    trainable_elements = sum(
        tensor["logical_numel"] for tensor in tensors if tensor["requires_grad"]
    )
    inventory = {
        "model_class": _qualified_class_name(model),
        "unique_tensor_count": len(tensors),
        "logical_parameter_elements": logical_elements,
        "trainable_parameter_elements": trainable_elements,
        "frozen_parameter_elements": logical_elements - trainable_elements,
        "tensors": tensors,
        "module_groups": module_groups,
        "largest_module_group": largest_module_group,
    }
    training_mode = "lora" if 0 < trainable_elements < logical_elements else "full"
    return validate_runtime_model_inventory(
        inventory,
        training_mode=training_mode,
    )


def write_runtime_model_manifest(
    model: torch.nn.Module,
    path: Path,
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    job_id: str,
    execution_attempt_id: str,
    training_mode: str,
    device_attestation: dict[str, Any],
) -> dict[str, Any]:
    inventory = build_runtime_model_inventory(model)
    checked_training_mode = validate_training_mode(training_mode)
    validate_runtime_model_inventory(
        inventory,
        training_mode=checked_training_mode,
    )
    manifest = {
        "schema": RUNTIME_MODEL_MANIFEST_SCHEMA,
        "schema_version": RUNTIME_MODEL_MANIFEST_SCHEMA_VERSION,
        "job_id": job_id,
        "execution_attempt_id": execution_attempt_id,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "training_mode": checked_training_mode,
        "inventory": inventory,
        "inventory_sha256": sha256_json(inventory),
        "device_attestation": device_attestation,
        "device_attestation_sha256": sha256_json(device_attestation),
    }
    validate_runtime_model_manifest(
        manifest,
        expected_job_id=job_id,
        expected_execution_attempt_id=execution_attempt_id,
        expected_rank=rank,
        expected_world_size=world_size,
        expected_training_mode=checked_training_mode,
    )
    _atomic_write_json(path, manifest)
    return manifest


TOKEN_COUNTER_KEYS = (
    "computed_tokens",
    "effective_tokens",
    "label_tokens",
    "logical_samples",
    "physical_batches",
    "computed_attention_token_pairs",
    "effective_attention_token_pairs",
)


def _sum_token_records(records: list[dict[str, int]]) -> dict[str, int]:
    return {
        key: sum(int(record[key]) for record in records)
        for key in TOKEN_COUNTER_KEYS
    }


def slice_consumed_token_ledger(
    records: list[dict[str, int]],
    *,
    gradient_accumulation_steps: int,
    completed_warmup_steps: int,
    completed_measured_steps: int,
) -> dict[str, Any]:
    """Separate consumed micro-batches from dataloader prefetch.

    A collator invocation proves that a batch was materialized, not that the
    trainer consumed it.  A completed optimizer step consumes exactly
    ``gradient_accumulation_steps`` batches per rank.  Therefore the prefix
    implied by completed optimizer steps is authoritative and any trailing
    collator records are prefetch-only evidence.
    """

    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if completed_warmup_steps < 0 or completed_measured_steps < 0:
        raise ValueError("completed step counts must be non-negative")
    normalized: list[dict[str, int]] = []
    for index, record in enumerate(records):
        normalized_record: dict[str, int] = {}
        for key in TOKEN_COUNTER_KEYS:
            value = record.get(key)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"token ledger record {index} has invalid {key}: {value!r}"
                )
            normalized_record[key] = value
        normalized.append(normalized_record)

    warmup_batch_count = completed_warmup_steps * gradient_accumulation_steps
    measured_batch_count = completed_measured_steps * gradient_accumulation_steps
    consumed_batch_count = warmup_batch_count + measured_batch_count
    ledger_complete = len(normalized) >= consumed_batch_count
    consumed = normalized[:consumed_batch_count]
    warmup = consumed[:warmup_batch_count]
    measured = consumed[warmup_batch_count:consumed_batch_count]
    prefetched = normalized[consumed_batch_count:]
    return {
        "schema": "consumed_token_ledger/v1",
        "authoritative": ledger_complete,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "completed_warmup_steps": completed_warmup_steps,
        "completed_measured_steps": completed_measured_steps,
        "collated_batch_count": len(normalized),
        "consumed_batch_count_required": consumed_batch_count,
        "consumed_batch_count_observed": len(consumed),
        "warmup_batch_count": len(warmup),
        "measured_batch_count": len(measured),
        "prefetched_not_consumed_batch_count": len(prefetched),
        "collated_totals": _sum_token_records(normalized),
        "consumed_totals": _sum_token_records(consumed),
        "warmup_totals": _sum_token_records(warmup),
        "measured_totals": _sum_token_records(measured),
        "prefetched_not_consumed_totals": _sum_token_records(prefetched),
    }


def slice_consumed_batch_shape_ledger(
    records: list[dict[str, Any]],
    *,
    gradient_accumulation_steps: int,
    completed_warmup_steps: int,
    completed_measured_steps: int,
) -> dict[str, Any]:
    """Bind per-collator batch shapes to the optimizer steps actually consumed.

    The collator can prefetch batches that the trainer never consumes.  As with
    the token ledger, only the prefix implied by completed optimizer steps is
    authoritative.  This ledger intentionally stores lengths, not token IDs or
    user content.
    """

    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if completed_warmup_steps < 0 or completed_measured_steps < 0:
        raise ValueError("completed step counts must be non-negative")
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        lengths = record.get("logical_sequence_lengths")
        if (
            not isinstance(lengths, list)
            or not lengths
            or not all(type(value) is int and value > 0 for value in lengths)
        ):
            raise ValueError(
                f"batch-shape ledger record {index} has invalid logical lengths"
            )
        physical_batch_size = record.get("physical_batch_size")
        padded_sequence_length = record.get("padded_sequence_length")
        padding_tokens = record.get("padding_tokens")
        if (
            type(physical_batch_size) is not int
            or physical_batch_size <= 0
            or type(padded_sequence_length) is not int
            or padded_sequence_length <= 0
            or type(padding_tokens) is not int
            or padding_tokens < 0
            or type(record.get("packing")) is not bool
        ):
            raise ValueError(f"batch-shape ledger record {index} is malformed")
        normalized.append(
            {
                "physical_batch_size": physical_batch_size,
                "padded_sequence_length": padded_sequence_length,
                "logical_sequence_lengths": [int(value) for value in lengths],
                "logical_sample_count": len(lengths),
                "maximum_logical_sequence_length": max(lengths),
                "minimum_logical_sequence_length": min(lengths),
                "sum_logical_sequence_lengths": sum(lengths),
                "padding_tokens": padding_tokens,
                "packing": bool(record["packing"]),
            }
        )

    warmup_batch_count = completed_warmup_steps * gradient_accumulation_steps
    measured_batch_count = completed_measured_steps * gradient_accumulation_steps
    consumed_batch_count = warmup_batch_count + measured_batch_count
    consumed = normalized[:consumed_batch_count]
    measured = consumed[warmup_batch_count:consumed_batch_count]
    steps = []
    for step_index in range(completed_measured_steps):
        left = step_index * gradient_accumulation_steps
        right = left + gradient_accumulation_steps
        microbatches = measured[left:right]
        logical_lengths = [
            length
            for microbatch in microbatches
            for length in microbatch["logical_sequence_lengths"]
        ]
        steps.append(
            {
                "measured_step_index": step_index,
                "microbatch_count": len(microbatches),
                "padded_sequence_length_max": max(
                    (row["padded_sequence_length"] for row in microbatches),
                    default=None,
                ),
                "logical_sequence_length_max": max(logical_lengths, default=None),
                "logical_sequence_length_min": min(logical_lengths, default=None),
                "logical_sequence_length_mean": (
                    sum(logical_lengths) / len(logical_lengths)
                    if logical_lengths
                    else None
                ),
                "logical_sample_count": len(logical_lengths),
                "padding_tokens": sum(row["padding_tokens"] for row in microbatches),
            }
        )
    return {
        "schema": "consumed_batch_shape_ledger/v1",
        "content_policy": "lengths_only_no_token_ids_or_user_content",
        "authoritative": len(normalized) >= consumed_batch_count,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "collated_batch_count": len(normalized),
        "consumed_batch_count_required": consumed_batch_count,
        "consumed_batch_count_observed": len(consumed),
        "prefetched_not_consumed_batch_count": max(
            0, len(normalized) - consumed_batch_count
        ),
        "measured_microbatches": measured,
        "measured_steps": steps,
    }


class CollatorCounters:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.computed_tokens = 0
        self.effective_tokens = 0
        self.label_tokens = 0
        self.logical_samples = 0
        self.physical_batches = 0
        self.computed_attention_token_pairs = 0
        self.effective_attention_token_pairs = 0
        self._ledger: list[dict[str, int]] = []
        self._batch_shape_ledger: list[dict[str, Any]] = []

    def add(
        self,
        computed: int,
        effective: int,
        labels: int,
        logical_samples: int,
        computed_attention_pairs: int,
        effective_attention_pairs: int,
        batch_shape: dict[str, Any],
    ) -> None:
        with self._lock:
            self.computed_tokens += computed
            self.effective_tokens += effective
            self.label_tokens += labels
            self.logical_samples += logical_samples
            self.physical_batches += 1
            self.computed_attention_token_pairs += computed_attention_pairs
            self.effective_attention_token_pairs += effective_attention_pairs
            self._ledger.append(
                {
                    "computed_tokens": computed,
                    "effective_tokens": effective,
                    "label_tokens": labels,
                    "logical_samples": logical_samples,
                    "physical_batches": 1,
                    "computed_attention_token_pairs": computed_attention_pairs,
                    "effective_attention_token_pairs": effective_attention_pairs,
                }
            )
            self._batch_shape_ledger.append(dict(batch_shape))

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "computed_tokens": self.computed_tokens,
                "effective_tokens": self.effective_tokens,
                "label_tokens": self.label_tokens,
                "logical_samples": self.logical_samples,
                "physical_batches": self.physical_batches,
                "computed_attention_token_pairs": self.computed_attention_token_pairs,
                "effective_attention_token_pairs": self.effective_attention_token_pairs,
            }

    def ledger_snapshot(self) -> list[dict[str, int]]:
        with self._lock:
            return [dict(record) for record in self._ledger]

    def batch_shape_ledger_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self._batch_shape_ledger]


COUNTERS = CollatorCounters()


class RuntimeBatchEvidence:
    """Bounded, process-local evidence for packing and real multimodal batches."""

    _MEDIA_TENSOR_KEYS = (
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.collator_batches = 0
        self.source_image_count = 0
        self.source_video_count = 0
        self.media_batches = 0
        self.image_grid_rows = 0
        self.video_grid_rows = 0
        self.pixel_value_elements = 0
        self.pixel_video_elements = 0
        self.media_tensor_shapes: dict[str, list[list[int]]] = {
            key: [] for key in self._MEDIA_TENSOR_KEYS
        }
        self.packing_features = 0
        self.packing_logical_samples = 0
        self.packing_multi_sample_features = 0
        self.packing_boundary_violations = 0
        self.packing_segment_mask_violations = 0
        self.packing_position_reset_violations = 0
        self.packing_padding_label_violations = 0
        self.packing_output_position_violations = 0
        self.packing_examples: list[dict[str, Any]] = []

    @staticmethod
    def _as_int_list(value: Any) -> list[int]:
        if torch.is_tensor(value):
            return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
        if isinstance(value, (list, tuple)):
            return [int(item) for item in value]
        return []

    @staticmethod
    def _shape(value: Any) -> list[int] | None:
        if torch.is_tensor(value):
            return [int(dimension) for dimension in value.shape]
        return None

    @staticmethod
    def _grid_rows(value: Any) -> int:
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
        return 0

    def observe(
        self,
        *,
        source_features: list[dict[str, Any]],
        batch: dict[str, Any],
        packing_observations: list[dict[str, Any]],
    ) -> None:
        source_images = sum(len(feature.get("images") or []) for feature in source_features)
        source_videos = sum(len(feature.get("videos") or []) for feature in source_features)
        image_grid_rows = self._grid_rows(batch.get("image_grid_thw"))
        video_grid_rows = self._grid_rows(batch.get("video_grid_thw"))
        pixel_values = batch.get("pixel_values")
        pixel_values_videos = batch.get("pixel_values_videos")
        pixel_elements = int(pixel_values.numel()) if torch.is_tensor(pixel_values) else 0
        pixel_video_elements = (
            int(pixel_values_videos.numel())
            if torch.is_tensor(pixel_values_videos)
            else 0
        )
        media_batch = bool(
            (source_images > 0 and image_grid_rows > 0 and pixel_elements > 0)
            or (source_videos > 0 and video_grid_rows > 0 and pixel_video_elements > 0)
        )

        with self._lock:
            self.collator_batches += 1
            self.source_image_count += source_images
            self.source_video_count += source_videos
            self.media_batches += int(media_batch)
            self.image_grid_rows += image_grid_rows
            self.video_grid_rows += video_grid_rows
            self.pixel_value_elements += pixel_elements
            self.pixel_video_elements += pixel_video_elements
            for key in self._MEDIA_TENSOR_KEYS:
                shape = self._shape(batch.get(key))
                examples = self.media_tensor_shapes[key]
                if shape is not None and shape not in examples and len(examples) < 8:
                    examples.append(shape)

            for observation in packing_observations:
                self.packing_features += 1
                logical_samples = int(observation["logical_samples"])
                self.packing_logical_samples += logical_samples
                self.packing_multi_sample_features += int(logical_samples > 1)
                self.packing_boundary_violations += int(
                    not observation["boundary_valid"]
                )
                self.packing_segment_mask_violations += int(
                    not observation["segment_mask_valid"]
                )
                self.packing_position_reset_violations += int(
                    not observation["position_reset_valid"]
                )
                self.packing_padding_label_violations += int(
                    not observation["padding_labels_valid"]
                )
                self.packing_output_position_violations += int(
                    not observation["output_position_reset_valid"]
                )
                if len(self.packing_examples) < 8:
                    self.packing_examples.append(dict(observation))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            violations = {
                "boundary": self.packing_boundary_violations,
                "segment_attention_mask": self.packing_segment_mask_violations,
                "input_position_reset": self.packing_position_reset_violations,
                "padding_labels": self.packing_padding_label_violations,
                "output_position_reset": self.packing_output_position_violations,
            }
            media = {
                "collator_batches": self.collator_batches,
                "source_image_count": self.source_image_count,
                "source_video_count": self.source_video_count,
                "media_batches": self.media_batches,
                "image_grid_rows": self.image_grid_rows,
                "video_grid_rows": self.video_grid_rows,
                "pixel_value_elements": self.pixel_value_elements,
                "pixel_video_elements": self.pixel_video_elements,
                "tensor_shapes": {
                    key: [list(shape) for shape in shapes]
                    for key, shapes in self.media_tensor_shapes.items()
                },
            }
            media["real_image_path_observed"] = bool(
                media["source_image_count"] > 0
                and media["media_batches"] > 0
                and media["image_grid_rows"] > 0
                and media["pixel_value_elements"] > 0
            )
            media["real_video_path_observed"] = bool(
                media["source_video_count"] > 0
                and media["media_batches"] > 0
                and media["video_grid_rows"] > 0
                and media["pixel_video_elements"] > 0
            )
            packing = {
                "features": self.packing_features,
                "logical_samples": self.packing_logical_samples,
                "multi_sample_features": self.packing_multi_sample_features,
                "violations": violations,
                "semantic_checks_passed": bool(
                    self.packing_features > 0 and all(value == 0 for value in violations.values())
                ),
                "examples": [dict(row) for row in self.packing_examples],
            }
            return {"media": media, "packing": packing}


BATCH_EVIDENCE = RuntimeBatchEvidence()
_PATCHED = False


def _packing_observation(
    feature: dict[str, Any],
    output_position_ids: Any,
) -> dict[str, Any] | None:
    params = feature.get("packing_params") or {}
    boundaries = RuntimeBatchEvidence._as_int_list(params.get("sequence_boundaries"))
    if not boundaries:
        return None
    input_ids = RuntimeBatchEvidence._as_int_list(feature.get("input_ids"))
    attention_mask = RuntimeBatchEvidence._as_int_list(feature.get("attention_mask"))
    position_ids = RuntimeBatchEvidence._as_int_list(feature.get("position_ids"))
    labels = RuntimeBatchEvidence._as_int_list(feature.get("labels"))
    right_padding_length = int(params.get("right_padding_length") or 0)
    boundary_valid = bool(
        len(boundaries) >= 3
        and boundaries[0] == 0
        and all(right > left for left, right in zip(boundaries, boundaries[1:]))
        and boundaries[-1] == len(input_ids)
        and boundaries[-1] - boundaries[-2] == right_padding_length
    )
    real_boundaries = boundaries[:-1] if len(boundaries) >= 2 else boundaries
    logical_samples = max(1, len(real_boundaries) - 1)
    expected_attention: list[int] = []
    expected_positions: list[int] = []
    for index, (left, right) in enumerate(zip(real_boundaries, real_boundaries[1:])):
        length = max(0, right - left)
        expected_attention.extend([index + 1] * length)
        expected_positions.extend(range(length))
    expected_attention.extend([0] * right_padding_length)
    expected_positions.extend([0] * right_padding_length)
    segment_mask_valid = bool(attention_mask == expected_attention)
    position_reset_valid = bool(position_ids == expected_positions)
    padding_start = real_boundaries[-1] if real_boundaries else 0
    padding_labels_valid = bool(
        len(labels) == len(input_ids)
        and all(value == -100 for value in labels[padding_start:])
    )
    output_positions = RuntimeBatchEvidence._as_int_list(output_position_ids)
    expected_unpadded_positions = expected_positions[:padding_start]
    output_position_reset_valid = bool(
        output_positions
        and output_positions[-len(expected_unpadded_positions) :] == expected_unpadded_positions
    )
    return {
        "boundaries": boundaries[:32],
        "right_padding_length": right_padding_length,
        "logical_samples": logical_samples,
        "boundary_valid": boundary_valid,
        "segment_mask_valid": segment_mask_valid,
        "position_reset_valid": position_reset_valid,
        "padding_labels_valid": padding_labels_valid,
        "output_position_reset_valid": output_position_reset_valid,
    }


def install_collator_instrumentation() -> None:
    """Patch only the process-local text collator; training math is unchanged."""
    global _PATCHED
    if _PATCHED:
        return
    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask

    original = SFTDataCollatorWith4DAttentionMask.__call__

    def instrumented(self: Any, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # The upstream collator pops media and packing fields.  Keep shallow
        # snapshots of the immutable lists before calling it.
        source_features = [dict(feature) for feature in features]
        logical_samples = 0
        source_lengths: list[int] = []
        logical_sequence_lengths: list[int] = []
        packed_attention_pairs = 0
        has_packing = False
        for feature in features:
            packing_params = feature.get("packing_params")
            boundaries = (packing_params or {}).get("sequence_boundaries") if packing_params else None
            # Packed processor appends one final right-padding boundary.
            if boundaries:
                has_packing = True
                real_boundaries = boundaries[:-1]
                subsequence_lengths = [right - left for left, right in zip(real_boundaries, real_boundaries[1:])]
                logical_samples += max(1, len(subsequence_lengths))
                logical_sequence_lengths.extend(
                    length for length in subsequence_lengths if length > 0
                )
                packed_attention_pairs += sum(length * length for length in subsequence_lengths)
            else:
                logical_samples += 1
                source_length = len(feature.get("input_ids") or [])
                source_lengths.append(source_length)
                logical_sequence_lengths.append(source_length)
        try:
            batch = original(self, features)
        except StopIteration as error:
            # Transformers treats StopIteration from the epoch iterator as an
            # exhausted dataset.  If it originates inside a multimodal
            # collator, that turns a real processor failure into a misleading
            # successful step-0 exit.  Preserve the causal traceback and fail
            # the benchmark explicitly instead.
            feature_keys = sorted(
                {str(key) for feature in source_features for key in feature.keys()}
            )
            image_count = sum(
                len(feature.get("images") or []) for feature in source_features
            )
            video_count = sum(
                len(feature.get("videos") or []) for feature in source_features
            )
            raise RuntimeError(
                "multimodal collator raised StopIteration "
                f"(features={len(features)}, images={image_count}, "
                f"videos={video_count}, keys={feature_keys})"
            ) from error
        output_position_ids = batch.get("position_ids")
        packing_observations: list[dict[str, Any]] = []
        for source_feature in source_features:
            observation = _packing_observation(source_feature, output_position_ids)
            if observation is not None:
                packing_observations.append(observation)
        BATCH_EVIDENCE.observe(
            source_features=source_features,
            batch=batch,
            packing_observations=packing_observations,
        )
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")
        computed = int(input_ids.numel()) if torch.is_tensor(input_ids) else 0
        if torch.is_tensor(attention_mask):
            effective = int((attention_mask != 0).sum().item())
        else:
            # FA3 neat packing removes right padding before returning the batch.
            effective = computed
        label_tokens = int((labels != -100).sum().item()) if torch.is_tensor(labels) else 0
        if has_packing:
            computed_attention_pairs = packed_attention_pairs
            effective_attention_pairs = packed_attention_pairs
        elif torch.is_tensor(input_ids):
            batch_size, padded_length = input_ids.shape[:2]
            computed_attention_pairs = int(batch_size * padded_length * padded_length)
            effective_attention_pairs = sum(length * length for length in source_lengths)
        else:
            computed_attention_pairs = 0
            effective_attention_pairs = 0
        if not torch.is_tensor(input_ids) or input_ids.ndim < 2:
            raise RuntimeError("collator returned no two-dimensional input_ids tensor")
        physical_batch_size = int(input_ids.shape[0])
        padded_sequence_length = int(input_ids.shape[1])
        if not logical_sequence_lengths:
            # A malformed packed boundary ledger must fail during evidence
            # collection instead of silently producing an unusable fit row.
            raise RuntimeError("collator observed no positive logical sequence lengths")
        COUNTERS.add(
            computed,
            effective,
            label_tokens,
            logical_samples,
            computed_attention_pairs,
            effective_attention_pairs,
            {
                "physical_batch_size": physical_batch_size,
                "padded_sequence_length": padded_sequence_length,
                "logical_sequence_lengths": logical_sequence_lengths,
                "padding_tokens": max(0, computed - effective),
                "packing": has_packing,
            },
        )
        return batch

    SFTDataCollatorWith4DAttentionMask.__call__ = instrumented
    _PATCHED = True


def delta(current: dict[str, int], previous: dict[str, int]) -> dict[str, int]:
    return {key: current[key] - previous.get(key, 0) for key in current}


class VisionPhaseMemoryProbe:
    """Record CUDA high-water marks when the frozen vision root runs.

    The probe is opt-in because forward hooks are evidence instrumentation, not
    part of the training implementation.  The enclosing callback resets CUDA
    peak counters once per optimizer step.  Reading those counters when the
    vision root returns gives the largest allocation observed up to the end of
    the visual phase; the ordinary step record remains the full-step peak.

    This does not add the visual and language peaks.  It exists specifically so
    a decomposition experiment can compare the two phase high-water marks.
    """

    _ROOT_NAMES: ClassVar[set[str]] = {"visual", "vision_tower"}

    def __init__(self, *, module_name: str, module: torch.nn.Module) -> None:
        self.module_name = module_name
        self.module_class = _qualified_class_name(module)
        self._step: int | None = None
        self._calls: list[dict[str, Any]] = []
        self._open_before: list[dict[str, int]] = []
        self._pre_handle = module.register_forward_pre_hook(self._before_forward)
        self._post_handle = module.register_forward_hook(self._after_forward)

    @staticmethod
    def _memory() -> dict[str, int]:
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated": int(torch.cuda.memory_allocated()),
            "reserved": int(torch.cuda.memory_reserved()),
            "max_allocated": int(torch.cuda.max_memory_allocated()),
            "max_reserved": int(torch.cuda.max_memory_reserved()),
        }

    @classmethod
    def install(cls, model: torch.nn.Module) -> VisionPhaseMemoryProbe:
        candidates: list[tuple[str, torch.nn.Module]] = []
        for name, module in model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if (
                leaf in cls._ROOT_NAMES
                and hasattr(module, "patch_embed")
                and hasattr(module, "blocks")
            ):
                candidates.append((name, module))
        # Parameter aliases can expose the same module through more than one
        # name.  Deduplicate by object identity while retaining a stable name.
        unique: dict[int, tuple[str, torch.nn.Module]] = {}
        for name, module in sorted(candidates, key=lambda item: item[0]):
            unique.setdefault(id(module), (name, module))
        if len(unique) != 1:
            names = [name for name, _ in unique.values()]
            raise RuntimeError(
                "Vision phase memory probe requires exactly one visual root; "
                f"found {len(unique)}: {names}"
            )
        name, module = next(iter(unique.values()))
        return cls(module_name=name, module=module)

    def _before_forward(self, module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        if self._step is not None:
            self._open_before.append(self._memory())

    def _after_forward(
        self,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if self._step is None:
            return
        before = self._open_before.pop() if self._open_before else {}
        self._calls.append({"before": before, "after": self._memory()})

    def start_step(self, global_step: int) -> None:
        if type(global_step) is not int or global_step <= 0:
            raise ValueError("global_step must be a positive integer")
        self._step = global_step
        self._calls = []
        self._open_before = []

    def finish_step(self, global_step: int) -> dict[str, Any]:
        if self._step != global_step:
            raise RuntimeError(
                "Vision phase probe step binding drifted: "
                f"started={self._step}, finished={global_step}"
            )
        calls = [dict(call) for call in self._calls]
        result = {
            "module_name": self.module_name,
            "module_class": self.module_class,
            "global_step": global_step,
            "forward_calls": len(calls),
            "max_allocated_during_vision": max(
                (int(call["after"].get("max_allocated", 0)) for call in calls),
                default=0,
            ),
            "max_reserved_during_vision": max(
                (int(call["after"].get("max_reserved", 0)) for call in calls),
                default=0,
            ),
            "calls": calls,
        }
        self._step = None
        self._calls = []
        self._open_before = []
        return result

    def summary(self, measured_steps: list[dict[str, Any]]) -> dict[str, Any]:
        phase_rows = [
            row["vision_phase_memory"]
            for row in measured_steps
            if row.get("vision_phase_memory") is not None
        ]
        return {
            "module_name": self.module_name,
            "module_class": self.module_class,
            "measured_steps": len(measured_steps),
            "measured_steps_with_visual_forward": sum(
                int(row["forward_calls"] > 0) for row in phase_rows
            ),
            "all_measured_steps_observed": bool(
                measured_steps
                and len(phase_rows) == len(measured_steps)
                and all(row["forward_calls"] > 0 for row in phase_rows)
            ),
            "max_allocated_during_vision": max(
                (int(row["max_allocated_during_vision"]) for row in phase_rows),
                default=0,
            ),
            "max_reserved_during_vision": max(
                (int(row["max_reserved_during_vision"]) for row in phase_rows),
                default=0,
            ),
        }


class ExperimentCallback(TrainerCallback):
    def __init__(self, metrics_dir: Path, warmup_steps: int, job_metadata: dict[str, Any]):
        self.metrics_dir = metrics_dir
        self.warmup_steps = warmup_steps
        self.job_metadata = job_metadata
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        validate_rank_binding(self.rank, self.local_rank, self.world_size)
        self.job_id = validate_job_id(job_metadata.get("job_id"))
        self.execution_attempt_id = validate_execution_attempt_id(
            job_metadata.get("_execution_attempt_id")
        )
        self.training_mode = validate_training_mode(job_metadata.get("train_type"))
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.metrics_dir / f"events.rank{self.rank}.jsonl"
        self.summary_path = self.metrics_dir / f"summary.rank{self.rank}.json"
        manifest_name = (
            f"runtime_model_manifest.{self.execution_attempt_id}.rank{self.rank}.json"
        )
        self.runtime_model_manifest_path = self.metrics_dir / manifest_name
        # Component-aware freeze/vision evidence is a sidecar so the strict
        # historical runtime-model-manifest v2 schema remains unchanged.
        self.runtime_structure_manifest_path = self.metrics_dir / (
            f"model_structure_manifest.{self.execution_attempt_id}.rank{self.rank}.json"
        )
        self._train_started = 0.0
        self._step_started = 0.0
        self._micro_started = 0.0
        self._optimizer_started = 0.0
        self._micro_times: list[float] = []
        self._current_micro_times: list[float] = []
        self._step_records: list[dict[str, Any]] = []
        self._last_counter_snapshot = {key: 0 for key in COUNTERS.snapshot()}
        self._pending_tokens = self._last_counter_snapshot.copy()
        self._initial_memory: dict[str, int] = {}
        self._failure: str | None = None
        self._device_attestation: dict[str, Any] | None = None
        self._vision_phase_probe: VisionPhaseMemoryProbe | None = None

    @staticmethod
    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _memory() -> dict[str, int]:
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
            "max_reserved": torch.cuda.max_memory_reserved(),
        }

    def _event(self, event: str, **payload: Any) -> None:
        record = {
            **payload,
            "time_unix": time.time(),
            "event": event,
            "job_id": self.job_id,
            "execution_attempt_id": self.execution_attempt_id,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
        }
        with self.events_path.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        model = kwargs.get("model")
        if model is None:
            raise RuntimeError(
                "Trainer did not provide model to on_train_begin; "
                "runtime model inventory cannot be proven"
            )
        self._device_attestation = capture_runtime_device_attestation(self.local_rank)
        manifest = write_runtime_model_manifest(
            model,
            self.runtime_model_manifest_path,
            rank=self.rank,
            local_rank=self.local_rank,
            world_size=self.world_size,
            job_id=self.job_id,
            execution_attempt_id=self.execution_attempt_id,
            training_mode=self.training_mode,
            device_attestation=self._device_attestation,
        )
        from model_structure_manifest import build_model_structure_manifest

        structure_manifest = build_model_structure_manifest(
            manifest,
            job_metadata=self.job_metadata,
            runtime_manifest_path=self.runtime_model_manifest_path,
        )
        _atomic_write_json(self.runtime_structure_manifest_path, structure_manifest)
        if self.job_metadata.get("enable_vision_phase_memory_probe") is True:
            self._vision_phase_probe = VisionPhaseMemoryProbe.install(model)
        self._sync()
        self._train_started = time.perf_counter()
        self._initial_memory = self._memory()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._event(
            "train_begin",
            world_size=self.world_size,
            local_rank=self.local_rank,
            initial_memory=self._initial_memory,
            metadata=self.job_metadata,
            runtime_model_manifest={
                "path": self.runtime_model_manifest_path.name,
                "inventory_sha256": manifest["inventory_sha256"],
                "device_attestation_sha256": manifest[
                    "device_attestation_sha256"
                ],
                "unique_tensor_count": manifest["inventory"]["unique_tensor_count"],
                "logical_parameter_elements": manifest["inventory"][
                    "logical_parameter_elements"
                ],
                "trainable_parameter_elements": manifest["inventory"][
                    "trainable_parameter_elements"
                ],
            },
            runtime_structure_manifest={
                "path": self.runtime_structure_manifest_path.name,
                "manifest_sha256": structure_manifest["manifest_sha256"],
                "visual_path_observed": structure_manifest["visual_path_observed"],
                "declaration_status": structure_manifest["declaration_status"],
            },
        )

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._sync()
        now = time.perf_counter()
        current = COUNTERS.snapshot()
        self._pending_tokens = delta(current, self._last_counter_snapshot)
        self._last_counter_snapshot = current
        self._step_started = now
        self._micro_started = now
        self._current_micro_times = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self._vision_phase_probe is not None:
            self._vision_phase_probe.start_step(int(state.global_step) + 1)

    def on_substep_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._sync()
        now = time.perf_counter()
        duration = now - self._micro_started
        self._current_micro_times.append(duration)
        self._micro_times.append(duration)
        self._micro_started = now

    def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._sync()
        now = time.perf_counter()
        duration = now - self._micro_started
        self._current_micro_times.append(duration)
        self._micro_times.append(duration)
        self._optimizer_started = now

    def on_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._sync()
        self._last_optimizer_seconds = time.perf_counter() - self._optimizer_started

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._sync()
        step_seconds = time.perf_counter() - self._step_started
        record = {
            "global_step": int(state.global_step),
            "step_seconds": step_seconds,
            "micro_step_seconds": self._current_micro_times,
            "optimizer_step_seconds": getattr(self, "_last_optimizer_seconds", None),
            "tokens": self._pending_tokens,
            "memory": self._memory(),
            "vision_phase_memory": (
                self._vision_phase_probe.finish_step(int(state.global_step))
                if self._vision_phase_probe is not None
                else None
            ),
            "is_warmup": int(state.global_step) <= self.warmup_steps,
        }
        self._step_records.append(record)
        self._event("step_end", **record)

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._event("trainer_log", global_step=int(state.global_step), logs=logs or {})

    def record_failure(self, error: BaseException) -> None:
        self._failure = f"{type(error).__name__}: {error}"
        self._event("failure", error=self._failure, memory=self._memory())

    def _write_summary(self) -> None:
        measured = [row for row in self._step_records if not row["is_warmup"]]
        warmup = [row for row in self._step_records if row["is_warmup"]]
        step_times = [row["step_seconds"] for row in measured]
        optimizer_times = [row["optimizer_step_seconds"] for row in measured if row["optimizer_step_seconds"] is not None]
        gradient_accumulation_steps = int(
            self.job_metadata.get("gradient_accumulation_steps", 1)
        )
        token_ledger_evidence = slice_consumed_token_ledger(
            COUNTERS.ledger_snapshot(),
            gradient_accumulation_steps=gradient_accumulation_steps,
            completed_warmup_steps=len(warmup),
            completed_measured_steps=len(measured),
        )
        batch_shape_evidence = slice_consumed_batch_shape_ledger(
            COUNTERS.batch_shape_ledger_snapshot(),
            gradient_accumulation_steps=gradient_accumulation_steps,
            completed_warmup_steps=len(warmup),
            completed_measured_steps=len(measured),
        )
        for shape_step, runtime_step in zip(
            batch_shape_evidence["measured_steps"], measured
        ):
            shape_step["global_step"] = runtime_step["global_step"]
            shape_step["memory"] = dict(runtime_step["memory"])
        token_totals = token_ledger_evidence["measured_totals"]
        token_evidence_authoritative = bool(token_ledger_evidence["authoritative"])
        measured_seconds = sum(step_times)
        runtime_batch_evidence = BATCH_EVIDENCE.snapshot()
        if self.runtime_model_manifest_path.is_file():
            from model_structure_manifest import build_model_structure_manifest

            with self.runtime_model_manifest_path.open("r", encoding="utf-8") as source:
                runtime_manifest = json.load(source)
            structure_manifest = build_model_structure_manifest(
                runtime_manifest,
                job_metadata=self.job_metadata,
                runtime_manifest_path=self.runtime_model_manifest_path,
                runtime_media_evidence=runtime_batch_evidence["media"],
            )
            _atomic_write_json(self.runtime_structure_manifest_path, structure_manifest)
        summary = {
            "job_id": self.job_id,
            "execution_attempt_id": self.execution_attempt_id,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "training_mode": self.training_mode,
            "device_attestation_sha256": (
                sha256_json(self._device_attestation)
                if self._device_attestation is not None
                else None
            ),
            "failure": self._failure,
            "initial_memory": self._initial_memory,
            "total_train_seconds": time.perf_counter() - self._train_started,
            "total_steps": len(self._step_records),
            "measured_steps": len(measured),
            "measured_seconds": measured_seconds,
            "median_step_seconds": statistics.median(step_times) if step_times else None,
            "p90_step_seconds": sorted(step_times)[min(len(step_times) - 1, int(0.9 * len(step_times)))] if step_times else None,
            "median_optimizer_seconds": statistics.median(optimizer_times) if optimizer_times else None,
            "measured_totals": token_totals,
            "token_ledger_evidence": token_ledger_evidence,
            "batch_shape_evidence": batch_shape_evidence,
            "computed_tokens_per_second": token_totals["computed_tokens"] / measured_seconds if measured_seconds and token_evidence_authoritative else None,
            "effective_tokens_per_second": token_totals["effective_tokens"] / measured_seconds if measured_seconds and token_evidence_authoritative else None,
            "logical_samples_per_second": token_totals["logical_samples"] / measured_seconds if measured_seconds and token_evidence_authoritative else None,
            "max_allocated": max((row["memory"].get("max_allocated", 0) for row in self._step_records), default=0),
            "max_reserved": max((row["memory"].get("max_reserved", 0) for row in self._step_records), default=0),
            "vision_phase_memory_probe": (
                self._vision_phase_probe.summary(measured)
                if self._vision_phase_probe is not None
                else None
            ),
            "metadata": self.job_metadata,
            "runtime_batch_evidence": runtime_batch_evidence,
            "runtime_structure_manifest": {
                "path": self.runtime_structure_manifest_path.name,
                "manifest_sha256": (
                    structure_manifest["manifest_sha256"]
                    if self.runtime_model_manifest_path.is_file()
                    else None
                ),
                "visual_path_observed": (
                    structure_manifest["visual_path_observed"]
                    if self.runtime_model_manifest_path.is_file()
                    else False
                ),
            },
        }
        _atomic_write_json(self.summary_path, summary)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._sync()
        self._event("train_end", memory=self._memory())
        self._write_summary()

    def finalize_after_failure(self) -> None:
        self._write_summary()
