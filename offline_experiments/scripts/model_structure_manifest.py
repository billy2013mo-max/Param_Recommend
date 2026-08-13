#!/usr/bin/env python3
"""Derive a component-aware model structure manifest from runtime evidence.

The existing ``sft_runtime_model_manifest`` schema is intentionally strict and
already consumed by historical exporters.  This adapter does not fork or
mutate that schema.  It derives a sidecar with independent language/vision /
projector freeze markers, parameter accounting and LoRA target-hit flags.

All component decisions are based on observed parameter names and
``requires_grad`` values.  ``training_mode=full`` is never used as a proxy for
vision-tower state.  An optional job declaration is compared to observation,
but a missing declaration remains an explicit ``observed_only`` state.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, ROOT, read_json, sha256_file, write_json
from runtime_evidence import (
    RUNTIME_MODEL_MANIFEST_SCHEMA,
    RUNTIME_MODEL_MANIFEST_SCHEMA_VERSION,
    RuntimeEvidenceError,
    sha256_json,
    validate_runtime_model_manifest,
)


SCHEMA = "sft_model_structure_manifest/v1"
SCHEMA_VERSION = 1
COMPONENTS = ("language_model", "vision_tower", "multimodal_projector", "other")
FREEZE_FIELDS = (
    "freeze_vision_tower",
    "freeze_multi_modal_projector",
    "freeze_language_model",
)

_VISION_MARKERS = ("visual", "vision", "vision_tower", "vision_model")
_PROJECTOR_MARKERS = (
    "projector",
    "merger",
    "multi_modal_projector",
    "multimodal_projector",
)
_ADAPTER_MARKERS = (
    ".lora_",
    "lora_a",
    "lora_b",
    "lora_embedding",
    ".adapter",
    "ia3",
)


def _name_of(tensor: Mapping[str, Any]) -> str:
    aliases = tensor.get("aliases") or []
    if aliases:
        return str(aliases[0]).lower()
    return str(tensor.get("canonical_name") or "").lower()


def _is_adapter(name: str) -> bool:
    return any(marker in name for marker in _ADAPTER_MARKERS)


def classify_component(name: str) -> str:
    """Classify a tensor name without using model ID or dataset ID."""

    lowered = str(name).lower()
    if any(marker in lowered for marker in _PROJECTOR_MARKERS):
        return "multimodal_projector"
    if any(marker in lowered for marker in _VISION_MARKERS):
        return "vision_tower"
    # A dense text checkpoint does not always spell out language_model.  Any
    # tensor not identified as visual/projector is conservatively language-side
    # unless it is an explicitly unknown auxiliary component.
    return "language_model"


def _empty_component() -> dict[str, Any]:
    return {
        "tensor_count": 0,
        "logical_parameter_elements": 0,
        "base_parameter_elements": 0,
        "base_trainable_parameter_elements": 0,
        "adapter_parameter_elements": 0,
        "adapter_trainable_parameter_elements": 0,
        "trainable_parameter_elements": 0,
        "frozen_parameter_elements": 0,
        "adapter_tensor_count": 0,
    }


def _component_totals(tensors: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {component: _empty_component() for component in COMPONENTS}
    for tensor in tensors:
        name = _name_of(tensor)
        component = classify_component(name)
        # Keep the schema extensible while ensuring the current classifier
        # never silently loses a tensor.
        if component not in result:
            component = "other"
        logical = int(tensor.get("logical_numel") or 0)
        trainable = bool(tensor.get("requires_grad"))
        adapter = _is_adapter(name)
        row = result[component]
        row["tensor_count"] += 1
        row["logical_parameter_elements"] += logical
        if adapter:
            row["adapter_tensor_count"] += 1
            row["adapter_parameter_elements"] += logical
            if trainable:
                row["adapter_trainable_parameter_elements"] += logical
        else:
            row["base_parameter_elements"] += logical
            if trainable:
                row["base_trainable_parameter_elements"] += logical
        if trainable:
            row["trainable_parameter_elements"] += logical
        else:
            row["frozen_parameter_elements"] += logical
    return result


def _freeze_marker(stats: Mapping[str, Any]) -> dict[str, Any]:
    total = int(stats["base_parameter_elements"])
    trainable = int(stats["base_trainable_parameter_elements"])
    adapter_trainable = int(stats["adapter_trainable_parameter_elements"])
    if int(stats["tensor_count"]) == 0:
        return {
            "value": None,
            "status": "not_applicable",
            "reason": "component_absent_from_runtime_inventory",
        }
    if total <= 0:
        # Adapter-only components cannot prove a base freeze state.
        return {
            "value": None,
            "status": "unknown",
            "reason": "component_has_no_base_tensors",
        }
    if trainable == 0:
        return {
            "value": True,
            "status": "base_frozen_adapter_trainable" if adapter_trainable else "frozen",
            "reason": "observed_requires_grad",
        }
    if trainable == total:
        return {
            "value": False,
            "status": "trainable",
            "reason": "observed_requires_grad",
        }
    return {
        "value": False,
        "status": "partial",
        "reason": "mixed_base_requires_grad",
    }


def _declaration_observed_value(
    stats: Mapping[str, Any],
    marker: Mapping[str, Any],
    training_mode: str | None,
) -> bool | None:
    """Interpret freeze declarations at the component, not base-tensor, level.

    LoRA freezes base tensors by construction.  A component with trainable LoRA
    tensors is therefore observed as enabled (freeze=false), while a component
    with frozen base tensors and no trainable adapter is observed as frozen.
    """

    if int(stats.get("tensor_count") or 0) == 0:
        return None
    if training_mode == "lora":
        if int(stats.get("adapter_trainable_parameter_elements") or 0) > 0:
            return False
        if int(stats.get("base_trainable_parameter_elements") or 0) == 0:
            return True
    value = marker.get("value")
    return value if value in (True, False, None) else None


def _declared_flags(job_metadata: Mapping[str, Any] | None) -> dict[str, bool | None]:
    metadata = job_metadata or {}
    result: dict[str, bool | None] = {}
    for field in FREEZE_FIELDS:
        value = metadata.get(field)
        result[field] = value if isinstance(value, bool) else None
    return result


def build_model_structure_manifest(
    runtime_manifest: Mapping[str, Any],
    *,
    job_metadata: Mapping[str, Any] | None = None,
    declared_model: Mapping[str, Any] | None = None,
    runtime_manifest_path: Path | None = None,
    runtime_media_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a sidecar manifest from one validated runtime rank manifest."""

    validate_runtime_model_manifest(runtime_manifest)
    if runtime_manifest.get("schema") != RUNTIME_MODEL_MANIFEST_SCHEMA:
        raise RuntimeEvidenceError("runtime manifest schema is not supported")
    if runtime_manifest.get("schema_version") != RUNTIME_MODEL_MANIFEST_SCHEMA_VERSION:
        raise RuntimeEvidenceError("runtime manifest version is not supported")

    inventory = runtime_manifest["inventory"]
    components = _component_totals(inventory["tensors"])
    observed_flags = {
        "freeze_vision_tower": _freeze_marker(components["vision_tower"]),
        "freeze_multi_modal_projector": _freeze_marker(
            components["multimodal_projector"]
        ),
        "freeze_language_model": _freeze_marker(components["language_model"]),
    }
    declaration_observed_flags = {
        "freeze_vision_tower": _declaration_observed_value(
            components["vision_tower"],
            observed_flags["freeze_vision_tower"],
            runtime_manifest.get("training_mode"),
        ),
        "freeze_multi_modal_projector": _declaration_observed_value(
            components["multimodal_projector"],
            observed_flags["freeze_multi_modal_projector"],
            runtime_manifest.get("training_mode"),
        ),
        "freeze_language_model": _declaration_observed_value(
            components["language_model"],
            observed_flags["freeze_language_model"],
            runtime_manifest.get("training_mode"),
        ),
    }
    declared_flags = _declared_flags(job_metadata)
    declaration_mismatches = []
    for field in FREEZE_FIELDS:
        declared = declared_flags[field]
        observed = declaration_observed_flags[field]
        if declared is not None and observed is not None and declared != observed:
            declaration_mismatches.append(field)

    visual_stats = components["vision_tower"]
    projector_stats = components["multimodal_projector"]
    visual_adapter_hit = bool(
        visual_stats["adapter_trainable_parameter_elements"]
        or projector_stats["adapter_trainable_parameter_elements"]
    )
    media_evidence = dict(runtime_media_evidence or {})
    real_image_path_observed = bool(
        media_evidence.get("real_image_path_observed") is True
        and int(media_evidence.get("source_image_count") or 0) > 0
        and int(media_evidence.get("media_batches") or 0) > 0
        and int(media_evidence.get("image_grid_rows") or 0) > 0
        and int(media_evidence.get("pixel_value_elements") or 0) > 0
    )
    real_video_path_observed = bool(
        media_evidence.get("real_video_path_observed") is True
        and int(media_evidence.get("source_video_count") or 0) > 0
        and int(media_evidence.get("media_batches") or 0) > 0
        and int(media_evidence.get("video_grid_rows") or 0) > 0
        and int(media_evidence.get("pixel_video_elements") or 0) > 0
    )
    model = declared_model or {}
    unsigned = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "job_id": runtime_manifest.get("job_id"),
        "execution_attempt_id": runtime_manifest.get("execution_attempt_id"),
        "rank": runtime_manifest.get("rank"),
        "world_size": runtime_manifest.get("world_size"),
        "training_mode": runtime_manifest.get("training_mode"),
        "model_id": (job_metadata or {}).get("model_id"),
        "model_class": inventory.get("model_class"),
        "architecture_role": model.get("architecture_role"),
        "is_vision_language": model.get("is_vision_language"),
        "vision_parameters_observed": bool(visual_stats["tensor_count"]),
        "visual_path_observed": bool(real_image_path_observed or real_video_path_observed),
        "runtime_media_evidence": media_evidence,
        "runtime_inventory_sha256": runtime_manifest.get("inventory_sha256"),
        "components": components,
        "freeze_flags": observed_flags,
        "declaration_observed_flags": declaration_observed_flags,
        "declared_freeze_flags": declared_flags,
        "declaration_status": (
            "matched"
            if not declaration_mismatches and any(value is not None for value in declared_flags.values())
            else "observed_only"
            if not any(value is not None for value in declared_flags.values())
            else "mismatch"
        ),
        "declaration_mismatches": declaration_mismatches,
        "lora_target_hits": {
            "vision_tower": bool(visual_stats["adapter_trainable_parameter_elements"]),
            "multimodal_projector": bool(
                projector_stats["adapter_trainable_parameter_elements"]
            ),
            "any_visual_component": visual_adapter_hit,
        },
        "binding": {
            "runtime_manifest_path": (
                str(runtime_manifest_path.resolve()) if runtime_manifest_path else None
            ),
            "runtime_manifest_sha256": (
                sha256_file(runtime_manifest_path)
                if runtime_manifest_path and runtime_manifest_path.is_file()
                else None
            ),
            "observed_requires_grad": True,
            "runtime_media_evidence_sha256": (
                sha256_json(media_evidence) if media_evidence else None
            ),
        },
    }
    return {**unsigned, "manifest_sha256": sha256_json(unsigned)}


def validate_model_structure_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise ValueError("model structure manifest schema mismatch")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("model structure manifest version mismatch")
    digest = value.get("manifest_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    if digest != sha256_json(unsigned):
        raise ValueError("model structure manifest checksum mismatch")
    for field in FREEZE_FIELDS:
        marker = (value.get("freeze_flags") or {}).get(field)
        if not isinstance(marker, Mapping) or marker.get("value") not in (True, False, None):
            raise ValueError(f"invalid freeze marker: {field}")
    return dict(value)


def _model_row(inventory: Mapping[str, Any], model_id: str | None) -> Mapping[str, Any] | None:
    if not model_id:
        return None
    for row in inventory.get("models") or []:
        if isinstance(row, Mapping) and row.get("id") == model_id:
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--job-metadata", type=Path)
    parser.add_argument("--model-inventory", type=Path, default=ARTIFACT_DIR / "model_inventory_vl_v1.json")
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "model_structure_manifest_v1.json")
    args = parser.parse_args()
    runtime_manifest = read_json(args.runtime_manifest)
    job_metadata = read_json(args.job_metadata) if args.job_metadata and args.job_metadata.is_file() else None
    inventory = read_json(args.model_inventory) if args.model_inventory.is_file() else {}
    model_id = (job_metadata or {}).get("model_id")
    result = build_model_structure_manifest(
        runtime_manifest,
        job_metadata=job_metadata,
        declared_model=_model_row(inventory, model_id),
        runtime_manifest_path=args.runtime_manifest,
    )
    validate_model_structure_manifest(result)
    write_json(args.output, result)
    print(
        f"wrote {args.output}; visual_path_observed={result['visual_path_observed']} "
        f"declaration_status={result['declaration_status']}"
    )


if __name__ == "__main__":
    main()
