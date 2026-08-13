#!/usr/bin/env python3
"""Validate selected local Qwen checkpoints and record structural fingerprints."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from safetensors import safe_open
from transformers import AutoTokenizer

from common import ARTIFACT_DIR, CONFIG_DIR, read_json, sha256_file, write_json


def product(shape: list[int]) -> int:
    return math.prod(shape)


def safetensor_files(model_path: Path) -> list[Path]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = read_json(index_path)
        names = sorted(set(index["weight_map"].values()))
        files = [model_path / name for name in names]
    else:
        files = sorted(model_path.glob("*.safetensors"))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint shards: {missing}")
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint found in {model_path}")
    return files


def count_parameters(files: list[Path]) -> tuple[int, int]:
    parameters = 0
    tensors = 0
    seen: set[str] = set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key in seen:
                    raise RuntimeError(f"Duplicate tensor {key} in checkpoint shards")
                seen.add(key)
                parameters += product(list(checkpoint.get_slice(key).get_shape()))
                tensors += 1
    return parameters, tensors


def _geometry_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the sub-config carrying transformer geometry.

    Dense Qwen3 exposes geometry at the top level; VL and newer nested
    architectures (qwen3_vl, qwen3_5) place the language tower under
    ``text_config``.  The returned geometry is always the language-tower
    geometry; vision geometry is collected separately by
    :func:`_vision_geometry`.
    """
    if config.get("hidden_size") is not None:
        return config
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and text_config.get("hidden_size") is not None:
        return text_config
    return config


VISION_GEOMETRY_FIELDS = (
    "model_type",
    "depth",
    "hidden_size",
    "intermediate_size",
    "num_heads",
    "patch_size",
    "spatial_merge_size",
    "temporal_patch_size",
    "out_hidden_size",
    "in_channels",
    "num_position_embeddings",
)


def _vision_config(config: dict[str, Any]) -> dict[str, Any] | None:
    value = config.get("vision_config")
    return value if isinstance(value, dict) else None


def _vision_geometry(config: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize the raw vision tower config without inventing defaults."""

    vision = _vision_config(config)
    if vision is None:
        return None
    # Some model generations call the layer count ``num_hidden_layers`` while
    # Qwen2.5/3-VL expose ``depth``.  Keep the canonical field name stable and
    # retain the raw config alongside it for processor/version binding.
    normalized = {
        field: vision.get(field)
        for field in VISION_GEOMETRY_FIELDS
    }
    if normalized["depth"] is None:
        normalized["depth"] = vision.get("num_hidden_layers")
    return normalized


def _is_vision_language_config(config: dict[str, Any]) -> bool:
    architectures = " ".join(
        str(value).lower() for value in (config.get("architectures") or [])
    )
    model_type = str(config.get("model_type") or "").lower()
    family = str(config.get("_name_or_path") or "").lower()
    return bool(
        _vision_config(config)
        or "vision" in architectures
        or "vl" in architectures
        or "vision" in model_type
        or "_vl" in model_type
        or "vl" in family
    )


def _tensor_component(tensor_name: str) -> str:
    """Classify checkpoint tensors for an auditable parameter breakdown."""

    lowered = tensor_name.lower()
    tokens = set(lowered.replace("/", ".").split("."))
    visual_tokens = {
        "visual",
        "vision",
        "vision_tower",
        "vision_model",
    }
    if tokens.intersection(visual_tokens) or "vision_tower" in lowered:
        if any(
            marker in lowered
            for marker in (
                "merger",
                "projector",
                "multi_modal_projector",
                "multimodal_projector",
            )
        ):
            return "projector_or_merger"
        return "vision_tower"
    if any(
        marker in lowered
        for marker in (
            "multi_modal_projector",
            "multimodal_projector",
            "vision_projector",
        )
    ):
        return "projector_or_merger"
    return "language_model_or_other"


def component_parameter_estimates(files: list[Path]) -> dict[str, int]:
    """Count logical checkpoint elements by component without model-specific multipliers."""

    counts = {
        "language_model_or_other": 0,
        "vision_tower": 0,
        "projector_or_merger": 0,
    }
    seen: set[str] = set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key in seen:
                    raise RuntimeError(f"Duplicate tensor {key} in checkpoint shards")
                seen.add(key)
                component = _tensor_component(key)
                counts[component] += product(list(checkpoint.get_slice(key).get_shape()))
    return counts


def inventory_model(entry: dict[str, Any]) -> dict[str, Any]:
    model_path = Path(entry["path"])
    tokenizer_path = Path(entry["tokenizer_path"])
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(tokenizer_path)
    config = read_json(model_path / "config.json")
    geometry = _geometry_config(config)
    vision_config = _vision_config(config)
    vision_geometry = _vision_geometry(config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
    files = safetensor_files(model_path)
    parameters, tensor_count = count_parameters(files)
    component_parameters = component_parameter_estimates(files)
    architectures = config.get("architectures") or []
    context = geometry.get("max_position_embeddings") or config.get("max_position_embeddings")
    if not isinstance(context, int) or context < 32768:
        raise RuntimeError(f"{entry['id']} has insufficient context limit: {context}")
    attention_heads = geometry.get("num_attention_heads")
    hidden_size = geometry.get("hidden_size")
    explicit_head_dim = geometry.get("head_dim")
    if explicit_head_dim is not None:
        head_dim = int(explicit_head_dim)
    elif isinstance(hidden_size, int) and isinstance(attention_heads, int) and attention_heads:
        head_dim = hidden_size // attention_heads
    else:
        head_dim = None
    vocab_size = geometry.get("vocab_size")
    checkpoint_manifest = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in files
    ]
    return {
        **entry,
        "actual_parameters": parameters,
        "actual_parameters_b": parameters / 1e9,
        "checkpoint_bytes": sum(path.stat().st_size for path in files),
        "checkpoint_shards": len(files),
        "checkpoint_manifest": checkpoint_manifest,
        "model_directory_name": model_path.name,
        "model_identity": str(model_path),
        "revision_policy": "The configured local model directory name/path is the authoritative model identity.",
        "tokenizer_identity": str(tokenizer_path),
        "tokenizer_directory_name": tokenizer_path.name,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_size": len(tokenizer),
        "tokenizer_max_id": max(tokenizer.get_vocab().values()),
        "tokenizer_fits_model_embeddings": max(tokenizer.get_vocab().values()) < int(vocab_size),
        "tensor_count": tensor_count,
        "model_type": config.get("model_type"),
        "architectures": architectures,
        "architecture_role": (
            "vision_language"
            if _is_vision_language_config(config)
            else "dense_language"
        ),
        "is_vision_language": _is_vision_language_config(config),
        "vision_config": vision_config,
        "vision_geometry": vision_geometry,
        "component_parameter_estimates": component_parameters,
        "vision_parameter_estimate": (
            component_parameters["vision_tower"]
            + component_parameters["projector_or_merger"]
        ),
        "hidden_size": hidden_size,
        "intermediate_size": geometry.get("intermediate_size"),
        "num_hidden_layers": geometry.get("num_hidden_layers"),
        "num_attention_heads": attention_heads,
        "num_key_value_heads": geometry.get("num_key_value_heads"),
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "max_position_embeddings": context,
        "torch_dtype": geometry.get("torch_dtype") or config.get("torch_dtype")
        or geometry.get("dtype") or config.get("dtype"),
        "geometry_source": "text_config" if geometry is not config else "top_level",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "model_inventory_vl_v1.json",
        help=(
            "Output manifest.  The default is the additive VL extension; "
            "writing the frozen model_inventory.json requires an explicit "
            "--output and a separate re-freeze."
        ),
    )
    arguments = parser.parse_args()
    catalog = read_json(CONFIG_DIR / "models.json")
    models = []
    for entry in catalog["models"]:
        print(f"Inspecting {entry['id']} at {entry['path']}", flush=True)
        models.append(inventory_model(entry))
    result = {
        "schema_version": 1,
        "catalog_sha256": sha256_file(CONFIG_DIR / "models.json"),
        "selection_policy": catalog["selection_policy"],
        "fixed_lora": catalog["fixed_lora"],
        "models": models,
    }
    write_json(arguments.output, result)
    print("id\tactual_B\tcontext\tshards\ttrain_types")
    for model in models:
        print(
            f"{model['id']}\t{model['actual_parameters_b']:.4f}\t{model['max_position_embeddings']}\t"
            f"{model['checkpoint_shards']}\t{','.join(model['train_types'])}"
        )


if __name__ == "__main__":
    main()
