#!/usr/bin/env python
"""Derive vision-tower structure from a VL model config, statically.

The runtime sidecar in :mod:`model_structure_manifest` can already split
language / vision / projector parameters and recover the three freeze flags --
but only from a *runtime* rank manifest, i.e. only after a training run exists.
Recommendation happens before any run, so the predictor path currently has no
vision-tower information at all: ``inventory_models._geometry_config`` reads
``text_config`` and its docstring states the vision tower is not inventoried.

That omission is why a VL request is priced with a dense-only formula.  This
module supplies the missing static half: it parses ``vision_config`` and derives
per-component parameter counts analytically, so a VL candidate can at least be
*conservatively* accounted for and explicitly downgraded rather than silently
under-counted.

Scope discipline, because this is the part that is easy to overclaim:

* Resident vision **weights** are analytic and verifiable -- and this module
  verifies them against the checkpoint when the safetensors index is available.
* Vision **activation**, multimodal workspace, ZeRO gather behaviour and the
  image-token pipeline are *not* modelled here.  On the measured Qwen3-VL
  evidence the resident vision weights account for roughly a twentieth of the
  observed memory gap, so this module must never be read as explaining it.

Two generations differ in ways that matter and are handled separately: Qwen3-VL
uses patch 16 with deepstack mergers, Qwen2.5-VL uses patch 14 with windowed
attention and a ``tokens_per_second`` video cadence.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, sha256_json, write_json

SCHEMA = "sft_vision_tower_structure/v1"
IMPLEMENTATION_VERSION = "sft_vision_tower_structure_impl/2026-08-01.v1"

DEFAULT_OUTPUT = ARTIFACT_DIR / "vision_tower_structure_v1.json"

BYTES_PER_BF16 = 2

# Generations whose block structure has been derived and verified against a real
# checkpoint.  Anything else fails closed: per the design rule, an unsupported
# vision model must be marked unsupported rather than priced with another
# generation's formula.  Qwen2-VL, for instance, uses a different config layout
# (``embed_dim`` / ``mlp_ratio`` instead of ``hidden_size`` /
# ``intermediate_size``) and is deliberately not inferred from Qwen2.5-VL.
VERIFIED_GENERATIONS = ("qwen3_vl", "qwen2_5_vl")

# Measured on Qwen3-VL-8B at mbs=1: reserved 36.41 GiB observed vs 15.22 GiB
# predicted by the dense skeleton.  Recorded so the report can state, in
# numbers, how little of that gap resident vision weights explain.
QWEN3_VL_8B_MBS1_OBSERVED_GAP_BYTES = int((36.41 - 15.22) * 1024**3)


class VisionStructureError(ValueError):
    """Raised when a config cannot be interpreted as a supported vision tower."""


def _int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise VisionStructureError(f"{name} must be an integer") from error
    if result <= 0:
        raise VisionStructureError(f"{name} must be positive")
    return result


def detect_generation(config: Mapping[str, Any]) -> str:
    """Identify the vision-tower generation from the model config.

    Ordering matters and is deliberate.  A recognised ``model_type`` always
    wins; the structural key fallbacks below only speak for configs that do not
    name a generation we know.  Two traps the fallbacks used to walk into:

    * Qwen3.5 declares ``model_type: "qwen3_5"`` and carries a *present but
      empty* ``deepstack_visual_indexes: []``.  A bare ``in`` test on that key
      reported ``qwen3_vl``, which then passed :data:`VERIFIED_GENERATIONS` and
      priced a brand-new generation with Qwen3-VL's formula.  A genuine
      deepstack tower has non-empty indexes, so the fallback requires that.
    * The tower may name its own generation in ``vision_config.model_type``.
      When that name is one we have no verified formula for, it *contradicts*
      any structural guess, so the guess must not be trusted.  The rejected
      name is not lost -- the report records the raw ``model_type`` next to the
      generation verdict.

    A top-level placeholder like ``model_type: "unknown"`` is not a
    contradiction: the structural fallbacks still speak for it.
    """

    model_type = str(config.get("model_type") or "").lower()
    architectures = " ".join(
        str(value).lower() for value in (config.get("architectures") or [])
    )
    vision = config.get("vision_config") or {}
    if "qwen3_vl" in model_type or "qwen3vl" in architectures:
        return "qwen3_vl"
    if "qwen2_5_vl" in model_type or "qwen2_5vl" in architectures:
        return "qwen2_5_vl"
    if "qwen2_vl" in model_type or "qwen2vl" in architectures:
        return "qwen2_vl"
    tower_type = str(vision.get("model_type") or "").lower()
    if tower_type and tower_type not in VERIFIED_GENERATIONS:
        # The tower names a generation we cannot price.  Never structurally
        # guess past that -- it is how Qwen3.5 was read as Qwen3-VL.
        return "unknown_vision_generation"
    if vision.get("deepstack_visual_indexes"):
        return "qwen3_vl"
    if "window_size" in vision or "fullatt_block_indexes" in vision:
        return "qwen2_5_vl"
    return "unknown_vision_generation"


def _merger_parameters(
    *,
    hidden_size: int,
    spatial_merge_size: int,
    out_hidden_size: int,
    gated_mlp: bool,
) -> int:
    """Parameters of one merger block (norm + two linears).

    Qwen2.5-VL normalises with RMSNorm over ``hidden_size`` (weight only);
    Qwen3-VL uses a LayerNorm over the merged width (weight and bias).
    """

    merged = hidden_size * spatial_merge_size * spatial_merge_size
    norm = hidden_size if gated_mlp else 2 * merged
    first = merged * merged + merged
    second = merged * out_hidden_size + out_hidden_size
    return norm + first + second


def vision_tower_parameters(
    vision_config: Mapping[str, Any], *, generation: str = "qwen3_vl"
) -> dict[str, int]:
    """Analytic per-component parameter counts for the vision tower.

    The two generations differ inside the block in two ways that together change
    the total by ~20%, so the generation must be passed rather than assumed:

    * Qwen2.5-VL blocks use a **gated** MLP (gate/up/down, all biased) and
      RMSNorm (weight only).
    * Qwen3-VL blocks use a two-layer MLP (fc1/fc2) and LayerNorm (weight+bias),
      plus learned position embeddings and one extra merger per deepstack tap.
    """

    depth = _int(vision_config.get("depth"), "depth")
    hidden = _int(vision_config.get("hidden_size"), "hidden_size")
    intermediate = _int(vision_config.get("intermediate_size"), "intermediate_size")
    patch = _int(vision_config.get("patch_size"), "patch_size")
    out_hidden = _int(vision_config.get("out_hidden_size"), "out_hidden_size")
    merge = _int(vision_config.get("spatial_merge_size"), "spatial_merge_size")
    temporal = _int(vision_config.get("temporal_patch_size", 2), "temporal_patch_size")
    channels = _int(
        vision_config.get("in_channels", vision_config.get("in_chans", 3)),
        "in_channels",
    )
    gated_mlp = generation in {"qwen2_5_vl", "qwen2_vl"}

    # Conv3d(in_channels, hidden, kernel=(temporal_patch, patch, patch)), no bias.
    patch_embed = channels * hidden * temporal * patch * patch

    attention = 3 * hidden * hidden + 3 * hidden + hidden * hidden + hidden
    if gated_mlp:
        # gate_proj + up_proj (both hidden -> intermediate) + down_proj, biased.
        mlp = 2 * (intermediate * hidden + intermediate) + (hidden * intermediate + hidden)
        norms = 2 * hidden  # two RMSNorms, weight only
    else:
        mlp = hidden * intermediate + intermediate + intermediate * hidden + hidden
        norms = 4 * hidden  # two LayerNorms, weight and bias
    blocks = depth * (attention + mlp + norms)

    merger = _merger_parameters(
        hidden_size=hidden,
        spatial_merge_size=merge,
        out_hidden_size=out_hidden,
        gated_mlp=gated_mlp,
    )

    # Qwen3-VL only: learned position embeddings plus one extra merger per
    # deepstack tap.  Omitting these under-counts the tower by ~21%.
    position_count = vision_config.get("num_position_embeddings")
    pos_embed = hidden * int(position_count) if position_count else 0
    deepstack_indexes = vision_config.get("deepstack_visual_indexes") or []
    deepstack = len(deepstack_indexes) * merger

    total = patch_embed + blocks + merger + pos_embed + deepstack
    return {
        "patch_embed": patch_embed,
        "blocks": blocks,
        "merger": merger,
        "position_embedding": pos_embed,
        "deepstack_mergers": deepstack,
        "total": total,
    }


def visual_tokens_per_image(
    vision_config: Mapping[str, Any],
    *,
    height: int,
    width: int,
) -> dict[str, Any]:
    """Visual tokens a single image contributes after spatial merging.

    This is the quantity that makes VL workload accounting differ from text: it
    is set by resolution and patch geometry, not by the sample's token length.
    """

    patch = _int(vision_config.get("patch_size"), "patch_size")
    merge = _int(vision_config.get("spatial_merge_size"), "spatial_merge_size")
    grid_h = height // patch
    grid_w = width // patch
    patches = grid_h * grid_w
    tokens = patches // (merge * merge)
    return {
        "input_height": height,
        "input_width": width,
        "patch_size": patch,
        "spatial_merge_size": merge,
        "grid_height": grid_h,
        "grid_width": grid_w,
        "patch_count": patches,
        "visual_tokens": tokens,
    }


def _checkpoint_vision_parameters(model_dir: Path) -> dict[str, Any]:
    """Count vision parameters actually present in the checkpoint, if readable.

    Verifying the analytic formula against the real tensors is what separates a
    trustworthy structural term from a plausible guess, so a mismatch is
    reported rather than smoothed over.
    """

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return {"available": False, "reason": "safetensors_index_missing"}
    try:
        from safetensors import safe_open
    except ImportError:
        return {"available": False, "reason": "safetensors_unavailable"}

    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    vision_keys = [key for key in weight_map if ".visual." in key or key.startswith("visual.")]
    if not vision_keys:
        return {"available": False, "reason": "no_vision_tensors_in_checkpoint"}

    by_file: dict[str, list[str]] = {}
    for key in vision_keys:
        by_file.setdefault(weight_map[key], []).append(key)

    total = 0
    components: dict[str, int] = {}
    try:
        for filename, keys in by_file.items():
            with safe_open(str(model_dir / filename), framework="pt") as handle:
                for key in keys:
                    count = 1
                    for dim in handle.get_slice(key).get_shape():
                        count *= dim
                    total += count
                    parts = key.split(".")
                    marker = parts[parts.index("visual") + 1] if "visual" in parts else parts[0]
                    components[marker] = components.get(marker, 0) + count
    except Exception as error:  # pragma: no cover - checkpoint read is best effort
        return {"available": False, "reason": f"checkpoint_read_failed: {error}"}

    return {
        "available": True,
        "tensor_count": len(vision_keys),
        "parameters": total,
        "components": dict(sorted(components.items())),
    }


def build_vision_structure(
    model_dir: Path,
    *,
    model_id: str | None = None,
    verify_against_checkpoint: bool = True,
    relative_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Derive the static vision-tower structure for one VL checkpoint."""

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise VisionStructureError(f"config.json not found under {model_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vision_config = config.get("vision_config")
    if not isinstance(vision_config, Mapping):
        raise VisionStructureError(
            "config has no vision_config; this is not a VL checkpoint"
        )

    generation = detect_generation(config)
    if generation not in VERIFIED_GENERATIONS:
        raise VisionStructureError(
            f"vision generation '{generation}' has no verified structural "
            "formula; mark the model unsupported instead of reusing another "
            "generation's vision formula"
        )
    parameters = vision_tower_parameters(vision_config, generation=generation)
    text_config = config.get("text_config") or config

    verification: dict[str, Any] = {"performed": False}
    if verify_against_checkpoint:
        observed = _checkpoint_vision_parameters(model_dir)
        verification = {"performed": True, **observed}
        if observed.get("available"):
            actual = int(observed["parameters"])
            analytic = parameters["total"]
            delta = analytic - actual
            verification.update(
                {
                    "analytic_parameters": analytic,
                    "checkpoint_parameters": actual,
                    "absolute_delta": delta,
                    "relative_delta": delta / actual if actual else None,
                    "matches_within_tolerance": bool(
                        actual and abs(delta) / actual <= relative_tolerance
                    ),
                }
            )

    resident_bytes = parameters["total"] * BYTES_PER_BF16
    report = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "model_id": model_id or model_dir.name,
        "model_directory": str(model_dir),
        "architectures": list(config.get("architectures") or []),
        "model_type": config.get("model_type"),
        "vision_generation": generation,
        "geometry": {
            "vision": {
                key: vision_config.get(key)
                for key in (
                    "depth",
                    "hidden_size",
                    "intermediate_size",
                    "out_hidden_size",
                    "num_heads",
                    "patch_size",
                    "spatial_merge_size",
                    "temporal_patch_size",
                    "num_position_embeddings",
                    "deepstack_visual_indexes",
                    "window_size",
                    "fullatt_block_indexes",
                    "tokens_per_second",
                )
                if key in vision_config
            },
            "language": {
                key: text_config.get(key)
                for key in (
                    "hidden_size",
                    "intermediate_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "vocab_size",
                )
                if key in text_config
            },
        },
        "vision_parameters": parameters,
        "vision_resident_bytes_bf16": resident_bytes,
        "vision_resident_gib_bf16": resident_bytes / 1024**3,
        "checkpoint_verification": verification,
        "media_token_markers": {
            key: config.get(key)
            for key in (
                "image_token_id",
                "video_token_id",
                "vision_start_token_id",
                "vision_end_token_id",
            )
            if key in config
        },
        "modelled": ["resident_vision_weights", "projector_and_merger_weights"],
        "not_modelled": [
            "vision_encoder_activation",
            "multimodal_workspace",
            "zero3_gather_behaviour_for_vision_parameters",
            "image_token_sequence_inflation",
            "dynamic_resolution_distribution",
        ],
        "calibration_status": "structural_only_uncalibrated",
        "guarantees": {
            "predicts_peak_memory": False,
            "admits_candidates": False,
            "explains_observed_vl_memory_gap": False,
            "creates_gpu_queue": False,
        },
    }

    # State the ceiling of this term explicitly, in the same units as the
    # measured gap, so nobody reads a 1 GiB structural term as a 21 GiB fix.
    if generation == "qwen3_vl":
        report["observed_gap_context"] = {
            "reference": "Qwen3-VL-8B LoRA ZeRO-3 GC mbs=1, 2xH800",
            "observed_minus_dense_predicted_bytes": (
                QWEN3_VL_8B_MBS1_OBSERVED_GAP_BYTES
            ),
            "resident_vision_weight_bytes": resident_bytes,
            "fraction_of_gap_explained": (
                resident_bytes / QWEN3_VL_8B_MBS1_OBSERVED_GAP_BYTES
            ),
            "conclusion": (
                "resident vision weights explain only a small fraction of the "
                "measured gap; activation, workspace and gather behaviour "
                "require their own calibration"
            ),
        }

    # The inventory's ``actual_parameters`` is a whole-checkpoint count, so for a
    # VL model it already contains the vision tower.  The dense skeleton derives
    # resident weight bytes from that number, which means the vision weights are
    # ALREADY counted.  Adding this module's total as a new memory term would
    # double-count it.  This block makes that explicit so a downstream basis
    # cannot quietly add the same bytes twice.
    report["double_counting_guard"] = {
        "checkpoint_total_includes_vision_tower": True,
        "safe_to_add_as_new_memory_term": False,
        "correct_use": (
            "Use these component counts to attribute and to reason about "
            "freeze/LoRA scope, not as an additive resident-memory term on top "
            "of a whole-checkpoint parameter count."
        ),
        "language_only_parameters_if_needed": (
            "checkpoint_total minus vision_parameters.total"
        ),
    }
    report["report_sha256"] = sha256_json(report)
    return report


def build_report(
    model_dirs: Sequence[Path], *, verify_against_checkpoint: bool = True
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for model_dir in model_dirs:
        try:
            entries.append(
                build_vision_structure(
                    model_dir, verify_against_checkpoint=verify_against_checkpoint
                )
            )
        except VisionStructureError as error:
            failures.append({"model_directory": str(model_dir), "error": str(error)})
    report = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "structural_only_uncalibrated",
        "model_count": len(entries),
        "models": entries,
        "failures": failures,
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dirs", nargs="+", type=Path)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(
        args.model_dirs, verify_against_checkpoint=not args.no_verify
    )
    if args.output is not None:
        write_json(args.output, report)

    for entry in report["models"]:
        params = entry["vision_parameters"]
        print(f"{entry['model_id']}  [{entry['vision_generation']}]")
        print(
            f"  vision params: {params['total'] / 1e6:.1f} M"
            f"  ({entry['vision_resident_gib_bf16']:.3f} GiB bf16)"
        )
        for key in (
            "patch_embed",
            "blocks",
            "merger",
            "position_embedding",
            "deepstack_mergers",
        ):
            if params[key]:
                print(f"    {key}: {params[key] / 1e6:.1f} M")
        verification = entry["checkpoint_verification"]
        if verification.get("available"):
            print(
                f"  checkpoint: {verification['checkpoint_parameters'] / 1e6:.1f} M"
                f"  delta={verification['relative_delta']:+.4%}"
                f"  match={verification['matches_within_tolerance']}"
            )
        else:
            print(f"  checkpoint: unverified ({verification.get('reason')})")
        context = entry.get("observed_gap_context")
        if context:
            print(
                "  explains "
                f"{context['fraction_of_gap_explained']:.1%} of the measured "
                "VL memory gap"
            )
    for failure in report["failures"]:
        print(f"failed: {failure['model_directory']}: {failure['error']}")
    if args.output is not None:
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
