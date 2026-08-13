#!/usr/bin/env python3
"""Instrumented worker entry point for one concrete LLaMA-Factory run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from metrics_callback import ExperimentCallback, install_collator_instrumentation


def _qwen35_rope_video_grid(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    video_grid_thw: torch.Tensor,
    video_token_id: int,
) -> torch.Tensor:
    """Match Qwen3.5 RoPE grids to Qwen3-VL's frame-separated token groups.

    LLaMA-Factory's Qwen3-VL plugin emits one timestamp-separated vision group
    per sampled frame, while its processor returns one ``[T, H, W]`` grid row
    per source video.  Transformers 5.3 Qwen3.5 consumes one grid row per
    contiguous video-token group, so only the RoPE call needs ``T`` rows of
    ``[1, H, W]``.  The original grid remains unchanged for the model forward.
    """

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise RuntimeError("Qwen3.5 RoPE compatibility requires aligned 2D token tensors")
    if video_grid_thw.ndim != 2 or video_grid_thw.shape[1] != 3:
        raise RuntimeError("Qwen3.5 RoPE compatibility requires video_grid_thw[N,3]")

    video_mask = (input_ids == int(video_token_id)) & attention_mask.bool()
    previous = torch.zeros_like(video_mask)
    previous[:, 1:] = video_mask[:, :-1]
    group_count = int((video_mask & ~previous).sum().item())
    grid_count = int(video_grid_thw.shape[0])
    if group_count == grid_count:
        return video_grid_thw

    temporal_counts = [int(row[0].item()) for row in video_grid_thw]
    if (
        group_count <= 0
        or any(count <= 0 for count in temporal_counts)
        or group_count != sum(temporal_counts)
    ):
        raise RuntimeError(
            "Qwen3.5 video token/grid groups are incompatible: "
            f"token_groups={group_count}, grid_rows={grid_count}, "
            f"temporal_counts={temporal_counts}"
        )

    expanded = []
    for row, temporal_count in zip(video_grid_thw, temporal_counts, strict=True):
        for _ in range(temporal_count):
            frame_grid = row.clone()
            frame_grid[0] = 1
            expanded.append(frame_grid)
    return torch.stack(expanded, dim=0)


def install_qwen35_video_rope_compat(*, collator_class: Any | None = None) -> bool:
    """Adapt only Qwen3.5 RoPE inputs without changing model-forward media."""

    if collator_class is None:
        from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq

        collator_class = MultiModalDataCollatorForSeq2Seq
    original = collator_class._compute_rope_position_ids
    if getattr(original, "_qwen35_video_rope_compat_v1", False):
        return True

    def compute_qwen35_compatible_rope(self: Any, features: dict[str, Any], mm_inputs: dict[str, Any]) -> None:
        model_type = getattr(getattr(self, "model", None), "config", None)
        model_type = getattr(model_type, "model_type", None)
        video_grid = mm_inputs.get("video_grid_thw")
        video_token_id = getattr(getattr(self.model, "config", None), "video_token_id", None)
        if model_type == "qwen3_5" and torch.is_tensor(video_grid) and video_token_id is not None:
            rope_grid = _qwen35_rope_video_grid(
                features["input_ids"],
                features["attention_mask"],
                video_grid,
                int(video_token_id),
            )
            if rope_grid is not video_grid:
                mm_inputs = dict(mm_inputs)
                mm_inputs["video_grid_thw"] = rope_grid
        return original(self, features, mm_inputs)

    compute_qwen35_compatible_rope._qwen35_video_rope_compat_v1 = True  # type: ignore[attr-defined]
    collator_class._compute_rope_position_ids = compute_qwen35_compatible_rope
    return True


def _uses_all_lora_targets(config: dict[str, Any]) -> bool:
    """Return whether the runtime config asks LLaMA-Factory to discover all LoRA layers."""

    target = config.get("lora_target")
    return target == "all" or target == ["all"]


def _projector_discovery_compat_required(config: dict[str, Any]) -> bool:
    """Limit the compatibility repair to projector+language LoRA runs."""

    return bool(
        config.get("finetuning_type") == "lora"
        and _uses_all_lora_targets(config)
        and config.get("freeze_vision_tower") is True
        and config.get("freeze_multi_modal_projector") is False
    )


def _projector_linear_module_suffixes(
    model: Any, projector_keys: list[str]
) -> set[str]:
    """Discover linear suffixes below registered paths at module boundaries.

    Some Transformers wrappers expose ``model.visual.merger`` while the
    LLaMA-Factory composite registry declares ``visual.merger``.  Matching in
    a dot-padded module path accepts that wrapper prefix without accepting a
    partial component such as ``not_visual.merger``.
    """

    suffixes: set[str] = set()
    for name, module in model.named_modules():
        inside_projector = any(
            f".{projector_key.strip('.')}." in f".{name.strip('.')}."
            for projector_key in projector_keys
        )
        if (
            inside_projector
            and "Linear" in module.__class__.__name__
            and "Embedding" not in module.__class__.__name__
        ):
            suffixes.add(name.rsplit(".", 1)[-1])
    return suffixes


def install_projector_lora_discovery_compat(
    config: dict[str, Any],
    *,
    adapter_module: Any | None = None,
    composite_models: dict[str, Any] | None = None,
) -> bool:
    """Repair LLaMA-Factory's ``all`` discovery for trainable projectors.

    The installed LLaMA-Factory version always excludes composite-model
    projectors in ``find_all_linear_modules``.  That is correct when a
    projector is frozen, but it silently turns the declared
    projector+language scope into language-only LoRA when the vision tower is
    frozen.  Keep the upstream discovery result and add only linear suffixes
    found below the model registry's exact projector paths.  The subsequent
    upstream ``patch_target_modules`` call still enforces all declared freeze
    flags and conflict keys.
    """

    if not _projector_discovery_compat_required(config):
        return False
    if adapter_module is None or composite_models is None:
        from llamafactory.model import adapter as adapter_module
        from llamafactory.model.model_utils.visual import COMPOSITE_MODELS

        composite_models = COMPOSITE_MODELS

    original = adapter_module.find_all_linear_modules
    if getattr(original, "_projector_lora_discovery_compat_v1", False):
        return True

    def find_all_linear_modules_with_trainable_projector(
        model: Any, freeze_vision_tower: bool
    ) -> list[str]:
        target_modules = set(original(model, freeze_vision_tower))
        model_type = getattr(model.config, "model_type", None)
        composite = composite_models.get(model_type)
        if composite is None:
            raise RuntimeError(
                "Projector+language LoRA discovery requires a registered "
                f"composite model, got model_type={model_type!r}"
            )
        projector_modules = _projector_linear_module_suffixes(
            model, list(composite.projector_keys)
        )
        if not projector_modules:
            raise RuntimeError(
                "Projector+language LoRA discovery found no linear module below "
                f"the registered projector paths {list(composite.projector_keys)!r}"
            )
        target_modules.update(projector_modules)
        return sorted(target_modules)

    find_all_linear_modules_with_trainable_projector._projector_lora_discovery_compat_v1 = True  # type: ignore[attr-defined]
    adapter_module.find_all_linear_modules = (
        find_all_linear_modules_with_trainable_projector
    )
    return True


def disable_final_weight_save() -> None:
    """Benchmark outputs must not duplicate multi-GB checkpoints after every probe."""
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

    def benchmark_save_model(self, output_dir=None, _internal_call=False):  # type: ignore[no-untyped-def]
        destination = Path(output_dir or self.args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        if self.is_world_process_zero():
            (destination / "BENCHMARK_NO_MODEL_SAVE").write_text(
                "Final weight serialization is disabled for offline efficiency probes.\n",
                encoding="utf-8",
            )

    CustomSeq2SeqTrainer.save_model = benchmark_save_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--job-metadata", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    metadata = json.loads(args.job_metadata.read_text(encoding="utf-8"))
    warmup_steps = int(metadata.get("warmup_steps", 0))
    install_projector_lora_discovery_compat(config)
    install_qwen35_video_rope_compat()
    install_collator_instrumentation()
    disable_final_weight_save()
    callback = ExperimentCallback(args.metrics_dir, warmup_steps=warmup_steps, job_metadata=metadata)
    callbacks = [callback]
    if metadata.get("enable_profiler"):
        from profiler_callback import CalibrationProfilerCallback

        profiler_active_steps = int(metadata.get("profiler_active_steps", 1))
        profiler_warmup_steps = 1
        total_steps = warmup_steps + int(metadata.get("measure_steps", 0))
        callbacks.append(
            CalibrationProfilerCallback(
                args.metrics_dir / "profiler",
                wait_steps=max(
                    0,
                    total_steps - profiler_warmup_steps - profiler_active_steps,
                ),
                warmup_steps=profiler_warmup_steps,
                active_steps=profiler_active_steps,
            )
        )

    # The venv's sitecustomize and launcher-side integrations assume this root.
    os.chdir("/fine-tuning-launcher")
    from llamafactory.train.tuner import run_exp

    try:
        run_exp(args=config, callbacks=callbacks)
    except BaseException as error:
        callback.record_failure(error)
        callback.finalize_after_failure()
        raise


if __name__ == "__main__":
    main()
