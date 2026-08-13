#!/usr/bin/env python3
"""Regression tests for projector+language LoRA target discovery."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_entry import (  # noqa: E402
    _projector_discovery_compat_required,
    _projector_linear_module_suffixes,
    install_projector_lora_discovery_compat,
)


class Linear:
    pass


class EmbeddingLinear:
    pass


class FakeModel:
    config = SimpleNamespace(model_type="qwen3_5")

    def named_modules(self):
        return [
            ("model.language_model.layers.0.q_proj", Linear()),
            ("model.visual.blocks.0.qkv", Linear()),
            ("model.visual.merger.linear_fc1", Linear()),
            ("model.visual.merger.linear_fc2", Linear()),
            ("model.visual.merger.embed", EmbeddingLinear()),
        ]


class WrappedQwen3VLModel:
    config = SimpleNamespace(model_type="qwen3_vl")

    def named_modules(self):
        return [
            ("model.visual.merger.linear_fc1", Linear()),
            ("model.visual.merger.linear_fc2", Linear()),
            ("model.not_visual.merger.wrong", Linear()),
        ]


@dataclass
class Composite:
    projector_keys: list[str]


class TrainEntryProjectorLoraCompatTest(unittest.TestCase):
    def test_patch_is_limited_to_exact_projector_language_scope(self) -> None:
        base = {
            "finetuning_type": "lora",
            "lora_target": "all",
            "freeze_vision_tower": True,
            "freeze_multi_modal_projector": False,
        }
        self.assertTrue(_projector_discovery_compat_required(base))
        for key, value in (
            ("finetuning_type", "full"),
            ("lora_target", "q_proj"),
            ("freeze_vision_tower", False),
            ("freeze_multi_modal_projector", True),
        ):
            changed = dict(base)
            changed[key] = value
            self.assertFalse(_projector_discovery_compat_required(changed))

    def test_patch_adds_only_registered_projector_linear_suffixes(self) -> None:
        adapter = SimpleNamespace(
            find_all_linear_modules=lambda model, freeze_vision_tower: ["q_proj"]
        )
        config = {
            "finetuning_type": "lora",
            "lora_target": ["all"],
            "freeze_vision_tower": True,
            "freeze_multi_modal_projector": False,
        }
        installed = install_projector_lora_discovery_compat(
            config,
            adapter_module=adapter,
            composite_models={
                "qwen3_5": Composite(projector_keys=["model.visual.merger"])
            },
        )
        self.assertTrue(installed)
        self.assertEqual(
            adapter.find_all_linear_modules(FakeModel(), True),
            ["linear_fc1", "linear_fc2", "q_proj"],
        )

    def test_patch_fails_closed_for_unregistered_model(self) -> None:
        adapter = SimpleNamespace(
            find_all_linear_modules=lambda model, freeze_vision_tower: ["q_proj"]
        )
        config = {
            "finetuning_type": "lora",
            "lora_target": "all",
            "freeze_vision_tower": True,
            "freeze_multi_modal_projector": False,
        }
        install_projector_lora_discovery_compat(
            config, adapter_module=adapter, composite_models={}
        )
        with self.assertRaisesRegex(RuntimeError, "registered composite model"):
            adapter.find_all_linear_modules(FakeModel(), True)

    def test_registered_path_allows_only_dot_delimited_wrapper_prefix(self) -> None:
        self.assertEqual(
            _projector_linear_module_suffixes(
                WrappedQwen3VLModel(), ["visual.merger"]
            ),
            {"linear_fc1", "linear_fc2"},
        )


if __name__ == "__main__":
    unittest.main()
