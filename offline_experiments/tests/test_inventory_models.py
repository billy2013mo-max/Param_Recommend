from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from inventory_models import (  # noqa: E402
    _is_vision_language_config,
    _tensor_component,
    _vision_geometry,
)


class InventoryVisionTests(unittest.TestCase):
    def test_normalizes_vision_geometry_without_inventing_missing_values(self) -> None:
        config = {
            "model_type": "qwen3_vl",
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "vision_config": {
                "model_type": "qwen3_vl",
                "depth": 27,
                "hidden_size": 1152,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
        }
        self.assertTrue(_is_vision_language_config(config))
        geometry = _vision_geometry(config)
        self.assertIsNotNone(geometry)
        assert geometry is not None
        self.assertEqual(geometry["depth"], 27)
        self.assertEqual(geometry["patch_size"], 16)
        self.assertIsNone(geometry["out_hidden_size"])

    def test_component_classifier_separates_visual_mergers(self) -> None:
        self.assertEqual(
            _tensor_component("model.visual.blocks.0.attn.qkv.weight"),
            "vision_tower",
        )
        self.assertEqual(
            _tensor_component("model.visual.merger.linear_fc1.weight"),
            "projector_or_merger",
        )
        self.assertEqual(
            _tensor_component("model.language_model.layers.0.mlp.down_proj.weight"),
            "language_model_or_other",
        )

    def test_dense_config_has_no_vision_role(self) -> None:
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 4096,
        }
        self.assertFalse(_is_vision_language_config(config))
        self.assertIsNone(_vision_geometry(config))


if __name__ == "__main__":
    unittest.main()

