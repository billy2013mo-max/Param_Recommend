from __future__ import annotations

import json
import struct
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from structural_linear_work import (  # noqa: E402
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    layer_type_counts,
    structural_linear_work,
)

MODEL_ROOT = Path("/wanqing-models")

# Every checkpoint the H800/RTX4090 campaigns have inventoried, with the
# actual/historical-formula ratio measured from tensor shapes on 2026-08-04.
# The ratio is recorded so a silent regression in either direction fails.
KNOWN_MODELS = {
    "Qwen3-0.6B": 1.1538,
    "Qwen3-1.7B": 1.0000,
    "Qwen3-4B": 1.0845,
    "Qwen3-8B": 1.0000,
    "Qwen3-14B": 1.0000,
    "Qwen3-32B": 1.0690,
    "Qwen2.5-14B": 1.0000,
    "Qwen2.5-32B": 1.0000,
    "Qwen3-VL-8B-Instruct": 1.0000,
    "Qwen3.5-4B": 1.2514,
    "Qwen3.6-27B": 1.1519,
}


def _safetensors_shapes(model_dir: Path) -> dict[str, list[int]]:
    """Read tensor shapes from the safetensors headers without torch."""
    shapes: dict[str, list[int]] = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with path.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            shapes[name] = entry["shape"]
    return shapes


def _layer_prefix(shapes: dict[str, list[int]]) -> str:
    if any(key.startswith("model.language_model.layers.") for key in shapes):
        return "model.language_model.layers."
    return "model.layers."


def _observed_layer_elements(shapes: dict[str, list[int]], prefix: str) -> int:
    """Sum 2-D weight elements in the decoder stack.

    A 2-D weight is exactly a per-token GEMM.  Norms and biases are 1-D and the
    depthwise ``conv1d`` is 3-D, so both drop out without an explicit filter.
    Vision-tower and MTP tensors live under different prefixes and are excluded
    by the prefix itself.
    """
    return sum(
        shape[0] * shape[1]
        for name, shape in shapes.items()
        if name.startswith(prefix) and len(shape) == 2 and "embed" not in name
    )


def _historical_formula(geom: dict) -> int:
    """The formula in h800_theory_basis.py:205-209, layer term only."""
    hidden = geom["hidden_size"]
    heads = geom["num_attention_heads"]
    kv_heads = geom.get("num_key_value_heads") or heads
    head_dim = geom.get("head_dim") or hidden // heads
    kv_width = kv_heads * head_dim
    return geom["num_hidden_layers"] * (
        2 * hidden * hidden
        + 2 * hidden * kv_width
        + 3 * hidden * geom["intermediate_size"]
    )


def _text_geometry(config: dict) -> dict:
    if config.get("hidden_size") is not None:
        return config
    return config.get("text_config") or config


def _available_models() -> list[tuple[str, Path]]:
    return [
        (name, MODEL_ROOT / name)
        for name in KNOWN_MODELS
        if (MODEL_ROOT / name / "config.json").is_file()
    ]


class StructuralLinearWorkAgainstCheckpoints(unittest.TestCase):
    """The computed geometry must match what the checkpoint actually stores."""

    def test_at_least_one_checkpoint_is_available(self) -> None:
        # Guards against the whole suite silently skipping on a host where
        # /wanqing-models is not mounted.
        self.assertTrue(
            _available_models(),
            f"no inventoried checkpoint found under {MODEL_ROOT}",
        )

    def test_layer_elements_match_checkpoint_shapes(self) -> None:
        for name, model_dir in _available_models():
            with self.subTest(model=name):
                config = json.loads((model_dir / "config.json").read_text())
                shapes = _safetensors_shapes(model_dir)
                if not shapes:
                    self.skipTest(f"{name} has no safetensors shards")
                prefix = _layer_prefix(shapes)
                observed = _observed_layer_elements(shapes, prefix)
                computed = structural_linear_work(config)[
                    "layer_elements_per_token"
                ]
                # Exact agreement is the intent; the tolerance only absorbs a
                # future checkpoint that stores a fused qkv tensor.
                self.assertAlmostEqual(
                    computed / observed,
                    1.0,
                    delta=0.005,
                    msg=(
                        f"{name}: computed {computed:,} vs checkpoint "
                        f"{observed:,} (ratio {computed / observed:.4f})"
                    ),
                )

    def test_historical_formula_deviation_is_reproduced(self) -> None:
        """Pin the measured bug magnitude per model.

        This is the regression guard: it documents that the historical formula
        is exact for uniform stacks with ``heads * head_dim == hidden`` and
        quantifies how far it drifts elsewhere.
        """
        for name, expected_ratio in KNOWN_MODELS.items():
            model_dir = MODEL_ROOT / name
            if not (model_dir / "config.json").is_file():
                continue
            with self.subTest(model=name):
                config = json.loads((model_dir / "config.json").read_text())
                geom = _text_geometry(config)
                historical = _historical_formula(geom)
                computed = structural_linear_work(config)[
                    "layer_elements_per_token"
                ]
                self.assertAlmostEqual(
                    computed / historical,
                    expected_ratio,
                    delta=0.0005,
                    msg=(
                        f"{name}: ratio {computed / historical:.4f} != "
                        f"recorded {expected_ratio}"
                    ),
                )


class LayerTypeAccounting(unittest.TestCase):
    def test_layer_types_are_counted_from_the_config(self) -> None:
        path = MODEL_ROOT / "Qwen3.5-4B" / "config.json"
        if not path.is_file():
            self.skipTest("Qwen3.5-4B is not available")
        config = json.loads(path.read_text())
        declared = Counter(_text_geometry(config)["layer_types"])
        counts = layer_type_counts(config)
        self.assertEqual(counts[FULL_ATTENTION], declared["full_attention"])
        self.assertEqual(counts[LINEAR_ATTENTION], declared["linear_attention"])

    def test_uniform_stack_reports_no_linear_layers(self) -> None:
        path = MODEL_ROOT / "Qwen3-8B" / "config.json"
        if not path.is_file():
            self.skipTest("Qwen3-8B is not available")
        config = json.loads(path.read_text())
        result = structural_linear_work(config)
        self.assertFalse(result["is_hybrid_attention"])
        self.assertEqual(result["linear_attention_layers"], 0)
        self.assertEqual(result["depthwise_elements_per_token"], 0)
        self.assertEqual(result["full_attention_layer_share"], 1.0)

    def test_hybrid_stack_reports_the_declared_share(self) -> None:
        path = MODEL_ROOT / "Qwen3.5-4B" / "config.json"
        if not path.is_file():
            self.skipTest("Qwen3.5-4B is not available")
        result = structural_linear_work(json.loads(path.read_text()))
        self.assertTrue(result["is_hybrid_attention"])
        self.assertTrue(result["attn_output_gate"])
        self.assertEqual(result["full_attention_layers"], 8)
        self.assertEqual(result["linear_attention_layers"], 24)
        self.assertAlmostEqual(result["full_attention_layer_share"], 0.25)
        # conv1d is depthwise, so it must stay out of the matmul term.
        self.assertGreater(result["depthwise_elements_per_token"], 0)

    def test_layer_types_length_must_match_layer_count(self) -> None:
        with self.assertRaises(ValueError):
            layer_type_counts(
                {
                    "hidden_size": 8,
                    "num_hidden_layers": 4,
                    "layer_types": ["full_attention"] * 3,
                }
            )

    def test_unknown_layer_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            layer_type_counts(
                {
                    "hidden_size": 8,
                    "num_hidden_layers": 2,
                    "layer_types": ["full_attention", "sliding_window"],
                }
            )

    def test_missing_linear_geometry_is_rejected(self) -> None:
        """A hybrid config without delta-net widths must fail, not guess."""
        with self.assertRaises(ValueError):
            structural_linear_work(
                {
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 16,
                    "vocab_size": 100,
                    "layer_types": ["linear_attention", "full_attention"],
                }
            )


class HistoricalEquivalence(unittest.TestCase):
    def test_matches_history_when_widths_align(self) -> None:
        """A uniform stack with heads*head_dim == hidden must be unchanged.

        This is what keeps the refit honest for Qwen3-1.7B/8B/14B: their
        features must not move at all.
        """
        config = {
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
        }
        computed = structural_linear_work(config)
        self.assertEqual(
            computed["layer_elements_per_token"], _historical_formula(config)
        )

    def test_head_term_follows_the_historical_convention(self) -> None:
        config = {
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 2,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
        }
        result = structural_linear_work(config)
        self.assertEqual(result["head_elements_per_token"], 151936 * 4096)
        self.assertEqual(
            result["linear_applications_per_pass"],
            result["layer_elements_per_token"]
            + result["head_elements_per_token"],
        )

    def test_text_config_nesting_is_followed(self) -> None:
        nested = {
            "model_type": "qwen3_5",
            "text_config": {
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
            },
        }
        flat = dict(nested["text_config"])
        self.assertEqual(
            structural_linear_work(nested)["linear_applications_per_pass"],
            structural_linear_work(flat)["linear_applications_per_pass"],
        )


if __name__ == "__main__":
    unittest.main()
