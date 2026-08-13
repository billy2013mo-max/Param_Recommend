from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from hybrid_attention_memory_features import (
    architecture_signature,
    build_dense_hybrid_features,
    lora_adapter_parameter_elements,
)


def _write_config(config: dict) -> tuple[tempfile.TemporaryDirectory, Path]:
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return temporary, path


def _full_config() -> dict:
    return {
        "model_type": "qwen3",
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "vocab_size": 32000,
    }


def _hybrid_config() -> dict:
    return {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 32000,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "full_attention_interval": 4,
            "attn_output_gate": True,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 8,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
        },
    }


class ArchitectureSignatureTests(unittest.TestCase):
    def test_implicit_dense_stack_is_all_full_attention(self) -> None:
        temporary, path = _write_config(_full_config())
        self.addCleanup(temporary.cleanup)
        signature = architecture_signature({"path": str(path)})
        self.assertEqual(signature["architecture_route"], "dense_full_attention")
        self.assertEqual(signature["num_full_attention_layers"], 4)
        self.assertEqual(signature["num_linear_attention_layers"], 0)
        self.assertEqual(signature["layer_pattern_source"], "implicit_all_full_attention")

    def test_explicit_hybrid_stack_keeps_layer_counts_and_state_dtype(self) -> None:
        temporary, path = _write_config(_hybrid_config())
        self.addCleanup(temporary.cleanup)
        signature = architecture_signature({"path": str(path)})
        self.assertEqual(signature["architecture_route"], "dense_hybrid_attention")
        self.assertEqual(signature["num_full_attention_layers"], 1)
        self.assertEqual(signature["num_linear_attention_layers"], 3)
        self.assertEqual(signature["linear_state_dtype_bytes"], 4)
        self.assertEqual(signature["max_consecutive_linear_attention_layers"], 3)

    def test_moe_config_is_a_hard_route_error(self) -> None:
        config = _full_config()
        config["num_experts"] = 8
        temporary, path = _write_config(config)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ValueError, "does not accept MoE"):
            architecture_signature({"path": str(path)})


class LoRAAndMemoryFeatureTests(unittest.TestCase):
    def test_full_attention_lora_count_specializes_to_historical_formula(self) -> None:
        temporary, path = _write_config(_full_config())
        self.addCleanup(temporary.cleanup)
        signature = architecture_signature({"path": str(path)})
        actual = lora_adapter_parameter_elements(signature, 32)["total_elements"]
        expected = 32 * 4 * (9 * 512 + 2 * 128 + 3 * 1024)
        self.assertEqual(actual, expected)

    def test_hybrid_checkpoint_counts_match_known_runtime_manifests(self) -> None:
        known = {
            "/wanqing-models/Qwen3.5-4B": 64_929_792,
            "/wanqing-models/Qwen3.5-9B": 86_556_672,
            "/wanqing-models/Qwen3.6-27B": 233_455_616,
        }
        available = 0
        for model_path, expected in known.items():
            if not (Path(model_path) / "config.json").is_file():
                continue
            available += 1
            with self.subTest(model=model_path):
                signature = architecture_signature({"path": model_path})
                actual = lora_adapter_parameter_elements(signature, 32)[
                    "total_elements"
                ]
                self.assertEqual(actual, expected)
        if not available:
            self.skipTest("known hybrid checkpoints are not mounted")

    def test_operator_workspaces_use_peak_not_sum(self) -> None:
        temporary, path = _write_config(_hybrid_config())
        self.addCleanup(temporary.cleanup)
        job = {
            "model_parameters": 100_000_000,
            "train_type": "lora",
            "gpu_count": 2,
            "zero": "zero2",
            "gc": False,
            "mbs": 1,
            "cutoff_len": 2048,
        }
        features = build_dense_hybrid_features(
            job,
            {"path": str(path), "actual_parameters": 100_000_000},
            {"rank": 32, "target": "all"},
            150_142_189_568,
        )
        memory = features["memory"]
        candidates = memory["workspace_candidates"]
        expected = max(
            candidates["full_attention_workspace_bytes"],
            candidates["linear_attention_workspace_bytes"],
            candidates["logits_workspace_bytes"],
            candidates["zero_collective_workspace_bytes"],
        )
        self.assertEqual(memory["peak_workspace_bytes"], expected)
        self.assertEqual(memory["workspace_aggregation"], "max_not_sum")


if __name__ == "__main__":
    unittest.main()
