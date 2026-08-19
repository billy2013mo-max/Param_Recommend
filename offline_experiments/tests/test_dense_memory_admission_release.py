"""Tests for the bounded dense text memory-admission production release."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import dense_memory_admission_release as dmar  # noqa: E402


def _dense_request(**overrides):
    request = {
        "model_id": "qwen3_8b",
        "hardware_id": "h800",
        "modality": "text",
        "stage": "sft",
        "dtype": "bf16",
        "training_mode": "lora",
        "lora_rank": 32,
        "cutoff_len": 4096,
        "gpu_count": 2,
        "target_gbs": 64,
        "packing": True,
    }
    request.update(overrides)
    return request


class LoadReleaseTest(unittest.TestCase):
    def setUp(self):
        self.release = dmar.load_dense_memory_admission_release()

    def test_contract_is_active_and_bounded(self):
        self.assertEqual(self.release["status"], "active_limited_production")
        self.assertIs(self.release["automatic_execution_allowed"], True)
        self.assertIs(self.release["fail_closed"], True)
        self.assertIs(self.release["vl_allowed"], False)

    def test_scope_is_dense_only(self):
        self.assertEqual(
            self.release["scope"]["architecture_routes"], ["dense_full_attention"]
        )
        for model_id in self.release["scope"]["model_ids"]:
            self.assertTrue(model_id.startswith("qwen3_"))
            self.assertFalse(model_id.startswith("qwen3p5_"))
            self.assertNotEqual(model_id, "qwen3_6_27b")


class ScopeAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.release = dmar.load_dense_memory_admission_release()

    def test_in_scope_dense_lora_admitted(self):
        self.assertEqual(dmar.scope_mismatches(_dense_request(), self.release), [])

    def test_in_scope_dense_full_boundary_admitted(self):
        request = _dense_request(
            model_id="qwen3_32b",
            training_mode="full",
            cutoff_len=512,
            gpu_count=4,
            target_gbs=256,
            packing=False,
        )
        request.pop("lora_rank", None)
        self.assertEqual(dmar.scope_mismatches(request, self.release), [])

    def test_hybrid_family_fails_closed(self):
        for model_id in ("qwen3p5_4b", "qwen3p5_9b", "qwen3_6_27b"):
            request = _dense_request(model_id=model_id)
            self.assertEqual(
                dmar.scope_mismatches(request, self.release),
                ["model_family_excluded"],
            )

    def test_vl_request_rejected(self):
        request = _dense_request(vl_workload_profile_path="/tmp/profile.json")
        self.assertIn("vl_workload_profile_path", dmar.scope_mismatches(request, self.release))

    def test_out_of_scope_scalars_rejected(self):
        self.assertIn(
            "cutoff_len",
            dmar.scope_mismatches(_dense_request(cutoff_len=999), self.release),
        )
        self.assertIn(
            "gpu_count",
            dmar.scope_mismatches(_dense_request(gpu_count=8), self.release),
        )
        self.assertIn(
            "offload",
            dmar.scope_mismatches(_dense_request(offload=True), self.release),
        )

    def test_bad_lora_rank_rejected(self):
        self.assertIn(
            "lora_rank",
            dmar.scope_mismatches(_dense_request(lora_rank=16), self.release),
        )


if __name__ == "__main__":
    unittest.main()
