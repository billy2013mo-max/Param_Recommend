from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from metrics_callback import capture_runtime_device_attestation, write_runtime_model_manifest  # noqa: E402
from model_structure_manifest import (  # noqa: E402
    build_model_structure_manifest,
    validate_model_structure_manifest,
)


class TinyVL(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = torch.nn.Linear(4, 4)
        self.multi_modal_projector = torch.nn.Linear(4, 4)
        self.language_model = torch.nn.Linear(4, 4)
        self.language_model.requires_grad_(False)
        self.language_model.lora_A = torch.nn.Parameter(torch.zeros(2, 4))
        self.language_model.lora_B = torch.nn.Parameter(torch.zeros(4, 2))
        self.visual.requires_grad_(False)
        self.multi_modal_projector.requires_grad_(False)


class ModelStructureManifestTests(unittest.TestCase):
    def test_component_freeze_and_visual_lora_hits_are_observed_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                device = capture_runtime_device_attestation(0)
            runtime_path = root / "runtime.json"
            runtime = write_runtime_model_manifest(
                TinyVL(),
                runtime_path,
                rank=0,
                local_rank=0,
                world_size=1,
                job_id="vl-job",
                execution_attempt_id="a" * 20,
                training_mode="lora",
                device_attestation=device,
            )
            result = build_model_structure_manifest(
                runtime,
                job_metadata={
                    "model_id": "tiny-vl",
                    "freeze_vision_tower": True,
                    "freeze_multi_modal_projector": True,
                    # LoRA adapters make this component trainable even though
                    # its base tensors remain frozen.
                    "freeze_language_model": False,
                },
                declared_model={
                    "architecture_role": "vision_language",
                    "is_vision_language": True,
                },
                runtime_manifest_path=runtime_path,
                runtime_media_evidence={
                    "real_image_path_observed": True,
                    "source_image_count": 1,
                    "media_batches": 1,
                    "image_grid_rows": 1,
                    "pixel_value_elements": 16,
                },
            )
            validate_model_structure_manifest(result)

        self.assertTrue(result["visual_path_observed"])
        self.assertTrue(result["vision_parameters_observed"])
        self.assertEqual(result["declaration_status"], "matched")
        self.assertTrue(result["freeze_flags"]["freeze_vision_tower"]["value"])
        self.assertTrue(result["freeze_flags"]["freeze_multi_modal_projector"]["value"])
        language_flag = result["freeze_flags"]["freeze_language_model"]
        self.assertTrue(language_flag["value"])
        self.assertEqual(language_flag["status"], "base_frozen_adapter_trainable")
        self.assertFalse(result["lora_target_hits"]["any_visual_component"])

    def test_declared_flag_mismatch_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                device = capture_runtime_device_attestation(0)
            runtime = write_runtime_model_manifest(
                TinyVL(),
                root / "runtime.json",
                rank=0,
                local_rank=0,
                world_size=1,
                job_id="vl-job",
                execution_attempt_id="b" * 20,
                training_mode="lora",
                device_attestation=device,
            )
            result = build_model_structure_manifest(
                runtime,
                job_metadata={"freeze_vision_tower": False},
            )
        self.assertEqual(result["declaration_status"], "mismatch")
        self.assertEqual(result["declaration_mismatches"], ["freeze_vision_tower"])


if __name__ == "__main__":
    unittest.main()
