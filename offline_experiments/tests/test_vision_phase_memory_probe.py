from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from metrics_callback import VisionPhaseMemoryProbe


class _TinyVision(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = torch.nn.Linear(2, 2)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.patch_embed(inputs)
        return self.blocks[0](hidden)


class _TinyVL(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _TinyVision()


class VisionPhaseMemoryProbeTest(unittest.TestCase):
    def test_records_visual_forward_without_cuda(self) -> None:
        model = _TinyVL()
        probe = VisionPhaseMemoryProbe.install(model)
        probe.start_step(1)
        model.visual(torch.ones(1, 2))
        row = probe.finish_step(1)

        self.assertEqual(row["module_name"], "visual")
        self.assertEqual(row["forward_calls"], 1)
        self.assertEqual(row["max_allocated_during_vision"], 0)
        self.assertEqual(row["max_reserved_during_vision"], 0)

        summary = probe.summary([{"vision_phase_memory": row}])
        self.assertTrue(summary["all_measured_steps_observed"])
        self.assertEqual(summary["measured_steps_with_visual_forward"], 1)

    def test_text_step_records_no_visual_forward(self) -> None:
        model = _TinyVL()
        probe = VisionPhaseMemoryProbe.install(model)
        probe.start_step(1)
        row = probe.finish_step(1)

        self.assertEqual(row["forward_calls"], 0)
        summary = probe.summary([{"vision_phase_memory": row}])
        self.assertFalse(summary["all_measured_steps_observed"])
        self.assertEqual(summary["measured_steps_with_visual_forward"], 0)

    def test_rejects_model_without_visual_root(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exactly one visual root"):
            VisionPhaseMemoryProbe.install(torch.nn.Linear(2, 2))


if __name__ == "__main__":
    unittest.main()
