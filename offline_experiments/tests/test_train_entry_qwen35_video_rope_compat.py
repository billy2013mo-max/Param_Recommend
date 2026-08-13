#!/usr/bin/env python3
"""Regression tests for Qwen3.5 frame-separated video RoPE grids."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_entry import _qwen35_rope_video_grid  # noqa: E402


class Qwen35VideoRopeCompatTests(unittest.TestCase):
    def test_frame_separated_groups_expand_one_video_grid(self) -> None:
        video_token_id = 9
        input_ids = torch.tensor([[1, 9, 9, 2, 9, 9, 3, 9, 9]])
        attention_mask = torch.ones_like(input_ids)
        grid = torch.tensor([[3, 12, 20]])

        expanded = _qwen35_rope_video_grid(
            input_ids, attention_mask, grid, video_token_id
        )

        self.assertEqual(expanded.tolist(), [[1, 12, 20]] * 3)
        self.assertEqual(grid.tolist(), [[3, 12, 20]])

    def test_contiguous_video_group_keeps_original_grid(self) -> None:
        input_ids = torch.tensor([[1, 9, 9, 9, 2]])
        attention_mask = torch.ones_like(input_ids)
        grid = torch.tensor([[3, 12, 20]])

        observed = _qwen35_rope_video_grid(input_ids, attention_mask, grid, 9)

        self.assertIs(observed, grid)

    def test_unexplained_group_mismatch_fails_closed(self) -> None:
        input_ids = torch.tensor([[1, 9, 2, 9, 3]])
        attention_mask = torch.ones_like(input_ids)
        grid = torch.tensor([[3, 12, 20]])

        with self.assertRaisesRegex(RuntimeError, "token/grid groups"):
            _qwen35_rope_video_grid(input_ids, attention_mask, grid, 9)


if __name__ == "__main__":
    unittest.main()
