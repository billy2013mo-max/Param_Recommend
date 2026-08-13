#!/usr/bin/env python3
"""Small, separate PyTorch Profiler callback for analytic FLOP calibration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback


class CalibrationProfilerCallback(TrainerCallback):
    def __init__(
        self,
        output_dir: Path,
        wait_steps: int = 0,
        warmup_steps: int = 1,
        active_steps: int = 1,
    ):
        self.rank = int(os.environ.get("RANK", "0"))
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wait_steps = max(0, int(wait_steps))
        self.warmup_steps = max(1, int(warmup_steps))
        self.active_steps = max(1, int(active_steps))
        self.profiler = None

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        # A repaired rerun must not retain a stale multi-gigabyte trace emitted
        # by an older callback revision.
        (self.output_dir / f"trace.rank{self.rank}.json").unlink(missing_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self.profiler = torch.profiler.profile(
            activities=activities,
            # Wait through the early training window, warm the profiler for one
            # optimizer step, then record only the final step.  The old 2+2+3
            # schedule never reached its active window in a four-step probe.
            schedule=torch.profiler.schedule(
                wait=self.wait_steps,
                warmup=self.warmup_steps,
                active=self.active_steps,
                repeat=1,
            ),
            # Calibration consumes the compact key-averages table below.
            # Exporting a Chrome trace for these model-sized steps creates
            # multi-gigabyte files and adds minutes of CPU post-processing
            # without contributing to the FLOP fit.
            record_shapes=True,
            profile_memory=False,
            with_flops=True,
            with_stack=False,
        )
        self.profiler.__enter__()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.profiler is not None:
            self.profiler.step()

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.profiler is None:
            return
        self.profiler.__exit__(None, None, None)
        rows = []
        for event in self.profiler.key_averages():
            rows.append(
                {
                    "key": event.key,
                    "count": event.count,
                    "cpu_time_total_us": event.cpu_time_total,
                    "device_time_total_us": event.device_time_total,
                    "self_device_time_total_us": event.self_device_time_total,
                    "flops": event.flops,
                }
            )
        rows.sort(key=lambda row: row["device_time_total_us"], reverse=True)
        (self.output_dir / f"operators.rank{self.rank}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
