#!/usr/bin/env python3
"""Small distributed runtime smoke test; does not load or train a model."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    value = torch.tensor(float(rank + 1), device="cuda")
    dist.all_reduce(value)
    expected = world_size * (world_size + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"NCCL all-reduce mismatch: got {value.item()}, expected {expected}")

    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "backend": dist.get_backend(),
                    "world_size": world_size,
                    "all_reduce": value.item(),
                    "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "nccl": list(torch.cuda.nccl.version()),
                },
                ensure_ascii=False,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
