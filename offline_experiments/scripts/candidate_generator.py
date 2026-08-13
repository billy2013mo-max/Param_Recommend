#!/usr/bin/env python
"""Generate the legal candidate set for one user training scenario.

Today the product side must hand the predictor a candidate list: the H800
predictor is inference-only and reads a ``candidates`` array.  Every existing
producer of that array is either a checked-in example or a hard-coded campaign
table, so an arbitrary user request has no way to obtain candidates.  This module
closes that gap.

It is deliberately *only* a generator.  It answers "which configurations are even
legal and worth pricing", and it never answers "which one is safe" or "which one
is fastest" -- those stay with the frozen memory gate and the ranking head, in
that order.  Concretely it will not:

* predict memory or admit a candidate,
* rank candidates or compare throughput,
* decide the GPU count (the scale-out policy owns that),
* enable packing on its own (the frozen static policy owns that decision).

The constraint math is reused from the experiment-matrix path rather than
reinvented: ``gpu_count * mbs`` must divide ``target_gbs`` for the unpacked case,
and an analytic optimizer-state lower bound prunes placements that cannot fit
before any model is consulted.  Pruning here is *static and conservative*: it may
only remove candidates that are impossible on arithmetic grounds, never
candidates that merely look slow or risky.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from common import read_json, sha256_json, write_json

SCHEMA = "sft_candidate_generation/v1"
IMPLEMENTATION_VERSION = "sft_candidate_generator_impl/2026-08-01.v1"

# Mirrors the frozen predictor's supported domain.  A candidate outside these
# sets is still emitted when explicitly requested, but is tagged so the caller
# cannot mistake it for an in-domain configuration.
SUPPORTED_GPU_COUNTS = (1, 2, 4)
SUPPORTED_TARGET_GBS = (32, 64, 128)
SUPPORTED_CUTOFF_LENS = (512, 2048, 4096, 8192, 32768)
SUPPORTED_MBS = (1, 2, 4, 8, 16)

# Cutoff-tiered micro-batch ladders, same shape as the matrix generator: long
# sequences make large micro batches pointless because activation dominates.
MBS_LADDER: tuple[tuple[int, tuple[int, ...]], ...] = (
    (512, (1, 2, 4, 8, 16, 32)),
    (2048, (1, 2, 4, 8, 16)),
    (4096, (1, 2, 4, 8)),
    (8192, (1, 2, 4)),
    (math.inf, (1, 2)),
)

BYTES_PER_BF16 = 2
# fp32 master weight + fp32 Adam exp_avg + fp32 exp_avg_sq per trainable element.
OPTIMIZER_BYTES_PER_TRAINABLE = 12


def _mbs_ladder(cutoff_len: int) -> tuple[int, ...]:
    for limit, ladder in MBS_LADDER:
        if cutoff_len <= limit:
            return ladder
    return MBS_LADDER[-1][1]


def _zero_options(gpu_count: int, *, include_zero1: bool) -> tuple[int, ...]:
    """Legal ZeRO stages for a card count.

    Single GPU has no data-parallel group to shard across, so stage 0 is the only
    legal choice; multi-GPU must shard.  ZeRO-1 is legal but carries limited
    evidence, so it is opt-in.
    """

    if gpu_count == 1:
        return (0,)
    return (1, 2, 3) if include_zero1 else (2, 3)


def _trainable_parameters(
    total_parameters: int, train_type: str, *, lora_fraction: float
) -> int:
    if train_type == "full":
        return total_parameters
    return max(1, int(total_parameters * lora_fraction))


def optimizer_state_lower_bound_bytes(
    *,
    total_parameters: int,
    train_type: str,
    gpu_count: int,
    zero_stage: int,
    lora_fraction: float,
) -> int:
    """Per-GPU lower bound on resident model state.

    This is a *lower* bound on purpose: it counts only weights plus sharded
    optimizer state, and deliberately omits activation, workspace, fragmentation
    and communication buffers.  A candidate whose lower bound already exceeds
    capacity cannot possibly fit, so pruning it is sound; a candidate that passes
    is *not* thereby declared safe.

    ZeRO-2 shards gradients and optimizer state but leaves the weights
    replicated on every rank.  For LoRA that means adding cards saves almost
    nothing, because the trainable set is only the adapter while the full base
    weights stay resident -- which is exactly why extra cards do not
    automatically loosen the memory limit on that path.
    """

    trainable = _trainable_parameters(
        total_parameters, train_type, lora_fraction=lora_fraction
    )
    weights = total_parameters * BYTES_PER_BF16
    optimizer = trainable * OPTIMIZER_BYTES_PER_TRAINABLE
    if zero_stage >= 3:
        # Parameters are also sharded, so weights shrink with the card count.
        return weights // gpu_count + optimizer // gpu_count
    if zero_stage >= 1:
        return weights + optimizer // gpu_count
    return weights + optimizer


def _derive_gradient_accumulation(
    *, target_gbs: int, gpu_count: int, physical_mbs: int
) -> int | None:
    """GA for the unpacked case, or None when the GBS contract cannot be met."""

    per_step = gpu_count * physical_mbs
    if per_step <= 0 or per_step > target_gbs:
        return None
    if target_gbs % per_step != 0:
        return None
    return target_gbs // per_step


def _request_field(request: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in request and request[name] is not None:
            return request[name]
    return default


def generate_candidates(
    request: Mapping[str, Any],
    *,
    capacity_bytes: int,
    static_prune_fraction: float = 0.95,
    include_zero1: bool = False,
    gradient_checkpointing_options: Sequence[bool] = (False, True),
    gpu_counts: Sequence[int] | None = None,
    lora_fraction: float = 0.01,
    restrict_to_supported_mbs: bool = True,
) -> dict[str, Any]:
    """Build every legal candidate for one fixed user scenario.

    The scenario fields -- model, train type, dataset profile, target GBS, cutoff
    and packing -- are held fixed, because they are the user's training
    semantics.  Only the resource and execution knobs are searched.

    ``restrict_to_supported_mbs`` keeps the ladder inside the micro-batch set the
    frozen memory model was calibrated on.  Leaving it on avoids emitting
    candidates the predictor can only reject as out-of-domain; turning it off is
    for experiment design, where probing an uncalibrated micro batch is the point.
    """

    model_id = str(request["model_id"])
    train_type = str(request["training_mode"]).strip().lower()
    if train_type not in {"full", "lora"}:
        raise ValueError("training_mode must be 'full' or 'lora'")
    target_gbs = int(request["target_gbs"])
    cutoff_len = int(request["cutoff_len"])
    total_parameters = int(
        _request_field(request, "actual_parameters", "model_parameters", default=0)
    )
    if total_parameters <= 0:
        raise ValueError("a positive parameter count is required for static pruning")
    packing = bool(request.get("packing", False))
    dataset_id = str(request["dataset_id"])
    if target_gbs <= 0 or cutoff_len <= 0:
        raise ValueError("target_gbs and cutoff_len must be positive")

    counts = tuple(gpu_counts) if gpu_counts else SUPPORTED_GPU_COUNTS
    prune_limit = capacity_bytes * static_prune_fraction

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    ladder = _mbs_ladder(cutoff_len)
    if restrict_to_supported_mbs:
        dropped = [value for value in ladder if value not in SUPPORTED_MBS]
        ladder = tuple(value for value in ladder if value in SUPPORTED_MBS)
        for value in dropped:
            rejected.append(
                {
                    "physical_mbs": value,
                    "reason": "mbs_outside_calibrated_support_grid",
                    "supported_mbs": list(SUPPORTED_MBS),
                }
            )

    for gpu_count in counts:
        for zero_stage in _zero_options(gpu_count, include_zero1=include_zero1):
            lower_bound = optimizer_state_lower_bound_bytes(
                total_parameters=total_parameters,
                train_type=train_type,
                gpu_count=gpu_count,
                zero_stage=zero_stage,
                lora_fraction=lora_fraction,
            )
            if lower_bound >= prune_limit:
                rejected.append(
                    {
                        "gpu_count": gpu_count,
                        "zero_stage": zero_stage,
                        "reason": "analytic_model_state_lower_bound_exceeds_capacity",
                        "model_state_lower_bound_bytes": lower_bound,
                        "static_prune_limit_bytes": int(prune_limit),
                    }
                )
                continue
            for physical_mbs in ladder:
                if packing:
                    # Neat packing fixes the physical micro batch at one; the
                    # sample-level batch is recovered from the pack occupancy by
                    # the frozen packing policy, not here.
                    if physical_mbs != 1:
                        continue
                    gradient_accumulation = None
                else:
                    gradient_accumulation = _derive_gradient_accumulation(
                        target_gbs=target_gbs,
                        gpu_count=gpu_count,
                        physical_mbs=physical_mbs,
                    )
                    if gradient_accumulation is None:
                        rejected.append(
                            {
                                "gpu_count": gpu_count,
                                "zero_stage": zero_stage,
                                "physical_mbs": physical_mbs,
                                "reason": "gbs_not_divisible_by_gpu_count_times_mbs",
                            }
                        )
                        continue
                for gradient_checkpointing in gradient_checkpointing_options:
                    candidate: dict[str, Any] = {
                        "model_id": model_id,
                        "training_mode": train_type,
                        "dataset_id": dataset_id,
                        "target_gbs": target_gbs,
                        "cutoff_len": cutoff_len,
                        "gpu_count": gpu_count,
                        "physical_mbs": physical_mbs,
                        "zero_stage": zero_stage,
                        "gradient_checkpointing": gradient_checkpointing,
                        "packing": packing,
                    }
                    for optional in (
                        "dataset_category",
                        "dtype",
                        "kernel_path",
                        "lora_rank",
                        "profile_tokenizer_id",
                        "profile_template_id",
                    ):
                        if request.get(optional) is not None:
                            candidate[optional] = request[optional]
                    if gradient_accumulation is not None:
                        candidate["gradient_accumulation_steps"] = gradient_accumulation
                    candidate["static_model_state_lower_bound_bytes"] = lower_bound
                    candidate["inside_predictor_supported_grid"] = bool(
                        gpu_count in SUPPORTED_GPU_COUNTS
                        and target_gbs in SUPPORTED_TARGET_GBS
                        and cutoff_len in SUPPORTED_CUTOFF_LENS
                        and physical_mbs in SUPPORTED_MBS
                    )
                    candidates.append(candidate)

    comparison_group = str(
        request.get("comparison_group")
        or "-".join(
            [
                model_id,
                train_type,
                dataset_id,
                f"gbs{target_gbs}",
                f"cut{cutoff_len}",
                f"pack{int(packing)}",
            ]
        )
    )
    for index, candidate in enumerate(candidates):
        candidate["comparison_group"] = comparison_group
        candidate.setdefault(
            "request_id",
            f"{comparison_group}-g{candidate['gpu_count']}"
            f"-m{candidate['physical_mbs']}-z{candidate['zero_stage']}"
            f"-gc{int(candidate['gradient_checkpointing'])}-{index:03d}",
        )

    by_gpu_count: dict[int, int] = {}
    for candidate in candidates:
        by_gpu_count[candidate["gpu_count"]] = (
            by_gpu_count.get(candidate["gpu_count"], 0) + 1
        )

    return {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "comparison_group": comparison_group,
        "scenario": {
            "model_id": model_id,
            "training_mode": train_type,
            "dataset_id": dataset_id,
            "target_gbs": target_gbs,
            "cutoff_len": cutoff_len,
            "packing": packing,
            "total_parameters": total_parameters,
        },
        "generation_policy": {
            "searched_fields": [
                "gpu_count",
                "zero_stage",
                "physical_mbs",
                "gradient_checkpointing",
            ],
            "fixed_fields": [
                "model_id",
                "training_mode",
                "dataset_id",
                "target_gbs",
                "cutoff_len",
                "packing",
            ],
            "static_prune_fraction": static_prune_fraction,
            "capacity_bytes": capacity_bytes,
            "include_zero1": include_zero1,
            "packed_physical_mbs_fixed_to_one": packing,
            "restrict_to_supported_mbs": restrict_to_supported_mbs,
            "lora_trainable_fraction_assumed": (
                lora_fraction if train_type == "lora" else None
            ),
        },
        "guarantees": {
            "predicts_memory": False,
            "admits_candidates": False,
            "ranks_candidates": False,
            "selects_gpu_count": False,
            "enables_packing": False,
            "creates_gpu_queue": False,
            "pruning_is_static_arithmetic_only": True,
        },
        "candidate_count": len(candidates),
        "candidates_by_gpu_count": dict(sorted(by_gpu_count.items())),
        "gpu_counts_with_at_least_two_candidates": sorted(
            gpu_count for gpu_count, count in by_gpu_count.items() if count >= 2
        ),
        "candidates": candidates,
        "statically_rejected": rejected,
        "next_step": (
            "Pass 'candidates' to the frozen memory gate first; only admitted "
            "candidates may be ranked, and the scale-out policy owns the final "
            "GPU count."
        ),
    }


def build_report(
    requests: Iterable[Mapping[str, Any]],
    *,
    capacity_bytes: int,
    **kwargs: Any,
) -> dict[str, Any]:
    groups = [
        generate_candidates(request, capacity_bytes=capacity_bytes, **kwargs)
        for request in requests
    ]
    report = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "candidate_generation_only",
        "group_count": len(groups),
        "total_candidates": sum(group["candidate_count"] for group in groups),
        "groups": groups,
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="JSON scenario or {'scenarios': []}")
    parser.add_argument(
        "--hardware",
        type=Path,
        default=None,
        help="hardware.json providing the per-GPU capacity",
    )
    parser.add_argument("--capacity-bytes", type=int, default=None)
    parser.add_argument("--include-zero1", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    capacity_bytes = args.capacity_bytes
    if capacity_bytes is None:
        if args.hardware is None:
            raise SystemExit("pass --capacity-bytes or --hardware")
        hardware = read_json(args.hardware)
        capacity_bytes = int(hardware["memory_bytes_reported_by_torch"])

    payload = read_json(args.request)
    requests = payload.get("scenarios") if isinstance(payload, Mapping) else payload
    if requests is None:
        requests = [payload]

    report = build_report(
        requests,
        capacity_bytes=capacity_bytes,
        include_zero1=args.include_zero1,
    )
    if args.output is not None:
        write_json(args.output, report)

    print(f"groups: {report['group_count']}")
    print(f"total candidates: {report['total_candidates']}")
    for group in report["groups"]:
        print(f"  {group['comparison_group']}: {group['candidate_count']}")
        print(f"    by gpu_count: {group['candidates_by_gpu_count']}")
        print(
            "    gpu counts with >=2 candidates: "
            f"{group['gpu_counts_with_at_least_two_candidates']}"
        )
        if group["statically_rejected"]:
            print(f"    statically rejected: {len(group['statically_rejected'])}")
    if args.output is not None:
        print(f"wrote {args.output}")
    else:
        print(json.dumps(report["groups"][0]["candidates"][:3], indent=1))


if __name__ == "__main__":
    main()
