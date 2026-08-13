#!/usr/bin/env python
"""Design the Phase-B preprocessing and runtime-consistency canary.

Phase B proves that the structured workload fed to the mathematical models
matches what the real training path actually executes.  Its checks are about
*software behaviour* -- tokenizer and template lengths, packing bin counts, loss
masking, cross-sample attention isolation, per-rank tensor inventory, event
counters -- so almost none of them depend on which Hopper SKU runs them.

That distinction is what makes this design useful right now.  The current pool
reports as H200 while the frozen coefficients are bound to an H800 runtime
fingerprint, so no run here may enter memory or throughput calibration.  A
consistency canary is different: it compares a static prediction against the
observed execution of the same run, on the same machine.  Both sides move
together, so the comparison stays valid.

Every check is therefore classified explicitly:

``hardware_independent``
    Valid on any CUDA device of the same software stack.  These may run on the
    current pool and their verdicts stand.

``hardware_bound``
    Numerically tied to the GPU SKU (memory peaks, step timing).  Recorded as
    diagnostics only, never promoted to evidence, and never compared against the
    frozen H800 artefacts.

The module writes a design and an approval request.  It does not launch GPUs,
does not write a queue, and does not grant itself authorisation: the approval
request is an input for a human decision, and ``launch_allowed`` stays false.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, CONFIG_DIR, ROOT, read_json, sha256_file, sha256_json, write_json
from prepare_h800_prospective_holdout import probe_hardware

SCHEMA = "sft_phase_b_consistency_canary_design/v1"
IMPLEMENTATION_VERSION = "sft_phase_b_canary_design_impl/2026-08-01.v1"

DEFAULT_OUTPUT = ARTIFACT_DIR / "phase_b_consistency_canary_design_v1.json"

CLASS_HARDWARE_INDEPENDENT = "hardware_independent"
CLASS_HARDWARE_BOUND = "hardware_bound"

# Checks whose verdict is a property of the software stack, not of the GPU SKU.
# Each states what it compares so a reviewer can see why the SKU cancels out.
CONSISTENCY_CHECKS: tuple[dict[str, Any], ...] = (
    {
        "check_id": "tokenized_length_matches_profile",
        "classification": CLASS_HARDWARE_INDEPENDENT,
        "compares": "static token-length profile vs lengths produced by the installed tokenizer and chat template",
        "rationale": "tokenization is deterministic CPU work; the GPU never changes a token count",
        "already_covered_by": "scripts/validate_preprocessing.py (CPU-only)",
        "requires_training_run": False,
    },
    {
        "check_id": "static_packing_matches_processor",
        "classification": CLASS_HARDWARE_INDEPENDENT,
        "compares": "static greedy-knapsack pack count and utilization vs the installed LLaMA-Factory processor",
        "rationale": "bin packing is deterministic given lengths, capacity and worker shard count",
        "already_covered_by": "scripts/validate_preprocessing.py (CPU-only)",
        "requires_training_run": False,
    },
    {
        "check_id": "packed_batch_loss_mask_correct",
        "classification": CLASS_HARDWARE_INDEPENDENT,
        "compares": "label mask of a packed batch vs the per-sample labels it was built from",
        "rationale": "masking is a collator property; a wrong mask is wrong on every device",
        "requires_training_run": True,
        "minimum_steps": 2,
    },
    {
        "check_id": "cross_sample_attention_isolated",
        "classification": CLASS_HARDWARE_INDEPENDENT,
        "compares": "attention reachability across packed sub-sequence boundaries vs the declared isolation",
        "rationale": "a leak across packed samples is a kernel/metadata defect, identical on any Hopper card",
        "requires_training_run": True,
        "minimum_steps": 2,
    },
    {
        "check_id": "computed_tokens_match_event_log",
        "classification": CLASS_HARDWARE_INDEPENDENT,
        "compares": "statically predicted computed tokens and attention pairs vs the per-rank event counters",
        "rationale": "both sides count the same work on the same run; the SKU cancels out of the ratio",
        "requires_training_run": True,
        "minimum_steps": 2,
    },
    {
        "check_id": "per_rank_inventory_consistent",
        "classification": CLASS_HARDWARE_INDEPENDENT,
        "compares": "logical tensor/module/trainable inventory across all ranks",
        "rationale": "rank disagreement is a sharding or configuration defect, not a device property",
        "requires_training_run": True,
        "minimum_steps": 1,
    },
    {
        "check_id": "freeze_flags_match_declaration",
        "classification": CLASS_HARDWARE_INDEPENDENT,
        "compares": "observed per-tensor requires_grad grouping vs the declared freeze flags",
        "rationale": "measured Qwen3-VL Full froze the vision tower, so the flag cannot be derived from train_type",
        "requires_training_run": True,
        "minimum_steps": 1,
    },
    {
        "check_id": "expected_gbs_reproduced",
        "classification": CLASS_HARDWARE_INDEPENDENT,
        "compares": "target GBS vs gpu_count * mbs * gradient_accumulation actually executed",
        "rationale": "the GBS contract is scheduling arithmetic, independent of the device",
        "requires_training_run": True,
        "minimum_steps": 2,
    },
    {
        "check_id": "execution_fingerprint_complete",
        "classification": CLASS_HARDWARE_INDEPENDENT,
        "compares": "presence and internal consistency of the two-stage execution evidence chain",
        "rationale": "completeness of the evidence chain is a property of the harness",
        "requires_training_run": True,
        "minimum_steps": 1,
    },
    {
        "check_id": "visual_path_observed",
        "classification": CLASS_HARDWARE_INDEPENDENT,
        "compares": "static image_grid_thw and visual token count vs the real image processor and collator",
        "rationale": "image geometry is deterministic; this is the check that lets visual_path_observed become true",
        "requires_training_run": True,
        "minimum_steps": 2,
        "vl_only": True,
    },
    {
        "check_id": "memory_peak_recorded",
        "classification": CLASS_HARDWARE_BOUND,
        "compares": "nothing; records max_allocated / max_reserved for diagnosis",
        "rationale": "absolute peaks depend on the SKU and allocator, so they cannot be reused as H800 evidence",
        "requires_training_run": True,
        "minimum_steps": 2,
        "evidence_eligible": False,
    },
    {
        "check_id": "step_timing_recorded",
        "classification": CLASS_HARDWARE_BOUND,
        "compares": "nothing; records step and micro-step seconds for diagnosis",
        "rationale": "timing is the quantity the throughput head models, so it must not cross hardware",
        "requires_training_run": True,
        "minimum_steps": 2,
        "evidence_eligible": False,
    },
)

# Small, bounded probes.  Deliberately not a matrix: this phase proves agreement,
# not performance, so each cell needs only enough steps to emit counters.
CANARY_CELLS: tuple[dict[str, Any], ...] = (
    {
        "cell_id": "text-unpacked-1gpu",
        "model_id": "qwen3_1p7b",
        "train_type": "lora",
        "dataset_id": "short_512",
        "cutoff_len": 512,
        "gpu_count": 1,
        "mbs": 2,
        "zero": "none",
        "gc": False,
        "packing": False,
        "target_gbs": 32,
        "purpose": "baseline unpacked counters and GBS contract",
    },
    {
        "cell_id": "text-packed-1gpu",
        "model_id": "qwen3_1p7b",
        "train_type": "lora",
        "dataset_id": "multiturn_4096",
        "cutoff_len": 4096,
        "gpu_count": 1,
        "mbs": 1,
        "zero": "none",
        "gc": False,
        "packing": True,
        "target_gbs": 32,
        "purpose": "packed loss mask, cross-sample attention isolation and pack counts",
    },
    {
        "cell_id": "text-unpacked-2gpu",
        "model_id": "qwen3_1p7b",
        "train_type": "lora",
        "dataset_id": "multiturn_4096",
        "cutoff_len": 4096,
        "gpu_count": 2,
        "mbs": 1,
        "zero": "zero2",
        "gc": False,
        "packing": False,
        "target_gbs": 32,
        "purpose": "per-rank inventory agreement under sharding",
    },
    {
        "cell_id": "vl-text-only-1gpu",
        "model_id": "qwen3_vl_8b",
        "train_type": "lora",
        "dataset_id": "multiturn_4096",
        "cutoff_len": 4096,
        "gpu_count": 1,
        "mbs": 1,
        "zero": "none",
        "gc": True,
        "packing": False,
        "target_gbs": 32,
        "purpose": "freeze-flag observation on a VL checkpoint; visual path stays unobserved",
        "expects_visual_path_observed": False,
    },
)

BOUND_SOURCES = (
    CONFIG_DIR / "experiment.json",
    CONFIG_DIR / "models.json",
    ARTIFACT_DIR / "preprocessing_validation.json",
    ARTIFACT_DIR / "static_workload_profiles.json",
)


def _source_bindings(paths: Sequence[Path] = BOUND_SOURCES) -> list[dict[str, Any]]:
    bindings = []
    for path in paths:
        bindings.append(
            {
                "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return bindings


def _approval_state(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Read the current approval and report whether it covers this canary."""

    path = config_dir / "APPROVED_TO_RUN.json"
    if not path.is_file():
        return {"present": False, "covers_canary": False, "reason": "approval_missing"}
    approval = read_json(path)
    allowed = [str(value) for value in (approval.get("allowed_job_ids") or [])]
    canary_ids = {cell["cell_id"] for cell in CANARY_CELLS}
    covered = sorted(canary_ids & set(allowed))
    return {
        "present": True,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "phase_id": approval.get("phase_id"),
        "approved_model_ids": approval.get("approved_model_ids"),
        "allowed_job_id_count": len(allowed),
        "covered_canary_cells": covered,
        # The existing approval lists finished generalization jobs, so it cannot
        # authorise a new canary no matter how permissive it looks.
        "covers_canary": len(covered) == len(canary_ids),
        "reason": (
            "current approval enumerates completed jobs from a different phase"
            if not covered
            else "partial coverage"
        ),
    }


def build_design(
    *,
    required_gpu_ids: Sequence[int] = (0, 1, 2, 3),
    hardware_probe: Mapping[str, Any] | None = None,
    include_vl: bool = True,
) -> dict[str, Any]:
    """Build the Phase-B canary design and its approval request."""

    probe = (
        dict(hardware_probe)
        if hardware_probe is not None
        else probe_hardware(required_gpu_ids=required_gpu_ids)
    )
    observed = probe.get("selected_gpu_rows") or []
    observed_names = sorted({str(row.get("name")) for row in observed if row.get("name")})
    # ``probe_hardware.selected_pool_idle`` deliberately folds "is an exact H800
    # pool" into its idle flag, because Phase D needs both at once.  Phase B needs
    # them separated: occupancy decides whether a canary can start, SKU decides
    # only whether its numbers may become H800 evidence.
    pool_busy = bool(probe.get("selected_gpu_compute_processes"))
    pool_idle = not pool_busy
    pool_complete = not probe.get("missing_gpu_ids")
    exact_h800_pool = probe.get("exact_h800_pool") is True

    cells = [
        cell
        for cell in CANARY_CELLS
        if include_vl or not str(cell["model_id"]).endswith("vl_8b")
    ]
    checks = [
        check
        for check in CONSISTENCY_CHECKS
        if include_vl or not check.get("vl_only")
    ]

    independent = [
        check for check in checks if check["classification"] == CLASS_HARDWARE_INDEPENDENT
    ]
    bound = [
        check for check in checks if check["classification"] == CLASS_HARDWARE_BOUND
    ]

    # The whole reason this phase can run on a mismatched pool: its verdicts do
    # not cross hardware, because both sides of every comparison come from the
    # same run.
    hardware_policy = {
        "expected_gpu_family_for_frozen_models": "H800",
        "observed_gpu_names": observed_names,
        "pool_matches_frozen_family": exact_h800_pool,
        "pool_complete": pool_complete,
        "pool_idle": pool_idle,
        "occupancy_and_sku_assessed_separately": True,
        "runs_may_enter_memory_calibration": False,
        "runs_may_enter_throughput_calibration": False,
        "runs_may_close_out_rank_validation": False,
        "consistency_verdicts_valid_on_this_pool": True,
        "justification": (
            "Each hardware-independent check compares a static prediction with "
            "the observed execution of the same run on the same machine, so the "
            "GPU SKU cancels out. Absolute memory and timing are recorded as "
            "diagnostics and are explicitly not evidence-eligible."
        ),
    }

    approval_state = _approval_state()
    approval_request = {
        "requested_phase_id": "phase_b_consistency_canary_20260801",
        "requested_job_ids": [cell["cell_id"] for cell in cells],
        "requested_model_ids": sorted({cell["model_id"] for cell in cells}),
        "requested_gpu_ids": list(required_gpu_ids),
        "maximum_gpu_count_per_job": max(cell["gpu_count"] for cell in cells),
        "maximum_steps_per_job": 2,
        "total_jobs": len(cells),
        "writes_frozen_artifacts": False,
        "enters_calibration": False,
        "enables_scale_out": False,
        "purpose": (
            "Prove static workload accounting matches real execution before any "
            "calibration or acceptance run is scheduled."
        ),
        "current_approval": approval_state,
        "approval_action_required": not approval_state.get("covers_canary"),
    }

    blockers: list[str] = []
    if not pool_complete:
        blockers.append("required_gpu_indices_missing")
    if not pool_idle:
        blockers.append("gpu_pool_not_idle")
    if approval_request["approval_action_required"]:
        blockers.append("canary_not_covered_by_current_approval")

    design = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "design_pending_approval",
        "phase": "B",
        "objective": (
            "Prove that the structured workload handed to the memory and "
            "throughput models matches the real training path."
        ),
        "hardware_policy": hardware_policy,
        "hardware_probe": probe,
        "check_counts": {
            CLASS_HARDWARE_INDEPENDENT: len(independent),
            CLASS_HARDWARE_BOUND: len(bound),
        },
        "checks": checks,
        "cells": cells,
        "cpu_only_checks_already_passing": [
            check["check_id"]
            for check in checks
            if not check["requires_training_run"]
        ],
        "source_bindings": _source_bindings(),
        "approval_request": approval_request,
        "guarantees": {
            "creates_gpu_queue": False,
            "writes_job_jsonl": False,
            "grants_own_approval": False,
            "mutates_frozen_artifacts": False,
            "promotes_hardware_bound_results_to_evidence": False,
        },
        "launch_allowed": False,
        "blockers": blockers,
        "next_step": (
            "A reviewer approves the requested phase_id and job ids; only then "
            "may these cells be materialized. Hardware-bound results stay "
            "diagnostic until the pool identity matches the frozen H800 family."
        ),
    }
    design["design_sha256"] = sha256_json(design)
    return design


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=(0, 1, 2, 3),
        help="GPU indices the canary would use",
    )
    parser.add_argument("--no-vl", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    design = build_design(
        required_gpu_ids=args.gpu_ids, include_vl=not args.no_vl
    )
    write_json(args.output, design)

    policy = design["hardware_policy"]
    print(f"phase: {design['phase']}  status: {design['status']}")
    print(f"observed GPUs: {policy['observed_gpu_names']}")
    print(f"matches frozen H800 family: {policy['pool_matches_frozen_family']}")
    print(f"pool complete: {policy['pool_complete']}  idle: {policy['pool_idle']}")
    print(f"check counts: {design['check_counts']}")
    print(f"cells: {len(design['cells'])}")
    print("consistency verdicts valid on this pool: "
          f"{policy['consistency_verdicts_valid_on_this_pool']}")
    print("runs may enter calibration: "
          f"{policy['runs_may_enter_memory_calibration']}")
    request = design["approval_request"]
    print(f"approval action required: {request['approval_action_required']}")
    print(f"  requested phase_id: {request['requested_phase_id']}")
    print(f"  requested jobs: {request['requested_job_ids']}")
    print(f"launch_allowed: {design['launch_allowed']}")
    for blocker in design["blockers"]:
        print(f"blocker: {blocker}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
