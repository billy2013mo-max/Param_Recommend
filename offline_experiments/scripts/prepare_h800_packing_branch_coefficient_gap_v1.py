#!/usr/bin/env python3
"""Design the packed-branch coefficient-gap campaign (CPU only, no GPU launch).

The branch-specific packed centre challenger
(``fit_packing_branch_specific_memory_centre_challenger_v1.py``) fits eight
coefficients on twelve packed arms.  Two gaps make that fit unusable for an
acceptance claim, and neither can be closed by re-analysis:

1. ``ZeRO-3 + GC-off`` has zero packed arms.  The existing mechanism grid is
   ZeRO-2/GC-on 6, ZeRO-2/GC-off 2, ZeRO-3/GC-on 2, ZeRO-3/GC-off **0**.  That
   missing cell is also the throughput-optimal direction (turning GC off is the
   single largest lever measured on this hardware), so the recommender is likely
   to select exactly the mechanism the memory model has never seen.
2. ``W5`` contributes a single packed arm, so the leave-one-workload-out fold
   that holds out W5 scores one point.  Its 0.14% error is not evidence.

The high-memory gap is deliberately NOT in this campaign: Phase-D Stage 1 already
supplies packed anchors at 112.4 and 137.6 GiB and they are ``role: calibration``,
so they are folded into the fit by re-analysis instead of new GPU time.

This script writes a design, a JSONL queue and a queue manifest.  It never
authorizes or launches training, never mutates a frozen artifact, and refuses to
overwrite an existing campaign.  Launching requires a separate explicit approval
promotion, per the standing rule that GPU work starts only when the user says so.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h800_theory_basis as theory
from common import (
    ARTIFACT_DIR,
    MATRIX_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from prepare_packing_dataprofile_v2 import _curve


SCHEMA = "sft_h800_packing_branch_coefficient_gap_design/v3"
JOB_SCHEMA = "sft_h800_packing_branch_coefficient_gap_job/v3"
CAMPAIGN_ID = "h800_packing_branch_coefficient_gap_b3_20260808_v1"
PHASE_ID = "h800_packing_branch_coefficient_gap_b3_v1"

GPU_COUNT = 2
MBS = 1  # neat packing asserts bsz==1 in the collator
# The user directed this campaign onto GPU 0-3.  The pool is fully NVLinked, so
# the two disjoint two-GPU masks are [0,1] and [2,3].
GPU_IDS = (0, 1, 2, 3)
TWO_GPU_MASKS = [[0, 1], [2, 3]]
WARMUP_STEPS = 1
MEASURE_STEPS = 4
TOTAL_STEPS = WARMUP_STEPS + MEASURE_STEPS
REPEATS = 2
EPSILON_GBS = 0.10
SAFE_LIMIT_GIB = 132.83927001953126

CHALLENGER = (
    ROOT
    / "diagnostics"
    / "packing_branch_specific_memory_centre_20260807"
    / "packing_branch_specific_memory_centre_challenger_v1.json"
)
MODEL_INVENTORY = ARTIFACT_DIR / "model_inventory.json"
HARDWARE = ROOT / "config" / "hardware.json"
PROFILE_DIR = ARTIFACT_DIR / "packing_profile_phase_b_v1" / "profiles"
DATA_DIR = ROOT / "data" / "packing_profile_phase_b_v1"
DATAPROFILE_DIR = ARTIFACT_DIR / "packing_dataprofile_v2"

# Observed peak reserved divided by the model's *centre* prediction.  This ratio
# is mechanism dependent and two batches were lost to under-estimating it:
#   ZeRO-2 packed (Stage 1 B1, batch-1 G3):  0.861, 0.862
#   ZeRO-3 packed (batch-2 G5, W3):          1.052
#   ZeRO-3 packed (batch-2 G4, W8):          > 1.137  (OOM, lower bound only)
# ZeRO-3 shards parameters, so its memory shape differs from ZeRO-2 and the 0.862
# figure does not transfer.  W8 packs 25.6 samples per pack against W3's 10.2,
# and more segments means more attention workspace, so W8 sits above W3.  Size
# against the conservative end, never the analytic reference (which is ~45% below
# the centre and caused the batch-1 OOMs).
OBSERVED_OVER_CENTRE = 1.25
# Keep a real margin below the safe limit rather than aiming at it.
TARGET_OBSERVED_CEILING_GIB = 125.0

QUEUE = MATRIX_DIR / "h800_packing_branch_coefficient_gap_b3_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_packing_branch_coefficient_gap_b3_design_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_branch_coefficient_gap_b3_queue_manifest_v1.json"

DATASET_CATEGORIES = {"W3": "multiturn", "W5": "longcontext", "W7": "longtail", "W8": "longtail"}
PROFILE_FILES = {
    "W3": "packing_w3_multiturn_probe_v2.qwen3_nothink.jsonl",
    "W5": "packing_w5_longcontext_probe_v2.qwen3_nothink.jsonl",
    "W7": "packing_w7_bimodal_v1.qwen3_nothink.jsonl",
    "W8": "packing_w8_code_structured_v1.qwen3_nothink.jsonl",
}
DATA_FILES = {
    "W3": "packing_w3_multiturn_probe_v2.jsonl",
    "W5": "packing_w5_longcontext_probe_v2.jsonl",
    "W7": "packing_w7_bimodal_v1.jsonl",
    "W8": "packing_w8_code_structured_v1.jsonl",
}

# Cutoffs are chosen so the analytic reference lands in the 90-130 GiB band --
# high enough to add information near the admission boundary, low enough that the
# observed peak should stay under the 132.84 GiB safe limit given the packed
# residual measured so far (0.75-1.00 of analytic).  Each setting states the gap
# it closes so a reviewer can check the campaign against the stated purpose.
SETTINGS: tuple[dict[str, Any], ...] = (
    {
        "setting_id": "G6_8B_LoRA_W8_Z3_GCoff-c8192",
        "gap": "zero3_gc_off_cell_lacks_a_second_workload",
        "workload_id": "W8",
        "model_id": "qwen3_8b",
        "cutoff_len": 8192,
        "zero_stage": 3,
        "gc": False,
        "target_gbs": 256,
        "reason": (
            "low anchor of a bracket. W8 OOMed at both cutoff 20480 (batch 1) and "
            "12288 (batch 2), so its observed/centre ratio is only known to exceed "
            "1.137; this point is sized to succeed even at a 1.25 ratio (85.8 -> "
            "107.2 GiB) and pins the ratio from below."
        ),
    },
    {
        "setting_id": "G7_8B_LoRA_W8_Z3_GCoff-c10240",
        "gap": "zero3_gc_off_cell_lacks_a_second_workload",
        "workload_id": "W8",
        "model_id": "qwen3_8b",
        "cutoff_len": 10240,
        "zero_stage": 3,
        "gc": False,
        "target_gbs": 256,
        "reason": (
            "high anchor of the same bracket (104.4 -> 118.7 GiB at the 1.137 lower "
            "bound). Running both anchors together brackets the boundary in one "
            "batch instead of guessing a single cutoff a third time; if the high "
            "anchor OOMs the low one still closes the mechanism cell."
        ),
    },
)


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _require_challenger() -> dict[str, Any]:
    """Read the challenger and re-derive the gap this batch claims to close.

    Batch 1 (2026-08-08, settings G1-G3) closed the single-arm W5 profile but its
    two ZeRO-3/GC-off settings OOMed at cutoff 20480, so that mechanism cell is
    still unfitted.  This batch retries it at cutoff 12288.  The guard therefore
    checks the cell is still empty rather than re-checking W5, and refuses to run
    if a parallel refit has already filled it.
    """
    report = read_json(CHALLENGER)
    if report.get("schema") != "sft_packing_branch_specific_memory_centre_challenger/v1":
        raise ValueError("challenger artifact has an unexpected schema")
    if report.get("publishable") is not False:
        raise ValueError("challenger is unexpectedly marked publishable")
    coverage = report["high_memory_coverage"]
    if coverage.get("zero3_gc_off_still_unfitted") is not True:
        raise ValueError(
            "the ZeRO-3/GC-off cell already has a fitted packed arm; this batch is "
            "redundant, re-derive the campaign before running it"
        )
    if coverage["extended_packed_reserved_gib_range"][1] < 130.0:
        raise ValueError(
            "the high-memory anchor is missing from the challenger; this campaign "
            "assumes Stage 1 already supplied it"
        )
    return report


def _models() -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in read_json(MODEL_INVENTORY)["models"]}


def _centre_prediction_gib(setting: dict[str, Any]) -> float:
    """Centre prediction from the frozen physical predictor, in GiB.

    This is the quantity a run must be sized against.  The analytic reference
    from ``memory_basis`` is roughly 45% lower and sizing against it is what made
    the first batch OOM.
    """
    from h800_physical_v4b_predictor import H800PhysicalV4BPredictor

    workload = str(setting["workload_id"])
    request = {
        "request_id": str(setting["setting_id"]),
        "comparison_group": str(setting["setting_id"]),
        "model_id": str(setting["model_id"]),
        "training_mode": "lora",
        "dataset_id": Path(DATA_FILES[workload]).stem,
        "dataset_category": DATASET_CATEGORIES[workload],
        "target_gbs": int(setting["target_gbs"]),
        "cutoff_len": int(setting["cutoff_len"]),
        "gpu_count": GPU_COUNT,
        "physical_mbs": MBS,
        # GA does not enter the memory basis; the contract-checked value is
        # recorded separately in the design.
        "gradient_accumulation_steps": 4,
        "zero_stage": int(setting["zero_stage"]),
        "gradient_checkpointing": bool(setting["gc"]),
        "packing": True,
        "offload": False,
        "dtype": "bf16",
        "kernel_path": "fa3_orig+liger_fused_ce+adamw_torch_fused",
        "lora_rank": 32,
        "profile_tokenizer_id": "qwen3_8b@local",
        "profile_template_id": "qwen3_nothink",
    }
    predictor = H800PhysicalV4BPredictor(
        model_inventory=MODEL_INVENTORY,
        strict_model_inventory_binding=False,
        additional_dataset_profile_dir=PROFILE_DIR,
    )
    memory = predictor.predict([request])["predictions"][0]["memory"]
    if memory.get("prediction_available") is not True:
        raise ValueError(f"{setting['setting_id']}: memory prediction unavailable")
    return float(memory["reserved_center_bytes"]) / float(2**30)


def _plan_setting(setting: dict[str, Any], models: dict[str, dict[str, Any]], inventory: dict[str, Any], capacity: int) -> dict[str, Any]:
    workload = str(setting["workload_id"])
    profile_path = PROFILE_DIR / PROFILE_FILES[workload]
    data_path = DATA_DIR / DATA_FILES[workload]
    dataprofile_path = DATAPROFILE_DIR / f"{workload.lower()}_packing_dataprofile_v2.json"
    for path in (profile_path, data_path, dataprofile_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    profile_rows = read_jsonl(profile_path)
    data_rows = read_jsonl(data_path)
    if len(profile_rows) != len(data_rows):
        raise ValueError(f"{setting['setting_id']}: data and profile row counts differ")
    cutoff = int(setting["cutoff_len"])
    curve = _curve([int(row["total_tokens"]) for row in profile_rows], [cutoff])[0]
    n_pack_mean = float(curve["samples_per_pack"]["mean"])

    # Integer GA chosen on the exact execution snapshot, same rule as Stage 1.
    target_gbs = float(setting["target_gbs"])
    raw_ga = target_gbs / (GPU_COUNT * n_pack_mean)
    candidates = sorted({max(1, math.floor(raw_ga)), max(1, math.ceil(raw_ga)), max(1, round(raw_ga))})
    packed_ga = min(
        candidates,
        key=lambda value: (abs(GPU_COUNT * value * n_pack_mean - target_gbs), value),
    )
    expected_gbs = GPU_COUNT * packed_ga * n_pack_mean
    gbs_error = abs(expected_gbs - target_gbs) / target_gbs
    step_p99 = GPU_COUNT * float(curve["samples_per_pack"]["p99"])
    gbs_gates = {
        "expected_sample_gbs": expected_gbs,
        "expected_sample_gbs_relative_error": gbs_error,
        "relative_error_within_epsilon": gbs_error <= EPSILON_GBS,
        "global_microstep_sample_p99": step_p99,
        # The hard per-step gate uses p99, not the mean: the mean only sizes GA
        # and expected epoch GBS, and a long tail in samples-per-pack can blow the
        # single-step contract even when the mean is on target.
        "step_p99_within_epsilon": step_p99 <= target_gbs * (1 + EPSILON_GBS),
    }
    if not gbs_gates["relative_error_within_epsilon"] or not gbs_gates["step_p99_within_epsilon"]:
        raise ValueError(f"{setting['setting_id']}: GBS contract failed: {gbs_gates}")

    required_packs = GPU_COUNT * packed_ga * TOTAL_STEPS
    epoch = {
        "available_packs": int(curve["packs"]),
        "required_packs": required_packs,
        "passed": int(curve["packs"]) >= required_packs,
    }
    if not epoch["passed"]:
        raise ValueError(f"{setting['setting_id']}: probe would cross an epoch boundary")

    job_spec = {
        "gpu_count": GPU_COUNT,
        "mbs": MBS,
        "cutoff_len": cutoff,
        "zero": f"zero{int(setting['zero_stage'])}",
        "gc": bool(setting["gc"]),
        "train_type": "lora",
        "model_id": str(setting["model_id"]),
    }
    geometry = theory._model_geometry(job_spec, models[str(setting["model_id"])], inventory["fixed_lora"])
    basis = theory.memory_basis(job_spec, geometry, capacity)
    analytic_gib = float(basis["analytic_reference_bytes"]) / float(2**30)
    centre_gib = _centre_prediction_gib(setting)
    # Size the run against the model's CENTRE, not the analytic reference.  The
    # first batch of this campaign sized against analytic and OOMed: for
    # 8B/LoRA/c20480 the analytic value is 110.1 GiB while the centre is 159.8,
    # and the observed peak came in at 137.6.
    expected_observed_gib = centre_gib * OBSERVED_OVER_CENTRE
    band = {
        "analytic_reference_gib": analytic_gib,
        "model_centre_gib": centre_gib,
        "observed_over_centre_ratio": OBSERVED_OVER_CENTRE,
        "expected_observed_gib": expected_observed_gib,
        "target_observed_ceiling_gib": TARGET_OBSERVED_CEILING_GIB,
        "safe_limit_gib": SAFE_LIMIT_GIB,
        "expected_observed_within_ceiling": expected_observed_gib <= TARGET_OBSERVED_CEILING_GIB,
        "sizing_basis": "model_centre_times_measured_observed_ratio",
    }
    if not band["expected_observed_within_ceiling"]:
        raise ValueError(
            f"{setting['setting_id']}: expected observed {expected_observed_gib:.1f} GiB "
            f"exceeds the {TARGET_OBSERVED_CEILING_GIB} GiB working ceiling "
            f"(centre {centre_gib:.1f}); pick a lower cutoff"
        )

    return {
        **setting,
        "profile_path": profile_path,
        "data_path": data_path,
        "dataprofile_path": dataprofile_path,
        "curve": curve,
        "n_pack_mean": n_pack_mean,
        "packed_ga": packed_ga,
        "gbs_gates": gbs_gates,
        "first_epoch_capacity": epoch,
        "memory_band": band,
        "records": len(data_rows),
    }


def prepare(*, force: bool) -> dict[str, Any]:
    if not force:
        for path in (QUEUE, DESIGN, QUEUE_MANIFEST):
            if path.exists():
                raise FileExistsError(
                    f"{path} already exists; refusing to overwrite a materialized campaign"
                )
    challenger = _require_challenger()
    inventory = read_json(MODEL_INVENTORY)
    models = _models()
    hardware = read_json(HARDWARE)
    capacity = int(hardware["memory_bytes_reported_by_torch"])

    plans = [_plan_setting(dict(s), models, inventory, capacity) for s in SETTINGS]

    jobs: list[dict[str, Any]] = []
    for index, plan in enumerate(plans):
        for repeat in range(REPEATS):
            payload = {
                "campaign_id": CAMPAIGN_ID,
                "phase_id": PHASE_ID,
                "setting_id": plan["setting_id"],
                "repeat": repeat,
            }
            job_id = stable_id("h800packgap", payload)
            jobs.append(
                {
                    "schema": JOB_SCHEMA,
                    "job_id": job_id,
                    "campaign_id": CAMPAIGN_ID,
                    "phase_id": PHASE_ID,
                    "setting_id": plan["setting_id"],
                    "coefficient_gap_closed": plan["gap"],
                    "selection_reason": plan["reason"],
                    "kind": "throughput",
                    "fidelity": "packed_coefficient_gap_1plus4",
                    "candidate_role": "packing_branch_coefficient_gap_fit_only",
                    "calibration_partition": {
                        "role": "calibration",
                        "policy": "packing_branch_coefficient_gap_never_acceptance_v1",
                        "split_unit_id": plan["setting_id"],
                    },
                    "model_id": plan["model_id"],
                    "model_path": f"/wanqing-models/Qwen3-{'8B' if plan['model_id']=='qwen3_8b' else '14B'}",
                    "tokenizer_path": f"/wanqing-models/Qwen3-{'8B' if plan['model_id']=='qwen3_8b' else '14B'}",
                    "model_family": "qwen3",
                    "train_type": "lora",
                    "workload_id": plan["workload_id"],
                    "dataset_id": Path(DATA_FILES[plan["workload_id"]]).stem,
                    "dataset_category": DATASET_CATEGORIES[plan["workload_id"]],
                    "template": "qwen3_nothink",
                    "data_path": str(plan["data_path"].resolve()),
                    "data_sha256": sha256_file(plan["data_path"]),
                    "dataset_profile_path": str(plan["profile_path"].resolve()),
                    "dataset_profile_sha256": sha256_file(plan["profile_path"]),
                    "packing_dataprofile_path": str(plan["dataprofile_path"].resolve()),
                    "packing_dataprofile_sha256": sha256_file(plan["dataprofile_path"]),
                    "gpu_type": "NVIDIA H800 140GB HBM3",
                    "required_runtime_gpu_name": "NVIDIA H800",
                    "hardware_id": "local_h800_140g",
                    "gpu_count": GPU_COUNT,
                    "mbs": MBS,
                    "cutoff_len": plan["cutoff_len"],
                    "zero": f"zero{int(plan['zero_stage'])}",
                    "zero_stage": int(plan["zero_stage"]),
                    "gc": bool(plan["gc"]),
                    "gradient_checkpointing": bool(plan["gc"]),
                    "packing": True,
                    "neat_packing": True,
                    "offload": False,
                    "gradient_accumulation_steps": plan["packed_ga"],
                    "target_gbs": plan["target_gbs"],
                    "expected_sample_gbs": plan["gbs_gates"]["expected_sample_gbs"],
                    "expected_sample_gbs_relative_error": plan["gbs_gates"][
                        "expected_sample_gbs_relative_error"
                    ],
                    "global_microstep_sample_p99": plan["gbs_gates"]["global_microstep_sample_p99"],
                    "n_pack_mean": plan["n_pack_mean"],
                    "n_pack_step_p99": float(plan["curve"]["samples_per_pack"]["p99"]),
                    "first_epoch_capacity": plan["first_epoch_capacity"],
                    "analytic_reference_gib": plan["memory_band"]["analytic_reference_gib"],
                    "model_centre_gib": plan["memory_band"]["model_centre_gib"],
                    "expected_observed_gib": plan["memory_band"]["expected_observed_gib"],
                    "warmup_steps": WARMUP_STEPS,
                    "measure_steps": MEASURE_STEPS,
                    "max_samples": plan["records"],
                    "repeat": repeat,
                    "execution_sequence_index": index * REPEATS + repeat,
                    "parallel_class": "gpu_partitionable",
                    "requires_external_node_idle": False,
                    "oom_role": "right_censored_lower_bound",
                    "declared_model_manifest_path": str(MODEL_INVENTORY.resolve()),
                    "declared_model_manifest_sha256": sha256_file(MODEL_INVENTORY),
                    "publication_allowed": False,
                    "automatic_packing_recommendation_allowed": False,
                }
            )

    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("duplicate job ids")
    write_jsonl(QUEUE, jobs)

    design: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "design_only_not_approved_not_queued_for_execution",
        "gpu_training_started": False,
        "publication_allowed": False,
        "required_gpu_pool": {
            "gpu_ids": list(GPU_IDS),
            "two_gpu_masks": TWO_GPU_MASKS,
            "max_gpu_count": GPU_COUNT,
        },
        "all_gbs_contracts_passed": all(
            p["gbs_gates"]["relative_error_within_epsilon"]
            and p["gbs_gates"]["step_p99_within_epsilon"]
            for p in plans
        ),
        "all_first_epoch_capacity_checks_passed": all(
            p["first_epoch_capacity"]["passed"] for p in plans
        ),
        "all_memory_bands_within_safe_limit": all(
            p["memory_band"]["expected_observed_within_ceiling"] for p in plans
        ),
        "objective": (
            "close the two coefficient gaps that block an acceptance claim for the "
            "branch-specific packed memory centre: the empty ZeRO-3/GC-off cell and "
            "the single-arm W5 profile"
        ),
        "motivating_evidence": {
            "challenger": {
                **_binding(CHALLENGER),
                "report_sha256": challenger["report_sha256"],
            },
            "packed_arms_before_campaign": challenger["challenger_packed_plus_stage1"]["arms"],
            "fitted_coefficients": challenger["fragility"]["fitted_coefficients"],
            "arms_per_coefficient_before": challenger["fragility"]["arms_per_coefficient"],
            "single_arm_profile_groups_before": challenger["fragility"][
                "single_arm_profile_groups"
            ],
        },
        "gaps_addressed": {
            "zero3_gc_off_cell_has_no_packed_arm": {
                "settings": [p["setting_id"] for p in plans if p["gap"].startswith("zero3")],
                "why_it_matters": (
                    "turning gradient checkpointing off is the largest measured "
                    "throughput lever on this hardware, so the recommender will "
                    "steer into a mechanism the packed memory model has never seen"
                ),
            },
            "w5_has_a_single_packed_arm": {
                "settings": [p["setting_id"] for p in plans if p["gap"].startswith("w5")],
                "why_it_matters": (
                    "a one-point leave-one-workload-out fold cannot support any "
                    "accuracy statement for that profile"
                ),
            },
        },
        "gaps_deliberately_not_addressed": {
            "high_memory_range": (
                "Phase-D Stage 1 already supplies packed anchors at 112.4 and 137.6 "
                "GiB with role=calibration, so the range is closed by re-analysis "
                "rather than new GPU time"
            ),
            "14b_coverage": (
                "Stage 1 contributes a 14B packed arm whose held-out error is 2.36%; "
                "adding more 14B arms is not the binding constraint"
            ),
        },
        "measurement_contract": {
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "total_optimizer_steps": TOTAL_STEPS,
            "repeats_per_setting": REPEATS,
            "mbs": MBS,
            "mbs_is_forced": "neat packing asserts bsz==1 in the collator",
            "epsilon_gbs": EPSILON_GBS,
            "reserved_reduction": "maximum over all rank summaries",
        },
        "settings": [
            {
                "setting_id": p["setting_id"],
                "coefficient_gap_closed": p["gap"],
                "workload_id": p["workload_id"],
                "model_id": p["model_id"],
                "cutoff_len": p["cutoff_len"],
                "zero_stage": p["zero_stage"],
                "gc": p["gc"],
                "target_gbs": p["target_gbs"],
                "gradient_accumulation_steps": p["packed_ga"],
                "n_pack_mean": p["n_pack_mean"],
                "gbs_gates": p["gbs_gates"],
                "first_epoch_capacity": p["first_epoch_capacity"],
                "memory_band": p["memory_band"],
                "selection_reason": p["reason"],
            }
            for p in plans
        ],
        "queue": {
            "path": str(QUEUE.resolve()),
            "sha256": sha256_file(QUEUE),
            "jobs": len(jobs),
            "gpu_job_equivalents": len(jobs) * GPU_COUNT,
            "ordered_job_ids": [job["job_id"] for job in jobs],
        },
        "source_bindings": {
            "model_inventory": _binding(MODEL_INVENTORY),
            "hardware": _binding(HARDWARE),
            "preparer": _binding(Path(__file__).resolve()),
        },
        "execution_contract": {
            "queue_materialized": True,
            "gpu_training_started": False,
            "automatic_gpu_launch_allowed": False,
            "requires_explicit_approval_promotion": True,
            "preemption_allowed": False,
            "join_busy_pool": True,
            "oom_role": "right_censored_lower_bound_not_regression_label",
            "software_failure_role": "repair_and_rerun_same_job_id_and_payload",
        },
        "safety_flags": {
            "publication_allowed": False,
            "automatic_packing_recommendation_allowed": False,
            "frozen_artifacts_modified": False,
            "gpu_experiments_launched": False,
            "queues_mutated_outside_this_campaign": False,
        },
        "acceptance_note": (
            "this campaign only makes the packed fit reportable; it is not itself an "
            "acceptance result. After it lands, the branch-specific challenger must "
            "be re-scored and a fresh prospective holdout designed before any "
            "recommendation is enabled."
        ),
        "next_step": (
            "review the design, then promote an approval for exactly these job ids "
            "when GPUs are free and no other campaign holds the approval lock"
        ),
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)

    manifest = {
        "schema": "sft_h800_packing_branch_coefficient_gap_queue_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": design["generated_at_utc"],
        "design": {"path": str(DESIGN.resolve()), "sha256": sha256_file(DESIGN)},
        "queue": {"path": str(QUEUE.resolve()), "sha256": sha256_file(QUEUE)},
        "jobs": len(jobs),
        "gpu_job_equivalents": len(jobs) * GPU_COUNT,
        "gpu_training_started": False,
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(QUEUE_MANIFEST, manifest)
    return design


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing design/queue (only for pre-approval iteration)",
    )
    args = parser.parse_args()
    design = prepare(force=args.force)
    print(
        json.dumps(
            {
                "design": str(DESIGN),
                "queue": str(QUEUE),
                "report_sha256": design["report_sha256"],
                "jobs": design["queue"]["jobs"],
                "gpu_job_equivalents": design["queue"]["gpu_job_equivalents"],
                "settings": [
                    {
                        "setting_id": s["setting_id"],
                        "gap": s["coefficient_gap_closed"],
                        "workload": s["workload_id"],
                        "model": s["model_id"],
                        "cutoff_len": s["cutoff_len"],
                        "zero_stage": s["zero_stage"],
                        "gc": s["gc"],
                        "ga": s["gradient_accumulation_steps"],
                        "gbs_error": s["gbs_gates"]["expected_sample_gbs_relative_error"],
                        "analytic_gib": s["memory_band"]["analytic_reference_gib"],
                        "model_centre_gib": s["memory_band"]["model_centre_gib"],
                        "expected_observed_gib": s["memory_band"]["expected_observed_gib"],
                    }
                    for s in design["settings"]
                ],
                "gpu_training_started": design["execution_contract"]["gpu_training_started"],
                "next_step": design["next_step"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
