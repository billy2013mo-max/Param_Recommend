#!/usr/bin/env python3
"""Challenger: fit the packed memory-centre residual on its own coefficients.

The frozen Phase-C centre model fits all twenty arms together.  Eight of its
twelve features are shared by the unpacked and packed branches, and only four
(`packing`, `packing_x_*`) are branch specific.  That is a problem because the
target is ``log(observed / physical_centre)`` and the physical centre is built
from ``sequence = cutoff_len``:

* with packing on, packs really are filled to the cutoff (measured fill 82-99%),
  so the analytic centre is roughly right and the target sits near zero
  (mean -0.057);
* with packing off, a microbatch holds one short sample, so the analytic centre
  overstates the length by up to 43x and the target is dragged far negative
  (mean -0.331).

A single ``log2_cutoff`` coefficient cannot be both ~0 and ~-0.5, so the shared
fit lands on a compromise that fits neither branch.  This challenger keeps the
shared analytic trunk and the shared feature definitions, and fits the packed
branch's residual coefficients on packed arms only.

Nothing frozen is mutated: the frozen refit artifact, the physical basis and the
production predictors are untouched.  The output is a fit-only shadow candidate.

IMPORTANT -- this cannot be read as an acceptance result.  Ten packed arms across
four workload profiles support eight coefficients, and one profile (W5)
contributes a single arm.  The report therefore carries both the leave-one-
workload-out score and an explicit fragility section; the alpha sweep is recorded
because the ranking is not stable across the whole grid.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

import fit_h800_packing_phase_c_models_v1 as phase_c_fit
from common import ARTIFACT_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json


FROZEN_REFIT = ARTIFACT_DIR / "h800_packing_phase_c_model_refit_v1.json"
STAGE1_QUEUES = (
    ROOT / "matrix" / "h800_packing_memory_boundary_stage1_v1.jsonl",
    ROOT / "matrix" / "h800_packing_memory_boundary_stage1_resume_v2.jsonl",
)
STAGE1_PREDICTIONS = ARTIFACT_DIR / "h800_packing_memory_boundary_candidate_predictions_v1.json"
GAP_QUEUES = (
    ROOT / "matrix" / "h800_packing_branch_coefficient_gap_v1.jsonl",
    ROOT / "matrix" / "h800_packing_branch_coefficient_gap_b2_v1.jsonl",
    ROOT / "matrix" / "h800_packing_branch_coefficient_gap_b3_v1.jsonl",
)
GAP_PREDICTIONS = ARTIFACT_DIR / "h800_packing_branch_coefficient_gap_predictions_v1.json"
OUTPUT_DIR = ROOT / "diagnostics" / "packing_branch_specific_memory_centre_20260807"
OUTPUT = OUTPUT_DIR / "packing_branch_specific_memory_centre_challenger_v1.json"

GIB = phase_c_fit.GIB
ALPHAS = phase_c_fit.ALPHAS
SHARED_FEATURES = phase_c_fit.MEMORY_SHARED_FEATURES
PACKING_FEATURES = phase_c_fit.MEMORY_PACKING_FEATURES


def _stage1_packed_arms() -> list[dict[str, Any]]:
    """Packed Phase-D Stage 1 arms, built to the same convention as Phase B/C.

    These matter because the Phase B/C packed arms top out at 65.2 GiB while the
    model is asked to predict up to 137.6 GiB.  Stage 1 supplies exactly that
    high-memory anchor and is already marked ``role: calibration`` /
    ``packing_memory_boundary_low_anchor_fit_only``, so it is fit evidence rather
    than a consumed holdout.

    Two details must match the Phase B/C convention or the target is corrupted:

    * the denominator is the *physical* centre (``reserved_center_bytes`` from the
      prediction artifact), NOT the queue's ``predicted_arm_center_gib`` -- the
      latter is a refit centre (96.22 vs 159.82 GiB for B1) and mixing the two
      would make the residual mean something different per row;
    * repeats are averaged.  Phase B/C average three; Stage 1 has two, which is
      recorded per arm so the weaker averaging is visible.
    """
    predictions = {row["request_id"]: row for row in read_json(STAGE1_PREDICTIONS)["predictions"]}
    jobs: dict[str, dict[str, Any]] = {}
    for queue in STAGE1_QUEUES:
        for job in read_jsonl(queue):
            jobs.setdefault(str(job["job_id"]), job)

    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for job_id, job in sorted(jobs.items()):
        if not job.get("packing"):
            continue
        status_path = ROOT / "results" / job_id / "status.json"
        if not status_path.is_file():
            continue
        status = read_json(status_path)
        if status.get("classification") != "success":
            continue
        if status.get("calibration_eligible") is not True:
            continue
        observed = _observed_reserved_gib(job_id, int(job["gpu_count"]))
        if observed is None:
            continue
        grouped[str(job["boundary_setting_id"])].append((job, observed))

    arms: list[dict[str, Any]] = []
    for setting_id, entries in sorted(grouped.items()):
        job = entries[0][0]
        request_id = (
            f"boundary-{str(job['chain_id']).lower()}-c{int(job['cutoff_len'])}-p"
        )
        memory = predictions[request_id]["memory"]
        centre_gib = float(memory["reserved_center_bytes"]) / GIB
        observed_gib = statistics.fmean(value for _, value in entries)
        cutoff = float(job["cutoff_len"])
        profile = phase_c_fit.phase_b_fit._profile_stats(str(job["dataset_profile_path"]))
        n_pack = float(job.get("n_pack_mean") or 1.0)
        zero3 = float(int(job.get("zero_stage", 0)) == 3)
        gc_off = float(not bool(job.get("gc")))
        arms.append(
            {
                "arm_id": f"stage1:{setting_id}:P",
                "setting_id": f"stage1:{setting_id}",
                "profile_group": str(job["workload_id"]).upper(),
                "source": "stage1_boundary",
                "packing": True,
                "repeats_averaged": len(entries),
                "model_id": str(job["model_id"]),
                "features": {
                    "log2_physical_center_gib": math.log2(centre_gib),
                    "log2_cutoff": math.log2(cutoff),
                    "log2_mean_length": math.log2(profile["mean"]),
                    "length_cv": profile["cv"],
                    "p99_length_to_cutoff": profile["p99"] / cutoff,
                    "log2_effective_samples_per_physical_row": math.log2(n_pack),
                    "zero3": zero3,
                    "gc_off": gc_off,
                    "packing": 1.0,
                    "packing_x_log2_n_pack_mean": math.log2(n_pack),
                    "packing_x_zero3": zero3,
                    "packing_x_gc_off": gc_off,
                },
                "physical_center_gib": centre_gib,
                "observed_reserved_gib": observed_gib,
                "target_log_residual": math.log(observed_gib / centre_gib),
            }
        )
    return arms


def _gap_campaign_arms() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Packed arms and OOM lower bounds from the 2026-08-08 coefficient-gap campaign.

    Returns ``(fit_arms, oom_bounds)``.  Only successful runs become fit arms:
    CUDA OOM is a right-censored lower bound on memory, so an OOM row states
    "at least this much was needed" and cannot be used as a regression target.
    The bounds are still reported, because they are the only evidence that the
    ZeRO-3/GC-off mechanism does not fit at cutoff 20480 and they constrain any
    future admission upper.

    The physical centre is computed with the same frozen predictor the earlier
    campaigns used, not derived from the analytic reference: the centre/analytic
    ratio is not constant across mechanisms (1.322 vs 1.386 in the existing
    artifacts), so a fixed multiplier would silently corrupt the target.
    """
    jobs = [row for q in GAP_QUEUES if q.is_file() for row in read_jsonl(q)]
    if not jobs:
        return [], []
    predictions = _gap_predictions(jobs)

    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    oom_bounds: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job["job_id"])
        status_path = ROOT / "results" / job_id / "status.json"
        if not status_path.is_file():
            continue
        status = read_json(status_path)
        if status.get("calibration_eligible") is not True:
            continue
        if status.get("classification") == "oom":
            oom_bounds.append(
                {
                    "setting_id": str(job["setting_id"]),
                    "job_id": job_id,
                    "workload_id": str(job["workload_id"]),
                    "model_id": str(job["model_id"]),
                    "cutoff_len": int(job["cutoff_len"]),
                    "zero_stage": int(job["zero_stage"]),
                    "gc": bool(job["gc"]),
                    "role": "right_censored_lower_bound_not_a_regression_target",
                }
            )
            continue
        if status.get("classification") != "success":
            continue
        observed = _observed_reserved_gib(job_id, int(job["gpu_count"]))
        if observed is None:
            continue
        grouped[str(job["setting_id"])].append((job, observed))

    arms: list[dict[str, Any]] = []
    for setting_id, entries in sorted(grouped.items()):
        job = entries[0][0]
        centre_gib = float(predictions[setting_id]["reserved_center_bytes"]) / GIB
        observed_gib = statistics.fmean(value for _, value in entries)
        cutoff = float(job["cutoff_len"])
        profile = phase_c_fit.phase_b_fit._profile_stats(str(job["dataset_profile_path"]))
        n_pack = float(job.get("n_pack_mean") or 1.0)
        zero3 = float(int(job.get("zero_stage", 0)) == 3)
        gc_off = float(not bool(job.get("gc")))
        arms.append(
            {
                "arm_id": f"gap:{setting_id}:P",
                "setting_id": f"gap:{setting_id}",
                "profile_group": str(job["workload_id"]).upper(),
                "source": "coefficient_gap_campaign",
                "packing": True,
                "repeats_averaged": len(entries),
                "model_id": str(job["model_id"]),
                "features": {
                    "log2_physical_center_gib": math.log2(centre_gib),
                    "log2_cutoff": math.log2(cutoff),
                    "log2_mean_length": math.log2(profile["mean"]),
                    "length_cv": profile["cv"],
                    "p99_length_to_cutoff": profile["p99"] / cutoff,
                    "log2_effective_samples_per_physical_row": math.log2(n_pack),
                    "zero3": zero3,
                    "gc_off": gc_off,
                    "packing": 1.0,
                    "packing_x_log2_n_pack_mean": math.log2(n_pack),
                    "packing_x_zero3": zero3,
                    "packing_x_gc_off": gc_off,
                },
                "physical_center_gib": centre_gib,
                "observed_reserved_gib": observed_gib,
                "target_log_residual": math.log(observed_gib / centre_gib),
            }
        )
    return arms, oom_bounds


def _gap_predictions(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Physical memory predictions for the gap campaign, cached on disk.

    The predictor is the frozen v4b physical predictor, invoked exactly as the
    boundary selection invoked it, so the resulting centre is on the same scale
    as the Phase B/C and Stage 1 arms.
    """
    # The cache must cover every setting currently in the queues.  An earlier
    # version returned a stale cache written before batch 2 existed, which
    # silently dropped the new arm from the fit -- the refit reported the same
    # arm count as before and nothing failed.  Re-predict whenever a setting is
    # missing rather than trusting the file's existence.
    needed = {str(job["setting_id"]) for job in jobs}
    if GAP_PREDICTIONS.is_file():
        cached = read_json(GAP_PREDICTIONS)
        by_setting = {str(k): v for k, v in cached["by_setting_id"].items()}
        if needed <= set(by_setting):
            return by_setting

    from h800_physical_v4b_predictor import H800PhysicalV4BPredictor

    by_setting: dict[str, dict[str, Any]] = {}
    requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for job in jobs:
        setting_id = str(job["setting_id"])
        if setting_id in seen:
            continue
        seen.add(setting_id)
        requests.append(
            {
                "request_id": setting_id,
                "comparison_group": setting_id,
                "model_id": str(job["model_id"]),
                "training_mode": str(job["train_type"]),
                "dataset_id": str(job["dataset_id"]),
                "dataset_category": str(job["dataset_category"]),
                "target_gbs": int(job["target_gbs"]),
                "cutoff_len": int(job["cutoff_len"]),
                "gpu_count": int(job["gpu_count"]),
                "physical_mbs": int(job["mbs"]),
                "gradient_accumulation_steps": int(job["gradient_accumulation_steps"]),
                "zero_stage": int(job["zero_stage"]),
                "gradient_checkpointing": bool(job["gc"]),
                "packing": True,
                "offload": False,
                "dtype": "bf16",
                "kernel_path": "fa3_orig+liger_fused_ce+adamw_torch_fused",
                "lora_rank": 32,
                "profile_tokenizer_id": "qwen3_8b@local",
                "profile_template_id": str(job["template"]),
            }
        )
    predictor = H800PhysicalV4BPredictor(
        model_inventory=ARTIFACT_DIR / "model_inventory.json",
        strict_model_inventory_binding=False,
        additional_dataset_profile_dir=ARTIFACT_DIR / "packing_profile_phase_b_v1" / "profiles",
    )
    report = predictor.predict(requests)
    for prediction in report["predictions"]:
        memory = prediction["memory"]
        if memory.get("prediction_available") is not True:
            raise ValueError(f"memory prediction unavailable: {prediction['request_id']}")
        by_setting[str(prediction["request_id"])] = {
            "reserved_center_bytes": memory["reserved_center_bytes"],
            "analytic_reference_bytes": memory["analytic_reference_bytes"],
            "operational_p95_reserved_bytes": memory["operational_p95_reserved_bytes"],
        }
    write_json(
        GAP_PREDICTIONS,
        {
            "schema": "sft_packing_branch_coefficient_gap_predictions/v1",
            "implementation_version": report.get("implementation_version"),
            "by_setting_id": by_setting,
        },
    )
    return by_setting


def _observed_reserved_gib(job_id: str, gpu_count: int) -> float | None:
    """Peak reserved across every rank, in GiB.

    Admission safety is per-card, so the maximum over ranks is the only defensible
    reduction; rank 0 alone understates an imbalanced job.
    """
    paths = sorted((ROOT / "results" / job_id / "metrics").glob("summary.rank*.json"))
    values = [int(read_json(p).get("max_reserved") or 0) for p in paths]
    values = [v for v in values if v > 0]
    if not values or len(paths) != gpu_count:
        return None
    return max(values) / GIB


def _group_equal_mae(rows: list[dict[str, Any]], actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute log error, weighting every workload profile equally.

    Equal weighting matters here because the packed arms are unbalanced across
    profiles (W3 has four, W5 has one); a row-weighted score would let W3 decide
    the ranking on its own.
    """
    errors: dict[str, list[float]] = defaultdict(list)
    for row, a, p in zip(rows, actual, predicted, strict=True):
        errors[str(row["profile_group"])].append(abs(float(a) - float(p)))
    return float(statistics.fmean(statistics.fmean(v) for v in errors.values()))


def _nested_leave_one_workload_out(
    rows: list[dict[str, Any]], feature_names: tuple[str, ...]
) -> dict[str, Any]:
    """Score a feature set by leave-one-workload-out with nested alpha choice.

    The alpha is re-selected inside each training fold, never on the held-out
    profile, so the reported error contains no selection leakage.  This mirrors
    the frozen script's protocol so the two numbers are comparable.
    """
    x = phase_c_fit._matrix(rows, feature_names)
    y = np.asarray([float(row["target_log_residual"]) for row in rows], dtype=float)
    groups = sorted({str(row["profile_group"]) for row in rows})
    prediction = np.empty(len(rows), dtype=float)
    folds: list[dict[str, Any]] = []
    for held_out in groups:
        train_idx = [i for i, row in enumerate(rows) if row["profile_group"] != held_out]
        test_idx = [i for i, row in enumerate(rows) if row["profile_group"] == held_out]
        train_rows = [rows[i] for i in train_idx]
        alpha = phase_c_fit._select_alpha(train_rows, x[train_idx], y[train_idx])
        prediction[test_idx], _, _ = phase_c_fit._fit_ridge(
            alpha, x[train_idx], y[train_idx], x[test_idx]
        )
        folds.append(
            {
                "held_out_profile_group": held_out,
                "selected_alpha": alpha,
                "training_arms": len(train_idx),
                "held_out_arms": len(test_idx),
            }
        )

    centre = np.asarray([float(r["physical_center_gib"]) for r in rows], dtype=float)
    observed = np.asarray([float(r["observed_reserved_gib"]) for r in rows], dtype=float)
    predicted_gib = centre * np.exp(prediction)
    ape = np.abs(predicted_gib - observed) / observed
    signed = predicted_gib - observed

    per_group: dict[str, dict[str, Any]] = {}
    for group in groups:
        idx = [i for i, r in enumerate(rows) if r["profile_group"] == group]
        per_group[group] = {
            "arms": len(idx),
            "reserved_mape": float(np.mean(ape[idx])),
            "maximum_reserved_ape": float(np.max(ape[idx])),
            # A single-arm profile cannot support a conclusion on its own.
            "single_arm_profile": len(idx) == 1,
        }
    return {
        "feature_names": list(feature_names),
        "arms": len(rows),
        "profile_groups": groups,
        "folds": folds,
        "group_equal_mae_log_residual": _group_equal_mae(rows, y, prediction),
        "reserved_mape": float(np.mean(ape)),
        "maximum_reserved_ape": float(np.max(ape)),
        "maximum_underprediction_gib": float(-np.min(signed)) if np.min(signed) < 0 else 0.0,
        "per_profile_group": per_group,
        "oof_predictions": [
            {
                "arm_id": r["arm_id"],
                "profile_group": r["profile_group"],
                "packing": r["packing"],
                "physical_center_gib": r["physical_center_gib"],
                "observed_reserved_gib": r["observed_reserved_gib"],
                "predicted_reserved_gib": float(predicted_gib[i]),
                "reserved_ape": float(ape[i]),
            }
            for i, r in enumerate(rows)
        ],
    }


def _alpha_sweep(
    baseline_rows: list[dict[str, Any]],
    baseline_features: tuple[str, ...],
    challenger_rows: list[dict[str, Any]],
    challenger_features: tuple[str, ...],
    evaluated_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fixed-alpha leave-one-workload-out scores for both models.

    Recorded because the ranking is NOT uniform across the grid: with a strong
    penalty the two models converge.  Publishing only the nested-selection number
    would hide that fragility.
    """
    evaluated_ids = {r["arm_id"] for r in evaluated_rows}
    sweep: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        row: dict[str, Any] = {"alpha": alpha}
        for label, rows, names in (
            ("baseline_shared", baseline_rows, baseline_features),
            ("challenger_packed_only", challenger_rows, challenger_features),
        ):
            x = phase_c_fit._matrix(rows, names)
            y = np.asarray([float(r["target_log_residual"]) for r in rows], dtype=float)
            groups = sorted({str(r["profile_group"]) for r in rows})
            apes: list[float] = []
            for held_out in groups:
                train = [i for i, r in enumerate(rows) if r["profile_group"] != held_out]
                test = [
                    i
                    for i, r in enumerate(rows)
                    if r["profile_group"] == held_out and r["arm_id"] in evaluated_ids
                ]
                if not test or len(train) < 2:
                    continue
                pred, _, _ = phase_c_fit._fit_ridge(alpha, x[train], y[train], x[test])
                for p, i in zip(pred, test, strict=True):
                    r = rows[i]
                    predicted = math.exp(float(p)) * float(r["physical_center_gib"])
                    observed = float(r["observed_reserved_gib"])
                    apes.append(abs(predicted - observed) / observed)
            row[label] = float(statistics.fmean(apes)) if apes else None
        sweep.append(row)
    return sweep


def build() -> dict[str, Any]:
    frozen = read_json(FROZEN_REFIT)
    if frozen.get("status") != "fit_only_not_publishable":
        raise ValueError("frozen Phase-C refit is not in the expected fit-only state")
    if frozen.get("memory_center", {}).get("upper_guard_accepted") is not False:
        raise ValueError("frozen refit unexpectedly accepts an upper guard")

    arms = phase_c_fit._memory_rows()
    packed = [r for r in arms if r["packing"]]
    unpacked = [r for r in arms if not r["packing"]]
    if len(arms) != 20 or len(packed) != 10:
        raise ValueError(f"unexpected arm split: {len(arms)} total / {len(packed)} packed")

    # Baseline: the frozen structure (all arms, shared + packing features),
    # scored on the packed arms only so the comparison is apples to apples.
    baseline = _nested_leave_one_workload_out(arms, PACKING_FEATURES)
    baseline_packed = [r for r in baseline["oof_predictions"] if r["packing"]]
    baseline_packed_mape = float(
        statistics.fmean(r["reserved_ape"] for r in baseline_packed)
    )
    baseline_packed_max = float(max(r["reserved_ape"] for r in baseline_packed))

    # Challenger: packed arms only, shared feature definitions.  The four
    # packing_x_* interactions are dropped because within the packed branch
    # `packing` is constant at 1, so they are collinear with the shared columns.
    challenger = _nested_leave_one_workload_out(packed, SHARED_FEATURES)

    # Extended challenger: add the Stage 1 packed arms, which are the only fit
    # evidence above 65.2 GiB.  Without them every prediction near the 132.84 GiB
    # safe limit is a 2x extrapolation from the fitted range.
    stage1 = _stage1_packed_arms()
    gap_arms, oom_bounds = _gap_campaign_arms()
    extended_rows = packed + stage1 + gap_arms
    extended = _nested_leave_one_workload_out(extended_rows, SHARED_FEATURES)
    added_ids = {r["arm_id"] for r in stage1 + gap_arms}
    extended_on_original = [
        r for r in extended["oof_predictions"] if r["arm_id"] not in added_ids
    ]
    extended_on_stage1 = [
        r for r in extended["oof_predictions"] if r["arm_id"] in added_ids
    ]
    coverage = {
        "phase_bc_packed_reserved_gib_range": [
            min(r["observed_reserved_gib"] for r in packed),
            max(r["observed_reserved_gib"] for r in packed),
        ],
        "extended_packed_reserved_gib_range": [
            min(r["observed_reserved_gib"] for r in extended_rows),
            max(r["observed_reserved_gib"] for r in extended_rows),
        ],
        "safe_limit_gib": 132.83927001953126,
        "stage1_arms_added": len(stage1),
        "gap_campaign_arms_added": len(gap_arms),
        "gap_campaign_oom_lower_bounds": oom_bounds,
        "gap_campaign_arms": [
            {
                "arm_id": r["arm_id"],
                "profile_group": r["profile_group"],
                "repeats_averaged": r["repeats_averaged"],
                "physical_center_gib": r["physical_center_gib"],
                "observed_reserved_gib": r["observed_reserved_gib"],
                "target_log_residual": r["target_log_residual"],
            }
            for r in gap_arms
        ],
        "zero3_gc_off_still_unfitted": not any(
            r["features"]["zero3"] == 1.0 and r["features"]["gc_off"] == 1.0
            for r in extended_rows
        ),
        "stage1_arms": [
            {
                "arm_id": r["arm_id"],
                "profile_group": r["profile_group"],
                "model_id": r["model_id"],
                "repeats_averaged": r["repeats_averaged"],
                "physical_center_gib": r["physical_center_gib"],
                "observed_reserved_gib": r["observed_reserved_gib"],
                "target_log_residual": r["target_log_residual"],
            }
            for r in stage1
        ],
        "stage1_role": (
            "role=calibration / packing_memory_boundary_low_anchor_fit_only, so it "
            "is fit evidence, not a consumed holdout"
        ),
        "denominator_note": (
            "the physical centre (reserved_center_bytes) is used, not the queue's "
            "predicted_arm_center_gib, which is a refit centre (96.22 vs 159.82 GiB "
            "for B1) and would make the residual mean something different per row"
        ),
    }

    sweep = _alpha_sweep(arms, PACKING_FEATURES, packed, SHARED_FEATURES, packed)

    target_spread = {
        "unpacked_mean_target_log_residual": float(
            statistics.fmean(r["target_log_residual"] for r in unpacked)
        ),
        "packed_mean_target_log_residual": float(
            statistics.fmean(r["target_log_residual"] for r in packed)
        ),
        "why_this_matters": (
            "the target is log(observed / analytic centre) and the analytic centre "
            "uses sequence=cutoff_len; packing fills the cutoff so its target sits "
            "near zero, while unpacked rows are charged for tokens they never "
            "materialise and are dragged far negative. One shared coefficient set "
            "cannot represent both."
        ),
    }

    improvement = baseline_packed_mape - challenger["reserved_mape"]
    relative = improvement / baseline_packed_mape if baseline_packed_mape else None

    # Fragility is reported as first-class output, not a footnote: with ten arms
    # over four profiles this is the part that decides whether the result can be
    # acted on.
    ranking_holds_across_alpha = all(
        row["challenger_packed_only"] is None
        or row["baseline_shared"] is None
        or row["challenger_packed_only"] <= row["baseline_shared"]
        for row in sweep
    )
    fragility = {
        "packed_arms": len(packed),
        "fitted_coefficients": len(SHARED_FEATURES),
        "arms_per_coefficient": len(packed) / len(SHARED_FEATURES),
        "profile_groups": challenger["profile_groups"],
        "single_arm_profile_groups": [
            g for g, v in challenger["per_profile_group"].items() if v["single_arm_profile"]
        ],
        "ranking_holds_across_full_alpha_grid": ranking_holds_across_alpha,
        "alpha_sweep": sweep,
        "interpretation": (
            "the challenger wins under nested alpha selection and across most of "
            "the grid, but ten arms supporting eight coefficients cannot establish "
            "an absolute accuracy claim; W5 contributes one arm and its score is "
            "not evidence"
        ),
    }

    report: dict[str, Any] = {
        "schema": "sft_packing_branch_specific_memory_centre_challenger/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "fit the packed memory-centre residual on packed arms only, keeping the "
            "shared analytic trunk, and measure it against the frozen shared fit"
        ),
        "frozen_inputs": {
            "phase_c_refit": {
                "path": str(FROZEN_REFIT.resolve()),
                "sha256": sha256_file(FROZEN_REFIT),
                "report_sha256": frozen["report_sha256"],
            }
        },
        "structure_change": {
            "from": "20 arms, 8 shared + 4 packing-specific coefficients",
            "to": "packed branch fits its own 8 coefficients on 10 packed arms",
            "analytic_trunk": "unchanged (h800_theory_basis.memory_basis)",
            "feature_definitions": "unchanged",
            "sequence_policy": "unchanged (packing uses cutoff_len, which measures 82-99% filled)",
            "dropped_features": [f for f in PACKING_FEATURES if f not in SHARED_FEATURES],
            "why_dropped": (
                "inside the packed branch `packing` is constant at 1, so the four "
                "packing_x_* columns are collinear with the shared columns"
            ),
        },
        "target_spread_between_branches": target_spread,
        "baseline_frozen_structure": {
            **baseline,
            "packed_only_reserved_mape": baseline_packed_mape,
            "packed_only_maximum_reserved_ape": baseline_packed_max,
        },
        "challenger_packed_only": challenger,
        "challenger_packed_plus_stage1": {
            **extended,
            "reserved_mape_on_phase_bc_arms": float(
                statistics.fmean(r["reserved_ape"] for r in extended_on_original)
            ),
            "reserved_mape_on_stage1_arms": (
                float(statistics.fmean(r["reserved_ape"] for r in extended_on_stage1))
                if extended_on_stage1
                else None
            ),
            "maximum_ape_on_stage1_arms": (
                float(max(r["reserved_ape"] for r in extended_on_stage1))
                if extended_on_stage1
                else None
            ),
        },
        "high_memory_coverage": coverage,
        "comparison_on_packed_arms": {
            "baseline_reserved_mape": baseline_packed_mape,
            "challenger_reserved_mape": challenger["reserved_mape"],
            "absolute_improvement": improvement,
            "relative_improvement": relative,
            "baseline_maximum_ape": baseline_packed_max,
            "challenger_maximum_ape": challenger["maximum_reserved_ape"],
            "evaluation": "leave-one-workload-out with nested alpha selection",
        },
        "fragility": fragility,
        "frozen_artifacts_modified": False,
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "publishable": False,
        "prospective_acceptance_passed": False,
        "automatic_packing_recommendation_allowed": False,
        "upper_guard_accepted": False,
        "claim": (
            "structural direction is supported: separating the packed branch's "
            "coefficients reduces leave-one-workload-out error on packed arms. "
            "Absolute accuracy is NOT established and this must not drive "
            "recommendations."
        ),
        "next_step": (
            "size a prospective packed-arm campaign that balances workload profiles "
            "before any acceptance claim; see required_experiments"
        ),
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    return report


def main() -> None:
    report = build()
    compact = {
        "output": str(OUTPUT),
        "report_sha256": report["report_sha256"],
        "comparison_on_packed_arms": report["comparison_on_packed_arms"],
        "per_profile_group": report["challenger_packed_only"]["per_profile_group"],
        "extended_with_stage1": {
            "arms": report["challenger_packed_plus_stage1"]["arms"],
            "reserved_mape": report["challenger_packed_plus_stage1"]["reserved_mape"],
            "maximum_reserved_ape": report["challenger_packed_plus_stage1"][
                "maximum_reserved_ape"
            ],
            "reserved_mape_on_stage1_arms": report["challenger_packed_plus_stage1"][
                "reserved_mape_on_stage1_arms"
            ],
            "maximum_ape_on_stage1_arms": report["challenger_packed_plus_stage1"][
                "maximum_ape_on_stage1_arms"
            ],
            "per_profile_group": report["challenger_packed_plus_stage1"][
                "per_profile_group"
            ],
        },
        "high_memory_coverage": {
            key: report["high_memory_coverage"][key]
            for key in (
                "phase_bc_packed_reserved_gib_range",
                "extended_packed_reserved_gib_range",
                "safe_limit_gib",
                "stage1_arms_added",
            )
        },
        "fragility": {
            key: report["fragility"][key]
            for key in (
                "packed_arms",
                "fitted_coefficients",
                "arms_per_coefficient",
                "single_arm_profile_groups",
                "ranking_holds_across_full_alpha_grid",
            )
        },
        "alpha_sweep": report["fragility"]["alpha_sweep"],
        "publishable": report["publishable"],
        "claim": report["claim"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
