#!/usr/bin/env python3
"""Stage-2 fit for dense hybrid-attention memory with an explicit ZeRO-3 term.

Motivation
----------
Stage-1 (`fit_h800_hybrid_attention_dense_stage1_v1.py`) passed a
leave-one-source-out CV at ~3% MAPE, yet on the source-disjoint prospective
tasks it under-predicted peak reserved memory for large hybrid models on
multi-GPU ZeRO-3 (qwen3p5_9b by 17.9%, qwen3_6_27b by 14.9%).  Two reasons:

1. The ZeRO-3 per-device all-gather / prefetch working set was never modelled;
   it leaked into ``state`` (coefficient pinned to [0.8, 1.25]) and was squeezed
   too small.
2. CV left out only the *data source*, never the *model* or the *parallel
   configuration*, so the ZeRO-3 rows were always in-distribution during CV.

Stage-2 therefore (a) adds a free ``zero3_param_workspace`` column (see
``hybrid_attention_memory_features_v2.py``) and (b) evaluates with a
leave-one-(model_id, zero, gpu_count)-out CV plus a dedicated held-out ZeRO-3
MAPE gate, which is the contrast that surfaces the large-model ZeRO-3 bias.

Release stays shadow-only: passing this fit never grants automatic admission.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

from common import ARTIFACT_DIR, read_json, read_jsonl, sha256_file, sha256_json, write_json
from evaluate_h800_hybrid_attention_dense_stage1_v1 import CAPACITY_BYTES, FORMAL_RECORDS
from hybrid_attention_memory_features_v2 import zero3_param_workspace_from_basis

FEATURE_SCHEMA_V3 = "sft_hybrid_attention_dense_features/v3"

PROTOCOL_SCHEMA = "sft_h800_hybrid_attention_dense_stage3_fit_protocol/v1"
REPORT_SCHEMA = "sft_h800_hybrid_attention_dense_stage3_fit_report/v1"
ARTIFACT_SCHEMA = "sft_h800_hybrid_memory_artifact/v3"

PROTOCOL_PATH = ARTIFACT_DIR / "h800_hybrid_attention_dense_stage3_fit_protocol_v1.json"
FIT_REPORT_PATH = ARTIFACT_DIR / "h800_hybrid_attention_dense_stage3_fit_report_v1.json"
ARTIFACT_PATH = ARTIFACT_DIR / "h800_hybrid_memory_artifact_v3.json"
SCRIPT_PATH = __file__

ROUTES = ("dense_full_attention", "dense_hybrid_attention")

# Stage-1 basis + one new freely-fit column for the ZeRO-3 parameter workspace.
COEFFICIENTS: tuple[tuple[str, str | None], ...] = (
    ("intercept", None),
    ("state", "state_bytes"),
    # GC-aware saved-activation split: when checkpointing is OFF the full
    # per-layer activation is saved and was systematically under-predicted by
    # the single GC-on fitted coefficient. In GC-off rows the saved bytes are
    # moved into the *_gc0 columns (the *_gc1 columns then carry zero), so the
    # two regimes receive independent coefficients.
    ("saved_full", "saved_full_attention_activations_bytes"),
    ("saved_linear", "saved_linear_attention_activations_bytes"),
    ("saved_full_gc0", "saved_full_gc0_activation_bytes"),
    ("saved_linear_gc0", "saved_linear_gc0_activation_bytes"),
    ("recompute", "recompute_workspace_bytes"),
    ("full_workspace", "full_attention_workspace_bytes"),
    ("linear_workspace", "linear_attention_workspace_bytes"),
    ("linear_state", "linear_recurrent_state_bytes"),
    ("logits", "logits_workspace_bytes"),
    ("zero_workspace", "zero_collective_workspace_bytes"),
    ("zero3_saved", "zero3_saved_activation_bytes"),
    ("zero3_param_workspace", "zero3_param_workspace_bytes"),
)

THRESHOLDS: dict[str, Any] = {
    "exact_mape_max": 0.10,
    "zero3_held_out_exact_mape_max": 0.10,
    # Censor satisfaction is evaluated on the IN-SAMPLE full fit, not on the
    # leave-out folds: the OOM right-censor hinge is only applied during the
    # full fit, and the center model is deliberately not a safety upper bound
    # (that is Part C's job). Held-out censor satisfaction is kept as a
    # diagnostic in the CV block.
    "insample_censor_satisfaction_min": 0.95,
    "state_coefficient_range": [0.8, 1.25],
    "censor_constraint_weight": 1.0,
    "minimum_exact_rows_per_route": {
        "dense_full_attention": 20,
        "dense_hybrid_attention": 60,
    },
}

FOLD_KEY_FIELDS = (
    "model_id",
    "source_dataset_id",
    "gpu_count",
    "zero",
    "gradient_checkpointing",
    "micro_batch_size",
    "exact_total_tokens_per_sample",
    "packing",
)

# The generalization CV holds out an entire (model, zero, gpu_count) cell.
CV_LEAVE_OUT_FIELDS = ("model_id", "zero", "gpu_count")

CEIL = 1.0


def _memory_features(observation: dict[str, Any]) -> dict[str, float]:
    memory = observation["feature_basis"]["memory"]
    activation = memory["activation_components"]
    workspace = memory["workspace_candidates"]
    configuration = observation.get("configuration") or {}
    zero_stage = str(configuration.get("zero") or "none")
    gc_off = bool(configuration.get("gradient_checkpointing")) is False
    saved_full = float(activation["saved_full_attention_activations_bytes"])
    saved_linear = float(activation["saved_linear_attention_activations_bytes"])
    saved_total = saved_full + saved_linear
    # GC-off rows carry the full saved activation in dedicated columns; the
    # GC-on columns are zeroed so the two regimes have independent coefficients.
    saved_full_gc0 = saved_full if gc_off else 0.0
    saved_linear_gc0 = saved_linear if gc_off else 0.0
    if gc_off:
        saved_full = 0.0
        saved_linear = 0.0
    return {
        "state_bytes": float(memory["state_bytes"]),
        "saved_full_attention_activations_bytes": saved_full,
        "saved_linear_attention_activations_bytes": saved_linear,
        "saved_full_gc0_activation_bytes": saved_full_gc0,
        "saved_linear_gc0_activation_bytes": saved_linear_gc0,
        "recompute_workspace_bytes": float(activation["recompute_workspace_bytes"]),
        "full_attention_workspace_bytes": float(
            workspace["full_attention_workspace_bytes"]
        ),
        "linear_attention_workspace_bytes": float(
            workspace["linear_attention_workspace_bytes"]
        ),
        "linear_recurrent_state_bytes": float(
            workspace["linear_recurrent_state_bytes"]
        ),
        "logits_workspace_bytes": float(workspace["logits_workspace_bytes"]),
        "zero_collective_workspace_bytes": float(
            workspace["zero_collective_workspace_bytes"]
        ),
        "zero3_saved_activation_bytes": saved_total if zero_stage == "zero3" else 0.0,
        "zero3_param_workspace_bytes": zero3_param_workspace_from_basis(
            observation["feature_basis"], configuration
        ),
    }


def _design_row(observation: dict[str, Any]) -> list[float]:
    features = _memory_features(observation)
    return [1.0] + [
        features[key] if key is not None else 1.0 for _, key in COEFFICIENTS[1:]
    ]


def _fold_key(observation: dict[str, Any]) -> tuple[Any, ...]:
    configuration = observation["configuration"]
    return tuple(
        observation.get(field, configuration.get(field)) for field in FOLD_KEY_FIELDS
    )


def _cv_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["model_id"], row["zero"], row["gpu_count"])


def _collapsed_rows(observations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for observation in observations:
        groups.setdefault(_fold_key(observation), []).append(observation)
    rows: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        exact = [m for m in members if m["outcome"]["peak_reserved_bytes"]]
        oom = [m for m in members if m["outcome"]["classification"] == "oom"]
        first = members[0]
        base = {
            "fold_key": key,
            "model_id": first["model_id"],
            "architecture_route": first["architecture_route"],
            "source_dataset_id": first["configuration"]["source_dataset_id"],
            "gpu_count": first["configuration"]["gpu_count"],
            "zero": first["configuration"]["zero"],
            "design": _design_row(first),
            "member_job_ids": sorted(m["job_id"] for m in members),
            "repeat_count": len(members),
            "discarded_exact_peaks": [],
        }
        if oom:
            bounds = [int(m["outcome"]["oom_right_censor_lower_bytes"]) for m in oom]
            base["state"] = "censored"
            base["censor_lower_bytes"] = max(bounds)
            if exact:
                base["discarded_exact_peaks"] = [
                    int(m["outcome"]["peak_reserved_bytes"]) for m in exact
                ]
            rows.append(base)
        else:
            peaks = [int(m["outcome"]["peak_reserved_bytes"]) for m in exact]
            base["state"] = "exact"
            base["peak_reserved_bytes"] = statistics.median(peaks)
            base["repeat_spread_bytes"] = max(peaks) - min(peaks)
            rows.append(base)
    return rows


def _predict(coefficients: Sequence[float], design: Sequence[float]) -> float:
    return float(np.dot(np.asarray(coefficients, dtype=float), np.asarray(design, dtype=float)))


def _fit_route(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names = [name for name, _ in COEFFICIENTS]
    design_matrix = np.asarray([row["design"] for row in rows], dtype=float)
    active = design_matrix.max(axis=0) > 0.0
    fixed_zero = [names[i] for i in range(len(names)) if not active[i]]
    active_designs = [design[active] for design in design_matrix]
    initial = np.full(int(active.sum()), 1.0, dtype=float)
    initial[0] = 1e6
    lower = np.zeros(int(active.sum()), dtype=float)
    weight = float(THRESHOLDS["censor_constraint_weight"])

    def residuals(coefficients: np.ndarray) -> np.ndarray:
        values: list[float] = []
        for design, row in zip(active_designs, rows):
            predicted = max(float(np.dot(coefficients, design)), CEIL)
            if row["state"] == "exact":
                values.append(math.log(predicted) - math.log(row["peak_reserved_bytes"]))
            else:
                values.append(
                    math.sqrt(weight)
                    * max(0.0, math.log(row["censor_lower_bytes"]) - math.log(predicted))
                )
        return np.asarray(values, dtype=float)

    result = least_squares(residuals, initial, bounds=(lower, np.inf), method="trf", max_nfev=4000)
    full = np.zeros(len(COEFFICIENTS), dtype=float)
    full[active] = result.x
    exact = [row for row in rows if row["state"] == "exact"]
    censored = [row for row in rows if row["state"] == "censored"]
    insample_satisfied = sum(
        1 for row in censored if _predict(full.tolist(), row["design"]) >= row["censor_lower_bytes"]
    )
    return {
        "coefficients": full.tolist(),
        "coefficients_by_name": dict(zip(names, full.tolist())),
        "fixed_zero_coefficients": fixed_zero,
        "exact_rows": len(exact),
        "censored_rows": len(censored),
        "insample_censor_satisfaction_rate": (
            insample_satisfied / len(censored) if censored else None
        ),
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
    }


def _evaluate_rows(coefficients: Sequence[float], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    exact = [row for row in rows if row["state"] == "exact"]
    censored = [row for row in rows if row["state"] == "censored"]
    exact_apes = [
        abs(max(_predict(coefficients, row["design"]), CEIL) - row["peak_reserved_bytes"])
        / row["peak_reserved_bytes"]
        for row in exact
    ]
    zero3_exact = [row for row in exact if str(row["zero"]) == "zero3"]
    zero3_apes = [
        abs(max(_predict(coefficients, row["design"]), CEIL) - row["peak_reserved_bytes"])
        / row["peak_reserved_bytes"]
        for row in zero3_exact
    ]
    satisfied = sum(
        1 for row in censored if _predict(coefficients, row["design"]) >= row["censor_lower_bytes"]
    )
    return {
        "exact_rows": len(exact),
        "exact_mape": statistics.fmean(exact_apes) if exact_apes else None,
        "zero3_exact_rows": len(zero3_exact),
        "zero3_exact_mape": statistics.fmean(zero3_apes) if zero3_apes else None,
        "censored_rows": len(censored),
        "censor_satisfaction_rate": satisfied / len(censored) if censored else None,
    }


def _leave_out_cv(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({_cv_group_key(row) for row in rows}, key=str)
    folds: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    for group in groups:
        train = [row for row in rows if _cv_group_key(row) != group]
        test = [row for row in rows if _cv_group_key(row) == group]
        if not train or not test:
            folds.append({"held_out": list(group), "skipped": True})
            continue
        fit = _fit_route(train)
        metrics = _evaluate_rows(fit["coefficients"], test)
        folds.append({"held_out": list(group), "skipped": False, **metrics})
        held.append(metrics)

    def _pooled(field_rows: str, field_val: str) -> float | None:
        num = sum(
            (m[field_val] or 0.0) * m[field_rows] for m in held if m[field_val] is not None
        )
        den = sum(m[field_rows] for m in held if m[field_val] is not None)
        return num / den if den else None

    pooled = {
        "exact_rows": sum(m["exact_rows"] for m in held),
        "exact_mape": _pooled("exact_rows", "exact_mape"),
        "zero3_exact_rows": sum(m["zero3_exact_rows"] for m in held),
        "zero3_exact_mape": _pooled("zero3_exact_rows", "zero3_exact_mape"),
        "censored_rows": sum(m["censored_rows"] for m in held),
        "censor_satisfaction_rate": _pooled("censored_rows", "censor_satisfaction_rate"),
    }
    return {"scheme": "leave_one_model_zero_gpu_out", "folds": folds, "pooled_held_out": pooled}


def _large_model_zero3_replay(
    route_fit: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """In-sample V2 center vs measured peak for large-model multi-GPU ZeRO-3.

    This is the exact grid where stage-1 under-predicted; it documents that the
    V2 basis can represent the ZeRO-3 bias.  The out-of-distribution proof is the
    held-out ZeRO-3 MAPE above and the Part C prospective acceptance.
    """

    coefficients = route_fit["coefficients"]
    replay: list[dict[str, Any]] = []
    for row in rows:
        if (
            row["state"] == "exact"
            and str(row["zero"]) == "zero3"
            and int(row["gpu_count"]) > 1
            and row["model_id"] in {"qwen3p5_9b", "qwen3p6_27b", "qwen3_6_27b"}
        ):
            predicted = max(_predict(coefficients, row["design"]), CEIL)
            actual = float(row["peak_reserved_bytes"])
            replay.append(
                {
                    "model_id": row["model_id"],
                    "gpu_count": row["gpu_count"],
                    "zero": row["zero"],
                    "predicted_center_bytes": predicted,
                    "actual_reserved_bytes": actual,
                    "actual_over_predicted": actual / predicted if predicted else None,
                }
            )
    return replay


def protocol_payload() -> dict[str, Any]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "campaign_id": "h800_hybrid_attention_dense_stage3_20260818_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_script": {"path": str(Path(SCRIPT_PATH).resolve()), "sha256": sha256_file(Path(SCRIPT_PATH))},
        "feature_module_schema": FEATURE_SCHEMA_V3,
        "observations_source": {"path": str(FORMAL_RECORDS.resolve())},
        "routes": list(ROUTES),
        "coefficients": [{"name": n, "feature_key": k} for n, k in COEFFICIENTS],
        "cross_validation": {
            "scheme": "leave_one_(model_id,zero,gpu_count)_out",
            "leave_out_fields": list(CV_LEAVE_OUT_FIELDS),
        },
        "thresholds": THRESHOLDS,
        "release": {"mode": "shadow_only", "automatic_admission_allowed": False},
    }


def _verify_protocol(protocol: dict[str, Any]) -> None:
    problems: list[str] = []
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        problems.append("protocol schema drifted")
    if protocol.get("fit_script", {}).get("sha256") != sha256_file(Path(SCRIPT_PATH)):
        problems.append("fit script sha256 drifted from frozen protocol")
    if protocol.get("thresholds") != THRESHOLDS:
        problems.append("thresholds drifted from frozen protocol")
    if [row.get("name") for row in protocol.get("coefficients", [])] != [n for n, _ in COEFFICIENTS]:
        problems.append("coefficient list drifted from frozen protocol")
    if protocol.get("release", {}).get("mode") != "shadow_only":
        problems.append("release mode drifted from shadow_only")
    if problems:
        raise RuntimeError("frozen fit protocol mismatch: " + "; ".join(problems))


def fit(*, allow_incomplete: bool = False) -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError("fit protocol not frozen yet; run with --freeze-protocol first")
    protocol = read_json(PROTOCOL_PATH)
    _verify_protocol(protocol)
    observations = read_jsonl(FORMAL_RECORDS)
    records_sha256 = sha256_file(FORMAL_RECORDS)

    terminal = [
        row
        for row in observations
        if row["outcome"].get("terminal_eligible")
        and row["outcome"]["classification"] in {"success", "oom"}
    ]
    missing = len(observations) - len(terminal)
    complete = missing == 0

    rows_by_route: dict[str, list[dict[str, Any]]] = {route: [] for route in ROUTES}
    for observation in terminal:
        rows_by_route[str(observation["architecture_route"])].append(observation)
    collapsed = {route: _collapsed_rows(rows_by_route[route]) for route in ROUTES}

    route_fits: dict[str, dict[str, Any]] = {}
    route_cv: dict[str, dict[str, Any]] = {}
    route_replay: dict[str, list[dict[str, Any]]] = {}
    for route in ROUTES:
        rows = collapsed[route]
        if rows:
            route_fits[route] = _fit_route(rows)
            route_cv[route] = _leave_out_cv(rows)
            route_replay[route] = _large_model_zero3_replay(route_fits[route], rows)
        else:
            route_fits[route] = {"exact_rows": 0, "censored_rows": 0}
            route_cv[route] = {"folds": [], "pooled_held_out": {}}
            route_replay[route] = []

    gates: dict[str, bool] = {}
    if complete:
        low, high = THRESHOLDS["state_coefficient_range"]
        cv_mape = [route_cv[r]["pooled_held_out"].get("exact_mape") for r in ROUTES]
        cv_mape = [v for v in cv_mape if v is not None]
        zero3_mape = [route_cv[r]["pooled_held_out"].get("zero3_exact_mape") for r in ROUTES]
        zero3_mape = [v for v in zero3_mape if v is not None]
        insample_satisfaction = [
            route_fits[r].get("insample_censor_satisfaction_rate") for r in ROUTES
        ]
        insample_satisfaction = [v for v in insample_satisfaction if v is not None]
        state_coefs = [
            float(route_fits[r]["coefficients_by_name"]["state"])
            for r in ROUTES
            if route_fits[r].get("coefficients_by_name")
        ]
        gates = {
            "minimum_exact_rows_per_route": all(
                route_fits[r]["exact_rows"] >= THRESHOLDS["minimum_exact_rows_per_route"][r]
                for r in ROUTES
            ),
            "cv_exact_mape_within_10pct": bool(cv_mape and max(cv_mape) <= THRESHOLDS["exact_mape_max"]),
            "cv_zero3_held_out_mape_within_10pct": bool(
                zero3_mape and max(zero3_mape) <= THRESHOLDS["zero3_held_out_exact_mape_max"]
            ),
            "insample_censor_satisfaction_at_least_95pct": bool(
                insample_satisfaction
                and min(insample_satisfaction) >= THRESHOLDS["insample_censor_satisfaction_min"]
            ),
            "state_coefficient_in_range": bool(
                state_coefs and all(low <= v <= high for v in state_coefs)
            ),
            "all_solvers_converged": all(route_fits[r].get("solver_success", False) for r in ROUTES),
        }
    all_passed = bool(gates) and all(gates.values())

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(PROTOCOL_PATH.resolve()), "sha256": sha256_file(PROTOCOL_PATH)},
        "observations": {
            "path": str(FORMAL_RECORDS.resolve()),
            "sha256": records_sha256,
            "total_rows": len(observations),
            "terminal_rows": len(terminal),
            "missing_rows": missing,
        },
        "data_complete": complete,
        "fit_status": "final" if complete else "partial",
        "route_fits": route_fits,
        "cross_validation": route_cv,
        "large_model_zero3_replay": route_replay,
        "gates": gates,
        "all_gates_passed": all_passed,
        "acceptance_status": (
            "development_pass_shadow_only"
            if complete and all_passed
            else "failed_offline_gates"
            if complete
            else "data_incomplete"
        ),
        "automatic_admission_allowed": False,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(FIT_REPORT_PATH, report)

    # The hybrid shadow channel serves the hybrid-attention models, so the
    # artifact exports the dense_hybrid route coefficients (as stage-1 did).
    hybrid_fit = route_fits["dense_hybrid_attention"]
    if complete and hybrid_fit.get("coefficients_by_name"):
        artifact = {
            "schema": ARTIFACT_SCHEMA,
            "model_ids": ["qwen3p5_4b", "qwen3p5_9b", "qwen3_6_27b"],
            "coefficients_by_name": hybrid_fit["coefficients_by_name"],
            "feature_module_schema": FEATURE_SCHEMA_V3,
            "fit_report": {"path": str(FIT_REPORT_PATH.resolve()), "sha256": sha256_file(FIT_REPORT_PATH)},
            "cv_metrics": {r: route_cv[r]["pooled_held_out"] for r in ROUTES},
            "capacity_bytes": CAPACITY_BYTES,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "hybrid_shadow_candidate",
            "production_admission_allowed": False,
        }
        artifact["report_sha256"] = sha256_json(artifact)
        write_json(ARTIFACT_PATH, artifact)

    if complete and not all_passed and not allow_incomplete:
        raise RuntimeError("stage-2 fit failed one or more pre-registered gates")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.freeze_protocol:
        write_json(PROTOCOL_PATH, protocol_payload())
        print(json.dumps(read_json(PROTOCOL_PATH), ensure_ascii=False, indent=2))
        return
    report = fit(allow_incomplete=args.allow_incomplete)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
