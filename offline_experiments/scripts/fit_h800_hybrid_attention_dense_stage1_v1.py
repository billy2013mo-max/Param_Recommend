#!/usr/bin/env python3
"""Pre-registered fit for the dense hybrid-attention stage-1 observations.

The fit protocol was frozen before the formal queue finished.  It fits one
set of non-negative memory coefficients per architecture route
(dense_full_attention / dense_hybrid_attention), learns no per-family
coefficients, treats OOM rows as right-censored lower bounds (pre-OOM
nvidia-smi watermark, capacity fallback), folds repeat pairs by physical
configuration, and evaluates by leave-one-source-out cross validation.

Release is shadow-only: a passing fit never grants automatic admission.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

from common import (
    ARTIFACT_DIR,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from evaluate_h800_hybrid_attention_dense_stage1_v1 import (
    CAPACITY_BYTES,
    FORMAL_RECORDS,
)

PROTOCOL_SCHEMA = "sft_h800_hybrid_attention_dense_stage1_fit_protocol/v1"
REPORT_SCHEMA = "sft_h800_hybrid_attention_dense_stage1_fit_report/v1"
PROTOCOL_PATH = (
    ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_fit_protocol_v1.json"
)
FIT_REPORT_PATH = (
    ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_fit_report_v1.json"
)
SCRIPT_PATH = __file__

ROUTES = ("dense_full_attention", "dense_hybrid_attention")

# Coefficient name -> memory-feature key inside the observation feature basis.
# The intercept absorbs allocator context / residual base overhead.
COEFFICIENTS: tuple[tuple[str, str | None], ...] = (
    ("intercept", None),
    ("state", "state_bytes"),
    ("saved_full", "saved_full_attention_activations_bytes"),
    ("saved_linear", "saved_linear_attention_activations_bytes"),
    ("recompute", "recompute_workspace_bytes"),
    ("full_workspace", "full_attention_workspace_bytes"),
    ("linear_workspace", "linear_attention_workspace_bytes"),
    ("linear_state", "linear_recurrent_state_bytes"),
    ("logits", "logits_workspace_bytes"),
    ("zero_workspace", "zero_collective_workspace_bytes"),
)

# Pre-registered thresholds, frozen before the formal queue completed.
THRESHOLDS: dict[str, Any] = {
    "exact_mape_max": 0.10,
    "censor_satisfaction_min": 0.95,
    "state_coefficient_range": [0.8, 1.25],
    "censor_constraint_weight": 1.0,
    "minimum_exact_rows_per_route": {
        "dense_full_attention": 20,
        "dense_hybrid_attention": 60,
    },
    "minimum_exact_rows_per_source": 8,
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

CEIL = 1.0  # bytes floor for log-space predictions


def protocol_payload() -> dict[str, Any]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "campaign_id": "h800_hybrid_attention_dense_stage1_20260812_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_script": {
            "path": str(__import__("pathlib").Path(SCRIPT_PATH).resolve()),
            "sha256": sha256_file(__import__("pathlib").Path(SCRIPT_PATH)),
        },
        "observations_source": {
            "path": str(FORMAL_RECORDS.resolve()),
            "note": (
                "evaluate_h800_hybrid_attention_dense_stage1_v1.py output; "
                "records sha256 is bound at fit time"
            ),
        },
        "routes": list(ROUTES),
        "coefficients": [
            {
                "name": name,
                "feature_key": key,
                "constraint": "non_negative",
            }
            for name, key in COEFFICIENTS
        ],
        "objective": {
            "exact_rows": "squared log error: (log(predicted) - log(peak))^2",
            "censored_rows": (
                "hinge squared log error: max(0, log(bound) - log(predicted))^2 "
                "with weight censor_constraint_weight"
            ),
            "scale": "log space (relative error)",
        },
        "repeat_folding": {
            "key_fields": list(FOLD_KEY_FIELDS),
            "exact_only": "median of exact peaks",
            "mixed_exact_and_oom": (
                "collapse to censored at the max OOM watermark; exact peaks "
                "are recorded under discarded_exact_peaks"
            ),
        },
        "cross_validation": {
            "scheme": "leave_one_source_out",
            "grouping": "source_dataset_id; rows of one source never split",
            "metrics_on_held_out_rows_only": True,
        },
        "thresholds": THRESHOLDS,
        "zeRO_pair_diagnostic": {
            "role": "diagnostic_only_no_gate",
            "definition": (
                "for exact rows differing only in zero stage at the same "
                "gpu_count, report observed peak ratio zero3 / zero2"
            ),
        },
        "throughput": {"role": "diagnostic_only_no_gate"},
        "release": {
            "mode": "shadow_only",
            "automatic_admission_allowed": False,
            "acceptance_allowed": False,
        },
    }


def _memory_features(observation: dict[str, Any]) -> dict[str, float]:
    memory = observation["feature_basis"]["memory"]
    activation = memory["activation_components"]
    workspace = memory["workspace_candidates"]
    return {
        "state_bytes": float(memory["state_bytes"]),
        "saved_full_attention_activations_bytes": float(
            activation["saved_full_attention_activations_bytes"]
        ),
        "saved_linear_attention_activations_bytes": float(
            activation["saved_linear_attention_activations_bytes"]
        ),
        "recompute_workspace_bytes": float(
            activation["recompute_workspace_bytes"]
        ),
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


def _collapsed_rows(
    observations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fold repeat pairs into one fit row per physical configuration."""

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for observation in observations:
        groups.setdefault(_fold_key(observation), []).append(observation)
    rows: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
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
        if oom and not exact:
            bounds = [
                int(m["outcome"]["oom_right_censor_lower_bytes"]) for m in oom
            ]
            base["state"] = "censored"
            base["censor_lower_bytes"] = max(bounds)
            rows.append(base)
        elif oom and exact:
            # Boundary instability: one run OOM'd while another succeeded.
            # Conservative collapse matches the memory mainline convention.
            bounds = [
                int(m["outcome"]["oom_right_censor_lower_bytes"]) for m in oom
            ]
            base["state"] = "censored"
            base["censor_lower_bytes"] = max(bounds)
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


def _predict(coefficients: np.ndarray, design: Sequence[float]) -> float:
    return float(np.dot(coefficients, design))


def _residuals(
    coefficients: np.ndarray,
    rows: Sequence[dict[str, Any]],
    weight: float,
) -> np.ndarray:
    residuals: list[float] = []
    for row in rows:
        predicted = max(_predict(coefficients, row["design"]), CEIL)
        if row["state"] == "exact":
            residuals.append(math.log(predicted) - math.log(row["peak_reserved_bytes"]))
        else:
            residuals.append(
                math.sqrt(weight)
                * max(
                    0.0,
                    math.log(row["censor_lower_bytes"]) - math.log(predicted),
                )
            )
    return np.asarray(residuals, dtype=float)


def _fit_route(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fit one coefficient set per architecture route (pre-registered)."""

    initial = np.full(len(COEFFICIENTS), 1.0, dtype=float)
    initial[0] = 1e6  # intercept starts at 1 MB
    lower = np.zeros(len(COEFFICIENTS), dtype=float)
    weight = float(THRESHOLDS["censor_constraint_weight"])
    result = least_squares(
        lambda coefficients: _residuals(coefficients, rows, weight),
        initial,
        bounds=(lower, np.inf),
        method="trf",
        max_nfev=2000,
    )
    exact = [row for row in rows if row["state"] == "exact"]
    censored = [row for row in rows if row["state"] == "censored"]
    return {
        "coefficients": result.x.tolist(),
        "coefficients_by_name": dict(
            zip((name for name, _ in COEFFICIENTS), result.x)
        ),
        "exact_rows": len(exact),
        "censored_rows": len(censored),
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
    }


def _evaluate_rows(
    coefficients: np.ndarray, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    exact = [row for row in rows if row["state"] == "exact"]
    censored = [row for row in rows if row["state"] == "censored"]
    exact_apes: list[float] = []
    for row in exact:
        predicted = max(_predict(coefficients, row["design"]), CEIL)
        exact_apes.append(abs(predicted - row["peak_reserved_bytes"]) / row["peak_reserved_bytes"])
    satisfied = 0
    for row in censored:
        if _predict(coefficients, row["design"]) >= row["censor_lower_bytes"]:
            satisfied += 1
    return {
        "exact_rows": len(exact),
        "exact_mape": statistics.fmean(exact_apes) if exact_apes else None,
        "censored_rows": len(censored),
        "censor_satisfaction_rate": (
            satisfied / len(censored) if censored else None
        ),
    }


def _leave_one_source_out(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    sources = sorted({row["source_dataset_id"] for row in rows})
    folds: list[dict[str, Any]] = []
    held_out: list[dict[str, Any]] = []
    for held_source in sources:
        train = [row for row in rows if row["source_dataset_id"] != held_source]
        test = [row for row in rows if row["source_dataset_id"] == held_source]
        if not train or not test:
            folds.append(
                {
                    "held_out_source": held_source,
                    "skipped": True,
                    "reason": "no training or held-out rows for this source",
                }
            )
            continue
        fit = _fit_route(train)
        metrics = _evaluate_rows(fit["coefficients"], test)
        folds.append(
            {
                "held_out_source": held_source,
                "skipped": False,
                **metrics,
            }
        )
        held_out.append(metrics)
    pooled = {
        "exact_rows": sum(m["exact_rows"] for m in held_out),
        "exact_mape": (
            statistics.fmean(
                m["exact_mape"] * m["exact_rows"]
                for m in held_out
                if m["exact_mape"] is not None
            )
            / sum(m["exact_rows"] for m in held_out if m["exact_mape"] is not None)
            if any(m["exact_mape"] is not None for m in held_out)
            else None
        ),
        "censored_rows": sum(m["censored_rows"] for m in held_out),
        "censor_satisfaction_rate": (
            sum(
                (m["censor_satisfaction_rate"] or 0) * m["censored_rows"]
                for m in held_out
            )
            / sum(m["censored_rows"] for m in held_out)
            if sum(m["censored_rows"] for m in held_out)
            else None
        ),
    }
    return {"folds": folds, "pooled_held_out": pooled}


def _zeRO_pair_diagnostic(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Observed ZeRO-3 / ZeRO-2 peak ratios at matched physical configs.

    Diagnostic only: the memory-side ZeRO-3 constant bias has never been
    measured (the throughput-side ÷2.0 finding was V5-specific).  Stage-1
    pairs ZeRO-2 and ZeRO-3 at the same gpu_count, which is the matched
    contrast that would surface such a bias.
    """

    pairs: dict[Any, dict[str, Any]] = {}
    for row in rows:
        if row["state"] != "exact":
            continue
        key = (
            row["model_id"],
            row["source_dataset_id"],
            row["gpu_count"],
            row["fold_key"][4],  # gradient_checkpointing
            row["fold_key"][5],  # micro_batch_size
            row["fold_key"][6],  # exact_total_tokens_per_sample
        )
        pairs.setdefault(key, {})[str(row["zero"])] = row["peak_reserved_bytes"]
    ratios: list[float] = []
    details: list[dict[str, Any]] = []
    for key, by_zero in sorted(pairs.items()):
        z2 = by_zero.get("zero2")
        z3 = by_zero.get("zero3")
        if z2 is None or z3 is None:
            continue
        ratio = z3 / z2
        ratios.append(ratio)
        details.append(
            {
                "model_id": key[0],
                "source_dataset_id": key[1],
                "gpu_count": key[2],
                "zero2_bytes": z2,
                "zero3_bytes": z3,
                "zero3_over_zero2_ratio": ratio,
            }
        )
    return {
        "matched_pairs": len(details),
        "details": details,
        "median_ratio": statistics.median(ratios) if ratios else None,
        "min_ratio": min(ratios) if ratios else None,
        "max_ratio": max(ratios) if ratios else None,
        "note": (
            "diagnostic only; a systematic ratio far from the analytic "
            "prediction would be evidence of a memory-side ZeRO-3 constant "
            "bias, which has never been measured on the memory side"
        ),
    }


def _throughput_diagnostic(
    observations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_family: dict[str, list[float]] = {}
    for observation in observations:
        outcome = observation["outcome"]
        if outcome["classification"] != "success":
            continue
        metrics = observation.get("metrics") or {}
        tokens_per_second = metrics.get("effective_tokens_per_second")
        if tokens_per_second is None:
            continue
        by_family.setdefault(str(observation["model_id"]), []).append(
            float(tokens_per_second)
        )
    return {
        "role": "diagnostic_only_no_gate",
        "by_model": {
            family: {
                "rows": len(values),
                "median_tokens_per_second": statistics.median(values),
            }
            for family, values in sorted(by_family.items())
        },
    }


def _verify_protocol(protocol: dict[str, Any]) -> None:
    """Fail closed if the frozen protocol does not match this script."""

    problems: list[str] = []
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        problems.append("protocol schema drifted")
    if protocol.get("fit_script", {}).get("sha256") != sha256_file(
        __import__("pathlib").Path(SCRIPT_PATH)
    ):
        problems.append("fit script sha256 drifted from frozen protocol")
    if protocol.get("thresholds") != THRESHOLDS:
        problems.append("thresholds drifted from frozen protocol")
    names = [name for name, _ in COEFFICIENTS]
    if [row.get("name") for row in protocol.get("coefficients", [])] != names:
        problems.append("coefficient list drifted from frozen protocol")
    if protocol.get("release", {}).get("mode") != "shadow_only":
        problems.append("release mode drifted from shadow_only")
    if problems:
        raise RuntimeError(
            "frozen fit protocol mismatch: " + "; ".join(problems)
        )


def fit(*, allow_incomplete: bool = False) -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError(
            "fit protocol not frozen yet; run with --freeze-protocol first"
        )
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
        rows_by_route[str(observation["architecture_route"])].append(
            observation
        )
    collapsed: dict[str, list[dict[str, Any]]] = {}
    for route in ROUTES:
        collapsed[route] = _collapsed_rows(rows_by_route[route])

    route_fits: dict[str, dict[str, Any]] = {}
    route_cv: dict[str, dict[str, Any]] = {}
    for route in ROUTES:
        rows = collapsed[route]
        if rows:
            route_fits[route] = _fit_route(rows)
            route_cv[route] = _leave_one_source_out(rows)
        else:
            route_fits[route] = {"exact_rows": 0, "censored_rows": 0}
            route_cv[route] = {"folds": [], "pooled_held_out": {}}

    gates: dict[str, bool] = {}
    if complete:
        cv_exact_rows = sum(
            route_cv[route]["pooled_held_out"].get("exact_rows", 0)
            for route in ROUTES
        )
        cv_mape = [
            route_cv[route]["pooled_held_out"].get("exact_mape")
            for route in ROUTES
        ]
        cv_mape = [v for v in cv_mape if v is not None]
        satisfaction_rates = [
            route_cv[route]["pooled_held_out"].get("censor_satisfaction_rate")
            for route in ROUTES
        ]
        satisfaction_rates = [v for v in satisfaction_rates if v is not None]
        state_coefficients = [
            float(route_fits[route]["coefficients_by_name"]["state"])
            for route in ROUTES
            if route_fits[route].get("coefficients_by_name")
        ]
        low, high = THRESHOLDS["state_coefficient_range"]
        gates = {
            "minimum_exact_rows_per_route": all(
                route_fits[route]["exact_rows"]
                >= THRESHOLDS["minimum_exact_rows_per_route"][route]
                for route in ROUTES
            ),
            "cv_exact_mape_within_10pct": bool(
                cv_mape
                and max(cv_mape) <= THRESHOLDS["exact_mape_max"]
            ),
            "cv_censor_satisfaction_at_least_95pct": bool(
                satisfaction_rates
                and min(satisfaction_rates) >= THRESHOLDS["censor_satisfaction_min"]
            ),
            "state_coefficient_in_range": bool(
                state_coefficients
                and all(low <= value <= high for value in state_coefficients)
            ),
            "all_solvers_converged": all(
                route_fits[route].get("solver_success", False)
                for route in ROUTES
            ),
        }
    all_passed = bool(gates) and all(gates.values())

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(PROTOCOL_PATH.resolve()),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
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
        "zeRO_pair_diagnostic": {
            route: _zeRO_pair_diagnostic(collapsed[route]) for route in ROUTES
        },
        "throughput_diagnostic": _throughput_diagnostic(terminal),
    }
    report["report_sha256"] = sha256_json(report)
    write_json(FIT_REPORT_PATH, report)
    if complete and not all_passed and not allow_incomplete:
        raise RuntimeError(
            "stage-1 fit failed one or more pre-registered gates"
        )
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
