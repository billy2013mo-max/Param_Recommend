#!/usr/bin/env python3
"""Fit auditable Packing profile estimators from frozen DataProfile-v2 curves.

The recommendation-time fallback is intentionally small and portable:

* a ridge head predicts pack utilization from aggregate profile features;
* epoch-mean samples/pack is derived from the physical identity
  ``capacity * utilization / mean_length``;
* a separate ridge head predicts the step-P99 pack count;
* family-LOO multiplicative guards provide conservative mean/P99 bounds.

Exact cached cutoff curves remain authoritative.  This model is a cache-miss
shadow fallback and cannot enable automatic Packing publication by itself.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from common import ARTIFACT_DIR, percentile, read_json, sha256_file, sha256_json, write_json, write_jsonl


SCHEMA = "sft_packing_profile_estimators/v1"
ACCEPTANCE_SCHEMA = "sft_packing_profile_estimators_acceptance/v1"
MEMBERSHIP_SCHEMA = "sft_packing_fit_membership/v1"
LABEL_SCHEMA = "sft_packing_cutoff_label/v1"

PROFILE_MANIFEST = ARTIFACT_DIR / "packing_data_profiles_w1_w9_manifest_v2.json"
PROFILE_SCHEMA = ARTIFACT_DIR / "packing_data_profile_schema_v2.json"
COMBINED_RESULTS = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_combined_results_v1.json"
GBS_CORRECTION = ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_gbs_contract_correction_v2.json"
REPAIR_RESULTS = ARTIFACT_DIR / "h800_packing_gbs_repair_batch_results_v1.json"

LABELS = ARTIFACT_DIR / "packing_cutoff_labels_w1_w9_v1.jsonl"
MEMBERSHIP = ARTIFACT_DIR / "packing_fit_membership_v1.json"
MODEL = ARTIFACT_DIR / "packing_profile_estimators_v1.json"
ACCEPTANCE = ARTIFACT_DIR / "packing_profile_estimators_acceptance_v1.json"
MARKDOWN = ARTIFACT_DIR / "packing_profile_estimators_acceptance_v1.md"

CALIBRATION_WORKLOADS = ("W1", "W2", "W3", "W4", "W5", "W7", "W8")
DIAGNOSTIC_WORKLOADS = ("W6",)
PROSPECTIVE_WORKLOADS = ("W9",)
NEW_LABEL_WORKLOADS = ("W3", "W5", "W7", "W8")
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
EPSILON = 1e-8

FEATURE_NAMES = (
    "log_capacity",
    "log_capacity_over_length_mean",
    "log_capacity_over_length_p50",
    "log_capacity_over_length_p90",
    "log_capacity_over_length_p99",
    "log_capacity_over_length_max",
    "length_cv",
    "length_p50_over_mean",
    "length_p90_over_mean",
    "length_p99_over_mean",
    "length_max_over_mean",
    "label_ratio",
    "turns_mean",
    "turns_cv",
    "log_records",
)


def _finite_positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _profile_partition(workload_id: str) -> str:
    if workload_id in CALIBRATION_WORKLOADS:
        return "calibration_fit"
    if workload_id in DIAGNOSTIC_WORKLOADS:
        return "diagnostic_excluded"
    if workload_id in PROSPECTIVE_WORKLOADS:
        return "prospective_holdout"
    raise ValueError(f"Unknown workload partition: {workload_id}")


def feature_vector(profile: dict[str, Any], cutoff_point: dict[str, Any]) -> list[float]:
    """Return recommendation-time features without reading label targets."""
    aggregate = profile["aggregate"]
    lengths = aggregate["length_tokens"]
    turns = aggregate["turns"]
    capacity = _finite_positive(cutoff_point["packing_capacity"], "packing_capacity")
    mean = _finite_positive(lengths["mean"], "length mean")
    p50 = _finite_positive(lengths["p50"], "length p50")
    p90 = _finite_positive(lengths["p90"], "length p90")
    p99 = _finite_positive(lengths["p99"], "length p99")
    maximum = _finite_positive(lengths["maximum"], "length maximum")
    turns_mean = _finite_positive(turns["mean"], "turns mean")
    records = _finite_positive(aggregate["records"], "records")
    values = [
        math.log(capacity),
        math.log(capacity / mean),
        math.log(capacity / p50),
        math.log(capacity / p90),
        math.log(capacity / p99),
        math.log(capacity / maximum),
        float(lengths["standard_deviation"]) / mean,
        p50 / mean,
        p90 / mean,
        p99 / mean,
        maximum / mean,
        float(aggregate["label_tokens"]["ratio_of_total"]),
        turns_mean,
        float(turns["standard_deviation"]) / turns_mean,
        math.log(records),
    ]
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite Packing estimator features")
    return values


def _profile_fingerprint_valid(profile: dict[str, Any]) -> bool:
    expected = profile.get("profile_fingerprint_sha256")
    content = dict(profile)
    content.pop("profile_fingerprint_sha256", None)
    return isinstance(expected, str) and expected == sha256_json(content)


def build_label_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema") != "sft_packing_data_profile_manifest/v2":
        raise ValueError("unexpected DataProfile manifest schema")
    rows: list[dict[str, Any]] = []
    seen_workloads: set[str] = set()
    for binding in manifest.get("profiles", []):
        path = Path(binding["path"])
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"DataProfile binding mismatch: {path}")
        profile = read_json(path)
        workload_id = str(profile["workload_id"])
        if workload_id in seen_workloads:
            raise ValueError(f"duplicate profile for {workload_id}")
        seen_workloads.add(workload_id)
        if not _profile_fingerprint_valid(profile):
            raise ValueError(f"profile fingerprint mismatch: {workload_id}")
        partition = _profile_partition(workload_id)
        length_mean = float(profile["aggregate"]["length_tokens"]["mean"])
        for point in profile["packing_curve"]:
            truncation_rate = float(point["sample_truncation_rate"])
            nontruncating = truncation_rate == 0.0
            samples = point["samples_per_pack"]
            row = {
                "schema": LABEL_SCHEMA,
                "label_id": f"{profile['profile_id']}@{int(point['cutoff_len'])}",
                "workload_id": workload_id,
                "profile_id": profile["profile_id"],
                "profile_role": profile["profile_role"],
                "profile_partition": partition,
                "profile_path": str(path.resolve()),
                "profile_sha256": binding["sha256"],
                "cutoff_len": int(point["cutoff_len"]),
                "packing_capacity": int(point["packing_capacity"]),
                "nontruncating": nontruncating,
                "sample_truncation_rate": truncation_rate,
                "fit_eligible": partition == "calibration_fit" and nontruncating,
                "features": dict(zip(FEATURE_NAMES, feature_vector(profile, point), strict=True)),
                "physics": {"length_mean": length_mean},
                "targets": {
                    "pack_utilization": float(point["pack_utilization"]),
                    "n_pack_mean": float(samples["mean"]),
                    "n_pack_step_p99": float(samples["p99"]),
                    "n_pack_maximum": float(samples["maximum"]),
                    "packs": int(point["packs"]),
                },
            }
            expected_mean = row["packing_capacity"] * row["targets"]["pack_utilization"] / length_mean
            if nontruncating and not math.isclose(
                expected_mean, row["targets"]["n_pack_mean"], rel_tol=1e-10, abs_tol=1e-10
            ):
                raise ValueError(f"mean physics identity failed: {row['label_id']}")
            rows.append(row)
    if seen_workloads != {f"W{index}" for index in range(1, 10)}:
        raise ValueError("W1-W9 membership is incomplete")
    label_ids = [row["label_id"] for row in rows]
    if len(label_ids) != len(set(label_ids)):
        raise ValueError("duplicate cutoff labels")
    return sorted(rows, key=lambda row: (int(row["workload_id"][1:]), row["cutoff_len"]))


def _job_semantics_valid(row: dict[str, Any]) -> bool:
    if row.get("classification") != "success":
        return False
    if row.get("calibration_eligible") is not True:
        return False
    if row.get("metrics_rank_count_exact") is not True or row.get("measured_steps_consistent") is not True:
        return False
    if row.get("authoritative_ledger_all_ranks") is not True:
        return False
    return not bool(row.get("packing")) or row.get("packing_semantics_all_ranks") is True


def build_fit_membership() -> dict[str, Any]:
    combined = read_json(COMBINED_RESULTS)
    correction = read_json(GBS_CORRECTION)
    repair = read_json(REPAIR_RESULTS)
    if combined.get("completion", {}).get("combined_success") != 24:
        raise ValueError("the 24-job interaction batch is incomplete")
    if repair.get("completion", {}).get("success") != 18:
        raise ValueError("the 18-job repair batch is incomplete")
    if repair.get("gates", {}).get("strict_target_gbs_fit_only_allowed") is not True:
        raise ValueError("repair batch has not passed the strict fit-only gate")
    correction_by_setting = {row["setting_id"]: row for row in correction.get("rows", [])}
    if set(correction_by_setting) != {"w1_gc_off", "w1_gc_on", "w4_zero2", "w4_zero3"}:
        raise ValueError("GBS correction membership is incomplete")

    rows = []
    for row in combined["job_results"]:
        contract = correction_by_setting[row["setting_id"]]["contract"]
        strict = contract["gates"]["candidate_admissible"] is True
        rows.append({
            "job_id": row["job_id"],
            "campaign_id": row["campaign_id"],
            "setting_id": row["setting_id"],
            "packing": bool(row["packing"]),
            "repeat": int(row["repeat"]),
            "evidence_role": "strict_target_gbs_fit_only" if strict else "distribution_labeled_shadow_only",
            "reason_codes": [] if strict else list(contract["gates"]["reason_codes"]),
            "target_gbs": float(contract["target_gbs"]),
            "expected_epoch_sample_gbs": float(contract["expected_epoch_sample_gbs"]),
            "n_pack_step_p99": float(contract["samples_per_pack"]["p99"]),
            "semantic_checks_passed": _job_semantics_valid(row),
            "source": "platform_v4_interactions_batch1_combined",
        })
    for row in repair["job_results"]:
        contract = row["gbs_contract_v2"]
        strict = contract["gates"]["candidate_admissible"] is True
        rows.append({
            "job_id": row["job_id"],
            "campaign_id": row["campaign_id"],
            "setting_id": row["setting_id"],
            "packing": bool(row["packing"]),
            "repeat": int(row["repeat"]),
            "evidence_role": "strict_target_gbs_fit_only" if strict else "distribution_labeled_shadow_only",
            "reason_codes": list(contract["gates"]["reason_codes"]),
            "target_gbs": float(contract["target_gbs"]),
            "expected_epoch_sample_gbs": float(contract["expected_epoch_sample_gbs"]),
            "n_pack_step_p99": float(contract["samples_per_pack"]["p99"]),
            "semantic_checks_passed": _job_semantics_valid(row),
            "source": "packing_gbs_repair_batch",
        })
    job_ids = [row["job_id"] for row in rows]
    if len(rows) != 42 or len(job_ids) != len(set(job_ids)):
        raise ValueError("GPU evidence membership must contain 42 unique jobs")
    if not all(row["semantic_checks_passed"] for row in rows):
        raise ValueError("membership contains invalid execution semantics")
    counts = Counter(row["evidence_role"] for row in rows)
    report: dict[str, Any] = {
        "schema": MEMBERSHIP_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_bindings": {
            "combined_results": {"path": str(COMBINED_RESULTS.resolve()), "sha256": sha256_file(COMBINED_RESULTS)},
            "gbs_correction": {"path": str(GBS_CORRECTION.resolve()), "sha256": sha256_file(GBS_CORRECTION)},
            "repair_results": {"path": str(REPAIR_RESULTS.resolve()), "sha256": sha256_file(REPAIR_RESULTS)},
        },
        "policy": {
            "strict_target_gbs_fit_only_may_fit_coefficients": True,
            "distribution_labeled_shadow_may_fit_strict_target_gbs_coefficients": False,
            "automatic_publication_allowed": False,
        },
        "counts": {"jobs": len(rows), **dict(sorted(counts.items()))},
        "rows": sorted(rows, key=lambda row: row["job_id"]),
    }
    report["report_sha256"] = sha256_json(report)
    return report


def _target_transform(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "logit":
        clipped = np.clip(values, EPSILON, 1.0 - EPSILON)
        return np.log(clipped / (1.0 - clipped))
    if kind == "log":
        return np.log(values)
    raise ValueError(kind)


def _target_inverse(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "logit":
        positive = values >= 0
        result = np.empty_like(values, dtype=float)
        result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exponential = np.exp(values[~positive])
        result[~positive] = exponential / (1.0 + exponential)
        return np.clip(result, EPSILON, 1.0 - EPSILON)
    if kind == "log":
        return np.exp(values)
    raise ValueError(kind)


def _fit_ridge(x: np.ndarray, y: np.ndarray, *, alpha: float, transform: str) -> dict[str, Any]:
    scaler = StandardScaler().fit(x)
    standardized = scaler.transform(x)
    ridge = Ridge(alpha=alpha).fit(standardized, _target_transform(y, transform))
    return {
        "model_type": "standardized_ridge",
        "alpha": float(alpha),
        "target_transform": transform,
        "feature_names": list(FEATURE_NAMES),
        "standardization": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "intercept": float(ridge.intercept_),
        "coefficients": ridge.coef_.tolist(),
    }


def predict_serialized(model: dict[str, Any], feature_rows: np.ndarray) -> np.ndarray:
    x = np.asarray(feature_rows, dtype=float)
    mean = np.asarray(model["standardization"]["mean"], dtype=float)
    scale = np.asarray(model["standardization"]["scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    raw = (x - mean) / scale @ coefficients + float(model["intercept"])
    return _target_inverse(raw, str(model["target_transform"]))


def _scenario_equal_relative_mae(truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> float:
    relative = np.abs(prediction - truth) / truth
    return float(np.mean([np.mean(relative[groups == group]) for group in sorted(set(groups))]))


def _select_and_oof(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, *, transform: str,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, float]]]:
    unique_groups = sorted(set(str(group) for group in groups))
    trials = []
    predictions_by_alpha: dict[float, np.ndarray] = {}
    for alpha in RIDGE_ALPHAS:
        predictions = np.zeros(len(y), dtype=float)
        for group in unique_groups:
            test = groups == group
            train = ~test
            model = _fit_ridge(x[train], y[train], alpha=alpha, transform=transform)
            predictions[test] = predict_serialized(model, x[test])
        score = _scenario_equal_relative_mae(y, predictions, groups)
        trials.append({"alpha": float(alpha), "scenario_equal_relative_mae": score})
        predictions_by_alpha[float(alpha)] = predictions
    selected = min(trials, key=lambda row: (row["scenario_equal_relative_mae"], row["alpha"]))
    alpha = float(selected["alpha"])
    model = _fit_ridge(x, y, alpha=alpha, transform=transform)
    model["selection"] = {
        "policy": "minimum_family_loo_scenario_equal_relative_mae",
        "candidate_trials": trials,
        "selected_alpha": alpha,
    }
    return model, predictions_by_alpha[alpha], trials


def _center_metrics(truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    relative = np.abs(prediction - truth) / truth
    signed = (prediction - truth) / truth
    per_family = {}
    for group in sorted(set(str(value) for value in groups)):
        mask = groups == group
        per_family[group] = {
            "rows": int(np.sum(mask)),
            "relative_mae": float(np.mean(relative[mask])),
            "relative_p95": float(np.percentile(relative[mask], 95)),
            "signed_relative_bias": float(np.mean(signed[mask])),
        }
    return {
        "rows": len(truth),
        "scenario_equal_relative_mae": float(np.mean([row["relative_mae"] for row in per_family.values()])),
        "row_weighted_relative_mae": float(np.mean(relative)),
        "row_weighted_relative_p95": float(np.percentile(relative, 95)),
        "maximum_relative_error": float(np.max(relative)),
        "signed_relative_bias": float(np.mean(signed)),
        "per_family": per_family,
    }


def _coverage(
    truth: np.ndarray, lower: np.ndarray | None, upper: np.ndarray, groups: np.ndarray,
) -> dict[str, Any]:
    covered = truth <= upper + 1e-12
    if lower is not None:
        covered &= truth + 1e-12 >= lower
    per_family = {
        group: float(np.mean(covered[groups == group]))
        for group in sorted(set(str(value) for value in groups))
    }
    return {
        "row_coverage": float(np.mean(covered)),
        "scenario_equal_coverage": float(np.mean(list(per_family.values()))),
        "underpredicted_rows": int(np.sum(~covered)),
        "per_family": per_family,
    }


def _row_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([[row["features"][name] for name in FEATURE_NAMES] for row in rows], dtype=float)
    utilization = np.asarray([row["targets"]["pack_utilization"] for row in rows], dtype=float)
    mean = np.asarray([row["targets"]["n_pack_mean"] for row in rows], dtype=float)
    p99 = np.asarray([row["targets"]["n_pack_step_p99"] for row in rows], dtype=float)
    capacity_over_mean = np.asarray([
        row["packing_capacity"] / row["physics"]["length_mean"] for row in rows
    ], dtype=float)
    groups = np.asarray([row["workload_id"] for row in rows])
    return x, utilization, mean, p99, capacity_over_mean, groups


def fit_estimators(label_rows: list[dict[str, Any]], membership: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration = [row for row in label_rows if row["fit_eligible"]]
    prospective = [
        row for row in label_rows
        if row["profile_partition"] == "prospective_holdout" and row["nontruncating"]
    ]
    if {row["workload_id"] for row in calibration} != set(CALIBRATION_WORKLOADS):
        raise ValueError("calibration workload membership mismatch")
    if {row["workload_id"] for row in prospective} != set(PROSPECTIVE_WORKLOADS):
        raise ValueError("prospective workload membership mismatch")
    x, utilization, mean, p99, capacity_over_mean, groups = _row_arrays(calibration)
    util_model, util_oof, _ = _select_and_oof(x, utilization, groups, transform="logit")
    p99_model, p99_oof, _ = _select_and_oof(x, p99, groups, transform="log")
    mean_oof = util_oof * capacity_over_mean

    mean_ratio = mean / mean_oof
    p99_ratio = p99 / p99_oof
    guard = {
        "policy": "family_loo_empirical_range_guard_v1",
        "mean_lower_multiplier": float(np.min(mean_ratio) * (1.0 - 1e-12)),
        "mean_upper_multiplier": float(np.max(mean_ratio) * (1.0 + 1e-12)),
        "p99_upper_multiplier": float(np.max(p99_ratio) * (1.0 + 1e-12)),
        "calibration_family_count": len(set(groups)),
    }

    px, putilization, pmean, pp99, pcapacity_over_mean, pgroups = _row_arrays(prospective)
    util_final = predict_serialized(util_model, x)
    p99_final = predict_serialized(p99_model, x)
    util_prospective = predict_serialized(util_model, px)
    p99_prospective = predict_serialized(p99_model, px)
    mean_prospective = util_prospective * pcapacity_over_mean

    calibration_metrics = {
        "evaluation_protocol": "leave_one_workload_family_out",
        "utilization_center": _center_metrics(utilization, util_oof, groups),
        "n_pack_mean_center": _center_metrics(mean, mean_oof, groups),
        "n_pack_mean_interval": _coverage(
            mean,
            mean_oof * guard["mean_lower_multiplier"],
            mean_oof * guard["mean_upper_multiplier"],
            groups,
        ),
        "n_pack_step_p99_center": _center_metrics(p99, p99_oof, groups),
        "n_pack_step_p99_upper": _coverage(
            p99, None, p99_oof * guard["p99_upper_multiplier"], groups
        ),
    }
    prospective_metrics = {
        "evaluation_protocol": "source_disjoint_frozen_W9_never_used_for_fit_or_guard",
        "utilization_center": _center_metrics(putilization, util_prospective, pgroups),
        "n_pack_mean_center": _center_metrics(pmean, mean_prospective, pgroups),
        "n_pack_mean_interval": _coverage(
            pmean,
            mean_prospective * guard["mean_lower_multiplier"],
            mean_prospective * guard["mean_upper_multiplier"],
            pgroups,
        ),
        "n_pack_step_p99_center": _center_metrics(pp99, p99_prospective, pgroups),
        "n_pack_step_p99_upper": _coverage(
            pp99, None, p99_prospective * guard["p99_upper_multiplier"], pgroups
        ),
    }

    model: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "usage": "cache_miss_shadow_fallback_only",
        "automatic_recommendation_allowed": False,
        "exact_cached_curve_preferred": True,
        "recommendation_time_contract": {
            "raw_dataset_read": False,
            "raw_length_vector_read": False,
            "tokenization": False,
            "full_packer_run": False,
            "aggregate_profile_only": True,
        },
        "membership": {
            "calibration_workloads": list(CALIBRATION_WORKLOADS),
            "diagnostic_excluded_workloads": list(DIAGNOSTIC_WORKLOADS),
            "prospective_holdout_workloads": list(PROSPECTIVE_WORKLOADS),
            "calibration_rows": len(calibration),
            "prospective_rows": len(prospective),
        },
        "source_bindings": {
            "profile_manifest": {"path": str(PROFILE_MANIFEST.resolve()), "sha256": sha256_file(PROFILE_MANIFEST)},
            "profile_schema": {"path": str(PROFILE_SCHEMA.resolve()), "sha256": sha256_file(PROFILE_SCHEMA)},
            "cutoff_labels": {"path": str(LABELS.resolve()), "sha256": sha256_file(LABELS)},
            "gpu_fit_membership": {
                "path": str(MEMBERSHIP.resolve()),
                "sha256": sha256_file(MEMBERSHIP),
                "report_sha256": membership["report_sha256"],
            },
        },
        "features": list(FEATURE_NAMES),
        "heads": {
            "pack_utilization_center": util_model,
            "n_pack_mean": {
                "model_type": "physical_identity",
                "formula": "packing_capacity * predicted_pack_utilization / length_mean",
            },
            "n_pack_step_p99_center": p99_model,
        },
        "safety_guard": guard,
        "fit_diagnostics": {
            "calibration_full_fit_utilization": _center_metrics(utilization, util_final, groups),
            "calibration_full_fit_p99": _center_metrics(p99, p99_final, groups),
        },
    }
    model["model_sha256"] = sha256_json(model)

    thresholds = {
        "calibration_scenario_equal_center_relative_mae_max": 0.05,
        "prospective_center_relative_mae_max": 0.05,
        "mean_interval_coverage_min": 0.95,
        "p99_upper_coverage_min": 0.95,
        "prospective_p99_underpredicted_rows_max": 0,
        "p99_center_scenario_equal_relative_mae_max_for_automatic_pruning": 0.25,
        "p99_upper_multiplier_max_for_automatic_pruning": 1.50,
    }
    gates = {
        "calibration_utilization_center": calibration_metrics["utilization_center"]["scenario_equal_relative_mae"] <= 0.05,
        "calibration_mean_center": calibration_metrics["n_pack_mean_center"]["scenario_equal_relative_mae"] <= 0.05,
        "calibration_mean_interval": calibration_metrics["n_pack_mean_interval"]["scenario_equal_coverage"] >= 0.95,
        "calibration_p99_upper": calibration_metrics["n_pack_step_p99_upper"]["scenario_equal_coverage"] >= 0.95,
        "prospective_utilization_center": prospective_metrics["utilization_center"]["scenario_equal_relative_mae"] <= 0.05,
        "prospective_mean_center": prospective_metrics["n_pack_mean_center"]["scenario_equal_relative_mae"] <= 0.05,
        "prospective_mean_interval": prospective_metrics["n_pack_mean_interval"]["scenario_equal_coverage"] >= 0.95,
        "prospective_p99_upper": (
            prospective_metrics["n_pack_step_p99_upper"]["scenario_equal_coverage"] >= 0.95
            and prospective_metrics["n_pack_step_p99_upper"]["underpredicted_rows"] == 0
        ),
        "source_disjoint_holdout": set(CALIBRATION_WORKLOADS).isdisjoint(PROSPECTIVE_WORKLOADS),
        "p99_center_sharpness": (
            calibration_metrics["n_pack_step_p99_center"]["scenario_equal_relative_mae"] <= 0.25
        ),
        "p99_guard_sharpness": guard["p99_upper_multiplier"] <= 1.50,
    }
    safety_gate_names = (
        "calibration_utilization_center", "calibration_mean_center",
        "calibration_mean_interval", "calibration_p99_upper",
        "prospective_utilization_center", "prospective_mean_center",
        "prospective_mean_interval", "prospective_p99_upper",
        "source_disjoint_holdout",
    )
    fit_only_ready = all(gates[name] for name in safety_gate_names)
    pruning_ready = gates["p99_center_sharpness"] and gates["p99_guard_sharpness"]
    acceptance: dict[str, Any] = {
        "schema": ACCEPTANCE_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_binding": {"path": str(MODEL.resolve()), "model_sha256": model["model_sha256"]},
        "thresholds": thresholds,
        "calibration": calibration_metrics,
        "prospective": prospective_metrics,
        "gates": {
            **gates,
            "fit_only_estimator_ready": fit_only_ready,
            "shadow_cache_miss_inference_allowed": fit_only_ready,
            "automatic_gbs_candidate_pruning_allowed": fit_only_ready and pruning_ready,
            "automatic_packing_recommendation_allowed": False,
            "automatic_publication_allowed": False,
            "automatic_gpu_batch_allowed": False,
        },
        "blockers": [
            "only_one_source_disjoint_prospective_profile_family",
            "p99_center_and_guard_are_safe_but_not_sharp_enough_for_automatic_pruning",
            "no_prospective_gpu_throughput_acceptance_for_W3_W5_W7_W8",
            "memory_boundary_not_calibrated_near_capacity",
        ],
    }
    acceptance["report_sha256"] = sha256_json(acceptance)
    return model, acceptance


def _write_markdown(acceptance: dict[str, Any], membership: dict[str, Any], labels: list[dict[str, Any]]) -> None:
    calibration = acceptance["calibration"]
    prospective = acceptance["prospective"]
    new_counts = Counter(
        row["workload_id"] for row in labels
        if row["workload_id"] in NEW_LABEL_WORKLOADS and row["nontruncating"]
    )
    pct = lambda value: f"{100.0 * float(value):.2f}%"
    lines = [
        "# Packing DataProfile estimator v1 验收", "",
        f"GPU 证据 membership：严格 fit-only {membership['counts'].get('strict_target_gbs_fit_only', 0)} jobs；shadow {membership['counts'].get('distribution_labeled_shadow_only', 0)} jobs。", "",
        "W3/W5/W7/W8 新整理的 non-truncating exact pack 标签："
        + "，".join(f"{key}={new_counts[key]}" for key in NEW_LABEL_WORKLOADS)
        + f"，合计 {sum(new_counts.values())} 个 cutoff 点。", "",
        "| 指标 | family-LOO calibration | W9 prospective | 门槛/结论 |", 
        "|---|---:|---:|---|",
        f"| utilization center relative MAE | {pct(calibration['utilization_center']['scenario_equal_relative_mae'])} | {pct(prospective['utilization_center']['scenario_equal_relative_mae'])} | ≤5% |",
        f"| n_pack_mean center relative MAE | {pct(calibration['n_pack_mean_center']['scenario_equal_relative_mae'])} | {pct(prospective['n_pack_mean_center']['scenario_equal_relative_mae'])} | ≤5% |",
        f"| n_pack_mean safety interval coverage | {pct(calibration['n_pack_mean_interval']['scenario_equal_coverage'])} | {pct(prospective['n_pack_mean_interval']['scenario_equal_coverage'])} | ≥95% |",
        f"| n_pack_step_p99 center relative MAE | {pct(calibration['n_pack_step_p99_center']['scenario_equal_relative_mae'])} | {pct(prospective['n_pack_step_p99_center']['scenario_equal_relative_mae'])} | 自动剪枝要求≤25% |",
        f"| n_pack_step_p99 upper coverage | {pct(calibration['n_pack_step_p99_upper']['scenario_equal_coverage'])} | {pct(prospective['n_pack_step_p99_upper']['scenario_equal_coverage'])} | ≥95%，prospective 不低估 |",
        "",
        f"fit-only estimator ready：`{acceptance['gates']['fit_only_estimator_ready']}`；cache-miss shadow inference：`{acceptance['gates']['shadow_cache_miss_inference_allowed']}`。", "",
        f"P99 safety guard multiplier 为 `{read_json(MODEL)['safety_guard']['p99_upper_multiplier']:.3f}×`；自动 GBS 候选剪枝：`{acceptance['gates']['automatic_gbs_candidate_pruning_allowed']}`。", "",
        "精确缓存曲线仍优先；本模型不得自动开启 Packing、发布模型或启动新 GPU 批次。", "",
    ]
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    manifest = read_json(PROFILE_MANIFEST)
    labels = build_label_rows(manifest)
    membership = build_fit_membership()
    write_jsonl(LABELS, labels)
    write_json(MEMBERSHIP, membership)
    model, acceptance = fit_estimators(labels, membership)
    write_json(MODEL, model)
    acceptance["model_binding"]["sha256"] = sha256_file(MODEL)
    acceptance["report_sha256"] = sha256_json({key: value for key, value in acceptance.items() if key != "report_sha256"})
    write_json(ACCEPTANCE, acceptance)
    _write_markdown(acceptance, membership, labels)
    return {
        "labels": str(LABELS),
        "membership": str(MEMBERSHIP),
        "model": str(MODEL),
        "acceptance": str(ACCEPTANCE),
        "markdown": str(MARKDOWN),
        "label_rows": len(labels),
        "membership_counts": membership["counts"],
        "gates": acceptance["gates"],
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
