#!/usr/bin/env python3
"""Fit and audit the H800 memory model with native-v2 calibration evidence.

This is a CPU-only stage-1 calibration command.  It keeps the predeclared
``calibration`` and ``holdout`` partitions separate:

* the candidate fit augments the historical memory-boundary basis with only
  native-v2 ``calibration`` rows;
* native-v2 ``holdout`` rows are touched only after the candidate is frozen;
* packing-pair evidence and MBS values outside the supported planner domain are
  excluded explicitly;
* the existing historical model is evaluated on the exact same native holdout
  rows, making before/after comparisons meaningful.

The output is an auditable, non-publishable candidate report.  It does not
create a planner profile, launch a GPU job, or mutate an experiment queue.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any

from audit_h800_calibration_readiness import (
    SUPPORTED_MBS,
    _selector as readiness_selector,
)
from common import ROOT, read_json, sha256_file, sha256_json
from h800_theory_basis import (
    ACTIVATION_LIVENESS_BOUNDS,
    RECORD_SCHEMA,
    _memory_observation,
    _model_geometry,
    memory_basis,
)
from h800_theory_calibration import (
    DEFAULT_ALPHA,
    MEMORY_INNER_OOF_FOLDS,
    _evaluate_memory,
    _fit_memory_center,
    _fit_memory_tail,
    _fit_reserved_model,
    _inner_scenario_oof,
    _make_memory_residual_collector,
    _observed_allocated,
    _observed_reserved,
    _predict_allocated_center,
    _predict_memory,
    _predicted_reserved_center,
    _safe_limit,
    scenario_id,
    scenario_material,
)


SCHEMA = "sft_h800_native_memory_calibration/v1"
OBSERVATION_SCHEMA = "sft_efficiency_observation/v2"
IMPLEMENTATION_VERSION = (
    "sft_h800_native_memory_calibration_impl/"
    "2026-07-27.augmented-calibration-frozen-holdout"
)
CALIBRATION_ROLES = {"calibration", "holdout"}
PACKING_EVIDENCE_CLASS = "packing_paired_only"
GIB = float(1024**3)


def _verify_bound_report(
    report: Mapping[str, Any], *, expected_schema: str, name: str
) -> None:
    if report.get("schema") != expected_schema:
        raise ValueError(
            f"{name} schema mismatch: {report.get('schema')!r} != {expected_schema!r}"
        )
    unsigned = dict(report)
    digest = unsigned.pop("report_sha256", None)
    if not isinstance(digest, str) or digest != sha256_json(unsigned):
        raise ValueError(f"{name} report SHA-256 mismatch")


def _read_observations(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("schema") != OBSERVATION_SCHEMA:
                raise ValueError(
                    f"Observation line {line_number} is not {OBSERVATION_SCHEMA}"
                )
            observation_id = row.get("observation_id")
            if not isinstance(observation_id, str) or not observation_id:
                raise ValueError(f"Observation line {line_number} has no id")
            if observation_id in seen:
                raise ValueError(f"Duplicate observation id {observation_id}")
            seen.add(observation_id)
            rows.append(row)
    return rows


def _job(row: Mapping[str, Any]) -> dict[str, Any]:
    configuration = row.get("configuration")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    job = configuration.get("job")
    return dict(job) if isinstance(job, Mapping) else {}


def _partition(row: Mapping[str, Any]) -> dict[str, Any]:
    configuration = row.get("configuration")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    partition = configuration.get("calibration_partition")
    return dict(partition) if isinstance(partition, Mapping) else {}


def native_admission_reason(row: Mapping[str, Any]) -> str:
    """Return ``admitted`` or one mutually exclusive exclusion reason."""

    hardware = row.get("hardware")
    hardware = hardware if isinstance(hardware, Mapping) else {}
    if hardware.get("gpu_family") != "H800":
        return "not_exact_h800"
    fingerprint = row.get("fingerprint")
    fingerprint = fingerprint if isinstance(fingerprint, Mapping) else {}
    if fingerprint.get("quality") != "complete":
        return "fingerprint_not_complete"
    if fingerprint.get("calibration_evidence_eligible") is not True:
        return "fingerprint_not_calibration_eligible"
    outcome = row.get("outcome")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    if outcome.get("class") not in {"success", "oom"}:
        return "outcome_not_success_or_oom"
    if outcome.get("usable_for_feasibility_calibration") is not True:
        return "not_usable_for_feasibility"
    partition = _partition(row)
    if str(partition.get("role") or "").lower() not in CALIBRATION_ROLES:
        return "partition_not_calibration_or_holdout"
    if not isinstance(partition.get("split_unit_id"), str) or not partition.get(
        "split_unit_id"
    ):
        return "partition_split_unit_missing"
    job = _job(row)
    if job.get("packing") is True or job.get(
        "calibration_evidence_class"
    ) == PACKING_EVIDENCE_CLASS:
        return "packing_effect_evidence_excluded"
    try:
        mbs = int(job.get("mbs"))
    except (TypeError, ValueError):
        return "mbs_missing_or_invalid"
    if mbs not in SUPPORTED_MBS:
        return "mbs_outside_supported_domain"
    selector_key, selector = readiness_selector(dict(row))
    if (
        not selector_key
        or not isinstance(selector.get("runtime_fingerprint"), str)
        or not selector.get("runtime_fingerprint")
        or selector.get("dtype") == "unknown"
        or "unknown" in str(selector.get("kernel_path"))
        or selector.get("training_mode") not in {"full", "lora"}
        or selector.get("zero_stage") not in {0, 1, 2, 3}
        or not isinstance(selector.get("gradient_checkpointing"), bool)
    ):
        return "mechanism_selector_incomplete"
    return "admitted"


def _inventory_models(
    inventory: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    models = inventory.get("models")
    if not isinstance(models, list):
        raise ValueError("Model inventory has no models")
    model_by_id = {
        str(model.get("id")): dict(model)
        for model in models
        if isinstance(model, Mapping)
    }
    if len(model_by_id) != len(models):
        raise ValueError("Model inventory ids must be present and unique")
    fixed_lora = inventory.get("fixed_lora")
    if not isinstance(fixed_lora, Mapping):
        raise ValueError("Model inventory has no fixed_lora contract")
    return model_by_id, dict(fixed_lora)


def build_native_record(
    row: dict[str, Any],
    *,
    model_by_id: Mapping[str, dict[str, Any]],
    fixed_lora: dict[str, Any],
    hardware: dict[str, Any],
) -> dict[str, Any]:
    if native_admission_reason(row) != "admitted":
        raise ValueError("Cannot build a native record from an excluded observation")
    job = _job(row)
    model_id = str(job.get("model_id") or "")
    if model_id not in model_by_id:
        raise ValueError(f"No model inventory for {model_id!r}")
    geometry = _model_geometry(job, model_by_id[model_id], fixed_lora)
    capacity = int(hardware.get("memory_bytes_reported_by_torch"))
    memory = memory_basis(job, geometry, capacity)
    memory["observed"] = _memory_observation(row)
    _key, native_selector = readiness_selector(row)
    runtime_fingerprint = str(native_selector["runtime_fingerprint"])
    partition = _partition(row)
    selector = {
        "runtime_cohort_id": runtime_fingerprint,
        "dtype": native_selector["dtype"],
        "kernel_path": native_selector["kernel_path"],
        "training_mode": native_selector["training_mode"],
        "zero_stage": native_selector["zero_stage"],
        "gradient_checkpointing": native_selector["gradient_checkpointing"],
        "packing": False,
    }
    scenario = {
        "model_id": job.get("model_id"),
        "train_type": job.get("train_type"),
        "dataset_id": job.get("dataset_id"),
        "target_gbs": job.get("target_gbs"),
        "gpu_count": job.get("gpu_count"),
        "physical_mbs": job.get("mbs"),
        "cutoff_len": job.get("cutoff_len"),
    }
    return {
        "schema": RECORD_SCHEMA,
        "observation_id": row["observation_id"],
        "job_id": job.get("job_id"),
        "source_observation_sha256": sha256_json(row),
        "recovery_id": None,
        "evidence_tier": "native_v2",
        "route": {"feasibility": True, "memory_boundary": True},
        "measurement_eligibility": {
            "feasibility": True,
            "memory_boundary": True,
        },
        "outcome": (row.get("outcome") or {}).get("class"),
        "scenario": scenario,
        "scenario_id": sha256_json(
            {
                "model_id": scenario["model_id"],
                "train_type": scenario["train_type"],
                "dataset_id": scenario["dataset_id"],
                "target_gbs": scenario["target_gbs"],
            }
        ),
        "selector": selector,
        "runtime": {
            "runtime_cohort_id": runtime_fingerprint,
            "runtime_cohort_material": {
                "runtime_mechanism_fingerprint_sha256": runtime_fingerprint
            },
        },
        "calibration_partition": {
            "policy": partition.get("policy"),
            "role": str(partition.get("role")).lower(),
            "split_unit_id": partition.get("split_unit_id"),
        },
        "model_basis": geometry,
        "memory": memory,
        "performance": None,
        "confidence": "native_v2",
        "publishable": False,
    }


def _route_enabled(record: Mapping[str, Any], route: str) -> bool:
    raw = record.get("route")
    return isinstance(raw, Mapping) and raw.get(route) is True


def _outcome(record: Mapping[str, Any]) -> str:
    raw = record.get("outcome")
    if isinstance(raw, Mapping):
        raw = raw.get("class")
    return str(raw or "").lower()


def _observation_id(record: Mapping[str, Any]) -> str:
    value = record.get("observation_id")
    if not isinstance(value, str) or not value:
        raise ValueError("Memory record has no observation id")
    return value


def _summary(values: Sequence[float]) -> dict[str, Any]:
    usable = [float(value) for value in values if math.isfinite(float(value))]
    if not usable:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    ordered = sorted(usable)

    def percentile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    return {
        "count": len(usable),
        "mean": statistics.fmean(usable),
        "median": statistics.median(usable),
        "p90": percentile(0.90),
        "max": max(usable),
    }


def _center_metrics(
    records: Sequence[Mapping[str, Any]],
    center: Mapping[str, Any],
    tail: Mapping[str, Any],
    *,
    include_details: bool,
) -> dict[str, Any]:
    allocated_ape: list[float] = []
    allocated_ae: list[float] = []
    allocated_bias: list[float] = []
    reserved_ape: list[float] = []
    reserved_ae: list[float] = []
    reserved_bias: list[float] = []
    details: list[dict[str, Any]] = []
    unavailable = 0
    reserved_model = {
        "center": center,
        "gap_buckets": tail.get("reserved_minus_allocated_center") or {},
    }
    for record in records:
        if _outcome(record) != "success":
            continue
        observed_allocated = _observed_allocated(record)
        observed_reserved = _observed_reserved(record)
        predicted_allocated, issues = _predict_allocated_center(record, center)
        predicted_reserved = _predicted_reserved_center(record, reserved_model)
        detail: dict[str, Any] = {
            "observation_id": _observation_id(record),
            "prediction_available": (
                predicted_allocated is not None
                and predicted_reserved is not None
                and observed_allocated is not None
                and observed_reserved is not None
            ),
        }
        if not detail["prediction_available"]:
            unavailable += 1
            detail["issues"] = issues or ["reserved_center_or_label_missing"]
            if include_details:
                details.append(detail)
            continue
        assert predicted_allocated is not None
        assert predicted_reserved is not None
        assert observed_allocated is not None
        assert observed_reserved is not None
        allocated_error = predicted_allocated - observed_allocated
        reserved_error = predicted_reserved - observed_reserved
        allocated_ae.append(abs(allocated_error))
        reserved_ae.append(abs(reserved_error))
        allocated_ape.append(abs(allocated_error) / observed_allocated)
        reserved_ape.append(abs(reserved_error) / observed_reserved)
        allocated_bias.append(allocated_error / observed_allocated)
        reserved_bias.append(reserved_error / observed_reserved)
        if include_details:
            detail.update(
                {
                    "observed_allocated_bytes": observed_allocated,
                    "predicted_allocated_center_bytes": predicted_allocated,
                    "allocated_absolute_percentage_error": allocated_ape[-1],
                    "observed_reserved_bytes": observed_reserved,
                    "predicted_reserved_center_bytes": predicted_reserved,
                    "reserved_absolute_percentage_error": reserved_ape[-1],
                }
            )
            details.append(detail)
    return {
        "success_rows": sum(_outcome(record) == "success" for record in records),
        "prediction_unavailable_rows": unavailable,
        "allocated_center": {
            "absolute_percentage_error": _summary(allocated_ape),
            "absolute_error_bytes": _summary(allocated_ae),
            "absolute_error_gib": _summary([value / GIB for value in allocated_ae]),
            "signed_percentage_error": _summary(allocated_bias),
        },
        "reserved_center": {
            "absolute_percentage_error": _summary(reserved_ape),
            "absolute_error_bytes": _summary(reserved_ae),
            "absolute_error_gib": _summary([value / GIB for value in reserved_ae]),
            "signed_percentage_error": _summary(reserved_bias),
        },
        "details": details if include_details else None,
    }


def _operational_metrics(
    records: Sequence[Mapping[str, Any]],
    center: Mapping[str, Any],
    tail: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = _evaluate_memory(records, center, tail)
    safe_successes = 0
    admitted_safe_successes = 0
    false_reject_safe_successes = 0
    p95_headroom: list[float] = []
    for record in records:
        if _outcome(record) != "success":
            continue
        observed = _observed_reserved(record)
        safe_limit = _safe_limit(record)
        prediction = _predict_memory(record, center, tail)
        if (
            observed is None
            or safe_limit is None
            or prediction.get("available") is not True
        ):
            continue
        p95 = float(prediction["p95_reserved_bytes"])
        p95_headroom.append(p95 - observed)
        if observed <= safe_limit:
            safe_successes += 1
            admitted = p95 <= safe_limit
            admitted_safe_successes += int(admitted)
            false_reject_safe_successes += int(not admitted)
    metrics.update(
        {
            "actual_safe_success_rows": safe_successes,
            "admitted_safe_success_rows": admitted_safe_successes,
            "false_reject_safe_success": false_reject_safe_successes,
            "safe_success_admission_recall": (
                admitted_safe_successes / safe_successes
                if safe_successes
                else None
            ),
            "p95_minus_observed_reserved_gib": _summary(
                [value / GIB for value in p95_headroom]
            ),
        }
    )
    return metrics


def _scenario_equal_operational(
    records: Sequence[Mapping[str, Any]],
    center: Mapping[str, Any],
    tail: Mapping[str, Any],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[scenario_id(record)].append(record)
    metrics = [
        _operational_metrics(group, center, tail)
        for _scenario, group in sorted(groups.items())
    ]

    def mean_field(name: str) -> float | None:
        values = [
            float(item[name])
            for item in metrics
            if item.get(name) is not None
        ]
        return statistics.fmean(values) if values else None

    return {
        "scenario_count": len(groups),
        "success_p95_coverage": mean_field("success_p95_coverage"),
        "false_safe_oom_rate": mean_field("false_safe_oom_rate"),
        "safe_success_admission_recall": mean_field(
            "safe_success_admission_recall"
        ),
    }


def _evaluate_model(
    records: Sequence[Mapping[str, Any]],
    center: Mapping[str, Any],
    tail: Mapping[str, Any],
    *,
    include_details: bool,
) -> dict[str, Any]:
    return {
        "center_accuracy": _center_metrics(
            records, center, tail, include_details=include_details
        ),
        "operational_safety": _operational_metrics(records, center, tail),
        "scenario_equal_operational_safety": _scenario_equal_operational(
            records, center, tail
        ),
    }


def _fit_model(
    records: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
) -> dict[str, Any]:
    center = _fit_memory_center(records)
    residuals = _inner_scenario_oof(
        records,
        frozenset(),
        fit_fn=_fit_reserved_model,
        collect_fn=_make_memory_residual_collector(),
        fit_cache={},
        folds=MEMORY_INNER_OOF_FOLDS,
    )
    tail = _fit_memory_tail(
        records,
        center,
        alpha=alpha,
        residual_observations=residuals,
    )
    return {
        "center": center,
        "tail": tail,
        "training_observation_ids_sha256": sha256_json(
            sorted(_observation_id(record) for record in records)
        ),
        "training_rows": len(records),
        "training_outcomes": dict(
            sorted(Counter(_outcome(record) for record in records).items())
        ),
        "training_scenarios": len({scenario_id(record) for record in records}),
    }


def _augmented_native_loocv(
    augmented_records: Sequence[Mapping[str, Any]],
    native_calibration: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
) -> dict[str, Any]:
    native_scenarios = sorted({scenario_id(record) for record in native_calibration})
    fit_cache: dict[frozenset[str], Any] = {}
    collector = _make_memory_residual_collector()
    fold_reports: list[dict[str, Any]] = []
    pooled_center_details: list[dict[str, Any]] = []
    success_rows = success_covered = oom_rows = false_safe = unavailable = 0
    safe_successes = admitted_safe = false_reject = 0
    for held_out in native_scenarios:
        train = [
            record
            for record in augmented_records
            if scenario_id(record) != held_out
        ]
        test = [
            record
            for record in native_calibration
            if scenario_id(record) == held_out
        ]
        center = _fit_memory_center(train)
        residuals = _inner_scenario_oof(
            augmented_records,
            frozenset({held_out}),
            fit_fn=_fit_reserved_model,
            collect_fn=collector,
            fit_cache=fit_cache,
            folds=MEMORY_INNER_OOF_FOLDS,
        )
        tail = _fit_memory_tail(
            train,
            center,
            alpha=alpha,
            residual_observations=residuals,
        )
        evaluation = _evaluate_model(
            test, center, tail, include_details=True
        )
        operational = evaluation["operational_safety"]
        center_details = evaluation["center_accuracy"]["details"] or []
        pooled_center_details.extend(center_details)
        success_rows += int(operational.get("success_rows") or 0)
        success_covered += int(operational.get("success_covered") or 0)
        oom_rows += int(operational.get("oom_rows") or 0)
        false_safe += int(operational.get("false_safe_oom") or 0)
        unavailable += int(operational.get("prediction_unavailable_rows") or 0)
        safe_successes += int(operational.get("actual_safe_success_rows") or 0)
        admitted_safe += int(operational.get("admitted_safe_success_rows") or 0)
        false_reject += int(operational.get("false_reject_safe_success") or 0)
        fold_reports.append(
            {
                "held_out_scenario_id": held_out,
                "held_out_scenario": scenario_material(test[0]),
                "train_rows": len(train),
                "test_rows": len(test),
                "train_observation_ids_sha256": sha256_json(
                    sorted(_observation_id(record) for record in train)
                ),
                "test_observation_ids_sha256": sha256_json(
                    sorted(_observation_id(record) for record in test)
                ),
                "center_accuracy": evaluation["center_accuracy"],
                "operational_safety": operational,
            }
        )

    def detail_values(field: str) -> list[float]:
        return [
            float(detail[field])
            for detail in pooled_center_details
            if detail.get(field) is not None
        ]

    scenario_coverage = [
        float(fold["operational_safety"]["success_p95_coverage"])
        for fold in fold_reports
        if fold["operational_safety"].get("success_p95_coverage") is not None
    ]
    scenario_false_safe = [
        float(fold["operational_safety"]["false_safe_oom_rate"])
        for fold in fold_reports
        if fold["operational_safety"].get("false_safe_oom_rate") is not None
    ]
    return {
        "policy": (
            "leave-one-native-(model,training_mode,dataset,target_gbs)-out; "
            "the same scenario is also removed from historical training rows"
        ),
        "scenario_folds": len(fold_reports),
        "aggregate": {
            "center_accuracy": {
                "allocated_center_absolute_percentage_error": _summary(
                    detail_values("allocated_absolute_percentage_error")
                ),
                "reserved_center_absolute_percentage_error": _summary(
                    detail_values("reserved_absolute_percentage_error")
                ),
            },
            "operational_safety": {
                "success_rows": success_rows,
                "success_covered": success_covered,
                "success_p95_coverage": (
                    success_covered / success_rows if success_rows else None
                ),
                "scenario_equal_success_p95_coverage": (
                    statistics.fmean(scenario_coverage)
                    if scenario_coverage
                    else None
                ),
                "oom_rows": oom_rows,
                "false_safe_oom": false_safe,
                "false_safe_oom_rate": (
                    false_safe / oom_rows if oom_rows else None
                ),
                "scenario_equal_false_safe_oom_rate": (
                    statistics.fmean(scenario_false_safe)
                    if scenario_false_safe
                    else None
                ),
                "prediction_unavailable_rows": unavailable,
                "actual_safe_success_rows": safe_successes,
                "admitted_safe_success_rows": admitted_safe,
                "false_reject_safe_success": false_reject,
                "safe_success_admission_recall": (
                    admitted_safe / safe_successes if safe_successes else None
                ),
            },
        },
        "folds": fold_reports,
    }


def _metric_comparison(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_center = baseline["center_accuracy"]
    candidate_center = candidate["center_accuracy"]
    baseline_safety = baseline["operational_safety"]
    candidate_safety = candidate["operational_safety"]

    def value(root: Mapping[str, Any], path: Sequence[str]) -> float | int | None:
        current: Any = root
        for key in path:
            if not isinstance(current, Mapping):
                return None
            current = current.get(key)
        return current if isinstance(current, (int, float)) else None

    definitions = {
        "allocated_center_mean_ape": (
            baseline_center,
            candidate_center,
            ("allocated_center", "absolute_percentage_error", "mean"),
            "lower_is_better",
        ),
        "reserved_center_mean_ape": (
            baseline_center,
            candidate_center,
            ("reserved_center", "absolute_percentage_error", "mean"),
            "lower_is_better",
        ),
        "success_p95_coverage": (
            baseline_safety,
            candidate_safety,
            ("success_p95_coverage",),
            "target_at_least_0.95_not_monotonic_accuracy",
        ),
        "false_safe_oom": (
            baseline_safety,
            candidate_safety,
            ("false_safe_oom",),
            "lower_is_better_target_zero",
        ),
        "false_reject_safe_success": (
            baseline_safety,
            candidate_safety,
            ("false_reject_safe_success",),
            "lower_is_better_conditional_on_safety",
        ),
        "prediction_unavailable_rows": (
            baseline_safety,
            candidate_safety,
            ("prediction_unavailable_rows",),
            "lower_is_better_target_zero",
        ),
    }
    result: dict[str, Any] = {}
    for name, (old_root, new_root, path, interpretation) in definitions.items():
        old = value(old_root, path)
        new = value(new_root, path)
        result[name] = {
            "historical_baseline": old,
            "augmented_candidate": new,
            "candidate_minus_baseline": (
                float(new) - float(old)
                if old is not None and new is not None
                else None
            ),
            "interpretation": interpretation,
        }
    return result


def build_report(
    *,
    observation_path: Path,
    historical_basis_path: Path,
    historical_calibration_path: Path,
    inventory_path: Path,
    hardware_path: Path,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    observations = _read_observations(observation_path)
    inventory = read_json(inventory_path)
    hardware = read_json(hardware_path)
    historical_basis = read_json(historical_basis_path)
    historical_calibration = read_json(historical_calibration_path)
    _verify_bound_report(
        historical_basis,
        expected_schema="sft_h800_theory_basis/v1",
        name="historical theory basis",
    )
    _verify_bound_report(
        historical_calibration,
        expected_schema="sft_h800_theory_calibration/v1",
        name="historical calibration",
    )
    if "h800" not in str(
        hardware.get("name_reported_by_driver") or ""
    ).lower():
        raise ValueError("Native H800 memory calibration requires H800 hardware")
    model_by_id, fixed_lora = _inventory_models(inventory)

    admission_counts: Counter[str] = Counter()
    native_records: list[dict[str, Any]] = []
    for row in observations:
        reason = native_admission_reason(row)
        admission_counts[reason] += 1
        if reason == "admitted":
            native_records.append(
                build_native_record(
                    row,
                    model_by_id=model_by_id,
                    fixed_lora=fixed_lora,
                    hardware=hardware,
                )
            )
    native_records.sort(key=_observation_id)
    native_calibration = [
        record
        for record in native_records
        if record["calibration_partition"]["role"] == "calibration"
    ]
    native_holdout = [
        record
        for record in native_records
        if record["calibration_partition"]["role"] == "holdout"
    ]
    if not native_calibration or not native_holdout:
        raise ValueError("Both native calibration and holdout rows are required")
    calibration_units = {
        record["calibration_partition"]["split_unit_id"]
        for record in native_calibration
    }
    holdout_units = {
        record["calibration_partition"]["split_unit_id"]
        for record in native_holdout
    }
    if calibration_units.intersection(holdout_units):
        raise ValueError("Calibration and holdout split units overlap")

    historical_records_raw = historical_basis.get("records")
    if not isinstance(historical_records_raw, list):
        raise ValueError("Historical theory basis has no records")
    historical_memory = [
        record
        for record in historical_records_raw
        if isinstance(record, Mapping)
        and _route_enabled(record, "memory_boundary")
        and _outcome(record) in {"success", "oom"}
    ]
    historical_ids = {_observation_id(record) for record in historical_memory}
    native_ids = {_observation_id(record) for record in native_records}
    overlap = historical_ids.intersection(native_ids)
    if overlap:
        raise ValueError(
            f"Historical/native memory observations overlap: {sorted(overlap)[:3]}"
        )
    augmented = [*historical_memory, *native_calibration]

    baseline_fit = (
        (historical_calibration.get("full_historical_bootstrap_fit") or {})
        .get("memory")
    )
    if not isinstance(baseline_fit, Mapping):
        raise ValueError("Historical calibration report has no frozen memory fit")
    baseline_center = baseline_fit.get("center")
    baseline_tail = baseline_fit.get("tail")
    if not isinstance(baseline_center, Mapping) or not isinstance(
        baseline_tail, Mapping
    ):
        raise ValueError("Historical calibration memory fit is incomplete")

    augmented_fit = _fit_model(augmented, alpha=alpha)
    native_only_fit = _fit_model(native_calibration, alpha=alpha)
    loocv = _augmented_native_loocv(
        augmented, native_calibration, alpha=alpha
    )
    baseline_holdout = _evaluate_model(
        native_holdout,
        baseline_center,
        baseline_tail,
        include_details=True,
    )
    augmented_holdout = _evaluate_model(
        native_holdout,
        augmented_fit["center"],
        augmented_fit["tail"],
        include_details=True,
    )
    native_only_holdout = _evaluate_model(
        native_holdout,
        native_only_fit["center"],
        native_only_fit["tail"],
        include_details=True,
    )
    comparison = _metric_comparison(baseline_holdout, augmented_holdout)

    holdout_safety = augmented_holdout["operational_safety"]
    holdout_acceptance = {
        "success_p95_coverage_at_least_0_95": (
            holdout_safety.get("success_p95_coverage") is not None
            and float(holdout_safety["success_p95_coverage"]) >= 0.95
        ),
        "false_safe_oom_is_zero": holdout_safety.get("false_safe_oom") == 0,
        "prediction_unavailable_rows_is_zero": (
            holdout_safety.get("prediction_unavailable_rows") == 0
        ),
    }
    holdout_acceptance["all_passed"] = all(holdout_acceptance.values())
    loocv_safety = loocv["aggregate"]["operational_safety"]
    dynamic_blockers: list[str] = []
    if not holdout_acceptance["all_passed"]:
        dynamic_blockers.append("native_holdout_memory_acceptance_failed")
    if (
        loocv_safety.get("success_p95_coverage") is None
        or float(loocv_safety["success_p95_coverage"]) < 0.95
    ):
        dynamic_blockers.append(
            "native_calibration_loocv_p95_coverage_below_0_95"
        )
    if int(loocv_safety.get("false_safe_oom") or 0) > 0:
        dynamic_blockers.append(
            "native_calibration_loocv_false_safe_oom_nonzero"
        )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "stage1_native_memory_candidate_evaluated",
        "gpu_family": "H800",
        "alpha": alpha,
        "publishable": False,
        "production_profile_generated": False,
        "coefficients_were_fit": True,
        "source_bindings": {
            "canonical_observations": {
                "path": str(observation_path),
                "sha256": sha256_file(observation_path),
                "rows": len(observations),
            },
            "historical_theory_basis": {
                "path": str(historical_basis_path),
                "sha256": sha256_file(historical_basis_path),
                "report_sha256": historical_basis.get("report_sha256"),
            },
            "historical_calibration": {
                "path": str(historical_calibration_path),
                "sha256": sha256_file(historical_calibration_path),
                "report_sha256": historical_calibration.get("report_sha256"),
            },
            "model_inventory": {
                "path": str(inventory_path),
                "sha256": sha256_file(inventory_path),
            },
            "hardware": {
                "path": str(hardware_path),
                "sha256": sha256_file(hardware_path),
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "data_admission": {
            "policy": (
                "exact H800 + complete native-v2 fingerprint + feasibility-usable "
                "+ predeclared calibration/holdout role + unpacked core domain "
                f"+ MBS in {sorted(SUPPORTED_MBS)}"
            ),
            "counts": dict(sorted(admission_counts.items())),
            "native_admitted": {
                "rows": len(native_records),
                "outcomes": dict(
                    sorted(Counter(_outcome(record) for record in native_records).items())
                ),
                "roles": {
                    role: sum(
                        record["calibration_partition"]["role"] == role
                        for record in native_records
                    )
                    for role in sorted(CALIBRATION_ROLES)
                },
            },
            "calibration": {
                "rows": len(native_calibration),
                "outcomes": dict(
                    sorted(
                        Counter(
                            _outcome(record) for record in native_calibration
                        ).items()
                    )
                ),
                "split_units": sorted(calibration_units),
                "observation_ids_sha256": sha256_json(
                    sorted(_observation_id(record) for record in native_calibration)
                ),
            },
            "holdout": {
                "rows": len(native_holdout),
                "outcomes": dict(
                    sorted(
                        Counter(
                            _outcome(record) for record in native_holdout
                        ).items()
                    )
                ),
                "split_units": sorted(holdout_units),
                "observation_ids_sha256": sha256_json(
                    sorted(_observation_id(record) for record in native_holdout)
                ),
                "used_for_fit": False,
                "touched_only_after_candidate_fit_frozen": True,
            },
            "partition_units_disjoint": True,
        },
        "model_contract": {
            "allocated_center_formula": (
                "analytic_non_activation_bytes + "
                "lambda[training_mode,gradient_checkpointing] * "
                "structural_activation_bytes + runtime_cohort_nuisance_bytes"
            ),
            "reserved_center_formula": (
                "allocated_center + hierarchical_median(reserved-allocated; "
                "selector -> mode_x_gc -> pooled)"
            ),
            "operational_p95_formula": (
                "reserved_center + max(success_scenario_OOF_conformal_q95, "
                "exact_selector_OOM_guard) + runtime_cohort_evidence_inflation"
            ),
            "oom_observation_contract": (
                "right-censored lower inequality; an OOM demand peak is never imputed"
            ),
            "admission_rule": "operational_p95_reserved_bytes <= 0.95 * device_capacity_bytes",
            "activation_liveness_bounds": list(ACTIVATION_LIVENESS_BOUNDS),
            "fit_features_exclude": ["model_id", "dataset_id", "target_gbs"],
        },
        "training_populations": {
            "historical_memory_rows": len(historical_memory),
            "native_calibration_rows": len(native_calibration),
            "augmented_candidate_rows": len(augmented),
            "native_holdout_rows": len(native_holdout),
            "candidate_contains_holdout_rows": False,
        },
        "augmented_candidate_fit": augmented_fit,
        "native_only_diagnostic_fit": native_only_fit,
        "native_calibration_augmented_loocv": loocv,
        "frozen_native_holdout": {
            "historical_baseline": baseline_holdout,
            "augmented_candidate": augmented_holdout,
            "native_only_diagnostic": native_only_holdout,
            "same_rows_for_all_models": True,
            "comparison": comparison,
            "augmented_candidate_acceptance": holdout_acceptance,
        },
        "publication_blockers": sorted(
            set(
                [
                    "stage1_report_is_diagnostic_and_never_auto_publishes",
                    "planner_trusted_exact_operator_manifest_missing",
                    "historical_rows_remain_bootstrap_evidence",
                    *dynamic_blockers,
                    *(
                        (augmented_fit["tail"].get("cohort_evidence_inflation") or {})
                        .get("blockers")
                        or []
                    ),
                ]
            )
        ),
    }
    report["report_sha256"] = sha256_json(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema") != SCHEMA:
        raise ValueError("Native memory calibration schema mismatch")
    unsigned = dict(report)
    digest = unsigned.pop("report_sha256", None)
    if digest != sha256_json(unsigned):
        raise ValueError("Native memory calibration report SHA-256 mismatch")
    if report.get("publishable") is not False:
        raise ValueError("Stage-1 native memory report must remain non-publishable")
    if report.get("production_profile_generated") is not False:
        raise ValueError("Stage-1 report cannot generate a production profile")
    admission = report.get("data_admission")
    admission = admission if isinstance(admission, Mapping) else {}
    if admission.get("partition_units_disjoint") is not True:
        raise ValueError("Calibration/holdout split units are not disjoint")
    holdout = admission.get("holdout")
    holdout = holdout if isinstance(holdout, Mapping) else {}
    if (
        holdout.get("used_for_fit") is not False
        or holdout.get("touched_only_after_candidate_fit_frozen") is not True
    ):
        raise ValueError("Holdout separation contract is missing")
    populations = report.get("training_populations")
    populations = populations if isinstance(populations, Mapping) else {}
    if populations.get("candidate_contains_holdout_rows") is not False:
        raise ValueError("Candidate fit contains holdout rows")
    json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    validate_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(
                report,
                output,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        type=Path,
        default=ROOT / "artifacts" / "canonical_h800_observations.jsonl",
    )
    parser.add_argument(
        "--historical-basis",
        type=Path,
        default=ROOT / "artifacts" / "h800_theory_basis.json",
    )
    parser.add_argument(
        "--historical-calibration",
        type=Path,
        default=ROOT / "artifacts" / "h800_theory_calibration.json",
    )
    parser.add_argument(
        "--model-inventory",
        type=Path,
        default=ROOT / "artifacts" / "model_inventory.json",
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=ROOT / "config" / "hardware.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "h800_native_memory_calibration.json",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()
    report = build_report(
        observation_path=args.observations,
        historical_basis_path=args.historical_basis,
        historical_calibration_path=args.historical_calibration,
        inventory_path=args.model_inventory,
        hardware_path=args.hardware,
        alpha=args.alpha,
    )
    write_report(args.output, report)
    holdout = report["frozen_native_holdout"]
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "output": str(args.output),
                "native_admitted": report["data_admission"]["native_admitted"],
                "training_populations": report["training_populations"],
                "holdout_comparison": holdout["comparison"],
                "holdout_acceptance": holdout[
                    "augmented_candidate_acceptance"
                ],
                "publishable": report["publishable"],
                "report_sha256": report["report_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
