#!/usr/bin/env python3
"""Backfill static virtual MBS for historical Qwen3-14B Packing evidence.

This analysis does not launch GPU work and does not refit either production
model.  It reconstructs the mean logical samples per pack from the frozen
dataset-side artifacts, then tests the zero-extra-coefficient hypothesis by
anchoring the current frozen unpacked throughput model at each experiment's
observed Unpacked arm.

The resulting evidence is intentionally labelled indirect: historical jobs
usually measured only one Unpacked MBS instead of the floor/ceil pair around
the virtual MBS.  The current frozen model is therefore used only to transfer
the observed Unpacked anchor from its measured MBS to the static virtual MBS.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze_h800_packing_virtual_mbs_v1 import FrozenUnpackedPredictor
from common import (
    ARTIFACT_DIR,
    RESULTS_DIR,
    ROOT,
    percentile,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)


MODEL_ID = "qwen3_14b"
LEGACY_EFFECT = ARTIFACT_DIR / "h800_packing_effect_candidate.json"
LEGACY_STATIC = ARTIFACT_DIR / "static_packing_policy_v1_calibration_replay.json"
ABBA_QUEUE = ROOT / "runtime/pipeline/pending-packing-abba.jsonl"
CALIBRATION_QUEUE = ROOT / "matrix/h800_packing_calibration_v1.jsonl"
CALIBRATION_STATIC = ARTIFACT_DIR / "h800_packing_calibration_frozen_decisions_v1.json"
UNIFIED_QUEUE = ROOT / "matrix/h800_unified_resource_evidence_jobs_v1.jsonl"
MAIN_MODEL = (
    ROOT
    / "diagnostics/h800_unified_resource_partial_refit_20260809/throughput_model.json"
)
OUTPUT = ARTIFACT_DIR / "h800_qwen14_historical_virtual_mbs_backfill_v1.json"
MARKDOWN = ARTIFACT_DIR / "h800_qwen14_historical_virtual_mbs_backfill_v1.md"
DIRECT_CONTROL_JOBS = {
    "short_512": {
        "floor": (2, "mem-8a8ab842f5971ae6-mbs2"),
        "ceil": (4, "mem-8a8ab842f5971ae6-mbs4"),
    },
    "multiturn_2048": {
        "floor": (1, "mem-e46aaef7c89abb5d-mbs1"),
        "ceil": (2, "mem-e46aaef7c89abb5d-mbs2"),
    },
    "longcontext_16384": {
        "floor": (1, "mem-fdf73366681b3eb7-mbs1"),
        "ceil": (2, "mem-fdf73366681b3eb7-mbs2"),
    },
}


def _positive(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"expected a positive finite value, got {value!r}")
    return result


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("geometric mean requires at least one value")
    return math.exp(statistics.fmean(math.log(_positive(value)) for value in values))


def _zero_stage(job: Mapping[str, Any]) -> int:
    if job.get("zero_stage") is not None:
        return int(job["zero_stage"])
    return {"none": 0, "zero2": 2, "zero3": 3}[str(job.get("zero"))]


def _gradient_accumulation(job: Mapping[str, Any], *, n_pack_mean: float | None = None) -> int:
    if job.get("gradient_accumulation_steps") is not None:
        return int(job["gradient_accumulation_steps"])
    denominator = int(job["gpu_count"]) * (
        float(n_pack_mean) if bool(job.get("packing")) else int(job["mbs"])
    )
    return max(1, int(round(float(job["target_gbs"]) / denominator)))


def _default_profile(dataset_id: str) -> Path:
    return ARTIFACT_DIR / f"dataset_profiles/{dataset_id}.qwen3_nothink.jsonl"


def _measurement(job_id: str, expected_gpu_count: int) -> dict[str, Any]:
    root = RESULTS_DIR / job_id
    status = read_json(root / "status.json")
    if status.get("classification") != "success":
        raise ValueError(f"historical job is not successful: {job_id}")
    summaries = [
        read_json(path) for path in sorted((root / "metrics").glob("summary.rank*.json"))
    ]
    if len(summaries) != expected_gpu_count:
        raise ValueError(
            f"rank count mismatch for {job_id}: {len(summaries)} != {expected_gpu_count}"
        )
    measured_seconds = max(_positive(row["measured_seconds"]) for row in summaries)
    logical_samples = sum(
        float(row["measured_totals"]["logical_samples"]) for row in summaries
    )
    effective_tokens = sum(
        float(row["measured_totals"]["effective_tokens"]) for row in summaries
    )
    physical_batches = sum(
        float(row["measured_totals"].get("physical_batches") or 0.0)
        for row in summaries
    )
    return {
        "job_id": job_id,
        "effective_tokens_per_second": effective_tokens / measured_seconds,
        "logical_samples_per_second": logical_samples / measured_seconds,
        "observed_mean_samples_per_physical_batch": (
            logical_samples / physical_batches if physical_batches > 0 else None
        ),
        "measured_seconds_conservative": measured_seconds,
        "execution_attempt_id": status.get("execution_attempt_id"),
    }


def _base_pair(
    *,
    packed: Mapping[str, Any],
    unpacked: Mapping[str, Any],
    n_pack_mean: float,
    profile_path: Path,
) -> dict[str, Any]:
    invariant_keys = (
        "model_id",
        "train_type",
        "dataset_id",
        "cutoff_len",
        "gpu_count",
        "gc",
    )
    mismatches = {
        key: {"packed": packed.get(key), "unpacked": unpacked.get(key)}
        for key in invariant_keys
        if packed.get(key) != unpacked.get(key)
    }
    if mismatches:
        raise ValueError(f"matched configuration drift: {mismatches}")
    if _zero_stage(packed) != _zero_stage(unpacked):
        raise ValueError("matched configuration has different ZeRO stages")
    if str(packed["model_id"]) != MODEL_ID:
        raise ValueError(f"unexpected model: {packed['model_id']}")
    if not profile_path.is_file():
        raise FileNotFoundError(profile_path)
    return {
        "model_id": str(packed["model_id"]),
        "train_type": str(packed["train_type"]),
        "dataset_id": str(packed["dataset_id"]),
        "dataset_profile_path": str(profile_path.resolve()),
        "cutoff_len": int(packed["cutoff_len"]),
        "gpu_count": int(packed["gpu_count"]),
        "zero_stage": _zero_stage(packed),
        "gc": bool(packed["gc"]),
        "n_pack_mean": float(n_pack_mean),
        "packed_ga": _gradient_accumulation(packed, n_pack_mean=n_pack_mean),
        "unpacked_mbs": int(unpacked["mbs"]),
        "unpacked_ga": _gradient_accumulation(unpacked),
        "target_gbs": int(packed["target_gbs"]),
    }


def _unit(
    *,
    cohort: str,
    evidence_unit_id: str,
    setting_id: str,
    packed: Mapping[str, Any],
    unpacked: Mapping[str, Any],
    n_pack_mean: float,
    profile_path: Path,
    packed_measurements: Sequence[Mapping[str, Any]],
    unpacked_measurements: Sequence[Mapping[str, Any]],
    static_source: str,
) -> dict[str, Any]:
    pair = _base_pair(
        packed=packed,
        unpacked=unpacked,
        n_pack_mean=n_pack_mean,
        profile_path=profile_path,
    )
    packed_tps = _geometric_mean(
        [float(row["effective_tokens_per_second"]) for row in packed_measurements]
    )
    unpacked_tps = _geometric_mean(
        [float(row["effective_tokens_per_second"]) for row in unpacked_measurements]
    )
    packed_logical_tps = _geometric_mean(
        [float(row["logical_samples_per_second"]) for row in packed_measurements]
    )
    unpacked_logical_tps = _geometric_mean(
        [float(row["logical_samples_per_second"]) for row in unpacked_measurements]
    )
    observed_pack_means = [
        float(row["observed_mean_samples_per_physical_batch"])
        for row in packed_measurements
        if row.get("observed_mean_samples_per_physical_batch") is not None
    ]
    return pair | {
        "cohort": cohort,
        "evidence_unit_id": evidence_unit_id,
        "setting_id": setting_id,
        "static_mean_source": static_source,
        "packed_job_ids": [str(row["job_id"]) for row in packed_measurements],
        "unpacked_job_ids": [str(row["job_id"]) for row in unpacked_measurements],
        "observed": {
            "packed_effective_tokens_per_second": packed_tps,
            "unpacked_effective_tokens_per_second": unpacked_tps,
            "packed_over_unpacked_effective_ratio": packed_tps / unpacked_tps,
            "packed_logical_samples_per_second": packed_logical_tps,
            "unpacked_logical_samples_per_second": unpacked_logical_tps,
            "packed_over_unpacked_logical_ratio": (
                packed_logical_tps / unpacked_logical_tps
            ),
            "mean_samples_per_physical_batch": (
                statistics.fmean(observed_pack_means) if observed_pack_means else None
            ),
        },
    }


def _legacy_non_abba_units() -> list[dict[str, Any]]:
    effect = read_json(LEGACY_EFFECT)
    static = read_json(LEGACY_STATIC)
    static_by_pair = {
        str(row["request_id"]): row
        for row in static["rows"]
        if str(row["model_id"]) == MODEL_ID
    }
    result = []
    for row in effect["pairs"]:
        config = row["configuration"]
        if str(config["model_id"]) != MODEL_ID:
            continue
        static_row = static_by_pair[str(row["pair_id"])]
        packed_job_id = str(row["job_ids"]["packed"])
        unpacked_job_id = str(row["job_ids"]["unpacked"])
        packed = {
            **config,
            "model_id": config["model_id"],
            "train_type": config["train_type"],
            "dataset_id": config["dataset_id"],
            "cutoff_len": config["cutoff_len"],
            "gpu_count": config["gpu_count"],
            "gc": config["gc"],
            "zero": config["zero"],
            "packing": True,
            "mbs": config["physical_mbs"]["packed"],
            "target_gbs": config["target_gbs"],
        }
        unpacked = dict(packed) | {
            "packing": False,
            "mbs": config["physical_mbs"]["unpacked"],
        }
        n_pack_mean = float(static_row["features"]["mean_samples_per_pack"])
        result.append(
            _unit(
                cohort="legacy_non_abba",
                evidence_unit_id=str(row["pair_id"]),
                setting_id=f"legacy:{config['dataset_id']}:c{config['cutoff_len']}",
                packed=packed,
                unpacked=unpacked,
                n_pack_mean=n_pack_mean,
                profile_path=_default_profile(str(config["dataset_id"])),
                packed_measurements=[
                    _measurement(packed_job_id, int(config["gpu_count"]))
                ],
                unpacked_measurements=[
                    _measurement(unpacked_job_id, int(config["gpu_count"]))
                ],
                static_source=str(LEGACY_STATIC.resolve()),
            )
        )
    return result


def _abba_units() -> list[dict[str, Any]]:
    jobs = [row for row in read_jsonl(ABBA_QUEUE) if str(row["model_id"]) == MODEL_ID]
    static = read_json(LEGACY_STATIC)
    static_by_dataset = {
        str(row["dataset_id"]): row
        for row in static["rows"]
        if str(row["model_id"]) == MODEL_ID
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        grouped[str(job["packing_pair"]["pair_id"])].append(job)
    result = []
    for pair_id, rows in sorted(grouped.items()):
        packed_rows = [row for row in rows if bool(row["packing"])]
        unpacked_rows = [row for row in rows if not bool(row["packing"])]
        if len(packed_rows) != 2 or len(unpacked_rows) != 2:
            raise ValueError(f"incomplete ABBA block: {pair_id}")
        representative = packed_rows[0]
        dataset_id = str(representative["dataset_id"])
        n_pack_mean = float(
            static_by_dataset[dataset_id]["features"]["mean_samples_per_pack"]
        )
        result.append(
            _unit(
                cohort="abba",
                evidence_unit_id=pair_id,
                setting_id=(
                    f"abba:{dataset_id}:c{representative['cutoff_len']}"
                ),
                packed=representative,
                unpacked=unpacked_rows[0],
                n_pack_mean=n_pack_mean,
                profile_path=_default_profile(dataset_id),
                packed_measurements=[
                    _measurement(str(row["job_id"]), int(row["gpu_count"]))
                    for row in packed_rows
                ],
                unpacked_measurements=[
                    _measurement(str(row["job_id"]), int(row["gpu_count"]))
                    for row in unpacked_rows
                ],
                static_source=str(LEGACY_STATIC.resolve()),
            )
        )
    return result


def _calibration_units() -> list[dict[str, Any]]:
    jobs = [
        row for row in read_jsonl(CALIBRATION_QUEUE) if str(row["model_id"]) == MODEL_ID
    ]
    decisions = read_json(CALIBRATION_STATIC)
    mean_by_family = {
        str(row["family_id"]): float(
            row["decision"]["features"]["mean_samples_per_pack"]
        )
        for row in decisions["families"]
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        grouped[str(job["packing_pair_id"])].append(job)
    result = []
    for pair_id, rows in sorted(grouped.items()):
        packed_rows = [row for row in rows if bool(row["packing"])]
        unpacked_rows = [row for row in rows if not bool(row["packing"])]
        if len(packed_rows) != 1 or len(unpacked_rows) != 1:
            raise ValueError(f"incomplete calibration pair: {pair_id}")
        packed = packed_rows[0]
        unpacked = unpacked_rows[0]
        family = str(packed["family_id"])
        result.append(
            _unit(
                cohort="packing_calibration",
                evidence_unit_id=pair_id,
                setting_id=f"calibration:{family}",
                packed=packed,
                unpacked=unpacked,
                n_pack_mean=mean_by_family[family],
                profile_path=Path(str(packed["dataset_profile_path"])),
                packed_measurements=[
                    _measurement(str(packed["job_id"]), int(packed["gpu_count"]))
                ],
                unpacked_measurements=[
                    _measurement(str(unpacked["job_id"]), int(unpacked["gpu_count"]))
                ],
                static_source=str(CALIBRATION_STATIC.resolve()),
            )
        )
    return result


def _unified_units() -> list[dict[str, Any]]:
    jobs = [
        row
        for row in read_jsonl(UNIFIED_QUEUE)
        if str(row["model_id"]) == MODEL_ID
        and str(row.get("evidence_role")) == "unified_packing_matched_formal_fit"
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        key = (
            str(job["dataset_id"]),
            str(job["train_type"]),
            int(job["gpu_count"]),
            _zero_stage(job),
            bool(job["gc"]),
            int(job["cutoff_len"]),
            int(job["repeat"]),
        )
        grouped[key].append(job)
    result = []
    for key, rows in sorted(grouped.items()):
        packed_rows = [row for row in rows if bool(row["packing"])]
        unpacked_rows = [row for row in rows if not bool(row["packing"])]
        if len(packed_rows) != 1 or len(unpacked_rows) != 1:
            raise ValueError(f"incomplete unified pair: {key}")
        packed = packed_rows[0]
        unpacked = unpacked_rows[0]
        contract = packed.get("packing_contract") or {}
        n_pack_mean = float(contract["expected_samples_per_pack"])
        dataset_id = str(packed["dataset_id"])
        result.append(
            _unit(
                cohort="unified_resource",
                evidence_unit_id=str(packed["scenario_id"]).replace("__packed", ""),
                setting_id=f"unified:{dataset_id}",
                packed=packed,
                unpacked=unpacked,
                n_pack_mean=n_pack_mean,
                profile_path=Path(str(packed["dataset_profile_path"])),
                packed_measurements=[
                    _measurement(str(packed["job_id"]), int(packed["gpu_count"]))
                ],
                unpacked_measurements=[
                    _measurement(str(unpacked["job_id"]), int(unpacked["gpu_count"]))
                ],
                static_source=f"{UNIFIED_QUEUE.resolve()}::packing_contract",
            )
        )
    return result


def _predict(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    predictor = FrozenUnpackedPredictor()
    result = []
    for unit in units:
        try:
            actual = predictor.predict(
                unit,
                physical_mbs=int(unit["unpacked_mbs"]),
                gradient_accumulation_steps=int(unit["unpacked_ga"]),
                packing=False,
            )
            virtual = predictor.virtual_prediction(unit)
        except ValueError as error:
            result.append(
                dict(unit)
                | {
                    "prediction": None,
                    "validation": {
                        "evaluable": False,
                        "evidence_class": "not_evaluable_by_frozen_v5",
                        "reason": str(error),
                    },
                }
            )
            continue
        predicted_effective_ratio = (
            float(virtual["predicted_effective_tokens_per_second"])
            / float(actual["predicted_effective_tokens_per_second"])
        )
        predicted_actual_logical_tps = float(
            actual["predicted_effective_tokens_per_second"]
        ) * float(actual["work_per_step"]["logical_samples"]) / float(
            actual["work_per_step"]["effective_tokens"]
        )
        predicted_virtual_logical_tps = float(
            virtual["predicted_effective_tokens_per_second"]
        ) * float(virtual["work_per_step"]["logical_samples"]) / float(
            virtual["work_per_step"]["effective_tokens"]
        )
        predicted_logical_ratio = (
            predicted_virtual_logical_tps / predicted_actual_logical_tps
        )
        observed_effective_ratio = float(
            unit["observed"]["packed_over_unpacked_effective_ratio"]
        )
        observed_logical_ratio = float(
            unit["observed"]["packed_over_unpacked_logical_ratio"]
        )
        effective_transfer_ratio = observed_effective_ratio / predicted_effective_ratio
        logical_transfer_ratio = observed_logical_ratio / predicted_logical_ratio
        observed_pack_mean = unit["observed"]["mean_samples_per_physical_batch"]
        distance = abs(
            math.log2(float(unit["unpacked_mbs"]) / float(unit["n_pack_mean"]))
        )
        result.append(
            dict(unit)
            | {
                "prediction": {
                    "unpacked_actual_mbs_effective_tokens_per_second": float(
                        actual["predicted_effective_tokens_per_second"]
                    ),
                    "unpacked_virtual_mbs_effective_tokens_per_second": float(
                        virtual["predicted_effective_tokens_per_second"]
                    ),
                    "unpacked_actual_mbs_logical_samples_per_second": (
                        predicted_actual_logical_tps
                    ),
                    "unpacked_virtual_mbs_logical_samples_per_second": (
                        predicted_virtual_logical_tps
                    ),
                    "virtual_over_actual_unpacked_effective_ratio": (
                        predicted_effective_ratio
                    ),
                    "virtual_over_actual_unpacked_logical_ratio": (
                        predicted_logical_ratio
                    ),
                    "virtual_mbs_lower": int(virtual["lower_mbs"]),
                    "virtual_mbs_upper": int(virtual["upper_mbs"]),
                },
                "validation": {
                    "evaluable": True,
                    "evidence_class": "v5_anchored_indirect",
                    "primary_metric": "logical_samples_per_second",
                    "t0_transfer_ratio": logical_transfer_ratio,
                    "t0_absolute_percentage_error": abs(
                        logical_transfer_ratio - 1.0
                    ),
                    "effective_tokens_secondary_transfer_ratio": (
                        effective_transfer_ratio
                    ),
                    "effective_tokens_secondary_absolute_percentage_error": abs(
                        effective_transfer_ratio - 1.0
                    ),
                    "unpacked_to_virtual_distance_log2": distance,
                    "unpacked_anchor_near_virtual": distance <= 0.15,
                    "static_to_observed_pack_mean_relative_error": (
                        abs(float(observed_pack_mean) / float(unit["n_pack_mean"]) - 1.0)
                        if observed_pack_mean is not None
                        else None
                    ),
                    "virtual_grid_extrapolation": bool(
                        virtual["is_grid_extrapolation"]
                    ),
                    "outside_h800_feature_support": list(
                        virtual["outside_h800_feature_support"]
                    ),
                    "template_gc_exact": bool(virtual["template_gc_exact"]),
                },
            }
        )
    return result


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in rows if row["validation"].get("evaluable", True)]
    if not rows:
        raise ValueError("metric summary has no evaluable rows")
    errors = [float(row["validation"]["t0_absolute_percentage_error"]) for row in rows]
    ratios = [float(row["validation"]["t0_transfer_ratio"]) for row in rows]
    static_errors = [
        float(row["validation"]["static_to_observed_pack_mean_relative_error"])
        for row in rows
        if row["validation"]["static_to_observed_pack_mean_relative_error"] is not None
    ]
    return {
        "units": len(rows),
        "t0_mape": statistics.fmean(errors),
        "t0_median_ape": percentile(errors, 50),
        "t0_p90_ape": percentile(errors, 90),
        "t0_max_ape": max(errors),
        "t0_geometric_mean_transfer_ratio": _geometric_mean(ratios),
        "static_pack_mean_mape_against_measured_window": (
            statistics.fmean(static_errors) if static_errors else None
        ),
        "near_virtual_anchors": sum(
            bool(row["validation"]["unpacked_anchor_near_virtual"]) for row in rows
        ),
        "outside_support_units": sum(
            bool(row["validation"]["outside_h800_feature_support"]) for row in rows
        ),
    }


def _collapse_settings(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["validation"].get("evaluable") is not True:
            continue
        grouped[(str(row["cohort"]), str(row["setting_id"]))].append(row)
    result = []
    for (cohort, setting_id), repeats in sorted(grouped.items()):
        transfer_ratio = _geometric_mean(
            [float(row["validation"]["t0_transfer_ratio"]) for row in repeats]
        )
        result.append(
            {
                "cohort": cohort,
                "setting_id": setting_id,
                "evidence_units": len(repeats),
                "model_id": MODEL_ID,
                "dataset_id": str(repeats[0]["dataset_id"]),
                "cutoff_len": int(repeats[0]["cutoff_len"]),
                "train_type": str(repeats[0]["train_type"]),
                "gpu_count": int(repeats[0]["gpu_count"]),
                "zero_stage": int(repeats[0]["zero_stage"]),
                "gc": bool(repeats[0]["gc"]),
                "n_pack_mean": float(repeats[0]["n_pack_mean"]),
                "unpacked_mbs": int(repeats[0]["unpacked_mbs"]),
                "t0_transfer_ratio": transfer_ratio,
                "t0_absolute_percentage_error": abs(transfer_ratio - 1.0),
            }
        )
    return result


def _direct_cross_campaign_abba(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reuse old floor/ceil Unpacked controls without a V5 MBS transfer.

    The controls are exact configuration matches but were not interleaved in
    the ABBA block, so this remains weaker than a prospective matched triple.
    """

    abba = [row for row in rows if str(row["cohort"]) == "abba"]
    result = []
    for dataset_id, controls in DIRECT_CONTROL_JOBS.items():
        setting_rows = [row for row in abba if str(row["dataset_id"]) == dataset_id]
        if not setting_rows:
            raise ValueError(f"missing ABBA rows for {dataset_id}")
        packed_job_ids = sorted(
            {
                str(job_id)
                for row in setting_rows
                for job_id in row["packed_job_ids"]
            }
        )
        packed_tps = _geometric_mean(
            [
                float(_measurement(job_id, 1)["logical_samples_per_second"])
                for job_id in packed_job_ids
            ]
        )
        floor_mbs, floor_job_id = controls["floor"]
        ceil_mbs, ceil_job_id = controls["ceil"]
        floor_tps = float(
            _measurement(floor_job_id, 1)["logical_samples_per_second"]
        )
        ceil_tps = float(
            _measurement(ceil_job_id, 1)["logical_samples_per_second"]
        )
        virtual_mbs = float(setting_rows[0]["n_pack_mean"])
        if not float(floor_mbs) <= virtual_mbs <= float(ceil_mbs):
            raise ValueError(
                f"virtual MBS is outside direct controls for {dataset_id}"
            )
        weight = (virtual_mbs - float(floor_mbs)) / (
            float(ceil_mbs) - float(floor_mbs)
        )
        interpolated = math.exp(
            (1.0 - weight) * math.log(floor_tps) + weight * math.log(ceil_tps)
        )
        ratio = packed_tps / interpolated
        result.append(
            {
                "dataset_id": dataset_id,
                "virtual_mbs": virtual_mbs,
                "packed_job_ids": packed_job_ids,
                "floor": {
                    "mbs": floor_mbs,
                    "job_id": floor_job_id,
                    "logical_samples_per_second": floor_tps,
                },
                "ceil": {
                    "mbs": ceil_mbs,
                    "job_id": ceil_job_id,
                    "logical_samples_per_second": ceil_tps,
                },
                "packed_logical_samples_per_second": packed_tps,
                "interpolated_unpacked_logical_samples_per_second": interpolated,
                "packed_over_interpolated_unpacked_ratio": ratio,
                "t0_absolute_percentage_error": abs(ratio - 1.0),
            }
        )
    errors = [float(row["t0_absolute_percentage_error"]) for row in result]
    ratios = [
        float(row["packed_over_interpolated_unpacked_ratio"]) for row in result
    ]
    scalar = _geometric_mean(ratios)
    scalar_errors = [abs(value / scalar - 1.0) for value in ratios]
    return {
        "evidence_class": "direct_mbs_interpolation_cross_campaign_not_interleaved",
        "settings": len(result),
        "t0_mape": statistics.fmean(errors),
        "t0_p90_ape": percentile(errors, 90),
        "t0_max_ape": max(errors),
        "t0_geometric_mean_transfer_ratio": scalar,
        "t1_in_sample_global_scalar": scalar,
        "t1_in_sample_mape": statistics.fmean(scalar_errors),
        "t1_in_sample_max_ape": max(scalar_errors),
        "rows": result,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    unit = report["metrics"]["unit_equal"]
    setting = report["metrics"]["setting_equal"]
    direct = report["metrics"]["direct_cross_campaign_abba"]
    lines = [
        "# Qwen3-14B 历史 Packing 虚拟 MBS 回填验证",
        "",
        "本报告只做离线回填和验证，没有启动 GPU，也没有重拟合模型。",
        "",
        "验证量定义：先用当前冻结的 Unpacked 吞吐模型，把实测 Unpacked MBS 的逻辑样本吞吐换算到静态平均每 pack 样本数；再计算实测 Packed 逻辑样本吞吐与该换算值的比值。T0 假设要求该比值接近 1。",
        "",
        "## 结果",
        "",
        "| 口径 | 数量 | T0 MAPE | P90 APE | 最大 APE | 几何平均比值 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 实验单元等权 | {unit['units']} | {unit['t0_mape']:.2%} | {unit['t0_p90_ape']:.2%} | {unit['t0_max_ape']:.2%} | {unit['t0_geometric_mean_transfer_ratio']:.4f} |",
        f"| 配置等权 | {setting['units']} | {setting['t0_mape']:.2%} | {setting['t0_p90_ape']:.2%} | {setting['t0_max_ape']:.2%} | {setting['t0_geometric_mean_transfer_ratio']:.4f} |",
        f"| 旧 Unpacked 两侧点直接插值 | {direct['settings']} | {direct['t0_mape']:.2%} | {direct['t0_p90_ape']:.2%} | {direct['t0_max_ape']:.2%} | {direct['t0_geometric_mean_transfer_ratio']:.4f} |",
        "",
        f"这 3 个 Qwen3-14B 直接插值配置中，Packed 平均是等效 Unpacked 的 {direct['t1_in_sample_global_scalar']:.4f} 倍。仅在这 3 个点上拟合该常数后，样本内 MAPE 为 {direct['t1_in_sample_mape']:.2%}；样本太少，不能直接发布成全局系数。",
        "",
        "## 分实验批次",
        "",
        "| 批次 | 实验单元 | T0 MAPE | P90 APE | 最大 APE |",
        "|---|---:|---:|---:|---:|",
    ]
    for cohort, metrics in report["metrics"]["by_cohort"].items():
        lines.append(
            f"| {cohort} | {metrics['units']} | {metrics['t0_mape']:.2%} | {metrics['t0_p90_ape']:.2%} | {metrics['t0_max_ape']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "多数旧实验只有一个 Unpacked MBS，因此主汇总依赖当前 V5 做 MBS 换算。另有 3 个数据集能复用历史 floor/ceil 点做直接插值，但这些控制点没有和 Packed 运行交错执行，仍弱于新实验中的直接三臂验证。旧数据可用于跨模型和跨机制的旁证，不应单独作为最终验收集。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    units = [
        *_legacy_non_abba_units(),
        *_abba_units(),
        *_calibration_units(),
        *_unified_units(),
    ]
    predicted = _predict(units)
    evaluated = [row for row in predicted if row["validation"].get("evaluable") is True]
    skipped = [row for row in predicted if row["validation"].get("evaluable") is not True]
    settings = _collapse_settings(predicted)
    direct_cross_campaign = _direct_cross_campaign_abba(predicted)
    by_cohort = {
        cohort: _metric_summary(
            [row for row in evaluated if str(row["cohort"]) == cohort]
        )
        for cohort in sorted({str(row["cohort"]) for row in evaluated})
    }
    model_report = read_json(MAIN_MODEL)
    report: dict[str, Any] = {
        "schema": "sft_h800_qwen14_historical_virtual_mbs_backfill/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "model_id": MODEL_ID,
            "gpu_family": "H800",
            "gpu_work_launched": False,
            "models_refit": False,
            "validation_class": "v5_anchored_indirect",
        },
        "hypothesis": {
            "t0": "Packed physical MBS=1 throughput equals Unpacked throughput at MBS=static mean samples per pack, without a new Packing coefficient",
            "primary_metric": "logical_samples_per_second",
            "test_ratio": "observed logical_samples_per_second(Packed/Unpacked_actual) divided by V5-derived logical_samples_per_second(Unpacked_virtual/Unpacked_actual)",
            "ideal_test_ratio": 1.0,
        },
        "source_bindings": {
            "legacy_effect": {
                "path": str(LEGACY_EFFECT.resolve()),
                "sha256": sha256_file(LEGACY_EFFECT),
            },
            "legacy_static": {
                "path": str(LEGACY_STATIC.resolve()),
                "sha256": sha256_file(LEGACY_STATIC),
            },
            "abba_queue": {
                "path": str(ABBA_QUEUE.resolve()),
                "sha256": sha256_file(ABBA_QUEUE),
            },
            "calibration_queue": {
                "path": str(CALIBRATION_QUEUE.resolve()),
                "sha256": sha256_file(CALIBRATION_QUEUE),
            },
            "calibration_static": {
                "path": str(CALIBRATION_STATIC.resolve()),
                "sha256": sha256_file(CALIBRATION_STATIC),
            },
            "unified_queue": {
                "path": str(UNIFIED_QUEUE.resolve()),
                "sha256": sha256_file(UNIFIED_QUEUE),
            },
            "frozen_throughput_model": {
                "path": str(MAIN_MODEL.resolve()),
                "sha256": sha256_file(MAIN_MODEL),
                "implementation_version": model_report.get("implementation_version"),
                "packing_primary_training_rows": model_report["model_contract"][
                    "packing_primary_training_rows"
                ],
            },
        },
        "metrics": {
            "inventory": {
                "historical_units": len(predicted),
                "evaluated_units": len(evaluated),
                "skipped_units": len(skipped),
                "skipped_reasons": sorted(
                    {str(row["validation"]["reason"]) for row in skipped}
                ),
            },
            "unit_equal": _metric_summary(evaluated),
            "setting_equal": _metric_summary(
                [
                    {
                        "validation": {
                            "evaluable": True,
                            "t0_absolute_percentage_error": row[
                                "t0_absolute_percentage_error"
                            ],
                            "t0_transfer_ratio": row["t0_transfer_ratio"],
                            "static_to_observed_pack_mean_relative_error": None,
                            "unpacked_anchor_near_virtual": False,
                            "outside_h800_feature_support": [],
                        }
                    }
                    for row in settings
                ]
            ),
            "direct_cross_campaign_abba": direct_cross_campaign,
            "by_cohort": by_cohort,
        },
        "limitations": [
            "Historical jobs usually have one measured Unpacked MBS, so V5 supplies the MBS transfer to virtual MBS.",
            "The old cohorts are not an untouched holdout for the current Unpacked model.",
            "Repeated runs are collapsed by setting to avoid treating repeats as independent workload coverage.",
            "Runtime observed samples per physical batch are audit-only; the prediction input is the frozen static dataset replay.",
            "The three direct floor/ceil controls match configuration but were not interleaved with the ABBA Packed runs.",
        ],
        "settings": settings,
        "units": predicted,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    MARKDOWN.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "markdown": str(MARKDOWN),
                "metrics": report["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
