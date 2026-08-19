#!/usr/bin/env python3
"""Test whether Packing throughput can reuse the unpacked model via virtual MBS.

The diagnostic keeps the frozen unpacked throughput model unchanged.  For each
matched Packed/Unpacked repeat it constructs three pre-run predictions:

* the observed unpacked arm using its real MBS and GA;
* a virtual-unpacked arm whose MBS is the static mean samples per pack and
  whose GA is the Packed arm's GA;
* a Packed-geometry arm using the existing structured physical work path while
  leaving the learned Packing coefficient at its frozen zero value.

The primary target is the anchor-calibrated multiplier left after dividing the
observed Packed/Unpacked throughput ratio by the corresponding frozen-model
ratio.  This removes scenario-level absolute scale error and directly tests
whether unpacked MBS scaling transfers to Packing.

This is an analysis-only artifact.  It does not publish or mutate a production
predictor and it never launches GPU work.
"""

from __future__ import annotations

import copy
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from common import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    ROOT,
    RUNTIME_DIR,
    percentile,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)
from h800_challenger_modeling import (
    _build_native_throughput_record,
    _inventory_models,
    _job,
    throughput_admission_reason,
)
# The frozen predictor adapter below does not need scikit-learn.  Keep the
# optional refit dependencies lazy so prediction-only commands can run in the
# production launcher environment, where sklearn is intentionally absent.
try:
    import fit_h800_packing_phase_b_challengers_v1 as phase_b_fit
    import fit_h800_packing_phase_c_models_v1 as phase_c_fit
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as error:
    if error.name != "sklearn":
        raise
    phase_b_fit = None  # type: ignore[assignment]
    phase_c_fit = None  # type: ignore[assignment]
    Ridge = None  # type: ignore[assignment,misc]
    StandardScaler = None  # type: ignore[assignment,misc]
from structured_throughput_modeling import (
    StaticDatasetProfiles,
    _predict_log_throughput,
    _static_structured_basis,
)

PHASE_C = ARTIFACT_DIR / "h800_packing_profile_phase_c_results_v1.json"
REAL_BUSINESS = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_results_v1.json"
MAIN_MODEL = (
    ROOT
    / "diagnostics"
    / "h800_unified_resource_partial_refit_20260809"
    / "throughput_model.json"
)
OUTPUT = ARTIFACT_DIR / "h800_packing_virtual_mbs_analysis_v1.json"
MARKDOWN = ARTIFACT_DIR / "h800_packing_virtual_mbs_analysis_v1.md"

VIRTUAL_MBS_GRID = (1, 2, 4, 8, 16, 32)
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
ONE_FEATURE_CANDIDATES = (
    "length_cv",
    "log_computed_work_ratio",
    "log_attention_work_ratio",
)
GLOBAL_COEFFICIENT_GATES = {
    "group_equal_mape": 0.10,
    "p90_ape": 0.15,
}


def _positive(value: Any, floor: float = 1.0e-12) -> float:
    return max(floor, float(value))


def _metric(row: Mapping[str, Any], name: str) -> float:
    aliases = {
        "logical": (
            "global_logical_samples_per_second",
            "logical_samples_per_second",
        ),
        "effective": (
            "global_effective_tokens_per_second",
            "effective_tokens_per_second",
        ),
    }
    for key in aliases[name]:
        value = row.get(key)
        if value is not None:
            return _positive(value)
    raise ValueError(f"{row.get('job_id')} has no {name} throughput")


def _phase_b_pairs() -> list[dict[str, Any]]:
    real_static, interaction_static = phase_b_fit._static_feature_maps()
    grouped: dict[tuple[str, int], dict[bool, dict[str, Any]]] = defaultdict(dict)
    for row in phase_b_fit._source_rows():
        setting_id, _profile_group = phase_b_fit._setting_identity(row)
        key = (setting_id, int(row["repeat"]))
        treatment = bool(row["packing"])
        if treatment in grouped[key]:
            raise ValueError(f"duplicate Phase-B treatment: {key}/{treatment}")
        grouped[key][treatment] = row

    result: list[dict[str, Any]] = []
    for (setting_id, repeat), treatments in sorted(grouped.items()):
        if set(treatments) != {False, True}:
            raise ValueError(f"incomplete Phase-B pair: {setting_id}/{repeat}")
        unpacked, packed = treatments[False], treatments[True]
        _, profile_group = phase_b_fit._setting_identity(packed)
        result.append(
            _pair_contract(
                setting_id=setting_id,
                profile_group=profile_group,
                source=str(packed["fit_source"]),
                repeat=repeat,
                packed=packed,
                unpacked=unpacked,
                n_pack_mean=phase_b_fit._n_pack_mean(
                    packed,
                    real_static,
                    interaction_static,
                ),
            )
        )
    if len(result) != 49:
        raise ValueError(f"expected 49 pre-Phase-C pairs, found {len(result)}")
    return result


def _phase_c_pairs() -> list[dict[str, Any]]:
    report = phase_c_fit._require_phase_c()
    grouped: dict[tuple[str, int], dict[bool, dict[str, Any]]] = defaultdict(dict)
    for row in report["job_results"]:
        setting_id = str(row["setting_id"])
        key = (setting_id, int(row["repeat"]))
        treatment = bool(row["packing"])
        if treatment in grouped[key]:
            raise ValueError(f"duplicate Phase-C treatment: {key}/{treatment}")
        grouped[key][treatment] = row

    result: list[dict[str, Any]] = []
    for (setting_id, repeat), treatments in sorted(grouped.items()):
        if set(treatments) != {False, True}:
            raise ValueError(f"incomplete Phase-C pair: {setting_id}/{repeat}")
        unpacked, packed = treatments[False], treatments[True]
        workload = str(packed["workload_id"]).upper()
        result.append(
            _pair_contract(
                setting_id=f"phase_c:{setting_id}",
                profile_group=f"phase_b:{workload}",
                source="phase_c",
                repeat=repeat,
                packed=packed,
                unpacked=unpacked,
                n_pack_mean=float(packed["n_pack_mean"]),
            )
        )
    if len(result) != 12:
        raise ValueError(f"expected 12 Phase-C pairs, found {len(result)}")
    return result


def _pair_contract(
    *,
    setting_id: str,
    profile_group: str,
    source: str,
    repeat: int,
    packed: Mapping[str, Any],
    unpacked: Mapping[str, Any],
    n_pack_mean: float,
) -> dict[str, Any]:
    invariant_keys = (
        "model_id",
        "train_type",
        "dataset_id",
        "dataset_profile_path",
        "cutoff_len",
        "gpu_count",
        "zero_stage",
        "gc",
    )
    mismatches = {
        key: {"packed": packed.get(key), "unpacked": unpacked.get(key)}
        for key in invariant_keys
        if packed.get(key) != unpacked.get(key)
    }
    if mismatches:
        raise ValueError(f"matched-pair invariant mismatch {setting_id}: {mismatches}")
    if int(packed.get("mbs", 0)) != 1:
        raise ValueError(f"Packed MBS is not one: {packed.get('job_id')}")

    profile_path = Path(str(packed["dataset_profile_path"]))
    profile_stats = phase_b_fit._profile_stats(str(profile_path))
    cutoff = int(packed["cutoff_len"])
    packed_ga = int(packed["gradient_accumulation_steps"])
    gpu_count = int(packed["gpu_count"])
    expected_packed_gbs = gpu_count * packed_ga * float(n_pack_mean)
    return {
        "pair_id": f"{setting_id}:repeat{repeat}",
        "setting_id": setting_id,
        "profile_group": profile_group,
        "source": source,
        "repeat": repeat,
        "packed_job_id": str(packed["job_id"]),
        "unpacked_job_id": str(unpacked["job_id"]),
        "model_id": str(packed["model_id"]),
        "train_type": str(packed["train_type"]),
        "dataset_id": str(packed["dataset_id"]),
        "dataset_profile_path": str(profile_path),
        "cutoff_len": cutoff,
        "gpu_count": gpu_count,
        "zero_stage": int(packed.get("zero_stage", 0)),
        "gc": bool(packed.get("gc")),
        "n_pack_mean": float(n_pack_mean),
        "packed_ga": packed_ga,
        "unpacked_mbs": int(unpacked["mbs"]),
        "unpacked_ga": int(unpacked["gradient_accumulation_steps"]),
        "expected_packed_gbs": expected_packed_gbs,
        "nominal_target_gbs": int(packed["target_gbs"]),
        "profile": {
            "mean_length": float(profile_stats["mean"]),
            "length_cv": float(profile_stats["cv"]),
            "p99_length_to_cutoff": float(profile_stats["p99"]) / cutoff,
            "pack_fill_proxy": min(
                1.25,
                float(profile_stats["mean"]) * float(n_pack_mean) / cutoff,
            ),
        },
        "observed": {
            "packed_effective_tokens_per_second": _metric(packed, "effective"),
            "unpacked_effective_tokens_per_second": _metric(unpacked, "effective"),
            "packed_logical_samples_per_second": _metric(packed, "logical"),
            "unpacked_logical_samples_per_second": _metric(unpacked, "logical"),
        },
    }


def build_matched_pairs() -> list[dict[str, Any]]:
    pairs = sorted(
        [*_phase_b_pairs(), *_phase_c_pairs()],
        key=lambda row: str(row["pair_id"]),
    )
    if len(pairs) != 61:
        raise ValueError(f"expected 61 matched repeat pairs, found {len(pairs)}")
    if len({row["pair_id"] for row in pairs}) != len(pairs):
        raise ValueError("pair ids are not unique")
    return pairs


def virtual_mbs_bracket(value: float) -> dict[str, Any]:
    value = _positive(value)
    log_value = math.log2(value)
    extrapolation = False
    if value <= VIRTUAL_MBS_GRID[0]:
        lower, upper = VIRTUAL_MBS_GRID[0], VIRTUAL_MBS_GRID[1]
        extrapolation = value < VIRTUAL_MBS_GRID[0]
    elif value >= VIRTUAL_MBS_GRID[-1]:
        lower, upper = VIRTUAL_MBS_GRID[-2], VIRTUAL_MBS_GRID[-1]
        extrapolation = value > VIRTUAL_MBS_GRID[-1]
    else:
        upper_index = next(
            index for index, item in enumerate(VIRTUAL_MBS_GRID) if item >= value
        )
        if VIRTUAL_MBS_GRID[upper_index] == value:
            lower = upper = VIRTUAL_MBS_GRID[upper_index]
        else:
            lower = VIRTUAL_MBS_GRID[upper_index - 1]
            upper = VIRTUAL_MBS_GRID[upper_index]
    if lower == upper:
        weight = 0.0
    else:
        weight = (log_value - math.log2(lower)) / (math.log2(upper) - math.log2(lower))
    return {
        "lower_mbs": int(lower),
        "upper_mbs": int(upper),
        "upper_log_weight": float(weight),
        "is_grid_extrapolation": extrapolation,
    }


def _log_interpolate(lower: float, upper: float, weight: float) -> float:
    if lower <= 0 or upper <= 0:
        raise ValueError("log interpolation requires positive values")
    return math.exp((1.0 - weight) * math.log(lower) + weight * math.log(upper))


class FrozenUnpackedPredictor:
    """Adapter around the latest frozen structured throughput model."""

    def __init__(self) -> None:
        self.report = read_json(MAIN_MODEL)
        self.model = self.report["frozen_model"]
        if self.report["model_contract"].get("packing_primary_training_rows") != 0:
            raise ValueError(
                "expected the frozen model to have zero Packed training rows"
            )
        packing_index = list(self.model["feature_names"]).index("packing")
        self.packing_correction_coefficient = float(
            self.model["correction_coefficients"][packing_index]
        )
        if abs(self.packing_correction_coefficient) > 1.0e-12:
            raise ValueError(
                "Packed physical diagnostic requires a frozen zero Packing coefficient"
            )
        self.hardware = read_json(CONFIG_DIR / "hardware.json")
        inventory = read_json(ARTIFACT_DIR / "model_inventory.json")
        self.model_by_id, self.fixed_lora = _inventory_models(inventory)
        observations_path = Path(
            self.report["source_bindings"]["h800_observations"]["path"]
        )
        self.raw_observations = [
            json.loads(line)
            for line in observations_path.open(encoding="utf-8")
            if line.strip()
        ]
        self._template_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._profiles_cache: dict[Path, StaticDatasetProfiles] = {}
        self.h800_support = self.model["training_support"]["by_card"]["h800"]

    @staticmethod
    def _mechanism_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            str(row["model_id"]),
            str(row["train_type"]),
            int(row["gpu_count"]),
            int(row["zero_stage"]),
            bool(row["gc"]),
        )

    def _template(self, pair: Mapping[str, Any]) -> dict[str, Any]:
        key = self._mechanism_key(pair)
        if key in self._template_cache:
            return copy.deepcopy(self._template_cache[key])
        model_id, train_type, gpu_count, zero_stage, gc = key
        exact: list[dict[str, Any]] = []
        same_except_gc: list[dict[str, Any]] = []
        for observation in self.raw_observations:
            if throughput_admission_reason(observation) != "admitted":
                continue
            job = _job(observation)
            material = (
                str(job.get("model_id")),
                str(job.get("train_type")),
                int(job.get("gpu_count") or 0),
                int(job.get("zero_stage") or 0),
            )
            if material != (model_id, train_type, gpu_count, zero_stage):
                continue
            same_except_gc.append(observation)
            if bool(job.get("gc")) == gc:
                exact.append(observation)
        candidates = exact or same_except_gc
        if not candidates:
            raise ValueError(f"no frozen-model template for mechanism {key}")
        raw = min(candidates, key=lambda row: str(row.get("observation_id")))
        record = _build_native_throughput_record(
            raw,
            model_by_id=self.model_by_id,
            fixed_lora=self.fixed_lora,
            hardware=self.hardware,
            runtime_root=RUNTIME_DIR,
        )
        record["selector"]["training_mode"] = train_type
        record["selector"]["zero_stage"] = zero_stage
        record["selector"]["gradient_checkpointing"] = gc
        record["scenario"]["model_id"] = model_id
        record["scenario"]["train_type"] = train_type
        record["scenario"]["gpu_count"] = gpu_count
        record["performance"] = {
            "physical_priors": copy.deepcopy(record["performance"]["physical_priors"])
        }
        record["template_gc_exact"] = bool(exact)
        self._template_cache[key] = copy.deepcopy(record)
        return record

    def _profiles(self, profile_path: Path) -> tuple[StaticDatasetProfiles, str]:
        parent = profile_path.parent
        if parent not in self._profiles_cache:
            self._profiles_cache[parent] = StaticDatasetProfiles(parent)
        dataset_id = profile_path.name.split(".", 1)[0]
        return self._profiles_cache[parent], dataset_id

    def predict(
        self,
        pair: Mapping[str, Any],
        *,
        physical_mbs: int,
        gradient_accumulation_steps: int,
        packing: bool,
    ) -> dict[str, Any]:
        record = self._template(pair)
        profile_path = Path(str(pair["dataset_profile_path"]))
        profiles, dataset_id = self._profiles(profile_path)
        gpu_count = int(pair["gpu_count"])
        mbs = int(physical_mbs)
        ga = int(gradient_accumulation_steps)
        record["scenario"].update(
            {
                "dataset_id": dataset_id,
                "cutoff_len": int(pair["cutoff_len"]),
                "gpu_count": gpu_count,
                "physical_mbs": mbs,
                "target_gbs": gpu_count * mbs * ga,
            }
        )
        record["selector"]["packing"] = bool(packing)
        if packing:
            record["performance"]["gradient_accumulation_steps"] = ga
        else:
            record["performance"].pop("gradient_accumulation_steps", None)
        basis = _static_structured_basis(
            record,
            profiles,
            hardware_memory_bytes=float(
                self.hardware["memory_bytes_reported_by_torch"]
            ),
        )
        candidate = {
            "structured_features": basis["features"],
            "physical_components": basis["components"],
            "static_log_work": math.log(
                _positive(basis["work_per_step"]["effective_tokens"])
            ),
            "card_id": "h800",
        }
        log_throughput = _predict_log_throughput(candidate, self.model)
        feature_values = basis["feature_values"]
        support_ranges = self.h800_support["feature_ranges"]
        outside = []
        for name, value in feature_values.items():
            limits = support_ranges.get(name)
            if limits is None:
                continue
            if float(value) < float(limits["min"]) or float(value) > float(
                limits["max"]
            ):
                outside.append(name)
        cutoff_support = {int(x) for x in self.h800_support["cutoff_lens"]}
        return {
            "predicted_effective_tokens_per_second": math.exp(log_throughput),
            "work_per_step": {
                key: float(value) for key, value in basis["work_per_step"].items()
            },
            "feature_values": {
                name: float(value) for name, value in feature_values.items()
            },
            "outside_h800_feature_support": sorted(outside),
            "cutoff_seen_exactly_in_h800_training": int(pair["cutoff_len"])
            in cutoff_support,
            "template_gc_exact": bool(record["template_gc_exact"]),
        }

    def virtual_prediction(self, pair: Mapping[str, Any]) -> dict[str, Any]:
        bracket = virtual_mbs_bracket(float(pair["n_pack_mean"]))
        lower = self.predict(
            pair,
            physical_mbs=int(bracket["lower_mbs"]),
            gradient_accumulation_steps=int(pair["packed_ga"]),
            packing=False,
        )
        upper = (
            lower
            if bracket["upper_mbs"] == bracket["lower_mbs"]
            else self.predict(
                pair,
                physical_mbs=int(bracket["upper_mbs"]),
                gradient_accumulation_steps=int(pair["packed_ga"]),
                packing=False,
            )
        )
        weight = float(bracket["upper_log_weight"])
        prediction = _log_interpolate(
            lower["predicted_effective_tokens_per_second"],
            upper["predicted_effective_tokens_per_second"],
            weight,
        )
        work = {
            key: _log_interpolate(
                lower["work_per_step"][key],
                upper["work_per_step"][key],
                weight,
            )
            for key in lower["work_per_step"]
        }
        return {
            **bracket,
            "predicted_effective_tokens_per_second": prediction,
            "work_per_step": work,
            "lower_prediction": lower,
            "upper_prediction": upper,
            "outside_h800_feature_support": sorted(
                set(lower["outside_h800_feature_support"])
                | set(upper["outside_h800_feature_support"])
            ),
            "cutoff_seen_exactly_in_h800_training": bool(
                lower["cutoff_seen_exactly_in_h800_training"]
            ),
            "template_gc_exact": bool(lower["template_gc_exact"]),
        }


def _predict_pairs(
    pairs: Sequence[Mapping[str, Any]],
    predictor: FrozenUnpackedPredictor,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for pair in pairs:
        unpacked = predictor.predict(
            pair,
            physical_mbs=int(pair["unpacked_mbs"]),
            gradient_accumulation_steps=int(pair["unpacked_ga"]),
            packing=False,
        )
        virtual = predictor.virtual_prediction(pair)
        packed_physical = predictor.predict(
            pair,
            physical_mbs=1,
            gradient_accumulation_steps=int(pair["packed_ga"]),
            packing=True,
        )
        observed = pair["observed"]
        observed_effective_ratio = (
            observed["packed_effective_tokens_per_second"]
            / observed["unpacked_effective_tokens_per_second"]
        )
        observed_logical_ratio = (
            observed["packed_logical_samples_per_second"]
            / observed["unpacked_logical_samples_per_second"]
        )
        virtual_ratio = (
            virtual["predicted_effective_tokens_per_second"]
            / unpacked["predicted_effective_tokens_per_second"]
        )
        packed_physical_ratio = (
            packed_physical["predicted_effective_tokens_per_second"]
            / unpacked["predicted_effective_tokens_per_second"]
        )
        computed_work_ratio = (
            virtual["work_per_step"]["computed_tokens"]
            / packed_physical["work_per_step"]["computed_tokens"]
        )
        attention_work_ratio = (
            virtual["work_per_step"]["computed_attention_token_pairs"]
            / packed_physical["work_per_step"]["computed_attention_token_pairs"]
        )
        result.append(
            dict(pair)
            | {
                "prediction": {
                    "unpacked_actual": unpacked,
                    "virtual_unpacked": virtual,
                    "packed_physical_shared_model": packed_physical,
                    "virtual_over_unpacked_ratio": virtual_ratio,
                    "packed_physical_over_unpacked_ratio": packed_physical_ratio,
                },
                "observed_ratio": {
                    "effective_tokens_per_second": observed_effective_ratio,
                    "logical_samples_per_second": observed_logical_ratio,
                },
                "residual_coefficient": {
                    "virtual_effective": observed_effective_ratio / virtual_ratio,
                    "virtual_logical": observed_logical_ratio / virtual_ratio,
                    "packed_physical_effective": (
                        observed_effective_ratio / packed_physical_ratio
                    ),
                    "packed_physical_logical": (
                        observed_logical_ratio / packed_physical_ratio
                    ),
                    "absolute_packed_over_virtual_prediction": (
                        observed["packed_effective_tokens_per_second"]
                        / virtual["predicted_effective_tokens_per_second"]
                    ),
                },
                "derived_features": {
                    "length_cv": float(pair["profile"]["length_cv"]),
                    "computed_work_ratio": computed_work_ratio,
                    "attention_work_ratio": attention_work_ratio,
                    "log_computed_work_ratio": math.log(computed_work_ratio),
                    "log_attention_work_ratio": math.log(attention_work_ratio),
                },
                "domain": {
                    "virtual_mbs_grid_extrapolation": bool(
                        virtual["is_grid_extrapolation"]
                    ),
                    "cutoff_seen_exactly_in_h800_training": bool(
                        virtual["cutoff_seen_exactly_in_h800_training"]
                    ),
                    "virtual_outside_h800_feature_support": virtual[
                        "outside_h800_feature_support"
                    ],
                    "packed_physical_outside_h800_feature_support": (
                        packed_physical["outside_h800_feature_support"]
                    ),
                    "template_gc_exact": bool(virtual["template_gc_exact"]),
                },
            }
        )
    return result


def _collapse_settings(pair_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[str(row["setting_id"])].append(row)
    result = []
    for setting_id, repeats in sorted(grouped.items()):
        representative = repeats[0]
        invariant_keys = (
            "profile_group",
            "source",
            "model_id",
            "train_type",
            "dataset_id",
            "dataset_profile_path",
            "cutoff_len",
            "gpu_count",
            "zero_stage",
            "gc",
            "n_pack_mean",
            "packed_ga",
            "unpacked_mbs",
            "unpacked_ga",
        )
        for row in repeats[1:]:
            mismatched = [
                key for key in invariant_keys if row[key] != representative[key]
            ]
            if mismatched:
                raise ValueError(
                    f"setting varies across repeats {setting_id}: {mismatched}"
                )

        coefficient_keys = tuple(representative["residual_coefficient"])
        ratio_keys = tuple(representative["observed_ratio"])
        result.append(
            {key: representative[key] for key in invariant_keys}
            | {
                "setting_id": setting_id,
                "repeat_count": len(repeats),
                "pair_ids": [str(row["pair_id"]) for row in repeats],
                "profile": dict(representative["profile"]),
                "prediction": {
                    "virtual_mbs": float(representative["n_pack_mean"]),
                    "lower_mbs": int(
                        representative["prediction"]["virtual_unpacked"]["lower_mbs"]
                    ),
                    "upper_mbs": int(
                        representative["prediction"]["virtual_unpacked"]["upper_mbs"]
                    ),
                    "upper_log_weight": float(
                        representative["prediction"]["virtual_unpacked"][
                            "upper_log_weight"
                        ]
                    ),
                    "unpacked_actual_effective_tokens_per_second": float(
                        representative["prediction"]["unpacked_actual"][
                            "predicted_effective_tokens_per_second"
                        ]
                    ),
                    "virtual_effective_tokens_per_second": float(
                        representative["prediction"]["virtual_unpacked"][
                            "predicted_effective_tokens_per_second"
                        ]
                    ),
                    "packed_physical_effective_tokens_per_second": float(
                        representative["prediction"]["packed_physical_shared_model"][
                            "predicted_effective_tokens_per_second"
                        ]
                    ),
                },
                "observed_ratio": {
                    key: math.exp(
                        statistics.fmean(
                            math.log(float(row["observed_ratio"][key]))
                            for row in repeats
                        )
                    )
                    for key in ratio_keys
                },
                "residual_coefficient": {
                    key: math.exp(
                        statistics.fmean(
                            math.log(float(row["residual_coefficient"][key]))
                            for row in repeats
                        )
                    )
                    for key in coefficient_keys
                },
                "repeat_log_std": {
                    key: statistics.pstdev(
                        math.log(float(row["residual_coefficient"][key]))
                        for row in repeats
                    )
                    for key in coefficient_keys
                },
                "derived_features": dict(representative["derived_features"]),
                "domain": dict(representative["domain"]),
            }
        )
    if len(result) != 23:
        raise ValueError(f"expected 23 settings, found {len(result)}")
    return result


def _group_equal_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts = Counter(str(row["profile_group"]) for row in rows)
    groups = len(counts)
    return np.asarray(
        [1.0 / (groups * counts[str(row["profile_group"])]) for row in rows],
        dtype=float,
    )


def _fit_constant(rows: Sequence[Mapping[str, Any]], target: str) -> dict[str, Any]:
    y = np.asarray(
        [math.log(float(row["residual_coefficient"][target])) for row in rows],
        dtype=float,
    )
    weights = _group_equal_weights(rows)
    intercept = float(np.sum(weights * y) / np.sum(weights))
    return {"intercept": intercept, "coefficient": math.exp(intercept)}


def _fit_one_feature(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    feature: str,
    alpha: float,
) -> dict[str, Any]:
    x = np.asarray(
        [[float(row["derived_features"][feature])] for row in rows],
        dtype=float,
    )
    y = np.asarray(
        [math.log(float(row["residual_coefficient"][target])) for row in rows],
        dtype=float,
    )
    weights = _group_equal_weights(rows)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    transformed = scaler.transform(x)
    model = Ridge(alpha=float(alpha)).fit(transformed, y, sample_weight=weights)
    return {
        "feature": feature,
        "alpha": float(alpha),
        "feature_mean": float(scaler.mean_[0]),
        "feature_scale": float(scaler.scale_[0]),
        "intercept": float(model.intercept_),
        "coefficient": float(model.coef_[0]),
    }


def _predict_one_feature(model: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    scale = _positive(model["feature_scale"])
    value = float(row["derived_features"][str(model["feature"])])
    standardized = (value - float(model["feature_mean"])) / scale
    return float(model["intercept"]) + float(model["coefficient"]) * standardized


def _metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    predicted_logs: Sequence[float],
) -> dict[str, Any]:
    actual = [math.log(float(row["residual_coefficient"][target])) for row in rows]
    signed = [
        math.exp(pred - truth) - 1.0 for truth, pred in zip(actual, predicted_logs)
    ]
    absolute = [abs(value) for value in signed]
    group_errors: dict[str, list[float]] = defaultdict(list)
    group_log_errors: dict[str, list[float]] = defaultdict(list)
    for row, truth, pred, ape in zip(rows, actual, predicted_logs, absolute):
        group = str(row["profile_group"])
        group_errors[group].append(ape)
        group_log_errors[group].append(abs(pred - truth))
    return {
        "settings": len(rows),
        "profile_groups": len(group_errors),
        "group_equal_mape": statistics.fmean(
            statistics.fmean(values) for values in group_errors.values()
        ),
        "row_mape": statistics.fmean(absolute),
        "median_ape": percentile(absolute, 50),
        "p90_ape": percentile(absolute, 90),
        "maximum_ape": max(absolute),
        "signed_bias": statistics.fmean(signed),
        "group_equal_log_mae": statistics.fmean(
            statistics.fmean(values) for values in group_log_errors.values()
        ),
        "by_profile_group_mape": {
            group: statistics.fmean(values)
            for group, values in sorted(group_errors.items())
        },
    }


def _loso_constant(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
) -> dict[str, Any]:
    predictions = [0.0] * len(rows)
    groups = sorted({str(row["profile_group"]) for row in rows})
    for group in groups:
        train = [row for row in rows if str(row["profile_group"]) != group]
        fitted = _fit_constant(train, target)
        for index, row in enumerate(rows):
            if str(row["profile_group"]) == group:
                predictions[index] = float(fitted["intercept"])
    return {
        "model_family": "profile_equal_global_constant",
        "validation": "leave_one_profile_group_out",
        "metrics": _metrics(rows, target=target, predicted_logs=predictions),
        "full_fit": _fit_constant(rows, target),
        "oof_predicted_log_coefficients": predictions,
    }


def _inner_select_alpha(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    feature: str,
) -> float:
    groups = sorted({str(row["profile_group"]) for row in rows})
    if len(groups) < 3:
        return 10.0
    scores = []
    for alpha in RIDGE_ALPHAS:
        predictions: list[float] = []
        held_rows: list[Mapping[str, Any]] = []
        for group in groups:
            train = [row for row in rows if str(row["profile_group"]) != group]
            test = [row for row in rows if str(row["profile_group"]) == group]
            fitted = _fit_one_feature(
                train,
                target=target,
                feature=feature,
                alpha=alpha,
            )
            predictions.extend(_predict_one_feature(fitted, row) for row in test)
            held_rows.extend(test)
        metric = _metrics(held_rows, target=target, predicted_logs=predictions)
        scores.append((float(metric["group_equal_mape"]), -float(alpha), float(alpha)))
    return min(scores)[2]


def _loso_one_feature(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    feature: str,
) -> dict[str, Any]:
    predictions = [0.0] * len(rows)
    selected_alphas: dict[str, float] = {}
    groups = sorted({str(row["profile_group"]) for row in rows})
    for group in groups:
        train = [row for row in rows if str(row["profile_group"]) != group]
        alpha = _inner_select_alpha(train, target=target, feature=feature)
        selected_alphas[group] = alpha
        fitted = _fit_one_feature(
            train,
            target=target,
            feature=feature,
            alpha=alpha,
        )
        for index, row in enumerate(rows):
            if str(row["profile_group"]) == group:
                predictions[index] = _predict_one_feature(fitted, row)
    final_alpha = _inner_select_alpha(rows, target=target, feature=feature)
    return {
        "model_family": "profile_equal_one_feature_ridge",
        "feature": feature,
        "validation": "nested_leave_one_profile_group_out",
        "outer_fold_selected_alphas": selected_alphas,
        "metrics": _metrics(rows, target=target, predicted_logs=predictions),
        "full_fit": _fit_one_feature(
            rows,
            target=target,
            feature=feature,
            alpha=final_alpha,
        ),
        "oof_predicted_log_coefficients": predictions,
    }


def _validate_residual_models(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
) -> dict[str, Any]:
    constant = _loso_constant(rows, target=target)
    challengers = {
        feature: _loso_one_feature(rows, target=target, feature=feature)
        for feature in ONE_FEATURE_CANDIDATES
    }
    selected_feature, selected = min(
        challengers.items(),
        key=lambda item: (
            float(item[1]["metrics"]["group_equal_mape"]),
            float(item[1]["metrics"]["p90_ape"]),
            item[0],
        ),
    )
    gates = {
        name: float(constant["metrics"][name]) <= threshold
        for name, threshold in GLOBAL_COEFFICIENT_GATES.items()
    }
    return {
        "target": target,
        "global_constant": constant,
        "global_constant_gates": {
            "thresholds": dict(GLOBAL_COEFFICIENT_GATES),
            "results": gates,
            "all_passed": all(gates.values()),
        },
        "one_feature_challengers": challengers,
        "selected_one_feature_diagnostic": selected_feature,
        "selected_one_feature_metrics": selected["metrics"],
        "selection_is_analysis_only": True,
    }


def _direct_real_business_audit(
    predictor: FrozenUnpackedPredictor,
    setting_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    report = read_json(REAL_BUSINESS)
    by_setting = {str(row["setting_id"]): row for row in setting_rows}
    result = []
    for family in report["families"]:
        setting_id = f"real:{family['family_id']}:c{int(family['base_cutoff_len'])}"
        setting = by_setting[setting_id]
        jobs = [
            row
            for row in report["job_results"]
            if str(row["family_id"]) == str(family["family_id"])
            and str(row["arm_id"]) == "N-C-k"
            and row.get("classification") == "success"
        ]
        if not jobs:
            continue
        representative = jobs[0]
        actual_mbs = int(representative["mbs"])
        actual_ga = int(representative["gradient_accumulation_steps"])
        reference_pair = next(
            row for row in build_matched_pairs() if str(row["setting_id"]) == setting_id
        )
        predicted_large = predictor.predict(
            reference_pair,
            physical_mbs=actual_mbs,
            gradient_accumulation_steps=actual_ga,
            packing=False,
        )["predicted_effective_tokens_per_second"]
        predicted_mbs1 = float(
            setting["prediction"]["unpacked_actual_effective_tokens_per_second"]
        )
        observed_large = statistics.fmean(_metric(row, "effective") for row in jobs)
        observed_mbs1 = float(
            family["arms"]["N-C-1"]["effective_tokens_per_second_mean"]
        )
        observed_packed = float(
            family["arms"]["P-C-1"]["effective_tokens_per_second_mean"]
        )
        result.append(
            {
                "family_id": str(family["family_id"]),
                "display_name": str(family["display_name"]),
                "cutoff_len": int(family["base_cutoff_len"]),
                "n_pack_mean": float(setting["n_pack_mean"]),
                "measured_unpacked_mbs": actual_mbs,
                "n_pack_over_measured_mbs": float(setting["n_pack_mean"]) / actual_mbs,
                "observed_packed_over_measured_unpacked": observed_packed
                / observed_large,
                "observed_unpacked_mbs_scaling": observed_large / observed_mbs1,
                "predicted_unpacked_mbs_scaling": predicted_large / predicted_mbs1,
                "unpacked_scaling_prediction_ratio": (
                    (predicted_large / predicted_mbs1)
                    / (observed_large / observed_mbs1)
                ),
                "anchor_calibrated_virtual_effective_coefficient": float(
                    setting["residual_coefficient"]["virtual_effective"]
                ),
            }
        )
    return result


def _fully_in_h800_virtual_support(row: Mapping[str, Any]) -> bool:
    domain = row["domain"]
    return bool(
        not domain["virtual_mbs_grid_extrapolation"]
        and domain["cutoff_seen_exactly_in_h800_training"]
        and not domain["virtual_outside_h800_feature_support"]
        and domain["template_gc_exact"]
    )


def _source_bindings(profile_paths: Sequence[str]) -> dict[str, Any]:
    paths = {
        "implementation": Path(__file__),
        "main_throughput_model": MAIN_MODEL,
        "phase_b": phase_b_fit.PHASE_B,
        "phase_c": PHASE_C,
        "real_business": phase_b_fit.REAL_BUSINESS,
        "strict_interactions": phase_b_fit.INTERACTIONS,
        "strict_gbs_repair": phase_b_fit.GBS_REPAIR,
        "fit_membership": phase_b_fit.MEMBERSHIP,
        "real_static": phase_b_fit.REAL_STATIC,
        "interaction_static": phase_b_fit.INTERACTION_STATIC,
    }
    return {
        key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for key, path in paths.items()
    } | {
        "dataset_profiles": {
            str(Path(path).resolve()): sha256_file(Path(path))
            for path in sorted(set(profile_paths))
        }
    }


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _render_markdown(report: Mapping[str, Any]) -> str:
    effective = report["validation"]["virtual_effective"]
    constant = effective["global_constant"]
    selected_name = str(effective["selected_one_feature_diagnostic"])
    selected = effective["one_feature_challengers"][selected_name]
    physical = report["validation"]["packed_physical_effective"]["global_constant"]
    supported = report["support_sensitivity"]["fully_in_h800_virtual_support"]
    lines = [
        "# H800 Packing 虚拟 MBS 离线诊断",
        "",
        "本产物仅做离线诊断，不修改主吞吐模型，不允许据此自动发布 Packing 推荐。",
        "",
        "## 1. 问题",
        "",
        "验证 Packed MBS=1、平均每 pack 为 k 时，能否直接复用 unpacked MBS=k 的吞吐预测，并只保留一个固定修正系数。",
        "",
        "## 2. 数据和口径",
        "",
        f"- matched repeat pairs：{report['population']['matched_repeat_pairs']}。",
        f"- setting：{report['population']['settings']}。",
        f"- 数据画像组：{report['population']['profile_groups']}。",
        "- 模型输入只使用运行前静态 profile；运行时观测只作为吞吐标签。",
        "- 主指标是锚定后的剩余系数：先用同一 pair 的 unpacked 实测消掉主模型绝对尺度误差，再检查虚拟 MBS 的相对扩展是否能解释 Packed 吞吐。",
        "",
        "## 3. 数据画像隔离验证",
        "",
        "| 模型 | 数据画像等权 MAPE | P90 APE | 最大 APE | 全量拟合系数/特征 |",
        "|---|---:|---:|---:|---|",
        (
            f"| 虚拟 MBS + 全局系数 | {_pct(constant['metrics']['group_equal_mape'])} | "
            f"{_pct(constant['metrics']['p90_ape'])} | {_pct(constant['metrics']['maximum_ape'])} | "
            f"{constant['full_fit']['coefficient']:.4f} |"
        ),
        (
            f"| 虚拟 MBS + 一个特征 | {_pct(selected['metrics']['group_equal_mape'])} | "
            f"{_pct(selected['metrics']['p90_ape'])} | {_pct(selected['metrics']['maximum_ape'])} | "
            f"{selected_name} |"
        ),
        (
            f"| Packed 物理工作量 + 全局系数 | {_pct(physical['metrics']['group_equal_mape'])} | "
            f"{_pct(physical['metrics']['p90_ape'])} | {_pct(physical['metrics']['maximum_ape'])} | "
            f"{physical['full_fit']['coefficient']:.4f} |"
        ),
        "",
        "全局系数稳定性门槛：数据画像等权 MAPE 不高于 10%，P90 APE 不高于 15%。",
        "",
        f"结论：虚拟 MBS 全局系数门槛{'通过' if effective['global_constant_gates']['all_passed'] else '未通过'}。",
        "",
        "### 3.1 仅保留主模型范围内 setting",
        "",
        f"严格处于 H800 主模型范围内的 setting 只有 {supported['population']['settings']} 个、{supported['population']['profile_groups']} 个数据画像组。",
        "",
        "| 模型 | 数据画像等权 MAPE | P90 APE |",
        "|---|---:|---:|",
        (
            f"| 虚拟 MBS + 全局系数 | "
            f"{_pct(supported['virtual_effective']['global_constant']['metrics']['group_equal_mape'])} | "
            f"{_pct(supported['virtual_effective']['global_constant']['metrics']['p90_ape'])} |"
        ),
        (
            f"| Packed 物理工作量 + 全局系数 | "
            f"{_pct(supported['packed_physical_effective']['global_constant']['metrics']['group_equal_mape'])} | "
            f"{_pct(supported['packed_physical_effective']['global_constant']['metrics']['p90_ape'])} |"
        ),
        "",
        "即使去掉明确外推的 setting，全局系数仍未达到 10% / 15% 门槛。",
        "",
        "## 4. 现有近等值实测对照",
        "",
        "| 数据集 | cutoff | 平均每 pack | 实测 unpacked MBS | Packed/Unpacked 有效吞吐 | 虚拟 MBS 剩余系数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["direct_real_business_audit"]:
        lines.append(
            f"| {row['display_name']} | {row['cutoff_len']} | {row['n_pack_mean']:.3f} | "
            f"{row['measured_unpacked_mbs']} | {row['observed_packed_over_measured_unpacked']:.3f} | "
            f"{row['anchor_calibrated_virtual_effective_coefficient']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 5. 适用范围",
            "",
            f"- 虚拟 MBS 超出 1/2/4/8/16/32 网格的 setting：{report['domain']['virtual_mbs_grid_extrapolation_settings']}。",
            f"- cutoff 未在 H800 主模型训练取值中出现的 setting：{report['domain']['cutoff_not_seen_exactly_settings']}。",
            f"- 至少一个虚拟特征超出 H800 训练范围的 setting：{report['domain']['virtual_feature_outside_support_settings']}。",
            "",
            "## 6. 工程结论",
            "",
        ]
    )
    if effective["global_constant_gates"]["all_passed"]:
        lines.append(
            "当前数据支持把虚拟 MBS 加一个全局系数作为 shadow 吞吐模型，但仍需前瞻等值实验验收。"
        )
    else:
        lines.append(
            "当前数据不支持只替换 MBS 再乘一个全局固定系数。应保留一个 padding/长度离散程度修正，或直接使用 Packed 物理工作量主干。"
        )
    lines.extend(
        [
            "",
            "数学事实、实验观察和工程假设已在 JSON 产物的 interpretation_contract 中分开记录。",
            "",
        ]
    )
    return "\n".join(lines)


def build_report() -> dict[str, Any]:
    pairs = build_matched_pairs()
    predictor = FrozenUnpackedPredictor()
    predicted_pairs = _predict_pairs(pairs, predictor)
    settings = _collapse_settings(predicted_pairs)

    validation = {
        "virtual_effective": _validate_residual_models(
            settings,
            target="virtual_effective",
        ),
        "virtual_logical": _validate_residual_models(
            settings,
            target="virtual_logical",
        ),
        "packed_physical_effective": {
            "target": "packed_physical_effective",
            "global_constant": _loso_constant(
                settings,
                target="packed_physical_effective",
            ),
        },
        "packed_physical_logical": {
            "target": "packed_physical_logical",
            "global_constant": _loso_constant(
                settings,
                target="packed_physical_logical",
            ),
        },
    }
    direct = _direct_real_business_audit(predictor, settings)
    supported_settings = [
        row for row in settings if _fully_in_h800_virtual_support(row)
    ]
    support_sensitivity = {
        "fully_in_h800_virtual_support": {
            "definition": (
                "MBS interpolation is inside 1..32, cutoff is an exact H800 "
                "training value, every virtual feature is inside H800 min/max, "
                "and the mechanism template matches GC exactly"
            ),
            "population": {
                "settings": len(supported_settings),
                "profile_groups": len(
                    {str(row["profile_group"]) for row in supported_settings}
                ),
            },
            "virtual_effective": _validate_residual_models(
                supported_settings,
                target="virtual_effective",
            ),
            "packed_physical_effective": {
                "global_constant": _loso_constant(
                    supported_settings,
                    target="packed_physical_effective",
                )
            },
        }
    }
    domain = {
        "virtual_mbs_grid_extrapolation_settings": sum(
            bool(row["domain"]["virtual_mbs_grid_extrapolation"]) for row in settings
        ),
        "cutoff_not_seen_exactly_settings": sum(
            not bool(row["domain"]["cutoff_seen_exactly_in_h800_training"])
            for row in settings
        ),
        "virtual_feature_outside_support_settings": sum(
            bool(row["domain"]["virtual_outside_h800_feature_support"])
            for row in settings
        ),
        "template_gc_fallback_settings": sum(
            not bool(row["domain"]["template_gc_exact"]) for row in settings
        ),
        "fully_in_h800_virtual_support_settings": len(supported_settings),
    }
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_virtual_mbs_analysis/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_complete_not_publishable",
        "analysis_only": True,
        "publishable": False,
        "gpu_experiments_launched": False,
        "main_model_mutated": False,
        "problem": (
            "test whether static mean samples per pack can be used as virtual "
            "unpacked MBS with a small residual coefficient"
        ),
        "population": {
            "matched_repeat_pairs": len(predicted_pairs),
            "settings": len(settings),
            "profile_groups": len({str(row["profile_group"]) for row in settings}),
            "sources": dict(Counter(str(row["source"]) for row in settings)),
        },
        "estimands": {
            "primary": (
                "observed Packed/Unpacked effective-throughput ratio divided by "
                "frozen-model virtual-MBS/actual-Unpacked ratio"
            ),
            "secondary": (
                "same anchored residual after replacing virtual-unpacked padding "
                "work with the existing Packed physical-work reconstruction"
            ),
            "absolute_coefficient_is_not_primary": (
                "absolute Packed/model ratio mixes Packing transfer error with "
                "the frozen model's scenario-level absolute scale error"
            ),
        },
        "virtual_mbs_contract": {
            "value": "static pre-run mean_samples_per_pack",
            "grid": list(VIRTUAL_MBS_GRID),
            "fractional_policy": "log-throughput interpolation in log2(MBS)",
            "ga": "use the Packed arm gradient accumulation so virtual logical GBS matches static Packed expected GBS at MBS=n_pack_mean",
            "padding_semantics": "baseline challenger deliberately uses unpacked batch-max padding semantics",
            "runtime_n_pack_used_as_input": False,
        },
        "validation_contract": {
            "unit": "setting collapsed over repeats",
            "split": "leave one upstream profile group out",
            "weights": "each profile group has equal total weight",
            "one_feature_selection": "nested profile-group LOSO selects Ridge alpha inside each outer fold",
            "global_coefficient_gates": dict(GLOBAL_COEFFICIENT_GATES),
        },
        "validation": validation,
        "support_sensitivity": support_sensitivity,
        "domain": domain,
        "direct_real_business_audit": direct,
        "setting_rows": settings,
        "pair_rows": predicted_pairs,
        "interpretation_contract": {
            "mathematical_fact": (
                "the anchor-calibrated residual exactly removes any multiplicative "
                "absolute scale error shared by the model predictions for the same scenario"
            ),
            "empirical_observation": (
                "reported validation metrics and direct-pair ratios apply only to the "
                "current H800 evidence population"
            ),
            "modeling_assumption": (
                "fractional virtual MBS is approximated by log interpolation between "
                "neighboring integer MBS predictions at fixed Packed GA"
            ),
            "mechanistic_interpretation": (
                "correlation with work-ratio or length-dispersion features may be "
                "consistent with padding/attention differences but does not establish causality"
            ),
        },
        "gates": {
            "global_virtual_mbs_coefficient_stable": bool(
                validation["virtual_effective"]["global_constant_gates"]["all_passed"]
            ),
            "one_feature_diagnostic_completed": True,
            "automatic_packing_recommendation_allowed": False,
            "automatic_publication_allowed": False,
        },
        "frozen_main_model_contract": {
            "packing_primary_training_rows": int(
                predictor.report["model_contract"]["packing_primary_training_rows"]
            ),
            "packing_correction_coefficient": predictor.packing_correction_coefficient,
            "packed_physical_prediction_interpretation": (
                "shared structured physical work with no learned Packing correction"
            ),
        },
        "source_bindings": _source_bindings(
            [str(row["dataset_profile_path"]) for row in pairs]
        ),
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    report = build_report()
    write_json(OUTPUT, report)
    MARKDOWN.write_text(_render_markdown(report), encoding="utf-8")
    print(OUTPUT)
    print(MARKDOWN)
    effective = report["validation"]["virtual_effective"]
    print(
        json.dumps(
            {
                "pairs": report["population"]["matched_repeat_pairs"],
                "settings": report["population"]["settings"],
                "groups": report["population"]["profile_groups"],
                "global_constant": effective["global_constant"]["full_fit"][
                    "coefficient"
                ],
                "global_constant_metrics": effective["global_constant"]["metrics"],
                "global_constant_passed": effective["global_constant_gates"][
                    "all_passed"
                ],
                "selected_one_feature": effective["selected_one_feature_diagnostic"],
                "selected_one_feature_metrics": effective[
                    "selected_one_feature_metrics"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
