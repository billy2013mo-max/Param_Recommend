#!/usr/bin/env python3
"""Unified inference API and CLI for the frozen structured throughput model.

The predictor converts a user-facing training configuration into the static
work, physical time components and structured features expected by
``structured_throughput_modeling.py``.  It then loads the frozen parameters,
predicts effective tokens/s, ranks comparable candidates and reports whether
the request is inside the empirical training support.

This module is inference-only.  It does not launch GPU work, mutate queues or
check whether a candidate is memory-safe.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

from common import (
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)
from h800_challenger_modeling import _inventory_models
from h800_theory_basis import _model_geometry
from rtx4090_challenger_modeling import (
    DEFAULT_COLLECTIVE_BANDWIDTH_BYTES_S as RTX4090_LINK_BANDWIDTH,
)
from rtx4090_challenger_modeling import (
    DEFAULT_COLLECTIVE_LATENCY_SECONDS as RTX4090_COLLECTIVE_LATENCY,
)
from rtx4090_challenger_modeling import (
    DEFAULT_HBM_BANDWIDTH_BYTES_S as RTX4090_HBM_BANDWIDTH,
)
from structured_throughput_modeling import (
    FEATURE_NAMES,
    StaticDatasetProfiles,
    _predict_log_throughput,
    _static_structured_basis,
    validate_report,
    validate_static_profile_report,
)


SCHEMA = "sft_throughput_prediction/v1"
IMPLEMENTATION_VERSION = (
    "sft_throughput_predictor_impl/2026-07-28.unified-api-cli-v1"
)

DEFAULT_MODEL_ARTIFACT = (
    ROOT / "artifacts" / "structured_throughput_modeling.json"
)
DEFAULT_STATIC_PROFILE_ARTIFACT = (
    ROOT / "artifacts" / "static_workload_profiles.json"
)
DEFAULT_DATASET_PROFILE_DIR = ROOT / "artifacts" / "dataset_profiles"
DEFAULT_MODEL_INVENTORY = ROOT / "artifacts" / "model_inventory.json"
DEFAULT_H800_HARDWARE = ROOT / "config" / "hardware.json"
DEFAULT_H800_THEORY_BASIS = (
    ROOT / "artifacts" / "h800_theory_basis.json"
)
DEFAULT_RTX4090_HARDWARE = (
    ROOT
    / "campaigns"
    / "rtx4090_20260717"
    / "config"
    / "hardware.json"
)

ZERO_STAGE_ALIASES = {
    "none": 0,
    "zero0": 0,
    "0": 0,
    "zero1": 1,
    "1": 1,
    "zero2": 2,
    "2": 2,
    "zero3": 3,
    "3": 3,
}


@dataclass(frozen=True)
class HardwareSpec:
    """Physical inputs required by the structured throughput formula."""

    hardware_id: str
    card_id: str
    display_name: str
    memory_bytes: float
    dense_bf16_peak_flops_per_gpu: float
    hbm_bandwidth_bytes_per_second: float
    intra_node_bandwidth_bytes_per_second: float
    collective_latency_seconds: float
    default_kernel_path: str | None
    builtin: bool

    def physical_priors(self) -> dict[str, float]:
        return {
            "dense_bf16_peak_flops_per_gpu": (
                self.dense_bf16_peak_flops_per_gpu
            ),
            "hbm_bandwidth_bytes_per_second": (
                self.hbm_bandwidth_bytes_per_second
            ),
            "intra_node_bandwidth_bytes_per_second": (
                self.intra_node_bandwidth_bytes_per_second
            ),
            "collective_latency_seconds": (
                self.collective_latency_seconds
            ),
        }

    def public_summary(self) -> dict[str, Any]:
        return {
            "hardware_id": self.hardware_id,
            "card_id": self.card_id,
            "display_name": self.display_name,
            "memory_bytes": self.memory_bytes,
            "dense_bf16_peak_flops_per_gpu": (
                self.dense_bf16_peak_flops_per_gpu
            ),
            "hbm_bandwidth_bytes_per_second": (
                self.hbm_bandwidth_bytes_per_second
            ),
            "intra_node_bandwidth_bytes_per_second": (
                self.intra_node_bandwidth_bytes_per_second
            ),
            "collective_latency_seconds": (
                self.collective_latency_seconds
            ),
            "default_kernel_path": self.default_kernel_path,
            "builtin": self.builtin,
        }


def _required_positive_int(
    value: Any,
    field: str,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a positive integer") from error
    if result <= 0 or float(value) != result:
        raise ValueError(f"{field} must be a positive integer")
    return result


def _required_positive_float(
    value: Any,
    field: str,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be positive") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _optional_bool(
    value: Any,
    field: str,
    *,
    default: bool,
) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false")
    return value


def _offload_requested(request: Mapping[str, Any]) -> bool:
    """Detect any optimizer/parameter offload without silently dropping it.

    V1 has no offload coefficients.  A configured mapping is therefore
    treated as enabled even when its exact DeepSpeed ratio is not understood;
    callers can reject it explicitly instead of evaluating it as no-offload.
    """

    keys = (
        "offload",
        "offload_optimizer",
        "offload_param",
        "optimizer_offload",
        "parameter_offload",
    )
    for key in keys:
        value = request.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                return True
            continue
        if isinstance(value, Mapping):
            return bool(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)) and float(value) > 0.0:
                return True
            continue
        text = str(value).strip().lower()
        if text not in {"", "0", "false", "none", "off", "disabled"}:
            return True
    for key in ("offload_optimizer_ratio", "offload_param_ratio", "optimizer_offload_ratio"):
        value = request.get(key)
        if value is None:
            continue
        try:
            if math.isfinite(float(value)) and float(value) > 0.0:
                return True
        except (TypeError, ValueError):
            return True
    return False


def _zero_stage(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("zero_stage must be 0, 1, 2 or 3")
    normalized = str(value if value is not None else "none").lower()
    if normalized not in ZERO_STAGE_ALIASES:
        raise ValueError(
            "zero_stage must be one of none/zero0/zero1/zero2/zero3"
        )
    return ZERO_STAGE_ALIASES[normalized]


def _first_present(
    mapping: Mapping[str, Any],
    names: Sequence[str],
) -> Any:
    for name in names:
        if mapping.get(name) is not None:
            return mapping[name]
    return None


def validate_prediction_report(
    report: Mapping[str, Any],
) -> None:
    """Validate checksum, safety flags and ranking invariants."""

    if report.get("schema") != SCHEMA:
        raise ValueError("Throughput prediction schema mismatch")
    unsigned = dict(report)
    digest = unsigned.pop("report_sha256", None)
    if digest != sha256_json(unsigned):
        raise ValueError("Throughput prediction report hash mismatch")
    if (
        report.get("gpu_experiments_launched") is not False
        or report.get("queues_mutated") is not False
        or report.get("requires_memory_safety_filter") is not True
        or report.get(
            "single_output_used_for_absolute_and_ranking"
        )
        is not True
    ):
        raise ValueError("Throughput prediction safety contract drifted")

    predictions = report.get("predictions")
    groups = report.get("ranking_groups")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Throughput prediction report has no rows")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Throughput prediction report has no groups")
    request_ids = [str(row.get("request_id")) for row in predictions]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Throughput prediction request ids drifted")

    rows_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        throughput = float(
            row.get("predicted_effective_tokens_per_second")
        )
        step_seconds = float(row.get("predicted_step_seconds"))
        if (
            not math.isfinite(throughput)
            or throughput <= 0.0
            or not math.isfinite(step_seconds)
            or step_seconds <= 0.0
            or row.get("memory_safety_checked") is not False
        ):
            raise ValueError(
                "Throughput prediction row violates numeric/safety contract"
            )
        confidence = row.get("confidence") or {}
        if confidence.get("label") not in {
            "supported",
            "caution",
            "low",
        }:
            raise ValueError(
                "Throughput prediction confidence label drifted"
            )
        rows_by_group[str(row.get("comparison_group"))].append(row)

    group_ids = set()
    for group in groups:
        group_id = str(group.get("comparison_group"))
        if group_id in group_ids or group_id not in rows_by_group:
            raise ValueError(
                "Throughput prediction ranking group drifted"
            )
        group_ids.add(group_id)
        rows = rows_by_group[group_id]
        ordered = sorted(
            rows,
            key=lambda row: int(row["rank_within_group"]),
        )
        if [int(row["rank_within_group"]) for row in ordered] != list(
            range(1, len(rows) + 1)
        ):
            raise ValueError(
                "Throughput prediction ranks are not contiguous"
            )
        ranked_ids = [str(row["request_id"]) for row in ordered]
        if ranked_ids != [
            str(value) for value in group.get("ranked_request_ids") or []
        ]:
            raise ValueError(
                "Throughput prediction group/request ordering drifted"
            )
        throughputs = [
            float(row["predicted_effective_tokens_per_second"])
            for row in ordered
        ]
        if throughputs != sorted(throughputs, reverse=True):
            raise ValueError(
                "Throughput prediction ranks are not throughput-descending"
            )
    if group_ids != set(rows_by_group):
        raise ValueError(
            "Throughput prediction rows contain an undeclared group"
        )


class ThroughputPredictor:
    """Load the frozen model and predict/rank user configurations."""

    def __init__(
        self,
        *,
        model_artifact: Path = DEFAULT_MODEL_ARTIFACT,
        static_profile_artifact: Path = (
            DEFAULT_STATIC_PROFILE_ARTIFACT
        ),
        dataset_profile_dir: Path = DEFAULT_DATASET_PROFILE_DIR,
        additional_dataset_profile_dir: Path | None = None,
        model_inventory: Path = DEFAULT_MODEL_INVENTORY,
        h800_hardware: Path = DEFAULT_H800_HARDWARE,
        h800_theory_basis: Path = DEFAULT_H800_THEORY_BASIS,
        rtx4090_hardware: Path = DEFAULT_RTX4090_HARDWARE,
        strict_bindings: bool = True,
    ) -> None:
        self.model_artifact_path = Path(model_artifact)
        self.static_profile_artifact_path = Path(
            static_profile_artifact
        )
        self.dataset_profile_dir = Path(dataset_profile_dir)
        self.additional_dataset_profile_dir = (
            Path(additional_dataset_profile_dir)
            if additional_dataset_profile_dir is not None
            else None
        )
        self.model_inventory_path = Path(model_inventory)
        self.h800_hardware_path = Path(h800_hardware)
        self.h800_theory_basis_path = Path(h800_theory_basis)
        self.rtx4090_hardware_path = Path(rtx4090_hardware)
        self.strict_bindings = bool(strict_bindings)

        self.report = read_json(self.model_artifact_path)
        validate_report(self.report)
        self.frozen_model = self.report["frozen_model"]

        self.static_profile_report = read_json(
            self.static_profile_artifact_path
        )
        validate_static_profile_report(self.static_profile_report)
        self.profiles = StaticDatasetProfiles(
            self.dataset_profile_dir
        )
        if self.additional_dataset_profile_dir is not None:
            additional_profiles = StaticDatasetProfiles(
                self.additional_dataset_profile_dir
            )
            duplicates = sorted(
                set(self.profiles.rows) & set(additional_profiles.rows)
            )
            if duplicates:
                raise ValueError(
                    "Additional dataset profiles must not override "
                    "frozen profile ids: "
                    + ", ".join(duplicates)
                )
            self.profiles.rows.update(additional_profiles.rows)

        inventory = read_json(self.model_inventory_path)
        self.models, self.fixed_lora = _inventory_models(inventory)
        self.hardware = self._load_builtin_hardware()
        self._validate_bindings()

    def _binding_matches(
        self,
        binding_name: str,
        path: Path,
    ) -> bool:
        binding = self.report["source_bindings"].get(binding_name)
        return bool(
            isinstance(binding, Mapping)
            and binding.get("sha256") == sha256_file(path)
        )

    def _validate_bindings(self) -> None:
        implementation_path = Path(
            self.report["source_bindings"]["implementation"]["path"]
        )
        if not self._binding_matches(
            "implementation",
            implementation_path,
        ):
            raise ValueError(
                "Frozen model implementation hash no longer matches "
                "structured_throughput_modeling.py; refit before inference"
            )

        required = (
            (
                "static_workload_profiles",
                self.static_profile_artifact_path,
            ),
            ("h800_model_inventory", self.model_inventory_path),
            ("h800_hardware", self.h800_hardware_path),
            ("rtx4090_hardware", self.rtx4090_hardware_path),
        )
        mismatches = [
            name
            for name, path in required
            if not self._binding_matches(name, path)
        ]
        profile_bindings = self.report["source_bindings"].get(
            "dataset_profiles"
        ) or {}
        for dataset_id, binding in profile_bindings.items():
            path = (
                self.dataset_profile_dir
                / f"{dataset_id}.qwen3_nothink.jsonl"
            )
            if (
                not path.is_file()
                or sha256_file(path) != binding.get("sha256")
            ):
                mismatches.append(
                    f"dataset_profile:{dataset_id}"
                )
        if mismatches and self.strict_bindings:
            raise ValueError(
                "Frozen model input binding mismatch: "
                + ", ".join(sorted(mismatches))
            )
        self.binding_mismatches = sorted(mismatches)

    def _load_builtin_hardware(
        self,
    ) -> dict[str, HardwareSpec]:
        h800 = read_json(self.h800_hardware_path)
        h800_theory = read_json(self.h800_theory_basis_path)
        h800_priors = h800_theory["physical_priors"]["values"]
        rtx4090 = read_json(self.rtx4090_hardware_path)

        h800_spec = HardwareSpec(
            hardware_id="h800",
            card_id="h800",
            display_name=str(
                h800.get("name_reported_by_driver") or "NVIDIA H800"
            ),
            memory_bytes=_required_positive_float(
                h800["memory_bytes_reported_by_torch"],
                "H800 memory_bytes",
            ),
            dense_bf16_peak_flops_per_gpu=_required_positive_float(
                h800[
                    "bf16_dense_peak_flops_per_second_for_mfu"
                ],
                "H800 dense peak",
            ),
            hbm_bandwidth_bytes_per_second=_required_positive_float(
                h800_priors["memory_bandwidth_bytes_per_s"],
                "H800 HBM bandwidth",
            ),
            intra_node_bandwidth_bytes_per_second=(
                _required_positive_float(
                    h800_priors[
                        "collective_bandwidth_bytes_per_s"
                    ],
                    "H800 link bandwidth",
                )
            ),
            collective_latency_seconds=_required_positive_float(
                h800_priors["collective_latency_seconds"],
                "H800 collective latency",
            ),
            default_kernel_path=(
                "fa3_orig+liger_fused_ce+adamw_torch_fused"
            ),
            builtin=True,
        )
        rtx4090_spec = HardwareSpec(
            hardware_id="rtx4090",
            card_id="rtx4090",
            display_name=str(
                rtx4090.get("name_reported_by_driver")
                or "NVIDIA GeForce RTX 4090"
            ),
            memory_bytes=_required_positive_float(
                rtx4090["memory_bytes_reported_by_torch"],
                "RTX 4090 memory_bytes",
            ),
            dense_bf16_peak_flops_per_gpu=_required_positive_float(
                rtx4090[
                    "bf16_dense_peak_flops_per_second_for_mfu"
                ],
                "RTX 4090 dense peak",
            ),
            hbm_bandwidth_bytes_per_second=(
                float(RTX4090_HBM_BANDWIDTH)
            ),
            intra_node_bandwidth_bytes_per_second=(
                float(RTX4090_LINK_BANDWIDTH)
            ),
            collective_latency_seconds=float(
                RTX4090_COLLECTIVE_LATENCY
            ),
            default_kernel_path=(
                "fa2+liger_fused_ce+adamw_torch_fused"
            ),
            builtin=True,
        )
        aliases = {
            "h800": h800_spec,
            "nvidia_h800": h800_spec,
            "rtx4090": rtx4090_spec,
            "4090": rtx4090_spec,
            "nvidia_rtx4090": rtx4090_spec,
        }
        return aliases

    def _resolve_hardware(
        self,
        request: Mapping[str, Any],
    ) -> HardwareSpec:
        custom = request.get("hardware")
        if custom is None:
            hardware_id = str(
                request.get("hardware_id")
                or request.get("gpu_type")
                or ""
            ).strip().lower()
            if not hardware_id:
                raise ValueError(
                    "hardware_id is required when hardware is absent"
                )
            if "h800" in hardware_id:
                hardware_id = "h800"
            elif "4090" in hardware_id:
                hardware_id = "rtx4090"
            if hardware_id not in self.hardware:
                raise ValueError(
                    f"Unknown hardware_id {hardware_id!r}; pass a "
                    "complete custom hardware object"
                )
            return self.hardware[hardware_id]
        if not isinstance(custom, Mapping):
            raise ValueError("hardware must be a JSON object")

        hardware_id = str(
            custom.get("hardware_id")
            or request.get("hardware_id")
            or "custom"
        )
        card_id = str(
            custom.get("card_id") or hardware_id
        ).strip().lower()
        memory = _first_present(
            custom,
            (
                "memory_bytes",
                "memory_bytes_reported_by_torch",
                "memory_capacity_bytes",
            ),
        )
        peak = _first_present(
            custom,
            (
                "dense_bf16_peak_flops_per_gpu",
                "dense_bf16_peak_flops_per_second",
                "dense_peak_flops_per_s",
            ),
        )
        hbm = _first_present(
            custom,
            (
                "hbm_bandwidth_bytes_per_second",
                "memory_bandwidth_bytes_per_s",
            ),
        )
        link = _first_present(
            custom,
            (
                "intra_node_bandwidth_bytes_per_second",
                "collective_bandwidth_bytes_per_s",
            ),
        )
        latency = _first_present(
            custom,
            ("collective_latency_seconds",),
        )
        return HardwareSpec(
            hardware_id=hardware_id,
            card_id=card_id,
            display_name=str(
                custom.get("display_name")
                or custom.get("name_reported_by_driver")
                or hardware_id
            ),
            memory_bytes=_required_positive_float(
                memory,
                "hardware.memory_bytes",
            ),
            dense_bf16_peak_flops_per_gpu=(
                _required_positive_float(
                    peak,
                    "hardware.dense_bf16_peak_flops_per_gpu",
                )
            ),
            hbm_bandwidth_bytes_per_second=(
                _required_positive_float(
                    hbm,
                    "hardware.hbm_bandwidth_bytes_per_second",
                )
            ),
            intra_node_bandwidth_bytes_per_second=(
                _required_positive_float(
                    link,
                    (
                        "hardware."
                        "intra_node_bandwidth_bytes_per_second"
                    ),
                )
            ),
            collective_latency_seconds=_required_positive_float(
                latency if latency is not None else 1.0e-5,
                "hardware.collective_latency_seconds",
            ),
            default_kernel_path=(
                str(custom["default_kernel_path"])
                if custom.get("default_kernel_path")
                else None
            ),
            builtin=False,
        )

    def _resolve_model(
        self,
        request: Mapping[str, Any],
        *,
        training_mode: str,
    ) -> tuple[str, dict[str, Any], dict[str, int], int]:
        custom = request.get("model")
        model_id = str(
            request.get("model_id")
            or (
                custom.get("id")
                if isinstance(custom, Mapping)
                else ""
            )
            or ""
        ).strip()
        if not model_id:
            raise ValueError("model_id is required")
        if custom is None:
            if model_id not in self.models:
                raise ValueError(
                    f"Unknown model_id {model_id!r}; pass a complete "
                    "custom model object"
                )
            model = dict(self.models[model_id])
            request_parameters = _first_present(
                request,
                ("model_parameters", "actual_parameters"),
            )
            if (
                request_parameters is not None
                and _required_positive_int(
                    request_parameters,
                    "model_parameters",
                )
                != int(model["actual_parameters"])
            ):
                raise ValueError(
                    f"model_parameters does not match catalog entry "
                    f"{model_id}"
                )
        else:
            if not isinstance(custom, Mapping):
                raise ValueError("model must be a JSON object")
            model = dict(custom)
            model.setdefault("id", model_id)
            if model.get("actual_parameters") is None:
                model["actual_parameters"] = _first_present(
                    request,
                    ("model_parameters", "actual_parameters"),
                )

        lora_rank = _required_positive_int(
            request.get("lora_rank")
            or self.fixed_lora.get("rank")
            or 32,
            "lora_rank",
        )
        lora_target = str(
            request.get("lora_target")
            or self.fixed_lora.get("target")
            or "all"
        )
        lora_contract = {
            **self.fixed_lora,
            "rank": lora_rank,
            "target": lora_target,
        }
        base_parameters = _required_positive_int(
            model.get("actual_parameters"),
            "model.actual_parameters",
        )
        geometry = _model_geometry(
            {
                "model_id": model_id,
                "model_parameters": base_parameters,
                "train_type": training_mode,
            },
            model,
            lora_contract,
        )
        return model_id, model, geometry, lora_rank

    def _normalized_request(
        self,
        request: Mapping[str, Any],
        *,
        input_index: int,
    ) -> dict[str, Any]:
        training_mode = str(
            request.get("training_mode")
            or request.get("train_type")
            or ""
        ).strip().lower()
        if training_mode not in {"full", "lora"}:
            raise ValueError("training_mode must be full or lora")

        hardware = self._resolve_hardware(request)
        model_id, model, geometry, lora_rank = self._resolve_model(
            request,
            training_mode=training_mode,
        )
        dataset_id = str(
            request.get("dataset_id") or ""
        ).strip()
        if not dataset_id:
            raise ValueError("dataset_id is required")
        if dataset_id not in self.profiles.rows:
            raise ValueError(
                f"No static profile for dataset_id {dataset_id!r} "
                f"under {self.dataset_profile_dir}"
            )

        gpu_count = _required_positive_int(
            request.get("gpu_count"),
            "gpu_count",
        )
        physical_mbs = _required_positive_int(
            request.get("physical_mbs")
            if request.get("physical_mbs") is not None
            else request.get("mbs"),
            "physical_mbs",
        )
        target_gbs = _required_positive_int(
            request.get("target_gbs"),
            "target_gbs",
        )
        cutoff_len = _required_positive_int(
            request.get("cutoff_len"),
            "cutoff_len",
        )
        packing = _optional_bool(
            request.get("packing"),
            "packing",
            default=False,
        )
        offload = _offload_requested(request)
        gradient_checkpointing = _optional_bool(
            (
                request.get("gradient_checkpointing")
                if request.get("gradient_checkpointing") is not None
                else request.get("gc")
            ),
            "gradient_checkpointing",
            default=False,
        )
        zero_stage = _zero_stage(
            request.get("zero_stage")
            if request.get("zero_stage") is not None
            else request.get("zero")
        )
        dtype = str(
            request.get("dtype") or "bf16"
        ).strip().lower()

        if not packing and target_gbs % (gpu_count * physical_mbs):
            raise ValueError(
                "For packing=false, target_gbs must be divisible by "
                "gpu_count * physical_mbs"
            )
        maximum_positions = (
            model.get("max_position_embeddings")
            or (
                model.get("text_config") or {}
            ).get("max_position_embeddings")
        )
        if (
            maximum_positions is not None
            and cutoff_len > int(maximum_positions)
        ):
            raise ValueError(
                f"cutoff_len {cutoff_len} exceeds model maximum "
                f"position length {maximum_positions}"
            )

        kernel_path = request.get("kernel_path")
        if kernel_path is None:
            kernel_path = hardware.default_kernel_path
        if not kernel_path:
            raise ValueError(
                "kernel_path is required for custom hardware"
            )
        kernel_path = str(kernel_path)

        request_id = str(
            request.get("request_id")
            or request.get("candidate_id")
            or f"candidate_{input_index + 1:04d}"
        )
        gradient_accumulation = request.get(
            "gradient_accumulation_steps"
        )
        if gradient_accumulation is not None:
            gradient_accumulation = _required_positive_int(
                gradient_accumulation,
                "gradient_accumulation_steps",
            )
            if not packing:
                expected = target_gbs // (
                    gpu_count * physical_mbs
                )
                if gradient_accumulation != expected:
                    raise ValueError(
                        "gradient_accumulation_steps conflicts with "
                        "target_gbs/(gpu_count*physical_mbs)"
                    )

        group_material = {
            "model_id": model_id,
            "model_geometry_sha256": sha256_json(geometry),
            "dataset_id": dataset_id,
            "training_mode": training_mode,
            "lora_rank": lora_rank,
            "target_gbs": target_gbs,
            "cutoff_len": cutoff_len,
            "packing": packing,
            "offload": offload,
            "dtype": dtype,
        }
        comparison_group = str(
            request.get("comparison_group")
            or (
                "auto-"
                + sha256_json(group_material)[:12]
            )
        )
        return {
            "input_index": input_index,
            "request_id": request_id,
            "comparison_group": comparison_group,
            "comparison_group_material": group_material,
            "model_id": model_id,
            "model": model,
            "model_geometry": geometry,
            "model_is_catalog_entry": (
                request.get("model") is None
                and model_id in self.models
            ),
            "dataset_id": dataset_id,
            "training_mode": training_mode,
            "hardware": hardware,
            "gpu_count": gpu_count,
            "physical_mbs": physical_mbs,
            "target_gbs": target_gbs,
            "cutoff_len": cutoff_len,
            "packing": packing,
            "offload": offload,
            "gradient_checkpointing": (
                gradient_checkpointing
            ),
            "zero_stage": zero_stage,
            "dtype": dtype,
            "kernel_path": kernel_path,
            "gradient_accumulation_steps": (
                gradient_accumulation
            ),
            "lora_rank": lora_rank,
            "planned_steps": request.get("planned_steps"),
            "total_effective_tokens": request.get(
                "total_effective_tokens"
            ),
        }

    @staticmethod
    def _model_family(model: Mapping[str, Any]) -> str:
        return str(
            model.get("family")
            or model.get("model_type")
            or (
                model.get("text_config") or {}
            ).get("model_type")
            or "unknown"
        ).lower()

    @staticmethod
    def _is_moe(model: Mapping[str, Any]) -> bool:
        values = [
            model.get("num_experts"),
            model.get("num_local_experts"),
            (model.get("text_config") or {}).get("num_experts"),
            (
                model.get("text_config") or {}
            ).get("num_local_experts"),
        ]
        return any(
            value is not None and int(value) > 1
            for value in values
        ) or "moe" in ThroughputPredictor._model_family(model)

    def _domain_assessment(
        self,
        normalized: Mapping[str, Any],
        basis: Mapping[str, Any],
    ) -> dict[str, Any]:
        reasons: list[dict[str, Any]] = []

        def add(
            code: str,
            severity: str,
            message: str,
        ) -> None:
            if code not in {item["code"] for item in reasons}:
                reasons.append(
                    {
                        "code": code,
                        "severity": severity,
                        "message": message,
                    }
                )

        hardware = normalized["hardware"]
        card_id = hardware.card_id
        fit_cards = set(self.frozen_model["fit_cards"])
        if card_id not in fit_cards:
            add(
                "unknown_card",
                "low",
                (
                    "No fitted card adapter exists; the predictor used "
                    "only shared physical parameters"
                ),
            )
        elif not hardware.builtin:
            add(
                "custom_physical_profile_for_known_card",
                "caution",
                (
                    "A known card adapter was combined with custom "
                    "hardware constants"
                ),
            )

        support = self.frozen_model["training_support"]
        by_card_support = (support.get("by_card") or {}).get(card_id)
        trained_models = set(support["model_ids"])
        if normalized["model_id"] not in trained_models:
            add(
                "unseen_model_id",
                "caution",
                (
                    "The model id was absent from fitting data; "
                    "prediction relies on structural transfer"
                ),
            )
        elif (
            by_card_support is not None
            and normalized["model_id"]
            not in set(by_card_support["model_ids"])
        ):
            add(
                "unseen_model_on_card",
                "caution",
                (
                    "This model scale was fitted on another card, not "
                    f"on {card_id}"
                ),
            )
        trained_datasets = set(support["dataset_ids"])
        if normalized["dataset_id"] not in trained_datasets:
            add(
                "unseen_dataset_id",
                "caution",
                (
                    "The dataset id was absent from fitting data; "
                    "prediction relies on its static token profile"
                ),
            )
        elif (
            by_card_support is not None
            and normalized["dataset_id"]
            not in set(by_card_support["dataset_ids"])
        ):
            add(
                "unseen_dataset_on_card",
                "caution",
                (
                    "This dataset distribution was fitted on another "
                    f"card, not on {card_id}"
                ),
            )

        family = self._model_family(normalized["model"])
        if family != "qwen3" or self._is_moe(normalized["model"]):
            add(
                "architecture_outside_training_evidence",
                "low",
                (
                    "Primary fitting evidence covers dense Qwen3 only"
                ),
            )
        if normalized["dtype"] != "bf16":
            add(
                "dtype_outside_training_evidence",
                "low",
                (
                    "The physical basis and fitting evidence are BF16"
                ),
            )
        if normalized["lora_rank"] != int(
            self.fixed_lora["rank"]
        ):
            add(
                "lora_rank_outside_training_evidence",
                "caution",
                (
                    "LoRA rank differs from the fitted rank "
                    f"{self.fixed_lora['rank']}"
                ),
            )

        base_parameters = float(
            normalized["model_geometry"]["base_parameters"]
        )
        parameter_support = support["base_parameters"]
        if not (
            float(parameter_support["min"])
            <= base_parameters
            <= float(parameter_support["max"])
        ):
            add(
                "model_scale_outside_training_support",
                "low",
                (
                    f"base_parameters={base_parameters:g} is outside "
                    f"[{parameter_support['min']:g}, "
                    f"{parameter_support['max']:g}]"
                ),
            )
        elif by_card_support is not None:
            card_parameter_support = by_card_support["base_parameters"]
            if not (
                float(card_parameter_support["min"])
                <= base_parameters
                <= float(card_parameter_support["max"])
            ):
                add(
                    "model_scale_outside_card_support",
                    "caution",
                    (
                        f"base_parameters={base_parameters:g} is within "
                        "global support but outside this card's fitted "
                        f"range [{card_parameter_support['min']:g}, "
                        f"{card_parameter_support['max']:g}]"
                    ),
                )

        discrete_checks = (
            (
                "gpu_count",
                normalized["gpu_count"],
                support["gpu_counts"],
            ),
            (
                "physical_mbs",
                normalized["physical_mbs"],
                support["physical_mbs"],
            ),
            (
                "target_gbs",
                normalized["target_gbs"],
                support["target_gbs"],
            ),
            (
                "cutoff_len",
                normalized["cutoff_len"],
                support["cutoff_lens"],
            ),
            (
                "zero_stage",
                normalized["zero_stage"],
                support["zero_stages"],
            ),
        )
        for name, value, supported in discrete_checks:
            if value not in supported:
                add(
                    f"unseen_{name}",
                    "caution",
                    (
                        f"{name}={value} was not observed; fitted values "
                        f"were {supported}"
                    ),
                )
        if by_card_support is not None:
            card_discrete_checks = (
                (
                    "gpu_count",
                    normalized["gpu_count"],
                    by_card_support["gpu_counts"],
                ),
                (
                    "physical_mbs",
                    normalized["physical_mbs"],
                    by_card_support["physical_mbs"],
                ),
                (
                    "target_gbs",
                    normalized["target_gbs"],
                    by_card_support["target_gbs"],
                ),
                (
                    "cutoff_len",
                    normalized["cutoff_len"],
                    by_card_support["cutoff_lens"],
                ),
                (
                    "zero_stage",
                    normalized["zero_stage"],
                    by_card_support["zero_stages"],
                ),
            )
            for name, value, supported in card_discrete_checks:
                if value not in supported:
                    add(
                        f"unseen_{name}_on_card",
                        "caution",
                        (
                            f"{name}={value} was fitted globally but not "
                            f"on {card_id}; card values were {supported}"
                        ),
                    )
        if normalized["packing"]:
            add(
                "packing_not_validated",
                "low",
                (
                    "The static packing contract is implemented, but "
                    "the frozen model has no packing=true training rows"
                ),
            )
        if normalized["offload"]:
            add(
                "unsupported_execution_mechanism",
                "low",
                (
                    "Optimizer/parameter offload has no V1 memory or throughput "
                    "coefficients; use the independent offload campaign."
                ),
            )

        outside_features = []
        outside_card_features = []
        feature_values = basis["feature_values"]
        for name in FEATURE_NAMES:
            value = float(feature_values[name])
            bounds = support["feature_ranges"][name]
            lower = float(bounds["min"])
            upper = float(bounds["max"])
            tolerance = max(
                1.0e-9,
                1.0e-8 * max(1.0, abs(lower), abs(upper)),
            )
            if value < lower - tolerance or value > upper + tolerance:
                outside_features.append(
                    {
                        "feature": name,
                        "value": value,
                        "training_min": lower,
                        "training_max": upper,
                    }
                )
            elif by_card_support is not None:
                card_bounds = by_card_support[
                    "feature_ranges"
                ][name]
                card_lower = float(card_bounds["min"])
                card_upper = float(card_bounds["max"])
                card_tolerance = max(
                    1.0e-9,
                    1.0e-8
                    * max(
                        1.0,
                        abs(card_lower),
                        abs(card_upper),
                    ),
                )
                if (
                    value < card_lower - card_tolerance
                    or value > card_upper + card_tolerance
                ):
                    outside_card_features.append(
                        {
                            "feature": name,
                            "value": value,
                            "card_training_min": card_lower,
                            "card_training_max": card_upper,
                        }
                    )
        if outside_features:
            add(
                "feature_outside_training_support",
                "low",
                (
                    f"{len(outside_features)} structured features are "
                    "outside their fitted min/max"
                ),
            )
        if outside_card_features:
            add(
                "feature_outside_card_support",
                "caution",
                (
                    f"{len(outside_card_features)} features were fitted "
                    "globally but are outside this card's min/max"
                ),
            )
        if self.binding_mismatches:
            add(
                "input_binding_mismatch",
                "low",
                (
                    "Inference inputs differ from frozen source bindings: "
                    + ", ".join(self.binding_mismatches)
                ),
            )

        severity_rank = {"caution": 1, "low": 2}
        maximum = max(
            (severity_rank[item["severity"]] for item in reasons),
            default=0,
        )
        label = (
            "supported"
            if maximum == 0
            else "caution"
            if maximum == 1
            else "low"
        )
        empirical_p90 = self._empirical_p90_reference(
            normalized,
            reasons,
        )
        return {
            "label": label,
            "inside_frozen_support": label == "supported",
            "known_card_adapter_used": card_id in fit_cards,
            "reasons": reasons,
            "outside_feature_support": outside_features,
            "outside_card_feature_support": (
                outside_card_features
            ),
            "empirical_p90_absolute_percentage_error_reference": (
                empirical_p90
            ),
            "reference_is_formal_prediction_interval": False,
        }

    def _empirical_p90_reference(
        self,
        normalized: Mapping[str, Any],
        reasons: Sequence[Mapping[str, Any]],
    ) -> float | None:
        codes = {str(reason["code"]) for reason in reasons}
        unsupported = {
            "architecture_outside_training_evidence",
            "dtype_outside_training_evidence",
            "model_scale_outside_training_support",
            "packing_not_validated",
            "unsupported_execution_mechanism",
            "feature_outside_training_support",
            "input_binding_mismatch",
            "unseen_gpu_count",
            "unseen_physical_mbs",
            "unseen_target_gbs",
            "unseen_cutoff_len",
            "unseen_zero_stage",
            "unseen_gpu_count_on_card",
            "unseen_physical_mbs_on_card",
            "unseen_target_gbs_on_card",
            "unseen_cutoff_len_on_card",
            "unseen_zero_stage_on_card",
            "feature_outside_card_support",
        }
        if codes & unsupported:
            return None

        references = self.report["uncertainty_reference"]
        values = []
        card_id = normalized["hardware"].card_id
        if card_id == "h800":
            values.append(
                references[
                    "known_h800_native_p90_absolute_percentage_error"
                ]
            )
        elif card_id == "rtx4090":
            values.append(
                references[
                    "known_rtx4090_scenario_p90_absolute_percentage_error"
                ]
            )
        else:
            values.extend(
                [
                    references[
                        "unseen_rtx4090_from_h800_p90_absolute_percentage_error"
                    ],
                    references[
                        "unseen_h800_from_rtx4090_p90_absolute_percentage_error"
                    ],
                ]
            )
        if (
            "unseen_model_id" in codes
            or "unseen_model_on_card" in codes
            or "model_scale_outside_card_support" in codes
        ):
            values.append(
                references[
                    "unseen_model_p90_absolute_percentage_error"
                ]
            )
        if (
            "unseen_dataset_id" in codes
            or "unseen_dataset_on_card" in codes
        ):
            values.append(
                references[
                    "unseen_dataset_p90_absolute_percentage_error"
                ]
            )
        return max(float(value) for value in values)

    def _predict_normalized(
        self,
        normalized: Mapping[str, Any],
        *,
        explain: bool,
    ) -> dict[str, Any]:
        performance: dict[str, Any] = {
            "physical_priors": (
                normalized["hardware"].physical_priors()
            )
        }
        if normalized["gradient_accumulation_steps"] is not None:
            performance["gradient_accumulation_steps"] = normalized[
                "gradient_accumulation_steps"
            ]
        record = {
            "scenario": {
                "model_id": normalized["model_id"],
                "dataset_id": normalized["dataset_id"],
                "train_type": normalized["training_mode"],
                "gpu_count": normalized["gpu_count"],
                "physical_mbs": normalized["physical_mbs"],
                "target_gbs": normalized["target_gbs"],
                "cutoff_len": normalized["cutoff_len"],
            },
            "selector": {
                "training_mode": normalized["training_mode"],
                "zero_stage": normalized["zero_stage"],
                "gradient_checkpointing": normalized[
                    "gradient_checkpointing"
                ],
                "packing": normalized["packing"],
                "offload": normalized["offload"],
                "kernel_path": normalized["kernel_path"],
                "dtype": normalized["dtype"],
            },
            "model_basis": normalized["model_geometry"],
            "performance": performance,
        }
        basis = _static_structured_basis(
            record,
            self.profiles,
            hardware_memory_bytes=(
                normalized["hardware"].memory_bytes
            ),
        )
        effective_tokens = float(
            basis["work_per_step"]["effective_tokens"]
        )
        candidate = {
            "card_id": normalized["hardware"].card_id,
            "structured_basis": basis,
            "structured_features": basis["features"],
            "physical_components": basis["components"],
            "static_log_work": math.log(effective_tokens),
        }
        predicted_log = _predict_log_throughput(
            candidate,
            self.frozen_model,
        )
        predicted_throughput = math.exp(predicted_log)
        predicted_step_seconds = (
            effective_tokens / predicted_throughput
        )
        work = {
            name: float(value)
            for name, value in basis["work_per_step"].items()
        }
        prediction = {
            "input_index": normalized["input_index"],
            "request_id": normalized["request_id"],
            "comparison_group": normalized["comparison_group"],
            "comparison_group_material": normalized[
                "comparison_group_material"
            ],
            "rank_within_group": None,
            "group_candidates": None,
            "predicted_effective_tokens_per_second": (
                predicted_throughput
            ),
            "predicted_log_effective_tokens_per_second": (
                predicted_log
            ),
            "predicted_step_seconds": predicted_step_seconds,
            "predicted_computed_tokens_per_second": (
                work["computed_tokens"] / predicted_step_seconds
            ),
            "predicted_logical_samples_per_second": (
                work["logical_samples"] / predicted_step_seconds
            ),
            "work_per_step": work,
            "configuration": {
                "model_id": normalized["model_id"],
                "dataset_id": normalized["dataset_id"],
                "hardware_id": normalized[
                    "hardware"
                ].hardware_id,
                "card_adapter_id": normalized[
                    "hardware"
                ].card_id,
                "training_mode": normalized["training_mode"],
                "gpu_count": normalized["gpu_count"],
                "physical_mbs": normalized["physical_mbs"],
                "target_gbs": normalized["target_gbs"],
                "cutoff_len": normalized["cutoff_len"],
                "gradient_accumulation_steps": basis[
                    "work_evidence"
                ]["gradient_accumulation_steps"],
                "gradient_checkpointing": normalized[
                    "gradient_checkpointing"
                ],
                "zero_stage": normalized["zero_stage"],
                "packing": normalized["packing"],
                "offload": normalized["offload"],
                "dtype": normalized["dtype"],
                "kernel_path": normalized["kernel_path"],
                "lora_rank": normalized["lora_rank"],
            },
            "confidence": self._domain_assessment(
                normalized,
                basis,
            ),
            "memory_safety_checked": False,
        }
        time_estimates = {}
        if normalized["planned_steps"] is not None:
            planned_steps = _required_positive_int(
                normalized["planned_steps"],
                "planned_steps",
            )
            time_estimates["planned_steps"] = planned_steps
            time_estimates[
                "seconds_for_planned_steps"
            ] = planned_steps * predicted_step_seconds
        if normalized["total_effective_tokens"] is not None:
            total_tokens = _required_positive_float(
                normalized["total_effective_tokens"],
                "total_effective_tokens",
            )
            time_estimates["total_effective_tokens"] = total_tokens
            time_estimates[
                "seconds_for_total_effective_tokens"
            ] = total_tokens / predicted_throughput
        if time_estimates:
            prediction["time_estimates"] = time_estimates
        if explain:
            prediction["explanation"] = {
                "model_geometry": dict(normalized["model_geometry"]),
                "hardware": normalized["hardware"].public_summary(),
                "physical_components_at_limits_seconds": (
                    basis[
                        "component_seconds_at_physical_limits"
                    ]
                ),
                "flops": basis["flops"],
                "traffic": basis["traffic"],
                "structured_feature_values": basis[
                    "feature_values"
                ],
            }
        return prediction

    def predict_many(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        explain: bool = False,
    ) -> dict[str, Any]:
        if isinstance(requests, (str, bytes)) or not isinstance(
            requests,
            Sequence,
        ):
            raise ValueError("requests must be a sequence of objects")
        normalized = []
        request_ids = set()
        for index, request in enumerate(requests):
            if not isinstance(request, Mapping):
                raise ValueError(
                    f"Candidate at index {index} is not an object"
                )
            item = self._normalized_request(
                request,
                input_index=index,
            )
            if item["request_id"] in request_ids:
                raise ValueError(
                    f"Duplicate request_id {item['request_id']!r}"
                )
            request_ids.add(item["request_id"])
            normalized.append(item)
        if not normalized:
            raise ValueError("At least one candidate is required")

        predictions = [
            self._predict_normalized(item, explain=explain)
            for item in normalized
        ]
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for prediction in predictions:
            by_group[prediction["comparison_group"]].append(
                prediction
            )
        ranking_groups = []
        for group_id, rows in sorted(by_group.items()):
            ordered = sorted(
                rows,
                key=lambda item: (
                    -float(
                        item[
                            "predicted_effective_tokens_per_second"
                        ]
                    ),
                    str(item["request_id"]),
                ),
            )
            for rank, row in enumerate(ordered, start=1):
                row["rank_within_group"] = rank
                row["group_candidates"] = len(ordered)
            ranking_groups.append(
                {
                    "comparison_group": group_id,
                    "comparison_group_material": ordered[0][
                        "comparison_group_material"
                    ],
                    "ranked_request_ids": [
                        row["request_id"] for row in ordered
                    ],
                    "predicted_effective_tokens_per_second": [
                        row[
                            "predicted_effective_tokens_per_second"
                        ]
                        for row in ordered
                    ],
                }
            )

        result: dict[str, Any] = {
            "schema": SCHEMA,
            "implementation_version": IMPLEMENTATION_VERSION,
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "target": "effective_tokens_per_second",
            "single_output_used_for_absolute_and_ranking": True,
            "requires_memory_safety_filter": True,
            "gpu_experiments_launched": False,
            "queues_mutated": False,
            "model_binding": {
                "path": str(self.model_artifact_path.resolve()),
                "file_sha256": sha256_file(
                    self.model_artifact_path
                ),
                "report_sha256": self.report["report_sha256"],
                "model_family": self.frozen_model[
                    "model_family"
                ],
            },
            "predictor_binding": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "ranking_groups": ranking_groups,
            "predictions": sorted(
                predictions,
                key=lambda item: int(item["input_index"]),
            ),
        }
        result["report_sha256"] = sha256_json(result)
        validate_prediction_report(result)
        return result

    def predict(
        self,
        request: Mapping[str, Any],
        *,
        explain: bool = False,
    ) -> dict[str, Any]:
        """Predict one configuration and return its prediction row."""

        return self.predict_many(
            [request],
            explain=explain,
        )["predictions"][0]

    def catalog(self) -> dict[str, Any]:
        trained_models = set(self.report["data"]["models"])
        return {
            "schema": "sft_throughput_predictor_catalog/v1",
            "models": [
                {
                    "model_id": model_id,
                    "family": self._model_family(model),
                    "actual_parameters": model.get(
                        "actual_parameters"
                    ),
                    "trained_by_frozen_model": (
                        model_id in trained_models
                    ),
                }
                for model_id, model in sorted(self.models.items())
            ],
            "datasets": [
                {
                    "dataset_id": dataset_id,
                    "trained_by_frozen_model": dataset_id
                    in set(self.report["data"]["datasets"]),
                }
                for dataset_id in sorted(self.profiles.rows)
            ],
            "hardware": [
                spec.public_summary()
                for spec in (
                    self.hardware["h800"],
                    self.hardware["rtx4090"],
                )
            ],
            "training_support": self.frozen_model[
                "training_support"
            ],
        }


def _read_input(path: str) -> Any:
    if path == "-":
        text = sys.stdin.read()
        source_name = "stdin"
    else:
        input_path = Path(path)
        text = input_path.read_text(encoding="utf-8")
        source_name = str(input_path)
    if not text.strip():
        raise ValueError(f"{source_name} is empty")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(
            text.splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{source_name}:{line_number} is invalid JSON"
                ) from error
            rows.append(row)
        return rows


def _requests_from_payload(
    payload: Any,
) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        if "candidates" in payload:
            candidates = payload["candidates"]
            if not isinstance(candidates, list):
                raise ValueError("candidates must be a JSON array")
            return candidates
        return [payload]
    raise ValueError(
        "Input must be one object, an array, or an object with candidates"
    )


def _write_output(path: str, payload: Mapping[str, Any]) -> None:
    if path == "-":
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    write_json(Path(path), payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        help=(
            "JSON/JSONL request path, or - for stdin. Optional with "
            "--show-catalog."
        ),
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Prediction JSON path, or - for stdout (default).",
    )
    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=DEFAULT_MODEL_ARTIFACT,
    )
    parser.add_argument(
        "--static-profile-artifact",
        type=Path,
        default=DEFAULT_STATIC_PROFILE_ARTIFACT,
    )
    parser.add_argument(
        "--dataset-profile-dir",
        type=Path,
        default=DEFAULT_DATASET_PROFILE_DIR,
    )
    parser.add_argument(
        "--additional-dataset-profile-dir",
        type=Path,
        help=(
            "Optional directory containing new dataset profiles. It is "
            "merged without changing frozen training-profile bindings."
        ),
    )
    parser.add_argument(
        "--model-inventory",
        type=Path,
        default=DEFAULT_MODEL_INVENTORY,
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Include physical components and all structured features.",
    )
    parser.add_argument(
        "--allow-binding-mismatch",
        action="store_true",
        help=(
            "Allow source/profile hashes that differ from frozen "
            "training inputs; predictions will be low confidence."
        ),
    )
    parser.add_argument(
        "--show-catalog",
        action="store_true",
        help="Print available models, datasets, hardware and support.",
    )
    args = parser.parse_args()

    try:
        predictor = ThroughputPredictor(
            model_artifact=args.model_artifact,
            static_profile_artifact=(
                args.static_profile_artifact
            ),
            dataset_profile_dir=args.dataset_profile_dir,
            additional_dataset_profile_dir=(
                args.additional_dataset_profile_dir
            ),
            model_inventory=args.model_inventory,
            strict_bindings=not args.allow_binding_mismatch,
        )
        if args.show_catalog:
            _write_output(args.output, predictor.catalog())
            return
        if not args.input:
            parser.error("--input is required without --show-catalog")
        requests = _requests_from_payload(
            _read_input(args.input)
        )
        result = predictor.predict_many(
            requests,
            explain=args.explain,
        )
        _write_output(args.output, result)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
