#!/usr/bin/env python3
"""H800 physical-shares memory gate plus frozen v4b ranking predictor.

The predictor is inference-only.  It accepts complete candidate
configurations for one or more fixed user scenarios, computes the H800
physical-shares memory center and operational P95, removes candidates that
fail memory or support-domain policy, and applies the frozen v4b two-head
ranker to the remaining candidates.  The recommendation policy first chooses
the minimum admitted GPU count and then selects the highest-ranked candidate
within that GPU count.  Cross-GPU throughput ranking remains diagnostic only.

The currently frozen artifacts are diagnostic and still require prospective
acceptance.  Every report is therefore explicitly marked ``shadow_only`` and
must not be used to launch or mutate training jobs automatically.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from common import ROOT, read_json, sha256_file, sha256_json, write_json
from cross_card_scaling import (
    DEFAULT_GPU_ORDER,
    DEFAULT_MINIMUM_THROUGHPUT_RATIO,
    POLICY_ID as SCALE_OUT_POLICY_ID,
    evaluate_scale_out_sequence,
)
from h800_challenger_modeling import _predict_memory_upper
from h800_theory_basis import memory_basis
from joint_throughput_modeling import (
    _predict_two_head_entries,
    _record_features,
    _record_log_analytic_anchor,
)
from structured_throughput_modeling import _static_structured_basis
from throughput_predictor import DEFAULT_MODEL_INVENTORY, ThroughputPredictor
from memory_anchor_registry import (
    evaluate_anchor_admission,
    load_registry,
)


SCHEMA = "sft_h800_physical_shares_v4b_prediction/v3"
IMPLEMENTATION_VERSION = "sft_h800_physical_shares_v4b_predictor/2026-07-30.v3"
SELECTION_POLICY = "minimum_admitted_gpu_count_then_v4b"
DEFAULT_MEMORY_ARTIFACT = ROOT / "artifacts" / "h800_challenger_modeling.json"
DEFAULT_THROUGHPUT_ARTIFACT = ROOT / "artifacts" / "joint_throughput_modeling.json"
DEFAULT_MEMORY_ANCHOR_REGISTRY = (
    ROOT / "artifacts" / "h800_memory_anchor_registry_v1.json"
)
MEMORY_MODEL_SCHEMA = "sft_h800_challenger_modeling/v1"
THROUGHPUT_MODEL_SCHEMA = "sft_joint_throughput_modeling/v1"
H800_RUNTIME_COHORT = "h800_static_inference_fa3_v1"
H800_RUNTIME_MECHANISM_COMPONENT_SHA256 = (
    "9723bdb98e55814c6a296cd9600161e195bc91e007736ba17b2e619f51688624"
)
H800_SUPPORTED_MODEL_IDS = {
    "qwen3_1p7b",
    "qwen3_4b",
    "qwen3_8b",
    "qwen3_14b",
}
SUPPORTED_DATASET_CATEGORIES = {
    "short",
    "multiturn",
    "longtail",
    "longcontext",
}
SUPPORTED_GPU_COUNTS = {1, 2, 4}
SUPPORTED_TARGET_GBS = {32, 64, 128}
SUPPORTED_CUTOFF_LENS = {512, 2048, 4096, 8192, 32768}
MIN_TRANSFER_PARAMETERS = 1_500_000_000
MAX_TRANSFER_PARAMETERS = 33_000_000_000
GIB = float(1024**3)


def _validate_checksum(
    report: Mapping[str, Any],
    *,
    artifact_name: str,
) -> None:
    unsigned = dict(report)
    expected = unsigned.pop("report_sha256", None)
    if not isinstance(expected, str) or expected != sha256_json(unsigned):
        raise ValueError(f"{artifact_name} checksum mismatch")


def _validate_memory_model(report: Mapping[str, Any]) -> None:
    if report.get("schema") != MEMORY_MODEL_SCHEMA:
        raise ValueError("H800 memory artifact schema mismatch")
    _validate_checksum(report, artifact_name="H800 memory artifact")
    if (
        report.get("gpu_family") != "H800"
        or report.get("gpu_experiments_launched") is not False
        or report.get("queues_mutated") is not False
    ):
        raise ValueError("H800 memory artifact safety contract drifted")
    frozen = (report.get("memory") or {}).get("frozen_model") or {}
    center = frozen.get("reserved_center") or {}
    tail = frozen.get("tail") or {}
    if (
        center.get("available") is not True
        or center.get("feature_set") != "physical_shares"
        or center.get("model_family") != "analytic_reference_log_residual_ridge"
        or len(center.get("feature_names") or []) != 28
        or tail.get("available") is not True
    ):
        raise ValueError("H800 physical-shares model contract drifted")


def _validate_throughput_model(
    report: Mapping[str, Any],
    *,
    memory_report: Mapping[str, Any],
    memory_artifact: Path,
) -> None:
    if report.get("schema") != THROUGHPUT_MODEL_SCHEMA:
        raise ValueError("H800 throughput artifact schema mismatch")
    _validate_checksum(report, artifact_name="H800 throughput artifact")
    if (
        report.get("gpu_experiments_launched") is not False
        or report.get("queues_mutated") is not False
    ):
        raise ValueError("H800 throughput artifact safety contract drifted")
    frozen = (report.get("h800") or {}).get("frozen_two_head_challenger") or {}
    absolute = frozen.get("absolute_head") or {}
    rank = frozen.get("rank_head") or {}
    if (
        frozen.get("available") is not True
        or frozen.get("model_family") != "set_aware_two_head_log_throughput_model"
        or frozen.get("absolute_deviation_blend") != 0.25
        or absolute.get("feature_dimension") != 70
        or rank.get("feature_dimension") != 70
    ):
        raise ValueError("H800 v4b model contract drifted")
    memory_binding = (report.get("source_bindings") or {}).get(
        "h800_hybrid_report"
    ) or {}
    if memory_binding.get("report_sha256") != memory_report.get(
        "report_sha256"
    ) or memory_binding.get("sha256") != sha256_file(memory_artifact):
        raise ValueError(
            "H800 v4b artifact is not bound to the selected memory artifact"
        )


def _zero_name(stage: int) -> str:
    return "none" if stage == 0 else f"zero{stage}"


def _gib(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) / GIB


def _dataset_category(
    request: Mapping[str, Any],
    dataset_id: str,
) -> str:
    explicit = request.get("dataset_category")
    if explicit is not None:
        category = str(explicit).strip().lower()
        if category not in SUPPORTED_DATASET_CATEGORIES:
            raise ValueError(
                "dataset_category must be short, multiturn, longtail or longcontext"
            )
        return category
    lowered = dataset_id.lower()
    for category in sorted(SUPPORTED_DATASET_CATEGORIES):
        if lowered.startswith(category):
            return category
    return "other"


def _feature_record(
    record: Mapping[str, Any],
    dataset_category: str,
) -> dict[str, Any]:
    """Return a shallow record copy with the frozen dataset-category prefix."""

    copied = dict(record)
    scenario = dict(record.get("scenario") or {})
    original_id = str(scenario.get("dataset_id") or "")
    scenario["dataset_id"] = f"{dataset_category}__{original_id}"
    copied["scenario"] = scenario
    return copied


def _model_family(model: Mapping[str, Any]) -> str:
    return (
        str(
            model.get("family")
            or model.get("model_type")
            or (model.get("text_config") or {}).get("model_type")
            or "unknown"
        )
        .strip()
        .lower()
    )


def _is_moe(model: Mapping[str, Any]) -> bool:
    values = [
        model.get("num_experts"),
        model.get("num_local_experts"),
        (model.get("text_config") or {}).get("num_experts"),
        (model.get("text_config") or {}).get("num_local_experts"),
    ]
    return any(
        value is not None and int(value) > 1 for value in values
    ) or "moe" in _model_family(model)


def _is_vision_language(model: Mapping[str, Any]) -> bool:
    family = _model_family(model)
    architectures = " ".join(
        str(value).lower() for value in (model.get("architectures") or [])
    )
    return bool(
        model.get("vision_config")
        or (model.get("text_config") and model.get("vision_config"))
        or "vision" in family
        or "vl" in family
        or "vision" in architectures
        or "vl" in architectures
    )


def validate_prediction_report(report: Mapping[str, Any]) -> None:
    """Validate checksums and the fail-closed ranking contract."""

    if report.get("schema") != SCHEMA:
        raise ValueError("H800 prediction schema mismatch")
    unsigned = dict(report)
    expected = unsigned.pop("report_sha256", None)
    if expected != sha256_json(unsigned):
        raise ValueError("H800 prediction report checksum mismatch")
    release = report.get("release") or {}
    if (
        report.get("hardware_id") != "h800"
        or report.get("gpu_experiments_launched") is not False
        or report.get("queues_mutated") is not False
        or release.get("mode") != "shadow_only"
        or release.get("automatic_execution_allowed") is not False
    ):
        raise ValueError("H800 prediction safety contract drifted")
    runtime_binding = report.get("runtime_binding") or {}
    if (
        runtime_binding.get("runtime_cohort_id") != H800_RUNTIME_COHORT
        or runtime_binding.get("gpu_family") != "H800"
        or runtime_binding.get("runtime_mechanism_component_sha256")
        != H800_RUNTIME_MECHANISM_COMPONENT_SHA256
    ):
        raise ValueError("H800 prediction runtime binding drifted")
    selection_policy = report.get("selection_policy") or {}
    if (
        selection_policy.get("id") != SELECTION_POLICY
        or selection_policy.get("scale_out_enabled") is not False
    ):
        raise ValueError("H800 selection policy contract drifted")
    anchor_binding = (report.get("model_artifacts") or {}).get("memory_anchors") or {}
    if not all(
        isinstance(anchor_binding.get(field), str) and bool(anchor_binding.get(field))
        for field in ("path", "sha256", "report_sha256", "registry_id")
    ):
        raise ValueError("H800 memory anchor binding is missing")

    predictions = report.get("predictions")
    groups = report.get("ranking_groups")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("H800 prediction report has no candidate rows")
    if not isinstance(groups, list) or not groups:
        raise ValueError("H800 prediction report has no ranking groups")

    request_ids = [str(row.get("request_id")) for row in predictions]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("H800 prediction request ids are not unique")

    rows_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        group_id = str(row.get("comparison_group"))
        rows_by_group[group_id].append(row)
        if not isinstance(row.get("scenario_material"), Mapping):
            raise ValueError("H800 prediction row lacks scenario_material binding")
        if row.get("runtime_mechanism_component_sha256") != runtime_binding.get(
            "runtime_mechanism_component_sha256"
        ):
            raise ValueError("H800 prediction row runtime mechanism binding drifted")
        memory = row.get("memory") or {}
        support = row.get("support") or {}
        throughput = row.get("throughput") or {}
        if support.get("label") not in {
            "supported",
            "caution",
            "unsupported",
        }:
            raise ValueError("H800 prediction support label drifted")
        if support.get("support_tier") not in {
            "native",
            "caution",
            "generic_structural",
            "conservative_fallback",
            "unsupported",
        }:
            raise ValueError("H800 prediction support tier drifted")
        if support.get("automatic_execution_allowed") is not False:
            raise ValueError("H800 support fallback must remain non-executable")
        safe_limit = float(memory.get("safe_limit_bytes") or 0.0)
        if safe_limit <= 0.0 or not math.isfinite(safe_limit):
            raise ValueError("H800 prediction has invalid safe limit")
        if memory.get("prediction_available") is True:
            for field in (
                "analytic_reference_bytes",
                "reserved_center_bytes",
                "operational_p95_reserved_bytes",
                "admission_upper_reserved_bytes",
            ):
                value = float(memory.get(field) or 0.0)
                if value <= 0.0 or not math.isfinite(value):
                    raise ValueError(f"H800 prediction has invalid {field}")
        expected_base_admission = bool(
            memory.get("prediction_available") is True
            and float(memory["operational_p95_reserved_bytes"]) <= safe_limit
        )
        if (
            memory.get("base_physical_model_admitted") is True
        ) != expected_base_admission:
            raise ValueError("H800 base physical admission contract drifted")
        anchor = memory.get("historical_anchor") or {}
        anchor_override = memory.get("anchor_override_applied") is True
        if anchor_override:
            if (
                expected_base_admission
                or anchor.get("matched") is not True
                or anchor.get("override_allowed") is not True
                or float(anchor.get("local_guarded_upper_bytes") or math.inf)
                > safe_limit
                or support.get("label") == "unsupported"
            ):
                raise ValueError("H800 anchor override contract drifted")
            if (
                float(memory["admission_upper_reserved_bytes"])
                != float(anchor["local_guarded_upper_bytes"])
                or memory.get("admission_source") != "versioned_historical_anchor"
            ):
                raise ValueError("H800 anchor admission upper drifted")
        elif expected_base_admission:
            if (
                float(memory["admission_upper_reserved_bytes"])
                != float(memory["operational_p95_reserved_bytes"])
                or memory.get("admission_source") != "physical_model_operational_p95"
            ):
                raise ValueError("H800 base admission upper drifted")
        expected_physical_admission = bool(expected_base_admission or anchor_override)
        if (
            memory.get("physical_model_admitted") is True
        ) != expected_physical_admission:
            raise ValueError("H800 physical admission contract drifted")
        admitted = memory.get("admitted") is True
        expected_policy_admission = bool(
            expected_physical_admission and support.get("label") != "unsupported"
        )
        if admitted != expected_policy_admission:
            raise ValueError("H800 support/memory admission contract drifted")
        if admitted != (throughput.get("prediction_available") is True):
            raise ValueError(
                "H800 throughput must exist exactly for admitted candidates"
            )
        if throughput.get("prediction_available") is True:
            score = float(throughput.get("ranking_score_log"))
            proxy = float(throughput.get("throughput_proxy_tokens_per_second"))
            if (
                not math.isfinite(score)
                or not math.isfinite(proxy)
                or proxy <= 0.0
                or throughput.get("absolute_scale_trusted") is not False
            ):
                raise ValueError("H800 throughput proxy contract drifted")

    declared_groups = set()
    for group in groups:
        group_id = str(group.get("comparison_group"))
        if group_id in declared_groups:
            raise ValueError("H800 prediction has duplicate ranking groups")
        declared_groups.add(group_id)
        ranked_ids = list(group.get("ranked_request_ids") or [])
        requested_rows = rows_by_group.get(group_id, [])
        ranked_rows = [
            row
            for row in requested_rows
            if (row.get("throughput") or {}).get("prediction_available") is True
        ]
        ranked_rows.sort(
            key=lambda row: (
                -float(row["throughput"]["throughput_proxy_tokens_per_second"]),
                int(row["input_index"]),
            )
        )
        expected_ids = [str(row["request_id"]) for row in ranked_rows]
        if ranked_ids != expected_ids:
            raise ValueError("H800 ranking group order drifted")
        for expected_rank, row in enumerate(ranked_rows, start=1):
            if row.get("rank_within_admitted_group") != expected_rank:
                raise ValueError("H800 candidate rank drifted")
        ranked_by_gpu: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in ranked_rows:
            ranked_by_gpu[int(row["configuration"]["gpu_count"])].append(row)
        for gpu_rows in ranked_by_gpu.values():
            for expected_rank, row in enumerate(gpu_rows, start=1):
                if row.get("rank_within_gpu_count") != expected_rank:
                    raise ValueError("H800 per-GPU-count rank drifted")
        if any(
            row.get("rank_within_admitted_group") is not None
            or row.get("rank_within_gpu_count") is not None
            for row in requested_rows
            if (row.get("throughput") or {}).get("prediction_available") is not True
        ):
            raise ValueError("H800 rejected candidate received a rank")
        minimum_gpu_count = min(ranked_by_gpu, default=None)
        selected = (
            str(ranked_by_gpu[minimum_gpu_count][0]["request_id"])
            if minimum_gpu_count is not None
            else None
        )
        throughput_top = expected_ids[0] if expected_ids else None
        if (
            group.get("selection_policy") != SELECTION_POLICY
            or group.get("minimum_admitted_gpu_count") != minimum_gpu_count
            or group.get("throughput_top_request_id") != throughput_top
            or group.get("selected_request_id") != selected
        ):
            raise ValueError("H800 selected request id drifted")
        physical_count = sum(
            (row.get("memory") or {}).get("physical_model_admitted") is True
            for row in requested_rows
        )
        memory_rejections = sum(
            (row.get("memory") or {}).get("rejection_reason")
            == "memory_upper_exceeds_safe_limit"
            for row in requested_rows
        )
        support_rejections = sum(
            (row.get("memory") or {}).get("rejection_reason")
            == "outside_supported_domain"
            for row in requested_rows
        )
        anchor_overrides = sum(
            (row.get("memory") or {}).get("anchor_override_applied") is True
            for row in requested_rows
        )
        expected_status = (
            "ranked_shadow_only" if ranked_rows else "no_admitted_candidate"
        )
        if (
            group.get("requested_candidates") != len(requested_rows)
            or group.get("physical_model_admitted_candidates") != physical_count
            or group.get("policy_admitted_candidates") != len(ranked_rows)
            or group.get("rejected_by_memory") != memory_rejections
            or group.get("rejected_by_support_domain") != support_rejections
            or group.get("admitted_by_historical_anchor") != anchor_overrides
            or group.get("status") != expected_status
            or group.get("automatic_execution_allowed") is not False
        ):
            raise ValueError("H800 ranking group summary drifted")
    if declared_groups != set(rows_by_group):
        raise ValueError("H800 prediction contains undeclared groups")


class H800PhysicalV4BPredictor:
    """Filter H800 candidates with physical-shares and rank with v4b."""

    def __init__(
        self,
        *,
        memory_artifact: Path = DEFAULT_MEMORY_ARTIFACT,
        throughput_artifact: Path = DEFAULT_THROUGHPUT_ARTIFACT,
        memory_anchor_registry: Path = DEFAULT_MEMORY_ANCHOR_REGISTRY,
        model_inventory: Path = DEFAULT_MODEL_INVENTORY,
        strict_model_inventory_binding: bool = True,
        additional_dataset_profile_dir: Path | None = None,
    ) -> None:
        self.memory_artifact_path = Path(memory_artifact)
        self.throughput_artifact_path = Path(throughput_artifact)
        self.memory_anchor_registry_path = Path(memory_anchor_registry)
        self.memory_report = read_json(self.memory_artifact_path)
        self.throughput_report = read_json(self.throughput_artifact_path)
        self.memory_anchor_registry = load_registry(self.memory_anchor_registry_path)
        _validate_memory_model(self.memory_report)
        _validate_throughput_model(
            self.throughput_report,
            memory_report=self.memory_report,
            memory_artifact=self.memory_artifact_path,
        )
        frozen_memory = self.memory_report["memory"]["frozen_model"]
        self.memory_center = frozen_memory["reserved_center"]
        self.memory_tail = frozen_memory["tail"]
        self.throughput_model = self.throughput_report["h800"][
            "frozen_two_head_challenger"
        ]
        self.supported_mbs = set(self.memory_report["protocol"]["supported_mbs"])
        self.model_inventory_path = Path(model_inventory)
        self.base = ThroughputPredictor(
            model_inventory=self.model_inventory_path,
            strict_bindings=strict_model_inventory_binding,
            additional_dataset_profile_dir=additional_dataset_profile_dir,
        )
        self.additional_dataset_profile_dir = (
            Path(additional_dataset_profile_dir)
            if additional_dataset_profile_dir is not None
            else None
        )

    def _profile_binding(self, dataset_id: str) -> dict[str, Any]:
        candidates = [
            self.base.dataset_profile_dir / f"{dataset_id}.qwen3_nothink.jsonl"
        ]
        if self.additional_dataset_profile_dir is not None:
            candidates.append(
                self.additional_dataset_profile_dir
                / f"{dataset_id}.qwen3_nothink.jsonl"
            )
        for path in candidates:
            if path.is_file():
                return {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "origin": (
                        "additional"
                        if (
                            self.additional_dataset_profile_dir is not None
                            and path.parent.resolve()
                            == self.additional_dataset_profile_dir.resolve()
                        )
                        else "frozen_catalog"
                    ),
                }
        raise ValueError(f"No profile binding found for {dataset_id!r}")

    def _support_assessment(
        self,
        normalized: Mapping[str, Any],
        *,
        dataset_category: str,
        profile_binding: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        reasons: list[dict[str, str]] = []

        def add(code: str, severity: str, message: str) -> None:
            if code not in {item["code"] for item in reasons}:
                reasons.append(
                    {
                        "code": code,
                        "severity": severity,
                        "message": message,
                    }
                )

        model = normalized["model"]
        family = _model_family(model)
        parameters = int(normalized["model_geometry"]["base_parameters"])
        if _is_moe(model):
            add(
                "moe_outside_supported_domain",
                "unsupported",
                "The frozen dense-Qwen models do not cover MoE routing.",
            )
        if _is_vision_language(model):
            add(
                "vision_language_outside_supported_domain",
                "unsupported",
                "The H800 physical-shares gate is not calibrated for VL.",
            )
        if "qwen" not in family:
            add(
                "architecture_outside_supported_domain",
                "unsupported",
                "The current H800 evidence covers dense Qwen models.",
            )
        if normalized["dtype"] != "bf16":
            add(
                "dtype_outside_supported_domain",
                "unsupported",
                "The frozen memory and throughput evidence is BF16.",
            )
        if normalized["packing"]:
            add(
                "packing_outside_supported_domain",
                "unsupported",
                "Packing requires a separate validated admission track.",
            )
        if normalized["offload"]:
            add(
                "unsupported_execution_mechanism",
                "unsupported",
                (
                    "Optimizer/parameter offload has no V1 memory or throughput "
                    "coefficients and requires an independent campaign."
                ),
            )
        if normalized["gpu_count"] not in SUPPORTED_GPU_COUNTS:
            add(
                "gpu_count_outside_supported_domain",
                "unsupported",
                "The first H800 product domain is limited to 1/2/4 GPUs.",
            )
        if normalized["physical_mbs"] not in self.supported_mbs:
            add(
                "mbs_outside_supported_domain",
                "unsupported",
                f"Supported MBS values are {sorted(self.supported_mbs)}.",
            )
        if not (MIN_TRANSFER_PARAMETERS <= parameters <= MAX_TRANSFER_PARAMETERS):
            add(
                "model_scale_outside_transfer_domain",
                "unsupported",
                "Dense-Qwen transfer evidence spans about 1.5B to 33B.",
            )
        if normalized["model_id"] not in H800_SUPPORTED_MODEL_IDS:
            add(
                "model_requires_transfer_evidence",
                "caution",
                "This model id relies on structural transfer evidence.",
            )
        if normalized["zero_stage"] == 1:
            add(
                "zero1_limited_evidence",
                "caution",
                "The primary H800 recommendation space uses ZeRO-2/3.",
            )
        if normalized["target_gbs"] not in SUPPORTED_TARGET_GBS:
            add(
                "target_gbs_unseen",
                "caution",
                "Target GBS is outside the primary 32/64/128 evidence.",
            )
        if normalized["cutoff_len"] not in SUPPORTED_CUTOFF_LENS:
            add(
                "cutoff_unseen",
                "caution",
                "Cutoff is outside the primary frozen evaluation grid.",
            )
        if dataset_category == "other":
            add(
                "dataset_category_unresolved",
                "caution",
                "Provide a frozen dataset_category for a custom profile.",
            )
        if profile_binding.get("origin") == "additional":
            add(
                "new_dataset_profile",
                "caution",
                "The dataset profile was not part of frozen model fitting.",
            )
            if not request.get("profile_tokenizer_id"):
                add(
                    "profile_tokenizer_binding_missing",
                    "caution",
                    "Custom profiles should bind the tokenizer identity.",
                )
            if not request.get("profile_template_id"):
                add(
                    "profile_template_binding_missing",
                    "caution",
                    "Custom profiles should bind the template identity.",
                )
        if "fa3" not in normalized["kernel_path"].lower():
            add(
                "kernel_outside_supported_domain",
                "unsupported",
                "The H800 frozen evidence uses the FA3 kernel path.",
            )
        if normalized["lora_rank"] != 32:
            add(
                "lora_rank_unseen",
                "caution",
                "The primary H800 LoRA evidence uses rank 32.",
            )

        maximum = max(
            ({"caution": 1, "unsupported": 2}[item["severity"]] for item in reasons),
            default=0,
        )
        label = (
            "supported"
            if maximum == 0
            else "caution"
            if maximum == 1
            else "unsupported"
        )
        reason_codes = {item["code"] for item in reasons}
        visual_path_observed = request.get("visual_path_observed") is True
        if "vision_language_outside_supported_domain" in reason_codes:
            support_tier = "conservative_fallback"
            required_experiment_family = "vl_image_memory_throughput"
            fallback_policy = {
                "max_physical_mbs": 1,
                "gradient_checkpointing": True,
                "packing": False,
                "offload": False,
                "automatic_execution_allowed": False,
                "reason": "VL visual path and processor-bound evidence are required first",
            }
        elif "unsupported_execution_mechanism" in reason_codes:
            support_tier = "unsupported"
            required_experiment_family = "offload_mechanism_calibration"
            fallback_policy = {
                "automatic_execution_allowed": False,
                "reason": "offload requires a separately calibrated mechanism track",
            }
        elif label == "unsupported":
            support_tier = "generic_structural"
            required_experiment_family = "dense_structural_transfer"
            fallback_policy = {
                "automatic_execution_allowed": False,
                "reason": "outside native evidence; collect a structural transfer holdout",
            }
        elif label == "caution":
            support_tier = "caution"
            required_experiment_family = "targeted_domain_holdout"
            fallback_policy = {
                "automatic_execution_allowed": False,
                "reason": "cautionary inputs require prospective validation",
            }
        else:
            support_tier = "native"
            required_experiment_family = None
            fallback_policy = {
                "automatic_execution_allowed": False,
                "reason": "native shadow prediction; publication gate remains separate",
            }
        return {
            "label": label,
            "support_tier": support_tier,
            "inside_initial_product_domain": label == "supported",
            "eligible_for_shadow_ranking": label != "unsupported",
            "automatic_execution_allowed": False,
            "required_experiment_family": required_experiment_family,
            "visual_path_observed": visual_path_observed,
            "fallback_policy": fallback_policy,
            "reasons": reasons,
        }

    @staticmethod
    def _validate_resource_geometry(
        normalized: Mapping[str, Any],
    ) -> None:
        gpu_count = int(normalized["gpu_count"])
        zero_stage = int(normalized["zero_stage"])
        if gpu_count == 1 and zero_stage != 0:
            raise ValueError("Single-GPU H800 candidates must use zero_stage=0")
        if gpu_count > 1 and zero_stage not in {1, 2, 3}:
            raise ValueError("Multi-GPU H800 candidates must use ZeRO-1/2/3")

    def _record(
        self,
        request: Mapping[str, Any],
        *,
        input_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        declared_hardware = str(
            request.get("hardware_id") or request.get("gpu_type") or "h800"
        ).lower()
        if "h800" not in declared_hardware or request.get("hardware"):
            raise ValueError("This predictor accepts builtin H800 only")
        normalized = self.base._normalized_request(
            {
                **dict(request),
                "hardware_id": "h800",
            },
            input_index=input_index,
        )
        if normalized["hardware"].card_id != "h800":
            raise ValueError("This predictor accepts H800 only")
        self._validate_resource_geometry(normalized)
        dataset_category = _dataset_category(
            request,
            normalized["dataset_id"],
        )
        profile_binding = self._profile_binding(normalized["dataset_id"])
        profile_binding = {
            **profile_binding,
            "tokenizer_id": (
                str(request["profile_tokenizer_id"])
                if request.get("profile_tokenizer_id")
                else None
            ),
            "template_id": (
                str(request["profile_template_id"])
                if request.get("profile_template_id")
                else None
            ),
        }
        support = self._support_assessment(
            normalized,
            dataset_category=dataset_category,
            profile_binding=profile_binding,
            request=request,
        )
        scenario_material = {
            "model_id": normalized["model_id"],
            "model_geometry_sha256": sha256_json(normalized["model_geometry"]),
            "dataset_id": normalized["dataset_id"],
            "dataset_category": dataset_category,
            "dataset_profile_sha256": profile_binding["sha256"],
            "profile_tokenizer_id": profile_binding["tokenizer_id"],
            "profile_template_id": profile_binding["template_id"],
            "training_mode": normalized["training_mode"],
            "lora_rank": normalized["lora_rank"],
            "target_gbs": normalized["target_gbs"],
            "cutoff_len": normalized["cutoff_len"],
            "packing": normalized["packing"],
            "offload": normalized["offload"],
            "dtype": normalized["dtype"],
        }
        scenario_id = str(
            request.get("comparison_group")
            or "auto-" + sha256_json(scenario_material)[:12]
        )
        scenario = {
            "model_id": normalized["model_id"],
            "train_type": normalized["training_mode"],
            "dataset_id": normalized["dataset_id"],
            "dataset_category": dataset_category,
            "target_gbs": normalized["target_gbs"],
            "gpu_count": normalized["gpu_count"],
            "physical_mbs": normalized["physical_mbs"],
            "cutoff_len": normalized["cutoff_len"],
        }
        selector = {
            "runtime_cohort_id": H800_RUNTIME_COHORT,
            "dtype": normalized["dtype"],
            "kernel_path": normalized["kernel_path"],
            "training_mode": normalized["training_mode"],
            "zero_stage": normalized["zero_stage"],
            "gradient_checkpointing": normalized["gradient_checkpointing"],
            "packing": normalized["packing"],
            "offload": normalized["offload"],
        }
        job = {
            "model_id": normalized["model_id"],
            "model_parameters": normalized["model_geometry"]["base_parameters"],
            "train_type": normalized["training_mode"],
            "dataset_id": normalized["dataset_id"],
            "target_gbs": normalized["target_gbs"],
            "gpu_count": normalized["gpu_count"],
            "mbs": normalized["physical_mbs"],
            "cutoff_len": normalized["cutoff_len"],
            "zero": _zero_name(normalized["zero_stage"]),
            "gc": normalized["gradient_checkpointing"],
            "packing": normalized["packing"],
        }
        memory = memory_basis(
            job,
            normalized["model_geometry"],
            int(normalized["hardware"].memory_bytes),
        )
        performance: dict[str, Any] = {
            "physical_priors": normalized["hardware"].physical_priors(),
        }
        if normalized["gradient_accumulation_steps"] is not None:
            performance["gradient_accumulation_steps"] = normalized[
                "gradient_accumulation_steps"
            ]
        record = {
            "schema": "sft_h800_static_inference_record/v1",
            "observation_id": normalized["request_id"],
            "job_id": normalized["request_id"],
            "outcome": "unknown",
            "scenario": scenario,
            "scenario_id": scenario_id,
            "scenario_material": scenario_material,
            "selector": selector,
            "runtime": {
                "runtime_cohort_id": H800_RUNTIME_COHORT,
                "gpu_family": "H800",
                "runtime_mechanism_component_sha256": (
                    H800_RUNTIME_MECHANISM_COMPONENT_SHA256
                ),
            },
            "model_basis": normalized["model_geometry"],
            "memory": memory,
            "performance": performance,
            "dataset_profile_binding": profile_binding,
            "support": support,
        }
        basis = _static_structured_basis(
            record,
            self.base.profiles,
            hardware_memory_bytes=normalized["hardware"].memory_bytes,
        )
        traffic = basis["traffic"]
        limits = basis["component_seconds_at_physical_limits"]
        performance.update(
            {
                "gradient_accumulation_steps": basis["work_evidence"][
                    "gradient_accumulation_steps"
                ],
                "work_per_step": basis["work_per_step"],
                "work_is_per_optimizer_step": True,
                "flops_per_step": basis["flops"],
                "traffic_bytes_per_rank_step": {
                    "kernel_total": traffic["kernel"],
                    "optimizer": traffic["optimizer"],
                },
                "communication": {
                    "payload_bytes_per_rank_step": traffic["communication"],
                    "collective_count": traffic["collective_count"],
                },
                "ideal_seconds": {
                    "compute_at_dense_peak": limits["compute"],
                    "kernel_hbm_at_physical_peak": limits["kernel_hbm"],
                    "optimizer_hbm_at_physical_peak": limits["optimizer_hbm"],
                    "collective_payload_at_link_peak": (
                        float(traffic["communication"])
                        / normalized["hardware"].intra_node_bandwidth_bytes_per_second
                    ),
                },
            }
        )
        normalized = {
            **normalized,
            "comparison_group": scenario_id,
            "dataset_category": dataset_category,
            "dataset_profile_binding": profile_binding,
            "scenario_material": scenario_material,
        }
        return record, normalized, support

    def _memory_result(
        self,
        record: Mapping[str, Any],
        support: Mapping[str, Any],
    ) -> dict[str, Any]:
        prediction = _predict_memory_upper(
            record,
            self.memory_center,
            self.memory_tail,
        )
        available = prediction.get("available") is True
        safe_limit = float(record["memory"]["safe_limit_bytes"])
        upper = prediction.get("operational_p95_reserved_bytes")
        base_physical_model_admitted = bool(available and float(upper) <= safe_limit)
        anchor = evaluate_anchor_admission(
            record,
            base_prediction=prediction,
            support=support,
            registry=self.memory_anchor_registry,
        )
        anchor_override_applied = bool(
            available
            and not base_physical_model_admitted
            and anchor.get("override_allowed") is True
        )
        physical_model_admitted = bool(
            base_physical_model_admitted or anchor_override_applied
        )
        policy_admitted = bool(
            physical_model_admitted and support.get("label") != "unsupported"
        )
        admission_upper = (
            float(upper)
            if base_physical_model_admitted
            else (
                float(anchor["local_guarded_upper_bytes"])
                if anchor_override_applied
                else (float(upper) if upper is not None else None)
            )
        )
        if not available:
            rejection_reason = "memory_prediction_unavailable"
        elif not physical_model_admitted:
            rejection_reason = "memory_upper_exceeds_safe_limit"
        elif support.get("label") == "unsupported":
            rejection_reason = "outside_supported_domain"
        else:
            rejection_reason = None
        center = prediction.get("reserved_center_bytes")
        reference = record["memory"]["analytic_reference_bytes"]
        headroom = safe_limit - admission_upper if admission_upper is not None else None
        return {
            "prediction_available": available,
            "analytic_reference_bytes": reference,
            "reserved_center_bytes": center,
            "operational_p95_reserved_bytes": upper,
            "admission_upper_reserved_bytes": admission_upper,
            "safe_limit_bytes": safe_limit,
            "headroom_to_safe_limit_bytes": headroom,
            "base_physical_model_admitted": base_physical_model_admitted,
            "anchor_override_applied": anchor_override_applied,
            "admission_source": (
                "physical_model_operational_p95"
                if base_physical_model_admitted
                else (
                    "versioned_historical_anchor"
                    if anchor_override_applied
                    else "rejected"
                )
            ),
            "physical_model_admitted": physical_model_admitted,
            "admitted": policy_admitted,
            "rejection_reason": rejection_reason,
            "tail_source": prediction.get("success_tail_source"),
            "success_log_residual_upper": prediction.get("success_log_residual_upper"),
            "oom_exact_selector_log_guard": prediction.get(
                "oom_exact_selector_log_guard"
            ),
            "issues": prediction.get("issues") or [],
            "historical_anchor": anchor,
            "gib": {
                "analytic_reference": _gib(reference),
                "reserved_center": _gib(center),
                "operational_p95": _gib(upper),
                "admission_upper": _gib(admission_upper),
                "safe_limit": _gib(safe_limit),
                "headroom_to_safe_limit": _gib(headroom),
            },
        }

    @staticmethod
    def _ensure_group_contract(
        normalized_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        material_by_group: dict[str, str] = {}
        candidate_keys: set[tuple[str, tuple[Any, ...]]] = set()
        for row in normalized_rows:
            group_id = str(row["comparison_group"])
            material_sha = sha256_json(row["scenario_material"])
            previous = material_by_group.setdefault(group_id, material_sha)
            if previous != material_sha:
                raise ValueError(
                    f"comparison_group combines different user scenarios: {group_id}"
                )
            candidate_key = (
                int(row["gpu_count"]),
                int(row["physical_mbs"]),
                int(row["zero_stage"]),
                bool(row["gradient_checkpointing"]),
                bool(row["packing"]),
                str(row["dtype"]),
                str(row["kernel_path"]),
            )
            identity = (group_id, candidate_key)
            if identity in candidate_keys:
                raise ValueError(
                    "comparison_group contains duplicate candidate "
                    f"configuration: {group_id}"
                )
            candidate_keys.add(identity)

    @staticmethod
    def _scale_out_report(
        requested_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> dict[str, Any]:
        """Expose the scale-out contract without enabling it prematurely.

        The current frozen v4b artifact has a point ranking proxy but no
        calibrated throughput lower/upper bounds.  The resulting decision is
        therefore intentionally ``conservative_bound_unavailable``; this
        method makes that missing evidence visible to callers and gives the
        later ratio calibration a stable input/output seam.
        """

        groups: list[dict[str, Any]] = []
        for group_id in sorted(requested_by_group):
            requested = requested_by_group[group_id]
            summaries: list[dict[str, Any]] = []
            by_gpu: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
            for row in requested:
                by_gpu[int(row["configuration"]["gpu_count"])].append(row)
            for gpu_count in DEFAULT_GPU_ORDER:
                rows_at_gpu = by_gpu.get(gpu_count, [])
                material_hashes = {
                    sha256_json(row.get("scenario_material"))
                    for row in requested
                    if row.get("scenario_material") is not None
                }
                runtime_hashes = {
                    str(row.get("runtime_mechanism_component_sha256"))
                    for row in requested
                    if row.get("runtime_mechanism_component_sha256")
                }
                scenario_material_sha256 = (
                    next(iter(material_hashes)) if len(material_hashes) == 1 else None
                )
                runtime_mechanism_sha256 = (
                    next(iter(runtime_hashes)) if len(runtime_hashes) == 1 else None
                )
                admitted = [
                    row
                    for row in rows_at_gpu
                    if (row.get("memory") or {}).get("admitted") is True
                ]
                ranked = [
                    row
                    for row in admitted
                    if (row.get("throughput") or {}).get(
                        "prediction_available"
                    )
                    is True
                ]
                ranked.sort(
                    key=lambda row: (
                        -float(
                            (row.get("throughput") or {}).get(
                                "throughput_proxy_tokens_per_second",
                                0.0,
                            )
                        ),
                        int(row.get("input_index", 0)),
                    )
                )
                best = ranked[0] if ranked else None
                summaries.append(
                    {
                        "gpu_count": gpu_count,
                        "memory_gate_passed": bool(admitted),
                        "admitted_candidate_count": len(admitted),
                        "best_candidate_request_id": (
                            str(best["request_id"]) if best is not None else None
                        ),
                        "scenario_material_sha256": scenario_material_sha256,
                        "runtime_mechanism_component_sha256": runtime_mechanism_sha256,
                        "predicted_throughput": (
                            float(
                                best["throughput"][
                                    "throughput_proxy_tokens_per_second"
                                ]
                            )
                            if best is not None
                            else None
                        ),
                        # v4b currently exposes no conservative endpoint
                        # bounds; never reuse its point proxy as one.
                        "conservative_lower_throughput": None,
                        "conservative_upper_throughput": None,
                    }
                )
            groups.append(
                {
                    "comparison_group": group_id,
                    **evaluate_scale_out_sequence(
                        summaries,
                        gpu_order=DEFAULT_GPU_ORDER,
                        minimum_ratio=DEFAULT_MINIMUM_THROUGHPUT_RATIO,
                    ),
                }
            )
        return {
            "schema": "sft_cross_card_scale_out_report/v1",
            "selection_policy": SCALE_OUT_POLICY_ID,
            "enabled": False,
            "minimum_ratio": DEFAULT_MINIMUM_THROUGHPUT_RATIO,
            "gpu_order": list(DEFAULT_GPU_ORDER),
            "status": "disabled_pending_conservative_ratio_calibration",
            "automatic_execution_allowed": False,
            "groups": groups,
        }

    def predict(
        self,
        requests: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if (
            isinstance(requests, (str, bytes, Mapping))
            or not isinstance(requests, Sequence)
            or not requests
        ):
            raise ValueError("At least one candidate is required")
        if not all(isinstance(request, Mapping) for request in requests):
            raise ValueError("Every candidate must be a JSON object")
        rows: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        normalized_rows: list[dict[str, Any]] = []
        support_rows: list[dict[str, Any]] = []
        request_ids: set[str] = set()

        for input_index, request in enumerate(requests):
            record, normalized, support = self._record(
                request,
                input_index=input_index,
            )
            request_id = str(normalized["request_id"])
            if request_id in request_ids:
                raise ValueError(f"Duplicate candidate request_id: {request_id}")
            request_ids.add(request_id)
            records.append(record)
            normalized_rows.append(normalized)
            support_rows.append(support)

        self._ensure_group_contract(normalized_rows)
        v4b_candidates = []
        for input_index, (record, normalized, support) in enumerate(
            zip(records, normalized_rows, support_rows)
        ):
            memory_result = self._memory_result(record, support)
            admitted = bool(memory_result["admitted"])
            work = record["performance"]["work_per_step"]
            row = {
                "input_index": input_index,
                "request_id": normalized["request_id"],
                "comparison_group": normalized["comparison_group"],
                "scenario_material": normalized["scenario_material"],
                "runtime_mechanism_component_sha256": (
                    H800_RUNTIME_MECHANISM_COMPONENT_SHA256
                ),
                "configuration": {
                    "hardware_id": "h800",
                    "model_id": normalized["model_id"],
                    "training_mode": normalized["training_mode"],
                    "lora_rank": normalized["lora_rank"],
                    "dataset_id": normalized["dataset_id"],
                    "dataset_category": normalized["dataset_category"],
                    "dataset_profile_sha256": normalized["dataset_profile_binding"][
                        "sha256"
                    ],
                    "target_gbs": normalized["target_gbs"],
                    "cutoff_len": normalized["cutoff_len"],
                    "gpu_count": normalized["gpu_count"],
                    "physical_mbs": normalized["physical_mbs"],
                    "gradient_accumulation_steps": record["performance"][
                        "gradient_accumulation_steps"
                    ],
                    "zero_stage": normalized["zero_stage"],
                    "gradient_checkpointing": normalized["gradient_checkpointing"],
                    "packing": normalized["packing"],
                    "offload": normalized["offload"],
                    "dtype": normalized["dtype"],
                    "kernel_path": normalized["kernel_path"],
                },
                "dataset_profile": normalized["dataset_profile_binding"],
                "support": support,
                "memory": memory_result,
                "throughput": {
                    "prediction_available": False,
                    "reason": (
                        "pending_candidate_set_ranking"
                        if admitted
                        else memory_result["rejection_reason"]
                    ),
                    "absolute_scale_trusted": False,
                },
                "rank_within_admitted_group": None,
                "rank_within_gpu_count": None,
            }
            rows.append(row)
            if not admitted:
                continue
            effective_work = float(work["effective_tokens"])
            v4b_candidates.append(
                {
                    "scenario_id": normalized["comparison_group"],
                    "scenario": {
                        "model_id": normalized["model_id"],
                        "training_mode": normalized["training_mode"],
                        "dataset_id": normalized["dataset_id"],
                        "target_gbs": normalized["target_gbs"],
                    },
                    "candidate_key": [
                        normalized["gpu_count"],
                        normalized["physical_mbs"],
                        normalized["zero_stage"],
                        normalized["gradient_checkpointing"],
                        normalized["packing"],
                        normalized["dtype"],
                        normalized["kernel_path"],
                    ],
                    "record": record,
                    "features": _record_features(
                        _feature_record(
                            record,
                            normalized["dataset_category"],
                        )
                    ),
                    "effective_log_work": math.log(effective_work),
                    "log_analytic_anchor": (_record_log_analytic_anchor(record)),
                    "request_id": normalized["request_id"],
                    "input_index": input_index,
                }
            )

        predicted = _predict_two_head_entries(
            v4b_candidates,
            self.throughput_model,
        )
        for candidate, log_score in predicted:
            score = float(log_score)
            proxy = math.exp(score)
            effective_work = math.exp(float(candidate["effective_log_work"]))
            row = rows[int(candidate["input_index"])]
            row["throughput"] = {
                "prediction_available": True,
                "ranking_score_log": score,
                "throughput_proxy_tokens_per_second": proxy,
                "proxy_step_seconds": effective_work / proxy,
                "static_effective_tokens_per_step": effective_work,
                "candidate_set_dependent": True,
                "absolute_scale_trusted": False,
                "reason": None,
            }

        ranked_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        requested_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            group_id = str(row["comparison_group"])
            requested_by_group[group_id].append(row)
            if row["throughput"]["prediction_available"] is True:
                ranked_by_group[group_id].append(row)

        groups = []
        for group_id in sorted(requested_by_group):
            ranked = sorted(
                ranked_by_group.get(group_id, []),
                key=lambda row: (
                    -float(row["throughput"]["throughput_proxy_tokens_per_second"]),
                    int(row["input_index"]),
                ),
            )
            for rank, row in enumerate(ranked, start=1):
                row["rank_within_admitted_group"] = rank
            ranked_by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in ranked:
                ranked_by_gpu[int(row["configuration"]["gpu_count"])].append(row)
            for gpu_rows in ranked_by_gpu.values():
                for rank, row in enumerate(gpu_rows, start=1):
                    row["rank_within_gpu_count"] = rank
            requested = requested_by_group[group_id]
            minimum_gpu_count = min(ranked_by_gpu, default=None)
            selected = (
                str(ranked_by_gpu[minimum_gpu_count][0]["request_id"])
                if minimum_gpu_count is not None
                else None
            )
            throughput_top = str(ranked[0]["request_id"]) if ranked else None
            groups.append(
                {
                    "comparison_group": group_id,
                    "requested_candidates": len(requested),
                    "physical_model_admitted_candidates": sum(
                        row["memory"]["physical_model_admitted"] is True
                        for row in requested
                    ),
                    "policy_admitted_candidates": len(ranked),
                    "rejected_by_memory": sum(
                        row["memory"]["rejection_reason"]
                        == "memory_upper_exceeds_safe_limit"
                        for row in requested
                    ),
                    "rejected_by_support_domain": sum(
                        row["memory"]["rejection_reason"] == "outside_supported_domain"
                        for row in requested
                    ),
                    "admitted_by_historical_anchor": sum(
                        row["memory"]["anchor_override_applied"] is True
                        for row in requested
                    ),
                    "status": (
                        "ranked_shadow_only" if ranked else "no_admitted_candidate"
                    ),
                    "selection_policy": SELECTION_POLICY,
                    "minimum_admitted_gpu_count": minimum_gpu_count,
                    "selected_request_id": selected,
                    "throughput_top_request_id": throughput_top,
                    "ranked_request_ids": [str(row["request_id"]) for row in ranked],
                    "automatic_execution_allowed": False,
                }
            )

        report: dict[str, Any] = {
            "schema": SCHEMA,
            "implementation_version": IMPLEMENTATION_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "hardware_id": "h800",
            "runtime_binding": {
                "runtime_cohort_id": H800_RUNTIME_COHORT,
                "gpu_family": "H800",
                "runtime_mechanism_component_sha256": (
                    H800_RUNTIME_MECHANISM_COMPONENT_SHA256
                ),
            },
            "release": {
                "mode": "shadow_only",
                "automatic_execution_allowed": False,
                "reason": (
                    "The frozen physical-shares and v4b artifacts still "
                    "require prospective publication acceptance."
                ),
                "memory_publishable": bool(self.memory_report.get("publishable")),
                "throughput_publishable": bool(
                    self.throughput_report.get("publishable")
                ),
                "memory_publication_blockers": self.memory_report.get(
                    "publication_blockers"
                )
                or [],
            },
            "model_artifacts": {
                "memory": {
                    "path": str(self.memory_artifact_path.resolve()),
                    "sha256": sha256_file(self.memory_artifact_path),
                    "report_sha256": self.memory_report["report_sha256"],
                },
                "throughput": {
                    "path": str(self.throughput_artifact_path.resolve()),
                    "sha256": sha256_file(self.throughput_artifact_path),
                    "report_sha256": self.throughput_report["report_sha256"],
                },
                "memory_anchors": {
                    "path": str(self.memory_anchor_registry_path.resolve()),
                    "sha256": sha256_file(self.memory_anchor_registry_path),
                    "report_sha256": self.memory_anchor_registry["report_sha256"],
                    "registry_id": self.memory_anchor_registry["registry_id"],
                },
            },
            "policy": (
                "H800 physical-shares operational P95, followed only when "
                "needed by a tightly matched versioned historical anchor; "
                "then choose the minimum admitted GPU count and use frozen "
                "v4b ranking within that GPU count"
            ),
            "selection_policy": {
                "id": SELECTION_POLICY,
                "primary_objective": "minimum_admitted_gpu_count",
                "secondary_objective": "v4b_rank_within_gpu_count",
                "cross_gpu_throughput_ranking": "diagnostic_only",
                "scale_out_enabled": False,
            },
            "scale_out": self._scale_out_report(requested_by_group),
            "absolute_throughput_scale_trusted": False,
            "gpu_experiments_launched": False,
            "queues_mutated": False,
            "predictions": rows,
            "ranking_groups": groups,
        }
        report["report_sha256"] = sha256_json(report)
        validate_prediction_report(report)
        return report


def _load_requests(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = payload.get("candidates") or payload.get("requests")
    else:
        rows = None
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Input must be a non-empty list or contain candidates/requests"
        )
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("Every predictor candidate must be an object")
    return [dict(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--memory-artifact",
        type=Path,
        default=DEFAULT_MEMORY_ARTIFACT,
    )
    parser.add_argument(
        "--throughput-artifact",
        type=Path,
        default=DEFAULT_THROUGHPUT_ARTIFACT,
    )
    parser.add_argument(
        "--memory-anchor-registry",
        type=Path,
        default=DEFAULT_MEMORY_ANCHOR_REGISTRY,
    )
    parser.add_argument(
        "--additional-dataset-profile-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model-inventory",
        type=Path,
        default=DEFAULT_MODEL_INVENTORY,
    )
    parser.add_argument(
        "--allow-unfrozen-model-inventory",
        action="store_true",
        help="Permit a separately bound transfer-only model inventory.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    predictor = H800PhysicalV4BPredictor(
        memory_artifact=args.memory_artifact,
        throughput_artifact=args.throughput_artifact,
        memory_anchor_registry=args.memory_anchor_registry,
        model_inventory=args.model_inventory,
        strict_model_inventory_binding=not args.allow_unfrozen_model_inventory,
        additional_dataset_profile_dir=(args.additional_dataset_profile_dir),
    )
    report = predictor.predict(_load_requests(args.input))
    if args.output is not None:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
