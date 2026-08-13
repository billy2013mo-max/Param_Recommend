#!/usr/bin/env python3
"""Active H800 unified-V3 memory gate plus structured throughput V5."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ROOT, read_json, sha256_file, sha256_json
from h800_physical_v4b_predictor import (
    H800_RUNTIME_COHORT,
    H800_RUNTIME_MECHANISM_COMPONENT_SHA256,
)
from h800_unified_bounded_memory_model import ARTIFACT_SCHEMA_V3, load_artifact
from h800_unified_bounded_memory_v3_data import RUNTIME_BASE_PARAMETERS
from h800_unified_v3_v4b_predictor import (
    ADMISSION_SOURCE,
    DEFAULT_MEMORY_ARTIFACT,
    DEFAULT_MEMORY_FEATURE_INVENTORY,
    V3_FEATURE_BUILDER,
    V3_FEATURE_BUILDER_SHA256,
    H800UnifiedV3V4BPredictor,
    _build_runtime_normalizer,
    _validate_inventory,
)
from throughput_predictor import (
    DEFAULT_MODEL_ARTIFACT as DEFAULT_THROUGHPUT_ARTIFACT,
)
from throughput_predictor import DEFAULT_MODEL_INVENTORY

SCHEMA = "sft_h800_unified_v3_throughput_v5_prediction/v1"
IMPLEMENTATION_VERSION = "sft_h800_unified_v3_throughput_v5_predictor/2026-08-11.v1"
SELECTION_POLICY = "minimum_admitted_gpu_count_then_throughput_v5"
THROUGHPUT_MODEL_ID = "structured_throughput_v5_single_output"
SUPPORTED_MBS = {1, 2, 4, 8, 16}


def validate_prediction_report(report: Mapping[str, Any]) -> None:
    """Validate V3 admission, V5 single-output ranking and selection."""

    if report.get("schema") != SCHEMA:
        raise ValueError("H800 V3+V5 prediction schema mismatch")
    unsigned = dict(report)
    expected = unsigned.pop("report_sha256", None)
    if not isinstance(expected, str) or expected != sha256_json(unsigned):
        raise ValueError("H800 V3+V5 prediction checksum mismatch")

    release = report.get("release") or {}
    activation = report.get("activation_record") or {}
    if (
        report.get("hardware_id") != "h800"
        or report.get("gpu_experiments_launched") is not False
        or report.get("queues_mutated") is not False
        or release.get("mode") != "active_recommendation"
        or release.get("automatic_execution_allowed") is not False
        or activation.get("authorization") != "explicit_user_acceptance_2026-08-11"
    ):
        raise ValueError("H800 V3+V5 release contract drifted")
    runtime = report.get("runtime_binding") or {}
    if (
        runtime.get("runtime_cohort_id") != H800_RUNTIME_COHORT
        or runtime.get("gpu_family") != "H800"
        or runtime.get("runtime_mechanism_component_sha256")
        != H800_RUNTIME_MECHANISM_COMPONENT_SHA256
    ):
        raise ValueError("H800 V3+V5 runtime binding drifted")
    artifacts = report.get("model_artifacts") or {}
    throughput_artifact = artifacts.get("throughput") or {}
    throughput_model = report.get("throughput_model") or {}
    if (
        throughput_artifact.get("schema") != "sft_structured_throughput_modeling/v1"
        or throughput_artifact.get("model_id") != THROUGHPUT_MODEL_ID
        or "throughput_legacy_memory_binding" in artifacts
        or throughput_model.get("single_output_used_for_absolute_and_ranking")
        is not True
        or throughput_model.get("dataset_id_used_as_model_feature") is not False
        or throughput_model.get("dataset_id_resolves_static_profile") is not True
    ):
        raise ValueError("H800 throughput V5 artifact binding drifted")
    selection = report.get("selection_policy") or {}
    if (
        selection.get("id") != SELECTION_POLICY
        or selection.get("scale_out_enabled") is not False
    ):
        raise ValueError("H800 V3+V5 selection policy drifted")

    predictions = report.get("predictions")
    groups = report.get("ranking_groups")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("H800 V3+V5 report has no predictions")
    if not isinstance(groups, list) or not groups:
        raise ValueError("H800 V3+V5 report has no ranking groups")
    request_ids = [str(row.get("request_id")) for row in predictions]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("H800 V3+V5 request ids are not unique")

    rows_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        rows_by_group[str(row["comparison_group"])].append(row)
        memory = row.get("memory") or {}
        support = row.get("support") or {}
        throughput = row.get("throughput") or {}
        if memory.get("anchor_override_applied") is not False:
            raise ValueError("unified V3 cannot use historical anchor overrides")
        available = memory.get("prediction_available") is True
        if available:
            center = float(memory["reserved_center_bytes"])
            risk = float(memory["risk_guard_bytes"])
            multiplier = float(memory["risk_guard_multiplier"])
            upper = float(memory["admission_upper_reserved_bytes"])
            safe_limit = float(memory["safe_limit_bytes"])
            if not all(
                math.isfinite(value) and value > 0.0
                for value in (center, risk, multiplier, upper, safe_limit)
            ):
                raise ValueError("H800 V3 memory values are invalid")
            if not math.isclose(
                upper,
                max(center, risk * multiplier),
                rel_tol=1.0e-12,
                abs_tol=1.0,
            ):
                raise ValueError("H800 V3 admission upper drifted")
            base_admitted = upper <= safe_limit
            if memory.get("admission_source") != ADMISSION_SOURCE:
                raise ValueError("H800 V3 admission source drifted")
        else:
            base_admitted = False
            if memory.get("admission_source") != "unavailable":
                raise ValueError("H800 V3 unavailable source drifted")
        policy_admitted = bool(base_admitted and support.get("label") != "unsupported")
        if memory.get("admitted") is not policy_admitted:
            raise ValueError("H800 V3 policy admission drifted")
        throughput_available = throughput.get("prediction_available") is True
        if throughput_available is not policy_admitted:
            raise ValueError("throughput V5 availability no longer follows V3")
        if throughput_available:
            predicted = float(throughput["predicted_effective_tokens_per_second"])
            step_seconds = float(throughput["predicted_step_seconds"])
            proxy = float(throughput["throughput_proxy_tokens_per_second"])
            if (
                not math.isfinite(predicted)
                or predicted <= 0.0
                or not math.isfinite(step_seconds)
                or step_seconds <= 0.0
                or not math.isclose(predicted, proxy, rel_tol=0.0, abs_tol=0.0)
                or throughput.get("model_id") != THROUGHPUT_MODEL_ID
                or throughput.get("single_output_used_for_absolute_and_ranking")
                is not True
                or throughput.get("memory_safety_checked_by_outer_gate") is not True
            ):
                raise ValueError("H800 throughput V5 output drifted")

    declared_groups = set()
    for group in groups:
        group_id = str(group["comparison_group"])
        if group_id in declared_groups:
            raise ValueError("H800 V3+V5 ranking group is duplicated")
        declared_groups.add(group_id)
        requested = rows_by_group.get(group_id, [])
        ranked = [
            row
            for row in requested
            if (row.get("throughput") or {}).get("prediction_available") is True
        ]
        ranked.sort(
            key=lambda row: (
                -float(row["throughput"]["predicted_effective_tokens_per_second"]),
                str(row["request_id"]),
            )
        )
        expected_ids = [str(row["request_id"]) for row in ranked]
        if list(group.get("ranked_request_ids") or []) != expected_ids:
            raise ValueError("H800 throughput V5 ranking order drifted")
        for rank, row in enumerate(ranked, start=1):
            if row.get("rank_within_admitted_group") != rank:
                raise ValueError("H800 throughput V5 group rank drifted")
        by_gpu: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in ranked:
            by_gpu[int(row["configuration"]["gpu_count"])].append(row)
        for gpu_rows in by_gpu.values():
            for rank, row in enumerate(gpu_rows, start=1):
                if row.get("rank_within_gpu_count") != rank:
                    raise ValueError("H800 throughput V5 GPU rank drifted")
        minimum_gpu = min(by_gpu, default=None)
        selected = (
            str(by_gpu[minimum_gpu][0]["request_id"])
            if minimum_gpu is not None
            else None
        )
        throughput_top = expected_ids[0] if expected_ids else None
        memory_rejections = sum(
            (row.get("memory") or {}).get("rejection_reason")
            == "memory_upper_exceeds_safe_limit"
            for row in requested
        )
        support_rejections = sum(
            (row.get("memory") or {}).get("rejection_reason")
            == "outside_supported_domain"
            for row in requested
        )
        expected_status = (
            "ranked_active_recommendation" if ranked else "no_admitted_candidate"
        )
        if (
            group.get("selection_policy") != SELECTION_POLICY
            or group.get("requested_candidates") != len(requested)
            or group.get("policy_admitted_candidates") != len(ranked)
            or group.get("rejected_by_memory") != memory_rejections
            or group.get("rejected_by_support_domain") != support_rejections
            or group.get("minimum_admitted_gpu_count") != minimum_gpu
            or group.get("selected_request_id") != selected
            or group.get("throughput_top_request_id") != throughput_top
            or group.get("status") != expected_status
            or group.get("automatic_execution_allowed") is not False
        ):
            raise ValueError("H800 V3+V5 group summary drifted")
    if declared_groups != set(rows_by_group):
        raise ValueError("H800 V3+V5 report contains undeclared groups")


class H800UnifiedV3ThroughputV5Predictor(H800UnifiedV3V4BPredictor):
    """Use unified memory V3, then structured throughput V5."""

    def __init__(
        self,
        *,
        memory_artifact: Path = DEFAULT_MEMORY_ARTIFACT,
        memory_feature_inventory: Path = DEFAULT_MEMORY_FEATURE_INVENTORY,
        throughput_artifact: Path = DEFAULT_THROUGHPUT_ARTIFACT,
        model_inventory: Path = DEFAULT_MODEL_INVENTORY,
        strict_model_inventory_binding: bool = True,
        additional_dataset_profile_dir: Path | None = None,
    ) -> None:
        self.throughput_artifact_path = Path(throughput_artifact)
        self.model_inventory_path = Path(model_inventory)
        self.base, self.structured_runtime_compatibility = _build_runtime_normalizer(
            model_inventory=self.model_inventory_path,
            strict_bindings=strict_model_inventory_binding,
            additional_dataset_profile_dir=additional_dataset_profile_dir,
            model_artifact=self.throughput_artifact_path,
        )
        self.throughput_report = self.base.report
        throughput_contract = self.throughput_report.get("model_contract") or {}
        if (
            throughput_contract.get("single_output_used_for_absolute_and_ranking")
            is not True
            or throughput_contract.get("dataset_id_used_as_model_feature") is not False
            or throughput_contract.get("runtime_outcome_features_used_as_inputs")
            is not False
            or throughput_contract.get("pre_run_static_profile_used") is not True
        ):
            raise ValueError("structured throughput V5 model contract drifted")
        self.supported_mbs = set(SUPPORTED_MBS)
        self.additional_dataset_profile_dir = (
            Path(additional_dataset_profile_dir)
            if additional_dataset_profile_dir is not None
            else None
        )

        self.memory_artifact_path = Path(memory_artifact)
        self.unified_memory_artifact = load_artifact(self.memory_artifact_path)
        if self.unified_memory_artifact.get("schema") != ARTIFACT_SCHEMA_V3:
            raise ValueError("active H800 memory artifact must be unified V3")
        for binding_name in ("implementation", "model_math", "data_builder"):
            binding = self.unified_memory_artifact["inputs"][binding_name]
            path = Path(str(binding["path"]))
            if not path.is_file() or sha256_file(path) != binding.get("sha256"):
                raise ValueError(f"unified V3 source binding drifted: {binding_name}")
        if sha256_file(V3_FEATURE_BUILDER) != V3_FEATURE_BUILDER_SHA256:
            raise ValueError("unified V3 runtime feature builder drifted")
        self.memory_feature_inventory_path = Path(memory_feature_inventory)
        self.memory_feature_inventory = read_json(self.memory_feature_inventory_path)
        _validate_inventory(self.memory_feature_inventory)
        self.memory_models = {
            str(row["id"]): dict(row) for row in self.memory_feature_inventory["models"]
        }
        for model_id, parameters in RUNTIME_BASE_PARAMETERS.items():
            if model_id in self.memory_models:
                self.memory_models[model_id]["actual_parameters"] = int(parameters)
        self.memory_fixed_lora = dict(self.memory_feature_inventory["fixed_lora"])
        self.memory_capacity_bytes = int(
            float(self.unified_memory_artifact["hardware_domain"]["capacity_bytes"])
        )
        runtime_capacity = int(float(self.base.hardware["h800"].memory_bytes))
        if runtime_capacity != self.memory_capacity_bytes:
            raise ValueError("unified V3 capacity does not match H800 runtime capacity")
        expected_safe_limit = self.memory_capacity_bytes * float(
            self.unified_memory_artifact["admission"]["safe_limit_fraction"]
        )
        artifact_safe_limit = float(
            self.unified_memory_artifact["hardware_domain"]["safe_limit_bytes"]
        )
        if not math.isclose(
            expected_safe_limit,
            artifact_safe_limit,
            rel_tol=0.0,
            abs_tol=1.0,
        ):
            raise ValueError("unified V3 safe-limit binding drifted")
        self._memory_profile_cache: dict[tuple[str, int, int], dict[str, Any]] = {}

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

        records = []
        normalized_rows = []
        support_rows = []
        request_ids = set()
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

        rows = []
        admitted_requests = []
        for input_index, (request, record, normalized, support) in enumerate(
            zip(requests, records, normalized_rows, support_rows)
        ):
            memory = self._memory_result(record, support)
            memory["admission_source"] = (
                ADMISSION_SOURCE
                if memory.get("prediction_available") is True
                else "unavailable"
            )
            admitted = memory["admitted"] is True
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
                "memory": memory,
                "throughput": {
                    "prediction_available": False,
                    "model_id": THROUGHPUT_MODEL_ID,
                    "reason": (
                        "pending_structured_throughput_v5"
                        if admitted
                        else memory["rejection_reason"]
                    ),
                    "single_output_used_for_absolute_and_ranking": True,
                },
                "rank_within_admitted_group": None,
                "rank_within_gpu_count": None,
            }
            rows.append(row)
            if admitted:
                admitted_requests.append(
                    {
                        **dict(request),
                        "request_id": normalized["request_id"],
                        "comparison_group": normalized["comparison_group"],
                        "hardware_id": "h800",
                        "model_id": normalized["model_id"],
                        "training_mode": normalized["training_mode"],
                        "lora_rank": normalized["lora_rank"],
                        "dataset_id": normalized["dataset_id"],
                        "target_gbs": normalized["target_gbs"],
                        "cutoff_len": normalized["cutoff_len"],
                        "gpu_count": normalized["gpu_count"],
                        "physical_mbs": normalized["physical_mbs"],
                        "zero_stage": normalized["zero_stage"],
                        "gradient_checkpointing": normalized["gradient_checkpointing"],
                        "packing": normalized["packing"],
                        "offload": normalized["offload"],
                        "dtype": normalized["dtype"],
                        "kernel_path": normalized["kernel_path"],
                    }
                )

        v5_report = (
            self.base.predict_many(admitted_requests) if admitted_requests else None
        )
        if v5_report is not None:
            v5_by_id = {
                str(item["request_id"]): item for item in v5_report["predictions"]
            }
            for row in rows:
                source = v5_by_id.get(str(row["request_id"]))
                if source is None:
                    continue
                predicted = float(source["predicted_effective_tokens_per_second"])
                row["throughput"] = {
                    "prediction_available": True,
                    "model_id": THROUGHPUT_MODEL_ID,
                    "predicted_effective_tokens_per_second": predicted,
                    "predicted_log_effective_tokens_per_second": float(
                        source["predicted_log_effective_tokens_per_second"]
                    ),
                    "predicted_step_seconds": float(source["predicted_step_seconds"]),
                    "predicted_computed_tokens_per_second": float(
                        source["predicted_computed_tokens_per_second"]
                    ),
                    "predicted_logical_samples_per_second": float(
                        source["predicted_logical_samples_per_second"]
                    ),
                    "throughput_proxy_tokens_per_second": predicted,
                    "work_per_step": source["work_per_step"],
                    "confidence": source["confidence"],
                    "single_output_used_for_absolute_and_ranking": True,
                    "absolute_scale_trusted": True,
                    "source_memory_safety_checked": bool(
                        source["memory_safety_checked"]
                    ),
                    "memory_safety_checked_by_outer_gate": True,
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
                    -float(row["throughput"]["predicted_effective_tokens_per_second"]),
                    str(row["request_id"]),
                ),
            )
            for rank, row in enumerate(ranked, start=1):
                row["rank_within_admitted_group"] = rank
            by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in ranked:
                by_gpu[int(row["configuration"]["gpu_count"])].append(row)
            for gpu_rows in by_gpu.values():
                for rank, row in enumerate(gpu_rows, start=1):
                    row["rank_within_gpu_count"] = rank
            requested = requested_by_group[group_id]
            minimum_gpu = min(by_gpu, default=None)
            selected = (
                str(by_gpu[minimum_gpu][0]["request_id"])
                if minimum_gpu is not None
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
                    "admitted_by_historical_anchor": 0,
                    "status": (
                        "ranked_active_recommendation"
                        if ranked
                        else "no_admitted_candidate"
                    ),
                    "selection_policy": SELECTION_POLICY,
                    "minimum_admitted_gpu_count": minimum_gpu,
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
                "structured_basis_compatibility": self.structured_runtime_compatibility,
            },
            "release": {
                "mode": "active_recommendation",
                "automatic_execution_allowed": False,
                "reason": (
                    "Unified memory V3 and structured throughput V5 are the "
                    "active recommendation models after explicit user acceptance."
                ),
                "memory_source_candidate_publishable": bool(
                    self.unified_memory_artifact.get("publishable")
                ),
                "throughput_source_candidate_publishable": bool(
                    self.throughput_report.get("publishable")
                ),
                "acceptance_monitoring_continues": True,
            },
            "activation_record": {
                "activated_at_utc": "2026-08-11T00:00:00+00:00",
                "authorization": "explicit_user_acceptance_2026-08-11",
                "scope": "recommendation_memory_and_throughput_models",
                "rollback_pipeline": "legacy_physical_memory_plus_v4b",
            },
            "memory_gate": {
                "id": "unified_v3_shared_center_independent_risk",
                "center": "shared_bounded_log_residual_center",
                "risk": "independent_shared_right_censored_risk_head",
                "upper_semantics": (
                    "max(center_bytes, risk_guard_bytes * upper_multiplier)"
                ),
                "safe_limit_fraction": float(
                    self.unified_memory_artifact["admission"]["safe_limit_fraction"]
                ),
                "historical_anchor_override_enabled": False,
            },
            "throughput_model": {
                "id": THROUGHPUT_MODEL_ID,
                "target": "effective_tokens_per_second",
                "single_output_used_for_absolute_and_ranking": True,
                "dataset_id_used_as_model_feature": False,
                "dataset_id_used_as_fitted_identity_feature": False,
                "dataset_id_resolves_static_profile": True,
                "runtime_outcome_features_used_as_inputs": False,
            },
            "model_artifacts": {
                "memory": {
                    "path": str(self.memory_artifact_path.resolve()),
                    "sha256": sha256_file(self.memory_artifact_path),
                    "artifact_sha256": self.unified_memory_artifact["artifact_sha256"],
                    "schema": self.unified_memory_artifact["schema"],
                },
                "memory_feature_inventory": {
                    "path": str(self.memory_feature_inventory_path.resolve()),
                    "sha256": sha256_file(self.memory_feature_inventory_path),
                    "report_sha256": self.memory_feature_inventory["report_sha256"],
                },
                "memory_feature_builder": {
                    "path": str(V3_FEATURE_BUILDER.resolve()),
                    "sha256": V3_FEATURE_BUILDER_SHA256,
                },
                "throughput": {
                    "path": str(self.throughput_artifact_path.resolve()),
                    "sha256": sha256_file(self.throughput_artifact_path),
                    "report_sha256": self.throughput_report["report_sha256"],
                    "schema": self.throughput_report["schema"],
                    "model_id": THROUGHPUT_MODEL_ID,
                },
                "throughput_predictor": {
                    "path": str(
                        (ROOT / "scripts" / "throughput_predictor.py").resolve()
                    ),
                    "sha256": sha256_file(ROOT / "scripts" / "throughput_predictor.py"),
                },
            },
            "policy": (
                "Use unified V3 for memory admission, then choose the minimum "
                "admitted GPU count and maximize the structured throughput V5 "
                "effective-tokens-per-second prediction within that GPU count"
            ),
            "selection_policy": {
                "id": SELECTION_POLICY,
                "primary_objective": "minimum_admitted_gpu_count",
                "secondary_objective": ("throughput_v5_effective_tokens_per_second"),
                "cross_gpu_throughput_ranking": "diagnostic_only",
                "scale_out_enabled": False,
            },
            "scale_out": self._scale_out_report(requested_by_group),
            "absolute_throughput_scale_trusted": True,
            "single_output_used_for_absolute_and_ranking": True,
            "gpu_experiments_launched": False,
            "queues_mutated": False,
            "predictions": rows,
            "ranking_groups": groups,
        }
        report["report_sha256"] = sha256_json(report)
        validate_prediction_report(report)
        return report
