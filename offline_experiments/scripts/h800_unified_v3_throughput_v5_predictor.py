#!/usr/bin/env python3
"""Active H800 unified-V3 memory gate plus structured throughput V5."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ROOT, read_json, sha256_file, sha256_json
from h800_frozen_vl_overlay_v1 import (
    load_artifact as load_vl_overlay_artifact,
)
from h800_frozen_vl_overlay_v1 import predict_shadow_overlay
from h800_physical_v4b_predictor import (
    H800_RUNTIME_COHORT,
    H800_RUNTIME_MECHANISM_COMPONENT_SHA256,
)
from h800_unified_bounded_memory_model import ARTIFACT_SCHEMA_V3, load_artifact
from h800_unified_bounded_memory_v3_data import RUNTIME_BASE_PARAMETERS
from h800_unified_v3_v4b_predictor import (
    ADMISSION_SOURCE,
    DEFAULT_HYBRID_MEMORY_ARTIFACT,
    DEFAULT_MEMORY_ARTIFACT,
    DEFAULT_MEMORY_FEATURE_INVENTORY,
    HYBRID_MEMORY_ARTIFACT_SCHEMA,
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
from vl_resource_features import build_vl_resource_features
import static_packing_predictor as static_packing
from packing_production_release import (
    DEFAULT_RELEASE_PATH as DEFAULT_PACKING_RELEASE,
    load_packing_release,
    scope_mismatches as packing_release_scope_mismatches,
)

SCHEMA = "sft_h800_unified_v3_throughput_v5_prediction/v1"
IMPLEMENTATION_VERSION = "sft_h800_unified_v3_throughput_v5_predictor/2026-08-18.v4"
SELECTION_POLICY = "minimum_admitted_gpu_count_then_throughput_v5"
THROUGHPUT_MODEL_ID = "structured_throughput_v5_single_output"
SUPPORTED_MBS = {1, 2, 4, 8, 16}
DEFAULT_VL_OVERLAY_ARTIFACT = (
    ROOT / "artifacts" / "h800_frozen_vl_residual_overlay_v1.json"
)
DEFAULT_VL_MODEL_INVENTORY = (
    ROOT / "artifacts" / "h800_qwen35_vl_supplement_model_inventory_v1.json"
)
DEFAULT_HYBRID_MODEL_INVENTORY = (
    ROOT / "artifacts" / "h800_bounded_memory_v2_model_inventory_with_hybrid_v1.json"
)
DEFAULT_HYBRID_VL_SAFETY_UPPER = (
    ROOT / "artifacts" / "h800_hybrid_vl_safety_upper_v2.json"
)
DEFAULT_PACKING_POLICY = ROOT / "artifacts" / "static_packing_policy_v1.json"
HYBRID_VL_SAFETY_UPPER_SCHEMA = "sft_h800_hybrid_vl_safety_upper/v2"
HYBRID_VL_V2_RELEASE_MODE = "candidate_pending_new_source_acceptance_v2"
VL_PROFILE_SCHEMA = "sft_vl_workload_profile/v2"


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("VL workload profile is empty")
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _vl_padding_statistics(
    profile: Mapping[str, Any],
    *,
    profile_path: Path,
    cutoff_len: int,
    physical_mbs: int,
) -> dict[str, Any]:
    """Build the V3 text-length basis directly from a VL V2 profile."""

    records = profile.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("VL workload profile has no records")
    lengths = [int((row or {}).get("total_tokens") or 0) for row in records]
    if min(lengths) <= 0:
        raise ValueError("VL workload profile has non-positive total_tokens")
    clipped = sorted(min(int(cutoff_len), value) for value in lengths)
    count = len(clipped)
    mbs = int(physical_mbs)
    expected_batch_max = sum(
        value
        * ((index / count) ** mbs - ((index - 1) / count) ** mbs)
        for index, value in enumerate(clipped, start=1)
    )
    mean = statistics.fmean(clipped)
    variance = statistics.fmean((value - mean) ** 2 for value in clipped)
    return {
        "path": str(profile_path.resolve()),
        "sha256": sha256_file(profile_path),
        "rows": count,
        "cutoff_len": int(cutoff_len),
        "physical_mbs": mbs,
        "mean_clipped_tokens": mean,
        "p50_clipped_tokens": _percentile(clipped, 50.0),
        "p90_clipped_tokens": _percentile(clipped, 90.0),
        "p99_clipped_tokens": _percentile(clipped, 99.0),
        "maximum_clipped_tokens": max(clipped),
        "truncation_fraction": sum(
            value >= int(cutoff_len) for value in lengths
        )
        / count,
        "coefficient_of_variation": (
            math.sqrt(variance) / mean if mean > 0.0 else None
        ),
        "expected_random_batch_max_tokens": expected_batch_max,
        "expected_random_batch_max_fraction_of_cutoff": (
            expected_batch_max / int(cutoff_len)
        ),
        "expected_padded_tokens_per_physical_batch": expected_batch_max * mbs,
        "formula": (
            "E[max(clipped_length_1..clipped_length_mbs)] from the empirical "
            "VL profile CDF"
        ),
    }


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
        or not isinstance(release.get("automatic_execution_allowed"), bool)
        or activation.get("authorization") != "explicit_user_acceptance_2026-08-11"
        or activation.get("packing_authorization")
        != "explicit_user_acceptance_2026-08-18"
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
    vl_contract = report.get("vl_overlay") or {}
    vl_artifact = artifacts.get("vl_overlay") or {}
    if (
        vl_contract.get("mode") != "shadow_only"
        or vl_contract.get("automatic_admission_allowed") is not False
        or vl_contract.get("automatic_ranking_allowed") is not False
        or vl_contract.get("packing_allowed") is not False
        or vl_contract.get("memory_output") != "candidate_v2_one_sided_upper"
        or vl_artifact.get("schema")
        != "sft_h800_frozen_vl_residual_overlay/v1"
        or vl_artifact.get("mode") != "shadow_only"
    ):
        raise ValueError("H800 VL shadow overlay contract drifted")
    hybrid_contract = report.get("hybrid_memory") or {}
    hybrid_artifact = artifacts.get("hybrid_memory") or {}
    if (
        hybrid_contract.get("mode") != "shadow_only"
        or hybrid_contract.get("automatic_admission_allowed") is not False
        or hybrid_contract.get("automatic_ranking_allowed") is not False
        or hybrid_contract.get("memory_output") != "candidate_v2_one_sided_upper"
        or hybrid_artifact.get("schema") != "sft_h800_hybrid_memory_artifact/v1"
        or hybrid_artifact.get("mode") != "shadow_only"
    ):
        raise ValueError("H800 hybrid-memory shadow contract drifted")
    packing_release_artifact = artifacts.get("packing_production_release") or {}
    packing_policy_artifact = artifacts.get("packing_policy") or {}
    if (
        packing_release_artifact.get("schema")
        != "sft_h800_text_packing_production_release/v1"
        or packing_release_artifact.get("release_id")
        != "h800_text_packing_limited_production/2026-08-18.v1"
        or packing_release_artifact.get("status") != "active_limited_production"
        or packing_policy_artifact.get("policy_id")
        != "h800_text_sft_static_packing/2026-07-30.v1"
    ):
        raise ValueError("H800 pure-text Packing artifact binding drifted")
    text_packing = report.get("pure_text_packing") or {}
    if (
        text_packing.get("input_combination") != "pure_text_plus_packing"
        or text_packing.get("vl_packing_allowed") is not False
        or text_packing.get("candidate_prediction_available") is not True
        or text_packing.get("candidate_ranking_available") is not True
        or text_packing.get("automatic_execution_allowed") is not True
        or text_packing.get("support_tier") != "limited_production"
        or text_packing.get("release_id")
        != "h800_text_packing_limited_production/2026-08-18.v1"
    ):
        raise ValueError("H800 pure-text Packing contract drifted")
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
        if support.get("automatic_execution_allowed") is True:
            packing_admission = support.get("packing_production_admission") or {}
            admission_mode = packing_admission.get("admission_mode")
            if (
                row.get("vl_shadow") is not None
                or (row.get("configuration") or {}).get("packing") is not True
                or packing_admission.get("verified") is not True
                or admission_mode
                not in {"explicit_packing_request", "static_policy_toggle"}
                or (
                    admission_mode == "static_policy_toggle"
                    and not packing_admission.get("static_policy_on_baseline_mbs")
                )
                or not packing_admission.get("profile_sha256")
            ):
                raise ValueError("Packing automatic execution escaped its release gate")
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
        vl_shadow = row.get("vl_shadow")
        if vl_shadow is not None:
            vl_memory = vl_shadow.get("memory") or {}
            vl_throughput = vl_shadow.get("throughput") or {}
            if (
                vl_shadow.get("mode") != "shadow_only"
                or vl_shadow.get("automatic_admission_allowed") is not False
                or vl_shadow.get("automatic_ranking_allowed") is not False
                or vl_shadow.get("fit_scope_matched") is not True
                or vl_shadow.get("recommendation_status")
                != HYBRID_VL_V2_RELEASE_MODE
                or vl_memory.get("safety_upper_bytes") is None
                or memory.get("admitted") is not False
                or throughput_available is not False
            ):
                raise ValueError("H800 VL row escaped the shadow-only gate")
            center = float(vl_memory["predicted_center_bytes"])
            upper = float(vl_memory["safety_upper_bytes"])
            step_seconds = float(vl_throughput["predicted_step_seconds"])
            tokens_per_second = float(
                vl_throughput["effective_tokens_per_second"]
            )
            if not all(
                math.isfinite(value) and value > 0.0
                for value in (center, upper, step_seconds, tokens_per_second)
            ):
                raise ValueError("H800 VL shadow prediction is invalid")
            if upper < center:
                raise ValueError("H800 VL safety upper is below its center")
        hybrid_shadow = row.get("hybrid_shadow")
        if hybrid_shadow is not None:
            center = float(hybrid_shadow["predicted_center_bytes"])
            if (
                hybrid_shadow.get("mode") != "shadow_only"
                or hybrid_shadow.get("automatic_admission_allowed") is not False
                or hybrid_shadow.get("automatic_ranking_allowed") is not False
                or hybrid_shadow.get("safety_upper_bytes") is None
                or memory.get("safety_upper_calibrated") is not True
                or memory.get("admitted") is not False
                or throughput_available is not False
                or not math.isfinite(center)
                or center <= 0.0
            ):
                raise ValueError("H800 hybrid-memory row escaped the shadow-only gate")
            if float(hybrid_shadow["safety_upper_bytes"]) < center:
                raise ValueError("H800 hybrid-memory upper is below its center")

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
            or group.get("automatic_execution_allowed")
            is not bool(
                selected
                and next(
                    row
                    for row in ranked
                    if str(row["request_id"]) == selected
                )["support"].get("automatic_execution_allowed")
                is True
            )
        ):
            raise ValueError("H800 V3+V5 group summary drifted")
    if declared_groups != set(rows_by_group):
        raise ValueError("H800 V3+V5 report contains undeclared groups")
    if release.get("automatic_execution_allowed") is not any(
        group.get("automatic_execution_allowed") is True for group in groups
    ):
        raise ValueError("H800 report-level automatic execution summary drifted")


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
        vl_overlay_artifact: Path = DEFAULT_VL_OVERLAY_ARTIFACT,
        vl_model_inventory: Path = DEFAULT_VL_MODEL_INVENTORY,
        hybrid_memory_artifact: Path = DEFAULT_HYBRID_MEMORY_ARTIFACT,
        hybrid_model_inventory: Path = DEFAULT_HYBRID_MODEL_INVENTORY,
        hybrid_vl_safety_upper: Path = DEFAULT_HYBRID_VL_SAFETY_UPPER,
        packing_policy: Path = DEFAULT_PACKING_POLICY,
        packing_release: Path = DEFAULT_PACKING_RELEASE,
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

        self.hybrid_memory_artifact_path = Path(hybrid_memory_artifact)
        self.hybrid_memory_artifact = read_json(self.hybrid_memory_artifact_path)
        if (
            self.hybrid_memory_artifact.get("schema")
            != HYBRID_MEMORY_ARTIFACT_SCHEMA
            or self.hybrid_memory_artifact.get("status")
            != "hybrid_shadow_candidate"
            or self.hybrid_memory_artifact.get("production_admission_allowed")
            is not False
        ):
            raise ValueError("hybrid memory artifact must remain shadow-only")
        self.hybrid_model_inventory_path = Path(hybrid_model_inventory)
        self.hybrid_model_inventory = read_json(self.hybrid_model_inventory_path)
        hybrid_inventory_models = {
            str(row["id"]): dict(row)
            for row in self.hybrid_model_inventory.get("models") or []
        }
        self.hybrid_model_ids = set(
            str(value) for value in self.hybrid_memory_artifact.get("model_ids") or []
        )
        missing_hybrid_models = sorted(
            self.hybrid_model_ids - set(hybrid_inventory_models)
        )
        if not self.hybrid_model_ids or missing_hybrid_models:
            raise ValueError(
                "hybrid memory inventory is incomplete: "
                + ", ".join(missing_hybrid_models)
            )
        for model_id in self.hybrid_model_ids:
            model = hybrid_inventory_models[model_id]
            self.memory_models[model_id] = dict(model)
            self.base.models[model_id] = dict(model)

        self.hybrid_vl_safety_upper_path = Path(hybrid_vl_safety_upper)
        self.hybrid_vl_safety_upper = read_json(self.hybrid_vl_safety_upper_path)
        if (
            self.hybrid_vl_safety_upper.get("schema")
            != HYBRID_VL_SAFETY_UPPER_SCHEMA
            or (
                self.hybrid_vl_safety_upper.get("release_contract") or {}
            ).get("mode")
            != HYBRID_VL_V2_RELEASE_MODE
            or (
                self.hybrid_vl_safety_upper.get("release_contract") or {}
            ).get("automatic_admission_allowed")
            is not False
            or (
                self.hybrid_vl_safety_upper.get("release_contract") or {}
            ).get("vl_packing_allowed")
            is not False
            or (
                self.hybrid_vl_safety_upper.get("release_contract") or {}
            ).get("pure_text_packing_governed_by_main_predictor")
            is not True
        ):
            raise ValueError("hybrid/VL safety upper must remain a pending candidate")

        self.vl_overlay_artifact_path = Path(vl_overlay_artifact)
        self.vl_overlay_artifact = load_vl_overlay_artifact(
            self.vl_overlay_artifact_path
        )
        self.vl_model_inventory_path = Path(vl_model_inventory)
        self.vl_model_inventory = read_json(self.vl_model_inventory_path)
        inventory_models = {
            str(row["id"]): dict(row)
            for row in self.vl_model_inventory.get("models") or []
        }
        self.vl_model_ids = set(
            str(value)
            for value in (self.vl_overlay_artifact.get("fit_scope") or {}).get(
                "model_ids"
            )
            or []
        )
        missing_models = sorted(self.vl_model_ids - set(inventory_models))
        if not self.vl_model_ids or missing_models:
            raise ValueError(
                "VL overlay inventory is incomplete: " + ", ".join(missing_models)
            )
        self.vl_models = {
            model_id: inventory_models[model_id] for model_id in self.vl_model_ids
        }
        # These rows are used only to construct the text-matched base and the
        # frozen visual residual.  The support policy below remains unsupported,
        # so extending the registry cannot turn the overlay into an admission
        # or ranking model.
        for model_id, model in self.vl_models.items():
            self.memory_models[model_id] = dict(model)
            self.base.models[model_id] = dict(model)
        self._vl_runtime_profile_bindings: dict[str, dict[str, Any]] = {}
        self.packing_policy_path = Path(packing_policy).resolve()
        self.packing_policy = static_packing.load_policy(self.packing_policy_path)
        self.packing_release_path = Path(packing_release).resolve()
        self.packing_release = load_packing_release(self.packing_release_path)
        static_binding = self.packing_release["evidence_bindings"]["static_policy"]
        if (
            sha256_file(self.packing_policy_path) != static_binding["sha256"]
            or self.packing_policy.get("policy_id") != static_binding["policy_id"]
        ):
            raise ValueError("Packing production release uses a different static policy")

    def _packing_production_admission(
        self,
        request: Mapping[str, Any],
        normalized: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Verify explicit or static-policy Packing production admission."""

        mismatches = packing_release_scope_mismatches(request, self.packing_release)
        if mismatches:
            return {
                "verified": False,
                "reason": "outside_release_scope",
                "mismatches": mismatches,
            }
        declared_release_id = request.get("packing_release_id")
        if (
            declared_release_id is not None
            and declared_release_id != self.packing_release["release_id"]
        ):
            return {"verified": False, "reason": "release_id_missing_or_mismatched"}
        raw_profile_path = request.get("packing_profile_path")
        if not isinstance(raw_profile_path, str) or not raw_profile_path:
            raw_profile_path = normalized["dataset_profile_binding"].get("path")
        if not isinstance(raw_profile_path, str) or not raw_profile_path:
            return {"verified": False, "reason": "packing_profile_path_missing"}
        profile_path = Path(raw_profile_path).expanduser().resolve()
        if not profile_path.is_file():
            return {"verified": False, "reason": "packing_profile_missing"}
        profile_sha256 = sha256_file(profile_path)
        declared_profile_sha256 = request.get("packing_profile_sha256")
        if (
            declared_profile_sha256 is not None
            and declared_profile_sha256 != profile_sha256
        ) or normalized["dataset_profile_binding"]["sha256"] != profile_sha256:
            return {"verified": False, "reason": "packing_profile_binding_mismatch"}
        raw_baselines = request.get("packing_policy_enabled_baseline_mbs")
        if raw_baselines is None:
            return {
                "verified": True,
                "admission_mode": "explicit_packing_request",
                "release_id": self.packing_release["release_id"],
                "profile_sha256": profile_sha256,
                "static_policy_id": self.packing_policy["policy_id"],
                "static_policy_on_baseline_mbs": [],
                "static_decision_ids": [],
                "memory_and_throughput_runtime_gates_pending": True,
            }
        if not isinstance(raw_baselines, list) or not raw_baselines:
            return {"verified": False, "reason": "packing_baseline_binding_invalid"}
        try:
            baselines = sorted({int(value) for value in raw_baselines})
        except (TypeError, ValueError):
            return {"verified": False, "reason": "packing_baseline_binding_invalid"}
        decision_ids: list[str] = []
        enabled: list[int] = []
        for baseline_mbs in baselines:
            policy_request = {
                "request_id": f"{normalized['request_id']}-packing-gate-mbs{baseline_mbs}",
                "gpu_family": "H800",
                "modality": "text",
                "stage": "sft",
                "dtype": normalized["dtype"],
                "model_id": normalized["model_id"],
                "train_type": normalized["training_mode"],
                "cutoff_len": normalized["cutoff_len"],
                "target_gbs": normalized["target_gbs"],
                "gpu_count": normalized["gpu_count"],
                "no_packing_mbs": baseline_mbs,
                "profile_path": str(profile_path),
                "require_gbs_controllable": True,
                "epsilon_gbs": float(request.get("packing_gbs_epsilon", 0.10)),
            }
            try:
                decision = static_packing.build_decision(
                    policy_request,
                    policy=self.packing_policy,
                    policy_path=self.packing_policy_path,
                    request_base=ROOT,
                )
            except (OSError, TypeError, ValueError) as error:
                return {
                    "verified": False,
                    "reason": "static_policy_recheck_failed",
                    "error": f"{type(error).__name__}:{error}",
                }
            decision_ids.append(str(decision["decision_id"]))
            if (decision.get("recommendation") or {}).get("packing") is True:
                enabled.append(baseline_mbs)
        if not enabled:
            return {"verified": False, "reason": "static_policy_not_on"}
        return {
            "verified": True,
            "admission_mode": "static_policy_toggle",
            "release_id": self.packing_release["release_id"],
            "profile_sha256": profile_sha256,
            "static_policy_id": self.packing_policy["policy_id"],
            "static_policy_on_baseline_mbs": enabled,
            "static_decision_ids": decision_ids,
            "memory_and_throughput_runtime_gates_pending": True,
        }

    def _record(
        self,
        request: Mapping[str, Any],
        *,
        input_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        record, normalized, support = super()._record(
            request,
            input_index=input_index,
        )
        if str(normalized["model_id"]) in self.hybrid_model_ids:
            support["label"] = "unsupported"
            support["support_tier"] = "shadow_only"
            support["inside_initial_product_domain"] = False
            support["eligible_for_shadow_ranking"] = False
            support["automatic_execution_allowed"] = False
            support["required_experiment_family"] = "hybrid_memory_risk_upper"
            if not any(
                reason.get("code") == "hybrid_memory_center_only"
                for reason in support["reasons"]
            ):
                support["reasons"].append(
                    {
                        "code": "hybrid_memory_center_only",
                        "severity": "unsupported",
                        "message": (
                            "The hybrid-memory V2 development upper is attached, "
                            "but new source-disjoint acceptance has not passed."
                        ),
                    }
                )
        elif bool(normalized["packing"]):
            admission = self._packing_production_admission(request, normalized)
            support["packing_production_admission"] = admission
            if admission["verified"]:
                support["support_tier"] = "limited_production"
                support["inside_limited_packing_release_domain"] = True
                support["automatic_execution_allowed"] = True
                support["required_experiment_family"] = None
                support["fallback_policy"] = {
                    "automatic_execution_allowed": True,
                    "reason": "bounded Packing release; V3 and V5 runtime gates still apply",
                }
                support["reasons"].append(
                    {
                        "code": "packing_limited_production_release",
                        "severity": "caution",
                        "message": (
                            "Pure-text Packing is inside the bounded production "
                            "release and its static policy decision was rechecked."
                        ),
                    }
                )
            else:
                support["automatic_execution_allowed"] = False
                support["reasons"].append(
                    {
                        "code": "packing_production_release_not_verified",
                        "severity": "caution",
                        "message": str(admission["reason"]),
                    }
                )
        return record, normalized, support

    def _memory_result(
        self,
        record: Mapping[str, Any],
        support: Mapping[str, Any],
    ) -> dict[str, Any]:
        memory = super()._memory_result(record, support)
        if memory.get("hybrid_model") is not True:
            return memory
        hybrid_spec = self.hybrid_vl_safety_upper["hybrid_memory"]
        scenario = record["scenario"]
        selector = record["selector"]
        upper_key = (
            f"{scenario['model_id']}::g{int(scenario['gpu_count'])}_"
            f"z{int(selector['zero_stage'])}_"
            f"gc{int(bool(selector['gradient_checkpointing']))}"
        )
        conditional = hybrid_spec["conditional_by_model_mechanism"]
        selected_spec = conditional.get(upper_key)
        multiplier = float(
            selected_spec["upper_multiplier"]
            if selected_spec is not None
            else hybrid_spec["fallback_v1_upper_multiplier"]
        )
        center = float(memory["reserved_center_bytes"])
        upper = center * multiplier
        safe_limit = float(memory["safe_limit_bytes"])
        base_admitted = upper <= safe_limit
        policy_admitted = bool(
            base_admitted and support.get("label") != "unsupported"
        )
        memory.update(
            {
                "risk_guard_bytes": center,
                "risk_guard_multiplier": multiplier,
                "operational_p95_reserved_bytes": upper,
                "admission_upper_reserved_bytes": upper,
                "headroom_to_safe_limit_bytes": safe_limit - upper,
                "base_physical_model_admitted": base_admitted,
                "physical_model_admitted": base_admitted,
                "admitted": policy_admitted,
                "rejection_reason": (
                    "memory_upper_exceeds_safe_limit"
                    if not base_admitted
                    else "outside_supported_domain"
                    if support.get("label") == "unsupported"
                    else None
                ),
                "safety_upper_key": upper_key,
                "safety_upper_key_observed_in_development": selected_spec is not None,
                "tail_source": "hybrid_v2_model_mechanism_upper_candidate",
                "safety_upper_calibrated": True,
                "safety_upper_development_only": True,
                "admission_upper_is_compatibility_center_only": False,
            }
        )
        memory["gib"].update(
            {
                "risk_guard": center / float(1 << 30),
                "operational_p95": upper / float(1 << 30),
                "admission_upper": upper / float(1 << 30),
                "headroom_to_safe_limit": (safe_limit - upper) / float(1 << 30),
            }
        )
        return memory

    @staticmethod
    def _vl_mechanism_id(request: Mapping[str, Any]) -> str:
        mbs = int(request["physical_mbs"])
        gc = bool(request["gradient_checkpointing"])
        if mbs == 1 and gc:
            return "SAFE"
        if mbs == 1 and not gc:
            return "NOGC"
        if mbs == 4 and not gc:
            return "PRESSURE"
        raise ValueError(
            "VL safety upper V2 supports SAFE(mbs1/gc), NOGC(mbs1/no-gc), "
            "or PRESSURE(mbs4/no-gc) only"
        )

    def _profile_binding(self, dataset_id: str) -> dict[str, Any]:
        binding = self._vl_runtime_profile_bindings.get(str(dataset_id))
        if binding is not None:
            return dict(binding)
        return super()._profile_binding(dataset_id)

    def _prepare_vl_requests(
        self,
        requests: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any] | None]]:
        prepared: list[dict[str, Any]] = []
        contexts: list[dict[str, Any] | None] = []
        fit_scope = self.vl_overlay_artifact.get("fit_scope") or {}
        for request in requests:
            item = dict(request)
            raw_path = item.get("vl_workload_profile_path")
            if raw_path is None:
                prepared.append(item)
                contexts.append(None)
                continue
            profile_path = Path(str(raw_path)).resolve()
            if not profile_path.is_file():
                raise ValueError(f"VL workload profile does not exist: {profile_path}")
            profile_sha256 = sha256_file(profile_path)
            declared_sha256 = item.get("vl_workload_profile_sha256")
            if declared_sha256 is not None and str(declared_sha256) != profile_sha256:
                raise ValueError("VL workload profile checksum mismatch")
            profile = read_json(profile_path)
            if profile.get("schema") != VL_PROFILE_SCHEMA:
                raise ValueError("VL overlay requires workload profile V2")
            model_id = str(item.get("model_id") or "")
            if model_id not in self.vl_model_ids:
                raise ValueError(f"model {model_id!r} is outside the VL overlay fit scope")
            if str((profile.get("model") or {}).get("model_id") or "") != model_id:
                raise ValueError("VL workload profile model binding mismatch")
            if str(item.get("training_mode") or "") != str(fit_scope["train_type"]):
                raise ValueError("VL overlay V2 supports LoRA only")
            if int(item.get("gpu_count") or 0) != int(fit_scope["gpu_count"]):
                raise ValueError("VL overlay V2 supports one GPU only")
            if int(item.get("zero_stage") or 0) != int(fit_scope["zero_stage"]):
                raise ValueError("VL overlay V2 supports ZeRO-0 only")
            if bool(item.get("packing")):
                raise ValueError("VL overlay V2 does not support packing")
            if item.get("freeze_vision_tower") is not True:
                raise ValueError("VL overlay V2 requires freeze_vision_tower=true")
            if item.get("freeze_multi_modal_projector") is not True:
                raise ValueError(
                    "VL overlay V2 requires freeze_multi_modal_projector=true"
                )
            records = profile.get("records")
            if not isinstance(records, list) or not records:
                raise ValueError("VL workload profile has no records")
            token_rows = [
                {
                    "total_tokens": int((row or {}).get("total_tokens") or 0),
                    "label_tokens": int((row or {}).get("label_tokens") or 0),
                }
                for row in records
            ]
            if min(row["total_tokens"] for row in token_rows) <= 0:
                raise ValueError("VL workload profile has non-positive total_tokens")
            runtime_dataset_id = f"vl_profile_{profile_sha256[:16]}"
            self.base.profiles.rows[runtime_dataset_id] = token_rows
            self._vl_runtime_profile_bindings[runtime_dataset_id] = {
                "path": str(profile_path),
                "sha256": profile_sha256,
                "origin": "vl_workload_profile_v2",
            }
            item["dataset_id"] = runtime_dataset_id
            # Dataset category is a legacy four-value request field.  V5 does
            # not use dataset identity as a fitted feature; the actual VL token
            # distribution comes from the bound V2 profile above.
            item["dataset_category"] = str(
                item.get("dataset_category") or "longtail"
            )
            cache_key = (
                str(profile_path),
                int(item["cutoff_len"]),
                int(item["physical_mbs"]),
            )
            self._memory_profile_cache[cache_key] = _vl_padding_statistics(
                profile,
                profile_path=profile_path,
                cutoff_len=int(item["cutoff_len"]),
                physical_mbs=int(item["physical_mbs"]),
            )
            features = build_vl_resource_features(
                profile,
                self.vl_models[model_id],
                physical_mbs=int(item["physical_mbs"]),
                freeze_vision_tower=True,
                freeze_multi_modal_projector=True,
            )
            mechanism_id = self._vl_mechanism_id(item)
            scale_key = f"{model_id}::{mechanism_id}"
            base_scales = self.hybrid_vl_safety_upper["vl_memory_by_modality"][
                "v1_text_base_scale_by_model_mechanism"
            ]
            if scale_key not in base_scales:
                raise ValueError(f"VL text-base scale is absent for {scale_key}")
            prepared.append(item)
            contexts.append(
                {
                    "profile_path": profile_path,
                    "profile_sha256": profile_sha256,
                    "profile": profile,
                    "features": features,
                    "model": self.vl_models[model_id],
                    "runtime_dataset_id": runtime_dataset_id,
                    "mechanism_id": mechanism_id,
                    "text_base_scale": float(base_scales[scale_key]),
                }
            )
        return prepared, contexts

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

        prepared_requests, vl_contexts = self._prepare_vl_requests(requests)

        records = []
        normalized_rows = []
        support_rows = []
        request_ids = set()
        for input_index, request in enumerate(prepared_requests):
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
            zip(prepared_requests, records, normalized_rows, support_rows)
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
            if memory.get("hybrid_model") is True:
                row["hybrid_shadow"] = {
                    "mode": "shadow_only",
                    "model_id": normalized["model_id"],
                    "predicted_center_bytes": memory["reserved_center_bytes"],
                    "safety_upper_bytes": memory["admission_upper_reserved_bytes"],
                    "upper_multiplier": memory["risk_guard_multiplier"],
                    "safety_upper_key": memory["safety_upper_key"],
                    "safety_upper_key_observed_in_development": memory[
                        "safety_upper_key_observed_in_development"
                    ],
                    "automatic_admission_allowed": False,
                    "automatic_ranking_allowed": False,
                    "reason": "V2 awaits new source-disjoint acceptance",
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

        vl_base_requests = [
            request
            for request, context in zip(prepared_requests, vl_contexts)
            if context is not None
        ]
        if vl_base_requests:
            vl_base_report = self.base.predict_many(vl_base_requests)
            vl_base_by_id = {
                str(item["request_id"]): item
                for item in vl_base_report["predictions"]
            }
            for row, context in zip(rows, vl_contexts):
                if context is None:
                    continue
                source = vl_base_by_id[str(row["request_id"])]
                work = source["work_per_step"]
                raw_text_center = row["memory"].get("reserved_center_bytes")
                if raw_text_center is None:
                    raise ValueError(
                        "VL shadow overlay requires an available text memory center"
                    )
                text_base_scale = float(context["text_base_scale"])
                text_center = float(raw_text_center) * text_base_scale
                shadow = predict_shadow_overlay(
                    text_memory_center_bytes=text_center,
                    text_step_seconds=float(source["predicted_step_seconds"]),
                    effective_tokens_per_step=float(work["effective_tokens"]),
                    logical_samples_per_step=float(work["logical_samples"]),
                    vl_features=context["features"],
                    model_family=str(context["model"]["family"]),
                    artifact=self.vl_overlay_artifact,
                )
                modality = (
                    "video"
                    if float(
                        context["features"]["feature_values"]["has_videos"]
                    )
                    > 0.0
                    else "image"
                )
                vl_specs = self.hybrid_vl_safety_upper["vl_memory_by_modality"]
                v1_overlay_center = float(
                    shadow["memory"]["predicted_center_bytes"]
                )
                visual_residual = float(shadow["memory"]["visual_residual_bytes"])
                base_guard_upper = (
                    float(row["memory"]["admission_upper_reserved_bytes"])
                    * max(1.0, text_base_scale)
                    + visual_residual
                )
                mechanism_key = (
                    f"{row['configuration']['model_id']}::"
                    f"{context['mechanism_id']}"
                )
                if modality == "image":
                    conditional = vl_specs["image"][
                        "conditional_by_model_mechanism"
                    ]
                    if mechanism_key not in conditional:
                        raise ValueError(
                            f"VL image V2 correction is absent for {mechanism_key}"
                        )
                    upper_spec = conditional[mechanism_key]
                    center_scale = float(upper_spec["total_center_scale"])
                    upper_multiplier = float(upper_spec["upper_multiplier"])
                    vl_center = v1_overlay_center * center_scale
                    conformal_upper = vl_center * upper_multiplier
                    # The legacy text-base guard is retained for diagnosis only.
                    # It was the direct cause of the Qwen3.5 PRESSURE false rejects
                    # and must not dominate the V2 image path.
                    safety_upper = conformal_upper
                    safety_upper_source = (
                        "image_v2_model_mechanism_cross_tier_oof_candidate"
                    )
                else:
                    upper_spec = vl_specs["video"]["v1_spec"]
                    center_scale = 1.0
                    upper_multiplier = float(upper_spec["upper_multiplier"])
                    vl_center = v1_overlay_center
                    conformal_upper = vl_center * upper_multiplier
                    safety_upper = max(conformal_upper, base_guard_upper)
                    safety_upper_source = "video_v1_oof_one_sided_upper_fallback"
                shadow["memory"].update(
                    {
                        "v1_overlay_center_bytes": v1_overlay_center,
                        "predicted_center_bytes": vl_center,
                        "raw_text_center_bytes": float(raw_text_center),
                        "text_base_scale": text_base_scale,
                        "total_center_scale": center_scale,
                        "model_mechanism_key": mechanism_key,
                        "upper_multiplier": upper_multiplier,
                        "conformal_upper_bytes": conformal_upper,
                        "legacy_v1_base_guard_upper_bytes": base_guard_upper,
                        "safety_upper_bytes": safety_upper,
                        "safe_limit_bytes": float(
                            row["memory"]["safe_limit_bytes"]
                        ),
                        "candidate_admitted_by_upper": safety_upper
                        <= float(row["memory"]["safe_limit_bytes"]),
                    }
                )
                shadow["profile_binding"] = {
                    "path": str(context["profile_path"]),
                    "sha256": context["profile_sha256"],
                    "schema": VL_PROFILE_SCHEMA,
                    "runtime_dataset_id": context["runtime_dataset_id"],
                }
                shadow["text_matched_base"] = {
                    "memory_model": "unified_v3_center_outside_admission_domain",
                    "throughput_model": THROUGHPUT_MODEL_ID,
                    "predicted_step_seconds": float(
                        source["predicted_step_seconds"]
                    ),
                    "effective_tokens_per_step": float(work["effective_tokens"]),
                    "logical_samples_per_step": float(work["logical_samples"]),
                }
                shadow["fit_scope_matched"] = True
                shadow["recommendation_status"] = (
                    HYBRID_VL_V2_RELEASE_MODE
                )
                shadow["safety_upper_source"] = safety_upper_source
                row["vl_shadow"] = shadow
                row["support"]["reasons"].append(
                    {
                        "code": "vl_overlay_shadow_only",
                        "severity": "unsupported",
                        "message": (
                            "The VL V2 development center and upper are attached, "
                            "but new source-disjoint acceptance has not passed."
                        ),
                    }
                )

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
            selected_row = (
                next(row for row in ranked if str(row["request_id"]) == selected)
                if selected is not None
                else None
            )
            automatic_execution_allowed = bool(
                selected_row is not None
                and selected_row["support"].get("automatic_execution_allowed")
                is True
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
                    "automatic_execution_allowed": automatic_execution_allowed,
                }
            )

        report_automatic_execution_allowed = any(
            group["automatic_execution_allowed"] is True for group in groups
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
                "automatic_execution_allowed": report_automatic_execution_allowed,
                "reason": (
                    "Unified memory V3 and structured throughput V5 are active; "
                    "only a verified bounded pure-text Packing selection may be "
                    "executed automatically."
                ),
                "memory_source_candidate_publishable": bool(
                    self.unified_memory_artifact.get("publishable")
                ),
                "throughput_source_candidate_publishable": bool(
                    self.throughput_report.get("publishable")
                ),
                "acceptance_monitoring_continues": True,
                "vl_overlay_mode": "shadow_only",
            },
            "activation_record": {
                "activated_at_utc": "2026-08-11T00:00:00+00:00",
                "authorization": "explicit_user_acceptance_2026-08-11",
                "scope": "recommendation_memory_and_throughput_models",
                "rollback_pipeline": "legacy_physical_memory_plus_v4b",
                "packing_authorization": "explicit_user_acceptance_2026-08-18",
                "packing_release_id": self.packing_release["release_id"],
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
            "vl_overlay": {
                "mode": "shadow_only",
                "automatic_admission_allowed": False,
                "automatic_ranking_allowed": False,
                "packing_allowed": False,
                "model_ids": sorted(self.vl_model_ids),
                "required_profile_schema": VL_PROFILE_SCHEMA,
                "memory_output": "candidate_v2_one_sided_upper",
                "reason": "V2 awaits new source-disjoint acceptance",
            },
            "pure_text_packing": {
                "input_combination": "pure_text_plus_packing",
                "vl_packing_allowed": False,
                "candidate_prediction_available": True,
                "candidate_ranking_available": True,
                "automatic_execution_allowed": True,
                "support_tier": "limited_production",
                "release_id": self.packing_release["release_id"],
                "release_scope": self.packing_release["scope"],
                "memory_source": "unified_v3_packing_features",
                "throughput_source": "accepted_frozen_unpacked_v5_at_virtual_mbs",
                "reason": (
                    "Pure-text Packing is enabled only inside the bounded release "
                    "scope after the static policy, V3 memory upper, and V5 "
                    "availability gates all pass."
                ),
                "known_limitation": self.packing_release["known_limitation"],
            },
            "hybrid_memory": {
                "mode": "shadow_only",
                "automatic_admission_allowed": False,
                "automatic_ranking_allowed": False,
                "model_ids": sorted(self.hybrid_model_ids),
                "memory_output": "candidate_v2_one_sided_upper",
                "reason": "V2 awaits new source-disjoint acceptance",
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
                "vl_overlay": {
                    "path": str(self.vl_overlay_artifact_path.resolve()),
                    "sha256": sha256_file(self.vl_overlay_artifact_path),
                    "artifact_sha256": self.vl_overlay_artifact["artifact_sha256"],
                    "schema": self.vl_overlay_artifact["schema"],
                    "mode": "shadow_only",
                },
                "vl_model_inventory": {
                    "path": str(self.vl_model_inventory_path.resolve()),
                    "sha256": sha256_file(self.vl_model_inventory_path),
                    "report_sha256": self.vl_model_inventory["report_sha256"],
                },
                "hybrid_memory": {
                    "path": str(self.hybrid_memory_artifact_path.resolve()),
                    "sha256": sha256_file(self.hybrid_memory_artifact_path),
                    "report_sha256": self.hybrid_memory_artifact["report_sha256"],
                    "schema": self.hybrid_memory_artifact["schema"],
                    "mode": "shadow_only",
                },
                "hybrid_model_inventory": {
                    "path": str(self.hybrid_model_inventory_path.resolve()),
                    "sha256": sha256_file(self.hybrid_model_inventory_path),
                    "report_sha256": self.hybrid_model_inventory["report_sha256"],
                },
                "hybrid_vl_safety_upper": {
                    "path": str(self.hybrid_vl_safety_upper_path.resolve()),
                    "sha256": sha256_file(self.hybrid_vl_safety_upper_path),
                    "report_sha256": self.hybrid_vl_safety_upper["report_sha256"],
                    "schema": self.hybrid_vl_safety_upper["schema"],
                    "mode": HYBRID_VL_V2_RELEASE_MODE,
                },
                "packing_policy": {
                    "path": str(self.packing_policy_path),
                    "sha256": sha256_file(self.packing_policy_path),
                    "policy_id": self.packing_policy["policy_id"],
                },
                "packing_production_release": {
                    "path": str(self.packing_release_path),
                    "sha256": sha256_file(self.packing_release_path),
                    "schema": self.packing_release["schema"],
                    "release_id": self.packing_release["release_id"],
                    "status": self.packing_release["status"],
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
