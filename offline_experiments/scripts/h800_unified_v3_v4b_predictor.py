#!/usr/bin/env python3
"""Active H800 unified-V3 memory gate plus the unchanged frozen v4b ranker.

This adapter deliberately leaves the historical predictor and its artifacts
immutable.  The historical memory artifact is still loaded only because the
frozen v4b throughput artifact is cryptographically bound to it.  Candidate
admission, however, is decided exclusively by the unified V3 shared center and
independent risk head.  Historical memory anchors cannot override V3.

The release is active for recommendation output after the explicit 2026-08-11
user acceptance.  Automatic training execution remains disabled.
"""

from __future__ import annotations

import ast
import hashlib
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from analyze_h800_fresh_memory_residual_v2 import profile_padding_statistics
from common import ROOT, read_json, sha256_file, sha256_json
from fit_h800_unified_resource_partial_v1 import _current_features
from h800_physical_v4b_predictor import (
    DEFAULT_MEMORY_ANCHOR_REGISTRY,
    DEFAULT_THROUGHPUT_ARTIFACT,
    H800PhysicalV4BPredictor,
    _validate_memory_model,
    _validate_throughput_model,
)
from h800_physical_v4b_predictor import (
    DEFAULT_MEMORY_ARTIFACT as DEFAULT_LEGACY_MEMORY_ARTIFACT,
)
from h800_physical_v4b_predictor import (
    SCHEMA as LEGACY_PREDICTION_SCHEMA,
)
from h800_physical_v4b_predictor import (
    validate_prediction_report as validate_legacy_prediction_report,
)
from h800_unified_bounded_memory_model import (
    ARTIFACT_SCHEMA_V3,
    load_artifact,
    predict_records,
)
from h800_unified_bounded_memory_v3_data import RUNTIME_BASE_PARAMETERS
from memory_anchor_registry import load_registry
from structured_throughput_modeling import (
    StaticDatasetProfiles,
    validate_static_profile_report,
)
from structured_throughput_modeling import (
    validate_report as validate_structured_report,
)
from throughput_predictor import (
    DEFAULT_DATASET_PROFILE_DIR,
    DEFAULT_H800_HARDWARE,
    DEFAULT_H800_THEORY_BASIS,
    DEFAULT_MODEL_INVENTORY,
    DEFAULT_RTX4090_HARDWARE,
    DEFAULT_STATIC_PROFILE_ARTIFACT,
    ThroughputPredictor,
    _inventory_models,
)
from throughput_predictor import (
    DEFAULT_MODEL_ARTIFACT as DEFAULT_STRUCTURED_MODEL_ARTIFACT,
)

SCHEMA = "sft_h800_unified_v3_v4b_prediction/v1"
IMPLEMENTATION_VERSION = "sft_h800_unified_v3_v4b_predictor/2026-08-11.v1"
MEMORY_GATE_ID = "unified_v3_shared_center_independent_risk"
ADMISSION_SOURCE = "unified_v3_center_risk_upper"
DEFAULT_MEMORY_ARTIFACT = (
    ROOT / "artifacts" / "h800_unified_bounded_memory_candidate_v3.json"
)
DEFAULT_MEMORY_FEATURE_INVENTORY = (
    ROOT / "artifacts" / "h800_bounded_memory_v2_model_inventory_v1.json"
)
V3_FEATURE_BUILDER = ROOT / "scripts" / "fit_h800_unified_resource_partial_v1.py"
V3_FEATURE_BUILDER_SHA256 = (
    "c21ca67a4275e9c0a0d5f0cec896f7b88e8f840d606fba89a9a1c5efdf25726d"
)
GIB = float(1 << 30)
FROZEN_STRUCTURED_IMPLEMENTATION = (
    ROOT.parent
    / "offline_experiments"
    / "campaigns"
    / "rtx4090_generalization_20260729"
    / "qwen3_8b"
    / "scripts"
    / "structured_throughput_modeling.py"
)


def _gib(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) / GIB


def _validate_inventory(report: Mapping[str, Any]) -> None:
    unsigned = dict(report)
    expected = unsigned.pop("report_sha256", None)
    if not isinstance(expected, str) or expected != sha256_json(unsigned):
        raise ValueError("H800 V3 memory feature inventory checksum mismatch")
    if not isinstance(report.get("models"), list) or not report["models"]:
        raise ValueError("H800 V3 memory feature inventory has no models")
    fixed_lora = report.get("fixed_lora") or {}
    if int(fixed_lora.get("rank") or 0) != 32:
        raise ValueError("H800 V3 memory feature inventory LoRA rank drifted")


def _runtime_prefix_fingerprint(path: Path) -> str:
    """Hash inference code while excluding release/build-only declarations."""

    source = path.read_text(encoding="utf-8")
    prefix = source.split("def build_report(", 1)[0]
    tree = ast.parse(prefix)
    kept = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and target.id == "IMPLEMENTATION_VERSION"
                for target in targets
            ):
                continue
        kept.append(node)
    tree.body = kept
    material = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _build_runtime_normalizer(
    *,
    model_inventory: Path,
    strict_bindings: bool,
    additional_dataset_profile_dir: Path | None,
    model_artifact: Path = DEFAULT_STRUCTURED_MODEL_ARTIFACT,
) -> tuple[ThroughputPredictor, dict[str, Any]]:
    """Build the frozen structured-throughput inference runtime.

    The structured model's fit/report builder gained analysis-only fields on
    2026-08-09, so its whole-file hash no longer matches the 2026-07-28
    artifact.  The inference prefix is nevertheless byte-semantically
    identical after removing the version label.  We verify that invariant
    before using the current runtime basis and still enforce every data/input
    binding that the original normalizer enforced.
    """

    current_implementation = ROOT / "scripts" / "structured_throughput_modeling.py"
    current_fingerprint = _runtime_prefix_fingerprint(current_implementation)
    frozen_fingerprint = _runtime_prefix_fingerprint(FROZEN_STRUCTURED_IMPLEMENTATION)
    if current_fingerprint != frozen_fingerprint:
        raise ValueError(
            "structured throughput inference prefix drifted from the frozen runtime"
        )

    base = ThroughputPredictor.__new__(ThroughputPredictor)
    base.model_artifact_path = Path(model_artifact)
    base.static_profile_artifact_path = Path(DEFAULT_STATIC_PROFILE_ARTIFACT)
    base.dataset_profile_dir = Path(DEFAULT_DATASET_PROFILE_DIR)
    base.additional_dataset_profile_dir = (
        Path(additional_dataset_profile_dir)
        if additional_dataset_profile_dir is not None
        else None
    )
    base.model_inventory_path = Path(model_inventory)
    base.h800_hardware_path = Path(DEFAULT_H800_HARDWARE)
    base.h800_theory_basis_path = Path(DEFAULT_H800_THEORY_BASIS)
    base.rtx4090_hardware_path = Path(DEFAULT_RTX4090_HARDWARE)
    base.strict_bindings = bool(strict_bindings)
    base.report = read_json(base.model_artifact_path)
    validate_structured_report(base.report)
    expected_implementation_sha256 = str(
        base.report["source_bindings"]["implementation"]["sha256"]
    )
    if sha256_file(FROZEN_STRUCTURED_IMPLEMENTATION) != expected_implementation_sha256:
        raise ValueError(
            "frozen structured-throughput reference no longer matches its artifact"
        )
    base.frozen_model = base.report["frozen_model"]
    base.static_profile_report = read_json(base.static_profile_artifact_path)
    validate_static_profile_report(base.static_profile_report)
    base.profiles = StaticDatasetProfiles(base.dataset_profile_dir)
    if base.additional_dataset_profile_dir is not None:
        additional = StaticDatasetProfiles(base.additional_dataset_profile_dir)
        duplicates = sorted(set(base.profiles.rows) & set(additional.rows))
        if duplicates:
            raise ValueError(
                "Additional dataset profiles must not override frozen profile ids: "
                + ", ".join(duplicates)
            )
        base.profiles.rows.update(additional.rows)
    inventory = read_json(base.model_inventory_path)
    base.models, base.fixed_lora = _inventory_models(inventory)
    base.hardware = base._load_builtin_hardware()

    required = (
        ("static_workload_profiles", base.static_profile_artifact_path),
        ("h800_model_inventory", base.model_inventory_path),
        ("h800_hardware", base.h800_hardware_path),
        ("rtx4090_hardware", base.rtx4090_hardware_path),
    )
    mismatches = [
        name for name, path in required if not base._binding_matches(name, path)
    ]
    profile_bindings = base.report["source_bindings"].get("dataset_profiles") or {}
    for dataset_id, binding in profile_bindings.items():
        profile_path = base.dataset_profile_dir / f"{dataset_id}.qwen3_nothink.jsonl"
        if not profile_path.is_file() or sha256_file(profile_path) != binding.get(
            "sha256"
        ):
            mismatches.append(f"dataset_profile:{dataset_id}")
    if mismatches and base.strict_bindings:
        raise ValueError(
            "Frozen model input binding mismatch: " + ", ".join(sorted(mismatches))
        )
    base.binding_mismatches = sorted(mismatches)
    audit = {
        "policy": "ast_equivalent_inference_prefix_excluding_version_label",
        "current_implementation": str(current_implementation.resolve()),
        "frozen_reference": str(FROZEN_STRUCTURED_IMPLEMENTATION.resolve()),
        "inference_prefix_sha256": current_fingerprint,
        "frozen_reference_sha256": expected_implementation_sha256,
        "current_whole_file_sha256": sha256_file(current_implementation),
        "whole_file_binding_matches": (
            sha256_file(current_implementation) == expected_implementation_sha256
        ),
        "input_binding_mismatches": list(base.binding_mismatches),
    }
    return base, audit


def _initialize_legacy_components(
    instance: H800PhysicalV4BPredictor,
    *,
    legacy_memory_artifact: Path,
    throughput_artifact: Path,
    memory_anchor_registry: Path,
    model_inventory: Path,
    strict_model_inventory_binding: bool,
    additional_dataset_profile_dir: Path | None,
) -> None:
    """Initialize the immutable legacy wrapper without its stale fit-only hash check."""

    instance.memory_artifact_path = Path(legacy_memory_artifact)
    instance.throughput_artifact_path = Path(throughput_artifact)
    instance.memory_anchor_registry_path = Path(memory_anchor_registry)
    instance.memory_report = read_json(instance.memory_artifact_path)
    instance.throughput_report = read_json(instance.throughput_artifact_path)
    instance.memory_anchor_registry = load_registry(
        instance.memory_anchor_registry_path
    )
    _validate_memory_model(instance.memory_report)
    _validate_throughput_model(
        instance.throughput_report,
        memory_report=instance.memory_report,
        memory_artifact=instance.memory_artifact_path,
    )
    frozen_memory = instance.memory_report["memory"]["frozen_model"]
    instance.memory_center = frozen_memory["reserved_center"]
    instance.memory_tail = frozen_memory["tail"]
    instance.throughput_model = instance.throughput_report["h800"][
        "frozen_two_head_challenger"
    ]
    instance.supported_mbs = set(instance.memory_report["protocol"]["supported_mbs"])
    instance.model_inventory_path = Path(model_inventory)
    instance.base, instance.structured_runtime_compatibility = (
        _build_runtime_normalizer(
            model_inventory=instance.model_inventory_path,
            strict_bindings=strict_model_inventory_binding,
            additional_dataset_profile_dir=additional_dataset_profile_dir,
        )
    )
    instance.additional_dataset_profile_dir = (
        Path(additional_dataset_profile_dir)
        if additional_dataset_profile_dir is not None
        else None
    )


class H800LegacyRollbackPredictor(H800PhysicalV4BPredictor):
    """Legacy gate kept only for explicit rollback from the stable entrypoint."""

    def __init__(
        self,
        *,
        memory_artifact: Path = DEFAULT_LEGACY_MEMORY_ARTIFACT,
        throughput_artifact: Path = DEFAULT_THROUGHPUT_ARTIFACT,
        memory_anchor_registry: Path = DEFAULT_MEMORY_ANCHOR_REGISTRY,
        model_inventory: Path = DEFAULT_MODEL_INVENTORY,
        strict_model_inventory_binding: bool = True,
        additional_dataset_profile_dir: Path | None = None,
    ) -> None:
        _initialize_legacy_components(
            self,
            legacy_memory_artifact=memory_artifact,
            throughput_artifact=throughput_artifact,
            memory_anchor_registry=memory_anchor_registry,
            model_inventory=model_inventory,
            strict_model_inventory_binding=strict_model_inventory_binding,
            additional_dataset_profile_dir=additional_dataset_profile_dir,
        )

    def predict(
        self,
        requests: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        report = super().predict(requests)
        report["runtime_binding"]["structured_basis_compatibility"] = deepcopy(
            self.structured_runtime_compatibility
        )
        report.pop("report_sha256", None)
        report["report_sha256"] = sha256_json(report)
        validate_legacy_prediction_report(report)
        return report


def _legacy_validation_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Project the active report onto the frozen ranker's validation schema.

    The historical validator remains the source of truth for candidate-set
    ordering, minimum-GPU selection, fail-closed throughput availability and
    report-level runtime bindings.  This projection changes validation labels
    only; it never changes a prediction value.
    """

    projected = deepcopy(dict(report))
    projected["schema"] = LEGACY_PREDICTION_SCHEMA
    projected["release"] = {
        "mode": "shadow_only",
        "automatic_execution_allowed": False,
    }
    artifacts = projected["model_artifacts"]
    projected["model_artifacts"] = {
        "memory": artifacts["throughput_legacy_memory_binding"],
        "throughput": artifacts["throughput"],
        "memory_anchors": artifacts["legacy_memory_anchors_disabled"],
    }
    for row in projected["predictions"]:
        memory = row["memory"]
        memory["admission_source"] = (
            "physical_model_operational_p95"
            if memory.get("base_physical_model_admitted") is True
            else "rejected"
        )
    for group in projected["ranking_groups"]:
        if group.get("status") == "ranked_active_recommendation":
            group["status"] = "ranked_shadow_only"
    projected.pop("memory_gate", None)
    projected.pop("activation_record", None)
    projected.pop("report_sha256", None)
    projected["report_sha256"] = sha256_json(projected)
    return projected


def validate_prediction_report(report: Mapping[str, Any]) -> None:
    """Validate the active V3 admission contract and unchanged v4b ranking."""

    if report.get("schema") != SCHEMA:
        raise ValueError("H800 unified-V3 prediction schema mismatch")
    unsigned = dict(report)
    expected = unsigned.pop("report_sha256", None)
    if not isinstance(expected, str) or expected != sha256_json(unsigned):
        raise ValueError("H800 unified-V3 prediction report checksum mismatch")

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
        raise ValueError("H800 unified-V3 release safety contract drifted")
    gate = report.get("memory_gate") or {}
    if (
        gate.get("id") != MEMORY_GATE_ID
        or gate.get("historical_anchor_override_enabled") is not False
        or gate.get("upper_semantics")
        != "max(center_bytes, risk_guard_bytes * upper_multiplier)"
    ):
        raise ValueError("H800 unified-V3 memory gate contract drifted")

    predictions = report.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("H800 unified-V3 prediction report has no rows")
    for row in predictions:
        memory = row.get("memory") or {}
        support = row.get("support") or {}
        throughput = row.get("throughput") or {}
        if memory.get("anchor_override_applied") is not False:
            raise ValueError("H800 unified-V3 cannot use historical anchor overrides")
        available = memory.get("prediction_available") is True
        if available:
            center = float(memory["reserved_center_bytes"])
            risk_guard = float(memory["risk_guard_bytes"])
            multiplier = float(memory["risk_guard_multiplier"])
            upper = float(memory["admission_upper_reserved_bytes"])
            compatibility_upper = float(memory["operational_p95_reserved_bytes"])
            safe_limit = float(memory["safe_limit_bytes"])
            values = (center, risk_guard, multiplier, upper, safe_limit)
            if not all(math.isfinite(value) and value > 0.0 for value in values):
                raise ValueError("H800 unified-V3 memory prediction is invalid")
            if not math.isclose(
                upper,
                max(center, risk_guard * multiplier),
                rel_tol=1.0e-12,
                abs_tol=1.0,
            ):
                raise ValueError("H800 unified-V3 admission upper drifted")
            if not math.isclose(upper, compatibility_upper, rel_tol=0.0, abs_tol=1.0):
                raise ValueError("H800 unified-V3 compatibility upper drifted")
            base_admitted = upper <= safe_limit
            if memory.get("base_physical_model_admitted") is not base_admitted:
                raise ValueError("H800 unified-V3 base admission drifted")
            if memory.get("physical_model_admitted") is not base_admitted:
                raise ValueError("H800 unified-V3 physical admission drifted")
            if memory.get("admission_source") != ADMISSION_SOURCE:
                raise ValueError("H800 unified-V3 admission source drifted")
        else:
            base_admitted = False
            if memory.get("admission_source") != "unavailable":
                raise ValueError("H800 unavailable memory source drifted")
        policy_admitted = bool(base_admitted and support.get("label") != "unsupported")
        if memory.get("admitted") is not policy_admitted:
            raise ValueError("H800 unified-V3 policy admission drifted")
        if (throughput.get("prediction_available") is True) is not policy_admitted:
            raise ValueError("H800 v4b availability no longer follows V3 admission")

    # Reuse the mature frozen validator for all ranking and group invariants.
    validate_legacy_prediction_report(_legacy_validation_projection(report))


class H800UnifiedV3V4BPredictor(H800PhysicalV4BPredictor):
    """Use unified V3 for memory admission and frozen v4b for ranking."""

    def __init__(
        self,
        *,
        memory_artifact: Path = DEFAULT_MEMORY_ARTIFACT,
        memory_feature_inventory: Path = DEFAULT_MEMORY_FEATURE_INVENTORY,
        legacy_memory_artifact: Path = DEFAULT_LEGACY_MEMORY_ARTIFACT,
        throughput_artifact: Path = DEFAULT_THROUGHPUT_ARTIFACT,
        memory_anchor_registry: Path = DEFAULT_MEMORY_ANCHOR_REGISTRY,
        model_inventory: Path = DEFAULT_MODEL_INVENTORY,
        strict_model_inventory_binding: bool = True,
        additional_dataset_profile_dir: Path | None = None,
    ) -> None:
        # The legacy memory artifact is loaded solely to satisfy the immutable
        # source binding of the frozen v4b throughput model.
        _initialize_legacy_components(
            self,
            legacy_memory_artifact=legacy_memory_artifact,
            throughput_artifact=throughput_artifact,
            memory_anchor_registry=memory_anchor_registry,
            model_inventory=model_inventory,
            strict_model_inventory_binding=strict_model_inventory_binding,
            additional_dataset_profile_dir=additional_dataset_profile_dir,
        )
        self.legacy_memory_artifact_path = self.memory_artifact_path
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
            expected_safe_limit, artifact_safe_limit, rel_tol=0.0, abs_tol=1.0
        ):
            raise ValueError("unified V3 safe-limit binding drifted")
        self._memory_profile_cache: dict[tuple[str, int, int], dict[str, Any]] = {}

    @staticmethod
    def _mechanism_id(record: Mapping[str, Any]) -> str:
        scenario = record["scenario"]
        selector = record["selector"]
        return (
            f"{selector['training_mode']}_zero{selector['zero_stage']}_"
            f"gc{int(bool(selector['gradient_checkpointing']))}_"
            f"{scenario['gpu_count']}gpu_pack{int(bool(selector['packing']))}"
        )

    def _v3_inference_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        scenario = record["scenario"]
        selector = record["selector"]
        model_id = str(scenario["model_id"])
        if model_id not in self.memory_models:
            raise ValueError(f"model {model_id!r} is absent from V3 inventory")
        profile_path = Path(str(record["dataset_profile_binding"]["path"]))
        cutoff = int(scenario["cutoff_len"])
        mbs = int(scenario["physical_mbs"])
        profile_key = (str(profile_path.resolve()), cutoff, mbs)
        if profile_key not in self._memory_profile_cache:
            self._memory_profile_cache[profile_key] = profile_padding_statistics(
                profile_path,
                cutoff_len=cutoff,
                physical_mbs=mbs,
            )
        padding = self._memory_profile_cache[profile_key]
        if bool(selector["packing"]):
            aligned_effective_sequence = cutoff
        else:
            raw_max = int(padding["maximum_clipped_tokens"])
            aligned_effective_sequence = 8 * ((min(cutoff, raw_max) + 7) // 8)
        model_parameters = int(self.memory_models[model_id]["actual_parameters"])
        job = {
            "model_id": model_id,
            "model_parameters": model_parameters,
            "train_type": str(selector["training_mode"]),
            "gpu_count": int(scenario["gpu_count"]),
            "mbs": mbs,
            "cutoff_len": cutoff,
            "aligned_effective_sequence": aligned_effective_sequence,
            "zero": (
                "none"
                if int(selector["zero_stage"]) == 0
                else f"zero{int(selector['zero_stage'])}"
            ),
            "zero_stage": int(selector["zero_stage"]),
            "gc": bool(selector["gradient_checkpointing"]),
            "packing": bool(selector["packing"]),
            "offload": bool(selector.get("offload")),
            "dataset_profile_path": str(profile_path),
            "mechanism_id": self._mechanism_id(record),
            "packing_contract": {"expected_samples_per_pack": 1.0},
        }
        reference, features = _current_features(
            job,
            model_by_id=self.memory_models,
            fixed_lora=self.memory_fixed_lora,
            capacity_bytes=self.memory_capacity_bytes,
            profile_cache=self._memory_profile_cache,
        )
        return {
            "record_id": str(record["observation_id"]),
            "reference_bytes": reference,
            "features": features,
            "model_id": model_id,
            "train_type": str(selector["training_mode"]),
            "gpu_count": int(scenario["gpu_count"]),
            "zero_stage": int(selector["zero_stage"]),
            "gc": bool(selector["gradient_checkpointing"]),
            "mbs": mbs,
            "cutoff_len": cutoff,
            "packing": bool(selector["packing"]),
            "effective_sequence_tokens": aligned_effective_sequence,
        }

    @staticmethod
    def _unavailable_memory_result(
        record: Mapping[str, Any],
        *,
        safe_limit: float,
        issue: str,
    ) -> dict[str, Any]:
        return {
            "prediction_available": False,
            "analytic_reference_bytes": record["memory"]["analytic_reference_bytes"],
            "reserved_center_bytes": None,
            "risk_guard_bytes": None,
            "risk_guard_multiplier": None,
            "operational_p95_reserved_bytes": None,
            "admission_upper_reserved_bytes": None,
            "safe_limit_bytes": safe_limit,
            "headroom_to_safe_limit_bytes": None,
            "base_physical_model_admitted": False,
            "anchor_override_applied": False,
            "admission_source": "rejected",
            "physical_model_admitted": False,
            "admitted": False,
            "rejection_reason": "memory_prediction_unavailable",
            "tail_source": "unified_v3_independent_risk_head",
            "success_log_residual_upper": None,
            "oom_exact_selector_log_guard": None,
            "issues": [issue],
            "historical_anchor": {
                "matched": False,
                "override_allowed": False,
                "status": "disabled_by_unified_v3",
            },
            "gib": {
                "analytic_reference": _gib(
                    record["memory"]["analytic_reference_bytes"]
                ),
                "reserved_center": None,
                "risk_guard": None,
                "operational_p95": None,
                "admission_upper": None,
                "safe_limit": _gib(safe_limit),
                "headroom_to_safe_limit": None,
            },
        }

    def _memory_result(
        self,
        record: Mapping[str, Any],
        support: Mapping[str, Any],
    ) -> dict[str, Any]:
        safe_limit = float(
            self.unified_memory_artifact["hardware_domain"]["safe_limit_bytes"]
        )
        try:
            inference_record = self._v3_inference_record(record)
            prediction = predict_records(
                [inference_record], self.unified_memory_artifact
            )[0]
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return self._unavailable_memory_result(
                record,
                safe_limit=safe_limit,
                issue=f"v3_feature_construction_failed:{type(exc).__name__}:{exc}",
            )
        center = float(prediction["center_bytes"])
        risk_guard = float(prediction["risk_guard_bytes"])
        risk_multiplier = float(prediction["risk_guard_multiplier"])
        upper = float(prediction["admission_upper_bytes"])
        base_admitted = upper <= safe_limit
        policy_admitted = bool(base_admitted and support.get("label") != "unsupported")
        if not base_admitted:
            rejection_reason = "memory_upper_exceeds_safe_limit"
        elif support.get("label") == "unsupported":
            rejection_reason = "outside_supported_domain"
        else:
            rejection_reason = None
        headroom = safe_limit - upper
        reference = float(inference_record["reference_bytes"])
        # During super().predict(), the historical validator sees
        # operational_p95 as a compatibility admission-upper field.  The
        # active report relabels its source and states the exact semantics.
        return {
            "prediction_available": True,
            "analytic_reference_bytes": reference,
            "legacy_cutoff_reference_bytes": float(
                record["memory"]["analytic_reference_bytes"]
            ),
            "effective_sequence_tokens": int(
                inference_record["effective_sequence_tokens"]
            ),
            "reserved_center_bytes": center,
            "risk_guard_bytes": risk_guard,
            "risk_guard_multiplier": risk_multiplier,
            "operational_p95_reserved_bytes": upper,
            "operational_p95_is_compatibility_alias": True,
            "admission_upper_reserved_bytes": upper,
            "safe_limit_bytes": safe_limit,
            "headroom_to_safe_limit_bytes": headroom,
            "base_physical_model_admitted": base_admitted,
            "anchor_override_applied": False,
            "admission_source": (
                "physical_model_operational_p95" if base_admitted else "rejected"
            ),
            "physical_model_admitted": base_admitted,
            "admitted": policy_admitted,
            "rejection_reason": rejection_reason,
            "tail_source": "unified_v3_independent_risk_head",
            "success_log_residual_upper": None,
            "oom_exact_selector_log_guard": None,
            "issues": list(prediction.get("issues") or []),
            "historical_anchor": {
                "matched": False,
                "override_allowed": False,
                "status": "disabled_by_unified_v3",
            },
            "gib": {
                "analytic_reference": _gib(reference),
                "legacy_cutoff_reference": _gib(
                    record["memory"]["analytic_reference_bytes"]
                ),
                "reserved_center": _gib(center),
                "risk_guard": _gib(risk_guard),
                "operational_p95": _gib(upper),
                "admission_upper": _gib(upper),
                "safe_limit": _gib(safe_limit),
                "headroom_to_safe_limit": _gib(headroom),
            },
        }

    def predict(
        self,
        requests: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        report = super().predict(requests)
        legacy_artifacts = deepcopy(report["model_artifacts"])
        report["schema"] = SCHEMA
        report["implementation_version"] = IMPLEMENTATION_VERSION
        report["release"] = {
            "mode": "active_recommendation",
            "automatic_execution_allowed": False,
            "reason": (
                "Unified V3 replaces the legacy memory gate after explicit "
                "user acceptance; prospective acceptance monitoring continues."
            ),
            "memory_source_candidate_publishable": bool(
                self.unified_memory_artifact.get("publishable")
            ),
            "throughput_publishable": bool(self.throughput_report.get("publishable")),
            "acceptance_monitoring_continues": True,
        }
        report["activation_record"] = {
            "activated_at_utc": "2026-08-11T00:00:00+00:00",
            "authorization": "explicit_user_acceptance_2026-08-11",
            "scope": "recommendation_memory_gate_only",
            "rollback_memory_gate": "legacy_physical_v1",
        }
        report["memory_gate"] = {
            "id": MEMORY_GATE_ID,
            "center": "shared_bounded_log_residual_center",
            "risk": "independent_shared_right_censored_risk_head",
            "upper_semantics": (
                "max(center_bytes, risk_guard_bytes * upper_multiplier)"
            ),
            "safe_limit_fraction": float(
                self.unified_memory_artifact["admission"]["safe_limit_fraction"]
            ),
            "historical_anchor_override_enabled": False,
        }
        report["runtime_binding"]["structured_basis_compatibility"] = deepcopy(
            self.structured_runtime_compatibility
        )
        report["model_artifacts"] = {
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
            "throughput": legacy_artifacts["throughput"],
            "throughput_legacy_memory_binding": {
                "path": str(self.legacy_memory_artifact_path.resolve()),
                "sha256": sha256_file(self.legacy_memory_artifact_path),
                "report_sha256": self.memory_report["report_sha256"],
            },
            "legacy_memory_anchors_disabled": legacy_artifacts["memory_anchors"],
        }
        report["policy"] = (
            "Use the unified V3 shared center and independent censored-risk "
            "upper for admission, without historical anchor overrides; then "
            "choose the minimum admitted GPU count and use the unchanged "
            "frozen v4b ranker within that GPU count"
        )
        for row in report["predictions"]:
            memory = row["memory"]
            memory["admission_source"] = (
                ADMISSION_SOURCE
                if memory.get("prediction_available") is True
                else "unavailable"
            )
        for group in report["ranking_groups"]:
            if group.get("status") == "ranked_shadow_only":
                group["status"] = "ranked_active_recommendation"
        report.pop("report_sha256", None)
        report["report_sha256"] = sha256_json(report)
        validate_prediction_report(report)
        return report
