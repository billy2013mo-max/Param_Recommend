#!/usr/bin/env python3
"""Profile-aware H800 memory challenger with the frozen v4b ranker.

This is a shadow-only predictor.  It reuses the existing physical allocated
anchor and v4b throughput model, replaces only memory admission with the
versioned two-stage profile-aware challenger, and intentionally disables
historical-anchor overrides during prospective acceptance.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from analyze_h800_fresh_memory_residual_v2 import profile_padding_statistics
from common import ARTIFACT_DIR, read_json, sha256_file, sha256_json, write_json
from h800_challenger_modeling import _predict_memory_center
from h800_physical_v4b_predictor import (
    DEFAULT_MEMORY_ANCHOR_REGISTRY,
    DEFAULT_MEMORY_ARTIFACT,
    DEFAULT_THROUGHPUT_ARTIFACT,
    H800PhysicalV4BPredictor,
    _gib,
    _load_requests,
)
from h800_profile_aware_memory_model import ARTIFACT_SCHEMA, predict_memory


SCHEMA = "sft_h800_profile_aware_memory_v4b_prediction/v1"
IMPLEMENTATION_VERSION = (
    "sft_h800_profile_aware_v4b_predictor/2026-08-03.shadow-v1"
)
DEFAULT_CHALLENGER = ARTIFACT_DIR / "h800_profile_aware_memory_challenger_v1.json"


def _validate_challenger(
    artifact: Mapping[str, Any],
    *,
    path: Path,
    allocated_anchor_path: Path,
    allocated_anchor_report: Mapping[str, Any],
) -> None:
    if artifact.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("profile-aware challenger schema mismatch")
    unsigned = dict(artifact)
    expected = unsigned.pop("report_sha256", None)
    if not isinstance(expected, str) or expected != sha256_json(unsigned):
        raise ValueError("profile-aware challenger checksum mismatch")
    if (
        artifact.get("gpu_family") != "H800"
        or artifact.get("publishable") is not False
        or artifact.get("production_override_allowed") is not False
        or artifact.get("gpu_experiments_launched") is not False
    ):
        raise ValueError("profile-aware challenger release contract drifted")
    anchor = (artifact.get("model") or {}).get("allocated_anchor") or {}
    if (
        anchor.get("source_artifact_sha256") != sha256_file(allocated_anchor_path)
        or anchor.get("source_report_sha256")
        != allocated_anchor_report.get("report_sha256")
    ):
        raise ValueError("profile-aware challenger allocated anchor drifted")
    if not path.is_file():
        raise FileNotFoundError(path)


class H800ProfileAwareV4BPredictor(H800PhysicalV4BPredictor):
    """Use the profile-aware memory gate before the unchanged v4b ranker."""

    def __init__(
        self,
        *,
        challenger_artifact: Path = DEFAULT_CHALLENGER,
        memory_artifact: Path = DEFAULT_MEMORY_ARTIFACT,
        throughput_artifact: Path = DEFAULT_THROUGHPUT_ARTIFACT,
        memory_anchor_registry: Path = DEFAULT_MEMORY_ANCHOR_REGISTRY,
        additional_dataset_profile_dir: Path | None = None,
    ) -> None:
        super().__init__(
            memory_artifact=memory_artifact,
            throughput_artifact=throughput_artifact,
            memory_anchor_registry=memory_anchor_registry,
            additional_dataset_profile_dir=additional_dataset_profile_dir,
        )
        self.challenger_artifact_path = Path(challenger_artifact)
        self.challenger = read_json(self.challenger_artifact_path)
        _validate_challenger(
            self.challenger,
            path=self.challenger_artifact_path,
            allocated_anchor_path=self.memory_artifact_path,
            allocated_anchor_report=self.memory_report,
        )
        self.allocated_anchor = self.memory_report["memory"]["frozen_model"][
            "allocated_center_diagnostic"
        ]

    def _memory_result(
        self,
        record: Mapping[str, Any],
        support: Mapping[str, Any],
    ) -> dict[str, Any]:
        profile_path = Path(record["dataset_profile_binding"]["path"])
        scenario = record["scenario"]
        padding = profile_padding_statistics(
            profile_path,
            cutoff_len=int(scenario["cutoff_len"]),
            physical_mbs=int(scenario["physical_mbs"]),
        )
        allocated_anchor = _predict_memory_center(record, self.allocated_anchor)
        prediction = predict_memory(
            record,
            padding,
            allocated_anchor_bytes=allocated_anchor,
            artifact=self.challenger,
        )
        available = prediction.get("available") is True
        safe_limit = float(record["memory"]["safe_limit_bytes"])
        upper = prediction.get("operational_upper_reserved_bytes")
        base_admitted = bool(available and float(upper) <= safe_limit)
        policy_admitted = bool(base_admitted and support.get("label") != "unsupported")
        if not available:
            rejection = "memory_prediction_unavailable"
        elif not base_admitted:
            rejection = "memory_upper_exceeds_safe_limit"
        elif support.get("label") == "unsupported":
            rejection = "outside_supported_domain"
        else:
            rejection = None
        center = prediction.get("reserved_center_bytes")
        admission_upper = float(upper) if upper is not None else None
        headroom = safe_limit - admission_upper if admission_upper is not None else None
        reference = float(record["memory"]["analytic_reference_bytes"])
        return {
            "prediction_available": available,
            "analytic_reference_bytes": reference,
            "allocated_anchor_bytes": prediction.get("allocated_anchor_bytes"),
            "allocated_center_bytes": prediction.get("allocated_center_bytes"),
            "reserved_center_bytes": center,
            # Keep the base predictor field name for its already-tested
            # fail-closed validation; the challenger artifact states that this
            # is an empirical operational guard, not an identified P95.
            "operational_p95_reserved_bytes": upper,
            "operational_upper_semantics": "selector_mbs_LOPO_empirical_guard",
            "admission_upper_reserved_bytes": admission_upper,
            "safe_limit_bytes": safe_limit,
            "headroom_to_safe_limit_bytes": headroom,
            "base_physical_model_admitted": base_admitted,
            "anchor_override_applied": False,
            "admission_source": (
                "physical_model_operational_p95" if base_admitted else "rejected"
            ),
            "physical_model_admitted": base_admitted,
            "admitted": policy_admitted,
            "rejection_reason": rejection,
            "tail_source": prediction.get("tail_source"),
            "tail_bucket": prediction.get("tail_bucket"),
            "operational_log_guard": prediction.get("operational_log_guard"),
            "profile_guard_log_residual": prediction.get(
                "profile_guard_log_residual"
            ),
            "legacy_oom_guard_log_residual": prediction.get(
                "legacy_oom_guard_log_residual"
            ),
            "allocated_profile_log_correction": prediction.get(
                "allocated_profile_log_correction"
            ),
            "reservation_ratio_log": prediction.get("reservation_ratio_log"),
            "padding_statistics": padding,
            "issues": prediction.get("issues") or [],
            "historical_anchor": {
                "matched": False,
                "override_allowed": False,
                "reason": "disabled_for_prospective_challenger_acceptance",
            },
            "gib": {
                "analytic_reference": _gib(reference),
                "allocated_anchor": _gib(prediction.get("allocated_anchor_bytes")),
                "allocated_center": _gib(prediction.get("allocated_center_bytes")),
                "reserved_center": _gib(center),
                "operational_upper": _gib(upper),
                "safe_limit": _gib(safe_limit),
                "headroom_to_safe_limit": _gib(headroom),
            },
        }

    def predict(self, requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        base = super().predict(requests)
        base_report_sha256 = str(base["report_sha256"])
        report = dict(base)
        report["schema"] = SCHEMA
        report["implementation_version"] = IMPLEMENTATION_VERSION
        report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["base_predictor_contract_validation_passed"] = True
        report["base_predictor_report_sha256_before_wrapping"] = base_report_sha256
        report["release"] = {
            **dict(report["release"]),
            "mode": "shadow_only",
            "automatic_execution_allowed": False,
            "reason": "profile-aware memory challenger requires frozen unseen holdout",
            "memory_publishable": False,
        }
        report["model_artifacts"] = {
            **dict(report["model_artifacts"]),
            "profile_aware_memory_challenger": {
                "path": str(self.challenger_artifact_path.resolve()),
                "sha256": sha256_file(self.challenger_artifact_path),
                "report_sha256": self.challenger["report_sha256"],
            },
        }
        report["policy"] = (
            "profile-aware physical allocated correction plus reservation ratio "
            "and selector-MBS operational guard; no historical-anchor override; "
            "unchanged frozen v4b ranking for admitted candidates"
        )
        report["absolute_throughput_scale_trusted"] = False
        report["gpu_experiments_launched"] = False
        report["queues_mutated"] = False
        report.pop("report_sha256", None)
        report["report_sha256"] = sha256_json(report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--challenger", type=Path, default=DEFAULT_CHALLENGER)
    parser.add_argument("--memory-artifact", type=Path, default=DEFAULT_MEMORY_ARTIFACT)
    parser.add_argument(
        "--throughput-artifact", type=Path, default=DEFAULT_THROUGHPUT_ARTIFACT
    )
    parser.add_argument(
        "--memory-anchor-registry",
        type=Path,
        default=DEFAULT_MEMORY_ANCHOR_REGISTRY,
    )
    parser.add_argument("--additional-dataset-profile-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    predictor = H800ProfileAwareV4BPredictor(
        challenger_artifact=args.challenger,
        memory_artifact=args.memory_artifact,
        throughput_artifact=args.throughput_artifact,
        memory_anchor_registry=args.memory_anchor_registry,
        additional_dataset_profile_dir=args.additional_dataset_profile_dir,
    )
    report = predictor.predict(_load_requests(args.input))
    if args.output is not None:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
