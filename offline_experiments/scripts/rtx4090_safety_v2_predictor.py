#!/usr/bin/env python3
"""Unified RTX 4090 physical-shares safety-v2 + v4b predictor."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from common import ROOT, read_json, sha256_file, sha256_json, write_json
from h800_challenger_modeling import (
    _memory_features,
    _predict_memory_center,
    _predict_memory_upper,
)
from rtx4090_physical_v4b_predictor import (
    RTX4090PhysicalV4BPredictor,
    _load_requests,
)
from throughput_predictor import ThroughputPredictor


SCHEMA = "sft_rtx4090_physical_shares_v4b_safety_v2_prediction/v1"
MODEL_SCHEMA = "sft_rtx4090_physical_shares_conditional_safety_v2/v1"
DEFAULT_MODEL_ARTIFACT = (
    ROOT
    / "campaigns"
    / "rtx4090_20260717"
    / "artifacts"
    / "rtx4090_physical_shares_v4b_safety_v2_2026-07-29.json"
)
IMPLEMENTATION_VERSION = (
    "sft_rtx4090_physical_shares_v4b_safety_v2_predictor/"
    "2026-07-29.v1"
)


def _validate_model(report: Mapping[str, Any]) -> None:
    if report.get("schema") != MODEL_SCHEMA:
        raise ValueError("RTX 4090 safety-v2 artifact schema mismatch")
    unsigned = dict(report)
    digest = unsigned.pop("report_sha256", None)
    if digest != sha256_json(unsigned):
        raise ValueError("RTX 4090 safety-v2 artifact checksum mismatch")
    if (
        report.get("freeze_status")
        != "frozen_before_prospective_generalization"
        or report.get("gpu_experiments_launched") is not False
        or report.get("queues_mutated") is not False
    ):
        raise ValueError("RTX 4090 prospective freeze contract drifted")
    memory = report.get("memory") or {}
    safety = memory.get("conditional_safety_head") or {}
    throughput = report.get("throughput") or {}
    if (
        (memory.get("center") or {}).get("feature_set")
        != "physical_shares"
        or safety.get("model_family")
        != "shallow_random_forest_binary_unsafe"
        or safety.get("feature_dimension") != 31
        or (throughput.get("frozen_model") or {}).get("model_family")
        != "set_aware_two_head_log_throughput_model"
    ):
        raise ValueError("RTX 4090 safety-v2 model contract drifted")


def _forest_score(
    features: np.ndarray,
    forest: Mapping[str, Any],
) -> float:
    total = 0.0
    trees = forest.get("trees") or []
    if not trees:
        raise ValueError("Frozen safety forest has no trees")
    for tree in trees:
        left = tree["children_left"]
        right = tree["children_right"]
        split_features = tree["feature"]
        thresholds = tree["threshold"]
        probabilities = tree["unsafe_probability"]
        node = 0
        while int(left[node]) != int(right[node]):
            node = (
                int(left[node])
                if float(features[int(split_features[node])])
                <= float(thresholds[node])
                else int(right[node])
            )
        total += float(probabilities[node])
    return total / len(trees)


class RTX4090SafetyV2Predictor(RTX4090PhysicalV4BPredictor):
    """Filter with conditional physical safety, then rank with frozen v4b."""

    def __init__(
        self,
        *,
        model_artifact: Path = DEFAULT_MODEL_ARTIFACT,
    ) -> None:
        self.model_artifact_path = Path(model_artifact)
        self.report = read_json(self.model_artifact_path)
        _validate_model(self.report)
        memory = self.report["memory"]
        self.memory_center = memory["center"]
        self.memory_tail = memory["legacy_tail_for_packing"]
        self.safety_head = memory["conditional_safety_head"]
        self.throughput_model = self.report["throughput"]["frozen_model"]
        self.base = ThroughputPredictor(strict_bindings=False)

        expected_profiles = self.report["source_bindings"][
            "static_dataset_profiles"
        ]
        mismatches = []
        for dataset_id, binding in expected_profiles.items():
            path = (
                self.base.dataset_profile_dir
                / f"{dataset_id}.qwen3_nothink.jsonl"
            )
            if (
                not path.is_file()
                or sha256_file(path) != binding.get("sha256")
            ):
                mismatches.append(dataset_id)
        if mismatches:
            raise ValueError(
                "Frozen dataset-profile bindings changed: "
                + ", ".join(sorted(mismatches))
            )

    def _memory_result(
        self,
        record: Mapping[str, Any],
    ) -> dict[str, Any]:
        center = _predict_memory_center(record, self.memory_center)
        safe_limit = float(record["memory"]["safe_limit_bytes"])
        packing = bool(record["selector"]["packing"])
        if packing:
            legacy = _predict_memory_upper(
                record,
                self.memory_center,
                self.memory_tail,
            )
            available = legacy.get("available") is True
            admitted = bool(
                available
                and float(legacy["operational_p95_reserved_bytes"])
                <= safe_limit
            )
            return {
                "prediction_available": available,
                "reserved_center_bytes": center,
                "operational_p95_reserved_bytes": legacy.get(
                    "operational_p95_reserved_bytes"
                ),
                "safe_limit_bytes": safe_limit,
                "admitted": admitted,
                "admission_policy": (
                    "legacy_p95_unseen_packing_fallback"
                ),
                "unsafe_score": None,
                "unsafe_score_threshold": None,
                "tail_source": legacy.get("success_tail_source"),
                "issues": legacy.get("issues") or [],
            }

        reference = float(record["memory"]["analytic_reference_bytes"])
        features = np.asarray(
            [
                *_memory_features(
                    record,
                    "physical_shares",
                ).tolist(),
                math.log(center / safe_limit),
                math.log(center / reference),
                center / safe_limit,
            ],
            dtype=float,
        )
        score = _forest_score(features, self.safety_head)
        threshold = float(self.safety_head["admission_threshold"])
        admitted = bool(center < safe_limit and score < threshold)
        return {
            "prediction_available": True,
            "reserved_center_bytes": center,
            "operational_p95_reserved_bytes": None,
            "safe_limit_bytes": safe_limit,
            "admitted": admitted,
            "admission_policy": "conditional_unsafe_score",
            "unsafe_score": score,
            "unsafe_score_threshold": threshold,
            "hard_center_guard_passed": center < safe_limit,
            "tail_source": None,
            "issues": [],
        }

    def predict(
        self,
        requests: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        report = super().predict(requests)
        report.pop("report_sha256", None)
        report.update(
            {
                "schema": SCHEMA,
                "implementation_version": IMPLEMENTATION_VERSION,
                "policy": (
                    "physical-shares center plus calibrated conditional "
                    "unsafe gate (legacy P95 for packing), then frozen v4b "
                    "candidate-set ranking"
                ),
            }
        )
        report["report_sha256"] = sha256_json(report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=DEFAULT_MODEL_ARTIFACT,
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = RTX4090SafetyV2Predictor(
        model_artifact=args.model_artifact
    ).predict(_load_requests(args.input))
    if args.output is not None:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
