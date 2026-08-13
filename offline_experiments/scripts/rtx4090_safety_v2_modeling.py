#!/usr/bin/env python3
"""Freeze a higher-recall RTX 4090 memory-safety head over physical-shares.

The physical-shares center and the v4b throughput model are inherited byte for
byte from the 2026-07-29 v1 artifact.  Only the admission boundary changes:

* fit a shallow random-forest unsafe classifier on the old memory probes;
* calibrate a strict zero-observed-unsafe threshold on throughput-screen rows;
* use the later formal population as a promotion check, never as fit data;
* retain the conservative v1 P95 gate for packing, which is absent from fit.

An unsafe row is either an OOM or a successful run whose observed reserved
memory exceeds the configured 95% capacity line.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from common import ROOT, read_json, sha256_file, sha256_json, write_json
from h800_challenger_modeling import (
    _memory_features,
    _predict_memory_center,
    _predict_memory_upper,
)
from h800_theory_calibration import _observed_reserved, _safe_limit
from rtx4090_physical_v4b_modeling import (
    DEFAULT_CAMPAIGN_ROOT,
    DEFAULT_OUTPUT as V1_ARTIFACT,
    _build_records,
    _read_rows,
)
from throughput_predictor import ThroughputPredictor


SCHEMA = "sft_rtx4090_physical_shares_conditional_safety_v2/v1"
IMPLEMENTATION_VERSION = (
    "sft_rtx4090_physical_shares_conditional_safety_v2/"
    "2026-07-29.shallow-rf-screen-calibrated"
)
DEFAULT_OUTPUT = (
    DEFAULT_CAMPAIGN_ROOT
    / "artifacts"
    / "rtx4090_physical_shares_v4b_safety_v2_2026-07-29.json"
)
SAFETY_FEATURE_NAMES = (
    "physical_shares::28",
    "log_center_over_safe_limit",
    "log_center_over_analytic_reference",
    "center_over_safe_limit",
)
RANDOM_STATE = 29
CANDIDATE_DEPTHS = (2, 3)
CANDIDATE_MIN_LEAVES = (10, 20, 30)
CANDIDATE_ESTIMATORS = (64, 96)


def _populations(
    campaign_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    rows = _read_rows(campaign_root)
    predictor = ThroughputPredictor(strict_bindings=False)

    def build(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return _build_records(
            predictor=predictor,
            campaign_root=campaign_root,
            rows=selected,
            require_throughput=False,
        )

    return {
        "old_fit": build(
            [row for row in rows if row.get("kind") == "memory_probe"]
        ),
        "screen_calibration": build(
            [
                row
                for row in rows
                if row.get("kind") == "throughput_screen"
            ]
        ),
        "formal_promotion": build(
            [
                row
                for row in rows
                if row.get("kind") == "throughput"
                and row.get("fidelity") == "formal"
            ]
        ),
        "packing_fallback_test": build(
            [
                row
                for row in rows
                if bool(row.get("packing"))
                and row.get("kind")
                in {"packing_memory_probe", "throughput"}
            ]
        ),
    }


def _unsafe(record: Mapping[str, Any]) -> bool:
    if str(record["outcome"]) == "oom":
        return True
    observed = _observed_reserved(record)
    limit = _safe_limit(record)
    return bool(
        observed is not None
        and limit is not None
        and float(observed) > float(limit)
    )


def _feature_vector(
    record: Mapping[str, Any],
    center_model: Mapping[str, Any],
) -> np.ndarray:
    center = _predict_memory_center(record, center_model)
    limit = float(_safe_limit(record) or 0.0)
    reference = float(record["memory"]["analytic_reference_bytes"])
    if center <= 0 or limit <= 0 or reference <= 0:
        raise ValueError("Safety features require positive memory quantities")
    return np.asarray(
        [
            *_memory_features(record, "physical_shares").tolist(),
            math.log(center / limit),
            math.log(center / reference),
            center / limit,
        ],
        dtype=float,
    )


def _matrix(
    records: Sequence[Mapping[str, Any]],
    center_model: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.vstack(
            [_feature_vector(record, center_model) for record in records]
        ),
        np.asarray([int(_unsafe(record)) for record in records], dtype=int),
    )


def _fit_forest(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: int,
    min_samples_leaf: int,
    n_estimators: int,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        bootstrap=False,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(x, y)
    if list(model.classes_) != [0, 1]:
        raise ValueError("Safety fit must contain both safe and unsafe rows")
    return model


def _calibrated_threshold(
    model: RandomForestClassifier,
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    unsafe_scores = model.predict_proba(x)[:, 1][y == 1]
    if not len(unsafe_scores):
        raise ValueError("Safety calibration contains no unsafe rows")
    # Admission is strict score < threshold.  Equal-score rows are rejected.
    return float(np.min(unsafe_scores))


def _decision_metrics(
    records: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    threshold: float,
    center_model: Mapping[str, Any],
    legacy_tail: Mapping[str, Any],
    packing_fallback: bool,
) -> dict[str, Any]:
    safe_success = admitted_safe = 0
    oom = admitted_oom = 0
    unsafe_success = admitted_unsafe_success = 0
    details = []
    for record, score in zip(records, scores, strict=True):
        center = _predict_memory_center(record, center_model)
        limit = float(_safe_limit(record) or 0.0)
        if packing_fallback and bool(record["selector"]["packing"]):
            legacy = _predict_memory_upper(
                record,
                center_model,
                legacy_tail,
            )
            admitted = bool(
                legacy.get("available") is True
                and float(legacy["operational_p95_reserved_bytes"])
                <= limit
            )
            policy = "legacy_p95_unseen_packing_fallback"
        else:
            admitted = bool(center < limit and float(score) < threshold)
            policy = "conditional_unsafe_score"
        outcome = str(record["outcome"])
        observed = _observed_reserved(record)
        actual_safe = bool(
            outcome == "success"
            and observed is not None
            and float(observed) <= limit
        )
        if actual_safe:
            safe_success += 1
            admitted_safe += int(admitted)
        elif outcome == "oom":
            oom += 1
            admitted_oom += int(admitted)
        else:
            unsafe_success += 1
            admitted_unsafe_success += int(admitted)
        details.append(
            {
                "observation_id": str(record["observation_id"]),
                "outcome": outcome,
                "actual_safe_success": actual_safe,
                "unsafe_score": float(score),
                "threshold": threshold,
                "admitted": admitted,
                "policy": policy,
                "center_over_safe_limit": center / limit,
            }
        )
    return {
        "rows": len(records),
        "actual_safe_success_rows": safe_success,
        "admitted_safe_success_rows": admitted_safe,
        "safe_success_admission_recall": (
            admitted_safe / safe_success if safe_success else None
        ),
        "oom_rows": oom,
        "false_safe_oom": admitted_oom,
        "unsafe_success_rows": unsafe_success,
        "admitted_unsafe_success_rows": admitted_unsafe_success,
        "all_observed_unsafe_rows": oom + unsafe_success,
        "admitted_observed_unsafe_rows": (
            admitted_oom + admitted_unsafe_success
        ),
        "details": details,
    }


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key != "details"
    }


def _serialize_forest(
    model: RandomForestClassifier,
) -> dict[str, Any]:
    trees = []
    unsafe_class_index = list(model.classes_).index(1)
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :]
        probabilities = values[:, unsafe_class_index] / np.maximum(
            values.sum(axis=1),
            1.0e-30,
        )
        trees.append(
            {
                "children_left": tree.children_left.astype(int).tolist(),
                "children_right": tree.children_right.astype(int).tolist(),
                "feature": tree.feature.astype(int).tolist(),
                "threshold": tree.threshold.astype(float).tolist(),
                "unsafe_probability": probabilities.astype(float).tolist(),
            }
        )
    return {
        "model_family": "shallow_random_forest_binary_unsafe",
        "feature_dimension": int(model.n_features_in_),
        "feature_contract": [
            *read_json(V1_ARTIFACT)["memory"]["frozen_model"]["center"][
                "feature_names"
            ],
            "log_center_over_safe_limit",
            "log_center_over_analytic_reference",
            "center_over_safe_limit",
        ],
        "classes": [0, 1],
        "unsafe_class": 1,
        "n_estimators": len(trees),
        "max_depth": model.max_depth,
        "min_samples_leaf": model.min_samples_leaf,
        "max_features": model.max_features,
        "bootstrap": model.bootstrap,
        "class_weight": model.class_weight,
        "random_state": model.random_state,
        "trees": trees,
    }


def _serialized_scores(
    x: np.ndarray,
    forest: Mapping[str, Any],
) -> np.ndarray:
    totals = np.zeros(len(x), dtype=float)
    for tree in forest["trees"]:
        left = tree["children_left"]
        right = tree["children_right"]
        features = tree["feature"]
        thresholds = tree["threshold"]
        probabilities = tree["unsafe_probability"]
        for row_index, row in enumerate(x):
            node = 0
            while int(left[node]) != int(right[node]):
                node = (
                    int(left[node])
                    if float(row[int(features[node])])
                    <= float(thresholds[node])
                    else int(right[node])
                )
            totals[row_index] += float(probabilities[node])
    return totals / len(forest["trees"])


def build_report(
    campaign_root: Path,
    v1_artifact: Path,
) -> dict[str, Any]:
    v1 = read_json(v1_artifact)
    unsigned_v1 = dict(v1)
    expected_v1_digest = unsigned_v1.pop("report_sha256", None)
    if expected_v1_digest != sha256_json(unsigned_v1):
        raise ValueError("RTX 4090 v1 artifact checksum mismatch")
    center = v1["memory"]["frozen_model"]["center"]
    legacy_tail = v1["memory"]["frozen_model"]["tail"]
    populations = _populations(campaign_root)
    matrices = {
        name: _matrix(records, center)
        for name, records in populations.items()
    }
    old_x, old_y = matrices["old_fit"]
    screen_x, screen_y = matrices["screen_calibration"]

    candidates = []
    fitted: dict[tuple[int, int, int], RandomForestClassifier] = {}
    for depth in CANDIDATE_DEPTHS:
        for leaf in CANDIDATE_MIN_LEAVES:
            for estimators in CANDIDATE_ESTIMATORS:
                model = _fit_forest(
                    old_x,
                    old_y,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    n_estimators=estimators,
                )
                threshold = _calibrated_threshold(
                    model,
                    screen_x,
                    screen_y,
                )
                old_scores = model.predict_proba(old_x)[:, 1]
                screen_scores = model.predict_proba(screen_x)[:, 1]
                old_metrics = _decision_metrics(
                    populations["old_fit"],
                    old_scores,
                    threshold=threshold,
                    center_model=center,
                    legacy_tail=legacy_tail,
                    packing_fallback=False,
                )
                screen_metrics = _decision_metrics(
                    populations["screen_calibration"],
                    screen_scores,
                    threshold=threshold,
                    center_model=center,
                    legacy_tail=legacy_tail,
                    packing_fallback=False,
                )
                settings = (depth, leaf, estimators)
                fitted[settings] = model
                candidates.append(
                    {
                        "max_depth": depth,
                        "min_samples_leaf": leaf,
                        "n_estimators": estimators,
                        "threshold": threshold,
                        "old_fit_diagnostic": _compact_metrics(old_metrics),
                        "screen_calibration": _compact_metrics(
                            screen_metrics
                        ),
                    }
                )

    eligible = [
        candidate
        for candidate in candidates
        if candidate["old_fit_diagnostic"][
            "admitted_observed_unsafe_rows"
        ]
        == 0
        and candidate["screen_calibration"][
            "admitted_observed_unsafe_rows"
        ]
        == 0
    ]
    if not eligible:
        raise ValueError("No zero-observed-unsafe safety candidate")
    selected = min(
        eligible,
        key=lambda candidate: (
            -float(
                candidate["screen_calibration"][
                    "safe_success_admission_recall"
                ]
            ),
            -float(
                candidate["old_fit_diagnostic"][
                    "safe_success_admission_recall"
                ]
            ),
            int(candidate["max_depth"]),
            -int(candidate["min_samples_leaf"]),
            int(candidate["n_estimators"]),
        ),
    )
    settings = (
        int(selected["max_depth"]),
        int(selected["min_samples_leaf"]),
        int(selected["n_estimators"]),
    )
    model = fitted[settings]
    threshold = float(selected["threshold"])
    serialized = _serialize_forest(model)

    evaluations = {}
    for name, records in populations.items():
        x, _ = matrices[name]
        sklearn_scores = model.predict_proba(x)[:, 1]
        frozen_scores = _serialized_scores(x, serialized)
        if not np.allclose(
            sklearn_scores,
            frozen_scores,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("Serialized forest does not reproduce sklearn")
        evaluations[name] = _decision_metrics(
            records,
            frozen_scores,
            threshold=threshold,
            center_model=center,
            legacy_tail=legacy_tail,
            packing_fallback=True,
        )

    formal = evaluations["formal_promotion"]
    if formal["admitted_observed_unsafe_rows"] != 0:
        raise ValueError("Selected safety head failed formal promotion")
    if evaluations["packing_fallback_test"][
        "admitted_observed_unsafe_rows"
    ] != 0:
        raise ValueError("Packing fallback admitted an unsafe row")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_id": "rtx4090",
        "gpu_family": "NVIDIA GeForce RTX 4090",
        "freeze_status": "frozen_before_prospective_generalization",
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "fit_contract": {
            "physical_center": "inherited byte-for-byte from v1",
            "safety_fit": "old kind=memory_probe only",
            "threshold_calibration": (
                "strictly below the minimum observed-unsafe score in "
                "throughput_screen"
            ),
            "candidate_selection": (
                "maximum screen safe-success recall subject to zero admitted "
                "observed-unsafe rows in old fit diagnostics and screen"
            ),
            "formal_role": "promotion check only; not fit or calibration",
            "packing_policy": (
                "v1 conservative P95 fallback because packing is absent "
                "from safety-head fit"
            ),
            "unsafe_definition": (
                "OOM or successful reserved memory above 95% capacity"
            ),
        },
        "selection": {
            "candidate_count": len(candidates),
            "selected": selected,
            "candidates": candidates,
        },
        "memory": {
            "center": center,
            "legacy_tail_for_packing": legacy_tail,
            "conditional_safety_head": {
                **serialized,
                "admission_threshold": threshold,
                "comparison": "unsafe_score < admission_threshold",
                "hard_center_guard": "reserved_center < safe_limit",
                "packing_fallback": "legacy physical-shares operational P95",
                "training_rows": len(populations["old_fit"]),
                "calibration_rows": len(
                    populations["screen_calibration"]
                ),
                "training_observation_ids_sha256": sha256_json(
                    sorted(
                        str(record["observation_id"])
                        for record in populations["old_fit"]
                    )
                ),
                "calibration_observation_ids_sha256": sha256_json(
                    sorted(
                        str(record["observation_id"])
                        for record in populations["screen_calibration"]
                    )
                ),
            },
            "evaluation": evaluations,
            "v1_reference": {
                "old_nested_cv": v1["memory"][
                    "old_experiment_nested_cv"
                ]["metrics"],
                "temporal_generalization": v1["memory"][
                    "temporal_generalization"
                ],
            },
        },
        "throughput": {
            "model_contract": v1["throughput"]["model_contract"],
            "frozen_model": v1["throughput"]["frozen_model"],
            "source_report_sha256": v1["report_sha256"],
        },
        "source_bindings": {
            "v1_artifact": {
                "path": str(v1_artifact.resolve()),
                "sha256": sha256_file(v1_artifact),
                "report_sha256": v1["report_sha256"],
            },
            "collected_results": v1["source_bindings"][
                "collected_results"
            ],
            "hardware": v1["source_bindings"]["hardware"],
            "static_dataset_profiles": v1["source_bindings"][
                "static_dataset_profiles"
            ],
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "sklearn_version": __import__("sklearn").__version__,
        },
        "limitations": [
            (
                "The screen set is used for threshold calibration and model "
                "selection, so its recall is a development metric."
            ),
            (
                "The formal population is a same-configuration temporal "
                "promotion check, not unseen-model generalization."
            ),
            (
                "The conditional head has no packing fit evidence; packing "
                "therefore remains on the conservative v1 gate."
            ),
            (
                "Qwen3 >4B, Qwen3.5 and other model families remain prospective "
                "tests and must not be claimed as validated by this artifact."
            ),
        ],
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT,
    )
    parser.add_argument(
        "--v1-artifact",
        type=Path,
        default=V1_ARTIFACT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        args.campaign_root.resolve(),
        args.v1_artifact.resolve(),
    )
    write_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "report_sha256": report["report_sha256"],
                "selected": report["selection"]["selected"],
                "evaluation": {
                    name: _compact_metrics(metrics)
                    for name, metrics in report["memory"][
                        "evaluation"
                    ].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
