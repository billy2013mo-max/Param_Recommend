#!/usr/bin/env python3
"""Fit rank-first challenger v2: add ZeRO/GC x MBS interaction terms.

Challenger v1 left one mechanism pair near chance level.  On the leave-one-
extension-scenario-out replay its ``Z2/GC-on vs Z3/GC-off`` ordering was 3/6 and
``Z2/GC-off vs Z3/GC-off`` was 9/12, while ``Z2/GC-off vs Z2/GC-on`` reached
47/48.  The failures were not near-ties: several mis-ordered pairs differed by
more than 40% in measured throughput.  The diagnosis recorded at the time was a
missing interaction between the sharding stage and the micro-batch size --- v1
carries ``zero3``, ``log2_mbs`` and ``zero3_x_log2_gpu_count`` but no
``zero3_x_log2_mbs``, so it cannot express that raising MBS pays off differently
under ZeRO-3 than under ZeRO-2.

v2 changes exactly one contract relative to v1: three interaction features are
appended.  Work reconstruction, dynamic padding, cutoff invariance, head
separation, the IRLS ridge, the alpha grid, the selection rules and every
evaluation definition are imported unchanged from v1 so that the comparison
isolates the feature change.

Identifiability was checked before fitting (see
``diagnose_zero_mbs_interaction_identifiability_v1.py``).  In the strict-fit
population ZeRO-3 spans MBS 1/2/4/8/16 across 53 candidates and each new term
retains 43--48% of its variation after projecting out the existing 44 features
(VIF 4.4--5.4), so the terms are estimable rather than collinear relabelings.

The script is offline and diagnostic.  It launches no GPU work, mutates no
queue, publishes no production profile, and does not modify the frozen V5
artifact or the v1 challenger artifact.  The consumed V5 dataset-extension
outcomes are used exactly as v1 used them and are never called a fresh holdout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import ROOT, read_json, sha256_file, sha256_json, write_json  # noqa: E402
from fit_rank_first_throughput_challenger_v1 import (  # noqa: E402
    ALPHA_GRID,
    MATERIAL_GAP,
    MATERIAL_PAIR_WEIGHT,
    PAD_TO_MULTIPLE_OF,
    REMOVED_CUTOFF_FEATURES,
    InvariantStaticProfiles,
    _evaluate_entries,
    _invariant_basis,
    _load_extension_candidates,
    _scenario_folds,
    invariant_profile_signature,
)
from fit_rank_first_throughput_challenger_v1 import (  # noqa: E402
    FEATURE_NAMES as V1_FEATURE_NAMES,
)
from structured_throughput_modeling import _load_h800  # noqa: E402

SCHEMA = "sft_rank_first_throughput_challenger/v2"
IMPLEMENTATION_VERSION = (
    "sft_rank_first_throughput_challenger_impl/"
    "2026-08-05.invariant-nonpacking-two-head-sharding-mbs-interactions-v2"
)

INTERACTION_FEATURE_NAMES = (
    "zero2_x_log2_mbs",
    "zero3_x_log2_mbs",
    "gc_x_log2_mbs",
)
FEATURE_NAMES = V1_FEATURE_NAMES + INTERACTION_FEATURE_NAMES

_V1_INDEX = {name: index for index, name in enumerate(V1_FEATURE_NAMES)}


def _interaction_values(v1_values: Mapping[str, float]) -> dict[str, float]:
    """Derive the new terms from v1 feature values only.

    Reusing ``zero2``/``zero3``/``gradient_checkpointing``/``log2_mbs`` as they
    were already computed keeps v2 consistent with v1 by construction: if v1
    changes how a base feature is derived, v2 follows automatically instead of
    silently disagreeing.
    """

    log2_mbs = float(v1_values["log2_mbs"])
    return {
        "zero2_x_log2_mbs": float(v1_values["zero2"]) * log2_mbs,
        "zero3_x_log2_mbs": float(v1_values["zero3"]) * log2_mbs,
        "gc_x_log2_mbs": float(v1_values["gradient_checkpointing"]) * log2_mbs,
    }


def _extend_features(basis: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(basis["invariant_feature_values"])
    values.update(_interaction_values(values))
    features = np.asarray([values[name] for name in FEATURE_NAMES], dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("v2 challenger features contain NaN/Inf")
    expected = np.asarray(basis["invariant_features"], dtype=float)
    if not np.allclose(features[: len(V1_FEATURE_NAMES)], expected, rtol=0.0, atol=0.0):
        raise ValueError("v2 must preserve the v1 feature prefix exactly")
    return {
        **basis,
        "v2_feature_values": values,
        "v2_features": features,
    }


def _v2_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Attach v2 features to a candidate produced by the v1 loaders."""

    basis = _extend_features(candidate["basis"])
    return {
        **candidate,
        "basis": basis,
        "features_v1": np.asarray(candidate["features"], dtype=float),
        "features": basis["v2_features"],
    }


def _candidate_from_pooled_v2(
    pooled: Mapping[str, Any],
    profiles: InvariantStaticProfiles,
    *,
    hardware_memory_bytes: float,
    source_role: str,
) -> dict[str, Any]:
    basis = _extend_features(
        _invariant_basis(
            pooled["record"],
            profiles,
            hardware_memory_bytes=hardware_memory_bytes,
        )
    )
    return {
        "scenario_id": str(pooled["scenario_id"]),
        "record": pooled["record"],
        "candidate_key": pooled.get("candidate_key"),
        "job_id": None,
        "source_role": source_role,
        "historical": bool(pooled.get("historical")),
        "features": basis["v2_features"],
        "features_v1": np.asarray(
            [
                basis["invariant_feature_values"][name]
                for name in V1_FEATURE_NAMES
            ],
            dtype=float,
        ),
        "basis": basis,
        "observed_log_throughput": float(pooled["observed_log_throughput"]),
    }


def _with_v1_features(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the same rows carrying v1 features, for the ablation arm."""

    return [
        {**row, "features": np.asarray(row["features_v1"], dtype=float)}
        for row in candidates
    ]


def _counterfactual_cutoff_audit_v2(
    template: Mapping[str, Any],
    profiles: InvariantStaticProfiles,
    absolute_head: Mapping[str, Any],
    rank_head: Mapping[str, Any],
    *,
    hardware_memory_bytes: float,
    linear_score: Any,
) -> dict[str, Any]:
    """Re-run the v1 invariance contract against the v2 feature vector.

    The new terms depend only on the sharding stage, checkpointing flag and
    micro-batch size, so they must not reintroduce cutoff sensitivity.  This
    asserts that rather than assuming it.
    """

    record = json.loads(json.dumps(template["record"]))
    dataset_id = str(record["scenario"]["dataset_id"])
    raw_maximum = max(
        int(row.get("total_tokens") or 0) for row in profiles.rows[dataset_id]
    )
    cutoffs = [512, 1024, 2048, 4096]
    cutoffs = [value for value in cutoffs if value >= raw_maximum] or [
        raw_maximum,
        raw_maximum * 2,
        raw_maximum * 4,
        raw_maximum * 8,
    ]

    signatures: list[dict[str, Any]] = []
    feature_vectors: list[list[float]] = []
    absolute_logs: list[float] = []
    rank_scores: list[float] = []
    for cutoff in cutoffs:
        current = json.loads(json.dumps(record))
        current["scenario"]["cutoff_len"] = int(cutoff)
        basis = _extend_features(
            _invariant_basis(
                current,
                profiles,
                hardware_memory_bytes=hardware_memory_bytes,
            )
        )
        candidate = {
            "scenario_id": "counterfactual",
            "record": current,
            "features": basis["v2_features"],
            "basis": basis,
            "observed_log_throughput": 0.0,
        }
        signatures.append(
            invariant_profile_signature(basis["work_evidence"]["profile"])
        )
        feature_vectors.append([float(value) for value in basis["v2_features"]])
        absolute_logs.append(linear_score(candidate, absolute_head))
        rank_scores.append(linear_score(candidate, rank_head))

    signatures_identical = all(
        signature == signatures[0] for signature in signatures[1:]
    )
    features_identical = all(
        np.allclose(np.asarray(vector), np.asarray(feature_vectors[0]), rtol=0.0, atol=0.0)
        for vector in feature_vectors[1:]
    )
    absolute_range = float(max(absolute_logs) - min(absolute_logs))
    rank_range = float(max(rank_scores) - min(rank_scores))
    return {
        "dataset_id": dataset_id,
        "profile_raw_maximum": raw_maximum,
        "cutoffs": cutoffs,
        "profile_signatures_identical": signatures_identical,
        "feature_vectors_identical": features_identical,
        "absolute_log_predictions": absolute_logs,
        "rank_scores": rank_scores,
        "absolute_prediction_range": absolute_range,
        "rank_score_range": rank_range,
        "interaction_terms_are_cutoff_free": features_identical,
        "passes": bool(
            signatures_identical
            and features_identical
            and absolute_range == 0.0
            and rank_range == 0.0
        ),
    }


def _coefficient_table(
    model: Mapping[str, Any],
    names: Sequence[str] = INTERACTION_FEATURE_NAMES,
) -> dict[str, float]:
    lookup = {
        name: float(value)
        for name, value in zip(model["feature_names"], model["coefficients"])
    }
    return {name: lookup[name] for name in names if name in lookup}


def _mechanism_pair_delta(
    v1_metrics: Mapping[str, Any],
    v2_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    keys = sorted(
        set(v1_metrics.get("mechanism_pair_breakdown") or {})
        | set(v2_metrics.get("mechanism_pair_breakdown") or {})
    )
    table: dict[str, Any] = {}
    for key in keys:
        before = (v1_metrics.get("mechanism_pair_breakdown") or {}).get(key) or {}
        after = (v2_metrics.get("mechanism_pair_breakdown") or {}).get(key) or {}
        table[key] = {
            "comparisons": after.get("comparisons", before.get("comparisons")),
            "v1_correct": before.get("correct"),
            "v2_correct": after.get("correct"),
            "v1_accuracy": before.get("accuracy"),
            "v2_accuracy": after.get("accuracy"),
            "accuracy_delta": (
                float(after["accuracy"]) - float(before["accuracy"])
                if after.get("accuracy") is not None
                and before.get("accuracy") is not None
                else None
            ),
            "v1_material_accuracy": before.get("material_accuracy"),
            "v2_material_accuracy": after.get("material_accuracy"),
        }
    return table


def _headline(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metrics.get(key)
        for key in (
            "candidate_rows",
            "scenario_rows",
            "all_pairwise_accuracy",
            "cross_mechanism_pairwise_comparisons",
            "cross_mechanism_pairwise_accuracy",
            "material_cross_mechanism_pairwise_accuracy",
            "scenario_equal_material_cross_mechanism_pairwise_accuracy",
            "mean_top1_regret",
            "worst_top1_regret",
            "exact_top1_fraction",
            "set_aware_absolute_mape",
            "absolute_head_mape",
        )
    }


def _acceptance(
    v1_metrics: Mapping[str, Any],
    v2_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Pre-registered decision rule for keeping the interaction terms.

    The user's release criterion is Top-1 regret below 10%, so regret must not
    get worse.  The purpose of this change is the weak mechanism pair, so at
    least one pair must improve and none may lose accuracy.
    """

    pairs = _mechanism_pair_delta(v1_metrics, v2_metrics)
    deltas = {
        key: value["accuracy_delta"]
        for key, value in pairs.items()
        if value["accuracy_delta"] is not None
    }
    improved = {key: value for key, value in deltas.items() if value > 1.0e-12}
    degraded = {key: value for key, value in deltas.items() if value < -1.0e-12}
    v1_worst = v1_metrics.get("worst_top1_regret")
    v2_worst = v2_metrics.get("worst_top1_regret")
    v1_mean = v1_metrics.get("mean_top1_regret")
    v2_mean = v2_metrics.get("mean_top1_regret")
    checks = {
        "some_mechanism_pair_improved": bool(improved),
        "no_mechanism_pair_degraded": not degraded,
        "mean_top1_regret_not_worse": (
            v2_mean is not None
            and v1_mean is not None
            and float(v2_mean) <= float(v1_mean) + 1.0e-12
        ),
        "worst_top1_regret_not_worse": (
            v2_worst is not None
            and v1_worst is not None
            and float(v2_worst) <= float(v1_worst) + 1.0e-12
        ),
        "worst_top1_regret_within_user_criterion": (
            v2_worst is not None and float(v2_worst) < 0.10
        ),
    }
    return {
        "rule": (
            "keep the interaction terms only if a weak mechanism pair improves, "
            "no pair degrades, and neither mean nor worst Top-1 regret worsens"
        ),
        "user_release_criterion": "worst_top1_regret < 0.10",
        "improved_mechanism_pairs": improved,
        "degraded_mechanism_pairs": degraded,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def build_report(
    *,
    profile_dirs: Sequence[Path],
    h800_observations_path: Path,
    h800_theory_basis_path: Path,
    h800_inventory_path: Path,
    h800_hardware_path: Path,
    h800_runtime_root: Path,
    extension_evaluation_path: Path,
    extension_predictions_path: Path,
    v5_artifact_path: Path,
    v1_artifact_path: Path,
) -> dict[str, Any]:
    import fit_rank_first_throughput_challenger_v1 as fitting

    profiles = InvariantStaticProfiles(profile_dirs)
    hardware = read_json(h800_hardware_path)
    hardware_memory_bytes = float(hardware["memory_bytes_reported_by_torch"])
    h800 = _load_h800(
        observation_path=h800_observations_path,
        theory_basis_path=h800_theory_basis_path,
        inventory_path=h800_inventory_path,
        hardware_path=h800_hardware_path,
        runtime_root=h800_runtime_root,
    )
    base = [
        _candidate_from_pooled_v2(
            candidate,
            profiles,
            hardware_memory_bytes=hardware_memory_bytes,
            source_role="h800_strict_fit",
        )
        for candidate in h800["strict_fit_candidates"]
    ]
    holdout = [
        _candidate_from_pooled_v2(
            candidate,
            profiles,
            hardware_memory_bytes=hardware_memory_bytes,
            source_role="h800_consumed_native_holdout",
        )
        for candidate in h800["holdout_candidates"]
    ]
    extension = [
        _v2_candidate(candidate)
        for candidate in _load_extension_candidates(
            extension_evaluation_path,
            extension_predictions_path,
            profiles,
            hardware_memory_bytes=hardware_memory_bytes,
        )
    ]

    folds = _scenario_folds(base)

    def _fit_arm(
        arm_base: Sequence[Mapping[str, Any]],
        arm_extension: Sequence[Mapping[str, Any]],
        arm_holdout: Sequence[Mapping[str, Any]],
        feature_names: Sequence[str],
    ) -> dict[str, Any]:
        """Run v1's full selection + fit + replay pipeline on one feature set.

        v1's fitting helpers read the module-level ``FEATURE_NAMES`` only to
        record it on the returned model, while the numeric work reads
        ``candidate["features"]``.  Rebinding it for the duration of an arm keeps
        each arm's recorded contract truthful without forking v1's solver.
        """

        original = fitting.FEATURE_NAMES
        fitting.FEATURE_NAMES = tuple(feature_names)
        try:
            width = len(feature_names)
            for population in (arm_base, arm_extension, arm_holdout):
                for row in population:
                    if len(np.asarray(row["features"], dtype=float)) != width:
                        raise ValueError(
                            "arm feature width disagrees with the declared "
                            "feature_names; basis construction must happen "
                            "before FEATURE_NAMES is rebound"
                        )
            absolute_selection = fitting._select_absolute_alpha(arm_base, folds)
            absolute_alpha = float(absolute_selection["selected"]["alpha"])
            rank_selection = fitting._select_rank_alpha(
                arm_base, folds, absolute_alpha=absolute_alpha
            )
            rank_alpha = float(rank_selection["selected"]["alpha"])
            base_absolute = fitting._fit_absolute_head(arm_base, alpha=absolute_alpha)
            base_rank = fitting._fit_rank_head(arm_base, alpha=rank_alpha)
            final_population = [*arm_base, *arm_extension]
            final_absolute = fitting._fit_absolute_head(
                final_population, alpha=absolute_alpha
            )
            final_rank = fitting._fit_rank_head(final_population, alpha=rank_alpha)
            return {
                "feature_dimension": len(feature_names),
                "absolute_alpha": absolute_alpha,
                "rank_alpha": rank_alpha,
                "absolute_selection": absolute_selection,
                "rank_selection": rank_selection,
                "models": {
                    "base_only_absolute_head": base_absolute,
                    "base_only_rank_head": base_rank,
                    "final_absolute_head": final_absolute,
                    "final_rank_head": final_rank,
                },
                "evaluation": {
                    "base_cross_validation": rank_selection["selected"]["metrics"],
                    "consumed_native_holdout_diagnostic": _evaluate_entries(
                        fitting._prediction_entries(
                            arm_holdout, base_absolute, base_rank
                        )
                    ),
                    "extension_base_only": _evaluate_entries(
                        fitting._prediction_entries(
                            arm_extension, base_absolute, base_rank
                        )
                    ),
                    "extension_leave_one_scenario_out": _evaluate_entries(
                        fitting._extension_oof_entries(
                            arm_base,
                            arm_extension,
                            absolute_alpha=absolute_alpha,
                            rank_alpha=rank_alpha,
                        )
                    ),
                    "extension_final_fit_in_sample": _evaluate_entries(
                        fitting._prediction_entries(
                            arm_extension, final_absolute, final_rank
                        )
                    ),
                },
            }
        finally:
            fitting.FEATURE_NAMES = original

    v2_arm = _fit_arm(base, extension, holdout, FEATURE_NAMES)
    v1_arm = _fit_arm(
        _with_v1_features(base),
        _with_v1_features(extension),
        _with_v1_features(holdout),
        V1_FEATURE_NAMES,
    )

    cutoff_audit = _counterfactual_cutoff_audit_v2(
        extension[0],
        profiles,
        v2_arm["models"]["final_absolute_head"],
        v2_arm["models"]["final_rank_head"],
        hardware_memory_bytes=hardware_memory_bytes,
        linear_score=fitting._linear_score,
    )
    if not cutoff_audit["passes"]:
        raise ValueError("v2 cutoff invariance audit failed")

    v1_artifact = read_json(v1_artifact_path)
    v1_published = v1_artifact["evaluation"]["extension"]

    primary_v1 = v1_arm["evaluation"]["extension_leave_one_scenario_out"]
    primary_v2 = v2_arm["evaluation"]["extension_leave_one_scenario_out"]

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sharding_mbs_interaction_ablation_complete",
        "analysis_only": True,
        "publishable": False,
        "production_profile_generated": False,
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "frozen_v1_artifact_modified": False,
        "model_contract": {
            "scope": "text SFT, packing=false, H800 diagnostic",
            "target": "effective_tokens_per_second",
            "feature_dimension": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "added_interaction_features": list(INTERACTION_FEATURE_NAMES),
            "inherited_feature_dimension": len(V1_FEATURE_NAMES),
            "removed_direct_cutoff_features": sorted(REMOVED_CUTOFF_FEATURES),
            "cutoff_is_only_a_truncation_control": True,
            "dynamic_padding_to_multiple_of": PAD_TO_MULTIPLE_OF,
            "two_independent_heads": True,
            "ranking_head_is_authoritative_for_order": True,
            "material_gap": MATERIAL_GAP,
            "material_pair_weight": MATERIAL_PAIR_WEIGHT,
            "alpha_grid": list(ALPHA_GRID),
            "unchanged_from_v1": [
                "work reconstruction and dynamic padding",
                "IRLS Huber ridge solver",
                "alpha grid and selection rules",
                "head separation and set-aware composition",
                "every evaluation metric definition",
            ],
        },
        "ablation": {
            "design": (
                "both arms share folds, solver, alpha grid, selection rules and "
                "evaluation code; they differ only in the feature set"
            ),
            "primary_comparison": "extension_leave_one_scenario_out",
            "arms": {
                "v1_features_refit_here": v1_arm,
                "v2_features": v2_arm,
            },
            "v1_published_reference": v1_published,
            "refit_reproduces_published_v1": {
                "published_cross_mechanism_pairwise_accuracy": (
                    v1_published["challenger_leave_one_extension_scenario_out"][
                        "cross_mechanism_pairwise_accuracy"
                    ]
                ),
                "refit_cross_mechanism_pairwise_accuracy": primary_v1[
                    "cross_mechanism_pairwise_accuracy"
                ],
                "matches": bool(
                    abs(
                        float(
                            v1_published[
                                "challenger_leave_one_extension_scenario_out"
                            ]["cross_mechanism_pairwise_accuracy"]
                        )
                        - float(primary_v1["cross_mechanism_pairwise_accuracy"])
                    )
                    < 1.0e-9
                ),
            },
        },
        "headline": {
            "v1_features": _headline(primary_v1),
            "v2_features": _headline(primary_v2),
        },
        "mechanism_pair_delta": _mechanism_pair_delta(primary_v1, primary_v2),
        "interaction_coefficients": {
            "base_only_rank_head": _coefficient_table(
                v2_arm["models"]["base_only_rank_head"]
            ),
            "final_rank_head": _coefficient_table(
                v2_arm["models"]["final_rank_head"]
            ),
            "base_only_absolute_head": _coefficient_table(
                v2_arm["models"]["base_only_absolute_head"]
            ),
            "final_absolute_head": _coefficient_table(
                v2_arm["models"]["final_absolute_head"]
            ),
        },
        "acceptance": _acceptance(primary_v1, primary_v2),
        "cutoff_invariance_audit": cutoff_audit,
        "data": {
            "base_fit_candidates": len(base),
            "base_fit_scenarios": len({str(row["scenario_id"]) for row in base}),
            "consumed_native_holdout_candidates": len(holdout),
            "extension_success_candidates": len(extension),
            "extension_scenarios": len(
                {str(row["scenario_id"]) for row in extension}
            ),
            "extension_results_are_fresh_holdout": False,
            "extension_results_used_in_final_fit": True,
        },
        "source_bindings": {
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "inherited_implementation": {
                "path": str(
                    (SCRIPTS / "fit_rank_first_throughput_challenger_v1.py").resolve()
                ),
                "sha256": sha256_file(
                    SCRIPTS / "fit_rank_first_throughput_challenger_v1.py"
                ),
            },
            "profiles": profiles.source_bindings(),
            "h800_observations": {
                "path": str(h800_observations_path.resolve()),
                "sha256": sha256_file(h800_observations_path),
            },
            "extension_evaluation": {
                "path": str(extension_evaluation_path.resolve()),
                "sha256": sha256_file(extension_evaluation_path),
            },
            "extension_predictions": {
                "path": str(extension_predictions_path.resolve()),
                "sha256": sha256_file(extension_predictions_path),
            },
            "frozen_v5": {
                "path": str(v5_artifact_path.resolve()),
                "sha256": sha256_file(v5_artifact_path),
            },
            "challenger_v1": {
                "path": str(v1_artifact_path.resolve()),
                "sha256": sha256_file(v1_artifact_path),
            },
        },
        "limitations": [
            "extension outcomes are consumed and are not a fresh release holdout",
            "the interaction terms were motivated by inspecting v1 failures on "
            "these same scenarios, so this is not a prospective validation",
            "ZeRO-3 appears at a single micro-batch size within the extension "
            "scenarios; cross-MBS ZeRO-3 evidence comes from the strict-fit pool",
            "matched contrasts holding gpu_count and mbs fixed while varying only "
            "ZeRO/GC are still missing and remain the next experimental step",
            "packing, VL, offload and non-H800 cards are out of scope",
            "this challenger does not replace the frozen V5 or the v1 artifact",
        ],
    }
    report["report_sha256"] = sha256_json(report)
    return report


def _render_markdown(report: Mapping[str, Any]) -> str:
    def pct(value: Any) -> str:
        if value is None:
            return "n/a"
        return f"{float(value) * 100:.2f}%"

    v1 = report["headline"]["v1_features"]
    v2 = report["headline"]["v2_features"]
    lines = [
        "# 吞吐 challenger v2：ZeRO/GC × MBS 交互项消融",
        "",
        f"- 生成时间：{report['generated_at_utc']}",
        f"- 实现版本：`{report['implementation_version']}`",
        f"- 特征维度：{len(V1_FEATURE_NAMES)} → {report['model_contract']['feature_dimension']}",
        f"- 新增交互项：{', '.join(report['model_contract']['added_interaction_features'])}",
        "- 未发布（`publishable=false`），未启动 GPU 实验，未改动 V5 与 v1 冻结产物。",
        "",
        "## 主对比口径：扩展场景逐场景留出（leave-one-scenario-out）",
        "",
        "| 指标 | v1 特征 | v2 特征 |",
        "|---|---:|---:|",
        f"| 跨机制排序正确率 | {pct(v1['cross_mechanism_pairwise_accuracy'])} | {pct(v2['cross_mechanism_pairwise_accuracy'])} |",
        f"| 重要差距跨机制排序正确率 | {pct(v1['material_cross_mechanism_pairwise_accuracy'])} | {pct(v2['material_cross_mechanism_pairwise_accuracy'])} |",
        f"| 场景等权重要差距正确率 | {pct(v1['scenario_equal_material_cross_mechanism_pairwise_accuracy'])} | {pct(v2['scenario_equal_material_cross_mechanism_pairwise_accuracy'])} |",
        f"| 平均 Top-1 损失 | {pct(v1['mean_top1_regret'])} | {pct(v2['mean_top1_regret'])} |",
        f"| 最差 Top-1 损失 | {pct(v1['worst_top1_regret'])} | {pct(v2['worst_top1_regret'])} |",
        f"| Top-1 完全命中率 | {pct(v1['exact_top1_fraction'])} | {pct(v2['exact_top1_fraction'])} |",
        f"| 绝对吞吐平均相对误差 | {pct(v1['set_aware_absolute_mape'])} | {pct(v2['set_aware_absolute_mape'])} |",
        "",
        "## 机制对逐项变化",
        "",
        "| 机制对 | 对比数 | v1 正确 | v2 正确 | v1 正确率 | v2 正确率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, value in report["mechanism_pair_delta"].items():
        lines.append(
            "| `{key}` | {n} | {c1} | {c2} | {a1} | {a2} |".format(
                key=key,
                n=value["comparisons"],
                c1=value["v1_correct"],
                c2=value["v2_correct"],
                a1=pct(value["v1_accuracy"]),
                a2=pct(value["v2_accuracy"]),
            )
        )
    acceptance = report["acceptance"]
    lines += [
        "",
        "## 预注册验收判定",
        "",
        f"- 规则：{acceptance['rule']}",
        f"- 用户口径：`{acceptance['user_release_criterion']}`",
        "",
        "| 检查项 | 结果 |",
        "|---|---|",
    ]
    for key, value in acceptance["checks"].items():
        lines.append(f"| `{key}` | {'通过' if value else '未通过'} |")
    lines += [
        "",
        f"**总判定：{'保留交互项' if acceptance['all_checks_pass'] else '不保留交互项'}**",
        "",
        "## 交互项系数（排序头，标准化尺度）",
        "",
        "| 特征 | 仅训练集拟合 | 含扩展的最终拟合 |",
        "|---|---:|---:|",
    ]
    base_rank = report["interaction_coefficients"]["base_only_rank_head"]
    final_rank = report["interaction_coefficients"]["final_rank_head"]
    for name in report["model_contract"]["added_interaction_features"]:
        lines.append(
            f"| `{name}` | {base_rank.get(name, float('nan')):.4f} | "
            f"{final_rank.get(name, float('nan')):.4f} |"
        )
    audit = report["cutoff_invariance_audit"]
    lines += [
        "",
        "## cutoff 不变量复核",
        "",
        f"- 数据集：`{audit['dataset_id']}`，样本最大长度 {audit['profile_raw_maximum']}",
        f"- 检查的 cutoff：{audit['cutoffs']}",
        f"- 特征向量完全一致：{audit['feature_vectors_identical']}",
        f"- 绝对预测极差：{audit['absolute_prediction_range']}",
        f"- 排序分极差：{audit['rank_score_range']}",
        f"- 结论：{'通过' if audit['passes'] else '未通过'}",
        "",
        "## 局限",
        "",
    ]
    lines += [f"- {item}" for item in report["limitations"]]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-profile-dir",
        type=Path,
        default=ROOT / "artifacts" / "dataset_profiles",
    )
    parser.add_argument(
        "--additional-dataset-profile-dir",
        type=Path,
        default=ROOT / "artifacts" / "h800_lora_safety_stage2_v1" / "profiles",
    )
    parser.add_argument(
        "--h800-observations",
        type=Path,
        default=ROOT / "artifacts" / "canonical_h800_observations.jsonl",
    )
    parser.add_argument(
        "--h800-theory-basis",
        type=Path,
        default=ROOT / "artifacts" / "h800_theory_basis.json",
    )
    parser.add_argument(
        "--h800-model-inventory",
        type=Path,
        default=ROOT / "artifacts" / "model_inventory.json",
    )
    parser.add_argument(
        "--h800-hardware", type=Path, default=ROOT / "config" / "hardware.json"
    )
    parser.add_argument("--h800-runtime-root", type=Path, default=ROOT / "runtime")
    parser.add_argument(
        "--extension-evaluation",
        type=Path,
        default=ROOT / "artifacts" / "v5_dataset_candidate_extension_h800_20260805.json",
    )
    parser.add_argument(
        "--extension-predictions",
        type=Path,
        default=ROOT
        / "artifacts"
        / "h800_v5_dataset_candidate_extension_frozen_predictions_v1.json",
    )
    parser.add_argument(
        "--v5-artifact",
        type=Path,
        default=ROOT / "artifacts" / "structured_throughput_modeling.json",
    )
    parser.add_argument(
        "--v1-artifact",
        type=Path,
        default=ROOT / "artifacts" / "rank_first_throughput_challenger_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "diagnostics"
        / "rank_first_challenger_sharding_mbs_v2"
        / "rank_first_throughput_challenger_v2.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT
        / "diagnostics"
        / "rank_first_challenger_sharding_mbs_v2"
        / "rank_first_throughput_challenger_v2.md",
    )
    args = parser.parse_args()

    report = build_report(
        profile_dirs=[args.dataset_profile_dir, args.additional_dataset_profile_dir],
        h800_observations_path=args.h800_observations,
        h800_theory_basis_path=args.h800_theory_basis,
        h800_inventory_path=args.h800_model_inventory,
        h800_hardware_path=args.h800_hardware,
        h800_runtime_root=args.h800_runtime_root,
        extension_evaluation_path=args.extension_evaluation,
        extension_predictions_path=args.extension_predictions,
        v5_artifact_path=args.v5_artifact,
        v1_artifact_path=args.v1_artifact,
    )
    write_json(args.output, report)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "markdown_output": str(args.markdown_output.resolve()),
                "report_sha256": report["report_sha256"],
                "headline": report["headline"],
                "mechanism_pair_delta": report["mechanism_pair_delta"],
                "acceptance": report["acceptance"],
                "cutoff_invariance_passes": report["cutoff_invariance_audit"]["passes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
