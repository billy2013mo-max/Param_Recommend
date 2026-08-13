#!/usr/bin/env python3
"""Per-term sweep of the proposed sharding/checkpointing x MBS interactions.

The bundled v2 change (three interaction terms at once) improved the six
consumed extension scenarios while degrading both larger evaluation populations.
That pattern is what fitting to a diagnosed failure looks like, so before either
keeping or discarding the direction we test each term on its own.

Every subset is fitted and evaluated with v1's unmodified solver, alpha grid,
selection rules and metric definitions; only the feature subset varies.  Each
subset is scored on three populations of very different size:

* ``base_cross_validation``          --- 238 candidates / 63 scenarios / 907 cross-mechanism pairs
* ``consumed_native_holdout``        ---  79 candidates / 16 scenarios / 399 cross-mechanism pairs
* ``extension_leave_one_scenario_out`` --- 39 candidates / 6 scenarios / 66 cross-mechanism pairs

The extension population is the one the terms were designed against, so it is
reported last and never treated as the deciding evidence.

Offline and diagnostic: no GPU work, no queue mutation, no production profile,
no modification of the frozen V5 artifact or the v1 challenger artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import ROOT, read_json, sha256_file, sha256_json, write_json  # noqa: E402
import fit_rank_first_throughput_challenger_v1 as fitting  # noqa: E402
from fit_rank_first_throughput_challenger_v1 import (  # noqa: E402
    InvariantStaticProfiles,
    _evaluate_entries,
    _load_extension_candidates,
    _scenario_folds,
)
from fit_rank_first_throughput_challenger_v1 import (  # noqa: E402
    FEATURE_NAMES as V1_FEATURE_NAMES,
)
from fit_rank_first_throughput_challenger_v2 import (  # noqa: E402
    INTERACTION_FEATURE_NAMES,
    _candidate_from_pooled_v2,
    _v2_candidate,
)
from structured_throughput_modeling import _load_h800  # noqa: E402

SCHEMA = "sft_rank_first_challenger_interaction_term_sweep/v1"
IMPLEMENTATION_VERSION = (
    "sft_rank_first_challenger_interaction_term_sweep/2026-08-05.per-term-v1"
)

HEADLINE_KEYS = (
    "cross_mechanism_pairwise_comparisons",
    "cross_mechanism_pairwise_accuracy",
    "material_cross_mechanism_pairwise_accuracy",
    "scenario_equal_material_cross_mechanism_pairwise_accuracy",
    "mean_top1_regret",
    "worst_top1_regret",
    "exact_top1_fraction",
    "set_aware_absolute_mape",
)


def _subsets() -> list[tuple[str, tuple[str, ...]]]:
    output: list[tuple[str, tuple[str, ...]]] = [("v1_baseline", ())]
    for size in range(1, len(INTERACTION_FEATURE_NAMES) + 1):
        for combo in combinations(INTERACTION_FEATURE_NAMES, size):
            label = "+".join(name.replace("_x_log2_mbs", "") for name in combo)
            output.append((label, combo))
    return output


def _project(
    candidates: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    full_names: Sequence[str],
) -> list[dict[str, Any]]:
    index = [full_names.index(name) for name in feature_names]
    picker = np.asarray(index, dtype=int)
    return [
        {**row, "features": np.asarray(row["features_full"], dtype=float)[picker]}
        for row in candidates
    ]


def _run_subset(
    feature_names: Sequence[str],
    base: Sequence[Mapping[str, Any]],
    extension: Sequence[Mapping[str, Any]],
    holdout: Sequence[Mapping[str, Any]],
    folds: Sequence[set[str]],
) -> dict[str, Any]:
    original = fitting.FEATURE_NAMES
    fitting.FEATURE_NAMES = tuple(feature_names)
    try:
        absolute_selection = fitting._select_absolute_alpha(base, folds)
        absolute_alpha = float(absolute_selection["selected"]["alpha"])
        rank_selection = fitting._select_rank_alpha(
            base, folds, absolute_alpha=absolute_alpha
        )
        rank_alpha = float(rank_selection["selected"]["alpha"])
        base_absolute = fitting._fit_absolute_head(base, alpha=absolute_alpha)
        base_rank = fitting._fit_rank_head(base, alpha=rank_alpha)
        evaluations = {
            "base_cross_validation": rank_selection["selected"]["metrics"],
            "consumed_native_holdout": _evaluate_entries(
                fitting._prediction_entries(holdout, base_absolute, base_rank)
            ),
            "extension_leave_one_scenario_out": _evaluate_entries(
                fitting._extension_oof_entries(
                    base,
                    extension,
                    absolute_alpha=absolute_alpha,
                    rank_alpha=rank_alpha,
                )
            ),
        }
        coefficients = {
            name: float(value)
            for name, value in zip(base_rank["feature_names"], base_rank["coefficients"])
            if name in INTERACTION_FEATURE_NAMES
        }
        return {
            "feature_dimension": len(feature_names),
            "absolute_alpha": absolute_alpha,
            "rank_alpha": rank_alpha,
            "interaction_rank_coefficients": coefficients,
            "evaluation": {
                name: {key: metrics.get(key) for key in HEADLINE_KEYS}
                for name, metrics in evaluations.items()
            },
            "wrong_cross_mechanism_pairs": {
                name: metrics.get("wrong_cross_mechanism_pairs")
                for name, metrics in evaluations.items()
                if name == "extension_leave_one_scenario_out"
            },
        }
    finally:
        fitting.FEATURE_NAMES = original


def _coverage_limitations(
    extension: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Describe the sharding/MBS coverage actually present, not a fixed guess.

    These two lines used to be hard-coded and went stale the moment matched
    contrasts were added: the report kept claiming ZeRO-3 was confined to one
    micro-batch size after that had ceased to be true, which reads as "the
    confound is still there" when in fact it had been removed.
    """

    levels: dict[int, set[int]] = defaultdict(set)
    for row in extension:
        selector = row["record"]["selector"]
        scenario = row["record"]["scenario"]
        levels[int(selector.get("zero_stage") or 0)].add(
            int(scenario.get("physical_mbs") or 0)
        )
    zero2 = sorted(levels.get(2, set()))
    zero3 = sorted(levels.get(3, set()))
    common = sorted(set(zero2) & set(zero3))
    out = [
        f"extension ZeRO-2 covers MBS {zero2}; ZeRO-3 covers MBS {zero3}",
    ]
    if len(common) >= 2:
        out.append(
            f"matched contrasts are present (common MBS {common}), so the "
            "sharding effect is separable from the batch-size effect here"
        )
    else:
        out.append(
            "matched contrasts holding gpu_count and mbs fixed are still missing"
        )
    return out


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
) -> dict[str, Any]:
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
    full_names = list(V1_FEATURE_NAMES) + list(INTERACTION_FEATURE_NAMES)

    def _prepare(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{**row, "features_full": np.asarray(row["features"], dtype=float)} for row in rows]

    base = _prepare(
        [
            _candidate_from_pooled_v2(
                candidate,
                profiles,
                hardware_memory_bytes=hardware_memory_bytes,
                source_role="h800_strict_fit",
            )
            for candidate in h800["strict_fit_candidates"]
        ]
    )
    holdout = _prepare(
        [
            _candidate_from_pooled_v2(
                candidate,
                profiles,
                hardware_memory_bytes=hardware_memory_bytes,
                source_role="h800_consumed_native_holdout",
            )
            for candidate in h800["holdout_candidates"]
        ]
    )
    extension = _prepare(
        [
            _v2_candidate(candidate)
            for candidate in _load_extension_candidates(
                extension_evaluation_path,
                extension_predictions_path,
                profiles,
                hardware_memory_bytes=hardware_memory_bytes,
            )
        ]
    )
    folds = _scenario_folds(base)

    results: dict[str, Any] = {}
    for label, combo in _subsets():
        names = list(V1_FEATURE_NAMES) + list(combo)
        results[label] = {
            "added_terms": list(combo),
            **_run_subset(
                names,
                _project(base, names, full_names),
                _project(extension, names, full_names),
                _project(holdout, names, full_names),
                folds,
            ),
        }

    baseline = results["v1_baseline"]["evaluation"]
    for label, entry in results.items():
        entry["delta_vs_v1_baseline"] = {
            population: {
                key: (
                    float(entry["evaluation"][population][key])
                    - float(baseline[population][key])
                )
                if entry["evaluation"][population].get(key) is not None
                and baseline[population].get(key) is not None
                else None
                for key in (
                    "cross_mechanism_pairwise_accuracy",
                    "material_cross_mechanism_pairwise_accuracy",
                    "scenario_equal_material_cross_mechanism_pairwise_accuracy",
                    "mean_top1_regret",
                    "worst_top1_regret",
                    "set_aware_absolute_mape",
                )
            }
            for population in baseline
        }

    dominating = [
        label
        for label, entry in results.items()
        if label != "v1_baseline"
        and all(
            (entry["delta_vs_v1_baseline"][population]["cross_mechanism_pairwise_accuracy"] or 0.0)
            >= -1.0e-12
            for population in ("base_cross_validation", "consumed_native_holdout")
        )
        and (
            entry["delta_vs_v1_baseline"]["extension_leave_one_scenario_out"][
                "cross_mechanism_pairwise_accuracy"
            ]
            or 0.0
        )
        > 1.0e-12
    ]

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "interaction_term_sweep_complete",
        "analysis_only": True,
        "publishable": False,
        "production_profile_generated": False,
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "design": {
            "question": (
                "does any subset of the sharding/checkpointing x MBS interaction "
                "terms improve ordering without degrading the larger populations"
            ),
            "invariant_across_arms": [
                "work reconstruction and dynamic padding",
                "IRLS Huber ridge solver",
                "alpha grid and both selection rules",
                "scenario folds",
                "every evaluation metric definition",
            ],
            "populations_by_size": {
                "base_cross_validation": "238 candidates / 63 scenarios",
                "consumed_native_holdout": "79 candidates / 16 scenarios",
                "extension_leave_one_scenario_out": "39 candidates / 6 scenarios",
            },
            "decision_rule": (
                "a term is worth keeping only if it does not reduce cross-mechanism "
                "ordering accuracy on either larger population while improving the "
                "extension population"
            ),
        },
        "results": results,
        "subsets_passing_decision_rule": dominating,
        "conclusion": (
            "no subset qualifies"
            if not dominating
            else f"qualifying subsets: {', '.join(dominating)}"
        ),
        "source_bindings": {
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "challenger_v1_implementation": {
                "path": str((SCRIPTS / "fit_rank_first_throughput_challenger_v1.py").resolve()),
                "sha256": sha256_file(SCRIPTS / "fit_rank_first_throughput_challenger_v1.py"),
            },
            "challenger_v2_implementation": {
                "path": str((SCRIPTS / "fit_rank_first_throughput_challenger_v2.py").resolve()),
                "sha256": sha256_file(SCRIPTS / "fit_rank_first_throughput_challenger_v2.py"),
            },
            "h800_observations": {
                "path": str(h800_observations_path.resolve()),
                "sha256": sha256_file(h800_observations_path),
            },
        },
        "limitations": [
            "the extension population is the one these terms were designed against",
            *_coverage_limitations(extension),
            "packing, VL, offload and non-H800 cards are out of scope",
        ],
    }
    report["report_sha256"] = sha256_json(report)
    return report


def _render_markdown(report: Mapping[str, Any]) -> str:
    def pct(value: Any) -> str:
        return "n/a" if value is None else f"{float(value) * 100:.2f}%"

    def sign(value: Any) -> str:
        return "n/a" if value is None else f"{float(value) * 100:+.2f}pp"

    lines = [
        "# 交互项逐项消融：ZeRO/GC × MBS",
        "",
        f"- 生成时间：{report['generated_at_utc']}",
        f"- 判定规则：{report['design']['decision_rule']}",
        "- 未发布，未启动 GPU 实验，未改动 V5 与 v1 冻结产物。",
        "",
        "## 跨机制排序正确率（三个样本量差异很大的口径）",
        "",
        "| 特征子集 | 维度 | 训练交叉验证 (907 对) | 原生留出 (399 对) | 扩展场景留出 (66 对) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, entry in report["results"].items():
        ev = entry["evaluation"]
        lines.append(
            "| `{label}` | {dim} | {a} | {b} | {c} |".format(
                label=label,
                dim=entry["feature_dimension"],
                a=pct(ev["base_cross_validation"]["cross_mechanism_pairwise_accuracy"]),
                b=pct(ev["consumed_native_holdout"]["cross_mechanism_pairwise_accuracy"]),
                c=pct(
                    ev["extension_leave_one_scenario_out"][
                        "cross_mechanism_pairwise_accuracy"
                    ]
                ),
            )
        )
    lines += [
        "",
        "## 相对 v1 基线的变化",
        "",
        "| 特征子集 | 训练交叉验证 | 原生留出 | 扩展场景留出 |",
        "|---|---:|---:|---:|",
    ]
    for label, entry in report["results"].items():
        if label == "v1_baseline":
            continue
        delta = entry["delta_vs_v1_baseline"]
        lines.append(
            "| `{label}` | {a} | {b} | {c} |".format(
                label=label,
                a=sign(delta["base_cross_validation"]["cross_mechanism_pairwise_accuracy"]),
                b=sign(delta["consumed_native_holdout"]["cross_mechanism_pairwise_accuracy"]),
                c=sign(
                    delta["extension_leave_one_scenario_out"][
                        "cross_mechanism_pairwise_accuracy"
                    ]
                ),
            )
        )
    lines += [
        "",
        "## Top-1 损失（用户口径：最差 < 10%）",
        "",
        "| 特征子集 | 训练平均 | 训练最差 | 留出最差 | 扩展最差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, entry in report["results"].items():
        ev = entry["evaluation"]
        lines.append(
            "| `{label}` | {a} | {b} | {c} | {d} |".format(
                label=label,
                a=pct(ev["base_cross_validation"]["mean_top1_regret"]),
                b=pct(ev["base_cross_validation"]["worst_top1_regret"]),
                c=pct(ev["consumed_native_holdout"]["worst_top1_regret"]),
                d=pct(ev["extension_leave_one_scenario_out"]["worst_top1_regret"]),
            )
        )
    lines += [
        "",
        f"**结论：{report['conclusion']}**",
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
        "--dataset-profile-dir", type=Path, default=ROOT / "artifacts" / "dataset_profiles"
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
        "--h800-theory-basis", type=Path, default=ROOT / "artifacts" / "h800_theory_basis.json"
    )
    parser.add_argument(
        "--h800-model-inventory", type=Path, default=ROOT / "artifacts" / "model_inventory.json"
    )
    parser.add_argument("--h800-hardware", type=Path, default=ROOT / "config" / "hardware.json")
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
        "--output",
        type=Path,
        default=ROOT
        / "diagnostics"
        / "rank_first_challenger_sharding_mbs_v2"
        / "interaction_term_sweep_v1.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT
        / "diagnostics"
        / "rank_first_challenger_sharding_mbs_v2"
        / "interaction_term_sweep_v1.md",
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
                "subsets_passing_decision_rule": report["subsets_passing_decision_rule"],
                "conclusion": report["conclusion"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
