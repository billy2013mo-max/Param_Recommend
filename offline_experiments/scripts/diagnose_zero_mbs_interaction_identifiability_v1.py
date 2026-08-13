#!/usr/bin/env python3
"""Diagnose whether ZeRO x MBS interaction terms are identifiable.

Before adding ``zero3_x_log2_mbs`` / ``gc_x_log2_mbs`` to the rank-first
challenger we must know whether the existing H800 evidence can estimate them at
all.  If ZeRO-3 only ever appears at a single micro-batch size, the proposed
interaction is collinear with the plain ``zero3`` indicator and fitting it would
merely relabel an existing coefficient instead of adding information.

This script is read-only.  It loads no model, refits nothing, writes no
artifact, launches no GPU work and mutates no queue.  It only prints a
cross-tabulation and the collinearity diagnostics implied by it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import ROOT, read_json  # noqa: E402
from fit_rank_first_throughput_challenger_v1 import (  # noqa: E402
    FEATURE_NAMES,
    InvariantStaticProfiles,
    _candidate_from_pooled,
    _load_extension_candidates,
)
from structured_throughput_modeling import _load_h800  # noqa: E402

PROPOSED = ("zero3_x_log2_mbs", "gc_x_log2_mbs", "zero2_x_log2_mbs")


def _selector(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    return candidate["record"]["selector"]


def _scenario(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    return candidate["record"]["scenario"]


def _cell(candidate: Mapping[str, Any]) -> tuple[int, int, bool]:
    selector = _selector(candidate)
    return (
        int(selector.get("zero_stage") or 0),
        int(_scenario(candidate).get("physical_mbs") or 0),
        bool(selector.get("gradient_checkpointing")),
    )


def _proposed_values(candidate: Mapping[str, Any]) -> dict[str, float]:
    selector = _selector(candidate)
    zero_stage = int(selector.get("zero_stage") or 0)
    gc = float(bool(selector.get("gradient_checkpointing")))
    mbs = int(_scenario(candidate).get("physical_mbs") or 0)
    log2_mbs = math.log2(mbs) if mbs > 0 else 0.0
    return {
        "zero3_x_log2_mbs": float(zero_stage == 3) * log2_mbs,
        "gc_x_log2_mbs": gc * log2_mbs,
        "zero2_x_log2_mbs": float(zero_stage == 2) * log2_mbs,
    }


def _crosstab(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    table: dict[int, Counter] = defaultdict(Counter)
    scenarios: dict[int, set[str]] = defaultdict(set)
    for candidate in candidates:
        zero_stage, mbs, _ = _cell(candidate)
        table[zero_stage][mbs] += 1
        scenarios[zero_stage].add(str(candidate["scenario_id"]))
    return {
        "counts": {
            str(stage): dict(sorted(counts.items()))
            for stage, counts in sorted(table.items())
        },
        "distinct_mbs_per_zero_stage": {
            str(stage): sorted(counts) for stage, counts in sorted(table.items())
        },
        "scenarios_per_zero_stage": {
            str(stage): len(names) for stage, names in sorted(scenarios.items())
        },
    }


def _gc_crosstab(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    table: dict[str, Counter] = defaultdict(Counter)
    for candidate in candidates:
        _, mbs, gc = _cell(candidate)
        table["gc_on" if gc else "gc_off"][mbs] += 1
    return {
        key: dict(sorted(counts.items())) for key, counts in sorted(table.items())
    }


def _within_scenario_contrasts(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count scenarios that can actually separate ZeRO from MBS.

    A scenario identifies the interaction only if it contains at least two
    distinct micro-batch sizes for the same ZeRO stage AND at least two ZeRO
    stages overall.  Otherwise the ZeRO effect and the MBS effect move together
    inside every candidate set the ranking head ever sees.
    """

    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_scenario[str(candidate["scenario_id"])].append(candidate)

    both_stages = 0
    mbs_varies_within_stage = 0
    identifying = 0
    detail: list[dict[str, Any]] = []
    for scenario_id, rows in sorted(by_scenario.items()):
        stages = {int(_selector(row).get("zero_stage") or 0) for row in rows}
        per_stage_mbs = defaultdict(set)
        for row in rows:
            per_stage_mbs[int(_selector(row).get("zero_stage") or 0)].add(
                int(_scenario(row).get("physical_mbs") or 0)
            )
        has_both = len(stages) >= 2
        varies = any(len(values) >= 2 for values in per_stage_mbs.values())
        if has_both:
            both_stages += 1
        if varies:
            mbs_varies_within_stage += 1
        if has_both and varies:
            identifying += 1
            detail.append(
                {
                    "scenario_id": scenario_id,
                    "zero_stages": sorted(stages),
                    "mbs_per_stage": {
                        str(stage): sorted(values)
                        for stage, values in sorted(per_stage_mbs.items())
                    },
                }
            )
    return {
        "scenarios_total": len(by_scenario),
        "scenarios_with_two_zero_stages": both_stages,
        "scenarios_with_mbs_variation_within_a_stage": mbs_varies_within_stage,
        "scenarios_identifying_the_interaction": identifying,
        "identifying_scenarios": detail,
    }


def _collinearity(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Regress each proposed term on the existing 44 features."""

    base = np.vstack(
        [np.asarray(row["features"], dtype=float) for row in candidates]
    )
    mean = base.mean(axis=0)
    scale = base.std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    standardized = (base - mean) / scale
    design = np.hstack([standardized, np.ones((standardized.shape[0], 1))])

    results: dict[str, Any] = {}
    for name in PROPOSED:
        target = np.asarray(
            [_proposed_values(row)[name] for row in candidates], dtype=float
        )
        if float(np.std(target)) < 1.0e-12:
            results[name] = {
                "constant_in_this_population": True,
                "note": "term has no variation, cannot be estimated",
            }
            continue
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        fitted = design @ coefficients
        residual = target - fitted
        total_ss = float(((target - target.mean()) ** 2).sum())
        residual_ss = float((residual**2).sum())
        r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else float("nan")
        results[name] = {
            "constant_in_this_population": False,
            "r_squared_on_existing_features": r_squared,
            "variance_inflation_factor": (
                float("inf") if r_squared >= 1.0 - 1.0e-12 else 1.0 / (1.0 - r_squared)
            ),
            "residual_std": float(np.std(residual)),
            "target_std": float(np.std(target)),
            "residual_share_of_target_std": (
                float(np.std(residual) / np.std(target))
                if float(np.std(target)) > 0
                else float("nan")
            ),
        }
    return results


def _mechanism_cells(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[int, int, bool], dict[str, Any]] = {}
    for candidate in candidates:
        key = _cell(candidate)
        entry = cells.setdefault(
            key,
            {
                "zero_stage": key[0],
                "physical_mbs": key[1],
                "gradient_checkpointing": key[2],
                "candidates": 0,
                "scenarios": set(),
                "datasets": set(),
                "cutoffs": set(),
            },
        )
        entry["candidates"] += 1
        entry["scenarios"].add(str(candidate["scenario_id"]))
        entry["datasets"].add(str(_scenario(candidate).get("dataset_id")))
        entry["cutoffs"].add(int(_scenario(candidate).get("cutoff_len") or 0))
    output = []
    for entry in cells.values():
        output.append(
            {
                **{
                    key: entry[key]
                    for key in (
                        "zero_stage",
                        "physical_mbs",
                        "gradient_checkpointing",
                        "candidates",
                    )
                },
                "scenarios": len(entry["scenarios"]),
                "datasets": len(entry["datasets"]),
                "distinct_cutoffs": sorted(entry["cutoffs"]),
            }
        )
    return sorted(
        output,
        key=lambda row: (row["zero_stage"], row["physical_mbs"], row["gradient_checkpointing"]),
    )


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
    args = parser.parse_args()

    profiles = InvariantStaticProfiles(
        [args.dataset_profile_dir, args.additional_dataset_profile_dir]
    )
    hardware = read_json(args.h800_hardware)
    hardware_memory_bytes = float(hardware["memory_bytes_reported_by_torch"])
    h800 = _load_h800(
        observation_path=args.h800_observations,
        theory_basis_path=args.h800_theory_basis,
        inventory_path=args.h800_model_inventory,
        hardware_path=args.h800_hardware,
        runtime_root=args.h800_runtime_root,
    )
    base = [
        _candidate_from_pooled(
            candidate,
            profiles,
            hardware_memory_bytes=hardware_memory_bytes,
            source_role="h800_strict_fit",
        )
        for candidate in h800["strict_fit_candidates"]
    ]
    extension = _load_extension_candidates(
        args.extension_evaluation,
        args.extension_predictions,
        profiles,
        hardware_memory_bytes=hardware_memory_bytes,
    )
    combined = [*base, *extension]

    report = {
        "analysis_only": True,
        "model_refit": False,
        "gpu_experiments_launched": False,
        "queues_mutated": False,
        "existing_feature_dimension": len(FEATURE_NAMES),
        "proposed_terms": list(PROPOSED),
        "populations": {},
    }
    for name, rows in (
        ("base_strict_fit", base),
        ("extension_only", extension),
        ("base_plus_extension", combined),
    ):
        report["populations"][name] = {
            "candidates": len(rows),
            "scenarios": len({str(row["scenario_id"]) for row in rows}),
            "zero_stage_by_mbs": _crosstab(rows),
            "gc_by_mbs": _gc_crosstab(rows),
            "within_scenario_identification": _within_scenario_contrasts(rows),
            "collinearity_with_existing_features": _collinearity(rows),
            "mechanism_cells": _mechanism_cells(rows),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
