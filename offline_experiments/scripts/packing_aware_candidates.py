#!/usr/bin/env python
"""Build a packing-aware candidate space for one scenario.

The plan requires packing on/off to compete inside a *single* candidate space
rather than being a separate global multiplier.  Two things stand in the way
today, and this module handles both explicitly instead of hiding them:

1. The packing decision is owned by a frozen, purely static policy.  This module
   never re-derives it, never overrides an ``off``, and never invents a gate.  It
   asks :mod:`static_packing_predictor` and then obeys the answer.
2. The frozen H800 memory and ranking heads have no ``packing=true`` training
   rows, so a packed candidate cannot be admitted or ranked yet.  Packed
   candidates are therefore emitted as an explicitly labelled *shadow* branch.

The resulting contract is deliberately asymmetric, and that asymmetry is the
point: unpacked candidates are ``rankable``; packed candidates are
``shadow_only`` and carry the reason why.  This keeps packing visible in the
candidate space -- so it stops being silently swallowed as an unknown field --
without letting an uncalibrated branch reach a recommendation.

One structural detail matters for callers.  The frozen predictor includes
``packing`` in its ``scenario_material`` and refuses a ``comparison_group`` that
mixes different user scenarios, so the two branches deliberately carry *different*
predictor group ids.  The "single candidate space" is expressed at the report
level -- one scenario in, one report out, both branches derived from the same
baseline shapes -- not by forcing both into one group id, which the predictor
would reject.

The module is CPU-only: it launches nothing, fits nothing, and mutates no frozen
artefact.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import candidate_generator as cg
import static_packing_predictor as sp
from common import ARTIFACT_DIR, read_json, sha256_file, sha256_json, write_json

SCHEMA = "sft_packing_aware_candidate_space/v1"
IMPLEMENTATION_VERSION = "sft_packing_aware_candidates_impl/2026-08-01.v1"

DEFAULT_POLICY_PATH = ARTIFACT_DIR / "static_packing_policy_v1.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "packing_aware_candidate_space_v1.json"

BRANCH_UNPACKED = "packing_off"
BRANCH_PACKED_SHADOW = "packing_on_shadow"

# The frozen H800 heads were fitted without any packing=true row, so a packed
# candidate has no calibrated memory bound and no ranking score.  Until the
# packing calibration stage closes, packed candidates may only be carried as
# shadow evidence.
PACKED_SHADOW_REASON = "packing_memory_and_throughput_heads_not_calibrated"


def _packing_request(
    scenario: Mapping[str, Any],
    *,
    no_packing_mbs: int,
    profile_path: str,
    gpu_count: int,
) -> dict[str, Any]:
    """Translate a scenario into the frozen policy's request contract."""

    return {
        "request_id": (
            f"{scenario['model_id']}-{scenario['training_mode']}"
            f"-{scenario['dataset_id']}-cut{scenario['cutoff_len']}"
            f"-mbs{no_packing_mbs}-g{gpu_count}"
        ),
        "modality": str(scenario.get("modality") or "text"),
        "gpu_family": str(scenario.get("gpu_family") or "H800"),
        "stage": str(scenario.get("stage") or "sft"),
        "dtype": str(scenario.get("dtype") or "bf16"),
        "model_id": scenario["model_id"],
        "train_type": scenario["training_mode"],
        "cutoff_len": int(scenario["cutoff_len"]),
        "target_gbs": int(scenario["target_gbs"]),
        "gpu_count": gpu_count,
        "no_packing_mbs": no_packing_mbs,
        "profile_path": profile_path,
        # Platform recommendations must enforce the GA-floor upper-tail gate.
        # Historical v1 replay requests omit this flag and remain reproducible.
        "require_gbs_controllable": True,
        "epsilon_gbs": float(scenario.get("packing_gbs_epsilon", 0.10)),
    }


def _decide_packing(
    scenario: Mapping[str, Any],
    *,
    baseline_shapes: Sequence[tuple[int, int]],
    policy: Mapping[str, Any],
    policy_path: Path,
    profile_path: str,
    request_base: Path,
) -> list[dict[str, Any]]:
    """Ask the frozen policy once per baseline (gpu_count, mbs) shape.

    Packing benefit depends on the unpacked micro batch it is replacing -- the
    same dataset can be worth packing against mbs=1 and not worth it against
    mbs=8 -- so the decision is per baseline shape, not per scenario.
    """

    decisions: list[dict[str, Any]] = []
    for gpu_count, no_packing_mbs in baseline_shapes:
        request = _packing_request(
            scenario,
            no_packing_mbs=no_packing_mbs,
            profile_path=profile_path,
            gpu_count=gpu_count,
        )
        try:
            decision = sp.build_decision(
                request,
                policy=dict(policy),
                policy_path=policy_path,
                request_base=request_base,
            )
        except (ValueError, FileNotFoundError) as error:
            decisions.append(
                {
                    "gpu_count": gpu_count,
                    "no_packing_mbs": no_packing_mbs,
                    "policy_query_failed": True,
                    "error": str(error),
                    "packing": False,
                    "decision": "off",
                    "reason_codes": ["static_policy_query_failed"],
                }
            )
            continue
        recommendation = decision.get("recommendation") or {}
        features = decision.get("features") or {}
        decisions.append(
            {
                "gpu_count": gpu_count,
                "no_packing_mbs": no_packing_mbs,
                "policy_query_failed": False,
                "packing": bool(recommendation.get("packing")),
                "decision": recommendation.get("decision"),
                "support_status": recommendation.get("support_status"),
                "confidence": recommendation.get("confidence"),
                "shadow_candidate_packing": recommendation.get(
                    "shadow_candidate_packing"
                ),
                "reason_codes": list(recommendation.get("reason_codes") or []),
                "released_domain_mismatches": list(
                    decision.get("released_domain_mismatches") or []
                ),
                "pack_utilization": features.get("pack_utilization"),
                "mean_samples_per_pack": features.get("mean_samples_per_pack"),
                "samples_per_pack": features.get("samples_per_pack"),
                "packed_batch_geometry_v2": features.get(
                    "packed_batch_geometry_v2"
                ),
                "sequence_reduction_ratio": features.get("sequence_reduction_ratio"),
                "decision_report_sha256": decision.get("report_sha256"),
            }
        )
    return decisions


def build_candidate_space(
    scenario: Mapping[str, Any],
    *,
    capacity_bytes: int,
    profile_path: str,
    policy_path: Path = DEFAULT_POLICY_PATH,
    request_base: Path = Path("."),
    **generator_kwargs: Any,
) -> dict[str, Any]:
    """Build the unpacked candidate set plus a labelled packed shadow branch."""

    if scenario.get("packing"):
        raise ValueError(
            "pass the scenario with packing unset; this module derives both "
            "branches from the frozen static policy"
        )

    unpacked = cg.generate_candidates(
        {**scenario, "packing": False},
        capacity_bytes=capacity_bytes,
        **generator_kwargs,
    )
    for candidate in unpacked["candidates"]:
        candidate["candidate_branch"] = BRANCH_UNPACKED
        candidate["eligibility"] = "rankable"

    # One policy query per distinct (gpu_count, mbs) baseline the unpacked branch
    # actually produced.  Packing is judged against real baselines only.
    baseline_shapes = sorted(
        {
            (candidate["gpu_count"], candidate["physical_mbs"])
            for candidate in unpacked["candidates"]
        }
    )
    policy = sp.load_policy(policy_path)
    decisions = _decide_packing(
        scenario,
        baseline_shapes=baseline_shapes,
        policy=policy,
        policy_path=policy_path,
        profile_path=profile_path,
        request_base=request_base,
    )

    enabled_shapes = {
        (row["gpu_count"], row["no_packing_mbs"])
        for row in decisions
        if row["packing"]
    }
    shadow_shapes = {
        (row["gpu_count"], row["no_packing_mbs"])
        for row in decisions
        if not row["packing"] and row.get("shadow_candidate_packing")
    }

    packed: list[dict[str, Any]] = []
    if enabled_shapes or shadow_shapes:
        # Packing fixes the physical micro batch at one, so the packed branch has
        # one candidate per (gpu_count, zero_stage, gc) -- the replaced baseline
        # mbs is recorded so the pair can be compared later.
        generated = cg.generate_candidates(
            {**scenario, "packing": True},
            capacity_bytes=capacity_bytes,
            **generator_kwargs,
        )
        for candidate in generated["candidates"]:
            gpu_count = candidate["gpu_count"]
            replaced = sorted(
                mbs
                for (count, mbs) in (enabled_shapes | shadow_shapes)
                if count == gpu_count
            )
            if not replaced:
                continue
            candidate["candidate_branch"] = BRANCH_PACKED_SHADOW
            # Asymmetric on purpose: the frozen heads cannot price this branch.
            candidate["eligibility"] = "shadow_only"
            candidate["shadow_reason"] = PACKED_SHADOW_REASON
            candidate["replaces_no_packing_mbs"] = replaced
            candidate["policy_enabled"] = any(
                (gpu_count, mbs) in enabled_shapes for mbs in replaced
            )
            packed.append(candidate)

    branch_counts = {
        BRANCH_UNPACKED: len(unpacked["candidates"]),
        BRANCH_PACKED_SHADOW: len(packed),
    }
    decision_summary = {
        "queried_baseline_shapes": len(baseline_shapes),
        "shapes_with_packing_on": sorted(enabled_shapes),
        "shapes_with_shadow_candidate": sorted(shadow_shapes),
        "distinct_decisions": sorted(
            {str(row.get("decision")) for row in decisions}
        ),
        "distinct_reason_codes": sorted(
            {code for row in decisions for code in row.get("reason_codes", [])}
        ),
    }

    report = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "comparison_group": unpacked["comparison_group"],
        "scenario": unpacked["scenario"],
        "policy_binding": {
            "path": str(policy_path),
            "file_sha256": sha256_file(policy_path),
            "policy_id": policy.get("policy_id"),
            "status": policy.get("status"),
            "packed_physical_mbs": policy.get("packed_physical_mbs"),
        },
        "profile_binding": {"path": profile_path},
        "guarantees": {
            "derives_packing_decision_itself": False,
            "overrides_policy_off": False,
            "predicts_memory": False,
            "ranks_candidates": False,
            "admits_packed_candidates": False,
            "creates_gpu_queue": False,
        },
        "branch_policy": {
            BRANCH_UNPACKED: "rankable by the frozen memory gate and ranking head",
            BRANCH_PACKED_SHADOW: (
                "emitted for visibility and future pairing only; not admissible "
                f"because {PACKED_SHADOW_REASON}"
            ),
        },
        "packing_decisions": decisions,
        "decision_summary": decision_summary,
        "branch_counts": branch_counts,
        "rankable_candidates": unpacked["candidates"],
        "shadow_candidates": packed,
        "statically_rejected": unpacked["statically_rejected"],
        "gpu_counts_with_at_least_two_rankable": unpacked[
            "gpu_counts_with_at_least_two_candidates"
        ],
        "next_step": (
            "Send only 'rankable_candidates' to the frozen predictor. The shadow "
            "branch becomes admissible after packing-aware memory and throughput "
            "calibration passes its own acceptance."
        ),
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="scenario JSON")
    parser.add_argument("--profile", required=True, help="token-length profile JSONL")
    parser.add_argument("--hardware", type=Path, default=None)
    parser.add_argument("--capacity-bytes", type=int, default=None)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    capacity_bytes = args.capacity_bytes
    if capacity_bytes is None:
        if args.hardware is None:
            raise SystemExit("pass --capacity-bytes or --hardware")
        capacity_bytes = int(
            read_json(args.hardware)["memory_bytes_reported_by_torch"]
        )

    scenario = read_json(args.request)
    if isinstance(scenario, Mapping) and "scenarios" in scenario:
        scenario = scenario["scenarios"][0]

    report = build_candidate_space(
        scenario,
        capacity_bytes=capacity_bytes,
        profile_path=args.profile,
        policy_path=args.policy,
    )
    if args.output is not None:
        write_json(args.output, report)

    print(f"group: {report['comparison_group']}")
    print(f"branch counts: {report['branch_counts']}")
    summary = report["decision_summary"]
    print(f"baseline shapes queried: {summary['queried_baseline_shapes']}")
    print(f"decisions: {summary['distinct_decisions']}")
    print(f"reason codes: {summary['distinct_reason_codes']}")
    print(f"packing-on shapes: {summary['shapes_with_packing_on']}")
    print(
        "gpu counts with >=2 rankable: "
        f"{report['gpu_counts_with_at_least_two_rankable']}"
    )
    if args.output is not None:
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
