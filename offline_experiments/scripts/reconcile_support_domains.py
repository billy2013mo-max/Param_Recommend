#!/usr/bin/env python
"""Reconcile the memory/ranking support domain with the packing release domain.

The plan flags an unresolved conflict: the predictor's ``SUPPORTED_CUTOFF_LENS``
omits 16384 while the frozen packing policy releases cutoffs up to 16384, and the
packing release domain is narrower on model and train type.  Left alone, this
produces requests that one component treats as in-domain and the other refuses,
which is how a recommendation ends up silently inconsistent.

This module does not resolve the conflict by editing either domain.  Both are
deliberate: the predictor grid reflects the cutoffs its frozen memory model was
calibrated on, and the packing domain reflects the ten paired points that
calibrated the packing gates.  Widening either one without evidence would be
exactly the kind of unbacked extrapolation the plan forbids.

What it does instead is compute the reconciliation explicitly: the intersection
where both components agree, the asymmetric regions and what each implies, and
the concrete evidence each widening would require.  A reviewer can then decide,
with the cost visible, rather than discovering the mismatch from a confusing
prediction.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, read_json, sha256_file, sha256_json, write_json
from h800_physical_v4b_predictor import (
    H800_SUPPORTED_MODEL_IDS,
    SUPPORTED_CUTOFF_LENS,
    SUPPORTED_GPU_COUNTS,
    SUPPORTED_TARGET_GBS,
)

SCHEMA = "sft_support_domain_reconciliation/v1"
IMPLEMENTATION_VERSION = "sft_support_domain_reconciliation_impl/2026-08-01.v1"

DEFAULT_POLICY = ARTIFACT_DIR / "static_packing_policy_v1.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "support_domain_reconciliation_v1.json"


def _packing_cutoffs(policy: Mapping[str, Any]) -> dict[str, Any]:
    domain = policy.get("released_enablement_domain") or {}
    window = domain.get("cutoff_len") or {}
    return {
        "minimum": window.get("minimum"),
        "maximum": window.get("maximum"),
        "expressed_as": "inclusive_range",
    }


def build_reconciliation(
    *, policy_path: Path = DEFAULT_POLICY
) -> dict[str, Any]:
    """Compare the two domains dimension by dimension."""

    policy = read_json(policy_path)
    packing_domain = policy.get("released_enablement_domain") or {}
    packing_cutoffs = _packing_cutoffs(policy)

    predictor_cutoffs = sorted(SUPPORTED_CUTOFF_LENS)
    lo = packing_cutoffs["minimum"]
    hi = packing_cutoffs["maximum"]

    # The packing side is a range, the predictor side an enumerated grid, so the
    # comparison is grid-vs-range rather than set-vs-set.
    in_both = [
        value
        for value in predictor_cutoffs
        if lo is not None and hi is not None and lo <= value <= hi
    ]
    predictor_only = [
        value
        for value in predictor_cutoffs
        if hi is not None and value > hi
    ]
    packing_range_not_on_grid = (
        [hi] if hi is not None and hi not in SUPPORTED_CUTOFF_LENS else []
    )

    packing_models = sorted(str(value) for value in packing_domain.get("model_ids") or [])
    predictor_models = sorted(H800_SUPPORTED_MODEL_IDS)
    packing_train_types = sorted(
        str(value) for value in packing_domain.get("train_types") or []
    )

    dimensions = [
        {
            "dimension": "cutoff_len",
            "predictor": predictor_cutoffs,
            "packing": packing_cutoffs,
            "agree_on": in_both,
            "predictor_only": predictor_only,
            "packing_boundary_absent_from_predictor_grid": packing_range_not_on_grid,
            "implication": (
                "A cutoff of 16384 is inside the packing release range but is "
                "not an enumerated predictor cutoff, so a 16384 request can "
                "receive a packing decision while the memory and ranking heads "
                "refuse it as out-of-domain. A 32768 request is the mirror case: "
                "the predictor accepts it, packing does not."
            ),
        },
        {
            "dimension": "model_id",
            "predictor": predictor_models,
            "packing": packing_models,
            "agree_on": sorted(set(predictor_models) & set(packing_models)),
            "predictor_only": sorted(set(predictor_models) - set(packing_models)),
            "packing_only": sorted(set(packing_models) - set(predictor_models)),
            "implication": (
                "Packing is released for a strict subset of the models the "
                "predictor ranks, so for the remaining models packing must stay "
                "off rather than be inferred from a nearby model."
            ),
        },
        {
            "dimension": "train_type",
            "predictor": ["full", "lora"],
            "packing": packing_train_types,
            "agree_on": sorted({"full", "lora"} & set(packing_train_types)),
            "predictor_only": sorted({"full", "lora"} - set(packing_train_types)),
            "implication": (
                "Full fine-tuning has no released packing evidence at all, so a "
                "Full request may be ranked but never packed."
            ),
        },
        {
            "dimension": "gpu_count",
            "predictor": sorted(SUPPORTED_GPU_COUNTS),
            "packing": "not constrained by the packing policy",
            "agree_on": sorted(SUPPORTED_GPU_COUNTS),
            "implication": "no conflict",
        },
        {
            "dimension": "target_gbs",
            "predictor": sorted(SUPPORTED_TARGET_GBS),
            "packing": "not constrained by the packing policy",
            "agree_on": sorted(SUPPORTED_TARGET_GBS),
            "implication": "no conflict",
        },
    ]

    conflicts = [
        item
        for item in dimensions
        if item["implication"] != "no conflict"
        and (
            item.get("predictor_only")
            or item.get("packing_only")
            or item.get("packing_boundary_absent_from_predictor_grid")
        )
    ]

    # The safe operating region is where both components agree, and it is what a
    # product should treat as jointly supported today.
    joint_domain = {
        "cutoff_len": in_both,
        "model_id": sorted(set(predictor_models) & set(packing_models)),
        "train_type": sorted({"full", "lora"} & set(packing_train_types)),
        "gpu_count": sorted(SUPPORTED_GPU_COUNTS),
        "target_gbs": sorted(SUPPORTED_TARGET_GBS),
        "note": (
            "Inside this region a request can receive both a packing decision "
            "and a memory/ranking verdict. Outside it, at least one component "
            "must fail closed."
        ),
    }

    resolution_options = [
        {
            "option": "keep_both_and_intersect",
            "changes_evidence_requirements": False,
            "effect": (
                "Product consumes the joint domain only; requests outside it get "
                "an explicit partial answer (ranked but packing off, or refused)."
            ),
            "recommended": True,
            "rationale": (
                "No new evidence needed and nothing is claimed beyond what was "
                "calibrated. It only makes the existing asymmetry explicit."
            ),
        },
        {
            "option": "add_16384_to_the_predictor_grid",
            "changes_evidence_requirements": True,
            "required_evidence": (
                "memory boundary and ranking evidence at cutoff 16384 for the "
                "models concerned; the frozen memory model was not calibrated "
                "there"
            ),
            "recommended": False,
        },
        {
            "option": "widen_the_packing_release_domain",
            "changes_evidence_requirements": True,
            "required_evidence": (
                "ABBA paired points for the additional models and for Full "
                "fine-tuning; the current release rests on ten resubstitution "
                "points with zero independent holdout"
            ),
            "recommended": False,
        },
    ]

    report = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "reconciliation_only",
        "policy_binding": {
            "path": str(policy_path),
            "sha256": sha256_file(policy_path),
            "policy_id": policy.get("policy_id"),
            "policy_status": policy.get("status"),
        },
        "guarantees": {
            "modifies_predictor_domain": False,
            "modifies_packing_domain": False,
            "widens_any_support_domain": False,
            "creates_gpu_queue": False,
        },
        "dimensions": dimensions,
        "conflict_count": len(conflicts),
        "conflicting_dimensions": [item["dimension"] for item in conflicts],
        "joint_supported_domain": joint_domain,
        "resolution_options": resolution_options,
        "required_next_step": (
            "Adopt the joint domain in the product layer, or approve one of the "
            "widening options together with the experiments it requires."
        ),
    }
    report["report_sha256"] = sha256_json(report)
    return report


def classify_request(
    *,
    cutoff_len: int,
    model_id: str,
    train_type: str,
    reconciliation: Mapping[str, Any],
) -> dict[str, Any]:
    """Say which components can serve a request, and why not the others."""

    joint = reconciliation["joint_supported_domain"]
    ranked = (
        cutoff_len in set(SUPPORTED_CUTOFF_LENS)
        and model_id in H800_SUPPORTED_MODEL_IDS
    )
    packable = (
        cutoff_len in set(joint["cutoff_len"])
        and model_id in set(joint["model_id"])
        and train_type in set(joint["train_type"])
    )
    reasons: list[str] = []
    if not ranked:
        reasons.append("outside_predictor_support_grid")
    if not packable:
        reasons.append("outside_packing_release_domain")
    return {
        "cutoff_len": cutoff_len,
        "model_id": model_id,
        "train_type": train_type,
        "memory_and_ranking_available": ranked,
        "packing_decision_available": packable,
        "jointly_supported": ranked and packable,
        "reasons": reasons,
    }


def _examples(reconciliation: Mapping[str, Any]) -> list[dict[str, Any]]:
    probes: Sequence[tuple[int, str, str]] = (
        (4096, "qwen3_8b", "lora"),
        (16384, "qwen3_8b", "lora"),
        (32768, "qwen3_8b", "lora"),
        (4096, "qwen3_8b", "full"),
        (4096, "qwen3_1p7b", "lora"),
    )
    return [
        classify_request(
            cutoff_len=cutoff,
            model_id=model_id,
            train_type=train_type,
            reconciliation=reconciliation,
        )
        for cutoff, model_id, train_type in probes
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_reconciliation(policy_path=args.policy)
    report["worked_examples"] = _examples(report)
    report["report_sha256"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    write_json(args.output, report)

    print(f"conflicts: {report['conflict_count']} "
          f"{report['conflicting_dimensions']}")
    for item in report["dimensions"]:
        if item["implication"] == "no conflict":
            continue
        print(f"  {item['dimension']}:")
        print(f"    predictor: {item['predictor']}")
        print(f"    packing:   {item['packing']}")
        print(f"    agree on:  {item['agree_on']}")
    print("--- joint supported domain ---")
    joint = report["joint_supported_domain"]
    for key in ("cutoff_len", "model_id", "train_type"):
        print(f"  {key}: {joint[key]}")
    print("--- worked examples ---")
    for example in report["worked_examples"]:
        print(
            f"  cutoff={example['cutoff_len']:>5} {example['model_id']}"
            f"/{example['train_type']}: rank={example['memory_and_ranking_available']}"
            f" pack={example['packing_decision_available']}"
            f" joint={example['jointly_supported']}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
