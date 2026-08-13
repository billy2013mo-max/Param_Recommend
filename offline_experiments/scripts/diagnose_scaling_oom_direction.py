#!/usr/bin/env python
"""Classify scenarios whose OOM pattern does not improve as GPUs are added.

A memory model is expected to become *more* permissive as the card count grows:
sharding moves optimizer state, gradients and (under ZeRO-3) parameters off each
rank.  When a scenario succeeds on ``N`` GPUs but OOMs on ``2N``, that
expectation is violated, and the record must not be fed to a boundary
calibration before the cause is known.

There are three very different causes, and conflating them corrupts the
right-censored memory evidence:

``configuration_confounded``
    The endpoints do not hold the tunables fixed -- typically the larger card
    count also raised ``mbs``.  Per-GPU activation grew faster than sharding
    saved, so the OOM is *physically expected* and is legitimate boundary
    evidence for its own configuration.  It is not an anomaly at all.

``sharding_ineffective``
    The tunables are fixed and the stage should have helped, but the sharded
    quantity is too small to matter -- most importantly LoRA under ZeRO-2,
    where only the tiny adapter optimizer state is sharded while the full
    base weights stay replicated on every rank.

``label_conflict_candidate``
    Nothing in the calibration keys explains it.  This is the pattern
    :mod:`h800_evidence_gap` already flags for identical cells: triage as
    software or infrastructure failure first, per the failure taxonomy.  Never
    silently keep it as a memory boundary label.

This module is read-only: it fits nothing, publishes nothing, mutates no frozen
artefact and creates no GPU queue.  It never invents a peak for an OOM -- an OOM
without an observed peak stays right-censored.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, sha256_file, write_json

SCHEMA = "sft_scaling_oom_direction_diagnosis/v1"
IMPLEMENTATION_VERSION = "sft_scaling_oom_direction_impl/2026-08-01.v1"

CANONICAL_OBSERVATIONS = ARTIFACT_DIR / "canonical_h800_observations.jsonl"
DEFAULT_OUTPUT = ARTIFACT_DIR / "scaling_oom_direction_diagnosis.json"

SCENARIO_FIELDS = (
    "model_id",
    "train_type",
    "dataset_id",
    "target_gbs",
    "cutoff_len",
    "packing",
)

# Under ZeRO-2 only gradients and optimizer state are sharded.  For LoRA the
# trainable set is the adapter alone, so the sharded quantity is a rounding error
# next to the replicated base weights.  Adding cards therefore buys almost no
# per-rank memory while a larger mbs costs activation linearly.
LOW_YIELD_SHARDING = {("lora", "zero2"), ("lora", "zero1"), ("lora", "none")}

CLASS_CONFIG_CONFOUNDED = "configuration_confounded"
CLASS_SHARDING_INEFFECTIVE = "sharding_ineffective"
CLASS_LABEL_CONFLICT = "label_conflict_candidate"


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _normalized_zero(value: Any) -> str:
    text = str(value or "none").strip().lower()
    return {"0": "none", "null": "none", "": "none"}.get(text, text)


def load_observations(
    path: Path = CANONICAL_OBSERVATIONS,
    *,
    job_prefixes: Sequence[str] = ("scale-",),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            job = ((record.get("configuration") or {}).get("job")) or {}
            job_id = str(job.get("job_id") or "")
            if job_prefixes and not any(
                job_id.startswith(prefix) for prefix in job_prefixes
            ):
                continue
            rows.append(record)
    return rows


def _endpoint(record: Mapping[str, Any]) -> dict[str, Any]:
    job = ((record.get("configuration") or {}).get("job")) or {}
    outcome = record.get("outcome") or {}
    memory = ((record.get("measurements") or {}).get("memory")) or {}
    censoring = record.get("censoring") or {}
    return {
        "job_id": job.get("job_id"),
        "scenario": {field: job.get(field) for field in SCENARIO_FIELDS},
        "gpu_count": _as_int(job.get("gpu_count")),
        "mbs": _as_int(job.get("mbs")),
        "zero": _normalized_zero(job.get("zero")),
        "gc": bool(job.get("gc")),
        "outcome_class": str(outcome.get("class") or "").strip().lower(),
        "max_reserved_bytes": _as_int(memory.get("max_reserved_bytes")),
        "device_capacity_bytes": _as_int(
            censoring.get("device_capacity_bytes_reported_in_error")
        ),
        "free_bytes_at_failure": _as_int(censoring.get("free_bytes_reported_in_error")),
        # An OOM peak is never imputed; the canonical export keeps it null and
        # records only the right-censoring inequality.
        "demand_peak_is_unknown_not_imputed": bool(
            censoring.get("demand_peak_is_unknown_not_imputed")
        ),
        "censoring_kind": censoring.get("kind"),
    }


def _scenario_key(scenario: Mapping[str, Any]) -> str:
    return json.dumps(
        {field: scenario.get(field) for field in SCENARIO_FIELDS},
        sort_keys=True,
        ensure_ascii=False,
    )


def _classify(
    low: Mapping[str, Any], high: Mapping[str, Any]
) -> tuple[str, list[str], str]:
    """Classify one success(N) -> OOM(2N) transition."""

    reasons: list[str] = []
    mbs_grew = (
        low["mbs"] is not None
        and high["mbs"] is not None
        and high["mbs"] > low["mbs"]
    )
    if mbs_grew:
        reasons.append(
            f"per_gpu_mbs_grew_{low['mbs']}_to_{high['mbs']}_activation_scales_with_mbs"
        )
    zero_changed = low["zero"] != high["zero"]
    if zero_changed:
        reasons.append(f"zero_stage_changed_{low['zero']}_to_{high['zero']}")
    if low["gc"] != high["gc"]:
        reasons.append(f"gradient_checkpointing_changed_{low['gc']}_to_{high['gc']}")

    train_type = str(low["scenario"].get("train_type") or "").strip().lower()
    low_yield = (train_type, high["zero"]) in LOW_YIELD_SHARDING
    if low_yield:
        reasons.append(
            f"sharding_low_yield_for_{train_type}_under_{high['zero']}"
            "_base_weights_stay_replicated"
        )

    if mbs_grew:
        # The larger card count was also given a larger micro batch, so the OOM
        # is the expected consequence of that configuration and remains valid
        # boundary evidence for the configuration actually run.
        return (
            CLASS_CONFIG_CONFOUNDED,
            reasons,
            "expected_under_larger_mbs_valid_boundary_evidence_for_its_own_config",
        )
    if low_yield or zero_changed:
        return (
            CLASS_SHARDING_INEFFECTIVE,
            reasons,
            "sharded_quantity_too_small_to_offset_per_gpu_cost",
        )
    return (
        CLASS_LABEL_CONFLICT,
        reasons or ["no_calibration_key_explains_the_direction"],
        "triage_as_software_or_infrastructure_failure_before_use",
    )


def build_transitions(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Find every success(N) -> OOM(2N) transition and classify it."""

    scenarios: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        endpoint = _endpoint(record)
        if endpoint["gpu_count"] is None:
            continue
        if endpoint["outcome_class"] not in {"success", "oom"}:
            continue
        scenarios[_scenario_key(endpoint["scenario"])][
            endpoint["gpu_count"]
        ].append(endpoint)

    transitions: list[dict[str, Any]] = []
    for key, by_gpu in sorted(scenarios.items()):
        for gpu_count in sorted(by_gpu):
            doubled = 2 * gpu_count
            if doubled not in by_gpu:
                continue
            lows = [
                item for item in by_gpu[gpu_count] if item["outcome_class"] == "success"
            ]
            highs = [item for item in by_gpu[doubled] if item["outcome_class"] == "oom"]
            if not lows or not highs:
                continue
            # Any success at N paired with an all-OOM 2N is a direction
            # violation; if 2N also has a success the scenario is not anomalous.
            if any(item["outcome_class"] == "success" for item in by_gpu[doubled]):
                continue
            low = lows[0]
            high = highs[0]
            classification, reasons, disposition = _classify(low, high)
            transitions.append(
                {
                    "scenario_key": key,
                    "scenario": low["scenario"],
                    "from_gpus": gpu_count,
                    "to_gpus": doubled,
                    "classification": classification,
                    "reasons": reasons,
                    "disposition": disposition,
                    "usable_as_boundary_evidence": classification
                    != CLASS_LABEL_CONFLICT,
                    "requires_failure_triage": classification
                    == CLASS_LABEL_CONFLICT,
                    "from_endpoint": low,
                    "to_endpoint": high,
                    "to_endpoint_oom_runs": len(highs),
                    "oom_peak_imputed": False,
                    "right_censored": all(
                        item["demand_peak_is_unknown_not_imputed"] for item in highs
                    ),
                }
            )
    return transitions


def build_report(
    *, observations_path: Path = CANONICAL_OBSERVATIONS
) -> dict[str, Any]:
    records = load_observations(observations_path)
    transitions = build_transitions(records)
    counts: dict[str, int] = defaultdict(int)
    for transition in transitions:
        counts[transition["classification"]] += 1

    findings: list[str] = []
    if counts[CLASS_CONFIG_CONFOUNDED]:
        findings.append(
            "some_direction_violations_are_explained_by_a_larger_per_gpu_mbs"
        )
    if counts[CLASS_SHARDING_INEFFECTIVE]:
        findings.append("some_scenarios_gain_almost_nothing_from_sharding")
    if counts[CLASS_LABEL_CONFLICT]:
        findings.append("unexplained_direction_violations_require_failure_triage")

    return {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "purpose": (
            "Separate physically expected OOM-on-more-GPUs cases from genuine "
            "label conflicts, so boundary calibration never ingests an "
            "unexplained direction violation."
        ),
        "status": "diagnostic_only",
        "guarantees": {
            "creates_gpu_queue": False,
            "proposes_campaign": False,
            "fits_or_publishes_coefficients": False,
            "mutates_frozen_artifacts": False,
            "imputes_oom_peaks": False,
        },
        "classification_policy": {
            CLASS_CONFIG_CONFOUNDED: (
                "larger card count also raised mbs; activation growth outpaced "
                "sharding. Valid boundary evidence for the configuration run."
            ),
            CLASS_SHARDING_INEFFECTIVE: (
                "tunables fixed but the sharded quantity is negligible, e.g. "
                "LoRA under ZeRO-2 where base weights stay replicated."
            ),
            CLASS_LABEL_CONFLICT: (
                "no calibration key explains the direction; triage as software "
                "or infrastructure failure first."
            ),
        },
        "sources": {
            "observations_path": str(observations_path),
            "observations_sha256": sha256_file(observations_path),
            "observations_considered": len(records),
        },
        "transition_count": len(transitions),
        "classification_counts": dict(sorted(counts.items())),
        "transitions": transitions,
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations", type=Path, default=CANONICAL_OBSERVATIONS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report(observations_path=args.observations)
    write_json(args.output, report)

    print(f"observations considered: {report['sources']['observations_considered']}")
    print(f"direction violations: {report['transition_count']}")
    print(f"classification: {report['classification_counts']}")
    for transition in report["transitions"]:
        scenario = transition["scenario"]
        print(
            f"  {transition['from_gpus']}->{transition['to_gpus']} "
            f"{scenario['model_id']}/{scenario['train_type']}/{scenario['dataset_id']}"
            f" => {transition['classification']}"
        )
        for reason in transition["reasons"]:
            print(f"      {reason}")
    for finding in report["findings"]:
        print(f"finding: {finding}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
