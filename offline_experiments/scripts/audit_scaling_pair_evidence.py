#!/usr/bin/env python
"""Audit how much adjacent-doubling scaling evidence the existing runs contain.

The calibration report only surfaced four scaling pairs, all of them from the
``longcontext_32768`` LoRA scenarios.  That is an artefact of where the pairs are
counted, not of what was measured: ``h800_theory_calibration._scaling_pairs``
only reports a pair when both endpoints happen to fall inside the same
leave-one-scenario-out test fold, and it keeps a single best candidate per GPU
count.

This module recomputes the pairs directly from the canonical observations so the
cross-card ratio evidence can be judged on scenario coverage rather than on fold
placement.  It is a read-only diagnostic:

* it never writes to a frozen artefact,
* it never fits or publishes a coefficient,
* it does not create a GPU queue or an approval,
* OOM and non-success endpoints are reported as unusable, never imputed.

The ratios produced here are measured point estimates from single runs.  They are
*not* a publishable scale-out claim: that additionally requires a conservative
lower bound and a fresh prospective measurement, per the release bars.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, read_json, sha256_file, write_json

SCHEMA = "sft_scaling_pair_evidence_audit/v1"
IMPLEMENTATION_VERSION = "sft_scaling_pair_audit_impl/2026-08-01.v1"

CANONICAL_OBSERVATIONS = ARTIFACT_DIR / "canonical_h800_observations.jsonl"
CALIBRATION_REPORT = ARTIFACT_DIR / "h800_theory_calibration.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "scaling_pair_evidence_audit.json"

# The scenario is the unit that must stay fixed across a doubling.  GBS, cutoff,
# packing and dtype are part of the user's training semantics; changing any of
# them would make the ratio meaningless (strong scaling only).
SCENARIO_FIELDS = (
    "model_id",
    "train_type",
    "dataset_id",
    "target_gbs",
    "cutoff_len",
    "packing",
)

# Within a scenario these may legally differ between the two endpoints, because
# they are exactly what the planner is allowed to re-tune when the card count
# changes.
ENDPOINT_TUNABLE_FIELDS = ("gpu_count", "mbs", "zero", "gc")

RATE_PRIMARY = "effective_tokens_per_second"
RATE_SECONDARY = "computed_tokens_per_second"

JOB_PREFIXES = ("scale-",)


def _as_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _positive(value: Any) -> float | None:
    result = _as_float(value)
    if result is None or result <= 0.0:
        return None
    return result


def load_scaling_observations(
    path: Path = CANONICAL_OBSERVATIONS,
    *,
    prefixes: Sequence[str] = JOB_PREFIXES,
) -> list[dict[str, Any]]:
    """Read canonical observations belonging to the scaling campaign."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            job = ((record.get("configuration") or {}).get("job")) or {}
            job_id = str(job.get("job_id") or "")
            if not any(job_id.startswith(prefix) for prefix in prefixes):
                continue
            rows.append(record)
    return rows


def _endpoint(record: Mapping[str, Any]) -> dict[str, Any]:
    job = ((record.get("configuration") or {}).get("job")) or {}
    outcome = record.get("outcome") or {}
    measurements = record.get("measurements") or {}
    rates = measurements.get("rates") or {}
    memory = measurements.get("memory") or {}
    return {
        "job_id": job.get("job_id"),
        "observation_id": record.get("observation_id"),
        "scenario": {field: job.get(field) for field in SCENARIO_FIELDS},
        "configuration": {field: job.get(field) for field in ENDPOINT_TUNABLE_FIELDS},
        "gpu_count": _as_int(job.get("gpu_count")),
        "outcome_class": outcome.get("class"),
        "effective_tokens_per_second": _positive(rates.get(RATE_PRIMARY)),
        "computed_tokens_per_second": _positive(rates.get(RATE_SECONDARY)),
        "mean_step_seconds": _positive(measurements.get("mean_step_seconds")),
        "measured_step_count": _as_int(measurements.get("measured_step_count")),
        "max_reserved_bytes": _as_int(memory.get("max_reserved_bytes")),
        "memory_observed_not_imputed": bool(
            memory.get("values_are_observed_not_imputed")
        ),
        "calibration_evidence_eligible": bool(
            (record.get("fingerprint") or {}).get("calibration_evidence_eligible")
        ),
    }


def _scenario_key(scenario: Mapping[str, Any]) -> str:
    return json.dumps(
        {field: scenario.get(field) for field in SCENARIO_FIELDS},
        sort_keys=True,
        ensure_ascii=False,
    )


def group_endpoints(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Group endpoints by scenario, then by GPU count.

    Repeats of the same ``(scenario, gpu_count, configuration)`` are kept
    together so a ratio can report how many independent runs back each endpoint.
    """

    scenarios: dict[str, dict[str, Any]] = {}
    for record in records:
        endpoint = _endpoint(record)
        gpu_count = endpoint["gpu_count"]
        if gpu_count is None:
            continue
        key = _scenario_key(endpoint["scenario"])
        bucket = scenarios.setdefault(
            key,
            {"scenario": endpoint["scenario"], "by_gpu_count": defaultdict(list)},
        )
        bucket["by_gpu_count"][gpu_count].append(endpoint)
    for bucket in scenarios.values():
        bucket["by_gpu_count"] = {
            gpu_count: sorted(items, key=lambda item: str(item["job_id"]))
            for gpu_count, items in sorted(bucket["by_gpu_count"].items())
        }
    return scenarios


def _select_endpoint(endpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one GPU count of one scenario.

    A GPU count is usable for a ratio only when at least one run succeeded with
    an observed rate.  When several runs succeeded the median rate is used and
    the spread is reported, so a ratio built on a noisy endpoint is visible
    rather than hidden behind a single point estimate.
    """

    successes = [
        item
        for item in endpoints
        if item["outcome_class"] == "success"
        and item[RATE_PRIMARY] is not None
    ]
    oom = [item for item in endpoints if item["outcome_class"] == "oom"]
    other = [
        item
        for item in endpoints
        if item["outcome_class"] not in {"success", "oom"}
    ]
    summary: dict[str, Any] = {
        "runs": len(endpoints),
        "successful_runs": len(successes),
        "oom_runs": len(oom),
        "other_runs": len(other),
        "outcome_classes": sorted(
            {str(item["outcome_class"]) for item in endpoints}
        ),
        "usable_for_ratio": bool(successes),
    }
    if not successes:
        summary["unusable_reason"] = (
            "all_runs_oom"
            if oom and not other
            else "no_successful_run_with_observed_rate"
        )
        return summary

    rates = sorted(item[RATE_PRIMARY] for item in successes)
    middle = len(rates) // 2
    median = (
        rates[middle]
        if len(rates) % 2 == 1
        else 0.5 * (rates[middle - 1] + rates[middle])
    )
    summary.update(
        {
            "effective_tokens_per_second_median": median,
            "effective_tokens_per_second_min": rates[0],
            "effective_tokens_per_second_max": rates[-1],
            "relative_spread": (
                (rates[-1] - rates[0]) / rates[0] if rates[0] else None
            ),
            "single_run_point_estimate": len(rates) == 1,
            "jobs": [item["job_id"] for item in successes],
            "configurations": [item["configuration"] for item in successes],
            "calibration_evidence_eligible": all(
                item["calibration_evidence_eligible"] for item in successes
            ),
        }
    )
    return summary


def build_pairs(scenarios: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build every adjacent-doubling pair the measured runs can support."""

    pairs: list[dict[str, Any]] = []
    for key, bucket in sorted(scenarios.items()):
        by_gpu = bucket["by_gpu_count"]
        summaries = {
            gpu_count: _select_endpoint(endpoints)
            for gpu_count, endpoints in by_gpu.items()
        }
        for gpu_count in sorted(by_gpu):
            doubled = 2 * gpu_count
            if doubled not in by_gpu:
                continue
            low = summaries[gpu_count]
            high = summaries[doubled]
            pair: dict[str, Any] = {
                "scenario_key": key,
                "scenario": bucket["scenario"],
                "from_gpus": gpu_count,
                "to_gpus": doubled,
                "from_endpoint": low,
                "to_endpoint": high,
            }
            if not (low["usable_for_ratio"] and high["usable_for_ratio"]):
                pair["ratio_available"] = False
                pair["blocked_reason"] = (
                    f"from={low.get('unusable_reason') or 'ok'};"
                    f"to={high.get('unusable_reason') or 'ok'}"
                )
                pairs.append(pair)
                continue
            low_rate = low["effective_tokens_per_second_median"]
            high_rate = high["effective_tokens_per_second_median"]
            ratio = high_rate / low_rate
            pair.update(
                {
                    "ratio_available": True,
                    "measured_ratio_point_estimate": ratio,
                    "clears_1p8_point_estimate": ratio >= 1.8,
                    "both_endpoints_single_run": bool(
                        low["single_run_point_estimate"]
                        and high["single_run_point_estimate"]
                    ),
                    "zero_stage_changed": (
                        low["configurations"][0].get("zero")
                        != high["configurations"][0].get("zero")
                    ),
                    "mbs_changed": (
                        low["configurations"][0].get("mbs")
                        != high["configurations"][0].get("mbs")
                    ),
                    "gc_changed": (
                        low["configurations"][0].get("gc")
                        != high["configurations"][0].get("gc")
                    ),
                }
            )
            # A doubling that also switches ZeRO stage mixes the scale-out effect
            # with the cost of introducing sharded communication.  The ratio is
            # still recorded, but it must not be read as a clean scaling
            # measurement.
            pair["clean_scaling_measurement"] = not pair["zero_stage_changed"]
            pairs.append(pair)
    return pairs


def _calibration_reported_pairs(path: Path = CALIBRATION_REPORT) -> dict[str, Any]:
    """Count the scaling pairs the existing calibration report surfaced."""

    if not path.exists():
        return {"available": False, "reason": "calibration_report_missing"}
    report = read_json(path)
    folds = ((report.get("global_scenario_loocv") or {}).get("folds")) or []
    unique: dict[tuple[int, int, float], dict[str, Any]] = {}
    training_pair_counts: set[int] = set()
    claims = 0
    valid_claims = 0
    for fold in folds:
        scaling = (
            ((fold.get("throughput") or {}).get("primary") or {}).get("scaling")
        ) or {}
        training = _as_int(scaling.get("training_pairs"))
        if training is not None:
            training_pair_counts.add(training)
        claims += _as_int(scaling.get("claims")) or 0
        valid_claims += _as_int(scaling.get("valid_claims")) or 0
        for pair in scaling.get("test_pairs") or []:
            observed = _as_float(pair.get("observed_ratio"))
            from_gpus = _as_int(pair.get("from_gpus"))
            to_gpus = _as_int(pair.get("to_gpus"))
            if observed is None or from_gpus is None or to_gpus is None:
                continue
            unique[(from_gpus, to_gpus, round(observed, 6))] = {
                "from_gpus": from_gpus,
                "to_gpus": to_gpus,
                "observed_ratio": observed,
                "predicted_center_ratio": _as_float(
                    pair.get("predicted_center_ratio")
                ),
                "conservative_ratio_lower": _as_float(
                    pair.get("conservative_ratio_lower")
                ),
                "measured_clears_1_8x": bool(pair.get("measured_clears_1_8x")),
            }
    return {
        "available": True,
        "report_path": str(path),
        "report_sha256": sha256_file(path),
        "outer_folds": len(folds),
        "distinct_training_pair_counts": sorted(training_pair_counts),
        "unique_test_pairs": sorted(
            unique.values(), key=lambda item: (item["from_gpus"], item["observed_ratio"])
        ),
        "unique_test_pair_count": len(unique),
        "claims": claims,
        "valid_claims": valid_claims,
        "conservative_lower_bound_identifiable": any(
            item["conservative_ratio_lower"] is not None for item in unique.values()
        ),
    }


def _coverage(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usable = [pair for pair in pairs if pair.get("ratio_available")]
    by_dataset: dict[str, int] = defaultdict(int)
    by_train_type: dict[str, int] = defaultdict(int)
    by_model: dict[str, int] = defaultdict(int)
    by_transition: dict[str, int] = defaultdict(int)
    for pair in usable:
        scenario = pair["scenario"]
        by_dataset[str(scenario.get("dataset_id"))] += 1
        by_train_type[str(scenario.get("train_type"))] += 1
        by_model[str(scenario.get("model_id"))] += 1
        by_transition[f"{pair['from_gpus']}->{pair['to_gpus']}"] += 1
    return {
        "usable_pairs": len(usable),
        "by_dataset_id": dict(sorted(by_dataset.items())),
        "by_train_type": dict(sorted(by_train_type.items())),
        "by_model_id": dict(sorted(by_model.items())),
        "by_transition": dict(sorted(by_transition.items())),
    }


def _ratio_stats(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usable = [pair for pair in pairs if pair.get("ratio_available")]
    clean = [pair for pair in usable if pair.get("clean_scaling_measurement")]
    ratios = sorted(pair["measured_ratio_point_estimate"] for pair in usable)
    clean_ratios = sorted(
        pair["measured_ratio_point_estimate"] for pair in clean
    )

    def _describe(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0}
        middle = len(values) // 2
        median = (
            values[middle]
            if len(values) % 2 == 1
            else 0.5 * (values[middle - 1] + values[middle])
        )
        return {
            "count": len(values),
            "min": values[0],
            "median": median,
            "max": values[-1],
            "clearing_1p8": sum(1 for value in values if value >= 1.8),
        }

    per_dataset: dict[str, list[float]] = defaultdict(list)
    per_transition: dict[str, list[float]] = defaultdict(list)
    for pair in clean:
        per_dataset[str(pair["scenario"].get("dataset_id"))].append(
            pair["measured_ratio_point_estimate"]
        )
        per_transition[f"{pair['from_gpus']}->{pair['to_gpus']}"].append(
            pair["measured_ratio_point_estimate"]
        )
    return {
        "all_usable": _describe(ratios),
        "clean_only": _describe(clean_ratios),
        "clean_by_dataset_id": {
            key: _describe(sorted(values)) for key, values in sorted(per_dataset.items())
        },
        "clean_by_transition": {
            key: _describe(sorted(values))
            for key, values in sorted(per_transition.items())
        },
    }


def build_report(
    *,
    observations_path: Path = CANONICAL_OBSERVATIONS,
    calibration_path: Path = CALIBRATION_REPORT,
) -> dict[str, Any]:
    records = load_scaling_observations(observations_path)
    scenarios = group_endpoints(records)
    pairs = build_pairs(scenarios)
    reported = _calibration_reported_pairs(calibration_path)
    coverage = _coverage(pairs)
    stats = _ratio_stats(pairs)

    usable = coverage["usable_pairs"]
    surfaced = reported.get("unique_test_pair_count") if reported.get("available") else None
    findings: list[str] = []
    if surfaced is not None and usable > surfaced:
        findings.append(
            "measured_runs_support_more_pairs_than_calibration_surfaced"
        )
    if any(
        pair.get("ratio_available") and pair.get("zero_stage_changed")
        for pair in pairs
    ):
        findings.append("some_pairs_change_zero_stage_and_are_not_clean_scaling")
    if all(
        pair.get("both_endpoints_single_run")
        for pair in pairs
        if pair.get("ratio_available")
    ):
        findings.append("every_usable_pair_rests_on_single_run_endpoints")
    if reported.get("available") and not reported.get(
        "conservative_lower_bound_identifiable"
    ):
        findings.append("conservative_ratio_lower_bound_not_identifiable")

    return {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "purpose": (
            "Recount adjacent-doubling scaling evidence from measured runs so "
            "cross-card experiment planning is driven by scenario coverage "
            "rather than by leave-one-scenario-out fold placement."
        ),
        "status": "diagnostic_only",
        "guarantees": {
            "creates_gpu_queue": False,
            "proposes_campaign": False,
            "fits_or_publishes_coefficients": False,
            "mutates_frozen_artifacts": False,
            "ratios_are_publishable_scale_out_claims": False,
        },
        "ratio_metric": {
            "primary": RATE_PRIMARY,
            "secondary_recorded": RATE_SECONDARY,
            "note": (
                "Point estimates from measured runs. A publishable scale-out "
                "claim additionally requires a conservative predicted lower "
                "bound and a fresh prospective measured lower bound >= 1.8."
            ),
        },
        "sources": {
            "observations_path": str(observations_path),
            "observations_sha256": sha256_file(observations_path),
            "scaling_observations": len(records),
        },
        "calibration_surfaced": reported,
        "scenario_count": len(scenarios),
        "pairs": pairs,
        "coverage": coverage,
        "ratio_statistics": stats,
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        type=Path,
        default=CANONICAL_OBSERVATIONS,
        help="canonical observations JSONL",
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=CALIBRATION_REPORT,
        help="existing calibration report used only to count surfaced pairs",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report(
        observations_path=args.observations,
        calibration_path=args.calibration_report,
    )
    write_json(args.output, report)

    coverage = report["coverage"]
    surfaced = report["calibration_surfaced"].get("unique_test_pair_count")
    print(f"scaling observations: {report['sources']['scaling_observations']}")
    print(f"scenarios: {report['scenario_count']}")
    print(f"usable adjacent pairs: {coverage['usable_pairs']}")
    print(f"calibration surfaced: {surfaced}")
    print(f"by dataset: {coverage['by_dataset_id']}")
    print(f"by train_type: {coverage['by_train_type']}")
    clean = report["ratio_statistics"]["clean_only"]
    print(f"clean ratios: {clean}")
    for key, value in report["ratio_statistics"]["clean_by_dataset_id"].items():
        print(f"  {key}: {value}")
    for finding in report["findings"]:
        print(f"finding: {finding}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
