#!/usr/bin/env python3
"""Common, CPU-only acceptance calculations for fresh prospective evidence.

This module deliberately consumes normalized observations rather than loading
models or touching the scheduler.  It keeps retrospective replay and fresh
acceptance on one set of gates while retaining the raw scenario-level details
needed to diagnose a failed selector.  ``publication_allowed`` is only a
reporting result; this module never promotes an artifact or launches a job.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from cross_card_scaling import evaluate_doubling
from common import ARTIFACT_DIR, sha256_file, write_json


SCHEMA = "sft_prospective_acceptance_report/v1"
DEFAULT_MEMORY_P95_COVERAGE = 0.95
DEFAULT_FALSE_SAFE_OOM = 0
DEFAULT_TOP1_REGRET = 0.10
DEFAULT_SCALE_RATIO = 1.8
SCALE_SCHEMA = "sft_scale_out_acceptance_report/v1"
DEFAULT_SCALE_OUTPUT = ARTIFACT_DIR / "scale_out_acceptance_report_v1.json"


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(q) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _scenario_id(row: Mapping[str, Any]) -> str:
    value = row.get("scenario_id") or row.get("comparison_group")
    if value is None or not str(value).strip():
        raise ValueError("every acceptance row requires scenario_id")
    return str(value)


def _actual_safe_success(row: Mapping[str, Any]) -> bool:
    if str(row.get("outcome")) != "success":
        return False
    explicit = row.get("actual_safe_success")
    if isinstance(explicit, bool):
        return explicit
    observed = _finite(
        row.get("observed_reserved_bytes")
        if row.get("observed_reserved_bytes") is not None
        else row.get("observed_reserved_gib")
    )
    safe_limit = _finite(
        row.get("safe_limit_bytes")
        if row.get("safe_limit_bytes") is not None
        else row.get("safe_limit_gib")
    )
    if observed is None or safe_limit is None:
        return False
    return observed <= safe_limit


def evaluate_memory_acceptance(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_coverage: float = DEFAULT_MEMORY_P95_COVERAGE,
    maximum_false_safe_oom: int = DEFAULT_FALSE_SAFE_OOM,
) -> dict[str, Any]:
    """Evaluate memory admission with OOM treated as right-censored evidence.

    ``upper_covers_observed`` is preferred because it can be produced by a
    model-specific evaluator.  If absent, the function compares explicit
    ``upper_*`` and observed peak fields.  It never imputes an OOM peak.
    """

    if not 0.0 < float(minimum_coverage) <= 1.0:
        raise ValueError("minimum_coverage must be in (0, 1]")
    if int(maximum_false_safe_oom) < 0:
        raise ValueError("maximum_false_safe_oom must be non-negative")
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    normalized: list[dict[str, Any]] = []
    for source in rows:
        if not isinstance(source, Mapping):
            raise ValueError("every memory acceptance row must be an object")
        scenario = _scenario_id(source)
        outcome = str(source.get("outcome"))
        predicted_admit = source.get("predicted_admit") is True
        false_safe = outcome == "oom" and predicted_admit
        upper_covers = source.get("upper_covers_observed")
        if not isinstance(upper_covers, bool):
            upper = _finite(
                source.get("upper_reserved_bytes")
                if source.get("upper_reserved_bytes") is not None
                else source.get("decision_upper_bytes")
                if source.get("decision_upper_bytes") is not None
                else source.get("decision_upper_gib")
            )
            observed = _finite(
                source.get("observed_reserved_bytes")
                if source.get("observed_reserved_bytes") is not None
                else source.get("observed_reserved_gib")
            )
            upper_covers = bool(
                outcome == "success"
                and upper is not None
                and observed is not None
                and upper >= observed
            ) if outcome == "success" and upper is not None and observed is not None else None
        actual_safe = _actual_safe_success(source)
        row = {
            "scenario_id": scenario,
            "outcome": outcome,
            "predicted_admit": predicted_admit,
            "actual_safe_success": actual_safe,
            "upper_covers_observed": upper_covers,
            "false_safe_oom": false_safe,
        }
        normalized.append(row)
        by_scenario[scenario].append(row)

    successes = [row for row in normalized if row["outcome"] == "success"]
    coverage_rows = [row for row in successes if isinstance(row["upper_covers_observed"], bool)]
    coverage_fraction = (
        sum(bool(row["upper_covers_observed"]) for row in coverage_rows) / len(coverage_rows)
        if coverage_rows
        else None
    )
    scenario_coverages: list[float] = []
    scenario_details: list[dict[str, Any]] = []
    for scenario, scenario_rows in sorted(by_scenario.items()):
        covered = [row for row in scenario_rows if isinstance(row["upper_covers_observed"], bool)]
        rate = (
            sum(bool(row["upper_covers_observed"]) for row in covered) / len(covered)
            if covered
            else None
        )
        if rate is not None:
            scenario_coverages.append(rate)
        scenario_details.append(
            {
                "scenario_id": scenario,
                "rows": len(scenario_rows),
                "success_rows": sum(row["outcome"] == "success" for row in scenario_rows),
                "oom_rows": sum(row["outcome"] == "oom" for row in scenario_rows),
                "coverage_rows": len(covered),
                "coverage_fraction": rate,
                "false_safe_oom": sum(row["false_safe_oom"] for row in scenario_rows),
            }
        )
    false_safe = sum(row["false_safe_oom"] for row in normalized)
    admitted_over_safe = sum(
        row["predicted_admit"] and row["outcome"] == "success" and not row["actual_safe_success"]
        for row in normalized
    )
    scenario_p05 = _percentile(scenario_coverages, 5.0)
    passes = bool(
        false_safe <= int(maximum_false_safe_oom)
        and scenario_p05 is not None
        and scenario_p05 >= float(minimum_coverage)
    )
    return {
        "rows": len(normalized),
        "success_rows": len(successes),
        "oom_rows": sum(row["outcome"] == "oom" for row in normalized),
        "false_safe_oom": false_safe,
        "admitted_observed_over_safe_success": admitted_over_safe,
        "memory_safety_failures": false_safe + admitted_over_safe,
        "upper_coverage_fraction": coverage_fraction,
        "scenario_equal_mean_coverage": fmean(scenario_coverages) if scenario_coverages else None,
        "scenario_equal_p05_coverage": scenario_p05,
        "minimum_coverage": float(minimum_coverage),
        "maximum_false_safe_oom": int(maximum_false_safe_oom),
        "passes": passes,
        "scenarios": scenario_details,
        "details": normalized,
    }


def evaluate_ranking_acceptance(
    groups: Sequence[Mapping[str, Any]],
    *,
    maximum_top1_regret: float = DEFAULT_TOP1_REGRET,
) -> dict[str, Any]:
    """Evaluate v4b selection after the memory gate, one complete scenario at a time."""

    if not 0.0 <= float(maximum_top1_regret) <= 1.0:
        raise ValueError("maximum_top1_regret must be in [0, 1]")
    scenarios: list[dict[str, Any]] = []
    for group in groups:
        scenario = str(group.get("scenario_id") or group.get("comparison_group") or "").strip()
        if not scenario:
            raise ValueError("every ranking group requires scenario_id")
        candidates = group.get("candidates") or group.get("rows") or []
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise ValueError("ranking group candidates must be a list")
        rows = [dict(row) for row in candidates]
        safe = [row for row in rows if _actual_safe_success(row)]
        admitted = [row for row in rows if row.get("predicted_admit") is True]
        predicted_score = lambda row: _finite(
            row.get("predicted_throughput")
            if row.get("predicted_throughput") is not None
            else row.get("predicted_score")
        )
        observed_rate = lambda row: _finite(
            row.get("observed_throughput")
            if row.get("observed_throughput") is not None
            else row.get("effective_tokens_per_second")
        )
        predicted = max(
            (row for row in admitted if predicted_score(row) is not None),
            key=lambda row: float(predicted_score(row)),
            default=None,
        )
        observed = max(
            (row for row in safe if observed_rate(row) is not None),
            key=lambda row: float(observed_rate(row)),
            default=None,
        )
        selected_rate = observed_rate(predicted) if predicted is not None else None
        best_rate = observed_rate(observed) if observed is not None else None
        regret = (
            1.0 - selected_rate / best_rate
            if selected_rate is not None and best_rate and best_rate > 0.0
            else None
        )
        scenarios.append(
            {
                "scenario_id": scenario,
                "candidate_count": len(rows),
                "safe_success_count": len(safe),
                "admitted_candidate_count": len(admitted),
                "predicted_winner": predicted.get("candidate_id") if predicted else None,
                "observed_best": observed.get("candidate_id") if observed else None,
                "top1_regret": regret,
                "status": (
                    "evaluated"
                    if len(safe) >= 2 and predicted is not None and observed is not None and regret is not None
                    else "insufficient_evidence"
                ),
            }
        )
    eligible = [row for row in scenarios if row["status"] == "evaluated"]
    regrets = [float(row["top1_regret"]) for row in eligible]
    passes = bool(eligible and max(regrets) <= float(maximum_top1_regret))
    return {
        "scenario_count": len(scenarios),
        "eligible_scenarios": len(eligible),
        "mean_top1_regret": fmean(regrets) if regrets else None,
        "worst_top1_regret": max(regrets) if regrets else None,
        "maximum_top1_regret": float(maximum_top1_regret),
        "passes": passes,
        "scenarios": scenarios,
    }


def evaluate_scale_out_acceptance(
    pairs: Sequence[Mapping[str, Any]],
    *,
    minimum_ratio: float = DEFAULT_SCALE_RATIO,
) -> dict[str, Any]:
    """Require both conservative prediction and fresh measured lower bounds."""

    if float(minimum_ratio) < 1.0 or not math.isfinite(float(minimum_ratio)):
        raise ValueError("minimum_ratio must be finite and >= 1")
    details: list[dict[str, Any]] = []
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("every scale-out pair must be an object")
        baseline = pair.get("baseline")
        expanded = pair.get("expanded")
        if not isinstance(baseline, Mapping) or not isinstance(expanded, Mapping):
            raise ValueError("scale-out pair requires baseline and expanded summaries")
        decision = evaluate_doubling(
            baseline,
            expanded,
            minimum_ratio=float(minimum_ratio),
        )
        measured = _finite(
            pair.get("measured_ratio_lower")
            if pair.get("measured_ratio_lower") is not None
            else pair.get("fresh_measured_ratio_lower")
        )
        measured_passes = measured is not None and measured >= float(minimum_ratio)
        details.append(
            {
                "pair_id": pair.get("pair_id"),
                "decision": decision,
                "measured_ratio_lower": measured,
                "measured_passes": measured_passes,
                "passes": bool(decision["passes"] and measured_passes),
            }
        )
    passes = bool(details and all(row["passes"] for row in details))
    return {
        "pair_count": len(details),
        "minimum_ratio": float(minimum_ratio),
        "passes": passes,
        "false_positive_scale_out_claims": sum(
            row["decision"]["passes"] and not row["measured_passes"] for row in details
        ),
        "pairs": details,
    }


def evaluate_prospective_acceptance(
    *,
    memory_rows: Sequence[Mapping[str, Any]],
    ranking_groups: Sequence[Mapping[str, Any]],
    scale_pairs: Sequence[Mapping[str, Any]] = (),
    fresh_split: bool,
    scenario_level_split: bool = True,
    minimum_memory_coverage: float = DEFAULT_MEMORY_P95_COVERAGE,
    maximum_false_safe_oom: int = DEFAULT_FALSE_SAFE_OOM,
    maximum_top1_regret: float = DEFAULT_TOP1_REGRET,
    minimum_scale_ratio: float = DEFAULT_SCALE_RATIO,
) -> dict[str, Any]:
    """Build one report and explicit blockers for a prospective campaign."""

    memory = evaluate_memory_acceptance(
        memory_rows,
        minimum_coverage=minimum_memory_coverage,
        maximum_false_safe_oom=maximum_false_safe_oom,
    )
    ranking = evaluate_ranking_acceptance(
        ranking_groups,
        maximum_top1_regret=maximum_top1_regret,
    )
    scaling = evaluate_scale_out_acceptance(
        scale_pairs,
        minimum_ratio=minimum_scale_ratio,
    ) if scale_pairs else {
        "pair_count": 0,
        "minimum_ratio": float(minimum_scale_ratio),
        "passes": False,
        "false_positive_scale_out_claims": 0,
        "pairs": [],
    }
    blockers: list[str] = []
    if not fresh_split:
        blockers.append("holdout_is_not_fresh")
    if not scenario_level_split:
        blockers.append("scenario_level_split_missing")
    if not memory["passes"]:
        blockers.append("memory_acceptance_failed_or_insufficient")
    if not ranking["passes"]:
        blockers.append("ranking_acceptance_failed_or_insufficient")
    if scale_pairs and not scaling["passes"]:
        blockers.append("scale_out_acceptance_failed_or_insufficient")
    return {
        "schema": SCHEMA,
        "fresh_split": bool(fresh_split),
        "scenario_level_split": bool(scenario_level_split),
        "memory": memory,
        "ranking": ranking,
        "scale_out": scaling,
        "publication_blockers": blockers,
        "publication_allowed": not blockers,
        "gpu_training_started": False,
        "queues_mutated": False,
    }


def build_scale_out_acceptance_report(
    report: Mapping[str, Any],
    *,
    input_path: Path | None = None,
) -> dict[str, Any]:
    """Project the independent scale-out gate without promoting anything."""

    scaling = report.get("scale_out") or {}
    blockers: list[str] = []
    if report.get("fresh_split") is not True:
        blockers.append("holdout_is_not_fresh")
    if report.get("scenario_level_split") is not True:
        blockers.append("scenario_level_split_missing")
    if int(scaling.get("pair_count") or 0) <= 0:
        blockers.append("scale_out_evidence_missing")
    if scaling.get("passes") is not True:
        blockers.append("scale_out_acceptance_failed_or_insufficient")
    return {
        "schema": SCALE_SCHEMA,
        "fresh_split": report.get("fresh_split") is True,
        "scenario_level_split": report.get("scenario_level_split") is True,
        "minimum_ratio": scaling.get("minimum_ratio"),
        "pair_count": scaling.get("pair_count", 0),
        "passes": scaling.get("passes") is True and not blockers,
        "false_positive_scale_out_claims": scaling.get(
            "false_positive_scale_out_claims", 0
        ),
        "pairs": scaling.get("pairs") or [],
        "publication_blockers": sorted(set(blockers)),
        "publication_allowed": not blockers,
        "automatic_execution_allowed": False,
        "gpu_training_started": False,
        "queues_mutated": False,
        "source_acceptance_report_sha256": sha256_file(input_path)
        if input_path and input_path.is_file()
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON containing memory_rows, ranking_groups, scale_pairs and split flags",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scale-output",
        type=Path,
        default=None,
        help="optionally write the independent scale-out acceptance projection",
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SystemExit("input must be a JSON object")
    report = evaluate_prospective_acceptance(
        memory_rows=payload.get("memory_rows") or [],
        ranking_groups=payload.get("ranking_groups") or [],
        scale_pairs=payload.get("scale_pairs") or [],
        fresh_split=payload.get("fresh_split") is True,
        scenario_level_split=payload.get("scenario_level_split", True) is True,
        minimum_memory_coverage=float(
            payload.get("minimum_memory_coverage", DEFAULT_MEMORY_P95_COVERAGE)
        ),
        maximum_false_safe_oom=int(
            payload.get("maximum_false_safe_oom", DEFAULT_FALSE_SAFE_OOM)
        ),
        maximum_top1_regret=float(
            payload.get("maximum_top1_regret", DEFAULT_TOP1_REGRET)
        ),
        minimum_scale_ratio=float(
            payload.get("minimum_scale_ratio", DEFAULT_SCALE_RATIO)
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    if args.scale_output is not None:
        scale_report = build_scale_out_acceptance_report(
            report,
            input_path=args.output,
        )
        write_json(args.scale_output, scale_report)
    print(f"wrote {args.output}; publication_allowed={report['publication_allowed']}")


if __name__ == "__main__":
    main()
