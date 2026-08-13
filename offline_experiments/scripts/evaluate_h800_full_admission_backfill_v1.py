#!/usr/bin/env python3
"""Evaluate the FULL admission backfill campaign.

Reads the per-attempt status and rank metrics produced by the 46-job queue and
answers the three questions the campaign was designed around:

1. Did each mechanism's low/high probe pair bracket the safe/OOM boundary?
2. Which mechanisms need the pre-registered adaptive third point?
3. Which mechanisms now have enough evidence to fit their own admission head?

Analysis only: it never refits a model, never mutates a frozen artifact, and
never launches GPU work.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, ROOT, sha256_json, write_json


GIB = float(1 << 30)
SCHEMA = "sft_h800_full_admission_backfill_results/v1"
CAMPAIGN_ID = "h800_full_admission_backfill_20260806_v1"
DEFAULT_QUEUE = ROOT / "matrix" / "h800_full_admission_backfill_jobs_v1.jsonl"
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_full_admission_backfill_results_v1.json"

# Same conservative planning thresholds the coverage audit used.
MIN_INDEPENDENT_SOURCES = 5
MIN_NEGATIVE_BOUNDARY_SOURCES = 2

# The H800 admission safety line, in bytes, exactly as carried by every V5
# observation (142635080089.6 B = 132.8393 GiB).  The job queue does not carry
# safe_limit_bytes -- reading it from there silently disabled the unsafe-success
# check and mislabelled two over-the-line successes as safe.
SAFE_LIMIT_BYTES = 142635080089.6


def _safe_limit(job: dict[str, Any]) -> float:
    """Prefer a per-job limit when present, else the canonical H800 line."""
    value = job.get("safe_limit_bytes")
    try:
        value = float(value or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    return value if value > 0 else SAFE_LIMIT_BYTES


def _peak_reserved(attempt_dir: Path) -> float | None:
    """Max reserved bytes across every rank; admission safety is a max, not rank0."""
    peaks: list[float] = []
    for path in attempt_dir.glob("metrics/summary.rank*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("max_reserved")
        except (OSError, ValueError):
            continue
        if value:
            peaks.append(float(value))
    return max(peaks) if peaks else None


def collect(queue_path: Path, results_root: Path) -> list[dict[str, Any]]:
    jobs = {
        str(row["job_id"]): row
        for row in (
            json.loads(line)
            for line in queue_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    # The scheduler has no resume mode: relaunching re-runs the whole queue, so a
    # job can hold several attempts.  Keep the newest attempt per job_id and
    # record how many attempts it had -- counting raw status files would inflate
    # progress (it once reported 30/46 when only 16 jobs had finished).
    latest: dict[str, tuple[float, Path, dict[str, Any]]] = {}
    attempt_counts: Counter[str] = Counter()
    repeat_agreement: dict[str, set[str]] = defaultdict(set)
    for status_path in sorted(
        glob.glob(str(results_root / "h800fullbf-*" / "attempts" / "*" / "status.json"))
    ):
        path = Path(status_path)
        job_id = path.parts[-4]
        if job_id not in jobs:
            continue
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        attempt_counts[job_id] += 1
        repeat_agreement[job_id].add(str(status.get("classification")))
        started = float(status.get("started_unix") or 0.0)
        if job_id not in latest or started > latest[job_id][0]:
            latest[job_id] = (started, path, status)

    observations: list[dict[str, Any]] = []
    for job_id, (_started, path, status) in sorted(latest.items()):
        job = jobs[job_id]
        classification = status.get("classification")
        peak = _peak_reserved(path.parent)
        limit = _safe_limit(job)
        observations.append(
            {
                "job_id": job_id,
                "mechanism_id": job["mechanism_id"],
                "mechanism_cn": job["mechanism_cn"],
                "probe_role": job["probe_role"],
                "experiment_group": job["experiment_group"],
                "source_id": job["split_unit_id"],
                "model_id": job["model_id"],
                "mbs": job["mbs"],
                "cutoff_len": job["cutoff_len"],
                "gpu_count": job["gpu_count"],
                "classification": classification,
                "calibration_eligible": bool(status.get("calibration_eligible")),
                "attempts": attempt_counts[job_id],
                "repeat_attempts_agreed": len(repeat_agreement[job_id]) == 1,
                "peak_reserved_bytes": peak,
                "peak_reserved_gib": round(peak / GIB, 2) if peak else None,
                "safe_limit_gib": round(limit / GIB, 2) if limit else None,
                # OOM is a right-censored lower bound, never an exact peak.
                "outcome_class": (
                    "oom"
                    if classification == "oom"
                    else (
                        "unsafe_success"
                        if peak is not None and limit and peak > limit
                        else "safe_success"
                        if classification == "success"
                        else str(classification)
                    )
                ),
            }
        )
    return observations


def summarize(observations: list[dict[str, Any]]) -> dict[str, Any]:
    by_mechanism: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_mechanism[row["mechanism_id"]].append(row)

    mechanisms = []
    for mechanism_id, rows in sorted(by_mechanism.items()):
        sources = {row["source_id"] for row in rows}
        negative_sources = {
            row["source_id"] for row in rows if row["outcome_class"] != "safe_success"
        }
        low = [r for r in rows if r["probe_role"] == "probe_low"]
        high = [r for r in rows if r["probe_role"].startswith("probe_high")]
        low_states = {r["outcome_class"] for r in low}
        high_states = {r["outcome_class"] for r in high}
        bracketed = bool(
            low_states
            and high_states
            and low_states <= {"safe_success"}
            and high_states & {"oom", "unsafe_success"}
        )
        pending = not low or not high
        # A low probe that OOMs means the pair was placed too high to bracket.
        low_probe_oom = bool(low_states & {"oom", "unsafe_success"})
        needs_third = (not pending) and (not bracketed)
        eligible = (
            len(sources) >= MIN_INDEPENDENT_SOURCES
            and len(negative_sources) >= MIN_NEGATIVE_BOUNDARY_SOURCES
            and bool({r["outcome_class"] for r in rows} & {"safe_success"})
        )
        mechanisms.append(
            {
                "mechanism_id": mechanism_id,
                "mechanism_cn": rows[0]["mechanism_cn"],
                "observations": len(rows),
                "independent_sources": len(sources),
                "negative_boundary_sources": len(negative_sources),
                "outcomes": dict(Counter(r["outcome_class"] for r in rows)),
                "low_probe_states": sorted(low_states),
                "high_probe_states": sorted(high_states),
                "boundary_bracketed": bracketed,
                "still_running": pending,
                "low_probe_unexpectedly_unsafe": low_probe_oom,
                "needs_adaptive_third_point": needs_third,
                "meets_planning_thresholds_for_own_head": eligible,
                "peak_reserved_gib_range": [
                    min(
                        (r["peak_reserved_gib"] for r in rows if r["peak_reserved_gib"]),
                        default=None,
                    ),
                    max(
                        (r["peak_reserved_gib"] for r in rows if r["peak_reserved_gib"]),
                        default=None,
                    ),
                ],
            }
        )

    completed = len(observations)
    safe_peaks = [
        r["peak_reserved_gib"]
        for r in observations
        if r["outcome_class"] == "safe_success" and r["peak_reserved_gib"]
    ]
    repeated = [r for r in observations if r["attempts"] > 1]
    return {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "analysis_only": True,
        "model_refit": False,
        "publishable": False,
        "completed_jobs": completed,
        "expected_jobs": 46,
        "outcome_counts": dict(Counter(r["outcome_class"] for r in observations)),
        "calibration_eligible_count": sum(
            1 for r in observations if r["calibration_eligible"]
        ),
        "repeat_reproducibility": {
            "jobs_run_more_than_once": len(repeated),
            "reason": (
                "the scheduler has no resume mode, so relaunching after the "
                "single-card launcher fix re-ran the already-finished 2-GPU jobs"
            ),
            "jobs_whose_repeats_disagreed": sum(
                1 for r in repeated if not r["repeat_attempts_agreed"]
            ),
        },
        "safe_success_peak_reserved_gib": {
            "min": min(safe_peaks) if safe_peaks else None,
            "max": max(safe_peaks) if safe_peaks else None,
            "count": len(safe_peaks),
        },
        "unsafe_success_count": sum(
            1 for r in observations if r["outcome_class"] == "unsafe_success"
        ),
        "planning_thresholds": {
            "minimum_independent_sources_per_mechanism": MIN_INDEPENDENT_SOURCES,
            "minimum_negative_boundary_sources_per_mechanism": (
                MIN_NEGATIVE_BOUNDARY_SOURCES
            ),
            "note": (
                "conservative experiment-planning assumptions, not a release gate; "
                "OOM stays a right-censored lower bound and never an exact peak"
            ),
        },
        "mechanisms": mechanisms,
        "mechanisms_needing_adaptive_third_point": [
            row["mechanism_id"] for row in mechanisms if row["needs_adaptive_third_point"]
        ],
        "mechanisms_meeting_planning_thresholds": [
            row["mechanism_id"]
            for row in mechanisms
            if row["meets_planning_thresholds_for_own_head"]
        ],
        "next_step": (
            "append the adaptive third points for any unbracketed mechanism; that "
            "requires a fresh freeze and approval promotion because the live "
            "approval authorizes only the original 46 job ids"
        ),
        "observations": observations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="summarize without writing the report (safe while the queue runs)",
    )
    args = parser.parse_args()

    observations = collect(args.queue, args.results)
    report = summarize(observations)
    report["report_sha256"] = sha256_json(
        {k: v for k, v in report.items() if k != "report_sha256"}
    )

    header = (
        f"{report['completed_jobs']}/{report['expected_jobs']} jobs  "
        f"{report['outcome_counts']}"
    )
    print(header)
    print(
        f"{'机制':30s}{'源':>4s}{'负边界源':>9s}{'夹住':>6s}{'需第三点':>9s}{'可建头':>7s}"
    )
    for row in report["mechanisms"]:
        print(
            f"{row['mechanism_cn'][:28]:30s}"
            f"{row['independent_sources']:>4d}"
            f"{row['negative_boundary_sources']:>9d}"
            f"{('是' if row['boundary_bracketed'] else ('跑' if row['still_running'] else '否')):>6s}"
            f"{('是' if row['needs_adaptive_third_point'] else '-'):>9s}"
            f"{('是' if row['meets_planning_thresholds_for_own_head'] else '否'):>7s}"
        )
    if not args.print_only:
        write_json(args.output, report)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
