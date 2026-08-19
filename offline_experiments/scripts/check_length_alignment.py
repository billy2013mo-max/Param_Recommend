#!/usr/bin/env python3
"""Pre-freeze length-alignment check for prospective queues.

Detects the blind spot that caused the V3 hybrid recall drop: the memory
center is computed assuming the sequence is padded to cutoff_len, but the
measured peak only reflects what the random sampling during warmup/measure
actually drew. Two misalignments are flagged:

1. ``max_tokens < cutoff``: no sample can fill the cutoff, so a center computed
   at full cutoff systematically OVER-estimates the real peak.
2. ``max_tokens >= cutoff`` with small measure coverage: long-tail samples that
   would fill the cutoff exist, but random sampling over a short measure window
   may never draw them, so the OBSERVED peak under-reports the true worst case
   and the admission-recall metric becomes misleading.

The check reads each job's dataset profile and prints (and returns) warnings;
it never raises, because a warning is a measurement-protocol caveat, not an
error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import read_json


def _profile_max_tokens(profile_path: Path) -> int | None:
    try:
        profile = read_json(profile_path)
    except Exception:
        return None
    for key in (
        "max_tokens_per_sample",
        "maximum_clipped_tokens",
        "total_tokens_max",
    ):
        value = profile.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    summary = profile.get("summary") or {}
    field = summary.get("total_tokens") or {}
    if isinstance(field, dict) and field.get("max"):
        return int(field["max"])
    return None


def check_length_alignment(
    jobs: list[dict[str, Any]],
    *,
    measure_steps_default: int = 10,
) -> list[dict[str, Any]]:
    """Return warnings for jobs whose cutoff/sample-length assumption is at risk."""

    warnings: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for job in jobs:
        cutoff = int(job.get("cutoff_len") or 0)
        if cutoff <= 0:
            continue
        profile_path = Path(str(job.get("dataset_profile_path") or ""))
        if not profile_path.is_file():
            continue
        max_tokens = _profile_max_tokens(profile_path)
        if max_tokens is None:
            continue
        key = (str(profile_path), cutoff)
        if key in seen:
            continue
        seen.add(key)

        mbs = int(job.get("mbs") or 1)
        measure_steps = int(job.get("measure_steps") or measure_steps_default)
        warmup = int(job.get("warmup_steps") or 3)
        drawn_samples = (warmup + measure_steps) * mbs

        if max_tokens < cutoff:
            warnings.append(
                {
                    "job_id": str(job.get("job_id")),
                    "model_id": str(job.get("model_id")),
                    "cutoff_len": cutoff,
                    "dataset_max_tokens": max_tokens,
                    "kind": "center_overestimates",
                    "message": (
                        f"Dataset max tokens {max_tokens} < cutoff {cutoff}: no "
                        f"sample can fill the cutoff, so a center computed at full "
                        f"cutoff over-estimates the real peak."
                    ),
                }
            )
        else:
            warnings.append(
                {
                    "job_id": str(job.get("job_id")),
                    "model_id": str(job.get("model_id")),
                    "cutoff_len": cutoff,
                    "dataset_max_tokens": max_tokens,
                    "drawn_samples_estimate": drawn_samples,
                    "kind": "long_tail_may_be_missed",
                    "message": (
                        f"Dataset has samples reaching cutoff {cutoff} "
                        f"(max_tokens={max_tokens}), but only ~{drawn_samples} "
                        f"samples are drawn during warmup+measure; the observed "
                        f"peak may under-report the true worst case and "
                        f"admission-recall can look worse than it is."
                    ),
                }
            )
    return warnings


def print_warnings(warnings: list[dict[str, Any]]) -> None:
    if not warnings:
        print("[length-alignment] no misalignment detected")
        return
    print(f"[length-alignment] {len(warnings)} warning(s):")
    for w in warnings:
        print(f"  - [{w['kind']}] {w['model_id']} cutoff={w['cutoff_len']} "
              f"max_tokens={w['dataset_max_tokens']}: {w['message']}")


if __name__ == "__main__":
    import sys

    from common import MATRIX_DIR, read_jsonl

    queue_name = sys.argv[1] if len(sys.argv) > 1 else (
        "h800_hybrid_vl_prospective_acceptance_v3.jsonl"
    )
    jobs = read_jsonl(MATRIX_DIR / queue_name)
    print_warnings(check_length_alignment(jobs))
