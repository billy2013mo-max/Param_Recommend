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

``check_length_alignment`` itself only reports; it never raises, because on a
ranking track a short-sample dataset is the intended business distribution
rather than an error.

``require_exact_length_basis`` is the hard gate for the admission track.  A
memory-admission metric is only meaningful when the measured run actually
reaches the cutoff the prediction assumed, so an admission queue must be built
from exact-length datasets (every sample equal to cutoff_len).  V3 shipped
without this gate: its hybrid rows ran on business data averaging ~805 tokens
against cutoff 8192, so "did not OOM" carried no evidence about whether the
model's refusals at cutoff were correct.  Call the gate from the freeze path of
any admission-track campaign.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import read_json

# The collator pads each batch up to a multiple of eight, matching
# DEFAULT_PAD_MULTIPLE in fit_h800_effective_sequence_v3.
PAD_MULTIPLE = 8


def _summary_profile_lengths(profile: dict[str, Any]) -> dict[str, Any] | None:
    """Read an aggregate profile that reports only summary statistics."""

    for key in (
        "max_tokens_per_sample",
        "maximum_clipped_tokens",
        "total_tokens_max",
    ):
        value = profile.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return {"max_tokens": int(value), "min_tokens": None, "rows": None}
    summary = profile.get("summary") or {}
    field = summary.get("total_tokens") or {}
    if isinstance(field, dict) and field.get("max"):
        minimum = field.get("min")
        return {
            "max_tokens": int(field["max"]),
            "min_tokens": int(minimum) if isinstance(minimum, (int, float)) else None,
            "rows": None,
        }
    return None


def _per_sample_profile_lengths(profile_path: Path) -> dict[str, Any] | None:
    """Read a per-sample JSONL profile such as the exact-length campaigns emit."""

    lengths: list[int] = []
    try:
        with profile_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                value = row.get("total_tokens")
                if not isinstance(value, (int, float)) or value <= 0:
                    return None
                lengths.append(int(value))
    except Exception:
        return None
    if not lengths:
        return None
    return {"max_tokens": max(lengths), "min_tokens": min(lengths), "rows": len(lengths)}


def _profile_lengths(profile_path: Path) -> dict[str, Any] | None:
    """Return ``max_tokens``/``min_tokens``/``rows`` for either profile layout.

    ``min_tokens`` is None when the profile only publishes a maximum, in which
    case exactness cannot be established and the admission gate must refuse.
    """

    try:
        profile = read_json(profile_path)
    except Exception:
        return _per_sample_profile_lengths(profile_path)
    if isinstance(profile, dict):
        return _summary_profile_lengths(profile)
    return _per_sample_profile_lengths(profile_path)


def _profile_max_tokens(profile_path: Path) -> int | None:
    lengths = _profile_lengths(profile_path)
    return None if lengths is None else int(lengths["max_tokens"])


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


class LengthAlignmentError(ValueError):
    """An admission-track queue whose measured peak cannot reach its cutoff."""

    def __init__(self, violations: list[dict[str, Any]]) -> None:
        self.violations = violations
        detail = "; ".join(
            f"{v['job_id']} ({v['model_id']} cutoff={v['cutoff_len']}): {v['reason']}"
            for v in violations
        )
        super().__init__(
            f"{len(violations)} admission-track job(s) are not on an exact-length "
            f"basis, so admission metrics would be unfalsifiable: {detail}"
        )


def audit_exact_length_basis(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one violation per job whose dataset cannot fill its cutoff.

    A job passes only when its profile proves every sample equals ``cutoff_len``.
    Padding to a multiple of eight means a cutoff that is not itself a multiple
    of eight can be reached by a slightly shorter sample, so the comparison
    allows that single alignment step.
    """

    violations: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job.get("job_id") or "")
        model_id = str(job.get("model_id") or "")
        cutoff = int(job.get("cutoff_len") or 0)
        if cutoff <= 0:
            violations.append({
                "job_id": job_id,
                "model_id": model_id,
                "cutoff_len": cutoff,
                "reason": "cutoff_len is missing or not positive",
            })
            continue
        raw_path = str(job.get("dataset_profile_path") or "")
        profile_path = Path(raw_path)
        if not raw_path or not profile_path.is_file():
            violations.append({
                "job_id": job_id,
                "model_id": model_id,
                "cutoff_len": cutoff,
                "reason": f"dataset profile is unreadable: {raw_path or '<unset>'}",
            })
            continue
        lengths = _profile_lengths(profile_path)
        if lengths is None:
            violations.append({
                "job_id": job_id,
                "model_id": model_id,
                "cutoff_len": cutoff,
                "reason": f"dataset profile carries no token lengths: {raw_path}",
            })
            continue
        max_tokens = int(lengths["max_tokens"])
        minimum = lengths["min_tokens"]
        if minimum is None:
            violations.append({
                "job_id": job_id,
                "model_id": model_id,
                "cutoff_len": cutoff,
                "dataset_max_tokens": max_tokens,
                "reason": (
                    "profile publishes only a maximum, so per-sample exactness "
                    "cannot be established; use an exact-length profile"
                ),
            })
            continue
        min_tokens = int(minimum)
        # A sample reaches the cutoff once padding to PAD_MULTIPLE lifts it
        # there, so an unaligned cutoff is satisfied by the aligned length just
        # below it.  Compare both ends against that floor.
        floor = cutoff - cutoff % PAD_MULTIPLE
        if min_tokens < floor or max_tokens < floor:
            violations.append({
                "job_id": job_id,
                "model_id": model_id,
                "cutoff_len": cutoff,
                "dataset_max_tokens": max_tokens,
                "dataset_min_tokens": min_tokens,
                "rows": lengths["rows"],
                "reason": (
                    f"samples span {min_tokens}..{max_tokens} tokens against "
                    f"cutoff {cutoff}: the measured peak need not reach the "
                    f"cutoff the prediction assumed"
                ),
            })
    return violations


def require_exact_length_basis(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Hard gate: raise unless every admission-track job is exact-length.

    Returns a binding summary for the freeze record when the queue passes.
    """

    violations = audit_exact_length_basis(jobs)
    if violations:
        raise LengthAlignmentError(violations)
    cutoffs = sorted({int(job["cutoff_len"]) for job in jobs})
    return {
        "policy": "every admission-track sample equals cutoff_len",
        "padding_multiple": PAD_MULTIPLE,
        "checked_jobs": len(jobs),
        "cutoff_lens": cutoffs,
        "exact_length_basis": True,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    from common import MATRIX_DIR, read_jsonl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "queue",
        nargs="?",
        default="h800_hybrid_vl_prospective_acceptance_v3.jsonl",
        help="queue file name under matrix/, or a path",
    )
    parser.add_argument(
        "--track",
        choices=("ranking", "admission"),
        default="ranking",
        help=(
            "ranking reports warnings and always exits 0; admission enforces the "
            "exact-length basis and exits 1 on any violation"
        ),
    )
    args = parser.parse_args(argv)

    queue_path = Path(args.queue)
    if not queue_path.is_file():
        queue_path = MATRIX_DIR / args.queue
    jobs = read_jsonl(queue_path)

    if args.track == "ranking":
        print_warnings(check_length_alignment(jobs))
        return 0

    try:
        binding = require_exact_length_basis(jobs)
    except LengthAlignmentError as error:
        print(f"[length-alignment] ADMISSION GATE FAILED: {error}")
        return 1
    print(f"[length-alignment] admission gate passed: {json.dumps(binding, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
