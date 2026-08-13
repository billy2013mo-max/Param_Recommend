#!/usr/bin/env python3
"""Diagnose which sequence-length rule reconstructs measured per-microbatch tokens.

The Packing memory center is anchored on a physical basis that assumes every
microbatch is filled to ``cutoff_len``.  That assumption is close to correct for
neat packing (packs are built near the cutoff) and badly wrong for unpacked SFT
on a short-tailed corpus, where a microbatch may hold a few hundred tokens under
a 20k cutoff.  The consequence is a center that is accurate on packed arms and
inflated on unpacked ones, which then inverts into an apparent "packing memory
explosion" when the two are divided.

This script scores the candidate length rules already implemented by
``ProfileLengths`` (cutoff / max / p99 / batch_max) against the *measured* tokens
per physical batch recorded in each run's rank summaries.  It is read-only: it
fits nothing, publishes nothing, and writes a single diagnostic report.

Ground truth per arm is ``computed_tokens / physical_batches`` summed across
ranks, i.e. the average model-facing sequence length actually processed.  The
comparison is deliberately made on tokens rather than on GiB so that the length
question is isolated from the memory model's own coefficients.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from typing import Any

from common import ARTIFACT_DIR, RESULTS_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from experiment_effective_sequence_memory_basis import ProfileLengths, load_profile_lengths


PHASE_B = ARTIFACT_DIR / "h800_packing_profile_phase_b_results_v1.json"
PHASE_C = ARTIFACT_DIR / "h800_packing_profile_phase_c_results_v1.json"
PROFILE_DIR = ARTIFACT_DIR / "packing_profile_phase_b_v1" / "profiles"
# The Packing campaigns are all mbs=1, which makes batch_max indistinguishable
# from a plain mean (see microbatch_size_coverage in the report).  The wider
# collected results carry unpacked mbs>=2 rows on datasets that have frozen
# length profiles, which is exactly the evidence needed to separate the two.
COLLECTED = ARTIFACT_DIR / "collected_results.json"
COLLECTED_PROFILE_DIRS = (
    ARTIFACT_DIR / "dataset_profiles",
    ARTIFACT_DIR / "fresh_holdout_v2" / "profiles",
    ARTIFACT_DIR / "h800_lora_safety_stage2_v1" / "profiles",
    ARTIFACT_DIR / "h800_14b_full_boundary_v1" / "profiles",
    ARTIFACT_DIR / "real_business_packing_cutoff_mbs_v1" / "profiles",
)
STAGE1_QUEUES = (
    ROOT / "matrix" / "h800_packing_memory_boundary_stage1_v1.jsonl",
    ROOT / "matrix" / "h800_packing_memory_boundary_stage1_resume_v2.jsonl",
)
OUTPUT_DIR = ROOT / "diagnostics" / "packing_effective_length_rule_20260807"
OUTPUT = OUTPUT_DIR / "packing_effective_length_rule_diagnostic_v1.json"

# Rules already implemented by ProfileLengths.  "cutoff" is the incumbent
# assumption; the others reduce it using the frozen length profile.
RULES = ("cutoff", "max", "p99", "batch_max")
PAD_MULTIPLE = 8


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _measured_tokens_per_batch(job_id: str) -> dict[str, Any] | None:
    """Average model-facing tokens per physical batch, summed over ranks.

    Returns None when no rank emitted a measured step, which is the normal case
    for a run that OOMed during warmup.  Such rows carry no length evidence.
    """
    paths = sorted((RESULTS_DIR / job_id / "metrics").glob("summary.rank*.json"))
    if not paths:
        return None
    computed = 0
    batches = 0
    logical = 0
    ranks = 0
    for path in paths:
        summary = read_json(path)
        totals = summary.get("measured_totals")
        if not totals:
            continue
        computed += int(totals.get("computed_tokens") or 0)
        batches += int(totals.get("physical_batches") or 0)
        logical += int(totals.get("logical_samples") or 0)
        ranks += 1
    if not batches or not computed:
        return None
    return {
        "rank_summaries_with_measured_steps": ranks,
        "computed_tokens": computed,
        "physical_batches": batches,
        "logical_samples": logical,
        "tokens_per_physical_batch": computed / batches,
        "logical_samples_per_physical_batch": logical / batches if batches else None,
    }


def _predicted_tokens(
    profiles: ProfileLengths,
    *,
    dataset_id: str,
    cutoff_len: int,
    mbs: int,
    packing: bool,
    rule: str,
) -> int | None:
    """Length a rule assigns, following the frozen non-packing contract.

    Packing concatenates samples into packs near the cutoff, so a single-sample
    statistic is not an upper bound and every rule degenerates to the cutoff.
    That degeneracy is the point of the diagnostic: it shows the incumbent rule
    is only defensible on the packed branch.
    """
    if packing:
        return int(cutoff_len)
    if rule == "cutoff":
        return int(cutoff_len)
    if not profiles.has(dataset_id):
        return None
    raw = profiles.sequence_for(dataset_id, cutoff_len=int(cutoff_len), mbs=int(mbs), rule=rule)
    return _round_up(min(int(cutoff_len), int(raw)), PAD_MULTIPLE)


def _collected_mbs_rows() -> list[dict[str, Any]]:
    """Unpacked mbs>=2 rows from the wider campaign results.

    Ground truth here is ``computed_tokens / logical_samples``: with packing off
    every logical sample occupies one padded row, so that ratio is the padded row
    length the collator actually produced -- which is precisely what batch_max
    predicts (E[max of mbs draws]).  These rows are what make the mbs=1 identity
    testable; without them batch_max cannot be told apart from a plain mean.
    """
    if not COLLECTED.is_file():
        return []
    payload = read_json(COLLECTED)
    raw = payload if isinstance(payload, list) else (payload.get("results") or payload.get("rows") or [])
    rows: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict) or row.get("packing"):
            continue
        if row.get("classification") != "success":
            continue
        mbs = int(row.get("mbs") or 0)
        samples = int(row.get("logical_samples") or 0)
        computed = int(row.get("computed_tokens") or 0)
        if mbs < 2 or samples <= 0 or computed <= 0:
            continue
        rows.append(
            {
                "source": "collected_unpacked_mbs_gt1",
                "job_id": str(row.get("job_id") or ""),
                "setting_id": f"{row.get('dataset_id')}-c{row.get('cutoff_len')}-mbs{mbs}",
                "workload_id": str(row.get("dataset_id")),
                "dataset_id": str(row.get("dataset_id")),
                "model_id": str(row.get("model_id")),
                "train_type": str(row.get("train_type")),
                "cutoff_len": int(row["cutoff_len"]),
                "mbs": mbs,
                "packing": False,
                "zero_stage": int(row.get("zero_stage") or 0),
                "gc": bool(row.get("gc")),
                "gpu_count": int(row.get("gpu_count") or 0),
                "repeat": int(row.get("repeat") or 0),
                "observed_max_reserved_gib": None,
                "n_pack_mean": None,
                "evidence_role": "collected_diagnostic_not_for_fit",
                "_measured": {
                    "rank_summaries_with_measured_steps": None,
                    "computed_tokens": computed,
                    "physical_batches": None,
                    "logical_samples": samples,
                    "tokens_per_physical_batch": computed / samples,
                    "logical_samples_per_physical_batch": 1.0,
                },
            }
        )
    return rows


def _arm_rows() -> list[dict[str, Any]]:
    """Collect every arm that carries both a length profile and measured tokens."""
    rows: list[dict[str, Any]] = []
    for source, path in (("phase_b", PHASE_B), ("phase_c", PHASE_C)):
        report = read_json(path)
        for job in report["job_results"]:
            if job.get("classification") != "success":
                continue
            if job.get("calibration_eligible") is not True:
                continue
            rows.append(
                {
                    "source": source,
                    "job_id": str(job["job_id"]),
                    "setting_id": str(job.get("setting_id") or job.get("family_id")),
                    "workload_id": str(job["workload_id"]),
                    "dataset_id": str(job["dataset_id"]),
                    "model_id": str(job["model_id"]),
                    "train_type": str(job["train_type"]),
                    "cutoff_len": int(job["cutoff_len"]),
                    "mbs": int(job["mbs"]),
                    "packing": bool(job["packing"]),
                    "zero_stage": int(job["zero_stage"]),
                    "gc": bool(job["gc"]),
                    "gpu_count": int(job["gpu_count"]),
                    "repeat": int(job.get("repeat") or 0),
                    "observed_max_reserved_gib": job.get("max_reserved_gib"),
                    "n_pack_mean": job.get("n_pack_mean"),
                    "evidence_role": "fit_only_training_arm",
                }
            )
    seen: set[str] = set()
    for queue in STAGE1_QUEUES:
        for job in read_jsonl(queue):
            job_id = str(job["job_id"])
            if job_id in seen:
                continue
            seen.add(job_id)
            status_path = RESULTS_DIR / job_id / "status.json"
            if not status_path.is_file():
                continue
            status = read_json(status_path)
            if status.get("calibration_eligible") is not True:
                continue
            if status.get("classification") != "success":
                continue
            rows.append(
                {
                    "source": "stage1_boundary",
                    "job_id": job_id,
                    "setting_id": str(job["boundary_setting_id"]),
                    "workload_id": str(job["workload_id"]),
                    "dataset_id": str(job["dataset_id"]),
                    "model_id": str(job["model_id"]),
                    "train_type": str(job["train_type"]),
                    "cutoff_len": int(job["cutoff_len"]),
                    "mbs": int(job["mbs"]),
                    "packing": bool(job["packing"]),
                    "zero_stage": int(job["zero_stage"]),
                    "gc": bool(job["gc"]),
                    "gpu_count": int(job["gpu_count"]),
                    "repeat": int(job.get("repeat") or 0),
                    "observed_max_reserved_gib": None,
                    "n_pack_mean": job.get("n_pack_mean"),
                    # Stage 1 is held out of any future fit by this diagnostic;
                    # it is scored here only to show out-of-range behaviour.
                    "evidence_role": "prospective_diagnostic_not_for_fit",
                }
            )
    rows.extend(_collected_mbs_rows())
    return rows


def _summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    logs = [math.log(value) for value in values]
    return {
        "count": len(values),
        "geometric_mean_ratio": math.exp(statistics.fmean(logs)),
        "log_ratio_stdev": statistics.stdev(logs) if len(logs) > 1 else 0.0,
        "minimum_ratio": min(values),
        "maximum_ratio": max(values),
        "median_absolute_percentage_error": statistics.median(
            [abs(value - 1.0) for value in values]
        ),
        "mean_absolute_percentage_error": statistics.fmean(
            [abs(value - 1.0) for value in values]
        ),
    }


def _merged_profiles() -> ProfileLengths:
    """Length profiles for the Packing campaigns plus the wider collected runs.

    ``load_profile_lengths`` reads one directory, but the mbs>1 rows come from
    datasets profiled under several campaign directories.  Merge them into one
    lookup; on a duplicate dataset id the Packing campaign's own profile wins so
    the primary evidence keeps its frozen lengths.
    """
    lengths: dict[str, list[int]] = {}
    for directory in (*COLLECTED_PROFILE_DIRS, PROFILE_DIR):
        if not directory.is_dir():
            continue
        merged = load_profile_lengths(directory)
        for dataset_id in merged.datasets():
            lengths[dataset_id] = merged.raw_lengths(dataset_id)
    return ProfileLengths(lengths)


def diagnose() -> dict[str, Any]:
    profiles = _merged_profiles()
    rows = _arm_rows()

    scored: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        # Collected rows carry their own measurement (no rank summaries on disk).
        measured = row.pop("_measured", None) or _measured_tokens_per_batch(row["job_id"])
        if measured is None:
            skipped.append(
                {
                    "job_id": row["job_id"],
                    "setting_id": row["setting_id"],
                    "reason": "no_measured_step_tokens",
                }
            )
            continue
        actual = measured["tokens_per_physical_batch"]
        entry = {**row, **measured, "rules": {}}
        for rule in RULES:
            predicted = _predicted_tokens(
                profiles,
                dataset_id=row["dataset_id"],
                cutoff_len=row["cutoff_len"],
                mbs=row["mbs"],
                packing=row["packing"],
                rule=rule,
            )
            entry["rules"][rule] = {
                "predicted_tokens": predicted,
                "ratio_to_measured": (predicted / actual) if predicted else None,
            }
        entry["fill_fraction_of_cutoff"] = actual / row["cutoff_len"]
        scored.append(entry)

    # Score each rule on the unpacked branch, where the rules actually differ,
    # and report the packed branch separately to make the degeneracy explicit.
    rule_reports: dict[str, Any] = {}
    for rule in RULES:
        by_branch: dict[str, list[float]] = {"unpacked": [], "packed": []}
        by_role: dict[str, list[float]] = {}
        for entry in scored:
            ratio = entry["rules"][rule]["ratio_to_measured"]
            if ratio is None:
                continue
            by_branch["packed" if entry["packing"] else "unpacked"].append(ratio)
            by_role.setdefault(entry["evidence_role"], []).append(ratio)
        rule_reports[rule] = {
            "unpacked": _summarize(by_branch["unpacked"]),
            "packed": _summarize(by_branch["packed"]),
            "by_evidence_role": {
                role: _summarize(values) for role, values in sorted(by_role.items())
            },
        }

    fit_only = [row for row in scored if row["evidence_role"] == "fit_only_training_arm"]
    unpacked_fit_only = [row for row in fit_only if not row["packing"]]
    ranked = sorted(
        RULES,
        key=lambda rule: _summarize(
            [
                row["rules"][rule]["ratio_to_measured"]
                for row in unpacked_fit_only
                if row["rules"][rule]["ratio_to_measured"] is not None
            ]
        ).get("mean_absolute_percentage_error", math.inf),
    )
    best_rule = ranked[0] if ranked else None

    # Any rule that clamps at the cutoff cannot represent a corpus whose samples
    # are far shorter than the cutoff.  Record how often each rule still lands on
    # the cutoff so a "winning" rule is not mistaken for a general fix.
    degenerate_counts = {
        rule: sum(
            1
            for row in scored
            if not row["packing"]
            and row["rules"][rule]["predicted_tokens"] == row["cutoff_len"]
        )
        for rule in RULES
    }

    per_workload: dict[str, Any] = {}
    for row in unpacked_fit_only:
        per_workload.setdefault(row["workload_id"], []).append(row)
    workload_reports = {
        workload: {
            "arms": len(entries),
            "median_fill_fraction_of_cutoff": statistics.median(
                [entry["fill_fraction_of_cutoff"] for entry in entries]
            ),
            "rules": {
                rule: _summarize(
                    [
                        entry["rules"][rule]["ratio_to_measured"]
                        for entry in entries
                        if entry["rules"][rule]["ratio_to_measured"] is not None
                    ]
                )
                for rule in RULES
            },
        }
        for workload, entries in sorted(per_workload.items())
    }

    # CRITICAL CAVEAT.  ``batch_max`` is E[max of mbs iid draws], which at mbs=1
    # collapses to the plain mean of clipped lengths -- and the ground truth
    # (computed_tokens / physical_batches) is that same mean when each physical
    # batch holds one sample.  So on an all-mbs=1 corpus the two are nearly the
    # same statistic and the low error is close to an identity, not a validated
    # prediction.  Discriminating batch_max from a plain mean needs mbs>1
    # evidence, which this evidence set does not contain.
    observed_mbs = sorted({int(row["mbs"]) for row in scored})
    unpacked_mbs = sorted({int(row["mbs"]) for row in scored if not row["packing"]})
    batch_max_is_near_tautological = unpacked_mbs == [1]

    report: dict[str, Any] = {
        "schema": "sft_packing_effective_length_rule_diagnostic/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "score candidate sequence-length rules against measured tokens per "
            "physical batch; select the unpacked centre-length definition before "
            "any refit"
        ),
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for key, path in (
                ("phase_b_results", PHASE_B),
                ("phase_c_results", PHASE_C),
                ("stage1_parent_queue", STAGE1_QUEUES[0]),
                ("stage1_resume_queue", STAGE1_QUEUES[1]),
            )
        },
        "profile_directory": str(PROFILE_DIR.resolve()),
        "ground_truth": "computed_tokens / physical_batches summed across ranks",
        "padding_multiple": PAD_MULTIPLE,
        "arms_scored": len(scored),
        "arms_skipped": skipped,
        "rule_reports": rule_reports,
        "unpacked_fit_only_rule_ranking": ranked,
        "best_unpacked_rule_on_fit_only_arms": best_rule,
        "unpacked_rows_where_rule_still_equals_cutoff": degenerate_counts,
        "per_workload_unpacked": workload_reports,
        "microbatch_size_coverage": {
            "observed_mbs_values": observed_mbs,
            "unpacked_mbs_values": unpacked_mbs,
            "batch_max_is_near_tautological_on_this_evidence": batch_max_is_near_tautological,
            "why": (
                "batch_max is E[max of mbs iid clipped lengths]; at mbs=1 it equals "
                "the plain mean of clipped lengths, and the ground truth "
                "computed_tokens/physical_batches is that same mean when a physical "
                "batch holds one sample. On an all-mbs=1 evidence set the low error "
                "is close to an identity rather than a validated prediction."
            ),
            "required_to_discriminate": (
                "unpacked arms at mbs>1, where batch_max and the plain mean diverge "
                "(W8 clipped: mbs=1 -> 464, mbs=2 -> 564, mbs=4 -> 672, mbs=8 -> 786)"
            ),
        },
        "packed_branch_note": (
            "every rule degenerates to cutoff when packing is on, so the packed "
            "branch cannot discriminate between rules; its accuracy reflects the "
            "physical assumption being correct there, not the rule being validated"
        ),
        "rows": scored,
        "fits_anything": False,
        "publishable": False,
        "automatic_packing_recommendation_allowed": False,
        "frozen_artifacts_modified": False,
        "next_step": (
            "use the selected unpacked length rule to re-anchor the Packing memory "
            "centre in a challenger; keep Stage 1 rows as prospective evidence"
        ),
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    return report


def main() -> None:
    report = diagnose()
    compact = {
        "output": str(OUTPUT),
        "report_sha256": report["report_sha256"],
        "arms_scored": report["arms_scored"],
        "arms_skipped": len(report["arms_skipped"]),
        "unpacked_branch_by_rule": {
            rule: {
                "count": report["rule_reports"][rule]["unpacked"].get("count"),
                "mean_ape": report["rule_reports"][rule]["unpacked"].get(
                    "mean_absolute_percentage_error"
                ),
                "geo_mean_ratio": report["rule_reports"][rule]["unpacked"].get(
                    "geometric_mean_ratio"
                ),
                "max_ratio": report["rule_reports"][rule]["unpacked"].get("maximum_ratio"),
            }
            for rule in RULES
        },
        "packed_branch_geo_mean_ratio": {
            rule: report["rule_reports"][rule]["packed"].get("geometric_mean_ratio")
            for rule in RULES
        },
        "unpacked_fit_only_rule_ranking": report["unpacked_fit_only_rule_ranking"],
        "unpacked_rows_where_rule_still_equals_cutoff": report[
            "unpacked_rows_where_rule_still_equals_cutoff"
        ],
        "microbatch_size_coverage": {
            "unpacked_mbs_values": report["microbatch_size_coverage"][
                "unpacked_mbs_values"
            ],
            "batch_max_is_near_tautological_on_this_evidence": report[
                "microbatch_size_coverage"
            ]["batch_max_is_near_tautological_on_this_evidence"],
        },
        "next_step": report["next_step"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
