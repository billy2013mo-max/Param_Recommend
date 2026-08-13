#!/usr/bin/env python3
"""Recount per-mechanism admission evidence after a campaign lands.

Answers one question for all twenty unpacked mechanisms (LoRA and FULL x 1/2/4
GPUs x ZeRO stage x gradient checkpointing): how many independent sources and
negative-boundary sources does each have, and how many are still missing before
its own admission head can be fitted?

Counts only what the head fit can actually consume.  A terminal result is not
enough: the head's features are ``log(center / safe_limit)``, so a mechanism
needs center predictions from the memory model, which in turn requires the run
to be an exported canonical observation.  Reporting raw campaign outcomes as
"sources" overstates readiness -- that mistake made two FULL mechanisms look
head-ready when their fit would have raised on a fold with no negative sample.

Analysis only: reads observations, writes a JSON report, never fits or mutates.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

from common import ARTIFACT_DIR, ROOT, sha256_json, write_json


SCHEMA = "sft_h800_admission_mechanism_evidence_recount/v1"
DEFAULT_OBSERVATIONS = (
    ARTIFACT_DIR / "canonical_h800_observations_with_backfill_v1.jsonl"
)
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_admission_mechanism_evidence_recount_v1.json"

# Planning thresholds, unchanged from the coverage audit.
MIN_INDEPENDENT_SOURCES = 5
MIN_NEGATIVE_BOUNDARY_SOURCES = 2

# Mechanisms that already carry a head in V5.1.
EXISTING_HEADS = {
    ("lora", 2, False, 2),
    ("full", 3, True, 2),
    ("lora", 2, False, 4),
}

# `split_unit_id` carries at least five historical conventions, and only some of
# them denote an independent dataset.  Counting distinct ids overstates coverage:
# `model_length_disjoint_h800_v1` splits ONE corpus by model and cutoff, so its
# nine "sources" are really five derived datasets drawn from three public corpora
# (alpaca_cleaned / ultrachat_200k / longalpaca_12k).  Policies whose own name
# says `fit_only` / `never_acceptance` are configuration slices by construction.
#
# The rule below keys off the declared policy rather than the id's spelling, and
# maps every non-dataset-disjoint partition onto its upstream corpus group so a
# connected group contributes exactly one cross-source unit.
DATASET_DISJOINT_POLICIES = {
    "business_source_disjoint_full_admission_backfill_v1",
    "remote_source_dataset_id_disjoint_v1",
    "remote_source_dataset_id_disjoint_stage2_v1",
    "profile_padding_stratified_scenario_disjoint_v1",
    "prospective_complete_s3_source_disjoint_v1",
    "prospective_unseen_dense_profiles_v1",
    "qwen3_14b_full_2gpu_boundary_profile_stratified_v1",
}

# Derived public datasets and the upstream corpus each was cut from.  Measured
# directly: pairwise sample overlap is 0%, but they descend from three corpora,
# and short_512 shares 125 sample_ids with longtail_8192.
UPSTREAM_CORPUS = {
    "short_512": "alpaca_cleaned",
    "multiturn_2048": "ultrachat_200k",
    "multiturn_4096": "ultrachat_200k",
    "longtail_8192": "mixed_public_longtail",
    "longcontext_16384": "longalpaca_12k",
    "longcontext_32768": "longalpaca_12k",
}


def _job(row: dict[str, Any]) -> dict[str, Any]:
    """Canonical observations nest the launch config under configuration.job."""
    configuration = row.get("configuration") or {}
    job = configuration.get("job")
    return job if isinstance(job, dict) else {}


def _partition(row: dict[str, Any]) -> dict[str, Any]:
    configuration = row.get("configuration") or {}
    partition = configuration.get("calibration_partition")
    if isinstance(partition, dict) and partition:
        return partition
    partition = _job(row).get("calibration_partition")
    return partition if isinstance(partition, dict) else {}


def _independent_source(row: dict[str, Any]) -> str | None:
    """Cross-source unit for this observation, or None when it is not one.

    Returns a dataset-level identity for dataset-disjoint campaigns, and folds
    everything else onto its upstream corpus group.  Two observations sharing a
    returned value must NOT be treated as independent evidence.
    """
    partition = _partition(row)
    raw = partition.get("split_unit_id") or _job(row).get("split_unit_id")
    if not isinstance(raw, str) or not raw:
        return None
    policy = str(partition.get("policy") or "")
    if policy in DATASET_DISJOINT_POLICIES:
        return raw
    dataset_id = str(_job(row).get("dataset_id") or "")
    if dataset_id in UPSTREAM_CORPUS:
        return f"public_corpus::{UPSTREAM_CORPUS[dataset_id]}"
    # Fit-only / never-acceptance partitions are configuration slices; collapse
    # each policy to a single unit so they cannot manufacture cross-source count.
    return f"non_dataset_partition::{policy or 'unknown'}"


def _mechanism(row: dict[str, Any]) -> tuple[str, int, bool, int] | None:
    job = _job(row)
    mode = job.get("train_type") or job.get("training_mode")
    if mode not in {"lora", "full"}:
        return None
    if job.get("packing"):
        return None
    zero = job.get("zero_stage")
    if zero is None:
        zero_name = str(job.get("zero") or "")
        zero = (
            int(zero_name.removeprefix("zero"))
            if zero_name.startswith("zero") and zero_name != "zero"
            else 0
        )
    gc = job.get("gradient_checkpointing")
    if gc is None:
        gc = job.get("gc")
    try:
        return (str(mode), int(zero or 0), bool(gc), int(job.get("gpu_count") or 0))
    except (TypeError, ValueError):
        return None


def _source_id(row: dict[str, Any]) -> str | None:
    return _independent_source(row)


def _outcome(row: dict[str, Any]) -> str:
    value = row.get("outcome")
    if isinstance(value, dict):
        value = value.get("class")
    return str(value or "").lower()


def _calibration_eligible(row: dict[str, Any]) -> bool:
    """Only base-eligible attempts may inform a published admission head."""
    outcome = row.get("outcome")
    return bool(isinstance(outcome, dict) and outcome.get("calibration_base_eligible"))


def _is_negative(row: dict[str, Any], safe_limit: float) -> bool:
    """OOM, or a success whose measured reserved peak crossed the safety line."""
    if "oom" in _outcome(row):
        return True
    memory = (row.get("measurements") or {}).get("memory") or {}
    peak = memory.get("max_reserved_bytes")
    limit = _job(row).get("safe_limit_bytes") or safe_limit
    try:
        return bool(peak and limit and float(peak) > float(limit))
    except (TypeError, ValueError):
        return False


def recount(
    rows: Iterable[dict[str, Any]], *, safe_limit: float
) -> dict[str, Any]:
    sources: dict[tuple, set[str]] = defaultdict(set)
    negatives: dict[tuple, set[str]] = defaultdict(set)
    raw_sources: dict[tuple, set[str]] = defaultdict(set)
    counts: dict[tuple, int] = defaultdict(int)
    skipped_no_source = 0
    skipped_not_eligible = 0
    for row in rows:
        mechanism = _mechanism(row)
        if mechanism is None:
            continue
        outcome = _outcome(row)
        if not outcome or ("oom" not in outcome and "success" not in outcome):
            continue
        if not _calibration_eligible(row):
            skipped_not_eligible += 1
            continue
        source = _source_id(row)
        if source is None:
            skipped_no_source += 1
            continue
        raw = (_partition(row).get("split_unit_id") or "") or source
        counts[mechanism] += 1
        sources[mechanism].add(source)
        raw_sources[mechanism].add(str(raw))
        if _is_negative(row, safe_limit):
            negatives[mechanism].add(source)

    mechanisms = []
    for mode in ("lora", "full"):
        for gpu_count in (1, 2, 4):
            zero_stages = (0,) if gpu_count == 1 else (2, 3)
            for zero in zero_stages:
                for gc in (False, True):
                    key = (mode, zero, gc, gpu_count)
                    have_sources = len(sources.get(key, ()))
                    have_negative = len(negatives.get(key, ()))
                    naive = len(raw_sources.get(key, ()))
                    missing_sources = max(0, MIN_INDEPENDENT_SOURCES - have_sources)
                    missing_negative = max(
                        0, MIN_NEGATIVE_BOUNDARY_SOURCES - have_negative
                    )
                    ready = missing_sources == 0 and missing_negative == 0
                    mechanisms.append(
                        {
                            "training_mode": mode,
                            "zero_stage": zero,
                            "gradient_checkpointing": gc,
                            "gpu_count": gpu_count,
                            "packing": False,
                            "label_cn": (
                                f"{mode.upper()} {gpu_count}卡 "
                                f"{'不切分' if zero == 0 else f'ZeRO-{zero}'} "
                                f"检查点{'开' if gc else '关'}"
                            ),
                            "observations": counts.get(key, 0),
                            "independent_sources": have_sources,
                            "negative_boundary_sources": have_negative,
                            "distinct_split_unit_ids_naive": naive,
                            "naive_count_inflation": naive - have_sources,
                            "missing_sources": missing_sources,
                            "missing_negative_boundary_sources": missing_negative,
                            "new_sources_needed": max(
                                missing_sources, missing_negative
                            ),
                            "has_head": key in EXISTING_HEADS,
                            "head_fittable_now": ready,
                        }
                    )

    pending = [
        row
        for row in mechanisms
        if not row["has_head"] and not row["head_fittable_now"]
    ]
    slots = sum(row["new_sources_needed"] for row in pending)
    return {
        "schema": SCHEMA,
        "analysis_only": True,
        "model_refit": False,
        "publishable": False,
        "planning_thresholds": {
            "minimum_independent_sources_per_mechanism": MIN_INDEPENDENT_SOURCES,
            "minimum_negative_boundary_sources_per_mechanism": (
                MIN_NEGATIVE_BOUNDARY_SOURCES
            ),
            "counting_rule": (
                "only exported canonical observations count; a terminal campaign "
                "result without a center prediction cannot feed an admission head"
            ),
        },
        "observations_skipped_without_source_id": skipped_no_source,
        "observations_skipped_not_calibration_base_eligible": skipped_not_eligible,
        "totals": {
            "mechanisms": len(mechanisms),
            "with_head": sum(1 for row in mechanisms if row["has_head"]),
            "fittable_now_without_new_experiments": sum(
                1 for row in mechanisms if row["head_fittable_now"] and not row["has_head"]
            ),
            "still_needing_experiments": len(pending),
            "mechanism_source_slots_missing": slots,
            "jobs_at_two_probes_per_slot": slots * 2,
            "jobs_with_adaptive_third_probe": slots * 3,
        },
        "mechanisms": mechanisms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--safe-limit-bytes", type=float, default=142635080089.6)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    rows = []
    with args.observations.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue

    report = recount(rows, safe_limit=args.safe_limit_bytes)
    report["observations_path"] = str(args.observations)
    report["report_sha256"] = sha256_json(
        {k: v for k, v in report.items() if k != "report_sha256"}
    )

    totals = report["totals"]
    print(f"读入观测 {len(rows)} 条")
    print(
        f"{'机制':30s}{'观测':>5s}{'真源':>5s}{'负源':>5s}{'裸计':>5s}{'虚高':>5s}"
        f"{'缺源':>5s}{'缺负':>5s}  状态"
    )
    for row in report["mechanisms"]:
        if row["has_head"]:
            state = "已建头"
        elif row["head_fittable_now"]:
            state = "可建头(零实验)"
        else:
            state = f"需补 {row['new_sources_needed']} 源"
        print(
            f"{row['label_cn']:30s}{row['observations']:>5d}"
            f"{row['independent_sources']:>5d}{row['negative_boundary_sources']:>5d}"
            f"{row['distinct_split_unit_ids_naive']:>5d}"
            f"{row['naive_count_inflation']:>5d}"
            f"{row['missing_sources']:>5d}{row['missing_negative_boundary_sources']:>5d}"
            f"  {state}"
        )
    print(
        f"\n已建头 {totals['with_head']}  可零成本建头 "
        f"{totals['fittable_now_without_new_experiments']}  待补数 "
        f"{totals['still_needing_experiments']}"
    )
    print(
        f"缺口槽位 {totals['mechanism_source_slots_missing']} -> "
        f"{totals['jobs_at_two_probes_per_slot']} 作业"
        f"(自适应上限 {totals['jobs_with_adaptive_third_probe']})"
    )
    if not args.print_only:
        write_json(args.output, report)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
