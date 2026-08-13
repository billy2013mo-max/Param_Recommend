#!/usr/bin/env python3
"""Prepare the LoRA / 2-GPU / ZeRO-3 / no-checkpointing memory boundary campaign.

Why
---
V5 cannot fit an admission head for this mechanism.  The recount counts 3
independent sources against a threshold of 5, so two more are needed.  Without a
head, every ZeRO-3 candidate in a throughput matrix has to be launched blind --
which is exactly how the 2026-08-06 acceptance lost 14 of 16 ZeRO-3 jobs to OOM
and burned four single-use holdout datasets for one usable cell.

The source shortage is not a shortage of measurements.  52 stage2 observations
already exist for this mechanism, including OOM boundaries on six distinct
datasets, and all 52 are rejected because their authorized job payload carries no
``calibration_partition``.  The exporter reads that block only from the frozen
payload and refuses to accept it after the fact, so those runs cannot be
recovered -- the annotation has to be present before the job starts.  This
campaign therefore re-measures two datasets *with* the annotation, and the
companion change to the throughput queue builder stops the leak for future runs.

Design
------
Two previously unconsumed stage2 datasets become two new independent sources.
Each is probed at three MBS values that bracket the known boundary: 14B at
cutoff 4096 OOMs at MBS8 (138.1 GiB observed) and fits at MBS4, so MBS 2/4/8
spans safe, marginal and unsafe.  A head needs both classes; a ladder that only
OOMs teaches it nothing.

``split_unit_id`` reuses the canonical per-dataset identity already established
for stage2 rather than minting a new one -- a fresh id for the same dataset would
count as an extra independent source and silently inflate coverage.

Offline: writes a queue and a design, starts no GPU work, mutates no live config,
and touches no frozen artifact.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from common import (
    ARTIFACT_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)

CAMPAIGN_ID = "h800_lora_2gpu_zero3_boundary_20260808_v1"
PHASE_ID = "h800_lora_2gpu_zero3_boundary_v1"
JOB_SCHEMA = "sft_h800_lora_2gpu_zero3_boundary_job/v1"
DESIGN_SCHEMA = "sft_h800_lora_2gpu_zero3_boundary_design/v1"

MODEL_ID = "qwen3_14b"
GPU_COUNT = 2
TARGET_GBS = 64
CUTOFF_LEN = 4096
REPEATS = 2

# Canonical stage2 policy.  Listed in DATASET_DISJOINT_POLICIES, so each distinct
# split_unit_id under it counts as one independent source.
SPLIT_POLICY = "remote_source_dataset_id_disjoint_stage2_v1"

# Role is `calibration`: these runs exist to teach the memory model, and nothing
# here is used to score a throughput prediction.  The datasets are picked from the
# pool never consumed by any throughput evidence, so spending them here does not
# compromise a future throughput holdout.
SPLIT_ROLE = "calibration"

# MBS ladder bracketing the known boundary at 14B / cutoff 4096:
# MBS4 fits (137.6 GiB at the margin), MBS8 OOMs (138.1 GiB).  Probing 2/4/8
# yields safe + marginal + unsafe, which is what a logistic head needs.
MBS_LADDER = (2, 4, 8)

# Two unconsumed stage2 datasets, the two largest available so the measurement
# window is as far from a dataloader wrap as this corpus allows.
TARGET_DATASETS: tuple[dict[str, Any], ...] = (
    {"dataset_id": "lora_s2_src11_flood_route_fallback", "samples": 768},
    {"dataset_id": "lora_s2_src13_text_safety", "samples": 768},
)

TEMPLATE_QUEUE = ROOT / "matrix" / "h800_lora_safety_stage2_jobs_v1.jsonl"
OBSERVATIONS = ARTIFACT_DIR / "canonical_h800_observations_with_backfill_v1.jsonl"
RECOUNT = ARTIFACT_DIR / "h800_admission_mechanism_evidence_recount_v1.json"
PROFILE_DIR = ARTIFACT_DIR / "h800_lora_safety_stage2_v1" / "profiles"
DATA_DIR = ROOT / "data" / "h800_lora_safety_stage2_v1"

DEFAULT_QUEUE = ROOT / "matrix" / "h800_lora_2gpu_zero3_boundary_jobs_v1.jsonl"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_lora_2gpu_zero3_boundary_design_v1.json"

# Datasets already spent on throughput evidence; probing one of these would make
# it unusable as a future throughput holdout for no gain here.
THROUGHPUT_CONSUMED = frozenset(
    {
        "lora_s2_src01_live_pk_script",
        "lora_s2_src02_comment_intent",
        "lora_s2_src03_quality_comment",
        "lora_s2_src05_beauty_slots",
        "lora_s2_src08_flood_topic",
        "lora_s2_src10_flood_route",
        "lora_s2_src04_live_punishment",
        "lora_s2_src12_flood_multilabel",
        "lora_s2_src14_ad_metrics",
        "lora_s2_src18_marketing_antifraud",
    }
)


def _canonical_split_unit_ids() -> dict[str, str]:
    """Per-dataset split_unit_id already established under the stage2 policy.

    Minting a new id for a dataset that already has one would present the same
    data as a second independent source, so the existing identity is mandatory
    rather than cosmetic.
    """

    mapping: dict[str, str] = {}
    with OBSERVATIONS.open(encoding="utf-8") as handle:
        for line in handle:
            if SPLIT_POLICY not in line:
                continue
            row = json.loads(line)
            configuration = row.get("configuration") or {}
            partition = configuration.get("calibration_partition") or {}
            job = configuration.get("job") or {}
            dataset_id = job.get("dataset_id")
            split_unit_id = partition.get("split_unit_id")
            if not dataset_id or not split_unit_id:
                continue
            previous = mapping.setdefault(str(dataset_id), str(split_unit_id))
            if previous != str(split_unit_id):
                raise ValueError(
                    f"{dataset_id} carries two split_unit_ids under {SPLIT_POLICY}: "
                    f"{previous} and {split_unit_id}"
                )
    return mapping


def _assert_sources_are_new(split_unit_ids: Mapping[str, str]) -> dict[str, Any]:
    """Fail closed unless both targets are genuinely new sources for the mechanism.

    A probe on a dataset that already contributes to this mechanism adds
    observations but not coverage, and the recount would not move.
    """

    existing: set[str] = set()
    with OBSERVATIONS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            configuration = row.get("configuration") or {}
            job = configuration.get("job") or {}
            if (
                job.get("train_type") != "lora"
                or int(job.get("zero_stage") or 0) != 3
                or bool(job.get("gradient_checkpointing"))
                or int(job.get("gpu_count") or 0) != GPU_COUNT
                or bool(job.get("packing"))
            ):
                continue
            partition = configuration.get("calibration_partition") or {}
            if not (row.get("outcome") or {}).get("calibration_base_eligible"):
                continue
            unit = partition.get("split_unit_id")
            if unit:
                existing.add(str(unit))

    planned = {split_unit_ids[str(entry["dataset_id"])] for entry in TARGET_DATASETS}
    collisions = sorted(planned & existing)
    if collisions:
        raise ValueError(f"these split units already count as sources: {collisions}")
    reused = sorted(
        str(entry["dataset_id"])
        for entry in TARGET_DATASETS
        if str(entry["dataset_id"]) in THROUGHPUT_CONSUMED
    )
    if reused:
        raise ValueError(f"refusing to spend throughput-consumed datasets: {reused}")
    return {
        "eligible_source_units_before": sorted(existing),
        "eligible_source_count_before": len(existing),
        "planned_new_units": sorted(planned),
        "planned_new_source_count": len(planned),
        "projected_source_count": len(existing) + len(planned),
    }


def _assert_measurement_window_fits(samples: int, mbs: int, steps: int) -> dict[str, Any]:
    """Report whether the window stays inside one epoch.

    Crossing an epoch boundary can leave the consumed-token ledger one batch
    short, which excludes the run.  For a memory boundary probe the peak is
    recorded regardless, so this is reported rather than enforced -- but it is
    reported, because it is the defect that silently dropped two acceptance jobs.
    """

    demand = steps * (TARGET_GBS // (GPU_COUNT * mbs)) * mbs * GPU_COUNT
    return {
        "samples_available": samples,
        "samples_required_for_single_epoch": demand,
        "stays_within_one_epoch": samples >= demand,
    }


def _job(
    template: Mapping[str, Any],
    dataset: Mapping[str, Any],
    split_unit_id: str,
    mbs: int,
    repeat: int,
) -> dict[str, Any]:
    dataset_id = str(dataset["dataset_id"])
    if TARGET_GBS % (GPU_COUNT * mbs):
        raise ValueError("target GBS is not divisible by GPU count times MBS")
    identity = {
        "campaign_id": CAMPAIGN_ID,
        "dataset_id": dataset_id,
        "mbs": mbs,
        "repeat": repeat,
    }
    job = dict(template)
    job.update(
        {
            "schema": JOB_SCHEMA,
            "job_id": stable_id("h800z3bnd", identity),
            "campaign_id": CAMPAIGN_ID,
            "phase_id": PHASE_ID,
            "experiment_group": "LORA_2GPU_ZERO3_MEMORY_BOUNDARY",
            "scenario_id": f"{dataset_id}__{MODEL_ID}_cutoff{CUTOFF_LEN}_mbs{mbs}",
            "purpose": "memory_admission_head_source_backfill",
            # The whole point of this campaign: present before the job starts, so
            # the exporter will accept the resulting observation.
            "calibration_partition": {
                "role": SPLIT_ROLE,
                "split_unit_id": split_unit_id,
                "policy": SPLIT_POLICY,
            },
            "model_id": MODEL_ID,
            "train_type": "lora",
            "dataset_id": dataset_id,
            "cutoff_len": CUTOFF_LEN,
            "data_path": str((DATA_DIR / f"{dataset_id}.jsonl").resolve()),
            "dataset_profile_path": str(
                (PROFILE_DIR / f"{dataset_id}.qwen3_nothink.jsonl").resolve()
            ),
            "data_sha256": sha256_file(DATA_DIR / f"{dataset_id}.jsonl"),
            "dataset_profile_sha256": sha256_file(
                PROFILE_DIR / f"{dataset_id}.qwen3_nothink.jsonl"
            ),
            "gpu_count": GPU_COUNT,
            "mbs": mbs,
            "zero_stage": 3,
            "zero": "zero3",
            "gc": False,
            "gradient_checkpointing": False,
            "gradient_accumulation_steps": TARGET_GBS // (GPU_COUNT * mbs),
            "target_gbs": TARGET_GBS,
            "max_samples": int(dataset["samples"]),
            "fidelity": "formal_3plus10",
            "warmup_steps": 3,
            "measure_steps": 10,
            "repeat": repeat,
            "packing": False,
            "offload": False,
            "parallel_class": "gpu_partitionable",
            "requires_external_node_idle": False,
            "publication_allowed": False,
            # OOM is an expected product here, not a failure to retry.
            "oom_role": "right_censored_lower_bound",
            "oom_is_expected_evidence": True,
        }
    )
    job.pop("evaluation_partition", None)
    job.pop("calibration_purpose", None)
    return job


def build_jobs(split_unit_ids: Mapping[str, str]) -> list[dict[str, Any]]:
    rows = read_jsonl(TEMPLATE_QUEUE)
    wanted = {str(entry["dataset_id"]) for entry in TARGET_DATASETS}
    templates: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset_id = str(row.get("dataset_id"))
        if dataset_id in wanted and dataset_id not in templates:
            templates[dataset_id] = row
    missing = wanted - set(templates)
    if missing:
        raise ValueError(f"no template row for: {sorted(missing)}")

    jobs: list[dict[str, Any]] = []
    for repeat in range(REPEATS):
        for dataset in TARGET_DATASETS:
            dataset_id = str(dataset["dataset_id"])
            if dataset_id not in split_unit_ids:
                raise ValueError(
                    f"{dataset_id} has no canonical split_unit_id under {SPLIT_POLICY}; "
                    "refusing to mint one because a new id would inflate the source count"
                )
            for mbs in MBS_LADDER:
                jobs.append(
                    _job(
                        templates[dataset_id],
                        dataset,
                        split_unit_ids[dataset_id],
                        mbs,
                        repeat,
                    )
                )
    expected = len(TARGET_DATASETS) * len(MBS_LADDER) * REPEATS
    if len(jobs) != expected or len({str(r["job_id"]) for r in jobs}) != expected:
        raise ValueError(f"queue must hold exactly {expected} unique jobs")
    counts = Counter(int(r["mbs"]) for r in jobs)
    if set(counts.values()) != {len(TARGET_DATASETS) * REPEATS}:
        raise ValueError("MBS ladder is unbalanced across datasets")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    args = parser.parse_args()
    existing = [str(p) for p in (args.queue, args.design) if p.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing outputs: {existing}")

    split_unit_ids = _canonical_split_unit_ids()
    source_audit = _assert_sources_are_new(split_unit_ids)
    jobs = build_jobs(split_unit_ids)
    write_jsonl(args.queue, jobs)

    recount = read_json(RECOUNT)
    mechanism = next(
        (
            entry
            for entry in recount.get("mechanisms", [])
            if entry.get("training_mode") == "lora"
            and int(entry.get("zero_stage") or 0) == 3
            and not bool(entry.get("gradient_checkpointing"))
            and int(entry.get("gpu_count") or 0) == GPU_COUNT
            and not bool(entry.get("packing"))
        ),
        None,
    )
    if mechanism is None:
        raise SystemExit("mechanism absent from the recount")

    windows = {
        f"{entry['dataset_id']}_mbs{mbs}": _assert_measurement_window_fits(
            int(entry["samples"]), mbs, 13
        )
        for entry in TARGET_DATASETS
        for mbs in MBS_LADDER
    }

    design = {
        "schema": DESIGN_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "status": "queue_prepared_waiting_for_exact_approval",
        "gpu_training_started": False,
        "execution_authorized": False,
        "publication_allowed": False,
        "objective": (
            "Add two independent sources to lora/2gpu/zero3/no-checkpointing so "
            "V5 can fit an admission head, removing the blind spot that cost 14 "
            "of 16 ZeRO-3 jobs in the 2026-08-06 acceptance."
        ),
        "mechanism": {
            "training_mode": "lora",
            "zero_stage": 3,
            "gradient_checkpointing": False,
            "gpu_count": GPU_COUNT,
            "packing": False,
            "label_cn": "LORA 2卡 ZeRO-3 检查点关",
        },
        "recount_before": mechanism,
        "source_audit": source_audit,
        "oom_policy": (
            "OOM is expected evidence for a boundary probe: it is recorded as a "
            "right-censored lower bound and never retried or interpolated"
        ),
        "measurement_windows": windows,
        "fixed_dimensions": {
            "gpu_type": "NVIDIA H800 140GB HBM3",
            "gpu_count_per_job": GPU_COUNT,
            "model_id": MODEL_ID,
            "cutoff_len": CUTOFF_LEN,
            "target_gbs": TARGET_GBS,
            "train_type": "lora",
            "zero_stage": 3,
            "gradient_checkpointing": False,
            "packing": False,
            "offload": False,
            "precision": "bf16",
            "warmup_steps": 3,
            "measure_steps": 10,
            "repeat_count": REPEATS,
        },
        "matrix": {
            "datasets": list(TARGET_DATASETS),
            "mbs_ladder": list(MBS_LADDER),
            "jobs": len(jobs),
            "bracketing_rationale": (
                "at 14B / cutoff 4096 the mechanism fits at MBS4 and OOMs at MBS8, "
                "so MBS 2/4/8 spans safe, marginal and unsafe"
            ),
        },
        "calibration_partition_bound_in_payload": {
            "role": SPLIT_ROLE,
            "policy": SPLIT_POLICY,
            "split_unit_ids": {
                str(entry["dataset_id"]): split_unit_ids[str(entry["dataset_id"])]
                for entry in TARGET_DATASETS
            },
            "why": (
                "the exporter reads the partition only from the authorized payload "
                "and refuses post-hoc annotation, which is why 52 existing stage2 "
                "observations for this mechanism are unusable"
            ),
        },
        "scheduler_design": {
            "gpu_ids": [4, 5, 6, 7],
            "disjoint_pairs": [[4, 5], [6, 7]],
            "parallel_jobs_per_wave": 2,
            "planned_waves": -(-len(jobs) // 2),
            "preemption_allowed": False,
            "reserved_for_others": [0, 1, 2, 3],
        },
        "bindings": {
            "queue": {
                "path": str(args.queue.resolve()),
                "sha256": sha256_file(args.queue),
            },
            "template_queue": {
                "path": str(TEMPLATE_QUEUE.resolve()),
                "sha256": sha256_file(TEMPLATE_QUEUE),
            },
        },
        "ordered_job_ids": [str(r["job_id"]) for r in jobs],
        "ordered_job_payload_sha256": sha256_json(jobs),
        "limitations": [
            "a fitted head will be scoped to 14B at cutoff 4096; other cutoffs "
            "and model sizes stay uncovered",
            "two sources reach the threshold exactly, leaving no margin if one "
            "source turns out to be unusable",
        ],
    }
    design["report_sha256"] = sha256_json(design)
    write_json(args.design, design)
    print(
        json.dumps(
            {
                "queue": str(args.queue.resolve()),
                "design": str(args.design.resolve()),
                "jobs": len(jobs),
                "source_audit": source_audit,
                "measurement_windows": windows,
                "gpu_training_started": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
