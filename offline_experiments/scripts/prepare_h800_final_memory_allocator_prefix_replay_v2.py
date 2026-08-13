#!/usr/bin/env python3
"""Prepare the LoRA allocator-history repair for peak replay.

Stage-one v1 proved that repeating only the largest real micro-batch reproduces
``max_memory_allocated`` but not ``max_memory_reserved`` for LoRA.  This repair
does not select a new model or touch the final blind partition.  It replays the
exact original sampler prefix until every rank has seen its own largest aligned
sequence length.  The prefix therefore preserves the allocator history that is
missing from cold peak-batch replay while remaining shorter than full coverage.

Only the two failed LoRA replay arms are materialized.  Their full-coverage
references remain the immutable successful v1 observations.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from common import (
    ARTIFACT_DIR,
    DATA_DIR,
    MATRIX_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from prepare_h800_final_memory_peak_replay_stage1_v1 import (
    DATA_SEED,
    DATASET_INFO,
    GPU_IDS,
    TARGET_GBS,
    _aligned,
    _sampler_batches,
)

CAMPAIGN_ID = "h800_final_memory_allocator_prefix_replay_20260810_v2"
PHASE_ID = "h800_final_memory_allocator_prefix_replay_v2"
JOB_SCHEMA = "sft_h800_final_memory_allocator_prefix_replay_job/v2"
DESIGN_SCHEMA = "sft_h800_final_memory_allocator_prefix_replay_design/v2"
DATA_SCHEMA = "sft_h800_final_memory_allocator_prefix_replay_data/v2"
EXPECTED_JOBS = 2

V1_QUEUE = MATRIX_DIR / "h800_final_memory_peak_replay_jobs_v1.jsonl"
V1_RESULTS = ARTIFACT_DIR / "h800_final_memory_peak_replay_results_v1.json"
OUTPUT_DATA_DIR = DATA_DIR / "final_memory_allocator_prefix_replay_v2"
PROFILE_DIR = ARTIFACT_DIR / "final_memory_allocator_prefix_replay_v2" / "profiles"
DEFAULT_QUEUE = MATRIX_DIR / "h800_final_memory_allocator_prefix_replay_jobs_v2.jsonl"
DEFAULT_DATA_BUNDLE = (
    ARTIFACT_DIR / "h800_final_memory_allocator_prefix_replay_data_v2.json"
)
DEFAULT_DESIGN = (
    ARTIFACT_DIR / "h800_final_memory_allocator_prefix_replay_design_v2.json"
)
STAGING_DIR = ROOT / "final_memory_allocator_prefix_replay_v2_staging"
DEFAULT_EXPERIMENT = (
    STAGING_DIR / "experiment.h800_final_memory_allocator_prefix_replay_v2.json"
)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _latest_success(job_id: str) -> dict[str, Any]:
    root = ROOT / "results" / job_id
    latest_path = root / "latest_attempt.json"
    if not latest_path.is_file():
        raise FileNotFoundError(f"missing v1 full baseline: {latest_path}")
    latest = read_json(latest_path)
    if (
        latest.get("state") != "complete"
        or latest.get("classification") != "success"
        or latest.get("calibration_eligible") is not True
    ):
        raise ValueError(f"v1 full baseline is not an eligible success: {job_id}")
    attempt_root = root / str(latest["attempt_path"])
    status_path = attempt_root / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(status_path)
    status = read_json(status_path)
    if status.get("classification") != "success":
        raise ValueError(f"v1 status is not success: {job_id}")
    summaries = sorted((attempt_root / "metrics").glob("summary.rank*.json"))
    if not summaries:
        raise FileNotFoundError(f"missing v1 summaries: {job_id}")
    return {
        "job_id": job_id,
        "latest_attempt_path": str(latest_path.resolve()),
        "latest_attempt_sha256": sha256_file(latest_path),
        "attempt_path": str(attempt_root.resolve()),
        "status_path": str(status_path.resolve()),
        "status_sha256": sha256_file(status_path),
        "summary_bindings": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in summaries
        ],
    }


def _source_index(sample_id: str) -> int:
    marker = ":source_row:"
    if marker not in sample_id:
        raise ValueError(f"source sample_id lacks {marker!r}: {sample_id}")
    return int(sample_id.rsplit(marker, 1)[1]) - 1


def _prefix_contract(
    lengths: list[int], *, cutoff_len: int, mbs: int, world_size: int, ga: int
) -> dict[str, Any]:
    by_rank = _sampler_batches(
        len(lengths), mbs=mbs, world_size=world_size, data_seed=DATA_SEED
    )
    rank_contracts = []
    for rank, batches in enumerate(by_rank):
        padded = [
            _aligned(max(min(lengths[index], cutoff_len) for index in batch))
            for batch in batches
        ]
        rank_max = max(padded)
        first_microbatch = padded.index(rank_max)
        rank_contracts.append(
            {
                "rank": rank,
                "rank_max_padded_sequence_length": rank_max,
                "first_rank_max_microbatch_index_zero_based": first_microbatch,
                "first_rank_max_optimizer_step_one_based": first_microbatch // ga + 1,
                "source_row_indices_zero_based": batches[first_microbatch],
                "clipped_sequence_lengths": [
                    min(lengths[index], cutoff_len)
                    for index in batches[first_microbatch]
                ],
            }
        )
    prefix_steps = max(
        int(row["first_rank_max_optimizer_step_one_based"])
        for row in rank_contracts
    )
    prefix_microbatches_per_rank = prefix_steps * ga
    return {
        "selection_rule": (
            "exact original sampler prefix through the earliest optimizer step "
            "by which every rank has observed its own full-plan maximum aligned "
            "padded sequence length"
        ),
        "data_seed": DATA_SEED,
        "gradient_accumulation_steps": ga,
        "prefix_optimizer_steps": prefix_steps,
        "prefix_microbatches_per_rank": prefix_microbatches_per_rank,
        "prefix_rows": prefix_steps * TARGET_GBS,
        "full_plan_optimizer_steps": (len(lengths) + TARGET_GBS - 1) // TARGET_GBS,
        "rank_contracts": rank_contracts,
    }


def _materialize_prefix(
    source_rows: list[dict[str, Any]],
    source_lengths: list[int],
    source_label_lengths: list[int],
    *,
    prefix_rows: int,
) -> tuple[list[dict[str, Any]], list[int], list[int], list[int]]:
    generator = torch.Generator().manual_seed(DATA_SEED)
    original_order = torch.randperm(len(source_rows), generator=generator).tolist()
    desired_source_indices = original_order[:prefix_rows]
    if len(desired_source_indices) != prefix_rows:
        raise ValueError("requested prefix exceeds the original sampler epoch")

    replay_generator = torch.Generator().manual_seed(DATA_SEED)
    replay_permutation = torch.randperm(prefix_rows, generator=replay_generator).tolist()
    output: list[dict[str, Any] | None] = [None] * prefix_rows
    output_lengths = [0] * prefix_rows
    output_labels = [0] * prefix_rows
    for emitted_position, storage_index in enumerate(replay_permutation):
        source_index = int(desired_source_indices[emitted_position])
        row = dict(source_rows[source_index])
        row["sample_id"] = (
            f"allocator_prefix_position:{emitted_position}:"
            f"source_row:{source_index + 1}"
        )
        output[storage_index] = row
        output_lengths[storage_index] = int(source_lengths[source_index])
        output_labels[storage_index] = int(source_label_lengths[source_index])
    if any(row is None for row in output):
        raise RuntimeError("inverse prefix permutation is incomplete")
    return (
        [dict(row) for row in output if row is not None],
        output_lengths,
        output_labels,
        [int(value) for value in desired_source_indices],
    )


def _verify_prefix(
    *,
    original_lengths: list[int],
    replay_rows: list[dict[str, Any]],
    replay_lengths: list[int],
    cutoff_len: int,
    mbs: int,
    world_size: int,
    prefix_microbatches_per_rank: int,
) -> dict[str, Any]:
    original = _sampler_batches(
        len(original_lengths), mbs=mbs, world_size=world_size, data_seed=DATA_SEED
    )
    replay = _sampler_batches(
        len(replay_lengths), mbs=mbs, world_size=world_size, data_seed=DATA_SEED
    )
    replay_source_indices = [_source_index(str(row["sample_id"])) for row in replay_rows]
    rank_checks = []
    all_exact = True
    for rank in range(world_size):
        expected = original[rank][:prefix_microbatches_per_rank]
        actual = [
            [replay_source_indices[index] for index in batch]
            for batch in replay[rank]
        ]
        expected_shapes = [
            [min(original_lengths[index], cutoff_len) for index in batch]
            for batch in expected
        ]
        actual_shapes = [
            [min(replay_lengths[index], cutoff_len) for index in batch]
            for batch in replay[rank]
        ]
        exact = actual == expected and actual_shapes == expected_shapes
        all_exact = all_exact and exact
        rank_checks.append(
            {
                "rank": rank,
                "expected_microbatches": len(expected),
                "observed_microbatches": len(actual),
                "source_indices_exact": actual == expected,
                "clipped_shapes_exact": actual_shapes == expected_shapes,
                "first_padded_sequence_length": _aligned(max(actual_shapes[0])),
                "last_padded_sequence_length": _aligned(max(actual_shapes[-1])),
            }
        )
    if not all_exact:
        raise RuntimeError("allocator-prefix replay does not reproduce the full sampler prefix")
    return {
        "sampler": "accelerate.SeedableRandomSampler+BatchSamplerShard",
        "data_seed": DATA_SEED,
        "all_rank_prefixes_exact": True,
        "rank_checks": rank_checks,
    }


def _experiment(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    source = read_json(
        ROOT
        / "final_memory_peak_replay_stage1_staging"
        / "experiment.h800_final_memory_peak_replay_stage1_v1.json"
    )
    experiment = copy.deepcopy(source)
    experiment["training_scope"] = {
        **dict(experiment.get("training_scope") or {}),
        "phase_id": PHASE_ID,
        "model_ids": sorted({str(row["model_id"]) for row in jobs}),
        "gpu_ids": list(GPU_IDS),
        "exclusive_node_gpu_ids": list(GPU_IDS),
        "max_gpu_count": 2,
        "objective": (
            "repair LoRA reserved-memory replay by preserving the exact "
            "pre-peak allocator history"
        ),
    }
    experiment["measurement"] = {
        **dict(experiment.get("measurement") or {}),
        "throughput_warmup_steps": 0,
        "throughput_measure_steps": max(int(row["measure_steps"]) for row in jobs),
        "scheduler_order_policy": "parallel_queue",
        "rerun_on_unhealthy_result": False,
    }
    experiment["datasets"] = [
        {
            "id": str(row["dataset_id"]),
            "category": "allocator_prefix_replay",
            "target_cutoffs": [int(row["cutoff_len"])],
        }
        for row in jobs
    ]
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--data-bundle", type=Path, default=DEFAULT_DATA_BUNDLE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    outputs = (args.queue, args.data_bundle, args.design, args.experiment)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing frozen outputs: {existing}")

    v1_results = read_json(V1_RESULTS)
    if v1_results.get("status") != "measurement_equivalence_failed":
        raise ValueError("v1 failure evidence is absent or changed")
    if v1_results.get("final_acceptance_outcomes_observed") != 0:
        raise ValueError("v1 unexpectedly observed final blind outcomes")

    v1_rows = read_jsonl(V1_QUEUE)
    full_lora = [
        dict(row)
        for row in v1_rows
        if row.get("train_type") == "lora" and row.get("replay_mode") == "full_coverage"
    ]
    if len(full_lora) != EXPECTED_JOBS:
        raise ValueError("v1 does not contain exactly two LoRA full baselines")

    registry = read_json(DATASET_INFO)
    jobs: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    for full in full_lora:
        source_rows = read_jsonl(Path(str(full["data_path"])))
        profile_rows = read_jsonl(Path(str(full["runtime_dataset_profile_path"])))
        if len(source_rows) != len(profile_rows):
            raise ValueError(f"full data/profile row mismatch: {full['job_id']}")
        source_lengths = [int(row["total_tokens"]) for row in profile_rows]
        source_label_lengths = [int(row["label_tokens"]) for row in profile_rows]
        ga = int(full["gradient_accumulation_steps"])
        contract = _prefix_contract(
            source_lengths,
            cutoff_len=int(full["cutoff_len"]),
            mbs=int(full["mbs"]),
            world_size=int(full["gpu_count"]),
            ga=ga,
        )
        replay_rows, replay_lengths, replay_labels, desired_indices = _materialize_prefix(
            source_rows,
            source_lengths,
            source_label_lengths,
            prefix_rows=int(contract["prefix_rows"]),
        )
        source_id = str(full["source_dataset_id"])
        dataset_id = f"prefix2_{source_id.replace('-', '_')}__allocator_history"
        data_path = OUTPUT_DATA_DIR / f"{source_id}__allocator_prefix.jsonl"
        profile_path = PROFILE_DIR / f"{source_id}__allocator_prefix.qwen3_nothink.jsonl"
        _atomic_jsonl(data_path, replay_rows)
        _atomic_jsonl(
            profile_path,
            [
                {
                    "sample_id": row["sample_id"],
                    "total_tokens": total,
                    "label_tokens": label,
                    "turns": 3 if row.get("system") else 2,
                    "assistant_turns": 1,
                }
                for row, total, label in zip(
                    replay_rows, replay_lengths, replay_labels
                )
            ],
        )
        verification = _verify_prefix(
            original_lengths=source_lengths,
            replay_rows=replay_rows,
            replay_lengths=replay_lengths,
            cutoff_len=int(full["cutoff_len"]),
            mbs=int(full["mbs"]),
            world_size=int(full["gpu_count"]),
            prefix_microbatches_per_rank=int(contract["prefix_microbatches_per_rank"]),
        )
        registry_entry = {
            "file_name": str(data_path.resolve().relative_to(DATA_DIR.resolve())),
            "columns": {"prompt": "prompt", "response": "response", "system": "system"},
        }
        existing_entry = registry.get(dataset_id)
        if existing_entry is not None and existing_entry != registry_entry:
            raise ValueError(f"dataset registry collision: {dataset_id}")
        registry[dataset_id] = registry_entry

        job = copy.deepcopy(full)
        job.update(
            {
                "schema": JOB_SCHEMA,
                "job_id": stable_id(
                    "h800prefix2",
                    {"v1_full_job_id": full["job_id"], "contract": contract},
                ),
                "campaign_id": CAMPAIGN_ID,
                "phase_id": PHASE_ID,
                "experiment_group": "FINAL_MEMORY_ALLOCATOR_PREFIX_REPLAY_V2",
                "evidence_role": "measurement_equivalence_repair_development_only",
                "purpose": "exact_sampler_prefix_allocator_history_replay",
                "replay_mode": "allocator_prefix_replay",
                "dataset_id": dataset_id,
                "runtime_dataset_id": dataset_id,
                "data_path": str(data_path.resolve()),
                "data_sha256": sha256_file(data_path),
                "runtime_dataset_profile_path": str(profile_path.resolve()),
                "runtime_dataset_profile_sha256": sha256_file(profile_path),
                "max_samples": int(contract["prefix_rows"]),
                "warmup_steps": 0,
                "measure_steps": int(contract["prefix_optimizer_steps"]),
                "publication_allowed": False,
                "oom_role": "measurement_repair_failure_not_model_evidence",
                "oom_is_expected_evidence": False,
                "calibration_partition": {
                    "role": "development_measurement",
                    "policy": "final_memory_allocator_prefix_replay_v2",
                    "split_unit_id": source_id,
                },
                "allocator_prefix_contract": contract,
                "allocator_prefix_sampler_verification": verification,
                "v1_full_baseline_job_id": str(full["job_id"]),
                "v1_pair_id": str(full["pair_id"]),
            }
        )
        job.pop("peak_batch_contract", None)
        job.pop("replay_sampler_verification", None)
        jobs.append(job)
        baselines.append(_latest_success(str(full["job_id"])))
        profiles.append(
            {
                "source_dataset_id": source_id,
                "dataset_id": dataset_id,
                "rows": len(replay_rows),
                "data_path": str(data_path.resolve()),
                "data_sha256": sha256_file(data_path),
                "profile_path": str(profile_path.resolve()),
                "profile_sha256": sha256_file(profile_path),
                "desired_original_source_indices_sha256": sha256_json(desired_indices),
                "contract": contract,
                "verification": verification,
            }
        )

    if len(jobs) != EXPECTED_JOBS or len({row["job_id"] for row in jobs}) != EXPECTED_JOBS:
        raise RuntimeError("repair queue size or uniqueness drifted")
    if any((ROOT / "results" / str(row["job_id"])).exists() for row in jobs):
        raise RuntimeError("one or more repair result directories already exist")

    write_json(DATASET_INFO, registry)
    write_jsonl(args.queue, jobs)
    write_json(args.experiment, _experiment(jobs))
    data_bundle: dict[str, Any] = {
        "schema": DATA_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "status": "prepared_before_v2_gpu_outcomes",
        "gpu_training_started": False,
        "gpu_outcomes_observed": 0,
        "final_acceptance_sources_read": 0,
        "profiles": profiles,
        "v1_full_baselines": baselines,
        "bindings": {
            "v1_queue": {"path": str(V1_QUEUE.resolve()), "sha256": sha256_file(V1_QUEUE)},
            "v1_results": {"path": str(V1_RESULTS.resolve()), "sha256": sha256_file(V1_RESULTS)},
            "dataset_registry": {"path": str(DATASET_INFO.resolve()), "sha256": sha256_file(DATASET_INFO)},
        },
    }
    data_bundle["report_sha256"] = sha256_json(data_bundle)
    write_json(args.data_bundle, data_bundle)
    design: dict[str, Any] = {
        "schema": DESIGN_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "status": "prepared_waiting_for_gpu_0_3_preflight",
        "gpu_training_started": False,
        "gpu_outcomes_observed": 0,
        "final_acceptance_sources_read": 0,
        "execution_authorized": True,
        "publication_allowed": False,
        "objective": (
            "Test whether an exact profile-derived sampler prefix repairs the two "
            "LoRA reserved-memory replay failures without rerunning full baselines."
        ),
        "authorized_gpu_ids": list(GPU_IDS),
        "old_missing_34_jobs_included": False,
        "jobs": EXPECTED_JOBS,
        "ordered_job_ids": [str(row["job_id"]) for row in jobs],
        "ordered_job_payload_sha256": sha256_json(jobs),
        "acceptance_thresholds": {
            "combined_four_pair_mean_absolute_replay_error_max": 0.03,
            "combined_four_pair_maximum_replay_underprediction": 0.03,
            "lora_no_directionally_consistent_underprediction": True,
            "all_allocator_prefixes_exact": True,
        },
        "bindings": {
            "queue": {"path": str(args.queue.resolve()), "sha256": sha256_file(args.queue)},
            "experiment": {"path": str(args.experiment.resolve()), "sha256": sha256_file(args.experiment)},
            "data_bundle": {"path": str(args.data_bundle.resolve()), "sha256": sha256_file(args.data_bundle)},
            "v1_results": {"path": str(V1_RESULTS.resolve()), "sha256": sha256_file(V1_RESULTS)},
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(args.design, design)


if __name__ == "__main__":
    main()
