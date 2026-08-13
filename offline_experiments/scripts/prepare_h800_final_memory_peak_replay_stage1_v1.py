#!/usr/bin/env python3
"""Prepare the four-pair full-coverage versus exact-peak-replay calibration.

The final-acceptance datasets remain outcome blind.  This script only reads the
four sources pre-registered as development/measurement data.  For every source
it reconstructs the exact seedable sampler order used by Transformers and
Accelerate, selects the highest-pressure real micro-batch, then constructs a
replay dataset by inverting the same random permutation.  Consequently the
runtime sampler yields the selected real batch on every replay micro-batch.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fit_h800_unified_resource_partial_v1 as base
import h800_unified_bounded_memory_v3_data as v3_data
import torch
from accelerate.data_loader import BatchSamplerShard, SeedableRandomSampler
from common import (
    ARTIFACT_DIR,
    DATA_DIR,
    MATRIX_DIR,
    ROOT,
    percentile,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from h800_unified_bounded_memory_model import load_artifact, predict_records
from prepare_h800_bounded_memory_v2_fresh_data_v1 import TrainingEncoder
from torch.utils.data import BatchSampler

CAMPAIGN_ID = "h800_final_memory_peak_replay_stage1_20260810_v1"
PHASE_ID = "h800_final_memory_peak_replay_stage1_v1"
JOB_SCHEMA = "sft_h800_final_memory_peak_replay_job/v1"
DESIGN_SCHEMA = "sft_h800_final_memory_peak_replay_design/v1"
DATA_SCHEMA = "sft_h800_final_memory_peak_replay_data/v1"
PREDICTION_SCHEMA = "sft_h800_final_memory_peak_replay_frozen_predictions/v1"
GPU_IDS = (0, 1, 2, 3)
TARGET_GBS = 32
DATA_SEED = 20260716
REPLAY_WARMUP_STEPS = 1
REPLAY_MEASURE_STEPS = 2
EXPECTED_PAIRS = 4
EXPECTED_JOBS = 8
MAX_SAFE_PRESSURE = 0.90

SELECTION = ARTIFACT_DIR / "h800_final_memory_acceptance_s3_selection_v1.json"
MODEL_INVENTORY = ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_v1.json"
V3_ARTIFACT = ARTIFACT_DIR / "h800_unified_bounded_memory_candidate_v3.json"
TEMPLATE_QUEUE = MATRIX_DIR / "h800_unified_resource_evidence_jobs_v1.jsonl"
DATASET_INFO = DATA_DIR / "dataset_info.json"
OUTPUT_DATA_DIR = DATA_DIR / "final_memory_peak_replay_stage1_v1"
PROFILE_DIR = ARTIFACT_DIR / "final_memory_peak_replay_stage1_v1" / "profiles"
DEFAULT_DATA_BUNDLE = ARTIFACT_DIR / "h800_final_memory_peak_replay_data_v1.json"
DEFAULT_QUEUE = MATRIX_DIR / "h800_final_memory_peak_replay_jobs_v1.jsonl"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_final_memory_peak_replay_design_v1.json"
DEFAULT_PREDICTIONS = (
    ARTIFACT_DIR / "h800_final_memory_peak_replay_frozen_predictions_v1.json"
)
STAGING_DIR = ROOT / "final_memory_peak_replay_stage1_staging"
DEFAULT_EXPERIMENT = (
    STAGING_DIR / "experiment.h800_final_memory_peak_replay_stage1_v1.json"
)

# The mechanisms are frozen by the acceptance plan.  MBS is selected before
# outcomes from a small legal set using only the already-frozen V3 upper bound.
WORKLOADS: tuple[dict[str, Any], ...] = (
    {
        "source_dataset_id": "dataset-5qtqde-1785830918",
        "profile_role": "极端长尾",
        "model_id": "qwen3_4b",
        "train_type": "lora",
        "gpu_count": 1,
        "zero_stage": 0,
        "gc": False,
        "cutoff_len": 8192,
        "mbs_candidates": (1, 2, 4),
    },
    {
        "source_dataset_id": "dataset-d3olaw-1785988468",
        "profile_role": "持续长文本",
        "model_id": "qwen3_8b",
        "train_type": "full",
        "gpu_count": 2,
        "zero_stage": 3,
        "gc": True,
        "cutoff_len": 6144,
        "mbs_candidates": (1, 2, 4),
    },
    {
        "source_dataset_id": "dataset-u6wwzq-1786083796",
        "profile_role": "集中中等长度",
        "model_id": "qwen3_4b",
        "train_type": "full",
        "gpu_count": 1,
        "zero_stage": 0,
        "gc": False,
        "cutoff_len": 2048,
        "mbs_candidates": (1, 2, 4),
    },
    {
        "source_dataset_id": "dataset-wfqhvw-1785941727",
        "profile_role": "中高长度",
        "model_id": "qwen3_8b",
        "train_type": "lora",
        "gpu_count": 2,
        "zero_stage": 2,
        "gc": True,
        "cutoff_len": 4096,
        "mbs_candidates": (1, 2, 4, 8),
    },
)

ACCEPTANCE_THRESHOLDS = {
    "mean_absolute_replay_error_max": 0.03,
    "maximum_replay_underprediction": 0.03,
    "full_and_lora_no_directionally_consistent_underprediction": True,
    "minimum_complete_pairs": 4,
}


def _normalize(value: Any, *, path: Path, line_number: int) -> dict[str, str]:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict):
        raise TypeError(f"{path}:{line_number} is not an SFT object")
    required = ("system", "prompt", "response")
    if not all(isinstance(value.get(field), str) for field in required):
        raise ValueError(f"{path}:{line_number} lacks string system/prompt/response")
    return {field: str(value[field]) for field in required}


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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


def _statistics(lengths: list[int], cutoff_len: int) -> dict[str, Any]:
    clipped = [min(value, cutoff_len) for value in lengths]
    return {
        "rows": len(lengths),
        "minimum_total_tokens": min(lengths),
        "mean_total_tokens": sum(lengths) / len(lengths),
        "p50_total_tokens": percentile(lengths, 50),
        "p90_total_tokens": percentile(lengths, 90),
        "p95_total_tokens": percentile(lengths, 95),
        "p99_total_tokens": percentile(lengths, 99),
        "maximum_total_tokens": max(lengths),
        "maximum_clipped_tokens": max(clipped),
        "cutoff_len": cutoff_len,
        "truncation_fraction": sum(value > cutoff_len for value in lengths)
        / len(lengths),
        "tokens_retained_ratio": sum(clipped) / sum(lengths),
    }


def _aligned(value: int) -> int:
    return int(math.ceil(value / 8.0) * 8)


def _sampler_batches(
    rows: int, *, mbs: int, world_size: int, data_seed: int
) -> list[list[list[int]]]:
    by_rank: list[list[list[int]]] = []
    for rank in range(world_size):
        sampler = SeedableRandomSampler(range(rows), data_seed=data_seed)
        batch_sampler = BatchSampler(sampler, batch_size=mbs, drop_last=False)
        shard = BatchSamplerShard(
            batch_sampler,
            num_processes=world_size,
            process_index=rank,
            split_batches=False,
            even_batches=True,
        )
        by_rank.append([[int(value) for value in batch] for batch in shard])
    if len({len(value) for value in by_rank}) != 1:
        raise RuntimeError("distributed sampler produced unequal rank lengths")
    return by_rank


def _select_peak_batch(
    lengths: list[int], *, cutoff_len: int, mbs: int, world_size: int
) -> dict[str, Any]:
    batches = _sampler_batches(
        len(lengths), mbs=mbs, world_size=world_size, data_seed=DATA_SEED
    )
    candidates = []
    for rank, rank_batches in enumerate(batches):
        for microbatch_index, indices in enumerate(rank_batches):
            clipped = [min(lengths[index], cutoff_len) for index in indices]
            candidates.append(
                {
                    "rank": rank,
                    "microbatch_index": microbatch_index,
                    "source_row_indices_zero_based": indices,
                    "raw_sequence_lengths": [lengths[index] for index in indices],
                    "clipped_sequence_lengths": clipped,
                    "padded_sequence_length": _aligned(max(clipped)),
                    "sum_clipped_sequence_lengths": sum(clipped),
                }
            )
    peak = max(
        candidates,
        key=lambda row: (
            int(row["padded_sequence_length"]),
            int(row["sum_clipped_sequence_lengths"]),
            -int(row["microbatch_index"]),
            -int(row["rank"]),
        ),
    )
    ga = TARGET_GBS // (world_size * mbs)
    peak["optimizer_step_one_based"] = int(peak["microbatch_index"]) // ga + 1
    peak["microbatch_within_optimizer_step_zero_based"] = (
        int(peak["microbatch_index"]) % ga
    )
    peak["rank_microbatches_per_epoch"] = len(batches[0])
    peak["gradient_accumulation_steps"] = ga
    return peak


def _inverse_permuted_replay_rows(
    source_rows: list[dict[str, Any]], peak_indices: list[int], *, replay_rows: int
) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(DATA_SEED)
    permutation = torch.randperm(replay_rows, generator=generator).tolist()
    desired_order = [
        peak_indices[index % len(peak_indices)] for index in range(replay_rows)
    ]
    output: list[dict[str, Any] | None] = [None] * replay_rows
    for emitted_position, storage_index in enumerate(permutation):
        source_index = desired_order[emitted_position]
        output[storage_index] = {
            "sample_id": f"replay_position:{emitted_position}:source_row:{source_index + 1}",
            **source_rows[source_index],
        }
    if any(row is None for row in output):
        raise RuntimeError("inverse replay permutation is incomplete")
    return [dict(row) for row in output if row is not None]


def _verify_replay(
    replay_lengths: list[int],
    *,
    expected_lengths: list[int],
    cutoff_len: int,
    mbs: int,
    world_size: int,
) -> dict[str, Any]:
    batches = _sampler_batches(
        len(replay_lengths), mbs=mbs, world_size=world_size, data_seed=DATA_SEED
    )
    expected_clipped = [min(value, cutoff_len) for value in expected_lengths]
    signatures = []
    for rank, rank_batches in enumerate(batches):
        for microbatch_index, indices in enumerate(rank_batches):
            actual = [min(replay_lengths[index], cutoff_len) for index in indices]
            signatures.append(
                {
                    "rank": rank,
                    "microbatch_index": microbatch_index,
                    "clipped_sequence_lengths": actual,
                    "exact_peak_batch_match": actual == expected_clipped,
                }
            )
    if not signatures or not all(row["exact_peak_batch_match"] for row in signatures):
        raise RuntimeError("replay sampler does not reproduce the exact peak batch")
    return {
        "sampler": "accelerate.SeedableRandomSampler+BatchSamplerShard",
        "data_seed": DATA_SEED,
        "rank_microbatches": len(signatures),
        "all_rank_microbatches_exact_peak_batch": True,
        "expected_clipped_sequence_lengths": expected_clipped,
        "padded_sequence_length": _aligned(max(expected_clipped)),
    }


def _template() -> dict[str, Any]:
    return dict(
        next(
            row
            for row in read_jsonl(TEMPLATE_QUEUE)
            if row["model_id"] == "qwen3_8b"
            and row["train_type"] == "lora"
            and int(row["gpu_count"]) == 2
            and not bool(row["packing"])
        )
    )


def _prediction_record(
    job: dict[str, Any],
    *,
    inventory: dict[str, Any],
    models: dict[str, dict[str, Any]],
    capacity: int,
    profile_cache: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    reference, features = base._current_features(
        job,
        model_by_id=models,
        fixed_lora=inventory["fixed_lora"],
        capacity_bytes=capacity,
        profile_cache=profile_cache,
    )
    return {
        "record_id": f"peak-replay-stage1::{job['pair_id']}",
        "source_id": str(job["source_dataset_id"]),
        "origin": CAMPAIGN_ID,
        "role": "development_measurement_calibration",
        "state": "exact",
        "reference_bytes": reference,
        "target_reserved_bytes": 1.0,
        "censor_lower_bytes": None,
        "features": features,
        "model_id": str(job["model_id"]),
        "train_type": str(job["train_type"]),
        "gpu_count": int(job["gpu_count"]),
        "zero_stage": int(job["zero_stage"]),
        "gc": bool(job["gc"]),
        "mbs": int(job["mbs"]),
        "cutoff_len": int(job["cutoff_len"]),
        "packing": False,
        "profile_sha256": str(job["dataset_profile_sha256"]),
    }


def _provisional_job(
    workload: dict[str, Any],
    *,
    mbs: int,
    model: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    zero_stage = int(workload["zero_stage"])
    return {
        "pair_id": stable_id("h800peakpair", {**workload, "mbs": mbs}),
        **workload,
        "dataset_id": str(profile["full_dataset_id"]),
        "data_path": str(profile["full_data_path"]),
        "data_sha256": str(profile["full_data_sha256"]),
        "dataset_profile_path": str(profile["full_profile_path"]),
        "dataset_profile_sha256": str(profile["full_profile_sha256"]),
        "aligned_effective_sequence": int(profile["aligned_effective_sequence"]),
        "model_family": str(model["family"]),
        "model_path": str(model["path"]),
        "tokenizer_path": str(model["path"]),
        "model_parameters": int(model["actual_parameters"]),
        "template": "qwen3_nothink",
        "mbs": mbs,
        "zero": "none" if zero_stage == 0 else f"zero{zero_stage}",
        "gradient_checkpointing": bool(workload["gc"]),
        "gradient_accumulation_steps": TARGET_GBS // (int(workload["gpu_count"]) * mbs),
        "target_gbs": TARGET_GBS,
        "packing": False,
        "mechanism_id": (
            f"{workload['train_type']}_zero{zero_stage}_gc"
            f"{int(bool(workload['gc']))}_{workload['gpu_count']}gpu_pack0"
        ),
    }


def _select_mbs(
    workload: dict[str, Any],
    *,
    model: dict[str, Any],
    profile: dict[str, Any],
    inventory: dict[str, Any],
    models: dict[str, dict[str, Any]],
    capacity: int,
    artifact: dict[str, Any],
    profile_cache: dict[tuple[str, int, int], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    choices = []
    for mbs in workload["mbs_candidates"]:
        if TARGET_GBS % (int(workload["gpu_count"]) * int(mbs)):
            continue
        job = _provisional_job(workload, mbs=int(mbs), model=model, profile=profile)
        record = _prediction_record(
            job,
            inventory=inventory,
            models=models,
            capacity=capacity,
            profile_cache=profile_cache,
        )
        prediction = predict_records([record], artifact)[0]
        pressure = float(prediction["admission_upper_bytes"]) / float(
            prediction["safe_limit_bytes"]
        )
        choices.append(
            {
                "mbs": int(mbs),
                "job": job,
                "record": record,
                "prediction": prediction,
                "upper_to_safe_limit": pressure,
                "eligible": bool(prediction["admitted"])
                and pressure <= MAX_SAFE_PRESSURE,
            }
        )
    eligible = [row for row in choices if row["eligible"]]
    if not eligible:
        summary = [
            {
                "mbs": row["mbs"],
                "admitted": row["prediction"]["admitted"],
                "upper_to_safe_limit": row["upper_to_safe_limit"],
            }
            for row in choices
        ]
        raise ValueError(
            f"no conservative MBS for {workload['source_dataset_id']}: {summary}"
        )
    selected = max(eligible, key=lambda row: int(row["mbs"]))
    audit = [
        {
            "mbs": row["mbs"],
            "center_gib": float(row["prediction"]["center_bytes"]) / (1 << 30),
            "upper_gib": float(row["prediction"]["admission_upper_bytes"]) / (1 << 30),
            "safe_limit_gib": float(row["prediction"]["safe_limit_bytes"]) / (1 << 30),
            "upper_to_safe_limit": row["upper_to_safe_limit"],
            "admitted": bool(row["prediction"]["admitted"]),
            "eligible": bool(row["eligible"]),
        }
        for row in choices
    ]
    return dict(selected["job"]), dict(selected["prediction"]), audit


def _job(
    provisional: dict[str, Any],
    *,
    mode: str,
    profile: dict[str, Any],
    template: dict[str, Any],
) -> dict[str, Any]:
    full = mode == "full_coverage"
    runtime_dataset_id = (
        str(profile["full_dataset_id"]) if full else str(profile["replay_dataset_id"])
    )
    runtime_data_path = (
        str(profile["full_data_path"]) if full else str(profile["replay_data_path"])
    )
    runtime_data_sha256 = (
        str(profile["full_data_sha256"]) if full else str(profile["replay_data_sha256"])
    )
    runtime_profile_path = (
        str(profile["full_profile_path"])
        if full
        else str(profile["replay_profile_path"])
    )
    runtime_profile_sha256 = (
        str(profile["full_profile_sha256"])
        if full
        else str(profile["replay_profile_sha256"])
    )
    full_steps = int(profile["full_plan_optimizer_steps"])
    job = dict(template)
    job.update(provisional)
    job.update(
        {
            "schema": JOB_SCHEMA,
            "job_id": stable_id(
                "h800peak1", {"pair_id": provisional["pair_id"], "mode": mode}
            ),
            "campaign_id": CAMPAIGN_ID,
            "phase_id": PHASE_ID,
            "kind": "throughput",
            "experiment_group": "FINAL_MEMORY_PEAK_REPLAY_STAGE1",
            "evidence_role": "measurement_equivalence_development_only",
            "purpose": "full_plan_peak_vs_exact_peak_batch_replay",
            "replay_mode": mode,
            "dataset_id": runtime_dataset_id,
            "runtime_dataset_id": runtime_dataset_id,
            "data_path": runtime_data_path,
            "data_sha256": runtime_data_sha256,
            "runtime_dataset_profile_path": runtime_profile_path,
            "runtime_dataset_profile_sha256": runtime_profile_sha256,
            # Model features intentionally describe the full business plan for
            # both arms.  The runtime-profile fields bind the physical replay.
            "dataset_profile_path": str(profile["full_profile_path"]),
            "dataset_profile_sha256": str(profile["full_profile_sha256"]),
            "dataset_category": str(provisional["profile_role"]),
            "max_samples": int(
                profile["source_rows"] if full else profile["replay_rows"]
            ),
            "warmup_steps": 0 if full else REPLAY_WARMUP_STEPS,
            "measure_steps": full_steps if full else REPLAY_MEASURE_STEPS,
            "repeat": 0,
            "offload": False,
            "parallel_class": "gpu_partitionable",
            "requires_external_node_idle": False,
            "publication_allowed": False,
            "oom_role": "measurement_calibration_failure_not_model_evidence",
            "oom_is_expected_evidence": False,
            "source_dataset_id": str(provisional["source_dataset_id"]),
            "split_unit_id": str(provisional["source_dataset_id"]),
            "calibration_partition": {
                "role": "development_measurement",
                "policy": "final_memory_peak_replay_stage1_v1",
                "split_unit_id": str(provisional["source_dataset_id"]),
            },
            "full_plan_optimizer_steps": full_steps,
            "peak_batch_contract": dict(profile["peak_batch"]),
            "replay_sampler_verification": dict(profile["replay_sampler_verification"]),
        }
    )
    for key in (
        "packing_contract",
        "packing_dataprofile_path",
        "packing_dataprofile_sha256",
        "samples_per_pack",
    ):
        job.pop(key, None)
    return job


def _experiment(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    experiment = dict(
        read_json(
            ROOT
            / "unified_resource_staging"
            / "experiment.h800_unified_resource_evidence_v1.json"
        )
    )
    experiment["training_scope"] = {
        "phase_id": PHASE_ID,
        "model_ids": sorted({str(row["model_id"]) for row in jobs}),
        "gpu_ids": list(GPU_IDS),
        "exclusive_node_gpu_ids": list(GPU_IDS),
        "max_gpu_count": 2,
        "stage": "sft",
        "precision": "bf16",
        "gpu_type": "NVIDIA H800 140GB HBM3",
        "gpu_counts": sorted({int(row["gpu_count"]) for row in jobs}),
        "global_batch_sizes": [TARGET_GBS],
        "gradient_checkpointing": [False, True],
        "zero_by_gpu_count": {"1": ["none"], "2": ["zero2", "zero3"]},
        "objective": "calibrate exact peak-batch replay against full development-dataset coverage",
    }
    experiment["measurement"] = {
        **dict(experiment.get("measurement") or {}),
        "memory_probe_max_steps": 1,
        "throughput_warmup_steps": REPLAY_WARMUP_STEPS,
        "throughput_measure_steps": REPLAY_MEASURE_STEPS,
        "performance_parallelism": "disjoint_gpu_masks",
        "scheduler_order_policy": "parallel_queue",
        "rerun_on_unhealthy_result": False,
    }
    experiment["datasets"] = [
        {
            "id": str(row["dataset_id"]),
            "category": str(row["replay_mode"]),
            "target_cutoffs": [int(row["cutoff_len"])],
        }
        for row in jobs
    ]
    return experiment


def _materialize_profiles(selection: dict[str, Any]) -> list[dict[str, Any]]:
    development = {
        str(row["dataset_id"]): dict(row)
        for row in selection["development_measurement_sources"]
    }
    expected = {str(row["source_dataset_id"]) for row in WORKLOADS}
    if set(development) != expected:
        raise ValueError("development source partition drifted")
    encoders = {
        "qwen3_4b": TrainingEncoder(Path("/wanqing-models/Qwen3-4B")),
        "qwen3_8b": TrainingEncoder(Path("/wanqing-models/Qwen3-8B")),
    }
    profiles = []
    for workload in WORKLOADS:
        source_id = str(workload["source_dataset_id"])
        source_spec = development[source_id]
        source_path = Path(str(source_spec["local_path"]))
        if (
            not source_path.is_file()
            or sha256_file(source_path) != source_spec["file_sha256"]
        ):
            raise ValueError(
                f"frozen development source is missing or changed: {source_path}"
            )
        rows: list[dict[str, Any]] = []
        lengths: list[int] = []
        label_lengths: list[int] = []
        with source_path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                normalized = _normalize(
                    json.loads(line), path=source_path, line_number=line_number
                )
                encoded = {
                    model_id: encoder.encode(normalized)
                    for model_id, encoder in encoders.items()
                }
                if len(set(encoded.values())) != 1:
                    raise ValueError(
                        f"Qwen3 token lengths differ at {source_path}:{line_number}: {encoded}"
                    )
                total_tokens, label_tokens = next(iter(encoded.values()))
                rows.append(
                    {
                        "sample_id": f"{source_id}:source_row:{line_number}",
                        **normalized,
                    }
                )
                lengths.append(int(total_tokens))
                label_lengths.append(int(label_tokens))
        if len(rows) != int(source_spec["rows"]):
            raise ValueError(f"source row count changed: {source_id}")
        cutoff = int(workload["cutoff_len"])
        provisional_profile = {
            "full_dataset_id": f"peak1_{source_id.replace('-', '_')}__full",
            "source_rows": len(rows),
            "full_data_path": str(
                (OUTPUT_DATA_DIR / f"{source_id}__full.jsonl").resolve()
            ),
            "full_profile_path": str(
                (PROFILE_DIR / f"{source_id}__full.qwen3_nothink.jsonl").resolve()
            ),
            "aligned_effective_sequence": _aligned(min(max(lengths), cutoff)),
        }
        full_data_path = Path(provisional_profile["full_data_path"])
        full_profile_path = Path(provisional_profile["full_profile_path"])
        _atomic_jsonl(full_data_path, rows)
        _atomic_jsonl(
            full_profile_path,
            (
                {
                    "sample_id": row["sample_id"],
                    "total_tokens": total,
                    "label_tokens": label,
                    "turns": 3 if row["system"] else 2,
                    "assistant_turns": 1,
                }
                for row, total, label in zip(rows, lengths, label_lengths)
            ),
        )
        provisional_profile.update(
            {
                "full_data_sha256": sha256_file(full_data_path),
                "full_profile_sha256": sha256_file(full_profile_path),
            }
        )
        profiles.append(
            {
                **provisional_profile,
                "source_dataset_id": source_id,
                "source_path": str(source_path.resolve()),
                "source_sha256": sha256_file(source_path),
                "cutoff_len": cutoff,
                "profile_statistics": _statistics(lengths, cutoff),
                "_rows": rows,
                "_lengths": lengths,
                "_label_lengths": label_lengths,
            }
        )
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--data-bundle", type=Path, default=DEFAULT_DATA_BUNDLE)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    outputs = (
        args.data_bundle,
        args.queue,
        args.design,
        args.predictions,
        args.experiment,
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing frozen outputs: {existing}")

    selection = read_json(args.selection)
    if (
        selection.get("schema") != "sft_h800_final_memory_acceptance_s3_selection/v1"
        or selection.get("gpu_training_started") is not False
        or selection.get("gpu_outcomes_observed") != 0
    ):
        raise ValueError("final acceptance S3 partition is not outcome-blind")
    profiles = _materialize_profiles(selection)
    profile_by_source = {str(row["source_dataset_id"]): row for row in profiles}
    inventory = read_json(MODEL_INVENTORY)
    inventory_models = {str(row["id"]): dict(row) for row in inventory["models"]}
    feature_models = copy.deepcopy(inventory_models)
    for model_id, parameters in v3_data.RUNTIME_BASE_PARAMETERS.items():
        feature_models[model_id]["actual_parameters"] = parameters
    capacity = int(read_json(base.DEFAULT_HARDWARE)["memory_bytes_reported_by_torch"])
    artifact = load_artifact(V3_ARTIFACT)
    profile_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    selected_jobs: list[dict[str, Any]] = []
    selected_predictions: dict[str, dict[str, Any]] = {}
    mbs_audits: dict[str, list[dict[str, Any]]] = {}
    for workload in WORKLOADS:
        source_id = str(workload["source_dataset_id"])
        profile = profile_by_source[source_id]
        selected, prediction, audit = _select_mbs(
            dict(workload),
            model=inventory_models[str(workload["model_id"])],
            profile=profile,
            inventory=inventory,
            models=feature_models,
            capacity=capacity,
            artifact=artifact,
            profile_cache=profile_cache,
        )
        selected_jobs.append(selected)
        selected_predictions[str(selected["pair_id"])] = prediction
        mbs_audits[source_id] = audit

        mbs = int(selected["mbs"])
        peak = _select_peak_batch(
            profile["_lengths"],
            cutoff_len=int(workload["cutoff_len"]),
            mbs=mbs,
            world_size=int(workload["gpu_count"]),
        )
        replay_rows = (REPLAY_WARMUP_STEPS + REPLAY_MEASURE_STEPS) * TARGET_GBS
        replay_data = _inverse_permuted_replay_rows(
            profile["_rows"],
            [int(value) for value in peak["source_row_indices_zero_based"]],
            replay_rows=replay_rows,
        )
        replay_path = OUTPUT_DATA_DIR / f"{source_id}__peak_replay.jsonl"
        _atomic_jsonl(replay_path, replay_data)
        replay_lengths = []
        replay_label_lengths = []
        for row in replay_data:
            source_row = int(str(row["sample_id"]).rsplit(":", 1)[-1])
            source_index = source_row - 1
            replay_lengths.append(int(profile["_lengths"][source_index]))
            replay_label_lengths.append(int(profile["_label_lengths"][source_index]))
        replay_profile_path = (
            PROFILE_DIR / f"{source_id}__peak_replay.qwen3_nothink.jsonl"
        )
        _atomic_jsonl(
            replay_profile_path,
            (
                {
                    "sample_id": row["sample_id"],
                    "total_tokens": total,
                    "label_tokens": label,
                    "turns": 3 if row["system"] else 2,
                    "assistant_turns": 1,
                }
                for row, total, label in zip(
                    replay_data, replay_lengths, replay_label_lengths
                )
            ),
        )
        replay_verification = _verify_replay(
            replay_lengths,
            expected_lengths=[
                int(profile["_lengths"][index])
                for index in peak["source_row_indices_zero_based"]
            ],
            cutoff_len=int(workload["cutoff_len"]),
            mbs=mbs,
            world_size=int(workload["gpu_count"]),
        )
        profile.update(
            {
                "replay_dataset_id": f"peak1_{source_id.replace('-', '_')}__replay",
                "replay_rows": replay_rows,
                "replay_data_path": str(replay_path.resolve()),
                "replay_data_sha256": sha256_file(replay_path),
                "replay_profile_path": str(replay_profile_path.resolve()),
                "replay_profile_sha256": sha256_file(replay_profile_path),
                "replay_profile_statistics": _statistics(
                    replay_lengths, int(workload["cutoff_len"])
                ),
                "peak_batch": peak,
                "replay_sampler_verification": replay_verification,
                "full_plan_optimizer_steps": math.ceil(
                    int(profile["source_rows"]) / TARGET_GBS
                ),
                "selected_mbs": mbs,
            }
        )

    registry = read_json(DATASET_INFO)
    for profile in profiles:
        entries = {
            str(profile["full_dataset_id"]): {
                "file_name": str(
                    Path(str(profile["full_data_path"]))
                    .resolve()
                    .relative_to(DATA_DIR.resolve())
                ),
                "columns": {
                    "prompt": "prompt",
                    "response": "response",
                    "system": "system",
                },
            },
            str(profile["replay_dataset_id"]): {
                "file_name": str(
                    Path(str(profile["replay_data_path"]))
                    .resolve()
                    .relative_to(DATA_DIR.resolve())
                ),
                "columns": {
                    "prompt": "prompt",
                    "response": "response",
                    "system": "system",
                },
            },
        }
        for dataset_id, expected_entry in entries.items():
            existing_entry = registry.get(dataset_id)
            if existing_entry is not None and existing_entry != expected_entry:
                raise ValueError(f"dataset registry collision: {dataset_id}")
            registry[dataset_id] = expected_entry
    write_json(DATASET_INFO, registry)

    template = _template()
    jobs = []
    for selected in selected_jobs:
        profile = profile_by_source[str(selected["source_dataset_id"])]
        jobs.extend(
            _job(selected, mode=mode, profile=profile, template=template)
            for mode in ("full_coverage", "peak_replay")
        )
    if (
        len(jobs) != EXPECTED_JOBS
        or len({row["job_id"] for row in jobs}) != EXPECTED_JOBS
    ):
        raise RuntimeError("stage-one queue size or uniqueness drifted")
    if any((ROOT / "results" / str(row["job_id"])).exists() for row in jobs):
        raise RuntimeError("one or more stage-one result directories already exist")

    frozen_rows = []
    for job in jobs:
        pair_id = str(job["pair_id"])
        frozen_rows.append(
            {
                "job_id": str(job["job_id"]),
                "pair_id": pair_id,
                "replay_mode": str(job["replay_mode"]),
                "source_dataset_id": str(job["source_dataset_id"]),
                "v3": selected_predictions[pair_id],
            }
        )
    frozen: dict[str, Any] = {
        "schema": PREDICTION_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_any_stage1_gpu_outcome",
        "outcomes_observed": 0,
        "model_or_margin_refit_allowed_during_measurement_equivalence": False,
        "production_model_mutated": False,
        "v3_artifact": {
            "path": str(V3_ARTIFACT.resolve()),
            "sha256": sha256_file(V3_ARTIFACT),
        },
        "ordered_job_payload_sha256": sha256_json(jobs),
        "rows": frozen_rows,
    }
    frozen["report_sha256"] = sha256_json(frozen)

    public_profiles = [
        {key: value for key, value in profile.items() if not key.startswith("_")}
        for profile in profiles
    ]
    data_bundle: dict[str, Any] = {
        "schema": DATA_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "gpu_training_started": False,
        "gpu_outcomes_observed": 0,
        "final_acceptance_sources_read": 0,
        "sampler_contract": {
            "implementation": "accelerate.SeedableRandomSampler+BatchSamplerShard",
            "data_seed": DATA_SEED,
            "split_batches": False,
            "even_batches": True,
            "padding_multiple": 8,
            "replay_construction": "inverse permutation yields the exact selected full-plan peak batch on every replay microbatch",
        },
        "profiles": public_profiles,
        "selection_binding": {
            "path": str(args.selection.resolve()),
            "sha256": sha256_file(args.selection),
        },
        "dataset_registry": {
            "path": str(DATASET_INFO.resolve()),
            "sha256": sha256_file(DATASET_INFO),
        },
    }
    data_bundle["report_sha256"] = sha256_json(data_bundle)

    args.data_bundle.parent.mkdir(parents=True, exist_ok=True)
    args.experiment.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.data_bundle, data_bundle)
    write_jsonl(args.queue, jobs)
    write_json(args.predictions, frozen)
    write_json(args.experiment, _experiment(jobs))
    design: dict[str, Any] = {
        "schema": DESIGN_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "status": "prepared_waiting_for_gpu_0_3_preflight",
        "gpu_training_started": False,
        "gpu_outcomes_observed": 0,
        "execution_authorized": True,
        "publication_allowed": False,
        "objective": "Validate that exact peak-batch replay reproduces the max CUDA reserved memory of full development-dataset coverage.",
        "evidence_contract": {
            "development_sources_only": True,
            "final_acceptance_sources_read": 0,
            "outcomes_may_only_calibrate_measurement_or_later_development_modeling": True,
            "not_final_model_acceptance_evidence": True,
            "full_arm_covers_at_least_one_complete_sampler_epoch": True,
            "replay_arm_uses_exact_selected_real_batch": True,
        },
        "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
        "authorized_gpu_ids": list(GPU_IDS),
        "old_missing_34_jobs_included": False,
        "pairs": EXPECTED_PAIRS,
        "jobs": EXPECTED_JOBS,
        "ordered_job_ids": [str(row["job_id"]) for row in jobs],
        "ordered_job_payload_sha256": sha256_json(jobs),
        "mbs_selection_policy": {
            "source": "frozen V3 admission upper bound only",
            "maximum_upper_to_safe_limit": MAX_SAFE_PRESSURE,
            "choose": "largest eligible legal MBS",
            "audits": mbs_audits,
        },
        "bindings": {
            "queue": {
                "path": str(args.queue.resolve()),
                "sha256": sha256_file(args.queue),
            },
            "experiment": {
                "path": str(args.experiment.resolve()),
                "sha256": sha256_file(args.experiment),
            },
            "frozen_predictions": {
                "path": str(args.predictions.resolve()),
                "sha256": sha256_file(args.predictions),
            },
            "data_bundle": {
                "path": str(args.data_bundle.resolve()),
                "sha256": sha256_file(args.data_bundle),
            },
            "selection": {
                "path": str(args.selection.resolve()),
                "sha256": sha256_file(args.selection),
            },
            "v3_artifact": {
                "path": str(V3_ARTIFACT.resolve()),
                "sha256": sha256_file(V3_ARTIFACT),
            },
            "dataset_registry": {
                "path": str(DATASET_INFO.resolve()),
                "sha256": sha256_file(DATASET_INFO),
            },
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(args.design, design)
    print(
        json.dumps(
            {
                "data_bundle": str(args.data_bundle),
                "queue": str(args.queue),
                "design": str(args.design),
                "predictions": str(args.predictions),
                "experiment": str(args.experiment),
                "jobs": len(jobs),
                "selected": [
                    {
                        "source_dataset_id": profile["source_dataset_id"],
                        "model_id": selected["model_id"],
                        "train_type": selected["train_type"],
                        "gpu_count": selected["gpu_count"],
                        "zero_stage": selected["zero_stage"],
                        "gc": selected["gc"],
                        "cutoff_len": selected["cutoff_len"],
                        "mbs": profile["selected_mbs"],
                        "full_plan_optimizer_steps": profile[
                            "full_plan_optimizer_steps"
                        ],
                        "peak_padded_sequence_length": profile["peak_batch"][
                            "padded_sequence_length"
                        ],
                    }
                    for selected, profile in zip(selected_jobs, public_profiles)
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
