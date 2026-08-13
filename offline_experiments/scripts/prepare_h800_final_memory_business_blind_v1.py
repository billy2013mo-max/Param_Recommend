#!/usr/bin/env python3
"""Freeze the 60-job six-dataset business-blind memory acceptance matrix.

The unified V3 model and all admission decisions are evaluated before any GPU
outcome from the six final sources is read.  Non-packing jobs use the validated
exact allocator-prefix replay.  The two packing ladders use full packed-dataset
coverage because allocator-prefix equivalence has not been established for the
packing preprocessor.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import fit_h800_unified_resource_partial_v1 as base
import h800_unified_bounded_memory_v3_data as v3_data
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
from h800_unified_bounded_memory_model import load_artifact, predict_records
from prepare_h800_bounded_memory_v2_fresh_data_v1 import TrainingEncoder
from prepare_h800_final_memory_peak_replay_stage1_v1 import _aligned, _sampler_batches
from prepare_packing_dataprofile_v2 import _curve

CAMPAIGN_ID = "h800_final_memory_business_blind_20260810_v1"
PHASE_ID = "h800_final_memory_business_blind_v1"
JOB_SCHEMA = "sft_h800_final_memory_business_blind_job/v1"
DESIGN_SCHEMA = "sft_h800_final_memory_business_blind_design/v1"
DATA_SCHEMA = "sft_h800_final_memory_business_blind_data/v1"
PREDICTION_SCHEMA = "sft_h800_final_memory_business_blind_predictions/v1"
GPU_IDS = (0, 1, 2, 3)
DATA_SEED = 20260810
TARGET_PRESSURES = (0.80, 0.93, 0.99, 1.01, 1.08)
EXPECTED_SCENARIOS = 12
EXPECTED_JOBS = EXPECTED_SCENARIOS * len(TARGET_PRESSURES)
BASE_TARGET_GBS = 32
FREEZE_STATUS = "frozen_before_any_final_gpu_outcome"
PREDICTION_ROLE = "final_business_blind_frozen_before_outcomes"
PRIOR_CAMPAIGN_GPU_OUTCOMES_OBSERVED_AT_FREEZE = 0

SELECTION = ARTIFACT_DIR / "h800_final_memory_acceptance_s3_selection_v1.json"
MEASUREMENT_GATE = ARTIFACT_DIR / "h800_final_memory_allocator_prefix_replay_results_v2.json"
V3_ARTIFACT = ARTIFACT_DIR / "h800_unified_bounded_memory_candidate_v3.json"
MODEL_INVENTORY = ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_v1.json"
QWEN35_RUNTIME_CONTRACT = ARTIFACT_DIR / "h800_qwen35_runtime_contract_v1.json"
DATASET_INFO = DATA_DIR / "dataset_info.json"
TEMPLATE_QUEUE = MATRIX_DIR / "h800_final_memory_peak_replay_jobs_v1.jsonl"
OUTPUT_DATA_DIR = DATA_DIR / "final_memory_business_blind_v1"
PROFILE_DIR = ARTIFACT_DIR / "final_memory_business_blind_v1" / "profiles"
DEFAULT_QUEUE = MATRIX_DIR / "h800_final_memory_business_blind_jobs_v1.jsonl"
DEFAULT_DATA_BUNDLE = ARTIFACT_DIR / "h800_final_memory_business_blind_data_v1.json"
DEFAULT_PREDICTIONS = ARTIFACT_DIR / "h800_final_memory_business_blind_frozen_predictions_v1.json"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_final_memory_business_blind_design_v1.json"
STAGING_DIR = ROOT / "final_memory_business_blind_v1_staging"
DEFAULT_EXPERIMENT = STAGING_DIR / "experiment.h800_final_memory_business_blind_v1.json"

SCENARIOS: tuple[dict[str, Any], ...] = (
    {"source_dataset_id": "dataset-quuxpd-1786009603", "scenario": "short_full", "model_id": "qwen3_1p7b", "train_type": "full", "gpu_count": 1, "zero_stage": 0, "gc": False, "packing": False},
    {"source_dataset_id": "dataset-quuxpd-1786009603", "scenario": "short_qwen35_lora", "model_id": "qwen3p5_4b", "train_type": "lora", "gpu_count": 1, "zero_stage": 0, "gc": False, "packing": False},
    {"source_dataset_id": "dataset-yviykz-1786010889", "scenario": "short_tail_full", "model_id": "qwen3_4b", "train_type": "full", "gpu_count": 1, "zero_stage": 0, "gc": True, "packing": False},
    {"source_dataset_id": "dataset-yviykz-1786010889", "scenario": "short_tail_lora", "model_id": "qwen3_4b", "train_type": "lora", "gpu_count": 1, "zero_stage": 0, "gc": False, "packing": False},
    {"source_dataset_id": "dataset-2udchx-1786025298", "scenario": "medium_full", "model_id": "qwen3_8b", "train_type": "full", "gpu_count": 2, "zero_stage": 2, "gc": True, "packing": False},
    {"source_dataset_id": "dataset-2udchx-1786025298", "scenario": "medium_lora", "model_id": "qwen3_8b", "train_type": "lora", "gpu_count": 2, "zero_stage": 2, "gc": False, "packing": False},
    {"source_dataset_id": "dataset-u119sh-1786072229", "scenario": "broad_tail_full", "model_id": "qwen3_14b", "train_type": "full", "gpu_count": 2, "zero_stage": 3, "gc": True, "packing": False},
    {"source_dataset_id": "dataset-u119sh-1786072229", "scenario": "broad_tail_lora", "model_id": "qwen3_14b", "train_type": "lora", "gpu_count": 2, "zero_stage": 3, "gc": False, "packing": False},
    {"source_dataset_id": "dataset-x52l60-1785911619", "scenario": "upper_tail_full", "model_id": "qwen3_32b", "train_type": "full", "gpu_count": 4, "zero_stage": 3, "gc": True, "packing": False},
    {"source_dataset_id": "dataset-x52l60-1785911619", "scenario": "upper_tail_lora", "model_id": "qwen3_32b", "train_type": "lora", "gpu_count": 4, "zero_stage": 3, "gc": True, "packing": False},
    {"source_dataset_id": "dataset-gy6hdc-1786278399", "scenario": "long_packed_full", "model_id": "qwen3_4b", "train_type": "full", "gpu_count": 1, "zero_stage": 0, "gc": False, "packing": True},
    {"source_dataset_id": "dataset-gy6hdc-1786278399", "scenario": "long_packed_qwen35_lora", "model_id": "qwen3p5_4b", "train_type": "lora", "gpu_count": 2, "zero_stage": 2, "gc": False, "packing": True},
)


def _qwen35_environment_overlay() -> dict[str, Any]:
    contract = read_json(QWEN35_RUNTIME_CONTRACT)
    environment = dict(contract.get("environment") or {})
    prefixes = [str(value) for value in environment.get("PYTHONPATH_prepend") or []]
    variables = {str(key): str(value) for key, value in (environment.get("variables") or {}).items()}
    if len(prefixes) != 2 or not variables:
        raise ValueError("Qwen3.5 runtime contract lacks the frozen environment overlay")
    return {
        "PYTHONPATH_prepend": prefixes,
        "variables": variables,
        "contract_path": str(QWEN35_RUNTIME_CONTRACT.resolve()),
        "contract_sha256": sha256_file(QWEN35_RUNTIME_CONTRACT),
    }


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
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


def _normalize(value: Any, *, path: Path, line_number: int) -> dict[str, str]:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict):
        raise TypeError(f"{path}:{line_number} is not an SFT object")
    required = ("system", "prompt", "response")
    if not all(isinstance(value.get(field), str) for field in required):
        raise ValueError(f"{path}:{line_number} lacks string system/prompt/response")
    return {field: str(value[field]) for field in required}


def _template() -> dict[str, Any]:
    return dict(read_jsonl(TEMPLATE_QUEUE)[0])


def _cutoff_candidates(maximum: int) -> list[int]:
    grid = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
    values = [value for value in grid if value < maximum]
    values.append(max(128, _aligned(maximum)))
    return sorted(set(values))


def _target_gbs(gpu_count: int, mbs: int) -> int:
    denominator = gpu_count * mbs
    return max(BASE_TARGET_GBS, denominator)


def _profile_job(
    scenario: dict[str, Any], *, mbs: int, cutoff: int,
    profile: dict[str, Any], model: dict[str, Any], packing_curve: dict[str, Any] | None,
) -> dict[str, Any]:
    gpu_count = int(scenario["gpu_count"])
    target_gbs = _target_gbs(gpu_count, mbs)
    ga = target_gbs // (gpu_count * mbs)
    zero_stage = int(scenario["zero_stage"])
    packing = bool(scenario["packing"])
    job = {
        **scenario,
        "model_family": str(model["family"]),
        "model_path": str(model["path"]),
        "tokenizer_path": str(model["path"]),
        "model_parameters": int(model["actual_parameters"]),
        "cutoff_len": cutoff,
        "aligned_effective_sequence": _aligned(min(int(profile["maximum_tokens"]), cutoff)),
        "mbs": mbs,
        "target_gbs": target_gbs,
        "gradient_accumulation_steps": ga,
        "gradient_checkpointing": bool(scenario["gc"]),
        "zero": "none" if zero_stage == 0 else f"zero{zero_stage}",
        "dataset_profile_path": str(profile["profile_path"]),
        "dataset_profile_sha256": str(profile["profile_sha256"]),
        "mechanism_id": (
            f"{scenario['train_type']}_zero{zero_stage}_gc{int(bool(scenario['gc']))}_"
            f"{gpu_count}gpu_pack{int(packing)}"
        ),
    }
    if packing:
        if packing_curve is None:
            raise ValueError("packing curve is required")
        job["packing_contract"] = {
            "expected_samples_per_pack": float(packing_curve["samples_per_pack"]["mean"]),
            "expected_packs": int(packing_curve["packs"]),
            "curve_policy": "exact frozen tokenizer profile and production worker-sharded greedy packer",
        }
    return job


def _prediction_record(
    job: dict[str, Any], *, inventory: dict[str, Any], models: dict[str, dict[str, Any]],
    capacity: int, profile_cache: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    reference, features = base._current_features(
        job, model_by_id=models, fixed_lora=inventory["fixed_lora"],
        capacity_bytes=capacity, profile_cache=profile_cache,
    )
    return {
        "record_id": stable_id("blindpred", {
            "scenario": job["scenario"], "mbs": job["mbs"], "cutoff": job["cutoff_len"]
        }),
        "source_id": str(job["source_dataset_id"]),
        "origin": CAMPAIGN_ID,
        "role": PREDICTION_ROLE,
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
        "packing": bool(job["packing"]),
        "profile_sha256": str(job["dataset_profile_sha256"]),
    }


def _select_ladder(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda row: (
        row["pressure"], int(row["job"]["cutoff_len"]), int(row["job"]["mbs"])
    ))
    count = len(TARGET_PRESSURES)
    if len(ordered) < count:
        raise ValueError("not enough unique pressure candidates")
    costs = [[math.inf] * len(ordered) for _ in range(count)]
    parents = [[-1] * len(ordered) for _ in range(count)]
    for index, row in enumerate(ordered):
        runtime_penalty = 0.001 * math.sqrt(float(row["planned_gpu_steps"]))
        costs[0][index] = (
            abs(float(row["pressure"]) - TARGET_PRESSURES[0]) + runtime_penalty
        )
    for target_index in range(1, count):
        best_cost = math.inf
        best_index = -1
        for index in range(len(ordered)):
            previous = index - 1
            if previous >= 0 and costs[target_index - 1][previous] < best_cost:
                best_cost = costs[target_index - 1][previous]
                best_index = previous
            if best_index >= 0:
                runtime_penalty = 0.001 * math.sqrt(
                    float(ordered[index]["planned_gpu_steps"])
                )
                costs[target_index][index] = best_cost + abs(
                    float(ordered[index]["pressure"]) - TARGET_PRESSURES[target_index]
                ) + runtime_penalty
                parents[target_index][index] = best_index
    end = min(range(len(ordered)), key=lambda index: costs[-1][index])
    indexes = [end]
    for target_index in range(count - 1, 0, -1):
        end = parents[target_index][end]
        if end < 0:
            raise RuntimeError("pressure ladder assignment failed")
        indexes.append(end)
    indexes.reverse()
    selected = []
    for target, index in zip(TARGET_PRESSURES, indexes):
        row = dict(ordered[index])
        row["target_pressure"] = target
        row["absolute_pressure_distance"] = abs(float(row["pressure"]) - target)
        selected.append(row)
    return selected


def _prefix_contract(
    lengths: list[int], *, cutoff: int, mbs: int, world_size: int,
    ga: int, target_gbs: int,
) -> tuple[dict[str, Any], list[int]]:
    by_rank = _sampler_batches(
        len(lengths), mbs=mbs, world_size=world_size, data_seed=DATA_SEED
    )
    rank_contracts = []
    for rank, batches in enumerate(by_rank):
        padded = [_aligned(max(min(lengths[index], cutoff) for index in batch)) for batch in batches]
        rank_max = max(padded)
        first_microbatch = padded.index(rank_max)
        rank_contracts.append({
            "rank": rank,
            "rank_max_padded_sequence_length": rank_max,
            "first_rank_max_microbatch_index_zero_based": first_microbatch,
            "first_rank_max_optimizer_step_one_based": first_microbatch // ga + 1,
        })
    steps = max(int(row["first_rank_max_optimizer_step_one_based"]) for row in rank_contracts)
    local_microbatches = steps * ga
    desired: list[int] = []
    for global_batch_index in range(local_microbatches * world_size):
        rank = global_batch_index % world_size
        local_index = global_batch_index // world_size
        desired.extend(by_rank[rank][local_index])
    if len(desired) != steps * target_gbs:
        raise RuntimeError("prefix rows do not match optimizer-step exposure")
    contract = {
        "selection_rule": "exact original sampler prefix until every rank first observes its full-plan maximum aligned padded length",
        "data_seed": DATA_SEED,
        "prefix_optimizer_steps": steps,
        "prefix_microbatches_per_rank": local_microbatches,
        "prefix_rows": len(desired),
        "full_plan_optimizer_steps": math.ceil(len(lengths) / target_gbs),
        "rank_contracts": rank_contracts,
    }
    return contract, desired


def _materialize_prefix(
    source_rows: list[dict[str, Any]], source_lengths: list[int], source_labels: list[int],
    desired: list[int], *, stem: str, mbs: int, world_size: int,
    expected_microbatches: int,
) -> tuple[Path, Path, dict[str, Any]]:
    replay_rows = len(desired)
    permutation = torch.randperm(
        replay_rows, generator=torch.Generator().manual_seed(DATA_SEED)
    ).tolist()
    output: list[dict[str, Any] | None] = [None] * replay_rows
    output_lengths = [0] * replay_rows
    output_labels = [0] * replay_rows
    for emitted, storage in enumerate(permutation):
        source_index = int(desired[emitted])
        output[storage] = {
            "sample_id": f"blind_prefix_position:{emitted}:source_row:{source_index + 1}",
            **{key: source_rows[source_index][key] for key in ("system", "prompt", "response")},
        }
        output_lengths[storage] = int(source_lengths[source_index])
        output_labels[storage] = int(source_labels[source_index])
    rows = [dict(row) for row in output if row is not None]
    if len(rows) != replay_rows:
        raise RuntimeError("prefix inverse permutation is incomplete")
    actual_batches = _sampler_batches(
        replay_rows, mbs=mbs, world_size=world_size, data_seed=DATA_SEED
    )
    replay_source = [int(str(row["sample_id"]).rsplit(":", 1)[1]) - 1 for row in rows]
    emitted_actual = []
    for global_batch_index in range(expected_microbatches * world_size):
        rank = global_batch_index % world_size
        local_index = global_batch_index // world_size
        emitted_actual.extend(replay_source[index] for index in actual_batches[rank][local_index])
    if emitted_actual != desired:
        raise RuntimeError("business-blind prefix sampler verification failed")
    data_path = OUTPUT_DATA_DIR / "prefixes" / f"{stem}.jsonl"
    profile_path = PROFILE_DIR / "prefixes" / f"{stem}.qwen3_nothink.jsonl"
    _atomic_jsonl(data_path, rows)
    _atomic_jsonl(profile_path, [
        {
            "sample_id": row["sample_id"], "total_tokens": total,
            "label_tokens": label, "turns": 3 if row["system"] else 2,
            "assistant_turns": 1,
        }
        for row, total, label in zip(rows, output_lengths, output_labels)
    ])
    return data_path, profile_path, {
        "all_rank_prefixes_exact": True,
        "desired_source_indices_sha256": sha256_json(desired),
        "emitted_source_indices_sha256": sha256_json(emitted_actual),
    }


def _experiment(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    experiment = read_json(
        ROOT / "final_memory_allocator_prefix_replay_v2_staging"
        / "experiment.h800_final_memory_allocator_prefix_replay_v2.json"
    )
    experiment = copy.deepcopy(experiment)
    experiment["training_scope"] = {
        "phase_id": PHASE_ID,
        "model_ids": sorted({str(row["model_id"]) for row in jobs}),
        "gpu_ids": list(GPU_IDS),
        "exclusive_node_gpu_ids": list(GPU_IDS),
        "max_gpu_count": 4,
        "stage": "sft",
        "precision": "bf16",
        "gpu_type": "NVIDIA H800 140GB HBM3",
        "gpu_counts": [1, 2, 4],
        "global_batch_sizes": sorted({int(row["target_gbs"]) for row in jobs}),
        "gradient_checkpointing": [False, True],
        "zero_by_gpu_count": {"1": ["none"], "2": ["none", "zero2", "zero3"], "4": ["zero3"]},
        "objective": "frozen V3 six-dataset final business-blind memory acceptance",
    }
    experiment["measurement"] = {
        **dict(experiment.get("measurement") or {}),
        "throughput_warmup_steps": 0,
        "throughput_measure_steps": max(int(row["measure_steps"]) for row in jobs),
        "performance_parallelism": "disjoint_gpu_masks",
        "scheduler_order_policy": "parallel_queue",
        "rerun_on_unhealthy_result": False,
    }
    experiment["datasets"] = [
        {"id": str(row["dataset_id"]), "category": str(row["dataset_category"]), "target_cutoffs": [int(row["cutoff_len"])]}
        for row in jobs
    ]
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--data-bundle", type=Path, default=DEFAULT_DATA_BUNDLE)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    outputs = (args.queue, args.data_bundle, args.predictions, args.design, args.experiment)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite frozen outputs: {existing}")

    gate = read_json(MEASUREMENT_GATE)
    if gate.get("status") != "measurement_equivalence_repaired" or gate.get("all_measurement_equivalence_checks_passed") is not True:
        raise ValueError("allocator-prefix measurement gate has not passed")
    selection = read_json(SELECTION)
    if selection.get("gpu_outcomes_observed") != 0 or selection.get("gpu_training_started") is not False:
        raise ValueError("final data partition is no longer outcome blind")
    all_selected_sources = {str(row["dataset_id"]): dict(row) for row in selection["final_acceptance_sources"]}
    expected_sources = {str(row["source_dataset_id"]) for row in SCENARIOS}
    if not expected_sources.issubset(all_selected_sources):
        raise ValueError("one or more requested final source IDs drifted")
    selected_sources = {source_id: all_selected_sources[source_id] for source_id in expected_sources}

    inventory = read_json(MODEL_INVENTORY)
    inventory_models = {str(row["id"]): dict(row) for row in inventory["models"]}
    feature_models = copy.deepcopy(inventory_models)
    for model_id, parameters in v3_data.RUNTIME_BASE_PARAMETERS.items():
        feature_models[model_id]["actual_parameters"] = parameters
    encoders = {
        "qwen3": TrainingEncoder(Path("/wanqing-models/Qwen3-8B"), "qwen3_nothink"),
        "qwen3_5": TrainingEncoder(Path("/wanqing-models/Qwen3.5-4B"), "qwen3_5_nothink"),
    }
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    profile_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    registry = read_json(DATASET_INFO)
    for source_id, source in selected_sources.items():
        source_path = Path(str(source["local_path"]))
        if not source_path.is_file() or sha256_file(source_path) != source["file_sha256"]:
            raise ValueError(f"final source missing or changed: {source_id}")
        normalized = []
        with source_path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if line.strip():
                    normalized.append({
                        "sample_id": f"{source_id}:source_row:{line_number}",
                        **_normalize(json.loads(line), path=source_path, line_number=line_number),
                    })
        if len(normalized) != int(source["rows"]):
            raise ValueError(f"final source row count drifted: {source_id}")
        rows_by_source[source_id] = normalized
        full_dataset_id = f"blind1_{source_id.replace('-', '_')}__full"
        full_path = OUTPUT_DATA_DIR / "full" / f"{source_id}.jsonl"
        _atomic_jsonl(full_path, normalized)
        registry[full_dataset_id] = {
            "file_name": str(full_path.resolve().relative_to(DATA_DIR.resolve())),
            "columns": {"prompt": "prompt", "response": "response", "system": "system"},
        }
        needed_families = {
            str(inventory_models[str(scenario["model_id"])]["family"])
            for scenario in SCENARIOS if scenario["source_dataset_id"] == source_id
        }
        for family in needed_families:
            encoder = encoders[family]
            lengths = []
            labels = []
            for row in normalized:
                total, label = encoder.encode({key: row[key] for key in ("system", "prompt", "response")})
                lengths.append(int(total)); labels.append(int(label))
            profile_path = PROFILE_DIR / "full" / f"{source_id}.{family}.jsonl"
            _atomic_jsonl(profile_path, [
                {"sample_id": row["sample_id"], "total_tokens": total, "label_tokens": label, "turns": 3 if row["system"] else 2, "assistant_turns": 1}
                for row, total, label in zip(normalized, lengths, labels)
            ])
            profile_by_key[(source_id, family)] = {
                "full_dataset_id": full_dataset_id,
                "full_data_path": str(full_path.resolve()),
                "full_data_sha256": sha256_file(full_path),
                "profile_path": str(profile_path.resolve()),
                "profile_sha256": sha256_file(profile_path),
                "lengths": lengths,
                "labels": labels,
                "maximum_tokens": max(lengths),
            }

    artifact = load_artifact(V3_ARTIFACT)
    capacity = int(artifact["hardware_domain"]["capacity_bytes"])
    profile_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    scenario_ladders = []
    selected_rows = []
    for scenario in SCENARIOS:
        source_id = str(scenario["source_dataset_id"])
        inventory_model = inventory_models[str(scenario["model_id"])]
        model = feature_models[str(scenario["model_id"])]
        family = str(inventory_model["family"])
        profile = profile_by_key[(source_id, family)]
        cutoffs = _cutoff_candidates(int(profile["maximum_tokens"]))
        mbs_values = [1] if scenario["packing"] else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        candidates = []
        for cutoff in cutoffs:
            curve = _curve(profile["lengths"], [cutoff])[0] if scenario["packing"] else None
            for mbs in mbs_values:
                if int(scenario["gpu_count"]) * mbs > len(rows_by_source[source_id]):
                    continue
                job = _profile_job(scenario, mbs=mbs, cutoff=cutoff, profile=profile, model=model, packing_curve=curve)
                record = _prediction_record(
                    job, inventory=inventory, models=feature_models,
                    capacity=capacity, profile_cache=profile_cache,
                )
                prediction = predict_records([record], artifact)[0]
                if scenario["packing"]:
                    assert curve is not None
                    physical_gbs = (
                        int(job["gpu_count"])
                        * int(job["mbs"])
                        * int(job["gradient_accumulation_steps"])
                    )
                    planned_steps = math.ceil(int(curve["packs"]) / physical_gbs)
                else:
                    prefix_contract, _ = _prefix_contract(
                        profile["lengths"],
                        cutoff=int(job["cutoff_len"]),
                        mbs=int(job["mbs"]),
                        world_size=int(job["gpu_count"]),
                        ga=int(job["gradient_accumulation_steps"]),
                        target_gbs=int(job["target_gbs"]),
                    )
                    planned_steps = int(prefix_contract["prefix_optimizer_steps"])
                candidates.append({
                    "job": job, "record": record, "prediction": prediction,
                    "pressure": float(prediction["admission_upper_bytes"]) / float(prediction["safe_limit_bytes"]),
                    "planned_optimizer_steps": planned_steps,
                    "planned_gpu_steps": planned_steps * int(job["gpu_count"]),
                })
        ladder = _select_ladder(candidates)
        scenario_ladders.append({
            "scenario": scenario["scenario"], "source_dataset_id": source_id,
            "candidate_count": len(candidates),
            "selected": [
                {"target_pressure": row["target_pressure"], "actual_pressure": row["pressure"], "absolute_pressure_distance": row["absolute_pressure_distance"], "mbs": row["job"]["mbs"], "cutoff_len": row["job"]["cutoff_len"], "admitted": row["prediction"]["admitted"], "planned_optimizer_steps": row["planned_optimizer_steps"], "planned_gpu_steps": row["planned_gpu_steps"]}
                for row in ladder
            ],
        })
        selected_rows.extend(ladder)

    template = _template()
    jobs = []
    frozen_predictions = []
    prefix_bindings = []
    for selected in selected_rows:
        logical = dict(selected["job"])
        source_id = str(logical["source_dataset_id"])
        model = inventory_models[str(logical["model_id"])]
        family = str(model["family"])
        profile = profile_by_key[(source_id, family)]
        packing = bool(logical["packing"])
        target_pressure = float(selected["target_pressure"])
        identity = {
            "scenario": logical["scenario"], "target_pressure": target_pressure,
            "mbs": logical["mbs"], "cutoff_len": logical["cutoff_len"],
        }
        job_id = stable_id("h800blind1", identity)
        if packing:
            curve = _curve(profile["lengths"], [int(logical["cutoff_len"])])[0]
            physical_gbs = int(logical["gpu_count"]) * int(logical["mbs"]) * int(logical["gradient_accumulation_steps"])
            steps = math.ceil(int(curve["packs"]) / physical_gbs)
            runtime_dataset_id = str(profile["full_dataset_id"])
            runtime_data_path = Path(str(profile["full_data_path"]))
            runtime_profile_path = Path(str(profile["profile_path"]))
            measurement_mode = "full_packed_dataset_coverage"
            measurement_contract = {
                "packs": int(curve["packs"]),
                "physical_packs_per_optimizer_step": physical_gbs,
                "full_plan_optimizer_steps": steps,
                "reason": "packing allocator-prefix equivalence is not established",
            }
            max_samples = len(rows_by_source[source_id])
        else:
            contract, desired = _prefix_contract(
                profile["lengths"], cutoff=int(logical["cutoff_len"]),
                mbs=int(logical["mbs"]), world_size=int(logical["gpu_count"]),
                ga=int(logical["gradient_accumulation_steps"]), target_gbs=int(logical["target_gbs"]),
            )
            stem = f"{job_id}__{source_id}"
            runtime_data_path, runtime_profile_path, verification = _materialize_prefix(
                rows_by_source[source_id], profile["lengths"], profile["labels"], desired,
                stem=stem, mbs=int(logical["mbs"]), world_size=int(logical["gpu_count"]),
                expected_microbatches=int(contract["prefix_microbatches_per_rank"]),
            )
            runtime_dataset_id = f"blind1_{job_id.replace('-', '_')}__prefix"
            registry[runtime_dataset_id] = {
                "file_name": str(runtime_data_path.resolve().relative_to(DATA_DIR.resolve())),
                "columns": {"prompt": "prompt", "response": "response", "system": "system"},
            }
            steps = int(contract["prefix_optimizer_steps"])
            max_samples = int(contract["prefix_rows"])
            measurement_mode = "validated_allocator_prefix_replay"
            measurement_contract = {**contract, "verification": verification}
            prefix_bindings.append({
                "job_id": job_id,
                "data_path": str(runtime_data_path.resolve()), "data_sha256": sha256_file(runtime_data_path),
                "profile_path": str(runtime_profile_path.resolve()), "profile_sha256": sha256_file(runtime_profile_path),
            })
        job = copy.deepcopy(template)
        job.update(logical)
        job.update({
            "schema": JOB_SCHEMA,
            "job_id": job_id,
            "campaign_id": CAMPAIGN_ID,
            "phase_id": PHASE_ID,
            "kind": "throughput",
            "experiment_group": "FINAL_MEMORY_BUSINESS_BLIND_V1",
            "evidence_role": "final_business_blind_acceptance_only",
            "purpose": "frozen_v3_three_metric_final_acceptance",
            "dataset_id": runtime_dataset_id,
            "runtime_dataset_id": runtime_dataset_id,
            "data_path": str(runtime_data_path.resolve()),
            "data_sha256": sha256_file(runtime_data_path),
            "runtime_dataset_profile_path": str(runtime_profile_path.resolve()),
            "runtime_dataset_profile_sha256": sha256_file(runtime_profile_path),
            "dataset_profile_path": str(profile["profile_path"]),
            "dataset_profile_sha256": str(profile["profile_sha256"]),
            "dataset_category": str(selected_sources[source_id]["profile_role"]),
            "max_samples": max_samples,
            "warmup_steps": 0,
            "measure_steps": steps,
            "repeat": 0,
            "offload": False,
            "parallel_class": "exclusive" if int(logical["gpu_count"]) == 4 else "gpu_partitionable",
            "requires_external_node_idle": False,
            "publication_allowed": False,
            "oom_role": "valid_final_acceptance_right_censored_outcome",
            "oom_is_expected_evidence": True,
            "split_unit_id": source_id,
            "calibration_partition": {
                "role": "final_acceptance_blind", "policy": "six_source_content_disjoint_v1", "split_unit_id": source_id,
            },
            "target_pressure": target_pressure,
            "actual_frozen_pressure": float(selected["pressure"]),
            "pressure_distance": float(selected["absolute_pressure_distance"]),
            "measurement_mode": measurement_mode,
            "measurement_contract": measurement_contract,
            "seed": DATA_SEED,
            "data_seed": DATA_SEED,
            "template": "qwen3_5_nothink" if family == "qwen3_5" else "qwen3_nothink",
            "fidelity": "final_business_blind_plan_coverage",
        })
        if family == "qwen3_5":
            job["environment_overlay"] = _qwen35_environment_overlay()
        for key in ("peak_batch_contract", "replay_sampler_verification", "allocator_prefix_contract", "allocator_prefix_sampler_verification"):
            job.pop(key, None)
        jobs.append(job)
        prediction = dict(selected["prediction"])
        frozen_predictions.append({
            "job_id": job_id, "scenario": logical["scenario"], "source_dataset_id": source_id,
            "target_pressure": target_pressure, "actual_pressure": float(selected["pressure"]),
            "center_bytes": float(prediction["center_bytes"]),
            "risk_head_bytes": float(prediction["risk_guard_bytes"]),
            "admission_upper_bytes": float(prediction["admission_upper_bytes"]),
            "safe_limit_bytes": float(prediction["safe_limit_bytes"]),
            "admitted": bool(prediction["admitted"]),
            "model_id": logical["model_id"], "train_type": logical["train_type"],
            "gpu_count": logical["gpu_count"], "zero_stage": logical["zero_stage"],
            "gc": logical["gc"], "packing": logical["packing"],
            "mbs": logical["mbs"], "cutoff_len": logical["cutoff_len"],
        })

    if len(jobs) != EXPECTED_JOBS or len({row["job_id"] for row in jobs}) != EXPECTED_JOBS:
        raise RuntimeError("final blind queue size or uniqueness drifted")
    if any((ROOT / "results" / str(row["job_id"])).exists() for row in jobs):
        raise RuntimeError("one or more final blind result directories already exist")
    write_json(DATASET_INFO, registry)
    write_jsonl(args.queue, jobs)
    write_json(args.experiment, _experiment(jobs))

    public_profiles = []
    for (source_id, family), profile in profile_by_key.items():
        public_profiles.append({
            "source_dataset_id": source_id, "model_family": family,
            "full_dataset_id": profile["full_dataset_id"],
            "full_data_path": profile["full_data_path"], "full_data_sha256": profile["full_data_sha256"],
            "profile_path": profile["profile_path"], "profile_sha256": profile["profile_sha256"],
            "rows": len(profile["lengths"]), "maximum_tokens": profile["maximum_tokens"],
        })
    data_bundle: dict[str, Any] = {
        "schema": DATA_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": FREEZE_STATUS,
        "campaign_id": CAMPAIGN_ID,
        "gpu_training_started": False,
        "gpu_outcomes_observed": 0,
        "prior_campaign_gpu_outcomes_observed_at_freeze": PRIOR_CAMPAIGN_GPU_OUTCOMES_OBSERVED_AT_FREEZE,
        "final_acceptance_sources_read_for_cpu_profiles": len(selected_sources),
        "final_acceptance_gpu_outcomes_read": 0,
        "profiles": public_profiles,
        "prefix_bindings": prefix_bindings,
        "selection_binding": {"path": str(SELECTION.resolve()), "sha256": sha256_file(SELECTION)},
        "measurement_gate_binding": {"path": str(MEASUREMENT_GATE.resolve()), "sha256": sha256_file(MEASUREMENT_GATE)},
        "dataset_registry": {"path": str(DATASET_INFO.resolve()), "sha256": sha256_file(DATASET_INFO)},
    }
    data_bundle["report_sha256"] = sha256_json(data_bundle)
    write_json(args.data_bundle, data_bundle)
    predictions: dict[str, Any] = {
        "schema": PREDICTION_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": FREEZE_STATUS,
        "outcomes_observed": 0,
        "prior_campaign_gpu_outcomes_observed_at_freeze": PRIOR_CAMPAIGN_GPU_OUTCOMES_OBSERVED_AT_FREEZE,
        "model_or_margin_refit_allowed": False,
        "production_model_mutated": False,
        "v3_artifact": {"path": str(V3_ARTIFACT.resolve()), "sha256": sha256_file(V3_ARTIFACT)},
        "ordered_job_payload_sha256": sha256_json(jobs),
        "rows": frozen_predictions,
    }
    predictions["report_sha256"] = sha256_json(predictions)
    write_json(args.predictions, predictions)
    design: dict[str, Any] = {
        "schema": DESIGN_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "status": "frozen_waiting_for_gpu_0_3_preflight",
        "gpu_training_started": False,
        "gpu_outcomes_observed": 0,
        "prior_campaign_gpu_outcomes_observed_at_freeze": PRIOR_CAMPAIGN_GPU_OUTCOMES_OBSERVED_AT_FREEZE,
        "production_model_mutated": False,
        "execution_authorized": True,
        "authorized_gpu_ids": list(GPU_IDS),
        "jobs": EXPECTED_JOBS,
        "scenarios": EXPECTED_SCENARIOS,
        "target_pressures": list(TARGET_PRESSURES),
        "old_missing_34_jobs_included": False,
        "measurement_policy": {
            "nonpacking": "validated exact allocator-prefix replay",
            "packing": "full packed-dataset coverage",
            "packing_shortcut_used": False,
        },
        "scenario_ladders": scenario_ladders,
        "ordered_job_ids": [str(row["job_id"]) for row in jobs],
        "ordered_job_payload_sha256": sha256_json(jobs),
        "acceptance_thresholds": {
            "dataset_equal_center_mape_max": 0.15,
            "absolute_center_bias_max": 0.05,
            "p90_ape_max": 0.30,
            "train_type_mape_max": 0.20,
            "per_dataset_mape_max": 0.25,
            "safe_admission_rate_min": 0.95,
            "boundary_safe_admission_rate_min": 0.90,
            "unsafe_success_admitted_max": 0,
            "oom_admitted_max": 0,
            "minimum_exact_successes": 40,
            "minimum_safe_successes": 40,
            "minimum_ooms": 60,
            "minimum_difficult_ooms": 20,
        },
        "bindings": {
            "queue": {"path": str(args.queue.resolve()), "sha256": sha256_file(args.queue)},
            "experiment": {"path": str(args.experiment.resolve()), "sha256": sha256_file(args.experiment)},
            "data_bundle": {"path": str(args.data_bundle.resolve()), "sha256": sha256_file(args.data_bundle)},
            "frozen_predictions": {"path": str(args.predictions.resolve()), "sha256": sha256_file(args.predictions)},
            "v3_artifact": {"path": str(V3_ARTIFACT.resolve()), "sha256": sha256_file(V3_ARTIFACT)},
            "measurement_gate": {"path": str(MEASUREMENT_GATE.resolve()), "sha256": sha256_file(MEASUREMENT_GATE)},
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(args.design, design)


if __name__ == "__main__":
    main()
