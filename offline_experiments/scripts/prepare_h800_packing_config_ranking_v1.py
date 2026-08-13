#!/usr/bin/env python3
"""Prepare the H800 Packing configuration-ranking experiment.

The product question is deliberately narrower than absolute throughput
prediction: for one fixed model, business dataset, GPU type and GPU count, can
the shared throughput model rank ``cutoff_len x ZeRO stage x GC`` candidates
after replacing unpacked MBS with the upload-time mean samples per pack?

This CPU-only script freezes every input needed both for later model fitting and
for source-disjoint validation.  It does not launch GPU work and it does not
read any training outcome.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from common import (
    ARTIFACT_DIR,
    DATA_DIR,
    MATRIX_DIR,
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from packing_gbs_contract import derive_packing_gbs_contract


CAMPAIGN_ID = "h800_packing_config_ranking_20260812_v2"
PHASE_ID = "h800_packing_config_ranking_full_utilization_v2"
JOB_SCHEMA = "sft_h800_packing_config_ranking_job/v2"
DESIGN_SCHEMA = "sft_h800_packing_config_ranking_design/v2"
STATIC_SCHEMA = "sft_h800_packing_config_ranking_static/v2"
EXECUTION_POLICY_VERSION = "homogeneous_full_utilization_waves_v2"

GPU_IDS = tuple(range(8))
GPU_COUNTS = (1, 2, 4, 8)
MODEL_ID = "qwen3_8b"
MODEL_PATH = Path("/wanqing-models/Qwen3-8B")
TOKENIZER_PATH = MODEL_PATH
TEMPLATE = "qwen3_nothink"
TRAIN_TYPE = "lora"
TARGET_GBS = 256
PHYSICAL_MBS = 1
WARMUP_STEPS = 3
MEASURE_STEPS = 12
MINIMUM_RUNTIME_RECORDS = 8_192
MAXIMUM_GBS_CENTER_RELATIVE_ERROR = 0.10
PACK_COUNT_P99_GBS_EPSILON = 0.15

PROFILE_MANIFEST = (
    ARTIFACT_DIR / "h800_packing_final_business_profile_manifest_v1.json"
)
PROFILE_DIR = ARTIFACT_DIR / "h800_packing_final_business_profiles_v1"
MODEL_INVENTORY = ARTIFACT_DIR / "model_inventory.json"
BASE_RANKER = ARTIFACT_DIR / "rank_first_throughput_challenger_v1.json"

CAMPAIGN_DATA_DIR = DATA_DIR / "h800_packing_config_ranking_v2"
DATASET_REGISTRY = CAMPAIGN_DATA_DIR / "dataset_info.json"
JOBS_DIR = ARTIFACT_DIR / "h800_packing_config_ranking_jobs_v2"
STATIC = ARTIFACT_DIR / "h800_packing_config_ranking_static_v2.json"
DESIGN = ARTIFACT_DIR / "h800_packing_config_ranking_design_v2.json"
QUEUE_ALL = MATRIX_DIR / "h800_packing_config_ranking_all_v2.jsonl"
QUEUE_FIT = MATRIX_DIR / "h800_packing_config_ranking_fit_v2.jsonl"
QUEUE_HOLDOUT = MATRIX_DIR / "h800_packing_config_ranking_holdout_v2.jsonl"
STAGING_DIR = ROOT / "packing_config_ranking_staging"
EXPERIMENT = STAGING_DIR / "experiment.h800_packing_config_ranking_v2.json"
HOLDOUT_PREDICTIONS = (
    ARTIFACT_DIR / "h800_packing_config_ranking_holdout_frozen_predictions_v2.json"
)
INVALIDATED_PILOT = (
    ARTIFACT_DIR / "h800_packing_config_ranking_invalidated_pilot_v1.json"
)

EXPECTED_DATASETS = 6
EXPECTED_CUTOFF_SCENARIOS = 18
EXPECTED_RANKING_GROUPS_PER_SPLIT = 12
EXPECTED_PRIMARY_JOBS_PER_SPLIT = 126
EXPECTED_FILL_REPEATS_PER_SPLIT = 6
EXPECTED_JOBS_PER_SPLIT = 132
EXPECTED_JOBS = 264


@dataclass(frozen=True)
class Workload:
    workload_id: str
    dataset_id: str
    split_role: str
    shape_role: str
    profile_name: str
    cutoffs: tuple[int, int, int]


# The split and cutoffs are frozen before this campaign sees any GPU outcome.
# Fit and holdout cover similar length regimes but contain disjoint business
# sources.  Cutoffs were chosen to span materially different virtual MBS values
# without making the target GBS impossible on eight GPUs.
WORKLOADS = (
    Workload(
        "PF02",
        "dataset-li2sye-1780465080",
        "fit",
        "short",
        "pf02_dataset-li2sye-1780465080.json",
        (1_024, 2_048, 4_096),
    ),
    Workload(
        "PF04",
        "dataset-udwtrx-1785229698",
        "fit",
        "medium_concentrated",
        "pf04_dataset-udwtrx-1785229698.json",
        (1_024, 2_048, 4_096),
    ),
    Workload(
        "PF06",
        "dataset-mooxf7-1778662463",
        "fit",
        "long_concentrated",
        "pf06_dataset-mooxf7-1778662463.json",
        (4_096, 8_192, 16_384),
    ),
    Workload(
        "PH02",
        "dataset-zefhaf-1780147888",
        "prospective_holdout",
        "short_tail",
        "ph02_dataset-zefhaf-1780147888.json",
        (1_024, 2_048, 4_096),
    ),
    Workload(
        "PH03",
        "dataset-bpgwrk-1784805111",
        "prospective_holdout",
        "medium_broad",
        "ph03_dataset-bpgwrk-1784805111.json",
        (1_024, 2_048, 4_096),
    ),
    Workload(
        "PH06",
        "dataset-51m5uv-1784027387",
        "prospective_holdout",
        "extreme_long_tail",
        "ph06_dataset-51m5uv-1784027387.json",
        (4_096, 8_192, 16_384),
    ),
)


def mechanisms(gpu_count: int) -> tuple[tuple[int, bool], ...]:
    """Return legal ZeRO/GC candidates for one fixed card count."""

    if gpu_count == 1:
        return ((0, False), (0, True))
    if gpu_count in {2, 4, 8}:
        return ((2, False), (2, True), (3, False), (3, True))
    raise ValueError(f"unsupported gpu_count={gpu_count}")


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _iter_source(path: Path) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            if not isinstance(value, dict) or not all(
                isinstance(value.get(field), str)
                for field in ("system", "prompt", "response")
            ):
                raise ValueError(
                    f"{path}:{line_number} is not pure system/prompt/response SFT"
                )
            yield {
                field: str(value[field])
                for field in ("system", "prompt", "response")
            }


def _materialize_runtime_data(
    workload: Workload,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    source = Path(str(profile["source_binding"]["local_path"]))
    source_rows = list(_iter_source(source))
    expected_records = int(profile["records"])
    if len(source_rows) != expected_records:
        raise ValueError(
            f"{workload.workload_id} source row count drifted: "
            f"{len(source_rows)} != {expected_records}"
        )
    repetition = max(1, math.ceil(MINIMUM_RUNTIME_RECORDS / expected_records))
    runtime_rows = [copy.deepcopy(row) for _ in range(repetition) for row in source_rows]
    path = CAMPAIGN_DATA_DIR / f"{workload.workload_id.lower()}_{workload.dataset_id}.jsonl"
    write_jsonl(path, runtime_rows)
    return {
        "dataset_id": f"packing_config_rank_{workload.workload_id.lower()}_v1",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "source_records": expected_records,
        "runtime_records": len(runtime_rows),
        "distribution_preserving_repetition": repetition > 1,
        "repetition_factor": repetition,
    }


def _curve(profile: Mapping[str, Any], cutoff: int) -> dict[str, Any]:
    rows = [
        row
        for row in profile["packing_curve"]
        if int(row["cutoff_len"]) == cutoff
    ]
    if len(rows) != 1:
        raise ValueError(
            f"profile {profile.get('workload_id')} lacks one curve for cutoff={cutoff}"
        )
    return copy.deepcopy(rows[0])


def _static_contracts() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest = read_json(PROFILE_MANIFEST)
    if manifest.get("schema") != "sft_h800_packing_final_business_profiles/v1":
        raise ValueError("business profile manifest schema drifted")
    if manifest.get("split_contract", {}).get("holdout_may_tune_coefficients_or_margins") is not False:
        raise ValueError("business holdout contract drifted")

    contracts: list[dict[str, Any]] = []
    runtime: dict[str, dict[str, Any]] = {}
    registry: dict[str, Any] = {}
    for workload in WORKLOADS:
        profile_path = PROFILE_DIR / workload.profile_name
        profile = read_json(profile_path)
        identity = (
            profile.get("workload_id"),
            profile.get("dataset_id"),
            profile.get("split_role"),
            profile.get("shape_role"),
        )
        expected = (
            workload.workload_id,
            workload.dataset_id,
            workload.split_role,
            workload.shape_role,
        )
        if identity != expected:
            raise ValueError(f"profile identity drifted: {identity} != {expected}")
        runtime_row = _materialize_runtime_data(workload, profile)
        runtime[workload.workload_id] = runtime_row
        registry[str(runtime_row["dataset_id"])] = {
            "file_name": str(
                Path(str(runtime_row["path"])).resolve().relative_to(
                    CAMPAIGN_DATA_DIR.resolve()
                )
            ),
            "columns": {
                "prompt": "prompt",
                "response": "response",
                "system": "system",
            },
        }
        for cutoff in workload.cutoffs:
            curve = _curve(profile, cutoff)
            samples = curve["samples_per_pack"]
            gbs_by_gpu: dict[str, Any] = {}
            for gpu_count in GPU_COUNTS:
                gbs = derive_packing_gbs_contract(
                    target_gbs=TARGET_GBS,
                    data_parallel=gpu_count,
                    samples_per_pack=samples,
                    epsilon_gbs=PACK_COUNT_P99_GBS_EPSILON,
                    maximum_center_relative_error=(
                        MAXIMUM_GBS_CENTER_RELATIVE_ERROR
                    ),
                )
                if gbs["gates"]["candidate_admissible"] is not True:
                    raise ValueError(
                        f"GBS contract rejected {workload.workload_id}/c{cutoff}/"
                        f"dp{gpu_count}: {gbs['gates']['reason_codes']}"
                    )
                gbs_by_gpu[str(gpu_count)] = gbs
            contracts.append(
                {
                    "workload_id": workload.workload_id,
                    "source_dataset_id": workload.dataset_id,
                    "runtime_dataset_id": runtime_row["dataset_id"],
                    "split_role": workload.split_role,
                    "shape_role": workload.shape_role,
                    "cutoff_len": cutoff,
                    "virtual_mbs": float(samples["mean"]),
                    "upload_time_packing_features": {
                        "packing_capacity": int(curve["packing_capacity"]),
                        "packs": int(curve["packs"]),
                        "pack_utilization": float(curve["pack_utilization"]),
                        "model_facing_pack_fill_ratio": float(
                            curve["model_facing_pack_fill_ratio"]
                        ),
                        "sequence_reduction_ratio": float(
                            curve["sequence_reduction_ratio"]
                        ),
                        "sample_truncation_rate": float(
                            curve["sample_truncation_rate"]
                        ),
                        "tokens_retained_ratio": float(
                            curve["tokens_retained_ratio"]
                        ),
                        "samples_per_pack": copy.deepcopy(samples),
                    },
                    "packing_gbs_contract_by_gpu_count": gbs_by_gpu,
                    "source_binding": {
                        "path": runtime_row["source_path"],
                        "sha256": runtime_row["source_sha256"],
                    },
                    "runtime_data_binding": {
                        "path": runtime_row["path"],
                        "sha256": runtime_row["sha256"],
                        "source_records": runtime_row["source_records"],
                        "runtime_records": runtime_row["runtime_records"],
                        "distribution_preserving_repetition": runtime_row[
                            "distribution_preserving_repetition"
                        ],
                        "repetition_factor": runtime_row["repetition_factor"],
                    },
                    "dataset_profile_binding": _binding(profile_path),
                }
            )

    write_json(DATASET_REGISTRY, registry)
    if len(contracts) != EXPECTED_CUTOFF_SCENARIOS:
        raise ValueError("static cutoff scenario count drifted")
    return contracts, runtime


def _model() -> dict[str, Any]:
    models = read_json(MODEL_INVENTORY)["models"]
    matches = [row for row in models if row.get("id") == MODEL_ID]
    if len(matches) != 1:
        raise ValueError(f"model inventory lacks exactly one {MODEL_ID}")
    return matches[0]


def _zero_name(stage: int) -> str:
    return "none" if stage == 0 else f"zero{stage}"


def _job(
    contract: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    gpu_count: int,
    zero_stage: int,
    gc: bool,
    registry_sha256: str,
) -> dict[str, Any]:
    gbs = contract["packing_gbs_contract_by_gpu_count"][str(gpu_count)]
    cutoff = int(contract["cutoff_len"])
    role = str(contract["split_role"])
    ranking_group = (
        f"{contract['workload_id']}-qwen3_8b-lora-dp{gpu_count}-gbs{TARGET_GBS}"
    )
    fixed_cutoff_group = f"{ranking_group}-c{cutoff}"
    arm_id = f"c{cutoff}-z{zero_stage}-gc{int(gc)}"
    job: dict[str, Any] = {
        "schema": JOB_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "candidate_role": (
            "model_fit_or_refit" if role == "fit" else "prospective_validation_only"
        ),
        "split_role": role,
        "split_unit_id": str(contract["source_dataset_id"]),
        "workload_id": str(contract["workload_id"]),
        "shape_role": str(contract["shape_role"]),
        "scenario_id": ranking_group,
        "ranking_group_id": ranking_group,
        "fixed_cutoff_mechanism_group_id": fixed_cutoff_group,
        "arm_id": arm_id,
        "repeat": 0,
        "model_id": MODEL_ID,
        "model_family": "qwen3",
        "model_path": str(MODEL_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "template": TEMPLATE,
        "model_parameters": int(model["actual_parameters"]),
        "train_type": TRAIN_TYPE,
        "dataset_id": str(contract["runtime_dataset_id"]),
        "source_dataset_id": str(contract["source_dataset_id"]),
        "data_path": str(contract["runtime_data_binding"]["path"]),
        "data_sha256": str(contract["runtime_data_binding"]["sha256"]),
        "dataset_dir": str(CAMPAIGN_DATA_DIR.resolve()),
        "dataset_registry_sha256": registry_sha256,
        "dataset_profile_path": str(contract["dataset_profile_binding"]["path"]),
        "dataset_profile_sha256": str(
            contract["dataset_profile_binding"]["sha256"]
        ),
        "cutoff_len": cutoff,
        "target_gbs": TARGET_GBS,
        "gpu_count": gpu_count,
        "zero": _zero_name(zero_stage),
        "zero_stage": zero_stage,
        "gc": gc,
        "gradient_checkpointing": gc,
        "mbs": PHYSICAL_MBS,
        "physical_mbs": PHYSICAL_MBS,
        "gradient_accumulation_steps": int(gbs["gradient_accumulation_steps"]),
        "packing": True,
        "neat_packing": True,
        "virtual_mbs": float(contract["virtual_mbs"]),
        "expected_sample_gbs": float(gbs["expected_epoch_sample_gbs"]),
        "expected_sample_gbs_relative_error": float(
            gbs["expected_epoch_sample_gbs_relative_error"]
        ),
        "upload_time_packing_features": copy.deepcopy(
            contract["upload_time_packing_features"]
        ),
        "packing_gbs_contract": copy.deepcopy(gbs),
        "model_input_projection": {
            "packing_physical_mbs": PHYSICAL_MBS,
            "throughput_virtual_mbs": float(contract["virtual_mbs"]),
            "virtual_mbs_source": "upload_time_samples_per_pack.mean",
            "packing_specific_throughput_coefficient": None,
            "use_shared_unpacked_throughput_features": True,
        },
        "ranking_contract": {
            "primary_group": ranking_group,
            "primary_candidates_vary": [
                "cutoff_len",
                "zero_stage",
                "gradient_checkpointing",
            ],
            "fixed_cutoff_diagnostic_group": fixed_cutoff_group,
            "primary_observed_metric": "global_logical_samples_per_second",
            "secondary_observed_metrics": [
                "global_effective_tokens_per_second",
                "global_computed_tokens_per_second",
            ],
            "cross_gpu_ranking_is_primary": False,
        },
        "offload": False,
        "gpu_type": "NVIDIA H800 140GB HBM3",
        "required_runtime_gpu_name": "NVIDIA H800",
        "hardware_id": "local_h800_140g",
        "kind": "throughput",
        "warmup_steps": WARMUP_STEPS,
        "measure_steps": MEASURE_STEPS,
        "max_samples": int(contract["runtime_data_binding"]["runtime_records"]),
        "fidelity": "formal_throughput_3plus12_no_repeat",
        "execution_policy_version": EXECUTION_POLICY_VERSION,
        "measurement_role": "primary_ranking_candidate",
        "ranking_eligible": True,
        "homogeneous_card_count_wave": True,
        "parallel_class": (
            "exclusive_pool"
            if gpu_count == 8
            else "disjoint_wave"
            if gpu_count == 4
            else "gpu_partitionable"
        ),
        "allow_disjoint_wave_for_large_job": gpu_count == 4,
        "requires_external_node_idle": False,
        "strict_queue_order": True,
        "publication_allowed": False,
    }
    if contract["runtime_data_binding"]["distribution_preserving_repetition"]:
        job.update(
            {
                "execution_data_distribution_preserving_repetition": True,
                "execution_data_repetition_factor": int(
                    contract["runtime_data_binding"]["repetition_factor"]
                ),
            }
        )
    job["job_id"] = stable_id("h800packrank", job)
    return job


def _arm_order_key(job: Mapping[str, Any]) -> str:
    material = "|".join(
        [
            str(job["ranking_group_id"]),
            str(job["cutoff_len"]),
            str(job["zero_stage"]),
            str(int(bool(job["gc"]))),
            "20260812",
        ]
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _round_robin_order(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        grouped.setdefault(str(job["ranking_group_id"]), []).append(job)
    for rows in grouped.values():
        rows.sort(key=_arm_order_key)
    ordered: list[dict[str, Any]] = []
    maximum = max(len(rows) for rows in grouped.values())
    for index in range(maximum):
        for group_id in sorted(grouped):
            rows = grouped[group_id]
            if index < len(rows):
                ordered.append(rows[index])
    return ordered


def _full_utilization_repeats(
    jobs: list[dict[str, Any]], *, split_role: str
) -> list[dict[str, Any]]:
    """Add six useful 1-GPU repeats so the one-GPU segment fills three waves."""

    candidates = [
        row
        for row in jobs
        if row["split_role"] == split_role
        and int(row["gpu_count"]) == 1
        and int(row["cutoff_len"])
        == sorted(
            int(current["cutoff_len"])
            for current in jobs
            if current["workload_id"] == row["workload_id"]
            and int(current["gpu_count"]) == 1
        )[2]
    ]
    if len(candidates) != EXPECTED_FILL_REPEATS_PER_SPLIT:
        raise ValueError(
            f"expected {EXPECTED_FILL_REPEATS_PER_SPLIT} one-GPU fill repeats, "
            f"got {len(candidates)}"
        )
    repeats: list[dict[str, Any]] = []
    for source in candidates:
        repeat = copy.deepcopy(source)
        repeat.pop("job_id", None)
        repeat["repeat"] = 1
        repeat["arm_id"] = f"{repeat['arm_id']}-repeat1"
        repeat["candidate_role"] = (
            "concurrency_noise_fit_repeat"
            if split_role == "fit"
            else "prospective_concurrency_noise_repeat"
        )
        repeat["measurement_role"] = "full_utilization_repeat"
        repeat["ranking_eligible"] = False
        repeat["job_id"] = stable_id("h800packrank", repeat)
        repeats.append(repeat)
    return repeats


def _homogeneous_full_utilization_order(
    jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    wave_index = 0
    for gpu_count in GPU_COUNTS:
        current = _round_robin_order(
            [row for row in jobs if int(row["gpu_count"]) == gpu_count]
        )
        capacity = len(GPU_IDS) // gpu_count
        if len(current) % capacity:
            raise ValueError(
                f"gpu_count={gpu_count} has {len(current)} jobs, which cannot "
                f"form full {capacity}-job waves"
            )
        for offset, job in enumerate(current):
            job["execution_wave_index"] = wave_index + offset // capacity
            job["execution_wave_gpu_count"] = gpu_count
            job["execution_wave_capacity"] = capacity
        wave_index += len(current) // capacity
        ordered.extend(current)
    for index, job in enumerate(ordered):
        job["execution_sequence_index"] = index
    return ordered


def _build_jobs(contracts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    model = _model()
    registry_sha256 = sha256_file(DATASET_REGISTRY)
    jobs: list[dict[str, Any]] = []
    for contract in contracts:
        for gpu_count in GPU_COUNTS:
            for zero_stage, gc in mechanisms(gpu_count):
                jobs.append(
                    _job(
                        contract,
                        model=model,
                        gpu_count=gpu_count,
                        zero_stage=zero_stage,
                        gc=gc,
                        registry_sha256=registry_sha256,
                    )
                )
    if len(jobs) != EXPECTED_PRIMARY_JOBS_PER_SPLIT * 2:
        raise ValueError("Packing primary ranking matrix must contain 252 jobs")
    jobs.extend(_full_utilization_repeats(jobs, split_role="fit"))
    jobs.extend(
        _full_utilization_repeats(jobs, split_role="prospective_holdout")
    )
    if len(jobs) != EXPECTED_JOBS or len({row["job_id"] for row in jobs}) != EXPECTED_JOBS:
        raise ValueError("Packing ranking matrix must contain 264 unique jobs")
    fit = _homogeneous_full_utilization_order(
        [copy.deepcopy(row) for row in jobs if row["split_role"] == "fit"]
    )
    holdout = _homogeneous_full_utilization_order(
        [copy.deepcopy(row) for row in jobs if row["split_role"] == "prospective_holdout"]
    )
    if len(fit) != EXPECTED_JOBS_PER_SPLIT or len(holdout) != EXPECTED_JOBS_PER_SPLIT:
        raise ValueError("fit/holdout queue size drifted")
    return {"fit": fit, "prospective_holdout": holdout, "all": fit + holdout}


def _experiment(runtime: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    current = read_json(ROOT / "config" / "experiment.json")
    return {
        "schema_version": 1,
        "training_scope": {
            "phase_id": PHASE_ID,
            "model_ids": [MODEL_ID],
            "gpu_ids": list(GPU_IDS),
            "exclusive_node_gpu_ids": list(GPU_IDS),
            "max_gpu_count": 8,
            "stage": "sft",
            "precision": "bf16",
            "gpu_type": "NVIDIA H800 140GB HBM3",
            "gpu_counts": list(GPU_COUNTS),
            "global_batch_sizes": [TARGET_GBS],
            "gradient_checkpointing": [False, True],
            "zero_by_gpu_count": {
                "1": ["none"],
                "2": ["zero2", "zero3"],
                "4": ["zero2", "zero3"],
                "8": ["zero2", "zero3"],
            },
            "objective": (
                "rank Packed cutoff_len x ZeRO x GC candidates independently "
                "inside each fixed H800 GPU count using upload-time virtual MBS"
            ),
        },
        "fixed_runtime": current["fixed_runtime"],
        "measurement": {
            **current["measurement"],
            "throughput_warmup_steps": WARMUP_STEPS,
            "throughput_measure_steps": MEASURE_STEPS,
            "performance_parallelism": "disjoint_gpu_masks",
            "scheduler_order_policy": "strict_homogeneous_card_count_waves",
            "formal_throughput_requires_exclusive_node": False,
        },
        "datasets": [
            {
                "id": row["dataset_id"],
                "workload_id": workload_id,
                "runtime_records": row["runtime_records"],
                "distribution_preserving_repetition": row[
                    "distribution_preserving_repetition"
                ],
            }
            for workload_id, row in sorted(runtime.items())
        ],
        "packing_static_gate": {
            "virtual_mbs_source": "upload_time samples_per_pack.mean",
            "maximum_expected_gbs_relative_error": (
                MAXIMUM_GBS_CENTER_RELATIVE_ERROR
            ),
            "pack_count_p99_gbs_epsilon": PACK_COUNT_P99_GBS_EPSILON,
            "recommendation_time_raw_rows_required": False,
        },
    }


def _queue_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    cutoffs: set[tuple[str, int]] = set()
    for row in rows:
        if row.get("ranking_eligible") is True:
            groups.setdefault(str(row["ranking_group_id"]), []).append(row)
        cutoffs.add((str(row["workload_id"]), int(row["cutoff_len"])))
    return {
        "jobs": len(rows),
        "ranking_groups": len(groups),
        "dataset_cutoff_scenarios": len(cutoffs),
        "candidates_per_ranking_group": {
            group: len(current) for group, current in sorted(groups.items())
        },
    }


def prepare() -> dict[str, Any]:
    contracts, runtime = _static_contracts()
    queues = _build_jobs(contracts)
    write_jsonl(QUEUE_FIT, queues["fit"])
    write_jsonl(QUEUE_HOLDOUT, queues["prospective_holdout"])
    write_jsonl(QUEUE_ALL, queues["all"])
    for job in queues["all"]:
        write_json(JOBS_DIR / f"{job['job_id']}.json", job)
    write_json(EXPERIMENT, _experiment(runtime))

    static: dict[str, Any] = {
        "schema": STATIC_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_before_campaign_gpu_execution": True,
        "campaign_id": CAMPAIGN_ID,
        "product_contract": {
            "physical_mbs_when_packing": PHYSICAL_MBS,
            "throughput_virtual_mbs": "samples_per_pack.mean",
            "gradient_accumulation_from_target_gbs": True,
            "recommendation_time_raw_dataset_required": False,
            "packing_specific_throughput_coefficient": None,
        },
        "tokenizer_and_packer_binding": {
            "source_profile_manifest": _binding(PROFILE_MANIFEST),
            "tokenizer": read_json(PROFILE_MANIFEST)["tokenizer_binding"],
            "packer": read_json(PROFILE_MANIFEST)["packer_binding"],
        },
        "contracts": contracts,
    }
    static["report_sha256"] = sha256_json(static)
    write_json(STATIC, static)

    bindings = {
        "static": _binding(STATIC),
        "fit_queue": _binding(QUEUE_FIT),
        "holdout_queue": _binding(QUEUE_HOLDOUT),
        "all_queue": _binding(QUEUE_ALL),
        "experiment": _binding(EXPERIMENT),
        "dataset_registry": _binding(DATASET_REGISTRY),
        "profile_manifest": _binding(PROFILE_MANIFEST),
        "model_inventory": _binding(MODEL_INVENTORY),
        "base_ranker": _binding(BASE_RANKER),
    }
    design: dict[str, Any] = {
        "schema": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_not_authorized_not_started",
        "gpu_training_started": False,
        "model": "Qwen3-8B LoRA",
        "hardware": "H800 140GB",
        "dimensions": {
            "gpu_counts": list(GPU_COUNTS),
            "cutoffs_per_dataset": 3,
            "single_gpu_mechanisms": ["ZeRO-0/GC-off", "ZeRO-0/GC-on"],
            "multi_gpu_mechanisms": [
                "ZeRO-2/GC-off",
                "ZeRO-2/GC-on",
                "ZeRO-3/GC-off",
                "ZeRO-3/GC-on",
            ],
            "datasets": EXPECTED_DATASETS,
            "dataset_cutoff_scenarios": EXPECTED_CUTOFF_SCENARIOS,
            "jobs": EXPECTED_JOBS,
            "primary_ranking_jobs": EXPECTED_PRIMARY_JOBS_PER_SPLIT * 2,
            "full_utilization_repeats": EXPECTED_FILL_REPEATS_PER_SPLIT * 2,
        },
        "split_contract": {
            "fit_dataset_ids": [
                workload.dataset_id
                for workload in WORKLOADS
                if workload.split_role == "fit"
            ],
            "prospective_holdout_dataset_ids": [
                workload.dataset_id
                for workload in WORKLOADS
                if workload.split_role == "prospective_holdout"
            ],
            "fit_queue_may_refit_shared_nonpacking_coefficients": True,
            "packing_specific_feature_or_coefficient_allowed": False,
            "holdout_may_tune_model_coefficients_or_margins": False,
            "holdout_requires_predictions_frozen_before_first_holdout_job": True,
            "holdout_predictions_path": str(HOLDOUT_PREDICTIONS.resolve()),
        },
        "ranking_contract": {
            "primary_group_fixed": [
                "gpu_type",
                "gpu_count",
                "model",
                "training_mode",
                "dataset",
                "target_gbs",
            ],
            "primary_group_candidates_vary": [
                "cutoff_len",
                "zero_stage",
                "gradient_checkpointing",
            ],
            "cross_gpu_ranking_is_primary": False,
            "primary_truth": "global logical samples per second",
            "secondary_truths": [
                "global effective tokens per second",
                "global computed tokens per second",
            ],
            "metrics": [
                "pairwise ordering accuracy",
                "exact Top1 hit rate",
                "Hit90: selected throughput >= 90% of oracle",
                "Top1 regret",
            ],
            "dataset_generalization": (
                "report holdout metrics per dataset and macro-average datasets"
            ),
        },
        "recording_contract": {
            "pre_run": [
                "raw dataset and profile SHA256",
                "tokenizer/template/packer fingerprint",
                "full upload-time samples-per-pack distribution per cutoff",
                "mean virtual MBS",
                "pack utilization, truncation and retained-token ratios",
                "target/expected sample GBS and integer GAS deviation",
                "GPU count, cutoff, ZeRO, GC and all fixed runtime fields",
                "fit/holdout split unit and both ranking group IDs",
            ],
            "runtime": [
                "logical/effective/computed tokens and samples for every measured step",
                "step time and optimizer time distributions",
                "allocated/reserved peak memory on every rank",
                "actual logical sample counts and sequence lengths per packed batch",
                "Packing semantic checks",
                "GPU UUID/topology/driver/software/runtime fingerprint",
                "thermal, clock, power, OOM and terminal classification evidence",
            ],
            "user_content_in_metric_files": False,
        },
        "measurement_contract": {
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "primary_repeats": 1,
            "one_gpu_middle_cutoff_noise_repeats_per_split": 6,
            "whole_node_exclusive_for_every_job": False,
            "full_node_utilization": True,
            "homogeneous_card_count_waves": True,
            "concurrency_by_gpu_count": {"1": 8, "2": 4, "4": 2, "8": 1},
            "deterministic_gpu_count_then_round_robin_arm_order": True,
            "automatic_retries_or_extra_repeats": False,
        },
        "invalidated_pilot": {
            "path": str(INVALIDATED_PILOT.resolve()),
            "reason": (
                "v1 ran whole-node-exclusive jobs serially; its measurements are "
                "not comparable with v2 concurrent-wave measurements"
            ),
            "may_enter_v2_fit_or_validation": False,
        },
        "queue_summaries": {
            "fit": _queue_summary(queues["fit"]),
            "prospective_holdout": _queue_summary(
                queues["prospective_holdout"]
            ),
        },
        "bindings": bindings,
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    return {
        "design": _binding(DESIGN),
        "static": _binding(STATIC),
        "experiment": _binding(EXPERIMENT),
        "fit_queue": _binding(QUEUE_FIT),
        "holdout_queue": _binding(QUEUE_HOLDOUT),
        "jobs": EXPECTED_JOBS,
        "fit_jobs": EXPECTED_JOBS_PER_SPLIT,
        "holdout_jobs": EXPECTED_JOBS_PER_SPLIT,
        "ranking_groups": EXPECTED_RANKING_GROUPS_PER_SPLIT * 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
