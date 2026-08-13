#!/usr/bin/env python3
"""Freeze the compact prospective RTX 4090 generalization campaign.

This script writes no training result and launches no GPU process.  It creates
two runtime-isolated sub-campaigns:

* Qwen3-8B uses the original RTX 4090 FA2 runtime.
* Qwen3.5-4B uses the resolved Transformers-5.3/FLA/TileLang runtime, while
  importing the ABI-matched FA2 wheel from the original 4090 environment.

Predictions are frozen before either queue is eligible to execute.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

from common import ROOT, read_json, sha256_file, sha256_json, stable_id, write_json, write_jsonl
from rtx4090_safety_v2_modeling import DEFAULT_OUTPUT as MODEL_ARTIFACT
from rtx4090_safety_v2_predictor import RTX4090SafetyV2Predictor


CAMPAIGN_ID = "rtx4090_generalization_20260729"
DEFAULT_CAMPAIGN_ROOT = ROOT / "campaigns" / CAMPAIGN_ID
OLD_4090_ROOT = ROOT / "campaigns" / "rtx4090_20260717"
QWEN35_OVERLAY = Path("/tmp/qwen35_tilelang_overlay_0.1.12")
QWEN35_OVERLAY_SHA256 = (
    "8496316be58be29f20a8dbf91a34279b79afeb13e5769ff5c3508e0587d04f09"
)
QWEN35_OVERLAY_FILES = 7064
QWEN35_PYTHON = (
    Path(__file__).resolve().parents[2] / "qwen36_venv" / "bin" / "python"
)
QWEN35_TORCHRUN = (
    Path(__file__).resolve().parents[2]
    / "qwen36_venv"
    / "bin"
    / "torchrun"
)
QWEN35_SITE_PACKAGES = (
    Path(__file__).resolve().parents[2]
    / "qwen36_venv"
    / "lib"
    / "python3.11"
    / "site-packages"
)
RTX4090_SITE_PACKAGES = Path(
    "/fine-tuning-launcher/.venv-4090/lib/python3.11/site-packages"
)
LIVE_APPROVAL_SOURCE = (
    ROOT / "scripts" / "freeze_rtx4090_generalization_live_approval.py"
)
DATASETS = {
    "short_512": 512,
    "multiturn_4096": 4096,
    "longtail_8192": 8192,
}
MODELS: dict[str, dict[str, Any]] = {
    "qwen3_8b": {
        "id": "qwen3_8b",
        "nominal_scale_b": 8,
        "path": "/wanqing-models/Qwen3-8B",
        "tokenizer_path": "/wanqing-models/Qwen3-8B",
        "family": "qwen3",
        "template": "qwen3_nothink",
        "train_types": ["lora"],
        "actual_parameters": 8_190_735_360,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "generalization_role": "unseen_same_family_scale",
    },
    "qwen3p5_4b": {
        "id": "qwen3p5_4b",
        "nominal_scale_b": 4,
        "path": "/wanqing-models/Qwen3.5-4B",
        "tokenizer_path": "/wanqing-models/Qwen3.5-4B",
        "family": "qwen3_5",
        "template": "qwen3_nothink",
        "train_types": ["lora"],
        "actual_parameters": 4_659_865_088,
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "vocab_size": 248320,
        "generalization_role": "unseen_hybrid_attention_family",
    },
}


def _zero(gpu_count: int, value: str | None = None) -> str:
    if gpu_count == 1:
        if value not in {None, "none"}:
            raise ValueError("Single-GPU job cannot use ZeRO")
        return "none"
    return value or "zero2"


def _job(
    *,
    prefix: str,
    model_id: str,
    phase: str,
    kind: str,
    fidelity: str,
    dataset_id: str,
    gpu_count: int,
    mbs: int,
    gc: bool,
    zero: str | None = None,
    packing: bool = False,
    repeat: int = 0,
    evaluation_group: str,
    candidate_label: str,
    warmup_steps: int | None = None,
    measure_steps: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> dict[str, Any]:
    model = MODELS[model_id]
    row: dict[str, Any] = {
        "schema": "sft_rtx4090_generalization_job/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": phase,
        "kind": kind,
        "fidelity": fidelity,
        "evaluation_group": evaluation_group,
        "candidate_label": candidate_label,
        "model_family": model["family"],
        "model_id": model_id,
        "model_parameters": model["actual_parameters"],
        "model_path": model["path"],
        "tokenizer_path": model["tokenizer_path"],
        "template": model["template"],
        "train_type": "lora",
        "dataset_id": dataset_id,
        "cutoff_len": DATASETS[dataset_id],
        "gpu_count": gpu_count,
        "zero": _zero(gpu_count, zero),
        "gc": gc,
        "mbs": mbs,
        "target_gbs": 64,
        "packing": packing,
        "repeat": repeat,
        "parallel_class": "gpu_partitionable",
        "requires_external_node_idle": False,
    }
    if warmup_steps is not None:
        row["warmup_steps"] = warmup_steps
    if measure_steps is not None:
        row["measure_steps"] = measure_steps
    if gradient_accumulation_steps is not None:
        row["gradient_accumulation_steps"] = (
            gradient_accumulation_steps
        )
    row["job_id"] = stable_id(prefix, row)
    row["prediction_request_id"] = stable_id(
        "pred",
        {
            key: row[key]
            for key in (
                "evaluation_group",
                "model_id",
                "train_type",
                "dataset_id",
                "gpu_count",
                "zero",
                "gc",
                "mbs",
                "target_gbs",
                "packing",
            )
        },
    )
    return row


def _canaries() -> list[dict[str, Any]]:
    return [
        _job(
            prefix="r4090canary",
            model_id=model_id,
            phase="compatibility_canary",
            kind="throughput_screen",
            fidelity="canary_0plus2",
            dataset_id="short_512",
            gpu_count=1,
            mbs=1,
            gc=True,
            evaluation_group=f"{model_id}-canary",
            candidate_label="one_gpu_lora_gc_mbs1",
            warmup_steps=0,
            measure_steps=2,
        )
        for model_id in MODELS
    ]


def _memory_boundaries() -> list[dict[str, Any]]:
    specs = [
        # Qwen3-8B: same-family scale extrapolation around the 24-GiB line.
        ("qwen3_8b", "multiturn_4096", 1, 1, True, None, "4k_1g_m1"),
        ("qwen3_8b", "multiturn_4096", 1, 2, True, None, "4k_1g_m2"),
        ("qwen3_8b", "multiturn_4096", 2, 1, True, "zero2", "4k_2g_m1"),
        ("qwen3_8b", "multiturn_4096", 4, 1, True, "zero2", "4k_4g_m1"),
        ("qwen3_8b", "longtail_8192", 1, 1, True, None, "8k_1g_m1"),
        # Qwen3.5-4B: safe/unsafe brackets at two token distributions.
        ("qwen3p5_4b", "multiturn_4096", 1, 8, True, None, "4k_1g_m8"),
        ("qwen3p5_4b", "multiturn_4096", 1, 16, True, None, "4k_1g_m16"),
        ("qwen3p5_4b", "longtail_8192", 1, 4, True, None, "8k_1g_m4"),
        ("qwen3p5_4b", "longtail_8192", 1, 8, True, None, "8k_1g_m8"),
    ]
    return [
        _job(
            prefix="r4090mem",
            model_id=model_id,
            phase="memory_boundary",
            kind="memory_probe",
            fidelity="memory_5step",
            dataset_id=dataset_id,
            gpu_count=gpu_count,
            mbs=mbs,
            gc=gc,
            zero=zero,
            evaluation_group=f"{model_id}-memory-boundary",
            candidate_label=label,
        )
        for model_id, dataset_id, gpu_count, mbs, gc, zero, label in specs
    ]


def _ranking() -> list[dict[str, Any]]:
    scenarios = {
        "qwen3_8b": {
            "dataset_id": "short_512",
            "candidates": [
                (1, 4, True, None, "1g_gc_m4"),
                (1, 1, False, None, "1g_nogc_m1"),
                (2, 1, False, "zero2", "2g_z2_nogc_m1"),
                (4, 1, False, "zero2", "4g_z2_nogc_m1"),
            ],
        },
        "qwen3p5_4b": {
            "dataset_id": "multiturn_4096",
            "candidates": [
                (1, 2, True, None, "1g_gc_m2"),
                (1, 4, True, None, "1g_gc_m4"),
                (2, 4, True, "zero2", "2g_z2_gc_m4"),
                (4, 4, True, "zero2", "4g_z2_gc_m4"),
            ],
        },
    }
    rows = []
    for model_id, scenario in scenarios.items():
        for repeat in (0, 1):
            for gpu_count, mbs, gc, zero, label in scenario[
                "candidates"
            ]:
                rows.append(
                    _job(
                        prefix="r4090rank",
                        model_id=model_id,
                        phase="multi_candidate_ranking",
                        kind="throughput",
                        fidelity="formal_3plus10",
                        dataset_id=str(scenario["dataset_id"]),
                        gpu_count=gpu_count,
                        mbs=mbs,
                        gc=gc,
                        zero=zero,
                        repeat=repeat,
                        evaluation_group=f"{model_id}-ranking",
                        candidate_label=label,
                        warmup_steps=3,
                        measure_steps=10,
                    )
                )
    return rows


def _packing_abba() -> list[dict[str, Any]]:
    # Keep packing to one same-family unseen-scale pair.  Qwen3.5 tokenization
    # needs a separately generated packing profile and is intentionally not
    # mixed into this compact queue.
    rows = []
    sequence = (
        (False, 0, "A0_off"),
        (True, 0, "B0_on"),
        (True, 1, "B1_on"),
        (False, 1, "A1_off"),
    )
    for packing, repeat, label in sequence:
        rows.append(
            _job(
                prefix="r4090pack",
                model_id="qwen3_8b",
                phase="packing_abba",
                kind="throughput",
                fidelity="packing_2plus8",
                dataset_id="multiturn_4096",
                gpu_count=1,
                mbs=1,
                gc=True,
                packing=packing,
                repeat=repeat,
                evaluation_group="qwen3_8b-packing-abba",
                candidate_label=label,
                warmup_steps=2,
                measure_steps=8,
                gradient_accumulation_steps=19 if packing else None,
            )
        )
    return rows


def _validate_jobs(rows: Sequence[Mapping[str, Any]]) -> None:
    ids = [str(row["job_id"]) for row in rows]
    if len(rows) != 31 or len(ids) != len(set(ids)):
        raise ValueError("Expected exactly 31 unique prospective jobs")
    counts = Counter(str(row["phase_id"]) for row in rows)
    expected = {
        "compatibility_canary": 2,
        "memory_boundary": 9,
        "multi_candidate_ranking": 16,
        "packing_abba": 4,
    }
    if dict(counts) != expected:
        raise ValueError(f"Prospective phase counts drifted: {counts}")
    for row in rows:
        gpu_count = int(row["gpu_count"])
        mbs = int(row["mbs"])
        if row["packing"]:
            if mbs != 1 or int(row["gradient_accumulation_steps"]) <= 0:
                raise ValueError(f"Invalid packing row {row['job_id']}")
        elif 64 % (gpu_count * mbs):
            raise ValueError(f"GBS divisibility failed: {row['job_id']}")
        if gpu_count == 1 and row["zero"] != "none":
            raise ValueError(f"Single-GPU ZeRO drift: {row['job_id']}")
        if gpu_count > 1 and row["zero"] not in {"zero2", "zero3"}:
            raise ValueError(f"Multi-GPU ZeRO drift: {row['job_id']}")


def _prediction_request(job: Mapping[str, Any]) -> dict[str, Any]:
    model = MODELS[str(job["model_id"])]
    return {
        "request_id": str(job["prediction_request_id"]),
        "comparison_group": str(job["evaluation_group"]),
        "hardware_id": "rtx4090",
        "model_id": str(job["model_id"]),
        "model": {
            key: model[key]
            for key in (
                "id",
                "actual_parameters",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
            )
        },
        "training_mode": str(job["train_type"]),
        "dataset_id": str(job["dataset_id"]),
        "target_gbs": int(job["target_gbs"]),
        "cutoff_len": int(job["cutoff_len"]),
        "gpu_count": int(job["gpu_count"]),
        "mbs": int(job["mbs"]),
        "zero": str(job["zero"]),
        "gc": bool(job["gc"]),
        "packing": bool(job["packing"]),
        **(
            {
                "gradient_accumulation_steps": int(
                    job["gradient_accumulation_steps"]
                )
            }
            if job.get("gradient_accumulation_steps") is not None
            else {}
        ),
    }


def _unique_prediction_requests(
    jobs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for job in jobs:
        request = _prediction_request(job)
        request_id = str(request["request_id"])
        previous = by_id.get(request_id)
        if previous is not None and previous != request:
            raise ValueError(f"Prediction ID collision: {request_id}")
        by_id[request_id] = request
    return list(by_id.values())


def _experiment_config(
    *,
    model_id: str,
    qwen35: bool,
) -> dict[str, Any]:
    if qwen35:
        bridge = (
            DEFAULT_CAMPAIGN_ROOT
            / "qwen3p5_4b"
            / "scripts"
        )
        fixed_runtime = {
            "python": str(QWEN35_PYTHON),
            "torchrun": str(QWEN35_TORCHRUN),
            "llamafactory_source": str(
                QWEN35_SITE_PACKAGES / "llamafactory"
            ),
            "flash_attn": "fa2",
            "fa3_variant": None,
            "enable_cce": False,
            "enable_liger_kernel": True,
            "optimizer": "adamw_torch_fused",
            "torch_compile": False,
            "dataloader_num_workers": 0,
            "preprocessing_num_workers": 8,
            "seed": 20260729,
            "data_seed": 20260729,
            "ddp_timeout": 180000000,
            "environment_overlay": {
                "PYTHONPATH_prepend": [
                    str(bridge),
                    str(QWEN35_OVERLAY),
                ],
                "FLA_TILELANG": "1",
                "TILELANG_CACHE_DIR": (
                    "/tmp/qwen35_tilelang_cache_rtx4090"
                ),
            },
        }
    else:
        fixed_runtime = dict(
            read_json(OLD_4090_ROOT / "config" / "experiment.json")[
                "fixed_runtime"
            ]
        )
        fixed_runtime["seed"] = 20260729
        fixed_runtime["data_seed"] = 20260729
    return {
        "schema_version": 2,
        "campaign_id": f"{CAMPAIGN_ID}_{model_id}",
        "hardware_id": "local_rtx4090_24g_pcie",
        "training_scope": {
            "phase_id": CAMPAIGN_ID,
            "model_ids": [model_id],
            "gpu_ids": [0, 1, 2, 3],
            "max_gpu_count": 4,
            "exclusive_node_gpu_ids": [0, 1, 2, 3],
            "deferred_model_ids": [],
            "stage": "sft",
            "precision": "bf16",
            "gpu_type": "NVIDIA GeForce RTX 4090 24GB",
            "gpu_counts": [1, 2, 4],
            "global_batch_sizes": [64],
            "gradient_checkpointing": [False, True],
            "zero_by_gpu_count": {
                "1": ["none"],
                "2": ["zero2", "zero3"],
                "4": ["zero2", "zero3"],
            },
        },
        "fixed_runtime": fixed_runtime,
        "measurement": {
            "memory_probe_max_steps": 5,
            "throughput_warmup_steps": 3,
            "throughput_measure_steps": 10,
            "throughput_screen_warmup_steps": 0,
            "throughput_screen_measure_steps": 2,
            "scaling_warmup_steps": 2,
            "scaling_measure_steps": 8,
            "packing_warmup_steps": 2,
            "packing_measure_steps": 8,
            "packing_memory_probe_steps": 3,
            "profiler_warmup_steps": 1,
            "profiler_measure_steps": 3,
            "throughput_repeats": 1,
            "rerun_on_unhealthy_result": True,
            "nvidia_smi_interval_seconds": 1.0,
            "performance_parallelism": "disjoint_gpu_masks",
            "formal_throughput_requires_exclusive_node": False,
        },
        "datasets": [
            {
                "id": dataset_id,
                "target_cutoff": cutoff,
            }
            for dataset_id, cutoff in DATASETS.items()
        ],
    }


def _write_subcampaign(
    campaign_root: Path,
    *,
    model_id: str,
    jobs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = campaign_root / model_id
    source_scripts = (
        Path(__file__).resolve().parents[2]
        / "offline_experiments_qwen35_tilelang_20260729"
        / "scripts"
        if model_id == "qwen3p5_4b"
        else ROOT / "scripts"
    )
    shutil.copytree(
        source_scripts,
        root / "scripts",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(
        LIVE_APPROVAL_SOURCE,
        root / "freeze_live_approval.py",
    )
    data_link = root / "data"
    if data_link.is_symlink():
        if data_link.resolve() != (ROOT / "data").resolve():
            raise ValueError(f"Unexpected data symlink: {data_link}")
    elif data_link.exists():
        raise ValueError(f"Campaign data path is not a symlink: {data_link}")
    else:
        os.symlink(
            (ROOT / "data").resolve(),
            data_link,
            target_is_directory=True,
        )
    write_json(
        root / "config" / "hardware.json",
        read_json(OLD_4090_ROOT / "config" / "hardware.json"),
    )
    write_json(
        root / "config" / "models.json",
        {
            "schema_version": 1,
            "selection_policy": (
                "Prospective RTX 4090 generalization; frozen before run"
            ),
            "fixed_lora": {
                "rank": 32,
                "alpha": 32,
                "dropout": 0.0,
                "target": "all",
            },
            "models": [MODELS[model_id]],
        },
    )
    write_json(
        root / "config" / "experiment.json",
        _experiment_config(
            model_id=model_id,
            qwen35=model_id == "qwen3p5_4b",
        ),
    )
    for stage in (2, 3):
        source = OLD_4090_ROOT / "config" / "deepspeed" / f"ds_z{stage}.json"
        write_json(
            root / "config" / "deepspeed" / f"ds_z{stage}.json",
            read_json(source),
        )
    source_inventory = read_json(
        (
            Path(__file__).resolve().parents[2]
            / "offline_experiments_qwen35_tilelang_20260729"
            / "artifacts"
            / "model_inventory.json"
        )
        if model_id == "qwen3p5_4b"
        else ROOT / "artifacts" / "model_inventory.json"
    )
    selected_inventory = [
        model
        for model in source_inventory["models"]
        if str(model.get("id")) == model_id
    ]
    if len(selected_inventory) != 1:
        raise ValueError(f"Model inventory missing {model_id}")
    write_json(
        root / "artifacts" / "model_inventory.json",
        {
            **{
                key: value
                for key, value in source_inventory.items()
                if key != "models"
            },
            "models": selected_inventory,
        },
    )
    write_json(
        root / "artifacts" / "dataset_analysis.json",
        read_json(ROOT / "artifacts" / "dataset_analysis.json"),
    )
    write_json(
        root
        / "artifacts"
        / "frozen_predictions_before_holdout.json",
        read_json(
            campaign_root
            / "artifacts"
            / "frozen_predictions_before_holdout.json"
        ),
    )
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        by_phase.setdefault(str(job["phase_id"]), []).append(dict(job))
    queue_bindings = {}
    for phase, phase_jobs in sorted(by_phase.items()):
        path = root / "matrix" / f"queue_{phase}.jsonl"
        write_jsonl(path, phase_jobs)
        queue_bindings[phase] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "jobs": len(phase_jobs),
            "gpu_slots": sum(int(job["gpu_count"]) for job in phase_jobs),
        }
    combined = root / "matrix" / "queue_all.jsonl"
    write_jsonl(combined, [dict(job) for job in jobs])
    return {
        "root": str(root.resolve()),
        "model_id": model_id,
        "runtime": _experiment_config(
            model_id=model_id,
            qwen35=model_id == "qwen3p5_4b",
        )["fixed_runtime"],
        "queues": queue_bindings,
        "combined_queue": {
            "path": str(combined.resolve()),
            "sha256": sha256_file(combined),
            "jobs": len(jobs),
        },
        "approval_status": (
            "not_frozen_until_live_4090_provenance_is_captured"
        ),
    }


def build(campaign_root: Path, model_artifact: Path) -> dict[str, Any]:
    jobs = [
        *_canaries(),
        *_memory_boundaries(),
        *_ranking(),
        *_packing_abba(),
    ]
    _validate_jobs(jobs)
    requests = _unique_prediction_requests(jobs)
    predictor = RTX4090SafetyV2Predictor(
        model_artifact=model_artifact
    )
    prediction = predictor.predict(requests)
    prediction_path = (
        campaign_root
        / "artifacts"
        / "frozen_predictions_before_holdout.json"
    )
    write_json(prediction_path, prediction)

    subcampaigns = []
    for model_id in MODELS:
        subcampaigns.append(
            _write_subcampaign(
                campaign_root,
                model_id=model_id,
                jobs=[
                    job
                    for job in jobs
                    if job["model_id"] == model_id
                ],
            )
        )
    request_path = campaign_root / "matrix" / "prediction_requests.json"
    write_json(request_path, {"candidates": requests})
    all_jobs_path = campaign_root / "matrix" / "all_jobs.jsonl"
    write_jsonl(all_jobs_path, jobs)

    summary: dict[str, Any] = {
        "schema": "sft_rtx4090_prospective_generalization_freeze/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_waiting_for_live_rtx4090_preflight",
        "gpu_experiments_launched": False,
        "results_consumed_for_model_fit": False,
        "gpu_pool": [0, 1, 2, 3],
        "model_artifact": {
            "path": str(model_artifact.resolve()),
            "sha256": sha256_file(model_artifact),
            "report_sha256": read_json(model_artifact)["report_sha256"],
        },
        "prediction_freeze": {
            "path": str(prediction_path.resolve()),
            "sha256": sha256_file(prediction_path),
            "report_sha256": prediction["report_sha256"],
            "unique_candidates": len(requests),
        },
        "jobs": {
            "path": str(all_jobs_path.resolve()),
            "sha256": sha256_file(all_jobs_path),
            "count": len(jobs),
            "gpu_slots": sum(int(job["gpu_count"]) for job in jobs),
            "by_phase": dict(
                Counter(str(job["phase_id"]) for job in jobs)
            ),
            "by_model": dict(
                Counter(str(job["model_id"]) for job in jobs)
            ),
        },
        "subcampaigns": subcampaigns,
        "execution_dag": [
            {
                "stage": 1,
                "phase": "compatibility_canary",
                "gate": "both models must produce complete metrics",
            },
            {
                "stage": 2,
                "phase": "memory_boundary",
                "gate": (
                    "collect success/OOM labels; do not refit before replay"
                ),
            },
            {
                "stage": 3,
                "phase": "multi_candidate_ranking",
                "gate": (
                    "run only the already frozen candidate set; two repeats"
                ),
            },
            {
                "stage": 4,
                "phase": "packing_abba",
                "gate": (
                    "Qwen3-8B only; A-B-B-A order with 2+8 steps"
                ),
            },
        ],
        "parallelism_policy": {
            "scheduler": "disjoint GPU masks inside each subcampaign",
            "one_gpu_jobs": "up to four concurrent jobs",
            "two_gpu_jobs": "up to two concurrent jobs",
            "four_gpu_jobs": "exclusive four-card wave",
            "cross_runtime_policy": (
                "never run qwen3_8b and qwen3p5_4b schedulers "
                "concurrently; prevents GPU-allocation races"
            ),
            "process_policy": (
                "never terminate or attach to an unrelated process; wait "
                "fail-closed when GPUs 0-3 are occupied"
            ),
        },
        "runtime_gate": {
            "required_gpu_name": "NVIDIA GeForce RTX 4090",
            "required_gpu_count": 4,
            "qwen3_runtime": "/fine-tuning-launcher/.venv-4090",
            "qwen35_runtime": str(QWEN35_PYTHON.parent.parent),
            "qwen35_overlay": str(QWEN35_OVERLAY),
            "qwen35_overlay_manifest_sha256": QWEN35_OVERLAY_SHA256,
            "qwen35_overlay_manifest_files": QWEN35_OVERLAY_FILES,
            "approval_rule": (
                "capture live provenance and freeze approvals on the RTX "
                "4090 node; never reuse the historical campaign approval"
            ),
        },
        "acceptance_metrics_after_run": {
            "memory": [
                "false-safe OOM",
                "admitted success above 95% capacity",
                "safe-success admission recall",
                "center MAPE and absolute GiB error",
            ],
            "throughput": [
                "pairwise ranking accuracy",
                "top-1 hit",
                "hit@90%-of-best",
                "top-1 regret",
                "absolute throughput APE as secondary evidence",
            ],
        },
        "source_bindings": {
            "generator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "old_4090_hardware": {
                "path": str(
                    (
                        OLD_4090_ROOT / "config" / "hardware.json"
                    ).resolve()
                ),
                "sha256": sha256_file(
                    OLD_4090_ROOT / "config" / "hardware.json"
                ),
            },
        },
    }
    summary["freeze_sha256"] = sha256_json(summary)
    write_json(campaign_root / "FREEZE.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT,
    )
    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=MODEL_ARTIFACT,
    )
    args = parser.parse_args()
    report = build(
        args.campaign_root.resolve(),
        args.model_artifact.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
