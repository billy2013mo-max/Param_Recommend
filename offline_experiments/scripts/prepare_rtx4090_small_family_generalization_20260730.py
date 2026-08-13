#!/usr/bin/env python3
"""Freeze a compact RTX 4090 Qwen2.5/Qwen3.5 generalization campaign.

The campaign is prospective: all memory and throughput predictions are
materialized before a GPU job is eligible to run.  Qwen2.5 uses the standard
RTX 4090 FA2 runtime.  Qwen3.5 uses the already validated
Transformers/FLA/TileLang compatibility runtime and imports the ABI-matched
FA2 package from the standard RTX 4090 environment.
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

from common import (
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from inventory_models import inventory_model
from rtx4090_safety_v2_predictor import RTX4090SafetyV2Predictor


CAMPAIGN_ID = "rtx4090_small_family_generalization_20260730"
DEFAULT_CAMPAIGN_ROOT = ROOT / "campaigns" / CAMPAIGN_ID
OLD_4090_ROOT = ROOT / "campaigns" / "rtx4090_20260717"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
QWEN35_SOURCE = (
    PROJECT_ROOT / "offline_experiments_qwen35_tilelang_20260729"
)
QWEN35_OVERLAY = Path("/tmp/qwen35_tilelang_overlay_0.1.12")
QWEN35_OVERLAY_SHA256 = (
    "8496316be58be29f20a8dbf91a34279b79afeb13e5769ff5c3508e0587d04f09"
)
QWEN35_OVERLAY_FILES = 7064
QWEN35_PYTHON = PROJECT_ROOT / "qwen36_venv" / "bin" / "python"
QWEN35_TORCHRUN = PROJECT_ROOT / "qwen36_venv" / "bin" / "torchrun"
QWEN35_SITE_PACKAGES = (
    PROJECT_ROOT
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
MODEL_ARTIFACT = (
    OLD_4090_ROOT
    / "artifacts"
    / "rtx4090_physical_shares_v4b_safety_v2_2026-07-29.json"
)
DATASETS = {
    "short_512": 512,
    "multiturn_4096": 4096,
    "longtail_8192": 8192,
}
MODELS: dict[str, dict[str, Any]] = {
    "qwen2p5_1p5b": {
        "id": "qwen2p5_1p5b",
        "nominal_scale_b": 1.5,
        "path": "/wanqing-models/Qwen2.5-1.5B",
        "tokenizer_path": "/wanqing-models/Qwen2.5-1.5B",
        "family": "qwen2.5",
        "template": "qwen",
        "train_types": ["full", "lora"],
        "actual_parameters": 1_543_714_304,
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": 151936,
        "generalization_role": "unseen_qwen2_dense_family_and_tokenizer",
    },
    "qwen3p5_0p8b": {
        "id": "qwen3p5_0p8b",
        "nominal_scale_b": 0.8,
        "path": "/wanqing-models/Qwen3.5-0.8B",
        "tokenizer_path": "/wanqing-models/Qwen3.5-0.8B",
        "family": "qwen3_5",
        "template": "qwen3_5_nothink",
        "train_types": ["full", "lora"],
        "actual_parameters": 873_438_784,
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 248320,
        "generalization_role": "unseen_hybrid_attention_small_scale",
    },
    "qwen3p5_4b": {
        "id": "qwen3p5_4b",
        "nominal_scale_b": 4,
        "path": "/wanqing-models/Qwen3.5-4B",
        "tokenizer_path": "/wanqing-models/Qwen3.5-4B",
        "family": "qwen3_5",
        "template": "qwen3_5_nothink",
        "train_types": ["full", "lora"],
        "actual_parameters": 4_659_865_088,
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "vocab_size": 248320,
        "generalization_role": "unseen_hybrid_attention_product_boundary",
    },
}


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


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
    train_type: str,
    gpu_count: int,
    mbs: int,
    gc: bool,
    evaluation_group: str,
    candidate_label: str,
    zero: str | None = None,
    repeat: int = 0,
    warmup_steps: int | None = None,
    measure_steps: int | None = None,
) -> dict[str, Any]:
    model = MODELS[model_id]
    row: dict[str, Any] = {
        "schema": "sft_rtx4090_small_family_generalization_job/v1",
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
        "train_type": train_type,
        "dataset_id": dataset_id,
        "cutoff_len": DATASETS[dataset_id],
        "gpu_count": gpu_count,
        "zero": _zero(gpu_count, zero),
        "gc": gc,
        "mbs": mbs,
        "target_gbs": 64,
        "packing": False,
        "repeat": repeat,
        "parallel_class": "gpu_partitionable",
        "requires_external_node_idle": False,
    }
    if warmup_steps is not None:
        row["warmup_steps"] = warmup_steps
    if measure_steps is not None:
        row["measure_steps"] = measure_steps
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
    rows = []
    full_layout = {
        "qwen2p5_1p5b": (2, "zero3"),
        "qwen3p5_0p8b": (2, "zero3"),
        "qwen3p5_4b": (4, "zero3"),
    }
    for model_id in MODELS:
        rows.append(
            _job(
                prefix="r4090sfcanary",
                model_id=model_id,
                phase="compatibility_canary",
                kind="throughput_screen",
                fidelity="canary_0plus2",
                dataset_id="short_512",
                train_type="lora",
                gpu_count=1,
                mbs=1,
                gc=True,
                evaluation_group=f"{model_id}-lora-canary",
                candidate_label="lora_1g_gc_m1",
                warmup_steps=0,
                measure_steps=2,
            )
        )
        gpu_count, zero = full_layout[model_id]
        rows.append(
            _job(
                prefix="r4090sfcanary",
                model_id=model_id,
                phase="compatibility_canary",
                kind="throughput_screen",
                fidelity="canary_0plus2",
                dataset_id="short_512",
                train_type="full",
                gpu_count=gpu_count,
                zero=zero,
                mbs=1,
                gc=True,
                evaluation_group=f"{model_id}-full-canary",
                candidate_label=f"full_{gpu_count}g_z3_gc_m1",
                warmup_steps=0,
                measure_steps=2,
            )
        )
    return rows


def _memory_boundaries() -> list[dict[str, Any]]:
    specs = (
        ("qwen2p5_1p5b", 2, "8k_1g_m2"),
        ("qwen2p5_1p5b", 8, "8k_1g_m8"),
        ("qwen3p5_0p8b", 4, "8k_1g_m4"),
        ("qwen3p5_0p8b", 16, "8k_1g_m16"),
        ("qwen3p5_4b", 1, "8k_1g_m1"),
        ("qwen3p5_4b", 8, "8k_1g_m8"),
    )
    return [
        _job(
            prefix="r4090sfmem",
            model_id=model_id,
            phase="memory_boundary",
            kind="memory_probe",
            fidelity="memory_5step",
            dataset_id="longtail_8192",
            train_type="lora",
            gpu_count=1,
            mbs=mbs,
            gc=True,
            evaluation_group=f"{model_id}-memory-boundary",
            candidate_label=label,
        )
        for model_id, mbs, label in specs
    ]


def _ranking() -> list[dict[str, Any]]:
    scenarios: dict[str, Sequence[tuple[int, int, bool, str | None, str]]] = {
        "qwen2p5_1p5b": (
            (1, 4, True, None, "1g_gc_m4"),
            (1, 1, False, None, "1g_nogc_m1"),
            (2, 4, True, "zero2", "2g_z2_gc_m4"),
            (4, 4, True, "zero2", "4g_z2_gc_m4"),
        ),
        "qwen3p5_0p8b": (
            (1, 8, True, None, "1g_gc_m8"),
            (1, 4, False, None, "1g_nogc_m4"),
            (2, 8, True, "zero2", "2g_z2_gc_m8"),
            (4, 8, True, "zero2", "4g_z2_gc_m8"),
        ),
        "qwen3p5_4b": (
            (1, 2, True, None, "1g_gc_m2"),
            (1, 1, False, None, "1g_nogc_m1"),
            (2, 2, True, "zero2", "2g_z2_gc_m2"),
            (4, 2, True, "zero2", "4g_z2_gc_m2"),
        ),
    }
    rows = []
    for model_id, candidates in scenarios.items():
        for repeat in (0, 1):
            for gpu_count, mbs, gc, zero, label in candidates:
                rows.append(
                    _job(
                        prefix="r4090sfrank",
                        model_id=model_id,
                        phase="multi_candidate_ranking",
                        kind="throughput",
                        fidelity="formal_3plus10",
                        dataset_id="multiturn_4096",
                        train_type="lora",
                        gpu_count=gpu_count,
                        mbs=mbs,
                        gc=gc,
                        zero=zero,
                        repeat=repeat,
                        evaluation_group=f"{model_id}-lora-ranking",
                        candidate_label=label,
                        warmup_steps=3,
                        measure_steps=10,
                    )
                )
    return rows


def _validate_jobs(rows: Sequence[Mapping[str, Any]]) -> None:
    ids = [str(row["job_id"]) for row in rows]
    if len(rows) != 36 or len(ids) != len(set(ids)):
        raise ValueError("Expected exactly 36 unique prospective jobs")
    expected = {
        "compatibility_canary": 6,
        "memory_boundary": 6,
        "multi_candidate_ranking": 24,
    }
    counts = Counter(str(row["phase_id"]) for row in rows)
    if dict(counts) != expected:
        raise ValueError(f"Prospective phase counts drifted: {counts}")
    for row in rows:
        gpu_count = int(row["gpu_count"])
        mbs = int(row["mbs"])
        if bool(row["packing"]):
            raise ValueError("Packing is outside this holdout")
        if 64 % (gpu_count * mbs):
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
        "packing": False,
    }


def _unique_prediction_requests(
    jobs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    requests: dict[str, dict[str, Any]] = {}
    for job in jobs:
        request = _prediction_request(job)
        request_id = str(request["request_id"])
        previous = requests.get(request_id)
        if previous is not None and previous != request:
            raise ValueError(f"Prediction ID collision: {request_id}")
        requests[request_id] = request
    return list(requests.values())


def _fixed_runtime(model_id: str, root: Path) -> dict[str, Any]:
    if not model_id.startswith("qwen3p5"):
        runtime = dict(
            read_json(OLD_4090_ROOT / "config" / "experiment.json")[
                "fixed_runtime"
            ]
        )
        runtime["seed"] = 20260730
        runtime["data_seed"] = 20260730
        return runtime
    return {
        "python": str(QWEN35_PYTHON),
        "torchrun": str(QWEN35_TORCHRUN),
        "llamafactory_source": str(QWEN35_SITE_PACKAGES / "llamafactory"),
        "flash_attn": "fa2",
        "fa3_variant": None,
        "enable_cce": False,
        "enable_liger_kernel": True,
        "optimizer": "adamw_torch_fused",
        "torch_compile": False,
        "dataloader_num_workers": 0,
        "preprocessing_num_workers": 8,
        "seed": 20260730,
        "data_seed": 20260730,
        "ddp_timeout": 180000000,
        "environment_overlay": {
            "PYTHONPATH_prepend": [
                str(root / "scripts"),
                str(QWEN35_OVERLAY),
            ],
            "FLA_TILELANG": "1",
            "TILELANG_CACHE_DIR": (
                "/tmp/qwen35_tilelang_cache_rtx4090_small_family"
            ),
        },
    }


def _experiment_config(model_id: str, root: Path) -> dict[str, Any]:
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
        "fixed_runtime": _fixed_runtime(model_id, root),
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
            {"id": dataset_id, "target_cutoff": cutoff}
            for dataset_id, cutoff in DATASETS.items()
        ],
    }


def _bridge_text() -> str:
    return f'''"""Append the ABI-matched RTX 4090 FA2 environment."""

from __future__ import annotations

import sys

RTX4090_SITE_PACKAGES = {str(RTX4090_SITE_PACKAGES)!r}
if RTX4090_SITE_PACKAGES not in sys.path:
    sys.path.append(RTX4090_SITE_PACKAGES)
'''


def _write_subcampaign(
    campaign_root: Path,
    model_id: str,
    jobs: Sequence[Mapping[str, Any]],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    root = campaign_root / model_id
    source_scripts = (
        QWEN35_SOURCE / "scripts"
        if model_id.startswith("qwen3p5")
        else ROOT / "scripts"
    )
    shutil.copytree(
        source_scripts,
        root / "scripts",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(LIVE_APPROVAL_SOURCE, root / "freeze_live_approval.py")
    if model_id.startswith("qwen3p5"):
        _write_text(root / "scripts" / "sitecustomize.py", _bridge_text())
    data_link = root / "data"
    if data_link.is_symlink():
        if data_link.resolve() != (ROOT / "data").resolve():
            raise ValueError(f"Unexpected data symlink: {data_link}")
    elif data_link.exists():
        raise ValueError(f"Campaign data path is not a symlink: {data_link}")
    else:
        data_link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink((ROOT / "data").resolve(), data_link, target_is_directory=True)

    model_config = {
        "schema_version": 1,
        "selection_policy": (
            "Prospective RTX 4090 cross-family generalization; frozen before run"
        ),
        "fixed_lora": {
            "rank": 32,
            "alpha": 32,
            "dropout": 0.0,
            "target": "all",
        },
        "models": [MODELS[model_id]],
    }
    write_json(root / "config" / "models.json", model_config)
    write_json(
        root / "config" / "experiment.json",
        _experiment_config(model_id, root),
    )
    write_json(
        root / "config" / "hardware.json",
        read_json(OLD_4090_ROOT / "config" / "hardware.json"),
    )
    for stage in (2, 3):
        source = (
            OLD_4090_ROOT
            / "config"
            / "deepspeed"
            / f"ds_z{stage}.json"
        )
        write_json(
            root / "config" / "deepspeed" / f"ds_z{stage}.json",
            read_json(source),
        )
    inventoried = inventory_model(MODELS[model_id])
    write_json(
        root / "artifacts" / "model_inventory.json",
        {
            "schema_version": 1,
            "catalog_sha256": sha256_file(root / "config" / "models.json"),
            "selection_policy": model_config["selection_policy"],
            "fixed_lora": model_config["fixed_lora"],
            "models": [inventoried],
        },
    )
    write_json(
        root / "artifacts" / "dataset_analysis.json",
        read_json(ROOT / "artifacts" / "dataset_analysis.json"),
    )
    write_json(
        root / "artifacts" / "frozen_predictions_before_holdout.json",
        dict(prediction),
    )
    design = (
        "# Experiment design\n\n"
        f"`{model_id}` is a prospective RTX 4090 holdout. The queue contains "
        "FULL and LoRA compatibility canaries, a two-point long-context LoRA "
        "memory bracket, and two repeats of a four-candidate 1/2/4-GPU LoRA "
        "ranking scenario. Packing is fixed off. Results must not be used to "
        "refit the predictor until the frozen replay report is complete.\n"
    )
    _write_text(root / "EXPERIMENT_DESIGN.md", design)

    by_phase: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        by_phase.setdefault(str(job["phase_id"]), []).append(dict(job))
    bindings = {}
    for phase, phase_jobs in sorted(by_phase.items()):
        path = root / "matrix" / f"queue_{phase}.jsonl"
        write_jsonl(path, phase_jobs)
        bindings[phase] = {
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
        "runtime": _fixed_runtime(model_id, root),
        "queues": bindings,
        "combined_queue": {
            "path": str(combined.resolve()),
            "sha256": sha256_file(combined),
            "jobs": len(jobs),
        },
        "approval_status": "requires_live_4090_provenance",
    }


def build(campaign_root: Path, model_artifact: Path) -> dict[str, Any]:
    jobs = [*_canaries(), *_memory_boundaries(), *_ranking()]
    _validate_jobs(jobs)
    requests = _unique_prediction_requests(jobs)
    prediction = RTX4090SafetyV2Predictor(
        model_artifact=model_artifact
    ).predict(requests)
    prediction_path = (
        campaign_root
        / "artifacts"
        / "frozen_predictions_before_holdout.json"
    )
    write_json(prediction_path, prediction)
    subcampaigns = [
        _write_subcampaign(
            campaign_root,
            model_id,
            [job for job in jobs if job["model_id"] == model_id],
            prediction,
        )
        for model_id in MODELS
    ]
    requests_path = campaign_root / "matrix" / "prediction_requests.json"
    write_json(requests_path, {"candidates": requests})
    jobs_path = campaign_root / "matrix" / "all_jobs.jsonl"
    write_jsonl(jobs_path, jobs)
    _write_text(
        campaign_root / "README.md",
        (
            "# RTX 4090 Qwen2.5/Qwen3.5 prospective generalization\n\n"
            "This campaign is frozen before any GPU result is observed. It "
            "contains Qwen2.5-1.5B, Qwen3.5-0.8B and Qwen3.5-4B, limited to "
            "1/2/4 RTX 4090 GPUs. FULL is covered by compatibility canaries; "
            "the memory and ranking evaluation uses LoRA. Packing is off.\n\n"
            "Execution fails closed unless GPU 0-3 are four idle RTX 4090s "
            "and both fixed runtimes pass their import probes.\n"
        ),
    )
    summary: dict[str, Any] = {
        "schema": "sft_rtx4090_small_family_generalization_freeze/v1",
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
            "path": str(jobs_path.resolve()),
            "sha256": sha256_file(jobs_path),
            "count": len(jobs),
            "gpu_slots": sum(int(job["gpu_count"]) for job in jobs),
            "by_phase": dict(Counter(str(job["phase_id"]) for job in jobs)),
            "by_model": dict(Counter(str(job["model_id"]) for job in jobs)),
            "by_train_type": dict(
                Counter(str(job["train_type"]) for job in jobs)
            ),
        },
        "subcampaigns": subcampaigns,
        "execution_dag": [
            {
                "stage": 1,
                "phase": "compatibility_canary",
                "gate": "all FULL/LoRA canaries must produce complete metrics",
            },
            {
                "stage": 2,
                "phase": "memory_boundary",
                "gate": "collect labels without refitting",
            },
            {
                "stage": 3,
                "phase": "multi_candidate_ranking",
                "gate": "run the frozen candidates with two repeats",
            },
        ],
        "parallelism_policy": {
            "scheduler": "disjoint GPU masks within one runtime",
            "one_gpu_jobs": "up to four concurrent jobs",
            "two_gpu_jobs": "up to two concurrent jobs",
            "four_gpu_jobs": "exclusive four-card wave",
            "cross_runtime_policy": "never overlap subcampaign schedulers",
            "process_policy": "never signal or take over unrelated processes",
        },
        "acceptance_metrics_after_run": {
            "memory": [
                "false-safe OOM or above-95%-capacity rate = 0",
                "safe-success admission recall",
                "center MAPE and absolute GiB error",
            ],
            "throughput": [
                "pairwise ranking accuracy",
                "top-1 hit",
                "hit@90%-of-best",
                "top-1 regret",
            ],
        },
        "runtime_gate": {
            "required_gpu_name": "NVIDIA GeForce RTX 4090",
            "required_gpu_count": 4,
            "qwen25_runtime": "/fine-tuning-launcher/.venv-4090",
            "qwen35_runtime": str(QWEN35_PYTHON.parent.parent),
            "qwen35_overlay": str(QWEN35_OVERLAY),
            "qwen35_overlay_manifest_sha256": QWEN35_OVERLAY_SHA256,
            "qwen35_overlay_manifest_files": QWEN35_OVERLAY_FILES,
        },
        "source_bindings": {
            "generator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "hardware": {
                "path": str(
                    (OLD_4090_ROOT / "config" / "hardware.json").resolve()
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
