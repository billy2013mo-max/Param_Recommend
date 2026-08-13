#!/usr/bin/env python3
"""Prove static lengths and packing match the installed LLaMA-Factory processors."""

from __future__ import annotations

import contextlib
import atexit
import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Any

import datasets

from common import ARTIFACT_DIR, CONFIG_DIR, DATA_DIR, RUNTIME_DIR, read_json, sha256_file, write_json


def ensure_torchrun() -> None:
    """Re-launch once under torchrun, as required by the installed LLaMA Factory parser."""
    if "LOCAL_RANK" in os.environ:
        def cleanup_process_group() -> None:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()

        atexit.register(cleanup_process_group)
        return
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=1",
            str(Path(__file__).resolve()),
        ],
    )


def profile_id(model: dict[str, Any]) -> str:
    return "qwen3_nothink"


def processor_run(model: dict[str, Any], dataset_id: str, cutoff: int, packing: bool) -> Any:
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    experiment = read_json(CONFIG_DIR / "experiment.json")
    preprocessing_workers = int(experiment["fixed_runtime"]["preprocessing_num_workers"])
    config = {
        "model_name_or_path": model["path"],
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "dataset": dataset_id,
        "dataset_dir": str(DATA_DIR),
        "template": model["template"],
        "cutoff_len": cutoff,
        "max_samples": 1000,
        "preprocessing_batch_size": 1000,
        "preprocessing_num_workers": preprocessing_workers,
        "overwrite_cache": True,
        "packing": packing,
        "neat_packing": packing,
        "output_dir": str(RUNTIME_DIR / "preprocessing_validation"),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-5,
        "max_steps": 1,
        "bf16": True,
        "report_to": "none",
        "disable_tqdm": True,
    }
    model_args, data_args, training_args, _, _ = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
        module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    return module["train_dataset"]


def main() -> None:
    ensure_torchrun()
    datasets.disable_progress_bars()
    analysis = read_json(ARTIFACT_DIR / "dataset_analysis.json")
    catalog = read_json(CONFIG_DIR / "models.json")
    enabled_ids = read_json(CONFIG_DIR / "experiment.json")["training_scope"]["model_ids"]
    representative_id = "qwen3_8b" if "qwen3_8b" in enabled_ids else enabled_ids[min(1, len(enabled_ids) - 1)]
    representative_models = [next(model for model in catalog["models"] if model["id"] == representative_id)]
    if any(model["template"] != "qwen3_nothink" for model in catalog["models"]):
        raise RuntimeError("Every model must use the unified qwen3_nothink template")
    checks = []
    for model in representative_models:
        pid = profile_id(model)
        for dataset_id, dataset_report in analysis["datasets"].items():
            expected = dataset_report["profiles"][pid]
            cutoff = expected["lengths"]["cutoff_len"]
            print(f"Validating {dataset_id}/{pid} no-packing", flush=True)
            plain = processor_run(model, dataset_id, cutoff, packing=False)
            actual_lengths = sorted(len(row["input_ids"]) for row in plain)
            profile_path = ARTIFACT_DIR / "dataset_profiles" / f"{dataset_id}.{pid}.jsonl"
            expected_lengths = sorted(
                __import__("json").loads(line)["total_tokens"] for line in profile_path.open(encoding="utf-8") if line.strip()
            )

            print(f"Validating {dataset_id}/{pid} neat-packing", flush=True)
            packed = processor_run(model, dataset_id, cutoff, packing=True)
            pack_counts = []
            packed_lengths_ok = True
            for row in packed:
                boundaries = row["packing_params"]["sequence_boundaries"]
                pack_counts.append(max(1, len(boundaries) - 2))
                packed_lengths_ok = packed_lengths_ok and len(row["input_ids"]) == cutoff
            check = {
                "dataset_id": dataset_id,
                "profile_id": pid,
                "no_packing_rows": len(plain),
                "lengths_exactly_match": actual_lengths == expected_lengths,
                "actual_max_tokens": max(actual_lengths),
                "expected_max_tokens": expected["lengths"]["max_tokens"],
                "packed_rows": len(packed),
                "expected_packs": expected["packing"]["packs"],
                "pack_count_matches": len(packed) == expected["packing"]["packs"],
                "logical_samples_in_packs": sum(pack_counts),
                "all_packed_processor_rows_equal_configured_cutoff": packed_lengths_ok,
            }
            check["passed"] = (
                check["no_packing_rows"] == 1000
                and check["lengths_exactly_match"]
                and check["pack_count_matches"]
                and check["logical_samples_in_packs"] == 1000
                and packed_lengths_ok
            )
            checks.append(check)
            if not check["passed"]:
                raise RuntimeError(f"Processor mismatch: {check}")
    report = {
        "schema_version": 1,
        "dataset_analysis_sha256": sha256_file(ARTIFACT_DIR / "dataset_analysis.json"),
        "models_config_sha256": sha256_file(CONFIG_DIR / "models.json"),
        "dataset_info_sha256": sha256_file(DATA_DIR / "dataset_info.json"),
        "llamafactory_version": importlib.metadata.version("llamafactory"),
        "preprocessing_num_workers": int(
            read_json(CONFIG_DIR / "experiment.json")["fixed_runtime"]["preprocessing_num_workers"]
        ),
        "tokenizer_policy": "Qwen3 tokenizer and qwen3_nothink template for every experiment",
        "representative_model_ids": [model["id"] for model in representative_models],
        "checks": checks,
        "all_passed": all(row["passed"] for row in checks),
    }
    write_json(ARTIFACT_DIR / "preprocessing_validation.json", report)
    print(f"Validated {len(checks)} tokenizer/dataset combinations against installed processors")


if __name__ == "__main__":
    main()
