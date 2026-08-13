#!/usr/bin/env python3
"""Validate all 15 frozen profiles against installed LLaMA-Factory preprocessing."""

from __future__ import annotations

import argparse
import atexit
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import datasets

from common import ARTIFACT_DIR, DATA_DIR, RUNTIME_DIR, read_json, sha256_file, sha256_json, write_json


DEFAULT_BUNDLE = ARTIFACT_DIR / "h800_lora_source_disjoint_bundle_v1.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_lora_source_disjoint_preprocessing_validation_v1.json"


def ensure_torchrun() -> None:
    if "LOCAL_RANK" in os.environ:
        def cleanup() -> None:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        atexit.register(cleanup)
        return
    os.execv(
        sys.executable,
        [
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc-per-node=1", str(Path(__file__).resolve()), *sys.argv[1:],
        ],
    )


def processor_run(scenario: dict[str, Any]) -> Any:
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    config = {
        "model_name_or_path": scenario["model_path"],
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "dataset": scenario["dataset_id"],
        "dataset_dir": str(DATA_DIR),
        "template": "qwen3_nothink",
        "cutoff_len": int(scenario["cutoff_len"]),
        "max_samples": int(scenario["profile_statistics"]["rows"]),
        "preprocessing_batch_size": 1000,
        # ``None`` keeps Hugging Face Dataset.map in-process.  The production
        # run uses eight workers, but the transform itself is identical and
        # the restricted validator cannot open the multiprocessing manager
        # socket.
        "preprocessing_num_workers": None,
        "overwrite_cache": True,
        "packing": False,
        "output_dir": str(RUNTIME_DIR / "h800_lora_source_disjoint_preprocessing_validation"),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-5,
        "max_steps": 1,
        # This validator only exercises dataset preprocessing.  Keep the
        # parser on CPU so it remains usable inside the restricted sandbox;
        # the tokenizer/template/cutoff path is identical to the GPU run.
        "bf16": False,
        "fp16": False,
        "use_cpu": True,
        "report_to": "none",
        "disable_tqdm": True,
    }
    model_args, data_args, training_args, _, _ = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
        module = get_dataset(
            template, model_args, data_args, training_args, stage="sft", **tokenizer_module
        )
    return module["train_dataset"]


def main() -> None:
    ensure_torchrun()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    datasets.disable_progress_bars()
    bundle = read_json(args.bundle)
    checks = []
    for index, scenario in enumerate(bundle["scenarios"], start=1):
        print(
            f"[{index:02d}/15] validating {scenario['dataset_id']} "
            f"with {scenario['model_id']} cutoff={scenario['cutoff_len']}",
            flush=True,
        )
        processed = processor_run(scenario)
        actual_lengths = sorted(len(row["input_ids"]) for row in processed)
        profile_rows = [
            json.loads(line)
            for line in Path(scenario["profile_path"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cutoff = int(scenario["cutoff_len"])
        expected_lengths = sorted(min(int(row["total_tokens"]), cutoff) for row in profile_rows)
        check = {
            "scenario_id": scenario["scenario_id"],
            "dataset_id": scenario["dataset_id"],
            "model_id": scenario["model_id"],
            "cutoff_len": cutoff,
            "profile_rows": len(profile_rows),
            "processor_rows": len(processed),
            "expected_max_after_cutoff": max(expected_lengths),
            "actual_max_after_cutoff": max(actual_lengths),
            "length_multiset_sha256": sha256_json(actual_lengths),
            "lengths_exactly_match": actual_lengths == expected_lengths,
        }
        check["passed"] = (
            check["profile_rows"] == check["processor_rows"]
            and check["lengths_exactly_match"]
        )
        checks.append(check)
        if not check["passed"]:
            raise RuntimeError(f"installed processor mismatch: {check}")
    report = {
        "schema": "sft_h800_lora_source_disjoint_preprocessing_validation/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "bundle": {"path": str(args.bundle.resolve()), "sha256": sha256_file(args.bundle)},
        "dataset_info": {
            "path": str((DATA_DIR / "dataset_info.json").resolve()),
            "sha256": sha256_file(DATA_DIR / "dataset_info.json"),
        },
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks),
    }
    report["report_sha256"] = sha256_json(report)
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "checks": len(checks),
                "all_passed": report["all_passed"],
                "report_sha256": report["report_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
