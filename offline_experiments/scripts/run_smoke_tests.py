#!/usr/bin/env python3
"""Run four tightly bounded training smoke tests and record a freshness manifest."""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any

from common import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    DATA_DIR,
    RESULTS_DIR,
    ROOT,
    RUNTIME_DIR,
    ensure_gpus_idle,
    read_json,
    sha256_file,
    write_json,
)


def base_job(name: str) -> dict[str, Any]:
    experiment = read_json(CONFIG_DIR / "experiment.json")
    model = next(row for row in read_json(CONFIG_DIR / "models.json")["models"] if row["id"] == "qwen3_1p7b")
    return {
        "job_id": f"smoke-{name}-{int(time.time())}",
        "kind": "smoke",
        "campaign_id": experiment.get("campaign_id"),
        "phase_id": experiment["training_scope"]["phase_id"],
        "hardware_id": experiment.get("hardware_id"),
        "gpu_type": experiment["training_scope"]["gpu_type"],
        "model_id": model["id"],
        "model_path": model["path"],
        "tokenizer_path": model["tokenizer_path"],
        "model_family": model["family"],
        "template": model["template"],
        "dataset_id": "short_512",
        "cutoff_len": 512,
        "mbs": 1,
        "target_gbs": 16,
        "packing": False,
        "warmup_steps": 0,
        "measure_steps": 1,
        "repeat": 0,
        "parallel_class": "exclusive_node",
    }


def smoke_jobs() -> list[tuple[dict[str, Any], list[int]]]:
    gpu_ids = read_json(CONFIG_DIR / "experiment.json")["training_scope"]["gpu_ids"]
    attention = read_json(CONFIG_DIR / "experiment.json")["fixed_runtime"]["flash_attn"]
    single = {
        **base_job(f"single-lora-{attention}"),
        "train_type": "lora",
        "gpu_count": 1,
        "zero": "none",
        "gc": False,
    }

    analysis = read_json(ARTIFACT_DIR / "dataset_analysis.json")
    packing = analysis["datasets"]["short_512"]["profiles"]["qwen3_nothink"]["packing"]
    ga = next(
        row
        for row in packing["ga_table"]
        if row["data_parallel"] == 1 and row["target_gbs"] == single["target_gbs"]
    )
    packed = {
        **base_job("single-lora-neat-packing"),
        "train_type": "lora",
        "gpu_count": 1,
        "zero": "none",
        "gc": False,
        "packing": True,
        "gradient_accumulation_steps": ga["gradient_accumulation_steps"],
        "expected_sample_gbs": ga["expected_sample_gbs"],
    }
    two = {
        **base_job("two-full-z2-gc-on"),
        "train_type": "full",
        "gpu_count": 2,
        "zero": "zero2",
        "gc": True,
    }
    four = {
        **base_job("four-lora-z3-gc-off"),
        "train_type": "lora",
        "gpu_count": 4,
        "zero": "zero3",
        "gc": False,
    }
    return [(single, gpu_ids[:1]), (packed, gpu_ids[:1]), (two, gpu_ids[:2]), (four, gpu_ids[:4])]


def freshness_manifest() -> dict[str, str]:
    paths = [
        ROOT / "scripts" / name
        for name in (
            "common.py",
            "metrics_callback.py",
            "run_job.py",
            "run_smoke_tests.py",
            "train_entry.py",
        )
    ]
    paths += [
        CONFIG_DIR / "experiment.json",
        CONFIG_DIR / "models.json",
        CONFIG_DIR / "deepspeed" / "ds_z2.json",
        CONFIG_DIR / "deepspeed" / "ds_z3.json",
        DATA_DIR / "dataset_info.json",
        DATA_DIR / "derived" / "short_512.jsonl",
        ARTIFACT_DIR / "dataset_analysis.json",
        ARTIFACT_DIR / "model_inventory.json",
        ARTIFACT_DIR / "provenance.json",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def main() -> None:
    experiment = read_json(CONFIG_DIR / "experiment.json")
    ensure_gpus_idle(experiment["training_scope"]["gpu_ids"])
    python = str(experiment["fixed_runtime"]["python"])
    rows = []
    for job, gpu_mask in smoke_jobs():
        job_path = RUNTIME_DIR / "smoke_jobs" / f"{job['job_id']}.json"
        write_json(job_path, job)
        command = [
            python,
            str(ROOT / "scripts" / "run_job.py"),
            "--job-file",
            str(job_path),
            "--gpu-mask",
            ",".join(map(str, gpu_mask)),
            "--execute-smoke",
        ]
        print(f"Running {job['job_id']} on GPUs {gpu_mask}", flush=True)
        process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        status_path = RESULTS_DIR / job["job_id"] / "status.json"
        status = read_json(status_path) if status_path.is_file() else {"classification": "missing_status"}
        summaries = sorted((RESULTS_DIR / job["job_id"] / "metrics").glob("summary.rank*.json"))
        summary_checks = []
        for path in summaries:
            summary = read_json(path)
            summary_checks.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "failure": summary.get("failure"),
                    "total_steps": summary.get("total_steps"),
                    "measured_steps": summary.get("measured_steps"),
                    "max_allocated": summary.get("max_allocated"),
                }
            )
        passed = (
            process.returncode == 0
            and status.get("classification") == "success"
            and len(summaries) == job["gpu_count"]
            and all(row["failure"] is None and int(row["measured_steps"] or 0) >= 1 for row in summary_checks)
        )
        rows.append(
            {
                "job_id": job["job_id"],
                "gpu_mask": gpu_mask,
                "train_type": job["train_type"],
                "zero": job["zero"],
                "gc": job["gc"],
                "packing": job["packing"],
                "return_code": process.returncode,
                "classification": status.get("classification"),
                "summaries": summary_checks,
                "launcher_output_tail": (process.stdout + process.stderr)[-4000:],
                "passed": passed,
            }
        )
        if not passed:
            print(json.dumps(rows[-1], ensure_ascii=False, indent=2), flush=True)
            break

    report = {
        "schema_version": 1,
        "scope": "Qwen3-1.7B, short_512, one optimizer step per case",
        "freshness_manifest": freshness_manifest(),
        "checks": rows,
        "all_passed": len(rows) == 4 and all(row["passed"] for row in rows),
    }
    write_json(ARTIFACT_DIR / "smoke_validation.json", report)
    print(json.dumps({"checks": len(rows), "all_passed": report["all_passed"]}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["all_passed"] else 2)


if __name__ == "__main__":
    main()
