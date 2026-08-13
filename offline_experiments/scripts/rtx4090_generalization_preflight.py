#!/usr/bin/env python3
"""Fail-closed preflight for the frozen RTX 4090 generalization campaign."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

from common import ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_rtx4090_generalization_20260729 import (
    DEFAULT_CAMPAIGN_ROOT,
    QWEN35_OVERLAY,
    QWEN35_OVERLAY_FILES,
    QWEN35_OVERLAY_SHA256,
    QWEN35_PYTHON,
    QWEN35_TORCHRUN,
)


SCHEMA = "sft_rtx4090_generalization_preflight/v1"


def _overlay_manifest(root: Path) -> tuple[str, int]:
    lines: list[bytes] = []
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix not in {".pyc", ".pyo"}
        and "__pycache__" not in path.relative_to(root).parts
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        lines.append(
            f"{sha256_file(path)}  ./{relative}\n".encode()
        )
    return hashlib.sha256(b"".join(lines)).hexdigest(), len(files)


def _command(command: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
            env=env,
        )
        return {
            "command": command,
            "return_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "passed": result.returncode == 0,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "command": command,
            "return_code": None,
            "stdout": "",
            "stderr": repr(error),
            "passed": False,
        }


def _hardware() -> dict[str, Any]:
    gpu_query = _command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    if gpu_query["passed"]:
        for line in gpu_query["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",", 3)]
            if len(parts) == 4:
                rows.append(
                    {
                        "index": int(parts[0]),
                        "uuid": parts[1],
                        "name": parts[2],
                        "memory_total_mib": float(parts[3]),
                    }
                )
    selected = [row for row in rows if row["index"] in {0, 1, 2, 3}]
    exact_pool = (
        [row["index"] for row in selected] == [0, 1, 2, 3]
        and all("RTX 4090" in row["name"] for row in selected)
    )

    process_query = _command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    selected_uuids = {row["uuid"] for row in selected}
    processes = []
    if process_query["passed"]:
        for line in process_query["stdout"].splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",", 2)]
            if len(parts) == 3 and parts[0] in selected_uuids:
                processes.append(
                    {
                        "gpu_uuid": parts[0],
                        "pid": int(parts[1]),
                        "process_name": parts[2],
                    }
                )
    return {
        "gpu_query": gpu_query,
        "process_query": process_query,
        "selected_gpu_rows": selected,
        "exact_four_card_rtx4090_pool": exact_pool,
        "selected_gpu_compute_processes": processes,
        "selected_pool_idle": exact_pool and not processes,
    }


def _runtime(
    campaign_root: Path,
) -> dict[str, Any]:
    q3_root = campaign_root / "qwen3_8b"
    q35_root = campaign_root / "qwen3p5_4b"
    q3_config = read_json(q3_root / "config" / "experiment.json")[
        "fixed_runtime"
    ]
    q35_config = read_json(q35_root / "config" / "experiment.json")[
        "fixed_runtime"
    ]
    q3_python = Path(q3_config["python"])
    q3_torchrun = Path(q3_config["torchrun"])
    q35_python = Path(q35_config["python"])
    q35_torchrun = Path(q35_config["torchrun"])
    q35_overlay = q35_config["environment_overlay"]
    bridge = q35_root / "scripts" / "sitecustomize.py"
    fa2_package = (
        Path("/fine-tuning-launcher/.venv-4090/lib/python3.11/site-packages")
        / "flash_attn"
    )
    if QWEN35_OVERLAY.is_dir():
        overlay_sha256, overlay_files = _overlay_manifest(QWEN35_OVERLAY)
    else:
        overlay_sha256, overlay_files = None, 0
    static = {
        "qwen3_python": q3_python.is_file(),
        "qwen3_torchrun": q3_torchrun.is_file(),
        "qwen35_python": q35_python.is_file(),
        "qwen35_torchrun": q35_torchrun.is_file(),
        "qwen35_tilelang_overlay": (
            QWEN35_OVERLAY / "tilelang"
        ).is_dir(),
        "qwen35_overlay_manifest": (
            overlay_sha256 == QWEN35_OVERLAY_SHA256
            and overlay_files == QWEN35_OVERLAY_FILES
        ),
        "qwen35_site_bridge": bridge.is_file(),
        "rtx4090_fa2_package": fa2_package.is_dir(),
    }
    q3_probe = None
    if static["qwen3_python"]:
        q3_probe = _command(
            [
                str(q3_python),
                "-c",
                (
                    "import torch,transformers,flash_attn;"
                    "print(torch.__version__,transformers.__version__,"
                    "flash_attn.__version__)"
                ),
            ]
        )
    q35_probe = None
    if (
        static["qwen35_python"]
        and static["qwen35_tilelang_overlay"]
        and static["qwen35_site_bridge"]
        and static["rtx4090_fa2_package"]
    ):
        env = dict(os.environ)
        prefixes = list(q35_overlay["PYTHONPATH_prepend"])
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            [*prefixes, *([existing] if existing else [])]
        )
        env["FLA_TILELANG"] = str(q35_overlay["FLA_TILELANG"])
        env["TILELANG_CACHE_DIR"] = str(
            q35_overlay["TILELANG_CACHE_DIR"]
        )
        q35_probe = _command(
            [
                str(q35_python),
                "-c",
                (
                    "import torch,transformers,flash_attn,tilelang,fla;"
                    "print(torch.__version__,transformers.__version__,"
                    "flash_attn.__version__,tilelang.__version__,"
                    "fla.__version__)"
                ),
            ],
            env=env,
        )
    return {
        "static_checks": static,
        "qwen35_overlay_manifest": {
            "sha256": overlay_sha256,
            "files": overlay_files,
            "expected_sha256": QWEN35_OVERLAY_SHA256,
            "expected_files": QWEN35_OVERLAY_FILES,
        },
        "qwen3_import_probe": q3_probe,
        "qwen35_import_probe": q35_probe,
        "all_passed": (
            all(static.values())
            and q3_probe is not None
            and q3_probe["passed"]
            and q35_probe is not None
            and q35_probe["passed"]
        ),
    }


def _freeze_integrity(campaign_root: Path) -> dict[str, Any]:
    path = campaign_root / "FREEZE.json"
    freeze = read_json(path)
    unsigned = dict(freeze)
    digest = unsigned.pop("freeze_sha256", None)
    checks: dict[str, bool] = {
        "freeze_checksum": digest == sha256_json(unsigned),
        "no_gpu_launch_recorded": (
            freeze.get("gpu_experiments_launched") is False
        ),
        "no_results_used_for_fit": (
            freeze.get("results_consumed_for_model_fit") is False
        ),
    }
    for name in ("model_artifact", "prediction_freeze"):
        binding = freeze[name]
        bound = Path(binding["path"])
        checks[f"{name}_sha256"] = (
            bound.is_file() and sha256_file(bound) == binding["sha256"]
        )
    jobs = freeze["jobs"]
    job_path = Path(jobs["path"])
    rows = read_jsonl(job_path)
    checks["job_queue_sha256"] = (
        sha256_file(job_path) == jobs["sha256"]
    )
    checks["job_count"] = len(rows) == int(jobs["count"]) == 31
    checks["job_ids_unique"] = len(
        {str(row["job_id"]) for row in rows}
    ) == len(rows)
    for subcampaign in freeze["subcampaigns"]:
        binding = subcampaign["combined_queue"]
        queue = Path(binding["path"])
        checks[
            f"{subcampaign['model_id']}_combined_queue"
        ] = (
            queue.is_file()
            and sha256_file(queue) == binding["sha256"]
            and len(read_jsonl(queue)) == int(binding["jobs"])
        )
    return {
        "path": str(path.resolve()),
        "freeze_sha256": digest,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def _schedule(campaign_root: Path) -> dict[str, Any]:
    reports = []
    total_waves = 0
    for model_id in ("qwen3_8b", "qwen3p5_4b"):
        root = campaign_root / model_id
        for path in sorted((root / "matrix").glob("queue_*.jsonl")):
            if path.name == "queue_all.jsonl":
                continue
            rows = read_jsonl(path)
            slots = sum(int(row["gpu_count"]) for row in rows)
            minimum_waves = math.ceil(slots / 4)
            # Each 4-GPU job consumes an exclusive wave.  The remaining lower
            # bound can still be tightly packed on disjoint masks.
            four_gpu = sum(int(row["gpu_count"]) == 4 for row in rows)
            lower_without_four = math.ceil(
                sum(
                    int(row["gpu_count"])
                    for row in rows
                    if int(row["gpu_count"]) < 4
                )
                / 4
            )
            minimum_waves = max(
                minimum_waves,
                four_gpu + lower_without_four,
            )
            total_waves += minimum_waves
            reports.append(
                {
                    "model_id": model_id,
                    "phase": path.stem.removeprefix("queue_"),
                    "jobs": len(rows),
                    "gpu_slots": slots,
                    "by_gpu_count": dict(
                        Counter(int(row["gpu_count"]) for row in rows)
                    ),
                    "minimum_fully_packed_waves": minimum_waves,
                }
            )
    return {
        "phases": reports,
        "minimum_sequential_runtime_waves": total_waves,
        "policy": (
            "pack disjoint masks within a runtime; never overlap the Qwen3 "
            "and Qwen3.5 schedulers"
        ),
    }


def build_report(campaign_root: Path) -> dict[str, Any]:
    integrity = _freeze_integrity(campaign_root)
    hardware = _hardware()
    runtime = _runtime(campaign_root)
    blockers = []
    if not integrity["all_passed"]:
        blockers.append("frozen_artifact_or_queue_integrity_failed")
    if not hardware["exact_four_card_rtx4090_pool"]:
        blockers.append("current_host_is_not_the_frozen_4x_rtx4090_pool")
    elif not hardware["selected_pool_idle"]:
        blockers.append("rtx4090_gpu_0_1_2_3_are_not_idle")
    if not runtime["all_passed"]:
        blockers.append("one_or_more_frozen_runtimes_are_unavailable")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_root": str(campaign_root.resolve()),
        "ready_to_launch": not blockers,
        "status": "ready" if not blockers else "blocked_fail_closed",
        "blockers": blockers,
        "freeze_integrity": integrity,
        "hardware": hardware,
        "runtime": runtime,
        "schedule": _schedule(campaign_root),
        "process_policy": (
            "No process is terminated or signalled. Occupied GPUs cause a "
            "fail-closed result."
        ),
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT,
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (
        args.campaign_root
        / "artifacts"
        / "preflight_current_host.json"
    )
    report = build_report(args.campaign_root.resolve())
    write_json(output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ready_to_launch"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
