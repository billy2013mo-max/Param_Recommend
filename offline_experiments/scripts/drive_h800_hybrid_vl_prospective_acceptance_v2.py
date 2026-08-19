#!/usr/bin/env python3
"""Partial-pool driver for the V2 hybrid/VL prospective acceptance queue.

The 54-job V2 queue was frozen and promoted (APPROVED_TO_RUN.json).  External
processes may occupy some of the 8 GPUs; this driver adaptively uses only the
idle GPUs and expands as the pool frees up:

  * jobs are submitted in approval order but GPU-count-desc first (so a 4-GPU
    job is never starved by 1/2-GPU jobs on the same idle subset);
  * a job is only submitted with an idle gpu mask;
  * once a job's results/<job_id>/status.json exists it counts as completed;
  * OOM is terminal (no retry) — it is a right-censored lower bound.

Run:  python drive_h800_hybrid_vl_prospective_acceptance_v2.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from common import ROOT, RUNTIME_DIR, write_json, read_json, read_jsonl
from prepare_h800_hybrid_vl_prospective_acceptance_v2 import COMBINED_QUEUE, PHASE_ID

PYTHON = Path("/fine-tuning-launcher/.venv/bin/python")
GPU_IDS = list(range(8))
STATUS = RUNTIME_DIR / "h800_hybrid_vl_prospective_acceptance_v2_driver_status.json"
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _idle_gpus() -> list[int]:
    """Return the GPU indices with no compute processes (excluding our own)."""

    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return []
    busy_uuids = {
        line.split(",")[0].strip()
        for line in output.strip().splitlines()
        if line.strip()
    }
    if not busy_uuids:
        return list(GPU_IDS)
    uuid_to_index: dict[str, int] = {}
    try:
        inventory = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        for line in inventory.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                uuid_to_index[parts[1].strip()] = int(parts[0].strip())
    except subprocess.CalledProcessError:
        return []
    return [index for index in GPU_IDS if uuid_to_index.get(index) not in busy_uuids]


def _find_mask(gpu_count: int, idle: set[int]) -> list[int] | None:
    if gpu_count == 1:
        for index in GPU_IDS:
            if index in idle:
                return [index]
        return None
    if gpu_count == 2:
        for left, right in PAIRS:
            if left in idle and right in idle:
                return [left, right]
        return None
    if gpu_count == 4:
        for block in ((0, 1, 2, 3), (4, 5, 6, 7)):
            if all(index in idle for index in block):
                return list(block)
        return None
    return None


def _iterate_pending(
    jobs: list[dict[str, Any]],
    running: dict[str, list[int]],
) -> list[dict[str, Any]]:
    done = {
        str(job["job_id"])
        for job in jobs
        if (ROOT / "results" / str(job["job_id"]) / "status.json").is_file()
    }
    pending = [
        job
        for job in jobs
        if str(job["job_id"]) not in done and str(job["job_id"]) not in running
    ]
    return pending


def _status_snapshot(jobs: list[dict[str, Any]], running: dict[str, list[int]]) -> dict[str, Any]:
    completed = sum(
        1
        for job in jobs
        if (ROOT / "results" / str(job["job_id"]) / "status.json").is_file()
    )
    return {
        "schema": "sft_h800_hybrid_vl_prospective_acceptance_v2_driver_status",
        "updated_at_utc": _now(),
        "phase_id": PHASE_ID,
        "total_jobs": len(jobs),
        "completed_jobs": completed,
        "running_jobs": len(running),
        "running_masks": running,
    }


def drive(*, dry_run: bool = False, poll_seconds: int = 8) -> dict[str, Any]:
    jobs = read_jsonl(COMBINED_QUEUE)
    order = [str(job["job_id"]) for job in jobs]
    running: dict[str, list[int]] = {}
    processes: dict[str, subprocess.Popen[bytes]] = {}

    while True:
        snapshot = _status_snapshot(jobs, running)
        write_json(STATUS, snapshot)
        if snapshot["completed_jobs"] >= len(jobs):
            final = {**snapshot, "final_state": "all_jobs_terminal"}
            return final

        # Reap finished processes.
        for job_id in list(processes):
            process = processes[job_id]
            if process.poll() is not None:
                processes.pop(job_id)
                running.pop(job_id, None)

        idle = set(_idle_gpus())
        # Free GPUs that our own running jobs hold but that became free.
        for job_id, mask in list(running.items()):
            if dry_run:
                # No real subprocess in dry-run; treat the pseudo-running job
                # as holding its mask so it is not re-submitted this round.
                for index in mask:
                    idle.discard(index)
                continue
            process = processes.get(job_id)
            if process is None:
                running.pop(job_id, None)
            elif process.poll() is None:
                for index in mask:
                    idle.discard(index)

        # Order pending jobs: GPU-count descending within approval order.
        pending = [
            job
            for job in _iterate_pending(jobs, running)
            if (ROOT / "results" / str(job["job_id"]) / "status.json").is_file() is False
        ]
        pending.sort(key=lambda row: (-int(row["gpu_count"]), order.index(str(row["job_id"]))))

        launched = 0
        for job in pending:
            gpu_count = int(job["gpu_count"])
            mask = _find_mask(gpu_count, idle)
            if mask is None:
                continue
            for index in mask:
                idle.discard(index)
            job_id = str(job["job_id"])
            job_path = RUNTIME_DIR / "jobs" / f"input-{job_id}.json"
            write_json(job_path, job)
            if dry_run:
                print(f"[dry-run] would launch {job_id} gpu_count={gpu_count} mask={mask}")
                running[job_id] = mask
                launched += 1
                continue
            command = [
                str(PYTHON),
                str(ROOT / "scripts" / "run_job.py"),
                "--job-file",
                str(job_path),
                "--gpu-mask",
                ",".join(map(str, mask)),
                "--execute",
            ]
            print(f"[launch] {job_id} gpu_count={gpu_count} mask={mask}", flush=True)
            processes[job_id] = subprocess.Popen(
                command, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            running[job_id] = mask
            launched += 1

        if dry_run:
            return snapshot
        if not dry_run and not processes and pending:
            print(f"[wait] no runnable job on idle {sorted(idle)}", flush=True)
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=8)
    args = parser.parse_args()
    result = drive(dry_run=args.dry_run, poll_seconds=args.poll_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
