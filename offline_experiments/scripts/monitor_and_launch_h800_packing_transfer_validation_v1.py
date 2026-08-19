#!/usr/bin/env python3
"""Wait for the full H800 pool to become idle, then launch exactly once."""

from __future__ import annotations

import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from common import ROOT, RUNTIME_DIR, write_json
from prepare_h800_packing_transfer_validation_v1 import GPU_IDS
from prepare_h800_prospective_holdout import probe_hardware


PYTHON = Path("/fine-tuning-launcher/.venv/bin/python")
LAUNCHER = ROOT / "packing_config_ranking_staging" / "launch_h800_packing_transfer_validation_v1.py"
LOCK = RUNTIME_DIR / "h800_packing_transfer_validation_monitor.lock"
STATUS = RUNTIME_DIR / "h800_packing_transfer_validation_monitor_status.json"
EVENTS = RUNTIME_DIR / "h800_packing_transfer_validation_monitor_events.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(payload: dict[str, Any]) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as output:
        output.write(json.dumps({"at_utc": _now(), **payload}, ensure_ascii=False) + "\n")


def main() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    lock = LOCK.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("a Packing transfer launch monitor is already active")
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    _event({"event": "monitor_started", "pid": os.getpid(), "gpu_ids": list(GPU_IDS)})
    while True:
        hardware = probe_hardware(required_gpu_ids=list(GPU_IDS))
        processes = hardware.get("selected_gpu_compute_processes") or []
        state = {
            "schema": "sft_h800_packing_transfer_validation_monitor_status/v1",
            "updated_at_utc": _now(),
            "monitor_pid": os.getpid(),
            "state": "waiting_for_full_pool_idle" if processes else "launching",
            "exact_h800_pool": hardware.get("exact_h800_pool") is True,
            "blocking_process_count": len(processes),
            "blocking_pids": sorted({int(row["pid"]) for row in processes}),
            "preemption_attempted": False,
        }
        write_json(STATUS, state)
        if hardware.get("exact_h800_pool") is True and not processes:
            break
        time.sleep(15.0)

    _event({"event": "full_pool_idle_launch_requested"})
    command = [str(PYTHON), str(LAUNCHER), "--execute"]
    result = subprocess.run(command, cwd=ROOT, text=True, check=False)
    final = {
        "schema": "sft_h800_packing_transfer_validation_monitor_status/v1",
        "updated_at_utc": _now(),
        "monitor_pid": os.getpid(),
        "state": "launcher_completed" if result.returncode == 0 else "launcher_failed",
        "launcher_return_code": result.returncode,
        "preemption_attempted": False,
    }
    write_json(STATUS, final)
    _event({"event": final["state"], "launcher_return_code": result.returncode})
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
