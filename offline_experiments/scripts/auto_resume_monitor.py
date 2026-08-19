#!/usr/bin/env python3
"""Watch the tangcan FA3 training; when it exits and GPUs drain,
resume the approved h800_hybrid_attention_dense_stage1 queue.

Keeps re-invoking resume_approved_queue.py until it returns 0, so an
interrupted resume is re-launched automatically (terminal jobs are skipped).
"""

from __future__ import annotations

import datetime
import subprocess
import sys
import time

PROJECT = "/wanqing-develop/luowenjing/Param_Recommend"
LOG = PROJECT + "/offline_experiments/runtime/auto_resume_monitor.log"
FA3_PID = 159599  # torchrun parent of the tangcan FA3 training
RESUME = [
    PROJECT + "/offline_experiments/scripts/resume_approved_queue.py",
    "--input",
    PROJECT + "/offline_experiments/matrix/h800_hybrid_attention_dense_stage1_formal_v1.jsonl",
    "--execute",
]
VENV_PY = "/fine-tuning-launcher/.venv/bin/python"


def log(msg: str) -> None:
    line = f"[{datetime.datetime.now().isoformat()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def fa3_alive() -> bool:
    try:
        out = subprocess.run(["ps", "-p", str(FA3_PID), "-o", "pid="],
                             capture_output=True, text=True)
        return bool(out.stdout.strip())
    except Exception:
        return True


def gpu_compute_pids() -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True)
    return [p.strip() for p in out.stdout.splitlines() if p.strip()]


def run_resume() -> int:
    cmd = [VENV_PY] + RESUME
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = PROJECT + "/offline_experiments/scripts"
    proc = subprocess.run(cmd, cwd=PROJECT, env=env,
                          capture_output=True, text=True)
    log(f"resume rc={proc.returncode} "
        f"stdout_tail={proc.stdout[-600:]!r} "
        f"stderr_tail={proc.stderr[-600:]!r}")
    return proc.returncode


def main() -> int:
    log("monitor started; waiting for FA3 training (pid %d) to finish" % FA3_PID)
    while fa3_alive():
        time.sleep(30)
    log("FA3 process gone; waiting for GPU compute processes to drain")
    deadline = time.time() + 30 * 60  # allow up to 30 min for drain
    pids = gpu_compute_pids()
    while pids and time.time() < deadline:
        log("still busy gpu pids=%s, sleeping 20s" % pids)
        time.sleep(20)
        pids = gpu_compute_pids()
    if pids:
        log("ERROR: GPU compute processes remain after 30 min: %s" % pids)
        return 1
    log("GPU free; launching resume loop")
    attempts = 0
    while True:
        attempts += 1
        log("resume launch #%d" % attempts)
        rc = run_resume()
        if rc == 0:
            log("resume finished cleanly (queue fully drained)")
            return 0
        log("resume rc=%d; waiting 60s before relaunching" % rc)
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
