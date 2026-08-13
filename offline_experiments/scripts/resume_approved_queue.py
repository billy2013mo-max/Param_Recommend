#!/usr/bin/env python3
"""Resume an approved queue without rerunning terminal success/OOM jobs.

The complete, originally approved queue is verified before filtering.  The
approval execution lock is held for the lifetime of the resumed scheduler, so
approval promotion cannot change the authorized job set between launches.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from approval_gate import execution_lock
from common import RESULTS_DIR, ROOT, RUNTIME_DIR, gpu_process_snapshot, read_json, read_jsonl
from run_job import verify_approval
from scheduler import (
    GPU_IDS,
    append_event,
    dashboard_phase,
    occupied_gpu_ids,
    run_scheduler,
    validate_jobs_in_gpu_scope,
)


TERMINAL_CLASSIFICATIONS = frozenset({"success", "oom"})


def result_classification(job_id: str) -> str | None:
    status_path = RESULTS_DIR / job_id / "status.json"
    if not status_path.is_file():
        return None
    classification = read_json(status_path).get("classification")
    return str(classification) if classification is not None else None


def resume_plan(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    terminal: dict[str, int] = {}
    pending: list[dict[str, Any]] = []
    for job in jobs:
        classification = result_classification(str(job["job_id"]))
        if classification in TERMINAL_CLASSIFICATIONS:
            terminal[classification] = terminal.get(classification, 0) + 1
        else:
            pending.append(job)
    return {
        "total": len(jobs),
        "terminal": dict(sorted(terminal.items())),
        "pending": pending,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--busy-poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    queue_path = args.input.resolve()
    jobs = read_jsonl(queue_path)
    if not jobs:
        raise ValueError("No jobs in the approved queue")
    validate_jobs_in_gpu_scope(jobs)
    plan = resume_plan(jobs)
    pending = plan.pop("pending")
    preview = {
        **plan,
        "pending": len(pending),
        "pending_job_ids": [str(job["job_id"]) for job in pending],
        "training_started": bool(args.execute),
    }
    if not args.execute:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    with execution_lock(RUNTIME_DIR, exclusive=False):
        approval_evidence = verify_approval(
            queue_path=queue_path,
            queue_rows=jobs,
            acquire_lock=False,
        )
        approval_design = read_json(Path(approval_evidence["approval_design_path"]))
        if (approval_design.get("scheduler_execution") or {}).get("join_busy_pool") is not False:
            raise PermissionError(
                "This resume entry supports only an approval frozen with join_busy_pool=false"
            )
        initial_blocked = occupied_gpu_ids()
        if initial_blocked:
            snapshot = gpu_process_snapshot(GPU_IDS)
            raise RuntimeError(
                "Refusing to resume: approved GPUs have compute processes: "
                f"{snapshot['processes']}"
            )
        if not pending:
            print(json.dumps(preview, ensure_ascii=False, indent=2))
            return

        event_path = RUNTIME_DIR / "scheduler_events.jsonl"
        execution_id = f"resume-scheduler-{int(time.time())}"
        append_event(
            event_path,
            {
                "event": "scheduler_start",
                "execution_id": execution_id,
                "input": str(queue_path),
                "jobs": len(pending),
                "approved_queue_jobs": len(jobs),
                "resume_terminal_skipped": plan["terminal"],
                "resume_mode": True,
                "phases": sorted({dashboard_phase(job) for job in pending}),
                "initial_blocked_gpus": [],
                "join_busy_pool": False,
            },
        )
        asyncio.run(
            run_scheduler(
                pending,
                event_path,
                initial_blocked=set(),
                busy_poll_seconds=args.busy_poll_seconds,
                execution_id=execution_id,
                require_execution_gate=True,
            )
        )
        append_event(
            event_path,
            {
                "event": "scheduler_complete",
                "execution_id": execution_id,
                "input": str(queue_path),
                "jobs": len(pending),
                "approved_queue_jobs": len(jobs),
                "resume_mode": True,
            },
        )
        print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
