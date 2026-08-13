#!/usr/bin/env python3
"""Rebuild a scheduler queue from durable results while excluding live jobs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from common import RESULTS_DIR, read_json, read_jsonl, write_json, write_jsonl


def active_job_ids(known_job_ids: Iterable[str], proc_root: Path = Path("/proc")) -> set[str]:
    known = set(known_job_ids)
    command_lines: list[str] = []
    for path in proc_root.glob("[0-9]*/cmdline"):
        try:
            command_lines.append(path.read_bytes().replace(b"\0", b" ").decode(errors="replace"))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    joined = "\n".join(command_lines)
    return {job_id for job_id in known if job_id in joined}


def classification(job_id: str, results_dir: Path = RESULTS_DIR) -> str | None:
    status_path = results_dir / job_id / "status.json"
    if not status_path.is_file():
        return None
    return str(read_json(status_path).get("classification") or "unknown")


def partition_jobs(
    jobs: list[dict[str, Any]],
    active: set[str],
    accepted: set[str],
    results_dir: Path = RESULTS_DIR,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()
    active_in_queue: list[str] = []
    for job in jobs:
        job_id = str(job["job_id"])
        outcome = classification(job_id, results_dir)
        if outcome in accepted:
            outcomes[f"accepted:{outcome}"] += 1
        elif job_id in active:
            outcomes["active"] += 1
            active_in_queue.append(job_id)
        else:
            outcomes[f"queued:{outcome or 'missing'}"] += 1
            pending.append(job)
    summary = {
        "input_jobs": len(jobs),
        "pending_jobs": len(pending),
        "active_job_ids": sorted(active_in_queue),
        "outcomes": dict(sorted(outcomes.items())),
    }
    return pending, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted", action="append", default=["success"])
    args = parser.parse_args()

    jobs = read_jsonl(args.input)
    known_ids = [str(job["job_id"]) for job in jobs]
    active = active_job_ids(known_ids)
    pending, summary = partition_jobs(jobs, active, set(args.accepted))
    write_jsonl(args.output, pending)
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    write_json(summary_path, {**summary, "input": str(args.input), "output": str(args.output)})
    print(json.dumps({**summary, "summary_path": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
