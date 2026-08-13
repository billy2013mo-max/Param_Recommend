#!/usr/bin/env python3
"""Resumable one-command controller for the staged SFT experiment.

The default mode is read-only status reporting.  ``--execute`` is required to
launch schedulers.  Completed job IDs are never launched again, and every phase
is checked before the next phase is materialized.

This file is deliberately not part of the currently frozen Stage-A manifest:
adding it does not invalidate the already-running approved memory experiment.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from common import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    MATRIX_DIR,
    RESULTS_DIR,
    ROOT,
    RUNTIME_DIR,
    read_json,
    read_jsonl,
    sha256_file,
    stable_id,
    write_json,
    write_jsonl,
)
from run_job import verify_approval
from stage_decisions import (
    build_reports,
    scaling_eligible_request_ids,
    stable_holdout_score,
)


PIPELINE_DIR = RUNTIME_DIR / "pipeline"
STATE_PATH = PIPELINE_DIR / "state.json"
LOCK_PATH = PIPELINE_DIR / "controller.lock"
EVENT_PATH = RUNTIME_DIR / "scheduler_events.jsonl"
PHASE_ORDER = (
    "memory",
    "throughput",
    "scaling",
    "packing",
    "profiler",
    "analysis",
)
CONTROLLER_SOURCES = (
    ROOT / "scripts" / "run_pipeline.py",
    ROOT / "scripts" / "stage_decisions.py",
)
MATERIALIZED_PATHS = {
    "throughput-screen": MATRIX_DIR / "throughput_screen_jobs.jsonl",
    "throughput": MATRIX_DIR / "throughput_jobs.jsonl",
    "scaling": MATRIX_DIR / "scaling_candidate_jobs.jsonl",
    "packing-memory": MATRIX_DIR / "packing_memory_jobs.jsonl",
    "packing-formal": MATRIX_DIR / "packing_pair_jobs.jsonl",
    "profiler": MATRIX_DIR / "profiler_jobs.jsonl",
}

PROFILER_HOLDOUT_MAX_GPU_COUNT = 2
PROFILER_HOLDOUT_MAX_GRADIENT_ACCUMULATION = 8


def now_state(stage: str, status: str, **details: Any) -> dict[str, Any]:
    previous = read_json(STATE_PATH) if STATE_PATH.is_file() else {"history": []}
    history = list(previous.get("history") or [])
    event = {"time_unix": time.time(), "stage": stage, "status": status, **details}
    history.append(event)
    state = {"schema_version": 1, "current": event, "history": history[-500:]}
    write_json(STATE_PATH, state)
    return state


def verify_controller_frozen(design_path: Path = RUNTIME_DIR / "approval_design.json") -> None:
    """Require downstream decision code to be part of the approved manifest."""

    design = read_json(design_path)
    manifest = design.get("file_sha256") or {}
    missing = []
    mismatched = []
    for path in CONTROLLER_SOURCES:
        relative = str(path.relative_to(ROOT))
        expected = manifest.get(relative)
        if expected is None:
            missing.append(relative)
        elif sha256_file(path) != expected:
            mismatched.append(relative)
    if missing or mismatched:
        raise PermissionError(
            "Pipeline controller is not covered by the frozen approval design; "
            f"re-freeze after Stage A and re-bind approval (missing={missing}, mismatched={mismatched})"
        )


def result_classification(job_id: str) -> str | None:
    path = RESULTS_DIR / job_id / "status.json"
    if not path.is_file():
        return None
    return str(read_json(path).get("classification"))


def successful_result_is_healthy(job: dict[str, Any]) -> bool:
    """Reject truncated or internally incomplete formal measurements.

    A healthy success has one summary per rank, at least the requested
    measurement window on every rank, positive measured time and positive
    logical-sample totals. This allows a previous longer run to satisfy a
    reduced design without wasting GPU time. Memory probes use success/OOM
    boundary semantics and are checked by their own stage gate instead.
    """

    job_id = str(job["job_id"])
    if result_classification(job_id) != "success":
        return False
    if job.get("kind") in {"memory_probe", "packing_memory_probe"}:
        return True
    expected_ranks = int(job.get("gpu_count") or 0)
    expected_steps = job.get("measure_steps")
    if expected_ranks <= 0 or expected_steps is None:
        return True
    summaries = [
        read_json(path)
        for path in sorted((RESULTS_DIR / job_id / "metrics").glob("summary.rank*.json"))
    ]
    if len(summaries) != expected_ranks:
        return False
    summaries_healthy = all(
        summary.get("failure") is None
        and int(summary.get("measured_steps") or 0) >= int(expected_steps)
        and float(summary.get("measured_seconds") or 0.0) > 0.0
        and int((summary.get("measured_totals") or {}).get("logical_samples") or 0) > 0
        for summary in summaries
    )
    if not summaries_healthy:
        return False
    if not job.get("enable_profiler"):
        return True
    operator_paths = sorted(
        (RESULTS_DIR / job_id / "metrics" / "profiler").glob("operators.rank*.json")
    )
    if len(operator_paths) != expected_ranks:
        return False
    return all(
        sum(float(row.get("flops") or 0.0) for row in read_json(path)) > 0.0
        for path in operator_paths
    )


def job_outcome(job: dict[str, Any]) -> str:
    classification = result_classification(str(job["job_id"])) or "missing"
    if classification == "success" and not successful_result_is_healthy(job):
        return "unhealthy_success"
    return classification


def pending_jobs(jobs: Iterable[dict[str, Any]], accepted: set[str] | None = None) -> list[dict[str, Any]]:
    terminal = accepted or {"success"}
    rerun_unhealthy = bool(
        read_json(CONFIG_DIR / "experiment.json")["measurement"].get("rerun_on_unhealthy_result", True)
    )
    return [
        job
        for job in jobs
        if result_classification(str(job["job_id"])) not in terminal
        or (
            rerun_unhealthy
            and result_classification(str(job["job_id"])) == "success"
            and not successful_result_is_healthy(job)
        )
    ]


def job_outcomes(jobs: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        status = job_outcome(job)
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def throughput_scenarios_without_success(
    jobs: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return formal-throughput scenarios that have no healthy finalist.

    OOM is a valid negative result for an individual finalist.  It must not
    abort the complete campaign when another finalist in the same scenario
    succeeded, but a scenario with no successful finalist still blocks
    downstream scaling and recommendation decisions.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        grouped.setdefault(str(job["request_id"]), []).append(job)

    unavailable = []
    for request_id, candidates in sorted(grouped.items()):
        outcomes = job_outcomes(candidates)
        if outcomes.get("success", 0) > 0:
            continue
        representative = candidates[0]
        unavailable.append(
            {
                "request_id": request_id,
                "model_id": representative.get("model_id"),
                "train_type": representative.get("train_type"),
                "dataset_id": representative.get("dataset_id"),
                "target_gbs": representative.get("target_gbs"),
                "outcomes": outcomes,
            }
        )
    return unavailable


def scheduler_active(event_path: Path = EVENT_PATH) -> bool:
    if not event_path.is_file():
        return False
    latest_start: dict[str, Any] | None = None
    completed: set[str] = set()
    with event_path.open(encoding="utf-8") as source:
        for line in source:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            execution_id = event.get("execution_id")
            if event.get("event") == "scheduler_start" and execution_id:
                latest_start = event
            elif event.get("event") == "scheduler_complete" and execution_id:
                completed.add(str(execution_id))
    return bool(latest_start and str(latest_start["execution_id"]) not in completed)


def memory_progress() -> dict[str, Any]:
    families = read_jsonl(MATRIX_DIR / "memory_boundary_families.jsonl")
    summarized = []
    missing = []
    for family in families:
        summary = RESULTS_DIR / "boundary_summaries" / f"{family['job_id']}.json"
        if summary.is_file():
            summarized.append(family)
            continue
        # Runtime/configuration failures are not feasibility evidence. Keep
        # the family retryable until a repaired runtime measures success/OOM.
        missing.append(family)
    return {
        "total": len(families),
        "complete": len(summarized),
        "summarized": len(summarized),
        "excluded": [],
        "missing": missing,
    }


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def scheduler_command(input_path: Path) -> list[str]:
    python = str(read_json(CONFIG_DIR / "experiment.json")["fixed_runtime"]["python"])
    return [
        python,
        str(ROOT / "scripts" / "scheduler.py"),
        "--input",
        str(input_path),
        "--execute",
        "--join-busy-pool",
    ]


def run_scheduler_jobs(
    jobs: list[dict[str, Any]],
    stage: str,
    accepted: set[str] | None = None,
) -> dict[str, int]:
    accepted = accepted or {"success"}
    outstanding = pending_jobs(jobs, accepted)
    if outstanding:
        input_path = PIPELINE_DIR / f"pending-{stage}.jsonl"
        write_jsonl(input_path, outstanding)
        now_state(stage, "running", jobs=len(outstanding), input=str(input_path))
        run_command(scheduler_command(input_path))
    outcomes = job_outcomes(jobs)
    unacceptable = {name: count for name, count in outcomes.items() if name not in accepted}
    if unacceptable:
        now_state(stage, "failed_gate", outcomes=outcomes)
        raise RuntimeError(f"Stage {stage} has unacceptable job outcomes: {unacceptable}")
    now_state(stage, "complete", outcomes=outcomes)
    return outcomes


def wait_for_running_memory(poll_seconds: float) -> None:
    while scheduler_active():
        progress = memory_progress()
        print(
            f"Attaching to active memory scheduler: {progress['complete']}/{progress['total']} families complete",
            flush=True,
        )
        time.sleep(poll_seconds)


def complete_memory_stage(poll_seconds: float, attach_existing: bool) -> None:
    progress = memory_progress()
    if not progress["missing"]:
        now_state("memory", "complete", families=progress["total"], excluded=progress["excluded"])
        return
    if scheduler_active():
        if not attach_existing:
            raise RuntimeError("Another scheduler is active; use the default attach behavior or wait for it to finish")
        now_state("memory", "attached", complete=progress["complete"], total=progress["total"])
        wait_for_running_memory(poll_seconds)
        progress = memory_progress()
    if progress["missing"]:
        # Resume only missing families.  This is also the safe recovery path if a
        # prior scheduler stopped after an infrastructure/code failure.
        input_path = PIPELINE_DIR / "pending-memory-resume.jsonl"
        write_jsonl(input_path, progress["missing"])
        now_state("memory-resume", "running", families=len(progress["missing"]), input=str(input_path))
        run_command(scheduler_command(input_path))
    progress = memory_progress()
    if progress["missing"]:
        raise RuntimeError(f"Memory stage ended with {len(progress['missing'])} boundary summaries missing")
    now_state("memory", "complete", families=progress["total"], excluded=progress["excluded"])


def materialize(phase: str) -> list[dict[str, Any]]:
    python = str(read_json(CONFIG_DIR / "experiment.json")["fixed_runtime"]["python"])
    run_command([python, str(ROOT / "scripts" / "materialize_jobs.py"), phase])
    path = MATERIALIZED_PATHS[phase]
    return read_jsonl(path) if path.is_file() else []


def write_reports() -> dict[str, Any]:
    report = build_reports(RESULTS_DIR)
    output = ARTIFACT_DIR / "stage_decisions.json"
    write_json(output, report)
    write_json(ARTIFACT_DIR / "profiler_calibration.json", report["profiler_calibration"])
    return report


def throughput_physical_key(job: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        job.get(field)
        for field in (
            "request_id", "model_id", "train_type", "dataset_id", "cutoff_len",
            "gpu_count", "zero", "gc", "mbs", "target_gbs", "packing",
        )
    )


def completed_formal_throughput_keys() -> set[tuple[Any, ...]]:
    completed = set()
    for rendered_path in RESULTS_DIR.glob("*/rendered_run.json"):
        try:
            job = read_json(rendered_path)["job"]
        except (KeyError, OSError, json.JSONDecodeError):
            continue
        if job.get("kind") != "throughput" or not successful_result_is_healthy(job):
            continue
        completed.add(throughput_physical_key(job))
    return completed


def prepare_throughput_screen_queue() -> dict[str, Any]:
    """Materialize and persist the resumable screen queue without training."""

    screen_jobs = materialize("throughput-screen")
    completed_formal = completed_formal_throughput_keys()
    require_screen = [
        job
        for job in screen_jobs
        if throughput_physical_key(job) not in completed_formal
    ]
    outstanding = pending_jobs(require_screen, accepted={"success", "oom"})
    input_path = PIPELINE_DIR / "pending-throughput-screen.jsonl"
    write_jsonl(input_path, outstanding)
    state = now_state(
        "throughput-screen",
        "queued_waiting_approval",
        jobs=len(outstanding),
        candidate_jobs=len(screen_jobs),
        substituted_by_healthy_formal=len(screen_jobs) - len(require_screen),
        already_screened=len(require_screen) - len(outstanding),
        input=str(input_path),
    )
    return {
        "stage": "throughput-screen",
        "status": "queued_waiting_approval",
        "candidate_jobs": len(screen_jobs),
        "queued_jobs": len(outstanding),
        "substituted_by_healthy_formal": len(screen_jobs) - len(require_screen),
        "already_screened": len(require_screen) - len(outstanding),
        "input": str(input_path),
        "pipeline_state": state["current"],
    }


def run_throughput() -> dict[str, Any]:
    screen_jobs = materialize("throughput-screen")
    if not screen_jobs:
        raise RuntimeError(
            "No throughput screening jobs were materialized; memory results are incomplete or infeasible"
        )
    completed_formal = completed_formal_throughput_keys()
    required_screen_jobs = [
        job for job in screen_jobs if throughput_physical_key(job) not in completed_formal
    ]
    if required_screen_jobs:
        run_scheduler_jobs(
            required_screen_jobs,
            "throughput-screen",
            accepted={"success", "oom"},
        )
    report = write_reports()
    incomplete = [
        row
        for row in report["throughput_screening"]["decisions"]
        if row["status"] == "screening_in_progress"
    ]
    if incomplete:
        raise RuntimeError(f"Throughput screening remains incomplete for {len(incomplete)} scenarios")

    jobs = materialize("throughput")
    if jobs:
        outcomes = run_scheduler_jobs(jobs, "throughput", accepted={"success", "oom"})
        unavailable = throughput_scenarios_without_success(jobs)
        if unavailable:
            now_state(
                "throughput",
                "failed_viability_gate",
                outcomes=outcomes,
                scenarios_without_success=unavailable,
            )
            raise RuntimeError(
                "Formal throughput has no successful finalist for "
                f"{len(unavailable)} scenarios"
            )
    else:
        now_state("throughput", "skipped", reason="screening produced no viable finalists")
    return write_reports()


def run_scaling() -> dict[str, Any]:
    jobs = materialize("scaling")
    requests = read_jsonl(MATRIX_DIR / "strong_scaling_requests.jsonl")
    request_ids = {str(request["request_id"]) for request in requests}
    report = write_reports()
    for gpu_count in (1, 2, 4):
        eligible = request_ids if gpu_count in (1, 2) else scaling_eligible_request_ids(report["scaling"], gpu_count)
        card_jobs = [
            job
            for job in jobs
            if int(job["gpu_count"]) == gpu_count and str(job["request_id"]) in eligible
        ]
        if card_jobs:
            # OOM is a valid negative scaling observation for one card count.
            # It must remain in the report as infeasible evidence while later
            # card counts are still allowed to run. Non-memory failures remain
            # outside the accepted set and continue to stop the pipeline.
            run_scheduler_jobs(
                card_jobs,
                f"scaling-{gpu_count}gpu",
                accepted={"success", "oom"},
            )
        else:
            now_state(f"scaling-{gpu_count}gpu", "skipped", reason="no eligible feasible candidates")
        report = write_reports()
    now_state("scaling", "complete", families=len(report["scaling"]["families"]))
    return report


def run_packing() -> dict[str, Any]:
    memory_jobs = materialize("packing-memory")
    if memory_jobs:
        # An explicit OOM is a valid negative memory label for packing; ordinary
        # failures still stop the pipeline.
        run_scheduler_jobs(memory_jobs, "packing-memory", accepted={"success", "oom"})
    formal_jobs = materialize("packing-formal")
    if formal_jobs:
        run_scheduler_jobs(formal_jobs, "packing-formal", accepted={"success", "oom"})
    report = write_reports()
    incomplete = [row for row in report["packing"]["decisions"] if row["status"] != "complete"]
    now_state(
        "packing",
        "complete",
        complete_pairs=len(report["packing"]["decisions"]) - len(incomplete),
        excluded_or_infeasible=len(incomplete),
        packing_on=report["packing"]["packing_on"],
    )
    return report


def matching_job(selected: dict[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any] | None:
    fields = ("campaign_id", "hardware_id", "request_id", "gpu_count", "zero", "gc", "mbs", "packing")
    return next(
        (
            job
            for job in jobs
            if all(job.get(field) == selected.get(field) for field in fields)
            and result_classification(str(job["job_id"])) == "success"
        ),
        None,
    )


def profiler_holdout_target_gbs(source: dict[str, Any]) -> int:
    gpu_count = max(1, int(source.get("gpu_count") or 1))
    mbs = max(1, int(source.get("mbs") or 1))
    source_target_gbs = int(source.get("target_gbs") or gpu_count * mbs)
    return min(
        source_target_gbs,
        gpu_count * mbs * PROFILER_HOLDOUT_MAX_GRADIENT_ACCUMULATION,
    )


def profiler_holdout_physical_key(source: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        profiler_holdout_target_gbs(source)
        if field == "target_gbs"
        else source.get(field)
        for field in (
            "campaign_id",
            "hardware_id",
            "model_id",
            "train_type",
            "dataset_id",
            "gpu_count",
            "zero",
            "gc",
            "mbs",
            "target_gbs",
        )
    )


def profiler_holdout_jobs(report: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    """Derive bounded independent profiler holdouts from throughput winners.

    A profiler point calibrates operator FLOPs; it does not need to inherit a
    throughput winner's potentially huge global batch.  Capping gradient
    accumulation keeps the advertised 1+3-step window genuinely small.
    Four-GPU winners are also excluded so a transiently busy approved GPU
    cannot block every remaining profiler holdout.
    """

    measurement = read_json(CONFIG_DIR / "experiment.json")["measurement"]
    throughput_path = MATERIALIZED_PATHS["throughput"]
    throughput_jobs = read_jsonl(throughput_path) if throughput_path.is_file() else []
    candidates = []
    for decision in report["throughput"]["decisions"]:
        selected = decision.get("selected")
        if not selected:
            continue
        source = matching_job(selected, throughput_jobs)
        if source is None:
            continue
        if int(source.get("gpu_count") or 1) > PROFILER_HOLDOUT_MAX_GPU_COUNT:
            continue
        if (
            source.get("model_id") == "qwen3_8b"
            and source.get("dataset_id") == "multiturn_4096"
            and int(source.get("target_gbs") or 0) == 64
        ):
            continue
        candidates.append(source)

    # Different throughput scenarios (most often different source GBS values)
    # can collapse to the same physical profiler configuration after the
    # accumulation cap.  Keep one representative so a job ID is never emitted
    # twice or rerun concurrently.
    unique_candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
    for source in candidates:
        unique_candidates.setdefault(profiler_holdout_physical_key(source), source)
    candidates = list(unique_candidates.values())

    chosen = []
    covered: dict[str, set[Any]] = {field: set() for field in ("model_id", "train_type", "dataset_id", "target_gbs", "gc")}
    remaining = {str(job["job_id"]): job for job in candidates}
    while remaining and len(chosen) < limit:
        def novelty(job: dict[str, Any]) -> tuple[int, str]:
            score = sum(job.get(field) not in covered[field] for field in covered)
            score += 2 if job.get("model_id") != "qwen3_8b" else 0
            score += 1 if job.get("dataset_id") != "multiturn_4096" else 0
            score += 1 if int(job.get("target_gbs") or 0) != 64 else 0
            return score, stable_holdout_score(job)

        selected_source = max(remaining.values(), key=novelty)
        remaining.pop(str(selected_source["job_id"]))
        chosen.append(selected_source)
        for field in covered:
            covered[field].add(selected_source.get(field))

    jobs = []
    for source in chosen:
        gpu_count = max(1, int(source.get("gpu_count") or 1))
        mbs = max(1, int(source.get("mbs") or 1))
        source_target_gbs = int(source.get("target_gbs") or gpu_count * mbs)
        profiler_target_gbs = profiler_holdout_target_gbs(source)
        identity = {
            field: (
                profiler_target_gbs
                if field == "target_gbs"
                else source.get(field)
            )
            for field in (
                "campaign_id",
                "hardware_id",
                "model_id",
                "train_type",
                "dataset_id",
                "gpu_count",
                "zero",
                "gc",
                "mbs",
                "target_gbs",
            )
        }
        job = copy.deepcopy(source)
        job.update(
            {
                "job_id": stable_id("profhold", identity),
                "kind": "profiler",
                "repeat": 0,
                "source_target_gbs": source_target_gbs,
                "target_gbs": profiler_target_gbs,
                "warmup_steps": int(measurement.get("profiler_warmup_steps", 1)),
                "measure_steps": int(measurement.get("profiler_measure_steps", 3)),
                "enable_profiler": True,
                "profiler_role": "holdout",
                "profiler_active_steps": 1,
                "parallel_class": "gpu_partitionable",
                "requires_external_node_idle": False,
            }
        )
        jobs.append(job)
    return jobs


def run_profiler(include_holdout: bool) -> dict[str, Any]:
    calibration_jobs = materialize("profiler")
    for job in calibration_jobs:
        source_target_gbs = int(job.get("target_gbs") or 1)
        job["source_target_gbs"] = source_target_gbs
        job["target_gbs"] = profiler_holdout_target_gbs(job)
        job["profiler_role"] = "calibration"
        job["profiler_active_steps"] = 1
    if calibration_jobs:
        run_scheduler_jobs(calibration_jobs, "profiler-calibration", accepted={"success", "oom"})
    report = write_reports()
    if include_holdout:
        holdout_jobs = profiler_holdout_jobs(report)
        write_jsonl(MATRIX_DIR / "profiler_holdout_jobs.jsonl", holdout_jobs)
        if holdout_jobs:
            run_scheduler_jobs(holdout_jobs, "profiler-holdout", accepted={"success", "oom"})
        else:
            now_state("profiler-holdout", "skipped", reason="no completed independent throughput winners")
    report = write_reports()
    now_state(
        "profiler",
        "complete",
        calibration_status=report["profiler_calibration"]["status"],
        resource_holdout_status=report["resource_holdout"]["status"],
    )
    return report


def status_report() -> dict[str, Any]:
    memory = memory_progress()
    state = read_json(STATE_PATH) if STATE_PATH.is_file() else None
    materialized = {}
    for phase, path in MATERIALIZED_PATHS.items():
        jobs = read_jsonl(path) if path.is_file() else []
        materialized[phase] = {"jobs": len(jobs), "outcomes": job_outcomes(jobs)}
    return {
        "schema_version": 1,
        "scheduler_active": scheduler_active(),
        "memory": {
            "total": memory["total"],
            "complete": memory["complete"],
            "summarized": memory["summarized"],
            "excluded": len(memory["excluded"]),
            "missing": len(memory["missing"]),
        },
        "materialized": materialized,
        "pipeline_state": state,
        "execution_requires_flag": "--execute",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Run/resume all stages; default is read-only status")
    mode.add_argument(
        "--prepare-throughput",
        action="store_true",
        help="Materialize the resumable throughput-screen queue without launching training",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--no-attach-existing", action="store_true")
    parser.add_argument("--skip-profiler-holdout", action="store_true")
    parser.add_argument("--stop-after", choices=PHASE_ORDER)
    args = parser.parse_args()
    if args.prepare_throughput:
        print(json.dumps(prepare_throughput_screen_queue(), ensure_ascii=False, indent=2))
        return
    if not args.execute:
        print(json.dumps(status_report(), ensure_ascii=False, indent=2))
        return

    verify_approval()
    verify_controller_frozen()
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another pipeline controller already holds the execution lock") from error

        complete_memory_stage(args.poll_seconds, not args.no_attach_existing)
        if args.stop_after == "memory":
            return
        run_throughput()
        if args.stop_after == "throughput":
            return
        run_scaling()
        if args.stop_after == "scaling":
            return
        run_packing()
        if args.stop_after == "packing":
            return
        run_profiler(not args.skip_profiler_holdout)
        if args.stop_after == "profiler":
            return
        report = write_reports()
        now_state(
            "analysis",
            "complete",
            output=str(ARTIFACT_DIR / "stage_decisions.json"),
            profiler_status=report["profiler_calibration"]["status"],
        )


if __name__ == "__main__":
    main()
