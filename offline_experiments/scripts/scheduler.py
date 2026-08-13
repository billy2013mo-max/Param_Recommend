#!/usr/bin/env python3
"""Bounded multi-GPU scheduler with adaptive MBS boundaries and an approval lock."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from approval_gate import execution_lock
from common import (
    CONFIG_DIR,
    MATRIX_DIR,
    RESULTS_DIR,
    ROOT,
    RUNTIME_DIR,
    gpu_process_snapshot,
    read_json,
    read_jsonl,
    sha256_json,
    write_json,
)
from run_job import validate_gpu_assignment, verify_approval


EXPERIMENT = read_json(CONFIG_DIR / "experiment.json")
GPU_IDS = tuple(EXPERIMENT["training_scope"]["gpu_ids"])
PERFORMANCE_PARALLELISM = EXPERIMENT["measurement"]["performance_parallelism"]


class ResourceBusy(RuntimeError):
    """The requested GPU set is temporarily occupied by an external process."""


class FatalLaunchError(RuntimeError):
    """The launcher or approval gate failed before a training status was written."""


class AdaptiveQueueError(RuntimeError):
    """The approved adaptive queue is malformed or cannot make safe progress."""


class RetryableAdaptiveOutcome(RuntimeError):
    """An adaptive probe did not produce the success/OOM evidence it requires."""


class CampaignBudgetExceeded(RuntimeError):
    """The approved campaign wall-clock budget no longer permits a new launch."""


ADAPTIVE_CAMPAIGN_ID = "h800_calibration_candidate_v1"
ADAPTIVE_JOB_SCHEMA = "sft_h800_calibration_job/v1"
ADAPTIVE_GPU_TYPE = "NVIDIA H800 140GB HBM3"
ADAPTIVE_GPU_NAME = "NVIDIA H800"
ADAPTIVE_HARDWARE_ID = "local_h800_140g"
ADAPTIVE_MBS_DOMAIN = (1, 2, 4, 8, 16)
ADAPTIVE_TERMINAL_OUTCOMES = frozenset({"success", "oom"})
CONDITIONAL_SKIPPED = "conditional_skipped"
EXECUTION_GATE_SCHEMA = "sft_h800_calibration_execution_gate/v1"
EXECUTION_GATE_TRANSFORM = "remove_false_authorization_add_exact_source_binding/v1"
ADAPTIVE_RUN_TIMEOUT_SECONDS = 45 * 60
ADAPTIVE_CAMPAIGN_HARD_WALL_SECONDS = 72 * 60 * 60
PROCESS_GROUP_TERM_GRACE_SECONDS = 10.0


def _canonical_jsonl_sha256(rows: list[dict[str, Any]]) -> str:
    payload = b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )
    return hashlib.sha256(payload).hexdigest()


def is_adaptive_job(job: dict[str, Any]) -> bool:
    """Return whether a row declares one of the H800 conditional protocols."""

    return "boundary_probe" in job or "packing_pair" in job


def _adaptive_metadata(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    kinds = [
        (name, job.get(name))
        for name in ("boundary_probe", "packing_pair")
        if name in job
    ]
    if len(kinds) != 1 or not isinstance(kinds[0][1], dict):
        raise AdaptiveQueueError(
            f"Adaptive job {job.get('job_id')} must declare exactly one metadata block"
        )
    return kinds[0][0], kinds[0][1]


def validate_adaptive_queue(
    jobs: list[dict[str, Any]], *, require_execution_gate: bool = False
) -> dict[str, Any] | None:
    """Compile and fail-closed validate the H800 condition dependency graph.

    The approval gate authenticates bytes; this validator gives the authenticated
    condition fields executable semantics.  A queue containing any adaptive row
    must consist entirely of the exact H800 calibration protocol so a generic or
    cross-campaign row cannot be smuggled into its dependency graph.
    """

    if not any(is_adaptive_job(job) for job in jobs):
        return None
    if not jobs or not all(is_adaptive_job(job) for job in jobs):
        raise AdaptiveQueueError(
            "Adaptive H800 queues may not mix conditional and ordinary jobs"
        )
    ids = [job.get("job_id") for job in jobs]
    if (
        not all(isinstance(job_id, str) and job_id for job_id in ids)
        or len(ids) != len(set(ids))
    ):
        raise AdaptiveQueueError("Adaptive queue job IDs must be non-empty and unique")

    by_id = {str(job["job_id"]): job for job in jobs}
    gate_presence = ["execution_gate" in job for job in jobs]
    if any(gate_presence) and not all(gate_presence):
        raise AdaptiveQueueError(
            "Adaptive execution gates must be present on every row or none"
        )
    if require_execution_gate and not all(gate_presence):
        raise AdaptiveQueueError(
            "Executing an adaptive H800 queue requires an exact execution_gate on every row"
        )
    if all(gate_presence):
        canonical_candidates: set[str] = set()
        canonical_job_sets: set[str] = set()
        reconstructed_source_rows: list[dict[str, Any]] = []
        for job in jobs:
            job_id = str(job["job_id"])
            gate = job.get("execution_gate")
            expected_fields = {
                "schema",
                "authorization_state",
                "source_candidate_canonical_sha256",
                "source_jobs_sha256",
                "source_job_payload_sha256",
                "transform_policy",
                "per_run_timeout_seconds",
                "campaign_hard_wall_time_seconds",
            }
            if not isinstance(gate, dict) or set(gate) != expected_fields:
                raise AdaptiveQueueError(
                    f"Adaptive job {job_id} has an incomplete execution gate"
                )
            canonical_candidate = gate.get("source_candidate_canonical_sha256")
            source_jobs_sha256 = gate.get("source_jobs_sha256")
            if (
                gate.get("schema") != EXECUTION_GATE_SCHEMA
                or gate.get("authorization_state")
                != "requires_exact_promoted_approval_design"
                or gate.get("transform_policy") != EXECUTION_GATE_TRANSFORM
                or not isinstance(canonical_candidate, str)
                or len(canonical_candidate) != 64
                or not isinstance(source_jobs_sha256, str)
                or len(source_jobs_sha256) != 64
                or gate.get("per_run_timeout_seconds")
                != ADAPTIVE_RUN_TIMEOUT_SECONDS
                or gate.get("campaign_hard_wall_time_seconds")
                != ADAPTIVE_CAMPAIGN_HARD_WALL_SECONDS
            ):
                raise AdaptiveQueueError(
                    f"Adaptive job {job_id} has a stale/unsupported execution gate"
                )
            source = copy.deepcopy(job)
            del source["execution_gate"]
            source["execution_authorized"] = False
            if gate.get("source_job_payload_sha256") != sha256_json(source):
                raise AdaptiveQueueError(
                    f"Adaptive job {job_id} does not match its bound source payload"
                )
            canonical_candidates.add(canonical_candidate)
            canonical_job_sets.add(source_jobs_sha256)
            reconstructed_source_rows.append(source)
        if len(canonical_candidates) != 1 or len(canonical_job_sets) != 1:
            raise AdaptiveQueueError(
                "Adaptive execution gates mix source candidates/job sets"
            )
        if _canonical_jsonl_sha256(reconstructed_source_rows) != next(
            iter(canonical_job_sets)
        ):
            raise AdaptiveQueueError(
                "Adaptive execution queue does not reconstruct its bound source JSONL"
            )
    probe_ids: dict[tuple[str, str], str] = {}
    group_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    row_metadata: dict[str, tuple[str, dict[str, Any], tuple[str, str]]] = {}
    for job in jobs:
        job_id = str(job["job_id"])
        identity = json.dumps(job, ensure_ascii=False, sort_keys=True).lower()
        if (
            job.get("schema") != ADAPTIVE_JOB_SCHEMA
            or job.get("campaign_id") != ADAPTIVE_CAMPAIGN_ID
            or job.get("gpu_type") != ADAPTIVE_GPU_TYPE
            or job.get("required_runtime_gpu_name") != ADAPTIVE_GPU_NAME
            or job.get("hardware_id") != ADAPTIVE_HARDWARE_ID
            or "4090" in identity
        ):
            raise AdaptiveQueueError(
                f"Adaptive job {job_id} is not bound to the exact H800 campaign"
            )
        gpu_count = job.get("gpu_count")
        mbs = job.get("mbs")
        if gpu_count not in {1, 2, 4} or mbs not in ADAPTIVE_MBS_DOMAIN:
            raise AdaptiveQueueError(
                f"Adaptive job {job_id} has an unsupported GPU count or MBS"
            )
        kind, metadata = _adaptive_metadata(job)
        sequence_index = metadata.get("sequence_index")
        condition = metadata.get("condition")
        if type(sequence_index) is not int or sequence_index < 0:
            raise AdaptiveQueueError(
                f"Adaptive job {job_id} has an invalid sequence index"
            )
        if not isinstance(condition, dict):
            raise AdaptiveQueueError(
                f"Adaptive job {job_id} has no structured condition"
            )
        group_field = "family_id" if kind == "boundary_probe" else "pair_id"
        group_id = metadata.get(group_field)
        if not isinstance(group_id, str) or not group_id:
            raise AdaptiveQueueError(
                f"Adaptive job {job_id} has no {group_field}"
            )
        group_key = (kind, group_id)
        group_rows.setdefault(group_key, []).append(job)
        row_metadata[job_id] = (kind, metadata, group_key)
        if kind == "boundary_probe":
            probe = metadata.get("probe")
            if not isinstance(probe, str) or not probe:
                raise AdaptiveQueueError(f"Boundary job {job_id} has no probe name")
            key = (group_id, probe)
            if key in probe_ids:
                raise AdaptiveQueueError(
                    f"Boundary family {group_id} has duplicate probe {probe}"
                )
            probe_ids[key] = job_id

    dependencies: dict[str, dict[str, str]] = {}
    for job_id, (kind, metadata, group_key) in row_metadata.items():
        condition = metadata["condition"]
        condition_type = condition.get("type")
        expected: dict[str, str]
        if condition_type == "always":
            if set(condition) != {"type"}:
                raise AdaptiveQueueError(
                    f"Adaptive job {job_id} has a malformed always condition"
                )
            expected = {}
        elif condition_type == "if_probe_outcome" and kind == "boundary_probe":
            if set(condition) != {"type", "probe", "outcome"}:
                raise AdaptiveQueueError(
                    f"Boundary job {job_id} has a malformed probe condition"
                )
            outcome = condition.get("outcome")
            target = probe_ids.get((group_key[1], str(condition.get("probe") or "")))
            if target is None or outcome not in ADAPTIVE_TERMINAL_OUTCOMES:
                raise AdaptiveQueueError(
                    f"Boundary job {job_id} references a missing or invalid probe outcome"
                )
            expected = {target: str(outcome)}
        elif condition_type == "all_jobs_succeeded" and kind == "packing_pair":
            if set(condition) != {"type", "job_ids"}:
                raise AdaptiveQueueError(
                    f"Packing job {job_id} has a malformed prerequisite condition"
                )
            targets = condition.get("job_ids")
            if (
                not isinstance(targets, list)
                or not targets
                or not all(isinstance(target, str) and target for target in targets)
                or len(targets) != len(set(targets))
            ):
                raise AdaptiveQueueError(
                    f"Packing job {job_id} has absent or duplicate prerequisites"
                )
            expected = {target: "success" for target in targets}
        else:
            raise AdaptiveQueueError(
                f"Adaptive job {job_id} has unsupported condition {condition_type!r}"
            )

        current_index = int(metadata["sequence_index"])
        for target in expected:
            target_row = row_metadata.get(target)
            if target_row is None:
                raise AdaptiveQueueError(
                    f"Adaptive job {job_id} depends on missing/cross-candidate job {target}"
                )
            _, target_metadata, target_group = target_row
            if target_group != group_key:
                raise AdaptiveQueueError(
                    f"Adaptive job {job_id} has a cross-family dependency on {target}"
                )
            if int(target_metadata["sequence_index"]) >= current_index:
                raise AdaptiveQueueError(
                    f"Adaptive job {job_id} depends on a non-prior job {target}"
                )
        dependencies[job_id] = expected

    # Validate the exact boundary and ABBA protocols, not merely an acyclic DAG.
    for (kind, group_id), rows in group_rows.items():
        rows.sort(key=lambda row: int(row[kind]["sequence_index"]))
        indices = [int(row[kind]["sequence_index"]) for row in rows]
        if indices != list(range(len(rows))):
            raise AdaptiveQueueError(
                f"Adaptive group {group_id} sequence indices are not contiguous"
            )
        if kind == "boundary_probe":
            anchor_rows = [row for row in rows if row[kind].get("probe") == "anchor"]
            if len(anchor_rows) != 1 or rows[0] is not anchor_rows[0]:
                raise AdaptiveQueueError(
                    f"Boundary family {group_id} must start with exactly one anchor"
                )
            anchor = anchor_rows[0]
            anchor_id = str(anchor["job_id"])
            anchor_mbs = int(anchor["mbs"])
            if dependencies[anchor_id]:
                raise AdaptiveQueueError(
                    f"Boundary family {group_id} anchor must be unconditional"
                )
            fallback = [
                row for row in rows if row[kind].get("probe") == "fallback_half"
            ]
            if anchor_mbs > 1:
                if (
                    len(fallback) != 1
                    or int(fallback[0]["mbs"]) * 2 != anchor_mbs
                    or dependencies[str(fallback[0]["job_id"])] != {anchor_id: "oom"}
                ):
                    raise AdaptiveQueueError(
                        f"Boundary family {group_id} has an invalid halving fallback"
                    )
            elif fallback:
                raise AdaptiveQueueError(
                    f"Boundary family {group_id} may not halve an MBS=1 anchor"
                )
            upward = []
            for row in rows:
                probe = str(row[kind].get("probe") or "")
                if probe.startswith("upward_"):
                    try:
                        upward.append((int(probe.removeprefix("upward_")), row))
                    except ValueError as error:
                        raise AdaptiveQueueError(
                            f"Boundary family {group_id} has an invalid upward probe name"
                        ) from error
                elif probe not in {"anchor", "fallback_half"}:
                    raise AdaptiveQueueError(
                        f"Boundary family {group_id} has unknown probe {probe!r}"
                    )
            upward.sort(key=lambda item: item[0])
            if [index for index, _ in upward] != list(range(1, len(upward) + 1)):
                raise AdaptiveQueueError(
                    f"Boundary family {group_id} upward probes are not contiguous"
                )
            previous = anchor
            for _, row in upward:
                row_id = str(row["job_id"])
                previous_id = str(previous["job_id"])
                if (
                    int(row["mbs"]) != int(previous["mbs"]) * 2
                    or dependencies[row_id] != {previous_id: "success"}
                ):
                    raise AdaptiveQueueError(
                        f"Boundary family {group_id} does not implement first-OOM stopping"
                    )
                previous = row
            expected_upward_mbs = []
            candidate_mbs = anchor_mbs * 2
            while candidate_mbs <= ADAPTIVE_MBS_DOMAIN[-1]:
                expected_upward_mbs.append(candidate_mbs)
                candidate_mbs *= 2
            if [int(row["mbs"]) for _, row in upward] != expected_upward_mbs:
                raise AdaptiveQueueError(
                    f"Boundary family {group_id} does not terminate exactly at MBS=16"
                )
        else:
            if len(rows) != 4:
                raise AdaptiveQueueError(f"Packing pair {group_id} is not a full ABBA")
            treatments = [row[kind].get("treatment") for row in rows]
            if treatments != ["unpacked", "packed", "packed", "unpacked"]:
                raise AdaptiveQueueError(
                    f"Packing pair {group_id} treatment order is not ABBA"
                )
            prior_ids: list[str] = []
            for index, row in enumerate(rows):
                row_id = str(row["job_id"])
                expected = {prior: "success" for prior in prior_ids}
                if dependencies[row_id] != expected:
                    raise AdaptiveQueueError(
                        f"Packing pair {group_id} row {index} does not require all prior successes"
                    )
                prior_ids.append(row_id)

    # Kahn's algorithm is intentionally retained even though prior-index checks
    # already exclude cycles: it fails closed if a future condition type changes.
    unresolved = {job_id: set(expected) for job_id, expected in dependencies.items()}
    completed: set[str] = set()
    while len(completed) < len(unresolved):
        ready = {
            job_id
            for job_id, prerequisites in unresolved.items()
            if job_id not in completed and prerequisites <= completed
        }
        if not ready:
            raise AdaptiveQueueError("Adaptive condition graph contains a cycle")
        completed.update(ready)
    return {
        "schema": "h800_adaptive_scheduler_plan/v1",
        "jobs": by_id,
        "dependencies": dependencies,
        "groups": group_rows,
        "execution_gate_required": require_execution_gate,
        "per_run_timeout_seconds": (
            ADAPTIVE_RUN_TIMEOUT_SECONDS if all(gate_presence) else None
        ),
        "campaign_hard_wall_time_seconds": (
            ADAPTIVE_CAMPAIGN_HARD_WALL_SECONDS if all(gate_presence) else None
        ),
    }


def adaptive_job_state(
    job_id: str,
    plan: dict[str, Any],
    outcomes: dict[str, str],
) -> tuple[str, str | None]:
    """Return ready/wait/skipped without treating a non-terminal value as ready."""

    expected = plan["dependencies"][job_id]
    for dependency, required in expected.items():
        observed = outcomes.get(dependency)
        if observed is None:
            return "wait", None
        if observed not in {*ADAPTIVE_TERMINAL_OUTCOMES, CONDITIONAL_SKIPPED}:
            raise AdaptiveQueueError(
                f"Dependency {dependency} has non-terminal outcome {observed!r}"
            )
        if observed != required:
            return (
                "skipped",
                f"prerequisite {dependency}={observed}, required={required}",
            )
    return "ready", None


def record_conditional_skip(
    job: dict[str, Any],
    event_path: Path,
    *,
    execution_id: str,
    reason: str,
) -> None:
    """Persist scheduler-terminal skip evidence outside training status.json."""

    payload = {
        "schema": "h800_adaptive_scheduler_terminal/v1",
        "job_id": job["job_id"],
        "scheduler_execution_id": execution_id,
        "classification": CONDITIONAL_SKIPPED,
        "terminal": True,
        "training_started": False,
        "calibration_observation_eligible": False,
        "reason": reason,
        "time_unix": time.time(),
        "training_status_json_was_not_written": True,
    }
    # The H800 exporter reads only status.json, so this explicit scheduler state
    # can never become a calibration observation.
    write_json(RESULTS_DIR / str(job["job_id"]) / "scheduler_terminal.json", payload)
    append_event(event_path, {"event": "family_skipped", **payload})


def validate_jobs_in_gpu_scope(jobs: list[dict[str, Any]]) -> None:
    for job in jobs:
        count = int(job["gpu_count"])
        validate_gpu_assignment(job, list(GPU_IDS[:count]), EXPERIMENT)


def requires_exclusive_pool(job: dict[str, Any]) -> bool:
    count = int(job["gpu_count"])
    disjoint_wave_large_job = bool(
        4 <= count <= 7
        and job.get("allow_disjoint_wave_for_large_job") is True
        and job.get("parallel_class") == "disjoint_wave"
        and job.get("requires_external_node_idle") is False
        and PERFORMANCE_PARALLELISM == "disjoint_gpu_masks"
    )
    available_pool_large_job = bool(
        4 <= count <= 7
        and job.get("allow_available_pool_for_large_job") is True
        and job.get("parallel_class") == "available_pool"
        and job.get("requires_external_node_idle") is False
        and PERFORMANCE_PARALLELISM == "disjoint_gpu_masks"
    )
    return bool(
        not (available_pool_large_job or disjoint_wave_large_job)
        and (
            job.get("requires_external_node_idle") is True
        # Jobs using four or more cards are deliberately pool-exclusive.  This
        # preserves comparable measurements and routes 4/5/6/7/8-card jobs
        # through ``allocate``'s generic exclusive branch; the partitionable
        # branch only packs 1/2-card jobs.
            or count >= 4
            or PERFORMANCE_PARALLELISM == "exclusive_pool"
        )
    )


def strict_queue_order(jobs: list[dict[str, Any]]) -> bool:
    """Fail closed if only part of a queue requests strict sequential order."""

    flags = [job.get("strict_queue_order") is True for job in jobs]
    if any(flags) and not all(flags):
        raise ValueError("strict_queue_order must be declared by every queue row")
    return bool(flags) and all(flags)


def allocate(
    job: dict[str, Any], available: set[int], running_count: int
) -> list[int] | None:
    count = int(job["gpu_count"])
    if requires_exclusive_pool(job):
        if running_count != 0 or available != set(GPU_IDS):
            return None
        return list(GPU_IDS[:count])
    if 4 <= count <= 7:
        if (
            job.get("allow_disjoint_wave_for_large_job") is True
            and job.get("parallel_class") == "disjoint_wave"
            and PERFORMANCE_PARALLELISM == "disjoint_gpu_masks"
        ):
            return sorted(available)[:count] if len(available) >= count else None
        if (
            running_count != 0
            or job.get("allow_available_pool_for_large_job") is not True
            or len(available) < count
        ):
            return None
        return sorted(available)[:count]
    if count == 2:
        for start in range(0, len(GPU_IDS) - 1, 2):
            pair = GPU_IDS[start : start + 2]
            if set(pair) <= available:
                return list(pair)
        return None
    if count == 1 and available:
        return [min(available)]
    return None


def preview_waves(jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    pending = jobs.copy()
    waves: list[list[dict[str, Any]]] = []
    preserve_order = strict_queue_order(jobs)
    while pending:
        available = set(GPU_IDS)
        wave: list[dict[str, Any]] = []
        # Fill partitionable waves with 2-card jobs first, then singles.
        candidates = list(enumerate(pending))
        if not preserve_order:
            candidates.sort(
                key=lambda item: (
                    requires_exclusive_pool(item[1]),
                    -int(item[1]["gpu_count"]),
                )
            )
        selected_indices = []
        for index, job in candidates:
            active_counts = {
                int(row["gpu_count"])
                for row in wave
                if row.get("homogeneous_card_count_wave") is True
            }
            if active_counts and (
                job.get("homogeneous_card_count_wave") is not True
                or int(job["gpu_count"]) not in active_counts
            ):
                continue
            mask = allocate(job, available, len(wave))
            if mask is None:
                continue
            available -= set(mask)
            wave.append(
                {
                    "job_id": job["job_id"],
                    "gpu_mask": mask,
                    "gpu_count": job["gpu_count"],
                    "homogeneous_card_count_wave": job.get(
                        "homogeneous_card_count_wave"
                    )
                    is True,
                }
            )
            selected_indices.append(index)
            if not available:
                break
        if not wave:
            raise RuntimeError("Scheduler could not place a pending job")
        for index in sorted(selected_indices, reverse=True):
            pending.pop(index)
        waves.append(wave)
    return waves


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(
            json.dumps(
                {"time_unix": time.time(), **event}, ensure_ascii=False, sort_keys=True
            )
            + "\n"
        )


def dashboard_phase(job: dict[str, Any]) -> str:
    """Stable phase label consumed by the read-only live dashboard."""
    job_id = str(job.get("job_id") or "")
    kind = str(job.get("kind") or "")
    if kind in {"memory_boundary", "memory_probe"} or job_id.startswith("mem-"):
        return "memory"
    if kind == "packing_memory_probe" or job_id.startswith(
        ("packmem-", "packon-", "packoff-")
    ):
        return "packing"
    if kind == "profiler" or job_id.startswith("prof-"):
        return "profiler"
    if job_id.startswith("scale-"):
        return "scaling"
    return "throughput"


def adaptive_run_timeout_seconds(job: dict[str, Any]) -> int | None:
    """Return the approved timeout, rejecting a conditional row without one."""

    if not is_adaptive_job(job):
        return None
    gate = job.get("execution_gate")
    if not isinstance(gate, dict):
        raise AdaptiveQueueError(
            f"Adaptive job {job.get('job_id')} has no execution timeout gate"
        )
    timeout = gate.get("per_run_timeout_seconds")
    if timeout != ADAPTIVE_RUN_TIMEOUT_SECONDS:
        raise AdaptiveQueueError(
            f"Adaptive job {job.get('job_id')} timeout is not the approved "
            f"{ADAPTIVE_RUN_TIMEOUT_SECONDS}s"
        )
    return int(timeout)


async def _terminate_process_group(
    process: asyncio.subprocess.Process,
    communication: "asyncio.Task[tuple[bytes, bytes | None]]",
    *,
    grace_seconds: float = PROCESS_GROUP_TERM_GRACE_SECONDS,
) -> tuple[bytes, bytes | None]:
    """TERM then KILL the launcher's process group and reap its output."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return await asyncio.wait_for(
            asyncio.shield(communication), timeout=grace_seconds
        )
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return await communication


async def communicate_with_process_group_timeout(
    process: asyncio.subprocess.Process,
    *,
    timeout_seconds: int | None,
) -> tuple[bytes, bytes | None, bool]:
    """Communicate with a bounded process group; ``timed_out`` is explicit."""

    communication = asyncio.create_task(process.communicate())
    if timeout_seconds is None:
        stdout, stderr = await communication
        return stdout, stderr, False
    try:
        done, _ = await asyncio.wait({communication}, timeout=timeout_seconds)
        if communication in done:
            stdout, stderr = communication.result()
            return stdout, stderr, False
        stdout, stderr = await _terminate_process_group(process, communication)
        return stdout, stderr, True
    except asyncio.CancelledError:
        # Scheduler hard-deadline cancellation must never orphan torchrun or a
        # DeepSpeed rank.  Reap the complete process group before propagating.
        await _terminate_process_group(process, communication)
        raise


async def execute_concrete(
    job: dict[str, Any], gpu_mask: list[int], event_path: Path
) -> str:
    job_path = RUNTIME_DIR / "jobs" / f"input-{job['job_id']}.json"
    write_json(job_path, job)
    command = [
        EXPERIMENT["fixed_runtime"]["python"],
        str(ROOT / "scripts" / "run_job.py"),
        "--job-file",
        str(job_path),
        "--gpu-mask",
        ",".join(map(str, gpu_mask)),
        "--execute",
    ]
    append_event(
        event_path,
        {"event": "trial_start", "job_id": job["job_id"], "gpu_mask": gpu_mask},
    )
    launched_at = time.time()
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # torchrun/deepspeed descendants share this process group.  Timeout
        # cleanup can therefore terminate the complete launch tree.
        start_new_session=True,
    )
    try:
        stdout, _, timed_out = await communicate_with_process_group_timeout(
            process,
            timeout_seconds=adaptive_run_timeout_seconds(job),
        )
    except asyncio.CancelledError:
        append_event(
            event_path,
            {
                "event": "trial_end",
                "job_id": job["job_id"],
                "gpu_mask": gpu_mask,
                "return_code": process.returncode,
                "classification": "campaign_budget_exceeded",
                "timed_out": False,
                "campaign_deadline_cancelled": True,
                "process_group_reaped": True,
                "per_run_timeout_seconds": adaptive_run_timeout_seconds(job),
            },
        )
        raise
    stdout_text = stdout.decode(errors="replace")
    status_path = RESULTS_DIR / job["job_id"] / "status.json"
    if timed_out:
        classification = "timeout"
    elif "Refusing to start: selected GPUs have compute processes" in stdout_text:
        classification = "resource_busy"
    elif any(
        marker in stdout_text
        for marker in (
            "Frozen design is stale",
            "Approval does not match the current frozen design",
            "Training is locked",
            "outside the approval scope",
            "Approval gate rejected",
            "Approval/design",
            "approval execution gate is busy",
            "canonical payload does not match",
        )
    ):
        classification = "approval_rejected"
    elif status_path.exists() and status_path.stat().st_mtime >= launched_at:
        classification = read_json(status_path)["classification"]
    else:
        classification = "launcher_failed"
    append_event(
        event_path,
        {
            "event": "trial_end",
            "job_id": job["job_id"],
            "gpu_mask": gpu_mask,
            "return_code": process.returncode,
            "classification": classification,
            "timed_out": timed_out,
            "per_run_timeout_seconds": adaptive_run_timeout_seconds(job),
            "launcher_output_tail": stdout_text[-2000:],
        },
    )
    return classification


async def execute_family(
    family: dict[str, Any], gpu_mask: list[int], event_path: Path
) -> str:
    append_event(
        event_path,
        {
            "event": "family_start",
            "job_id": family["job_id"],
            "phase": dashboard_phase(family),
            "gpu_mask": gpu_mask,
            "candidate_mbs": family.get("mbs_candidates"),
        },
    )
    if family["kind"] != "memory_boundary":
        classification = await execute_concrete(family, gpu_mask, event_path)
        if classification == "resource_busy":
            raise ResourceBusy(f"GPU mask {gpu_mask} became busy")
        if classification in {"approval_rejected", "launcher_failed"}:
            raise FatalLaunchError(
                f"Job {family['job_id']} could not pass the launcher gate ({classification})"
            )
        if is_adaptive_job(family) and classification not in ADAPTIVE_TERMINAL_OUTCOMES:
            raise RetryableAdaptiveOutcome(
                f"Adaptive job {family['job_id']} produced {classification!r}; "
                "only success/OOM is terminal, so the same job remains retryable "
                "and every successor remains blocked"
            )
        append_event(
            event_path,
            {
                "event": "family_end",
                "job_id": family["job_id"],
                "phase": dashboard_phase(family),
                "classification": classification,
            },
        )
        return classification
    trial_results = []
    for mbs in family["mbs_candidates"]:
        concrete = copy.deepcopy(family)
        concrete["kind"] = "memory_probe"
        concrete["family_job_id"] = family["job_id"]
        concrete["job_id"] = f"{family['job_id']}-mbs{mbs}"
        concrete["mbs"] = mbs
        classification = await execute_concrete(concrete, gpu_mask, event_path)
        if classification == "resource_busy":
            raise ResourceBusy(f"GPU mask {gpu_mask} became busy")
        if classification in {"approval_rejected", "launcher_failed"}:
            raise FatalLaunchError(
                f"Job {concrete['job_id']} could not pass the launcher gate ({classification})"
            )
        trial_results.append({"mbs": mbs, "classification": classification})
        if classification == "oom":
            break
        if classification != "success":
            raise RuntimeError(
                f"Family {family['job_id']} trial MBS={mbs} failed as {classification}; refusing to label a boundary"
            )
    successful = [
        row["mbs"] for row in trial_results if row["classification"] == "success"
    ]
    summary = {
        "family_job_id": family["job_id"],
        "gpu_mask": gpu_mask,
        "trials": trial_results,
        "max_feasible_mbs": max(successful) if successful else None,
        "first_failed_mbs": next(
            (row["mbs"] for row in trial_results if row["classification"] != "success"),
            None,
        ),
    }
    write_json(RESULTS_DIR / "boundary_summaries" / f"{family['job_id']}.json", summary)
    append_event(
        event_path,
        {
            "event": "family_end",
            "job_id": family["job_id"],
            "phase": "memory",
            "classification": "boundary_found"
            if summary["max_feasible_mbs"] is not None
            else "infeasible",
            "max_feasible_mbs": summary["max_feasible_mbs"],
            "first_failed_mbs": summary["first_failed_mbs"],
        },
    )
    return str(
        "boundary_found" if summary["max_feasible_mbs"] is not None else "infeasible"
    )


def occupied_gpu_ids() -> set[int]:
    snapshot = gpu_process_snapshot(GPU_IDS)
    return {int(row["gpu_index"]) for row in snapshot["processes"]}


def refresh_pool_state(
    available: set[int],
    blocked: set[int],
    running: dict[asyncio.Task[None], tuple[dict[str, Any], list[int]]],
) -> tuple[set[int], set[int]]:
    managed = {gpu for _, mask in running.values() for gpu in mask}
    externally_busy = occupied_gpu_ids() - managed
    refreshed_blocked = externally_busy & set(GPU_IDS)
    refreshed_available = (set(GPU_IDS) - managed) - refreshed_blocked
    return refreshed_available, refreshed_blocked


async def run_scheduler(
    jobs: list[dict[str, Any]],
    event_path: Path,
    initial_blocked: set[int] | None = None,
    busy_poll_seconds: float = 5.0,
    execution_id: str | None = None,
    require_execution_gate: bool = False,
    monotonic_clock: Any = time.monotonic,
    wall_clock: Any = time.time,
    campaign_deadline_unix: float | None = None,
) -> None:
    preserve_order = strict_queue_order(jobs)
    adaptive_plan = validate_adaptive_queue(
        jobs, require_execution_gate=require_execution_gate
    )
    adaptive_execution_id = execution_id or f"scheduler-{time.time_ns()}"
    adaptive_outcomes: dict[str, str] = {}
    campaign_started_monotonic = monotonic_clock()
    campaign_budget_seconds = (
        adaptive_plan.get("campaign_hard_wall_time_seconds")
        if adaptive_plan is not None
        else None
    )
    if (
        require_execution_gate
        and campaign_budget_seconds is not None
        and campaign_deadline_unix is None
    ):
        raise AdaptiveQueueError(
            "Executing an adaptive queue requires the approval-derived absolute "
            "campaign deadline"
        )
    remaining_at_start = campaign_budget_seconds
    if campaign_budget_seconds is not None and campaign_deadline_unix is not None:
        remaining_at_start = min(
            float(campaign_budget_seconds),
            max(0.0, float(campaign_deadline_unix) - float(wall_clock())),
        )
    campaign_deadline_monotonic = (
        campaign_started_monotonic + float(remaining_at_start)
        if remaining_at_start is not None
        else None
    )
    campaign_budget_reported = False
    pending = jobs.copy()
    running: dict[asyncio.Task[None], tuple[dict[str, Any], list[int]]] = {}
    blocked = set(initial_blocked or ())
    available = set(GPU_IDS) - blocked
    fatal_error: BaseException | None = None
    while pending or running:
        now_monotonic = monotonic_clock()
        campaign_elapsed = (
            float(campaign_budget_seconds)
            - max(0.0, campaign_deadline_monotonic - now_monotonic)
            if campaign_budget_seconds is not None
            and campaign_deadline_monotonic is not None
            else now_monotonic - campaign_started_monotonic
        )
        if (
            campaign_deadline_monotonic is not None
            and now_monotonic >= campaign_deadline_monotonic
        ):
            budget_error = CampaignBudgetExceeded(
                f"H800 campaign reached its approved {campaign_budget_seconds}s "
                "hard wall-clock deadline"
            )
            if not campaign_budget_reported:
                append_event(
                    event_path,
                    {
                        "event": "campaign_budget_exhausted",
                        "execution_id": adaptive_execution_id,
                        "elapsed_seconds": campaign_elapsed,
                        "hard_wall_time_seconds": campaign_budget_seconds,
                        "approval_derived_deadline_unix": campaign_deadline_unix,
                        "pending": len(pending),
                        "running": len(running),
                        "new_launches_disabled": True,
                        "running_jobs_cancelled_at_deadline": True,
                        "process_group_cleanup": "SIGTERM_then_SIGKILL_and_reap",
                    },
                )
                campaign_budget_reported = True
            cancelled = list(running.items())
            for task, _ in cancelled:
                task.cancel()
            if cancelled:
                await asyncio.gather(
                    *(task for task, _ in cancelled), return_exceptions=True
                )
                for task, (job, mask) in cancelled:
                    running.pop(task, None)
                    available.update(mask)
                    append_event(
                        event_path,
                        {
                            "event": "family_cancelled",
                            "job_id": job["job_id"],
                            "gpu_mask": mask,
                            "classification": "campaign_budget_exceeded",
                            "terminal_calibration_outcome": False,
                            "retryable_under_a_new_approval": True,
                        },
                    )
            append_event(
                event_path,
                {
                    "event": "scheduler_halted",
                    "pending": len(pending),
                    "running": 0,
                    "error": repr(budget_error),
                    "campaign_hard_deadline_enforced": True,
                },
            )
            raise budget_error
        available, blocked = refresh_pool_state(available, blocked, running)
        if adaptive_plan is not None and fatal_error is None:
            # A mismatch is terminal for the conditional row, but is not a
            # training failure and does not create a calibration observation.
            changed = True
            while changed:
                changed = False
                for job in list(pending):
                    state, reason = adaptive_job_state(
                        str(job["job_id"]), adaptive_plan, adaptive_outcomes
                    )
                    if state != "skipped":
                        continue
                    pending.remove(job)
                    adaptive_outcomes[str(job["job_id"])] = CONDITIONAL_SKIPPED
                    record_conditional_skip(
                        job,
                        event_path,
                        execution_id=adaptive_execution_id,
                        reason=str(reason),
                    )
                    changed = True
            if not pending and not running:
                break
        launched = False
        if fatal_error is None:
            active_homogeneous_counts = {
                int(current_job["gpu_count"])
                for current_job, _ in running.values()
                if current_job.get("homogeneous_card_count_wave") is True
            }
            if len(active_homogeneous_counts) > 1:
                raise RuntimeError(
                    "homogeneous-card-count queue is running mixed GPU counts"
                )
            eligible = [
                (index, job)
                for index, job in enumerate(pending)
                if (
                    adaptive_plan is None
                    or adaptive_job_state(
                        str(job["job_id"]), adaptive_plan, adaptive_outcomes
                    )[0]
                    == "ready"
                )
                and (
                    not active_homogeneous_counts
                    or (
                        job.get("homogeneous_card_count_wave") is True
                        and int(job["gpu_count"])
                        in active_homogeneous_counts
                    )
                )
            ]
            candidates = list(eligible)
            if not preserve_order:
                candidates.sort(
                    key=lambda item: (
                        requires_exclusive_pool(item[1]),
                        -int(item[1]["gpu_count"]),
                    )
                )
            for index, job in candidates:
                mask = allocate(job, available, len(running))
                if mask is None:
                    continue
                available -= set(mask)
                pending.pop(index)
                task = asyncio.create_task(execute_family(job, mask, event_path))
                running[task] = (job, mask)
                append_event(
                    event_path,
                    {
                        "event": "queue_updated",
                        "pending": len(pending),
                        "running": len(running),
                        "available_gpus": sorted(available),
                        "blocked_gpus": sorted(blocked),
                    },
                )
                launched = True
                break
        if launched:
            continue
        if not running:
            if fatal_error is not None:
                append_event(
                    event_path,
                    {
                        "event": "scheduler_halted",
                        "pending": len(pending),
                        "error": repr(fatal_error),
                    },
                )
                raise fatal_error
            if adaptive_plan is not None and pending and not eligible:
                waiting = {
                    str(job["job_id"]): adaptive_plan["dependencies"][str(job["job_id"])]
                    for job in pending
                }
                error = AdaptiveQueueError(
                    "Adaptive queue cannot make progress because prerequisites are "
                    f"non-terminal: {waiting}"
                )
                append_event(
                    event_path,
                    {
                        "event": "scheduler_halted",
                        "pending": len(pending),
                        "error": repr(error),
                    },
                )
                raise error
            if pending and blocked:
                append_event(
                    event_path,
                    {
                        "event": "pool_wait",
                        "pending": len(pending),
                        "blocked_gpus": sorted(blocked),
                    },
                )
                sleep_seconds = busy_poll_seconds
                if campaign_budget_seconds is not None:
                    remaining = max(
                        0.0,
                        campaign_deadline_monotonic - monotonic_clock(),
                    )
                    sleep_seconds = min(sleep_seconds, remaining)
                await asyncio.sleep(sleep_seconds)
                continue
            raise RuntimeError("No runnable job and no running work")
        wait_timeout = busy_poll_seconds if blocked else None
        if campaign_budget_seconds is not None:
            remaining = max(
                0.0,
                campaign_deadline_monotonic - monotonic_clock(),
            )
            wait_timeout = (
                remaining if wait_timeout is None else min(wait_timeout, remaining)
            )
        done, _ = await asyncio.wait(
            running,
            timeout=wait_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            continue
        for task in done:
            job, mask = running.pop(task)
            available.update(mask)
            try:
                classification = task.result()
                if adaptive_plan is not None:
                    if classification not in ADAPTIVE_TERMINAL_OUTCOMES:
                        raise AdaptiveQueueError(
                            f"Adaptive job {job['job_id']} returned non-terminal "
                            f"classification {classification!r}"
                        )
                    adaptive_outcomes[str(job["job_id"])] = classification
            except ResourceBusy as error:
                pending.append(job)
                blocked.update(mask)
                available.difference_update(mask)
                append_event(
                    event_path,
                    {
                        "event": "family_requeued",
                        "job_id": job["job_id"],
                        "reason": str(error),
                    },
                )
            except FatalLaunchError as error:
                pending.append(job)
                fatal_error = error
                append_event(
                    event_path,
                    {
                        "event": "family_requeued",
                        "job_id": job["job_id"],
                        "reason": str(error),
                        "fatal": True,
                    },
                )
            except Exception as error:
                pending.append(job)
                fatal_error = error
                append_event(
                    event_path,
                    {
                        "event": "family_exception",
                        "job_id": job["job_id"],
                        "error": repr(error),
                        "requeued": True,
                        "fatal": True,
                    },
                )
            append_event(
                event_path,
                {
                    "event": "queue_updated",
                    "pending": len(pending),
                    "running": len(running),
                    "available_gpus": sorted(available),
                    "blocked_gpus": sorted(blocked),
                },
            )


def main(*, _execution_gate_held: bool = False) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=MATRIX_DIR / "memory_boundary_families.jsonl"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Requires the separately created approval file",
    )
    parser.add_argument(
        "--join-busy-pool",
        action="store_true",
        help="Start on currently idle GPUs in the approved pool and add busy pool GPUs when they become idle",
    )
    parser.add_argument("--busy-poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    jobs = read_jsonl(args.input)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        raise ValueError("No jobs to schedule")
    validate_jobs_in_gpu_scope(jobs)
    # Compile conditions before preview, approval verification, or any GPU
    # allocation so malformed/cross-candidate DAGs fail closed without launch.
    adaptive_plan = validate_adaptive_queue(
        jobs, require_execution_gate=args.execute
    )

    if not args.execute:
        waves = preview_waves(jobs)
        preview = {
            "training_started": False,
            "jobs": len(jobs),
            "waves": len(waves),
            "parallel_policy": {
                "memory": (
                    "pack all approved disjoint 1-card/2-card masks; "
                    f"the {len(GPU_IDS)}-GPU pool supports up to "
                    f"{len(GPU_IDS) // 2} 2-card jobs"
                ),
                "formal_throughput": "pack disjoint GPU masks concurrently; only 4-GPU jobs are pool-exclusive",
                "selected_gpu_ids": list(GPU_IDS),
                "four_card": "always exclusive",
            },
            "first_20_waves": waves[:20],
        }
        write_json(MATRIX_DIR / "schedule_preview.json", preview)
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    if not _execution_gate_held:
        # Re-read and revalidate the queue under the process-lifetime shared
        # gate.  Promotion cannot switch approval/design until this scheduler
        # returns, including exceptional exits.
        with execution_lock(RUNTIME_DIR, exclusive=False):
            return main(_execution_gate_held=True)
    approval_evidence = verify_approval(
        queue_path=args.input,
        queue_rows=jobs,
        acquire_lock=False,
    )
    approval_design = read_json(Path(approval_evidence["approval_design_path"]))
    approved_join_busy_pool = (
        (approval_design.get("scheduler_execution") or {}).get("join_busy_pool")
        is True
    )
    if args.join_busy_pool is not approved_join_busy_pool:
        raise PermissionError(
            "Scheduler --join-busy-pool mode must exactly match the frozen approval design"
        )
    campaign_deadline_unix: float | None = None
    if adaptive_plan is not None:
        approved_unix = (approval_evidence.get("approval") or {}).get(
            "approved_unix"
        )
        if not isinstance(approved_unix, (int, float)) or isinstance(
            approved_unix, bool
        ):
            raise PermissionError(
                "Adaptive H800 approval lacks its absolute approval timestamp"
            )
        campaign_deadline_unix = float(approved_unix) + float(
            adaptive_plan["campaign_hard_wall_time_seconds"]
        )
    initial_blocked = occupied_gpu_ids()
    if initial_blocked and not args.join_busy_pool:
        snapshot = gpu_process_snapshot(GPU_IDS)
        raise RuntimeError(
            f"Refusing to start: selected GPUs have compute processes: {snapshot['processes']}"
        )
    event_path = RUNTIME_DIR / "scheduler_events.jsonl"
    execution_id = f"scheduler-{int(time.time())}"
    append_event(
        event_path,
        {
            "event": "scheduler_start",
            "execution_id": execution_id,
            "input": str(args.input),
            "jobs": len(jobs),
            "phases": sorted({dashboard_phase(job) for job in jobs}),
            "initial_blocked_gpus": sorted(initial_blocked),
            "join_busy_pool": args.join_busy_pool,
        },
    )
    asyncio.run(
        run_scheduler(
            jobs,
            event_path,
            initial_blocked=initial_blocked,
            busy_poll_seconds=args.busy_poll_seconds,
            execution_id=execution_id,
            require_execution_gate=True,
            campaign_deadline_unix=campaign_deadline_unix,
        )
    )
    append_event(
        event_path,
        {
            "event": "scheduler_complete",
            "execution_id": execution_id,
            "input": str(args.input),
            "jobs": len(jobs),
        },
    )


if __name__ == "__main__":
    main()
