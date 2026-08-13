#!/usr/bin/env python3
"""Build deterministic GPU-full scheduling plans without launching jobs.

The planner is deliberately independent from ``offline_experiments`` so it can
be reused by future campaigns without changing the source fingerprint of a
running experiment.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


PLAN_SCHEMA = "full_gpu_schedule_plan/v1"
STRATEGY_SCHEMA = "full_gpu_schedule_strategy/v1"


class SchedulingError(ValueError):
    """The requested policy cannot produce a valid schedule."""


@dataclass(frozen=True)
class GPUContext:
    available_gpu_ids: tuple[int, ...]
    selected_gpu_ids: tuple[int, ...]
    gpu_limit: int


def parse_gpu_ids(value: str) -> tuple[int, ...]:
    """Parse ``0,2,4-7`` into a deterministic tuple of GPU IDs."""

    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start < 0 or end < start:
                raise SchedulingError(f"invalid GPU range: {token!r}")
            result.extend(range(start, end + 1))
        else:
            current = int(token)
            if current < 0:
                raise SchedulingError("GPU IDs must be non-negative")
            result.append(current)
    if not result:
        raise SchedulingError("at least one GPU ID is required")
    if len(result) != len(set(result)):
        raise SchedulingError("GPU IDs must be unique")
    return tuple(result)


def resolve_gpu_context(
    *, gpu_ids: Sequence[int] | None, gpu_limit: int | None
) -> GPUContext:
    """Resolve the available pool and select at most ``gpu_limit`` cards."""

    if gpu_ids is None:
        if gpu_limit is None or gpu_limit <= 0:
            raise SchedulingError(
                "provide a positive --gpu-limit, or provide --gpu-ids"
            )
        available = tuple(range(gpu_limit))
    else:
        available = tuple(gpu_ids)
        if not available or len(available) != len(set(available)):
            raise SchedulingError("available GPU IDs must be non-empty and unique")
        if any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in available):
            raise SchedulingError("available GPU IDs must be non-negative integers")
    limit = len(available) if gpu_limit is None else gpu_limit
    if type(limit) is not int or limit <= 0 or limit > len(available):
        raise SchedulingError(
            f"gpu_limit must be in [1, {len(available)}], got {limit!r}"
        )
    return GPUContext(
        available_gpu_ids=available,
        selected_gpu_ids=available[:limit],
        gpu_limit=limit,
    )


def _prefer_combination(
    candidate: tuple[int, ...], current: tuple[int, ...] | None
) -> bool:
    if current is None:
        return True
    candidate_key = (len(candidate), tuple(-size for size in sorted(candidate, reverse=True)))
    current_key = (len(current), tuple(-size for size in sorted(current, reverse=True)))
    return candidate_key < current_key


def unbounded_fill(target: int, job_sizes: Iterable[int]) -> tuple[int, ...] | None:
    """Return a deterministic exact fill using reusable sizes, if one exists."""

    if target < 0:
        return None
    sizes = tuple(sorted({int(size) for size in job_sizes if int(size) > 0}, reverse=True))
    if target == 0:
        return ()
    if not sizes:
        return None
    plans: list[tuple[int, ...] | None] = [None] * (target + 1)
    plans[0] = ()
    for used in range(target + 1):
        current = plans[used]
        if current is None:
            continue
        for size in sizes:
            destination = used + size
            if destination > target:
                continue
            candidate = current + (size,)
            if _prefer_combination(candidate, plans[destination]):
                plans[destination] = candidate
    return plans[target]


def build_strategy(
    context: GPUContext, job_sizes: Iterable[int]
) -> dict[str, Any]:
    """Describe homogeneous concurrency and an unbounded mixed exact fill."""

    sizes = tuple(sorted({int(size) for size in job_sizes}))
    if not sizes or any(size <= 0 for size in sizes):
        raise SchedulingError("job sizes must be positive integers")
    homogeneous = []
    supported = []
    for size in sizes:
        concurrency = context.gpu_limit // size
        used = concurrency * size
        if concurrency:
            supported.append(size)
        homogeneous.append(
            {
                "job_gpu_count": size,
                "supported": concurrency > 0,
                "max_concurrency": concurrency,
                "used_gpu_count": used,
                "idle_gpu_count": context.gpu_limit - used,
                "full_utilization": used == context.gpu_limit,
            }
        )
    mixed = unbounded_fill(context.gpu_limit, supported)
    divisor = math.gcd(*supported) if supported else None
    return {
        "schema": STRATEGY_SCHEMA,
        "available_gpu_ids": list(context.available_gpu_ids),
        "selected_gpu_ids": list(context.selected_gpu_ids),
        "gpu_limit": context.gpu_limit,
        "homogeneous": homogeneous,
        "mixed_exact_fill": {
            "possible": mixed is not None,
            "job_gpu_counts": list(mixed) if mixed is not None else None,
            "greatest_common_divisor": divisor,
            "reason_if_impossible": (
                None
                if mixed is not None
                else "the allowed GPU counts cannot sum to the selected GPU limit"
            ),
        },
    }


def _validate_jobs(
    rows: Sequence[Mapping[str, Any]], *, gpu_limit: int
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    observed: set[str] = set()
    for index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise SchedulingError(f"job row {index} is not an object")
        row = copy.deepcopy(dict(source))
        job_id = str(row.get("job_id") or "")
        gpu_count = row.get("gpu_count")
        if not job_id or job_id in observed:
            raise SchedulingError(f"job row {index} has an absent or duplicate job_id")
        if type(gpu_count) is not int or gpu_count <= 0 or gpu_count > gpu_limit:
            raise SchedulingError(
                f"job {job_id} gpu_count must be in [1, {gpu_limit}]"
            )
        observed.add(job_id)
        row["job_id"] = job_id
        row["gpu_count"] = gpu_count
        row["_input_index"] = index
        jobs.append(row)
    if not jobs:
        raise SchedulingError("the job list is empty")
    return jobs


def _repeat_job(
    source: Mapping[str, Any], *, serial: int, existing_ids: set[str]
) -> dict[str, Any]:
    base = f"{source['job_id']}__fullgpu_repeat_{serial}"
    job_id = base
    suffix = 1
    while job_id in existing_ids:
        suffix += 1
        job_id = f"{base}_{suffix}"
    existing_ids.add(job_id)
    row = copy.deepcopy(dict(source))
    row["job_id"] = job_id
    row["synthetic_repeat"] = True
    row["repeat_of_job_id"] = str(source["job_id"])
    row["scheduler_repeat_serial"] = serial
    row["ranking_eligible"] = False
    row["_input_index"] = None
    return row


def _materialize_wave(
    jobs: Sequence[Mapping[str, Any]],
    *,
    selected_gpu_ids: Sequence[int],
    wave_index: int,
    policy: str,
) -> dict[str, Any]:
    offset = 0
    placements = []
    for job in jobs:
        count = int(job["gpu_count"])
        mask = list(selected_gpu_ids[offset : offset + count])
        if len(mask) != count:
            raise SchedulingError("internal error: wave exceeds the selected GPU pool")
        placements.append(
            {
                "job_id": str(job["job_id"]),
                "gpu_count": count,
                "gpu_mask": mask,
                "synthetic_repeat": job.get("synthetic_repeat") is True,
                "repeat_of_job_id": job.get("repeat_of_job_id"),
                "input_index": job.get("_input_index"),
            }
        )
        offset += count
    idle = list(selected_gpu_ids[offset:])
    counts = {int(job["gpu_count"]) for job in jobs}
    return {
        "wave_index": wave_index,
        "policy": policy,
        "homogeneous_gpu_count": next(iter(counts)) if len(counts) == 1 else None,
        "used_gpu_count": offset,
        "idle_gpu_count": len(idle),
        "idle_gpu_ids": idle,
        "full_utilization": not idle,
        "jobs": placements,
    }


def _homogeneous_plan(
    jobs: list[dict[str, Any]],
    *,
    context: GPUContext,
    tail_policy: str,
    require_full: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for job in jobs:
        groups.setdefault(int(job["gpu_count"]), []).append(job)
    waves: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    existing_ids = {str(job["job_id"]) for job in jobs}
    repeat_serial = 0
    for size, current in groups.items():
        concurrency = context.gpu_limit // size
        static_idle = context.gpu_limit % size
        if require_full and static_idle:
            raise SchedulingError(
                f"homogeneous {size}-GPU waves cannot fill a {context.gpu_limit}-GPU "
                "limit; use --policy mixed, change the limit, or use --allow-partial"
            )
        for start in range(0, len(current), concurrency):
            wave_jobs = list(current[start : start + concurrency])
            missing = concurrency - len(wave_jobs)
            if missing and tail_policy == "error":
                raise SchedulingError(
                    f"{size}-GPU group has a {len(wave_jobs)}/{concurrency} tail; "
                    "predeclare repeats or use --tail-policy repeat/partial"
                )
            if missing and tail_policy == "repeat":
                for repeat_index in range(missing):
                    source = current[(start + repeat_index) % len(current)]
                    repeat_serial += 1
                    repeated = _repeat_job(
                        source, serial=repeat_serial, existing_ids=existing_ids
                    )
                    wave_jobs.append(repeated)
                    repeats.append(repeated)
            wave = _materialize_wave(
                wave_jobs,
                selected_gpu_ids=context.selected_gpu_ids,
                wave_index=len(waves),
                policy="homogeneous",
            )
            if require_full and not wave["full_utilization"]:
                raise SchedulingError("internal error: homogeneous wave is not full")
            waves.append(wave)
    return waves, repeats


def _finite_subsets(
    jobs: Sequence[Mapping[str, Any]], capacity: int
) -> dict[int, tuple[int, ...]]:
    plans: dict[int, tuple[int, ...]] = {0: ()}
    for index, job in enumerate(jobs):
        size = int(job["gpu_count"])
        for used, selected in sorted(list(plans.items()), reverse=True):
            destination = used + size
            if destination > capacity:
                continue
            candidate = selected + (index,)
            current = plans.get(destination)
            if current is None or (len(candidate), candidate) < (len(current), current):
                plans[destination] = candidate
    return plans


def _mixed_plan(
    jobs: list[dict[str, Any]],
    *,
    context: GPUContext,
    tail_policy: str,
    require_full: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pending = list(jobs)
    waves: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    existing_ids = {str(job["job_id"]) for job in jobs}
    templates: dict[int, dict[str, Any]] = {}
    for job in jobs:
        templates.setdefault(int(job["gpu_count"]), job)
    repeat_serial = 0
    while pending:
        subsets = _finite_subsets(pending, context.gpu_limit)
        selected_indices: tuple[int, ...] | None = subsets.get(context.gpu_limit)
        repeat_sizes: tuple[int, ...] = ()
        if selected_indices is None and tail_policy == "repeat":
            options = []
            for used, indices in subsets.items():
                if used <= 0:
                    continue
                filler = unbounded_fill(
                    context.gpu_limit - used, templates
                )
                if filler is not None:
                    options.append((used, -len(filler), indices, filler))
            if options:
                _, _, selected_indices, repeat_sizes = max(options)
        if selected_indices is None:
            best_used = max(subsets)
            selected_indices = subsets[best_used]
            if tail_policy == "error" or require_full:
                allowed = sorted(templates)
                divisor = math.gcd(*allowed)
                raise SchedulingError(
                    f"remaining jobs cannot fill {context.gpu_limit} GPUs exactly; "
                    f"best finite use is {best_used}, allowed-size gcd is {divisor}"
                )
        selected = [pending[index] for index in selected_indices]
        for size in repeat_sizes:
            repeat_serial += 1
            repeated = _repeat_job(
                templates[size], serial=repeat_serial, existing_ids=existing_ids
            )
            selected.append(repeated)
            repeats.append(repeated)
        wave = _materialize_wave(
            selected,
            selected_gpu_ids=context.selected_gpu_ids,
            wave_index=len(waves),
            policy="mixed",
        )
        if require_full and not wave["full_utilization"]:
            raise SchedulingError("internal error: mixed wave is not full")
        waves.append(wave)
        for index in sorted(selected_indices, reverse=True):
            pending.pop(index)
    return waves, repeats


def plan_jobs(
    rows: Sequence[Mapping[str, Any]],
    *,
    context: GPUContext,
    policy: str = "homogeneous",
    tail_policy: str = "error",
    require_full: bool = True,
) -> dict[str, Any]:
    """Plan every input job exactly once, plus explicit repeats if requested."""

    if policy not in {"homogeneous", "mixed"}:
        raise SchedulingError(f"unsupported policy: {policy}")
    if tail_policy not in {"error", "repeat", "partial"}:
        raise SchedulingError(f"unsupported tail policy: {tail_policy}")
    if require_full and tail_policy == "partial":
        raise SchedulingError("--tail-policy partial requires --allow-partial")
    jobs = _validate_jobs(rows, gpu_limit=context.gpu_limit)
    if policy == "homogeneous":
        waves, repeats = _homogeneous_plan(
            jobs,
            context=context,
            tail_policy=tail_policy,
            require_full=require_full,
        )
    else:
        waves, repeats = _mixed_plan(
            jobs,
            context=context,
            tail_policy=tail_policy,
            require_full=require_full,
        )
    original_ids = {str(job["job_id"]) for job in jobs}
    scheduled_original_ids = [
        str(placement["job_id"])
        for wave in waves
        for placement in wave["jobs"]
        if not placement["synthetic_repeat"]
    ]
    if (
        set(scheduled_original_ids) != original_ids
        or len(scheduled_original_ids) != len(original_ids)
    ):
        raise SchedulingError("internal error: original-job coverage is not exact")
    strategy = build_strategy(
        context, (int(job["gpu_count"]) for job in jobs)
    )
    return {
        "schema": PLAN_SCHEMA,
        "policy": policy,
        "tail_policy": tail_policy,
        "require_full_utilization": require_full,
        "available_gpu_ids": list(context.available_gpu_ids),
        "selected_gpu_ids": list(context.selected_gpu_ids),
        "gpu_limit": context.gpu_limit,
        "strategy": strategy,
        "summary": {
            "input_jobs": len(jobs),
            "scheduled_original_jobs": len(scheduled_original_ids),
            "synthetic_repeats": len(repeats),
            "waves": len(waves),
            "full_utilization_waves": sum(
                wave["full_utilization"] is True for wave in waves
            ),
            "partial_waves": sum(
                wave["full_utilization"] is not True for wave in waves
            ),
        },
        "synthetic_repeat_jobs": [
            {key: value for key, value in row.items() if key != "_input_index"}
            for row in repeats
        ],
        "waves": waves,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SchedulingError(
                    f"{path}:{line_number} is not valid JSON"
                ) from error
            if not isinstance(row, dict):
                raise SchedulingError(f"{path}:{line_number} must be an object")
            rows.append(row)
    return rows


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parse_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    except ValueError as error:
        raise SchedulingError("job sizes must be comma-separated integers") from error
    if not sizes:
        raise SchedulingError("at least one job size is required")
    return sizes


def _context_from_args(args: argparse.Namespace) -> GPUContext:
    gpu_ids = parse_gpu_ids(args.gpu_ids) if args.gpu_ids else None
    return resolve_gpu_context(gpu_ids=gpu_ids, gpu_limit=args.gpu_limit)


def _emit(payload: Mapping[str, Any], output: Path | None) -> None:
    if output is not None:
        write_json_atomic(output, payload)
        rendered: Mapping[str, Any] = {
            "output": str(output.resolve()),
            "schema": payload.get("schema"),
            "gpu_limit": payload.get("gpu_limit"),
            "summary": payload.get("summary"),
        }
    else:
        rendered = payload
    print(json.dumps(rendered, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    strategy_parser = subparsers.add_parser(
        "strategy", help="show concurrency and exact-fill feasibility"
    )
    strategy_parser.add_argument("--gpu-ids", help="available IDs, e.g. 0-7 or 0,2,4")
    strategy_parser.add_argument("--gpu-limit", type=int)
    strategy_parser.add_argument("--job-sizes", required=True, help="e.g. 1,2,4,8")
    strategy_parser.add_argument("--output", type=Path)

    plan_parser = subparsers.add_parser("plan", help="plan a JSONL job queue")
    plan_parser.add_argument("--jobs", type=Path, required=True)
    plan_parser.add_argument("--gpu-ids", help="available IDs, e.g. 0-7 or 0,2,4")
    plan_parser.add_argument("--gpu-limit", type=int)
    plan_parser.add_argument(
        "--policy", choices=("homogeneous", "mixed"), default="homogeneous"
    )
    plan_parser.add_argument(
        "--tail-policy", choices=("error", "repeat", "partial"), default="error"
    )
    plan_parser.add_argument("--allow-partial", action="store_true")
    plan_parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    try:
        context = _context_from_args(args)
        if args.command == "strategy":
            payload = build_strategy(context, _parse_sizes(args.job_sizes))
        else:
            payload = plan_jobs(
                read_jsonl(args.jobs),
                context=context,
                policy=args.policy,
                tail_policy=args.tail_policy,
                require_full=not args.allow_partial,
            )
        _emit(payload, args.output)
    except SchedulingError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
