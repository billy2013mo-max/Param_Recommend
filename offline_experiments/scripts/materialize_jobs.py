#!/usr/bin/env python3
"""Materialize runnable jobs from staged requests and completed memory labels."""

from __future__ import annotations

import argparse
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, RESULTS_DIR, read_json, read_jsonl, stable_id, write_json, write_jsonl


def load_context() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    models = {row["id"]: row for row in read_json(ARTIFACT_DIR / "model_inventory.json")["models"]}
    families = {row["job_id"]: row for row in read_jsonl(MATRIX_DIR / "memory_boundary_families.jsonl")}
    summaries = {}
    summary_dir = RESULTS_DIR / "boundary_summaries"
    if summary_dir.exists():
        summaries = {path.stem: read_json(path) for path in summary_dir.glob("*.json")}
    return models, families, summaries


def compatible_mbs_values(max_feasible: int | None, gpu_count: int, target_gbs: int) -> list[int]:
    if max_feasible is None:
        return []
    return [
        mbs
        for mbs in (1, 2, 4, 8, 16, 32)
        if mbs <= max_feasible and target_gbs % (gpu_count * mbs) == 0
    ]


def compatible_mbs(max_feasible: int | None, gpu_count: int, target_gbs: int) -> int | None:
    candidates = compatible_mbs_values(max_feasible, gpu_count, target_gbs)
    return max(candidates) if candidates else None


def find_boundaries(
    request: dict[str, Any],
    families: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    gpu_count: int | None = None,
    zero: str | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    matches = []
    for family_id, family in families.items():
        if family_id not in summaries:
            continue
        required = ("model_id", "train_type", "dataset_id")
        if any(family[key] != request[key] for key in required):
            continue
        if gpu_count is not None and family["gpu_count"] != gpu_count:
            continue
        if zero is not None and family["zero"] != zero:
            continue
        matches.append((family, summaries[family_id]))
    return matches


def concrete_job(
    prefix: str,
    request: dict[str, Any],
    model: dict[str, Any],
    family: dict[str, Any],
    mbs: int,
    repeat: int,
    kind: str = "throughput",
    packing: bool = False,
    ga: int | None = None,
) -> dict[str, Any]:
    identity = {
        "request_id": request["request_id"],
        "gpu_count": family["gpu_count"],
        "zero": family["zero"],
        "gc": family["gc"],
        "mbs": mbs,
        "target_gbs": request["target_gbs"],
        "packing": packing,
        "repeat": repeat,
    }
    if request.get("campaign_id") is not None:
        identity = {
            "campaign_id": request["campaign_id"],
            "hardware_id": request.get("hardware_id"),
            **identity,
        }
    scoped_fields = {
        key: request[key]
        for key in ("campaign_id", "phase_id", "hardware_id", "gpu_type")
        if request.get(key) is not None
    }
    job = {
        "job_id": stable_id(prefix, identity),
        "kind": kind,
        **scoped_fields,
        "request_id": request["request_id"],
        "model_id": model["id"],
        "model_path": model["path"],
        "tokenizer_path": model["tokenizer_path"],
        "model_family": model["family"],
        "model_parameters": model["actual_parameters"],
        "template": model["template"],
        "train_type": request["train_type"],
        "dataset_id": request["dataset_id"],
        "cutoff_len": request["cutoff_len"],
        "gpu_count": family["gpu_count"],
        "zero": family["zero"],
        "gc": family["gc"],
        "mbs": mbs,
        "target_gbs": request["target_gbs"],
        "packing": packing,
        "repeat": repeat,
        "warmup_steps": request.get("warmup_steps", 20),
        "measure_steps": request.get("measure_steps", 100),
        "parallel_class": request.get(
            "parallel_class", "exclusive_pool" if family["gpu_count"] == 4 else "gpu_partitionable"
        ),
        "requires_external_node_idle": False,
    }
    if ga is not None:
        job["gradient_accumulation_steps"] = ga
    return job


def throughput_candidates(
    request: dict[str, Any],
    families: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> dict[tuple[int, str, bool, int], tuple[dict[str, Any], int]]:
    mbs_points = max(1, int(request.get("mbs_points_per_strategy", 2)))
    matches = find_boundaries(request, families, summaries)
    feasible: dict[tuple[int, str, bool, int], tuple[dict[str, Any], int]] = {}
    for family, summary in matches:
        candidates = compatible_mbs_values(
            summary.get("max_feasible_mbs"),
            int(family["gpu_count"]),
            int(request["target_gbs"]),
        )
        for mbs in candidates[-mbs_points:]:
            key = (int(family["gpu_count"]), str(family["zero"]), bool(family["gc"]), mbs)
            feasible[key] = (family, mbs)
    return feasible


def static_candidate_priority(
    key: tuple[int, str, bool, int]
) -> tuple[bool, int, int, int]:
    """Prefer low-overhead strategies before any throughput measurement exists."""
    gpu_count, zero, gc, mbs = key
    zero_rank = {"none": 0, "zero2": 0, "zero3": 1}.get(zero, 2)
    return gc, zero_rank, -mbs, gpu_count


def bounded_screen_candidates(
    feasible: dict[tuple[int, str, bool, int], tuple[dict[str, Any], int]],
    limit: int,
) -> list[tuple[int, str, bool, int]]:
    """Keep GPU-count coverage, then fill a small deterministic contrast set."""
    selected: list[tuple[int, str, bool, int]] = []
    selected_set: set[tuple[int, str, bool, int]] = set()
    selected_strategies: set[tuple[int, str, bool]] = set()
    gpu_counts = sorted({key[0] for key in feasible})
    effective_limit = min(len(feasible), max(limit, len(gpu_counts)))
    for gpu_count in gpu_counts:
        best = min(
            (key for key in feasible if key[0] == gpu_count),
            key=static_candidate_priority,
        )
        selected.append(best)
        selected_set.add(best)
        selected_strategies.add(best[:3])
    contrast_strategies = sorted(
        {key[:3] for key in feasible} - selected_strategies,
        key=lambda strategy: static_candidate_priority((*strategy, 1)),
    )
    contrasts = [
        min(
            (key for key in feasible if key[:3] == strategy),
            key=static_candidate_priority,
        )
        for strategy in contrast_strategies
    ]
    for key in [
        *sorted(contrasts, key=static_candidate_priority),
        *sorted(feasible, key=static_candidate_priority),
    ]:
        if len(selected) >= effective_limit:
            break
        if key not in selected_set:
            selected.append(key)
            selected_set.add(key)
    return selected


def materialize_throughput_screen(
    models: dict[str, dict[str, Any]],
    families: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = []
    skipped = []
    for request in read_jsonl(MATRIX_DIR / "throughput_requests.jsonl"):
        # A recommendation still needs alternatives, but exhaustively crossing
        # every ZeRO/GC/MBS choice is too expensive for long-context scenarios.
        # Keep one statically efficient option per GPU count and a bounded set
        # of contrasts; the short screen supplies the actual performance rank.
        feasible = throughput_candidates(request, families, summaries)
        if not feasible:
            skipped.append(
                {
                    "request_id": request["request_id"],
                    "reason": "no compatible successful memory boundary",
                }
            )
            continue
        screen_request = {
            **request,
            "warmup_steps": int(request.get("screen_warmup_steps", 10)),
            "measure_steps": int(request.get("screen_measure_steps", 20)),
        }
        candidate_keys = bounded_screen_candidates(
            feasible,
            int(request.get("screen_max_candidates", len(feasible))),
        )
        for key in candidate_keys:
            family, mbs = feasible[key]
            job = concrete_job(
                "tputscreen",
                screen_request,
                models[request["model_id"]],
                family,
                mbs,
                0,
                kind="throughput_screen",
            )
            job["fidelity"] = "screen"
            jobs.append(job)
    return jobs, skipped


def materialize_throughput(
    models: dict[str, dict[str, Any]],
    families: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize only the Top-K configurations selected by screening."""

    report_path = ARTIFACT_DIR / "stage_decisions.json"
    report = read_json(report_path) if report_path.is_file() else {}
    screening = report.get("throughput_screening") or {}
    decisions = {
        str(row.get("request_id")): row
        for row in screening.get("decisions", [])
    }
    jobs = []
    skipped = []
    for request in read_jsonl(MATRIX_DIR / "throughput_requests.jsonl"):
        request_id = str(request["request_id"])
        decision = decisions.get(request_id) or {}
        shortlisted = decision.get("shortlisted") or []
        if decision.get("status") != "shortlisted" or not shortlisted:
            skipped.append(
                {
                    "request_id": request_id,
                    "reason": "screening is not complete or produced no successful candidate",
                }
            )
            continue
        feasible = throughput_candidates(request, families, summaries)
        for selected in shortlisted:
            key = (
                int(selected["gpu_count"]),
                str(selected["zero"]),
                bool(selected["gc"]),
                int(selected["mbs"]),
            )
            match = feasible.get(key)
            if match is None:
                skipped.append(
                    {
                        "request_id": request_id,
                        "reason": f"shortlisted configuration no longer feasible: {key}",
                    }
                )
                continue
            family, mbs = match
            for repeat in range(int(request["repeats"])):
                job = concrete_job(
                    "tput",
                    request,
                    models[request["model_id"]],
                    family,
                    mbs,
                    repeat,
                )
                job["fidelity"] = "formal"
                jobs.append(job)
    return jobs, skipped


def materialize_scaling(
    models: dict[str, dict[str, Any]],
    families: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = []
    skipped = []
    for request in read_jsonl(MATRIX_DIR / "strong_scaling_requests.jsonl"):
        request = {
            **request,
            "warmup_steps": int(request.get("warmup_steps", 2)),
            "measure_steps": int(request.get("measure_steps", 8)),
        }
        for gpu_count in request["gpu_sequence"]:
            matches = find_boundaries(request, families, summaries, gpu_count=gpu_count)
            feasible = []
            for family, summary in matches:
                mbs = compatible_mbs(summary["max_feasible_mbs"], gpu_count, request["target_gbs"])
                if mbs is not None:
                    feasible.append((family, mbs))
            if not feasible:
                skipped.append({"request_id": request["request_id"], "gpu_count": gpu_count, "reason": "no feasible candidate"})
                continue
            # Throughput screening already compared strategy choices. Scaling
            # needs one representative per card count, not another full sweep.
            family, mbs = min(
                feasible,
                key=lambda item: static_candidate_priority(
                    (
                        int(item[0]["gpu_count"]),
                        str(item[0]["zero"]),
                        bool(item[0]["gc"]),
                        int(item[1]),
                    )
                ),
            )
            for repeat in range(request["repeats"]):
                jobs.append(
                    concrete_job(
                        "scale",
                        request,
                        models[request["model_id"]],
                        family,
                        mbs,
                        repeat,
                    )
                )
    return jobs, skipped


def materialize_packing_memory(
    models: dict[str, dict[str, Any]],
    families: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = []
    skipped = []
    for request in read_jsonl(MATRIX_DIR / "packing_pair_requests.jsonl"):
        matches = find_boundaries(request, families, summaries, request["gpu_count"], request["zero"])
        feasible = [
            family
            for family, summary in matches
            if compatible_mbs(summary["max_feasible_mbs"], request["gpu_count"], request["target_gbs"]) is not None
        ]
        feasible.sort(key=lambda family: family["gc"])
        if not feasible:
            skipped.append({"request_id": request["request_id"], "reason": "no no-packing memory baseline"})
            continue
        family = feasible[0]
        memory_request = {
            **request,
            "warmup_steps": 0,
            "measure_steps": int(request.get("memory_probe_steps", 3)),
        }
        jobs.append(
            concrete_job(
                "packmem",
                memory_request,
                models[request["model_id"]],
                family,
                1,
                0,
                kind="packing_memory_probe",
                packing=True,
                ga=request["gradient_accumulation_steps"],
            )
        )
    return jobs, skipped


def materialize_packing_formal(
    models: dict[str, dict[str, Any]],
    families: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = []
    skipped = []
    memory_jobs_path = MATRIX_DIR / "packing_memory_jobs.jsonl"
    if not memory_jobs_path.exists():
        return [], [{"reason": "packing-memory phase has not been materialized"}]
    memory_jobs = {job["request_id"]: job for job in read_jsonl(memory_jobs_path)}
    for request in read_jsonl(MATRIX_DIR / "packing_pair_requests.jsonl"):
        memory_job = memory_jobs.get(request["request_id"])
        if memory_job is None:
            skipped.append({"request_id": request["request_id"], "reason": "no packing memory job"})
            continue
        status_path = RESULTS_DIR / memory_job["job_id"] / "status.json"
        if not status_path.exists() or read_json(status_path)["classification"] != "success":
            skipped.append({"request_id": request["request_id"], "reason": "packing MBS=1 memory probe not successful"})
            continue
        matches = find_boundaries(request, families, summaries, request["gpu_count"], request["zero"])
        matches = [(family, summary) for family, summary in matches if family["gc"] == memory_job["gc"]]
        feasible = []
        for family, summary in matches:
            mbs = compatible_mbs(summary["max_feasible_mbs"], request["gpu_count"], request["target_gbs"])
            if mbs is not None:
                feasible.append((family, mbs))
        if not feasible:
            skipped.append({"request_id": request["request_id"], "reason": "paired no-packing baseline unavailable"})
            continue
        family, no_pack_mbs = max(feasible, key=lambda item: item[1])
        for repeat in range(request["repeats"]):
            jobs.append(
                concrete_job(
                    "packoff",
                    request,
                    models[request["model_id"]],
                    family,
                    no_pack_mbs,
                    repeat,
                    kind="throughput",
                    packing=False,
                )
            )
            jobs.append(
                concrete_job(
                    "packon",
                    request,
                    models[request["model_id"]],
                    family,
                    1,
                    repeat,
                    kind="throughput",
                    packing=True,
                    ga=request["gradient_accumulation_steps"],
                )
            )
    return jobs, skipped


def materialize_profiler(
    models: dict[str, dict[str, Any]],
    families: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = []
    skipped = []
    for request in read_jsonl(MATRIX_DIR / "profiler_requests.jsonl"):
        matches = find_boundaries(request, families, summaries, request["gpu_count"], request["zero"])
        matches = [(family, summary) for family, summary in matches if family["gc"] == request["gc"]]
        feasible = []
        for family, summary in matches:
            mbs = compatible_mbs(summary["max_feasible_mbs"], family["gpu_count"], request["target_gbs"])
            if mbs is not None:
                feasible.append((family, mbs))
        if not feasible:
            skipped.append({"request_id": request["request_id"], "reason": "no matching memory boundary"})
            continue
        family, mbs = max(feasible, key=lambda item: item[1])
        job = concrete_job("prof", request, models[request["model_id"]], family, mbs, 0, kind="profiler")
        job["enable_profiler"] = True
        job["parallel_class"] = "gpu_partitionable"
        jobs.append(job)
    return jobs, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(
            "throughput-screen",
            "throughput",
            "scaling",
            "packing-memory",
            "packing-formal",
            "profiler",
        ),
    )
    args = parser.parse_args()
    models, families, summaries = load_context()
    if args.phase == "throughput-screen":
        jobs, skipped = materialize_throughput_screen(models, families, summaries)
        output = MATRIX_DIR / "throughput_screen_jobs.jsonl"
    elif args.phase == "throughput":
        jobs, skipped = materialize_throughput(models, families, summaries)
        output = MATRIX_DIR / "throughput_jobs.jsonl"
    elif args.phase == "scaling":
        jobs, skipped = materialize_scaling(models, families, summaries)
        output = MATRIX_DIR / "scaling_candidate_jobs.jsonl"
    elif args.phase == "packing-memory":
        jobs, skipped = materialize_packing_memory(models, families, summaries)
        output = MATRIX_DIR / "packing_memory_jobs.jsonl"
    elif args.phase == "packing-formal":
        jobs, skipped = materialize_packing_formal(models, families, summaries)
        output = MATRIX_DIR / "packing_pair_jobs.jsonl"
    else:
        jobs, skipped = materialize_profiler(models, families, summaries)
        output = MATRIX_DIR / "profiler_jobs.jsonl"
    write_jsonl(output, jobs)
    report = {"phase": args.phase, "jobs": len(jobs), "skipped": skipped, "output": str(output)}
    write_json(MATRIX_DIR / f"materialize_{args.phase}.json", report)
    print(report)


if __name__ == "__main__":
    main()
