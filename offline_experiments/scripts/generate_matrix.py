#!/usr/bin/env python3
"""Generate a hardware-aware, pruned experimental design; never launches training."""

from __future__ import annotations

from collections import Counter
from typing import Any

from common import ARTIFACT_DIR, CONFIG_DIR, MATRIX_DIR, read_json, stable_id, write_json, write_jsonl


LEGACY_PLACEMENTS = (
    (1, "none"),
    (2, "zero2"),
    (2, "zero3"),
    (4, "zero2"),
    (4, "zero3"),
)
LEGACY_ANCHOR_DATASETS = {"short_512", "multiturn_4096", "longcontext_32768"}
LEGACY_CONTRAST_MODELS = {"qwen3_8b", "qwen3_14b", "qwen3_32b", "qwen2p5_72b"}
LEGACY_ANALYSIS_MODELS = {"qwen3_8b", "qwen3_14b", "qwen3_32b"}


def optimizer_state_lower_bound_bytes(model: dict[str, Any], train_type: str, gpu_count: int, zero: str) -> float:
    parameters = model["actual_parameters"]
    if train_type == "lora":
        return 2.0 * parameters / (gpu_count if zero == "zero3" else 1)
    if zero == "none":
        return 16.0 * parameters
    if zero == "zero2":
        return 2.0 * parameters + 14.0 * parameters / gpu_count
    if zero == "zero3":
        return 16.0 * parameters / gpu_count
    raise ValueError(zero)


def design_context(experiment: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
    scope = experiment["training_scope"]
    policy = experiment.get("matrix_policy", {})
    measurement = experiment["measurement"]
    formal_repeats = int(measurement.get("throughput_repeats", 1))
    if formal_repeats < 1:
        raise ValueError("measurement.throughput_repeats must be at least 1")
    throughput_mbs_points = int(policy.get("throughput_mbs_points_per_strategy", 2))
    if throughput_mbs_points < 1:
        raise ValueError("matrix_policy.throughput_mbs_points_per_strategy must be at least 1")
    throughput_shortlist_top_k = int(policy.get("throughput_shortlist_top_k", 3))
    if throughput_shortlist_top_k < 1:
        raise ValueError("matrix_policy.throughput_shortlist_top_k must be at least 1")
    screen_candidate_limit = int(
        policy.get("throughput_screen_max_candidates_per_scenario", 4)
    )
    if screen_candidate_limit < 2:
        raise ValueError(
            "matrix_policy.throughput_screen_max_candidates_per_scenario must be at least 2"
        )
    screen_warmup_steps = int(measurement.get("throughput_screen_warmup_steps", 2))
    screen_measure_steps = int(measurement.get("throughput_screen_measure_steps", 4))
    if screen_warmup_steps < 0 or screen_measure_steps < 1:
        raise ValueError("throughput screening requires non-negative warmup and at least one measured step")
    windows = {
        "formal": (
            int(measurement.get("throughput_warmup_steps", 3)),
            int(measurement.get("throughput_measure_steps", 10)),
        ),
        "scaling": (
            int(measurement.get("scaling_warmup_steps", 2)),
            int(measurement.get("scaling_measure_steps", 8)),
        ),
        "packing": (
            int(measurement.get("packing_warmup_steps", 2)),
            int(measurement.get("packing_measure_steps", 8)),
        ),
        "profiler": (
            int(measurement.get("profiler_warmup_steps", 1)),
            int(measurement.get("profiler_measure_steps", 3)),
        ),
    }
    if any(warmup < 0 or measured < 1 for warmup, measured in windows.values()):
        raise ValueError("measurement windows require non-negative warmup and measured steps")
    gpu_counts = set(scope["gpu_counts"])
    zero_by_count = {int(key): values for key, values in scope["zero_by_gpu_count"].items()}
    placements = [
        (gpu_count, zero)
        for gpu_count, zero in LEGACY_PLACEMENTS
        if gpu_count in gpu_counts and zero in zero_by_count[gpu_count]
    ]
    memory_bytes = int(
        hardware.get("memory_bytes_reported_by_torch")
        or hardware.get("per_gpu", {}).get("torch_total_memory_bytes")
    )
    return {
        "legacy_unscoped_identity": "campaign_id" not in experiment,
        "campaign_id": experiment.get("campaign_id", scope["phase_id"]),
        "phase_id": scope["phase_id"],
        "hardware_id": hardware.get("hardware_id", hardware.get("gpu_id", "unknown")),
        "gpu_type": scope["gpu_type"],
        "gpu_pool": scope.get("gpu_pool") or scope["gpu_ids"],
        "memory_bytes": memory_bytes,
        "placements": placements,
        "preferred_fraction": float(policy.get("preferred_static_memory_fraction", 0.90)),
        "possible_fraction": float(policy.get("analytic_possible_fraction", 0.98)),
        "anchor_dataset_ids": set(policy.get("anchor_dataset_ids", LEGACY_ANCHOR_DATASETS)),
        "contrast_model_ids": set(policy.get("contrast_model_ids", LEGACY_CONTRAST_MODELS)),
        "multi_gbs_model_ids": set(policy.get("multi_gbs_model_ids", LEGACY_ANALYSIS_MODELS)),
        "scaling_model_ids": set(policy.get("scaling_model_ids", LEGACY_ANALYSIS_MODELS)),
        "packing_model_ids": set(policy.get("packing_model_ids", LEGACY_ANALYSIS_MODELS)),
        "profiler_model_ids": set(policy.get("profiler_model_ids", {"qwen3_8b", "qwen3_32b"})),
        "profiler_dataset_id": policy.get("profiler_dataset_id", "multiturn_4096"),
        "formal_repeats": formal_repeats,
        "throughput_mbs_points_per_strategy": throughput_mbs_points,
        "throughput_shortlist_top_k": throughput_shortlist_top_k,
        "throughput_screen_max_candidates_per_scenario": screen_candidate_limit,
        "screen_warmup_steps": screen_warmup_steps,
        "screen_measure_steps": screen_measure_steps,
        "formal_warmup_steps": windows["formal"][0],
        "formal_measure_steps": windows["formal"][1],
        "scaling_warmup_steps": windows["scaling"][0],
        "scaling_measure_steps": windows["scaling"][1],
        "packing_warmup_steps": windows["packing"][0],
        "packing_measure_steps": windows["packing"][1],
        "packing_memory_probe_steps": int(
            measurement.get("packing_memory_probe_steps", 3)
        ),
        "profiler_warmup_steps": windows["profiler"][0],
        "profiler_measure_steps": windows["profiler"][1],
    }


def scoped_identity(context: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    # The original H800 campaign predates campaign-scoped IDs. Preserve those
    # IDs so regenerated one-run requests continue to join existing boundaries
    # and completed throughput results. New campaigns (including RTX 4090)
    # retain the collision-safe scoped identity.
    if context["legacy_unscoped_identity"]:
        return identity
    return {
        "campaign_id": context["campaign_id"],
        "phase_id": context["phase_id"],
        "hardware_id": context["hardware_id"],
        "gpu_type": context["gpu_type"],
        **identity,
    }


def analytically_possible(
    context: dict[str, Any], model: dict[str, Any], train_type: str, gpu_count: int, zero: str
) -> bool:
    lower = optimizer_state_lower_bound_bytes(model, train_type, gpu_count, zero)
    return lower < context["memory_bytes"] * context["possible_fraction"]


def preferred_placement(context: dict[str, Any], model: dict[str, Any], train_type: str) -> tuple[int, str]:
    for gpu_count, zero in context["placements"]:
        lower = optimizer_state_lower_bound_bytes(model, train_type, gpu_count, zero)
        if lower < context["memory_bytes"] * context["preferred_fraction"]:
            return gpu_count, zero
    possible = [
        placement
        for placement in context["placements"]
        if analytically_possible(context, model, train_type, *placement)
    ]
    if possible:
        return possible[0]
    raise RuntimeError(f"No analytically possible placement for {model['id']}/{train_type}")


def mbs_candidates(cutoff: int, gpu_count: int, target_gbs: int = 64) -> list[int]:
    if cutoff <= 512:
        candidates = [1, 2, 4, 8, 16, 32]
    elif cutoff <= 2048:
        candidates = [1, 2, 4, 8, 16]
    elif cutoff <= 4096:
        candidates = [1, 2, 4, 8]
    elif cutoff <= 8192:
        candidates = [1, 2, 4]
    else:
        candidates = [1, 2]
    return [mbs for mbs in candidates if gpu_count * mbs <= target_gbs and target_gbs % (gpu_count * mbs) == 0]


def profile_name(model: dict[str, Any]) -> str:
    return "qwen3_nothink"


def make_memory_family(
    context: dict[str, Any],
    model: dict[str, Any],
    train_type: str,
    dataset_id: str,
    cutoff: int,
    gpu_count: int,
    zero: str,
    gc: bool,
    sampling_role: str,
) -> dict[str, Any]:
    identity = scoped_identity(
        context,
        {
            "model_id": model["id"],
            "train_type": train_type,
            "dataset_id": dataset_id,
            "gpu_count": gpu_count,
            "zero": zero,
            "gc": gc,
            "packing": False,
            "target_gbs": 64,
        },
    )
    return {
        "job_id": stable_id("mem", identity),
        "kind": "memory_boundary",
        **identity,
        "model_path": model["path"],
        "tokenizer_path": model["tokenizer_path"],
        "model_family": model["family"],
        "model_parameters": model["actual_parameters"],
        "template": model["template"],
        "cutoff_len": cutoff,
        "mbs_candidates": mbs_candidates(cutoff, gpu_count),
        "analytic_state_lower_bound_bytes_per_gpu": optimizer_state_lower_bound_bytes(
            model, train_type, gpu_count, zero
        ),
        "memory_total_bytes_per_gpu": context["memory_bytes"],
        "sampling_roles": [sampling_role],
        "parallel_class": "exclusive_node" if gpu_count == len(context["gpu_pool"]) else "gpu_partitionable",
        "repeats": 1,
    }


def merge_family(families: dict[str, dict[str, Any]], family: dict[str, Any]) -> None:
    existing = families.get(family["job_id"])
    if existing is None:
        families[family["job_id"]] = family
    else:
        existing["sampling_roles"] = sorted(set(existing["sampling_roles"] + family["sampling_roles"]))


def throughput_requests(
    context: dict[str, Any],
    models: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    requests: dict[str, dict[str, Any]] = {}
    for model in models:
        for train_type in model["train_types"]:
            gpu_count, zero = preferred_placement(context, model, train_type)
            for dataset in datasets:
                profile = analysis["datasets"][dataset["id"]]["profiles"][profile_name(model)]
                global_batches = [64]
                if model["id"] in context["multi_gbs_model_ids"]:
                    global_batches += [16, 256]
                for target_gbs in global_batches:
                    identity = scoped_identity(
                        context,
                        {
                            "model_id": model["id"],
                            "train_type": train_type,
                            "dataset_id": dataset["id"],
                            "gpu_count": gpu_count,
                            "zero": zero,
                            "target_gbs": target_gbs,
                        },
                    )
                    request = {
                        "request_id": stable_id("tput", identity),
                        "kind": "throughput_after_memory",
                        **identity,
                        "cutoff_len": profile["lengths"]["cutoff_len"],
                        "candidate_scope": "all_feasible_memory_strategies",
                        "gc_selection": "all_feasible",
                        "mbs_selection": "largest_compatible_points_per_strategy",
                        "mbs_points_per_strategy": context["throughput_mbs_points_per_strategy"],
                        "screen_max_candidates": context[
                            "throughput_screen_max_candidates_per_scenario"
                        ],
                        "screen_warmup_steps": context["screen_warmup_steps"],
                        "screen_measure_steps": context["screen_measure_steps"],
                        "shortlist_top_k": context["throughput_shortlist_top_k"],
                        "repeats": context["formal_repeats"],
                        "warmup_steps": context["formal_warmup_steps"],
                        "measure_steps": context["formal_measure_steps"],
                        "parallel_class": (
                            "exclusive_pool" if gpu_count == len(context["gpu_pool"]) else "gpu_partitionable"
                        ),
                    }
                    requests[request["request_id"]] = request
    return sorted(requests.values(), key=lambda row: row["request_id"])


def strong_scaling_requests(
    context: dict[str, Any],
    models: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    requests = []
    gpu_sequence = sorted({gpu_count for gpu_count, _ in context["placements"]})
    for model in models:
        if model["id"] not in context["scaling_model_ids"]:
            continue
        for train_type in model["train_types"]:
            for dataset in datasets:
                if dataset["id"] not in context["anchor_dataset_ids"]:
                    continue
                cutoff = analysis["datasets"][dataset["id"]]["profiles"][profile_name(model)]["lengths"]["cutoff_len"]
                identity = scoped_identity(
                    context,
                    {
                        "model_id": model["id"],
                        "train_type": train_type,
                        "dataset_id": dataset["id"],
                        "target_gbs": 64,
                    },
                )
                requests.append(
                    {
                        "request_id": stable_id("scale", identity),
                        "kind": "strong_scaling_family",
                        **identity,
                        "cutoff_len": cutoff,
                        "gpu_sequence": gpu_sequence,
                        "selection": "for each card count use fastest feasible ZeRO/GC/MBS at identical GBS",
                        "stop_rule": "stop after conservative throughput ratio lower bound is below 1.8 (80% gain)",
                        "minimum_throughput_ratio_per_doubling": 1.8,
                        "repeats": context["formal_repeats"],
                        "warmup_steps": context["scaling_warmup_steps"],
                        "measure_steps": context["scaling_measure_steps"],
                        "parallel_class": "gpu_partitionable",
                    }
                )
    return requests


def packing_requests(
    context: dict[str, Any],
    models: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    requests = []
    for model in models:
        if model["id"] not in context["packing_model_ids"]:
            continue
        for train_type in model["train_types"]:
            gpu_count, zero = preferred_placement(context, model, train_type)
            for dataset in datasets:
                profile = analysis["datasets"][dataset["id"]]["profiles"][profile_name(model)]
                if not profile["packing"]["packing_eligible_for_paired_test"]:
                    continue
                ga_rows = [
                    row
                    for row in profile["packing"]["ga_table"]
                    if row["data_parallel"] == gpu_count
                    and row["target_gbs"] in {16, 64, 256}
                    and row["relative_error"] <= 0.05
                ]
                if not ga_rows:
                    continue
                selected_ga = min(
                    ga_rows,
                    key=lambda row: ({64: 0, 256: 1, 16: 2}[row["target_gbs"]], row["relative_error"]),
                )
                identity = scoped_identity(
                    context,
                    {
                        "model_id": model["id"],
                        "train_type": train_type,
                        "dataset_id": dataset["id"],
                        "gpu_count": gpu_count,
                        "zero": zero,
                        "target_gbs": selected_ga["target_gbs"],
                    },
                )
                requests.append(
                    {
                        "request_id": stable_id("pack", identity),
                        "kind": "packing_pair_after_memory",
                        **identity,
                        "cutoff_len": profile["lengths"]["cutoff_len"],
                        "no_packing_mbs": "fastest feasible no-packing MBS",
                        "packing_physical_mbs": 1,
                        "packing_effective_mbs": profile["packing"]["mean_samples_per_pack"],
                        "gradient_accumulation_steps": selected_ga["gradient_accumulation_steps"],
                        "expected_sample_gbs": selected_ga["expected_sample_gbs"],
                        "expected_gbs_relative_error": selected_ga["relative_error"],
                        "repeats": context["formal_repeats"],
                        "warmup_steps": context["packing_warmup_steps"],
                        "measure_steps": context["packing_measure_steps"],
                        "memory_probe_steps": context["packing_memory_probe_steps"],
                        "parallel_class": (
                            "exclusive_pool" if gpu_count == len(context["gpu_pool"]) else "gpu_partitionable"
                        ),
                    }
                )
    return requests


def profiler_requests(
    context: dict[str, Any],
    models: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    requests = []
    target_dataset = next(row for row in datasets if row["id"] == context["profiler_dataset_id"])
    for model in models:
        if model["id"] not in context["profiler_model_ids"]:
            continue
        for train_type in model["train_types"]:
            gpu_count, zero = preferred_placement(context, model, train_type)
            cutoff = analysis["datasets"][target_dataset["id"]]["profiles"][profile_name(model)]["lengths"]["cutoff_len"]
            for gc in (False, True):
                identity = scoped_identity(
                    context,
                    {
                        "model_id": model["id"],
                        "train_type": train_type,
                        "dataset_id": target_dataset["id"],
                        "gpu_count": gpu_count,
                        "zero": zero,
                        "gc": gc,
                    },
                )
                requests.append(
                    {
                        "request_id": stable_id("prof", identity),
                        "kind": "profiler_after_memory",
                        **identity,
                        "cutoff_len": cutoff,
                        "target_gbs": 64,
                        "warmup_steps": context["profiler_warmup_steps"],
                        "measure_steps": context["profiler_measure_steps"],
                        "repeats": 1,
                        "parallel_class": "gpu_partitionable",
                    }
                )
    return requests


def main() -> None:
    experiment = read_json(CONFIG_DIR / "experiment.json")
    hardware = read_json(CONFIG_DIR / "hardware.json")
    inventory = read_json(ARTIFACT_DIR / "model_inventory.json")
    analysis = read_json(ARTIFACT_DIR / "dataset_analysis.json")
    context = design_context(experiment, hardware)
    enabled_model_ids = experiment["training_scope"]["model_ids"]
    inventory_by_id = {model["id"]: model for model in inventory["models"]}
    missing_model_ids = [model_id for model_id in enabled_model_ids if model_id not in inventory_by_id]
    if missing_model_ids:
        raise RuntimeError(f"Current phase references absent model inventory entries: {missing_model_ids}")
    models = [inventory_by_id[model_id] for model_id in enabled_model_ids]
    datasets = experiment["datasets"]

    families: dict[str, dict[str, Any]] = {}
    for model in models:
        for train_type in model["train_types"]:
            base_placement = preferred_placement(context, model, train_type)
            for dataset in datasets:
                cutoff = analysis["datasets"][dataset["id"]]["profiles"][profile_name(model)]["lengths"]["cutoff_len"]
                for gc in (False, True):
                    merge_family(
                        families,
                        make_memory_family(
                            context, model, train_type, dataset["id"], cutoff, *base_placement, gc, "base_grid"
                        ),
                    )

            if model["id"] in context["contrast_model_ids"]:
                for dataset in datasets:
                    if dataset["id"] not in context["anchor_dataset_ids"]:
                        continue
                    cutoff = analysis["datasets"][dataset["id"]]["profiles"][profile_name(model)]["lengths"]["cutoff_len"]
                    for gpu_count, zero in context["placements"]:
                        if not analytically_possible(context, model, train_type, gpu_count, zero):
                            continue
                        for gc in (False, True):
                            merge_family(
                                families,
                                make_memory_family(
                                    context,
                                    model,
                                    train_type,
                                    dataset["id"],
                                    cutoff,
                                    gpu_count,
                                    zero,
                                    gc,
                                    "gpu_zero_contrast",
                                ),
                            )

    memory = sorted(families.values(), key=lambda row: row["job_id"])
    throughput = throughput_requests(context, models, datasets, analysis)
    scaling = strong_scaling_requests(context, models, datasets, analysis)
    packing = packing_requests(context, models, datasets, analysis)
    profiler = profiler_requests(context, models, datasets, analysis)
    write_jsonl(MATRIX_DIR / "memory_boundary_families.jsonl", memory)
    write_jsonl(MATRIX_DIR / "throughput_requests.jsonl", throughput)
    write_jsonl(MATRIX_DIR / "strong_scaling_requests.jsonl", scaling)
    write_jsonl(MATRIX_DIR / "packing_pair_requests.jsonl", packing)
    write_jsonl(MATRIX_DIR / "profiler_requests.jsonl", profiler)

    summary = {
        "schema_version": 2,
        "campaign_id": context["campaign_id"],
        "phase_id": context["phase_id"],
        "hardware_id": context["hardware_id"],
        "gpu_type": context["gpu_type"],
        "enabled_model_ids": enabled_model_ids,
        "deferred_model_ids": experiment["training_scope"]["deferred_model_ids"],
        "training_is_not_started": True,
        "memory_boundary_families": len(memory),
        "memory_probe_upper_bound": sum(len(row["mbs_candidates"]) for row in memory),
        "throughput_configurations_before_repeats": len(throughput),
        "throughput_scenarios_before_materialization": len(throughput),
        "throughput_mbs_points_per_strategy": context["throughput_mbs_points_per_strategy"],
        "throughput_screen_max_candidates_per_scenario": context[
            "throughput_screen_max_candidates_per_scenario"
        ],
        "throughput_shortlist_top_k": context["throughput_shortlist_top_k"],
        "throughput_screen_warmup_steps": context["screen_warmup_steps"],
        "throughput_screen_measure_steps": context["screen_measure_steps"],
        "throughput_formal_warmup_steps": context["formal_warmup_steps"],
        "throughput_formal_measure_steps": context["formal_measure_steps"],
        "scaling_warmup_steps": context["scaling_warmup_steps"],
        "scaling_measure_steps": context["scaling_measure_steps"],
        "packing_warmup_steps": context["packing_warmup_steps"],
        "packing_measure_steps": context["packing_measure_steps"],
        "formal_repeats_per_configuration": context["formal_repeats"],
        "throughput_runs_after_repeats": len(throughput) * context["formal_repeats"],
        "strong_scaling_families": len(scaling),
        "packing_pair_families": len(packing),
        "profiler_calibration_configurations": len(profiler),
        "memory_by_sampling_role": dict(Counter(role for row in memory for role in row["sampling_roles"])),
        "memory_by_gpu_count": dict(Counter(str(row["gpu_count"]) for row in memory)),
        "memory_by_train_type": dict(Counter(row["train_type"] for row in memory)),
        "notes": [
            "Memory jobs are adaptive boundary families: MBS doubles and stops at the first OOM.",
            "Formal throughput jobs are materialized only after memory results exist.",
            "Each throughput scenario keeps one statically efficient candidate per feasible GPU count, then fills contrast slots up to the configured cap.",
            "Reduced candidates run a 2+4-step screen; only the lowest-resource and fastest Top-2 finalists run a 3+10-step formal window.",
            "Strong scaling measures one statically preferred strategy per GPU count with a 2+8-step window.",
            "Throughput, strong-scaling and paired packing configurations run once by default; only failed or unhealthy results are rerun.",
            f"All phases pack disjoint masks across GPU pool {context['gpu_pool']}.",
        ],
    }
    write_json(MATRIX_DIR / "design_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
