#!/usr/bin/env python3
"""Refit H800 M1 with stage two and score every unused validation source.

The fit set is fixed to the original 158 historical calibration observations,
the 105 source-disjoint observations used by the previous M1 fit, and the new
60-job stage-two campaign.  Model and safety-variant selection use only nested
leave-source-out predictions from that fit set.

Validation is reported in three explicitly separated scopes:

* strict dataset-disjoint validation: every complete H800 memory holdout source
  whose sample content never entered the fit;
* historical configuration holdout: the original 167 observations, which were
  not fitted but share datasets with the historical calibration rows;
* overlap replay: holdout campaigns whose datasets later entered the fit.  They
  are audited and scored, but never presented as independent validation.

OOM observations remain right-censored constraints and are never converted to
exact peak-memory regression labels.  This script is CPU-only and does not
publish or mutate the production predictor.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any

from analyze_h800_fresh_memory_residual_v2 import profile_padding_statistics
from calibrate_h800_m1_safety_upper_v2 import (
    CANDIDATE_SCHEMA,
    SAFETY_VARIANTS,
    S2_RESERVED_OOM,
    VARIANT_M1,
    _critical_tail_entry,
    _detail,
    _fit_safety_bundle,
    _nested_safety_ablation,
    _predict_safety,
    _prepare_records,
    _scope_metrics,
    _select_safety_variant,
)
from common import (
    ARTIFACT_DIR,
    MATRIX_DIR,
    RESULTS_DIR,
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)
from export_h800_observations import (
    _load_json,
    _one_observation,
    _provenance_index,
    validate_canonical_observation,
    write_jsonl,
)
from fit_h800_lora_source_disjoint_recalibration_v1 import (
    DEFAULT_COVERAGE,
    DEFAULT_FROZEN_BASELINE,
    DEFAULT_HARDWARE,
    DEFAULT_INVENTORY,
    _cluster_id,
    _collapse_records,
    _evaluate,
    _input_binding,
    _inventory_models,
    _is_critical_lora,
    _outcome,
    _read_jsonl,
    _round_up,
    _source_id,
)
from fit_h800_profile_aware_memory_challenger_v1 import _export_exact_jobs
from h800_theory_basis import _memory_observation, _model_geometry, memory_basis
from fit_h800_m1_separate_admission_v4 import (
    S4_SEPARATE_ADMISSION,
    _fit_admission_head,
    _nested_separate_admission,
    _separate_admission_detail,
)
from fit_h800_m1_full_admission_v5 import (
    S5_LORA_FULL_ADMISSION,
    _fit_v5_heads,
    _nested_v5,
    _v5_detail,
)
from migrate_refit_h800_historical_memory_v1 import (
    DEFAULT_CANONICAL,
    DEFAULT_CURRENT_OLD,
    DEFAULT_DATASET_ANALYSIS,
    DEFAULT_NEW_QUEUE,
    DEFAULT_THEORY_BASIS,
    _build_pairs,
    _content_hashes,
    _current_data_paths,
    _dataset_registry,
)


SCHEMA = "sft_h800_m1_all_unused_validation_refit/v3"
SCHEMA_V4 = "sft_h800_m1_separate_admission_all_unused_validation/v4"
SCHEMA_V5 = "sft_h800_m1_lora_full_separate_admission_validation/v5"
IMPLEMENTATION_VERSION = (
    "sft_h800_m1_all_unused_validation_refit/"
    "2026-08-05.stage2-40-fit-sources-all-unused-validation-v3"
)
IMPLEMENTATION_VERSION_V4 = (
    "sft_h800_m1_separate_admission/"
    "2026-08-05.direct-reserved-upper-binary-risk-head-v4"
)
IMPLEMENTATION_VERSION_V5 = (
    "sft_h800_m1_lora_full_separate_admission/"
    "2026-08-05.full-zero3-gc-two-gpu-plus-lora-four-gpu-"
    "stacked-fail-closed-v5.1"
)
DEFAULT_STAGE2_QUEUE = MATRIX_DIR / "h800_lora_safety_stage2_jobs_v1.jsonl"
DEFAULT_VALIDATION_INVENTORY = (
    ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_v1.json"
)
DEFAULT_PREVIOUS_CANDIDATE = (
    ROOT
    / "diagnostics"
    / "h800_m1_safety_upper_v2_20260805"
    / "candidate_model_safety_upper_v2.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "diagnostics" / "h800_m1_all_unused_validation_refit_20260805"
)

HOLDOUT_CAMPAIGNS = (
    {
        "name": "fresh_holdout_v2",
        "queue": MATRIX_DIR / "h800_fresh_holdout_jobs_v2.jsonl",
        "snapshot": ARTIFACT_DIR / "h800_fresh_holdout_observations_v2.json",
    },
    {
        "name": "bounded_memory_v2_fresh_holdout_v1",
        "queue": MATRIX_DIR / "h800_bounded_memory_v2_fresh_holdout_jobs_v1.jsonl",
        "snapshot": ARTIFACT_DIR
        / "h800_bounded_memory_v2_fresh_holdout_observations_v1.json",
    },
    {
        "name": "final_unseen_holdout_v1",
        "queue": MATRIX_DIR / "h800_final_unseen_holdout_jobs_v1.jsonl",
        "snapshot": ARTIFACT_DIR / "h800_final_unseen_holdout_observations_v1.json",
    },
)


def _job(observation: Mapping[str, Any]) -> dict[str, Any]:
    configuration = observation.get("configuration") or {}
    value = configuration.get("job") if isinstance(configuration, Mapping) else None
    if not isinstance(value, Mapping):
        raise ValueError("canonical observation has no bound job")
    return dict(value)


def _profile_path(job: Mapping[str, Any]) -> Path:
    value = job.get("dataset_profile_path")
    if not value:
        pressure = job.get("expected_padding_pressure") or {}
        value = pressure.get("path") if isinstance(pressure, Mapping) else None
    path = Path(str(value or ""))
    if not path.is_file():
        raise FileNotFoundError(f"profile is absent: {path}")
    expected = job.get("dataset_profile_sha256")
    if expected and sha256_file(path) != str(expected):
        raise ValueError(f"profile SHA-256 drifted: {path}")
    return path


def _raw_profile_max(path: Path) -> int:
    maximum = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                maximum = max(maximum, int(json.loads(line)["total_tokens"]))
    if maximum <= 0:
        raise ValueError(f"profile is empty: {path}")
    return maximum


def _validation_partition(
    observation: dict[str, Any], job: Mapping[str, Any]
) -> dict[str, str]:
    configuration = observation.setdefault("configuration", {})
    existing = configuration.get("calibration_partition") or {}
    declared = job.get("calibration_partition") or {}
    split_unit_id = (
        existing.get("split_unit_id")
        or declared.get("split_unit_id")
        or str(job.get("dataset_id") or "").split("__", 1)[0]
    )
    if not split_unit_id:
        raise ValueError(f"validation job has no source id: {job.get('job_id')}")
    partition = {
        "role": "holdout",
        "policy": "all_unused_dataset_validation_v3",
        "split_unit_id": str(split_unit_id),
    }
    configuration["calibration_partition"] = partition
    return partition


def _export_validation_jobs(
    queue_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Export exact old holdout attempts without reclassifying them as fit rows."""

    hardware = _load_json(ROOT / "config" / "hardware.json")
    experiment = _load_json(ROOT / "config" / "experiment.json")
    provenance = _provenance_index(ROOT)
    observations: list[dict[str, Any]] = []
    for job in sorted(queue_rows, key=lambda row: str(row["job_id"])):
        job_id = str(job["job_id"])
        latest = _load_json(RESULTS_DIR / job_id / "latest_attempt.json")
        if (
            latest.get("job_id") != job_id
            or latest.get("state") != "complete"
            or latest.get("calibration_eligible") is not True
        ):
            raise ValueError(f"validation attempt is not evidence-complete: {job_id}")
        attempt = RESULTS_DIR / job_id / str(latest["attempt_path"])
        row = _one_observation(
            attempt,
            ROOT,
            hardware,
            experiment,
            provenance,
            job_id_hint=job_id,
            attempt_scoped=True,
        )
        if row is None:
            raise ValueError(f"canonical exporter rejected validation job: {job_id}")
        errors = validate_canonical_observation(row)
        if errors:
            raise ValueError(f"canonical validation failed for {job_id}: {errors}")
        _validation_partition(row, job)
        observations.append(row)
    return observations


def _build_validation_record(
    observation: dict[str, Any],
    *,
    model_by_id: Mapping[str, dict[str, Any]],
    fixed_lora: Mapping[str, Any],
    hardware: Mapping[str, Any],
    origin: str,
    padding_cache: dict[tuple[str, int, int], dict[str, Any]],
    raw_max_cache: dict[str, int],
) -> dict[str, Any]:
    job = _job(observation)
    partition = _validation_partition(observation, job)
    model_id = str(job.get("model_id") or "")
    if model_id not in model_by_id:
        raise ValueError(f"model inventory is missing {model_id}")
    geometry = _model_geometry(job, model_by_id[model_id], dict(fixed_lora))
    profile_path = _profile_path(job)
    cutoff = int(job["cutoff_len"])
    mbs = int(job["mbs"])
    cache_key = (str(profile_path.resolve()), cutoff, mbs)
    if cache_key not in padding_cache:
        padding_cache[cache_key] = profile_padding_statistics(
            profile_path, cutoff_len=cutoff, physical_mbs=mbs
        )
    padding = dict(padding_cache[cache_key])
    raw_key = str(profile_path.resolve())
    if raw_key not in raw_max_cache:
        raw_max_cache[raw_key] = _raw_profile_max(profile_path)
    raw_max = int(raw_max_cache[raw_key])
    clipped_max = int(padding["maximum_clipped_tokens"])
    sequence = _round_up(min(cutoff, clipped_max), 8)
    zero_stage = int(job.get("zero_stage") or 0)
    gc = bool(
        job["gradient_checkpointing"]
        if "gradient_checkpointing" in job
        else job.get("gc")
    )
    memory = memory_basis(
        {
            "gpu_count": int(job["gpu_count"]),
            "mbs": mbs,
            "cutoff_len": sequence,
            "zero": f"zero{zero_stage}",
            "gc": gc,
        },
        geometry,
        int(hardware["memory_bytes_reported_by_torch"]),
    )
    memory["observed"] = _memory_observation(observation)
    record: dict[str, Any] = {
        "schema": "sft_h800_memory_calibration_record/v3",
        "observation_id": observation["observation_id"],
        "job_id": job.get("job_id"),
        "outcome": (observation.get("outcome") or {}).get("class"),
        "route": {"feasibility": True, "memory_boundary": True},
        "scenario": {
            "model_id": model_id,
            "train_type": job.get("train_type"),
            "dataset_id": job.get("dataset_id"),
            "target_gbs": job.get("target_gbs"),
            "gpu_count": int(job["gpu_count"]),
            "physical_mbs": mbs,
            "cutoff_len": cutoff,
        },
        "selector": {
            "training_mode": str(job.get("train_type") or ""),
            "zero_stage": zero_stage,
            "gradient_checkpointing": gc,
            "packing": bool(job.get("packing")),
        },
        "calibration_partition": dict(partition),
        "model_basis": geometry,
        "memory": memory,
        "effective_sequence": {
            "policy": "round_up(min(cutoff_len, profile_max), 8)",
            "tokens": sequence,
            "cutoff_len": cutoff,
            "fraction_of_cutoff": sequence / float(cutoff),
            "raw_profile_max": raw_max,
            "raw_profile_max_over_cutoff": raw_max / float(cutoff),
            "maximum_clipped_tokens": clipped_max,
            "profile_path": str(profile_path.resolve()),
            "profile_sha256": sha256_file(profile_path),
        },
        "recalibration": {
            "origin": origin,
            "campaign_id": job.get("campaign_id"),
            "job_id": job.get("job_id"),
            "observation_id": observation.get("observation_id"),
            "padding": padding,
        },
    }
    record["cluster_id"] = _cluster_id(record) + f"-{_outcome(record)}"
    return record


def _paths_from_queue(rows: Sequence[Mapping[str, Any]]) -> set[Path]:
    return {
        Path(str(row["data_path"])).resolve()
        for row in rows
        if row.get("data_path")
    }


def _content_union(paths: Sequence[Path] | set[Path]) -> set[str]:
    values: set[str] = set()
    for path in sorted(set(paths), key=str):
        values |= _content_hashes(path)
    return values


def _content_audit(
    *,
    training_paths: set[Path],
    campaigns: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    training_hashes = _content_union(training_paths)
    sources: list[dict[str, Any]] = []
    for campaign in campaigns:
        by_source: dict[str, set[Path]] = defaultdict(set)
        for job in campaign["queue_rows"]:
            partition = job.get("calibration_partition") or {}
            source_id = (
                partition.get("split_unit_id")
                or str(job.get("dataset_id") or "").split("__", 1)[0]
            )
            by_source[str(source_id)].add(Path(str(job["data_path"])).resolve())
        for source_id, paths in sorted(by_source.items()):
            hashes = _content_union(paths)
            overlap = len(hashes & training_hashes)
            sources.append(
                {
                    "campaign": campaign["name"],
                    "source_id": source_id,
                    "data_paths": [str(path) for path in sorted(paths, key=str)],
                    "unique_samples": len(hashes),
                    "fit_sample_overlap": overlap,
                    "strict_dataset_disjoint": overlap == 0,
                }
            )
    return {
        "training_data_paths": len(training_paths),
        "training_unique_samples": len(training_hashes),
        "sources": sources,
        "strict_source_ids": sorted(
            row["source_id"] for row in sources if row["strict_dataset_disjoint"]
        ),
        "overlap_source_ids": sorted(
            row["source_id"] for row in sources if not row["strict_dataset_disjoint"]
        ),
    }


def _metrics_by_source(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[str(row["source_id"])].append(row)
    return {source_id: _evaluate(rows) for source_id, rows in sorted(grouped.items())}


def _score(
    records: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    selected: str,
) -> list[dict[str, Any]]:
    if selected == S5_LORA_FULL_ADMISSION:
        return [_v5_detail(row, bundle) for row in records]
    if selected == S4_SEPARATE_ADMISSION:
        return [_separate_admission_detail(row, bundle) for row in records]
    return [
        _detail(
            row,
            _predict_safety(row, bundle, safety_variant=selected),
            safety_variant=selected,
        )
        for row in records
    ]


def _origin_metrics(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[str(row.get("origin"))].append(row)
    return {
        origin: {
            "all_mechanisms": _evaluate(rows),
            "critical_lora": _evaluate(
                [row for row in rows if row.get("critical_lora") is True]
            ),
        }
        for origin, rows in sorted(grouped.items())
    }


def _gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "prediction_availability_100_percent": math.isclose(
            float(metrics.get("prediction_availability") or 0.0), 1.0
        ),
        "source_equal_upper_coverage_at_least_95_percent": float(
            metrics.get("reserved_upper_source_equal_coverage") or 0.0
        )
        >= 0.95,
        "unsafe_success_admitted_zero": int(
            metrics.get("unsafe_success_admitted") or 0
        )
        == 0,
        "false_safe_oom_zero": int(metrics.get("false_safe_oom") or 0) == 0,
    }
    return {"checks": checks, "all_passed": all(checks.values())}


def _metric_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, float | None]:
    keys = (
        "allocated_center_source_equal_mape",
        "reserved_center_source_equal_mape",
        "reserved_upper_source_equal_coverage",
        "admission_recall",
        "unsafe_success_admitted",
        "false_safe_oom",
    )
    result: dict[str, float | None] = {}
    for key in keys:
        old = before.get(key)
        new = after.get(key)
        result[key] = (
            float(new) - float(old) if old is not None and new is not None else None
        )
    return result


def _select_with_separate_admission(
    scope_metrics: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    selection = copy.deepcopy(dict(baseline))
    s4_metrics = scope_metrics[S4_SEPARATE_ADMISSION][
        "selection_scope_combined_critical_lora"
    ]
    s4_gate = _gate(s4_metrics)
    selection["gates"][S4_SEPARATE_ADMISSION] = s4_gate
    selection["selectable_variants"] = [
        *selection["selectable_variants"],
        S4_SEPARATE_ADMISSION,
    ]
    eligible = list(selection["eligible_variants"])
    if s4_gate["all_passed"]:
        eligible.append(S4_SEPARATE_ADMISSION)
    selection["eligible_variants"] = eligible
    ranked: list[tuple[float, int, str]] = []
    for variant in eligible:
        recall = scope_metrics[variant][
            "selection_scope_combined_critical_lora"
        ].get("admission_recall")
        if recall is not None:
            ranked.append(
                (
                    -float(recall),
                    0 if variant == S4_SEPARATE_ADMISSION else 1,
                    variant,
                )
            )
    selection["selected_variant"] = min(ranked)[2] if ranked else None
    selection["policy"] = (
        str(selection["policy"])
        + "; S4 separates the memory upper from binary admission and is ranked "
        "by the same gates and safe-success recall"
    )
    return selection


def _is_supported_full_detail(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("training_mode") == "full"
        and int(row.get("zero_stage") or 0) == 3
        and bool(row.get("gradient_checkpointing"))
        and int(row.get("gpu_count") or 0) == 2
        and not bool(row.get("packing"))
    )


def _is_supported_lora_four_gpu_detail(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("training_mode") == "lora"
        and int(row.get("zero_stage") or 0) == 2
        and not bool(row.get("gradient_checkpointing"))
        and int(row.get("gpu_count") or 0) == 4
        and not bool(row.get("packing"))
    )


def _full_metrics(
    details: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant, rows in details.items():
        full = [row for row in rows if row.get("training_mode") == "full"]
        supported = [row for row in full if _is_supported_full_detail(row)]
        supported_lora_four = [
            row for row in rows if _is_supported_lora_four_gpu_detail(row)
        ]
        result[variant] = {
            "all_full_diagnostic": _evaluate(full),
            "supported_full_zero3_gc_two_gpu": _evaluate(supported),
            "supported_lora_zero2_no_gc_four_gpu": _evaluate(
                supported_lora_four
            ),
        }
    return result


def _select_with_full_admission(
    scope_metrics: Mapping[str, Any],
    full_metrics: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    selection = copy.deepcopy(dict(baseline))
    critical = scope_metrics[S5_LORA_FULL_ADMISSION][
        "selection_scope_combined_critical_lora"
    ]
    supported_full = full_metrics[S5_LORA_FULL_ADMISSION][
        "supported_full_zero3_gc_two_gpu"
    ]
    critical_gate = _gate(critical)
    full_gate = _gate(supported_full)
    supported_lora_four = full_metrics[S5_LORA_FULL_ADMISSION][
        "supported_lora_zero2_no_gc_four_gpu"
    ]
    lora_four_gate = _gate(supported_lora_four)
    combined_pass = (
        critical_gate["all_passed"]
        and full_gate["all_passed"]
        and lora_four_gate["all_passed"]
    )
    selection["gates"][S5_LORA_FULL_ADMISSION] = {
        "critical_lora": critical_gate,
        "supported_full": full_gate,
        "supported_lora_zero2_no_gc_four_gpu": lora_four_gate,
        "all_passed": combined_pass,
    }
    selection["selectable_variants"] = [
        *selection["selectable_variants"],
        S5_LORA_FULL_ADMISSION,
    ]
    eligible = list(selection["eligible_variants"])
    if combined_pass:
        eligible.append(S5_LORA_FULL_ADMISSION)
    selection["eligible_variants"] = eligible
    if combined_pass:
        selection["selected_variant"] = S5_LORA_FULL_ADMISSION
    selection["policy"] = (
        str(selection["policy"])
        + "; S5 must additionally pass the supported FULL nested gate and the "
        "supported LoRA ZeRO-2/GC-off/four-GPU nested gate: availability=100%, "
        "upper coverage>=95%, zero unsafe-success and zero OOM admission"
    )
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--theory-basis", type=Path, default=DEFAULT_THEORY_BASIS)
    parser.add_argument(
        "--dataset-analysis", type=Path, default=DEFAULT_DATASET_ANALYSIS
    )
    parser.add_argument("--new-queue", type=Path, default=DEFAULT_NEW_QUEUE)
    parser.add_argument("--current-old", type=Path, default=DEFAULT_CURRENT_OLD)
    parser.add_argument("--stage2-queue", type=Path, default=DEFAULT_STAGE2_QUEUE)
    parser.add_argument(
        "--inventory", type=Path, default=DEFAULT_VALIDATION_INVENTORY
    )
    parser.add_argument(
        "--previous-candidate", type=Path, default=DEFAULT_PREVIOUS_CANDIDATE
    )
    parser.add_argument("--hardware", type=Path, default=DEFAULT_HARDWARE)
    parser.add_argument(
        "--frozen-baseline", type=Path, default=DEFAULT_FROZEN_BASELINE
    )
    parser.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    parser.add_argument(
        "--separate-admission-head-v4",
        action="store_true",
        help=(
            "evaluate and select the non-stacked direct-reserved upper plus "
            "separate binary admission head"
        ),
    )
    parser.add_argument(
        "--full-admission-head-v5",
        action="store_true",
        help=(
            "extend separate admission to supported FULL ZeRO-3 + GC + two-GPU "
            "configurations and fail closed for other FULL mechanisms"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not 0.5 < args.coverage < 1.0:
        raise ValueError("coverage must be strictly between 0.5 and 1")

    prepared = _prepare_records(args)
    base_training = list(prepared["training"])
    historical_holdout = list(prepared["holdout"])

    inventory = read_json(args.inventory)
    model_by_id, fixed_lora = _inventory_models(inventory)
    hardware = read_json(args.hardware)
    padding_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    raw_max_cache: dict[str, int] = {}

    stage2_queue = _read_jsonl(args.stage2_queue)
    if len(stage2_queue) != 60:
        raise ValueError("stage-two queue must contain exactly 60 jobs")
    print("exporting exact stage-two 60-job observations", flush=True)
    stage2_observations = _export_exact_jobs(stage2_queue)
    if Counter((row.get("outcome") or {}).get("class") for row in stage2_observations) != {
        "success": 57,
        "oom": 3,
    }:
        raise ValueError("stage-two terminal outcomes drifted")
    stage2_cutoff, stage2_effective = _build_pairs(
        stage2_observations,
        model_by_id=model_by_id,
        fixed_lora=fixed_lora,
        hardware=hardware,
        origin="stage2_20_new_sources",
        padding_cache=padding_cache,
        raw_max_cache=raw_max_cache,
    )
    _, stage2_collapsed, stage2_collapse = _collapse_records(
        stage2_cutoff, stage2_effective
    )
    training = [*base_training, *stage2_collapsed]
    base_sources = {_source_id(row) for row in base_training}
    stage2_sources = {_source_id(row) for row in stage2_collapsed}
    if base_sources & stage2_sources:
        raise ValueError("stage-two source ids overlap the prior fit")
    if len(stage2_sources) != 20:
        raise ValueError("stage two must contribute exactly 20 sources")
    cluster_ids = [str(row["cluster_id"]) for row in training]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("fit configuration ids overlap after stage-two merge")

    registry, _ = _dataset_registry(args.dataset_analysis)
    training_paths = {Path(row["data_path"]).resolve() for row in registry.values()}
    stage1_queue = _read_jsonl(args.new_queue)
    training_paths |= _paths_from_queue(stage1_queue)
    training_paths |= _paths_from_queue(stage2_queue)
    old_observations = _read_jsonl(args.current_old)
    training_paths |= set(_current_data_paths([], old_observations).values())

    campaigns: list[dict[str, Any]] = []
    for spec in HOLDOUT_CAMPAIGNS:
        queue_rows = _read_jsonl(Path(spec["queue"]))
        snapshot = read_json(Path(spec["snapshot"]))
        if snapshot.get("immutable") is not True:
            raise ValueError(f"holdout snapshot is not immutable: {spec['snapshot']}")
        if snapshot.get("source_queue_sha256") != sha256_file(Path(spec["queue"])):
            raise ValueError(f"holdout snapshot queue binding drifted: {spec['name']}")
        if len(snapshot.get("rows") or []) != len(queue_rows):
            raise ValueError(f"holdout row count drifted: {spec['name']}")
        campaigns.append({**dict(spec), "queue_rows": queue_rows})
    content_audit = _content_audit(
        training_paths=training_paths,
        campaigns=campaigns,
    )
    strict_source_ids = set(content_audit["strict_source_ids"])
    overlap_source_ids = set(content_audit["overlap_source_ids"])
    if len(strict_source_ids) != 5 or len(overlap_source_ids) != 4:
        raise ValueError(
            "expected five strict unused sources and four later-consumed replay sources"
        )

    strict_validation: list[dict[str, Any]] = []
    overlap_replay: list[dict[str, Any]] = []
    holdout_bindings: list[dict[str, Any]] = []
    for campaign in campaigns:
        observations = _export_validation_jobs(campaign["queue_rows"])
        built = [
            _build_validation_record(
                observation,
                model_by_id=model_by_id,
                fixed_lora=fixed_lora,
                hardware=hardware,
                origin=str(campaign["name"]),
                padding_cache=padding_cache,
                raw_max_cache=raw_max_cache,
            )
            for observation in observations
        ]
        if len({str(row["job_id"]) for row in built}) != len(built):
            raise ValueError(f"duplicate holdout job ids: {campaign['name']}")
        feature_counts = Counter(str(row["cluster_id"]) for row in built)
        feature_collision_rows = sum(
            count for count in feature_counts.values() if count > 1
        )
        for row in built:
            if _source_id(row) in strict_source_ids:
                strict_validation.append(row)
            elif _source_id(row) in overlap_source_ids:
                overlap_replay.append(row)
            else:
                raise ValueError(f"unclassified validation source: {_source_id(row)}")
        holdout_bindings.append(
            {
                "campaign": campaign["name"],
                "queue": _input_binding(Path(campaign["queue"])),
                "snapshot": _input_binding(Path(campaign["snapshot"])),
                "observations": len(observations),
                "outcomes": dict(Counter(_outcome(row) for row in built)),
                "unique_m1_feature_configurations": len(feature_counts),
                "m1_feature_collision_rows": feature_collision_rows,
                "feature_collision_interpretation": (
                    "distinct validation datasets/jobs can share every M1 feature; "
                    "retain every observation and expose this as model information loss"
                ),
            }
        )

    fit_source_ids = {_source_id(row) for row in training}
    if fit_source_ids & {_source_id(row) for row in strict_validation}:
        raise ValueError("strict validation source ids overlap fit sources")
    if {str(row["cluster_id"]) for row in training} & {
        str(row["cluster_id"]) for row in strict_validation
    }:
        raise ValueError("strict validation configurations overlap fit configurations")

    print(
        f"fit unique={len(training)} sources={len(fit_source_ids)}; "
        f"strict validation={len(strict_validation)} sources="
        f"{len({_source_id(row) for row in strict_validation})}; "
        f"historical config holdout={len(historical_holdout)}",
        flush=True,
    )
    nested = _nested_safety_ablation(training, coverage=args.coverage)
    separate_nested = None
    v5_nested = None
    if args.full_admission_head_v5:
        v5_nested = _nested_v5(training, coverage=args.coverage)
        nested["details"][S5_LORA_FULL_ADMISSION] = v5_nested["details"]
    elif args.separate_admission_head_v4:
        separate_nested = _nested_separate_admission(
            training,
            coverage=args.coverage,
        )
        nested["details"][S4_SEPARATE_ADMISSION] = separate_nested["details"]
    scope_metrics = _scope_metrics(nested["details"])
    training_full_metrics = _full_metrics(nested["details"])
    selection = _select_safety_variant(scope_metrics)
    if args.full_admission_head_v5:
        selection = _select_with_full_admission(
            scope_metrics,
            training_full_metrics,
            selection,
        )
    elif args.separate_admission_head_v4:
        selection = _select_with_separate_admission(scope_metrics, selection)
    selected = selection["selected_variant"]
    if selected is None:
        raise ValueError("no safety variant passed nested training-only gates")
    full_bundle = _fit_safety_bundle(
        training,
        coverage=args.coverage,
        diagnostic_max_fallback=False,
    )
    if args.full_admission_head_v5:
        _fit_v5_heads(training, full_bundle)
    elif args.separate_admission_head_v4:
        full_bundle["separate_admission_head"] = _fit_admission_head(
            training,
            full_bundle,
        )

    strict_details = _score(strict_validation, full_bundle, selected)
    historical_details = _score(historical_holdout, full_bundle, selected)
    overlap_details = _score(overlap_replay, full_bundle, selected)
    strict_critical = [
        row for row in strict_details if row.get("critical_lora") is True
    ]
    historical_critical = [
        row for row in historical_details if row.get("critical_lora") is True
    ]
    combined_details = [*strict_details, *historical_details]
    combined_critical = [
        row for row in combined_details if row.get("critical_lora") is True
    ]

    def mechanism_scopes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        full = [row for row in rows if row.get("training_mode") == "full"]
        supported_full = [row for row in full if _is_supported_full_detail(row)]
        supported_lora_four = [
            row for row in rows if _is_supported_lora_four_gpu_detail(row)
        ]
        return {
            "full": _evaluate(full),
            "supported_full_zero3_gc_two_gpu": _evaluate(supported_full),
            "supported_lora_zero2_no_gc_four_gpu": _evaluate(
                supported_lora_four
            ),
        }

    validation_metrics = {
        "strict_unused_datasets": {
            "all_mechanisms": _evaluate(strict_details),
            "critical_lora": _evaluate(strict_critical),
            "by_campaign": _origin_metrics(strict_details),
            "by_source": _metrics_by_source(strict_details),
            "gate_all_mechanisms": _gate(_evaluate(strict_details)),
            "gate_critical_lora": _gate(_evaluate(strict_critical)),
            **mechanism_scopes(strict_details),
        },
        "historical_unfitted_configurations": {
            "all_mechanisms": _evaluate(historical_details),
            "critical_lora": _evaluate(historical_critical),
            "by_source": _metrics_by_source(historical_details),
            "dataset_level_independent": False,
            **mechanism_scopes(historical_details),
        },
        "all_unfitted_observations_combined": {
            "all_mechanisms": _evaluate(combined_details),
            "critical_lora": _evaluate(combined_critical),
            "note": (
                "includes strict unused datasets and historical configurations; "
                "the historical portion is not dataset-level independent"
            ),
            **mechanism_scopes(combined_details),
        },
        "overlap_replay_not_validation": {
            "all_mechanisms": _evaluate(overlap_details),
            "critical_lora": _evaluate(
                [row for row in overlap_details if row.get("critical_lora") is True]
            ),
            "by_source": _metrics_by_source(overlap_details),
            **mechanism_scopes(overlap_details),
        },
    }

    previous_candidate = read_json(args.previous_candidate)
    previous_selected = str(previous_candidate["selected_safety_variant"])
    previous_bundle = previous_candidate["model"]
    previous_strict_details = _score(
        strict_validation, previous_bundle, previous_selected
    )
    previous_historical_details = _score(
        historical_holdout, previous_bundle, previous_selected
    )
    previous_combined_details = [
        *previous_strict_details,
        *previous_historical_details,
    ]
    baseline_comparison: dict[str, Any] = {
        "previous_candidate": {
            "safety_variant": previous_selected,
            "source_count": previous_candidate.get("source_count"),
            "candidate_sha256": previous_candidate.get("candidate_sha256"),
        },
        "scopes": {},
    }
    for scope, before_details, after_details in (
        ("strict_unused_datasets", previous_strict_details, strict_details),
        (
            "historical_unfitted_configurations",
            previous_historical_details,
            historical_details,
        ),
        (
            "all_unfitted_observations_combined",
            previous_combined_details,
            combined_details,
        ),
    ):
        before = _evaluate(before_details)
        after = _evaluate(after_details)
        before_critical = _evaluate(
            [row for row in before_details if row.get("critical_lora") is True]
        )
        after_critical = _evaluate(
            [row for row in after_details if row.get("critical_lora") is True]
        )
        before_full = _evaluate(
            [row for row in before_details if row.get("training_mode") == "full"]
        )
        after_full = _evaluate(
            [row for row in after_details if row.get("training_mode") == "full"]
        )
        before_supported_full = _evaluate(
            [row for row in before_details if _is_supported_full_detail(row)]
        )
        after_supported_full = _evaluate(
            [row for row in after_details if _is_supported_full_detail(row)]
        )
        baseline_comparison["scopes"][scope] = {
            "before": before,
            "after": after,
            "after_minus_before": _metric_delta(before, after),
            "critical_lora": {
                "before": before_critical,
                "after": after_critical,
                "after_minus_before": _metric_delta(
                    before_critical, after_critical
                ),
            },
            "full": {
                "before": before_full,
                "after": after_full,
                "after_minus_before": _metric_delta(before_full, after_full),
            },
            "supported_full_zero3_gc_two_gpu": {
                "before": before_supported_full,
                "after": after_supported_full,
                "after_minus_before": _metric_delta(
                    before_supported_full, after_supported_full
                ),
            },
        }

    if selected in (S4_SEPARATE_ADMISSION, S5_LORA_FULL_ADMISSION):
        critical_key = json.dumps(
            ["lora", 2, False, 2, False], separators=(",", ":")
        )
        tail = {
            "path": "direct_reserved_success_residual_separate_from_admission",
            **dict(
                (full_bundle["reserved_residual_upper"].get("exact") or {}).get(
                    critical_key
                )
                or {}
            ),
        }
    else:
        tail = _critical_tail_entry(full_bundle, selected)
    training_outcomes = dict(Counter(_outcome(row) for row in training))
    if args.full_admission_head_v5:
        run_schema = SCHEMA_V5
        implementation_version = IMPLEMENTATION_VERSION_V5
    elif args.separate_admission_head_v4:
        run_schema = SCHEMA_V4
        implementation_version = IMPLEMENTATION_VERSION_V4
    else:
        run_schema = SCHEMA
        implementation_version = IMPLEMENTATION_VERSION
    report: dict[str, Any] = {
        "schema": run_schema,
        "implementation_version": implementation_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "shadow_refit_evaluated",
        "publishable": False,
        "production_model_mutated": False,
        "center_model": {
            "variant": VARIANT_M1,
            "effective_sequence_policy": (
                "round_up(min(cutoff_len, maximum_clipped_profile_length), 8)"
            ),
            "oom_treatment": (
                "right-censored lower-bound safety constraint; never a center label"
            ),
            "admission_design": (
                "separate LoRA and supported-FULL risk heads; unsupported FULL "
                "fails closed; no allocated-tail times reservation-expansion-tail "
                "product"
                if args.full_admission_head_v5
                else (
                    "separate source-balanced binary risk head; no allocated-tail "
                    "times reservation-expansion-tail product"
                    if args.separate_admission_head_v4
                    else "selected legacy safety variant"
                )
            ),
        },
        "fit_set": {
            "raw_observations": 158 + 105 + 60,
            "unique_configuration_results": len(training),
            "independent_sources": len(fit_source_ids),
            "outcomes": training_outcomes,
            "source_ids": sorted(fit_source_ids),
            "dataset_ids": sorted(
                {str((row.get("scenario") or {}).get("dataset_id")) for row in training}
            ),
            "components": {
                "prior_fit": prepared["audit"],
                "stage2": {
                    "raw_observations": len(stage2_observations),
                    "sources": len(stage2_sources),
                    "collapse": stage2_collapse,
                },
            },
        },
        "validation_set": {
            "strict_unused_dataset_sources": sorted(strict_source_ids),
            "strict_unused_dataset_configurations": len(strict_validation),
            "historical_unfitted_configuration_results": len(historical_holdout),
            "later_consumed_replay_sources_excluded_from_validation": sorted(
                overlap_source_ids
            ),
            "holdout_bindings": holdout_bindings,
            "content_overlap_audit": content_audit,
            "duplicate_subset_queues_not_double_counted": [
                "matrix/h800_fresh_holdout_remaining_0_3_v2.jsonl",
                "matrix/h800_bounded_memory_v2_fresh_holdout_canary_v1.jsonl",
                "matrix/h800_bounded_memory_v2_fresh_holdout_formal_v1.jsonl",
            ],
            "out_of_scope_holdouts": {
                "matrix/profiler_holdout_jobs.jsonl": "profiler route, not memory admission",
                "matrix/h800_thermal_validation_jobs.jsonl": "thermal validation route",
            },
        },
        "training_nested_protocol": nested["protocol"],
        "separate_admission_nested_protocol": (
            v5_nested["protocol"]
            if v5_nested
            else (separate_nested["protocol"] if separate_nested else None)
        ),
        "training_nested_folds": nested["folds"],
        "training_nested_metrics": scope_metrics,
        "training_nested_full_metrics": training_full_metrics,
        "selection": selection,
        "selected_full_fit": {
            "safety_variant": selected,
            "critical_tail": tail,
            "bundle": full_bundle,
        },
        "validation_metrics": validation_metrics,
        "previous_m1_comparison_on_identical_validation": baseline_comparison,
        "release_interpretation": {
            "strict_unused_dataset_gate_passed": validation_metrics[
                "strict_unused_datasets"
            ]["gate_all_mechanisms"]["all_passed"],
            "strict_unused_critical_lora_gate_passed": validation_metrics[
                "strict_unused_datasets"
            ]["gate_critical_lora"]["all_passed"],
            "strict_unused_supported_full_gate_passed": _gate(
                validation_metrics["strict_unused_datasets"][
                    "supported_full_zero3_gc_two_gpu"
                ]
            )["all_passed"],
            "production_replacement_allowed": False,
            "reason": (
                "all strict validation campaigns predate this refit and have already "
                "been inspected; use as complete diagnostic evidence, not a new "
                "post-freeze prospective acceptance"
            ),
        },
        "inputs": {
            "canonical": _input_binding(args.canonical),
            "theory_basis": _input_binding(args.theory_basis),
            "dataset_analysis": _input_binding(args.dataset_analysis),
            "stage1_queue": _input_binding(args.new_queue),
            "current_old": _input_binding(args.current_old),
            "stage2_queue": _input_binding(args.stage2_queue),
            "inventory": _input_binding(args.inventory),
            "previous_candidate": _input_binding(args.previous_candidate),
            "hardware": _input_binding(args.hardware),
            "frozen_baseline": _input_binding(args.frozen_baseline),
        },
    }
    report["report_sha256"] = sha256_json(report)

    candidate: dict[str, Any] = {
        "schema": (
            "sft_h800_m1_lora_full_admission_shadow_candidate/v5"
            if args.full_admission_head_v5
            else (
                "sft_h800_m1_separate_admission_shadow_candidate/v4"
                if args.separate_admission_head_v4
                else CANDIDATE_SCHEMA
            )
        ),
        "implementation_version": implementation_version,
        "generated_at_utc": report["generated_at_utc"],
        "status": "shadow_candidate_evaluated_not_published",
        "publishable": False,
        "production_override_allowed": False,
        "scope": {
            "gpu_family": "H800",
            "supported_routes": (
                [
                    {
                        "training_mode": "lora",
                        "zero_stage": 2,
                        "gradient_checkpointing": False,
                        "gpu_count": 2,
                        "packing": False,
                    },
                    {
                        "training_mode": "full",
                        "zero_stage": 3,
                        "gradient_checkpointing": True,
                        "gpu_count": 2,
                        "packing": False,
                    },
                ]
                if args.full_admission_head_v5
                else [
                    {
                        "training_mode": "lora",
                        "zero_stage": 2,
                        "gradient_checkpointing": False,
                        "gpu_count": 2,
                        "packing": False,
                    }
                ]
            ),
            "packing": False,
            "outside_scope_policy": (
                "FULL fails closed; other routes keep current frozen model"
                if args.full_admission_head_v5
                else "keep_current_frozen_model"
            ),
        },
        "center_variant": VARIANT_M1,
        "selected_safety_variant": selected,
        "source_count": len(fit_source_ids),
        "strict_unused_validation_sources": len(strict_source_ids),
        "refit_report_sha256": report["report_sha256"],
        "model": full_bundle,
    }
    candidate["candidate_sha256"] = sha256_json(candidate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "refit_and_all_unused_validation_report.json"
    candidate_path = args.output_dir / (
        "candidate_model_m1_lora_full_admission_v5.json"
        if args.full_admission_head_v5
        else (
            "candidate_model_m1_separate_admission_v4.json"
            if args.separate_admission_head_v4
            else "candidate_model_m1_refit_v3.json"
        )
    )
    training_path = args.output_dir / "training_nested_oof_predictions.jsonl"
    validation_path = args.output_dir / "all_unused_validation_predictions.jsonl"
    replay_path = args.output_dir / "overlap_replay_predictions_not_validation.jsonl"
    write_json(report_path, report)
    write_json(candidate_path, candidate)
    write_jsonl(
        training_path,
        [
            row
            for variant in (
                (*SAFETY_VARIANTS, S5_LORA_FULL_ADMISSION)
                if args.full_admission_head_v5
                else (
                    (*SAFETY_VARIANTS, S4_SEPARATE_ADMISSION)
                    if args.separate_admission_head_v4
                    else SAFETY_VARIANTS
                )
            )
            for row in nested["details"][variant]
        ],
    )
    write_jsonl(
        validation_path,
        [
            {**dict(row), "validation_scope": "strict_unused_dataset"}
            for row in strict_details
        ]
        + [
            {**dict(row), "validation_scope": "historical_unfitted_configuration"}
            for row in historical_details
        ],
    )
    write_jsonl(replay_path, overlap_details)
    manifest = {
        "schema": (
            "sft_h800_m1_lora_full_admission_manifest/v5"
            if args.full_admission_head_v5
            else (
                "sft_h800_m1_separate_admission_manifest/v4"
                if args.separate_admission_head_v4
                else "sft_h800_m1_all_unused_validation_manifest/v3"
            )
        ),
        "generated_at_utc": report["generated_at_utc"],
        "selected_safety_variant": selected,
        "candidate_sha256": candidate["candidate_sha256"],
        "files": {
            path.name: _input_binding(path)
            for path in (
                report_path,
                candidate_path,
                training_path,
                validation_path,
                replay_path,
            )
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(args.output_dir / "output_manifest.json", manifest)

    strict_metrics = validation_metrics["strict_unused_datasets"]["all_mechanisms"]
    critical_metrics = validation_metrics["strict_unused_datasets"]["critical_lora"]
    print(
        "selected="
        f"{selected} strict_sources={len(strict_source_ids)} "
        f"strict_reserved_mape={strict_metrics.get('reserved_center_source_equal_mape')} "
        f"strict_coverage={strict_metrics.get('reserved_upper_source_equal_coverage')} "
        f"strict_recall={strict_metrics.get('admission_recall')} "
        f"critical_coverage={critical_metrics.get('reserved_upper_source_equal_coverage')} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
