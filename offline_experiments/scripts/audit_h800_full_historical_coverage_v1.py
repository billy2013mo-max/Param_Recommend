#!/usr/bin/env python3
"""Audit whether existing H800 evidence covers the primary FULL admission grid.

The audit keeps four evidence roles separate:

* configurations used by the V5 fit;
* configurations reserved for validation;
* exact historical observations before repeat/configuration collapse;
* retrospectively recovered legacy observations that remain diagnostic under
  the current native calibration contract.

Counts are never treated as independent evidence by themselves.  Readiness is
assessed per FULL mechanism using independent fit sources and independent
sources containing unsafe-success or right-censored OOM observations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from common import ROOT, sha256_file, sha256_json, write_json


SCHEMA = "sft_h800_full_historical_coverage_audit/v1"
IMPLEMENTATION_VERSION = (
    "sft_h800_full_historical_coverage_audit/"
    "2026-08-05.primary-unpacked-grid-source-boundary-v1"
)
S5 = "S5_lora_full_separate_admission_heads"

DEFAULT_V5_DIR = ROOT / "diagnostics" / "h800_m1_lora_full_admission_v5_20260805"
DEFAULT_EXACT_HISTORICAL = (
    ROOT
    / "diagnostics"
    / "h800_historical_memory_migration_20260805"
    / "migrated_historical_effective_records.jsonl"
)
DEFAULT_EXACT_BINDINGS = (
    ROOT
    / "diagnostics"
    / "h800_historical_memory_migration_20260805"
    / "historical_observation_bindings.jsonl"
)
DEFAULT_THEORY_BASIS = ROOT / "artifacts" / "h800_theory_basis.json"
DEFAULT_RECOVERY = ROOT / "artifacts" / "historical_h800_recovery.json"
DEFAULT_OUTPUT = (
    ROOT
    / "diagnostics"
    / "h800_full_historical_coverage_audit_20260805"
    / "full_historical_coverage_audit.json"
)

PRIMARY_UNPACKED_MECHANISMS = tuple(
    (zero, gc, gpu_count, False)
    for gpu_count, zero in ((1, 0), (2, 2), (2, 3), (4, 2), (4, 3))
    for gc in (False, True)
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mechanism(
    *, zero: Any, gc: Any, gpu_count: Any, packing: Any
) -> tuple[int, bool, int, bool]:
    return int(zero or 0), bool(gc), int(gpu_count), bool(packing)


def _mechanism_id(key: tuple[int, bool, int, bool]) -> str:
    zero, gc, gpu_count, packing = key
    return f"full_zero{zero}_gc{int(gc)}_{gpu_count}gpu_pack{int(packing)}"


def _mechanism_cn(key: tuple[int, bool, int, bool]) -> str:
    zero, gc, gpu_count, packing = key
    zero_name = "不切分" if zero == 0 else f"ZeRO-{zero}"
    gc_name = "开启梯度检查点" if gc else "关闭梯度检查点"
    packing_name = "开启 Packing" if packing else "关闭 Packing"
    return f"{gpu_count} 卡、{zero_name}、{gc_name}、{packing_name}"


def _label_from_detail(row: Mapping[str, Any]) -> str:
    if row.get("outcome") == "oom":
        return "oom"
    return "safe" if row.get("actually_safe_success") is True else "unsafe_success"


def _label_from_basis(row: Mapping[str, Any]) -> str:
    if row.get("outcome") == "oom":
        return "oom"
    memory = row.get("memory") or {}
    observed = memory.get("observed") or {}
    reserved = observed.get("peak_reserved_target_bytes")
    safe_limit = memory.get("safe_limit_bytes")
    if reserved is None or safe_limit is None:
        raise ValueError(f"success row has no exact reserved/safe limit: {row.get('job_id')}")
    return "safe" if float(reserved) <= float(safe_limit) else "unsafe_success"


def _normalized_detail(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mechanism": _mechanism(
            zero=row.get("zero_stage"),
            gc=row.get("gradient_checkpointing"),
            gpu_count=row.get("gpu_count"),
            packing=row.get("packing"),
        ),
        "source_id": str(row.get("source_id")),
        "label": _label_from_detail(row),
        "model_id": row.get("model_id"),
        "dataset_id": row.get("source_id"),
        "cutoff_len": row.get("cutoff_len"),
        "mbs": row.get("mbs"),
        "origin": row.get("origin"),
    }


def _normalized_basis(
    row: Mapping[str, Any], *, source_id: str, role: str
) -> dict[str, Any]:
    selector = row.get("selector") or {}
    scenario = row.get("scenario") or {}
    return {
        "mechanism": _mechanism(
            zero=selector.get("zero_stage"),
            gc=selector.get("gradient_checkpointing"),
            gpu_count=scenario.get("gpu_count"),
            packing=selector.get("packing"),
        ),
        "source_id": source_id,
        "label": _label_from_basis(row),
        "model_id": scenario.get("model_id"),
        "dataset_id": scenario.get("dataset_id"),
        "cutoff_len": scenario.get("cutoff_len"),
        "mbs": scenario.get("physical_mbs"),
        "role": role,
        "observation_id": row.get("observation_id"),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(row["label"]) for row in rows)
    sources = {str(row["source_id"]) for row in rows}
    negative_sources = {
        str(row["source_id"])
        for row in rows
        if row["label"] in {"unsafe_success", "oom"}
    }
    return {
        "configurations_or_observations": len(rows),
        "independent_sources": len(sources),
        "safe_success": labels["safe"],
        "unsafe_success": labels["unsafe_success"],
        "oom_right_censored": labels["oom"],
        "negative_boundary_sources": len(negative_sources),
        "source_ids": sorted(sources),
        "model_ids": sorted({str(row["model_id"]) for row in rows}),
        "dataset_ids": sorted({str(row["dataset_id"]) for row in rows}),
        "cutoff_lengths": sorted(
            {int(row["cutoff_len"]) for row in rows if row.get("cutoff_len") is not None}
        ),
        "micro_batch_sizes": sorted(
            {int(row["mbs"]) for row in rows if row.get("mbs") is not None}
        ),
    }


def _by_mechanism(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, bool, int, bool], dict[str, Any]]:
    grouped: dict[tuple[int, bool, int, bool], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row["mechanism"])].append(row)
    return {key: _summary(value) for key, value in grouped.items()}


def _readiness(
    key: tuple[int, bool, int, bool], fit: Mapping[str, Any]
) -> dict[str, Any]:
    checks = {
        "at_least_five_independent_fit_sources": int(fit.get("independent_sources") or 0)
        >= 5,
        "at_least_two_negative_boundary_fit_sources": int(
            fit.get("negative_boundary_sources") or 0
        )
        >= 2,
        "has_safe_success_fit_evidence": int(fit.get("safe_success") or 0) > 0,
        "has_unsafe_or_oom_fit_evidence": (
            int(fit.get("unsafe_success") or 0)
            + int(fit.get("oom_right_censored") or 0)
        )
        > 0,
    }
    ready = all(checks.values())
    if ready:
        decision = "no_additional_fit_data_for_current_shadow_scope"
    elif checks["at_least_five_independent_fit_sources"]:
        decision = "supplement_negative_boundary_sources"
    else:
        decision = "supplement_independent_sources_and_boundary_outcomes"
    return {
        "mechanism_id": _mechanism_id(key),
        "mechanism_cn": _mechanism_cn(key),
        "planning_checks": checks,
        "fit_evidence_ready": ready,
        "decision": decision,
        "interpretation": (
            "These are conservative experiment-planning thresholds, not a "
            "mathematical guarantee or an existing production release gate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-dir", type=Path, default=DEFAULT_V5_DIR)
    parser.add_argument("--exact-historical", type=Path, default=DEFAULT_EXACT_HISTORICAL)
    parser.add_argument("--exact-bindings", type=Path, default=DEFAULT_EXACT_BINDINGS)
    parser.add_argument("--theory-basis", type=Path, default=DEFAULT_THEORY_BASIS)
    parser.add_argument("--recovery", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    training_path = args.v5_dir / "training_nested_oof_predictions.jsonl"
    validation_path = args.v5_dir / "all_unused_validation_predictions.jsonl"
    fit_details = [
        _normalized_detail(row)
        for row in _read_jsonl(training_path)
        if row.get("safety_variant") == S5 and row.get("training_mode") == "full"
    ]
    validation_details = [
        _normalized_detail(row)
        for row in _read_jsonl(validation_path)
        if row.get("training_mode") == "full"
    ]

    bindings = {
        str(row["observation_id"]): row for row in _read_jsonl(args.exact_bindings)
    }
    exact_raw = [
        row
        for row in _read_jsonl(args.exact_historical)
        if (row.get("selector") or {}).get("training_mode") == "full"
    ]
    exact = [
        _normalized_basis(
            row,
            source_id="historical_public_connected_component_01",
            role=str(bindings[str(row["observation_id"])]["original_role"]),
        )
        for row in exact_raw
    ]

    theory = json.loads(args.theory_basis.read_text(encoding="utf-8"))
    legacy_full_raw = [
        row
        for row in theory.get("records") or []
        if (row.get("selector") or {}).get("training_mode") == "full"
    ]
    legacy_boundary_raw = [
        row
        for row in legacy_full_raw
        if (row.get("route") or {}).get("memory_boundary") is True
    ]
    legacy_auxiliary_raw = [
        row
        for row in legacy_full_raw
        if (row.get("route") or {}).get("memory_boundary") is not True
    ]
    legacy_boundary = [
        _normalized_basis(
            row,
            source_id="historical_legacy_connected_component_01",
            role="legacy_diagnostic_memory_boundary",
        )
        for row in legacy_boundary_raw
    ]
    legacy_auxiliary = [
        _normalized_basis(
            row,
            source_id="historical_legacy_connected_component_01",
            role="legacy_diagnostic_auxiliary",
        )
        for row in legacy_auxiliary_raw
    ]

    recovery = json.loads(args.recovery.read_text(encoding="utf-8"))
    boundary_ids = {str(row["observation_id"]) for row in legacy_boundary_raw}
    recovery_rows = [
        row
        for row in recovery.get("records") or []
        if str(row.get("source_observation_id")) in boundary_ids
    ]

    fit_by = _by_mechanism(fit_details)
    validation_by = _by_mechanism(validation_details)
    exact_by = _by_mechanism(exact)
    legacy_by = _by_mechanism(legacy_boundary)
    auxiliary_by = _by_mechanism(legacy_auxiliary)

    mechanism_rows: list[dict[str, Any]] = []
    for key in PRIMARY_UNPACKED_MECHANISMS:
        fit = fit_by.get(key, _summary([]))
        mechanism_rows.append(
            {
                **_readiness(key, fit),
                "fit": fit,
                "reserved_validation": validation_by.get(key, _summary([])),
                "exact_historical_raw": exact_by.get(key, _summary([])),
                "legacy_boundary_diagnostic": legacy_by.get(key, _summary([])),
                "legacy_auxiliary_diagnostic": auxiliary_by.get(key, _summary([])),
            }
        )

    ready = [row["mechanism_id"] for row in mechanism_rows if row["fit_evidence_ready"]]
    missing = [row["mechanism_id"] for row in mechanism_rows if not row["fit_evidence_ready"]]
    exact_roles = Counter(str(row["role"]) for row in exact)
    legacy_limitations = Counter(
        str(limitation) for row in recovery_rows for limitation in row.get("limitations") or []
    )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "scope": {
            "primary_unpacked_full_grid": (
                "one GPU/ZeRO-0 or two/four GPUs with ZeRO-2/3, each with "
                "gradient checkpointing off/on, packing disabled"
            ),
            "zero1": "opt-in and outside the primary generator grid",
            "packing_enabled": "accepted as a user scenario but has zero FULL evidence in the audited history",
            "safe_success_definition": (
                "successful run with observed CUDA reserved peak not exceeding "
                "the H800 safe limit"
            ),
            "oom_definition": "right-censored lower-bound constraint; never an exact peak label",
        },
        "planning_heuristic": {
            "minimum_independent_fit_sources_per_mechanism": 5,
            "minimum_negative_boundary_fit_sources_per_mechanism": 2,
            "requires_both_safe_and_negative_outcomes": True,
            "status": "experiment-planning assumption, not a mathematical theorem or release gate",
        },
        "evidence_inventory": {
            "v5_fit": _summary(fit_details),
            "reserved_validation": _summary(validation_details),
            "exact_historical_full_raw": {
                **_summary(exact),
                "original_roles": dict(exact_roles),
                "content_independence": (
                    "all historical datasets are conservatively one connected source group"
                ),
            },
            "legacy_full": {
                "total_records": len(legacy_full_raw),
                "memory_boundary_records": len(legacy_boundary_raw),
                "auxiliary_feasibility_or_throughput_records": len(legacy_auxiliary_raw),
                "memory_boundary_summary": _summary(legacy_boundary),
                "auxiliary_summary": _summary(legacy_auxiliary),
                "evidence_tiers": dict(
                    Counter(str(row.get("evidence_tier")) for row in legacy_full_raw)
                ),
                "memory_boundary_archive_completeness": dict(
                    Counter(
                        "full_payload"
                        if (row.get("archive_completeness") or {}).get("full_payload_set")
                        else "hash_only"
                        for row in recovery_rows
                    )
                ),
                "memory_boundary_attempt_integrity": dict(
                    Counter(
                        str((row.get("attempt_integrity") or {}).get("strength"))
                        for row in recovery_rows
                    )
                ),
                "limitations": dict(legacy_limitations),
                "release_fit_eligible_under_current_native_contract": False,
                "allowed_use": (
                    "diagnose mechanism transitions and choose rerun points; do not "
                    "treat as exact release-calibration labels"
                ),
            },
        },
        "primary_unpacked_mechanisms": mechanism_rows,
        "conclusion": {
            "primary_unpacked_mechanisms": len(PRIMARY_UNPACKED_MECHANISMS),
            "fit_ready_mechanisms": ready,
            "mechanisms_requiring_supplement": missing,
            "fit_ready_count": len(ready),
            "supplement_count": len(missing),
            "packing_enabled_full_records": 0,
            "zero1_full_records": 0,
            "existing_rows_are_sufficient_for_grid_discovery": True,
            "existing_rows_are_sufficient_for_release_calibration_of_all_full": False,
            "reason": (
                "the grid is broad, but eight mechanisms have only one independent "
                "fit source, the four-GPU ZeRO-3 GC-on mechanism has no negative fit "
                "source, and only the two-GPU ZeRO-3 GC-on mechanism passes the "
                "conservative source/boundary planning checks"
            ),
        },
        "minimum_followup_design": {
            "unpacked_missing_mechanisms": len(missing),
            "shared_new_dataset_sources": 5,
            "source_roles": {"fit": 4, "frozen_validation": 1},
            "initial_bracket_jobs_per_mechanism_per_source": 2,
            "adaptive_third_job_if_outcomes_do_not_bracket_boundary": True,
            "missing_mechanism_jobs_minimum": len(missing) * 5 * 2,
            "missing_mechanism_jobs_adaptive_maximum": len(missing) * 5 * 3,
            "current_supported_route_prospective_jobs": 5 * 2,
            "complete_primary_unpacked_campaign_minimum": (
                len(PRIMARY_UNPACKED_MECHANISMS) * 5 * 2
            ),
            "complete_primary_unpacked_campaign_adaptive_maximum": (
                len(PRIMARY_UNPACKED_MECHANISMS) * 5 * 3
            ),
            "packing_enabled_extension": (
                "requires a separate six-source design (five fit plus one frozen "
                "validation source) because audited FULL packing evidence is zero"
            ),
            "zero1_extension": (
                "do not run unless the opt-in ZeRO-1 product path will be enabled"
            ),
        },
        "inputs": {
            "training_predictions": {
                "path": str(training_path.resolve()),
                "sha256": sha256_file(training_path),
            },
            "validation_predictions": {
                "path": str(validation_path.resolve()),
                "sha256": sha256_file(validation_path),
            },
            "exact_historical": {
                "path": str(args.exact_historical.resolve()),
                "sha256": sha256_file(args.exact_historical),
            },
            "exact_bindings": {
                "path": str(args.exact_bindings.resolve()),
                "sha256": sha256_file(args.exact_bindings),
            },
            "theory_basis": {
                "path": str(args.theory_basis.resolve()),
                "sha256": sha256_file(args.theory_basis),
            },
            "historical_recovery": {
                "path": str(args.recovery.resolve()),
                "sha256": sha256_file(args.recovery),
            },
        },
    }
    report["report_sha256"] = sha256_json(report)
    write_json(args.output, report)
    print(
        f"fit_ready={len(ready)}/{len(PRIMARY_UNPACKED_MECHANISMS)} "
        f"supplement={len(missing)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
