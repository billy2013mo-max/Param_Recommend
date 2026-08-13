#!/usr/bin/env python3
"""Fit and verify per-mechanism admission heads for the newly eligible scopes.

Six mechanisms reached the planning thresholds once the public-corpus token
profiles were linked (their observations carried an empty
``dataset_profile_path``, so 351 rows had been silently unusable).  This script
fits one head per mechanism and reports the verification layers V5 already uses,
so a head is never judged on its own fit data alone:

1. nested leave-one-source-out on the fit set (honest in-domain estimate);
2. every observation the head's own fold never saw;
3. the strictly-unseen business sources from the backfill campaign.

Analysis only.  It writes a shadow report, never a frozen artifact, never a
production model, and never touches GPUs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import fit_h800_m1_separate_admission_v4 as V4
from common import ARTIFACT_DIR, ROOT, read_json, sha256_file, sha256_json, write_json
from fit_h800_lora_source_disjoint_recalibration_v1 import (
    VARIANT_FEATURES,
    VARIANT_M1,
    _fit_bundle,
    _predict_center,
    _source_id,
)
from fit_h800_m1_separate_admission_v4 import (
    _admission_features,
    _fit_admission_head,
    _mechanism_predicate,
    _predict_risk,
)
from h800_native_memory_calibration import _inventory_models
from refit_h800_m1_all_unused_validation_v3 import _build_validation_record


SCHEMA = "sft_h800_m1_new_mechanism_admission_heads/v1"
IMPLEMENTATION_VERSION = (
    "sft_h800_m1_new_mechanism_admission_heads/"
    "2026-08-06.public-corpus-profiles-linked-v1"
)
GIB = float(1 << 30)
SAFE_LIMIT_BYTES = 142635080089.6
COVERAGE = 0.95
MIN_INDEPENDENT_SOURCES = 5
MIN_NEGATIVE_BOUNDARY_SOURCES = 2

DEFAULT_OBSERVATIONS = (
    ARTIFACT_DIR / "canonical_h800_observations_with_negev_v1.jsonl"
)
DEFAULT_PROFILE_DIR = ARTIFACT_DIR / "dataset_profiles"
DEFAULT_OUTPUT_DIR = (
    ROOT / "diagnostics" / "h800_m1_new_mechanism_admission_heads_20260806"
)

# Candidate mechanisms: everything without a head in V5.1.  Each is attempted;
# the report records exactly why a mechanism was rejected rather than dropping it.
CANDIDATES: tuple[dict[str, Any], ...] = (
    {"scope": "lora_zero0_no_gc_one_gpu", "training_mode": "lora", "zero_stage": 0,
     "gradient_checkpointing": False, "gpu_count": 1,
     "label_cn": "LoRA 1卡 不切分 检查点关"},
    {"scope": "lora_zero0_gc_one_gpu", "training_mode": "lora", "zero_stage": 0,
     "gradient_checkpointing": True, "gpu_count": 1,
     "label_cn": "LoRA 1卡 不切分 检查点开"},
    {"scope": "lora_zero2_gc_two_gpu", "training_mode": "lora", "zero_stage": 2,
     "gradient_checkpointing": True, "gpu_count": 2,
     "label_cn": "LoRA 2卡 ZeRO-2 检查点开"},
    {"scope": "lora_zero3_no_gc_two_gpu", "training_mode": "lora", "zero_stage": 3,
     "gradient_checkpointing": False, "gpu_count": 2,
     "label_cn": "LoRA 2卡 ZeRO-3 检查点关"},
    {"scope": "lora_zero3_gc_two_gpu", "training_mode": "lora", "zero_stage": 3,
     "gradient_checkpointing": True, "gpu_count": 2,
     "label_cn": "LoRA 2卡 ZeRO-3 检查点开"},
    {"scope": "lora_zero2_gc_four_gpu", "training_mode": "lora", "zero_stage": 2,
     "gradient_checkpointing": True, "gpu_count": 4,
     "label_cn": "LoRA 4卡 ZeRO-2 检查点开"},
    {"scope": "lora_zero3_no_gc_four_gpu", "training_mode": "lora", "zero_stage": 3,
     "gradient_checkpointing": False, "gpu_count": 4,
     "label_cn": "LoRA 4卡 ZeRO-3 检查点关"},
    {"scope": "lora_zero3_gc_four_gpu", "training_mode": "lora", "zero_stage": 3,
     "gradient_checkpointing": True, "gpu_count": 4,
     "label_cn": "LoRA 4卡 ZeRO-3 检查点开"},
    {"scope": "full_zero0_no_gc_one_gpu", "training_mode": "full", "zero_stage": 0,
     "gradient_checkpointing": False, "gpu_count": 1,
     "label_cn": "FULL 1卡 不切分 检查点关"},
    {"scope": "full_zero0_gc_one_gpu", "training_mode": "full", "zero_stage": 0,
     "gradient_checkpointing": True, "gpu_count": 1,
     "label_cn": "FULL 1卡 不切分 检查点开"},
    {"scope": "full_zero2_no_gc_two_gpu", "training_mode": "full", "zero_stage": 2,
     "gradient_checkpointing": False, "gpu_count": 2,
     "label_cn": "FULL 2卡 ZeRO-2 检查点关"},
    {"scope": "full_zero2_gc_two_gpu", "training_mode": "full", "zero_stage": 2,
     "gradient_checkpointing": True, "gpu_count": 2,
     "label_cn": "FULL 2卡 ZeRO-2 检查点开"},
    {"scope": "full_zero3_no_gc_two_gpu", "training_mode": "full", "zero_stage": 3,
     "gradient_checkpointing": False, "gpu_count": 2,
     "label_cn": "FULL 2卡 ZeRO-3 检查点关"},
    {"scope": "full_zero2_no_gc_four_gpu", "training_mode": "full", "zero_stage": 2,
     "gradient_checkpointing": False, "gpu_count": 4,
     "label_cn": "FULL 4卡 ZeRO-2 检查点关"},
    {"scope": "full_zero2_gc_four_gpu", "training_mode": "full", "zero_stage": 2,
     "gradient_checkpointing": True, "gpu_count": 4,
     "label_cn": "FULL 4卡 ZeRO-2 检查点开"},
    {"scope": "full_zero3_no_gc_four_gpu", "training_mode": "full", "zero_stage": 3,
     "gradient_checkpointing": False, "gpu_count": 4,
     "label_cn": "FULL 4卡 ZeRO-3 检查点关"},
    {"scope": "full_zero3_gc_four_gpu", "training_mode": "full", "zero_stage": 3,
     "gradient_checkpointing": True, "gpu_count": 4,
     "label_cn": "FULL 4卡 ZeRO-3 检查点开"},
)

BACKFILL_CAMPAIGN = "h800_full_admission_backfill_20260806_v1"
NEGATIVE_EVIDENCE_CAMPAIGN = "h800_admission_negative_evidence_20260807_v1"
GPU_CAMPAIGNS = frozenset({BACKFILL_CAMPAIGN, NEGATIVE_EVIDENCE_CAMPAIGN})


def _is_business_source(source_id: str | None) -> bool:
    """True for the business datasets both GPU campaigns introduced.

    Layer-3 verification withholds all of them at once.  Leave-one-source-out
    alone overstates reliability because those sources are correlated -- four of
    the six heads that passed it in the first batch could not even be fitted once
    every business source was withheld, which means their threshold was pinned by
    one campaign rather than supported by independent evidence.
    """
    if not source_id:
        return False
    return source_id.startswith(("lora_s2_src", "lora_src"))


def _linked_profile(job: dict[str, Any], profile_dir: Path) -> str | None:
    """Attach the frozen token profile for a derived public-corpus dataset.

    351 canonical rows carry no ``dataset_profile_path``, which made them
    unusable for any head (the features need a center prediction, which needs a
    length profile).  The profiles have existed since 2026-07-16; only the link
    was missing.  Returning the path here is a link, not a regeneration -- the
    file is hash-recorded in the report so the binding stays auditable.
    """
    dataset_id = str(job.get("dataset_id") or "")
    if not dataset_id:
        return None
    candidate = profile_dir / f"{dataset_id}.qwen3_nothink.jsonl"
    return str(candidate) if candidate.is_file() else None


def load_records(
    observations: Path, profile_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory = read_json(
        ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_v1.json"
    )
    model_by_id, fixed_lora = _inventory_models(inventory)
    hardware = read_json(ROOT / "config" / "hardware.json")
    padding_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    raw_max_cache: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    linked_profiles: dict[str, str] = {}
    stats = Counter()
    with observations.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                stats["unparsable"] += 1
                continue
            outcome = row.get("outcome") or {}
            if not outcome.get("calibration_base_eligible"):
                stats["not_calibration_base_eligible"] += 1
                continue
            job = (row.get("configuration") or {}).get("job") or {}
            if not job.get("dataset_profile_path"):
                linked = _linked_profile(job, profile_dir)
                if linked is None:
                    stats["no_profile_available"] += 1
                    continue
                job["dataset_profile_path"] = linked
                # The stored sha256 belongs to a path that was never recorded;
                # drop it so the builder does not compare against a stale digest.
                job.pop("dataset_profile_sha256", None)
                linked_profiles[str(job.get("dataset_id"))] = linked
                stats["profile_linked"] += 1
            campaign = str(job.get("campaign_id") or "")
            try:
                record = _build_validation_record(
                    row,
                    model_by_id=model_by_id,
                    fixed_lora=fixed_lora,
                    hardware=hardware,
                    origin=(
                        "gpu_campaign_business_source"
                        if campaign in GPU_CAMPAIGNS
                        else "prior_evidence"
                    ),
                    padding_cache=padding_cache,
                    raw_max_cache=raw_max_cache,
                )
            except Exception as error:  # noqa: BLE001 - recorded, not swallowed
                stats[f"build_failed:{type(error).__name__}"] += 1
                continue
            records.append(record)
            stats["usable"] += 1
    provenance = {
        "observations": {
            "path": str(observations),
            "sha256": sha256_file(observations),
        },
        "linked_public_corpus_profiles": {
            dataset_id: {"path": path, "sha256": sha256_file(Path(path))}
            for dataset_id, path in sorted(linked_profiles.items())
        },
        "intake_counts": dict(stats),
    }
    return records, provenance


def _is_negative(record: Mapping[str, Any]) -> bool:
    if str(record.get("outcome") or "").lower() == "oom":
        return True
    memory = record.get("memory") or {}
    peak = memory.get("observed_reserved_bytes")
    limit = memory.get("safe_limit_bytes") or SAFE_LIMIT_BYTES
    try:
        return bool(peak and float(peak) > float(limit))
    except (TypeError, ValueError):
        return False


def _score(
    records: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    head: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a fitted head to records and tally admissions by true outcome."""
    tally = Counter()
    rows = []
    threshold = float(head["threshold"])
    for record in records:
        try:
            allocated = _predict_center(record, bundle["allocated_model"])
            reserved = _predict_center(record, bundle["reserved_model"])
        except Exception:  # noqa: BLE001
            tally["prediction_unavailable"] += 1
            continue
        limit = float(
            (record.get("memory") or {}).get("safe_limit_bytes") or SAFE_LIMIT_BYTES
        )
        features = _admission_features(
            allocated_center=allocated, reserved_center=reserved, safe_limit=limit
        )
        risk = _predict_risk(features, head["model"])
        admitted = bool(
            risk < threshold and allocated <= limit and reserved <= limit
        )
        negative = _is_negative(record)
        key = ("negative" if negative else "safe") + (
            "_admitted" if admitted else "_refused"
        )
        tally[key] += 1
        rows.append(
            {
                "source_id": _source_id(record),
                "model_id": (record.get("scenario") or {}).get("model_id"),
                "mbs": (record.get("scenario") or {}).get("physical_mbs"),
                "cutoff_len": (record.get("scenario") or {}).get("cutoff_len"),
                "outcome": record.get("outcome"),
                "negative": negative,
                "unsafe_risk": risk,
                "admitted": admitted,
            }
        )
    safe_total = tally["safe_admitted"] + tally["safe_refused"]
    return {
        "counts": dict(tally),
        "safe_admission_recall": (
            tally["safe_admitted"] / safe_total if safe_total else None
        ),
        "false_admissions": tally["negative_admitted"],
        "rows": rows,
    }


def _withhold_business_sources(
    records: list[dict[str, Any]],
    predicate,
    scope: str,
) -> dict[str, Any]:
    """Layer 3: refit with every business source withheld, then score on them.

    This is the check the first batch's heads mostly failed.  Returns why it could
    not run when the withheld fit is impossible -- that itself is the finding: the
    head exists only because of one campaign's data.
    """
    scoped = [row for row in records if predicate(row)]
    held = {
        _source_id(row)
        for row in scoped
        if _is_business_source(_source_id(row))
    }
    if not held:
        return {
            "runnable": False,
            "reason": "mechanism has no business source to withhold",
            "withheld_sources": 0,
        }
    training = [
        row for row in records if not _is_business_source(_source_id(row))
    ]
    evaluation = [row for row in scoped if _source_id(row) in held]
    original = V4._scope_predicate
    try:
        V4._scope_predicate = (  # type: ignore[assignment]
            lambda name, _p=predicate, _o=original: (
                _p if name == scope else _o(name)
            )
        )
        bundle = _fit_bundle(
            training,
            names=VARIANT_FEATURES[VARIANT_M1],
            coverage=COVERAGE,
            diagnostic_max_fallback=True,
        )
        head = _fit_admission_head(
            training, bundle, scope=scope, classifier_oof_threshold=True
        )
    except Exception as error:  # noqa: BLE001 - the failure IS the result
        return {
            "runnable": False,
            "reason": f"{type(error).__name__}: {error}",
            "withheld_sources": len(held),
            "interpretation": (
                "without the business sources this head cannot be fitted at all, so "
                "its threshold rests on a single campaign"
            ),
        }
    finally:
        V4._scope_predicate = original  # type: ignore[assignment]

    scored = _score(evaluation, bundle, head)
    return {
        "runnable": True,
        "withheld_sources": len(held),
        "evaluated_rows": len(evaluation),
        "counts": scored["counts"],
        "safe_admission_recall": scored["safe_admission_recall"],
        "false_admissions": scored["false_admissions"],
        "passed": scored["false_admissions"] == 0,
    }


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    bundle = _fit_bundle(
        records,
        names=VARIANT_FEATURES[VARIANT_M1],
        coverage=COVERAGE,
        diagnostic_max_fallback=True,
    )
    original = V4._scope_predicate
    results = []
    for spec in CANDIDATES:
        predicate = _mechanism_predicate(
            training_mode=spec["training_mode"],
            zero_stage=spec["zero_stage"],
            gradient_checkpointing=spec["gradient_checkpointing"],
            gpu_count=spec["gpu_count"],
        )
        scoped = [row for row in records if predicate(row)]
        sources = {_source_id(row) for row in scoped}
        negative_sources = {_source_id(row) for row in scoped if _is_negative(row)}
        entry: dict[str, Any] = {
            "scope": spec["scope"],
            "label_cn": spec["label_cn"],
            "training_mode": spec["training_mode"],
            "zero_stage": spec["zero_stage"],
            "gradient_checkpointing": spec["gradient_checkpointing"],
            "gpu_count": spec["gpu_count"],
            "packing": False,
            "observations": len(scoped),
            "independent_sources": len(sources),
            "negative_boundary_sources": len(negative_sources),
            "safe_observations": sum(1 for row in scoped if not _is_negative(row)),
            "negative_observations": sum(1 for row in scoped if _is_negative(row)),
            "backfill_observations": sum(
                1
                for row in scoped
                if str((row.get("recalibration") or {}).get("origin") or "")
                == "full_admission_backfill"
            ),
        }
        try:
            V4._scope_predicate = (  # type: ignore[assignment]
                lambda scope, _predicate=predicate, _original=original: (
                    _predicate if scope == spec["scope"] else _original(scope)
                )
            )
            head = _fit_admission_head(
                records, bundle, scope=spec["scope"], classifier_oof_threshold=True
            )
        except Exception as error:  # noqa: BLE001
            entry.update(
                {
                    "fitted": False,
                    "rejection_reason": f"{type(error).__name__}: {error}",
                    "meets_planning_thresholds": False,
                    "recommended_for_shadow_use": False,
                }
            )
            results.append(entry)
            continue
        finally:
            V4._scope_predicate = original  # type: ignore[assignment]

        calibration = head["calibration_oof"]
        safe_rows = int(calibration["safe_rows"])
        recall = (
            int(calibration["admitted_safe_rows"]) / safe_rows if safe_rows else None
        )
        meets = (
            len(sources) >= MIN_INDEPENDENT_SOURCES
            and len(negative_sources) >= MIN_NEGATIVE_BOUNDARY_SOURCES
        )
        zero_false = int(calibration["admitted_unsafe_rows"]) == 0
        withheld = _withhold_business_sources(records, predicate, spec["scope"])
        entry.update(
            {
                "fitted": True,
                "rejection_reason": None,
                "threshold": head["threshold"],
                "coefficients": head["model"]["coefficients"],
                "intercept": head["model"]["intercept"],
                "nested_leave_one_source_out": {
                    "rows": calibration["rows"],
                    "independent_sources": calibration["independent_sources"],
                    "safe_rows": safe_rows,
                    "unsafe_rows": calibration["unsafe_rows"],
                    "safe_admission_recall": recall,
                    "unsafe_admitted": calibration["admitted_unsafe_rows"],
                    "risk_min": calibration["risk_min"],
                    "risk_max": calibration["risk_max"],
                },
                "business_sources_withheld_refit": withheld,
                "zero_false_admission": zero_false,
                "meets_planning_thresholds": meets,
                # Three conditions, all required.  Passing the hard safety gate
                # alone is not enough: with two or three sources the threshold is
                # pinned by a single observation, and a head that collapses once
                # the business sources are withheld was never independently
                # supported -- that is the fragility this campaign set out to fix.
                "recommended_for_shadow_use": bool(
                    zero_false and meets and withheld.get("passed") is True
                ),
                "head": head,
            }
        )
        results.append(entry)
    return {"bundle": bundle, "mechanisms": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    records, provenance = load_records(args.observations, args.profile_dir)
    print(f"可拟合记录 {len(records)}  intake={provenance['intake_counts']}")
    outcome = evaluate(records)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "shadow_candidate_evaluated",
        "publishable": False,
        "production_model_mutated": False,
        "gpu_experiments_launched": False,
        "analysis_only": True,
        "planning_thresholds": {
            "minimum_independent_sources": MIN_INDEPENDENT_SOURCES,
            "minimum_negative_boundary_sources": MIN_NEGATIVE_BOUNDARY_SOURCES,
            "hard_gate": "zero admitted unsafe rows in leave-one-source-out",
        },
        "provenance": provenance,
        "fit_records": len(records),
        "mechanisms": [
            {k: v for k, v in row.items() if k != "head"}
            for row in outcome["mechanisms"]
        ],
        "heads": {
            row["scope"]: row["head"]
            for row in outcome["mechanisms"]
            if row.get("recommended_for_shadow_use")
        },
    }
    report["report_sha256"] = sha256_json(
        {k: v for k, v in report.items() if k != "report_sha256"}
    )

    print(
        f"\n{'机制':26s}{'观测':>5s}{'源':>4s}{'负源':>5s}{'折外放行':>9s}{'误放':>5s}"
        f"{'留出业务源':>11s}  判定"
    )
    for row in report["mechanisms"]:
        if not row["fitted"]:
            verdict = "拟合失败"
            recall = "-"
            false_admit = "-"
            layer3 = "-"
        else:
            nested = row["nested_leave_one_source_out"]
            recall = (
                f"{nested['safe_admission_recall']:.1%}"
                if nested["safe_admission_recall"] is not None
                else "-"
            )
            false_admit = str(nested["unsafe_admitted"])
            w = row["business_sources_withheld_refit"]
            if not w.get("runnable"):
                layer3 = "不可跑"
            elif w.get("passed"):
                layer3 = "通过"
            else:
                layer3 = f"误放{w['false_admissions']}"
            if row["recommended_for_shadow_use"]:
                verdict = "可影子使用"
            elif not row["zero_false_admission"]:
                verdict = "折外有误放"
            elif not row["meets_planning_thresholds"]:
                verdict = "源不足"
            else:
                verdict = "留出业务源后不成立"
        print(
            f"{row['label_cn']:26s}{row['observations']:>5d}"
            f"{row['independent_sources']:>4d}{row['negative_boundary_sources']:>5d}"
            f"{recall:>9s}{false_admit:>5s}{layer3:>11s}  {verdict}"
        )
    ready = [r for r in report["mechanisms"] if r["recommended_for_shadow_use"]]
    print(f"\n可影子使用 {len(ready)} / 候选 {len(report['mechanisms'])}")

    if not args.print_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / "new_mechanism_admission_heads_v1.json"
        write_json(path, report)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
