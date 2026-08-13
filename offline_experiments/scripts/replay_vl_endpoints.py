#!/usr/bin/env python
"""Replay the existing VL endpoints against the frozen memory model, on CPU.

The plan's Phase A-VL asks for a CPU replay of the eighteen Qwen3-VL endpoints so
the failure pattern is recorded as a reproducible artefact rather than as prose.
This module reads the frozen replay evidence and restates it as an explicit
diagnostic report.

Two boundaries are load-bearing:

* This is **diagnosis, not acceptance**.  Those eighteen endpoints have been
  inspected repeatedly, so by the project's own rule they are spent as an
  acceptance set.  The report says so in a field, not only in a comment.
* The endpoints ran on a text-only dataset with a frozen vision tower, so they
  measure the vision tower's *resident* cost and nothing about real image input.
  Any number here is a floor on the real VL gap, never an estimate of it.

It fits nothing, publishes nothing and touches no frozen artefact.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, read_json, sha256_file, sha256_json, write_json

SCHEMA = "sft_vl_endpoint_replay/v1"
IMPLEMENTATION_VERSION = "sft_vl_endpoint_replay_impl/2026-08-01.v1"

DEFAULT_REPLAY = ARTIFACT_DIR / "memory_model_generations_replay_2026-07-29.json"
DEFAULT_THROUGHPUT_REPLAY = (
    ARTIFACT_DIR / "throughput_model_generations_generalization_2026-07-28.json"
)
DEFAULT_OUTPUT = ARTIFACT_DIR / "vl_endpoint_replay_v1.json"

VL_SLICES = ("qwen3_vl_8b/lora", "qwen3_vl_8b/full")
MEMORY_HEADS = ("legacy_11d_log_reserved", "native_augmented", "physical_shares")
SERVING_HEAD = "physical_shares"


def _stat(value: Any, key: str = "mean") -> float | None:
    """Pull one number out of a nested {count, mean, median, p90, max} block."""

    if isinstance(value, Mapping):
        item = value.get(key)
        return float(item) if isinstance(item, (int, float)) else None
    return float(value) if isinstance(value, (int, float)) else None


def _memory_slice(entry: Mapping[str, Any]) -> dict[str, Any]:
    center = entry.get("center_accuracy") or {}
    safety = entry.get("safety") or {}
    signed = center.get("signed_percentage_error_percent")
    return {
        "success_rows": safety.get("success_rows"),
        "oom_rows": safety.get("oom_rows"),
        "center_mape_percent": _stat(
            center.get("absolute_percentage_error_percent")
        ),
        "center_signed_mean_percent": _stat(signed),
        # The worst case matters more than the mean for a safety head: if even the
        # least-negative endpoint under-predicts, the bias is systematic rather
        # than an average of over- and under-predictions.
        "center_signed_max_percent": _stat(signed, "max"),
        "mean_absolute_error_gib": _stat(center.get("absolute_error_gib")),
        "success_upper_coverage": safety.get("success_upper_coverage"),
        "success_upper_coverage_percent": safety.get(
            "success_upper_coverage_percent"
        ),
        "false_safe_oom": safety.get("false_safe_oom"),
        "unique_false_safe_oom_configurations": safety.get(
            "unique_false_safe_oom_configurations"
        ),
        "memory_admission_safety_failures": safety.get(
            "memory_admission_safety_failures"
        ),
        "admitted_observed_over_safe_line_success_rows": safety.get(
            "admitted_observed_over_safe_line_success_rows"
        ),
        "false_reject_safe_success": safety.get("false_reject_safe_success"),
        "safe_success_admission_recall_percent": safety.get(
            "safe_success_admission_recall_percent"
        ),
        "coverage_is_formal_p95": safety.get("coverage_is_formal_p95"),
    }


def _sum_field(slices: Mapping[str, Mapping[str, Any]], field: str) -> int | None:
    total = 0
    seen = False
    for value in slices.values():
        item = value.get(field)
        if isinstance(item, (int, float)):
            total += int(item)
            seen = True
    return total if seen else None


def build_report(
    *,
    replay_path: Path = DEFAULT_REPLAY,
    throughput_path: Path = DEFAULT_THROUGHPUT_REPLAY,
) -> dict[str, Any]:
    replay = read_json(replay_path)
    models = (
        (replay.get("new_generalization_evidence") or {}).get("models")
    ) or {}

    by_head: dict[str, Any] = {}
    for head in MEMORY_HEADS:
        table = ((models.get(head) or {}).get("by_model_and_train_type")) or {}
        slices = {
            name: _memory_slice(table[name])
            for name in VL_SLICES
            if name in table
        }
        if not slices:
            continue
        by_head[head] = {
            "slices": slices,
            "totals": {
                "success_rows": _sum_field(slices, "success_rows"),
                "oom_rows": _sum_field(slices, "oom_rows"),
                "false_safe_oom": _sum_field(slices, "false_safe_oom"),
                "memory_admission_safety_failures": _sum_field(
                    slices, "memory_admission_safety_failures"
                ),
            },
        }

    serving = by_head.get(SERVING_HEAD, {})
    totals = serving.get("totals") or {}

    # Direction matters more than magnitude here: every VL signed error being
    # negative means the model under-predicts, i.e. it fails toward admitting a
    # configuration that will not fit.
    signed_means = [
        value["center_signed_mean_percent"]
        for value in (serving.get("slices") or {}).values()
        if isinstance(value.get("center_signed_mean_percent"), (int, float))
    ]
    signed_worst = [
        value["center_signed_max_percent"]
        for value in (serving.get("slices") or {}).values()
        if isinstance(value.get("center_signed_max_percent"), (int, float))
    ]
    # Even the least-negative endpoint under-predicts, so this is a one-sided
    # bias rather than scatter that happens to average negative.
    all_under_predicted = bool(signed_worst) and all(
        item < 0 for item in signed_worst
    )

    throughput: dict[str, Any] = {"available": False}
    if throughput_path.is_file():
        payload = read_json(throughput_path)
        text = sha256_json(payload)
        throughput = {
            "available": True,
            "path": str(throughput_path),
            "sha256": sha256_file(throughput_path),
            "payload_digest": text,
            "note": (
                "Ranking transferred on VL while absolute scale did not; the "
                "ranking head is therefore reusable and the memory head is not."
            ),
        }

    report = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "diagnostic_replay_only",
        "evidence_role": {
            "is_fresh_acceptance_set": False,
            "reason": (
                "these endpoints have been inspected across several reports, so "
                "they are spent as an acceptance set and may only diagnose"
            ),
            "may_refit_coefficients": False,
            "may_publish": False,
        },
        "endpoint_scope": {
            "model": "qwen3_vl_8b",
            "dataset_id": "multiturn_4096",
            "input_modality": "text_only",
            "vision_tower_state": "frozen_in_both_lora_and_full",
            "gpu_count": 2,
            "cutoff_len": 4096,
            "packing": False,
            "consequence": (
                "measures resident vision cost only; with real images the gap "
                "can only grow, so every number here is a floor"
            ),
        },
        "sources": {
            "memory_replay_path": str(replay_path),
            "memory_replay_sha256": sha256_file(replay_path),
        },
        "memory_heads": by_head,
        "serving_head": SERVING_HEAD,
        "serving_head_summary": {
            **totals,
            "all_slices_under_predict_even_at_best_case": all_under_predicted,
            "signed_mean_percentage_errors": signed_means,
            "signed_best_case_percentage_errors": signed_worst,
        },
        "throughput_context": throughput,
        "findings": [],
        "guarantees": {
            "fits_or_publishes_coefficients": False,
            "mutates_frozen_artifacts": False,
            "creates_gpu_queue": False,
            "claims_real_image_coverage": False,
        },
    }

    findings: list[str] = []
    if all_under_predicted:
        findings.append("vl_memory_is_systematically_under_predicted")
    if (totals.get("false_safe_oom") or 0) > 0:
        findings.append("frozen_memory_head_admitted_configurations_that_oomed")
    if (totals.get("memory_admission_safety_failures") or 0) > 0:
        findings.append("admitted_runs_exceeded_the_safe_line")
    findings.append("text_only_endpoints_cannot_validate_the_visual_path")
    report["findings"] = findings
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--throughput", type=Path, default=DEFAULT_THROUGHPUT_REPLAY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report(replay_path=args.replay, throughput_path=args.throughput)
    write_json(args.output, report)

    scope = report["endpoint_scope"]
    print(f"scope: {scope['model']} / {scope['dataset_id']} / "
          f"{scope['input_modality']} / vision={scope['vision_tower_state']}")
    print(f"is_fresh_acceptance_set: "
          f"{report['evidence_role']['is_fresh_acceptance_set']}")
    print()
    for head, payload in report["memory_heads"].items():
        marker = " (serving)" if head == report["serving_head"] else ""
        print(f"{head}{marker}:")
        for name, value in payload["slices"].items():
            print(
                f"  {name}: success={value['success_rows']} oom={value['oom_rows']}"
                f" mape={value['center_mape_percent']:.2f}%"
                f" coverage={value['success_upper_coverage_percent']}%"
                f" false_safe={value['false_safe_oom']}"
                f" safety_failures={value['memory_admission_safety_failures']}"
            )
        print(f"  totals: {payload['totals']}")
    print()
    summary = report["serving_head_summary"]
    print(f"serving head totals: success={summary.get('success_rows')} "
          f"oom={summary.get('oom_rows')} "
          f"false_safe={summary.get('false_safe_oom')} "
          f"safety_failures={summary.get('memory_admission_safety_failures')}")
    print(
        "all slices under-predict even at best case: "
        f"{summary['all_slices_under_predict_even_at_best_case']}"
    )
    print(f"  signed means: {summary['signed_mean_percentage_errors']}")
    print(f"  signed best case: {summary['signed_best_case_percentage_errors']}")
    for finding in report["findings"]:
        print(f"finding: {finding}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
