#!/usr/bin/env python
"""Report exactly what must be restored to regain the calibrated runtime cohort.

The frozen H800 coefficients are bound to a runtime mechanism fingerprint.  When
the execution environment drifts, results from the new environment belong to a
different cohort and cannot be pooled with the historical evidence.  The chosen
course of action is to *restore* the original environment rather than open a new
cohort, so this module states precisely what "restored" means, item by item.

It compares three things per field:

* the value recorded in a pre-drift provenance snapshot (the restore target),
* the value in the current provenance,
* whether the difference actually changes execution semantics.

That last column is the point.  Not every difference matters equally: a container
id changes on every pod restart and carries no semantics on its own, whereas a
missing DeepSpeed patch has been *measured* in this project to flip identical
configurations between success and OOM.  Sorting by that distinction keeps the
restore list short and honest instead of demanding bit-identity for its own sake.

Read-only: it restores nothing, fits nothing and launches nothing.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, read_json, sha256_file, sha256_json, write_json

SCHEMA = "sft_runtime_cohort_restore_checklist/v1"
IMPLEMENTATION_VERSION = "sft_runtime_cohort_restore_impl/2026-08-02.v1"

DEFAULT_BASELINE = ARTIFACT_DIR / "provenance.pre-refresh-20260801.json"
DEFAULT_CURRENT = ARTIFACT_DIR / "provenance.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "runtime_cohort_restore_checklist_v1.json"

DEEPSPEED_PATCH = Path(
    "/fine-tuning-launcher/hack/patches/apply_deepspeed_zero3_mixed_dtype_fix.py"
)
INSTALLED_DEEPSPEED_SOURCE = Path(
    "/fine-tuning-launcher/.venv/lib/python3.11/site-packages/"
    "deepspeed/runtime/zero/partition_parameters.py"
)

SEVERITY_BLOCKING = "blocking_changes_execution_semantics"
SEVERITY_REVIEW = "needs_review_may_change_results"
SEVERITY_BENIGN = "benign_no_execution_semantics"

# Why each field matters, so the checklist explains itself rather than listing
# hashes.  ``severity`` decides whether restoration is mandatory.
FIELD_POLICY: dict[str, dict[str, Any]] = {
    "launcher_patch_sha256": {
        "severity": SEVERITY_BLOCKING,
        "why": (
            "The DeepSpeed ZeRO-3 mixed-dtype all-gather patch. This project "
            "measured 10 conflicting configurations that re-ran 20/20 OOM once "
            "the runtime was corrected, so its absence can flip a configuration "
            "between success and OOM. ZeRO-3 evidence collected without it is "
            "not comparable to the calibrated cohort."
        ),
        "restore_action": (
            "Restore /fine-tuning-launcher/hack/patches/"
            "apply_deepspeed_zero3_mixed_dtype_fix.py and the launcher dev tree, "
            "then re-verify the patch sha matches the baseline."
        ),
    },
    "framework_source_sha256": {
        "severity": SEVERITY_BLOCKING,
        "why": (
            "Hash of deepspeed/runtime/zero/partition_parameters.py, i.e. the "
            "exact file the mixed-dtype patch modifies. A different hash at the "
            "same package version means a different build, not an upgrade."
        ),
        "restore_action": (
            "Restore the patched DeepSpeed build so the partition_parameters "
            "hash matches the baseline."
        ),
    },
    "launcher_commits": {
        "severity": SEVERITY_REVIEW,
        "why": (
            "Launcher git commits pin the wrapper and LLaMA-Factory checkout. "
            "Nulls mean the git metadata is unavailable in this image, which "
            "removes provenance rather than proving a code change."
        ),
        "restore_action": (
            "Restore the launcher checkouts so commits are readable again, or "
            "record explicitly that provenance is degraded for this cohort."
        ),
    },
    "packages": {
        "severity": SEVERITY_REVIEW,
        "why": (
            "Framework versions define kernel and optimizer behaviour. Any "
            "difference here needs an explicit judgement rather than a default."
        ),
        "restore_action": "Reinstall the missing or changed distributions.",
    },
    "nvidia_driver": {
        "severity": SEVERITY_REVIEW,
        "why": (
            "The driver affects kernel selection and allocator behaviour. It is "
            "not part of the calibration key, but a major-version jump alongside "
            "a GPU inventory change is evidence the host itself changed."
        ),
        "restore_action": (
            "Confirm whether the original host/driver is available; if not, this "
            "difference must be carried into the cohort decision."
        ),
    },
    "container_runtime_id": {
        "severity": SEVERITY_BENIGN,
        "why": (
            "Changes on every pod restart. On its own it proves nothing; it is "
            "only corroborating evidence when other fields also moved."
        ),
        "restore_action": "No action; informational only.",
    },
    "project_source_snapshot_sha256": {
        "severity": SEVERITY_BENIGN,
        "why": (
            "Digest of this repository's own sources, which changed because new "
            "CPU-only analysis modules were added on purpose. Unrelated to the "
            "execution environment."
        ),
        "restore_action": "No action; expected to differ.",
    },
}


def _classify(field: str) -> dict[str, Any]:
    return FIELD_POLICY.get(
        field,
        {
            "severity": SEVERITY_REVIEW,
            "why": "no policy recorded for this field; treat as review-needed",
            "restore_action": "decide explicitly before collecting evidence",
        },
    )


def _diff_identity(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for field in sorted(set(baseline) | set(current)):
        before, after = baseline.get(field), current.get(field)
        if before == after:
            continue
        policy = _classify(field)
        entry: dict[str, Any] = {
            "field": field,
            "severity": policy["severity"],
            "why_it_matters": policy["why"],
            "restore_action": policy["restore_action"],
        }
        if isinstance(before, Mapping) or isinstance(after, Mapping):
            before_map = before if isinstance(before, Mapping) else {}
            after_map = after if isinstance(after, Mapping) else {}
            entry["sub_differences"] = {
                key: {
                    "baseline": before_map.get(key),
                    "current": after_map.get(key),
                }
                for key in sorted(set(before_map) | set(after_map))
                if before_map.get(key) != after_map.get(key)
            }
        else:
            entry["baseline"] = before
            entry["current"] = after
        entries.append(entry)
    return entries


def build_checklist(
    *,
    baseline_path: Path = DEFAULT_BASELINE,
    current_path: Path = DEFAULT_CURRENT,
) -> dict[str, Any]:
    baseline = read_json(baseline_path)
    current = read_json(current_path)
    base_identity = baseline.get("runtime_identity") or {}
    cur_identity = current.get("runtime_identity") or {}

    differences = _diff_identity(base_identity, cur_identity)
    blocking = [item for item in differences if item["severity"] == SEVERITY_BLOCKING]
    review = [item for item in differences if item["severity"] == SEVERITY_REVIEW]
    benign = [item for item in differences if item["severity"] == SEVERITY_BENIGN]

    # Report what still matches, because a short restore list is only credible if
    # the unchanged surface is stated too.
    unchanged = sorted(
        field
        for field in set(base_identity) & set(cur_identity)
        if base_identity[field] == cur_identity[field]
    )

    restore_targets = [
        {
            "item": "deepspeed_mixed_dtype_patch_file",
            "path": str(DEEPSPEED_PATCH),
            "present_now": DEEPSPEED_PATCH.is_file(),
            "current_sha256": (
                sha256_file(DEEPSPEED_PATCH) if DEEPSPEED_PATCH.is_file() else None
            ),
            "expected_sha256": base_identity.get("launcher_patch_sha256"),
            "blocking": True,
        },
        {
            "item": "patched_deepspeed_build",
            "file": str(INSTALLED_DEEPSPEED_SOURCE),
            "expected_sha256": (base_identity.get("framework_source_sha256") or {}).get(
                "deepspeed_zero_partition_parameters"
            ),
            "current_sha256": (
                sha256_file(INSTALLED_DEEPSPEED_SOURCE)
                if INSTALLED_DEEPSPEED_SOURCE.is_file()
                else None
            ),
            "recorded_current_sha256": (
                cur_identity.get("framework_source_sha256") or {}
            ).get("deepspeed_zero_partition_parameters"),
            "blocking": True,
        },
        {
            "item": "launcher_dev_tree",
            "path": "/fine-tuning-launcher/dev/finetuning-launcher",
            "present_now": Path(
                "/fine-tuning-launcher/dev/finetuning-launcher"
            ).is_dir(),
            "blocking": True,
        },
        {
            "item": "gpu_pool_identity",
            "expected": "8 x NVIDIA H800 140GB HBM3 matching the recorded UUIDs",
            "note": (
                "Historical run-bound evidence names 8 GPU UUIDs; the current "
                "pool exposes 4 with no overlap. Restoring the cohort includes "
                "restoring the pool, or the mechanism key still differs."
            ),
            "blocking": True,
        },
    ]

    checklist = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "restore_required",
        "decision": "restore_original_cohort_rather_than_open_a_new_one",
        "baseline": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
            "role": "pre-drift snapshot used as the restore target",
        },
        "current": {
            "path": str(current_path),
            "sha256": sha256_file(current_path),
            "role": (
                "last captured provenance; intentionally not refreshed until "
                "runtime and hardware restoration decisions are complete"
            ),
            "live_restore_targets_override_stale_recorded_values": True,
        },
        "difference_counts": {
            SEVERITY_BLOCKING: len(blocking),
            SEVERITY_REVIEW: len(review),
            SEVERITY_BENIGN: len(benign),
        },
        "differences": differences,
        "unchanged_identity_fields": unchanged,
        "restore_targets": restore_targets,
        "acceptance_of_restore": {
            "checks": [
                "launcher_patch_sha256 equals the baseline value",
                "framework_source_sha256.deepspeed_zero_partition_parameters "
                "equals the baseline value",
                "live_runtime_patch().all_passed is true",
                "the GPU pool matches the recorded H800 UUID set",
                "capture_provenance is re-run only after all of the above hold",
            ],
            "note": (
                "capture_provenance must be the last step: capturing before the "
                "environment is restored would freeze the drifted state as the "
                "new baseline."
            ),
        },
        "guarantees": {
            "restores_anything": False,
            "modifies_runtime": False,
            "fits_or_publishes_coefficients": False,
            "creates_gpu_queue": False,
        },
        "until_restored": {
            "phase_b_canary_may_run": False,
            "reason": (
                "Phase B verdicts are hardware-independent but mechanism-bound. "
                "Changes to the live patch/build/tree can alter execution "
                "semantics, so their runtime fingerprint must be accepted "
                "together before consistency evidence is collected."
            ),
            "calibration_or_acceptance_may_run": False,
        },
    }
    checklist["checklist_sha256"] = sha256_json(checklist)
    return checklist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    checklist = build_checklist(
        baseline_path=args.baseline, current_path=args.current
    )
    write_json(args.output, checklist)

    counts = checklist["difference_counts"]
    print(f"decision: {checklist['decision']}")
    print(
        f"differences: blocking={counts[SEVERITY_BLOCKING]} "
        f"review={counts[SEVERITY_REVIEW]} benign={counts[SEVERITY_BENIGN]}"
    )
    print(f"unchanged identity fields: {len(checklist['unchanged_identity_fields'])}")
    print()
    for severity in (SEVERITY_BLOCKING, SEVERITY_REVIEW, SEVERITY_BENIGN):
        items = [
            item for item in checklist["differences"] if item["severity"] == severity
        ]
        if not items:
            continue
        print(f"[{severity}]")
        for item in items:
            print(f"  {item['field']}")
            if "sub_differences" in item:
                for key, value in item["sub_differences"].items():
                    print(
                        f"    {key}: {str(value['baseline'])[:34]} -> "
                        f"{str(value['current'])[:34]}"
                    )
            else:
                print(
                    f"    {str(item['baseline'])[:40]} -> {str(item['current'])[:40]}"
                )
        print()
    print("restore targets:")
    for target in checklist["restore_targets"]:
        mark = "BLOCKING" if target["blocking"] else "optional"
        present = target.get("present_now")
        state = "" if present is None else f" present_now={present}"
        print(f"  [{mark}] {target['item']}{state}")
    print()
    print(f"phase_b_canary_may_run: "
          f"{checklist['until_restored']['phase_b_canary_may_run']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
