#!/usr/bin/env python
"""Materialize the Phase-B canary queue and the exact approval diff it needs.

The approval gate binds a queue by SHA-256, so an approval cannot be written
before the queue file exists, and the queue must not be executable before the
approval names it.  This module breaks that circularity the same way the rest of
the repo does: it produces the queue rows and the *proposed* approval document as
review artefacts, computes every hash the gate will check, and refuses to install
either one.

What it will not do:

* write ``config/APPROVED_TO_RUN.json`` (that is the human approval act),
* write the runnable queue JSONL unless ``--write-queue`` is passed *and* every
  precondition already holds,
* launch training, or mark any run as calibration evidence.

The output is designed to be reviewable as a diff: it reports the current
approval, the proposed approval, and a field-by-field delta, so the reviewer sees
exactly which authorisations are being requested rather than a rewritten file.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from approval_gate import canonical_job_sha256
from common import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    DATA_DIR,
    MATRIX_DIR,
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from prepare_phase_b_consistency_canary import (
    CLASS_HARDWARE_BOUND,
    build_design,
)

SCHEMA = "sft_phase_b_canary_queue_manifest/v1"
IMPLEMENTATION_VERSION = "sft_phase_b_canary_materializer_impl/2026-08-01.v1"

DEFAULT_OUTPUT = ARTIFACT_DIR / "phase_b_canary_queue_manifest_v1.json"
DEFAULT_QUEUE = MATRIX_DIR / "phase_b_canary_jobs.jsonl"

CAMPAIGN_ID = "phase_b_consistency_canary_20260801"

# Deliberately tiny: this phase proves agreement between prediction and
# execution, so a cell needs only enough steps to emit stable counters.
WARMUP_STEPS = 0
MEASURE_STEPS = 2


def _model_row(inventory: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    for row in inventory.get("models") or []:
        if row.get("id") == model_id:
            return row
    raise KeyError(f"model {model_id} is not in the inventory")


def build_jobs(
    cells: Sequence[Mapping[str, Any]],
    *,
    inventory: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Turn design cells into concrete job rows, reporting missing inputs."""

    jobs: list[dict[str, Any]] = []
    blockers: list[str] = []
    dataset_info = (
        read_json(DATA_DIR / "dataset_info.json")
        if (DATA_DIR / "dataset_info.json").is_file()
        else {}
    )
    for cell in cells:
        model_id = str(cell["model_id"])
        try:
            model = _model_row(inventory, model_id)
        except KeyError:
            blockers.append(f"model_not_in_inventory:{model_id}")
            continue
        dataset_id = str(cell["dataset_id"])
        derived = DATA_DIR / "derived" / f"{dataset_id}.jsonl"
        if not derived.is_file():
            blockers.append(f"dataset_missing:{dataset_id}")
            continue
        if dataset_id not in dataset_info:
            blockers.append(f"dataset_not_registered:{dataset_id}")
            continue
        job = {
            "schema": "sft_phase_b_consistency_canary_job/v1",
            "job_id": str(cell["cell_id"]),
            "campaign_id": CAMPAIGN_ID,
            "phase_id": CAMPAIGN_ID,
            "kind": "consistency_canary",
            "fidelity": f"canary_{WARMUP_STEPS}plus{MEASURE_STEPS}",
            "purpose": cell.get("purpose"),
            "model_id": model_id,
            "model_family": model.get("family"),
            "model_path": model.get("path"),
            "tokenizer_path": model.get("tokenizer_path"),
            "template": model.get("template"),
            "model_parameters": model.get("actual_parameters"),
            "train_type": str(cell["train_type"]),
            "dataset_id": dataset_id,
            "cutoff_len": int(cell["cutoff_len"]),
            "target_gbs": int(cell["target_gbs"]),
            "gpu_count": int(cell["gpu_count"]),
            "mbs": int(cell["mbs"]),
            "zero": str(cell["zero"]),
            "gc": bool(cell["gc"]),
            "packing": bool(cell["packing"]),
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "repeat": 0,
            "parallel_class": (
                "gpu_partitionable" if int(cell["gpu_count"]) < 4 else "exclusive_node"
            ),
            "requires_external_node_idle": False,
            # The consistency verdicts survive a hardware mismatch; the numbers
            # that do not are marked here so no downstream step can promote them.
            "calibration_evidence_eligible": False,
            "evidence_role": "consistency_only",
            "hardware_bound_outputs_are_diagnostic": True,
        }
        if not job["packing"]:
            per_step = job["gpu_count"] * job["mbs"]
            if per_step <= 0 or job["target_gbs"] % per_step != 0:
                blockers.append(f"gbs_contract_violated:{job['job_id']}")
                continue
            job["gradient_accumulation_steps"] = job["target_gbs"] // per_step
        jobs.append(job)
    return jobs, blockers


def _provenance_state(root: Path = ROOT) -> dict[str, Any]:
    """Report whether the provenance binding the gate needs is currently valid.

    ``build_provenance_binding`` refuses to emit a binding while the recorded
    source manifest disagrees with the working tree, which is the correct
    behaviour: that binding is the tamper-check root.  Rather than route around
    it, this reports the exact reason so the reviewer sees that a provenance
    refresh -- a separate, deliberate act -- is a precondition.
    """

    try:
        from approval_gate import build_provenance_binding

        binding = build_provenance_binding(root)
    except ValueError as error:
        text = str(error)
        return {
            "binding_available": False,
            "reason": "provenance_stale_or_incomplete",
            "detail_truncated": text[:400],
            "refresh_command": (
                "/fine-tuning-launcher/.venv/bin/python scripts/capture_provenance.py"
            ),
            "note": (
                "New scripts added to the tree also change the expected source "
                "path set, so the manifest must be recaptured before any new "
                "approval can bind it."
            ),
        }
    return {
        "binding_available": True,
        "sha256": binding.get("sha256"),
        "runtime_fingerprint_sha256": binding.get("runtime_fingerprint_sha256"),
        "project_source_path_count": len(binding.get("project_source_paths") or []),
    }


def build_proposed_approval(
    jobs: Sequence[Mapping[str, Any]],
    *,
    queue_path: Path,
    current: Mapping[str, Any] | None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the approval document a reviewer would install, unwritten.

    Fields the current approval carries for tamper-checking are *preserved*, not
    dropped: an approval that omits the provenance and runtime-fingerprint
    bindings would be weaker than the one it replaces, which is exactly the kind
    of silent regression the gate exists to prevent.
    """

    job_ids = [str(job["job_id"]) for job in jobs]
    payload_hashes = [canonical_job_sha256(dict(job)) for job in jobs]
    proposed = {
        "schema_version": 1,
        "approved": True,
        "phase_id": CAMPAIGN_ID,
        "approved_model_ids": sorted({str(job["model_id"]) for job in jobs}),
        "allowed_job_ids": job_ids,
        "resource_scope": {
            "gpu_ids": [0, 1, 2, 3],
            "max_gpu_count": max((int(job["gpu_count"]) for job in jobs), default=0),
            "allow_gpu_ids_outside_pool": False,
            "performance_parallelism": "disjoint_gpu_masks",
        },
        "queue_binding": {
            "schema_version": 1,
            "path": str(queue_path.relative_to(ROOT))
            if queue_path.is_relative_to(ROOT)
            else str(queue_path),
            "sha256": sha256_file(queue_path) if queue_path.is_file() else None,
            "ordered_job_ids": job_ids,
            "ordered_job_payload_sha256": payload_hashes,
            "job_payload_sha256": dict(zip(job_ids, payload_hashes, strict=True)),
        },
        "evidence_policy": {
            "calibration_evidence_eligible": False,
            "may_enter_memory_calibration": False,
            "may_enter_throughput_calibration": False,
            "may_close_out_rank_validation": False,
            "reason": (
                "pool identity does not match the frozen H800 runtime "
                "fingerprint; only hardware-independent consistency verdicts "
                "from these runs are usable"
            ),
        },
    }

    # Carry the tamper-check fields forward.  Provenance and the runtime
    # fingerprint come straight from the refreshed binding.  ``runtime_patch``
    # and ``design`` hashes cannot be synthesised here: run_job derives them from
    # an approval *design* document (``runtime/approval_design.json``), which has
    # no ``runtime_patch`` block for this campaign yet.  Naming the source beats
    # emitting a plausible-looking hash that the gate would reject at launch.
    if provenance and provenance.get("binding_available"):
        proposed["provenance_sha256"] = provenance.get("sha256")
        proposed["runtime_fingerprint_sha256"] = provenance.get(
            "runtime_fingerprint_sha256"
        )
    else:
        for field in ("provenance_sha256", "runtime_fingerprint_sha256"):
            proposed[field] = "<must_be_regenerated_after_provenance_refresh>"

    proposed["runtime_patch_sha256"] = (
        "<derive_from_approval_design.runtime_patch; absent for this campaign>"
    )
    proposed["design_sha256"] = (
        "<sha256 of the campaign approval design once it is authored>"
    )

    delta: list[dict[str, Any]] = []
    if current:
        for key in sorted(set(proposed) | set(current)):
            before = current.get(key)
            after = proposed.get(key)
            if key == "allowed_job_ids":
                before_ids = list(before or [])
                after_ids = list(after or [])
                delta.append(
                    {
                        "field": key,
                        "change": "replaced",
                        "before_count": len(before_ids),
                        "after_count": len(after_ids),
                        "removed_sample": before_ids[:3],
                        "added": after_ids,
                        "note": (
                            "existing ids are completed jobs from another phase; "
                            "they are not re-authorised by this request"
                        ),
                    }
                )
                continue
            if before != after:
                delta.append(
                    {
                        "field": key,
                        "change": "added" if key not in current else "modified",
                        "before": before
                        if not isinstance(before, (dict, list))
                        else "<structured>",
                        "after": after
                        if not isinstance(after, (dict, list))
                        else "<structured>",
                    }
                )
    return {"document": proposed, "delta_vs_current": delta}


def build_manifest(
    *,
    required_gpu_ids: Sequence[int] = (0, 1, 2, 3),
    include_vl: bool = True,
    queue_path: Path = DEFAULT_QUEUE,
) -> dict[str, Any]:
    design = build_design(
        required_gpu_ids=required_gpu_ids, include_vl=include_vl
    )
    inventory = read_json(ARTIFACT_DIR / "model_inventory.json")
    jobs, job_blockers = build_jobs(design["cells"], inventory=inventory)

    approval_path = CONFIG_DIR / "APPROVED_TO_RUN.json"
    current = read_json(approval_path) if approval_path.is_file() else None
    provenance = _provenance_state()
    approval = build_proposed_approval(
        jobs, queue_path=queue_path, current=current, provenance=provenance
    )

    blockers = sorted(set(job_blockers) | set(design["blockers"]))
    if not provenance.get("binding_available"):
        blockers.append("provenance_binding_unavailable")
    blockers = sorted(set(blockers))
    # The queue rows may be written once their own inputs resolve, but installing
    # the approval is always a human act, and the provenance refresh is a separate
    # deliberate step, so launch stays closed regardless.
    materialization_allowed = not job_blockers and len(jobs) == len(design["cells"])

    hardware_bound = [
        check["check_id"]
        for check in design["checks"]
        if check["classification"] == CLASS_HARDWARE_BOUND
    ]

    manifest = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "design_sha256": design["design_sha256"],
        "hardware_policy": design["hardware_policy"],
        "job_count": len(jobs),
        "jobs": jobs,
        "queue": {
            "path": str(queue_path.relative_to(ROOT))
            if queue_path.is_relative_to(ROOT)
            else str(queue_path),
            "exists": queue_path.is_file(),
            "binding_sha256_of_rows": sha256_json(jobs),
        },
        "current_approval": {
            "path": str(approval_path.relative_to(ROOT)),
            "sha256": sha256_file(approval_path) if approval_path.is_file() else None,
            "phase_id": (current or {}).get("phase_id"),
            "allowed_job_id_count": len((current or {}).get("allowed_job_ids") or []),
            "covers_this_campaign": False,
        },
        "provenance_state": provenance,
        "proposed_approval": approval["document"],
        "approval_delta_vs_current": approval["delta_vs_current"],
        "hardware_bound_checks_not_evidence": hardware_bound,
        "guarantees": {
            "installs_approval": False,
            "launches_training": False,
            "marks_runs_as_calibration_evidence": False,
            "mutates_frozen_artifacts": False,
            "writes_queue_without_flag": False,
            "refreshes_provenance": False,
            "weakens_existing_tamper_checks": False,
        },
        "materialization_allowed": materialization_allowed,
        "launch_allowed": False,
        "blockers": blockers,
        "required_next_step": (
            "1) refresh provenance (capture_provenance.py) so the new scripts "
            "enter the source manifest; 2) review 'proposed_approval' and "
            "'approval_delta_vs_current'; 3) install the approval manually with "
            "regenerated provenance/runtime hashes; 4) rerun with --write-queue "
            "so the bound queue SHA matches the installed document."
        ),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=(0, 1, 2, 3))
    parser.add_argument("--no-vl", action="store_true")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--write-queue",
        action="store_true",
        help="write the runnable queue JSONL; refused while blockers remain",
    )
    args = parser.parse_args()

    manifest = build_manifest(
        required_gpu_ids=args.gpu_ids,
        include_vl=not args.no_vl,
        queue_path=args.queue,
    )
    write_json(args.output, manifest)

    print(f"campaign: {manifest['campaign_id']}")
    print(f"jobs: {manifest['job_count']}")
    for job in manifest["jobs"]:
        ga = job.get("gradient_accumulation_steps")
        print(
            f"  {job['job_id']}: {job['model_id']}/{job['train_type']}"
            f"/{job['dataset_id']} gpu={job['gpu_count']} mbs={job['mbs']}"
            f" zero={job['zero']} pack={job['packing']}"
            + (f" ga={ga}" if ga else "")
        )
    print(f"materialization_allowed: {manifest['materialization_allowed']}")
    print(f"launch_allowed: {manifest['launch_allowed']}")
    print("--- approval delta vs current ---")
    for entry in manifest["approval_delta_vs_current"]:
        if entry["field"] == "allowed_job_ids":
            print(
                f"  {entry['field']}: {entry['before_count']} -> "
                f"{entry['after_count']} ({entry['change']})"
            )
            print(f"    added: {entry['added']}")
        else:
            print(
                f"  {entry['field']}: {entry['change']}"
                f"  {entry['before']} -> {entry['after']}"
            )
    for blocker in manifest["blockers"]:
        print(f"blocker: {blocker}")

    if args.write_queue:
        if not manifest["materialization_allowed"]:
            raise SystemExit(
                "refusing --write-queue: " + ", ".join(manifest["blockers"])
            )
        write_jsonl(args.queue, manifest["jobs"])
        print(f"wrote queue {args.queue}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
