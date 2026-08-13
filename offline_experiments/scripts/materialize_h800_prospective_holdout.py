#!/usr/bin/env python3
"""Gate and materialize the fresh H800 holdout without launching it.

The legacy ``materialize_jobs.py`` consumes the old memory-boundary matrix and
cannot safely infer new processor/data bindings.  This entry point consumes
the versioned fresh-holdout design instead.  By default it writes only a
manifest/report; ``--write-queue`` is accepted only when the campaign gate is
green and every scenario binds model, profile and data files.  It never calls
the scheduler or a training launcher.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from common import (
    ARTIFACT_DIR,
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)


SCHEMA = "sft_h800_prospective_queue_manifest/v1"
DESIGN_SCHEMA = "sft_h800_prospective_holdout_design/v1"
GATE_SCHEMA = "sft_h800_campaign_gate_report/v1"
PROFILE_REQUIREMENTS_SCHEMA = "sft_h800_fresh_profile_requirements/v1"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_fresh_holdout_design_v1.json"
DEFAULT_GATE = ARTIFACT_DIR / "h800_campaign_gate_report_v1.json"
DEFAULT_PROFILE_REQUIREMENTS = ARTIFACT_DIR / "h800_fresh_profile_requirements_v1.json"
DEFAULT_APPROVAL = ROOT / "config" / "APPROVED_TO_RUN.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_prospective_queue_manifest_v1.json"
DEFAULT_QUEUE = ROOT / "matrix" / "h800_fresh_holdout_jobs.jsonl"


def queue_binding_sha256(jobs: Sequence[Mapping[str, Any]]) -> str:
    """Hash only the deterministic job payload that approval authorizes."""

    return sha256_json([dict(job) for job in jobs])


def approval_binds_queue(
    approval: Mapping[str, Any] | None,
    expected_queue_binding_sha256: str,
) -> tuple[bool, str]:
    """Require an approved, exact queue binding before writing JSONL."""

    if not isinstance(approval, Mapping):
        return False, "approval_missing"
    if approval.get("approved") is not True:
        return False, "approval_not_promoted"
    observed = approval.get("queue_binding_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        return False, "approval_queue_binding_invalid"
    try:
        int(observed, 16)
    except ValueError:
        return False, "approval_queue_binding_invalid"
    if observed != expected_queue_binding_sha256:
        return False, "approval_queue_binding_mismatch"
    return True, "approval_queue_binding_matches"


def _path_binding(raw: Any, root: Path) -> dict[str, Any] | None:
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = root / path
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _model_map(inventory: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["id"]): row
        for row in inventory.get("models") or []
        if isinstance(row, Mapping) and row.get("id")
    }


def build_queue_manifest(
    *,
    design: Mapping[str, Any],
    gate_report: Mapping[str, Any],
    model_inventory: Mapping[str, Any],
    fresh_profile_requirements: Mapping[str, Any] | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Build a non-executable queue manifest and explicit blockers."""

    blockers: list[str] = []
    if design.get("schema") != DESIGN_SCHEMA:
        blockers.append("holdout_design_schema_mismatch")
    if gate_report.get("schema") != GATE_SCHEMA:
        blockers.append("campaign_gate_schema_mismatch")
    if gate_report.get("all_prerequisites_passed") is not True:
        blockers.append("campaign_gate_not_passed")
    if design.get("materialization_allowed") is not True:
        blockers.append("design_materialization_not_allowed")
    if design.get("gpu_training_started") is not False or design.get("queues_mutated") is not False:
        blockers.append("design_safety_invariant_drifted")
    requirements_by_scenario: dict[str, Mapping[str, Any]] = {}
    if fresh_profile_requirements is not None:
        if fresh_profile_requirements.get("schema") != PROFILE_REQUIREMENTS_SCHEMA:
            blockers.append("fresh_profile_requirements_schema_mismatch")
        for row in fresh_profile_requirements.get("scenarios") or []:
            if isinstance(row, Mapping) and row.get("scenario_id"):
                requirements_by_scenario[str(row["scenario_id"])] = row
    elif "prospective_holdout" in str(design.get("schema") or ""):
        # Backward-compatible direct binding path for synthetic/unit fixtures
        # and already-frozen designs.  The generated campaign design has no
        # direct bindings, so it still fails closed until the intake contract
        # is filled.
        direct_bindings_complete = all(
            bool(
                (scenario.get("profile_path") or (scenario.get("freshness") or {}).get("profile_path"))
                and bool(
                    scenario.get("data_path")
                    or (scenario.get("freshness") or {}).get("data_path")
                )
                and (scenario.get("freshness") or {}).get("runtime_dataset_registered") is True
            )
            for scenario in design.get("scenarios") or []
            if isinstance(scenario, Mapping)
        )
        if not direct_bindings_complete:
            blockers.append("fresh_profile_requirements_missing")

    models = _model_map(model_inventory)
    jobs: list[dict[str, Any]] = []
    scenarios = {str(row.get("scenario_id")): row for row in design.get("scenarios") or []}
    if not scenarios:
        blockers.append("no_scenarios")
    for scenario_id, scenario in sorted(scenarios.items()):
        model_id = str(scenario.get("model_id") or "")
        model = models.get(model_id)
        if model is None:
            blockers.append(f"missing_model_inventory:{model_id}")
        freshness = scenario.get("freshness") or {}
        requirement = requirements_by_scenario.get(scenario_id) or {}
        requirement_bindings = requirement.get("required_bindings") or {}
        profile_binding = _path_binding(
            scenario.get("profile_path")
            or freshness.get("profile_path")
            or requirement_bindings.get("profile_path"),
            root,
        )
        data_binding = _path_binding(
            scenario.get("data_path")
            or freshness.get("data_path")
            or requirement_bindings.get("data_path"),
            root,
        )
        if not profile_binding or not profile_binding["exists"]:
            blockers.append(f"missing_fresh_profile:{scenario_id}")
        if not data_binding or not data_binding["exists"]:
            blockers.append(f"missing_fresh_data:{scenario_id}")
        if (
            freshness.get("runtime_dataset_registered") is not True
            and requirement_bindings.get("runtime_dataset_registered") is not True
        ):
            blockers.append(f"runtime_dataset_registration_missing:{scenario_id}")
        for binding_name, actual in (
            ("profile_sha256", profile_binding),
            ("data_sha256", data_binding),
        ):
            expected = requirement_bindings.get(binding_name)
            if expected and actual and actual.get("sha256") != expected:
                blockers.append(f"{binding_name}_mismatch:{scenario_id}")
        if requirement and requirement.get("dataset_profile_id") != scenario.get("dataset_profile_id"):
            blockers.append(f"profile_requirement_mismatch:{scenario_id}")
        scenario["resolved_profile_binding"] = profile_binding
        scenario["resolved_data_binding"] = data_binding

    for slot in design.get("candidate_slots") or []:
        scenario_id = str(slot.get("scenario_id") or "")
        scenario = scenarios.get(scenario_id)
        if scenario is None:
            blockers.append(f"slot_references_unknown_scenario:{scenario_id}")
            continue
        if slot.get("packing") is not False or slot.get("offload") is not False:
            blockers.append(f"unsupported_mechanism_in_slot:{slot.get('candidate_slot_id')}")
            continue
        model = models.get(str(slot.get("model_id"))) or {}
        identity = {
            "campaign_id": design.get("campaign_id"),
            "scenario_id": scenario_id,
            "template_id": slot.get("template_id"),
            "dataset_profile_id": slot.get("dataset_profile_id"),
            "gpu_count": slot.get("gpu_count"),
            "zero_stage": slot.get("zero_stage"),
            "physical_mbs": slot.get("physical_mbs"),
        }
        jobs.append(
            {
                "job_id": stable_id("h800fresh", identity),
                "campaign_id": design.get("campaign_id"),
                # Keep the scenario and slot identities in the executable
                # manifest.  Result recovery must join observations to the
                # frozen prospective scenario, never infer a scenario from a
                # dataset/model name after the run.
                "scenario_id": scenario_id,
                "candidate_slot_id": slot.get("candidate_slot_id"),
                "scale_out_transition": scenario.get("scale_out_transition"),
                "phase_id": "h800_frozen_physical_v4b_fresh_holdout",
                "hardware_id": "h800",
                "gpu_type": "NVIDIA H800",
                "model_id": slot.get("model_id"),
                "model_path": model.get("path"),
                "tokenizer_path": model.get("tokenizer_path"),
                "model_family": model.get("family"),
                "model_parameters": model.get("actual_parameters"),
                "template": model.get("template"),
                "train_type": "full",
                "dataset_id": slot.get("dataset_profile_id"),
                "dataset_profile_path": scenario.get("resolved_profile_binding"),
                "data_path": scenario.get("resolved_data_binding"),
                "cutoff_len": slot.get("cutoff_len"),
                "target_gbs": slot.get("target_gbs"),
                "gpu_count": slot.get("gpu_count"),
                "zero_stage": slot.get("zero_stage"),
                "zero": f"zero{slot.get('zero_stage')}" if slot.get("zero_stage") else "none",
                "gc": slot.get("gradient_checkpointing"),
                "mbs": slot.get("physical_mbs"),
                "packing": False,
                "offload": False,
                "gradient_checkpointing": slot.get("gradient_checkpointing"),
                "parallel_class": "exclusive_pool" if slot.get("gpu_count") == 4 else "gpu_partitionable",
                "execution_state": "awaiting_exact_promoted_approval",
            }
        )

    # A queue must never be considered ready if a design slot was silently
    # dropped while building it.
    expected_slots = len(design.get("candidate_slots") or [])
    if expected_slots != len(jobs):
        blockers.append(f"slot_count_mismatch:{expected_slots}!={len(jobs)}")
    return {
        "schema": SCHEMA,
        "campaign_id": design.get("campaign_id"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "path": str(Path(DEFAULT_DESIGN).resolve()),
            "sha256": None,
        },
        "gate": {
            "path": str(DEFAULT_GATE.resolve()),
            "passed": gate_report.get("all_prerequisites_passed") is True,
        },
        "gpu_training_started": False,
        "queues_mutated": False,
        "materialization_allowed": not blockers,
        "launch_allowed": False,
        "publication_allowed": False,
        "candidate_count": len(jobs),
        "jobs": jobs,
        "queue_binding_sha256": queue_binding_sha256(jobs),
        "blockers": sorted(set(blockers)),
        "required_next_step": (
            "write queue only after a separately promoted approval and a final gate recheck"
            if not blockers
            else "resolve blockers; no queue may be written"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument(
        "--profile-requirements",
        type=Path,
        default=DEFAULT_PROFILE_REQUIREMENTS,
    )
    parser.add_argument(
        "--model-inventory",
        type=Path,
        default=ARTIFACT_DIR / "model_inventory.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument(
        "--write-queue",
        action="store_true",
        help="write JSONL only if every gate and binding passes; never launches",
    )
    args = parser.parse_args()
    design = read_json(args.design)
    gate = read_json(args.gate)
    approval = read_json(args.approval) if args.approval.is_file() else None
    profile_requirements = (
        read_json(args.profile_requirements)
        if args.profile_requirements.is_file()
        else None
    )
    inventory = read_json(args.model_inventory)
    manifest = build_queue_manifest(
        design=design,
        gate_report=gate,
        model_inventory=inventory,
        fresh_profile_requirements=profile_requirements,
        root=ROOT,
    )
    manifest["design"] = {"path": str(args.design.resolve()), "sha256": sha256_file(args.design)}
    manifest["gate"] = {"path": str(args.gate.resolve()), "sha256": sha256_file(args.gate), "passed": gate.get("all_prerequisites_passed") is True}
    manifest["fresh_profile_requirements"] = (
        {
            "path": str(args.profile_requirements.resolve()),
            "sha256": sha256_file(args.profile_requirements),
        }
        if args.profile_requirements.is_file()
        else None
    )
    approval_matches, approval_reason = approval_binds_queue(
        approval,
        manifest["queue_binding_sha256"],
    )
    manifest["approval"] = {
        "path": str(args.approval.resolve()),
        "present": approval is not None,
        "queue_binding_sha256": (approval or {}).get("queue_binding_sha256")
        if isinstance(approval, Mapping)
        else None,
        "expected_queue_binding_sha256": manifest["queue_binding_sha256"],
        "queue_binding_matches": approval_matches,
        "reason": approval_reason,
    }
    if not approval_matches:
        manifest["materialization_allowed"] = False
        manifest["blockers"] = sorted(
            set(manifest["blockers"]) | {approval_reason}
        )
    if args.write_queue:
        if manifest["materialization_allowed"] is not True:
            raise SystemExit(
                "refusing --write-queue: " + ", ".join(manifest["blockers"])
            )
        if args.queue.exists():
            raise SystemExit(
                f"refusing to overwrite existing queue; choose a new path: {args.queue}"
            )
        write_jsonl(args.queue, manifest["jobs"])
        manifest["queue_path"] = str(args.queue.resolve())
        manifest["queues_mutated"] = True
    write_json(args.output, manifest)
    print(
        f"wrote {args.output}; materialization_allowed={manifest['materialization_allowed']} "
        f"candidate_count={manifest['candidate_count']}"
    )


if __name__ == "__main__":
    main()
