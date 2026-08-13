#!/usr/bin/env python3
"""Prepare (but never launch) the frozen H800 prospective holdout design.

The existing H800 challenger report has already inspected its native holdout,
so it cannot be reused as publication acceptance.  This module creates a
pre-declared design for a new scenario-level holdout and records the exact
frozen artifacts it is allowed to consume.  Dataset profiles and approval are
deliberately separate inputs: until they are supplied and the hardware gate
passes, this file is a design manifest rather than an executable queue.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from common import (
    ARTIFACT_DIR,
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)


SCHEMA = "sft_h800_prospective_holdout_design/v1"
CAMPAIGN_ID = "h800_frozen_physical_v4b_fresh_holdout_20260731"
EXPECTED_GPU_NAME = "H800"
DEFAULT_REQUIRED_GPU_IDS = (4, 5, 6, 7)
MINIMUM_RATIO = 1.8
TARGET_CANDIDATE_COUNT = 24


# These are design slots, not measured recommendations.  The transition-specific
# subsets below give each declared ratio endpoint at least two sortable
# candidates while keeping the fresh campaign within its 20-30 run budget.
CANDIDATE_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "template_id": "1gpu_none_mbs16_gc_on",
        "gpu_count": 1,
        "zero_stage": 0,
        "physical_mbs": 16,
        "gradient_checkpointing": True,
    },
    {
        "template_id": "1gpu_none_mbs8_gc_on",
        "gpu_count": 1,
        "zero_stage": 0,
        "physical_mbs": 8,
        "gradient_checkpointing": True,
    },
    {
        "template_id": "2gpu_zero2_mbs8_gc_on",
        "gpu_count": 2,
        "zero_stage": 2,
        "physical_mbs": 8,
        "gradient_checkpointing": True,
    },
    {
        "template_id": "2gpu_zero3_mbs8_gc_on",
        "gpu_count": 2,
        "zero_stage": 3,
        "physical_mbs": 8,
        "gradient_checkpointing": True,
    },
    {
        "template_id": "4gpu_zero2_mbs4_gc_on",
        "gpu_count": 4,
        "zero_stage": 2,
        "physical_mbs": 4,
        "gradient_checkpointing": True,
    },
    {
        "template_id": "4gpu_zero3_mbs4_gc_on",
        "gpu_count": 4,
        "zero_stage": 3,
        "physical_mbs": 4,
        "gradient_checkpointing": True,
    },
)

# Keep the prospective campaign small while making each declared ratio
# endpoint sortable.  8B is the realistic 1->2-card family; 14B is the
# realistic 2->4-card family.  The remaining card count is intentionally not
# implied by a single unpaired slot.
TEMPLATES_BY_TRANSITION: dict[str, tuple[str, ...]] = {
    "1_to_2": (
        "1gpu_none_mbs16_gc_on",
        "1gpu_none_mbs8_gc_on",
        "2gpu_zero2_mbs8_gc_on",
        "2gpu_zero3_mbs8_gc_on",
    ),
    "2_to_4": (
        "2gpu_zero2_mbs8_gc_on",
        "2gpu_zero3_mbs8_gc_on",
        "4gpu_zero2_mbs4_gc_on",
        "4gpu_zero3_mbs4_gc_on",
    ),
}


DEFAULT_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "scenario_id": "fresh_qwen3_8b_short_512_mix",
        "model_id": "qwen3_8b",
        "dataset_profile_id": "fresh_qwen3_8b_short_512_mix_v1",
        "cutoff_len": 512,
        "task_family": "causal_dense_sft",
        "scale_out_transition": "1_to_2",
    },
    {
        "scenario_id": "fresh_qwen3_8b_multiturn_4096_mix",
        "model_id": "qwen3_8b",
        "dataset_profile_id": "fresh_qwen3_8b_multiturn_4096_mix_v1",
        "cutoff_len": 4096,
        "task_family": "causal_dense_sft",
        "scale_out_transition": "1_to_2",
    },
    {
        "scenario_id": "fresh_qwen3_8b_longcontext_32768_mix",
        "model_id": "qwen3_8b",
        "dataset_profile_id": "fresh_qwen3_8b_longcontext_32768_mix_v1",
        "cutoff_len": 32768,
        "task_family": "causal_dense_sft",
        "scale_out_transition": "1_to_2",
    },
    {
        "scenario_id": "fresh_qwen3_14b_short_512_mix",
        "model_id": "qwen3_14b",
        "dataset_profile_id": "fresh_qwen3_14b_short_512_mix_v1",
        "cutoff_len": 512,
        "task_family": "causal_dense_sft",
        "scale_out_transition": "2_to_4",
    },
    {
        "scenario_id": "fresh_qwen3_14b_multiturn_4096_mix",
        "model_id": "qwen3_14b",
        "dataset_profile_id": "fresh_qwen3_14b_multiturn_4096_mix_v1",
        "cutoff_len": 4096,
        "task_family": "causal_dense_sft",
        "scale_out_transition": "2_to_4",
    },
    {
        "scenario_id": "fresh_qwen3_14b_longcontext_32768_mix",
        "model_id": "qwen3_14b",
        "dataset_profile_id": "fresh_qwen3_14b_longcontext_32768_mix_v1",
        "cutoff_len": 32768,
        "task_family": "causal_dense_sft",
        "scale_out_transition": "2_to_4",
    },
)


def _command(command: Sequence[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(command),
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "command": list(command),
            "return_code": None,
            "stdout": "",
            "stderr": repr(error),
            "passed": False,
        }
    return {
        "command": list(command),
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "passed": result.returncode == 0,
    }


def probe_hardware(
    *,
    required_gpu_ids: Sequence[int] = DEFAULT_REQUIRED_GPU_IDS,
    expected_gpu_name: str = EXPECTED_GPU_NAME,
) -> dict[str, Any]:
    """Return a read-only H800 pool probe suitable for an approval manifest."""

    required = tuple(int(value) for value in required_gpu_ids)
    query = _command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, Any]] = []
    if query["passed"]:
        for line in query["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",", 3)]
            if len(parts) != 4:
                continue
            try:
                rows.append(
                    {
                        "index": int(parts[0]),
                        "uuid": parts[1],
                        "name": parts[2],
                        "memory_total_mib": float(parts[3]),
                    }
                )
            except ValueError:
                continue
    selected = [row for row in rows if row["index"] in set(required)]
    missing = sorted(set(required) - {row["index"] for row in selected})

    process_query = _command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    selected_uuids = {row["uuid"] for row in selected}
    processes: list[dict[str, Any]] = []
    if process_query["passed"]:
        for line in process_query["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",", 2)]
            if len(parts) != 3 or parts[0] not in selected_uuids:
                continue
            try:
                pid = int(parts[1])
            except ValueError:
                continue
            processes.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": pid,
                    "process_name": parts[2],
                }
            )

    exact_pool = (
        not missing
        and len(selected) == len(required)
        and all(expected_gpu_name.lower() in row["name"].lower() for row in selected)
    )
    return {
        "required_gpu_ids": list(required),
        "expected_gpu_name": expected_gpu_name,
        "gpu_query": query,
        "process_query": process_query,
        "selected_gpu_rows": selected,
        "missing_gpu_ids": missing,
        "selected_gpu_compute_processes": processes,
        "exact_h800_pool": exact_pool,
        "selected_pool_idle": exact_pool and not processes,
    }


def _binding(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _frozen_bindings(root: Path) -> dict[str, Any]:
    files = {
        "experiment_config": root / "config" / "experiment.json",
        "hardware_config": root / "config" / "hardware.json",
        "h800_challenger": root / "artifacts" / "h800_challenger_modeling.json",
        "structured_throughput": root / "artifacts" / "structured_throughput_modeling.json",
        "joint_throughput": root / "artifacts" / "joint_throughput_modeling.json",
        "memory_anchor_registry": root / "artifacts" / "h800_memory_anchor_registry_v1.json",
        "model_inventory": root / "artifacts" / "model_inventory.json",
        "cross_card_policy": root / "scripts" / "cross_card_scaling.py",
        "physical_predictor": root / "scripts" / "h800_physical_v4b_predictor.py",
        "campaign_gate": root / "scripts" / "check_h800_campaign_gate.py",
        "prospective_acceptance": root / "scripts" / "prospective_acceptance.py",
        "acceptance_input_builder": root / "scripts" / "build_h800_prospective_acceptance_input.py",
        "model_structure_manifest": root / "scripts" / "model_structure_manifest.py",
        "holdout_materializer": root / "scripts" / "materialize_h800_prospective_holdout.py",
        "holdout_design_implementation": Path(__file__).resolve(),
    }
    absent = [name for name, path in files.items() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Required frozen binding is absent: {absent}")
    return {name: _binding(path) for name, path in sorted(files.items())}


def _validate_scenarios(scenarios: Sequence[Mapping[str, Any]]) -> None:
    if not scenarios:
        raise ValueError("at least one prospective scenario is required")
    ids: set[str] = set()
    profile_ids: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            raise ValueError("every scenario must be an object")
        scenario_id = str(scenario.get("scenario_id") or "").strip()
        profile_id = str(scenario.get("dataset_profile_id") or "").strip()
        if not scenario_id or scenario_id in ids or not str(scenario.get("model_id") or "").strip():
            raise ValueError("scenario_id must be non-empty and unique")
        if not profile_id:
            raise ValueError("dataset_profile_id is required")
        if profile_id in profile_ids:
            # Reusing one fresh profile across model rows is allowed only when
            # the profile itself declares a model-independent processor/data
            # contract.  Keep the default conservative and require explicit
            # opt-in in a supplied manifest.
            if scenario.get("allow_profile_reuse") is not True:
                raise ValueError(
                    f"fresh profile {profile_id!r} is reused without explicit opt-in"
                )
        ids.add(scenario_id)
        profile_ids.add(profile_id)
        if str(scenario.get("task_family") or "") != "causal_dense_sft":
            raise ValueError("this V1 holdout design only accepts causal_dense_sft")
        if scenario.get("packing") is True or scenario.get("offload") is True:
            raise ValueError("packing/offload require separate evidence tracks")
        transition = str(scenario.get("scale_out_transition") or "").strip()
        if transition not in TEMPLATES_BY_TRANSITION:
            raise ValueError(
                "scale_out_transition must be one of "
                f"{sorted(TEMPLATES_BY_TRANSITION)}"
            )
        cutoff = scenario.get("cutoff_len")
        if type(cutoff) is not int or cutoff <= 0:
            raise ValueError("cutoff_len must be a positive integer")


def build_design(
    scenarios: Sequence[Mapping[str, Any]] = DEFAULT_SCENARIOS,
    *,
    root: Path = ROOT,
    hardware_probe: Mapping[str, Any] | None = None,
    required_gpu_ids: Sequence[int] = DEFAULT_REQUIRED_GPU_IDS,
) -> dict[str, Any]:
    """Build a deterministic, non-executable holdout design manifest."""

    _validate_scenarios(scenarios)
    candidate_slots: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for source in scenarios:
        scenario = dict(source)
        scenario_id = str(scenario["scenario_id"])
        model_id = str(scenario["model_id"])
        profile_id = str(scenario["dataset_profile_id"])
        transition = str(scenario["scale_out_transition"])
        template_ids = set(TEMPLATES_BY_TRANSITION[transition])
        row = {
            "scenario_id": scenario_id,
            "model_id": model_id,
            "dataset_profile_id": profile_id,
            "cutoff_len": int(scenario["cutoff_len"]),
            "target_gbs": 64,
            "task_family": "causal_dense_sft",
            "scale_out_transition": transition,
            "scale_out_endpoint_candidate_minimum": 2,
            "freshness": {
                "required": True,
                "split_unit": profile_id,
                "profile_status": "external_fresh_profile_required",
                "must_not_reuse_prior_holdout": True,
            },
            "runtime_mechanism": {
                "hardware_id": "h800",
                "packing": False,
                "offload": False,
                "dtype": "bf16",
                "kernel_path": "fa3",
            },
        }
        scenario_rows.append(row)
        for template in CANDIDATE_TEMPLATES:
            if template["template_id"] not in template_ids:
                continue
            material = {
                "campaign_id": CAMPAIGN_ID,
                "scenario_id": scenario_id,
                "template_id": template["template_id"],
            }
            candidate_slots.append(
                {
                    "candidate_slot_id": "ph-" + sha256_json(material)[:16],
                    "scenario_id": scenario_id,
                    "template_id": template["template_id"],
                    "model_id": model_id,
                    "dataset_profile_id": profile_id,
                    "target_gbs": 64,
                    "cutoff_len": int(scenario["cutoff_len"]),
                    "training_mode": "full",
                    "packing": False,
                    "offload": False,
                    "gpu_count": int(template["gpu_count"]),
                    "zero_stage": int(template["zero_stage"]),
                    "physical_mbs": int(template["physical_mbs"]),
                    "gradient_checkpointing": bool(template["gradient_checkpointing"]),
                    "status": "awaiting_fresh_profile_and_materialization",
                }
            )

    if len(candidate_slots) < 20 or len(candidate_slots) > 30:
        raise ValueError(
            f"prospective design must contain 20-30 candidate slots, got {len(candidate_slots)}"
        )
    config = read_json(root / "config" / "experiment.json")
    scaling_rule = config.get("scaling_rule") or {}
    if float(scaling_rule.get("minimum_throughput_ratio_per_doubling", 0.0)) != MINIMUM_RATIO:
        raise ValueError("experiment scaling rule is not the frozen 1.8 ratio contract")
    if [int(value) for value in scaling_rule.get("gpu_order") or []] != [1, 2, 4]:
        raise ValueError("experiment scaling rule gpu_order must be [1, 2, 4]")
    approval_path = root / "config" / "APPROVED_TO_RUN.json"
    probe = dict(
        hardware_probe
        if hardware_probe is not None
        else probe_hardware(required_gpu_ids=required_gpu_ids)
    )
    challenger = read_json(root / "artifacts" / "h800_challenger_modeling.json")
    approval_ready = approval_path.is_file()
    blocker_codes = list(challenger.get("publication_blockers") or [])
    if not probe.get("selected_pool_idle"):
        blocker_codes.append("required_h800_pool_not_idle_or_not_present")
    if not approval_ready:
        blocker_codes.append("new_campaign_approval_missing")
    blocker_codes.append("fresh_dataset_profiles_not_materialized")

    return {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "queues_mutated": False,
        "materialization_allowed": False,
        "publication_allowed": False,
        "predecessors": {
            "frozen_challenger_status": challenger.get("status"),
            "frozen_challenger_is_fresh_acceptance": (
                challenger.get("protocol", {}).get("holdout_is_fresh_publication_acceptance")
                is True
            ),
            "historical_challenger_blockers": blocker_codes,
        },
        "acceptance_contract": {
            "memory_false_safe_oom": 0,
            "memory_p95_coverage_minimum": 0.95,
            "minimum_throughput_ratio_per_doubling": MINIMUM_RATIO,
            "ratio_definition": (
                "conservative_lower_throughput_at_2N_divided_by_"
                "conservative_upper_throughput_at_N"
            ),
            "gpu_order": [1, 2, 4],
            "stop_after_first_failed_doubling": True,
            "scenario_level_split_required": True,
            "minimum_safe_candidates_per_declared_endpoint": 2,
        },
        "required_gpu_pool": {
            "gpu_ids": [int(value) for value in required_gpu_ids],
            "expected_name_contains": EXPECTED_GPU_NAME,
            "hardware_probe": probe,
        },
        "approval": {
            "path": str(approval_path.resolve()),
            "present": approval_ready,
            "must_bind_this_design": True,
            "must_bind_new_queue_and_runtime_fingerprint": True,
        },
        "frozen_bindings": _frozen_bindings(root),
        "scenarios": scenario_rows,
        "candidate_templates": [dict(template) for template in CANDIDATE_TEMPLATES],
        "candidate_slots": candidate_slots,
        "required_before_materialization": [
            "provide six new processor-bound dataset profiles",
            "prove fresh split units do not overlap the inspected challenger holdout",
            "run H800 hardware/runtime preflight on the required pool",
            "complete the existing four-card rank closeout",
            "freeze approval design and queue before touching any holdout result",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "h800_fresh_holdout_design_v1.json",
    )
    parser.add_argument(
        "--required-gpu-ids",
        default=",".join(str(value) for value in DEFAULT_REQUIRED_GPU_IDS),
        help="physical GPU IDs reserved for the H800 campaign",
    )
    args = parser.parse_args()
    try:
        required_gpu_ids = tuple(
            int(token.strip())
            for token in str(args.required_gpu_ids).split(",")
            if token.strip()
        )
    except ValueError as error:
        raise SystemExit(f"invalid --required-gpu-ids: {args.required_gpu_ids!r}") from error
    if not required_gpu_ids:
        raise SystemExit("--required-gpu-ids must not be empty")
    design = build_design(required_gpu_ids=required_gpu_ids)
    write_json(args.output, design)
    print(
        f"wrote {args.output} with {len(design['scenarios'])} scenarios and "
        f"{len(design['candidate_slots'])} non-executable candidate slots"
    )


if __name__ == "__main__":
    main()
