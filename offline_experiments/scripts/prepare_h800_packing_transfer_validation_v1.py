#!/usr/bin/env python3
"""Prepare a small prospective transfer validation for the accepted Packing ranker.

The current Packing ranking rule is already fixed.  This campaign therefore
does not repeat the full candidate Cartesian product.  It selects six representative
pure-text model/training/data/GPU scenes and four predeclared candidates per
scene.  The 24 outcomes may validate transfer, but may not tune the fitted
model, safety margin, candidate selection rule, or acceptance thresholds.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from common import (
    ARTIFACT_DIR,
    MATRIX_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from prepare_h800_packing_config_ranking_v1 import (
    DATASET_REGISTRY,
    GPU_IDS,
    QUEUE_HOLDOUT as CANDIDATE_BASIS,
    _round_robin_order,
)


CAMPAIGN_ID = "h800_packing_transfer_validation_20260817_v1"
PHASE_ID = "h800_packing_transfer_validation_v1"
JOB_SCHEMA = "sft_h800_packing_transfer_validation_job/v1"
DESIGN_SCHEMA = "sft_h800_packing_transfer_validation_design/v1"

MODEL_INVENTORY = (
    ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_with_hybrid_v1.json"
)
FITTED_PACKING_MODEL = ARTIFACT_DIR / "h800_packing_phase_c_model_refit_v1.json"
PACKING_RANKING_ACCEPTANCE = ARTIFACT_DIR / "h800_packing_ranking_acceptance_v1.json"
QUEUE = MATRIX_DIR / "h800_packing_transfer_validation_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_packing_transfer_validation_design_v1.json"
EXPERIMENT = (
    ROOT
    / "packing_config_ranking_staging"
    / "experiment.h800_packing_transfer_validation_v1.json"
)
FROZEN_PREDICTIONS = (
    ARTIFACT_DIR / "h800_packing_transfer_validation_frozen_predictions_v1.json"
)


@dataclass(frozen=True)
class Scene:
    scene_id: str
    model_id: str
    train_type: str
    workload_id: str
    gpu_count: int
    reason: str


# Six scenes across five models cover the dense-Qwen domain that the current frozen throughput
# ranker can actually score.  Qwen3.5/Qwen3.6 hybrid attention is deliberately
# excluded here: its throughput head still needs training, so pretending to
# validate it with an unavailable prediction would be invalid.
SCENES = (
    Scene("S1", "qwen3_1p7b", "full", "PH02", 1, "small dense/full + short-tail data"),
    Scene("S2", "qwen3_1p7b", "lora", "PH03", 1, "small dense/LoRA + medium data"),
    Scene("S3", "qwen3_4b", "full", "PH06", 2, "medium dense/full + long-tail data"),
    Scene("S4", "qwen3_8b", "full", "PH03", 2, "anchor scale/full + medium data"),
    Scene("S5", "qwen3_14b", "lora", "PH02", 4, "large dense/LoRA + short-tail data"),
    Scene("S6", "qwen3_32b", "lora", "PH06", 4, "largest frozen-template dense/LoRA + long-tail data"),
)

EXPECTED_SCENES = 6
CANDIDATES_PER_SCENE = 4
EXPECTED_JOBS = EXPECTED_SCENES * CANDIDATES_PER_SCENE


def _execution_order(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build full waves only for card counts present in this small campaign."""

    ordered: list[dict[str, Any]] = []
    wave_index = 0
    for gpu_count in sorted({int(row["gpu_count"]) for row in jobs}):
        current = _round_robin_order(
            [row for row in jobs if int(row["gpu_count"]) == gpu_count]
        )
        capacity = len(GPU_IDS) // gpu_count
        if len(current) % capacity:
            raise ValueError(
                f"gpu_count={gpu_count} has {len(current)} jobs, which cannot "
                f"form full {capacity}-job waves"
            )
        for offset, job in enumerate(current):
            job["execution_wave_index"] = wave_index + offset // capacity
            job["execution_wave_gpu_count"] = gpu_count
            job["execution_wave_capacity"] = capacity
        wave_index += len(current) // capacity
        ordered.extend(current)
    for index, job in enumerate(ordered):
        job["execution_sequence_index"] = index
    return ordered


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _models() -> dict[str, dict[str, Any]]:
    inventory = read_json(MODEL_INVENTORY)
    by_id = {str(row.get("id")): row for row in inventory.get("models") or []}
    selected: dict[str, dict[str, Any]] = {}
    for scene in SCENES:
        model = by_id.get(scene.model_id)
        if not isinstance(model, dict):
            raise ValueError(f"model inventory lacks {scene.model_id}")
        if scene.train_type not in set(model.get("train_types") or []):
            raise ValueError(f"{scene.model_id} does not support {scene.train_type}")
        if not Path(str(model["path"])).is_dir():
            raise FileNotFoundError(model["path"])
        selected[scene.model_id] = copy.deepcopy(model)
    return selected


def _four_candidates(scene: Scene, basis: list[dict[str, Any]]) -> list[dict[str, Any]]:
    available = [
        row
        for row in basis
        if row.get("ranking_eligible") is True
        and row.get("workload_id") == scene.workload_id
        and int(row.get("gpu_count", 0)) == scene.gpu_count
    ]
    cutoffs = sorted({int(row["cutoff_len"]) for row in available})
    if len(cutoffs) != 3:
        raise ValueError(f"{scene.scene_id} does not have three candidate cutoffs")
    low, high = cutoffs[0], cutoffs[-1]
    if scene.model_id == "qwen3_32b":
        # The frozen V5 template has one exact 32B mechanism (4-card ZeRO-3
        # with GC on).  Validate all three cutoff choices and spend the fourth
        # slot on a predeclared repeat instead of inventing an unsupported
        # mechanism prediction.
        selected = [
            copy.deepcopy(by_row)
            for cutoff in cutoffs
            for by_row in available
            if int(by_row["cutoff_len"]) == cutoff
            and int(by_row["zero_stage"]) == 3
            and bool(by_row["gc"]) is True
        ]
        if len(selected) != 3:
            raise ValueError(f"{scene.scene_id} lacks the exact 32B cutoff triplet")
        repeat = copy.deepcopy(selected[1])
        repeat["transfer_noise_repeat"] = True
        return [*selected, repeat]

    keys = (
        ((low, 0, False), (low, 0, True), (high, 0, False), (high, 0, True))
        if scene.gpu_count == 1
        else (
            (low, 2, False),
            (low, 2, True),
            (high, 3, False),
            (high, 2, False),
        )
        if scene.model_id == "qwen3_4b" and scene.train_type == "full"
        else (
            (low, 2, False),
            (low, 3, True),
            (high, 2, True),
            (high, 3, False),
        )
    )
    by_key = {
        (int(row["cutoff_len"]), int(row["zero_stage"]), bool(row["gc"])): row
        for row in available
    }
    missing = [key for key in keys if key not in by_key]
    if missing:
        raise ValueError(f"{scene.scene_id} lacks candidate arms {missing}")
    return [copy.deepcopy(by_key[key]) for key in keys]


def _job(
    source: Mapping[str, Any],
    *,
    scene: Scene,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    row = copy.deepcopy(dict(source))
    row.pop("job_id", None)
    is_repeat = bool(row.pop("transfer_noise_repeat", False))
    family = str(model["family"])
    cutoff = int(row["cutoff_len"])
    group_id = f"{scene.scene_id}-{scene.model_id}-{scene.train_type}-{scene.workload_id}-dp{scene.gpu_count}"
    row.update(
        {
            "schema": JOB_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "phase_id": PHASE_ID,
            "candidate_role": "prospective_transfer_validation_only",
            "scene_id": scene.scene_id,
            "scene_selection_reason": scene.reason,
            "model_id": scene.model_id,
            "model_family": family,
            "release_family": family,
            "architecture_route": "dense_full_attention",
            "model_path": str(Path(str(model["path"])).resolve()),
            "tokenizer_path": str(Path(str(model["tokenizer_path"])).resolve()),
            "template": "qwen3_nothink",
            "model_parameters": int(model["actual_parameters"]),
            "train_type": scene.train_type,
            "scenario_id": group_id,
            "ranking_group_id": group_id,
            "fixed_cutoff_mechanism_group_id": f"{group_id}-c{cutoff}",
            "measurement_role": "transfer_noise_repeat" if is_repeat else "primary_ranking_candidate",
            "repeat": 1 if is_repeat else 0,
            "ranking_eligible": not is_repeat,
            "publication_allowed": False,
        }
    )
    row["ranking_contract"]["primary_group"] = group_id
    row["ranking_contract"]["fixed_cutoff_diagnostic_group"] = row[
        "fixed_cutoff_mechanism_group_id"
    ]
    row["model_input_projection"].update(
        {
            "accepted_virtual_mbs_ranking_rule_transfer": True,
            "packing_specific_coefficient": 0.0,
            "validation_scene_was_selected_before_outcomes": True,
        }
    )
    row.pop("environment_overlay", None)
    row["job_id"] = stable_id("h800packxfer", row)
    return row


def _experiment() -> dict[str, Any]:
    current = read_json(ROOT / "config" / "experiment.json")
    return {
        "schema_version": 1,
        "training_scope": {
            "phase_id": PHASE_ID,
            "model_ids": sorted({scene.model_id for scene in SCENES}),
            "train_types": sorted({scene.train_type for scene in SCENES}),
            "gpu_ids": list(GPU_IDS),
            "exclusive_node_gpu_ids": list(GPU_IDS),
            "max_gpu_count": max(scene.gpu_count for scene in SCENES),
            "stage": "sft",
            "precision": "bf16",
            "gpu_type": "NVIDIA H800 140GB HBM3",
            "gpu_counts": sorted({scene.gpu_count for scene in SCENES}),
            "global_batch_sizes": [256],
            "objective": "small prospective transfer validation for the accepted zero-coefficient virtual-MBS Packing ranker",
        },
        "fixed_runtime": current["fixed_runtime"],
        "measurement": {
            **current["measurement"],
            "throughput_warmup_steps": 3,
            "throughput_measure_steps": 12,
            "performance_parallelism": "disjoint_gpu_masks",
            "scheduler_order_policy": "strict_homogeneous_card_count_waves",
            "formal_throughput_requires_exclusive_node": False,
        },
    }


def prepare() -> dict[str, Any]:
    fitted = read_json(FITTED_PACKING_MODEL)
    center = fitted.get("throughput_effect_center") or {}
    if center.get("status") != "paired_effect_center_refit_complete_fit_only":
        raise ValueError("the fitted Packing effect model binding drifted")
    if int(center.get("matched_repeat_pairs", 0)) != 61:
        raise ValueError("expected the 61-pair fitted Packing model")
    accepted_ranker = read_json(PACKING_RANKING_ACCEPTANCE)
    decision = accepted_ranker.get("decision") or {}
    if (
        decision.get("current_empirical_ranking_gate_passed") is not True
        or decision.get("packing_coefficient_required_for_current_ranking") is not False
    ):
        raise ValueError("the accepted zero-coefficient Packing ranking rule drifted")

    basis = read_jsonl(CANDIDATE_BASIS)
    models = _models()
    rows = [
        _job(source, scene=scene, model=models[scene.model_id])
        for scene in SCENES
        for source in _four_candidates(scene, basis)
    ]
    if len(rows) != EXPECTED_JOBS or len({row["job_id"] for row in rows}) != EXPECTED_JOBS:
        raise ValueError("transfer validation must contain 24 unique jobs")
    rows = _execution_order(rows)
    write_jsonl(QUEUE, rows)
    write_json(EXPERIMENT, _experiment())

    design: dict[str, Any] = {
        "schema": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_not_authorized_not_started",
        "gpu_training_started": False,
        "problem": "validate transfer of the accepted zero-coefficient virtual-MBS Packing ranker without re-running the product Cartesian space",
        "supersession": {
            "exhaustive_1584_job_design_must_not_run": True,
            "narrow_132_job_holdout_must_not_be_treated_as_the_only_acceptance_scope": True,
        },
        "upstream_fit_diagnostic_not_used_as_production_ranker": {
            "matched_repeat_pairs": 61,
            "setting_rows": 23,
            "profile_groups": 10,
            "binding": _binding(FITTED_PACKING_MODEL),
        },
        "accepted_ranking_rule_evidence": {
            "binding": _binding(PACKING_RANKING_ACCEPTANCE),
            "prediction_contract": "frozen Unpacked V5 at static mean samples per pack; Packing coefficient is zero",
            "historical_scenarios": int(accepted_ranker["historical_end_to_end"]["scenario_count"]),
            "final_direct_transfer_scenarios": int(accepted_ranker["final_direct_transfer"]["scenario_count"]),
            "combined_pairwise_accuracy_is_diagnostic": float(accepted_ranker["combined_diagnostic_counts"]["pairwise_accuracy"]),
            "previous_allowed_scope": str(decision["allowed_scope"]),
        },
        "validation_scope": {
            "scenes": EXPECTED_SCENES,
            "candidates_per_scene": CANDIDATES_PER_SCENE,
            "jobs": EXPECTED_JOBS,
            "models": len({scene.model_id for scene in SCENES}),
            "dataset_sources": len({scene.workload_id for scene in SCENES}),
            "gpu_counts": sorted({scene.gpu_count for scene in SCENES}),
            "training_modes": sorted({scene.train_type for scene in SCENES}),
        },
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "model_id": scene.model_id,
                "train_type": scene.train_type,
                "workload_id": scene.workload_id,
                "gpu_count": scene.gpu_count,
                "reason": scene.reason,
                "candidate_count": CANDIDATES_PER_SCENE,
            }
            for scene in SCENES
        ],
        "prospective_contract": {
            "scene_and_candidate_arms_selected_before_outcomes": True,
            "predictions_must_be_frozen_before_first_job": True,
            "frozen_predictions_path": str(FROZEN_PREDICTIONS.resolve()),
            "validation_outcomes_may_tune_model_or_margin": False,
            "failed_transfer_requires_new_holdout_after_repair": True,
        },
        "acceptance_contract": {
            "all_six_scenes_must_have_four_terminal_jobs": True,
            "ranking_candidates": 23,
            "noise_repeats": 1,
            "pairwise_ordering_accuracy_min": 0.90,
            "hit90_required": "6/6",
            "worst_top1_regret_max_exclusive": 0.10,
            "false_safe_oom_required": 0,
            "oom_gate_is_reported_vacuous_if_no_oom_occurs": True,
            "no_pooled_average_may_hide_a_failed_scene": True,
        },
        "scope_limit": "H800 only; the local machine cannot prospectively validate another GPU type",
        "bindings": {
            "queue": _binding(QUEUE),
            "experiment": _binding(EXPERIMENT),
            "candidate_basis": _binding(CANDIDATE_BASIS),
            "dataset_registry": _binding(DATASET_REGISTRY),
            "model_inventory": _binding(MODEL_INVENTORY),
            "packing_ranking_acceptance": _binding(PACKING_RANKING_ACCEPTANCE),
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    return {
        "design": _binding(DESIGN),
        "queue": _binding(QUEUE),
        "experiment": _binding(EXPERIMENT),
        "scenes": EXPECTED_SCENES,
        "jobs": EXPECTED_JOBS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
