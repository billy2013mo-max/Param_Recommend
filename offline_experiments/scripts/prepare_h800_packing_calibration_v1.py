#!/usr/bin/env python3
"""Materialize the prospective H800 neat-Packing calibration queue."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    DATA_DIR,
    MATRIX_DIR,
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from h800_physical_v4b_predictor import H800PhysicalV4BPredictor
from inventory_models import inventory_model
from static_packing_predictor import build_decision, load_policy


SCHEMA = "sft_h800_packing_calibration_design/v1"
JOB_SCHEMA = "sft_h800_packing_calibration_job/v1"
CAMPAIGN_ID = "h800_packing_calibration_20260803_v1"
PHASE_ID = "h800_packing_calibration_v1"
PROFILE_DIR = ARTIFACT_DIR / "bounded_memory_v2_fresh_holdout_v1" / "profiles"
DATA_BUNDLE_DIR = DATA_DIR / "bounded_memory_v2_fresh_holdout_v1"
POLICY_PATH = ARTIFACT_DIR / "static_packing_policy_v1.json"
QUEUE = MATRIX_DIR / "h800_packing_calibration_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_packing_calibration_design_v1.json"
DECISIONS = ARTIFACT_DIR / "h800_packing_calibration_frozen_decisions_v1.json"
PREDICTIONS = ARTIFACT_DIR / "h800_packing_calibration_unpacked_predictions_v1.json"
INVENTORY = ARTIFACT_DIR / "h800_packing_calibration_model_inventory_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_calibration_queue_manifest_v1.json"


MODELS = (
    {
        "id": "qwen3_8b",
        "nominal_scale_b": 8,
        "path": "/wanqing-models/Qwen3-8B",
        "tokenizer_path": "/wanqing-models/Qwen3-8B",
        "family": "qwen3",
        "template": "qwen3_nothink",
        "train_types": ["lora", "full"],
    },
    {
        "id": "qwen3_14b",
        "nominal_scale_b": 14,
        "path": "/wanqing-models/Qwen3-14B",
        "tokenizer_path": "/wanqing-models/Qwen3-14B",
        "family": "qwen3",
        "template": "qwen3_nothink",
        "train_types": ["lora", "full"],
    },
)


FAMILIES = (
    {
        "id": "C1",
        "model_id": "qwen3_8b",
        "train_type": "lora",
        "dataset_id": "fresh_s3_rare_tail_short_v1",
        "dataset_category": "short",
        "cutoff_len": 4096,
        "gpu_count": 1,
        "zero": "none",
        "gc": False,
        "unpacked_mbs": 2,
        "unpacked_basis": "fresh holdout success h800boundedv2fresh-2ab38aeb3d6f5279; MBS4 rejected above the 132.84-GiB safety line",
        "sequence": ("unpacked", "packed", "packed", "unpacked", "unpacked", "packed"),
    },
    {
        "id": "C2",
        "model_id": "qwen3_14b",
        "train_type": "lora",
        "dataset_id": "fresh_s3_broad_nontruncated_longtail_v1",
        "dataset_category": "longtail",
        "cutoff_len": 8192,
        "gpu_count": 2,
        "zero": "zero2",
        "gc": False,
        "unpacked_mbs": 1,
        "unpacked_basis": "maximum historically justified boundary: cutoff4096 MBS2 used 116.48 GiB; cutoff32768 MBS1 OOM; frozen physical upper is conservative and rejects this calibration point",
        "sequence": ("packed", "unpacked", "unpacked", "packed", "packed", "unpacked"),
    },
    {
        "id": "C3",
        "model_id": "qwen3_14b",
        "train_type": "full",
        "dataset_id": "fresh_s3_rare_tail_short_v1",
        "dataset_category": "short",
        "cutoff_len": 4096,
        "gpu_count": 2,
        "zero": "zero3",
        "gc": True,
        "unpacked_mbs": 2,
        "unpacked_basis": "fresh holdout success h800boundedv2fresh-c1bd8d539bcb43ca; MBS2 was faster than MBS1 and MBS4 is rejected by the frozen safety model",
        "sequence": ("unpacked", "packed", "packed", "unpacked", "unpacked", "packed"),
    },
    {
        "id": "C4",
        "model_id": "qwen3_8b",
        "train_type": "full",
        "dataset_id": "fresh_s3_concentrated_truncated_long_v1",
        "dataset_category": "longtail",
        "cutoff_len": 2048,
        "gpu_count": 2,
        "zero": "zero3",
        "gc": True,
        "unpacked_mbs": 4,
        "unpacked_basis": "frozen physical-shares admits MBS1-16 and frozen v4b ranks MBS4 first; MBS32 is policy-rejected",
        "sequence": ("packed", "unpacked", "unpacked", "packed", "packed", "unpacked"),
    },
)


def _binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _profile_path(dataset_id: str) -> Path:
    return PROFILE_DIR / f"{dataset_id}.qwen3_nothink.jsonl"


def _data_path(dataset_id: str) -> Path:
    return DATA_BUNDLE_DIR / f"{dataset_id}.jsonl"


def _inventory() -> dict[str, Any]:
    rows = [inventory_model(dict(model)) for model in MODELS]
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_calibration_model_inventory/v1",
        "campaign_id": CAMPAIGN_ID,
        "fixed_lora": {"rank": 32, "alpha": 32, "dropout": 0.0, "target": "all"},
        "models": rows,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(INVENTORY, report)
    return report


def _static_decisions() -> dict[str, Any]:
    policy = load_policy(POLICY_PATH.resolve())
    decisions = []
    for family in FAMILIES:
        request = {
            "request_id": f"h800-packing-calibration-{family['id'].lower()}-v1",
            "gpu_family": "H800",
            "modality": "text",
            "stage": "sft",
            "dtype": "bf16",
            "model_id": family["model_id"],
            "train_type": family["train_type"],
            "profile_path": str(_profile_path(str(family["dataset_id"])).resolve()),
            "cutoff_len": int(family["cutoff_len"]),
            "no_packing_mbs": int(family["unpacked_mbs"]),
            "gpu_count": int(family["gpu_count"]),
            "data_parallel": int(family["gpu_count"]),
            "target_gbs": 64,
            "preprocessing_num_workers": 8,
            "packing_algorithm_id": policy["packing_algorithm"]["id"],
        }
        decision = build_decision(
            request,
            policy=policy,
            policy_path=POLICY_PATH.resolve(),
            request_base=ROOT.parent,
        )
        geometry = decision["features"]["packed_batch_geometry"]
        if float(geometry["relative_error"]) > 0.05:
            raise ValueError(f"{family['id']} packed sample-GBS error exceeds 5%")
        decisions.append({"family_id": family["id"], "decision": decision})
    expected = {"C1": "off", "C2": "on", "C3": "off", "C4": "off"}
    actual = {
        row["family_id"]: row["decision"]["recommendation"]["decision"]
        for row in decisions
    }
    if actual != expected:
        raise ValueError(f"static decision boundary drifted: {actual}")
    report: dict[str, Any] = {
        "schema": "sft_h800_packing_calibration_frozen_decisions/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_before_gpu": True,
        "policy": _binding(POLICY_PATH),
        "observed_decision_map": actual,
        "draft_expectation_correction": {
            "original_labels_not_used_as_ground_truth": True,
            "reason": "the product comparison uses the fastest justified unpacked MBS; static decisions are recomputed from that baseline",
        },
        "families": decisions,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(DECISIONS, report)
    return report


def _unpacked_predictions(inventory: dict[str, Any]) -> dict[str, Any]:
    models = {str(row["id"]): row for row in inventory["models"]}
    requests = []
    for family in FAMILIES:
        gpu_count = int(family["gpu_count"])
        mbs = int(family["unpacked_mbs"])
        requests.append(
            {
                "request_id": f"packing-calibration-{family['id'].lower()}-unpacked-v1",
                "comparison_group": f"packing-calibration-{family['id'].lower()}",
                "model_id": family["model_id"],
                "training_mode": family["train_type"],
                "dataset_id": family["dataset_id"],
                "dataset_category": family["dataset_category"],
                "target_gbs": 64,
                "cutoff_len": int(family["cutoff_len"]),
                "actual_parameters": int(models[str(family["model_id"])]["actual_parameters"]),
                "gpu_count": gpu_count,
                "physical_mbs": mbs,
                "gradient_accumulation_steps": 64 // (gpu_count * mbs),
                "zero_stage": 0 if family["zero"] == "none" else int(str(family["zero"])[-1]),
                "gradient_checkpointing": bool(family["gc"]),
                "packing": False,
                "offload": False,
                "dtype": "bf16",
                "kernel_path": "fa3_orig+liger_fused_ce+adamw_torch_fused",
                "lora_rank": 32,
                "profile_tokenizer_id": "qwen3_shared_local",
                "profile_template_id": "qwen3_nothink",
            }
        )
    predictor = H800PhysicalV4BPredictor(
        model_inventory=INVENTORY,
        strict_model_inventory_binding=False,
        additional_dataset_profile_dir=PROFILE_DIR,
    )
    report = predictor.predict(requests)
    write_json(PREDICTIONS, report)
    return report


def _jobs(
    inventory: dict[str, Any], decisions: dict[str, Any], predictions: dict[str, Any]
) -> list[dict[str, Any]]:
    models = {str(row["id"]): row for row in inventory["models"]}
    decisions_by_family = {row["family_id"]: row["decision"] for row in decisions["families"]}
    predictions_by_family = {
        str(row["request_id"]).split("-")[2].upper(): row
        for row in predictions["predictions"]
    }
    treatment_repeat = {family["id"]: {"unpacked": 0, "packed": 0} for family in FAMILIES}
    jobs = []
    # Interleave families within each counterbalance position so every family
    # spans the campaign clock.  Adjacent families use mirrored treatment order.
    for position in range(6):
        for family in FAMILIES:
            family_id = str(family["id"])
            treatment = str(family["sequence"][position])
            repeat = treatment_repeat[family_id][treatment]
            treatment_repeat[family_id][treatment] += 1
            decision = decisions_by_family[family_id]
            geometry = decision["features"]["packed_batch_geometry"]
            packed = treatment == "packed"
            model = models[str(family["model_id"])]
            mbs = 1 if packed else int(family["unpacked_mbs"])
            ga = (
                int(geometry["gradient_accumulation_steps"])
                if packed
                else 64 // (int(family["gpu_count"]) * mbs)
            )
            pair_id = stable_id(
                "h800packcalpair",
                {"campaign_id": CAMPAIGN_ID, "family_id": family_id, "repeat": repeat},
            )
            row: dict[str, Any] = {
                "schema": JOB_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "phase_id": PHASE_ID,
                "candidate_role": f"packing_calibration_{family_id.lower()}_{treatment}",
                "family_id": family_id,
                "scenario_id": f"packing-calibration-{family_id.lower()}",
                "packing_pair_id": pair_id,
                "packing_treatment": treatment,
                "counterbalance_position": position,
                "repeat": repeat,
                "model_id": model["id"],
                "model_family": model["family"],
                "model_path": model["path"],
                "tokenizer_path": model["tokenizer_path"],
                "template": model["template"],
                "model_parameters": int(model["actual_parameters"]),
                "train_type": family["train_type"],
                "dataset_id": family["dataset_id"],
                "profile_id": family["dataset_id"],
                "dataset_category": family["dataset_category"],
                "data_path": str(_data_path(str(family["dataset_id"])).resolve()),
                "data_sha256": sha256_file(_data_path(str(family["dataset_id"]))),
                "dataset_profile_path": str(_profile_path(str(family["dataset_id"])).resolve()),
                "dataset_profile_sha256": sha256_file(_profile_path(str(family["dataset_id"]))),
                "cutoff_len": int(family["cutoff_len"]),
                "target_gbs": 64,
                "gpu_count": int(family["gpu_count"]),
                "zero": family["zero"],
                "zero_stage": 0 if family["zero"] == "none" else int(str(family["zero"])[-1]),
                "gc": bool(family["gc"]),
                "gradient_checkpointing": bool(family["gc"]),
                "mbs": mbs,
                "gradient_accumulation_steps": ga,
                "packing": packed,
                "expected_sample_gbs": float(geometry["expected_sample_gbs"]) if packed else 64.0,
                "expected_sample_gbs_relative_error": float(geometry["relative_error"]) if packed else 0.0,
                "offload": False,
                "gpu_type": "NVIDIA H800 140GB HBM3",
                "required_runtime_gpu_name": "NVIDIA H800",
                "hardware_id": "local_h800_140g",
                "kind": "throughput",
                "warmup_steps": 2,
                "measure_steps": 8,
                "fidelity": "formal_packing_calibration_2plus8",
                "parallel_class": "exclusive_node",
                "requires_external_node_idle": True,
                "strict_queue_order": True,
                "execution_sequence_index": len(jobs),
                "calibration_partition": {
                    "role": "calibration",
                    "split_unit_id": family_id,
                    "policy": "fit_only_never_acceptance_v1",
                },
                "declared_model_manifest_path": str(INVENTORY.resolve()),
                "declared_model_manifest_sha256": sha256_file(INVENTORY),
                "static_packing_decision_id": decision["decision_id"],
                "static_packing_decision_report_sha256": decision["report_sha256"],
                "unpacked_mbs_basis": family["unpacked_basis"],
                "unpacked_frozen_prediction_request_id": predictions_by_family[family_id]["request_id"],
                "packed_model_prediction_available_before_calibration": False,
            }
            row["job_id"] = stable_id("h800packcal", row)
            jobs.append(row)
    if len(jobs) != 24 or any(value != 3 for family in treatment_repeat.values() for value in family.values()):
        raise ValueError("Packing calibration must contain exactly three U and three P runs per family")
    return jobs


def prepare() -> dict[str, Any]:
    for family in FAMILIES:
        for path in (_profile_path(str(family["dataset_id"])), _data_path(str(family["dataset_id"]))):
            if not path.is_file():
                raise FileNotFoundError(path)
    inventory = _inventory()
    decisions = _static_decisions()
    predictions = _unpacked_predictions(inventory)
    jobs = _jobs(inventory, decisions, predictions)
    write_jsonl(QUEUE, jobs)
    jobs_dir = ARTIFACT_DIR / "h800_packing_calibration_jobs_v1"
    for job in jobs:
        write_json(jobs_dir / f"{job['job_id']}.json", job)

    source_files = {
        "strict_design": ARTIFACT_DIR / "h800_text_neat_packing_strict_acceptance_design_v1.md",
        "canary_closeout": ARTIFACT_DIR / "h800_packing_vl_canary_closeout_v2.json",
        "packing_policy": POLICY_PATH,
        "experiment_config": ROOT / "config" / "experiment.json",
        "dataset_registry": DATA_DIR / "dataset_info.json",
        "inventory": INVENTORY,
        "decisions": DECISIONS,
        "unpacked_predictions": PREDICTIONS,
        "preparer": Path(__file__).resolve(),
        "approval_freezer": ROOT / "scripts" / "freeze_h800_packing_calibration_v1.py",
        "evaluator": ROOT / "scripts" / "evaluate_h800_packing_calibration_v1.py",
        "run_job": ROOT / "scripts" / "run_job.py",
        "scheduler": ROOT / "scripts" / "scheduler.py",
        "train_entry": ROOT / "scripts" / "train_entry.py",
        "metrics_callback": ROOT / "scripts" / "metrics_callback.py",
    }
    source_files.update({f"profile_{family['id']}": _profile_path(str(family["dataset_id"])) for family in FAMILIES})
    source_files.update({f"data_{family['id']}": _data_path(str(family["dataset_id"])) for family in FAMILIES})
    design: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_gpu_waiting_for_exact_approval",
        "gpu_training_started": False,
        "fit_allowed_after_complete_results": True,
        "acceptance_allowed": False,
        "publication_allowed": False,
        "required_gpu_pool": {"gpu_ids": [0, 1], "max_gpu_count": 2, "preemption_allowed": False},
        "queue": {**_binding(QUEUE), "job_count": len(jobs), "ordered_job_ids": [job["job_id"] for job in jobs]},
        "counterbalance": {
            "blocks": 6,
            "families_per_block": 4,
            "family_sequences": {str(family["id"]): list(family["sequence"]) for family in FAMILIES},
            "all_jobs_exclusive_within_gpu_0_1_pool": True,
            "strict_queue_order": True,
        },
        "static_decisions": _binding(DECISIONS),
        "unpacked_predictions": _binding(PREDICTIONS),
        "model_inventory": _binding(INVENTORY),
        "source_bindings": {name: _binding(path) for name, path in sorted(source_files.items())},
        "measurement_contract": {
            "warmup_steps": 2,
            "measure_steps": 8,
            "token_source": "consumed_token_ledger/v1",
            "sample_gbs_relative_error_max": 0.05,
            "packed_semantics_required": True,
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    queue_manifest: dict[str, Any] = {
        "schema": "sft_h800_packing_calibration_queue_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "design": _binding(DESIGN),
        "queue": {**_binding(QUEUE), "job_count": len(jobs), "ordered_job_ids": [job["job_id"] for job in jobs]},
    }
    queue_manifest["report_sha256"] = sha256_json(queue_manifest)
    write_json(QUEUE_MANIFEST, queue_manifest)
    return {"design": _binding(DESIGN), "queue": _binding(QUEUE), "jobs": len(jobs), "decisions": decisions["observed_decision_map"]}


def main() -> None:
    import json

    print(json.dumps(prepare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
