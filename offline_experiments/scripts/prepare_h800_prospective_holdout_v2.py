#!/usr/bin/env python3
"""Freeze a predictor-gated 24-run H800 prospective business holdout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from candidate_generator import generate_candidates
from common import ARTIFACT_DIR, ROOT, read_json, sha256_file, sha256_json, write_json
from h800_physical_v4b_predictor import H800PhysicalV4BPredictor
from prepare_h800_prospective_holdout import probe_hardware


SCHEMA = "sft_h800_prospective_holdout_design/v2"
CAMPAIGN_ID = "h800_fresh_business_physical_v4b_holdout_20260802"
DEFAULT_BUNDLE = ARTIFACT_DIR / "h800_fresh_business_data_bundle_v2.json"
DEFAULT_PREDICTIONS = ARTIFACT_DIR / "h800_frozen_predictions_before_fresh_holdout_v2.json"
DEFAULT_REQUIREMENTS = ARTIFACT_DIR / "h800_fresh_profile_requirements_v2.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_fresh_holdout_design_v2.json"
REQUIRED_GPU_IDS = (0, 1, 2, 3)
ENDPOINT_CANDIDATES = 3
MINIMUM_ENDPOINT_CANDIDATES = 2


def _binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _model_map() -> dict[str, Mapping[str, Any]]:
    inventory = read_json(ARTIFACT_DIR / "model_inventory.json")
    return {str(row["id"]): row for row in inventory["models"]}


def _endpoint_counts(transition: str) -> tuple[int, int]:
    if transition == "1_to_2":
        return 1, 2
    if transition == "2_to_4":
        return 2, 4
    raise ValueError(f"unsupported transition: {transition}")


def _endpoint_quota(scenario: Mapping[str, Any], gpu_count: int) -> int:
    """Keep the 14B two-card anchor pair and spend its extra slot at four cards."""

    if scenario["model_id"] == "qwen3_14b" and scenario["training_mode"] == "full":
        return 2 if gpu_count == 2 else 4
    return ENDPOINT_CANDIDATES


def _rank_key(row: Mapping[str, Any]) -> tuple[int, float, str]:
    rank = row.get("rank_within_gpu_count")
    proxy = (row.get("throughput") or {}).get("throughput_proxy_tokens_per_second")
    return (
        int(rank) if rank is not None else 10**9,
        -float(proxy) if proxy is not None else 0.0,
        str(row.get("request_id")),
    )


def _select_diverse(
    rows: list[dict[str, Any]],
    *,
    count: int,
    require_14b_full_anchor_neighborhood: bool,
) -> list[dict[str, Any]]:
    admitted = [
        row for row in rows
        if (row.get("memory") or {}).get("admitted") is True
        and (row.get("throughput") or {}).get("prediction_available") is True
    ]
    admitted.sort(key=_rank_key)
    selected: list[dict[str, Any]] = []

    def add(row: dict[str, Any]) -> None:
        if row not in selected:
            selected.append(row)

    if require_14b_full_anchor_neighborhood:
        for mbs in (1, 2):
            matches = [
                row for row in admitted
                if int(row["configuration"]["physical_mbs"]) == mbs
                and int(row["configuration"]["zero_stage"]) == 3
                and row["configuration"]["gradient_checkpointing"] is True
            ]
            if not matches:
                raise ValueError(
                    "14B Full two-GPU endpoint lacks admitted ZeRO-3+GC "
                    f"MBS={mbs} anchor-neighborhood candidate"
                )
            add(matches[0])

    if admitted:
        add(admitted[0])
    while len(selected) < count:
        remaining = [row for row in admitted if row not in selected]
        if not remaining:
            break
        seen_mbs = {row["configuration"]["physical_mbs"] for row in selected}
        seen_gc = {row["configuration"]["gradient_checkpointing"] for row in selected}
        seen_zero = {row["configuration"]["zero_stage"] for row in selected}

        def diversity_key(row: Mapping[str, Any]) -> tuple[int, int, int, tuple[int, float, str]]:
            config = row["configuration"]
            return (
                0 if config["gradient_checkpointing"] not in seen_gc else 1,
                0 if config["physical_mbs"] not in seen_mbs else 1,
                0 if config["zero_stage"] not in seen_zero else 1,
                _rank_key(row),
            )

        add(min(remaining, key=diversity_key))
    if len(selected) != count:
        raise ValueError(f"endpoint has only {len(admitted)} admitted candidates; {count} required")
    return selected


def _frozen_bindings(bundle_path: Path, prediction_path: Path) -> dict[str, Any]:
    files = {
        "experiment_config": ROOT / "config" / "experiment.json",
        "hardware_config": ROOT / "config" / "hardware.json",
        "model_inventory": ARTIFACT_DIR / "model_inventory.json",
        "memory_model": ARTIFACT_DIR / "h800_challenger_modeling.json",
        "throughput_model": ARTIFACT_DIR / "joint_throughput_modeling.json",
        "memory_anchor_registry": ARTIFACT_DIR / "h800_memory_anchor_registry_v1.json",
        "business_data_bundle": bundle_path,
        "frozen_predictions": prediction_path,
        "candidate_generator": ROOT / "scripts" / "candidate_generator.py",
        "predictor": ROOT / "scripts" / "h800_physical_v4b_predictor.py",
        "campaign_gate": ROOT / "scripts" / "check_h800_campaign_gate.py",
        "materializer": ROOT / "scripts" / "materialize_h800_prospective_holdout_v2.py",
        "approval_freezer": ROOT / "scripts" / "freeze_h800_prospective_live_approval_v2.py",
        "design_implementation": Path(__file__).resolve(),
        "run_job": ROOT / "scripts" / "run_job.py",
        "scheduler": ROOT / "scripts" / "scheduler.py",
    }
    return {name: _binding(path) for name, path in sorted(files.items())}


def build_design(bundle_path: Path, prediction_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = read_json(bundle_path)
    if bundle.get("schema") != "sft_h800_fresh_business_data_bundle/v2":
        raise ValueError("fresh business bundle schema mismatch")
    models = _model_map()
    predictor = H800PhysicalV4BPredictor(
        additional_dataset_profile_dir=Path(bundle["scenarios"][0]["profile_path"]).parent
    )
    all_requests: list[dict[str, Any]] = []
    generated_by_scenario: dict[str, dict[str, Any]] = {}
    for scenario in bundle["scenarios"]:
        model = models[str(scenario["model_id"])]
        endpoints = _endpoint_counts(str(scenario["scale_out_transition"]))
        request = {
            "model_id": scenario["model_id"],
            "training_mode": scenario["training_mode"],
            "dataset_id": scenario["dataset_id"],
            "dataset_category": scenario["dataset_category"],
            "target_gbs": scenario["target_gbs"],
            "cutoff_len": scenario["cutoff_len"],
            "actual_parameters": model["actual_parameters"],
            "packing": False,
            "dtype": "bf16",
            "kernel_path": "fa3_orig+liger_fused_ce+adamw_torch_fused",
            "lora_rank": 32,
            "profile_tokenizer_id": f"{scenario['model_id']}@local",
            "profile_template_id": "qwen3_nothink",
            "comparison_group": scenario["scenario_id"],
        }
        generated = generate_candidates(
            request,
            capacity_bytes=int(read_json(ROOT / "config" / "hardware.json")["memory_bytes_reported_by_torch"]),
            gpu_counts=endpoints,
            gradient_checkpointing_options=(False, True),
        )
        generated_by_scenario[str(scenario["scenario_id"])] = generated
        all_requests.extend(generated["candidates"])
    predictions = predictor.predict(all_requests)
    write_json(prediction_path, predictions)
    prediction_rows = {
        str(row["request_id"]): row for row in predictions["predictions"]
    }

    candidate_slots: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for scenario in bundle["scenarios"]:
        scenario_id = str(scenario["scenario_id"])
        endpoints = _endpoint_counts(str(scenario["scale_out_transition"]))
        scenario_prediction_rows = [
            prediction_rows[str(candidate["request_id"])]
            for candidate in generated_by_scenario[scenario_id]["candidates"]
        ]
        endpoint_summary: list[dict[str, Any]] = []
        for gpu_count in endpoints:
            endpoint_rows = [
                row for row in scenario_prediction_rows
                if int(row["configuration"]["gpu_count"]) == gpu_count
            ]
            quota = _endpoint_quota(scenario, gpu_count)
            selected = _select_diverse(
                endpoint_rows,
                count=quota,
                require_14b_full_anchor_neighborhood=(
                    scenario["model_id"] == "qwen3_14b"
                    and scenario["training_mode"] == "full"
                    and gpu_count == 2
                ),
            )
            endpoint_summary.append(
                {
                    "gpu_count": gpu_count,
                    "generated_candidates": len(endpoint_rows),
                    "memory_admitted_candidates": sum(
                        (row.get("memory") or {}).get("admitted") is True
                        for row in endpoint_rows
                    ),
                    "selected_candidates": len(selected),
                }
            )
            for row in selected:
                config = row["configuration"]
                slot_material = {
                    "campaign_id": CAMPAIGN_ID,
                    "scenario_id": scenario_id,
                    "request_id": row["request_id"],
                }
                candidate_slots.append(
                    {
                        "candidate_slot_id": "ph2-" + sha256_json(slot_material)[:16],
                        "scenario_id": scenario_id,
                        "request_id": row["request_id"],
                        "model_id": scenario["model_id"],
                        "dataset_id": scenario["dataset_id"],
                        "training_mode": scenario["training_mode"],
                        "target_gbs": scenario["target_gbs"],
                        "cutoff_len": scenario["cutoff_len"],
                        "gpu_count": config["gpu_count"],
                        "zero_stage": config["zero_stage"],
                        "physical_mbs": config["physical_mbs"],
                        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
                        "gradient_checkpointing": config["gradient_checkpointing"],
                        "packing": False,
                        "offload": False,
                        "memory_admitted": True,
                        "memory_admission_source": row["memory"]["admission_source"],
                        "memory_upper_reserved_bytes": row["memory"]["admission_upper_reserved_bytes"],
                        "safe_limit_bytes": row["memory"]["safe_limit_bytes"],
                        "v4b_rank_within_gpu_count": row["rank_within_gpu_count"],
                        "v4b_throughput_proxy": row["throughput"]["throughput_proxy_tokens_per_second"],
                    }
                )
        scenario_rows.append(
            {
                **scenario,
                "runtime_dataset_registered": True,
                "scale_out_endpoint_candidate_minimum": MINIMUM_ENDPOINT_CANDIDATES,
                "endpoint_summary": endpoint_summary,
                "freshness": {
                    "required": True,
                    "split_unit": scenario["dataset_id"],
                    "profile_path": scenario["profile_path"],
                    "data_path": scenario["data_path"],
                    "runtime_dataset_registered": True,
                    "must_not_reuse_prior_gpu_outcomes": True,
                },
            }
        )
    if len(candidate_slots) != 24:
        raise ValueError(f"expected exactly 24 selected slots, got {len(candidate_slots)}")
    probe = probe_hardware(required_gpu_ids=REQUIRED_GPU_IDS)
    design = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "queues_mutated": False,
        "materialization_allowed": True,
        "publication_allowed": False,
        "required_gpu_pool": {
            "gpu_ids": list(REQUIRED_GPU_IDS),
            "expected_name_contains": "H800",
            "availability_policy": "exact_idle_pool",
            "join_busy_pool_allowed": False,
            "hardware_probe_at_design": probe,
        },
        "acceptance_contract": {
            "memory_false_safe_oom": 0,
            "memory_p95_coverage_minimum": 0.95,
            "minimum_throughput_ratio_per_doubling": 1.8,
            "gpu_order": [1, 2, 4],
            "scenario_level_split_required": True,
            "minimum_safe_candidates_per_declared_endpoint": MINIMUM_ENDPOINT_CANDIDATES,
            "top1_regret_target": 0.10,
        },
        "frozen_bindings": _frozen_bindings(bundle_path, prediction_path),
        "scenarios": scenario_rows,
        "candidate_slots": candidate_slots,
        "candidate_count": len(candidate_slots),
        "selection_policy": (
            "physical-shares/anchor admission first; then v4b rank with explicit "
            "GC/MBS/ZeRO diversity and a mandatory 14B Full two-card MBS=1/2 neighborhood"
        ),
    }
    return design, predictions


def build_requirements(design: Mapping[str, Any], design_path: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    processor = bundle["processor_contract"]
    split = bundle["split_manifest"]
    rows = []
    for scenario in design["scenarios"]:
        rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "dataset_profile_id": scenario["dataset_profile_id"],
                "model_id": scenario["model_id"],
                "cutoff_len": scenario["cutoff_len"],
                "target_gbs": scenario["target_gbs"],
                "scale_out_transition": scenario["scale_out_transition"],
                "required_bindings": {
                    "profile_path": scenario["profile_path"],
                    "profile_sha256": scenario["profile_sha256"],
                    "data_path": scenario["data_path"],
                    "data_sha256": scenario["data_sha256"],
                    "runtime_dataset_registered": True,
                    "processor_contract_sha256": processor["sha256"],
                    "split_manifest_sha256": split["sha256"],
                },
                "required_profile_row_fields": [
                    "sample_id", "total_tokens", "label_tokens", "turns", "assistant_turns"
                ],
            }
        )
    return {
        "schema": "sft_h800_fresh_profile_requirements/v1",
        "campaign_id": design["campaign_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_binding": _binding(design_path),
        "processor_contract": processor,
        "split_manifest": split,
        "scenarios": rows,
        "ready_for_materialization": True,
        "publication_allowed": False,
        "gpu_training_started": False,
        "queues_mutated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    design, _ = build_design(args.bundle, args.predictions)
    write_json(args.output, design)
    requirements = build_requirements(design, args.output, read_json(args.bundle))
    write_json(args.requirements, requirements)
    print(
        f"wrote {args.output}; scenarios={len(design['scenarios'])}; "
        f"selected_safe_slots={len(design['candidate_slots'])}"
    )


if __name__ == "__main__":
    main()
