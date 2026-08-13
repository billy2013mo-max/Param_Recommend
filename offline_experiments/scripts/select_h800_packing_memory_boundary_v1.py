#!/usr/bin/env python3
"""Select an eight-setting H800 Packing memory-boundary DOE without launching GPUs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from common import ARTIFACT_DIR, MATRIX_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from h800_physical_v4b_predictor import H800PhysicalV4BPredictor


SCHEMA = "sft_h800_packing_memory_boundary_selection/v1"
CAMPAIGN_ID = "h800_packing_memory_boundary_20260805_v1"
MODEL_INVENTORY = ARTIFACT_DIR / "model_inventory.json"
MODEL_REFIT = ARTIFACT_DIR / "h800_packing_phase_c_model_refit_v1.json"
PHASE_C = ARTIFACT_DIR / "h800_packing_profile_phase_c_results_v1.json"
PHASE_B_QUEUE = MATRIX_DIR / "h800_packing_profile_phase_b_v1.jsonl"
PROFILE_DIR = ARTIFACT_DIR / "packing_profile_phase_b_v1/profiles"
DATAPROFILE_DIR = ARTIFACT_DIR / "packing_dataprofile_v2"
HARDWARE = ROOT / "config/hardware.json"

PREDICTIONS = ARTIFACT_DIR / "h800_packing_memory_boundary_candidate_predictions_v1.json"
OUTPUT = ARTIFACT_DIR / "h800_packing_memory_boundary_selection_v1.json"
MARKDOWN = ARTIFACT_DIR / "h800_packing_memory_boundary_selection_v1.md"

PHYSICAL_MBS = 1
EPSILON_GBS = 0.10
GIB = float(2**30)

CHAIN_SPECS = (
    {
        "chain_id": "B1_8B_LoRA_W8_Z2_GCoff",
        "workload_id": "W8",
        "model_id": "qwen3_8b",
        "training_mode": "lora",
        "gpu_count": 2,
        "zero_stage": 2,
        "gc": False,
        "target_gbs": 256,
        "target_fractions": (0.69, 0.95),
        "refit_domain_relation": "same_model_and_training_mode_cutoff_extrapolation",
        "selection_reason": "Revisit the high Phase-C W8 GC-off Packing memory arm and bracket its cutoff boundary.",
    },
    {
        "chain_id": "B2_14B_LoRA_W7_Z2_GCoff",
        "workload_id": "W7",
        "model_id": "qwen3_14b",
        "training_mode": "lora",
        "gpu_count": 2,
        "zero_stage": 2,
        "gc": False,
        "target_gbs": 128,
        "target_fractions": (0.89, 0.94),
        "refit_domain_relation": "model_scale_transfer_ood",
        "selection_reason": "Add model-scale transfer on high-P99 W7 using the high-activation ZeRO-2/GC-off boundary mechanism.",
    },
    {
        "chain_id": "B3_14B_Full_W3_Z3_GCoff_G2",
        "workload_id": "W3",
        "model_id": "qwen3_14b",
        "training_mode": "full",
        "gpu_count": 2,
        "zero_stage": 3,
        "gc": False,
        "target_gbs": 128,
        "target_fractions": (0.87, 0.96),
        "refit_domain_relation": "model_scale_and_full_training_transfer_ood",
        "selection_reason": "Add Full-training model-state and activation pressure under ZeRO-3/GC-off on a natural multiturn profile.",
    },
    {
        "chain_id": "B4_14B_Full_W3_Z2_GCon_G2",
        "workload_id": "W3",
        "model_id": "qwen3_14b",
        "training_mode": "full",
        "gpu_count": 2,
        "zero_stage": 2,
        "gc": True,
        "target_gbs": 128,
        "target_fractions": (0.76, 0.81),
        "refit_domain_relation": "model_scale_and_full_training_transfer_ood",
        "selection_reason": "Provide the GC-on Full-training boundary control under ZeRO-2 for the same W3 profile.",
    },
)


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _models() -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in read_json(MODEL_INVENTORY)["models"]}


def _workloads() -> dict[str, dict[str, Any]]:
    rows = read_jsonl(PHASE_B_QUEUE)
    result: dict[str, dict[str, Any]] = {}
    for workload in {str(spec["workload_id"]) for spec in CHAIN_SPECS}:
        row = next(item for item in rows if str(item["workload_id"]).upper() == workload)
        result[workload] = row
    return result


def _dataprofile(workload_id: str) -> dict[str, Any]:
    return read_json(DATAPROFILE_DIR / f"{workload_id.lower()}_packing_dataprofile_v2.json")


def _curve(workload_id: str) -> dict[int, dict[str, Any]]:
    return {
        int(row["cutoff_len"]): row
        for row in _dataprofile(workload_id)["packing_curve"]
        if 8192 <= int(row["cutoff_len"]) <= 40960
    }


def _requests() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    models = _models()
    workloads = _workloads()
    metadata: dict[str, dict[str, Any]] = {}
    requests: list[dict[str, Any]] = []
    for spec in CHAIN_SPECS:
        model = models[str(spec["model_id"])]
        workload = workloads[str(spec["workload_id"])]
        max_context = int(model["max_position_embeddings"])
        for cutoff, curve in sorted(_curve(str(spec["workload_id"])).items()):
            if cutoff > max_context:
                continue
            n_pack = float(curve["samples_per_pack"]["mean"])
            target_gbs = int(spec["target_gbs"])
            for packing in (False, True):
                request_id = (
                    f"boundary-{spec['chain_id'].lower()}-c{cutoff}-"
                    f"{'p' if packing else 'u'}"
                )
                ga = (
                    max(1, round(target_gbs / (int(spec["gpu_count"]) * n_pack)))
                    if packing
                    else target_gbs // (int(spec["gpu_count"]) * PHYSICAL_MBS)
                )
                request = {
                    "request_id": request_id,
                    # The frozen predictor requires one immutable user scenario
                    # per ranking group; Packing U/P are paired after prediction.
                    "comparison_group": request_id,
                    "model_id": spec["model_id"],
                    "training_mode": spec["training_mode"],
                    "dataset_id": workload["dataset_id"],
                    "dataset_category": workload["dataset_category"],
                    "target_gbs": target_gbs,
                    "cutoff_len": cutoff,
                    "actual_parameters": int(model["actual_parameters"]),
                    "gpu_count": int(spec["gpu_count"]),
                    "physical_mbs": PHYSICAL_MBS,
                    "gradient_accumulation_steps": ga,
                    "zero_stage": int(spec["zero_stage"]),
                    "gradient_checkpointing": bool(spec["gc"]),
                    "packing": packing,
                    "offload": False,
                    "dtype": "bf16",
                    "kernel_path": "fa3_orig+liger_fused_ce+adamw_torch_fused",
                    "lora_rank": 32 if spec["training_mode"] == "lora" else 0,
                    "profile_tokenizer_id": "qwen3_8b@local",
                    "profile_template_id": workload["template"],
                }
                requests.append(request)
                metadata[request_id] = {
                    **spec,
                    "cutoff_len": cutoff,
                    "packing": packing,
                    "n_pack_mean": n_pack,
                    "n_pack_p99": float(curve["samples_per_pack"]["p99"]),
                    "pack_utilization": float(curve["pack_utilization"]),
                    "expected_sample_gbs": n_pack * int(spec["gpu_count"]) * ga if packing else target_gbs,
                    "gradient_accumulation_steps": ga,
                    "dataset_id": workload["dataset_id"],
                    "dataset_profile_path": workload["dataset_profile_path"],
                    "data_path": workload["data_path"],
                    "model_path": model["path"],
                    "tokenizer_path": model["tokenizer_path"],
                    "actual_parameters": int(model["actual_parameters"]),
                    "max_position_embeddings": max_context,
                }
    return requests, metadata


def _memory_model() -> tuple[dict[str, Any], float]:
    report = read_json(MODEL_REFIT)
    memory = report["memory_center"]
    selected = str(memory["selection"]["selected"])
    model = memory["candidates"][selected]
    residuals = []
    for row in model["oof_predictions"]:
        predicted = float(row["predicted_reserved_gib"])
        observed = float(row["observed_reserved_gib"])
        residuals.append(math.log(observed / predicted))
    # With 20 calibration arms, the finite-sample one-sided 95% rank is the
    # maximum.  This remains a provisional DOE envelope, not an accepted guard.
    q95 = max(residuals)
    return model, q95


def _profile_stats(path_text: str) -> dict[str, float]:
    path = Path(path_text)
    lengths = [
        float(json.loads(line)["total_tokens"])
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]
    mean = statistics.fmean(lengths)
    return {
        "mean": mean,
        "cv": statistics.pstdev(lengths) / mean,
        "p99": float(np.percentile(np.asarray(lengths, dtype=float), 99)),
    }


def _refit_center(
    physical_center_gib: float,
    metadata: dict[str, Any],
    model: dict[str, Any],
) -> float:
    profile = _profile_stats(str(metadata["dataset_profile_path"]))
    packing = float(bool(metadata["packing"]))
    zero3 = float(int(metadata["zero_stage"]) == 3)
    gc_off = float(not bool(metadata["gc"]))
    n_pack = float(metadata["n_pack_mean"])
    feature_values = {
        "log2_physical_center_gib": math.log2(physical_center_gib),
        "log2_cutoff": math.log2(float(metadata["cutoff_len"])),
        "log2_mean_length": math.log2(profile["mean"]),
        "length_cv": profile["cv"],
        "p99_length_to_cutoff": profile["p99"] / float(metadata["cutoff_len"]),
        "log2_effective_samples_per_physical_row": math.log2(n_pack if packing else 1.0),
        "zero3": zero3,
        "gc_off": gc_off,
        "packing": packing,
        "packing_x_log2_n_pack_mean": packing * math.log2(n_pack),
        "packing_x_zero3": packing * zero3,
        "packing_x_gc_off": packing * gc_off,
    }
    names = tuple(model["feature_names"])
    fitted = model["final_fit_only_model"]
    raw = np.asarray([feature_values[name] for name in names], dtype=float)
    mean = np.asarray(fitted["standardizer_mean"], dtype=float)
    scale = np.asarray(fitted["standardizer_scale"], dtype=float)
    coefficient = np.asarray(fitted["coefficients"], dtype=float)
    log_residual = float(fitted["intercept"]) + float(np.dot((raw - mean) / scale, coefficient))
    return physical_center_gib * math.exp(log_residual)


def _scan() -> tuple[list[dict[str, Any]], float]:
    requests, metadata = _requests()
    predictor = H800PhysicalV4BPredictor(
        model_inventory=MODEL_INVENTORY,
        strict_model_inventory_binding=False,
        additional_dataset_profile_dir=PROFILE_DIR,
    )
    report = predictor.predict(requests)
    write_json(PREDICTIONS, report)
    model, q95 = _memory_model()
    capacity_gib = float(read_json(HARDWARE)["memory_bytes_reported_by_torch"]) / GIB
    rows = []
    for prediction in report["predictions"]:
        request_id = str(prediction["request_id"])
        memory = prediction["memory"]
        if memory.get("prediction_available") is not True:
            raise ValueError(f"memory prediction unavailable: {request_id}")
        meta = metadata[request_id]
        physical_center = float(memory["reserved_center_bytes"]) / GIB
        refit_center = _refit_center(physical_center, meta, model)
        physical_p95 = float(memory["operational_p95_reserved_bytes"]) / GIB
        provisional_upper = max(physical_p95, refit_center * math.exp(q95))
        rows.append(
            {
                "request_id": request_id,
                **meta,
                "physical_center_gib": physical_center,
                "refit_center_gib": refit_center,
                "physical_operational_p95_gib": physical_p95,
                "provisional_design_upper_gib": provisional_upper,
                "refit_center_capacity_fraction": refit_center / capacity_gib,
                "physical_p95_capacity_fraction": physical_p95 / capacity_gib,
                "provisional_upper_capacity_fraction": provisional_upper / capacity_gib,
                "support_label": prediction["support"]["label"],
            }
        )
    return rows, q95


def _select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for spec in CHAIN_SPECS:
        chain_rows = [row for row in rows if row["chain_id"] == spec["chain_id"]]
        by_cutoff: dict[int, list[dict[str, Any]]] = {}
        for cutoff in sorted({int(row["cutoff_len"]) for row in chain_rows}):
            arms = [row for row in chain_rows if int(row["cutoff_len"]) == cutoff]
            if len(arms) != 2:
                raise ValueError(f"boundary cutoff is not a U/P pair: {spec['chain_id']}/{cutoff}")
            packed = next(row for row in arms if row["packing"])
            target_gbs = float(packed["target_gbs"])
            expected_error = abs(float(packed["expected_sample_gbs"]) - target_gbs) / target_gbs
            p99_global = float(packed["n_pack_p99"]) * int(packed["gpu_count"])
            if expected_error > EPSILON_GBS or p99_global > target_gbs * (1.0 + EPSILON_GBS):
                continue
            for arm in arms:
                arm["expected_sample_gbs_relative_error"] = (
                    abs(float(arm["expected_sample_gbs"]) - float(arm["target_gbs"]))
                    / float(arm["target_gbs"])
                )
                arm["global_microstep_sample_p99"] = (
                    float(arm["n_pack_p99"]) * int(arm["gpu_count"])
                    if arm["packing"]
                    else float(arm["gpu_count"])
                )
                arm["gbs_contract_passed"] = True
            by_cutoff[cutoff] = arms
        used: set[int] = set()
        for target in spec["target_fractions"]:
            candidates = []
            for cutoff, arms in by_cutoff.items():
                if cutoff in used:
                    continue
                pair_fraction = max(float(row["refit_center_capacity_fraction"]) for row in arms)
                candidates.append((abs(pair_fraction - target), cutoff, pair_fraction, arms))
            _, cutoff, pair_fraction, arms = min(candidates, key=lambda item: (item[0], item[1]))
            used.add(cutoff)
            selected.append(
                {
                    "boundary_setting_id": f"{spec['chain_id']}-c{cutoff}",
                    "chain_id": spec["chain_id"],
                    "target_capacity_fraction": target,
                    "selected_cutoff_len": cutoff,
                    "pair_refit_center_capacity_fraction": pair_fraction,
                    "selection_reason": spec["selection_reason"],
                    "execution_order_within_chain": len(used),
                    "arms": sorted(arms, key=lambda row: bool(row["packing"])),
                }
            )
    if len(selected) != 8:
        raise ValueError(f"expected eight boundary settings, got {len(selected)}")
    return selected


def select() -> dict[str, Any]:
    phase_c = read_json(PHASE_C)
    if phase_c.get("gates", {}).get("phase_c_complete_without_extra_repeats") is not True:
        raise PermissionError("Phase C is incomplete")
    rows, q95 = _scan()
    selected = _select(rows)
    capacity_gib = float(read_json(HARDWARE)["memory_bytes_reported_by_torch"]) / GIB
    selected_fractions = [
        float(row["pair_refit_center_capacity_fraction"]) for row in selected
    ]
    capacity_bin_coverage = {
        "70_to_80_percent": sum(0.70 <= value < 0.80 for value in selected_fractions),
        "80_to_90_percent": sum(0.80 <= value < 0.90 for value in selected_fractions),
        "90_to_98_percent": sum(0.90 <= value <= 0.98 for value in selected_fractions),
    }
    if any(count < 1 for count in capacity_bin_coverage.values()):
        raise ValueError(f"selected DOE misses a required capacity bin: {capacity_bin_coverage}")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "design_only_not_approved_not_queued",
        "hardware_scope": {
            "hardware_id": "local_h800_140g",
            "capacity_gib": capacity_gib,
            "h800_80g_claim_allowed": False,
        },
        "inputs": {
            path.name: _binding(path)
            for path in (MODEL_INVENTORY, MODEL_REFIT, PHASE_C, PHASE_B_QUEUE, HARDWARE)
        },
        "design": {
            "chain_specs": list(CHAIN_SPECS),
            "target_capacity_fractions_by_chain": {
                str(spec["chain_id"]): list(spec["target_fractions"])
                for spec in CHAIN_SPECS
            },
            "required_capacity_bin_coverage": capacity_bin_coverage,
            "selected_settings": len(selected),
            "treatments_per_setting": 2,
            "repeats_per_treatment": 2,
            "planned_jobs": len(selected) * 4,
            "planned_gpu_job_equivalents": sum(
                int(setting["arms"][0]["gpu_count"]) * 4 for setting in selected
            ),
            "provisional_one_sided_q95_log_residual": q95,
            "provisional_upper_is_accepted_guard": False,
            "epsilon_gbs": EPSILON_GBS,
            "all_selected_gbs_contracts_passed": all(
                bool(arm["gbs_contract_passed"])
                for setting in selected
                for arm in setting["arms"]
            ),
        },
        "selected_settings": selected,
        "candidate_scan": {
            "rows": len(rows),
            "prediction_artifact": _binding(PREDICTIONS),
            "all_rows": rows,
        },
        "execution_contract": {
            "queue_materialized": False,
            "gpu_training_started": False,
            "first_run_lower_target_per_chain": True,
            "near_boundary_requires_all_lower_target_u_p_success": True,
            "treatment_order": "counterbalanced U-P-P-U or P-U-U-P",
            "cuda_oom_role": "right_censored_lower_bound_not_regression_label",
            "software_failure_role": "repair_and_rerun_same_job_not_memory_evidence",
            "automatic_publication_allowed": False,
        },
        "gates": {
            "phase_c_complete": True,
            "memory_center_refit_complete": True,
            "memory_upper_guard_complete": False,
            "boundary_queue_approved": False,
            "automatic_gpu_launch_allowed": False,
            "automatic_packing_recommendation_allowed": False,
        },
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)

    lines = [
        "# H800 Packing 显存边界实验点选择 v1",
        "",
        "本产物只选择实验点，不生成队列、不启动 GPU，也不把 provisional upper 当作已验收显存上界。",
        "",
        f"选择 {len(selected)} 个 setting；每个 Packing U/P 各 2 次，共 {report['design']['planned_jobs']} jobs、"
        f"{report['design']['planned_gpu_job_equivalents']} GPU-job equivalents。",
        "",
        "| setting | 模型/训练 | 机制 | profile | cutoff | 目标/实际占用 | target/expected GBS | step-P99 | U/P center GiB | U/P physical P95 GiB |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in selected:
        arms = setting["arms"]
        unpacked = next(row for row in arms if not row["packing"])
        packed = next(row for row in arms if row["packing"])
        lines.append(
            f"| {setting['boundary_setting_id']} | {unpacked['model_id']}/{unpacked['training_mode']} | "
            f"G{unpacked['gpu_count']}/Z{unpacked['zero_stage']}/GC={'on' if unpacked['gc'] else 'off'} | "
            f"{unpacked['workload_id']} | {setting['selected_cutoff_len']} | "
            f"{setting['target_capacity_fraction']:.0%}/{setting['pair_refit_center_capacity_fraction']:.1%} | "
            f"{packed['target_gbs']}/{packed['expected_sample_gbs']:.1f} | "
            f"{packed['global_microstep_sample_p99']:.1f} | "
            f"{unpacked['refit_center_gib']:.1f}/{packed['refit_center_gib']:.1f} | "
            f"{unpacked['physical_operational_p95_gib']:.1f}/{packed['physical_operational_p95_gib']:.1f} |"
        )
    lines.extend(
        (
            "",
            "每条链先执行较低目标点；只有该点 U/P 全部成功，才允许执行近边界点。CUDA OOM 只作为右删失边界。",
            "",
            "B2～B4 是模型规模或 Full 训练模式的 OOD 转移点；其目的正是获取校准证据，预测中心不构成执行安全承诺。",
            "",
            "本轮仅覆盖本机 H800 140 GiB；不形成 H800 80 GiB 发布证据。",
            "",
        )
    )
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = select()
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "markdown": str(MARKDOWN),
                "status": result["status"],
                "design": result["design"],
                "selected": [
                    {
                        "setting": row["boundary_setting_id"],
                        "cutoff": row["selected_cutoff_len"],
                        "target": row["target_capacity_fraction"],
                        "center_fraction": row["pair_refit_center_capacity_fraction"],
                    }
                    for row in result["selected_settings"]
                ],
                "gates": result["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
