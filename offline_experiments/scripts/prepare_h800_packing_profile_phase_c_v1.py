#!/usr/bin/env python3
"""Materialize the safe 24-job H800 Packing interaction Phase-C batch.

Phase C estimates two ratio-of-ratios interactions against the completed
Phase-B ZeRO-2/GC-on baseline.  The originally proposed long-cutoff GC-off
arms are retained as explicit failed preflight candidates and never enter the
GPU queue.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

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
from h800_physical_v4b_predictor import H800PhysicalV4BPredictor


SCHEMA = "sft_h800_packing_profile_phase_c_design/v1"
JOB_SCHEMA = "sft_h800_packing_profile_phase_c_job/v1"
CAMPAIGN_ID = "h800_packing_profile_phase_c_20260805_v1"
PHASE_ID = "h800_packing_profile_phase_c_v1"
GPU_POOL = tuple(range(8))
TWO_GPU_MASKS = ((0, 1), (2, 3), (4, 5), (6, 7))
TARGET_GBS = 128
MEMORY_LIMIT_FRACTION = 0.90
BASELINE_OBSERVED_MULTIPLIER = 1.50

BASELINE_QUEUE = MATRIX_DIR / "h800_packing_profile_phase_b_v1.jsonl"
BASELINE_RESULTS = ARTIFACT_DIR / "h800_packing_profile_phase_b_results_v1.json"
MODEL_SELECTION = ARTIFACT_DIR / "h800_packing_phase_b_model_selection_v1.json"
INVENTORY = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_model_inventory_v1.json"
PROFILE_DIR = ARTIFACT_DIR / "packing_profile_phase_b_v1/profiles"
QUEUE = MATRIX_DIR / "h800_packing_profile_phase_c_v1.jsonl"
STATIC = ARTIFACT_DIR / "h800_packing_profile_phase_c_static_v1.json"
PREDICTIONS = ARTIFACT_DIR / "h800_packing_profile_phase_c_memory_predictions_v1.json"
DESIGN = ARTIFACT_DIR / "h800_packing_profile_phase_c_design_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_profile_phase_c_queue_manifest_v1.json"


@dataclass(frozen=True)
class Setting:
    setting_id: str
    baseline_family_id: str
    interaction_axis: str
    zero: str
    gc: bool
    sequence: tuple[bool, ...]
    selection_reason: str


UPPUUP = (False, True, True, False, False, True)
PUUPPU = (True, False, False, True, True, False)
SETTINGS = (
    Setting(
        "w7-c20480-z3-gcon",
        "w7-c20480-dp2-g128",
        "packing_x_zero",
        "zero3",
        True,
        UPPUUP,
        "W7 is the high-P99/bimodal profile and has high Phase-B OOF residual.",
    ),
    Setting(
        "w3-c40960-z3-gcon",
        "w3-c40960-dp2-g128",
        "packing_x_zero",
        "zero3",
        True,
        PUUPPU,
        "W3@40960 is the Phase-B top-1 cutoff and stresses high n_pack_mean.",
    ),
    Setting(
        "w3-c4096-z2-gcoff",
        "w3-c4096-dp2-g128",
        "packing_x_gc",
        "zero2",
        False,
        UPPUUP,
        "Natural W3 fallback with an existing Phase-B baseline and safe GC-off P95.",
    ),
    Setting(
        "w8-c10240-z2-gcoff",
        "w8-c10240-dp2-g128",
        "packing_x_gc",
        "zero2",
        False,
        PUUPPU,
        "Structured W8 fallback maximizes safe cutoff diversity under the 90% limit.",
    ),
)

EXCLUDED_GC_OFF = (
    ("w7-c20480-dp2-g128", "W7 long-cutoff GC-off"),
    ("w3-c40960-dp2-g128", "W3 long-cutoff GC-off"),
)


def _binding(path: Path, **extra: Any) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), **extra}


def _baseline() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(BASELINE_QUEUE)
    report = read_json(BASELINE_RESULTS)
    if len(rows) != 36 or report.get("gates", {}).get(
        "phase_b_complete_without_extra_repeats"
    ) is not True:
        raise PermissionError("the exact completed Phase-B baseline is unavailable")
    expected = {setting.baseline_family_id for setting in SETTINGS}
    if not expected.issubset({str(row["family_id"]) for row in rows}):
        raise ValueError("a selected Phase-C family is absent from Phase B")
    return rows, report


def _representatives(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for family_id in {setting.baseline_family_id for setting in SETTINGS} | {
        family_id for family_id, _ in EXCLUDED_GC_OFF
    }:
        subset = [row for row in rows if row["family_id"] == family_id]
        if len(subset) != 6:
            raise ValueError(f"{family_id}: Phase-B baseline is not six jobs")
        selected[family_id] = next(row for row in subset if row["packing"] is False)
    return selected


def _predict_request(
    request_id: str,
    base: dict[str, Any],
    *,
    zero: str,
    gc: bool,
    packing: bool,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "comparison_group": request_id,
        "model_id": base["model_id"],
        "training_mode": base["train_type"],
        "dataset_id": base["dataset_id"],
        "dataset_category": base["dataset_category"],
        "target_gbs": base["target_gbs"],
        "cutoff_len": base["cutoff_len"],
        "actual_parameters": base["model_parameters"],
        "gpu_count": base["gpu_count"],
        "physical_mbs": base["mbs"],
        "gradient_accumulation_steps": (
            next(
                row["gradient_accumulation_steps"]
                for row in read_jsonl(BASELINE_QUEUE)
                if row["family_id"] == base["family_id"]
                and bool(row["packing"]) is packing
            )
        ),
        "zero_stage": int(zero[-1]),
        "gradient_checkpointing": gc,
        "packing": packing,
        "offload": False,
        "dtype": "bf16",
        "kernel_path": "fa3_orig+liger_fused_ce+adamw_torch_fused",
        "lora_rank": 32,
        "profile_tokenizer_id": "qwen3_8b@local",
        "profile_template_id": base["template"],
    }


def _baseline_memory(report: dict[str, Any]) -> dict[tuple[str, bool], float]:
    values: dict[tuple[str, bool], float] = {}
    for family in report["families"]:
        for packing, arm in ((False, "unpacked"), (True, "packed")):
            value = family["arms"][arm]["max_reserved_gib_mean"]
            if value is None:
                raise ValueError(f"missing Phase-B memory anchor: {family['family_id']}")
            values[(str(family["family_id"]), packing)] = float(value) * 2**30
    return values


def _memory_preflight(
    base_rows: list[dict[str, Any]], baseline_report: dict[str, Any]
) -> dict[str, Any]:
    representatives = _representatives(base_rows)
    requests: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    request_family: dict[str, str] = {}
    for setting in SETTINGS:
        base = representatives[setting.baseline_family_id]
        for packing in (False, True):
            request_id = f"phasec-{setting.setting_id}-{'p' if packing else 'u'}"
            selected_ids.add(request_id)
            request_family[request_id] = setting.baseline_family_id
            requests.append(
                _predict_request(
                    request_id,
                    base,
                    zero=setting.zero,
                    gc=setting.gc,
                    packing=packing,
                )
            )
    for family_id, _ in EXCLUDED_GC_OFF:
        base = representatives[family_id]
        for packing in (False, True):
            request_id = f"phasec-excluded-{family_id}-gcoff-{'p' if packing else 'u'}"
            request_family[request_id] = family_id
            requests.append(
                _predict_request(
                    request_id,
                    base,
                    zero="zero2",
                    gc=False,
                    packing=packing,
                )
            )
    predictor = H800PhysicalV4BPredictor(
        model_inventory=INVENTORY,
        strict_model_inventory_binding=False,
        additional_dataset_profile_dir=PROFILE_DIR,
    )
    predictions = predictor.predict(requests)
    write_json(PREDICTIONS, predictions)
    capacity = int(read_json(ROOT / "config/hardware.json")["memory_bytes_reported_by_torch"])
    limit = capacity * MEMORY_LIMIT_FRACTION
    anchors = _baseline_memory(baseline_report)
    rows: list[dict[str, Any]] = []
    for prediction in predictions["predictions"]:
        request_id = str(prediction["request_id"])
        configuration = prediction["configuration"]
        memory = prediction["memory"]
        if memory.get("prediction_available") is not True:
            raise ValueError(f"memory prediction unavailable: {request_id}")
        family_id = request_family[request_id]
        packing = bool(configuration["packing"])
        physical_p95 = float(memory["operational_p95_reserved_bytes"])
        baseline_guard = anchors[(family_id, packing)] * BASELINE_OBSERVED_MULTIPLIER
        guarded = max(physical_p95, baseline_guard)
        selected = request_id in selected_ids
        rows.append(
            {
                "request_id": request_id,
                "family_id": family_id,
                "packing": packing,
                "selected_for_queue": selected,
                "physical_reserved_center_bytes": float(memory["reserved_center_bytes"]),
                "physical_operational_p95_reserved_bytes": physical_p95,
                "baseline_observed_guard_bytes": baseline_guard,
                "execution_guarded_upper_bytes": guarded,
                "execution_limit_bytes": limit,
                "headroom_bytes": limit - guarded,
                "execution_preflight_passed": guarded <= limit,
                "support": prediction["support"],
            }
        )
    selected_rows = [row for row in rows if row["selected_for_queue"]]
    excluded_rows = [row for row in rows if not row["selected_for_queue"]]
    if len(selected_rows) != 8 or not all(
        row["execution_preflight_passed"] for row in selected_rows
    ):
        raise ValueError(f"selected Phase-C memory preflight failed: {selected_rows}")
    if len(excluded_rows) != 4 or not all(
        not row["execution_preflight_passed"] for row in excluded_rows
    ):
        raise ValueError("long-cutoff GC-off exclusions no longer match the frozen gate")
    return {
        "status": "selected_arms_pass_and_long_cutoff_gc_off_fail",
        "prediction_artifact": _binding(PREDICTIONS),
        "hardware_capacity_bytes": capacity,
        "execution_limit_fraction": MEMORY_LIMIT_FRACTION,
        "baseline_observed_guard_multiplier": BASELINE_OBSERVED_MULTIPLIER,
        "rows": rows,
        "selected_all_passed": True,
        "excluded_all_failed": True,
        "automatic_packing_admission_allowed": False,
    }


def prepare() -> dict[str, Any]:
    base_rows, baseline_report = _baseline()
    memory_preflight = _memory_preflight(base_rows, baseline_report)
    static: dict[str, Any] = {
        "schema": "sft_h800_packing_profile_phase_c_static/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_before_gpu": True,
        "recommendation_path_reads_raw_data": False,
        "recommendation_path_runs_full_packer": False,
        "baseline": _binding(BASELINE_RESULTS),
        "model_selection": _binding(MODEL_SELECTION),
        "memory_preflight": memory_preflight,
        "estimand": "ratio_of_packing_ratios_against_phase_b_baseline",
        "publication_allowed": False,
    }
    static["report_sha256"] = sha256_json(static)
    write_json(STATIC, static)
    memory_by_id = {
        str(row["request_id"]): row
        for row in memory_preflight["rows"]
        if row["selected_for_queue"]
    }
    jobs: list[dict[str, Any]] = []
    counters = {setting.setting_id: {False: 0, True: 0} for setting in SETTINGS}
    for block in range(6):
        for setting in SETTINGS:
            packing = setting.sequence[block]
            repeat = counters[setting.setting_id][packing]
            counters[setting.setting_id][packing] += 1
            source = next(
                row
                for row in base_rows
                if row["family_id"] == setting.baseline_family_id
                and bool(row["packing"]) is packing
                and int(row["repeat"]) == repeat
            )
            row = json.loads(json.dumps(source))
            request_id = f"phasec-{setting.setting_id}-{'p' if packing else 'u'}"
            memory = memory_by_id[request_id]
            row.update(
                {
                    "schema": JOB_SCHEMA,
                    "campaign_id": CAMPAIGN_ID,
                    "phase_id": PHASE_ID,
                    "candidate_role": "packing_profile_interaction_fit_only",
                    "setting_id": setting.setting_id,
                    "family_id": setting.setting_id,
                    "baseline_family_id": setting.baseline_family_id,
                    "baseline_phase_b_job_id": source["job_id"],
                    "baseline_phase_b_results_path": str(BASELINE_RESULTS.resolve()),
                    "baseline_phase_b_results_sha256": sha256_file(BASELINE_RESULTS),
                    "display_name": f"{source['workload_id']}/{setting.interaction_axis}/{setting.zero}/gc-{setting.gc}",
                    "scenario_id": f"phasec-{setting.setting_id}-qwen3_8b-lora",
                    "interaction_axis": setting.interaction_axis,
                    "interaction_reference": "phase_b_zero2_gc_on_same_family",
                    "selection_reason": setting.selection_reason,
                    "packing_pair_id": stable_id(
                        "h800packphasecpair",
                        {
                            "campaign_id": CAMPAIGN_ID,
                            "setting_id": setting.setting_id,
                            "repeat": repeat,
                        },
                    ),
                    "packing_treatment": "packed" if packing else "unpacked",
                    "counterbalance_block": block,
                    "repeat": repeat,
                    "zero": setting.zero,
                    "zero_stage": int(setting.zero[-1]),
                    "gc": setting.gc,
                    "gradient_checkpointing": setting.gc,
                    "memory_preflight_request_id": request_id,
                    "memory_execution_guarded_upper_bytes": memory["execution_guarded_upper_bytes"],
                    "memory_execution_limit_bytes": memory["execution_limit_bytes"],
                    "static_features_path": str(STATIC.resolve()),
                    "static_features_sha256": sha256_file(STATIC),
                    "execution_sequence_index": len(jobs),
                    "calibration_partition": {
                        "role": "calibration",
                        "split_unit_id": setting.setting_id,
                        "policy": "phase_c_interaction_fit_only_never_acceptance_v1",
                    },
                    "matched_interaction_pair": True,
                    "final_route_effect_claim_allowed": False,
                    "publication_allowed": False,
                }
            )
            row.pop("job_id", None)
            row["job_id"] = stable_id("h800packphasec", row)
            jobs.append(row)
    if len(jobs) != 24 or len({row["job_id"] for row in jobs}) != 24:
        raise ValueError("Phase-C batch must contain 24 unique jobs")
    for setting in SETTINGS:
        subset = [row for row in jobs if row["setting_id"] == setting.setting_id]
        repeats = Counter((bool(row["packing"]), int(row["repeat"])) for row in subset)
        if len(subset) != 6 or set(repeats.values()) != {1} or len(repeats) != 6:
            raise ValueError(f"{setting.setting_id}: unbalanced U/P repeats")
    write_jsonl(QUEUE, jobs)
    jobs_dir = ARTIFACT_DIR / "h800_packing_profile_phase_c_jobs_v1"
    for job in jobs:
        write_json(jobs_dir / f"{job['job_id']}.json", job)

    source_files = {
        "execution_plan": ROOT.parent / "项目文档/04_Packing与参数搜索/Packing后续补充实验执行计划_2026-08-05.md",
        "joint_modeling_plan": ROOT.parent / "项目文档/04_Packing与参数搜索/Neat_Packing联合搜索实验与建模计划_2026-08-04.md",
        "experiment_config": ROOT / "config/experiment.json",
        "hardware_config": ROOT / "config/hardware.json",
        "dataset_registry": ROOT / "data/dataset_info.json",
        "baseline_queue": BASELINE_QUEUE,
        "baseline_results": BASELINE_RESULTS,
        "model_selection": MODEL_SELECTION,
        "model_inventory": INVENTORY,
        "memory_predictions": PREDICTIONS,
        "preparer": Path(__file__).resolve(),
        "freezer": ROOT / "scripts/freeze_h800_packing_profile_phase_c_v1.py",
        "evaluator": ROOT / "scripts/evaluate_h800_packing_profile_phase_c_v1.py",
        "run_job": ROOT / "scripts/run_job.py",
        "scheduler": ROOT / "scripts/scheduler.py",
        "train_entry": ROOT / "scripts/train_entry.py",
        "metrics_callback": ROOT / "scripts/metrics_callback.py",
    }
    for setting in SETTINGS:
        representative = next(row for row in jobs if row["setting_id"] == setting.setting_id)
        source_files[f"data_{setting.setting_id}"] = Path(representative["data_path"])
        source_files[f"profile_{setting.setting_id}"] = Path(representative["dataset_profile_path"])
        source_files[f"dataprofile_{setting.setting_id}"] = Path(representative["packing_dataprofile_path"])
    design: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_gpu_waiting_for_exact_approval",
        "gpu_training_started": False,
        "fit_allowed_after_complete_results": True,
        "automatic_next_batch_allowed": False,
        "publication_allowed": False,
        "objective": "Estimate Packing×ZeRO and Packing×GC ratio-of-ratios against exact Phase-B baselines.",
        "safety_amendment": {
            "original_long_cutoff_gc_off_admitted": False,
            "reason": "Frozen operational-P95 memory preflight exceeds 90% of 140-GiB H800 capacity.",
            "replacement_policy": "use the highest-diversity Phase-B families that pass the same frozen gate",
            "excluded_candidates": [label for _, label in EXCLUDED_GC_OFF],
            "gc_interaction_long_cutoff_generalization_allowed": False,
        },
        "required_gpu_pool": {
            "gpu_ids": list(GPU_POOL),
            "max_gpu_count_per_job": 2,
            "maximum_parallel_gpu_slots": len(GPU_POOL),
            "maximum_parallel_jobs": 4,
            "preview_two_gpu_masks_when_all_idle": [list(mask) for mask in TWO_GPU_MASKS],
            "two_gpu_mask_policy": "any_disjoint_pair_within_fully_nvlinked_approved_pool",
            "preemption_allowed": False,
            "join_busy_pool": True,
        },
        "queue": {
            **_binding(QUEUE),
            "job_count": 24,
            "gpu_job_equivalents": 48,
            "ordered_job_ids": [row["job_id"] for row in jobs],
        },
        "matrix": {
            "settings": [
                {
                    "setting_id": setting.setting_id,
                    "baseline_family_id": setting.baseline_family_id,
                    "interaction_axis": setting.interaction_axis,
                    "zero": setting.zero,
                    "gc": setting.gc,
                    "sequence": ["P" if value else "U" for value in setting.sequence],
                    "treatment_repeats": 3,
                    "selection_reason": setting.selection_reason,
                }
                for setting in SETTINGS
            ],
            "jobs": 24,
            "gpu_job_equivalents": 48,
        },
        "measurement_contract": {
            "warmup_steps": 2,
            "measure_steps": 8,
            "token_source": "consumed_token_ledger/v1",
            "packed_semantics_required_on_every_rank": True,
            "authoritative_ledger_required_on_every_rank": True,
            "all_measured_steps_must_fit_before_first_epoch_boundary": True,
            "counterbalance": "U-P-P-U-U-P_or_mirror",
            "stop_on_semantic_or_ledger_failure_before_next_batch": True,
        },
        "inference_contract": {
            "estimand": "ratio_of_packing_ratios_against_phase_b_zero2_gc_on",
            "best_branch_route_effect_claim_allowed": False,
            "gc_interaction_at_long_cutoff_claim_allowed": False,
        },
        "static_features": _binding(STATIC),
        "memory_predictions": _binding(PREDICTIONS),
        "model_inventory": _binding(INVENTORY),
        "source_bindings": {name: _binding(path) for name, path in sorted(source_files.items())},
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    manifest: dict[str, Any] = {
        "schema": "sft_h800_packing_profile_phase_c_queue_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "design": _binding(DESIGN),
        "queue": {
            **_binding(QUEUE),
            "job_count": 24,
            "ordered_job_ids": [row["job_id"] for row in jobs],
        },
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(QUEUE_MANIFEST, manifest)
    return {
        "design": _binding(DESIGN),
        "queue": _binding(QUEUE),
        "jobs": 24,
        "gpu_job_equivalents": 48,
        "selected_preflight_max_guarded_gib": max(
            row["execution_guarded_upper_bytes"]
            for row in memory_preflight["rows"]
            if row["selected_for_queue"]
        ) / 2**30,
        "excluded_preflight_min_guarded_gib": min(
            row["execution_guarded_upper_bytes"]
            for row in memory_preflight["rows"]
            if not row["selected_for_queue"]
        ) / 2**30,
    }


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
