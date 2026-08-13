#!/usr/bin/env python3
"""Materialize the exact 36-job H800 Packing profile Phase-B batch.

This script is CPU-only.  It reconstructs the W7/W8 training snapshots from
the same frozen sources used by DataProfile v2, verifies the exact pack curves,
runs a frozen execution-only memory preflight, and writes a queue.  It never
launches a GPU job and never enables automatic Packing publication.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    DATA_DIR,
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
from prepare_packing_dataprofile_v2 import (
    OPEN_CODE,
    SAMPLE_RECORDS,
    SEED,
    TrainingEncoder,
    _curve,
    _reservoir,
)


SCHEMA = "sft_h800_packing_profile_phase_b_design/v1"
JOB_SCHEMA = "sft_h800_packing_profile_phase_b_job/v1"
CAMPAIGN_ID = "h800_packing_profile_phase_b_20260805_v1"
PHASE_ID = "h800_packing_profile_phase_b_v1"
GPU_POOL = (0, 1, 2, 3, 4, 5, 6, 7)
TWO_GPU_MASKS = ((0, 1), (2, 3), (4, 5), (6, 7))
MODEL_ID = "qwen3_8b"
MODEL_PATH = Path("/wanqing-models/Qwen3-8B")
TEMPLATE = "qwen3_nothink"
TARGET_GBS = 128
GPU_COUNT = 2
ZERO = "zero2"
WARMUP_STEPS = 2
MEASURE_STEPS = 8
UNPACKED_GA = 64
MEMORY_EXECUTION_LIMIT_FRACTION = 0.90
PACKING_HISTORY_MULTIPLIER = 1.50

CANDIDATE_DESIGN = ARTIFACT_DIR / "packing_information_gain_design_v1.json"
PROFILE_MANIFEST = ARTIFACT_DIR / "packing_data_profiles_w1_w9_manifest_v2.json"
CPU_SCREEN = ARTIFACT_DIR / "packing_cutoff_dp_gbs_screen_w1_w9_v2.json"
CANARY_RESULTS = ARTIFACT_DIR / "h800_packing_platform_v4_semantic_canary_results_v1.json"
INVENTORY = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_model_inventory_v1.json"
MEMORY_MODEL = ARTIFACT_DIR / "h800_challenger_modeling.json"
THROUGHPUT_MODEL = ARTIFACT_DIR / "joint_throughput_modeling.json"
MEMORY_ANCHORS = ARTIFACT_DIR / "h800_memory_anchor_registry_v1.json"
HISTORICAL_RESULTS = (
    ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_results_v1.json",
    ARTIFACT_DIR / "h800_packing_gbs_repair_batch_results_v1.json",
    ARTIFACT_DIR / "h800_packing_platform_v4_interactions_batch1_continuation_combined_results_v1.json",
)

DATA_BUNDLE_DIR = DATA_DIR / "packing_profile_phase_b_v1"
ADDITIONAL_PROFILE_DIR = ARTIFACT_DIR / "packing_profile_phase_b_v1/profiles"
W7_DATA = DATA_BUNDLE_DIR / "packing_w7_bimodal_v1.jsonl"
W8_DATA = DATA_BUNDLE_DIR / "packing_w8_code_structured_v1.jsonl"
W3_DATA = DATA_BUNDLE_DIR / "packing_w3_multiturn_probe_v2.jsonl"
W5_DATA = DATA_BUNDLE_DIR / "packing_w5_longcontext_probe_v2.jsonl"
W7_PROFILE = ADDITIONAL_PROFILE_DIR / "packing_w7_bimodal_v1.qwen3_nothink.jsonl"
W8_PROFILE = ADDITIONAL_PROFILE_DIR / "packing_w8_code_structured_v1.qwen3_nothink.jsonl"
W3_PROFILE = ADDITIONAL_PROFILE_DIR / "packing_w3_multiturn_probe_v2.qwen3_nothink.jsonl"
W5_PROFILE = ADDITIONAL_PROFILE_DIR / "packing_w5_longcontext_probe_v2.qwen3_nothink.jsonl"

W3_SOURCE_DATA = DATA_DIR / "derived/multiturn_4096.jsonl"
W5_SOURCE_DATA = DATA_DIR / "derived/longcontext_16384.jsonl"
W3_SOURCE_PROFILE = ARTIFACT_DIR / "dataset_profiles/multiturn_4096.qwen3_nothink.jsonl"
W5_SOURCE_PROFILE = ARTIFACT_DIR / "dataset_profiles/longcontext_16384.qwen3_nothink.jsonl"

STATIC = ARTIFACT_DIR / "h800_packing_profile_phase_b_static_v1.json"
PREDICTIONS = ARTIFACT_DIR / "h800_packing_profile_phase_b_memory_predictions_v1.json"
QUEUE = MATRIX_DIR / "h800_packing_profile_phase_b_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_packing_profile_phase_b_design_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_packing_profile_phase_b_queue_manifest_v1.json"

RUNTIME_DATASETS: dict[str, dict[str, Any]] = {
    "W3": {
        "dataset_id": "packing_w3_multiturn_probe_v2",
        "dataset_category": "multiturn",
        "data_path": W3_DATA,
        "profile_path": W3_PROFILE,
        "source_data_path": W3_SOURCE_DATA,
        "source_profile_path": W3_SOURCE_PROFILE,
        "probe_repetition_factor": 2,
    },
    "W5": {
        "dataset_id": "packing_w5_longcontext_probe_v2",
        "dataset_category": "longcontext",
        "data_path": W5_DATA,
        "profile_path": W5_PROFILE,
        "source_data_path": W5_SOURCE_DATA,
        "source_profile_path": W5_SOURCE_PROFILE,
        "probe_repetition_factor": 2,
    },
    "W7": {
        "dataset_id": "packing_w7_bimodal_v1",
        "dataset_category": "longtail",
        "data_path": W7_DATA,
        "profile_path": W7_PROFILE,
    },
    "W8": {
        "dataset_id": "packing_w8_code_structured_v1",
        # The frozen predictor has no code/structured category.  Longtail is
        # an explicit execution-only transfer assumption, not a release label.
        "dataset_category": "longtail",
        "data_path": W8_DATA,
        "profile_path": W8_PROFILE,
    },
}


def _binding(path: Path, **extra: Any) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), **extra}


def _messages(row: dict[str, Any]) -> list[dict[str, str]]:
    value = row.get("messages")
    if not isinstance(value, list) or not value:
        raise ValueError("training row has no messages")
    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in value
    ]


def _materialize_data() -> dict[str, Any]:
    short_data = DATA_DIR / "derived/short_512.jsonl"
    long_data = DATA_DIR / "derived/longcontext_16384.jsonl"
    short_profile = ARTIFACT_DIR / "dataset_profiles/short_512.qwen3_nothink.jsonl"
    long_profile = ARTIFACT_DIR / "dataset_profiles/longcontext_16384.qwen3_nothink.jsonl"
    for path in (short_data, long_data, short_profile, long_profile, OPEN_CODE):
        if not path.is_file():
            raise FileNotFoundError(path)

    # W3/W5 originally contain 1,000 rows.  With DP=2 and Unpacked GA=64,
    # ten optimizer steps require 1,280 raw examples before the first epoch
    # boundary.  Repeating the exact frozen snapshot twice keeps its empirical
    # sample distribution unchanged while preventing a partial accumulation
    # step.  Sample IDs are namespaced so runtime/profile order remains auditable.
    for workload_id, source_data, source_profile, target_data, target_profile in (
        ("W3", W3_SOURCE_DATA, W3_SOURCE_PROFILE, W3_DATA, W3_PROFILE),
        ("W5", W5_SOURCE_DATA, W5_SOURCE_PROFILE, W5_DATA, W5_PROFILE),
    ):
        for path in (source_data, source_profile):
            if not path.is_file():
                raise FileNotFoundError(path)
        source_data_rows = read_jsonl(source_data)
        source_profile_rows = read_jsonl(source_profile)
        if len(source_data_rows) != 1_000 or len(source_profile_rows) != 1_000:
            raise ValueError(f"{workload_id}: expected exact 1,000-row source snapshot")
        repeated_data: list[dict[str, Any]] = []
        repeated_profile: list[dict[str, Any]] = []
        for cycle in range(2):
            for data_row, profile_row in zip(source_data_rows, source_profile_rows):
                source_id = str(data_row.get("sample_id") or "")
                if not source_id or source_id != str(profile_row.get("sample_id") or ""):
                    raise ValueError(f"{workload_id}: source raw/profile sample order mismatch")
                probe_id = f"{source_id}:phaseb-cycle-{cycle}"
                data_copy = dict(data_row)
                profile_copy = dict(profile_row)
                data_copy.update(
                    {
                        "sample_id": probe_id,
                        "phase_b_source_sample_id": source_id,
                        "phase_b_repetition_cycle": cycle,
                    }
                )
                profile_copy.update(
                    {
                        "sample_id": probe_id,
                        "phase_b_source_sample_id": source_id,
                        "phase_b_repetition_cycle": cycle,
                    }
                )
                repeated_data.append(data_copy)
                repeated_profile.append(profile_copy)
        write_jsonl(target_data, repeated_data)
        write_jsonl(target_profile, repeated_profile)

    short_rows = read_jsonl(short_data)
    long_rows = read_jsonl(long_data)
    short_profile_rows = read_jsonl(short_profile)
    long_profile_rows = read_jsonl(long_profile)
    if not all(len(rows) >= 1_000 for rows in (short_rows, long_rows, short_profile_rows, long_profile_rows)):
        raise ValueError("W7 sources must each contain at least 1,000 rows")
    write_jsonl(W7_DATA, short_rows[:1_000] + long_rows[:1_000])
    write_jsonl(W7_PROFILE, short_profile_rows[:1_000] + long_profile_rows[:1_000])

    raw_code = _reservoir(
        OPEN_CODE,
        fields=("id", "input", "output"),
        limit=SAMPLE_RECORDS,
        seed=SEED + 8,
    )
    encoder = TrainingEncoder()
    w8_data_rows: list[dict[str, Any]] = []
    w8_profile_rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_code):
        messages = [
            {"role": "user", "content": str(row["input"])},
            {"role": "assistant", "content": str(row["output"])},
        ]
        sample_id = str(row.get("id") or f"W8:{index}")
        total_tokens, label_tokens = encoder.encode(messages)
        w8_data_rows.append(
            {
                "messages": messages,
                "sample_id": sample_id,
                "source_dataset": "nvidia/OpenCodeInstruct",
                "source_revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d",
                "source_index_in_frozen_reservoir": index,
            }
        )
        w8_profile_rows.append(
            {
                "sample_id": sample_id,
                "total_tokens": total_tokens,
                "label_tokens": label_tokens,
                "turns": len(messages),
            }
        )
    write_jsonl(W8_DATA, w8_data_rows)
    write_jsonl(W8_PROFILE, w8_profile_rows)

    rows_by_workload: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for workload_id, runtime in RUNTIME_DATASETS.items():
        data_rows = read_jsonl(Path(runtime["data_path"]))
        profile_rows = read_jsonl(Path(runtime["profile_path"]))
        if len(data_rows) != len(profile_rows):
            raise ValueError(f"{workload_id}: raw/profile row count mismatch")
        data_ids = [str(row.get("sample_id") or "") for row in data_rows]
        profile_ids = [str(row.get("sample_id") or "") for row in profile_rows]
        if not all(data_ids) or data_ids != profile_ids:
            raise ValueError(f"{workload_id}: raw/profile sample order mismatch")
        if any(not _messages(row) for row in data_rows):
            raise ValueError(f"{workload_id}: malformed conversation")
        rows_by_workload[workload_id] = (data_rows, profile_rows)

    return {
        workload_id: {
            "dataset_id": RUNTIME_DATASETS[workload_id]["dataset_id"],
            "dataset_category": RUNTIME_DATASETS[workload_id]["dataset_category"],
            "records": len(rows[0]),
            "source_records": (
                len(read_jsonl(Path(RUNTIME_DATASETS[workload_id]["source_profile_path"])))
                if RUNTIME_DATASETS[workload_id].get("source_profile_path")
                else len(rows[0])
            ),
            "probe_repetition_factor": int(
                RUNTIME_DATASETS[workload_id].get("probe_repetition_factor", 1)
            ),
            "data": _binding(Path(RUNTIME_DATASETS[workload_id]["data_path"])),
            "token_profile": _binding(Path(RUNTIME_DATASETS[workload_id]["profile_path"])),
        }
        for workload_id, rows in sorted(rows_by_workload.items())
    }


def _selected_families() -> list[dict[str, Any]]:
    design = read_json(CANDIDATE_DESIGN)
    rows = list(design.get("selected_families") or [])
    expected = {
        "w3-c4096-dp2-g128",
        "w3-c40960-dp2-g128",
        "w5-c16384-dp2-g128",
        "w7-c20480-dp2-g128",
        "w8-c2048-dp2-g128",
        "w8-c10240-dp2-g128",
    }
    if (
        design.get("schema") != "sft_packing_information_gain_design/v1"
        or design.get("status") != "candidate_design_only_not_materialized"
        or {str(row.get("candidate_id")) for row in rows} != expected
        or len(rows) != 6
    ):
        raise ValueError("information-gain candidate design drifted")
    return rows


def _verify_exact_profiles(
    selected: list[dict[str, Any]], data_bindings: dict[str, Any]
) -> dict[str, Any]:
    checks = []
    by_candidate: dict[str, dict[str, Any]] = {}
    total_probe_steps = WARMUP_STEPS + MEASURE_STEPS
    for family in selected:
        workload_id = str(family["workload_id"])
        profile_path = Path(data_bindings[workload_id]["token_profile"]["path"])
        rows = read_jsonl(profile_path)
        lengths = [int(row["total_tokens"]) for row in rows]
        cutoff_len = int(family["cutoff_len"])
        runtime_curve = _curve(lengths, [cutoff_len])[0]
        dataprofile = read_json(Path(family["profile_path"]))
        exact = [
            point
            for point in dataprofile["packing_curve"]
            if int(point["cutoff_len"]) == cutoff_len
        ]
        runtime = RUNTIME_DATASETS[workload_id]
        source_profile_path = Path(runtime.get("source_profile_path", profile_path))
        source_rows = read_jsonl(source_profile_path)
        source_lengths = [int(row["total_tokens"]) for row in source_rows]
        source_curve = _curve(source_lengths, [cutoff_len])[0]
        repetition_factor = int(runtime.get("probe_repetition_factor", 1))
        if len(exact) != 1 or source_curve != exact[0]:
            raise ValueError(f"{family['candidate_id']}: source profile/pack curve mismatch")
        if lengths != source_lengths * repetition_factor:
            raise ValueError(f"{family['candidate_id']}: probe profile is not an exact repetition")
        aggregate = dataprofile["aggregate"]
        if (
            int(aggregate["records"]) != len(source_rows)
            or not math.isclose(
                float(aggregate["length_tokens"]["mean"]),
                sum(source_lengths) / len(source_lengths),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"{family['candidate_id']}: aggregate profile mismatch")
        packed_ga = int(family["packing_contract"]["gradient_accumulation_steps"])
        required_unpacked_examples = GPU_COUNT * UNPACKED_GA * total_probe_steps
        required_packed_packs = GPU_COUNT * packed_ga * total_probe_steps
        first_epoch_capacity = {
            "total_probe_optimizer_steps": total_probe_steps,
            "unpacked": {
                "available_examples": len(rows),
                "required_examples": required_unpacked_examples,
                "passed": len(rows) >= required_unpacked_examples,
            },
            "packed": {
                "available_packs": int(runtime_curve["packs"]),
                "required_packs": required_packed_packs,
                "passed": int(runtime_curve["packs"]) >= required_packed_packs,
            },
        }
        if not all(branch["passed"] for branch in first_epoch_capacity.values() if isinstance(branch, dict)):
            raise ValueError(f"{family['candidate_id']}: probe crosses an epoch boundary")
        expected_probe_gbs = (
            GPU_COUNT
            * packed_ga
            * float(runtime_curve["samples_per_pack"]["mean"])
        )
        expected_probe_gbs_error = abs(expected_probe_gbs - TARGET_GBS) / TARGET_GBS
        if expected_probe_gbs_error > 0.05:
            raise ValueError(f"{family['candidate_id']}: repeated probe GBS exceeds 5% error")
        check = {
                "candidate_id": str(family["candidate_id"]),
                "workload_id": workload_id,
                "cutoff_len": cutoff_len,
                "source_records": len(source_rows),
                "runtime_records": len(rows),
                "probe_repetition_factor": repetition_factor,
                "source_profile": _binding(source_profile_path),
                "runtime_profile": _binding(profile_path),
                "packing_dataprofile": _binding(Path(family["profile_path"])),
                "source_exact_curve_sha256": sha256_json(source_curve),
                "source_exact_curve_match": True,
                "runtime_curve": runtime_curve,
                "runtime_curve_sha256": sha256_json(runtime_curve),
                "first_epoch_capacity": first_epoch_capacity,
                "expected_probe_window_sample_gbs": expected_probe_gbs,
                "expected_probe_window_sample_gbs_relative_error": expected_probe_gbs_error,
        }
        checks.append(check)
        by_candidate[str(family["candidate_id"])] = check
    return {
        "checks": checks,
        "by_candidate": by_candidate,
        "all_passed": len(checks) == 6,
    }


def _model_parameters() -> int:
    inventory = read_json(INVENTORY)
    models = {str(row["id"]): row for row in inventory["models"]}
    if MODEL_ID not in models:
        raise ValueError("Qwen3-8B is absent from the frozen inventory")
    return int(models[MODEL_ID]["actual_parameters"])


def _historical_packed_anchor() -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for path in HISTORICAL_RESULTS:
        report = read_json(path)
        for row in report.get("job_results") or []:
            if (
                row.get("classification") == "success"
                and row.get("packing") is True
                and int(row.get("gpu_count") or 0) == 2
                and row.get("zero") == "zero2"
                and row.get("gc") is True
                and row.get("model_id") == MODEL_ID
                and row.get("train_type") == "lora"
                and row.get("max_reserved_gib") is not None
            ):
                values.append(
                    {
                        "source": str(path.resolve()),
                        "job_id": row.get("job_id"),
                        "max_reserved_gib": float(row["max_reserved_gib"]),
                    }
                )
    if not values:
        raise ValueError("no relevant DP2/ZeRO-2/GC-on historical Packed anchor")
    maximum = max(values, key=lambda row: row["max_reserved_gib"])
    return {
        "selector": {
            "model_id": MODEL_ID,
            "train_type": "lora",
            "gpu_count": 2,
            "zero": "zero2",
            "gc": True,
            "packing": True,
        },
        "successful_observations": len(values),
        "maximum": maximum,
        "maximum_reserved_bytes": maximum["max_reserved_gib"] * 2**30,
        "execution_guard_multiplier": PACKING_HISTORY_MULTIPLIER,
        "interpretation": (
            "Empirical execution-only floor.  The 1.50x multiplier is an engineering "
            "stress buffer, not a calibrated statistical quantile."
        ),
    }


def _memory_preflight(selected: list[dict[str, Any]]) -> dict[str, Any]:
    parameters = _model_parameters()
    requests = []
    for family in selected:
        runtime = RUNTIME_DATASETS[str(family["workload_id"])]
        for packed in (False, True):
            contract = family["packing_contract"]
            requests.append(
                {
                    "request_id": f"phaseb-{family['candidate_id']}-{'p' if packed else 'u'}",
                    "comparison_group": (
                        f"phaseb-{family['candidate_id']}-{'p' if packed else 'u'}"
                    ),
                    "model_id": MODEL_ID,
                    "training_mode": "lora",
                    "dataset_id": runtime["dataset_id"],
                    "dataset_category": runtime["dataset_category"],
                    "target_gbs": TARGET_GBS,
                    "cutoff_len": int(family["cutoff_len"]),
                    "actual_parameters": parameters,
                    "gpu_count": GPU_COUNT,
                    "physical_mbs": 1,
                    "gradient_accumulation_steps": (
                        int(contract["gradient_accumulation_steps"])
                        if packed
                        else UNPACKED_GA
                    ),
                    "zero_stage": 2,
                    "gradient_checkpointing": True,
                    "packing": packed,
                    "offload": False,
                    "dtype": "bf16",
                    "kernel_path": "fa3_orig+liger_fused_ce+adamw_torch_fused",
                    "lora_rank": 32,
                    "profile_tokenizer_id": "qwen3_8b@local",
                    "profile_template_id": TEMPLATE,
                }
            )
    predictor = H800PhysicalV4BPredictor(
        model_inventory=INVENTORY,
        strict_model_inventory_binding=False,
        additional_dataset_profile_dir=ADDITIONAL_PROFILE_DIR,
    )
    predictions = predictor.predict(requests)
    write_json(PREDICTIONS, predictions)

    hardware = read_json(ROOT / "config/hardware.json")
    capacity = int(hardware["memory_bytes_reported_by_torch"])
    execution_limit = capacity * MEMORY_EXECUTION_LIMIT_FRACTION
    history = _historical_packed_anchor()
    history_floor = float(history["maximum_reserved_bytes"]) * PACKING_HISTORY_MULTIPLIER
    rows = []
    for prediction in predictions["predictions"]:
        memory = prediction["memory"]
        packed = bool(prediction["configuration"]["packing"])
        if memory.get("prediction_available") is not True:
            raise ValueError(f"memory prediction unavailable: {prediction['request_id']}")
        physical_upper = float(memory["operational_p95_reserved_bytes"])
        guarded_upper = max(physical_upper, history_floor if packed else 0.0)
        passed = bool(guarded_upper <= execution_limit)
        rows.append(
            {
                "request_id": prediction["request_id"],
                "comparison_group": prediction["comparison_group"],
                "packing": packed,
                "physical_reserved_center_bytes": float(memory["reserved_center_bytes"]),
                "physical_operational_p95_reserved_bytes": physical_upper,
                "historical_guard_floor_bytes": history_floor if packed else None,
                "execution_guarded_upper_bytes": guarded_upper,
                "execution_limit_bytes": execution_limit,
                "execution_limit_fraction": MEMORY_EXECUTION_LIMIT_FRACTION,
                "headroom_bytes": execution_limit - guarded_upper,
                "base_physical_model_admitted_at_95pct_line": memory.get(
                    "base_physical_model_admitted"
                ),
                "product_policy_admitted": memory.get("admitted"),
                "support": prediction["support"],
                "experimental_track_override": packed,
                "experimental_track_reason": (
                    "Packing is intentionally outside the frozen product admission support; "
                    "this fit-only batch supplies that missing evidence."
                    if packed
                    else None
                ),
                "execution_preflight_passed": passed,
            }
        )
    if len(rows) != 12 or not all(row["execution_preflight_passed"] for row in rows):
        raise ValueError(f"Phase-B memory execution preflight failed: {rows}")
    return {
        "status": "fit_only_execution_preflight_not_product_admission",
        "prediction_artifact": _binding(PREDICTIONS),
        "hardware_capacity_bytes": capacity,
        "execution_limit_fraction": MEMORY_EXECUTION_LIMIT_FRACTION,
        "historical_packed_anchor": history,
        "rows": rows,
        "all_passed": True,
        "automatic_packing_admission_allowed": False,
    }


def _memory_by_request(preflight: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["request_id"]): row for row in preflight["rows"]}


def prepare() -> dict[str, Any]:
    canary = read_json(CANARY_RESULTS)
    if canary.get("gates", {}).get("packing_semantics_and_ledger_passed") is not True:
        raise PermissionError("Packing semantic canary is not healthy")
    selected = _selected_families()
    data_bindings = _materialize_data()
    profile_consistency = _verify_exact_profiles(selected, data_bindings)
    if profile_consistency["all_passed"] is not True:
        raise ValueError("runtime profiles do not reproduce the exact cached curves")
    memory_preflight = _memory_preflight(selected)

    static: dict[str, Any] = {
        "schema": "sft_h800_packing_profile_phase_b_static/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_before_gpu": True,
        "recommendation_path_reads_raw_data": False,
        "recommendation_path_reads_raw_lengths": False,
        "recommendation_path_runs_full_packer": False,
        "materialization_path_reads_frozen_sources": True,
        "candidate_design": _binding(CANDIDATE_DESIGN),
        "profile_manifest": _binding(PROFILE_MANIFEST),
        "cpu_screen": _binding(CPU_SCREEN),
        "runtime_datasets": data_bindings,
        "exact_profile_consistency": profile_consistency,
        "memory_preflight": memory_preflight,
        "production_optimizer_step_gate": "not_applicable_fit_only_probe",
        "fixed_probe_optimizer_steps": WARMUP_STEPS + MEASURE_STEPS,
        "final_route_effect_claim_allowed": False,
    }
    static["report_sha256"] = sha256_json(static)
    write_json(STATIC, static)

    model_parameters = _model_parameters()
    memory_rows = _memory_by_request(memory_preflight)
    counters = {
        str(family["candidate_id"]): {False: 0, True: 0}
        for family in selected
    }
    jobs: list[dict[str, Any]] = []
    for block in range(6):
        for family in selected:
            family_id = str(family["candidate_id"])
            treatment = str(family["counterbalanced_treatment_order"][block])
            packed = treatment == "packed"
            repeat = counters[family_id][packed]
            counters[family_id][packed] += 1
            contract = family["packing_contract"]
            execution_profile = profile_consistency["by_candidate"][family_id]
            runtime_curve = execution_profile["runtime_curve"]
            ga = int(contract["gradient_accumulation_steps"]) if packed else UNPACKED_GA
            expected_gbs = (
                float(execution_profile["expected_probe_window_sample_gbs"])
                if packed
                else float(TARGET_GBS)
            )
            expected_error = (
                float(
                    execution_profile[
                        "expected_probe_window_sample_gbs_relative_error"
                    ]
                )
                if packed
                else 0.0
            )
            runtime_contract = json.loads(json.dumps(contract))
            runtime_contract["execution_snapshot"] = {
                "source_records": int(execution_profile["source_records"]),
                "runtime_records": int(execution_profile["runtime_records"]),
                "probe_repetition_factor": int(
                    execution_profile["probe_repetition_factor"]
                ),
                "packs": int(runtime_curve["packs"]),
                "samples_per_pack": runtime_curve["samples_per_pack"],
                "expected_probe_window_sample_gbs": float(expected_gbs),
                "expected_probe_window_sample_gbs_relative_error": float(
                    expected_error
                ),
                "first_epoch_capacity": execution_profile[
                    "first_epoch_capacity"
                ],
            }
            workload_id = str(family["workload_id"])
            runtime = RUNTIME_DATASETS[workload_id]
            memory_request_id = f"phaseb-{family_id}-{'p' if packed else 'u'}"
            memory = memory_rows[memory_request_id]
            pair_id = stable_id(
                "h800packphasebpair",
                {"campaign_id": CAMPAIGN_ID, "family_id": family_id, "repeat": repeat},
            )
            row: dict[str, Any] = {
                "schema": JOB_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "phase_id": PHASE_ID,
                "candidate_role": "packing_profile_matched_mechanism_effect_fit_only",
                "family_id": family_id,
                "setting_id": family_id,
                "workload_id": workload_id,
                "profile_family_id": str(family["profile_role"]),
                "display_name": f"{workload_id}/{family['profile_role']}/cutoff{family['cutoff_len']}",
                "scenario_id": f"phaseb-{family_id}-qwen3_8b-lora",
                "interaction_axis": "packing_mechanism_x_profile_cutoff_at_mbs1",
                "packing_pair_id": pair_id,
                "packing_treatment": treatment,
                "counterbalance_block": block,
                "arm_id": "P-C-1-gP" if packed else "N-C-1-gN",
                "repeat": repeat,
                "model_id": MODEL_ID,
                "model_family": "qwen3",
                "model_path": str(MODEL_PATH),
                "tokenizer_path": str(MODEL_PATH),
                "template": TEMPLATE,
                "model_parameters": model_parameters,
                "train_type": "lora",
                "dataset_id": runtime["dataset_id"],
                "profile_id": family["profile_id"],
                "dataset_category": runtime["dataset_category"],
                "source_dataset_records": int(
                    data_bindings[workload_id]["source_records"]
                ),
                "frozen_slice_records": int(data_bindings[workload_id]["records"]),
                "probe_repetition_factor": int(
                    data_bindings[workload_id]["probe_repetition_factor"]
                ),
                "data_path": str(Path(runtime["data_path"]).resolve()),
                "data_sha256": sha256_file(Path(runtime["data_path"])),
                "dataset_profile_path": str(Path(runtime["profile_path"]).resolve()),
                "dataset_profile_sha256": sha256_file(Path(runtime["profile_path"])),
                "packing_dataprofile_path": str(Path(family["profile_path"]).resolve()),
                "packing_dataprofile_sha256": sha256_file(Path(family["profile_path"])),
                "cutoff_label": "information_gain_exact_cache",
                "base_cutoff_len": int(family["cutoff_len"]),
                "cutoff_scale": 1,
                "cutoff_len": int(family["cutoff_len"]),
                "target_gbs": TARGET_GBS,
                "gpu_count": GPU_COUNT,
                "zero": ZERO,
                "zero_stage": 2,
                "gc": True,
                "gradient_checkpointing": True,
                "mbs": 1,
                "gradient_accumulation_steps": ga,
                "packing": packed,
                "expected_sample_gbs": expected_gbs,
                "expected_epoch_sample_gbs": (
                    float(contract["expected_epoch_sample_gbs"])
                    if packed
                    else float(TARGET_GBS)
                ),
                "expected_probe_window_sample_gbs": expected_gbs,
                "expected_sample_gbs_relative_error": expected_error,
                "n_pack_mean": family["exact_cached_curve"]["samples_per_pack"]["mean"],
                "execution_snapshot_n_pack_mean": runtime_curve[
                    "samples_per_pack"
                ]["mean"],
                "n_pack_step_p99": family["exact_cached_curve"]["samples_per_pack"]["p99"],
                "gbs_contract_v2": runtime_contract,
                "offload": False,
                "gpu_type": "NVIDIA H800 140GB HBM3",
                "required_runtime_gpu_name": "NVIDIA H800",
                "hardware_id": "local_h800_140g",
                "kind": "throughput",
                "warmup_steps": WARMUP_STEPS,
                "measure_steps": MEASURE_STEPS,
                "max_samples": int(data_bindings[workload_id]["records"]),
                "fidelity": "formal_profile_mechanism_effect_2plus8",
                "parallel_class": "gpu_partitionable",
                "requires_external_node_idle": False,
                "execution_sequence_index": len(jobs),
                "calibration_partition": {
                    "role": "calibration",
                    "split_unit_id": family_id,
                    "policy": "profile_mechanism_effect_fit_only_never_acceptance_v1",
                },
                "declared_model_manifest_path": str(INVENTORY.resolve()),
                "declared_model_manifest_sha256": sha256_file(INVENTORY),
                "candidate_design_path": str(CANDIDATE_DESIGN.resolve()),
                "candidate_design_sha256": sha256_file(CANDIDATE_DESIGN),
                "static_features_path": str(STATIC.resolve()),
                "static_features_sha256": sha256_file(STATIC),
                "memory_preflight_request_id": memory_request_id,
                "memory_execution_guarded_upper_bytes": memory["execution_guarded_upper_bytes"],
                "memory_execution_limit_bytes": memory["execution_limit_bytes"],
                "matched_mechanism_pair": True,
                "final_route_effect_claim_allowed": False,
                "production_optimizer_step_gate": "not_applicable_fit_only_probe",
                "fixed_probe_optimizer_steps": WARMUP_STEPS + MEASURE_STEPS,
                "first_epoch_capacity": execution_profile[
                    "first_epoch_capacity"
                ],
                "publication_allowed": False,
            }
            row["job_id"] = stable_id("h800packphaseb", row)
            jobs.append(row)

    if len(jobs) != 36 or len({str(row["job_id"]) for row in jobs}) != 36:
        raise ValueError("Phase-B batch must contain 36 unique jobs")
    for family in selected:
        family_id = str(family["candidate_id"])
        subset = [row for row in jobs if row["family_id"] == family_id]
        counts = Counter(bool(row["packing"]) for row in subset)
        repeats = Counter((bool(row["packing"]), int(row["repeat"])) for row in subset)
        if len(subset) != 6 or counts != {False: 3, True: 3} or any(value != 1 for value in repeats.values()):
            raise ValueError(f"{family_id}: unbalanced U/P repeats")
    write_jsonl(QUEUE, jobs)
    job_dir = ARTIFACT_DIR / "h800_packing_profile_phase_b_jobs_v1"
    for job in jobs:
        write_json(job_dir / f"{job['job_id']}.json", job)

    source_files = {
        "execution_plan": ROOT.parent / "项目文档/04_Packing与参数搜索/Packing后续补充实验执行计划_2026-08-05.md",
        "joint_modeling_plan": ROOT.parent / "项目文档/04_Packing与参数搜索/Neat_Packing联合搜索实验与建模计划_2026-08-04.md",
        "packing_decision": ROOT.parent / "项目文档/04_Packing与参数搜索/Packing决策逻辑V2_修订版_2026-08-04.md",
        "candidate_design": CANDIDATE_DESIGN,
        "profile_manifest": PROFILE_MANIFEST,
        "cpu_screen": CPU_SCREEN,
        "canary_results": CANARY_RESULTS,
        "experiment_config": ROOT / "config/experiment.json",
        "hardware_config": ROOT / "config/hardware.json",
        "dataset_registry": DATA_DIR / "dataset_info.json",
        "model_inventory": INVENTORY,
        "memory_model": MEMORY_MODEL,
        "throughput_model": THROUGHPUT_MODEL,
        "memory_anchor_registry": MEMORY_ANCHORS,
        "memory_predictions": PREDICTIONS,
        "static_features": STATIC,
        "preparer": Path(__file__).resolve(),
        "freezer": ROOT / "scripts/freeze_h800_packing_profile_phase_b_v1.py",
        "evaluator": ROOT / "scripts/evaluate_h800_packing_profile_phase_b_v1.py",
        "run_job": ROOT / "scripts/run_job.py",
        "scheduler": ROOT / "scripts/scheduler.py",
        "train_entry": ROOT / "scripts/train_entry.py",
        "metrics_callback": ROOT / "scripts/metrics_callback.py",
    }
    for workload_id, runtime in RUNTIME_DATASETS.items():
        source_files[f"data_{workload_id.lower()}"] = Path(runtime["data_path"])
        source_files[f"runtime_profile_{workload_id.lower()}"] = Path(runtime["profile_path"])
        if runtime.get("source_data_path"):
            source_files[f"source_data_{workload_id.lower()}"] = Path(
                runtime["source_data_path"]
            )
        if runtime.get("source_profile_path"):
            source_files[f"source_profile_{workload_id.lower()}"] = Path(
                runtime["source_profile_path"]
            )
    for index, path in enumerate(HISTORICAL_RESULTS, start=1):
        source_files[f"historical_packing_results_{index}"] = path
    for family in selected:
        source_files[f"dataprofile_{str(family['candidate_id']).replace('-', '_')}"] = Path(family["profile_path"])

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
        "objective": (
            "Estimate the MBS=1 matched Packing mechanism effect across W3/W5/W7/W8; "
            "do not interpret this batch as best-branch route effect."
        ),
        "required_gpu_pool": {
            "gpu_ids": list(GPU_POOL),
            "max_gpu_count_per_job": 2,
            "maximum_parallel_gpu_slots": len(GPU_POOL),
            "maximum_parallel_jobs": 4,
            "preview_two_gpu_masks_when_all_idle": [list(mask) for mask in TWO_GPU_MASKS],
            "two_gpu_mask_policy": "any_disjoint_pair_within_fully_nvlinked_approved_pool",
            "excluded_busy_gpu_ids": [],
            "preemption_allowed": False,
            "join_busy_pool": True,
        },
        "queue": {
            **_binding(QUEUE),
            "job_count": 36,
            "gpu_job_equivalents": 72,
            "ordered_job_ids": [str(row["job_id"]) for row in jobs],
        },
        "matrix": {
            "families": [
                {
                    "family_id": family["candidate_id"],
                    "workload_id": family["workload_id"],
                    "profile_role": family["profile_role"],
                    "cutoff_len": family["cutoff_len"],
                    "packed_gradient_accumulation_steps": family["packing_contract"]["gradient_accumulation_steps"],
                    "unpacked_gradient_accumulation_steps": UNPACKED_GA,
                    "source_records": profile_consistency["by_candidate"][str(family["candidate_id"])]["source_records"],
                    "runtime_records": profile_consistency["by_candidate"][str(family["candidate_id"])]["runtime_records"],
                    "first_epoch_capacity": profile_consistency["by_candidate"][str(family["candidate_id"])]["first_epoch_capacity"],
                    "expected_probe_window_sample_gbs": profile_consistency["by_candidate"][str(family["candidate_id"])]["expected_probe_window_sample_gbs"],
                    "step_p99_times_dp": family["packing_contract"]["global_microstep_sample_gbs"]["p99"],
                    "treatment_order": family["counterbalanced_treatment_order"],
                    "treatment_repeats": 3,
                }
                for family in selected
            ],
            "jobs": 36,
            "gpu_job_equivalents": 72,
        },
        "measurement_contract": {
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "token_source": "consumed_token_ledger/v1",
            "expected_epoch_gbs_separate_from_probe_window_gbs": True,
            "packed_semantics_required_on_every_rank": True,
            "authoritative_ledger_required_on_every_rank": True,
            "all_measured_steps_must_fit_before_first_epoch_boundary": True,
            "first_epoch_capacity_formula": (
                "records_or_packs >= DP * gradient_accumulation_steps * "
                "(warmup_steps + measure_steps)"
            ),
            "global_metrics_sum_across_ranks": True,
            "counterbalance": "U-P-P-U-U-P_or_mirror",
            "production_optimizer_step_gate": "not_applicable_fit_only_probe",
        },
        "inference_contract": {
            "estimand": "matched_packing_mechanism_effect_at_mbs1",
            "best_branch_route_effect_claim_allowed": False,
            "reason": "Neither branch is independently tuned over MBS/cutoff in this batch.",
        },
        "static_features": _binding(STATIC),
        "memory_predictions": _binding(PREDICTIONS),
        "model_inventory": _binding(INVENTORY),
        "source_bindings": {
            name: _binding(path) for name, path in sorted(source_files.items())
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    manifest: dict[str, Any] = {
        "schema": "sft_h800_packing_profile_phase_b_queue_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "design": _binding(DESIGN),
        "queue": {
            **_binding(QUEUE),
            "job_count": 36,
            "ordered_job_ids": [str(row["job_id"]) for row in jobs],
        },
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(QUEUE_MANIFEST, manifest)
    return {
        "design": _binding(DESIGN),
        "queue": _binding(QUEUE),
        "jobs": 36,
        "gpu_job_equivalents": 72,
        "gpu_pool": list(GPU_POOL),
        "maximum_parallel_jobs": 4,
        "memory_preflight_max_guarded_gib": max(
            float(row["execution_guarded_upper_bytes"]) for row in memory_preflight["rows"]
        )
        / 2**30,
    }


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
