#!/usr/bin/env python3
"""Prepare matched text/image probes for frozen-vision VL decomposition.

This is a CPU-only materializer.  It creates text controls, immutable workload
profiles, a six-job semantic canary, and a 45-job formal queue.  It never
promotes an approval and never starts GPU work.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    DATA_DIR,
    MATRIX_DIR,
    percentile,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from prepare_h800_qwen35_vl_supplement_v1 import (
    MEDIA_MANIFEST,
    MODEL_SPECS,
    RUNTIME_CONTRACT,
    VL_DATA,
)
from prepare_h800_vl_calibration_profiles_v1 import TrainingEncoder

CAMPAIGN_ID = "h800_frozen_vl_decomposition_20260811_v1"
CANARY_PHASE_ID = "h800_frozen_vl_decomposition_canary_v1"
FORMAL_PHASE_ID = "h800_frozen_vl_decomposition_formal_v1"
JOB_SCHEMA = "sft_h800_frozen_vl_decomposition_job/v1"
DESIGN_SCHEMA = "sft_h800_frozen_vl_decomposition_design/v1"
TARGET_GBS = 64
CUTOFF_LEN = 8192
ROWS = 1000

INVENTORY = ARTIFACT_DIR / "h800_qwen35_vl_supplement_model_inventory_v1.json"
PROCESSOR_PROFILES = (
    ARTIFACT_DIR / "h800_qwen35_vl_supplement_processor_profiles_manifest_v1.json"
)
DATA_OUTPUT_DIR = DATA_DIR / "frozen_vl_decomposition_v1"
DATASET_REGISTRY_DIR = DATA_OUTPUT_DIR / "registry"
DATASET_REGISTRY = DATASET_REGISTRY_DIR / "dataset_info.json"
PROFILE_OUTPUT_DIR = ARTIFACT_DIR / "h800_frozen_vl_decomposition_profiles_v1"
PROFILE_MANIFEST = ARTIFACT_DIR / "h800_frozen_vl_decomposition_profiles_manifest_v1.json"
CANARY_QUEUE = MATRIX_DIR / "h800_frozen_vl_decomposition_canary_v1.jsonl"
FORMAL_QUEUE = MATRIX_DIR / "h800_frozen_vl_decomposition_formal_v1.jsonl"
CANARY_DESIGN = ARTIFACT_DIR / "h800_frozen_vl_decomposition_canary_design_v1.json"
FORMAL_DESIGN = ARTIFACT_DIR / "h800_frozen_vl_decomposition_formal_design_v1.json"

REPRESENTATIVES = (
    "qwen2p5_vl_3b",
    "qwen3_vl_4b",
    "qwen3p5_4b",
)

MECHANISMS: dict[str, dict[str, Any]] = {
    # SAFE vs NOGC isolates gradient checkpointing at MBS=1.
    "SAFE": {"gpu_count": 1, "zero": "none", "gc": True, "mbs": 1},
    "NOGC": {"gpu_count": 1, "zero": "none", "gc": False, "mbs": 1},
    # NOGC vs PRESSURE isolates MBS at fixed GC-off.
    "PRESSURE": {"gpu_count": 1, "zero": "none", "gc": False, "mbs": 4},
}


def _statistics(values: list[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("profile statistics require non-empty values")
    mean = sum(values) / len(values)
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": mean,
        "mean_squared": sum(value * value for value in values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "maximum": max(values),
    }


def _model_specs() -> dict[str, dict[str, Any]]:
    specs = {str(row["id"]): dict(row) for row in MODEL_SPECS}
    missing = sorted(set(REPRESENTATIVES) - set(specs))
    if missing:
        raise ValueError(f"representative model specs missing: {missing}")
    return {model_id: specs[model_id] for model_id in REPRESENTATIVES}


def _inventory_rows() -> dict[str, dict[str, Any]]:
    report = read_json(INVENTORY)
    rows = {str(row["id"]): row for row in report.get("models") or []}
    missing = sorted(set(REPRESENTATIVES) - set(rows))
    if missing:
        raise ValueError(f"representative inventory rows missing: {missing}")
    return {model_id: rows[model_id] for model_id in REPRESENTATIVES}


def _processor_profiles() -> dict[tuple[str, str], dict[str, Any]]:
    report = read_json(PROCESSOR_PROFILES)
    if (
        report.get("all_actual_processor_checks_passed") is not True
        or report.get("all_alias_equivalence_checks_passed") is not True
    ):
        raise ValueError("processor-bound VL profiles are not accepted")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in report.get("profiles") or []:
        model_id = str(row["model_id"])
        tier = str(row["tier"])
        if model_id not in REPRESENTATIVES:
            continue
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"processor profile drifted: {path}")
        rows[(model_id, tier)] = row
    expected = {(model_id, tier) for model_id in REPRESENTATIVES for tier in ("low", "high")}
    if set(rows) != expected:
        raise ValueError("representative low/high processor profile coverage is incomplete")
    return rows


def _remove_image_placeholders(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = copy.deepcopy(messages)
    for message in output:
        content = str(message.get("content") or "")
        message["content"] = content.replace("<image>", "")
    if not any(message.get("role") == "user" for message in output):
        raise ValueError("matched text row has no user message")
    return output


def _append_exact_filler(
    encoder: TrainingEncoder,
    messages: list[dict[str, Any]],
    *,
    target_total_tokens: int,
) -> tuple[list[dict[str, Any]], int, int]:
    base = copy.deepcopy(messages)
    base_total, base_labels = encoder.encode(base)
    if target_total_tokens < base_total:
        raise ValueError(
            f"target total {target_total_tokens} is below text base {base_total}"
        )
    user_index = max(
        index for index, message in enumerate(base) if message.get("role") == "user"
    )
    original = str(base[user_index].get("content") or "")
    filler_tokens = target_total_tokens - base_total
    for _ in range(8):
        candidate = copy.deepcopy(base)
        candidate[user_index]["content"] = original + (" x" * filler_tokens)
        total, labels = encoder.encode(candidate)
        if labels != base_labels:
            raise ValueError("user-side filler changed assistant label token count")
        if total == target_total_tokens:
            return candidate, total, labels
        filler_tokens += target_total_tokens - total
        if filler_tokens < 0:
            raise ValueError("exact text filler search crossed below zero")
    raise ValueError(
        f"failed to construct exact matched text length {target_total_tokens}"
    )


def _write_text_profile(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    encoder: TrainingEncoder,
    arm_id: str,
    target_tokens: list[int] | None,
) -> dict[str, Any]:
    profile_rows: list[dict[str, Any]] = []
    totals: list[int] = []
    labels: list[int] = []
    for index, row in enumerate(rows):
        total, label = encoder.encode(row["messages"])
        totals.append(total)
        labels.append(label)
        profile_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "total_tokens": total,
                "label_tokens": label,
                "turns": len(row["messages"]),
                "assistant_turns": sum(
                    int(message.get("role") == "assistant")
                    for message in row["messages"]
                ),
                "arm_id": arm_id,
            }
        )
        if target_tokens is not None and total != target_tokens[index]:
            raise ValueError(
                f"row {index} matched total drifted: {total} != {target_tokens[index]}"
            )
    write_jsonl(path, profile_rows)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(profile_rows),
        "total_tokens": _statistics(totals),
        "label_tokens": _statistics(labels),
        "recordwise_total_match": target_tokens is None or totals == target_tokens,
    }


def _materialize_text_controls(
    specs: dict[str, dict[str, Any]],
    processor_profiles: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    source_rows = read_jsonl(VL_DATA)
    if len(source_rows) != ROWS:
        raise ValueError(f"expected {ROWS} VL source rows, got {len(source_rows)}")
    outputs: dict[tuple[str, str], dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for model_id, spec in specs.items():
        encoder = TrainingEncoder(spec["path"], spec["template"])
        base_rows = [
            {
                "sample_id": f"{model_id}:text_base:{index}",
                "messages": _remove_image_placeholders(row["messages"]),
            }
            for index, row in enumerate(source_rows)
        ]
        base_data = DATA_OUTPUT_DIR / f"{model_id}.text_base.jsonl"
        write_jsonl(base_data, base_rows)
        base_profile = PROFILE_OUTPUT_DIR / f"{model_id}.text_base.jsonl"
        base_binding = _write_text_profile(
            path=base_profile,
            rows=base_rows,
            encoder=encoder,
            arm_id="text_base",
            target_tokens=None,
        )
        base_output = {
            "model_id": model_id,
            "arm_id": "text_base",
            "tier": None,
            "data_path": str(base_data.resolve()),
            "data_sha256": sha256_file(base_data),
            "profile": base_binding,
        }
        outputs[(model_id, "text_base")] = base_output
        manifest_rows.append(base_output)

        for tier in ("low", "high"):
            vl_binding = processor_profiles[(model_id, tier)]
            vl_profile = read_json(Path(vl_binding["path"]))
            target_tokens = [int(row["total_tokens"]) for row in vl_profile["records"]]
            if len(target_tokens) != ROWS:
                raise ValueError(f"{model_id}/{tier} target profile row count drifted")
            matched_rows = []
            for index, (source, target) in enumerate(zip(source_rows, target_tokens)):
                messages = _remove_image_placeholders(source["messages"])
                matched, observed_total, _ = _append_exact_filler(
                    encoder,
                    messages,
                    target_total_tokens=target,
                )
                if observed_total != target:
                    raise AssertionError("exact filler helper returned a mismatched total")
                matched_rows.append(
                    {
                        "sample_id": f"{model_id}:text_matched:{tier}:{index}",
                        "messages": matched,
                    }
                )
            data_path = DATA_OUTPUT_DIR / f"{model_id}.{tier}.text_matched.jsonl"
            write_jsonl(data_path, matched_rows)
            profile_path = PROFILE_OUTPUT_DIR / f"{model_id}.{tier}.text_matched.jsonl"
            profile_binding = _write_text_profile(
                path=profile_path,
                rows=matched_rows,
                encoder=encoder,
                arm_id="text_length_matched",
                target_tokens=target_tokens,
            )
            output = {
                "model_id": model_id,
                "arm_id": "text_length_matched",
                "tier": tier,
                "data_path": str(data_path.resolve()),
                "data_sha256": sha256_file(data_path),
                "profile": profile_binding,
                "matched_vl_profile_path": str(Path(vl_binding["path"]).resolve()),
                "matched_vl_profile_sha256": vl_binding["sha256"],
                "recordwise_total_match": profile_binding["recordwise_total_match"],
            }
            outputs[(model_id, f"text_length_matched:{tier}")] = output
            manifest_rows.append(output)

    report: dict[str, Any] = {
        "schema": "sft_h800_frozen_vl_decomposition_profiles/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_vl_data": {
            "path": str(VL_DATA.resolve()),
            "sha256": sha256_file(VL_DATA),
            "rows": len(source_rows),
        },
        "profiles": manifest_rows,
        "all_recordwise_total_matches_exact": all(
            row.get("recordwise_total_match", True) is True for row in manifest_rows
        ),
        "raw_images_copied": False,
        "gpu_training_started": False,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(PROFILE_MANIFEST, report)
    return outputs


def _runtime_overlay() -> dict[str, Any]:
    contract = read_json(RUNTIME_CONTRACT)
    return {
        "PYTHONPATH_prepend": list(contract["environment"]["PYTHONPATH_prepend"]),
        "variables": dict(contract["environment"]["variables"]),
        "contract_path": str(RUNTIME_CONTRACT.resolve()),
        "contract_sha256": sha256_file(RUNTIME_CONTRACT),
    }


def _sharegpt_entry(*, data_path: Path, images: bool) -> dict[str, Any]:
    columns = {"messages": "messages"}
    if images:
        columns["images"] = "images"
    return {
        "file_name": str(data_path.resolve()),
        "formatting": "sharegpt",
        "columns": columns,
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }


def _materialize_dataset_registry(
    controls: dict[tuple[str, str], dict[str, Any]],
) -> None:
    registry: dict[str, Any] = {
        "vl_pzfj38_calibration_v1": _sharegpt_entry(
            data_path=VL_DATA, images=True
        )
    }
    for model_id in REPRESENTATIVES:
        for key in ("text_base", "text_length_matched:low", "text_length_matched:high"):
            binding = controls[(model_id, key)]
            dataset_id = (
                f"frozen_vl_decomposition_{model_id}_{key.replace(':', '_')}"
            )
            registry[dataset_id] = _sharegpt_entry(
                data_path=Path(binding["data_path"]), images=False
            )
    write_json(DATASET_REGISTRY, registry)


def _base_job(
    *,
    model: dict[str, Any],
    inventory: dict[str, Any],
    phase_id: str,
    arm_id: str,
    tier: str | None,
    mechanism_id: str,
    repeat: int,
    warmup_steps: int,
    measure_steps: int,
    max_samples: int,
) -> dict[str, Any]:
    mechanism = MECHANISMS[mechanism_id]
    denominator = int(mechanism["gpu_count"]) * int(mechanism["mbs"])
    if TARGET_GBS % denominator:
        raise ValueError("target GBS is not divisible by gpu_count * MBS")
    family = str(model["family"])
    row: dict[str, Any] = {
        "schema": JOB_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": phase_id,
        "track": "frozen_vision_decomposition",
        "evidence_role": (
            "semantic_canary" if phase_id == CANARY_PHASE_ID else "paired_fit_diagnostic"
        ),
        "arm_id": arm_id,
        "media_tier": tier,
        "model_id": model["id"],
        "model_family": family,
        "model_path": model["path"],
        "tokenizer_path": model["tokenizer_path"],
        "model_parameters": int(inventory["actual_parameters"]),
        "template": model["template"],
        "train_type": "lora",
        "train_scope_id": "language_only",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "freeze_language_model": False,
        "target_gbs": TARGET_GBS,
        "cutoff_len": CUTOFF_LEN,
        "gpu_count": int(mechanism["gpu_count"]),
        "zero": str(mechanism["zero"]),
        "zero_stage": 0,
        "gc": bool(mechanism["gc"]),
        "gradient_checkpointing": bool(mechanism["gc"]),
        "mbs": int(mechanism["mbs"]),
        "gradient_accumulation_steps": TARGET_GBS // denominator,
        "packing": False,
        "offload": False,
        "kind": "throughput",
        "mechanism_id": mechanism_id,
        "repeat": repeat,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "max_samples": max_samples,
        "fidelity": f"{warmup_steps}plus{measure_steps}",
        "enable_vision_phase_memory_probe": True,
        "enable_liger_kernel": bool(model["enable_liger_kernel"]),
        "effective_kernel_path": (
            "fa3_orig+liger_fused_ce+adamw_torch_fused"
            if model["enable_liger_kernel"]
            else "fa3_orig+native_ce+adamw_torch_fused"
        ),
        "gpu_type": "NVIDIA H800 140GB HBM3",
        "required_runtime_gpu_name": "NVIDIA H800",
        "hardware_id": "local_h800_140g",
        "parallel_class": "gpu_partitionable",
        "requires_external_node_idle": False,
        "declared_model_manifest_path": str(INVENTORY.resolve()),
        "declared_model_manifest_sha256": sha256_file(INVENTORY),
        "dataset_dir": str(DATASET_REGISTRY_DIR.resolve()),
        "dataset_registry_sha256": sha256_file(DATASET_REGISTRY),
    }
    if family == "qwen3_5":
        row["environment_overlay"] = _runtime_overlay()
    return row


def _text_job(
    *,
    model: dict[str, Any],
    inventory: dict[str, Any],
    controls: dict[tuple[str, str], dict[str, Any]],
    phase_id: str,
    arm_id: str,
    tier: str | None,
    mechanism_id: str,
    repeat: int = 0,
    warmup_steps: int = 2,
    measure_steps: int = 8,
    max_samples: int = ROWS,
) -> dict[str, Any]:
    key = "text_base" if arm_id == "text_base" else f"text_length_matched:{tier}"
    binding = controls[(str(model["id"]), key)]
    row = _base_job(
        model=model,
        inventory=inventory,
        phase_id=phase_id,
        arm_id=arm_id,
        tier=tier,
        mechanism_id=mechanism_id,
        repeat=repeat,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        max_samples=max_samples,
    )
    row.update(
        {
            "dataset_id": f"frozen_vl_decomposition_{model['id']}_{key.replace(':', '_')}",
            "dataset_category": "controlled_text_base" if arm_id == "text_base" else "controlled_text_length_matched",
            "data_path": binding["data_path"],
            "data_sha256": binding["data_sha256"],
            "dataset_profile_path": binding["profile"]["path"],
            "dataset_profile_sha256": binding["profile"]["sha256"],
            "expected_images_per_sample": 0,
            "visual_runtime_evidence_required": False,
        }
    )
    row["scenario_id"] = (
        f"{model['id']}__{tier or 'base'}__{mechanism_id}__repeat{repeat}"
    )
    row["pair_id"] = row["scenario_id"]
    row["job_id"] = stable_id("h800vlfix", row)
    return row


def _image_job(
    *,
    model: dict[str, Any],
    inventory: dict[str, Any],
    processor_profiles: dict[tuple[str, str], dict[str, Any]],
    phase_id: str,
    tier: str,
    mechanism_id: str,
    repeat: int = 0,
    warmup_steps: int = 2,
    measure_steps: int = 8,
    max_samples: int = ROWS,
) -> dict[str, Any]:
    binding = processor_profiles[(str(model["id"]), tier)]
    row = _base_job(
        model=model,
        inventory=inventory,
        phase_id=phase_id,
        arm_id="real_image",
        tier=tier,
        mechanism_id=mechanism_id,
        repeat=repeat,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        max_samples=max_samples,
    )
    row.update(
        {
            "dataset_id": "vl_pzfj38_calibration_v1",
            "dataset_category": "vl_full_reference_image_quality",
            "data_path": str(VL_DATA.resolve()),
            "data_sha256": sha256_file(VL_DATA),
            "dataset_profile_path": str(Path(binding["path"]).resolve()),
            "dataset_profile_sha256": binding["sha256"],
            "media_manifest_path": str(MEDIA_MANIFEST.resolve()),
            "media_manifest_sha256": sha256_file(MEDIA_MANIFEST),
            "image_min_pixels": int(binding["image_min_pixels"]),
            "image_max_pixels": int(binding["image_max_pixels"]),
            "expected_images_per_sample": 2,
            "visual_runtime_evidence_required": True,
        }
    )
    row["scenario_id"] = f"{model['id']}__{tier}__{mechanism_id}__repeat{repeat}"
    row["pair_id"] = row["scenario_id"]
    row["job_id"] = stable_id("h800vlfix", row)
    return row


def _canary_jobs(
    specs: dict[str, dict[str, Any]],
    inventory: dict[str, dict[str, Any]],
    controls: dict[tuple[str, str], dict[str, Any]],
    processor_profiles: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for model_id in REPRESENTATIVES:
        model = specs[model_id]
        jobs.append(
            _text_job(
                model=model,
                inventory=inventory[model_id],
                controls=controls,
                phase_id=CANARY_PHASE_ID,
                arm_id="text_length_matched",
                tier="high",
                mechanism_id="SAFE",
                warmup_steps=0,
                measure_steps=2,
                max_samples=64,
            )
        )
        jobs.append(
            _image_job(
                model=model,
                inventory=inventory[model_id],
                processor_profiles=processor_profiles,
                phase_id=CANARY_PHASE_ID,
                tier="high",
                mechanism_id="SAFE",
                warmup_steps=0,
                measure_steps=2,
                max_samples=64,
            )
        )
    if len(jobs) != 6 or len({job["job_id"] for job in jobs}) != 6:
        raise ValueError("semantic canary must contain six unique jobs")
    return jobs


def _formal_jobs(
    specs: dict[str, dict[str, Any]],
    inventory: dict[str, dict[str, Any]],
    controls: dict[tuple[str, str], dict[str, Any]],
    processor_profiles: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for model_id in REPRESENTATIVES:
        model = specs[model_id]
        for mechanism_id in ("SAFE", "NOGC", "PRESSURE"):
            jobs.append(
                _text_job(
                    model=model,
                    inventory=inventory[model_id],
                    controls=controls,
                    phase_id=FORMAL_PHASE_ID,
                    arm_id="text_base",
                    tier=None,
                    mechanism_id=mechanism_id,
                )
            )
        for tier, mechanism_ids in (
            ("low", ("SAFE", "PRESSURE")),
            ("high", ("SAFE", "NOGC", "PRESSURE")),
        ):
            for mechanism_id in mechanism_ids:
                jobs.append(
                    _text_job(
                        model=model,
                        inventory=inventory[model_id],
                        controls=controls,
                        phase_id=FORMAL_PHASE_ID,
                        arm_id="text_length_matched",
                        tier=tier,
                        mechanism_id=mechanism_id,
                    )
                )
                jobs.append(
                    _image_job(
                        model=model,
                        inventory=inventory[model_id],
                        processor_profiles=processor_profiles,
                        phase_id=FORMAL_PHASE_ID,
                        tier=tier,
                        mechanism_id=mechanism_id,
                    )
                )
        for arm_id in ("text_length_matched", "real_image"):
            if arm_id == "text_length_matched":
                job = _text_job(
                    model=model,
                    inventory=inventory[model_id],
                    controls=controls,
                    phase_id=FORMAL_PHASE_ID,
                    arm_id=arm_id,
                    tier="high",
                    mechanism_id="SAFE",
                    repeat=1,
                )
            else:
                job = _image_job(
                    model=model,
                    inventory=inventory[model_id],
                    processor_profiles=processor_profiles,
                    phase_id=FORMAL_PHASE_ID,
                    tier="high",
                    mechanism_id="SAFE",
                    repeat=1,
                )
            jobs.append(job)
    if len(jobs) != 45 or len({job["job_id"] for job in jobs}) != 45:
        raise ValueError(f"formal decomposition queue must contain 45 jobs, got {len(jobs)}")
    return jobs


def _design(*, phase_id: str, jobs: list[dict[str, Any]], queue: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": phase_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue": {"path": str(queue.resolve()), "sha256": sha256_file(queue)},
        "jobs": len(jobs),
        "by_model": dict(sorted(Counter(str(row["model_id"]) for row in jobs).items())),
        "by_arm": dict(sorted(Counter(str(row["arm_id"]) for row in jobs).items())),
        "by_mechanism": dict(
            sorted(Counter(str(row["mechanism_id"]) for row in jobs).items())
        ),
        "fixed_semantics": {
            "train_type": "lora",
            "train_scope_id": "language_only",
            "freeze_vision_tower": True,
            "freeze_multi_modal_projector": True,
            "target_gbs": TARGET_GBS,
            "cutoff_len": CUTOFF_LEN,
            "packing": False,
            "offload": False,
            "dtype": "bf16",
        },
        "hypotheses": {
            "memory": (
                "At matched post-processor total sequence lengths, the real-image minus "
                "text-control residual is a bounded family/grid/mechanism correction; "
                "visual and full-step peaks are compared as phase maxima, never summed."
            ),
            "throughput": (
                "Real-image step time equals a text-like mixed-sequence component plus "
                "a family/grid-dependent frozen-vision forward component."
            ),
            "packing": "Outside product V1 and fixed false in every arm.",
        },
        "pairing": {
            "text_base": "original prompt with image placeholders removed",
            "text_length_matched": "no media; total token count matched record-by-record to real image",
            "real_image": "same source task with two decoded images",
        },
        "decision_metrics": {
            "data_quality": [
                "recordwise total-token equality for text_length_matched vs real_image",
                "real image/pixel/grid evidence on every real-image rank",
                "vision phase observed only on real-image measured steps",
                "repeat throughput CV <= 5%",
            ],
            "memory": [
                "max allocated/reserved bytes per rank and per phase",
                "OOM mismatch inside each paired arm",
                "family/tier/mechanism real-image-to-matched-text peak ratio",
                "leave-family-out correction MAPE target <= 10% as an engineering gate",
            ],
            "throughput": [
                "effective and computed tokens/s",
                "real-image minus matched-text step-time residual",
                "mechanism pairwise order accuracy target >= 90%",
                "top-1 regret target < 10%",
            ],
        },
        "usage": "fit_and_hypothesis_diagnostic_only_never_prospective_acceptance",
        "gpu_training_started": False,
    }
    report["report_sha256"] = sha256_json(report)
    return report


def prepare() -> dict[str, Any]:
    if not VL_DATA.is_file() or not MEDIA_MANIFEST.is_file():
        raise FileNotFoundError("frozen real-image data or media manifest is missing")
    specs = _model_specs()
    inventory = _inventory_rows()
    processor_profiles = _processor_profiles()
    controls = _materialize_text_controls(specs, processor_profiles)
    _materialize_dataset_registry(controls)
    canary = _canary_jobs(specs, inventory, controls, processor_profiles)
    formal = _formal_jobs(specs, inventory, controls, processor_profiles)
    write_jsonl(CANARY_QUEUE, canary)
    write_jsonl(FORMAL_QUEUE, formal)
    canary_design = _design(phase_id=CANARY_PHASE_ID, jobs=canary, queue=CANARY_QUEUE)
    formal_design = _design(phase_id=FORMAL_PHASE_ID, jobs=formal, queue=FORMAL_QUEUE)
    write_json(CANARY_DESIGN, canary_design)
    write_json(FORMAL_DESIGN, formal_design)
    return {
        "campaign_id": CAMPAIGN_ID,
        "profile_manifest": str(PROFILE_MANIFEST.resolve()),
        "canary_queue": str(CANARY_QUEUE.resolve()),
        "canary_jobs": len(canary),
        "formal_queue": str(FORMAL_QUEUE.resolve()),
        "formal_jobs": len(formal),
        "gpu_training_started": False,
    }


def main() -> None:
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
