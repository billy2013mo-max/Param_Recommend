#!/usr/bin/env python3
"""Prepare matched text/video probes for frozen-vision VL decomposition.

This CPU-only materializer creates a training-valid video view, exact
post-processor length-matched text controls, a campaign-local dataset registry,
and canary/formal queues.  It never starts GPU work or creates an approval.
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
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from prepare_h800_frozen_vl_decomposition_v1 import (
    INVENTORY,
    MECHANISMS,
    REPRESENTATIVES,
    _append_exact_filler,
    _runtime_overlay,
    _statistics,
)
from prepare_h800_qwen35_vl_supplement_v1 import MODEL_SPECS
from prepare_vl_business_workload_profiles_v2 import (
    TrainingEncoderV2,
    VIDEO_CALIBRATION_RESPONSE,
    VIDEO_DATA,
    VIDEO_TIERS,
)


CAMPAIGN_ID = "h800_frozen_video_decomposition_20260811_v1"
CANARY_PHASE_ID = "h800_frozen_video_decomposition_canary_v1"
FORMAL_PHASE_ID = "h800_frozen_video_decomposition_formal_v1"
JOB_SCHEMA = "sft_h800_frozen_video_decomposition_job/v1"
DESIGN_SCHEMA = "sft_h800_frozen_video_decomposition_design/v1"
TARGET_GBS = 64
CANARY_GBS = 8
ROWS = 100

WORKLOAD_MANIFEST = ARTIFACT_DIR / "h800_vl_business_workload_profiles_manifest_v2.json"
MEDIA_MANIFEST = ARTIFACT_DIR / "h800_vl_media_business_download_manifest_v1.json"
DATA_OUTPUT_DIR = DATA_DIR / "frozen_video_decomposition_v1"
VIDEO_TRAIN_DATA = DATA_OUTPUT_DIR / "zltbjg_v2_videos_trainable.jsonl"
DATASET_REGISTRY_DIR = DATA_OUTPUT_DIR / "registry"
DATASET_REGISTRY = DATASET_REGISTRY_DIR / "dataset_info.json"
PROFILE_OUTPUT_DIR = ARTIFACT_DIR / "h800_frozen_video_decomposition_profiles_v1"
PROFILE_MANIFEST = ARTIFACT_DIR / "h800_frozen_video_decomposition_profiles_manifest_v1.json"
CANARY_QUEUE = MATRIX_DIR / "h800_frozen_video_decomposition_canary_v1.jsonl"
FORMAL_QUEUE = MATRIX_DIR / "h800_frozen_video_decomposition_formal_v1.jsonl"
CANARY_DESIGN = ARTIFACT_DIR / "h800_frozen_video_decomposition_canary_design_v1.json"
FORMAL_DESIGN = ARTIFACT_DIR / "h800_frozen_video_decomposition_formal_design_v1.json"

TIER_IDS = tuple(str(row["id"]) for row in VIDEO_TIERS)
TIER_SETTINGS = {str(row["id"]): dict(row) for row in VIDEO_TIERS}


def _model_specs() -> dict[str, dict[str, Any]]:
    specs = {str(row["id"]): dict(row) for row in MODEL_SPECS}
    missing = sorted(set(REPRESENTATIVES) - set(specs))
    if missing:
        raise ValueError(f"representative model specs missing: {missing}")
    return {model_id: specs[model_id] for model_id in REPRESENTATIVES}


def _inventory_rows() -> dict[str, dict[str, Any]]:
    rows = {str(row["id"]): row for row in read_json(INVENTORY)["models"]}
    missing = sorted(set(REPRESENTATIVES) - set(rows))
    if missing:
        raise ValueError(f"representative inventory rows missing: {missing}")
    return {model_id: rows[model_id] for model_id in REPRESENTATIVES}


def _workload_profiles() -> dict[tuple[str, str], dict[str, Any]]:
    manifest = read_json(WORKLOAD_MANIFEST)
    if manifest.get("schema") != "sft_h800_vl_business_workload_profiles/v2":
        raise ValueError("video workload manifest schema drifted")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in manifest.get("profiles") or []:
        if row.get("source") != "zltbjg_video":
            continue
        key = (str(row["model_id"]), str(row["tier"]))
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"video workload profile drifted: {path}")
        profile = read_json(path)
        binding = profile.get("source_binding") or {}
        if (
            binding.get("data_sha256") != sha256_file(VIDEO_DATA)
            or binding.get("media_manifest_sha256") != sha256_file(MEDIA_MANIFEST)
        ):
            raise ValueError(f"video workload source binding drifted: {path}")
        if len(profile.get("records") or []) != ROWS:
            raise ValueError(f"video profile must contain {ROWS} records: {path}")
        rows[key] = row
    expected = {
        (model_id, tier) for model_id in REPRESENTATIVES for tier in TIER_IDS
    }
    if set(rows) != expected:
        raise ValueError("video workload profile coverage is incomplete")
    return rows


def _materialize_trainable_video_data() -> list[dict[str, Any]]:
    source_rows = read_jsonl(VIDEO_DATA)
    if len(source_rows) != ROWS:
        raise ValueError(f"expected {ROWS} video rows, got {len(source_rows)}")
    outputs = []
    for index, source in enumerate(source_rows):
        prompt = str(source.get("prompt") or "")
        videos = source.get("videos") or []
        if prompt.count("<video>") != 1 or len(videos) != 1:
            raise ValueError(f"video row {index} violates one-placeholder/one-video contract")
        video_path = Path(str(videos[0]))
        if not video_path.is_file():
            raise FileNotFoundError(f"video row {index} media is missing: {video_path}")
        outputs.append(
            {
                "sample_id": str(source.get("sample_id") or index),
                # Keep the real-video arm on the same ShareGPT conversion path
                # as the already validated image and text-control arms.  The
                # previous Alpaca-only video view was the sole format outlier
                # in the Qwen3.5 runtime that stopped at trainer step zero.
                "messages": [
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": (
                            str(source.get("response") or "").strip()
                            or VIDEO_CALIBRATION_RESPONSE
                        ),
                    },
                ],
                "videos": [str(video_path.resolve())],
            }
        )
    write_jsonl(VIDEO_TRAIN_DATA, outputs)
    return outputs


def _text_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = copy.deepcopy(row["messages"])
    if len(messages) != 2:
        raise ValueError("text control requires one user and one assistant message")
    if [message.get("role") for message in messages] != ["user", "assistant"]:
        raise ValueError("text control message roles drifted")
    prompt = str(messages[0].get("content") or "")
    if prompt.count("<video>") != 1:
        raise ValueError("text control requires exactly one video placeholder")
    messages[0]["content"] = prompt.replace("<video>", "")
    return messages


def _write_profile(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    encoder: TrainingEncoderV2,
    targets: list[int],
    model_id: str,
    tier: str,
) -> dict[str, Any]:
    totals: list[int] = []
    labels: list[int] = []
    records = []
    for index, row in enumerate(rows):
        total, label = encoder.encode(row["messages"])
        if total != targets[index]:
            raise ValueError(
                f"{model_id}/{tier} text match drifted at row {index}: {total} != {targets[index]}"
            )
        totals.append(total)
        labels.append(label)
        records.append(
            {
                "sample_id": str(row["sample_id"]),
                "total_tokens": total,
                "label_tokens": label,
                "arm_id": "text_length_matched",
                "matched_video_tier": tier,
            }
        )
    write_jsonl(path, records)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(records),
        "total_tokens": _statistics(totals),
        "label_tokens": _statistics(labels),
        "recordwise_total_match": totals == targets,
    }


def _materialize_text_controls(
    specs: dict[str, dict[str, Any]],
    workloads: dict[tuple[str, str], dict[str, Any]],
    video_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    outputs: dict[tuple[str, str], dict[str, Any]] = {}
    manifest_rows = []
    for model_id, spec in specs.items():
        encoder = TrainingEncoderV2(spec["path"], spec["template"])
        base_messages = [_text_messages(row) for row in video_rows]
        for tier in TIER_IDS:
            workload_binding = workloads[(model_id, tier)]
            workload = read_json(Path(workload_binding["path"]))
            targets = [int(row["total_tokens"]) for row in workload["records"]]
            matched_rows = []
            for index, (messages, target) in enumerate(zip(base_messages, targets)):
                matched, observed, _ = _append_exact_filler(
                    encoder, messages, target_total_tokens=target
                )
                if observed != target:
                    raise AssertionError("exact filler returned a mismatched token count")
                matched_rows.append(
                    {
                        "sample_id": f"{model_id}:{tier}:{index}",
                        "messages": matched,
                    }
                )
            data_path = DATA_OUTPUT_DIR / f"{model_id}.{tier}.text_matched.jsonl"
            write_jsonl(data_path, matched_rows)
            profile_path = PROFILE_OUTPUT_DIR / f"{model_id}.{tier}.text_matched.jsonl"
            profile = _write_profile(
                path=profile_path,
                rows=matched_rows,
                encoder=encoder,
                targets=targets,
                model_id=model_id,
                tier=tier,
            )
            output = {
                "model_id": model_id,
                "tier": tier,
                "data_path": str(data_path.resolve()),
                "data_sha256": sha256_file(data_path),
                "profile": profile,
                "matched_video_profile_path": str(Path(workload_binding["path"]).resolve()),
                "matched_video_profile_sha256": workload_binding["sha256"],
                "recordwise_total_match": profile["recordwise_total_match"],
            }
            outputs[(model_id, tier)] = output
            manifest_rows.append(output)

    report: dict[str, Any] = {
        "schema": "sft_h800_frozen_video_decomposition_profiles/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_video_data": {
            "path": str(VIDEO_DATA.resolve()),
            "sha256": sha256_file(VIDEO_DATA),
            "trainable_view_path": str(VIDEO_TRAIN_DATA.resolve()),
            "trainable_view_sha256": sha256_file(VIDEO_TRAIN_DATA),
            "empty_responses_replaced_with_calibration_sentinel": True,
            "calibration_sentinel": VIDEO_CALIBRATION_RESPONSE,
            "all_training_responses_nonempty": all(
                bool(str(row["messages"][1].get("content") or "").strip())
                for row in video_rows
            ),
            "training_view_format": "sharegpt",
            "semantic_labels_fabricated": False,
        },
        "profiles": manifest_rows,
        "all_recordwise_total_matches_exact": all(
            row["recordwise_total_match"] is True for row in manifest_rows
        ),
        "gpu_training_started": False,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(PROFILE_MANIFEST, report)
    return outputs


def _text_registry_entry(path: Path) -> dict[str, Any]:
    return {
        "file_name": str(path.resolve()),
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }


def _materialize_registry(controls: dict[tuple[str, str], dict[str, Any]]) -> None:
    registry: dict[str, Any] = {
        "frozen_video_decomposition_zltbjg_v2": {
            "file_name": str(VIDEO_TRAIN_DATA.resolve()),
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "videos": "videos"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    for (model_id, tier), binding in controls.items():
        registry[f"frozen_video_decomposition_{model_id}_{tier}_text_matched"] = (
            _text_registry_entry(Path(binding["data_path"]))
        )
    write_json(DATASET_REGISTRY, registry)


def _base_job(
    *,
    model: dict[str, Any],
    inventory: dict[str, Any],
    phase_id: str,
    arm_id: str,
    tier: str,
    mechanism_id: str,
    repeat: int,
    warmup_steps: int,
    measure_steps: int,
    max_samples: int,
    target_gbs: int,
) -> dict[str, Any]:
    mechanism = MECHANISMS[mechanism_id]
    denominator = int(mechanism["gpu_count"]) * int(mechanism["mbs"])
    if target_gbs % denominator:
        raise ValueError("target GBS is not divisible by gpu_count * MBS")
    cutoff_len = 16384 if tier == "upscaled_64" else 8192
    row: dict[str, Any] = {
        "schema": JOB_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": phase_id,
        "track": "frozen_video_decomposition",
        "evidence_role": "semantic_canary" if phase_id == CANARY_PHASE_ID else "paired_fit_diagnostic",
        "arm_id": arm_id,
        "media_tier": tier,
        "model_id": model["id"],
        "model_family": model["family"],
        "model_path": model["path"],
        "tokenizer_path": model["tokenizer_path"],
        "model_parameters": int(inventory["actual_parameters"]),
        "template": model["template"],
        "train_type": "lora",
        "train_scope_id": "language_only",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "freeze_language_model": False,
        "target_gbs": target_gbs,
        "cutoff_len": cutoff_len,
        "gpu_count": int(mechanism["gpu_count"]),
        "zero": str(mechanism["zero"]),
        "zero_stage": 0,
        "gc": bool(mechanism["gc"]),
        "gradient_checkpointing": bool(mechanism["gc"]),
        "mbs": int(mechanism["mbs"]),
        "gradient_accumulation_steps": target_gbs // denominator,
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
    if model["family"] == "qwen3_5":
        row["environment_overlay"] = _runtime_overlay()
    return row


def _text_job(
    *,
    model: dict[str, Any],
    inventory: dict[str, Any],
    controls: dict[tuple[str, str], dict[str, Any]],
    phase_id: str,
    tier: str,
    mechanism_id: str,
    repeat: int = 0,
    warmup_steps: int = 2,
    measure_steps: int = 8,
    max_samples: int = ROWS,
    target_gbs: int = TARGET_GBS,
) -> dict[str, Any]:
    binding = controls[(str(model["id"]), tier)]
    row = _base_job(
        model=model,
        inventory=inventory,
        phase_id=phase_id,
        arm_id="text_length_matched",
        tier=tier,
        mechanism_id=mechanism_id,
        repeat=repeat,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        max_samples=max_samples,
        target_gbs=target_gbs,
    )
    row.update(
        {
            "dataset_id": f"frozen_video_decomposition_{model['id']}_{tier}_text_matched",
            "dataset_category": "controlled_text_length_matched_to_video",
            "data_path": binding["data_path"],
            "data_sha256": binding["data_sha256"],
            "dataset_profile_path": binding["profile"]["path"],
            "dataset_profile_sha256": binding["profile"]["sha256"],
            "expected_images_per_sample": 0,
            "expected_videos_per_sample": 0,
            "visual_runtime_evidence_required": False,
        }
    )
    row["scenario_id"] = f"{model['id']}__{tier}__{mechanism_id}__repeat{repeat}"
    row["pair_id"] = row["scenario_id"]
    row["job_id"] = stable_id("h800vidfix", row)
    return row


def _video_job(
    *,
    model: dict[str, Any],
    inventory: dict[str, Any],
    workloads: dict[tuple[str, str], dict[str, Any]],
    phase_id: str,
    tier: str,
    mechanism_id: str,
    repeat: int = 0,
    warmup_steps: int = 2,
    measure_steps: int = 8,
    max_samples: int = ROWS,
    target_gbs: int = TARGET_GBS,
) -> dict[str, Any]:
    binding = workloads[(str(model["id"]), tier)]
    settings = TIER_SETTINGS[tier]
    row = _base_job(
        model=model,
        inventory=inventory,
        phase_id=phase_id,
        arm_id="real_video",
        tier=tier,
        mechanism_id=mechanism_id,
        repeat=repeat,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        max_samples=max_samples,
        target_gbs=target_gbs,
    )
    row.update(
        {
            "dataset_id": "frozen_video_decomposition_zltbjg_v2",
            "dataset_category": "vl_single_video_business_quality",
            "data_path": str(VIDEO_TRAIN_DATA.resolve()),
            "data_sha256": sha256_file(VIDEO_TRAIN_DATA),
            "dataset_profile_path": str(Path(binding["path"]).resolve()),
            "dataset_profile_sha256": binding["sha256"],
            "media_manifest_path": str(MEDIA_MANIFEST.resolve()),
            "media_manifest_sha256": sha256_file(MEDIA_MANIFEST),
            "video_min_pixels": int(settings["video_min_pixels"]),
            "video_max_pixels": int(settings["video_max_pixels"]),
            "video_fps": float(settings["video_fps"]),
            "video_maxlen": int(settings["video_maxlen"]),
            "expected_images_per_sample": 0,
            "expected_videos_per_sample": 1,
            "visual_runtime_evidence_required": True,
            "requires_full_video_decode_canary": True,
        }
    )
    row["scenario_id"] = f"{model['id']}__{tier}__{mechanism_id}__repeat{repeat}"
    row["pair_id"] = row["scenario_id"]
    row["job_id"] = stable_id("h800vidfix", row)
    return row


def _paired_jobs(
    *,
    model: dict[str, Any],
    inventory: dict[str, Any],
    controls: dict[tuple[str, str], dict[str, Any]],
    workloads: dict[tuple[str, str], dict[str, Any]],
    phase_id: str,
    tier: str,
    mechanism_id: str,
    repeat: int = 0,
    warmup_steps: int = 2,
    measure_steps: int = 8,
    max_samples: int = ROWS,
    target_gbs: int = TARGET_GBS,
) -> list[dict[str, Any]]:
    common = dict(
        model=model,
        inventory=inventory,
        phase_id=phase_id,
        tier=tier,
        mechanism_id=mechanism_id,
        repeat=repeat,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        max_samples=max_samples,
        target_gbs=target_gbs,
    )
    return [
        _text_job(controls=controls, **common),
        _video_job(workloads=workloads, **common),
    ]


def _queues(
    specs: dict[str, dict[str, Any]],
    inventory: dict[str, dict[str, Any]],
    controls: dict[tuple[str, str], dict[str, Any]],
    workloads: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canary: list[dict[str, Any]] = []
    formal: list[dict[str, Any]] = []
    for model_id in REPRESENTATIVES:
        model = specs[model_id]
        canary.extend(
            _paired_jobs(
                model=model,
                inventory=inventory[model_id],
                controls=controls,
                workloads=workloads,
                phase_id=CANARY_PHASE_ID,
                tier="native_16",
                mechanism_id="SAFE",
                warmup_steps=0,
                measure_steps=1,
                max_samples=16,
                target_gbs=CANARY_GBS,
            )
        )
        for tier in TIER_IDS:
            formal.extend(
                _paired_jobs(
                    model=model,
                    inventory=inventory[model_id],
                    controls=controls,
                    workloads=workloads,
                    phase_id=FORMAL_PHASE_ID,
                    tier=tier,
                    mechanism_id="SAFE",
                )
            )
        for mechanism_id in ("NOGC", "PRESSURE"):
            formal.extend(
                _paired_jobs(
                    model=model,
                    inventory=inventory[model_id],
                    controls=controls,
                    workloads=workloads,
                    phase_id=FORMAL_PHASE_ID,
                    tier="native_64",
                    mechanism_id=mechanism_id,
                )
            )
        formal.extend(
            _paired_jobs(
                model=model,
                inventory=inventory[model_id],
                controls=controls,
                workloads=workloads,
                phase_id=FORMAL_PHASE_ID,
                tier="native_64",
                mechanism_id="SAFE",
                repeat=1,
            )
        )
    if len(canary) != 6 or len({row["job_id"] for row in canary}) != 6:
        raise ValueError("video canary must contain six unique paired jobs")
    if len(formal) != 42 or len({row["job_id"] for row in formal}) != 42:
        raise ValueError("video formal queue must contain 42 unique jobs")
    return canary, formal


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
        "by_tier": dict(sorted(Counter(str(row["media_tier"]) for row in jobs).items())),
        "by_mechanism": dict(sorted(Counter(str(row["mechanism_id"]) for row in jobs).items())),
        "fixed_semantics": {
            "train_type": "lora",
            "train_scope_id": "language_only",
            "freeze_vision_tower": True,
            "freeze_multi_modal_projector": True,
            "packing": False,
            "offload": False,
            "dtype": "bf16",
            "formal_target_gbs": TARGET_GBS,
            "canary_target_gbs": CANARY_GBS,
        },
        "hypotheses": {
            "memory": (
                "At recordwise matched language-backbone lengths, real-video minus "
                "text-control peak isolates the frozen video decode/vision-forward residual. "
                "The full-step peak is max(language phase, vision phase), never their sum."
            ),
            "throughput": (
                "Step time is a shared mixed-token language component plus a video-forward "
                "component driven by raw patch units, sampled frames, and media launches."
            ),
            "resolution_arm": (
                "upscaled_64 is a controlled processor-resolution mechanism arm, not evidence "
                "that the downloaded source distribution contains high-resolution video."
            ),
        },
        "decision_metrics": {
            "data_quality": [
                "recordwise text/video total-token equality",
                "one decoded video and nonzero video_grid_thw per real-video sample",
                "measured sampled-frame count and pixel tensor size",
                "repeat throughput CV <= 5%",
            ],
            "memory": [
                "rank/phase max allocated and reserved bytes",
                "paired OOM mismatch",
                "leave-family-out residual MAPE target <= 10%",
            ],
            "throughput": [
                "effective/computed tokens per second",
                "real-video minus matched-text step-time residual",
                "mechanism order accuracy target >= 90%",
                "top-1 regret target < 10%",
            ],
        },
        "coefficient_policy": "no_vl_coefficient_is_fit_before_approved_paired_gpu_evidence",
        "usage": "fit_and_hypothesis_diagnostic_only_never_prospective_acceptance",
        "gpu_training_started": False,
    }
    report["report_sha256"] = sha256_json(report)
    return report


def prepare() -> dict[str, Any]:
    for required in (VIDEO_DATA, WORKLOAD_MANIFEST, MEDIA_MANIFEST, INVENTORY):
        if not required.is_file():
            raise FileNotFoundError(f"required input is missing: {required}")
    specs = _model_specs()
    inventory = _inventory_rows()
    workloads = _workload_profiles()
    video_rows = _materialize_trainable_video_data()
    controls = _materialize_text_controls(specs, workloads, video_rows)
    _materialize_registry(controls)
    canary, formal = _queues(specs, inventory, controls, workloads)
    write_jsonl(CANARY_QUEUE, canary)
    write_jsonl(FORMAL_QUEUE, formal)
    write_json(CANARY_DESIGN, _design(phase_id=CANARY_PHASE_ID, jobs=canary, queue=CANARY_QUEUE))
    write_json(FORMAL_DESIGN, _design(phase_id=FORMAL_PHASE_ID, jobs=formal, queue=FORMAL_QUEUE))
    return {
        "campaign_id": CAMPAIGN_ID,
        "profile_manifest": str(PROFILE_MANIFEST.resolve()),
        "dataset_registry": str(DATASET_REGISTRY.resolve()),
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
