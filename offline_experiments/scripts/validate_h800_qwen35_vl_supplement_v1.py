#!/usr/bin/env python3
"""Fail-closed static validation for the Qwen3.5/VL supplement campaign."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, DATA_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json
from run_job import gradient_accumulation, validate_job
from prepare_h800_qwen35_vl_supplement_v1 import (
    CAMPAIGN_ID,
    CANARY_DESIGN,
    CANARY_MANIFEST,
    CANARY_PHASE_ID,
    CANARY_QUEUE,
    DESIGN_SCHEMA,
    FORMAL_DESIGN,
    FORMAL_MANIFEST,
    FORMAL_PHASE_ID,
    FORMAL_QUEUE,
    FROZEN_SELECTION,
    INVENTORY,
    JOB_SCHEMA,
    MEDIA_MANIFEST,
    MODEL_SPECS,
    PROFILE_MANIFEST,
    RUNTIME_CONTRACT,
    STAGING_CONFIG,
    TEXT_PROFILE_MANIFEST,
    VL_DATA,
)


OUTPUT = ARTIFACT_DIR / "h800_qwen35_vl_supplement_static_validation_v1.json"
EXPECTED_FORMAL_TRACKS = {
    "qwen35_text_cross_scale": 28,
    "vl_anchor_mechanism_calibration": 16,
    "qwen35_real_image_cross_scale": 16,
    "vl_dense_scale_transfer": 14,
    "vl_moe_transfer": 2,
    "visual_train_scope_ablation": 4,
    "repeat_variance_diagnostic": 6,
}


def _internal_hash(report: dict[str, Any]) -> bool:
    unsigned = dict(report)
    expected = unsigned.pop("report_sha256", None)
    return expected == sha256_json(unsigned)


def _binding_exact(binding: dict[str, Any], path: Path) -> bool:
    return (
        path.is_file()
        and Path(binding.get("path", "")).resolve() == path.resolve()
        and binding.get("sha256") == sha256_file(path)
    )


def _result_collisions(rows: list[dict[str, Any]]) -> list[str]:
    return [
        str(ROOT / "results" / str(row["job_id"]))
        for row in rows
        if (ROOT / "results" / str(row["job_id"])).exists()
    ]


def _validate_stage(
    *, queue: Path, design_path: Path, manifest_path: Path, phase_id: str, expected: int
) -> dict[str, Any]:
    rows = read_jsonl(queue)
    design = read_json(design_path)
    manifest = read_json(manifest_path)
    row_errors = []
    for row in rows:
        try:
            validate_job(row)
            if gradient_accumulation(row) != int(row["gradient_accumulation_steps"]):
                raise ValueError("stored and derived gradient accumulation differ")
        except (KeyError, TypeError, ValueError) as error:
            row_errors.append({"job_id": row.get("job_id"), "error": repr(error)})
    ids = [str(row.get("job_id") or "") for row in rows]
    required_files = set()
    for row in rows:
        for key in (
            "model_path",
            "tokenizer_path",
            "data_path",
            "dataset_profile_path",
            "declared_model_manifest_path",
        ):
            required_files.add((key, str(row.get(key) or "")))
        if row.get("media_manifest_path"):
            required_files.add(("media_manifest_path", str(row["media_manifest_path"])))
        overlay = row.get("environment_overlay") or {}
        if overlay:
            required_files.add(("contract_path", str(overlay.get("contract_path") or "")))
    missing = []
    for key, path_text in sorted(required_files):
        path = Path(path_text)
        expects_directory = key in {"model_path", "tokenizer_path"}
        exists = path.is_dir() if expects_directory else path.is_file()
        if not exists:
            missing.append({"field": key, "path": path_text})
    expected_role = "canary_excluded" if phase_id == CANARY_PHASE_ID else "calibration"
    checks = {
        "queue_count_and_ids_exact": len(rows) == expected
        and len(ids) == len(set(ids))
        and all(ids),
        "row_identity_exact": all(
            row.get("schema") == JOB_SCHEMA
            and row.get("campaign_id") == CAMPAIGN_ID
            and row.get("phase_id") == phase_id
            for row in rows
        ),
        "job_mechanisms_valid": not row_errors,
        "all_paths_present": not missing,
        "bf16_sft_domain_exact": all(
            row.get("packing") is False
            and row.get("offload") is False
            and row.get("train_type") in {"lora", "full"}
            for row in rows
        ),
        "partition_role_exact": all(
            (row.get("calibration_partition") or {}).get("role") == expected_role
            for row in rows
        ),
        "visual_jobs_use_real_media": all(
            row.get("visual_runtime_evidence_required") is True
            and int(row.get("expected_images_per_sample") or 0) == 2
            and row.get("dataset_id") == "vl_pzfj38_calibration_v1"
            for row in rows
            if row.get("track") != "qwen35_text_cross_scale"
        ),
        "qwen35_overlay_bound": all(
            bool(row.get("environment_overlay"))
            for row in rows
            if row.get("model_family") == "qwen3_5"
        ),
        "non_qwen35_has_no_overlay": all(
            not row.get("environment_overlay")
            for row in rows
            if row.get("model_family") != "qwen3_5"
        ),
        "design_identity_and_hash_exact": design.get("schema") == DESIGN_SCHEMA
        and design.get("campaign_id") == CAMPAIGN_ID
        and design.get("phase_id") == phase_id
        and _internal_hash(design),
        "design_binds_queue": design.get("queue", {}).get("sha256")
        == sha256_file(queue)
        and design.get("queue", {}).get("ordered_job_ids") == ids
        and design.get("queue", {}).get("ordered_job_payload_sha256")
        == sha256_json(rows),
        "manifest_hash_and_bindings_exact": _internal_hash(manifest)
        and _binding_exact(manifest.get("design") or {}, design_path)
        and manifest.get("queue", {}).get("sha256") == sha256_file(queue)
        and manifest.get("queue", {}).get("ordered_job_ids") == ids,
        "pre_gpu_flags_fail_closed": design.get("gpu_training_started") is False
        and design.get("execution_authorized") is False
        and design.get("publication_allowed") is False,
    }
    return {
        "phase_id": phase_id,
        "queue": {"path": str(queue.resolve()), "sha256": sha256_file(queue)},
        "jobs": len(rows),
        "gpu_job_equivalents": sum(int(row["gpu_count"]) for row in rows),
        "tracks": dict(sorted(Counter(str(row["track"]) for row in rows).items())),
        "checks": checks,
        "row_errors": row_errors,
        "missing_paths": missing,
        "result_collisions": _result_collisions(rows),
        "all_passed": all(checks.values()),
    }


def validate(*, allow_results: bool = False) -> dict[str, Any]:
    required = (
        INVENTORY,
        PROFILE_MANIFEST,
        TEXT_PROFILE_MANIFEST,
        MEDIA_MANIFEST,
        VL_DATA,
        RUNTIME_CONTRACT,
        FROZEN_SELECTION,
        CANARY_QUEUE,
        FORMAL_QUEUE,
        CANARY_DESIGN,
        FORMAL_DESIGN,
        CANARY_MANIFEST,
        FORMAL_MANIFEST,
        STAGING_CONFIG,
        DATA_DIR / "dataset_info.json",
    )
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        return {
            "schema": "sft_h800_qwen35_vl_supplement_static_validation/v1",
            "all_passed": False,
            "missing_required_artifacts": absent,
        }
    canary = _validate_stage(
        queue=CANARY_QUEUE,
        design_path=CANARY_DESIGN,
        manifest_path=CANARY_MANIFEST,
        phase_id=CANARY_PHASE_ID,
        expected=15,
    )
    formal = _validate_stage(
        queue=FORMAL_QUEUE,
        design_path=FORMAL_DESIGN,
        manifest_path=FORMAL_MANIFEST,
        phase_id=FORMAL_PHASE_ID,
        expected=86,
    )
    inventory = read_json(INVENTORY)
    profiles = read_json(PROFILE_MANIFEST)
    selection = read_json(FROZEN_SELECTION)
    config = read_json(STAGING_CONFIG)
    all_ids = [str(row["job_id"]) for row in read_jsonl(CANARY_QUEUE)] + [
        str(row["job_id"]) for row in read_jsonl(FORMAL_QUEUE)
    ]
    checks = {
        "stages_passed": canary["all_passed"] and formal["all_passed"],
        "no_cross_stage_job_id_collision": len(all_ids) == len(set(all_ids)) == 101,
        "formal_track_counts_exact": formal["tracks"] == EXPECTED_FORMAL_TRACKS,
        "inventory_exact": inventory.get("schema")
        == "sft_h800_qwen35_vl_supplement_model_inventory/v1"
        and _internal_hash(inventory)
        and len(inventory.get("models") or []) == len(MODEL_SPECS)
        and {row["id"] for row in inventory["models"]}
        == {row["id"] for row in MODEL_SPECS},
        "profile_manifest_exact": profiles.get("schema")
        == "sft_h800_qwen35_vl_supplement_processor_profiles/v1"
        and _internal_hash(profiles)
        and profiles.get("all_actual_processor_checks_passed") is True
        and profiles.get("all_alias_equivalence_checks_passed") is True
        and len(profiles.get("profiles") or []) == 22,
        "selection_frozen_before_gpu": selection.get("generated_before_gpu") is True
        and selection.get("gpu_training_started") is False
        and selection.get("automatic_execution_from_prediction") is False
        and _internal_hash(selection),
        "config_scope_exact": config.get("training_scope", {}).get("gpu_ids")
        == list(range(8))
        and config.get("training_scope", {}).get("max_gpu_count") == 4
        and config.get("training_scope", {}).get("gpu_counts") == [1, 2, 4]
        and config.get("measurement", {}).get("performance_parallelism")
        == "disjoint_gpu_masks",
        "no_preexisting_results": allow_results
        or (not canary["result_collisions"] and not formal["result_collisions"]),
    }
    report: dict[str, Any] = {
        "schema": "sft_h800_qwen35_vl_supplement_static_validation/v1",
        "campaign_id": CAMPAIGN_ID,
        "validation_mode": "allow_results" if allow_results else "strict_pre_gpu",
        "checks": checks,
        "all_passed": all(checks.values()),
        "canary": canary,
        "formal": formal,
        "interpretation": {
            "canary_is_fit_evidence": False,
            "formal_is_prospective_acceptance": False,
            "confirmed_oom_is_right_censored": True,
            "software_failure_is_oom": False,
        },
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-results", action="store_true")
    args = parser.parse_args()
    report = validate(allow_results=args.allow_results)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
