#!/usr/bin/env python3
"""Fail-closed static validation for the frozen-vision decomposition queues."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, read_json, read_jsonl, sha256_file, write_json
from prepare_h800_frozen_vl_decomposition_v1 import (
    CANARY_DESIGN,
    CANARY_PHASE_ID,
    CANARY_QUEUE,
    FORMAL_DESIGN,
    FORMAL_PHASE_ID,
    FORMAL_QUEUE,
    PROFILE_MANIFEST,
    REPRESENTATIVES,
)
from run_job import gradient_accumulation, resolve_dataset_dir, validate_job

OUTPUT = ARTIFACT_DIR / "h800_frozen_vl_decomposition_static_validation_v1.json"


def _check_file_binding(path_text: str, expected: str, label: str) -> None:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} drifted: {path}: {actual} != {expected}")


def _validate_queue(
    *,
    path: Path,
    design_path: Path,
    expected_phase: str,
    expected_jobs: int,
) -> list[dict[str, Any]]:
    jobs = read_jsonl(path)
    if len(jobs) != expected_jobs:
        raise ValueError(f"{path} contains {len(jobs)} jobs, expected {expected_jobs}")
    if len({str(row["job_id"]) for row in jobs}) != len(jobs):
        raise ValueError(f"{path} contains duplicate job IDs")
    design = read_json(design_path)
    if design.get("phase_id") != expected_phase or design.get("jobs") != len(jobs):
        raise ValueError(f"{design_path} does not bind the exact phase/job count")
    if design.get("queue", {}).get("sha256") != sha256_file(path):
        raise ValueError(f"{design_path} queue hash drifted")

    for row in jobs:
        if row.get("phase_id") != expected_phase:
            raise ValueError(f"job {row.get('job_id')} phase drifted")
        if row.get("model_id") not in REPRESENTATIVES:
            raise ValueError(f"job {row.get('job_id')} uses an unregistered model")
        if row.get("train_scope_id") != "language_only":
            raise ValueError(f"job {row.get('job_id')} is not language-only")
        if not (
            row.get("freeze_vision_tower") is True
            and row.get("freeze_multi_modal_projector") is True
            and row.get("freeze_language_model") is False
        ):
            raise ValueError(f"job {row.get('job_id')} freeze contract drifted")
        if row.get("packing") is not False or row.get("offload") is not False:
            raise ValueError(f"job {row.get('job_id')} enabled packing/offload")
        if row.get("enable_vision_phase_memory_probe") is not True:
            raise ValueError(f"job {row.get('job_id')} disabled phase memory evidence")
        expected_ga = gradient_accumulation(row)
        if int(row["gradient_accumulation_steps"]) != expected_ga:
            raise ValueError(f"job {row.get('job_id')} GA drifted")
        validate_job(row)
        resolve_dataset_dir(row)
        _check_file_binding(row["data_path"], row["data_sha256"], "dataset")
        _check_file_binding(
            row["dataset_profile_path"],
            row["dataset_profile_sha256"],
            "dataset profile",
        )
        if row["arm_id"] == "real_image":
            if row.get("visual_runtime_evidence_required") is not True:
                raise ValueError(f"real-image job {row['job_id']} relaxed visual evidence")
            if int(row.get("expected_images_per_sample", -1)) != 2:
                raise ValueError(f"real-image job {row['job_id']} image count drifted")
            _check_file_binding(
                row["media_manifest_path"], row["media_manifest_sha256"], "media manifest"
            )
        else:
            if row.get("visual_runtime_evidence_required") is not False:
                raise ValueError(f"text job {row['job_id']} requires a visual path")
            if int(row.get("expected_images_per_sample", -1)) != 0:
                raise ValueError(f"text job {row['job_id']} declares images")
    return jobs


def _validate_pairing(jobs: list[dict[str, Any]]) -> None:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in jobs:
        if row["arm_id"] != "text_base":
            by_pair[str(row["pair_id"])].append(row)
    for pair_id, rows in by_pair.items():
        arms = Counter(str(row["arm_id"]) for row in rows)
        if arms != Counter({"text_length_matched": 1, "real_image": 1}):
            raise ValueError(f"pair {pair_id} is incomplete: {dict(arms)}")
        left, right = sorted(rows, key=lambda row: str(row["arm_id"]))
        for key in (
            "model_id",
            "model_family",
            "media_tier",
            "mechanism_id",
            "gpu_count",
            "zero_stage",
            "gc",
            "mbs",
            "gradient_accumulation_steps",
            "repeat",
            "target_gbs",
            "cutoff_len",
        ):
            if left.get(key) != right.get(key):
                raise ValueError(f"pair {pair_id} differs on {key}")


def validate() -> dict[str, Any]:
    profiles = read_json(PROFILE_MANIFEST)
    if profiles.get("all_recordwise_total_matches_exact") is not True:
        raise ValueError("matched text profiles are not recordwise exact")
    if len(profiles.get("profiles") or []) != 9:
        raise ValueError("profile manifest must contain three base and six matched profiles")
    for row in profiles["profiles"]:
        _check_file_binding(row["data_path"], row["data_sha256"], "control dataset")
        profile = row["profile"]
        _check_file_binding(profile["path"], profile["sha256"], "control profile")
        if int(profile["rows"]) != 1000:
            raise ValueError("every control profile must contain 1000 rows")

    canary = _validate_queue(
        path=CANARY_QUEUE,
        design_path=CANARY_DESIGN,
        expected_phase=CANARY_PHASE_ID,
        expected_jobs=6,
    )
    formal = _validate_queue(
        path=FORMAL_QUEUE,
        design_path=FORMAL_DESIGN,
        expected_phase=FORMAL_PHASE_ID,
        expected_jobs=45,
    )
    _validate_pairing(canary)
    _validate_pairing(formal)

    formal_models = Counter(str(row["model_id"]) for row in formal)
    formal_arms = Counter(str(row["arm_id"]) for row in formal)
    formal_mechanisms = Counter(str(row["mechanism_id"]) for row in formal)
    if formal_models != Counter({model_id: 15 for model_id in REPRESENTATIVES}):
        raise ValueError(f"formal model balance drifted: {dict(formal_models)}")
    if formal_arms != Counter(
        {"text_base": 9, "text_length_matched": 18, "real_image": 18}
    ):
        raise ValueError(f"formal arm balance drifted: {dict(formal_arms)}")
    if formal_mechanisms != Counter({"SAFE": 21, "NOGC": 9, "PRESSURE": 15}):
        raise ValueError(f"formal mechanism balance drifted: {dict(formal_mechanisms)}")

    report = {
        "schema": "sft_h800_frozen_vl_decomposition_static_validation/v1",
        "all_passed": True,
        "gpu_training_started": False,
        "profiles": len(profiles["profiles"]),
        "canary_jobs": len(canary),
        "formal_jobs": len(formal),
        "formal_by_model": dict(sorted(formal_models.items())),
        "formal_by_arm": dict(sorted(formal_arms.items())),
        "formal_by_mechanism": dict(sorted(formal_mechanisms.items())),
        "packing_enabled_jobs": sum(int(row["packing"]) for row in canary + formal),
        "exact_text_image_pairs": sum(
            int(row["arm_id"] == "real_image") for row in formal
        ),
    }
    write_json(OUTPUT, report)
    return report


def main() -> None:
    print(json.dumps(validate(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
