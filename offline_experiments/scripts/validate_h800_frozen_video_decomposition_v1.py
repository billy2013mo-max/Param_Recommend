#!/usr/bin/env python3
"""Fail-closed static validation for frozen-video decomposition queues."""

from __future__ import annotations

import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, read_json, read_jsonl, sha256_file, write_json
from prepare_h800_frozen_video_decomposition_v1 import (
    CANARY_DESIGN,
    CANARY_GBS,
    CANARY_PHASE_ID,
    CANARY_QUEUE,
    FORMAL_DESIGN,
    FORMAL_PHASE_ID,
    FORMAL_QUEUE,
    PROFILE_MANIFEST,
    REPRESENTATIVES,
    ROWS,
    TARGET_GBS,
    TIER_IDS,
    VIDEO_TRAIN_DATA,
)
from run_job import (
    gradient_accumulation,
    render_config,
    resolve_dataset_dir,
    validate_job,
)


OUTPUT = ARTIFACT_DIR / "h800_frozen_video_decomposition_static_validation_v1.json"


def _check_file(path_text: str, expected: str, label: str) -> Path:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} drifted: {actual} != {expected}")
    return path


def _validate_media_files() -> dict[str, Any]:
    rows = read_jsonl(VIDEO_TRAIN_DATA)
    if len(rows) != ROWS:
        raise ValueError(f"trainable video view must contain {ROWS} rows")
    sizes = []
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"video row {index} is not a two-message ShareGPT sample")
        if [message.get("role") for message in messages] != ["user", "assistant"]:
            raise ValueError(f"video row {index} message roles drifted")
        if not str(messages[1].get("content") or "").strip():
            raise ValueError(f"video row {index} has no supervised target tokens")
        if str(messages[0].get("content") or "").count("<video>") != 1:
            raise ValueError(f"video row {index} placeholder count drifted")
        videos = row.get("videos") or []
        if len(videos) != 1:
            raise ValueError(f"video row {index} must contain one video")
        path = Path(str(videos[0]))
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"video row {index} media is missing/empty: {path}")
        sizes.append(path.stat().st_size)
    return {
        "rows": len(rows),
        "all_files_present": True,
        "minimum_video_bytes": min(sizes),
        "maximum_video_bytes": max(sizes),
        "training_view_format": "sharegpt",
        "full_decode_deferred_to_runtime_canary": True,
    }


def _validate_queue(
    *,
    queue_path: Path,
    design_path: Path,
    phase_id: str,
    expected_jobs: int,
) -> list[dict[str, Any]]:
    jobs = read_jsonl(queue_path)
    if len(jobs) != expected_jobs or len({row["job_id"] for row in jobs}) != expected_jobs:
        raise ValueError(f"{queue_path} job count/uniqueness drifted")
    design = read_json(design_path)
    if (
        design.get("phase_id") != phase_id
        or design.get("jobs") != expected_jobs
        or design.get("queue", {}).get("sha256") != sha256_file(queue_path)
    ):
        raise ValueError(f"{design_path} does not bind the exact queue")

    for row in jobs:
        if row.get("phase_id") != phase_id or row.get("model_id") not in REPRESENTATIVES:
            raise ValueError(f"job {row.get('job_id')} phase/model drifted")
        if row.get("media_tier") not in TIER_IDS:
            raise ValueError(f"job {row.get('job_id')} video tier drifted")
        if row.get("train_scope_id") != "language_only" or not (
            row.get("freeze_vision_tower") is True
            and row.get("freeze_multi_modal_projector") is True
            and row.get("freeze_language_model") is False
        ):
            raise ValueError(f"job {row.get('job_id')} freeze scope drifted")
        if row.get("packing") is not False or row.get("offload") is not False:
            raise ValueError(f"job {row.get('job_id')} packing/offload drifted")
        expected_gbs = CANARY_GBS if phase_id == CANARY_PHASE_ID else TARGET_GBS
        if int(row["target_gbs"]) != expected_gbs:
            raise ValueError(f"job {row.get('job_id')} GBS drifted")
        validate_job(row)
        resolve_dataset_dir(row)
        if int(row["gradient_accumulation_steps"]) != gradient_accumulation(row):
            raise ValueError(f"job {row.get('job_id')} GA drifted")
        _check_file(row["data_path"], row["data_sha256"], "dataset")
        profile_path = _check_file(
            row["dataset_profile_path"], row["dataset_profile_sha256"], "profile"
        )
        if row["arm_id"] == "real_video":
            if row.get("visual_runtime_evidence_required") is not True:
                raise ValueError(f"real-video job {row['job_id']} relaxed visual evidence")
            if int(row.get("expected_videos_per_sample", -1)) != 1:
                raise ValueError(f"real-video job {row['job_id']} video count drifted")
            profile = read_json(profile_path)
            binding = profile["processor_binding"]
            expected_options = {
                "video_min_pixels": int(binding["video_min_pixels"]),
                "video_max_pixels": int(binding["video_max_pixels"]),
                "video_maxlen": int(binding["video_max_frames"]),
                "video_fps": float(binding["video_sample_fps"]),
            }
            for key, value in expected_options.items():
                if row.get(key) != value:
                    raise ValueError(f"real-video job {row['job_id']} {key} drifted")
            maximum_total = int(profile["summary"]["total_tokens"]["max"])
            if maximum_total > int(row["cutoff_len"]):
                raise ValueError(
                    f"real-video job {row['job_id']} truncates matched workload: "
                    f"{maximum_total} > {row['cutoff_len']}"
                )
            _check_file(
                row["media_manifest_path"], row["media_manifest_sha256"], "media manifest"
            )
        else:
            if row.get("visual_runtime_evidence_required") is not False:
                raise ValueError(f"text job {row['job_id']} requires visual evidence")
            if int(row.get("expected_videos_per_sample", -1)) != 0:
                raise ValueError(f"text job {row['job_id']} declares videos")
    return jobs


def _validate_pairing(jobs: list[dict[str, Any]]) -> None:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in jobs:
        by_pair[str(row["pair_id"])].append(row)
    for pair_id, rows in by_pair.items():
        arms = Counter(str(row["arm_id"]) for row in rows)
        if arms != Counter({"text_length_matched": 1, "real_video": 1}):
            raise ValueError(f"pair {pair_id} is incomplete: {dict(arms)}")
        left, right = rows
        for key in (
            "model_id",
            "model_family",
            "media_tier",
            "mechanism_id",
            "gpu_count",
            "gc",
            "mbs",
            "gradient_accumulation_steps",
            "repeat",
            "target_gbs",
            "cutoff_len",
        ):
            if left.get(key) != right.get(key):
                raise ValueError(f"pair {pair_id} differs on {key}")


def _render_canary(jobs: list[dict[str, Any]]) -> int:
    rendered = 0
    with tempfile.TemporaryDirectory(prefix="vl-video-render-") as temporary:
        root = Path(temporary)
        for index, row in enumerate(jobs):
            config = render_config(dict(row), root / str(index))
            if row["arm_id"] == "real_video":
                for key in (
                    "video_min_pixels",
                    "video_max_pixels",
                    "video_fps",
                    "video_maxlen",
                ):
                    if config.get(key) != row[key]:
                        raise ValueError(f"rendered config dropped {key}")
            rendered += 1
    return rendered


def validate() -> dict[str, Any]:
    profiles = read_json(PROFILE_MANIFEST)
    if profiles.get("all_recordwise_total_matches_exact") is not True:
        raise ValueError("matched text/video profiles are not recordwise exact")
    if len(profiles.get("profiles") or []) != 12:
        raise ValueError("video profile manifest must contain twelve matched controls")
    for row in profiles["profiles"]:
        _check_file(row["data_path"], row["data_sha256"], "text control")
        _check_file(row["profile"]["path"], row["profile"]["sha256"], "text profile")
        if int(row["profile"]["rows"]) != ROWS:
            raise ValueError("text control profile row count drifted")
        if int(row["profile"]["label_tokens"]["minimum"]) <= 0:
            raise ValueError("text control profile has no supervised label tokens")

    media = _validate_media_files()
    canary = _validate_queue(
        queue_path=CANARY_QUEUE,
        design_path=CANARY_DESIGN,
        phase_id=CANARY_PHASE_ID,
        expected_jobs=6,
    )
    formal = _validate_queue(
        queue_path=FORMAL_QUEUE,
        design_path=FORMAL_DESIGN,
        phase_id=FORMAL_PHASE_ID,
        expected_jobs=42,
    )
    _validate_pairing(canary)
    _validate_pairing(formal)
    rendered = _render_canary(canary)

    report = {
        "schema": "sft_h800_frozen_video_decomposition_static_validation/v1",
        "all_passed": True,
        "gpu_training_started": False,
        "profiles": len(profiles["profiles"]),
        "media": media,
        "canary_jobs": len(canary),
        "formal_jobs": len(formal),
        "canary_configs_rendered": rendered,
        "formal_by_model": dict(sorted(Counter(row["model_id"] for row in formal).items())),
        "formal_by_arm": dict(sorted(Counter(row["arm_id"] for row in formal).items())),
        "formal_by_tier": dict(sorted(Counter(row["media_tier"] for row in formal).items())),
        "formal_by_mechanism": dict(sorted(Counter(row["mechanism_id"] for row in formal).items())),
        "packing_enabled_jobs": sum(int(row["packing"]) for row in canary + formal),
        "exact_text_video_pairs": sum(int(row["arm_id"] == "real_video") for row in formal),
    }
    write_json(OUTPUT, report)
    return report


def main() -> None:
    print(json.dumps(validate(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
