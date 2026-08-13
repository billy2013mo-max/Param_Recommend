#!/usr/bin/env python3
"""Freeze one ordered image-then-video formal calibration queue and config."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json, write_jsonl
from prepare_h800_frozen_video_decomposition_v1 import FORMAL_QUEUE as VIDEO_FORMAL_QUEUE
from prepare_h800_frozen_vl_combined_canary_v1 import AUTHORIZED_GPU_IDS, STAGING_DIR
from prepare_h800_frozen_vl_decomposition_v1 import FORMAL_QUEUE as IMAGE_FORMAL_QUEUE


CAMPAIGN_ID = "h800_frozen_vl_combined_formal_20260811_v1"
PHASE_ID = "h800_frozen_vl_combined_formal_v1"
QUEUE = MATRIX_DIR / "h800_frozen_vl_combined_formal_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_frozen_vl_combined_formal_design_v1.json"
STAGING_CONFIG = STAGING_DIR / "experiment.h800_frozen_vl_combined_formal_v1.json"
BASE_CONFIG = ROOT / "qwen35_vl_staging/experiment.h800_qwen35_vl_supplement_v1.json"
CANARY_ACCEPTANCE = ARTIFACT_DIR / "h800_frozen_vl_combined_canary_acceptance_v1.json"


def prepare() -> dict[str, Any]:
    image_jobs = read_jsonl(IMAGE_FORMAL_QUEUE)
    video_jobs = read_jsonl(VIDEO_FORMAL_QUEUE)
    if len(image_jobs) != 45 or len(video_jobs) != 42:
        raise ValueError("combined formal requires exact 45 image and 42 video jobs")
    jobs = [*image_jobs, *video_jobs]
    if len({str(row["job_id"]) for row in jobs}) != 87:
        raise ValueError("combined formal job IDs are not unique")
    if any(int(row["gpu_count"]) != 1 for row in jobs):
        raise ValueError("combined formal GPU 0-6 staging supports only single-card jobs")
    write_jsonl(QUEUE, jobs)

    design: dict[str, Any] = {
        "schema": "sft_h800_frozen_vl_combined_formal_design/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue": {
            "path": str(QUEUE.resolve()),
            "sha256": sha256_file(QUEUE),
            "ordered_job_ids": [str(row["job_id"]) for row in jobs],
        },
        "source_queues": [
            {"modality": "image", "path": str(IMAGE_FORMAL_QUEUE.resolve()), "sha256": sha256_file(IMAGE_FORMAL_QUEUE)},
            {"modality": "video", "path": str(VIDEO_FORMAL_QUEUE.resolve()), "sha256": sha256_file(VIDEO_FORMAL_QUEUE)},
        ],
        "jobs": len(jobs),
        "by_model": dict(sorted(Counter(str(row["model_id"]) for row in jobs).items())),
        "by_track": dict(sorted(Counter(str(row["track"]) for row in jobs).items())),
        "by_arm": dict(sorted(Counter(str(row["arm_id"]) for row in jobs).items())),
        "by_mechanism": dict(sorted(Counter(str(row["mechanism_id"]) for row in jobs).items())),
        "execution_order_policy": "all_image_diagnostics_then_all_video_diagnostics",
        "authorized_gpu_ids": AUTHORIZED_GPU_IDS,
        "canary_acceptance_required": str(CANARY_ACCEPTANCE.resolve()),
        "packing": False,
        "usage": "fit_and_diagnostic_only_not_prospective_acceptance",
        "gpu_training_started": False,
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)

    config = read_json(BASE_CONFIG)
    config["training_scope"] = {
        "phase_id": PHASE_ID,
        "model_ids": ["qwen2p5_vl_3b", "qwen3_vl_4b", "qwen3p5_4b"],
        "gpu_ids": AUTHORIZED_GPU_IDS,
        "exclusive_node_gpu_ids": AUTHORIZED_GPU_IDS,
        "max_gpu_count": 1,
        "stage": "sft",
        "precision": "bf16",
        "gpu_type": "NVIDIA H800 140GB HBM3",
        "gpu_counts": [1],
        "global_batch_sizes": [64],
        "gradient_checkpointing": [False, True],
        "zero_by_gpu_count": {"1": ["none"]},
        "objective": "fit frozen-vision image/video memory and throughput residuals",
    }
    deduplicated: dict[tuple[str, int], dict[str, Any]] = {}
    for row in jobs:
        key = (str(row["dataset_id"]), int(row["cutoff_len"]))
        deduplicated[key] = {
            "id": key[0],
            "category": str(row["dataset_category"]),
            "target_cutoffs": [key[1]],
        }
    config["datasets"] = list(deduplicated.values())
    config["measurement"]["throughput_warmup_steps"] = 2
    config["measurement"]["throughput_measure_steps"] = 8
    config["measurement"]["rerun_on_unhealthy_result"] = False
    config["combined_vl_formal_policy"] = {
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "queue_sha256": sha256_file(QUEUE),
        "gpu_ids": AUTHORIZED_GPU_IDS,
        "preemption_allowed": False,
        "gpu7_excluded": True,
        "canary_acceptance_required": str(CANARY_ACCEPTANCE.resolve()),
        "prospective_acceptance": False,
    }
    config.pop("supplement_policy", None)
    write_json(STAGING_CONFIG, config)
    return {
        "campaign_id": CAMPAIGN_ID,
        "queue": str(QUEUE.resolve()),
        "queue_sha256": sha256_file(QUEUE),
        "jobs": len(jobs),
        "design": str(DESIGN.resolve()),
        "staging_config": str(STAGING_CONFIG.resolve()),
        "authorized_gpu_ids": AUTHORIZED_GPU_IDS,
        "canary_acceptance_required": str(CANARY_ACCEPTANCE.resolve()),
        "gpu_training_started": False,
    }


def main() -> None:
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
