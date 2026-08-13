#!/usr/bin/env python3
"""Freeze one ordered image-then-video semantic canary queue and staging config."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, MATRIX_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json, write_jsonl
from prepare_h800_frozen_video_decomposition_v1 import CANARY_QUEUE as VIDEO_CANARY_QUEUE
from prepare_h800_frozen_vl_decomposition_v1 import CANARY_QUEUE as IMAGE_CANARY_QUEUE


CAMPAIGN_ID = "h800_frozen_vl_combined_canary_20260811_v1"
PHASE_ID = "h800_frozen_vl_combined_canary_v1"
QUEUE = MATRIX_DIR / "h800_frozen_vl_combined_canary_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_frozen_vl_combined_canary_design_v1.json"
STAGING_DIR = ROOT / "frozen_vl_staging"
STAGING_CONFIG = STAGING_DIR / "experiment.h800_frozen_vl_combined_canary_v1.json"
BASE_CONFIG = ROOT / "qwen35_vl_staging/experiment.h800_qwen35_vl_supplement_v1.json"
AUTHORIZED_GPU_IDS = list(range(7))


def prepare() -> dict[str, Any]:
    image_jobs = read_jsonl(IMAGE_CANARY_QUEUE)
    video_jobs = read_jsonl(VIDEO_CANARY_QUEUE)
    if len(image_jobs) != 6 or len(video_jobs) != 6:
        raise ValueError("combined canary requires exact six-job image and video queues")
    jobs = [*image_jobs, *video_jobs]
    if len({str(row["job_id"]) for row in jobs}) != 12:
        raise ValueError("combined canary job IDs are not unique")
    write_jsonl(QUEUE, jobs)

    design: dict[str, Any] = {
        "schema": "sft_h800_frozen_vl_combined_canary_design/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue": {
            "path": str(QUEUE.resolve()),
            "sha256": sha256_file(QUEUE),
            "ordered_job_ids": [str(row["job_id"]) for row in jobs],
        },
        "source_queues": [
            {"modality": "image", "path": str(IMAGE_CANARY_QUEUE.resolve()), "sha256": sha256_file(IMAGE_CANARY_QUEUE)},
            {"modality": "video", "path": str(VIDEO_CANARY_QUEUE.resolve()), "sha256": sha256_file(VIDEO_CANARY_QUEUE)},
        ],
        "jobs": len(jobs),
        "by_model": dict(sorted(Counter(str(row["model_id"]) for row in jobs).items())),
        "by_arm": dict(sorted(Counter(str(row["arm_id"]) for row in jobs).items())),
        "execution_order_policy": "all_image_pairs_then_all_video_pairs",
        "authorized_gpu_ids": AUTHORIZED_GPU_IDS,
        "packing": False,
        "formal_execution_authorized": False,
        "formal_requires_canary_acceptance": True,
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
        "global_batch_sizes": [8, 64],
        "gradient_checkpointing": [True],
        "zero_by_gpu_count": {"1": ["none"]},
        "objective": "run exact paired frozen-vision image/video semantic canaries",
    }
    config["datasets"] = [
        {
            "id": str(row["dataset_id"]),
            "category": str(row["dataset_category"]),
            "target_cutoffs": [int(row["cutoff_len"])],
        }
        for row in jobs
    ]
    config["measurement"]["throughput_warmup_steps"] = 0
    config["measurement"]["throughput_measure_steps"] = 2
    config["measurement"]["rerun_on_unhealthy_result"] = False
    config["combined_vl_canary_policy"] = {
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "queue_sha256": sha256_file(QUEUE),
        "gpu_ids": AUTHORIZED_GPU_IDS,
        "preemption_allowed": False,
        "gpu7_excluded": True,
        "formal_requires_acceptance": True,
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
        "gpu_training_started": False,
    }


def main() -> None:
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
