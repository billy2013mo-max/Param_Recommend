#!/usr/bin/env python3
"""Evaluate exact paired image/video frozen-vision semantic canaries."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from common import ARTIFACT_DIR, read_jsonl, sha256_file, sha256_json, write_json
from evaluate_h800_qwen35_vl_supplement_v1 import _result
from prepare_h800_frozen_vl_combined_canary_v1 import CAMPAIGN_ID, PHASE_ID, QUEUE


OUTPUT = ARTIFACT_DIR / "h800_frozen_vl_combined_canary_acceptance_v1.json"


def _rank_semantics(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    rank_rows = []
    for summary, structure in zip(
        result.get("summaries") or [], result.get("structures") or []
    ):
        media = (summary.get("runtime_batch_evidence") or {}).get("media") or {}
        phase = summary.get("vision_phase_memory_probe") or {}
        components = structure.get("components") or {}
        vision = components.get("vision_tower") or {}
        projector = components.get("multimodal_projector") or {}
        language = components.get("language_model") or {}
        checks = {
            "freeze_declaration_matched": structure.get("declaration_status") == "matched",
            "vision_is_frozen": int(vision.get("trainable_parameter_elements") or 0) == 0,
            "projector_is_frozen": int(projector.get("trainable_parameter_elements") or 0) == 0,
            "language_adapter_is_trainable": int(language.get("adapter_trainable_parameter_elements") or 0) > 0,
            "no_visual_lora_target_hit": (structure.get("lora_target_hits") or {}).get("any_visual_component") is False,
        }
        if job["arm_id"] == "real_image":
            checks.update(
                {
                    "real_image_path_observed": media.get("real_image_path_observed") is True,
                    "image_grid_rows_positive": int(media.get("image_grid_rows") or 0) > 0,
                    "image_pixel_tensor_positive": int(media.get("pixel_value_elements") or 0) > 0,
                    "vision_forward_on_every_measured_step": phase.get("all_measured_steps_observed") is True,
                }
            )
        elif job["arm_id"] == "real_video":
            checks.update(
                {
                    "real_video_path_observed": media.get("real_video_path_observed") is True,
                    "video_grid_rows_positive": int(media.get("video_grid_rows") or 0) > 0,
                    "video_pixel_tensor_positive": int(media.get("pixel_video_elements") or 0) > 0,
                    "vision_forward_on_every_measured_step": phase.get("all_measured_steps_observed") is True,
                }
            )
        else:
            checks.update(
                {
                    "no_source_media": int(media.get("source_image_count") or 0) == 0
                    and int(media.get("source_video_count") or 0) == 0,
                    "dummy_media_is_not_counted_as_real": int(
                        media.get("media_batches") or 0
                    )
                    == 0
                    and media.get("real_image_path_observed") is False
                    and media.get("real_video_path_observed") is False,
                    "vl_text_dummy_image_baseline_observed": int(
                        media.get("image_grid_rows") or 0
                    )
                    > 0
                    and int(media.get("pixel_value_elements") or 0) > 0,
                    "vl_text_dummy_vision_forward_observed": phase.get(
                        "all_measured_steps_observed"
                    )
                    is True,
                }
            )
        rank_rows.append(
            {
                "rank": summary.get("rank"),
                "checks": checks,
                "all_passed": all(checks.values()),
            }
        )
    return {
        "job_id": job["job_id"],
        "model_id": job["model_id"],
        "arm_id": job["arm_id"],
        "rank_semantics": rank_rows,
        "all_ranks_passed": len(rank_rows) == int(job["gpu_count"])
        and all(row["all_passed"] for row in rank_rows),
    }


def evaluate() -> dict[str, Any]:
    jobs = read_jsonl(QUEUE)
    results = [_result(job) for job in jobs]
    semantics = [
        _rank_semantics(job, result) for job, result in zip(jobs, results)
    ]
    classifications = Counter(str(row["classification"]) for row in results)
    checks = {
        "queue_exact": len(jobs) == 12,
        "three_models_exact": Counter(str(row["model_id"]) for row in jobs)
        == Counter({"qwen2p5_vl_3b": 4, "qwen3_vl_4b": 4, "qwen3p5_4b": 4}),
        "four_arms_exact": Counter(str(row["arm_id"]) for row in jobs)
        == Counter(
            {
                "text_length_matched": 6,
                "real_image": 3,
                "real_video": 3,
            }
        ),
        "all_jobs_success_and_calibration_eligible": all(
            row["classification"] == "success"
            and row["terminal_eligible"]
            and row["result_artifacts_complete"]
            for row in results
        ),
        "all_rank_media_and_freeze_semantics_passed": all(
            row["all_ranks_passed"] for row in semantics
        ),
    }
    report: dict[str, Any] = {
        "schema": "sft_h800_frozen_vl_combined_canary_acceptance/v1",
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "queue": {"path": str(QUEUE.resolve()), "sha256": sha256_file(QUEUE)},
        "checks": checks,
        "all_passed": all(checks.values()),
        "classifications": dict(sorted(classifications.items())),
        "semantics": semantics,
        "results": results,
        "formal_stage_allowed": all(checks.values()),
        "fit_allowed": False,
        "recommendation_release_allowed": False,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    return report


def main() -> None:
    report = evaluate()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["all_passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
