#!/usr/bin/env python3
"""Freeze the 24 Packing-transfer scores before any validation outcome exists."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from analyze_h800_packing_config_ranking_partial_virtual_mbs_v1 import (
    BusinessVirtualMBSPredictor,
    _score,
)
from common import (
    ARTIFACT_DIR,
    RESULTS_DIR,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
)
from prepare_h800_packing_transfer_validation_v1 import (
    DESIGN,
    FROZEN_PREDICTIONS,
    PACKING_RANKING_ACCEPTANCE,
    QUEUE,
)
from throughput_predictor import DEFAULT_MODEL_ARTIFACT as THROUGHPUT_ARTIFACT


SCHEMA = "sft_h800_packing_transfer_validation_frozen_predictions/v1"


def freeze(output: Path = FROZEN_PREDICTIONS) -> dict[str, Any]:
    jobs = read_jsonl(QUEUE)
    if len(jobs) != 24 or len({str(row["job_id"]) for row in jobs}) != 24:
        raise ValueError("transfer-validation queue is not the exact 24-job design")
    existing = [
        str(row["job_id"])
        for row in jobs
        if (RESULTS_DIR / str(row["job_id"]) / "latest_attempt.json").is_file()
    ]
    if existing:
        raise ValueError(
            "validation outcomes already exist; predictions are no longer prospective: "
            f"{existing[:5]}"
        )

    predictor = BusinessVirtualMBSPredictor()
    predictions: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        prediction = _score(job, predictor)
        score = float(prediction["predicted_logical_samples_per_second"])
        if not math.isfinite(score) or score <= 0.0:
            raise ValueError(f"invalid frozen score for {job['job_id']}")
        row = {
            "job_id": str(job["job_id"]),
            "scene_id": str(job["scene_id"]),
            "ranking_group_id": str(job["ranking_group_id"]),
            "predicted_ranking_score": score,
            # The small transfer campaign intentionally runs all four
            # predeclared candidates.  An OOM, especially on predicted Top1,
            # is therefore observed as a failed safety transfer instead of
            # being hidden by retrospective filtering.
            "admitted": True,
            "admission_contract": "all_predeclared_candidates_run; any OOM is retained as failure evidence",
            "prediction": prediction,
        }
        predictions.append(row)
        if job.get("ranking_eligible") is True:
            groups[str(job["ranking_group_id"])].append(row)

    ranking_freeze = {
        group_id: {
            "predicted_top1_job_id": max(
                rows, key=lambda row: float(row["predicted_ranking_score"])
            )["job_id"],
            "ordered_job_ids": [
                row["job_id"]
                for row in sorted(
                    rows,
                    key=lambda row: -float(row["predicted_ranking_score"]),
                )
            ],
        }
        for group_id, rows in sorted(groups.items())
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_before_first_validation_gpu_execution": True,
        "validation_outcomes_read": False,
        "coefficients_fitted": False,
        "queue_binding": {
            "path": str(QUEUE.resolve()),
            "sha256": sha256_file(QUEUE),
            "jobs": len(jobs),
        },
        "design_binding": {
            "path": str(DESIGN.resolve()),
            "sha256": sha256_file(DESIGN),
        },
        "model_bindings": {
            "accepted_packing_ranking_rule": {
                "path": str(PACKING_RANKING_ACCEPTANCE.resolve()),
                "sha256": sha256_file(PACKING_RANKING_ACCEPTANCE),
            },
            "frozen_unpacked_throughput": {
                "path": str(Path(THROUGHPUT_ARTIFACT).resolve()),
                "sha256": sha256_file(Path(THROUGHPUT_ARTIFACT)),
            },
            "prediction_rule": "accepted frozen Unpacked V5 at upload-time samples_per_pack.mean; Packing coefficient is fixed at zero",
        },
        "ranking_freeze": ranking_freeze,
        "predictions": predictions,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(output, report)
    return report


def main() -> None:
    report = freeze()
    print(
        json.dumps(
            {
                "output": str(FROZEN_PREDICTIONS.resolve()),
                "jobs": len(report["predictions"]),
                "groups": len(report["ranking_freeze"]),
                "report_sha256": report["report_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
