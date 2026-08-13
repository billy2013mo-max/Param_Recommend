#!/usr/bin/env python3
"""Freeze exact holdout scores before the first holdout GPU job.

The input JSONL is produced by the selected shared throughput model and must
contain one row per holdout job with ``job_id``, ``predicted_ranking_score`` and
boolean ``admitted``.  This script validates and binds those scores; it never
fits a model and never reads a training result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from common import RESULTS_DIR, read_jsonl, sha256_file, sha256_json, write_json
from prepare_h800_packing_config_ranking_v1 import (
    HOLDOUT_PREDICTIONS,
    QUEUE_HOLDOUT,
)


SCHEMA = "sft_h800_packing_config_ranking_frozen_predictions/v2"


def freeze(
    input_path: Path,
    *,
    model_artifact: Path,
    model_implementation: Path,
    output: Path = HOLDOUT_PREDICTIONS,
) -> dict[str, Any]:
    for path in (input_path, model_artifact, model_implementation, QUEUE_HOLDOUT):
        if not path.is_file():
            raise FileNotFoundError(path)
    jobs = read_jsonl(QUEUE_HOLDOUT)
    expected = {str(job["job_id"]) for job in jobs}
    existing = [
        job_id
        for job_id in sorted(expected)
        if (RESULTS_DIR / job_id / "latest_attempt.json").is_file()
    ]
    if existing:
        raise ValueError(
            "holdout outcomes already exist; predictions can no longer be frozen "
            f"prospectively: {existing[:5]}"
        )
    source = read_jsonl(input_path)
    predictions: list[dict[str, Any]] = []
    observed: set[str] = set()
    for row in source:
        job_id = str(row.get("job_id") or "")
        score = row.get("predicted_ranking_score")
        admitted = row.get("admitted")
        if (
            job_id not in expected
            or job_id in observed
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not isinstance(admitted, bool)
        ):
            raise ValueError(f"invalid or duplicate prediction for {job_id}")
        observed.add(job_id)
        prediction = {
            "job_id": job_id,
            "predicted_ranking_score": float(score),
            "admitted": admitted,
        }
        for optional in (
            "predicted_memory_bytes",
            "memory_safety_margin_bytes",
            "prediction_notes",
        ):
            if optional in row:
                prediction[optional] = row[optional]
        predictions.append(prediction)
    if observed != expected:
        raise ValueError(
            f"prediction coverage drifted: missing={len(expected - observed)}, "
            f"unexpected={len(observed - expected)}"
        )
    order = {str(job["job_id"]): index for index, job in enumerate(jobs)}
    predictions.sort(key=lambda row: order[str(row["job_id"])])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_before_first_holdout_gpu_execution": True,
        "training_outcomes_read": False,
        "queue_binding": {
            "path": str(QUEUE_HOLDOUT.resolve()),
            "sha256": sha256_file(QUEUE_HOLDOUT),
            "jobs": len(jobs),
        },
        "model_binding": {
            "artifact_path": str(model_artifact.resolve()),
            "artifact_sha256": sha256_file(model_artifact),
            "implementation_path": str(model_implementation.resolve()),
            "implementation_sha256": sha256_file(model_implementation),
            "packing_specific_feature_or_coefficient": False,
            "packing_projection": "samples_per_pack.mean is substituted as virtual MBS",
        },
        "source_scores": {
            "path": str(input_path.resolve()),
            "sha256": sha256_file(input_path),
        },
        "predictions": predictions,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--model-implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HOLDOUT_PREDICTIONS)
    args = parser.parse_args()
    report = freeze(
        args.input,
        model_artifact=args.model_artifact,
        model_implementation=args.model_implementation,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "jobs": len(report["predictions"]),
                "model_binding": report["model_binding"],
                "report_sha256": report["report_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
