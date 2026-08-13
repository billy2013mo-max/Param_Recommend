#!/usr/bin/env python3
"""Merge the matched ZeRO/GC contrast into the prior extension evidence.

The interaction-term sweep takes a single extension evaluation plus its frozen
predictions.  The matched-contrast batch on its own contains only three
scenarios, which is too few for the sweep's leave-one-scenario-out validation to
mean anything, so it must be evaluated together with the six scenarios from the
2026-08-05 extension rather than in isolation.

This script produces that merged pair of files.  It is offline and read-only
with respect to every input: it fits nothing, launches no GPU work, and writes
only the two merged artifacts under diagnostics/.

Both source batches sealed their predictions before their own GPU outcomes, and
each row keeps the prediction that was sealed for it, so merging does not
weaken the prospective property of either batch.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, ROOT, read_json, sha256_file, sha256_json, write_json

PRIOR_EVAL = ARTIFACT_DIR / "v5_dataset_candidate_extension_h800_20260805.json"
PRIOR_PRED = (
    ARTIFACT_DIR / "h800_v5_dataset_candidate_extension_frozen_predictions_v1.json"
)
MATCHED_EVAL = ARTIFACT_DIR / "h800_zero_gc_matched_contrast_results_v1.json"
MATCHED_PRED = (
    ARTIFACT_DIR / "h800_zero_gc_matched_contrast_frozen_predictions_v1.json"
)
OUT_DIR = ROOT / "diagnostics" / "zero_gc_matched_contrast_merge"
DEFAULT_EVAL_OUT = OUT_DIR / "merged_extension_evaluation_v1.json"
DEFAULT_PRED_OUT = OUT_DIR / "merged_frozen_predictions_v1.json"


def _binding(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-evaluation", type=Path, default=PRIOR_EVAL)
    parser.add_argument("--prior-predictions", type=Path, default=PRIOR_PRED)
    parser.add_argument("--matched-evaluation", type=Path, default=MATCHED_EVAL)
    parser.add_argument("--matched-predictions", type=Path, default=MATCHED_PRED)
    parser.add_argument("--evaluation-output", type=Path, default=DEFAULT_EVAL_OUT)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PRED_OUT)
    args = parser.parse_args()

    prior_eval = read_json(args.prior_evaluation)
    prior_pred = read_json(args.prior_predictions)
    matched_eval = read_json(args.matched_evaluation)
    matched_pred = read_json(args.matched_predictions)

    if matched_eval.get("state_counts", {}).get("success_non_authoritative"):
        raise ValueError("matched batch contains non-authoritative rows")

    prior_rows = list(prior_eval["rows"])
    matched_rows = list(matched_eval["rows"])
    for row in prior_rows:
        row.setdefault("batch", "v5_dataset_candidate_extension_20260805")
    for row in matched_rows:
        row["batch"] = "zero_gc_matched_contrast_20260806"

    prior_ids = {str(row["job_id"]) for row in prior_rows}
    matched_ids = {str(row["job_id"]) for row in matched_rows}
    overlap = prior_ids & matched_ids
    if overlap:
        raise ValueError(f"job id collision between batches: {sorted(overlap)[:5]}")

    merged_predictions = {**prior_pred["predictions"], **matched_pred["predictions"]}
    merged_rows = [*prior_rows, *matched_rows]
    covered = {str(row["job_id"]) for row in merged_rows}
    missing = covered - set(merged_predictions)
    if missing:
        raise ValueError(f"rows without a sealed prediction: {sorted(missing)[:5]}")

    scenarios = sorted({str(row["scenario_id"]) for row in merged_rows})
    authoritative = [
        row for row in merged_rows if row.get("state") == "success_authoritative"
    ]

    evaluation = {
        "schema": "sft_merged_extension_evaluation/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "model_refit": False,
        "publishable": False,
        "purpose": (
            "single evaluation input for the interaction-term sweep, combining "
            "the 2026-08-05 extension with the 2026-08-06 matched ZeRO/GC contrast"
        ),
        "sources": {
            "prior_extension": _binding(args.prior_evaluation),
            "prior_predictions": _binding(args.prior_predictions),
            "matched_contrast": _binding(args.matched_evaluation),
            "matched_predictions": _binding(args.matched_predictions),
        },
        "composition": {
            "prior_rows": len(prior_rows),
            "matched_rows": len(matched_rows),
            "total_rows": len(merged_rows),
            "authoritative_rows": len(authoritative),
            "scenarios": len(scenarios),
            "scenario_ids": scenarios,
        },
        "provenance_note": (
            "each row carries the prediction sealed by its own batch before that "
            "batch ran, so the merge does not weaken either batch's prospective "
            "property; the matched batch is nonetheless a designed contrast, not a "
            "fresh release holdout"
        ),
        "rows": merged_rows,
    }
    evaluation["report_sha256"] = sha256_json(evaluation)

    predictions = {
        "schema": "sft_merged_frozen_predictions/v1",
        "generated_at_utc": evaluation["generated_at_utc"],
        "temporal_status": {
            "new_24_predictions_frozen_before_new_gpu_outcomes": True,
            "predictions_frozen_before_gpu_outcomes": True,
            "merged_from_two_sealed_batches": True,
        },
        "sources": {
            "prior_predictions": _binding(args.prior_predictions),
            "matched_predictions": _binding(args.matched_predictions),
        },
        "job_count": len(merged_predictions),
        "predictions": merged_predictions,
    }
    predictions["report_sha256"] = sha256_json(predictions)

    write_json(args.evaluation_output, evaluation)
    write_json(args.predictions_output, predictions)
    print(
        json.dumps(
            {
                "evaluation_output": str(args.evaluation_output.resolve()),
                "predictions_output": str(args.predictions_output.resolve()),
                "composition": evaluation["composition"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
