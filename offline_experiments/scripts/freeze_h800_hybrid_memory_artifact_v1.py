#!/usr/bin/env python3
"""Freeze the stage-1 dense hybrid memory coefficients into an artifact.

The artifact holds the fitted dense_hybrid coefficients (intercept + 9
structural byte terms + the ZeRO-3 saved-activation interaction) plus the
model ids the hybrid channel applies to.  It binds the fit report SHA so the
coefficients are traceable to the exact fit that produced them.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from common import ARTIFACT_DIR, read_json, sha256_file, sha256_json, write_json

DEFAULT_FIT_REPORT = (
    ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_fit_report_v1.json"
)
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_hybrid_memory_artifact_v1.json"

HYBRID_MODEL_IDS = ("qwen3p5_4b", "qwen3p5_9b", "qwen3_6_27b")


def freeze(fit_report_path: Path) -> dict:
    report = read_json(fit_report_path)
    route_fit = report["route_fits"]["dense_hybrid_attention"]
    coefficients = route_fit["coefficients_by_name"]
    expected = {
        "intercept",
        "state",
        "saved_full",
        "saved_linear",
        "recompute",
        "full_workspace",
        "linear_workspace",
        "linear_state",
        "logits",
        "zero_workspace",
        "zero3_saved",
    }
    if set(coefficients) != expected:
        raise ValueError(
            f"dense_hybrid coefficients drifted: {set(coefficients) ^ expected}"
        )
    if not report.get("all_gates_passed"):
        raise ValueError("cannot freeze hybrid artifact from a fit that failed gates")
    artifact = {
        "schema": "sft_h800_hybrid_memory_artifact/v1",
        "model_ids": list(HYBRID_MODEL_IDS),
        "coefficients_by_name": coefficients,
        "fit_report": {
            "path": str(fit_report_path.resolve()),
            "sha256": sha256_file(fit_report_path),
        },
        "cv_metrics": {
            route: report["cross_validation"][route]["pooled_held_out"]
            for route in report["cross_validation"]
        },
        "capacity_bytes": 150142189568,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "hybrid_shadow_candidate",
        "production_admission_allowed": False,
    }
    artifact["report_sha256"] = sha256_json(artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-report", type=Path, default=DEFAULT_FIT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = freeze(args.fit_report)
    write_json(args.output, artifact)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
