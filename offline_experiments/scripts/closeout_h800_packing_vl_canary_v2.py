#!/usr/bin/env python3
"""Close out the Packing/VL canaries without reinterpreting either stage."""

from __future__ import annotations

import json

from common import ARTIFACT_DIR, sha256_file, sha256_json, write_json


PACKING = ARTIFACT_DIR / "h800_packing_semantic_canary_acceptance_v2_prefetch_corrected.json"
VL = ARTIFACT_DIR / "h800_vl_media_canary_acceptance_v1.json"
V1_COMBINED = ARTIFACT_DIR / "h800_packing_vl_canary_acceptance_v1.json"
OUTPUT = ARTIFACT_DIR / "h800_packing_vl_canary_closeout_v2.json"


def main() -> None:
    packing = json.loads(PACKING.read_text(encoding="utf-8"))
    vl = json.loads(VL.read_text(encoding="utf-8"))
    v1 = json.loads(V1_COMBINED.read_text(encoding="utf-8"))
    if (
        packing.get("all_passed") is not True
        or vl.get("all_passed") is not True
        or v1.get("all_passed") is not False
    ):
        raise ValueError("stage reports do not describe the exact transparent v2 closeout")
    report = {
        "schema": "sft_h800_packing_vl_canary_closeout/v2",
        "campaign_id": "h800_packing_vl_canary_20260803_v1",
        "packing": {
            "path": str(PACKING.resolve()),
            "sha256": sha256_file(PACKING),
            "report_sha256": packing["report_sha256"],
            "all_passed": True,
            "observed_sample_gbs": {
                row["packing_treatment"]: row["observed_sample_gbs"]
                for row in packing["corrections"]
            },
            "note": "v1 counted one collated but unconsumed batch; v2 removes exactly that prefetch and leaves the 5% gate unchanged",
        },
        "vl": {
            "path": str(VL.resolve()),
            "sha256": sha256_file(VL),
            "report_sha256": vl["report_sha256"],
            "all_passed": True,
            "models": [
                {
                    "job_id": result["job_id"],
                    "classification": result["classification"],
                    "all_ranks_real_image_and_freeze_passed": next(
                        row["all_ranks_passed"]
                        for row in vl["media_and_freeze_checks"]
                        if row["job_id"] == result["job_id"]
                    ),
                }
                for result in vl["results"]
            ],
        },
        "v1_combined_report_retained": {
            "path": str(V1_COMBINED.resolve()),
            "sha256": sha256_file(V1_COMBINED),
            "all_passed": False,
            "reason": "it intentionally retains the prefetch-biased Packing v1 verdict",
        },
        "all_passed": True,
        "fit_allowed": False,
        "recommendation_release_allowed": False,
        "next_stage_allowed": "packing_and_vl_calibration_materialization",
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
