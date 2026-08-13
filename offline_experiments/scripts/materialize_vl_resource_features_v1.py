#!/usr/bin/env python3
"""Materialize frozen VL memory/throughput feature rows for candidate MBS values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, read_json, sha256_file, sha256_json, write_json, write_jsonl
from vl_resource_features import build_vl_resource_features


PROFILE_MANIFEST = ARTIFACT_DIR / "h800_vl_business_workload_profiles_manifest_v2.json"
MODEL_INVENTORY = ARTIFACT_DIR / "h800_qwen35_vl_supplement_model_inventory_v1.json"
OUTPUT = ARTIFACT_DIR / "h800_vl_resource_features_v1.jsonl"
OUTPUT_MANIFEST = ARTIFACT_DIR / "h800_vl_resource_features_manifest_v1.json"
MBS_VALUES = (1, 4)


def materialize() -> dict[str, Any]:
    profiles = read_json(PROFILE_MANIFEST)
    models = {str(row["id"]): row for row in read_json(MODEL_INVENTORY)["models"]}
    rows = []
    for binding in profiles["profiles"]:
        path = Path(binding["path"])
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"workload profile drifted: {path}")
        profile = read_json(path)
        model_id = str(binding["model_id"])
        for mbs in MBS_VALUES:
            features = build_vl_resource_features(
                profile,
                models[model_id],
                physical_mbs=mbs,
                freeze_vision_tower=True,
                freeze_multi_modal_projector=True,
            )
            rows.append(
                {
                    "model_id": model_id,
                    "source": binding["source"],
                    "modality": binding["modality"],
                    "tier": binding["tier"],
                    "physical_mbs": mbs,
                    "workload_profile_path": str(path.resolve()),
                    "workload_profile_sha256": binding["sha256"],
                    "features": features,
                }
            )
    write_jsonl(OUTPUT, rows)
    report: dict[str, Any] = {
        "schema": "sft_h800_vl_resource_features_manifest/v1",
        "profile_manifest": {
            "path": str(PROFILE_MANIFEST.resolve()),
            "sha256": sha256_file(PROFILE_MANIFEST),
            "report_sha256": profiles["report_sha256"],
        },
        "model_inventory": {
            "path": str(MODEL_INVENTORY.resolve()),
            "sha256": sha256_file(MODEL_INVENTORY),
        },
        "features": {
            "path": str(OUTPUT.resolve()),
            "sha256": sha256_file(OUTPUT),
            "rows": len(rows),
            "mbs_values": list(MBS_VALUES),
        },
        "shared_text_coefficients_modified": False,
        "vl_coefficients_fitted": False,
        "gpu_training_started": False,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT_MANIFEST, report)
    return report


def main() -> None:
    report = materialize()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
