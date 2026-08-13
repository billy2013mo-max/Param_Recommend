#!/usr/bin/env python3
"""Build a transfer-only inventory adding Qwen3.5-4B to the frozen H800 catalog."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from common import ARTIFACT_DIR, read_json, sha256_file, sha256_json, write_json


DEFAULT_BASE = ARTIFACT_DIR / "model_inventory.json"
DEFAULT_Q35 = (
    ARTIFACT_DIR.parent
    / "campaigns"
    / "rtx4090_small_family_generalization_20260730"
    / "qwen3p5_4b"
    / "artifacts"
    / "model_inventory.json"
)
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_v1.json"


def build(base_path: Path, q35_path: Path) -> dict:
    base = read_json(base_path)
    q35 = read_json(q35_path)
    if len(q35.get("models") or []) != 1:
        raise ValueError("Qwen3.5 source inventory must contain exactly one model")
    model = dict(q35["models"][0])
    if (
        model.get("id") != "qwen3p5_4b"
        or model.get("path") != "/wanqing-models/Qwen3.5-4B"
        or model.get("family") != "qwen3_5"
    ):
        raise ValueError("Qwen3.5 transfer model identity drifted")
    ids = {str(row["id"]) for row in base["models"]}
    if model["id"] in ids:
        raise ValueError("Qwen3.5 transfer model already exists in base inventory")
    report = {
        "schema_version": base["schema_version"],
        "catalog_sha256": sha256_json([*base["models"], model]),
        "selection_policy": (
            "Frozen H800 catalog plus separately bound Qwen3.5-4B transfer-only "
            "diagnostic; this sidecar is not a production inventory replacement."
        ),
        "fixed_lora": base["fixed_lora"],
        "models": [*base["models"], model],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fresh_holdout_transfer_sidecar_only",
        "production_override_allowed": False,
        "source_bindings": {
            "frozen_h800_inventory": {
                "path": str(base_path.resolve()),
                "sha256": sha256_file(base_path),
            },
            "qwen35_4b_runtime_inventory": {
                "path": str(q35_path.resolve()),
                "sha256": sha256_file(q35_path),
            },
        },
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--qwen35", type=Path, default=DEFAULT_Q35)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.base, args.qwen35)
    write_json(args.output, report)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
