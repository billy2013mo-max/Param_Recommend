#!/usr/bin/env python3
"""Add the Qwen3.5-9B hybrid model to the V3 memory feature inventory.

Produces a fresh transfer-only sidecar that extends the frozen V3 inventory
with the Qwen3.5-9B row (the only stage-1 hybrid model absent from it).
Qwen3.5-4B and Qwen3.6-27B are already present.  This is a pure additive
artifact: the frozen base inventory and the experiment inventory are read
read-only, nothing existing is overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from common import ARTIFACT_DIR, read_json, sha256_json, write_json


DEFAULT_BASE = ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_v1.json"
DEFAULT_HYBRID = (
    ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_model_inventory_v1.json"
)
DEFAULT_OUTPUT = (
    ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_with_hybrid_v1.json"
)


def build(base_path: Path, hybrid_path: Path) -> dict:
    base = read_json(base_path)
    hybrid = read_json(hybrid_path)
    hybrid_models = hybrid.get("models", [])
    by_id = {str(m.get("id")): m for m in hybrid_models}
    if "qwen3p5_9b" not in by_id:
        raise ValueError("hybrid experiment inventory is missing qwen3p5_9b")
    model = dict(by_id["qwen3p5_9b"])
    if model.get("path") != "/wanqing-models/Qwen3.5-9B" or model.get("family") != "qwen3_5":
        raise ValueError("qwen3p5_9b identity drifted")
    existing = {str(row["id"]) for row in base["models"]}
    if model["id"] in existing:
        raise ValueError("qwen3p5_9b already exists in base inventory")
    report = {
        "schema_version": base["schema_version"],
        "catalog_sha256": sha256_json([*base["models"], model]),
        "selection_policy": (
            "Frozen V3 H800 catalog plus separately bound Qwen3.5 transfer-only "
            "diagnostic models (4B/9B/27B); this sidecar is not a production "
            "inventory replacement."
        ),
        "fixed_lora": base["fixed_lora"],
        "models": [*base["models"], model],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "hybrid_transfer_sidecar_only",
        "production_override_allowed": False,
        "source_bindings": {
            "frozen_v3_inventory": {
                "path": str(base_path.resolve()),
                "sha256": sha256_json(base.get("models", [])),
            },
            "hybrid_experiment_inventory": {
                "path": str(hybrid_path.resolve()),
                "sha256": sha256_json(hybrid_models),
            },
        },
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--hybrid", type=Path, default=DEFAULT_HYBRID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.base, args.hybrid)
    write_json(args.output, report)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
