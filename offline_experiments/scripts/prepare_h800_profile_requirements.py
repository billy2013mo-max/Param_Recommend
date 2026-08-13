#!/usr/bin/env python3
"""Generate the CPU-only intake contract for fresh H800 profiles.

This file does not fabricate data or mark any profile fresh.  It turns the
holdout design into an explicit hand-off for the data/runtime owner: each row
must provide a scenario-level split, processor-bound static profile, source
data file, and a registration record consumed by the eventual materializer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from common import ARTIFACT_DIR, sha256_file, write_json


SCHEMA = "sft_h800_fresh_profile_requirements/v1"
DEFAULT_DESIGN = ARTIFACT_DIR / "h800_fresh_holdout_design_v1.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_fresh_profile_requirements_v1.json"


def build_requirements(*, design: Mapping[str, Any], design_path: Path) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for source in design.get("scenarios") or []:
        scenario = dict(source)
        freshness = scenario.get("freshness") or {}
        scenarios.append(
            {
                "scenario_id": scenario.get("scenario_id"),
                "dataset_profile_id": scenario.get("dataset_profile_id"),
                "model_id": scenario.get("model_id"),
                "cutoff_len": scenario.get("cutoff_len"),
                "target_gbs": scenario.get("target_gbs", 64),
                "scale_out_transition": scenario.get("scale_out_transition"),
                "required_bindings": {
                    "profile_path": None,
                    "profile_sha256": None,
                    "data_path": None,
                    "data_sha256": None,
                    "runtime_dataset_registered": False,
                    "processor_contract_sha256": None,
                    "split_manifest_sha256": None,
                },
                "required_profile_row_fields": [
                    "sample_id",
                    "total_tokens",
                    "label_tokens",
                    "turns",
                    "assistant_turns",
                ],
                "required_metadata": {
                    "schema": "sft_static_workload_profiles/v1",
                    "profile_id": scenario.get("dataset_profile_id"),
                    "model_id": scenario.get("model_id"),
                    "cutoff_len": scenario.get("cutoff_len"),
                    "tokenizer_id": "Qwen3 tokenizer + qwen3_nothink template",
                    "processor_path": "Qwen3 text processor/collator",
                    "fresh_split_unit": freshness.get("split_unit"),
                    "must_not_reuse_prior_holdout": True,
                },
            }
        )
    return {
        "schema": SCHEMA,
        "campaign_id": design.get("campaign_id"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_binding": {
            "path": str(design_path.resolve()),
            "sha256": sha256_file(design_path),
        },
        "freshness_policy": {
            "split_unit": "complete scenario/profile, not random rows",
            "must_be_disjoint_from": [
                "inspected challenger native holdout",
                "all rows used in physical-shares/v4b fitting",
            ],
            "profile_and_data_sha_must_be_frozen_before_gpu": True,
        },
        "required_processor_contract": {
            "text_family": "Qwen3",
            "template": "qwen3_nothink",
            "profile_schema": "sft_static_workload_profiles/v1",
            "runtime_registration_required": True,
        },
        "scenarios": scenarios,
        "ready_for_materialization": False,
        "publication_allowed": False,
        "gpu_training_started": False,
        "queues_mutated": False,
        "required_next_action": "data/runtime owner fills bindings and proves split disjointness",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    import json

    design = json.loads(args.design.read_text(encoding="utf-8"))
    requirements = build_requirements(design=design, design_path=args.design)
    write_json(args.output, requirements)
    print(f"wrote {args.output}; scenarios={len(requirements['scenarios'])}; ready_for_materialization=False")


if __name__ == "__main__":
    main()
