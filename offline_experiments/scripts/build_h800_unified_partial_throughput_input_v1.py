#!/usr/bin/env python3
"""Build the exact baseline-plus-unified-campaign throughput observation set.

The workspace-wide canonical export contains later experiments whose model
inventory contracts are intentionally outside the frozen throughput V5 input
domain.  For an apples-to-apples partial refit, keep the canonical observation
set used by the frozen V5 artifact and append only observations from the sealed
unified-resource campaign.  Observation IDs are de-duplicated and the source
files remain immutable.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any

from common import ARTIFACT_DIR, ROOT, read_jsonl, sha256_file, sha256_json, write_json, write_jsonl


SCHEMA = "sft_h800_unified_partial_throughput_input/v1"
CAMPAIGN_ID = "h800_unified_resource_evidence_20260809_v1"
DEFAULT_BASELINE = ARTIFACT_DIR / "canonical_h800_observations.jsonl"
DEFAULT_CURRENT_EXPORT = (
    ROOT
    / "diagnostics"
    / "h800_unified_resource_partial_refit_20260809"
    / "canonical_observations.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "diagnostics"
    / "h800_unified_resource_partial_refit_20260809"
    / "throughput_observations.jsonl"
)
DEFAULT_BASELINE_PROFILE_DIR = ARTIFACT_DIR / "dataset_profiles"
DEFAULT_OUTPUT_PROFILE_DIR = DEFAULT_OUTPUT.parent / "throughput_profiles"


def _campaign_id(row: dict[str, Any]) -> str | None:
    return (
        ((row.get("configuration") or {}).get("job") or {}).get("campaign_id")
    )


def _materialize_profiles(
    rows: list[dict[str, Any]],
    *,
    baseline_profile_dir: Path,
    output_profile_dir: Path,
) -> dict[str, dict[str, str]]:
    """Copy the exact flat profile set consumed by the throughput fitter."""

    source_candidates: dict[str, set[Path]] = {}
    for row in rows:
        job = ((row.get("configuration") or {}).get("job") or {})
        dataset_id = str(job.get("dataset_id") or "").strip()
        if not dataset_id:
            raise ValueError(f"observation lacks dataset_id: {row.get('observation_id')}")
        configured_path = job.get("dataset_profile_path")
        source_path = (
            Path(str(configured_path)).resolve()
            if configured_path
            else (
                baseline_profile_dir
                / f"{dataset_id}.qwen3_nothink.jsonl"
            ).resolve()
        )
        if not source_path.is_file():
            raise FileNotFoundError(
                f"dataset profile missing for {dataset_id}: {source_path}"
            )
        source_candidates.setdefault(dataset_id, set()).add(source_path)

    output_profile_dir.mkdir(parents=True, exist_ok=True)
    bindings: dict[str, dict[str, str]] = {}
    for dataset_id, paths in sorted(source_candidates.items()):
        hashes = {sha256_file(path): path for path in paths}
        if len(hashes) != 1:
            detail = {str(path): sha256_file(path) for path in sorted(paths)}
            raise ValueError(f"profile payload drift for {dataset_id}: {detail}")
        source_sha, source_path = next(iter(hashes.items()))
        target_path = output_profile_dir / f"{dataset_id}.qwen3_nothink.jsonl"
        if target_path.exists() and sha256_file(target_path) != source_sha:
            raise ValueError(f"refusing to overwrite drifted profile: {target_path}")
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
        bindings[dataset_id] = {
            "source_path": str(source_path),
            "source_sha256": source_sha,
            "materialized_path": str(target_path.resolve()),
            "materialized_sha256": sha256_file(target_path),
        }
    return bindings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--current-export", type=Path, default=DEFAULT_CURRENT_EXPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--baseline-profile-dir", type=Path, default=DEFAULT_BASELINE_PROFILE_DIR
    )
    parser.add_argument(
        "--output-profile-dir", type=Path, default=DEFAULT_OUTPUT_PROFILE_DIR
    )
    args = parser.parse_args()

    baseline = read_jsonl(args.baseline)
    current = [
        row for row in read_jsonl(args.current_export) if _campaign_id(row) == CAMPAIGN_ID
    ]
    if len(current) != 186:
        raise ValueError(f"unified campaign export drifted: {len(current)} != 186")
    if Counter((row.get("outcome") or {}).get("class") for row in current) != {
        "success": 152,
        "oom": 34,
    }:
        raise ValueError("unified campaign terminal outcome counts drifted")
    if sum(
        (row.get("outcome") or {}).get("usable_for_throughput_calibration") is True
        for row in current
    ) != 136:
        raise ValueError("unified campaign throughput-usable count drifted")

    combined: dict[str, dict[str, Any]] = {}
    for row in [*baseline, *current]:
        observation_id = str(row["observation_id"])
        existing = combined.get(observation_id)
        if existing is not None and existing != row:
            raise ValueError(f"observation payload collision: {observation_id}")
        combined[observation_id] = row
    ordered = sorted(combined.values(), key=lambda row: str(row["observation_id"]))
    write_jsonl(args.output, ordered)
    profile_bindings = _materialize_profiles(
        ordered,
        baseline_profile_dir=args.baseline_profile_dir,
        output_profile_dir=args.output_profile_dir,
    )
    manifest = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "path": str(args.baseline.resolve()),
            "sha256": sha256_file(args.baseline),
            "observations": len(baseline),
        },
        "current_export": {
            "path": str(args.current_export.resolve()),
            "sha256": sha256_file(args.current_export),
            "selected_campaign_id": CAMPAIGN_ID,
            "selected_observations": len(current),
            "throughput_usable_successes": 136,
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "observations": len(ordered),
        },
        "dataset_profiles": {
            "directory": str(args.output_profile_dir.resolve()),
            "count": len(profile_bindings),
            "bindings": profile_bindings,
        },
        "selection_policy": (
            "frozen V5 canonical baseline plus exact unified-resource campaign only"
        ),
        "publishable": False,
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(
        {
            "baseline": len(baseline),
            "current_campaign": len(current),
            "combined": len(ordered),
            "throughput_usable_current": 136,
            "dataset_profiles": len(profile_bindings),
            "output": str(args.output.resolve()),
        }
    )


if __name__ == "__main__":
    main()
