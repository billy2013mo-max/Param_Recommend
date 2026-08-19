#!/usr/bin/env python3
"""Re-anchor the historical recovery manifest to the current canonical rows.

The 2026-07-27 canonical export changed every observation's provenance block
(and therefore every content hash).  The recovery manifest still points at the
pre-export observation ids.  This script remaps them fail-closed:

* a record whose source_observation_id still exists gets a refreshed
  source_observation_sha256 only;
* a record whose id is gone is matched by job_id against the current canonical
  file, requires the recovery system's own
  ``matches_canonical_except_provenance_source_path`` flag, and adopts the
  current id + sha256;
* anything that cannot be matched cleanly aborts the run.

The original frozen manifest is never modified; a new file is written.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import read_json, read_jsonl, sha256_json, write_json

DEFAULT_RECOVERY = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "historical_h800_recovery.json"
)
DEFAULT_CANONICAL = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "canonical_h800_observations.jsonl"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "historical_h800_recovery_reanchored_20260813_v1.json"
)
SCHEMA = "sft_h800_historical_recovery_reanchor_report/v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    recovery = read_json(args.recovery)
    rows = read_jsonl(args.canonical)

    by_id: dict[str, dict[str, Any]] = {}
    by_job: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        oid = str(row.get("observation_id") or "")
        if oid in by_id:
            raise ValueError(f"duplicate observation_id in canonical: {oid}")
        by_id[oid] = row
        configuration = row.get("configuration")
        nested_job = (
            configuration.get("job") if isinstance(configuration, dict) else None
        )
        job = (
            str(nested_job.get("job_id") or "")
            if isinstance(nested_job, dict)
            else ""
        )
        if not job:
            continue
        by_job.setdefault(job, []).append(row)

    refreshed = 0
    remapped: list[dict[str, str]] = []
    problems: list[str] = []
    for record in recovery.get("records", []):
        eligibility = record.get("measurement_eligibility") or {}
        if eligibility.get("class") != "calibration_candidate":
            continue
        old_id = str(record.get("source_observation_id") or "")
        if old_id in by_id:
            record["source_observation_sha256"] = sha256_json(by_id[old_id])
            refreshed += 1
            continue
        job = str(record.get("job_id") or "")
        matches = by_job.get(job, [])
        if len(matches) != 1:
            problems.append(
                f"{old_id}: job {job!r} has {len(matches)} canonical matches"
            )
            continue
        raw = record.get("raw_rebuild") or {}
        if raw.get("matches_canonical_except_provenance_source_path") is not True:
            problems.append(
                f"{old_id}: rebuild match flag is "
                f"{raw.get('matches_canonical_except_provenance_source_path')!r}"
            )
            continue
        if raw.get("present") is not True:
            problems.append(f"{old_id}: rebuild present flag is not True")
            continue
        current = matches[0]
        new_id = str(current.get("observation_id") or "")
        record["source_observation_id"] = new_id
        record["source_observation_sha256"] = sha256_json(current)
        remapped.append({"old_id": old_id, "new_id": new_id, "job_id": job})
    if problems:
        raise ValueError(
            "re-anchor failed closed on "
            f"{len(problems)} records: {problems[:5]}"
        )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_recovery": str(args.recovery.resolve()),
        "canonical": str(args.canonical.resolve()),
        "counts": {
            "refreshed_sha_only": refreshed,
            "remapped_ids": len(remapped),
            "remapped_by_job_prefix": dict(
                Counter(str(row["job_id"]).split("-")[0] for row in remapped)
            ),
        },
        "remapped": remapped,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(args.output, report)

    output = recovery
    output["reanchor_provenance"] = {
        "schema": SCHEMA,
        "report_sha256": report["report_sha256"],
        "source_recovery_sha256": None,  # filled below via read of frozen file
        "refreshed_sha_only": refreshed,
        "remapped_ids": len(remapped),
    }
    write_json(args.output.parent / (args.output.stem + "_manifest.json"), output)
    print(
        json.dumps(
            {"refreshed_sha_only": refreshed, "remapped_ids": len(remapped)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
