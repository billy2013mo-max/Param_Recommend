#!/usr/bin/env python3
"""Audit fresh S3 memory-acceptance candidates against prior local data.

The upstream candidate audit validates schema and token-length profiles.  This
second, CPU-only audit checks exact normalized SFT-row overlap against the data
that has already been available to earlier fitting, diagnostics, or business
validation.  It deliberately emits fingerprints and counts only; raw business
text is never copied into the report.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from common import ARTIFACT_DIR, ROOT, read_json, sha256_json, write_json


DEFAULT_CANDIDATE_AUDIT = (
    ARTIFACT_DIR / "h800_final_memory_acceptance_s3_candidate_audit_v1.json"
)
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_final_memory_acceptance_s3_independence_audit_v1.json"
DEFAULT_REFERENCE_ROOTS = (
    ROOT / "data" / "remote_candidate_screen_20260804",
    ROOT / "data" / "source_pools",
    ROOT / "data" / "bounded_memory_v2_fresh_holdout_v1",
    ROOT.parent / "real_business_validation_20260730" / "datasets",
)
REQUIRED_FIELDS = ("system", "prompt", "response")


def _iter_values(path: Path) -> Iterable[Any]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if isinstance(value, list):
                yield from value
            else:
                yield value


def _normalized_fingerprint(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    if not all(isinstance(value.get(field), str) for field in REQUIRED_FIELDS):
        return None
    normalized = "\0".join(str(value[field]).strip() for field in REQUIRED_FIELDS)
    return hashlib.sha256(normalized.encode()).hexdigest()


def _candidate_fingerprints(path: Path) -> set[str]:
    fingerprints = {
        fingerprint
        for value in _iter_values(path)
        if (fingerprint := _normalized_fingerprint(value)) is not None
    }
    if not fingerprints:
        raise ValueError(f"candidate has no normalized SFT rows: {path}")
    return fingerprints


def audit(candidate_audit_path: Path, reference_roots: tuple[Path, ...]) -> dict[str, Any]:
    candidate_audit = read_json(candidate_audit_path)
    candidates: dict[str, dict[str, Any]] = {}
    fingerprint_owners: dict[str, set[str]] = defaultdict(set)
    for row in candidate_audit["datasets"]:
        if row.get("accepted_pure_text_sft") is not True:
            continue
        dataset_id = str(row["dataset_id"])
        path = Path(row["local_path"])
        fingerprints = _candidate_fingerprints(path)
        candidates[dataset_id] = {
            "path": path,
            "fingerprints": fingerprints,
            "source_rows": int(row["raw_rows"]),
            "unique_rows": len(fingerprints),
            "file_sha256": str(row["file_sha256"]),
            "token_profile": row["token_profile"],
        }
        for fingerprint in fingerprints:
            fingerprint_owners[fingerprint].add(dataset_id)

    candidate_root = Path(candidate_audit["input_dir"]).resolve()
    reference_fingerprints: set[str] = set()
    reference_files = 0
    reference_values = 0
    reference_sft_rows = 0
    matched_files_by_dataset: dict[str, set[str]] = defaultdict(set)
    for root in reference_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            resolved = path.resolve()
            if candidate_root == resolved or candidate_root in resolved.parents:
                continue
            reference_files += 1
            relative = str(resolved)
            for value in _iter_values(path):
                reference_values += 1
                fingerprint = _normalized_fingerprint(value)
                if fingerprint is None:
                    continue
                reference_sft_rows += 1
                reference_fingerprints.add(fingerprint)
                for dataset_id in fingerprint_owners.get(fingerprint, ()):
                    matched_files_by_dataset[dataset_id].add(relative)

    cross_candidate_overlap: dict[str, dict[str, int]] = defaultdict(dict)
    dataset_ids = sorted(candidates)
    for left_index, left in enumerate(dataset_ids):
        for right in dataset_ids[left_index + 1 :]:
            shared = len(candidates[left]["fingerprints"] & candidates[right]["fingerprints"])
            if shared:
                cross_candidate_overlap[left][right] = shared
                cross_candidate_overlap[right][left] = shared

    rows = []
    for dataset_id in dataset_ids:
        candidate = candidates[dataset_id]
        fingerprints = candidate["fingerprints"]
        reference_overlap = len(fingerprints & reference_fingerprints)
        peer_containments = {
            peer: shared / len(fingerprints)
            for peer, shared in cross_candidate_overlap.get(dataset_id, {}).items()
        }
        max_peer_containment = max(peer_containments.values(), default=0.0)
        reference_containment = reference_overlap / len(fingerprints)
        rows.append(
            {
                "dataset_id": dataset_id,
                "local_path": str(candidate["path"].resolve()),
                "file_sha256": candidate["file_sha256"],
                "source_rows": candidate["source_rows"],
                "unique_normalized_rows": candidate["unique_rows"],
                "token_profile": candidate["token_profile"],
                "reference_overlap_rows": reference_overlap,
                "reference_containment": reference_containment,
                "matched_reference_files": sorted(matched_files_by_dataset[dataset_id]),
                "peer_overlap_rows": dict(sorted(cross_candidate_overlap.get(dataset_id, {}).items())),
                "max_peer_containment": max_peer_containment,
                "independent_candidate": (
                    reference_containment <= 0.05 and max_peer_containment <= 0.05
                ),
            }
        )

    report: dict[str, Any] = {
        "schema": "sft_h800_final_memory_acceptance_s3_independence_audit/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "candidate_audit": str(candidate_audit_path.resolve()),
        "reference_roots": [str(path.resolve()) for path in reference_roots],
        "reference_scan": {
            "jsonl_files": reference_files,
            "json_values": reference_values,
            "normalized_sft_rows": reference_sft_rows,
            "unique_normalized_sft_rows": len(reference_fingerprints),
        },
        "accepted_schema_candidates": len(rows),
        "independent_candidates": sum(row["independent_candidate"] is True for row in rows),
        "datasets": rows,
        "selection_contract": {
            "maximum_reference_containment": 0.05,
            "maximum_peer_containment": 0.05,
            "one_dataset_per_fully_overlapping_peer_component": True,
            "gpu_outcomes_observed": 0,
        },
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-audit", type=Path, default=DEFAULT_CANDIDATE_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-root", type=Path, action="append")
    args = parser.parse_args()
    roots = tuple(args.reference_root) if args.reference_root else DEFAULT_REFERENCE_ROOTS
    report = audit(args.candidate_audit, roots)
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "accepted_schema_candidates": report["accepted_schema_candidates"],
                "independent_candidates": report["independent_candidates"],
                "reference_scan": report["reference_scan"],
                "report_sha256": report["report_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
