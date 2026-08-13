#!/usr/bin/env python3
"""Select a deterministic, previously-unseen BS3 dataset screen for Packing.

The selector is intentionally metadata-only.  It chooses the newest published
revision for each opaque ``dataset-*`` ID, excludes IDs already mentioned by
the project, and balances the screen over remote file-size bands.  Content,
schema, token lengths, and source overlap are audited only after download.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LISTING = Path("/tmp/packing_final_bs3_datasets.tsv")
DEFAULT_OUTPUT = ROOT / "artifacts" / "h800_packing_business_screen_selection_v1.json"
DEFAULT_DOWNLOAD_LISTING = Path("/tmp/packing_final_business_candidates.tsv")
DATASET_ID_PATTERN = re.compile(r"dataset-[a-z0-9]+-[0-9]+")
PUBLISH_PATTERN = re.compile(
    r"^(?P<size>[0-9]+)\t"
    r"(?P<key>datasets/(?P<dataset_id>dataset-[a-z0-9]+-(?P<stamp>[0-9]+))/"
    r"(?P<revision>[0-9]+)/publish/[^/]+\.jsonl)$"
)
SIZE_BANDS = (
    ("tiny", 200_000, 750_000),
    ("small", 750_000, 2_000_000),
    ("medium", 2_000_000, 5_000_000),
    ("large", 5_000_000, 12_000_001),
)


def _referenced_dataset_ids() -> set[str]:
    ids: set[str] = set()
    for root in (ROOT / "artifacts", ROOT.parent / "项目文档"):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                ids.update(DATASET_ID_PATTERN.findall(path.read_text(errors="ignore")))
            except OSError:
                continue
    data_root = ROOT / "data"
    if data_root.exists():
        for path in data_root.rglob("dataset-*"):
            if path.is_dir() and DATASET_ID_PATTERN.fullmatch(path.name):
                ids.add(path.name)
    return ids


def _latest_rows(listing: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for line in listing.read_text(errors="replace").splitlines():
        match = PUBLISH_PATTERN.match(line)
        if not match:
            continue
        row: dict[str, Any] = match.groupdict()
        row["size_bytes"] = int(row.pop("size"))
        row["creation_stamp"] = int(row.pop("stamp"))
        row["revision"] = int(row["revision"])
        previous = latest.get(str(row["dataset_id"]))
        if previous is None or row["revision"] > previous["revision"]:
            latest[str(row["dataset_id"])] = row
    return list(latest.values())


def select(listing: Path, per_band: int) -> dict[str, Any]:
    excluded = _referenced_dataset_ids()
    rows = [row for row in _latest_rows(listing) if row["dataset_id"] not in excluded]
    selected: list[dict[str, Any]] = []
    for band, lower, upper in SIZE_BANDS:
        candidates = [row for row in rows if lower <= row["size_bytes"] < upper]
        candidates.sort(key=lambda row: (-row["creation_stamp"], row["dataset_id"]))
        seen_sizes: set[int] = set()
        chosen: list[dict[str, Any]] = []
        for row in candidates:
            if row["size_bytes"] in seen_sizes:
                continue
            seen_sizes.add(row["size_bytes"])
            chosen.append({**row, "size_band": band})
            if len(chosen) == per_band:
                break
        if len(chosen) != per_band:
            raise ValueError(f"not enough unique-size candidates in band {band}: {len(chosen)}")
        selected.extend(chosen)

    report: dict[str, Any] = {
        "schema": "h800_packing_business_screen_selection/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_listing": str(listing.resolve()),
        "selection_role": "download_screen_only_not_train_or_acceptance_assignment",
        "rules": {
            "publish_extension": ".jsonl",
            "latest_revision_per_dataset_id": True,
            "exclude_any_project_referenced_dataset_id": True,
            "sort_within_band": "creation_stamp_desc_then_dataset_id",
            "deduplicate_exact_file_size_within_band": True,
            "per_band": per_band,
            "size_bands_half_open_bytes": {
                name: [lower, upper] for name, lower, upper in SIZE_BANDS
            },
        },
        "excluded_dataset_id_count": len(excluded),
        "eligible_latest_unreferenced_count": len(rows),
        "selected_file_count": len(selected),
        "selected_bytes": sum(row["size_bytes"] for row in selected),
        "selected": selected,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listing", type=Path, default=DEFAULT_LISTING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--download-listing", type=Path, default=DEFAULT_DOWNLOAD_LISTING)
    parser.add_argument("--per-band", type=int, default=16)
    args = parser.parse_args()

    report = select(args.listing, args.per_band)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    args.download_listing.write_text(
        "".join(f"{row['size_bytes']}\t{row['key']}\n" for row in report["selected"])
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "download_listing": str(args.download_listing.resolve()),
                "selected_file_count": report["selected_file_count"],
                "selected_bytes": report["selected_bytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
