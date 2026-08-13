#!/usr/bin/env python3
"""Build a versioned VL workload profile from normalized/sample JSONL rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vl_workload_profile import build_workload_profile


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain JSON objects")
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--processor-name", required=True)
    parser.add_argument("--processor-version", required=True)
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--spatial-merge-size", type=int, required=True)
    parser.add_argument("--temporal-patch-size", type=int, default=1)
    parser.add_argument("--image-min-pixels", type=int)
    parser.add_argument("--image-max-pixels", type=int)
    parser.add_argument("--default-task-family", default="causal_vl_sft")
    args = parser.parse_args()

    profile = build_workload_profile(
        _read_jsonl(args.input),
        model_id=args.model_id,
        processor_name=args.processor_name,
        processor_version=args.processor_version,
        patch_size=args.patch_size,
        spatial_merge_size=args.spatial_merge_size,
        temporal_patch_size=args.temporal_patch_size,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        default_task_family=args.default_task_family,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(profile["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

