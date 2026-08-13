#!/usr/bin/env python3
"""Summarize and pre-declare the unfinished H800 four-card rank closeout.

This is a read-only design tool.  It never reruns the old shell scripts and it
does not treat a different GPU SKU as a substitute for the H800 evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from common import ARTIFACT_DIR, ROOT, sha256_file, write_json
from prepare_h800_prospective_holdout import probe_hardware


SCHEMA = "sft_h800_four_card_rank_closeout_design/v1"
DEFAULT_VALIDATION_ROOT = ROOT.parent / "qwen3_14b_4gpu_rank_validation_20260730"
DEFAULT_GPU_IDS = (4, 5, 6, 7)
REVERSE_RETEST_GAP = 0.03

CANDIDATES = (
    ("rank1_z3_gc_mbs16_ga2", 1),
    ("rank2_z2_gc_mbs8_ga4", 2),
    ("rank3_z3_gc_mbs8_ga4", 3),
)


def _meta(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _step_throughputs(log_path: Path) -> list[float]:
    pattern = re.compile(r"'train_tokens_per_second':\s*([0-9]+(?:\.[0-9]+)?)")
    return [float(match.group(1)) for match in pattern.finditer(log_path.read_text(errors="replace"))]


def summarize_validation(root: Path = DEFAULT_VALIDATION_ROOT) -> dict[str, Any]:
    """Read the existing closeout directory without changing it."""

    if not root.is_dir():
        raise FileNotFoundError(f"rank validation directory is absent: {root}")
    rows: list[dict[str, Any]] = []
    for tag, predicted_rank in CANDIDATES:
        meta_path = root / "logs" / f"{tag}_meta.txt"
        log_path = root / "logs" / f"{tag}.log"
        yaml_path = root / f"{tag}.yaml"
        if not (meta_path.is_file() and log_path.is_file() and yaml_path.is_file()):
            raise FileNotFoundError(f"incomplete closeout files for {tag}")
        meta = _meta(meta_path)
        throughputs = _step_throughputs(log_path)
        rows.append(
            {
                "tag": tag,
                "predicted_rank": predicted_rank,
                "yaml": {"path": str(yaml_path.resolve()), "sha256": sha256_file(yaml_path)},
                "meta": {
                    key: meta.get(key)
                    for key in ("cuda_visible_devices", "master_port", "start_utc", "end_utc", "exit_code")
                },
                "exit_code": int(meta["exit_code"]) if meta.get("exit_code", "").lstrip("-").isdigit() else None,
                "measured_steps": len(throughputs),
                "last_train_tokens_per_second": throughputs[-1] if throughputs else None,
                "terminal_success": meta.get("exit_code") == "0" and len(throughputs) >= 45,
                "log": {"path": str(log_path.resolve()), "sha256": sha256_file(log_path)},
            }
        )
    rank2 = next(row for row in rows if row["tag"] == "rank2_z2_gc_mbs8_ga4")
    rank3 = next(row for row in rows if row["tag"] == "rank3_z3_gc_mbs8_ga4")
    left = rank2["last_train_tokens_per_second"]
    right = rank3["last_train_tokens_per_second"]
    relative_gap = None
    if left and right:
        relative_gap = abs(float(left) - float(right)) / max(float(left), float(right))
    reruns: list[dict[str, Any]] = []
    rank1 = next(row for row in rows if row["tag"] == "rank1_z3_gc_mbs16_ga2")
    if rank1["terminal_success"] is not True:
        reruns.append(
            {
                "rerun_id": "rank1_recovery",
                "tag": rank1["tag"],
                "reason": "predicted winner did not complete all 45 optimizer steps",
                "order": 1,
            }
        )
    if relative_gap is not None and relative_gap < REVERSE_RETEST_GAP:
        reruns.extend(
            [
                {
                    "rerun_id": "rank2_reverse_retest",
                    "tag": rank2["tag"],
                    "reason": "rank2/rank3 gap is below the predeclared 3% repeat threshold",
                    "order": 2,
                },
                {
                    "rerun_id": "rank3_reverse_retest",
                    "tag": rank3["tag"],
                    "reason": "rank2/rank3 gap is below the predeclared 3% repeat threshold",
                    "order": 3,
                },
            ]
        )
    return {
        "validation_root": str(root.resolve()),
        "expected_ranking_path": str((root / "EXPECTED_RANKING.md").resolve()),
        "candidates": rows,
        "rank2_rank3_relative_gap": relative_gap,
        "reverse_retest_threshold": REVERSE_RETEST_GAP,
        "required_reruns": reruns,
        "closeout_status": "pending" if reruns else "ready_for_review",
    }


def build_closeout_design(
    *,
    validation_root: Path = DEFAULT_VALIDATION_ROOT,
    required_gpu_ids: Sequence[int] = DEFAULT_GPU_IDS,
    hardware_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summarize_validation(validation_root)
    probe = dict(
        hardware_probe
        if hardware_probe is not None
        else probe_hardware(required_gpu_ids=required_gpu_ids)
    )
    hardware_ready = probe.get("selected_pool_idle") is True
    return {
        "schema": SCHEMA,
        "campaign_id": "h800_qwen3_14b_four_card_rank_closeout_20260731",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "queues_mutated": False,
        "automatic_execution_allowed": False,
        "hardware_gate": {
            "required_gpu_ids": [int(value) for value in required_gpu_ids],
            "expected_gpu_name_contains": "H800",
            "probe": probe,
            "passed": hardware_ready,
        },
        "design_implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "campaign_gate": {
            "path": str((ROOT / "scripts" / "check_h800_campaign_gate.py").resolve()),
            "sha256": sha256_file(ROOT / "scripts" / "check_h800_campaign_gate.py"),
        },
        "source_evidence": summary,
        "required_before_launch": [
            "hardware_gate_passed",
            "new approval design binds exact YAML/logging/runtime sources",
            "rank1 recovery and rank2/rank3 reverse retest are submitted as a new attempt",
            "terminal summaries and runtime fingerprints are complete",
        ],
        "launch_status": (
            "blocked_by_hardware_or_approval"
            if not hardware_ready or summary["required_reruns"]
            else "pending_new_approval"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "h800_four_card_rank_closeout_design_v1.json",
    )
    args = parser.parse_args()
    design = build_closeout_design(validation_root=args.validation_root)
    write_json(args.output, design)
    print(
        f"wrote {args.output}; status={design['launch_status']}; "
        f"reruns={len(design['source_evidence']['required_reruns'])}"
    )


if __name__ == "__main__":
    main()
