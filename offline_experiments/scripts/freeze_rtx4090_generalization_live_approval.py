#!/usr/bin/env python3
"""Freeze and optionally promote one live RTX 4090 prospective queue.

Copy this file to the root of an isolated campaign and invoke it with that
campaign's configured training interpreter after live provenance is captured.
It never reuses an approval from another hardware/runtime cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from approval_gate import build_provenance_binding, build_queue_binding  # noqa: E402
from common import read_json, read_jsonl, sha256_file, sha256_json, write_json  # noqa: E402
from promote_approval_candidate import promote_candidate  # noqa: E402
from run_job import live_runtime_identity, live_runtime_patch  # noqa: E402


AUTHORIZATION = (
    "用户明确要求在空闲的 RTX 4090 GPU 0,1,2,3 上执行已冻结的泛化实验；"
    "不得终止、接管或干扰其他人的进程。"
)


def freeze(queue: Path, stage: str) -> Path:
    queue = queue.resolve()
    rows = read_jsonl(queue)
    campaign_id = str(
        read_json(PROJECT_ROOT / "config" / "experiment.json")[
            "campaign_id"
        ]
    )
    queue_binding = build_queue_binding(queue, rows, PROJECT_ROOT)
    provenance_binding = build_provenance_binding(PROJECT_ROOT)
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    ids = [str(row["job_id"]) for row in rows]
    manifest_paths = [
        queue,
        PROJECT_ROOT / "artifacts" / "provenance.json",
        PROJECT_ROOT
        / "artifacts"
        / "frozen_predictions_before_holdout.json",
        PROJECT_ROOT / "EXPERIMENT_DESIGN.md",
    ]
    manifest = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in manifest_paths
    }
    stage_binding = {
        "allowed_job_ids": ids,
        "queue_path": queue_binding["path"],
        "queue_sha256": queue_binding["sha256"],
        "ordered_job_ids": queue_binding["ordered_job_ids"],
        "ordered_job_payload_sha256": queue_binding[
            "ordered_job_payload_sha256"
        ],
        "job_payload_sha256": queue_binding["job_payload_sha256"],
    }
    design = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{campaign_id}:{stage}",
        "file_sha256": manifest,
        "execution_order": [stage],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": [0, 1, 2, 3],
        "max_gpu_count": 4,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        # The generic validator retains this historical field name.  It is an
        # exact mirror of the bound prospective queue, not a semantic claim.
        "throughput_screen_delta": stage_binding,
    }
    candidate = (
        PROJECT_ROOT
        / "artifacts"
        / f"approval_design_{stage}_candidate.json"
    )
    write_json(candidate, design)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    candidate = freeze(args.queue, args.stage)
    digest = sha256_file(candidate)
    report = promote_candidate(
        candidate_path=candidate,
        expected_candidate_sha256=digest,
        project_root=PROJECT_ROOT,
        authorization=AUTHORIZATION,
        approved_by="user",
        promote=args.promote,
    )
    print(
        json.dumps(
            {
                "candidate": str(candidate),
                "candidate_sha256": digest,
                "promotion": report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report.get("all_passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
