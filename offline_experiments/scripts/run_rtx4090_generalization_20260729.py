#!/usr/bin/env python3
"""Execute the frozen RTX 4090 prospective campaign when preflight is ready."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from common import read_json, read_jsonl, sha256_file, sha256_json, write_json
from prepare_rtx4090_generalization_20260729 import DEFAULT_CAMPAIGN_ROOT
from rtx4090_generalization_preflight import build_report as preflight
from rtx4090_generalization_preflight import _hardware


SCHEMA = "sft_rtx4090_generalization_orchestrator/v1"
PHASES = (
    ("qwen3_8b", "compatibility_canary"),
    ("qwen3p5_4b", "compatibility_canary"),
    ("qwen3_8b", "memory_boundary"),
    ("qwen3p5_4b", "memory_boundary"),
    ("qwen3_8b", "multi_candidate_ranking"),
    ("qwen3p5_4b", "multi_candidate_ranking"),
    ("qwen3_8b", "packing_abba"),
)


def _state_path(campaign_root: Path) -> Path:
    return campaign_root / "runtime" / "orchestrator_state.json"


def _set_state(
    campaign_root: Path,
    state: str,
    **details: Any,
) -> None:
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(_state_path(campaign_root), report)


def _environment(root: Path) -> tuple[Path, dict[str, str]]:
    fixed = read_json(root / "config" / "experiment.json")[
        "fixed_runtime"
    ]
    python = Path(fixed["python"])
    env = dict(os.environ)
    overlay = fixed.get("environment_overlay") or {}
    prefixes = [str(value) for value in overlay.get("PYTHONPATH_prepend") or []]
    existing = env.get("PYTHONPATH")
    if prefixes:
        env["PYTHONPATH"] = os.pathsep.join(
            [*prefixes, *([existing] if existing else [])]
        )
    for key in ("FLA_TILELANG", "TILELANG_CACHE_DIR"):
        if overlay.get(key) is not None:
            env[key] = str(overlay[key])
    return python, env


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {result.returncode}: {command}"
        )
    return {
        "command": command,
        "return_code": result.returncode,
    }


def _capture_provenance(root: Path) -> dict[str, Any]:
    python, env = _environment(root)
    result = _run(
        [str(python), str(root / "scripts" / "capture_provenance.py")],
        cwd=root,
        env=env,
    )
    path = root / "artifacts" / "provenance.json"
    result.update(
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "runtime_fingerprint_sha256": read_json(path)[
                "runtime_fingerprint_sha256"
            ],
        }
    )
    return result


def _phase_result(
    root: Path,
    queue: Path,
) -> dict[str, Any]:
    rows = read_jsonl(queue)
    outcomes = {}
    complete_metrics = {}
    for row in rows:
        job_id = str(row["job_id"])
        result_root = root / "results" / job_id
        status_path = result_root / "status.json"
        if not status_path.is_file():
            outcomes[job_id] = "missing"
            complete_metrics[job_id] = False
            continue
        status = read_json(status_path)
        outcomes[job_id] = str(
            status.get("classification") or "unknown"
        )
        summaries = list(
            (result_root / "metrics").glob("summary.rank*.json")
        )
        complete_metrics[job_id] = (
            outcomes[job_id] == "success"
            and len(summaries) == int(row["gpu_count"])
        )
    return {
        "jobs": len(rows),
        "outcomes": outcomes,
        "complete_metrics": complete_metrics,
    }


def _run_phase(
    campaign_root: Path,
    model_id: str,
    phase: str,
) -> dict[str, Any]:
    root = campaign_root / model_id
    queue = root / "matrix" / f"queue_{phase}.jsonl"
    hardware = _hardware()
    if not hardware["selected_pool_idle"]:
        raise RuntimeError(
            "GPU 0-3 are not an idle four-card RTX 4090 pool; "
            "the orchestrator will not touch any process"
        )
    python, env = _environment(root)
    _run(
        [
            str(python),
            str(root / "freeze_live_approval.py"),
            "--queue",
            str(queue),
            "--stage",
            phase,
            "--promote",
        ],
        cwd=root,
        env=env,
    )
    _run(
        [
            str(python),
            str(root / "scripts" / "scheduler.py"),
            "--input",
            str(queue),
            "--execute",
        ],
        cwd=root,
        env=env,
    )
    result = _phase_result(root, queue)
    result.update(
        {
            "model_id": model_id,
            "phase": phase,
            "queue": str(queue),
            "queue_sha256": sha256_file(queue),
        }
    )
    if phase == "compatibility_canary" and not all(
        result["complete_metrics"].values()
    ):
        raise RuntimeError(
            f"{model_id} compatibility canary did not produce complete metrics"
        )
    return result


def execute(campaign_root: Path) -> dict[str, Any]:
    initial = preflight(campaign_root)
    if not initial["ready_to_launch"]:
        _set_state(
            campaign_root,
            "blocked_preflight",
            blockers=initial["blockers"],
            preflight_report_sha256=initial["report_sha256"],
        )
        raise RuntimeError(
            "RTX 4090 launch is blocked: "
            + ", ".join(initial["blockers"])
        )
    _set_state(
        campaign_root,
        "capturing_live_provenance",
        preflight_report_sha256=initial["report_sha256"],
    )
    provenance = {
        model_id: _capture_provenance(campaign_root / model_id)
        for model_id in ("qwen3_8b", "qwen3p5_4b")
    }
    completed = []
    for model_id, phase in PHASES:
        _set_state(
            campaign_root,
            "running",
            active_model_id=model_id,
            active_phase=phase,
            completed=completed,
            provenance=provenance,
        )
        completed.append(
            _run_phase(campaign_root, model_id, phase)
        )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "state": "measurement_complete_pending_replay",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "phases": completed,
        "results_used_for_refit": False,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(_state_path(campaign_root), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute only after the fail-closed live preflight passes.",
    )
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    if args.execute:
        try:
            report = execute(root)
        except RuntimeError as error:
            print(
                json.dumps(
                    {
                        "launched": False,
                        "error": str(error),
                        "state": str(_state_path(root)),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from error
    else:
        report = preflight(root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
