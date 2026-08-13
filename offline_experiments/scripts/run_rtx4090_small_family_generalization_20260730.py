#!/usr/bin/env python3
"""Fail-closed preflight and executor for the small-family RTX 4090 holdout."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

from common import read_json, read_jsonl, sha256_file, sha256_json, write_json
from evaluate_rtx4090_small_family_generalization_20260730 import (
    evaluate as evaluate_frozen_replay,
)
from prepare_rtx4090_small_family_generalization_20260730 import (
    DEFAULT_CAMPAIGN_ROOT,
    MODELS,
    QWEN35_OVERLAY,
    QWEN35_OVERLAY_FILES,
    QWEN35_OVERLAY_SHA256,
)


SCHEMA = "sft_rtx4090_small_family_generalization_orchestrator/v1"
MODEL_ORDER = ("qwen2p5_1p5b", "qwen3p5_0p8b", "qwen3p5_4b")
PHASE_ORDER = (
    "compatibility_canary",
    "memory_boundary",
    "multi_candidate_ranking",
)


def _command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            env=env,
        )
        return {
            "command": command,
            "return_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "passed": result.returncode == 0,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "command": command,
            "return_code": None,
            "stdout": "",
            "stderr": repr(error),
            "passed": False,
        }


def _visible_indices() -> set[int] | None:
    value = os.environ.get("NVIDIA_VISIBLE_DEVICES", "").strip()
    if not value or value in {"all", "void", "none"}:
        return None
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if tokens and all(token.isdigit() for token in tokens):
        return {int(token) for token in tokens}
    return None


def _hardware(campaign_root: Path) -> dict[str, Any]:
    runtime = read_json(
        campaign_root
        / "qwen2p5_1p5b"
        / "config"
        / "experiment.json"
    )["fixed_runtime"]
    torch_probe = _command(
        [
            str(runtime["python"]),
            "-c",
            (
                "import json,torch;"
                "print(json.dumps(["
                "{'logical_index':i,'name':torch.cuda.get_device_name(i),"
                "'memory_total_bytes':torch.cuda.get_device_properties(i).total_memory,"
                "'capability':'.'.join(map(str,torch.cuda.get_device_capability(i)))}"
                " for i in range(torch.cuda.device_count())]))"
            ),
        ]
    )
    logical_rows: list[dict[str, Any]] = []
    if torch_probe["passed"]:
        try:
            logical_rows = json.loads(torch_probe["stdout"].splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            logical_rows = []
    exact_logical_pool = (
        [row.get("logical_index") for row in logical_rows] == [0, 1, 2, 3]
        and all("RTX 4090" in str(row.get("name")) for row in logical_rows)
        and all(
            int(row.get("memory_total_bytes", 0)) >= 23 * 2**30
            for row in logical_rows
        )
    )

    gpu_query = _command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    physical_rows = []
    if gpu_query["passed"]:
        for line in gpu_query["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",", 3)]
            if len(parts) == 4:
                physical_rows.append(
                    {
                        "index": int(parts[0]),
                        "uuid": parts[1],
                        "name": parts[2],
                        "memory_total_mib": float(parts[3]),
                    }
                )
    visible = _visible_indices()
    if visible is not None:
        selected = [row for row in physical_rows if row["index"] in visible]
    elif len(physical_rows) == 4:
        selected = physical_rows
    else:
        selected = [
            row
            for row in physical_rows
            if "RTX 4090" in row["name"]
        ]
    exact_physical_pool = (
        len(selected) == 4
        and all("RTX 4090" in row["name"] for row in selected)
    )

    process_query = _command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    selected_uuids = {row["uuid"] for row in selected}
    processes = []
    if process_query["passed"]:
        for line in process_query["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",", 2)]
            if len(parts) == 3 and parts[0] in selected_uuids:
                try:
                    pid = int(parts[1])
                except ValueError:
                    continue
                processes.append(
                    {
                        "gpu_uuid": parts[0],
                        "pid": pid,
                        "process_name": parts[2],
                    }
                )
    exact_pool = exact_logical_pool and exact_physical_pool
    return {
        "torch_probe": torch_probe,
        "logical_gpu_rows": logical_rows,
        "gpu_query": gpu_query,
        "physical_gpu_rows": physical_rows,
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "selected_physical_gpu_rows": selected,
        "process_query": process_query,
        "selected_gpu_compute_processes": processes,
        "exact_four_card_rtx4090_pool": exact_pool,
        "selected_pool_idle": exact_pool and not processes,
    }


def _overlay_manifest(root: Path) -> tuple[str | None, int]:
    if not root.is_dir():
        return None, 0
    lines: list[bytes] = []
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix not in {".pyc", ".pyo"}
        and "__pycache__" not in path.relative_to(root).parts
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        lines.append(
            f"{sha256_file(path)}  ./{relative}\n".encode()
        )
    return hashlib.sha256(b"".join(lines)).hexdigest(), len(files)


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


def _runtime(campaign_root: Path) -> dict[str, Any]:
    overlay_sha, overlay_files = _overlay_manifest(QWEN35_OVERLAY)
    overlay_passed = (
        overlay_sha == QWEN35_OVERLAY_SHA256
        and overlay_files == QWEN35_OVERLAY_FILES
    )
    reports = {}
    for model_id in MODEL_ORDER:
        root = campaign_root / model_id
        fixed = read_json(root / "config" / "experiment.json")[
            "fixed_runtime"
        ]
        python, env = _environment(root)
        torchrun = Path(fixed["torchrun"])
        model_path = Path(MODELS[model_id]["path"])
        static = {
            "python": python.is_file(),
            "torchrun": torchrun.is_file(),
            "model": model_path.is_dir(),
            "tokenizer": Path(MODELS[model_id]["tokenizer_path"]).is_dir(),
            "fa2_declared": fixed.get("flash_attn") == "fa2",
            "site_bridge": (
                not model_id.startswith("qwen3p5")
                or (root / "scripts" / "sitecustomize.py").is_file()
            ),
            "qwen35_overlay": (
                not model_id.startswith("qwen3p5") or overlay_passed
            ),
        }
        probe = None
        if all(static.values()):
            imports = (
                "torch,transformers,flash_attn,deepspeed,peft,trl,triton,"
                "llamafactory"
            )
            if model_id.startswith("qwen3p5"):
                imports += ",tilelang,fla"
            probe = _command(
                [
                    str(python),
                    "-c",
                    (
                        f"import {imports};"
                        "print(torch.__version__,transformers.__version__,"
                        "flash_attn.__version__)"
                    ),
                ],
                env=env,
            )
        reports[model_id] = {
            "fixed_runtime": fixed,
            "static_checks": static,
            "import_probe": probe,
            "all_passed": (
                all(static.values())
                and probe is not None
                and probe["passed"]
            ),
        }
    return {
        "models": reports,
        "qwen35_overlay_manifest": {
            "path": str(QWEN35_OVERLAY),
            "sha256": overlay_sha,
            "files": overlay_files,
            "expected_sha256": QWEN35_OVERLAY_SHA256,
            "expected_files": QWEN35_OVERLAY_FILES,
            "passed": overlay_passed,
        },
        "all_passed": all(
            report["all_passed"] for report in reports.values()
        ),
    }


def _freeze_integrity(campaign_root: Path) -> dict[str, Any]:
    path = campaign_root / "FREEZE.json"
    freeze = read_json(path)
    unsigned = dict(freeze)
    digest = unsigned.pop("freeze_sha256", None)
    checks: dict[str, bool] = {
        "freeze_checksum": digest == sha256_json(unsigned),
        "no_gpu_launch_recorded": (
            freeze.get("gpu_experiments_launched") is False
        ),
        "no_results_used_for_fit": (
            freeze.get("results_consumed_for_model_fit") is False
        ),
    }
    for name in ("model_artifact", "prediction_freeze"):
        binding = freeze[name]
        bound = Path(binding["path"])
        checks[f"{name}_sha256"] = (
            bound.is_file() and sha256_file(bound) == binding["sha256"]
        )
    jobs = freeze["jobs"]
    jobs_path = Path(jobs["path"])
    rows = read_jsonl(jobs_path)
    checks["job_queue_sha256"] = (
        jobs_path.is_file() and sha256_file(jobs_path) == jobs["sha256"]
    )
    checks["job_count"] = len(rows) == int(jobs["count"]) == 36
    checks["job_ids_unique"] = (
        len({str(row["job_id"]) for row in rows}) == len(rows)
    )
    generator = freeze["source_bindings"]["generator"]
    generator_path = Path(generator["path"])
    checks["generator_sha256"] = (
        generator_path.is_file()
        and sha256_file(generator_path) == generator["sha256"]
    )
    prediction_sha = freeze["prediction_freeze"]["sha256"]
    for subcampaign in freeze["subcampaigns"]:
        model_id = str(subcampaign["model_id"])
        binding = subcampaign["combined_queue"]
        queue = Path(binding["path"])
        checks[f"{model_id}_queue"] = (
            queue.is_file()
            and sha256_file(queue) == binding["sha256"]
            and len(read_jsonl(queue)) == int(binding["jobs"])
        )
        copied_prediction = (
            campaign_root
            / model_id
            / "artifacts"
            / "frozen_predictions_before_holdout.json"
        )
        checks[f"{model_id}_prediction"] = (
            copied_prediction.is_file()
            and sha256_file(copied_prediction) == prediction_sha
        )
    return {
        "path": str(path.resolve()),
        "freeze_sha256": digest,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def _schedule(campaign_root: Path) -> dict[str, Any]:
    reports = []
    total_waves = 0
    for phase in PHASE_ORDER:
        for model_id in MODEL_ORDER:
            path = (
                campaign_root
                / model_id
                / "matrix"
                / f"queue_{phase}.jsonl"
            )
            rows = read_jsonl(path)
            four_gpu = sum(int(row["gpu_count"]) == 4 for row in rows)
            lower_slots = sum(
                int(row["gpu_count"])
                for row in rows
                if int(row["gpu_count"]) < 4
            )
            waves = four_gpu + math.ceil(lower_slots / 4)
            total_waves += waves
            reports.append(
                {
                    "phase": phase,
                    "model_id": model_id,
                    "jobs": len(rows),
                    "gpu_slots": sum(int(row["gpu_count"]) for row in rows),
                    "by_gpu_count": dict(
                        Counter(int(row["gpu_count"]) for row in rows)
                    ),
                    "minimum_fully_packed_waves": waves,
                }
            )
    return {
        "phases": reports,
        "minimum_sequential_runtime_waves": total_waves,
        "policy": "parallel disjoint masks within one queue; queues are sequential",
    }


def preflight(campaign_root: Path) -> dict[str, Any]:
    integrity = _freeze_integrity(campaign_root)
    hardware = _hardware(campaign_root)
    runtime = _runtime(campaign_root)
    blockers = []
    if not integrity["all_passed"]:
        blockers.append("frozen_artifact_or_queue_integrity_failed")
    if not hardware["exact_four_card_rtx4090_pool"]:
        blockers.append("current_host_is_not_a_usable_4x_rtx4090_pool")
    elif not hardware["selected_pool_idle"]:
        blockers.append("selected_rtx4090_pool_is_not_idle")
    if not runtime["all_passed"]:
        blockers.append("one_or_more_frozen_runtimes_are_unavailable")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_root": str(campaign_root.resolve()),
        "ready_to_launch": not blockers,
        "status": "ready" if not blockers else "blocked_fail_closed",
        "blockers": blockers,
        "freeze_integrity": integrity,
        "hardware": hardware,
        "runtime": runtime,
        "schedule": _schedule(campaign_root),
        "process_policy": (
            "No process is terminated or signalled; occupied or unavailable "
            "GPUs cause a fail-closed result."
        ),
    }
    report["report_sha256"] = sha256_json(report)
    write_json(
        campaign_root / "artifacts" / "preflight_current_host.json",
        report,
    )
    for model_id in MODEL_ORDER:
        write_json(
            campaign_root
            / model_id
            / "artifacts"
            / "preflight_current_host.json",
            report,
        )
    return report


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
    write_json(
        campaign_root / "runtime" / "orchestrator_state.json",
        report,
    )
    for model_id in MODEL_ORDER:
        write_json(
            campaign_root
            / model_id
            / "runtime"
            / "orchestrator_state.json",
            report,
        )


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> None:
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


def _capture_provenance(root: Path) -> dict[str, Any]:
    python, env = _environment(root)
    _run(
        [str(python), str(root / "scripts" / "capture_provenance.py")],
        cwd=root,
        env=env,
    )
    path = root / "artifacts" / "provenance.json"
    report = read_json(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "runtime_fingerprint_sha256": report[
            "runtime_fingerprint_sha256"
        ],
    }


def _phase_result(
    campaign_root: Path,
    root: Path,
    queue: Path,
) -> dict[str, Any]:
    predictions = read_json(
        campaign_root
        / "artifacts"
        / "frozen_predictions_before_holdout.json"
    )
    admitted = {
        str(row["request_id"]): bool(row["memory"]["admitted"])
        for row in predictions["predictions"]
    }
    rows = read_jsonl(queue)
    outcomes = {}
    complete_metrics = {}
    terminal = {"success", "oom"}
    canary_errors = []
    for row in rows:
        job_id = str(row["job_id"])
        result_root = root / "results" / job_id
        status_path = result_root / "status.json"
        if not status_path.is_file():
            outcomes[job_id] = "missing"
            complete_metrics[job_id] = False
            canary_errors.append(f"{job_id}:missing")
            continue
        status = read_json(status_path)
        outcome = str(status.get("classification") or "unknown")
        outcomes[job_id] = outcome
        summaries = list(
            (result_root / "metrics").glob("summary.rank*.json")
        )
        complete = (
            outcome == "success"
            and len(summaries) == int(row["gpu_count"])
        )
        complete_metrics[job_id] = complete
        if row["phase_id"] == "compatibility_canary":
            expected_safe = admitted[str(row["prediction_request_id"])]
            if outcome not in terminal:
                canary_errors.append(f"{job_id}:{outcome}")
            elif expected_safe and not complete:
                canary_errors.append(
                    f"{job_id}:predicted_safe_but_{outcome}"
                )
            elif row["train_type"] == "lora" and not complete:
                canary_errors.append(
                    f"{job_id}:required_lora_canary_{outcome}"
                )
    return {
        "jobs": len(rows),
        "outcomes": outcomes,
        "complete_metrics": complete_metrics,
        "canary_errors": canary_errors,
    }


def _run_phase(
    campaign_root: Path,
    model_id: str,
    phase: str,
) -> dict[str, Any]:
    root = campaign_root / model_id
    queue = root / "matrix" / f"queue_{phase}.jsonl"
    hardware = _hardware(campaign_root)
    if not hardware["selected_pool_idle"]:
        raise RuntimeError(
            "The selected four-card RTX 4090 pool is unavailable or occupied"
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
    result = _phase_result(campaign_root, root, queue)
    result.update(
        {
            "model_id": model_id,
            "phase": phase,
            "queue": str(queue),
            "queue_sha256": sha256_file(queue),
        }
    )
    if phase == "compatibility_canary" and result["canary_errors"]:
        raise RuntimeError(
            f"{model_id} canary gate failed: {result['canary_errors']}"
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
        for model_id in MODEL_ORDER
    }
    completed = []
    for phase in PHASE_ORDER:
        for model_id in MODEL_ORDER:
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
    replay = evaluate_frozen_replay(campaign_root)
    if replay["status"] != "complete":
        raise RuntimeError(
            "All queues finished but frozen replay still has missing results"
        )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "state": "measurement_and_frozen_replay_complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "phases": completed,
        "frozen_replay": {
            "path": str(
                campaign_root
                / "artifacts"
                / "frozen_replay_evaluation.json"
            ),
            "report_sha256": replay["report_sha256"],
            "promotion_ready": replay["promotion_ready"],
        },
        "results_used_for_refit": False,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(
        campaign_root / "runtime" / "orchestrator_state.json",
        report,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT,
    )
    parser.add_argument("--execute", action="store_true")
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
                        "state": str(
                            root / "runtime" / "orchestrator_state.json"
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from error
    else:
        report = preflight(root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("ready_to_launch", True) and not args.execute:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
