#!/usr/bin/env python3
"""Run one post-hoc full-dataset diagnostic for the business long-tail outlier."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from approval_gate import build_provenance_binding, build_queue_binding
from common import (
    ARTIFACT_DIR,
    MATRIX_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from evaluate_h800_unified_bounded_canary_v3 import _attempt_dir, _structure_audit, _summaries
from prepare_h800_prospective_holdout import probe_hardware
from prepare_h800_unified_bounded_business_generalization_v3 import (
    DEFAULT_EXPERIMENT as BUSINESS_EXPERIMENT,
    DEFAULT_PREDICTIONS as BUSINESS_PREDICTIONS,
    DEFAULT_QUEUE as BUSINESS_QUEUE,
    MODEL_INVENTORY,
    V3_ARTIFACT,
)
from promote_approval_candidate import active_execution_processes, promote_candidate
from run_job import live_runtime_identity, live_runtime_patch

CAMPAIGN_ID = "h800_unified_bounded_business_longtail_diagnostic_20260810_v3"
PHASE_ID = "h800_unified_bounded_business_longtail_diagnostic_v3"
SOURCE_JOB_ID = "h800ubg3-8c2d86226fe9f996"
GPU_IDS = (0,)
WARMUP_STEPS = 1
MEASURE_STEPS = 256
MAX_SAMPLES = 8192
QUEUE = MATRIX_DIR / "h800_unified_bounded_business_longtail_diagnostic_jobs_v3.jsonl"
PREDICTIONS = ARTIFACT_DIR / "h800_unified_bounded_business_longtail_diagnostic_frozen_prediction_v3.json"
DESIGN = ARTIFACT_DIR / "h800_unified_bounded_business_longtail_diagnostic_design_v3.json"
RESULT = ARTIFACT_DIR / "h800_unified_bounded_business_longtail_diagnostic_results_v3.json"
MARKDOWN = ARTIFACT_DIR / "h800_unified_bounded_business_longtail_diagnostic_results_v3.md"
STAGING = ROOT / "unified_bounded_business_longtail_diagnostic_staging"
EXPERIMENT = STAGING / "experiment.h800_unified_bounded_business_longtail_diagnostic_v3.json"
CANDIDATE = ARTIFACT_DIR / "approval_design_h800_unified_bounded_business_longtail_diagnostic_v3_candidate.json"
LIVE_CONFIG = ROOT / "config" / "experiment.json"
PYTHON = Path("/fine-tuning-launcher/.venv/bin/python")
RUNTIME_PARAMETERS = 4_022_468_096
AUTHORIZATION = (
    "用户要求在真实业务数据集上补充实验并验收V3中心精度。12任务业务验收出现一条短尾数据上的离群结果；"
    "日志证明1+4步只采到1293 token，而完整业务profile存在4096 token截断样本。"
    "本批准仅允许GPU0运行同一冻结V3配置的单任务全数据覆盖诊断：8192条真实业务样本，1步warmup+256步measure，"
    "不packing、不offload、不重试；该任务是post-hoc测量口径诊断，不得计作独立前瞻验收，不得发布或覆盖线上模型。"
)


def _atomic_install(source: Path, destination: Path) -> None:
    payload = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _run(command: list[str]) -> None:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "scripts"), *([existing] if existing else [])])
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with code {result.returncode}: {command}")


def _prepare() -> tuple[dict[str, Any], dict[str, Any]]:
    outputs = (QUEUE, PREDICTIONS, DESIGN, EXPERIMENT, CANDIDATE, RESULT, MARKDOWN)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite diagnostic outputs: {existing}")
    source_jobs = {str(row["job_id"]): dict(row) for row in read_jsonl(BUSINESS_QUEUE)}
    source_job = source_jobs[SOURCE_JOB_ID]
    if (
        source_job["dataset_id"] != "real_71014_short_tail_qwen3_v1"
        or source_job["model_id"] != "qwen3_4b"
        or source_job["train_type"] != "lora"
        or int(source_job["gpu_count"]) != 1
        or int(source_job["mbs"]) != 4
        or int(source_job["cutoff_len"]) != 4096
        or int(source_job["zero_stage"]) != 0
        or bool(source_job["gc"])
    ):
        raise ValueError("source outlier configuration drifted")
    identity = {"source_job_id": SOURCE_JOB_ID, "campaign_id": CAMPAIGN_ID, "measurement_steps": MEASURE_STEPS}
    job = dict(source_job)
    job.update({
        "job_id": stable_id("h800ubg3tail", identity),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "experiment_group": "UNIFIED_BOUNDED_BUSINESS_LONGTAIL_DIAGNOSTIC_V3",
        "evidence_role": "posthoc_measurement_horizon_diagnostic",
        "purpose": "exercise_full_real_business_long_tail_after_short_probe_outlier",
        "fidelity": "posthoc_full_dataset_memory_peak_1plus256",
        "max_samples": MAX_SAMPLES,
        "warmup_steps": WARMUP_STEPS,
        "measure_steps": MEASURE_STEPS,
        "calibration_partition": {
            "role": "diagnostic_only",
            "policy": "posthoc_full_business_tail_measurement_v1",
            "split_unit_id": str(source_job["dataset_id"]),
        },
    })

    frozen = read_json(BUSINESS_PREDICTIONS)
    source_prediction = next(row for row in frozen["rows"] if row["job_id"] == SOURCE_JOB_ID)
    prediction = {
        "schema": "sft_h800_unified_bounded_business_longtail_diagnostic_prediction/v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_full_dataset_diagnostic_outcome",
        "outcomes_observed": 0,
        "posthoc_after_short_probe": True,
        "independent_acceptance_evidence": False,
        "v3_artifact": {"path": str(V3_ARTIFACT.resolve()), "sha256": sha256_file(V3_ARTIFACT)},
        "job_id": str(job["job_id"]),
        "source_short_probe_job_id": SOURCE_JOB_ID,
        "record": {**dict(source_prediction["record"]), "record_id": f"posthoc_longtail::{job['job_id']}"},
        "v3": dict(source_prediction["v3"]),
        "ordered_job_payload_sha256": sha256_json([job]),
    }
    prediction["report_sha256"] = sha256_json(prediction)
    experiment = dict(read_json(BUSINESS_EXPERIMENT))
    experiment["training_scope"] = {
        **dict(experiment["training_scope"]),
        "phase_id": PHASE_ID,
        "model_ids": ["qwen3_4b"],
        "gpu_ids": list(GPU_IDS),
        "exclusive_node_gpu_ids": list(GPU_IDS),
        "max_gpu_count": 1,
        "gpu_counts": [1],
        "zero_by_gpu_count": {"1": ["none"]},
        "objective": "post-hoc full-dataset peak diagnostic for frozen-v3 business long-tail outlier",
    }
    experiment["measurement"] = {
        **dict(experiment.get("measurement") or {}),
        "memory_probe_max_steps": WARMUP_STEPS + MEASURE_STEPS,
        "throughput_warmup_steps": WARMUP_STEPS,
        "throughput_measure_steps": MEASURE_STEPS,
        "performance_parallelism": "exclusive_pool",
        "scheduler_order_policy": "single_job",
        "rerun_on_unhealthy_result": False,
    }
    experiment["datasets"] = [{"id": str(job["dataset_id"]), "category": str(job["dataset_category"]), "target_cutoffs": [4096]}]
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    write_jsonl(QUEUE, [job])
    write_json(PREDICTIONS, prediction)
    write_json(EXPERIMENT, experiment)
    design = {
        "schema": "sft_h800_unified_bounded_business_longtail_diagnostic_design/v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "status": "frozen_before_full_dataset_diagnostic_outcome",
        "posthoc_after_short_probe": True,
        "independent_acceptance_evidence": False,
        "production_model_mutated": False,
        "old_missing_34_jobs_included": False,
        "measurement_contract": {
            "dataset_rows": MAX_SAMPLES,
            "global_batch_size": int(job["target_gbs"]),
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "logical_samples_consumed": (WARMUP_STEPS + MEASURE_STEPS) * int(job["target_gbs"]),
            "required_observed_sequence_max": 4096,
        },
        "bindings": {
            "queue": {"path": str(QUEUE.resolve()), "sha256": sha256_file(QUEUE)},
            "predictions": {"path": str(PREDICTIONS.resolve()), "sha256": sha256_file(PREDICTIONS)},
            "experiment": {"path": str(EXPERIMENT.resolve()), "sha256": sha256_file(EXPERIMENT)},
            "v3_artifact": {"path": str(V3_ARTIFACT.resolve()), "sha256": sha256_file(V3_ARTIFACT)},
        },
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    return job, prediction


def _approve(job: dict[str, Any]) -> None:
    if active_execution_processes():
        raise RuntimeError("another training controller or job is active")
    hardware = probe_hardware(required_gpu_ids=GPU_IDS)
    if hardware.get("exact_h800_pool") is not True or hardware.get("selected_gpu_compute_processes"):
        raise RuntimeError("GPU0 is unavailable for the diagnostic")
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    if runtime_patch.get("all_passed") is not True:
        raise RuntimeError("live runtime patch is unhealthy")
    source_paths = [
        ROOT / "scripts" / name
        for name in (
            "approval_gate.py",
            "common.py",
            "metrics_callback.py",
            "promote_approval_candidate.py",
            "run_job.py",
            "runtime_evidence.py",
            "scheduler.py",
            "train_entry.py",
            Path(__file__).name,
        )
    ] + [ROOT / "config" / "experiment.json", ROOT / "config" / "hardware.json", ROOT / "config" / "models.json"]
    queue_binding = build_queue_binding(QUEUE, [job], ROOT)
    provenance_binding = build_provenance_binding(ROOT, source_paths=source_paths)
    bound_files = {
        QUEUE,
        PREDICTIONS,
        DESIGN,
        EXPERIMENT,
        MODEL_INVENTORY,
        V3_ARTIFACT,
        ARTIFACT_DIR / "provenance.json",
        ARTIFACT_DIR / "model_inventory.json",
        ROOT / "data" / "dataset_info.json",
        ROOT / "config" / "experiment.json",
        ROOT / "config" / "models.json",
        Path(str(job["data_path"])),
        Path(str(job["dataset_profile_path"])),
        Path(str(job["model_path"])),
        *source_paths,
    }
    missing = [str(path) for path in bound_files if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    manifest = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in sorted(bound_files, key=str)
        if path.is_file() and path.resolve().is_relative_to(ROOT.resolve())
    }
    approval = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{CAMPAIGN_ID}:{PHASE_ID}",
        "authorization": AUTHORIZATION,
        "campaign_design": {"path": str(DESIGN.resolve().relative_to(ROOT.resolve())), "sha256": sha256_file(DESIGN)},
        "frozen_predictions": {"path": str(PREDICTIONS.resolve().relative_to(ROOT.resolve())), "sha256": sha256_file(PREDICTIONS)},
        "file_sha256": manifest,
        "execution_order": [PHASE_ID],
        "allowed_job_ids": [str(job["job_id"])],
        "authorized_gpu_ids": list(GPU_IDS),
        "max_gpu_count": 1,
        "scheduler_execution": {"join_busy_pool": False, "preemption_allowed": False, "policy": "single diagnostic on GPU0"},
        "oom_policy": "no_retry; OOM is right-censored diagnostic evidence",
        "hardware_preflight": hardware,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        "throughput_screen_delta": {
            "allowed_job_ids": queue_binding["ordered_job_ids"],
            "queue_path": queue_binding["path"],
            "queue_sha256": queue_binding["sha256"],
            "ordered_job_ids": queue_binding["ordered_job_ids"],
            "ordered_job_payload_sha256": queue_binding["ordered_job_payload_sha256"],
            "job_payload_sha256": queue_binding["job_payload_sha256"],
        },
    }
    write_json(CANDIDATE, approval)
    promote_candidate(
        candidate_path=CANDIDATE,
        expected_candidate_sha256=sha256_file(CANDIDATE),
        project_root=ROOT,
        authorization=AUTHORIZATION,
        approved_by="user",
        promote=True,
    )


def _evaluate(job: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    status = read_json(ROOT / "results" / str(job["job_id"]) / "status.json")
    classification = str(status.get("classification") or "")
    if classification not in {"success", "oom"}:
        raise RuntimeError(f"diagnostic did not terminate: {status}")
    attempt = _attempt_dir(str(job["job_id"]))
    if attempt is None:
        raise RuntimeError("diagnostic attempt pointer is missing")
    structure = _structure_audit(attempt, expected_parameters=RUNTIME_PARAMETERS, expected_world_size=1)
    if not structure["all_passed"]:
        raise RuntimeError(f"runtime identity mismatch: {structure}")
    summaries = _summaries(attempt)
    max_reserved = max((float(row["max_reserved"]) for row in summaries), default=None)
    measured_steps = [step for row in summaries for step in ((row.get("batch_shape_evidence") or {}).get("measured_steps") or [])]
    max_sequence = max((int(step["logical_sequence_length_max"]) for step in measured_steps), default=None)
    center = float(prediction["v3"]["center_bytes"])
    result: dict[str, Any] = {
        "schema": "sft_h800_unified_bounded_business_longtail_diagnostic_results/v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "posthoc_after_short_probe": True,
        "independent_acceptance_evidence": False,
        "production_model_mutated": False,
        "job_id": str(job["job_id"]),
        "classification": classification,
        "runtime_model_identity": structure,
        "measured_steps": len(measured_steps),
        "observed_logical_sequence_max": max_sequence,
        "required_logical_sequence_max": 4096,
        "tail_exercised": max_sequence is not None and max_sequence >= 4096,
        "frozen_v3_center_bytes": center,
        "frozen_v3_upper_bytes": float(prediction["v3"]["admission_upper_bytes"]),
        "max_reserved_bytes": max_reserved,
    }
    if classification == "success" and max_reserved is not None:
        signed = center / max_reserved - 1.0
        result.update({"center_ape": abs(signed), "center_signed_error": signed})
    result["report_sha256"] = sha256_json(result)
    write_json(RESULT, result)
    lines = [
        "# V3 真实业务短文本长尾：完整数据覆盖诊断",
        "",
        "该任务是在短探针离群结果出现后的 post-hoc 诊断，不属于独立前瞻验收。",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 任务结果 | {classification} |",
        f"| 测量步数 | {result['measured_steps']} |",
        f"| 实际采到的最长序列 | {max_sequence} tokens |",
        f"| V3 冻结中心 | {center / (1 << 30):.2f} GiB |",
        f"| 实际 reserved 峰值 | {max_reserved / (1 << 30):.2f} GiB |" if max_reserved is not None else "| 实际 reserved 峰值 | OOM/right-censored |",
        f"| 中心 APE | {result.get('center_ape', float('nan')):.2%} |" if result.get("center_ape") is not None else "| 中心 APE | n/a |",
        "",
    ]
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    job, prediction = _prepare()
    STAGING.mkdir(parents=True, exist_ok=True)
    backup = STAGING / "prelaunch_backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup.mkdir(parents=True, exist_ok=False)
    if LIVE_CONFIG.is_file():
        shutil.copy2(LIVE_CONFIG, backup / "experiment.json")
    _atomic_install(EXPERIMENT, LIVE_CONFIG)
    _run([str(PYTHON), str(ROOT / "scripts" / "capture_provenance.py")])
    _approve(job)
    _run([str(PYTHON), str(ROOT / "scripts" / "scheduler.py"), "--input", str(QUEUE), "--execute"])
    print(json.dumps(_evaluate(job, prediction), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
