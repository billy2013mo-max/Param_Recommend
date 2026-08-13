#!/usr/bin/env python3
"""Prepare the bounded Qwen3-14B Full two-H800 memory calibration."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from llamafactory.data.template import TEMPLATES
from transformers import AutoTokenizer

from analyze_h800_fresh_memory_residual_v2 import profile_padding_statistics
from common import (
    ARTIFACT_DIR,
    DATA_DIR,
    RESULTS_DIR,
    ROOT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
)


SCHEMA = "sft_h800_qwen3_14b_full_boundary_calibration_design/v1"
CAMPAIGN_ID = "h800_qwen3_14b_full_2gpu_boundary_calibration_20260802_v1"
PHASE_ID = "h800_qwen3_14b_full_2gpu_boundary_calibration_v1"
SOURCE_QUEUE = ROOT / "matrix" / "h800_profile_aware_memory_calibration_jobs_v1.jsonl"
SOURCE_DESIGN = ARTIFACT_DIR / "h800_profile_aware_memory_calibration_design_v1.json"
SOURCE_ACCEPTANCE = ARTIFACT_DIR / "h800_fresh_holdout_acceptance_v2.json"
PROFILE_DIR = ARTIFACT_DIR / "h800_14b_full_boundary_v1" / "profiles"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_14b_full_boundary_calibration_design_v1.json"
MODEL_PATH = Path("/wanqing-models/Qwen3-14B")
DATASET_REPEATS = {
    "fresh_business_education_1000": (0, 1),
    "fresh_business_relevance_1000": (0,),
    "fresh_business_title_1000": (0,),
}


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class Encoder:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True,
        )
        self.template = copy.deepcopy(TEMPLATES["qwen3_nothink"])
        self.template.fix_special_tokens(self.tokenizer)

    def encode(self, row: dict[str, Any]) -> tuple[int, int]:
        conversation = [
            {"role": "user", "content": str(row["prompt"])},
            {"role": "assistant", "content": str(row["response"])},
        ]
        pairs = self.template.encode_multiturn(
            self.tokenizer,
            conversation,
            system=str(row.get("system") or "") or None,
            tools=None,
        )
        source_tokens = sum(len(source_ids) for source_ids, _ in pairs)
        label_tokens = sum(len(target_ids) for _, target_ids in pairs)
        if self.template.efficient_eos:
            label_tokens += 1
        return source_tokens + label_tokens, label_tokens


def _completed_source_campaign() -> dict[str, Any]:
    jobs = read_jsonl(SOURCE_QUEUE)
    if len(jobs) != 20:
        raise ValueError("source profile-aware calibration queue must contain 20 jobs")
    attempts = []
    for job in jobs:
        result_dir = RESULTS_DIR / str(job["job_id"])
        latest_path = result_dir / "latest_attempt.json"
        if not latest_path.is_file():
            raise FileNotFoundError(f"incomplete source calibration result: {job['job_id']}")
        latest = read_json(latest_path)
        attempt_dir = result_dir / str(latest["attempt_path"])
        status_path = attempt_dir / "status.json"
        fingerprint_path = attempt_dir / "execution_fingerprint.json"
        if not status_path.is_file() or not fingerprint_path.is_file():
            raise FileNotFoundError(f"incomplete source calibration attempt: {job['job_id']}")
        status = read_json(status_path)
        if (
            status.get("classification") != "success"
            or status.get("execution_fingerprint_quality") != "complete"
            or status.get("calibration_eligible") is not True
        ):
            raise ValueError(f"source calibration attempt is ineligible: {job['job_id']}")
        attempts.append(
            {
                "job_id": job["job_id"],
                "latest_attempt_path": str(latest_path.resolve()),
                "latest_attempt_sha256": sha256_file(latest_path),
                "status_path": str(status_path.resolve()),
                "status_sha256": sha256_file(status_path),
                "execution_fingerprint_path": str(fingerprint_path.resolve()),
                "execution_fingerprint_sha256": sha256_file(fingerprint_path),
                "execution_attempt_id": status["execution_attempt_id"],
            }
        )
    return {
        "queue_path": str(SOURCE_QUEUE.resolve()),
        "queue_sha256": sha256_file(SOURCE_QUEUE),
        "job_count": len(jobs),
        "success_count": len(attempts),
        "complete_fingerprint_count": len(attempts),
        "calibration_eligible_count": len(attempts),
        "attempts": attempts,
        "attempt_set_sha256": sha256_json(attempts),
    }


def _build_profile(dataset_id: str, data_path: Path, encoder: Encoder) -> Path:
    profile_path = PROFILE_DIR / f"{dataset_id}.qwen3_14b.qwen3_nothink.jsonl"
    if profile_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen profile: {profile_path}")
    rows = []
    with data_path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            row = json.loads(line)
            total_tokens, label_tokens = encoder.encode(row)
            rows.append(
                {
                    "sample_id": str(row.get("sample_id") or f"{dataset_id}:{index}"),
                    "total_tokens": total_tokens,
                    "label_tokens": label_tokens,
                    "turns": 3 if row.get("system") else 2,
                    "assistant_turns": 1,
                }
            )
    if len(rows) != 1000:
        raise ValueError(f"{dataset_id} must contain exactly 1000 rows, got {len(rows)}")
    _atomic_jsonl(profile_path, rows)
    return profile_path


def build_design() -> dict[str, Any]:
    completion = _completed_source_campaign()
    registry = read_json(DATA_DIR / "dataset_info.json")
    inventory = read_json(ARTIFACT_DIR / "model_inventory.json")
    model = next(row for row in inventory["models"] if row["id"] == "qwen3_14b")
    encoder = Encoder()
    scenarios = []
    slots = []
    for dataset_id, repeats in DATASET_REPEATS.items():
        entry = registry[dataset_id]
        data_path = DATA_DIR / entry["file_name"]
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        profile_path = _build_profile(dataset_id, data_path, encoder)
        scenario_id = f"{dataset_id}__qwen3_14b_full_2gpu_cutoff2048"
        statistics = {
            f"mbs{mbs}": profile_padding_statistics(
                profile_path,
                cutoff_len=2048,
                physical_mbs=mbs,
            )
            for mbs in (1, 2)
        }
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "dataset_id": dataset_id,
                "data_path": str(data_path.resolve()),
                "data_sha256": sha256_file(data_path),
                "profile_path": str(profile_path.resolve()),
                "profile_sha256": sha256_file(profile_path),
                "profile_statistics": statistics,
                "model_id": "qwen3_14b",
                "train_type": "full",
                "cutoff_len": 2048,
                "target_gbs": 64,
            }
        )
        for repeat in repeats:
            for mbs in (1, 2):
                material = {
                    "campaign_id": CAMPAIGN_ID,
                    "scenario_id": scenario_id,
                    "mbs": mbs,
                    "repeat": repeat,
                }
                slots.append(
                    {
                        "slot_id": stable_id("h80014bfull", material),
                        **material,
                        "purpose": (
                            "high_padding_boundary_repeat"
                            if dataset_id == "fresh_business_education_1000"
                            else "length_distribution_boundary_control"
                        ),
                        "model_id": "qwen3_14b",
                        "model_path": model["path"],
                        "train_type": "full",
                        "dataset_id": dataset_id,
                        "data_path": str(data_path.resolve()),
                        "data_sha256": sha256_file(data_path),
                        "dataset_profile_path": str(profile_path.resolve()),
                        "dataset_profile_sha256": sha256_file(profile_path),
                        "cutoff_len": 2048,
                        "target_gbs": 64,
                        "gpu_count": 2,
                        "zero_stage": 3,
                        "zero": "zero3",
                        "gc": True,
                        "gradient_checkpointing": True,
                        "mbs": mbs,
                        "gradient_accumulation_steps": 64 // (2 * mbs),
                        "packing": False,
                        "offload": False,
                        "fidelity": "formal_3plus10",
                        "warmup_steps": 3,
                        "measure_steps": 10,
                        "calibration_partition": {
                            "role": "calibration",
                            "split_unit_id": dataset_id,
                            "policy": "qwen3_14b_full_2gpu_boundary_profile_stratified_v1",
                        },
                        "expected_padding_pressure": statistics[f"mbs{mbs}"],
                    }
                )
    if len(scenarios) != 3 or len(slots) != 8:
        raise ValueError(f"expected 3 scenarios and 8 slots, got {len(scenarios)} and {len(slots)}")
    approval = read_json(ROOT / "config" / "APPROVED_TO_RUN.json")
    receipt_path = ROOT / str(approval["promotion"]["receipt_path"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "status": "design_only_waiting_for_exact_approval",
        "gpu_training_started": False,
        "queues_mutated": False,
        "execution_authorized": False,
        "objective": (
            "Calibrate the previously unfit Qwen3-14B Full two-GPU ZeRO-3+GC boundary "
            "without using the old prospective holdout as training data."
        ),
        "source_bindings": {
            "fresh_acceptance": {
                "path": str(SOURCE_ACCEPTANCE.resolve()),
                "sha256": sha256_file(SOURCE_ACCEPTANCE),
            },
            "profile_aware_design": {
                "path": str(SOURCE_DESIGN.resolve()),
                "sha256": sha256_file(SOURCE_DESIGN),
            },
            "source_campaign_approval_receipt": {
                "path": str(receipt_path.resolve()),
                "sha256": sha256_file(receipt_path),
            },
        },
        "source_campaign_completion": completion,
        "design": {
            "scenario_count": len(scenarios),
            "job_count": len(slots),
            "gpu_job_equivalents": sum(slot["gpu_count"] for slot in slots),
            "selector": "Qwen3-14B Full, 2 H800, ZeRO-3, GC on, no offload, no packing",
            "distribution_strata": list(DATASET_REPEATS),
            "high_padding_repeats": 2,
        },
        "scenarios": scenarios,
        "candidate_slots": slots,
        "governance": {
            "calibration_only": True,
            "old_24_row_holdout_enters_fit": False,
            "new_job_bound_calibration_partition_required": True,
            "final_unseen_holdout_still_required": True,
        },
        "next_step": "materialize a new eight-job queue and promote an exact GPU-4,5 approval",
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    if DEFAULT_OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite frozen design: {DEFAULT_OUTPUT}")
    design = build_design()
    write_json(DEFAULT_OUTPUT, design)
    print(
        json.dumps(
            {
                "output": str(DEFAULT_OUTPUT),
                "scenarios": design["design"]["scenario_count"],
                "jobs": design["design"]["job_count"],
                "execution_authorized": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
