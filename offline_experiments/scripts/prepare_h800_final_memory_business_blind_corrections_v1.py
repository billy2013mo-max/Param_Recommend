#!/usr/bin/env python3
"""Prepare the prospective correction set for invalid final-blind executor rows.

The frozen V3 artifact is not refit.  Qwen3.5 rows bind the previously validated
runtime overlay and the native Qwen3.5 template.  Packing rows use the executor's
required physical MBS=1; their pressure ladder is varied through cutoff length.
"""

from __future__ import annotations

from common import ARTIFACT_DIR, DATA_DIR, MATRIX_DIR, ROOT, read_json, read_jsonl
from prepare_h800_final_memory_business_blind_v2 import implementation

implementation.CAMPAIGN_ID = "h800_final_memory_business_blind_corrections_20260810_v1"
implementation.PHASE_ID = "h800_final_memory_business_blind_corrections_v1"
implementation.JOB_SCHEMA = "sft_h800_final_memory_business_blind_correction_job/v1"
implementation.DESIGN_SCHEMA = "sft_h800_final_memory_business_blind_correction_design/v1"
implementation.DATA_SCHEMA = "sft_h800_final_memory_business_blind_correction_data/v1"
implementation.PREDICTION_SCHEMA = "sft_h800_final_memory_business_blind_correction_predictions/v1"
implementation.FREEZE_STATUS = "frozen_before_any_correction_gpu_outcome"
implementation.PREDICTION_ROLE = "prospective_executor_corrected_supplement_frozen_before_own_outcomes"
implementation.GPU_IDS = (4, 5, 6, 7)


def _prior_scored_outcomes() -> int:
    count = 0
    for row in read_jsonl(implementation.DEFAULT_QUEUE):
        status_path = ROOT / "results" / str(row["job_id"]) / "status.json"
        if status_path.is_file() and read_json(status_path).get("classification") in {"success", "oom"}:
            count += 1
    return count


implementation.PRIOR_CAMPAIGN_GPU_OUTCOMES_OBSERVED_AT_FREEZE = _prior_scored_outcomes()
implementation.SCENARIOS = (
    {
        "source_dataset_id": "dataset-quuxpd-1786009603",
        "scenario": "short_qwen35_lora_corrected_runtime",
        "model_id": "qwen3p5_4b",
        "train_type": "lora",
        "gpu_count": 1,
        "zero_stage": 0,
        "gc": False,
        "packing": False,
    },
    {
        "source_dataset_id": "dataset-gy6hdc-1786278399",
        "scenario": "long_packed_full_corrected_physical_mbs",
        "model_id": "qwen3_4b",
        "train_type": "full",
        "gpu_count": 1,
        "zero_stage": 0,
        "gc": False,
        "packing": True,
    },
    {
        "source_dataset_id": "dataset-gy6hdc-1786278399",
        "scenario": "long_packed_qwen35_lora_corrected_runtime_and_mechanism",
        "model_id": "qwen3p5_4b",
        "train_type": "lora",
        "gpu_count": 2,
        "zero_stage": 2,
        "gc": False,
        "packing": True,
    },
)
implementation.EXPECTED_SCENARIOS = len(implementation.SCENARIOS)
implementation.EXPECTED_JOBS = implementation.EXPECTED_SCENARIOS * len(implementation.TARGET_PRESSURES)
implementation.OUTPUT_DATA_DIR = DATA_DIR / "final_memory_business_blind_corrections_v1"
implementation.PROFILE_DIR = ARTIFACT_DIR / "final_memory_business_blind_corrections_v1" / "profiles"
implementation.DEFAULT_QUEUE = MATRIX_DIR / "h800_final_memory_business_blind_correction_jobs_v1.jsonl"
implementation.DEFAULT_DATA_BUNDLE = ARTIFACT_DIR / "h800_final_memory_business_blind_correction_data_v1.json"
implementation.DEFAULT_PREDICTIONS = ARTIFACT_DIR / "h800_final_memory_business_blind_correction_predictions_v1.json"
implementation.DEFAULT_DESIGN = ARTIFACT_DIR / "h800_final_memory_business_blind_correction_design_v1.json"
implementation.STAGING_DIR = ROOT / "final_memory_business_blind_corrections_v1_staging"
implementation.DEFAULT_EXPERIMENT = implementation.STAGING_DIR / "experiment.h800_final_memory_business_blind_corrections_v1.json"

CAMPAIGN_ID = implementation.CAMPAIGN_ID
PHASE_ID = implementation.PHASE_ID
EXPECTED_JOBS = implementation.EXPECTED_JOBS
GPU_IDS = implementation.GPU_IDS
DEFAULT_QUEUE = implementation.DEFAULT_QUEUE
DEFAULT_DATA_BUNDLE = implementation.DEFAULT_DATA_BUNDLE
DEFAULT_PREDICTIONS = implementation.DEFAULT_PREDICTIONS
DEFAULT_DESIGN = implementation.DEFAULT_DESIGN
DEFAULT_EXPERIMENT = implementation.DEFAULT_EXPERIMENT
V3_ARTIFACT = implementation.V3_ARTIFACT
MEASUREMENT_GATE = implementation.MEASUREMENT_GATE
DATASET_INFO = implementation.DATASET_INFO
QWEN35_RUNTIME_CONTRACT = implementation.QWEN35_RUNTIME_CONTRACT


if __name__ == "__main__":
    if implementation.PRIOR_CAMPAIGN_GPU_OUTCOMES_OBSERVED_AT_FREEZE != 45:
        raise SystemExit(
            "correction preparation is blocked until all 45 executor-valid original rows are terminal; "
            f"observed={implementation.PRIOR_CAMPAIGN_GPU_OUTCOMES_OBSERVED_AT_FREEZE}"
        )
    implementation.main()
