#!/usr/bin/env python3
"""Prepare the pre-outcome audited v2 business-blind matrix."""

from __future__ import annotations

from common import ARTIFACT_DIR, MATRIX_DIR, ROOT
import prepare_h800_final_memory_business_blind_v1 as implementation

implementation.CAMPAIGN_ID = "h800_final_memory_business_blind_20260810_v2"
implementation.PHASE_ID = "h800_final_memory_business_blind_v2"
implementation.JOB_SCHEMA = "sft_h800_final_memory_business_blind_job/v2"
implementation.DESIGN_SCHEMA = "sft_h800_final_memory_business_blind_design/v2"
implementation.DATA_SCHEMA = "sft_h800_final_memory_business_blind_data/v2"
implementation.PREDICTION_SCHEMA = "sft_h800_final_memory_business_blind_predictions/v2"
implementation.DEFAULT_QUEUE = MATRIX_DIR / "h800_final_memory_business_blind_jobs_v2.jsonl"
implementation.DEFAULT_DATA_BUNDLE = ARTIFACT_DIR / "h800_final_memory_business_blind_data_v2.json"
implementation.DEFAULT_PREDICTIONS = ARTIFACT_DIR / "h800_final_memory_business_blind_frozen_predictions_v2.json"
implementation.DEFAULT_DESIGN = ARTIFACT_DIR / "h800_final_memory_business_blind_design_v2.json"
implementation.STAGING_DIR = ROOT / "final_memory_business_blind_v2_staging"
implementation.DEFAULT_EXPERIMENT = implementation.STAGING_DIR / "experiment.h800_final_memory_business_blind_v2.json"


if __name__ == "__main__":
    implementation.main()
