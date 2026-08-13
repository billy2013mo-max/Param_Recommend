from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_h800_memory_center_models_v1 as benchmark
from h800_unified_bounded_memory_model import (
    load_artifact,
    predict_records,
    validate_artifact,
)

ARTIFACT = ROOT / "artifacts" / "h800_unified_bounded_memory_candidate_v2.json"


class UnifiedBoundedMemoryV2Tests(unittest.TestCase):
    def test_lora_zero3_nogc_activation_feature_is_physical_interaction(self) -> None:
        records, _strict, _audit = benchmark._load_records()
        row = next(
            record
            for record in records
            if record["record_id"] == "combined::5e49f2e42c89d1dd0125420a"
        )
        value = benchmark._feature_value(
            row, benchmark.LORA_ZERO3_NOGC_ACTIVATION_FEATURE
        )
        expected = (
            row["reference_bytes"]
            / benchmark.DEVICE_CAPACITY_BYTES
            * row["features"]["activation_share"]
        )
        self.assertAlmostEqual(value, expected, places=12)

    def test_frozen_artifact_hash_and_release_contract(self) -> None:
        artifact = load_artifact(ARTIFACT)
        validate_artifact(artifact)
        self.assertFalse(artifact["publishable"])
        self.assertFalse(artifact["production_override_allowed"])

    def test_known_lora_zero3_oom_is_rejected(self) -> None:
        artifact = load_artifact(ARTIFACT)
        records, _strict, _audit = benchmark._load_records()
        row = next(
            record
            for record in records
            if record["record_id"] == "combined::5e49f2e42c89d1dd0125420a"
        )
        prediction = predict_records([row], artifact)[0]
        self.assertGreater(
            prediction["admission_upper_bytes"], prediction["safe_limit_bytes"]
        )
        self.assertFalse(prediction["admitted"])

    def test_all_known_oom_rows_are_rejected_by_final_artifact(self) -> None:
        artifact = load_artifact(ARTIFACT)
        records, _strict, _audit = benchmark._load_records()
        oom_rows = [row for row in records if row["state"] == "censored"]
        predictions = predict_records(oom_rows, artifact)
        admitted = [row["record_id"] for row in predictions if row["admitted"]]
        self.assertEqual(admitted, [])

    def test_mutated_artifact_is_rejected(self) -> None:
        artifact = json.loads(ARTIFACT.read_text())
        artifact["admission"]["upper_multiplier"] = 1.0
        with self.assertRaises(ValueError):
            validate_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
