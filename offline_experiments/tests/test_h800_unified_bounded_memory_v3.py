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
import h800_unified_bounded_memory_v3_data as v3_data
from h800_unified_bounded_memory_model import (
    ARTIFACT_SCHEMA_V3,
    load_artifact,
    predict_records,
    validate_artifact,
)

ARTIFACT = ROOT / "artifacts" / "h800_unified_bounded_memory_candidate_v3.json"


class UnifiedBoundedMemoryV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = load_artifact(ARTIFACT)
        cls.records, _audit = v3_data.development_records()

    def test_frozen_v3_contract_and_checksum(self) -> None:
        validate_artifact(self.artifact)
        self.assertEqual(self.artifact["schema"], ARTIFACT_SCHEMA_V3)
        self.assertFalse(self.artifact["publishable"])
        self.assertFalse(self.artifact["production_override_allowed"])

    def test_v3_mechanism_features_are_single_shared_basis(self) -> None:
        center_names = set(self.artifact["model"]["raw_feature_names"])
        risk_names = set(self.artifact["admission"]["risk_model"]["raw_feature_names"])
        self.assertTrue(set(benchmark.V3_MECHANISM_FEATURE_NAMES) <= center_names)
        self.assertEqual(center_names, risk_names)
        self.assertFalse(self.artifact["model"]["source_or_dataset_id_used_as_feature"])

    def test_all_known_oom_rows_are_rejected(self) -> None:
        oom = [row for row in self.records if row["state"] == "censored"]
        predictions = predict_records(oom, self.artifact)
        self.assertEqual(len(oom), 106)
        self.assertEqual(
            [row["record_id"] for row in predictions if row["admitted"]], []
        )

    def test_checksum_mutation_is_rejected(self) -> None:
        mutated = json.loads(ARTIFACT.read_text())
        mutated["admission"]["upper_multiplier"] = 1.0
        with self.assertRaises(ValueError):
            validate_artifact(mutated)


if __name__ == "__main__":
    unittest.main()
