from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fit_h800_unified_bounded_memory_dual_head_v4 as dual  # noqa: E402
from common import sha256_json  # noqa: E402
from h800_unified_bounded_memory_model import (  # noqa: E402
    ARTIFACT_SCHEMA_V4,
    predict_records,
    validate_artifact,
)


def _record() -> dict[str, object]:
    return {
        "record_id": "unit::dual-head",
        "source_id": "unit-source",
        "origin": "unit",
        "role": "unit",
        "state": "exact",
        "model_id": "unit-model",
        "train_type": "lora",
        "gpu_count": 1,
        "zero_stage": 0,
        "gc": False,
        "packing": False,
        "mbs": 1,
        "cutoff_len": 1024,
        "reference_bytes": 100.0,
        "target_allocated_bytes": 80.0,
        "target_reserved_bytes": 100.0,
        "censor_lower_bytes": None,
        "features": {},
    }


def _artifact() -> dict[str, object]:
    artifact: dict[str, object] = {
        "schema": ARTIFACT_SCHEMA_V4,
        "status": "frozen_shadow_candidate_waiting_validation",
        "immutable": True,
        "publishable": False,
        "production_override_allowed": False,
        "hardware_domain": {"capacity_bytes": 200.0},
        "model": {
            "kind": "allocated_plus_positive_allocator_gap",
            "composition": (
                "reserved=allocated*exp(max(0,predicted_log_reserved_over_allocated))"
            ),
            "allocated_model": {"unit": "allocated"},
            "allocator_gap_model": {"unit": "gap"},
        },
        "admission": {
            "kind": "independent_shared_risk_head",
            "risk_model": {"unit": "risk"},
            "upper_multiplier": 1.0,
            "safe_limit_fraction": 0.95,
        },
    }
    artifact["artifact_sha256"] = sha256_json(artifact)
    return artifact


class UnifiedBoundedMemoryDualHeadV4Tests(unittest.TestCase):
    def test_target_transform_separates_allocated_and_allocator_gap(self) -> None:
        records = [_record(), {**_record(), "record_id": "unit::dual-head-2"}]
        allocated = dual._exact_target_records(records, target="allocated")
        with patch.object(
            dual.bm,
            "_predict_correction",
            return_value=[math.log(0.8), math.log(0.8)],
        ):
            gap = dual._exact_target_records(
                records,
                target="allocator_gap",
                allocated_model={"unit": "allocated"},
            )
        self.assertEqual(allocated[0]["target_reserved_bytes"], 80.0)
        self.assertEqual(gap[0]["target_reserved_bytes"], 125.0)
        self.assertEqual(
            math.log(gap[0]["target_reserved_bytes"] / gap[0]["reference_bytes"]),
            math.log(100.0 / 80.0),
        )

    def test_reserved_only_row_constrains_gap_without_fake_allocated_label(self) -> None:
        reserved_only = {
            **_record(),
            "record_id": "unit::reserved-only",
            "target_allocated_bytes": None,
        }
        records = [
            _record(),
            {**_record(), "record_id": "unit::observed-2"},
            reserved_only,
        ]
        allocated = dual._exact_target_records(records, target="allocated")
        with patch.object(
            dual.bm,
            "_predict_correction",
            return_value=[math.log(0.8), math.log(0.8), math.log(0.8)],
        ):
            gap = dual._exact_target_records(
                records,
                target="allocator_gap",
                allocated_model={"unit": "allocated"},
            )
        self.assertEqual(len(allocated), 2)
        self.assertEqual(len(gap), 3)
        self.assertEqual(
            gap[2]["allocator_gap_target_kind"],
            "composition_residual_reserved_only_partial_label",
        )
        self.assertAlmostEqual(gap[2]["target_reserved_bytes"], 125.0)

    def test_composition_predicts_reserved_from_allocated_and_positive_gap(self) -> None:
        model = {
            "allocated_model": {"unit": "allocated"},
            "allocator_gap_model": {"unit": "gap"},
        }
        with patch.object(
            dual.bm,
            "_predict_correction",
            side_effect=[[math.log(0.8)], [math.log(1.25)]],
        ):
            prediction = dual._predict_dual_center([_record()], model)[0]
        self.assertAlmostEqual(prediction["allocated_center_bytes"], 80.0)
        self.assertAlmostEqual(prediction["reserved_center_bytes"], 100.0)
        self.assertAlmostEqual(prediction["allocator_gap_bytes"], 20.0)

    def test_negative_gap_is_clipped_so_reserved_cannot_below_allocated(self) -> None:
        artifact = _artifact()
        validate_artifact(artifact)
        with patch(
            "h800_unified_bounded_memory_model._predict_correction",
            side_effect=[[math.log(0.8)], [-0.7], [math.log(0.7)]],
        ):
            prediction = predict_records([_record()], artifact)[0]
        self.assertAlmostEqual(prediction["allocated_center_bytes"], 80.0)
        self.assertAlmostEqual(prediction["reserved_center_bytes"], 80.0)
        self.assertGreaterEqual(
            prediction["reserved_center_bytes"],
            prediction["allocated_center_bytes"],
        )

    def test_checksum_mutation_is_rejected(self) -> None:
        artifact = _artifact()
        mutated = json.loads(json.dumps(artifact))
        mutated["model"]["composition"] = "invalid"
        with self.assertRaises(ValueError):
            validate_artifact(mutated)


if __name__ == "__main__":
    unittest.main()
