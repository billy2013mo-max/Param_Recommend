from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path

OFFLINE = Path(__file__).resolve().parents[1]
SCRIPTS = OFFLINE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import read_jsonl
from evaluate_h800_unified_resource_evidence_v1 import (
    collapse_memory_fit_units,
)
from prepare_h800_unified_resource_evidence_v1 import QUEUE


def _load_launcher():
    path = (
        OFFLINE
        / "unified_resource_staging"
        / "launch_h800_unified_resource_evidence_v1.py"
    )
    spec = importlib.util.spec_from_file_location("unified_resource_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load unified-resource launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UnifiedResourceEvidenceEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.jobs = read_jsonl(QUEUE)

    def test_required_repeat_collapse_produces_156_memory_units(self) -> None:
        observations = []
        for index, job in enumerate(self.jobs):
            observations.append(
                {
                    **job,
                    "classification": "success",
                    "calibration_eligible": True,
                    "exact_center_target_bytes": 1000 + index,
                }
            )
        units = collapse_memory_fit_units(observations)
        self.assertEqual(len(units), 156)
        kinds = Counter(unit["fit_unit_kind"] for unit in units)
        self.assertEqual(
            kinds,
            {
                "single_nonpacking_configuration": 128,
                "critical_profile": 4,
                "packing_arm": 24,
            },
        )
        self.assertEqual(
            Counter(unit["member_count"] for unit in units),
            {1: 128, 3: 24, 5: 4},
        )

    def test_mixed_repeat_outcome_is_not_an_exact_center(self) -> None:
        packing = [
            job
            for job in self.jobs
            if job["evidence_role"] == "unified_packing_matched_formal_fit"
        ][:3]
        observations = []
        for index, job in enumerate(packing):
            success = index < 2
            observations.append(
                {
                    **job,
                    "classification": "success" if success else "oom",
                    "calibration_eligible": True,
                    "exact_center_target_bytes": 1000 if success else None,
                }
            )
        unit = collapse_memory_fit_units(observations)[0]
        self.assertEqual(unit["fit_state"], "mixed_success_oom_instability")
        self.assertIsNone(unit["exact_center_target_bytes"])

    def test_scheduler_preview_uses_disjoint_masks_and_all_eight_gpu_ids(self) -> None:
        launcher = _load_launcher()
        waves = launcher._preview_waves(self.jobs)
        used = set()
        for wave in waves:
            masks = [set(item["gpu_mask"]) for item in wave]
            for index, mask in enumerate(masks):
                used |= mask
                self.assertTrue(mask <= set(range(8)))
                self.assertTrue(
                    all(mask.isdisjoint(other) for other in masks[index + 1 :])
                )
            if any(item["gpu_count"] == 4 for item in wave):
                self.assertEqual(len(wave), 1)
        self.assertEqual(used, set(range(8)))


if __name__ == "__main__":
    unittest.main()
