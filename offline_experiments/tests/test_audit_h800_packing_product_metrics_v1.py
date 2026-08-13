from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_h800_packing_product_metrics_v1 import (
    OUTPUT,
    memory_boundary_replay,
    memory_source_group_oof,
    throughput_packed_ranking,
)
from common import read_json, sha256_json


class PackingProductMetricsAuditTest(unittest.TestCase):
    def test_memory_source_group_oof_keeps_zero_oom_undefined(self) -> None:
        metrics = memory_source_group_oof()
        self.assertEqual(6, metrics["exact_success_rows"])
        self.assertEqual(3, metrics["profile_or_source_groups"])
        self.assertAlmostEqual(0.08704672477356966, metrics["center_source_equal_mape"])
        self.assertEqual(6, metrics["actual_safe_success_rows"])
        self.assertEqual(4, metrics["admitted_safe_success_rows"])
        self.assertAlmostEqual(2.0 / 3.0, metrics["safe_success_admission_rate"])
        self.assertEqual(0, metrics["oom_rows"])
        self.assertIsNone(metrics["oom_admission_rate"])

    def test_retrospective_boundary_replay_has_unique_oom_units(self) -> None:
        metrics = memory_boundary_replay()
        self.assertEqual(4, metrics["rows"])
        self.assertEqual(2, metrics["exact_success_rows"])
        self.assertAlmostEqual(
            0.010525832804642365, metrics["center_source_equal_mape"]
        )
        self.assertEqual(1, metrics["actual_safe_success_rows"])
        self.assertEqual(1, metrics["admitted_safe_success_rows"])
        self.assertEqual(2, metrics["oom_rows"])
        self.assertEqual(0, metrics["admitted_oom_rows"])
        self.assertEqual(0.0, metrics["oom_admission_rate"])

    def test_packed_only_ranking_population_is_explicit(self) -> None:
        metrics = throughput_packed_ranking()
        self.assertEqual(6, metrics["scenarios"])
        self.assertEqual(12, metrics["candidates"])
        self.assertEqual(2, metrics["minimum_candidates_per_scenario"])
        self.assertEqual(2, metrics["maximum_candidates_per_scenario"])
        self.assertEqual(6, metrics["pairwise_rows"])
        self.assertEqual(6, metrics["pairwise_correct_rows"])
        self.assertEqual(1.0, metrics["scenario_equal_pairwise_accuracy"])
        self.assertEqual(0.0, metrics["scenario_equal_top1_regret"])
        self.assertEqual(1.0, metrics["scenario_equal_hit_at_10_percent"])

    def test_artifact_is_not_publishable_and_checksum_is_valid(self) -> None:
        report = read_json(OUTPUT)
        body = dict(report)
        stored = body.pop("report_sha256")
        self.assertEqual(stored, sha256_json(body))
        self.assertFalse(report["publishable"])
        self.assertFalse(report["production_model_mutated"])
        self.assertFalse(
            report["memory"]["one_joint_prospective_packing_row_available"]
        )
        self.assertFalse(report["decision"]["automatic_packing_recommendation_allowed"])


if __name__ == "__main__":
    unittest.main()
