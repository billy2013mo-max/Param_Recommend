from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import read_json, read_jsonl, sha256_file  # noqa: E402
from evaluate_h800_hybrid_vl_prospective_acceptance_v1 import evaluate  # noqa: E402
from prepare_h800_hybrid_vl_prospective_acceptance_v1 import (  # noqa: E402
    COMBINED_QUEUE,
    DESIGN,
    FROZEN_PREDICTIONS,
    HYBRID_QUEUE,
    VL_QUEUE,
)


class H800HybridVLProspectiveAcceptanceTests(unittest.TestCase):
    def test_combined_queue_is_exact_ordered_9_plus_18(self) -> None:
        hybrid = read_jsonl(HYBRID_QUEUE)
        vl = read_jsonl(VL_QUEUE)
        combined = read_jsonl(COMBINED_QUEUE)
        self.assertEqual(len(hybrid), 9)
        self.assertEqual(len(vl), 18)
        self.assertEqual(combined, [*hybrid, *vl])
        self.assertEqual(len({row["job_id"] for row in combined}), 27)

    def test_design_binds_queue_and_predictions_before_outcomes(self) -> None:
        design = read_json(DESIGN)
        self.assertFalse(design["gpu_training_started"])
        self.assertFalse(design["scope"]["automatic_execution_allowed"])
        self.assertEqual(design["queues"]["combined"]["sha256"], sha256_file(COMBINED_QUEUE))
        self.assertEqual(design["frozen_predictions"]["sha256"], sha256_file(FROZEN_PREDICTIONS))

    def test_acceptance_state_always_fails_closed_unless_gates_pass(self) -> None:
        report = evaluate(allow_incomplete=True)
        if report["complete"]:
            self.assertEqual(sum(report["classifications"].values()), 27)
            self.assertEqual(
                report["release_scope"]["hybrid_non_packing_automatic_admission"],
                report["tracks"]["hybrid"]["memory"]["all_passed"],
            )
            self.assertEqual(
                report["release_scope"]["vl_image_non_packing_automatic_admission"],
                report["tracks"]["vl_image"]["memory"]["all_passed"]
                and report["tracks"]["vl_image"]["visual_semantics_passed"],
            )
        else:
            self.assertFalse(report["release_scope"]["hybrid_non_packing_automatic_admission"])
            self.assertFalse(report["release_scope"]["vl_image_non_packing_automatic_admission"])
        self.assertFalse(report["release_scope"]["packing_automatic_recommendation"])


if __name__ == "__main__":
    unittest.main()
