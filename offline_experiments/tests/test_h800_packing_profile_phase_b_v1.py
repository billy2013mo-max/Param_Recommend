from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import read_json, read_jsonl, sha256_file  # noqa: E402
from freeze_h800_packing_profile_phase_b_v1 import (  # noqa: E402
    DESIGN,
    QUEUE,
    QUEUE_MANIFEST,
    STATIC,
    _validate_queue,
)


class PackingProfilePhaseBTest(unittest.TestCase):
    def test_exact_matrix_and_bindings(self) -> None:
        rows = read_jsonl(QUEUE)
        ids = _validate_queue(rows)
        self.assertEqual(len(ids), 36)
        design = read_json(DESIGN)
        manifest = read_json(QUEUE_MANIFEST)
        self.assertEqual(design["queue"]["ordered_job_ids"], ids)
        self.assertEqual(design["queue"]["sha256"], sha256_file(QUEUE))
        self.assertEqual(manifest["design"]["sha256"], sha256_file(DESIGN))
        self.assertFalse(
            design["inference_contract"]["best_branch_route_effect_claim_allowed"]
        )

    def test_exact_profiles_and_memory_preflight(self) -> None:
        static = read_json(STATIC)
        self.assertTrue(static["exact_profile_consistency"]["all_passed"])
        self.assertEqual(
            len(static["exact_profile_consistency"]["checks"]), 6
        )
        checks = static["exact_profile_consistency"]["checks"]
        self.assertTrue(
            all(
                check["first_epoch_capacity"]["unpacked"]["passed"]
                and check["first_epoch_capacity"]["packed"]["passed"]
                for check in checks
            )
        )
        repeated = {
            check["workload_id"]: check
            for check in checks
            if check["workload_id"] in {"W3", "W5"}
        }
        self.assertEqual(set(repeated), {"W3", "W5"})
        self.assertTrue(
            all(
                check["source_records"] == 1_000
                and check["runtime_records"] == 2_000
                and check["probe_repetition_factor"] == 2
                for check in repeated.values()
            )
        )
        preflight = static["memory_preflight"]
        self.assertTrue(preflight["all_passed"])
        self.assertFalse(preflight["automatic_packing_admission_allowed"])
        self.assertEqual(len(preflight["rows"]), 12)
        self.assertTrue(
            all(
                row["execution_guarded_upper_bytes"]
                <= row["execution_limit_bytes"]
                for row in preflight["rows"]
            )
        )


if __name__ == "__main__":
    unittest.main()
