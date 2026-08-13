from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import read_json, read_jsonl  # noqa: E402
from freeze_h800_packing_profile_phase_c_v1 import (  # noqa: E402
    EXPECTED_SETTINGS,
    _validate_queue,
)
from prepare_h800_packing_profile_phase_c_v1 import (  # noqa: E402
    DESIGN,
    QUEUE,
    STATIC,
)


class PackingProfilePhaseCTest(unittest.TestCase):
    def test_exact_balanced_queue(self) -> None:
        rows = read_jsonl(QUEUE)
        ids = _validate_queue(rows)
        self.assertEqual(24, len(ids))
        self.assertEqual(set(EXPECTED_SETTINGS), {row["setting_id"] for row in rows})

    def test_memory_gate_selects_only_safe_arms(self) -> None:
        preflight = read_json(STATIC)["memory_preflight"]
        selected = [row for row in preflight["rows"] if row["selected_for_queue"]]
        excluded = [row for row in preflight["rows"] if not row["selected_for_queue"]]
        self.assertEqual(8, len(selected))
        self.assertEqual(4, len(excluded))
        self.assertTrue(all(row["execution_preflight_passed"] for row in selected))
        self.assertTrue(all(not row["execution_preflight_passed"] for row in excluded))

    def test_design_forbids_long_cutoff_gc_claim(self) -> None:
        design = read_json(DESIGN)
        self.assertFalse(design["safety_amendment"]["original_long_cutoff_gc_off_admitted"])
        self.assertFalse(design["inference_contract"]["gc_interaction_at_long_cutoff_claim_allowed"])
        self.assertFalse(design["publication_allowed"])


if __name__ == "__main__":
    unittest.main()
