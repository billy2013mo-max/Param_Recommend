from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import sha256_json  # noqa: E402
from memory_anchor_registry import (  # noqa: E402
    build_registry,
    load_registry,
    validate_registry,
)


REGISTRY_PATH = ROOT / "artifacts" / "h800_memory_anchor_registry_v1.json"


class MemoryAnchorRegistryTests(unittest.TestCase):
    def test_frozen_registry_has_one_guarded_active_anchor(self) -> None:
        registry = load_registry(REGISTRY_PATH)
        self.assertTrue(registry["immutable"])
        self.assertEqual(registry["active_anchor_count"], 1)
        anchor = registry["anchors"][0]
        self.assertEqual(anchor["evidence"]["successes"], 3)
        self.assertEqual(
            anchor["evidence"]["complete_calibration_eligible_successes"],
            2,
        )
        self.assertEqual(anchor["evidence"]["ooms"], 0)
        self.assertLess(
            anchor["evidence"]["empirical_guard_bytes"],
            anchor["evidence"]["safe_limit_bytes"],
        )

    def test_registry_rebuild_reproduces_evidence(self) -> None:
        report = build_registry(
            ROOT / "config" / "memory_anchor_candidates_v1.json",
            ROOT / "artifacts" / "canonical_h800_observations.jsonl",
            ROOT / "config" / "hardware.json",
        )
        anchor = report["anchors"][0]
        self.assertTrue(anchor["memory_gate_override_allowed"])
        self.assertEqual(
            anchor["evidence"]["complete_job_ids"],
            [
                "tputh-190fe44c243eef61",
                "tputh2-a361aff82b9d5a5",
            ],
        )

    def test_registry_checksum_tamper_is_rejected(self) -> None:
        registry = load_registry(REGISTRY_PATH)
        tampered = copy.deepcopy(registry)
        tampered["governance"]["observed_peak_relative_guard"] = 0.0
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_registry(tampered)

    def test_active_anchor_with_oom_is_rejected_even_with_valid_checksum(
        self,
    ) -> None:
        registry = load_registry(REGISTRY_PATH)
        tampered = copy.deepcopy(registry)
        tampered["anchors"][0]["evidence"]["ooms"] = 1
        unsigned = {
            key: value for key, value in tampered.items() if key != "report_sha256"
        }
        tampered["report_sha256"] = sha256_json(unsigned)
        with self.assertRaisesRegex(ValueError, "contains an OOM"):
            validate_registry(tampered)

    def test_saved_registry_can_be_loaded_from_an_independent_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registry.json"
            path.write_text(REGISTRY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            loaded = load_registry(path)
        self.assertEqual(
            loaded["report_sha256"],
            json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["report_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
