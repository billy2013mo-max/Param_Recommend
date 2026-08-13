from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_h800_full_historical_coverage_v1 import (  # noqa: E402
    PRIMARY_UNPACKED_MECHANISMS,
    _readiness,
)


class FullHistoricalCoverageAuditTests(unittest.TestCase):
    def test_primary_unpacked_grid_has_ten_mechanisms(self) -> None:
        self.assertEqual(len(PRIMARY_UNPACKED_MECHANISMS), 10)
        self.assertIn((0, False, 1, False), PRIMARY_UNPACKED_MECHANISMS)
        self.assertIn((3, True, 4, False), PRIMARY_UNPACKED_MECHANISMS)

    def test_readiness_requires_source_and_outcome_coverage(self) -> None:
        result = _readiness(
            (3, True, 2, False),
            {
                "independent_sources": 5,
                "negative_boundary_sources": 2,
                "safe_success": 8,
                "unsafe_success": 1,
                "oom_right_censored": 1,
            },
        )
        self.assertTrue(result["fit_evidence_ready"])

    def test_many_rows_from_one_source_are_not_ready(self) -> None:
        result = _readiness(
            (2, False, 4, False),
            {
                "independent_sources": 1,
                "negative_boundary_sources": 1,
                "safe_success": 100,
                "unsafe_success": 20,
                "oom_right_censored": 20,
            },
        )
        self.assertFalse(result["fit_evidence_ready"])
        self.assertEqual(
            result["decision"],
            "supplement_independent_sources_and_boundary_outcomes",
        )


if __name__ == "__main__":
    unittest.main()
