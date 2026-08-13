"""Tests for the read-only scaling-pair evidence audit."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import audit_scaling_pair_evidence as audit  # noqa: E402


def _observation(
    job_id: str,
    *,
    model_id: str = "qwen3_8b",
    train_type: str = "lora",
    dataset_id: str = "short_512",
    target_gbs: int = 64,
    cutoff_len: int = 512,
    gpu_count: int = 1,
    mbs: int = 4,
    zero: str = "none",
    gc: bool = False,
    outcome: str = "success",
    effective_rate: float | None = 1000.0,
    reserved_bytes: int = 10_000,
) -> dict:
    rates = {}
    if effective_rate is not None:
        rates = {
            "effective_tokens_per_second": effective_rate,
            "computed_tokens_per_second": effective_rate * 1.5,
        }
    return {
        "observation_id": f"obs-{job_id}",
        "configuration": {
            "job": {
                "job_id": job_id,
                "model_id": model_id,
                "train_type": train_type,
                "dataset_id": dataset_id,
                "target_gbs": target_gbs,
                "cutoff_len": cutoff_len,
                "packing": False,
                "gpu_count": gpu_count,
                "mbs": mbs,
                "zero": zero,
                "gc": gc,
            }
        },
        "outcome": {"class": outcome},
        "measurements": {
            "mean_step_seconds": 1.0,
            "measured_step_count": 8,
            "rates": rates,
            "memory": {
                "max_reserved_bytes": reserved_bytes,
                "values_are_observed_not_imputed": True,
            },
        },
        "fingerprint": {"calibration_evidence_eligible": False},
    }


def _write_jsonl(rows: list[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for row in rows:
        handle.write(json.dumps(row) + "\n")
    handle.close()
    return Path(handle.name)


class TestLoadAndGroup(unittest.TestCase):
    def test_only_scale_prefixed_jobs_are_loaded(self) -> None:
        path = _write_jsonl(
            [
                _observation("scale-a"),
                _observation("tput-b"),
                _observation("mem-c"),
            ]
        )
        rows = audit.load_scaling_observations(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["configuration"]["job"]["job_id"], "scale-a")

    def test_endpoints_group_by_scenario_then_gpu_count(self) -> None:
        rows = [
            _observation("scale-a", gpu_count=1),
            _observation("scale-b", gpu_count=2),
            _observation("scale-c", gpu_count=2, dataset_id="multiturn_4096"),
        ]
        grouped = audit.group_endpoints(rows)
        self.assertEqual(len(grouped), 2)
        counts = {
            key: sorted(bucket["by_gpu_count"]) for key, bucket in grouped.items()
        }
        self.assertIn([1, 2], counts.values())
        self.assertIn([2], counts.values())


class TestPairConstruction(unittest.TestCase):
    def test_adjacent_doubling_ratio_uses_effective_tokens(self) -> None:
        rows = [
            _observation("scale-a", gpu_count=2, zero="zero2", effective_rate=1000.0),
            _observation("scale-b", gpu_count=4, zero="zero2", effective_rate=1900.0),
        ]
        pairs = audit.build_pairs(audit.group_endpoints(rows))
        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        self.assertTrue(pair["ratio_available"])
        self.assertAlmostEqual(pair["measured_ratio_point_estimate"], 1.9)
        self.assertTrue(pair["clears_1p8_point_estimate"])
        self.assertTrue(pair["clean_scaling_measurement"])

    def test_non_adjacent_counts_do_not_pair(self) -> None:
        rows = [
            _observation("scale-a", gpu_count=1, zero="none"),
            _observation("scale-b", gpu_count=4, zero="zero2"),
        ]
        pairs = audit.build_pairs(audit.group_endpoints(rows))
        self.assertEqual(pairs, [])

    def test_zero_stage_switch_is_flagged_as_not_clean(self) -> None:
        rows = [
            _observation("scale-a", gpu_count=1, zero="none", effective_rate=1000.0),
            _observation("scale-b", gpu_count=2, zero="zero2", effective_rate=1900.0),
        ]
        pair = audit.build_pairs(audit.group_endpoints(rows))[0]
        self.assertTrue(pair["ratio_available"])
        self.assertTrue(pair["zero_stage_changed"])
        self.assertFalse(pair["clean_scaling_measurement"])

    def test_oom_endpoint_blocks_the_ratio_and_is_never_imputed(self) -> None:
        rows = [
            _observation("scale-a", gpu_count=2, zero="zero2"),
            _observation(
                "scale-b",
                gpu_count=4,
                zero="zero2",
                outcome="oom",
                effective_rate=None,
            ),
        ]
        pair = audit.build_pairs(audit.group_endpoints(rows))[0]
        self.assertFalse(pair["ratio_available"])
        self.assertIn("all_runs_oom", pair["blocked_reason"])
        self.assertNotIn("measured_ratio_point_estimate", pair)

    def test_repeats_use_median_and_report_spread(self) -> None:
        rows = [
            _observation("scale-a1", gpu_count=2, zero="zero2", effective_rate=1000.0),
            _observation("scale-a2", gpu_count=2, zero="zero2", effective_rate=1200.0),
            _observation("scale-b", gpu_count=4, zero="zero2", effective_rate=2200.0),
        ]
        pair = audit.build_pairs(audit.group_endpoints(rows))[0]
        self.assertEqual(pair["from_endpoint"]["successful_runs"], 2)
        self.assertAlmostEqual(
            pair["from_endpoint"]["effective_tokens_per_second_median"], 1100.0
        )
        self.assertAlmostEqual(pair["from_endpoint"]["relative_spread"], 0.2)
        self.assertFalse(pair["from_endpoint"]["single_run_point_estimate"])
        self.assertFalse(pair["both_endpoints_single_run"])

    def test_scenario_fields_must_match_to_form_a_pair(self) -> None:
        rows = [
            _observation("scale-a", gpu_count=2, zero="zero2", target_gbs=64),
            _observation("scale-b", gpu_count=4, zero="zero2", target_gbs=128),
        ]
        self.assertEqual(audit.build_pairs(audit.group_endpoints(rows)), [])


class TestReport(unittest.TestCase):
    def test_report_is_diagnostic_and_declares_no_side_effects(self) -> None:
        path = _write_jsonl(
            [
                _observation("scale-a", gpu_count=2, zero="zero2", effective_rate=1000.0),
                _observation("scale-b", gpu_count=4, zero="zero2", effective_rate=1900.0),
            ]
        )
        report = audit.build_report(
            observations_path=path, calibration_path=Path("/nonexistent.json")
        )
        self.assertEqual(report["schema"], audit.SCHEMA)
        self.assertEqual(report["status"], "diagnostic_only")
        guarantees = report["guarantees"]
        self.assertFalse(guarantees["creates_gpu_queue"])
        self.assertFalse(guarantees["fits_or_publishes_coefficients"])
        self.assertFalse(guarantees["mutates_frozen_artifacts"])
        self.assertFalse(guarantees["ratios_are_publishable_scale_out_claims"])
        self.assertFalse(report["calibration_surfaced"]["available"])
        self.assertEqual(report["coverage"]["usable_pairs"], 1)

    def test_coverage_counts_by_dataset_and_train_type(self) -> None:
        path = _write_jsonl(
            [
                _observation("scale-a", gpu_count=2, zero="zero2"),
                _observation("scale-b", gpu_count=4, zero="zero2"),
                _observation(
                    "scale-c",
                    gpu_count=2,
                    zero="zero2",
                    dataset_id="multiturn_4096",
                    train_type="full",
                ),
                _observation(
                    "scale-d",
                    gpu_count=4,
                    zero="zero2",
                    dataset_id="multiturn_4096",
                    train_type="full",
                ),
            ]
        )
        report = audit.build_report(
            observations_path=path, calibration_path=Path("/nonexistent.json")
        )
        coverage = report["coverage"]
        self.assertEqual(coverage["usable_pairs"], 2)
        self.assertEqual(coverage["by_dataset_id"]["short_512"], 1)
        self.assertEqual(coverage["by_dataset_id"]["multiturn_4096"], 1)
        self.assertEqual(coverage["by_train_type"]["lora"], 1)
        self.assertEqual(coverage["by_train_type"]["full"], 1)

    def test_single_run_endpoints_are_reported_as_a_finding(self) -> None:
        path = _write_jsonl(
            [
                _observation("scale-a", gpu_count=2, zero="zero2"),
                _observation("scale-b", gpu_count=4, zero="zero2"),
            ]
        )
        report = audit.build_report(
            observations_path=path, calibration_path=Path("/nonexistent.json")
        )
        self.assertIn(
            "every_usable_pair_rests_on_single_run_endpoints", report["findings"]
        )


class TestRealArtifacts(unittest.TestCase):
    def test_real_runs_support_more_pairs_than_calibration_surfaced(self) -> None:
        if not audit.CANONICAL_OBSERVATIONS.exists():
            self.skipTest("canonical observations not present")
        report = audit.build_report()
        surfaced = report["calibration_surfaced"]
        self.assertTrue(surfaced["available"])
        self.assertGreater(
            report["coverage"]["usable_pairs"], surfaced["unique_test_pair_count"]
        )
        self.assertIn(
            "measured_runs_support_more_pairs_than_calibration_surfaced",
            report["findings"],
        )
        # The audit must widen dataset coverage beyond the single long-context
        # slice the calibration report exposed.
        self.assertGreaterEqual(len(report["coverage"]["by_dataset_id"]), 3)
        self.assertIn("full", report["coverage"]["by_train_type"])


if __name__ == "__main__":
    unittest.main()
