"""Tests for the scaling OOM-direction diagnosis."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import diagnose_scaling_oom_direction as diag  # noqa: E402


def _observation(
    job_id: str,
    *,
    model_id: str = "qwen3_14b",
    train_type: str = "lora",
    dataset_id: str = "multiturn_4096",
    target_gbs: int = 64,
    cutoff_len: int = 4096,
    gpu_count: int = 1,
    mbs: int = 2,
    zero: str = "none",
    gc: bool = False,
    outcome: str = "success",
    reserved_bytes: int | None = 120_000_000_000,
    censored: bool = False,
) -> dict:
    censoring = None
    if censored:
        censoring = {
            "kind": "right_censored_memory_demand",
            "constraint": "required_memory_exceeded_available_capacity_at_failure",
            "demand_peak_bytes": None,
            "demand_peak_is_unknown_not_imputed": True,
            "device_capacity_bytes_reported_in_error": 150_141_319_249,
            "free_bytes_reported_in_error": 200_000_000,
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
        "measurements": {"memory": {"max_reserved_bytes": reserved_bytes}},
        "censoring": censoring,
    }


def _write(rows: list[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for row in rows:
        handle.write(json.dumps(row) + "\n")
    handle.close()
    return Path(handle.name)


class TestDetection(unittest.TestCase):
    def test_success_then_oom_on_double_is_detected(self) -> None:
        rows = [
            _observation("scale-a", gpu_count=1, outcome="success"),
            _observation("scale-b", gpu_count=2, outcome="oom", censored=True),
        ]
        transitions = diag.build_transitions(rows)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["from_gpus"], 1)
        self.assertEqual(transitions[0]["to_gpus"], 2)

    def test_double_with_any_success_is_not_a_violation(self) -> None:
        rows = [
            _observation("scale-a", gpu_count=1, outcome="success"),
            _observation("scale-b", gpu_count=2, outcome="oom", censored=True),
            _observation("scale-c", gpu_count=2, outcome="success"),
        ]
        self.assertEqual(diag.build_transitions(rows), [])

    def test_oom_at_low_card_count_is_not_a_direction_violation(self) -> None:
        rows = [
            _observation("scale-a", gpu_count=2, outcome="oom", censored=True),
            _observation("scale-b", gpu_count=4, outcome="success"),
        ]
        self.assertEqual(diag.build_transitions(rows), [])

    def test_non_adjacent_card_counts_are_ignored(self) -> None:
        rows = [
            _observation("scale-a", gpu_count=1, outcome="success"),
            _observation("scale-b", gpu_count=4, outcome="oom", censored=True),
        ]
        self.assertEqual(diag.build_transitions(rows), [])


class TestClassification(unittest.TestCase):
    def test_larger_mbs_is_configuration_confounded(self) -> None:
        rows = [
            _observation("scale-a", gpu_count=1, mbs=2, zero="none"),
            _observation(
                "scale-b",
                gpu_count=2,
                mbs=4,
                zero="zero2",
                outcome="oom",
                censored=True,
            ),
        ]
        transition = diag.build_transitions(rows)[0]
        self.assertEqual(
            transition["classification"], diag.CLASS_CONFIG_CONFOUNDED
        )
        self.assertTrue(transition["usable_as_boundary_evidence"])
        self.assertFalse(transition["requires_failure_triage"])
        self.assertTrue(
            any("per_gpu_mbs_grew" in reason for reason in transition["reasons"])
        )

    def test_lora_zero2_with_fixed_mbs_is_sharding_ineffective(self) -> None:
        rows = [
            _observation("scale-a", gpu_count=2, mbs=2, zero="zero2"),
            _observation(
                "scale-b",
                gpu_count=4,
                mbs=2,
                zero="zero2",
                outcome="oom",
                censored=True,
            ),
        ]
        transition = diag.build_transitions(rows)[0]
        self.assertEqual(
            transition["classification"], diag.CLASS_SHARDING_INEFFECTIVE
        )
        self.assertTrue(
            any("sharding_low_yield" in reason for reason in transition["reasons"])
        )

    def test_full_zero3_fixed_config_is_an_unexplained_label_conflict(self) -> None:
        rows = [
            _observation(
                "scale-a", train_type="full", gpu_count=2, mbs=2, zero="zero3"
            ),
            _observation(
                "scale-b",
                train_type="full",
                gpu_count=4,
                mbs=2,
                zero="zero3",
                outcome="oom",
                censored=True,
            ),
        ]
        transition = diag.build_transitions(rows)[0]
        self.assertEqual(transition["classification"], diag.CLASS_LABEL_CONFLICT)
        self.assertFalse(transition["usable_as_boundary_evidence"])
        self.assertTrue(transition["requires_failure_triage"])
        self.assertIn("triage", transition["disposition"])

    def test_oom_peak_is_never_imputed_and_stays_right_censored(self) -> None:
        rows = [
            _observation("scale-a", gpu_count=1, mbs=2, zero="none"),
            _observation(
                "scale-b",
                gpu_count=2,
                mbs=2,
                zero="none",
                outcome="oom",
                censored=True,
            ),
        ]
        transition = diag.build_transitions(rows)[0]
        self.assertFalse(transition["oom_peak_imputed"])
        self.assertTrue(transition["right_censored"])
        self.assertTrue(
            transition["to_endpoint"]["demand_peak_is_unknown_not_imputed"]
        )

    def test_scenario_fields_must_match(self) -> None:
        rows = [
            _observation("scale-a", gpu_count=1, target_gbs=64),
            _observation(
                "scale-b",
                gpu_count=2,
                target_gbs=128,
                outcome="oom",
                censored=True,
            ),
        ]
        self.assertEqual(diag.build_transitions(rows), [])


class TestReport(unittest.TestCase):
    def test_report_declares_no_side_effects(self) -> None:
        path = _write(
            [
                _observation("scale-a", gpu_count=1, mbs=2),
                _observation(
                    "scale-b", gpu_count=2, mbs=4, outcome="oom", censored=True
                ),
            ]
        )
        report = diag.build_report(observations_path=path)
        self.assertEqual(report["schema"], diag.SCHEMA)
        self.assertEqual(report["status"], "diagnostic_only")
        guarantees = report["guarantees"]
        self.assertFalse(guarantees["creates_gpu_queue"])
        self.assertFalse(guarantees["fits_or_publishes_coefficients"])
        self.assertFalse(guarantees["mutates_frozen_artifacts"])
        self.assertFalse(guarantees["imputes_oom_peaks"])
        self.assertEqual(report["transition_count"], 1)


class TestRealArtifacts(unittest.TestCase):
    def test_real_scaling_runs_have_no_unexplained_direction_violation(self) -> None:
        if not diag.CANONICAL_OBSERVATIONS.exists():
            self.skipTest("canonical observations not present")
        report = diag.build_report()
        counts = report["classification_counts"]
        # Both real violations are physically explained; if this ever becomes
        # non-zero a genuine software/infrastructure triage is required before
        # the record may be used as boundary evidence.
        self.assertEqual(counts.get(diag.CLASS_LABEL_CONFLICT, 0), 0)
        self.assertGreaterEqual(report["transition_count"], 1)
        for transition in report["transitions"]:
            self.assertTrue(transition["usable_as_boundary_evidence"])


if __name__ == "__main__":
    unittest.main()
