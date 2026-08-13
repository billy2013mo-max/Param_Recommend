from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gpu_telemetry import (
    add_clock_adjusted_mfu,
    aggregate_gpu_telemetry,
    read_nvidia_smi_rows,
)
from collect_results import aggregate_result
from stage_decisions import (
    aggregate_configurations,
    select_fastest,
    throughput_screening_decisions,
)


def telemetry_row(
    *,
    clock_mhz: float = 2640.0,
    thermal: int | None = 0,
    power_cap: int | None = 0,
    sample: int = 0,
) -> dict[str, object]:
    return {
        "time_unix": float(sample),
        "gpu_index": 0,
        "memory_used_mib": 1000.0,
        "utilization_gpu": 100.0,
        "power_draw_w": 350.0,
        "clock_sm_mhz": clock_mhz,
        "temperature_gpu_c": 70.0,
        "fan_speed_percent": 60.0,
        "sw_thermal_slowdown_active": thermal,
        "hw_thermal_slowdown_active": 0 if thermal is not None else None,
        "sw_power_cap_active": power_cap,
    }


class GpuTelemetryTests(unittest.TestCase):
    def test_reported_max_clock_is_not_a_training_baseline(self) -> None:
        rows = [telemetry_row(sample=index) for index in range(10)]
        summary = aggregate_gpu_telemetry(
            rows,
            {
                "max_sm_clock_mhz_reported_by_nvidia_smi": 3105,
                "power_limit_w": 450,
            },
        )

        self.assertEqual(summary["clock_status"], "normal")
        self.assertIsNone(summary["sm_clock_reference_mhz"])
        self.assertIsNone(summary["sm_clock_reference_source"])
        self.assertEqual(summary["sm_clock_spec_max_mhz"], 3105.0)
        self.assertIsNone(summary["busy_clock_p50_ratio"])
        self.assertAlmostEqual(
            summary["busy_clock_p50_to_spec_max_ratio"],
            2640.0 / 3105.0,
        )
        metrics = {"mfu": 0.5, **summary}
        add_clock_adjusted_mfu(metrics)
        self.assertNotIn("clock_adjusted_mfu", metrics)

    def test_explicit_thermal_reason_is_authoritative(self) -> None:
        rows = [telemetry_row(thermal=1)]
        summary = aggregate_gpu_telemetry(rows)

        self.assertEqual(summary["clock_status"], "thermal_limited")
        self.assertEqual(
            summary["clock_status_source"],
            "observed_throttle_reason",
        )
        self.assertAlmostEqual(
            summary["sw_thermal_slowdown_busy_fraction"],
            1.0,
        )

    def test_clock_only_drop_requires_a_calibrated_healthy_reference(self) -> None:
        rows = [
            telemetry_row(
                clock_mhz=2000.0,
                thermal=None,
                power_cap=None,
                sample=index,
            )
            for index in range(10)
        ]
        without_reference = aggregate_gpu_telemetry(rows)
        summary = aggregate_gpu_telemetry(
            rows,
            {"healthy_busy_sm_clock_mhz": 2640},
        )

        self.assertEqual(without_reference["clock_status"], "insufficient_data")
        self.assertEqual(summary["clock_status"], "downclocked")
        self.assertEqual(summary["clock_status_source"], "clock_only")
        self.assertEqual(summary["sm_clock_reference_mhz"], 2640.0)
        self.assertAlmostEqual(summary["busy_clock_p50_ratio"], 2000.0 / 2640.0)

    def test_explicit_power_reason_is_recorded(self) -> None:
        summary = aggregate_gpu_telemetry(
            [telemetry_row(power_cap=1)],
        )

        self.assertEqual(summary["clock_status"], "power_limited")
        self.assertEqual(
            summary["clock_status_source"],
            "observed_throttle_reason",
        )

    def test_partial_limiter_support_is_not_reported_as_normal(self) -> None:
        summary = aggregate_gpu_telemetry(
            [telemetry_row(power_cap=None)],
        )

        self.assertEqual(summary["clock_status"], "insufficient_data")
        self.assertEqual(
            summary["clock_status_source"],
            "partial_throttle_reason_support",
        )
        self.assertFalse(summary["throttle_reason_data_complete"])

    def test_historical_six_column_csv_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nvidia_smi.csv"
            path.write_text(
                "timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm\n"
                "2026/07/20 13:00:00.000,0,1024,99,300,2640\n",
                encoding="utf-8",
            )
            rows = read_nvidia_smi_rows(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gpu_index"], 0)
        self.assertIsNone(rows[0]["temperature_gpu_c"])
        self.assertIsNone(rows[0]["sw_thermal_slowdown_active"])
        summary = aggregate_gpu_telemetry(rows)
        self.assertEqual(summary["clock_status"], "insufficient_data")

    def test_clock_adjusted_mfu_uses_only_a_calibrated_reference_ratio(self) -> None:
        metrics = {"mfu": 0.5, "busy_clock_p50_ratio": 0.95}
        add_clock_adjusted_mfu(metrics)
        self.assertAlmostEqual(metrics["clock_adjusted_mfu"], 0.5 / 0.95)

    def test_thermal_status_does_not_exclude_a_successful_configuration(self) -> None:
        row = {
            "request_id": "request",
            "model_id": "model",
            "train_type": "lora",
            "dataset_id": "dataset",
            "cutoff_len": 512,
            "gpu_count": 1,
            "zero": "zero2",
            "gc": False,
            "mbs": 1,
            "target_gbs": 1,
            "packing": False,
            "repeat": 0,
            "kind": "throughput",
            "job_id": "job",
            "classification": "success",
            "samples_per_second": 2.0,
            "clock_status": "thermal_limited",
        }
        configurations = aggregate_configurations([row])

        self.assertTrue(configurations[0]["complete"])
        self.assertEqual(
            select_fastest(configurations)["clock_status"],
            "thermal_limited",
        )

    def test_zero_second_oom_summary_is_collected_without_reclassification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary)
            (result_dir / "metrics").mkdir()
            (result_dir / "status.json").write_text(
                json.dumps({"classification": "oom"}),
                encoding="utf-8",
            )
            (result_dir / "rendered_run.json").write_text(
                json.dumps(
                    {
                        "job": {
                            "job_id": "oom-job",
                            "model_id": "model",
                            "train_type": "lora",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (result_dir / "metrics" / "summary.rank0.json").write_text(
                json.dumps(
                    {
                        "measured_seconds": 0,
                        "max_allocated": 123,
                        "max_reserved": 456,
                        "measured_totals": {
                            "computed_tokens": 0,
                            "effective_tokens": 0,
                            "logical_samples": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            row = aggregate_result(result_dir, {}, {})

        self.assertEqual(row["classification"], "oom")
        self.assertFalse(row["metrics_available"])
        self.assertEqual(
            row["metrics_unavailable_reason"],
            "non_positive_measured_seconds",
        )
        self.assertEqual(row["max_allocated_bytes"], 123)

    def test_thermal_status_is_visible_but_does_not_block_screen_shortlist(self) -> None:
        row = {
            "request_id": "request",
            "model_id": "model",
            "train_type": "lora",
            "dataset_id": "dataset",
            "cutoff_len": 512,
            "gpu_count": 1,
            "zero": "zero2",
            "gc": False,
            "mbs": 1,
            "target_gbs": 1,
            "packing": False,
            "repeat": 0,
            "kind": "throughput_screen",
            "job_id": "job",
            "classification": "success",
            "samples_per_second": 2.0,
            "clock_status": "thermal_limited",
            "sw_thermal_slowdown_busy_fraction": 0.25,
        }
        report = throughput_screening_decisions(
            [row],
            [{"request_id": "request"}],
            [row],
            top_k=1,
        )
        candidate = report["decisions"][0]["shortlisted"][0]

        self.assertEqual(candidate["clock_status"], "thermal_limited")
        self.assertEqual(candidate["sw_thermal_slowdown_busy_fraction"], 0.25)


if __name__ == "__main__":
    unittest.main()
