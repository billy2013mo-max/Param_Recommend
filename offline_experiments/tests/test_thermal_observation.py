from __future__ import annotations

import csv
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_job import (  # noqa: E402
    NVIDIA_SMI_FIELDS,
    classify,
    monitor_nvidia_smi,
    summarize_nvidia_smi,
)


class ThermalObservationTests(unittest.TestCase):
    def test_monitor_fields_capture_clock_and_throttle_reasons(self) -> None:
        self.assertIn("clocks.sm", NVIDIA_SMI_FIELDS)
        self.assertIn("temperature.gpu", NVIDIA_SMI_FIELDS)
        self.assertIn("fan.speed", NVIDIA_SMI_FIELDS)
        self.assertIn(
            "clocks_event_reasons.sw_thermal_slowdown",
            NVIDIA_SMI_FIELDS,
        )
        self.assertIn(
            "clocks_event_reasons.hw_thermal_slowdown",
            NVIDIA_SMI_FIELDS,
        )
        self.assertIn("clocks_event_reasons.sw_power_cap", NVIDIA_SMI_FIELDS)

    def test_summary_is_record_only_and_reports_busy_thermal_samples(self) -> None:
        rows = [
            [
                "2026/07/20 12:00:00.000",
                "0",
                "20000",
                "100",
                "420",
                "2700",
                "80",
                "90",
                "Not Active",
                "Not Active",
                "Active",
            ],
            [
                "2026/07/20 12:00:01.000",
                "0",
                "20000",
                "100",
                "360",
                "1800",
                "89",
                "100",
                "Active",
                "Not Active",
                "Not Active",
            ],
            [
                "2026/07/20 12:00:02.000",
                "0",
                "1",
                "0",
                "80",
                "2715",
                "55",
                "45",
                "Not Active",
                "Not Active",
                "Not Active",
            ],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nvidia_smi.csv"
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(NVIDIA_SMI_FIELDS)
                writer.writerows(rows)
            summary = summarize_nvidia_smi(path, {0})

        self.assertEqual(summary["policy"], "record_only")
        self.assertFalse(summary["affects_job_classification"])
        self.assertTrue(summary["monitor_data_available"])
        self.assertTrue(summary["any_sw_thermal_slowdown"])
        gpu = summary["gpus"][0]
        self.assertEqual(gpu["samples"], 3)
        self.assertEqual(gpu["busy_samples"], 2)
        self.assertEqual(gpu["max_temperature_c"], 89.0)
        self.assertEqual(gpu["max_fan_percent"], 100.0)
        self.assertEqual(gpu["min_busy_sm_clock_mhz"], 1800.0)
        self.assertEqual(gpu["median_busy_sm_clock_mhz"], 2250.0)
        self.assertEqual(gpu["sw_thermal_slowdown_busy_samples"], 1)
        self.assertEqual(gpu["sw_thermal_slowdown_busy_fraction"], 0.5)
        self.assertEqual(gpu["hw_thermal_slowdown_samples"], 0)
        self.assertEqual(gpu["sw_power_cap_samples"], 1)

    def test_monitor_queries_expected_fields_and_filters_physical_gpus(self) -> None:
        stop = threading.Event()
        health = {}
        output_rows = "\n".join(
            (
                "2026/07/20 12:00:00.000, 0, 100, 90, 300, 2600, 80, 90, Not Active, Not Active, Active",
                "2026/07/20 12:00:00.000, 1, 200, 100, 350, 1900, 89, 100, Active, Not Active, Not Active",
            )
        )

        def fake_run(command, **kwargs):
            stop.set()
            self.assertIn(
                "--query-gpu=" + ",".join(NVIDIA_SMI_FIELDS),
                command,
            )
            self.assertEqual(kwargs["timeout"], 5)
            return CompletedProcess(command, 0, stdout=output_rows, stderr="")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nvidia_smi.csv"
            with patch("run_job.subprocess.run", side_effect=fake_run):
                monitor_nvidia_smi(stop, path, {1}, 0.01, health)
            with path.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))

        self.assertEqual([row["index"] for row in rows], ["1"])
        self.assertEqual(health["attempts"], 1)
        self.assertEqual(health["failures"], 0)
        self.assertEqual(health["samples_written"], 1)

    def test_thermal_observation_never_changes_training_classification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            log_path = result_dir / "train.log"
            log_path.write_text("thermal slowdown active\n", encoding="utf-8")
            metrics = result_dir / "metrics"
            metrics.mkdir()
            (metrics / "summary.rank0.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(classify(0, log_path, 1), "success")
            self.assertEqual(classify(1, log_path, 1), "failed")
            log_path.write_text("CUDA out of memory\n", encoding="utf-8")
            self.assertEqual(classify(1, log_path, 1), "oom")


if __name__ == "__main__":
    unittest.main()
