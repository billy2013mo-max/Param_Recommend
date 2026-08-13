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

from common import sha256_file, sha256_json  # noqa: E402
from h800_packing_effect import (  # noqa: E402
    OBSERVATION_SCHEMA,
    RECOVERY_SCHEMA,
    build_report,
    validate_report,
)


def observation(index: int, *, packed: bool, effective_rate: float) -> dict:
    return {
        "schema": OBSERVATION_SCHEMA,
        "observation_id": f"observation-{index}",
        "configuration": {
            "job": {
                "job_id": f"job-{index}",
                "model_id": "qwen",
                "train_type": "lora",
                "dataset_id": "data",
                "target_gbs": 64,
                "gpu_count": 1,
                "zero": "none",
                "gc": False,
                "mbs": 1,
                "cutoff_len": 2048,
                "packing": packed,
            }
        },
        "outcome": {"class": "success"},
        "measurements": {
            "rates": {
                "effective_tokens_per_second": effective_rate,
                "logical_samples_per_second": effective_rate / 100,
                "computed_tokens_per_second": effective_rate * 1.1,
            },
            "memory": {"max_reserved_bytes": 10_000 + index},
        },
    }


def recovery_record(row: dict, *, pair_id: str, treatment: str) -> dict:
    return {
        "source_observation_id": row["observation_id"],
        "measurement_eligibility": {"packing_pair": True},
        "validation_design": {"pair_id": pair_id, "treatment": treatment},
        "runtime": {"runtime_cohort_id": "runtime"},
    }


class PackingEffectTests(unittest.TestCase):
    def build_fixture(self, directory: Path) -> tuple[Path, Path]:
        unpacked = observation(1, packed=False, effective_rate=100)
        packed = observation(2, packed=True, effective_rate=125)
        observations = directory / "observations.jsonl"
        observations.write_text(
            "".join(json.dumps(row) + "\n" for row in (unpacked, packed)),
            encoding="utf-8",
        )
        recovery = {
            "schema": RECOVERY_SCHEMA,
            "report_sha256": "recovery",
            "records": [
                recovery_record(unpacked, pair_id="pair", treatment="unpacked"),
                recovery_record(packed, pair_id="pair", treatment="packed"),
            ],
        }
        recovery_path = directory / "recovery.json"
        recovery_path.write_text(json.dumps(recovery), encoding="utf-8")
        return observations, recovery_path

    def test_builds_low_confidence_non_enabling_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            observation_path, recovery_path = self.build_fixture(Path(temporary))
            report = build_report(observation_path, recovery_path)

        self.assertEqual(report["aggregate"]["pairs"], 1)
        self.assertEqual(
            report["aggregate"]["median_effective_token_throughput_ratio"],
            1.25,
        )
        self.assertFalse(report["automatic_enablement_allowed"])
        self.assertFalse(report["publishable"])
        validate_report(report)

    def test_rejects_non_packing_dimension_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observation_path, recovery_path = self.build_fixture(root)
            rows = [json.loads(line) for line in observation_path.read_text().splitlines()]
            rows[1]["configuration"]["job"]["gpu_count"] = 2
            observation_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-packing dimension"):
                build_report(observation_path, recovery_path)

    def test_validation_rejects_enabling_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            observation_path, recovery_path = self.build_fixture(Path(temporary))
            report = build_report(observation_path, recovery_path)
        tampered = copy.deepcopy(report)
        tampered["automatic_enablement_allowed"] = True
        content = dict(tampered)
        content.pop("report_sha256")
        tampered["report_sha256"] = sha256_json(content)
        with self.assertRaisesRegex(ValueError, "never enable"):
            validate_report(tampered)

    def test_source_files_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            observation_path, recovery_path = self.build_fixture(Path(temporary))
            report = build_report(observation_path, recovery_path)
            self.assertEqual(
                report["source_bindings"]["observations"]["sha256"],
                sha256_file(observation_path),
            )
            self.assertEqual(
                report["source_bindings"]["historical_recovery"]["sha256"],
                sha256_file(recovery_path),
            )


if __name__ == "__main__":
    unittest.main()
