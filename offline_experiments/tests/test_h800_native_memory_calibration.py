from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import h800_native_memory_calibration as native  # noqa: E402
from common import sha256_json  # noqa: E402


def observation() -> dict:
    return {
        "schema": native.OBSERVATION_SCHEMA,
        "observation_id": "native-row",
        "hardware": {"gpu_family": "H800"},
        "configuration": {
            "job": {
                "train_type": "full",
                "zero": "none",
                "gc": False,
                "packing": False,
                "mbs": 4,
            },
            "environment": {"FA3_VARIANT": "orig", "ENABLE_CCE": "0"},
            "calibration_partition": {
                "policy": "model_length_disjoint_h800_v1",
                "role": "calibration",
                "split_unit_id": "model@length",
            },
        },
        "fingerprint": {
            "quality": "complete",
            "calibration_evidence_eligible": True,
            "runtime_mechanism_fingerprint_sha256": "runtime-native",
            "runtime_config": {
                "payload": {
                    "bf16": True,
                    "fp16": False,
                    "flash_attn": "fa3",
                    "enable_liger_kernel": True,
                    "torch_compile": False,
                    "optim": "adamw_torch_fused",
                    "use_reentrant_gc": True,
                    "gradient_checkpointing": False,
                }
            },
        },
        "outcome": {
            "class": "success",
            "usable_for_feasibility_calibration": True,
        },
    }


class H800NativeMemoryCalibrationTests(unittest.TestCase):
    def test_native_admission_requires_predeclared_supported_core_row(self) -> None:
        row = observation()
        self.assertEqual(native.native_admission_reason(row), "admitted")

        no_partition = copy.deepcopy(row)
        no_partition["configuration"].pop("calibration_partition")
        self.assertEqual(
            native.native_admission_reason(no_partition),
            "partition_not_calibration_or_holdout",
        )

        out_of_domain = copy.deepcopy(row)
        out_of_domain["configuration"]["job"]["mbs"] = 32
        self.assertEqual(
            native.native_admission_reason(out_of_domain),
            "mbs_outside_supported_domain",
        )

    def test_packing_pair_rows_never_enter_core_memory_fit(self) -> None:
        packed = observation()
        packed["configuration"]["job"]["packing"] = True
        self.assertEqual(
            native.native_admission_reason(packed),
            "packing_effect_evidence_excluded",
        )

        paired_unpacked = observation()
        paired_unpacked["configuration"]["job"][
            "calibration_evidence_class"
        ] = native.PACKING_EVIDENCE_CLASS
        self.assertEqual(
            native.native_admission_reason(paired_unpacked),
            "packing_effect_evidence_excluded",
        )

    def test_comparison_does_not_call_p95_coverage_monotonic_accuracy(self) -> None:
        def evaluation(center_mape: float, coverage: float) -> dict:
            return {
                "center_accuracy": {
                    "allocated_center": {
                        "absolute_percentage_error": {"mean": center_mape}
                    },
                    "reserved_center": {
                        "absolute_percentage_error": {"mean": center_mape}
                    },
                },
                "operational_safety": {
                    "success_p95_coverage": coverage,
                    "false_safe_oom": 0,
                    "false_reject_safe_success": 0,
                    "prediction_unavailable_rows": 0,
                },
            }

        compared = native._metric_comparison(  # noqa: SLF001
            evaluation(0.10, 0.95), evaluation(0.12, 0.99)
        )
        self.assertEqual(
            compared["success_p95_coverage"]["interpretation"],
            "target_at_least_0.95_not_monotonic_accuracy",
        )
        self.assertGreater(
            compared["allocated_center_mean_ape"]["candidate_minus_baseline"], 0
        )

    def test_report_validator_enforces_holdout_separation(self) -> None:
        report = {
            "schema": native.SCHEMA,
            "publishable": False,
            "production_profile_generated": False,
            "data_admission": {
                "partition_units_disjoint": True,
                "holdout": {
                    "used_for_fit": False,
                    "touched_only_after_candidate_fit_frozen": True,
                },
            },
            "training_populations": {"candidate_contains_holdout_rows": False},
        }
        report["report_sha256"] = sha256_json(report)
        native.validate_report(report)

        leaked = copy.deepcopy(report)
        leaked["data_admission"]["holdout"]["used_for_fit"] = True
        leaked["report_sha256"] = sha256_json(
            {key: value for key, value in leaked.items() if key != "report_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "Holdout separation"):
            native.validate_report(leaked)


if __name__ == "__main__":
    unittest.main()
