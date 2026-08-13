from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import sha256_json  # noqa: E402
import h800_challenger_modeling as challenger  # noqa: E402


def native_throughput_observation() -> dict:
    return {
        "schema": challenger.OBSERVATION_SCHEMA,
        "observation_id": "throughput-row",
        "hardware": {"gpu_family": "H800"},
        "configuration": {
            "job": {
                "train_type": "full",
                "zero": "zero2",
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
            "usable_for_throughput_calibration": True,
        },
    }


def throughput_record(*, mbs: int, gc: bool = False) -> dict:
    return {
        "selector": {
            "training_mode": "full",
            "zero_stage": 2,
            "gradient_checkpointing": gc,
            "packing": False,
        },
        "scenario": {
            "gpu_count": 2,
            "physical_mbs": mbs,
            "cutoff_len": 512,
        },
        "model_basis": {"base_parameters": 2_031_739_904},
        "performance": {
            "gradient_accumulation_steps": max(1, 32 // mbs),
            "ideal_seconds": {
                "compute_at_dense_peak": 0.1 * mbs,
                "kernel_hbm_at_physical_peak": 0.2,
                "optimizer_hbm_at_physical_peak": 0.01,
                "collective_payload_at_link_peak": 0.01,
            },
            "communication": {"payload_bytes_per_rank_step": 1_000_000},
            "traffic_bytes_per_rank_step": {
                "kernel_total": 2_000_000,
                "optimizer": 100_000,
            },
        },
    }


def candidate(
    scenario: str,
    *,
    mbs: int,
    observed: float,
    physics: float,
) -> dict:
    return {
        "scenario_id": scenario,
        "scenario": {"model_id": scenario},
        "candidate_key": [2, mbs],
        "record": throughput_record(mbs=mbs),
        "replicates": 1,
        "observed_log_throughput": math.log(observed),
        "physics_log_throughput": math.log(physics),
        "historical": False,
        "observation_ids": [f"{scenario}-{mbs}"],
    }


class H800ChallengerModelingTests(unittest.TestCase):
    def test_throughput_admission_is_core_partition_only(self) -> None:
        row = native_throughput_observation()
        self.assertEqual(
            challenger.throughput_admission_reason(row), "admitted"
        )

        packing = copy.deepcopy(row)
        packing["configuration"]["job"]["packing"] = True
        self.assertEqual(
            challenger.throughput_admission_reason(packing),
            "packing_effect_evidence_excluded",
        )

        unspecified = copy.deepcopy(row)
        unspecified["configuration"].pop("calibration_partition")
        self.assertEqual(
            challenger.throughput_admission_reason(unspecified),
            "partition_not_calibration_or_holdout",
        )

    def test_finite_q95_requires_nineteen_scores(self) -> None:
        insufficient = challenger._finite_upper_quantile(  # noqa: SLF001
            list(range(18))
        )
        sufficient = challenger._finite_upper_quantile(  # noqa: SLF001
            list(range(19))
        )
        self.assertFalse(insufficient["available"])
        self.assertIsNone(insufficient["log_residual_upper"])
        self.assertTrue(sufficient["available"])
        self.assertEqual(sufficient["log_residual_upper"], 18)

    def test_pairwise_ranker_learns_ordering_not_only_absolute_scale(self) -> None:
        rows = [
            candidate("scenario-a", mbs=1, observed=10.0, physics=10.0),
            candidate("scenario-a", mbs=2, observed=20.0, physics=11.0),
            candidate("scenario-b", mbs=1, observed=100.0, physics=100.0),
            candidate("scenario-b", mbs=2, observed=200.0, physics=110.0),
        ]
        model = challenger._fit_pairwise_ranker(  # noqa: SLF001
            rows,
            feature_set="basic",
            alpha=0.01,
            historical_weight=0.0,
            use_physics_base=True,
        )
        metrics = challenger._ranking_evaluation(  # noqa: SLF001
            rows, model, include_details=False
        )
        self.assertEqual(metrics["pooled_pairwise_accuracy"], 1.0)
        self.assertEqual(metrics["scenario_equal_top1_regret"], 0.0)

    def test_validator_rejects_holdout_use(self) -> None:
        report = {
            "schema": challenger.SCHEMA,
            "gpu_experiments_launched": False,
            "queues_mutated": False,
            "publishable": False,
            "production_profile_generated": False,
            "protocol": {
                "holdout_used_for_fit_or_selection": False,
                "holdout_touched_after_both_challengers_frozen": True,
            },
            "data_admission": {
                "memory": {"split": {"disjoint": True, "overlap": []}},
                "throughput": {"split": {"disjoint": True, "overlap": []}},
            },
            "memory": {"frozen_model": {"holdout_rows_in_training": 0}},
            "throughput": {"frozen_model": {"holdout_rows_in_training": 0}},
        }
        report["report_sha256"] = sha256_json(report)
        challenger.validate_report(report)

        leaked = copy.deepcopy(report)
        leaked["protocol"]["holdout_used_for_fit_or_selection"] = True
        leaked["report_sha256"] = sha256_json(
            {
                key: value
                for key, value in leaked.items()
                if key != "report_sha256"
            }
        )
        with self.assertRaisesRegex(ValueError, "Holdout separation"):
            challenger.validate_report(leaked)


if __name__ == "__main__":
    unittest.main()
