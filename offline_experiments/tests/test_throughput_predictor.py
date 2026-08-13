from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = EXPERIMENT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from throughput_predictor import (  # noqa: E402
    SCHEMA,
    ThroughputPredictor,
    validate_prediction_report,
)


class ThroughputPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = ThroughputPredictor()

    @staticmethod
    def request(
        request_id: str,
        *,
        model_id: str = "qwen3_4b",
        hardware_id: str = "h800",
        gpu_count: int = 1,
        physical_mbs: int = 4,
        zero_stage: int = 0,
        gc: bool = False,
        packing: bool = False,
        offload: bool = False,
    ) -> dict[str, object]:
        return {
            "request_id": request_id,
            "model_id": model_id,
            "dataset_id": "multiturn_2048",
            "hardware_id": hardware_id,
            "training_mode": "lora",
            "gpu_count": gpu_count,
            "physical_mbs": physical_mbs,
            "target_gbs": 64,
            "cutoff_len": 2048,
            "gradient_checkpointing": gc,
            "zero_stage": zero_stage,
            "packing": packing,
            "offload": offload,
        }

    def test_supported_prediction_is_positive_and_complete(self) -> None:
        prediction = self.predictor.predict(
            self.request("supported")
        )
        self.assertGreater(
            prediction["predicted_effective_tokens_per_second"],
            0.0,
        )
        self.assertGreater(prediction["predicted_step_seconds"], 0.0)
        self.assertEqual(
            prediction["confidence"]["label"],
            "supported",
        )
        self.assertTrue(
            prediction["confidence"]["known_card_adapter_used"]
        )
        self.assertFalse(prediction["memory_safety_checked"])
        self.assertEqual(prediction["rank_within_group"], 1)

    def test_offload_is_explicitly_outside_v1_domain(self) -> None:
        prediction = self.predictor.predict(
            self.request("optimizer-offload", offload=True)
        )
        codes = {reason["code"] for reason in prediction["confidence"]["reasons"]}
        self.assertTrue(prediction["configuration"]["offload"])
        self.assertIn("unsupported_execution_mechanism", codes)
        self.assertIsNone(
            prediction["confidence"][
                "empirical_p90_absolute_percentage_error_reference"
            ]
        )

    def test_existing_matrix_field_aliases_are_accepted(self) -> None:
        prediction = self.predictor.predict(
            {
                "request_id": "matrix-style",
                "model_id": "qwen3_4b",
                "model_parameters": 4_022_468_096,
                "dataset_id": "multiturn_2048",
                "gpu_type": "NVIDIA H800 140GB HBM3",
                "train_type": "lora",
                "gpu_count": 1,
                "mbs": 4,
                "target_gbs": 64,
                "cutoff_len": 2048,
                "gc": False,
                "zero": "none",
                "packing": False,
            }
        )
        self.assertEqual(
            prediction["configuration"]["hardware_id"],
            "h800",
        )
        self.assertEqual(
            prediction["confidence"]["label"],
            "supported",
        )

    def test_catalog_parameter_mismatch_is_rejected(self) -> None:
        request = self.request("bad-parameters")
        request["model_parameters"] = 123
        with self.assertRaisesRegex(
            ValueError,
            "does not match catalog",
        ):
            self.predictor.predict(request)

    def test_batch_ranks_candidates_with_one_shared_output(self) -> None:
        report = self.predictor.predict_many(
            [
                self.request("one"),
                self.request(
                    "two",
                    gpu_count=2,
                    physical_mbs=2,
                    zero_stage=2,
                ),
                self.request(
                    "three",
                    gpu_count=2,
                    physical_mbs=4,
                    zero_stage=3,
                    gc=True,
                ),
            ]
        )
        self.assertEqual(report["schema"], SCHEMA)
        validate_prediction_report(report)
        self.assertTrue(
            report["single_output_used_for_absolute_and_ranking"]
        )
        self.assertTrue(report["requires_memory_safety_filter"])
        self.assertEqual(len(report["ranking_groups"]), 1)
        self.assertEqual(
            {row["rank_within_group"] for row in report["predictions"]},
            {1, 2, 3},
        )
        ranked = report["ranking_groups"][0][
            "predicted_effective_tokens_per_second"
        ]
        self.assertEqual(ranked, sorted(ranked, reverse=True))

    def test_outside_model_scale_is_low_confidence(self) -> None:
        prediction = self.predictor.predict(
            self.request("qwen3-32b", model_id="qwen3_32b")
        )
        codes = {
            reason["code"]
            for reason in prediction["confidence"]["reasons"]
        }
        self.assertEqual(prediction["confidence"]["label"], "low")
        self.assertIn("unseen_model_id", codes)
        self.assertIn("model_scale_outside_training_support", codes)
        self.assertIsNone(
            prediction["confidence"][
                "empirical_p90_absolute_percentage_error_reference"
            ]
        )

    def test_model_seen_only_on_other_card_is_caution(self) -> None:
        prediction = self.predictor.predict(
            self.request(
                "cross-card-model",
                model_id="qwen3_8b",
                hardware_id="rtx4090",
            )
        )
        codes = {
            reason["code"]
            for reason in prediction["confidence"]["reasons"]
        }
        self.assertEqual(
            prediction["confidence"]["label"],
            "caution",
        )
        self.assertIn("unseen_model_on_card", codes)
        self.assertIn("model_scale_outside_card_support", codes)

    def test_packing_emits_numeric_but_low_confidence(self) -> None:
        prediction = self.predictor.predict(
            self.request("packing", packing=True)
        )
        self.assertTrue(
            math.isfinite(
                prediction[
                    "predicted_effective_tokens_per_second"
                ]
            )
        )
        codes = {
            reason["code"]
            for reason in prediction["confidence"]["reasons"]
        }
        self.assertEqual(prediction["confidence"]["label"], "low")
        self.assertIn("packing_not_validated", codes)

    def test_unknown_hardware_uses_shared_model_and_is_low(self) -> None:
        request = self.request("custom-card")
        request.pop("hardware_id")
        request["hardware"] = {
            "hardware_id": "example_gpu",
            "card_id": "example_gpu",
            "memory_bytes": 80_000_000_000,
            "dense_bf16_peak_flops_per_gpu": 500_000_000_000_000,
            "hbm_bandwidth_bytes_per_second": 2_000_000_000_000,
            "intra_node_bandwidth_bytes_per_second": 200_000_000_000,
            "collective_latency_seconds": 0.00001,
            "default_kernel_path": (
                "fa3_orig+liger_fused_ce+adamw_torch_fused"
            ),
        }
        prediction = self.predictor.predict(request)
        codes = {
            reason["code"]
            for reason in prediction["confidence"]["reasons"]
        }
        self.assertIn("unknown_card", codes)
        self.assertFalse(
            prediction["confidence"]["known_card_adapter_used"]
        )
        self.assertGreater(
            prediction["predicted_effective_tokens_per_second"],
            0.0,
        )

    def test_invalid_unpacked_batch_geometry_fails(self) -> None:
        request = self.request("invalid")
        request["target_gbs"] = 63
        with self.assertRaisesRegex(
            ValueError,
            "target_gbs must be divisible",
        ):
            self.predictor.predict(request)

    def test_additional_dataset_profile_is_merged_as_caution(
        self,
    ) -> None:
        lengths = [
            200,
            300,
            400,
            600,
            800,
            1000,
            1200,
            1500,
            1800,
            2000,
        ]
        with tempfile.TemporaryDirectory() as temporary:
            profile_path = (
                Path(temporary)
                / "custom_dataset.qwen3_nothink.jsonl"
            )
            profile_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "total_tokens": length,
                            "label_tokens": int(length * 0.7),
                            "turns": 4,
                        }
                    )
                    + "\n"
                    for length in lengths
                ),
                encoding="utf-8",
            )
            predictor = ThroughputPredictor(
                additional_dataset_profile_dir=Path(temporary)
            )
            request = self.request("new-dataset")
            request["dataset_id"] = "custom_dataset"
            prediction = predictor.predict(request)
        codes = {
            reason["code"]
            for reason in prediction["confidence"]["reasons"]
        }
        self.assertEqual(
            prediction["confidence"]["label"],
            "caution",
        )
        self.assertIn("unseen_dataset_id", codes)
        self.assertAlmostEqual(
            prediction["confidence"][
                "empirical_p90_absolute_percentage_error_reference"
            ],
            0.48601122643698924,
        )

    def test_cli_writes_valid_ranked_report(self) -> None:
        example = (
            EXPERIMENT_ROOT
            / "examples"
            / "throughput_predictor_request.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prediction.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "throughput_predictor.py"),
                    "--input",
                    str(example),
                    "--output",
                    str(output),
                ],
                cwd=EXPERIMENT_ROOT.parent,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], SCHEMA)
            self.assertEqual(len(report["predictions"]), 3)
            self.assertEqual(
                sorted(
                    row["rank_within_group"]
                    for row in report["predictions"]
                ),
                [1, 2, 3],
            )


if __name__ == "__main__":
    unittest.main()
