#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = EXPERIMENT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import sha256_json  # noqa: E402
from h800_physical_v4b_predictor import (  # noqa: E402
    H800PhysicalV4BPredictor,
    SCHEMA,
    SELECTION_POLICY,
    validate_prediction_report,
)


class H800PhysicalV4BPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = H800PhysicalV4BPredictor()
        example = EXPERIMENT_ROOT / "examples" / "h800_physical_v4b_request.json"
        cls.requests = json.loads(example.read_text(encoding="utf-8"))["candidates"]

    def test_memory_gate_then_v4b_ranking(self) -> None:
        report = self.predictor.predict(self.requests)
        validate_prediction_report(report)

        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["release"]["mode"], "shadow_only")
        self.assertFalse(report["release"]["automatic_execution_allowed"])
        group = report["ranking_groups"][0]
        self.assertEqual(group["requested_candidates"], 4)
        self.assertEqual(group["policy_admitted_candidates"], 3)
        self.assertEqual(group["rejected_by_memory"], 1)
        self.assertEqual(
            group["selection_policy"],
            SELECTION_POLICY,
        )
        self.assertEqual(group["minimum_admitted_gpu_count"], 4)
        self.assertEqual(
            group["selected_request_id"],
            "qwen3-14b-full-gpu4-z3-mbs8",
        )
        self.assertEqual(
            group["throughput_top_request_id"],
            "qwen3-14b-full-gpu4-z3-mbs8",
        )
        self.assertEqual(
            group["ranked_request_ids"],
            [
                "qwen3-14b-full-gpu4-z3-mbs8",
                "qwen3-14b-full-gpu4-z2-mbs8-gc",
                "qwen3-14b-full-gpu4-z3-mbs4",
            ],
        )

        ranked = [
            row
            for row in report["predictions"]
            if row["throughput"]["prediction_available"]
        ]
        rejected = [
            row for row in report["predictions"] if not row["memory"]["admitted"]
        ]
        self.assertEqual(
            sorted(row["rank_within_admitted_group"] for row in ranked),
            [1, 2, 3],
        )
        self.assertEqual(
            sorted(row["rank_within_gpu_count"] for row in ranked),
            [1, 2, 3],
        )
        self.assertTrue(
            all(row["throughput"]["absolute_scale_trusted"] is False for row in ranked)
        )
        self.assertEqual(len(rejected), 1)
        self.assertEqual(
            rejected[0]["memory"]["rejection_reason"],
            "memory_upper_exceeds_safe_limit",
        )
        self.assertFalse(rejected[0]["throughput"]["prediction_available"])
        self.assertIsNone(rejected[0]["rank_within_gpu_count"])

    def test_minimum_admitted_gpu_count_precedes_global_top(self) -> None:
        candidates = []
        for gpu_count, zero_stage, accumulation in (
            (1, 0, 32),
            (2, 2, 16),
            (4, 2, 8),
        ):
            candidate = deepcopy(self.requests[0])
            candidate.update(
                {
                    "request_id": f"qwen3-8b-lora-gpu{gpu_count}",
                    "comparison_group": "qwen3-8b-lora-multiturn4096",
                    "model_id": "qwen3_8b",
                    "training_mode": "lora",
                    "dataset_id": "multiturn_4096",
                    "dataset_category": "multiturn",
                    "cutoff_len": 4096,
                    "gpu_count": gpu_count,
                    "physical_mbs": 2,
                    "zero_stage": zero_stage,
                    "gradient_accumulation_steps": accumulation,
                    "gradient_checkpointing": False,
                }
            )
            candidates.append(candidate)

        report = self.predictor.predict(candidates)
        group = report["ranking_groups"][0]
        rows = {row["request_id"]: row for row in report["predictions"]}

        self.assertEqual(group["minimum_admitted_gpu_count"], 1)
        self.assertEqual(
            rows[group["selected_request_id"]]["configuration"]["gpu_count"],
            1,
        )
        self.assertEqual(
            rows[group["selected_request_id"]]["rank_within_gpu_count"],
            1,
        )
        self.assertEqual(
            rows[group["throughput_top_request_id"]]["rank_within_admitted_group"],
            1,
        )

    def test_scale_out_contract_is_visible_but_remains_disabled(self) -> None:
        report = self.predictor.predict(self.requests)
        scale_out = report["scale_out"]
        self.assertFalse(scale_out["enabled"])
        self.assertFalse(scale_out["automatic_execution_allowed"])
        self.assertEqual(
            scale_out["selection_policy"],
            "thresholded_doubling_then_v4b",
        )
        self.assertEqual(
            scale_out["status"],
            "disabled_pending_conservative_ratio_calibration",
        )
        self.assertEqual(len(scale_out["groups"]), 1)

    def test_offload_is_rejected_instead_of_treated_as_no_offload(self) -> None:
        request = deepcopy(self.requests[0])
        request["request_id"] = "qwen3-14b-full-offload"
        request["offload_optimizer"] = {"ratio": 0.4}
        report = self.predictor.predict([request])
        row = report["predictions"][0]
        self.assertTrue(row["configuration"]["offload"])
        self.assertEqual(row["support"]["support_tier"], "unsupported")
        self.assertEqual(
            row["support"]["required_experiment_family"],
            "offload_mechanism_calibration",
        )
        self.assertFalse(row["memory"]["admitted"])
        self.assertEqual(
            row["memory"]["rejection_reason"],
            "outside_supported_domain",
        )
        self.assertFalse(row["throughput"]["prediction_available"])

    def test_versioned_anchor_recovers_exact_14b_two_gpu_false_reject(self) -> None:
        request = deepcopy(self.requests[0])
        request.update(
            {
                "request_id": "qwen3-14b-full-anchor-recovery",
                "comparison_group": "qwen3-14b-full-anchor-recovery",
                "dataset_id": "multiturn_4096",
                "dataset_category": "multiturn",
                "target_gbs": 64,
                "cutoff_len": 4096,
                "gpu_count": 2,
                "physical_mbs": 2,
                "zero_stage": 3,
                "gradient_checkpointing": True,
                "packing": False,
            }
        )

        report = self.predictor.predict([request])
        row = report["predictions"][0]
        memory = row["memory"]

        self.assertFalse(memory["base_physical_model_admitted"])
        self.assertTrue(memory["anchor_override_applied"])
        self.assertEqual(
            memory["admission_source"],
            "versioned_historical_anchor",
        )
        self.assertTrue(memory["physical_model_admitted"])
        self.assertTrue(memory["admitted"])
        self.assertGreater(
            memory["operational_p95_reserved_bytes"],
            memory["safe_limit_bytes"],
        )
        self.assertLessEqual(
            memory["admission_upper_reserved_bytes"],
            memory["safe_limit_bytes"],
        )
        self.assertEqual(
            report["ranking_groups"][0]["admitted_by_historical_anchor"],
            1,
        )
        self.assertEqual(
            report["runtime_binding"]["runtime_mechanism_component_sha256"],
            self.predictor.memory_anchor_registry["anchors"][0]["runtime_match"][
                "runtime_mechanism_component_sha256"
            ],
        )

    def test_anchor_fails_closed_when_runtime_mechanism_changes(self) -> None:
        request = deepcopy(self.requests[0])
        request.update(
            {
                "request_id": "qwen3-14b-full-runtime-mismatch",
                "comparison_group": "qwen3-14b-full-runtime-mismatch",
                "dataset_id": "multiturn_4096",
                "dataset_category": "multiturn",
                "target_gbs": 64,
                "cutoff_len": 4096,
                "gpu_count": 2,
                "physical_mbs": 2,
                "zero_stage": 3,
                "gradient_checkpointing": True,
                "packing": False,
            }
        )
        record, _, support = self.predictor._record(request, input_index=0)
        record["runtime"]["runtime_mechanism_component_sha256"] = "0" * 64

        memory = self.predictor._memory_result(record, support)

        self.assertFalse(memory["anchor_override_applied"])
        self.assertFalse(memory["physical_model_admitted"])
        diagnostics = memory["historical_anchor"]["match_diagnostics"]
        self.assertIn(
            "runtime_mechanism_mismatch",
            diagnostics[0]["issues"],
        )

    def test_anchor_does_not_transfer_to_larger_cutoff(self) -> None:
        request = deepcopy(self.requests[0])
        request.update(
            {
                "request_id": "qwen3-14b-full-cutoff-too-large",
                "comparison_group": "qwen3-14b-full-cutoff-too-large",
                "dataset_id": "multiturn_4096",
                "dataset_category": "multiturn",
                "target_gbs": 64,
                "cutoff_len": 8192,
                "gpu_count": 2,
                "physical_mbs": 2,
                "zero_stage": 3,
                "gradient_checkpointing": True,
                "packing": False,
            }
        )

        report = self.predictor.predict([request])
        memory = report["predictions"][0]["memory"]

        self.assertFalse(memory["anchor_override_applied"])
        diagnostics = memory["historical_anchor"]["match_diagnostics"]
        self.assertIn("cutoff_len_exceeds_anchor", diagnostics[0]["issues"])

    def test_unsupported_packing_fails_closed(self) -> None:
        request = deepcopy(self.requests[2])
        request["request_id"] = "packing-unsupported"
        request["comparison_group"] = "packing-unsupported"
        request["packing"] = True

        report = self.predictor.predict([request])
        row = report["predictions"][0]
        codes = {reason["code"] for reason in row["support"]["reasons"]}
        self.assertEqual(row["support"]["label"], "unsupported")
        self.assertEqual(row["support"]["support_tier"], "generic_structural")
        self.assertEqual(
            row["support"]["required_experiment_family"],
            "dense_structural_transfer",
        )
        self.assertIn("packing_outside_supported_domain", codes)
        self.assertFalse(row["memory"]["admitted"])
        self.assertFalse(row["throughput"]["prediction_available"])
        self.assertEqual(
            row["memory"]["rejection_reason"],
            "outside_supported_domain",
        )

    def test_group_must_describe_one_fixed_user_scenario(self) -> None:
        requests = deepcopy(self.requests[:2])
        requests[1]["dataset_id"] = "multiturn_2048"
        requests[1]["dataset_category"] = "multiturn"
        requests[1]["cutoff_len"] = 2048
        with self.assertRaisesRegex(
            ValueError,
            "combines different user scenarios",
        ):
            self.predictor.predict(requests)

    def test_duplicate_candidate_configuration_is_rejected(self) -> None:
        requests = deepcopy(self.requests[:2])
        requests[1]["request_id"] = "same-config-new-id"
        for field in (
            "gpu_count",
            "physical_mbs",
            "zero_stage",
            "gradient_checkpointing",
            "packing",
        ):
            requests[1][field] = requests[0][field]
        with self.assertRaisesRegex(
            ValueError,
            "duplicate candidate configuration",
        ):
            self.predictor.predict(requests)

    def test_custom_dataset_profile_is_bound_and_cautioned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_path = Path(temporary) / "customer_chat.qwen3_nothink.jsonl"
            profile_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "sample_id": f"sample-{index}",
                            "total_tokens": length,
                            "label_tokens": int(length * 0.6),
                            "turns": 3,
                        }
                    )
                    + "\n"
                    for index, length in enumerate(
                        [64, 96, 128, 160, 192, 224, 256, 320]
                    )
                ),
                encoding="utf-8",
            )
            predictor = H800PhysicalV4BPredictor(
                additional_dataset_profile_dir=Path(temporary)
            )
            request = deepcopy(self.requests[2])
            request.update(
                {
                    "request_id": "custom-profile",
                    "comparison_group": "custom-profile",
                    "dataset_id": "customer_chat",
                    "dataset_category": "multiturn",
                    "profile_tokenizer_id": "Qwen3-14B@local",
                    "profile_template_id": "qwen3_nothink@v1",
                }
            )
            report = predictor.predict([request])

        row = report["predictions"][0]
        self.assertEqual(row["dataset_profile"]["origin"], "additional")
        self.assertEqual(row["support"]["label"], "caution")
        self.assertEqual(
            {reason["code"] for reason in row["support"]["reasons"]},
            {"new_dataset_profile"},
        )
        self.assertTrue(row["memory"]["admitted"])
        self.assertTrue(row["throughput"]["prediction_available"])

    def test_tampered_report_is_rejected(self) -> None:
        report = self.predictor.predict(self.requests)
        report["predictions"][0]["memory"]["admitted"] = False
        report["report_sha256"] = sha256_json(
            {key: value for key, value in report.items() if key != "report_sha256"}
        )
        with self.assertRaisesRegex(
            ValueError,
            "support/memory admission contract drifted",
        ):
            validate_prediction_report(report)


if __name__ == "__main__":
    unittest.main()
