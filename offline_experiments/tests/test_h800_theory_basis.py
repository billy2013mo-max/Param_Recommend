from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import sha256_json  # noqa: E402
from h800_theory_basis import (  # noqa: E402
    _memory_observation,
    _model_geometry,
    memory_basis,
    validate_report,
)


def model_fixture() -> dict:
    return {
        "hidden_size": 512,
        "intermediate_size": 1_024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": 32_000,
        "actual_parameters": 100_000_000,
    }


def job_fixture(**overrides: object) -> dict:
    job = {
        "model_id": "qwen-fixture",
        "model_parameters": 100_000_000,
        "train_type": "lora",
        "gpu_count": 4,
        "mbs": 2,
        "cutoff_len": 2_048,
        "zero": "zero2",
        "gc": True,
    }
    job.update(overrides)
    return job


def memory_row(*, outcome: str) -> dict:
    row = {
        "outcome": {"class": outcome},
        "measurements": {
            "memory": {
                "max_allocated_bytes": 80,
                "max_reserved_bytes": 90,
                "values_are_observed_not_imputed": True,
            }
        },
    }
    if outcome == "oom":
        row["censoring"] = {
            "kind": "right_censored_memory_demand",
            "demand_peak_bytes": None,
            "requested_allocation_bytes": 30,
            "device_capacity_bytes_reported_in_error": 120,
            "free_bytes_reported_in_error": 10,
        }
    return row


class H800TheoryBasisTests(unittest.TestCase):
    def test_lora_parameter_count_is_structural_not_fitted(self) -> None:
        geometry = _model_geometry(
            job_fixture(), model_fixture(), {"rank": 32, "target": "all"}
        )
        expected = 32 * 4 * (9 * 512 + 2 * 128 + 3 * 1_024)
        self.assertEqual(geometry["adapter_parameters"], expected)
        self.assertEqual(geometry["trainable_parameters"], expected)
        self.assertEqual(geometry["loaded_parameters"], 100_000_000 + expected)

    def test_full_training_uses_base_parameters_as_trainable(self) -> None:
        geometry = _model_geometry(
            job_fixture(train_type="full"),
            model_fixture(),
            {"rank": 32, "target": "all"},
        )
        self.assertEqual(geometry["adapter_parameters"], 0)
        self.assertEqual(geometry["trainable_parameters"], 100_000_000)
        self.assertEqual(geometry["frozen_parameters"], 0)

    def test_geometry_reads_from_text_config_when_nested(self) -> None:
        # VL / nested architectures place the language tower under text_config.
        flat = model_fixture()
        nested = {
            "actual_parameters": flat["actual_parameters"],
            "text_config": {k: v for k, v in flat.items() if k != "actual_parameters"},
        }
        base = _model_geometry(
            job_fixture(), flat, {"rank": 32, "target": "all"}
        )
        via_text = _model_geometry(
            job_fixture(), nested, {"rank": 32, "target": "all"}
        )
        self.assertEqual(via_text, base)

    def test_geometry_honors_explicit_head_dim(self) -> None:
        # When hidden is not divisible by heads, an explicit head_dim must be
        # used rather than hidden // heads (which would raise or mis-size KV).
        model = model_fixture()
        model["num_attention_heads"] = 24  # 512 % 24 != 0
        model["head_dim"] = 256
        geometry = _model_geometry(
            job_fixture(), model, {"rank": 32, "target": "all"}
        )
        self.assertEqual(geometry["head_dim"], 256)
        self.assertEqual(geometry["kv_width"], model["num_key_value_heads"] * 256)

    def test_zero2_shards_gradient_and_optimizer_but_not_parameters(self) -> None:
        geometry = _model_geometry(
            job_fixture(), model_fixture(), {"rank": 32, "target": "all"}
        )
        basis = memory_basis(job_fixture(), geometry, 150_000_000_000)
        components = basis["components"]
        self.assertEqual(
            components["parameters_bytes"], 2 * geometry["loaded_parameters"]
        )
        self.assertEqual(
            components["gradients_bytes"], 2 * geometry["trainable_parameters"] / 4
        )
        self.assertEqual(
            components["optimizer_bytes"],
            12 * geometry["trainable_parameters"] / 4,
        )
        self.assertLessEqual(
            basis["analytic_lower_bound_bytes"],
            basis["analytic_reference_bytes"],
        )

    def test_oom_is_right_censored_and_never_an_exact_target(self) -> None:
        observed = _memory_observation(memory_row(outcome="oom"))
        self.assertIsNone(observed["peak_reserved_target_bytes"])
        self.assertEqual(observed["right_censor_lower_bytes"], 140)
        self.assertEqual(
            observed["right_censor_components"][
                "capacity_minus_free_plus_requested_bytes"
            ],
            140,
        )

    def test_oom_rejects_imputed_demand(self) -> None:
        row = memory_row(outcome="oom")
        row["censoring"]["demand_peak_bytes"] = 141
        with self.assertRaisesRegex(ValueError, "never imputed"):
            _memory_observation(row)

    def test_real_artifact_is_nonpublishable_and_mechanism_only(self) -> None:
        artifact = ROOT / "artifacts" / "h800_theory_basis.json"
        report = json.loads(artifact.read_text(encoding="utf-8"))
        validate_report(report)
        self.assertEqual(report["counts"]["records"], 1_111)
        self.assertEqual(report["counts"]["outcomes"], {"oom": 106, "success": 1_005})
        self.assertFalse(report["publishable"])
        for record in report["records"]:
            self.assertNotIn("model_id", record["selector"])
            self.assertNotIn("dataset_id", record["selector"])

    def test_validation_rejects_exact_oom_target(self) -> None:
        artifact = ROOT / "artifacts" / "h800_theory_basis.json"
        report = json.loads(artifact.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(report)
        oom = next(record for record in tampered["records"] if record["outcome"] == "oom")
        oom["memory"]["observed"]["peak_reserved_target_bytes"] = 1.0
        content = dict(tampered)
        content.pop("report_sha256")
        tampered["report_sha256"] = sha256_json(content)
        with self.assertRaisesRegex(ValueError, "cannot be used as an exact target"):
            validate_report(tampered)


if __name__ == "__main__":
    unittest.main()
