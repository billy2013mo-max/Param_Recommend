from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fit_h800_m1_full_admission_v5 import (  # noqa: E402
    _is_architecture_outside_v5_scope,
    _predict_v5,
    _v5_detail,
)
from fit_h800_lora_source_disjoint_recalibration_v1 import (  # noqa: E402
    _is_critical_lora,
)
from fit_h800_m1_separate_admission_v4 import (  # noqa: E402
    _is_bounded_full,
    _is_lora_zero2_no_gc_four_gpu,
    _scope_predicate,
)


def full_record(*, zero: int = 3, gc: bool = True, gpu_count: int = 2) -> dict:
    return {
        "selector": {
            "training_mode": "full",
            "zero_stage": zero,
            "gradient_checkpointing": gc,
            "packing": False,
        },
        "scenario": {"gpu_count": gpu_count},
        "memory": {"safe_limit_bytes": 100.0},
    }


def lora_record(
    *, zero: int = 2, gc: bool = False, gpu_count: int = 4, packing: bool = False
) -> dict:
    return {
        "selector": {
            "training_mode": "lora",
            "zero_stage": zero,
            "gradient_checkpointing": gc,
            "packing": packing,
        },
        "scenario": {"gpu_count": gpu_count},
        "memory": {"safe_limit_bytes": 100.0},
    }


class FullAdmissionV5Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.direct = {
            "available": True,
            "allocated_center_bytes": 40.0,
            "reserved_center_bytes": 50.0,
            "upper_bytes": 80.0,
            "upper_components": {
                "direct_reserved_upper": 80.0,
                "oom_upper": 60.0,
            },
        }
        self.bundle = {
            "full_admission_head": {
                "threshold": 0.5,
                "model": {
                    "feature_means": [0.0, 0.0],
                    "feature_scales": [1.0, 1.0],
                    "intercept": -2.0,
                    "coefficients": [0.0, 0.0],
                },
            },
            "lora_zero2_no_gc_four_gpu_admission_head": {
                "threshold": 0.5,
                "model": {
                    "feature_means": [0.0, 0.0],
                    "feature_scales": [1.0, 1.0],
                    "intercept": -2.0,
                    "coefficients": [0.0, 0.0],
                },
            },
        }

    def test_scope_is_full_zero3_gc_two_gpu(self) -> None:
        self.assertTrue(_is_bounded_full(full_record()))
        self.assertFalse(_is_bounded_full(full_record(zero=2)))
        self.assertFalse(_is_bounded_full(full_record(gc=False)))
        self.assertFalse(_is_bounded_full(full_record(gpu_count=4)))

    def test_supported_full_uses_separate_head_without_stacking(self) -> None:
        with patch(
            "fit_h800_m1_full_admission_v5._predict_safety",
            return_value=self.direct,
        ):
            prediction = _predict_v5(full_record(), self.bundle)
        self.assertTrue(prediction["admitted_by_separate_head"])
        self.assertEqual(prediction["admission_scope"], "full_zero3_gc_two_gpu")
        self.assertNotIn(
            "allocated_expansion_upper", prediction["upper_components"]
        )

    def test_unsupported_full_fails_closed(self) -> None:
        with patch(
            "fit_h800_m1_full_admission_v5._predict_safety",
            return_value=self.direct,
        ):
            prediction = _predict_v5(full_record(zero=2), self.bundle)
        self.assertFalse(prediction["admitted_by_separate_head"])
        self.assertIn("fail_closed", prediction["admission_scope"])
        self.assertNotIn(
            "allocated_expansion_upper", prediction["upper_components"]
        )

    def test_supported_full_rejects_center_above_safe_limit(self) -> None:
        direct = {
            **self.direct,
            "reserved_center_bytes": 101.0,
            "upper_bytes": 120.0,
            "upper_components": {"direct_reserved_upper": 120.0},
        }
        with patch(
            "fit_h800_m1_full_admission_v5._predict_safety",
            return_value=direct,
        ):
            prediction = _predict_v5(full_record(), self.bundle)
        self.assertFalse(prediction["admitted_by_separate_head"])
        self.assertTrue(prediction["hard_center_guard"])
        self.assertFalse(prediction["center_within_safe_limit"])


class LoraFourGpuHeadTests(FullAdmissionV5Tests):
    """The second LoRA mechanism gets its own head instead of the S0 fallback."""

    def test_scope_is_lora_zero2_no_gc_four_gpu(self) -> None:
        self.assertTrue(_is_lora_zero2_no_gc_four_gpu(lora_record()))
        self.assertFalse(_is_lora_zero2_no_gc_four_gpu(lora_record(zero=3)))
        self.assertFalse(_is_lora_zero2_no_gc_four_gpu(lora_record(gc=True)))
        self.assertFalse(_is_lora_zero2_no_gc_four_gpu(lora_record(gpu_count=2)))
        self.assertFalse(
            _is_lora_zero2_no_gc_four_gpu(lora_record(packing=True))
        )

    def test_scope_predicate_registers_new_scope(self) -> None:
        self.assertIs(
            _scope_predicate("lora_zero2_no_gc_four_gpu"),
            _is_lora_zero2_no_gc_four_gpu,
        )

    def test_lora_four_gpu_uses_own_head_without_stacking(self) -> None:
        with patch(
            "fit_h800_m1_full_admission_v5._predict_safety",
            return_value=self.direct,
        ):
            prediction = _predict_v5(lora_record(), self.bundle)
        self.assertEqual(
            prediction["admission_scope"], "lora_zero2_no_gc_four_gpu"
        )
        self.assertTrue(prediction["admitted_by_separate_head"])
        self.assertNotIn(
            "allocated_expansion_upper", prediction["upper_components"]
        )

    def test_lora_four_gpu_rejects_center_above_safe_limit(self) -> None:
        direct = {
            **self.direct,
            "reserved_center_bytes": 101.0,
            "upper_bytes": 120.0,
            "upper_components": {"direct_reserved_upper": 120.0},
        }
        with patch(
            "fit_h800_m1_full_admission_v5._predict_safety",
            return_value=direct,
        ):
            prediction = _predict_v5(lora_record(), self.bundle)
        self.assertFalse(prediction["admitted_by_separate_head"])
        self.assertTrue(prediction["hard_center_guard"])


class StackedFallbackFailClosedTests(unittest.TestCase):
    """Mechanisms still on S0 must not admit once the stacked product revives."""

    def setUp(self) -> None:
        self.stacked = {
            "available": True,
            "allocated_center_bytes": 40.0,
            "reserved_center_bytes": 50.0,
            "upper_bytes": 80.0,
            "upper_components": {"direct_reserved_upper": 80.0},
        }
        # LoRA ZeRO-3 / GC-off / two GPUs has no head of its own in V5.1.
        self.record = lora_record(zero=3, gpu_count=2)

    def _bundle(self, *, expansion_available: bool) -> dict:
        entry = {"available": expansion_available}
        if expansion_available:
            entry["expansion_upper"] = 2.3008
        return {
            "reservation_expansion": {
                "entries": {'["lora",3,false,2,false]': entry}
            }
        }

    def test_inert_expansion_keeps_current_s0_behaviour(self) -> None:
        with patch(
            "fit_h800_m1_full_admission_v5._predict_safety",
            return_value=self.stacked,
        ):
            prediction = _predict_v5(
                self.record, self._bundle(expansion_available=False)
            )
        self.assertEqual(
            prediction["admission_scope"],
            "outside_lora_and_full_keep_current_s0",
        )
        self.assertFalse(prediction["stacked_expansion_available"])
        self.assertIsNone(prediction["admitted_by_separate_head"])
        self.assertNotIn(
            "stacked_expansion_upper_active_fail_closed",
            prediction["admission_issues"],
        )

    def test_live_expansion_fails_closed(self) -> None:
        with patch(
            "fit_h800_m1_full_admission_v5._predict_safety",
            return_value=self.stacked,
        ):
            prediction = _predict_v5(
                self.record, self._bundle(expansion_available=True)
            )
        self.assertTrue(prediction["stacked_expansion_available"])
        self.assertFalse(prediction["admitted_by_separate_head"])
        self.assertIn(
            "stacked_expansion_upper_active_fail_closed",
            prediction["admission_issues"],
        )

    def test_detail_refuses_admission_when_stacked_product_is_live(self) -> None:
        record = {
            **self.record,
            "calibration_partition": {"split_unit_id": "unit-test-source"},
            "memory": {
                "safe_limit_bytes": 100.0,
                "observed_allocated_bytes": 30.0,
                "observed_reserved_bytes": 40.0,
            },
            "outcome": "success",
        }
        with patch(
            "fit_h800_m1_full_admission_v5._predict_safety",
            return_value=self.stacked,
        ):
            live = _v5_detail(record, self._bundle(expansion_available=True))
            inert = _v5_detail(record, self._bundle(expansion_available=False))
        self.assertFalse(live["admitted"])
        self.assertTrue(live["memory_upper_would_admit"])
        # The inert path is unchanged, so the guard costs nothing today.
        self.assertTrue(inert["admitted"])


class ArchitectureFailClosedTests(unittest.TestCase):
    """V5 routes on mechanism alone, so architecture must gate it separately.

    Every V5 head is fitted on dense, text-only, uniform-softmax-attention Qwen3
    checkpoints.  A vision-language or hybrid-attention request satisfies those
    mechanism predicates just as well as a text job does, so without this gate
    it is admitted on a head that never saw its architecture.
    """

    def setUp(self) -> None:
        self.direct = {
            "available": True,
            "allocated_center_bytes": 40.0,
            "reserved_center_bytes": 50.0,
            "upper_bytes": 80.0,
            "upper_components": {"direct_reserved_upper": 80.0},
        }
        self.bundle = {
            "separate_admission_head": {
                "threshold": 0.5,
                "model": {
                    "feature_means": [0.0, 0.0],
                    "feature_scales": [1.0, 1.0],
                    # Strongly negative intercept: this head admits everything
                    # it is asked about, so anything refused below is refused by
                    # the architecture gate and not by the risk model.
                    "intercept": -8.0,
                    "coefficients": [0.0, 0.0],
                },
            }
        }

    def _record(self, model_id: str, model_path: str) -> dict:
        # Mechanism chosen to match _is_critical_lora exactly: LoRA, ZeRO-2,
        # GC off, two GPUs, no packing.
        return {
            "selector": {
                "training_mode": "lora",
                "zero_stage": 2,
                "gradient_checkpointing": False,
                "packing": False,
            },
            "scenario": {"gpu_count": 2},
            "memory": {
                "safe_limit_bytes": 100.0,
                "observed_allocated_bytes": 30.0,
                "observed_reserved_bytes": 40.0,
            },
            "outcome": "success",
            "calibration_partition": {"split_unit_id": "unit-test-source"},
            "configuration": {
                "job": {
                    "model_id": model_id,
                    "model_path": model_path,
                    "model_family": model_id.rsplit("_", 1)[0],
                }
            },
        }

    def test_text_baseline_on_this_mechanism_is_admitted(self) -> None:
        # The control: same mechanism, text-only checkpoint, must still admit.
        # Without this the tests below would pass even if the gate refused
        # everything.
        record = self._record("qwen3_14b", "/wanqing-models/Qwen3-14B")
        self.assertTrue(_is_critical_lora(record))
        self.assertFalse(_is_architecture_outside_v5_scope(record))
        with patch(
            "fit_h800_m1_full_admission_v5._predict_safety",
            return_value=self.direct,
        ):
            prediction = _predict_v5(record, self.bundle)
            detail = _v5_detail(record, self.bundle)
        self.assertEqual(prediction["admission_scope"], "critical_lora")
        self.assertTrue(prediction["admitted_by_separate_head"])
        self.assertTrue(detail["admitted"])

    def test_vision_language_is_refused_on_the_critical_lora_mechanism(self) -> None:
        # This is the case the gate exists for: a VL job whose mechanism is
        # exactly the one Critical-LoRA covers.  A mechanism-first router hands
        # it to a text-only head and admits it.
        for model_id, path in (
            ("qwen3_vl_8b", "/wanqing-models/Qwen3-VL-8B-Instruct"),
            ("qwen2p5_vl_7b", "/wanqing-models/Qwen2.5-VL-7B-Instruct"),
        ):
            record = self._record(model_id, path)
            with self.subTest(model_id=model_id):
                self.assertTrue(_is_critical_lora(record))
                self.assertTrue(_is_architecture_outside_v5_scope(record))
                with patch(
                    "fit_h800_m1_full_admission_v5._predict_safety",
                    return_value=self.direct,
                ):
                    prediction = _predict_v5(record, self.bundle)
                    detail = _v5_detail(record, self.bundle)
                self.assertEqual(
                    prediction["admission_scope"],
                    "architecture_outside_v5_scope_fail_closed",
                )
                self.assertFalse(prediction["admitted_by_separate_head"])
                self.assertFalse(detail["admitted"])

    def test_hybrid_attention_is_refused(self) -> None:
        record = self._record("qwen3p5_4b", "/wanqing-models/Qwen3.5-4B")
        self.assertTrue(_is_architecture_outside_v5_scope(record))
        with patch(
            "fit_h800_m1_full_admission_v5._predict_safety",
            return_value=self.direct,
        ):
            detail = _v5_detail(record, self.bundle)
        self.assertFalse(detail["admitted"])
        self.assertEqual(
            detail["admission_scope"], "architecture_outside_v5_scope_fail_closed"
        )

    def test_refusal_survives_the_detail_layer(self) -> None:
        # The load-bearing assertion.  _v5_detail recomputes ``admitted`` from
        # ``head_backed``, which is false for an out-of-domain architecture -- so
        # if the gate lived only in _predict_v5, the memory upper's verdict would
        # be restored here and the row admitted anyway.  ``upper_would_admit``
        # being true is what proves the refusal is doing the work.
        record = self._record("qwen3_vl_8b", "/wanqing-models/Qwen3-VL-8B-Instruct")
        with patch(
            "fit_h800_m1_full_admission_v5._predict_safety",
            return_value=self.direct,
        ):
            detail = _v5_detail(record, self.bundle)
        self.assertTrue(detail["memory_upper_would_admit"])
        self.assertFalse(detail["admitted"])

    def test_mechanism_only_record_is_not_refused_for_architecture(self) -> None:
        # Absence of an architecture claim is not a claim.  Mechanism-only
        # records (as the rest of this suite uses) must route by mechanism, or
        # the whole calibrated fleet would fall out of domain.
        self.assertFalse(_is_architecture_outside_v5_scope(lora_record()))
        self.assertFalse(_is_architecture_outside_v5_scope(full_record()))


if __name__ == "__main__":
    unittest.main()
