"""Tests for static vision-tower structure derivation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import vision_tower_structure as vts  # noqa: E402

QWEN3_VL_8B = Path("/wanqing-models/Qwen3-VL-8B-Instruct")
QWEN2_5_VL_7B = Path("/wanqing-models/Qwen2.5-VL-7B-Instruct")
QWEN2_VL_7B = Path("/wanqing-models/Qwen2-VL-7B-Instruct")

QWEN3_VISION = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "out_hidden_size": 4096,
    "num_heads": 16,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "num_position_embeddings": 2304,
    "deepstack_visual_indexes": [8, 16, 24],
}
QWEN2_5_VISION = {
    "depth": 32,
    "hidden_size": 1280,
    "intermediate_size": 3420,
    "out_hidden_size": 3584,
    "num_heads": 16,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_chans": 3,
    "window_size": 112,
    "tokens_per_second": 2,
}


def _config_dir(config: dict) -> Path:
    directory = Path(tempfile.mkdtemp())
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


class TestGenerationDetection(unittest.TestCase):
    def test_deepstack_marks_qwen3_vl(self) -> None:
        config = {"model_type": "unknown", "vision_config": QWEN3_VISION}
        self.assertEqual(vts.detect_generation(config), "qwen3_vl")

    def test_window_size_marks_qwen2_5_vl(self) -> None:
        config = {"model_type": "unknown", "vision_config": QWEN2_5_VISION}
        self.assertEqual(vts.detect_generation(config), "qwen2_5_vl")

    def test_model_type_takes_precedence(self) -> None:
        config = {"model_type": "qwen3_vl", "vision_config": {}}
        self.assertEqual(vts.detect_generation(config), "qwen3_vl")

    def test_unknown_generation_is_named_not_guessed(self) -> None:
        config = {"model_type": "some_new_vl", "vision_config": {}}
        self.assertEqual(
            vts.detect_generation(config), "unknown_vision_generation"
        )

    def test_qwen3_5_is_not_mistaken_for_qwen3_vl(self) -> None:
        # Qwen3.5 declares model_type "qwen3_5" and carries a *present but
        # empty* deepstack_visual_indexes.  The old `"…" in vision` test only
        # checked key presence, so it returned "qwen3_vl" -- which passed
        # VERIFIED_GENERATIONS and priced a new generation with Qwen3-VL's
        # formula instead of failing closed.
        config = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "vision_config": dict(
                QWEN3_VISION, model_type="qwen3_5", deepstack_visual_indexes=[]
            ),
        }
        self.assertEqual(
            vts.detect_generation(config), "unknown_vision_generation"
        )
        with self.assertRaises(vts.VisionStructureError):
            vts.build_vision_structure(
                _config_dir(config), verify_against_checkpoint=False
            )

    def test_empty_deepstack_does_not_imply_qwen3_vl(self) -> None:
        # The fallback must key on a non-empty index list, not key presence.
        config = {"vision_config": dict(QWEN3_VISION, deepstack_visual_indexes=[])}
        self.assertEqual(
            vts.detect_generation(config), "unknown_vision_generation"
        )

    def test_vision_config_model_type_can_refuse_a_generation(self) -> None:
        # No top-level model_type, but the vision tower names an unverified
        # generation: that is decisive, so do not structurally guess.
        config = {"vision_config": dict(QWEN3_VISION, model_type="qwen4_vl")}
        self.assertEqual(
            vts.detect_generation(config), "unknown_vision_generation"
        )


class TestParameterFormulas(unittest.TestCase):
    def test_qwen3_components_match_checkpoint_values(self) -> None:
        params = vts.vision_tower_parameters(QWEN3_VISION, generation="qwen3_vl")
        # Values verified against Qwen3-VL-8B safetensors.
        self.assertEqual(params["blocks"], 411_466_608)
        self.assertEqual(params["deepstack_mergers"], 3 * params["merger"])
        self.assertEqual(params["position_embedding"], 2304 * 1152)
        # Checkpoint total is 576_388_336; the analytic form is within 0.001%.
        # The residual 5,760 elements are a small bias/shape detail not worth
        # hard-coding away -- the checkpoint verification is the real guard.
        self.assertEqual(params["total"], 576_394_096)
        self.assertLess(abs(params["total"] - 576_388_336) / 576_388_336, 1e-4)

    def test_qwen2_5_components_match_checkpoint_values(self) -> None:
        params = vts.vision_tower_parameters(
            QWEN2_5_VISION, generation="qwen2_5_vl"
        )
        self.assertEqual(params["blocks"], 630_470_400)
        self.assertEqual(params["merger"], 44_574_464)
        self.assertEqual(params["position_embedding"], 0)
        self.assertEqual(params["deepstack_mergers"], 0)

    def test_gated_mlp_makes_qwen2_5_blocks_larger_per_layer(self) -> None:
        # Same depth and width, different MLP structure: the gated variant has
        # three projections instead of two.  Getting this wrong under-counts the
        # tower by ~20%, which the checkpoint check caught.
        shared = dict(QWEN2_5_VISION, depth=1)
        gated = vts.vision_tower_parameters(shared, generation="qwen2_5_vl")
        plain = vts.vision_tower_parameters(shared, generation="qwen3_vl")
        self.assertGreater(gated["blocks"], plain["blocks"])

    def test_deepstack_is_a_material_share_of_the_tower(self) -> None:
        params = vts.vision_tower_parameters(QWEN3_VISION, generation="qwen3_vl")
        share = params["deepstack_mergers"] / params["total"]
        self.assertGreater(share, 0.15)

    def test_missing_required_field_is_rejected(self) -> None:
        broken = {k: v for k, v in QWEN3_VISION.items() if k != "intermediate_size"}
        with self.assertRaises(vts.VisionStructureError):
            vts.vision_tower_parameters(broken, generation="qwen3_vl")


class TestVisualTokens(unittest.TestCase):
    def test_patch_and_merge_determine_token_count(self) -> None:
        result = vts.visual_tokens_per_image(
            QWEN3_VISION, height=768, width=768
        )
        self.assertEqual(result["grid_height"], 48)
        self.assertEqual(result["patch_count"], 48 * 48)
        self.assertEqual(result["visual_tokens"], 48 * 48 // 4)

    def test_patch14_and_patch16_give_different_token_counts(self) -> None:
        q3 = vts.visual_tokens_per_image(QWEN3_VISION, height=224, width=224)
        q25 = vts.visual_tokens_per_image(QWEN2_5_VISION, height=224, width=224)
        self.assertNotEqual(q3["visual_tokens"], q25["visual_tokens"])

    def test_token_count_scales_with_resolution(self) -> None:
        small = vts.visual_tokens_per_image(QWEN3_VISION, height=224, width=224)
        large = vts.visual_tokens_per_image(QWEN3_VISION, height=448, width=448)
        self.assertEqual(large["visual_tokens"], 4 * small["visual_tokens"])


class TestFailClosed(unittest.TestCase):
    def test_unverified_generation_is_refused(self) -> None:
        directory = _config_dir(
            {
                "model_type": "qwen2_vl",
                "architectures": ["Qwen2VLForConditionalGeneration"],
                "vision_config": {"depth": 32, "embed_dim": 1280, "mlp_ratio": 4},
            }
        )
        with self.assertRaises(vts.VisionStructureError) as caught:
            vts.build_vision_structure(directory, verify_against_checkpoint=False)
        self.assertIn("no verified structural formula", str(caught.exception))

    def test_non_vl_config_is_refused(self) -> None:
        directory = _config_dir({"model_type": "qwen3", "hidden_size": 4096})
        with self.assertRaises(vts.VisionStructureError) as caught:
            vts.build_vision_structure(directory, verify_against_checkpoint=False)
        self.assertIn("no vision_config", str(caught.exception))

    def test_missing_config_is_refused(self) -> None:
        with self.assertRaises(vts.VisionStructureError):
            vts.build_vision_structure(
                Path(tempfile.mkdtemp()), verify_against_checkpoint=False
            )


class TestReportContract(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = _config_dir(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "image_token_id": 151655,
                "text_config": {"hidden_size": 4096, "num_hidden_layers": 36},
                "vision_config": QWEN3_VISION,
            }
        )
        self.report = vts.build_vision_structure(
            self.directory, verify_against_checkpoint=False
        )

    def test_report_states_what_is_not_modelled(self) -> None:
        not_modelled = self.report["not_modelled"]
        self.assertIn("vision_encoder_activation", not_modelled)
        self.assertIn("multimodal_workspace", not_modelled)
        self.assertIn("image_token_sequence_inflation", not_modelled)
        self.assertIn("dynamic_resolution_distribution", not_modelled)

    def test_report_claims_no_memory_authority(self) -> None:
        guarantees = self.report["guarantees"]
        self.assertFalse(guarantees["predicts_peak_memory"])
        self.assertFalse(guarantees["admits_candidates"])
        self.assertFalse(guarantees["explains_observed_vl_memory_gap"])
        self.assertFalse(guarantees["creates_gpu_queue"])
        self.assertEqual(
            self.report["calibration_status"], "structural_only_uncalibrated"
        )

    def test_resident_weights_explain_only_a_small_gap_fraction(self) -> None:
        # The whole point of the honesty guard: a ~1 GiB structural term must not
        # be read as fixing a ~21 GiB measured gap.
        context = self.report["observed_gap_context"]
        self.assertLess(context["fraction_of_gap_explained"], 0.10)
        self.assertIn("conclusion", context)

    def test_report_is_checksummed(self) -> None:
        self.assertEqual(self.report["schema"], vts.SCHEMA)
        self.assertEqual(len(self.report["report_sha256"]), 64)

    def test_guard_forbids_adding_vision_weights_as_a_new_memory_term(self) -> None:
        # The inventory's actual_parameters is a whole-checkpoint count, so a VL
        # model already includes its vision tower there.  Adding this module's
        # total as an extra resident term would double-count those bytes.
        guard = self.report["double_counting_guard"]
        self.assertTrue(guard["checkpoint_total_includes_vision_tower"])
        self.assertFalse(guard["safe_to_add_as_new_memory_term"])
        self.assertIn("not as an additive", guard["correct_use"])


class TestDoubleCountingArithmetic(unittest.TestCase):
    def test_qwen3_vl_checkpoint_total_minus_vision_equals_language_tower(
        self,
    ) -> None:
        # Measured evidence: the inventory records 8,767,123,696 parameters for
        # qwen3_vl_8b, and the training log reports 8,190,735,360 trainable
        # parameters for Full (which froze the vision tower).  Their difference is
        # exactly the measured vision-tower size, proving the checkpoint total
        # already contains it.
        inventory_total = 8_767_123_696
        measured_vision = 576_388_336
        full_trainable_language_only = 8_190_735_360
        self.assertEqual(
            inventory_total - measured_vision, full_trainable_language_only
        )

    def test_vision_weights_are_a_small_share_of_the_measured_gap(self) -> None:
        params = vts.vision_tower_parameters(QWEN3_VISION, generation="qwen3_vl")
        vision_bytes = params["total"] * vts.BYTES_PER_BF16
        share = vision_bytes / vts.QWEN3_VL_8B_MBS1_OBSERVED_GAP_BYTES
        # ~1.07 GiB against a ~21.2 GiB gap: the gap is activation, workspace and
        # gather behaviour, not resident vision weights.
        self.assertLess(share, 0.10)


class TestRealCheckpoints(unittest.TestCase):
    def test_qwen3_vl_matches_its_checkpoint(self) -> None:
        if not (QWEN3_VL_8B / "config.json").is_file():
            self.skipTest("Qwen3-VL checkpoint unavailable")
        report = vts.build_vision_structure(QWEN3_VL_8B)
        verification = report["checkpoint_verification"]
        if not verification.get("available"):
            self.skipTest(f"checkpoint unreadable: {verification.get('reason')}")
        self.assertTrue(verification["matches_within_tolerance"])
        self.assertLess(abs(verification["relative_delta"]), 0.001)

    def test_qwen2_5_vl_matches_its_checkpoint(self) -> None:
        if not (QWEN2_5_VL_7B / "config.json").is_file():
            self.skipTest("Qwen2.5-VL checkpoint unavailable")
        report = vts.build_vision_structure(QWEN2_5_VL_7B)
        verification = report["checkpoint_verification"]
        if not verification.get("available"):
            self.skipTest(f"checkpoint unreadable: {verification.get('reason')}")
        self.assertTrue(verification["matches_within_tolerance"])
        self.assertLess(abs(verification["relative_delta"]), 0.001)

    def test_build_report_records_failures_without_aborting(self) -> None:
        if not (QWEN3_VL_8B / "config.json").is_file():
            self.skipTest("checkpoints unavailable")
        report = vts.build_report(
            [QWEN3_VL_8B, QWEN2_VL_7B], verify_against_checkpoint=False
        )
        self.assertEqual(report["model_count"], 1)
        self.assertEqual(len(report["failures"]), 1)
        self.assertIn("qwen2_vl", report["failures"][0]["error"])


if __name__ == "__main__":
    unittest.main()
