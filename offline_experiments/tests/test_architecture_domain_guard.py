"""Tests for the shared fail-closed architecture predicates.

The point of these predicates is that a refusal is *symmetric*: whatever the
memory side refuses, the throughput side refuses too.  So the tests here pin
two things -- that the calibrated text fleet stays in domain (a false positive
would refuse work we can actually price), and that every architecture we cannot
price is refused even when its config is malformed.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import architecture_domain_guard as guard  # noqa: E402

MODELS_JSON = Path(__file__).resolve().parents[1] / "config" / "models.json"

# The four models the H800 memory and throughput models are actually fitted on.
# If any of these is ever refused, the guard is broken, not the fleet.
CALIBRATED_TEXT_MODEL_IDS = ("qwen3_1p7b", "qwen3_4b", "qwen3_8b", "qwen3_14b")

QWEN3_5_4B = Path("/wanqing-models/Qwen3.5-4B")
QWEN3_VL_8B = Path("/wanqing-models/Qwen3-VL-8B-Instruct")
QWEN2_5_VL_7B = Path("/wanqing-models/Qwen2.5-VL-7B-Instruct")


def _load(path: Path) -> dict:
    return json.loads((path / "config.json").read_text(encoding="utf-8"))


def _registry() -> dict[str, dict]:
    raw = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    entries = raw["models"] if isinstance(raw, dict) and "models" in raw else raw
    return {entry["id"]: entry for entry in entries}


class TestCalibratedFleetStaysInDomain(unittest.TestCase):
    def test_the_four_fitted_models_are_never_refused(self) -> None:
        registry = _registry()
        for model_id in CALIBRATED_TEXT_MODEL_IDS:
            entry = registry[model_id]
            config = _load(Path(entry["path"]))
            config["model_id"] = model_id
            config["family"] = entry.get("family")
            with self.subTest(model_id=model_id):
                self.assertEqual(guard.architecture_domain_refusals(config), [])
                self.assertFalse(guard.is_vision_language(config))
                self.assertFalse(guard.is_hybrid_attention(config))

    def test_a_dense_text_config_without_layer_types_is_uniform(self) -> None:
        # Absent layer_types means a uniform softmax stack; it must not be read
        # as hybrid, or the whole calibrated fleet would be refused.
        config = {"hidden_size": 4096, "num_hidden_layers": 32}
        self.assertFalse(guard.is_hybrid_attention(config))


class TestVisionLanguageRefusal(unittest.TestCase):
    def test_vision_config_is_decisive(self) -> None:
        self.assertTrue(guard.is_vision_language({"vision_config": {"depth": 27}}))

    def test_real_vl_checkpoints_are_refused(self) -> None:
        for path in (QWEN3_VL_8B, QWEN2_5_VL_7B):
            with self.subTest(path=path.name):
                self.assertTrue(guard.is_vision_language(_load(path)))

    def test_media_token_markers_alone_are_enough(self) -> None:
        # A config that hides its tower but still reserves image token ids is
        # multimodal; refuse it rather than pricing it as text.
        self.assertTrue(guard.is_vision_language({"image_token_id": 151655}))

    def test_model_id_alone_is_enough(self) -> None:
        # Registry rows sometimes carry the architecture only in the id.
        self.assertTrue(guard.is_vision_language({"model_id": "qwen3_vl_8b"}))

    def test_text_only_model_is_not_flagged(self) -> None:
        self.assertFalse(
            guard.is_vision_language(
                {"model_id": "qwen3_14b", "hidden_size": 5120, "architectures": ["Qwen3ForCausalLM"]}
            )
        )


class TestHybridAttentionRefusal(unittest.TestCase):
    def test_linear_attention_layers_are_refused(self) -> None:
        config = {
            "hidden_size": 2560,
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        }
        self.assertTrue(guard.is_hybrid_attention(config))

    def test_real_qwen3_5_is_refused_on_both_axes(self) -> None:
        # Qwen3.5 is genuinely multimodal *and* hybrid, so it must trip both
        # predicates -- not merely be caught incidentally by the VL one.  Before
        # this guard, hybrid attention had no fail-closed protection at all, so a
        # text-only hybrid stack would have been priced silently.
        config = _load(QWEN3_5_4B)
        self.assertTrue(guard.is_vision_language(config))
        self.assertTrue(guard.is_hybrid_attention(config))
        reasons = [code for code, _ in guard.architecture_domain_refusals(config)]
        self.assertEqual(
            reasons, [guard.VISION_LANGUAGE_REASON, guard.HYBRID_ATTENTION_REASON]
        )

    def test_hybrid_is_detected_without_any_vision_tower(self) -> None:
        # The load-bearing case: strip the tower from a hybrid config and the
        # hybrid predicate must still fire on its own.
        config = _load(QWEN3_5_4B)
        for key in (
            "vision_config",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ):
            config.pop(key, None)
        self.assertFalse(guard.is_vision_language(config))
        self.assertTrue(guard.is_hybrid_attention(config))

    def test_layer_types_nested_under_text_config_is_read(self) -> None:
        config = _load(QWEN3_5_4B)
        self.assertIsNotNone((config.get("text_config") or {}).get("layer_types"))
        self.assertTrue(guard.is_hybrid_attention(config))


class TestNativeRecordShapeIsCovered(unittest.TestCase):
    """The record shape V5's fit loop actually predicts on must be readable.

    ``build_native_record`` drops the job payload: a native record has no
    ``configuration`` key at all, only ``scenario.model_id`` plus a
    ``model_basis`` of plain geometry integers.  An earlier version of the view
    read only ``configuration.job``, so it returned ``None`` for every native
    record -- the gate was live in the unit tests (which use observation-shaped
    fixtures) and inert on the real fit path.  These tests exist so that cannot
    silently return.
    """

    def _native_record(self, model_id: str) -> dict:
        # Faithful to build_native_record's output: no "configuration" key.
        return {
            "schema": "sft_h800_native_memory_record/v2",
            "scenario": {"model_id": model_id, "gpu_count": 2},
            "selector": {"training_mode": "lora", "zero_stage": 2},
            "model_basis": {"hidden_size": 4096, "num_layers": 36},
        }

    def test_native_record_resolves_to_a_checkpoint_config(self) -> None:
        view = guard.architecture_view_of_observation(
            self._native_record("qwen3_8b")
        )
        self.assertIsNotNone(view)
        # Proof the registry lookup reached the real config.json rather than
        # merely echoing the id back.
        self.assertIsNotNone(view.get("hidden_size"))
        self.assertEqual(view.get("model_id"), "qwen3_8b")

    def test_calibrated_models_are_in_domain_on_the_native_shape(self) -> None:
        for model_id in CALIBRATED_TEXT_MODEL_IDS:
            with self.subTest(model_id=model_id):
                self.assertEqual(
                    guard.observation_architecture_refusals(
                        self._native_record(model_id)
                    ),
                    [],
                )

    def test_out_of_domain_models_are_refused_on_the_native_shape(self) -> None:
        # qwen3_vl_8b is registered as a text-only proxy, but its checkpoint is a
        # real VL tower; the guard must read the checkpoint, not the role.
        self.assertIn(
            guard.VISION_LANGUAGE_REASON,
            [
                code
                for code, _ in guard.observation_architecture_refusals(
                    self._native_record("qwen3_vl_8b")
                )
            ],
        )
        reasons = [
            code
            for code, _ in guard.observation_architecture_refusals(
                self._native_record("qwen3_6_27b")
            )
        ]
        self.assertIn(guard.HYBRID_ATTENTION_REASON, reasons)

    def test_unresolvable_model_id_is_refused_not_assumed_text(self) -> None:
        # A model_id naming no registry entry yields no checkpoint path, so the
        # architecture cannot be verified -- refuse.  This costs nothing real:
        # build_native_record raises on an id absent from the inventory, and all
        # canonical observation rows carry their own checkpoint path, so neither
        # calibrated path depends on registry resolution.
        for unknown in ("some_new_text_model", "some_new_vl_model"):
            with self.subTest(model_id=unknown):
                self.assertTrue(
                    guard.observation_architecture_refusals(
                        self._native_record(unknown)
                    )
                )

    def test_qwen3_5_refusal_reads_its_real_layer_schedule(self) -> None:
        # qwen3p5_4b is registered only in the Qwen3.5 campaign's own inventory
        # (that campaign runs in a separate venv).  Before the registry search
        # included campaign inventories, the id resolved to no path, the view was
        # an empty config, and ``is_hybrid_attention({})`` refused it for *absent*
        # evidence -- the same verdict, but unfalsifiable, and it would have
        # survived unchanged if the checkpoint were in fact uniform.  Assert the
        # refusal is grounded in the real 24-linear/8-full schedule instead.
        view = guard.architecture_view_of_observation(
            self._native_record("qwen3p5_4b")
        )
        self.assertIsNotNone(view)
        layer_types = view.get("layer_types") or (
            view.get("text_config") or {}
        ).get("layer_types")
        self.assertIsNotNone(layer_types, "registry lookup never reached config.json")
        self.assertIn("linear_attention", layer_types)
        reasons = [
            code
            for code, _ in guard.observation_architecture_refusals(
                self._native_record("qwen3p5_4b")
            )
        ]
        self.assertEqual(
            reasons, [guard.VISION_LANGUAGE_REASON, guard.HYBRID_ATTENTION_REASON]
        )

    def test_qwen2_5_holdout_models_are_not_refused(self) -> None:
        # The same campaign inventory also registers two dense Qwen2.5 anchors.
        # Widening the registry must not drag them out of domain: they are
        # uniform softmax text stacks, and refusing them would shrink the
        # throughput holdout the Qwen3.5 comparison rests on.
        for model_id in ("qwen2p5_14b", "qwen2p5_32b"):
            with self.subTest(model_id=model_id):
                view = guard.architecture_view_of_observation(
                    self._native_record(model_id)
                )
                self.assertIsNotNone(view.get("hidden_size"))
                self.assertEqual(
                    guard.observation_architecture_refusals(
                        self._native_record(model_id)
                    ),
                    [],
                )


class TestUnparseableConfigsFailClosed(unittest.TestCase):
    def test_unknown_layer_kind_is_refused_not_raised(self) -> None:
        # layer_type_counts raises on an unknown kind.  The guard must convert
        # that into a refusal: an exception here would crash the predictor
        # instead of downgrading the candidate.
        config = {
            "hidden_size": 2560,
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "some_future_attention"],
        }
        self.assertTrue(guard.is_hybrid_attention(config))

    def test_inconsistent_layer_types_length_is_refused(self) -> None:
        config = {
            "hidden_size": 2560,
            "num_hidden_layers": 8,
            "layer_types": ["full_attention"],
        }
        self.assertTrue(guard.is_hybrid_attention(config))

    def test_non_mapping_is_refused_on_both_axes(self) -> None:
        for bad in (None, [], "qwen3_14b", 0):
            with self.subTest(bad=bad):
                self.assertTrue(guard.is_vision_language(bad))
                self.assertTrue(guard.is_hybrid_attention(bad))

    def test_empty_config_is_refused_as_hybrid(self) -> None:
        # No geometry at all cannot be verified as uniform, so refuse.
        self.assertTrue(guard.is_hybrid_attention({}))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
