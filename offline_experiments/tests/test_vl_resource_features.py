from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from vl_resource_features import (  # noqa: E402
    build_vl_resource_features,
    extend_throughput_work_per_step,
    phase_max_reference_bytes,
)
from vl_workload_profile_v2 import build_workload_profile  # noqa: E402


MODEL = {
    "id": "qwen3_vl_4b",
    "hidden_size": 2560,
    "vision_geometry": {
        "depth": 24,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "out_hidden_size": 2560,
    },
    "vision_config": {"deepstack_visual_indexes": [0, 1, 2]},
    "component_parameter_estimates": {
        "vision_tower": 306242560,
        "projector_or_merger": 109105152,
    },
}


class VLResourceFeatureTests(unittest.TestCase):
    def _profile(self):
        return build_workload_profile(
            [
                {
                    "images": [{"grid_thw": [1, 4, 4]}],
                    "videos": [{"grid_thw": [8, 4, 4], "sampled_frames": 16}],
                    "text_tokens": 100,
                    "label_tokens": 10,
                }
            ],
            model_id="qwen3_vl_4b",
            processor_name="processor",
            processor_version="test",
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )

    def test_phase_max_does_not_sum_language_and_vision(self) -> None:
        self.assertEqual(
            phase_max_reference_bytes(
                persistent_bytes=10,
                common_dynamic_bytes=5,
                language_dynamic_bytes=60,
                vision_dynamic_bytes=20,
            ),
            75,
        )

    def test_frozen_vl_features_keep_both_workloads(self) -> None:
        result = build_vl_resource_features(
            self._profile(),
            MODEL,
            physical_mbs=2,
            freeze_vision_tower=True,
            freeze_multi_modal_projector=True,
        )
        self.assertEqual(result["language_work"]["mean_visual_tokens_per_sample"], 36)
        self.assertEqual(result["vision_work"]["mean_raw_patch_units_per_sample"], 144)
        self.assertGreater(
            result["memory_proxies_bytes"]["vision_dynamic_microbatch"], 0
        )
        self.assertTrue(
            result["semantics"]["vision_and_language_dynamic_peaks_are_phase_max_not_sum"]
        )

    def test_unfrozen_scope_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen vision tower"):
            build_vl_resource_features(
                self._profile(),
                MODEL,
                physical_mbs=1,
                freeze_vision_tower=False,
                freeze_multi_modal_projector=True,
            )

    def test_throughput_extension_preserves_base_work(self) -> None:
        features = build_vl_resource_features(
            self._profile(),
            MODEL,
            physical_mbs=1,
            freeze_vision_tower=True,
            freeze_multi_modal_projector=True,
        )
        work = extend_throughput_work_per_step(
            {"computed_tokens": 1234},
            features,
            logical_samples_per_step=64,
        )
        self.assertEqual(work["computed_tokens"], 1234)
        self.assertGreater(work["vision_forward_flops"], 0)


if __name__ == "__main__":
    unittest.main()
