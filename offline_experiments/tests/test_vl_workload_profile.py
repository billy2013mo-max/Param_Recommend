from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from vl_workload_profile import (  # noqa: E402
    build_workload_profile,
    dimensions_to_grid_thw,
    visual_tokens_from_grid,
)


class VLWorkloadProfileTests(unittest.TestCase):
    def test_qwen_grid_formula_and_dimension_conversion(self) -> None:
        grid = dimensions_to_grid_thw(
            width=980,
            height=616,
            patch_size=14,
            spatial_merge_size=2,
        )
        self.assertEqual(grid, [1, 44, 70])
        self.assertEqual(visual_tokens_from_grid(grid, spatial_merge_size=2), 770)

    def test_profile_separates_visual_text_and_task_fields(self) -> None:
        profile = build_workload_profile(
            [
                {
                    "record_index": 7,
                    "task_family": "causal_vl_sft",
                    "images": [
                        {"grid_thw": [1, 44, 70], "visual_tokens": 770},
                        {"grid_thw": [1, 44, 70], "visual_tokens": 770},
                    ],
                    "text_tokens": 1000,
                    "label_tokens": 6,
                    "repeated_prompt_tokens": 500,
                },
                {
                    "record_index": 8,
                    "task_family": "classification_multilabel",
                    "images": [
                        {"width": 980, "height": 616, "visual_tokens": 770}
                    ],
                    "base_total_tokens_one_image_pad_each": 20,
                    "label_tokens": 5,
                },
            ],
            model_id="qwen3_vl_8b",
            processor_name="qwen3_vl_image_processor",
            processor_version="4.57.0.dev0",
            patch_size=14,
            spatial_merge_size=2,
            image_min_pixels=3136,
            image_max_pixels=12845056,
        )
        self.assertEqual(profile["schema"], "sft_vl_workload_profile/v1")
        binding = profile["processor_binding"]
        self.assertEqual(
            binding["contract_sha256"],
            __import__("runtime_evidence").sha256_json(
                {key: value for key, value in binding.items() if key != "contract_sha256"}
            ),
        )
        self.assertEqual(profile["summary"]["records"], 2)
        self.assertEqual(
            profile["summary_by_task_family"]["classification_multilabel"][
                "visual_tokens_total"
            ]["mean"],
            770,
        )
        first = profile["records"][0]
        self.assertEqual(first["visual_tokens_total"], 1540)
        self.assertEqual(first["text_tokens"], 1000)
        self.assertEqual(first["repeated_prompt_tokens"], 500)
        self.assertAlmostEqual(first["image_token_ratio"], 1540 / 2540)

    def test_declared_visual_tokens_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "declared visual_tokens"):
            build_workload_profile(
                [
                    {
                        "images": [
                            {"grid_thw": [1, 44, 70], "visual_tokens": 771}
                        ],
                        "text_tokens": 10,
                    }
                ],
                model_id="qwen3_vl_8b",
                processor_name="processor",
                processor_version="v1",
                patch_size=14,
                spatial_merge_size=2,
            )


if __name__ == "__main__":
    unittest.main()
