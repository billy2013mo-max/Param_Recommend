from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from runtime_evidence import sha256_json  # noqa: E402
from vl_workload_profile_v2 import build_workload_profile  # noqa: E402


class VLWorkloadProfileV2Tests(unittest.TestCase):
    def test_mixed_image_video_keeps_language_and_vision_work_separate(self) -> None:
        report = build_workload_profile(
            [
                {
                    "sample_id": "mixed-1",
                    "images": [{"grid_thw": [1, 44, 70]}],
                    "videos": [
                        {
                            "grid_thw": [8, 12, 20],
                            "sampled_frames": 16,
                            "duration_seconds": 8.0,
                            "source_fps": 30.0,
                            "sample_fps": 2.0,
                        }
                    ],
                    "text_tokens": 100,
                    "label_tokens": 10,
                }
            ],
            model_id="qwen3_vl_4b",
            processor_name="processor",
            processor_version="v2-test",
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
            video_sample_fps=2.0,
            video_max_frames=32,
        )
        self.assertEqual(report["schema"], "sft_vl_workload_profile/v2")
        row = report["records"][0]
        self.assertEqual(row["image_visual_tokens"], 770)
        self.assertEqual(row["video_visual_tokens"], 480)
        self.assertEqual(row["visual_tokens_total"], 1250)
        self.assertEqual(row["raw_patch_units_total"], 3080 + 1920)
        self.assertEqual(row["sampled_video_frames_total"], 16)
        self.assertEqual(row["total_tokens"], 1350)
        binding = dict(report["processor_binding"])
        digest = binding.pop("contract_sha256")
        self.assertEqual(digest, sha256_json(binding))

    def test_video_frame_grid_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sampled_frames disagrees"):
            build_workload_profile(
                [
                    {
                        "videos": [
                            {"grid_thw": [8, 12, 20], "sampled_frames": 15}
                        ],
                        "text_tokens": 10,
                    }
                ],
                model_id="qwen3_vl_4b",
                processor_name="processor",
                processor_version="v2-test",
                patch_size=14,
                spatial_merge_size=2,
                temporal_patch_size=2,
            )

    def test_declared_pixel_tensor_size_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "pixel_values_elements"):
            build_workload_profile(
                [
                    {
                        "images": [
                            {
                                "grid_thw": [1, 4, 4],
                                "pixel_values_elements": 1,
                            }
                        ],
                        "text_tokens": 10,
                    }
                ],
                model_id="qwen3_vl_4b",
                processor_name="processor",
                processor_version="v2-test",
                patch_size=14,
                spatial_merge_size=2,
                temporal_patch_size=2,
            )


if __name__ == "__main__":
    unittest.main()
