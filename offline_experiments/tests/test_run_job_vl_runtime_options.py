from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_job import (  # noqa: E402
    apply_multimodal_runtime_options,
    resolve_dataset_dir,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VLRuntimeOptionTests(unittest.TestCase):
    def test_video_processor_options_are_copied_with_types(self) -> None:
        config: dict[str, object] = {}
        apply_multimodal_runtime_options(
            config,
            {
                "freeze_vision_tower": True,
                "freeze_multi_modal_projector": True,
                "freeze_language_model": False,
                "video_min_pixels": 256,
                "video_max_pixels": 65_536,
                "video_maxlen": 64,
                "video_fps": 2,
            },
        )
        self.assertEqual(config["video_min_pixels"], 256)
        self.assertEqual(config["video_max_pixels"], 65_536)
        self.assertEqual(config["video_maxlen"], 64)
        self.assertEqual(config["video_fps"], 2.0)
        self.assertIs(config["freeze_vision_tower"], True)

    def test_invalid_video_ranges_and_fps_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            apply_multimodal_runtime_options(
                {}, {"video_min_pixels": 1024, "video_max_pixels": 256}
            )
        for invalid in (0, -1, float("inf"), float("nan")):
            with self.subTest(video_fps=invalid):
                with self.assertRaisesRegex(ValueError, "video_fps"):
                    apply_multimodal_runtime_options({}, {"video_fps": invalid})

    def test_campaign_local_registry_is_hash_and_path_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "video.jsonl"
            data_path.write_text('{"prompt":"x"}\n', encoding="utf-8")
            registry_path = root / "dataset_info.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "video_probe": {
                            "file_name": str(data_path.resolve()),
                            "columns": {"prompt": "prompt"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            job = {
                "dataset_id": "video_probe",
                "dataset_dir": str(root),
                "dataset_registry_sha256": sha256_file(registry_path),
                "data_path": str(data_path),
                "data_sha256": sha256_file(data_path),
            }
            self.assertEqual(resolve_dataset_dir(job), root.resolve())

            registry_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "registry drifted"):
                resolve_dataset_dir(job)


if __name__ == "__main__":
    unittest.main()
