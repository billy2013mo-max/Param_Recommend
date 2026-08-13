from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_h800_profile_requirements import build_requirements  # noqa: E402


class H800ProfileRequirementsTests(unittest.TestCase):
    def test_requirements_are_non_executable_and_preserve_scenario_contract(self) -> None:
        design = {
            "campaign_id": "c",
            "scenarios": [
                {
                    "scenario_id": "s",
                    "dataset_profile_id": "p",
                    "model_id": "qwen3_8b",
                    "cutoff_len": 512,
                    "scale_out_transition": "1_to_2",
                    "freshness": {"split_unit": "p"},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "design.json"
            path.write_text("{}\n", encoding="utf-8")
            result = build_requirements(design=design, design_path=path)
        self.assertEqual(result["schema"], "sft_h800_fresh_profile_requirements/v1")
        self.assertFalse(result["ready_for_materialization"])
        self.assertFalse(result["publication_allowed"])
        self.assertFalse(result["gpu_training_started"])
        row = result["scenarios"][0]
        self.assertEqual(row["required_metadata"]["profile_id"], "p")
        self.assertIn("sample_id", row["required_profile_row_fields"])
        self.assertFalse(row["required_bindings"]["runtime_dataset_registered"])


if __name__ == "__main__":
    unittest.main()
