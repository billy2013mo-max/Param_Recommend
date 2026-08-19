from __future__ import annotations

from collections import Counter, defaultdict
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_h800_packing_transfer_validation_v1 as campaign  # noqa: E402
from common import read_json, read_jsonl, sha256_file  # noqa: E402
from run_job import validate_job  # noqa: E402


class PackingTransferValidationPreparationTests(unittest.TestCase):
    def test_queue_is_small_balanced_and_executable(self) -> None:
        rows = read_jsonl(campaign.QUEUE)
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({row["job_id"] for row in rows}), 24)
        self.assertEqual(
            [row["execution_sequence_index"] for row in rows], list(range(24))
        )
        self.assertEqual(
            Counter(int(row["gpu_count"]) for row in rows),
            Counter({1: 8, 2: 8, 4: 8}),
        )
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            validate_job(row)
            self.assertTrue(row["packing"])
            self.assertEqual(row["split_role"], "prospective_holdout")
            self.assertFalse(row["publication_allowed"])
            self.assertNotIn("vl", row["model_id"].lower())
            self.assertEqual(
                row["dataset_registry_sha256"], sha256_file(campaign.DATASET_REGISTRY)
            )
            if row["ranking_eligible"] is True:
                groups[row["ranking_group_id"]].append(row)
        self.assertEqual(len(groups), 6)
        self.assertEqual(
            sorted(len(current) for current in groups.values()),
            [3, 4, 4, 4, 4, 4],
        )
        self.assertEqual(
            Counter(row["measurement_role"] for row in rows),
            Counter({"primary_ranking_candidate": 23, "transfer_noise_repeat": 1}),
        )

    def test_scope_spans_models_data_training_and_gpu_counts(self) -> None:
        rows = read_jsonl(campaign.QUEUE)
        self.assertEqual(len({row["model_id"] for row in rows}), 5)
        self.assertEqual({row["train_type"] for row in rows}, {"full", "lora"})
        self.assertEqual(len({row["source_dataset_id"] for row in rows}), 3)
        self.assertEqual({int(row["gpu_count"]) for row in rows}, {1, 2, 4})
        self.assertEqual({row["model_family"] for row in rows}, {"qwen3"})
        self.assertTrue(all(row["template"] == "qwen3_nothink" for row in rows))

    def test_design_forbids_exhaustive_queue_and_post_outcome_tuning(self) -> None:
        design = read_json(campaign.DESIGN)
        self.assertEqual(design["validation_scope"]["jobs"], 24)
        self.assertTrue(
            design["supersession"]["exhaustive_1584_job_design_must_not_run"]
        )
        self.assertFalse(
            design["prospective_contract"][
                "validation_outcomes_may_tune_model_or_margin"
            ]
        )


if __name__ == "__main__":
    unittest.main()
