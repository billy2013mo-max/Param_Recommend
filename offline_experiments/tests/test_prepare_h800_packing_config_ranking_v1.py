from __future__ import annotations

from collections import Counter, defaultdict
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_h800_packing_config_ranking_v1 as campaign  # noqa: E402
from common import read_jsonl, sha256_file  # noqa: E402
from run_job import validate_job  # noqa: E402


class PackingConfigRankingPreparationTests(unittest.TestCase):
    def test_mechanism_grid_matches_product_space(self) -> None:
        self.assertEqual(campaign.mechanisms(1), ((0, False), (0, True)))
        for gpu_count in (2, 4, 8):
            self.assertEqual(
                campaign.mechanisms(gpu_count),
                ((2, False), (2, True), (3, False), (3, True)),
            )

    def test_materialized_queues_are_balanced_and_executable(self) -> None:
        for queue in (campaign.QUEUE_FIT, campaign.QUEUE_HOLDOUT):
            rows = read_jsonl(queue)
            self.assertEqual(len(rows), campaign.EXPECTED_JOBS_PER_SPLIT)
            self.assertEqual(len({row["job_id"] for row in rows}), len(rows))
            self.assertEqual(
                [row["execution_sequence_index"] for row in rows],
                list(range(len(rows))),
            )
            self.assertEqual(
                Counter(row["gpu_count"] for row in rows),
                Counter({1: 24, 2: 36, 4: 36, 8: 36}),
            )
            self.assertEqual(
                Counter(bool(row["gc"]) for row in rows),
                Counter({False: 66, True: 66}),
            )
            self.assertEqual(
                Counter(row["measurement_role"] for row in rows),
                Counter(
                    {
                        "primary_ranking_candidate": 126,
                        "full_utilization_repeat": 6,
                    }
                ),
            )
            self.assertEqual(
                Counter(row["execution_wave_capacity"] for row in rows),
                Counter({8: 24, 4: 36, 2: 36, 1: 36}),
            )
            groups: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                validate_job(row)
                self.assertTrue(row["packing"])
                self.assertEqual(row["mbs"], 1)
                self.assertEqual(
                    row["dataset_registry_sha256"],
                    sha256_file(campaign.DATASET_REGISTRY),
                )
                if row["ranking_eligible"]:
                    groups[row["ranking_group_id"]].append(row)
            self.assertEqual(len(groups), campaign.EXPECTED_RANKING_GROUPS_PER_SPLIT)
            self.assertEqual(
                sorted({len(current) for current in groups.values()}), [6, 12]
            )


if __name__ == "__main__":
    unittest.main()
