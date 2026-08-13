from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_job import idle_check_gpu_ids, validate_gpu_assignment, validate_job  # noqa: E402


def experiment_scope() -> dict:
    return {
        "training_scope": {
            "gpu_ids": [1, 2, 3, 4],
            "gpu_counts": [1, 2, 4],
            "max_gpu_count": 4,
            "exclusive_node_gpu_ids": [1, 2, 3, 4],
        },
        "measurement": {"performance_parallelism": "disjoint_gpu_masks"},
    }


class GpuScopeTests(unittest.TestCase):
    def test_allows_assignments_inside_shared_node_pool(self) -> None:
        validate_gpu_assignment({"gpu_count": 2}, [1, 2], experiment_scope())

    def test_allows_five_to_seven_card_assignments_when_explicitly_scoped(self) -> None:
        experiment = experiment_scope()
        experiment["training_scope"].update(
            {
                "gpu_ids": list(range(8)),
                "gpu_counts": [4, 5, 6, 7],
                "max_gpu_count": 7,
                "exclusive_node_gpu_ids": list(range(8)),
            }
        )
        for gpu_count in (5, 6, 7):
            validate_gpu_assignment(
                {"gpu_count": gpu_count}, list(range(gpu_count)), experiment
            )

    def test_runner_accepts_five_to_seven_card_zero_jobs(self) -> None:
        for gpu_count in (5, 6, 7):
            validate_job(
                {
                    "gpu_count": gpu_count,
                    "zero": "zero3",
                    "packing": False,
                    "mbs": 1,
                    "target_gbs": gpu_count * 8,
                }
            )

    def test_runner_accepts_eight_card_zero_job(self) -> None:
        validate_job(
            {
                "gpu_count": 8,
                "zero": "zero3",
                "packing": True,
                "mbs": 1,
                "gradient_accumulation_steps": 32,
                "target_gbs": 256,
            }
        )

    def test_rejects_eight_gpu_job(self) -> None:
        with self.assertRaises(PermissionError):
            validate_gpu_assignment({"gpu_count": 8}, list(range(8)), experiment_scope())

    def test_rejects_mask_outside_pool(self) -> None:
        with self.assertRaises(PermissionError):
            validate_gpu_assignment({"gpu_count": 2}, [1, 5], experiment_scope())

    def test_rejects_exclusive_check_outside_pool(self) -> None:
        experiment = experiment_scope()
        experiment["training_scope"]["exclusive_node_gpu_ids"] = list(range(8))
        with self.assertRaises(PermissionError):
            validate_gpu_assignment({"gpu_count": 1}, [1], experiment)

    def test_disjoint_policy_checks_only_assigned_mask(self) -> None:
        job = {"gpu_count": 1, "requires_external_node_idle": True}
        self.assertEqual(idle_check_gpu_ids(job, [3], experiment_scope()), [3])

    def test_exclusive_policy_checks_entire_pool(self) -> None:
        experiment = experiment_scope()
        experiment["measurement"]["performance_parallelism"] = "exclusive_pool"
        job = {"gpu_count": 1, "requires_external_node_idle": True}
        self.assertEqual(idle_check_gpu_ids(job, [3], experiment), [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
