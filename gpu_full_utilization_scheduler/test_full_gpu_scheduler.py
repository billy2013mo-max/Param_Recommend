from __future__ import annotations

import unittest

from gpu_full_utilization_scheduler.full_gpu_scheduler import (
    SchedulingError,
    build_strategy,
    parse_gpu_ids,
    plan_jobs,
    resolve_gpu_context,
)


class FullGpuSchedulerTests(unittest.TestCase):
    def test_parse_gpu_ids_supports_ranges_and_rejects_duplicates(self) -> None:
        self.assertEqual(parse_gpu_ids("0,2,4-7"), (0, 2, 4, 5, 6, 7))
        with self.assertRaisesRegex(SchedulingError, "unique"):
            parse_gpu_ids("0,1,1")

    def test_gpu_limit_selects_prefix_of_available_ids(self) -> None:
        context = resolve_gpu_context(gpu_ids=(2, 4, 6, 8), gpu_limit=3)
        self.assertEqual(context.available_gpu_ids, (2, 4, 6, 8))
        self.assertEqual(context.selected_gpu_ids, (2, 4, 6))
        self.assertEqual(context.gpu_limit, 3)

    def test_eight_gpu_strategy_matches_current_campaign(self) -> None:
        context = resolve_gpu_context(gpu_ids=tuple(range(8)), gpu_limit=8)
        strategy = build_strategy(context, (1, 2, 4, 8))
        observed = {
            row["job_gpu_count"]: (
                row["max_concurrency"],
                row["idle_gpu_count"],
            )
            for row in strategy["homogeneous"]
        }
        self.assertEqual(observed, {1: (8, 0), 2: (4, 0), 4: (2, 0), 8: (1, 0)})
        self.assertEqual(
            strategy["mixed_exact_fill"]["job_gpu_counts"], [8]
        )

    def test_six_gpu_strategy_exposes_homogeneous_gap_and_mixed_fill(self) -> None:
        context = resolve_gpu_context(gpu_ids=None, gpu_limit=6)
        strategy = build_strategy(context, (1, 2, 4))
        four = next(
            row for row in strategy["homogeneous"] if row["job_gpu_count"] == 4
        )
        self.assertEqual(four["max_concurrency"], 1)
        self.assertEqual(four["idle_gpu_count"], 2)
        self.assertFalse(four["full_utilization"])
        self.assertEqual(
            sum(strategy["mixed_exact_fill"]["job_gpu_counts"]), 6
        )

    def test_homogeneous_repeat_fills_tail_and_marks_repeat_nonranking(self) -> None:
        context = resolve_gpu_context(gpu_ids=None, gpu_limit=8)
        jobs = [
            {"job_id": f"one-{index}", "gpu_count": 1}
            for index in range(10)
        ]
        plan = plan_jobs(
            jobs,
            context=context,
            policy="homogeneous",
            tail_policy="repeat",
        )
        self.assertEqual(plan["summary"]["waves"], 2)
        self.assertEqual(plan["summary"]["synthetic_repeats"], 6)
        self.assertEqual(plan["summary"]["partial_waves"], 0)
        self.assertTrue(all(wave["used_gpu_count"] == 8 for wave in plan["waves"]))
        self.assertTrue(
            all(
                row["synthetic_repeat"] is True
                and row["ranking_eligible"] is False
                for row in plan["synthetic_repeat_jobs"]
            )
        )

    def test_homogeneous_rejects_four_gpu_jobs_under_six_gpu_limit(self) -> None:
        context = resolve_gpu_context(gpu_ids=None, gpu_limit=6)
        jobs = [{"job_id": "four", "gpu_count": 4}]
        with self.assertRaisesRegex(SchedulingError, "cannot fill"):
            plan_jobs(
                jobs,
                context=context,
                policy="homogeneous",
                tail_policy="repeat",
            )

    def test_mixed_plan_fills_six_with_four_plus_two(self) -> None:
        context = resolve_gpu_context(gpu_ids=(1, 3, 5, 7, 9, 11), gpu_limit=6)
        jobs = [
            {"job_id": "four", "gpu_count": 4},
            {"job_id": "two", "gpu_count": 2},
        ]
        plan = plan_jobs(
            jobs,
            context=context,
            policy="mixed",
            tail_policy="error",
        )
        self.assertEqual(plan["summary"]["waves"], 1)
        self.assertEqual(plan["waves"][0]["used_gpu_count"], 6)
        self.assertEqual(
            [row["gpu_mask"] for row in plan["waves"][0]["jobs"]],
            [[1, 3, 5, 7], [9, 11]],
        )

    def test_mixed_plan_rejects_mathematically_impossible_five_gpu_fill(self) -> None:
        context = resolve_gpu_context(gpu_ids=None, gpu_limit=5)
        jobs = [
            {"job_id": "four", "gpu_count": 4},
            {"job_id": "two", "gpu_count": 2},
        ]
        with self.assertRaisesRegex(SchedulingError, "cannot fill 5 GPUs"):
            plan_jobs(
                jobs,
                context=context,
                policy="mixed",
                tail_policy="repeat",
            )

    def test_partial_requires_explicit_allowance(self) -> None:
        context = resolve_gpu_context(gpu_ids=None, gpu_limit=8)
        jobs = [{"job_id": "one", "gpu_count": 1}]
        with self.assertRaisesRegex(SchedulingError, "allow-partial"):
            plan_jobs(
                jobs,
                context=context,
                policy="homogeneous",
                tail_policy="partial",
                require_full=True,
            )
        plan = plan_jobs(
            jobs,
            context=context,
            policy="homogeneous",
            tail_policy="partial",
            require_full=False,
        )
        self.assertEqual(plan["summary"]["partial_waves"], 1)


if __name__ == "__main__":
    unittest.main()
