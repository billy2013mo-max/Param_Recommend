from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
import json
import os
import signal
from pathlib import Path
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import scheduler  # noqa: E402
from approval_gate import (  # noqa: E402
    acquire_execution_lock,
    reject_retired_approval_plan,
)


def job(name: str, gpu_count: int) -> dict:
    return {
        "job_id": name,
        "gpu_count": gpu_count,
        "parallel_class": "gpu_partitionable",
    }


def adaptive_base(name: str, *, mbs: int = 4) -> dict:
    return {
        "schema": "sft_h800_calibration_job/v1",
        "campaign_id": "h800_calibration_candidate_v1",
        "phase_id": "developer_h800_1p7b_to_14b",
        "hardware_id": "local_h800_140g",
        "gpu_type": "NVIDIA H800 140GB HBM3",
        "required_runtime_gpu_name": "NVIDIA H800",
        "job_id": name,
        "kind": "throughput_screen",
        "gpu_count": 1,
        "mbs": mbs,
        "parallel_class": "gpu_partitionable",
    }


def boundary_family() -> list[dict]:
    family_id = "h800calfam-test"
    specifications = (
        ("anchor", 0, 4, {"type": "always"}),
        (
            "fallback_half",
            1,
            2,
            {"type": "if_probe_outcome", "probe": "anchor", "outcome": "oom"},
        ),
        (
            "upward_1",
            2,
            8,
            {"type": "if_probe_outcome", "probe": "anchor", "outcome": "success"},
        ),
        (
            "upward_2",
            3,
            16,
            {
                "type": "if_probe_outcome",
                "probe": "upward_1",
                "outcome": "success",
            },
        ),
    )
    rows = []
    for probe, sequence_index, mbs, condition in specifications:
        row = adaptive_base(f"{family_id}-{probe}", mbs=mbs)
        row["boundary_probe"] = {
            "family_id": family_id,
            "probe": probe,
            "sequence_index": sequence_index,
            "condition": condition,
        }
        rows.append(row)
    return rows


def packing_pair() -> list[dict]:
    pair_id = "h800packpair-test"
    treatments = ("unpacked", "packed", "packed", "unpacked")
    rows = []
    prior_ids = []
    for sequence_index, treatment in enumerate(treatments):
        row = adaptive_base(f"{pair_id}-{sequence_index}", mbs=1)
        row["kind"] = "throughput"
        row["packing_pair"] = {
            "pair_id": pair_id,
            "order": "ABBA",
            "sequence_index": sequence_index,
            "treatment": treatment,
            "condition": (
                {"type": "always"}
                if not prior_ids
                else {"type": "all_jobs_succeeded", "job_ids": list(prior_ids)}
            ),
        }
        prior_ids.append(row["job_id"])
        rows.append(row)
    return rows


class SchedulerTests(unittest.TestCase):
    def test_adaptive_graph_rejects_missing_cross_group_duplicate_and_4090(self) -> None:
        valid = boundary_family()
        self.assertIsNotNone(scheduler.validate_adaptive_queue(valid))

        missing = json.loads(json.dumps(valid))
        missing[2]["boundary_probe"]["condition"]["probe"] = "missing"
        with self.assertRaisesRegex(scheduler.AdaptiveQueueError, "missing"):
            scheduler.validate_adaptive_queue(missing)

        duplicate = json.loads(json.dumps(valid))
        duplicate[1]["job_id"] = duplicate[0]["job_id"]
        with self.assertRaisesRegex(scheduler.AdaptiveQueueError, "unique"):
            scheduler.validate_adaptive_queue(duplicate)

        crossed = [*valid, *packing_pair()]
        crossed[-1]["packing_pair"]["condition"]["job_ids"] = [valid[0]["job_id"]]
        with self.assertRaisesRegex(scheduler.AdaptiveQueueError, "cross-family"):
            scheduler.validate_adaptive_queue(crossed)

        non_h800 = json.loads(json.dumps(valid))
        non_h800[0]["gpu_type"] = "NVIDIA GeForce RTX 4090"
        with self.assertRaisesRegex(scheduler.AdaptiveQueueError, "H800"):
            scheduler.validate_adaptive_queue(non_h800)

    def test_non_terminal_dependency_never_becomes_ready(self) -> None:
        rows = boundary_family()
        plan = scheduler.validate_adaptive_queue(rows)
        self.assertIsNotNone(plan)
        upward = rows[2]["job_id"]
        anchor = rows[0]["job_id"]
        self.assertEqual(
            scheduler.adaptive_job_state(upward, plan, {}),  # type: ignore[arg-type]
            ("wait", None),
        )
        with self.assertRaisesRegex(scheduler.AdaptiveQueueError, "non-terminal"):
            scheduler.adaptive_job_state(  # type: ignore[arg-type]
                upward, plan, {anchor: "software_failure"}
            )

    def test_packing_abba_requires_every_prior_success(self) -> None:
        rows = packing_pair()
        plan = scheduler.validate_adaptive_queue(rows)
        self.assertIsNotNone(plan)
        first, second, third, fourth = [row["job_id"] for row in rows]
        self.assertEqual(
            scheduler.adaptive_job_state(second, plan, {first: "success"})[0],  # type: ignore[arg-type]
            "ready",
        )
        state, reason = scheduler.adaptive_job_state(  # type: ignore[arg-type]
            fourth,
            plan,
            {first: "success", second: "oom", third: scheduler.CONDITIONAL_SKIPPED},
        )
        self.assertEqual(state, "skipped")
        self.assertIn("required=success", str(reason))

    def test_single_gpu_jobs_fill_the_configured_pool_in_one_wave(self) -> None:
        with patch.object(
            scheduler, "PERFORMANCE_PARALLELISM", "disjoint_gpu_masks"
        ):
            waves = scheduler.preview_waves(
                [job(f"one-{index}", 1) for index in range(len(scheduler.GPU_IDS))]
            )
        self.assertEqual(len(waves), 1)
        self.assertEqual(
            {tuple(row["gpu_mask"]) for row in waves[0]},
            {(value,) for value in scheduler.GPU_IDS},
        )

    def test_two_dual_gpu_jobs_share_one_wave(self) -> None:
        with patch.object(
            scheduler, "PERFORMANCE_PARALLELISM", "disjoint_gpu_masks"
        ):
            waves = scheduler.preview_waves([job("two-a", 2), job("two-b", 2)])
        self.assertEqual(len(waves), 1)
        self.assertEqual(
            {tuple(row["gpu_mask"]) for row in waves[0]},
            {tuple(scheduler.GPU_IDS[:2]), tuple(scheduler.GPU_IDS[2:4])},
        )

    def test_mixed_wave_fills_the_configured_pool(self) -> None:
        with patch.object(
            scheduler, "PERFORMANCE_PARALLELISM", "disjoint_gpu_masks"
        ):
            waves = scheduler.preview_waves(
                [job("two", 2)]
                + [
                    job(f"one-{index}", 1)
                    for index in range(len(scheduler.GPU_IDS) - 2)
                ]
            )
        self.assertEqual(len(waves), 1)
        assigned = {gpu for row in waves[0] for gpu in row["gpu_mask"]}
        self.assertEqual(assigned, set(scheduler.GPU_IDS))

    def test_four_gpu_job_is_pool_exclusive(self) -> None:
        self.assertIsNone(
            scheduler.allocate(
                job("four", 4), set(scheduler.GPU_IDS[1:]), running_count=1
            )
        )
        self.assertEqual(
            scheduler.allocate(
                job("four", 4), set(scheduler.GPU_IDS), running_count=0
            ),
            list(scheduler.GPU_IDS[:4]),
        )

    def test_two_opted_in_four_gpu_jobs_fill_disjoint_wave(self) -> None:
        rows = []
        for name in ("four-a", "four-b"):
            current = job(name, 4)
            current.update(
                {
                    "parallel_class": "disjoint_wave",
                    "requires_external_node_idle": False,
                    "allow_disjoint_wave_for_large_job": True,
                    "homogeneous_card_count_wave": True,
                    "strict_queue_order": True,
                }
            )
            rows.append(current)
        with patch.object(
            scheduler, "PERFORMANCE_PARALLELISM", "disjoint_gpu_masks"
        ):
            waves = scheduler.preview_waves(rows)
        self.assertEqual(len(waves), 1)
        self.assertEqual(
            {tuple(row["gpu_mask"]) for row in waves[0]},
            {tuple(scheduler.GPU_IDS[:4]), tuple(scheduler.GPU_IDS[4:8])},
        )

    def test_five_to_seven_gpu_jobs_are_pool_exclusive_with_exact_masks(self) -> None:
        gpu_pool = tuple(range(8))
        with patch.object(scheduler, "GPU_IDS", gpu_pool):
            for gpu_count in (5, 6, 7):
                requested = job(f"many-{gpu_count}", gpu_count)
                self.assertIsNone(
                    scheduler.allocate(
                        requested, set(gpu_pool[1:]), running_count=1
                    )
                )
                self.assertEqual(
                    scheduler.allocate(requested, set(gpu_pool), running_count=0),
                    list(gpu_pool[:gpu_count]),
                )

    def test_opt_in_large_job_uses_only_currently_available_gpus(self) -> None:
        requested = job("available-four", 4)
        requested.update(
            {
                "parallel_class": "available_pool",
                "requires_external_node_idle": False,
                "allow_available_pool_for_large_job": True,
            }
        )
        available = {1, 2, 3, 5, 6}
        with patch.object(scheduler, "PERFORMANCE_PARALLELISM", "disjoint_gpu_masks"):
            self.assertFalse(scheduler.requires_exclusive_pool(requested))
            self.assertEqual(
                scheduler.allocate(requested, available, running_count=0),
                [1, 2, 3, 5],
            )
            self.assertIsNone(
                scheduler.allocate(requested, available, running_count=1)
            )

    def test_execute_verifies_exact_input_queue_before_scheduler_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "bound.jsonl"
            row = job("one", 1)
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            def reject_while_scheduler_lock_is_held(*args, **kwargs):
                with self.assertRaisesRegex(PermissionError, "gate is busy"):
                    acquire_execution_lock(root / "runtime", exclusive=True)
                raise PermissionError("queue rejected")

            with (
                patch.object(
                    sys,
                    "argv",
                    ["scheduler.py", "--input", str(input_path), "--execute"],
                ),
                patch.object(scheduler, "RUNTIME_DIR", root / "runtime"),
                patch.object(
                    scheduler,
                    "verify_approval",
                    side_effect=reject_while_scheduler_lock_is_held,
                ) as verify,
                patch.object(scheduler, "run_scheduler") as launch,
            ):
                with self.assertRaisesRegex(PermissionError, "queue rejected"):
                    scheduler.main()
            verify.assert_called_once_with(
                queue_path=input_path,
                queue_rows=[row],
                acquire_lock=False,
            )
            launch.assert_not_called()
            released = acquire_execution_lock(root / "runtime", exclusive=True)
            released.close()

    def test_execute_never_launches_a_retired_h800_approval_design(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "retired.jsonl"
            row = job("retired-one", 1)
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            def retired_gate(*args, **kwargs):
                reject_retired_approval_plan(
                    {
                        "h800_calibration": {
                            "schema": "sft_h800_calibration_approval_freeze/v1"
                        }
                    },
                    operation="authorize a run or scheduler launch",
                )

            with (
                patch.object(
                    sys,
                    "argv",
                    ["scheduler.py", "--input", str(input_path), "--execute"],
                ),
                patch.object(scheduler, "RUNTIME_DIR", root / "runtime"),
                patch.object(scheduler, "verify_approval", side_effect=retired_gate),
                patch.object(scheduler, "run_scheduler") as launch,
            ):
                with self.assertRaisesRegex(
                    PermissionError, "Retired H800 calibration"
                ):
                    scheduler.main()
            launch.assert_not_called()


class SchedulerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_group_timeout_escalates_term_then_kill_and_reaps(self) -> None:
        killed = asyncio.Event()

        class Process:
            pid = 424242

            async def communicate(self):
                await killed.wait()
                return b"timed out", None

        def killpg(pid, sig):
            self.assertEqual(pid, 424242)
            if sig == signal.SIGKILL:
                killed.set()

        with patch.object(os, "killpg", side_effect=killpg) as terminate:
            process = Process()
            communication = asyncio.create_task(process.communicate())
            stdout, stderr = await scheduler._terminate_process_group(
                process,  # type: ignore[arg-type]
                communication,
                grace_seconds=0,
            )
        self.assertEqual((stdout, stderr), (b"timed out", None))
        self.assertEqual(
            [call.args[1] for call in terminate.call_args_list],
            [signal.SIGTERM, signal.SIGKILL],
        )

    async def test_cancelling_long_running_child_reaps_its_process_group(self) -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        pid = process.pid
        communication = asyncio.create_task(
            scheduler.communicate_with_process_group_timeout(
                process,
                timeout_seconds=scheduler.ADAPTIVE_RUN_TIMEOUT_SECONDS,
            )
        )
        await asyncio.sleep(0.05)
        communication.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await communication
        self.assertIsNotNone(process.returncode)
        with self.assertRaises(ProcessLookupError):
            os.killpg(pid, 0)

    async def test_campaign_deadline_stops_new_launches_and_halts_after_current(self) -> None:
        rows = boundary_family()
        plan = scheduler.validate_adaptive_queue(rows)
        self.assertIsNotNone(plan)
        plan["campaign_hard_wall_time_seconds"] = 1  # type: ignore[index]
        clock_values = iter((0.0, 0.0, 0.0, 2.0, 2.0, 2.0))
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def long_running(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launch = AsyncMock(side_effect=long_running)
            with (
                patch.object(scheduler, "validate_adaptive_queue", return_value=plan),
                patch.object(scheduler, "execute_family", new=launch),
                patch.object(scheduler, "occupied_gpu_ids", return_value=set()),
                patch.object(scheduler, "RESULTS_DIR", root / "results"),
            ):
                with self.assertRaises(scheduler.CampaignBudgetExceeded):
                    await scheduler.run_scheduler(
                        rows,
                        root / "events.jsonl",
                        monotonic_clock=lambda: next(clock_values),
                    )
            events = (root / "events.jsonl").read_text(encoding="utf-8")
        self.assertEqual(launch.await_count, 1)
        self.assertTrue(started.is_set())
        self.assertTrue(cancelled.is_set())
        self.assertIn('"event": "campaign_budget_exhausted"', events)
        self.assertIn('"new_launches_disabled": true', events)
        self.assertIn('"event": "family_cancelled"', events)
        self.assertIn('"event": "scheduler_halted"', events)

    async def test_first_upward_oom_skips_later_probe_without_training_status(self) -> None:
        rows = boundary_family()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event_path = root / "events.jsonl"
            launch = AsyncMock(side_effect=["success", "oom"])
            with (
                patch.object(scheduler, "execute_family", new=launch),
                patch.object(scheduler, "occupied_gpu_ids", return_value=set()),
                patch.object(scheduler, "RESULTS_DIR", root / "results"),
            ):
                await scheduler.run_scheduler(
                    rows, event_path, execution_id="scheduler-test"
                )
            launched_ids = [call.args[0]["job_id"] for call in launch.await_args_list]
            self.assertEqual(launched_ids, [rows[0]["job_id"], rows[2]["job_id"]])
            for skipped in (rows[1], rows[3]):
                terminal = json.loads(
                    (
                        root
                        / "results"
                        / skipped["job_id"]
                        / "scheduler_terminal.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(terminal["classification"], "conditional_skipped")
                self.assertFalse(terminal["calibration_observation_eligible"])
                self.assertFalse(
                    (root / "results" / skipped["job_id"] / "status.json").exists()
                )

    async def test_mbs16_success_caps_family_without_out_of_domain_launch(self) -> None:
        rows = boundary_family()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launch = AsyncMock(side_effect=["success", "success", "success"])
            with (
                patch.object(scheduler, "execute_family", new=launch),
                patch.object(scheduler, "occupied_gpu_ids", return_value=set()),
                patch.object(scheduler, "RESULTS_DIR", root / "results"),
            ):
                await scheduler.run_scheduler(rows, root / "events.jsonl")
        launched_mbs = [call.args[0]["mbs"] for call in launch.await_args_list]
        self.assertEqual(launched_mbs, [4, 8, 16])
        self.assertNotIn(32, launched_mbs)

    async def test_adaptive_software_failure_halts_retryably_and_keeps_successors_blocked(
        self,
    ) -> None:
        rows = boundary_family()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launch = AsyncMock(
                side_effect=scheduler.RetryableAdaptiveOutcome("software failure")
            )
            with (
                patch.object(scheduler, "execute_family", new=launch),
                patch.object(scheduler, "occupied_gpu_ids", return_value=set()),
                patch.object(scheduler, "RESULTS_DIR", root / "results"),
            ):
                with self.assertRaises(scheduler.RetryableAdaptiveOutcome):
                    await scheduler.run_scheduler(rows, root / "events.jsonl")
            self.assertEqual(launch.await_count, 1)
            self.assertEqual(launch.await_args.args[0]["job_id"], rows[0]["job_id"])
            self.assertFalse(any((root / "results").rglob("scheduler_terminal.json")))

    async def test_launcher_failure_is_fatal_instead_of_consumed(self) -> None:
        family = {
            **job("one", 1),
            "kind": "throughput",
        }
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "events.jsonl"
            with patch.object(
                scheduler,
                "execute_concrete",
                new=AsyncMock(return_value="launcher_failed"),
            ):
                with self.assertRaises(scheduler.FatalLaunchError):
                    await scheduler.execute_family(family, [1], event_path)

    async def test_approval_rejection_is_fatal_instead_of_consumed(self) -> None:
        family = {
            **job("one", 1),
            "kind": "throughput",
        }
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "events.jsonl"
            with patch.object(
                scheduler,
                "execute_concrete",
                new=AsyncMock(return_value="approval_rejected"),
            ):
                with self.assertRaises(scheduler.FatalLaunchError):
                    await scheduler.execute_family(family, [1], event_path)

    async def test_scheduler_requeues_once_and_halts_after_fatal_launch_error(
        self,
    ) -> None:
        family = {
            **job("one", 1),
            "kind": "throughput",
        }
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "events.jsonl"
            launch = AsyncMock(side_effect=scheduler.FatalLaunchError("stale approval"))
            with (
                patch.object(scheduler, "execute_family", new=launch),
                patch.object(scheduler, "occupied_gpu_ids", return_value=set()),
            ):
                with self.assertRaises(scheduler.FatalLaunchError):
                    await scheduler.run_scheduler([family], event_path)
            events = event_path.read_text(encoding="utf-8")
        self.assertEqual(launch.await_count, 1)
        self.assertIn('"event": "family_requeued"', events)
        self.assertIn('"event": "scheduler_halted"', events)

    async def test_busy_gpu_is_skipped_while_idle_pool_gpu_runs(self) -> None:
        family = {
            **job("one", 1),
            "kind": "throughput",
        }
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "events.jsonl"
            launch = AsyncMock(return_value=None)
            idle_gpu = scheduler.GPU_IDS[0]
            busy_gpu = scheduler.GPU_IDS[1]
            with (
                patch.object(scheduler, "execute_family", new=launch),
                patch.object(scheduler, "occupied_gpu_ids", return_value={busy_gpu}),
                patch.object(
                    scheduler,
                    "PERFORMANCE_PARALLELISM",
                    "disjoint_gpu_masks",
                ),
            ):
                await scheduler.run_scheduler(
                    [family], event_path, initial_blocked={busy_gpu}
                )
        self.assertEqual(launch.await_args.args[1], [idle_gpu])


if __name__ == "__main__":
    unittest.main()
