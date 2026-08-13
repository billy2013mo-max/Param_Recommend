from __future__ import annotations

import math
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from stage_decisions import (  # noqa: E402
    best_configuration_decisions,
    build_reports,
    fit_profiler_calibration,
    group_holdout_resource_evaluation,
    packing_decisions,
    profiler_points,
    scaling_decisions,
    scaling_eligible_request_ids,
    throughput_decisions,
    throughput_screening_decisions,
)


def result_row(
    request_id: str,
    repeat: int,
    rate: float,
    *,
    gpu_count: int = 1,
    zero: str = "none",
    gc: bool = False,
    mbs: int = 1,
    packing: bool = False,
) -> dict:
    return {
        "job_id": f"job-{request_id}-{gpu_count}-{mbs}-{int(packing)}-{repeat}",
        "kind": "throughput",
        "request_id": request_id,
        "model_id": "qwen3_8b",
        "train_type": "full",
        "dataset_id": "short_512",
        "cutoff_len": 512,
        "gpu_count": gpu_count,
        "zero": zero,
        "gc": gc,
        "mbs": mbs,
        "target_gbs": 64,
        "packing": packing,
        "repeat": repeat,
        "classification": "success",
        "samples_per_second": rate,
        "effective_tokens_per_second": rate * 100,
        "computed_tokens_per_second": rate * 120,
        "measured_seconds": 10.0,
        "max_allocated_bytes": 10_000,
        "max_reserved_bytes": 12_000,
    }


class StageDecisionTests(unittest.TestCase):
    def test_screening_shortlist_preserves_gpu_count_diversity(self) -> None:
        request = {"request_id": "screen", "repeats": 1}
        planned = []
        rows = []
        for gpu_count, mbs, rate in (
            (1, 1, 100.0),
            (1, 2, 120.0),
            (2, 1, 140.0),
            (2, 2, 150.0),
        ):
            job = result_row("screen", 0, rate, gpu_count=gpu_count, mbs=mbs)
            job["job_id"] = f"screen-{gpu_count}-{mbs}"
            job["kind"] = "throughput_screen"
            planned.append(job)
            rows.append(job)

        decision = throughput_screening_decisions(rows, [request], planned, top_k=3)[
            "decisions"
        ][0]

        self.assertEqual(decision["status"], "shortlisted")
        self.assertEqual(len(decision["shortlisted"]), 3)
        self.assertEqual(
            {row["gpu_count"] for row in decision["shortlisted"]},
            {1, 2},
        )
        selected = {(row["gpu_count"], row["mbs"]) for row in decision["shortlisted"]}
        self.assertIn((1, 2), selected)
        self.assertIn((2, 2), selected)

    def test_top_two_keep_lowest_resource_and_fastest_overall(self) -> None:
        request = {"request_id": "screen", "repeats": 1}
        rows = []
        for gpu_count, rate in ((1, 100.0), (2, 180.0), (4, 320.0)):
            job = result_row("screen", 0, rate, gpu_count=gpu_count)
            job["job_id"] = f"screen-{gpu_count}"
            job["kind"] = "throughput_screen"
            rows.append(job)

        decision = throughput_screening_decisions(rows, [request], rows, top_k=2)[
            "decisions"
        ][0]

        self.assertEqual(decision["status"], "shortlisted")
        self.assertEqual(
            {row["gpu_count"] for row in decision["shortlisted"]},
            {1, 4},
        )

    def test_inactive_historical_screen_cannot_measure_or_fail_candidate(self) -> None:
        request = {"request_id": "screen", "repeats": 1}
        planned = result_row("screen", 0, 0.0)
        planned.update({"job_id": "screen-active", "kind": "throughput_screen"})
        stale_success = dict(planned)
        stale_success.update(
            {
                "job_id": "screen-stale-success",
                "classification": "success",
                "samples_per_second": 999.0,
            }
        )
        stale_oom = dict(planned)
        stale_oom.update(
            {
                "job_id": "screen-stale-oom",
                "classification": "oom",
                "samples_per_second": None,
            }
        )

        report = throughput_screening_decisions(
            [stale_success, stale_oom],
            [request],
            [planned],
            top_k=1,
        )

        candidate = report["configurations"][0]
        self.assertEqual(candidate["status"], "pending")
        self.assertIsNone(candidate["samples_per_second"])
        self.assertEqual(candidate["job_ids"], [])

    def test_exact_active_screen_wins_over_inactive_screen_history(self) -> None:
        request = {"request_id": "screen", "repeats": 1}
        active = result_row("screen", 0, 100.0)
        active.update({"job_id": "screen-active", "kind": "throughput_screen"})
        stale = dict(active)
        stale.update({"job_id": "screen-stale", "samples_per_second": 999.0})

        candidate = throughput_screening_decisions(
            [stale, active],
            [request],
            [active],
            top_k=1,
        )["configurations"][0]

        self.assertEqual(candidate["measurement_source"], "screen")
        self.assertEqual(candidate["samples_per_second"], 100.0)
        self.assertEqual(candidate["job_ids"], ["screen-active"])

    def test_screen_formal_substitute_prefers_exact_active_formal(self) -> None:
        request = {"request_id": "screen", "repeats": 1}
        screen_plan = result_row("screen", 0, 0.0, mbs=2)
        screen_plan.update({"job_id": "screen-active", "kind": "throughput_screen"})
        active_formal = result_row("screen", 1, 100.0, mbs=2)
        active_formal["job_id"] = "formal-active"
        active_formal_plan = dict(active_formal)
        historical = result_row("screen", 0, 999.0, mbs=2)
        historical["job_id"] = "tput-historical"

        first = throughput_screening_decisions(
            [historical, active_formal],
            [request],
            [screen_plan],
            top_k=1,
            active_formal_jobs=[active_formal_plan],
        )["configurations"][0]
        second = throughput_screening_decisions(
            [active_formal, historical],
            [request],
            [screen_plan],
            top_k=1,
            active_formal_jobs=[active_formal_plan],
        )["configurations"][0]

        self.assertEqual(first["measurement_source"], "formal")
        self.assertEqual(first["formal_substitute_scope"], "active")
        self.assertEqual(first["job_ids"], ["formal-active"])
        self.assertEqual(first["samples_per_second"], 100.0)
        self.assertEqual(first, second)

    def test_healthy_active_formal_only_configuration_remains_eligible(self) -> None:
        request = {"request_id": "screen", "repeats": 1}
        screen = result_row("screen", 0, 300.0, gpu_count=4, mbs=4)
        screen.update({"job_id": "screen-four", "kind": "throughput_screen"})
        formal = result_row("screen", 0, 100.0, gpu_count=1, mbs=2)
        formal["job_id"] = "formal-one"

        report = throughput_screening_decisions(
            [screen, formal],
            [request],
            [screen],
            top_k=2,
            active_formal_jobs=[dict(formal)],
        )

        decision = report["decisions"][0]
        self.assertEqual(decision["status"], "shortlisted")
        self.assertEqual(
            {row["gpu_count"] for row in decision["shortlisted"]},
            {1, 4},
        )
        formal_candidate = next(
            row for row in decision["shortlisted"] if row["gpu_count"] == 1
        )
        self.assertEqual(formal_candidate["candidate_plan_source"], "active_formal")
        self.assertEqual(formal_candidate["measurement_source"], "formal")
        self.assertEqual(formal_candidate["formal_substitute_scope"], "active")
        self.assertEqual(formal_candidate["job_ids"], ["formal-one"])

    def test_historical_formal_substitute_is_one_deterministic_result(self) -> None:
        request = {"request_id": "screen", "repeats": 1}
        screen_plan = result_row("screen", 0, 0.0, mbs=2)
        screen_plan.update({"job_id": "screen-active", "kind": "throughput_screen"})
        historical_b = result_row("screen", 0, 300.0, mbs=2)
        historical_b["job_id"] = "tput-history-b"
        historical_a = result_row("screen", 0, 100.0, mbs=2)
        historical_a["job_id"] = "tput-history-a"

        first = throughput_screening_decisions(
            [historical_b, historical_a],
            [request],
            [screen_plan],
            top_k=1,
        )["configurations"][0]
        second = throughput_screening_decisions(
            [historical_a, historical_b],
            [request],
            [screen_plan],
            top_k=1,
        )["configurations"][0]

        self.assertEqual(first["formal_substitute_scope"], "historical")
        self.assertEqual(first["job_ids"], ["tput-history-a"])
        self.assertEqual(first["samples_per_second"], 100.0)
        self.assertEqual(first, second)

    def test_formal_substitute_requires_exact_physical_key(self) -> None:
        request = {"request_id": "screen", "repeats": 1}
        screen_plan = result_row("screen", 0, 0.0, mbs=2)
        screen_plan.update({"job_id": "screen-active", "kind": "throughput_screen"})
        wrong_physical = result_row("screen", 0, 999.0, mbs=4)
        wrong_physical["job_id"] = "tput-wrong-physical"

        candidate = throughput_screening_decisions(
            [wrong_physical],
            [request],
            [screen_plan],
            top_k=1,
        )["configurations"][0]

        self.assertEqual(candidate["status"], "pending")
        self.assertIsNone(candidate["measurement_source"])
        self.assertEqual(candidate["job_ids"], [])

    def test_invalid_current_formal_id_cannot_masquerade_as_history(self) -> None:
        request = {"request_id": "screen", "repeats": 1}
        screen_plan = result_row("screen", 0, 0.0, mbs=4)
        screen_plan.update({"job_id": "screen-active", "kind": "throughput_screen"})
        formal_plan = result_row("screen", 0, 0.0, mbs=2)
        formal_plan["job_id"] = "formal-active"
        corrupted_formal = result_row("screen", 0, 999.0, mbs=4)
        corrupted_formal["job_id"] = "formal-active"

        candidate = throughput_screening_decisions(
            [corrupted_formal],
            [request],
            [screen_plan],
            top_k=1,
            active_formal_jobs=[formal_plan],
        )["configurations"][0]

        self.assertEqual(candidate["status"], "pending")
        self.assertEqual(candidate["job_ids"], [])

    def test_single_planned_run_is_complete_and_selectable(self) -> None:
        request = {"request_id": "single", "repeats": 1}
        report = throughput_decisions([result_row("single", 0, 100.0)], [request])
        selected = report["decisions"][0]["selected"]
        self.assertEqual(selected["planned_repeats"], 1)
        self.assertEqual(selected["successful_runs"], 1)
        self.assertTrue(selected["complete"])

    def test_fast_incomplete_configuration_cannot_win(self) -> None:
        request = {"request_id": "t", "repeats": 3}
        rows = [result_row("t", repeat, 100.0, mbs=1) for repeat in range(3)]
        rows += [result_row("t", repeat, 200.0, mbs=2) for repeat in range(2)]
        report = throughput_decisions(rows, [request])
        selected = report["decisions"][0]["selected"]
        self.assertEqual(selected["mbs"], 1)
        self.assertEqual(selected["successful_runs"], 3)

    def test_formal_decisions_only_accept_exact_current_matrix_jobs(self) -> None:
        request = {"request_id": "active", "repeats": 99}
        planned = [result_row("active", repeat, 0.0, mbs=2) for repeat in range(2)]
        active = [
            result_row("active", 0, 100.0, mbs=2),
            result_row("active", 1, 120.0, mbs=2),
        ]
        stale = result_row("active", 0, 999.0, mbs=2)
        stale["job_id"] = "tput-stale-same-physical"
        profiler = result_row("active", 0, 2_000.0, mbs=2)
        profiler.update({"job_id": "profhold-stale", "kind": "profiler"})
        wrong_physical = dict(active[0])
        wrong_physical.update({"job_id": planned[0]["job_id"], "mbs": 8})
        wrong_repeat = dict(active[0])
        wrong_repeat.update({"job_id": planned[0]["job_id"], "repeat": 7})

        report = throughput_decisions(
            [stale, profiler, wrong_physical, wrong_repeat, *active],
            [request],
            planned,
        )

        self.assertEqual(report["requests"], 1)
        self.assertEqual(report["selected"], 1)
        self.assertEqual(len(report["configurations"]), 1)
        selected = report["decisions"][0]["selected"]
        self.assertEqual(selected["planned_repeats"], 2)
        self.assertEqual(selected["successful_runs"], 2)
        self.assertEqual(selected["samples_per_second"], 110.0)
        self.assertEqual(
            set(selected["job_ids"]),
            {job["job_id"] for job in planned},
        )

    def test_current_matrix_missing_repeat_stays_incomplete(self) -> None:
        request = {"request_id": "active", "repeats": 1}
        planned = [result_row("active", repeat, 0.0, mbs=2) for repeat in range(2)]
        valid = result_row("active", 0, 100.0, mbs=2)
        wrong_physical = result_row("active", 1, 200.0, mbs=8)
        wrong_physical["job_id"] = planned[1]["job_id"]
        wrong_kind = result_row("active", 1, 300.0, mbs=2)
        wrong_kind.update({"job_id": planned[1]["job_id"], "kind": "profiler"})

        report = throughput_decisions(
            [valid, wrong_physical, wrong_kind],
            [request],
            planned,
        )

        config = report["configurations"][0]
        self.assertEqual(config["planned_repeats"], 2)
        self.assertEqual(config["observed_runs"], 1)
        self.assertFalse(config["complete"])
        self.assertIsNone(report["decisions"][0]["selected"])

    def test_planned_mode_reports_only_requests_in_active_matrix(self) -> None:
        requests = [
            {"request_id": "active", "repeats": 1},
            {"request_id": "inactive", "repeats": 1},
        ]
        planned = [result_row("active", 0, 0.0)]
        report = throughput_decisions(
            [result_row("active", 0, 100.0)],
            requests,
            planned,
        )
        self.assertEqual(report["requests"], 1)
        self.assertEqual(
            [decision["request_id"] for decision in report["decisions"]],
            ["active"],
        )

    def test_best_configuration_selects_measured_zero_gc_mbs_tuple(self) -> None:
        requests = [
            {"request_id": "a", "repeats": 3},
            {"request_id": "b", "repeats": 3},
        ]
        rows = [
            result_row("a", repeat, 100.0, zero="zero2", gc=False, mbs=2)
            for repeat in range(3)
        ]
        rows += [
            result_row("b", repeat, 120.0, zero="zero3", gc=True, mbs=4)
            for repeat in range(3)
        ]
        report = best_configuration_decisions(rows, requests)
        winner = report["decisions"][0]["selected"]
        self.assertEqual(
            (winner["zero"], winner["gc"], winner["mbs"]), ("zero3", True, 4)
        )

    def test_best_configuration_uses_only_current_plan_union(self) -> None:
        requests = [
            {"request_id": "formal", "repeats": 99},
            {"request_id": "scaling", "repeats": 99},
        ]
        formal = result_row("formal", 0, 100.0, zero="zero2", mbs=2)
        scaling = result_row("scaling", 0, 120.0, zero="zero3", mbs=4)
        plans = [dict(formal), dict(scaling)]
        stale = result_row("formal", 0, 900.0, zero="none", mbs=8)
        stale["job_id"] = "tput-stale-best"
        profiler = result_row("scaling", 0, 1_000.0, zero="none", mbs=16)
        profiler.update({"job_id": "profhold-stale-best", "kind": "profiler"})

        report = best_configuration_decisions(
            [stale, profiler, formal, scaling],
            requests,
            plans,
        )

        decision = report["decisions"][0]
        self.assertEqual(decision["candidate_configurations"], 2)
        self.assertEqual(
            (decision["selected"]["zero"], decision["selected"]["mbs"]),
            ("zero3", 4),
        )
        self.assertEqual(
            decision["selected"]["job_ids"],
            [scaling["job_id"]],
        )

    def test_build_reports_wires_current_formal_and_scaling_matrices(self) -> None:
        requests = [
            {"request_id": "active", "repeats": 99},
            {"request_id": "inactive", "repeats": 1},
        ]
        active = result_row("active", 0, 100.0, mbs=2)
        active["job_id"] = "tput-active"
        stale = result_row("active", 0, 999.0, mbs=8)
        stale["job_id"] = "tput-stale"
        profiler = result_row("active", 0, 2_000.0, mbs=16)
        profiler.update({"job_id": "profhold-stale", "kind": "profiler"})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "matrix"
            config = root / "config"
            matrix.mkdir()
            config.mkdir()

            def write_jsonl(name: str, rows: list[dict]) -> None:
                (matrix / name).write_text(
                    "".join(f"{json.dumps(row)}\n" for row in rows),
                    encoding="utf-8",
                )

            write_jsonl("throughput_requests.jsonl", requests)
            write_jsonl("throughput_screen_jobs.jsonl", [])
            write_jsonl("throughput_jobs.jsonl", [active])
            write_jsonl("strong_scaling_requests.jsonl", [])
            write_jsonl("scaling_candidate_jobs.jsonl", [])
            write_jsonl("packing_pair_requests.jsonl", [])
            (config / "experiment.json").write_text(
                json.dumps(
                    {
                        "matrix_policy": {"throughput_shortlist_top_k": 1},
                        "scaling_rule": {"minimum_throughput_gain_per_doubling": 0.7},
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("stage_decisions.MATRIX_DIR", matrix),
                patch("stage_decisions.CONFIG_DIR", config),
                patch(
                    "stage_decisions.load_result_rows",
                    return_value=([stale, profiler, active], {}),
                ),
            ):
                report = build_reports(root / "results")

        formal = report["throughput"]
        self.assertEqual((formal["requests"], formal["selected"]), (1, 1))
        self.assertEqual(len(formal["configurations"]), 1)
        self.assertEqual(
            formal["configurations"][0]["job_ids"],
            ["tput-active"],
        )
        best = report["best_configurations"]["decisions"]
        self.assertEqual(len(best), 1)
        self.assertEqual(best[0]["candidate_configurations"], 1)
        self.assertEqual(best[0]["selected"]["job_ids"], ["tput-active"])

    def test_build_reports_accepts_explicit_formal_plan_snapshot(self) -> None:
        request = {"request_id": "active", "repeats": 1}
        current = result_row("active", 0, 80.0, mbs=1)
        current["job_id"] = "tput-current"
        snapshot = result_row("active", 0, 120.0, mbs=2)
        snapshot["job_id"] = "tput-snapshot"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "matrix"
            config = root / "config"
            matrix.mkdir()
            config.mkdir()

            def write_jsonl(name: str, rows: list[dict]) -> Path:
                path = matrix / name
                path.write_text(
                    "".join(f"{json.dumps(row)}\n" for row in rows),
                    encoding="utf-8",
                )
                return path

            write_jsonl("throughput_requests.jsonl", [request])
            write_jsonl("throughput_screen_jobs.jsonl", [])
            write_jsonl("throughput_jobs.jsonl", [current])
            snapshot_path = write_jsonl("formal-snapshot.jsonl", [snapshot])
            write_jsonl("strong_scaling_requests.jsonl", [])
            write_jsonl("scaling_candidate_jobs.jsonl", [])
            write_jsonl("packing_pair_requests.jsonl", [])
            (config / "experiment.json").write_text(
                json.dumps(
                    {
                        "matrix_policy": {"throughput_shortlist_top_k": 1},
                        "scaling_rule": {"minimum_throughput_gain_per_doubling": 0.7},
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("stage_decisions.MATRIX_DIR", matrix),
                patch("stage_decisions.CONFIG_DIR", config),
                patch(
                    "stage_decisions.load_result_rows",
                    return_value=([current, snapshot], {}),
                ),
            ):
                report = build_reports(
                    root / "results", throughput_jobs_path=snapshot_path
                )

        self.assertEqual(
            report["throughput"]["configurations"][0]["job_ids"],
            ["tput-snapshot"],
        )

    def test_scaling_stops_before_four_gpus_after_failed_gain(self) -> None:
        request = {"request_id": "s", "repeats": 3, "gpu_sequence": [1, 2, 4]}
        rows = []
        for gpu_count, rate in ((1, 100.0), (2, 160.0), (4, 400.0)):
            rows += [
                result_row("s", repeat, rate, gpu_count=gpu_count)
                for repeat in range(3)
            ]
        report = scaling_decisions(rows, [request], minimum_gain=0.70)
        family = report["families"][0]
        self.assertTrue(family["stopped_early"])
        self.assertEqual(family["points"][1]["gain_from_previous"], 0.60)
        self.assertEqual(family["points"][2]["status"], "skipped_by_early_stop")
        self.assertEqual(family["recommended"]["gpu_count"], 1)
        self.assertNotIn("s", scaling_eligible_request_ids(report, 4))

    def test_scaling_continues_after_passing_gain(self) -> None:
        request = {"request_id": "s", "repeats": 3, "gpu_sequence": [1, 2, 4]}
        rows = []
        for gpu_count, rate in ((1, 100.0), (2, 180.0)):
            rows += [
                result_row("s", repeat, rate, gpu_count=gpu_count)
                for repeat in range(3)
            ]
        report = scaling_decisions(rows, [request], minimum_gain=0.70)
        self.assertIn("s", scaling_eligible_request_ids(report, 4))

    def test_packing_uses_worst_paired_gain_and_mbs_threshold(self) -> None:
        requests = [
            {"request_id": "p1", "repeats": 3, "expected_gbs_relative_error": 0.01},
            {"request_id": "p2", "repeats": 3, "expected_gbs_relative_error": 0.01},
        ]
        rows = []
        for request_id, mbs, gains in (
            ("p1", 1, (0.12, 0.11, 0.15)),
            ("p2", 2, (0.21, 0.19, 0.25)),
        ):
            for repeat, gain in enumerate(gains):
                rows.append(
                    result_row(request_id, repeat, 100.0, mbs=mbs, packing=False)
                )
                rows.append(
                    result_row(
                        request_id, repeat, 100.0 / (1.0 - gain), mbs=1, packing=True
                    )
                )
        report = packing_decisions(rows, requests)
        decisions = {row["request_id"]: row for row in report["decisions"]}
        self.assertEqual(decisions["p1"]["decision"], "on")
        self.assertAlmostEqual(
            decisions["p1"]["conservative_time_gain_lower_bound"], 0.11
        )
        self.assertEqual(decisions["p2"]["decision_threshold"], 0.20)
        self.assertEqual(decisions["p2"]["decision"], "off")

    def test_single_packing_pair_uses_its_measured_gain(self) -> None:
        request = {
            "request_id": "single-pair",
            "repeats": 1,
            "expected_gbs_relative_error": 0.01,
        }
        rows = [
            result_row("single-pair", 0, 100.0, mbs=1, packing=False),
            result_row("single-pair", 0, 125.0, mbs=1, packing=True),
        ]
        decision = packing_decisions(rows, [request])["decisions"][0]
        self.assertEqual(decision["complete_pairs"], 1)
        self.assertEqual(decision["lower_bound_method"], "single paired time gain")
        self.assertEqual(decision["decision"], "on")

    def test_profiler_fit_uses_independent_holdout(self) -> None:
        points = []
        for train_type in ("full", "lora"):
            for gc in (False, True):
                ratio = (
                    1.10
                    * (1.20 if train_type == "lora" else 1.0)
                    * (1.30 if gc else 1.0)
                )
                points.append(
                    {
                        "job_id": f"cal-{train_type}-{gc}",
                        "role": "calibration",
                        "train_type": train_type,
                        "gc": gc,
                        "analytic_flops": 1000.0,
                        "observed_profiler_flops": 1000.0 * ratio,
                        "observed_to_analytic_ratio": ratio,
                    }
                )
        points.append(
            {
                "job_id": "holdout",
                "role": "holdout",
                "train_type": "lora",
                "gc": True,
                "analytic_flops": 5000.0,
                "observed_profiler_flops": 5000.0 * 1.10 * 1.20 * 1.30,
                "observed_to_analytic_ratio": 1.10 * 1.20 * 1.30,
            }
        )
        report = fit_profiler_calibration(points)
        self.assertEqual(report["status"], "evaluated")
        self.assertEqual(report["evaluation_method"], "independent_holdout")
        self.assertLess(report["evaluation_mape"], 1.0e-5)
        self.assertTrue(
            math.isclose(report["multipliers"]["lora_gc_on"], 1.716, rel_tol=1.0e-5)
        )

    def test_profiler_fit_promotes_minimum_rank_improving_holdout(self) -> None:
        points = [
            {
                "job_id": "cal-lora-off",
                "role": "calibration",
                "train_type": "lora",
                "gc": False,
                "analytic_flops": 1000.0,
                "observed_profiler_flops": 1200.0,
                "observed_to_analytic_ratio": 1.2,
            },
            {
                "job_id": "cal-lora-on",
                "role": "calibration",
                "train_type": "lora",
                "gc": True,
                "analytic_flops": 1000.0,
                "observed_profiler_flops": 1320.0,
                "observed_to_analytic_ratio": 1.32,
            },
            {
                "job_id": "holdout-full",
                "role": "holdout",
                "train_type": "full",
                "gc": False,
                "analytic_flops": 2000.0,
                "observed_profiler_flops": 2000.0,
                "observed_to_analytic_ratio": 1.0,
            },
            {
                "job_id": "holdout-lora",
                "role": "holdout",
                "train_type": "lora",
                "gc": False,
                "analytic_flops": 5000.0,
                "observed_profiler_flops": 6000.0,
                "observed_to_analytic_ratio": 1.2,
            },
        ]

        report = fit_profiler_calibration(points)

        self.assertEqual(report["status"], "evaluated")
        self.assertEqual(
            report["fallback_calibration_job_ids"],
            ["holdout-full"],
        )
        self.assertEqual(report["evaluation_method"], "independent_holdout")
        self.assertEqual(
            [point["job_id"] for point in report["evaluation_points"]],
            ["holdout-lora"],
        )

    def test_profiler_point_reads_active_steps_and_operator_flops(self) -> None:
        model = {
            "actual_parameters": 1000,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
        }
        row = {
            **result_row("prof", 0, 1.0),
            "job_id": "prof-test",
            "classification": "success",
            "enable_profiler": True,
            "profiler_active_steps": 3,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "prof-test"
            (result / "metrics" / "profiler").mkdir(parents=True)
            (result / "metrics" / "profiler" / "operators.rank0.json").write_text(
                json.dumps([{"flops": 1000}, {"flops": 2000}])
            )
            with (result / "metrics" / "events.rank0.jsonl").open("w") as output:
                for step in range(1, 5):
                    output.write(
                        json.dumps(
                            {
                                "event": "step_end",
                                "global_step": step,
                                "tokens": {
                                    "computed_tokens": 10,
                                    "effective_tokens": 9,
                                    "computed_attention_token_pairs": 100,
                                    "effective_attention_token_pairs": 90,
                                },
                            }
                        )
                        + "\n"
                    )
            points = profiler_points([row], {"qwen3_8b": model}, Path(directory))
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["computed_tokens"], 30)
        self.assertEqual(points[0]["observed_profiler_flops"], 3000)

    def test_resource_models_use_group_holdout(self) -> None:
        models = {
            "qwen3_8b": {"actual_parameters": 8_000_000_000},
            "qwen3_14b": {"actual_parameters": 14_000_000_000},
        }
        rows = []
        for scenario in range(4):
            model_id = "qwen3_8b" if scenario < 2 else "qwen3_14b"
            for mbs in (1, 2, 4, 8):
                row = result_row(
                    f"resource-{scenario}", 0, (scenario + 1) * mbs * 10.0, mbs=mbs
                )
                row.update(
                    {
                        "model_id": model_id,
                        "dataset_id": f"data-{scenario}",
                        "target_gbs": 64,
                        "max_reserved_bytes": float((scenario + 2) * mbs * 1_000_000),
                    }
                )
                rows.append(row)
        report = group_holdout_resource_evaluation(rows, models)
        self.assertEqual(report["status"], "evaluated")
        self.assertEqual(len(report["folds"]), 4)
        self.assertIsNotNone(report["throughput_mape"])


if __name__ == "__main__":
    unittest.main()
