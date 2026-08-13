from __future__ import annotations

import csv
import asyncio
import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from dashboard.app import create_app, job_series
from dashboard.analytics import DashboardAnalytics
from dashboard.ingest import DashboardIngestor, MultiCampaignIngestor
from dashboard.store import DashboardStore


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class DashboardIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for directory in ("matrix", "runtime/jobs", "results", "artifacts", "config"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        write_json(
            self.root / "artifacts/model_inventory.json",
            {
                "models": [
                    {
                        "id": "qwen-test", "hidden_size": 64, "intermediate_size": 128,
                        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                        "vocab_size": 1024,
                    }
                ]
            },
        )
        write_json(
            self.root / "config/hardware.json",
            {
                "bf16_dense_peak_flops_per_second_for_standard_mfu": 1000e12,
                "clock_normalized_dense_peak_flops_per_second": 900e12,
                "max_sm_clock_mhz_reported_by_nvidia_smi": 1980,
                "healthy_busy_sm_clock_mhz": 1800,
                "power_limit_w": 700,
            },
        )
        write_json(
            self.root / "config/experiment.json",
            {
                "training_scope": {
                    "stage": "sft",
                    "gpu_ids": [1, 2, 3, 4],
                    "max_gpu_count": 4,
                },
                "measurement": {
                    "performance_parallelism": "disjoint_gpu_masks"
                },
            },
        )
        write_json(
            self.root / "matrix/design_summary.json",
            {
                "memory_boundary_families": 1,
                "formal_repeats_per_configuration": 1,
                "throughput_runs_after_repeats": 1,
            },
        )
        family = {
            "job_id": "mem-family", "kind": "memory_boundary", "model_id": "qwen-test",
            "train_type": "lora", "dataset_id": "short", "cutoff_len": 512, "gpu_count": 1,
            "zero": "none", "gc": False, "target_gbs": 16, "mbs_candidates": [1, 2],
        }
        write_jsonl(self.root / "matrix/memory_boundary_families.jsonl", [family])
        job = {
            "job_id": "tput-test", "kind": "throughput", "request_id": "request-test",
            "model_id": "qwen-test", "model_family": "qwen", "train_type": "lora",
            "dataset_id": "short", "cutoff_len": 512, "gpu_count": 1, "zero": "none",
            "gc": False, "mbs": 2, "target_gbs": 16, "packing": False, "repeat": 0,
            "warmup_steps": 1, "measure_steps": 2, "max_steps": 3,
        }
        write_jsonl(self.root / "matrix/throughput_jobs.jsonl", [job])
        write_json(self.root / "runtime/jobs/tput-test.json", job)
        result = self.root / "results/tput-test"
        write_json(result / "rendered_run.json", {"job": job, "gpu_mask": "0", "command": ["python"]})
        write_json(
            result / "status.json",
            {
                "job_id": "tput-test", "classification": "success", "return_code": 0,
                "started_unix": 1000, "finished_unix": 1010, "wall_seconds": 10, "gpu_mask": "0",
            },
        )
        events = []
        for step in (1, 2, 3):
            events.append(
                {
                    "time_unix": 1000 + step, "event": "step_end", "rank": 0, "global_step": step,
                    "step_seconds": 2.0, "optimizer_step_seconds": 0.1,
                    "micro_step_seconds": [1.0, 1.0], "is_warmup": step == 1,
                    "tokens": {
                        "computed_tokens": 200, "effective_tokens": 160, "label_tokens": 80,
                        "logical_samples": 16, "physical_batches": 2,
                        "computed_attention_token_pairs": 20_000,
                        "effective_attention_token_pairs": 15_000,
                    },
                    "memory": {
                        "allocated": 1024**3, "reserved": 2 * 1024**3,
                        "max_allocated": 3 * 1024**3, "max_reserved": 4 * 1024**3,
                    },
                }
            )
        write_jsonl(result / "metrics/events.rank0.jsonl", events)
        (result / "metrics").mkdir(parents=True, exist_ok=True)
        with (result / "nvidia_smi.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["timestamp", "index", "memory.used", "utilization.gpu", "power.draw", "clocks.sm"])
            writer.writerow(["2026/07/16 09:00:01.000", 0, 3072, 98, 600, 1800])
            writer.writerow(["2026/07/16 09:00:02.000", 0, 4096, 99, 610, 1815])
        write_json(
            self.root / "results/boundary_summaries/mem-family.json",
            {
                "family_job_id": "mem-family", "gpu_mask": [0],
                "trials": [{"mbs": 1, "classification": "success"}, {"mbs": 2, "classification": "oom"}],
                "max_feasible_mbs": 1, "first_failed_mbs": 2,
            },
        )
        write_jsonl(
            self.root / "runtime/scheduler_events.jsonl",
            [
                {"time_unix": 1000, "event": "trial_start", "job_id": "tput-test", "gpu_mask": [0]},
                {"time_unix": 1010, "event": "trial_end", "job_id": "tput-test", "classification": "success", "return_code": 0},
            ],
        )

    def activate_queue(
        self,
        rows: list[dict],
        *,
        queue_relative: str = "runtime/pipeline/approved-queue.jsonl",
    ) -> tuple[Path, dict, dict]:
        rows = [
            {**row, "gpu_count": int(row.get("gpu_count") or 1)} for row in rows
        ]
        queue_path = (self.root / queue_relative).resolve()
        write_jsonl(queue_path, rows)
        job_ids = [str(row["job_id"]) for row in rows]
        payload_hashes = [sha256_json(row) for row in rows]
        binding = {
            "schema_version": 1,
            "path": queue_relative,
            "sha256": sha256_file(queue_path),
            "ordered_job_ids": job_ids,
            "ordered_job_payload_sha256": payload_hashes,
            "job_payload_sha256": dict(
                zip(job_ids, payload_hashes, strict=True)
            ),
        }
        provenance_path = self.root / "artifacts/provenance.json"
        write_json(provenance_path, {"fixture": "dashboard-display-binding"})
        runtime_identity = {"fixture": "runtime-identity-v1"}
        runtime_patch = {"all_passed": True, "fixture": "runtime-patch-v1"}
        execution_order = ["h800_calibration"]
        design = {
            "schema_version": 1,
            "training_started": False,
            "allowed_job_ids": job_ids,
            "execution_order": execution_order,
            "authorized_gpu_ids": [1, 2, 3, 4],
            "max_gpu_count": 4,
            "runtime_identity": runtime_identity,
            "runtime_fingerprint_sha256": sha256_json(runtime_identity),
            "runtime_patch": runtime_patch,
            "provenance_binding": {
                "path": "artifacts/provenance.json",
                "sha256": sha256_file(provenance_path),
            },
            "file_sha256": {
                queue_relative: binding["sha256"],
                "artifacts/provenance.json": sha256_file(provenance_path),
                "config/experiment.json": sha256_file(
                    self.root / "config/experiment.json"
                ),
            },
            "queue_binding": binding,
        }
        design_path = self.root / "runtime/approval_design.json"
        write_json(design_path, design)
        approval = {
            "schema_version": 2,
            "approved": True,
            "design_sha256": sha256_file(design_path),
            "allowed_job_ids": job_ids,
            "execution_order": execution_order,
            "phase_id": execution_order[0],
            "queue_binding_sha256": sha256_json(binding),
            "runtime_fingerprint_sha256": design[
                "runtime_fingerprint_sha256"
            ],
            "runtime_patch_sha256": sha256_json(runtime_patch),
            "provenance_sha256": design["provenance_binding"]["sha256"],
            "resource_scope": {
                "gpu_ids": [1, 2, 3, 4],
                "max_gpu_count": 4,
                "allow_gpu_ids_outside_pool": False,
                "performance_parallelism": "disjoint_gpu_masks",
            },
        }
        write_json(self.root / "config/APPROVED_TO_RUN.json", approval)
        return queue_path, design, approval

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_multi_campaign_ingestor_discovers_nested_isolated_campaigns(
        self,
    ) -> None:
        immediate = self.root / "campaigns" / "immediate"
        nested = self.root / "campaigns" / "prospective" / "model-a"
        for campaign_root, campaign_id in (
            (immediate, "immediate-campaign"),
            (nested, "nested-model-campaign"),
        ):
            write_json(
                campaign_root / "config" / "experiment.json",
                {
                    "campaign_id": campaign_id,
                    "hardware_id": "local_rtx4090_24g_pcie",
                    "training_scope": {
                        "gpu_ids": [0, 1, 2, 3],
                        "gpu_type": "NVIDIA GeForce RTX 4090 24GB",
                    },
                },
            )
        write_jsonl(
            nested / "matrix" / "queue_compatibility_canary.jsonl",
            [
                {
                    "job_id": "nested-canary",
                    "campaign_id": "prospective-parent",
                    "kind": "throughput_screen",
                    "model_id": "qwen-test",
                    "gpu_count": 1,
                }
            ],
        )
        store = DashboardStore(
            self.root / "runtime" / "dashboard" / "nested.sqlite"
        )
        ingestor = MultiCampaignIngestor(store, root=self.root)
        campaign_ids = {
            item.campaign_id for item in ingestor.ingestors
        }
        self.assertIn("immediate-campaign", campaign_ids)
        self.assertIn("nested-model-campaign", campaign_ids)
        ingestor.scan_once()
        nested_job = store.get_job("nested-canary")
        self.assertEqual(
            nested_job["campaign_id"],
            "nested-model-campaign",
        )
        store.close()

    def test_ingestion_and_derived_metrics(self) -> None:
        store = DashboardStore(self.root / "runtime/dashboard/test.sqlite")
        ingestor = DashboardIngestor(store, root=self.root)
        result = ingestor.scan_once()
        self.assertTrue(result["changed"])
        job = store.get_job("tput-test")
        self.assertEqual(job["status"], "success")
        self.assertEqual(job["current_step"], 3)
        self.assertAlmostEqual(job["metrics"]["samples_per_second"], 8.0)
        self.assertAlmostEqual(job["metrics"]["padding_efficiency"], 0.8)
        self.assertEqual(job["metrics"]["nvidia_smi_peak_mib"], 4096)
        self.assertEqual(job["metrics"]["clock_status"], "insufficient_data")
        family = store.get_job("mem-family")
        self.assertEqual(family["status"], "boundary_found")
        self.assertEqual(family["boundary"]["max_feasible_mbs"], 1)
        second = ingestor.scan_once()
        self.assertEqual(second["dirty_jobs"], 0)
        store.close()

    def test_formal_manifest_only_uses_current_throughput_matrix(self) -> None:
        write_jsonl(
            self.root / "matrix/scaling_candidate_jobs.jsonl",
            [
                {
                    "job_id": "scale-formal-lookalike",
                    "kind": "throughput",
                    "model_id": "qwen-test",
                }
            ],
        )
        rows = [
            json.loads(line)
            for line in (self.root / "matrix/throughput_jobs.jsonl").read_text().splitlines()
        ]
        rows.append(
            {
                "job_id": "non-formal-in-formal-file",
                "kind": "other",
                "model_id": "qwen-test",
            }
        )
        write_jsonl(self.root / "matrix/throughput_jobs.jsonl", rows)

        store = DashboardStore(self.root / "runtime/dashboard/formal-manifest.sqlite")
        DashboardIngestor(store, root=self.root).scan_once()

        campaign_id = str(
            json.loads(
                (self.root / "config/experiment.json").read_text(encoding="utf-8")
            ).get("campaign_id")
            or self.root.name
        )
        self.assertEqual(
            store.get_meta(
                f"campaign:{campaign_id}:active_throughput_formal_job_ids",
                None,
            ),
            ["tput-test"],
        )
        store.close()

    def test_bound_live_queue_replaces_matrix_progress_and_is_rescanned_immediately(
        self,
    ) -> None:
        write_json(
            self.root / "matrix/design_summary.json",
            {
                "memory_boundary_families": 999,
                "packing_pair_families": 777,
            },
        )
        unapproved = {
            "job_id": "artifact-only",
            "boundary_probe": {"condition": {"type": "always"}},
        }
        write_jsonl(
            self.root / "artifacts/candidates/unapproved.jobs.jsonl",
            [unapproved],
        )
        rows = [
            {
                "job_id": "approved-boundary",
                "kind": "throughput_screen",
                "boundary_probe": {"condition": {"type": "always"}},
                "model_id": "qwen-test",
                "train_type": "lora",
                "cutoff_len": 512,
                "gpu_count": 1,
                "mbs": 4,
            },
            {
                "job_id": "approved-packing",
                "kind": "throughput",
                "packing_pair": {"condition": {"type": "always"}},
                "model_id": "qwen-test",
                "train_type": "lora",
                "cutoff_len": 512,
                "gpu_count": 1,
                "mbs": 1,
                "packing": True,
            },
        ]

        store = DashboardStore(self.root / "runtime/dashboard/live-queue.sqlite")
        ingestor = DashboardIngestor(store, root=self.root)
        store.set_meta(
            "campaigns",
            [{"campaign_id": ingestor.campaign_id, "gpu_ids": [1, 2, 3, 4]}],
        )
        ingestor.scan_once()
        self.assertIsNotNone(store.get_job("mem-family"))

        self.activate_queue(rows)
        ingestor._last_registry_scan = time.time()
        ingestor.scan_once()

        self.assertEqual(store.get_job("approved-boundary")["phase"], "memory")
        self.assertEqual(store.get_job("approved-packing")["phase"], "packing")
        self.assertIsNone(store.get_job("artifact-only"))
        # Historical terminal evidence remains queryable, but is filtered out
        # of the current approved-manifest denominator below.
        self.assertEqual(store.get_job("mem-family")["status"], "boundary_found")
        overview = DashboardAnalytics(store).overview(ingestor.campaign_id)
        self.assertEqual(
            overview["active_manifest_job_ids"][ingestor.campaign_id],
            ["approved-boundary", "approved-packing"],
        )
        self.assertEqual(
            overview["phase_progress"]["memory"],
            {"planned": 1, "completed": 0},
        )
        self.assertEqual(
            overview["phase_progress"]["packing"],
            {"planned": 1, "completed": 0},
        )
        self.assertEqual(overview["runs_by_status"], {"planned": 2})
        validation = store.get_meta(
            f"campaign:{ingestor.campaign_id}:active_queue_validation", {}
        )
        self.assertEqual(validation["mode"], "approved_queue")
        self.assertTrue(all(validation["checks"].values()))
        store.close()

    def test_invalid_live_queue_binding_fails_closed_instead_of_using_matrix(
        self,
    ) -> None:
        row = {
            "job_id": "approved-only",
            "kind": "throughput_screen",
            "boundary_probe": {"condition": {"type": "always"}},
        }

        def approval_false() -> None:
            _, _, approval = self.activate_queue([row])
            approval["approved"] = False
            write_json(self.root / "config/APPROVED_TO_RUN.json", approval)

        def wrong_design_hash() -> None:
            self.activate_queue([row])
            approval_path = self.root / "config/APPROVED_TO_RUN.json"
            approval = json.loads(approval_path.read_text(encoding="utf-8"))
            approval["design_sha256"] = "0" * 64
            write_json(approval_path, approval)

        def mismatched_execution_order() -> None:
            self.activate_queue([row])
            approval_path = self.root / "config/APPROVED_TO_RUN.json"
            approval = json.loads(approval_path.read_text(encoding="utf-8"))
            approval["execution_order"] = ["different-stage"]
            write_json(approval_path, approval)

        def stale_runtime_binding() -> None:
            self.activate_queue([row])
            approval_path = self.root / "config/APPROVED_TO_RUN.json"
            approval = json.loads(approval_path.read_text(encoding="utf-8"))
            approval["runtime_fingerprint_sha256"] = "1" * 64
            write_json(approval_path, approval)

        def mismatched_resource_scope() -> None:
            self.activate_queue([row])
            approval_path = self.root / "config/APPROVED_TO_RUN.json"
            approval = json.loads(approval_path.read_text(encoding="utf-8"))
            approval["resource_scope"]["gpu_ids"] = [0]
            write_json(approval_path, approval)

        def stale_provenance_binding() -> None:
            self.activate_queue([row])
            approval_path = self.root / "config/APPROVED_TO_RUN.json"
            approval = json.loads(approval_path.read_text(encoding="utf-8"))
            approval["provenance_sha256"] = "2" * 64
            write_json(approval_path, approval)

        def tampered_queue_bytes() -> None:
            queue_path, _, _ = self.activate_queue([row])
            with queue_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps({**row, "job_id": "injected"}) + "\n")

        def tampered_order_and_payload_binding() -> None:
            _, design, approval = self.activate_queue(
                [row, {**row, "job_id": "approved-second"}]
            )
            binding = design["queue_binding"]
            binding["ordered_job_ids"] = list(
                reversed(binding["ordered_job_ids"])
            )
            binding["ordered_job_payload_sha256"][0] = "f" * 64
            design_path = self.root / "runtime/approval_design.json"
            write_json(design_path, design)
            approval["design_sha256"] = sha256_file(design_path)
            approval["queue_binding_sha256"] = sha256_json(binding)
            write_json(self.root / "config/APPROVED_TO_RUN.json", approval)

        outside_path = self.root.parent / f"{self.root.name}-outside-queue.jsonl"

        def outside_queue_path() -> None:
            relative = f"../{outside_path.name}"
            self.activate_queue([row], queue_relative=relative)

        self.addCleanup(lambda: outside_path.unlink(missing_ok=True))
        mutations = (
            approval_false,
            wrong_design_hash,
            mismatched_execution_order,
            stale_runtime_binding,
            mismatched_resource_scope,
            stale_provenance_binding,
            tampered_queue_bytes,
            tampered_order_and_payload_binding,
            outside_queue_path,
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=mutation.__name__):
                mutation()
                store = DashboardStore(
                    self.root / f"runtime/dashboard/invalid-{index}.sqlite"
                )
                ingestor = DashboardIngestor(store, root=self.root)
                ingestor._scan_registry()
                self.assertIsNone(store.get_job("approved-only"))
                self.assertIsNone(store.get_job("mem-family"))
                self.assertEqual(
                    store.get_meta(
                        f"campaign:{ingestor.campaign_id}:active_manifest_job_ids",
                        None,
                    ),
                    [],
                )
                validation = store.get_meta(
                    f"campaign:{ingestor.campaign_id}:active_queue_validation",
                    {},
                )
                self.assertEqual(validation["mode"], "invalid_approval")
                self.assertFalse(all(validation.get("checks", {}).values()))
                store.close()

    def test_conditional_skips_are_terminal_non_training_and_halt_is_not_running(
        self,
    ) -> None:
        boundary = {
            "job_id": "skip-boundary",
            "kind": "throughput_screen",
            "boundary_probe": {"condition": {"type": "always"}},
            "model_id": "qwen-test",
            "train_type": "lora",
            "dataset_id": "short",
            "cutoff_len": 512,
            "gpu_count": 1,
            "mbs": 8,
        }
        packing = {
            "job_id": "skip-packing",
            "kind": "throughput",
            "packing_pair": {"condition": {"type": "always"}},
            "model_id": "qwen-test",
            "train_type": "lora",
            "dataset_id": "short",
            "cutoff_len": 512,
            "gpu_count": 1,
            "mbs": 1,
            "packing": True,
        }
        self.activate_queue([boundary, packing])

        def terminal(job_id: str, timestamp: float) -> dict:
            return {
                "schema": "h800_adaptive_scheduler_terminal/v1",
                "job_id": job_id,
                "scheduler_execution_id": "scheduler-test",
                "classification": "conditional_skipped",
                "terminal": True,
                "training_started": False,
                "calibration_observation_eligible": False,
                "reason": "prerequisite did not match",
                "time_unix": timestamp,
                "training_status_json_was_not_written": True,
            }

        boundary_terminal = terminal("skip-boundary", 1001)
        noise = [
            {"time_unix": 1100 + index, "event": "pool_wait"}
            for index in range(300)
        ]
        write_jsonl(
            self.root / "runtime/scheduler_events.jsonl",
            [
                {"time_unix": 1000, "event": "scheduler_start"},
                {"event": "family_skipped", **boundary_terminal},
                *noise,
                {
                    "time_unix": 1500,
                    "event": "campaign_budget_exhausted",
                    "new_launches_disabled": True,
                },
                {"time_unix": 1501, "event": "scheduler_halted", "pending": 0},
            ],
        )
        # Both skips are older than the cold scheduler tail. The authoritative
        # scheduler-only terminal files must reconstruct them without status.json.
        write_json(
            self.root / "results/skip-boundary/scheduler_terminal.json",
            boundary_terminal,
        )
        write_json(
            self.root / "results/skip-packing/scheduler_terminal.json",
            terminal("skip-packing", 1002),
        )

        store = DashboardStore(self.root / "runtime/dashboard/skips.sqlite")
        ingestor = DashboardIngestor(store, root=self.root)
        store.set_meta(
            "campaigns", [{"campaign_id": ingestor.campaign_id}]
        )
        ingestor.scan_once()

        for job_id in ("skip-boundary", "skip-packing"):
            job = store.get_job(job_id)
            self.assertEqual(job["status"], "conditional_skipped")
            self.assertEqual(job["current_step"], 0)
            self.assertEqual(job["metrics"], {})
        scheduler = store.get_meta(
            f"campaign:{ingestor.campaign_id}:scheduler_state", {}
        )
        self.assertEqual(scheduler["status"], "halted")
        overview = DashboardAnalytics(store).overview(ingestor.campaign_id)
        self.assertEqual(
            overview["runs_by_status"], {"conditional_skipped": 2}
        )
        self.assertEqual(
            overview["phase_progress"]["memory"],
            {"planned": 1, "completed": 1},
        )
        self.assertEqual(
            overview["phase_progress"]["packing"],
            {"planned": 1, "completed": 1},
        )
        self.assertTrue(overview["active_manifest_complete"])
        self.assertEqual(overview["current_phase_label"], "全部实验完成")
        configs = DashboardAnalytics(store).throughput(
            "packing", ingestor.campaign_id
        )["configurations"]
        self.assertEqual(configs[0]["successful_runs"], 0)
        self.assertEqual(configs[0]["failed_runs"], 0)
        self.assertEqual(configs[0]["skipped_runs"], 1)
        store.close()

    def test_forged_scheduler_terminal_is_ignored(self) -> None:
        row = {
            "job_id": "forged-skip",
            "kind": "throughput_screen",
            "boundary_probe": {"condition": {"type": "always"}},
        }
        self.activate_queue([row])
        write_json(
            self.root / "results/forged-skip/scheduler_terminal.json",
            {
                "schema": "h800_adaptive_scheduler_terminal/v1",
                "job_id": "different-job",
                "classification": "conditional_skipped",
                "terminal": True,
                "training_started": False,
                "calibration_observation_eligible": True,
                "training_status_json_was_not_written": True,
            },
        )
        store = DashboardStore(self.root / "runtime/dashboard/forged-skip.sqlite")
        ingestor = DashboardIngestor(store, root=self.root)
        ingestor.scan_once()
        self.assertEqual(store.get_job("forged-skip")["status"], "planned")
        self.assertIsNone(store.get_job("different-job"))
        store.close()

    def test_extended_gpu_telemetry_is_classified_and_retained(self) -> None:
        path = self.root / "results/tput-test/nvidia_smi.csv"
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(
                [
                    "timestamp", "index", "memory.used", "utilization.gpu",
                    "power.draw", "clocks.sm", "temperature.gpu", "fan.speed",
                    "clocks_event_reasons.sw_thermal_slowdown",
                    "clocks_event_reasons.hw_thermal_slowdown",
                    "clocks_event_reasons.sw_power_cap",
                ]
            )
            writer.writerows(
                [
                    ["2026/07/16 09:00:01.000", 0, 3072, 98, 690, 1500, 74, 70, "Not Active", "Not Active", "Active"],
                    ["2026/07/16 09:00:02.000", 0, 4096, 99, 695, 1560, 77, 75, "Not Active", "Not Active", "Active"],
                    ["2026/07/16 09:00:03.000", 0, 4096, 100, 700, 1600, 79, 80, "Not Active", "Not Active", "Active"],
                ]
            )

        store = DashboardStore(self.root / "runtime/dashboard/extended.sqlite")
        ingestor = DashboardIngestor(store, root=self.root)
        ingestor.scan_once()

        metrics = store.get_job("tput-test")["metrics"]
        self.assertEqual(metrics["busy_gpu_samples"], 3)
        self.assertEqual(metrics["busy_clock_p50_mhz"], 1560)
        self.assertAlmostEqual(metrics["busy_clock_p50_ratio"], 1560 / 1800)
        self.assertAlmostEqual(
            metrics["busy_clock_p50_to_spec_max_ratio"],
            1560 / 1980,
        )
        self.assertEqual(metrics["power_limit_busy_fraction"], 1.0)
        self.assertEqual(metrics["sw_power_cap_busy_fraction"], 1.0)
        self.assertEqual(metrics["clock_status"], "power_limited")
        self.assertGreater(metrics["clock_adjusted_mfu"], metrics["mfu"])
        samples = store.get_gpu_samples("tput-test")
        self.assertEqual(samples[-1]["temperature_gpu_c"], 79)
        self.assertEqual(samples[-1]["sw_power_cap_active"], 1)
        evidence = job_series(store, "tput-test")["limiter_evidence"]
        self.assertEqual(evidence["busy_samples"], 3)
        self.assertTrue(evidence["reason_coverage_complete"])
        self.assertEqual(
            evidence["flags"]["sw_power_cap_active"]["active_samples"],
            3,
        )
        self.assertEqual(evidence["per_gpu"][0]["clock_status"], "power_limited")
        store.close()

    def test_pipeline_completion_refreshes_without_waiting_for_registry_scan(self) -> None:
        state_path = self.root / "runtime/pipeline/state.json"
        write_json(
            state_path,
            {
                "current": {
                    "stage": "profiler-holdout",
                    "status": "running",
                }
            },
        )
        holdout = {
            "job_id": "profhold-test",
            "kind": "profiler",
            "model_id": "qwen-test",
            "train_type": "lora",
            "dataset_id": "short",
            "cutoff_len": 512,
            "gpu_count": 1,
            "zero": "none",
            "gc": False,
            "mbs": 1,
            "target_gbs": 8,
        }
        write_jsonl(
            self.root / "matrix/profiler_holdout_jobs.jsonl",
            [holdout],
        )
        store = DashboardStore(self.root / "runtime/dashboard/pipeline.sqlite")
        ingestor = DashboardIngestor(store, root=self.root)
        store.set_meta(
            "campaigns",
            [{"campaign_id": ingestor.campaign_id, "gpu_ids": [1, 2, 3, 4]}],
        )
        ingestor.scan_once()
        self.assertEqual(store.get_job("profhold-test")["phase"], "profiler")

        write_json(
            state_path,
            {
                "current": {
                    "stage": "analysis",
                    "status": "complete",
                    "profiler_status": "evaluated",
                }
            },
        )
        ingestor._last_registry_scan = time.time()
        ingestor.scan_once()

        pipeline = store.get_meta(
            f"campaign:{ingestor.campaign_id}:pipeline_state",
            {},
        )
        self.assertEqual(pipeline["current"]["stage"], "analysis")
        overview = DashboardAnalytics(store).overview(ingestor.campaign_id)
        self.assertIsNone(overview["current_phase"])
        self.assertEqual(overview["current_phase_label"], "全部实验完成")
        store.close()

    def test_retry_shows_only_latest_attempt_steps(self) -> None:
        store = DashboardStore(self.root / "runtime/dashboard/retry.sqlite")
        ingestor = DashboardIngestor(store, root=self.root)
        ingestor.scan_once()
        self.assertEqual(store.get_job("tput-test")["current_step"], 3)

        with (self.root / "runtime/scheduler_events.jsonl").open(
            "a", encoding="utf-8"
        ) as output:
            output.write(
                json.dumps(
                    {
                        "time_unix": 2000,
                        "event": "trial_start",
                        "job_id": "tput-test",
                        "gpu_mask": [0],
                    }
                )
                + "\n"
            )
        with (
            self.root / "results/tput-test/metrics/events.rank0.jsonl"
        ).open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    {
                        "time_unix": 2001,
                        "event": "train_begin",
                        "rank": 0,
                    }
                )
                + "\n"
            )
            output.write(
                json.dumps(
                    {
                        "time_unix": 2002,
                        "event": "step_end",
                        "rank": 0,
                        "global_step": 1,
                        "step_seconds": 1.0,
                        "is_warmup": True,
                        "tokens": {
                            "computed_tokens": 100,
                            "effective_tokens": 80,
                            "logical_samples": 4,
                        },
                        "memory": {},
                    }
                )
                + "\n"
            )

        ingestor.scan_once()

        job = store.get_job("tput-test")
        self.assertEqual(job["status"], "running")
        self.assertEqual(job["current_step"], 1)
        self.assertEqual(job["metrics"]["observed_steps"], 1)
        self.assertEqual(len(store.get_steps("tput-test")), 1)
        store.close()

    def test_retry_after_restart_rejects_old_status_until_current_attempt_finishes(self) -> None:
        status_path = self.root / "results/tput-test/status.json"
        scheduler_path = self.root / "runtime/scheduler_events.jsonl"
        gpu_path = self.root / "results/tput-test/nvidia_smi.csv"
        os.utime(gpu_path, (1005, 1005))
        write_json(
            status_path,
            {
                "job_id": "tput-test",
                "classification": "failed",
                "return_code": 1,
                "started_unix": 1000,
                "finished_unix": 1010,
                "wall_seconds": 10,
                "gpu_mask": "0",
            },
        )
        write_jsonl(
            scheduler_path,
            [
                {
                    "time_unix": 1000,
                    "event": "trial_start",
                    "job_id": "tput-test",
                    "gpu_mask": [0],
                },
                {
                    "time_unix": 1010,
                    "event": "trial_end",
                    "job_id": "tput-test",
                    "classification": "failed",
                    "return_code": 1,
                },
            ],
        )
        db_path = self.root / "runtime/dashboard/retry-restart.sqlite"
        store = DashboardStore(db_path)
        DashboardIngestor(store, root=self.root).scan_once()
        old = store.get_job("tput-test")
        self.assertEqual(old["status"], "failed")
        self.assertEqual(old["current_step"], 3)
        self.assertTrue(old["metrics"])
        store.close()

        metrics_path = self.root / "results/tput-test/metrics/events.rank0.jsonl"
        with metrics_path.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    {
                        "time_unix": 1011,
                        "event": "failure",
                        "rank": 0,
                        "error": "old attempt failure",
                    }
                )
                + "\n"
            )

        with scheduler_path.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    {
                        "time_unix": 2000,
                        "event": "trial_start",
                        "job_id": "tput-test",
                        "gpu_mask": [0],
                    }
                )
                + "\n"
            )

        # A service restart rebuilds the in-memory settled and mtime caches
        # while retaining SQLite cursors.  The directory still contains the
        # previous attempt's failed status at this point.
        store = DashboardStore(db_path)
        ingestor = DashboardIngestor(store, root=self.root)
        self.assertIn("tput-test", ingestor._settled_result_ids)
        ingestor.scan_once()

        running = store.get_job("tput-test")
        self.assertEqual(running["status"], "running")
        self.assertEqual(running["started_unix"], 2000)
        self.assertIsNone(running["finished_unix"])
        self.assertIsNone(running["wall_seconds"])
        self.assertIsNone(running["return_code"])
        self.assertEqual(running["current_step"], 0)
        self.assertEqual(running["metrics"]["observed_steps"], 0)
        self.assertIsNone(running["metrics"]["samples_per_second"])
        self.assertEqual(running["metrics"]["max_allocated_bytes"], 0)
        self.assertEqual(store.get_gpu_samples("tput-test"), [])
        self.assertIsNone(running["last_error"])
        self.assertNotIn("tput-test", ingestor._settled_result_ids)

        retry_events = [
            {"time_unix": 2001, "event": "train_begin", "rank": 0},
            {
                "time_unix": 2002,
                "event": "step_end",
                "rank": 0,
                "global_step": 1,
                "step_seconds": 1.0,
                "optimizer_step_seconds": 0.1,
                "is_warmup": True,
                "tokens": {
                    "computed_tokens": 100,
                    "effective_tokens": 80,
                    "label_tokens": 40,
                    "logical_samples": 4,
                    "computed_attention_token_pairs": 10_000,
                    "effective_attention_token_pairs": 8_000,
                },
                "memory": {
                    "allocated": 2 * 1024**3,
                    "reserved": 3 * 1024**3,
                    "max_allocated": 4 * 1024**3,
                    "max_reserved": 5 * 1024**3,
                },
            },
            {
                "time_unix": 2003,
                "event": "step_end",
                "rank": 0,
                "global_step": 2,
                "step_seconds": 1.0,
                "optimizer_step_seconds": 0.1,
                "is_warmup": False,
                "tokens": {
                    "computed_tokens": 120,
                    "effective_tokens": 100,
                    "label_tokens": 50,
                    "logical_samples": 5,
                    "computed_attention_token_pairs": 12_000,
                    "effective_attention_token_pairs": 10_000,
                },
                "memory": {
                    "allocated": 3 * 1024**3,
                    "reserved": 4 * 1024**3,
                    "max_allocated": 6 * 1024**3,
                    "max_reserved": 7 * 1024**3,
                },
            },
        ]
        with (
            self.root / "results/tput-test/metrics/events.rank0.jsonl"
        ).open("a", encoding="utf-8") as output:
            for event in retry_events:
                output.write(json.dumps(event) + "\n")

        old_gpu_inode = gpu_path.stat().st_ino
        _, old_gpu_offset = store.get_cursor(gpu_path)
        with gpu_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(
                [
                    "timestamp",
                    "index",
                    "memory.used",
                    "utilization.gpu",
                    "power.draw",
                    "clocks.sm",
                ]
            )
            for sample in range(20):
                writer.writerow(
                    [
                        f"2026/07/16 09:01:{sample:02d}.000",
                        0,
                        8000 + sample,
                        95,
                        600,
                        1800,
                    ]
                )
        self.assertEqual(gpu_path.stat().st_ino, old_gpu_inode)
        self.assertGreaterEqual(gpu_path.stat().st_size, old_gpu_offset)

        ingestor.scan_once()
        live = store.get_job("tput-test")
        self.assertEqual(live["status"], "running")
        self.assertEqual(live["current_step"], 2)
        self.assertEqual([row["global_step"] for row in store.get_steps("tput-test")], [1, 2])
        self.assertEqual(live["metrics"]["observed_steps"], 2)
        self.assertAlmostEqual(live["metrics"]["samples_per_second"], 5.0)
        self.assertEqual(live["metrics"]["max_allocated_bytes"], 6 * 1024**3)
        self.assertEqual(live["metrics"]["nvidia_smi_peak_mib"], 8019)
        self.assertEqual(len(store.get_gpu_samples("tput-test")), 20)
        self.assertNotIn("tput-test", ingestor._settled_result_ids)

        write_json(
            status_path,
            {
                "job_id": "tput-test",
                "classification": "success",
                "return_code": 0,
                "started_unix": 2000.5,
                "finished_unix": 2010,
                "wall_seconds": 9.5,
                "gpu_mask": "0",
            },
        )
        with scheduler_path.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    {
                        "time_unix": 2010.1,
                        "event": "trial_end",
                        "job_id": "tput-test",
                        "classification": "success",
                        "return_code": 0,
                    }
                )
                + "\n"
            )
        ingestor.scan_once()

        completed = store.get_job("tput-test")
        self.assertEqual(completed["status"], "success")
        self.assertEqual(completed["started_unix"], 2000.5)
        self.assertEqual(completed["finished_unix"], 2010)
        self.assertEqual(completed["current_step"], 2)
        self.assertIsNone(completed["last_error"])
        self.assertAlmostEqual(completed["metrics"]["samples_per_second"], 5.0)
        self.assertEqual(len(store.get_steps("tput-test")), 2)
        self.assertIn("tput-test", ingestor._settled_result_ids)
        store.close()

    def test_trial_end_can_settle_retry_without_reusing_old_status(self) -> None:
        db_path = self.root / "runtime/dashboard/trial-end.sqlite"
        scheduler_path = self.root / "runtime/scheduler_events.jsonl"
        store = DashboardStore(db_path)
        ingestor = DashboardIngestor(store, root=self.root)
        ingestor.scan_once()
        self.assertEqual(store.get_job("tput-test")["status"], "success")

        with scheduler_path.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    {
                        "time_unix": 2000,
                        "event": "trial_start",
                        "job_id": "tput-test",
                        "gpu_mask": [0],
                    }
                )
                + "\n"
            )
        ingestor.scan_once()
        self.assertEqual(store.get_job("tput-test")["status"], "running")
        self.assertNotIn("tput-test", ingestor._settled_result_ids)

        # The launcher can fail before replacing status.json.  trial_end is
        # emitted after the process exits and is the authoritative terminal
        # record for this attempt; the old success file must remain ignored.
        with scheduler_path.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    {
                        "time_unix": 2001,
                        "event": "trial_end",
                        "job_id": "tput-test",
                        "classification": "launcher_failed",
                        "return_code": 2,
                    }
                )
                + "\n"
            )
        ingestor.scan_once()
        terminal = store.get_job("tput-test")
        self.assertEqual(terminal["status"], "launcher_failed")
        self.assertEqual(terminal["classification"], "launcher_failed")
        self.assertEqual(terminal["started_unix"], 2000)
        self.assertEqual(terminal["finished_unix"], 2001)
        self.assertIn("tput-test", ingestor._settled_result_ids)

        ingestor._registry_signature = None
        ingestor._scan_registry()
        self.assertEqual(store.get_job("tput-test")["status"], "launcher_failed")
        store.close()

        # The terminal scheduler classification also survives an ingestor
        # restart even though status.json still belongs to the old attempt.
        store = DashboardStore(db_path)
        restarted = DashboardIngestor(store, root=self.root)
        self.assertIn("tput-test", restarted._settled_result_ids)
        restarted.scan_once()
        self.assertEqual(store.get_job("tput-test")["status"], "launcher_failed")
        store.close()

    def test_scheduler_cursor_replay_does_not_reset_completed_samples(self) -> None:
        scheduler_path = self.root / "runtime/scheduler_events.jsonl"
        store = DashboardStore(self.root / "runtime/dashboard/replay.sqlite")
        ingestor = DashboardIngestor(store, root=self.root)
        ingestor.scan_once()
        before = store.get_job("tput-test")
        before_steps = store.get_steps("tput-test")
        before_gpu = store.get_gpu_samples("tput-test")

        store.set_cursor(scheduler_path, scheduler_path.stat().st_ino, 0)
        ingestor.scan_once()

        after = store.get_job("tput-test")
        self.assertEqual(after["status"], "success")
        self.assertEqual(after["current_step"], before["current_step"])
        self.assertEqual(after["metrics"], before["metrics"])
        self.assertEqual(store.get_steps("tput-test"), before_steps)
        self.assertEqual(store.get_gpu_samples("tput-test"), before_gpu)
        store.close()

    def test_retry_status_without_timestamps_uses_attempt_mtime(self) -> None:
        status_path = self.root / "results/tput-test/status.json"
        scheduler_path = self.root / "runtime/scheduler_events.jsonl"
        db_path = self.root / "runtime/dashboard/status-mtime.sqlite"
        store = DashboardStore(db_path)
        DashboardIngestor(store, root=self.root).scan_once()
        store.close()

        write_json(
            status_path,
            {
                "job_id": "tput-test",
                "classification": "failed",
                "return_code": 1,
                "gpu_mask": "0",
            },
        )
        os.utime(status_path, (1010, 1010))
        with scheduler_path.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    {
                        "time_unix": 2000,
                        "event": "trial_start",
                        "job_id": "tput-test",
                        "gpu_mask": [0],
                    }
                )
                + "\n"
            )

        store = DashboardStore(db_path)
        ingestor = DashboardIngestor(store, root=self.root)
        ingestor.scan_once()
        self.assertEqual(store.get_job("tput-test")["status"], "running")
        self.assertNotIn("tput-test", ingestor._settled_result_ids)

        write_json(
            status_path,
            {
                "job_id": "tput-test",
                "classification": "success",
                "return_code": 0,
                "gpu_mask": "0",
            },
        )
        os.utime(status_path, (2001, 2001))
        ingestor.scan_once()
        current = store.get_job("tput-test")
        self.assertEqual(current["status"], "success")
        self.assertEqual(current["started_unix"], 2000)
        self.assertIn("tput-test", ingestor._settled_result_ids)
        store.close()

    def test_restart_reindexes_a_mixed_attempt_terminal_row(self) -> None:
        status_path = self.root / "results/tput-test/status.json"
        metrics_path = self.root / "results/tput-test/metrics/events.rank0.jsonl"
        db_path = self.root / "runtime/dashboard/mixed-attempt.sqlite"
        store = DashboardStore(db_path)
        DashboardIngestor(store, root=self.root).scan_once()
        self.assertEqual(store.get_job("tput-test")["current_step"], 3)

        with metrics_path.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    {"time_unix": 2001, "event": "train_begin", "rank": 0}
                )
                + "\n"
            )
            output.write(
                json.dumps(
                    {
                        "time_unix": 2002,
                        "event": "step_end",
                        "rank": 0,
                        "global_step": 1,
                        "step_seconds": 1.0,
                        "is_warmup": False,
                        "tokens": {
                            "computed_tokens": 120,
                            "effective_tokens": 100,
                            "logical_samples": 6,
                        },
                        "memory": {
                            "max_allocated": 8 * 1024**3,
                            "max_reserved": 9 * 1024**3,
                        },
                    }
                )
                + "\n"
            )
        write_json(
            status_path,
            {
                "job_id": "tput-test",
                "classification": "success",
                "return_code": 0,
                "started_unix": 2000,
                "finished_unix": 2010,
                "wall_seconds": 10,
                "gpu_mask": "0",
            },
        )
        # Reproduce the pre-fix DB mix: old status start/wall plus the current
        # scheduler's terminal timestamp.
        store.update_job(
            "tput-test",
            status="success",
            classification="success",
            started_unix=1000,
            finished_unix=2010,
            wall_seconds=10,
        )
        store.close()

        # More than one result-scan batch of newer incomplete directories used
        # to starve the poisoned row forever.  A known repair must outrank them.
        for index in range(12):
            blocker = self.root / f"results/tput-blocker-{index:02d}"
            blocker.mkdir()
            write_json(
                blocker / "rendered_run.json",
                {
                    "job": {
                        "job_id": blocker.name,
                        "kind": "throughput_screen",
                    },
                    "gpu_mask": "0",
                },
            )

        store = DashboardStore(db_path)
        ingestor = DashboardIngestor(store, root=self.root)
        self.assertNotIn("tput-test", ingestor._settled_result_ids)
        self.assertIn("tput-test", ingestor._repair_result_ids)
        ingestor.scan_once()

        repaired = store.get_job("tput-test")
        self.assertEqual(repaired["status"], "success")
        self.assertEqual(repaired["started_unix"], 2000)
        self.assertEqual(repaired["finished_unix"], 2010)
        self.assertEqual(repaired["current_step"], 1)
        self.assertEqual(len(store.get_steps("tput-test")), 1)
        self.assertAlmostEqual(repaired["metrics"]["samples_per_second"], 6.0)
        self.assertEqual(repaired["metrics"]["max_allocated_bytes"], 8 * 1024**3)
        self.assertIn("tput-test", ingestor._settled_result_ids)
        self.assertNotIn("tput-test", ingestor._repair_result_ids)
        store.close()

    def test_historical_status_without_timestamps_is_still_imported(self) -> None:
        status_path = self.root / "results/tput-test/status.json"
        scheduler_path = self.root / "runtime/scheduler_events.jsonl"
        write_json(
            status_path,
            {
                "job_id": "tput-test",
                "classification": "failed",
                "return_code": 1,
                "gpu_mask": "0",
            },
        )
        write_jsonl(scheduler_path, [])
        original = status_path.read_bytes()
        original_mtime = status_path.stat().st_mtime_ns

        store = DashboardStore(self.root / "runtime/dashboard/historical.sqlite")
        ingestor = DashboardIngestor(store, root=self.root)
        ingestor.scan_once()

        self.assertEqual(store.get_job("tput-test")["status"], "failed")
        self.assertIn("tput-test", ingestor._settled_result_ids)
        self.assertEqual(status_path.read_bytes(), original)
        self.assertEqual(status_path.stat().st_mtime_ns, original_mtime)
        store.close()

    def test_registry_prunes_only_obsolete_planned_jobs(self) -> None:
        store = DashboardStore(self.root / "runtime/dashboard/prune.sqlite")
        ingestor = DashboardIngestor(store, root=self.root)
        obsolete = {
            "job_id": "obsolete-planned",
            "campaign_id": ingestor.campaign_id,
            "kind": "throughput",
        }
        completed = {
            "job_id": "historical-success",
            "campaign_id": ingestor.campaign_id,
            "kind": "throughput",
        }
        store.upsert_job(obsolete, phase="throughput", status="planned")
        store.upsert_job(completed, phase="throughput", status="success")

        ingestor.scan_once()

        self.assertIsNone(store.get_job("obsolete-planned"))
        self.assertEqual(store.get_job("historical-success")["status"], "success")
        store.close()

    def test_empty_active_manifest_counts_all_jobs(self) -> None:
        # Regression: a campaign whose active_manifest_job_ids is present but an
        # EMPTY set must NOT hide every job from the overview progress counts.
        # Empty manifest = no active-batch scoping -> count all indexed jobs.
        store = DashboardStore(self.root / "runtime/dashboard/emptymanifest.sqlite")
        ingestor = DashboardIngestor(store, root=self.root)
        for jid, status in (("done-a", "success"), ("done-b", "oom")):
            store.upsert_job(
                {"job_id": jid, "campaign_id": ingestor.campaign_id, "kind": "throughput"},
                phase="throughput",
                status=status,
            )
        store.set_meta(
            f"campaign:{ingestor.campaign_id}:active_manifest_job_ids", []
        )
        overview = DashboardAnalytics(store).overview(ingestor.campaign_id)
        self.assertEqual(overview["runs_by_status"], {"success": 1, "oom": 1})
        self.assertEqual(overview["phase_counts"]["throughput"]["indexed"], 2)
        store.close()

    def test_empty_throughput_registries_count_all_stage_progress(self) -> None:
        # Regression: present-but-empty active screen/formal registries must not
        # zero the 吞吐初筛/正式复验 stage progress.  Empty = "no active batch
        # scoping" -> count all indexed throughput jobs.
        store = DashboardStore(self.root / "runtime/dashboard/tputreg.sqlite")
        ingestor = DashboardIngestor(store, root=self.root)
        for jid, kind, status in (
            ("scr-1", "throughput_screen", "success"),
            ("scr-2", "throughput_screen", "success"),
            ("fml-1", "throughput", "success"),
        ):
            store.upsert_job(
                {"job_id": jid, "campaign_id": ingestor.campaign_id, "kind": kind},
                phase="throughput",
                status=status,
            )
        store.set_meta(
            f"campaign:{ingestor.campaign_id}:active_throughput_screen_job_ids", []
        )
        store.set_meta(
            f"campaign:{ingestor.campaign_id}:active_throughput_formal_job_ids", []
        )
        a = DashboardAnalytics(store)
        self.assertIsNone(a._active_throughput_screen_job_ids([{"campaign_id": ingestor.campaign_id}]))
        self.assertIsNone(a._active_throughput_formal_job_ids([{"campaign_id": ingestor.campaign_id}]))
        tp = a.overview(ingestor.campaign_id)["throughput_progress"]
        self.assertGreaterEqual(tp["screening"]["executed"], 1)
        self.assertGreaterEqual(tp["formal"]["completed"], 1)
        store.close()

    def test_completed_scheduler_closes_stale_running_jobs(self) -> None:
        store = DashboardStore(self.root / "runtime/dashboard/stale.sqlite")
        ingestor = DashboardIngestor(store, root=self.root)
        store.upsert_job(
            {
                "job_id": "stale-running",
                "campaign_id": ingestor.campaign_id,
                "kind": "throughput",
            },
            phase="throughput",
            status="running",
        )
        store.update_job(
            "stale-running",
            status="running",
            started_unix=100.0,
        )
        store.set_meta(
            f"campaign:{ingestor.campaign_id}:scheduler_state",
            {"status": "complete", "time_unix": 200.0},
        )

        ingestor._scan_registry()

        job = store.get_job("stale-running")
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["classification"], "scheduler_interrupted")
        store.close()

    def test_failure_memory_is_indexed_before_first_step(self) -> None:
        job = {
            "job_id": "oom-before-step", "kind": "memory_probe", "family_job_id": "mem-family",
            "model_id": "qwen-test", "train_type": "full", "dataset_id": "short",
            "cutoff_len": 512, "gpu_count": 1, "zero": "none", "gc": False,
            "mbs": 4, "target_gbs": 16, "warmup_steps": 0, "max_steps": 5,
        }
        result = self.root / "results/oom-before-step"
        write_json(result / "rendered_run.json", {"job": job, "gpu_mask": "1", "command": ["python"]})
        write_json(
            result / "status.json",
            {
                "job_id": job["job_id"], "classification": "oom", "return_code": 1,
                "started_unix": 2000, "finished_unix": 2010, "wall_seconds": 10, "gpu_mask": "1",
            },
        )
        error = (
            "OutOfMemoryError: CUDA out of memory. Tried to allocate 7.63 GiB. "
            "GPU 1 has a total capacity of 139.83 GiB of which 7.03 GiB is free. "
            "Of the allocated memory 122.83 GiB is allocated by PyTorch, and 7.94 GiB "
            "is reserved by PyTorch but unallocated."
        )
        write_jsonl(
            result / "metrics/events.rank0.jsonl",
            [
                {
                    "time_unix": 2005, "event": "failure", "rank": 0, "error": error,
                    "memory": {
                        "allocated": 6 * 1024**3, "reserved": 7 * 1024**3,
                        "max_allocated": 7 * 1024**3, "max_reserved": 8 * 1024**3,
                    },
                }
            ],
        )
        with (result / "nvidia_smi.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["timestamp", "index", "memory.used", "utilization.gpu", "power.draw", "clocks.sm"])
            writer.writerow(["2026/07/16 09:10:01.000", 1, 140_000, 100, 600, 1800])

        store = DashboardStore(self.root / "runtime/dashboard/oom.sqlite")
        ingestor = DashboardIngestor(store, root=self.root)
        ingestor.scan_once()
        indexed = store.get_job(job["job_id"])
        self.assertEqual(indexed["status"], "oom")
        self.assertEqual(indexed["current_step"], 0)
        self.assertEqual(indexed["metrics"]["max_allocated_bytes"], 7 * 1024**3)
        self.assertEqual(indexed["metrics"]["max_reserved_bytes"], 8 * 1024**3)
        self.assertEqual(indexed["metrics"]["nvidia_smi_peak_mib"], 140_000)
        self.assertEqual(indexed["metrics"]["oom_requested_bytes"], round(7.63 * 1024**3))
        self.assertEqual(indexed["metrics"]["oom_free_bytes"], round(7.03 * 1024**3))
        self.assertEqual(len(store.get_failures(job["job_id"])), 1)
        store.close()

    def test_api_and_html(self) -> None:
        app = create_app(
            root=self.root,
            db_path=self.root / "runtime/dashboard/api.sqlite",
            scan_interval=60,
        )
        app.state.dashboard_ingestor.scan_once()

        async def exercise_api() -> None:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/")
                self.assertEqual(response.status_code, 200)
                self.assertIn("多硬件 SFT 实验控制台", response.text)
                overview = (await client.get("/api/v1/overview")).json()
                self.assertEqual(overview["runs_by_status"]["success"], 1)
                self.assertEqual(overview["gpu_ids"], [1, 2, 3, 4])
                self.assertEqual(overview["throughput_progress"]["screening"]["total"], 0)
                self.assertEqual(overview["throughput_progress"]["formal"]["completed"], 1)
                self.assertEqual(overview["throughput_progress"]["formal"]["total"], 1)
                runs = (await client.get("/api/v1/runs?status=success")).json()
                self.assertEqual(runs["total"], 1)
                detail = (await client.get("/api/v1/runs/tput-test")).json()
                self.assertEqual(detail["job"]["model_id"], "qwen-test")
                series = (await client.get("/api/v1/runs/tput-test/series")).json()
                self.assertEqual(len(series["steps"]), 3)
                memory = (await client.get("/api/v1/analysis/memory")).json()
                self.assertEqual(memory["rows"][0]["max_feasible_mbs"], 1)
                self.assertIn("mfu", memory["rows"][0])
                self.assertIn("samples_per_second", memory["rows"][0])

        try:
            asyncio.run(exercise_api())
        finally:
            app.state.dashboard_store.close()

    def test_recommendations_compare_distinct_physical_candidates(self) -> None:
        store = DashboardStore(self.root / "runtime/dashboard/recommendations.sqlite")
        base = {
            "campaign_id": "campaign",
            "hardware_id": "hardware",
            "gpu_type": "Test GPU",
            "model_id": "qwen-test",
            "train_type": "lora",
            "dataset_id": "short",
            "cutoff_len": 512,
            "target_gbs": 16,
            "packing": False,
        }
        jobs = [
            {
                **base,
                "job_id": "candidate-a-repeat-0",
                "kind": "throughput",
                "request_id": "throughput-request",
                "gpu_count": 1,
                "zero": "none",
                "gc": False,
                "mbs": 2,
            },
            {
                **base,
                "job_id": "candidate-a-repeat-1",
                "kind": "throughput",
                "request_id": "scaling-request",
                "gpu_count": 1,
                "zero": "none",
                "gc": False,
                "mbs": 2,
            },
            {
                **base,
                "job_id": "candidate-b",
                "kind": "throughput",
                "request_id": "throughput-request",
                "gpu_count": 2,
                "zero": "zero2",
                "gc": True,
                "mbs": 4,
            },
        ]
        for index, job in enumerate(jobs):
            phase = "scaling" if index == 1 else "throughput"
            store.upsert_job(job, phase=phase, record_type="run", status="success")
            store.update_job(
                job["job_id"],
                status="success",
                metrics_json=json.dumps(
                    {
                        "samples_per_second": (
                            100.0
                            if phase == "scaling"
                            else (8.0 if job["gpu_count"] == 1 else 15.0)
                        ),
                        "mfu": 0.4,
                    }
                ),
            )
        store.upsert_job(
            {
                **base,
                "job_id": "candidate-c-pending",
                "kind": "throughput",
                "request_id": "throughput-request",
                "gpu_count": 4,
                "zero": "zero3",
                "gc": True,
                "mbs": 2,
            },
            phase="throughput",
            record_type="run",
            status="planned",
        )
        store.upsert_job(
            {
                **base,
                "job_id": "removed-old-finalist",
                "kind": "throughput",
                "request_id": "old-throughput-request",
                "gpu_count": 8,
                "zero": "zero3",
                "gc": True,
                "mbs": 1,
            },
            phase="throughput",
            record_type="run",
            status="success",
        )
        store.update_job(
            "removed-old-finalist",
            status="success",
            metrics_json=json.dumps({"samples_per_second": 200.0, "mfu": 0.4}),
        )
        store.set_meta("campaigns", [{"campaign_id": "campaign"}])
        store.set_meta(
            "campaign:campaign:active_throughput_formal_job_ids",
            ["candidate-a-repeat-0", "candidate-b", "candidate-c-pending"],
        )

        scenarios = DashboardAnalytics(store).recommendations("campaign")["scenarios"]
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0]["candidate_count"], 3)
        self.assertEqual(scenarios[0]["measured_candidate_count"], 2)
        self.assertEqual(scenarios[0]["pending_candidate_count"], 1)
        self.assertEqual(scenarios[0]["status"], "formal_in_progress")
        self.assertEqual(len(scenarios[0]["candidates"]), 3)
        single_gpu = next(row for row in scenarios[0]["candidates"] if row["gpu_count"] == 1)
        self.assertEqual(single_gpu["runs"], 1)
        self.assertEqual(single_gpu["successful_runs"], 1)
        self.assertEqual(single_gpu["request_ids"], ["throughput-request"])
        # This is the same physical-key projection used by stage_decisions.
        fastest = scenarios[0]["fastest"]
        self.assertEqual(
            (
                fastest["gpu_count"],
                fastest["zero_name"],
                bool(fastest["gc"]),
                fastest["mbs"],
            ),
            (2, "zero2", True, 4),
        )
        store.close()

    def test_overview_separates_screening_and_formal_progress(self) -> None:
        store = DashboardStore(self.root / "runtime/dashboard/progress.sqlite")
        campaign_id = "campaign"
        store.set_meta("campaigns", [{"campaign_id": campaign_id, "gpu_ids": [1, 2, 3, 4]}])
        store.set_meta(
            f"campaign:{campaign_id}:design_summary",
            {"throughput_shortlist_top_k": 2},
        )
        base = {
            "campaign_id": campaign_id,
            "hardware_id": "hardware",
            "gpu_type": "Test GPU",
            "model_id": "qwen-test",
            "train_type": "lora",
            "dataset_id": "short",
            "cutoff_len": 512,
            "target_gbs": 16,
            "packing": False,
        }
        candidates = [
            {**base, "gpu_count": 1, "zero": "none", "gc": False, "mbs": 1},
            {**base, "gpu_count": 2, "zero": "zero2", "gc": False, "mbs": 2},
            {**base, "gpu_count": 4, "zero": "zero3", "gc": True, "mbs": 4},
        ]
        for index, (candidate, status) in enumerate(
            zip(candidates, ("success", "planned", "running"), strict=True)
        ):
            store.upsert_job(
                {
                    **candidate,
                    "job_id": f"screen-{index}",
                    "kind": "throughput_screen",
                    "request_id": "screen-request",
                },
                phase="throughput",
                status=status,
            )
        store.set_meta(
            f"campaign:{campaign_id}:active_throughput_screen_job_ids",
            [f"screen-{index}" for index in range(3)],
        )
        store.upsert_job(
            {
                **candidates[0],
                "job_id": "obsolete-screen",
                "kind": "throughput_screen",
                "request_id": "old-screen-request",
                "mbs": 8,
            },
            phase="throughput",
            status="success",
        )
        store.upsert_job(
            {
                **candidates[1],
                "job_id": "formal-reused",
                "kind": "throughput",
                "request_id": "formal-request",
            },
            phase="throughput",
            status="success",
        )
        store.upsert_job(
            {
                **candidates[2],
                "job_id": "current-formal",
                "kind": "throughput",
                "request_id": "current-formal-request",
            },
            phase="throughput",
            status="planned",
        )
        store.upsert_job(
            {
                **candidates[2],
                "job_id": "scaling-lookalike",
                "kind": "throughput",
                "request_id": "scaling-request",
            },
            phase="scaling",
            status="success",
        )
        store.upsert_job(
            {
                **base,
                "job_id": "obsolete-formal-scenario",
                "kind": "throughput",
                "request_id": "old-formal-request",
                "dataset_id": "removed-from-current-matrix",
                "gpu_count": 1,
                "zero": "none",
                "gc": False,
                "mbs": 1,
            },
            phase="throughput",
            status="success",
        )
        store.update_job(
            "obsolete-formal-scenario",
            status="success",
            metrics_json=json.dumps({"samples_per_second": 10.0, "mfu": 0.5}),
        )
        store.set_meta(
            f"campaign:{campaign_id}:active_throughput_formal_job_ids",
            ["current-formal"],
        )

        overview = DashboardAnalytics(store).overview(campaign_id)
        screening = overview["throughput_progress"]["screening"]
        formal = overview["throughput_progress"]["formal"]
        self.assertEqual(
            screening,
            {
                "completed": 2,
                "total": 3,
                "executed": 1,
                "reused_formal": 1,
                "running": 1,
                "remaining": 0,
            },
        )
        self.assertEqual(formal["completed"], 0)
        self.assertEqual(formal["total"], 1)
        self.assertEqual(formal["successful"], 0)
        self.assertEqual(formal["materialized"], 1)
        self.assertTrue(formal["waiting_for_screening"])
        self.assertFalse(formal["total_is_top_k_upper_bound"])
        self.assertEqual(overview["throughput_progress"]["current_stage"], "screening")
        self.assertEqual(overview["current_phase_label"], "吞吐初筛")
        store.close()

    def test_current_formal_success_can_complete_equivalent_screen(self) -> None:
        store = DashboardStore(self.root / "runtime/dashboard/formal-reuse.sqlite")
        campaign_id = "campaign"
        store.set_meta("campaigns", [{"campaign_id": campaign_id}])
        base = {
            "campaign_id": campaign_id,
            "hardware_id": "hardware",
            "model_id": "qwen-test",
            "train_type": "lora",
            "dataset_id": "short",
            "cutoff_len": 512,
            "gpu_count": 2,
            "zero": "zero3",
            "gc": True,
            "mbs": 4,
            "target_gbs": 16,
            "packing": False,
        }
        store.upsert_job(
            {
                **base,
                "job_id": "screen-current",
                "kind": "throughput_screen",
                "request_id": "screen-request",
            },
            phase="throughput",
            status="planned",
        )
        store.upsert_job(
            {
                **base,
                "job_id": "formal-current",
                "kind": "throughput",
                "request_id": "formal-request",
            },
            phase="throughput",
            status="success",
        )
        store.set_meta(
            f"campaign:{campaign_id}:active_throughput_screen_job_ids",
            ["screen-current"],
        )
        store.set_meta(
            f"campaign:{campaign_id}:active_throughput_formal_job_ids",
            ["formal-current"],
        )

        progress = DashboardAnalytics(store).overview(campaign_id)["throughput_progress"]
        self.assertEqual(progress["screening"]["completed"], 1)
        self.assertEqual(progress["screening"]["executed"], 0)
        self.assertEqual(progress["screening"]["reused_formal"], 1)
        self.assertEqual(progress["formal"]["completed"], 1)
        self.assertEqual(progress["formal"]["total"], 1)
        store.close()

    def test_skipped_formal_is_resolved_but_never_recommendation_evidence(self) -> None:
        store = DashboardStore(self.root / "runtime/dashboard/formal-skip.sqlite")
        campaign_id = "campaign"
        store.set_meta("campaigns", [{"campaign_id": campaign_id}])
        job = {
            "campaign_id": campaign_id,
            "hardware_id": "hardware",
            "job_id": "formal-skipped",
            "kind": "throughput",
            "request_id": "formal-request",
            "model_id": "qwen-test",
            "train_type": "lora",
            "dataset_id": "short",
            "cutoff_len": 512,
            "gpu_count": 1,
            "zero": "none",
            "gc": False,
            "mbs": 1,
            "target_gbs": 16,
            "packing": False,
        }
        store.upsert_job(
            job,
            phase="throughput",
            status="conditional_skipped",
        )
        store.set_meta(
            f"campaign:{campaign_id}:active_throughput_formal_job_ids",
            [job["job_id"]],
        )

        overview = DashboardAnalytics(store).overview(campaign_id)
        formal = overview["throughput_progress"]["formal"]
        self.assertEqual(formal["completed"], 1)
        self.assertEqual(formal["successful"], 0)
        recommendations = DashboardAnalytics(store).recommendations(campaign_id)
        scenario = recommendations["scenarios"][0]
        self.assertEqual(scenario["status"], "comparison_complete")
        self.assertEqual(scenario["measured_candidate_count"], 0)
        self.assertIsNone(scenario["default"])
        candidate = scenario["candidates"][0]
        self.assertTrue(candidate["formal_resolved"])
        self.assertEqual(candidate["formal_successful_runs"], 0)
        self.assertEqual(candidate["failed_runs"], 0)
        self.assertEqual(candidate["skipped_runs"], 1)
        store.close()


if __name__ == "__main__":
    unittest.main()
