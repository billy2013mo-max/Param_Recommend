"""Incrementally index scheduler, trainer and GPU telemetry files."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from scripts.common import ROOT, sha256_file, sha256_json
from scripts.gpu_telemetry import add_clock_adjusted_mfu, normalize_nvidia_smi_row

from .analytics import aggregate_failure_rows, aggregate_gpu_rows, aggregate_step_rows
from .store import FINAL_STATUSES, SKIPPED_STATUSES, DashboardStore


REGISTRY_SOURCES = (
    ("h800_thermal_validation_jobs.jsonl", "validation", "run"),
    ("memory_boundary_families.jsonl", "memory", "family"),
    ("throughput_screen_jobs.jsonl", "throughput", "run"),
    ("throughput_jobs.jsonl", "throughput", "run"),
    ("scaling_candidate_jobs.jsonl", "scaling", "run"),
    ("packing_memory_jobs.jsonl", "packing", "run"),
    ("packing_pair_jobs.jsonl", "packing", "run"),
    ("profiler_jobs.jsonl", "profiler", "run"),
    ("profiler_holdout_jobs.jsonl", "profiler", "run"),
    ("queue_compatibility_canary.jsonl", "validation", "run"),
    ("queue_memory_boundary.jsonl", "memory", "run"),
    ("queue_multi_candidate_ranking.jsonl", "throughput", "run"),
)
COLD_SCHEDULER_EVENT_LIMIT = 256
RESULT_SCAN_BATCH_SIZE = 8
# A started-but-not-terminal run is only shown as "running" if its result tree
# was written within this window; older orphans are marked interrupted.
RUNNING_FRESHNESS_SECONDS = 600


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def phase_for_job(job: dict[str, Any], default: str | None = None) -> str:
    if default:
        return default
    if isinstance(job.get("boundary_probe"), dict):
        return "memory"
    if isinstance(job.get("packing_pair"), dict):
        return "packing"
    job_id = str(job.get("job_id") or "")
    kind = str(job.get("kind") or "")
    if job_id.startswith("preflight-"):
        return "preflight"
    if kind == "thermal_validation" or job_id.startswith("h800val-"):
        return "validation"
    if kind == "smoke" or job_id.startswith("smoke-"):
        return "validation"
    if kind in {"memory_boundary", "memory_probe"} or job_id.startswith("mem-"):
        return "memory"
    if kind == "packing_memory_probe" or job_id.startswith(("packmem-", "packon-", "packoff-")):
        return "packing"
    if kind == "profiler" or job_id.startswith("prof-"):
        return "profiler"
    if job_id.startswith("scale-"):
        return "scaling"
    if job_id.startswith(("tput-", "tputscreen-", "tputd-")) or kind in {"throughput", "throughput_screen"}:
        return "throughput"
    return "other"


class DashboardIngestor:
    def __init__(self, store: DashboardStore, root: Path = ROOT):
        self.store = store
        self.root = root
        self.matrix_dir = root / "matrix"
        self.runtime_dir = root / "runtime"
        self.results_dir = root / "results"
        self.artifact_dir = root / "artifacts"
        self.config_dir = root / "config"
        self.last_error: str | None = None
        self.last_scan_unix = 0.0
        self._last_registry_scan = 0.0
        self._registry_signature: tuple[tuple[Any, ...], ...] | None = None
        self._approval_surface_signature: tuple[tuple[Any, ...], ...] | None = None
        self._mtime_cache: dict[str, tuple[int, int]] = {}
        self._live_job_ids: set[str] | None = None
        inventory_path = self.artifact_dir / "model_inventory.json"
        inventory = read_json(inventory_path) if inventory_path.is_file() else {"models": []}
        self.models = {row["id"]: row for row in inventory.get("models", [])}
        hardware_path = self.config_dir / "hardware.json"
        self.hardware = read_json(hardware_path) if hardware_path.is_file() else {
            "bf16_dense_peak_flops_per_second_for_standard_mfu": 989.5e12,
            "clock_normalized_dense_peak_flops_per_second": 989.5e12,
        }
        experiment_path = self.config_dir / "experiment.json"
        self.experiment = read_json(experiment_path) if experiment_path.is_file() else {}
        scope = self.experiment.get("training_scope", {})
        self.campaign_id = str(self.experiment.get("campaign_id") or scope.get("phase_id") or root.name)
        self.hardware_id = str(
            self.experiment.get("hardware_id")
            or self.hardware.get("hardware_id")
            or self.hardware.get("gpu_id")
            or "unknown"
        )
        self.gpu_type = str(scope.get("gpu_type") or self.hardware.get("name") or self.hardware_id)
        indexed_jobs = self.store.all_jobs(campaign_id=self.campaign_id)
        # Completed result directories are immutable. Reuse the disposable
        # index across service restarts instead of re-reading thousands of
        # metrics files before a new registry can become visible.
        self._repair_result_ids = {
            str(job["job_id"])
            for job in indexed_jobs
            if job.get("status") in FINAL_STATUSES
            and not self._terminal_attempt_is_consistent(job)
        }
        self._settled_result_ids = {
            str(job["job_id"])
            for job in indexed_jobs
            if job.get("status") in FINAL_STATUSES
            and self._terminal_attempt_is_consistent(job)
            and (
                job.get("record_type") != "run"
                or bool(job.get("metrics"))
            )
        }
        # A stable job ID may be retried in the same result directory.  Keep
        # the current attempt boundary independently from status.json: until
        # run_job replaces that file it still describes the previous attempt.
        # Running rows make this state recoverable across dashboard restarts.
        self._active_attempt_started_unix: dict[str, float | None] = {
            str(job["job_id"]): (
                float(job["started_unix"])
                if job.get("started_unix") is not None
                else None
            )
            for job in indexed_jobs
            if job.get("record_type") == "run" and job.get("status") == "running"
        }
        # A current terminal event/status permits settling only after this scan
        # has consumed the attempt's final metric and GPU files.
        self._current_attempt_terminal_ids: set[str] = set()
        # nvidia_smi.csv is truncated in place for a retry, so inode/size alone
        # cannot distinguish it from the previous attempt.  Reset its cursor
        # once the file mtime proves that the new monitor has taken ownership.
        self._gpu_cursor_reset_attempt_ids: set[str] = set()

    @staticmethod
    def _timestamp(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _terminal_attempt_is_consistent(cls, job: dict[str, Any]) -> bool:
        """Keep a previously poisoned retry eligible for one source re-index.

        The old retry race produced a distinctive mixed attempt: ``started``
        and ``wall_seconds`` came from the old status file, while ``finished``
        came from the new trial_end.  Healthy historical rows agree within a
        small scheduler/process timestamp tolerance and remain settled.
        """

        if job.get("record_type") != "run":
            return True
        started = cls._timestamp(job.get("started_unix"))
        finished = cls._timestamp(job.get("finished_unix"))
        wall_seconds = cls._timestamp(job.get("wall_seconds"))
        if started is None or finished is None or wall_seconds is None:
            return True
        observed_wall = finished - started
        tolerance = max(10.0, max(0.0, wall_seconds) * 0.25)
        return observed_wall >= 0 and abs(observed_wall - wall_seconds) <= tolerance

    def _valid_conditional_skip(
        self, payload: dict[str, Any], *, expected_job_id: str | None = None
    ) -> bool:
        job_id = str(payload.get("job_id") or "")
        return bool(
            job_id
            and (expected_job_id is None or job_id == expected_job_id)
            and self._live_job_ids is not None
            and job_id in self._live_job_ids
            and payload.get("schema") == "h800_adaptive_scheduler_terminal/v1"
            and payload.get("classification") == "conditional_skipped"
            and payload.get("terminal") is True
            and payload.get("training_started") is False
            and payload.get("calibration_observation_eligible") is False
            and payload.get("training_status_json_was_not_written") is True
        )

    def _status_is_for_current_attempt(
        self,
        job_id: str,
        status: dict[str, Any],
        status_mtime: float | None = None,
    ) -> bool:
        """Reject a terminal status file left behind by an earlier retry.

        ``started_unix`` is the primary attempt identity.  ``finished_unix``
        is also accepted when the dashboard restarted after ``train_begin``:
        that event can be later than the process-level start recorded by
        run_job, while a terminal time after the known attempt boundary still
        proves that the status belongs to the current run.
        """

        if job_id not in self._active_attempt_started_unix:
            return True
        attempt_started = self._active_attempt_started_unix[job_id]
        if attempt_started is None:
            return False
        status_started = self._timestamp(status.get("started_unix"))
        if status_started is not None and status_started >= attempt_started:
            return True
        status_finished = self._timestamp(status.get("finished_unix"))
        if status_finished is not None:
            return status_finished >= attempt_started
        return status_started is None and bool(
            status_mtime is not None and status_mtime >= attempt_started
        )

    def _decorate(
        self, job: dict[str, Any], *, force_campaign: bool = False
    ) -> dict[str, Any]:
        job_id = str(job.get("job_id") or "")
        active_job = bool(
            self._live_job_ids is not None and job_id in self._live_job_ids
        )
        return {
            **job,
            "campaign_id": (
                self.campaign_id
                if force_campaign or active_job
                else job.get("campaign_id") or self.campaign_id
            ),
            "hardware_id": job.get("hardware_id") or self.hardware_id,
            "gpu_type": job.get("gpu_type") or self.gpu_type,
            "results_root": str(self.results_dir),
        }

    @staticmethod
    def _strict_job_ids(value: Any) -> list[str] | None:
        if not isinstance(value, list) or not value:
            return None
        if not all(isinstance(item, str) and item for item in value):
            return None
        return value if len(value) == len(set(value)) else None

    def _project_path(self, value: Any) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        root = self.root.resolve()
        declared = Path(value)
        candidate = (
            declared.resolve()
            if declared.is_absolute()
            else (root / declared).resolve()
        )
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate

    def _approved_queue(self) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
        """Return the one live, fully bound queue or fail closed.

        ``None`` means this is a legacy campaign with no live approval files,
        so its matrix remains authoritative.  An empty list means an approval
        surface exists but is invalid; in that case no planned rows are active.
        Candidate artifacts are intentionally never inspected here.
        """

        declared_design_path = self.runtime_dir / "approval_design.json"
        declared_approval_path = self.config_dir / "APPROVED_TO_RUN.json"
        approval_surface_present = (
            declared_design_path.exists() or declared_approval_path.exists()
        )
        diagnostics: dict[str, Any] = {
            "mode": "legacy_matrix" if not approval_surface_present else "invalid_approval",
            "checks": {},
        }
        if not approval_surface_present:
            return None, diagnostics

        design_path = self._project_path("runtime/approval_design.json")
        approval_path = self._project_path("config/APPROVED_TO_RUN.json")
        if design_path is None or approval_path is None:
            diagnostics["read_error"] = "approval/design path escapes campaign root"
            return [], diagnostics

        try:
            design = read_json(design_path)
            approval = read_json(approval_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            diagnostics["read_error"] = repr(error)
            return [], diagnostics
        if not isinstance(design, dict) or not isinstance(approval, dict):
            diagnostics["read_error"] = "approval and design must be JSON objects"
            return [], diagnostics

        binding = design.get("queue_binding")
        binding = binding if isinstance(binding, dict) else {}
        queue_path = self._project_path(binding.get("path"))
        rows: list[dict[str, Any]] = []
        queue_error: str | None = None
        if queue_path is not None and queue_path.is_file():
            try:
                values = read_jsonl(queue_path)
                if not all(isinstance(row, dict) for row in values):
                    raise ValueError("queue rows must be JSON objects")
                rows = values
            except (OSError, ValueError, json.JSONDecodeError) as error:
                queue_error = repr(error)
        else:
            queue_error = "bound queue is absent or outside the campaign root"

        job_ids = [str(row.get("job_id") or "") for row in rows]
        valid_ids = self._strict_job_ids(job_ids)
        payload_hashes = [sha256_json(row) for row in rows]
        payload_by_id = (
            dict(zip(job_ids, payload_hashes, strict=True))
            if valid_ids is not None
            else {}
        )
        actual_queue_sha256 = (
            sha256_file(queue_path)
            if queue_path is not None and queue_path.is_file()
            else None
        )
        actual_design_sha256 = sha256_file(design_path)
        actual_approval_sha256 = sha256_file(approval_path)
        relative_queue_path = (
            str(queue_path.resolve().relative_to(self.root.resolve()))
            if queue_path is not None
            else None
        )
        design_manifest = design.get("file_sha256")
        design_manifest = design_manifest if isinstance(design_manifest, dict) else {}
        design_allowed = self._strict_job_ids(design.get("allowed_job_ids"))
        approval_allowed = self._strict_job_ids(approval.get("allowed_job_ids"))
        design_order = self._strict_job_ids(design.get("execution_order"))
        approval_order = self._strict_job_ids(approval.get("execution_order"))
        runtime_identity = design.get("runtime_identity")
        runtime_identity = runtime_identity if isinstance(runtime_identity, dict) else {}
        runtime_patch = design.get("runtime_patch")
        runtime_patch = runtime_patch if isinstance(runtime_patch, dict) else {}
        provenance_binding = design.get("provenance_binding")
        provenance_binding = (
            provenance_binding if isinstance(provenance_binding, dict) else {}
        )
        provenance_path = self._project_path(provenance_binding.get("path"))
        actual_provenance_sha256 = (
            sha256_file(provenance_path)
            if provenance_path is not None and provenance_path.is_file()
            else None
        )
        relative_provenance_path = (
            str(provenance_path.resolve().relative_to(self.root.resolve()))
            if provenance_path is not None
            else None
        )
        experiment_path = self.config_dir / "experiment.json"
        experiment = read_json(experiment_path) if experiment_path.is_file() else {}
        experiment = experiment if isinstance(experiment, dict) else {}
        training_scope = experiment.get("training_scope")
        training_scope = training_scope if isinstance(training_scope, dict) else {}
        measurement = experiment.get("measurement")
        measurement = measurement if isinstance(measurement, dict) else {}
        resource_scope = approval.get("resource_scope")
        resource_scope = resource_scope if isinstance(resource_scope, dict) else {}
        configured_gpu_ids = training_scope.get("gpu_ids")
        design_gpu_ids = design.get("authorized_gpu_ids")
        approval_gpu_ids = resource_scope.get("gpu_ids")
        configured_max_gpu_count = training_scope.get("max_gpu_count")
        design_max_gpu_count = design.get("max_gpu_count")
        approval_max_gpu_count = resource_scope.get("max_gpu_count")
        experiment_relative = "config/experiment.json"
        queue_rows_within_gpu_scope = False
        try:
            maximum = int(design_max_gpu_count)
            queue_rows_within_gpu_scope = maximum > 0 and all(
                0 < int(row.get("gpu_count") or 0) <= maximum for row in rows
            )
        except (TypeError, ValueError):
            pass
        checks = {
            "approval_schema_supported": approval.get("schema_version") == 2,
            "design_schema_supported": design.get("schema_version") == 1,
            "design_training_not_started": design.get("training_started") is False,
            "approval_true": approval.get("approved") is True,
            "design_hash_matches_approval": approval.get("design_sha256")
            == actual_design_sha256,
            "execution_order_exact": design_order is not None
            and approval_order == design_order,
            "approval_phase_matches_first_stage": design_order is not None
            and approval.get("phase_id") == design_order[0],
            "binding_schema_supported": binding.get("schema_version") == 1,
            "queue_path_inside_root": queue_path is not None,
            "queue_readable_and_nonempty": queue_error is None and bool(rows),
            "queue_sha256_matches": actual_queue_sha256 is not None
            and binding.get("sha256") == actual_queue_sha256,
            "design_manifest_binds_queue": relative_queue_path is not None
            and design_manifest.get(relative_queue_path) == actual_queue_sha256,
            "queue_job_ids_unique": valid_ids is not None,
            "design_allowed_ids_exact": design_allowed is not None
            and design_allowed == job_ids,
            "approval_allowed_ids_exact": approval_allowed is not None
            and approval_allowed == job_ids,
            "ordered_job_ids_exact": binding.get("ordered_job_ids") == job_ids,
            "ordered_payload_hashes_exact": binding.get(
                "ordered_job_payload_sha256"
            )
            == payload_hashes,
            "payload_hash_map_exact": binding.get("job_payload_sha256")
            == payload_by_id,
            "approval_binds_queue_binding": approval.get("queue_binding_sha256")
            == sha256_json(binding),
            "runtime_identity_internal": bool(runtime_identity)
            and design.get("runtime_fingerprint_sha256")
            == sha256_json(runtime_identity),
            "approval_binds_runtime_identity": approval.get(
                "runtime_fingerprint_sha256"
            )
            == design.get("runtime_fingerprint_sha256"),
            "runtime_patch_declared_healthy": bool(runtime_patch)
            and runtime_patch.get("all_passed") is True,
            "approval_binds_runtime_patch": approval.get("runtime_patch_sha256")
            == sha256_json(runtime_patch),
            "provenance_path_inside_root": provenance_path is not None,
            "provenance_file_sha256_matches": actual_provenance_sha256 is not None
            and provenance_binding.get("sha256") == actual_provenance_sha256,
            "design_manifest_binds_provenance": (
                relative_provenance_path is not None
                and design_manifest.get(relative_provenance_path)
                == actual_provenance_sha256
            ),
            "approval_binds_provenance": approval.get("provenance_sha256")
            == provenance_binding.get("sha256"),
            "design_manifest_binds_experiment": experiment_path.is_file()
            and design_manifest.get(experiment_relative)
            == sha256_file(experiment_path),
            "design_gpu_scope_matches_config": isinstance(configured_gpu_ids, list)
            and configured_gpu_ids == design_gpu_ids
            and configured_max_gpu_count == design_max_gpu_count,
            "approval_gpu_scope_matches_design": approval_gpu_ids == design_gpu_ids
            and approval_max_gpu_count == design_max_gpu_count,
            "approval_forbids_outside_gpu_pool": resource_scope.get(
                "allow_gpu_ids_outside_pool"
            )
            is False,
            "parallelism_matches_config": resource_scope.get(
                "performance_parallelism"
            )
            == measurement.get("performance_parallelism")
            and measurement.get("performance_parallelism")
            in {"disjoint_gpu_masks", "exclusive_pool"},
            "queue_rows_within_gpu_scope": queue_rows_within_gpu_scope,
        }
        diagnostics.update(
            {
                "checks": checks,
                "queue_path": relative_queue_path,
                "queue_sha256": actual_queue_sha256,
                "design_sha256": actual_design_sha256,
                "approval_sha256": actual_approval_sha256,
                "read_error": queue_error,
                "job_count": len(rows),
                "authorization_scope": "read_only_display_binding_subset",
                "runtime_probe_performed": False,
            }
        )
        if not all(checks.values()):
            return [], diagnostics
        diagnostics["mode"] = "approved_queue"
        return rows, diagnostics

    def _approval_surface_changed(self) -> bool:
        """Detect approval promotion/tampering without waiting 60 seconds."""

        declared_paths = [
            ("runtime/approval_design.json", self.runtime_dir / "approval_design.json"),
            (
                "config/APPROVED_TO_RUN.json",
                self.config_dir / "APPROVED_TO_RUN.json",
            ),
        ]
        paths: list[Path] = []
        markers: list[tuple[Any, ...]] = []
        for relative, declared in declared_paths:
            safe = self._project_path(relative)
            if safe is not None:
                paths.append(safe)
                continue
            try:
                stat = declared.lstat()
                markers.append(
                    (str(declared), "outside_root", stat.st_mtime_ns, stat.st_size)
                )
            except OSError:
                markers.append((str(declared), "missing", None, None))
        design_path = self._project_path("runtime/approval_design.json")
        if design_path is not None and design_path.is_file():
            try:
                design = read_json(design_path)
                binding = design.get("queue_binding") if isinstance(design, dict) else None
                queue_path = self._project_path(
                    binding.get("path") if isinstance(binding, dict) else None
                )
                if queue_path is not None:
                    paths.append(queue_path)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        for path in paths:
            try:
                stat = path.stat()
                markers.append(
                    (str(path.resolve()), stat.st_ino, stat.st_mtime_ns, stat.st_size)
                )
            except OSError:
                markers.append((str(path), None, None, None))
        signature = tuple(markers)
        changed = signature != self._approval_surface_signature
        self._approval_surface_signature = signature
        return changed

    def _tail_complete_lines(self, path: Path) -> tuple[list[tuple[int, str]], int | None, int]:
        if not path.is_file():
            return [], None, 0
        stat = path.stat()
        old_inode, offset = self.store.get_cursor(path)
        if old_inode != stat.st_ino or stat.st_size < offset:
            offset = 0
        if stat.st_size == offset:
            return [], stat.st_ino, offset
        with path.open("rb") as source:
            source.seek(offset)
            data = source.read()
        last_newline = data.rfind(b"\n")
        if last_newline < 0:
            return [], stat.st_ino, offset
        complete = data[: last_newline + 1]
        rows: list[tuple[int, str]] = []
        cursor = offset
        for raw in complete.splitlines(keepends=True):
            line_offset = cursor
            cursor += len(raw)
            text = raw.decode("utf-8", errors="replace").strip()
            if text:
                rows.append((line_offset, text))
        return rows, stat.st_ino, offset + len(complete)

    def _scan_registry(self) -> bool:
        approved_rows, approval_diagnostics = self._approved_queue()
        registry_mode = str(approval_diagnostics.get("mode") or "invalid_approval")
        signature_rows: list[tuple[Any, ...]] = []
        for filename, _, _ in REGISTRY_SOURCES:
            path = self.matrix_dir / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            signature_rows.append((filename, stat.st_mtime_ns, stat.st_size))
        signature_rows.append(
            (
                "live_approval",
                registry_mode,
                sha256_json(approval_diagnostics),
            )
        )
        signature = tuple(signature_rows)
        if signature == self._registry_signature:
            return False
        self._registry_signature = signature

        changed = False
        jobs_to_upsert: list[tuple[dict[str, Any], str, str, str]] = []
        live_job_ids: set[str] = set()
        active_throughput_formal_job_ids: set[str] = set()
        if approved_rows is not None:
            # A present approval surface is authoritative even when invalid.
            # Invalid bindings therefore expose zero planned rows instead of
            # silently falling back to an unrelated matrix or artifact.
            live_job_ids = {
                str(job["job_id"])
                for job in approved_rows
            }
            self._live_job_ids = live_job_ids
            for job in approved_rows:
                phase = phase_for_job(job)
                jobs_to_upsert.append(
                    (
                        self._decorate(job, force_campaign=True),
                        phase,
                        "run",
                        "planned",
                    )
                )
                if phase == "throughput" and job.get("kind") == "throughput":
                    active_throughput_formal_job_ids.add(str(job["job_id"]))
            changed = True
        else:
            for filename, phase, record_type in REGISTRY_SOURCES:
                path = self.matrix_dir / filename
                if not path.is_file():
                    continue
                for job in read_jsonl(path):
                    jobs_to_upsert.append(
                        (
                            self._decorate(
                                job,
                                force_campaign=filename.startswith("queue_"),
                            ),
                            phase,
                            record_type,
                            "planned",
                        )
                    )
                    live_job_ids.add(str(job["job_id"]))
                    # scaling_candidate_jobs.jsonl also uses kind=throughput.
                    # Only the authoritative formal matrix contributes to the
                    # current formal denominator/recommendations.
                    if (
                        filename == "throughput_jobs.jsonl"
                        and job.get("kind") == "throughput"
                    ):
                        active_throughput_formal_job_ids.add(str(job["job_id"]))
                changed = True
            self._live_job_ids = live_job_ids
            for path in sorted((self.runtime_dir / "jobs").glob("*.json")):
                try:
                    job = read_json(path)
                    # runtime/jobs is a historical render cache, not a queue.
                    # Re-indexing every old planned render would resurrect
                    # candidates removed by a newly materialized shortlist.
                    if str(job["job_id"]) not in live_job_ids:
                        continue
                    jobs_to_upsert.append(
                        (self._decorate(job), phase_for_job(job), "run", "planned")
                    )
                except (OSError, json.JSONDecodeError, KeyError):
                    continue
        self.store.upsert_jobs(jobs_to_upsert)
        self._live_job_ids = live_job_ids
        active_manifest_job_ids = (
            sorted(live_job_ids) if approved_rows is not None else None
        )
        self.store.set_meta(
            f"campaign:{self.campaign_id}:active_manifest_job_ids",
            active_manifest_job_ids,
        )
        self.store.set_meta(
            f"campaign:{self.campaign_id}:active_registry_mode",
            registry_mode,
        )
        self.store.set_meta(
            f"campaign:{self.campaign_id}:active_queue_validation",
            approval_diagnostics,
        )
        self.store.set_meta(
            f"campaign:{self.campaign_id}:active_throughput_screen_job_ids",
            sorted(
                str(job["job_id"])
                for job, phase, record_type, _ in jobs_to_upsert
                if phase == "throughput"
                and record_type == "run"
                and job.get("kind") == "throughput_screen"
            ),
        )
        self.store.set_meta(
            f"campaign:{self.campaign_id}:active_throughput_formal_job_ids",
            sorted(active_throughput_formal_job_ids),
        )
        changed |= bool(self.store.prune_planned_jobs(self.campaign_id, live_job_ids))
        design_path = self.matrix_dir / "design_summary.json"
        if design_path.is_file():
            self.store.set_meta(f"campaign:{self.campaign_id}:design_summary", read_json(design_path))
        experiment_path = self.config_dir / "experiment.json"
        if experiment_path.is_file():
            self.store.set_meta(f"campaign:{self.campaign_id}:experiment", read_json(experiment_path))
        preflight_paths = (
            self.artifact_dir / "preflight_report.json",
            self.artifact_dir / "preflight_current_host.json",
        )
        for preflight_path in preflight_paths:
            if preflight_path.is_file():
                self.store.set_meta(
                    f"campaign:{self.campaign_id}:preflight",
                    read_json(preflight_path),
                )
                break
        scheduler_state = self.store.get_meta(
            f"campaign:{self.campaign_id}:scheduler_state", {}
        )
        if scheduler_state.get("status") == "complete":
            changed |= bool(
                self.store.finalize_stale_running_jobs(
                    self.campaign_id,
                    float(scheduler_state.get("time_unix") or time.time()),
                )
            )
        return changed

    def _scan_scheduler(self, dirty_jobs: set[str]) -> bool:
        path = self.runtime_dir / "scheduler_events.jsonl"
        old_inode, _ = self.store.get_cursor(path)
        lines, inode, new_offset = self._tail_complete_lines(path)
        if old_inode is None and len(lines) > COLD_SCHEDULER_EVENT_LIMIT:
            # Result status files and boundary summaries are authoritative for
            # historical terminal outcomes. Replaying tens of thousands of
            # scheduler events one transaction at a time makes a disposable
            # dashboard index take many minutes to rebuild while adding no
            # value to the recent-events UI. Keep enough tail context to
            # reconstruct the current/latest execution, then advance the
            # cursor to the real end so subsequent scans remain incremental.
            lines = lines[-COLD_SCHEDULER_EVENT_LIMIT:]
        for offset, line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = {
                **event,
                "campaign_id": self.campaign_id,
                "hardware_id": self.hardware_id,
                "gpu_type": self.gpu_type,
            }
            event_key = hashlib.sha256(f"{path}:{offset}:{line}".encode()).hexdigest()
            if not self.store.add_scheduler_event(event_key, event):
                # A cursor reset (for example after log replacement) may replay
                # an old trial_start.  The event table is the idempotency key;
                # never let a duplicate reset a newer attempt's samples.
                continue
            scheduler_key = f"campaign:{self.campaign_id}:scheduler_state"
            event_kind = str(event.get("event") or "")
            if event_kind == "scheduler_start":
                self.store.set_meta("scheduler_state", {**event, "status": "running"})
                self.store.set_meta(scheduler_key, {**event, "status": "running"})
            elif event_kind == "queue_updated":
                current = self.store.get_meta(scheduler_key, {})
                state = {**current, **event, "status": "running"}
                self.store.set_meta("scheduler_state", state)
                self.store.set_meta(scheduler_key, state)
            elif event_kind == "campaign_budget_exhausted":
                current = self.store.get_meta(scheduler_key, {})
                state = {**current, **event, "status": "budget_exhausted"}
                self.store.set_meta("scheduler_state", state)
                self.store.set_meta(scheduler_key, state)
            elif event_kind == "scheduler_halted":
                current = self.store.get_meta(scheduler_key, {})
                state = {**current, **event, "status": "halted"}
                self.store.set_meta("scheduler_state", state)
                self.store.set_meta(scheduler_key, state)
                self.store.finalize_stale_running_jobs(
                    self.campaign_id,
                    float(event.get("time_unix") or time.time()),
                )
            elif event_kind == "scheduler_complete":
                current = self.store.get_meta(scheduler_key, {})
                state = {**current, **event, "status": "complete"}
                self.store.set_meta("scheduler_state", state)
                self.store.set_meta(scheduler_key, state)
                self.store.finalize_stale_running_jobs(
                    self.campaign_id,
                    float(event.get("time_unix") or time.time()),
                )
            job_id = event.get("job_id")
            if not job_id:
                continue
            if not self.store.job_exists(job_id):
                self.store.upsert_job(
                    self._decorate({"job_id": job_id, "kind": "unknown"}),
                    phase_for_job({"job_id": job_id}),
                    record_type="run",
                )
            if event_kind == "trial_start":
                event_started = self._timestamp(event.get("time_unix"))
                current = self.store.get_job(job_id) or {}
                known_times = [
                    value
                    for value in (
                        self._timestamp(current.get("started_unix")),
                        self._timestamp(current.get("finished_unix")),
                        self._active_attempt_started_unix.get(job_id),
                    )
                    if value is not None
                ]
                if (
                    event_started is not None
                    and known_times
                    and event_started <= max(known_times)
                ):
                    dirty_jobs.add(job_id)
                    continue
                self._settled_result_ids.discard(job_id)
                self._current_attempt_terminal_ids.discard(job_id)
                self._gpu_cursor_reset_attempt_ids.discard(job_id)
                self._active_attempt_started_unix[job_id] = event_started
                self.store.reset_job_samples(job_id, include_gpu=True)
                self.store.update_job(
                    job_id,
                    status="running",
                    classification=None,
                    gpu_mask=json.dumps(event.get("gpu_mask") or []),
                    started_unix=event.get("time_unix") or time.time(),
                    finished_unix=None,
                    wall_seconds=None,
                    return_code=None,
                )
            elif event_kind == "trial_end":
                attempt_started = self._active_attempt_started_unix.get(job_id)
                event_time = self._timestamp(event.get("time_unix"))
                current = self.store.get_job(job_id) or {}
                known_finished = self._timestamp(current.get("finished_unix"))
                if (
                    (
                        job_id in self._active_attempt_started_unix
                        and attempt_started is not None
                        and (event_time is None or event_time < attempt_started)
                    )
                    or (
                        job_id not in self._active_attempt_started_unix
                        and event_time is not None
                        and known_finished is not None
                        and event_time <= known_finished
                    )
                ):
                    # An out-of-order terminal event from the previous attempt
                    # must not close the retry that is currently running.
                    dirty_jobs.add(job_id)
                    continue
                classification = str(event.get("classification") or "failed")
                self.store.update_job(
                    job_id,
                    status=classification,
                    classification=classification,
                    return_code=event.get("return_code"),
                    finished_unix=event.get("time_unix") or time.time(),
                    last_error=event.get("launcher_output_tail") if classification not in {"success", "oom"} else None,
                )
                self._settled_result_ids.discard(job_id)
                self._current_attempt_terminal_ids.add(job_id)
            elif event_kind in {"family_skipped", "conditional_skipped"}:
                if self._valid_conditional_skip(event):
                    self.store.update_job(
                        job_id,
                        status="conditional_skipped",
                        classification="conditional_skipped",
                        finished_unix=event.get("time_unix") or time.time(),
                        last_error=None,
                    )
                    self._settled_result_ids.discard(job_id)
                    self._current_attempt_terminal_ids.add(job_id)
                # No trainer ran, so do not synthesize an empty training metric
                # summary or count this row as observed training evidence.
                continue
            elif event_kind == "family_exception":
                self.store.update_job(job_id, status="failed", classification="family_exception", last_error=event.get("error"))
            elif event_kind == "family_start":
                self.store.update_job(
                    job_id,
                    status="running",
                    gpu_mask=json.dumps(event.get("gpu_mask") or []),
                    started_unix=event.get("time_unix") or time.time(),
                )
            elif event_kind == "family_end":
                classification = str(event.get("classification") or "success")
                fields: dict[str, Any] = {
                    "status": classification,
                    "classification": classification,
                    "finished_unix": event.get("time_unix") or time.time(),
                }
                if event.get("phase") == "memory":
                    fields["boundary_json"] = json.dumps(
                        {
                            "family_job_id": job_id,
                            "max_feasible_mbs": event.get("max_feasible_mbs"),
                            "first_failed_mbs": event.get("first_failed_mbs"),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                self.store.update_job(job_id, **fields)
            dirty_jobs.add(job_id)
        if lines and inode is not None:
            self.store.set_cursor(path, inode, new_offset)
        return bool(lines)

    def _scan_metric_events(self, result_dir: Path, dirty_jobs: set[str]) -> bool:
        changed = False
        job_id = result_dir.name
        metrics_dir = result_dir / "metrics"
        attempt_started = self._active_attempt_started_unix.get(job_id)
        if attempt_started is None and job_id in self._repair_result_ids:
            current = self.store.get_job(job_id) or {}
            attempt_started = self._timestamp(current.get("started_unix"))
        for path in sorted(metrics_dir.glob("events.rank*.jsonl")):
            match = re.search(r"rank(\d+)", path.name)
            rank = int(match.group(1)) if match else 0
            lines, inode, new_offset = self._tail_complete_lines(path)
            if lines:
                changed = True
                dirty_jobs.add(job_id)
            for _, line in lines:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_time = self._timestamp(event.get("time_unix"))
                if (
                    attempt_started is not None
                    and event_time is not None
                    and event_time < attempt_started
                ):
                    # A cursor replay must not temporarily resurrect failures,
                    # steps, or peaks from the previous attempt.
                    continue
                kind = event.get("event")
                if kind == "step_end":
                    self.store.add_step(job_id, rank, event)
                elif kind == "train_begin":
                    self.store.reset_job_samples(job_id, rank=rank)
                    current = self.store.get_job(job_id)
                    if current and current.get("status") not in FINAL_STATUSES:
                        fields: dict[str, Any] = {"status": "running"}
                        if current.get("started_unix") is None:
                            fields["started_unix"] = event.get("time_unix") or time.time()
                        self.store.update_job(job_id, **fields)
                        if job_id not in self._active_attempt_started_unix:
                            self._active_attempt_started_unix[job_id] = self._timestamp(
                                current.get("started_unix")
                                or event.get("time_unix")
                            )
                elif kind == "failure":
                    self.store.add_failure(job_id, rank, event)
                    if self.store.job_exists(job_id):
                        self.store.update_job(job_id, last_error=str(event.get("error") or "training failure"))
            if lines and inode is not None:
                self.store.set_cursor(path, inode, new_offset)
        return changed

    def _scan_gpu_csv(self, result_dir: Path, dirty_jobs: set[str]) -> bool:
        path = result_dir / "nvidia_smi.csv"
        job_id = result_dir.name
        attempt_started = self._active_attempt_started_unix.get(job_id)
        if job_id in self._repair_result_ids:
            try:
                stat = path.stat()
            except OSError:
                stat = None
            if stat is not None:
                # A poisoned terminal row no longer has an in-memory active
                # attempt after restart.  Its current CSV nevertheless belongs
                # entirely to the retried run because run_job truncates it.
                self.store.set_cursor(path, stat.st_ino, 0)
                self._gpu_cursor_reset_attempt_ids.add(job_id)
        if (
            job_id in self._active_attempt_started_unix
            and job_id not in self._gpu_cursor_reset_attempt_ids
            and attempt_started is not None
        ):
            try:
                stat = path.stat()
            except OSError:
                stat = None
            if stat is not None and stat.st_mtime >= attempt_started:
                # run_job opens this file with "w".  It can retain the old inode
                # and grow beyond the previous cursor before our next scan, in
                # which case the generic inode/size check would miss its prefix.
                self.store.set_cursor(path, stat.st_ino, 0)
                self._gpu_cursor_reset_attempt_ids.add(job_id)
        lines, inode, new_offset = self._tail_complete_lines(path)
        if not lines:
            return False
        try:
            with path.open(encoding="utf-8", newline="") as source:
                header = [field.strip() for field in next(csv.reader(source))]
        except (OSError, StopIteration):
            return False
        if "timestamp" not in header or "index" not in header:
            return False
        samples = []
        for offset, line in lines:
            values = next(csv.reader([line]))
            if offset == 0 or (values and values[0].strip() == "timestamp"):
                continue
            normalized = normalize_nvidia_smi_row(
                dict(zip(header, values)),
                path.stat().st_mtime,
            )
            if normalized is not None:
                samples.append(normalized)
        self.store.add_gpu_samples(job_id, samples)
        if inode is not None:
            self.store.set_cursor(path, inode, new_offset)
        dirty_jobs.add(job_id)
        return True

    def _changed_file(self, path: Path) -> bool:
        if not path.is_file():
            return False
        stat = path.stat()
        marker = (stat.st_mtime_ns, stat.st_size)
        key = str(path)
        if self._mtime_cache.get(key) == marker:
            return False
        self._mtime_cache[key] = marker
        return True

    def _scan_pipeline_state(self) -> bool:
        """Publish controller progress immediately instead of every registry scan."""

        candidates = (
            self.runtime_dir / "pipeline" / "state.json",
            self.runtime_dir / "orchestrator_state.json",
        )
        path = next(
            (candidate for candidate in candidates if candidate.is_file()),
            candidates[0],
        )
        if not self._changed_file(path):
            return False
        self.store.set_meta(
            f"campaign:{self.campaign_id}:pipeline_state",
            read_json(path),
        )
        return True

    def _refresh_live_result_statuses(self) -> bool:
        """Restore current-matrix terminal progress before deep metric backfill."""

        if not self._live_job_ids:
            return False
        rows = []
        for job_id in self._live_job_ids:
            path = self.results_dir / job_id / "status.json"
            if not path.is_file():
                continue
            try:
                status = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            classification = str(status.get("classification") or "failed")
            if classification in SKIPPED_STATUSES:
                # Conditional skips are scheduler-only and must pass the exact
                # scheduler_terminal/event attestation checked elsewhere.
                continue
            rows.append(
                {
                    **status,
                    "job_id": str(status.get("job_id") or job_id),
                    "classification": classification,
                }
            )
        self.store.update_result_statuses(rows)
        return bool(rows)

    def _result_dir_is_fresh(self, result_dir: Path) -> bool:
        """A running job writes metrics/events continuously; treat the result
        tree as live only if something was modified within
        RUNNING_FRESHNESS_SECONDS.  Prevents orphaned half-finished runs (a
        rendered_run.json with no terminal status and no recent writes) from
        being shown as running."""
        newest = 0.0
        candidates = [result_dir, result_dir / "metrics"]
        try:
            attempts = result_dir / "attempts"
            if attempts.is_dir():
                candidates.extend(p for p in attempts.iterdir() if p.is_dir())
                candidates.extend(attempts.glob("*/metrics"))
        except OSError:
            pass
        for base in candidates:
            try:
                newest = max(newest, base.stat().st_mtime)
                if base.is_dir():
                    for entry in os.scandir(base):
                        try:
                            newest = max(newest, entry.stat(follow_symlinks=False).st_mtime)
                        except OSError:
                            continue
            except OSError:
                continue
        return (time.time() - newest) <= RUNNING_FRESHNESS_SECONDS

    def _scan_result_metadata(self, result_dir: Path, dirty_jobs: set[str]) -> bool:
        changed = False
        rendered_path = result_dir / "rendered_run.json"
        if self._changed_file(rendered_path):
            try:
                rendered = read_json(rendered_path)
                job = rendered["job"]
                self.store.upsert_job(self._decorate(job), phase_for_job(job), record_type="run")
                self.store.update_job(job["job_id"], gpu_mask=str(rendered.get("gpu_mask") or ""))
                # rendered_run.json is written when a run actually starts.  If no
                # terminal status.json exists yet, decide running-vs-interrupted
                # by freshness: a live run writes metrics/events continuously, so
                # a recently-touched result tree is running, while a stale one
                # (no writes for RUNNING_FRESHNESS_SECONDS) is an orphan left by a
                # killed run and must not be shown as running.
                if not (result_dir / "status.json").is_file():
                    existing = self.store.get_job(job["job_id"]) or {}
                    if str(existing.get("status") or "planned") in {"planned", "running"}:
                        fresh = self._result_dir_is_fresh(result_dir)
                        self.store.update_job(
                            job["job_id"],
                            status="running" if fresh else "interrupted",
                            **({} if fresh else {"classification": "stale_orphan_no_terminal_status"}),
                        )
                changed = True
            except (OSError, json.JSONDecodeError, KeyError):
                pass
        scheduler_terminal_path = result_dir / "scheduler_terminal.json"
        if self._changed_file(scheduler_terminal_path):
            try:
                terminal = read_json(scheduler_terminal_path)
                job_id = str(terminal.get("job_id") or result_dir.name)
                if self._valid_conditional_skip(
                    terminal, expected_job_id=result_dir.name
                ):
                    if not self.store.job_exists(job_id):
                        self.store.upsert_job(
                            self._decorate({"job_id": job_id}),
                            phase_for_job({"job_id": job_id}),
                        )
                    self.store.update_job(
                        job_id,
                        status="conditional_skipped",
                        classification="conditional_skipped",
                        finished_unix=terminal.get("time_unix") or time.time(),
                        last_error=None,
                    )
                    self._settled_result_ids.discard(job_id)
                    self._current_attempt_terminal_ids.add(job_id)
                    changed = True
            except (OSError, AttributeError, json.JSONDecodeError):
                pass
        status_path = result_dir / "status.json"
        if self._changed_file(status_path):
            try:
                status = read_json(status_path)
                job_id = str(status.get("job_id") or result_dir.name)
                classification = str(status.get("classification") or "failed")
                if classification in SKIPPED_STATUSES:
                    raise ValueError(
                        "conditional skip must come from scheduler_terminal.json"
                    )
                if not self.store.job_exists(job_id):
                    self.store.upsert_job(self._decorate({"job_id": job_id}), phase_for_job({"job_id": job_id}))
                if self._status_is_for_current_attempt(
                    job_id,
                    status,
                    status_path.stat().st_mtime,
                ):
                    fields: dict[str, Any] = {
                        "status": classification,
                        "classification": classification,
                    }
                    if classification in {"success", "oom"}:
                        fields["last_error"] = None
                    for key in (
                        "started_unix",
                        "finished_unix",
                        "wall_seconds",
                        "return_code",
                    ):
                        if status.get(key) is not None:
                            fields[key] = status[key]
                    if status.get("gpu_mask") is not None:
                        fields["gpu_mask"] = str(status["gpu_mask"])
                    self.store.update_job(job_id, **fields)
                    self._settled_result_ids.discard(job_id)
                    if classification not in {"planned", "running"}:
                        self._current_attempt_terminal_ids.add(job_id)
                    dirty_jobs.add(job_id)
                    changed = True
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        for path in sorted((result_dir / "metrics").glob("summary.rank*.json")):
            if not self._changed_file(path):
                continue
            try:
                summary = read_json(path)
                if summary.get("failure"):
                    self.store.update_job(result_dir.name, last_error=str(summary["failure"]))
                dirty_jobs.add(result_dir.name)
                changed = True
            except (OSError, json.JSONDecodeError):
                continue
        return changed

    def _scan_boundaries(self) -> bool:
        changed = False
        boundary_dir = self.results_dir / "boundary_summaries"
        for path in sorted(boundary_dir.glob("*.json")):
            if not self._changed_file(path):
                continue
            try:
                summary = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            family_id = str(summary.get("family_job_id") or path.stem)
            if not self.store.job_exists(family_id):
                self.store.upsert_job(
                    self._decorate({"job_id": family_id, "kind": "memory_boundary"}),
                    "memory",
                    record_type="family",
                )
            status = "boundary_found" if summary.get("max_feasible_mbs") is not None else "infeasible"
            self.store.update_job(
                family_id,
                status=status,
                classification=status,
                boundary_json=json.dumps(summary, ensure_ascii=False, sort_keys=True),
                gpu_mask=json.dumps(summary.get("gpu_mask") or []),
            )
            changed = True
        return changed

    def _refresh_metrics(self, job_ids: set[str]) -> None:
        for job_id in job_ids:
            job = self.store.get_job(job_id)
            if not job or job.get("record_type") != "run":
                continue
            steps = self.store.get_steps(job_id)
            metrics = aggregate_step_rows(steps, job, self.models.get(job.get("model_id")), self.hardware)
            metrics.update(aggregate_failure_rows(self.store.get_failures(job_id), metrics))
            metrics.update(
                aggregate_gpu_rows(
                    self.store.get_gpu_samples(job_id, limit=100_000),
                    self.hardware,
                )
            )
            add_clock_adjusted_mfu(metrics)
            self.store.update_metrics(job_id, metrics)

    def _pending_result_dirs(self, running_job_ids: set[str]) -> list[Path]:
        """Return the next result directories without stat-ing settled history.

        Result trees live on shared storage and can contain thousands of
        directories.  ``Path.iterdir()`` followed by ``Path.is_dir()`` and a
        second ``Path.stat()`` used to issue filesystem metadata calls for the
        entire history on every scan, even though almost every result was
        already immutable and indexed.  ``os.scandir`` lets us discard settled
        names first and only fetch mtimes for the small pending set.
        """

        candidates: list[tuple[tuple[int, int, int, str], Path]] = []
        try:
            entries = os.scandir(self.results_dir)
        except OSError:
            return []
        with entries:
            for entry in entries:
                name = entry.name
                if name == "boundary_summaries" or name in self._settled_result_ids:
                    continue
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    mtime_ns = entry.stat(follow_symlinks=False).st_mtime_ns
                except OSError:
                    continue
                if name.startswith("h800val-"):
                    phase_priority = 0
                elif name.startswith(("tput-", "tputscreen-", "tputd-")):
                    phase_priority = 1
                elif name.startswith(("prof-", "profhold-")):
                    phase_priority = 2
                elif name.startswith("scale-"):
                    phase_priority = 3
                elif name.startswith(("packmem-", "packon-", "packoff-")):
                    phase_priority = 4
                elif name.startswith("mem-"):
                    phase_priority = 5
                else:
                    phase_priority = 6
                priority = (
                    (
                        0
                        if name in running_job_ids
                        else 1
                        if self._live_job_ids is not None
                        and name in self._live_job_ids
                        else 2
                    ),
                    phase_priority,
                    -mtime_ns,
                    name,
                )
                candidates.append((priority, Path(entry.path)))
        candidates.sort(key=lambda row: row[0])
        return [path for _, path in candidates[:RESULT_SCAN_BATCH_SIZE]]

    def scan_once(self) -> dict[str, Any]:
        started = time.time()
        changed = False
        any_changed = False
        dirty_jobs: set[str] = set()
        try:
            approval_changed = self._approval_surface_changed()
            if (
                started - self._last_registry_scan >= 60
                or self._last_registry_scan == 0
                or approval_changed
            ):
                registry_changed = self._scan_registry()
                any_changed |= registry_changed
                if registry_changed:
                    # Publish a new matrix immediately. A cold historical
                    # result scan must not delay candidate visibility.
                    self.store.mark_changed()
                self._last_registry_scan = started
            changed |= self._scan_pipeline_state()
            changed |= self._scan_scheduler(dirty_jobs)
            # Scheduler events and the bounded result-directory scan below are
            # the incremental status sources.  A former periodic pass opened
            # every live job's status.json; on shared storage that made each
            # 4090 scan take 10+ minutes and prevented genuinely live updates.
            if self.results_dir.exists():
                running_job_ids = self.store.job_ids_with_status(
                    "running",
                    self.campaign_id,
                )
                running_job_ids.update(self._current_attempt_terminal_ids)
                # Repair rows must not be starved by a fixed batch of newer,
                # incomplete historical directories that never becomes
                # settled.  Treat them like active work for one scan.
                running_job_ids.update(self._repair_result_ids)
                for result_dir in self._pending_result_dirs(running_job_ids):
                    changed |= self._scan_result_metadata(result_dir, dirty_jobs)
                    changed |= self._scan_metric_events(result_dir, dirty_jobs)
                    changed |= self._scan_gpu_csv(result_dir, dirty_jobs)
                    indexed = self.store.get_job(result_dir.name)
                    terminal = bool(
                        indexed
                        and indexed.get("status") not in {"planned", "running"}
                    )
                    current_attempt_drained = (
                        result_dir.name not in self._active_attempt_started_unix
                        or result_dir.name in self._current_attempt_terminal_ids
                    )
                    if terminal and current_attempt_drained:
                        self._settled_result_ids.add(result_dir.name)
                        self._active_attempt_started_unix.pop(result_dir.name, None)
                        self._current_attempt_terminal_ids.discard(result_dir.name)
                        self._gpu_cursor_reset_attempt_ids.discard(result_dir.name)
                        self._repair_result_ids.discard(result_dir.name)
            changed |= self._scan_boundaries()
            if self._live_job_ids is not None:
                # Partial historical result directories can contain a rendered
                # job without any terminal status. Do not let those files
                # resurrect work removed from the current matrix.
                changed |= bool(
                    self.store.prune_planned_jobs(
                        self.campaign_id, self._live_job_ids
                    )
                )
            self._refresh_metrics(dirty_jobs)
            any_changed |= changed
            self.last_error = None
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.last_scan_unix = time.time()
            self.store.set_meta("last_scan_unix", self.last_scan_unix)
        if changed or dirty_jobs:
            self.store.mark_changed()
        return {
            "changed": any_changed,
            "dirty_jobs": len(dirty_jobs),
            "revision": self.store.revision(),
            "scan_seconds": time.time() - started,
        }


class MultiCampaignIngestor:
    """Scan the legacy campaign plus every isolated hardware campaign."""

    def __init__(self, store: DashboardStore, root: Path = ROOT):
        self.store = store
        self.root = root
        roots = []
        campaigns_dir = root / "campaigns"
        if campaigns_dir.is_dir():
            experiment_paths = [
                *campaigns_dir.glob("*/config/experiment.json"),
                *campaigns_dir.glob("*/*/config/experiment.json"),
            ]
            roots.extend(
                sorted(
                    {
                        experiment_path.parent.parent
                        for experiment_path in experiment_paths
                    }
                )
            )
        # Fresh hardware campaigns must become visible immediately even when the
        # legacy campaign contains hundreds of cold shared-filesystem results.
        roots.append(root)
        self.ingestors = [DashboardIngestor(store, campaign_root) for campaign_root in roots]
        scan_campaign_ids = {
            value.strip()
            for value in os.environ.get("DASHBOARD_SCAN_CAMPAIGN_IDS", "").split(",")
            if value.strip()
        }
        self.scan_ingestors = [
            ingestor
            for ingestor in self.ingestors
            if not scan_campaign_ids or ingestor.campaign_id in scan_campaign_ids
        ]
        if not self.scan_ingestors:
            raise ValueError(
                "DASHBOARD_SCAN_CAMPAIGN_IDS does not match any configured campaign"
            )
        self.last_error: str | None = None
        self.last_scan_unix = 0.0
        self._publish_campaigns()

    def _publish_campaigns(self) -> None:
        campaigns = []
        for ingestor in self.ingestors:
            campaigns.append(
                {
                    "campaign_id": ingestor.campaign_id,
                    "hardware_id": ingestor.hardware_id,
                    "gpu_type": ingestor.gpu_type,
                    "root": str(ingestor.root),
                    "gpu_ids": ingestor.experiment.get("training_scope", {}).get("gpu_ids", []),
                    "memory_total_bytes": (
                        ingestor.hardware.get("memory_bytes_reported_by_torch")
                        or ingestor.hardware.get("per_gpu", {}).get("torch_total_memory_bytes")
                    ),
                    "max_sm_clock_mhz": ingestor.hardware.get(
                        "max_sm_clock_mhz_reported_by_nvidia_smi"
                    ),
                    "healthy_busy_sm_clock_mhz": ingestor.hardware.get(
                        "healthy_busy_sm_clock_mhz"
                    ),
                    "power_limit_w": ingestor.hardware.get("power_limit_w"),
                    "attention_backend": ingestor.experiment.get("fixed_runtime", {}).get("flash_attn"),
                }
            )
        self.store.set_meta("campaigns", campaigns)

    def scan_once(self) -> dict[str, Any]:
        started = time.time()
        reports = []
        errors = []
        with ThreadPoolExecutor(max_workers=max(1, len(self.scan_ingestors))) as executor:
            futures = {
                executor.submit(ingestor.scan_once): ingestor
                for ingestor in self.scan_ingestors
            }
            for future in as_completed(futures):
                ingestor = futures[future]
                try:
                    reports.append({"campaign_id": ingestor.campaign_id, **future.result()})
                except Exception as error:
                    errors.append(f"{ingestor.campaign_id}: {type(error).__name__}: {error}")
        reports.sort(key=lambda row: str(row["campaign_id"]))
        self.last_scan_unix = time.time()
        self.last_error = "; ".join(errors) or None
        health = {
            "last_scan_unix": self.last_scan_unix,
            "last_error": self.last_error,
            "scan_seconds": time.time() - started,
            "campaigns": reports,
        }
        self.store.set_meta("ingestor_health", health)
        self._publish_campaigns()
        if errors and not reports:
            raise RuntimeError(self.last_error)
        return health
