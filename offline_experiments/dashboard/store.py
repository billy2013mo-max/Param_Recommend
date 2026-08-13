"""Disposable SQLite index for dashboard queries.

Experiment files remain authoritative.  This database can be removed and rebuilt
without changing or losing a training result.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


SKIPPED_STATUSES = {
    "conditional_skipped",
    "family_skipped",
}
FINAL_STATUSES = {
    "success",
    "oom",
    "failed",
    "incomplete_metrics",
    "resource_busy",
    "approval_rejected",
    "launcher_failed",
    "scheduler_interrupted",
    "boundary_found",
    "infeasible",
    *SKIPPED_STATUSES,
}
DASHBOARD_SCHEMA_VERSION = 5


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _decode(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class DashboardStore:
    """Thread-safe query/index layer shared by ingestion and FastAPI."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._initialize()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _initialize(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            version_row = self._conn.execute(
                "SELECT value FROM meta WHERE key='dashboard_schema_version'"
            ).fetchone()
            version = int(version_row[0]) if version_row else None
            if version != DASHBOARD_SCHEMA_VERSION:
                for table in (
                    "source_cursors", "jobs", "scheduler_events", "step_samples",
                    "failure_samples", "gpu_samples",
                ):
                    self._conn.execute(f"DROP TABLE IF EXISTS {table}")
                self._conn.execute("DELETE FROM meta")
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES('dashboard_schema_version', ?)",
                    (str(DASHBOARD_SCHEMA_VERSION),),
                )
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS source_cursors (
                    path TEXT PRIMARY KEY,
                    inode INTEGER,
                    byte_offset INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    campaign_id TEXT,
                    hardware_id TEXT,
                    gpu_type TEXT,
                    results_root TEXT,
                    parent_job_id TEXT,
                    request_id TEXT,
                    phase TEXT NOT NULL,
                    kind TEXT,
                    record_type TEXT NOT NULL DEFAULT 'run',
                    status TEXT NOT NULL DEFAULT 'planned',
                    classification TEXT,
                    model_id TEXT,
                    model_family TEXT,
                    train_type TEXT,
                    dataset_id TEXT,
                    cutoff_len INTEGER,
                    gpu_count INTEGER,
                    zero_name TEXT,
                    gc INTEGER,
                    mbs INTEGER,
                    target_gbs INTEGER,
                    packing INTEGER,
                    repeat_index INTEGER,
                    gpu_mask TEXT,
                    warmup_steps INTEGER,
                    measure_steps INTEGER,
                    max_steps INTEGER,
                    current_step INTEGER NOT NULL DEFAULT 0,
                    started_unix REAL,
                    finished_unix REAL,
                    wall_seconds REAL,
                    return_code INTEGER,
                    last_error TEXT,
                    metrics_json TEXT,
                    boundary_json TEXT,
                    raw_job_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_phase_status ON jobs(phase, status);
                CREATE INDEX IF NOT EXISTS jobs_dimensions ON jobs(model_id, train_type, dataset_id, gpu_count);
                CREATE INDEX IF NOT EXISTS jobs_request_id ON jobs(request_id);
                CREATE INDEX IF NOT EXISTS jobs_campaign_hardware ON jobs(campaign_id, hardware_id);

                CREATE TABLE IF NOT EXISTS scheduler_events (
                    event_key TEXT PRIMARY KEY,
                    campaign_id TEXT,
                    time_unix REAL,
                    event TEXT,
                    job_id TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS scheduler_events_time ON scheduler_events(time_unix);

                CREATE TABLE IF NOT EXISTS step_samples (
                    job_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    global_step INTEGER NOT NULL,
                    time_unix REAL,
                    step_seconds REAL,
                    optimizer_seconds REAL,
                    is_warmup INTEGER NOT NULL DEFAULT 0,
                    computed_tokens INTEGER NOT NULL DEFAULT 0,
                    effective_tokens INTEGER NOT NULL DEFAULT 0,
                    label_tokens INTEGER NOT NULL DEFAULT 0,
                    logical_samples INTEGER NOT NULL DEFAULT 0,
                    physical_batches INTEGER NOT NULL DEFAULT 0,
                    computed_attention_pairs INTEGER NOT NULL DEFAULT 0,
                    effective_attention_pairs INTEGER NOT NULL DEFAULT 0,
                    allocated_bytes INTEGER NOT NULL DEFAULT 0,
                    reserved_bytes INTEGER NOT NULL DEFAULT 0,
                    max_allocated_bytes INTEGER NOT NULL DEFAULT 0,
                    max_reserved_bytes INTEGER NOT NULL DEFAULT 0,
                    micro_times_json TEXT NOT NULL DEFAULT '[]',
                    PRIMARY KEY(job_id, rank, global_step)
                );
                CREATE INDEX IF NOT EXISTS step_samples_job_step ON step_samples(job_id, global_step);

                CREATE TABLE IF NOT EXISTS failure_samples (
                    job_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    time_unix REAL,
                    error TEXT,
                    allocated_bytes INTEGER NOT NULL DEFAULT 0,
                    reserved_bytes INTEGER NOT NULL DEFAULT 0,
                    max_allocated_bytes INTEGER NOT NULL DEFAULT 0,
                    max_reserved_bytes INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(job_id, rank)
                );
                CREATE INDEX IF NOT EXISTS failure_samples_job ON failure_samples(job_id);

                CREATE TABLE IF NOT EXISTS gpu_samples (
                    job_id TEXT NOT NULL,
                    timestamp_text TEXT NOT NULL,
                    time_unix REAL NOT NULL,
                    gpu_index INTEGER NOT NULL,
                    memory_used_mib REAL,
                    utilization_gpu REAL,
                    power_draw_w REAL,
                    clock_sm_mhz REAL,
                    temperature_gpu_c REAL,
                    fan_speed_percent REAL,
                    sw_thermal_slowdown_active INTEGER,
                    hw_thermal_slowdown_active INTEGER,
                    sw_power_cap_active INTEGER,
                    PRIMARY KEY(job_id, timestamp_text, gpu_index)
                );
                CREATE INDEX IF NOT EXISTS gpu_samples_job_time ON gpu_samples(job_id, time_unix);
                CREATE INDEX IF NOT EXISTS gpu_samples_gpu_time ON gpu_samples(gpu_index, time_unix);
                """
            )
            version_row = self._conn.execute(
                "SELECT value FROM meta WHERE key='dashboard_schema_version'"
            ).fetchone()
            version = int(version_row[0]) if version_row else None
            if version != DASHBOARD_SCHEMA_VERSION:
                # The index is deliberately disposable.  Rebuild it from source
                # files after a schema change instead of attempting lossy migrations.
                for table in (
                    "source_cursors", "jobs", "scheduler_events", "step_samples",
                    "failure_samples", "gpu_samples",
                ):
                    self._conn.execute(f"DROP TABLE IF EXISTS {table}")
                self._conn.execute("DELETE FROM meta")
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES('dashboard_schema_version', ?)",
                    (str(DASHBOARD_SCHEMA_VERSION),),
                )
                self._conn.execute("INSERT INTO meta(key, value) VALUES('revision', '0')")
                self._conn.execute("INSERT INTO meta(key, value) VALUES('last_scan_unix', '0')")
                self._initialize()
                return
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('dashboard_schema_version', ?)",
                (str(DASHBOARD_SCHEMA_VERSION),),
            )
            self._conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('revision', '0')")
            self._conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('last_scan_unix', '0')")

    def revision(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key='revision'").fetchone()
            return int(row[0]) if row else 0

    def mark_changed(self) -> int:
        with self._lock, self._conn:
            revision = self.revision() + 1
            self._conn.execute("UPDATE meta SET value=? WHERE key='revision'", (str(revision),))
            self._conn.execute("UPDATE meta SET value=? WHERE key='last_scan_unix'", (str(time.time()),))
            return revision

    def set_meta(self, key: str, value: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, _json(value)),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            if not row:
                return default
            return _decode(row[0], row[0])

    def get_cursor(self, path: Path) -> tuple[int | None, int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT inode, byte_offset FROM source_cursors WHERE path=?", (str(path),)
            ).fetchone()
            return (int(row[0]), int(row[1])) if row else (None, 0)

    def set_cursor(self, path: Path, inode: int, offset: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO source_cursors(path, inode, byte_offset, updated_at) VALUES(?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET inode=excluded.inode,
                    byte_offset=excluded.byte_offset, updated_at=excluded.updated_at
                """,
                (str(path), inode, offset, time.time()),
            )

    def upsert_job(
        self,
        job: dict[str, Any],
        phase: str,
        record_type: str = "run",
        status: str = "planned",
    ) -> None:
        self.upsert_jobs([(job, phase, record_type, status)])

    @staticmethod
    def _job_values(
        job: dict[str, Any], phase: str, record_type: str, status: str
    ) -> dict[str, Any]:
        job_id = str(job["job_id"])
        max_steps = job.get("max_steps")
        if max_steps is None and job.get("measure_steps") is not None:
            max_steps = int(job.get("warmup_steps", 0)) + int(job["measure_steps"])
        return {
            "job_id": job_id,
            "campaign_id": job.get("campaign_id"),
            "hardware_id": job.get("hardware_id"),
            "gpu_type": job.get("gpu_type"),
            "results_root": job.get("results_root"),
            "parent_job_id": job.get("family_job_id"),
            "request_id": job.get("request_id"),
            "phase": phase,
            "kind": job.get("kind"),
            "record_type": record_type,
            "status": status,
            "model_id": job.get("model_id"),
            "model_family": job.get("model_family"),
            "train_type": job.get("train_type"),
            "dataset_id": job.get("dataset_id"),
            "cutoff_len": job.get("cutoff_len"),
            "gpu_count": job.get("gpu_count"),
            "zero_name": job.get("zero"),
            "gc": int(bool(job.get("gc"))) if job.get("gc") is not None else None,
            "mbs": job.get("mbs"),
            "target_gbs": job.get("target_gbs"),
            "packing": int(bool(job.get("packing"))) if job.get("packing") is not None else None,
            "repeat_index": job.get("repeat"),
            "warmup_steps": job.get("warmup_steps"),
            "measure_steps": job.get("measure_steps"),
            "max_steps": max_steps,
            "raw_job_json": _json(job),
            "updated_at": time.time(),
        }

    def upsert_jobs(self, rows: list[tuple[dict[str, Any], str, str, str]]) -> None:
        if not rows:
            return
        values = [self._job_values(*row) for row in rows]
        columns = ", ".join(values[0])
        placeholders = ", ".join("?" for _ in values[0])
        update_columns = [key for key in values[0] if key not in {"job_id", "status"}]
        updates = ", ".join(f"{key}=COALESCE(excluded.{key}, jobs.{key})" for key in update_columns)
        with self._lock, self._conn:
            self._conn.executemany(
                f"""
                INSERT INTO jobs({columns}) VALUES({placeholders})
                ON CONFLICT(job_id) DO UPDATE SET {updates},
                    status=CASE WHEN jobs.status != 'planned'
                        THEN jobs.status ELSE excluded.status END
                """,
                [tuple(row.values()) for row in values],
            )

    def prune_planned_jobs(self, campaign_id: str, live_job_ids: set[str]) -> int:
        """Remove obsolete registry entries without touching executed results.

        Matrix files are authoritative for work that has not started. Completed,
        failed and running rows remain indexed even after a shortlist replaces
        the larger screening matrix.
        """

        with self._lock, self._conn:
            planned = {
                str(row[0])
                for row in self._conn.execute(
                    "SELECT job_id FROM jobs WHERE campaign_id=? AND status='planned'",
                    (campaign_id,),
                ).fetchall()
            }
            stale = sorted(planned - live_job_ids)
            for start in range(0, len(stale), 500):
                chunk = stale[start : start + 500]
                placeholders = ", ".join("?" for _ in chunk)
                self._conn.execute(
                    f"DELETE FROM jobs WHERE campaign_id=? AND status='planned' "
                    f"AND job_id IN ({placeholders})",
                    (campaign_id, *chunk),
                )
        return len(stale)

    def finalize_stale_running_jobs(
        self, campaign_id: str, scheduler_finished_unix: float
    ) -> int:
        """Close jobs left running after their scheduler execution completed."""

        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE jobs
                SET status='failed',
                    classification='scheduler_interrupted',
                    finished_unix=?,
                    last_error='Scheduler completed without a terminal trial event',
                    updated_at=?
                WHERE campaign_id=?
                  AND status='running'
                  AND COALESCE(started_unix, 0) <= ?
                """,
                (
                    scheduler_finished_unix,
                    time.time(),
                    campaign_id,
                    scheduler_finished_unix,
                ),
            )
        return int(cursor.rowcount)

    def update_job(self, job_id: str, **fields: Any) -> None:
        allowed = {
            "campaign_id", "hardware_id", "gpu_type", "results_root",
            "parent_job_id", "request_id", "phase", "kind", "record_type", "status", "classification",
            "gpu_mask", "current_step", "started_unix", "finished_unix", "wall_seconds", "return_code",
            "last_error", "metrics_json", "boundary_json", "updated_at",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unsupported job fields: {sorted(unknown)}")
        fields.setdefault("updated_at", time.time())
        assignments = ", ".join(f"{key}=?" for key in fields)
        with self._lock, self._conn:
            self._conn.execute(f"UPDATE jobs SET {assignments} WHERE job_id=?", (*fields.values(), job_id))

    def update_result_statuses(self, rows: list[dict[str, Any]]) -> None:
        """Apply a cold-start status pass in one transaction."""

        if not rows:
            return
        now = time.time()
        with self._lock, self._conn:
            self._conn.executemany(
                """
                UPDATE jobs
                SET status=?,
                    classification=?,
                    started_unix=?,
                    finished_unix=?,
                    wall_seconds=?,
                    return_code=?,
                    gpu_mask=?,
                    updated_at=?
                WHERE job_id=?
                  AND status != 'running'
                """,
                [
                    (
                        row["classification"],
                        row["classification"],
                        row.get("started_unix"),
                        row.get("finished_unix"),
                        row.get("wall_seconds"),
                        row.get("return_code"),
                        str(row.get("gpu_mask") or ""),
                        now,
                        row["job_id"],
                    )
                    for row in rows
                ],
            )

    def add_scheduler_event(self, event_key: str, event: dict[str, Any]) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO scheduler_events VALUES(?, ?, ?, ?, ?, ?)",
                (
                    event_key,
                    event.get("campaign_id"),
                    event.get("time_unix"),
                    event.get("event"),
                    event.get("job_id"),
                    _json(event),
                ),
            )
        return bool(cursor.rowcount)

    def add_step(self, job_id: str, rank: int, event: dict[str, Any]) -> None:
        tokens = event.get("tokens") or {}
        memory = event.get("memory") or {}
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO step_samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(job_id, rank, global_step) DO UPDATE SET
                    time_unix=excluded.time_unix, step_seconds=excluded.step_seconds,
                    optimizer_seconds=excluded.optimizer_seconds, is_warmup=excluded.is_warmup,
                    computed_tokens=excluded.computed_tokens, effective_tokens=excluded.effective_tokens,
                    label_tokens=excluded.label_tokens, logical_samples=excluded.logical_samples,
                    physical_batches=excluded.physical_batches,
                    computed_attention_pairs=excluded.computed_attention_pairs,
                    effective_attention_pairs=excluded.effective_attention_pairs,
                    allocated_bytes=excluded.allocated_bytes, reserved_bytes=excluded.reserved_bytes,
                    max_allocated_bytes=excluded.max_allocated_bytes,
                    max_reserved_bytes=excluded.max_reserved_bytes,
                    micro_times_json=excluded.micro_times_json
                """,
                (
                    job_id,
                    rank,
                    int(event.get("global_step", 0)),
                    event.get("time_unix"),
                    event.get("step_seconds"),
                    event.get("optimizer_step_seconds"),
                    int(bool(event.get("is_warmup"))),
                    int(tokens.get("computed_tokens", 0)),
                    int(tokens.get("effective_tokens", 0)),
                    int(tokens.get("label_tokens", 0)),
                    int(tokens.get("logical_samples", 0)),
                    int(tokens.get("physical_batches", 0)),
                    int(tokens.get("computed_attention_token_pairs", 0)),
                    int(tokens.get("effective_attention_token_pairs", 0)),
                    int(memory.get("allocated", 0)),
                    int(memory.get("reserved", 0)),
                    int(memory.get("max_allocated", 0)),
                    int(memory.get("max_reserved", 0)),
                    _json(event.get("micro_step_seconds") or []),
                ),
            )
            self._conn.execute(
                "UPDATE jobs SET current_step=MAX(current_step, ?), updated_at=? WHERE job_id=?",
                (int(event.get("global_step", 0)), time.time(), job_id),
            )

    def reset_job_samples(
        self,
        job_id: str,
        rank: int | None = None,
        *,
        include_gpu: bool = False,
    ) -> None:
        """Discard disposable dashboard samples from an earlier retry.

        Result event files are append-only, while a retried job deliberately
        keeps the same stable job ID.  Resetting at the new trial/train boundary
        prevents old high step numbers, failures, and GPU peaks from leaking
        into the latest attempt.
        """

        with self._lock, self._conn:
            if rank is None:
                self._conn.execute("DELETE FROM step_samples WHERE job_id=?", (job_id,))
                self._conn.execute("DELETE FROM failure_samples WHERE job_id=?", (job_id,))
            else:
                self._conn.execute(
                    "DELETE FROM step_samples WHERE job_id=? AND rank=?",
                    (job_id, rank),
                )
                self._conn.execute(
                    "DELETE FROM failure_samples WHERE job_id=? AND rank=?",
                    (job_id, rank),
                )
            if include_gpu:
                self._conn.execute("DELETE FROM gpu_samples WHERE job_id=?", (job_id,))
            current = self._conn.execute(
                "SELECT COALESCE(MAX(global_step), 0) FROM step_samples WHERE job_id=?",
                (job_id,),
            ).fetchone()
            self._conn.execute(
                """
                UPDATE jobs
                SET current_step=?, metrics_json='{}', last_error=NULL, updated_at=?
                WHERE job_id=?
                """,
                (int(current[0]) if current else 0, time.time(), job_id),
            )

    def add_gpu_sample(self, job_id: str, sample: dict[str, Any]) -> None:
        self.add_gpu_samples(job_id, [sample])

    def add_failure(self, job_id: str, rank: int, event: dict[str, Any]) -> None:
        memory = event.get("memory") or {}
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO failure_samples VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(job_id, rank) DO UPDATE SET
                    time_unix=excluded.time_unix, error=excluded.error,
                    allocated_bytes=excluded.allocated_bytes,
                    reserved_bytes=excluded.reserved_bytes,
                    max_allocated_bytes=excluded.max_allocated_bytes,
                    max_reserved_bytes=excluded.max_reserved_bytes
                """,
                (
                    job_id,
                    rank,
                    event.get("time_unix"),
                    str(event.get("error") or "training failure"),
                    int(memory.get("allocated", 0)),
                    int(memory.get("reserved", 0)),
                    int(memory.get("max_allocated", 0)),
                    int(memory.get("max_reserved", 0)),
                ),
            )

    def add_gpu_samples(self, job_id: str, samples: list[dict[str, Any]]) -> None:
        if not samples:
            return
        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO gpu_samples(
                    job_id, timestamp_text, time_unix, gpu_index,
                    memory_used_mib, utilization_gpu, power_draw_w, clock_sm_mhz,
                    temperature_gpu_c, fan_speed_percent,
                    sw_thermal_slowdown_active, hw_thermal_slowdown_active,
                    sw_power_cap_active
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        job_id,
                        sample["timestamp_text"],
                        float(sample["time_unix"]),
                        int(sample["gpu_index"]),
                        sample.get("memory_used_mib"),
                        sample.get("utilization_gpu"),
                        sample.get("power_draw_w"),
                        sample.get("clock_sm_mhz"),
                        sample.get("temperature_gpu_c"),
                        sample.get("fan_speed_percent"),
                        sample.get("sw_thermal_slowdown_active"),
                        sample.get("hw_thermal_slowdown_active"),
                        sample.get("sw_power_cap_active"),
                    )
                    for sample in samples
                ],
            )

    def job_exists(self, job_id: str) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1 FROM jobs WHERE job_id=?", (job_id,)).fetchone() is not None

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for field, default in (("raw_job_json", {}), ("metrics_json", {}), ("boundary_json", {})):
            result[field.removesuffix("_json")] = _decode(result.pop(field, None), default)
        for field in ("gc", "packing"):
            if result.get(field) is not None:
                result[field] = bool(result[field])
        return result

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._row(self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())

    def list_jobs(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        filters = filters or {}
        clauses = ["1=1"]
        values: list[Any] = []
        column_map = {
            "campaign_id": "campaign_id", "hardware_id": "hardware_id",
            "phase": "phase", "status": "status", "model_id": "model_id", "train_type": "train_type",
            "dataset_id": "dataset_id", "gpu_count": "gpu_count", "zero": "zero_name",
            "packing": "packing", "record_type": "record_type",
        }
        for key, column in column_map.items():
            value = filters.get(key)
            if value is None or value == "":
                continue
            clauses.append(f"{column}=?")
            if key == "packing":
                value = int(str(value).lower() in {"1", "true", "yes", "on"})
            values.append(value)
        if filters.get("q"):
            clauses.append("(job_id LIKE ? OR model_id LIKE ? OR dataset_id LIKE ?)")
            needle = f"%{filters['q']}%"
            values.extend([needle, needle, needle])
        where = " AND ".join(clauses)
        with self._lock:
            total = int(self._conn.execute(f"SELECT COUNT(*) FROM jobs WHERE {where}", values).fetchone()[0])
            rows = self._conn.execute(
                f"""SELECT * FROM jobs WHERE {where}
                ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'failed' THEN 1 WHEN 'oom' THEN 2 ELSE 3 END,
                    COALESCE(started_unix, updated_at) DESC LIMIT ? OFFSET ?""",
                (*values, min(max(limit, 1), 1000), max(offset, 0)),
            ).fetchall()
        return [self._row(row) for row in rows if row is not None], total

    def all_jobs(
        self,
        phase: str | None = None,
        record_type: str | None = None,
        campaign_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        values: list[Any] = []
        if phase:
            clauses.append("phase=?")
            values.append(phase)
        if record_type:
            clauses.append("record_type=?")
            values.append(record_type)
        if campaign_id:
            clauses.append("campaign_id=?")
            values.append(campaign_id)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM jobs WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC", values
            ).fetchall()
        return [self._row(row) for row in rows if row is not None]

    def job_ids_with_status(
        self,
        status: str,
        campaign_id: str | None = None,
    ) -> set[str]:
        clauses = ["status=?"]
        values: list[Any] = [status]
        if campaign_id:
            clauses.append("campaign_id=?")
            values.append(campaign_id)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT job_id FROM jobs WHERE {' AND '.join(clauses)}",
                values,
            ).fetchall()
        return {str(row["job_id"]) for row in rows}

    def get_steps(self, job_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM step_samples WHERE job_id=? ORDER BY global_step, rank"
        values: list[Any] = [job_id]
        if limit:
            sql = """SELECT * FROM step_samples WHERE job_id=? AND global_step >=
                MAX(0, (SELECT MAX(global_step) FROM step_samples WHERE job_id=?)-?)
                ORDER BY global_step, rank"""
            values = [job_id, job_id, limit]
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, values).fetchall()]

    def get_gpu_samples(self, job_id: str, limit: int = 1200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM (SELECT * FROM gpu_samples WHERE job_id=? ORDER BY time_unix DESC LIMIT ?)
                ORDER BY time_unix, gpu_index""",
                (job_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_failures(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM failure_samples WHERE job_id=? ORDER BY rank", (job_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_gpus(self, campaign_id: str | None = None) -> list[dict[str, Any]]:
        campaign_clause = "AND j.campaign_id=?" if campaign_id else ""
        values: tuple[Any, ...] = (campaign_id,) if campaign_id else ()
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT g.*, j.campaign_id, j.hardware_id, j.gpu_type, j.status,
                    j.model_id, j.dataset_id, j.current_step, j.max_steps
                FROM gpu_samples g
                JOIN (
                    SELECT j2.campaign_id, g2.gpu_index, MAX(g2.time_unix) AS latest
                    FROM gpu_samples g2 JOIN jobs j2 ON j2.job_id=g2.job_id
                    GROUP BY j2.campaign_id, g2.gpu_index
                ) x ON x.gpu_index=g.gpu_index AND x.latest=g.time_unix
                LEFT JOIN jobs j ON j.job_id=g.job_id
                WHERE x.campaign_id=j.campaign_id
                {campaign_clause}
                ORDER BY g.gpu_index
                """,
                values,
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_scheduler_events(
        self, limit: int = 50, campaign_id: str | None = None
    ) -> list[dict[str, Any]]:
        where = "WHERE campaign_id=?" if campaign_id else ""
        values: tuple[Any, ...] = (campaign_id, limit) if campaign_id else (limit,)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT payload_json FROM scheduler_events {where} ORDER BY time_unix DESC LIMIT ?",
                values,
            ).fetchall()
        return [_decode(row[0], {}) for row in rows]

    def counts(self, campaign_id: str | None = None) -> dict[str, Any]:
        campaign_clause = " AND campaign_id=?" if campaign_id else ""
        values: tuple[Any, ...] = (campaign_id,) if campaign_id else ()
        with self._lock:
            status_rows = self._conn.execute(
                f"""SELECT status, COUNT(*) AS count FROM jobs
                WHERE record_type='run'{campaign_clause} GROUP BY status""",
                values,
            ).fetchall()
            phase_rows = self._conn.execute(
                f"""SELECT phase, status, record_type, COUNT(*) AS count FROM jobs
                WHERE 1=1{campaign_clause} GROUP BY phase, status, record_type""",
                values,
            ).fetchall()
        return {
            "runs_by_status": {row["status"]: row["count"] for row in status_rows},
            "phase_rows": [dict(row) for row in phase_rows],
        }

    def update_metrics(self, job_id: str, metrics: dict[str, Any]) -> None:
        self.update_job(job_id, metrics_json=_json(metrics), current_step=int(metrics.get("current_step", 0)))

    def log_tail(self, job_id: str, results_root: Path | None = None, max_bytes: int = 64_000) -> str:
        job = self.get_job(job_id)
        if not job:
            return ""
        results_root = Path(job.get("results_root") or results_root or "")
        root = results_root.resolve()
        path = (root / job_id / "train.log").resolve()
        if root not in path.parents or not path.is_file():
            return ""
        with path.open("rb") as source:
            source.seek(max(0, path.stat().st_size - max_bytes))
            return source.read().decode("utf-8", errors="replace")
