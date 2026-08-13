"""FastAPI application for the live, read-only experiment dashboard."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from scripts.common import ROOT
from scripts.gpu_telemetry import aggregate_gpu_telemetry

from .analytics import DashboardAnalytics
from .ingest import MultiCampaignIngestor
from .store import DashboardStore


DASHBOARD_DIR = Path(__file__).resolve().parent


def compact_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key not in {"raw_job", "boundary"}}


LIMITER_FLAG_KEYS = (
    "sw_power_cap_active",
    "sw_thermal_slowdown_active",
    "hw_thermal_slowdown_active",
)


def _limiter_flag_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key in LIMITER_FLAG_KEYS:
        known = [int(row[key]) for row in rows if row.get(key) is not None]
        active = sum(known)
        result[key] = {
            "known_samples": len(known),
            "active_samples": active,
            "active_fraction": active / len(known) if known else None,
            "coverage_fraction": len(known) / len(rows) if rows else None,
        }
    return result


def _limiter_evidence(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    busy_rows = [
        row for row in rows if float(row.get("utilization_gpu") or 0) >= 90.0
    ]
    hardware = {
        "power_limit_w": metrics.get("power_limit_w"),
        "max_sm_clock_mhz_reported_by_nvidia_smi": metrics.get(
            "sm_clock_spec_max_mhz"
        ),
        "healthy_busy_sm_clock_mhz": metrics.get("sm_clock_reference_mhz"),
    }
    per_gpu = []
    for gpu_index in sorted({int(row["gpu_index"]) for row in rows}):
        gpu_rows = [row for row in rows if int(row["gpu_index"]) == gpu_index]
        gpu_busy_rows = [
            row
            for row in gpu_rows
            if float(row.get("utilization_gpu") or 0) >= 90.0
        ]
        aggregate = aggregate_gpu_telemetry(gpu_rows, hardware)
        per_gpu.append(
            {
                "gpu_index": gpu_index,
                "telemetry_samples": len(gpu_rows),
                "busy_samples": len(gpu_busy_rows),
                "flags": _limiter_flag_counts(gpu_busy_rows),
                **{
                    key: aggregate.get(key)
                    for key in (
                        "temperature_max_c",
                        "busy_temperature_p95_c",
                        "busy_clock_p5_mhz",
                        "busy_clock_p50_mhz",
                        "busy_clock_p95_mhz",
                        "busy_power_p50_w",
                        "busy_power_p95_w",
                        "power_limit_busy_fraction",
                        "sw_power_cap_busy_fraction",
                        "sw_thermal_slowdown_busy_fraction",
                        "hw_thermal_slowdown_busy_fraction",
                        "throttle_reason_data_available",
                        "throttle_reason_data_complete",
                        "clock_status",
                        "clock_status_source",
                    )
                },
            }
        )
    flags = _limiter_flag_counts(busy_rows)
    return {
        "telemetry_samples": len(rows),
        "busy_samples": len(busy_rows),
        "busy_utilization_threshold_percent": 90.0,
        "flags": flags,
        "reason_coverage_complete": bool(busy_rows)
        and all(
            flag["known_samples"] == len(busy_rows) for flag in flags.values()
        ),
        "per_gpu": per_gpu,
    }


def _downsample_gpu_rows(
    rows: list[dict[str, Any]], max_points: int = 2400
) -> list[dict[str, Any]]:
    if len(rows) <= max_points:
        return rows
    by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_gpu[int(row["gpu_index"])].append(row)
    budget = max(1, max_points // max(1, len(by_gpu)))
    sampled: list[dict[str, Any]] = []
    for gpu_rows in by_gpu.values():
        stride = max(1, (len(gpu_rows) + budget - 1) // budget)
        selected = gpu_rows[::stride]
        if selected and selected[-1] is not gpu_rows[-1]:
            selected.append(gpu_rows[-1])
        sampled.extend(selected)
    return sorted(sampled, key=lambda row: (float(row["time_unix"]), int(row["gpu_index"])))


def job_series(store: DashboardStore, job_id: str) -> dict[str, Any]:
    steps = store.get_steps(job_id, limit=500)
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in steps:
        by_step[int(row["global_step"])].append(row)
    step_series = []
    for step, ranks in sorted(by_step.items()):
        seconds = max(float(row.get("step_seconds") or 0) for row in ranks)
        computed = sum(int(row.get("computed_tokens") or 0) for row in ranks)
        effective = sum(int(row.get("effective_tokens") or 0) for row in ranks)
        samples = sum(int(row.get("logical_samples") or 0) for row in ranks)
        step_series.append(
            {
                "step": step,
                "time_unix": max(float(row.get("time_unix") or 0) for row in ranks),
                "step_seconds": seconds,
                "computed_tokens_per_second": computed / seconds if seconds else None,
                "effective_tokens_per_second": effective / seconds if seconds else None,
                "samples_per_second": samples / seconds if seconds else None,
                "max_allocated_gib": max(int(row.get("max_allocated_bytes") or 0) for row in ranks) / 1024**3,
                "is_warmup": all(bool(row.get("is_warmup")) for row in ranks),
            }
        )
    job = store.get_job(job_id) or {}
    gpu_rows = store.get_gpu_samples(job_id, limit=100_000)
    gpu_series = _downsample_gpu_rows(gpu_rows)
    return {
        "job_id": job_id,
        "steps": step_series,
        "gpus": gpu_series,
        "limiter_evidence": _limiter_evidence(
            gpu_rows,
            job.get("metrics") or {},
        ),
        "failures": store.get_failures(job_id),
    }


def create_app(
    root: Path = ROOT,
    db_path: Path | None = None,
    scan_interval: float | None = None,
) -> FastAPI:
    root = Path(root).resolve()
    db_path = db_path or Path(
        os.environ.get("DASHBOARD_DB_PATH", root / "runtime" / "dashboard" / "dashboard.sqlite")
    )
    scan_interval = float(scan_interval if scan_interval is not None else os.environ.get("DASHBOARD_SCAN_INTERVAL", "2"))
    store = DashboardStore(db_path)
    ingestor = MultiCampaignIngestor(store, root=root)
    analytics = DashboardAnalytics(store)

    async def ingest_loop() -> None:
        while True:
            try:
                await asyncio.to_thread(ingestor.scan_once)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Health endpoint exposes the exact ingestion error.  A malformed
                # partial file must not terminate the display service.
                pass
            await asyncio.sleep(max(0.5, scan_interval))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Always expose the HTTP service immediately. A first multi-campaign
        # rebuild may traverse hundreds of historical result directories, so it
        # belongs in the background just like subsequent incremental scans.
        task = asyncio.create_task(ingest_loop(), name="dashboard-ingestor")
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            store.close()

    app = FastAPI(
        title="Multi-GPU SFT Experiment Dashboard",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.dashboard_store = store
    app.state.dashboard_ingestor = ingestor
    app.state.dashboard_analytics = analytics
    templates = Jinja2Templates(directory=DASHBOARD_DIR / "templates")
    app.mount("/static", StaticFiles(directory=DASHBOARD_DIR / "static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"app_version": app.version, "root": str(root)},
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        health, revision = await asyncio.gather(
            asyncio.to_thread(store.get_meta, "ingestor_health", {}),
            asyncio.to_thread(store.revision),
        )
        stale = time.time() - float(health.get("last_scan_unix") or 0)
        stale_limit = max(
            300.0,
            scan_interval * 5,
            float(health.get("scan_seconds") or 0) * 2.5,
        )
        healthy = not health.get("last_error") and stale <= stale_limit
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={"status": "ok" if healthy else "degraded", "revision": revision, **health},
        )

    @app.get("/api/v1/overview")
    async def overview(campaign_id: str | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(analytics.overview, campaign_id)

    @app.get("/api/v1/campaigns")
    async def campaigns() -> dict[str, Any]:
        rows = await asyncio.to_thread(store.get_meta, "campaigns", [])
        return {"rows": rows}

    @app.get("/api/v1/gpus")
    async def gpus(campaign_id: str | None = None) -> dict[str, Any]:
        snapshot = await asyncio.to_thread(analytics.overview, campaign_id)
        return {"rows": snapshot["gpus"]}

    @app.get("/api/v1/runs")
    async def runs(
        phase: str | None = None,
        status: str | None = None,
        model_id: str | None = None,
        train_type: str | None = None,
        dataset_id: str | None = None,
        gpu_count: int | None = None,
        zero: str | None = None,
        packing: str | None = None,
        record_type: str | None = "run",
        q: str | None = None,
        campaign_id: str | None = None,
        hardware_id: str | None = None,
        limit: int = Query(200, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        filters = {
            "phase": phase, "status": status, "model_id": model_id, "train_type": train_type,
            "dataset_id": dataset_id, "gpu_count": gpu_count, "zero": zero, "packing": packing,
            "record_type": record_type, "q": q,
            "campaign_id": campaign_id, "hardware_id": hardware_id,
        }
        rows, total = await asyncio.to_thread(
            store.list_jobs, filters, limit, offset
        )
        return {"rows": [compact_job(row) for row in rows], "total": total, "limit": limit, "offset": offset}

    @app.get("/api/v1/runs/{job_id}")
    async def run_detail(job_id: str) -> dict[str, Any]:
        job = await asyncio.to_thread(store.get_job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        children = []
        if job.get("record_type") == "family":
            child_rows, _ = await asyncio.to_thread(
                store.list_jobs,
                {"q": f"{job_id}-mbs", "record_type": "run"},
                100,
                0,
            )
            children = [compact_job(row) for row in child_rows if row.get("parent_job_id") == job_id or row["job_id"].startswith(job_id)]
        return {"job": job, "children": children, "series_summary": job.get("metrics") or {}}

    @app.get("/api/v1/runs/{job_id}/series")
    async def run_series(job_id: str) -> dict[str, Any]:
        if not await asyncio.to_thread(store.job_exists, job_id):
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return await asyncio.to_thread(job_series, store, job_id)

    @app.get("/api/v1/runs/{job_id}/log", response_class=PlainTextResponse)
    async def run_log(job_id: str) -> PlainTextResponse:
        if not await asyncio.to_thread(store.job_exists, job_id):
            raise HTTPException(status_code=404, detail="Unknown job_id")
        content = await asyncio.to_thread(store.log_tail, job_id, root / "results")
        return PlainTextResponse(content)

    @app.get("/api/v1/analysis/memory")
    async def memory_analysis(campaign_id: str | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(analytics.memory, campaign_id)

    @app.get("/api/v1/analysis/throughput")
    async def throughput_analysis(campaign_id: str | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(
            analytics.throughput, campaign_id=campaign_id
        )

    @app.get("/api/v1/analysis/scaling")
    async def scaling_analysis(campaign_id: str | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(analytics.scaling, campaign_id)

    @app.get("/api/v1/analysis/packing")
    async def packing_analysis(campaign_id: str | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(analytics.packing, campaign_id)

    @app.get("/api/v1/recommendations")
    async def recommendations(campaign_id: str | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(analytics.recommendations, campaign_id)

    @app.get("/api/v1/system")
    async def system() -> dict[str, Any]:
        def snapshot() -> dict[str, Any]:
            return {
                "root": str(root),
                "campaigns": [
                    {
                        **campaign,
                        "design": store.get_meta(f"campaign:{campaign['campaign_id']}:design_summary", {}),
                        "experiment": store.get_meta(f"campaign:{campaign['campaign_id']}:experiment", {}),
                        "preflight": store.get_meta(f"campaign:{campaign['campaign_id']}:preflight", {}),
                        "pipeline_state": store.get_meta(
                            f"campaign:{campaign['campaign_id']}:pipeline_state", {}
                        ),
                    }
                    for campaign in store.get_meta("campaigns", [])
                ],
                "ingestor": store.get_meta("ingestor_health", {}),
                "database": str(db_path),
                "read_only_experiment_access": True,
            }

        return await asyncio.to_thread(snapshot)

    @app.get("/api/v1/events")
    async def events(request: Request) -> StreamingResponse:
        async def stream() -> AsyncIterator[str]:
            previous = -1
            last_heartbeat = 0.0
            while not await request.is_disconnected():
                revision = await asyncio.to_thread(store.revision)
                now = time.time()
                if revision != previous:
                    # Keep SSE fan-out tiny.  Sending a multi-megabyte overview
                    # per client made reconnect storms monopolize the event
                    # loop; clients fetch the latest snapshot once, debounced.
                    payload = {"revision": revision}
                    yield f"event: snapshot\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    previous = revision
                    last_heartbeat = now
                elif now - last_heartbeat >= 15:
                    yield f"event: heartbeat\ndata: {{\"time_unix\": {now}}}\n\n"
                    last_heartbeat = now
                await asyncio.sleep(1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    return app


app = create_app()
