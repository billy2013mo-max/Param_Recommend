#!/usr/bin/env python3
"""Evaluate the H800 memory/throughput predictors on the business-data blind
queue: compare predicted memory/throughput against the observed outcomes.

Metrics mirror the production gates: safe admission rate, OOM admission rate,
memory MAPE, and throughput ranking accuracy, split by model family and by
packing.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from common import ARTIFACT_DIR, MATRIX_DIR, read_json, read_jsonl
from h800_unified_v3_v4b_predictor import H800UnifiedV3V4BPredictor

QUEUE = MATRIX_DIR / "h800_business_blind_v1.jsonl"
RESULTS_DIR = ARTIFACT_DIR.parent / "results"
SAFE_LIMIT = 150142189568 * 0.95  # 0.95 x H800 capacity
INVENTORY = (
    ARTIFACT_DIR / "h800_bounded_memory_v2_model_inventory_with_hybrid_v1.json"
)


def _candidate(job: dict) -> dict:
    zero = {"none": 0, "zero2": 2, "zero3": 3}[job["zero"]]
    return {
        "request_id": job["job_id"],
        "comparison_group": f"blind-{job['job_id']}",
        "hardware_id": "h800",
        "model_id": job["model_id"],
        "training_mode": "lora",
        "dataset_id": "short_512",
        "dataset_category": "short",
        "target_gbs": job["target_gbs"],
        "cutoff_len": job["cutoff_len"],
        "gpu_count": job["gpu_count"],
        "physical_mbs": job["mbs"],
        "zero_stage": zero,
        "gradient_checkpointing": job["gc"],
        "packing": bool(job["packing"]),
    }


def _observed(job_id: str, results_dir: Path = RESULTS_DIR) -> dict | None:
    d = results_dir / job_id
    status = d / "status.json"
    if not status.is_file():
        return None
    s = read_json(status)
    cls = s.get("classification")
    if cls != "success":
        return {"classification": cls}
    summary = d / "metrics" / "summary.rank0.json"
    if not summary.is_file():
        return None
    m = read_json(summary)
    peak = None
    for field in ("max_reserved_bytes", "peak_reserved_bytes", "max_reserved"):
        if m.get(field):
            peak = float(m[field])
            break
    return {
        "classification": "success",
        "peak_reserved_bytes": peak,
        "tokens_per_second": float(m.get("effective_tokens_per_second") or 0),
    }


def evaluate(
    queue_path: Path = QUEUE,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    predictor = H800UnifiedV3V4BPredictor(
        memory_feature_inventory=str(INVENTORY),
    )
    jobs = read_jsonl(queue_path)
    candidates = [_candidate(job) for job in jobs]
    report = predictor.predict(candidates)
    by_id = {r["request_id"]: r for r in report["predictions"]}

    rows = []
    for job in jobs:
        jid = job["job_id"]
        pred = by_id[jid]
        obs = _observed(jid, results_dir)
        if not obs:
            continue
        mem = pred["memory"]
        rows.append(
            {
                "job_id": jid,
                "model_id": job["model_id"],
                "packing": bool(job["packing"]),
                "zero": job["zero"],
                "gpu": job["gpu_count"],
                "gc": job["gc"],
                "cutoff": job["cutoff_len"],
                "actual_class": obs.get("classification"),
                "actual_peak_gb": (obs.get("peak_reserved_bytes") or 0) / 1e9,
                "pred_center_gb": (mem.get("reserved_center_bytes") or 0) / 1e9,
                "pred_upper_gb": (mem.get("admission_upper_reserved_bytes") or 0) / 1e9,
                "pred_admitted": bool(mem.get("admitted")),
                "pred_available": bool(mem.get("prediction_available")),
            }
        )

    success = [r for r in rows if r["actual_class"] == "success"]
    admitted = [r for r in success if r["pred_admitted"]]
    # safe admission: predicted-admitted AND actually safe (no OOM)
    safe_admission_rate = len(admitted) / len(success) if success else None
    oom = [r for r in rows if r["actual_class"] == "oom"]
    oom_admitted = [r for r in oom if r["pred_admitted"]]
    # memory MAPE on predicted centers vs actual peaks
    mape = (
        statistics.fmean(
            abs(r["pred_center_gb"] - r["actual_peak_gb"]) / r["actual_peak_gb"]
            for r in success
            if r["actual_peak_gb"] > 0
        )
        if success
        else None
    )
    by_model = {}
    for r in success:
        by_model.setdefault(r["model_id"], []).append(r)
    model_stats = {}
    for mid, ms in by_model.items():
        adm = [r for r in ms if r["pred_admitted"]]
        model_stats[mid] = {
            "n": len(ms),
            "safe_admission_rate": len(adm) / len(ms),
            "mape": statistics.fmean(
                abs(r["pred_center_gb"] - r["actual_peak_gb"]) / r["actual_peak_gb"]
                for r in ms
                if r["actual_peak_gb"] > 0
            ),
        }
    by_packing = {}
    for pk in (True, False):
        ps = [r for r in success if r["packing"] == pk]
        if ps:
            adm = [r for r in ps if r["pred_admitted"]]
            by_packing[pk] = {
                "n": len(ps),
                "safe_admission_rate": len(adm) / len(ps),
            }
    return {
        "queue_total": len(jobs),
        "success": len(success),
        "safe_admission_rate": safe_admission_rate,
        "oom_admission_rate": len(oom_admitted) / len(oom) if oom else 0.0,
        "memory_mape": mape,
        "by_model": model_stats,
        "by_packing": {str(k): v for k, v in by_packing.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "h800_business_blind_eval_v1.json")
    args = parser.parse_args()
    result = evaluate(queue_path=args.queue, results_dir=args.results_dir)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
