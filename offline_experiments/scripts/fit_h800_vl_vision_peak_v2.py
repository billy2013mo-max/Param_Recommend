#!/usr/bin/env python3
"""Vision-tower per-step peak model v2: per-mechanism linear in patch count.

The V3 27-row grid proves peak = base(mechanism, model) + slope(mechanism, model) * patches.
Fit one linear regression per (model, mechanism) pair on the 3-frame points.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from common import ARTIFACT_DIR, RESULTS_DIR, ROOT, read_json, read_jsonl, sha256_file, sha256_json, write_json

SCHEMA = "sft_h800_vl_vision_peak/v2"
OUTPUT = ARTIFACT_DIR / "h800_vl_vision_peak_v2.json"
V3_QUEUE = ROOT / "matrix" / "h800_hybrid_vl_prospective_acceptance_v3.jsonl"
MODEL_INVENTORY = ARTIFACT_DIR / "h800_qwen35_vl_supplement_model_inventory_v1.json"


def _load_rows() -> list[dict[str, Any]]:
    inventory = read_json(MODEL_INVENTORY)
    params = {
        str(m["id"]): float(m.get("actual_parameters") or m.get("total_parameters") or 0)
        for m in inventory["models"]
    }
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(V3_QUEUE):
        if not str(row.get("design_arm") or "").startswith("vl_image"):
            continue
        job_id = str(row["job_id"])
        summary = list(
            (RESULTS_DIR / job_id / "attempts").glob("*/metrics/summary.rank0.json")
        )
        if not summary:
            continue
        probe = read_json(summary[0]).get("vision_phase_memory_probe") or {}
        peak = probe.get("max_reserved_during_vision")
        if not peak:
            continue
        profile = read_json(Path(str(row["dataset_profile_path"])))
        record = profile["records"][0]
        patches = float(record["raw_patch_units_total"])
        rows.append(
            {
                "job_id": job_id,
                "model_id": row["model_id"],
                "mechanism": str(row.get("mechanism_id") or "?"),
                "gc": bool(row["gc"]),
                "mbs": int(row["mbs"]),
                "params": params.get(row["model_id"], 0),
                "patches": patches,
                "peak_reserved_bytes": float(peak),
            }
        )
    return rows


def fit() -> dict[str, Any]:
    rows = _load_rows()
    if len(rows) != 27:
        raise ValueError(f"expected 27 V3 grid rows, got {len(rows)}")
    models = sorted({r["model_id"] for r in rows})
    mechanisms = sorted({r["mechanism"] for r in rows})
    per_pair: dict[str, Any] = {}
    for model_id in models:
        for mech in mechanisms:
            sub = [r for r in rows if r["model_id"] == model_id and r["mechanism"] == mech]
            if len(sub) < 2:
                continue
            patches = np.asarray([r["patches"] for r in sub])
            peaks = np.asarray([r["peak_reserved_bytes"] for r in sub])
            # linear fit: peak = a + b * patches
            A = np.vstack([np.ones_like(patches), patches]).T
            coef, *_ = np.linalg.lstsq(A, peaks, rcond=None)
            pred = coef[0] + coef[1] * patches
            ape = np.abs(pred - peaks) / peaks
            per_pair[f"{model_id}::{mech}"] = {
                "intercept_bytes": float(coef[0]),
                "slope_bytes_per_patch": float(coef[1]),
                "n": len(sub),
                "in_sample_mape": float(np.mean(ape)),
                "in_sample_max_abs_pct": float(np.max(ape)),
            }
    # Overall in-sample
    all_pred = []
    all_true = []
    for r in rows:
        key = f"{r['model_id']}::{r['mechanism']}"
        if key not in per_pair:
            continue
        p = per_pair[key]
        pred = p["intercept_bytes"] + p["slope_bytes_per_patch"] * r["patches"]
        all_pred.append(pred)
        all_true.append(r["peak_reserved_bytes"])
    all_pred = np.asarray(all_pred)
    all_true = np.asarray(all_true)
    ape_all = np.abs(all_pred - all_true) / all_true
    payload = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_source": {
            "queue": str(V3_QUEUE.resolve()),
            "sha256": sha256_file(V3_QUEUE),
            "role": "v3_grid_calibration_only_not_acceptance",
        },
        "per_model_mechanism": per_pair,
        "in_sample_mape": float(np.mean(ape_all)),
        "in_sample_max_abs_pct": float(np.max(ape_all)),
        "coverage_margin": 1.05,
        "release": {"mode": "shadow_only", "automatic_admission_allowed": False},
    }
    payload["report_sha256"] = sha256_json(payload)
    write_json(OUTPUT, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(fit(), ensure_ascii=False, indent=2))
