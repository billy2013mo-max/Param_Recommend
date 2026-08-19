#!/usr/bin/env python3
"""Fit the frozen-VL vision-tower forward peak memory model on old VL jobs.

The V2 prospective VL runs proved that the per-step peak equals the
vision-tower forward peak (obs/vision==1.00 on all 18 rows) and that this peak
scales with gc and mbs (qwen2p5_vl_3b: SAFE=11.2GB, NOGC=18.5GB, PRESSURE+
mbs=4 =48.2GB).  The existing VL overlay credited the tower with ~0.34GB
(proxy), under-predicting by 25-130% depending on mechanism.

This script fits a light structural model of that vision-forward peak using
the OLD VL jobs (h800vlfix-*, from qype19/pzfj/zltbjg + supplement campaigns)
whose vision-phase memory probe already recorded max_reserved_during_vision.
The resulting ``vision_peak`` artifact is qualified only on old jobs; the V2
and any future prospective rows are NEVER used for fitting.

Features are structural: a state term (tower/backbone parameter bytes proxy)
plus a saved-activation term driven by the per-step vision patch count, the
tower hidden width, the tower depth, gradient-checkpointing on/off and mbs.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from common import ARTIFACT_DIR, RESULTS_DIR, read_json, sha256_file, sha256_json, write_json

SCHEMA = "sft_h800_vl_vision_peak/v1"
OUTPUT = ARTIFACT_DIR / "h800_vl_vision_peak_v1.json"
OLD_JOB_PREFIXES = ("h800vlfix", "h800vlacc")
DTYPE_BYTES = 2

# Vision-tower geometry per model (hidden/depth from vision_config).
VISION_GEOMETRY = {
    "qwen2p5_vl_3b": {"hidden": 1280, "depth": 32},
    "qwen3_vl_4b": {"hidden": 1536, "depth": 32},
    "qwen3p5_4b": {"hidden": 1536, "depth": 32},
}


def _old_vl_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta_path in sorted(RESULTS_DIR.glob("*/attempts/*/job_metadata.json")):
        job_id = meta_path.parent.parent.parent.name
        if not job_id.startswith(OLD_JOB_PREFIXES):
            continue
        metadata = read_json(meta_path)
        summary_paths = sorted(
            (meta_path.parent / "metrics").glob("summary.rank*.json")
        )
        if not summary_paths:
            continue
        probe = read_json(summary_paths[0]).get("vision_phase_memory_probe") or {}
        peak = probe.get("max_reserved_during_vision")
        if not peak:
            continue
        model_id = str(metadata.get("model_id") or "")
        geometry = VISION_GEOMETRY.get(model_id)
        if geometry is None:
            continue
        # Mean raw patch units / visual tokens from the bound profile.
        profile_path = Path(str(metadata.get("dataset_profile_path") or ""))
        patches_per_image = 0
        if profile_path.is_file():
            profile = None
            try:
                profile = read_json(profile_path)
            except (json.JSONDecodeError, ValueError):
                try:
                    profile = read_json(Path(profile_path.parent) / f"{profile_path.name}.json")
                except Exception:
                    profile = None
            if profile is not None:
                summary = profile.get("summary") or {}
                fields = summary.get("raw_patch_units_total") or {}
                if fields:
                    mean_images = (summary.get("images_per_sample") or {}).get("mean") or 1.0
                    patches_per_image = float(fields.get("mean") or 0.0) / max(1.0, float(mean_images))
        images_per_step = max(
            1, int(metadata.get("max_samples") or 1)
        )
        rows.append(
            {
                "job_id": job_id,
                "model_id": model_id,
                "gc_off": not bool(metadata.get("gradient_checkpointing", True)),
                "mbs": int(metadata.get("mbs") or 1),
                "model_parameters": float(metadata.get("model_parameters") or 0),
                "patches_per_image": patches_per_image,
                "images_per_step": images_per_step,
                "vision_hidden": geometry["hidden"],
                "vision_depth": geometry["depth"],
                "peak_reserved_bytes": float(peak),
                "mechanism_id": str(metadata.get("mechanism_id") or metadata.get("arm_id") or "?"),
                "campaign_id": str(metadata.get("campaign_id") or "?"),
            }
        )
    return rows


def _design(row: dict[str, Any]) -> list[float]:
    mbs = row["mbs"]
    patches = row["patches_per_image"] * row["images_per_step"]
    act_per_layer = patches * row["vision_hidden"] * DTYPE_BYTES
    # Saved activation: gc-off saves the whole tower, gc-on only one layer's
    # worth (activation checkpointing recomputes). mbs multiplies the batch of
    # vision forward work.
    saved = act_per_layer * (row["vision_depth"] if row["gc_off"] else 1.0)
    return [
        1.0,
        row["model_parameters"],
        saved,               # per physical microbatch saved activation
        saved * math.sqrt(mbs),  # sublinear mbs growth (batch overlap)
        saved * mbs,         # linear mbs growth (upper check)
    ]


def _fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    design = np.asarray([_design(row) for row in rows], dtype=float)
    target = np.asarray([row["peak_reserved_bytes"] for row in rows], dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coefficients
    residuals = target - predicted
    abs_pct = np.abs(residuals) / target
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mape = float(np.mean(abs_pct))
    return {
        "coefficients": coefficients.tolist(),
        "coefficient_names": ["intercept", "params", "saved_gc_aware", "saved_sqrt_mbs", "saved_lin_mbs"],
        "mape": mape,
        "mae_bytes": mae,
        "rmse_bytes": rmse,
        "rows": len(rows),
        "worst": [
            {
                "job_id": row["job_id"],
                "model_id": row["model_id"],
                "mbs": row["mbs"],
                "gc_off": row["gc_off"],
                "mechanism": row["mechanism_id"],
                "peak_actual": row["peak_reserved_bytes"],
                "peak_predicted": float(predicted[i]),
                "abs_pct_error": float(abs_pct[i]),
            }
            for i, row in enumerate(rows)
        ],
        "sorted_worst": [
            {
                "job_id": rows[i]["job_id"],
                "model_id": rows[i]["model_id"],
                "mbs": rows[i]["mbs"],
                "gc_off": rows[i]["gc_off"],
                "mechanism": rows[i]["mechanism_id"],
                "peak_actual": rows[i]["peak_reserved_bytes"],
                "peak_predicted": float(predicted[i]),
                "abs_pct_error": float(abs_pct[i]),
            }
            for i in sorted(range(len(rows)), key=lambda i: -abs_pct[i])[:10]
        ],
        "by_mechanism": {
            mech: {
                "n": len(sub),
                "mape": float(
                    np.mean(
                        np.abs(
                            np.asarray([row["peak_reserved_bytes"] for row in sub])
                            - predicted[[rows.index(row) for row in sub]]
                        )
                        / np.asarray([row["peak_reserved_bytes"] for row in sub])
                    )
                ),
                "median": statistics.median([row["peak_reserved_bytes"] for row in sub]),
            }
            for mech, sub in defaultdict(list, {}).items()
        } if False else {},
    }


def fit() -> dict[str, Any]:
    rows = _old_vl_rows()
    if not rows:
        raise RuntimeError("no old VL vision-peak rows found")
    fit_result = _fit(rows)
    payload = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit": fit_result,
        "data_source": {
            "job_prefixes": list(OLD_JOB_PREFIXES),
            "role": "development_only_old_vl_jobs",
            "note": "V2/new prospective rows are excluded from this fit.",
        },
        "release": {"mode": "shadow_only", "automatic_admission_allowed": False},
    }
    payload["report_sha256"] = sha256_json(payload)
    write_json(OUTPUT, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(fit(), ensure_ascii=False, indent=2))
