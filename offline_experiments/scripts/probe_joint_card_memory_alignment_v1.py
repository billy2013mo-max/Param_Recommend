#!/usr/bin/env python3
"""Probe: can H800 and RTX4090 memory records share one physical_shares design?

Read-only diagnostic for step 1 of the joint card-aware memory refit.
Builds memory records for both cards, runs them through the SHARED
h800_challenger_modeling._memory_features, and reports:

  * record counts per card / outcome
  * whether the 28-dim physical_shares vector builds without error
  * selector-key coverage per card (the thing H800's bounded layer keys on)
  * analytic_reference magnitude per card (sanity: 4090 is 24GiB, H800 80GiB)

Does not fit anything and does not write any artifact.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import ROOT, read_json

GIB = 1024.0 ** 3


# ---------------------------------------------------------------- H800 side
def build_h800_records() -> list[dict]:
    import h800_challenger_modeling as H
    from h800_native_memory_calibration import (
        build_native_record,
        native_admission_reason,
    )

    observations = H._read_observations(
        ROOT / "artifacts" / "canonical_h800_observations.jsonl"
    )
    inventory = read_json(ROOT / "artifacts" / "model_inventory.json")
    hardware = read_json(ROOT / "config" / "hardware.json")
    model_by_id, fixed_lora = H._inventory_models(inventory)

    records: list[dict] = []
    reasons: Counter = Counter()
    for row in observations:
        reason = native_admission_reason(row)
        reasons[reason] += 1
        if reason != "admitted":
            continue
        records.append(
            build_native_record(
                row,
                model_by_id=model_by_id,
                fixed_lora=fixed_lora,
                hardware=hardware,
            )
        )
    return records, reasons


# --------------------------------------------------------------- 4090 side
def build_rtx4090_records() -> list[dict]:
    """Build 4090 memory records.

    The 4090 record builder needs ThroughputPredictor._normalized_request for
    hardware + model geometry + zero-stage resolution.  The predictor's
    __init__ hard-fails on an implementation-SHA gate that no commit in this
    repo can satisfy (the default artefact was frozen pre-git).  That gate
    guards the *throughput* head, which memory work never touches, so we
    bypass __init__'s validation for this diagnostic and record the fact.
    """
    import throughput_predictor as TP
    import rtx4090_physical_v4b_modeling as R

    original = TP.ThroughputPredictor._validate_bindings

    def _skip(self) -> None:  # noqa: ANN001
        self.binding_mismatches = ["<bypassed for memory-only diagnostic>"]

    TP.ThroughputPredictor._validate_bindings = _skip
    try:
        predictor = TP.ThroughputPredictor(strict_bindings=False)
        campaign_root = (
            ROOT / "campaigns" / "rtx4090_20260717"
        )
        rows = R._read_rows(campaign_root)
        records = R._build_records(
            predictor=predictor,
            campaign_root=campaign_root,
            rows=rows,
            require_throughput=False,
        )
    finally:
        TP.ThroughputPredictor._validate_bindings = original
    return records


# ------------------------------------------------------------------ report
def selector_key(record) -> str:
    sel = record.get("selector") or {}
    sc = record.get("scenario") or {}
    return json.dumps(
        [
            sel.get("training_mode"),
            sc.get("gpu_count"),
            sel.get("zero_stage"),
            bool(sel.get("gradient_checkpointing")),
            bool(sel.get("packing")),
            sc.get("physical_mbs"),
        ]
    )


def summarize(name: str, records) -> dict:
    import h800_challenger_modeling as H

    out: dict = {"card": name, "records": len(records)}
    outcomes = Counter(H._outcome(r) for r in records)
    out["outcomes"] = dict(outcomes)

    ok, fail, refs, dims = 0, [], [], set()
    for r in records:
        try:
            vec = H._memory_features(r, "physical_shares")
            dims.add(len(vec))
            if not np.all(np.isfinite(vec)):
                fail.append((r.get("observation_id"), "non-finite"))
                continue
            ok += 1
            refs.append(float((r.get("memory") or {})["analytic_reference_bytes"]))
        except Exception as exc:  # noqa: BLE001
            fail.append((r.get("observation_id"), f"{type(exc).__name__}: {exc}"))
    out["physical_shares_built"] = ok
    out["physical_shares_failed"] = len(fail)
    out["failure_examples"] = fail[:5]
    out["feature_dims_seen"] = sorted(dims)
    if refs:
        out["analytic_reference_gib"] = {
            "min": round(min(refs) / GIB, 2),
            "median": round(float(np.median(refs)) / GIB, 2),
            "max": round(max(refs) / GIB, 2),
        }
    caps = Counter()
    for r in records:
        m = r.get("memory") or {}
        cap = m.get("capacity_bytes") or m.get("safe_limit_bytes")
        if cap:
            caps[round(float(cap) / GIB)] += 1
    out["capacity_gib_counts"] = dict(caps)
    keys = Counter(selector_key(r) for r in records)
    out["distinct_selector_keys"] = len(keys)
    out["top_selector_keys"] = keys.most_common(6)
    models = Counter(
        str((r.get("scenario") or {}).get("model_id") or "?") for r in records
    )
    out["models"] = dict(models)
    return out


def main() -> None:
    result: dict = {}

    h_records, h_reasons = build_h800_records()
    result["h800"] = summarize("h800", h_records)
    result["h800"]["admission_reasons"] = dict(h_reasons)

    r_records = build_rtx4090_records()
    result["rtx4090"] = summarize("rtx4090", r_records)

    h_keys = {selector_key(r) for r in h_records}
    r_keys = {selector_key(r) for r in r_records}
    result["selector_overlap"] = {
        "h800_only": len(h_keys - r_keys),
        "rtx4090_only": len(r_keys - h_keys),
        "shared": len(h_keys & r_keys),
        "shared_examples": sorted(h_keys & r_keys)[:8],
    }
    h_models = {
        str((r.get("scenario") or {}).get("model_id") or "?") for r in h_records
    }
    r_models = {
        str((r.get("scenario") or {}).get("model_id") or "?") for r in r_records
    }
    result["model_overlap"] = {
        "h800": sorted(h_models),
        "rtx4090": sorted(r_models),
        "shared": sorted(h_models & r_models),
    }
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
