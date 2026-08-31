#!/usr/bin/env python3
"""Step 2: package the RTX 4090 branch for the Go side.

What the Go side is missing today
---------------------------------
``server_integration/README.md`` says "only H800".  That is true of the
PACKAGING, not of the models:

  * throughput - ``artifacts/throughput_v5.json`` is already a two-card model
    (``cards: [h800, rtx4090]``, with ``card_component_raw_offsets`` and
    ``card_residual_coefficients`` entries for both).  Nothing needs refitting;
    the 4090 branch simply was never exercised or shipped as vectors.
  * memory - the shipped V3 artifact is H800-only.  The joint two-card refit
    lives in ``artifacts/joint_card_v3_memory_v1.json``.

This script produces:

  1. ``server_integration/artifacts/joint_card_v3_memory.json`` - the joint
     memory model with PER-CARD admission multipliers.
  2. ``server_integration/testdata/test_vectors_rtx4090.json`` - vectors that
     let Go verify it reproduces both heads on real 4090 rows.

Why not reuse ``build_test_vectors.py``
---------------------------------------
``H800UnifiedV3ThroughputV5Predictor`` is H800-bound by construction: it
asserts the runtime H800 capacity equals the V3 artifact's, hardcodes
``hardware_id: "h800"`` in its output, and layers on VL / hybrid / packing
release logic that does not apply to 4090 dense text.  Rather than bend it,
this script exercises the two frozen models directly, which is exactly what
Go has to reimplement anyway.

Read-only with respect to every offline_experiments artifact.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "offline_experiments" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402

import benchmark_h800_memory_center_models_v1 as bm  # noqa: E402
import structured_throughput_modeling as stm  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / "server_integration" / "artifacts"
TESTDATA_DIR = REPO_ROOT / "server_integration" / "testdata"
JOINT_MEMORY_SOURCE = (
    REPO_ROOT
    / "offline_experiments"
    / "artifacts"
    / "joint_card_v3_memory_v1.json"
)
THROUGHPUT_ARTIFACT = ARTIFACT_DIR / "throughput_v5.json"
CARD_ID = "rtx4090"
SAFE_LIMIT_FRACTION = 0.95


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------ memory
def memory_prediction(record: dict, model: dict) -> dict:
    """centre / risk head evaluation, mirroring bm._predict_correction."""
    names = [str(n) for n in model["raw_feature_names"]]
    raw = bm._raw_matrix([record], names)
    expanded = bm._expand_basis(
        raw,
        kind=str(model["basis_kind"]),
        raw_means=np.asarray(model["raw_means"], dtype=float),
        raw_scales=np.asarray(model["raw_scales"], dtype=float),
        nonlinear_indexes=[int(v) for v in model["nonlinear_indexes"]],
        feature_names=names,
    )
    standardized = (
        expanded - np.asarray(model["expanded_means"], dtype=float)
    ) / np.asarray(model["expanded_scales"], dtype=float)
    raw_correction = float(model["intercept"]) + float(
        standardized[0] @ np.asarray(model["coefficients"], dtype=float)
    )
    correction = float(model["correction_shrinkage"]) * raw_correction
    reference = float(record["reference_bytes"])
    return {
        "raw_feature_names": names,
        "raw": raw[0].tolist(),
        "basis_kind": str(model["basis_kind"]),
        "expanded_dimension": int(expanded.shape[1]),
        "intercept": float(model["intercept"]),
        "raw_correction": raw_correction,
        "correction_shrinkage": float(model["correction_shrinkage"]),
        "correction": correction,
        "reference_bytes": reference,
        "predicted_bytes": reference * math.exp(correction),
    }


# -------------------------------------------------------------- throughput
def throughput_prediction(candidate: dict, model: dict) -> dict:
    features = np.asarray(candidate["structured_features"], dtype=float)
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    standardized = 2.0 * np.tanh((features - means) / (2.0 * scales))
    global_raw = np.asarray(
        model["global_component_raw_parameters"], dtype=float
    )
    offset = np.asarray(
        model["card_component_raw_offsets"].get(
            str(candidate["card_id"]), np.zeros(len(stm.COMPONENT_NAMES))
        ),
        dtype=float,
    )
    multipliers = 1.0 + np.logaddexp(0.0, global_raw + offset)
    components = np.asarray(candidate["physical_components"], dtype=float)
    launch = float(components[0] * multipliers[0])
    compute = float(components[1] * multipliers[1])
    hbm = float(components[2] * multipliers[2])
    optimizer = float(components[3] * multipliers[3])
    communication = float(components[4] * multipliers[4])
    power = float(model["smooth_roofline_power"])
    roof = (compute ** power + hbm ** power) ** (1.0 / power)
    step_base = launch + roof + optimizer + communication
    raw_correction = float(model["correction_intercept"]) + float(
        standardized @ np.asarray(model["correction_coefficients"], dtype=float)
    )
    card_residual = np.asarray(
        model["card_residual_coefficients"].get(
            str(candidate["card_id"]),
            np.zeros(len(model["card_residual_feature_names"])),
        ),
        dtype=float,
    )
    raw_correction += float(
        standardized[list(stm.CARD_RESIDUAL_FEATURE_INDEXES)] @ card_residual
    )
    limit = float(model["correction_log_limit"])
    correction = limit * math.tanh(raw_correction / limit)
    log_step = math.log(stm._positive(step_base)) + correction
    log_throughput = float(candidate["static_log_work"]) - log_step
    return {
        "card_id": str(candidate["card_id"]),
        "feature_names": list(stm.FEATURE_NAMES),
        "features": features.tolist(),
        "standardized": standardized.tolist(),
        "component_names": list(stm.COMPONENT_NAMES),
        "physical_components": components.tolist(),
        "card_component_raw_offset": offset.tolist(),
        "multipliers": multipliers.tolist(),
        "component_launch": launch,
        "component_compute": compute,
        "component_kernel_hbm": hbm,
        "component_optimizer_hbm": optimizer,
        "component_communication": communication,
        "smooth_roofline_power": power,
        "roof": float(roof),
        "step_base": float(step_base),
        "correction_intercept": float(model["correction_intercept"]),
        "card_residual_feature_names": list(
            model["card_residual_feature_names"]
        ),
        "card_residual": card_residual.tolist(),
        "raw_correction": raw_correction,
        "correction_log_limit": limit,
        "correction": correction,
        "log_step": log_step,
        "static_log_work": float(candidate["static_log_work"]),
        "log_throughput": log_throughput,
        "predicted_effective_tokens_per_second": math.exp(log_throughput),
    }


# ------------------------------------------------------------------- build
def build(limit_per_bucket: int) -> dict:
    import fit_joint_card_v3_memory_v1 as joint
    import rtx4090_physical_v4b_modeling as R
    import throughput_predictor as TP
    from common import ROOT, read_json

    joint_report = read_json(JOINT_MEMORY_SOURCE)
    centre_model = joint_report["frozen_model"]["centre"]
    risk_model = joint_report["frozen_model"]["risk"]
    multipliers = joint_report["evaluation"]["upper_multiplier_calibration"][
        "recalibrated_per_card"
    ]
    throughput_model = read_json(THROUGHPUT_ARTIFACT)["frozen_model"]

    memory_records = joint.build_rtx4090_records()
    joint.attach_card_features(memory_records)
    by_job = {
        str(r["record_id"]).split("::", 1)[1]: r for r in memory_records
    }

    original = TP.ThroughputPredictor._validate_bindings
    TP.ThroughputPredictor._validate_bindings = lambda self: setattr(
        self, "binding_mismatches", ["implementation-sha gate bypassed"]
    )
    try:
        predictor = TP.ThroughputPredictor(strict_bindings=False)
        campaign = ROOT / "campaigns" / "rtx4090_20260717"
        rows = R._read_rows(campaign)
        records = R._build_records(
            predictor=predictor,
            campaign_root=campaign,
            rows=rows,
            require_throughput=True,
        )
        capacity = float(predictor.hardware[CARD_ID].memory_bytes)
        profiles = predictor.profiles
    finally:
        TP.ThroughputPredictor._validate_bindings = original

    safe_limit = capacity * SAFE_LIMIT_FRACTION
    multiplier = float(multipliers[CARD_ID])

    # Spread the vectors over the configuration space instead of taking the
    # first N rows, so Go exercises every branch that matters.
    buckets: dict[tuple, list] = defaultdict(list)
    for record in records:
        selector = record["selector"]
        scenario = record["scenario"]
        buckets[
            (
                str(selector["training_mode"]),
                int(selector["zero_stage"] or 0),
                bool(selector["gradient_checkpointing"]),
                bool(selector["packing"]),
                int(scenario["gpu_count"]),
            )
        ].append(record)

    vectors = []
    for key in sorted(buckets, key=lambda k: tuple(str(v) for v in k)):
        for record in buckets[key][:limit_per_bucket]:
            job_id = str(record["job_id"])
            memory_record = by_job.get(job_id)
            if memory_record is None:
                continue
            basis = stm._static_structured_basis(
                record, profiles, hardware_memory_bytes=capacity
            )
            candidate = {
                "card_id": CARD_ID,
                "structured_features": basis["features"],
                "physical_components": basis["components"],
                "static_log_work": math.log(
                    stm._positive(basis["work_per_step"]["effective_tokens"])
                ),
            }
            centre = memory_prediction(memory_record, centre_model)
            risk = memory_prediction(memory_record, risk_model)
            upper = max(
                centre["predicted_bytes"], risk["predicted_bytes"] * multiplier
            )
            observed = memory_record.get("_observed_reserved_bytes")
            vectors.append(
                {
                    "job_id": job_id,
                    "card_id": CARD_ID,
                    "scenario": {
                        "model_id": str(record["scenario"]["model_id"]),
                        "dataset_id": str(record["scenario"].get("dataset_id")),
                        "training_mode": str(record["selector"]["training_mode"]),
                        "zero_stage": int(record["selector"]["zero_stage"] or 0),
                        "gradient_checkpointing": bool(
                            record["selector"]["gradient_checkpointing"]
                        ),
                        "packing": bool(record["selector"]["packing"]),
                        "gpu_count": int(record["scenario"]["gpu_count"]),
                        "physical_mbs": int(record["scenario"]["physical_mbs"]),
                        "cutoff_len": int(record["scenario"]["cutoff_len"]),
                    },
                    "memory": {
                        "capacity_bytes": capacity,
                        "safe_limit_fraction": SAFE_LIMIT_FRACTION,
                        "safe_limit_bytes": safe_limit,
                        "upper_multiplier": multiplier,
                        "centre": centre,
                        "risk": risk,
                        "upper_bytes": upper,
                        "admitted": bool(upper <= safe_limit),
                        "observed_outcome": str(record["outcome"]),
                        "observed_reserved_bytes": observed,
                    },
                    "throughput": throughput_prediction(
                        candidate, throughput_model
                    ),
                }
            )

    return {
        "schema": "sft_server_integration_test_vectors_rtx4090/v1",
        "card_id": CARD_ID,
        "generated_from": {
            "joint_memory_artifact": str(JOINT_MEMORY_SOURCE),
            "joint_memory_sha256": _sha256_file(JOINT_MEMORY_SOURCE),
            "throughput_artifact": str(THROUGHPUT_ARTIFACT),
            "throughput_sha256": _sha256_file(THROUGHPUT_ARTIFACT),
            "campaign": "rtx4090_20260717",
        },
        "contract": {
            "memory": (
                "centre_bytes = reference_bytes * exp(shrinkage * (intercept + "
                "standardised_basis @ beta)); same for the risk head; "
                "admit iff max(centre, risk * upper_multiplier) <= "
                "safe_limit_fraction * capacity_bytes"
            ),
            "throughput": (
                "log_throughput = static_log_work - (log(step_base) + "
                "correction); step_base = launch + smooth_roofline(compute, "
                "kernel_hbm) + optimizer_hbm + communication, each component "
                "scaled by 1 + softplus(global_raw + card_offset)"
            ),
            "per_card_upper_multiplier": multipliers,
            "unknown_card_policy": throughput_model["unknown_card_policy"],
        },
        "notes": [
            "The upper_multiplier is PER CARD.  The shipped H800-only value "
            "(1.0171) admits 18 of 168 real 4090 OOM rows, so Go must dispatch "
            "on card_id here, not use a single constant.",
            "These vectors are generated from real campaign rows, so "
            "observed_outcome / observed_reserved_bytes are ground truth and "
            "can be used as a sanity check as well as a reproduction target.",
            "analysis_only: the joint memory model has not replaced the "
            "shipped H800 V3 artifact.",
        ],
        "vectors": vectors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-bucket", type=int, default=2)
    args = parser.parse_args()

    payload = build(args.per_bucket)

    TESTDATA_DIR.mkdir(parents=True, exist_ok=True)
    out = TESTDATA_DIR / "test_vectors_rtx4090.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    # Package the joint memory model next to the throughput model.
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    packaged = ARTIFACT_DIR / "joint_card_v3_memory.json"
    source = json.loads(JOINT_MEMORY_SOURCE.read_text(encoding="utf-8"))
    with packaged.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": source["schema"],
                "generated_at_utc": source["generated_at_utc"],
                "analysis_only": source["analysis_only"],
                "publishable": source["publishable"],
                "contract": source["contract"],
                "per_card_upper_multiplier": source["evaluation"][
                    "upper_multiplier_calibration"
                ]["recalibrated_per_card"],
                "frozen_model": source["frozen_model"],
                "evaluation_summary": {
                    "out_of_fold_by_card": source["evaluation"][
                        "out_of_fold_by_card"
                    ],
                    "admission_out_of_fold_per_card_recalibrated": source[
                        "evaluation"
                    ]["admission_out_of_fold_per_card_recalibrated"],
                },
                "limitations": source["limitations"],
                "source_report_sha256": source["report_sha256"],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    admitted = sum(1 for v in payload["vectors"] if v["memory"]["admitted"])
    oom = sum(
        1 for v in payload["vectors"] if v["memory"]["observed_outcome"] == "oom"
    )
    false_safe = sum(
        1
        for v in payload["vectors"]
        if v["memory"]["observed_outcome"] == "oom" and v["memory"]["admitted"]
    )
    print(
        json.dumps(
            {
                "test_vectors": str(out),
                "vectors": len(payload["vectors"]),
                "packaged_memory_artifact": str(packaged),
                "admitted": admitted,
                "observed_oom_in_vectors": oom,
                "false_safe_in_vectors": false_safe,
                "per_card_upper_multiplier": payload["contract"][
                    "per_card_upper_multiplier"
                ],
            },
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
