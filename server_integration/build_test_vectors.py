#!/usr/bin/env python
"""Generate ``testdata/test_vectors.json`` for the Go param-recommender.

Read-only driver: imports the frozen H800 predictor from
``offline_experiments/scripts/`` and runs it against 5 fixed Qwen3 dense-text
scenarios. Captures per-candidate intermediate values by monkey-patching two
internal helpers at runtime (memory correction / structured throughput). The
predictor .py files are NOT modified; the patches wrap the original callables
with a tap that records their inputs and outputs into a side channel.

Emits:
- ``server_integration/testdata/test_vectors.json`` — one entry per scenario,
  each with the request, the full candidate list, and per-candidate memory /
  throughput intermediates + final row from the predictor report.

Run:
    cd Param_Recommend
    python server_integration/build_test_vectors.py

The output is stable given the same frozen artifacts; regenerate only when
those artifacts change.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "offline_experiments" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402

import benchmark_h800_memory_center_models_v1 as memory_center  # noqa: E402
import candidate_generator  # noqa: E402
import structured_throughput_modeling as stm  # noqa: E402
from h800_unified_v3_throughput_v5_predictor import (  # noqa: E402
    H800UnifiedV3ThroughputV5Predictor,
)


CAPACITY_BYTES = 150_142_189_568  # H800 140 GiB HBM3
PREDICTOR_SOURCE = SCRIPTS_DIR / "h800_unified_v3_throughput_v5_predictor.py"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Monkey-patch tap for the memory correction pipeline.
#
# ``_predict_correction`` builds raw feature matrix -> expanded basis ->
# standardized -> intercept + coefs @ standardized -> shrinkage. We wrap it to
# record every intermediate for each (records, model) call. The model dict has
# a ``correction_shrinkage`` field, so we can tell apart the "center" model and
# the "risk" model even though the same function services both.
# ---------------------------------------------------------------------------

memory_calls: list[dict] = []
_orig_predict_correction = memory_center._predict_correction


def _tapped_predict_correction(records, model):
    feature_names = [str(name) for name in model["raw_feature_names"]]
    raw = memory_center._raw_matrix(records, feature_names)
    expanded = memory_center._expand_basis(
        raw,
        kind=str(model["basis_kind"]),
        raw_means=np.asarray(model["raw_means"], dtype=float),
        raw_scales=np.asarray(model["raw_scales"], dtype=float),
        nonlinear_indexes=[int(v) for v in model["nonlinear_indexes"]],
        feature_names=feature_names,
    )
    standardized = (
        expanded - np.asarray(model["expanded_means"], dtype=float)
    ) / np.asarray(model["expanded_scales"], dtype=float)
    coeffs = np.asarray(model["coefficients"], dtype=float)
    intercept = float(model["intercept"])
    raw_correction = intercept + standardized @ coeffs
    shrinkage = float(model["correction_shrinkage"])
    correction = shrinkage * raw_correction

    memory_calls.append(
        {
            "model_shrinkage": shrinkage,
            "basis_kind": str(model["basis_kind"]),
            "feature_names": feature_names,
            "raw": raw.tolist(),
            "expanded": expanded.tolist(),
            "standardized": standardized.tolist(),
            "intercept": intercept,
            "raw_correction": raw_correction.tolist(),
            "correction": correction.tolist(),
        }
    )
    # Reuse the original to keep the numerical result byte-identical.
    return _orig_predict_correction(records, model)


memory_center._predict_correction = _tapped_predict_correction
# Re-bind the alias used by the wrapper module so both paths hit the tap.
import h800_unified_bounded_memory_model as bm  # noqa: E402

bm._predict_correction = _tapped_predict_correction


# ---------------------------------------------------------------------------
# Monkey-patch tap for the throughput structured predictor.
#
# ``_predict_log_throughput`` in structured_throughput_modeling.py returns a
# single float. We record all intermediates before returning.
# ---------------------------------------------------------------------------

throughput_calls: list[dict] = []
_orig_predict_log_throughput = stm._predict_log_throughput


def _tapped_predict_log_throughput(candidate, model):
    import math

    features = np.asarray(candidate["structured_features"], dtype=float)
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    standardized = 2.0 * np.tanh((features - means) / (2.0 * scales))
    global_raw = np.asarray(
        model["global_component_raw_parameters"], dtype=float
    )
    card_offsets = model["card_component_raw_offsets"]
    offset = np.asarray(
        card_offsets.get(
            str(candidate["card_id"]),
            np.zeros(len(stm.COMPONENT_NAMES)),
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
    roof = (
        compute ** stm.SMOOTH_ROOFLINE_POWER + hbm ** stm.SMOOTH_ROOFLINE_POWER
    ) ** (1.0 / stm.SMOOTH_ROOFLINE_POWER)
    step_base = launch + roof + optimizer + communication
    correction_coeffs = np.asarray(
        model["correction_coefficients"], dtype=float
    )
    raw_correction = float(model["correction_intercept"]) + float(
        standardized @ correction_coeffs
    )
    card_residuals = model["card_residual_coefficients"]
    card_residual = np.asarray(
        card_residuals.get(
            str(candidate["card_id"]),
            np.zeros(len(stm.CARD_RESIDUAL_FEATURE_NAMES)),
        ),
        dtype=float,
    )
    raw_correction += float(
        standardized[list(stm.CARD_RESIDUAL_FEATURE_INDEXES)] @ card_residual
    )
    correction = stm.CORRECTION_LOG_LIMIT * math.tanh(
        raw_correction / stm.CORRECTION_LOG_LIMIT
    )
    log_step = math.log(stm._positive(step_base)) + correction
    log_throughput = float(candidate["static_log_work"]) - log_step

    key = str(candidate.get("request_id", ""))
    throughput_calls.append(
        {
            "call_key": key,
            "card_id": str(candidate["card_id"]),
            "features_names": list(stm.FEATURE_NAMES),
            "features": features.tolist(),
            "feature_means": means.tolist(),
            "feature_scales": scales.tolist(),
            "standardized": standardized.tolist(),
            "component_names": list(stm.COMPONENT_NAMES),
            "physical_components": components.tolist(),
            "multipliers": multipliers.tolist(),
            "component_launch": launch,
            "component_compute": compute,
            "component_hbm": hbm,
            "component_optimizer": optimizer,
            "component_communication": communication,
            "smooth_roofline_power": stm.SMOOTH_ROOFLINE_POWER,
            "roof": float(roof),
            "step_base": float(step_base),
            "correction_intercept": float(model["correction_intercept"]),
            "correction_coefficients": correction_coeffs.tolist(),
            "card_residual_feature_names": list(stm.CARD_RESIDUAL_FEATURE_NAMES),
            "card_residual": card_residual.tolist(),
            "raw_correction": raw_correction,
            "correction_log_limit": stm.CORRECTION_LOG_LIMIT,
            "correction": correction,
            "log_step": log_step,
            "static_log_work": float(candidate["static_log_work"]),
            "log_throughput": log_throughput,
        }
    )
    return _orig_predict_log_throughput(candidate, model)


stm._predict_log_throughput = _tapped_predict_log_throughput
# throughput_predictor imported _predict_log_throughput by name, so its own
# module-level binding still points at the un-tapped original. Rebind it too.
import throughput_predictor as tp  # noqa: E402

tp._predict_log_throughput = _tapped_predict_log_throughput


# ---------------------------------------------------------------------------
# Scenarios: 5 Qwen3 dense-text SFT configurations.
# Parameter counts pulled from
# offline_experiments/artifacts/model_inventory.json.
# ---------------------------------------------------------------------------

MODEL_PARAMETERS = {
    "qwen3_1p7b": 2_031_739_904,
    "qwen3_4b": 4_022_468_096,
    "qwen3_8b": 8_190_735_360,
    "qwen3_14b": 14_768_307_200,
}


SCENARIOS = [
    {
        "id": "S1-qwen3_8b-lora-short",
        "request": {
            "model_id": "qwen3_8b",
            "training_mode": "lora",
            "dataset_id": "short_512",
            "target_gbs": 64,
            "cutoff_len": 512,
            "actual_parameters": MODEL_PARAMETERS["qwen3_8b"],
            "packing": False,
        },
    },
    {
        "id": "S2-qwen3_8b-full-multiturn",
        "request": {
            "model_id": "qwen3_8b",
            "training_mode": "full",
            "dataset_id": "multiturn_2048",
            "target_gbs": 64,
            "cutoff_len": 2048,
            "actual_parameters": MODEL_PARAMETERS["qwen3_8b"],
            "packing": False,
        },
    },
    {
        "id": "S3-qwen3_8b-lora-longtail-packing",
        "request": {
            "model_id": "qwen3_8b",
            "training_mode": "lora",
            "dataset_id": "longtail_8192",
            "target_gbs": 64,
            "cutoff_len": 8192,
            "actual_parameters": MODEL_PARAMETERS["qwen3_8b"],
            "packing": True,
        },
    },
    {
        "id": "S4-qwen3_14b-lora-multiturn",
        "request": {
            "model_id": "qwen3_14b",
            "training_mode": "lora",
            "dataset_id": "multiturn_2048",
            "target_gbs": 128,
            "cutoff_len": 2048,
            "actual_parameters": MODEL_PARAMETERS["qwen3_14b"],
            "packing": False,
        },
    },
    {
        "id": "S5-qwen3_1p7b-lora-short",
        "request": {
            "model_id": "qwen3_1p7b",
            "training_mode": "lora",
            "dataset_id": "short_512",
            "target_gbs": 32,
            "cutoff_len": 512,
            "actual_parameters": MODEL_PARAMETERS["qwen3_1p7b"],
            "packing": False,
        },
    },
]


def _index_memory_calls_for_request(
    request_id: str, memory_calls: list[dict]
) -> dict:
    """Filter tap entries down to the calls that touched this request.

    The memory tap sees every _predict_correction call. In V3 each `predict`
    invocation triggers one center call and one risk call over the batch of
    admitted records; but the predictor pipeline calls _memory_result per
    record with a length-1 record list. Recover the (center, risk) pair for
    this request by looking up the calls whose record set has this request_id.
    """
    # No direct request_id in memory tap payload -- the driver instead
    # produces one call per record via _memory_result loop, so we key on the
    # order the predictor iterates candidates. See ``call_index`` below.
    raise NotImplementedError


def main() -> None:
    print("=== initializing H800 V5 predictor (uses offline_experiments artifacts) ===", flush=True)
    predictor = H800UnifiedV3ThroughputV5Predictor()
    print(
        f"predictor init ok, capacity_bytes={predictor.memory_capacity_bytes}",
        flush=True,
    )

    scenarios_out = []
    for scenario in SCENARIOS:
        scen_id = scenario["id"]
        req = scenario["request"]
        print(f"\n=== scenario {scen_id} ===", flush=True)

        gen = candidate_generator.generate_candidates(
            req, capacity_bytes=CAPACITY_BYTES
        )
        candidates = gen["candidates"]
        print(
            f"  generated {len(candidates)} candidates, "
            f"pruned {len(gen.get('rejected', []))}",
            flush=True,
        )

        # Reset per-scenario taps so we can associate correction calls with
        # the right rows using call ordering.
        memory_calls.clear()
        throughput_calls.clear()

        try:
            report = predictor.predict(candidates)
        except Exception:
            print("  predict FAILED:", flush=True)
            traceback.print_exc()
            raise

        # V3 memory: two calls per _memory_result loop iteration -- one for
        # the center model, one for the risk model. Both length-1. Order is
        # (center, risk) per input_index.
        mem_by_index = {}
        for i in range(0, len(memory_calls), 2):
            center = memory_calls[i]
            risk = (
                memory_calls[i + 1] if i + 1 < len(memory_calls) else None
            )
            mem_by_index[i // 2] = {"center": center, "risk": risk}

        # Throughput: _predict_log_throughput is invoked once per admitted
        # request, in the same order as they appear in admitted_requests,
        # which itself preserves input_index order. Match by the request_id
        # published in the final report.
        predictions = report["predictions"] if "predictions" in report else report["rows"]
        admitted_ids_in_order = [
            str(row["request_id"])
            for row in sorted(predictions, key=lambda r: r["input_index"])
            if row["throughput"].get("prediction_available")
        ]
        thr_by_request_id = {}
        for admitted_id, call in zip(admitted_ids_in_order, throughput_calls):
            thr_by_request_id[admitted_id] = call

        results = []
        for row in predictions:
            input_index = int(row["input_index"])
            request_id = str(row["request_id"])
            memory_intermediates = mem_by_index.get(input_index)
            throughput_intermediates = thr_by_request_id.get(request_id)
            results.append(
                {
                    "input_index": input_index,
                    "request_id": request_id,
                    "candidate": candidates[input_index],
                    "memory": {
                        "final_row": row.get("memory"),
                        "intermediates": memory_intermediates,
                    },
                    "throughput": {
                        "final_row": row.get("throughput"),
                        "intermediates": throughput_intermediates,
                    },
                    "final_report_row": row,
                }
            )

        scenarios_out.append(
            {
                "scenario_id": scen_id,
                "request": req,
                "capacity_bytes": CAPACITY_BYTES,
                "candidates_count": len(candidates),
                "generator_output": {
                    "comparison_group": gen["comparison_group"],
                    "scenario": gen["scenario"],
                    "generation_policy": gen["generation_policy"],
                    "rejected": gen.get("rejected", []),
                },
                "results": results,
            }
        )

        n_admitted = sum(
            1 for r in results if r["final_report_row"]["memory"].get("admitted")
        )
        n_throughput = sum(
            1
            for r in results
            if r["throughput"]["final_row"].get("prediction_available")
        )
        print(
            f"  admitted {n_admitted}/{len(results)}, "
            f"throughput_predicted {n_throughput}",
            flush=True,
        )

    out_path = REPO_ROOT / "server_integration" / "testdata" / "test_vectors.json"
    payload = {
        "schema": "sft_server_integration_test_vectors/v1",
        "generated_at_utc": (
            os.environ.get("SOURCE_DATE_EPOCH_ISO")
            or "2026-08-26T00:00:00+00:00"
        ),
        "python_version": sys.version,
        "predictor_source_sha256": _sha256_file(PREDICTOR_SOURCE),
        "capacity_bytes": CAPACITY_BYTES,
        "scenarios": scenarios_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path} ({out_path.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
