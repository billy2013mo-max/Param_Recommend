#!/usr/bin/env python3
"""Probe: can RTX 4090 rows be expressed in the PLATFORM's V3 memory contract?

The platform (server_integration) reads
``artifacts/h800_unified_bounded_memory_candidate_v3.json``, schema
``sft_h800_unified_bounded_memory_shadow_candidate/v3`` - model family
``physics_anchored_shared_bounded_residual``, 38 raw features expanded to 144
basis features, plus a separate risk head for admission.  That is NOT the
``physical_shares`` 28-feature contract.

This probe answers one question before any refitting: can a 4090 campaign row
be turned into a valid V3 job dict and pushed through the V3 raw-feature
builder without error?

Read-only.  Fits nothing, writes nothing.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, read_json

CAMPAIGN = ROOT / "campaigns" / "rtx4090_20260717"
PROFILE_DIR = CAMPAIGN / "artifacts" / "dataset_profiles"
V3_ARTIFACT = ROOT / "artifacts" / "h800_unified_bounded_memory_candidate_v3.json"
V3_INVENTORY = ROOT / "artifacts" / "h800_bounded_memory_v2_model_inventory_v1.json"


def _load_4090_jobs():
    import throughput_predictor as TP
    import rtx4090_physical_v4b_modeling as R

    original = TP.ThroughputPredictor._validate_bindings

    def _skip(self) -> None:  # noqa: ANN001
        self.binding_mismatches = [
            "implementation-sha gate bypassed: memory-only probe"
        ]

    TP.ThroughputPredictor._validate_bindings = _skip
    try:
        predictor = TP.ThroughputPredictor(strict_bindings=False)
        rows = R._read_rows(CAMPAIGN)
        out = []
        for index, row in enumerate(rows):
            classification = str(row.get("classification") or "")
            if classification not in {"success", "oom"}:
                continue
            job = R._rendered_job(CAMPAIGN, row)
            normalized = predictor._normalized_request(
                R._request(job), input_index=index
            )
            out.append((row, job, normalized))
    finally:
        TP.ThroughputPredictor._validate_bindings = original
    return out


def v3_job_from_4090(job, normalized, *, padding_stats) -> dict:
    """Translate a 4090 campaign job into the V3 job contract.

    Mapping decisions, each one explicit:

      zero_stage                 <- normalized["zero_stage"]  (4090 job carries
                                    "zero" as a string like "zero2"/"none")
      aligned_effective_sequence <- the canonical recipe copied verbatim from
                                    h800_unified_v3_v4b_predictor: cutoff when
                                    packing, else 8-aligned min(cutoff,
                                    maximum_clipped_tokens)
      dataset_profile_path       <- the 4090 campaign's own profile for the
                                    dataset id
      mechanism_id               <- absent.  It only drives
                                    critical_lora_x_log_effective_fraction,
                                    which fires on ONE named H800 mechanism;
                                    for 4090 rows the correct value is 0.
      packing_contract           <- absent -> expected_samples_per_pack = 1.0
    """
    cutoff = int(job["cutoff_len"])
    packing = bool(job.get("packing"))
    if packing:
        aligned = cutoff
    else:
        raw_max = int(padding_stats["maximum_clipped_tokens"])
        aligned = 8 * ((min(cutoff, raw_max) + 7) // 8)
    return {
        "model_id": str(job["model_id"]),
        "model_parameters": int(normalized["model_geometry"]["base_parameters"]),
        "train_type": str(job["train_type"]),
        "gpu_count": int(job["gpu_count"]),
        "mbs": int(job["mbs"]),
        "cutoff_len": cutoff,
        "aligned_effective_sequence": int(aligned),
        "zero": str(job.get("zero") or "none"),
        "zero_stage": int(normalized["zero_stage"]),
        "gc": bool(job["gc"]),
        "packing": packing,
        "dataset_id": str(job["dataset_id"]),
        "dataset_profile_path": str(
            PROFILE_DIR / f"{job['dataset_id']}.qwen3_nothink.jsonl"
        ),
    }


def capacity_normalised_features(
    values: dict, *, reference_bytes: float, capacity_bytes: float, model_id: str
) -> dict:
    """The 9 features that ``_current_features`` does NOT produce.

    They live in ``benchmark_h800_memory_center_models_v1._feature_value``,
    which divides by a MODULE-LEVEL constant:

        DEVICE_CAPACITY_BYTES = read_json(DEFAULT_HARDWARE)
                                    ["memory_bytes_reported_by_torch"]

    i.e. the H800's 150142189568 bytes, hardcoded.  A 4090 row pushed through
    that code gets reference_fraction = ref / 150GB instead of ref / 23.6GB -
    wrong by 5.91x, which makes every one of these features meaningless off
    H800.  This reimplements them with the ROW's own capacity.

    For an H800 row capacity_bytes == DEVICE_CAPACITY_BYTES, so the values are
    bit-identical to the shipped model.  This is a strict generalisation, not a
    behaviour change.
    """
    reference_fraction = float(reference_bytes) / float(capacity_bytes)
    activation_share = float(values["activation_share"])
    is_lora = float(values["is_lora"])
    zero2 = float(values["zero2"])
    zero3 = float(values["zero3"])
    gc = float(values["gradient_checkpointing"])
    single_gpu = float(abs(float(values["log2_gpu_count"])) < 1.0e-12)
    zero0 = max(0.0, 1.0 - zero2 - zero3)
    tied = float(model_id in {"qwen3_1p7b", "qwen3_4b"})
    return {
        "reference_fraction_of_capacity": reference_fraction,
        "activation_fraction_of_capacity": reference_fraction * activation_share,
        "nonactivation_fraction_of_capacity": reference_fraction
        * (1.0 - activation_share),
        "lora_zero3_nogc_activation_fraction_of_capacity": (
            is_lora * zero3 * (1.0 - gc) * reference_fraction * activation_share
        ),
        "lora_zero3_nogc_truncation_activation_pressure": (
            is_lora
            * zero3
            * (1.0 - gc)
            * reference_fraction
            * activation_share
            * float(values["profile_truncation_fraction"])
        ),
        "lora_zero3_nogc_rare_tail_activation_pressure": (
            is_lora
            * zero3
            * (1.0 - gc)
            * reference_fraction
            * activation_share
            * float(values["profile_rare_tail_gap"])
        ),
        "full_zero3_gc_reference_fraction_of_capacity": (
            (1.0 - is_lora) * zero3 * gc * reference_fraction
        ),
        "full_single_gpu_zero0_nogc_mbs_reference_pressure": (
            (1.0 - is_lora)
            * single_gpu
            * zero0
            * (1.0 - gc)
            * reference_fraction
            * float(values["log2_mbs"])
        ),
        "tied_embedding_gc_reference_pressure": (
            tied * gc * reference_fraction
        ),
    }


def main() -> None:
    import fit_h800_unified_resource_partial_v1 as V3
    from analyze_h800_fresh_memory_residual_v2 import profile_padding_statistics

    artifact = read_json(V3_ARTIFACT)
    expected_names = list(artifact["model"]["raw_feature_names"])
    inventory = read_json(V3_INVENTORY)
    model_by_id = {str(m["id"]): dict(m) for m in inventory["models"]}
    fixed_lora = dict(inventory["fixed_lora"])

    triples = _load_4090_jobs()
    capacity_by_row = {}
    profile_cache: dict = {}
    padding_cache: dict = {}

    built, failures = 0, []
    dataset_counts: Counter = Counter()
    missing_profiles = set()
    feature_dims = set()
    name_mismatch = None
    sample_features = None

    for row, job, normalized in triples:
        dataset_id = str(job["dataset_id"])
        dataset_counts[dataset_id] += 1
        profile_path = PROFILE_DIR / f"{dataset_id}.qwen3_nothink.jsonl"
        if not profile_path.is_file():
            missing_profiles.add(dataset_id)
            failures.append((job.get("job_id"), f"no profile for {dataset_id}"))
            continue
        capacity = int(normalized["hardware"].memory_bytes)
        capacity_by_row[str(job["job_id"])] = capacity
        key = (str(profile_path.resolve()), int(job["cutoff_len"]), int(job["mbs"]))
        if key not in padding_cache:
            padding_cache[key] = profile_padding_statistics(
                profile_path,
                cutoff_len=int(job["cutoff_len"]),
                physical_mbs=int(job["mbs"]),
            )
        try:
            v3_job = v3_job_from_4090(
                job, normalized, padding_stats=padding_cache[key]
            )
            reference, values = V3._current_features(
                v3_job,
                model_by_id=model_by_id,
                fixed_lora=fixed_lora,
                capacity_bytes=capacity,
                profile_cache=profile_cache,
            )
            values.update(
                capacity_normalised_features(
                    values,
                    reference_bytes=reference,
                    capacity_bytes=capacity,
                    model_id=str(job["model_id"]),
                )
            )
            names = sorted(values)
            if name_mismatch is None:
                got, want = set(names), set(expected_names)
                name_mismatch = {
                    "missing_vs_artifact": sorted(want - got),
                    "extra_vs_artifact": sorted(got - want),
                }
            feature_dims.add(len(values))
            if sample_features is None:
                sample_features = {
                    "job_id": str(job["job_id"]),
                    "capacity_gib": round(capacity / 1024 ** 3, 2),
                    "analytic_reference_gib": round(reference / 1024 ** 3, 2),
                    "aligned_effective_sequence": v3_job[
                        "aligned_effective_sequence"
                    ],
                    "cutoff_len": v3_job["cutoff_len"],
                    "reference_fraction_of_capacity": round(
                        values["reference_fraction_of_capacity"], 4
                    ),
                    "activation_share": round(values["activation_share"], 4),
                }
            if not all(math.isfinite(v) for v in values.values()):
                failures.append((job.get("job_id"), "non-finite feature"))
                continue
            built += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(
                (job.get("job_id"), f"{type(exc).__name__}: {exc}")
            )

    print(
        json.dumps(
            {
                "rtx4090_rows_considered": len(triples),
                "v3_raw_features_built": built,
                "failed": len(failures),
                "failure_examples": failures[:6],
                "feature_dims_seen": sorted(feature_dims),
                "artifact_expects_raw_features": len(expected_names),
                "feature_name_check_vs_artifact": name_mismatch,
                "datasets": dict(dataset_counts),
                "datasets_missing_profile": sorted(missing_profiles),
                "distinct_capacity_bytes": sorted(set(capacity_by_row.values())),
                "sample_row": sample_features,
            },
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
