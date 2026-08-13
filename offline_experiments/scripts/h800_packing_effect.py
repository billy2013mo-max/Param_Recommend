#!/usr/bin/env python3
"""Build a non-publishable packing-effect diagnostic from historical pairs.

The historical campaign contains one packed and one unpacked run per pair, not
an ABBA experiment.  This command therefore records the observed paired effect
without enabling packing or feeding the effect into the main planner fit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from common import ROOT, percentile, read_json, sha256_file, sha256_json


SCHEMA = "dsplanner.packing_effect_candidate/v1"
OBSERVATION_SCHEMA = "sft_efficiency_observation/v2"
RECOVERY_SCHEMA = "sft_h800_historical_recovery/v1"


def _read_observations(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("schema") != OBSERVATION_SCHEMA:
                raise ValueError(
                    f"Observation line {line_number} is not {OBSERVATION_SCHEMA}"
                )
            observation_id = row.get("observation_id")
            if not isinstance(observation_id, str) or not observation_id:
                raise ValueError(f"Observation line {line_number} has no id")
            if observation_id in rows:
                raise ValueError(f"Duplicate observation id {observation_id}")
            rows[observation_id] = row
    return rows


def _finite_positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _configuration_key(row: dict[str, Any]) -> dict[str, Any]:
    job = (row.get("configuration") or {}).get("job") or {}
    return {
        "model_id": job.get("model_id"),
        "train_type": job.get("train_type"),
        "dataset_id": job.get("dataset_id"),
        "target_gbs": job.get("target_gbs"),
        "gpu_count": job.get("gpu_count"),
        "zero": job.get("zero"),
        "gc": job.get("gc"),
        "cutoff_len": job.get("cutoff_len"),
    }


def _metric(row: dict[str, Any], key: str) -> float:
    rates = (row.get("measurements") or {}).get("rates") or {}
    return _finite_positive(rates.get(key), key)


def _pair_effect(
    pair_id: str,
    records: list[dict[str, Any]],
    observations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if len(records) != 2:
        raise ValueError(f"Packing pair {pair_id} must contain exactly two rows")
    by_treatment: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for record in records:
        design = record.get("validation_design") or {}
        treatment = design.get("treatment")
        if treatment not in {"packed", "unpacked"} or treatment in by_treatment:
            raise ValueError(f"Packing pair {pair_id} has invalid treatments")
        observation_id = record.get("source_observation_id")
        if observation_id not in observations:
            raise ValueError(f"Packing pair {pair_id} references a missing observation")
        row = observations[observation_id]
        if (row.get("outcome") or {}).get("class") != "success":
            raise ValueError(f"Packing pair {pair_id} contains a non-success run")
        packing = bool(((row.get("configuration") or {}).get("job") or {}).get("packing"))
        if packing != (treatment == "packed"):
            raise ValueError(f"Packing pair {pair_id} treatment disagrees with job")
        by_treatment[treatment] = (record, row)
    if set(by_treatment) != {"packed", "unpacked"}:
        raise ValueError(f"Packing pair {pair_id} is incomplete")

    unpacked_record, unpacked = by_treatment["unpacked"]
    packed_record, packed = by_treatment["packed"]
    if _configuration_key(unpacked) != _configuration_key(packed):
        raise ValueError(f"Packing pair {pair_id} changes a non-packing dimension")
    unpacked_runtime = (unpacked_record.get("runtime") or {}).get(
        "runtime_cohort_id"
    )
    packed_runtime = (packed_record.get("runtime") or {}).get("runtime_cohort_id")
    if not unpacked_runtime or unpacked_runtime != packed_runtime:
        raise ValueError(f"Packing pair {pair_id} crosses runtime cohorts")

    effective_ratio = _metric(packed, "effective_tokens_per_second") / _metric(
        unpacked, "effective_tokens_per_second"
    )
    sample_ratio = _metric(packed, "logical_samples_per_second") / _metric(
        unpacked, "logical_samples_per_second"
    )
    computed_ratio = _metric(packed, "computed_tokens_per_second") / _metric(
        unpacked, "computed_tokens_per_second"
    )
    packed_memory = _finite_positive(
        ((packed.get("measurements") or {}).get("memory") or {}).get(
            "max_reserved_bytes"
        ),
        "packed max_reserved_bytes",
    )
    unpacked_memory = _finite_positive(
        ((unpacked.get("measurements") or {}).get("memory") or {}).get(
            "max_reserved_bytes"
        ),
        "unpacked max_reserved_bytes",
    )
    return {
        "pair_id": pair_id,
        "configuration": {
            **_configuration_key(unpacked),
            "physical_mbs": {
                "unpacked": ((unpacked.get("configuration") or {}).get("job") or {}).get(
                    "mbs"
                ),
                "packed": ((packed.get("configuration") or {}).get("job") or {}).get(
                    "mbs"
                ),
            },
        },
        "runtime_cohort_id": unpacked_runtime,
        "observation_ids": {
            "unpacked": unpacked["observation_id"],
            "packed": packed["observation_id"],
        },
        "job_ids": {
            "unpacked": ((unpacked.get("configuration") or {}).get("job") or {}).get(
                "job_id"
            ),
            "packed": ((packed.get("configuration") or {}).get("job") or {}).get(
                "job_id"
            ),
        },
        "effects": {
            "effective_token_throughput_ratio": effective_ratio,
            "logical_sample_throughput_ratio": sample_ratio,
            "computed_token_throughput_ratio": computed_ratio,
            "log_effective_token_throughput_ratio": math.log(effective_ratio),
            "reserved_memory_ratio": packed_memory / unpacked_memory,
            "reserved_memory_delta_bytes": packed_memory - unpacked_memory,
        },
    }


def build_report(
    observation_path: Path,
    recovery_path: Path,
) -> dict[str, Any]:
    observations = _read_observations(observation_path)
    recovery = read_json(recovery_path)
    if not isinstance(recovery, dict) or recovery.get("schema") != RECOVERY_SCHEMA:
        raise ValueError(f"Recovery report is not {RECOVERY_SCHEMA}")
    records = recovery.get("records")
    if not isinstance(records, list):
        raise ValueError("Recovery report has no records")

    pairs: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        eligibility = record.get("measurement_eligibility") or {}
        if eligibility.get("packing_pair") is not True:
            continue
        design = record.get("validation_design") or {}
        pair_id = design.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("Packing record has no pair id")
        pairs.setdefault(pair_id, []).append(record)

    effects = [
        _pair_effect(pair_id, pairs[pair_id], observations)
        for pair_id in sorted(pairs)
    ]
    ratios = [
        item["effects"]["effective_token_throughput_ratio"] for item in effects
    ]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "gpu_family": "H800",
        "confidence": "low",
        "usage": "diagnostic_prior_only",
        "automatic_enablement_allowed": False,
        "publishable": False,
        "source_bindings": {
            "observations": {
                "path": str(observation_path),
                "sha256": sha256_file(observation_path),
            },
            "historical_recovery": {
                "path": str(recovery_path),
                "sha256": sha256_file(recovery_path),
                "report_sha256": recovery.get("report_sha256"),
            },
            "observation_ids_sha256": sha256_json(
                sorted(
                    observation_id
                    for effect in effects
                    for observation_id in effect["observation_ids"].values()
                )
            ),
        },
        "design": {
            "observed_pairs": len(effects),
            "runs_per_treatment_per_pair": 1,
            "is_abba": False,
        },
        "aggregate": {
            "pairs": len(effects),
            "median_effective_token_throughput_ratio": (
                percentile(ratios, 50) if ratios else None
            ),
            "p10_effective_token_throughput_ratio": (
                percentile(ratios, 10) if ratios else None
            ),
            "minimum_effective_token_throughput_ratio": min(ratios) if ratios else None,
            "maximum_effective_token_throughput_ratio": max(ratios) if ratios else None,
            "pairs_faster_with_packing": sum(ratio > 1 for ratio in ratios),
            "pairs_slower_with_packing": sum(ratio < 1 for ratio in ratios),
        },
        "pairs": effects,
        "blockers": [
            "historical_pairs_are_not_abba",
            "no_independent_packing_holdout",
            "packing_effect_is_dataset_profile_dependent",
            "prospective_acceptance_required",
        ],
    }
    report["report_sha256"] = sha256_json(report)
    return report


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported packing report schema {report.get('schema')}")
    digest = report.get("report_sha256")
    content = dict(report)
    content.pop("report_sha256", None)
    if digest != sha256_json(content):
        raise ValueError("Packing report SHA-256 does not match canonical content")
    if report.get("automatic_enablement_allowed") is not False:
        raise ValueError("Historical single pairs must never enable packing")
    if (report.get("design") or {}).get("is_abba") is not False:
        raise ValueError("Historical packing design must remain non-ABBA")
    pairs = report.get("pairs") or []
    if (report.get("aggregate") or {}).get("pairs") != len(pairs):
        raise ValueError("Packing aggregate pair count mismatch")
    encoded = json.dumps(report, allow_nan=False, sort_keys=True)
    if not encoded:
        raise ValueError("Packing report cannot be empty")


def write_report(path: Path, report: dict[str, Any]) -> None:
    validate_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(
                report,
                output,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        type=Path,
        default=ROOT / "artifacts" / "canonical_h800_observations.jsonl",
    )
    parser.add_argument(
        "--historical-recovery",
        type=Path,
        default=ROOT / "artifacts" / "historical_h800_recovery.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "h800_packing_effect_candidate.json",
    )
    args = parser.parse_args()
    report = build_report(args.observations, args.historical_recovery)
    write_report(args.output, report)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "output": str(args.output),
                "pairs": report["aggregate"]["pairs"],
                "automatic_enablement_allowed": report[
                    "automatic_enablement_allowed"
                ],
                "report_sha256": report["report_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
