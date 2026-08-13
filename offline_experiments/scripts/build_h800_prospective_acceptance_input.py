#!/usr/bin/env python3
"""Join frozen H800 predictions with fresh terminal observations.

This is the result-recovery seam between the training runner and
``prospective_acceptance.py``.  It consumes a queue manifest, a prediction
report frozen before the campaign, and canonical observation JSONL exported
after the campaign.  It never imputes OOM throughput, derives a scale lower
bound from point measurements, launches jobs, or promotes an artifact.

The generated input is intentionally allowed to be ``blocked``.  Missing
scenario/slot joins, stale campaign IDs, incomplete runtime evidence, and
missing explicit cross-card confidence bounds are reported as blockers rather
than silently dropped.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, read_json, sha256_file, sha256_json, write_json


SCHEMA = "sft_prospective_acceptance_input/v1"
QUEUE_SCHEMA = "sft_h800_prospective_queue_manifest/v1"
PREDICTION_SCHEMA = "sft_h800_physical_shares_v4b_prediction/v3"
OBSERVATION_SCHEMA = "sft_efficiency_observation/v2"
DEFAULT_QUEUE = ARTIFACT_DIR / "h800_prospective_queue_manifest_v1.json"
DEFAULT_PREDICTION = ARTIFACT_DIR / "h800_fresh_prediction_report_v1.json"
DEFAULT_OBSERVATIONS = ARTIFACT_DIR / "h800_fresh_observations.jsonl"
DEFAULT_SCALE_EVIDENCE = ARTIFACT_DIR / "h800_fresh_scale_lower_bounds_v1.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "prospective_acceptance_input_v1.json"

QUALITY_REQUIREMENTS = (
    "evidence_verified",
    "event_attempt_binding_complete",
    "terminal_label_verified",
)
CORE_JOB_FIELDS = (
    "model_id",
    "dataset_id",
    "gpu_count",
    "cutoff_len",
    "packing",
    "offload",
)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"observation line {line_number} is not an object")
        rows.append(dict(value))
    return rows


def _job_from_observation(row: Mapping[str, Any]) -> Mapping[str, Any]:
    configuration = row.get("configuration")
    if isinstance(configuration, Mapping) and isinstance(configuration.get("job"), Mapping):
        return configuration["job"]
    if isinstance(configuration, Mapping):
        return configuration
    fingerprint = row.get("fingerprint")
    if isinstance(fingerprint, Mapping) and isinstance(fingerprint.get("bound_job"), Mapping):
        return fingerprint["bound_job"]
    return {}


def _observation_id(row: Mapping[str, Any]) -> str | None:
    job = _job_from_observation(row)
    value = job.get("job_id") or (row.get("observation_identity") or {}).get("job_id")
    return str(value) if value else None


def _outcome(row: Mapping[str, Any]) -> str:
    value = row.get("outcome")
    if isinstance(value, Mapping):
        value = value.get("class") or value.get("label")
    return str(value or "").strip().lower()


def _memory_measurement(row: Mapping[str, Any]) -> int | None:
    measurements = row.get("measurements")
    memory = measurements.get("memory") if isinstance(measurements, Mapping) else None
    if not isinstance(memory, Mapping):
        return None
    value = memory.get("max_reserved_bytes")
    if value is None:
        value = memory.get("max_allocated_bytes")
    number = _finite(value)
    return int(number) if number is not None and number >= 0 else None


def _throughput_measurement(row: Mapping[str, Any]) -> float | None:
    measurements = row.get("measurements")
    rates = measurements.get("rates") if isinstance(measurements, Mapping) else None
    if not isinstance(rates, Mapping):
        return None
    return _finite(rates.get("effective_tokens_per_second"))


def _prediction_rows(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if report.get("schema") != PREDICTION_SCHEMA:
        raise ValueError("prediction report schema mismatch")
    expected_hash = report.get("report_sha256")
    if expected_hash:
        unsigned = dict(report)
        unsigned.pop("report_sha256", None)
        if expected_hash != sha256_json(unsigned):
            raise ValueError("prediction report checksum mismatch")
    rows = report.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("prediction report has no predictions")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("request_id"):
            raise ValueError("prediction row lacks request_id")
        request_id = str(row["request_id"])
        if request_id in indexed:
            raise ValueError(f"duplicate prediction request_id: {request_id}")
        indexed[request_id] = row
    return indexed


def _scale_evidence_rows(payload: Any) -> dict[str, Mapping[str, Any]]:
    if payload is None:
        return {}
    rows = payload.get("pairs") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError("scale evidence must contain a pairs list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("pair_id"):
            raise ValueError("scale evidence row lacks pair_id")
        pair_id = str(row["pair_id"])
        if pair_id in indexed:
            raise ValueError(f"duplicate scale evidence pair_id: {pair_id}")
        indexed[pair_id] = row
    return indexed


def _scenario_contract(prediction_rows: Sequence[Mapping[str, Any]]) -> str | None:
    materials = {
        sha256_json(row.get("scenario_material"))
        for row in prediction_rows
        if row.get("scenario_material") is not None
    }
    runtime = {
        str(row.get("runtime_mechanism_component_sha256"))
        for row in prediction_rows
        if row.get("runtime_mechanism_component_sha256")
    }
    if len(materials) == 1 and len(runtime) == 1:
        return f"{next(iter(materials))}:{next(iter(runtime))}"
    return None


def _prediction_throughput(prediction: Mapping[str, Any]) -> tuple[float | None, float | None, float | None]:
    throughput = prediction.get("throughput")
    if not isinstance(throughput, Mapping):
        throughput = {}
    point = _finite(throughput.get("throughput_proxy_tokens_per_second"))
    lower = _finite(
        throughput.get("conservative_lower_throughput")
        or prediction.get("conservative_lower_throughput")
    )
    upper = _finite(
        throughput.get("conservative_upper_throughput")
        or prediction.get("conservative_upper_throughput")
    )
    return point, lower, upper


def _candidate_row(
    *,
    job: Mapping[str, Any],
    observation: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    outcome = _outcome(observation)
    memory = prediction.get("memory") if isinstance(prediction.get("memory"), Mapping) else {}
    safe_limit = _finite(memory.get("safe_limit_bytes"))
    observed = _memory_measurement(observation)
    actual_safe = bool(
        outcome == "success"
        and observed is not None
        and safe_limit is not None
        and observed <= safe_limit
    )
    point, lower, upper = _prediction_throughput(prediction)
    return {
        "scenario_id": job.get("scenario_id") or prediction.get("comparison_group"),
        "candidate_id": str(job.get("job_id")),
        "job_id": str(job.get("job_id")),
        "candidate_slot_id": job.get("candidate_slot_id"),
        "gpu_count": job.get("gpu_count"),
        "outcome": outcome,
        "predicted_admit": memory.get("admitted") is True,
        "predicted_throughput": point,
        "conservative_lower_throughput": lower,
        "conservative_upper_throughput": upper,
        "observed_throughput": _throughput_measurement(observation),
        "observed_reserved_bytes": observed,
        "upper_reserved_bytes": _finite(
            memory.get("admission_upper_reserved_bytes")
            if memory.get("admission_upper_reserved_bytes") is not None
            else memory.get("operational_p95_reserved_bytes")
        ),
        "safe_limit_bytes": safe_limit,
        "actual_safe_success": actual_safe,
        "scenario_contract_sha256": _scenario_contract([prediction]),
        "runtime_mechanism_component_sha256": prediction.get(
            "runtime_mechanism_component_sha256"
        ),
        "configuration": dict(prediction.get("configuration") or {}),
    }


def _endpoint_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    gpu_count: int,
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("predicted_admit") is True]
    safe = [row for row in candidates if row.get("actual_safe_success") is True]
    ranked = [row for row in candidates if _finite(row.get("predicted_throughput")) is not None]
    ranked.sort(key=lambda row: (-float(row["predicted_throughput"]), str(row.get("candidate_id"))))
    best = ranked[0] if ranked else None
    contract = _scenario_contract(prediction_rows)
    return {
        "gpu_count": int(gpu_count),
        "memory_gate_passed": bool(len(candidates) >= 2 and len(safe) == len(candidates)),
        "admitted_candidate_count": len(candidates),
        "best_candidate_request_id": best.get("candidate_id") if best else None,
        "scenario_contract_sha256": contract,
        "predicted_throughput": best.get("predicted_throughput") if best else None,
        "conservative_lower_throughput": best.get("conservative_lower_throughput") if best else None,
        "conservative_upper_throughput": best.get("conservative_upper_throughput") if best else None,
    }


def build_acceptance_input(
    *,
    queue_manifest: Mapping[str, Any],
    prediction_report: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    scale_evidence: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    strict_quality: bool = True,
) -> dict[str, Any]:
    """Build normalized evaluator input and explicit recovery blockers."""

    blockers: list[str] = []
    if queue_manifest.get("schema") != QUEUE_SCHEMA:
        blockers.append("queue_manifest_schema_mismatch")
    if queue_manifest.get("gpu_training_started") is not False:
        blockers.append("queue_manifest_execution_invariant_drifted")
    # ``queues_mutated=true`` is expected when this manifest was materialized
    # into the explicitly approved JSONL queue.  The adapter is still
    # non-executable; it only requires the source field to be a boolean so a
    # malformed status cannot be mistaken for a clean campaign.
    if not isinstance(queue_manifest.get("queues_mutated"), bool):
        blockers.append("queue_manifest_queue_status_missing")
    jobs = [row for row in queue_manifest.get("jobs") or [] if isinstance(row, Mapping)]
    if not jobs:
        blockers.append("queue_manifest_has_no_jobs")
    job_by_id: dict[str, Mapping[str, Any]] = {}
    for job in jobs:
        job_id = str(job.get("job_id") or "")
        if not job_id:
            blockers.append("queue_job_missing_job_id")
        elif job_id in job_by_id:
            blockers.append(f"duplicate_queue_job:{job_id}")
        else:
            job_by_id[job_id] = job

    predictions = _prediction_rows(prediction_report)
    expected_campaign = str(queue_manifest.get("campaign_id") or "")
    observations_by_id: dict[str, Mapping[str, Any]] = {}
    scenario_by_job: dict[str, str] = {}
    scenario_blockers: list[str] = []
    for observation in observations:
        if observation.get("schema") != OBSERVATION_SCHEMA:
            blockers.append("observation_schema_mismatch")
        job_id = _observation_id(observation)
        if not job_id:
            blockers.append("observation_missing_job_id")
            continue
        if job_id in observations_by_id:
            blockers.append(f"duplicate_observation:{job_id}")
            continue
        observations_by_id[job_id] = observation
        job = _job_from_observation(observation)
        outcome_class = _outcome(observation)
        if outcome_class not in {"success", "oom"}:
            blockers.append(f"unsupported_observation_outcome:{job_id}:{outcome_class or 'missing'}")
        if str(job.get("campaign_id") or "") != expected_campaign:
            blockers.append(f"observation_campaign_mismatch:{job_id}")
        hardware = observation.get("hardware")
        hardware_identity = " ".join(
            str(value or "")
            for value in (
                (hardware or {}).get("gpu_type") if isinstance(hardware, Mapping) else None,
                (hardware or {}).get("gpu_family") if isinstance(hardware, Mapping) else None,
                job.get("gpu_type"),
            )
        ).lower()
        if "h800" not in hardware_identity:
            blockers.append(f"observation_hardware_mismatch:{job_id}")
        scenario_id = str(job.get("scenario_id") or "").strip()
        if not scenario_id:
            scenario_blockers.append(f"missing_observation_scenario_id:{job_id}")
        else:
            scenario_by_job[job_id] = scenario_id
        if strict_quality:
            quality = observation.get("quality") if isinstance(observation.get("quality"), Mapping) else {}
            outcome = observation.get("outcome") if isinstance(observation.get("outcome"), Mapping) else {}
            for field in QUALITY_REQUIREMENTS:
                value = quality.get(field)
                if field == "terminal_label_verified":
                    value = bool(value is True or outcome.get(field) is True)
                if value is not True:
                    blockers.append(f"observation_quality:{job_id}:{field}")
            fingerprint = observation.get("fingerprint") if isinstance(observation.get("fingerprint"), Mapping) else {}
            runtime_fingerprint = (
                observation.get("runtime_mechanism_fingerprint_sha256")
                or fingerprint.get("runtime_mechanism_fingerprint_sha256")
            )
            if runtime_fingerprint is None:
                blockers.append(f"observation_runtime_fingerprint_missing:{job_id}")

    expected_ids = set(job_by_id)
    observed_ids = set(observations_by_id)
    for job_id in sorted(expected_ids - observed_ids):
        blockers.append(f"missing_observation:{job_id}")
    for job_id in sorted(observed_ids - expected_ids):
        blockers.append(f"observation_not_in_queue:{job_id}")
    prediction_ids = set(predictions)
    for job_id in sorted(expected_ids - prediction_ids):
        blockers.append(f"missing_prediction:{job_id}")
    for job_id in sorted(prediction_ids - expected_ids):
        blockers.append(f"prediction_not_in_queue:{job_id}")

    rows: list[dict[str, Any]] = []
    ranking_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job_id in sorted(expected_ids & observed_ids & prediction_ids):
        job = job_by_id[job_id]
        prediction = predictions[job_id]
        observed_job = _job_from_observation(observations_by_id[job_id])
        for field in CORE_JOB_FIELDS:
            if field in job and field in observed_job and job.get(field) != observed_job.get(field):
                blockers.append(f"job_binding_mismatch:{job_id}:{field}")
        scenario_id = str(job.get("scenario_id") or prediction.get("comparison_group") or "").strip()
        if not scenario_id:
            scenario_blockers.append(f"missing_queue_scenario_id:{job_id}")
            continue
        if scenario_by_job.get(job_id) not in (None, scenario_id):
            blockers.append(f"scenario_binding_mismatch:{job_id}")
        candidate = _candidate_row(job=job, observation=observations_by_id[job_id], prediction=prediction)
        candidate["scenario_id"] = scenario_id
        rows.append(candidate)
        ranking_by_scenario[scenario_id].append(candidate)

    scenario_ids = {str(job.get("scenario_id") or "") for job in jobs if job.get("scenario_id")}
    observed_scenario_ids = {scenario_by_job[job_id] for job_id in observed_ids if job_id in scenario_by_job}
    scenario_level_split = bool(scenario_ids and observed_scenario_ids == scenario_ids and not scenario_blockers)
    if scenario_blockers:
        blockers.extend(scenario_blockers)
    fresh_split = bool(
        expected_campaign
        and expected_ids
        and observed_ids == expected_ids
        and all(str(_job_from_observation(observations_by_id[job_id]).get("campaign_id") or "") == expected_campaign for job_id in expected_ids)
    )

    ranking_groups = [
        {"scenario_id": scenario_id, "candidates": sorted(candidates, key=lambda row: str(row["candidate_id"]))}
        for scenario_id, candidates in sorted(ranking_by_scenario.items())
    ]

    scale_rows = _scale_evidence_rows(scale_evidence)
    scale_pairs: list[dict[str, Any]] = []
    jobs_by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for job in jobs:
        if job.get("scenario_id"):
            jobs_by_scenario[str(job["scenario_id"])].append(job)
    row_by_scenario_gpu: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    pred_by_scenario_gpu: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("scenario_id") and isinstance(row.get("gpu_count"), int):
            key = (str(row["scenario_id"]), int(row["gpu_count"]))
            row_by_scenario_gpu[key].append(row)
            pred_by_scenario_gpu[key].append(predictions[str(row["job_id"])])
    for scenario_id, scenario_jobs in sorted(jobs_by_scenario.items()):
        transition = str((scenario_jobs[0].get("scale_out_transition") or "")).strip()
        try:
            from_gpu, to_gpu = (int(part) for part in transition.split("_to_", 1))
        except (TypeError, ValueError):
            blockers.append(f"scale_transition_missing:{scenario_id}")
            continue
        pair_id = f"{scenario_id}:{from_gpu}_to_{to_gpu}"
        baseline_rows = row_by_scenario_gpu.get((scenario_id, from_gpu), [])
        expanded_rows = row_by_scenario_gpu.get((scenario_id, to_gpu), [])
        if not baseline_rows or not expanded_rows:
            blockers.append(f"scale_endpoint_observation_missing:{pair_id}")
        baseline = _endpoint_summary(
            baseline_rows,
            gpu_count=from_gpu,
            prediction_rows=pred_by_scenario_gpu.get((scenario_id, from_gpu), []),
        )
        expanded = _endpoint_summary(
            expanded_rows,
            gpu_count=to_gpu,
            prediction_rows=pred_by_scenario_gpu.get((scenario_id, to_gpu), []),
        )
        evidence = scale_rows.get(pair_id)
        if evidence is None:
            blockers.append(f"scale_measured_lower_bound_missing:{pair_id}")
        scale_pairs.append(
            {
                "pair_id": pair_id,
                "scenario_id": scenario_id,
                "baseline": baseline,
                "expanded": expanded,
                "measured_ratio_lower": evidence.get("measured_ratio_lower") if evidence else None,
                "fresh_measured_ratio_lower": evidence.get("fresh_measured_ratio_lower") if evidence else None,
            }
        )

    return {
        "schema": SCHEMA,
        "campaign_id": expected_campaign,
        "fresh_split": fresh_split,
        "scenario_level_split": scenario_level_split,
        "minimum_memory_coverage": 0.95,
        "maximum_false_safe_oom": 0,
        "maximum_top1_regret": 0.1,
        "minimum_scale_ratio": 1.8,
        "memory_rows": rows,
        "ranking_groups": ranking_groups,
        "scale_pairs": scale_pairs,
        "status": "ready_for_evaluator" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "publication_allowed": False,
        "gpu_training_started": False,
        "queues_mutated": False,
        "required_next_action": (
            "run prospective_acceptance.py on this input; publication remains separately gated"
            if not blockers
            else "resolve result-recovery blockers; do not treat this input as acceptance evidence"
        ),
    }


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _blocked_missing_sources(
    *,
    queue_path: Path,
    prediction_path: Path,
    observations_path: Path,
) -> dict[str, Any]:
    """Return a safe, explicit blocked input before GPU evidence exists."""

    missing = [
        label
        for label, path in (
            ("queue_manifest", queue_path),
            ("prediction_report", prediction_path),
            ("observations", observations_path),
        )
        if not path.is_file()
    ]
    campaign_id = None
    if queue_path.is_file():
        try:
            queue = read_json(queue_path)
            campaign_id = queue.get("campaign_id") if isinstance(queue, Mapping) else None
        except (OSError, ValueError, json.JSONDecodeError):
            missing.append("queue_manifest_parseable")
    blockers = [f"source_file_missing:{label}" for label in missing]
    return {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "fresh_split": False,
        "scenario_level_split": False,
        "minimum_memory_coverage": 0.95,
        "maximum_false_safe_oom": 0,
        "maximum_top1_regret": 0.1,
        "minimum_scale_ratio": 1.8,
        "memory_rows": [],
        "ranking_groups": [],
        "scale_pairs": [],
        "status": "blocked",
        "blockers": sorted(set(blockers)),
        "publication_allowed": False,
        "gpu_training_started": False,
        "queues_mutated": False,
        "required_next_action": "provide frozen prediction report and canonical fresh observations before evaluation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--scale-evidence", type=Path, default=DEFAULT_SCALE_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-unverified-observations",
        action="store_true",
        help="diagnostic only; do not require runtime/evidence quality flags",
    )
    args = parser.parse_args()
    required_sources = (args.queue, args.prediction, args.observations)
    if any(not path.is_file() for path in required_sources):
        result = _blocked_missing_sources(
            queue_path=args.queue,
            prediction_path=args.prediction,
            observations_path=args.observations,
        )
    else:
        queue = read_json(args.queue)
        prediction = read_json(args.prediction)
        observations = _read_jsonl(args.observations)
        scale_payload = read_json(args.scale_evidence) if args.scale_evidence.is_file() else None
        result = build_acceptance_input(
            queue_manifest=queue,
            prediction_report=prediction,
            observations=observations,
            scale_evidence=scale_payload,
            strict_quality=not args.allow_unverified_observations,
        )
    result["source_bindings"] = {
        "queue_manifest": _binding(args.queue),
        "prediction_report": _binding(args.prediction),
        "observations": _binding(args.observations),
        "scale_evidence": _binding(args.scale_evidence) if args.scale_evidence.is_file() else None,
    }
    write_json(args.output, result)
    print(f"wrote {args.output}; status={result['status']} blockers={len(result['blockers'])}")


if __name__ == "__main__":
    main()
