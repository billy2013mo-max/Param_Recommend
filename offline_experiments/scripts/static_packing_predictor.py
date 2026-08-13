#!/usr/bin/env python3
"""Pure-static packing decisions from token lengths and a frozen policy.

The online path in this module never launches training or reads runtime GPU
measurements.  It reproduces LLaMA-Factory's neat-packing knapsack scope,
estimates the no-packing padding baseline, applies frozen monotonic gates, and
fails closed outside the released enablement domain.
"""

from __future__ import annotations

import argparse
import bisect
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile
from typing import Any

from common import percentile, sha256_file, sha256_json, stable_id
from packing_gbs_contract import derive_packing_gbs_contract


POLICY_SCHEMA = "sft_static_packing_policy/v1"
REQUEST_SCHEMA = "sft_static_packing_requests/v1"
DECISION_SCHEMA = "sft_static_packing_decision/v1"
PREDICTIONS_SCHEMA = "sft_static_packing_predictions/v1"
REPLAY_SCHEMA = "sft_static_packing_calibration_replay/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _finite_positive(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def load_policy(path: Path) -> dict[str, Any]:
    policy = _read_json(path)
    if not isinstance(policy, dict) or policy.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"Policy must use schema {POLICY_SCHEMA}")
    if policy.get("online_gpu_execution_allowed") is not False:
        raise ValueError("Static policy must prohibit online GPU execution")
    if policy.get("decision_semantics", {}).get("fail_closed") is not True:
        raise ValueError("Static policy must fail closed")
    if _positive_integer(policy.get("packed_physical_mbs"), "packed_physical_mbs") != 1:
        raise ValueError("Version 1 only supports packed physical MBS=1")
    return policy


def _search_for_fit(numbers: list[int], capacity: int) -> int:
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else index - 1


def greedy_knapsack(numbers: list[int], capacity: int) -> list[list[int]]:
    """Mirror LLaMA-Factory's largest-fitting-item greedy implementation."""
    remaining = sorted(numbers)
    knapsacks: list[list[int]] = []
    while remaining:
        current: list[int] = []
        available = capacity
        while True:
            index = _search_for_fit(remaining, available)
            if index == -1:
                break
            value = remaining.pop(index)
            current.append(value)
            available -= value
        if not current:
            raise ValueError("A profiled sequence exceeds the packing capacity")
        knapsacks.append(current)
    return knapsacks


def _load_lengths(
    profile_path: Path,
    *,
    length_field: str,
) -> list[int]:
    lengths: list[int] = []
    with profile_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{profile_path}:{line_number} is not an object")
            value = row.get(length_field)
            lengths.append(
                _positive_integer(
                    value,
                    f"{profile_path}:{line_number}.{length_field}",
                )
            )
    if not lengths:
        raise ValueError(f"Profile contains no lengths: {profile_path}")
    return lengths


def _pack_in_worker_shards(
    lengths: list[int],
    *,
    capacity: int,
    workers: int,
) -> list[list[int]]:
    worker_count = min(workers, len(lengths))
    knapsacks: list[list[int]] = []
    for worker in range(worker_count):
        start = len(lengths) * worker // worker_count
        stop = len(lengths) * (worker + 1) // worker_count
        knapsacks.extend(greedy_knapsack(lengths[start:stop], capacity))
    return knapsacks


def _unpacked_padding_utilization(
    lengths: list[int],
    *,
    mbs: int,
    seeds: list[int],
) -> dict[str, Any]:
    runs: list[float] = []
    for seed in seeds:
        shuffled = lengths.copy()
        random.Random(seed).shuffle(shuffled)
        effective = sum(shuffled)
        computed = 0
        for start in range(0, len(shuffled), mbs):
            batch = shuffled[start : start + mbs]
            computed += max(batch) * len(batch)
        runs.append(effective / computed)
    return {
        "mean": statistics.fmean(runs),
        "minimum": min(runs),
        "maximum": max(runs),
        "seeds": seeds,
    }


def _derive_ga(
    *,
    target_gbs: float,
    data_parallel: int,
    effective_mbs: float,
) -> dict[str, Any]:
    raw = target_gbs / (data_parallel * effective_mbs)
    candidates = sorted(
        {
            max(1, math.floor(raw)),
            max(1, math.ceil(raw)),
        }
    )
    gradient_accumulation_steps = min(
        candidates,
        key=lambda ga: (
            abs(data_parallel * ga * effective_mbs - target_gbs),
            ga,
        ),
    )
    expected = data_parallel * gradient_accumulation_steps * effective_mbs
    return {
        "target_gbs": target_gbs,
        "data_parallel": data_parallel,
        "raw_gradient_accumulation_steps": raw,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "expected_sample_gbs": expected,
        "relative_error": abs(expected - target_gbs) / target_gbs,
    }


def _profile_features(
    request: dict[str, Any],
    policy: dict[str, Any],
    profile_path: Path,
) -> dict[str, Any]:
    cutoff_len = _positive_integer(request.get("cutoff_len"), "cutoff_len")
    no_packing_mbs = _positive_integer(
        request.get("no_packing_mbs"),
        "no_packing_mbs",
    )
    data_parallel = _positive_integer(
        request.get("data_parallel", request.get("gpu_count")),
        "data_parallel",
    )
    target_gbs = _finite_positive(request.get("target_gbs"), "target_gbs")
    workers = _positive_integer(
        request.get(
            "preprocessing_num_workers",
            policy["packing_algorithm"]["preprocessing_num_workers"],
        ),
        "preprocessing_num_workers",
    )
    length_field = str(
        request.get(
            "length_field",
            policy["profile_contract"]["length_field"],
        )
    )
    raw_lengths = _load_lengths(profile_path, length_field=length_field)
    capacity = cutoff_len - 1
    if capacity <= 0:
        raise ValueError("cutoff_len must be at least 2")

    no_packing_lengths = [min(length, cutoff_len) for length in raw_lengths]
    packing_lengths = [min(length, capacity) for length in raw_lengths]
    knapsacks = _pack_in_worker_shards(
        packing_lengths,
        capacity=capacity,
        workers=workers,
    )
    used_tokens = [sum(knapsack) for knapsack in knapsacks]
    counts = [len(knapsack) for knapsack in knapsacks]
    records = len(raw_lengths)
    packs = len(knapsacks)
    effective_mbs = records / packs
    pack_utilization = sum(used_tokens) / (packs * capacity)
    model_facing_fill = sum(used_tokens) / (packs * cutoff_len)
    padding = _unpacked_padding_utilization(
        no_packing_lengths,
        mbs=no_packing_mbs,
        seeds=[
            int(seed) for seed in policy["profile_contract"]["padding_shuffle_seeds"]
        ],
    )
    ga = _derive_ga(
        target_gbs=target_gbs,
        data_parallel=data_parallel,
        effective_mbs=effective_mbs,
    )
    samples_per_pack = {
        "minimum": min(counts),
        "mean": effective_mbs,
        "standard_deviation": statistics.pstdev(counts),
        "p10": percentile(counts, 10),
        "p50": percentile(counts, 50),
        "p90": percentile(counts, 90),
        "p95": percentile(counts, 95),
        "p99": percentile(counts, 99),
        "maximum": max(counts),
    }
    gbs_contract_v2 = derive_packing_gbs_contract(
        target_gbs=target_gbs,
        data_parallel=data_parallel,
        samples_per_pack=samples_per_pack,
        epsilon_gbs=float(request.get("epsilon_gbs", 0.10)),
        maximum_center_relative_error=float(
            policy["hard_gates"]["maximum_expected_gbs_relative_error"]
        ),
    )
    return {
        "records": records,
        "raw_length_tokens": {
            "minimum": min(raw_lengths),
            "mean": statistics.fmean(raw_lengths),
            "p50": percentile(raw_lengths, 50),
            "p90": percentile(raw_lengths, 90),
            "p95": percentile(raw_lengths, 95),
            "p99": percentile(raw_lengths, 99),
            "maximum": max(raw_lengths),
        },
        "cutoff_len": cutoff_len,
        "packing_capacity": capacity,
        "samples_truncated_without_packing": sum(
            length > cutoff_len for length in raw_lengths
        ),
        "sample_truncation_rate_without_packing": (
            sum(length > cutoff_len for length in raw_lengths) / records
        ),
        "tokens_retained_ratio_without_packing": (
            sum(no_packing_lengths) / sum(raw_lengths)
        ),
        "preprocessing_num_workers": min(workers, records),
        "packs": packs,
        "pack_utilization": pack_utilization,
        "model_facing_pack_fill_ratio": model_facing_fill,
        "sequence_reduction_ratio": 1.0 - packs / records,
        "mean_samples_per_pack": effective_mbs,
        "samples_per_pack": samples_per_pack,
        "used_tokens_per_pack": {
            "minimum": min(used_tokens),
            "maximum": max(used_tokens),
        },
        "no_packing_mbs": no_packing_mbs,
        "no_packing_padding_utilization": padding,
        "linear_token_efficiency_ratio": (model_facing_fill / padding["mean"]),
        "cutoff_tokens_per_no_packing_mbs": (cutoff_len / no_packing_mbs),
        "packed_batch_geometry": ga,
        "packed_batch_geometry_v2": gbs_contract_v2,
    }


def _released_domain_mismatches(
    request: dict[str, Any],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    domain = policy["released_enablement_domain"]
    dimensions = {
        "gpu_family": "gpu_families",
        "modality": "modalities",
        "stage": "stages",
        "dtype": "dtypes",
        "model_id": "model_ids",
        "train_type": "train_types",
    }
    mismatches: list[dict[str, Any]] = []
    for request_key, policy_key in dimensions.items():
        actual = request.get(request_key)
        allowed = domain[policy_key]
        if actual not in allowed:
            mismatches.append(
                {
                    "dimension": request_key,
                    "actual": actual,
                    "allowed": allowed,
                }
            )
    cutoff_len = request.get("cutoff_len")
    cutoff_domain = domain["cutoff_len"]
    if (
        not isinstance(cutoff_len, int)
        or isinstance(cutoff_len, bool)
        or cutoff_len < cutoff_domain["minimum"]
        or cutoff_len > cutoff_domain["maximum"]
    ):
        mismatches.append(
            {
                "dimension": "cutoff_len",
                "actual": cutoff_len,
                "allowed": cutoff_domain,
            }
        )
    return mismatches


def _gate(
    *,
    name: str,
    actual: float,
    threshold: float,
    comparator: str,
) -> dict[str, Any]:
    if comparator == ">=":
        passed = actual >= threshold
    elif comparator == "<=":
        passed = actual <= threshold
    else:
        raise ValueError(f"Unsupported gate comparator {comparator}")
    return {
        "name": name,
        "actual": actual,
        "comparator": comparator,
        "threshold": threshold,
        "passed": passed,
    }


def build_decision(
    request: dict[str, Any],
    *,
    policy: dict[str, Any],
    policy_path: Path,
    request_base: Path,
) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("Each request must be an object")
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("Each request needs a non-empty request_id")

    modality = request.get("modality")
    algorithm_id = request.get(
        "packing_algorithm_id",
        policy["packing_algorithm"]["id"],
    )
    configured_workers = request.get(
        "preprocessing_num_workers",
        policy["packing_algorithm"]["preprocessing_num_workers"],
    )
    structural_reasons: list[str] = []
    if modality != "text":
        structural_reasons.append("multimodal_packing_not_supported")
    if algorithm_id != policy["packing_algorithm"]["id"]:
        structural_reasons.append("packing_algorithm_mismatch")
    if configured_workers != policy["packing_algorithm"]["preprocessing_num_workers"]:
        structural_reasons.append("preprocessing_worker_count_mismatch")

    profile_binding: dict[str, Any] | None = None
    features: dict[str, Any] | None = None
    gate_evaluations: list[dict[str, Any]] = []
    if not structural_reasons:
        profile_value = request.get("profile_path")
        if not isinstance(profile_value, str) or not profile_value:
            raise ValueError(f"{request_id}: text request needs profile_path")
        profile_path = Path(profile_value).expanduser()
        if not profile_path.is_absolute():
            profile_path = (request_base / profile_path).resolve()
        if not profile_path.is_file():
            raise FileNotFoundError(profile_path)
        profile_binding = {
            "path": str(profile_path),
            "sha256": sha256_file(profile_path),
            "length_field": request.get(
                "length_field",
                policy["profile_contract"]["length_field"],
            ),
        }
        features = _profile_features(request, policy, profile_path)
        gates = policy["hard_gates"]
        efficiency_threshold = (
            gates["minimum_linear_token_efficiency_when_no_packing_mbs_is_one"]
            if features["no_packing_mbs"] == 1
            else gates[
                "minimum_linear_token_efficiency_when_no_packing_mbs_exceeds_one"
            ]
        )
        gate_evaluations = [
            _gate(
                name="minimum_records",
                actual=features["records"],
                comparator=">=",
                threshold=policy["profile_contract"]["minimum_records"],
            ),
            _gate(
                name="pack_utilization",
                actual=features["pack_utilization"],
                comparator=">=",
                threshold=gates["minimum_pack_utilization"],
            ),
            _gate(
                name="sequence_reduction_ratio",
                actual=features["sequence_reduction_ratio"],
                comparator=">=",
                threshold=gates["minimum_sequence_reduction_ratio"],
            ),
            _gate(
                name="mean_samples_per_pack",
                actual=features["mean_samples_per_pack"],
                comparator=">=",
                threshold=gates["minimum_mean_samples_per_pack"],
            ),
            _gate(
                name="expected_gbs_relative_error",
                actual=features["packed_batch_geometry"]["relative_error"],
                comparator="<=",
                threshold=gates["maximum_expected_gbs_relative_error"],
            ),
            _gate(
                name="cutoff_tokens_per_no_packing_mbs",
                actual=features["cutoff_tokens_per_no_packing_mbs"],
                comparator=">=",
                threshold=gates["minimum_cutoff_tokens_per_no_packing_mbs"],
            ),
            _gate(
                name="linear_token_efficiency_ratio",
                actual=features["linear_token_efficiency_ratio"],
                comparator=">=",
                threshold=efficiency_threshold,
            ),
        ]
        if request.get("require_gbs_controllable") is True:
            gate_evaluations.append(
                _gate(
                    name="gbs_controllable_at_ga_floor",
                    actual=float(
                        features["packed_batch_geometry_v2"]["gates"][
                            "gbs_controllable_at_ga_floor"
                        ]
                    ),
                    comparator=">=",
                    threshold=1.0,
                )
            )

    static_candidate = (
        not structural_reasons
        and bool(gate_evaluations)
        and all(item["passed"] for item in gate_evaluations)
    )
    domain_mismatches = _released_domain_mismatches(request, policy)
    if structural_reasons:
        decision = "off"
        packing = False
        support_status = "unsupported_structural_off"
        confidence = "high"
        automatic_action_allowed = True
    elif not static_candidate:
        decision = "off"
        packing = False
        support_status = (
            "released_off"
            if not domain_mismatches
            else "fail_closed_off_outside_enablement_domain"
        )
        confidence = "medium"
        automatic_action_allowed = True
    elif domain_mismatches:
        decision = "hold_off"
        packing = False
        support_status = "shadow_positive_outside_enablement_domain"
        confidence = "low"
        automatic_action_allowed = False
    else:
        decision = "on"
        packing = True
        support_status = "released_on"
        confidence = "medium"
        automatic_action_allowed = True

    failed_gates = [item["name"] for item in gate_evaluations if not item["passed"]]
    reason_codes = (
        structural_reasons
        or failed_gates
        or (
            ["outside_released_enablement_domain"]
            if domain_mismatches
            else ["all_static_gates_passed"]
        )
    )
    identity = {
        "request_id": request_id,
        "policy_id": policy["policy_id"],
        "profile_sha256": (profile_binding["sha256"] if profile_binding else None),
        "request": request,
    }
    decision_report: dict[str, Any] = {
        "schema": DECISION_SCHEMA,
        "decision_id": stable_id("packing", identity),
        "request_id": request_id,
        "policy": {
            "policy_id": policy["policy_id"],
            "version": policy["version"],
            "path": str(policy_path.resolve()),
            "sha256": sha256_file(policy_path),
            "status": policy["status"],
        },
        "profile_binding": profile_binding,
        "recommendation": {
            "packing": packing,
            "decision": decision,
            "support_status": support_status,
            "confidence": confidence,
            "automatic_action_allowed": automatic_action_allowed,
            "shadow_candidate_packing": (True if decision == "hold_off" else None),
            "reason_codes": reason_codes,
        },
        "features": features,
        "gate_evaluations": gate_evaluations,
        "released_domain_mismatches": domain_mismatches,
        "operational_contract": {
            "prediction_kind": "pure_static",
            "pre_recommendation_gpu_execution": False,
            "runtime_measurements_consumed": False,
            "fail_closed": True,
        },
    }
    decision_report["report_sha256"] = sha256_json(decision_report)
    return decision_report


def build_predictions(
    request_path: Path,
    *,
    policy: dict[str, Any],
    policy_path: Path,
) -> dict[str, Any]:
    document = _read_json(request_path)
    if not isinstance(document, dict) or document.get("schema") != REQUEST_SCHEMA:
        raise ValueError(f"Request document must use schema {REQUEST_SCHEMA}")
    requests = document.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("Request document must contain a non-empty requests list")
    decisions = [
        build_decision(
            request,
            policy=policy,
            policy_path=policy_path,
            request_base=request_path.parent,
        )
        for request in requests
    ]
    report: dict[str, Any] = {
        "schema": PREDICTIONS_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_binding": {
            "path": str(policy_path.resolve()),
            "sha256": sha256_file(policy_path),
            "policy_id": policy["policy_id"],
        },
        "request_binding": {
            "path": str(request_path.resolve()),
            "sha256": sha256_file(request_path),
        },
        "online_gpu_execution_performed": False,
        "decisions": decisions,
    }
    report["report_sha256"] = sha256_json(report)
    return report


def _verify_policy_source_bindings(policy: dict[str, Any]) -> None:
    bindings = policy["calibration"]["source_bindings"]
    for name, binding in bindings.items():
        path = Path(binding["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Missing calibration source {name}: {path}")
        actual = sha256_file(path)
        if actual != binding["sha256"]:
            raise ValueError(
                f"Calibration source {name} changed: expected "
                f"{binding['sha256']}, got {actual}"
            )


def build_calibration_replay(
    *,
    policy: dict[str, Any],
    policy_path: Path,
    stage_decisions_path: Path,
    packing_requests_path: Path,
    dataset_profile_dir: Path,
) -> dict[str, Any]:
    _verify_policy_source_bindings(policy)
    stage = _read_json(stage_decisions_path)["packing"]["decisions"]
    request_rows = {
        row["request_id"]: row for row in _read_jsonl(packing_requests_path)
    }
    rows: list[dict[str, Any]] = []
    for observed in stage:
        if observed["status"] != "complete":
            continue
        experiment = request_rows[observed["request_id"]]
        profile_path = (
            dataset_profile_dir / f"{experiment['dataset_id']}.qwen3_nothink.jsonl"
        )
        request = {
            "request_id": observed["request_id"],
            "gpu_family": "H800",
            "modality": "text",
            "stage": "sft",
            "dtype": "bf16",
            "model_id": experiment["model_id"],
            "train_type": experiment["train_type"],
            "profile_path": str(profile_path),
            "cutoff_len": experiment["cutoff_len"],
            "no_packing_mbs": observed["baseline_mbs"],
            "gpu_count": experiment["gpu_count"],
            "data_parallel": experiment["gpu_count"],
            "target_gbs": experiment["target_gbs"],
            "preprocessing_num_workers": 8,
            "packing_algorithm_id": policy["packing_algorithm"]["id"],
        }
        prediction = build_decision(
            request,
            policy=policy,
            policy_path=policy_path,
            request_base=PROJECT_ROOT,
        )
        predicted = prediction["recommendation"]["decision"]
        expected = observed["decision"]
        rows.append(
            {
                "request_id": observed["request_id"],
                "model_id": experiment["model_id"],
                "train_type": experiment["train_type"],
                "dataset_id": experiment["dataset_id"],
                "cutoff_len": experiment["cutoff_len"],
                "no_packing_mbs": observed["baseline_mbs"],
                "observed_time_gain": observed["median_time_gain"],
                "observed_decision": expected,
                "predicted_decision": predicted,
                "correct": predicted == expected,
                "features": {
                    key: prediction["features"][key]
                    for key in (
                        "pack_utilization",
                        "sequence_reduction_ratio",
                        "mean_samples_per_pack",
                        "linear_token_efficiency_ratio",
                        "cutoff_tokens_per_no_packing_mbs",
                    )
                },
                "failed_gates": [
                    gate["name"]
                    for gate in prediction["gate_evaluations"]
                    if not gate["passed"]
                ],
            }
        )

    true_positive = sum(
        row["observed_decision"] == "on" and row["predicted_decision"] == "on"
        for row in rows
    )
    false_positive = sum(
        row["observed_decision"] == "off" and row["predicted_decision"] == "on"
        for row in rows
    )
    true_negative = sum(
        row["observed_decision"] == "off" and row["predicted_decision"] == "off"
        for row in rows
    )
    false_negative = sum(
        row["observed_decision"] == "on" and row["predicted_decision"] != "on"
        for row in rows
    )
    report: dict[str, Any] = {
        "schema": REPLAY_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_binding": {
            "path": str(policy_path.resolve()),
            "sha256": sha256_file(policy_path),
            "policy_id": policy["policy_id"],
        },
        "evaluation_kind": "calibration_resubstitution_not_holdout",
        "online_gpu_execution_performed": False,
        "metrics": {
            "points": len(rows),
            "correct": sum(row["correct"] for row in rows),
            "accuracy": (
                sum(row["correct"] for row in rows) / len(rows) if rows else None
            ),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
            "precision_on": (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else None
            ),
            "recall_on": (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else None
            ),
        },
        "limitations": [
            "The thresholds were calibrated from these same ten paired points.",
            "Each treatment has only one historical run and is not ABBA.",
            "There is no independent packing holdout.",
            "Full SFT, multimodal workloads, other GPU families, and new model families are not promoted by this replay.",
        ],
        "rows": rows,
    }
    report["report_sha256"] = sha256_json(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=PROJECT_ROOT
        / "offline_experiments"
        / "artifacts"
        / "static_packing_policy_v1.json",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--requests", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)

    replay = subparsers.add_parser("replay")
    replay.add_argument(
        "--stage-decisions",
        type=Path,
        default=PROJECT_ROOT
        / "offline_experiments"
        / "artifacts"
        / "stage_decisions.json",
    )
    replay.add_argument(
        "--packing-requests",
        type=Path,
        default=PROJECT_ROOT
        / "offline_experiments"
        / "matrix"
        / "packing_pair_requests.jsonl",
    )
    replay.add_argument(
        "--dataset-profile-dir",
        type=Path,
        default=PROJECT_ROOT / "offline_experiments" / "artifacts" / "dataset_profiles",
    )
    replay.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    policy_path = args.policy.expanduser().resolve()
    policy = load_policy(policy_path)
    if args.command == "predict":
        report = build_predictions(
            args.requests.expanduser().resolve(),
            policy=policy,
            policy_path=policy_path,
        )
    else:
        report = build_calibration_replay(
            policy=policy,
            policy_path=policy_path,
            stage_decisions_path=args.stage_decisions.expanduser().resolve(),
            packing_requests_path=args.packing_requests.expanduser().resolve(),
            dataset_profile_dir=args.dataset_profile_dir.expanduser().resolve(),
        )
    _write_json_atomic(args.output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
