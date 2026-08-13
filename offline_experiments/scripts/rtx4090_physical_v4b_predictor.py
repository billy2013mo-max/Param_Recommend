#!/usr/bin/env python3
"""Inference wrapper for the frozen RTX 4090 physical-shares + v4b model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from common import ROOT, read_json, sha256_file, sha256_json, write_json
from h800_challenger_modeling import _predict_memory_upper
from h800_theory_basis import memory_basis
from joint_throughput_modeling import (
    _predict_two_head_entries,
    _record_features,
    _record_log_analytic_anchor,
)
from rtx4090_physical_v4b_modeling import (
    DEFAULT_OUTPUT as DEFAULT_MODEL_ARTIFACT,
    KERNEL_PATH,
    SCHEMA as MODEL_SCHEMA,
)
from structured_throughput_modeling import _static_structured_basis
from throughput_predictor import ThroughputPredictor


SCHEMA = "sft_rtx4090_physical_shares_v4b_prediction/v1"
IMPLEMENTATION_VERSION = (
    "sft_rtx4090_physical_shares_v4b_predictor/2026-07-29.v1"
)


def _validate_model(report: Mapping[str, Any]) -> None:
    if report.get("schema") != MODEL_SCHEMA:
        raise ValueError("RTX 4090 model artifact schema mismatch")
    unsigned = dict(report)
    digest = unsigned.pop("report_sha256", None)
    if digest != sha256_json(unsigned):
        raise ValueError("RTX 4090 model artifact checksum mismatch")
    if (
        report.get("gpu_experiments_launched") is not False
        or report.get("queues_mutated") is not False
    ):
        raise ValueError("RTX 4090 model safety contract drifted")
    memory = report.get("memory") or {}
    throughput = report.get("throughput") or {}
    frozen_memory = memory.get("frozen_model") or {}
    frozen_throughput = throughput.get("frozen_model") or {}
    if (
        (frozen_memory.get("center") or {}).get("feature_set")
        != "physical_shares"
        or frozen_throughput.get("model_family")
        != "set_aware_two_head_log_throughput_model"
        or (frozen_throughput.get("absolute_head") or {}).get(
            "feature_dimension"
        )
        != 70
    ):
        raise ValueError("RTX 4090 frozen model contract drifted")


def _zero_name(stage: int) -> str:
    return "none" if stage == 0 else f"zero{stage}"


class RTX4090PhysicalV4BPredictor:
    """Filter RTX 4090 candidates for memory safety and rank admitted rows."""

    def __init__(
        self,
        *,
        model_artifact: Path = DEFAULT_MODEL_ARTIFACT,
    ) -> None:
        self.model_artifact_path = Path(model_artifact)
        self.report = read_json(self.model_artifact_path)
        _validate_model(self.report)
        self.memory_center = self.report["memory"]["frozen_model"][
            "center"
        ]
        self.memory_tail = self.report["memory"]["frozen_model"]["tail"]
        self.throughput_model = self.report["throughput"]["frozen_model"]
        self.base = ThroughputPredictor(strict_bindings=False)
        expected_profiles = self.report["source_bindings"][
            "static_dataset_profiles"
        ]
        mismatches = []
        for dataset_id, binding in expected_profiles.items():
            path = (
                self.base.dataset_profile_dir
                / f"{dataset_id}.qwen3_nothink.jsonl"
            )
            if (
                not path.is_file()
                or sha256_file(path) != binding.get("sha256")
            ):
                mismatches.append(dataset_id)
        if mismatches:
            raise ValueError(
                "Frozen dataset-profile bindings changed: "
                + ", ".join(sorted(mismatches))
            )

    @staticmethod
    def _automatic_group(normalized: Mapping[str, Any]) -> str:
        material = {
            "model_id": normalized["model_id"],
            "model_geometry": normalized["model_geometry"],
            "dataset_id": normalized["dataset_id"],
            "training_mode": normalized["training_mode"],
            "target_gbs": normalized["target_gbs"],
            "cutoff_len": normalized["cutoff_len"],
            "dtype": normalized["dtype"],
        }
        return "auto-" + sha256_json(material)[:12]

    def _record(
        self,
        request: Mapping[str, Any],
        *,
        input_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        normalized = self.base._normalized_request(
            {
                **dict(request),
                "hardware_id": "rtx4090",
                "kernel_path": (
                    request.get("kernel_path") or KERNEL_PATH
                ),
            },
            input_index=input_index,
        )
        if normalized["hardware"].card_id != "rtx4090":
            raise ValueError("This predictor accepts RTX 4090 only")
        scenario = {
            "model_id": normalized["model_id"],
            "train_type": normalized["training_mode"],
            "dataset_id": normalized["dataset_id"],
            "target_gbs": normalized["target_gbs"],
            "gpu_count": normalized["gpu_count"],
            "physical_mbs": normalized["physical_mbs"],
            "cutoff_len": normalized["cutoff_len"],
        }
        selector = {
            "runtime_cohort_id": "rtx4090_20260717_fa2",
            "dtype": normalized["dtype"],
            "kernel_path": normalized["kernel_path"],
            "training_mode": normalized["training_mode"],
            "zero_stage": normalized["zero_stage"],
            "gradient_checkpointing": normalized[
                "gradient_checkpointing"
            ],
            "packing": normalized["packing"],
        }
        job = {
            "model_id": normalized["model_id"],
            "model_parameters": normalized["model_geometry"][
                "base_parameters"
            ],
            "train_type": normalized["training_mode"],
            "dataset_id": normalized["dataset_id"],
            "target_gbs": normalized["target_gbs"],
            "gpu_count": normalized["gpu_count"],
            "mbs": normalized["physical_mbs"],
            "cutoff_len": normalized["cutoff_len"],
            "zero": _zero_name(normalized["zero_stage"]),
            "gc": normalized["gradient_checkpointing"],
            "packing": normalized["packing"],
        }
        memory = memory_basis(
            job,
            normalized["model_geometry"],
            int(normalized["hardware"].memory_bytes),
        )
        performance: dict[str, Any] = {
            "physical_priors": normalized["hardware"].physical_priors(),
        }
        if normalized["gradient_accumulation_steps"] is not None:
            performance["gradient_accumulation_steps"] = normalized[
                "gradient_accumulation_steps"
            ]
        record = {
            "schema": "sft_rtx4090_static_inference_record/v1",
            "observation_id": normalized["request_id"],
            "job_id": normalized["request_id"],
            "outcome": "unknown",
            "scenario": scenario,
            "scenario_id": (
                str(request.get("comparison_group"))
                if request.get("comparison_group") is not None
                else self._automatic_group(normalized)
            ),
            "selector": selector,
            "runtime": {
                "runtime_cohort_id": "rtx4090_20260717_fa2",
            },
            "model_basis": normalized["model_geometry"],
            "memory": memory,
            "performance": performance,
        }
        basis = _static_structured_basis(
            record,
            self.base.profiles,
            hardware_memory_bytes=normalized["hardware"].memory_bytes,
        )
        traffic = basis["traffic"]
        limits = basis["component_seconds_at_physical_limits"]
        performance.update(
            {
                "gradient_accumulation_steps": basis["work_evidence"][
                    "gradient_accumulation_steps"
                ],
                "work_per_step": basis["work_per_step"],
                "work_is_per_optimizer_step": True,
                "flops_per_step": basis["flops"],
                "traffic_bytes_per_rank_step": {
                    "kernel_total": traffic["kernel"],
                    "optimizer": traffic["optimizer"],
                },
                "communication": {
                    "payload_bytes_per_rank_step": traffic[
                        "communication"
                    ],
                    "collective_count": traffic["collective_count"],
                },
                "ideal_seconds": {
                    "compute_at_dense_peak": limits["compute"],
                    "kernel_hbm_at_physical_peak": limits[
                        "kernel_hbm"
                    ],
                    "optimizer_hbm_at_physical_peak": limits[
                        "optimizer_hbm"
                    ],
                    "collective_payload_at_link_peak": (
                        float(traffic["communication"])
                        / normalized[
                            "hardware"
                        ].intra_node_bandwidth_bytes_per_second
                    ),
                },
            }
        )
        return record, normalized

    def _memory_result(
        self,
        record: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return the v1 admission decision for one static record.

        Keeping this policy behind a method lets later, checksum-bound model
        generations reuse the request normalization and v4b ranking path
        without changing v1 behavior.
        """

        memory = _predict_memory_upper(
            record,
            self.memory_center,
            self.memory_tail,
        )
        available = memory.get("available") is True
        safe_limit = float(record["memory"]["safe_limit_bytes"])
        admitted = bool(
            available
            and float(memory["operational_p95_reserved_bytes"])
            <= safe_limit
        )
        return {
            "prediction_available": available,
            "reserved_center_bytes": memory.get(
                "reserved_center_bytes"
            ),
            "operational_p95_reserved_bytes": memory.get(
                "operational_p95_reserved_bytes"
            ),
            "safe_limit_bytes": safe_limit,
            "admitted": admitted,
            "tail_source": memory.get("success_tail_source"),
            "issues": memory.get("issues") or [],
        }

    def predict(
        self,
        requests: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not requests:
            raise ValueError("At least one candidate is required")
        rows = []
        candidates = []
        for input_index, request in enumerate(requests):
            record, normalized = self._record(
                request,
                input_index=input_index,
            )
            memory_result = self._memory_result(record)
            admitted = bool(memory_result["admitted"])
            row = {
                "input_index": input_index,
                "request_id": normalized["request_id"],
                "comparison_group": record["scenario_id"],
                "configuration": {
                    "model_id": normalized["model_id"],
                    "training_mode": normalized["training_mode"],
                    "dataset_id": normalized["dataset_id"],
                    "target_gbs": normalized["target_gbs"],
                    "cutoff_len": normalized["cutoff_len"],
                    "gpu_count": normalized["gpu_count"],
                    "mbs": normalized["physical_mbs"],
                    "zero_stage": normalized["zero_stage"],
                    "gradient_checkpointing": normalized[
                        "gradient_checkpointing"
                    ],
                    "packing": normalized["packing"],
                },
                "memory": memory_result,
                "throughput": {
                    "prediction_available": False,
                    "reason": (
                        "pending_candidate_set_ranking"
                        if admitted
                        else "rejected_by_memory_p95"
                    ),
                },
                "rank_within_admitted_group": None,
            }
            rows.append(row)
            if admitted:
                effective_work = float(
                    record["performance"]["work_per_step"][
                        "effective_tokens"
                    ]
                )
                candidates.append(
                    {
                        "scenario_id": record["scenario_id"],
                        "scenario": {
                            "model_id": normalized["model_id"],
                            "training_mode": normalized[
                                "training_mode"
                            ],
                            "dataset_id": normalized["dataset_id"],
                            "target_gbs": normalized["target_gbs"],
                        },
                        "candidate_key": [
                            normalized["gpu_count"],
                            normalized["physical_mbs"],
                            normalized["zero_stage"],
                            normalized["gradient_checkpointing"],
                            normalized["packing"],
                            normalized["dtype"],
                            normalized["kernel_path"],
                        ],
                        "record": record,
                        "features": _record_features(record),
                        "effective_log_work": math.log(effective_work),
                        "log_analytic_anchor": (
                            _record_log_analytic_anchor(record)
                        ),
                        "request_id": normalized["request_id"],
                        "input_index": input_index,
                    }
                )

        predicted = _predict_two_head_entries(
            candidates,
            self.throughput_model,
        )
        for candidate, log_rate in predicted:
            rate = math.exp(float(log_rate))
            work = math.exp(float(candidate["effective_log_work"]))
            row = rows[int(candidate["input_index"])]
            row["throughput"] = {
                "prediction_available": True,
                "predicted_effective_tokens_per_second": rate,
                "predicted_step_seconds": work / rate,
                "static_effective_tokens_per_step": work,
                "candidate_set_dependent": True,
                "reason": None,
            }

        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["throughput"]["prediction_available"] is True:
                by_group[str(row["comparison_group"])].append(row)
        groups = []
        all_group_ids = sorted(
            {str(row["comparison_group"]) for row in rows}
        )
        for group_id in all_group_ids:
            admitted = sorted(
                by_group.get(group_id, []),
                key=lambda row: (
                    -float(
                        row["throughput"][
                            "predicted_effective_tokens_per_second"
                        ]
                    ),
                    int(row["input_index"]),
                ),
            )
            for rank, row in enumerate(admitted, start=1):
                row["rank_within_admitted_group"] = rank
            requested = [
                row
                for row in rows
                if str(row["comparison_group"]) == group_id
            ]
            groups.append(
                {
                    "comparison_group": group_id,
                    "requested_candidates": len(requested),
                    "admitted_candidates": len(admitted),
                    "status": (
                        "ranked" if admitted else "no_memory_safe_candidate"
                    ),
                    "ranked_request_ids": [
                        str(row["request_id"]) for row in admitted
                    ],
                }
            )
        report: dict[str, Any] = {
            "schema": SCHEMA,
            "implementation_version": IMPLEMENTATION_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_artifact": {
                "path": str(self.model_artifact_path.resolve()),
                "sha256": sha256_file(self.model_artifact_path),
                "report_sha256": self.report["report_sha256"],
            },
            "hardware_id": "rtx4090",
            "policy": (
                "physical-shares operational P95 filter, then v4b "
                "candidate-set recentering and ranking"
            ),
            "gpu_experiments_launched": False,
            "queues_mutated": False,
            "predictions": rows,
            "ranking_groups": groups,
        }
        report["report_sha256"] = sha256_json(report)
        return report


def _load_requests(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = payload.get("candidates") or payload.get("requests")
    else:
        rows = None
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Input must be a non-empty list or contain candidates/requests"
        )
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("Every predictor candidate must be an object")
    return [dict(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=DEFAULT_MODEL_ARTIFACT,
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    predictor = RTX4090PhysicalV4BPredictor(
        model_artifact=args.model_artifact
    )
    report = predictor.predict(_load_requests(args.input))
    if args.output is not None:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
