#!/usr/bin/env python3
"""Replay every frozen H800 throughput model on later experiments.

The comparison deliberately reconstructs every workload from information that
was available before a run started: model geometry, static dataset profiles,
hardware constants, and the requested training configuration.  Observed
throughput and step time are used only as evaluation labels.

There are two complementary evaluation cohorts:

* ``common_unseen_model_holdout`` uses all currently successful Qwen3-VL/32B
  campaign rows.  It gives every model the same rows, but some early Qwen3-VL
  outcomes predate the newest model artifacts.
* ``strict_post_artifact`` uses only rows that finished after the corresponding
  artifact was frozen.  It is temporally cleaner, but the number of rows differs
  across model generations.

The RTX4090 branch is catalogued but is not scored on H800 outcomes.  Its
configuration-only ranker was validated by CV but no full-data coefficient
vector was serialized, so reconstructing it here would silently train a new
model rather than replay the historical one.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any

from common import (
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)
from h800_challenger_modeling import (
    _physics_throughput_score,
    _ranker_score,
)
from joint_throughput_modeling import (
    _predict_log_throughput as _predict_joint_log_throughput,
    _predict_two_head_entries,
    _record_features,
    _record_log_analytic_anchor,
)
from structured_throughput_modeling import _static_structured_basis
from throughput_predictor import ThroughputPredictor
from validate_throughput_predictor_generalization import (
    _absolute_metrics,
    _ranking_metrics,
)


SCHEMA = "sft_throughput_model_generation_replay/v1"
IMPLEMENTATION_VERSION = (
    "sft_throughput_model_generation_replay/2026-07-28.v1"
)

THEORY_ARTIFACT = ROOT / "artifacts" / "h800_theory_calibration.json"
HYBRID_ARTIFACT = ROOT / "artifacts" / "h800_challenger_modeling.json"
PURE_ARTIFACT = ROOT / "artifacts" / "h800_pure_ranker_modeling.json"
JOINT_ARTIFACT = ROOT / "artifacts" / "joint_throughput_modeling.json"
STRUCTURED_ARTIFACT = (
    ROOT / "artifacts" / "structured_throughput_modeling.json"
)
RTX4090_ARTIFACT = (
    ROOT
    / "campaigns"
    / "rtx4090_20260717"
    / "artifacts"
    / "rtx4090_challenger_modeling.json"
)
DEFAULT_LEDGER = (
    ROOT
    / "artifacts"
    / "throughput_predictor_generalization_ledger_2026-07-28.json"
)
DEFAULT_VALIDATION = (
    ROOT
    / "artifacts"
    / "throughput_predictor_generalization_validation_2026-07-28.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "throughput_model_generations_generalization_2026-07-28.json"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_time(
    path: Path,
    payload: Mapping[str, Any],
) -> tuple[str, str]:
    generated = payload.get("generated_at_utc")
    if generated:
        return str(generated), "artifact.generated_at_utc"
    return (
        datetime.fromtimestamp(
            path.stat().st_mtime,
            timezone.utc,
        ).isoformat(),
        "filesystem_mtime_artifact_has_no_generated_at",
    )


def _iso_to_unix(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    if "report_sha256" in payload:
        raise ValueError("report_sha256 must be absent before sealing")
    result = dict(payload)
    result["report_sha256"] = sha256_json(result)
    return result


def _request(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **job,
        "request_id": str(job["job_id"]),
        "hardware_id": "h800",
        "dtype": "bf16",
    }


def _static_record(
    predictor: ThroughputPredictor,
    job: Mapping[str, Any],
    *,
    input_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = predictor._normalized_request(
        _request(job),
        input_index=input_index,
    )
    performance: dict[str, Any] = {
        "physical_priors": normalized["hardware"].physical_priors(),
    }
    if normalized["gradient_accumulation_steps"] is not None:
        performance["gradient_accumulation_steps"] = normalized[
            "gradient_accumulation_steps"
        ]
    record: dict[str, Any] = {
        "comparison_job_id": str(job["job_id"]),
        "scenario": {
            "model_id": normalized["model_id"],
            "dataset_id": normalized["dataset_id"],
            "train_type": normalized["training_mode"],
            "gpu_count": normalized["gpu_count"],
            "physical_mbs": normalized["physical_mbs"],
            "target_gbs": normalized["target_gbs"],
            "cutoff_len": normalized["cutoff_len"],
        },
        "selector": {
            "training_mode": normalized["training_mode"],
            "zero_stage": normalized["zero_stage"],
            "gradient_checkpointing": normalized[
                "gradient_checkpointing"
            ],
            "packing": normalized["packing"],
            "kernel_path": normalized["kernel_path"],
            "dtype": normalized["dtype"],
        },
        "model_basis": normalized["model_geometry"],
        "performance": performance,
    }
    basis = _static_structured_basis(
        record,
        predictor.profiles,
        hardware_memory_bytes=normalized["hardware"].memory_bytes,
    )
    traffic = basis["traffic"]
    limit_seconds = basis["component_seconds_at_physical_limits"]
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
                "compute_at_dense_peak": limit_seconds["compute"],
                "kernel_hbm_at_physical_peak": limit_seconds[
                    "kernel_hbm"
                ],
                "optimizer_hbm_at_physical_peak": limit_seconds[
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
    return record, basis


def _joint_candidate(
    record: Mapping[str, Any],
    *,
    scenario_id: str,
) -> dict[str, Any]:
    effective_work = float(
        record["performance"]["work_per_step"]["effective_tokens"]
    )
    return {
        "comparison_job_id": str(record["comparison_job_id"]),
        "scenario_id": scenario_id,
        "record": record,
        "features": _record_features(record),
        "effective_log_work": math.log(effective_work),
        "log_analytic_anchor": _record_log_analytic_anchor(record),
    }


def _prediction_maps(
    *,
    ledger: Mapping[str, Any],
    predictor: ThroughputPredictor,
    theory: Mapping[str, Any],
    hybrid: Mapping[str, Any],
    pure: Mapping[str, Any],
    joint: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, Any]],
]:
    physical_model = theory["full_historical_bootstrap_fit"][
        "throughput"
    ]["center"]
    old_priors = theory["physical_priors"]["values"]
    hybrid_model = hybrid["throughput"]["frozen_model"]
    pure_model = pure["frozen_model"]
    joint_model = joint["h800"]["frozen_model"]
    two_head_model = joint["h800"]["frozen_two_head_challenger"]

    maps: dict[str, dict[str, float]] = {
        "v1_physical_roofline": {},
        "v2_physics_plus_pairwise_residual": {},
        "v3_configuration_only_pairwise": {},
        "v4_full_factor_single_head": {},
        "v4b_full_factor_two_head_rejected": {},
        "v5_structured_cross_card_single_head": {},
    }
    basis_by_job: dict[str, dict[str, Any]] = {}
    joint_candidates = []
    ledger_prediction_by_job = {}
    for input_index, ledger_row in enumerate(ledger["jobs"]):
        job = ledger_row["job"]
        job_id = str(job["job_id"])
        record, basis = _static_record(
            predictor,
            job,
            input_index=input_index,
        )
        basis_by_job[job_id] = basis
        physical_log = _physics_throughput_score(
            record,
            physical_model,
            old_priors,
        )
        old_candidate = {
            "record": record,
            "physics_log_throughput": physical_log,
        }
        maps["v1_physical_roofline"][job_id] = math.exp(physical_log)
        maps["v2_physics_plus_pairwise_residual"][job_id] = math.exp(
            _ranker_score(old_candidate, hybrid_model)
        )
        maps["v3_configuration_only_pairwise"][job_id] = math.exp(
            _ranker_score(old_candidate, pure_model)
        )

        scenario_id = str(
            ledger_row["prediction"]["comparison_group"]
        )
        current_joint = _joint_candidate(
            record,
            scenario_id=scenario_id,
        )
        joint_candidates.append(current_joint)
        maps["v4_full_factor_single_head"][job_id] = math.exp(
            _predict_joint_log_throughput(
                current_joint,
                joint_model,
            )
        )
        ledger_prediction_by_job[job_id] = float(
            ledger_row["prediction"][
                "predicted_effective_tokens_per_second"
            ]
        )

    for candidate, predicted_log in _predict_two_head_entries(
        joint_candidates,
        two_head_model,
    ):
        maps["v4b_full_factor_two_head_rejected"][
            str(candidate["comparison_job_id"])
        ] = math.exp(float(predicted_log))
    maps["v5_structured_cross_card_single_head"] = (
        ledger_prediction_by_job
    )
    return maps, basis_by_job


def _evaluation_row(
    source: Mapping[str, Any],
    *,
    predicted_rate: float,
    static_work: float,
) -> dict[str, Any]:
    observed = source["observed"]
    observed_rate = float(observed["effective_tokens_per_second"])
    observed_step = float(observed["step_seconds"])
    observed_work = float(observed["effective_tokens_per_step"])
    predicted_step = static_work / predicted_rate

    throughput_signed = 100.0 * (
        predicted_rate - observed_rate
    ) / observed_rate
    step_signed = 100.0 * (
        predicted_step - observed_step
    ) / observed_step
    work_signed = 100.0 * (
        static_work - observed_work
    ) / observed_work
    return {
        "job_id": str(source["job_id"]),
        "finished_unix": float(source["finished_unix"]),
        "finished_at_utc": str(source["finished_at_utc"]),
        "configuration": source["configuration"],
        "observed": observed,
        "predicted": {
            "effective_tokens_per_second": predicted_rate,
            "step_seconds": predicted_step,
            "effective_tokens_per_step": static_work,
        },
        "error": {
            "throughput_signed_percentage": throughput_signed,
            "throughput_absolute_percentage": abs(throughput_signed),
            "step_time_signed_percentage": step_signed,
            "step_time_absolute_percentage": abs(step_signed),
            "static_work_signed_percentage": work_signed,
            "static_work_absolute_percentage": abs(work_signed),
            "predicted_to_observed_throughput_ratio": (
                predicted_rate / observed_rate
            ),
        },
    }


def _add_hit_at_90(
    ranking: dict[str, Any],
) -> dict[str, Any]:
    evaluated = [
        row
        for row in ranking["scenarios"]
        if row["status"] == "evaluated"
    ]
    ranking["top1_at_least_90pct_of_best_fraction"] = (
        statistics.fmean(
            float(float(row["top1_regret_percent"]) <= 10.0)
            for row in evaluated
        )
        if evaluated
        else None
    )
    return ranking


def _metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "absolute": _absolute_metrics(rows),
        "ranking": _add_hit_at_90(_ranking_metrics(rows)),
    }


def _cohort_breakdown(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        config = row["configuration"]
        key = "/".join(
            (
                str(config["model_id"]),
                str(config["train_type"]),
                str(config["dataset_id"]),
            )
        )
        groups[key].append(row)
    return {
        key: _metrics(values)
        for key, values in sorted(groups.items())
    }


def _v4_prediction_inventory(
    *,
    ledger: Mapping[str, Any],
    validation: Mapping[str, Any],
    predictions: Mapping[str, float],
) -> dict[str, Any]:
    state_by_job: dict[str, str] = {}
    observed_by_job: dict[str, Mapping[str, Any]] = {}
    for row in validation["successful_evaluations"]:
        job_id = str(row["job_id"])
        state_by_job[job_id] = "success"
        observed_by_job[job_id] = row["observed"]
    for row in validation["excluded_terminal_outcomes"]:
        state_by_job[str(row["job_id"])] = str(
            row["classification"]
        )
    for row in validation["sealed_pending_predictions"]:
        state_by_job[str(row["job_id"])] = str(row["state"])

    job_rows = []
    by_configuration: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    for ledger_row in ledger["jobs"]:
        job = ledger_row["job"]
        job_id = str(job["job_id"])
        configuration = {
            "model_id": str(job["model_id"]),
            "train_type": str(job["train_type"]),
            "gpu_count": int(job["gpu_count"]),
            "target_gbs": int(job["target_gbs"]),
            "cutoff_len": int(job["cutoff_len"]),
            "mbs": int(job["mbs"]),
            "gradient_checkpointing": bool(job["gc"]),
            "zero_stage": str(job["zero"]),
            "packing": bool(job["packing"]),
            "kernel_path": str(
                ledger_row["prediction"]["configuration"][
                    "kernel_path"
                ]
            ),
        }
        predicted = float(predictions[job_id])
        observed = observed_by_job.get(job_id)
        observed_rate = (
            float(observed["effective_tokens_per_second"])
            if observed is not None
            else None
        )
        row = {
            "job_id": job_id,
            "repeat": int(job["repeat"]),
            "state": state_by_job.get(job_id, "unknown"),
            "configuration": configuration,
            "predicted_effective_tokens_per_second": predicted,
            "observed_effective_tokens_per_second": observed_rate,
            "absolute_percentage_error": (
                100.0
                * abs(predicted - observed_rate)
                / observed_rate
                if observed_rate is not None
                else None
            ),
        }
        job_rows.append(row)
        by_configuration[
            json.dumps(
                configuration,
                sort_keys=True,
                separators=(",", ":"),
            )
        ].append(row)

    configuration_rows = []
    for key, repeat_rows in sorted(by_configuration.items()):
        predicted_values = [
            float(row["predicted_effective_tokens_per_second"])
            for row in repeat_rows
        ]
        if max(predicted_values) - min(predicted_values) > 1.0e-8:
            raise ValueError(
                "V4 prediction changed across exact repeats"
            )
        observed_values = [
            float(row["observed_effective_tokens_per_second"])
            for row in repeat_rows
            if row["observed_effective_tokens_per_second"] is not None
        ]
        observed_mean = (
            statistics.fmean(observed_values)
            if observed_values
            else None
        )
        predicted = statistics.fmean(predicted_values)
        states = sorted({str(row["state"]) for row in repeat_rows})
        configuration_rows.append(
            {
                "configuration": json.loads(key),
                "repeat_jobs": len(repeat_rows),
                "job_ids": [
                    str(row["job_id"]) for row in repeat_rows
                ],
                "state_counts": {
                    state: sum(
                        str(row["state"]) == state
                        for row in repeat_rows
                    )
                    for state in states
                },
                "predicted_effective_tokens_per_second": predicted,
                "successful_repeats": len(observed_values),
                "observed_mean_effective_tokens_per_second": (
                    observed_mean
                ),
                "absolute_percentage_error_against_repeat_mean": (
                    100.0
                    * abs(predicted - observed_mean)
                    / observed_mean
                    if observed_mean is not None
                    else None
                ),
            }
        )
    return {
        "model_id": "v4_full_factor_single_head",
        "job_predictions": sorted(
            job_rows,
            key=lambda row: str(row["job_id"]),
        ),
        "unique_configuration_predictions": configuration_rows,
        "jobs": len(job_rows),
        "unique_configurations": len(configuration_rows),
        "repeat_policy": (
            "Exact repeats share one deterministic pre-run prediction; "
            "observed successful repeats are averaged in the compact table."
        ),
    }


def _model_catalog(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    paths: Mapping[str, Path],
) -> list[dict[str, Any]]:
    definitions = [
        (
            "v1_physical_roofline",
            "旧物理 roofline/效率曲线",
            "theory",
            "absolute_and_ranking",
            True,
        ),
        (
            "v2_physics_plus_pairwise_residual",
            "物理基线 + 28维配置残差排序",
            "hybrid",
            "absolute_and_ranking",
            True,
        ),
        (
            "v3_configuration_only_pairwise",
            "21维纯配置 pairwise 排序",
            "pure",
            "ranking_primary_absolute_is_diagnostic_only",
            True,
        ),
        (
            "v4_full_factor_single_head",
            "70维全因素直接 log(step-time) 单头",
            "joint",
            "absolute_and_ranking",
            True,
        ),
        (
            "v4b_full_factor_two_head_rejected",
            "70维绝对头 + pairwise 头（已淘汰候选）",
            "joint",
            "absolute_and_ranking_set_dependent",
            False,
        ),
        (
            "v5_structured_cross_card_single_head",
            "35维结构化物理组件 + 分层卡型适配器",
            "structured",
            "absolute_and_ranking",
            True,
        ),
    ]
    catalog = []
    for model_id, label, artifact_key, semantics, mainline in definitions:
        artifact = artifacts[artifact_key]
        path = paths[artifact_key]
        generated, source = _artifact_time(path, artifact)
        catalog.append(
            {
                "model_id": model_id,
                "label": label,
                "artifact_path": str(path.resolve()),
                "artifact_sha256": sha256_file(path),
                "artifact_frozen_at_utc": generated,
                "artifact_time_source": source,
                "prediction_semantics": semantics,
                "h800_mainline_generation": mainline,
                "replayed": True,
            }
        )
    return catalog


def build_report(
    *,
    ledger_path: Path,
    validation_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    ledger = read_json(ledger_path)
    validation = read_json(validation_path)
    successful = validation["successful_evaluations"]
    predictor = ThroughputPredictor()

    artifacts = {
        "theory": read_json(THEORY_ARTIFACT),
        "hybrid": read_json(HYBRID_ARTIFACT),
        "pure": read_json(PURE_ARTIFACT),
        "joint": read_json(JOINT_ARTIFACT),
        "structured": read_json(STRUCTURED_ARTIFACT),
        "rtx4090": read_json(RTX4090_ARTIFACT),
    }
    paths = {
        "theory": THEORY_ARTIFACT,
        "hybrid": HYBRID_ARTIFACT,
        "pure": PURE_ARTIFACT,
        "joint": JOINT_ARTIFACT,
        "structured": STRUCTURED_ARTIFACT,
        "rtx4090": RTX4090_ARTIFACT,
    }
    catalog = _model_catalog(artifacts=artifacts, paths=paths)
    predictions, bases = _prediction_maps(
        ledger=ledger,
        predictor=predictor,
        theory=artifacts["theory"],
        hybrid=artifacts["hybrid"],
        pure=artifacts["pure"],
        joint=artifacts["joint"],
    )
    successful_by_job = {
        str(row["job_id"]): row for row in successful
    }
    result_models = []
    detailed_predictions: dict[str, list[dict[str, Any]]] = {}
    for entry in catalog:
        model_id = str(entry["model_id"])
        model_rows = []
        for job_id, source in successful_by_job.items():
            static_work = float(
                bases[job_id]["work_per_step"]["effective_tokens"]
            )
            model_rows.append(
                _evaluation_row(
                    source,
                    predicted_rate=predictions[model_id][job_id],
                    static_work=static_work,
                )
            )
        frozen_unix = _iso_to_unix(
            str(entry["artifact_frozen_at_utc"])
        )
        strict_rows = [
            row
            for row in model_rows
            if float(row["finished_unix"]) > frozen_unix
        ]
        post_ledger_unix = _iso_to_unix(
            str(ledger["generated_at_utc"])
        )
        prospective_rows = [
            row
            for row in strict_rows
            if float(row["finished_unix"]) > post_ledger_unix
        ]
        result_models.append(
            {
                **entry,
                "prediction_registration_status": (
                    "predictions_sealed_before_outcomes"
                    if model_id
                    == "v5_structured_cross_card_single_head"
                    else (
                        "artifact_frozen_before_outcomes_but_predictions_"
                        "replayed_after_outcomes"
                    )
                ),
                "common_unseen_model_holdout": _metrics(model_rows),
                "strict_post_artifact": _metrics(strict_rows),
                "rows_finished_after_v5_prediction_ledger": _metrics(
                    prospective_rows
                ),
                "common_holdout_breakdown": _cohort_breakdown(
                    model_rows
                ),
            }
        )
        detailed_predictions[model_id] = model_rows

    rtx4090 = artifacts["rtx4090"]
    report = _seal(
        {
            "schema": SCHEMA,
            "implementation_version": IMPLEMENTATION_VERSION,
            "generated_at_utc": _utc_now(),
            "question_answered": (
                "How many throughput model generations exist, and how "
                "do their frozen parameters generalize to later H800 "
                "Qwen3-VL/32B experiments?"
            ),
            "version_count": {
                "h800_mainline_generations": 5,
                "h800_serialized_prediction_heads_replayed": 6,
                "reason_for_difference": (
                    "The joint full-factor artifact contains both the "
                    "selected single head and one rejected two-head "
                    "challenger."
                ),
                "rtx4090_separate_card_branch": 1,
            },
            "evaluation_contract": {
                "inputs_available_before_run_only": True,
                "static_dataset_profile_used": True,
                "observed_step_time_used_as_feature": False,
                "observed_throughput_used_as_feature": False,
                "runtime_counters_used_as_feature": False,
                "old_runtime_cohort_nuisance_policy": (
                    "Unknown pre-run cohort; no fitted cohort nuisance "
                    "term is injected."
                ),
                "ranking_candidate_unit": (
                    "Exact repeats are averaged before pairwise and "
                    "Top-1 metrics."
                ),
                "oom_policy": (
                    "OOM/failed/nonterminal rows are excluded; ranking "
                    "is conditional on a separate memory-safety filter."
                ),
                "common_holdout_warning": (
                    "All rows are unseen model IDs for every frozen "
                    "fit, but some Qwen3-VL outcomes predate V4/V5. Use "
                    "strict_post_artifact for temporal claims."
                ),
                "prediction_registration_warning": (
                    "Only V5 predictions were sealed in the common "
                    "ledger before the prospective outcomes. V1-V4/V4b "
                    "parameters were frozen, but their values in this "
                    "report are deterministic after-outcome replays."
                ),
            },
            "source_bindings": {
                "ledger": {
                    "path": str(ledger_path.resolve()),
                    "sha256": sha256_file(ledger_path),
                    "generated_at_utc": ledger["generated_at_utc"],
                },
                "validation": {
                    "path": str(validation_path.resolve()),
                    "sha256": sha256_file(validation_path),
                    "report_sha256": validation["report_sha256"],
                    "generated_at_utc": validation[
                        "generated_at_utc"
                    ],
                },
                "implementation": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
            },
            "campaign_snapshot": {
                "queue_jobs": validation["queue_snapshot"]["jobs"],
                "state_counts": validation["queue_snapshot"][
                    "state_counts"
                ],
                "successful_rows_evaluated": len(successful),
                "successful_model_ids": sorted(
                    {
                        str(row["configuration"]["model_id"])
                        for row in successful
                    }
                ),
            },
            "v4_full_factor_prediction_inventory": (
                _v4_prediction_inventory(
                    ledger=ledger,
                    validation=validation,
                    predictions=predictions[
                        "v4_full_factor_single_head"
                    ],
                )
            ),
            "h800_models": result_models,
            "rtx4090_branch": {
                "artifact_path": str(RTX4090_ARTIFACT.resolve()),
                "artifact_sha256": sha256_file(RTX4090_ARTIFACT),
                "generated_at_utc": rtx4090["generated_at_utc"],
                "serialized_hybrid_model_available": True,
                "configuration_only_ranker_was_trained": True,
                "configuration_only_full_data_coefficients_serialized": (
                    False
                ),
                "later_same_card_holdout_rows": 0,
                "evaluation_status": (
                    "not_scored_no_later_rtx4090_holdout"
                ),
                "why_not_scored_on_h800": (
                    "Card-specific parameters cannot be called a valid "
                    "same-card generalization test on H800 outcomes."
                ),
            },
            "detailed_success_predictions": detailed_predictions,
        }
    )
    write_json(output_path, report)
    return report


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _summary(report: Mapping[str, Any]) -> str:
    lines = [
        (
            "versions: "
            f"{report['version_count']['h800_mainline_generations']} "
            "H800 mainline / "
            f"{report['version_count']['h800_serialized_prediction_heads_replayed']} "
            "serialized heads replayed"
        ),
        (
            "campaign: "
            f"{report['campaign_snapshot']['successful_rows_evaluated']} "
            "successful rows"
        ),
    ]
    for row in report["h800_models"]:
        common = row["common_unseen_model_holdout"]
        absolute = common["absolute"]
        ranking = common["ranking"]
        lines.append(
            f"{row['model_id']}: "
            f"MAPE={_fmt(absolute.get('throughput_mape_percent'))}% "
            f"pairwise={_fmt(ranking.get('pooled_pairwise_accuracy_percent'))}% "
            f"top1_regret={_fmt(ranking.get('mean_top1_regret_percent'))}% "
            f"hit90={_fmt(ranking.get('top1_at_least_90pct_of_best_fraction'))}"
        )
    lines.append(f"report_sha256={report['report_sha256']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--validation",
        type=Path,
        default=DEFAULT_VALIDATION,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        ledger_path=args.ledger.resolve(),
        validation_path=args.validation.resolve(),
        output_path=args.output.resolve(),
    )
    print(_summary(report))


if __name__ == "__main__":
    main()
