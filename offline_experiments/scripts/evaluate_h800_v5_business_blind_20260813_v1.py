#!/usr/bin/env python3
"""Business blind test for the structured throughput model (V5).

The stage-1 dense hybrid campaign measured real business data
(业务短样本 / 业务长尾样本) on Qwen3-8B — the only stage-1 model inside the
V5 training domain.  This script builds V5-format static dataset profiles from
the stage-1 exact-token profiles, predicts each measured configuration with
the frozen production V5 (and optionally the FLOPs-fixed challenger), and
compares predictions against the measured throughput.

Protocol rules (frozen via --freeze-protocol before the queue completed):

* test rows: stage-1 formal queue, model qwen3_8b only, terminal rows
  (success with measured throughput; OOM rows excluded from throughput
  ranking but recorded for admission diagnostics);
* ranking groups: (source_dataset_id, gpu_count); the primary metric ranks
  only memory-admitted rows (production behaviour), pure-throughput ranking
  is reported as a diagnostic;
* thresholds: worst top-1 regret < 0.10, group pairwise accuracy >= 0.90;
  absolute MAPE is diagnostic-only because stage-1 runs the fa3+liger kernel
  path which is outside the V5 training kernel set;
* a pass is business-transfer evidence, NOT a production release — release
  still requires explicit user acceptance.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from common import (
    ARTIFACT_DIR,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from evaluate_h800_hybrid_attention_dense_stage1_v1 import FORMAL_RECORDS

PROTOCOL_SCHEMA = "sft_h800_v5_business_blind_protocol/v1"
REPORT_SCHEMA = "sft_h800_v5_business_blind_report/v1"
PROTOCOL_PATH = ARTIFACT_DIR / "h800_v5_business_blind_protocol_20260813_v1.json"
REPORT_PATH = ARTIFACT_DIR / "h800_v5_business_blind_report_20260813_v1.json"
PROFILE_DIR = ARTIFACT_DIR / "h800_v5_business_profiles_20260813"
STAGE1_PROFILE_DIR = (
    ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_profiles_v1"
)
PRODUCTION_V5 = ARTIFACT_DIR / "structured_throughput_modeling.json"
CHALLENGER_V5 = (
    ARTIFACT_DIR / "structured_throughput_modeling_flopsfix_20260813_v1.json"
)
FORMAL_MATRIX = (
    ARTIFACT_DIR.parent
    / "matrix"
    / "h800_hybrid_attention_dense_stage1_formal_v1.jsonl"
)
TEST_MODEL_ID = "qwen3_8b"

# Pre-registered thresholds (frozen before the queue completes).
THRESHOLDS: dict[str, Any] = {
    "worst_top1_regret_max": 0.10,
    "group_pairwise_accuracy_min": 0.90,
    "primary_ranking_scope": "memory_admitted_rows_only",
    "absolute_mape": "diagnostic_only_kernel_path_outside_v5_training",
}

SOURCES = {"业务短样本": "business_short", "业务长尾样本": "business_tail"}
# The production predictor only supports the four training categories; the
# honest closest mapping for the two business sources is fixed here.
CATEGORY_MAP = {"业务短样本": "short", "业务长尾样本": "longtail"}
CUTOFFS = (2048, 8192, 16384)

_ZERO_MAP = {"none": 0, "zero0": 0, "zero1": 1, "zero2": 2, "zero3": 3}


def protocol_payload() -> dict[str, Any]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "test_rows": {
            "source": str(FORMAL_RECORDS.resolve()),
            "filter": {
                "model_id": TEST_MODEL_ID,
                "terminal": "success rows with measured throughput",
                "oom_rows": "recorded for admission diagnostics, excluded from ranking",
            },
        },
        "ranking_groups": "(source_dataset_id, cutoff_len, gpu_count)",
        "repeat_folding": (
            "repeat runs of the same physical configuration fold to one "
            "request with the median measured throughput (the production "
            "group contract forbids duplicate candidates)"
        ),
        "target_gbs_source": (
            "frozen formal matrix h800_hybrid_attention_dense_stage1_formal_v1"
            ".jsonl, keyed by job_id"
        ),
        "models_under_test": [
            {
                "name": "production_v5",
                "path": str(PRODUCTION_V5.resolve()),
                "sha256": sha256_file(PRODUCTION_V5),
            },
            {
                "name": "flopsfix_challenger",
                "path": str(CHALLENGER_V5.resolve()),
                "sha256": sha256_file(CHALLENGER_V5),
            },
        ],
        "thresholds": THRESHOLDS,
        "dataset_category_mapping": CATEGORY_MAP,
        "category_mapping_note": (
            "the production predictor only accepts the four training "
            "categories; business sources are mapped to their closest "
            "training category (业务短样本->short, 业务长尾样本->longtail)"
        ),
        "challenger_runtime_note": (
            "the flopsfix challenger artifact is not yet promoted, so it runs "
            "through the base throughput predictor and inherits the "
            "production memory-admission flags for the primary metric"
        ),
        "kernel_path_caveat": (
            "stage-1 runs fa3+liger fused kernels; V5 training data covers "
            "fa2-class kernels only, so absolute MAPE is diagnostic-only"
        ),
        "release": {
            "mode": "business_transfer_evidence_only",
            "automatic_admission_allowed": False,
        },
    }


def build_business_profiles() -> dict[str, Path]:
    """Derive V5-format static profiles from the stage-1 exact-token profiles."""

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    built: dict[str, Path] = {}
    for source_id, slug in SOURCES.items():
        for cutoff in CUTOFFS:
            stage1_file = (
                STAGE1_PROFILE_DIR / f"qwen3_8b.source_{'a' if source_id == '业务短样本' else 'b'}.{cutoff}.jsonl"
            )
            if not stage1_file.is_file():
                continue
            rows = read_jsonl(stage1_file)
            dataset_id = f"{slug}_{cutoff}"
            output = PROFILE_DIR / f"{dataset_id}.qwen3_nothink.jsonl"
            write_jsonl(
                output,
                [
                    {
                        "sample_id": str(row.get("sample_id") or ""),
                        "total_tokens": int(row.get("total_tokens") or cutoff),
                        "label_tokens": int(row.get("label_tokens") or 0),
                    }
                    for row in rows
                ],
            )
            built[dataset_id] = output
    return built


def _predictor(artifact: Path) -> Any:
    from h800_unified_v3_throughput_v5_predictor import (
        H800UnifiedV3ThroughputV5Predictor,
    )

    return H800UnifiedV3ThroughputV5Predictor(
        throughput_artifact=artifact,
        additional_dataset_profile_dir=PROFILE_DIR,
    )


def _requests(
    observations: Sequence[dict[str, Any]],
    gbs_by_job: dict[str, int],
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for row in observations:
        if str(row.get("model_id")) != TEST_MODEL_ID:
            continue
        outcome = row["outcome"]
        if outcome.get("classification") not in {"success", "oom"}:
            continue
        configuration = row["configuration"]
        source_id = str(configuration["source_dataset_id"])
        if source_id not in SOURCES:
            raise ValueError(f"unknown business source {source_id!r}")
        cutoff = int(configuration["exact_total_tokens_per_sample"])
        job_id = str(row["job_id"])
        if job_id not in gbs_by_job:
            raise ValueError(f"no frozen matrix target_gbs for {job_id}")
        requests.append(
            {
                "request_id": job_id,
                "comparison_group": (
                    f"{source_id}__c{cutoff}__gpu{configuration['gpu_count']}"
                ),
                "hardware_id": "h800",
                "model_id": TEST_MODEL_ID,
                "training_mode": "lora",
                "lora_rank": 32,
                "dataset_id": f"{SOURCES[source_id]}_{cutoff}",
                "dataset_category": CATEGORY_MAP[source_id],
                "target_gbs": int(gbs_by_job[job_id]),
                "cutoff_len": cutoff,
                "gpu_count": int(configuration["gpu_count"]),
                "physical_mbs": int(configuration["micro_batch_size"]),
                "zero_stage": int(_ZERO_MAP[str(configuration["zero"])]),
                "gradient_checkpointing": bool(
                    configuration["gradient_checkpointing"]
                ),
                "packing": False,
                "offload": False,
                "dtype": "bf16",
                "kernel_path": str(configuration.get("kernel_path") or ""),
            }
        )
    return requests


def _metrics_for(
    observations: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    *,
    admitted_only: bool,
) -> dict[str, Any]:
    measured: dict[str, float] = {}
    for row in observations:
        if str(row.get("model_id")) != TEST_MODEL_ID:
            continue
        if row["outcome"].get("classification") != "success":
            continue
        metrics = row.get("metrics") or {}
        tps = metrics.get("effective_tokens_per_second")
        if tps is not None:
            measured[str(row["job_id"])] = float(tps)
    predicted: dict[str, float] = {}
    admitted: dict[str, bool] = {}
    group_of: dict[str, str] = {}
    for item in predictions:
        request_id = str(item.get("request_id"))
        throughput = item.get("throughput") or {}
        if throughput.get("prediction_available") is True:
            predicted[request_id] = float(
                throughput["predicted_effective_tokens_per_second"]
            )
        memory = item.get("memory") or {}
        admitted[request_id] = bool(memory.get("admitted") is True)
        group_of[request_id] = str(item.get("comparison_group") or "?")

    groups: dict[str, list[str]] = defaultdict(list)
    for request_id in measured:
        if admitted_only and admitted.get(request_id) is not True:
            continue
        groups[group_of.get(request_id, "?")].append(request_id)

    group_regrets: list[float] = []
    group_pairwise: list[float] = []
    aps: list[float] = []
    for group_id, members in sorted(groups.items()):
        members = [m for m in members if m in predicted]
        if len(members) < 2:
            continue
        oracle = max(members, key=lambda m: measured[m])
        selected = max(members, key=lambda m: predicted[m])
        regret = max(0.0, 1.0 - measured[selected] / measured[oracle])
        group_regrets.append(regret)
        pairs = 0
        correct = 0
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                pairs += 1
                if (predicted[left] - predicted[right]) * (
                    measured[left] - measured[right]
                ) >= 0:
                    correct += 1
        if pairs:
            group_pairwise.append(correct / pairs)
        for member in members:
            aps.append(
                abs(predicted[member] - measured[member]) / measured[member]
            )
    return {
        "rows_ranked": sum(len(group) for group in groups.values()),
        "groups": len(groups),
        "worst_top1_regret": max(group_regrets) if group_regrets else None,
        "mean_top1_regret": (
            statistics.fmean(group_regrets) if group_regrets else None
        ),
        "exact_top1_fraction": (
            sum(1 for r in group_regrets if r == 0.0) / len(group_regrets)
            if group_regrets
            else None
        ),
        "group_pairwise_accuracy": (
            statistics.fmean(group_pairwise) if group_pairwise else None
        ),
        "absolute_mape": statistics.fmean(aps) if aps else None,
        "regret_by_group": {
            group_id: regret
            for group_id, regret in zip(
                (g for g in sorted(groups)), group_regrets
            )
        },
    }


def _oom_admission_diagnostic(
    observations: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Production safety check: an OOM row must never be admitted."""

    oom_jobs = {
        str(row["job_id"])
        for row in observations
        if str(row.get("model_id")) == TEST_MODEL_ID
        and row["outcome"].get("classification") == "oom"
    }
    admitted_oom = []
    for item in predictions:
        request_id = str(item.get("request_id"))
        if request_id in oom_jobs:
            memory = item.get("memory") or {}
            if bool(memory.get("admitted") is True):
                admitted_oom.append(request_id)
    return {
        "oom_rows": len(oom_jobs),
        "oom_rows_admitted_by_memory_model": len(admitted_oom),
        "admitted_oom_job_ids": admitted_oom,
    }


def _folded_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold repeat runs of the same physical configuration to one request.

    The repeat-variance arm re-runs the core-orthogonal centre, so the same
    physical config appears twice.  The production group contract forbids
    duplicate candidates; measured throughput folds to the median.
    """

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        configuration = row["configuration"]
        groups[
            (
                configuration["source_dataset_id"],
                configuration["exact_total_tokens_per_sample"],
                configuration["gpu_count"],
                configuration["micro_batch_size"],
                configuration["zero"],
                configuration["gradient_checkpointing"],
            )
        ].append(row)
    folded: list[dict[str, Any]] = []
    for members in groups.values():
        successes = [
            row
            for row in members
            if row["outcome"]["classification"] == "success"
        ]
        if not successes:
            folded.append(members[0])
            continue
        measured = [
            float((row.get("metrics") or {}).get("effective_tokens_per_second"))
            for row in successes
            if (row.get("metrics") or {}).get("effective_tokens_per_second")
        ]
        representative = dict(successes[0])
        representative["metrics"] = {
            **(successes[0].get("metrics") or {}),
            "effective_tokens_per_second": statistics.median(measured),
            "folded_repeats": len(measured),
        }
        representative["member_job_ids"] = sorted(
            str(row["job_id"]) for row in members
        )
        folded.append(representative)
    return folded


def evaluate(*, allow_incomplete: bool = False) -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError("protocol not frozen; run --freeze-protocol first")
    protocol = read_json(PROTOCOL_PATH)
    observations = read_jsonl(FORMAL_RECORDS)
    records_sha256 = sha256_file(FORMAL_RECORDS)

    eight_b = [
        row
        for row in observations
        if str(row.get("model_id")) == TEST_MODEL_ID
    ]
    terminal = [
        row
        for row in eight_b
        if row["outcome"].get("terminal_eligible")
        and row["outcome"]["classification"] in {"success", "oom"}
    ]
    complete = len(terminal) == len(eight_b)

    build_business_profiles()
    gbs_by_job = {
        str(row["job_id"]): int(row.get("target_gbs") or 0)
        for row in read_jsonl(FORMAL_MATRIX)
    }
    folded = _folded_rows(terminal)
    requests = _requests(folded, gbs_by_job)
    group_by_request = {
        str(request["request_id"]): str(request["comparison_group"])
        for request in requests
    }
    results: dict[str, Any] = {}
    production_admitted: dict[str, bool] = {}
    for entry in protocol["models_under_test"]:
        if sha256_file(Path(str(entry["path"]))) != entry["sha256"]:
            raise ValueError(
                f"model under test drifted from protocol: {entry['name']}"
            )
        if entry["name"] == "production_v5":
            predictions = _predictor(Path(str(entry["path"]))).predict(requests)[
                "predictions"
            ]
            production_admitted = {
                str(item.get("request_id")): bool(
                    (item.get("memory") or {}).get("admitted") is True
                )
                for item in predictions
            }
        else:
            from throughput_predictor import ThroughputPredictor

            challenger_report = read_json(Path(str(entry["path"])))
            challenger_static = Path(
                str(
                    challenger_report["source_bindings"][
                        "static_workload_profiles"
                    ]["path"]
                )
            )
            base = ThroughputPredictor(
                model_artifact=Path(str(entry["path"])),
                static_profile_artifact=challenger_static,
                additional_dataset_profile_dir=PROFILE_DIR,
            )
            raw = base.predict_many(requests)["predictions"]
            predictions = [
                {
                    "request_id": str(item.get("request_id")),
                    "comparison_group": group_by_request.get(
                        str(item.get("request_id")), "?"
                    ),
                    "throughput": {
                        "prediction_available": True,
                        "predicted_effective_tokens_per_second": float(
                            item.get("predicted_effective_tokens_per_second")
                        ),
                    },
                    "memory": {
                        "admitted": production_admitted.get(
                            str(item.get("request_id")), False
                        )
                    },
                }
                for item in raw
            ]
        results[entry["name"]] = {
            "admitted_only": _metrics_for(
                folded, predictions, admitted_only=True
            ),
            "pure_throughput": _metrics_for(
                folded, predictions, admitted_only=False
            ),
            "oom_admission_diagnostic": _oom_admission_diagnostic(
                observations, predictions
            ),
        }

    gates: dict[str, bool] = {}
    if complete:
        for name, entry in results.items():
            admitted_only = entry["admitted_only"]
            gates[name] = {
                "worst_top1_regret_within_10pct": bool(
                    admitted_only["worst_top1_regret"] is not None
                    and admitted_only["worst_top1_regret"]
                    <= THRESHOLDS["worst_top1_regret_max"]
                ),
                "group_pairwise_at_least_90pct": bool(
                    admitted_only["group_pairwise_accuracy"] is not None
                    and admitted_only["group_pairwise_accuracy"]
                    >= THRESHOLDS["group_pairwise_accuracy_min"]
                ),
            }
    all_passed = bool(gates) and all(
        check for entry in gates.values() for check in entry.values()
    )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(PROTOCOL_PATH.resolve()),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "observations": {
            "path": str(FORMAL_RECORDS.resolve()),
            "sha256": records_sha256,
            "eight_b_rows": len(eight_b),
            "terminal_rows": len(terminal),
        },
        "data_complete": complete,
        "results": results,
        "gates": gates,
        "all_gates_passed": all_passed,
        "status": (
            "business_transfer_evidence_pass"
            if complete and all_passed
            else "failed_gates"
            if complete
            else "data_incomplete"
        ),
        "release": protocol["release"],
        "kernel_path_caveat": protocol["kernel_path_caveat"],
    }
    report["report_sha256"] = sha256_json(report)
    write_json(REPORT_PATH, report)
    if complete and not all_passed and not allow_incomplete:
        raise RuntimeError("business blind failed one or more gates")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.freeze_protocol:
        write_json(PROTOCOL_PATH, protocol_payload())
        print(json.dumps(read_json(PROTOCOL_PATH), ensure_ascii=False, indent=2))
        return
    report = evaluate(allow_incomplete=args.allow_incomplete)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
