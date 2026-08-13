#!/usr/bin/env python3
"""Evaluate completed Packing-ranking rows with frozen virtual-MBS V5 logic.

No coefficient is fitted.  The score is the frozen Unpacked model prediction
log-interpolated at ``samples_per_pack.mean`` on the 1/2/4/8/16/32 MBS grid.
The report keeps complete-scenario metrics separate from incomplete pair-only
diagnostics and never treats observed OOM filtering as a prospective memory
prediction.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
import statistics
from typing import Any

from analyze_h800_packing_virtual_mbs_v1 import FrozenUnpackedPredictor
from common import ARTIFACT_DIR, read_json, read_jsonl, sha256_file, sha256_json, write_json
from evaluate_h800_packing_config_ranking_v1 import collect_job
from prepare_h800_packing_config_ranking_v1 import CAMPAIGN_ID, QUEUE_FIT
from prepare_h800_packing_final_business_profiles_v1 import Encoder, _iter_objects
from structured_throughput_modeling import StaticDatasetProfiles


OUTPUT = ARTIFACT_DIR / "h800_packing_config_ranking_partial_virtual_mbs_v1.json"
MATERIAL_GAP = 0.03


class BusinessVirtualMBSPredictor(FrozenUnpackedPredictor):
    """Supply exact in-memory business token profiles to the frozen predictor."""

    def __init__(self) -> None:
        super().__init__()
        self._business_cache: dict[Path, tuple[StaticDatasetProfiles, str]] = {}
        self._encoder: Encoder | None = None

    def _profiles(self, profile_path: Path) -> tuple[StaticDatasetProfiles, str]:
        resolved = profile_path.resolve()
        cached = self._business_cache.get(resolved)
        if cached is not None:
            return cached
        aggregate = read_json(resolved)
        source_path = Path(str(aggregate["source_binding"]["local_path"]))
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if self._encoder is None:
            self._encoder = Encoder()
        token_rows = []
        for value in _iter_objects(source_path):
            if not isinstance(value, dict):
                raise ValueError(f"business row is not an object: {source_path}")
            row = {
                key: str(value.get(key) or "")
                for key in ("system", "prompt", "response")
            }
            total_tokens, label_tokens = self._encoder.encode(row)
            token_rows.append(
                {
                    "total_tokens": total_tokens,
                    "label_tokens": label_tokens,
                    "turns": 3 if row["system"] else 2,
                }
            )
        if len(token_rows) != int(aggregate["records"]):
            raise ValueError(f"business profile row count drifted: {resolved}")
        dataset_id = resolved.stem
        profiles = StaticDatasetProfiles.__new__(StaticDatasetProfiles)
        profiles.profile_dir = resolved.parent
        profiles.rows = {dataset_id: token_rows}
        profiles._cache = {}
        result = (profiles, dataset_id)
        self._business_cache[resolved] = result
        return result


def _pair(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": str(job["model_id"]),
        "train_type": str(job["train_type"]),
        "gpu_count": int(job["gpu_count"]),
        "zero_stage": int(job["zero_stage"]),
        "gc": bool(job["gc"]),
        "dataset_profile_path": str(job["dataset_profile_path"]),
        "cutoff_len": int(job["cutoff_len"]),
        "n_pack_mean": float(job["virtual_mbs"]),
        "packed_ga": int(job["gradient_accumulation_steps"]),
    }


def _score(job: Mapping[str, Any], predictor: BusinessVirtualMBSPredictor) -> dict[str, Any]:
    prediction = predictor.virtual_prediction(_pair(job))
    effective = float(prediction["predicted_effective_tokens_per_second"])
    work = prediction["work_per_step"]
    logical = effective * float(work["logical_samples"]) / float(work["effective_tokens"])
    return {
        "predicted_logical_samples_per_second": logical,
        "predicted_effective_tokens_per_second": effective,
        "virtual_mbs": float(job["virtual_mbs"]),
        "lower_mbs": int(prediction["lower_mbs"]),
        "upper_mbs": int(prediction["upper_mbs"]),
        "upper_log_weight": float(prediction["upper_log_weight"]),
        "grid_extrapolation": bool(prediction["is_grid_extrapolation"]),
        "outside_h800_feature_support": list(
            prediction["outside_h800_feature_support"]
        ),
        "cutoff_seen_exactly_in_h800_training": bool(
            prediction["cutoff_seen_exactly_in_h800_training"]
        ),
        "template_gc_exact": bool(prediction["template_gc_exact"]),
    }


def _pair_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = total = material_correct = material_total = 0
    for left, right in itertools.combinations(rows, 2):
        left_truth = float(left["observed_logical_samples_per_second"])
        right_truth = float(right["observed_logical_samples_per_second"])
        if math.isclose(left_truth, right_truth, rel_tol=0.0, abs_tol=1.0e-12):
            continue
        left_score = float(left["prediction"]["predicted_logical_samples_per_second"])
        right_score = float(right["prediction"]["predicted_logical_samples_per_second"])
        is_correct = (left_truth > right_truth) == (left_score > right_score)
        correct += int(is_correct)
        total += 1
        gap = abs(left_truth - right_truth) / max(left_truth, right_truth)
        if gap >= MATERIAL_GAP:
            material_correct += int(is_correct)
            material_total += 1
    return {
        "pairwise_correct": correct,
        "pairwise_total": total,
        "pairwise_accuracy": correct / total if total else None,
        "material_gap": MATERIAL_GAP,
        "material_pairwise_correct": material_correct,
        "material_pairwise_total": material_total,
        "material_pairwise_accuracy": (
            material_correct / material_total if material_total else None
        ),
    }


def _complete_group(group_id: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["classification"] == "success"]
    metrics = _pair_metrics(successful)
    oracle = max(
        successful,
        key=lambda row: float(row["observed_logical_samples_per_second"]),
    )
    selected = max(
        successful,
        key=lambda row: float(
            row["prediction"]["predicted_logical_samples_per_second"]
        ),
    )
    oracle_tps = float(oracle["observed_logical_samples_per_second"])
    selected_tps = float(selected["observed_logical_samples_per_second"])
    regret = max(0.0, 1.0 - selected_tps / oracle_tps)
    unfiltered = max(
        rows,
        key=lambda row: float(
            row["prediction"]["predicted_logical_samples_per_second"]
        ),
    )
    return {
        "group_id": group_id,
        "workload_id": str(rows[0]["workload_id"]),
        "gpu_count": int(rows[0]["gpu_count"]),
        "candidate_count": len(rows),
        "successful_candidates": len(successful),
        "oom_candidates": sum(row["classification"] == "oom" for row in rows),
        **metrics,
        "success_conditioned_predicted_top1_job_id": str(selected["job_id"]),
        "observed_top1_job_id": str(oracle["job_id"]),
        "exact_top1": selected["job_id"] == oracle["job_id"],
        "hit90": selected_tps >= 0.90 * oracle_tps,
        "top1_regret": regret,
        "unfiltered_predicted_top1_job_id": str(unfiltered["job_id"]),
        "unfiltered_predicted_top1_classification": str(
            unfiltered["classification"]
        ),
        "rows": list(rows),
    }


def analyze() -> dict[str, Any]:
    jobs = [
        row
        for row in read_jsonl(QUEUE_FIT)
        if row.get("ranking_eligible") is True
    ]
    observations = {str(job["job_id"]): collect_job(job) for job in jobs}
    snapshot = []
    for job in jobs:
        observation = observations[str(job["job_id"])]
        outcome = observation["outcome"]
        snapshot.append(
            {
                "job": job,
                "classification": str(outcome.get("classification") or "missing"),
                "usable": outcome.get("usable_for_modeling") is True,
                "observed": (
                    (outcome.get("throughput") or {}).get(
                        "global_logical_samples_per_second"
                    )
                ),
                "execution_attempt_id": outcome.get("execution_attempt_id"),
            }
        )
    terminal = [row for row in snapshot if row["classification"] in {"success", "oom"}]
    predictor = BusinessVirtualMBSPredictor()
    predicted_rows = []
    for state in terminal:
        job = state["job"]
        if state["classification"] == "success" and (
            not state["usable"] or state["observed"] is None
        ):
            continue
        predicted_rows.append(
            {
                "job_id": str(job["job_id"]),
                "ranking_group_id": str(job["ranking_group_id"]),
                "fixed_cutoff_mechanism_group_id": str(
                    job["fixed_cutoff_mechanism_group_id"]
                ),
                "workload_id": str(job["workload_id"]),
                "gpu_count": int(job["gpu_count"]),
                "cutoff_len": int(job["cutoff_len"]),
                "zero_stage": int(job["zero_stage"]),
                "gc": bool(job["gc"]),
                "classification": state["classification"],
                "execution_attempt_id": state["execution_attempt_id"],
                "observed_logical_samples_per_second": state["observed"],
                "prediction": _score(job, predictor),
            }
        )

    planned_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    terminal_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for job in jobs:
        planned_by_group[str(job["ranking_group_id"])].append(job)
    for row in predicted_rows:
        terminal_by_group[str(row["ranking_group_id"])].append(row)

    complete_groups = []
    incomplete_pair_groups = []
    for group_id, planned in sorted(planned_by_group.items()):
        current = terminal_by_group.get(group_id, [])
        if len(current) == len(planned) and len(current) >= 2:
            complete_groups.append(_complete_group(group_id, current))
        else:
            successful = [row for row in current if row["classification"] == "success"]
            if len(successful) >= 2:
                incomplete_pair_groups.append(
                    {
                        "group_id": group_id,
                        "planned_candidates": len(planned),
                        "terminal_candidates": len(current),
                        "successful_candidates": len(successful),
                        **_pair_metrics(successful),
                    }
                )

    pair_correct = sum(int(row["pairwise_correct"]) for row in complete_groups)
    pair_total = sum(int(row["pairwise_total"]) for row in complete_groups)
    material_correct = sum(
        int(row["material_pairwise_correct"]) for row in complete_groups
    )
    material_total = sum(
        int(row["material_pairwise_total"]) for row in complete_groups
    )
    exact = sum(bool(row["exact_top1"]) for row in complete_groups)
    hit90 = sum(bool(row["hit90"]) for row in complete_groups)
    regrets = [float(row["top1_regret"]) for row in complete_groups]
    unfiltered_oom = sum(
        row["unfiltered_predicted_top1_classification"] == "oom"
        for row in complete_groups
    )
    partial_correct = sum(
        int(row["pairwise_correct"]) for row in incomplete_pair_groups
    )
    partial_total = sum(int(row["pairwise_total"]) for row in incomplete_pair_groups)

    report: dict[str, Any] = {
        "schema": "sft_h800_packing_config_ranking_partial_virtual_mbs/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "analysis_only": True,
        "gpu_work_launched": False,
        "coefficients_fitted": False,
        "prediction_contract": {
            "base_model": "frozen Unpacked structured throughput V5",
            "packing_specific_coefficient": 0.0,
            "virtual_mbs": "upload-time samples_per_pack.mean",
            "fractional_rule": "log interpolation on MBS 1/2/4/8/16/32",
            "score": "predicted logical samples per second",
        },
        "snapshot": {
            "planned_primary_jobs": len(jobs),
            "terminal_primary_jobs": len(terminal),
            "predicted_terminal_rows": len(predicted_rows),
            "complete_ranking_groups": len(complete_groups),
            "incomplete_pair_only_groups": len(incomplete_pair_groups),
        },
        "complete_group_metrics": {
            "scope": "only groups whose every primary candidate is success or OOM",
            "scenario_count": len(complete_groups),
            "pairwise_correct": pair_correct,
            "pairwise_total": pair_total,
            "pairwise_accuracy": pair_correct / pair_total if pair_total else None,
            "material_pairwise_correct": material_correct,
            "material_pairwise_total": material_total,
            "material_pairwise_accuracy": (
                material_correct / material_total if material_total else None
            ),
            "exact_top1_hits": exact,
            "exact_top1_accuracy": exact / len(complete_groups) if complete_groups else None,
            "hit90_hits": hit90,
            "hit90_accuracy": hit90 / len(complete_groups) if complete_groups else None,
            "mean_top1_regret": statistics.fmean(regrets) if regrets else None,
            "worst_top1_regret": max(regrets) if regrets else None,
            "unfiltered_predicted_top1_is_oom_count": unfiltered_oom,
            "top1_metrics_are_success_conditioned": True,
            "top1_metrics_are_not_a_prospective_memory_gate_test": True,
        },
        "incomplete_group_pair_diagnostic": {
            "group_count": len(incomplete_pair_groups),
            "pairwise_correct": partial_correct,
            "pairwise_total": partial_total,
            "pairwise_accuracy": (
                partial_correct / partial_total if partial_total else None
            ),
            "top1_metrics_intentionally_not_computed": True,
        },
        "complete_groups": complete_groups,
        "incomplete_pair_groups": incomplete_pair_groups,
        "source_bindings": {
            "queue": {"path": str(QUEUE_FIT.resolve()), "sha256": sha256_file(QUEUE_FIT)},
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "frozen_virtual_mbs_predictor": {
                "path": str(
                    (Path(__file__).parent / "analyze_h800_packing_virtual_mbs_v1.py").resolve()
                ),
                "sha256": sha256_file(
                    Path(__file__).parent / "analyze_h800_packing_virtual_mbs_v1.py"
                ),
            },
        },
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = analyze()
    print(
        json.dumps(
            {
                "output": str(OUTPUT.resolve()),
                "snapshot": report["snapshot"],
                "complete_group_metrics": report["complete_group_metrics"],
                "incomplete_group_pair_diagnostic": report[
                    "incomplete_group_pair_diagnostic"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
