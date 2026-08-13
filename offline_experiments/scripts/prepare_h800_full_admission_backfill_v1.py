#!/usr/bin/env python3
"""Materialize the FULL admission backfill campaign (business-source first).

Nine unpacked FULL mechanisms currently fail closed because each has only one
independent fit source in the V5 evidence.  Recounting the historical data by
upstream corpus raises it to three, but three is the ceiling: every historical
FULL run used the same three public corpora.  This campaign adds two *business*
sources per mechanism and probes two points each, bracketing the safe/OOM
boundary the historical rows already located.

The script only writes a design, a JSONL queue, and a queue manifest.  It never
authorizes or launches GPU training, and it refuses to overwrite existing
campaign artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

from common import (
    ARTIFACT_DIR,
    ROOT,
    read_json,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)


CAMPAIGN_ID = "h800_full_admission_backfill_20260806_v1"
PHASE_ID = "h800_full_admission_backfill_v1"
DESIGN_SCHEMA = "sft_h800_full_admission_backfill_experiment_design/v1"
JOB_SCHEMA = "sft_h800_full_admission_backfill_job/v1"
QUEUE_MANIFEST_SCHEMA = "sft_h800_full_admission_backfill_queue_manifest/v1"
TARGET_GBS = 64

DEFAULT_DESIGN = ARTIFACT_DIR / "h800_full_admission_backfill_design_v1.json"
DEFAULT_QUEUE = ROOT / "matrix" / "h800_full_admission_backfill_jobs_v1.jsonl"
DEFAULT_MANIFEST = (
    ARTIFACT_DIR / "h800_full_admission_backfill_queue_manifest_v1.json"
)

# Business sources, none of them consumed by any FULL experiment so far.  Every
# one already has a frozen tokenizer profile, so no re-tokenization is needed.
DATA_ROOT = ROOT / "data"
PROFILE_ROOT = ARTIFACT_DIR
SOURCES: dict[str, dict[str, Any]] = {
    "lora_s2_src04_live_punishment": {
        "band": "short",
        "profile_max": 765,
        "campaign_dir": "h800_lora_safety_stage2_v1",
    },
    "lora_src04_account_risk_short": {
        "band": "short",
        "profile_max": 668,
        "campaign_dir": "h800_lora_source_disjoint_v1",
    },
    "lora_s2_src12_flood_multilabel": {
        "band": "medium",
        "profile_max": 3668,
        "campaign_dir": "h800_lora_safety_stage2_v1",
    },
    "lora_src07_flood_event_classify": {
        "band": "medium",
        "profile_max": 3729,
        "campaign_dir": "h800_lora_source_disjoint_v1",
    },
    "lora_s2_src18_marketing_antifraud": {
        "band": "long",
        "profile_max": 9804,
        "campaign_dir": "h800_lora_safety_stage2_v1",
    },
    "lora_src12_marketing_antifraud_long": {
        "band": "long",
        "profile_max": 13223,
        "campaign_dir": "h800_lora_source_disjoint_v1",
    },
}

# The nine unpacked FULL mechanisms that currently fail closed.  `probe_low` is
# a clearly-safe point, `probe_high` sits at the historical safe/OOM transition.
# Both are read off the migrated historical records, so this campaign brackets a
# known boundary instead of rescanning a grid.
MECHANISMS: tuple[dict[str, Any], ...] = (
    {
        "mechanism_id": "full_zero0_gc0_1gpu_pack0",
        "mechanism_cn": "1 卡、不切分、关闭梯度检查点",
        "gpu_count": 1,
        "zero_stage": 0,
        "gradient_checkpointing": False,
        "band": "short",
        "negative_boundary_short": False,
        "probe_low": {"model_id": "qwen3_1p7b", "mbs": 1, "cutoff_len": 512},
        "probe_high": {"model_id": "qwen3_4b", "mbs": 2, "cutoff_len": 4096},
    },
    {
        "mechanism_id": "full_zero0_gc1_1gpu_pack0",
        "mechanism_cn": "1 卡、不切分、开启梯度检查点",
        "gpu_count": 1,
        "zero_stage": 0,
        "gradient_checkpointing": True,
        "band": "medium",
        "negative_boundary_short": True,
        "probe_low": {"model_id": "qwen3_1p7b", "mbs": 2, "cutoff_len": 2048},
        "probe_high": {"model_id": "qwen3_8b", "mbs": 1, "cutoff_len": 4096},
    },
    {
        "mechanism_id": "full_zero2_gc0_2gpu_pack0",
        "mechanism_cn": "2 卡、ZeRO-2、关闭梯度检查点",
        "gpu_count": 2,
        "zero_stage": 2,
        "gradient_checkpointing": False,
        "band": "short",
        "negative_boundary_short": False,
        "probe_low": {"model_id": "qwen3_1p7b", "mbs": 1, "cutoff_len": 512},
        "probe_high": {"model_id": "qwen3_8b", "mbs": 4, "cutoff_len": 4096},
    },
    {
        "mechanism_id": "full_zero2_gc1_2gpu_pack0",
        "mechanism_cn": "2 卡、ZeRO-2、开启梯度检查点",
        "gpu_count": 2,
        "zero_stage": 2,
        "gradient_checkpointing": True,
        "band": "medium",
        "negative_boundary_short": False,
        "probe_low": {"model_id": "qwen3_1p7b", "mbs": 1, "cutoff_len": 512},
        "probe_high": {"model_id": "qwen3_14b", "mbs": 1, "cutoff_len": 2048},
    },
    {
        "mechanism_id": "full_zero3_gc0_2gpu_pack0",
        "mechanism_cn": "2 卡、ZeRO-3、关闭梯度检查点",
        "gpu_count": 2,
        "zero_stage": 3,
        "gradient_checkpointing": False,
        "band": "medium",
        "negative_boundary_short": False,
        "probe_low": {"model_id": "qwen3_8b", "mbs": 2, "cutoff_len": 2048},
        "probe_high": {"model_id": "qwen3_8b", "mbs": 4, "cutoff_len": 4096},
    },
    {
        "mechanism_id": "full_zero2_gc0_4gpu_pack0",
        "mechanism_cn": "4 卡、ZeRO-2、关闭梯度检查点",
        "gpu_count": 4,
        "zero_stage": 2,
        "gradient_checkpointing": False,
        "band": "medium",
        "negative_boundary_short": False,
        "probe_low": {"model_id": "qwen3_8b", "mbs": 1, "cutoff_len": 2048},
        "probe_high": {"model_id": "qwen3_14b", "mbs": 2, "cutoff_len": 4096},
    },
    {
        "mechanism_id": "full_zero2_gc1_4gpu_pack0",
        "mechanism_cn": "4 卡、ZeRO-2、开启梯度检查点",
        "gpu_count": 4,
        "zero_stage": 2,
        "gradient_checkpointing": True,
        "band": "long",
        "negative_boundary_short": True,
        "probe_low": {"model_id": "qwen3_8b", "mbs": 1, "cutoff_len": 4096},
        "probe_high": {"model_id": "qwen3_14b", "mbs": 4, "cutoff_len": 8192},
    },
    {
        "mechanism_id": "full_zero3_gc0_4gpu_pack0",
        "mechanism_cn": "4 卡、ZeRO-3、关闭梯度检查点",
        "gpu_count": 4,
        "zero_stage": 3,
        "gradient_checkpointing": False,
        "band": "medium",
        "negative_boundary_short": False,
        "probe_low": {"model_id": "qwen3_8b", "mbs": 2, "cutoff_len": 2048},
        "probe_high": {"model_id": "qwen3_8b", "mbs": 8, "cutoff_len": 4096},
    },
    {
        "mechanism_id": "full_zero3_gc1_4gpu_pack0",
        "mechanism_cn": "4 卡、ZeRO-3、开启梯度检查点",
        "gpu_count": 4,
        "zero_stage": 3,
        "gradient_checkpointing": True,
        "band": "long",
        "negative_boundary_short": True,
        "probe_low": {"model_id": "qwen3_8b", "mbs": 2, "cutoff_len": 4096},
        "probe_high": {"model_id": "qwen3_14b", "mbs": 8, "cutoff_len": 8192},
    },
)

# Already-supported mechanism: no fit backfill needed, but the audit asks for
# post-freeze prospective evidence on genuinely new sources.
SUPPORTED_MECHANISM: dict[str, Any] = {
    "mechanism_id": "full_zero3_gc1_2gpu_pack0",
    "mechanism_cn": "2 卡、ZeRO-3、开启梯度检查点（已支持，前瞻验收）",
    "gpu_count": 2,
    "zero_stage": 3,
    "gradient_checkpointing": True,
    "band": "medium",
    "probe_low": {"model_id": "qwen3_8b", "mbs": 2, "cutoff_len": 2048},
    "probe_high": {"model_id": "qwen3_14b", "mbs": 4, "cutoff_len": 4096},
}

BAND_SOURCES: dict[str, tuple[str, ...]] = {
    "short": (
        "lora_s2_src04_live_punishment",
        "lora_src04_account_risk_short",
    ),
    "medium": (
        "lora_s2_src12_flood_multilabel",
        "lora_src07_flood_event_classify",
    ),
    "long": (
        "lora_s2_src18_marketing_antifraud",
        "lora_src12_marketing_antifraud_long",
    ),
}


def _binding(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _models() -> dict[str, Mapping[str, Any]]:
    inventory = read_json(ARTIFACT_DIR / "model_inventory.json")
    return {str(row["id"]): row for row in inventory["models"]}


def _data_path(source_id: str) -> Path:
    return DATA_ROOT / SOURCES[source_id]["campaign_dir"] / f"{source_id}.jsonl"


def _profile_path(source_id: str) -> Path:
    return (
        PROFILE_ROOT
        / SOURCES[source_id]["campaign_dir"]
        / "profiles"
        / f"{source_id}.qwen3_nothink.jsonl"
    )


def _align8(value: int) -> int:
    return 8 * ((int(value) + 7) // 8)


def _effective_sequence(source_id: str, cutoff_len: int) -> int:
    """round_up(min(cutoff_len, profile_max), 8) - the V5 M1 length policy."""
    return _align8(min(int(cutoff_len), int(SOURCES[source_id]["profile_max"])))


def _job(
    mechanism: Mapping[str, Any],
    source_id: str,
    probe: Mapping[str, Any],
    probe_role: str,
    models: Mapping[str, Mapping[str, Any]],
    *,
    experiment_group: str,
) -> dict[str, Any]:
    gpu_count = int(mechanism["gpu_count"])
    mbs = int(probe["mbs"])
    cutoff_len = int(probe["cutoff_len"])
    model_id = str(probe["model_id"])
    if TARGET_GBS % (gpu_count * mbs):
        raise ValueError(
            f"target GBS {TARGET_GBS} is not divisible by {gpu_count} GPUs x MBS {mbs}"
        )
    model = models[model_id]
    data_path = _data_path(source_id)
    profile_path = _profile_path(source_id)
    identity = {
        "campaign_id": CAMPAIGN_ID,
        "mechanism_id": mechanism["mechanism_id"],
        "source_id": source_id,
        "probe_role": probe_role,
        "model_id": model_id,
        "mbs": mbs,
        "cutoff_len": cutoff_len,
    }
    effective = _effective_sequence(source_id, cutoff_len)
    return {
        "schema": JOB_SCHEMA,
        "job_id": stable_id("h800fullbf", identity),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "experiment_group": experiment_group,
        "mechanism_id": mechanism["mechanism_id"],
        "mechanism_cn": mechanism["mechanism_cn"],
        "probe_role": probe_role,
        "scenario_id": (
            f"{source_id}__{mechanism['mechanism_id']}__{model_id}"
            f"_mbs{mbs}_cutoff{cutoff_len}"
        ),
        "split_unit_id": source_id,
        "calibration_purpose": f"full_admission_backfill_{probe_role}",
        "calibration_partition": {
            "role": "calibration",
            "split_unit_id": source_id,
            "policy": "business_source_disjoint_full_admission_backfill_v1",
        },
        "hardware_id": "local_h800_140g",
        "gpu_type": "NVIDIA H800 140GB HBM3",
        "required_runtime_gpu_name": "NVIDIA H800",
        "model_id": model_id,
        "model_path": model["path"],
        "tokenizer_path": model["tokenizer_path"],
        "model_family": model["family"],
        "model_parameters": model["actual_parameters"],
        "template": model["template"],
        "train_type": "full",
        "dataset_id": source_id,
        "dataset_category": SOURCES[source_id]["band"],
        "dataset_profile_path": str(profile_path.resolve()),
        "dataset_profile_sha256": sha256_file(profile_path),
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "cutoff_len": cutoff_len,
        "raw_profile_max": int(SOURCES[source_id]["profile_max"]),
        "aligned_effective_sequence": effective,
        "target_gbs": TARGET_GBS,
        "gpu_count": gpu_count,
        "zero_stage": int(mechanism["zero_stage"]),
        # run_job.validate_job rejects single-card jobs that carry a DeepSpeed
        # config: one GPU means no sharding, and the launcher spells that
        # "none" rather than "zero0".
        "zero": (
            "none"
            if gpu_count == 1
            else f"zero{int(mechanism['zero_stage'])}"
        ),
        "gc": bool(mechanism["gradient_checkpointing"]),
        "gradient_checkpointing": bool(mechanism["gradient_checkpointing"]),
        "mbs": mbs,
        "gradient_accumulation_steps": TARGET_GBS // (gpu_count * mbs),
        "packing": False,
        "offload": False,
        # "throughput" is the scheduler's fixed-configuration execution path.
        # The scheduler reserves kind="memory_boundary" for adaptive families
        # that carry an `mbs_candidates` sweep; every probe here is a single
        # pre-registered point, so it must use the concrete path -- the same
        # kind every prior memory-boundary campaign used.
        "kind": "throughput",
        "evidence_role": "memory_admission_boundary",
        "fidelity": "formal_3plus10",
        "warmup_steps": 3,
        "measure_steps": 10,
        "repeat": 0,
        "parallel_class": "gpu_partitionable",
        "requires_external_node_idle": False,
    }


def build_jobs(models: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for mechanism in MECHANISMS:
        for source_id in BAND_SOURCES[str(mechanism["band"])]:
            for role in ("probe_low", "probe_high"):
                jobs.append(
                    _job(
                        mechanism,
                        source_id,
                        mechanism[role],
                        role,
                        models,
                        experiment_group="backfill",
                    )
                )
        # Mechanisms whose historical evidence has fewer than two negative
        # boundary sources get one extra high-pressure probe, so the fitted head
        # sees at least two independent unsafe sources.  Long-band mechanisms
        # already probe the long sources, so reusing them would re-run an
        # identical configuration; give those a heavier point instead.
        if mechanism.get("negative_boundary_short"):
            high = dict(mechanism["probe_high"])
            if str(mechanism["band"]) == "long":
                extra_sources = BAND_SOURCES["long"]
                high["mbs"] = int(high["mbs"]) * 2
            else:
                extra_sources = BAND_SOURCES["long"]
            for source_id in extra_sources:
                jobs.append(
                    _job(
                        mechanism,
                        source_id,
                        high,
                        "probe_high_negative_boundary",
                        models,
                        experiment_group="negative_boundary",
                    )
                )

    for source_id in BAND_SOURCES[str(SUPPORTED_MECHANISM["band"])]:
        for role in ("probe_low", "probe_high"):
            jobs.append(
                _job(
                    SUPPORTED_MECHANISM,
                    source_id,
                    SUPPORTED_MECHANISM[role],
                    role,
                    models,
                    experiment_group="prospective_supported",
                )
            )

    job_ids = {job["job_id"] for job in jobs}
    if len(job_ids) != len(jobs):
        raise ValueError("queue contains duplicate job identities")
    # job_id also hashes probe_role, so two differently-labelled probes could
    # still describe the same physical run.  Reject that: it would burn GPU time
    # re-running an identical configuration.
    physical = Counter(
        (
            job["mechanism_id"],
            job["split_unit_id"],
            job["model_id"],
            job["mbs"],
            job["cutoff_len"],
        )
        for job in jobs
    )
    repeated = {key: count for key, count in physical.items() if count > 1}
    if repeated:
        raise ValueError(
            f"queue repeats physically identical configurations: {sorted(repeated)}"
        )
    groups = Counter(job["experiment_group"] for job in jobs)
    if groups["backfill"] != 36:
        raise ValueError(f"expected 36 backfill jobs, got {groups['backfill']}")
    if groups["negative_boundary"] != 6:
        raise ValueError(
            f"expected 6 negative-boundary jobs, got {groups['negative_boundary']}"
        )
    if groups["prospective_supported"] != 4:
        raise ValueError(
            "expected 4 prospective jobs for the already-supported mechanism, "
            f"got {groups['prospective_supported']}"
        )
    if len(jobs) != 46:
        raise ValueError(f"expected 46 jobs in total, got {len(jobs)}")
    covered = {job["mechanism_id"] for job in jobs}
    expected = {row["mechanism_id"] for row in MECHANISMS} | {
        SUPPORTED_MECHANISM["mechanism_id"]
    }
    if covered != expected:
        raise ValueError(f"mechanism coverage mismatch: {covered ^ expected}")
    for mechanism in MECHANISMS:
        sources = {
            job["split_unit_id"]
            for job in jobs
            if job["mechanism_id"] == mechanism["mechanism_id"]
            and job["experiment_group"] == "backfill"
        }
        if len(sources) != 2:
            raise ValueError(
                f"{mechanism['mechanism_id']} must add exactly two business "
                f"sources, got {sorted(sources)}"
            )
    if any(job["train_type"] != "full" for job in jobs):
        raise ValueError("every backfill job must be a FULL fine-tuning job")
    if any(job["packing"] for job in jobs):
        raise ValueError("this campaign is unpacked only")
    return jobs


def build_design(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    outcome = {
        "schema": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "status": "design_and_queue_materialized_waiting_for_exact_approval",
        "gpu_training_started": False,
        "queues_mutated": True,
        "execution_authorized": False,
        "publication_allowed": False,
        "objective": (
            "Give the nine currently fail-closed unpacked FULL mechanisms two "
            "independent business sources each, probing the safe/OOM boundary "
            "the historical rows already located, so their admission heads can "
            "be fitted without widening any safety rule."
        ),
        "evidence_rationale": {
            "historical_full_rows": 596,
            "historical_upstream_corpora": [
                "alpaca_cleaned",
                "ultrachat_200k",
                "longalpaca_12k",
            ],
            "historical_independent_sources_recounted": 3,
            "historical_independent_sources_as_currently_hardcoded": 1,
            "measured_pairwise_content_overlap": 0.0,
            "known_cross_membership": (
                "short_512 and longtail_8192 share 125 sample_id values"
            ),
            "why_new_data_is_still_required": (
                "three upstream corpora is the ceiling for every historical FULL "
                "run, so no recount reaches the five-source planning threshold"
            ),
            "why_business_sources": (
                "all three historical corpora are public; the served workload is "
                "business data, so every source added here is business data"
            ),
            "legacy_442_rows_excluded_from_fit": (
                "hash-only provenance without attempt-bound runtime identity; "
                "usable for locating boundaries, not as release-grade labels"
            ),
        },
        "design": {
            "backfill_mechanisms": len(MECHANISMS),
            "new_business_sources_per_mechanism": 2,
            "probe_points_per_source_per_mechanism": 2,
            "job_count": len(jobs),
            "gpu_job_equivalents": sum(int(job["gpu_count"]) for job in jobs),
            "job_groups": dict(Counter(job["experiment_group"] for job in jobs)),
            "adaptive_third_point_if_bracket_fails": True,
            "adaptive_maximum_jobs": len(jobs) + 18,
            "target_gbs": TARGET_GBS,
            "fidelity": "warmup 3 plus measure 10 optimizer steps",
            "business_sources": sorted(
                {job["split_unit_id"] for job in jobs}
            ),
            "mechanisms": [
                {
                    "mechanism_id": row["mechanism_id"],
                    "mechanism_cn": row["mechanism_cn"],
                    "gpu_count": row["gpu_count"],
                    "zero_stage": row["zero_stage"],
                    "gradient_checkpointing": row["gradient_checkpointing"],
                    "packing": False,
                    "probe_low": row["probe_low"],
                    "probe_high": row["probe_high"],
                    "extra_negative_boundary_probe": bool(
                        row.get("negative_boundary_short")
                    ),
                }
                for row in MECHANISMS
            ],
            "prospective_supported_mechanism": {
                "mechanism_id": SUPPORTED_MECHANISM["mechanism_id"],
                "mechanism_cn": SUPPORTED_MECHANISM["mechanism_cn"],
                "purpose": (
                    "post-freeze prospective acceptance; not fit backfill"
                ),
            },
        },
        "interpretation": {
            "success_and_oom_are_both_valid_safety_evidence": True,
            "oom_is_a_right_censored_lower_bound_never_a_peak_label": True,
            "infra_failure_is_not_oom": True,
            "source_disjoint_outer_validation_required": True,
            "admission_safety_uses_the_maximum_across_all_ranks": True,
            "publication_requires_a_separate_gate": True,
            "no_safety_threshold_is_relaxed_by_this_campaign": True,
        },
        "ordered_job_ids": [job["job_id"] for job in jobs],
        "ordered_job_payload_sha256": sha256_json(jobs),
    }
    outcome["report_sha256"] = sha256_json(outcome)
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    existing = [
        path for path in (args.design, args.queue, args.manifest) if path.exists()
    ]
    if existing:
        raise SystemExit(
            f"refusing to overwrite existing campaign artifacts: {existing}"
        )
    for source_id in SOURCES:
        for path in (_data_path(source_id), _profile_path(source_id)):
            if not path.exists():
                raise SystemExit(f"missing bound input for {source_id}: {path}")

    jobs = build_jobs(_models())
    design = build_design(jobs)
    write_json(args.design, design)
    write_jsonl(args.queue, jobs)

    manifest = {
        "schema": QUEUE_MANIFEST_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "gpu_training_started": False,
        "execution_authorized": False,
        "publication_allowed": False,
        "design": _binding(args.design),
        "queue": {
            **_binding(args.queue),
            "job_count": len(jobs),
            "gpu_job_equivalents": sum(int(job["gpu_count"]) for job in jobs),
            "ordered_job_ids": [job["job_id"] for job in jobs],
            "ordered_job_payload_sha256": sha256_json(jobs),
        },
        "next_step": (
            "review the probe placement against the historical boundaries, "
            "capture fresh H800 occupancy, then request the exact "
            "non-preemptive execution approval; this script never launches GPU work"
        ),
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(args.manifest, manifest)

    print(
        json.dumps(
            {
                "design": _binding(args.design),
                "queue": _binding(args.queue),
                "manifest": _binding(args.manifest),
                "jobs": len(jobs),
                "gpu_job_equivalents": sum(int(job["gpu_count"]) for job in jobs),
                "job_groups": dict(
                    Counter(job["experiment_group"] for job in jobs)
                ),
                "business_sources": len(SOURCES),
                "execution_authorized": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
