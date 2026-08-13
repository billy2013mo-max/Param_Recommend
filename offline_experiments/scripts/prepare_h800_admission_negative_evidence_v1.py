#!/usr/bin/env python3
"""Materialize the negative-evidence admission campaign (second batch).

The first backfill batch added independent business sources and lifted FULL
coverage from 1/10 to 7/10 heads.  Fitting then exposed a different gap: the
remaining mechanisms do not lack sources, they lack *failures*.
``LoRA 1 GPU / gradient checkpointing on`` has 13 independent sources and 55 safe
observations, yet its head cannot be fitted at all, because a binary risk head
with no unsafe example never learns where to refuse.

So this batch deliberately drives configurations to OOM.  Probe placement is
derived from the analytic memory basis at the *effective* sequence length
(``round_up(min(cutoff, profile_max), 8)``) -- using a nominal 32768 cutoff on a
business source whose longest sample is 9077 tokens would silently keep the
pressure low, which is how these mechanisms ended up with no failures.

Writes a design, queue and manifest only.  Never authorizes or launches GPU work.
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


CAMPAIGN_ID = "h800_admission_negative_evidence_20260807_v1"
PHASE_ID = "h800_admission_negative_evidence_v1"
DESIGN_SCHEMA = "sft_h800_admission_negative_evidence_experiment_design/v1"
JOB_SCHEMA = "sft_h800_admission_negative_evidence_job/v1"
QUEUE_MANIFEST_SCHEMA = "sft_h800_admission_negative_evidence_queue_manifest/v1"
TARGET_GBS = 64
SAFE_LIMIT_BYTES = 142635080089.6

DEFAULT_DESIGN = ARTIFACT_DIR / "h800_admission_negative_evidence_design_v1.json"
DEFAULT_QUEUE = ROOT / "matrix" / "h800_admission_negative_evidence_jobs_v1.jsonl"
DEFAULT_MANIFEST = (
    ARTIFACT_DIR / "h800_admission_negative_evidence_queue_manifest_v1.json"
)

DATA_ROOT = ROOT / "data"

# Three business sources untouched by the first batch, chosen because p95 sits
# close to max: a dense tail is what actually raises pressure under the
# min(cutoff, profile_max) policy.  A source with max 3668 but p95 3054 (used
# last batch) cannot be pushed to OOM no matter how large the cutoff.
SOURCES: dict[str, dict[str, Any]] = {
    "lora_s2_src16_product_match": {
        "profile_max": 8210,
        "p95": 8210,
        "campaign_dir": "h800_lora_safety_stage2_v1",
        "rows": 500,
    },
    "lora_s2_src17_content_risk": {
        "profile_max": 8509,
        "p95": 8167,
        "campaign_dir": "h800_lora_safety_stage2_v1",
        "rows": 198,
    },
    "lora_src13_private_trade_compliance": {
        "profile_max": 9077,
        "p95": 8484,
        "campaign_dir": "h800_lora_source_disjoint_v1",
        "rows": 205,
    },
}

# Group A -- mechanisms whose head cannot be fitted for lack of any failure.
# `high` is chosen so the analytic reference exceeds the safety line at the
# effective sequence length; `low` is a deliberate safe control, without which a
# mechanism can end up with only failures and still be unfittable (exactly what
# happened to LoRA 2 GPU / ZeRO-3 / checkpointing off).
GROUP_A: tuple[dict[str, Any], ...] = (
    {
        "mechanism_id": "lora_zero0_gc1_1gpu_pack0",
        "mechanism_cn": "LoRA 1 卡、不切分、开启梯度检查点",
        "training_mode": "lora", "zero_stage": 0, "gradient_checkpointing": True,
        "gpu_count": 1,
        "high": {"model_id": "qwen3_8b", "mbs": 16, "cutoff_len": 16384},
        "low": {"model_id": "qwen3_1p7b", "mbs": 2, "cutoff_len": 2048},
        "gap": "no_unsafe_observation",
    },
    {
        "mechanism_id": "lora_zero2_gc1_2gpu_pack0",
        "mechanism_cn": "LoRA 2 卡、ZeRO-2、开启梯度检查点",
        "training_mode": "lora", "zero_stage": 2, "gradient_checkpointing": True,
        "gpu_count": 2,
        "high": {"model_id": "qwen3_8b", "mbs": 16, "cutoff_len": 16384},
        "low": {"model_id": "qwen3_1p7b", "mbs": 2, "cutoff_len": 2048},
        "gap": "no_unsafe_observation",
    },
    {
        "mechanism_id": "lora_zero3_gc1_2gpu_pack0",
        "mechanism_cn": "LoRA 2 卡、ZeRO-3、开启梯度检查点",
        "training_mode": "lora", "zero_stage": 3, "gradient_checkpointing": True,
        "gpu_count": 2,
        "high": {"model_id": "qwen3_32b", "mbs": 8, "cutoff_len": 16384},
        "low": {"model_id": "qwen3_8b", "mbs": 2, "cutoff_len": 2048},
        "gap": "no_unsafe_observation",
    },
    {
        "mechanism_id": "lora_zero2_gc1_4gpu_pack0",
        "mechanism_cn": "LoRA 4 卡、ZeRO-2、开启梯度检查点",
        "training_mode": "lora", "zero_stage": 2, "gradient_checkpointing": True,
        "gpu_count": 4,
        "high": {"model_id": "qwen3_8b", "mbs": 16, "cutoff_len": 16384},
        "low": {"model_id": "qwen3_1p7b", "mbs": 2, "cutoff_len": 2048},
        "gap": "no_observation_at_all",
    },
    {
        "mechanism_id": "lora_zero3_gc1_4gpu_pack0",
        "mechanism_cn": "LoRA 4 卡、ZeRO-3、开启梯度检查点",
        "training_mode": "lora", "zero_stage": 3, "gradient_checkpointing": True,
        "gpu_count": 4,
        "high": {"model_id": "qwen3_8b", "mbs": 16, "cutoff_len": 16384},
        "low": {"model_id": "qwen3_8b", "mbs": 2, "cutoff_len": 2048},
        "gap": "no_unsafe_observation",
    },
    {
        "mechanism_id": "lora_zero3_gc0_2gpu_pack0",
        "mechanism_cn": "LoRA 2 卡、ZeRO-3、关闭梯度检查点",
        "training_mode": "lora", "zero_stage": 3, "gradient_checkpointing": False,
        "gpu_count": 2,
        # Inverted: this mechanism has six failures and zero safe runs, so the
        # control point matters more than the high one.
        "high": {"model_id": "qwen3_14b", "mbs": 2, "cutoff_len": 16384},
        "low": {"model_id": "qwen3_1p7b", "mbs": 1, "cutoff_len": 1024},
        "gap": "no_safe_observation",
    },
    {
        "mechanism_id": "lora_zero3_gc0_4gpu_pack0",
        "mechanism_cn": "LoRA 4 卡、ZeRO-3、关闭梯度检查点",
        "training_mode": "lora", "zero_stage": 3, "gradient_checkpointing": False,
        "gpu_count": 4,
        "high": {"model_id": "qwen3_14b", "mbs": 2, "cutoff_len": 16384},
        "low": {"model_id": "qwen3_8b", "mbs": 1, "cutoff_len": 2048},
        "gap": "insufficient_sources",
    },
    {
        "mechanism_id": "full_zero2_gc0_2gpu_pack0",
        "mechanism_cn": "FULL 2 卡、ZeRO-2、关闭梯度检查点",
        "training_mode": "full", "zero_stage": 2, "gradient_checkpointing": False,
        "gpu_count": 2,
        "high": {"model_id": "qwen3_8b", "mbs": 2, "cutoff_len": 16384},
        "low": {"model_id": "qwen3_1p7b", "mbs": 1, "cutoff_len": 2048},
        "gap": "insufficient_sources",
    },
)

# Group B -- heads that fit and pass the hard gate, but whose failures come
# almost entirely from the first batch's six sources.  Holding those out makes
# the fit collapse, so the threshold is effectively pinned by one campaign.
# Probe parameters mirror the confirmed points from batch one so old and new
# evidence stay directly comparable; only the data source changes.
GROUP_B: tuple[dict[str, Any], ...] = (
    {
        "mechanism_id": "full_zero0_gc1_1gpu_pack0",
        "mechanism_cn": "FULL 1 卡、不切分、开启梯度检查点",
        "training_mode": "full", "zero_stage": 0, "gradient_checkpointing": True,
        "gpu_count": 1,
        "high": {"model_id": "qwen3_8b", "mbs": 1, "cutoff_len": 4096},
        "low": {"model_id": "qwen3_1p7b", "mbs": 2, "cutoff_len": 2048},
        "gap": "unsafe_evidence_from_single_campaign",
    },
    {
        "mechanism_id": "full_zero2_gc1_2gpu_pack0",
        "mechanism_cn": "FULL 2 卡、ZeRO-2、开启梯度检查点",
        "training_mode": "full", "zero_stage": 2, "gradient_checkpointing": True,
        "gpu_count": 2,
        "high": {"model_id": "qwen3_14b", "mbs": 1, "cutoff_len": 2048},
        "low": {"model_id": "qwen3_1p7b", "mbs": 1, "cutoff_len": 512},
        "gap": "unsafe_evidence_from_single_campaign",
    },
    {
        "mechanism_id": "full_zero3_gc0_4gpu_pack0",
        "mechanism_cn": "FULL 4 卡、ZeRO-3、关闭梯度检查点",
        "training_mode": "full", "zero_stage": 3, "gradient_checkpointing": False,
        "gpu_count": 4,
        "high": {"model_id": "qwen3_8b", "mbs": 8, "cutoff_len": 4096},
        "low": {"model_id": "qwen3_8b", "mbs": 2, "cutoff_len": 2048},
        "gap": "unsafe_evidence_from_single_campaign",
    },
    {
        "mechanism_id": "full_zero3_gc1_4gpu_pack0",
        "mechanism_cn": "FULL 4 卡、ZeRO-3、开启梯度检查点",
        "training_mode": "full", "zero_stage": 3, "gradient_checkpointing": True,
        "gpu_count": 4,
        "high": {"model_id": "qwen3_14b", "mbs": 16, "cutoff_len": 8192},
        "low": {"model_id": "qwen3_8b", "mbs": 2, "cutoff_len": 4096},
        "gap": "unsafe_evidence_from_single_campaign",
    },
)


def _binding(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _models() -> dict[str, Mapping[str, Any]]:
    inventory = read_json(ARTIFACT_DIR / "model_inventory.json")
    return {str(row["id"]): row for row in inventory["models"]}


def _data_path(source_id: str) -> Path:
    return DATA_ROOT / SOURCES[source_id]["campaign_dir"] / f"{source_id}.jsonl"


def _profile_path(source_id: str) -> Path:
    return (
        ARTIFACT_DIR
        / SOURCES[source_id]["campaign_dir"]
        / "profiles"
        / f"{source_id}.qwen3_nothink.jsonl"
    )


def _align8(value: int) -> int:
    return 8 * ((int(value) + 7) // 8)


def _effective_sequence(source_id: str, cutoff_len: int) -> int:
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
    zero_stage = int(mechanism["zero_stage"])
    identity = {
        "campaign_id": CAMPAIGN_ID,
        "mechanism_id": mechanism["mechanism_id"],
        "source_id": source_id,
        "probe_role": probe_role,
        "model_id": model_id,
        "mbs": mbs,
        "cutoff_len": cutoff_len,
    }
    return {
        "schema": JOB_SCHEMA,
        "job_id": stable_id("h800negev", identity),
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "experiment_group": experiment_group,
        "mechanism_id": mechanism["mechanism_id"],
        "mechanism_cn": mechanism["mechanism_cn"],
        "evidence_gap": mechanism["gap"],
        "probe_role": probe_role,
        "scenario_id": (
            f"{source_id}__{mechanism['mechanism_id']}__{model_id}"
            f"_mbs{mbs}_cutoff{cutoff_len}"
        ),
        "split_unit_id": source_id,
        "calibration_purpose": f"admission_negative_evidence_{probe_role}",
        "calibration_partition": {
            "role": "calibration",
            "split_unit_id": source_id,
            "policy": "business_source_disjoint_negative_evidence_v1",
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
        "train_type": str(mechanism["training_mode"]),
        "dataset_id": source_id,
        "dataset_category": "long",
        "dataset_profile_path": str(profile_path.resolve()),
        "dataset_profile_sha256": sha256_file(profile_path),
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "cutoff_len": cutoff_len,
        "raw_profile_max": int(SOURCES[source_id]["profile_max"]),
        "aligned_effective_sequence": _effective_sequence(source_id, cutoff_len),
        "target_gbs": TARGET_GBS,
        "gpu_count": gpu_count,
        "zero_stage": zero_stage,
        # Single-card jobs must declare no DeepSpeed; run_job.validate_job
        # rejects "zero0" there, which killed the first batch mid-run.
        "zero": "none" if gpu_count == 1 else f"zero{zero_stage}",
        "gc": bool(mechanism["gradient_checkpointing"]),
        "gradient_checkpointing": bool(mechanism["gradient_checkpointing"]),
        "mbs": mbs,
        "gradient_accumulation_steps": TARGET_GBS // (gpu_count * mbs),
        "packing": False,
        "offload": False,
        # The scheduler reserves kind="memory_boundary" for adaptive families
        # carrying an mbs_candidates sweep; fixed probes use the concrete path.
        "kind": "throughput",
        "evidence_role": "memory_admission_negative_evidence",
        "fidelity": "formal_3plus10",
        "warmup_steps": 3,
        "measure_steps": 10,
        "repeat": 0,
        "max_samples": int(SOURCES[source_id]["rows"]),
        "parallel_class": "gpu_partitionable",
        "requires_external_node_idle": False,
    }


def build_jobs(models: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for group, mechanisms in (("group_a_missing_failures", GROUP_A),
                              ("group_b_single_campaign_failures", GROUP_B)):
        for mechanism in mechanisms:
            for source_id in SOURCES:
                for role in ("high", "low"):
                    jobs.append(
                        _job(
                            mechanism,
                            source_id,
                            mechanism[role],
                            f"probe_{role}",
                            models,
                            experiment_group=group,
                        )
                    )

    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("queue contains duplicate job identities")
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
    repeated = sorted(key for key, count in physical.items() if count > 1)
    if repeated:
        raise ValueError(f"queue repeats identical physical configurations: {repeated}")

    groups = Counter(job["experiment_group"] for job in jobs)
    expected_a = len(GROUP_A) * len(SOURCES) * 2
    expected_b = len(GROUP_B) * len(SOURCES) * 2
    if groups["group_a_missing_failures"] != expected_a:
        raise ValueError(
            f"group A must hold {expected_a} jobs, got {groups['group_a_missing_failures']}"
        )
    if groups["group_b_single_campaign_failures"] != expected_b:
        raise ValueError(
            f"group B must hold {expected_b} jobs, got "
            f"{groups['group_b_single_campaign_failures']}"
        )
    for mechanism in (*GROUP_A, *GROUP_B):
        scoped = [
            job for job in jobs if job["mechanism_id"] == mechanism["mechanism_id"]
        ]
        if len({job["split_unit_id"] for job in scoped}) != len(SOURCES):
            raise ValueError(
                f"{mechanism['mechanism_id']} must cover all {len(SOURCES)} new sources"
            )
        if {job["probe_role"] for job in scoped} != {"probe_high", "probe_low"}:
            raise ValueError(
                f"{mechanism['mechanism_id']} needs both a high and a low probe; a "
                "high-only design can leave a mechanism with failures but no safe run"
            )
    if any(job["packing"] or job["offload"] for job in jobs):
        raise ValueError("this campaign is unpacked and offload-free")
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
            "Produce independent failure evidence for mechanisms whose admission "
            "head cannot be fitted (no unsafe observation) or whose unsafe "
            "evidence comes from a single campaign, using three business sources "
            "untouched by the first batch."
        ),
        "why_more_sources_alone_would_not_help": (
            "LoRA 1-GPU with gradient checkpointing already has 13 independent "
            "sources and 55 safe observations, and still cannot fit a head: a "
            "binary risk head needs unsafe examples, not more volume."
        ),
        "probe_placement_basis": (
            "analytic memory basis evaluated at the effective sequence "
            "round_up(min(cutoff_len, profile_max), 8); a nominal 32768 cutoff on "
            "a 9077-token source would leave pressure unchanged"
        ),
        "oom_is_expected_output": True,
        "design": {
            "group_a_mechanisms": len(GROUP_A),
            "group_b_mechanisms": len(GROUP_B),
            "new_business_sources": sorted(SOURCES),
            "probes_per_mechanism_per_source": 2,
            "job_count": len(jobs),
            "gpu_job_equivalents": sum(int(job["gpu_count"]) for job in jobs),
            "job_groups": dict(Counter(job["experiment_group"] for job in jobs)),
            "adaptive_third_probe_if_bracket_fails": True,
            "adaptive_maximum_jobs": len(jobs) + len(jobs) // 2,
            "target_gbs": TARGET_GBS,
            "safe_limit_bytes": SAFE_LIMIT_BYTES,
            "mechanisms": [
                {
                    "mechanism_id": row["mechanism_id"],
                    "mechanism_cn": row["mechanism_cn"],
                    "evidence_gap": row["gap"],
                    "training_mode": row["training_mode"],
                    "zero_stage": row["zero_stage"],
                    "gradient_checkpointing": row["gradient_checkpointing"],
                    "gpu_count": row["gpu_count"],
                    "packing": False,
                    "high_probe": row["high"],
                    "low_probe": row["low"],
                }
                for row in (*GROUP_A, *GROUP_B)
            ],
        },
        "acceptance_additions_over_batch_one": {
            "hold_out_all_business_sources_and_refit": (
                "a head must still fit and admit zero unsafe rows after every "
                "business source is withheld; four of batch one's six passing "
                "heads fail this check"
            ),
            "unsafe_evidence_must_span_campaigns": True,
            "leave_one_source_out_alone_is_insufficient": True,
        },
        "interpretation": {
            "oom_is_a_right_censored_lower_bound_never_a_peak_label": True,
            "infra_failure_is_not_oom": True,
            "admission_safety_uses_the_maximum_across_all_ranks": True,
            "no_safety_threshold_is_relaxed_by_this_campaign": True,
            "publication_requires_a_separate_gate": True,
            "high_probe_staying_safe_is_an_informative_result": (
                "it would indicate the mechanism cannot OOM inside the candidate "
                "space, which is a separate finding and not a failed experiment"
            ),
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
            "confirm the competing campaign has finished, rewrite the training "
            "scope, then freeze and promote an approval; this script never "
            "launches GPU work"
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
                "new_business_sources": len(SOURCES),
                "execution_authorized": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
