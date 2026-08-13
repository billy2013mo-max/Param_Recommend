#!/usr/bin/env python3
"""Freeze four real-business slices and materialize the 40-job interaction matrix.

The experiment isolates whether neat Packing with physical MBS=1 and a larger
cutoff beats ordinary batching with a larger physical MBS.  All jobs use the
same Qwen3-8B LoRA-SFT runtime and one H800.  This script never launches GPUs.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
from typing import Any, Iterable

from transformers import AutoTokenizer
from llamafactory.data.template import TEMPLATES

from common import (
    ARTIFACT_DIR,
    DATA_DIR,
    MATRIX_DIR,
    ROOT,
    percentile,
    read_json,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from inventory_models import inventory_model
from static_packing_predictor import build_decision, load_policy


SCHEMA = "sft_h800_real_business_packing_cutoff_mbs_design/v1"
JOB_SCHEMA = "sft_h800_real_business_packing_cutoff_mbs_job/v1"
CAMPAIGN_ID = "h800_real_business_packing_cutoff_mbs_20260804_v1"
PHASE_ID = "h800_real_business_packing_cutoff_mbs_v1"
MODEL_ID = "qwen3_8b"
MODEL_PATH = Path("/wanqing-models/Qwen3-8B")
TEMPLATE = "qwen3_nothink"
SEED = 20260804
WORKERS = 8
REPEATS = 2
WARMUP_STEPS = 2
MEASURE_STEPS = 8
GPU_POOL = (0, 1, 4, 5, 6, 7)

DATA_BUNDLE_DIR = DATA_DIR / "real_business_packing_cutoff_mbs_v1"
PROFILE_DIR = ARTIFACT_DIR / "real_business_packing_cutoff_mbs_v1" / "profiles"
SUMMARY_DIR = ARTIFACT_DIR / "real_business_packing_cutoff_mbs_v1" / "profile_summaries"
QUEUE = MATRIX_DIR / "h800_real_business_packing_cutoff_mbs_v1.jsonl"
DESIGN = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_design_v1.json"
QUEUE_MANIFEST = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_queue_manifest_v1.json"
STATIC_FEATURES = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_static_features_v1.json"
INVENTORY = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_model_inventory_v1.json"
POLICY_PATH = ARTIFACT_DIR / "static_packing_policy_v1.json"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset_id: str
    display_name: str
    category: str
    source: Path
    expected_source_records: int
    slice_records: int
    cutoff: int
    expanded_cutoff: int
    scale: int
    target_gbs: int


SPECS = (
    DatasetSpec(
        key="d177870",
        dataset_id="real_177870_short_qwen3_v1",
        display_name="177870条-极短集中",
        category="very_short_concentrated",
        source=ROOT.parent
        / "real_business_validation_20260730/datasets/dataset-cl63xd-1785209042/1/publish/dataset-cl63xd-1785209042-V1.jsonl",
        expected_source_records=177_870,
        slice_records=8_192,
        cutoff=1_024,
        expanded_cutoff=4_096,
        scale=4,
        target_gbs=56,
    ),
    DatasetSpec(
        key="d71014",
        dataset_id="real_71014_short_tail_qwen3_v1",
        display_name="71014条-短文本长尾",
        category="short_long_tail",
        source=ROOT.parent
        / "real_business_validation_20260730/datasets/dataset-bxoblq-1785311859/1/publish/dataset-bxoblq-1785311859-V1.jsonl",
        expected_source_records=71_014,
        slice_records=8_192,
        cutoff=4_096,
        expanded_cutoff=16_384,
        scale=4,
        target_gbs=56,
    ),
    DatasetSpec(
        key="d4500edu",
        dataset_id="real_4500_education_qwen3_v1",
        display_name="4500条-教育会话中长",
        category="education_medium_long",
        source=ROOT.parent
        / "real_business_validation_20260730/datasets/dataset-yt0jrk-1775555547/2/publish/dataset-yt0jrk-1775555547-V2.jsonl",
        expected_source_records=4_500,
        slice_records=4_500,
        cutoff=8_192,
        expanded_cutoff=32_768,
        scale=4,
        target_gbs=60,
    ),
    DatasetSpec(
        key="d4500content",
        dataset_id="real_4500_content_longtail_qwen3_v1",
        display_name="4500条-内容评测宽长尾",
        category="content_broad_long_tail",
        source=ROOT.parent
        / "real_business_validation_20260730/datasets/dataset-flieht-1763953980/4/publish/dataset-flieht-1763953980-V4.jsonl",
        expected_source_records=4_500,
        slice_records=4_500,
        cutoff=16_384,
        expanded_cutoff=32_768,
        scale=2,
        target_gbs=60,
    ),
)

ARMS = (
    {"id": "N-C-1", "packing": False, "cutoff": "base", "mbs": "one"},
    {"id": "N-C-k", "packing": False, "cutoff": "base", "mbs": "scale"},
    {"id": "P-C-1", "packing": True, "cutoff": "base", "mbs": "one"},
    {"id": "N-kC-1", "packing": False, "cutoff": "expanded", "mbs": "one"},
    {"id": "P-kC-1", "packing": True, "cutoff": "expanded", "mbs": "one"},
)


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _unwrap(line: str, *, path: Path, line_number: int) -> dict[str, str]:
    raw = json.loads(line)
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
        raw = raw[0]
    if not isinstance(raw, dict):
        raise ValueError(f"{path}:{line_number} is not an object or one-item wrapper")
    fields = {name: raw.get(name) for name in ("system", "prompt", "response")}
    if not all(isinstance(value, str) for value in fields.values()):
        raise ValueError(f"{path}:{line_number} lacks string system/prompt/response")
    return fields  # type: ignore[return-value]


def _reservoir_slice(spec: DatasetSpec) -> tuple[list[tuple[int, dict[str, str]]], int]:
    if not spec.source.is_file():
        raise FileNotFoundError(spec.source)
    rng = random.Random(SEED + sum(map(ord, spec.key)))
    selected: list[tuple[int, dict[str, str]]] = []
    records = 0
    with spec.source.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = _unwrap(line, path=spec.source, line_number=line_number)
            item = (records, row)
            records += 1
            if len(selected) < spec.slice_records:
                selected.append(item)
            else:
                replacement = rng.randrange(records)
                if replacement < spec.slice_records:
                    selected[replacement] = item
    if records != spec.expected_source_records:
        raise ValueError(
            f"{spec.display_name}: expected {spec.expected_source_records} records, got {records}"
        )
    if len(selected) != spec.slice_records:
        raise ValueError(f"{spec.display_name}: frozen slice has {len(selected)} rows")
    rng.shuffle(selected)
    return selected, records


class TrainingEncoder:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True,
        )
        self.template = copy.deepcopy(TEMPLATES[TEMPLATE])
        self.template.fix_special_tokens(self.tokenizer)

    def encode(self, row: dict[str, str]) -> tuple[int, int]:
        messages: list[dict[str, str]] = []
        if row["system"]:
            messages.append({"role": "system", "content": row["system"]})
        messages.extend(
            (
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            )
        )
        system = messages[0]["content"] if messages[0]["role"] == "system" else None
        conversation = messages[1:] if system is not None else messages
        pairs = self.template.encode_multiturn(
            self.tokenizer,
            conversation,
            system=system,
            tools=None,
        )
        source_tokens = sum(len(source_ids) for source_ids, _ in pairs)
        label_tokens = sum(len(target_ids) for _, target_ids in pairs)
        if self.template.efficient_eos:
            label_tokens += 1
        return source_tokens + label_tokens, label_tokens


def _write_dataset_and_profile(
    spec: DatasetSpec,
    encoder: TrainingEncoder,
    *,
    reuse: bool,
) -> dict[str, Any]:
    data_path = DATA_BUNDLE_DIR / f"{spec.dataset_id}.jsonl"
    profile_path = PROFILE_DIR / f"{spec.dataset_id}.qwen3_nothink.jsonl"
    summary_path = SUMMARY_DIR / f"{spec.dataset_id}.json"
    if reuse and data_path.is_file() and profile_path.is_file() and summary_path.is_file():
        summary = read_json(summary_path)
        if (
            summary.get("source", {}).get("sha256") == sha256_file(spec.source)
            and summary.get("slice", {}).get("records") == spec.slice_records
            and summary.get("slice", {}).get("data_sha256") == sha256_file(data_path)
            and summary.get("profile", {}).get("sha256") == sha256_file(profile_path)
        ):
            return summary

    selected, source_records = _reservoir_slice(spec)
    write_jsonl(data_path, (row for _, row in selected))
    profile_rows: list[dict[str, Any]] = []
    for source_index, row in selected:
        total_tokens, label_tokens = encoder.encode(row)
        if total_tokens <= 0 or not 0 <= label_tokens <= total_tokens:
            raise ValueError(f"invalid token profile for {spec.display_name}:{source_index}")
        profile_rows.append(
            {
                "sample_id": f"{spec.dataset_id}:{source_index}",
                "total_tokens": total_tokens,
                "label_tokens": label_tokens,
                "turns": 3 if row["system"] else 2,
                "assistant_turns": 1,
            }
        )
    write_jsonl(profile_path, profile_rows)
    lengths = [int(row["total_tokens"]) for row in profile_rows]
    labels = [int(row["label_tokens"]) for row in profile_rows]
    summary: dict[str, Any] = {
        "schema": "sft_real_business_frozen_slice_profile/v1",
        "dataset_key": spec.key,
        "dataset_id": spec.dataset_id,
        "display_name": spec.display_name,
        "category": spec.category,
        "source": {
            **_binding(spec.source),
            "records": source_records,
        },
        "slice": {
            "selection": "deterministic reservoir sample followed by deterministic shuffle",
            "seed": SEED + sum(map(ord, spec.key)),
            "records": len(selected),
            "source_indices": [index for index, _ in selected],
            "data_path": str(data_path.resolve()),
            "data_sha256": sha256_file(data_path),
        },
        "model_binding": {
            "model_id": MODEL_ID,
            "model_path": str(MODEL_PATH),
            "template": TEMPLATE,
            "tokenizer_config_sha256": sha256_file(MODEL_PATH / "tokenizer_config.json"),
            "model_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        },
        "profile": {
            "path": str(profile_path.resolve()),
            "sha256": sha256_file(profile_path),
            "length_tokens": {
                "minimum": min(lengths),
                "mean": statistics.fmean(lengths),
                "std": statistics.pstdev(lengths),
                "p50": percentile(lengths, 50),
                "p90": percentile(lengths, 90),
                "p95": percentile(lengths, 95),
                "p99": percentile(lengths, 99),
                "maximum": max(lengths),
            },
            "label_tokens": {
                "mean": statistics.fmean(labels),
                "ratio_of_total": sum(labels) / sum(lengths),
            },
        },
        "experiment_geometry": {
            "cutoff": spec.cutoff,
            "expanded_cutoff": spec.expanded_cutoff,
            "scale": spec.scale,
            "target_gbs": spec.target_gbs,
        },
        "gpu_training_started": False,
    }
    summary["report_sha256"] = sha256_json(summary)
    write_json(summary_path, summary)
    return summary


def _update_dataset_registry() -> None:
    registry_path = DATA_DIR / "dataset_info.json"
    registry = read_json(registry_path)
    for spec in SPECS:
        registry[spec.dataset_id] = {
            "file_name": f"real_business_packing_cutoff_mbs_v1/{spec.dataset_id}.jsonl",
            "columns": {"prompt": "prompt", "response": "response", "system": "system"},
        }
    write_json(registry_path, registry)


def _inventory() -> dict[str, Any]:
    model = inventory_model(
        {
            "id": MODEL_ID,
            "nominal_scale_b": 8,
            "path": str(MODEL_PATH),
            "tokenizer_path": str(MODEL_PATH),
            "family": "qwen3",
            "template": TEMPLATE,
            "train_types": ["lora"],
        }
    )
    report: dict[str, Any] = {
        "schema": "sft_h800_real_business_packing_cutoff_mbs_model_inventory/v1",
        "campaign_id": CAMPAIGN_ID,
        "fixed_lora": {"rank": 32, "alpha": 32, "dropout": 0.0, "target": "all"},
        "models": [model],
    }
    report["report_sha256"] = sha256_json(report)
    write_json(INVENTORY, report)
    return report


def _static_features(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    policy = load_policy(POLICY_PATH.resolve())
    rows = []
    for spec in SPECS:
        profile_path = Path(summaries[spec.key]["profile"]["path"])
        for label, cutoff in (("C", spec.cutoff), ("kC", spec.expanded_cutoff)):
            request = {
                "request_id": f"{CAMPAIGN_ID}-{spec.key}-{label}",
                "gpu_family": "H800",
                "modality": "text",
                "stage": "sft",
                "dtype": "bf16",
                "model_id": MODEL_ID,
                "train_type": "lora",
                "profile_path": str(profile_path.resolve()),
                "cutoff_len": cutoff,
                "no_packing_mbs": spec.scale,
                "gpu_count": 1,
                "data_parallel": 1,
                "target_gbs": spec.target_gbs,
                "preprocessing_num_workers": WORKERS,
                "packing_algorithm_id": policy["packing_algorithm"]["id"],
            }
            decision = build_decision(
                request,
                policy=policy,
                policy_path=POLICY_PATH.resolve(),
                request_base=ROOT.parent,
            )
            features = decision["features"]
            if float(features["sample_truncation_rate_without_packing"]) > 0.01:
                raise ValueError(f"{spec.display_name}/{label}: sample truncation exceeds 1%")
            if float(features["tokens_retained_ratio_without_packing"]) < 0.99:
                raise ValueError(f"{spec.display_name}/{label}: retained tokens below 99%")
            geometry = features["packed_batch_geometry"]
            if float(geometry["relative_error"]) > 0.05:
                raise ValueError(
                    f"{spec.display_name}/{label}: packed logical GBS error exceeds 5%: {geometry}"
                )
            rows.append(
                {
                    "dataset_key": spec.key,
                    "display_name": spec.display_name,
                    "cutoff_label": label,
                    "cutoff_len": cutoff,
                    "decision": decision,
                }
            )
    report: dict[str, Any] = {
        "schema": "sft_h800_real_business_packing_cutoff_mbs_static_features/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_before_gpu": True,
        "policy": _binding(POLICY_PATH),
        "gates": {
            "maximum_sample_truncation_rate": 0.01,
            "minimum_token_retention_ratio": 0.99,
            "maximum_expected_logical_gbs_error": 0.05,
        },
        "target_gbs_note": (
            "Per-dataset targets 56/56/60/60 are the nearest values jointly representable "
            "by integer unpacked GA and profile-derived packed GA; this replaces a nominal "
            "GBS64 that would cause 6-10% packed GBS mismatch in three expanded-cutoff arms."
        ),
        "rows": rows,
    }
    report["report_sha256"] = sha256_json(report)
    write_json(STATIC_FEATURES, report)
    return report


def _jobs(
    summaries: dict[str, dict[str, Any]],
    inventory: dict[str, Any],
    static: dict[str, Any],
) -> list[dict[str, Any]]:
    model_parameters = int(inventory["models"][0]["actual_parameters"])
    geometry = {
        (row["dataset_key"], row["cutoff_label"]): row["decision"]["features"][
            "packed_batch_geometry"
        ]
        for row in static["rows"]
    }
    features_sha = sha256_file(STATIC_FEATURES)
    arm_rows: list[tuple[DatasetSpec, dict[str, Any]]] = [
        (spec, arm) for arm in ARMS for spec in SPECS
    ]
    jobs: list[dict[str, Any]] = []
    for repeat in range(REPEATS):
        ordered = arm_rows[repeat * 3 :] + arm_rows[: repeat * 3]
        for spec, arm in ordered:
            cutoff_label = "C" if arm["cutoff"] == "base" else "kC"
            cutoff = spec.cutoff if cutoff_label == "C" else spec.expanded_cutoff
            packing = bool(arm["packing"])
            mbs = 1 if arm["mbs"] == "one" else spec.scale
            if packing:
                packed_geometry = geometry[(spec.key, cutoff_label)]
                ga = int(packed_geometry["gradient_accumulation_steps"])
                expected_gbs = float(packed_geometry["expected_sample_gbs"])
                gbs_error = float(packed_geometry["relative_error"])
            else:
                if spec.target_gbs % mbs:
                    raise ValueError(f"{spec.key}/{arm['id']}: GBS is not divisible by MBS")
                ga = spec.target_gbs // mbs
                expected_gbs = float(spec.target_gbs)
                gbs_error = 0.0
            summary = summaries[spec.key]
            row: dict[str, Any] = {
                "schema": JOB_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "phase_id": PHASE_ID,
                "candidate_role": "real_business_packing_cutoff_mbs_interaction",
                "family_id": spec.key,
                "display_name": spec.display_name,
                "scenario_id": f"{spec.key}-qwen3_8b-lora",
                "arm_id": arm["id"],
                "repeat": repeat,
                "model_id": MODEL_ID,
                "model_family": "qwen3",
                "model_path": str(MODEL_PATH),
                "tokenizer_path": str(MODEL_PATH),
                "template": TEMPLATE,
                "model_parameters": model_parameters,
                "train_type": "lora",
                "dataset_id": spec.dataset_id,
                "profile_id": spec.dataset_id,
                "dataset_category": spec.category,
                "source_dataset_records": spec.expected_source_records,
                "frozen_slice_records": spec.slice_records,
                "data_path": summary["slice"]["data_path"],
                "data_sha256": summary["slice"]["data_sha256"],
                "dataset_profile_path": summary["profile"]["path"],
                "dataset_profile_sha256": summary["profile"]["sha256"],
                "cutoff_label": cutoff_label,
                "base_cutoff_len": spec.cutoff,
                "cutoff_scale": spec.scale,
                "cutoff_len": cutoff,
                "target_gbs": spec.target_gbs,
                "gpu_count": 1,
                "zero": "none",
                "zero_stage": 0,
                "gc": True,
                "gradient_checkpointing": True,
                "mbs": mbs,
                "gradient_accumulation_steps": ga,
                "packing": packing,
                "expected_sample_gbs": expected_gbs,
                "expected_sample_gbs_relative_error": gbs_error,
                "offload": False,
                "gpu_type": "NVIDIA H800 140GB HBM3",
                "required_runtime_gpu_name": "NVIDIA H800",
                "hardware_id": "local_h800_140g",
                "kind": "throughput",
                "warmup_steps": WARMUP_STEPS,
                "measure_steps": MEASURE_STEPS,
                "max_samples": spec.slice_records,
                "fidelity": "formal_real_business_interaction_2plus8",
                "parallel_class": "gpu_partitionable",
                "requires_external_node_idle": False,
                "execution_sequence_index": len(jobs),
                "calibration_partition": {
                    "role": "calibration",
                    "split_unit_id": spec.key,
                    "policy": "real_business_distribution_interaction_fit_only_v1",
                },
                "declared_model_manifest_path": str(INVENTORY.resolve()),
                "declared_model_manifest_sha256": sha256_file(INVENTORY),
                "static_features_path": str(STATIC_FEATURES.resolve()),
                "static_features_sha256": features_sha,
                "matched_nominal_slot_pair": arm["id"] in {"N-C-k", "P-kC-1"},
                "publication_allowed": False,
            }
            row["job_id"] = stable_id("h800realpack", row)
            jobs.append(row)
    if len(jobs) != 40 or len({row["job_id"] for row in jobs}) != 40:
        raise ValueError("interaction matrix must contain 40 unique jobs")
    counts: dict[tuple[str, str], int] = {}
    for row in jobs:
        key = (str(row["family_id"]), str(row["arm_id"]))
        counts[key] = counts.get(key, 0) + 1
    expected = {(spec.key, arm["id"]): 2 for spec in SPECS for arm in ARMS}
    if counts != expected:
        raise ValueError(f"matrix balance drifted: {counts}")
    return jobs


def prepare(*, reuse: bool) -> dict[str, Any]:
    encoder = TrainingEncoder()
    summaries = {
        spec.key: _write_dataset_and_profile(spec, encoder, reuse=reuse) for spec in SPECS
    }
    _update_dataset_registry()
    inventory = _inventory()
    static = _static_features(summaries)
    jobs = _jobs(summaries, inventory, static)
    write_jsonl(QUEUE, jobs)
    jobs_dir = ARTIFACT_DIR / "h800_real_business_packing_cutoff_mbs_jobs_v1"
    for job in jobs:
        write_json(jobs_dir / f"{job['job_id']}.json", job)

    source_files = {
        "design_document": ROOT.parent / "真实业务Packing_Cutoff_MBS交互实验设计_2026-08-04.md",
        "experiment_config": ROOT / "config" / "experiment.json",
        "dataset_registry": DATA_DIR / "dataset_info.json",
        "packing_policy": POLICY_PATH,
        "static_features": STATIC_FEATURES,
        "model_inventory": INVENTORY,
        "preparer": Path(__file__).resolve(),
        "freezer": ROOT / "scripts" / "freeze_h800_real_business_packing_cutoff_mbs_v1.py",
        "evaluator": ROOT / "scripts" / "evaluate_h800_real_business_packing_cutoff_mbs_v1.py",
        "run_job": ROOT / "scripts" / "run_job.py",
        "scheduler": ROOT / "scripts" / "scheduler.py",
        "train_entry": ROOT / "scripts" / "train_entry.py",
        "metrics_callback": ROOT / "scripts" / "metrics_callback.py",
    }
    for spec in SPECS:
        source_files[f"raw_{spec.key}"] = spec.source
        source_files[f"slice_{spec.key}"] = Path(summaries[spec.key]["slice"]["data_path"])
        source_files[f"profile_{spec.key}"] = Path(summaries[spec.key]["profile"]["path"])
        source_files[f"summary_{spec.key}"] = SUMMARY_DIR / f"{spec.dataset_id}.json"

    design: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase_id": PHASE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_gpu_waiting_for_exact_approval",
        "gpu_training_started": False,
        "fit_allowed_after_complete_results": True,
        "acceptance_allowed": False,
        "publication_allowed": False,
        "objective": (
            "Measure how real token-length distributions change the trade-off between "
            "unpacked larger MBS and neat Packing MBS=1 with jointly searched cutoff."
        ),
        "required_gpu_pool": {
            "gpu_ids": list(GPU_POOL),
            "max_gpu_count_per_job": 1,
            "max_parallel_jobs": len(GPU_POOL),
            "preemption_allowed": False,
            "excluded_busy_gpu_ids": [2, 3],
        },
        "queue": {
            **_binding(QUEUE),
            "job_count": len(jobs),
            "ordered_job_ids": [row["job_id"] for row in jobs],
        },
        "matrix": {
            "datasets": [
                {
                    "key": spec.key,
                    "display_name": spec.display_name,
                    "source_records": spec.expected_source_records,
                    "slice_records": spec.slice_records,
                    "C": spec.cutoff,
                    "kC": spec.expanded_cutoff,
                    "k": spec.scale,
                    "target_gbs": spec.target_gbs,
                }
                for spec in SPECS
            ],
            "arms": [dict(arm) for arm in ARMS],
            "repeats": REPEATS,
            "jobs": 40,
        },
        "measurement_contract": {
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "token_source": "consumed_token_ledger/v1",
            "logical_sample_gbs_relative_error_max": 0.05,
            "packed_semantics_required": True,
            "parallel_execution_recorded": True,
            "primary_comparison": "N-C-k versus P-kC-1",
        },
        "static_features": _binding(STATIC_FEATURES),
        "model_inventory": _binding(INVENTORY),
        "source_bindings": {name: _binding(path) for name, path in sorted(source_files.items())},
    }
    design["report_sha256"] = sha256_json(design)
    write_json(DESIGN, design)
    manifest: dict[str, Any] = {
        "schema": "sft_h800_real_business_packing_cutoff_mbs_queue_manifest/v1",
        "campaign_id": CAMPAIGN_ID,
        "design": _binding(DESIGN),
        "queue": {
            **_binding(QUEUE),
            "job_count": len(jobs),
            "ordered_job_ids": [row["job_id"] for row in jobs],
        },
    }
    manifest["report_sha256"] = sha256_json(manifest)
    write_json(QUEUE_MANIFEST, manifest)
    return {
        "design": _binding(DESIGN),
        "queue": _binding(QUEUE),
        "jobs": len(jobs),
        "gpu_pool": list(GPU_POOL),
        "profiles": {
            spec.display_name: summaries[spec.key]["profile"]["length_tokens"] for spec in SPECS
        },
        "logical_gbs": {
            f"{row['dataset_key']}/{row['cutoff_label']}": row["decision"]["features"][
                "packed_batch_geometry"
            ]
            for row in static["rows"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(reuse=args.reuse), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
