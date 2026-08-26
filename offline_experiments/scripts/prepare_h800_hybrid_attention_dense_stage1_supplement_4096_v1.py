#!/usr/bin/env python3
"""Supplement stage1 hybrid-attention fit data with cutoff=4096 for qwen3.6-27B.

Rationale
---------
The original stage1 (208 formal jobs) covers exact_total_tokens ∈ {2048, 8192,
16384}.  The v3 prospective acceptance runs at cutoff=4096 for 27B and the
predictor has to *interpolate* across that gap.  For dense_hybrid_attention on
qwen3p6_27b this interpolation predicts 163-187 GB while the actual observations
sit at 95-116 GB — a symptom of an unobserved training point rather than a
fitting-data quality issue.

This CPU-only preparation adds a single anchor:
  - model    = qwen3p6_27b (dense_hybrid_attention)
  - length   = 4096 (exact-length filler, same construction as stage1)
  - source   = 业务短样本 (only; matches stage1's core-orthogonal arm)
  - factors  = 5 parallel routes × gc {True, False} × mbs {1, 2} = 20 jobs
    (mirrors stage1's "核心正交" + "微批扩展" arms for cutoff=4096)

No GPU work is started.  All output files carry the ``supplement_4096_v1``
suffix so they never collide with the original stage1 208-job products.  The
resulting observations are meant to be *appended* to
``h800_hybrid_attention_dense_stage1_observations_v1.jsonl`` before refitting.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_DIR,
    DATA_DIR,
    MATRIX_DIR,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
    write_jsonl,
)
from prepare_h800_hybrid_attention_dense_stage1_v1 import (
    AUTHORIZED_GPU_IDS,
    FEATURE_SOURCE,
    FIXED_LORA,
    MODEL_SPECS,
    PARALLEL_ROUTES,
    PROFILE_SCHEMA,
    ROWS_PER_DATASET,
    SOURCE_DATASETS,
    TARGET_GBS,
    _append_exact_filler,
    _base_job,
    _existing_inventory_rows,
    _messages,
    _source_rows,
    _statistics,
)
from hybrid_attention_memory_features import architecture_signature
from inventory_models import inventory_model
from prepare_h800_qwen35_vl_supplement_v1 import RUNTIME_CONTRACT
from prepare_h800_vl_calibration_profiles_v1 import TrainingEncoder

CAMPAIGN_ID = "h800_hybrid_attention_dense_stage1_supplement_4096_20260825_v1"
FORMAL_PHASE_ID = "h800_hybrid_attention_dense_stage1_supplement_4096_formal_v1"
DESIGN_SCHEMA = "sft_h800_hybrid_attention_dense_stage1_supplement_4096_design/v1"
GENERATED_AT_UTC = "2026-08-25T00:00:00+00:00"

SUPPLEMENT_LENGTH = 4096
SUPPLEMENT_MODEL_IDS = ("qwen3p6_27b",)
SUPPLEMENT_SOURCE_IDS = ("业务短样本",)

MODEL_INVENTORY = (
    ARTIFACT_DIR
    / "h800_hybrid_attention_dense_stage1_supplement_4096_model_inventory_v1.json"
)
DATA_OUTPUT_DIR = DATA_DIR / "hybrid_attention_dense_stage1_supplement_4096_v1"
DATASET_REGISTRY_DIR = DATA_OUTPUT_DIR / "registry"
DATASET_REGISTRY = DATASET_REGISTRY_DIR / "dataset_info.json"
PROFILE_OUTPUT_DIR = (
    ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_supplement_4096_profiles_v1"
)
PROFILE_MANIFEST = (
    ARTIFACT_DIR
    / "h800_hybrid_attention_dense_stage1_supplement_4096_profiles_manifest_v1.json"
)
FORMAL_QUEUE = (
    MATRIX_DIR / "h800_hybrid_attention_dense_stage1_supplement_4096_formal_v1.jsonl"
)
DESIGN = (
    ARTIFACT_DIR / "h800_hybrid_attention_dense_stage1_supplement_4096_design_v1.json"
)

EXPECTED_JOB_COUNT = (
    len(SUPPLEMENT_MODEL_IDS)
    * len(SUPPLEMENT_SOURCE_IDS)
    * len(PARALLEL_ROUTES)
    * 2  # gc {True, False}
    * 2  # mbs {1, 2}
)


def _materialize_inventory() -> dict[str, Any]:
    """Materialize a minimal inventory covering only the supplement model set."""
    existing = _existing_inventory_rows()
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        if spec["id"] not in SUPPLEMENT_MODEL_IDS:
            continue
        if spec["source_inventory"] == "new_scan":
            entry = inventory_model(dict(spec))
        else:
            if spec["id"] not in existing:
                raise ValueError(f"missing source inventory row {spec['id']}")
            entry = copy.deepcopy(existing[spec["id"]])
            for field in (
                "family",
                "release_family",
                "path",
                "tokenizer_path",
                "template",
                "enable_liger_kernel",
            ):
                entry[field] = spec[field]
        entry["architecture_signature"] = architecture_signature(entry)
        rows.append(entry)
    if len(rows) != len(SUPPLEMENT_MODEL_IDS):
        raise ValueError(
            f"inventory produced {len(rows)} rows, expected {len(SUPPLEMENT_MODEL_IDS)}"
        )
    report_core = {
        "schema": (
            "sft_h800_hybrid_attention_dense_stage1_supplement_4096_model_inventory/v1"
        ),
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": GENERATED_AT_UTC,
        "gpu_training_started": False,
        "fixed_lora": FIXED_LORA,
        "supplement_of": "h800_hybrid_attention_dense_stage1_20260812_v1",
        "models": rows,
    }
    report = {**report_core, "report_sha256": sha256_json(report_core)}
    write_json(MODEL_INVENTORY, report)
    return report


def _materialize_profiles(
    inventory: dict[str, Any],
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, Any]]:
    """Build the exact-length training profile(s) for the supplement.

    The construction is a byte-for-byte replay of stage1's core-orthogonal
    profile builder, restricted to the supplement (model, source, length) grid.
    """
    source_rows_cache = {
        source_id: _source_rows(SOURCE_DATASETS[source_id])
        for source_id in SUPPLEMENT_SOURCE_IDS
    }
    registry: dict[str, Any] = {}
    profiles: dict[tuple[str, str, int], dict[str, Any]] = {}
    profile_manifest_rows: list[dict[str, Any]] = []
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model in inventory["models"]:
        model_id = str(model["id"])
        if model_id not in SUPPLEMENT_MODEL_IDS:
            continue
        encoder = TrainingEncoder(
            str(model["tokenizer_path"]), str(model["template"])
        )
        for source_id in SUPPLEMENT_SOURCE_IDS:
            raw_rows = source_rows_cache[source_id]
            source_path = SOURCE_DATASETS[source_id]
            target = SUPPLEMENT_LENGTH
            output_rows: list[dict[str, Any]] = []
            profile_rows: list[dict[str, Any]] = []
            for raw in raw_rows:
                base_messages = _messages(raw)
                base_total, _ = encoder.encode(base_messages)
                if base_total > target:
                    continue
                exact_messages, total, labels = _append_exact_filler(
                    encoder,
                    base_messages,
                    target_total_tokens=target,
                )
                sample_id = str(
                    raw.get("sample_id") or f"source-{len(output_rows)}"
                )
                output_rows.append(
                    {
                        "messages": exact_messages,
                        "sample_id": sample_id,
                        "source_dataset_id": source_id,
                    }
                )
                profile_rows.append(
                    {
                        "schema": PROFILE_SCHEMA,
                        "sample_id": sample_id,
                        "source_dataset_id": source_id,
                        "model_id": model_id,
                        "total_tokens": total,
                        "label_tokens": labels,
                        "target_tokens": target,
                    }
                )
                if len(output_rows) == ROWS_PER_DATASET:
                    break
            if len(output_rows) != ROWS_PER_DATASET:
                raise ValueError(
                    f"not enough <= {target} rows for {model_id}/{source_id}"
                )
            if any(row["total_tokens"] != target for row in profile_rows):
                raise ValueError("exact-length profile construction failed")

            slug = "source_a" if source_id == "业务短样本" else "source_b"
            dataset_id = (
                f"hybrid_dense_stage1_supplement_4096_{model_id}_{slug}_{target}"
            )
            data_path = DATA_OUTPUT_DIR / f"{model_id}.{slug}.{target}.jsonl"
            profile_path = (
                PROFILE_OUTPUT_DIR / f"{model_id}.{slug}.{target}.jsonl"
            )
            write_jsonl(data_path, output_rows)
            write_jsonl(profile_path, profile_rows)
            registry[dataset_id] = {
                "file_name": str(data_path.resolve()),
                "formatting": "sharegpt",
                "columns": {"messages": "messages"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                    "system_tag": "system",
                },
            }
            label_values = [int(row["label_tokens"]) for row in profile_rows]
            profile = {
                "schema": (
                    "sft_h800_hybrid_attention_dense_stage1_supplement_4096_profile/v1"
                ),
                "model_id": model_id,
                "template": model["template"],
                "tokenizer_path": model["tokenizer_path"],
                "tokenizer_json_sha256": sha256_file(
                    Path(model["tokenizer_path"]) / "tokenizer.json"
                ),
                "source_dataset_id": source_id,
                "source_path": str(source_path.resolve()),
                "source_sha256": sha256_file(source_path),
                "dataset_id": dataset_id,
                "data_path": str(data_path.resolve()),
                "data_sha256": sha256_file(data_path),
                "profile_path": str(profile_path.resolve()),
                "profile_sha256": sha256_file(profile_path),
                "rows": ROWS_PER_DATASET,
                "target_total_tokens": target,
                "all_total_tokens_exact": True,
                "label_token_statistics": _statistics(label_values),
                "source_sample_ids_sha256": sha256_json(
                    [row["sample_id"] for row in profile_rows]
                ),
            }
            profiles[(model_id, source_id, target)] = profile
            profile_manifest_rows.append(profile)

    DATASET_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    write_json(DATASET_REGISTRY, registry)
    manifest_core = {
        "schema": (
            "sft_h800_hybrid_attention_dense_stage1_supplement_4096_profiles_manifest/v1"
        ),
        "campaign_id": CAMPAIGN_ID,
        "gpu_training_started": False,
        "profiles": profile_manifest_rows,
        "checks": {
            "profile_count": len(profile_manifest_rows),
            "expected_profile_count": (
                len(SUPPLEMENT_MODEL_IDS)
                * len(SUPPLEMENT_SOURCE_IDS)
            ),
            "all_exact": all(
                row["all_total_tokens_exact"] for row in profile_manifest_rows
            ),
            "all_64_rows": all(
                row["rows"] == ROWS_PER_DATASET for row in profile_manifest_rows
            ),
        },
        "dataset_registry": {
            "path": str(DATASET_REGISTRY.resolve()),
            "sha256": sha256_file(DATASET_REGISTRY),
        },
    }
    manifest = {**manifest_core, "report_sha256": sha256_json(manifest_core)}
    write_json(PROFILE_MANIFEST, manifest)
    return profiles, manifest


def _formal_jobs(
    models: dict[str, dict[str, Any]],
    profiles: dict[tuple[str, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the 20 supplement jobs: core-orthogonal + micro-batch, at 4096."""
    jobs: list[dict[str, Any]] = []
    for model_id in SUPPLEMENT_MODEL_IDS:
        model = models[model_id]
        profile = profiles[(model_id, "业务短样本", SUPPLEMENT_LENGTH)]
        # 核心正交 @ 4096: 5 routes × gc {T, F} × mbs=1
        for gc in (True, False):
            for route_id in PARALLEL_ROUTES:
                route = PARALLEL_ROUTES[route_id]
                jobs.append(
                    _base_job(
                        model=model,
                        profile=profile,
                        phase_id=FORMAL_PHASE_ID,
                        evidence_role="dense_hybrid_memory_calibration",
                        design_arm="核心正交_4096补数",
                        parallel_route_id=route_id,
                        gpu_count=int(route["gpu_count"]),
                        zero=str(route["zero"]),
                        gc=gc,
                        mbs=1,
                        warmup_steps=3,
                        measure_steps=10,
                        repeat=0,
                    )
                )
        # 微批扩展 @ 4096: 5 routes × gc {T, F} × mbs=2
        for gc in (True, False):
            for route_id in PARALLEL_ROUTES:
                route = PARALLEL_ROUTES[route_id]
                jobs.append(
                    _base_job(
                        model=model,
                        profile=profile,
                        phase_id=FORMAL_PHASE_ID,
                        evidence_role="dense_hybrid_memory_calibration",
                        design_arm="微批扩展_4096补数",
                        parallel_route_id=route_id,
                        gpu_count=int(route["gpu_count"]),
                        zero=str(route["zero"]),
                        gc=gc,
                        mbs=2,
                        warmup_steps=3,
                        measure_steps=10,
                        repeat=0,
                    )
                )
    if len(jobs) != EXPECTED_JOB_COUNT:
        raise ValueError(
            f"supplement queue must have {EXPECTED_JOB_COUNT} jobs, got {len(jobs)}"
        )
    if len({row["job_id"] for row in jobs}) != EXPECTED_JOB_COUNT:
        raise ValueError("supplement queue has duplicate job ids")
    # _base_job hard-codes the stage-1 v1 registry paths (its module holds a
    # different DATASET_REGISTRY_DIR / DATASET_REGISTRY constant).  For the
    # supplement we ship a scoped one-entry registry beside the supplement
    # data, so redirect the fields llamafactory reads at load time.  Every
    # other _base_job-provided field is intentionally identical so ledger,
    # provenance and observation joins match the parent campaign.
    supplement_registry_dir = str(DATASET_REGISTRY_DIR.resolve())
    supplement_registry_sha = sha256_file(DATASET_REGISTRY)
    # After the container rebuild on 2026-08-19, the qwen36_venv Torch/FA3
    # wheels no longer share a process-global libcuda: FA3's compiled
    # extension resolves ``cuDriverGetVersion`` from the global symbol table,
    # but neither libc10_cuda.so nor libtorch_cuda.so link libcuda, and
    # torch dlopens it RTLD_LOCAL, so the symbol never becomes visible to
    # dependent extensions loaded after torch.  LD_PRELOAD-ing the host
    # driver library repairs this without touching the pinned contract or
    # the qwen35 tilelang overlay.
    libcuda_path = Path("/lib/x86_64-linux-gnu/libcuda.so.1")
    if not libcuda_path.is_file():
        raise FileNotFoundError(
            f"host libcuda.so.1 is absent, cannot ship LD_PRELOAD: {libcuda_path}"
        )
    libcuda_str = str(libcuda_path)
    for job in jobs:
        job["dataset_dir"] = supplement_registry_dir
        job["dataset_registry_sha256"] = supplement_registry_sha
        overlay = job.get("environment_overlay") or {}
        variables = dict(overlay.get("variables") or {})
        variables["LD_PRELOAD"] = libcuda_str
        overlay["variables"] = variables
        job["environment_overlay"] = overlay
    return jobs


def _counts(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "jobs": len(jobs),
        "by_model": dict(sorted(Counter(row["model_id"] for row in jobs).items())),
        "by_architecture_route": dict(
            sorted(Counter(row["architecture_route"] for row in jobs).items())
        ),
        "by_gpu_count": dict(
            sorted(Counter(str(row["gpu_count"]) for row in jobs).items())
        ),
        "by_zero": dict(sorted(Counter(row["zero"] for row in jobs).items())),
        "by_gc": dict(sorted(Counter(str(row["gc"]) for row in jobs).items())),
        "by_mbs": dict(sorted(Counter(str(row["mbs"]) for row in jobs).items())),
        "by_cutoff": dict(
            sorted(Counter(str(row["cutoff_len"]) for row in jobs).items())
        ),
        "by_arm": dict(
            sorted(Counter(row["design_arm"] for row in jobs).items())
        ),
    }


def _audit(jobs: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "total_20": len(jobs) == EXPECTED_JOB_COUNT,
        "unique_ids": len({row["job_id"] for row in jobs}) == EXPECTED_JOB_COUNT,
        "all_27b": all(row["model_id"] == "qwen3p6_27b" for row in jobs),
        "all_4096": all(row["cutoff_len"] == SUPPLEMENT_LENGTH for row in jobs),
        "all_short_source": all(
            row["source_dataset_id"] == "业务短样本" for row in jobs
        ),
        "gbs_divisible": all(
            (TARGET_GBS % (row["gpu_count"] * row["mbs"])) == 0 for row in jobs
        ),
        "warmup_measure_stage1_shape": all(
            row["warmup_steps"] == 3 and row["measure_steps"] == 10 for row in jobs
        ),
        "no_packing": all(row["packing"] is False for row in jobs),
    }


def prepare() -> dict[str, Any]:
    if not RUNTIME_CONTRACT.is_file():
        raise FileNotFoundError(RUNTIME_CONTRACT)
    inventory = _materialize_inventory()
    models = {str(row["id"]): row for row in inventory["models"]}
    profiles, profile_manifest = _materialize_profiles(inventory)
    formal = _formal_jobs(models, profiles)
    checks = _audit(formal)
    if not all(checks.values()):
        raise ValueError(f"supplement queue audit failed: {checks}")
    write_jsonl(FORMAL_QUEUE, formal)

    design_core = {
        "schema": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": GENERATED_AT_UTC,
        "gpu_training_started": False,
        "approval_promoted": False,
        "supplement_of": "h800_hybrid_attention_dense_stage1_20260812_v1",
        "rationale": (
            "Original stage1 fit data has no exact_total_tokens=4096 training "
            "point.  Predictor interpolates across the {2048, 8192, 16384} gap "
            "and overshoots on qwen3p6_27b at cutoff=4096 during v3 acceptance "
            "(163-187 GB pred vs 95-116 GB obs).  This anchor closes the gap."
        ),
        "scope": {
            "included": [
                "qwen3p6_27b × cutoff=4096 anchor",
                "core-orthogonal + micro-batch design arms",
            ],
            "excluded": [
                "smaller models (4B/9B/8B)",
                "source-invariance (长尾) at 4096",
                "repeat-variance at 4096",
                "packing",
            ],
        },
        "factors": {
            "exact_total_tokens": [SUPPLEMENT_LENGTH],
            "gradient_checkpointing": [True, False],
            "micro_batch_size": [1, 2],
            "parallel_routes": PARALLEL_ROUTES,
            "source_datasets": {
                source_id: {
                    "path": str(SOURCE_DATASETS[source_id].resolve()),
                    "sha256": sha256_file(SOURCE_DATASETS[source_id]),
                }
                for source_id in SUPPLEMENT_SOURCE_IDS
            },
            "packing": [False],
        },
        "required_gpu_pool": {
            "gpu_ids": list(AUTHORIZED_GPU_IDS),
            "max_gpu_count_per_job": 4,
            "allow_gpu_ids_outside_pool": False,
            "source": "reuses stage1 authorization",
        },
        "design_arms": {
            "核心正交_4096补数": 10,
            "微批扩展_4096补数": 10,
        },
        "formal": {
            "queue_path": str(FORMAL_QUEUE.resolve()),
            "queue_sha256": sha256_file(FORMAL_QUEUE),
            "counts": _counts(formal),
            "checks": checks,
            "oom_policy": (
                "right-censored lower bound; never copy capacity as exact peak"
            ),
        },
        "bindings": {
            "model_inventory_path": str(MODEL_INVENTORY.resolve()),
            "model_inventory_sha256": sha256_file(MODEL_INVENTORY),
            "profile_manifest_path": str(PROFILE_MANIFEST.resolve()),
            "profile_manifest_sha256": sha256_file(PROFILE_MANIFEST),
            "dataset_registry_path": str(DATASET_REGISTRY.resolve()),
            "dataset_registry_sha256": sha256_file(DATASET_REGISTRY),
            "feature_basis_path": str(FEATURE_SOURCE),
            "feature_basis_sha256": sha256_file(FEATURE_SOURCE),
            "runtime_contract_path": str(RUNTIME_CONTRACT.resolve()),
            "runtime_contract_sha256": sha256_file(RUNTIME_CONTRACT),
        },
        "execution": {
            "prepare_command": (
                "PYTHONPATH=offline_experiments/scripts "
                "/fine-tuning-launcher/.venv/bin/python "
                "offline_experiments/scripts/"
                "prepare_h800_hybrid_attention_dense_stage1_supplement_4096_v1.py"
            ),
            "formal_preview_command": (
                "python offline_experiments/scripts/scheduler.py --input "
                f"{FORMAL_QUEUE}"
            ),
        },
    }
    design = {**design_core, "report_sha256": sha256_json(design_core)}
    write_json(DESIGN, design)

    return {
        "campaign_id": CAMPAIGN_ID,
        "formal_queue_path": str(FORMAL_QUEUE.resolve()),
        "formal_queue_sha256": sha256_file(FORMAL_QUEUE),
        "formal_job_count": len(formal),
        "counts": _counts(formal),
        "checks": checks,
        "design_path": str(DESIGN.resolve()),
        "profile_manifest_path": str(PROFILE_MANIFEST.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    summary = prepare()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
