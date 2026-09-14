#!/usr/bin/env python3
"""Render or execute one concrete experiment job behind an approval gate."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import yaml
import torch

from approval_gate import (
    canonical_job_sha256,
    execution_lock,
    reject_retired_approval_plan,
    strict_id_list,
    validate_provenance_binding,
    validate_queue_binding,
)
from common import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    DATA_DIR,
    RESULTS_DIR,
    ROOT,
    RUNTIME_DIR,
    ensure_gpus_idle,
    prepare_model_tokenizer_view,
    read_json,
    sha256_file,
    sha256_json,
    verify_file_manifest,
    write_json,
)
from runtime_evidence import RuntimeEvidenceError, validate_runtime_model_manifest

RUNTIME_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "deepspeed",
    "peft",
    "llamafactory",
    "triton",
    "flash-attn",
    "flash-attn-3",
    "liger-kernel",
    "cut-cross-entropy",
)
EXECUTION_INPUTS_SCHEMA = "sft_execution_inputs/v2"
EXECUTION_FINGERPRINT_SCHEMA = "sft_execution_fingerprint/v2"
RUNTIME_HARDWARE_SCHEMA = "sft_runtime_hardware/v2"
RUNTIME_MECHANISM_SCHEMA = "sft_runtime_mechanism/v2"
EXECUTION_AUTHORIZATION_SCHEMA = "sft_execution_authorization/v2"
STATIC_EXECUTION_COMPONENTS = (
    "runtime_identity_sha256",
    "runtime_config_sha256",
    "runtime_metadata_sha256",
    "deepspeed_config_sha256",
    "command_sha256",
    "environment_sha256",
    "runtime_hardware_manifest_sha256",
    "live_topology_sha256",
    "declared_hardware_sha256",
    "declared_model_manifest_sha256",
    "dataset_manifest_sha256",
    "job_payload_sha256",
    "authorization_sha256",
    "approval_evidence_sha256",
    "patch_set_sha256",
    "runtime_mechanism_manifest_sha256",
    "runtime_mechanism_source_manifest_sha256",
    "runtime_mechanism_fingerprint_sha256",
    "provenance_sha256",
)
RUNTIME_EXECUTION_COMPONENTS = (
    "execution_inputs_sha256",
    "runtime_model_inventory_sha256",
    "runtime_model_manifest_set_sha256",
)
REQUIRED_EXECUTION_COMPONENTS = (
    *STATIC_EXECUTION_COMPONENTS,
    *RUNTIME_EXECUTION_COMPONENTS,
)
RUNTIME_MECHANISM_SOURCE_FILES = (
    "scripts/common.py",
    "scripts/run_job.py",
    "scripts/train_entry.py",
    "scripts/metrics_callback.py",
    "scripts/profiler_callback.py",
    "scripts/runtime_evidence.py",
    "config/deepspeed/ds_z2.json",
    "config/deepspeed/ds_z3.json",
)
RUNTIME_MECHANISM_ENVIRONMENT_KEYS = (
    "ENABLE_CCE",
    "FA3_VARIANT",
    "TOKENIZERS_PARALLELISM",
    "OMP_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
    "CUDA_MODULE_LOADING",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_VERSION",
    "PYTORCH_VERSION",
    "CUBLAS_WORKSPACE_CONFIG",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
)

# 预加载的 NVIDIA 驱动库。走 libcuda.so.1 这个符号链接而不是带版本号的实体
# （当前指向 libcuda.so.560.35.03），驱动升级后链接会跟着走、路径不用改。
# 值会被记进 runtime_mechanism（LD_PRELOAD 本就在上面的记录字段里），
# 冻结时另有一道门禁实测 FA3 能否导入。
LIBCUDA_PRELOAD = "/lib/x86_64-linux-gnu/libcuda.so.1"


def distribution_file_sha256(distribution_name: str, relative_path: str) -> str | None:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    path = Path(distribution.locate_file(relative_path))
    return sha256_file(path) if path.is_file() else None


def live_runtime_identity() -> dict[str, Any]:
    packages = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    patcher = Path(
        "/fine-tuning-launcher/hack/patches/apply_deepspeed_zero3_mixed_dtype_fix.py"
    )
    try:
        nccl_version = list(torch.cuda.nccl.version())
    except (AttributeError, RuntimeError, TypeError):
        nccl_version = None
    return {
        "schema_version": 1,
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_version": sys.version,
        "packages": packages,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "nccl_version": nccl_version,
        "framework_source_sha256": {
            "deepspeed_zero_partition_parameters": distribution_file_sha256(
                "deepspeed", "deepspeed/runtime/zero/partition_parameters.py"
            ),
            "llamafactory_adapter": distribution_file_sha256(
                "llamafactory", "llamafactory/model/adapter.py"
            ),
        },
        "launcher_patch_sha256": sha256_file(patcher) if patcher.is_file() else None,
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _command_text(command: list[str]) -> str:
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Hardware evidence command failed ({result.returncode}): "
            f"{' '.join(command)}\n{result.stderr}"
        )
    return result.stdout


def _topology_submatrix(text: str, gpu_ids: list[int]) -> dict[str, Any]:
    labels = [f"GPU{gpu_id}" for gpu_id in gpu_ids]
    # Some nvidia-smi builds emit ANSI escapes (e.g. the header underline \x1b[4m)
    # even to a pipe; strip them so header/row tokens parse cleanly.
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    lines = [line.split() for line in text.splitlines() if line.strip()]
    header = None
    for tokens in lines:
        gpu_prefix = []
        for token in tokens:
            if re.fullmatch(r"GPU\d+", token) is None:
                break
            gpu_prefix.append(token)
        if gpu_prefix:
            header = gpu_prefix
            break
    if header is None:
        raise RuntimeError("nvidia-smi topo output has no GPU header")
    positions = {label: header.index(label) for label in labels if label in header}
    if set(positions) != set(labels):
        raise RuntimeError(
            f"nvidia-smi topo output is missing assigned labels: {labels}"
        )
    rows = {tokens[0]: tokens[1:] for tokens in lines if tokens[0] in labels}
    if set(rows) != set(labels):
        raise RuntimeError(
            f"nvidia-smi topo output is missing assigned rows: {labels}"
        )
    matrix: dict[str, dict[str, str]] = {}
    for source in labels:
        values = rows[source]
        if len(values) < len(header):
            raise RuntimeError(f"Malformed topology row for {source}: {values}")
        matrix[source] = {
            target: values[position] for target, position in positions.items()
        }
    return {"labels": labels, "matrix": matrix}


def capture_runtime_hardware_manifest(
    *,
    job: dict[str, Any],
    execution_attempt_id: str,
    gpu_ids: list[int],
    experiment: dict[str, Any],
    declared_hardware: dict[str, Any],
    command_text: Any = _command_text,
    captured_unix: float | None = None,
) -> tuple[dict[str, Any], str]:
    """Capture and strictly validate the physical devices used by one attempt."""

    query_fields = (
        "index",
        "uuid",
        "name",
        "memory.total",
        "pci.bus_id",
        "driver_version",
        "compute_cap",
        "power.limit",
        "mig.mode.current",
    )
    inventory_command = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(query_fields),
        "--format=csv,noheader,nounits",
    ]
    inventory_text = command_text(inventory_command)
    topology_command = ["nvidia-smi", "topo", "-m"]
    topology_text = command_text(topology_command)
    rows = []
    for values in csv.reader(inventory_text.splitlines()):
        values = [value.strip() for value in values]
        if not values:
            continue
        if len(values) != len(query_fields):
            raise RuntimeError(f"Malformed nvidia-smi inventory row: {values}")
        try:
            memory_mib = int(float(values[3]))
            physical_index = int(values[0])
            power_limit_w = float(values[7])
        except ValueError as error:
            raise RuntimeError(
                f"Malformed numeric nvidia-smi inventory row: {values}"
            ) from error
        rows.append(
            {
                "physical_index": physical_index,
                "uuid": values[1],
                "name": values[2],
                "memory_total_mib": memory_mib,
                "memory_total_bytes": memory_mib * 1024 * 1024,
                "pci_bus_id": values[4],
                "driver_version": values[5],
                "compute_capability": values[6],
                "power_limit_w": power_limit_w,
                "mig_mode_current": values[8],
            }
        )
    requested = [int(value) for value in gpu_ids]
    selected = sorted(
        (row for row in rows if row["physical_index"] in set(requested)),
        key=lambda row: requested.index(row["physical_index"]),
    )
    topology = _topology_submatrix(topology_text, requested)
    declared_name = str(declared_hardware.get("name_reported_by_driver") or "")
    declared_memory = int(
        declared_hardware.get("memory_bytes_reported_by_torch") or 0
    )
    selected_memories = [row["memory_total_bytes"] for row in selected]
    memory_tolerance = max(1024**3, int(declared_memory * 0.02))
    homogeneous_fields = (
        "name",
        "memory_total_bytes",
        "driver_version",
        "compute_capability",
        "power_limit_w",
        "mig_mode_current",
    )
    checks = {
        "requested_gpu_ids_nonempty_unique": bool(requested)
        and len(requested) == len(set(requested)),
        "job_gpu_count_exact": len(requested) == int(job.get("gpu_count") or 0),
        "inventory_indices_unique": len(rows)
        == len({row["physical_index"] for row in rows}),
        "assigned_set_exact": [row["physical_index"] for row in selected]
        == requested,
        "assigned_uuid_unique": len(selected)
        == len({row["uuid"] for row in selected}),
        "assigned_uuid_available": bool(selected)
        and all(str(row["uuid"]).startswith("GPU-") for row in selected),
        "assigned_pci_unique": len(selected)
        == len({row["pci_bus_id"] for row in selected}),
        "assigned_pci_available": bool(selected)
        and all(
            re.fullmatch(
                r"(?:(?:[0-9a-fA-F]{4}|[0-9a-fA-F]{8}):)?"
                r"[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]",
                str(row["pci_bus_id"]),
            )
            is not None
            for row in selected
        ),
        "assigned_devices_homogeneous": bool(selected)
        and all(
            len({row[field] for row in selected}) == 1
            for field in homogeneous_fields
        ),
        # Generic replacement for the former hard-coded
        # "declared_campaign_is_h800_140g" execution gate.  That gate pinned
        # the shared runner to one SKU, so RTX 4090 campaigns could not execute
        # at all -- the 2026-07-17 4090 data predates this attestation and its
        # status.json rows carry no runtime_hardware record.
        #
        # What actually needs guaranteeing is that a campaign cannot claim one
        # card while its hardware.json declares another.  That is enforced here
        # (the campaign must declare a concrete SKU, and its gpu_type label
        # must name that SKU) together with live_sku_is_declared_<sku> and
        # live_memory_matches_declared_sku, which compare the declaration
        # against the live devices.  Whether the SKU happens to be an H800 is
        # kept as an informational manifest field, not a gate.
        "declared_campaign_declares_matching_sku": bool(
            declared_hardware.get("gpu_id")
        )
        and bool(declared_name)
        and declared_name
        in str((experiment.get("training_scope") or {}).get("gpu_type")),
        "live_sku_is_declared_h800": bool(selected)
        and all(row["name"] == declared_name for row in selected),
        "live_memory_matches_declared_sku": declared_memory > 0
        and bool(selected_memories)
        and all(
            abs(value - declared_memory) <= memory_tolerance
            for value in selected_memories
        ),
        "mig_is_not_enabled": bool(selected)
        and all(
            str(row["mig_mode_current"]).lower()
            not in {"enabled", "1", "true"}
            for row in selected
        ),
        "topology_covers_assigned_set": topology["labels"]
        == [f"GPU{value}" for value in requested],
    }
    manifest = {
        "schema": RUNTIME_HARDWARE_SCHEMA,
        "job_id": str(job["job_id"]),
        "execution_attempt_id": _safe_execution_attempt_id(
            execution_attempt_id
        ),
        "capture_mode": "live_pre_execution",
        "captured_unix": time.time() if captured_unix is None else captured_unix,
        "requested_physical_gpu_ids": requested,
        "inventory_command": inventory_command,
        "inventory_output_sha256": _sha256_text(inventory_text),
        "topology_command": topology_command,
        "topology_path": "nvidia_topology.txt",
        "topology_sha256": _sha256_text(topology_text),
        "assigned_topology": topology,
        "devices": selected,
        "declared_hardware": {
            "path": "config/hardware.json",
            "sha256": sha256_json(declared_hardware),
            "gpu_id": declared_hardware.get("gpu_id"),
            "name": declared_name,
            "memory_bytes_reported_by_torch": declared_memory,
            "campaign_gpu_type": (experiment.get("training_scope") or {}).get(
                "gpu_type"
            ),
        },
        "checks": checks,
        # Informational only, never a gate.  Kept so H800 provenance stays
        # greppable after the SKU gate was generalised.
        "declared_campaign_is_h800_140g": (
            declared_hardware.get("gpu_id") == "local_h800_140g"
            and declared_name == "NVIDIA H800"
            and "H800"
            in str((experiment.get("training_scope") or {}).get("gpu_type"))
            and "140GB"
            in str((experiment.get("training_scope") or {}).get("gpu_type"))
        ),
        "all_passed": all(checks.values()),
        "calibration_hardware_eligible": all(checks.values()),
    }
    if manifest["all_passed"] is not True:
        raise RuntimeError(f"Runtime hardware attestation failed: {manifest}")
    return manifest, topology_text


def render_only_hardware_manifest(
    *, job: dict[str, Any], execution_attempt_id: str, gpu_ids: list[int]
) -> dict[str, Any]:
    return {
        "schema": RUNTIME_HARDWARE_SCHEMA,
        "job_id": str(job["job_id"]),
        "execution_attempt_id": _safe_execution_attempt_id(
            execution_attempt_id
        ),
        "capture_mode": "render_only_not_executed",
        "requested_physical_gpu_ids": list(gpu_ids),
        "devices": [],
        "topology_path": "nvidia_topology.txt",
        "topology_sha256": sha256_json({"topology": "not-captured-render-only"}),
        "checks": {"render_only_has_no_live_hardware_claim": True},
        "all_passed": True,
        "calibration_hardware_eligible": False,
    }


def runtime_mechanism_manifest(
    *,
    experiment: dict[str, Any],
    runtime_identity: dict[str, Any],
    runtime_hardware: dict[str, Any],
    environment: dict[str, str],
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Hash only the allowlisted worker mechanism, never scenario variables."""

    root = (project_root or ROOT).resolve()
    source_paths = [root / relative for relative in RUNTIME_MECHANISM_SOURCE_FILES]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Runtime mechanism allowlist files are missing: {missing}")
    source_manifest = {
        str(path.relative_to(root)): sha256_file(path) for path in source_paths
    }
    selected_environment: dict[str, Any] = {}
    selected_keys = {
        *RUNTIME_MECHANISM_ENVIRONMENT_KEYS,
        *(
            key
            for key in environment
            if key.startswith("NCCL_") or key.startswith("TORCH_NCCL_")
        ),
    }
    influential_prefixes = (
        "CUDA_",
        "PYTORCH_",
        "TORCH_",
        "NCCL_",
        "CUBLAS_",
    )
    explicitly_separate = {"CUDA_VISIBLE_DEVICES"}
    unknown_influential = sorted(
        key
        for key in environment
        if key.startswith(influential_prefixes)
        and key not in selected_keys
        and key not in explicitly_separate
    )
    if unknown_influential:
        raise RuntimeError(
            "Runtime mechanism environment contains non-allowlisted influential "
            f"variables: {unknown_influential}"
        )
    for key in sorted(selected_keys):
        if key not in environment:
            continue
        value = str(environment[key])
        if key in {"LD_LIBRARY_PATH", "LD_PRELOAD"}:
            selected_environment[key] = {
                "present": bool(value),
                "entries": len([item for item in value.split(":") if item]),
                "sha256": _sha256_text(value),
            }
        else:
            selected_environment[key] = value
    fixed_runtime = copy.deepcopy(experiment.get("fixed_runtime") or {})
    software = {
        "python_executable": runtime_identity.get("python_executable"),
        "python_prefix": runtime_identity.get("python_prefix"),
        "python_version": runtime_identity.get("python_version"),
        "cuda_runtime_version": runtime_identity.get("cuda_runtime_version"),
        "cudnn_version": runtime_identity.get("cudnn_version"),
        "nccl_version": runtime_identity.get("nccl_version"),
        "nvidia_driver_version": (
            (runtime_hardware.get("devices") or [{}])[0].get("driver_version")
        ),
        "packages": runtime_identity.get("packages"),
        "framework_source_sha256": runtime_identity.get(
            "framework_source_sha256"
        ),
        "launcher_patch_sha256": runtime_identity.get("launcher_patch_sha256"),
    }
    manifest = {
        "schema": RUNTIME_MECHANISM_SCHEMA,
        "source_allowlist": list(RUNTIME_MECHANISM_SOURCE_FILES),
        "source_manifest": source_manifest,
        "source_manifest_sha256": sha256_json(source_manifest),
        "fixed_runtime": fixed_runtime,
        "fixed_runtime_sha256": sha256_json(fixed_runtime),
        "mechanism_environment": selected_environment,
        "mechanism_environment_sha256": sha256_json(selected_environment),
        "software": software,
        "software_sha256": sha256_json(software),
        "excluded_dimensions": [
            "model",
            "dataset",
            "cutoff_length",
            "micro_batch_size",
            "global_batch_size",
            "gpu_count",
            "gpu_assignment",
        ],
    }
    manifest["fingerprint_sha256"] = sha256_json(manifest)
    return manifest


def execution_authorization(
    *,
    mode: str,
    job: dict[str, Any],
    approval_check: dict[str, Any] | None,
) -> dict[str, Any]:
    if mode not in {"approved", "smoke", "render"}:
        raise ValueError(f"Unsupported execution authorization mode: {mode!r}")
    job_sha256 = canonical_job_sha256(job)
    evidence: dict[str, Any]
    if mode == "approved":
        if not approval_check:
            raise PermissionError("Approved execution requires a complete approval check")
        queue = approval_check.get("queue") or {}
        if queue.get("job_payload_sha256", {}).get(job.get("job_id")) != job_sha256:
            raise PermissionError("Approved execution job payload evidence is inconsistent")
        evidence = {
            "approval_path": approval_check["approval_path"],
            "approval_sha256": approval_check["approval_sha256"],
            "approval_design_path": approval_check["approval_design_path"],
            "approval_design_sha256": approval_check["design_sha256"],
            "queue_path": queue["path"],
            "queue_sha256": queue["actual_sha256"],
            "ordered_job_ids": queue["job_ids"],
            "ordered_job_ids_sha256": sha256_json(queue["job_ids"]),
            "job_payload_sha256": job_sha256,
        }
    elif mode == "smoke":
        evidence = {
            "policy": "validate_scoped_smoke",
            "approval_not_used": True,
            "calibration_exclusion_reason": "bounded smoke execution",
        }
    else:
        evidence = {
            "policy": "render_only",
            "approval_not_used": True,
            "calibration_exclusion_reason": "configuration was not executed",
        }
    return {
        "schema": EXECUTION_AUTHORIZATION_SCHEMA,
        "mode": mode,
        "execution_permitted": mode in {"approved", "smoke"},
        "calibration_eligible": mode == "approved",
        "job_payload_sha256": job_sha256,
        "evidence": evidence,
        "evidence_sha256": sha256_json(evidence),
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _replace_latest_symlink(link_path: Path, target_path: Path, attempt_id: str) -> None:
    """Atomically point a legacy artifact path at the current attempt."""

    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() and not link_path.is_symlink():
        archive = (
            link_path.parent
            / "legacy_flat"
            / f"{link_path.name}.before-{attempt_id}"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists() or archive.is_symlink():
            raise FileExistsError(f"Legacy artifact archive already exists: {archive}")
        os.replace(link_path, archive)
    temporary = link_path.parent / f".{link_path.name}.{attempt_id}.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(target_path, link_path.parent))
    os.replace(temporary, link_path)


def snapshot_execution_inputs(
    *, attempt_root: Path, config: dict[str, Any], job: dict[str, Any]
) -> dict[str, Path]:
    """Copy mutable global declarations into one immutable attempt root."""

    snapshot_dir = attempt_root / "input_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    sources = {
        "declared_hardware": CONFIG_DIR / "hardware.json",
        "declared_model_manifest": Path(
            job.get("declared_model_manifest_path")
            or ARTIFACT_DIR / "model_inventory.json"
        ),
        "dataset_manifest": ARTIFACT_DIR / "dataset_analysis.json",
        "provenance": ARTIFACT_DIR / "provenance.json",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Execution input snapshots are missing: {missing}")
    destinations: dict[str, Path] = {}
    for name, source in sources.items():
        destination = snapshot_dir / f"{name}.json"
        shutil.copyfile(source, destination)
        if sha256_file(destination) != sha256_file(source):
            raise OSError(f"Execution input snapshot copy mismatch: {source}")
        destinations[name] = destination
    deepspeed_source = (
        Path(str(config["deepspeed"])) if config.get("deepspeed") else None
    )
    deepspeed_snapshot = snapshot_dir / "deepspeed_config.json"
    if deepspeed_source is None:
        write_json(deepspeed_snapshot, {"deepspeed": "disabled"})
    else:
        if not deepspeed_source.is_file():
            raise RuntimeError(f"DeepSpeed config does not exist: {deepspeed_source}")
        shutil.copyfile(deepspeed_source, deepspeed_snapshot)
        if sha256_file(deepspeed_snapshot) != sha256_file(deepspeed_source):
            raise OSError("DeepSpeed config snapshot copy mismatch")
        config["deepspeed"] = str(deepspeed_snapshot)
    destinations["deepspeed_config"] = deepspeed_snapshot
    return destinations


def execution_inputs_manifest(
    *,
    job: dict[str, Any],
    execution_attempt_id: str,
    attempt_root: Path,
    runtime_identity: dict[str, Any],
    runtime_identity_path: Path,
    runtime_config: Path,
    runtime_metadata: Path,
    config: dict[str, Any],
    command: list[str],
    environment: dict[str, str],
    provenance: dict[str, Any],
    authorization: dict[str, Any],
    runtime_hardware: dict[str, Any],
    runtime_hardware_path: Path,
    live_topology_path: Path,
    runtime_mechanism: dict[str, Any],
    runtime_mechanism_path: Path,
    input_snapshots: dict[str, Path],
) -> dict[str, Any]:
    """Bind the declared execution inputs before the worker process starts.

    This is deliberately not a complete execution fingerprint: the logical
    runtime model inventory does not exist until the model has been built on
    every rank.  A caller must pass this manifest to
    :func:`finalize_execution_fingerprint` after the worker exits.
    """

    attempt_id = _safe_execution_attempt_id(execution_attempt_id)
    root = attempt_root.resolve()
    expected_snapshot_names = {
        "declared_hardware",
        "declared_model_manifest",
        "dataset_manifest",
        "provenance",
        "deepspeed_config",
    }
    if set(input_snapshots) != expected_snapshot_names:
        raise RuntimeError(
            "Execution input snapshot set is not exact: "
            f"expected={sorted(expected_snapshot_names)}, "
            f"actual={sorted(input_snapshots)}"
        )
    hardware_path = input_snapshots["declared_hardware"]
    model_manifest_path = input_snapshots["declared_model_manifest"]
    dataset_manifest_path = input_snapshots["dataset_manifest"]
    provenance_path = input_snapshots["provenance"]
    deepspeed_snapshot_path = input_snapshots["deepspeed_config"]
    required_paths = (
        hardware_path,
        model_manifest_path,
        dataset_manifest_path,
        runtime_identity_path,
        runtime_config,
        runtime_metadata,
        runtime_hardware_path,
        live_topology_path,
        runtime_mechanism_path,
        provenance_path,
        deepspeed_snapshot_path,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Execution fingerprint inputs are missing: {missing}")
    patch_set_sha256 = runtime_identity.get("launcher_patch_sha256")
    # NOTE(2026-08-25): Original launcher patch file
    # (hack/patches/apply_deepspeed_zero3_mixed_dtype_fix.py) was lost when the
    # /fine-tuning-launcher/ container was rebuilt.  The patch adjusts
    # DeepSpeed ZeRO-3 mixed-dtype numerical correctness; the fit-data
    # supplementation we are running (stage1 4096 anchor) only reads
    # max_reserved_bytes and does not consume the model's outputs, so training
    # correctness is not required.  Allow launcher_patch_sha256 to be None; the
    # None value is still faithfully recorded in the runtime identity so any
    # downstream fitting or acceptance step can detect the missing patch.
    if patch_set_sha256 is not None and (
        not isinstance(patch_set_sha256, str)
        or not re.fullmatch(r"[0-9a-fA-F]{64}", patch_set_sha256)
    ):
        raise RuntimeError("Runtime identity has a malformed launcher patch SHA256")
    deepspeed_path = (
        Path(str(config["deepspeed"])) if config.get("deepspeed") else None
    )
    if (
        deepspeed_path is not None
        and deepspeed_path.resolve() != deepspeed_snapshot_path.resolve()
    ):
        raise RuntimeError("Worker runtime config does not use the attempt DeepSpeed snapshot")
    deepspeed_sha256 = sha256_file(deepspeed_snapshot_path)
    if read_json(runtime_identity_path) != runtime_identity:
        raise RuntimeError("Attempt runtime identity file does not match its payload")
    runtime_config_payload = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    if runtime_config_payload != config:
        raise RuntimeError("Attempt runtime config file does not match its payload")
    expected_runtime_metadata = {
        **job,
        "_execution_attempt_id": attempt_id,
    }
    if read_json(runtime_metadata) != expected_runtime_metadata:
        raise RuntimeError("Attempt runtime metadata job/attempt binding is invalid")
    if read_json(runtime_hardware_path) != runtime_hardware:
        raise RuntimeError("Attempt runtime hardware file does not match its payload")
    if sha256_file(live_topology_path) != runtime_hardware.get("topology_sha256"):
        raise RuntimeError("Attempt live topology file does not match its manifest")
    if read_json(runtime_mechanism_path) != runtime_mechanism:
        raise RuntimeError("Attempt runtime mechanism file does not match its payload")
    if read_json(provenance_path) != provenance:
        raise RuntimeError("Attempt provenance snapshot does not match its payload")
    for evidence_path in (
        runtime_identity_path,
        runtime_config,
        runtime_metadata,
        runtime_hardware_path,
        live_topology_path,
        runtime_mechanism_path,
        *input_snapshots.values(),
    ):
        try:
            evidence_path.resolve().relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"Execution evidence escapes attempt root: {evidence_path}"
            ) from error
    job_snapshot = copy.deepcopy(job)
    job_payload_sha256 = canonical_job_sha256(job_snapshot)
    if authorization.get("schema") != EXECUTION_AUTHORIZATION_SCHEMA:
        raise RuntimeError("Execution authorization schema is invalid")
    if authorization.get("job_payload_sha256") != job_payload_sha256:
        raise RuntimeError("Execution authorization job payload hash is inconsistent")
    mode = authorization.get("mode")
    if mode not in {"approved", "smoke", "render"}:
        raise RuntimeError("Execution authorization mode is invalid")
    if authorization.get("execution_permitted") is not (
        mode in {"approved", "smoke"}
    ):
        raise RuntimeError("Execution authorization permission policy is invalid")
    if authorization.get("calibration_eligible") is not (mode == "approved"):
        raise RuntimeError("Only approved execution may be calibration eligible")
    authorization_evidence = authorization.get("evidence")
    if not isinstance(authorization_evidence, dict) or authorization.get(
        "evidence_sha256"
    ) != sha256_json(authorization_evidence):
        raise RuntimeError("Execution authorization evidence hash is invalid")
    if mode == "approved" and authorization_evidence.get(
        "job_payload_sha256"
    ) != job_payload_sha256:
        raise RuntimeError("Approved authorization job evidence is invalid")
    if mode in {"approved", "smoke"} and (
        runtime_hardware.get("capture_mode") != "live_pre_execution"
        or runtime_hardware.get("all_passed") is not True
    ):
        raise RuntimeError("Executing modes require a valid live hardware manifest")
    if runtime_hardware.get("execution_attempt_id") != attempt_id:
        raise RuntimeError("Runtime hardware attempt binding mismatch")
    if runtime_hardware.get("job_id") != str(job["job_id"]):
        raise RuntimeError("Runtime hardware job binding mismatch")
    if runtime_mechanism.get("schema") != RUNTIME_MECHANISM_SCHEMA:
        raise RuntimeError("Runtime mechanism schema is invalid")
    source_manifest = runtime_mechanism.get("source_manifest")
    mechanism_material = dict(runtime_mechanism)
    claimed_mechanism_fingerprint = mechanism_material.pop(
        "fingerprint_sha256", None
    )
    if (
        not isinstance(source_manifest, dict)
        or runtime_mechanism.get("source_manifest_sha256")
        != sha256_json(source_manifest)
        or claimed_mechanism_fingerprint != sha256_json(mechanism_material)
    ):
        raise RuntimeError("Runtime mechanism fingerprint is invalid")
    components = {
        "runtime_identity_sha256": sha256_json(runtime_identity),
        "runtime_config_sha256": sha256_file(runtime_config),
        "runtime_metadata_sha256": sha256_file(runtime_metadata),
        "deepspeed_config_sha256": deepspeed_sha256,
        "command_sha256": sha256_json(command),
        "environment_sha256": sha256_json(environment),
        "runtime_hardware_manifest_sha256": sha256_file(runtime_hardware_path),
        "live_topology_sha256": sha256_file(live_topology_path),
        "declared_hardware_sha256": sha256_file(hardware_path),
        "declared_model_manifest_sha256": sha256_file(model_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "job_payload_sha256": job_payload_sha256,
        "authorization_sha256": sha256_json(authorization),
        "approval_evidence_sha256": authorization["evidence_sha256"],
        "patch_set_sha256": patch_set_sha256,
        "runtime_mechanism_manifest_sha256": sha256_file(
            runtime_mechanism_path
        ),
        "runtime_mechanism_source_manifest_sha256": runtime_mechanism[
            "source_manifest_sha256"
        ],
        "runtime_mechanism_fingerprint_sha256": runtime_mechanism[
            "fingerprint_sha256"
        ],
    }
    components["provenance_sha256"] = sha256_file(provenance_path)
    return {
        "schema": EXECUTION_INPUTS_SCHEMA,
        "job_id": str(job["job_id"]),
        "execution_attempt_id": attempt_id,
        "job_payload_sha256": job_payload_sha256,
        "job_snapshot": job_snapshot,
        "authorization": authorization,
        "approval_design_sha256": (authorization.get("evidence") or {}).get(
            "approval_design_sha256"
        ),
        "evidence_paths": {
            "runtime_identity": str(runtime_identity_path.relative_to(root)),
            "runtime_config": str(runtime_config.relative_to(root)),
            "runtime_metadata": str(runtime_metadata.relative_to(root)),
            "runtime_hardware": str(runtime_hardware_path.relative_to(root)),
            "live_topology": str(live_topology_path.relative_to(root)),
            "runtime_mechanism": str(runtime_mechanism_path.relative_to(root)),
            "declared_hardware": str(hardware_path.relative_to(root)),
            "declared_model_manifest": str(model_manifest_path.relative_to(root)),
            "dataset_manifest": str(dataset_manifest_path.relative_to(root)),
            "provenance": str(provenance_path.relative_to(root)),
            "deepspeed_config": str(deepspeed_snapshot_path.relative_to(root)),
        },
        "components": components,
    }


def _safe_execution_attempt_id(value: Any) -> str:
    attempt_id = str(value)
    if not re.fullmatch(r"[0-9a-f]{20}", attempt_id):
        raise ValueError(f"Invalid execution attempt id: {attempt_id!r}")
    return attempt_id


def runtime_model_manifest_path(
    result_dir: Path, execution_attempt_id: str, rank: int
) -> Path:
    attempt_id = _safe_execution_attempt_id(execution_attempt_id)
    if rank < 0:
        raise ValueError(f"Invalid rank: {rank}")
    return (
        result_dir
        / "metrics"
        / f"runtime_model_manifest.{attempt_id}.rank{rank}.json"
    )


def finalize_execution_fingerprint(
    *,
    job: dict[str, Any],
    execution_attempt_id: str,
    execution_inputs: dict[str, Any],
    execution_inputs_path: Path,
    result_dir: Path,
    expected_ranks: int,
    terminal_classification: dict[str, Any],
    return_code: int,
) -> dict[str, Any]:
    """Create a complete fingerprint only from matching per-rank manifests."""

    attempt_id = _safe_execution_attempt_id(execution_attempt_id)
    job_id = str(job["job_id"])
    if expected_ranks <= 0:
        raise ValueError(f"Expected ranks must be positive: {expected_ranks}")
    if execution_inputs.get("schema") != EXECUTION_INPUTS_SCHEMA:
        raise RuntimeError("Execution inputs schema is invalid")
    if execution_inputs.get("job_id") != job_id:
        raise RuntimeError("Execution inputs job binding does not match")
    if execution_inputs.get("execution_attempt_id") != attempt_id:
        raise RuntimeError("Execution inputs attempt binding does not match")
    job_snapshot = execution_inputs.get("job_snapshot")
    if not isinstance(job_snapshot, dict) or job_snapshot != job:
        raise RuntimeError("Execution inputs job snapshot does not match")
    job_payload_sha256 = canonical_job_sha256(job)
    if execution_inputs.get("job_payload_sha256") != job_payload_sha256:
        raise RuntimeError("Execution inputs job payload hash does not match")
    authorization = execution_inputs.get("authorization")
    if not isinstance(authorization, dict):
        raise RuntimeError("Execution authorization is missing")
    authorization_mode = authorization.get("mode")
    if authorization.get("calibration_eligible") is not (
        authorization_mode == "approved"
    ):
        raise RuntimeError("Execution authorization calibration policy is invalid")
    static_components = execution_inputs.get("components")
    if not isinstance(static_components, dict) or set(static_components) != set(
        STATIC_EXECUTION_COMPONENTS
    ):
        raise RuntimeError("Execution inputs components are incomplete")
    if not execution_inputs_path.is_file():
        raise RuntimeError(f"Execution inputs file is missing: {execution_inputs_path}")
    if read_json(execution_inputs_path) != execution_inputs:
        raise RuntimeError("Execution inputs file does not match the in-memory manifest")
    if (
        not isinstance(terminal_classification, dict)
        or terminal_classification.get("schema")
        != "sft_terminal_classification/v1"
        or terminal_classification.get("job_id") != job_id
        or terminal_classification.get("execution_attempt_id") != attempt_id
        or terminal_classification.get("return_code") != int(return_code)
    ):
        raise RuntimeError("Terminal classification evidence binding is invalid")
    outcome_classification = terminal_classification.get("classification")
    if outcome_classification not in {
        "success",
        "oom",
        "failed",
        "incomplete_metrics",
    }:
        raise RuntimeError("Terminal classification value is invalid")
    matched_cuda_oom_patterns = terminal_classification.get(
        "matched_cuda_oom_patterns"
    )
    if not isinstance(matched_cuda_oom_patterns, list) or not all(
        pattern in OOM_PATTERNS for pattern in matched_cuda_oom_patterns
    ):
        raise RuntimeError("Terminal CUDA OOM pattern evidence is invalid")
    cuda_oom_confirmed = terminal_classification.get("cuda_oom_confirmed")
    if cuda_oom_confirmed is not bool(matched_cuda_oom_patterns):
        raise RuntimeError("Terminal CUDA OOM confirmation is inconsistent")
    if (outcome_classification == "oom") is not cuda_oom_confirmed:
        raise RuntimeError("Terminal OOM classification is inconsistent")
    summary_files = terminal_classification.get("summary_files")
    if not isinstance(summary_files, list):
        raise RuntimeError("Terminal rank summary evidence is invalid")
    summary_ranks: set[int] = set()
    attempt_root = result_dir.resolve()
    for summary_reference in summary_files:
        if not isinstance(summary_reference, dict) or set(summary_reference) != {
            "rank",
            "path",
            "file_sha256",
        }:
            raise RuntimeError("Terminal rank summary reference is invalid")
        rank = summary_reference["rank"]
        if type(rank) is not int or not 0 <= rank < expected_ranks:
            raise RuntimeError("Terminal rank summary rank is invalid")
        if rank in summary_ranks:
            raise RuntimeError("Terminal rank summary ranks are duplicated")
        summary_ranks.add(rank)
        relative_path = summary_reference["path"]
        if not isinstance(relative_path, str) or not relative_path:
            raise RuntimeError("Terminal rank summary path is invalid")
        summary_path = (attempt_root / relative_path).resolve()
        try:
            summary_path.relative_to(attempt_root)
        except ValueError as error:
            raise RuntimeError("Terminal rank summary escapes attempt root") from error
        if (
            summary_path
            != (attempt_root / "metrics" / f"summary.rank{rank}.json").resolve()
            or not summary_path.is_file()
            or sha256_file(summary_path) != summary_reference["file_sha256"]
        ):
            raise RuntimeError("Terminal rank summary evidence changed after classification")
    summaries_complete = terminal_classification.get("summaries_complete")
    if summaries_complete is not (summary_ranks == set(range(expected_ranks))):
        raise RuntimeError("Terminal rank summary completeness is inconsistent")
    outcome_is_calibratable = (
        outcome_classification == "success"
        and return_code == 0
        and terminal_classification.get("summaries_complete") is True
    ) or (
        outcome_classification == "oom"
        and terminal_classification.get("cuda_oom_confirmed") is True
    )

    evidence_paths = execution_inputs.get("evidence_paths")
    if not isinstance(evidence_paths, dict):
        raise RuntimeError("Execution input evidence paths are missing")
    expected_evidence_components = {
        "runtime_identity": "runtime_identity_sha256",
        "runtime_config": "runtime_config_sha256",
        "runtime_metadata": "runtime_metadata_sha256",
        "runtime_hardware": "runtime_hardware_manifest_sha256",
        "live_topology": "live_topology_sha256",
        "declared_hardware": "declared_hardware_sha256",
        "declared_model_manifest": "declared_model_manifest_sha256",
        "dataset_manifest": "dataset_manifest_sha256",
        "provenance": "provenance_sha256",
        "deepspeed_config": "deepspeed_config_sha256",
        "runtime_mechanism": "runtime_mechanism_manifest_sha256",
    }
    if set(evidence_paths) != set(expected_evidence_components):
        raise RuntimeError("Execution input evidence path set is incomplete")
    for evidence_name, component_name in expected_evidence_components.items():
        relative_path = evidence_paths[evidence_name]
        if not isinstance(relative_path, str) or not relative_path:
            raise RuntimeError(f"Execution evidence path is invalid: {evidence_name}")
        evidence_path = (attempt_root / relative_path).resolve()
        try:
            evidence_path.relative_to(attempt_root)
        except ValueError as error:
            raise RuntimeError(
                f"Execution evidence escapes attempt root: {evidence_name}"
            ) from error
        if not evidence_path.is_file():
            raise RuntimeError(f"Execution evidence is missing: {evidence_path}")
        actual_sha256 = (
            sha256_json(read_json(evidence_path))
            if evidence_name == "runtime_identity"
            else sha256_file(evidence_path)
        )
        if static_components.get(component_name) != actual_sha256:
            raise RuntimeError(
                f"Execution evidence changed after launch: {evidence_name}"
            )
    hardware_relative_path = evidence_paths.get("runtime_hardware")
    if not isinstance(hardware_relative_path, str) or not hardware_relative_path:
        raise RuntimeError("Execution runtime hardware evidence path is invalid")
    hardware_path = (attempt_root / hardware_relative_path).resolve()
    try:
        hardware_path.relative_to(attempt_root)
    except ValueError as error:
        raise RuntimeError("Runtime hardware evidence escapes attempt root") from error
    if not hardware_path.is_file():
        raise RuntimeError(f"Runtime hardware evidence is missing: {hardware_path}")
    runtime_hardware = read_json(hardware_path)
    if (
        not isinstance(runtime_hardware, dict)
        or runtime_hardware.get("schema") != RUNTIME_HARDWARE_SCHEMA
        or runtime_hardware.get("job_id") != job_id
        or runtime_hardware.get("execution_attempt_id") != attempt_id
        or runtime_hardware.get("capture_mode") != "live_pre_execution"
        or runtime_hardware.get("all_passed") is not True
        or runtime_hardware.get("calibration_hardware_eligible") is not True
    ):
        raise RuntimeError("Runtime hardware evidence is not execution eligible")
    devices = runtime_hardware.get("devices")
    if not isinstance(devices, list) or len(devices) != expected_ranks:
        raise RuntimeError("Runtime hardware assigned device set is incomplete")

    references: list[dict[str, Any]] = []
    common_inventory_sha256: str | None = None
    seen_local_ranks: set[int] = set()
    for rank in range(expected_ranks):
        path = runtime_model_manifest_path(result_dir, attempt_id, rank)
        if not path.is_file():
            raise RuntimeError(f"Runtime model manifest is missing for rank {rank}: {path}")
        manifest = validate_runtime_model_manifest(
            read_json(path),
            expected_job_id=job_id,
            expected_execution_attempt_id=attempt_id,
            expected_rank=rank,
            expected_world_size=expected_ranks,
            expected_training_mode=str(job["train_type"]),
            allow_unavailable_device=False,
        )
        local_rank = manifest["local_rank"]
        if local_rank in seen_local_ranks:
            raise RuntimeError(
                f"Runtime model manifests repeat local rank {local_rank}"
            )
        seen_local_ranks.add(local_rank)
        if local_rank != rank:
            raise RuntimeError(
                "Single-node runtime model manifest rank/local-rank mismatch: "
                f"rank={rank}, local_rank={local_rank}"
            )
        parent_device = devices[local_rank]
        attestation = manifest["device_attestation"]
        expected_capability = str(parent_device.get("compute_capability") or "")
        observed_capability = (
            f"{attestation['compute_capability']['major']}."
            f"{attestation['compute_capability']['minor']}"
        )
        memory_tolerance = max(
            1024**3,
            int(int(parent_device.get("memory_total_bytes") or 0) * 0.02),
        )
        device_checks = {
            "visible_device_index_matches_local_rank": attestation[
                "visible_device_index"
            ]
            == local_rank,
            "uuid_exact": bool(attestation.get("uuid"))
            and attestation.get("uuid") == parent_device.get("uuid"),
            "name_exact": attestation.get("name") == parent_device.get("name"),
            "memory_matches": abs(
                int(attestation.get("total_memory_bytes") or 0)
                - int(parent_device.get("memory_total_bytes") or 0)
            )
            <= memory_tolerance,
            "compute_capability_exact": observed_capability
            == expected_capability,
        }
        if not all(device_checks.values()):
            raise RuntimeError(
                "Rank CUDA device attestation does not match pre-execution "
                f"hardware evidence for rank {rank}: {device_checks}"
            )
        inventory_sha256 = manifest["inventory_sha256"]
        if common_inventory_sha256 is None:
            common_inventory_sha256 = inventory_sha256
        elif common_inventory_sha256 != inventory_sha256:
            raise RuntimeError("Runtime model inventories differ across ranks")
        references.append(
            {
                "rank": rank,
                "path": str(path.relative_to(result_dir)),
                "file_sha256": sha256_file(path),
                "inventory_sha256": inventory_sha256,
                "local_rank": local_rank,
                "device_attestation_sha256": manifest[
                    "device_attestation_sha256"
                ],
                "parent_physical_gpu_index": parent_device["physical_index"],
                "parent_runtime_hardware_sha256": sha256_file(hardware_path),
                "device_cross_attestation": device_checks,
            }
        )

    if seen_local_ranks != set(range(expected_ranks)):
        raise RuntimeError("Runtime model manifest local-rank set is incomplete")
    if common_inventory_sha256 is None:  # pragma: no cover - guarded above
        raise RuntimeError("No runtime model inventory was collected")
    components = dict(static_components)
    components.update(
        {
            "execution_inputs_sha256": sha256_file(execution_inputs_path),
            "runtime_model_inventory_sha256": common_inventory_sha256,
            "runtime_model_manifest_set_sha256": sha256_json(references),
        }
    )
    if set(components) != set(REQUIRED_EXECUTION_COMPONENTS):
        raise RuntimeError("Final execution fingerprint components are incomplete")
    return {
        "schema": EXECUTION_FINGERPRINT_SCHEMA,
        "job_id": job_id,
        "execution_attempt_id": attempt_id,
        "job_payload_sha256": job_payload_sha256,
        "job_snapshot": job_snapshot,
        "authorization": authorization,
        "authorization_mode": authorization_mode,
        "calibration_eligible": authorization_mode == "approved"
        and outcome_is_calibratable,
        "approval_design_sha256": execution_inputs.get("approval_design_sha256"),
        "execution_inputs_path": str(execution_inputs_path.relative_to(result_dir)),
        "runtime_model_manifests": references,
        "outcome": copy.deepcopy(terminal_classification),
        "outcome_sha256": sha256_json(terminal_classification),
        "components": components,
    }


APPROVAL_FILE = CONFIG_DIR / "APPROVED_TO_RUN.json"
OOM_PATTERNS = (
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    "CUDNN_STATUS_ALLOC_FAILED",
    "CUDA error: out of memory",
)
CUDA_OOM_REGEXES = (
    (OOM_PATTERNS[0], r"(?<!no )\bCUDA out of memory\b"),
    (OOM_PATTERNS[1], r"\btorch\.cuda\.OutOfMemoryError\b"),
    (OOM_PATTERNS[2], r"\bCUDNN_STATUS_ALLOC_FAILED\b"),
    (OOM_PATTERNS[3], r"\bCUDA error:\s*out of memory\b"),
)
NVIDIA_SMI_FIELDS = (
    "timestamp",
    "index",
    "memory.used",
    "utilization.gpu",
    "power.draw",
    "clocks.sm",
    "temperature.gpu",
    "fan.speed",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.sw_power_cap",
)


def live_runtime_patch() -> dict[str, Any]:
    """Re-run the installed/build patch audit without creating artifacts."""

    # Delayed to avoid the import cycle: freeze_lora_zero3_fix imports
    # live_runtime_identity from this module when it constructs candidates.
    from freeze_lora_zero3_fix import validate_patch

    return validate_patch()


def verify_approval(
    job: dict[str, Any] | None = None,
    *,
    queue_path: Path | None = None,
    queue_rows: list[dict[str, Any]] | None = None,
    acquire_lock: bool = True,
    project_root: Path | None = None,
    approval_file: Path | None = None,
    approval_design_path: Path | None = None,
    config_dir: Path | None = None,
    runtime_dir: Path | None = None,
    current_runtime_identity: dict[str, Any] | None = None,
    current_runtime_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = (project_root or ROOT).resolve()
    runtime_root = (runtime_dir or RUNTIME_DIR).resolve()
    approval_path = (approval_file or APPROVAL_FILE).resolve()
    design_path = (
        approval_design_path or runtime_root / "approval_design.json"
    ).resolve()
    configuration_root = (config_dir or CONFIG_DIR).resolve()
    if acquire_lock:
        with execution_lock(runtime_root, exclusive=False):
            return verify_approval(
                job,
                queue_path=queue_path,
                queue_rows=queue_rows,
                acquire_lock=False,
                project_root=root,
                approval_file=approval_path,
                approval_design_path=design_path,
                config_dir=configuration_root,
                runtime_dir=runtime_root,
                current_runtime_identity=current_runtime_identity,
                current_runtime_patch=current_runtime_patch,
            )
    if not approval_path.is_file():
        raise PermissionError(
            f"Training is locked. Missing approval file: {approval_path}"
        )
    approval = read_json(approval_path)
    if approval.get("approved") is not True:
        raise PermissionError("Approval file does not contain approved=true")
    if not design_path.is_file() or approval.get("design_sha256") != sha256_file(
        design_path
    ):
        raise PermissionError("Approval does not match the current frozen design")
    design = read_json(design_path)
    reject_retired_approval_plan(
        design,
        operation="authorize a run or scheduler launch",
    )
    manifest = design.get("file_sha256") or {}
    manifest_check = verify_file_manifest(root, manifest)
    if (
        not isinstance(manifest, dict)
        or not manifest
        or not manifest_check["all_passed"]
    ):
        raise PermissionError(f"Frozen design is stale: {manifest_check}")
    approval_ids = strict_id_list(approval.get("allowed_job_ids"))
    design_ids = strict_id_list(design.get("allowed_job_ids"))
    if approval_ids is None or design_ids is None or approval_ids != design_ids:
        raise PermissionError(
            "Approval/design allowed_job_ids must be non-empty, unique and exactly ordered"
        )
    approval_order = strict_id_list(approval.get("execution_order"))
    design_order = strict_id_list(design.get("execution_order"))
    if approval_order is None or design_order is None or approval_order != design_order:
        raise PermissionError("Approval/design execution_order is absent or mismatched")
    queue_check = validate_queue_binding(
        design,
        root,
        queue_path=queue_path,
        queue_rows=queue_rows,
    )
    if queue_check.get("all_passed") is not True:
        raise PermissionError(f"Approval gate rejected the bound queue: {queue_check}")
    queue_binding = design.get("queue_binding") or {}
    if approval.get("queue_binding_sha256") != sha256_json(queue_binding):
        raise PermissionError("Approval does not bind the frozen queue metadata")

    expected_runtime_identity = design.get("runtime_identity") or {}
    expected_runtime_fingerprint = design.get("runtime_fingerprint_sha256")
    live_identity = (
        live_runtime_identity()
        if current_runtime_identity is None
        else current_runtime_identity
    )
    if (
        not isinstance(expected_runtime_identity, dict)
        or not expected_runtime_identity
        or expected_runtime_fingerprint != sha256_json(expected_runtime_identity)
        or approval.get("runtime_fingerprint_sha256") != expected_runtime_fingerprint
        or live_identity != expected_runtime_identity
        or sha256_json(live_identity) != expected_runtime_fingerprint
    ):
        raise PermissionError(
            "Approval gate rejected stale live runtime identity/fingerprint"
        )

    expected_patch = design.get("runtime_patch") or {}
    live_patch = (
        live_runtime_patch() if current_runtime_patch is None else current_runtime_patch
    )
    if (
        not isinstance(expected_patch, dict)
        or expected_patch.get("all_passed") is not True
        or live_patch.get("all_passed") is not True
        or live_patch != expected_patch
        or approval.get("runtime_patch_sha256") != sha256_json(expected_patch)
    ):
        raise PermissionError(
            "Approval gate rejected stale live runtime patch evidence"
        )

    provenance_check = validate_provenance_binding(design, root)
    provenance_binding = design.get("provenance_binding") or {}
    if provenance_check.get("all_passed") is not True or approval.get(
        "provenance_sha256"
    ) != provenance_binding.get("sha256"):
        raise PermissionError(
            f"Approval gate rejected stale provenance evidence: {provenance_check}"
        )
    experiment = read_json(configuration_root / "experiment.json")
    training_scope = experiment["training_scope"]
    resource_scope = approval.get("resource_scope") or {}
    approved_gpu_ids = [int(value) for value in resource_scope.get("gpu_ids") or ()]
    configured_gpu_ids = [int(value) for value in training_scope["gpu_ids"]]
    if approved_gpu_ids != configured_gpu_ids:
        raise PermissionError(
            f"Approval GPU scope {approved_gpu_ids} does not match configured GPU pool {configured_gpu_ids}"
        )
    if int(resource_scope.get("max_gpu_count", 0)) != int(
        training_scope["max_gpu_count"]
    ):
        raise PermissionError(
            "Approval max_gpu_count does not match the configured training scope"
        )
    if resource_scope.get("allow_gpu_ids_outside_pool") is not False:
        raise PermissionError(
            "Approval must explicitly forbid GPU IDs outside the configured pool"
        )
    configured_parallelism = experiment["measurement"]["performance_parallelism"]
    if resource_scope.get("performance_parallelism") != configured_parallelism:
        raise PermissionError(
            "Approval performance_parallelism does not match the configured scheduling policy"
        )
    if job is not None:
        job_id = job.get("job_id")
        if job_id not in set(approval_ids):
            raise PermissionError(f"Job {job_id} is outside the approval scope")
        expected_payload = queue_check["job_payload_sha256"].get(job_id)
        actual_payload = canonical_job_sha256(job)
        if not expected_payload or actual_payload != expected_payload:
            raise PermissionError(
                f"Job {job_id} canonical payload does not match the approved queue"
            )
    return {
        "approval": approval,
        "approval_path": str(approval_path),
        "approval_sha256": sha256_file(approval_path),
        "approval_design_path": str(design_path),
        "design_sha256": sha256_file(design_path),
        "runtime_identity": live_identity,
        "runtime_fingerprint_sha256": expected_runtime_fingerprint,
        "runtime_patch": live_patch,
        "manifest": manifest_check,
        "queue": queue_check,
        "provenance": provenance_check,
    }


def validate_gpu_assignment(
    job: dict[str, Any], gpu_ids: list[int], experiment: dict[str, Any]
) -> None:
    """Reject jobs or masks outside the explicitly approved shared-node GPU pool."""

    scope = experiment["training_scope"]
    allowed_ids = {int(value) for value in scope["gpu_ids"]}
    allowed_counts = {int(value) for value in scope["gpu_counts"]}
    max_gpu_count = int(scope["max_gpu_count"])
    exclusive_ids = {int(value) for value in scope["exclusive_node_gpu_ids"]}
    gpu_count = int(job["gpu_count"])

    if not exclusive_ids <= allowed_ids:
        raise PermissionError(
            f"Exclusive performance scope {sorted(exclusive_ids)} must stay inside GPU pool {sorted(allowed_ids)}"
        )
    if gpu_count not in allowed_counts or gpu_count > max_gpu_count:
        raise PermissionError(
            f"Job gpu_count={gpu_count} is outside approved counts {sorted(allowed_counts)} "
            f"with max_gpu_count={max_gpu_count}"
        )
    if len(gpu_ids) != gpu_count or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"GPU mask {gpu_ids} does not match gpu_count={gpu_count}")
    unexpected = sorted(set(gpu_ids) - allowed_ids)
    if unexpected:
        raise PermissionError(
            f"GPU mask {gpu_ids} contains IDs outside approved pool {sorted(allowed_ids)}"
        )


def idle_check_gpu_ids(
    job: dict[str, Any], gpu_ids: list[int], experiment: dict[str, Any]
) -> list[int]:
    """Return the physical GPUs that must be idle immediately before launch."""

    parallelism = experiment["measurement"]["performance_parallelism"]
    if parallelism not in {"disjoint_gpu_masks", "exclusive_pool"}:
        raise ValueError(f"Unsupported performance_parallelism={parallelism!r}")
    if (
        job.get("requires_external_node_idle") is True
        and parallelism == "exclusive_pool"
    ):
        return [
            int(value)
            for value in experiment["training_scope"]["exclusive_node_gpu_ids"]
        ]
    return gpu_ids


def validate_scoped_smoke(job: dict[str, Any]) -> None:
    if job.get("kind") != "smoke":
        raise PermissionError("The scoped smoke path accepts only kind=smoke")
    if job.get("model_id") != "qwen3_1p7b" or job.get("dataset_id") != "short_512":
        raise PermissionError(
            "Smoke tests are restricted to Qwen3-1.7B and short_512 in this phase"
        )
    if int(job.get("mbs", 0)) != 1:
        raise PermissionError("Smoke tests are restricted to physical MBS=1")
    if (
        int(job.get("warmup_steps", 0)) != 0
        or not 1 <= int(job.get("measure_steps", 0)) <= 2
    ):
        raise PermissionError(
            "Smoke tests are restricted to one or two optimizer steps with no warmup"
        )


def gradient_accumulation(job: dict[str, Any]) -> int:
    if job.get("packing"):
        ga = int(job["gradient_accumulation_steps"])
        if int(job["mbs"]) != 1:
            raise ValueError("Neat packing requires physical MBS=1")
        return ga
    denominator = int(job["gpu_count"]) * int(job["mbs"])
    target = int(job["target_gbs"])
    if target % denominator != 0:
        raise ValueError(f"GBS {target} is not divisible by DP×MBS={denominator}")
    return target // denominator


EXECUTABLE_GPU_COUNTS = frozenset({1, 2, 4, 5, 6, 7, 8})


def validate_job(job: dict[str, Any]) -> None:
    gpu_count = int(job["gpu_count"])
    zero = job["zero"]
    if gpu_count not in EXECUTABLE_GPU_COUNTS:
        raise ValueError(f"Unsupported configured GPU count: {gpu_count}")
    if gpu_count == 1 and zero != "none":
        raise ValueError("Single-card jobs must not use DeepSpeed")
    if gpu_count > 1 and zero not in {"zero2", "zero3"}:
        raise ValueError("Multi-card jobs support only ZeRO-2 or ZeRO-3")
    if job.get("packing") and int(job["mbs"]) != 1:
        raise ValueError("Neat packing requires physical MBS=1")
    gradient_accumulation(job)


def resolve_dataset_dir(job: dict[str, Any]) -> Path:
    """Resolve and validate an optional campaign-local dataset registry.

    Campaign-local registries keep new calibration datasets from mutating the
    shared ``data/dataset_info.json`` used by already frozen campaigns.  The
    registry content is bound into the immutable job payload by its SHA256.
    """

    dataset_dir = Path(job.get("dataset_dir") or DATA_DIR).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir is missing: {dataset_dir}")
    registry_path = dataset_dir / "dataset_info.json"
    if not registry_path.is_file():
        raise FileNotFoundError(f"dataset registry is missing: {registry_path}")

    expected_sha256 = job.get("dataset_registry_sha256")
    if "dataset_dir" in job:
        if not isinstance(expected_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", expected_sha256
        ) is None:
            raise ValueError(
                "campaign-local dataset_dir requires dataset_registry_sha256"
            )
        actual_sha256 = sha256_file(registry_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"dataset registry drifted: {actual_sha256} != {expected_sha256}"
            )

    registry = read_json(registry_path)
    dataset_id = str(job["dataset_id"])
    entry = registry.get(dataset_id)
    if not isinstance(entry, dict):
        raise ValueError(
            f"dataset_id={dataset_id!r} is absent from {registry_path}"
        )
    file_name = entry.get("file_name")
    if not isinstance(file_name, str) or not file_name:
        raise ValueError(f"dataset_id={dataset_id!r} has no file_name")
    registered_path = Path(file_name)
    if not registered_path.is_absolute():
        registered_path = dataset_dir / registered_path
    registered_path = registered_path.resolve()
    if not registered_path.is_file():
        raise FileNotFoundError(
            f"registered dataset is missing for {dataset_id!r}: {registered_path}"
        )
    if job.get("data_path"):
        declared_path = Path(str(job["data_path"])).resolve()
        if registered_path != declared_path:
            raise ValueError(
                f"dataset registry path mismatch for {dataset_id!r}: "
                f"{registered_path} != {declared_path}"
            )
        expected_data_sha256 = job.get("data_sha256")
        if expected_data_sha256 and sha256_file(registered_path) != expected_data_sha256:
            raise ValueError(f"dataset content drifted for {dataset_id!r}")
    return dataset_dir


def apply_multimodal_runtime_options(
    config: dict[str, Any], job: dict[str, Any]
) -> None:
    """Copy validated image/video processor controls into a runtime config."""

    for optional_bool in (
        "freeze_vision_tower",
        "freeze_multi_modal_projector",
        "freeze_language_model",
    ):
        if optional_bool in job:
            value = job[optional_bool]
            if not isinstance(value, bool):
                raise ValueError(f"{optional_bool} must be boolean")
            config[optional_bool] = value
    for optional_int in (
        "image_min_pixels",
        "image_max_pixels",
        "video_min_pixels",
        "video_max_pixels",
        "video_maxlen",
    ):
        if optional_int in job:
            value = int(job[optional_int])
            if value <= 0:
                raise ValueError(f"{optional_int} must be positive")
            config[optional_int] = value
    if "video_fps" in job:
        value = float(job["video_fps"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("video_fps must be finite and positive")
        config["video_fps"] = value
    for modality in ("image", "video"):
        minimum = f"{modality}_min_pixels"
        maximum = f"{modality}_max_pixels"
        if (
            minimum in config
            and maximum in config
            and config[minimum] > config[maximum]
        ):
            raise ValueError(f"{minimum} cannot exceed {maximum}")


def render_config(job: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    validate_job(job)
    experiment = read_json(CONFIG_DIR / "experiment.json")
    runtime = experiment["fixed_runtime"]
    measurement = experiment["measurement"]
    kind = job["kind"]
    if kind in {"memory_probe", "packing_memory_probe"}:
        max_steps = int(measurement["memory_probe_max_steps"])
        warmup_steps = 0
    else:
        warmup_steps = int(
            job.get("warmup_steps", measurement["throughput_warmup_steps"])
        )
        measure_steps = int(
            job.get("measure_steps", measurement["throughput_measure_steps"])
        )
        max_steps = warmup_steps + measure_steps

    gc_enabled = bool(job["gc"])
    runtime_model_path = prepare_model_tokenizer_view(
        Path(job["model_path"]),
        Path(job.get("tokenizer_path") or job["model_path"]),
        output_dir / "model_with_qwen3_tokenizer",
    )
    job["runtime_model_path"] = str(runtime_model_path)
    config: dict[str, Any] = {
        "model_name_or_path": str(runtime_model_path),
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": job["train_type"],
        "dataset": job["dataset_id"],
        "dataset_dir": str(resolve_dataset_dir(job)),
        "template": job["template"],
        "cutoff_len": int(job["cutoff_len"]),
        "max_samples": int(job.get("max_samples", 1000)),
        "preprocessing_num_workers": int(runtime["preprocessing_num_workers"]),
        "dataloader_num_workers": int(runtime["dataloader_num_workers"]),
        "overwrite_cache": False,
        "packing": bool(job.get("packing", False)),
        "neat_packing": bool(job.get("packing", False)),
        "output_dir": str(output_dir / "trainer_output"),
        "overwrite_output_dir": True,
        "logging_steps": 1,
        "logging_strategy": "steps",
        "save_strategy": "no",
        "eval_strategy": "no",
        "report_to": "none",
        "plot_loss": False,
        "disable_tqdm": True,
        "per_device_train_batch_size": int(job["mbs"]),
        "gradient_accumulation_steps": gradient_accumulation(job),
        "learning_rate": 1.0e-5,
        "max_steps": max_steps,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "flash_attn": runtime["flash_attn"],
        "enable_liger_kernel": bool(
            job.get("enable_liger_kernel", runtime["enable_liger_kernel"])
        ),
        "optim": runtime["optimizer"],
        "torch_compile": bool(runtime["torch_compile"]),
        "gradient_checkpointing": gc_enabled,
        "disable_gradient_checkpointing": not gc_enabled,
        "use_reentrant_gc": True,
        "include_num_input_tokens_seen": "all",
        "seed": int(job.get("seed", runtime["seed"])),
        "data_seed": int(job.get("data_seed", runtime["data_seed"])),
        "ddp_timeout": int(runtime["ddp_timeout"]),
    }
    apply_multimodal_runtime_options(config, job)
    if job["train_type"] == "lora":
        lora = read_json(CONFIG_DIR / "models.json")["fixed_lora"]
        config.update(
            {
                "lora_rank": int(lora["rank"]),
                "lora_alpha": int(lora["alpha"]),
                "lora_dropout": float(lora["dropout"]),
                "lora_target": lora["target"],
            }
        )
    if job["zero"] != "none":
        config["deepspeed"] = str(
            CONFIG_DIR / "deepspeed" / f"ds_z{job['zero'][-1]}.json"
        )
    job["warmup_steps"] = warmup_steps
    job["max_steps"] = max_steps
    return config


def monitor_nvidia_smi(
    stop: threading.Event,
    path: Path,
    physical_gpu_ids: set[int],
    interval: float,
    health: dict[str, Any] | None = None,
) -> None:
    health = health if health is not None else {}
    health.update(
        {
            "attempts": 0,
            "failures": 0,
            "malformed_rows": 0,
            "samples_written": 0,
            "last_error": None,
        }
    )
    try:
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(NVIDIA_SMI_FIELDS)
            while not stop.is_set():
                command = [
                    "nvidia-smi",
                    "--query-gpu=" + ",".join(NVIDIA_SMI_FIELDS),
                    "--format=csv,noheader,nounits",
                ]
                health["attempts"] += 1
                try:
                    result = subprocess.run(
                        command,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=5,
                    )
                except Exception as error:
                    health["failures"] += 1
                    health["last_error"] = repr(error)
                    stop.wait(interval)
                    continue
                if result.returncode != 0:
                    health["failures"] += 1
                    health["last_error"] = (
                        result.stderr.strip()
                        or f"nvidia-smi returned exit code {result.returncode}"
                    )
                    stop.wait(interval)
                    continue
                for line in result.stdout.splitlines():
                    values = [value.strip() for value in line.split(",")]
                    try:
                        gpu_index = int(values[1])
                    except (IndexError, ValueError):
                        health["malformed_rows"] += 1
                        continue
                    if len(values) != len(NVIDIA_SMI_FIELDS):
                        health["malformed_rows"] += 1
                        continue
                    if gpu_index in physical_gpu_ids:
                        writer.writerow(values)
                        health["samples_written"] += 1
                output.flush()
                stop.wait(interval)
    except Exception as error:
        health["failures"] += 1
        health["last_error"] = repr(error)


def optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def active_flag(value: str | None) -> bool:
    return str(value or "").strip().lower() == "active"


def flag_counts(rows: list[dict[str, str]], field: str) -> tuple[int, int]:
    values = [str(row.get(field) or "").strip().lower() for row in rows]
    supported = [value for value in values if value in {"active", "not active"}]
    return len(supported), sum(value == "active" for value in supported)


def summarize_nvidia_smi(path: Path, physical_gpu_ids: set[int]) -> dict[str, Any]:
    """Summarize thermal and clock observations without changing job validity."""

    rows_by_gpu: dict[int, list[dict[str, str]]] = {
        int(gpu_id): [] for gpu_id in sorted(physical_gpu_ids)
    }
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                try:
                    gpu_index = int(row.get("index") or "")
                except ValueError:
                    continue
                if gpu_index in rows_by_gpu:
                    rows_by_gpu[gpu_index].append(row)

    gpu_summaries = []
    for gpu_index, rows in rows_by_gpu.items():
        busy_rows = [
            row
            for row in rows
            if (optional_float(row.get("utilization.gpu")) or 0.0) >= 90.0
        ]
        temperatures = [
            value
            for row in rows
            if (value := optional_float(row.get("temperature.gpu"))) is not None
        ]
        fan_speeds = [
            value
            for row in rows
            if (value := optional_float(row.get("fan.speed"))) is not None
        ]
        clocks = [
            value
            for row in rows
            if (value := optional_float(row.get("clocks.sm"))) is not None
        ]
        busy_clocks = [
            value
            for row in busy_rows
            if (value := optional_float(row.get("clocks.sm"))) is not None
        ]
        sw_thermal_supported, sw_thermal_samples = flag_counts(
            rows,
            "clocks_event_reasons.sw_thermal_slowdown",
        )
        sw_thermal_busy_supported, sw_thermal_busy_samples = flag_counts(
            busy_rows,
            "clocks_event_reasons.sw_thermal_slowdown",
        )
        hw_thermal_supported, hw_thermal_samples = flag_counts(
            rows,
            "clocks_event_reasons.hw_thermal_slowdown",
        )
        sw_power_cap_supported, sw_power_cap_samples = flag_counts(
            rows,
            "clocks_event_reasons.sw_power_cap",
        )
        gpu_summaries.append(
            {
                "gpu_index": gpu_index,
                "samples": len(rows),
                "busy_samples": len(busy_rows),
                "max_temperature_c": max(temperatures) if temperatures else None,
                "max_fan_percent": max(fan_speeds) if fan_speeds else None,
                "min_sm_clock_mhz": min(clocks) if clocks else None,
                "max_sm_clock_mhz": max(clocks) if clocks else None,
                "min_busy_sm_clock_mhz": min(busy_clocks) if busy_clocks else None,
                "median_busy_sm_clock_mhz": (
                    statistics.median(busy_clocks) if busy_clocks else None
                ),
                "sw_thermal_slowdown_supported_samples": sw_thermal_supported,
                "sw_thermal_slowdown_samples": sw_thermal_samples,
                "sw_thermal_slowdown_busy_supported_samples": sw_thermal_busy_supported,
                "sw_thermal_slowdown_busy_samples": sw_thermal_busy_samples,
                "sw_thermal_slowdown_busy_fraction": (
                    sw_thermal_busy_samples / sw_thermal_busy_supported
                    if sw_thermal_busy_supported
                    else None
                ),
                "hw_thermal_slowdown_supported_samples": hw_thermal_supported,
                "hw_thermal_slowdown_samples": hw_thermal_samples,
                "sw_power_cap_supported_samples": sw_power_cap_supported,
                "sw_power_cap_samples": sw_power_cap_samples,
            }
        )

    return {
        "schema_version": 1,
        "policy": "record_only",
        "affects_job_classification": False,
        "source": str(path),
        "fields": list(NVIDIA_SMI_FIELDS),
        "monitor_data_available": any(row["samples"] for row in gpu_summaries),
        "any_sw_thermal_slowdown": any(
            row["sw_thermal_slowdown_samples"] for row in gpu_summaries
        ),
        "any_hw_thermal_slowdown": any(
            row["hw_thermal_slowdown_samples"] for row in gpu_summaries
        ),
        "gpus": gpu_summaries,
    }


def classify_execution(
    return_code: int,
    log_path: Path,
    expected_ranks: int,
    *,
    job_id: str | None = None,
    execution_attempt_id: str | None = None,
) -> dict[str, Any]:
    text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.exists()
        else ""
    )
    matched_oom_patterns = [
        label
        for label, pattern in CUDA_OOM_REGEXES
        if re.search(pattern, text, flags=re.IGNORECASE) is not None
    ]
    expected_paths = [
        log_path.parent / "metrics" / f"summary.rank{rank}.json"
        for rank in range(expected_ranks)
    ]
    discovered_paths = sorted(
        (log_path.parent / "metrics").glob("summary.rank*.json")
    )
    summary_checks = []
    summary_files = []
    for rank, path in enumerate(expected_paths):
        checks = {"exists": path.is_file()}
        summary: dict[str, Any] = {}
        if path.is_file():
            try:
                value = read_json(path)
                summary = value if isinstance(value, dict) else {}
            except (OSError, ValueError, json.JSONDecodeError):
                summary = {}
            checks.update(
                {
                    "object": bool(summary),
                    "job_id": job_id is None or summary.get("job_id") == job_id,
                    "execution_attempt_id": execution_attempt_id is None
                    or summary.get("execution_attempt_id")
                    == execution_attempt_id,
                    "rank": summary.get("rank") == rank
                    if job_id is not None or execution_attempt_id is not None
                    else True,
                    "world_size": summary.get("world_size") == expected_ranks
                    if job_id is not None or execution_attempt_id is not None
                    else True,
                    "measured_steps_positive": type(summary.get("measured_steps"))
                    is int
                    and int(summary["measured_steps"]) > 0,
                    "measured_seconds_positive": isinstance(
                        summary.get("measured_seconds"), (int, float)
                    )
                    and not isinstance(summary.get("measured_seconds"), bool)
                    and float(summary["measured_seconds"]) > 0,
                    "measured_tokens_positive": isinstance(
                        summary.get("measured_totals"), dict
                    )
                    and int(
                        (summary.get("measured_totals") or {}).get(
                            "effective_tokens", 0
                        )
                    )
                    > 0,
                }
            )
        summary_checks.append(
            {
                "rank": rank,
                "path": str(path.relative_to(log_path.parent)),
                "checks": checks,
                "all_passed": all(checks.values()),
            }
        )
        if path.is_file():
            summary_files.append(
                {
                    "rank": rank,
                    "path": str(path.relative_to(log_path.parent)),
                    "file_sha256": sha256_file(path),
                }
            )
    summary_set_exact = discovered_paths == expected_paths
    summaries_valid = summary_set_exact and all(
        row["all_passed"] for row in summary_checks
    )
    if matched_oom_patterns:
        classification = "oom"
    elif return_code != 0:
        classification = "failed"
    elif summaries_valid:
        classification = "success"
    else:
        classification = "incomplete_metrics"
    evidence = {
        "schema": "sft_terminal_classification/v1",
        "classification": classification,
        "return_code": int(return_code),
        "cuda_oom_confirmed": bool(matched_oom_patterns),
        "matched_cuda_oom_patterns": matched_oom_patterns,
        "summaries_complete": summaries_valid,
        "summary_files": summary_files,
        "job_id": job_id,
        "execution_attempt_id": execution_attempt_id,
        "policy": "cuda_allocator_and_positive_measurement_v3",
        "host_oom_and_oomkill_are_not_cuda_oom": True,
        "log_path": str(log_path.name),
        "log_sha256": sha256_file(log_path) if log_path.is_file() else None,
        "expected_ranks": expected_ranks,
        "summary_set_exact": summary_set_exact,
        "summary_checks": summary_checks,
    }
    return {"classification": classification, "evidence": evidence}


def classify(return_code: int, log_path: Path, expected_ranks: int) -> str:
    """Compatibility wrapper for legacy callers without attempt identity."""

    text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.exists()
        else ""
    )
    if any(
        re.search(pattern, text, flags=re.IGNORECASE) is not None
        for _, pattern in CUDA_OOM_REGEXES
    ):
        return "oom"
    if return_code != 0:
        return "failed"
    summaries = list((log_path.parent / "metrics").glob("summary.rank*.json"))
    return "success" if len(summaries) == expected_ranks else "incomplete_metrics"


def main(*, _execution_gate_held: bool = False) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file", type=Path, required=True)
    parser.add_argument(
        "--gpu-mask", required=True, help="Physical GPU IDs, for example 0,1"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument(
        "--execute-smoke",
        action="store_true",
        help="Run a tightly bounded 0.6B/short-data smoke test",
    )
    args = parser.parse_args()

    job = read_json(args.job_file)
    experiment = read_json(CONFIG_DIR / "experiment.json")
    if args.execute_smoke:
        validate_scoped_smoke(job)
    if re.fullmatch(r"[0-9]+(?:,[0-9]+)*", args.gpu_mask) is None:
        raise ValueError(
            "--gpu-mask must be a canonical comma-separated list of GPU indices"
        )
    gpu_ids = [int(value) for value in args.gpu_mask.split(",")]
    gpu_mask = ",".join(str(value) for value in gpu_ids)
    validate_gpu_assignment(job, gpu_ids, experiment)
    if (args.execute or args.execute_smoke) and not _execution_gate_held:
        # Re-enter under a context-managed process-lifetime shared lock.  The
        # second pass is still before any result/runtime artifact write, and
        # the context releases deterministically on every exception/SystemExit.
        with execution_lock(RUNTIME_DIR, exclusive=False):
            return main(_execution_gate_held=True)
    authorized_job = copy.deepcopy(job)
    approval_check = (
        verify_approval(authorized_job, acquire_lock=False) if args.execute else None
    )
    authorization_mode = (
        "approved" if args.execute else "smoke" if args.execute_smoke else "render"
    )
    execution_attempt_id = sha256_json(
        {
            "job_id": str(authorized_job["job_id"]),
            "time_ns": time.time_ns(),
            "pid": os.getpid(),
        }
    )[:20]
    expected_prefix = Path(experiment["fixed_runtime"]["python"]).parent.parent
    if Path(sys.prefix) != expected_prefix:
        raise RuntimeError(
            "run_job.py must use the configured training interpreter: "
            f"{experiment['fixed_runtime']['python']} "
            f"(running prefix={sys.prefix!r}, expected={str(expected_prefix)!r})"
        )
    declared_hardware = read_json(CONFIG_DIR / "hardware.json")
    if args.render_only:
        topology_text = "Live topology not captured for render-only mode.\n"
        runtime_hardware = render_only_hardware_manifest(
            job=authorized_job,
            execution_attempt_id=execution_attempt_id,
            gpu_ids=gpu_ids,
        )
        runtime_hardware["topology_sha256"] = _sha256_text(topology_text)
    else:
        runtime_hardware, topology_text = capture_runtime_hardware_manifest(
            job=authorized_job,
            execution_attempt_id=execution_attempt_id,
            gpu_ids=gpu_ids,
            experiment=experiment,
            declared_hardware=declared_hardware,
        )

    job_result_dir = RESULTS_DIR / str(authorized_job["job_id"])
    attempt_dir = (
        job_result_dir / "attempts" / execution_attempt_id
    )
    attempt_dir.mkdir(parents=True, exist_ok=False)
    runtime_config = attempt_dir / "runtime_config.yaml"
    runtime_metadata = attempt_dir / "job_metadata.json"
    runtime_identity_path = attempt_dir / "runtime_identity.json"
    runtime_hardware_path = attempt_dir / "runtime_hardware.json"
    runtime_mechanism_path = attempt_dir / "runtime_mechanism.json"
    topology_path = attempt_dir / "nvidia_topology.txt"
    execution_inputs_path = attempt_dir / "execution_inputs.json"
    execution_manifest_path = attempt_dir / "execution_fingerprint.json"
    rendered_path = attempt_dir / "rendered_run.json"
    status_path = attempt_dir / "status.json"
    metrics_dir = attempt_dir / "metrics"
    log_path = attempt_dir / "train.log"
    nvidia_smi_path = attempt_dir / "nvidia_smi.csv"
    thermal_summary_path = attempt_dir / "thermal_summary.json"

    config = render_config(job, attempt_dir)
    input_snapshots = snapshot_execution_inputs(
        attempt_root=attempt_dir,
        config=config,
        job=job,
    )
    if read_json(input_snapshots["declared_hardware"]) != declared_hardware:
        raise RuntimeError(
            "Declared hardware changed between live attestation and attempt snapshot"
        )
    runtime_config.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    _atomic_write_json(
        runtime_metadata,
        {
            **authorized_job,
            "_execution_attempt_id": execution_attempt_id,
        },
    )
    topology_path.write_text(topology_text, encoding="utf-8")
    if sha256_file(topology_path) != runtime_hardware["topology_sha256"]:
        raise RuntimeError("Runtime topology snapshot hash mismatch")
    _atomic_write_json(runtime_hardware_path, runtime_hardware)
    runtime_identity = live_runtime_identity()
    if approval_check is not None and (
        runtime_identity != approval_check["runtime_identity"]
        or sha256_json(runtime_identity)
        != approval_check["runtime_fingerprint_sha256"]
    ):
        raise PermissionError(
            "Live runtime identity changed after the approval check and before "
            "attempt evidence capture"
        )
    runtime_fingerprint = sha256_json(runtime_identity)
    _atomic_write_json(runtime_identity_path, runtime_identity)

    fixed = experiment["fixed_runtime"]
    entry = ROOT / "scripts" / "train_entry.py"
    worker_args = [
        str(entry),
        "--config",
        str(runtime_config),
        "--metrics-dir",
        str(metrics_dir),
        "--job-metadata",
        str(runtime_metadata),
    ]
    if len(gpu_ids) == 1:
        command = [fixed["python"], *worker_args]
    else:
        port = 20000 + int(re.sub(r"\D", "", job["job_id"])[-4:] or "0") % 20000
        command = [
            fixed["torchrun"],
            "--standalone",
            f"--master-port={port}",
            f"--nproc-per-node={len(gpu_ids)}",
            *worker_args,
        ]
    job_environment = {
        "CUDA_VISIBLE_DEVICES": gpu_mask,
        "ENABLE_CCE": "1" if fixed["enable_cce"] else "0",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "8",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "PYTHONUNBUFFERED": "1",
        # flash_attn_3/_C.abi3.so 需要 cuDriverGetVersion，但它的 NEEDED 列表里
        # 没有 libcuda——该符号本应由先加载的 torch 带进全局符号表。2026-09-01
        # 之后这条链断了（.so 文件 6-25 起未变、两个 venv 同一份，libcuda 本身
        # 也有该符号），FA3 在两个环境里都 ImportError，任何用 FA3 的作业直接
        # 失败。显式预加载驱动库把符号放进全局符号表即可恢复。
        #
        # 只影响符号可见性，不改 FA3 内核与计算逻辑，因此与历史 440 个
        # fa3_orig 作业的测量结果仍然可比。
        "LD_PRELOAD": LIBCUDA_PRELOAD,
    }
    if fixed.get("fa3_variant"):
        job_environment["FA3_VARIANT"] = str(fixed["fa3_variant"])
    overlay = job.get("environment_overlay") or {}
    if overlay:
        contract_path = Path(str(overlay.get("contract_path") or ""))
        contract_sha256 = str(overlay.get("contract_sha256") or "")
        if not contract_path.is_file() or sha256_file(contract_path) != contract_sha256:
            raise RuntimeError("Job environment-overlay contract is absent or changed")
        contract = read_json(contract_path)
        for path_text, expected_sha256 in (contract.get("file_sha256") or {}).items():
            bound_path = Path(path_text)
            if not bound_path.is_file() or sha256_file(bound_path) != expected_sha256:
                raise RuntimeError(f"Qwen3.5 runtime file binding changed: {bound_path}")
        prefixes = [str(value) for value in overlay.get("PYTHONPATH_prepend") or []]
        if prefixes:
            existing = os.environ.get("PYTHONPATH")
            job_environment["PYTHONPATH"] = os.pathsep.join(
                [*prefixes, *([existing] if existing else [])]
            )
        for key, value in (overlay.get("variables") or {}).items():
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", str(key)):
                raise RuntimeError(f"Invalid environment-overlay variable: {key!r}")
            job_environment[str(key)] = str(value)
    env = os.environ.copy()
    env.update(job_environment)
    runtime_mechanism = runtime_mechanism_manifest(
        experiment=experiment,
        runtime_identity=runtime_identity,
        runtime_hardware=runtime_hardware,
        environment=env,
        project_root=ROOT,
    )
    _atomic_write_json(runtime_mechanism_path, runtime_mechanism)
    authorization = execution_authorization(
        mode=authorization_mode,
        job=authorized_job,
        approval_check=approval_check,
    )
    provenance_path = input_snapshots["provenance"]
    provenance = read_json(provenance_path)
    execution_inputs = execution_inputs_manifest(
        job=authorized_job,
        execution_attempt_id=execution_attempt_id,
        attempt_root=attempt_dir,
        runtime_identity=runtime_identity,
        runtime_identity_path=runtime_identity_path,
        runtime_config=runtime_config,
        runtime_metadata=runtime_metadata,
        config=config,
        command=command,
        environment=env,
        provenance=provenance,
        authorization=authorization,
        runtime_hardware=runtime_hardware,
        runtime_hardware_path=runtime_hardware_path,
        live_topology_path=topology_path,
        runtime_mechanism=runtime_mechanism,
        runtime_mechanism_path=runtime_mechanism_path,
        input_snapshots=input_snapshots,
    )
    _atomic_write_json(execution_inputs_path, execution_inputs)
    render = {
        "schema": "sft_rendered_run/v2",
        "job": authorized_job,
        "execution_attempt_id": execution_attempt_id,
        "attempt_root": ".",
        "authorization": authorization,
        "approval_design_sha256": (authorization.get("evidence") or {}).get(
            "approval_design_sha256"
        ),
        "calibration_eligible": False,
        "gpu_mask": gpu_mask,
        "config_path": str(runtime_config),
        "runtime_metadata_path": str(runtime_metadata),
        "command": command,
        "environment": job_environment,
        "provenance_sha256": sha256_file(provenance_path),
        "runtime_identity_path": str(runtime_identity_path),
        "runtime_fingerprint_sha256": runtime_fingerprint,
        "runtime_hardware_path": str(runtime_hardware_path),
        "runtime_hardware_sha256": sha256_file(runtime_hardware_path),
        "runtime_mechanism_path": str(runtime_mechanism_path),
        "runtime_mechanism_sha256": sha256_file(runtime_mechanism_path),
        "execution_inputs_path": str(execution_inputs_path),
        "execution_inputs_sha256": sha256_file(execution_inputs_path),
        "execution_fingerprint_path": str(execution_manifest_path),
        "execution_fingerprint_sha256": None,
        "execution_fingerprint_quality": "pending_runtime_model_manifests",
        "execution_fingerprint_errors": [],
    }
    _atomic_write_json(rendered_path, render)
    _replace_latest_symlink(
        job_result_dir / "rendered_run.json",
        rendered_path,
        execution_attempt_id,
    )
    _atomic_write_json(
        job_result_dir / "latest_attempt.json",
        {
            "schema": "sft_latest_attempt/v1",
            "job_id": str(authorized_job["job_id"]),
            "execution_attempt_id": execution_attempt_id,
            "attempt_path": str(attempt_dir.relative_to(job_result_dir)),
            "state": "rendered" if args.render_only else "running",
            "calibration_evidence_root": str(
                attempt_dir.relative_to(job_result_dir)
            ),
        },
    )
    _replace_latest_symlink(
        job_result_dir / "runtime_identity.json",
        runtime_identity_path,
        execution_attempt_id,
    )
    if args.render_only:
        print(json.dumps(render, ensure_ascii=False, indent=2))
        return

    _replace_latest_symlink(
        job_result_dir / "metrics", metrics_dir, execution_attempt_id
    )
    _replace_latest_symlink(
        job_result_dir / "train.log", log_path, execution_attempt_id
    )
    _replace_latest_symlink(
        job_result_dir / "nvidia_smi.csv",
        nvidia_smi_path,
        execution_attempt_id,
    )
    _replace_latest_symlink(
        job_result_dir / "thermal_summary.json",
        thermal_summary_path,
        execution_attempt_id,
    )
    ensure_gpus_idle(idle_check_gpu_ids(job, gpu_ids, experiment))
    stop_monitor = threading.Event()
    monitor_health: dict[str, Any] = {}
    monitor = threading.Thread(
        target=monitor_nvidia_smi,
        args=(
            stop_monitor,
            nvidia_smi_path,
            set(gpu_ids),
            float(experiment["measurement"]["nvidia_smi_interval_seconds"]),
            monitor_health,
        ),
        daemon=True,
    )
    started = time.time()
    monitor.start()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd="/fine-tuning-launcher",
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    stop_monitor.set()
    monitor.join(timeout=6)
    monitor_health["thread_alive_after_join"] = monitor.is_alive()
    if monitor.is_alive():
        thermal_summary = {
            "schema_version": 1,
            "policy": "record_only",
            "affects_job_classification": False,
            "source": "nvidia_smi.csv",
            "monitor_data_available": False,
            "monitor_error": "Monitor thread was still active after the bounded join; summary skipped.",
            "gpus": [],
        }
    else:
        try:
            thermal_summary = summarize_nvidia_smi(
                nvidia_smi_path,
                set(gpu_ids),
            )
            thermal_summary["source"] = "nvidia_smi.csv"
        except Exception as error:
            thermal_summary = {
                "schema_version": 1,
                "policy": "record_only",
                "affects_job_classification": False,
                "source": "nvidia_smi.csv",
                "monitor_data_available": False,
                "monitor_error": repr(error),
                "gpus": [],
            }
    thermal_summary["monitor_health"] = monitor_health
    thermal_summary_write_error = None
    try:
        _atomic_write_json(thermal_summary_path, thermal_summary)
    except Exception as error:
        thermal_summary_write_error = repr(error)
    finished = time.time()
    classification = classify_execution(
        process.returncode,
        log_path,
        len(gpu_ids),
        job_id=str(authorized_job["job_id"]),
        execution_attempt_id=execution_attempt_id,
    )
    execution_fingerprint: str | None = None
    execution_fingerprint_errors: list[str] = []
    try:
        execution_manifest = finalize_execution_fingerprint(
            job=authorized_job,
            execution_attempt_id=execution_attempt_id,
            execution_inputs=execution_inputs,
            execution_inputs_path=execution_inputs_path,
            result_dir=attempt_dir,
            expected_ranks=len(gpu_ids),
            terminal_classification=classification["evidence"],
            return_code=process.returncode,
        )
        _atomic_write_json(execution_manifest_path, execution_manifest)
        execution_fingerprint = sha256_json(execution_manifest)
        execution_fingerprint_quality = "complete"
    except (
        OSError,
        RuntimeError,
        RuntimeEvidenceError,
        TypeError,
        ValueError,
    ) as error:
        execution_fingerprint_quality = "incomplete"
        execution_fingerprint_errors.append(f"{type(error).__name__}: {error}")
    calibratable_outcome = bool(
        (
            classification["classification"] == "success"
            and process.returncode == 0
            and classification["evidence"]["summaries_complete"] is True
        )
        or (
            classification["classification"] == "oom"
            and classification["evidence"]["cuda_oom_confirmed"] is True
        )
    )
    calibration_eligible = bool(
        authorization["calibration_eligible"]
        and runtime_hardware["calibration_hardware_eligible"]
        and execution_fingerprint_quality == "complete"
        and calibratable_outcome
    )
    render.update(
        {
            "calibration_eligible": calibration_eligible,
            "execution_fingerprint_sha256": execution_fingerprint,
            "execution_fingerprint_quality": execution_fingerprint_quality,
            "execution_fingerprint_errors": execution_fingerprint_errors,
        }
    )
    status = {
        "schema": "sft_execution_status/v2",
        "job_id": authorized_job["job_id"],
        "return_code": process.returncode,
        "classification": classification["classification"],
        "classification_evidence": classification["evidence"],
        "started_unix": started,
        "finished_unix": finished,
        "wall_seconds": finished - started,
        "gpu_mask": gpu_mask,
        "authorization_mode": authorization_mode,
        "calibration_eligible": calibration_eligible,
        "approval_design_sha256": approval_check["design_sha256"]
        if approval_check
        else None,
        "provenance_sha256": render["provenance_sha256"],
        "runtime_fingerprint_sha256": runtime_fingerprint,
        "execution_attempt_id": execution_attempt_id,
        "execution_inputs_sha256": render["execution_inputs_sha256"],
        "execution_fingerprint_sha256": execution_fingerprint,
        "execution_fingerprint_quality": execution_fingerprint_quality,
        "execution_fingerprint_errors": execution_fingerprint_errors,
        "evidence_paths": {
            "rendered_run": "rendered_run.json",
            "runtime_identity": "runtime_identity.json",
            "runtime_hardware": "runtime_hardware.json",
            "runtime_mechanism": "runtime_mechanism.json",
            "execution_inputs": "execution_inputs.json",
            "execution_fingerprint": "execution_fingerprint.json",
            "runtime_config": "runtime_config.yaml",
            "runtime_metadata": "job_metadata.json",
            "metrics": "metrics",
            "log": "train.log",
            "nvidia_smi": "nvidia_smi.csv",
            "thermal_summary": "thermal_summary.json",
        },
        "thermal_observation": {
            "policy": "record_only",
            "affects_job_classification": False,
            "summary_path": "thermal_summary.json",
            "summary_write_error": thermal_summary_write_error,
            "monitor_data_available": thermal_summary.get(
                "monitor_data_available", False
            ),
            "any_sw_thermal_slowdown": thermal_summary.get(
                "any_sw_thermal_slowdown", False
            ),
            "any_hw_thermal_slowdown": thermal_summary.get(
                "any_hw_thermal_slowdown", False
            ),
        },
    }
    _atomic_write_json(rendered_path, render)
    _atomic_write_json(status_path, status)
    _replace_latest_symlink(
        job_result_dir / "status.json",
        status_path,
        execution_attempt_id,
    )
    _atomic_write_json(
        job_result_dir / "latest_attempt.json",
        {
            "schema": "sft_latest_attempt/v1",
            "job_id": str(authorized_job["job_id"]),
            "execution_attempt_id": execution_attempt_id,
            "attempt_path": str(attempt_dir.relative_to(job_result_dir)),
            "state": "complete",
            "classification": status["classification"],
            "calibration_eligible": calibration_eligible,
            "calibration_evidence_root": str(
                attempt_dir.relative_to(job_result_dir)
            ),
        },
    )
    print(json.dumps(status, ensure_ascii=False))
    raise SystemExit(0 if status["classification"] == "success" else 2)


if __name__ == "__main__":
    main()
