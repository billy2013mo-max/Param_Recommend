#!/usr/bin/env python3
"""Static preflight for environment, data, models, matrix and launch safety."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import torch

from common import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    DATA_DIR,
    MATRIX_DIR,
    ROOT,
    RUNTIME_DIR,
    TOKENIZER_FILES,
    command_output,
    prepare_model_tokenizer_view,
    read_json,
    read_jsonl,
    sha256_file,
    verify_file_manifest,
    write_json,
)
from run_job import APPROVAL_FILE, render_config


def package_versions() -> dict[str, str]:
    names = (
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "deepspeed",
        "peft",
        "trl",
        "tokenizers",
        "triton",
        "liger-kernel",
        "flash-attn",
        "flash-attn-3",
        "cut-cross-entropy",
        "llamafactory",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed-under-this-distribution-name"
    return versions


def gpu_snapshot() -> dict[str, Any]:
    selected_gpu_indices = read_json(CONFIG_DIR / "experiment.json")["training_scope"]["gpu_ids"]
    expected_hardware = read_json(CONFIG_DIR / "hardware.json")
    query = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,pci.bus_id,power.limit,clocks.max.sm,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in query.splitlines():
        values = [value.strip() for value in line.split(",")]
        gpus.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "uuid": values[2],
                "memory_mib": int(values[3]),
                "pci_bus_id": values[4],
                "power_limit_w": float(values[5]),
                "max_sm_clock_mhz": int(values[6]),
                "driver_version": values[7],
            }
        )
    topology = command_output(["nvidia-smi", "topo", "-m"])
    (ARTIFACT_DIR / "nvidia_topology.txt").write_text(topology + "\n", encoding="utf-8")

    process_output = command_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"]
    )
    processes = []
    for line in process_output.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        processes.append({"pid": int(values[0]), "gpu_uuid": values[1], "used_memory_mib": int(values[2])})
    selected_uuids = {gpu["uuid"] for gpu in gpus if gpu["index"] in set(selected_gpu_indices)}
    selected_processes = [process for process in processes if process["gpu_uuid"] in selected_uuids]
    selected_gpus = [gpu for gpu in gpus if gpu["index"] in set(selected_gpu_indices)]
    available_indices = {gpu["index"] for gpu in gpus}
    missing_gpu_indices = sorted(set(selected_gpu_indices) - available_indices)
    torch_devices = []
    for index in selected_gpu_indices:
        # A preflight must report a hardware mismatch, not crash while
        # probing an index from a different node's frozen GPU pool.
        if index not in available_indices:
            continue
        properties = torch.cuda.get_device_properties(index)
        torch_devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "multiprocessors": properties.multi_processor_count,
                "total_memory_bytes": properties.total_memory,
            }
        )
    hardware_matches_config = (
        not missing_gpu_indices
        and len(selected_gpus) == int(expected_hardware.get("gpu_count", len(selected_gpu_indices)))
        and len(torch_devices) == len(selected_gpu_indices)
        and all(device["name"] == expected_hardware["name_reported_by_driver"] for device in torch_devices)
        and all(
            device["total_memory_bytes"] == int(expected_hardware["memory_bytes_reported_by_torch"])
            for device in torch_devices
        )
        and all(device["compute_capability"] == expected_hardware["compute_capability"] for device in torch_devices)
    )
    return {
        "gpus": gpus,
        "selected_gpus": selected_gpus,
        "selected_gpu_indices": selected_gpu_indices,
        "missing_gpu_indices": missing_gpu_indices,
        "selected_gpu_processes": selected_processes,
        "selected_gpus_idle": not selected_processes,
        "hardware_matches_config": hardware_matches_config,
        "topology_file": str(ARTIFACT_DIR / "nvidia_topology.txt"),
        "torch_devices": torch_devices,
        "torch_device": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "multiprocessors": torch.cuda.get_device_properties(0).multi_processor_count,
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
    }


def data_checks() -> dict[str, Any]:
    analysis = read_json(ARTIFACT_DIR / "dataset_analysis.json")
    checks = []
    for dataset_id, dataset in analysis["datasets"].items():
        path = Path(dataset["file"])
        rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        profile_checks = {}
        for profile_id, profile in dataset["profiles"].items():
            lengths = profile["lengths"]
            profile_checks[profile_id] = {
                "cutoff_len": lengths["cutoff_len"],
                "max_tokens": lengths["max_tokens"],
                "zero_truncation": lengths["truncated_samples_at_cutoff"] == 0,
            }
        checks.append(
            {
                "dataset_id": dataset_id,
                "rows": rows,
                "rows_ok": rows == 1000,
                "sha256_matches": sha256_file(path) == dataset["sha256"],
                "zero_truncation": all(row["zero_truncation"] for row in profile_checks.values()),
                "profiles": profile_checks,
            }
        )
    return {
        "datasets": checks,
        "all_passed": all(
            row["rows_ok"] and row["sha256_matches"] and row["zero_truncation"] for row in checks
        ),
    }


def model_checks() -> dict[str, Any]:
    inventory_path = ARTIFACT_DIR / "model_inventory.json"
    if not inventory_path.is_file():
        return {"models": [], "all_passed": False, "reason": "model_inventory.json is absent"}
    inventory = read_json(inventory_path)
    catalog_path = CONFIG_DIR / "models.json"
    catalog = read_json(catalog_path)
    catalog_matches = inventory.get("catalog_sha256") == sha256_file(catalog_path)
    expected_ids = {row["id"] for row in catalog["models"]}
    checks = []
    for model in inventory.get("models") or []:
        model_path = Path(model["path"])
        tokenizer_path = Path(model["tokenizer_path"])
        runtime_model_path = prepare_model_tokenizer_view(
            model_path, tokenizer_path, RUNTIME_DIR / "preflight_model_views" / model["id"]
        )
        runtime_view_ok = (
            (runtime_model_path / "config.json").resolve() == (model_path / "config.json").resolve()
            and all(
                (runtime_model_path / name).resolve() == (tokenizer_path / name).resolve()
                for name in TOKENIZER_FILES
            )
        )
        shards_ok = True
        for shard in model.get("checkpoint_manifest") or []:
            path = model_path / shard["name"]
            shards_ok = shards_ok and path.is_file() and path.stat().st_size == shard["size"]
        catalog_entry = next(row for row in catalog["models"] if row["id"] == model["id"])
        check = {
            "id": model["id"],
            "path": str(model_path),
            "path_matches_catalog": model["path"] == catalog_entry["path"],
            "tokenizer_path_matches_catalog": model["tokenizer_path"] == catalog_entry["tokenizer_path"],
            "directory_exists": model_path.is_dir(),
            "tokenizer_directory_exists": tokenizer_path.is_dir(),
            "shards_ok": shards_ok,
            "config_exists": (model_path / "config.json").is_file(),
            "tokenizer_exists": (model_path / "tokenizer.json").is_file(),
            "directory_identity_matches": model.get("model_directory_name") == model_path.name
            and model.get("model_identity") == str(model_path),
            "tokenizer_identity_matches": model.get("tokenizer_directory_name") == tokenizer_path.name
            and model.get("tokenizer_identity") == str(tokenizer_path),
            "qwen3_tokenizer_policy": tokenizer_path.name.lower().startswith("qwen3")
            and model.get("template") == "qwen3_nothink",
            "tokenizer_fits_model_embeddings": model.get("tokenizer_fits_model_embeddings") is True,
            "runtime_model_tokenizer_view_ok": runtime_view_ok,
            "context_ok": int(model.get("max_position_embeddings") or 0) >= 32768,
        }
        check["passed"] = all(value is True for key, value in check.items() if key not in {"id", "path"})
        checks.append(check)
    ids_match = {row["id"] for row in checks} == expected_ids
    return {
        "inventory_sha256": sha256_file(inventory_path),
        "catalog_matches": catalog_matches,
        "ids_match": ids_match,
        "models": checks,
        "all_passed": catalog_matches and ids_match and all(row["passed"] for row in checks),
    }


def preprocessing_checks() -> dict[str, Any]:
    path = ARTIFACT_DIR / "preprocessing_validation.json"
    if not path.is_file():
        return {"path": str(path), "all_passed": False, "reason": "validation artifact is absent"}
    report = read_json(path)
    freshness = {
        "dataset_analysis": report.get("dataset_analysis_sha256") == sha256_file(ARTIFACT_DIR / "dataset_analysis.json"),
        "models_config": report.get("models_config_sha256") == sha256_file(CONFIG_DIR / "models.json"),
        "dataset_info": report.get("dataset_info_sha256") == sha256_file(DATA_DIR / "dataset_info.json"),
        "llamafactory": report.get("llamafactory_version") == importlib.metadata.version("llamafactory"),
        "preprocessing_num_workers": report.get("preprocessing_num_workers")
        == read_json(CONFIG_DIR / "experiment.json")["fixed_runtime"]["preprocessing_num_workers"],
    }
    return {
        "path": str(path),
        "checks": len(report.get("checks") or []),
        "freshness": freshness,
        "all_passed": report.get("all_passed") is True
        and len(report.get("checks") or []) == 6
        and all(freshness.values()),
    }


def smoke_checks() -> dict[str, Any]:
    path = ARTIFACT_DIR / "smoke_validation.json"
    if not path.is_file():
        return {"path": str(path), "all_passed": False, "reason": "smoke validation artifact is absent"}
    report = read_json(path)
    freshness = verify_file_manifest(ROOT, report.get("freshness_manifest") or {})
    return {
        "path": str(path),
        "checks": len(report.get("checks") or []),
        "freshness": freshness,
        "all_passed": report.get("all_passed") is True
        and len(report.get("checks") or []) == 4
        and freshness["all_passed"],
    }


def provenance_checks() -> dict[str, Any]:
    path = ARTIFACT_DIR / "provenance.json"
    if not path.is_file():
        return {"path": str(path), "all_passed": False, "reason": "provenance artifact is absent"}
    report = read_json(path)
    source = verify_file_manifest(ROOT, report.get("project_source_manifest") or {})
    return {
        "path": str(path),
        "runtime_fingerprint_sha256": report.get("runtime_fingerprint_sha256"),
        "image_digest_available": report.get("image_digest_available"),
        "source_manifest": source,
        "all_passed": report.get("reproducibility_identity_complete") is True and source["all_passed"],
    }


def matrix_checks() -> dict[str, Any]:
    expected = read_json(MATRIX_DIR / "design_summary.json")
    matrix_rows = {
        "memory": read_jsonl(MATRIX_DIR / "memory_boundary_families.jsonl"),
        "throughput": read_jsonl(MATRIX_DIR / "throughput_requests.jsonl"),
        "scaling": read_jsonl(MATRIX_DIR / "strong_scaling_requests.jsonl"),
        "packing": read_jsonl(MATRIX_DIR / "packing_pair_requests.jsonl"),
        "profiler": read_jsonl(MATRIX_DIR / "profiler_requests.jsonl"),
    }
    counts = {
        "memory_boundary_families": len(matrix_rows["memory"]),
        "throughput_configurations_before_repeats": len(matrix_rows["throughput"]),
        "strong_scaling_families": len(matrix_rows["scaling"]),
        "packing_pair_families": len(matrix_rows["packing"]),
        "profiler_calibration_configurations": len(matrix_rows["profiler"]),
    }
    matches = {key: counts[key] == expected[key] for key in counts}
    actual_model_ids = sorted(
        {row["model_id"] for rows in matrix_rows.values() for row in rows if "model_id" in row}
    )
    configured_model_ids = read_json(CONFIG_DIR / "experiment.json")["training_scope"]["model_ids"]
    scope_matches = actual_model_ids == sorted(configured_model_ids) == sorted(expected["enabled_model_ids"])
    return {
        "summary": expected,
        "actual_counts": counts,
        "matches": matches,
        "actual_model_ids": actual_model_ids,
        "scope_matches": scope_matches,
        "all_passed": all(matches.values()) and scope_matches,
    }


def deepspeed_checks() -> dict[str, Any]:
    checks = {}
    for stage in (2, 3):
        path = CONFIG_DIR / "deepspeed" / f"ds_z{stage}.json"
        config = read_json(path)
        zero = config["zero_optimization"]
        checks[f"zero{stage}"] = {
            "path": str(path),
            "stage": zero["stage"],
            "overlap_comm": zero["overlap_comm"],
            "offload_present": any("offload" in key for key in zero),
            "matches_production_launcher_default": zero["stage"] == stage and zero["overlap_comm"] is False,
        }
    return checks


def rendered_config_checks() -> dict[str, Any]:
    expected_attention = read_json(CONFIG_DIR / "experiment.json")["fixed_runtime"]["flash_attn"]
    families = read_jsonl(MATRIX_DIR / "memory_boundary_families.jsonl")
    selected: list[dict[str, Any]] = []
    wanted = {
        (1, "none", False, "full"),
        (1, "none", True, "lora"),
        (2, "zero2", False, "full"),
        (2, "zero3", True, "lora"),
        (4, "zero2", True, "full"),
        (4, "zero3", False, "lora"),
    }
    for family in families:
        key = (family["gpu_count"], family["zero"], family["gc"], family["train_type"])
        if key in wanted and all(
            (row["gpu_count"], row["zero"], row["gc"], row["train_type"]) != key for row in selected
        ):
            selected.append(family)

    checks = []
    for family in selected:
        concrete = dict(family)
        concrete["kind"] = "memory_probe"
        concrete["mbs"] = concrete["mbs_candidates"][0]
        config = render_config(concrete, RUNTIME_DIR / "preflight_only")
        gc = bool(family["gc"])
        checks.append(
            {
                "gpu_count": family["gpu_count"],
                "zero": family["zero"],
                "gc": gc,
                "train_type": family["train_type"],
                "single_has_no_deepspeed": family["gpu_count"] != 1 or "deepspeed" not in config,
                "multi_has_expected_deepspeed": family["gpu_count"] == 1
                or config.get("deepspeed", "").endswith(f"ds_z{family['zero'][-1]}.json"),
                "gc_flags_consistent": config["gradient_checkpointing"] is gc
                and config["disable_gradient_checkpointing"] is (not gc),
                "attention_backend_matches": config["flash_attn"] == expected_attention,
                "liger_fixed": config["enable_liger_kernel"] is True,
                "fused_adamw": config["optim"] == "adamw_torch_fused",
            }
        )
    return {"representative_configs": checks, "count": len(checks)}


def environment_snapshot(gpu: dict[str, Any]) -> dict[str, Any]:
    distribution = importlib.metadata.distribution("llamafactory")
    direct_url_path = Path(
        distribution.locate_file(f"llamafactory-{distribution.version}.dist-info/direct_url.json")
    )
    direct_url = read_json(direct_url_path) if direct_url_path.exists() else {}
    disk = shutil.disk_usage(ROOT)
    return {
        "python_executable": os.sys.executable,
        "python_version": os.sys.version,
        "packages": package_versions(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": torch.cuda.nccl.version(),
        "llamafactory_import_path": __import__("llamafactory").__file__,
        "llamafactory_direct_url": direct_url,
        "runtime_flags": read_json(CONFIG_DIR / "experiment.json")["fixed_runtime"],
        "gpu": gpu,
        "workspace_disk_free_bytes": disk.free,
    }


def freeze_approval_design(
    extra_files: Iterable[Path] = (),
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    files: list[Path] = []
    files.extend(path for path in CONFIG_DIR.rglob("*") if path.is_file() and path != APPROVAL_FILE)
    files.extend(path for path in (ROOT / "scripts").glob("*.py") if path.is_file())
    files.extend(
        MATRIX_DIR / name
        for name in (
            "design_summary.json",
            "memory_boundary_families.jsonl",
            "packing_pair_requests.jsonl",
            "profiler_requests.jsonl",
            "strong_scaling_requests.jsonl",
            "throughput_requests.jsonl",
        )
    )
    files.extend(
        [
            ARTIFACT_DIR / "dataset_analysis.json",
            ARTIFACT_DIR / "model_inventory.json",
            ARTIFACT_DIR / "preprocessing_validation.json",
            ARTIFACT_DIR / "provenance.json",
            ARTIFACT_DIR / "smoke_validation.json",
            DATA_DIR / "dataset_info.json",
            ROOT / "README.md",
            ROOT / "EXPERIMENT_DESIGN.md",
        ]
    )
    files.extend((DATA_DIR / "derived").glob("*.jsonl"))
    files.extend(extra_files)
    absent = [str(path) for path in files if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Cannot freeze approval design; required files are absent: {absent}")
    manifest = {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(set(files))}
    design = {
        "schema_version": 1,
        "training_started": False,
        "file_sha256": manifest,
        "matrix_summary": read_json(MATRIX_DIR / "design_summary.json"),
        "approval_instruction": "Approval must bind to the SHA256 of this exact file.",
    }
    design.update(extra_metadata or {})
    write_json(RUNTIME_DIR / "approval_design.json", design)
    return {"path": str(RUNTIME_DIR / "approval_design.json"), "sha256": sha256_file(RUNTIME_DIR / "approval_design.json")}


def write_markdown(report: dict[str, Any]) -> None:
    gpu = report["environment"]["gpu"]
    selected = ",".join(map(str, gpu["selected_gpu_indices"]))
    missing = ",".join(map(str, gpu.get("missing_gpu_indices") or []))
    if missing:
        pool_status = f"缺失配置 GPU {missing}（实际可见卡池不完整）"
    else:
        pool_status = f"空闲：{'是' if gpu['selected_gpus_idle'] else '否'}"
    lines = [
        "# 开跑前静态校验",
        "",
        f"- 总体状态：{'可进入人工审批' if report['ready_for_approval'] else '存在静态阻塞项'}",
        f"- 训练锁：{'仍锁定（符合预期）' if report['approval_lock_active'] else '已存在批准文件'}",
        f"- 当前卡池 {selected}：{pool_status}",
        f"- 数据切片校验：{'通过' if report['data']['all_passed'] else '失败'}",
        f"- 模型快照校验：{'通过' if report['models']['all_passed'] else '失败'}",
        f"- Processor 对齐：{'通过' if report['preprocessing']['all_passed'] else '失败'}",
        f"- 四组训练 smoke：{'通过' if report['smoke']['all_passed'] else '失败'}",
        f"- 复现指纹：{'通过' if report['provenance']['all_passed'] else '失败'}",
        f"- 冻结设计 SHA256：`{report['approval_design']['sha256']}`",
        "",
        f"所有阶段在 GPU {selected} 内按不相交 mask 并行填满；仅 4 卡任务独占该卡池，池外 GPU 不会被占用。",
        "",
    ]
    (ARTIFACT_DIR / "preflight_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    gpu = gpu_snapshot()
    data = data_checks()
    models = model_checks()
    preprocessing = preprocessing_checks()
    smoke = smoke_checks()
    provenance = provenance_checks()
    ds = deepspeed_checks()
    rendered = rendered_config_checks()
    matrix = matrix_checks()
    environment = environment_snapshot(gpu)
    approval_design = freeze_approval_design()
    config_passed = all(
        all(
            value is True
            for key, value in row.items()
            if key not in {"gpu_count", "zero", "gc", "train_type"}
        )
        for row in rendered["representative_configs"]
    )
    deepspeed_passed = all(row["matches_production_launcher_default"] for row in ds.values())
    static_checks_passed = all(
        (
            data["all_passed"],
            models["all_passed"],
            preprocessing["all_passed"],
            smoke["all_passed"],
            provenance["all_passed"],
            deepspeed_passed,
            config_passed,
            matrix["all_passed"],
            environment["gpu"]["hardware_matches_config"],
        )
    )
    approval_binding_valid = False
    if APPROVAL_FILE.is_file():
        approval = read_json(APPROVAL_FILE)
        approval_binding_valid = approval.get("approved") is True and approval.get("design_sha256") == approval_design["sha256"]
    report = {
        "schema_version": 1,
        "training_started": False,
        "approval_lock_active": not APPROVAL_FILE.exists(),
        "static_checks_passed": static_checks_passed,
        "ready_for_approval": static_checks_passed and gpu["selected_gpus_idle"],
        "approval_binding_valid": approval_binding_valid,
        "launch_ready_now": static_checks_passed and gpu["selected_gpus_idle"] and approval_binding_valid,
        "environment": environment,
        "data": data,
        "models": models,
        "preprocessing": preprocessing,
        "smoke": smoke,
        "provenance": provenance,
        "deepspeed": ds,
        "rendered_configs": rendered,
        "matrix": matrix,
        "approval_design": approval_design,
    }
    write_json(ARTIFACT_DIR / "preflight_report.json", report)
    write_markdown(report)
    print(json.dumps({
        "static_checks_passed": report["static_checks_passed"],
        "ready_for_approval": report["ready_for_approval"],
        "launch_ready_now": report["launch_ready_now"],
        "selected_gpu_processes": gpu["selected_gpu_processes"],
        "approval_design": approval_design,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
