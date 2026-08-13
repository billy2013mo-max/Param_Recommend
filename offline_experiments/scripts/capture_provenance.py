#!/usr/bin/env python3
"""Capture reproducibility identities for code, runtime, data, models and container."""

from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from common import ARTIFACT_DIR, CONFIG_DIR, ROOT, read_json, sha256_file, sha256_json, write_json


PACKAGE_NAMES = (
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


def git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def container_id() -> str | None:
    text = Path("/proc/self/cgroup").read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"cri-containerd-([0-9a-f]{64})\.scope", text)
    return matches[0] if matches else None


def image_digest() -> tuple[str | None, str]:
    for name in ("CONTAINER_IMAGE_DIGEST", "K8S_IMAGE_DIGEST", "IMAGE_DIGEST"):
        value = os.environ.get(name)
        if value:
            return value, f"environment:{name}"
    return None, "unavailable: pod imageID is not exposed by the container environment or ServiceAccount RBAC"


def source_manifest() -> dict[str, str]:
    paths = list((ROOT / "scripts").glob("*.py"))
    paths += [path for path in CONFIG_DIR.rglob("*") if path.is_file() and "APPROVED" not in path.name]
    paths += [ROOT / "README.md", ROOT / "EXPERIMENT_DESIGN.md"]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)}


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def optional_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def installed_vcs_commit(distribution_name: str) -> str | None:
    distribution = importlib.metadata.distribution(distribution_name)
    direct_url = Path(
        distribution.locate_file(
            f"{distribution_name}-{distribution.version}.dist-info/direct_url.json"
        )
    )
    if not direct_url.is_file():
        return None
    return (
        (read_json(direct_url).get("vcs_info") or {}).get("commit_id")
    )


def distribution_file_sha256(distribution_name: str, relative_path: str) -> str | None:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    path = Path(distribution.locate_file(relative_path))
    return sha256_file(path) if path.is_file() else None


def main() -> None:
    experiment = read_json(CONFIG_DIR / "experiment.json")
    fixed_runtime = experiment["fixed_runtime"]
    expected_prefix = Path(fixed_runtime["python"]).parent.parent
    if Path(sys.prefix) != expected_prefix:
        raise RuntimeError(
            "Provenance must be captured with the configured training interpreter: "
            f"{fixed_runtime['python']} (running prefix={sys.prefix!r}, expected={str(expected_prefix)!r})"
        )
    manifest = source_manifest()
    digest, digest_source = image_digest()
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()[0].strip()
    packages = package_versions()
    launcher_commits = {
        "finetuning_launcher": git_commit(Path("/fine-tuning-launcher/dev/finetuning-launcher")),
        "large_model_customization_server": git_commit(
            Path("/fine-tuning-launcher/dev/large-model-customization-server")
        ),
        "llamafactory_checkout": git_commit(
            Path(fixed_runtime.get("llamafactory_checkout", "/fine-tuning-launcher/LlamaFactory"))
        ),
        "llamafactory_installed": (
            fixed_runtime.get("llamafactory_commit")
            or installed_vcs_commit("llamafactory")
        ),
    }
    runtime_identity: dict[str, Any] = {
        "container_image_digest": digest,
        "container_image_digest_source": digest_source,
        "container_runtime_id": container_id(),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python": sys.version,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": list(torch.cuda.nccl.version()),
        "nvidia_driver": driver,
        "packages": packages,
        "framework_source_sha256": {
            "deepspeed_zero_partition_parameters": distribution_file_sha256(
                "deepspeed", "deepspeed/runtime/zero/partition_parameters.py"
            ),
            "llamafactory_adapter": distribution_file_sha256(
                "llamafactory", "llamafactory/model/adapter.py"
            ),
        },
        "launcher_patch_sha256": optional_sha256(
            Path("/fine-tuning-launcher/hack/patches/apply_deepspeed_zero3_mixed_dtype_fix.py")
        ),
        "launcher_commits": launcher_commits,
        "project_git_commit": git_commit(ROOT),
        "project_source_snapshot_sha256": sha256_json(manifest),
        "model_inventory_sha256": optional_sha256(ARTIFACT_DIR / "model_inventory.json"),
        "dataset_analysis_sha256": optional_sha256(ARTIFACT_DIR / "dataset_analysis.json"),
    }
    report = {
        "schema_version": 1,
        "identity_policy": (
            "Use container image digest when the platform exposes it; otherwise bind the run to the container runtime ID, "
            "package versions, CUDA libraries, launcher commits and the complete project source snapshot."
        ),
        "image_digest_available": digest is not None,
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "project_source_manifest": manifest,
        "model_snapshot_policy": "Configured local model directory names/paths are authoritative; only required-file presence and structural metadata are checked.",
        "dataset_snapshot_policy": "Local source/derived SHA256 values are authoritative; future downloads enforce Hub revision.",
        "reproducibility_identity_complete": bool(
            runtime_identity["model_inventory_sha256"]
            and runtime_identity["dataset_analysis_sha256"]
            and launcher_commits["llamafactory_installed"]
        ),
    }
    provenance_path = ARTIFACT_DIR / "provenance.json"
    if provenance_path.is_file():
        previous = read_json(provenance_path)
        previous_fingerprint = previous.get("runtime_fingerprint_sha256")
        if previous_fingerprint and previous_fingerprint != report["runtime_fingerprint_sha256"]:
            history_path = ARTIFACT_DIR / "provenance_history" / f"{previous_fingerprint}.json"
            if not history_path.exists():
                write_json(history_path, previous)
    write_json(provenance_path, report)
    print(
        {
            "runtime_fingerprint_sha256": report["runtime_fingerprint_sha256"],
            "image_digest_available": report["image_digest_available"],
            "project_git_commit": runtime_identity["project_git_commit"],
        }
    )


if __name__ == "__main__":
    main()
