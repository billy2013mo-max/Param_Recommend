#!/usr/bin/env python3
"""Shared helpers for the offline SFT efficiency experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
# Keep source/data paths anchored at the project root while allowing each
# hardware campaign to own config, artifacts, matrices, runtime state and
# results. With the environment variable unset, the legacy H800 layout is
# unchanged.
CAMPAIGN_ROOT = Path(os.environ.get("OFFLINE_EXPERIMENT_CAMPAIGN_ROOT", ROOT)).expanduser().resolve()
CONFIG_DIR = CAMPAIGN_ROOT / "config"
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = CAMPAIGN_ROOT / "artifacts"
MATRIX_DIR = CAMPAIGN_ROOT / "matrix"
RUNTIME_DIR = CAMPAIGN_ROOT / "runtime"
RESULTS_DIR = CAMPAIGN_ROOT / "results"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def percentile(values: list[float] | list[int], q: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def aligned_cutoff(max_tokens: int) -> int:
    return max(512, int(math.ceil(max_tokens / 512.0) * 512))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, payload: dict[str, Any], length: int = 16) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:length]}"


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_file_manifest(root: Path, manifest: dict[str, str]) -> dict[str, Any]:
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    for relative, expected in sorted(manifest.items()):
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual = sha256_file(path)
        if actual != expected:
            mismatched.append({"path": relative, "expected": expected, "actual": actual})
    return {
        "entries": len(manifest),
        "missing": missing,
        "mismatched": mismatched,
        "all_passed": not missing and not mismatched,
    }


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}")
    return result.stdout.strip()


def gpu_process_snapshot(selected_indices: Iterable[int]) -> dict[str, Any]:
    selected = sorted(set(int(index) for index in selected_indices))
    gpu_rows = command_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    )
    uuid_to_index: dict[str, int] = {}
    for line in gpu_rows.splitlines():
        index, uuid = (value.strip() for value in line.split(",", 1))
        uuid_to_index[uuid] = int(index)

    process_rows = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = []
    for line in process_rows.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",", 3)]
        if len(values) != 4 or values[1] not in uuid_to_index:
            continue
        index = uuid_to_index[values[1]]
        if index in selected:
            processes.append(
                {
                    "gpu_index": index,
                    "gpu_uuid": values[1],
                    "pid": int(values[0]),
                    "used_memory_mib": int(values[2]),
                    "process_name": values[3],
                }
            )
    return {"gpu_indices": selected, "processes": processes, "all_idle": not processes}


def ensure_gpus_idle(selected_indices: Iterable[int]) -> dict[str, Any]:
    snapshot = gpu_process_snapshot(selected_indices)
    if not snapshot["all_idle"]:
        raise RuntimeError(f"Refusing to start: selected GPUs have compute processes: {snapshot['processes']}")
    return snapshot


def prepare_model_tokenizer_view(model_path: Path, tokenizer_path: Path, view_path: Path) -> Path:
    """Create a symlink-only model view when weights and tokenizer live in different directories."""
    if model_path.resolve() == tokenizer_path.resolve():
        return model_path
    if not model_path.is_dir() or not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Missing model/tokenizer directory: {model_path}, {tokenizer_path}")
    missing = [name for name in TOKENIZER_FILES if not (tokenizer_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Tokenizer directory {tokenizer_path} is missing {missing}")

    view_path.mkdir(parents=True, exist_ok=True)
    tokenizer_names = {name for name in TOKENIZER_FILES if (tokenizer_path / name).is_file()}
    for source in model_path.iterdir():
        if source.is_file() and source.name not in tokenizer_names:
            destination = view_path / source.name
            if destination.is_symlink() and destination.resolve() == source.resolve():
                continue
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.symlink_to(source)
    for name in sorted(tokenizer_names):
        source = tokenizer_path / name
        destination = view_path / name
        if destination.is_symlink() and destination.resolve() == source.resolve():
            continue
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source)
    write_json(
        view_path / "model_tokenizer_view.json",
        {
            "model_path": str(model_path),
            "tokenizer_path": str(tokenizer_path),
            "policy": "Model files and Qwen3 tokenizer files are exposed through symlinks; weights are not copied.",
        },
    )
    return view_path
