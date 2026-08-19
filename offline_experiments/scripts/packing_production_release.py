#!/usr/bin/env python3
"""Load and enforce the bounded pure-text Packing production release."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from common import ROOT, read_json, sha256_file, sha256_json


RELEASE_SCHEMA = "sft_h800_text_packing_production_release/v1"
DEFAULT_RELEASE_PATH = ROOT / "config" / "h800_text_packing_production_release_v1.json"


def _binding_path(release_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Bindings are expressed relative to offline_experiments so the repository
    # can move without changing the release contract.
    return (ROOT / path).resolve()


def load_packing_release(path: Path = DEFAULT_RELEASE_PATH) -> dict[str, Any]:
    release_path = Path(path).resolve()
    release = read_json(release_path)
    if release.get("schema") != RELEASE_SCHEMA:
        raise ValueError("Packing production release schema mismatch")
    if (
        release.get("status") != "active_limited_production"
        or release.get("authorization") != "explicit_user_acceptance_2026-08-18"
        or release.get("automatic_execution_allowed") is not True
        or release.get("fail_closed") is not True
        or release.get("vl_packing_allowed") is not False
    ):
        raise ValueError("Packing production release contract is not active and bounded")

    bindings = release.get("evidence_bindings") or {}
    loaded: dict[str, dict[str, Any]] = {}
    for name in ("static_policy", "packing_ranking_acceptance", "transfer_validation"):
        binding = bindings.get(name) or {}
        bound_path = _binding_path(release_path, str(binding.get("path") or ""))
        if not bound_path.is_file() or sha256_file(bound_path) != binding.get("sha256"):
            raise ValueError(f"Packing release evidence binding drifted: {name}")
        loaded[name] = read_json(bound_path)

    policy = loaded["static_policy"]
    ranking = loaded["packing_ranking_acceptance"]
    transfer = loaded["transfer_validation"]
    unsigned_transfer = dict(transfer)
    transfer_checksum = unsigned_transfer.pop("report_sha256", None)
    if (
        policy.get("policy_id")
        != bindings["static_policy"].get("policy_id")
        or policy.get("decision_semantics", {}).get("fail_closed") is not True
        or ranking.get("decision", {}).get("current_empirical_ranking_gate_passed")
        is not True
        or transfer.get("accepted") is not True
        or transfer_checksum != bindings["transfer_validation"].get("report_sha256")
        or transfer_checksum != sha256_json(unsigned_transfer)
    ):
        raise ValueError("Packing release acceptance evidence is not valid")
    return release


def scope_mismatches(
    request: Mapping[str, Any],
    release: Mapping[str, Any],
) -> list[str]:
    scope = release["scope"]
    mismatches: list[str] = []
    exact = (
        ("hardware_id", "hardware_ids", "h800"),
        ("modality", "modalities", "text"),
        ("stage", "stages", "sft"),
        ("dtype", "dtypes", "bf16"),
        ("model_id", "model_ids", None),
        ("training_mode", "training_modes", None),
        ("lora_rank", "lora_ranks", 32),
        ("cutoff_len", "cutoff_lens", None),
        ("gpu_count", "gpu_counts", None),
        ("target_gbs", "target_gbs_values", None),
    )
    for request_key, scope_key, default in exact:
        actual = request.get(request_key, default)
        if actual not in scope[scope_key]:
            mismatches.append(request_key)
    if request.get("packing") is not True:
        mismatches.append("packing")
    if int(request.get("physical_mbs") or 0) != int(scope["packed_physical_mbs"]):
        mismatches.append("physical_mbs")
    if bool(request.get("offload")) is not bool(scope["offload_allowed"]):
        mismatches.append("offload")
    if str(scope["required_kernel_substring"]).lower() not in str(
        request.get("kernel_path") or "fa3"
    ).lower():
        mismatches.append("kernel_path")
    if request.get("vl_workload_profile_path") is not None:
        mismatches.append("vl_workload_profile_path")
    return mismatches
