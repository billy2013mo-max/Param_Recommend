#!/usr/bin/env python3
"""Load and enforce the bounded pure-text dense memory-admission release.

This release makes the dense (full-attention) text memory-admission capability
an independently executable production contract, in parallel to the pure-text
Packing production release.  It is deliberately fail-closed: hybrid-attention,
VL and MoE requests are never admitted here, and every evidence binding is
verified by sha256 before the contract is considered active.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from common import ROOT, read_json, sha256_file, sha256_json


RELEASE_SCHEMA = "sft_h800_text_dense_memory_admission_release/v1"
DEFAULT_RELEASE_PATH = (
    ROOT / "config" / "h800_text_dense_memory_admission_release_v1.json"
)


def _binding_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Bindings are expressed relative to offline_experiments so the repository
    # can move without changing the release contract.
    return (ROOT / path).resolve()


def load_dense_memory_admission_release(
    path: Path = DEFAULT_RELEASE_PATH,
) -> dict[str, Any]:
    release_path = Path(path).resolve()
    release = read_json(release_path)
    if release.get("schema") != RELEASE_SCHEMA:
        raise ValueError("Dense memory-admission release schema mismatch")
    if (
        release.get("status") != "active_limited_production"
        or release.get("authorization") != "explicit_user_acceptance_2026-08-18"
        or release.get("automatic_execution_allowed") is not True
        or release.get("fail_closed") is not True
        or release.get("vl_allowed") is not False
    ):
        raise ValueError(
            "Dense memory-admission release contract is not active and bounded"
        )

    bindings = release.get("evidence_bindings") or {}
    loaded: dict[str, dict[str, Any]] = {}
    for name in ("business_blind", "transfer_validation"):
        binding = bindings.get(name) or {}
        bound_path = _binding_path(str(binding.get("path") or ""))
        if not bound_path.is_file() or sha256_file(bound_path) != binding.get("sha256"):
            raise ValueError(
                f"Dense memory-admission release evidence binding drifted: {name}"
            )
        loaded[name] = read_json(bound_path)

    business = loaded["business_blind"]
    transfer = loaded["transfer_validation"]

    # The business-blind evaluation carries no internal signature; the transfer
    # validation report does, so verify it the same way the Packing release does.
    unsigned_transfer = dict(transfer)
    transfer_checksum = unsigned_transfer.pop("report_sha256", None)
    # The aggregate business-blind safe-admission rate is intentionally NOT used:
    # it is diluted by out-of-scope hybrid families (qwen3p5_9b was 0/4). Dense
    # admission is validated per dense family below plus a zero OOM-admission gate.
    oom_admission_rate = business.get("oom_admission_rate")
    if (
        oom_admission_rate is None
        or float(oom_admission_rate) != 0.0
        or transfer.get("accepted") is not True
        or transfer_checksum != bindings["transfer_validation"].get("report_sha256")
        or transfer_checksum != sha256_json(unsigned_transfer)
    ):
        raise ValueError(
            "Dense memory-admission release acceptance evidence is not valid"
        )

    # Dense business-blind safe admission must hold on every dense model family
    # that is both in scope and present in the evidence, not only in aggregate.
    dense_models = set(release["scope"]["model_ids"])
    by_model = business.get("by_model") or {}
    covered_dense = [m for m in by_model if m in dense_models]
    if not covered_dense:
        raise ValueError("Dense memory-admission evidence covers no dense family")
    for model_id in covered_dense:
        if float(by_model[model_id].get("safe_admission_rate") or 0.0) < 1.0:
            raise ValueError(
                f"Dense memory-admission evidence has a sub-1.0 dense family: {model_id}"
            )
    return release


def _is_excluded_family(model_id: str, release: Mapping[str, Any]) -> bool:
    excluded = release.get("excluded_model_families") or {}
    if model_id in set(excluded.get("model_ids") or []):
        return True
    return any(model_id.startswith(prefix) for prefix in excluded.get("prefixes") or [])


def scope_mismatches(
    request: Mapping[str, Any],
    release: Mapping[str, Any],
) -> list[str]:
    """Return the list of request fields that fall outside the dense scope.

    An empty list means the request is admissible under the contract, still
    subject to the runtime unified-V3 memory upper gate.
    """

    scope = release["scope"]
    mismatches: list[str] = []

    # Hybrid / VL / MoE families fail closed before any scalar scope check.
    if _is_excluded_family(str(request.get("model_id") or ""), release):
        mismatches.append("model_family_excluded")
        return mismatches

    exact = (
        ("hardware_id", "hardware_ids", "h800"),
        ("modality", "modalities", "text"),
        ("stage", "stages", "sft"),
        ("dtype", "dtypes", "bf16"),
        ("model_id", "model_ids", None),
        ("training_mode", "training_modes", None),
        ("cutoff_len", "cutoff_lens", None),
        ("gpu_count", "gpu_counts", None),
        ("target_gbs", "target_gbs_values", None),
    )
    for request_key, scope_key, default in exact:
        actual = request.get(request_key, default)
        if actual not in scope[scope_key]:
            mismatches.append(request_key)

    if str(request.get("training_mode")) == "lora":
        if int(request.get("lora_rank") or 0) not in scope["lora_ranks"]:
            mismatches.append("lora_rank")

    if bool(request.get("packing")) not in scope["packing_allowed_values"]:
        mismatches.append("packing")
    if bool(request.get("offload")) is not bool(scope["offload_allowed"]):
        mismatches.append("offload")
    if str(scope["required_kernel_substring"]).lower() not in str(
        request.get("kernel_path") or "fa3"
    ).lower():
        mismatches.append("kernel_path")
    if request.get("vl_workload_profile_path") is not None:
        mismatches.append("vl_workload_profile_path")
    return mismatches


if __name__ == "__main__":
    import json

    contract = load_dense_memory_admission_release()
    print(json.dumps({"loaded": True, "release_id": contract["release_id"]}, ensure_ascii=False, indent=2))
