#!/usr/bin/env python
"""Freeze the three unified contracts named by the plan.

The plan asks for ``ModelStructureManifest/v1``, ``WorkloadProfile/v1`` and
``RuntimeMechanism/v1`` to be frozen before any calibration work continues.  A
code inventory shows the substance already exists under other names:

* ``sft_static_workload_profiles/v1`` in :mod:`structured_throughput_modeling`
  (plus the VL extension in :mod:`vl_workload_profile`),
* ``sft_model_structure_manifest/v1`` in :mod:`model_structure_manifest`
  (runtime-derived) and the static geometry in ``model_inventory.json`` plus
  :mod:`vision_tower_structure`,
* ``sft_runtime_mechanism/v2`` produced per run and validated in
  :mod:`export_h800_observations`.

So this module deliberately does **not** author new schemas.  Re-declaring these
contracts would fork three live formats that frozen artefacts already bind by
SHA, which is the specific failure the plan warns against ("不 fork 出平行
WorkloadProfile/ManifestModel").  Instead it freezes a *registry*: for each
contract it records the authoritative schema id, the module that owns it, the
fields the contract guarantees, and the SHA-256 of the artefacts that currently
realise it.

The registry is what downstream code should reference when it needs to know
"which contract version am I speaking", and it is what a reviewer reads to see
that the three names in the plan map onto real, bound implementations rather
than aspirations.  It fits nothing and launches nothing.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, ROOT, read_json, sha256_file, sha256_json, write_json

SCHEMA = "sft_unified_contract_registry/v1"
IMPLEMENTATION_VERSION = "sft_unified_contract_registry_impl/2026-08-01.v1"

DEFAULT_OUTPUT = ARTIFACT_DIR / "unified_contract_registry_v1.json"

# Plan name -> authoritative implementation.  ``schema_id`` is the string that
# actually appears in artefacts; the plan name is an alias, never a new format.
CONTRACTS: tuple[dict[str, Any], ...] = (
    {
        "plan_name": "ModelStructureManifest/v1",
        "schema_id": "sft_model_structure_manifest/v1",
        "owning_module": "scripts/model_structure_manifest.py",
        "derivation": "runtime rank manifest (post-run)",
        "static_companions": [
            "artifacts/model_inventory.json",
            "scripts/vision_tower_structure.py",
        ],
        "guaranteed_fields": [
            "component parameter totals for language_model / vision_tower / "
            "multimodal_projector / other",
            "freeze markers: freeze_vision_tower, freeze_multi_modal_projector, "
            "freeze_language_model",
            "declaration-vs-observation mismatch list",
            "tied-weight deduplication by parameter identity",
            "LoRA adapter target-hit flags",
        ],
        "known_limitations": [
            "freeze state requires a real run; it cannot be derived from "
            "train_type (measured Qwen3-VL Full froze the vision tower)",
            "static vision geometry lives in vision_tower_structure, and only "
            "qwen3_vl / qwen2_5_vl have checkpoint-verified formulas",
        ],
        "realising_artifacts": [
            "artifacts/model_inventory.json",
            "artifacts/vision_tower_structure_v1.json",
        ],
    },
    {
        "plan_name": "WorkloadProfile/v1",
        "schema_id": "sft_static_workload_profiles/v1",
        "owning_module": "scripts/structured_throughput_modeling.py",
        "derivation": "static token-length profile + packing simulation (pre-run)",
        "static_companions": [
            "scripts/vl_workload_profile.py",
            "scripts/static_packing_predictor.py",
            "scripts/metrics_callback.py",
        ],
        "guaranteed_fields": [
            "logical_samples, physical_sequences, computed_tokens, "
            "effective_tokens, padding utilisation",
            "computed_attention_token_pairs and effective_attention_token_pairs",
            "pack utilisation, packing_fill_ratio, mean_samples_per_pack",
            "gradient_accumulation_steps and physical_sequences_per_step",
            "sha256-pinned dataset provenance",
        ],
        "known_limitations": [
            "the VL extension (sft_vl_workload_profile/v1) is a separate schema "
            "because image count, resolution and visual tokens are not text "
            "fields; it stays uncalibrated until real-image evidence exists",
            "runtime counterpart CollatorCounters measures the same quantities "
            "live, and Phase B is what proves the two agree",
        ],
        "realising_artifacts": [
            "artifacts/static_workload_profiles.json",
            "artifacts/vl_workload_profile_pzfj38_v1.json",
        ],
    },
    {
        "plan_name": "RuntimeMechanism/v1",
        "schema_id": "sft_runtime_mechanism/v2",
        "owning_module": "scripts/run_job.py",
        "derivation": "per-run execution evidence, validated on export",
        "static_companions": [
            "scripts/export_h800_observations.py",
            "scripts/runtime_evidence.py",
        ],
        "guaranteed_fields": [
            "self-verifying fingerprint_sha256 over the mechanism payload",
            "source_manifest plus its own hash",
            "excluded_dimensions covering model, dataset, micro_batch_size, "
            "global_batch_size and gpu_count",
        ],
        "known_limitations": [
            "the plan calls this v1; the live format is already v2, and the "
            "higher number is authoritative -- renaming it would invalidate "
            "every bound observation",
            "model, dataset and batch dimensions are excluded by contract so a "
            "mechanism key can never memorise a model or dataset id",
        ],
        "realising_artifacts": [
            "artifacts/canonical_h800_observations.jsonl",
        ],
    },
)


def _artifact_binding(relative: str, root: Path = ROOT) -> dict[str, Any]:
    path = root / relative
    binding: dict[str, Any] = {"path": relative, "exists": path.is_file()}
    if path.is_file():
        binding["sha256"] = sha256_file(path)
        if path.suffix == ".json":
            try:
                value = read_json(path)
            except (OSError, ValueError):
                binding["declared_schema"] = None
            else:
                binding["declared_schema"] = (
                    value.get("schema") if isinstance(value, Mapping) else None
                )
    return binding


def build_registry(
    contracts: Sequence[Mapping[str, Any]] = CONTRACTS, *, root: Path = ROOT
) -> dict[str, Any]:
    """Freeze the contract registry against whatever currently realises it."""

    entries: list[dict[str, Any]] = []
    unrealised: list[str] = []
    for contract in contracts:
        realising = [
            _artifact_binding(relative, root)
            for relative in contract["realising_artifacts"]
        ]
        modules = [
            _artifact_binding(contract["owning_module"], root),
            *(
                _artifact_binding(relative, root)
                for relative in contract.get("static_companions") or []
            ),
        ]
        missing = [item["path"] for item in realising if not item["exists"]]
        if missing:
            unrealised.extend(missing)
        entries.append(
            {
                **{
                    key: value
                    for key, value in contract.items()
                    if key not in {"realising_artifacts", "static_companions"}
                },
                "implementation_bindings": modules,
                "realising_artifacts": realising,
                "fully_realised": not missing,
                "missing_artifacts": missing,
            }
        )

    registry = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "contracts_frozen_by_reference",
        "policy": {
            "authors_new_schemas": False,
            "renames_live_schemas": False,
            "forks_existing_formats": False,
            "note": (
                "Plan names are aliases onto existing schema ids. The live id is "
                "authoritative; frozen artefacts already bind those ids by SHA, "
                "so redeclaring them would fork the format."
            ),
        },
        "contract_count": len(entries),
        "contracts": entries,
        "all_contracts_realised": not unrealised,
        "unrealised_artifacts": sorted(set(unrealised)),
        "guarantees": {
            "fits_or_publishes_coefficients": False,
            "mutates_frozen_artifacts": False,
            "creates_gpu_queue": False,
        },
    }
    registry["registry_sha256"] = sha256_json(registry)
    return registry


def resolve_schema(plan_name: str) -> str:
    """Map a plan contract name onto the authoritative live schema id."""

    for contract in CONTRACTS:
        if contract["plan_name"] == plan_name:
            return str(contract["schema_id"])
    raise KeyError(f"unknown contract name: {plan_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    registry = build_registry()
    write_json(args.output, registry)

    print(f"contracts: {registry['contract_count']}")
    for contract in registry["contracts"]:
        print(f"  {contract['plan_name']}")
        print(f"    -> {contract['schema_id']}  ({contract['owning_module']})")
        print(f"    realised: {contract['fully_realised']}")
        for item in contract["realising_artifacts"]:
            mark = "ok" if item["exists"] else "MISSING"
            print(f"      [{mark}] {item['path']}")
        for limitation in contract["known_limitations"]:
            print(f"    limitation: {limitation[:88]}")
    print(f"all_contracts_realised: {registry['all_contracts_realised']}")
    for missing in registry["unrealised_artifacts"]:
        print(f"unrealised: {missing}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
