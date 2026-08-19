#!/usr/bin/env python3
"""Surgically correct the projection-width FLOPs fields in the theory basis.

The fixed `_model_geometry` (heads*head_dim projection widths) cannot be
applied by regenerating the theory basis from canonical observations: the
2026-07-27 canonical export changed observation content, and 51 historical
recovery references no longer resolve to byte-identical measurements.

This script instead derives a new theory basis artifact from the frozen one,
changing ONLY the fields the structured throughput training consumes:

* model_basis.linear_applications_per_pass
* model_basis.adapter_parameters / loaded / trainable / frozen
* model_basis.max_layer_parameter_elements / max_module_parameter_elements
* model_basis.persistent_parameter_elements_structural

Every other field (measurements, performance, memory, scenario, selector) is
copied bit-for-bit.  The script fails closed if the stored geometry cannot be
reproduced from the stored fields, and verifies that unaffected models
(heads*head_dim == hidden) are byte-identical before and after.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import read_json, sha256_json, write_json

DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "h800_theory_basis.json"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "h800_theory_basis_flopsfix_20260813_v1.json"
)
DEFAULT_INVENTORY = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "model_inventory.json"
)
SCHEMA = "sft_h800_theory_basis_flopsfix_report/v1"


def _int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _checkpoint_head_dim(inventory: dict[str, Any]) -> dict[tuple[int, int], int]:
    """Map (hidden_size, num_attention_heads) to checkpoint-authoritative head_dim.

    The frozen artifact stores a wrong head_dim for Qwen3-4B (80 instead of
    128); the checkpoint config.json is authoritative.
    """

    mapping: dict[tuple[int, int], int] = {}
    for model in inventory.get("models") or []:
        path = model.get("path")
        if not path:
            continue
        config_path = Path(str(path)) / "config.json"
        if not config_path.is_file():
            continue
        config = read_json(config_path)
        text = config.get("text_config") or config
        hidden = _int(text.get("hidden_size"), "hidden_size")
        heads = _int(text.get("num_attention_heads"), "num_attention_heads")
        head_dim = _int(text.get("head_dim") or hidden // heads, "head_dim")
        mapping[(hidden, heads)] = head_dim
    return mapping


def _patched_model_basis(
    model_basis: dict[str, Any], true_head_dim: int
) -> dict[str, Any]:
    hidden = _int(model_basis["hidden_size"], "hidden_size")
    heads = _int(model_basis["num_attention_heads"], "num_attention_heads")
    stored_head_dim = _int(model_basis["head_dim"], "head_dim")
    kv_heads = _int(model_basis["num_key_value_heads"], "num_key_value_heads")
    layers = _int(model_basis["num_layers"], "num_layers")
    intermediate = _int(
        model_basis["intermediate_size"], "intermediate_size"
    )
    vocab = _int(model_basis["vocab_size"], "vocab_size")
    stored_kv_width = kv_heads * stored_head_dim
    kv_width = kv_heads * true_head_dim
    q_width = heads * true_head_dim

    stored_linear = int(model_basis["linear_applications_per_pass"])
    # Reproduce the OLD formula from the stored geometry; fail closed on drift.
    old_linear = layers * (
        2 * hidden * hidden
        + 2 * hidden * stored_kv_width
        + 3 * hidden * intermediate
    ) + vocab * hidden
    if old_linear != stored_linear:
        raise ValueError(
            f"stored linear_applications_per_pass {stored_linear} does not "
            f"reproduce from stored geometry ({old_linear})"
        )

    new_linear = layers * (
        2 * hidden * q_width + 2 * hidden * kv_width + 3 * hidden * intermediate
    ) + vocab * hidden
    new_max_layer = (
        2 * hidden * q_width
        + 2 * hidden * kv_width
        + 3 * hidden * intermediate
        + 2 * hidden
    )
    new_max_module = max(new_max_layer, vocab * hidden)

    old_adapter = int(model_basis["adapter_parameters"])
    old_loaded = int(model_basis["loaded_parameters"])
    base = old_loaded - old_adapter
    new_adapter = old_adapter
    if old_adapter > 0:
        # LoRA target=all: old count was rank*layers*(9h + 2kv + 3i).
        per_layer_old = 9 * hidden + 2 * stored_kv_width + 3 * intermediate
        if old_adapter % (layers * per_layer_old) != 0:
            raise ValueError("LoRA adapter count is not rank-divisible")
        rank = old_adapter // (layers * per_layer_old)
        new_adapter = rank * layers * (
            (hidden + q_width)
            + (q_width + hidden)
            + 2 * (hidden + kv_width)
            + 3 * (hidden + intermediate)
        )
    new_loaded = base + new_adapter
    new_trainable = new_adapter if old_adapter > 0 else new_loaded
    new_frozen = new_loaded - new_trainable
    new_persistent = min(
        new_loaded,
        layers * (2 * hidden + 2 * true_head_dim) + hidden,
    )

    return {
        "head_dim": true_head_dim,
        "kv_width": kv_width,
        "linear_applications_per_pass": new_linear,
        "adapter_parameters": new_adapter,
        "loaded_parameters": new_loaded,
        "trainable_parameters": new_trainable,
        "frozen_parameters": new_frozen,
        "max_layer_parameter_elements": new_max_layer,
        "max_module_parameter_elements": new_max_module,
        "persistent_parameter_elements_structural": new_persistent,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-inventory", type=Path, default=DEFAULT_INVENTORY)
    args = parser.parse_args()

    artifact = read_json(args.input)
    records = artifact.get("records")
    if not isinstance(records, list):
        raise ValueError("theory basis artifact has no records")
    source_sha256 = sha256_json(artifact)
    head_dim_by_geometry = _checkpoint_head_dim(read_json(args.model_inventory))

    changed_by_model: Counter[str] = Counter()
    untouched = 0
    patched_fields = [
        "head_dim",
        "kv_width",
        "linear_applications_per_pass",
        "adapter_parameters",
        "loaded_parameters",
        "trainable_parameters",
        "frozen_parameters",
        "max_layer_parameter_elements",
        "max_module_parameter_elements",
        "persistent_parameter_elements_structural",
    ]
    for record in records:
        model_basis = record.get("model_basis")
        if not isinstance(model_basis, dict):
            continue
        hidden = int(model_basis["hidden_size"])
        heads = int(model_basis["num_attention_heads"])
        if (hidden, heads) not in head_dim_by_geometry:
            raise ValueError(
                f"record {record.get('job_id')} geometry ({hidden}, {heads}) "
                "has no checkpoint-authoritative head_dim"
            )
        true_head_dim = head_dim_by_geometry[(hidden, heads)]
        patch = _patched_model_basis(model_basis, true_head_dim)
        changed = any(
            int(model_basis.get(key, -1)) != value
            for key, value in patch.items()
        )
        geometry_affected = (
            heads * true_head_dim != hidden
            or int(model_basis["head_dim"]) != true_head_dim
        )
        if geometry_affected and not changed:
            raise ValueError(
                f"record {record.get('job_id')} has affected geometry but no "
                f"field change under the fix: {patch}"
            )
        if not geometry_affected and changed:
            raise ValueError(
                f"record {record.get('job_id')} has unaffected geometry "
                f"({hidden}, {heads}, head_dim {true_head_dim}) but fields "
                f"changed under the fix: {patch}"
            )
        if not changed:
            untouched += 1
            continue
        model_id = str(
            (record.get("scenario") or {}).get("model_id")
            or record.get("job_id")
        )
        changed_by_model[model_id] += 1
        model_basis.update(patch)

    output = dict(artifact)
    output["flopsfix_provenance"] = {
        "schema": SCHEMA,
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact_sha256": source_sha256,
        "patch_script": str(Path(__file__).resolve()),
        "changed_fields": patched_fields,
        "changed_records_by_model": dict(changed_by_model),
        "untouched_records": untouched,
        "reason": (
            "projection widths heads*head_dim replace the historical "
            "hidden x hidden assumption in linear_applications_per_pass and "
            "per-layer parameter elements; measurements and all other fields "
            "are byte-identical to the frozen artifact"
        ),
    }
    output["report_sha256"] = sha256_json(
        {key: value for key, value in output.items() if key != "report_sha256"}
    )
    write_json(args.output, output)
    print(
        json.dumps(
            {
                "records": len(records),
                "changed_records_by_model": dict(changed_by_model),
                "untouched_records": untouched,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
